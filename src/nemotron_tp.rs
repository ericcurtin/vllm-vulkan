//! Nemotron-75B TP=2 (Mamba2 head-shard) / EP=2 (latent-MoE) sharding scaffold.
//!
//! **Increment 1:** the reusable col/row shard primitives (lifted from
//! `q35_tp_shard`, tp.rs:297–311), a combine-mode shard-invariance test
//! harness, and a name-map dispatch stub. Proves the machinery + both combine
//! modes on a bare GEMM before any Mamba2/MoE intricacy (see
//! `scratchpad/nemotron-tp-ep-implementation-plan.md` Inc 1).
//!
//! **Increment 2 (this file, current state):** the Mamba2 TP=2 head-shard —
//! [`mamba_head_shard`] (5-segment in_proj column-shard, channel-partitioned
//! conv1d, head/group-partitioned A_log/D/dt_bias/norm, row-sharded out_proj)
//! and [`mamba_state_shard`], wired into [`nem_tp_shard`]'s `.mixer.*` arms.
//! **No kernel change:** the existing `mamba2_decode_step` (nemotron.rs:534)
//! runs verbatim on the LOCAL (per-rank) dims + weight slices + state; its
//! out_proj matmul only ever contracts over the LOCAL `scan[inter/n]`, so its
//! return value already IS the out_proj partial — no separate plumbing needed.
//!
//! **Increment 3 (this file, current state): EP=2 latent-MoE.** Router +
//! fc1(latent) + fc2 + shared expert stay REPLICATED (both ranks run them
//! bit-identically, per the design in the plan §4); each rank owns a disjoint
//! contiguous HALF of the routed experts ([`moe_expert_half`]) and accumulates
//! a `routed[lat]` PARTIAL over `selected ∩ owned`
//! ([`latent_moe_routed_partial`]) — the two ranks' partials sum-combine
//! (all-reduce stand-in) BEFORE `fc2`. The resident-loader byte-layout wiring
//! of `.moe_experts.{up,down}` into `nem_tp_shard`'s name-map dispatch is
//! Increment 4 (cluster-gated numeric; this file proves the partition math on
//! the dequantized f32 free-fn path only — see plan §0/§4).
//!
//! **Quant-scale finding (FP8/NVFP4), resolved statically — see plan §0:**
//! FP8 mamba in/out_proj + shared-expert weights use `num_bits:8`, per-tensor,
//! with `.weight_scale` an F32 **scalar** (shape `[]`) — see
//! `nemotron_loader.rs:22–27`. A per-tensor scalar is replicated unchanged to
//! both shards; there is no per-row/per-block scale layout to re-slice. NVFP4
//! routed-expert `.weight_scale` (F8, per-group `[out,in/16]`) is sharded
//! per-WHOLE-EXPERT under EP (Increment 3), so its per-group scales travel with
//! the expert byte-slice untouched. This module's tests validate the partition
//! math on the dequantized f32 free-fn path only; quant-scale carry in the
//! resident-loader byte layout is Increment 4 (Mac-testable as a layout
//! assertion; NOT wired to a GPU dispatch — see below).
//!
//! **Cluster seams (NOT implemented here — GPU/Vulkan or live comm required):**
//! - GPU sharded-matvec dispatch: routing the sharded weights through the
//!   resident matvec path — `nem_matvec` (nemotron.rs:1259) /
//!   `nem_expert_matvec` (nemotron.rs:1426) — needs a live Vulkan device.
//! - Real 2-rank pairwise all-reduce: replacing this module's host-side
//!   `combine_sum` with `send_f32`/`recv_f32` (vccl_ffi.rs:491/516),
//!   deadlock-safe even-send-first order over a live vCCL comm. The single-
//!   process `combine_sum` used by the tests below is mathematically
//!   identical to that exchange (each rank's partial is what it would send);
//!   only the transport is deferred.

/// Column-shard a `[out, in_feat]` row-major weight: keep OUTPUT rows
/// `[r*out/n, (r+1)*out/n)` (column-parallel — each rank computes a disjoint
/// slice of the output vector). Lifted from the `col` closure in
/// `q35_tp_shard` (tp.rs:297–303).
#[allow(dead_code)] // consumed by nem_tp_shard's mamba/moe arms, Increment 2/3
pub(crate) fn col_shard(w: &[f32], in_feat: usize, r: usize, n: usize) -> Vec<f32> {
    let out = w.len() / in_feat;
    let per = out / n;
    let lo = r * per;
    w[lo * in_feat..(lo + per) * in_feat].to_vec()
}

/// Row-shard a `[out, in_feat]` row-major weight: keep INPUT columns
/// `[r*in_feat/n, (r+1)*in_feat/n)` of every output row (row-parallel — each
/// rank computes a partial sum over a disjoint slice of the contraction
/// dimension). Lifted from the `row` closure in `q35_tp_shard` (tp.rs:305–311).
#[allow(dead_code)] // consumed by nem_tp_shard's mamba/moe arms, Increment 2/3
pub(crate) fn row_shard(w: &[f32], in_feat: usize, r: usize, n: usize) -> Vec<f32> {
    let out = w.len() / in_feat;
    let per = in_feat / n;
    let lo = r * per;
    let mut o = Vec::with_capacity(out * per);
    for rr in 0..out {
        o.extend_from_slice(&w[rr * in_feat + lo..rr * in_feat + lo + per]);
    }
    o
}

use crate::nemotron::{mlp_relu2, LatentMoeDims, Mamba2Dims, Mamba2State, Mamba2Weights};

/// Row ranges `(start_row, len)` of the 5-segment in_proj `[in_proj_out, *]`
/// column-shard for rank `r` of `n`, in `gate | conv-x | conv-B | conv-C | dt`
/// order — mirrors `Mamba2Dims::in_proj_out`'s layout (nemotron.rs:333),
/// consumed at nemotron.rs:545–556 (`gate`/`raw_bc`/`dt` slicing). `conv-x`/
/// `gate` are partitioned BY HEAD (`inter/n` wide); `conv-B`/`conv-C` BY GROUP
/// (`(n_groups/n)*ssm_state` wide, group-shared across `heads_per_group`
/// heads); `dt` BY HEAD (`num_heads/n` wide).
///
/// Shared by [`mamba_head_shard`] (sliced at `hidden_size` row-width, to shard
/// the in_proj WEIGHT rows) and the shard-invariance test (sliced at row-width
/// 1, to gather the corresponding elements of a monolithic in_proj OUTPUT
/// vector by global index — the plan's "compare by global index, not by
/// position" note).
fn in_proj_seg_ranges(d: &Mamba2Dims, r: usize, n: usize) -> [(usize, usize); 5] {
    let inter = d.intermediate();
    let ng = d.n_groups;
    let ss = d.ssm_state_size;
    let inter_l = inter / n;
    let ng_l = ng / n;
    let nh_l = d.num_heads / n;
    let conv_dim = d.conv_dim();
    [
        (r * inter_l, inter_l),                           // gate, by head
        (inter + r * inter_l, inter_l),                   // conv-x, by head
        (2 * inter + r * ng_l * ss, ng_l * ss),           // conv-B, by group
        (2 * inter + ng * ss + r * ng_l * ss, ng_l * ss), // conv-C, by group
        (inter + conv_dim + r * nh_l, nh_l),              // dt, by head
    ]
}

/// Channel ranges `(start_channel, len)` of the conv1d `[conv_dim, *]`
/// channel-shard for rank `r` of `n`, in `x | B | C` order — the same
/// head/group partition as [`in_proj_seg_ranges`]'s `conv-x`/`conv-B`/`conv-C`
/// segments (raw_bc's channel layout, nemotron.rs:552–556). Used for
/// `conv1d.weight`/`conv1d.bias` (row-width `conv_kernel`/1) and `conv_state`
/// (row-width `conv_kernel`).
fn conv_chan_ranges(d: &Mamba2Dims, r: usize, n: usize) -> [(usize, usize); 3] {
    let inter = d.intermediate();
    let ng = d.n_groups;
    let ss = d.ssm_state_size;
    let inter_l = inter / n;
    let ng_l = ng / n;
    [
        (r * inter_l, inter_l),                       // x, by head
        (inter + r * ng_l * ss, ng_l * ss),           // B, by group
        (inter + ng * ss + r * ng_l * ss, ng_l * ss), // C, by group
    ]
}

/// Gather row-ranges of a flat row-major buffer (`row_width` elements/row)
/// into one contiguous `Vec`, in range order — the shared slicing primitive
/// for both [`in_proj_seg_ranges`] and [`conv_chan_ranges`].
fn gather_rows(data: &[f32], row_width: usize, ranges: &[(usize, usize)]) -> Vec<f32> {
    let mut out = Vec::with_capacity(ranges.iter().map(|&(_, len)| len).sum::<usize>() * row_width);
    for &(lo, len) in ranges {
        out.extend_from_slice(&data[lo * row_width..(lo + len) * row_width]);
    }
    out
}

/// Owned LOCAL (per-rank) Mamba2 weight buffers produced by
/// [`mamba_head_shard`]. Field shapes mirror `Mamba2Weights` but each buffer
/// covers only this rank's head/group partition; [`Self::borrow`] builds the
/// `Mamba2Weights<'_>` that `mamba2_decode_step` takes, unchanged.
#[allow(dead_code)] // consumed by the resident TP loader wiring, Increment 4 (cluster)
pub(crate) struct Mamba2WeightBufsLocal {
    pub in_proj: Vec<f32>,
    pub conv_w: Vec<f32>,
    pub conv_b: Option<Vec<f32>>,
    pub a_log: Vec<f32>,
    pub d_skip: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm_w: Vec<f32>,
    pub out_proj: Vec<f32>,
}

impl Mamba2WeightBufsLocal {
    #[allow(dead_code)] // consumed by the resident TP loader wiring, Increment 4 (cluster)
    pub(crate) fn borrow(&self) -> Mamba2Weights<'_> {
        Mamba2Weights {
            in_proj: &self.in_proj,
            conv1d_weight: &self.conv_w,
            conv1d_bias: self.conv_b.as_deref(),
            a_log: &self.a_log,
            d: &self.d_skip,
            dt_bias: &self.dt_bias,
            norm_weight: &self.norm_w,
            out_proj: &self.out_proj,
        }
    }
}

/// TP=2(+) Mamba2 head-shard: partitions one layer's mixer weights across `n`
/// ranks by HEAD (and, for the group-shared B/C in_proj/conv1d segments, by
/// GROUP), returning this rank's LOCAL weight buffers + LOCAL dims. Rank `r`'s
/// `mamba2_decode_step(x, &local.borrow(), &mut state_local, &dims_local)`
/// then runs **verbatim, no kernel change**, and its return value already IS
/// the out_proj PARTIAL (row-sharded out_proj — see module doc / plan §3 for
/// the all-reduce seam that sum-combines the `n` ranks' partials).
///
/// **Requires** `d.num_heads % n == 0 && d.n_groups % n == 0` (asserted). A TP
/// factor that doesn't divide `n_groups` would split a group's shared B/C
/// across ranks, which `mamba2_recurrence_and_norm`'s group-shared read
/// (nemotron.rs:477–524) cannot express without a kernel change — out of
/// scope for this increment (see plan §3).
#[allow(dead_code)] // consumed by the resident TP loader wiring, Increment 4 (cluster)
pub(crate) fn mamba_head_shard(
    w: &Mamba2Weights,
    d: &Mamba2Dims,
    r: usize,
    n: usize,
) -> (Mamba2WeightBufsLocal, Mamba2Dims) {
    assert!(n >= 1 && r < n);
    assert_eq!(
        d.num_heads % n,
        0,
        "mamba_head_shard: num_heads {} not divisible by n {}",
        d.num_heads,
        n
    );
    assert_eq!(
        d.n_groups % n,
        0,
        "mamba_head_shard: n_groups {} not divisible by n {}",
        d.n_groups,
        n
    );

    let dl = Mamba2Dims {
        hidden_size: d.hidden_size,
        num_heads: d.num_heads / n,
        head_dim: d.head_dim,
        ssm_state_size: d.ssm_state_size,
        n_groups: d.n_groups / n,
        conv_kernel: d.conv_kernel,
        time_step_min: d.time_step_min,
        eps: d.eps,
    };

    let in_proj_l = gather_rows(w.in_proj, d.hidden_size, &in_proj_seg_ranges(d, r, n));
    let conv_ranges = conv_chan_ranges(d, r, n);
    let conv_w = gather_rows(w.conv1d_weight, d.conv_kernel, &conv_ranges);
    let conv_b = w.conv1d_bias.map(|b| gather_rows(b, 1, &conv_ranges));

    let nh_l = dl.num_heads;
    let head_range = [(r * nh_l, nh_l)];
    let a_log = gather_rows(w.a_log, 1, &head_range);
    let d_skip = gather_rows(w.d, 1, &head_range);
    let dt_bias = gather_rows(w.dt_bias, 1, &head_range);
    // norm_weight is by-head over `intermediate` (each rank's heads form a
    // whole number of gated-norm groups — see the fixture note below), i.e.
    // the same channel range as the conv-x segment.
    let norm_w = gather_rows(w.norm_weight, 1, &[conv_ranges[0]]);

    // out_proj [hidden_size, intermediate] ROW-shard (input cols = intermediate
    // channels, by head) — reuse the Inc 1 primitive directly.
    let out_proj = row_shard(w.out_proj, d.intermediate(), r, n);

    (
        Mamba2WeightBufsLocal {
            in_proj: in_proj_l,
            conv_w,
            conv_b,
            a_log,
            d_skip,
            dt_bias,
            norm_w,
            out_proj,
        },
        dl,
    )
}

/// Shard a Mamba2 recurrent state (`conv_state`/`ssm_state`) to rank `r` of
/// `n`, matching [`mamba_head_shard`]'s channel/head partition. For the
/// zero-start case, callers should use `Mamba2State::zeros(&dims_local)`
/// directly instead (no need to shard an all-zero buffer).
#[allow(dead_code)] // consumed by the resident TP loader wiring, Increment 4 (cluster)
pub(crate) fn mamba_state_shard(s: &Mamba2State, d: &Mamba2Dims, r: usize, n: usize) -> Mamba2State {
    assert_eq!(d.num_heads % n, 0);
    assert_eq!(d.n_groups % n, 0);
    let conv_ranges = conv_chan_ranges(d, r, n);
    let conv_state = gather_rows(&s.conv_state, d.conv_kernel, &conv_ranges);
    let nh_l = d.num_heads / n;
    let ssm_row_width = d.head_dim * d.ssm_state_size; // per-head chunk of ssm_state
    let ssm_state = gather_rows(&s.ssm_state, ssm_row_width, &[(r * nh_l, nh_l)]);
    Mamba2State { conv_state, ssm_state }
}

/// EP=2(+) whole-expert partition of the routed-expert weight buffers: keep
/// the contiguous per-expert slices for experts `[r*ne/n, (r+1)*ne/n)`.
/// `expert_up` is `[n_routed_experts, moe_intermediate_size, moe_latent_size]`
/// row-major (per-expert stride `inter*lat`, nemotron.rs `expert_up_stride`,
/// `latent_moe_forward` body); `expert_down` is
/// `[n_routed_experts, moe_latent_size, moe_intermediate_size]` (per-expert
/// stride `lat*inter`, `expert_down_stride`). Each expert's weight matrix is a
/// single atomic byte-slice — there is no per-expert re-slicing, unlike the
/// mamba head/group shard — so **assert** `ne % n == 0` (an EP factor that
/// doesn't divide `n_routed_experts` has no meaning here).
#[allow(dead_code)] // consumed by the resident TP loader wiring, Increment 4 (cluster)
pub(crate) fn moe_expert_half(
    expert_up: &[f32],
    expert_down: &[f32],
    ne: usize,
    lat: usize,
    inter: usize,
    r: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(n >= 1 && r < n);
    assert_eq!(
        ne % n,
        0,
        "moe_expert_half: n_routed_experts {} not divisible by n {}",
        ne,
        n
    );
    let per = ne / n;
    let lo = r * per;
    let up_stride = inter * lat;
    let down_stride = lat * inter;
    let up_local = expert_up[lo * up_stride..(lo + per) * up_stride].to_vec();
    let down_local = expert_down[lo * down_stride..(lo + per) * down_stride].to_vec();
    (up_local, down_local)
}

/// The routed-expert loop (`latent_moe_forward`'s per-token inner loop,
/// nemotron.rs ~789–797) restricted to the OWNED expert id range
/// `[owned_lo, owned_hi)` of this rank's `up_local`/`down_local` (produced by
/// [`moe_expert_half`]). `indices`/`weights` are the token's FULL top-k
/// selection from the (REPLICATED) `router_forward` call — both ranks receive
/// the identical selection; this rank simply skips any selected expert it
/// does not own (its EP peer owns it and contributes that term to ITS
/// partial). Each owned expert's contribution uses the exact same
/// `weights[k]` gate value the monolithic loop would apply — **no
/// re-normalization within the owned half** (that would silently change the
/// math; the plan flags this as the common trap). Calls the SAME
/// [`mlp_relu2`] the monolithic path calls, so per-expert numerics are
/// byte-identical.
///
/// Returns this rank's PARTIAL `routed[moe_latent_size]` accumulator. The two
/// (or `n`) ranks' partials sum-combine (the all-reduce stand-in, plan §1 SUM
/// row: cos-tolerance, not bit-exact — FP add is non-associative) BEFORE
/// `fc2`.
#[allow(dead_code)] // consumed by the resident TP loader wiring, Increment 4 (cluster)
pub(crate) fn latent_moe_routed_partial(
    latent: &[f32],
    indices: &[usize],
    weights: &[f32],
    up_local: &[f32],
    down_local: &[f32],
    d: &LatentMoeDims,
    owned_lo: usize,
    owned_hi: usize,
) -> Vec<f32> {
    let lat_size = d.moe_latent_size;
    let inter = d.moe_intermediate_size;
    let up_stride = inter * lat_size;
    let down_stride = lat_size * inter;
    let mut routed = vec![0.0f32; lat_size];
    for (k, &e) in indices.iter().enumerate() {
        if e < owned_lo || e >= owned_hi {
            continue; // owned by a different EP rank
        }
        let el = e - owned_lo;
        let up = &up_local[el * up_stride..(el + 1) * up_stride];
        let down = &down_local[el * down_stride..(el + 1) * down_stride];
        let eout = mlp_relu2(latent, up, down, lat_size, inter);
        let wk = weights[k];
        for (rv, &o) in routed.iter_mut().zip(&eout) {
            *rv += o * wk;
        }
    }
    routed
}

/// Nemotron TP/EP shard name-map dispatch. `n <= 1` is the no-op (single-rank)
/// case. **Increment 2:** the mamba `.mixer.*` arms are wired (in_proj
/// 5-segment column-shard, conv1d channel-partition, A_log/D/dt_bias/norm
/// head-partition, out_proj row-shard) via [`mamba_head_shard`]'s range
/// helpers. **Increment 3** adds the EP=2 expert-half partition math
/// ([`moe_expert_half`] / [`latent_moe_routed_partial`]), proven by the
/// dequantized f32 shard-invariance test below — but this fn's `.moe_experts.*`
/// arm is deliberately NOT wired here: `nem_tp_shard` is keyed on `Mamba2Dims`
/// (the mamba per-layer dims), whereas the expert partition needs
/// `n_routed_experts`/`moe_latent_size`/`moe_intermediate_size`
/// (`LatentMoeDims`), and the real routed-expert tensors are resident
/// NVFP4-quantized byte buffers, not this f32 free-fn path. That resident
/// byte-layout wiring (`gpu_experts[layer]` partitioned to owned expert ids)
/// is Increment 4 (cluster-gated numeric; plan §5). Mirrors the structure of
/// `q35_tp_shard` (tp.rs:283–366).
#[allow(dead_code)] // dispatch surface wired to a real loader call in Increment 4 (cluster)
pub(crate) fn nem_tp_shard(name: &str, data: Vec<f32>, d: &Mamba2Dims, r: usize, n: usize) -> Vec<f32> {
    if n <= 1 {
        return data;
    }
    if name.ends_with(".mixer.in_proj.weight") {
        return gather_rows(&data, d.hidden_size, &in_proj_seg_ranges(d, r, n));
    }
    if name.ends_with(".mixer.conv1d.weight") {
        return gather_rows(&data, d.conv_kernel, &conv_chan_ranges(d, r, n));
    }
    if name.ends_with(".mixer.conv1d.bias") {
        return gather_rows(&data, 1, &conv_chan_ranges(d, r, n));
    }
    if name.ends_with(".mixer.A_log") || name.ends_with(".mixer.D") || name.ends_with(".mixer.dt_bias") {
        let nh_l = d.num_heads / n;
        return gather_rows(&data, 1, &[(r * nh_l, nh_l)]);
    }
    if name.ends_with(".mixer.norm.weight") {
        return gather_rows(&data, 1, &[conv_chan_ranges(d, r, n)[0]]);
    }
    if name.ends_with(".mixer.out_proj.weight") {
        return row_shard(&data, d.intermediate(), r, n);
    }
    // `.moe_experts.{up,down}` (whole-expert EP partition): partition math is
    // `moe_expert_half` (proven above); the resident quantized-byte-layout
    // wiring into this name-map is Increment 4 (cluster; see fn doc above).
    // Everything else (router, fc1/fc2, shared expert, embed_tokens, lm_head,
    // all layernorms) is REPLICATED.
    data
}

// ── Config-driven TP shard dispatcher (attention + shared-expert + mamba) ────
//
// The resident loader (`load_nemotron_resident`) stages every sharded weight
// through an f32 buffer before (re)quantizing it GPU-resident: the f16 arm
// `decode_plain`s BF16→f32 (attn q/k/v/o, fc1/fc2), and the q8_0 arm
// `dequantize_fp8`→f32→requant (mamba in/out_proj, shared_experts up/down).
// So sharding on that f32 buffer with [`nem_tp_shard_full`] (before the
// re-encode) needs NO quantized-byte col/row slicing for the numeric weights.
// The NVFP4 routed experts are the ONE exception — they stay packed and are
// partitioned whole-expert (EP=2) via [`expert_owned_range`] in the loader's
// per-expert streaming, never touching this f32 path (there is no per-expert
// re-slice under EP — the trap the plan flags).

use crate::nemotron::NemotronConfig;

/// EP rank `r`'s owned routed-expert id range `(owned_lo, owned_count)` out of
/// `n` ranks. Whole-expert partition (see [`moe_expert_half`] for the math /
/// the "no re-normalization within the owned half" invariant). Asserts
/// `ne % n == 0` (an EP factor that doesn't divide the expert count is
/// meaningless — each expert is one atomic byte-slice).
#[allow(dead_code)] // consumed by the resident loader's EP=2 streaming (Increment 4)
pub(crate) fn expert_owned_range(ne: usize, r: usize, n: usize) -> (usize, usize) {
    assert_eq!(ne % n, 0, "EP: n_routed_experts {ne} not divisible by n {n}");
    let per = ne / n;
    (r * per, per)
}

/// Local (per-rank) `(out_features, in_features)` of a nemotron matmul weight
/// after the TP shard, so the resident loader can record the sharded matvec
/// with the right dims. Column-parallel (q/k/v, mamba in_proj gate/x/dt +
/// conv/head segments, shared up) shrink `out`; row-parallel (o_proj, mamba
/// out_proj, shared down) shrink `in`; replicated weights are unchanged. Mirrors
/// exactly the partition [`nem_tp_shard_full`] applies.
#[allow(dead_code)] // consumed by the resident TP loader wiring (Increment 4, cluster)
pub(crate) fn nem_tp_local_shape(
    name: &str, out_f: usize, in_f: usize, cfg: &NemotronConfig, n: usize,
) -> (usize, usize) {
    if n <= 1 {
        return (out_f, in_f);
    }
    // Column-parallel: output-row slice (out/n) — shared-expert up_proj.
    if name.ends_with(".shared_experts.up_proj.weight") {
        return (out_f / n, in_f);
    }
    // Row-parallel: input-col slice (in/n) — shared-expert down_proj.
    if name.ends_with(".shared_experts.down_proj.weight") {
        return (out_f, in_f / n);
    }
    // REPLICATED under this TP scope: attention q/k/v/o and the mamba mixer.
    // Head-sharding either would require halving the runtime head/scan/conv/
    // gated-norm dims (and the KV-cache size) throughout the resident kernels —
    // a separate, larger change; the routed experts (EP=2) dominate the
    // footprint, and the shared-expert col/row split is dim-local to moe_tail.
    (out_f, in_f) // replicated
}

/// Config-driven TP shard of a nemotron matmul weight on the DEQUANTIZED f32
/// buffer. Under this build's TP scope only the **shared-expert MLP** is
/// f32-sharded here (Megatron column/row split): up_proj col-shard
/// (out=`shared_inter`), down_proj row-shard (in=`shared_inter`) — a
/// dim-LOCAL split (`moe_tail` uses `shared_inter/n` for the sup/sact buffers),
/// so it needs no head/scan runtime plumbing.
///
/// The dominant lever — the **routed experts** — is EP=2 whole-expert
/// ([`expert_owned_range`]) in the loader's per-expert stream, NOT here.
///
/// REPLICATED (returned unchanged): attention q/k/v/o and the mamba mixer
/// (head-sharding either needs LOCAL head/scan/conv/gated-norm dims + KV-cache
/// resizing throughout the resident kernels — a separate change; the
/// `mamba_head_shard`/attn head-shard MATH is Mac-proven in this module's tests
/// but not wired into the resident forward), plus `fc1_latent_proj`/
/// `fc2_latent_proj`, router `gate`/`e_score`, layernorms, embeddings, lm_head.
#[allow(dead_code)] // consumed by the resident TP loader wiring (Increment 4, cluster)
pub(crate) fn nem_tp_shard_full(
    name: &str, data: Vec<f32>, cfg: &NemotronConfig, r: usize, n: usize,
) -> Vec<f32> {
    if n <= 1 {
        return data;
    }
    let h = cfg.hidden_size;
    // Shared-expert MLP (Megatron col/row split).
    if name.ends_with(".shared_experts.up_proj.weight") {
        // out = shared_inter (col-shard), in = hidden.
        return col_shard(&data, h, r, n);
    }
    if name.ends_with(".shared_experts.down_proj.weight") {
        let in_f = cfg.moe_shared_expert_intermediate_size;
        assert_eq!(in_f % n, 0,
            "shared down_proj: shared_inter {in_f} not divisible by tp {n}");
        return row_shard(&data, in_f, r, n);
    }
    // REPLICATED under this TP scope: mamba mixer (in/out_proj, conv1d, A_log/D/
    // dt_bias, norm), latent-MoE fc1/fc2, router gate/e_score, all layernorms,
    // embeddings, lm_head. Returned unchanged. (The mamba head-shard math lives
    // in `nem_tp_shard`/`mamba_head_shard` and is Mac-proven, but wiring it into
    // the resident scan requires halving the runtime dims — deferred.)
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cpu_matmul;
    use crate::nemotron::{
        latent_moe_forward, mamba2_decode_step, router_forward, LatentMoeDims, LatentMoeWeights, RouterDims,
    };
    use serde_json::Value;

    // Reuse the existing mamba golden fixture (nh=4/ng=2 — TP=2-clean per the
    // plan §0 table: n_groups/2=1 whole group/rank, norm_group_size=64/2=32
    // preserved) rather than synthesizing a new one.
    const MAMBA_FIXTURE: &str = include_str!("../nemotron_ref/fixtures/mamba2_decode_step.json");
    // Reuse the existing latent-moe golden fixture (ne=6/top_k=2/n_group=1 —
    // EP=2-clean per the plan §0 table: 3 experts/rank, plain top-k matching
    // the real model's n_group=1) rather than synthesizing a new one.
    const LATENT_MOE_FIXTURE: &str = include_str!("../nemotron_ref/fixtures/latent_moe_forward.json");

    fn arr(v: &Value, key: &str) -> Vec<f32> {
        v[key]["data"]
            .as_array()
            .unwrap_or_else(|| panic!("fixture key '{key}' has no data array"))
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }
    fn su(v: &Value, key: &str) -> usize {
        v[key].as_u64().unwrap() as usize
    }
    fn sf(v: &Value, key: &str) -> f32 {
        v[key].as_f64().unwrap() as f32
    }
    fn sb(v: &Value, key: &str) -> bool {
        v[key].as_bool().unwrap()
    }

    /// Deterministic pseudo-random f32 stream (LCG), same pattern as
    /// `nemotron.rs:2264` `lcg_fill` — no `rand` dep, bit-reproducible.
    fn lcg_fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * scale
            })
            .collect()
    }

    /// Elementwise host-add of the two ranks' partials — the all-reduce
    /// equivalent for the SUM (row-parallel) combine mode. On the cluster this
    /// becomes `send_f32`/`recv_f32(tp_peer_global_rank)` + add
    /// (vccl_ffi.rs:491/516); the numeric result is identical, only the
    /// transport differs (see module doc).
    fn combine_sum(parts: &[Vec<f32>]) -> Vec<f32> {
        assert!(!parts.is_empty());
        let mut out = parts[0].clone();
        for p in &parts[1..] {
            assert_eq!(p.len(), out.len());
            for (o, &v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        out
    }

    /// CONCAT (column-parallel) combine: every element is computed identically
    /// regardless of which shard produced it (no term reordering), so this is
    /// the BIT-EXACT gate.
    fn assert_bit_exact(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "length mismatch: got {} want {}", got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g, w, "element {i} not bit-exact: got {g} want {w}");
        }
    }

    /// SUM (row-parallel / additive) combine: the monolithic reduction is
    /// `Σ_{k=0..K}` sequentially, the sharded reduction is
    /// `(Σ_{k=0..K/2}) + (Σ_{K/2..K})` then added — FP add is non-associative,
    /// so bit-exactness does not hold. This is exactly the reorder a real
    /// all-reduce introduces, so cosine similarity is the honest gate.
    fn assert_cos_ge(got: &[f32], want: &[f32], thr: f32) {
        assert_eq!(got.len(), want.len());
        let dot: f32 = got.iter().zip(want).map(|(&a, &b)| a * b).sum();
        let na: f32 = got.iter().map(|&a| a * a).sum::<f32>().sqrt();
        let nb: f32 = want.iter().map(|&b| b * b).sum::<f32>().sqrt();
        let c = dot / (na * nb);
        assert!(c >= thr, "cos {c} < threshold {thr}");
    }

    /// **Shard-reassembly bit-exact gate (the Mac-validatable TP correctness
    /// anchor, mirroring gemma's tp-shard gate).** For a REPRESENTATIVE
    /// nemotron matmul in each quantization it ships in — NVFP4 (routed
    /// experts / EP whole-expert), FP8-E4M3 (mamba proj / shared experts
    /// pre-requant), and f16 (attn q/k/v, fc1/fc2) — take the QUANTIZED weight,
    /// column-shard it into TP=2 halves at the byte layer, DEQUANT each half
    /// independently with the real production dequant fn, reassemble (concat),
    /// and assert the result is BIT-IDENTICAL (`to_bits()`, maxdiff 0) to the
    /// dequant of the full tensor. This is exact because every dequant here is
    /// per-output-row independent (NVFP4: own packed row + own wscale row;
    /// FP8/f16: own row), so a contiguous output-row (column-parallel) split
    /// introduces no reordering. It directly proves the load-time shard's
    /// byte-layout carry (packed nibbles, F8 wscale rows, F8 per-tensor scalar,
    /// f16 bits) is correct — the exact class of bug (wrong stride / dropped
    /// group / scale mis-slice) that would silently corrupt a TP run.
    #[test]
    fn tp2_quant_shard_reassembly_bit_exact() {
        // Representative small shapes standing in for the real nemotron tensors
        // (out divisible by 2 for the TP=2 output-row split; in divisible by the
        // NVFP4 group size 16 and even for the nibble pack).
        let n = 2usize;

        // ── NVFP4 (E2M1 nibbles + per-group F8 wscale + global F32) — routed
        //    experts. dequantize_nvfp4(packed[out,in/2], wscale[out,in/gs] bytes,
        //    global, out, in, gs). Column-shard = output-row byte slice. ──
        {
            let (out_f, in_f, gs) = (8usize, 32usize, 16usize);
            let bpr = in_f / 2;
            let groups = in_f / gs;
            let packed: Vec<u8> = (0..out_f * bpr).map(|i| (i * 37 + 11) as u8).collect();
            let wscale: Vec<u8> = (0..out_f * groups).map(|i| (i * 53 + 7) as u8).collect();
            let global = 0.137f32;
            let full = crate::model::dequantize_nvfp4(&packed, &wscale, global, out_f, in_f, gs);
            let per = out_f / n;
            let mut reassembled = Vec::with_capacity(out_f * in_f);
            for r in 0..n {
                let lo = r * per;
                let p_half = &packed[lo * bpr..(lo + per) * bpr];
                let s_half = &wscale[lo * groups..(lo + per) * groups];
                let deq = crate::model::dequantize_nvfp4(p_half, s_half, global, per, in_f, gs);
                reassembled.extend_from_slice(&deq);
            }
            assert_eq!(reassembled.len(), full.len());
            for (i, (&g, &w)) in reassembled.iter().zip(&full).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(),
                    "NVFP4 col-shard elem {i}: {g} != {w} (not bit-exact)");
            }
        }

        // ── FP8-E4M3 (per-tensor SCALAR scale, W8A16) — mamba/shared pre-requant.
        //    dequantize_fp8(weight[out,in] bytes, scale, out, in). Scalar scale is
        //    replicated to both shards unchanged. ──
        {
            let (out_f, in_f) = (8usize, 24usize);
            let weight: Vec<u8> = (0..out_f * in_f).map(|i| (i * 41 + 3) as u8).collect();
            let scale = vec![0.0219f32]; // per-tensor scalar (shape []), len 1
            let full = crate::model::dequantize_fp8(&weight, &scale, out_f, in_f);
            let per = out_f / n;
            let mut reassembled = Vec::with_capacity(out_f * in_f);
            for r in 0..n {
                let lo = r * per;
                let w_half = &weight[lo * in_f..(lo + per) * in_f];
                let deq = crate::model::dequantize_fp8(w_half, &scale, per, in_f); // scalar replicated
                reassembled.extend_from_slice(&deq);
            }
            for (i, (&g, &w)) in reassembled.iter().zip(&full).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(),
                    "FP8 col-shard elem {i}: {g} != {w} (not bit-exact)");
            }
        }

        // ── f16 (BF16→f16 bits) — attn q/k/v, fc1/fc2. The loader decodes to f32
        //    and re-encodes f16; sharding the f32 by output rows is a pure slice
        //    → bit-exact reassembly (this is the col_shard primitive on f32). ──
        {
            let (out_f, in_f) = (8usize, 16usize);
            let w: Vec<f32> = lcg_fill(out_f * in_f, 0xC0DE, 1.0);
            let per = out_f / n;
            let mut reassembled = Vec::with_capacity(out_f * in_f);
            for r in 0..n {
                reassembled.extend_from_slice(&col_shard(&w, in_f, r, n));
            }
            for (i, (&g, &o)) in reassembled.iter().zip(&w).enumerate() {
                assert_eq!(g.to_bits(), o.to_bits(),
                    "f16/f32 col-shard elem {i}: not bit-exact");
            }
        }
    }

    /// EP whole-expert range partition covers `[0, ne)` exactly, contiguous,
    /// disjoint, and balanced — guards the loader's per-expert streaming skip.
    #[test]
    fn ep_owned_range_covers_and_disjoint() {
        for (ne, n) in [(512usize, 2usize), (6, 2), (512, 4), (8, 4)] {
            let mut seen = vec![false; ne];
            let mut prev_hi = 0usize;
            for r in 0..n {
                let (lo, cnt) = expert_owned_range(ne, r, n);
                assert_eq!(lo, prev_hi, "ne={ne} n={n} r={r}: not contiguous");
                assert_eq!(cnt, ne / n, "ne={ne} n={n} r={r}: unbalanced");
                for e in lo..lo + cnt {
                    assert!(!seen[e], "expert {e} owned twice");
                    seen[e] = true;
                }
                prev_hi = lo + cnt;
            }
            assert_eq!(prev_hi, ne, "ne={ne} n={n}: did not cover all experts");
            assert!(seen.iter().all(|&b| b), "ne={ne} n={n}: gap in coverage");
        }
    }

    /// Inc 1 machinery smoke test: proves both combine modes on a bare GEMM,
    /// no mamba/moe intricacy. Column-parallel output rows concat to a
    /// BIT-EXACT match (no reorder — see `assert_bit_exact` doc); row-parallel
    /// input-column partials all-reduce (host-add here; `send_f32`/`recv_f32`
    /// on the cluster — see module doc) to a cos >= 0.99999 match (FP-add
    /// reorder — see `assert_cos_ge` doc).
    #[test]
    fn tp_gemm_shard_invariance() {
        let (in_feat, out_feat, t, n) = (64usize, 32usize, 1usize, 2usize);
        let x = lcg_fill(t * in_feat, 0xF00D, 1.0);
        let w = lcg_fill(out_feat * in_feat, 0xBEEF, 0.1);

        let monolithic = cpu_matmul(&x, &w, t, in_feat, out_feat);

        // Column-parallel: each rank keeps a contiguous output-row slice of W,
        // computes its own output rows, concat -> compare by global index.
        let mut concatenated = Vec::with_capacity(out_feat);
        for r in 0..n {
            let w_local = col_shard(&w, in_feat, r, n);
            let out_local = out_feat / n;
            let part = cpu_matmul(&x, &w_local, t, in_feat, out_local);
            concatenated.extend_from_slice(&part);
        }
        assert_bit_exact(&concatenated, &monolithic);

        // Row-parallel: each rank keeps a contiguous input-column slice of x
        // and W, computes a full-length partial, combine_sum -> compare cos.
        let mut partials = Vec::with_capacity(n);
        for r in 0..n {
            let w_local = row_shard(&w, in_feat, r, n);
            let in_local = in_feat / n;
            let lo = r * in_local;
            let x_local: Vec<f32> = x[lo..lo + in_local].to_vec();
            let part = cpu_matmul(&x_local, &w_local, t, in_local, out_feat);
            partials.push(part);
        }
        let summed = combine_sum(&partials);
        assert_cos_ge(&summed, &monolithic, 0.99999);
    }

    /// Increment 2 shard-invariance proof: TP=2 Mamba2 head-shard ==
    /// monolithic, on the existing golden fixture (nh=4/ng=2, TP=2-clean).
    ///
    /// (a) in_proj CONCAT is BIT-EXACT by global index (no reorder — the
    ///     5-segment column split just partitions which output rows each
    ///     rank computes).
    /// (b) the full decode-step SUM-combine (this rank's `mamba2_decode_step`
    ///     out_proj-partial return values, host-added) is cos >= 0.99999 vs
    ///     the monolithic decode step, for both a zero-start and a
    ///     nonzero-start state (mirrors the prefill test's two-state pattern,
    ///     nemotron.rs:2050–2062).
    /// (c) [strong check] the two ranks' advanced `ssm_state`, concatenated by
    ///     head, is BIT-EXACT vs the monolithic advanced state — each head's
    ///     recurrence is fully independent, so no reorder is introduced here
    ///     either; this guards a silent head/group index-map bug that (b)'s
    ///     cos tolerance alone might mask.
    #[test]
    fn mamba2_head_shard_tp2_matches_monolithic() {
        let v: Value = serde_json::from_str(MAMBA_FIXTURE).unwrap();
        let dims = Mamba2Dims {
            hidden_size: su(&v, "hidden_size"),
            num_heads: su(&v, "num_heads"),
            head_dim: su(&v, "head_dim"),
            ssm_state_size: su(&v, "ssm_state_size"),
            n_groups: su(&v, "n_groups"),
            conv_kernel: su(&v, "conv_kernel_size"),
            time_step_min: sf(&v, "time_step_min"),
            eps: sf(&v, "layer_norm_epsilon"),
        };
        let x = arr(&v, "x");
        let in_proj = arr(&v, "in_proj_weight");
        let conv_w = arr(&v, "conv1d_weight");
        let conv_b = arr(&v, "conv1d_bias");
        let a_log = arr(&v, "A_log");
        let d_skip = arr(&v, "D");
        let dt_bias = arr(&v, "dt_bias");
        let norm_w = arr(&v, "norm_weight");
        let out_proj = arr(&v, "out_proj_weight");
        let weights = Mamba2Weights {
            in_proj: &in_proj,
            conv1d_weight: &conv_w,
            conv1d_bias: Some(&conv_b),
            a_log: &a_log,
            d: &d_skip,
            dt_bias: &dt_bias,
            norm_weight: &norm_w,
            out_proj: &out_proj,
        };
        let n = 2usize;
        assert_eq!(dims.num_heads % n, 0);
        assert_eq!(dims.n_groups % n, 0);

        // (a) in_proj CONCAT — BIT-EXACT by global index.
        let monolithic_in_proj = cpu_matmul(&x, &in_proj, 1, dims.hidden_size, dims.in_proj_out());
        for r in 0..n {
            let (local, _dl) = mamba_head_shard(&weights, &dims, r, n);
            let local_out = local.in_proj.len() / dims.hidden_size;
            let proj_local = cpu_matmul(&x, &local.in_proj, 1, dims.hidden_size, local_out);
            let expected = gather_rows(&monolithic_in_proj, 1, &in_proj_seg_ranges(&dims, r, n));
            assert_bit_exact(&proj_local, &expected);
        }

        // (b) full decode-step SUM-combine — cos >= 0.99999 — and
        // (c) ssm_state CONCAT — BIT-EXACT — for both zero- and nonzero-start.
        for nonzero_start in [false, true] {
            let mono_state_start = if nonzero_start {
                Mamba2State {
                    conv_state: arr(&v, "conv_state_in"),
                    ssm_state: arr(&v, "ssm_state_in"),
                }
            } else {
                Mamba2State::zeros(&dims)
            };
            let mut state_mono = mono_state_start.clone();
            let mono_out = mamba2_decode_step(&x, &weights, &mut state_mono, &dims);

            let mut partials = Vec::with_capacity(n);
            let mut ssm_concat = vec![0.0f32; dims.num_heads * dims.head_dim * dims.ssm_state_size];
            let mut nh_l_seen = 0usize;
            for r in 0..n {
                let (local, dl) = mamba_head_shard(&weights, &dims, r, n);
                let mut state_local = if nonzero_start {
                    mamba_state_shard(&mono_state_start, &dims, r, n)
                } else {
                    Mamba2State::zeros(&dl)
                };
                let part = mamba2_decode_step(&x, &local.borrow(), &mut state_local, &dl);
                partials.push(part);

                let per = dl.num_heads * dl.head_dim * dl.ssm_state_size;
                let lo = r * per;
                ssm_concat[lo..lo + per].copy_from_slice(&state_local.ssm_state);
                nh_l_seen = dl.num_heads;
            }
            assert_eq!(nh_l_seen * n, dims.num_heads);

            let summed = combine_sum(&partials);
            assert_cos_ge(&summed, &mono_out, 0.99999);
            assert_bit_exact(&ssm_concat, &state_mono.ssm_state);
        }
    }

    /// Increment 3 shard-invariance proof: EP=2 latent-MoE (expert-half
    /// partition + routed sum-combine) == monolithic `latent_moe_forward`, on
    /// the existing golden fixture (ne=6/top_k=2/n_group=1, EP=2-clean per
    /// plan §0: 3 experts/rank, plain top-k).
    ///
    /// - Router replication check: both "ranks" `router_forward` on the same
    ///   token give IDENTICAL `indices`/`weights` (replicated, no sharding —
    ///   trivially equal, but guards a divergence bug).
    /// - (a) the routed `[lat]` accumulator: `combine_sum` of the two owned-half
    ///   partials (`moe_expert_half` + `latent_moe_routed_partial`) vs the
    ///   monolithic routed accumulator (same loop, unrestricted) — cos >=
    ///   0.99999 (SUM combine, plan §1 — FP add reordered across the
    ///   expert-half boundary, not bit-exact).
    /// - (b) the full mixer output (`fc2(routed_full) + shared`, shared
    ///   REPLICATED bit-identically) vs `latent_moe_forward` — cos >= 0.99999.
    #[test]
    fn latent_moe_ep2_matches_monolithic() {
        let v: Value = serde_json::from_str(LATENT_MOE_FIXTURE).unwrap();
        let hs = su(&v, "hidden_size");
        let lat = su(&v, "moe_latent_size");
        let inter = su(&v, "moe_intermediate_size");
        let shared_inter = su(&v, "moe_shared_expert_intermediate_size");
        let ne = su(&v, "n_routed_experts");
        let dims = LatentMoeDims {
            hidden_size: hs,
            moe_latent_size: lat,
            moe_intermediate_size: inter,
            moe_shared_expert_intermediate_size: shared_inter,
            router: RouterDims {
                n_routed_experts: ne,
                top_k: su(&v, "num_experts_per_tok"),
                routed_scaling_factor: sf(&v, "routed_scaling_factor"),
                n_group: su(&v, "n_group"),
                topk_group: su(&v, "topk_group"),
                norm_topk_prob: sb(&v, "norm_topk_prob"),
            },
        };

        let x = arr(&v, "x");
        let n_tokens = x.len() / hs;
        let gate_weight = arr(&v, "gate_weight");
        let e_bias = arr(&v, "e_score_correction_bias");
        let fc1 = arr(&v, "fc1_latent_proj");
        let fc2 = arr(&v, "fc2_latent_proj");
        let expert_up = arr(&v, "expert_up_proj");
        let expert_down = arr(&v, "expert_down_proj");
        let shared_up = arr(&v, "shared_up_proj");
        let shared_down = arr(&v, "shared_down_proj");

        let mono_weights = LatentMoeWeights {
            gate_weight: &gate_weight,
            e_score_correction_bias: &e_bias,
            fc1_latent_proj: &fc1,
            fc2_latent_proj: &fc2,
            expert_up: &expert_up,
            expert_down: &expert_down,
            shared_up: &shared_up,
            shared_down: &shared_down,
        };

        let n = 2usize;
        assert_eq!(ne % n, 0, "fixture n_routed_experts must be EP=2-clean");
        let per = ne / n;

        // Pre-shard the expert-half buffers once (per rank) — mirrors the
        // resident loader partitioning `gpu_experts[layer]` once at load time,
        // not per-token (plan §4 "Load-time" note).
        let halves: Vec<(Vec<f32>, Vec<f32>)> =
            (0..n).map(|r| moe_expert_half(&expert_up, &expert_down, ne, lat, inter, r, n)).collect();

        let expected_out = latent_moe_forward(&x, n_tokens, &mono_weights, &dims);

        for t in 0..n_tokens {
            let htok = &x[t * hs..(t + 1) * hs];

            // Router replication check + monolithic router selection (both
            // ranks and the monolithic reference all call the SAME
            // `router_forward` — replicated, so trivially identical; guards a
            // divergence bug rather than testing a real split).
            let (mono_idx, mono_w) = router_forward(htok, &gate_weight, &e_bias, &dims.router);
            for r in 0..n {
                let (idx_r, w_r) = router_forward(htok, &gate_weight, &e_bias, &dims.router);
                let _ = r;
                assert_eq!(idx_r, mono_idx, "router indices diverged between replicated ranks");
                assert_eq!(w_r, mono_w, "router weights diverged between replicated ranks");
            }

            // Monolithic routed[lat] accumulator (same math as
            // `latent_moe_forward`'s inner loop, unrestricted to any expert
            // half) — the (a) intermediate-check baseline.
            let latent = cpu_matmul(htok, &fc1, 1, hs, lat);
            let expert_up_stride = inter * lat;
            let expert_down_stride = lat * inter;
            let mut routed_mono = vec![0.0f32; lat];
            for (k, &e) in mono_idx.iter().enumerate() {
                let up = &expert_up[e * expert_up_stride..(e + 1) * expert_up_stride];
                let down = &expert_down[e * expert_down_stride..(e + 1) * expert_down_stride];
                let eout = mlp_relu2(&latent, up, down, lat, inter);
                let wk = mono_w[k];
                for (r_acc, &o) in routed_mono.iter_mut().zip(&eout) {
                    *r_acc += o * wk;
                }
            }

            // Sharded EP=2: each rank restricts the SAME (idx, weights)
            // selection to its owned expert-id half, using the SAME router
            // gate weight per owned expert (no re-normalization within the
            // half — the plan's flagged common trap).
            let mut partials = Vec::with_capacity(n);
            for r in 0..n {
                let (up_local, down_local) = &halves[r];
                let owned_lo = r * per;
                let owned_hi = owned_lo + per;
                let part = latent_moe_routed_partial(
                    &latent, &mono_idx, &mono_w, up_local, down_local, &dims, owned_lo, owned_hi,
                );
                partials.push(part);
            }
            let routed_full = combine_sum(&partials);
            assert_cos_ge(&routed_full, &routed_mono, 0.99999);

            // (b) full mixer output: fc2(routed_full) + shared (REPLICATED,
            // bit-identical on both ranks) vs `latent_moe_forward`.
            let moe_out = cpu_matmul(&routed_full, &fc2, 1, lat, hs);
            let shared = mlp_relu2(htok, &shared_up, &shared_down, hs, shared_inter);
            let mut out_sharded = vec![0.0f32; hs];
            for i in 0..hs {
                out_sharded[i] = moe_out[i] + shared[i];
            }
            let expected_tok = &expected_out[t * hs..(t + 1) * hs];
            assert_cos_ge(&out_sharded, expected_tok, 0.99999);
        }
    }
}
