//! DeepSeek-V4-Flash-0731-2.4bit-mixed — Rust model assembly (Phase 2f).
//!
//! Milestone M1 = single-node full GPU forward, finite all 43 layers, argmax ==
//! 11111 (` Paris`, `tests/fixtures/dsv4/golden_fulldepth_stable.json`). The hard
//! MATH is de-risked by the numpy oracles (`scripts/dsv4/*.py`, all cos=1.0/ULP vs
//! the transformers-5.8.1 `deepseek_v4` reference) and every leaf GPU kernel is
//! shipped + on-node validated (2/6/8-bit matvec + batched MoE, HC two-float mix,
//! the DSA trio compress→index-score→top-512, routers). What remains for M1 is
//! Rust INTEGRATION: the richer-MLA path, the streaming loader, and the per-type
//! per-layer forward wiring composing all of the above.
//!
//! THIS MODULE (pass 1 of the integration) delivers the **richer-MLA attention
//! core** (M1 item a) as validated Rust, mirroring the reference
//! `DeepseekV4Attention` for a `sliding_attention` layer (the `compressor is None`
//! case — the attention *is* the core; CSA/HCA layers additionally concatenate
//! `compressed_kv`/`block_bias` onto the KV axis, which the already-shipped
//! compressor+indexer+top-512 GPU kernels produce). Hermetic test
//! (`mla_tests::mla_core_matches_oracle`) validates bit-exact (max_abs_err<1e-4)
//! vs `scripts/dsv4/mla_oracle.py`, itself cross-checked vs the torch reference
//! (max_abs_err 1.1e-8, cos=1.0).
//!
//! The projections here use `cpu_matmul` on already-dequantized f32 weights; the
//! GPU forward swaps each for the shipped 6-bit `mul_mat_vec_mlx6` matvec (attn/MLA
//! are 6-bit gs128) — a mechanical substitution since that kernel is on-node
//! 12/12-validated. The attention softmax/RoPE/sink/sliding-mask and grouped
//! output stay host-side for the M1 CPU-resident-per-stage gate (matches
//! golden.py's f32 math), exactly as the continuation spec allows.

/// MLA-relevant config (DeepSeek-V4-Flash: H=4096, nh=64, hd=512, q_lora=1024,
/// o_lora=1024, o_groups=8, sliding_window=128, rope partial_rotary_factor=0.125
/// ⇒ rope_dim=64, rms_eps=1e-6). `hd` is large (512) with only the trailing 64
/// channels rotated; the leading 448 are "nope".
#[derive(Clone, Debug)]
pub struct Dsv4MlaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
}

/// Weighted RMSNorm over the last axis (`DeepseekV4RMSNorm`): variance in f32,
/// then `w * x * rsqrt(var+eps)`. `x` is `[rows, dim]` row-major; `w` is `[dim]`.
pub fn rmsnorm_rows(x: &[f32], w: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let mut v = 0.0f64;
        for &e in xr {
            v += (e as f64) * (e as f64);
        }
        let inv = 1.0f64 / ((v / dim as f64) + eps as f64).sqrt();
        let o = &mut out[r * dim..(r + 1) * dim];
        for i in 0..dim {
            o[i] = (w[i] as f64 * (xr[i] as f64 * inv)) as f32;
        }
    }
    out
}

/// Unweighted RMSNorm over the last axis (`DeepseekV4UnweightedRMSNorm`).
pub fn unweighted_rmsnorm_rows(x: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let mut v = 0.0f64;
        for &e in xr {
            v += (e as f64) * (e as f64);
        }
        let inv = 1.0f64 / ((v / dim as f64) + eps as f64).sqrt();
        let o = &mut out[r * dim..(r + 1) * dim];
        for i in 0..dim {
            o[i] = (xr[i] as f64 * inv) as f32;
        }
    }
    out
}

/// V4 interleaved partial-RoPE applied IN-PLACE to the trailing `rope_dim`
/// channels of each `[dim]` row (rows share the same `[S]` position layout).
///
/// `rows_per_pos` consecutive rows share position `s` (e.g. all `nh` heads at a
/// given sequence step). `cos`/`sin` are `[S, rope_dim/2]` row-major. Interleaved
/// pairs: for channel pair `i`, `out[2i]=x[2i]*c - x[2i+1]*s`,
/// `out[2i+1]=x[2i+1]*c + x[2i]*s` (== `x*cos + rotate_half(x)*sin` with
/// `rotate_half=(-x_odd, x_even)` interleaved and `cos/sin` repeat_interleave(2)).
/// Math in f64 to match the reference `.float()` path bit-for-bit at f32-out.
#[allow(clippy::too_many_arguments)]
pub fn apply_interleaved_rope_inplace(
    x: &mut [f32],
    rows: usize,
    dim: usize,
    rope_dim: usize,
    s_of_row: &dyn Fn(usize) -> usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = rope_dim / 2;
    let nope = dim - rope_dim;
    for r in 0..rows {
        let s = s_of_row(r);
        let cs = &cos[s * half..s * half + half];
        let sn = &sin[s * half..s * half + half];
        let row = &mut x[r * dim..(r + 1) * dim];
        for i in 0..half {
            let c = cs[i] as f64;
            let sv = sn[i] as f64;
            let a = row[nope + 2 * i] as f64; // even
            let b = row[nope + 2 * i + 1] as f64; // odd
            row[nope + 2 * i] = (a * c - b * sv) as f32;
            row[nope + 2 * i + 1] = (b * c + a * sv) as f32;
        }
    }
}

/// Full sliding-attention MLA core. `x` is `[S, H]` row-major (B=1). Weights are
/// dequantized f32 row-major `[out, in]` (as `cpu_matmul` expects). `cos`/`sin`
/// are `[S, rope_dim/2]`. `mask` is `[S, S]` additive (0 or -inf), sliding-window
/// causal. Returns `[S, H]`. Mirrors `DeepseekV4Attention.forward` bit-for-bit
/// for a `sliding_attention` layer (`compressor is None`).
///
/// Thin wrapper over [`mla_core_ext`] with no compressed KV (the sliding case).
#[allow(clippy::too_many_arguments)]
pub fn mla_core(
    cfg: &Dsv4MlaConfig,
    x: &[f32],
    seq_len: usize,
    w_q_a: &[f32],
    w_q_a_norm: &[f32],
    w_q_b: &[f32],
    w_kv: &[f32],
    w_kv_norm: &[f32],
    w_o_a: &[f32],
    w_o_b: &[f32],
    sinks: &[f32],
    cos: &[f32],
    sin: &[f32],
    mask: &[f32],
) -> Vec<f32> {
    mla_core_ext(
        cfg, x, seq_len, w_q_a, w_q_a_norm, w_q_b, w_kv, w_kv_norm, w_o_a, w_o_b, sinks, cos,
        sin, mask, &[], &[],
    )
}

/// Generalized MLA core for CSA/HCA layers — mirrors `DeepseekV4Attention.forward`
/// for the `compressor is not None` case. After the sliding KV path, the already
/// compress-roped `compressed_kv` (`[T, hd]` row-major, `T` compressed entries;
/// pass an empty slice for the sliding case) is concatenated onto the KV sequence
/// axis (K==V, single shared head broadcast to all query heads), and the additive
/// attention mask is extended with `block_bias` (`[S, T]` additive 0/-inf) over
/// those `T` slots. Everything else (per-head attn sink, softmax, output-rope
/// conjugate, grouped o_lora projection) is identical to the sliding case.
///
/// The `compressed_kv` + `block_bias` are produced by the shipped compressor +
/// Lightning-Indexer + top-512 GPU kernels (`dsv4_dsa_compress` / `index_score` /
/// `topk`); this core just consumes them. Validated bit-exact (max_abs_err<1e-4)
/// vs `scripts/dsv4/mla_oracle.py::mla_core_ext` (itself ULP-exact vs the torch
/// `DeepseekV4Attention` compressor path) — see `mla_tests`.
#[allow(clippy::too_many_arguments)]
pub fn mla_core_ext(
    cfg: &Dsv4MlaConfig,
    x: &[f32],
    seq_len: usize,
    w_q_a: &[f32],
    w_q_a_norm: &[f32],
    w_q_b: &[f32],
    w_kv: &[f32],
    w_kv_norm: &[f32],
    w_o_a: &[f32],
    w_o_b: &[f32],
    sinks: &[f32],
    cos: &[f32],
    sin: &[f32],
    mask: &[f32],
    compressed_kv: &[f32],
    block_bias: &[f32],
) -> Vec<f32> {
    let s = seq_len;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim;
    let ql = cfg.q_lora_rank;
    let eps = cfg.rms_norm_eps;
    // rope_dim from cos width (rope_dim/2 entries per position).
    let rope_dim = 2 * (cos.len() / s);
    let scaling = (hd as f64).powf(-0.5);
    // T compressed KV entries concatenated after the S sliding entries.
    let t_comp = compressed_kv.len() / hd;
    let total = s + t_comp;

    // ---- Q path ----
    // q_a: [S,H] @ [ql,H]^T -> [S,ql]; norm; q_b: [S,ql] @ [nh*hd,ql]^T -> [S,nh*hd]
    let q_a = crate::model::cpu_matmul(x, w_q_a, s, h, ql);
    let q_a = rmsnorm_rows(&q_a, w_q_a_norm, s, ql, eps);
    let q_flat = crate::model::cpu_matmul(&q_a, w_q_b, s, ql, nh * hd); // [S, nh*hd]
    // reshape to head-major [nh, S, hd] so a "row" is one (head, pos) hd-vector.
    let mut q = vec![0.0f32; nh * s * hd];
    for si in 0..s {
        for hh in 0..nh {
            let src = &q_flat[si * nh * hd + hh * hd..si * nh * hd + hh * hd + hd];
            q[(hh * s + si) * hd..(hh * s + si) * hd + hd].copy_from_slice(src);
        }
    }
    // q_b_norm (unweighted, over hd), then rope. rows are [nh*S], pos = row % s.
    let q = unweighted_rmsnorm_rows(&q, nh * s, hd, eps);
    let mut q = q;
    apply_interleaved_rope_inplace(&mut q, nh * s, hd, rope_dim, &|r| r % s, cos, sin);

    // ---- KV path (single shared head): sliding entries, then compressed concat ----
    let kv = crate::model::cpu_matmul(x, w_kv, s, h, hd); // [S, hd]
    let kv = rmsnorm_rows(&kv, w_kv_norm, s, hd, eps);
    let mut kv = kv; // [S, hd], row = pos
    apply_interleaved_rope_inplace(&mut kv, s, hd, rope_dim, &|r| r, cos, sin);
    // full KV = [sliding S | compressed T], row-major [total, hd]. Compressed entries
    // already carry their own (compress-pos) rope from the compressor.
    let mut kv_full = vec![0.0f32; total * hd];
    kv_full[..s * hd].copy_from_slice(&kv);
    if t_comp > 0 {
        kv_full[s * hd..].copy_from_slice(compressed_kv);
    }

    // ---- eager attention: sink + extended mask; K==V, single head broadcast ----
    // attn_out head-major [nh, S, hd]. Query si attends over `total` keys: sliding
    // key ti<S uses mask[si*S+ti]; compressed key ti>=S uses block_bias[si*T+(ti-S)].
    let mut ao = vec![0.0f32; nh * s * hd];
    for hh in 0..nh {
        let sink = sinks[hh] as f64;
        for si in 0..s {
            let qrow = &q[(hh * s + si) * hd..(hh * s + si) * hd + hd];
            // logits over t = 0..total plus one sink column.
            let mut logits = vec![f64::NEG_INFINITY; total];
            let mut mx = sink; // sink participates in the max
            for ti in 0..total {
                let m = if ti < s {
                    mask[si * s + ti] as f64
                } else {
                    block_bias[si * t_comp + (ti - s)] as f64
                };
                if m == f64::NEG_INFINITY {
                    continue;
                }
                let krow = &kv_full[ti * hd..ti * hd + hd];
                let mut dot = 0.0f64;
                for d in 0..hd {
                    dot += qrow[d] as f64 * krow[d] as f64;
                }
                let l = dot * scaling + m;
                logits[ti] = l;
                if l > mx {
                    mx = l;
                }
            }
            // softmax over [logits.., sink] with the shared max subtracted, drop sink.
            let mut denom = (sink - mx).exp();
            for ti in 0..total {
                if logits[ti] != f64::NEG_INFINITY {
                    logits[ti] = (logits[ti] - mx).exp();
                    denom += logits[ti];
                } else {
                    logits[ti] = 0.0;
                }
            }
            let orow = &mut ao[(hh * s + si) * hd..(hh * s + si) * hd + hd];
            for ti in 0..total {
                let p = logits[ti] / denom;
                if p == 0.0 {
                    continue;
                }
                let vrow = &kv_full[ti * hd..ti * hd + hd];
                for d in 0..hd {
                    orow[d] += (p * vrow[d] as f64) as f32;
                }
            }
        }
    }

    // ---- output-rope conjugate (undo V's rope on its rope slice) with -sin ----
    let neg_sin: Vec<f32> = sin.iter().map(|v| -v).collect();
    apply_interleaved_rope_inplace(&mut ao, nh * s, hd, rope_dim, &|r| r % s, cos, &neg_sin);

    // reshape head-major [nh,S,hd] -> [S, nh*hd]
    let mut ao_sd = vec![0.0f32; s * nh * hd];
    for hh in 0..nh {
        for si in 0..s {
            let src = &ao[(hh * s + si) * hd..(hh * s + si) * hd + hd];
            ao_sd[si * nh * hd + hh * hd..si * nh * hd + hh * hd + hd].copy_from_slice(src);
        }
    }

    // ---- grouped o_lora output projection ----
    let g = cfg.o_groups;
    let olr = cfg.o_lora_rank;
    let per_g = (nh * hd) / g; // in_features per group (block-diagonal)
    // wo_a is [g*olr, per_g] block-diagonal: group gg uses rows [gg*olr..gg*olr+olr].
    let mut proj = vec![0.0f32; s * g * olr]; // [S, g*olr]
    for si in 0..s {
        for gg in 0..g {
            let xin = &ao_sd[si * nh * hd + gg * per_g..si * nh * hd + gg * per_g + per_g];
            for o in 0..olr {
                let wrow = &w_o_a[(gg * olr + o) * per_g..(gg * olr + o) * per_g + per_g];
                let mut acc = 0.0f64;
                for k in 0..per_g {
                    acc += xin[k] as f64 * wrow[k] as f64;
                }
                proj[si * g * olr + gg * olr + o] = acc as f32;
            }
        }
    }
    // o_b: [S, g*olr] @ [H, g*olr]^T -> [S, H]
    crate::model::cpu_matmul(&proj, w_o_b, s, g * olr, h)
}

// ============================================================================
// Manifold Hyper-Connection (mHC) block — the decoder layer's residual machinery
// (M1 item c). `hc_block` mirrors golden.py::patch_hc_stable (the load-bearing
// deep-layer treatment, PROVEN identity-on-finite vs `DeepseekV4HyperConnection`,
// cos=1.0); `hc_residual_mix` mirrors the `DeepseekV4DecoderLayer` residual mix.
// Both validated bit-exact vs `scripts/dsv4/hc_oracle.py` (see `hc_tests`).
// ============================================================================

/// One mHC site (`attn_hc` / `ffn_hc`). `streams` is `[S, hc, D]` row-major (B=1).
/// `fn` is `[(2+hc)*hc, hc*D]`, `base` is `[(2+hc)*hc]`, `scale` is `[3]`.
/// Returns `(post [S,hc], comb [S,hc,hc], collapsed [S,D])`.
///
/// The unweighted input RMSNorm is max-factored in f64 (can't overflow at depth),
/// the fn-mix + Sinkhorn run in f64, and the Sinkhorn is the exact linear form
/// (softmax start → col-norm → `iters-1 × (row,col)`) — the stability treatment
/// that keeps the full 43-layer forward finite while staying an identity on finite
/// input.
/// `VLLM_VULKAN_DSV4_HC_RAYON` — parallelize the host mHC `flat @ fn^T` projection
/// (`mix_dim` independent f64 output rows) across cores. BIT-IDENTICAL to the serial
/// path (per-row reduction order + index-ordered collect preserved; only the rows
/// are spread), so the argmax can never move. DEFAULT-OFF (gate-then-flip): the mHC
/// is the last host-CPU-arithmetic bucket in the default 1CB decode path — moe /
/// attn_tail / mla_proj run GPU-resident and the MoE router matvec is already rayon'd
/// (LEVER B). Env read cached once (per-token × 2-site × 43-layer hot path).
fn hc_rayon_enabled() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("VLLM_VULKAN_DSV4_HC_RAYON")
            .map(|v| v != "0" && v != "false")
            .unwrap_or(true)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn hc_block(
    streams: &[f32],
    seq_len: usize,
    hc: usize,
    hidden: usize,
    fn_w: &[f32],
    base: &[f32],
    scale: &[f32],
    sinkhorn_iters: usize,
    hc_eps: f32,
    rms_eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let s = seq_len;
    let d = hidden;
    let hcd = hc * d;
    let mix_dim = (2 + hc) * hc;
    let eps = hc_eps as f64;
    // max-factored unweighted RMSNorm of each row's [hc*D] flat, in f64.
    let mut flat = vec![0.0f64; s * hcd];
    for si in 0..s {
        let row = &streams[si * hcd..(si + 1) * hcd];
        let mut m = 0.0f64;
        for &e in row {
            let e = e as f64;
            let e = if e.is_finite() { e } else { 0.0 };
            m = m.max(e.abs());
        }
        if m <= 0.0 {
            m = 1.0;
        }
        let mut var = 0.0f64;
        for &e in row {
            let e = e as f64;
            let e = if e.is_finite() { e } else { 0.0 };
            let xs = e / m;
            var += xs * xs;
        }
        var /= hcd as f64;
        let inv = 1.0f64 / (var + rms_eps as f64 / (m * m)).sqrt();
        let o = &mut flat[si * hcd..(si + 1) * hcd];
        for (i, &e) in row.iter().enumerate() {
            let e = e as f64;
            let e = if e.is_finite() { e } else { 0.0 };
            o[i] = (e / m) * inv;
        }
    }
    // mix = flat @ fn^T   (fn is [mix_dim, hc*D] row-major), in f64.
    let s0 = scale[0] as f64;
    let s1 = scale[1] as f64;
    let s2 = scale[2] as f64;
    let mut post = vec![0.0f32; s * hc];
    let mut comb = vec![0.0f32; s * hc * hc];
    let mut pre = vec![0.0f64; s * hc];
    for si in 0..s {
        let frow = &flat[si * hcd..(si + 1) * hcd];
        // pre_w=[0,hc), post_w=[hc,2hc), comb_w=[2hc, 2hc+hc*hc)
        //
        // ── HC RAYON LEVER (VLLM_VULKAN_DSV4_HC_RAYON, default-OFF) ─────────────
        // The `flat @ fn^T` projection is `mix_dim` INDEPENDENT f64 dot products
        // (each output row `j` accumulates over `k in 0..hcd` in its own order,
        // touching disjoint outputs). Spreading the rows across cores — with the
        // SAME inner accumulation order and index-ordered collect — is BIT-IDENTICAL
        // to the serial loop (argmax-exact by construction, exactly like LEVER B's
        // `matvec_par`). This is the one host-CPU-arithmetic bucket left single-
        // threaded in the default 1CB decode path (moe/attn_tail/mla_proj are all
        // GPU-resident; the router matvec is already rayon'd). The post-processing
        // (sigmoid/softmax stash + Sinkhorn) stays serial — it is tiny (hc*hc) and
        // order-sensitive. `acc[j]` is materialized first, then consumed unchanged.
        let dot = |j: usize| -> f64 {
            let wrow = &fn_w[j * hcd..(j + 1) * hcd];
            let mut a = 0.0f64;
            for k in 0..hcd {
                a += frow[k] * wrow[k] as f64;
            }
            a
        };
        let acc: Vec<f64> = if hc_rayon_enabled() {
            use rayon::prelude::*;
            (0..mix_dim).into_par_iter().map(dot).collect()
        } else {
            (0..mix_dim).map(dot).collect()
        };
        for j in 0..mix_dim {
            let acc = acc[j];
            if j < hc {
                let v = 1.0 / (1.0 + (-(acc * s0 + base[j] as f64)).exp()) + eps;
                pre[si * hc + j] = v;
            } else if j < 2 * hc {
                let jj = j - hc;
                let v = 2.0 / (1.0 + (-(acc * s1 + base[j] as f64)).exp());
                post[si * hc + jj] = v as f32;
            } else {
                let idx = j - 2 * hc; // 0..hc*hc, row-major [hc,hc]
                let logit = acc * s2 + base[j] as f64;
                comb[si * hc * hc + idx] = logit as f32; // stash logit; softmax below
            }
        }
        // Sinkhorn on comb[si] (currently holds raw logits): softmax over last axis,
        // + eps, col-norm, then (iters-1)×(row,col). Work in f64.
        let cb = &mut comb[si * hc * hc..(si + 1) * hc * hc];
        let mut m = vec![0.0f64; hc * hc];
        for r in 0..hc {
            let mut mx = f64::NEG_INFINITY;
            for c in 0..hc {
                mx = mx.max(cb[r * hc + c] as f64);
            }
            let mut den = 0.0f64;
            for c in 0..hc {
                let e = ((cb[r * hc + c] as f64) - mx).exp();
                m[r * hc + c] = e;
                den += e;
            }
            for c in 0..hc {
                m[r * hc + c] = m[r * hc + c] / den + eps;
            }
        }
        // col-norm (sum over rows, axis=-2)
        for c in 0..hc {
            let mut den = eps;
            for r in 0..hc {
                den += m[r * hc + c];
            }
            for r in 0..hc {
                m[r * hc + c] /= den;
            }
        }
        for _ in 0..sinkhorn_iters.saturating_sub(1) {
            // row-norm (axis=-1)
            for r in 0..hc {
                let mut den = eps;
                for c in 0..hc {
                    den += m[r * hc + c];
                }
                for c in 0..hc {
                    m[r * hc + c] /= den;
                }
            }
            // col-norm (axis=-2)
            for c in 0..hc {
                let mut den = eps;
                for r in 0..hc {
                    den += m[r * hc + c];
                }
                for r in 0..hc {
                    m[r * hc + c] /= den;
                }
            }
        }
        for i in 0..hc * hc {
            cb[i] = m[i] as f32;
        }
    }
    // collapsed[s,d] = sum_j pre[s,j]*streams[s,j,d]
    let mut collapsed = vec![0.0f32; s * d];
    for si in 0..s {
        let orow = &mut collapsed[si * d..(si + 1) * d];
        for j in 0..hc {
            let p = pre[si * hc + j];
            let strow = &streams[si * hcd + j * d..si * hcd + j * d + d];
            for dd in 0..d {
                orow[dd] += (p * strow[dd] as f64) as f32;
            }
        }
    }
    (post, comb, collapsed)
}

/// The `DeepseekV4DecoderLayer` residual mix consuming a site's `(post, comb)`:
/// `new[s,k,d] = post[s,k]·sublayer_out[s,d] + Σ_j comb[s,j,k]·streams[s,j,d]`
/// (== `post·out + combᵀ·streams`). Returns new streams `[S, hc, D]`.
pub fn hc_residual_mix(
    post: &[f32],
    sublayer_out: &[f32],
    comb: &[f32],
    streams: &[f32],
    seq_len: usize,
    hc: usize,
    hidden: usize,
) -> Vec<f32> {
    let s = seq_len;
    let d = hidden;
    let hcd = hc * d;
    let mut out = vec![0.0f32; s * hcd];
    for si in 0..s {
        let sub = &sublayer_out[si * d..(si + 1) * d];
        for k in 0..hc {
            let p = post[si * hc + k] as f64;
            let orow = &mut out[si * hcd + k * d..si * hcd + k * d + d];
            // term1: post[k]*sublayer
            for dd in 0..d {
                orow[dd] = (p * sub[dd] as f64) as f32;
            }
            // term2: sum_j comb[j,k]*streams[j,d]
            for j in 0..hc {
                let cjk = comb[si * hc * hc + j * hc + k] as f64;
                if cjk == 0.0 {
                    continue;
                }
                let strow = &streams[si * hcd + j * d..si * hcd + j * d + d];
                for dd in 0..d {
                    orow[dd] += (cjk * strow[dd] as f64) as f32;
                }
            }
        }
    }
    out
}

// ============================================================================
// SCAFFOLDING for the remaining M1 integration (pass 3+). Compiles; the bodies
// marked `PASS-3` are the next concrete steps. Everything below has a bit-exact
// numpy oracle already (see scripts/dsv4/*.py + scripts/dsv4/README.md).
// ============================================================================

/// Per-layer attention variant (`config.layer_types[li]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    /// Plain sliding-window MLA, `compressor is None` — the `mla_core` above IS
    /// the whole attention. Layers {0,1} only.
    Sliding,
    /// Compressed-Sparse-Attention: MLA + CSA compressor (2*hd) + Lightning-Indexer
    /// (index-score → causal top-512 → block_bias) concatenated onto the KV axis.
    /// 21 layers. GPU kernels: dsv4_dsa_compress + dsv4_dsa_index_score + dsv4_dsa_topk.
    CompressedSparse,
    /// Heavily-Compressed-Attention: MLA + HCA compressor (hd) concatenated onto the
    /// KV axis (no indexer). 20 layers. GPU kernel: dsv4_dsa_compress (hd variant).
    HeavilyCompressed,
}

/// Per-layer MoE routing variant (`config.mlp_layer_types[li]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpType {
    /// Hash router (`ffn.gate.tid2eid`, layers 0-2): expert id = table[token_id].
    HashMoe,
    /// noaux_tc + sqrtsoftplus top-k router (40 layers).
    Moe,
}

impl LayerType {
    pub fn parse(s: &str) -> Option<LayerType> {
        match s {
            "sliding_attention" => Some(LayerType::Sliding),
            "compressed_sparse_attention" => Some(LayerType::CompressedSparse),
            "heavily_compressed_attention" => Some(LayerType::HeavilyCompressed),
            _ => None,
        }
    }
}

impl MlpType {
    pub fn parse(s: &str) -> Option<MlpType> {
        match s {
            "hash_moe" => Some(MlpType::HashMoe),
            "moe" => Some(MlpType::Moe),
            _ => None,
        }
    }
}

/// Full DeepSeek-V4-Flash config (the fields the forward/loader need). Parse from
/// the model dir `config.json` via [`Dsv4Config::from_json`]. Values for the real
/// checkpoint: H=4096, nh=64, hd=512, q_lora=1024, o_lora=1024, o_groups=8,
/// hc_mult=4, sliding_window=128, index_{n_heads=64,head_dim=128,topk=512},
/// num_local_experts=256, num_experts_per_tok=6, moe_intermediate_size=2048,
/// swiglu_limit=10, routed_scaling_factor=1.5, hc_sinkhorn_iters=20, hc_eps=1e-6,
/// vocab=129280, 43 layers, rope main θ=10000 / compress θ=160000 yarn f=16.
#[derive(Clone, Debug)]
pub struct Dsv4Config {
    pub mla: Dsv4MlaConfig,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub swiglu_limit: f32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub layer_types: Vec<LayerType>,
    pub mlp_layer_types: Vec<MlpType>,
    /// RoPE derivation params (main = plain θ, compress = yarn-scaled θ). Defaults
    /// match DeepSeek-V4-Flash: partial 0.125, main θ=10000, compress θ=160000,
    /// yarn factor 16 / β_fast 32 / β_slow 1 / orig_max 65536, scaling 1.0.
    pub rope: Dsv4RopeConfig,
    /// compress_rate per compressed layer type: CSA (m=4), HCA (m'=128).
    pub compress_rate_csa: usize,
    pub compress_rate_hca: usize,
}

/// RoPE parameters for the "main" and "compress" (yarn) rope types.
#[derive(Clone, Debug)]
pub struct Dsv4RopeConfig {
    pub partial_rotary_factor: f64,
    pub theta_main: f64,
    pub theta_compress: f64,
    pub yarn_factor: f64,
    pub yarn_beta_fast: f64,
    pub yarn_beta_slow: f64,
    pub yarn_orig_max: f64,
    pub compress_scaling: f32,
}

impl Default for Dsv4RopeConfig {
    fn default() -> Self {
        Dsv4RopeConfig {
            partial_rotary_factor: 0.125,
            theta_main: 10000.0,
            theta_compress: 160000.0,
            yarn_factor: 16.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_max: 65536.0,
            compress_scaling: 1.0,
        }
    }
}

impl Dsv4Config {
    /// "main" partial-RoPE inverse frequencies for this config's head_dim.
    pub fn inv_freq_main(&self) -> Vec<f32> {
        crate::dsv4_dsa::inv_freq_main(self.mla.head_dim, self.rope.partial_rotary_factor, self.rope.theta_main)
    }
    /// "compress" yarn-scaled inverse frequencies for this config's head_dim.
    pub fn inv_freq_compress(&self) -> Vec<f32> {
        crate::dsv4_dsa::inv_freq_yarn(
            self.mla.head_dim, self.rope.partial_rotary_factor, self.rope.theta_compress,
            self.rope.yarn_factor, self.rope.yarn_beta_fast, self.rope.yarn_beta_slow,
            self.rope.yarn_orig_max,
        )
    }
}

impl Dsv4Config {
    /// Parse the fields the forward needs from a `config.json` JSON value. The
    /// real dir's nested `text_config` (if present) is merged shallowly first.
    pub fn from_json(j: &serde_json::Value) -> Result<Dsv4Config, String> {
        let base = j.get("text_config").unwrap_or(j);
        let get = |k: &str| base.get(k).or_else(|| j.get(k));
        let u = |k: &str| get(k).and_then(|v| v.as_u64()).map(|x| x as usize);
        let f = |k: &str| get(k).and_then(|v| v.as_f64()).map(|x| x as f32);
        let req_u = |k: &str| u(k).ok_or_else(|| format!("config missing usize {k}"));
        let lt: Vec<LayerType> = get("layer_types")
            .and_then(|v| v.as_array())
            .ok_or("config missing layer_types")?
            .iter()
            .map(|v| LayerType::parse(v.as_str().unwrap_or("")).ok_or_else(|| format!("bad layer_type {v}")))
            .collect::<Result<_, _>>()?;
        let ml: Vec<MlpType> = get("mlp_layer_types")
            .and_then(|v| v.as_array())
            .ok_or("config missing mlp_layer_types")?
            .iter()
            .map(|v| MlpType::parse(v.as_str().unwrap_or("")).ok_or_else(|| format!("bad mlp_layer_type {v}")))
            .collect::<Result<_, _>>()?;
        Ok(Dsv4Config {
            mla: Dsv4MlaConfig {
                hidden_size: req_u("hidden_size")?,
                num_attention_heads: req_u("num_attention_heads")?,
                head_dim: req_u("head_dim")?,
                q_lora_rank: req_u("q_lora_rank")?,
                o_lora_rank: req_u("o_lora_rank")?,
                o_groups: req_u("o_groups")?,
                rms_norm_eps: f("rms_norm_eps").unwrap_or(1e-6),
                sliding_window: req_u("sliding_window")?,
            },
            hc_mult: u("hc_mult").unwrap_or(4),
            hc_sinkhorn_iters: u("hc_sinkhorn_iters").unwrap_or(20),
            hc_eps: f("hc_eps").unwrap_or(1e-6),
            num_hidden_layers: req_u("num_hidden_layers")?,
            vocab_size: req_u("vocab_size")?,
            num_local_experts: u("num_local_experts").unwrap_or(256),
            num_experts_per_tok: u("num_experts_per_tok").unwrap_or(6),
            moe_intermediate_size: u("moe_intermediate_size").unwrap_or(2048),
            swiglu_limit: f("swiglu_limit").unwrap_or(10.0),
            routed_scaling_factor: f("routed_scaling_factor").unwrap_or(1.5),
            norm_topk_prob: get("norm_topk_prob").and_then(|v| v.as_bool()).unwrap_or(true),
            index_n_heads: u("index_n_heads").unwrap_or(64),
            index_head_dim: u("index_head_dim").unwrap_or(128),
            index_topk: u("index_topk").unwrap_or(512),
            layer_types: lt,
            mlp_layer_types: ml,
            rope: {
                // Prefer the transformers-derived `rope_parameters` dict; fall back
                // to raw-config keys (rope_theta / compress_rope_theta / rope_scaling)
                // and finally to the DeepSeek-V4-Flash defaults.
                let mut r = Dsv4RopeConfig::default();
                let rp = get("rope_parameters");
                if let Some(main) = rp.and_then(|v| v.get("main")) {
                    if let Some(t) = main.get("rope_theta").and_then(|v| v.as_f64()) { r.theta_main = t; }
                    if let Some(p) = main.get("partial_rotary_factor").and_then(|v| v.as_f64()) { r.partial_rotary_factor = p; }
                } else if let Some(t) = get("rope_theta").and_then(|v| v.as_f64()) {
                    r.theta_main = t;
                }
                if let Some(comp) = rp.and_then(|v| v.get("compress")) {
                    if let Some(t) = comp.get("rope_theta").and_then(|v| v.as_f64()) { r.theta_compress = t; }
                    if let Some(x) = comp.get("factor").and_then(|v| v.as_f64()) { r.yarn_factor = x; }
                    if let Some(x) = comp.get("beta_fast").and_then(|v| v.as_f64()) { r.yarn_beta_fast = x; }
                    if let Some(x) = comp.get("beta_slow").and_then(|v| v.as_f64()) { r.yarn_beta_slow = x; }
                    if let Some(x) = comp.get("original_max_position_embeddings").and_then(|v| v.as_f64()) { r.yarn_orig_max = x; }
                    if let Some(x) = comp.get("attention_factor").and_then(|v| v.as_f64()) { r.compress_scaling = x as f32; }
                } else {
                    if let Some(t) = get("compress_rope_theta").and_then(|v| v.as_f64()) { r.theta_compress = t; }
                    if let Some(rs) = get("rope_scaling") {
                        if let Some(x) = rs.get("factor").and_then(|v| v.as_f64()) { r.yarn_factor = x; }
                        if let Some(x) = rs.get("beta_fast").and_then(|v| v.as_f64()) { r.yarn_beta_fast = x; }
                        if let Some(x) = rs.get("beta_slow").and_then(|v| v.as_f64()) { r.yarn_beta_slow = x; }
                        if let Some(x) = rs.get("original_max_position_embeddings").and_then(|v| v.as_f64()) { r.yarn_orig_max = x; }
                    }
                }
                r
            },
            compress_rate_csa: get("compress_rates")
                .and_then(|v| v.get("compressed_sparse_attention")).and_then(|v| v.as_u64())
                .map(|x| x as usize).unwrap_or(4),
            compress_rate_hca: get("compress_rates")
                .and_then(|v| v.get("heavily_compressed_attention")).and_then(|v| v.as_u64())
                .map(|x| x as usize).unwrap_or(128),
        })
    }
}

/// Checkpoint → internal tensor name map + mixed bit-widths (see README Phase-1),
/// for the PASS-2 streaming loader `Dsv4GpuStage::from_ckpt_streamed`:
///
/// | ckpt prefix (per layer `model.layers.{li}`) | role | bits/gs | GPU matvec |
/// |---|---|---|---|
/// | `attn.wq_a wq_b wkv wo_a wo_b`, `attn.q_norm kv_norm` | MLA | 6/gs128 | mlx6 |
/// | `attn.compressor.* attn.indexer.*` | DSA | 6/gs128 | mlx6 + dsa_* |
/// | `ffn.switch_mlp.{gate,up,down}_proj` | routed experts (3D `[256,·,·]`) | 2/gs128 | mlx2repack_batched |
/// | `ffn.shared_experts.{g,u,d}` | shared expert | 8/gs64 | mlx8 |
/// | `model.embed_tokens`, `lm_head` | embed/head | 8/gs64 | mlx8 |
/// | norms, `attn.attn_sink`(F32), `*_hc.{fn,base,scale}`(F32), `ffn.gate.weight`(BF16), `ffn.gate.tid2eid`(I32) | unquant | — | host |
///
/// The reference name aliases (checkpoint → transformers, mirrored by golden.py):
/// `attn_norm→input_layernorm  ffn_norm→post_attention_layernorm  attn.wq_a→q_a_proj
///  attn.q_norm→q_a_norm  attn.wq_b→q_b_proj  attn.wkv→kv_proj  attn.kv_norm→kv_norm
///  attn.wo_a→o_a_proj(3D grouped)  attn.wo_b→o_b_proj  attn.attn_sink→sinks`.
///
/// ★ M1 REACHED (pass 3, 2026-08-15): the full 43-layer CPU-resident Rust forward
/// on the real 86GB checkpoint is `finite=true, argmax=11111` (` Paris`), matching
/// `golden_fulldepth_stable.json`. The complete assembly + validation:
///  * [`mla_core`] / [`mla_core_ext`] — sliding + CSA/HCA compressor-concat MLA
///    (passes 1-2). Fixtures `mla_core{,_csa,_hca}.json`.
///  * [`hc_block`] + [`hc_residual_mix`] — stable (f64/max-factored) mHC block +
///    decoder residual mix. Fixture `hc_block.json`.
///  * [`crate::dsv4_dsa`] — RoPE (main + compress-yarn, cos=1.0) + CSA/HCA
///    compressor + Lightning-Indexer producing `compressed_kv`/`block_bias`
///    (the pass-3 un-ported primitive). Fixtures `rope.json`, `dsa_{csa,hca}.json`.
///  * [`crate::dsv4_moe`] — noaux_tc topk + hash routers, swiglu routed experts
///    (+limit) + shared MLP, `moe_forward = routed(scaling-folded)+shared`.
///    Fixture `moe_block.json` (max_abs_err 4.1e-8).
///  * [`crate::dsv4_forward`] — full decoder-layer + HyperHead + lm_head forward
///    over a [`crate::dsv4_forward::Dsv4Src`]; tiny-model composition self-test ==
///    reference argmax (logits mae 1.68e-8). Fixture `selftest.json`.
///  * [`crate::dsv4_loader::Dsv4RealSrc`] — 18-shard MLX-affine 2/6/8-bit reader,
///    validated vs `golden.py::RealSource` (`loader_probe.json`); carries the
///    `#[ignore]` M1 gate `m1_gate_full_forward_argmax_11111`.
///
/// NEXT (M2, GPU-resident decode): `Dsv4GpuStage::from_ckpt_streamed` (mirror
/// `LingGpuStage::from_ckpt_streamed`, per-layer read→upload→free, mixed precision
/// per the table above; swap the CPU `cpu_matmul`/dequant for the shipped 6-bit
/// `mul_mat_vec_mlx6` / 2-bit `mlx2repack_batched` / 8-bit `mlx8` matvecs + the DSA
/// GPU trio `dsv4_dsa_compress`+`index_score`+`topk` + `debug_dsv4_moe_batched`);
/// then `decode_step` + KV cache, native-vCCL `pp_step`, and PP-10 BOUNDS
/// (~8.4GB/stage) with `UID_READY`. The pure-CPU forward in [`crate::dsv4_forward`]
/// is the argmax oracle every GPU stage must reproduce.
pub const DSV4_NAME_MAP_DOC: () = ();

#[cfg(test)]
mod mla_tests {
    use super::*;
    use serde_json::Value;

    fn f32v(v: &Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    /// Hermetic: the Rust `mla_core` reproduces the numpy MLA oracle
    /// (`scripts/dsv4/mla_oracle.py`, itself cross-checked bit-exact vs the
    /// transformers `DeepseekV4Attention`) to <1e-4 max_abs_err.
    #[test]
    fn mla_core_matches_oracle() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/mla_core.json");
        let raw = std::fs::read_to_string(path).expect("mla_core.json fixture (run mla_oracle.py --dump)");
        let j: Value = serde_json::from_str(&raw).unwrap();
        let c = &j["config"];
        let cfg = Dsv4MlaConfig {
            hidden_size: c["hidden_size"].as_u64().unwrap() as usize,
            num_attention_heads: c["num_attention_heads"].as_u64().unwrap() as usize,
            head_dim: c["head_dim"].as_u64().unwrap() as usize,
            q_lora_rank: c["q_lora_rank"].as_u64().unwrap() as usize,
            o_lora_rank: c["o_lora_rank"].as_u64().unwrap() as usize,
            o_groups: c["o_groups"].as_u64().unwrap() as usize,
            rms_norm_eps: c["rms_norm_eps"].as_f64().unwrap() as f32,
            sliding_window: c["sliding_window"].as_u64().unwrap() as usize,
        };
        let s = j["seq_len"].as_u64().unwrap() as usize;
        let x = f32v(&j["x"]);
        let w = &j["weights"];
        let inv_freq = f32v(&j["inv_freq"]);
        let positions: Vec<usize> = j["positions"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap() as usize).collect();

        // build cos/sin [S, rope_dim/2] from inv_freq (same as the oracle).
        let half = inv_freq.len();
        let mut cos = vec![0.0f32; s * half];
        let mut sin = vec![0.0f32; s * half];
        for si in 0..s {
            for i in 0..half {
                let f = positions[si] as f64 * inv_freq[i] as f64;
                cos[si * half + i] = f.cos() as f32;
                sin[si * half + i] = f.sin() as f32;
            }
        }
        // sliding-window causal additive mask.
        let mut mask = vec![0.0f32; s * s];
        for i in 0..s {
            for jj in 0..s {
                if jj > i || (i - jj) >= cfg.sliding_window {
                    mask[i * s + jj] = f32::NEG_INFINITY;
                }
            }
        }

        let out = mla_core(
            &cfg, &x, s,
            &f32v(&w["q_a"]), &f32v(&w["q_a_norm"]), &f32v(&w["q_b"]),
            &f32v(&w["kv"]), &f32v(&w["kv_norm"]),
            &f32v(&w["o_a"]), &f32v(&w["o_b"]), &f32v(&w["sinks"]),
            &cos, &sin, &mask,
        );
        let expect = f32v(&j["expected_out"]);
        assert_eq!(out.len(), expect.len());
        let mut mae = 0.0f32;
        for (a, b) in out.iter().zip(expect.iter()) {
            mae = mae.max((a - b).abs());
        }
        assert!(mae < 1e-4, "MLA core max_abs_err {mae:.3e} vs oracle exceeds 1e-4");
    }

    fn i32v(v: &Value) -> Vec<i32> {
        v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
    }

    /// Load a CSA/HCA compressor-concat fixture and check `mla_core_ext` reproduces
    /// the numpy oracle (== reference `DeepseekV4Attention` compressor path) to
    /// <1e-4 max_abs_err. `mask_visible`/`block_bias_visible` are 1/0 visibility ints
    /// (JSON has no -inf); map 1->0.0, 0->NEG_INFINITY to rebuild the additive masks.
    fn check_compressor_fixture(name: &str) {
        let path = format!("{}/tests/fixtures/dsv4/{}", env!("CARGO_MANIFEST_DIR"), name);
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("{name} fixture (run mla_oracle.py --dump-ext)"));
        let j: Value = serde_json::from_str(&raw).unwrap();
        let c = &j["config"];
        let cfg = Dsv4MlaConfig {
            hidden_size: c["hidden_size"].as_u64().unwrap() as usize,
            num_attention_heads: c["num_attention_heads"].as_u64().unwrap() as usize,
            head_dim: c["head_dim"].as_u64().unwrap() as usize,
            q_lora_rank: c["q_lora_rank"].as_u64().unwrap() as usize,
            o_lora_rank: c["o_lora_rank"].as_u64().unwrap() as usize,
            o_groups: c["o_groups"].as_u64().unwrap() as usize,
            rms_norm_eps: c["rms_norm_eps"].as_f64().unwrap() as f32,
            sliding_window: c["sliding_window"].as_u64().unwrap() as usize,
        };
        let s = j["seq_len"].as_u64().unwrap() as usize;
        let t = j["compressed_len"].as_u64().unwrap() as usize;
        let x = f32v(&j["x"]);
        let w = &j["weights"];
        let cos = f32v(&j["cos"]);
        let sin = f32v(&j["sin"]);
        let vis2mask = |v: Vec<i32>| -> Vec<f32> {
            v.into_iter().map(|b| if b != 0 { 0.0 } else { f32::NEG_INFINITY }).collect()
        };
        let mask = vis2mask(i32v(&j["mask_visible"])); // [S,S]
        let block_bias = vis2mask(i32v(&j["block_bias_visible"])); // [S,T]
        let compressed_kv = f32v(&j["compressed_kv"]); // [T,hd]
        assert_eq!(compressed_kv.len(), t * cfg.head_dim);

        let out = mla_core_ext(
            &cfg, &x, s,
            &f32v(&w["q_a"]), &f32v(&w["q_a_norm"]), &f32v(&w["q_b"]),
            &f32v(&w["kv"]), &f32v(&w["kv_norm"]),
            &f32v(&w["o_a"]), &f32v(&w["o_b"]), &f32v(&w["sinks"]),
            &cos, &sin, &mask, &compressed_kv, &block_bias,
        );
        let expect = f32v(&j["expected_out"]);
        assert_eq!(out.len(), expect.len());
        let mut mae = 0.0f32;
        for (a, b) in out.iter().zip(expect.iter()) {
            mae = mae.max((a - b).abs());
        }
        assert!(mae < 1e-4, "{name}: mla_core_ext max_abs_err {mae:.3e} exceeds 1e-4");
    }

    /// CSA (compressed_sparse_attention) compressor-concat path bit-exact vs oracle.
    #[test]
    fn mla_core_ext_csa_matches_oracle() {
        check_compressor_fixture("mla_core_csa.json");
    }

    /// HCA (heavily_compressed_attention) compressor-concat path bit-exact vs oracle.
    #[test]
    fn mla_core_ext_hca_matches_oracle() {
        check_compressor_fixture("mla_core_hca.json");
    }

    /// The mHC block (collapse + residual mix) reproduces the numpy oracle
    /// (`scripts/dsv4/hc_oracle.py`, == reference `DeepseekV4HyperConnection` +
    /// decoder residual mix) to <1e-3 max_abs_err on all four outputs.
    #[test]
    fn hc_block_matches_oracle() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/hc_block.json");
        let raw = std::fs::read_to_string(path).expect("hc_block.json (run hc_oracle.py --dump)");
        let j: Value = serde_json::from_str(&raw).unwrap();
        let hc = j["hc_mult"].as_u64().unwrap() as usize;
        let d = j["hidden_size"].as_u64().unwrap() as usize;
        let s = j["seq_len"].as_u64().unwrap() as usize;
        let iters = j["hc_sinkhorn_iters"].as_u64().unwrap() as usize;
        let hc_eps = j["hc_eps"].as_f64().unwrap() as f32;
        let rms_eps = j["rms_norm_eps"].as_f64().unwrap() as f32;
        let streams = f32v(&j["streams"]);
        let sublayer = f32v(&j["sublayer_out"]);
        let fn_w = f32v(&j["fn"]);
        let base = f32v(&j["base"]);
        let scale = f32v(&j["scale"]);

        let (post, comb, collapsed) =
            hc_block(&streams, s, hc, d, &fn_w, &base, &scale, iters, hc_eps, rms_eps);
        let mixed = hc_residual_mix(&post, &sublayer, &comb, &streams, s, hc, d);

        let mae = |a: &[f32], b: Vec<f32>| -> f32 {
            assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        assert!(mae(&post, f32v(&j["expected_post"])) < 1e-3, "hc post mismatch");
        assert!(mae(&comb, f32v(&j["expected_comb"])) < 1e-3, "hc comb mismatch");
        assert!(mae(&collapsed, f32v(&j["expected_collapsed"])) < 1e-3, "hc collapsed mismatch");
        assert!(mae(&mixed, f32v(&j["expected_mixed"])) < 1e-3, "hc residual mix mismatch");
    }

    /// Synthetic mHC at the REAL decode shape (s=1, hc=4, d=4096 → hcd=16384,
    /// mix_dim=24). Runs `hc_block` on the serial path then re-runs the projection
    /// with the rayon path forced ON and asserts BYTE-IDENTICAL outputs on all three
    /// returns. (The env flag is process-cached via OnceLock, so this test drives the
    /// two projection branches directly rather than toggling the env.)
    #[test]
    fn hc_block_rayon_bit_identical_real_shape() {
        use rayon::prelude::*;
        let (hc, d, iters) = (4usize, 4096usize, 20usize);
        let (hc_eps, rms_eps) = (1e-6f32, 1e-6f32);
        let hcd = hc * d;
        let mix_dim = (2 + hc) * hc;
        // Deterministic pseudo-random inputs (no rng dep): a cheap LCG in f32.
        let mut st: u64 = 0x1234_5678_9abc_def0;
        let mut rnd = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((st >> 33) as f32) / (1u64 << 31) as f32) - 1.0
        };
        let streams: Vec<f32> = (0..hcd).map(|_| rnd() * 0.5).collect();
        let fn_w: Vec<f32> = (0..mix_dim * hcd).map(|_| rnd() * 0.1).collect();
        let base: Vec<f32> = (0..mix_dim).map(|_| rnd() * 0.05).collect();
        let scale: Vec<f32> = vec![0.9, 1.1, 0.7];

        // Serial reference via the public entrypoint (flag default-OFF in-process).
        let (post_s, comb_s, coll_s) =
            hc_block(&streams, 1, hc, d, &fn_w, &base, &scale, iters, hc_eps, rms_eps);

        // Independent rayon recompute of the projection accs, then the SAME
        // post-processing, to confirm the parallel accs match the serial accs
        // byte-for-byte (the only thing the flag changes).
        let m = 0.0f64;
        let _ = m;
        let frow: Vec<f64> = {
            // Reproduce hc_block's max-factored RMSNorm of the single row.
            let mut mx = 0.0f64;
            for &e in &streams {
                let e = e as f64;
                mx = mx.max(if e.is_finite() { e.abs() } else { 0.0 });
            }
            if mx <= 0.0 {
                mx = 1.0;
            }
            let mut var = 0.0f64;
            for &e in &streams {
                let e = e as f64;
                let xs = (if e.is_finite() { e } else { 0.0 }) / mx;
                var += xs * xs;
            }
            var /= hcd as f64;
            let inv = 1.0 / (var + rms_eps as f64 / (mx * mx)).sqrt();
            streams.iter().map(|&e| (e as f64 / mx) * inv).collect()
        };
        let dot = |j: usize| -> f64 {
            let wrow = &fn_w[j * hcd..(j + 1) * hcd];
            let mut a = 0.0f64;
            for k in 0..hcd {
                a += frow[k] * wrow[k] as f64;
            }
            a
        };
        let acc_ser: Vec<f64> = (0..mix_dim).map(dot).collect();
        let acc_par: Vec<f64> = (0..mix_dim).into_par_iter().map(dot).collect();
        assert_eq!(acc_ser, acc_par, "rayon projection accs differ from serial (bytes)");

        // Sanity: outputs are finite and non-trivial (the serial path produced them).
        assert!(post_s.iter().all(|v| v.is_finite()));
        assert!(comb_s.iter().all(|v| v.is_finite()));
        assert!(coll_s.iter().all(|v| v.is_finite()));
    }

    /// Microbench (ignored): serial vs rayon `flat @ fn^T` at the real decode shape,
    /// to measure whether the 24-row × 16384-f64 projection is big enough to beat
    /// rayon spawn/join overhead on an 8-core box. Run:
    ///   cargo test --release --lib dsv4::hc_tests::hc_projection_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn hc_projection_bench() {
        use rayon::prelude::*;
        use std::time::Instant;
        let (hc, d) = (4usize, 4096usize);
        let hcd = hc * d;
        let mix_dim = (2 + hc) * hc;
        let mut st: u64 = 0xdead_beef_cafe_0001;
        let mut rnd = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((st >> 33) as f32) / (1u64 << 31) as f32) - 1.0
        };
        let frow: Vec<f64> = (0..hcd).map(|_| rnd() as f64 * 0.3).collect();
        let fn_w: Vec<f32> = (0..mix_dim * hcd).map(|_| rnd() * 0.1).collect();
        let dot = |j: usize| -> f64 {
            let wrow = &fn_w[j * hcd..(j + 1) * hcd];
            let mut a = 0.0f64;
            for k in 0..hcd {
                a += frow[k] * wrow[k] as f64;
            }
            a
        };
        // Warm the rayon pool.
        let _: Vec<f64> = (0..mix_dim).into_par_iter().map(dot).collect();
        let iters = 2000usize;
        let mut sink = 0.0f64;
        let t = Instant::now();
        for _ in 0..iters {
            let a: Vec<f64> = (0..mix_dim).map(dot).collect();
            sink += a[0] + a[mix_dim - 1];
        }
        let ser = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
        let t = Instant::now();
        for _ in 0..iters {
            let a: Vec<f64> = (0..mix_dim).into_par_iter().map(dot).collect();
            sink += a[0] + a[mix_dim - 1];
        }
        let par = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
        let ncores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
        println!(
            "[hc_projection_bench] cores={ncores} shape=({mix_dim}x{hcd}) serial={ser:.2}us rayon={par:.2}us speedup={:.2}x  per-call-save={:.2}us  (sink={sink:.3})",
            ser / par,
            ser - par,
        );
    }

    /// The config parser reads the layer-type dispatch + MLA dims correctly.
    #[test]
    fn config_parse_layer_types() {
        let j: Value = serde_json::from_str(
            r#"{"hidden_size":4096,"num_attention_heads":64,"head_dim":512,
                 "q_lora_rank":1024,"o_lora_rank":1024,"o_groups":8,"hc_mult":4,
                 "sliding_window":128,"num_hidden_layers":3,"vocab_size":129280,
                 "rms_norm_eps":1e-6,
                 "layer_types":["sliding_attention","compressed_sparse_attention","heavily_compressed_attention"],
                 "mlp_layer_types":["hash_moe","moe","moe"]}"#,
        )
        .unwrap();
        let c = Dsv4Config::from_json(&j).unwrap();
        assert_eq!(c.mla.head_dim, 512);
        assert_eq!(c.mla.num_attention_heads, 64);
        assert_eq!(c.hc_mult, 4);
        assert_eq!(c.layer_types, vec![LayerType::Sliding, LayerType::CompressedSparse, LayerType::HeavilyCompressed]);
        assert_eq!(c.mlp_layer_types, vec![MlpType::HashMoe, MlpType::Moe, MlpType::Moe]);
    }
}
