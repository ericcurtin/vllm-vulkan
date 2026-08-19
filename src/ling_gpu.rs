// SPDX-License-Identifier: Apache-2.0
//! Ling-3.0-flash (BailingMoeV3) GPU QUANT-RESIDENT per-PP-stage decode — the perf
//! port of the CPU-resident `ling::LingModel` decode (which is the bit-exact golden).
//!
//! Modeled on `src/kimi_gpu.rs::KimiGpuStage`. A `LingGpuStage` owns a
//! `ComputeEngine` and holds its PP window `[layer_start, layer_end)` resident:
//!   - **KDA** (35/42 layers, the bulk): 7 f32 projections (q/k/v/f/g/b/o) held
//!     GPU-resident + host conv/L2-norm/decay glue (VERBATIM from the bit-exact
//!     `ling::LingKdaWeights::decode_step`) + the REUSED `kda_gdn_step.comp`
//!     recurrence (Ling feeds a pre-computed per-key-channel `decay` and raw
//!     beta/gate, so the Kimi kernel is bit-compatible) advancing a resident state
//!     buffer + f32 o_proj.
//!   - **MoE** (40 layers): host grouped-topk route + resident INT4 mlx4 concatenated
//!     switch experts (int4-symmetric packed words verbatim) + f32 shared expert +
//!     `q35_moe_accum`. Overflow-streamed experts (MOE_STREAM_OVERFLOW) are uploaded
//!     transient per token, so the fit-to-validate harness still composes.
//!   - **Dense MLP** (layers 0..first_k_dense_replace): f32 SwiGLU on the GPU.
//!   - **lm_head**: f32 matvec on the GPU (last stage).
//!   - **MLA** (7/42 layers): the HOST bit-exact seam (`ling::LingMlaWeights::decode_step`)
//!     — mirrors Kimi's default `MlaImpl::Host`. GPU-MLA projections are a documented
//!     follow-on lever (interleaved-RoPE + head-gate glue is small; the int4 KDA/MoE
//!     bandwidth is the decode floor).
//!
//! Only routed experts are int4 in this checkpoint; every other tensor (attention,
//! MLA, dense, shared, embed, lm_head, router) is BF16 → held/matvec'd as f32.
//!
//! Bit-exact oracle = the CPU resident decode (`ling::LingModel::decode_step` /
//! `forward_pp_stage`): resident-held GPU == CPU, argmax-exact (GPU-float tolerance).

use crate::compute;
use crate::device;
use crate::model;
use crate::push_constants::*;
use crate::ling::{self, LingAttn, LingConfig, LingExpertQ, LingKdaState, LingLayerKind, LingMlp, LingModel};

use ash::vk;
use std::collections::HashMap;

/// int4-symmetric group size for Ling routed experts (compressed-tensors group-32).
const EXPERT_GROUP: usize = 32;

/// A dense (bf16-origin) matvec weight `[n, k]` held GPU-resident. Stored as f32
/// by default, or as f16 when the `VLLM_VULKAN_LING_F16_ATTN` lever is on (`f16=true`)
/// — halving the resident bytes + the per-token read. The weights are BF16 on disk
/// (7 mantissa bits) so f16 storage (10 mantissa bits) is effectively lossless for
/// the O(1) weight magnitudes; the dispatch picks the f16 vs f32 matvec shader by
/// this flag (both share the `matvec_pc13` layout + 3-binding `[W,x,out]` form).
struct GpuMatF32 {
    buf: compute::Buffer,
    k: usize,
    n: usize,
    f16: bool,
}

/// A resident packed int4-sym mlx4 switch: the RESIDENT experts of one MoE
/// sub-projection concatenated (packed words verbatim + `-8·scale` biases).
/// `slot[e]` maps expert `e` to its concatenated slot, or `-1` when that expert
/// was overflow-streamed (then `strm[e]` carries `(shard_path, base)`).
struct SwitchR {
    p: compute::Buffer,
    s: compute::Buffer,
    b: compute::Buffer,
    out: usize,
    inn: usize,
    slot: Vec<i32>,
    strm: Vec<Option<(String, String)>>,
    resident_slots: usize,
    /// GPU-resident copy of `slot` (i32[e]) — built only under the Phase-1
    /// `moe_indirect` lever so the GPU meta-builder can map expert idx -> slot.
    slot_buf: Option<compute::Buffer>,
}

impl SwitchR {
    /// True when every expert is resident (no MOE_STREAM_OVERFLOW), the
    /// precondition for the fully-GPU-driven fused route->meta->matvec path
    /// (the host can't detect a streamed expert once routing is on-device).
    fn fully_resident(&self) -> bool {
        self.strm.iter().all(|s| s.is_none())
    }
}

struct KdaGpuR {
    q: GpuMatF32,
    k: GpuMatF32,
    v: GpuMatF32,
    f: GpuMatF32,
    g: GpuMatF32,
    b: GpuMatF32,
    o: GpuMatF32,
    // host glue (verbatim from decode_step)
    q_conv: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<f32>,
    nh: usize,
    hd: usize,
    kern: usize,
    eps: f32,
    safe_gate: bool,
    lower_bound: f32,
    /// Persistent recurrence state [nv,kd,vd], advanced in place by `kda_gdn_step`.
    g_state: compute::Buffer,
    /// Host conv sliding-windows (advanced host-side each token — the OFF /
    /// 2-submit `kda_step_resident` path only).
    conv: LingKdaState,
    // --- Phase-3 fused-KDA (`VLLM_VULKAN_LING_KDA_FUSED`) resident glue buffers.
    // Held GPU-resident so the depthwise conv / L2-norm qknorm / safe_gate decay
    // record into the KDA command buffer instead of round-tripping to host, so
    // the projections->conv->qknorm->decay->gdn_step->o_proj chain is ONE submit.
    // Built regardless of the flag (tiny: ~0.35 MB/KDA-layer); `reset_state`
    // zeros the conv windows so the flag can be honoured per session.
    /// Depthwise conv taps [proj, kern] (== the host `*_conv` Vecs, uploaded once).
    q_conv_buf: compute::Buffer,
    k_conv_buf: compute::Buffer,
    v_conv_buf: compute::Buffer,
    /// Persistent GPU conv sliding-windows [proj, kern-1] (channel-major; the GPU
    /// analog of `conv.conv_*`, advanced in place by `q35_dn_conv_step`).
    conv_state_q: compute::Buffer,
    conv_state_k: compute::Buffer,
    conv_state_v: compute::Buffer,
    /// Per-head A_log [nh] and per-channel dt_bias [proj] for `ling_kda_decay`
    /// (== the host `a_log`/`dt_bias` Vecs, uploaded once).
    a_log_buf: compute::Buffer,
    dt_bias_buf: compute::Buffer,
    /// `kda_gdn_step` params [2*nh | o_norm(proj)] (the leading 2*nh is unused —
    /// beta/gate/decay are fed via separate bindings). Built once.
    g_params: compute::Buffer,
}

struct MoeGpuR {
    gate: SwitchR,
    up: SwitchR,
    down: SwitchR,
    router_gate: Vec<f32>, // [e, h]
    expert_bias: Vec<f32>, // [e]
    // Phase-1 GPU-router (VLLM_VULKAN_LING_MOE_INDIRECT): router_gate/expert_bias
    // held GPU-resident so the grouped-topk selection runs on-device (only the
    // top-k idx+weights is read back). None when the lever is OFF (no upload).
    router_gate_buf: Option<compute::Buffer>,
    expert_bias_buf: Option<compute::Buffer>,
    // shared expert (ungated, f32)
    sh_gate: GpuMatF32,
    sh_up: GpuMatF32,
    sh_down: GpuMatF32,
    sh_inter: usize,
    inter: usize,
    e: usize,
    top_k: usize,
    scale: f32,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
}

struct DenseGpuR {
    gate: GpuMatF32,
    up: GpuMatF32,
    down: GpuMatF32,
    h: usize,
    inter: usize,
}

/// Resident MLA attention (`VLLM_VULKAN_LING_MLA_RESIDENT`): the 6 MLA projections
/// held GPU-resident as `GpuMatF32` (f32, or f16 under `LING_F16_ATTN`) + the host
/// interleaved-RoPE / kv_a_layernorm / causal-SDPA / head-gate seam. `q_proj`
/// `[nh*(nope+pe), h]`, `kv_a_proj` (kv_a_proj_with_mqa) `[r+pe, h]`, `g_proj`
/// `[nh, h]`, `dense` `[h, nh*v]` map 1:1 to a matvec. The compressed-KV decompress
/// splits into TWO matvecs against the layer-normed latent `c_kv [r]`: `embed_q`
/// (transposed at upload to matvec-ready `[nh*nope, r]`) → `k_nope [nh*nope]`, and
/// `unembed_out` (already `[nh*v, r]` on disk) → `v [nh*v]`. `kv_a_layernorm` stays
/// host f32. The MLA KV cache advances token-to-token host-side (materialized-MHA).
struct MlaGpuR {
    q_proj: GpuMatF32,          // [nh*(nope+pe), h]
    kv_a_proj: GpuMatF32,       // [r+pe, h]
    embed_q: GpuMatF32,         // [nh*nope, r]  (transposed at upload)
    unembed_out: GpuMatF32,     // [nh*v, r]
    g_proj: Option<GpuMatF32>,  // [nh, h]  (None when !head_gate)
    dense: GpuMatF32,           // [h, nh*v]
    kv_a_layernorm: Vec<f32>,
    h: usize,
    nh: usize,
    nope: usize,
    pe: usize,
    v: usize,
    r: usize,
    eps: f32,
    rope_theta: f32,
    head_gate: bool,
    /// Materialized-MHA KV cache, advanced in place across decode steps.
    cache: ling::LingMlaCache,
}

/// MLA backend for a resident layer: GPU-resident projections (lever ON) or the
/// legacy all-host bit-exact seam (`ling::LingMlaWeights::decode_step`, lever OFF).
enum LMlaR {
    Host(ling::LingMlaWeights, ling::LingMlaCache),
    Gpu(MlaGpuR),
}

enum LAttnR {
    Kda(KdaGpuR),
    Mla(LMlaR),
}
enum LMlpR {
    Dense(DenseGpuR),
    Moe(MoeGpuR),
}

struct LLayerR {
    input_ln: Vec<f32>,
    post_ln: Vec<f32>,
    attn: LAttnR,
    mlp: LMlpR,
    /// GPU-resident copies of the two RMSNorm weights, so the resident-layer
    /// path (`VLLM_VULKAN_LING_RESIDENT_LAYER`) can run `input_layernorm` /
    /// `post_attention_layernorm` on the GPU (`rms_norm_f32_mul`) over the
    /// resident hidden buffer instead of the host `ling::rmsnorm`. Uploaded once
    /// (tiny: h·4 B each); unused when the lever is OFF.
    input_ln_buf: compute::Buffer,
    post_ln_buf: compute::Buffer,
}

/// One PP window of Ling held GPU-resident, decoded a token at a time.
pub struct LingGpuStage {
    layers: Vec<LLayerR>,
    embed: Option<Vec<f32>>,
    final_norm: Option<Vec<f32>>,
    /// GPU-resident copy of `final_norm` (tail stage only) for the resident-layer
    /// path's GPU model-norm. `None` on non-tail stages / when unused.
    final_norm_buf: Option<compute::Buffer>,
    lm_head: Option<GpuMatF32>,
    eng: compute::ComputeEngine,
    _dev: device::ComputeDevice,
    cfg: LingConfig,
    pub layer_start: usize,
    pub layer_end: usize,
    pub first: bool,
    pub last: bool,
    h: usize,
    eps: f32,
    /// Expert-batched MoE decode lever (`VLLM_VULKAN_LING_MOE_BATCH`, default OFF):
    /// collapse the 8-experts × {gate,up,down} = 24 per-expert matvec dispatches
    /// into 3 batched dispatches through `mul_mat_vec_mlx4repack_batched_f32_f32`.
    moe_batch: bool,
    /// Phase-3 fused-KDA decode lever (`VLLM_VULKAN_LING_KDA_FUSED`, default OFF):
    /// collapse each KDA layer's 2-submit host-seam path (6 projections → host
    /// conv/L2norm/decay glue → gdn_step+o_proj) into ONE submit by moving the
    /// conv (`q35_dn_conv_step`) + L2-norm qknorm (`ling_kda_l2norm`) + safe_gate
    /// decay (`ling_kda_decay`) onto the GPU over resident conv-ring/tap/param
    /// buffers — killing ~1 fence + the host round-trip on 35/42 layers.
    kda_fused: bool,
    /// Phase-1 GPU-router decode lever (`VLLM_VULKAN_LING_MOE_INDIRECT`, default
    /// OFF): run the grouped-topk router (`ling_moe_router`) on the GPU instead
    /// of the host `cpu_matmul` + `grouped_topk_route`; only the top-k idx+weights
    /// is read back. Composes with `moe_batch` (the batched gather-matvec).
    moe_indirect: bool,
    /// Resident single-CB-layer decode lever (`VLLM_VULKAN_LING_RESIDENT_LAYER`,
    /// default OFF): thread the hidden state through a GPU-resident buffer across
    /// the whole layer (and across layers) — `input_layernorm` + the attn-residual
    /// + `post_attention_layernorm` + the mlp-residual all run on the GPU and the
    /// whole KDA/Dense+MoE layer (input_ln → attn → residual → post_ln → mlp →
    /// residual) records into ONE command buffer, so the host stops orchestrating
    /// each layer op-by-op (no per-op `hn`-upload / `[h]`-readback). MLA layers
    /// (7/42) keep the host-seam op-by-op path (interleaved-RoPE / SDPA / head-gate
    /// cannot be recorded). Composes with all the stacked levers.
    resident_layer: bool,
}

/// Per-layer upload context: the resident-layout levers + dims needed to mirror
/// ONE host `LingLayer` into GPU-resident buffers. Shared by `from_cpu` (consume a
/// fully-loaded CPU window) and `from_ckpt_streamed` (per-layer read→upload→free),
/// so both produce byte-identical GPU buffers from the same host `LingLayer`.
struct UpCtx {
    f16_dense: bool,
    mla_resident: bool,
    moe_indirect: bool,
    h: usize,
    proj: usize,
}

/// The resident-layout decode levers, read from env once per stage build.
struct LingFlags {
    // f16-dense lever (`VLLM_VULKAN_LING_F16_ATTN`, default-ON): hold every
    // bf16-origin DENSE weight (KDA attn q/k/v/f/g/b/o, dense-MLP, shared-expert,
    // lm_head) as f16 instead of f32 — effectively lossless (bf16→f16 keeps all 7
    // disk mantissa bits), halves the per-token dense read. int4 routed experts
    // untouched (separate lever).
    f16_dense: bool,
    // MLA-resident lever (`VLLM_VULKAN_LING_MLA_RESIDENT`, default-ON): the 7/42
    // MLA layers run on GPU-resident projections instead of the host bit-exact seam.
    mla_resident: bool,
    // Phase-1 GPU-router lever (`VLLM_VULKAN_LING_MOE_INDIRECT`, default-ON): run
    // the grouped-topk router on the GPU; uploads router_gate/expert_bias resident.
    moe_indirect: bool,
    moe_batch: bool,
    kda_fused: bool,
    resident_layer: bool,
}

fn read_ling_flags() -> LingFlags {
    LingFlags {
        f16_dense: std::env::var("VLLM_VULKAN_LING_F16_ATTN").ok().as_deref() != Some("0"),
        mla_resident: std::env::var("VLLM_VULKAN_LING_MLA_RESIDENT").ok().as_deref() != Some("0"),
        moe_indirect: std::env::var("VLLM_VULKAN_LING_MOE_INDIRECT").ok().as_deref() != Some("0"),
        moe_batch: std::env::var("VLLM_VULKAN_LING_MOE_BATCH").ok().as_deref() != Some("0"),
        kda_fused: std::env::var("VLLM_VULKAN_LING_KDA_FUSED").ok().as_deref() != Some("0"),
        resident_layer: std::env::var("VLLM_VULKAN_LING_RESIDENT_LAYER").ok().as_deref() == Some("1"),
    }
}

/// Create the compute engine + device for a stage (identical setup for both
/// constructors, so the resident kernels behave the same regardless of loader).
fn make_engine(device_idx: usize) -> Result<(compute::ComputeEngine, device::ComputeDevice), String> {
    let dev = device::ComputeDevice::create(device_idx)?;
    let shader_spvs = crate::include_all_shaders();
    let refs: HashMap<&str, &[u8]> =
        shader_spvs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
    let eng = compute::ComputeEngine::new(
        dev.instance.clone(), dev.physical_device, dev.device.clone(),
        dev.compute_queue, dev.compute_queue_family, dev.caps(), &refs,
    )?;
    Ok((eng, dev))
}

/// Upload a dense `[n,k]` matvec weight (f16 under `f16_dense`, else f32).
fn up_f32(
    eng: &mut compute::ComputeEngine, w: &[f32], k: usize, n: usize, f16_dense: bool,
) -> Result<GpuMatF32, String> {
    debug_assert_eq!(w.len(), k * n, "dense weight [{n},{k}] len {}", w.len());
    let bb = if f16_dense { f32_to_f16_bytes(w) } else { f32_slice_to_bytes(w) };
    let buf = eng.alloc_host_coherent_storage(bb.len().max(4) as u64)?;
    buf.write(&bb)?;
    Ok(GpuMatF32 { buf, k, n, f16: f16_dense })
}

/// Upload a raw f32 storage buffer (conv taps / windows / decay params — always
/// f32, read as `float[]` by the fused-KDA glue shaders regardless of f16_dense).
fn up_raw(eng: &mut compute::ComputeEngine, w: &[f32]) -> Result<compute::Buffer, String> {
    let bb = f32_slice_to_bytes(w);
    let buf = eng.alloc_host_coherent_storage(bb.len().max(4) as u64)?;
    buf.write(&bb)?;
    Ok(buf)
}

/// Build a resident int4-sym switch from a Vec<LingExpertQ> (one MoE sub-proj).
fn build_switch(
    eng: &mut compute::ComputeEngine, experts: Vec<LingExpertQ>, moe_indirect: bool,
) -> Result<SwitchR, String> {
    let want_slot_buf = moe_indirect;
    let e = experts.len();
    let (out, inn) = (experts[0].out, experts[0].inn);
    let groups = inn / EXPERT_GROUP;
    let words_per_row = inn / 8;
    let mut packed: Vec<u32> = Vec::new();
    let mut scales: Vec<f32> = Vec::new();
    let mut biases: Vec<f32> = Vec::new();
    let mut slot = vec![-1i32; e];
    let mut strm: Vec<Option<(String, String)>> = vec![None; e];
    let mut resident_slots = 0usize;
    for (ei, ex) in experts.into_iter().enumerate() {
        debug_assert_eq!(ex.out, out);
        debug_assert_eq!(ex.inn, inn);
        match ex.store {
            ling::ExpertStore::Resident { packed: p, scales: s, biases: b } => {
                debug_assert_eq!(p.len(), out * words_per_row);
                debug_assert_eq!(s.len(), out * groups);
                slot[ei] = resident_slots as i32;
                resident_slots += 1;
                packed.extend_from_slice(&p);
                scales.extend_from_slice(&s);
                biases.extend_from_slice(&b);
            }
            ling::ExpertStore::Streamed { shard_path, base } => {
                strm[ei] = Some((shard_path, base));
            }
        }
    }
    // Allocate at least one slot's worth so the buffer is never zero-sized.
    if packed.is_empty() {
        packed = vec![0u32; out * words_per_row];
        scales = vec![0f32; out * groups];
        biases = vec![0f32; out * groups];
    }
    let pb = bytemuck::cast_slice::<u32, u8>(&packed).to_vec();
    let p = eng.alloc_host_coherent_storage(pb.len().max(4) as u64)?;
    p.write(&pb)?;
    let sb = f32_slice_to_bytes(&scales);
    let s = eng.alloc_host_coherent_storage(sb.len().max(4) as u64)?;
    s.write(&sb)?;
    let bb = f32_slice_to_bytes(&biases);
    let b = eng.alloc_host_coherent_storage(bb.len().max(4) as u64)?;
    b.write(&bb)?;
    let slot_buf = if want_slot_buf {
        let sbytes = bytemuck::cast_slice::<i32, u8>(&slot).to_vec();
        let buf = eng.alloc_host_coherent_storage(sbytes.len().max(4) as u64)?;
        buf.write(&sbytes)?;
        Some(buf)
    } else {
        None
    };
    Ok(SwitchR { p, s, b, out, inn, slot, strm, resident_slots, slot_buf })
}

/// Mirror ONE host `LingLayer` into GPU-resident buffers, consuming (and thus
/// freeing) the host weights. This is the per-layer upload core: both `from_cpu`
/// and `from_ckpt_streamed` call it, so the resident GPU layout is byte-identical
/// regardless of whether the whole window was pre-loaded on host or streamed.
fn upload_ling_layer(
    eng: &mut compute::ComputeEngine, ly: ling::LingLayer, ctx: &UpCtx,
) -> Result<LLayerR, String> {
    let (h, proj, f16_dense, mla_resident, moe_indirect) =
        (ctx.h, ctx.proj, ctx.f16_dense, ctx.mla_resident, ctx.moe_indirect);
    let attn = match ly.attn {
        LingAttn::Kda(w) => {
            let (nh, hd, kern) = (w.nh, w.hd, w.kern);
            let win = kern.saturating_sub(1);
            // fused-KDA resident glue buffers (proj == nh*hd == key_dim == value_dim)
            let q_conv_buf = up_raw(eng, &w.q_conv)?;
            let k_conv_buf = up_raw(eng, &w.k_conv)?;
            let v_conv_buf = up_raw(eng, &w.v_conv)?;
            let conv_state_q = up_raw(eng, &vec![0f32; proj * win])?;
            let conv_state_k = up_raw(eng, &vec![0f32; proj * win])?;
            let conv_state_v = up_raw(eng, &vec![0f32; proj * win])?;
            let a_log_buf = up_raw(eng, &w.a_log)?;
            let dt_bias_buf = up_raw(eng, &w.dt_bias)?;
            let g_params = {
                let mut params = vec![0f32; 2 * nh];
                params.extend_from_slice(&w.o_norm);
                up_raw(eng, &params)?
            };
            let kda = KdaGpuR {
                q: up_f32(eng, &w.q_proj, h, proj, f16_dense)?,
                k: up_f32(eng, &w.k_proj, h, proj, f16_dense)?,
                v: up_f32(eng, &w.v_proj, h, proj, f16_dense)?,
                f: up_f32(eng, &w.f_proj, h, proj, f16_dense)?,
                g: up_f32(eng, &w.g_proj, h, proj, f16_dense)?,
                b: up_f32(eng, &w.b_proj, h, nh, f16_dense)?,
                o: up_f32(eng, &w.o_proj, proj, h, f16_dense)?,
                q_conv: w.q_conv,
                k_conv: w.k_conv,
                v_conv: w.v_conv,
                a_log: w.a_log,
                dt_bias: w.dt_bias,
                o_norm: w.o_norm,
                nh, hd, kern,
                eps: w.eps,
                safe_gate: w.safe_gate,
                lower_bound: w.lower_bound,
                g_state: {
                    let zb = f32_slice_to_bytes(&vec![0f32; nh * hd * hd]);
                    let buf = eng.alloc_host_coherent_storage(zb.len() as u64)?;
                    buf.write(&zb)?;
                    buf
                },
                conv: LingKdaState::new(nh, hd, kern),
                q_conv_buf, k_conv_buf, v_conv_buf,
                conv_state_q, conv_state_k, conv_state_v,
                a_log_buf, dt_bias_buf, g_params,
            };
            LAttnR::Kda(kda)
        }
        LingAttn::Mla(w) => {
            if mla_resident {
                let (nh, nope, pe, vv, r) = (w.nh, w.nope, w.pe, w.v, w.r);
                // Transpose embed_q [nh, r, nope] (row `(hh*r+rr)*nope+n`) →
                // matvec-ready [nh*nope, r] (row `(hh*nope+n)*r+rr`), so ONE
                // matvec against c_kv[r] yields k_nope[nh*nope]. unembed_out is
                // already [nh*v, r] on disk (row `(hh*v+d)*r+rr`) = matvec-ready.
                let mut eq_t = vec![0f32; nh * nope * r];
                for hh in 0..nh {
                    for rr in 0..r {
                        for n in 0..nope {
                            eq_t[(hh * nope + n) * r + rr] = w.embed_q[(hh * r + rr) * nope + n];
                        }
                    }
                }
                let g_proj = if w.head_gate {
                    Some(up_f32(eng, &w.g_proj, w.h, nh, f16_dense)?)
                } else {
                    None
                };
                let mla = MlaGpuR {
                    q_proj: up_f32(eng, &w.q_proj, w.h, nh * (nope + pe), f16_dense)?,
                    kv_a_proj: up_f32(eng, &w.kv_a_proj, w.h, r + pe, f16_dense)?,
                    embed_q: up_f32(eng, &eq_t, r, nh * nope, f16_dense)?,
                    unembed_out: up_f32(eng, &w.unembed_out, r, nh * vv, f16_dense)?,
                    g_proj,
                    dense: up_f32(eng, &w.dense, nh * vv, w.h, f16_dense)?,
                    kv_a_layernorm: w.kv_a_layernorm,
                    h: w.h, nh, nope, pe, v: vv, r,
                    eps: w.eps, rope_theta: w.rope_theta, head_gate: w.head_gate,
                    cache: ling::LingMlaCache::new(),
                };
                LAttnR::Mla(LMlaR::Gpu(mla))
            } else {
                LAttnR::Mla(LMlaR::Host(w, ling::LingMlaCache::new()))
            }
        }
    };

    let mlp = match ly.mlp {
        LingMlp::Dense(d) => LMlpR::Dense(DenseGpuR {
            gate: up_f32(eng, &d.gate, d.h, d.inter, f16_dense)?,
            up: up_f32(eng, &d.up, d.h, d.inter, f16_dense)?,
            down: up_f32(eng, &d.down, d.inter, d.h, f16_dense)?,
            h: d.h,
            inter: d.inter,
        }),
        LingMlp::Moe(m) => {
            let sh_inter = m.shared_inter;
            let (router_gate_buf, expert_bias_buf) = if moe_indirect {
                (Some(up_raw(eng, &m.gate)?), Some(up_raw(eng, &m.expert_bias)?))
            } else {
                (None, None)
            };
            LMlpR::Moe(MoeGpuR {
                gate: build_switch(eng, m.ew_gate, moe_indirect)?,
                up: build_switch(eng, m.ew_up, moe_indirect)?,
                down: build_switch(eng, m.ew_down, moe_indirect)?,
                router_gate: m.gate,
                expert_bias: m.expert_bias,
                router_gate_buf,
                expert_bias_buf,
                sh_gate: up_f32(eng, &m.sh_gate, h, sh_inter, f16_dense)?,
                sh_up: up_f32(eng, &m.sh_up, h, sh_inter, f16_dense)?,
                sh_down: up_f32(eng, &m.sh_down, sh_inter, h, f16_dense)?,
                sh_inter,
                inter: m.inter,
                e: m.e,
                top_k: m.top_k,
                scale: m.scale,
                n_group: m.n_group,
                topk_group: m.topk_group,
                norm_topk_prob: m.norm_topk_prob,
            })
        }
    };
    // Resident RMSNorm weights (always uploaded, tiny — used only under
    // VLLM_VULKAN_LING_RESIDENT_LAYER; OFF path still reads the host Vecs).
    let input_ln_buf = up_raw(eng, &ly.input_ln)?;
    let post_ln_buf = up_raw(eng, &ly.post_ln)?;
    Ok(LLayerR {
        input_ln: ly.input_ln, post_ln: ly.post_ln, attn, mlp,
        input_ln_buf, post_ln_buf,
    })
}

impl LingGpuStage {
    /// Build the resident GPU stage by CONSUMING a CPU-loaded window (`load_cpu`,
    /// which already applied the MOE_STREAM_OVERFLOW budget). Uploads all resident
    /// weights ONCE; MLA weights and streamed-expert descriptors stay on host.
    pub fn from_cpu(model: LingModel, device_idx: usize) -> Result<LingGpuStage, String> {
        let cfg = model.cfg.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let (layer_start, layer_end) = (model.layer_start, model.layer_end);
        let first = layer_start == 0;
        let last = layer_end == cfg.num_hidden_layers;

        let (mut eng, dev) = make_engine(device_idx)?;
        let fl = read_ling_flags();
        let ctx = UpCtx {
            f16_dense: fl.f16_dense,
            mla_resident: fl.mla_resident,
            moe_indirect: fl.moe_indirect,
            h,
            proj: cfg.kda_num_heads * cfg.kda_head_dim,
        };

        // Consume the pre-loaded host window layer-by-layer (each `ly` is uploaded
        // then dropped by `into_iter`, so the host window drains as GTT fills). The
        // full host window was already materialized by `load_cpu` — for the memory-
        // lean load path use `from_ckpt_streamed`, which never holds the whole window.
        let mut layers: Vec<LLayerR> = Vec::with_capacity(layer_end - layer_start);
        for ly in model.layers.into_iter() {
            layers.push(upload_ling_layer(&mut eng, ly, &ctx)?);
        }

        // edges
        let embed = model.embed; // host row lookup (first stage)
        let final_norm = model.final_norm;
        let final_norm_buf = match &final_norm {
            Some(w) => Some(up_raw(&mut eng, w)?),
            None => None,
        };
        let lm_head = match model.lm_head {
            Some(w) => Some(up_f32(&mut eng, &w, h, w.len() / h, fl.f16_dense)?),
            None => None,
        };

        Ok(LingGpuStage {
            layers, embed, final_norm, final_norm_buf, lm_head, eng, _dev: dev,
            cfg, layer_start, layer_end, first, last, h, eps,
            moe_batch: fl.moe_batch, kda_fused: fl.kda_fused, moe_indirect: fl.moe_indirect,
            resident_layer: fl.resident_layer,
        })
    }

    /// Memory-lean loader: build the resident GPU stage by STREAMING the checkpoint
    /// window layer-by-layer — read one layer's host weights, upload them to GTT,
    /// free the host copy — via `ling::LingModel::load_window_streaming`. The full
    /// host PP-window is NEVER materialized at once, so on the UMA BC-250 nodes
    /// (GTT === system DRAM) peak DRAM ≈ (resident GTT footprint) + (one layer's
    /// working set), instead of the ~2.5-3x blow-up of `load_cpu` (whole host
    /// window) + `from_cpu` (GTT upload) coexisting that OOMed the PP-8 all-resident
    /// bring-up at LOAD. Produces byte-identical resident buffers to `from_cpu`
    /// (same `upload_ling_layer`, same host `LingLayer` read by the same loader),
    /// including the MOE_STREAM_OVERFLOW budget (the streaming core accumulates it
    /// across the single pass). All 6 decode levers behave identically.
    pub fn from_ckpt_streamed(
        ckpt_dir: &str,
        cfg: &LingConfig,
        layer_start: usize,
        layer_end: usize,
        load_edges: bool,
        device_idx: usize,
    ) -> Result<LingGpuStage, String> {
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let first = layer_start == 0;
        let last = layer_end == cfg.num_hidden_layers;

        let (mut eng, dev) = make_engine(device_idx)?;
        let fl = read_ling_flags();
        let ctx = UpCtx {
            f16_dense: fl.f16_dense,
            mla_resident: fl.mla_resident,
            moe_indirect: fl.moe_indirect,
            h,
            proj: cfg.kda_num_heads * cfg.kda_head_dim,
        };

        // Stream: each layer is read → uploaded → freed by the sink; the host window
        // never fully materializes. `layers` and `eng` are borrowed by the sink for
        // the duration of the call, then reused for the edge uploads below.
        let mut layers: Vec<LLayerR> = Vec::with_capacity(layer_end - layer_start);
        let edges = LingModel::load_window_streaming(
            ckpt_dir, cfg, layer_start, layer_end, load_edges,
            |ly| {
                layers.push(upload_ling_layer(&mut eng, ly, &ctx)?);
                Ok(())
            },
        )?;

        // edges (uploaded after the layer sink drains — same order/layout as from_cpu)
        let embed = edges.embed; // host row lookup (first stage)
        let final_norm = edges.final_norm;
        let final_norm_buf = match &final_norm {
            Some(w) => Some(up_raw(&mut eng, w)?),
            None => None,
        };
        let lm_head = match edges.lm_head {
            Some(w) => Some(up_f32(&mut eng, &w, h, w.len() / h, fl.f16_dense)?),
            None => None,
        };

        Ok(LingGpuStage {
            layers, embed, final_norm, final_norm_buf, lm_head, eng, _dev: dev,
            cfg: cfg.clone(), layer_start, layer_end, first, last, h, eps,
            moe_batch: fl.moe_batch, kda_fused: fl.kda_fused, moe_indirect: fl.moe_indirect,
            resident_layer: fl.resident_layer,
        })
    }

    /// Reset every layer's persistent recurrence/KV state (fresh decode session).
    pub fn reset_state(&mut self) -> Result<(), String> {
        for layer in self.layers.iter_mut() {
            match &mut layer.attn {
                LAttnR::Kda(kda) => {
                    let zb = f32_slice_to_bytes(&vec![0f32; kda.nh * kda.hd * kda.hd]);
                    kda.g_state.write(&zb)?;
                    kda.conv = LingKdaState::new(kda.nh, kda.hd, kda.kern);
                    // zero the GPU conv-ring windows (fused path state).
                    let win = kda.kern.saturating_sub(1);
                    let zc = f32_slice_to_bytes(&vec![0f32; kda.nh * kda.hd * win]);
                    kda.conv_state_q.write(&zc)?;
                    kda.conv_state_k.write(&zc)?;
                    kda.conv_state_v.write(&zc)?;
                }
                LAttnR::Mla(m) => match m {
                    LMlaR::Host(_, c) => *c = ling::LingMlaCache::new(),
                    LMlaR::Gpu(g) => g.cache = ling::LingMlaCache::new(),
                },
            }
        }
        Ok(())
    }

    /// One PP-stage single-token decode step (the GPU-resident `forward_pp_stage`).
    /// First stage embeds `token_id`; else consumes `hidden_in[H]`. Last stage
    /// returns `[vocab]` logits; else the `[H]` hidden to ship onward.
    pub fn forward_pp_stage(&mut self, token_id: u32, hidden_in: &[f32]) -> Result<Vec<f32>, String> {
        if self.resident_layer {
            return self.forward_pp_stage_resident(token_id, hidden_in);
        }
        let h = self.h;
        let eps = self.eps;
        let mut x = if self.first {
            let emb = self.embed.as_ref().ok_or("stage 0 requires embed")?;
            let row = token_id as usize * h;
            emb[row..row + h].to_vec()
        } else {
            if hidden_in.len() != h {
                return Err(format!("PP hidden_in {} != {h}", hidden_in.len()));
            }
            hidden_in.to_vec()
        };

        let eng = &mut self.eng;
        for layer in self.layers.iter_mut() {
            let xn = ling::rmsnorm(&x, 1, h, &layer.input_ln, eps);
            let attn = match &mut layer.attn {
                LAttnR::Kda(kda) => {
                    if self.kda_fused {
                        kda_step_resident_fused(eng, kda, &xn, eps)?
                    } else {
                        kda_step_resident(eng, kda, &xn)?
                    }
                }
                LAttnR::Mla(m) => match m {
                    LMlaR::Host(w, c) => w.decode_step(&xn, c),
                    LMlaR::Gpu(g) => mla_step_resident(eng, g, &xn)?,
                },
            };
            let mut hres = vec![0f32; h];
            for i in 0..h { hres[i] = x[i] + attn[i]; }
            let hn = ling::rmsnorm(&hres, 1, h, &layer.post_ln, eps);
            let mlp = match &layer.mlp {
                LMlpR::Dense(d) => dense_step_resident(eng, d, &hn)?,
                LMlpR::Moe(m) => {
                    // Fully-GPU-driven route->meta->matvec (one CB, no host index
                    // readback) when the lever is on AND every expert is resident
                    // (else the host can't detect a streamed expert). Otherwise the
                    // host-route batched/per-expert path (the standalone GPU router
                    // as its own submit is a known regression — not used).
                    if self.moe_indirect
                        && m.router_gate_buf.is_some()
                        && m.gate.fully_resident()
                        && m.up.fully_resident()
                        && m.down.fully_resident()
                    {
                        moe_combine_batched_fused(eng, m, &hn, h)?
                    } else if self.moe_batch {
                        moe_combine_batched(eng, m, &hn, h, false)?
                    } else {
                        moe_combine_resident(eng, m, &hn, h, false)?
                    }
                }
            };
            let mut out = vec![0f32; h];
            for i in 0..h { out[i] = hres[i] + mlp[i]; }
            x = out;
        }

        if self.last {
            let fnorm = self.final_norm.as_ref().ok_or("tail stage requires final_norm")?;
            let normed = ling::rmsnorm(&x, 1, h, fnorm, eps);
            let lm = self.lm_head.as_ref().ok_or("tail stage requires lm_head")?;
            f32_matvec_once(&mut self.eng, lm, &normed)
        } else {
            Ok(x)
        }
    }

    /// Resident single-CB-layer decode (`VLLM_VULKAN_LING_RESIDENT_LAYER`). The
    /// hidden state lives in a GPU-resident buffer threaded across all layers of
    /// this PP stage; `input_layernorm`, the attn residual, `post_attention_layernorm`
    /// and the mlp residual run on the GPU, and each KDA/Dense+MoE layer records
    /// (input_ln → attn → residual → post_ln → mlp → residual) into ONE command
    /// buffer — so the host stops orchestrating each layer op-by-op (no per-op
    /// `hn`-upload / `[h]`-readback; ONE submit/fence per such layer instead of the
    /// prior attn-submit+fence+readback → host-glue → mlp-submit+fence+readback).
    /// MLA layers (7/42) keep the host-seam op-by-op path — their interleaved-RoPE /
    /// causal-SDPA / head-gate glue cannot be recorded — reading the resident hidden
    /// to host and writing the result back. Bit-exact-equivalent to the OFF path:
    /// the GPU RMSNorm (`rms_norm_f32_mul`, the same argmax-exact kernel qwen3.6 uses)
    /// and the GPU residual add replace host `ling::rmsnorm` / host `+`; the attn/mlp
    /// dispatches are the identical stacked-lever kernels.
    fn forward_pp_stage_resident(&mut self, token_id: u32, hidden_in: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.h;
        let eps = self.eps;
        let (moe_indirect, moe_batch, kda_fused) = (self.moe_indirect, self.moe_batch, self.kda_fused);
        let first = self.first;

        // initial hidden -> resident GPU buffer
        let x0: Vec<f32> = if first {
            let emb = self.embed.as_ref().ok_or("stage 0 requires embed")?;
            let row = token_id as usize * h;
            emb[row..row + h].to_vec()
        } else {
            if hidden_in.len() != h {
                return Err(format!("PP hidden_in {} != {h}", hidden_in.len()));
            }
            hidden_in.to_vec()
        };

        let eng = &mut self.eng;
        let x_buf = alloc(eng, h)?;
        x_buf.write(&f32_slice_to_bytes(&x0))?;

        for layer in self.layers.iter_mut() {
            // Single-CB capable = KDA attn AND an MLP with no host seam (Dense, or a
            // fully-resident GPU-driven MoE). MLA layers + overflow-streamed / non-
            // indirect MoE fall back to the host-seam op-by-op path below.
            let kda_attn = matches!(layer.attn, LAttnR::Kda(_));
            let moe_single_cb = match &layer.mlp {
                LMlpR::Dense(_) => true,
                LMlpR::Moe(m) => {
                    moe_indirect
                        && m.router_gate_buf.is_some()
                        && m.gate.fully_resident()
                        && m.up.fully_resident()
                        && m.down.fully_resident()
                }
            };

            if kda_attn && moe_single_cb {
                // ---- ONE command buffer for the whole layer ----
                let mut g: Vec<compute::Buffer> = Vec::new();
                let xn = alloc(eng, h)?;
                let attn = alloc(eng, h)?;
                let hres = alloc(eng, h)?;
                let hn = alloc(eng, h)?;
                let mlp = alloc(eng, h)?;
                let cb = eng.begin_batch()?;
                // input_layernorm
                record_rmsnorm(eng, cb, &x_buf, &layer.input_ln_buf, &xn, h, eps)?;
                eng.record_barrier_to(cb);
                // attention (KDA, fused-resident)
                match &layer.attn {
                    LAttnR::Kda(kda) => kda_record(eng, cb, kda, &xn, &attn, eps, &mut g)?,
                    LAttnR::Mla(_) => unreachable!(),
                }
                eng.record_barrier_to(cb);
                // residual: hres = x + attn
                record_add(eng, cb, &x_buf, &attn, &hres, h)?;
                eng.record_barrier_to(cb);
                // post_attention_layernorm
                record_rmsnorm(eng, cb, &hres, &layer.post_ln_buf, &hn, h, eps)?;
                eng.record_barrier_to(cb);
                // MLP
                match &layer.mlp {
                    LMlpR::Dense(d) => dense_record(eng, cb, d, &hn, &mlp, &mut g)?,
                    LMlpR::Moe(m) => moe_record(eng, cb, m, &hn, &mlp, h, &mut g)?,
                }
                eng.record_barrier_to(cb);
                // residual: x = hres + mlp (in place — resident hidden for next layer)
                record_add(eng, cb, &hres, &mlp, &x_buf, h)?;
                eng.submit_batch(cb)?;
                for b in [xn, attn, hres, hn, mlp] { eng.return_to_pool(b); }
                for b in g { eng.return_to_pool(b); }
            } else {
                // ---- host-seam op-by-op path (MLA / non-resident MoE) ----
                let x = read_f32_buf(&x_buf, h);
                let xn = ling::rmsnorm(&x, 1, h, &layer.input_ln, eps);
                let attn = match &mut layer.attn {
                    LAttnR::Kda(kda) => {
                        if kda_fused { kda_step_resident_fused(eng, kda, &xn, eps)? }
                        else { kda_step_resident(eng, kda, &xn)? }
                    }
                    LAttnR::Mla(m) => match m {
                        LMlaR::Host(w, c) => w.decode_step(&xn, c),
                        LMlaR::Gpu(gp) => mla_step_resident(eng, gp, &xn)?,
                    },
                };
                let mut hres = vec![0f32; h];
                for i in 0..h { hres[i] = x[i] + attn[i]; }
                let hn = ling::rmsnorm(&hres, 1, h, &layer.post_ln, eps);
                let mlp = match &layer.mlp {
                    LMlpR::Dense(d) => dense_step_resident(eng, d, &hn)?,
                    LMlpR::Moe(m) => {
                        if moe_indirect && m.router_gate_buf.is_some()
                            && m.gate.fully_resident() && m.up.fully_resident() && m.down.fully_resident() {
                            moe_combine_batched_fused(eng, m, &hn, h)?
                        } else if moe_batch {
                            moe_combine_batched(eng, m, &hn, h, false)?
                        } else {
                            moe_combine_resident(eng, m, &hn, h, false)?
                        }
                    }
                };
                let mut out = vec![0f32; h];
                for i in 0..h { out[i] = hres[i] + mlp[i]; }
                x_buf.write(&f32_slice_to_bytes(&out))?;
            }
        }

        let x = read_f32_buf(&x_buf, h);
        self.eng.return_to_pool(x_buf);

        if self.last {
            let fnorm = self.final_norm.as_ref().ok_or("tail stage requires final_norm")?;
            let normed = ling::rmsnorm(&x, 1, h, fnorm, eps);
            let lm = self.lm_head.as_ref().ok_or("tail stage requires lm_head")?;
            f32_matvec_once(&mut self.eng, lm, &normed)
        } else {
            Ok(x)
        }
    }

    /// Router-selection gate (Phase-1): for the FIRST MoE layer in this window,
    /// run BOTH the host grouped-topk route and the GPU `ling_moe_router` on the
    /// SAME hidden `hn`, returning `(host_inds, gpu_inds, host_w, gpu_w)`. The GPU
    /// router must select the identical top-k SET (a routing mismatch = wrong
    /// experts). Requires the stage built with `VLLM_VULKAN_LING_MOE_INDIRECT=1`
    /// (else the resident router buffers are absent and the GPU path falls back to
    /// the host route, trivially matching but not exercising the shader).
    pub fn debug_router_compare(
        &mut self,
        hn: &[f32],
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<f32>, Vec<f32>), String> {
        let h = self.h;
        let eng = &mut self.eng;
        for layer in self.layers.iter_mut() {
            if let LMlpR::Moe(m) = &layer.mlp {
                if m.router_gate_buf.is_none() {
                    return Err("MoE layer has no resident router buffer (set \
                                VLLM_VULKAN_LING_MOE_INDIRECT=1)".into());
                }
                let (hi, hw) = moe_route(eng, m, hn, h, false)?;
                let (gi, gw) = gpu_router_topk(eng, m, hn, h)?;
                return Ok((hi, gi, hw, gw));
            }
        }
        Err("no MoE layer in this window".into())
    }
}

// ---------------- primitives ----------------

fn alloc(eng: &mut compute::ComputeEngine, n: usize) -> Result<compute::Buffer, String> {
    eng.alloc_host_coherent_storage((n.max(1) * 4) as u64)
}

/// Record a dense matvec `out[n] = W[n,k] · xin[k]` into an OPEN command buffer.
/// Picks the f16 vs f32 matvec shader by the weight's stored format (both share the
/// `matvec_pc13` layout + `[W,x,out]` bindings; f16 halves the weight read). The f16
/// shader is selected via `matvec_variant_by_format(F16,..)` so it is correct even
/// under a global `VLLM_VULKAN_QUANT` (which `matvec_variant` would otherwise honor).
fn record_matvec_f32(
    eng: &mut compute::ComputeEngine,
    cb: vk::CommandBuffer,
    m: &GpuMatF32,
    xin: &compute::Buffer,
    out: &compute::Buffer,
) -> Result<(), String> {
    let (shader, r) = if m.f16 {
        matvec_variant_by_format(crate::flags::QuantFormat::F16, m.n)
    } else {
        matvec_f32_variant(m.n)
    };
    let wg = (m.n as u32 + r - 1) / r;
    let pc = matvec_pc13(m.k, m.n);
    eng.record_to(cb, &shader, &[&m.buf, xin, out], &pc, (wg, 1, 1))
}

/// One f32 matvec, own submit+fence, returns `[n]`.
fn f32_matvec_once(eng: &mut compute::ComputeEngine, m: &GpuMatF32, x: &[f32]) -> Result<Vec<f32>, String> {
    if x.len() != m.k { return Err(format!("matvec x {} != k {}", x.len(), m.k)); }
    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let o = alloc(eng, m.n)?;
    let cb = eng.begin_batch()?;
    record_matvec_f32(eng, cb, m, &xbuf, &o)?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o, m.n);
    eng.return_to_pool(xbuf);
    eng.return_to_pool(o);
    Ok(out)
}

/// Record `out[h] = rmsnorm(x[h])·w[h]` into an OPEN command buffer (one row,
/// `rms_norm_f32_mul` — the identical argmax-exact kernel qwen3.6's resident stage
/// uses). Matches Ling's host `ling::rmsnorm` (standard RMSNorm; eps carried in the
/// pc) to GPU subgroup-reduce order (cos≈1.0, the accepted LingGpuStage tolerance).
fn record_rmsnorm(
    eng: &mut compute::ComputeEngine,
    cb: vk::CommandBuffer,
    x: &compute::Buffer,
    w: &compute::Buffer,
    out: &compute::Buffer,
    h: usize,
    eps: f32,
) -> Result<(), String> {
    let pc = rmsnorm_pc(h, eps);
    eng.record_to(cb, "rms_norm_f32_mul", &[x, w, out], &pc, (1, 1, 1))
}

/// Record `out[h] = a[h] + b[h]` into an OPEN command buffer (`add_f32_f32_f32`).
/// Binding 3 is the shader's partial-sum scratch (dead `ADD_RMS` path, never
/// written here) so `out` harmlessly doubles as the dummy (cf. qwen3.6 resident).
fn record_add(
    eng: &mut compute::ComputeEngine,
    cb: vk::CommandBuffer,
    a: &compute::Buffer,
    b: &compute::Buffer,
    out: &compute::Buffer,
    h: usize,
) -> Result<(), String> {
    let pc = ew_mul_pc(h as u32);
    let wg = (h as u32 + 255) / 256;
    eng.record_to(cb, "add_f32_f32_f32", &[a, b, out, out], &pc, (wg, 1, 1))
}

/// Record the fused-KDA decode step (the `VLLM_VULKAN_LING_KDA_FUSED` stage A-E
/// chain) into an OPEN command buffer: read the input-norm hidden from `x`, write
/// the o_proj output to `out`, advance the resident conv-ring + recurrence state in
/// place. The recording is IDENTICAL to `kda_step_resident_fused`; only the outer
/// upload/submit/readback wrapper is hoisted to the caller so the whole layer is one
/// CB. Scratch buffers are pushed to `g` for freeing after the caller's submit.
fn kda_record(
    eng: &mut compute::ComputeEngine,
    cb: vk::CommandBuffer,
    kda: &KdaGpuR,
    x: &compute::Buffer,
    out: &compute::Buffer,
    eps: f32,
    g: &mut Vec<compute::Buffer>,
) -> Result<(), String> {
    let (nh, hd, kern) = (kda.nh, kda.hd, kda.kern);
    let proj = nh * hd;
    let key_dim = proj;
    let value_dim = proj;
    let conv_dim = 2 * key_dim + value_dim;
    let v_off = 2 * key_dim;
    let scale = (hd as f32).powf(-0.5);

    let o_q = alloc(eng, kda.q.n)?;
    let o_k = alloc(eng, kda.k.n)?;
    let o_v = alloc(eng, kda.v.n)?;
    let o_f = alloc(eng, kda.f.n)?;
    let o_g = alloc(eng, kda.g.n)?;
    let o_b = alloc(eng, kda.b.n)?;
    let b_conv = alloc(eng, conv_dim)?;
    let b_decay = alloc(eng, proj)?;
    let b_q = alloc(eng, key_dim)?;
    let b_k = alloc(eng, key_dim)?;
    let b_gated = alloc(eng, value_dim)?;

    let conv_pc = q35_conv_pc(key_dim, kern);
    let qk_pc = q35_qknorm_pc(nh, hd, key_dim, kda.eps, scale);
    let decay_pc = ling_kda_decay_pc(nh, hd, kda.lower_bound);
    let gdn_pc = q35_gdn_pc(hd, hd, 1, v_off, eps, nh);
    let conv_wg = ((key_dim as u32) + 255) / 256;
    let decay_wg = ((proj as u32) + 255) / 256;

    // stage A: 6 direct projections (read x).
    record_matvec_f32(eng, cb, &kda.q, x, &o_q)?;
    record_matvec_f32(eng, cb, &kda.k, x, &o_k)?;
    record_matvec_f32(eng, cb, &kda.v, x, &o_v)?;
    record_matvec_f32(eng, cb, &kda.f, x, &o_f)?;
    record_matvec_f32(eng, cb, &kda.g, x, &o_g)?;
    record_matvec_f32(eng, cb, &kda.b, x, &o_b)?;
    eng.record_barrier_to(cb);
    // stage B: depthwise conv+silu q/k/v (advances resident windows) + safe_gate decay.
    eng.record_to_off(cb, "q35_dn_conv_step",
        &[(&kda.q_conv_buf, 0), (&o_q, 0), (&kda.conv_state_q, 0), (&b_conv, 0)],
        &conv_pc, (conv_wg, 1, 1))?;
    eng.record_to_off(cb, "q35_dn_conv_step",
        &[(&kda.k_conv_buf, 0), (&o_k, 0), (&kda.conv_state_k, 0), (&b_conv, (key_dim * 4) as u64)],
        &conv_pc, (conv_wg, 1, 1))?;
    eng.record_to_off(cb, "q35_dn_conv_step",
        &[(&kda.v_conv_buf, 0), (&o_v, 0), (&kda.conv_state_v, 0), (&b_conv, (v_off * 4) as u64)],
        &conv_pc, (conv_wg, 1, 1))?;
    eng.record_to(cb, "ling_kda_decay",
        &[&o_f, &kda.dt_bias_buf, &kda.a_log_buf, &b_decay],
        &decay_pc, (decay_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage C: L2-norm qknorm (q scaled).
    eng.record_to(cb, "ling_kda_l2norm",
        &[&b_conv, &b_q, &b_k],
        &qk_pc, (2 * nh as u32, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage D: delta-rule recurrence + gated o_norm (advances g_state in place).
    eng.record_to(cb, "kda_gdn_step",
        &[&b_q, &b_k, &b_conv, &b_decay, &o_b, &o_g, &kda.g_params, &kda.g_state, &b_gated],
        &gdn_pc, (nh as u32, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage E: o_proj -> out.
    record_matvec_f32(eng, cb, &kda.o, &b_gated, out)?;

    for b in [o_q, o_k, o_v, o_f, o_g, o_b, b_conv, b_decay, b_q, b_k, b_gated] {
        g.push(b);
    }
    Ok(())
}

/// Record a resident dense SwiGLU MLP (`silu(gate·x) ⊙ (up·x) → down`) into an OPEN
/// command buffer: read `hn`, write `out`. Recording identical to
/// `dense_step_resident` sans the outer submit/readback. Scratch pushed to `g`.
fn dense_record(
    eng: &mut compute::ComputeEngine,
    cb: vk::CommandBuffer,
    d: &DenseGpuR,
    hn: &compute::Buffer,
    out: &compute::Buffer,
    g: &mut Vec<compute::Buffer>,
) -> Result<(), String> {
    let inter = d.inter;
    let b_g = alloc(eng, inter)?;
    let b_u = alloc(eng, inter)?;
    let b_a = alloc(eng, inter)?;
    let b_m = alloc(eng, inter)?;
    let silu_pc = ew_unary_pc(inter as u32);
    let mul_pc = ew_mul_pc(inter as u32);
    let silu_wg = (inter as u32 + 511) / 512;
    let mul_wg = (inter as u32 + 255) / 256;
    record_matvec_f32(eng, cb, &d.gate, hn, &b_g)?;
    record_matvec_f32(eng, cb, &d.up, hn, &b_u)?;
    eng.record_barrier_to(cb);
    eng.record_to(cb, "silu_f32", &[&b_g, &b_a], &silu_pc, (silu_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_a, &b_u, &b_m], &mul_pc, (mul_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    record_matvec_f32(eng, cb, &d.down, &b_m, out)?;
    for b in [b_g, b_u, b_a, b_m] { g.push(b); }
    Ok(())
}

/// Record the fully-GPU-driven MoE combine (`VLLM_VULKAN_LING_MOE_INDIRECT` +
/// expert-batch: matvec-router → grouped-topk → GPU gather descriptors → batched
/// expert gate/up/silu/mul/down → weighted accum + shared expert) into an OPEN
/// command buffer: read `hn`, write `out`. Recording identical to
/// `moe_combine_batched_fused` sans the outer submit/readback. Requires every
/// expert resident + the router/slot buffers (the caller gates on that). Scratch
/// pushed to `g`.
#[allow(clippy::too_many_arguments)]
fn moe_record(
    eng: &mut compute::ComputeEngine,
    cb: vk::CommandBuffer,
    m: &MoeGpuR,
    hn: &compute::Buffer,
    out: &compute::Buffer,
    h: usize,
    g: &mut Vec<compute::Buffer>,
) -> Result<(), String> {
    let top_k = m.top_k;
    if top_k != 8 { return Err(format!("ling moe top_k {top_k} != 8 (q35_moe_accum is fixed-8)")); }
    let n_ex = top_k as u32;
    let inter = m.inter;
    let sh_inter = m.sh_inter;

    let (rg_buf, eb_buf) = match (&m.router_gate_buf, &m.expert_bias_buf) {
        (Some(gb), Some(bb)) => (gb, bb),
        _ => return Err("moe_record requires resident router buffers".into()),
    };
    let (slot_g, slot_u, slot_d) = match (&m.gate.slot_buf, &m.up.slot_buf, &m.down.slot_buf) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return Err("moe_record requires resident slot buffers".into()),
    };

    let logits = alloc(eng, m.e)?;
    let router_out = alloc(eng, 2 * top_k)?;
    let meta_gate = alloc(eng, top_k * 4)?;
    let meta_up = alloc(eng, top_k * 4)?;
    let meta_down = alloc(eng, top_k * 4)?;
    let scores = alloc(eng, top_k)?;

    let b_g_all = alloc(eng, top_k * inter)?;
    let b_u_all = alloc(eng, top_k * inter)?;
    let b_act_all = alloc(eng, top_k * inter)?;
    let b_mid_all = alloc(eng, top_k * inter)?;
    let dwn_all = alloc(eng, top_k * h)?;

    let (sgu_shader, sgu_r) = if m.sh_gate.f16 {
        matvec_variant_by_format(crate::flags::QuantFormat::F16, sh_inter)
    } else { matvec_f32_variant(sh_inter) };
    let s_wg_i = (sh_inter as u32 + sgu_r - 1) / sgu_r;
    let (sd_shader, sd_r) = if m.sh_down.f16 {
        matvec_variant_by_format(crate::flags::QuantFormat::F16, h)
    } else { matvec_f32_variant(h) };
    let s_wg_h = (h as u32 + sd_r - 1) / sd_r;
    let pc_sgu = matvec_pc13(h, sh_inter);
    let pc_sd = matvec_pc13(sh_inter, h);
    let ss_pc = ew_unary_pc(sh_inter as u32);
    let sm_pc = ew_mul_pc(sh_inter as u32);
    let silu_wg_s = (sh_inter as u32 + 511) / 512;
    let mul_wg_s = (sh_inter as u32 + 255) / 256;
    let b_sg = alloc(eng, sh_inter)?; let b_su = alloc(eng, sh_inter)?;
    let b_sa = alloc(eng, sh_inter)?; let b_sm = alloc(eng, sh_inter)?;
    let b_sop = alloc(eng, h)?;

    let all_i = (top_k * inter) as u32;
    let silu_pc = ew_unary_pc(all_i);
    let mul_pc = ew_mul_pc(all_i);
    let silu_wg = (all_i + 511) / 512;
    let mul_wg = (all_i + 255) / 256;

    let slog = { let b = alloc(eng, 1)?; b.write(&f32_slice_to_bytes(&[30.0]))?; b };
    let acc_pc = q35_moe_accum_batched_pc(1, h, top_k);

    let ps_gu = m.gate.out * (m.gate.inn / 8);
    let sb_gu = m.gate.out * (m.gate.inn / EXPERT_GROUP);
    let ps_dn = m.down.out * (m.down.inn / 8);
    let sb_dn = m.down.out * (m.down.inn / EXPERT_GROUP);
    let router_pc = ling_moe_router_pc(
        m.e, top_k, m.n_group, m.topk_group, m.scale, m.norm_topk_prob);
    let meta_pc = ling_moe_meta_pc(
        top_k, m.gate.out, ps_gu, sb_gu, m.down.out, ps_dn, sb_dn, m.down.inn);
    let (mv_shader, mv_r) = matvec_f32_variant(m.e);
    let mv_wg = (m.e as u32 + mv_r - 1) / mv_r;
    let mv_pc = matvec_pc13(h, m.e);

    // GPU route (fast matvec -> grouped-topk) -> GPU gather descriptors.
    eng.record_to(cb, &mv_shader, &[rg_buf, hn, &logits], &mv_pc, (mv_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    eng.record_to(cb, "ling_moe_router", &[&logits, eb_buf, &router_out], &router_pc, (1, 1, 1))?;
    eng.record_barrier_to(cb);
    eng.record_to(cb, "ling_moe_meta",
        &[&router_out, slot_g, slot_u, slot_d, &meta_gate, &meta_up, &meta_down, &scores],
        &meta_pc, (1, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage A: batched gate + up (routed) + shared gate/up
    record_batched_matvec(eng, cb, &m.gate, hn, &b_g_all, &meta_gate, n_ex)?;
    record_batched_matvec(eng, cb, &m.up, hn, &b_u_all, &meta_up, n_ex)?;
    eng.record_to(cb, &sgu_shader, &[&m.sh_gate.buf, hn, &b_sg], &pc_sgu, (s_wg_i, 1, 1))?;
    eng.record_to(cb, &sgu_shader, &[&m.sh_up.buf, hn, &b_su], &pc_sgu, (s_wg_i, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage B: silu (routed batched + shared)
    eng.record_to(cb, "silu_f32", &[&b_g_all, &b_act_all], &silu_pc, (silu_wg, 1, 1))?;
    eng.record_to(cb, "silu_f32", &[&b_sg, &b_sa], &ss_pc, (silu_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage C: mul(silu, up)
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_act_all, &b_u_all, &b_mid_all], &mul_pc, (mul_wg, 1, 1))?;
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_sa, &b_su, &b_sm], &sm_pc, (mul_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage D: batched down (routed) + shared down
    record_batched_matvec(eng, cb, &m.down, &b_mid_all, &dwn_all, &meta_down, n_ex)?;
    eng.record_to(cb, &sd_shader, &[&m.sh_down.buf, &b_sm, &b_sop], &pc_sd, (s_wg_h, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage E: weighted accum + ungated shared (sigmoid(30)) -> out.
    eng.record_to(cb, "q35_moe_accum_batched",
        &[&dwn_all, &scores, &b_sop, &slog, out],
        &acc_pc, ((h as u32 + 255) / 256, 1, 1))?;

    for b in [logits, router_out, meta_gate, meta_up, meta_down, scores,
              b_g_all, b_u_all, b_act_all, b_mid_all, dwn_all,
              b_sg, b_su, b_sa, b_sm, b_sop, slog] {
        g.push(b);
    }
    Ok(())
}

/// The resident KDA decode step: f32 matvec projections (q/k/v/f/g/b) → host
/// conv/L2-norm/decay glue (VERBATIM from `ling::LingKdaWeights::decode_step`) →
/// the reused `kda_gdn_step` recurrence advancing the resident state → f32 o_proj.
fn kda_step_resident(
    eng: &mut compute::ComputeEngine,
    kda: &mut KdaGpuR,
    x: &[f32],
) -> Result<Vec<f32>, String> {
    let (nh, hd, kern, eps) = (kda.nh, kda.hd, kda.kern, kda.eps);
    let proj = nh * hd;
    let scale = (hd as f32).powf(-0.5);
    if x.len() != kda.q.k { return Err(format!("kda x {} != k {}", x.len(), kda.q.k)); }

    // ---- SEGMENT 1: 6 projections in ONE command buffer / ONE fence ----
    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let o_q = alloc(eng, kda.q.n)?;
    let o_k = alloc(eng, kda.k.n)?;
    let o_v = alloc(eng, kda.v.n)?;
    let o_f = alloc(eng, kda.f.n)?;
    let o_g = alloc(eng, kda.g.n)?;
    let o_b = alloc(eng, kda.b.n)?;
    let cb = eng.begin_batch()?;
    record_matvec_f32(eng, cb, &kda.q, &xbuf, &o_q)?;
    record_matvec_f32(eng, cb, &kda.k, &xbuf, &o_k)?;
    record_matvec_f32(eng, cb, &kda.v, &xbuf, &o_v)?;
    record_matvec_f32(eng, cb, &kda.f, &xbuf, &o_f)?;
    record_matvec_f32(eng, cb, &kda.g, &xbuf, &o_g)?;
    record_matvec_f32(eng, cb, &kda.b, &xbuf, &o_b)?;
    eng.submit_batch(cb)?;
    let qc = read_f32_buf(&o_q, kda.q.n);
    let kc = read_f32_buf(&o_k, kda.k.n);
    let vc = read_f32_buf(&o_v, kda.v.n);
    let f_dec = read_f32_buf(&o_f, kda.f.n);
    let gate = read_f32_buf(&o_g, kda.g.n); // pre-sigmoid g_proj(x)
    let b_in = read_f32_buf(&o_b, kda.b.n); // pre-sigmoid beta [nh]
    for buf in [xbuf, o_q, o_k, o_v, o_f, o_g, o_b] { eng.return_to_pool(buf); }

    // ---- host glue: conv sliding-window + L2-norm + decay (verbatim decode_step) ----
    let mut q = ling::ling_conv_step(&mut kda.conv.conv_q, &qc, proj, &kda.q_conv, kern);
    let mut k = ling::ling_conv_step(&mut kda.conv.conv_k, &kc, proj, &kda.k_conv, kern);
    let v = ling::ling_conv_step(&mut kda.conv.conv_v, &vc, proj, &kda.v_conv, kern);
    for hh in 0..nh {
        let qh = &mut q[hh * hd..(hh + 1) * hd];
        let qn = 1.0 / (qh.iter().map(|z| z * z).sum::<f32>() + eps).sqrt();
        for z in qh.iter_mut() { *z *= qn * scale; }
        let kh = &mut k[hh * hd..(hh + 1) * hd];
        let kn = 1.0 / (kh.iter().map(|z| z * z).sum::<f32>() + eps).sqrt();
        for z in kh.iter_mut() { *z *= kn; }
    }
    let mut decay = vec![0f32; proj];
    for hh in 0..nh {
        let a_hh = kda.a_log[hh];
        for dk in 0..hd {
            let fb = f_dec[hh * hd + dk] + kda.dt_bias[hh * hd + dk];
            let g = if kda.safe_gate {
                kda.lower_bound * ling::sigmoid(a_hh.exp() * fb)
            } else {
                -(a_hh.exp()) * ling::softplus(fb)
            };
            decay[hh * hd + dk] = g.exp();
        }
    }

    // ---- SEGMENT 2: kda_gdn_step (recurrence + gated o_norm) + o_proj ----
    let key_dim = proj;
    let value_dim = proj;
    let conv_dim = 2 * key_dim + value_dim;
    let v_off = 2 * key_dim;
    let mut params = vec![0f32; 2 * nh];
    params.extend_from_slice(&kda.o_norm);
    let g_params = {
        let pb = f32_slice_to_bytes(&params);
        let bf = eng.alloc_host_coherent_storage(pb.len() as u64)?;
        bf.write(&pb)?; bf
    };
    let b_q = alloc(eng, key_dim)?;
    let b_k = alloc(eng, key_dim)?;
    let b_conv = alloc(eng, conv_dim)?;
    let b_decay = alloc(eng, proj)?;
    let b_b = alloc(eng, nh)?;
    let b_gate = alloc(eng, value_dim)?;
    let b_gated = alloc(eng, value_dim)?;
    let o_out = alloc(eng, kda.o.n)?;
    let mut conv_scratch = vec![0f32; conv_dim];
    conv_scratch[v_off..v_off + value_dim].copy_from_slice(&v);
    b_q.write(&f32_slice_to_bytes(&q))?;
    b_k.write(&f32_slice_to_bytes(&k))?;
    b_conv.write(&f32_slice_to_bytes(&conv_scratch))?;
    b_decay.write(&f32_slice_to_bytes(&decay))?;
    b_b.write(&f32_slice_to_bytes(&b_in))?;
    b_gate.write(&f32_slice_to_bytes(&gate))?;
    let pc = q35_gdn_pc(hd, hd, 1, v_off, eps, nh);
    let cb = eng.begin_batch()?;
    eng.record_to(cb, "kda_gdn_step",
        &[&b_q, &b_k, &b_conv, &b_decay, &b_b, &b_gate, &g_params, &kda.g_state, &b_gated],
        &pc, (nh as u32, 1, 1))?;
    eng.record_barrier_to(cb);
    record_matvec_f32(eng, cb, &kda.o, &b_gated, &o_out)?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o_out, kda.o.n);
    for buf in [g_params, b_q, b_k, b_conv, b_decay, b_b, b_gate, b_gated, o_out] {
        eng.return_to_pool(buf);
    }
    Ok(out)
}

/// Fused Ling KDA decode step (Phase 3): the ENTIRE KDA layer in ONE command
/// buffer / ONE fence. The 6 direct projections (q/k/v/f/g/b) → depthwise
/// conv+silu (`q35_dn_conv_step`, advancing the GPU-resident conv windows) →
/// L2-norm qknorm (`ling_kda_l2norm`) + safe_gate decay (`ling_kda_decay`) →
/// delta-rule recurrence + gated o_norm (`kda_gdn_step`) → o_proj all record
/// into ONE submit — no host round-trip (vs `kda_step_resident`'s 2 submits +
/// fence + readback/re-upload). Ling is `no_kda_lora` (full-rank f/g), so there
/// are no low-rank tails — f_proj/g_proj are direct. Numerically identical to
/// the 2-submit path (SAME projection/recurrence/o_proj kernels; the
/// moved-to-GPU conv-silu / L2-sqrt / decay-exp differ only at the last ulp,
/// GPU intrinsic vs libm, accumulation order bit-identical) → argmax-exact /
/// cos≈1.0, the accepted LingGpuStage decode tolerance.
///
/// Buffer layout mirrors `kimi_gpu::kda_step_resident_fused`: the three q/k/v
/// conv outputs are placed into ONE combined `b_conv`
/// `[conv_dim = 2*key_dim + value_dim]` (q at `[0,key_dim)`, k at
/// `[key_dim,2*key_dim)`, v at `[2*key_dim,conv_dim)`) via `record_to_off`
/// output offsets — exactly what `ling_kda_l2norm` (reads q/k) and `kda_gdn_step`
/// (reads v at `v_off`) expect.
fn kda_step_resident_fused(
    eng: &mut compute::ComputeEngine,
    kda: &mut KdaGpuR,
    x: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    let (nh, hd, kern) = (kda.nh, kda.hd, kda.kern);
    let proj = nh * hd;
    let key_dim = proj;   // KDA: nk == nh, kd == hd
    let value_dim = proj; // nv == nh, vd == hd
    let conv_dim = 2 * key_dim + value_dim;
    let v_off = 2 * key_dim;
    let scale = (hd as f32).powf(-0.5);
    if x.len() != kda.q.k {
        return Err(format!("kda x {} != k {}", x.len(), kda.q.k));
    }

    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    // projection outputs
    let o_q = alloc(eng, kda.q.n)?;
    let o_k = alloc(eng, kda.k.n)?;
    let o_v = alloc(eng, kda.v.n)?;
    let o_f = alloc(eng, kda.f.n)?; // f_proj(x) [proj] -> decay input
    let o_g = alloc(eng, kda.g.n)?; // g_proj(x) [value_dim] (pre-sigmoid gate)
    let o_b = alloc(eng, kda.b.n)?; // b_proj(x) [nh] (pre-sigmoid beta)
    // fused intermediates
    let b_conv = alloc(eng, conv_dim)?;
    let b_decay = alloc(eng, proj)?;
    let b_q = alloc(eng, key_dim)?;
    let b_k = alloc(eng, key_dim)?;
    let b_gated = alloc(eng, value_dim)?;
    let o_out = alloc(eng, kda.o.n)?;

    let conv_pc = q35_conv_pc(key_dim, kern); // key_dim == value_dim == proj
    let qk_pc = q35_qknorm_pc(nh, hd, key_dim, kda.eps, scale);
    let decay_pc = ling_kda_decay_pc(nh, hd, kda.lower_bound);
    let gdn_pc = q35_gdn_pc(hd, hd, 1, v_off, eps, nh);
    let conv_wg = ((key_dim as u32) + 255) / 256;
    let decay_wg = ((proj as u32) + 255) / 256;

    // ---- ENTIRE KDA layer in ONE command buffer / ONE fence ----
    let cb = eng.begin_batch()?;
    // stage A: 6 direct projections (read x).
    record_matvec_f32(eng, cb, &kda.q, &xbuf, &o_q)?;
    record_matvec_f32(eng, cb, &kda.k, &xbuf, &o_k)?;
    record_matvec_f32(eng, cb, &kda.v, &xbuf, &o_v)?;
    record_matvec_f32(eng, cb, &kda.f, &xbuf, &o_f)?;
    record_matvec_f32(eng, cb, &kda.g, &xbuf, &o_g)?;
    record_matvec_f32(eng, cb, &kda.b, &xbuf, &o_b)?;
    eng.record_barrier_to(cb); // projections written -> conv + decay
    // stage B: depthwise conv+silu q/k/v into the combined b_conv (offset outputs;
    // advances the resident conv windows in place) + safe_gate per-channel decay.
    eng.record_to_off(cb, "q35_dn_conv_step",
        &[(&kda.q_conv_buf, 0), (&o_q, 0), (&kda.conv_state_q, 0), (&b_conv, 0)],
        &conv_pc, (conv_wg, 1, 1))?;
    eng.record_to_off(cb, "q35_dn_conv_step",
        &[(&kda.k_conv_buf, 0), (&o_k, 0), (&kda.conv_state_k, 0), (&b_conv, (key_dim * 4) as u64)],
        &conv_pc, (conv_wg, 1, 1))?;
    eng.record_to_off(cb, "q35_dn_conv_step",
        &[(&kda.v_conv_buf, 0), (&o_v, 0), (&kda.conv_state_v, 0), (&b_conv, (v_off * 4) as u64)],
        &conv_pc, (conv_wg, 1, 1))?;
    eng.record_to(cb, "ling_kda_decay",
        &[&o_f, &kda.dt_bias_buf, &kda.a_log_buf, &b_decay],
        &decay_pc, (decay_wg, 1, 1))?;
    eng.record_barrier_to(cb); // conv (q/k) + decay written -> qknorm
    // stage C: L2-norm qknorm (reads q/k from b_conv, writes b_q/b_k; q scaled).
    eng.record_to(cb, "ling_kda_l2norm",
        &[&b_conv, &b_q, &b_k],
        &qk_pc, (2 * nh as u32, 1, 1))?;
    eng.record_barrier_to(cb); // b_q/b_k written -> gdn_step
    // stage D: KDA delta-rule recurrence + gated o_norm (advances g_state in place;
    // v read from b_conv at v_off; beta from o_b; gate from o_g).
    eng.record_to(cb, "kda_gdn_step",
        &[&b_q, &b_k, &b_conv, &b_decay, &o_b, &o_g, &kda.g_params, &kda.g_state, &b_gated],
        &gdn_pc, (nh as u32, 1, 1))?;
    eng.record_barrier_to(cb); // b_gated written -> o_proj
    // stage E: o_proj.
    record_matvec_f32(eng, cb, &kda.o, &b_gated, &o_out)?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o_out, kda.o.n);

    for buf in [
        xbuf, o_q, o_k, o_v, o_f, o_g, o_b,
        b_conv, b_decay, b_q, b_k, b_gated, o_out,
    ] {
        eng.return_to_pool(buf);
    }
    Ok(out)
}

/// The resident MLA decode step (`VLLM_VULKAN_LING_MLA_RESIDENT`): the 6 MLA
/// projections dispatch `record_matvec_f32` against HELD `GpuMatF32`s; the
/// interleaved-RoPE / kv_a_layernorm / causal-SDPA / head-gate seam stays on the
/// host via `ling::{mla_rope_split, mla_attend_gate}` — the SAME code the CPU
/// golden `LingMlaWeights::decode_step` runs. The two paths therefore differ ONLY
/// in the projection accumulation order (GPU subgroup-reduce vs host sequential
/// dot) → argmax-exact / cos≈1.0, the accepted LingGpuStage decode tolerance.
///
/// Three fenced submits split at the two host seams: {q_proj, kv_a_proj, g_proj}
/// all read `x`; then host RoPE/layernorm produces the latent `c_kv` →
/// {embed_q, unembed_out} decompress against it; then host softmax-SDPA + head
/// gate produce the attention output → {dense}. The RoPE position is the cache
/// length before the append (== the query row index in prefill).
fn mla_step_resident(
    eng: &mut compute::ComputeEngine,
    mla: &mut MlaGpuR,
    x: &[f32],
) -> Result<Vec<f32>, String> {
    let (h, nh, nope, pe, v, r) = (mla.h, mla.nh, mla.nope, mla.pe, mla.v, mla.r);
    if x.len() != h {
        return Err(format!("mla x {} != h {h}", x.len()));
    }
    let pos = mla.cache.len(); // RoPE pos = cache len before append.

    // ---- SEGMENT 1: q_proj + kv_a_proj (+ g_proj) — all read x — ONE submit ----
    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let o_q = alloc(eng, mla.q_proj.n)?;
    let o_kva = alloc(eng, mla.kv_a_proj.n)?;
    let o_g = match mla.g_proj.as_ref() {
        Some(gp) => Some(alloc(eng, gp.n)?),
        None => None,
    };
    let cb = eng.begin_batch()?;
    record_matvec_f32(eng, cb, &mla.q_proj, &xbuf, &o_q)?;
    record_matvec_f32(eng, cb, &mla.kv_a_proj, &xbuf, &o_kva)?;
    if let (Some(gp), Some(og)) = (mla.g_proj.as_ref(), o_g.as_ref()) {
        record_matvec_f32(eng, cb, gp, &xbuf, og)?;
    }
    eng.submit_batch(cb)?;
    let q = read_f32_buf(&o_q, mla.q_proj.n); // [nh*qhd]
    let kva = read_f32_buf(&o_kva, mla.kv_a_proj.n); // [r+pe]
    let g = o_g.as_ref().map(|og| read_f32_buf(og, mla.g_proj.as_ref().unwrap().n));
    eng.return_to_pool(xbuf);
    eng.return_to_pool(o_q);
    eng.return_to_pool(o_kva);
    if let Some(og) = o_g {
        eng.return_to_pool(og);
    }

    // host glue: interleaved-RoPE + split + kv_a_layernorm (shared with the golden).
    let (q_nope, q_pe, c_kv, kpe_new) = ling::mla_rope_split(
        &q, &kva, pos, nh, nope, pe, r, mla.eps, mla.rope_theta, &mla.kv_a_layernorm,
    );

    // ---- SEGMENT 2: decompress k_nope + v — both matvec the layer-normed latent ----
    let lb = f32_slice_to_bytes(&c_kv);
    let latbuf = eng.alloc_host_coherent_storage(lb.len().max(4) as u64)?;
    latbuf.write(&lb)?;
    let o_kn = alloc(eng, mla.embed_q.n)?;
    let o_v = alloc(eng, mla.unembed_out.n)?;
    let cb = eng.begin_batch()?;
    record_matvec_f32(eng, cb, &mla.embed_q, &latbuf, &o_kn)?;
    record_matvec_f32(eng, cb, &mla.unembed_out, &latbuf, &o_v)?;
    eng.submit_batch(cb)?;
    let kn_new = read_f32_buf(&o_kn, mla.embed_q.n); // [nh*nope]
    let v_new = read_f32_buf(&o_v, mla.unembed_out.n); // [nh*v]
    eng.return_to_pool(latbuf);
    eng.return_to_pool(o_kn);
    eng.return_to_pool(o_v);

    // host softmax-SDPA seam (append + causal softmax + value combine + head gate) —
    // the bit-exact code shared with the CPU golden `decode_step`.
    let attn = ling::mla_attend_gate(
        &mut mla.cache, &q_nope, &q_pe, &kn_new, &kpe_new, &v_new,
        g.as_deref(), nh, nope, pe, v, mla.head_gate,
    );

    // ---- SEGMENT 3: dense output projection ----
    if attn.len() != mla.dense.k {
        return Err(format!("mla dense in {} != k {}", attn.len(), mla.dense.k));
    }
    let ab = f32_slice_to_bytes(&attn);
    let abuf = eng.alloc_host_coherent_storage(ab.len().max(4) as u64)?;
    abuf.write(&ab)?;
    let o_out = alloc(eng, mla.dense.n)?;
    let cb = eng.begin_batch()?;
    record_matvec_f32(eng, cb, &mla.dense, &abuf, &o_out)?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o_out, mla.dense.n);
    eng.return_to_pool(abuf);
    eng.return_to_pool(o_out);
    Ok(out)
}

/// Resident dense SwiGLU MLP decode: `silu(gate·x) ⊙ (up·x) → down`, f32 on the GPU.
fn dense_step_resident(
    eng: &mut compute::ComputeEngine,
    d: &DenseGpuR,
    hn: &[f32],
) -> Result<Vec<f32>, String> {
    let (h, inter) = (d.h, d.inter);
    if hn.len() != h { return Err(format!("dense hn {} != h {h}", hn.len())); }
    let xb = f32_slice_to_bytes(hn);
    let inp = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    inp.write(&xb)?;
    let b_g = alloc(eng, inter)?;
    let b_u = alloc(eng, inter)?;
    let b_a = alloc(eng, inter)?;
    let b_m = alloc(eng, inter)?;
    let b_out = alloc(eng, h)?;
    let silu_pc = ew_unary_pc(inter as u32);
    let mul_pc = ew_mul_pc(inter as u32);
    let silu_wg = (inter as u32 + 511) / 512;
    let mul_wg = (inter as u32 + 255) / 256;
    let cb = eng.begin_batch()?;
    record_matvec_f32(eng, cb, &d.gate, &inp, &b_g)?;
    record_matvec_f32(eng, cb, &d.up, &inp, &b_u)?;
    eng.record_barrier_to(cb);
    eng.record_to(cb, "silu_f32", &[&b_g, &b_a], &silu_pc, (silu_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_a, &b_u, &b_m], &mul_pc, (mul_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    record_matvec_f32(eng, cb, &d.down, &b_m, &b_out)?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&b_out, h);
    for buf in [inp, b_g, b_u, b_a, b_m, b_out] { eng.return_to_pool(buf); }
    Ok(out)
}

/// Record one selected expert's gate/up (or down) mlx4 matvec — resident slot in the
/// concatenated switch, or a transient upload for an overflow-streamed expert. Any
/// transient buffers are pushed to `transient` for freeing after submit.
#[allow(clippy::too_many_arguments)]
fn record_expert_matvec(
    eng: &mut compute::ComputeEngine,
    cb: vk::CommandBuffer,
    sw: &SwitchR,
    ei: usize,
    xin: &compute::Buffer,
    out: &compute::Buffer,
    transient: &mut Vec<compute::Buffer>,
) -> Result<(), String> {
    let (k, n) = (sw.inn, sw.out);
    let (shader, r) = matvec_mlx4_variant_k(k, n);
    let wg = (n as u32 + r - 1) / r;
    let slot = sw.slot[ei];
    if slot >= 0 {
        let s = slot as usize;
        let pack_stride = n * (k / 8);
        let sb_stride = n * (k / EXPERT_GROUP);
        let pc = matvec_mlx4_pc_off(k, n, EXPERT_GROUP, s * pack_stride, s * sb_stride);
        eng.record_to(cb, &shader, &[&sw.p, &sw.s, &sw.b, xin, out], &pc, (wg, 1, 1))
    } else {
        // overflow-streamed: read the on-disk bytes on demand, upload transient.
        let (shard, base) = sw.strm[ei].as_ref().ok_or("streamed expert has no descriptor")?;
        let (packed, scales, biases, o2, i2) = LingExpertQ::read_streamed(shard, base)?;
        debug_assert_eq!((o2, i2), (n, k));
        let pb = bytemuck::cast_slice::<u32, u8>(&packed).to_vec();
        let p = eng.alloc_host_coherent_storage(pb.len().max(4) as u64)?;
        p.write(&pb)?;
        let sbuf = eng.alloc_host_coherent_storage(f32_slice_to_bytes(&scales).len().max(4) as u64)?;
        sbuf.write(&f32_slice_to_bytes(&scales))?;
        let bbuf = eng.alloc_host_coherent_storage(f32_slice_to_bytes(&biases).len().max(4) as u64)?;
        bbuf.write(&f32_slice_to_bytes(&biases))?;
        let pc = matvec_mlx4_pc_off(k, n, EXPERT_GROUP, 0, 0);
        eng.record_to(cb, &shader, &[&p, &sbuf, &bbuf, xin, out], &pc, (wg, 1, 1))?;
        transient.push(p);
        transient.push(sbuf);
        transient.push(bbuf);
        Ok(())
    }
}

/// Phase-1 GPU grouped-topk router: dispatch `ling_moe_router` (matvec + sigmoid
/// + bias + grouped-topk) over the resident router_gate/expert_bias, read back the
/// top-k idx+weights. Numerically the same SELECTION as `grouped_topk_route`
/// (argmax-exact; the router matvec's reduction order differs by ~1 ULP but does
/// not flip the top-k set — the on-node gate arbitrates). Falls through to the
/// host route when the resident router buffers are absent (lever OFF / not built).
fn gpu_router_topk(
    eng: &mut compute::ComputeEngine,
    m: &MoeGpuR,
    hn: &[f32],
    h: usize,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let (gbuf, bbuf) = match (&m.router_gate_buf, &m.expert_bias_buf) {
        (Some(g), Some(b)) => (g, b),
        _ => {
            // resident router buffers not built — host fallback.
            let logits = model::cpu_matmul(hn, &m.router_gate, 1, h, m.e);
            return Ok(ling::grouped_topk_route(
                &logits, &m.expert_bias, m.top_k, m.n_group, m.topk_group, m.scale, m.norm_topk_prob,
            ));
        }
    };
    let top_k = m.top_k;
    let xb = f32_slice_to_bytes(hn);
    let inp = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    inp.write(&xb)?;
    let logits = alloc(eng, m.e)?;
    let out = alloc(eng, 2 * top_k)?;
    // router matvec via the fast coalesced kernel: logits[e] = router_gate[e,h]·hn.
    let (mv_shader, mv_r) = matvec_f32_variant(m.e);
    let mv_wg = (m.e as u32 + mv_r - 1) / mv_r;
    let mv_pc = matvec_pc13(h, m.e);
    let pc = ling_moe_router_pc(
        m.e, top_k, m.n_group, m.topk_group, m.scale, m.norm_topk_prob,
    );
    let cb = eng.begin_batch()?;
    eng.record_to(cb, &mv_shader, &[gbuf, &inp, &logits], &mv_pc, (mv_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    eng.record_to(cb, "ling_moe_router", &[&logits, bbuf, &out], &pc, (1, 1, 1))?;
    eng.submit_batch(cb)?;
    let raw = read_f32_buf(&out, 2 * top_k);
    eng.return_to_pool(inp);
    eng.return_to_pool(logits);
    eng.return_to_pool(out);
    let inds: Vec<usize> = raw[..top_k].iter().map(|&f| f as usize).collect();
    let weights: Vec<f32> = raw[top_k..2 * top_k].to_vec();
    Ok((inds, weights))
}

/// Route selection for one MoE layer — GPU grouped-topk router (`moe_indirect`)
/// or the host `cpu_matmul` + `grouped_topk_route`. Both return argmax-exact
/// (idx, weights); the GPU path keeps the router matvec off the host.
fn moe_route(
    eng: &mut compute::ComputeEngine,
    m: &MoeGpuR,
    hn: &[f32],
    h: usize,
    gpu_router: bool,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if gpu_router {
        gpu_router_topk(eng, m, hn, h)
    } else {
        let logits = model::cpu_matmul(hn, &m.router_gate, 1, h, m.e);
        Ok(ling::grouped_topk_route(
            &logits, &m.expert_bias, m.top_k, m.n_group, m.topk_group, m.scale, m.norm_topk_prob,
        ))
    }
}

/// Resident MoE combine: host grouped-topk route + GPU int4 routed experts (resident
/// switch or transient stream) + f32 ungated shared expert + `q35_moe_accum`.
fn moe_combine_resident(
    eng: &mut compute::ComputeEngine,
    m: &MoeGpuR,
    hn: &[f32],
    h: usize,
    gpu_router: bool,
) -> Result<Vec<f32>, String> {
    // router selection (grouped-topk, argmax-exact vs CPU block; GPU or host).
    let (inds, weights) = moe_route(eng, m, hn, h, gpu_router)?;
    moe_combine_resident_with_route(eng, m, hn, h, inds, weights)
}

/// The per-expert MoE combine body, given an ALREADY-selected route (idx+weights).
/// Split out so the batched path's overflow-streamed fallback reuses the same
/// GPU-router selection instead of re-routing.
fn moe_combine_resident_with_route(
    eng: &mut compute::ComputeEngine,
    m: &MoeGpuR,
    hn: &[f32],
    h: usize,
    inds: Vec<usize>,
    weights: Vec<f32>,
) -> Result<Vec<f32>, String> {
    let top_k = inds.len();
    if top_k != 8 { return Err(format!("ling moe top_k {top_k} != 8 (q35_moe_accum is fixed-8)")); }
    let inter = m.inter;
    let sh_inter = m.sh_inter;

    let xb = f32_slice_to_bytes(hn);
    let inp = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    inp.write(&xb)?;

    // shared-expert (bf16-origin dense, ungated) shaders/pcs — format-aware so an
    // f16-held shared expert (the LING_F16_ATTN lever) dispatches the f16 matvec.
    let (sgu_shader, sgu_r) = if m.sh_gate.f16 {
        matvec_variant_by_format(crate::flags::QuantFormat::F16, sh_inter)
    } else {
        matvec_f32_variant(sh_inter)
    };
    let s_wg_i = (sh_inter as u32 + sgu_r - 1) / sgu_r;
    let (sd_shader, sd_r) = if m.sh_down.f16 {
        matvec_variant_by_format(crate::flags::QuantFormat::F16, h)
    } else {
        matvec_f32_variant(h)
    };
    let s_wg_h = (h as u32 + sd_r - 1) / sd_r;
    let pc_sgu = matvec_pc13(h, sh_inter);
    let pc_sd = matvec_pc13(sh_inter, h);
    let ss_pc = ew_unary_pc(sh_inter as u32);
    let sm_pc = ew_mul_pc(sh_inter as u32);
    let silu_wg_i = (inter as u32 + 511) / 512;
    let mul_wg_i = (inter as u32 + 255) / 256;
    let silu_wg_s = (sh_inter as u32 + 511) / 512;
    let mul_wg_s = (sh_inter as u32 + 255) / 256;
    let silu_pc = ew_unary_pc(inter as u32);
    let mul_pc = ew_mul_pc(inter as u32);

    // per-slot intermediate buffers
    let mut b_g: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    let mut b_u: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    let mut b_act: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    let mut b_mid: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    let mut dwn: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    for _ in 0..top_k {
        b_g.push(alloc(eng, inter)?);
        b_u.push(alloc(eng, inter)?);
        b_act.push(alloc(eng, inter)?);
        b_mid.push(alloc(eng, inter)?);
        dwn.push(alloc(eng, h)?);
    }
    let b_sg = alloc(eng, sh_inter)?; let b_su = alloc(eng, sh_inter)?;
    let b_sa = alloc(eng, sh_inter)?; let b_sm = alloc(eng, sh_inter)?;
    let b_sop = alloc(eng, h)?;
    let slp = { let bf = alloc(eng, 1)?; bf.write(&f32_slice_to_bytes(&[30.0]))?; bf };
    let h1z = { let bf = alloc(eng, h)?; bf.write(&f32_slice_to_bytes(&vec![0f32; h]))?; bf };
    let b_out = alloc(eng, h)?;
    let acc_pc = q35_moe_accum_pc(h, &weights);

    let mut transient: Vec<compute::Buffer> = Vec::new();
    let cb = eng.begin_batch()?;
    // stage A: gate + up (routed experts + shared)
    for slot in 0..top_k {
        let ei = inds[slot];
        record_expert_matvec(eng, cb, &m.gate, ei, &inp, &b_g[slot], &mut transient)?;
        record_expert_matvec(eng, cb, &m.up, ei, &inp, &b_u[slot], &mut transient)?;
    }
    eng.record_to(cb, &sgu_shader, &[&m.sh_gate.buf, &inp, &b_sg], &pc_sgu, (s_wg_i, 1, 1))?;
    eng.record_to(cb, &sgu_shader, &[&m.sh_up.buf, &inp, &b_su], &pc_sgu, (s_wg_i, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage B: silu(gate)
    for slot in 0..top_k {
        eng.record_to(cb, "silu_f32", &[&b_g[slot], &b_act[slot]], &silu_pc, (silu_wg_i, 1, 1))?;
    }
    eng.record_to(cb, "silu_f32", &[&b_sg, &b_sa], &ss_pc, (silu_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage C: mul(silu, up)
    for slot in 0..top_k {
        eng.record_to(cb, "mul_f32_f32_f32", &[&b_act[slot], &b_u[slot], &b_mid[slot]], &mul_pc, (mul_wg_i, 1, 1))?;
    }
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_sa, &b_su, &b_sm], &sm_pc, (mul_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage D: down
    for slot in 0..top_k {
        let ei = inds[slot];
        record_expert_matvec(eng, cb, &m.down, ei, &b_mid[slot], &dwn[slot], &mut transient)?;
    }
    eng.record_to(cb, &sd_shader, &[&m.sh_down.buf, &b_sm, &b_sop], &pc_sd, (s_wg_h, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage E: weighted accum + ungated shared + residual (h1z=0)
    let binds: Vec<&compute::Buffer> = vec![
        &dwn[0], &dwn[1], &dwn[2], &dwn[3], &dwn[4], &dwn[5], &dwn[6], &dwn[7],
        &b_sop, &slp, &h1z, &b_out,
    ];
    eng.record_to(cb, "q35_moe_accum", &binds, &acc_pc, ((h as u32 + 255) / 256, 1, 1))?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&b_out, h);

    for buf in [inp, b_sg, b_su, b_sa, b_sm, b_sop, slp, h1z, b_out] { eng.return_to_pool(buf); }
    for buf in b_g { eng.return_to_pool(buf); }
    for buf in b_u { eng.return_to_pool(buf); }
    for buf in b_act { eng.return_to_pool(buf); }
    for buf in b_mid { eng.return_to_pool(buf); }
    for buf in dwn { eng.return_to_pool(buf); }
    for buf in transient { eng.return_to_pool(buf); }
    Ok(out)
}

/// Expert-batched geometry pick for `mul_mat_vec_mlx4repack_batched_f32_f32`. bs64/
/// r4 is the wired default (matches the single-expert repack winner); the
/// `VLLM_VULKAN_LING_MOE_BATCH_BS`/`_R` envs drive the on-node micro-sweep (both
/// must be a compiled combo — see `geom_combos_for`).
fn moe_batch_variant() -> (String, u32) {
    let bs = std::env::var("VLLM_VULKAN_LING_MOE_BATCH_BS").ok()
        .and_then(|v| v.parse::<u32>().ok()).unwrap_or(64);
    let r = std::env::var("VLLM_VULKAN_LING_MOE_BATCH_R").ok()
        .and_then(|v| v.parse::<u32>().ok()).unwrap_or(4);
    (format!("mul_mat_vec_mlx4repack_batched_f32_f32_bs{bs}_r{r}"), r)
}

/// Build the per-expert `meta[]` (uvec4 = packed_off, sb_off, x_off, dst_off) for
/// one MoE sub-projection over the `inds` selected experts, or `None` if ANY
/// selected expert is overflow-streamed (slot < 0) — in which case the caller must
/// fall back to the per-expert path so `MOE_STREAM_OVERFLOW` still composes.
/// `x_stride` = 0 for gate/up (all experts read the same activation) or `sw.inn`
/// for down (each expert reads its own [inter]-wide intermediate).
fn expert_meta_bytes(sw: &SwitchR, inds: &[usize], x_stride: usize) -> Option<Vec<u8>> {
    let pack_stride = sw.out * (sw.inn / 8);
    let sb_stride = sw.out * (sw.inn / EXPERT_GROUP);
    let mut meta: Vec<u32> = Vec::with_capacity(inds.len() * 4);
    for (e, &ei) in inds.iter().enumerate() {
        let slot = sw.slot[ei];
        if slot < 0 { return None; }
        let s = slot as usize;
        meta.push((s * pack_stride) as u32); // packed_off (words)
        meta.push((s * sb_stride) as u32);   // sb_off (elements)
        meta.push((e * x_stride) as u32);    // x_off (floats)
        meta.push((e * sw.out) as u32);      // dst_off (floats)
    }
    Some(bytemuck::cast_slice::<u32, u8>(&meta).to_vec())
}

/// Record one expert-batched mlx4-repack matvec (all `n_ex` selected experts of a
/// sub-projection in ONE dispatch): grid `(ceil(n/r), n_ex, 1)`, each workgroup a
/// single-expert repack matvec reading its base from `meta[gl_WorkGroupID.y]`.
fn record_batched_matvec(
    eng: &mut compute::ComputeEngine,
    cb: vk::CommandBuffer,
    sw: &SwitchR,
    activ: &compute::Buffer,
    dst: &compute::Buffer,
    meta: &compute::Buffer,
    n_ex: u32,
) -> Result<(), String> {
    let (k, n) = (sw.inn, sw.out);
    let (shader, r) = moe_batch_variant();
    let wg = (n as u32 + r - 1) / r;
    let pc = matvec_mlx4_pc(k, n, EXPERT_GROUP);
    eng.record_to(cb, &shader, &[&sw.p, &sw.s, &sw.b, activ, dst, meta], &pc, (wg, n_ex, 1))
}

/// Expert-batched MoE combine — the dispatch-collapse decode lever. Numerically
/// identical to `moe_combine_resident` (same host grouped-topk route, same repack
/// per-byte kernel, same fixed slot-ascending accumulation) but the 8 experts ×
/// {gate,up,down} = 24 per-expert matvec dispatches become 3 batched dispatches
/// (and the 8 silu / 8 mul become 1 each over the concatenated [n_ex,inter]
/// buffer). Falls back to `moe_combine_resident` when any selected expert is
/// overflow-streamed (keeps MOE_STREAM_OVERFLOW composing).
fn moe_combine_batched(
    eng: &mut compute::ComputeEngine,
    m: &MoeGpuR,
    hn: &[f32],
    h: usize,
    gpu_router: bool,
) -> Result<Vec<f32>, String> {
    let (inds, weights) = moe_route(eng, m, hn, h, gpu_router)?;
    let top_k = inds.len();
    if top_k != 8 { return Err(format!("ling moe top_k {top_k} != 8 (q35_moe_accum is fixed-8)")); }

    // Build all three meta tables; if ANY selected expert is streamed, fall back.
    let inter = m.inter;
    let sh_inter = m.sh_inter;
    let (mb_gate, mb_up, mb_down) = match (
        expert_meta_bytes(&m.gate, &inds, 0),
        expert_meta_bytes(&m.up, &inds, 0),
        expert_meta_bytes(&m.down, &inds, m.down.inn),
    ) {
        (Some(g), Some(u), Some(d)) => (g, u, d),
        // an overflow-streamed expert: fall back to the per-expert path, reusing
        // the SAME (already-selected) route so the GPU-router selection composes.
        _ => return moe_combine_resident_with_route(eng, m, hn, h, inds, weights),
    };
    let n_ex = top_k as u32;

    let inp = {
        let xb = f32_slice_to_bytes(hn);
        let b = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
        b.write(&xb)?; b
    };
    let up_meta = |eng: &mut compute::ComputeEngine, bytes: &[u8]| -> Result<compute::Buffer, String> {
        let b = eng.alloc_host_coherent_storage(bytes.len().max(4) as u64)?;
        b.write(bytes)?; Ok(b)
    };
    let meta_gate = up_meta(eng, &mb_gate)?;
    let meta_up = up_meta(eng, &mb_up)?;
    let meta_down = up_meta(eng, &mb_down)?;

    // concatenated [n_ex, inter] / [n_ex, h] work buffers
    let b_g_all = alloc(eng, top_k * inter)?;
    let b_u_all = alloc(eng, top_k * inter)?;
    let b_act_all = alloc(eng, top_k * inter)?;
    let b_mid_all = alloc(eng, top_k * inter)?;
    let dwn_all = alloc(eng, top_k * h)?;

    // shared expert (bf16-origin dense, ungated) — same as moe_combine_resident.
    let (sgu_shader, sgu_r) = if m.sh_gate.f16 {
        matvec_variant_by_format(crate::flags::QuantFormat::F16, sh_inter)
    } else { matvec_f32_variant(sh_inter) };
    let s_wg_i = (sh_inter as u32 + sgu_r - 1) / sgu_r;
    let (sd_shader, sd_r) = if m.sh_down.f16 {
        matvec_variant_by_format(crate::flags::QuantFormat::F16, h)
    } else { matvec_f32_variant(h) };
    let s_wg_h = (h as u32 + sd_r - 1) / sd_r;
    let pc_sgu = matvec_pc13(h, sh_inter);
    let pc_sd = matvec_pc13(sh_inter, h);
    let ss_pc = ew_unary_pc(sh_inter as u32);
    let sm_pc = ew_mul_pc(sh_inter as u32);
    let silu_wg_s = (sh_inter as u32 + 511) / 512;
    let mul_wg_s = (sh_inter as u32 + 255) / 256;
    let b_sg = alloc(eng, sh_inter)?; let b_su = alloc(eng, sh_inter)?;
    let b_sa = alloc(eng, sh_inter)?; let b_sm = alloc(eng, sh_inter)?;
    let b_sop = alloc(eng, h)?;

    // concatenated silu/mul over the whole [n_ex, inter] block in one dispatch each.
    let all_i = (top_k * inter) as u32;
    let silu_pc = ew_unary_pc(all_i);
    let mul_pc = ew_mul_pc(all_i);
    let silu_wg = (all_i + 511) / 512;
    let mul_wg = (all_i + 255) / 256;

    // accum: q35_moe_accum_batched (T=1) reads down_out[n_ex,h], scores[n_ex],
    // shared[h], logits[1]; produces sum(w*down) + sigmoid(30)*shared (ungated
    // shared, == the slp=30 path in moe_combine_resident), no residual (caller adds).
    let scores = { let b = alloc(eng, top_k)?; b.write(&f32_slice_to_bytes(&weights))?; b };
    let slog = { let b = alloc(eng, 1)?; b.write(&f32_slice_to_bytes(&[30.0]))?; b };
    let b_out = alloc(eng, h)?;
    let acc_pc = q35_moe_accum_batched_pc(1, h, top_k);

    let cb = eng.begin_batch()?;
    // stage A: batched gate + up (routed) + shared gate/up
    record_batched_matvec(eng, cb, &m.gate, &inp, &b_g_all, &meta_gate, n_ex)?;
    record_batched_matvec(eng, cb, &m.up, &inp, &b_u_all, &meta_up, n_ex)?;
    eng.record_to(cb, &sgu_shader, &[&m.sh_gate.buf, &inp, &b_sg], &pc_sgu, (s_wg_i, 1, 1))?;
    eng.record_to(cb, &sgu_shader, &[&m.sh_up.buf, &inp, &b_su], &pc_sgu, (s_wg_i, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage B: silu (routed batched + shared)
    eng.record_to(cb, "silu_f32", &[&b_g_all, &b_act_all], &silu_pc, (silu_wg, 1, 1))?;
    eng.record_to(cb, "silu_f32", &[&b_sg, &b_sa], &ss_pc, (silu_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage C: mul(silu, up) (routed batched + shared)
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_act_all, &b_u_all, &b_mid_all], &mul_pc, (mul_wg, 1, 1))?;
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_sa, &b_su, &b_sm], &sm_pc, (mul_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage D: batched down (routed) + shared down
    record_batched_matvec(eng, cb, &m.down, &b_mid_all, &dwn_all, &meta_down, n_ex)?;
    eng.record_to(cb, &sd_shader, &[&m.sh_down.buf, &b_sm, &b_sop], &pc_sd, (s_wg_h, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage E: weighted accum + ungated shared (sigmoid(30)) — one batched dispatch.
    eng.record_to(cb, "q35_moe_accum_batched",
        &[&dwn_all, &scores, &b_sop, &slog, &b_out],
        &acc_pc, ((h as u32 + 255) / 256, 1, 1))?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&b_out, h);

    for buf in [inp, meta_gate, meta_up, meta_down, b_g_all, b_u_all, b_act_all,
                b_mid_all, dwn_all, b_sg, b_su, b_sa, b_sm, b_sop, scores, slog, b_out] {
        eng.return_to_pool(buf);
    }
    Ok(out)
}

/// Phase-1 FULLY-GPU-DRIVEN MoE combine (`VLLM_VULKAN_LING_MOE_INDIRECT`): the
/// route (`ling_moe_router`) + gather-descriptor build (`ling_moe_meta`) + the 3
/// batched expert matvecs + shared expert + accum all record into ONE command
/// buffer — the host never learns which experts were picked (no index readback,
/// no extra submit/fence, in contrast to the separate-submit GPU router which is
/// a large regression). The only host↔GPU traffic is the `hn` upload + the `[h]`
/// output readback (inherent to the host-orchestrated layer loop — removing THAT
/// needs the resident single-CB *layer*, the next lever). Numerically identical
/// to `moe_combine_batched` (same grouped-topk selection, same repack matvec).
///
/// Requires every selected expert resident (`fully_resident`) + the resident
/// router/slot buffers; the caller gates on this and otherwise uses the host
/// route. `top_k` (=8) fixes the batched grid, so no `vkCmdDispatchIndirect` is
/// needed — only the meta CONTENTS are GPU-computed.
fn moe_combine_batched_fused(
    eng: &mut compute::ComputeEngine,
    m: &MoeGpuR,
    hn: &[f32],
    h: usize,
) -> Result<Vec<f32>, String> {
    let top_k = m.top_k;
    if top_k != 8 { return Err(format!("ling moe top_k {top_k} != 8 (q35_moe_accum is fixed-8)")); }
    let n_ex = top_k as u32;
    let inter = m.inter;
    let sh_inter = m.sh_inter;

    let (rg_buf, eb_buf) = match (&m.router_gate_buf, &m.expert_bias_buf) {
        (Some(g), Some(b)) => (g, b),
        _ => return Err("moe_indirect fused path requires resident router buffers".into()),
    };
    let (slot_g, slot_u, slot_d) = match (&m.gate.slot_buf, &m.up.slot_buf, &m.down.slot_buf) {
        (Some(g), Some(u), Some(d)) => (g, u, d),
        _ => return Err("moe_indirect fused path requires resident slot buffers".into()),
    };

    let inp = {
        let xb = f32_slice_to_bytes(hn);
        let b = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
        b.write(&xb)?; b
    };

    // GPU-built router logits + top-k output + gather descriptors + routed scores.
    let logits = alloc(eng, m.e)?;
    let router_out = alloc(eng, 2 * top_k)?;
    let meta_gate = alloc(eng, top_k * 4)?; // top_k uvec4
    let meta_up = alloc(eng, top_k * 4)?;
    let meta_down = alloc(eng, top_k * 4)?;
    let scores = alloc(eng, top_k)?;

    // concatenated [n_ex, inter] / [n_ex, h] work buffers
    let b_g_all = alloc(eng, top_k * inter)?;
    let b_u_all = alloc(eng, top_k * inter)?;
    let b_act_all = alloc(eng, top_k * inter)?;
    let b_mid_all = alloc(eng, top_k * inter)?;
    let dwn_all = alloc(eng, top_k * h)?;

    // shared expert (bf16-origin dense, ungated) — same as moe_combine_batched.
    let (sgu_shader, sgu_r) = if m.sh_gate.f16 {
        matvec_variant_by_format(crate::flags::QuantFormat::F16, sh_inter)
    } else { matvec_f32_variant(sh_inter) };
    let s_wg_i = (sh_inter as u32 + sgu_r - 1) / sgu_r;
    let (sd_shader, sd_r) = if m.sh_down.f16 {
        matvec_variant_by_format(crate::flags::QuantFormat::F16, h)
    } else { matvec_f32_variant(h) };
    let s_wg_h = (h as u32 + sd_r - 1) / sd_r;
    let pc_sgu = matvec_pc13(h, sh_inter);
    let pc_sd = matvec_pc13(sh_inter, h);
    let ss_pc = ew_unary_pc(sh_inter as u32);
    let sm_pc = ew_mul_pc(sh_inter as u32);
    let silu_wg_s = (sh_inter as u32 + 511) / 512;
    let mul_wg_s = (sh_inter as u32 + 255) / 256;
    let b_sg = alloc(eng, sh_inter)?; let b_su = alloc(eng, sh_inter)?;
    let b_sa = alloc(eng, sh_inter)?; let b_sm = alloc(eng, sh_inter)?;
    let b_sop = alloc(eng, h)?;

    let all_i = (top_k * inter) as u32;
    let silu_pc = ew_unary_pc(all_i);
    let mul_pc = ew_mul_pc(all_i);
    let silu_wg = (all_i + 511) / 512;
    let mul_wg = (all_i + 255) / 256;

    let slog = { let b = alloc(eng, 1)?; b.write(&f32_slice_to_bytes(&[30.0]))?; b };
    let b_out = alloc(eng, h)?;
    let acc_pc = q35_moe_accum_batched_pc(1, h, top_k);

    // meta strides (gate/up share shape; down is transposed) — see expert_meta_bytes.
    let ps_gu = m.gate.out * (m.gate.inn / 8);
    let sb_gu = m.gate.out * (m.gate.inn / EXPERT_GROUP);
    let ps_dn = m.down.out * (m.down.inn / 8);
    let sb_dn = m.down.out * (m.down.inn / EXPERT_GROUP);
    let router_pc = ling_moe_router_pc(
        m.e, top_k, m.n_group, m.topk_group, m.scale, m.norm_topk_prob);
    let meta_pc = ling_moe_meta_pc(
        top_k, m.gate.out, ps_gu, sb_gu, m.down.out, ps_dn, sb_dn, m.down.inn);
    let (mv_shader, mv_r) = matvec_f32_variant(m.e);
    let mv_wg = (m.e as u32 + mv_r - 1) / mv_r;
    let mv_pc = matvec_pc13(h, m.e);

    let cb = eng.begin_batch()?;
    // GPU route (fast matvec -> grouped-topk) -> GPU gather descriptors (no host readback).
    eng.record_to(cb, &mv_shader, &[rg_buf, &inp, &logits], &mv_pc, (mv_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    eng.record_to(cb, "ling_moe_router", &[&logits, eb_buf, &router_out], &router_pc, (1, 1, 1))?;
    eng.record_barrier_to(cb);
    eng.record_to(cb, "ling_moe_meta",
        &[&router_out, slot_g, slot_u, slot_d, &meta_gate, &meta_up, &meta_down, &scores],
        &meta_pc, (1, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage A: batched gate + up (routed) + shared gate/up
    record_batched_matvec(eng, cb, &m.gate, &inp, &b_g_all, &meta_gate, n_ex)?;
    record_batched_matvec(eng, cb, &m.up, &inp, &b_u_all, &meta_up, n_ex)?;
    eng.record_to(cb, &sgu_shader, &[&m.sh_gate.buf, &inp, &b_sg], &pc_sgu, (s_wg_i, 1, 1))?;
    eng.record_to(cb, &sgu_shader, &[&m.sh_up.buf, &inp, &b_su], &pc_sgu, (s_wg_i, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage B: silu (routed batched + shared)
    eng.record_to(cb, "silu_f32", &[&b_g_all, &b_act_all], &silu_pc, (silu_wg, 1, 1))?;
    eng.record_to(cb, "silu_f32", &[&b_sg, &b_sa], &ss_pc, (silu_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage C: mul(silu, up) (routed batched + shared)
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_act_all, &b_u_all, &b_mid_all], &mul_pc, (mul_wg, 1, 1))?;
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_sa, &b_su, &b_sm], &sm_pc, (mul_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage D: batched down (routed) + shared down
    record_batched_matvec(eng, cb, &m.down, &b_mid_all, &dwn_all, &meta_down, n_ex)?;
    eng.record_to(cb, &sd_shader, &[&m.sh_down.buf, &b_sm, &b_sop], &pc_sd, (s_wg_h, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage E: weighted accum + ungated shared (sigmoid(30)) — GPU-built scores.
    eng.record_to(cb, "q35_moe_accum_batched",
        &[&dwn_all, &scores, &b_sop, &slog, &b_out],
        &acc_pc, ((h as u32 + 255) / 256, 1, 1))?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&b_out, h);

    for buf in [inp, logits, router_out, meta_gate, meta_up, meta_down, scores,
                b_g_all, b_u_all, b_act_all, b_mid_all, dwn_all,
                b_sg, b_su, b_sa, b_sm, b_sop, slog, b_out] {
        eng.return_to_pool(buf);
    }
    Ok(out)
}
