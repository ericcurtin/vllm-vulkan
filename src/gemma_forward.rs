// SPDX-License-Identifier: Apache-2.0
//! Gemma4 GPU forward path: per-op dispatch (`forward_gpu`) and the fused
//! GPU-resident decode layer (`forward_gemma_gpu_resident`). Extracted
//! verbatim from lib.rs (M1).

use crate::compute;
use crate::model;
use crate::VulkanModel;
use crate::{
    matvec_pc13, matvec_shader, matvec_variant, matvec_variant_core, matvec_mlx4_pc,
    matvec_mlx4_variant_k, nvfp4_dispatch, matvec_fp8_variant, matvec_fp8_pc,
    use_subgroup_flag, matvec_rows_override,
    f32_slice_to_bytes, read_f32_buf,
    sdpa_pc, attn_decode_kernel,
};
use crate::{
    ACT_QKV_IN, ACT_Q_OUT, ACT_K_OUT, ACT_V_OUT, ACT_O_IN, ACT_O_OUT, ACT_FFIN,
    ACT_GATE, ACT_UP, ACT_MID, ACT_DOWN, ACT_GELU, ACT_PLE_G, ACT_PLE_C,
    ACT_COUNT,
};
use crate::{MvKind, QuantAux};
use crate::layer_core::{
    self, FrontParams, GpuRecorder, LayerPtrs, LayerRecorder, NormW, ProjW, Slot,
    TailParams, FRONT_PROJS, NORM_COUNT, SLOT_COUNT, TAIL_PROJS,
};
use crate::flags::QuantFormat;
use crate::prof_add;
use std::time::Instant;

/// Look up a `GpuWeight`'s dispatch kind (per-buffer `format` + packed `aux`)
/// for the resident-layer matvec path. Unlike the older `matvec_variant`
/// (which reads the GLOBAL `VLLM_VULKAN_QUANT` snapshot), this reads the
/// weight's OWN recorded format, so mlx4-packed attn projections and q8_0
/// MLP projections both dispatch the matching kernel regardless of that env
/// var — see `gemma4_unified`'s loader in lib.rs, which uploads attn as Mlx4
/// and MLP as Q8_0 unconditionally.
pub(crate) fn gemma_res_mv_kind(w: &crate::GpuWeight) -> (QuantFormat, MvKind) {
    let kind = match &w.aux {
        None => MvKind::Plain,
        Some(QuantAux::Nvfp4 { scales, group_size, e4m3, global }) =>
            MvKind::Nvfp4 { s: scales as *const _, gs: *group_size, e4m3: *e4m3, global: *global },
        Some(QuantAux::Fp8 { scale, per_row }) =>
            MvKind::Fp8 { s: scale as *const _, per_row: *per_row },
        Some(QuantAux::Mlx4 { scales, biases, group_size }) =>
            MvKind::Mlx4 { s: scales as *const _, b: biases as *const _, gs: *group_size },
    };
    (w.format, kind)
}

/// Record one resident-layer matvec dispatch into `cb`, branching on the
/// weight's own format/aux (via `kind`/`format` from `gemma_res_mv_kind`)
/// instead of the global quant snapshot.
///
/// Whole-buffer form: the input and output are bound at offset 0. Callers that
/// slice one token's row out of a T-wide activation buffer want
/// `record_gemma_mv_off`.
///
/// Deliberately NOT a thin wrapper over `record_gemma_mv_off`. This is the
/// GPU-resident DECODE hot path (7 projections x every layer, every token) and
/// `record_to` binds through fixed-size stack arrays, while `record_to_off`
/// heap-allocates two `Vec`s per dispatch — the exact allocation churn
/// `record_to` was rewritten to remove. The offset form pays that only on the
/// batched-verify LM head (T <= 8 dispatches per call).
pub(crate) unsafe fn record_gemma_mv(
    eng: &mut compute::ComputeEngine,
    cb: ash::vk::CommandBuffer,
    wptr: *const compute::Buffer,
    format: QuantFormat,
    kind: MvKind,
    inp: *const compute::Buffer,
    out: *const compute::Buffer,
    k: usize,
    n: usize,
) {
    match kind {
        MvKind::Mlx4 { s, b, gs } => {
            let (shader, r) = matvec_mlx4_variant_k(k, n);
            let wg = (n as u32 + r - 1) / r;
            let pc = matvec_mlx4_pc(k, n, gs as usize);
            eng.record_to(cb, &shader, &[&*wptr, &*s, &*b, &*inp, &*out], &pc, (wg, 1, 1)).unwrap();
        }
        MvKind::Nvfp4 { s, gs, e4m3, global } => {
            let (shader, r, pc) = nvfp4_dispatch(k, n, gs, e4m3, global);
            let wg = (n as u32 + r - 1) / r;
            eng.record_to(cb, &shader, &[&*wptr, &*s, &*inp, &*out], &pc, (wg, 1, 1)).unwrap();
        }
        MvKind::Fp8 { s, per_row } => {
            let (shader, r) = matvec_fp8_variant(n);
            let wg = (n as u32 + r - 1) / r;
            let pc = matvec_fp8_pc(k, n, per_row);
            eng.record_to(cb, &shader, &[&*wptr, &*s, &*inp, &*out], &pc, (wg, 1, 1)).unwrap();
        }
        MvKind::Plain => {
            let (shader, r) = matvec_variant_core(format, use_subgroup_flag(), true, matvec_rows_override(), n);
            let wg = (n as u32 + r - 1) / r;
            let pc = matvec_pc13(k, n);
            eng.record_to(cb, &shader, &[&*wptr, &*inp, &*out], &pc, (wg, 1, 1)).unwrap();
        }
    }
}

/// Byte-offset form of `record_gemma_mv`: identical dispatch selection, but the
/// INPUT and OUTPUT buffers are bound at `in_off`/`out_off` instead of 0 (the
/// weight and its scale/bias aux buffers are always whole-buffer).
///
/// THE INVARIANT this exists to hold: the shader and its BINDING COUNT come
/// from the weight's OWN recorded `format`/`aux`, never from the process-wide
/// `VLLM_VULKAN_QUANT` snapshot. The three packed formats each need extra
/// buffers the plain matvec does not bind — Mlx4 needs `scales` AND `biases`
/// (5 bindings), Nvfp4/Fp8 need `scales` (4) — so a global-keyed selector that
/// picks a plain/f16 matvec for a packed weight reads 4-bit nibbles as f16 and
/// yields NaN logits, not merely inaccurate ones. That was `22ee4a9`'s defect
/// in the unified per-op path and (via `matvec_variant`/`matvec_cols_variant`)
/// the same defect in `batched_forward`'s LM head.
///
/// `in_off`/`out_off` are BYTE offsets and must satisfy the device's
/// `minStorageBufferOffsetAlignment`; a row stride of `dim * 4` bytes over the
/// dims this path uses (multiples of 64) always does.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn record_gemma_mv_off(
    eng: &mut compute::ComputeEngine,
    cb: ash::vk::CommandBuffer,
    wptr: *const compute::Buffer,
    format: QuantFormat,
    kind: MvKind,
    inp: *const compute::Buffer,
    in_off: u64,
    out: *const compute::Buffer,
    out_off: u64,
    k: usize,
    n: usize,
) {
    match kind {
        MvKind::Mlx4 { s, b, gs } => {
            let (shader, r) = matvec_mlx4_variant_k(k, n);
            let wg = (n as u32 + r - 1) / r;
            let pc = matvec_mlx4_pc(k, n, gs as usize);
            eng.record_to_off(cb, &shader,
                &[(&*wptr, 0), (&*s, 0), (&*b, 0), (&*inp, in_off), (&*out, out_off)],
                &pc, (wg, 1, 1)).unwrap();
        }
        MvKind::Nvfp4 { s, gs, e4m3, global } => {
            // Flag-routed: e4m3-resident kernel or the f32-fold kernel (same 4
            // bindings, differing only in shader name + push constants).
            let (shader, r, pc) = nvfp4_dispatch(k, n, gs, e4m3, global);
            let wg = (n as u32 + r - 1) / r;
            eng.record_to_off(cb, &shader,
                &[(&*wptr, 0), (&*s, 0), (&*inp, in_off), (&*out, out_off)],
                &pc, (wg, 1, 1)).unwrap();
        }
        MvKind::Fp8 { s, per_row } => {
            let (shader, r) = matvec_fp8_variant(n);
            let wg = (n as u32 + r - 1) / r;
            let pc = matvec_fp8_pc(k, n, per_row);
            eng.record_to_off(cb, &shader,
                &[(&*wptr, 0), (&*s, 0), (&*inp, in_off), (&*out, out_off)],
                &pc, (wg, 1, 1)).unwrap();
        }
        MvKind::Plain => {
            let (shader, r) = matvec_variant_core(format, use_subgroup_flag(), true, matvec_rows_override(), n);
            let wg = (n as u32 + r - 1) / r;
            let pc = matvec_pc13(k, n);
            eng.record_to_off(cb, &shader,
                &[(&*wptr, 0), (&*inp, in_off), (&*out, out_off)],
                &pc, (wg, 1, 1)).unwrap();
        }
    }
}

/// LEVER 1 helper — for the TP replicated-KV + column-sharded-Q regime, decide
/// whether this rank's `r_num_q` LOCAL query heads map onto a CONTIGUOUS block
/// of GLOBAL kv heads that the plain-GQA GPU decode kernel
/// (`paged_attn_decode_f32`, `kv_head = q_head / (num_q/num_kv)`) can reproduce
/// exactly. Returns `Some((g_kv_start, local_num_kv))` — the first global kv
/// head this rank touches and how many — when it can (so the caller head-slices
/// the replicated cache to `[vlen, local_num_kv, head_dim]` and dispatches with
/// `num_q=r_num_q, num_kv=local_num_kv`, whose implied ratio `r_num_q/local_num_kv`
/// reproduces the exact `(q_head_offset + qh)/gqa_ratio` mapping); `None`
/// otherwise (→ CPU `cpu_sdpa_gqa` fallback, no accuracy change). For the
/// Gemma-31B TP-4 target this always succeeds: sliding (gqa_ratio 2, r_num_q 8 →
/// local_num_kv 4, g_kv_start rank*4) and global (gqa_ratio 8, r_num_q 8 →
/// local_num_kv 1, g_kv_start rank) both form clean blocks.
fn tp_gpu_kv_block(r_num_q: usize, gqa_ratio: usize, q_head_offset: usize) -> Option<(usize, usize)> {
    if gqa_ratio == 0 || r_num_q == 0 { return None; }
    let first = q_head_offset / gqa_ratio;
    let last = (q_head_offset + r_num_q - 1) / gqa_ratio;
    let local_num_kv = last - first + 1;
    if local_num_kv == 0 || r_num_q % local_num_kv != 0 { return None; }
    let shader_ratio = r_num_q / local_num_kv;
    if shader_ratio == 0 { return None; }
    // Verify the plain-GQA kernel mapping (qh/shader_ratio) reproduces the TP
    // mapping ((q_head_offset+qh)/gqa_ratio - first) for EVERY local q head.
    for qh in 0..r_num_q {
        let want = (q_head_offset + qh) / gqa_ratio - first;
        if qh / shader_ratio != want { return None; }
    }
    Some((first, local_num_kv))
}

const GR_HA:    usize = 0;  // hidden A (layer input / ffn-add output)         [h]
const GR_HB:    usize = 1;  // hidden B (attn-add output / ffn residual)       [h]
const GR_X:     usize = 2;  // input_layernorm output (q/k/v input)            [h]
const GR_Q:     usize = 3;  // q proj → q-norm → rope (in place)              [q_dim]
const GR_K:     usize = 4;  // k proj → k-norm → rope (in place)             [kv_dim]
const GR_V:     usize = 5;  // v proj → v-norm (no-weight, in place)         [kv_dim]
const GR_ATTN:  usize = 6;  // attention output (host sdpa → uploaded)        [q_dim]
const GR_O:     usize = 7;  // o proj output                                   [h]
const GR_ON:    usize = 8;  // post_attention_layernorm(o_proj) output         [h]
const GR_FFIN:  usize = 9;  // pre_feedforward_layernorm output (ffn input)    [h]
const GR_GATE:  usize = 10; // gate proj output                       [ffn_inter]
const GR_UP:    usize = 11; // up proj output                         [ffn_inter]
const GR_GELU:  usize = 12; // gelu(gate) output                      [ffn_inter]
const GR_MID:   usize = 13; // gelu(gate)*up                          [ffn_inter]
const GR_DOWN:  usize = 14; // down proj output                                [h]
const GR_DOWNN: usize = 15; // post_feedforward_layernorm(down) output         [h]
const GR_POS:   usize = 16; // rope position (1 int)
const GR_FF:    usize = 17; // rope freq-factors dummy (1 f32)
const GR_IDX:   usize = 18; // rope set_rows idx dummy (2 u32)
const GR_DUMMY: usize = 19; // harmless binding-3 buf for add_f32_f32_f32       [h]
const GR_PLE_G: usize = 20; // PLE gate output                          [ple_dim]
const GR_PLE_C: usize = 21; // PLE projection contribution                     [h]
const GR_COUNT: usize = 22;

// `layer_core::Slot` is used directly as an index into `gres_bufs`, so the two
// orderings MUST agree. If a slot is ever inserted or reordered here, these fail
// the BUILD rather than silently binding the wrong buffer to every dispatch in
// the layer. (`GR_GELU` is the gemma name for the arch-neutral `Slot::Act`.)
const _: () = {
    assert!(GR_HA == Slot::Ha as usize);
    assert!(GR_HB == Slot::Hb as usize);
    assert!(GR_X == Slot::X as usize);
    assert!(GR_Q == Slot::Q as usize);
    assert!(GR_K == Slot::K as usize);
    assert!(GR_V == Slot::V as usize);
    assert!(GR_ATTN == Slot::Attn as usize);
    assert!(GR_O == Slot::O as usize);
    assert!(GR_ON == Slot::On as usize);
    assert!(GR_FFIN == Slot::Ffin as usize);
    assert!(GR_GATE == Slot::Gate as usize);
    assert!(GR_UP == Slot::Up as usize);
    assert!(GR_GELU == Slot::Act as usize);
    assert!(GR_MID == Slot::Mid as usize);
    assert!(GR_DOWN == Slot::Down as usize);
    assert!(GR_DOWNN == Slot::Downn as usize);
    assert!(GR_POS == Slot::Pos as usize);
    assert!(GR_FF == Slot::Ff as usize);
    assert!(GR_IDX == Slot::Idx as usize);
    assert!(GR_DUMMY == Slot::Dummy as usize);
    assert!(GR_COUNT >= SLOT_COUNT);
};

/// The gemma tensor name for each `layer_core::NormW` role, in `NormW` order.
///
/// The FfnIn role is gemma's `pre_feedforward_layernorm` and qwen's
/// `post_attention_layernorm` — same role in the graph, different tensor. Keeping
/// that mapping beside the ARCH (rather than inside `layer_core`) is what lets one
/// body serve both without a per-arch branch inside the body.
const GEMMA_NORM_NAMES: [&str; NORM_COUNT] = [
    "input_layernorm.weight",            // InputLn
    "self_attn.q_norm.weight",           // QNorm
    "self_attn.k_norm.weight",           // KNorm
    "pre_feedforward_layernorm.weight",  // FfnIn (gemma SANDWICH name)
    "post_attention_layernorm.weight",   // PostAttn
    "post_feedforward_layernorm.weight", // PostFfn
];

/// `layer_core` description of gemma layer `layer_idx`'s ATTENTION FRONT.
///
/// `num_q` is THIS RANK's query-head count: `cfg.num_attention_heads` on the
/// single-node paths, `num_attention_heads / n` under TP. K and V stay replicated
/// at the full per-layer `layer_num_kv_heads`.
///
/// Every per-layer quantity is read off `layer_idx`, never off the model-level
/// config: global (full-attention) layers differ from sliding ones in head dim,
/// KV-head count (value-less MQA(1) vs GQA(8)), RoPE theta (1e6 vs 1e4) and rotary
/// width (PARTIAL head_dim/4 vs full). Reading any of them from the model level
/// mis-shapes half the network — that was one of the five review blockers, and it
/// had to be fixed in each copy separately.
pub(crate) fn gemma_front_params(
    cfg: &model::Gemma4Config, layer_idx: usize, input_norm: bool, num_q: usize,
) -> FrontParams {
    let is_full = cfg.is_full_attention(layer_idx);
    let head_dim = cfg.layer_head_dim(layer_idx);
    let num_kv = cfg.layer_num_kv_heads(layer_idx);
    let is_kv_shared = cfg.is_kv_shared(layer_idx);
    // global → theta 1e6, PARTIAL rotary head_dim/4 (the frequency basis stays the
    // full head_dim); sliding → 1e4, full rotary.
    let (theta, rotary_dim) = if is_full {
        (1_000_000.0f32, head_dim / 4)
    } else {
        (10_000.0f32, head_dim)
    };
    FrontParams {
        input_norm,
        hidden: cfg.hidden_size,
        head_dim,
        num_q,
        num_kv,
        q_dim: num_q * head_dim,
        kv_dim: num_kv * head_dim,
        eps: cfg.rms_norm_eps,
        // Gemma applies the weighted k-norm and the weightless v-norm on exactly
        // the same layers (the non-KV-shared ones).
        k_norm: !is_kv_shared,
        v_norm: !is_kv_shared,
        uses_k_eq_v: cfg.layer_uses_k_eq_v(layer_idx),
        rotary_dim,
        freq_dim: head_dim,
        theta,
    }
}

/// `layer_core` description of gemma layer `layer_idx`'s TAIL. `o_in_dim` and
/// `inter` are THIS RANK's widths (sharded under TP).
pub(crate) fn gemma_tail_params(
    cfg: &model::Gemma4Config, o_in_dim: usize, inter: usize,
) -> TailParams {
    TailParams {
        hidden: cfg.hidden_size,
        o_in_dim,
        inter,
        eps: cfg.rms_norm_eps,
        sandwich: true, // gemma norms each sublayer OUTPUT before the residual add
        act_shader: "gelu_f32",
    }
}

impl VulkanModel {
    /// Host-f32 weight slice for the Gemma model (mirror of `qwen_w`).
    pub(crate) fn gemma_w(&self, name: &str) -> Vec<f32> {
        self.inner.weights.f32_slice(name).to_vec()
    }
    /// Single matvec `[1,n] = x[1,k] @ W[n,k]^T` reading the GEMMA weight store
    /// (`self.inner.weights` / `self.gpu_weights`). Mirror of `qwen_matvec`, used
    /// by `forward_tp_gemma`. f16/quant variant chosen via `matvec_variant`.
    pub(crate) fn gemma_matvec(&mut self, weight_name: &str, x: &[f32], k: usize, n: usize, f16_weight: bool) -> Vec<f32> {
        if let (Some(eng), Some(w_ptr)) = (
            self.engine.as_mut(),
            self.gpu_weights.get(weight_name).map(|w| &w.buffer as *const compute::Buffer),
        ) {
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let out_p = &out as *const compute::Buffer;
            let (shader, r) = matvec_variant(f16_weight, n);
            let wg = (n as u32 + r - 1) / r;
            let pc = matvec_pc13(k, n);
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, &shader, &[&*w_ptr, &*inp_p, &*out_p], &pc, (wg, 1, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            let result = read_f32_buf(&out, n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
            result
        } else {
            let w = self.inner.weights.f32_slice(weight_name);
            model::cpu_matmul(x, w, 1, k, n)
        }
    }
    /// Format-aware single matvec `[1,n] = x[1,k] @ W[n,k]^T` for the TP path
    /// (`forward_tp_gemma`). Unlike `gemma_matvec` (blind 3-binding f16/f32
    /// `record_to`, wrong for NVFP4/FP8/Mlx4 packed+scales weights — same class
    /// of bug as the 12B's earlier RADV segfault), this reads the loaded
    /// `GpuWeight`'s OWN format/aux via `gemma_res_mv_kind` and dispatches
    /// through `record_gemma_mv`, exactly like `forward_pp_gemma`/
    /// `forward_layer_gpu_matmuls`/`gemma_resident_layer` already do. Falls
    /// back to `gemma_matvec`'s f16/CPU path when the weight carries no GPU
    /// buffer (e.g. small replicated PLE/lm_head tables), so those callers are
    /// unaffected.
    pub(crate) fn gemma_tp_matvec(&mut self, weight_name: &str, x: &[f32], k: usize, n: usize, f16_weight: bool) -> Vec<f32> {
        let meta = self.gpu_weights.get(weight_name).map(|w| {
            let (fmt, kind) = gemma_res_mv_kind(w);
            (&w.buffer as *const compute::Buffer, fmt, kind)
        });
        if let (Some(eng), Some((w_ptr, fmt, kind))) = (self.engine.as_mut(), meta) {
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let out_p = &out as *const compute::Buffer;
            let cb = eng.begin_batch().unwrap();
            unsafe {
                record_gemma_mv(eng, cb, w_ptr, fmt, kind, inp_p, out_p, k, n);
            }
            // PROFILE bucket gemma_cb_fence (f): the submit + fence-drain ONLY
            // (NESTED subset of gemma_attn_mv/gemma_mlp_mv — reveals how much of
            // the matvec wall is CB submit/fence vs host alloc/write/readback).
            let _t_fence = Instant::now();
            eng.submit_batch(cb).unwrap();
            prof_add("gemma_cb_fence", _t_fence);
            let result = read_f32_buf(&out, n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
            result
        } else {
            self.gemma_matvec(weight_name, x, k, n, f16_weight)
        }
    }

    /// Row-batched projection/FFN matmul for the Gemma4 batched prefill.
    /// `[t,k] @ W[n,k]^T -> [t,n]`. CRITICAL: `gpu_gemm` is a blind **f16-only**
    /// tiled GEMM (A_TYPE=f16 on binding 0), which silently misinterprets the
    /// g12b/g31b MLX-affine 4-bit / 8-bit / NVFP4 packed weights as f16 and
    /// yields NaN. Until the quantized batched-matmul kernel lands, dispatch each
    /// of the T rows through the FORMAT-AWARE single-row `gemma_tp_matvec` (reads
    /// the weight's own recorded format/aux via `gemma_res_mv_kind`, exactly like
    /// the decode path). Correct for every quant format; the per-row submits are
    /// a correctness-first trade (the batched-quant GEMM is the follow-on perf
    /// lever). For an f16 weight this is bit-identical to the tiled GEMM's math.
    pub(crate) fn gemma_prefill_matmul(&mut self, weight_name: &str, x: &[f32],
                            t: usize, k: usize, n: usize) -> Vec<f32> {
        // VLLM_VULKAN_GEMMA_PREFILL_COLS (default ON — see flags.rs; `=0`
        // reverts to the serial per-row loop below): stream the resident weight
        // ONCE per <=8-column tile through the shared single-stream cols kernels
        // (`mul_mat_vec_{mlx4,q8_0,f16}_cols`) instead of the T per-row matvecs
        // below — the same validated primitive the qwen35 cols prefill uses
        // (`qwen35_matvec_cols_tiled`), reused verbatim here because gemma's
        // mlx4-affine attn + q8_0 MLP weights are the SAME `GpuWeight`
        // format/aux those kernels read (the helper touches only `gpu_weights`
        // /`engine`, nothing qwen-specific). Each output column equals the exact
        // single-column projection of its token, so column-order concatenation
        // rebuilds `[t,n]` identically to the per-row loop up to the cols kernel's
        // f32 reduction order (argmax-exact / cos=1.0 on-node gate). The helper
        // returns None — falling through to the per-row loop, byte-identical — for
        // t<2, Nvfp4/Fp8 or f16-CPU weights (no dequant-cols sibling), or a
        // geometry the cols kernel can't take. Per-layer KV-head/head-dim
        // variation and the value-less global layers need no handling here: the
        // caller passes the already-correct per-layer (k,n) and omits the v_proj
        // call on value-less globals, so this dispatch only ever sees real
        // projections. Independent of (and stacks with) the windowed-ring KV
        // prefill.
        // The column-tiled prefill matvec now lives in a shared
        // `#[cfg(any(feature = "gemma", feature = "qwen35"))]` impl block
        // (qwen35_forward.rs), so this default-ON lever is compiled in for a
        // `--features gemma` build that does NOT enable `qwen35`. It used to
        // sit behind `#[cfg(feature = "qwen35")]` here, which silently
        // disengaged it in exactly that configuration.
        if self.flags.gemma_prefill_cols {
            if let Some(out) = self.qwen35_matvec_cols_tiled(weight_name, x, t, k, n) {
                return out;
            }
        }
        let mut out = vec![0f32; t * n];
        for ti in 0..t {
            let row = self.gemma_tp_matvec(weight_name, &x[ti * k..(ti + 1) * k], k, n, false);
            out[ti * n..(ti + 1) * n].copy_from_slice(&row);
        }
        out
    }

    /// TP-sharded attention sub-block for one Gemma4 layer at one position
    /// (INC-5b piece 2 building block). Column-shards q (this rank's
    /// `r_num_q = num_q/n` heads); k/v stay REPLICATED — each rank holds the
    /// whole per-layer K/V (full `kv_dim = layer_num_kv_heads*layer_head_dim`,
    /// the loader replicates k_proj/v_proj; KV cache is allocated full at
    /// lib.rs KvCache::new). Note num_kv VARIES per layer on g31b (16 sliding /
    /// 4 global), so it MUST come from `layer_num_kv_heads`, not the constant
    /// `cfg.num_key_value_heads` (that stale const was the global-layer OOB
    /// bug). Row-shards o_proj → returns the `[h]` PARTIAL (pre-all-reduce).
    /// Extracted from `forward_tp_gemma`'s per-layer body so the single-token
    /// TP path and the batched T-token verify path (`forward_tp_gemma_verify`)
    /// share ONE implementation — same reasoning as qwen's
    /// `qwen35_gated_attention_tp` being reused by both `forward_tp_qwen35`
    /// and `forward_tp_qwen35_verify_impl`. `x` is the ALREADY input-normed
    /// hidden state at this position.
    pub(crate) fn gemma_attn_tp(
        &mut self,
        cfg: &model::Gemma4Config,
        layer_idx: usize,
        x: &[f32],
        pos: usize,
        n: usize,
    ) -> Vec<f32> {
        // LEVER 2 (VLLM_VULKAN_GEMMA_1CB): fold this rank's attention sub-block
        // into the fused resident-CB dispatch (GPU norms/rope, one QKV submit,
        // one o_proj submit) instead of the per-projection host round-trips
        // below. Falls through to the CPU-orchestrated path when the GPU/weights/
        // buffers aren't ready.
        if self.flags.gemma_1cb {
            if let Some(partial) = self.gemma_attn_tp_1cb(cfg, layer_idx, x, pos, n) {
                return partial;
            }
        }
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let num_q = cfg.num_attention_heads;
        // Per-layer-varying on g31b: 16 KV @ head_dim 256 on sliding layers,
        // 4 KV @ head_dim 512 on the period-6 global layers. Using the constant
        // `cfg.num_key_value_heads` (16) here made kv_dim=16*512=8192 on global
        // layers whose k_proj/v_proj only have 4*512=2048 rows → the matvec
        // dispatched n=8192 read 4x past the buffer (robustBufferAccess→0),
        // manufacturing 12 phantom-zero KV heads → corrupted global-layer attn.
        let num_kv = cfg.layer_num_kv_heads(layer_idx);
        let r_num_q = num_q / n;
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let r_q_dim = r_num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let is_kv_shared = cfg.is_kv_shared(layer_idx);
        // Value-less global attention (g31b period-6 full-attention layers):
        // there is NO v_proj tensor on disk — V derives from the RAW (pre-k_norm)
        // k_proj output via weightless rms-norm. Indexing v_proj unconditionally
        // panics ("model.layers.N.self_attn.v_proj.weight not found"). Mirrors
        // cfg.layer_uses_k_eq_v / forward_layer_gpu_matmuls (gemma_forward.rs
        // ~L530,660) / the CPU reference (model.rs ~L1011).
        let uses_k_eq_v = cfg.layer_uses_k_eq_v(layer_idx);
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

        // q: column-sharded (r_q_dim rows); k/v: replicated (full kv_dim).
        // PROFILE bucket gemma_attn_mv (a): q/k/v FP8 projection matvecs
        // (submit+fence+readback per proj).
        let _t_mv = Instant::now();
        let mut q = self.gemma_tp_matvec(&ln("self_attn.q_proj.weight"), x, h, r_q_dim, true);
        let mut k_final = self.gemma_tp_matvec(&ln("self_attn.k_proj.weight"), x, h, kv_dim, true);
        // Value-less global layers: V = raw (pre-k_norm) K, cloned BEFORE the
        // k_norm loop below mutates k_final in place. Mirrors the CPU ref
        // (model.rs ~L1011: `v_raw = k_raw.clone()`) and forward_layer_gpu_matmuls
        // (gemma_forward.rs ~L660). Sliding/non-k_eq_v layers keep the v_proj matvec.
        let mut v_final = if uses_k_eq_v {
            k_final.clone()
        } else {
            self.gemma_tp_matvec(&ln("self_attn.v_proj.weight"), x, h, kv_dim, true)
        };
        prof_add("gemma_attn_mv", _t_mv);

        // Q-norm (per head_dim, on THIS rank's heads); K/V-norm only when the
        // layer computes its own KV (not kv_shared), matching single-node.
        // PROFILE bucket gemma_host (d): CPU q/k/v-norm + RoPE.
        let _t_host = Instant::now();
        let q_norm_w = self.gemma_w(&ln("self_attn.q_norm.weight"));
        for hi in 0..r_num_q {
            let s = &mut q[hi * head_dim..(hi + 1) * head_dim];
            let nn = model::cpu_rms_norm(s, &q_norm_w, eps);
            s.copy_from_slice(&nn);
        }
        if !is_kv_shared {
            let k_norm_w = self.gemma_w(&ln("self_attn.k_norm.weight"));
            for hi in 0..num_kv {
                let s = &mut k_final[hi * head_dim..(hi + 1) * head_dim];
                let nn = model::cpu_rms_norm(s, &k_norm_w, eps);
                s.copy_from_slice(&nn);
            }
            for hi in 0..num_kv {
                let s = &mut v_final[hi * head_dim..(hi + 1) * head_dim];
                let nn = model::cpu_rms_norm_no_weight(s, head_dim, eps);
                s.copy_from_slice(&nn);
            }
        }

        // RoPE (per-layer-type theta/rotary_dim) — over this rank's q heads
        // and the (replicated) kv head. Position-only, so per-rank safe.
        let (theta, rotary_dim) = if is_full {
            (1_000_000.0f32, head_dim / 4)
        } else {
            (10_000.0f32, head_dim)
        };
        model::cpu_rope_with_basis(&mut q, &mut k_final, pos, r_num_q, num_kv, head_dim, rotary_dim, head_dim, theta);
        prof_add("gemma_host", _t_host);

        // KV cache: each rank holds the (replicated) full kv head for every
        // layer it computes. KV-shared layers read the target layer's cache,
        // which is already local on this rank → no cross-rank gather.
        let target_cache_idx = if is_kv_shared {
            self.inner.kv_shared_target(layer_idx)
        } else {
            self.inner.kv_caches[layer_idx].append(&k_final, &v_final);
            layer_idx
        };
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        // Replicated KV + column-sharded Q: this rank owns local q heads
        // [tp_rank*r_num_q, +r_num_q) of the GLOBAL `num_q`. Map each local q
        // head to its GLOBAL kv head via the global GQA ratio (num_q/num_kv)
        // and a tp_rank*r_num_q offset so rank r>0 attends the correct KV
        // heads — and so the ratio never degenerates to 0 (r_num_q<num_kv at
        // TP≥2, e.g. TP-4 sliding 8/16). At TP-1 this reduces to plain cpu_sdpa.
        let gqa_ratio = num_q / num_kv;
        let q_head_offset = self.tp_rank * r_num_q;
        let attn_out = self.gemma_tp_sdpa(
            &q, target_cache_idx, r_num_q, num_kv, head_dim, window, gqa_ratio, q_head_offset,
        );

        // o_proj: row-sharded → partial sum over this rank's q_dim slice.
        let _t_o = Instant::now();
        let o_partial = self.gemma_tp_matvec(&ln("self_attn.o_proj.weight"), &attn_out, r_q_dim, h, true);
        prof_add("gemma_attn_mv", _t_o);
        o_partial
    }

    /// TP-sharded MLP sub-block for one Gemma4 layer (INC-5b piece 2 building
    /// block, mirrors `gemma_attn_tp`). Column-shards gate/up
    /// (`r_inter = ffn_inter/n` wide each); row-shards down_proj → returns the
    /// `[h]` PARTIAL (pre-all-reduce). `x` is the already pre-ffn-normed input.
    pub(crate) fn gemma_mlp_tp(
        &mut self,
        cfg: &model::Gemma4Config,
        layer_idx: usize,
        x: &[f32],
        n: usize,
    ) -> Vec<f32> {
        // LEVER 2 (VLLM_VULKAN_GEMMA_1CB): fuse gate+up+GELU+mul+down into one
        // resident-CB submit (mirrors gemma_resident_layer's FFN), replacing the
        // three separate host round-trips + CPU gelu/mul below.
        if self.flags.gemma_1cb {
            if let Some(partial) = self.gemma_mlp_tp_1cb(cfg, layer_idx, x, n) {
                return partial;
            }
        }
        let h = cfg.hidden_size;
        let ffn_inter = cfg.layer_intermediate_size(layer_idx);
        let r_inter = ffn_inter / n;
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
        // PROFILE bucket gemma_mlp_mv (c): gate/up NVFP4 matvecs.
        let _t_gu = Instant::now();
        let gate = self.gemma_tp_matvec(&ln("mlp.gate_proj.weight"), x, h, r_inter, true);
        let up   = self.gemma_tp_matvec(&ln("mlp.up_proj.weight"),   x, h, r_inter, true);
        prof_add("gemma_mlp_mv", _t_gu);
        // gemma_host (d): CPU GELU + elementwise mul glue.
        let _t_g = Instant::now();
        let gate_act = model::cpu_gelu(&gate);
        let mid: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
        prof_add("gemma_host", _t_g);
        // gemma_mlp_mv (c): down NVFP4 matvec.
        let _t_d = Instant::now();
        let down = self.gemma_tp_matvec(&ln("mlp.down_proj.weight"), &mid, r_inter, h, true);
        prof_add("gemma_mlp_mv", _t_d);
        down
    }

    /// LEVER 1 — shared TP SDPA: run this rank's decode attention either on the
    /// GPU decode kernel (VLLM_VULKAN_GPU_SDPA, when this rank's local q heads
    /// map to a clean contiguous kv-head block — see `tp_gpu_kv_block`) or on the
    /// host `cpu_sdpa_gqa` reference. `q` is this rank's `r_num_q*head_dim`
    /// post-norm/rope query; `num_kv` is the FULL replicated kv-head count.
    /// The GPU path head-slices the replicated cache to `[vlen, local_num_kv,
    /// head_dim]` (window + head block) so the plain-GQA kernel reproduces the
    /// TP `(q_head_offset+qh)/gqa_ratio` mapping exactly. GPU online-softmax vs
    /// the host single-accumulator dot → cos≈1.0, argmax-exact gate (not
    /// bit-exact). Used by BOTH the base `gemma_attn_tp` and the 1cb path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemma_tp_sdpa(
        &mut self,
        q: &[f32],
        target_cache_idx: usize,
        r_num_q: usize,
        num_kv: usize,
        head_dim: usize,
        window: Option<usize>,
        gqa_ratio: usize,
        q_head_offset: usize,
    ) -> Vec<f32> {
        // PROFILE bucket gemma_sdpa (e): SDPA compute (gather+upload+kernel on the
        // GPU path, or the host cpu_sdpa_gqa reduction).
        let _t_sdpa = Instant::now();
        let use_gpu_sdpa = self.engine.is_some() && self.flags.gpu_sdpa;
        if use_gpu_sdpa {
            if let Some((g_kv_start, local_num_kv)) =
                tp_gpu_kv_block(r_num_q, gqa_ratio, q_head_offset)
            {
                let (k_slice, v_slice, vlen) = {
                    let cache = &self.inner.kv_caches[target_cache_idx];
                    // full replicated per-position stride ([pos, num_kv, head_dim]).
                    let stride = num_kv * head_dim;
                    // Per-layer-sized KV: a sliding layer's cache may be a
                    // `window`-sized ring. `windowed_view` compacts the last
                    // `window` positions into ascending-absolute order (index 0
                    // == absolute `seq_len-window`), which is byte-identical to
                    // the old `k_up_to_now()` slice `[kv_start..slen)` for an
                    // unwrapped cache and correct once the ring wraps. Full
                    // layers keep the zero-copy absolute slice (`window=None`).
                    let (kbuf, vbuf, vlen) = cache.sdpa_view(window);
                    let mut ks = Vec::with_capacity(vlen * local_num_kv * head_dim);
                    let mut vs = Vec::with_capacity(vlen * local_num_kv * head_dim);
                    for pos_i in 0..vlen {
                        let base = pos_i * stride + g_kv_start * head_dim;
                        for lh in 0..local_num_kv {
                            let o = base + lh * head_dim;
                            ks.extend_from_slice(&kbuf[o..o + head_dim]);
                            vs.extend_from_slice(&vbuf[o..o + head_dim]);
                        }
                    }
                    (ks, vs, vlen)
                };
                let out = self.gpu_sdpa(q, &k_slice, &v_slice, r_num_q, local_num_kv, head_dim, vlen, 1.0);
                prof_add("gemma_sdpa", _t_sdpa);
                return out;
            }
        }
        let out = {
            let cache = &self.inner.kv_caches[target_cache_idx];
            // Sliding layers read the ring via `windowed_view` (+ `None`), which
            // is bit-for-bit identical to attending the full absolute cache with
            // `Some(window)` (see `KvCache::windowed_view`); full layers keep the
            // zero-copy absolute path.
            match window {
                Some(w) => {
                    let (kw, vw, vlen) = cache.windowed_view(w);
                    model::cpu_sdpa_gqa(
                        q, &kw, &vw,
                        r_num_q, num_kv, head_dim, vlen, 1.0, None,
                        gqa_ratio, q_head_offset,
                    )
                }
                None => model::cpu_sdpa_gqa(
                    q, cache.k_up_to_now(), cache.v_up_to_now(),
                    r_num_q, num_kv, head_dim, cache.seq_len, 1.0, None,
                    gqa_ratio, q_head_offset,
                ),
            }
        };
        prof_add("gemma_sdpa", _t_sdpa);
        out
    }

    /// Are the fused-CB resident buffers + norm weights ready for the Gemma TP
    /// 1cb path (Lever 2)? Mirrors `forward_gemma_gpu_resident`'s `ready` gate.
    pub(crate) fn gemma_1cb_ready(&mut self, layer_idx: usize) -> bool {
        self.engine.is_some()
            && self.gpu_weights.contains_key(&format!("model.layers.{layer_idx}.self_attn.q_proj.weight"))
            && self.init_gres_bufs()
            && self.ensure_gemma_norm_weights()
    }

    /// LEVER 2 — TP attention sub-block via the fused resident-CB dispatch. `x`
    /// is the ALREADY input-normed hidden (same contract as `gemma_attn_tp`).
    /// Returns `Some(partial)` (the row-sharded o_proj partial, pre-all-reduce)
    /// when the GPU path ran, or `None` to signal the caller to use the base
    /// CPU-orchestrated path. Mirrors `gemma_resident_layer`'s CB1 exactly
    /// (input_layernorm is kept on the caller/CPU so `x` stays bit-identical;
    /// this path folds only q/k/v-norm + RoPE onto the GPU), sharded to this
    /// rank's `r_num_q` query heads. K/V stay replicated (full `kv_dim`).
    fn gemma_attn_tp_1cb(
        &mut self,
        cfg: &model::Gemma4Config,
        layer_idx: usize,
        x: &[f32],
        pos: usize,
        n: usize,
    ) -> Option<Vec<f32>> {
        if !self.gemma_1cb_ready(layer_idx) { return None; }
        let h = cfg.hidden_size;
        let num_q = cfg.num_attention_heads;
        let num_kv = cfg.layer_num_kv_heads(layer_idx);
        let r_num_q = num_q / n;
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let r_q_dim = r_num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let is_kv_shared = cfg.is_kv_shared(layer_idx);
        let uses_k_eq_v = cfg.layer_uses_k_eq_v(layer_idx);

        // Persistent buffers (reused from the single-node resident pool — the TP
        // path never runs concurrently with forward_gemma_gpu_resident, and the
        // GR_* slots are sized for the MAX (full) head geometry, so the sharded
        // r_q_dim/kv_dim always fit).
        //
        // `input_norm: false` — this path's contract is that the CALLER already ran
        // input_layernorm on the CPU so `x` stays bit-identical with
        // `gemma_attn_tp`; only q/k/v-norm + RoPE fold onto the GPU here.
        let front = gemma_front_params(cfg, layer_idx, false, r_num_q);
        let front_norms = Self::gemma_front_norms(cfg, layer_idx, false);
        let front_ptrs = self.gemma_layer_ptrs(layer_idx, uses_k_eq_v, &front_norms, &FRONT_PROJS);
        let qp = self.gres_ptr(GR_Q);
        let kp = self.gres_ptr(GR_K);
        let vp = self.gres_ptr(GR_V);

        // Write x (input-normed hidden) + this token's position into resident bufs.
        unsafe {
            (*self.gres_ptr_mut(GR_X)).write(&f32_slice_to_bytes(x)).unwrap();
            (*self.gres_ptr_mut(GR_POS)).write(&(pos as i32).to_le_bytes()).unwrap();
        }

        // ── CB1: q/k/v proj → q/k/v-norm → RoPE (one submit) ─────────────────
        // PROFILE bucket gemma_attn_mv (b): the fused QKV+norm+rope CB (1cb path).
        let _t_mv = Instant::now();
        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            let mut rec = GpuRecorder { eng, cb, p: &front_ptrs };
            layer_core::record_front(&mut rec, &front).unwrap();
        }
        let _t_fence = Instant::now();
        eng.submit_batch(cb).unwrap();
        prof_add("gemma_cb_fence", _t_fence);
        prof_add("gemma_attn_mv", _t_mv);

        // ── Host boundary: KV cache append (post-norm/rope k/v) ──────────────
        let q_host = read_f32_buf(unsafe { &*qp }, r_q_dim);
        let k_host = read_f32_buf(unsafe { &*kp }, kv_dim);
        let v_host = read_f32_buf(unsafe { &*vp }, kv_dim);
        let target_cache_idx = if is_kv_shared {
            self.inner.kv_shared_target(layer_idx)
        } else {
            self.inner.kv_caches[layer_idx].append(&k_host, &v_host);
            layer_idx
        };
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        let gqa_ratio = num_q / num_kv;
        let q_head_offset = self.tp_rank * r_num_q;
        let attn_out = self.gemma_tp_sdpa(
            &q_host, target_cache_idx, r_num_q, num_kv, head_dim, window, gqa_ratio, q_head_offset,
        );

        // ── CB2: o_proj (row-sharded) → partial ──────────────────────────────
        let op = self.gres_ptr(GR_O);
        let o_ptrs = self.gemma_layer_ptrs(layer_idx, uses_k_eq_v, &[], &[ProjW::O]);
        let o_tail = gemma_tail_params(cfg, r_q_dim, cfg.layer_intermediate_size(layer_idx));
        let _t_o = Instant::now();
        unsafe { (*self.gres_ptr_mut(GR_ATTN)).write(&f32_slice_to_bytes(&attn_out)).unwrap(); }
        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            let mut rec = GpuRecorder { eng, cb, p: &o_ptrs };
            layer_core::record_o_proj(&mut rec, &o_tail).unwrap();
        }
        let _t_fence_o = Instant::now();
        eng.submit_batch(cb).unwrap();
        prof_add("gemma_cb_fence", _t_fence_o);
        let out = read_f32_buf(unsafe { &*op }, h);
        prof_add("gemma_attn_mv", _t_o);
        Some(out)
    }

    /// LEVER 2 — TP MLP sub-block via one fused resident-CB submit (gate+up→
    /// GELU→mul→down), the analog of `gemma_resident_layer`'s FFN sequence,
    /// sharded to this rank's `r_inter`. `x` is the already pre-ffn-normed input
    /// (same contract as `gemma_mlp_tp`). Returns `Some(down partial)` or `None`
    /// to fall back to the base path.
    fn gemma_mlp_tp_1cb(
        &mut self,
        cfg: &model::Gemma4Config,
        layer_idx: usize,
        x: &[f32],
        n: usize,
    ) -> Option<Vec<f32>> {
        if !self.gemma_1cb_ready(layer_idx) { return None; }
        let h = cfg.hidden_size;
        let ffn_inter = cfg.layer_intermediate_size(layer_idx);
        let r_inter = ffn_inter / n;
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
        if !self.gpu_weights.contains_key(&ln("mlp.gate_proj.weight"))
            || !self.gpu_weights.contains_key(&ln("mlp.down_proj.weight")) {
            return None;
        }

        let down = self.gres_ptr(GR_DOWN);
        // The FFN BODY alone: no norms and no residual adds — under TP those live
        // on the caller's side of the all-reduce.
        let ffn_ptrs = self.gemma_layer_ptrs(
            layer_idx, cfg.layer_uses_k_eq_v(layer_idx), &[],
            &[ProjW::Gate, ProjW::Up, ProjW::Down]);
        let tail = gemma_tail_params(
            cfg, cfg.num_attention_heads * cfg.layer_head_dim(layer_idx), r_inter);

        // PROFILE bucket gemma_mlp_mv (c): the fused gate+up+gelu+mul+down CB (1cb).
        let _t_mv = Instant::now();
        unsafe { (*self.gres_ptr_mut(GR_FFIN)).write(&f32_slice_to_bytes(x)).unwrap(); }
        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            let mut rec = GpuRecorder { eng, cb, p: &ffn_ptrs };
            layer_core::record_ffn_body(&mut rec, &tail).unwrap();
        }
        let _t_fence = Instant::now();
        eng.submit_batch(cb).unwrap();
        prof_add("gemma_cb_fence", _t_fence);
        let out = read_f32_buf(unsafe { &*down }, h);
        prof_add("gemma_mlp_mv", _t_mv);
        Some(out)
    }

    /// LEVER 1 — seed the resident hidden (GR_HA) from a host `[h]` vector at the
    /// layer boundary (initial embedding, and after each layer's host
    /// PLE/layer_scalar tail). Encapsulates the GR_HA slot for the lib.rs
    /// orchestration branch.
    pub(crate) fn gemma_full_seed_hidden(&mut self, hidden: &[f32]) {
        unsafe { (*self.gres_ptr_mut(GR_HA)).write(&f32_slice_to_bytes(hidden)).unwrap(); }
    }
    /// LEVER 1 — write the all-reduced o_proj `[h]` back into GR_O for
    /// `gemma_mlp_tp_full`'s post_attn_norm.
    pub(crate) fn gemma_full_set_oproj(&mut self, o_proj: &[f32]) {
        unsafe { (*self.gres_ptr_mut(GR_O)).write(&f32_slice_to_bytes(o_proj)).unwrap(); }
    }
    /// LEVER 1 — write the all-reduced down output `[h]` back into GR_DOWN for
    /// `gemma_layer_tail_full`'s post_ffn_norm.
    pub(crate) fn gemma_full_set_ffout(&mut self, ff_out: &[f32]) {
        unsafe { (*self.gres_ptr_mut(GR_DOWN)).write(&f32_slice_to_bytes(ff_out)).unwrap(); }
    }

    /// LEVER 1 (VLLM_VULKAN_GEMMA_1CB_FULL) — attention FRONT of the fully-fused
    /// TP layer. Unlike `gemma_attn_tp_1cb` (which takes a host-normed `x` and
    /// leaves input_layernorm on the caller's CPU), this reads the layer-input
    /// hidden from the GPU-resident GR_HA and folds `input_layernorm` onto the
    /// GPU as the first dispatch of CB-A — so NO host norm runs and GR_HA stays
    /// the intact residual for `gemma_mlp_tp_full`'s residual add. CB-A does
    /// input_layernorm(HA→X) → q/k/v(X) → q/k/v-norm → RoPE; then the host KV
    /// append + SDPA; then CB-B does the row-sharded o_proj → returns the `[h]`
    /// PARTIAL (pre-all-reduce). Mirrors `gemma_resident_layer`'s CB1 + o_proj,
    /// sharded to this rank's `r_num_q` query heads (K/V replicated). Caller
    /// guarantees readiness (`gemma_1cb_ready`) + that GR_HA already holds this
    /// layer's input hidden.
    pub(crate) fn gemma_attn_tp_full_front(
        &mut self,
        cfg: &model::Gemma4Config,
        layer_idx: usize,
        pos: usize,
        n: usize,
    ) -> Vec<f32> {
        let h = cfg.hidden_size;
        let num_q = cfg.num_attention_heads;
        let num_kv = cfg.layer_num_kv_heads(layer_idx);
        let r_num_q = num_q / n;
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let r_q_dim = r_num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let is_kv_shared = cfg.is_kv_shared(layer_idx);
        let uses_k_eq_v = cfg.layer_uses_k_eq_v(layer_idx);

        // `input_norm: true` — unlike `gemma_attn_tp_1cb`, this reads the intact
        // layer-input hidden from GR_HA and folds input_layernorm onto the GPU as
        // CB-A's first dispatch, so GR_HA stays the residual for the middle piece.
        let front = gemma_front_params(cfg, layer_idx, true, r_num_q);
        let front_norms = Self::gemma_front_norms(cfg, layer_idx, true);
        let front_ptrs = self.gemma_layer_ptrs(layer_idx, uses_k_eq_v, &front_norms, &FRONT_PROJS);
        let qp = self.gres_ptr(GR_Q);
        let kp = self.gres_ptr(GR_K);
        let vp = self.gres_ptr(GR_V);

        // This token's position (hidden is already resident in GR_HA).
        unsafe { (*self.gres_ptr_mut(GR_POS)).write(&(pos as i32).to_le_bytes()).unwrap(); }

        // ── CB-A: input_layernorm → q/k/v proj → q/k/v-norm → RoPE ───────────
        let _t_mv = Instant::now();
        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            let mut rec = GpuRecorder { eng, cb, p: &front_ptrs };
            layer_core::record_front(&mut rec, &front).unwrap();
        }
        let _t_fence = Instant::now();
        eng.submit_batch(cb).unwrap();
        prof_add("gemma_cb_fence", _t_fence);
        prof_add("gemma_attn_mv", _t_mv);

        // ── Host boundary: KV cache append (post-norm/rope k/v) + SDPA ────────
        let q_host = read_f32_buf(unsafe { &*qp }, r_q_dim);
        let k_host = read_f32_buf(unsafe { &*kp }, kv_dim);
        let v_host = read_f32_buf(unsafe { &*vp }, kv_dim);
        let target_cache_idx = if is_kv_shared {
            self.inner.kv_shared_target(layer_idx)
        } else {
            self.inner.kv_caches[layer_idx].append(&k_host, &v_host);
            layer_idx
        };
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        let gqa_ratio = num_q / num_kv;
        let q_head_offset = self.tp_rank * r_num_q;
        let attn_out = self.gemma_tp_sdpa(
            &q_host, target_cache_idx, r_num_q, num_kv, head_dim, window, gqa_ratio, q_head_offset,
        );

        // ── CB-B: o_proj (row-sharded) → partial ─────────────────────────────
        let op = self.gres_ptr(GR_O);
        let o_ptrs = self.gemma_layer_ptrs(layer_idx, uses_k_eq_v, &[], &[ProjW::O]);
        let o_tail = gemma_tail_params(cfg, r_q_dim, cfg.layer_intermediate_size(layer_idx));
        let _t_o = Instant::now();
        unsafe { (*self.gres_ptr_mut(GR_ATTN)).write(&f32_slice_to_bytes(&attn_out)).unwrap(); }
        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            let mut rec = GpuRecorder { eng, cb, p: &o_ptrs };
            layer_core::record_o_proj(&mut rec, &o_tail).unwrap();
        }
        let _t_fence_o = Instant::now();
        eng.submit_batch(cb).unwrap();
        prof_add("gemma_cb_fence", _t_fence_o);
        let out = read_f32_buf(unsafe { &*op }, h);
        prof_add("gemma_attn_mv", _t_o);
        out
    }

    /// LEVER 1 — the fused MIDDLE of the fully-fused TP layer: post_attn_norm +
    /// residual#1 + pre_ffn_norm + gate/up + GELU/mul + down, ONE submit. Reads
    /// the all-reduced o_proj (written by the caller into GR_O) and the intact
    /// layer-input residual from GR_HA; writes the post-attn residual to GR_HB
    /// (kept for `gemma_layer_tail_full`). Returns the `[h]` down PARTIAL
    /// (pre-all-reduce). Mirrors the head of `gemma_resident_layer`'s CB2 up to
    /// the down_proj, replacing the CPU post_attn_norm/residual/pre_ffn_norm.
    pub(crate) fn gemma_mlp_tp_full(
        &mut self,
        cfg: &model::Gemma4Config,
        layer_idx: usize,
        n: usize,
    ) -> Vec<f32> {
        let h = cfg.hidden_size;
        let ffn_inter = cfg.layer_intermediate_size(layer_idx);
        let r_inter = ffn_inter / n;

        let down = self.gres_ptr(GR_DOWN);
        // Tail pieces 2, 3a and 3b — post_attn_norm + residual#1 + pre_ffn_norm +
        // the FFN body — stopping BEFORE post_ffn_norm/residual#2, which wait on
        // the caller's all-reduce and land in `gemma_layer_tail_full`.
        let ptrs = self.gemma_layer_ptrs(
            layer_idx, cfg.layer_uses_k_eq_v(layer_idx),
            &[NormW::PostAttn, NormW::FfnIn],
            &[ProjW::Gate, ProjW::Up, ProjW::Down]);
        let tail = gemma_tail_params(
            cfg, cfg.num_attention_heads * cfg.layer_head_dim(layer_idx), r_inter);

        let _t_mv = Instant::now();
        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            let mut rec = GpuRecorder { eng, cb, p: &ptrs };
            layer_core::record_attn_residual(&mut rec, &tail).unwrap();
            rec.step(layer_core::Step::Barrier).unwrap();
            layer_core::record_ffn_in_norm(&mut rec, &tail).unwrap();
            rec.step(layer_core::Step::Barrier).unwrap();
            layer_core::record_ffn_body(&mut rec, &tail).unwrap();
        }
        let _t_fence = Instant::now();
        eng.submit_batch(cb).unwrap();
        prof_add("gemma_cb_fence", _t_fence);
        let out = read_f32_buf(unsafe { &*down }, h);
        prof_add("gemma_mlp_mv", _t_mv);
        out
    }

    /// LEVER 1 — the fused TAIL of the fully-fused TP layer: post_ffn_norm +
    /// residual#2, ONE submit. Reads the all-reduced down output (written by the
    /// caller into GR_DOWN) and the post-attn residual from GR_HB; writes the
    /// pre-scalar hidden to GR_HA and returns it on the host so the caller's
    /// `gemma_ple_add_tp` applies PLE (E2B only) + layer_scalar — exactly like
    /// `gemma_resident_layer`'s host PLE/scalar tail. The returned Vec is the
    /// layer output BEFORE layer_scalar; the caller writes the scaled result
    /// back into GR_HA for the next layer's front.
    pub(crate) fn gemma_layer_tail_full(&mut self, cfg: &model::Gemma4Config, layer_idx: usize) -> Vec<f32> {
        let h = cfg.hidden_size;
        let ha = self.gres_ptr(GR_HA);
        // Tail piece 4 alone: post_ffn_norm + residual#2.
        let ptrs = self.gemma_layer_ptrs(
            layer_idx, cfg.layer_uses_k_eq_v(layer_idx), &[NormW::PostFfn], &[]);
        let tail = gemma_tail_params(
            cfg, cfg.num_attention_heads * cfg.layer_head_dim(layer_idx),
            cfg.layer_intermediate_size(layer_idx));

        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            let mut rec = GpuRecorder { eng, cb, p: &ptrs };
            layer_core::record_ffn_residual(&mut rec, &tail).unwrap();
        }
        let _t_fence = Instant::now();
        eng.submit_batch(cb).unwrap();
        prof_add("gemma_cb_fence", _t_fence);
        read_f32_buf(unsafe { &*ha }, h)
    }

    /// PLE contribution + layer_scalar (INC-5b piece 2 building block,
    /// replicated — identical per TP rank, no all-reduce). Mirrors the tail of
    /// `forward_tp_gemma`'s per-layer body. The PLE contribution
    /// (per_layer_input_gate / per_layer_projection / post_per_layer_input_norm)
    /// only exists on has_ple() checkpoints (E2B) — g31b/g12b carry no
    /// per_layer* tensors at all, so fetching them by name panics
    /// ("Weight not found"). Gated exactly like `forward_layer_gpu_matmuls`
    /// (gemma_forward.rs ~L861): skip the whole contribution block for
    /// !has_ple(), but the layer_scalar multiply still runs unconditionally
    /// (present on every layer regardless of PLE). Mutates `hidden3` in place.
    pub(crate) fn gemma_ple_add_tp(
        &mut self,
        cfg: &model::Gemma4Config,
        layer_idx: usize,
        hidden3: &mut Vec<f32>,
        layer_ple: &[f32],
    ) {
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let ple_dim = cfg.hidden_size_per_layer_input;
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
        if cfg.has_ple() {
            let gate_ple = self.gemma_matvec(&ln("per_layer_input_gate.weight"), hidden3, h, ple_dim, true);
            let gate_ple_act = model::cpu_gelu(&gate_ple);
            let gated: Vec<f32> = gate_ple_act.iter().zip(layer_ple.iter())
                .map(|(&g, &p)| g * p).collect();
            let contrib = self.gemma_matvec(&ln("per_layer_projection.weight"), &gated, ple_dim, h, true);
            let ple_norm_w = self.gemma_w(&ln("post_per_layer_input_norm.weight"));
            let contrib_normed = model::cpu_rms_norm(&contrib, &ple_norm_w, eps);
            hidden3.iter_mut().zip(contrib_normed.iter()).for_each(|(hv, &c)| *hv += c);
        }
        let layer_scalar = self.gemma_w(&ln("layer_scalar"))[0];
        hidden3.iter_mut().for_each(|v| *v *= layer_scalar);
    }

    /// Small ring cap for `gemma_hidden_ring` (INC-5b piece 3) — only the
    /// last few committed-frontier positions are ever looked up by the
    /// drafter's step-0 seed, mirrors qwen's `PRENORM_RING_CAP`.
    const GEMMA_HIDDEN_RING_CAP: usize = 8;

    /// Record the pre-`model.norm` hidden at absolute position `pos` into the
    /// position-keyed `gemma_hidden_ring` (INC-5b piece 3: the EAGLE drafter's
    /// step-0 `recurrent_hidden` seed — `GEMMA31B_SPEC_PLAN.md` §1.1). Called
    /// once per position from `forward_tp_gemma` (single-token) and
    /// `forward_tp_gemma_verify` (once per verified position) — mirrors
    /// qwen's `stash_verify_prenorm`.
    pub(crate) fn stash_gemma_hidden(&mut self, pos: usize, hidden: &[f32]) {
        self.gemma_hidden_ring.push_back((pos, hidden.to_vec()));
        while self.gemma_hidden_ring.len() > Self::GEMMA_HIDDEN_RING_CAP {
            self.gemma_hidden_ring.pop_front();
        }
    }

    /// Look up the pre-`model.norm` hidden stashed at absolute position `pos`
    /// (INC-5b piece 3). Returns `None` if it fell off the ring or was never
    /// stashed (e.g. no verify/decode call has run yet at that position).
    pub(crate) fn gemma_hidden_at(&self, pos: usize) -> Option<Vec<f32>> {
        self.gemma_hidden_ring.iter().rev()
            .find(|(p, _)| *p == pos)
            .map(|(_, h)| h.clone())
    }

    /// GPU-accelerated forward pass: matmuls on GPU, norms + attention on CPU.
    pub(crate) fn forward_gpu(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let cfg = self.inner.config.clone();
        let ple_dim = cfg.hidden_size_per_layer_input;

        let (mut hidden, ple_inputs) = self.gemma_embed_and_ple(token_id);

        // Decoder layers (honors pp_start..pp_end; all layers for single-node).
        for layer_idx in self.pp_start..self.pp_end {
            let layer_ple = &ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim];
            hidden = self.forward_layer_gpu_matmuls(layer_idx, &hidden, position, layer_ple);
        }
        self.gemma_final(&hidden)
    }
    /// Embedding + PLE preprocessing — runs on the PLE-owning (first) stage.
    /// Returns (hidden, ple_inputs) covering ALL layers; in PP the ple_inputs
    /// are forwarded downstream so later stages need no PLE weights.
    pub(crate) fn gemma_embed_and_ple(&mut self, token_id: u32) -> (Vec<f32>, Vec<f32>) {
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        // g12b (gemma4_unified loader): the tied embed table is kept host-side
        // as f16 (~2GB vs ~4GB f32; the biggest full-48L single-node OOM
        // driver) in the generic `q35_f16_host` table (mirrors qwen3.6's
        // embed/lm_head f16-host convention) — widen just this token's row
        // back to f32. Falls back to the f32 store when unset (E2B / test
        // constructions that never populate q35_f16_host).
        let hidden: Vec<f32> = if let Some(f16v) = self.q35_f16_host.get("model.embed_tokens.weight") {
            f16v[token_id as usize * h..(token_id as usize + 1) * h]
                .iter().map(|&b| half::f16::from_bits(b).to_f32() * cfg.embed_scale).collect()
        } else {
            let embed_w = self.inner.weights.f32_slice("model.embed_tokens.weight");
            embed_w[token_id as usize * h..(token_id as usize + 1) * h]
                .iter().map(|&v| v * cfg.embed_scale).collect()
        };
        // g12b: hidden_size_per_layer_input=0, no per-layer-embedding table
        // (`gemma_ple_bf16` stays None) — skip the PLE preprocessing entirely
        // rather than unwrap a table that was never loaded.
        if !cfg.has_ple() {
            return (hidden, Vec::new());
        }
        let total_ple = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input;
        // bf16-resident PLE table: convert just this token's row to f32.
        let ple_bits = self.gemma_ple_bf16.as_ref()
            .expect("gemma_ple_bf16 must be loaded on the PLE-owning stage");
        let row0 = token_id as usize * total_ple;
        let ple_embeds_flat: Vec<f32> = ple_bits[row0..row0 + total_ple]
            .iter().map(|&b| half::bf16::from_bits(b).to_f32() * cfg.ple_scale).collect();
        let proj_w = self.inner.weights.f32_slice("model.per_layer_model_projection.weight");
        let ple_proj = model::cpu_matmul(&hidden, proj_w, 1, h, total_ple);
        let ple_proj: Vec<f32> = ple_proj.iter().map(|&v| v * cfg.per_layer_projection_scale).collect();
        let pn_w = self.inner.weights.f32_slice("model.per_layer_projection_norm.weight");
        let ple_proj_normed = model::cpu_rms_norm(&ple_proj, pn_w, eps);
        let ple_inputs: Vec<f32> = ple_proj_normed.iter().zip(ple_embeds_flat.iter())
            .map(|(&p, &e)| (p + e) * cfg.per_layer_input_scale).collect();
        (hidden, ple_inputs)
    }
    /// Tied embed/lm_head table widened to f32 for a CPU-fallback matmul.
    /// g12b (gemma4_unified loader) keeps this table host-side as f16 in
    /// `q35_f16_host` (see `gemma_embed_and_ple`); this path is only hit when
    /// the GPU LM-head dispatch is unavailable, so the transient full f32
    /// materialization here is an acceptable degraded-path cost.
    pub(crate) fn gemma_lm_head_host_f32(&self) -> std::borrow::Cow<'_, [f32]> {
        if let Some(f16v) = self.q35_f16_host.get("model.embed_tokens.weight") {
            let v: Vec<f32> = f16v.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            std::borrow::Cow::Owned(v)
        } else {
            std::borrow::Cow::Borrowed(self.inner.weights.f32_slice("model.embed_tokens.weight"))
        }
    }
    /// Final RMS norm + GPU LM head + logit softcap — runs on the last stage.
    pub(crate) fn gemma_final(&mut self, hidden: &[f32]) -> Vec<f32> {
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let norm_w = self.inner.weights.f32_slice("model.norm.weight");
        let normed = model::cpu_rms_norm(hidden, norm_w, eps);
        let vocab = cfg.vocab_size;
        let cap = cfg.final_logit_softcapping;
        // Tied LM-head dispatch descriptor: (weight buffer ptr, its own format,
        // its own MvKind). Read from the weight's recorded format/aux (like the
        // resident-layer path) so the H2 lever (VLLM_VULKAN_GEMMA_LMHEAD_Q4)
        // uploaded as Mlx4 dispatches the 4-bit matvec, while the default f16
        // upload keeps the legacy f16 matvec — no global-quant-snapshot coupling.
        let lm_dispatch: Option<(*const compute::Buffer, QuantFormat, MvKind)> =
            self.gpu_weights.get("model.embed_tokens.weight").map(|w| {
                let (fmt, kind) = gemma_res_mv_kind(w);
                (&w.buffer as *const compute::Buffer, fmt, kind)
            });
        let mut logits = if let (Some(_eng), Some((lm_w_ptr, lm_fmt, lm_kind))) = (
            self.engine.as_ref(),
            lm_dispatch,
        ) {
            let normed_bytes = f32_slice_to_bytes(&normed);
            let logit_size = (vocab * 4) as u64;
            // The logits buffer is APPENDED to `act_bufs` past the fixed
            // ACT_* slots, so only look for it (and only create it) beyond
            // `ACT_COUNT`. A plain `size == logit_size` scan over the whole Vec
            // aliases a fixed slot whenever `ffn_inter == vocab` (ACT_GATE /
            // ACT_UP / ACT_MID / ACT_GELU are all `ffn_inter*4` bytes) — the
            // LM head would then write its logits over a live FFN activation.
            if !self.act_bufs.iter().skip(ACT_COUNT).any(|b| b.size == logit_size) {
                if let Ok(buf) = self.engine.as_mut().unwrap().alloc_host_coherent_storage(logit_size) {
                    self.act_bufs.push(buf);
                }
            }
            // Both pointers are taken AFTER the push above. `act_bufs` is a
            // `Vec<Buffer>`: pushing can REALLOCATE its backing store and
            // invalidate every previously-taken element pointer, so taking
            // `inp_p` first (as this did) leaves it dangling on the call that
            // actually allocates the logits buffer.
            let inp_p = self.act_ptr_mut(ACT_QKV_IN);
            let logit_p: *const compute::Buffer = self.act_bufs.iter().skip(ACT_COUNT)
                .find(|b| b.size == logit_size)
                .map(|b| b as *const compute::Buffer)
                .unwrap_or(std::ptr::null());
            if logit_p.is_null() {
                let lm_w = self.gemma_lm_head_host_f32();
                model::cpu_matmul(&normed, &lm_w, 1, h, vocab)
            } else {
                unsafe { (*inp_p).write(&normed_bytes).unwrap(); }
                let eng = self.engine.as_mut().unwrap();
                // PROFILE bucket gemma_lmhead: the tied-embed/lm_head matvec
                // record+submit+fence+readback over the full vocab. Under the H2
                // lever this is the ~2GB-f16 -> ~0.6GB-mlx4 stream we are cutting.
                let _t_lmhead = Instant::now();
                let cb = eng.begin_batch().unwrap();
                let inp_ref = inp_p as *const compute::Buffer;
                match lm_kind {
                    // H2: mlx4 4-bit lm_head — 5 bindings [packed,scales,biases,
                    // inp,out], mlx4 push constants (built inside record_gemma_mv).
                    MvKind::Mlx4 { .. } => unsafe {
                        record_gemma_mv(eng, cb, lm_w_ptr, lm_fmt, lm_kind,
                            inp_ref, logit_p, h, vocab);
                    },
                    // Default f16 (or any non-mlx4) tied-embed lm_head: legacy
                    // 3-binding f16 matvec + 13-word push constants.
                    _ => {
                        let pc = matvec_pc13(h, vocab);
                        let (lms, lmr) = matvec_variant(true, vocab);
                        unsafe {
                            eng.record_to(cb, &lms,
                                &[&*lm_w_ptr, &*inp_ref, &*logit_p],
                                &pc, ((vocab as u32 + lmr - 1)/lmr, 1, 1)).unwrap();
                        }
                    }
                }
                eng.submit_batch(cb).unwrap();
                let out = read_f32_buf(unsafe { &*logit_p }, vocab);
                prof_add("gemma_lmhead", _t_lmhead);
                out
            }
        } else {
            let lm_w = self.gemma_lm_head_host_f32();
            model::cpu_matmul(&normed, &lm_w, 1, h, vocab)
        };
        logits.iter_mut().for_each(|l| *l = (*l / cap).tanh() * cap);
        logits
    }
    /// One decoder layer: norms on CPU, matmuls on GPU via execute_batch.
    pub(crate) fn forward_layer_gpu_matmuls(
        &mut self,
        layer_idx: usize,
        hidden: &[f32],
        pos: usize,
        layer_ple: &[f32],
    ) -> Vec<f32> {
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let num_q = cfg.num_attention_heads;
        // Per-layer KV head count: global (full-attention) layers on g12b are
        // value-less MQA(1) — was hardcoded to cfg.num_key_value_heads (the
        // sliding count), mis-sizing every global layer's K/V. Mirrors
        // gemma_resident_layer's identical fix.
        let num_kv = cfg.layer_num_kv_heads(layer_idx);
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let is_kv_shared = cfg.is_kv_shared(layer_idx);
        // Value-less global attention (g12b full-attention layers): no v_proj
        // tensor on disk; V is derived from the RAW k_proj output (before
        // k_norm) via weightless rms-norm. Mirrors cfg.layer_uses_k_eq_v /
        // gemma_resident_layer / the CPU reference.
        let uses_k_eq_v = cfg.layer_uses_k_eq_v(layer_idx);
        let ffn_inter = cfg.layer_intermediate_size(layer_idx);
        let ple_dim = cfg.hidden_size_per_layer_input;
        let t = 1usize;

        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

        // Helper: pack matvec push constants
        let mv_pc = |k: usize, n: usize| -> Vec<u8> {
            use std::io::Write;
            let mut v = Vec::with_capacity(13 * 4);
            for x in [k as u32, k as u32, k as u32, n as u32,
                       (k * n) as u32, k as u32, n as u32,
                       0u32, 0u32, 1u32, t as u32, t as u32, 1u32] {
                v.write_all(&x.to_le_bytes()).unwrap();
            }
            v
        };

        // Pre-extract all needed weight slices (avoids borrow conflicts later).
        macro_rules! w {
            ($name:expr) => { self.inner.weights.f32_slice(&ln($name)).to_vec() };
        }

        let inln_w   = w!("input_layernorm.weight");
        let q_norm_w = w!("self_attn.q_norm.weight");
        let k_norm_w = if !is_kv_shared { Some(w!("self_attn.k_norm.weight")) } else { None };
        let pa_w     = w!("post_attention_layernorm.weight");
        let pf_w     = w!("pre_feedforward_layernorm.weight");
        let postff_w = w!("post_feedforward_layernorm.weight");
        // gate_ple_w and ple_proj_w now dispatched via GPU. layer_scalar IS
        // present on every g12b layer; the PLE tensors (per_layer_input_gate/
        // per_layer_projection/post_per_layer_input_norm) are NOT — they only
        // exist for has_ple() models (E2B), so the PLE contribution block below
        // (and post_per_layer_input_norm) is gated on cfg.has_ple().
        let layer_scalar = w!("layer_scalar")[0];

        // ── ATTENTION ──────────────────────────────────────────────────────
        let residual = hidden.to_vec();
        let _t_layer = std::time::Instant::now();

        // CPU: input_layernorm
        let x = model::cpu_rms_norm(hidden, &inln_w, eps);

        let xb = f32_slice_to_bytes(&x);
        let _ = matvec_shader(true);  // variants now chosen per-call via matvec_variant()

        // Init persistent activation buffers on first call.
        let use_gpu = self.engine.is_some()
            && self.gpu_weights.contains_key(&ln("self_attn.q_proj.weight"))
            && self.init_act_bufs();

        // ── GPU BATCH: ALL 7 MATMULS IN ONE vkQueueSubmit ───────────────────
        // We batch QKV + o_proj + gate + up + down into a single command buffer.
        // The fence wait happens ONCE per layer instead of 4 times.
        // Between QKV submit and down_proj, CPU runs: Q/K/V norms, RoPE, SDPA.
        // We split into 2 submits at the attention boundary:
        //   Submit 1: Q + K + V  (before attention)
        //   Submit 2: o_proj + gate + up + down  (after attention, combined)

        let (q_vec, k_vec, v_vec) = if use_gpu {
            // Write input to persistent buffer.
            unsafe { (*self.act_ptr_mut(ACT_QKV_IN)).write(&xb).unwrap(); }

            let inp = self.act_ptr(ACT_QKV_IN);
            let q_p = self.act_ptr(ACT_Q_OUT);
            let k_p = self.act_ptr(ACT_K_OUT);
            let v_p = self.act_ptr(ACT_V_OUT);

            // Pre-extract weight buffer ptr + format/aux kind BEFORE borrowing
            // `eng` (`MvW` is Copy raw pointers, so the `gpu_weights` borrow ends
            // before `self.engine.as_mut()` takes its mutable one).
            //
            // This body keeps its own CPU orchestration (norms, RoPE and SDPA run
            // on the host here; only the matmuls are GPU) so it has no recordable
            // sequence in common with the resident paths — but it resolves its
            // weights through the SAME `layer_proj_weights`, which reads each
            // weight's own recorded format and drops `v_proj` on a value-less
            // layer. Those two rules are the ones that had to be fixed three times
            // across copies; there is now one copy of them.
            let mv = self.layer_proj_weights(
                layer_idx, uses_k_eq_v, &[ProjW::Q, ProjW::K, ProjW::V]);
            let q = mv[ProjW::Q as usize].expect("q_proj is unconditional");
            let k = mv[ProjW::K as usize].expect("k_proj is unconditional");
            let (q_w, q_fmt, q_kind) = (q.ptr, q.format, q.kind);
            let (k_w, k_fmt, k_kind) = (k.ptr, k.format, k.kind);
            // `None` exactly on a value-less global layer (no v_proj tensor at all).
            let v_meta = mv[ProjW::V as usize].map(|v| (v.ptr, v.format, v.kind));

            // SUBMIT 1: Q, K, V in one command buffer (no barriers needed — independent)
            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                record_gemma_mv(eng, cb, q_w, q_fmt, q_kind, inp, q_p, h, q_dim);
                record_gemma_mv(eng, cb, k_w, k_fmt, k_kind, inp, k_p, h, kv_dim);
                if let Some((vwp, v_fmt, v_kind)) = v_meta {
                    record_gemma_mv(eng, cb, vwp, v_fmt, v_kind, inp, v_p, h, kv_dim);
                }
            }
            eng.submit_batch(cb).unwrap();  // Fence wait 1: QKV
            if layer_idx == 0 { log::debug!("L{layer_idx} QKV submit: {}µs", _t_layer.elapsed().as_micros()); }

            let q_v = read_f32_buf(unsafe { &*q_p }, t * q_dim);
            let k_v = read_f32_buf(unsafe { &*k_p }, t * kv_dim);
            // Value-less global layers: no v_proj output to read back — V is
            // derived below from the raw (pre-k_norm) k_v.
            let v_v = if uses_k_eq_v { Vec::new() } else { read_f32_buf(unsafe { &*v_p }, t * kv_dim) };
            (q_v, k_v, v_v)
        } else {
            let q_w = self.inner.weights.f32_slice(&ln("self_attn.q_proj.weight"));
            let k_w = self.inner.weights.f32_slice(&ln("self_attn.k_proj.weight"));
            let q_out = model::cpu_matmul(&x, q_w, 1, h, q_dim);
            let k_out = model::cpu_matmul(&x, k_w, 1, h, kv_dim);
            let v_out = if uses_k_eq_v {
                Vec::new()
            } else {
                let v_w = self.inner.weights.f32_slice(&ln("self_attn.v_proj.weight"));
                model::cpu_matmul(&x, v_w, 1, h, kv_dim)
            };
            (q_out, k_out, v_out)
        };

        let mut q = q_vec;
        let mut k_final = k_vec;
        // Value-less global layers: V = raw (pre-k_norm) K, cloned BEFORE the
        // k_norm loop below mutates k_final in place. Mirrors the resident
        // path's `v_raw = k_raw.clone()` / the CPU reference.
        let mut v_final = if uses_k_eq_v { k_final.clone() } else { v_vec };

        // CPU: Q-norm, K-norm, V-norm (using pre-extracted weights)
        for hi in 0..num_q {
            let s = &mut q[hi * head_dim..(hi + 1) * head_dim];
            let n = model::cpu_rms_norm(s, &q_norm_w, eps);
            s.copy_from_slice(&n);
        }
        if !is_kv_shared {
            let k_norm = k_norm_w.as_ref().unwrap();
            for hi in 0..num_kv {
                let s = &mut k_final[hi * head_dim..(hi + 1) * head_dim];
                let n = model::cpu_rms_norm(s, k_norm, eps);
                s.copy_from_slice(&n);
            }
            for hi in 0..num_kv {
                let s = &mut v_final[hi * head_dim..(hi + 1) * head_dim];
                let n = model::cpu_rms_norm_no_weight(s, head_dim, eps);
                s.copy_from_slice(&n);
            }
        }

        // CPU: RoPE
        let (theta, rotary_dim) = if is_full {
            (1_000_000.0f32, head_dim / 4)
        } else {
            (10_000.0f32, head_dim)
        };
        model::cpu_rope_with_basis(&mut q, &mut k_final, pos, num_q, num_kv, head_dim, rotary_dim, head_dim, theta);

        // CPU: KV cache update + SDPA
        let target_cache_idx = if is_kv_shared {
            self.inner.kv_shared_target(layer_idx)
        } else {
            self.inner.kv_caches[layer_idx].append(&k_final, &v_final);
            layer_idx
        };
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        // GPU decode attention for Gemma4 (roadmap P2). The sliding window is a
        // contiguous KV slice: cpu_sdpa attends keys [seq_len-window, seq_len),
        // and the decode query (single, at the last position) attends ALL of an
        // uploaded KV range causally — so slicing the cache to that range and
        // running the validated gpu_sdpa (paged_attn_decode, cos=1.0) is exactly
        // equivalent (global layers: full range). Per-layer head_dim (256 sliding
        // / 512 global) flows through unchanged. KV-sharing is already handled by
        // target_cache_idx. Gated behind VLLM_VULKAN_GEMMA_GPU_ATTN=1.
        let gemma_gpu_attn = self.engine.is_some() && self.flags.gemma_gpu_attn;
        // Per-layer-sized KV: sliding layers read the ring's last-`window`
        // positions via `windowed_view` (ascending-absolute, `window=None`),
        // bit-identical to the old absolute `k_up_to_now()[kv_start..]` slice
        // for an unwrapped cache and correct once the ring wraps. Full layers
        // keep the absolute path.
        let attn_out = if gemma_gpu_attn {
            let (kw, vw, vlen) = {
                let cache = &self.inner.kv_caches[target_cache_idx];
                match window {
                    Some(w) => cache.windowed_view(w),
                    None => (cache.k_up_to_now().to_vec(), cache.v_up_to_now().to_vec(), cache.seq_len),
                }
            };
            self.gpu_sdpa(&q, &kw, &vw, num_q, num_kv, head_dim, vlen, 1.0)
        } else {
            let cache = &self.inner.kv_caches[target_cache_idx];
            match window {
                Some(w) => {
                    let (kw, vw, vlen) = cache.windowed_view(w);
                    model::cpu_sdpa(&q, &kw, &vw, num_q, num_kv, head_dim, vlen, 1.0, None)
                }
                None => model::cpu_sdpa(
                    &q, cache.k_up_to_now(), cache.v_up_to_now(),
                    num_q, num_kv, head_dim, cache.seq_len, 1.0, None,
                ),
            }
        };

        // GPU: o_proj — use persistent buffers (no alloc/free overhead)
        let o_proj = if use_gpu && self.gpu_weights.contains_key(&ln("self_attn.o_proj.weight")) {
            let attnb = f32_slice_to_bytes(&attn_out);
            unsafe { (*self.act_ptr_mut(ACT_O_IN)).write(&attnb).unwrap(); }
            let oi   = self.act_ptr(ACT_O_IN);
            let oo   = self.act_ptr(ACT_O_OUT);
            let o = self.layer_proj_mv(layer_idx, ProjW::O);
            let (ow, o_fmt, o_kind) = (o.ptr, o.format, o.kind);
            let eng  = self.engine.as_mut().unwrap();
            let cb   = eng.begin_batch().unwrap();
            unsafe {
                record_gemma_mv(eng, cb, ow, o_fmt, o_kind, oi, oo, q_dim, h);
            }
            eng.submit_batch(cb).unwrap();
            if layer_idx == 0 { log::debug!("L{layer_idx} o_proj submit: {}µs total since layer start", _t_layer.elapsed().as_micros()); }
            read_f32_buf(unsafe { &*oo }, t * h)
        } else {
            let ow = self.inner.weights.f32_slice(&ln("self_attn.o_proj.weight"));
            model::cpu_matmul(&attn_out, ow, 1, q_dim, h)
        };

        // CPU: post_attn_norm + residual (using pre-extracted weight)
        let pa_normed = model::cpu_rms_norm(&o_proj, &pa_w, eps);
        let hidden2: Vec<f32> = residual.iter().zip(pa_normed.iter())
            .map(|(&r, &a)| r + a).collect();
        let residual2 = hidden2.clone();

        // CPU: pre_ffn_norm
        let ff_in = model::cpu_rms_norm(&hidden2, &pf_w, eps);

        // SUBMIT 2 (FUSED FFN): gate + up + gelu(gate) + gelu×up + down in ONE command buffer.
        // This eliminates 3 separate submits + CPU gelu/multiply overhead.
        // Pipeline: gate_proj → [barrier] → gelu(gate) → [barrier] → gelu×up → [barrier] → down_proj
        //           up_proj runs in parallel with gate_proj (no barrier between them)
        let ff_out = if use_gpu
            && self.gpu_weights.contains_key(&ln("mlp.gate_proj.weight"))
            && self.gpu_weights.contains_key(&ln("mlp.down_proj.weight"))
        {
            let ffb = f32_slice_to_bytes(&ff_in);
            unsafe { (*self.act_ptr_mut(ACT_FFIN)).write(&ffb).unwrap(); }

            let ffi    = self.act_ptr(ACT_FFIN);
            let gp     = self.act_ptr(ACT_GATE);   // gate_proj output (src for gelu)
            let gelu_p = self.act_ptr(ACT_GELU);   // gelu(gate) output (dst from gelu, src for mul)
            let up_p   = self.act_ptr(ACT_UP);
            let mid_p  = self.act_ptr(ACT_MID);
            let down_p = self.act_ptr(ACT_DOWN);

            let g = self.layer_proj_mv(layer_idx, ProjW::Gate);
            let u = self.layer_proj_mv(layer_idx, ProjW::Up);
            let d = self.layer_proj_mv(layer_idx, ProjW::Down);
            let (gw, g_fmt, g_kind) = (g.ptr, g.format, g.kind);
            let (uw, u_fmt, u_kind) = (u.ptr, u.format, u.kind);
            let (dw, d_fmt, d_kind) = (d.ptr, d.format, d.kind);

            // Push constants for elementwise ops over ffn_inter elements
            // gelu_f32: local_size_x=512 (gelu.comp — the wg math below MUST
            // match its width; see the under-dispatch RCA note in that shader),
            // mul_f32_f32_f32: local_size_x=256
            let gelu_wg = ((ffn_inter + 511) / 512) as u32;
            let mul_wg  = ((ffn_inter + 255) / 256) as u32;
            let ew_wg = gelu_wg; // used below for gelu dispatch
            let gelu_pc = {
                use std::io::Write;
                let mut v = Vec::with_capacity(6 * 4);
                let kx = ffn_inter as u32;
                v.write_all(&kx.to_le_bytes()).unwrap();       // KX = num elements
                v.write_all(&1u32.to_le_bytes()).unwrap();     // KY = 1
                for _ in 0..4 { v.write_all(&0u32.to_le_bytes()).unwrap(); } // param1-4
                v
            };
            // mul.comp generic_binary_head push constants for elementwise [ffn_inter] × [ffn_inter] → [ffn_inter]
            // Format: ne(uint), ne00-ne03(4 uint), nb00-nb03(4 uint), [same for src1], [same for dst], misalign(uint), param1(f32), param2(f32), param3(i32)
            // nb values are in ELEMENTS (ggml convention: nb00=1, nb01=n for a [n] flat tensor)
            let mul_pc = {
                use std::io::Write;
                let n = ffn_inter as u32;
                let mut v = Vec::with_capacity(29 * 4);
                // ne, ne00-ne03, nb00-nb03 (src0)
                for &x in &[n, n,1u32,1,1, 1u32,n,n,n] { v.write_all(&x.to_le_bytes()).unwrap(); }
                // ne10-ne13, nb10-nb13 (src1)
                for &x in &[n, 1u32,1,1, 1u32,n,n,n] { v.write_all(&x.to_le_bytes()).unwrap(); }
                // ne20-ne23, nb20-nb23 (dst)
                for &x in &[n, 1u32,1,1, 1u32,n,n,n] { v.write_all(&x.to_le_bytes()).unwrap(); }
                // misalign, param1, param2, param3
                for &x in &[0u32, 0u32, 0u32, 0u32] { v.write_all(&x.to_le_bytes()).unwrap(); }
                v
            };

            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                // Step 1: gate and up matmuls are independent (same input ffi, different outputs)
                record_gemma_mv(eng, cb, gw, g_fmt, g_kind, ffi, gp, h, ffn_inter);
                record_gemma_mv(eng, cb, uw, u_fmt, u_kind, ffi, up_p, h, ffn_inter);
                eng.record_barrier_to(cb);
                // Step 2: gelu(gate) → gelu_p  (gelu_f32: binding0=src, binding1=dst)
                eng.record_to(cb, "gelu_f32", &[&*gp, &*gelu_p], &gelu_pc, (ew_wg, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                // Step 3: mid = gelu(gate) * up
                eng.record_to(cb, "mul_f32_f32_f32", &[&*gelu_p, &*up_p, &*mid_p], &mul_pc, (mul_wg, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                // Step 4: ff_out = down_proj(mid)
                record_gemma_mv(eng, cb, dw, d_fmt, d_kind, mid_p, down_p, ffn_inter, h);
            }
            eng.submit_batch(cb).unwrap();  // ONE fence wait for all FFN ops
            if layer_idx == 0 { log::debug!("L{layer_idx} FFN submit: {}µs total since layer start", _t_layer.elapsed().as_micros()); }

            read_f32_buf(unsafe { &*down_p }, t * h)
        } else {
            // CPU fallback
            let gate_w = self.inner.weights.f32_slice(&ln("mlp.gate_proj.weight")).to_vec();
            let up_w   = self.inner.weights.f32_slice(&ln("mlp.up_proj.weight")).to_vec();
            let gate = model::cpu_matmul(&ff_in, &gate_w, 1, h, ffn_inter);
            let up   = model::cpu_matmul(&ff_in, &up_w,   1, h, ffn_inter);
            let gate_act = model::cpu_gelu(&gate);
            let mid: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
            self.gpu_matmul_or_cpu(&ln("mlp.down_proj.weight"), &mid, t, ffn_inter, h, &mv_pc(ffn_inter, h))
        };

        // CPU: post_ffn_norm + residual (using pre-extracted weight)
        let ff_normed = model::cpu_rms_norm(&ff_out, &postff_w, eps);
        let mut hidden3: Vec<f32> = residual2.iter().zip(ff_normed.iter())
            .map(|(&r, &f)| r + f).collect();

        // PLE: gate_ple matmul [H→ple_dim] on GPU (persistent buf), then CPU gelu+elementwise,
        //      then proj [ple_dim→H] on GPU (persistent buf). PLE-only: g12b has
        //      no per_layer* tensors on disk, so skip the entire contribution
        //      for it (has_ple()==false) — the layer_scalar step below still runs.
        if cfg.has_ple() {
        let ple_norm_w = w!("post_per_layer_input_norm.weight");
        let gate_ple = if use_gpu && self.gpu_weights.contains_key(&ln("per_layer_input_gate.weight")) {
            let h3b = f32_slice_to_bytes(&hidden3);
            unsafe { (*self.act_ptr_mut(ACT_FFIN)).write(&h3b).unwrap(); }  // reuse ACT_FFIN as PLE input
            let inp_p = self.act_ptr(ACT_FFIN);
            let pg_p  = self.act_ptr(ACT_PLE_G);
            let pgw   = &self.gpu_weights[&ln("per_layer_input_gate.weight")].buffer as *const compute::Buffer;
            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                let (ps, pr) = matvec_variant(true, ple_dim);
                eng.record_to(cb, &ps, &[&*pgw, &*inp_p, &*pg_p], &mv_pc(h, ple_dim), ((ple_dim as u32 + pr - 1)/pr, t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            read_f32_buf(unsafe { &*pg_p }, t * ple_dim)
        } else {
            let pgw = self.inner.weights.f32_slice(&ln("per_layer_input_gate.weight"));
            model::cpu_matmul(&hidden3, pgw, 1, h, ple_dim)
        };
        let gate_ple_act = model::cpu_gelu(&gate_ple);
        let gated: Vec<f32> = gate_ple_act.iter().zip(layer_ple.iter())
            .map(|(&g, &p)| g * p).collect();
        let contrib = if use_gpu && self.gpu_weights.contains_key(&ln("per_layer_projection.weight")) {
            let gb = f32_slice_to_bytes(&gated);
            unsafe { (*self.act_ptr_mut(ACT_PLE_G)).write(&gb).unwrap(); }  // reuse ACT_PLE_G as gated input
            let gat_p = self.act_ptr(ACT_PLE_G);
            let pc_p  = self.act_ptr(ACT_PLE_C);
            let ppw   = &self.gpu_weights[&ln("per_layer_projection.weight")].buffer as *const compute::Buffer;
            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                let (pps, ppr) = matvec_variant(true, h);
                eng.record_to(cb, &pps, &[&*ppw, &*gat_p, &*pc_p], &mv_pc(ple_dim, h), ((h as u32 + ppr - 1)/ppr, t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            read_f32_buf(unsafe { &*pc_p }, t * h)
        } else {
            let ppw = self.inner.weights.f32_slice(&ln("per_layer_projection.weight"));
            model::cpu_matmul(&gated, ppw, 1, ple_dim, h)
        };
        let contrib_normed = model::cpu_rms_norm(&contrib, &ple_norm_w, eps);
        hidden3.iter_mut().zip(contrib_normed.iter()).for_each(|(hv, &c)| *hv += c);
        } // end has_ple() PLE contribution (skipped for g12b)

        // Layer scalar (pre-extracted; present on every g12b layer)
        hidden3.iter_mut().for_each(|v| *v *= layer_scalar);
        if layer_idx == 0 { log::debug!("L{layer_idx} END: {}µs total", _t_layer.elapsed().as_micros()); }
        hidden3
    }
    // ─── Fused GPU-resident Gemma4 decode (roadmap P4) ──────────────────────
    /// Stable raw pointer to one persistent `GR_*` activation buffer.
    ///
    /// Raw, not a reference, so the immutable borrow of `self.gres_bufs` ends
    /// before `self.engine.as_mut()` takes its mutable borrow — one recording
    /// needs both. Valid only while `gres_bufs` is not reallocated:
    /// `init_gres_bufs` fills it once and never resizes it, so nothing may
    /// allocate activation buffers part-way through a command buffer.
    pub(crate) fn gres_ptr(&self, slot: usize) -> *const compute::Buffer {
        &self.gres_bufs[slot] as *const compute::Buffer
    }
    /// Mutable twin of `gres_ptr`, for the few slots a dispatch writes host-side
    /// (rope position, dummies) before recording. Same non-reallocation
    /// requirement, and the caller must not hold it across an `init_gres_bufs`.
    pub(crate) fn gres_ptr_mut(&mut self, slot: usize) -> *mut compute::Buffer {
        &mut self.gres_bufs[slot] as *mut compute::Buffer
    }
    /// The `GR_*` activation arena as a `layer_core` slot table. Sound because
    /// `Slot`'s discriminants are asserted equal to the `GR_*` indices above.
    fn gemma_slot_bufs(&self) -> [*const compute::Buffer; SLOT_COUNT] {
        let mut a = [std::ptr::null(); SLOT_COUNT];
        for (i, e) in a.iter_mut().enumerate() {
            *e = &self.gres_bufs[i] as *const compute::Buffer;
        }
        a
    }
    /// Bind the gemma arena plus exactly the norms and projections ONE recording
    /// reads. Callers name only what they touch: `gemma_norm_w` does not stage
    /// `k_norm` for KV-shared layers, and a TP sub-block CB records a subset of the
    /// projections — indexing either map for something a layer type or a checkpoint
    /// does not have is a panic, not a lookup.
    ///
    /// `v_proj` is dropped for value-less layers inside `layer_proj_weights`, so no
    /// caller can re-introduce the unconditional index that broke those layers.
    fn gemma_layer_ptrs(
        &self, layer_idx: usize, uses_k_eq_v: bool, norms: &[NormW], projs: &[ProjW],
    ) -> LayerPtrs {
        let mut p = LayerPtrs { bufs: self.gemma_slot_bufs(), ..Default::default() };
        for &n in norms {
            let name = format!("model.layers.{layer_idx}.{}", GEMMA_NORM_NAMES[n as usize]);
            p.set_norm(n, &self.gemma_norm_w[&name]);
        }
        p.projs = self.layer_proj_weights(layer_idx, uses_k_eq_v, projs);
        p
    }
    /// The front's norm set for `layer_idx`: q-norm always, `input_layernorm` when
    /// the caller folds it onto the GPU, k-norm only on non-KV-shared layers.
    fn gemma_front_norms(cfg: &model::Gemma4Config, layer_idx: usize, input_norm: bool) -> Vec<NormW> {
        let mut v = Vec::with_capacity(3);
        if input_norm { v.push(NormW::InputLn); }
        v.push(NormW::QNorm);
        if !cfg.is_kv_shared(layer_idx) { v.push(NormW::KNorm); }
        v
    }
    /// Allocate the persistent activation buffers for the fused Gemma layer once.
    /// Sized for the MAX over layer types: global head_dim (512) and double-wide
    /// ffn_inter (intermediate_size*2 on the KV-shared layers).
    pub(crate) fn init_gres_bufs(&mut self) -> bool {
        if self.gres_ready { return true; }
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.global_head_dim;   // 8*512
        let kv_dim = cfg.num_key_value_heads * cfg.global_head_dim;  // 1*512
        let ffn_inter = cfg.intermediate_size * 2;                   // double-wide max
        let ple_dim = cfg.hidden_size_per_layer_input;
        let sizes: [u64; GR_COUNT] = [
            (h * 4) as u64,          // GR_HA
            (h * 4) as u64,          // GR_HB
            (h * 4) as u64,          // GR_X
            (q_dim * 4) as u64,      // GR_Q
            (kv_dim * 4) as u64,     // GR_K
            (kv_dim * 4) as u64,     // GR_V
            (q_dim * 4) as u64,      // GR_ATTN
            (h * 4) as u64,          // GR_O
            (h * 4) as u64,          // GR_ON
            (h * 4) as u64,          // GR_FFIN
            (ffn_inter * 4) as u64,  // GR_GATE
            (ffn_inter * 4) as u64,  // GR_UP
            (ffn_inter * 4) as u64,  // GR_GELU
            (ffn_inter * 4) as u64,  // GR_MID
            (h * 4) as u64,          // GR_DOWN
            (h * 4) as u64,          // GR_DOWNN
            4,                       // GR_POS  (1 int)
            4,                       // GR_FF   (1 f32)
            8,                       // GR_IDX  (2 u32)
            (h * 4) as u64,          // GR_DUMMY
            // g12b has ple_dim==0 (no PLE); a 0-byte alloc risks failing on
            // some Vulkan implementations and would sink `ready` to false for
            // every layer, so floor it at 4 bytes (never read: the PLE tail
            // is skipped entirely when `!cfg.has_ple()`).
            (ple_dim * 4).max(4) as u64, // GR_PLE_G
            (h * 4) as u64,          // GR_PLE_C
        ];
        let eng = match self.engine.as_mut() { Some(e) => e, None => return false };
        let mut bufs = Vec::with_capacity(GR_COUNT);
        for &sz in &sizes {
            match eng.alloc_host_coherent_storage(sz) {
                Ok(b) => bufs.push(b),
                Err(e) => { log::warn!("init_gres_bufs alloc failed: {e}"); return false; }
            }
        }
        bufs[GR_FF].write(&1.0f32.to_le_bytes()).ok();
        bufs[GR_IDX].write(&0u64.to_le_bytes()).ok();
        self.gres_bufs = bufs;
        self.gres_ready = true;
        true
    }
    /// Upload every f32 norm weight the fused Gemma layer reads into
    /// `gemma_norm_w` ONCE (no per-token inserts → stable Buffer pointers).
    /// v_norm is NOT included: it is no-weight (rms_norm_f32). Returns false if
    /// any expected weight is absent.
    pub(crate) fn ensure_gemma_norm_weights(&mut self) -> bool {
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let ple_dim = cfg.hidden_size_per_layer_input;
        let mut names: Vec<(String, usize)> = Vec::new();
        for li in self.pp_start..self.pp_end {
            let hd = cfg.layer_head_dim(li);
            names.push((format!("model.layers.{li}.input_layernorm.weight"), h));
            names.push((format!("model.layers.{li}.self_attn.q_norm.weight"), hd));
            if !cfg.is_kv_shared(li) {
                names.push((format!("model.layers.{li}.self_attn.k_norm.weight"), hd));
            }
            names.push((format!("model.layers.{li}.post_attention_layernorm.weight"), h));
            names.push((format!("model.layers.{li}.pre_feedforward_layernorm.weight"), h));
            names.push((format!("model.layers.{li}.post_feedforward_layernorm.weight"), h));
            // g12b has no PLE (hidden_size_per_layer_input=0) and the
            // checkpoint carries no `per_layer*` tensors at all — fetching
            // this by name would panic (`Weight not found`). Only present
            // for has_ple() models (E2B).
            if cfg.has_ple() {
                names.push((format!("model.layers.{li}.post_per_layer_input_norm.weight"), ple_dim));
            }
        }
        for (name, n) in names {
            if self.gemma_norm_w.contains_key(&name) { continue; }
            let w = self.inner.weights.f32_slice(&name).to_vec();
            if w.len() < n { return false; }
            let eng = self.engine.as_mut().unwrap();
            let buf = match eng.alloc_host_coherent_storage((n * 4) as u64) {
                Ok(b) => b, Err(_) => return false,
            };
            if buf.write(&f32_slice_to_bytes(&w[..n])).is_err() { return false; }
            self.gemma_norm_w.insert(name, buf);
        }
        true
    }
    /// FUSED GPU-resident Gemma4 decode forward (roadmap P4). The expensive part
    /// of each layer — input-norm → q/k/v → q/k/v-norm → per-layer-type RoPE →
    /// (attention) → o → post_attn_norm → residual → pre_ffn_norm → GELU FFN →
    /// post_ffn_norm → residual — runs through PERSISTENT GPU buffers recorded
    /// into TWO command buffers (split only at the attention boundary). Gemma's
    /// SANDWICH norms are applied to each sublayer OUTPUT before the residual
    /// add (GR_ON/GR_DOWNN). Hidden ping-pongs GR_HA↔GR_HB. The PLE tail keeps
    /// the reference's CPU gelu/mul glue (matmuls stay on GPU). Numerically
    /// tracks forward_layer_gpu_matmuls (argmax-identical, cos≈1.0).
    /// Gated behind VLLM_VULKAN_GEMMA_RESIDENT=1; falls back to forward_gpu if
    /// the GPU/weights/buffers aren't ready.
    pub(crate) fn forward_gemma_gpu_resident(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let num_q = cfg.num_attention_heads;
        let num_kv = cfg.num_key_value_heads;
        let ple_dim = cfg.hidden_size_per_layer_input;

        let l0_q = "model.layers.0.self_attn.q_proj.weight".to_string();
        let ready = self.engine.is_some()
            && self.gpu_weights.contains_key(&l0_q)
            && self.init_gres_bufs()
            && self.ensure_gemma_norm_weights()
            // gemma_final's GPU LM-head path reads act_ptr_mut(ACT_QKV_IN),
            // but the resident path never otherwise touches self.act_bufs —
            // without this, gemma_final panics on an empty act_bufs Vec.
            && self.init_act_bufs();
        if !ready {
            return self.forward_gpu(token_id, position);
        }

        // Embedding (scaled) + PLE inputs — identical to forward_gpu's setup.
        let (hidden0, ple_inputs) = self.gemma_embed_and_ple(token_id);
        unsafe { (*self.gres_ptr_mut(GR_HA)).write(&f32_slice_to_bytes(&hidden0)).unwrap(); }
        unsafe { (*self.gres_ptr_mut(GR_POS)).write(&(position as i32).to_le_bytes()).unwrap(); }

        // LEVER (VLLM_VULKAN_GEMMA_RESIDENT_1CB): fold each layer's 2 CBs + host
        // SDPA round-trip into ONE CB with GPU-resident KV + in-CB attention.
        // Only for models without PLE (g12b/g31b): the E2B PLE tail keeps its
        // host glue, so it stays on the proven 2-CB path.
        let use_1cb = self.flags.gemma_resident_1cb && !cfg.has_ple();
        for layer_idx in self.pp_start..self.pp_end {
            let layer_ple = &ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim];
            if use_1cb {
                self.gemma_resident_layer_1cb(layer_idx, position, layer_ple);
            } else {
                self.gemma_resident_layer(layer_idx, position, layer_ple);
            }
        }
        // All layers now hold token `position` in their gpu_kv (appended in-CB).
        if use_1cb {
            self.gemma_kv_filled = position + 1;
        }

        // Final norm + LM head + softcap (reuse the validated gemma_final). It
        // expects hidden in host memory; hidden currently lives in GR_HA.
        let hidden = read_f32_buf(unsafe { &*self.gres_ptr(GR_HA) }, h);
        let _ = (num_q, num_kv, eps);
        self.gemma_final(&hidden)
    }
    /// One fused Gemma decoder layer over the persistent GR_* buffers. Hidden
    /// enters in GR_HA and leaves in GR_HA (ping-pong via GR_HB).
    pub(crate) fn gemma_resident_layer(&mut self, layer_idx: usize, pos: usize, layer_ple: &[f32]) {
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let num_q = cfg.num_attention_heads;
        // Per-layer KV head count: global (full-attention) layers on g12b are
        // value-less MQA(1); E2B is MQA(1) everywhere too, so this is a no-op
        // there. Was hardcoded to `cfg.num_key_value_heads` (=8 sliding count
        // on g12b), which mis-sized every global layer's K/V by 8x.
        let num_kv = cfg.layer_num_kv_heads(layer_idx);
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let is_kv_shared = cfg.is_kv_shared(layer_idx);
        // Value-less global attention (g12b full-attention layers): there is
        // no `v_proj` tensor on disk; V is derived from the RAW k_proj output
        // (before k_norm) through a weightless rms-norm. Mirrors
        // `Gemma4Config::layer_uses_k_eq_v` / the CPU reference at
        // model.rs's `forward_layer` (`v_raw = k_raw.clone()` taken BEFORE
        // k_norm mutates it).
        let uses_k_eq_v = cfg.layer_uses_k_eq_v(layer_idx);
        let ffn_inter = cfg.layer_intermediate_size(layer_idx);
        let ple_dim = cfg.hidden_size_per_layer_input;
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

        // ── CB1: input_layernorm → q/k/v → q/k/v-norm → RoPE ─────────────────
        // The dispatch sequence is `layer_core::record_front`; everything
        // per-layer (head dim, KV heads, RoPE theta / rotary width, the value-less
        // V derivation) comes from `gemma_front_params(cfg, layer_idx, ..)`.
        let front = gemma_front_params(&cfg, layer_idx, true, num_q);
        let front_norms = Self::gemma_front_norms(&cfg, layer_idx, true);
        let front_ptrs = self.gemma_layer_ptrs(layer_idx, uses_k_eq_v, &front_norms, &FRONT_PROJS);
        let qp = self.gres_ptr(GR_Q);
        let kp = self.gres_ptr(GR_K);
        let vp = self.gres_ptr(GR_V);

        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            let mut rec = GpuRecorder { eng, cb, p: &front_ptrs };
            layer_core::record_front(&mut rec, &front).unwrap();
        }
        eng.submit_batch(cb).unwrap();

        // ── Host boundary: KV cache append + windowed attention ──────────────
        let q_host = read_f32_buf(unsafe { &*qp }, q_dim);
        let k_host = read_f32_buf(unsafe { &*kp }, kv_dim);
        let v_host = read_f32_buf(unsafe { &*vp }, kv_dim);
        let target_cache_idx = if is_kv_shared {
            self.inner.kv_shared_target(layer_idx)
        } else {
            self.inner.kv_caches[layer_idx].append(&k_host, &v_host);
            layer_idx
        };
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        let gemma_gpu_attn = self.flags.gemma_gpu_attn;
        // Per-layer-sized KV: sliding layers read the ring's last-`window`
        // positions via `windowed_view` (ascending-absolute, `window=None`),
        // bit-identical to the old absolute slice for an unwrapped cache and
        // correct once the ring wraps; full layers keep the absolute path.
        let attn = if gemma_gpu_attn {
            let (kw_s, vw_s, vlen) = {
                let cache = &self.inner.kv_caches[target_cache_idx];
                match window {
                    Some(w) => cache.windowed_view(w),
                    None => (cache.k_up_to_now().to_vec(), cache.v_up_to_now().to_vec(), cache.seq_len),
                }
            };
            self.gpu_sdpa(&q_host, &kw_s, &vw_s, num_q, num_kv, head_dim, vlen, 1.0)
        } else {
            let cache = &self.inner.kv_caches[target_cache_idx];
            match window {
                Some(w) => {
                    let (kw, vw, vlen) = cache.windowed_view(w);
                    model::cpu_sdpa(&q_host, &kw, &vw, num_q, num_kv, head_dim, vlen, 1.0, None)
                }
                None => model::cpu_sdpa(&q_host, cache.k_up_to_now(), cache.v_up_to_now(),
                    num_q, num_kv, head_dim, cache.seq_len, 1.0, None),
            }
        };

        // ── CB2: o → post_attn_norm → +residual → pre_ffn_norm → GELU FFN →
        //         post_ffn_norm → +residual2 ──────────────────────────────────
        unsafe { (*self.gres_ptr_mut(GR_ATTN)).write(&f32_slice_to_bytes(&attn)).unwrap(); }
        let tail = gemma_tail_params(&cfg, q_dim, ffn_inter);
        let tail_ptrs = self.gemma_layer_ptrs(
            layer_idx, uses_k_eq_v,
            &[NormW::PostAttn, NormW::FfnIn, NormW::PostFfn], &TAIL_PROJS);

        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            let mut rec = GpuRecorder { eng, cb, p: &tail_ptrs };
            layer_core::record_tail(&mut rec, &tail).unwrap();
        }
        eng.submit_batch(cb).unwrap();

        // ── PLE tail (E2B only; g12b has hidden_size_per_layer_input=0 and
        //    no `per_layer*` tensors on disk at all — gate the whole block on
        //    `cfg.has_ple()` or this panics on the g12b checkpoint fetching
        //    per_layer_input_gate/per_layer_projection/post_per_layer_input_norm,
        //    none of which exist there). `layer_scalar` IS present on every
        //    layer of BOTH E2B and g12b (verified against the checkpoint and
        //    already applied unconditionally by the CPU reference,
        //    `Gemma4Model::forward_layer`), so it stays unconditional here too.
        // hidden3 currently lives in GR_HA. gate_ple = hidden3 @ per_layer_input_gate.
        let mut hidden3 = read_f32_buf(unsafe { &*self.gres_ptr(GR_HA) }, h);
        if cfg.has_ple() {
            let (ps, prr) = matvec_variant(true, ple_dim);
            let mv_pg = matvec_pc13(h, ple_dim);
            let pgw = &self.gpu_weights[&ln("per_layer_input_gate.weight")].buffer as *const compute::Buffer;
            let ffin_p = self.gres_ptr(GR_FFIN);   // reuse FFIN as scratch input
            let pg_p = self.gres_ptr(GR_PLE_G);
            unsafe { (*self.gres_ptr_mut(GR_FFIN)).write(&f32_slice_to_bytes(&hidden3)).unwrap(); }
            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, &ps, &[&*pgw, &*ffin_p, &*pg_p], &mv_pg, ((ple_dim as u32 + prr - 1)/prr, 1, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            let gate_ple = read_f32_buf(unsafe { &*pg_p }, ple_dim);
            let gate_ple_act = model::cpu_gelu(&gate_ple);
            let gated: Vec<f32> = gate_ple_act.iter().zip(layer_ple.iter()).map(|(&g, &p)| g * p).collect();

            let (pps, pprr) = matvec_variant(true, h);
            let mv_pp = matvec_pc13(ple_dim, h);
            let ppw = &self.gpu_weights[&ln("per_layer_projection.weight")].buffer as *const compute::Buffer;
            let pg_in = self.gres_ptr(GR_PLE_G);
            let pc_p = self.gres_ptr(GR_PLE_C);
            unsafe { (*self.gres_ptr_mut(GR_PLE_G)).write(&f32_slice_to_bytes(&gated)).unwrap(); }
            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, &pps, &[&*ppw, &*pg_in, &*pc_p], &mv_pp, ((h as u32 + pprr - 1)/pprr, 1, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            let contrib = read_f32_buf(unsafe { &*pc_p }, h);

            let ple_norm_w = self.inner.weights.f32_slice(&ln("post_per_layer_input_norm.weight")).to_vec();
            let contrib_normed = model::cpu_rms_norm(&contrib, &ple_norm_w, eps);
            hidden3.iter_mut().zip(contrib_normed.iter()).for_each(|(hv, &c)| *hv += c);
        }
        let layer_scalar = self.inner.weights.f32_slice(&ln("layer_scalar"))[0];
        hidden3.iter_mut().for_each(|v| *v *= layer_scalar);

        // Write the final hidden back to GR_HA for the next layer.
        unsafe { (*self.gres_ptr_mut(GR_HA)).write(&f32_slice_to_bytes(&hidden3)).unwrap(); }
    }

    /// Seed the resident KV plane `gpu_kv[layer]` with host-cache positions
    /// [from..to) — used ONLY when a batched-prefill path advanced the position
    /// past `gemma_kv_filled` without touching gpu_kv (never in the token-by-token
    /// decode flow). Host-coherent mapped write; no submit. Value-less global
    /// layers rely on the host cache already holding the derived (k_eq_v) V.
    ///
    /// `capacity` is the plane's physical slot count. For a full/global layer this
    /// is `max_seq` and every position `p` lands at absolute slot `p` (identical to
    /// the pre-ring bulk copy). For a `window`-sized sliding-layer RING plane the
    /// host cache is ALSO a `window`-sized ring now (Phase-0 host half): absolute
    /// position `p` lives at host slot `p % host_capacity`, so the read gathers
    /// per-position from the host ring (byte-identical to the old absolute slice
    /// for a full cache) and writes to gpu ring slot `p % capacity`. Only the last
    /// `capacity` positions before `to` need seeding (earlier ones are outside the
    /// window), so the copy is bounded to `[max(from, to-capacity) .. to)`.
    pub(crate) fn gemma_seed_gpu_kv(&mut self, layer: usize, num_kv: usize, head_dim: usize,
                         capacity: usize, from: usize, to: usize) {
        if to <= from { return; }
        let stride = num_kv * head_dim;
        let plane = capacity * stride;
        let _ = self.ensure_gpu_kv(layer, num_kv, head_dim, capacity);
        // Only the most-recent `capacity` positions can be resident in the ring.
        let start = from.max(to.saturating_sub(capacity));
        let (kslice, vslice) = {
            let cache = &self.inner.kv_caches[layer];
            if cache.seq_len < to { return; } // host lacks these positions — skip (misuse)
            // Gather [start..to) from the host ring by slot (`p % host_capacity`).
            // For a full/global host cache host_capacity == max_seq so slot == p
            // and this is the same contiguous run the old absolute slice produced.
            let hcap = cache.capacity;
            let n = to - start;
            let mut ks = vec![0f32; n * stride];
            let mut vs = vec![0f32; n * stride];
            for (i, p) in (start..to).enumerate() {
                let hslot = p % hcap;
                ks[i * stride..(i + 1) * stride]
                    .copy_from_slice(&cache.k[hslot * stride..(hslot + 1) * stride]);
                vs[i * stride..(i + 1) * stride]
                    .copy_from_slice(&cache.v[hslot * stride..(hslot + 1) * stride]);
            }
            (ks, vs)
        };
        if let Some(buf) = self.gpu_kv.get(&layer) {
            if let Some(mp) = buf.mapped_ptr {
                let base = mp as *mut u8;
                let (kb, vb) = (f32_slice_to_bytes(&kslice), f32_slice_to_bytes(&vslice));
                // Write each position into its ring slot. For full layers slot == p
                // and this is a contiguous run (identical to the old bulk copy).
                for (i, p) in (start..to).enumerate() {
                    let slot = p % capacity;
                    let src = i * stride * 4;
                    let cpy = stride * 4;
                    unsafe {
                        std::ptr::copy_nonoverlapping(kb.as_ptr().add(src), base.add(slot * stride * 4), cpy);
                        std::ptr::copy_nonoverlapping(vb.as_ptr().add(src), base.add((plane + slot * stride) * 4), cpy);
                    }
                }
            }
        }
    }

    /// ONE-CB Gemma4 decode layer (VLLM_VULKAN_GEMMA_RESIDENT_1CB). Identical
    /// math to `gemma_resident_layer` but the per-layer HOST SDPA round-trip is
    /// eliminated: after RoPE, this token's K/V are buffer-copied into the
    /// GPU-resident `gpu_kv[layer]` plane IN-CB, then the `_sg` decode-attn
    /// dispatch reads that resident KV IN-CB (sliding-window via `window_start`,
    /// per-layer kv-heads/head_dim, value-less global V from GR_V), and the whole
    /// layer — norms/qkv/qk-norm/RoPE → KV-append → attention → o_proj/FFN — runs
    /// through a SINGLE command buffer (1 submit/layer vs the 2-CB path's 2). Only
    /// the tiny host `layer_scalar` [h] readback remains (no PLE: gated on
    /// `!has_ple()`). Not implemented for kv-shared layers (falls back to the
    /// 2-CB path; never taken on g12b/g31b).
    pub(crate) fn gemma_resident_layer_1cb(&mut self, layer_idx: usize, pos: usize, layer_ple: &[f32]) {
        let _t_setup = Instant::now();
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let num_q = cfg.num_attention_heads;
        let num_kv = cfg.layer_num_kv_heads(layer_idx);
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let is_kv_shared = cfg.is_kv_shared(layer_idx);
        let uses_k_eq_v = cfg.layer_uses_k_eq_v(layer_idx);
        let ffn_inter = cfg.layer_intermediate_size(layer_idx);
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

        // KV-shared layers (E2B only) reuse another layer's KV — the 1-CB fold
        // does not implement that seam, so defer to the proven 2-CB path. Never
        // taken on g12b/g31b (first_kv_shared == num_hidden_layers).
        if is_kv_shared {
            return self.gemma_resident_layer(layer_idx, pos, layer_ple);
        }

        let max_seq = self.inner.kv_caches[layer_idx].max_seq_len;
        let seq_len = pos + 1;
        // Per-layer-sized KV (Phase 0): a sliding-window layer only ever attends
        // the last `sliding_window` positions, so its resident plane is a
        // `window`-sized RING (capacity = window); absolute pos `t` lives at ring
        // slot `t % capacity`. Global/full layers keep a full max_seq plane
        // (capacity == max_seq, ring_capacity 0 = absolute addressing → allocation,
        // append offsets and shader addressing are all byte-for-byte unchanged).
        // KV_RING_DISABLE forces the pre-ring uniform allocation (capacity ==
        // max_seq, absolute addressing) for the on-cluster A/B reference.
        let capacity = if self.flags.kv_ring_disable {
            max_seq
        } else {
            cfg.layer_kv_capacity(layer_idx, max_seq)
        };
        let ring_capacity = if is_full || self.flags.kv_ring_disable { 0 } else { capacity };
        // Sliding layers attend the last `sliding_window` positions; global
        // (full) layers attend everything. Matches cpu_sdpa's `kv_start`.
        let window_start = if is_full { 0 } else { seq_len.saturating_sub(cfg.sliding_window) };
        let plane = capacity * kv_dim;

        // Per-layer RoPE / norms / the value-less V all come from the SAME
        // `gemma_front_params` the 2-CB path uses — this path differs only in
        // WHERE the attention runs, never in what the front records.
        let front = gemma_front_params(&cfg, layer_idx, true, num_q);
        let tail = gemma_tail_params(&cfg, q_dim, ffn_inter);

        // Defensive: if a batched-prefill path left gpu_kv behind (host cache
        // ahead), seed the gap once. No-op in the token-by-token decode flow.
        if pos > self.gemma_kv_filled {
            self.gemma_seed_gpu_kv(layer_idx, num_kv, head_dim, capacity, self.gemma_kv_filled, pos);
        }
        let kv_ptr = match self.ensure_gpu_kv(layer_idx, num_kv, head_dim, capacity) {
            Some(p) => p,
            None => return self.gemma_resident_layer(layer_idx, pos, layer_ple),
        };

        // ── Gather ALL buffer/weight pointers BEFORE recording (no map inserts
        //    mid-record → stable Buffer pointers). `is_kv_shared` returned above,
        //    so k_norm is always staged for this layer. ────────────────────────
        let mut ptrs = self.gemma_layer_ptrs(
            layer_idx, uses_k_eq_v,
            &Self::gemma_front_norms(&cfg, layer_idx, true), &FRONT_PROJS);
        {
            let t = self.gemma_layer_ptrs(
                layer_idx, uses_k_eq_v,
                &[NormW::PostAttn, NormW::FfnIn, NormW::PostFfn], &TAIL_PROJS);
            for (i, m) in t.projs.iter().enumerate() {
                if m.is_some() { ptrs.projs[i] = *m; }
            }
            for (i, n) in t.norms.iter().enumerate() {
                if !n.is_null() { ptrs.norms[i] = *n; }
            }
        }
        let qp = self.gres_ptr(GR_Q);
        let kp = self.gres_ptr(GR_K);
        let vp = self.gres_ptr(GR_V);
        let idxp = self.gres_ptr(GR_IDX);
        let attnp = self.gres_ptr(GR_ATTN);

        // Resident-KV in-CB append geometry (this token at absolute `pos`).
        // Ring slot = pos % capacity; for full layers capacity == max_seq and
        // pos < max_seq, so slot == pos (byte-identical to the old absolute offset).
        let slot = pos % capacity;
        let k_dst_off = (slot * kv_dim * 4) as u64;
        let v_dst_off = ((plane + slot * kv_dim) * 4) as u64;
        let kv_copy_sz = (kv_dim * 4) as u64;
        // In-CB decode attention (scale 1.0, matching the 2-CB path's SDPA calls).
        let sdpa_kernel = attn_decode_kernel();
        let sdpa_wg = match sdpa_kernel {
            "paged_attn_decode_f32_sg"   => (num_q as u32, 1u32, 1u32),
            "paged_attn_decode_f32_coop" => (num_q as u32, (head_dim as u32 + 255) / 256, 1u32),
            _                            => ((q_dim as u32 + 255) / 256, 1u32, 1u32),
        };
        let sdpa_pc_v = sdpa_pc(seq_len, num_q, num_kv, head_dim, capacity, plane, 1.0, window_start, ring_capacity);

        unsafe { (*self.gres_ptr_mut(GR_POS)).write(&(pos as i32).to_le_bytes()).unwrap(); }

        // ── ONE command buffer: CB1 + in-CB KV append + in-CB attn + CB2. ──────
        prof_add("g1cb_setup", _t_setup);
        let _t_rec = Instant::now();
        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        {
            // input_layernorm → q/k/v → q/k/v-norm (value-less V derived from the
            // RAW K) → RoPE. THE shared body; see `layer_core::record_front`.
            let mut rec = GpuRecorder { eng, cb, p: &ptrs };
            layer_core::record_front(&mut rec, &front).unwrap();
        }
        unsafe {
            // RoPE'd Q (SHADER_WRITE, COMPUTE) feeds the SDPA dispatch below
            // (SHADER_READ, COMPUTE). The COMPUTE→TRANSFER + TRANSFER→COMPUTE
            // pair below covers only K/V: it makes TRANSFER_WRITEs visible to
            // SHADER_READ, never the COMPUTE SHADER_WRITE of Q. Without this
            // barrier Q's write is never made visible to the SDPA read.
            eng.record_barrier_to(cb);
            // ── Resident-KV append: RoPE'd K/V (COMPUTE write) → buffer-copy
            //    into gpu_kv[layer] at ring slot `pos % capacity` (TRANSFER). ───
            eng.record_compute_to_transfer_barrier(cb);
            eng.record_copy_to(cb, &*kp, &*kv_ptr, 0, k_dst_off, kv_copy_sz);
            eng.record_copy_to(cb, &*vp, &*kv_ptr, 0, v_dst_off, kv_copy_sz);
            eng.record_transfer_to_compute_barrier(cb);

            // ── In-CB decode attention over the resident KV → GR_ATTN. Q read
            //    from GR_Q (RoPE'd this CB); block-table = GR_IDX (=0). ──────────
            eng.record_to(cb, sdpa_kernel, &[&*qp, &*idxp, &*kv_ptr, &*attnp], &sdpa_pc_v, sdpa_wg).unwrap();
            eng.record_barrier_to(cb);
        }
        {
            // o_proj → residual → pre_ffn_norm → FFN → residual2.
            let mut rec = GpuRecorder { eng, cb, p: &ptrs };
            layer_core::record_tail(&mut rec, &tail).unwrap();
        }
        prof_add("g1cb_record", _t_rec);
        eng.submit_batch(cb).unwrap();

        // ── Host tail: layer_scalar only (has_ple() is false here). Tiny [h]
        //    readback — no submit, no attention round-trip. Mirrors
        //    gemma_resident_layer's tail with the PLE block elided. ─────────────
        let _t_tail = Instant::now();
        let _ = layer_ple;
        let mut hidden3 = read_f32_buf(unsafe { &*self.gres_ptr(GR_HA) }, h);
        let layer_scalar = self.inner.weights.f32_slice(&ln("layer_scalar"))[0];
        hidden3.iter_mut().for_each(|v| *v *= layer_scalar);
        unsafe { (*self.gres_ptr_mut(GR_HA)).write(&f32_slice_to_bytes(&hidden3)).unwrap(); }
        prof_add("g1cb_tail", _t_tail);
    }
}

// ── HOST bit-exact validation of the GEMMA_PREFILL_COLS decomposition ────────
//
// The gemma prefill cols lever (`gemma_prefill_matmul` with
// `VLLM_VULKAN_GEMMA_PREFILL_COLS=1`) routes each `[t,k]@W[n,k]^T -> [t,n]`
// projection through `qwen35_matvec_cols_tiled`, which splits the `t` activation
// columns into <=8-column tiles (`cols_tile_schedule`), dispatches each tile
// through the single-stream cols kernel, and concatenates the tiles in column
// order. The kernel itself needs the GFX1013 GPU (no GPU on the build host), and
// its per-column vs single-row reduction equivalence is argmax-exact / cos=1.0
// — the DEFERRED on-node A/B gate. What IS host-checkable, and what these tests
// pin, is the GEMMA-SPECIFIC wiring risk: that the column-tiling schedule +
// concatenation + lone-trailing-column fold reconstruct `[t,n]` such that every
// output column equals the SAME primitive applied to that column alone (i.e. the
// per-row `gemma_prefill_matmul` oracle) BIT-FOR-BIT — no dropped, duplicated,
// or mis-placed column. We model the cols kernel with a deterministic CPU
// per-column dot product (the exact per-column math a correct kernel computes,
// with a column-independent reduction order identical between the tiled and
// per-row calls), over the real g12b/g31b projection (k,n) shapes and a T sweep
// that straddles the tile boundary. Bit-exactness is asserted via `f32::to_bits`
// (the strongest form) plus argmax-per-row.
#[cfg(test)]
mod gemma_prefill_cols_tests {
    use crate::qwen35_forward::cols_tile_schedule;

    /// Deterministic pseudo-random f32 in [-1, 1) from a 64-bit splitmix hash of
    /// (a, b) — no allocation, reproducible, no rand dep.
    ///
    /// It must stay a PURE function of (a, b): the oracle and the tiled
    /// reconstruction below derive the same weight element independently, from
    /// the same (row, col) pair, instead of sharing an n*k table. Any hidden
    /// state here would make the two disagree for reasons unrelated to tiling.
    fn h32(a: u64, b: u64) -> f32 {
        let mut z = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_add(0x1234_5678_9ABC_DEF0);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 24-bit mantissa -> [-1, 1)
        ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// One output column's projection: out_row j = sum_i W[j*k+i] * xcol[i].
    ///
    /// `n_check` caps how many output rows we materialize (columns are fully
    /// independent, so a fixed slice per column suffices to catch any placement
    /// corruption while keeping the real `k` reduction length). `wseed` derives
    /// the weight so every shape gets a distinct deterministic W without storing
    /// an n*k table.
    ///
    /// The reduction order is fixed and COLUMN-INDEPENDENT. That is what makes
    /// the bitwise comparison below meaningful: any difference between the
    /// tiled and per-row results can then only come from tiling, never from a
    /// reordered sum.
    fn project_col(wseed: u64, xcol: &[f32], k: usize, n_check: usize) -> Vec<f32> {
        let mut o = vec![0f32; n_check];
        for (j, oj) in o.iter_mut().enumerate() {
            let mut acc = 0f32;
            for i in 0..k {
                acc += h32(wseed.wrapping_add(j as u64), i as u64) * xcol[i];
            }
            *oj = acc;
        }
        o
    }

    /// Per-row oracle == today's `gemma_prefill_matmul` loop (one column at a
    /// time). This is the reference the tiled reconstruction must match
    /// BIT-for-bit, not merely approximately.
    fn oracle(wseed: u64, x: &[f32], t: usize, k: usize, n_check: usize) -> Vec<f32> {
        let mut out = vec![0f32; t * n_check];
        for ti in 0..t {
            let col = project_col(wseed, &x[ti * k..(ti + 1) * k], k, n_check);
            out[ti * n_check..(ti + 1) * n_check].copy_from_slice(&col);
        }
        out
    }

    /// Cols-tiled reconstruction == `qwen35_matvec_cols_tiled`'s schedule (the
    /// live code path via `cols_tile_schedule`), each tile's columns projected
    /// independently and written at the tile's column offset.
    ///
    /// The SCHEDULE is the live one, imported from the shipping code — only the
    /// per-column arithmetic is modelled here. A local copy of the tiling rule
    /// would gate the test against itself and pass through a schedule change.
    fn cols_tiled(wseed: u64, x: &[f32], t: usize, k: usize, n_check: usize, tile: usize) -> Vec<f32> {
        let mut out = vec![0f32; t * n_check];
        for (c0, ct) in cols_tile_schedule(t, tile) {
            for lc in 0..ct {
                let gc = c0 + lc;
                let col = project_col(wseed, &x[gc * k..(gc + 1) * k], k, n_check);
                out[gc * n_check..(gc + 1) * n_check].copy_from_slice(&col);
            }
        }
        out
    }

    /// First-maximum argmax (strict `>`, so ties keep the lowest index — the
    /// driver's own tie-break). Used as a second, coarser check alongside the
    /// bitwise one: it is the property a caller actually observes.
    fn argmax(row: &[f32]) -> usize {
        let mut best = 0usize;
        for j in 1..row.len() {
            if row[j] > row[best] { best = j; }
        }
        best
    }

    // Real g12b / g31b prefill projection (k, n) shapes — the columns dimension
    // (t) and the reduction length (k) are what drive the decomposition; n is
    // capped by `n_check` at compute time (placement is n-independent).
    // g12b: h=3840, sliding q/kv_dim=4096/2048 (hd256, 16q/8kv), global
    //   q/kv_dim=8192/512 (hd512, MQA1), inter=15360.
    // g31b: h=5376, sliding q/kv_dim=8192/4096 (hd256, 32q/16kv), global
    //   q/kv_dim=16384/2048 (hd512, GQA4), inter=21504.
    const SHAPES: &[(&str, usize, usize)] = &[
        ("g12b.sliding.qkv_in", 3840, 4096),
        ("g12b.sliding.o",      4096, 3840),
        ("g12b.sliding.kv",     3840, 2048),
        ("g12b.global.q",       3840, 8192),
        ("g12b.global.kv",      3840, 512),
        ("g12b.global.o",       8192, 3840),
        ("g12b.mlp.gate_up",    3840, 15360),
        ("g12b.mlp.down",       15360, 3840),
        ("g31b.sliding.qkv_in", 5376, 8192),
        ("g31b.sliding.o",      8192, 5376),
        ("g31b.global.q",       5376, 16384),
        ("g31b.global.kv",      5376, 2048),
        ("g31b.mlp.gate_up",    5376, 21504),
        ("g31b.mlp.down",       21504, 5376),
    ];

    /// The schedule must (a) emit only 2..=tile-wide tiles, (b) cover [0,t)
    /// exactly once per column except the single idempotent lone-column
    /// overlap, and (c) never index past t. Pure index logic — EXHAUSTIVE over
    /// the sweep, not sampled, because the failure this guards is a single
    /// awkward `t` that drops or double-writes one column.
    ///
    /// At most ONE column may be covered twice, and only by the trailing fold,
    /// which recomputes it identically. Any further overlap means a column is
    /// written by two different tiles and the last writer silently wins.
    #[test]
    fn schedule_covers_all_columns() {
        for t in 2..=260usize {
            for tile in 2..=8usize {
                let sched = cols_tile_schedule(t, tile);
                let mut cover = vec![0u32; t];
                for &(c0, ct) in &sched {
                    assert!((2..=tile).contains(&ct), "t={t} tile={tile}: bad ct={ct}");
                    assert!(c0 + ct <= t, "t={t} tile={tile}: tile ({c0},{ct}) past end");
                    for c in c0..c0 + ct { cover[c] += 1; }
                }
                // Every column covered >=1; overlap only ever from the trailing
                // fold, so total extra coverage is 0 or 1 column.
                let overlaps: usize = cover.iter().map(|&c| (c as usize).saturating_sub(1)).sum();
                assert!(cover.iter().all(|&c| c >= 1), "t={t} tile={tile}: uncovered column");
                assert!(overlaps <= 1, "t={t} tile={tile}: {overlaps} overlapped columns (>1)");
            }
        }
    }

    /// The load-bearing gate: cols-tiled projection == per-row oracle, BITWISE
    /// (`f32::to_bits`), over the real g12b/g31b projection shapes.
    ///
    /// Bitwise, not a tolerance: the tiled and per-row paths compute the same
    /// per-column dot product in the same order, so ANY difference is a
    /// placement defect (a dropped, duplicated or mis-offset column), never
    /// accumulated rounding. A tolerance here would hide exactly the class of
    /// bug the test exists for.
    ///
    /// The `t` sweep straddles the tile boundary on purpose — single-tile
    /// (t <= 8), exact multiples, and remainders that trigger the lone-column
    /// fold. A sweep that stayed under one tile would pass on an untiled path.
    #[test]
    fn cols_tiled_bit_exact_vs_per_row_oracle() {
        // Straddle the <=8 tile boundary: single-tile (t<=8), exact multiples,
        // and awkward remainders that trigger the lone-column fold.
        let ts = [2usize, 3, 5, 7, 8, 9, 15, 16, 17, 24, 33, 64];
        let tiles = [2usize, 3, 4, 6, 8];
        let n_check = 24usize; // rows materialized per column (placement check)
        for (si, &(name, k, n)) in SHAPES.iter().enumerate() {
            let nc = n_check.min(n);
            let wseed = 0xA5A5_0000u64 ^ (si as u64).wrapping_mul(0x1_0001);
            for &t in &ts {
                // Deterministic activations [t,k].
                let mut x = vec![0f32; t * k];
                for (idx, xv) in x.iter_mut().enumerate() {
                    *xv = h32(0xC0FFEE ^ si as u64, idx as u64);
                }
                let want = oracle(wseed, &x, t, k, nc);
                for &tile in &tiles {
                    let got = cols_tiled(wseed, &x, t, k, nc, tile);
                    assert_eq!(got.len(), want.len());
                    // Strongest check: exact IEEE-754 bit pattern, every element.
                    for idx in 0..want.len() {
                        assert_eq!(
                            got[idx].to_bits(), want[idx].to_bits(),
                            "{name}: bitwise mismatch t={t} tile={tile} at flat idx {idx} \
                             (col {}, row {})", idx / nc, idx % nc,
                        );
                    }
                    // And argmax-per-row parity (the on-node gate's acceptance metric).
                    for ti in 0..t {
                        assert_eq!(
                            argmax(&got[ti * nc..(ti + 1) * nc]),
                            argmax(&want[ti * nc..(ti + 1) * nc]),
                            "{name}: argmax mismatch t={t} tile={tile} row {ti}",
                        );
                    }
                }
            }
        }
    }
}
