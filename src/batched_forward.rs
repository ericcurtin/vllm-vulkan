// SPDX-License-Identifier: Apache-2.0
//! EAGLE3 Phase-0 fast-verify: `forward_batched` — a T-token (T=2..8) batched
//! forward for spec-decode verify, whose fixed overhead is ~1 submit/layer
//! (not ~7/layer like `forward_prefill`'s per-projection `gpu_gemm` calls).
//!
//! Each layer is recorded as ONE command buffer over persistent T-wide
//! activation buffers (`BR_*`, this file's slot layout — distinct from the
//! per-token `UR_*` unified-decode slots in `unified_layer.rs`):
//!   - projections: batched multi-column matvec (`_r{rows}_c{t}`, NUM_COLS
//!     spec constant), reading each weight row ONCE per workgroup for all T
//!     token-columns (the bn8 GEMM was occupancy-starved at these shapes on
//!     GFX1013) — row-batched norms piggyback on the same [T,*] row-major
//!     layout with no offset primitive needed. lm_head stays on the bn8 GEMM.
//!   - RoPE / attention are inherently per-token (position, causal seq_len),
//!     so they use `record_to_off` (a byte-offset-capable dispatch sibling of
//!     `record_to`) to slice token i's row out of the shared T-wide buffer.
//!   - K/V append is ONE contiguous `record_copy_to` per K/V (T positions are
//!     contiguous starting at `start_pos`), into the SAME resident-KV plane
//!     `gpu_layer_1cb` uses (`ensure_gpu_kv`).
//!
//! Dense/Qwen3-unified only (V1 scope — the design plan's §7 defers the
//! qwen3_5 hybrid: GatedDeltaNet's recurrence is an inherently sequential
//! host-side scan, no batching win there).

use crate::compute;
use crate::gpu_error::GpuResult;
use crate::VulkanModel;
use crate::{
    matvec_cols_variant_by_format, matvec_pc13, rmsnorm_pc, rope_neox_pc, sdpa_pc,
    ew_unary_pc, ew_mul_pc, f32_slice_to_bytes, read_f32_buf,
};
use crate::{attn_decode_kernel, prof_add};
use crate::flags::QuantFormat;
use crate::gemma_forward::{gemma_res_mv_kind, record_gemma_mv_off};

/// Max tokens a single `forward_batched` call may verify. Buffers are sized
/// for this up front (`init_bres_bufs`) and reused across calls; a bigger `t`
/// is rejected rather than silently truncated.
pub(crate) const BR_MAX_T: usize = 8;

// BR_* slot indices for the batched-verify T-wide activation buffers. Sized
// `max_t * per-token-element-count` (POS is `max_t` ints; FF/IDX stay tiny
// per-call dummies, never T-wide — RoPE reads them at offset 0 for every
// token, same as the per-token unified path).
const BR_HA:    usize = 0;  // hidden A (layer input / ffn-add output)   [T,h]
const BR_HB:    usize = 1;  // hidden B (attn-add output / ffn residual) [T,h]
const BR_X:     usize = 2;  // input_layernorm output (q/k/v input)      [T,h]
const BR_Q:     usize = 3;  // q proj -> q-norm -> rope (in place)   [T,q_dim]
const BR_K:     usize = 4;  // k proj -> k-norm -> rope (in place)  [T,kv_dim]
const BR_V:     usize = 5;  // v proj                                [T,kv_dim]
const BR_ATTN:  usize = 6;  // attention output (T resident-KV decodes) [T,q_dim]
const BR_O:     usize = 7;  // o proj output                             [T,h]
const BR_FFIN:  usize = 8;  // ffn-input norm output                     [T,h]
const BR_GATE:  usize = 9;  // gate proj output                     [T,inter]
const BR_UP:    usize = 10; // up proj output                       [T,inter]
const BR_ACT:   usize = 11; // silu(gate) output                    [T,inter]
const BR_MID:   usize = 12; // silu(gate)*up                        [T,inter]
const BR_DOWN:  usize = 13; // down proj output                          [T,h]
const BR_NORM:  usize = 14; // final model.norm output (LM head input)   [T,h]
const BR_POS:   usize = 15; // per-token rope position (T ints)
const BR_FF:    usize = 16; // rope freq-factors dummy (1 f32, offset-0 always)
const BR_IDX:   usize = 17; // rope set_rows idx / attn block-table dummy (2 u32)
const BR_DUMMY: usize = 18; // harmless binding-3 buf for add_f32_f32_f32  [T,h]
const BR_LOGITS:usize = 19; // final lm_head output                  [T,vocab]
const BR_COUNT: usize = 20;

impl VulkanModel {
    /// Stable raw pointer to one persistent `BR_*` activation buffer.
    ///
    /// Raw, not a reference, so the immutable borrow of `self.bres_bufs` ends
    /// before `self.engine.as_mut()` takes its mutable borrow — the recording
    /// loop needs both at once. Valid only while `bres_bufs` is not
    /// reallocated, which `init_bres_bufs` does exactly once (it is idempotent
    /// and never resizes afterwards), so no dispatch may allocate buffers
    /// mid-recording.
    fn bres_ptr(&self, slot: usize) -> *const compute::Buffer {
        &self.bres_bufs[slot] as *const compute::Buffer
    }

    /// Allocate the persistent T-wide (max_t=`BR_MAX_T`) activation buffers
    /// once, sized from the loaded Qwen3-dense config. Idempotent (mirrors
    /// `init_ures_bufs`). Returns false if there is no engine or no Qwen3
    /// model loaded (this V1 is dense-only — see module docs).
    pub(crate) fn init_bres_bufs(&mut self) -> bool {
        if self.bres_ready { return true; }
        let (h, q_dim, kv_dim, inter, vocab) = match self.qwen.as_ref() {
            Some(q) => {
                let c = &q.config;
                (c.hidden_size, c.num_attention_heads * c.head_dim,
                 c.num_key_value_heads * c.head_dim, c.intermediate_size, c.vocab_size)
            }
            None => return false,
        };
        let mt = BR_MAX_T;
        let sizes: [u64; BR_COUNT] = [
            (mt * h * 4) as u64,        // BR_HA
            (mt * h * 4) as u64,        // BR_HB
            (mt * h * 4) as u64,        // BR_X
            (mt * q_dim * 4) as u64,    // BR_Q
            (mt * kv_dim * 4) as u64,   // BR_K
            (mt * kv_dim * 4) as u64,   // BR_V
            (mt * q_dim * 4) as u64,    // BR_ATTN
            (mt * h * 4) as u64,        // BR_O
            (mt * h * 4) as u64,        // BR_FFIN
            (mt * inter * 4) as u64,    // BR_GATE
            (mt * inter * 4) as u64,    // BR_UP
            (mt * inter * 4) as u64,    // BR_ACT
            (mt * inter * 4) as u64,    // BR_MID
            (mt * h * 4) as u64,        // BR_DOWN
            (mt * h * 4) as u64,        // BR_NORM
            (mt * 4) as u64,            // BR_POS
            4,                          // BR_FF
            8,                          // BR_IDX
            (mt * h * 4) as u64,        // BR_DUMMY
            (mt * vocab * 4) as u64,    // BR_LOGITS
        ];
        let eng = match self.engine.as_mut() { Some(e) => e, None => return false };
        let mut bufs = Vec::with_capacity(BR_COUNT);
        for &sz in &sizes {
            match eng.alloc_host_coherent_storage(sz.max(4)) {
                Ok(b) => bufs.push(b),
                Err(e) => { log::warn!("init_bres_bufs alloc failed: {e}"); return false; }
            }
        }
        bufs[BR_FF].write(&1.0f32.to_le_bytes()).ok();
        bufs[BR_IDX].write(&0u64.to_le_bytes()).ok();
        self.bres_bufs = bufs;
        self.bres_ready = true;
        true
    }

    /// Rewind every layer's resident KV cache to `accepted` valid positions
    /// (spec-decode reject/rollback). GPU-resident planes are overwrite-in-
    /// place (§5 of the design plan): no GPU work is needed, the next append
    /// simply overwrites the abandoned slots and SDPA only reads `0..seq_len`.
    pub(crate) fn truncate_kv(&mut self, accepted: usize) {
        if let Some(qm) = self.qwen.as_mut() {
            for c in qm.kv_caches.iter_mut() { c.truncate(accepted); }
        }
    }

    /// Fetch one projection weight for the batched path as
    /// `(buffer_ptr, its OWN recorded format)`.
    ///
    /// THE INVARIANT: the multi-column `_r*_c{t}` matvec kernel is chosen from
    /// the returned format, NOT from the process-wide `VLLM_VULKAN_QUANT`
    /// snapshot. The two are not interchangeable even inside a single loader —
    /// the dense loader falls back to `F16` for any tensor whose element count
    /// is not 32-aligned while the global flag still says `q8_0`, and the
    /// global-keyed `matvec_cols_variant` would then hand those f16 bytes to
    /// the q8_0 dequant kernel (garbage logits, no error). Same class of defect
    /// `22ee4a9` fixed in the unified per-op path.
    ///
    /// PACKED formats are REFUSED rather than mis-dispatched: Mlx4/Nvfp4/Fp8
    /// need extra scale/bias bindings and have NO compiled `_c{t}` sibling, so
    /// there is no correct multi-column dispatch to select. Failing here is the
    /// point — the alternative is a plain matvec reading 4-bit nibbles as f16.
    fn bres_proj(&self, name: &str, layer_idx: usize)
        -> GpuResult<(*const compute::Buffer, QuantFormat)>
    {
        let w = self.gpu_weights.get(name)
            .ok_or_else(|| format!("forward_batched: missing {name} (layer {layer_idx})"))?;
        if w.aux.is_some() {
            return Err(format!(
                "forward_batched: {name} (layer {layer_idx}) is packed-quantized ({:?}); the \
                 batched multi-column matvec has no packed `_c{{t}}` kernel. Load this model \
                 without the packed-weight lever, or verify one token at a time.",
                w.format).into());
        }
        Ok((&w.buffer as *const compute::Buffer, w.format))
    }

    /// T-token batched forward for spec-decode verify (dense Qwen3-unified
    /// only). Appends T tokens at positions `start_pos..start_pos+T` and
    /// returns `[T*vocab]` logits (next-token logits at EVERY position, not
    /// just the last — that is the semantic a real verify pass consumes).
    ///
    /// GPU path: each layer is ONE command buffer (1 submit/layer) — see
    /// module docs. CPU-fallback (no engine, or no Qwen3 model): T sequential
    /// calls to the existing bit-exact per-token `Qwen3Model::forward`, which
    /// is the same math (`cpu_matmul`/`cpu_rms_norm`/`cpu_rope`/`cpu_sdpa`)
    /// `forward_prefill`'s CPU path already relies on — this keeps the tests
    /// deterministic on Mac and T=1 exactly bit-identical to `forward` by
    /// construction (same function, not a re-derivation of its math).
    pub(crate) fn forward_batched_impl(&mut self, tokens: Vec<u32>, start_pos: usize) -> GpuResult<Vec<f32>> {
        let t = tokens.len();
        if t == 0 { return Err("forward_batched: empty tokens".to_string().into()); }
        if t > BR_MAX_T {
            return Err(format!("forward_batched: t={t} exceeds max {BR_MAX_T}").into());
        }

        // ── CPU fallback (no GPU engine, or a non-Qwen3 model) ───────────────
        if self.engine.is_none() || self.qwen.is_none() {
            let qm = self.qwen.as_mut()
                .ok_or_else(|| "forward_batched needs a Qwen3 model".to_string())?;
            let vocab = qm.config.vocab_size;
            let mut out = Vec::with_capacity(t * vocab);
            for (i, &tok) in tokens.iter().enumerate() {
                out.extend(qm.forward(tok, start_pos + i));
            }
            return Ok(out);
        }

        // ── GPU path: 1 command buffer (1 submit) per layer ──────────────────
        let cfg = self.qwen.as_ref().unwrap().config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let num_q = cfg.num_attention_heads;
        let num_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let max_seq = self.max_seq_len;

        // Bound the WHOLE appended span against the PHYSICAL resident plane
        // before any offset is computed. `k_dst_off`/`v_dst_off` below are raw
        // byte offsets into the gpu_kv plane, consumed by `vkCmdCopyBuffer`:
        // an out-of-range `start_pos` writes `t*kv_dim*4` bytes past the end of
        // the buffer (memory corruption, not a Rust panic). `start_pos` arrives
        // straight from Python via `forward_batched`, so it is untrusted.
        // Same precedent as `forward_prefill_gemma`.
        if start_pos + t > max_seq {
            return Err(format!(
                "forward_batched: start_pos {start_pos} + t {t} exceeds max_seq_len {max_seq} — \
                 construct VulkanModel with a larger max_seq_len").into());
        }

        if !self.init_bres_bufs() {
            return Err("forward_batched: failed to allocate T-wide buffers".to_string().into());
        }

        // Embed all T tokens -> BR_HA [T,h] (host embedding table lookup, as
        // `forward_prefill` does — the embed table is not GPU-resident).
        let host_prep_ts = std::time::Instant::now();
        {
            let embed_w = self.qwen.as_ref().unwrap().weights.f32_slice("model.embed_tokens.weight");
            let mut emb = Vec::with_capacity(t * h);
            for &tok in &tokens { emb.extend_from_slice(&embed_w[tok as usize * h..(tok as usize + 1) * h]); }
            self.bres_bufs[BR_HA].write(&f32_slice_to_bytes(&emb))?;
        }
        // Per-token rope positions -> BR_POS (T i32).
        {
            let pos: Vec<i32> = (0..t).map(|i| (start_pos + i) as i32).collect();
            let mut bytes = Vec::with_capacity(t * 4);
            for p in &pos { bytes.extend_from_slice(&p.to_le_bytes()); }
            self.bres_bufs[BR_POS].write(&bytes)?;
        }
        prof_add("batched_host_prep", host_prep_ts);

        let rope_pc_q = rope_neox_pc(num_q, head_dim, head_dim, head_dim, cfg.rope_theta);
        let rope_pc_k = rope_neox_pc(num_kv, head_dim, head_dim, head_dim, cfg.rope_theta);
        let rope_wgy = (((head_dim / 2) as u32) + 255) / 256;
        let sdpa_kernel = attn_decode_kernel();

        let ha = self.bres_ptr(BR_HA);
        let hb = self.bres_ptr(BR_HB);
        let xp = self.bres_ptr(BR_X);
        let qp = self.bres_ptr(BR_Q);
        let kp = self.bres_ptr(BR_K);
        let vp = self.bres_ptr(BR_V);
        let posp = self.bres_ptr(BR_POS);
        let ffp = self.bres_ptr(BR_FF);
        let idxp = self.bres_ptr(BR_IDX);
        let attnp = self.bres_ptr(BR_ATTN);
        let op = self.bres_ptr(BR_O);
        let ffin = self.bres_ptr(BR_FFIN);
        let gate = self.bres_ptr(BR_GATE);
        let up = self.bres_ptr(BR_UP);
        let act = self.bres_ptr(BR_ACT);
        let mid = self.bres_ptr(BR_MID);
        let down = self.bres_ptr(BR_DOWN);
        let dummy = self.bres_ptr(BR_DUMMY);

        let add_pc_h = ew_mul_pc((t * h) as u32);
        let act_pc = ew_unary_pc((t * inter) as u32);
        let mul_pc = ew_mul_pc((t * inter) as u32);
        let rms_h = rmsnorm_pc(h, eps);
        let rms_hd = rmsnorm_pc(head_dim, eps);
        let add_wg_h = ((t * h) as u32 + 255) / 256;
        let act_wg = ((t * inter) as u32 + 511) / 512;
        let mul_wg = ((t * inter) as u32 + 255) / 256;

        // Per-layer projections: batched multi-column matvec (`_r{rows}_c{t}`,
        // rows*t<=8). Each workgroup streams its weight rows ONCE across all T
        // token-columns — the bn8 GEMM ran these at 16-48 workgroups on GFX1013
        // (occupancy-starved, ~18GB/s) vs the matvec family's ~185GB/s. Column j
        // reads B at j*batch_stride_b(=k) and writes D at j*batch_stride_d(=n),
        // exactly the row-major [T,k]->[T,n] layout of the BR_* buffers, and
        // matvec_pc13 already encodes those strides. Bit-exact per column vs the
        // serial decode matvec (same kernel/accumulation order). The LM head is
        // NOT on this kernel — it is T single-column r8 matvecs, see the final
        // block below.
        //
        // The shader is picked PER WEIGHT (`bres_proj` -> its own recorded
        // format), not from the global `VLLM_VULKAN_QUANT` snapshot; see
        // `bres_proj`'s doc comment for why those two diverge in practice.
        let mv_q = matvec_pc13(h, q_dim);
        let mv_kv = matvec_pc13(h, kv_dim);
        let mv_o = matvec_pc13(q_dim, h);
        let mv_gu = matvec_pc13(h, inter);
        let mv_d = matvec_pc13(inter, h);

        for layer_idx in 0..cfg.num_hidden_layers {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

            // Norm weights (stable pointers once uploaded; reused every call).
            if !self.ensure_unified_norm(&ln("input_layernorm.weight"), h, true)
                || !self.ensure_unified_norm(&ln("self_attn.q_norm.weight"), head_dim, true)
                || !self.ensure_unified_norm(&ln("self_attn.k_norm.weight"), head_dim, true)
                || !self.ensure_unified_norm(&ln("post_attention_layernorm.weight"), h, true)
            {
                return Err(format!("forward_batched: missing norm weight for layer {layer_idx}").into());
            }

            // KV bookkeeping (qwen dense: every layer appends its own KV).
            self.qwen.as_mut().unwrap().kv_caches[layer_idx].seq_len = start_pos + t;
            let kv_ptr = match self.ensure_gpu_kv(layer_idx, num_kv, head_dim, max_seq) {
                Some(p) => p,
                None => return Err("forward_batched: resident KV buffer unavailable".to_string().into()),
            };
            let plane = max_seq * kv_dim;

            // Gather every buffer/weight pointer BEFORE recording (no map
            // inserts mid-record — same discipline as `gpu_layer_1cb`).
            let inln_p = &self.unified_norm_w[&ln("input_layernorm.weight")] as *const compute::Buffer;
            let qnorm_p = &self.unified_norm_w[&ln("self_attn.q_norm.weight")] as *const compute::Buffer;
            let knorm_p = &self.unified_norm_w[&ln("self_attn.k_norm.weight")] as *const compute::Buffer;
            let ffn_in_p = &self.unified_norm_w[&ln("post_attention_layernorm.weight")] as *const compute::Buffer;
            let (qw, q_fmt) = self.bres_proj(&ln("self_attn.q_proj.weight"), layer_idx)?;
            let (kw, k_fmt) = self.bres_proj(&ln("self_attn.k_proj.weight"), layer_idx)?;
            let (vw, v_fmt) = self.bres_proj(&ln("self_attn.v_proj.weight"), layer_idx)?;
            let (ow, o_fmt) = self.bres_proj(&ln("self_attn.o_proj.weight"), layer_idx)?;
            let (gw, g_fmt) = self.bres_proj(&ln("mlp.gate_proj.weight"), layer_idx)?;
            let (uw, u_fmt) = self.bres_proj(&ln("mlp.up_proj.weight"), layer_idx)?;
            let (dw, d_fmt) = self.bres_proj(&ln("mlp.down_proj.weight"), layer_idx)?;

            // One (shader, rows) pair per projection, keyed on THAT weight's
            // format. `rows` feeds the workgroup count, so it must come from
            // the same call that chose the shader.
            let (mv_n_q, r_q) = matvec_cols_variant_by_format(q_fmt, h, t);
            let (mv_n_k, r_k) = matvec_cols_variant_by_format(k_fmt, h, t);
            let (mv_n_v, r_v) = matvec_cols_variant_by_format(v_fmt, h, t);
            let (mv_n_o, r_o) = matvec_cols_variant_by_format(o_fmt, h, t);
            let (mv_n_g, r_g) = matvec_cols_variant_by_format(g_fmt, h, t);
            let (mv_n_u, r_u) = matvec_cols_variant_by_format(u_fmt, h, t);
            let (mv_n_d, r_d) = matvec_cols_variant_by_format(d_fmt, h, t);
            let wg_q = ((q_dim as u32).div_ceil(r_q), 1u32, 1u32);
            let wg_k = ((kv_dim as u32).div_ceil(r_k), 1u32, 1u32);
            let wg_v = ((kv_dim as u32).div_ceil(r_v), 1u32, 1u32);
            let wg_o = ((h as u32).div_ceil(r_o), 1u32, 1u32);
            let wg_g = ((inter as u32).div_ceil(r_g), 1u32, 1u32);
            let wg_u = ((inter as u32).div_ceil(r_u), 1u32, 1u32);
            let wg_d = ((h as u32).div_ceil(r_d), 1u32, 1u32);

            let sdpa_wg = match sdpa_kernel {
                "paged_attn_decode_f32_sg"   => (num_q as u32, 1u32, 1u32),
                "paged_attn_decode_f32_coop" => (num_q as u32, (head_dim as u32 + 255) / 256, 1u32),
                _                            => ((q_dim as u32 + 255) / 256, 1u32, 1u32),
            };

            let k_dst_off = (start_pos * kv_dim * 4) as u64;
            let v_dst_off = ((plane + start_pos * kv_dim) * 4) as u64;
            let kv_copy_sz = (t * kv_dim * 4) as u64;

            let eng = self.engine.as_mut().expect("invariant: checked engine.is_some() above");
            let cb = eng.begin_batch()?;
            unsafe {
                // input_layernorm, row-batched over [T,h]: (T,1,1).
                eng.record_to(cb, "rms_norm_f32_mul", &[&*ha, &*inln_p, &*xp], &rms_h, (t as u32, 1, 1))?;
                eng.record_barrier_to(cb);
                // q/k/v projections — multi-column matvec, weight rows read
                // ONCE per workgroup and reused across all T token-columns.
                eng.record_to(cb, &mv_n_q, &[&*qw, &*xp, &*qp], &mv_q, wg_q)?;
                eng.record_to(cb, &mv_n_k, &[&*kw, &*xp, &*kp], &mv_kv, wg_k)?;
                eng.record_to(cb, &mv_n_v, &[&*vw, &*xp, &*vp], &mv_kv, wg_v)?;
                eng.record_barrier_to(cb);
                // q/k-norm, row-batched over all heads x tokens (Q/K are
                // row-major [T,num_*,head_dim] == contiguous [T*num_*,head_dim]).
                eng.record_to(cb, "rms_norm_f32_mul", &[&*qp, &*qnorm_p, &*qp], &rms_hd, ((t * num_q) as u32, 1, 1))?;
                eng.record_to(cb, "rms_norm_f32_mul", &[&*kp, &*knorm_p, &*kp], &rms_hd, ((t * num_kv) as u32, 1, 1))?;
                eng.record_barrier_to(cb);
                // RoPE, per token (position start_pos+i via BR_POS[i]).
                for i in 0..t {
                    let q_off = (i * q_dim * 4) as u64;
                    let k_off = (i * kv_dim * 4) as u64;
                    let pos_off = (i * 4) as u64;
                    eng.record_to_off(cb, "rope_neox_f32_f32",
                        &[(&*qp, q_off), (&*posp, pos_off), (&*ffp, 0), (&*qp, q_off), (&*idxp, 0)],
                        &rope_pc_q, (num_q as u32, rope_wgy, 1))?;
                    eng.record_to_off(cb, "rope_neox_f32_f32",
                        &[(&*kp, k_off), (&*posp, pos_off), (&*ffp, 0), (&*kp, k_off), (&*idxp, 0)],
                        &rope_pc_k, (num_kv as u32, rope_wgy, 1))?;
                }

                // RoPE'd Q (SHADER_WRITE, COMPUTE) feeds the per-token SDPA
                // dispatches below (SHADER_READ, COMPUTE). The
                // COMPUTE→TRANSFER + TRANSFER→COMPUTE pair below covers only
                // K/V: it makes TRANSFER_WRITEs visible to SHADER_READ, never
                // the COMPUTE SHADER_WRITE of Q.
                eng.record_barrier_to(cb);
                // ── Resident-KV append: T positions are CONTIGUOUS -> ONE
                // copy each for K and V (RoPE wrote them, COMPUTE; the copy
                // reads them, TRANSFER).
                eng.record_compute_to_transfer_barrier(cb);
                eng.record_copy_to(cb, &*kp, &*kv_ptr, 0, k_dst_off, kv_copy_sz);
                eng.record_copy_to(cb, &*vp, &*kv_ptr, 0, v_dst_off, kv_copy_sz);
                eng.record_transfer_to_compute_barrier(cb);

                // Attention: T single-query resident-KV decode dispatches,
                // causal via per-query seq_len = start_pos+i+1.
                for i in 0..t {
                    let q_off = (i * q_dim * 4) as u64;
                    let attn_off = (i * q_dim * 4) as u64;
                    let pc = sdpa_pc(start_pos + i + 1, num_q, num_kv, head_dim, max_seq, plane, scale, 0, 0);
                    eng.record_to_off(cb, sdpa_kernel,
                        &[(&*qp, q_off), (&*idxp, 0), (&*kv_ptr, 0), (&*attnp, attn_off)],
                        &pc, sdpa_wg)?;
                }
                eng.record_barrier_to(cb);

                // o_proj matvec -> residual add -> ffn_in_norm -> FFN -> residual2.
                eng.record_to(cb, &mv_n_o, &[&*ow, &*attnp, &*op], &mv_o, wg_o)?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "add_f32_f32_f32", &[&*ha, &*op, &*hb, &*dummy], &add_pc_h, (add_wg_h, 1, 1))?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "rms_norm_f32_mul", &[&*hb, &*ffn_in_p, &*ffin], &rms_h, (t as u32, 1, 1))?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, &mv_n_g, &[&*gw, &*ffin, &*gate], &mv_gu, wg_g)?;
                eng.record_to(cb, &mv_n_u, &[&*uw, &*ffin, &*up], &mv_gu, wg_u)?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "silu_f32", &[&*gate, &*act], &act_pc, (act_wg, 1, 1))?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "mul_f32_f32_f32", &[&*act, &*up, &*mid], &mul_pc, (mul_wg, 1, 1))?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, &mv_n_d, &[&*dw, &*mid, &*down], &mv_d, wg_d)?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "add_f32_f32_f32", &[&*hb, &*down, &*ha, &*dummy], &add_pc_h, (add_wg_h, 1, 1))?;
            }
            let ts = std::time::Instant::now();
            eng.submit_batch(cb)?;
            prof_add("batched_layer_submit_fence", ts);
        }

        // ── Final norm + LM head: T single-column r8 matvecs (one per token),
        // all T rows returned (a real k-token verify needs logits at every
        // provisional position, unlike `forward_prefill`'s last-row-only). ──
        if !self.ensure_unified_norm("model.norm.weight", h, true) {
            return Err("forward_batched: missing model.norm.weight".to_string().into());
        }
        let lm_name = self.qwen.as_ref().unwrap().lm_head_name.clone();
        // The LM head dispatch is keyed on the uploaded weight's OWN
        // `format`/`aux`, exactly like the resident per-op path
        // (`gemma_forward::gemma_res_mv_kind`, added by 22ee4a9). It used to
        // call the global-quant-keyed `matvec_variant` and bind only
        // (weight, input, output): for an Mlx4-packed head — which the
        // gemma4_unified loader produces under VLLM_VULKAN_GEMMA_LMHEAD_Q4 —
        // that picks a plain f16 matvec, leaves the mandatory `scales` and
        // `biases` buffers unbound, and reads 4-bit nibbles as f16. NaN
        // logits, no error.
        let (lm_fmt, lm_kind, lm_w) = {
            let lm_gw = self.gpu_weights.get(&lm_name)
                .ok_or_else(|| "forward_batched: missing lm_head weight".to_string())?;
            let (f, k) = gemma_res_mv_kind(lm_gw);
            // Raw pointers into `gpu_weights`, which is not mutated between
            // here and the record loop (same discipline as the per-layer
            // projections above).
            (f, k, &lm_gw.buffer as *const compute::Buffer)
        };
        let norm_w_p = &self.unified_norm_w["model.norm.weight"] as *const compute::Buffer;
        let ha = self.bres_ptr(BR_HA);
        let normp = self.bres_ptr(BR_NORM);
        let logitsp = self.bres_ptr(BR_LOGITS);
        // The M=vocab GEMM ran at the bn8 mul_mm bandwidth ceiling (~50GB/s,
        // 311MB f16 weight → ~6.4ms). A single-column matvec over M=vocab needs
        // rows LARGE enough to stay under the 65535-workgroup limit: r8 →
        // ceil(151936/8)=18992 wg (the SAME variant the per-token decode lm_head
        // already uses). The per-layer multi-COLUMN _r*_c* variants can't serve
        // the lm_head — rows*cols<=8 forces rows<8 for cols>1 (rows=2 at cols=3 →
        // 75968 wg > 65535). So loop T INDEPENDENT r8 dispatches, each slicing
        // token j's normed-hidden input row (j*h) and logits output row (j*vocab)
        // via `record_gemma_mv_off`. This reuses the EXACT decode lm_head
        // recorder (shader, bindings and push constants all chosen by
        // `record_gemma_mv`'s own match), so every column is bit-identical to a
        // serial decode by construction, for packed and plain heads alike.
        // The weight streams once per dispatch (T reads) but at the matvec
        // family's ~185GB/s vs the GEMM's one read at its ~50GB/s occupancy
        // ceiling.

        let lmhead_submit_ts = std::time::Instant::now();
        let eng = self.engine.as_mut().expect("invariant: checked engine.is_some() above");
        let cb = eng.begin_batch()?;
        unsafe {
            eng.record_to(cb, "rms_norm_f32_mul", &[&*ha, &*norm_w_p, &*normp], &rms_h, (t as u32, 1, 1))?;
            eng.record_barrier_to(cb);
            // T independent single-column matvecs — no inter-dispatch barrier
            // (each writes a disjoint logits output row).
            for j in 0..t {
                let in_off = (j * h * 4) as u64;
                let out_off = (j * vocab * 4) as u64;
                record_gemma_mv_off(eng, cb, lm_w, lm_fmt, lm_kind,
                    normp, in_off, logitsp, out_off, h, vocab);
            }
        }
        eng.submit_batch(cb)?;
        prof_add("batched_lmhead_submit", lmhead_submit_ts);
        let logits_readback_ts = std::time::Instant::now();
        let logits = read_f32_buf(&self.bres_bufs[BR_LOGITS], t * vocab);
        prof_add("batched_logits_readback", logits_readback_ts);
        Ok(logits)
    }
}
