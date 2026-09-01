// SPDX-License-Identifier: Apache-2.0
//! Qwen3 (`qwen_*`) and Qwen3.5/3.6 (`qwen35_*`) GPU forward paths — same
//! family, two generations of the format. Extracted verbatim from lib.rs
//! (M1).

use crate::compute;
use crate::model;
use crate::moe;
#[cfg(feature = "qwen35")]
use crate::qwen35;
use crate::gpu_error::GpuResult;
use crate::VulkanModel;
use crate::{DnGpuLayer, MoeGpuLayer, MoeGpuProj, MoeSharedGpu, MvKind, QuantAux, SpecSlot};
use crate::{
    matvec_pc13, matvec_variant, matvec_variant_geom, matvec_variant_q35geom, matvec_mlx4_pc, matvec_mlx4_pc_off,
    matvec_mlx4_variant_k,
    nvfp4_dispatch, matvec_fp8_variant, matvec_fp8_pc, matvec_f32_variant,
    matvec_f32_variant_k,
    gemm_pc, gemm_variant_k, gemm_pc_mlx4, gemm_pc_mlx4_id, gemm_pc_mlx4_id_gateup,
    gemm_variant_quant_k, gemm_variant_quant_id_bn, gemm_quant_flag,
    ew_unary_pc, ew_mul_pc, rmsnorm_pc, rope_neox_pc,
    q35_conv_pc, q35_qknorm_pc, q35_gdn_pc, q35_gdn_scan_pc, q35_moe_accum_pc,
    q35_moe_accum_batched_pc, glu_split_pc,
    f32_slice_to_bytes, read_f32_buf,
};
use crate::{dn_gpu_enabled, moe_gpu_enabled, moe_gemm_enabled, moe_gemm_fused_enabled, moe_gemm_combine_enabled, q35_1cb_enabled, q35_geom_enabled, q35_tp_fused_enabled, q35_gpu_attn_enabled, q35_tstamp_enabled, q35_kv_boundary_diag_enabled, par_deltanet, prof_add, prof_add_ns};
use crate::flags;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const QR_HA:    usize = 0;  // hidden A (layer input / ffn-add output)   [h]
const QR_HB:    usize = 1;  // hidden B (attn-add output / ffn residual) [h]
const QR_X:     usize = 2;  // normed hidden (input/final norm output)   [h]
const QR_Q:     usize = 3;  // q proj (then q-norm, rope, in place)      [q_dim]
const QR_K:     usize = 4;  // k proj (then k-norm, rope, in place)      [kv_dim]
const QR_V:     usize = 5;  // v proj                                    [kv_dim]
const QR_ATTN:  usize = 6;  // attention output (host sdpa → uploaded)   [q_dim]
const QR_O:     usize = 7;  // o proj output                            [h]
const QR_FFIN:  usize = 8;  // post-attn norm output (ffn input)        [h]
const QR_GATE:  usize = 9;  // gate proj output                         [inter]
const QR_UP:    usize = 10; // up proj output                           [inter]
const QR_SILU:  usize = 11; // silu(gate) output                        [inter]
const QR_MID:   usize = 12; // silu(gate)*up                            [inter]
const QR_DOWN:  usize = 13; // down proj output                         [h]
const QR_POS:   usize = 14; // rope position (1 int)
const QR_FF:    usize = 15; // rope freq-factors dummy (1 f32)
const QR_IDX:   usize = 16; // rope set_rows idx dummy (2 u32)
const QR_LOGITS:usize = 17; // final lm_head output                     [vocab]
const QR_COUNT: usize = 18;

// Slot indices for the WS3 qwen3.6 resident stage buffers (q35res_bufs).
// Hidden/attn/MoE intermediates stay in these persistent device buffers
// between layers AND tokens; the residual stream lives in Q35R_H and is only
// read back at the stage boundary (or never, on the lm_head tail).
const Q35R_H:     usize = 0;  // hidden / residual stream                [h]
const Q35R_X:     usize = 1;  // input-norm output                       [h]
const Q35R_ATTN:  usize = 2;  // attn block output (out_proj / o_proj)   [h]
const Q35R_H1:    usize = 3;  // post-attn residual (H + ATTN)           [h]
const Q35R_FFIN:  usize = 4;  // post-attn-norm output (MoE input)       [h]
const Q35R_RLOG:  usize = 5;  // router gate logits                      [E]
const Q35R_QKV:   usize = 6;  // deltanet in_proj_qkv output             [conv_dim]
const Q35R_Z:     usize = 7;  // deltanet in_proj_z output               [value_dim]
const Q35R_A:     usize = 8;  // deltanet in_proj_a output               [nv]
const Q35R_B:     usize = 9;  // deltanet in_proj_b output               [nv]
const Q35R_CONV:  usize = 10; // conv1d+SiLU output                      [conv_dim]
const Q35R_Q:     usize = 11; // normed+scaled q                         [key_dim]
const Q35R_K:     usize = 12; // normed+scaled k                         [key_dim]
const Q35R_GATED: usize = 13; // delta-rule gated output                 [value_dim]
const Q35R_QG:    usize = 14; // full-attn q_proj [query|gate] output    [2*q_dim]
const Q35R_AK:    usize = 15; // full-attn k_proj output                 [kv_dim]
const Q35R_AV:    usize = 16; // full-attn v_proj output                 [kv_dim]
const Q35R_GIN:   usize = 17; // full-attn gated SDPA out (host upload)  [q_dim]
const Q35R_SG:    usize = 18; // shared expert gate proj                 [si]
const Q35R_SU:    usize = 19; // shared expert up proj                   [si]
const Q35R_SA:    usize = 20; // silu(shared gate)                       [si]
const Q35R_SM:    usize = 21; // shared mid                              [si]
const Q35R_SO:    usize = 22; // shared expert down proj                 [h]
const Q35R_SL:    usize = 23; // shared_expert_gate logit                [1]
const Q35R_NORMED:usize = 24; // final-norm output (lm tail)             [h]
const Q35R_VLOG:  usize = 25; // lm_head logits (pp_last only)           [vocab]
const Q35R_GU0:   usize = 26; // routed expert gate/up: gate=GU0+2*slot, up=+1 (16 × [mi])
const Q35R_ACT0:  usize = 42; // silu(gate) per routed expert            (8 × [mi])
const Q35R_MID0:  usize = 50; // act*up per routed expert                (8 × [mi])
const Q35R_DOWN0: usize = 58; // down proj per routed expert             (8 × [h])
const Q35R_COUNT: usize = 66;

/// Column-tiling schedule for the batched cols matvec: split `t` activation
/// columns into contiguous `(c0, ct)` tiles of at most `tile` columns each
/// (`2 <= ct <= tile`). A lone trailing column (`ct == 1`) is folded back into
/// the previous tile by stepping `c0` back one and widening to `ct = 2` — the
/// overlapped column is recomputed with the identical (idempotent) result, so
/// every tile stays `>= 2` (the cols kernel requires `t >= 2`). The tiles are
/// COLUMN-DISJOINT in their output writes except for that single idempotent
/// overlap, and their union covers `[0, t)`, so writing each tile's `[ct, n]`
/// block at column offset `c0` reconstructs the full `[t, n]` output identically
/// to computing every column independently. REQUIRES `t >= 2` (callers guard
/// `t < 2`; `t == 1` would underflow the `c0 -= 1` fold). Shared by
/// `qwen35_matvec_cols_tiled` (the live cols path) and the gemma prefill cols
/// wiring; extracted so the decomposition is host-testable without a GPU.
pub(crate) fn cols_tile_schedule(t: usize, tile: usize) -> Vec<(usize, usize)> {
    debug_assert!(t >= 2 && tile >= 2);
    let mut sched = Vec::new();
    let mut c0 = 0usize;
    while c0 < t {
        let mut ct = (t - c0).min(tile);
        if ct == 1 {
            c0 -= 1;
            ct = 2;
        }
        sched.push((c0, ct));
        c0 += ct;
    }
    sched
}

// ── Base Qwen3 dense GPU path (`qwen_*`) — the `gemma`/base-dense feature ────
#[cfg(feature = "gemma")]
impl VulkanModel {
    /// Copy a Qwen3 weight tensor (CPU reference copy) into an owned Vec.
    pub(crate) fn qwen_w(&self, name: &str) -> Vec<f32> {
        self.qwen.as_ref().unwrap().weights.f32_slice(name).to_vec()
    }
    /// Single matrix-vector product `[1, n] = x[1, k] @ W[n, k]^T`.
    /// Runs on the GPU when the weight is resident there, else CPU.
    /// `f16_weight` selects the f16- vs f32-weight shader variant to match how
    /// the weight was uploaded (projection weights are f16; embed/lm_head f32).
    pub(crate) fn qwen_matvec(&mut self, weight_name: &str, x: &[f32], k: usize, n: usize, f16_weight: bool) -> Vec<f32> {
        if let (Some(eng), Some(w_ptr)) = (
            self.engine.as_mut(),
            self.gpu_weights.get(weight_name).map(|w| &w.buffer as *const compute::Buffer),
        ) {
            let th = std::time::Instant::now();
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
            prof_add("mv_host_in", th);
            let ts = std::time::Instant::now();
            eng.submit_batch(cb).unwrap();
            prof_add("mv_submit_fence", ts);
            let tr = std::time::Instant::now();
            let result = read_f32_buf(&out, n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
            prof_add("mv_host_out", tr);
            result
        } else {
            let w = self.qwen.as_ref().unwrap().weights.f32_slice(weight_name);
            model::cpu_matmul(x, w, 1, k, n)
        }
    }
    /// Batched matvec: several weight matrices that all read the SAME input `x`
    /// (and share contraction dim `k`) are dispatched into ONE command buffer and
    /// submitted with a SINGLE fence-wait, instead of one submit per matmul.
    /// On GFX1013 each `submit_batch` blocks the GPU to idle, so collapsing the
    /// independent q/k/v (and gate/up) projections from N submits to 1 removes the
    /// dominant per-token overhead. `jobs` = (weight_name, output_dim n). Returns
    /// one output vector per job, in order.
    pub(crate) fn qwen_matvec_multi(&mut self, x: &[f32], k: usize, jobs: &[(&str, usize)], f16_weight: bool) -> Vec<Vec<f32>> {
        let w_ptrs: Option<Vec<*const compute::Buffer>> = jobs.iter()
            .map(|(name, _)| self.gpu_weights.get(*name).map(|w| &w.buffer as *const compute::Buffer))
            .collect();
        if let (Some(eng), Some(w_ptrs)) = (self.engine.as_mut(), w_ptrs) {
            let th = std::time::Instant::now();
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let outs: Vec<compute::Buffer> = jobs.iter()
                .map(|(_, n)| eng.alloc_host_coherent_storage((*n * 4) as u64).unwrap())
                .collect();
            let cb = eng.begin_batch().unwrap();
            for (i, (_, n)) in jobs.iter().enumerate() {
                let (shader, r) = matvec_variant(f16_weight, *n);
                let wg = (*n as u32 + r - 1) / r;
                let pc = matvec_pc13(k, *n);
                let out_p = &outs[i] as *const compute::Buffer;
                unsafe {
                    eng.record_to(cb, &shader, &[&*w_ptrs[i], &*inp_p, &*out_p], &pc, (wg, 1, 1)).unwrap();
                }
            }
            prof_add("mv_host_in", th);
            let ts = std::time::Instant::now();
            eng.submit_batch(cb).unwrap();
            prof_add("mv_submit_fence", ts);
            let tr = std::time::Instant::now();
            let results: Vec<Vec<f32>> = jobs.iter().enumerate()
                .map(|(i, (_, n))| read_f32_buf(&outs[i], *n))
                .collect();
            eng.return_to_pool(inp);
            for o in outs { eng.return_to_pool(o); }
            prof_add("mv_host_out", tr);
            results
        } else {
            jobs.iter().map(|(name, n)| {
                let w = self.qwen.as_ref().unwrap().weights.f32_slice(name);
                model::cpu_matmul(x, w, 1, k, *n)
            }).collect()
        }
    }
    /// GPU-accelerated Qwen3 forward pass for one token.
    ///
    /// Mirrors `model::Qwen3Model::forward` exactly, but routes the large
    /// projection matmuls to the GPU.  Norms, Q/K-norm, RoPE and attention run
    /// on CPU (cheap relative to the projections), matching the Gemma4 path.
    pub(crate) fn forward_qwen_gpu(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        self.forward_qwen_gpu_cap(token_id, pos, None)
    }
    /// As `forward_qwen_gpu`, but optionally records the hidden state after each
    /// decoder layer into `cap` (debug: localise GPU vs reference divergence).
    pub(crate) fn forward_qwen_gpu_cap(&mut self, token_id: u32, pos: usize, mut cap: Option<&mut Vec<Vec<f32>>>) -> Vec<f32> {
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

        // Embedding (no scaling).
        let mut hidden: Vec<f32> = {
            let w = self.qwen.as_ref().unwrap().weights.f32_slice("model.embed_tokens.weight");
            w[token_id as usize * h..(token_id as usize + 1) * h].to_vec()
        };

        for layer_idx in 0..cfg.num_hidden_layers {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

            // ── Attention ──────────────────────────────────────────────────
            let residual = hidden.clone();
            let tw = std::time::Instant::now();
            let inln = self.qwen_w(&ln("input_layernorm.weight"));
            prof_add("host_weight_fetch", tw);
            let tn = std::time::Instant::now();
            let x = model::cpu_rms_norm(&hidden, &inln, eps);
            prof_add("cpu_rmsnorm", tn);

            let (qn, kn, vn) = (ln("self_attn.q_proj.weight"), ln("self_attn.k_proj.weight"), ln("self_attn.v_proj.weight"));
            let mut qkv = self.qwen_matvec_multi(&x, h, &[(&qn, q_dim), (&kn, kv_dim), (&vn, kv_dim)], true);
            let v = qkv.pop().unwrap();
            let mut k = qkv.pop().unwrap();
            let mut q = qkv.pop().unwrap();

            let tw2 = std::time::Instant::now();
            let q_norm = self.qwen_w(&ln("self_attn.q_norm.weight"));
            let k_norm = self.qwen_w(&ln("self_attn.k_norm.weight"));
            prof_add("host_weight_fetch", tw2);
            let tqk = std::time::Instant::now();
            for hi in 0..num_q {
                let s = &mut q[hi * head_dim..(hi + 1) * head_dim];
                let n = model::cpu_rms_norm(s, &q_norm, eps);
                s.copy_from_slice(&n);
            }
            for hi in 0..num_kv {
                let s = &mut k[hi * head_dim..(hi + 1) * head_dim];
                let n = model::cpu_rms_norm(s, &k_norm, eps);
                s.copy_from_slice(&n);
            }
            prof_add("cpu_qknorm", tqk);

            let tr = std::time::Instant::now();
            model::cpu_rope(&mut q, &mut k, pos, num_q, num_kv, head_dim, head_dim, cfg.rope_theta);
            prof_add("cpu_rope", tr);

            let tsd = std::time::Instant::now();
            let attn_out = {
                let qm = self.qwen.as_mut().unwrap();
                qm.kv_caches[layer_idx].append(&k, &v);
                let cache = &qm.kv_caches[layer_idx];
                model::cpu_sdpa(
                    &q, cache.k_up_to_now(), cache.v_up_to_now(),
                    num_q, num_kv, head_dim, cache.seq_len, scale, None,
                )
            };
            prof_add("cpu_sdpa", tsd);

            let attn_proj = self.qwen_matvec(&ln("self_attn.o_proj.weight"), &attn_out, q_dim, h, true);
            let hidden2: Vec<f32> = residual.iter().zip(attn_proj.iter())
                .map(|(&r, &a)| r + a).collect();
            let residual2 = hidden2.clone();

            // ── MLP (SwiGLU) ───────────────────────────────────────────────
            let pa = self.qwen_w(&ln("post_attention_layernorm.weight"));
            let ff_in = model::cpu_rms_norm(&hidden2, &pa, eps);
            let (gn, un) = (ln("mlp.gate_proj.weight"), ln("mlp.up_proj.weight"));
            let mut gu = self.qwen_matvec_multi(&ff_in, h, &[(&gn, inter), (&un, inter)], true);
            let up = gu.pop().unwrap();
            let gate = gu.pop().unwrap();
            let gate_act = model::cpu_silu(&gate);
            let mid: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
            let ff_out = self.qwen_matvec(&ln("mlp.down_proj.weight"), &mid, inter, h, true);

            hidden = residual2.iter().zip(ff_out.iter()).map(|(&r, &f)| r + f).collect();
            if let Some(c) = cap.as_deref_mut() {
                c.push(hidden.clone());
            }
        }

        // Final norm + LM head (no softcapping).  The LM head is kept on the
        // CPU at full f32 precision: it is the largest and most precision-
        // sensitive projection, and its host f32 copy is retained even in
        // lean-host mode (embeddings/lm_head are never dropped).
        let norm_w = self.qwen_w("model.norm.weight");
        let normed = model::cpu_rms_norm(&hidden, &norm_w, eps);
        let lm_name = self.qwen.as_ref().unwrap().lm_head_name.clone();
        self.qwen_matvec(&lm_name, &normed, h, vocab, true)
    }
    /// GPU-resident SwiGLU FFN (Qwen): gate+up matmuls → silu → mul → down all
    /// chain through GPU buffers in ONE submit — no host round-trips for
    /// gate/up/mid and no CPU silu/mul (vs the current per-matvec read-back +
    /// CPU elementwise path). x is [t,h]; returns [t,h]. Falls back to CPU.
    pub(crate) fn qwen_ffn_gpu(&mut self, x: &[f32], gate_name: &str, up_name: &str, down_name: &str,
                    t: usize, h: usize, inter: usize) -> Vec<f32> {
        let gw = self.gpu_weights.get(gate_name).map(|w| &w.buffer as *const compute::Buffer);
        let uw = self.gpu_weights.get(up_name).map(|w| &w.buffer as *const compute::Buffer);
        let dw = self.gpu_weights.get(down_name).map(|w| &w.buffer as *const compute::Buffer);
        if let (Some(eng), Some(gw), Some(uw), Some(dw)) = (self.engine.as_mut(), gw, uw, dw) {
            let a = |eng: &mut compute::ComputeEngine, sz: usize| eng.alloc_host_coherent_storage((sz * 4) as u64).unwrap();
            let ffi = a(eng, t * h); ffi.write(&f32_slice_to_bytes(x)).unwrap();
            let gp = a(eng, t * inter); let sp = a(eng, t * inter); let up = a(eng, t * inter);
            let mid = a(eng, t * inter); let down = a(eng, t * h);
            let (ffi_p, gp_p, sp_p, up_p, mid_p, dn_p) = (
                &ffi as *const compute::Buffer, &gp as *const compute::Buffer, &sp as *const compute::Buffer,
                &up as *const compute::Buffer, &mid as *const compute::Buffer, &down as *const compute::Buffer);
            let elem = (t * inter) as u32;
            let (silu_pc, mul_pc) = (ew_unary_pc(elem), ew_mul_pc(elem));
            // Use the tiled GEMM for the matmuls — correct for any t (the
            // matvec batch dim is unvalidated for t>1). out[t,N] = in[t,K] @
            // W[N,K]^T -> gemm_pc(t, N, K), bindings [W, in, out], wg
            // (ceil(N/BM), ceil(t/BN), 1). Default matmul_f16_f32_fp32
            // (BM=BN=64), or the GEMM-campaign Phase 1 winner on the swept
            // (K,N) shapes -- see gemm_variant_k.
            let gpc = gemm_pc(t, inter, h);   // gate/up: N=inter, K=h
            let dpc = gemm_pc(t, h, inter);   // down:    N=h,     K=inter
            let (g_variant, g_bm, g_bn) = gemm_variant_k(h, inter);
            let (d_variant, d_bm, d_bn) = gemm_variant_k(inter, h);
            let gwg = (((inter + g_bm as usize - 1) / g_bm as usize) as u32,
                       ((t + g_bn as usize - 1) / g_bn as usize) as u32, 1u32);
            let dwg = (((h + d_bm as usize - 1) / d_bm as usize) as u32,
                       ((t + d_bn as usize - 1) / d_bn as usize) as u32, 1u32);
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, &g_variant, &[&*gw, &*ffi_p, &*gp_p], &gpc, gwg).unwrap();
                eng.record_to(cb, &g_variant, &[&*uw, &*ffi_p, &*up_p], &gpc, gwg).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "silu_f32", &[&*gp_p, &*sp_p], &silu_pc, ((elem + 511)/512, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "mul_f32_f32_f32", &[&*sp_p, &*up_p, &*mid_p], &mul_pc, ((elem + 255)/256, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, &d_variant, &[&*dw, &*mid_p, &*dn_p], &dpc, dwg).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            let out = read_f32_buf(&down, t * h);
            for b in [ffi, gp, sp, up, mid, down] { eng.return_to_pool(b); }
            out
        } else {
            let gate = model::cpu_matmul(x, self.qwen.as_ref().unwrap().weights.f32_slice(gate_name), t, h, inter);
            let up = model::cpu_matmul(x, self.qwen.as_ref().unwrap().weights.f32_slice(up_name), t, h, inter);
            let act = model::cpu_silu(&gate);
            let mid: Vec<f32> = act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
            model::cpu_matmul(&mid, self.qwen.as_ref().unwrap().weights.f32_slice(down_name), t, inter, h)
        }
    }
    /// GPU RMSNorm with weight: out[r,c] = x[r,c] * rsqrt(mean(x[r]^2)+eps) * w[c].
    /// `n_rows` rows of `n_cols` (one workgroup/row). Used both for the input/
    /// post-attn norms (n_rows=t, n_cols=h) and the per-head q/k norms
    /// (n_rows=num_heads, n_cols=head_dim). Matches model::cpu_rms_norm.
    pub(crate) fn qwen_rmsnorm_gpu(&mut self, x: &[f32], weight: &[f32], n_rows: usize,
                        n_cols: usize, eps: f32) -> Vec<f32> {
        if let Some(eng) = self.engine.as_mut() {
            let inp = eng.alloc_host_coherent_storage((n_rows * n_cols * 4) as u64).unwrap();
            inp.write(&f32_slice_to_bytes(x)).unwrap();
            let wbuf = eng.alloc_host_coherent_storage((n_cols * 4) as u64).unwrap();
            wbuf.write(&f32_slice_to_bytes(weight)).unwrap();
            let out = eng.alloc_host_coherent_storage((n_rows * n_cols * 4) as u64).unwrap();
            let (ip, wp, op) = (&inp as *const compute::Buffer, &wbuf as *const compute::Buffer,
                                &out as *const compute::Buffer);
            let pc = rmsnorm_pc(n_cols, eps);
            let cb = eng.begin_batch().unwrap();
            unsafe { eng.record_to(cb, "rms_norm_f32_mul", &[&*ip, &*wp, &*op], &pc, (n_rows as u32, 1, 1)).unwrap(); }
            eng.submit_batch(cb).unwrap();
            let result = read_f32_buf(&out, n_rows * n_cols);
            eng.return_to_pool(inp); eng.return_to_pool(wbuf); eng.return_to_pool(out);
            result
        } else {
            model::cpu_rms_norm(x, weight, eps)
        }
    }
    /// GPU NeoX RoPE applied in place to q/k. q[num_q*head_dim], k[num_kv*head_dim].
    /// rotary_dim==head_dim for Qwen. Output is seeded with the input so the
    /// non-rotary tail (if any) passes through. Matches model::cpu_rope.
    pub(crate) fn qwen_rope_gpu(&mut self, q: &mut [f32], k: &mut [f32], pos: usize, num_q: usize,
                     num_kv: usize, head_dim: usize, rotary_dim: usize, theta: f32) {
        // q — Qwen is always full-rotary here, so freq_dim == rotary_dim.
        self.rope_one(q, pos, num_q, head_dim, rotary_dim, rotary_dim, theta);
        // k
        self.rope_one(k, pos, num_kv, head_dim, rotary_dim, rotary_dim, theta);
    }
    pub(crate) fn qres_ptr(&self, slot: usize) -> *const compute::Buffer {
        &self.qres_bufs[slot] as *const compute::Buffer
    }
    pub(crate) fn qres_ptr_mut(&mut self, slot: usize) -> *mut compute::Buffer {
        &mut self.qres_bufs[slot] as *mut compute::Buffer
    }
    /// Allocate the persistent activation buffers for the fused Qwen layer (once).
    pub(crate) fn init_qres_bufs(&mut self) -> bool {
        if self.qres_ready { return true; }
        let cfg = match self.qwen.as_ref() { Some(q) => q.config.clone(), None => return false };
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let sizes: [u64; QR_COUNT] = [
            (h * 4) as u64,       // QR_HA
            (h * 4) as u64,       // QR_HB
            (h * 4) as u64,       // QR_X
            (q_dim * 4) as u64,   // QR_Q
            (kv_dim * 4) as u64,  // QR_K
            (kv_dim * 4) as u64,  // QR_V
            (q_dim * 4) as u64,   // QR_ATTN
            (h * 4) as u64,       // QR_O
            (h * 4) as u64,       // QR_FFIN
            (inter * 4) as u64,   // QR_GATE
            (inter * 4) as u64,   // QR_UP
            (inter * 4) as u64,   // QR_SILU
            (inter * 4) as u64,   // QR_MID
            (h * 4) as u64,       // QR_DOWN
            4,                    // QR_POS  (1 int)
            4,                    // QR_FF   (1 f32)
            8,                    // QR_IDX  (2 u32)
            (vocab * 4) as u64,   // QR_LOGITS
        ];
        let eng = match self.engine.as_mut() { Some(e) => e, None => return false };
        let mut bufs = Vec::with_capacity(QR_COUNT);
        for &sz in &sizes {
            match eng.alloc_host_coherent_storage(sz) {
                Ok(b) => bufs.push(b),
                Err(e) => { log::warn!("init_qres_bufs alloc failed: {e}"); return false; }
            }
        }
        // Seed rope's dummy freq-factors (1.0) and set_rows idx (0) once.
        bufs[QR_FF].write(&1.0f32.to_le_bytes()).ok();
        bufs[QR_IDX].write(&0u64.to_le_bytes()).ok();
        self.qres_bufs = bufs;
        self.qres_ready = true;
        true
    }
    /// Upload every f32 norm weight the fused Qwen layer reads (input/post-attn/
    /// q/k per layer + final) into `gpu_norm_w` ONCE. Done up-front so no inserts
    /// happen during a forward (which would rehash the map and invalidate the raw
    /// Buffer pointers gathered in the per-layer command-buffer recording).
    /// Returns false if any expected weight is absent.
    pub(crate) fn ensure_qwen_norm_weights(&mut self) -> bool {
        let cfg = self.qwen.as_ref().unwrap().config.clone();
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let mut names: Vec<(String, usize)> = Vec::new();
        for li in 0..cfg.num_hidden_layers {
            names.push((format!("model.layers.{li}.input_layernorm.weight"), h));
            names.push((format!("model.layers.{li}.post_attention_layernorm.weight"), h));
            names.push((format!("model.layers.{li}.self_attn.q_norm.weight"), head_dim));
            names.push((format!("model.layers.{li}.self_attn.k_norm.weight"), head_dim));
        }
        names.push(("model.norm.weight".to_string(), h));
        for (name, n) in names {
            if self.gpu_norm_w.contains_key(&name) { continue; }
            let w = self.qwen.as_ref().unwrap().weights.f32_slice(&name).to_vec();
            if w.len() < n { return false; }
            let eng = self.engine.as_mut().unwrap();
            let buf = match eng.alloc_host_coherent_storage((n * 4) as u64) {
                Ok(b) => b, Err(_) => return false,
            };
            if buf.write(&f32_slice_to_bytes(&w[..n])).is_err() { return false; }
            self.gpu_norm_w.insert(name, buf);
        }
        true
    }
    /// FUSED GPU-resident Qwen3 decode forward (roadmap #2). The whole layer
    /// (input-norm → q/k/v → q/k-norm → RoPE → attention → o → residual →
    /// post-norm → SwiGLU FFN → residual) runs through PERSISTENT GPU buffers,
    /// recorded into just TWO command buffers per layer (split only at the
    /// attention boundary, where K/V must reach the host KV cache + SDPA). Norms
    /// and RoPE are chained INTO those command buffers — not separate submits —
    /// so the per-token submit/round-trip count drops from ~10/layer (the
    /// previous resident version) to 2/layer. Residuals ping-pong HA↔HB so no
    /// buffer copies are needed. Hidden never leaves the GPU between layers.
    ///
    /// Quantized weights work transparently (matvec_variant picks the dequant
    /// kernel from VLLM_VULKAN_QUANT). Falls back to the proven forward_qwen_gpu
    /// if the GPU/weights aren't ready. Gated behind VLLM_VULKAN_RESIDENT=1.
    /// Numerically tracks forward_qwen_gpu (argmax-identical, cos≈1.0).
    pub(crate) fn forward_qwen_gpu_resident(&mut self, token_id: u32, pos: usize) -> GpuResult<Vec<f32>> {
        let cfg = self.qwen.as_ref().expect("invariant: forward_qwen_gpu_resident only called when self.qwen is Some").config.clone();
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
        let theta = cfg.rope_theta;
        let gpu_sdpa = self.flags.gpu_sdpa;

        // Readiness: GPU up, projection weights resident, buffers + norm weights
        // staged. Otherwise use the validated non-fused path.
        let l0_q = "model.layers.0.self_attn.q_proj.weight".to_string();
        let ready = self.engine.is_some()
            && self.gpu_weights.contains_key(&l0_q)
            && self.init_qres_bufs()
            && self.ensure_qwen_norm_weights();
        if !ready {
            return Ok(self.forward_qwen_gpu(token_id, pos));
        }

        // M5b: pipelined CB/fence ring (opt-in). Drain leftovers from the
        // previous token and reset the descriptor cursor exactly once per token.
        let use_ring = self.engine.as_ref().map_or(false, |e| e.ring_active());
        if use_ring {
            let eng = self.engine.as_mut().unwrap();
            eng.begin_forward_ring()?;
        }

        // Embedding row → QR_HA (the only host write of hidden; it stays GPU-
        // resident, ping-ponging HA↔HB, for the rest of the forward).
        {
            let emb = {
                let w = self.qwen.as_ref().unwrap().weights.f32_slice("model.embed_tokens.weight");
                f32_slice_to_bytes(&w[token_id as usize * h..(token_id as usize + 1) * h])
            };
            unsafe { (*self.qres_ptr_mut(QR_HA)).write(&emb)?; }
        }
        // RoPE position (shared by q and k) — write once per token.
        unsafe { (*self.qres_ptr_mut(QR_POS)).write(&(pos as i32).to_le_bytes())?; }

        let rope_pc_q = rope_neox_pc(num_q, head_dim, head_dim, head_dim, theta);
        let rope_pc_k = rope_neox_pc(num_kv, head_dim, head_dim, head_dim, theta);
        let rope_wgy = (((head_dim / 2) as u32) + 255) / 256;

        for layer_idx in 0..cfg.num_hidden_layers {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

            // ── CB1: input-norm → q/k/v → q/k-norm → RoPE (one submit) ───────
            // Gather raw buffer pointers (immutable borrows released before we
            // take &mut engine).
            let ha = self.qres_ptr(QR_HA);
            let xp = self.qres_ptr(QR_X);
            let qp = self.qres_ptr(QR_Q);
            let kp = self.qres_ptr(QR_K);
            let vp = self.qres_ptr(QR_V);
            let posp = self.qres_ptr(QR_POS);
            let ffp = self.qres_ptr(QR_FF);
            let idxp = self.qres_ptr(QR_IDX);
            let inln_p = &self.gpu_norm_w[&ln("input_layernorm.weight")] as *const compute::Buffer;
            let qnorm_p = &self.gpu_norm_w[&ln("self_attn.q_norm.weight")] as *const compute::Buffer;
            let knorm_p = &self.gpu_norm_w[&ln("self_attn.k_norm.weight")] as *const compute::Buffer;
            let qw = &self.gpu_weights[&ln("self_attn.q_proj.weight")].buffer as *const compute::Buffer;
            let kw = &self.gpu_weights[&ln("self_attn.k_proj.weight")].buffer as *const compute::Buffer;
            let vw = &self.gpu_weights[&ln("self_attn.v_proj.weight")].buffer as *const compute::Buffer;
            let (qs, qrr) = matvec_variant(true, q_dim);
            let (ks, krr) = matvec_variant(true, kv_dim);
            let mv_q = matvec_pc13(h, q_dim);
            let mv_kv = matvec_pc13(h, kv_dim);
            let rms_h = rmsnorm_pc(h, eps);
            let rms_hd = rmsnorm_pc(head_dim, eps);

            let eng = self.engine.as_mut().expect("invariant: forward_qwen_gpu_resident only called when self.engine is Some");
            let cb = if use_ring { eng.begin_batch_pipelined()? } else { eng.begin_batch()? };
            unsafe {
                // input_layernorm: HA → X
                eng.record_to(cb, "rms_norm_f32_mul", &[&*ha, &*inln_p, &*xp], &rms_h, (1, 1, 1))?;
                eng.record_barrier_to(cb);
                // q/k/v projections (independent, all read X)
                eng.record_to(cb, &qs, &[&*qw, &*xp, &*qp], &mv_q, ((q_dim as u32 + qrr - 1)/qrr, 1, 1))?;
                eng.record_to(cb, &ks, &[&*kw, &*xp, &*kp], &mv_kv, ((kv_dim as u32 + krr - 1)/krr, 1, 1))?;
                eng.record_to(cb, &ks, &[&*vw, &*xp, &*vp], &mv_kv, ((kv_dim as u32 + krr - 1)/krr, 1, 1))?;
                eng.record_barrier_to(cb);
                // per-head q/k RMSNorm (in place: binding 0 == binding 2)
                eng.record_to(cb, "rms_norm_f32_mul", &[&*qp, &*qnorm_p, &*qp], &rms_hd, (num_q as u32, 1, 1))?;
                eng.record_to(cb, "rms_norm_f32_mul", &[&*kp, &*knorm_p, &*kp], &rms_hd, (num_kv as u32, 1, 1))?;
                eng.record_barrier_to(cb);
                // NeoX RoPE (in place; rotary_dim == head_dim so full coverage)
                eng.record_to(cb, "rope_neox_f32_f32", &[&*qp, &*posp, &*ffp, &*qp, &*idxp], &rope_pc_q, (num_q as u32, rope_wgy, 1))?;
                eng.record_to(cb, "rope_neox_f32_f32", &[&*kp, &*posp, &*ffp, &*kp, &*idxp], &rope_pc_k, (num_kv as u32, rope_wgy, 1))?;
            }
            // CB1 waits immediately even on the ring — its output (q/k/v) is read
            // on the host right below for the SDPA; no overlap by construction.
            if use_ring { eng.submit_batch_pipelined(cb, Vec::new())?; eng.wait_batch_pipelined()?; }
            else { eng.submit_batch(cb)?; }

            // K/V → host KV cache + attention (the one host boundary per layer).
            let q_host = read_f32_buf(unsafe { &*qp }, q_dim);
            let k_host = read_f32_buf(unsafe { &*kp }, kv_dim);
            let v_host = read_f32_buf(unsafe { &*vp }, kv_dim);
            let (ck, cv, slen) = {
                let qm = self.qwen.as_mut().unwrap();
                qm.kv_caches[layer_idx].append(&k_host, &v_host);
                let cache = &qm.kv_caches[layer_idx];
                (cache.k_up_to_now().to_vec(), cache.v_up_to_now().to_vec(), cache.seq_len)
            };
            let attn = if gpu_sdpa {
                self.gpu_sdpa(&q_host, &ck, &cv, num_q, num_kv, head_dim, slen, scale)
            } else {
                model::cpu_sdpa(&q_host, &ck, &cv, num_q, num_kv, head_dim, slen, scale, None)
            };

            // ── CB2: o → residual → post-norm → SwiGLU FFN → residual (1 submit)
            unsafe { (*self.qres_ptr_mut(QR_ATTN)).write(&f32_slice_to_bytes(&attn))?; }
            let ha = self.qres_ptr(QR_HA);
            let hb = self.qres_ptr(QR_HB);
            let attnp = self.qres_ptr(QR_ATTN);
            let op = self.qres_ptr(QR_O);
            let ffin = self.qres_ptr(QR_FFIN);
            let gate = self.qres_ptr(QR_GATE);
            let up = self.qres_ptr(QR_UP);
            let silu = self.qres_ptr(QR_SILU);
            let mid = self.qres_ptr(QR_MID);
            let down = self.qres_ptr(QR_DOWN);
            let pa_p = &self.gpu_norm_w[&ln("post_attention_layernorm.weight")] as *const compute::Buffer;
            let ow = &self.gpu_weights[&ln("self_attn.o_proj.weight")].buffer as *const compute::Buffer;
            let gw = &self.gpu_weights[&ln("mlp.gate_proj.weight")].buffer as *const compute::Buffer;
            let uw = &self.gpu_weights[&ln("mlp.up_proj.weight")].buffer as *const compute::Buffer;
            let dw = &self.gpu_weights[&ln("mlp.down_proj.weight")].buffer as *const compute::Buffer;
            let (os, orr) = matvec_variant(true, h);
            let (gs, grr) = matvec_variant(true, inter);
            let (ds, drr) = matvec_variant(true, h);
            let mv_o = matvec_pc13(q_dim, h);
            let mv_gu = matvec_pc13(h, inter);
            let mv_d = matvec_pc13(inter, h);
            let add_pc = ew_mul_pc(h as u32);
            let silu_pc = ew_unary_pc(inter as u32);
            let mul_pc = ew_mul_pc(inter as u32);
            let rms_h2 = rmsnorm_pc(h, eps);
            let elem = inter as u32;

            let eng = self.engine.as_mut().expect("invariant: forward_qwen_gpu_resident only called when self.engine is Some");
            let cb = if use_ring { eng.begin_batch_pipelined()? } else { eng.begin_batch()? };
            unsafe {
                // o_proj: ATTN → O
                eng.record_to(cb, &os, &[&*ow, &*attnp, &*op], &mv_o, ((h as u32 + orr - 1)/orr, 1, 1))?;
                eng.record_barrier_to(cb);
                // residual: HB = HA + O  (binding 3 = harmless dummy; add.comp
                // declares a PartialBuf at binding 3, used only under ADD_RMS)
                eng.record_to(cb, "add_f32_f32_f32", &[&*ha, &*op, &*hb, &*hb], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
                eng.record_barrier_to(cb);
                // post_attention_layernorm: HB → FFIN
                eng.record_to(cb, "rms_norm_f32_mul", &[&*hb, &*pa_p, &*ffin], &rms_h2, (1, 1, 1))?;
                eng.record_barrier_to(cb);
                // gate/up (independent)
                eng.record_to(cb, &gs, &[&*gw, &*ffin, &*gate], &mv_gu, ((inter as u32 + grr - 1)/grr, 1, 1))?;
                eng.record_to(cb, &gs, &[&*uw, &*ffin, &*up], &mv_gu, ((inter as u32 + grr - 1)/grr, 1, 1))?;
                eng.record_barrier_to(cb);
                // SwiGLU: silu(gate) → mul by up
                eng.record_to(cb, "silu_f32", &[&*gate, &*silu], &silu_pc, ((elem + 511)/512, 1, 1))?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "mul_f32_f32_f32", &[&*silu, &*up, &*mid], &mul_pc, ((elem + 255)/256, 1, 1))?;
                eng.record_barrier_to(cb);
                // down_proj: MID → DOWN
                eng.record_to(cb, &ds, &[&*dw, &*mid, &*down], &mv_d, ((h as u32 + drr - 1)/drr, 1, 1))?;
                eng.record_barrier_to(cb);
                // residual: HA = HB + DOWN (hidden back in HA for next layer)
                eng.record_to(cb, "add_f32_f32_f32", &[&*hb, &*down, &*ha, &*ha], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
            }
            // CB2 is submitted WITHOUT waiting on the ring: its execution overlaps
            // the next layer's CB1 recording (same queue, FIFO, so ordering holds).
            if use_ring { eng.submit_batch_pipelined(cb, Vec::new())?; }
            else { eng.submit_batch(cb)?; }
        }

        // ── Final norm + LM head (one submit) ────────────────────────────────
        let lm_name = self.qwen.as_ref().unwrap().lm_head_name.clone();
        let ha = self.qres_ptr(QR_HA);
        let xp = self.qres_ptr(QR_X);
        let logitp = self.qres_ptr(QR_LOGITS);
        let norm_p = &self.gpu_norm_w["model.norm.weight"] as *const compute::Buffer;
        let lmw = &self.gpu_weights[&lm_name].buffer as *const compute::Buffer;
        let (lms, lmr) = matvec_variant(true, vocab);
        let rms_f = rmsnorm_pc(h, eps);
        let mv_lm = matvec_pc13(h, vocab);
        let eng = self.engine.as_mut().expect("invariant: forward_qwen_gpu_resident only called when self.engine is Some");
        let cb = if use_ring { eng.begin_batch_pipelined()? } else { eng.begin_batch()? };
        unsafe {
            eng.record_to(cb, "rms_norm_f32_mul", &[&*ha, &*norm_p, &*xp], &rms_f, (1, 1, 1))?;
            eng.record_barrier_to(cb);
            eng.record_to(cb, &lms, &[&*lmw, &*xp, &*logitp], &mv_lm, ((vocab as u32 + lmr - 1)/lmr, 1, 1))?;
        }
        // Final CB waits before the host logits read.
        if use_ring { eng.submit_batch_pipelined(cb, Vec::new())?; eng.wait_batch_pipelined()?; }
        else { eng.submit_batch(cb)?; }
        Ok(read_f32_buf(unsafe { &*logitp }, vocab))
    }
}

// ── Column-batched prefill matvec — shared by the `gemma` and `qwen35`
// model paths ──────────────────────────────────────────────────────────────
//
// These two helpers carry the `qwen35_` name for history (they were written for
// the qwen3.6 batched-prefill work) but touch NO qwen3.5/3.6 state: they read
// only `self.gpu_weights` and `self.engine`, both foundation-level. The gemma
// prefill (`gemma_forward.rs`, `flags.gemma_prefill_cols`, default ON) calls
// `qwen35_matvec_cols_tiled` directly. Keeping them inside the
// `#[cfg(feature = "qwen35")]` impl block made that call site
// `#[cfg(feature = "qwen35")]` too, so a `--features gemma` build WITHOUT
// `qwen35` silently lost a default-ON lever and fell back to one
// submit+fence+readback per token per projection. Gate on either feature
// instead of making `gemma` depend on the whole qwen3.6 hybrid.
#[cfg(any(feature = "gemma", feature = "qwen35"))]
impl VulkanModel {
    /// Batched GEMM `[T,n] = xs[T,k] @ W[n,k]^T` reading the qwen3_5 weight
    /// store (`gpu_weights` f16, else host f32 from `self.qwen35`). Mirrors
    /// `VulkanModel::gpu_gemm` exactly, just retargeted at the qwen3_5 weight
    /// store/CPU-fallback (the Gemma `gpu_gemm`'s CPU fallback reads
    /// `self.inner`, which is an empty placeholder on a qwen3_5 model — reusing
    /// it here would silently read garbage, hence this qwen35-specific copy).
    /// Design-A batched-verify projection primitive (MTP re-gate, spec §6):
    /// dispatch the SINGLE-STREAM `mul_mat_vec_{f16,q8_0}_cols` kernel — each
    /// weight element is read/dequantized ONCE and reused across all `t`
    /// token-columns (the Phase-0-confirmed GFX1013 amortization, blended
    /// T_verify(5)/T_verify(1)=1.3963 → GO), unlike `qwen35_gemm`'s
    /// occupancy-starved decode-shape GEMM or its T-serial matvec fallback that
    /// RE-streams the weight per column (the 0.951× Design-B failure). The
    /// resident f16 / q8_0 weight buffers already match the cols kernels' layouts
    /// exactly (f16 `[n,k]`, q8_0 ggml blocks `[n,k/32]`) — no repack.
    ///
    /// Returns `None` (caller falls back to `qwen35_gemm`) when: the resident
    /// weight is not a plain f16/q8_0 buffer (mlx4/nvfp4/fp8 DENSE residency has
    /// no cols dequantizing sibling — set `VLLM_VULKAN_DENSE_Q4_RESIDENT=0`, the
    /// default, so mlx4 projections dequant→requant to q8_0/f16 and DO amortize);
    /// `t` is outside `2..=8` (t==1 has no reuse and stays on the bit-exact serial
    /// path; the `rows*t<=8` LDS envelope caps t); q8_0 with `k % 32 != 0`; or the
    /// pre-registered `_r{rows}_c{t}` variant is missing. `x` = `[t,k]` row-major,
    /// output `[t,n]` row-major (identical convention to `qwen35_gemm`).
    pub(crate) fn qwen35_matvec_cols(&mut self, weight_name: &str, x: &[f32], t: usize, k: usize, n: usize) -> Option<Vec<f32>> {
        // STEP-7 bisect kill-switch: `VLLM_VULKAN_SPEC_NO_COLS=1` forces the
        // qwen35_gemm fallback (GEMM / T-serial matvec) so the harness can
        // localize a garbage-verify to the cols kernel vs the batched mixers.
        if std::env::var("VLLM_VULKAN_SPEC_NO_COLS").map(|v| v != "0").unwrap_or(false) {
            return None;
        }
        if !(2..=8).contains(&t) { return None; }
        // Per-format: base shader, push-constants, and (mlx4 only) the aux
        // scale/bias buffers bound between the packed weight and the activation.
        // All pointers are gathered from `self.gpu_weights` BEFORE
        // `engine.as_mut()` so the immutable borrow ends before the mutable one
        // (the same discipline as `qwen35_matvec`).
        enum ColsAux { None, Mlx4(*const compute::Buffer, *const compute::Buffer) }
        let (base, pc, aux, w_ptr) = {
            let w = self.gpu_weights.get(weight_name)?;
            match (&w.format, &w.aux) {
                (crate::flags::QuantFormat::F16, None) => (
                    "mul_mat_vec_f16_cols",
                    crate::push_constants::matvec_cols_pc2(k, n),
                    ColsAux::None,
                    &w.buffer as *const compute::Buffer,
                ),
                (crate::flags::QuantFormat::Q8_0, None) if k % 32 == 0 => (
                    "mul_mat_vec_q8_0_cols",
                    crate::push_constants::matvec_cols_pc2(k, n),
                    ColsAux::None,
                    &w.buffer as *const compute::Buffer,
                ),
                // MLX-affine 4-bit (the fleet's REAL weight format): the packed
                // nibbles live in `buffer`, the per-(row,group) scales/biases in
                // the Mlx4 aux. Bindings [packed, scales, biases, x, out] + the
                // 5-word matvec_mlx4 pc — identical to mul_mat_vec_mlx4.comp, so
                // NUM_COLS=1 is byte-identical to the serial decode matvec and
                // column c of NUM_COLS=t is that same single-column projection.
                // Requires k%8==0 (nibble packing) and k%group_size==0 (affine
                // groups); a shape violating either falls back to serial.
                (crate::flags::QuantFormat::Mlx4, Some(QuantAux::Mlx4 { scales, biases, group_size }))
                    if k % 8 == 0 && k % (*group_size as usize) == 0 => (
                    "mul_mat_vec_mlx4_cols",
                    matvec_mlx4_pc(k, n, *group_size as usize),
                    ColsAux::Mlx4(scales as *const compute::Buffer, biases as *const compute::Buffer),
                    &w.buffer as *const compute::Buffer,
                ),
                _ => return None,
            }
        };
        let rows = (8 / t as u32).max(1);           // matches the registered set
        let variant = format!("{base}_r{rows}_c{t}");
        let eng = self.engine.as_mut()?;
        if !eng.has_pipeline(&variant) { return None; }  // fall back to qwen35_gemm
        let xb = f32_slice_to_bytes(x);
        let inp = eng.alloc_host_coherent_storage((t * k * 4) as u64).ok()?;
        inp.write(&xb).ok()?;
        let out = eng.alloc_host_coherent_storage((t * n * 4) as u64).ok()?;
        let inp_p = &inp as *const compute::Buffer;
        let out_p = &out as *const compute::Buffer;
        // 2D grid over ceil(n/rows) row-groups (n may exceed the 65535 limit).
        let row_groups = (n as u32 + rows - 1) / rows;
        let wg = if row_groups <= 65535 {
            (row_groups, 1u32, 1u32)
        } else {
            let gx = 32768u32;
            (gx, (row_groups + gx - 1) / gx, 1u32)
        };
        let cb = eng.begin_batch().ok()?;
        unsafe {
            // Binding order matches mul_mat_vec_mlx4.comp / the cols shaders:
            // plain = [weight, x, out]; mlx4 = [packed, scales, biases, x, out].
            match aux {
                ColsAux::None =>
                    eng.record_to(cb, &variant, &[&*w_ptr, &*inp_p, &*out_p], &pc, wg).ok()?,
                ColsAux::Mlx4(s, b) =>
                    eng.record_to(cb, &variant, &[&*w_ptr, &*s, &*b, &*inp_p, &*out_p], &pc, wg).ok()?,
            }
        }
        eng.submit_batch(cb).ok()?;
        let result = read_f32_buf(&out, t * n);
        eng.return_to_pool(inp);
        eng.return_to_pool(out);
        Some(result)
    }

    /// Column-tiled `qwen35_matvec_cols`: process `t` activation columns in
    /// tiles of at most `VLLM_VULKAN_QWEN35_COLS_TILE` (default 8 — the on-node
    /// LDS/T sweet spot; mlx4-cols ACO gate n53) so a PREFILL projection of ANY
    /// prompt length reuses the streamed weight across each <=8-token tile
    /// instead of `t` serial per-token matvecs. Output columns are INDEPENDENT
    /// (each is the exact single-column projection of its token), so
    /// concatenating the per-tile `[tile,n]` blocks in column order rebuilds
    /// `[t,n]` BIT-IDENTICALLY to one hypothetical t-wide dispatch — each column
    /// equals the serial `qwen35_matvec` result up to f32 reduction order (the
    /// on-node argmax/cos gate). Returns None (caller keeps the serial path) if
    /// a tile declines (format/pipeline/geometry), so a run never MIXES cols and
    /// serial projections for one weight. `t<=tile` collapses to a single
    /// `qwen35_matvec_cols` dispatch (the pre-tiling spec-verify behaviour).
    pub(crate) fn qwen35_matvec_cols_tiled(&mut self, weight_name: &str, x: &[f32], t: usize, k: usize, n: usize) -> Option<Vec<f32>> {
        if t < 2 { return None; }
        let tile = std::env::var("VLLM_VULKAN_QWEN35_COLS_TILE").ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(2, 8))
            .unwrap_or(8);
        if t <= tile {
            return self.qwen35_matvec_cols(weight_name, x, t, k, n);
        }
        let mut out = vec![0.0f32; t * n];
        // The (c0, ct) tiling — including the lone-trailing-column fold that
        // keeps every tile >=2 — is `cols_tile_schedule` (host-tested, and
        // reused by the gemma prefill cols path).
        for (c0, ct) in cols_tile_schedule(t, tile) {
            let xt = &x[c0 * k..(c0 + ct) * k];
            // Any tile declining (weight/geometry ineligible, or a transient
            // alloc failure) => uniform serial fallback: never emit a
            // partially-cols result. Format/pipeline eligibility is
            // tile-invariant, so the first tile decides for the whole weight.
            let part = self.qwen35_matvec_cols(weight_name, xt, ct, k, n)?;
            out[c0 * n..(c0 + ct) * n].copy_from_slice(&part);
        }
        Some(out)
    }
}

// ── Qwen3.6 (qwen3_5) hybrid GPU path (`qwen35_*`) — the `qwen35` feature ────
#[cfg(feature = "qwen35")]
impl VulkanModel {
    // ── Qwen3.6 (qwen3_5) GPU helpers ───────────────────────────────────────
    /// Host-f32 weight slice for the qwen3_5 model (mirror of `qwen_w`).
    pub(crate) fn qwen35_w(&self, name: &str) -> Vec<f32> {
        self.qwen35.as_ref().unwrap().weights.f32_slice(name).to_vec()
    }
    /// Single matvec `[1,n] = x[1,k] @ W[n,k]^T` reading the qwen3_5 weight store
    /// (`gpu_weights` f16, else host f32 from `self.qwen35`). One submit.
    pub(crate) fn qwen35_matvec(&mut self, weight_name: &str, x: &[f32], k: usize, n: usize) -> Vec<f32> {
        // Chunked-alloc dispatch (plan §5, P2; VLLM_VULKAN_MAX_ALLOC_MB, default
        // OFF): several row-range buffers, several `record_to_off` dispatches
        // into ONE command buffer + ONE submit, each writing its contiguous row
        // range of a single output buffer. Not wired for nvfp4/fp8 (chunking is
        // lm_head-only in v1, always the Plain matvec kind) and explicitly NOT
        // consulted by the resident WS3 1-CB path (`q35r_meta`/`q35r_rec_mv`
        // read `gpu_weights` only) — `ensure_q35res`'s probe treats a chunked
        // lm_head as resident-unavailable so that path falls back here instead.
        if self.chunked_weights.contains_key(weight_name) {
            let (buf_ptrs, rows): (Vec<*const compute::Buffer>, Vec<usize>) = {
                let cw = self.chunked_weights.get(weight_name).unwrap();
                (cw.buffers.iter().map(|b| b as *const compute::Buffer).collect(), cw.rows.clone())
            };
            let total_rows: usize = rows.iter().sum();
            debug_assert_eq!(total_rows, n, "chunked weight {weight_name}: rows {total_rows} != n {n}");
            if let Some(eng) = self.engine.as_mut() {
                let th = std::time::Instant::now();
                let xb = f32_slice_to_bytes(x);
                let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
                inp.write(&xb).unwrap();
                let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
                let inp_p = &inp as *const compute::Buffer;
                let out_p = &out as *const compute::Buffer;
                let cb = eng.begin_batch().unwrap();
                let mut row_off = 0usize;
                for (i, &r) in rows.iter().enumerate() {
                    let (shader, rr) = matvec_variant(true, r);
                    let wg = (r as u32 + rr - 1) / rr;
                    let pc = matvec_pc13(k, r);
                    unsafe {
                        eng.record_to_off(cb, &shader,
                            &[(&*buf_ptrs[i], 0u64), (&*inp_p, 0u64), (&*out_p, (row_off * 4) as u64)],
                            &pc, (wg, 1, 1)).unwrap();
                    }
                    row_off += r;
                }
                prof_add("mv_host_in", th);
                let ts = std::time::Instant::now();
                eng.submit_batch(cb).unwrap();
                prof_add("mv_submit_fence", ts);
                let tr = std::time::Instant::now();
                let result = read_f32_buf(&out, n);
                eng.return_to_pool(inp);
                eng.return_to_pool(out);
                prof_add("mv_host_out", tr);
                return result;
            }
        }
        let meta = self.gpu_weights.get(weight_name).map(|w| (
            &w.buffer as *const compute::Buffer,
            w.format,
            match &w.aux {
                None => MvKind::Plain,
                Some(QuantAux::Nvfp4 { scales, group_size, e4m3, global }) =>
                    MvKind::Nvfp4 { s: scales as *const _, gs: *group_size, e4m3: *e4m3, global: *global },
                Some(QuantAux::Fp8 { scale, per_row }) =>
                    MvKind::Fp8 { s: scale as *const _, per_row: *per_row },
                Some(QuantAux::Mlx4 { scales, biases, group_size }) =>
                    MvKind::Mlx4 { s: scales as *const _, b: biases as *const _, gs: *group_size },
            },
        ));
        if let (Some(eng), Some((w_ptr, wfmt, kind))) = (self.engine.as_mut(), meta) {
            let th = std::time::Instant::now();
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let out_p = &out as *const compute::Buffer;
            let cb = eng.begin_batch().unwrap();
            unsafe {
                match kind {
                    MvKind::Nvfp4 { s, gs, e4m3, global } => {
                        // Flag-routed: e4m3-resident kernel (raw byte scale +
                        // global pc) or the f32-fold kernel. Same 4 bindings.
                        let (shader, r, pc) = nvfp4_dispatch(k, n, gs, e4m3, global);
                        let wg = (n as u32 + r - 1) / r;
                        eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*inp_p, &*out_p], &pc, (wg,1,1)).unwrap();
                    }
                    MvKind::Fp8 { s, per_row } => {
                        let (shader, r) = matvec_fp8_variant(n);
                        let wg = (n as u32 + r - 1) / r;
                        let pc = matvec_fp8_pc(k, n, per_row);
                        eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*inp_p, &*out_p], &pc, (wg,1,1)).unwrap();
                    }
                    MvKind::Mlx4 { s, b, gs } => {
                        let (shader, r) = matvec_mlx4_variant_k(k, n);
                        let wg = (n as u32 + r - 1) / r;
                        let pc = matvec_mlx4_pc(k, n, gs as usize);
                        eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*b, &*inp_p, &*out_p], &pc, (wg,1,1)).unwrap();
                    }
                    MvKind::Plain => {
                        // RECONCILE union of two features on this dispatch line:
                        //  - Q35_GEOM perf lever (default OFF): swept geometry for
                        //    the TP-4 sharded q8_0 projection shapes.
                        //  - format-aware default (replaces the old
                        //    `matvec_variant(true, n)`): dispatch by the WEIGHT's
                        //    own format so an F16 head projection under a q8_0 main
                        //    model reads with the F16 shader (else garbage).
                        // Geom stays the opt-in override; the format-aware path is
                        // the corrected default (what incoming replaced the old
                        // default with).
                        let (shader, r) = if q35_geom_enabled() {
                            matvec_variant_q35geom(k, n)
                        } else {
                            crate::push_constants::matvec_variant_by_format(wfmt, n)
                        };
                        let wg = (n as u32 + r - 1) / r;
                        let pc = matvec_pc13(k, n);
                        eng.record_to(cb, &shader, &[&*w_ptr, &*inp_p, &*out_p], &pc, (wg,1,1)).unwrap();
                    }
                }
            }
            prof_add("mv_host_in", th);
            let ts = std::time::Instant::now();
            eng.submit_batch(cb).unwrap();
            prof_add("mv_submit_fence", ts);
            let tr = std::time::Instant::now();
            let result = read_f32_buf(&out, n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
            prof_add("mv_host_out", tr);
            result
        } else if let Some(f16w) = self.q35_f16_host.get(weight_name) {
            // f16 host table (lm_head / lean-dropped weight): matmul in f32.
            let mut out = vec![0.0f32; n];
            for j in 0..n {
                let row = &f16w[j * k..(j + 1) * k];
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += x[i] * half::f16::from_bits(row[i]).to_f32();
                }
                out[j] = acc;
            }
            out
        } else {
            let w = self.qwen35.as_ref().unwrap().weights.f32_slice(weight_name);
            model::cpu_matmul(x, w, 1, k, n)
        }
    }
    /// Batched matvec: several qwen3_5 weights sharing the SAME input `x` and
    /// contraction dim `k` go into ONE command buffer + ONE submit/fence (the
    /// host-bound win: collapses N submits → 1). `jobs` = (weight_name, n).
    pub(crate) fn qwen35_matvec_multi(&mut self, x: &[f32], k: usize, jobs: &[(&str, usize)]) -> Vec<Vec<f32>> {
        let metas: Option<Vec<(*const compute::Buffer, MvKind)>> = jobs.iter()
            .map(|(name, _)| self.gpu_weights.get(*name).map(|w| (
                &w.buffer as *const compute::Buffer,
                match &w.aux {
                    None => MvKind::Plain,
                    Some(QuantAux::Nvfp4 { scales, group_size, e4m3, global }) =>
                        MvKind::Nvfp4 { s: scales as *const _, gs: *group_size, e4m3: *e4m3, global: *global },
                    Some(QuantAux::Fp8 { scale, per_row }) =>
                        MvKind::Fp8 { s: scale as *const _, per_row: *per_row },
                    Some(QuantAux::Mlx4 { scales, biases, group_size }) =>
                        MvKind::Mlx4 { s: scales as *const _, b: biases as *const _, gs: *group_size },
                },
            )))
            .collect();
        if let (Some(eng), Some(metas)) = (self.engine.as_mut(), metas) {
            let th = std::time::Instant::now();
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let outs: Vec<compute::Buffer> = jobs.iter()
                .map(|(_, n)| eng.alloc_host_coherent_storage((*n * 4) as u64).unwrap())
                .collect();
            let cb = eng.begin_batch().unwrap();
            for (i, (_, n)) in jobs.iter().enumerate() {
                let out_p = &outs[i] as *const compute::Buffer;
                let (w_ptr, kind) = metas[i];
                unsafe {
                    match kind {
                        MvKind::Nvfp4 { s, gs, e4m3, global } => {
                            let (shader, r, pc) = nvfp4_dispatch(k, *n, gs, e4m3, global);
                            let wg = (*n as u32 + r - 1) / r;
                            eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*inp_p, &*out_p], &pc, (wg,1,1)).unwrap();
                        }
                        MvKind::Fp8 { s, per_row } => {
                            let (shader, r) = matvec_fp8_variant(*n);
                            let wg = (*n as u32 + r - 1) / r;
                            let pc = matvec_fp8_pc(k, *n, per_row);
                            eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*inp_p, &*out_p], &pc, (wg,1,1)).unwrap();
                        }
                        MvKind::Mlx4 { s, b, gs } => {
                            let (shader, r) = matvec_mlx4_variant_k(k, *n);
                            let wg = (*n as u32 + r - 1) / r;
                            let pc = matvec_mlx4_pc(k, *n, gs as usize);
                            eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*b, &*inp_p, &*out_p], &pc, (wg,1,1)).unwrap();
                        }
                        MvKind::Plain => {
                            // Swept geometry (VLLM_VULKAN_Q35_GEOM); see
                            // qwen35_matvec. OFF = byte-identical legacy dispatch.
                            let (shader, r) = if q35_geom_enabled() {
                                matvec_variant_q35geom(k, *n)
                            } else {
                                matvec_variant(true, *n)
                            };
                            let wg = (*n as u32 + r - 1) / r;
                            let pc = matvec_pc13(k, *n);
                            eng.record_to(cb, &shader, &[&*w_ptr, &*inp_p, &*out_p], &pc, (wg,1,1)).unwrap();
                        }
                    }
                }
            }
            prof_add("mv_host_in", th);
            let ts = std::time::Instant::now();
            eng.submit_batch(cb).unwrap();
            prof_add("mv_submit_fence", ts);
            let tr = std::time::Instant::now();
            let results: Vec<Vec<f32>> = jobs.iter().enumerate()
                .map(|(i, (_, n))| read_f32_buf(&outs[i], *n))
                .collect();
            eng.return_to_pool(inp);
            for o in outs { eng.return_to_pool(o); }
            prof_add("mv_host_out", tr);
            results
        } else {
            jobs.iter().map(|(name, n)| {
                let w = self.qwen35.as_ref().unwrap().weights.f32_slice(name);
                model::cpu_matmul(x, w, 1, k, *n)
            }).collect()
        }
    }
    /// Phase 2: GPU-accelerated qwen3_5 (Qwen3.6) forward for one token.
    ///
    /// Mirrors `qwen35::Qwen35Model::forward` EXACTLY, but routes the large
    /// projection matmuls (GatedDeltaNet in_proj_*/out_proj, GatedAttention
    /// q/k/v/o_proj, dense MLP gate/up/down_proj, lm_head) to the GPU via
    /// `qwen35_matvec*`. The cheap per-token math — depthwise conv1d, the
    /// delta-rule recurrence, gated/RMS norms, partial RoPE, the SDPA — runs on
    /// the CPU and replicates `qwen35.rs` bit-for-bit (f32). Per layer the
    /// independent projections share their input and collapse to FEW submits.
    /// f32 embedding row for `token_id` (`h` elems). Prefers the packed-4bit
    /// resident embed (per-row on-demand mlx-affine decode, bit-exact to the old
    /// whole-table f16 path); falls back to the legacy whole f16 host table.
    /// This is the single embed-lookup accessor for every qwen3_5 forward.
    pub(crate) fn q35_embed_row(&self, token_id: usize, h: usize) -> Vec<f32> {
        if let Some(q) = self.qwen35.as_ref() {
            if let Some(pe) = q.embed_packed.as_ref() {
                return pe.row_f32(token_id);
            }
        }
        let w = self
            .q35_f16_host
            .get("model.embed_tokens.weight")
            .expect("qwen3_5 embed lookup: neither packed embed nor f16 host present");
        w[token_id * h..(token_id + 1) * h]
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect()
    }

    pub(crate) fn forward_qwen35_gpu(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        let cfg = self.qwen35.as_ref().unwrap().config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let vocab = cfg.vocab_size;

        // Embedding (no scaling) — packed per-row decode or f16 host lookup.
        let mut hidden: Vec<f32> = self.q35_embed_row(token_id as usize, h);

        // WS3 resident stage (VLLM_VULKAN_Q35_1CB): whole span + lm_head tail
        // through persistent GPU buffers. pp_last-only here because this
        // function's contract is "return logits" (single-node full model).
        if self.pp_last {
            if let Some(out) = self.forward_qwen35_span_resident(&cfg, &hidden, pos) {
                return out;
            }
        }

        // Resident layers only (pp_start..pp_end == 0..num_hidden_layers on a
        // single node); shares the layer body with forward_pp_qwen35.
        for layer_idx in self.pp_start..self.pp_end {
            hidden = self.qwen35_layer_gpu(&cfg, layer_idx, &hidden, pos);
        }

        // Final norm + LM head (GPU matvec).
        let norm_w = self.qwen35_w("model.norm.weight");
        let normed = model::cpu_rms_norm(&hidden, &norm_w, eps);
        let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
        self.qwen35_matvec(&lm_name, &normed, h, vocab)
    }
    /// Run ONE qwen3.6 decoder layer (GLOBAL `layer_idx`) on a hidden state,
    /// GPU projections + CPU recurrence/attn/norms. Mirrors the body of
    /// `forward_qwen35_gpu`'s loop; per-layer state is resolved via `state_idx`
    /// inside the sub-block helpers, so this is PP-safe (the layer need only be
    /// resident on this stage).
    pub(crate) fn qwen35_layer_gpu(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, hidden: &[f32], pos: usize) -> Vec<f32> {
        let eps = cfg.rms_norm_eps;
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
        // Attention sub-block.
        let residual = hidden.to_vec();
        let in_ln = self.qwen35_w(&ln("input_layernorm.weight"));
        let x = model::cpu_rms_norm(hidden, &in_ln, eps);
        let attn_out = match cfg.layer_types[layer_idx] {
            qwen35::LayerType::FullAttention => {
                let t = std::time::Instant::now();
                let r = self.qwen35_gated_attention_gpu(cfg, layer_idx, &x, pos);
                prof_add("q35_full_attn_block", t); r
            }
            qwen35::LayerType::LinearAttention => {
                let t = std::time::Instant::now();
                let r = self.qwen35_delta_net_gpu(cfg, layer_idx, &x);
                prof_add("q35_deltanet_block", t); r
            }
        };
        let h1: Vec<f32> = residual.iter().zip(&attn_out).map(|(&r, &a)| r + a).collect();
        // MLP sub-block (dense SwiGLU).
        let residual2 = h1.clone();
        let post_ln = self.qwen35_w(&ln("post_attention_layernorm.weight"));
        let ff_in = model::cpu_rms_norm(&h1, &post_ln, eps);
        let tm = std::time::Instant::now();
        let mlp_out = if cfg.is_moe() {
            self.qwen35_moe_mlp_gpu(cfg, layer_idx, &ff_in)
        } else {
            self.qwen35_dense_mlp_gpu(cfg, layer_idx, &ff_in)
        };
        prof_add("q35_mlp_block", tm);
        residual2.iter().zip(&mlp_out).map(|(&r, &m)| r + m).collect()
    }
    /// Pipeline-parallel forward for one qwen3.6 (qwen3_5) token, cross-stage.
    /// Stage holds resident global layers `[pp_start, pp_end)` + their dual
    /// per-layer state (DeltaNetState / KvCache), advanced IN PLACE every call.
    ///
    /// - First stage (`pp_first`): embed `token_id` (qwen3.6 embed has NO
    ///   scaling) into the hidden vector (`hidden_in` ignored — pass `[]`).
    /// - Every stage: run the resident layers via the GPU layer path.
    /// - Non-last stage: return the hidden vector (vCCL-send to the next stage).
    /// - Last stage (`pp_last`): final `model.norm` RMSNorm + lm_head matvec →
    ///   full logits.
    ///
    /// UNLIKE gemma PP, the inter-stage message is JUST the hidden state — the
    /// dual state is resident and never shipped (each stage advances its own).
    pub(crate) fn forward_pp_qwen35_impl(&mut self, token_id: u32, hidden_in: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        let cfg = self
            .qwen35
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forward_pp_qwen35 needs a qwen3_5 model"))?
            .config
            .clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let vocab = cfg.vocab_size;
        let (first, last) = (self.pp_first, self.pp_last);

        // First stage embeds the token (no scaling); else continue from hidden_in.
        let mut hidden: Vec<f32> = if first {
            self.q35_embed_row(token_id as usize, h)
        } else {
            hidden_in
        };

        // WS3 resident stage (VLLM_VULKAN_Q35_1CB): the whole span through
        // persistent GPU buffers — returns the stage hidden (or logits on
        // the last stage) directly. Falls through to the per-block path when
        // the readiness probe fails.
        if let Some(out) = self.forward_qwen35_span_resident(&cfg, &hidden, pos) {
            return Ok(out);
        }

        // Resident decoder layers (GPU matmuls; state advanced in place).
        for layer_idx in self.pp_start..self.pp_end {
            hidden = self.qwen35_layer_gpu(&cfg, layer_idx, &hidden, pos);
        }

        if last {
            // P3: stash the pre-`model.norm` residual for the MTP draft head
            // (only when a head is loaded — else the clone is pure overhead).
            if self.mtp_head.is_some() {
                self.q35_last_prenorm = Some(hidden.clone());
            }
            // Final norm (f32 host) + lm_head (GPU matvec) → full logits.
            let norm_w = self.qwen35_w("model.norm.weight");
            let normed = model::cpu_rms_norm(&hidden, &norm_w, eps);
            let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
            Ok(self.qwen35_matvec(&lm_name, &normed, h, vocab))
        } else {
            Ok(hidden)
        }
    }
    /// Gated full attention (full_attention layers), GPU projections + CPU SDPA.
    /// Mirrors `Qwen35Model::gated_attention`.
    pub(crate) fn qwen35_gated_attention_gpu(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, x: &[f32], pos: usize) -> Vec<f32> {
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let rotary = cfg.rotary_dim();
        let theta = cfg.rope_theta;
        let ln = |s: &str| format!("model.layers.{layer_idx}.self_attn.{s}");

        // q_proj is double-width [query(hd)|gate(hd)] per head; k/v normal.
        // All three share input x → ONE submit.
        let (qn_w, kn_w, vn_w) = (ln("q_proj.weight"), ln("k_proj.weight"), ln("v_proj.weight"));
        let mut qkv = self.qwen35_matvec_multi(x, h, &[
            (&qn_w, nq * hd * 2), (&kn_w, kv_dim), (&vn_w, kv_dim),
        ]);
        let v = qkv.pop().unwrap();
        let mut k = qkv.pop().unwrap();
        let q_and_gate = qkv.pop().unwrap();

        let mut q = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for head in 0..nq {
            let base = head * 2 * hd;
            q[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base..base + hd]);
            gate[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base + hd..base + 2 * hd]);
        }

        // Per-head Q/K RMSNorm (before RoPE).
        let tqk = std::time::Instant::now();
        let qn = self.qwen35_w(&ln("q_norm.weight"));
        let kn = self.qwen35_w(&ln("k_norm.weight"));
        for hi in 0..nq {
            let s = &mut q[hi * hd..(hi + 1) * hd];
            let n = model::cpu_rms_norm(s, &qn, eps);
            s.copy_from_slice(&n);
        }
        for hi in 0..nkv {
            let s = &mut k[hi * hd..(hi + 1) * hd];
            let n = model::cpu_rms_norm(s, &kn, eps);
            s.copy_from_slice(&n);
        }

        // Partial RoPE.
        model::cpu_rope(&mut q, &mut k, pos, nq, nkv, hd, rotary, theta);
        prof_add("q35_gattn_qknorm_rope", tqk);

        // KV cache append (host, retained for KvStore export regardless of
        // which SDPA path runs) + geometry needed by the GPU seam, resolved
        // before any &mut self GPU call to avoid overlapping the qwen35/self
        // borrows.
        let tsd = std::time::Instant::now();
        let (max_seq_len, seq_len) = {
            let qm = self.qwen35.as_mut().unwrap();
            let si = qm.state_idx(layer_idx);
            let cache = match &mut qm.layer_state[si] {
                qwen35::LayerState::Full(c) => c,
                _ => unreachable!("full_attention layer has a KV cache"),
            };
            cache.append(&k, &v);
            (cache.max_seq_len, cache.seq_len)
        };

        // item-4a: GPU-resident SDPA (`_sg` kernel) replaces `cpu_sdpa` when
        // the flag is on and the `sg` decode kernel is selected. Falls back
        // to `cpu_sdpa` over the host KvCache on any `None` (no engine, or
        // resident kv-plane alloc failed) — cpu_sdpa is the always-live
        // correctness oracle.
        let gpu_attn_out = if q35_gpu_attn_enabled() && self.flags.attn == flags::AttnKernel::Sg {
            self.gpu_kv_append(layer_idx, &k, &v, pos, nkv, hd, max_seq_len);
            self.gpu_sdpa_resident(layer_idx, &q, nq, nkv, hd, seq_len, scale, max_seq_len)
        } else {
            None
        };
        let attn_out = match gpu_attn_out {
            Some(out) => out,
            None => {
                let qm = self.qwen35.as_mut().unwrap();
                let si = qm.state_idx(layer_idx);
                let cache = match &mut qm.layer_state[si] {
                    qwen35::LayerState::Full(c) => c,
                    _ => unreachable!("full_attention layer has a KV cache"),
                };
                model::cpu_sdpa(&q, cache.k_up_to_now(), cache.v_up_to_now(), nq, nkv, hd, cache.seq_len, scale, None)
            }
        };
        prof_add("q35_gattn_sdpa", tsd);

        // Output gate then o_proj (GPU).
        let gated: Vec<f32> = attn_out.iter().zip(&gate)
            .map(|(&a, &g)| a * (1.0 / (1.0 + (-g).exp()))).collect();
        self.qwen35_matvec(&ln("o_proj.weight"), &gated, q_dim, h)
    }
    /// GatedDeltaNet linear attention (linear_attention layers), decode step.
    /// GPU projections + CPU conv1d/delta-rule/gated-norm. Mirrors
    /// `Qwen35Model::delta_net` bit-for-bit on the recurrence.
    pub(crate) fn qwen35_delta_net_gpu(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, x: &[f32]) -> Vec<f32> {
        // WS1b: fused single-CB GPU deltanet (VLLM_VULKAN_DN_GPU, default ON).
        // Falls through to the CPU conv/recurrence path below when the flag is
        // off, the engine is absent, or any weight is not GPU-resident.
        if dn_gpu_enabled() && self.engine.is_some() {
            if let Some(out) = self.qwen35_delta_net_gpu_fused(cfg, layer_idx, x) {
                return out;
            }
        }
        let eps = cfg.rms_norm_eps;
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let kern = cfg.linear_conv_kernel_dim;
        let ratio = nv / nk;
        let h = cfg.hidden_size;
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");

        // Projections (GPU). in_proj_qkv/z/a/b all share input x → ONE submit.
        // WS1(a) attribution: bracket the two projection submits separately so
        // the block's cost splits into proj-tax vs CPU recurrence (they also
        // still land in the mv_* buckets — these sub-buckets attribute the same
        // time WITHIN q35_deltanet_block).
        let (qkv_n, z_n, a_n, b_n) = (
            ln("in_proj_qkv.weight"), ln("in_proj_z.weight"),
            ln("in_proj_a.weight"), ln("in_proj_b.weight"),
        );
        let tip = std::time::Instant::now();
        let mut proj = self.qwen35_matvec_multi(x, h, &[
            (&qkv_n, conv_dim), (&z_n, value_dim), (&a_n, nv), (&b_n, nv),
        ]);
        prof_add("q35_dn_inproj", tip);
        let b = proj.pop().unwrap();
        let a = proj.pop().unwrap();
        let z = proj.pop().unwrap();
        let qkv = proj.pop().unwrap();

        // CPU host weights for the recurrence (extract before touching state).
        let twf = std::time::Instant::now();
        let conv_w = self.qwen35_w(&ln("conv1d.weight")); // [conv_dim, 1, kern]
        let a_log = self.qwen35_w(&ln("A_log"));
        let dt_bias = self.qwen35_w(&ln("dt_bias"));
        let norm_w = self.qwen35_w(&ln("norm.weight"));
        prof_add("q35_weight_fetch", twf);

        // Causal depthwise conv1d + SiLU, updating the sliding window state.
        let tconv = std::time::Instant::now();
        let win = kern - 1;
        let mut conv_out = vec![0.0f32; conv_dim];
        {
            use rayon::prelude::*;
            let qm = self.qwen35.as_mut().unwrap();
            let si = qm.state_idx(layer_idx);
            let st = match &mut qm.layer_state[si] {
                qwen35::LayerState::Linear(d) => d,
                _ => unreachable!("linear_attention layer has a DeltaNet state"),
            };
            // Depthwise conv: each channel `c` is independent (own conv_state row
            // + own conv_out element). Parallelize across cores (bit-exact).
            let chan = |c: usize, cs: &mut [f32], out_c: &mut f32| {
                let mut acc = 0.0f32;
                for t in 0..win { acc += cs[t] * conv_w[c * kern + t]; }
                acc += qkv[c] * conv_w[c * kern + win];
                *out_c = acc / (1.0 + (-acc).exp()); // silu
                if win > 0 {
                    for t in 0..win - 1 { cs[t] = cs[t + 1]; }
                    cs[win - 1] = qkv[c];
                }
            };
            if win == 0 {
                conv_out.iter_mut().enumerate().for_each(|(c, o)| {
                    *o = { let a = qkv[c] * conv_w[c * kern]; a / (1.0 + (-a).exp()) };
                });
            } else if par_deltanet() {
                st.conv_state.par_chunks_mut(win)
                    .zip(conv_out.par_iter_mut())
                    .enumerate()
                    .for_each(|(c, (cs, o))| chan(c, cs, o));
            } else {
                st.conv_state.chunks_mut(win)
                    .zip(conv_out.iter_mut())
                    .enumerate()
                    .for_each(|(c, (cs, o))| chan(c, cs, o));
            }
        }

        prof_add("q35_conv1d", tconv);
        // Split + per-head RMSNorm(no weight) with inv_scale / inv_scale^2.
        // WS1(a): bracketed separately from the recurrence.
        let tqk = std::time::Instant::now();
        let inv = 1.0 / (kd as f32).sqrt();
        let q_flat = &conv_out[..key_dim];
        let k_flat = &conv_out[key_dim..2 * key_dim];
        let v_flat = &conv_out[2 * key_dim..];
        let mut q = vec![0.0f32; key_dim];
        let mut k = vec![0.0f32; key_dim];
        for hi in 0..nk {
            let qn = model::cpu_rms_norm_no_weight(&q_flat[hi * kd..(hi + 1) * kd], kd, 1e-6);
            let kn = model::cpu_rms_norm_no_weight(&k_flat[hi * kd..(hi + 1) * kd], kd, 1e-6);
            for j in 0..kd {
                q[hi * kd + j] = qn[j] * inv * inv;
                k[hi * kd + j] = kn[j] * inv;
            }
        }
        prof_add("q35_dn_qknorm", tqk);
        let trec = std::time::Instant::now();

        // Recurrent delta rule per v-head, gated norm. The v-heads are FULLY
        // INDEPENDENT (each owns a contiguous `[kd,vd]` slice of `state` and a
        // `[vd]` slice of `gated`; all other inputs are read-only), so the loop
        // parallelizes across CPU cores with NO change to the per-head f32 math
        // → bit-exact regardless of thread count. This is the dominant decode
        // cost on the BC-250 (~3ms/layer × 48 linear layers, single-core).
        // Gated by VLLM_VULKAN_Q35_PARREC (default ON; "0" forces the serial path
        // for A/B + as a fallback).
        let mut gated = vec![0.0f32; value_dim];
        {
            use rayon::prelude::*;
            let qm = self.qwen35.as_mut().unwrap();
            let si = qm.state_idx(layer_idx);
            let st = match &mut qm.layer_state[si] {
                qwen35::LayerState::Linear(d) => d,
                _ => unreachable!(),
            };
            // One closure computing head `j` into its own state/gated slices.
            let head = |j: usize, state_j: &mut [f32], gated_j: &mut [f32]| {
                let kh = j / ratio;
                let q_j = &q[kh * kd..(kh + 1) * kd];
                let k_j = &k[kh * kd..(kh + 1) * kd];
                let v_j = &v_flat[j * vd..(j + 1) * vd];
                let g = -(a_log[j].exp()) * {
                    let xx = a[j] + dt_bias[j];
                    if xx > 20.0 { xx } else { (1.0 + xx.exp()).ln() }
                };
                let decay = g.exp();
                let beta = 1.0 / (1.0 + (-b[j]).exp());

                for e in 0..kd * vd { state_j[e] *= decay; }
                let mut kv_mem = vec![0.0f32; vd];
                for kk in 0..kd {
                    let kv = k_j[kk];
                    for vv in 0..vd { kv_mem[vv] += state_j[kk * vd + vv] * kv; }
                }
                let mut delta = vec![0.0f32; vd];
                for vv in 0..vd { delta[vv] = (v_j[vv] - kv_mem[vv]) * beta; }
                for kk in 0..kd {
                    let kv = k_j[kk];
                    for vv in 0..vd { state_j[kk * vd + vv] += kv * delta[vv]; }
                }
                let mut out_j = vec![0.0f32; vd];
                for kk in 0..kd {
                    let qv = q_j[kk];
                    for vv in 0..vd { out_j[vv] += state_j[kk * vd + vv] * qv; }
                }
                let normed = model::cpu_rms_norm(&out_j, &norm_w, eps);
                for vv in 0..vd {
                    let zz = z[j * vd + vv];
                    gated_j[vv] = normed[vv] * (zz / (1.0 + (-zz).exp()));
                }
            };
            if par_deltanet() {
                st.state.par_chunks_mut(kd * vd)
                    .zip(gated.par_chunks_mut(vd))
                    .enumerate()
                    .for_each(|(j, (state_j, gated_j))| head(j, state_j, gated_j));
            } else {
                st.state.chunks_mut(kd * vd)
                    .zip(gated.chunks_mut(vd))
                    .enumerate()
                    .for_each(|(j, (state_j, gated_j))| head(j, state_j, gated_j));
            }
        }

        prof_add("q35_deltanet_recur", trec);
        // out_proj (GPU). WS1(a): bracketed (see in_proj note above).
        let top = std::time::Instant::now();
        let r = self.qwen35_matvec(&ln("out_proj.weight"), &gated, value_dim, h);
        prof_add("q35_dn_outproj", top);
        r
    }
    /// Upload one GatedDeltaNet layer's decode constants + state to resident
    /// GPU buffers (WS1b). Idempotent. The conv/delta state is SEEDED from the
    /// CPU `DeltaNetState` (zeros at sequence start; also correct if the GPU
    /// path engages after CPU-stepped tokens) and is GPU-authoritative from
    /// then on — the CPU copy goes stale (see `dn_gpu_enabled`). Returns false
    /// if there is no engine, a tensor is missing, or the dims exceed the
    /// `q35_gdn_step` shader's 128-column workgroup.
    // ── P1: speculative-pipelining rollback (spec_snapshot / spec_restore) ──
    //
    // Snapshot + bit-exact restore of ALL per-token mutable decode state on the
    // qwen3_5 stage, so a rejected speculative token can be rolled back exactly.
    // Two authority regimes, both covered:
    //   * DN_GPU on  (node resident path): each GatedDeltaNet layer's conv/delta
    //     state is authoritative in its `DnGpuLayer` device buffers → snapshotted
    //     device→device via `vkCmdCopyBuffer` (one fenced batch). The stale host
    //     `DeltaNetState` copy for those layers is skipped.
    //   * DN_GPU off / engine-less (Mac): host `DeltaNetState` is authoritative
    //     → captured by `Qwen35Model::spec_snapshot_host`.
    // Full-attention KV is host-resident on this path (`cpu_sdpa` over `KvCache`)
    // in BOTH regimes: rollback is a `seq_len` counter rewind (overwrite-in-place
    // K/V planes ⇒ complete). The state buffers are quiescent between tokens
    // (every deltanet CB submits fenced/blocking), so no extra sync is needed.

    /// Stage-local `state_idx` set of the linear layers whose DeltaNet state is
    /// GPU-authoritative right now (so `spec_snapshot_host` skips their stale
    /// host copy). Empty when engine-less or `DN_GPU` off.
    fn dn_gpu_skip_set(&self) -> std::collections::HashSet<usize> {
        let mut skip = std::collections::HashSet::new();
        if self.engine.is_some() {
            if let Some(m) = self.qwen35.as_ref() {
                for &g in self.dn_gpu.keys() {
                    skip.insert(m.state_idx(g));
                }
            }
        }
        skip
    }

    /// Snapshot ALL per-token mutable decode state into ring `slot` (P1). See
    /// the block comment above. Cost lands in the `q35_spec_snapshot` prof bucket.
    pub(crate) fn spec_snapshot_impl(&mut self, slot: usize) -> Result<(), String> {
        let t = std::time::Instant::now();
        let skip = self.dn_gpu_skip_set();
        let host = match self.qwen35.as_ref() {
            Some(m) => m.spec_snapshot_host(&skip),
            None => return Err("spec_snapshot: no qwen3_5 model resident".into()),
        };
        while self.spec_slots.len() <= slot {
            self.spec_slots.push(SpecSlot::default());
        }
        // GPU-authoritative deltanet buffers: device→device copy, one fenced CB.
        if self.engine.is_some() && !self.dn_gpu.is_empty() {
            let Self { engine, dn_gpu, spec_slots, .. } = self;
            let eng = engine.as_mut().ok_or("spec_snapshot: engine gone")?;
            let sl = &mut spec_slots[slot];
            // Phase 1: lazily allocate this slot's snapshot buffers (once).
            for (l, layer) in dn_gpu.iter() {
                if !sl.dn_gpu.contains_key(l) {
                    let c = eng.alloc_host_coherent_storage(layer.conv_state.size)?;
                    let s = eng.alloc_host_coherent_storage(layer.state.size)?;
                    sl.dn_gpu.insert(*l, (c, s));
                }
            }
            // Phase 2: record all copies (live buffer → snapshot buffer), submit.
            let cb = eng.begin_batch()?;
            for (l, layer) in dn_gpu.iter() {
                let (c, s) = sl.dn_gpu.get(l).ok_or("spec_snapshot: slot buf missing")?;
                eng.record_copy_to(cb, &layer.conv_state, c, 0, 0, layer.conv_state.size);
                eng.record_copy_to(cb, &layer.state, s, 0, 0, layer.state.size);
            }
            eng.submit_batch(cb)?;
        }
        self.spec_slots[slot].host = host;
        self.spec_slots[slot].valid = true;
        prof_add("q35_spec_snapshot", t);
        Ok(())
    }

    /// Restore ALL per-token mutable decode state from ring `slot` (P1). Cost
    /// lands in the `q35_spec_restore` prof bucket.
    pub(crate) fn spec_restore_impl(&mut self, slot: usize) -> Result<(), String> {
        let t = std::time::Instant::now();
        if slot >= self.spec_slots.len() || !self.spec_slots[slot].valid {
            return Err(format!("spec_restore: slot {slot} not populated"));
        }
        // GPU-authoritative deltanet buffers: reverse copy (snapshot → live).
        if self.engine.is_some() && !self.dn_gpu.is_empty() {
            let Self { engine, dn_gpu, spec_slots, .. } = self;
            let eng = engine.as_mut().ok_or("spec_restore: engine gone")?;
            let sl = &spec_slots[slot];
            let cb = eng.begin_batch()?;
            for (l, layer) in dn_gpu.iter() {
                if let Some((c, s)) = sl.dn_gpu.get(l) {
                    eng.record_copy_to(cb, c, &layer.conv_state, 0, 0, layer.conv_state.size);
                    eng.record_copy_to(cb, s, &layer.state, 0, 0, layer.state.size);
                }
            }
            eng.submit_batch(cb)?;
        }
        // Host-authoritative state (KV counters + CPU-path deltanet). Clone the
        // small snapshot out first so `spec_slots` isn't borrowed across the
        // `&mut qwen35`.
        let host = self.spec_slots[slot].host.clone();
        if let Some(m) = self.qwen35.as_mut() {
            m.spec_restore_host(&host);
        }
        prof_add("q35_spec_restore", t);
        Ok(())
    }

    /// FNV-1a fingerprint of ALL resident mutable state, one u64 per resident
    /// layer (in `[pp_start, pp_end)` order): GatedDeltaNet layers hash their
    /// live conv+delta state (from the GPU buffer when authoritative, else the
    /// host copy); full-attn layers hash `seq_len ++ live K ++ live V`. The
    /// P1 gate compares this across the restore for exact rollback.
    pub(crate) fn spec_state_fingerprint_impl(&self) -> Result<Vec<u64>, String> {
        fn fnv(acc: &mut u64, bytes: &[u8]) {
            for &b in bytes {
                *acc ^= b as u64;
                *acc = acc.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        let m = self.qwen35.as_ref().ok_or("fingerprint: no qwen3_5 model")?;
        let mut out = Vec::with_capacity(m.layer_state.len());
        for (si, s) in m.layer_state.iter().enumerate() {
            let g = m.pp_start + si; // global layer idx
            let mut acc = 0xcbf2_9ce4_8422_2325u64;
            match s {
                qwen35::LayerState::Linear(d) => {
                    if let Some(layer) = self.dn_gpu.get(&g) {
                        // GPU-authoritative: read the live device buffers.
                        let mut cbuf = vec![0u8; layer.conv_state.size as usize];
                        let mut sbuf = vec![0u8; layer.state.size as usize];
                        layer.conv_state.read(&mut cbuf)?;
                        layer.state.read(&mut sbuf)?;
                        fnv(&mut acc, &cbuf);
                        fnv(&mut acc, &sbuf);
                    } else {
                        fnv(&mut acc, &f32_slice_to_bytes(&d.conv_state));
                        fnv(&mut acc, &f32_slice_to_bytes(&d.state));
                    }
                }
                qwen35::LayerState::Full(c) => {
                    fnv(&mut acc, &(c.seq_len as u64).to_le_bytes());
                    fnv(&mut acc, &f32_slice_to_bytes(c.k_up_to_now()));
                    fnv(&mut acc, &f32_slice_to_bytes(c.v_up_to_now()));
                }
            }
            out.push(acc);
        }
        Ok(out)
    }

    // ── KV-prefix export/import device-aware readback (LMCache-NAS follow-up) ──
    //
    // `Qwen35Model::export_prefix`/`import_prefix` only serialize the HOST
    // `layer_state` copy. Under `VLLM_VULKAN_DN_GPU=1` (and transitively
    // `VLLM_VULKAN_Q35_1CB=1`, whose fused resident-stage kernels read/write
    // the SAME `dn_gpu` buffers directly — see `forward_qwen35_span_resident`'s
    // `LinearAttention` arm) the host copy is stale; the GPU buffers are
    // authoritative, exactly the regime `spec_snapshot_impl`/`spec_restore_impl`
    // already handle for speculative rollback. Full-attention KV stays
    // host-resident in BOTH regimes (`cpu_sdpa` over the host `KvCache`, used
    // unchanged by the resident path — see the `FullAttention` arm above), so
    // only the linear/DeltaNet layers need this readback.
    //
    // These two helpers sync the host `layer_state` DeltaNet copy against the
    // live `dn_gpu` device buffers immediately before/after the existing
    // host-only (de)serialization, so `export_prefix`/`import_prefix` never
    // need to know about device residency at all. `dn_gpu` buffers are
    // host-coherent/mapped (`alloc_host_coherent_storage`), so this is a
    // direct memcpy via `read_f32_buf`/`Buffer::write` — no command buffer or
    // fence needed, same as `spec_state_fingerprint_impl`'s direct `.read()`
    // (the buffers are quiescent between tokens: every deltanet CB submits
    // fenced/blocking).

    /// Download every GPU-authoritative GatedDeltaNet layer's live conv/delta
    /// state into this model's HOST `layer_state` copy. Call this
    /// immediately before `Qwen35Model::export_prefix` so its host-only
    /// serialization captures the authoritative device values instead of the
    /// stale host copy. No-op (not an error) when `dn_gpu` is empty (engine-less
    /// Mac tests / `DN_GPU` off) — the host copy is already authoritative there.
    pub(crate) fn dn_gpu_sync_to_host(&mut self) -> Result<(), String> {
        // Live-tip fingerprint (export path): the reference an independent
        // fresh-N run emits for the on-cluster off-by-one-vs-drift probe.
        self.kv_boundary_diag_log(usize::MAX, "live_export");
        let Self { dn_gpu, qwen35, .. } = self;
        if dn_gpu.is_empty() {
            return Ok(());
        }
        let m = qwen35
            .as_mut()
            .ok_or("dn_gpu_sync_to_host: no qwen3_5 model resident")?;
        for (&g, layer) in dn_gpu.iter() {
            let si = m.state_idx(g);
            if let qwen35::LayerState::Linear(d) = &mut m.layer_state[si] {
                let conv = read_f32_buf(&layer.conv_state, d.conv_state.len());
                let state = read_f32_buf(&layer.state, d.state.len());
                d.conv_state.copy_from_slice(&conv);
                d.state.copy_from_slice(&state);
            }
        }
        Ok(())
    }

    /// On-cluster KV-boundary diagnostic (`VLLM_VULKAN_KV_BOUNDARY_DIAG=1`).
    /// Log a stable per-resident-linear-layer fingerprint of the DEVICE
    /// conv/delta state (FNV-1a over the raw f32 bit pattern — order- and
    /// value-exact — plus the L2 norm and first element), tagged with the call
    /// site and the token `pos`, reading the SAME host-coherent `dn_gpu`
    /// buffers as `capture_gdn_boundary_resident`/`dn_gpu_sync_to_host`.
    /// Default-off → no overhead, no behavior change (the CPU boundary-snapshot
    /// unit test is unaffected). See `q35_kv_boundary_diag_enabled` for the
    /// off-by-one-vs-drift localization protocol this feeds.
    fn kv_boundary_diag_log(&self, pos: usize, tag: &str) {
        if !q35_kv_boundary_diag_enabled() {
            return;
        }
        let Some(m) = self.qwen35.as_ref() else { return };
        if self.dn_gpu.is_empty() {
            return;
        }
        // FNV-1a (64-bit) over the f32 bit pattern + L2 norm: a change of even
        // one ulp in one element flips the hash, so equal fingerprints across
        // two runs prove byte-identical state; the L2/x0 give a human-readable
        // magnitude to compare against the gate's reported max_abs.
        let fp = |v: &[f32]| -> (u64, f32) {
            let mut hsh: u64 = 0xcbf29ce484222325;
            let mut l2 = 0.0f64;
            for &x in v {
                for b in x.to_bits().to_le_bytes() {
                    hsh ^= b as u64;
                    hsh = hsh.wrapping_mul(0x100000001b3);
                }
                l2 += (x as f64) * (x as f64);
            }
            (hsh, l2.sqrt() as f32)
        };
        // Deterministic layer order (HashMap iteration is not stable across
        // processes, and the operator diffs two processes' logs line-by-line).
        let mut layers: Vec<(&usize, &DnGpuLayer)> = self.dn_gpu.iter().collect();
        layers.sort_by_key(|(g, _)| **g);
        for (&g, layer) in layers {
            let si = m.state_idx(g);
            if let qwen35::LayerState::Linear(d) = &m.layer_state[si] {
                let conv = read_f32_buf(&layer.conv_state, d.conv_state.len());
                let state = read_f32_buf(&layer.state, d.state.len());
                let (ch, cn) = fp(&conv);
                let (sh, sn) = fp(&state);
                log::warn!(
                    "KVDIAG {tag} pos={pos} layer={g} si={si} \
                     conv[n={} l2={:.6} fnv={:016x} x0={:.6}] \
                     state[n={} l2={:.6} fnv={:016x} x0={:.6}]",
                    conv.len(), cn, ch, conv.first().copied().unwrap_or(0.0),
                    state.len(), sn, sh, state.first().copied().unwrap_or(0.0),
                );
            }
        }
    }

    /// KV-offload chunk-boundary capture, resident/device path: at a
    /// `kvstore::CHUNK` boundary (`pos>0 && pos%CHUNK==0`), read back every
    /// GPU-authoritative GatedDeltaNet layer's live conv/delta state — which
    /// at this fenced instant (called right after `begin_forward_ring`
    /// drains the previous token's in-flight CBs, before this token mutates
    /// anything) reflects exactly `[0, pos)` — into
    /// `qwen35.gdn_boundary[pos]`, for `Qwen35Model::export_prefix`'s Linear
    /// arm to serve later. Same read primitive as `dn_gpu_sync_to_host`
    /// (`read_f32_buf` off the host-coherent `dn_gpu` buffers), different
    /// destination (the boundary map, not the live host `d.state`). No-op
    /// when `dn_gpu` is empty (engine-less path — the CPU capture in
    /// `forward_pp_range` covers that) or off-boundary.
    pub(crate) fn capture_gdn_boundary_resident(&mut self, pos: usize) -> Result<(), String> {
        if pos == 0 || pos % crate::kvstore::CHUNK != 0 {
            return Ok(());
        }
        // Fingerprint the exact device state this call is about to snapshot
        // (the `[0, pos)` boundary state). Compared on-cluster against a fresh
        // `[0, pos)` run's `live_export` fingerprint, this confirms whether the
        // capture is bit-identical (→ mismatch is reference-side drift) or off.
        self.kv_boundary_diag_log(pos, "boundary");
        let Self { dn_gpu, qwen35, .. } = self;
        if dn_gpu.is_empty() {
            return Ok(());
        }
        let m = qwen35
            .as_mut()
            .ok_or("capture_gdn_boundary_resident: no qwen3_5 model resident")?;
        let mut snap = std::collections::HashMap::new();
        for (&g, layer) in dn_gpu.iter() {
            let si = m.state_idx(g);
            if let qwen35::LayerState::Linear(d) = &m.layer_state[si] {
                let conv = read_f32_buf(&layer.conv_state, d.conv_state.len());
                let state = read_f32_buf(&layer.state, d.state.len());
                snap.insert(si, (conv, state));
            }
        }
        m.gdn_boundary.retain(|&b, _| b == pos);
        m.gdn_boundary.entry(pos).or_default().extend(snap);
        Ok(())
    }

    /// Upload this model's HOST `layer_state` DeltaNet snapshot — just
    /// restored by `Qwen35Model::import_prefix` — back into the
    /// GPU-authoritative `dn_gpu` buffers, so the next resident-path token
    /// reads the imported state instead of the stale device copy. Call this
    /// immediately after `import_prefix`. Reverse of `dn_gpu_sync_to_host`;
    /// no-op when `dn_gpu` is empty.
    pub(crate) fn dn_gpu_sync_from_host(&mut self) -> Result<(), String> {
        let Self { dn_gpu, qwen35, .. } = self;
        if dn_gpu.is_empty() {
            return Ok(());
        }
        let m = qwen35
            .as_ref()
            .ok_or("dn_gpu_sync_from_host: no qwen3_5 model resident")?;
        for (&g, layer) in dn_gpu.iter() {
            let si = m.state_idx(g);
            if let qwen35::LayerState::Linear(d) = &m.layer_state[si] {
                layer.conv_state.write(&f32_slice_to_bytes(&d.conv_state))?;
                layer.state.write(&f32_slice_to_bytes(&d.state))?;
            }
        }
        Ok(())
    }

    pub(crate) fn ensure_dn_gpu_layer(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize) -> bool {
        if self.dn_gpu.contains_key(&layer_idx) {
            return true;
        }
        let nv = cfg.linear_num_value_heads;
        let vd = cfg.linear_value_head_dim;
        let kern = cfg.linear_conv_kernel_dim;
        let conv_dim = cfg.conv_dim();
        // q35_gdn_step runs one thread per value column (128-thread workgroup)
        // and stages out_j in a 128-slot shared array.
        if vd > 128 || kern == 0 {
            return false;
        }
        let Self { engine, qwen35, dn_gpu, .. } = self;
        let eng = match engine.as_mut() { Some(e) => e, None => return false };
        let m = match qwen35.as_ref() { Some(m) => m, None => return false };
        let p = format!("model.layers.{layer_idx}.linear_attn");
        let t = |name: String| m.weights.tensors.get(&name).map(|t| t.data.as_slice());
        let (conv_w, a_log, dt_bias, norm_w) = match (
            t(format!("{p}.conv1d.weight")),
            t(format!("{p}.A_log")),
            t(format!("{p}.dt_bias")),
            t(format!("{p}.norm.weight")),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return false,
        };
        if conv_w.len() < conv_dim * kern || a_log.len() < nv || dt_bias.len() < nv
            || norm_w.len() < vd
        {
            return false;
        }
        let upload = |eng: &mut compute::ComputeEngine, data: &[f32]| -> Option<compute::Buffer> {
            let bytes = f32_slice_to_bytes(data);
            // .max(4): a kern==1 model has an EMPTY conv window; Vulkan buffers
            // can't be zero-sized.
            let buf = eng.alloc_host_coherent_storage(bytes.len().max(4) as u64).ok()?;
            buf.write(&bytes).ok()?;
            Some(buf)
        };
        let conv_w_buf = match upload(eng, &conv_w[..conv_dim * kern]) {
            Some(b) => b, None => return false,
        };
        // Per-head constants packed as [A_log(nv) | dt_bias(nv) | norm.weight(vd)]
        // — one binding instead of three (q35_gdn_step binds 9 buffers total).
        let mut params = Vec::with_capacity(2 * nv + vd);
        params.extend_from_slice(&a_log[..nv]);
        params.extend_from_slice(&dt_bias[..nv]);
        params.extend_from_slice(&norm_w[..vd]);
        let params_buf = match upload(eng, &params) { Some(b) => b, None => return false };
        let si = m.state_idx(layer_idx);
        let st = match &m.layer_state[si] {
            qwen35::LayerState::Linear(d) => d,
            _ => return false,
        };
        let conv_state_buf = match upload(eng, &st.conv_state) { Some(b) => b, None => return false };
        let state_buf = match upload(eng, &st.state) { Some(b) => b, None => return false };
        dn_gpu.insert(layer_idx, DnGpuLayer {
            conv_w: conv_w_buf,
            params: params_buf,
            conv_state: conv_state_buf,
            state: state_buf,
        });
        true
    }
    /// WS1b: the WHOLE GatedDeltaNet decode block in ONE fenced submit.
    /// in_proj qkv/z/a/b → conv1d+SiLU (window shift in place) → per-head Q/K
    /// RMSNorm+inv-scale → per-v-head delta-rule recurrence + gated RMSNorm →
    /// out_proj, recorded into a single command buffer with barriers between
    /// stages. The conv/delta state stays GTT-resident between tokens (never
    /// read back); the only host traffic is x in (h floats) and the block
    /// output out (h floats). This replaces the CPU path's 2 fenced projection
    /// submits + host conv/recurrence (the measured 64% of q35_deltanet_block).
    ///
    /// Math matches `Qwen35Model::delta_net` with identical f32 accumulation
    /// order everywhere except GPU exp/log/sqrt intrinsics (last-ulp vs libm)
    /// — validated cos >= 0.99999 by `debug_qwen35_gdn_gpu` and argmax-exact
    /// at the PP-5 level. Returns None to fall back to the CPU path.
    pub(crate) fn qwen35_delta_net_gpu_fused(
        &mut self,
        cfg: &qwen35::Qwen35Config,
        layer_idx: usize,
        x: &[f32],
    ) -> Option<Vec<f32>> {
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let kern = cfg.linear_conv_kernel_dim;
        let ratio = nv / nk;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        if x.len() != h {
            return None;
        }
        if !self.ensure_dn_gpu_layer(cfg, layer_idx) {
            return None;
        }
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");
        // Gather the 5 projection weights' GPU meta up front. Raw pointers stay
        // valid through the recording: gpu_weights is never mutated during a
        // forward (same invariant as qwen35_matvec_multi).
        let meta = |vm: &Self, name: &str| -> Option<(*const compute::Buffer, MvKind)> {
            vm.gpu_weights.get(name).map(|w| (
                &w.buffer as *const compute::Buffer,
                match &w.aux {
                    None => MvKind::Plain,
                    Some(QuantAux::Nvfp4 { scales, group_size, e4m3, global }) =>
                        MvKind::Nvfp4 { s: scales as *const _, gs: *group_size, e4m3: *e4m3, global: *global },
                    Some(QuantAux::Fp8 { scale, per_row }) =>
                        MvKind::Fp8 { s: scale as *const _, per_row: *per_row },
                    Some(QuantAux::Mlx4 { scales, biases, group_size }) =>
                        MvKind::Mlx4 { s: scales as *const _, b: biases as *const _, gs: *group_size },
                },
            ))
        };
        let w_qkv = meta(self, &ln("in_proj_qkv.weight"))?;
        let w_z = meta(self, &ln("in_proj_z.weight"))?;
        let w_a = meta(self, &ln("in_proj_a.weight"))?;
        let w_b = meta(self, &ln("in_proj_b.weight"))?;
        let w_out = meta(self, &ln("out_proj.weight"))?;

        let Self { engine, dn_gpu, .. } = self;
        let eng = engine.as_mut()?;
        let layer = &dn_gpu[&layer_idx];

        let t0 = std::time::Instant::now();
        // Pooled per-token buffers (returned below; sizes repeat every layer so
        // the pool serves them without fresh vkAllocate churn).
        let alloc = |eng: &mut compute::ComputeEngine, n: usize| -> Option<compute::Buffer> {
            eng.alloc_host_coherent_storage((n * 4) as u64).ok()
        };
        let inp = alloc(eng, h)?;
        inp.write(&f32_slice_to_bytes(x)).ok()?;
        let b_qkv = alloc(eng, conv_dim)?;
        let b_z = alloc(eng, value_dim)?;
        let b_a = alloc(eng, nv)?;
        let b_b = alloc(eng, nv)?;
        let b_conv = alloc(eng, conv_dim)?;
        let b_q = alloc(eng, key_dim)?;
        let b_k = alloc(eng, key_dim)?;
        let b_gated = alloc(eng, value_dim)?;
        let b_out = alloc(eng, h)?;

        // One matvec dispatch, format-routed exactly as qwen35_matvec_multi.
        let rec_mv = |eng: &mut compute::ComputeEngine,
                      cb: ash::vk::CommandBuffer,
                      (w_ptr, kind): (*const compute::Buffer, MvKind),
                      ip: &compute::Buffer,
                      op: &compute::Buffer,
                      k: usize,
                      n: usize| -> Option<()> {
            unsafe {
                match kind {
                    MvKind::Nvfp4 { s, gs, e4m3, global } => {
                        let (shader, r, pc) = nvfp4_dispatch(k, n, gs, e4m3, global);
                        let wg = (n as u32 + r - 1) / r;
                        eng.record_to(cb, &shader, &[&*w_ptr, &*s, ip, op], &pc, (wg, 1, 1)).ok()?;
                    }
                    MvKind::Fp8 { s, per_row } => {
                        let (shader, r) = matvec_fp8_variant(n);
                        let wg = (n as u32 + r - 1) / r;
                        let pc = matvec_fp8_pc(k, n, per_row);
                        eng.record_to(cb, &shader, &[&*w_ptr, &*s, ip, op], &pc, (wg, 1, 1)).ok()?;
                    }
                    MvKind::Mlx4 { s, b, gs } => {
                        let (shader, r) = matvec_mlx4_variant_k(k, n);
                        let wg = (n as u32 + r - 1) / r;
                        let pc = matvec_mlx4_pc(k, n, gs as usize);
                        eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*b, ip, op], &pc, (wg, 1, 1)).ok()?;
                    }
                    MvKind::Plain => {
                        let (shader, r) = matvec_variant(true, n);
                        let wg = (n as u32 + r - 1) / r;
                        let pc = matvec_pc13(k, n);
                        eng.record_to(cb, &shader, &[&*w_ptr, ip, op], &pc, (wg, 1, 1)).ok()?;
                    }
                }
            }
            Some(())
        };

        let cb = eng.begin_batch().ok()?;
        // Stage 1: the four in_proj matvecs (all read INP; independent).
        rec_mv(eng, cb, w_qkv, &inp, &b_qkv, h, conv_dim)?;
        rec_mv(eng, cb, w_z, &inp, &b_z, h, value_dim)?;
        rec_mv(eng, cb, w_a, &inp, &b_a, h, nv)?;
        rec_mv(eng, cb, w_b, &inp, &b_b, h, nv)?;
        eng.record_barrier_to(cb);
        // Stage 2: depthwise conv1d + SiLU; shifts the resident window in place.
        let pc_conv = q35_conv_pc(conv_dim, kern);
        eng.record_to(cb, "q35_dn_conv_step",
            &[&layer.conv_w, &b_qkv, &layer.conv_state, &b_conv],
            &pc_conv, ((conv_dim as u32 + 255) / 256, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        // Stage 3: per-head Q/K RMSNorm(no-weight) + inv-scale.
        let inv = 1.0f32 / (kd as f32).sqrt();
        let pc_qk = q35_qknorm_pc(nk, kd, key_dim, 1e-6, inv);
        eng.record_to(cb, "q35_gdn_qknorm",
            &[&b_conv, &b_q, &b_k],
            &pc_qk, (2 * nk as u32, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        // Stage 4: delta-rule recurrence + gated RMSNorm (state in place).
        let pc_gdn = q35_gdn_pc(kd, vd, ratio, 2 * key_dim, eps, nv);
        eng.record_to(cb, "q35_gdn_step",
            &[&b_q, &b_k, &b_conv, &b_a, &b_b, &b_z, &layer.params, &layer.state, &b_gated],
            &pc_gdn, (nv as u32, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        // Stage 5: out_proj.
        rec_mv(eng, cb, w_out, &b_gated, &b_out, value_dim, h)?;
        prof_add("q35_dn_gpu_record", t0);

        let ts = std::time::Instant::now();
        eng.submit_batch(cb).ok()?;
        prof_add("q35_dn_gpu_submit", ts);

        let tr = std::time::Instant::now();
        let out = read_f32_buf(&b_out, h);
        for buf in [inp, b_qkv, b_z, b_a, b_b, b_conv, b_q, b_k, b_gated, b_out] {
            eng.return_to_pool(buf);
        }
        prof_add("q35_dn_gpu_readback", tr);
        Some(out)
    }
    /// Dense SwiGLU MLP: gate/up (one submit) on GPU, CPU silu/mul, down on GPU.
    pub(crate) fn qwen35_dense_mlp_gpu(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, ff_in: &[f32]) -> Vec<f32> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let ln = |s: &str| format!("model.layers.{layer_idx}.mlp.{s}");
        let (gn, un) = (ln("gate_proj.weight"), ln("up_proj.weight"));
        let mut gu = self.qwen35_matvec_multi(ff_in, h, &[(&gn, inter), (&un, inter)]);
        let up = gu.pop().unwrap();
        let gate = gu.pop().unwrap();
        let act = model::cpu_silu(&gate);
        let mid: Vec<f32> = act.iter().zip(&up).map(|(&g, &u)| g * u).collect();
        self.qwen35_matvec(&ln("down_proj.weight"), &mid, inter, h)
    }

    /// Q35_TP_FUSED dense SwiGLU MLP: gate/up -> `swiglu_f32` -> down all in ONE
    /// command buffer / ONE submit (was 2: gate/up submit, host silu+mul, down
    /// submit). Rank-local `inter = intermediate_size/n`; sharded gate/up (col)
    /// + down (row) weights already live in `gpu_weights`. Returns the PARTIAL
    /// down_proj (the caller all-reduces it, unchanged). The matvec dispatches
    /// are byte-identical to `qwen35_matvec*` (same variant/pc, geom-flag-aware);
    /// the ONLY numeric delta vs the host path is silu computed with the GPU
    /// `exp` intrinsic instead of libm (last-ulp; cos>=0.99999, argmax-exact).
    /// `None` -> caller falls back to `qwen35_dense_mlp_tp` (the oracle).
    pub(crate) fn qwen35_dense_mlp_tp_fused(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, ff_in: &[f32], n: usize) -> Option<Vec<f32>> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size / n;    // rank-local intermediate
        if ff_in.len() != h { return None; }
        let ln = |s: &str| format!("model.layers.{layer_idx}.mlp.{s}");
        let meta = |vm: &Self, name: &str| -> Option<(*const compute::Buffer, MvKind)> {
            vm.gpu_weights.get(name).map(|w| (
                &w.buffer as *const compute::Buffer,
                match &w.aux {
                    None => MvKind::Plain,
                    Some(QuantAux::Nvfp4 { scales, group_size, e4m3, global }) =>
                        MvKind::Nvfp4 { s: scales as *const _, gs: *group_size, e4m3: *e4m3, global: *global },
                    Some(QuantAux::Fp8 { scale, per_row }) =>
                        MvKind::Fp8 { s: scale as *const _, per_row: *per_row },
                    Some(QuantAux::Mlx4 { scales, biases, group_size }) =>
                        MvKind::Mlx4 { s: scales as *const _, b: biases as *const _, gs: *group_size },
                },
            ))
        };
        let w_gate = meta(self, &ln("gate_proj.weight"))?;
        let w_up   = meta(self, &ln("up_proj.weight"))?;
        let w_down = meta(self, &ln("down_proj.weight"))?;

        let eng = self.engine.as_mut()?;
        let t0 = std::time::Instant::now();
        let alloc = |eng: &mut compute::ComputeEngine, m: usize| -> Option<compute::Buffer> {
            eng.alloc_host_coherent_storage((m * 4) as u64).ok()
        };
        let inp = alloc(eng, h)?;
        inp.write(&f32_slice_to_bytes(ff_in)).ok()?;
        let b_gate = alloc(eng, inter)?;
        let b_up   = alloc(eng, inter)?;
        let b_mid  = alloc(eng, inter)?;
        let b_out  = alloc(eng, h)?;

        let cb = eng.begin_batch().ok()?;
        // Stage 1: gate + up (share INP; independent) — byte-identical dispatch
        // to qwen35_matvec_multi's Plain/quant arms.
        Self::qwen35_tp_fused_mv(eng, cb, w_gate, &inp, &b_gate, h, inter)?;
        Self::qwen35_tp_fused_mv(eng, cb, w_up,   &inp, &b_up,   h, inter)?;
        eng.record_barrier_to(cb);
        // Stage 2: fused swiglu = silu(gate)*up -> b_mid (split mode).
        eng.record_to(cb, "swiglu_f32", &[&b_gate, &b_up, &b_mid],
            &glu_split_pc(inter), ((inter as u32 + 511) / 512, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        // Stage 3: down_proj (row-sharded) -> PARTIAL.
        Self::qwen35_tp_fused_mv(eng, cb, w_down, &b_mid, &b_out, inter, h)?;
        prof_add("q35_tp_mlp_record", t0);

        let ts = std::time::Instant::now();
        eng.submit_batch(cb).ok()?;
        prof_add("q35_tp_mlp_submit", ts);
        let tr = std::time::Instant::now();
        let out = read_f32_buf(&b_out, h);
        for buf in [inp, b_gate, b_up, b_mid, b_out] { eng.return_to_pool(buf); }
        prof_add("q35_tp_mlp_readback", tr);
        Some(out)
    }

    /// ONE matvec dispatch for the Q35_TP_FUSED CBs, format-routed EXACTLY as
    /// `qwen35_matvec`/`qwen35_matvec_multi` (including the Q35_GEOM-flag-aware
    /// Plain arm) so the recorded matvec is byte-identical to the host-
    /// orchestrated path — the fused CB changes only WHERE the intervening
    /// elementwise math runs, never the matvec numerics.
    fn qwen35_tp_fused_mv(
        eng: &mut compute::ComputeEngine,
        cb: ash::vk::CommandBuffer,
        (w_ptr, kind): (*const compute::Buffer, MvKind),
        ip: &compute::Buffer,
        op: &compute::Buffer,
        k: usize,
        n: usize,
    ) -> Option<()> {
        unsafe {
            match kind {
                MvKind::Nvfp4 { s, gs, e4m3, global } => {
                    let (shader, r, pc) = nvfp4_dispatch(k, n, gs, e4m3, global);
                    let wg = (n as u32 + r - 1) / r;
                    eng.record_to(cb, &shader, &[&*w_ptr, &*s, ip, op], &pc, (wg, 1, 1)).ok()?;
                }
                MvKind::Fp8 { s, per_row } => {
                    let (shader, r) = matvec_fp8_variant(n);
                    let wg = (n as u32 + r - 1) / r;
                    let pc = matvec_fp8_pc(k, n, per_row);
                    eng.record_to(cb, &shader, &[&*w_ptr, &*s, ip, op], &pc, (wg, 1, 1)).ok()?;
                }
                MvKind::Mlx4 { s, b, gs } => {
                    let (shader, r) = matvec_mlx4_variant_k(k, n);
                    let wg = (n as u32 + r - 1) / r;
                    let pc = matvec_mlx4_pc(k, n, gs as usize);
                    eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*b, ip, op], &pc, (wg, 1, 1)).ok()?;
                }
                MvKind::Plain => {
                    // Match qwen35_matvec's Plain arm exactly (geom-flag-aware).
                    let (shader, r) = if q35_geom_enabled() {
                        matvec_variant_q35geom(k, n)
                    } else {
                        matvec_variant(true, n)
                    };
                    let wg = (n as u32 + r - 1) / r;
                    let pc = matvec_pc13(k, n);
                    eng.record_to(cb, &shader, &[&*w_ptr, ip, op], &pc, (wg, 1, 1)).ok()?;
                }
            }
        }
        Some(())
    }

    /// Rank-aware variant of `ensure_dn_gpu_layer` for the Q35_TP_FUSED path:
    /// builds the GPU-resident GatedDeltaNet state for THIS rank's 1/n head
    /// shard. The conv1d/A_log/dt_bias host weights are already sharded by the
    /// loader (`q35_tp_shard`), and `layer_state` is already sized per-rank, so
    /// the only difference vs the single-node `ensure_dn_gpu_layer` is that the
    /// dim checks use `nv/n`, `conv_dim/n` (norm.weight stays full `vd`,
    /// replicated). Reuses the same `dn_gpu` map (only one of the two paths runs
    /// per deploy). Returns false -> caller falls back to the host GDN path.
    pub(crate) fn ensure_dn_gpu_layer_tp(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, n: usize) -> bool {
        if self.dn_gpu.contains_key(&layer_idx) { return true; }
        let nv_r = cfg.linear_num_value_heads / n;
        let vd = cfg.linear_value_head_dim;
        let kern = cfg.linear_conv_kernel_dim;
        let conv_dim_r = cfg.conv_dim() / n;
        if vd > 128 || kern == 0 { return false; }
        let Self { engine, qwen35, dn_gpu, .. } = self;
        let eng = match engine.as_mut() { Some(e) => e, None => return false };
        let m = match qwen35.as_ref() { Some(m) => m, None => return false };
        let p = format!("model.layers.{layer_idx}.linear_attn");
        let t = |name: String| m.weights.tensors.get(&name).map(|t| t.data.as_slice());
        let (conv_w, a_log, dt_bias, norm_w) = match (
            t(format!("{p}.conv1d.weight")),
            t(format!("{p}.A_log")),
            t(format!("{p}.dt_bias")),
            t(format!("{p}.norm.weight")),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return false,
        };
        // Sharded lengths: conv1d = conv_dim/n * kern; A_log/dt_bias = nv/n;
        // norm.weight = vd (replicated).
        if conv_w.len() < conv_dim_r * kern || a_log.len() < nv_r || dt_bias.len() < nv_r
            || norm_w.len() < vd
        {
            return false;
        }
        let upload = |eng: &mut compute::ComputeEngine, data: &[f32]| -> Option<compute::Buffer> {
            let bytes = f32_slice_to_bytes(data);
            let buf = eng.alloc_host_coherent_storage(bytes.len().max(4) as u64).ok()?;
            buf.write(&bytes).ok()?;
            Some(buf)
        };
        let conv_w_buf = match upload(eng, &conv_w[..conv_dim_r * kern]) { Some(b) => b, None => return false };
        // [A_log(nv_r) | dt_bias(nv_r) | norm.weight(vd)] — one binding.
        let mut params = Vec::with_capacity(2 * nv_r + vd);
        params.extend_from_slice(&a_log[..nv_r]);
        params.extend_from_slice(&dt_bias[..nv_r]);
        params.extend_from_slice(&norm_w[..vd]);
        let params_buf = match upload(eng, &params) { Some(b) => b, None => return false };
        let si = m.state_idx(layer_idx);
        let st = match &m.layer_state[si] {
            qwen35::LayerState::Linear(d) => d,
            _ => return false,
        };
        let conv_state_buf = match upload(eng, &st.conv_state) { Some(b) => b, None => return false };
        let state_buf = match upload(eng, &st.state) { Some(b) => b, None => return false };
        dn_gpu.insert(layer_idx, DnGpuLayer {
            conv_w: conv_w_buf, params: params_buf,
            conv_state: conv_state_buf, state: state_buf,
        });
        true
    }

    /// Q35_TP_FUSED GatedDeltaNet: the WHOLE per-rank GDN block in ONE fenced
    /// submit — in_proj qkv/z/a/b -> `q35_dn_conv_step` -> `q35_gdn_qknorm` ->
    /// `q35_gdn_step` -> out_proj, all rank-local (1/n heads). Mirrors the
    /// single-node `qwen35_delta_net_gpu_fused` (validated cos>=0.99999,
    /// argmax-exact) but with per-rank nk/nv/key_dim/value_dim/conv_dim; the
    /// conv/delta state stays GTT-resident between tokens. Returns the PARTIAL
    /// out_proj (caller all-reduces, unchanged). `None` -> host GDN fallback.
    pub(crate) fn qwen35_delta_net_tp_fused(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, x: &[f32], n: usize) -> Option<Vec<f32>> {
        let nk_r = cfg.linear_num_key_heads / n;
        let nv_r = cfg.linear_num_value_heads / n;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim_r = nk_r * kd;
        let value_dim_r = nv_r * vd;
        let conv_dim_r = key_dim_r * 2 + value_dim_r;
        let kern = cfg.linear_conv_kernel_dim;
        let ratio = nv_r / nk_r;                 // == nv/nk (GVA ratio, rank-invariant)
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        if x.len() != h { return None; }
        if !self.ensure_dn_gpu_layer_tp(cfg, layer_idx, n) { return None; }
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");
        let meta = |vm: &Self, name: &str| -> Option<(*const compute::Buffer, MvKind)> {
            vm.gpu_weights.get(name).map(|w| (
                &w.buffer as *const compute::Buffer,
                match &w.aux {
                    None => MvKind::Plain,
                    Some(QuantAux::Nvfp4 { scales, group_size, e4m3, global }) =>
                        MvKind::Nvfp4 { s: scales as *const _, gs: *group_size, e4m3: *e4m3, global: *global },
                    Some(QuantAux::Fp8 { scale, per_row }) =>
                        MvKind::Fp8 { s: scale as *const _, per_row: *per_row },
                    Some(QuantAux::Mlx4 { scales, biases, group_size }) =>
                        MvKind::Mlx4 { s: scales as *const _, b: biases as *const _, gs: *group_size },
                },
            ))
        };
        let w_qkv = meta(self, &ln("in_proj_qkv.weight"))?;
        let w_z = meta(self, &ln("in_proj_z.weight"))?;
        let w_a = meta(self, &ln("in_proj_a.weight"))?;
        let w_b = meta(self, &ln("in_proj_b.weight"))?;
        let w_out = meta(self, &ln("out_proj.weight"))?;

        let Self { engine, dn_gpu, .. } = self;
        let eng = engine.as_mut()?;
        let layer = &dn_gpu[&layer_idx];

        let t0 = std::time::Instant::now();
        let alloc = |eng: &mut compute::ComputeEngine, m: usize| -> Option<compute::Buffer> {
            eng.alloc_host_coherent_storage((m * 4) as u64).ok()
        };
        let inp = alloc(eng, h)?;
        inp.write(&f32_slice_to_bytes(x)).ok()?;
        let b_qkv = alloc(eng, conv_dim_r)?;
        let b_z = alloc(eng, value_dim_r)?;
        let b_a = alloc(eng, nv_r)?;
        let b_b = alloc(eng, nv_r)?;
        let b_conv = alloc(eng, conv_dim_r)?;
        let b_q = alloc(eng, key_dim_r)?;
        let b_k = alloc(eng, key_dim_r)?;
        let b_gated = alloc(eng, value_dim_r)?;
        let b_out = alloc(eng, h)?;

        let cb = eng.begin_batch().ok()?;
        // Stage 1: four in_proj matvecs (all read INP; independent).
        Self::qwen35_tp_fused_mv(eng, cb, w_qkv, &inp, &b_qkv, h, conv_dim_r)?;
        Self::qwen35_tp_fused_mv(eng, cb, w_z, &inp, &b_z, h, value_dim_r)?;
        Self::qwen35_tp_fused_mv(eng, cb, w_a, &inp, &b_a, h, nv_r)?;
        Self::qwen35_tp_fused_mv(eng, cb, w_b, &inp, &b_b, h, nv_r)?;
        eng.record_barrier_to(cb);
        // Stage 2: depthwise conv1d + SiLU; shifts the resident window in place.
        let pc_conv = q35_conv_pc(conv_dim_r, kern);
        eng.record_to(cb, "q35_dn_conv_step",
            &[&layer.conv_w, &b_qkv, &layer.conv_state, &b_conv],
            &pc_conv, ((conv_dim_r as u32 + 255) / 256, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        // Stage 3: per-head Q/K RMSNorm(no-weight) + inv-scale.
        let inv = 1.0f32 / (kd as f32).sqrt();
        let pc_qk = q35_qknorm_pc(nk_r, kd, key_dim_r, 1e-6, inv);
        eng.record_to(cb, "q35_gdn_qknorm",
            &[&b_conv, &b_q, &b_k],
            &pc_qk, (2 * nk_r as u32, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        // Stage 4: delta-rule recurrence + gated RMSNorm (state in place).
        let pc_gdn = q35_gdn_pc(kd, vd, ratio, 2 * key_dim_r, eps, nv_r);
        eng.record_to(cb, "q35_gdn_step",
            &[&b_q, &b_k, &b_conv, &b_a, &b_b, &b_z, &layer.params, &layer.state, &b_gated],
            &pc_gdn, (nv_r as u32, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        // Stage 5: out_proj -> PARTIAL (caller all-reduces).
        Self::qwen35_tp_fused_mv(eng, cb, w_out, &b_gated, &b_out, value_dim_r, h)?;
        prof_add("q35_tp_dn_record", t0);

        let ts = std::time::Instant::now();
        eng.submit_batch(cb).ok()?;
        prof_add("q35_tp_dn_submit", ts);
        let tr = std::time::Instant::now();
        let out = read_f32_buf(&b_out, h);
        for buf in [inp, b_qkv, b_z, b_a, b_b, b_conv, b_q, b_k, b_gated, b_out] {
            eng.return_to_pool(buf);
        }
        prof_add("q35_tp_dn_readback", tr);
        Some(out)
    }

    /// MoE MLP (35B-A3B), one token. The router gate + the 8 active routed
    /// experts + shared expert are computed on the CPU from the host-f32 expert
    /// tensors (`moe::moe_forward_token_borrowed`): each expert matmul is tiny
    /// (gate/up 2048->512, down 512->2048) and only 8 of 256 + shared run per
    /// token, so the ~3B active params/token are CPU-cheap and avoid uploading
    /// the full ~18GB expert set to the GPU. The big GPU win (per-layer fused
    /// dense projections) does not apply to sparse MoE; this is the correct
    /// wiring. Expert tensors live in `self.qwen35.weights` (host f32, kept off
    /// the GPU matvec sink by `is_qwen35_moe_weight_name`).
    pub(crate) fn qwen35_moe_mlp_gpu(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, ff_in: &[f32]) -> Vec<f32> {
        let dims = moe::MoeDims {
            hidden: cfg.hidden_size,
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            moe_inter: cfg.moe_intermediate_size,
            shared_inter: cfg.shared_expert_intermediate_size,
        };
        // GPU-resident 4-bit expert path (the decode win): only when the layer's
        // experts are loaded packed (quant_moe) AND the flag is on AND we have an
        // engine. Falls through to the CPU path otherwise (the A/B baseline).
        if moe_gpu_enabled()
            && self.engine.is_some()
            && self.qwen35.as_ref().unwrap().quant_moe.gate.contains_key(&layer_idx)
        {
            if let Some(out) = self.qwen35_moe_mlp_gpu_resident(layer_idx, ff_in, dims) {
                return out;
            }
        }

        let m = self.qwen35.as_ref().unwrap();
        let (_routing, out) = if m.quant_moe.gate.contains_key(&layer_idx) {
            moe::moe_forward_token_quant(ff_in, &m.weights, &m.quant_moe, layer_idx, dims)
        } else {
            moe::moe_forward_token_borrowed(ff_in, &m.weights, layer_idx, dims)
        };
        out
    }

    // ── P1b/P2: qwen3.6 batched-prefill GPU helpers (plan-batched-prefill.md) ──


    pub(crate) fn qwen35_gemm(&mut self, weight_name: &str, x: &[f32], t: usize, k: usize, n: usize) -> Vec<f32> {
        // Design-A batched-verify projection swap (spec §6): during a verify pass
        // (`spec_verify_cols`, set only around the attn/GDN mixers in
        // `forward_qwen35_verify_core`), route eligible f16/q8_0 projections
        // through the single-stream cols kernel — weight streamed ONCE for all `t`
        // columns. MoE keeps its grouped MUL_MAT_ID GEMM (the flag is off there),
        // and prefill/decode are untouched (flag off), so numerics never change
        // outside the verify projections. Falls through to the GEMM/serial paths
        // for any weight/geometry the cols kernel can't take.
        if self.spec_verify_cols {
            // Column-tiled so a long PREFILL projection (t = prompt/chunk len,
            // possibly >8) reuses the streamed weight across <=8-token tiles;
            // spec-verify (t<=8) collapses to a single dispatch, unchanged.
            if let Some(out) = self.qwen35_matvec_cols_tiled(weight_name, x, t, k, n) {
                return out;
            }
        }
        let (variant, bm, bn) = gemm_variant_k(k, n);
        let (bm, bn) = (bm as usize, bn as usize);
        // The `matmul_f16_f32_*` GEMM family only knows how to read a PLAIN
        // f16/f32 weight buffer — no batched-T GEMM variant exists for
        // Q8_0/Nvfp4/Fp8-resident weights (unlike `qwen35_matvec`, which picks
        // the correct dequantizing shader per `format`/`aux`). A real load with
        // `VLLM_VULKAN_QUANT=q8_0` or an NVFP4/FP8 checkpoint would otherwise
        // silently reinterpret quantized bytes as f16 here — wrong numbers, not
        // a crash. Gate the fast batched path on "plain" gpu_weights and fall
        // through to `t` serial `qwen35_matvec` calls (which also cover the
        // chunked-lm_head / f16-host / f32-host sources) for everything else —
        // correct for every real loader configuration, just not batched.
        let plain_gpu = self.gpu_weights.get(weight_name)
            .filter(|w| w.aux.is_none() && matches!(w.format, crate::flags::QuantFormat::F16 | crate::flags::QuantFormat::F32))
            .map(|w| &w.buffer as *const compute::Buffer);
        if let (Some(eng), Some(w_ptr)) = (self.engine.as_mut(), plain_gpu) {
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((t * k * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let out = eng.alloc_host_coherent_storage((t * n * 4) as u64).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let out_p = &out as *const compute::Buffer;
            let pc = gemm_pc(t, n, k);
            let wg = (((n + bm - 1) / bm) as u32, ((t + bn - 1) / bn) as u32, 1u32);
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, &variant, &[&*w_ptr, &*inp_p, &*out_p], &pc, wg).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            let result = read_f32_buf(&out, t * n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
            result
        } else if gemm_quant_flag() && t >= 2 {
            // Phase A quant-batched-GEMM (plan-quant-batched-matmul.md,
            // VLLM_VULKAN_GEMM_QUANT=1, default OFF): the "plain_gpu" fast
            // path above only covers f16/f32 weights with no aux buffers --
            // q8_0/mlx4-resident weights fell through to `t` serial
            // qwen35_matvec calls. Route Q8_0 (no aux) and Mlx4 (aux =
            // separate scales/biases buffers) through the new quant `mul_mm`
            // variants instead, when there's more than one token to amortize
            // the dequant over (T=1 gets no reuse benefit -- see the campaign
            // plan's crossover-T argument -- and stays on the serial matvec
            // path, which is already bit-exact-by-construction at T=1).
            let quant_meta = self.gpu_weights.get(weight_name).and_then(|w| match (&w.format, &w.aux) {
                (crate::flags::QuantFormat::Q8_0, None) =>
                    Some(("matmul_q8_0_f32_fp32", &w.buffer as *const compute::Buffer, None)),
                (crate::flags::QuantFormat::Mlx4, Some(QuantAux::Mlx4 { scales, biases, group_size })) =>
                    Some(("matmul_mlx4_f32_fp32", &w.buffer as *const compute::Buffer,
                          Some((scales as *const compute::Buffer, biases as *const compute::Buffer, *group_size)))),
                _ => None,
            });
            if let (Some(eng), Some((base, w_ptr, mlx4))) = (self.engine.as_mut(), quant_meta) {
                let (variant, qbm, qbn) = gemm_variant_quant_k(base, k, n);
                let (qbm, qbn) = (qbm as usize, qbn as usize);
                let xb = f32_slice_to_bytes(x);
                let inp = eng.alloc_host_coherent_storage((t * k * 4) as u64).unwrap();
                inp.write(&xb).unwrap();
                let out = eng.alloc_host_coherent_storage((t * n * 4) as u64).unwrap();
                let inp_p = &inp as *const compute::Buffer;
                let out_p = &out as *const compute::Buffer;
                let wg = (((n + qbm - 1) / qbm) as u32, ((t + qbn - 1) / qbn) as u32, 1u32);
                let cb = eng.begin_batch().unwrap();
                unsafe {
                    match mlx4 {
                        Some((s, b, gs)) => {
                            let pc = gemm_pc_mlx4(t, n, k, gs as usize, 0, 0);
                            eng.record_to(cb, &variant, &[&*w_ptr, &*inp_p, &*out_p, &*s, &*b], &pc, wg).unwrap();
                        }
                        None => {
                            let pc = gemm_pc(t, n, k);
                            eng.record_to(cb, &variant, &[&*w_ptr, &*inp_p, &*out_p], &pc, wg).unwrap();
                        }
                    }
                }
                eng.submit_batch(cb).unwrap();
                let result = read_f32_buf(&out, t * n);
                eng.return_to_pool(inp);
                eng.return_to_pool(out);
                result
            } else {
                let mut out = vec![0.0f32; t * n];
                for ti in 0..t {
                    let xi = &x[ti * k..(ti + 1) * k];
                    let oi = self.qwen35_matvec(weight_name, xi, k, n);
                    out[ti * n..(ti + 1) * n].copy_from_slice(&oi);
                }
                out
            }
        } else {
            let mut out = vec![0.0f32; t * n];
            for ti in 0..t {
                let xi = &x[ti * k..(ti + 1) * k];
                let oi = self.qwen35_matvec(weight_name, xi, k, n);
                out[ti * n..(ti + 1) * n].copy_from_slice(&oi);
            }
            out
        }
    }

    /// P1b: ONE full-attention (`FullAttention`) layer's prefill output for
    /// `t_count` tokens starting at `start_pos`, via batched GPU GEMM
    /// projections + `gpu_flash_attn`. Mirrors `Qwen35Model::gated_attention`
    /// (qwen35.rs:446) exactly, just batched over T instead of one token:
    ///   1. Batched double-width q_proj -> split [query(hd)|gate(hd)] per head
    ///      (interleaved layout, `base = head*2*hd`, NOT two contiguous halves
    ///      — see `gated_attention`).
    ///   2. Batched k_proj/v_proj.
    ///   3. Per-token per-head Q/K RMSNorm (before RoPE), then partial RoPE
    ///      (rotary = `cfg.rotary_dim()`, theta = `cfg.rope_theta`) at each
    ///      token's ABSOLUTE position `start_pos + ti` — cheap CPU elementwise,
    ///      same order as `gated_attention`.
    ///   4. Append all T K/V to this layer's resident `KvCache` (leaves it in
    ///      the exact state serial decode expects to continue from).
    ///   5. `gpu_flash_attn` with a pure causal `[T x kv_len]` mask (no window
    ///      — qwen3.6 full-attn is non-sliding); Q laid out `[head][pos][hd]`.
    ///   6. Output gate `attn_out[i] * sigmoid(gate[i])`, THEN batched o_proj.
    ///
    /// `xs`: `[t_count * hidden_size]` row-major (already input-norm'd by the
    /// caller, matching `gated_attention`'s contract). Returns `[t_count *
    /// hidden_size]` row-major.
    pub(crate) fn qwen35_full_attn_prefill_gpu(
        &mut self,
        cfg: &qwen35::Qwen35Config,
        layer_idx: usize,
        xs: &[f32],
        t_count: usize,
        start_pos: usize,
    ) -> Vec<f32> {
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let rotary = cfg.rotary_dim();
        let theta = cfg.rope_theta;
        let ln = |s: &str| format!("model.layers.{layer_idx}.self_attn.{s}");

        // 1-2. Batched double-width q_proj + k_proj + v_proj GEMMs.
        let q_and_gate = self.qwen35_gemm(&ln("q_proj.weight"), xs, t_count, h, nq * hd * 2);
        let mut k = self.qwen35_gemm(&ln("k_proj.weight"), xs, t_count, h, kv_dim);
        let v = self.qwen35_gemm(&ln("v_proj.weight"), xs, t_count, h, kv_dim);

        let mut q = vec![0.0f32; t_count * q_dim];
        let mut gate = vec![0.0f32; t_count * q_dim];
        for ti in 0..t_count {
            let row = &q_and_gate[ti * nq * hd * 2..(ti + 1) * nq * hd * 2];
            for head in 0..nq {
                let base = head * 2 * hd;
                q[ti * q_dim + head * hd..ti * q_dim + (head + 1) * hd]
                    .copy_from_slice(&row[base..base + hd]);
                gate[ti * q_dim + head * hd..ti * q_dim + (head + 1) * hd]
                    .copy_from_slice(&row[base + hd..base + 2 * hd]);
            }
        }

        // 3. Per-token per-head Q/K RMSNorm then partial RoPE at the token's
        // absolute position.
        let qn = self.qwen35_w(&ln("q_norm.weight"));
        let kn = self.qwen35_w(&ln("k_norm.weight"));
        for ti in 0..t_count {
            {
                let qrow = &mut q[ti * q_dim..(ti + 1) * q_dim];
                for hi in 0..nq {
                    let s = &mut qrow[hi * hd..(hi + 1) * hd];
                    let n = model::cpu_rms_norm(s, &qn, eps);
                    s.copy_from_slice(&n);
                }
            }
            {
                let krow = &mut k[ti * kv_dim..(ti + 1) * kv_dim];
                for hi in 0..nkv {
                    let s = &mut krow[hi * hd..(hi + 1) * hd];
                    let n = model::cpu_rms_norm(s, &kn, eps);
                    s.copy_from_slice(&n);
                }
            }
            let pos = start_pos + ti;
            let mut qr = q[ti * q_dim..(ti + 1) * q_dim].to_vec();
            let mut kr = k[ti * kv_dim..(ti + 1) * kv_dim].to_vec();
            model::cpu_rope(&mut qr, &mut kr, pos, nq, nkv, hd, rotary, theta);
            q[ti * q_dim..(ti + 1) * q_dim].copy_from_slice(&qr);
            k[ti * kv_dim..(ti + 1) * kv_dim].copy_from_slice(&kr);
        }

        // 4. Append all T K/V to the resident KvCache (state_idx resolved
        // fresh here — PP-safe, mirrors `qwen35_gated_attention_gpu`).
        let si = self.qwen35.as_ref().unwrap().state_idx(layer_idx);
        let kv_len = {
            let qm = self.qwen35.as_mut().unwrap();
            let cache = match &mut qm.layer_state[si] {
                qwen35::LayerState::Full(c) => c,
                _ => unreachable!("full_attention layer has a KV cache"),
            };
            for ti in 0..t_count {
                cache.append(&k[ti * kv_dim..(ti + 1) * kv_dim], &v[ti * kv_dim..(ti + 1) * kv_dim]);
            }
            cache.seq_len
        };

        // 5. Flash attention: Q reshaped [head][pos][head_dim]; K/V read back
        // from the cache up to kv_len; pure causal mask (no sliding window).
        let mut qflash = vec![0.0f32; nq * t_count * hd];
        for ti in 0..t_count {
            for hi in 0..nq {
                let src = &q[ti * q_dim + hi * hd..ti * q_dim + (hi + 1) * hd];
                let dst = (hi * t_count + ti) * hd;
                qflash[dst..dst + hd].copy_from_slice(src);
            }
        }
        let (kbuf, vbuf) = {
            let qm = self.qwen35.as_ref().unwrap();
            let cache = match &qm.layer_state[si] {
                qwen35::LayerState::Full(c) => c,
                _ => unreachable!("full_attention layer has a KV cache"),
            };
            (cache.k_up_to_now()[..kv_len * kv_dim].to_vec(),
             cache.v_up_to_now()[..kv_len * kv_dim].to_vec())
        };
        let neg = f32::NEG_INFINITY;
        let mut mask = vec![0.0f32; t_count * kv_len];
        for i in 0..t_count {
            let qpos = start_pos + i;
            for j in 0..kv_len {
                if j > qpos { mask[i * kv_len + j] = neg; }
            }
        }
        // STEP-7: the batched full-attn SDPA is per-token f32 `cpu_sdpa` BY
        // DEFAULT — bit-exact vs the serial single-token decode reference, which
        // uses `cpu_sdpa` too (GPU-resident SDPA / `gpu_flash_attn` is
        // default-OFF; the flash kernel additionally casts K/V to f16, so it is
        // NOT bit-exact vs the f32 decode — the residual full-attn divergence the
        // harness saw). The projections stay batched (the weight-streaming win);
        // only the cheap attention COMPUTE is per-token. Opt back into the
        // batched f16 `gpu_flash_attn` with `VLLM_VULKAN_SPEC_ATTN_FLASH=1` once
        // it is validated argmax-exact.
        if !std::env::var("VLLM_VULKAN_SPEC_ATTN_FLASH").map(|v| v != "0").unwrap_or(false) {
            // Perf review P1b: the T (=DEPTH+1) per-position `cpu_sdpa` calls are
            // mutually independent (each reads only its own query row + the
            // shared, already-materialized k/v prefix; none writes shared
            // state) — fan them across cores with rayon exactly like the MoE
            // per-token routing above. CRITICAL: only the ACROSS-calls loop is
            // parallelized; each individual `cpu_sdpa` call's internal
            // summation order is untouched, so per-position output — and thus
            // the concatenated `attn_out` — is bit-identical to the serial loop.
            let rows: Vec<Vec<f32>> = {
                use rayon::prelude::*;
                (0..t_count).into_par_iter()
                    .map(|ti| {
                        let causal = start_pos + ti + 1;          // this query's KV length
                        model::cpu_sdpa(&q[ti * q_dim..(ti + 1) * q_dim],
                            &kbuf[..causal * kv_dim], &vbuf[..causal * kv_dim],
                            nq, nkv, hd, causal, scale, None)
                    })
                    .collect()
            };
            let mut attn_out = vec![0.0f32; t_count * q_dim];
            for (ti, o) in rows.into_iter().enumerate() {
                attn_out[ti * q_dim..(ti + 1) * q_dim].copy_from_slice(&o);
            }
            let gated: Vec<f32> = attn_out.iter().zip(&gate)
                .map(|(&a, &g)| a * (1.0 / (1.0 + (-g).exp()))).collect();
            return self.qwen35_gemm(&ln("o_proj.weight"), &gated, t_count, q_dim, h);
        }

        let attn_out_hm = self.gpu_flash_attn(&qflash, &kbuf, &vbuf, nq, nkv, hd,
                                           t_count, kv_len, scale, Some(&mask));
        // `gpu_flash_attn` returns O in the SAME HEAD-MAJOR `[head][query][hd]`
        // layout as its `qflash` input, but the output gate + `o_proj` below are
        // TOKEN-MAJOR `[pos][head][hd]` (like `gate`). Transpose back. WITHOUT
        // this, T>1 pairs each token's gate/o_proj row with the WRONG attention
        // output — scrambled logits at every position — while T==1 is unaffected
        // (the two layouts coincide when there is a single query), which is
        // exactly why the single-token decode + the per-token CPU verify path
        // (Mac identity gate) never exposed it and the batched GPU verify did.
        let mut attn_out = vec![0.0f32; t_count * q_dim];
        for hi in 0..nq {
            for ti in 0..t_count {
                let src = (hi * t_count + ti) * hd;
                let dst = ti * q_dim + hi * hd;
                attn_out[dst..dst + hd].copy_from_slice(&attn_out_hm[src..src + hd]);
            }
        }

        // 6. Output gate then batched o_proj.
        let gated: Vec<f32> = attn_out.iter().zip(&gate)
            .map(|(&a, &g)| a * (1.0 / (1.0 + (-g).exp()))).collect();
        self.qwen35_gemm(&ln("o_proj.weight"), &gated, t_count, q_dim, h)
    }

    /// P2 GPU-resident path: ONE `LinearAttention` layer's prefill output for
    /// `t_count` tokens. CPU `gdn_scan_prepass` (batched projections + causal
    /// conv1d+SiLU + qk-norm, advancing the CPU-side `DeltaNetState.conv_state`
    /// across all T tokens) feeds the GPU `q35_gdn_scan` shader, which
    /// dispatches DIRECTLY against the persistent `dn_gpu[layer_idx].state`
    /// buffer (read-modify-write in place — the SAME buffer
    /// `qwen35_delta_net_gpu_fused`'s decode path reads/writes), so decode
    /// continues seamlessly after this call with no extra `state` sync.
    ///
    /// `conv_state`, however, is NOT touched by `q35_gdn_scan` (it consumes
    /// already-resolved `conv_out` from the CPU prepass) — so after the
    /// prepass advances the CPU-side window, we explicitly re-upload it into
    /// `dn_gpu[layer_idx].conv_state` so the NEXT per-token GPU decode call
    /// (whose conv1d shader reads that buffer) sees the correct window.
    ///
    /// FLAG: this assumes the CPU-side `DeltaNetState.conv_state` is
    /// authoritative going INTO this call (true for prefill running before any
    /// GPU-resident decode on this layer, or from a freshly reset sequence).
    /// If a GPU-resident decode step already advanced
    /// `dn_gpu[layer_idx].conv_state` independently (per WS1b: the CPU copy
    /// "goes stale" once DN_GPU decode has run for a layer), calling this
    /// method afterward would feed `gdn_scan_prepass` a STALE conv window
    /// instead of the GPU-authoritative one — a real staleness bug for
    /// "prefill resumes after decode on the same layer" call patterns. Fixing
    /// it would need a device→host `conv_state` readback before the prepass
    /// (symmetric with the upload direction here); out of scope for this
    /// pass — flagging per the plan rather than guessing at the right sync
    /// point.
    ///
    /// Returns `None` if the GDN-scan shader can't engage (dims exceed its
    /// 128-column workgroup, or a weight is missing) — the caller should fall
    /// back to T serial `qwen35_delta_net_gpu` calls.
    pub(crate) fn qwen35_linear_prefill_gpu(
        &mut self,
        cfg: &qwen35::Qwen35Config,
        layer_idx: usize,
        xs: &[f32],
        t_count: usize,
    ) -> Option<Vec<f32>> {
        let h = cfg.hidden_size;
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let ratio = nv / cfg.linear_num_key_heads;
        let eps = cfg.rms_norm_eps;
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");

        if !self.ensure_dn_gpu_layer(cfg, layer_idx) {
            return None;
        }

        // STEP-7 conv-window staleness fix (resolves the FLAG above): seed the
        // CPU `DeltaNetState.conv_state` from the GPU-AUTHORITATIVE
        // `dn_gpu[layer].conv_state` before the prepass. The single-token decode
        // path (`qwen35_delta_net_gpu_fused`, DN_GPU default ON) advances the
        // device conv buffer in place and leaves the CPU copy stale, so
        // `gdn_scan_conv_qknorm` below would convolve a STALE causal window for
        // the batch's early tokens (cos worst at position 0, recovering as the
        // in-batch window fills — the residual batched-verify divergence). This
        // is the device→host readback the FLAG describes, symmetric with the
        // post-prepass CPU→GPU upload. No-op engine-less (dn_gpu empty ⇒ CPU copy
        // already authoritative), so the Mac identity path is unchanged.
        if let Some(layer) = self.dn_gpu.get(&layer_idx) {
            let mut cbytes = vec![0u8; layer.conv_state.size as usize];
            if layer.conv_state.read(&mut cbytes).is_ok() {
                let conv: &[f32] = bytemuck::cast_slice(&cbytes);
                let qm = self.qwen35.as_mut().unwrap();
                let si = qm.state_idx(layer_idx);
                if let qwen35::LayerState::Linear(d) = &mut qm.layer_state[si] {
                    if d.conv_state.len() == conv.len() {
                        d.conv_state.copy_from_slice(conv);
                    }
                }
            }
        }

        // Batched GPU projections (mirrors `qwen35_delta_net_gpu`'s decode-path
        // `qwen35_matvec_multi` call): `in_proj_*` are matvec weights the
        // split loader streams to the GPU and DROPS from the host f32 store on
        // a real lean load, so — UNLIKE `gdn_scan_prepass` (CPU-only path,
        // reads `self.weights` directly) — this must go through `qwen35_gemm`,
        // not the host store. `a`/`b`/`z` need no further CPU processing.
        let (qkv_n, z_n, a_n, b_n) = (
            ln("in_proj_qkv.weight"), ln("in_proj_z.weight"),
            ln("in_proj_a.weight"), ln("in_proj_b.weight"),
        );
        let qkv = self.qwen35_gemm(&qkv_n, xs, t_count, h, conv_dim);
        let z = self.qwen35_gemm(&z_n, xs, t_count, h, value_dim);
        let a = self.qwen35_gemm(&a_n, xs, t_count, h, nv);
        let b = self.qwen35_gemm(&b_n, xs, t_count, h, nv);

        // CPU tail: causal conv1d+SiLU + qk-norm over all T tokens, advancing
        // `layer_state[si].conv_state` in place (`conv1d.weight` is NOT a
        // matvec weight, always host-f32-resident — safe to read here).
        let (q, k, conv_out) =
            self.qwen35.as_mut().unwrap().gdn_scan_conv_qknorm(layer_idx, &qkv, t_count);

        // Re-sync the resident conv window post-prepass (see FLAG above).
        {
            let qm = self.qwen35.as_ref().unwrap();
            let si = qm.state_idx(layer_idx);
            let cpu_conv_state = match &qm.layer_state[si] {
                qwen35::LayerState::Linear(d) => d.conv_state.clone(),
                _ => unreachable!("linear_attention layer has a DeltaNet state"),
            };
            self.dn_gpu.get(&layer_idx)?
                .conv_state.write(&f32_slice_to_bytes(&cpu_conv_state)).ok()?;
        }

        let Self { engine, dn_gpu, .. } = self;
        let eng = engine.as_mut()?;
        let layer = &dn_gpu[&layer_idx];

        let upload = |eng: &mut compute::ComputeEngine, data: &[f32]| -> Option<compute::Buffer> {
            let bytes = f32_slice_to_bytes(data);
            let buf = eng.alloc_host_coherent_storage(bytes.len().max(4) as u64).ok()?;
            buf.write(&bytes).ok()?;
            Some(buf)
        };
        let g_q = upload(eng, &q)?;
        let g_k = upload(eng, &k)?;
        let g_conv = upload(eng, &conv_out)?;
        let g_a = upload(eng, &a)?;
        let g_b = upload(eng, &b)?;
        let g_z = upload(eng, &z)?;
        let g_gated = eng.alloc_host_coherent_storage((t_count * value_dim * 4) as u64).ok()?;

        let pc = q35_gdn_scan_pc(kd, vd, ratio, 2 * key_dim, eps, nv, t_count, conv_dim, key_dim, value_dim);
        let cb = eng.begin_batch().ok()?;
        eng.record_to(cb, "q35_gdn_scan",
            &[&g_q, &g_k, &g_conv, &g_a, &g_b, &g_z, &layer.params, &layer.state, &g_gated],
            &pc, (nv as u32, 1, 1)).ok()?;
        eng.submit_batch(cb).ok()?;
        let gated = read_f32_buf(&g_gated, t_count * value_dim);
        for buf in [g_q, g_k, g_conv, g_a, g_b, g_z, g_gated] {
            eng.return_to_pool(buf);
        }

        // out_proj (batched GEMM); `engine`/`dn_gpu` borrows end here so this
        // is free to re-borrow `self` mutably.
        Some(self.qwen35_gemm(&ln("out_proj.weight"), &gated, t_count, value_dim, h))
    }

    /// P2 GPU-resident path: dense SwiGLU MLP batched over `t_count` tokens
    /// (mirrors `forward_prefill_gemma`'s FFN block, retargeted at qwen3_5
    /// weights via `qwen35_gemm`).
    pub(crate) fn qwen35_dense_mlp_prefill_gpu(
        &mut self,
        cfg: &qwen35::Qwen35Config,
        layer_idx: usize,
        ff_in: &[f32],
        t_count: usize,
    ) -> Vec<f32> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let ln = |s: &str| format!("model.layers.{layer_idx}.mlp.{s}");
        let gate = self.qwen35_gemm(&ln("gate_proj.weight"), ff_in, t_count, h, inter);
        let up = self.qwen35_gemm(&ln("up_proj.weight"), ff_in, t_count, h, inter);
        let act = model::cpu_silu(&gate);
        let mid: Vec<f32> = act.iter().zip(&up).map(|(&g, &u)| g * u).collect();
        self.qwen35_gemm(&ln("down_proj.weight"), &mid, t_count, inter, h)
    }

    /// P2 GPU-resident path: MoE MLP for `t_count` tokens.
    //
    // Phase B (item-1-phase-3): when VLLM_VULKAN_MOE_GEMM=1 (default OFF) and
    // this layer's experts are 4-bit-resident, the routed-expert matmuls are
    // batched across ALL T tokens via the grouped MUL_MAT_ID GEMM
    // (`qwen35_moe_mlp_prefill_gpu_grouped`) — one dispatch per gate/up/down
    // over every routed (token,slot) pair instead of T*top_k per-token matvecs.
    // Falls through to the per-token loop below when the flag is off, there is
    // no engine, the layer isn't quant-resident, or the grouped path returns
    // None (the unchanged A/B baseline).
    pub(crate) fn qwen35_moe_mlp_prefill_gpu(
        &mut self,
        cfg: &qwen35::Qwen35Config,
        layer_idx: usize,
        ff_in: &[f32],
        t_count: usize,
    ) -> Vec<f32> {
        let h = cfg.hidden_size;
        // The grouped MUL_MAT_ID prefill GEMM (`matmul_mlx4_id`) reads the resident
        // expert scales as f32. Under VLLM_VULKAN_MOE_F16_SCALES those buffers are
        // f16, so skip it and use the per-token resident matvec path (which reads
        // the f16 scales via `mul_mat_vec_mlx4_f16scale_f32_f32`). Prefill for the
        // near-empty 122B prompt is a handful of tokens — correctness over the
        // grouped-GEMM speedup here. (Reconciliation: an f16-scale matmul_mlx4_id
        // twin would restore batched prefill; noted, not needed for first-fit.)
        if moe_gemm_enabled()
            && !crate::moe_f16_scales_flag()
            && self.engine.is_some()
            && self.qwen35.as_ref().unwrap().quant_moe.gate.contains_key(&layer_idx)
        {
            if let Some(o) = self.qwen35_moe_mlp_prefill_gpu_grouped(cfg, layer_idx, ff_in, t_count) {
                return o;
            }
        }
        let mut out = vec![0.0f32; t_count * h];
        for ti in 0..t_count {
            let fi = &ff_in[ti * h..(ti + 1) * h];
            let o = self.qwen35_moe_mlp_gpu(cfg, layer_idx, fi);
            out[ti * h..(ti + 1) * h].copy_from_slice(&o);
        }
        out
    }

    /// Phase B: batched MoE MLP for `t_count` tokens via the grouped
    /// MUL_MAT_ID + MLX4 GEMM (`matmul_mlx4_id_f32_fp32`). Mirrors
    /// `qwen35_moe_mlp_gpu_resident`'s fused command buffer but batched over T:
    /// the routed gate/up/down each collapse to ONE grouped dispatch that reads
    /// the on-GPU `ids`/`counts` routing tables, gathers the right B rows,
    /// dequantizes the per-expert 4-bit weight in-shader, and scatters results
    /// straight to (token,slot) order — no host-side token sort/permute. The
    /// dense f32 shared expert is batched alongside via `matmul_f32_f32_fp32`
    /// in the same CB. Routing is now ALSO on GPU: a separate submit-1 runs a
    /// single batched `[T,h]x[e,h]->[T,e]` GEMM (`matmul_f32_f32_fp32`) over
    /// the resident `shared.router` weight — the same buffer
    /// `qwen35_moe_mlp_gpu_resident`'s per-token decode path reads — instead of
    /// T unbatched per-token `moe::route()` calls on the host (that loop alone
    /// was 71% of the MoE block, diagnosed as the whole Phase B regression).
    /// Only softmax/top-k/renorm (`route_from_logits`, ~256 floats/token) and
    /// the final score/sigmoid combine stay on the host — negligible, and
    /// bit-identical to `moe_forward_token_quant`. Returns None to fall back
    /// to the per-token loop.
    ///
    /// The `ne11` push-constant is THE correctness pivot (decoded from the
    /// shader): gate/up use `ne11=1` (broadcast a token's x across all top_k
    /// slots), down uses `ne11=top_k` (each slot reads its own `mid` row). See
    /// `gemm_pc_mlx4_id`.
    pub(crate) fn qwen35_moe_mlp_prefill_gpu_grouped(
        &mut self,
        cfg: &qwen35::Qwen35Config,
        layer_idx: usize,
        ff_in: &[f32],
        t_count: usize,
    ) -> Option<Vec<f32>> {
        let h = cfg.hidden_size;
        let mi = cfg.moe_intermediate_size;
        let si = cfg.shared_expert_intermediate_size;
        let e = cfg.num_experts;
        let top_k = cfg.num_experts_per_tok;
        let t = t_count;
        if t == 0 { return Some(Vec::new()); }
        if !self.ensure_moe_gpu_layer(layer_idx) { return None; }
        if !self.ensure_moe_shared_gpu_layer(layer_idx) { return None; }

        let Self { engine, moe_gpu, moe_shared_gpu, .. } = self;
        let eng = engine.as_mut()?;
        let layer = &moe_gpu[&layer_idx];
        let shared = &moe_shared_gpu[&layer_idx];
        let group_size = layer.gate.group_size;

        // Token activations — read by the router GEMM, all routed experts AND
        // the shared expert (uploaded once, shared across every submit below).
        let inp = eng.alloc_host_coherent_storage((t * h * 4) as u64).ok()?;
        inp.write(&f32_slice_to_bytes(ff_in)).ok()?;

        // ── Submit-1 (GPU router): ONE batched [T,h]x[e,h]->[T,e] GEMM over
        //    the already-resident `shared.router` weight — the SAME buffer
        //    `qwen35_moe_mlp_gpu_resident`'s per-token decode path reads at its
        //    submit 1. Replaces the old per-token `moe::route()` host loop (T
        //    calls into unbatched single-core `cpu_matmul`, each re-reading the
        //    2MB router weight): that loop alone was 71% of the MoE block and
        //    more than accounted for the whole Phase B regression. softmax /
        //    top-k / renorm (`route_from_logits`) stays on the host — it's a
        //    per-token reduction over only `e`=256 floats, negligible.
        const RBM: u32 = 64;
        const RBN: u32 = 64;
        let logits_buf = eng.alloc_host_coherent_storage((t * e * 4) as u64).ok()?;
        let pc_router = gemm_pc(t, e, h);
        let wg_router = (((e as u32) + RBM - 1) / RBM, ((t as u32) + RBN - 1) / RBN, 1);
        {
            let cb = eng.begin_batch().ok()?;
            eng.record_to(cb, "matmul_f32_f32_fp32", &[&shared.router, &inp, &logits_buf], &pc_router, wg_router).ok()?;
            eng.submit_batch(cb).ok()?;
        }
        let logits = read_f32_buf(&logits_buf, t * e);
        eng.return_to_pool(logits_buf);
        // Per-token softmax/top-k/renorm is independent across tokens; fan it
        // across cores (at T=2048 this is 2048 single-core reductions over
        // e experts each — measurably super-linear on one core at large T).
        // Order-independent → bit-identical routing (keeps cos == 1.0).
        let routings: Vec<moe::Routing> = {
            use rayon::prelude::*;
            (0..t).into_par_iter()
                .map(|ti| moe::route_from_logits(&logits[ti * e..(ti + 1) * e], top_k))
                .collect()
        };

        // ── Step 1: ids[t*top_k] (token→expert, ascending-expert-id slot order
        //    matching Routing.indices) + counts[e] histogram.
        let mut ids = vec![0i32; t * top_k];
        let mut counts = vec![0i32; e];
        for ti in 0..t {
            for slot in 0..top_k {
                let ex = routings[ti].indices[slot];
                ids[ti * top_k + slot] = ex as i32;
                counts[ex] += 1;
            }
        }
        let max_count = *counts.iter().max().unwrap_or(&0);
        if max_count <= 0 { return None; }
        let wy = |bn: u32| ((max_count as u32) + bn - 1) / bn;

        // Uploads: ids, counts (token activations already uploaded above).
        let ids_buf = eng.alloc_host_coherent_storage((ids.len() * 4) as u64).ok()?;
        ids_buf.write(bytemuck::cast_slice::<i32, u8>(&ids)).ok()?;
        let counts_buf = eng.alloc_host_coherent_storage((counts.len() * 4) as u64).ok()?;
        counts_buf.write(bytemuck::cast_slice::<i32, u8>(&counts)).ok()?;

        // Intermediate/output buffers (routed are [t*top_k*·], shared [t*·]).
        // Epilogue-fused MoE GEMM (VLLM_VULKAN_MOE_GEMM_FUSED, default OFF —
        // see plan-epilogue-fused-moe-gemm.md): when fused, the routed
        // gu_gate/gu_up/act_r buffers are never allocated -- gate+up+silu+mul
        // collapse into ONE dispatch of matmul_mlx4_id_gateup_silu_f32_fp32
        // that writes mid_r directly (silu(gate)*up computed in its store
        // epilogue). The shared (dense) expert path is untouched either way.
        let fused = moe_gemm_fused_enabled();
        // Phase 3 (plan §5): fuse the score-weighted routed combine +
        // sigmoid-gated shared add into ONE batched on-GPU dispatch
        // (`q35_moe_accum_batched`) so `down_out` never leaves VRAM (readback
        // 16384->2048 f/tok) and the host rayon combine is gone. Independent of
        // `fused` (works whether or not gate/up was epilogue-fused: it consumes
        // `down_out`/`s_out`/`s_logit`, produced identically either way).
        let combine = moe_gemm_combine_enabled();
        let mut a = |n: usize| eng.alloc_host_coherent_storage((n * 4) as u64);
        let gu_gate = if fused { None } else { Some(a(t * top_k * mi).ok()?) };
        let gu_up = if fused { None } else { Some(a(t * top_k * mi).ok()?) };
        let act_r = if fused { None } else { Some(a(t * top_k * mi).ok()?) };
        let mid_r = a(t * top_k * mi).ok()?;
        let down_out = a(t * top_k * h).ok()?;
        // Phase-3-only buffers: per-(token,slot) routed scores (uploaded) and
        // the on-GPU out[T,h] the batched combine writes (the sole readback).
        let (scores_buf, out_buf) = if combine {
            let mut scores = vec![0.0f32; t * top_k];
            for ti in 0..t {
                scores[ti * top_k..(ti + 1) * top_k].copy_from_slice(&routings[ti].scores);
            }
            let sb = a(t * top_k).ok()?;
            sb.write(&f32_slice_to_bytes(&scores)).ok()?;
            (Some(sb), Some(a(t * h).ok()?))
        } else {
            (None, None)
        };
        let s_gate = a(t * si).ok()?;
        let s_up = a(t * si).ok()?;
        let s_act = a(t * si).ok()?;
        let s_mid = a(t * si).ok()?;
        let s_out = a(t * h).ok()?;
        let s_logit = a(t).ok()?;

        // Grouped-GEMM geometry: gate/up (M=mi,K=h), down (M=h,K=mi). Base
        // (BM=BN=64) unless a swept sibling covers the (k,n) shape.
        let base = "matmul_mlx4_id_f32_fp32";
        let gateup_base = "matmul_mlx4_id_gateup_silu_f32_fp32";
        let (gu_variant, gu_bm, gu_bn) = gemm_variant_quant_k(base, h, mi);
        let (dn_variant, dn_bm, dn_bn) = gemm_variant_quant_k(base, mi, h);
        // Phase 2 sparse-BN (plan-epilogue-fused-moe-gemm.md §3): the fused
        // gate+up dispatch is ONLY ever recorded when `fused` is true (see
        // below), so picking its BN by tokens/expert here cannot affect the
        // `fused=false` grouped path at all -- gu_variant/dn_variant above
        // stay on the shape-keyed `gemm_variant_quant_k` picker unchanged,
        // preserving the "VLLM_VULKAN_MOE_GEMM_FUSED=0 -> existing grouped
        // path verbatim" contract. `avg_tokens_per_expert` uses the mean
        // (t*top_k/e) rather than `max_count` -- routing skew means the
        // busiest expert can still need the BN=64 column tile even when the
        // mean is small, but the per-workgroup early-return
        // (`ic*BN >= count`) already makes an over-sized BN choice merely
        // wasteful (not incorrect) for any one expert, so a mean-based pick
        // is a safe first cut (a max_count-based picker is a possible
        // follow-up refinement, not required for correctness).
        let avg_tokens_per_expert = (t * top_k) / e.max(1);
        let (gateup_variant, gateup_bm, gateup_bn) = gemm_variant_quant_id_bn(gateup_base, avg_tokens_per_expert);
        let wg_gu = (((mi as u32) + gu_bm - 1) / gu_bm, wy(gu_bn), e as u32);
        let wg_dn = (((h as u32) + dn_bm - 1) / dn_bm, wy(dn_bn), e as u32);
        let wg_gateup = (((mi as u32) + gateup_bm - 1) / gateup_bm, wy(gateup_bn), e as u32);

        // ne11=1 broadcast (gate/up): each token's x is `h` apart in `inp`.
        let pc_gate = gemm_pc_mlx4_id(mi, h, top_k, t, 1, h, h, mi, top_k * mi,
            group_size, layer.gate.pack_stride, layer.gate.sb_stride);
        let pc_up = gemm_pc_mlx4_id(mi, h, top_k, t, 1, h, h, mi, top_k * mi,
            group_size, layer.up.pack_stride, layer.up.sb_stride);
        let pc_gateup = gemm_pc_mlx4_id_gateup(mi, h, top_k, t, 1, h, h, mi, top_k * mi,
            group_size, layer.gate.pack_stride, layer.gate.sb_stride,
            layer.up.pack_stride, layer.up.sb_stride);
        // ne11=top_k per-slot (down): mid rows are `mi` apart, `top_k*mi` per token.
        let pc_down = gemm_pc_mlx4_id(h, mi, top_k, t, top_k, mi, top_k * mi, h, top_k * h,
            group_size, layer.down.pack_stride, layer.down.sb_stride);

        // Shared dense f32 GEMMs (BM=BN=64 base geometry) + expert-gate logit.
        const SBM: u32 = 64;
        const SBN: u32 = 64;
        let pc_sgu = gemm_pc(t, si, h);
        let pc_sdown = gemm_pc(t, h, si);
        let pc_slogit = gemm_pc(t, 1, h);
        let wg_sgu = (((si as u32) + SBM - 1) / SBM, ((t as u32) + SBN - 1) / SBN, 1);
        let wg_sdown = (((h as u32) + SBM - 1) / SBM, ((t as u32) + SBN - 1) / SBN, 1);
        let wg_slogit = (1u32, ((t as u32) + SBN - 1) / SBN, 1);

        let n_s = (t * si) as u32;
        let silu_s = ew_unary_pc(n_s);
        let mul_s = ew_mul_pc(n_s);

        // n_r/silu_r/mul_r only exist (and are only recorded) on the
        // non-fused path — declared here so both the dispatch-recording
        // block and nothing else needs them in scope.
        let n_r = (t * top_k * mi) as u32;
        let silu_r = ew_unary_pc(n_r);
        let mul_r = ew_mul_pc(n_r);

        let cb = eng.begin_batch().ok()?;
        if fused {
            // ONE dispatch replaces gate-GEMM + up-GEMM + silu_f32 + mul_f32
            // (4 dispatches in the `else` arm below) -- gate.packed is
            // binding 0/A, layer.up.packed is the SECOND A-binding (7/8/9),
            // sharing `inp` (binding 1/B) between both projections. Writes
            // mid_r directly; no barrier needed before the shared dispatches
            // below (disjoint buffers), same as the `else` arm's gate/up.
            eng.record_to(cb, &gateup_variant,
                &[&layer.gate.packed, &inp, &mid_r, &ids_buf, &counts_buf,
                  &layer.gate.scales, &layer.gate.biases,
                  &layer.up.packed, &layer.up.scales, &layer.up.biases],
                &pc_gateup, wg_gateup).ok()?;
        } else {
            eng.record_to(cb, &gu_variant,
                &[&layer.gate.packed, &inp, gu_gate.as_ref().unwrap(), &ids_buf, &counts_buf, &layer.gate.scales, &layer.gate.biases],
                &pc_gate, wg_gu).ok()?;
            eng.record_to(cb, &gu_variant,
                &[&layer.up.packed, &inp, gu_up.as_ref().unwrap(), &ids_buf, &counts_buf, &layer.up.scales, &layer.up.biases],
                &pc_up, wg_gu).ok()?;
        }
        // Shared (dense) expert path -- identical either way, unaffected by
        // the routed gate/up/silu/mul fusion above. Same 3-barrier structure
        // as before fusion (barrier/silu/barrier/mul/barrier/down) -- the
        // fused path just has nothing routed left to gate on the first two
        // barriers (mid_r is already final), while the non-fused path's
        // routed silu/mul ride along with the shared ones under the SAME
        // barriers, exactly as they did pre-fusion.
        eng.record_to(cb, "matmul_f32_f32_fp32", &[&shared.gate, &inp, &s_gate], &pc_sgu, wg_sgu).ok()?;
        eng.record_to(cb, "matmul_f32_f32_fp32", &[&shared.up, &inp, &s_up], &pc_sgu, wg_sgu).ok()?;
        eng.record_to(cb, "matmul_f32_f32_fp32", &[&shared.expert_gate, &inp, &s_logit], &pc_slogit, wg_slogit).ok()?;
        eng.record_barrier_to(cb);
        if !fused {
            eng.record_to(cb, "silu_f32", &[gu_gate.as_ref().unwrap(), act_r.as_ref().unwrap()], &silu_r, ((n_r + 511) / 512, 1, 1)).ok()?;
        }
        eng.record_to(cb, "silu_f32", &[&s_gate, &s_act], &silu_s, ((n_s + 511) / 512, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        if !fused {
            eng.record_to(cb, "mul_f32_f32_f32", &[act_r.as_ref().unwrap(), gu_up.as_ref().unwrap(), &mid_r], &mul_r, ((n_r + 255) / 256, 1, 1)).ok()?;
        }
        eng.record_to(cb, "mul_f32_f32_f32", &[&s_act, &s_up, &s_mid], &mul_s, ((n_s + 255) / 256, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        eng.record_to(cb, &dn_variant,
            &[&layer.down.packed, &mid_r, &down_out, &ids_buf, &counts_buf, &layer.down.scales, &layer.down.biases],
            &pc_down, wg_dn).ok()?;
        eng.record_to(cb, "matmul_f32_f32_fp32", &[&shared.down, &s_mid, &s_out], &pc_sdown, wg_sdown).ok()?;
        if combine {
            // Phase 3: batched on-GPU combine. Barrier: `down_out` (routed
            // down GEMM) and `s_out`/`s_logit` (shared expert) must be fully
            // written before the reduce reads them. Gather-reduce over the
            // top_k contiguous down rows per token -> out[T,h], no scatter/
            // atomics; fixed slot order == the host loop below -> cos=1.0.
            eng.record_barrier_to(cb);
            let pc_acc = q35_moe_accum_batched_pc(t, h, top_k);
            eng.record_to(cb, "q35_moe_accum_batched",
                &[&down_out, scores_buf.as_ref().unwrap(), &s_out, &s_logit, out_buf.as_ref().unwrap()],
                &pc_acc, (((t * h) as u32 + 255) / 256, 1, 1)).ok()?;
        }
        eng.submit_batch(cb).ok()?;

        // ── Step 6: score-weighted routed combine + sigmoid-gated shared add.
        if combine {
            // Phase 3: the batched dispatch already produced out[T,h] on-GPU;
            // read it back (the ONLY routed readback — `down_out` stayed in
            // VRAM) and return. No host reduction.
            let out = read_f32_buf(out_buf.as_ref().unwrap(), t * h);
            for b in gu_gate.into_iter().chain(gu_up).chain(act_r)
                .chain(scores_buf).chain(out_buf)
                .chain([inp, ids_buf, counts_buf, mid_r, down_out,
                        s_gate, s_up, s_act, s_mid, s_out, s_logit]) {
                eng.return_to_pool(b);
            }
            return Some(out);
        }

        let down_host = read_f32_buf(&down_out, t * top_k * h);
        let shared_host = read_f32_buf(&s_out, t * h);
        let logit_host = read_f32_buf(&s_logit, t);

        for b in gu_gate.into_iter().chain(gu_up).chain(act_r)
            .chain([inp, ids_buf, counts_buf, mid_r, down_out,
                    s_gate, s_up, s_act, s_mid, s_out, s_logit]) {
            eng.return_to_pool(b);
        }

        // Weighted combine on the already-un-permuted [t,top_k,h] down output
        // (the shader scattered to (token,slot) order) + the sigmoid-gated
        // shared expert. Embarrassingly parallel over tokens (each `ti` writes
        // a disjoint `out[ti*h..]` slice); fan it across cores. At large T this
        // host reduction over `t*top_k*h` is a measurable slice of the
        // grouped-MoE block. Bit-identical to the serial loop (per-token
        // reduction order unchanged) → keeps cos == 1.0.
        let mut out = vec![0.0f32; t * h];
        {
            use rayon::prelude::*;
            out.par_chunks_mut(h).enumerate().for_each(|(ti, o)| {
                let r = &routings[ti];
                for slot in 0..top_k {
                    let sc = r.scores[slot];
                    let base_o = (ti * top_k + slot) * h;
                    let src = &down_host[base_o..base_o + h];
                    for (oj, &d) in o.iter_mut().zip(src) { *oj += sc * d; }
                }
                let sg = 1.0f32 / (1.0 + (-logit_host[ti]).exp());
                let src = &shared_host[ti * h..(ti + 1) * h];
                for (oj, &s) in o.iter_mut().zip(src) { *oj += sg * s; }
            });
        }
        Some(out)
    }

    /// P2: `forward_qwen35_prefill` — batched prefill entry point for a
    /// qwen3.6 (qwen3_5) model. Embeds `tokens` (`[T]`), runs THIS STAGE's
    /// resident decoder layers (`[self.pp_start, self.pp_end)`) batched over
    /// T, and — only on the last PP stage — returns the LAST token's logits;
    /// a non-last stage returns the raw `[T,h]` hidden output instead.
    ///
    /// PP-range-aware (mirrors the decode path, `forward_pp_qwen35_impl`):
    /// only the stage's resident layers run (no full-model residency
    /// required), and `state_idx`/`dn_gpu`/KV lookups inside the per-layer
    /// prefill helpers already resolve stage-locally from the GLOBAL
    /// `layer_idx` this loop passes them — same as the decode path. This is
    /// what makes a single resident stage (the r0solo pattern, `pp_start=0,
    /// pp_end=8` on an 8-layer stage) run without needing the other 32
    /// layers' MoE experts dequantized to host f32 (the OOM this fixes).
    ///
    /// SCOPE: this unblocks the SINGLE-stage resident prefill benchmark only.
    /// Wiring the full PP-5 batched-prefill HOP — shipping a non-last stage's
    /// `[T,h]` hidden to the next stage, and accepting a `hidden_in` on a
    /// non-first stage instead of always embedding from `tokens` — is a
    /// separate, larger follow-up (the embed-side FLAG below).
    ///
    /// Two execution paths:
    ///   - CPU fallback (`self.engine.is_none()`): delegates to
    ///     `Qwen35Model::forward_pp_range_batched(.., pp_start, pp_end)`,
    ///     which reproduces `forward_pp_range` called `t_count` times EXACTLY
    ///     (see that method's doc comment) — the required correctness gate.
    ///   - GPU-resident (`self.engine.is_some()`): batched GEMM projections
    ///     (`qwen35_full_attn_prefill_gpu` / `qwen35_dense_mlp_prefill_gpu`)
    ///     + the resident GDN-scan shader (`qwen35_linear_prefill_gpu`) for
    ///     `LinearAttention` layers, falling back to T serial
    ///     `qwen35_delta_net_gpu` calls if the scan shader can't engage. MoE
    ///     stays per-token (see the TODO on `qwen35_moe_mlp_prefill_gpu`).
    ///
    /// lm_head: computed ONLY for the last token's hidden state (not all T) —
    /// decode only ever needs the last position's logits to pick the next
    /// token, so an all-T lm_head GEMM is both wasted work and, at large T,
    /// hangs on GFX1013 — and ONLY on `self.pp_last` (mirrors
    /// `forward_pp_qwen35_impl`'s `if last { .. lm_head .. } else { hidden }`;
    /// a non-last stage doesn't have `model.norm`/lm_head weights resident).
    ///
    /// `start_pos`: the KV/DeltaNet position this batch continues from (NOT
    /// reset here — unlike `forward_prefill`/`forward_prefill_gemma`, callers
    /// that want a fresh sequence must call `reset_kv_cache()` themselves;
    /// this mirrors `forward_batched_impl`'s "resume from start_pos" contract
    /// so the same entry point can serve both a first prefill (`start_pos=0`
    /// on a freshly reset model) and a continuation).
    pub(crate) fn forward_qwen35_prefill_impl(&mut self, tokens: Vec<u32>, start_pos: usize) -> PyResult<Vec<f32>> {
        let t = tokens.len();
        if t == 0 {
            return Err(PyRuntimeError::new_err("forward_qwen35_prefill: empty prompt"));
        }
        let cfg = self.qwen35.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forward_qwen35_prefill needs a qwen3_5 model"))?
            .config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let vocab = cfg.vocab_size;

        // Embed all T tokens -> hidden[T,h]. `embed_tokens` is ALWAYS split
        // into the f16 host table (`q35_f16_host`, per `want_f16` in
        // `load_qwen35_weights_split` — never the f32 store), exactly like the
        // decode path (`forward_pp_qwen35_impl`/`forward_qwen35_gpu`) reads it
        // — dequantize f16->f32 per row, NOT `weights.f32_slice`, which would
        // panic on any real (split-loader) load.
        //
        // FLAG (scoped out of this fix): this always embeds from `tokens`,
        // i.e. it assumes `self.pp_first` (the resident stage owns
        // `embed_tokens`, per `lib.rs`'s `pp_first || pp_last` weight-filter).
        // A true middle PP stage would need a `hidden_in` parameter instead
        // (mirroring `forward_pp_qwen35_impl`'s `first`-gated embed) — that's
        // part of the same inter-stage batched-prefill HOP noted below as a
        // separate follow-up. This fix only guarantees correctness for a
        // single resident stage that is both first and last (the r0solo
        // pattern), which is what the queued benchmark exercises.
        let mut hidden: Vec<f32> = {
            let mut hv = vec![0.0f32; t * h];
            for (ti, &tok) in tokens.iter().enumerate() {
                hv[ti * h..(ti + 1) * h].copy_from_slice(&self.q35_embed_row(tok as usize, h));
            }
            hv
        };

        // ── CPU fallback (no GPU engine) ────────────────────────────────────
        // PP-range-aware (mirrors the decode path's `self.pp_start..self.pp_end`
        // + `self.pp_last`-gated final norm/lm_head): only this stage's
        // resident layers run, and only the last stage produces logits — a
        // non-last stage returns the raw stage-output hidden `[T,h]` (the
        // inter-stage batched-prefill HOP that would consume this on the next
        // stage is NOT wired yet; that's a separate, larger follow-up item).
        let (pp_start, pp_end, pp_last) = (self.pp_start, self.pp_end, self.pp_last);
        if self.engine.is_none() {
            let qm = self.qwen35.as_mut().unwrap();
            hidden = qm.forward_pp_range_batched(&hidden, start_pos, t, pp_start, pp_end);
            if !pp_last {
                return Ok(hidden);
            }
            let norm_w = qm.weights.f32_slice("model.norm.weight").to_vec();
            let last = &hidden[(t - 1) * h..t * h];
            let normed = model::cpu_rms_norm(last, &norm_w, eps);
            let lm_name = qm.lm_head_name.clone();
            // lm_head is ALWAYS split into the f16 host table by the loader
            // (`want_f16` in `load_qwen35_weights_split` is unconditional on
            // engine presence), never `weights.f32_slice` — mirrors the
            // GPU-resident path's `qwen35_matvec` f16-host fallback tier, and
            // is what the (engine-less) decode path effectively falls back to
            // as well.
            let lm_w = self.q35_f16_host.get(&lm_name)
                .expect("qwen3_5 lm_head f16 host missing (prefill CPU-fallback last stage)");
            let lm_f32: Vec<f32> = lm_w.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            return Ok(model::cpu_matmul(&normed, &lm_f32, 1, h, vocab));
        }

        // ── GPU-resident path ────────────────────────────────────────────────
        // PREFILL cols batching (VLLM_VULKAN_QWEN35_PREFILL_COLS, default OFF):
        // route the attn q/k/v/o + GDN in/out_proj + DENSE-MLP gate/up/down
        // weight projections through the single-stream `mul_mat_vec_*_cols`
        // kernels (weight streamed ONCE per <=8-token tile) instead of the
        // T-serial per-token matvecs these projections fall through to today for
        // mlx4/q8_0-resident weights. Scoped with `spec_verify_cols` EXACTLY as
        // `forward_qwen35_verify_core` does (proven bit-exact-per-column):
        // set around the attn/GDN mixer + the dense MLP, and LEFT OFF around the
        // MoE block so experts keep their grouped MUL_MAT_ID GEMM. OFF =>
        // untouched serial prefill (byte-identical); the cols dispatch is
        // argmax-exact / cos=1.0 (mlx4-cols ACO gate n53) but reorders the f32
        // reduction, so default-on awaits the on-node prefill tok/s + argmax
        // gate. NOTE: `qwen35_matvec_cols_tiled` caps the projection batch at
        // <=8 columns/tile, so a whole-prompt call with T>8 still engages cols
        // (per <=8-token tile); a caller may also drive fixed <=8-token prompt
        // tiles (see the deferred on-node gate notes).
        let cols = self.flags.qwen35_prefill_cols;
        for layer_idx in pp_start..pp_end {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
            let residual = hidden.clone();
            let in_ln = self.qwen35_w(&ln("input_layernorm.weight"));
            let x = model::cpu_rms_norm(&hidden, &in_ln, eps);
            if cols { self.spec_verify_cols = true; }
            let attn_out = match cfg.layer_types[layer_idx] {
                qwen35::LayerType::FullAttention =>
                    self.qwen35_full_attn_prefill_gpu(&cfg, layer_idx, &x, t, start_pos),
                qwen35::LayerType::LinearAttention => {
                    match self.qwen35_linear_prefill_gpu(&cfg, layer_idx, &x, t) {
                        Some(out) => out,
                        None => {
                            // Resident GDN-scan couldn't engage: fall back to
                            // T serial fused-decode calls (same per-token
                            // math, just not batched into one dispatch). Each is
                            // a t=1 projection, so cols no-ops (t<2) regardless.
                            let mut out = vec![0.0f32; t * h];
                            for ti in 0..t {
                                let xt = &x[ti * h..(ti + 1) * h];
                                let o = self.qwen35_delta_net_gpu(&cfg, layer_idx, xt);
                                out[ti * h..(ti + 1) * h].copy_from_slice(&o);
                            }
                            out
                        }
                    }
                }
            };
            if cols { self.spec_verify_cols = false; }
            let h1: Vec<f32> = residual.iter().zip(&attn_out).map(|(&r, &a)| r + a).collect();

            let residual2 = h1.clone();
            let post_ln = self.qwen35_w(&ln("post_attention_layernorm.weight"));
            let ff_in = model::cpu_rms_norm(&h1, &post_ln, eps);
            let mlp_out = if cfg.is_moe() {
                // MoE: grouped MUL_MAT_ID GEMM handles its own T-batching — the
                // cols flag stays OFF here (never routes experts through cols).
                self.qwen35_moe_mlp_prefill_gpu(&cfg, layer_idx, &ff_in, t)
            } else {
                // DENSE MLP: gate/up/down go through `qwen35_gemm` — cols-batch
                // them (the largest 4-bit weights in the dense 27B).
                if cols { self.spec_verify_cols = true; }
                let o = self.qwen35_dense_mlp_prefill_gpu(&cfg, layer_idx, &ff_in, t);
                if cols { self.spec_verify_cols = false; }
                o
            };
            hidden = residual2.iter().zip(&mlp_out).map(|(&r, &m)| r + m).collect();
            // KvStore 256-token-boundary snapshot hook: NOT WIRED (HELD,
            // pending a KV device-capture fix) — would fire once per
            // 256-boundary crossed in [start_pos, start_pos+t) here (see the
            // batched-prefill plan §5).
        }

        if !pp_last {
            // Non-last stage: return the stage's `[T,h]` hidden output (no
            // norm/lm_head — only the last stage owns those weights). Feeding
            // this to the next stage is the PP-hop follow-up (out of scope
            // here; this unblocks the SINGLE-stage resident benchmark).
            return Ok(hidden);
        }
        // Last stage: only the LAST token's logits are needed to seed decode
        // (mirrors `forward_pp_qwen35_impl`'s lm_head handling) — running
        // lm_head over all T tokens is both wasted GEMM work and, at large T,
        // hangs the lm_head matvec on GFX1013. Uses `qwen35_matvec` directly
        // (the EXACT call `forward_pp_qwen35_impl` makes), not `qwen35_gemm`
        // — lm_head is the one weight that can be chunked across several GPU
        // buffers (`chunked_weights`, the ~2.5GB alloc mitigation) or live
        // ONLY in the f16 host table (upload failure / CPU-only); `qwen35_gemm`
        // only ever reads a single `gpu_weights` entry, so it can't see either
        // of those sources — `qwen35_matvec` already resolves all of them.
        let norm_w = self.qwen35_w("model.norm.weight");
        let last = &hidden[(t - 1) * h..t * h];
        let normed = model::cpu_rms_norm(last, &norm_w, eps);
        let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
        Ok(self.qwen35_matvec(&lm_name, &normed, h, vocab))
    }

    /// Design-A batched VERIFY entry (MTP re-gate, spec §2). First/single stage:
    /// embed `tokens` (`[s_R, d_1..d_D]`, T=D+1) then run the verify stack. See
    /// `forward_qwen35_verify_core` for the all-T-position logits + GDN-capture
    /// contract. Middle/last PP stages take the previous stage's `[T,h]` via
    /// `forward_qwen35_verify_core` directly (see `pp_step_qwen35_verify`).
    pub(crate) fn forward_qwen35_verify_impl(&mut self, tokens: Vec<u32>, start_pos: usize) -> PyResult<Vec<f32>> {
        let t = tokens.len();
        if t == 0 {
            return Err(PyRuntimeError::new_err("forward_qwen35_verify: empty verify batch"));
        }
        let h = self.qwen35.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forward_qwen35_verify needs a qwen3_5 model"))?
            .config.hidden_size;
        // Embed all T tokens -> hidden[T,h], f16 host table (identical to
        // `forward_qwen35_prefill_impl`; the resident/first stage owns embed).
        let hidden: Vec<f32> = {
            let w = self.q35_f16_host.get("model.embed_tokens.weight")
                .expect("qwen3_5 embed_tokens f16 host missing (verify first stage)");
            let mut hv = vec![0.0f32; t * h];
            for (ti, &tok) in tokens.iter().enumerate() {
                hv[ti * h..(ti + 1) * h]
                    .iter_mut()
                    .zip(&w[tok as usize * h..(tok as usize + 1) * h])
                    .for_each(|(dst, &b)| *dst = half::f16::from_bits(b).to_f32());
            }
            hv
        };
        self.forward_qwen35_verify_core(hidden, start_pos, t)
    }

    /// Design-A batched VERIFY core (MTP re-gate, spec §2/§3). Runs the same
    /// T-token hybrid stack as `forward_qwen35_prefill_impl`, but:
    ///   (2) the LAST stage emits logits for ALL T positions (`[T*vocab]`, row
    ///       `ti` = argmax candidate `out(ti)`), not just the last — the chain
    ///       verifier needs every position's argmax (`a_out, b_1..b_D`);
    ///   (3) each `LinearAttention` layer's batched input `x` is captured into
    ///       `spec_verify_gdn_inputs` (with `spec_verify_span=(start_pos,T)`) so
    ///       a partial-accept `qwen35_verify_rollback_impl` re-scans the
    ///       committed prefix through the GDN layers only (option-B).
    /// A non-last stage returns the stage `[T,h]` hidden (the batched-PP-hop
    /// payload). `hidden` is the stage input (embedded tokens on the first
    /// stage, previous stage's `[T,h]` on a middle/last stage).
    pub(crate) fn forward_qwen35_verify_core(&mut self, mut hidden: Vec<f32>, start_pos: usize, t: usize) -> PyResult<Vec<f32>> {
        let cfg = self.qwen35.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forward_qwen35_verify needs a qwen3_5 model"))?
            .config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let vocab = cfg.vocab_size;
        self.spec_verify_gdn_inputs.clear();
        self.spec_verify_span = Some((start_pos, t));
        let (pp_start, pp_end, pp_last) = (self.pp_start, self.pp_end, self.pp_last);

        // ── CPU fallback (no GPU engine) — Mac identity-gate path ────────────
        if self.engine.is_none() {
            let mut cap: Vec<(usize, Vec<f32>)> = Vec::new();
            let qm = self.qwen35.as_mut().unwrap();
            hidden = qm.forward_pp_range_batched_capture(&hidden, start_pos, t, pp_start, pp_end, Some(&mut cap));
            self.spec_verify_gdn_inputs = cap;
            if !pp_last {
                return Ok(hidden);
            }
            let qm = self.qwen35.as_mut().unwrap();
            let norm_w = qm.weights.f32_slice("model.norm.weight").to_vec();
            let lm_name = qm.lm_head_name.clone();
            let lm_w = self.q35_f16_host.get(&lm_name)
                .expect("qwen3_5 lm_head f16 host missing (verify CPU-fallback last stage)");
            let lm_f32: Vec<f32> = lm_w.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            self.stash_verify_prenorm(&hidden, start_pos, t, h);
            let mut logits = vec![0.0f32; t * vocab];
            for ti in 0..t {
                let normed = model::cpu_rms_norm(&hidden[ti * h..(ti + 1) * h], &norm_w, eps);
                let l = model::cpu_matmul(&normed, &lm_f32, 1, h, vocab);
                logits[ti * vocab..(ti + 1) * vocab].copy_from_slice(&l);
            }
            return Ok(logits);
        }

        // ── GPU-resident path (mirrors forward_qwen35_prefill_impl's loop,
        //    plus the GDN-input capture) ──────────────────────────────────────
        for layer_idx in pp_start..pp_end {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
            let residual = hidden.clone();
            let in_ln = self.qwen35_w(&ln("input_layernorm.weight"));
            let x = model::cpu_rms_norm(&hidden, &in_ln, eps);
            // §6 projection swap: route the attn q/k/v/o + GDN in/out_proj
            // matvecs (all through `qwen35_gemm`) onto the single-stream cols
            // kernel for the duration of the mixer. Reset before the MoE/MLP
            // block so MoE keeps its grouped MUL_MAT_ID GEMM untouched.
            self.spec_verify_cols = true;
            let attn_out = match cfg.layer_types[layer_idx] {
                qwen35::LayerType::FullAttention =>
                    self.qwen35_full_attn_prefill_gpu(&cfg, layer_idx, &x, t, start_pos),
                qwen35::LayerType::LinearAttention => {
                    // §3 capture: stash this GDN layer's batched input for the
                    // option-B rollback BEFORE the scan advances resident state.
                    self.spec_verify_gdn_inputs.push((layer_idx, x.clone()));
                    match self.qwen35_linear_prefill_gpu(&cfg, layer_idx, &x, t) {
                        Some(out) => out,
                        None => {
                            let mut out = vec![0.0f32; t * h];
                            for ti in 0..t {
                                let xt = &x[ti * h..(ti + 1) * h];
                                let o = self.qwen35_delta_net_gpu(&cfg, layer_idx, xt);
                                out[ti * h..(ti + 1) * h].copy_from_slice(&o);
                            }
                            out
                        }
                    }
                }
            };
            self.spec_verify_cols = false;
            let h1: Vec<f32> = residual.iter().zip(&attn_out).map(|(&r, &a)| r + a).collect();
            let residual2 = h1.clone();
            let post_ln = self.qwen35_w(&ln("post_attention_layernorm.weight"));
            let ff_in = model::cpu_rms_norm(&h1, &post_ln, eps);
            let mlp_out = if cfg.is_moe() {
                self.qwen35_moe_mlp_prefill_gpu(&cfg, layer_idx, &ff_in, t)
            } else {
                self.qwen35_dense_mlp_prefill_gpu(&cfg, layer_idx, &ff_in, t)
            };
            hidden = residual2.iter().zip(&mlp_out).map(|(&r, &m)| r + m).collect();
        }

        if !pp_last {
            return Ok(hidden);
        }
        // Stash the pre-`model.norm` residual for the MTP draft head (§5 refill).
        self.stash_verify_prenorm(&hidden, start_pos, t, h);
        // Last stage: lm_head over ALL T positions (T independent matvecs, one
        // per row — the exact per-column pattern the dense batched path uses;
        // `qwen35_matvec` resolves the chunked/f16-host lm_head sources the
        // `qwen35_gemm` path can't, same as the prefill last-token branch).
        let norm_w = self.qwen35_w("model.norm.weight");
        let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
        let normed_all: Vec<f32> = (0..t)
            .flat_map(|ti| model::cpu_rms_norm(&hidden[ti * h..(ti + 1) * h], &norm_w, eps))
            .collect();
        // Single-stream lm_head (perf review P1a): the T verify positions were
        // T serial `qwen35_matvec` calls, each re-streaming the whole ~330MB
        // q8_0/f16 lm_head weight — T passes over the same bytes. Route through
        // `qwen35_matvec_cols` (the same single-stream-weight kernel the
        // attn/GDN mixers use under `spec_verify_cols`) so the weight is
        // streamed ONCE for all T rows. GUARDED off the chunked-lm_head path
        // (`chunked_weights`, the >~2.5GB row-split alloc): that source isn't a
        // plain `gpu_weights` buffer the cols kernel can read, so it falls
        // through to the existing per-row `qwen35_matvec` loop below, exactly
        // as before this change. Pure dispatch swap — same weight bytes, same
        // per-row dequant math, so output is bit-identical.
        let logits = if !self.chunked_weights.contains_key(&lm_name) {
            self.qwen35_matvec_cols(&lm_name, &normed_all, t, h, vocab)
        } else {
            None
        };
        let logits = match logits {
            Some(l) => l,
            None => {
                let mut logits = vec![0.0f32; t * vocab];
                for ti in 0..t {
                    let normed = &normed_all[ti * h..(ti + 1) * h];
                    let l = self.qwen35_matvec(&lm_name, normed, h, vocab);
                    logits[ti * vocab..(ti + 1) * vocab].copy_from_slice(&l);
                }
                logits
            }
        };
        Ok(logits)
    }

    /// Record the pre-`model.norm` residual for EACH of the T verify positions
    /// into the position-keyed `q35_prenorm_ring` (+ the freshest into
    /// `q35_last_prenorm`), so the MTP draft refill sources the EXACT producing
    /// hidden via `prenorm_for_pos(new_pos-1)` — the P4 α fix the single-token
    /// `forward_pp_qwen35_argmax` does per pass, which the batched verify must do
    /// for all T positions at once. Row `ti` is the pre-norm residual at position
    /// `start_pos+ti` (the pass FED the token at `start_pos+ti`, producing the
    /// token at `start_pos+ti+1`). Without this every refill falls back to the
    /// STALE prefill `q35_last_prenorm` and acceptance collapses to ~0. Gated on
    /// a loaded head (only the last PP stage owns it); a no-op otherwise, so the
    /// Mac/engine-less identity path is unchanged.
    pub(crate) fn stash_verify_prenorm(&mut self, hidden: &[f32], start_pos: usize, t: usize, h: usize) {
        if self.mtp_head.is_none() {
            return;
        }
        for ti in 0..t {
            let pn = hidden[ti * h..(ti + 1) * h].to_vec();
            self.q35_prenorm_ring.push_back((start_pos + ti, pn));
        }
        while self.q35_prenorm_ring.len() > Self::PRENORM_RING_CAP {
            self.q35_prenorm_ring.pop_front();
        }
        self.q35_last_prenorm = Some(hidden[(t - 1) * h..t * h].to_vec());
    }

    /// Design-A partial-accept ROLLBACK (MTP re-gate, spec §3, option-B). After
    /// a batched verify of `[s_R, d_1..d_D]` and `resolve_chain` yielding
    /// `accept_len=k`, roll the resident recurrent state back to exactly the
    /// `k+1` committed tokens:
    ///   - full accept (`k=D`): the verify already left state at `R+D+1` — no-op;
    ///   - full-attn: `spec_restore` rewinds the KV counter to `R`, then
    ///     `set_full_attn_seq_len(R+k+1)` re-exposes the committed prefix's
    ///     already-written K/V (overwrite-in-place ⇒ no recompute);
    ///   - GDN: `spec_restore` restores the recurrent state to `R`, then re-scan
    ///     the first `k+1` captured inputs through the GDN layers only (same
    ///     kernel path the verify used ⇒ bit-exact to committing `k+1` tokens).
    /// `slot` must hold a `spec_snapshot` taken at `R` (pre-verify). Requires a
    /// `forward_qwen35_verify_*` call since the last rollback.
    pub(crate) fn qwen35_verify_rollback_impl(&mut self, slot: usize, accept_len: usize) -> Result<(), String> {
        let (r, t_span) = self.spec_verify_span
            .ok_or("qwen35_verify_rollback: no verify pending (call forward_qwen35_verify first)")?;
        let commit_len = accept_len + 1;
        if commit_len > t_span {
            return Err(format!("qwen35_verify_rollback: accept_len {accept_len} exceeds verify T {t_span}"));
        }
        if commit_len == t_span {
            // Full accept: verify already advanced state to R+T = R+commit_len.
            self.spec_verify_gdn_inputs.clear();
            self.spec_verify_span = None;
            return Ok(());
        }
        // 1) Restore GDN (device+host) and rewind full-attn KV counter to R.
        self.spec_restore_impl(slot)?;
        // 2) Re-expose the committed full-attn K/V (bytes at R..R+commit_len are
        //    still the verify's committed tokens; counter-only advance).
        let cfg = self.qwen35.as_ref().ok_or("verify_rollback: no qwen3_5 model")?.config.clone();
        let h = cfg.hidden_size;
        if let Some(m) = self.qwen35.as_mut() { m.set_full_attn_seq_len(r + commit_len); }
        // 3) GDN-only re-scan of the committed prefix (option-B). Uses the SAME
        //    projection dispatch the verify forward used (§6 cols kernel on the
        //    GPU path), so `cols(x[0..k+1])` == the first k+1 rows of the verify's
        //    `cols(x[0..T])` and the re-scanned recurrent state is BIT-identical
        //    to the committed frontier — no cols-vs-GEMM drift across cycles.
        let inputs = std::mem::take(&mut self.spec_verify_gdn_inputs);
        self.spec_verify_cols = true;
        for (layer_idx, x) in &inputs {
            let x_prefix = &x[..commit_len * h];
            if !self.dn_gpu.contains_key(layer_idx) {
                // Host-authoritative GDN (CPU path or non-uploaded layer).
                self.qwen35.as_mut().unwrap().delta_net_scan(*layer_idx, x_prefix, commit_len);
            } else if self.qwen35_linear_prefill_gpu(&cfg, *layer_idx, x_prefix, commit_len).is_none() {
                for ti in 0..commit_len {
                    let xt = &x_prefix[ti * h..(ti + 1) * h];
                    self.qwen35_delta_net_gpu(&cfg, *layer_idx, xt);
                }
            }
        }
        self.spec_verify_cols = false;
        self.spec_verify_span = None;
        Ok(())
    }

    /// Upload one MoE layer's 4-bit experts (all `num_experts`) to resident GPU
    /// buffers, mirroring `moe::QuantSwitch`'s packed layout exactly. Idempotent:
    /// no-op if already resident. Returns false if no engine / weights missing.
    pub(crate) fn ensure_moe_gpu_layer(&mut self, layer_idx: usize) -> bool {
        if self.moe_gpu.contains_key(&layer_idx) {
            return true;
        }
        // Pull the three QuantSwitch tensors out (clone the small metadata; the
        // big Vecs are referenced for upload then dropped — engine owns the GPU copy).
        let build = |eng: &mut compute::ComputeEngine, qs: &moe::QuantSwitch| -> Option<MoeGpuProj> {
            let per_word = 32 / qs.bits;
            let words_per_row = qs.in_features / per_word;
            let groups = qs.in_features / qs.group_size;
            let pack_stride = qs.out_features * words_per_row;
            let sb_stride = qs.out_features * groups;
            assert!(pack_stride > 0 && qs.packed.len() % pack_stride == 0,
                "MoE packed len {} not a multiple of per-expert stride {}",
                qs.packed.len(), pack_stride);
            assert!(sb_stride > 0 && qs.scales.len() % sb_stride == 0,
                "MoE scales len {} not a multiple of per-expert stride {}",
                qs.scales.len(), sb_stride);
            // packed is Vec<u32>; reinterpret as bytes for upload.
            let packed_bytes = bytemuck::cast_slice::<u32, u8>(&qs.packed).to_vec();
            // VLLM_VULKAN_MOE_F16_SCALES: store the affine scales+biases f16 in the
            // resident GTT buffer (half the f32 footprint — the 122B PP-6 fit-
            // enabler). f16 holds every normal-range bf16-sourced scale EXACTLY;
            // the measured 122B switch tensors also carry a tiny subnormal tail
            // (~3e-7) that rounds with negligible error (argmax-exact). The safe
            // converter rejects ONLY genuine overflow (|x|>65504→Inf) / non-finite
            // — that layer then stays host/CPU (ensure returns false, resident path
            // falls back, never corrupts). Measured max|x|≈0.6 → no rejections.
            let (scales_bytes, biases_bytes) = if crate::moe_f16_scales_flag() {
                match (
                    crate::f32_scales_to_f16_bytes_safe(&qs.scales),
                    crate::f32_scales_to_f16_bytes_safe(&qs.biases),
                ) {
                    (Some(s), Some(b)) => (s, b),
                    _ => {
                        eprintln!(
                            "[moe_f16_scales] f16-OVERFLOW (|x|>65504) or non-finite affine scale/bias in MoE expert tensor (out={}, in={}); refusing f16 — layer stays host/CPU. Unset VLLM_VULKAN_MOE_F16_SCALES for f32-resident.",
                            qs.out_features, qs.in_features
                        );
                        return None;
                    }
                }
            } else {
                (f32_slice_to_bytes(&qs.scales), f32_slice_to_bytes(&qs.biases))
            };
            let pbuf = eng.alloc_host_coherent_storage(packed_bytes.len() as u64).ok()?;
            pbuf.write(&packed_bytes).ok()?;
            let sbuf = eng.alloc_host_coherent_storage(scales_bytes.len() as u64).ok()?;
            sbuf.write(&scales_bytes).ok()?;
            let bbuf = eng.alloc_host_coherent_storage(biases_bytes.len() as u64).ok()?;
            bbuf.write(&biases_bytes).ok()?;
            Some(MoeGpuProj {
                packed: pbuf, scales: sbuf, biases: bbuf,
                out_features: qs.out_features, in_features: qs.in_features,
                group_size: qs.group_size, pack_stride, sb_stride,
            })
        };

        // Disjoint field borrows: &mut engine alongside &qwen35.quant_moe +
        // &mut moe_gpu, so the three per-expert tensors are referenced for
        // upload instead of .cloned() (avoids a ~0.4 GB/layer transient copy).
        let Self { engine, qwen35, moe_gpu, .. } = self;
        let eng = match engine.as_mut() { Some(e) => e, None => return false };
        let m = match qwen35.as_ref() { Some(m) => m, None => return false };
        let (qg, qu, qd) = match (
            m.quant_moe.gate.get(&layer_idx),
            m.quant_moe.up.get(&layer_idx),
            m.quant_moe.down.get(&layer_idx),
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => return false,
        };
        let gate = match build(eng, qg) { Some(p) => p, None => return false };
        let up = match build(eng, qu) { Some(p) => p, None => return false };
        let down = match build(eng, qd) { Some(p) => p, None => return false };
        moe_gpu.insert(layer_idx, MoeGpuLayer { gate, up, down });

        // All three projections are now fully mirrored in GTT-resident GPU
        // buffers. On a unified-memory box the host copy is pure waste from
        // here on (contains_key above makes this layer's upload idempotent —
        // it can never be re-read), so free it under VLLM_VULKAN_MOE_HOST_FREE
        // (default ON). NOTE: the CPU quant-MoE fallback in `qwen35_moe_mlp_gpu`
        // (used when `qwen35_moe_mlp_gpu_resident` returns None on a LATER call,
        // e.g. a transient GPU submit failure after this layer already went
        // resident) reads these same Vecs — with the flag on, that fallback is
        // unsupported for an already-freed layer and will panic on an
        // out-of-bounds/empty-slice index instead of silently degrading to CPU.
        // This is an accepted tradeoff: avoiding the deterministic PP-2 OOM
        // matters more than a fallback path that only exists for GPU errors
        // that would likely recur on retry anyway.
        if crate::moe_host_free_enabled() {
            if let Some(m) = qwen35.as_mut() {
                if let Some(qs) = m.quant_moe.gate.get_mut(&layer_idx) { qs.free_host_data(); }
                if let Some(qs) = m.quant_moe.up.get_mut(&layer_idx) { qs.free_host_data(); }
                if let Some(qs) = m.quant_moe.down.get_mut(&layer_idx) { qs.free_host_data(); }
            }
        }
        true
    }
    /// Upload one MoE layer's CPU-glue weights — shared expert gate/up/down,
    /// the 1-row shared_expert_gate and the router gate — to resident f32 GPU
    /// buffers (WS2). Idempotent; the host f32 copies stay in `ModelWeights`
    /// (they are small and keep the `MOE_GPU=0` CPU fallback intact). Returns
    /// false if there is no engine or any tensor is missing.
    pub(crate) fn ensure_moe_shared_gpu_layer(&mut self, layer_idx: usize) -> bool {
        if self.moe_shared_gpu.contains_key(&layer_idx) {
            return true;
        }
        let Self { engine, qwen35, moe_shared_gpu, .. } = self;
        let eng = match engine.as_mut() { Some(e) => e, None => return false };
        let m = match qwen35.as_ref() { Some(m) => m, None => return false };
        let p = format!("model.layers.{layer_idx}.mlp");
        let upload = |eng: &mut compute::ComputeEngine, name: String| -> Option<compute::Buffer> {
            let t = m.weights.tensors.get(&name)?;
            let bytes = f32_slice_to_bytes(&t.data);
            let buf = eng.alloc_host_coherent_storage(bytes.len() as u64).ok()?;
            buf.write(&bytes).ok()?;
            Some(buf)
        };
        let gate = match upload(eng, format!("{p}.shared_expert.gate_proj.weight")) {
            Some(b) => b, None => return false,
        };
        let up = match upload(eng, format!("{p}.shared_expert.up_proj.weight")) {
            Some(b) => b, None => return false,
        };
        let down = match upload(eng, format!("{p}.shared_expert.down_proj.weight")) {
            Some(b) => b, None => return false,
        };
        let expert_gate = match upload(eng, format!("{p}.shared_expert_gate.weight")) {
            Some(b) => b, None => return false,
        };
        let router = match upload(eng, format!("{p}.gate.weight")) {
            Some(b) => b, None => return false,
        };
        moe_shared_gpu.insert(layer_idx, MoeSharedGpu { gate, up, down, expert_gate, router });
        true
    }
    /// GPU-resident MoE forward for one token (WS2). The router gate matvec,
    /// the 8 routed experts' SwiGLU (4-bit mlx4, dequant in-shader), the
    /// silu(gate)·up glue AND the dense f32 shared expert ALL run on the GPU;
    /// the host only does the top-8 selection over 256 logits, the
    /// score-weighted accumulate and the final sigmoid-gated combine.
    /// Two submits per layer:
    ///   1. router gate matvec h→E (must complete first — the top-8 indices
    ///      select which packed-expert offsets get recorded), then
    ///   2. ONE command buffer: routed gate/up ‖ shared gate/up ‖ shared
    ///      sigmoid-gate logit → barrier → silu → barrier → mul → barrier →
    ///      routed downs ‖ shared down.
    /// The shared expert only reads `ff_in`, so recording it into the same CB
    /// lets the GPU scheduler overlap it with the routed experts for free —
    /// this RETIRED the VLLM_VULKAN_MOE_OVERLAP std::thread::scope machinery.
    /// Math is cos≈1.0-comparable to `moe_forward_token_quant` (GPU silu/exp
    /// and reduction order differ from libm in the last ulp). Returns None to
    /// fall back to the CPU path.
    pub(crate) fn qwen35_moe_mlp_gpu_resident(
        &mut self,
        layer_idx: usize,
        ff_in: &[f32],
        d: moe::MoeDims,
    ) -> Option<Vec<f32>> {
        if !self.ensure_moe_gpu_layer(layer_idx) {
            return None;
        }
        if !self.ensure_moe_shared_gpu_layer(layer_idx) {
            return None;
        }
        let h = d.hidden;
        let mi = d.moe_inter;
        let si = d.shared_inter;

        let Self { engine, moe_gpu, moe_shared_gpu, .. } = self;
        let eng = engine.as_mut()?;
        let layer = &moe_gpu[&layer_idx];
        let shared = &moe_shared_gpu[&layer_idx];
        let group_size = layer.gate.group_size;

        // Token input — read by the router, all routed experts AND the shared
        // expert (uploaded once, shared across both submits).
        let t_route = std::time::Instant::now();
        let xb = f32_slice_to_bytes(ff_in);
        let inp = eng.alloc_host_coherent_storage((ff_in.len() * 4) as u64).ok()?;
        inp.write(&xb).ok()?;
        let inp_p = &inp as *const compute::Buffer;

        // ── Submit 1: router gate matvec (h→E) on GPU; top-8 on host. The
        //    readback is E=256 floats (tiny); the host tail is `route`'s
        //    softmax → top-k → renorm, byte-identical logic.
        let logits_buf = eng.alloc_host_coherent_storage((d.num_experts * 4) as u64).ok()?;
        {
            let (rshader, rr) = matvec_f32_variant(d.num_experts);
            let wg = (d.num_experts as u32 + rr - 1) / rr;
            let pc = matvec_pc13(h, d.num_experts);
            let lb = &logits_buf as *const compute::Buffer;
            let cb = eng.begin_batch().ok()?;
            unsafe {
                eng.record_to(cb, &rshader, &[&shared.router, &*inp_p, &*lb], &pc, (wg, 1, 1)).ok()?;
            }
            eng.submit_batch(cb).ok()?;
        }
        let logits = read_f32_buf(&logits_buf, d.num_experts);
        eng.return_to_pool(logits_buf);
        let routing = moe::route_from_logits(&logits, d.top_k);
        prof_add("moe_route_gpu", t_route);

        // ── Submit 2: the WHOLE MLP in one command buffer. ─────────────────
        let t_fused = std::time::Instant::now();
        let n_exp = routing.indices.len();
        // Pooled per-token buffers. Routed: gate+up (2·mi per expert), act
        // (mi), mid (mi), down (h). Shared: gate/up/act/mid (si), out (h),
        // sigmoid-gate logit (1).
        let mut gu_outs: Vec<compute::Buffer> = Vec::with_capacity(n_exp * 2);
        for _ in 0..(n_exp * 2) {
            gu_outs.push(eng.alloc_host_coherent_storage((mi * 4) as u64).ok()?);
        }
        let mut acts: Vec<compute::Buffer> = Vec::with_capacity(n_exp);
        let mut mids: Vec<compute::Buffer> = Vec::with_capacity(n_exp);
        let mut down_outs: Vec<compute::Buffer> = Vec::with_capacity(n_exp);
        for _ in 0..n_exp {
            acts.push(eng.alloc_host_coherent_storage((mi * 4) as u64).ok()?);
            mids.push(eng.alloc_host_coherent_storage((mi * 4) as u64).ok()?);
            down_outs.push(eng.alloc_host_coherent_storage((h * 4) as u64).ok()?);
        }
        let s_gate = eng.alloc_host_coherent_storage((si * 4) as u64).ok()?;
        let s_up = eng.alloc_host_coherent_storage((si * 4) as u64).ok()?;
        let s_act = eng.alloc_host_coherent_storage((si * 4) as u64).ok()?;
        let s_mid = eng.alloc_host_coherent_storage((si * 4) as u64).ok()?;
        let s_out = eng.alloc_host_coherent_storage((h * 4) as u64).ok()?;
        let s_logit = eng.alloc_host_coherent_storage(4).ok()?;
        let (sg_p, su_p, sa_p, sm_p, so_p, sl_p) = (
            &s_gate as *const compute::Buffer, &s_up as *const compute::Buffer,
            &s_act as *const compute::Buffer, &s_mid as *const compute::Buffer,
            &s_out as *const compute::Buffer, &s_logit as *const compute::Buffer,
        );

        let (gu_shader, gu_r) = crate::matvec_mlx4_moe_variant_k(h, mi);
        let wg_mi = (mi as u32 + gu_r - 1) / gu_r;
        let (down_shader, down_r) = crate::matvec_mlx4_moe_variant_k(mi, h);
        let wg_h = (h as u32 + down_r - 1) / down_r;
        let (sgu_shader, sgu_r) = matvec_f32_variant_k(h, si);
        let swg_si = (si as u32 + sgu_r - 1) / sgu_r;
        let (sd_shader, sd_r) = matvec_f32_variant_k(si, h);
        let swg_h = (h as u32 + sd_r - 1) / sd_r;
        let (sl_shader, _) = matvec_f32_variant(1);
        let pc_sgu = matvec_pc13(h, si);
        let pc_sd = matvec_pc13(si, h);
        let pc_sl = matvec_pc13(h, 1);
        let silu_mi = ew_unary_pc(mi as u32);
        let mul_mi = ew_mul_pc(mi as u32);
        let silu_si = ew_unary_pc(si as u32);
        let mul_si = ew_mul_pc(si as u32);

        let cb = eng.begin_batch().ok()?;
        unsafe {
            // Phase A: every projection that reads INP (all independent) —
            // routed gate/up per expert, shared gate/up, shared gate logit.
            for (slot, &e) in routing.indices.iter().enumerate() {
                let pc_g = matvec_mlx4_pc_off(
                    h, mi, group_size, e * layer.gate.pack_stride, e * layer.gate.sb_stride);
                let pc_u = matvec_mlx4_pc_off(
                    h, mi, group_size, e * layer.up.pack_stride, e * layer.up.sb_stride);
                let go = &gu_outs[slot * 2] as *const compute::Buffer;
                let uo = &gu_outs[slot * 2 + 1] as *const compute::Buffer;
                eng.record_to(cb, &gu_shader,
                    &[&layer.gate.packed, &layer.gate.scales, &layer.gate.biases, &*inp_p, &*go],
                    &pc_g, (wg_mi, 1, 1)).ok()?;
                eng.record_to(cb, &gu_shader,
                    &[&layer.up.packed, &layer.up.scales, &layer.up.biases, &*inp_p, &*uo],
                    &pc_u, (wg_mi, 1, 1)).ok()?;
            }
            eng.record_to(cb, &sgu_shader, &[&shared.gate, &*inp_p, &*sg_p], &pc_sgu, (swg_si, 1, 1)).ok()?;
            eng.record_to(cb, &sgu_shader, &[&shared.up, &*inp_p, &*su_p], &pc_sgu, (swg_si, 1, 1)).ok()?;
            eng.record_to(cb, &sl_shader, &[&shared.expert_gate, &*inp_p, &*sl_p], &pc_sl, (1, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            // silu(gate) — routed + shared (same shader/geometry as the dense
            // resident path at forward_qwen_gpu_resident).
            for slot in 0..n_exp {
                let go = &gu_outs[slot * 2] as *const compute::Buffer;
                let ao = &acts[slot] as *const compute::Buffer;
                eng.record_to(cb, "silu_f32", &[&*go, &*ao], &silu_mi, ((mi as u32 + 511) / 512, 1, 1)).ok()?;
            }
            eng.record_to(cb, "silu_f32", &[&*sg_p, &*sa_p], &silu_si, ((si as u32 + 511) / 512, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            // mid = silu(gate) * up — routed + shared.
            for slot in 0..n_exp {
                let ao = &acts[slot] as *const compute::Buffer;
                let uo = &gu_outs[slot * 2 + 1] as *const compute::Buffer;
                let mo = &mids[slot] as *const compute::Buffer;
                eng.record_to(cb, "mul_f32_f32_f32", &[&*ao, &*uo, &*mo], &mul_mi, ((mi as u32 + 255) / 256, 1, 1)).ok()?;
            }
            eng.record_to(cb, "mul_f32_f32_f32", &[&*sa_p, &*su_p, &*sm_p], &mul_si, ((si as u32 + 255) / 256, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            // down projections — routed (per-expert offsets) + shared.
            for (slot, &e) in routing.indices.iter().enumerate() {
                let pc_d = matvec_mlx4_pc_off(
                    mi, h, group_size, e * layer.down.pack_stride, e * layer.down.sb_stride);
                let mo = &mids[slot] as *const compute::Buffer;
                let dobuf = &down_outs[slot] as *const compute::Buffer;
                eng.record_to(cb, &down_shader,
                    &[&layer.down.packed, &layer.down.scales, &layer.down.biases, &*mo, &*dobuf],
                    &pc_d, (wg_h, 1, 1)).ok()?;
            }
            eng.record_to(cb, &sd_shader, &[&shared.down, &*sm_p, &*so_p], &pc_sd, (swg_h, 1, 1)).ok()?;
        }
        eng.submit_batch(cb).ok()?;
        prof_add("moe_fused_submit", t_fused);

        // Host: score-weighted routed accumulate + sigmoid-gated shared add.
        let t_acc = std::time::Instant::now();
        let mut out = vec![0.0f32; h];
        for slot in 0..n_exp {
            let eo = read_f32_buf(&down_outs[slot], h);
            let sc = routing.scores[slot];
            for (r, &o) in out.iter_mut().zip(&eo) { *r += sc * o; }
        }
        let shared_out = read_f32_buf(&s_out, h);
        let sg_logit = read_f32_buf(&s_logit, 1)[0];
        let sg = 1.0f32 / (1.0 + (-sg_logit).exp());
        for (r, &sh) in out.iter_mut().zip(&shared_out) { *r += sg * sh; }

        eng.return_to_pool(inp);
        for b in gu_outs { eng.return_to_pool(b); }
        for b in acts { eng.return_to_pool(b); }
        for b in mids { eng.return_to_pool(b); }
        for b in down_outs { eng.return_to_pool(b); }
        for b in [s_gate, s_up, s_act, s_mid, s_out, s_logit] { eng.return_to_pool(b); }
        prof_add("moe_readback", t_acc);

        Some(out)
    }

    /// MTP draft head MoE, GPU-resident (plan §P2 D-gate). ONE fused command
    /// buffer runs the top-k routed experts' f16 SwiGLU + the dense f32 shared
    /// expert; the host does only the h→256 route (off `mtp_moe_gpu.router`),
    /// the top-k score-weighted accumulate and the sigmoid-gated shared add.
    /// Structurally identical to `qwen35_moe_mlp_gpu_resident`, with two
    /// deliberate simplifications for the single-layer head:
    ///   * routed experts are f16 (not 4-bit): the whole 256-expert set is only
    ///     ~1.6 GB resident, so a per-expert `record_to_off` byte offset into
    ///     one big f16 buffer + the proven `mul_mat_vec_f16` shader beats writing
    ///     an mlx4 encoder, and keeps the draft numerically closest to the f32
    ///     CPU parity path;
    ///   * the router runs on the CPU (negligible at one layer), so there is one
    ///     submit, not two — no router-readback round trip to pick the offsets.
    /// Math is cos≈1.0-comparable to the head's CPU `moe_forward_token_rayon`
    /// (GPU silu/exp + f16 weight rounding differ in the last ulp). Returns None
    /// (→ CPU fallback) if the head MoE was not uploaded or any alloc/submit fails.
    #[cfg(feature = "mtp")]
    pub(crate) fn mtp_moe_mlp_gpu(&mut self, ff_in: &[f32], d: moe::MoeDims) -> Option<Vec<f32>> {
        let h = d.hidden;
        let mi = d.moe_inter;
        let si = d.shared_inter;

        // Router on the CPU (h→E, ~0.5M MACs), off the resident host copy.
        let t_route = std::time::Instant::now();
        let routing = {
            let m = self.mtp_moe_gpu.as_ref()?;
            moe::route_par(ff_in, &m.router, h, d.num_experts, d.top_k)
        };
        prof_add("mtp_moe_route", t_route);

        let Self { engine, mtp_moe_gpu, .. } = self;
        let eng = engine.as_mut()?;
        let m = mtp_moe_gpu.as_ref()?;
        let n_exp = routing.indices.len();

        // Token input — read by every routed expert AND the shared expert.
        let t_fused = std::time::Instant::now();
        let xb = f32_slice_to_bytes(ff_in);
        let inp = eng.alloc_host_coherent_storage((ff_in.len() * 4) as u64).ok()?;
        inp.write(&xb).ok()?;
        let inp_p = &inp as *const compute::Buffer;

        // Pooled per-token scratch (mirrors the resident path): routed gate+up
        // (2·mi/expert), act (mi), mid (mi), down (h); shared gate/up/act/mid
        // (si), out (h), sigmoid-gate logit (1).
        let mut gu_outs: Vec<compute::Buffer> = Vec::with_capacity(n_exp * 2);
        for _ in 0..(n_exp * 2) {
            gu_outs.push(eng.alloc_host_coherent_storage((mi * 4) as u64).ok()?);
        }
        let mut acts: Vec<compute::Buffer> = Vec::with_capacity(n_exp);
        let mut mids: Vec<compute::Buffer> = Vec::with_capacity(n_exp);
        let mut down_outs: Vec<compute::Buffer> = Vec::with_capacity(n_exp);
        for _ in 0..n_exp {
            acts.push(eng.alloc_host_coherent_storage((mi * 4) as u64).ok()?);
            mids.push(eng.alloc_host_coherent_storage((mi * 4) as u64).ok()?);
            down_outs.push(eng.alloc_host_coherent_storage((h * 4) as u64).ok()?);
        }
        let s_gate = eng.alloc_host_coherent_storage((si * 4) as u64).ok()?;
        let s_up = eng.alloc_host_coherent_storage((si * 4) as u64).ok()?;
        let s_act = eng.alloc_host_coherent_storage((si * 4) as u64).ok()?;
        let s_mid = eng.alloc_host_coherent_storage((si * 4) as u64).ok()?;
        let s_out = eng.alloc_host_coherent_storage((h * 4) as u64).ok()?;
        let s_logit = eng.alloc_host_coherent_storage(4).ok()?;
        let (sg_p, su_p, sa_p, sm_p, so_p, sl_p) = (
            &s_gate as *const compute::Buffer, &s_up as *const compute::Buffer,
            &s_act as *const compute::Buffer, &s_mid as *const compute::Buffer,
            &s_out as *const compute::Buffer, &s_logit as *const compute::Buffer,
        );

        // Shaders: f16 routed experts (gate/up k=h→mi, down k=mi→h), f32 shared.
        // The routed experts (m.gate/m.up/m.down) are ALWAYS uploaded F16 by the
        // MTP loader, so they MUST use the F16 shader explicitly — NOT the
        // global-quant-keyed `matvec_variant(true, ..)`, which under
        // VLLM_VULKAN_QUANT=q8_0 picks the q8_0 dequant shader and reads the f16
        // expert bytes as q8_0 blocks → a ~20000× output blow-up (garbage head
        // hidden → acc_rate=0). Same class as the dense-projection fix; this path
        // dispatches matvec directly instead of through qwen35_matvec, so it
        // needs the format-explicit selector of its own.
        let (gu_shader, gu_r) = crate::push_constants::matvec_variant_by_format(crate::flags::QuantFormat::F16, mi);
        let wg_mi = (mi as u32 + gu_r - 1) / gu_r;
        let (down_shader, down_r) = crate::push_constants::matvec_variant_by_format(crate::flags::QuantFormat::F16, h);
        let wg_h = (h as u32 + down_r - 1) / down_r;
        let (sgu_shader, sgu_r) = matvec_f32_variant_k(h, si);
        let swg_si = (si as u32 + sgu_r - 1) / sgu_r;
        let (sd_shader, sd_r) = matvec_f32_variant_k(si, h);
        let swg_h = (h as u32 + sd_r - 1) / sd_r;
        let (sl_shader, _) = matvec_f32_variant(1);
        let pc_gu = matvec_pc13(h, mi);
        let pc_d = matvec_pc13(mi, h);
        let pc_sgu = matvec_pc13(h, si);
        let pc_sd = matvec_pc13(si, h);
        let pc_sl = matvec_pc13(h, 1);
        let silu_mi = ew_unary_pc(mi as u32);
        let mul_mi = ew_mul_pc(mi as u32);
        let silu_si = ew_unary_pc(si as u32);
        let mul_si = ew_mul_pc(si as u32);
        // Per-expert f16 weight byte offsets into the [E*out, in] buffers.
        let gu_expert_bytes = (mi * h * 2) as u64; // one expert of gate/up [mi,h]
        let dn_expert_bytes = (h * mi * 2) as u64; // one expert of down    [h,mi]

        let cb = eng.begin_batch().ok()?;
        unsafe {
            // Phase A: every projection reading INP (all independent) — routed
            // gate/up per expert (f16, per-expert offset), shared gate/up, and
            // the shared sigmoid-gate logit.
            for (slot, &e) in routing.indices.iter().enumerate() {
                let goff = e as u64 * gu_expert_bytes;
                let go = &gu_outs[slot * 2] as *const compute::Buffer;
                let uo = &gu_outs[slot * 2 + 1] as *const compute::Buffer;
                eng.record_to_off(cb, &gu_shader,
                    &[(&m.gate, goff), (&*inp_p, 0), (&*go, 0)], &pc_gu, (wg_mi, 1, 1)).ok()?;
                eng.record_to_off(cb, &gu_shader,
                    &[(&m.up, goff), (&*inp_p, 0), (&*uo, 0)], &pc_gu, (wg_mi, 1, 1)).ok()?;
            }
            eng.record_to(cb, &sgu_shader, &[&m.s_gate, &*inp_p, &*sg_p], &pc_sgu, (swg_si, 1, 1)).ok()?;
            eng.record_to(cb, &sgu_shader, &[&m.s_up, &*inp_p, &*su_p], &pc_sgu, (swg_si, 1, 1)).ok()?;
            eng.record_to(cb, &sl_shader, &[&m.s_expert_gate, &*inp_p, &*sl_p], &pc_sl, (1, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            // silu(gate) — routed + shared.
            for slot in 0..n_exp {
                let go = &gu_outs[slot * 2] as *const compute::Buffer;
                let ao = &acts[slot] as *const compute::Buffer;
                eng.record_to(cb, "silu_f32", &[&*go, &*ao], &silu_mi, ((mi as u32 + 511) / 512, 1, 1)).ok()?;
            }
            eng.record_to(cb, "silu_f32", &[&*sg_p, &*sa_p], &silu_si, ((si as u32 + 511) / 512, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            // mid = silu(gate) * up — routed + shared.
            for slot in 0..n_exp {
                let ao = &acts[slot] as *const compute::Buffer;
                let uo = &gu_outs[slot * 2 + 1] as *const compute::Buffer;
                let mo = &mids[slot] as *const compute::Buffer;
                eng.record_to(cb, "mul_f32_f32_f32", &[&*ao, &*uo, &*mo], &mul_mi, ((mi as u32 + 255) / 256, 1, 1)).ok()?;
            }
            eng.record_to(cb, "mul_f32_f32_f32", &[&*sa_p, &*su_p, &*sm_p], &mul_si, ((si as u32 + 255) / 256, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            // down projections — routed (f16, per-expert offset) + shared (f32).
            for (slot, &e) in routing.indices.iter().enumerate() {
                let doff = e as u64 * dn_expert_bytes;
                let mo = &mids[slot] as *const compute::Buffer;
                let dobuf = &down_outs[slot] as *const compute::Buffer;
                eng.record_to_off(cb, &down_shader,
                    &[(&m.down, doff), (&*mo, 0), (&*dobuf, 0)], &pc_d, (wg_h, 1, 1)).ok()?;
            }
            eng.record_to(cb, &sd_shader, &[&m.s_down, &*sm_p, &*so_p], &pc_sd, (swg_h, 1, 1)).ok()?;
        }
        eng.submit_batch(cb).ok()?;
        prof_add("mtp_moe_fused_submit", t_fused);

        // Host: score-weighted routed accumulate + sigmoid-gated shared add
        // (byte-identical reduction shape to `moe_forward_token`).
        let t_acc = std::time::Instant::now();
        let mut out = vec![0.0f32; h];
        for slot in 0..n_exp {
            let eo = read_f32_buf(&down_outs[slot], h);
            let sc = routing.scores[slot];
            for (r, &o) in out.iter_mut().zip(&eo) { *r += sc * o; }
        }
        let shared_out = read_f32_buf(&s_out, h);
        let sg_logit = read_f32_buf(&s_logit, 1)[0];
        let sg = 1.0f32 / (1.0 + (-sg_logit).exp());
        for (r, &sh) in out.iter_mut().zip(&shared_out) { *r += sg * sh; }

        eng.return_to_pool(inp);
        for b in gu_outs { eng.return_to_pool(b); }
        for b in acts { eng.return_to_pool(b); }
        for b in mids { eng.return_to_pool(b); }
        for b in down_outs { eng.return_to_pool(b); }
        for b in [s_gate, s_up, s_act, s_mid, s_out, s_logit] { eng.return_to_pool(b); }
        prof_add("mtp_moe_readback", t_acc);

        Some(out)
    }
    // ── WS3: qwen3.6 resident stage (VLLM_VULKAN_Q35_1CB) ───────────────────
    pub(crate) fn q35r_ptr(&self, slot: usize) -> *const compute::Buffer {
        &self.q35res_bufs[slot] as *const compute::Buffer
    }
    pub(crate) fn q35r_ptr_mut(&mut self, slot: usize) -> *mut compute::Buffer {
        &mut self.q35res_bufs[slot] as *mut compute::Buffer
    }
    /// GPU weight meta (buffer + dequant kind) for one resident projection.
    /// Raw pointers stay valid through a recording: gpu_weights is never
    /// mutated during a forward (same invariant as qwen35_matvec_multi).
    fn q35r_meta(&self, name: &str) -> Option<(*const compute::Buffer, MvKind)> {
        self.gpu_weights.get(name).map(|w| (
            &w.buffer as *const compute::Buffer,
            match &w.aux {
                None => MvKind::Plain,
                Some(QuantAux::Nvfp4 { scales, group_size, e4m3, global }) =>
                    MvKind::Nvfp4 { s: scales as *const _, gs: *group_size, e4m3: *e4m3, global: *global },
                Some(QuantAux::Fp8 { scale, per_row }) =>
                    MvKind::Fp8 { s: scale as *const _, per_row: *per_row },
                Some(QuantAux::Mlx4 { scales, biases, group_size }) =>
                    MvKind::Mlx4 { s: scales as *const _, b: biases as *const _, gs: *group_size },
            },
        ))
    }
    /// Record ONE format-routed matvec dispatch (shared by the WS3 resident
    /// CBs). Identical dispatch logic to qwen35_matvec/_multi/_fused.
    fn q35r_rec_mv(
        eng: &mut compute::ComputeEngine,
        cb: ash::vk::CommandBuffer,
        (w_ptr, kind): (*const compute::Buffer, MvKind),
        ip: *const compute::Buffer,
        op: *const compute::Buffer,
        k: usize,
        n: usize,
    ) -> Option<()> {
        unsafe {
            match kind {
                MvKind::Nvfp4 { s, gs, e4m3, global } => {
                    let (shader, r, pc) = nvfp4_dispatch(k, n, gs, e4m3, global);
                    let wg = (n as u32 + r - 1) / r;
                    eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
                }
                MvKind::Fp8 { s, per_row } => {
                    let (shader, r) = matvec_fp8_variant(n);
                    let wg = (n as u32 + r - 1) / r;
                    let pc = matvec_fp8_pc(k, n, per_row);
                    eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
                }
                MvKind::Mlx4 { s, b, gs } => {
                    let (shader, r) = matvec_mlx4_variant_k(k, n);
                    let wg = (n as u32 + r - 1) / r;
                    let pc = matvec_mlx4_pc(k, n, gs as usize);
                    eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*b, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
                }
                MvKind::Plain => {
                    // Shape-aware geometry for the bf16 deltanet projections
                    // (swept winners; every other quant format is unchanged).
                    let (shader, r) = matvec_variant_geom(true, k, n);
                    let wg = (n as u32 + r - 1) / r;
                    let pc = matvec_pc13(k, n);
                    eng.record_to(cb, &shader, &[&*w_ptr, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
                }
            }
        }
        Some(())
    }
    /// Upload the qwen3.6 per-layer norm weights (input/post-attn + the final
    /// model.norm on the last stage) into `gpu_norm_w` ONCE, up front, so no
    /// map inserts happen during a forward (raw Buffer pointers gathered in
    /// the per-layer recording must not be invalidated by a rehash).
    pub(crate) fn ensure_qwen35_norm_weights(&mut self) -> bool {
        let h = match self.qwen35.as_ref() {
            Some(m) => m.config.hidden_size,
            None => return false,
        };
        let mut names: Vec<(String, usize)> = Vec::new();
        for li in self.pp_start..self.pp_end {
            names.push((format!("model.layers.{li}.input_layernorm.weight"), h));
            names.push((format!("model.layers.{li}.post_attention_layernorm.weight"), h));
        }
        if self.pp_last {
            names.push(("model.norm.weight".to_string(), h));
        }
        for (name, n) in names {
            if self.gpu_norm_w.contains_key(&name) {
                continue;
            }
            let data = match self.qwen35.as_ref().unwrap().weights.tensors.get(&name) {
                Some(t) if t.data.len() >= n => f32_slice_to_bytes(&t.data[..n]),
                _ => return false,
            };
            let eng = match self.engine.as_mut() { Some(e) => e, None => return false };
            let buf = match eng.alloc_host_coherent_storage((n * 4) as u64) {
                Ok(b) => b,
                Err(_) => return false,
            };
            if buf.write(&data).is_err() {
                return false;
            }
            self.gpu_norm_w.insert(name, buf);
        }
        true
    }
    /// Allocate the persistent activation buffers for the WS3 resident stage
    /// (once). Sizes come from the live config; slots the stage's layer mix
    /// never touches (e.g. full-attn buffers on an all-linear stage) still
    /// allocate — they are tiny and keep the slot layout fixed.
    pub(crate) fn init_q35res_bufs(&mut self, cfg: &qwen35::Qwen35Config) -> bool {
        if self.q35res_ready {
            return true;
        }
        let h = cfg.hidden_size;
        let e_num = cfg.num_experts;
        let conv_dim = cfg.conv_dim();
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let nv = cfg.linear_num_value_heads;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let mi = cfg.moe_intermediate_size;
        let si = cfg.shared_expert_intermediate_size;
        let vocab = if self.pp_last { cfg.vocab_size } else { 1 };
        // .max(4): Vulkan buffers can't be zero-sized (mirrors ensure_dn_gpu_layer).
        let f4 = |n: usize| ((n * 4).max(4)) as u64;
        let mut sizes: Vec<u64> = vec![
            f4(h), f4(h), f4(h), f4(h), f4(h), f4(e_num),                  // H,X,ATTN,H1,FFIN,RLOG
            f4(conv_dim), f4(value_dim), f4(nv), f4(nv), f4(conv_dim),     // QKV,Z,A,B,CONV
            f4(key_dim), f4(key_dim), f4(value_dim),                       // Q,K,GATED
            f4(2 * q_dim), f4(kv_dim), f4(kv_dim), f4(q_dim),              // QG,AK,AV,GIN
            f4(si), f4(si), f4(si), f4(si), f4(h), f4(1),                  // SG,SU,SA,SM,SO,SL
            f4(h), f4(vocab),                                              // NORMED,VLOG
        ];
        for _ in 0..16 { sizes.push(f4(mi)); }  // GU
        for _ in 0..8 { sizes.push(f4(mi)); }   // ACT
        for _ in 0..8 { sizes.push(f4(mi)); }   // MID
        for _ in 0..8 { sizes.push(f4(h)); }    // DOWN
        debug_assert_eq!(sizes.len(), Q35R_COUNT);
        let eng = match self.engine.as_mut() { Some(e) => e, None => return false };
        let mut bufs = Vec::with_capacity(Q35R_COUNT);
        for &sz in &sizes {
            match eng.alloc_host_coherent_storage(sz) {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    log::warn!("init_q35res_bufs alloc failed: {e}");
                    return false;
                }
            }
        }
        self.q35res_bufs = bufs;
        self.q35res_ready = true;
        true
    }
    /// One-time readiness probe for the WS3 resident stage, cached in
    /// `q35res_ok`. A false verdict is permanent for the process (the stage
    /// falls back to the WS1b/WS2 per-block submit path).
    pub(crate) fn ensure_q35res(&mut self, cfg: &qwen35::Qwen35Config) -> bool {
        if let Some(ok) = self.q35res_ok {
            return ok;
        }
        let ok = self.q35res_probe(cfg);
        if !ok {
            log::warn!(
                "qwen3.6 resident stage (VLLM_VULKAN_Q35_1CB) unavailable on \
                 layers [{}, {}); using the per-block submit path",
                self.pp_start, self.pp_end
            );
        }
        self.q35res_ok = Some(ok);
        ok
    }
    fn q35res_probe(&mut self, cfg: &qwen35::Qwen35Config) -> bool {
        if self.engine.is_none() || self.qwen35.is_none() {
            return false;
        }
        // The resident stage folds the fused deltanet + fused MoE recordings;
        // both sub-paths must be enabled and serviceable.
        if !moe_gpu_enabled() || !dn_gpu_enabled() {
            return false;
        }
        // q35_moe_accum is a fixed top-8 kernel.
        if !cfg.is_moe() || cfg.num_experts_per_tok != 8 {
            return false;
        }
        for layer_idx in self.pp_start..self.pp_end {
            match cfg.layer_types[layer_idx] {
                qwen35::LayerType::LinearAttention => {
                    let p = format!("model.layers.{layer_idx}.linear_attn");
                    for w in ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] {
                        if !self.gpu_weights.contains_key(&format!("{p}.{w}.weight")) {
                            return false;
                        }
                    }
                    if !self.ensure_dn_gpu_layer(cfg, layer_idx) {
                        return false;
                    }
                }
                qwen35::LayerType::FullAttention => {
                    let p = format!("model.layers.{layer_idx}.self_attn");
                    for w in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                        if !self.gpu_weights.contains_key(&format!("{p}.{w}.weight")) {
                            return false;
                        }
                    }
                }
            }
            if !self.ensure_moe_gpu_layer(layer_idx) {
                return false;
            }
            if !self.ensure_moe_shared_gpu_layer(layer_idx) {
                return false;
            }
        }
        if !self.ensure_qwen35_norm_weights() {
            return false;
        }
        // Chunked-alloc lm_head (§5, P2) is NOT wired into the resident tail
        // (`q35r_meta`/`q35r_rec_mv` read `gpu_weights` only, one binding) — if
        // the last stage's lm_head went into `chunked_weights` instead (opt-in
        // VLLM_VULKAN_MAX_ALLOC_MB), the resident stage is unavailable here and
        // must fall back to the per-block path, which DOES check
        // `chunked_weights` in `qwen35_matvec`. Non-last stages don't own
        // lm_head, so this is a no-op for them.
        if self.pp_last {
            let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
            if self.chunked_weights.contains_key(&lm_name) {
                return false;
            }
        }
        self.init_q35res_bufs(cfg)
    }
    /// WS3: run this stage's whole layer span through PERSISTENT GPU buffers.
    /// Ports the proven `forward_qwen_gpu_resident` scheme to the qwen3.6
    /// hybrid layer: input/post-attn RMSNorms, residual adds, the whole
    /// GatedDeltaNet block, the router matvec, the fused MoE (routed + shared)
    /// AND its score-weighted accumulate all record into per-layer command
    /// buffers over resident buffers — the hidden state never visits the host
    /// between layers. Per layer:
    ///   - linear-attn: ONE fenced CB (in-norm → deltanet → residual →
    ///     post-norm → router), host top-8 over 256 logits (the only host
    ///     boundary), then ONE MoE CB (…→ q35_moe_accum → next hidden) that
    ///     is submitted WITHOUT a fence on the CB ring (VLLM_VULKAN_CB_RING)
    ///     so its execution overlaps the next layer's recording.
    ///   - full-attn: one extra fenced CB + host boundary for the SDPA
    ///     (q/k/v → host qknorm/RoPE/KV/SDPA → o_proj CB), then the same MoE
    ///     CB. Full-attn is ~2/8 layers per stage.
    /// The lm_head tail (last stage) records final-norm + lm_head into one
    /// more CB when the lm weight is GPU-resident, else falls back to the
    /// host tail after draining.
    ///
    /// Math is unchanged vs the WS1b/WS2 path except: the two RMSNorms + the
    /// two residual adds move from libm/host-f32 to the (already-validated)
    /// rms_norm_f32_mul/add_f32_f32_f32 kernels, and the MoE tail's sigmoid
    /// moves in-shader — all last-ulp exp/rsqrt differences, gated by the
    /// on-node cos + argmax A/B. Returns None (fall back) only when the
    /// readiness probe fails; a mid-token GPU error also returns None (the
    /// caller re-runs the stage on the fallback path — same accepted posture
    /// as the WS1b/WS2 fused paths, where a post-submit failure can leave
    /// advanced state; such errors have never been observed and would recur
    /// on retry anyway).
    pub(crate) fn forward_qwen35_span_resident(
        &mut self,
        cfg: &qwen35::Qwen35Config,
        hidden_in: &[f32],
        pos: usize,
    ) -> Option<Vec<f32>> {
        if !q35_1cb_enabled() {
            return None;
        }
        let h = cfg.hidden_size;
        if hidden_in.len() != h {
            return None;
        }
        if !self.ensure_q35res(cfg) {
            return None;
        }
        let eps = cfg.rms_norm_eps;
        let e_num = cfg.num_experts;
        let top_k = cfg.num_experts_per_tok;
        let mi = cfg.moe_intermediate_size;
        let si_dim = cfg.shared_expert_intermediate_size;
        let vocab = cfg.vocab_size;
        // GatedDeltaNet dims.
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let kern = cfg.linear_conv_kernel_dim;
        let ratio = if nk > 0 { nv / nk } else { 0 };
        // GatedAttention dims.
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let rotary = cfg.rotary_dim();
        let theta = cfg.rope_theta;

        // M5b ring: drain the previous token's in-flight CBs BEFORE the host
        // writes Q35R_H, and reset the descriptor cursor once per token.
        let use_ring = self.engine.as_ref().map_or(false, |e| e.ring_active());
        if use_ring {
            self.engine.as_mut()?.begin_forward_ring().ok()?;
        }
        // KV-offload chunk-boundary capture: the ring drain above (or, off
        // the ring, the previous token's fenced/blocking submits) leaves
        // dn_gpu quiescent and holding exactly `[0, pos)` — the instant
        // `export_prefix` needs for `kv_cache_store`'s aligned-down boundary.
        // Must happen BEFORE the Q35R_H write / layer loop below mutates it.
        let _ = self.capture_gdn_boundary_resident(pos);
        // Off-by-one localization (diag-only): fingerprint the device state at
        // the boundary's immediate neighbours — `[0, N-1)` and `[0, N+1)` — so
        // the on-cluster re-gate can see which of `pre(-1)`/`boundary`/`adv(+1)`
        // matches its reference (and the per-token state delta between them).
        if q35_kv_boundary_diag_enabled() {
            let c = crate::kvstore::CHUNK;
            if pos > 1 && pos % c == c - 1 {
                self.kv_boundary_diag_log(pos, "pre(-1)");
            } else if pos % c == 1 {
                self.kv_boundary_diag_log(pos, "adv(+1)");
            }
        }
        // The only host write of hidden; it stays GPU-resident all stage.
        unsafe { (*self.q35r_ptr_mut(Q35R_H)).write(&f32_slice_to_bytes(hidden_in)).ok()?; }

        // Layer-invariant push constants / shader picks.
        let rms_h = rmsnorm_pc(h, eps);
        let add_pc = ew_mul_pc(h as u32);
        let (rt_shader, rt_r) = matvec_f32_variant_k(h, e_num);
        let rt_wg = (e_num as u32 + rt_r - 1) / rt_r;
        let pc_rt = matvec_pc13(h, e_num);
        let (gu_shader, gu_r) = crate::matvec_mlx4_moe_variant_k(h, mi);
        let wg_mi = (mi as u32 + gu_r - 1) / gu_r;
        let (down_shader, down_r) = crate::matvec_mlx4_moe_variant_k(mi, h);
        let wg_h = (h as u32 + down_r - 1) / down_r;
        let (sgu_shader, sgu_r) = matvec_f32_variant_k(h, si_dim);
        let swg_si = (si_dim as u32 + sgu_r - 1) / sgu_r;
        let (sd_shader, sd_r) = matvec_f32_variant_k(si_dim, h);
        let swg_h = (h as u32 + sd_r - 1) / sd_r;
        let (sl_shader, _) = matvec_f32_variant(1);
        let pc_sgu = matvec_pc13(h, si_dim);
        let pc_sd = matvec_pc13(si_dim, h);
        let pc_sl = matvec_pc13(h, 1);
        let silu_mi = ew_unary_pc(mi as u32);
        let mul_mi = ew_mul_pc(mi as u32);
        let silu_si = ew_unary_pc(si_dim as u32);
        let mul_si = ew_mul_pc(si_dim as u32);

        for layer_idx in self.pp_start..self.pp_end {
            match cfg.layer_types[layer_idx] {
                // ── linear layer CB_A: in-norm → deltanet → residual →
                //    post-norm → router, ONE fenced submit. ─────────────────
                qwen35::LayerType::LinearAttention => {
                    let t = std::time::Instant::now();
                    let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");
                    let w_qkv = self.q35r_meta(&ln("in_proj_qkv.weight"))?;
                    let w_z = self.q35r_meta(&ln("in_proj_z.weight"))?;
                    let w_a = self.q35r_meta(&ln("in_proj_a.weight"))?;
                    let w_b = self.q35r_meta(&ln("in_proj_b.weight"))?;
                    let w_out = self.q35r_meta(&ln("out_proj.weight"))?;
                    let inln = &self.gpu_norm_w[&format!("model.layers.{layer_idx}.input_layernorm.weight")] as *const compute::Buffer;
                    let postln = &self.gpu_norm_w[&format!("model.layers.{layer_idx}.post_attention_layernorm.weight")] as *const compute::Buffer;
                    let dnl = &self.dn_gpu[&layer_idx];
                    let dn_convw = &dnl.conv_w as *const compute::Buffer;
                    let dn_params = &dnl.params as *const compute::Buffer;
                    let dn_cstate = &dnl.conv_state as *const compute::Buffer;
                    let dn_state = &dnl.state as *const compute::Buffer;
                    let router_p = &self.moe_shared_gpu[&layer_idx].router as *const compute::Buffer;
                    let hp = self.q35r_ptr(Q35R_H);
                    let xp = self.q35r_ptr(Q35R_X);
                    let qkvp = self.q35r_ptr(Q35R_QKV);
                    let zp = self.q35r_ptr(Q35R_Z);
                    let apb = self.q35r_ptr(Q35R_A);
                    let bpb = self.q35r_ptr(Q35R_B);
                    let convp = self.q35r_ptr(Q35R_CONV);
                    let qp = self.q35r_ptr(Q35R_Q);
                    let kp = self.q35r_ptr(Q35R_K);
                    let gatedp = self.q35r_ptr(Q35R_GATED);
                    let attnp = self.q35r_ptr(Q35R_ATTN);
                    let h1p = self.q35r_ptr(Q35R_H1);
                    let ffinp = self.q35r_ptr(Q35R_FFIN);
                    let rlogp = self.q35r_ptr(Q35R_RLOG);
                    let pc_conv = q35_conv_pc(conv_dim, kern);
                    let inv = 1.0f32 / (kd as f32).sqrt();
                    let pc_qk = q35_qknorm_pc(nk, kd, key_dim, 1e-6, inv);
                    let pc_gdn = q35_gdn_pc(kd, vd, ratio, 2 * key_dim, eps, nv);

                    let eng = self.engine.as_mut()?;
                    let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
                    // Q35_TSTAMP attribution: GPU-exec wall of this CB (fenced
                    // path only — a deferred ring fence has no read-back point).
                    let ts_on = q35_tstamp_enabled() && !use_ring && eng.ensure_ts_pool(16);
                    if ts_on {
                        eng.ts_cmd_reset(cb, 0, 2);
                        eng.ts_cmd_mark(cb, 0, true);
                    }
                    // Leading barrier: orders this CB after the previous MoE
                    // CB's H write on the same queue (spec-correct cross-CB
                    // visibility; the previous CB may still be in flight on
                    // the ring).
                    eng.record_barrier_to(cb);
                    unsafe {
                        eng.record_to(cb, "rms_norm_f32_mul", &[&*hp, &*inln, &*xp], &rms_h, (1, 1, 1)).ok()?;
                        eng.record_barrier_to(cb);
                        Self::q35r_rec_mv(eng, cb, w_qkv, xp, qkvp, h, conv_dim)?;
                        Self::q35r_rec_mv(eng, cb, w_z, xp, zp, h, value_dim)?;
                        Self::q35r_rec_mv(eng, cb, w_a, xp, apb, h, nv)?;
                        Self::q35r_rec_mv(eng, cb, w_b, xp, bpb, h, nv)?;
                        eng.record_barrier_to(cb);
                        eng.record_to(cb, "q35_dn_conv_step",
                            &[&*dn_convw, &*qkvp, &*dn_cstate, &*convp],
                            &pc_conv, ((conv_dim as u32 + 255) / 256, 1, 1)).ok()?;
                        eng.record_barrier_to(cb);
                        eng.record_to(cb, "q35_gdn_qknorm",
                            &[&*convp, &*qp, &*kp],
                            &pc_qk, (2 * nk as u32, 1, 1)).ok()?;
                        eng.record_barrier_to(cb);
                        eng.record_to(cb, "q35_gdn_step",
                            &[&*qp, &*kp, &*convp, &*apb, &*bpb, &*zp, &*dn_params, &*dn_state, &*gatedp],
                            &pc_gdn, (nv as u32, 1, 1)).ok()?;
                        eng.record_barrier_to(cb);
                        Self::q35r_rec_mv(eng, cb, w_out, gatedp, attnp, value_dim, h)?;
                        eng.record_barrier_to(cb);
                        eng.record_to(cb, "add_f32_f32_f32", &[&*hp, &*attnp, &*h1p, &*h1p], &add_pc, ((h as u32 + 255) / 256, 1, 1)).ok()?;
                        eng.record_barrier_to(cb);
                        eng.record_to(cb, "rms_norm_f32_mul", &[&*h1p, &*postln, &*ffinp], &rms_h, (1, 1, 1)).ok()?;
                        eng.record_barrier_to(cb);
                        eng.record_to(cb, &rt_shader, &[&*router_p, &*ffinp, &*rlogp], &pc_rt, (rt_wg, 1, 1)).ok()?;
                    }
                    if ts_on { eng.ts_cmd_mark(cb, 1, false); }
                    prof_add("q35r_cba_record", t);
                    let ts = std::time::Instant::now();
                    if use_ring {
                        eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
                        eng.wait_batch_pipelined().ok()?;
                    } else {
                        eng.submit_batch(cb).ok()?;
                    }
                    prof_add("q35r_cba_fence", ts);
                    if ts_on {
                        if let Ok(v) = eng.ts_read_ns(0, 2) {
                            prof_add_ns("q35ts_cba_total", (v[1] - v[0]).max(0.0) as u128);
                        }
                    }
                }
                // ── full-attn layer: proj CB → host SDPA → o/router CB. ─────
                qwen35::LayerType::FullAttention => {
                    let t = std::time::Instant::now();
                    let ln = |s: &str| format!("model.layers.{layer_idx}.self_attn.{s}");
                    let w_q = self.q35r_meta(&ln("q_proj.weight"))?;
                    let w_k = self.q35r_meta(&ln("k_proj.weight"))?;
                    let w_v = self.q35r_meta(&ln("v_proj.weight"))?;
                    let w_o = self.q35r_meta(&ln("o_proj.weight"))?;
                    let inln = &self.gpu_norm_w[&format!("model.layers.{layer_idx}.input_layernorm.weight")] as *const compute::Buffer;
                    let postln = &self.gpu_norm_w[&format!("model.layers.{layer_idx}.post_attention_layernorm.weight")] as *const compute::Buffer;
                    let router_p = &self.moe_shared_gpu[&layer_idx].router as *const compute::Buffer;
                    let hp = self.q35r_ptr(Q35R_H);
                    let xp = self.q35r_ptr(Q35R_X);
                    let qgp = self.q35r_ptr(Q35R_QG);
                    let akp = self.q35r_ptr(Q35R_AK);
                    let avp = self.q35r_ptr(Q35R_AV);
                    let ginp = self.q35r_ptr(Q35R_GIN);
                    let attnp = self.q35r_ptr(Q35R_ATTN);
                    let h1p = self.q35r_ptr(Q35R_H1);
                    let ffinp = self.q35r_ptr(Q35R_FFIN);
                    let rlogp = self.q35r_ptr(Q35R_RLOG);

                    // CB_A1: in-norm → q/k/v projections (q is double-width
                    // [query|gate] per head), one fenced submit.
                    {
                        let eng = self.engine.as_mut()?;
                        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
                        eng.record_barrier_to(cb);
                        unsafe {
                            eng.record_to(cb, "rms_norm_f32_mul", &[&*hp, &*inln, &*xp], &rms_h, (1, 1, 1)).ok()?;
                            eng.record_barrier_to(cb);
                            Self::q35r_rec_mv(eng, cb, w_q, xp, qgp, h, 2 * q_dim)?;
                            Self::q35r_rec_mv(eng, cb, w_k, xp, akp, h, kv_dim)?;
                            Self::q35r_rec_mv(eng, cb, w_v, xp, avp, h, kv_dim)?;
                        }
                        if use_ring {
                            eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
                            eng.wait_batch_pipelined().ok()?;
                        } else {
                            eng.submit_batch(cb).ok()?;
                        }
                    }
                    prof_add("q35r_attn_proj_fence", t);

                    // Host boundary: split [q|gate], per-head q/k RMSNorm,
                    // partial RoPE, KV append, causal GQA SDPA, sigmoid gate
                    // (mirrors qwen35_gated_attention_gpu bit-for-bit).
                    let th = std::time::Instant::now();
                    let q_and_gate = read_f32_buf(unsafe { &*qgp }, 2 * q_dim);
                    let mut k = read_f32_buf(unsafe { &*akp }, kv_dim);
                    let v = read_f32_buf(unsafe { &*avp }, kv_dim);
                    let mut q = vec![0.0f32; q_dim];
                    let mut gate = vec![0.0f32; q_dim];
                    for head in 0..nq {
                        let base = head * 2 * hd;
                        q[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base..base + hd]);
                        gate[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base + hd..base + 2 * hd]);
                    }
                    let qn = self.qwen35_w(&ln("q_norm.weight"));
                    let kn = self.qwen35_w(&ln("k_norm.weight"));
                    for hi in 0..nq {
                        let s = &mut q[hi * hd..(hi + 1) * hd];
                        let n2 = model::cpu_rms_norm(s, &qn, eps);
                        s.copy_from_slice(&n2);
                    }
                    for hi in 0..nkv {
                        let s = &mut k[hi * hd..(hi + 1) * hd];
                        let n2 = model::cpu_rms_norm(s, &kn, eps);
                        s.copy_from_slice(&n2);
                    }
                    model::cpu_rope(&mut q, &mut k, pos, nq, nkv, hd, rotary, theta);
                    // Append to the host KvCache (retained for KvStore export
                    // either way) and resolve geometry before any &mut self
                    // GPU call — mirrors qwen35_gated_attention_gpu.
                    let (max_seq_len, seq_len) = {
                        let qm = self.qwen35.as_mut().unwrap();
                        let sidx = qm.state_idx(layer_idx);
                        let cache = match &mut qm.layer_state[sidx] {
                            qwen35::LayerState::Full(c) => c,
                            _ => unreachable!("full_attention layer has a KV cache"),
                        };
                        cache.append(&k, &v);
                        (cache.max_seq_len, cache.seq_len)
                    };
                    // item-4a: same GPU-SDPA seam as the per-block path.
                    let gpu_attn_out = if q35_gpu_attn_enabled() && self.flags.attn == flags::AttnKernel::Sg {
                        self.gpu_kv_append(layer_idx, &k, &v, pos, nkv, hd, max_seq_len);
                        self.gpu_sdpa_resident(layer_idx, &q, nq, nkv, hd, seq_len, scale, max_seq_len)
                    } else {
                        None
                    };
                    let attn_out = match gpu_attn_out {
                        Some(out) => out,
                        None => {
                            let qm = self.qwen35.as_mut().unwrap();
                            let sidx = qm.state_idx(layer_idx);
                            let cache = match &mut qm.layer_state[sidx] {
                                qwen35::LayerState::Full(c) => c,
                                _ => unreachable!("full_attention layer has a KV cache"),
                            };
                            model::cpu_sdpa(&q, cache.k_up_to_now(), cache.v_up_to_now(), nq, nkv, hd, cache.seq_len, scale, None)
                        }
                    };
                    let gated: Vec<f32> = attn_out.iter().zip(&gate)
                        .map(|(&a, &g)| a * (1.0 / (1.0 + (-g).exp()))).collect();
                    unsafe { (*self.q35r_ptr_mut(Q35R_GIN)).write(&f32_slice_to_bytes(&gated)).ok()?; }
                    prof_add("q35r_attn_host", th);

                    // CB_A2: o_proj → residual → post-norm → router, fenced.
                    let t2 = std::time::Instant::now();
                    let eng = self.engine.as_mut()?;
                    let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
                    eng.record_barrier_to(cb);
                    unsafe {
                        Self::q35r_rec_mv(eng, cb, w_o, ginp, attnp, q_dim, h)?;
                        eng.record_barrier_to(cb);
                        eng.record_to(cb, "add_f32_f32_f32", &[&*hp, &*attnp, &*h1p, &*h1p], &add_pc, ((h as u32 + 255) / 256, 1, 1)).ok()?;
                        eng.record_barrier_to(cb);
                        eng.record_to(cb, "rms_norm_f32_mul", &[&*h1p, &*postln, &*ffinp], &rms_h, (1, 1, 1)).ok()?;
                        eng.record_barrier_to(cb);
                        eng.record_to(cb, &rt_shader, &[&*router_p, &*ffinp, &*rlogp], &pc_rt, (rt_wg, 1, 1)).ok()?;
                    }
                    if use_ring {
                        eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
                        eng.wait_batch_pipelined().ok()?;
                    } else {
                        eng.submit_batch(cb).ok()?;
                    }
                    prof_add("q35r_attn_o_fence", t2);
                }
            }

            // ── routing on host: top-8 over E logits (tiny). ────────────────
            let tr = std::time::Instant::now();
            let logits = read_f32_buf(unsafe { &*self.q35r_ptr(Q35R_RLOG) }, e_num);
            let routing = moe::route_from_logits(&logits, top_k);
            if routing.indices.len() != 8 {
                return None; // guarded by the probe's top_k == 8 check
            }
            prof_add("q35r_route_host", tr);

            // ── MoE CB_B: routed gate/up ‖ shared gate/up/logit → silu → mul
            //    → downs → q35_moe_accum (weighted accumulate + sigmoid-gated
            //    shared add + residual) → next hidden IN PLACE. On the ring
            //    this submits WITHOUT a fence: its execution overlaps the next
            //    layer's CB_A recording; the next fenced wait covers it (same
            //    FIFO queue + leading barriers). ───────────────────────────
            let tb = std::time::Instant::now();
            let layer = &self.moe_gpu[&layer_idx];
            let group_size = layer.gate.group_size;
            let (g_pack_stride, g_sb_stride) = (layer.gate.pack_stride, layer.gate.sb_stride);
            let (u_pack_stride, u_sb_stride) = (layer.up.pack_stride, layer.up.sb_stride);
            let (d_pack_stride, d_sb_stride) = (layer.down.pack_stride, layer.down.sb_stride);
            let g_packed = &layer.gate.packed as *const compute::Buffer;
            let g_scales = &layer.gate.scales as *const compute::Buffer;
            let g_biases = &layer.gate.biases as *const compute::Buffer;
            let u_packed = &layer.up.packed as *const compute::Buffer;
            let u_scales = &layer.up.scales as *const compute::Buffer;
            let u_biases = &layer.up.biases as *const compute::Buffer;
            let d_packed = &layer.down.packed as *const compute::Buffer;
            let d_scales = &layer.down.scales as *const compute::Buffer;
            let d_biases = &layer.down.biases as *const compute::Buffer;
            let shared = &self.moe_shared_gpu[&layer_idx];
            let s_gatep = &shared.gate as *const compute::Buffer;
            let s_upp = &shared.up as *const compute::Buffer;
            let s_downp = &shared.down as *const compute::Buffer;
            let s_egp = &shared.expert_gate as *const compute::Buffer;
            let hp = self.q35r_ptr(Q35R_H);
            let h1p = self.q35r_ptr(Q35R_H1);
            let ffinp = self.q35r_ptr(Q35R_FFIN);
            let sgp = self.q35r_ptr(Q35R_SG);
            let sup = self.q35r_ptr(Q35R_SU);
            let sap = self.q35r_ptr(Q35R_SA);
            let smp = self.q35r_ptr(Q35R_SM);
            let sop = self.q35r_ptr(Q35R_SO);
            let slp = self.q35r_ptr(Q35R_SL);
            let gu: Vec<*const compute::Buffer> = (0..16).map(|i| self.q35r_ptr(Q35R_GU0 + i)).collect();
            let act: Vec<*const compute::Buffer> = (0..8).map(|i| self.q35r_ptr(Q35R_ACT0 + i)).collect();
            let mid: Vec<*const compute::Buffer> = (0..8).map(|i| self.q35r_ptr(Q35R_MID0 + i)).collect();
            let dwn: Vec<*const compute::Buffer> = (0..8).map(|i| self.q35r_ptr(Q35R_DOWN0 + i)).collect();
            let pc_acc = q35_moe_accum_pc(h, &routing.scores);

            let eng = self.engine.as_mut()?;
            let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
            // Q35_TSTAMP attribution: bracket each MoE phase with a GPU
            // timestamp (7 slots: top, routed-gu, shared-gu, silu, mul, down,
            // accum). Phases are separated by full compute barriers, so each
            // BOTTOM_OF_PIPE delta is that phase's GPU-exec + its launch
            // overhead. Fenced (ring-off) path only. NOTE: routed-gu and
            // shared-gu have NO barrier between them, so the GPU may overlap
            // them — read t1/t2 as "all-routed-done" / "all-phase-1-done".
            let ts_on = q35_tstamp_enabled() && !use_ring && eng.ensure_ts_pool(16);
            if ts_on {
                eng.ts_cmd_reset(cb, 8, 7);
                eng.ts_cmd_mark(cb, 8, true);
            }
            eng.record_barrier_to(cb);
            unsafe {
                for (slot, &e) in routing.indices.iter().enumerate() {
                    let pc_g = matvec_mlx4_pc_off(h, mi, group_size, e * g_pack_stride, e * g_sb_stride);
                    let pc_u = matvec_mlx4_pc_off(h, mi, group_size, e * u_pack_stride, e * u_sb_stride);
                    eng.record_to(cb, &gu_shader,
                        &[&*g_packed, &*g_scales, &*g_biases, &*ffinp, &*gu[slot * 2]],
                        &pc_g, (wg_mi, 1, 1)).ok()?;
                    eng.record_to(cb, &gu_shader,
                        &[&*u_packed, &*u_scales, &*u_biases, &*ffinp, &*gu[slot * 2 + 1]],
                        &pc_u, (wg_mi, 1, 1)).ok()?;
                }
                if ts_on { eng.ts_cmd_mark(cb, 9, false); }
                eng.record_to(cb, &sgu_shader, &[&*s_gatep, &*ffinp, &*sgp], &pc_sgu, (swg_si, 1, 1)).ok()?;
                eng.record_to(cb, &sgu_shader, &[&*s_upp, &*ffinp, &*sup], &pc_sgu, (swg_si, 1, 1)).ok()?;
                eng.record_to(cb, &sl_shader, &[&*s_egp, &*ffinp, &*slp], &pc_sl, (1, 1, 1)).ok()?;
                if ts_on { eng.ts_cmd_mark(cb, 10, false); }
                eng.record_barrier_to(cb);
                for slot in 0..8 {
                    eng.record_to(cb, "silu_f32", &[&*gu[slot * 2], &*act[slot]], &silu_mi, ((mi as u32 + 511) / 512, 1, 1)).ok()?;
                }
                eng.record_to(cb, "silu_f32", &[&*sgp, &*sap], &silu_si, ((si_dim as u32 + 511) / 512, 1, 1)).ok()?;
                if ts_on { eng.ts_cmd_mark(cb, 11, false); }
                eng.record_barrier_to(cb);
                for slot in 0..8 {
                    eng.record_to(cb, "mul_f32_f32_f32", &[&*act[slot], &*gu[slot * 2 + 1], &*mid[slot]], &mul_mi, ((mi as u32 + 255) / 256, 1, 1)).ok()?;
                }
                eng.record_to(cb, "mul_f32_f32_f32", &[&*sap, &*sup, &*smp], &mul_si, ((si_dim as u32 + 255) / 256, 1, 1)).ok()?;
                if ts_on { eng.ts_cmd_mark(cb, 12, false); }
                eng.record_barrier_to(cb);
                for (slot, &e) in routing.indices.iter().enumerate() {
                    let pc_d = matvec_mlx4_pc_off(mi, h, group_size, e * d_pack_stride, e * d_sb_stride);
                    eng.record_to(cb, &down_shader,
                        &[&*d_packed, &*d_scales, &*d_biases, &*mid[slot], &*dwn[slot]],
                        &pc_d, (wg_h, 1, 1)).ok()?;
                }
                eng.record_to(cb, &sd_shader, &[&*s_downp, &*smp, &*sop], &pc_sd, (swg_h, 1, 1)).ok()?;
                if ts_on { eng.ts_cmd_mark(cb, 13, false); }
                eng.record_barrier_to(cb);
                eng.record_to(cb, "q35_moe_accum",
                    &[&*dwn[0], &*dwn[1], &*dwn[2], &*dwn[3], &*dwn[4], &*dwn[5], &*dwn[6], &*dwn[7],
                      &*sop, &*slp, &*h1p, &*hp],
                    &pc_acc, ((h as u32 + 255) / 256, 1, 1)).ok()?;
                if ts_on { eng.ts_cmd_mark(cb, 14, false); }
            }
            prof_add("q35r_cbb_record", tb);
            let ts = std::time::Instant::now();
            if use_ring {
                eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
            } else {
                eng.submit_batch(cb).ok()?;
            }
            prof_add("q35r_cbb_submit", ts);
            if ts_on {
                if let Ok(v) = eng.ts_read_ns(8, 7) {
                    let d = |a: usize, b: usize| (v[b] - v[a]).max(0.0) as u128;
                    prof_add_ns("q35ts_moe_rgu",   d(0, 1)); // 16 routed 4-bit gate/up
                    prof_add_ns("q35ts_moe_sgu",   d(1, 2)); // 3 shared f32 gu+logit
                    prof_add_ns("q35ts_moe_silu",  d(2, 3)); // 9 silu (tiny compute)
                    prof_add_ns("q35ts_moe_mul",   d(3, 4)); // 9 mul (tiny compute)
                    prof_add_ns("q35ts_moe_down",  d(4, 5)); // 8 routed + 1 shared down
                    prof_add_ns("q35ts_moe_accum", d(5, 6)); // 1 accum
                    prof_add_ns("q35ts_moe_total", d(0, 6));
                }
            }
        }

        // ── stage tail ───────────────────────────────────────────────────────
        let tt = std::time::Instant::now();
        if self.pp_last {
            let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
            if let Some(w_lm) = self.q35r_meta(&lm_name) {
                // Final norm + lm_head in one more CB over resident hidden.
                let normp = &self.gpu_norm_w["model.norm.weight"] as *const compute::Buffer;
                let hp = self.q35r_ptr(Q35R_H);
                let np = self.q35r_ptr(Q35R_NORMED);
                let vp = self.q35r_ptr(Q35R_VLOG);
                let eng = self.engine.as_mut()?;
                let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
                eng.record_barrier_to(cb);
                unsafe {
                    eng.record_to(cb, "rms_norm_f32_mul", &[&*hp, &*normp, &*np], &rms_h, (1, 1, 1)).ok()?;
                    eng.record_barrier_to(cb);
                    Self::q35r_rec_mv(eng, cb, w_lm, np, vp, h, vocab)?;
                }
                if use_ring {
                    eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
                    eng.wait_batch_pipelined().ok()?;
                } else {
                    eng.submit_batch(cb).ok()?;
                }
                let out = read_f32_buf(unsafe { &*vp }, vocab);
                // P3: stash the pre-`model.norm` residual (Q35R_H) for the MTP
                // draft head. Only when a head is loaded — the 8KB readback is
                // pure overhead otherwise.
                if self.mtp_head.is_some() {
                    self.q35_last_prenorm = Some(read_f32_buf(unsafe { &*hp }, h));
                }
                prof_add("q35r_tail_fence", tt);
                Some(out)
            } else {
                // lm_head lives in the f16 host table: drain, read the hidden
                // back and run the proven host norm + matvec tail.
                if use_ring {
                    self.engine.as_mut()?.wait_batch_pipelined().ok()?;
                }
                let hidden = read_f32_buf(unsafe { &*self.q35r_ptr(Q35R_H) }, h);
                if self.mtp_head.is_some() {
                    self.q35_last_prenorm = Some(hidden.clone());
                }
                prof_add("q35r_tail_fence", tt);
                let norm_w = self.qwen35_w("model.norm.weight");
                let normed = model::cpu_rms_norm(&hidden, &norm_w, eps);
                Some(self.qwen35_matvec(&lm_name, &normed, h, vocab))
            }
        } else {
            // Non-last stage: drain and ship the hidden to the next stage.
            if use_ring {
                self.engine.as_mut()?.wait_batch_pipelined().ok()?;
            }
            let hidden = read_f32_buf(unsafe { &*self.q35r_ptr(Q35R_H) }, h);
            prof_add("q35r_hidden_out", tt);
            Some(hidden)
        }
    }
    /// CPU reference forward for qwen3_5, one token. f32 projections (host) via
    /// `Qwen35Model::forward_layers_from_hidden` (bit-identical per-layer math),
    /// with the embed lookup + final lm_head matmul done from the f16 host
    /// tables (q35_f16_host). This is the parity baseline vs forward_qwen35_gpu;
    /// it shares `layer_state` so decode-step state persists across both paths.
    pub(crate) fn forward_qwen35_cpu_ref(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        let cfg = self.qwen35.as_ref().unwrap().config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let vocab = cfg.vocab_size;
        let n_layers = cfg.num_hidden_layers;

        // Embedding (packed per-row decode or f16 host → f32, one row).
        let hidden0: Vec<f32> = self.q35_embed_row(token_id as usize, h);

        // Decoder layers (f32 host projections, exact CPU recurrence).
        let hidden = self.qwen35.as_mut().unwrap()
            .forward_layers_from_hidden(&hidden0, pos, n_layers);

        // Final norm (f32 host) + lm_head matmul from the f16 host table.
        let norm_w = self.qwen35_w("model.norm.weight");
        let normed = model::cpu_rms_norm(&hidden, &norm_w, eps);
        let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
        let lm = self.q35_f16_host.get(&lm_name)
            .expect("qwen3_5 lm_head f16 host missing");
        // logits[j] = sum_i normed[i] * lm[j*h + i]  (row-major [vocab, h]).
        let mut logits = vec![0.0f32; vocab];
        for j in 0..vocab {
            let row = &lm[j * h..(j + 1) * h];
            let mut acc = 0.0f32;
            for i in 0..h {
                acc += normed[i] * half::f16::from_bits(row[i]).to_f32();
            }
            logits[j] = acc;
        }
        logits
    }
}
