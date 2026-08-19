// SPDX-License-Identifier: Apache-2.0
//! Per-model pyo3 seam for `laguna` — moved verbatim out of the monolithic
//! `VulkanModel` `#[pymethods]` block in `lib.rs` (Phase A upstream refactor).
//! Behavior-preserving code motion: method bodies are byte-for-byte identical.
//! Kept as separate `#[pymethods] impl VulkanModel` block(s) via pyo3's
//! `multiple-pymethods` feature so a per-model upstream PR can carve this file.
#![allow(clippy::all)]

use crate::*;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;


#[pymethods]
impl VulkanModel {

    /// FULL layer gate: `LagunaGpuModel::layer_forward` vs `laguna_layer_forward`.
    fn debug_laguna_layer(&mut self, layer_idx: usize, seq: usize) -> PyResult<(f64, f64, bool)> {
        let (dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let hidden = laguna_gate_fill(seq * hs, 0x1A6u64 ^ layer_idx as u64);
        let oracle = laguna::load_owned_layer_cpu(&dir, &cfg, layer_idx)
            .map_err(PyRuntimeError::new_err)?;
        let out_cpu = laguna::laguna_layer_forward(&hidden, seq, layer_idx, &oracle, &cfg);
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        let out_gpu = g.layer_forward(&hidden, seq, layer_idx);
        Ok(laguna_cos_maxdiff(&out_gpu, &out_cpu))
    }


    /// Attention-seam gate: `LagunaGpuModel::attn` vs `laguna_attn` on a random
    /// post-input_layernorm `hidden_normed` (isolates q/k/v/o/g f16 matvec +
    /// qk-norm + YaRN/sliding rope + softplus gate + SDPA).
    fn debug_laguna_attn(&mut self, layer_idx: usize, seq: usize) -> PyResult<(f64, f64, bool)> {
        let (dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let h = laguna_gate_fill(seq * hs, 0xA77u64 ^ layer_idx as u64);
        let oracle = laguna::load_owned_layer_cpu(&dir, &cfg, layer_idx)
            .map_err(PyRuntimeError::new_err)?;
        let out_cpu = laguna::laguna_attn(&h, seq, layer_idx, &oracle.attn, &cfg);
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        let out_gpu = g.attn(&h, seq, layer_idx);
        Ok(laguna_cos_maxdiff(&out_gpu, &out_cpu))
    }


    /// MoE-block gate (single token): `LagunaGpuModel::moe_token` vs
    /// `laguna_moe_token` (isolates router + nvfp4-e4m3 expert matvec + shared
    /// f16 + top-k accumulate). MoE layers only.
    fn debug_laguna_moe(&mut self, layer_idx: usize) -> PyResult<(f64, f64, bool)> {
        let (dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let h = laguna_gate_fill(hs, 0x30Eu64 ^ layer_idx as u64);
        let oracle = laguna::load_owned_layer_cpu(&dir, &cfg, layer_idx)
            .map_err(PyRuntimeError::new_err)?;
        let moe = match &oracle.mlp {
            laguna::OwnedMlp::Moe(m) => m,
            laguna::OwnedMlp::Dense(_) =>
                return Err(PyRuntimeError::new_err(format!("layer {layer_idx} is dense, not MoE"))),
        };
        let out_cpu = laguna::laguna_moe_token(&h, moe, &cfg);
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        let out_gpu = g.moe_token(&h, layer_idx);
        Ok(laguna_cos_maxdiff(&out_gpu, &out_cpu))
    }


    /// MULTI-LAYER chain gate: run `[start,end)` GPU `layer_forward`s in sequence
    /// vs the CPU oracle chain over the same layers (composition + residual
    /// accumulation). Loads ONE oracle layer at a time (memory-safe: ~1.3GB, not
    /// the whole window). The resident model must own `[start,end)`.
    fn debug_laguna_chain(&mut self, start: usize, end: usize, seq: usize) -> PyResult<(f64, f64, bool)> {
        let (dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let h0 = laguna_gate_fill(seq * hs, 0xC4A1u64 ^ ((start * 131 + end) as u64));
        // CPU oracle chain (one layer resident at a time).
        let mut h_cpu = h0.clone();
        for l in start..end {
            let oracle = laguna::load_owned_layer_cpu(&dir, &cfg, l)
                .map_err(PyRuntimeError::new_err)?;
            h_cpu = laguna::laguna_layer_forward(&h_cpu, seq, l, &oracle, &cfg);
        }
        // GPU resident chain.
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        let mut h_gpu = h0;
        for l in start..end {
            h_gpu = g.layer_forward(&h_gpu, seq, l);
        }
        Ok(laguna_cos_maxdiff(&h_gpu, &h_cpu))
    }


    /// RAW multi-layer chain outputs for the int8-attn accuracy A/B: identical
    /// control flow to `debug_laguna_chain` (same deterministic `h0`, one oracle
    /// layer resident at a time) but returns the flat `[seq*hidden]` GPU and CPU
    /// window outputs so python can score PER-POSITION cosine + top-1 argmax
    /// agreement over `seq` (>=16) positions. `h0` depends only on (start,end,seq)
    /// so it is byte-identical across the int8-OFF and int8-ON processes, and the
    /// CPU oracle is full-precision in BOTH — so the per-position argmax/cos delta
    /// between the two runs is exactly the q8_0 weight-quant effect. Format-
    /// agnostic: the GPU chain picks up whatever quant the resident loader stored.
    fn debug_laguna_chain_raw(
        &mut self, start: usize, end: usize, seq: usize,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let (dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let h0 = laguna_gate_fill(seq * hs, 0xC4A1u64 ^ ((start * 131 + end) as u64));
        let mut h_cpu = h0.clone();
        for l in start..end {
            let oracle = laguna::load_owned_layer_cpu(&dir, &cfg, l)
                .map_err(PyRuntimeError::new_err)?;
            h_cpu = laguna::laguna_layer_forward(&h_cpu, seq, l, &oracle, &cfg);
        }
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        let mut h_gpu = h0;
        for l in start..end {
            h_gpu = g.layer_forward(&h_gpu, seq, l);
        }
        Ok((h_gpu, h_cpu))
    }


    /// KV-CACHE decode gate (the whole point — a KV bug is invisible at T=1 and
    /// only shows at T≥2). Over the resident MID window `[pp_start, pp_end)`,
    /// compares the CACHED path (a `forward_prefill_hidden` over `prefill_seq`
    /// then `steps` single-token `forward_decode_hidden` calls) against the
    /// validated STATELESS `forward_hidden` recompute at every position. Uses a
    /// deterministic-random `[prefill_seq+steps, hidden]` input sequence; the
    /// per-token hidden is arbitrary (this isolates the K/V cache + windowed SDPA
    /// + abs-pos rope, the only new logic). Returns the WORST (cos, max_abs_diff,
    /// argmax_match) across all compared positions — expected bit-exact (cos=1,
    /// maxdiff≈0) since the cached and stateless paths run the identical ops.
    /// A MID window (`pp_end < num_layers`, no lm_head) is required so the output
    /// is `[seq*hidden]` at every step. Exercises BOTH layer types when the
    /// window spans a full/sliding mix (e.g. `[0,8)`).
    fn debug_laguna_kvcache(&mut self, prefill_seq: usize, steps: usize) -> PyResult<(f64, f64, bool)> {
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let total = prefill_seq + steps;
        let seqh = laguna_gate_fill(total * hs, 0x9C5Eu64);
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if g.pp_last {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_kvcache needs a MID window (pp_end < num_layers, no lm_head)",
            ));
        }
        let mut worst_cos = 1.0f64;
        let mut worst_maxd = 0.0f64;
        let mut all_ok = true;
        let mut acc = |a: &[f32], b: &[f32]| {
            let (c, d, ok) = laguna_cos_maxdiff(a, b);
            worst_cos = worst_cos.min(c);
            worst_maxd = worst_maxd.max(d);
            all_ok &= ok;
        };
        // CACHED prefill over [0, prefill_seq); compare its LAST position against
        // the stateless recompute over the same prefix.
        let pref = g.forward_prefill_hidden(&seqh[..prefill_seq * hs], prefill_seq);
        let refp = g.forward_hidden(&seqh[..prefill_seq * hs], prefill_seq);
        acc(
            &pref[(prefill_seq - 1) * hs..prefill_seq * hs],
            &refp[(prefill_seq - 1) * hs..prefill_seq * hs],
        );
        // Single-token decode each subsequent position vs stateless recompute of
        // that position's prefix (the T≥2 test).
        for i in 0..steps {
            let p = prefill_seq + i;
            let dec = g.forward_decode_hidden(&seqh[p * hs..(p + 1) * hs]);
            let refo = g.forward_hidden(&seqh[..(p + 1) * hs], p + 1);
            acc(&dec, &refo[p * hs..(p + 1) * hs]);
        }
        Ok((worst_cos, worst_maxd, all_ok))
    }


    /// RESIDENT 1-CB FOLD gate (the whole point of the fold — a batched-submit
    /// bug is invisible at T=1 and only shows across ≥2 decode steps with the
    /// cache advancing). Over the resident MID window `[pp_start, pp_end)`,
    /// compares the FOLDED single-token decode (`forward_decode_hidden_1cb`:
    /// batched q/k/v/g CB + GPU `swiglu_f32` MoE CB + o_proj CB) against the
    /// validated STATELESS `forward_hidden` recompute at every position — the
    /// same reference `debug_laguna_kvcache` uses for the per-op decode, so a
    /// pass transitively proves folded == per-op == stateless. A per-op prefill
    /// populates the caches; then `steps` folded decodes advance them. Exercises
    /// BOTH layer types on a full/sliding window (e.g. `[0,8)`), the top-k MoE,
    /// the shared expert, and the softplus gate. Expected argmax-exact every
    /// step, cos≥0.999 (the only numeric delta is `swiglu_f32` silu vs host
    /// libm). Requires a MID window (no lm_head) so output is `[seq*hidden]`.
    fn debug_laguna_1cb(&mut self, prefill_seq: usize, steps: usize) -> PyResult<(f64, f64, bool)> {
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let total = prefill_seq + steps;
        let seqh = laguna_gate_fill(total * hs, 0x9C5Eu64);
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if g.pp_last {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_1cb needs a MID window (pp_end < num_layers, no lm_head)",
            ));
        }
        let mut worst_cos = 1.0f64;
        let mut worst_maxd = 0.0f64;
        let mut all_ok = true;
        let mut acc = |a: &[f32], b: &[f32]| {
            let (c, d, ok) = laguna_cos_maxdiff(a, b);
            worst_cos = worst_cos.min(c);
            worst_maxd = worst_maxd.max(d);
            all_ok &= ok;
        };
        // 1-CB prefill populates the DEVICE-RESIDENT K/V planes (the store the
        // folded decode reads); bit-exact vs stateless already.
        let _ = g.forward_prefill_hidden_1cb(&seqh[..prefill_seq * hs], prefill_seq);
        // Each subsequent position: FOLDED decode vs stateless recompute prefix.
        for i in 0..steps {
            let p = prefill_seq + i;
            let dec = g.forward_decode_hidden_1cb(&seqh[p * hs..(p + 1) * hs]);
            let refo = g.forward_hidden(&seqh[..(p + 1) * hs], p + 1);
            acc(&dec, &refo[p * hs..(p + 1) * hs]);
        }
        Ok((worst_cos, worst_maxd, all_ok))
    }


    /// PIECE-3 GATE: DEVICE-RESIDENT K/V planes + GPU softplus gate. The resident
    /// arena is invisible at T=1 — an append write-offset/stride bug or a
    /// resident-vs-host softplus divergence only shows across ≥2 decode steps with
    /// the plane advancing. Runs a 1-CB (resident-plane) prefill over
    /// `[0, prefill_seq)`, then `steps` folded single-token decodes (each appends
    /// its post-rope K/V into the resident plane on-GPU and reads it back via SDPA
    /// with no round-trip, gate applied by the `laguna_softplus_gate` shader), and
    /// compares every decode position against the validated STATELESS `forward_hidden`
    /// recompute of that prefix (the ground truth the committed host-KV/host-softplus
    /// 1-CB path is already gated against — so a pass proves resident+GPU-softplus ==
    /// host-KV+host-softplus transitively). Exercises BOTH attn types over the
    /// resident window (report the full/sliding split), the top-k MoE, the shared
    /// expert, and the softplus gate. Expected argmax-exact every step, cos≥0.999.
    /// Requires a MID window (no lm_head) so the compared output is `[seq*hidden]`.
    /// Returns `(worst_cos, worst_maxdiff, all_argmax_ok, n_full_layers, n_sliding_layers)`.
    fn debug_laguna_residentkv(
        &mut self,
        prefill_seq: usize,
        steps: usize,
    ) -> PyResult<(f64, f64, bool, usize, usize)> {
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let total = prefill_seq + steps;
        let seqh = laguna_gate_fill(total * hs, 0x9C5Eu64);
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if g.pp_last {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_residentkv needs a MID window (pp_end < num_layers, no lm_head)",
            ));
        }
        // Report the attn-type split of the resident window (both must be present).
        let mut n_full = 0usize;
        let mut n_slide = 0usize;
        for li in g.pp_start..g.pp_end {
            if cfg.layer_is_full[li] {
                n_full += 1;
            } else {
                n_slide += 1;
            }
        }
        let mut worst_cos = 1.0f64;
        let mut worst_maxd = 0.0f64;
        let mut all_ok = true;
        let mut acc = |a: &[f32], b: &[f32]| {
            let (c, d, ok) = laguna_cos_maxdiff(a, b);
            worst_cos = worst_cos.min(c);
            worst_maxd = worst_maxd.max(d);
            all_ok &= ok;
        };
        // Resident-plane 1-CB prefill (fills the GPU K/V planes across the window).
        let _ = g.forward_prefill_hidden_1cb(&seqh[..prefill_seq * hs], prefill_seq);
        for i in 0..steps {
            let p = prefill_seq + i;
            let dec = g.forward_decode_hidden_1cb(&seqh[p * hs..(p + 1) * hs]);
            let refo = g.forward_hidden(&seqh[..(p + 1) * hs], p + 1);
            acc(&dec, &refo[p * hs..(p + 1) * hs]);
        }
        Ok((worst_cos, worst_maxd, all_ok, n_full, n_slide))
    }


    /// SPAN-FOLD FINAL-PIECE GATE: GPU decode-SDPA over the resident K/V planes
    /// (`VLLM_VULKAN_LAGUNA_GPU_SDPA=1`). An SDPA bug is invisible at T=1 — a
    /// wrong GQA map, a plane stride/offset error, or a botched sliding clamp only
    /// shows across ≥2 decode steps with the plane advancing and BOTH attn types
    /// present. Identical structure to `debug_laguna_residentkv`: a resident 1-CB
    /// prefill over `[0, prefill_seq)`, then `steps` folded single-token decodes
    /// (each appends its post-rope K/V into the resident plane and attends it with
    /// the `laguna_gpu_sdpa` subgroup kernel — NO host readback), each compared to
    /// the validated STATELESS `forward_hidden` recompute of that prefix (which
    /// uses host `cpu_sdpa`). So a pass proves GPU-SDPA == host-SDPA transitively.
    /// Requires `VLLM_VULKAN_LAGUNA_GPU_SDPA=1` (else this just re-runs the host
    /// path). Exercises BOTH attn regimes over the resident window (returns the
    /// full/sliding split). Expected argmax-exact every step, cos≥0.999 (the only
    /// delta is GPU exp/subgroupAdd last-ulp vs host softmax — a nonzero maxdiff is
    /// EXPECTED). Requires a MID window (no lm_head).
    /// Returns `(worst_cos, worst_maxdiff, all_argmax_ok, n_full_layers, n_sliding_layers)`.
    fn debug_laguna_gpusdpa(
        &mut self,
        prefill_seq: usize,
        steps: usize,
    ) -> PyResult<(f64, f64, bool, usize, usize)> {
        if !crate::flags::flags_global().laguna_gpu_sdpa {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_gpusdpa needs VLLM_VULKAN_LAGUNA_GPU_SDPA=1 (this gates the GPU-SDPA path)",
            ));
        }
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let total = prefill_seq + steps;
        let seqh = laguna_gate_fill(total * hs, 0x9C5Eu64);
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if g.pp_last {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_gpusdpa needs a MID window (pp_end < num_layers, no lm_head)",
            ));
        }
        let mut n_full = 0usize;
        let mut n_slide = 0usize;
        for li in g.pp_start..g.pp_end {
            if cfg.layer_is_full[li] {
                n_full += 1;
            } else {
                n_slide += 1;
            }
        }
        let mut worst_cos = 1.0f64;
        let mut worst_maxd = 0.0f64;
        let mut all_ok = true;
        let mut acc = |a: &[f32], b: &[f32]| {
            let (c, d, ok) = laguna_cos_maxdiff(a, b);
            worst_cos = worst_cos.min(c);
            worst_maxd = worst_maxd.max(d);
            all_ok &= ok;
        };
        let _ = g.forward_prefill_hidden_1cb(&seqh[..prefill_seq * hs], prefill_seq);
        for i in 0..steps {
            let p = prefill_seq + i;
            let dec = g.forward_decode_hidden_1cb(&seqh[p * hs..(p + 1) * hs]);
            let refo = g.forward_hidden(&seqh[..(p + 1) * hs], p + 1);
            acc(&dec, &refo[p * hs..(p + 1) * hs]);
        }
        Ok((worst_cos, worst_maxd, all_ok, n_full, n_slide))
    }


    /// LEVER-2 GATE: GPU per-layer attention MATH — GPU qk-norm + GPU sliding-rope
    /// (`VLLM_VULKAN_LAGUNA_GPU_ATTNMATH=1`). Identical structure/oracle to
    /// `debug_laguna_gpusdpa`: a resident 1-CB prefill over `[0, prefill_seq)`,
    /// then `steps` folded single-token decodes (each does GPU qk-norm before
    /// rope, GPU-YaRN rope for full-attn layers, GPU plain rope for sliding
    /// layers), each compared to the STATELESS `forward_hidden` recompute of that
    /// prefix (which uses host qk-norm + host rope). A pass proves the on-GPU
    /// attn-math == host attn-math transitively, over BOTH attn regimes (returns
    /// the full/sliding split), across the plane-advancing window where a
    /// position/GQA bug would surface. Requires `VLLM_VULKAN_LAGUNA_GPU_ATTNMATH=1`
    /// and a MID window (no lm_head). Expected argmax-exact every step, cos≥0.999
    /// (delta = GPU sin/cos last-ulp for rope + shared-mem-tree reduction order
    /// for qk-norm vs host sequential — a nonzero maxdiff is EXPECTED).
    /// Returns `(worst_cos, worst_maxdiff, all_argmax_ok, n_full_layers, n_sliding_layers)`.
    fn debug_laguna_attnmath(
        &mut self,
        prefill_seq: usize,
        steps: usize,
    ) -> PyResult<(f64, f64, bool, usize, usize)> {
        if !crate::flags::flags_global().laguna_gpu_attnmath {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_attnmath needs VLLM_VULKAN_LAGUNA_GPU_ATTNMATH=1 (this gates the GPU attn-math path)",
            ));
        }
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let total = prefill_seq + steps;
        let seqh = laguna_gate_fill(total * hs, 0x9C5Eu64);
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if g.pp_last {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_attnmath needs a MID window (pp_end < num_layers, no lm_head)",
            ));
        }
        let mut n_full = 0usize;
        let mut n_slide = 0usize;
        for li in g.pp_start..g.pp_end {
            if cfg.layer_is_full[li] {
                n_full += 1;
            } else {
                n_slide += 1;
            }
        }
        let mut worst_cos = 1.0f64;
        let mut worst_maxd = 0.0f64;
        let mut all_ok = true;
        let mut acc = |a: &[f32], b: &[f32]| {
            let (c, d, ok) = laguna_cos_maxdiff(a, b);
            worst_cos = worst_cos.min(c);
            worst_maxd = worst_maxd.max(d);
            all_ok &= ok;
        };
        let _ = g.forward_prefill_hidden_1cb(&seqh[..prefill_seq * hs], prefill_seq);
        for i in 0..steps {
            let p = prefill_seq + i;
            let dec = g.forward_decode_hidden_1cb(&seqh[p * hs..(p + 1) * hs]);
            let refo = g.forward_hidden(&seqh[..(p + 1) * hs], p + 1);
            acc(&dec, &refo[p * hs..(p + 1) * hs]);
        }
        Ok((worst_cos, worst_maxd, all_ok, n_full, n_slide))
    }


    /// QK-NORM micro-gate: the `rms_norm_f32_mul` per-head norm (as used by the
    /// LEVER-2 GPU attn-math path) vs the host `cpu_rms_norm_inplace`, in
    /// isolation over `num_heads` random rows of `head_dim`. Elementwise RMSNorm,
    /// so this is expected ~bit-exact (the only delta is the shared-mem tree
    /// reduction order vs the host sequential sum). Returns
    /// `(cos, maxdiff, bit_exact)` — `bit_exact` true iff every element matches to
    /// the last bit.
    fn debug_laguna_qknorm(
        &mut self,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
        seed: u64,
    ) -> PyResult<(f64, f64, bool)> {
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        Ok(g.qknorm_micro(num_heads, head_dim, eps, seed))
    }


    /// SOFTPLUS micro-gate: the `laguna_softplus_gate` shader vs the host `softplus`
    /// in isolation, over `n` values spanning `[-lo, hi]` PLUS the edge cases the
    /// stable form must nail — the ln2 crossover (x≈0), large positive x (softplus
    /// → x), large negative x (softplus → e^x → 0). Returns `(cos, maxdiff,
    /// bit_exact)` where `bit_exact` is true iff every element matches to the last
    /// bit (else the tiny exp/log last-ulp noise is in `maxdiff`).
    fn debug_laguna_softplus(&mut self) -> PyResult<(f64, f64, bool)> {
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        // Dense sweep [-40,40] + explicit edge cases around the ln2 crossover and
        // the large-|x| tails.
        let mut xs: Vec<f32> = (0..801).map(|i| -40.0 + (i as f32) * 0.1).collect();
        for &e in &[
            -88.0f32, -30.0, -1e-3, -1e-6, 0.0, 1e-6, 1e-3, 0.6931472, 30.0, 60.0, 88.0,
        ] {
            xs.push(e);
        }
        let host: Vec<f32> = xs
            .iter()
            .map(|&x| x.max(0.0) + (-x.abs()).exp().ln_1p())
            .collect();
        let gpu = g.softplus_gpu(&xs);
        let (cos, maxd, _argmax) = laguna_cos_maxdiff(&gpu, &host);
        let bit_exact = gpu.iter().zip(&host).all(|(&a, &b)| a.to_bits() == b.to_bits());
        Ok((cos, maxd, bit_exact))
    }


    /// A/B the GPU `laguna_moe_accum` fold vs the HOST weighted-accumulate loop
    /// (== commit bb33073), ISOLATED to the MoE tail. For `ntok` distinct
    /// post-norm hidden vectors at MoE `layer_idx`, run `moe_token_1cb` with the
    /// GPU accumulate ON and OFF and compare per token. The routed top-10
    /// matvecs + ungated shared matvec are identical between the two calls; only
    /// the accumulate differs, so this pinpoints the kernel. Returns
    /// `(worst_cos, worst_maxdiff, all_ok, gpu_path_tokens)` — `gpu_path_tokens`
    /// counts tokens where the router selected exactly 10 experts (the GPU
    /// accumulate path engaged); it MUST equal `ntok` for a valid gate.
    fn debug_laguna_moe_accum(&mut self, layer_idx: usize, ntok: usize) -> PyResult<(f64, f64, bool, usize)> {
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        if cfg.mlp_only_layers.contains(&layer_idx) {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_moe_accum needs a MoE layer (layer 0 is dense)",
            ));
        }
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if layer_idx < g.pp_start || layer_idx >= g.pp_end {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_moe_accum: layer_idx outside this stage's [pp_start, pp_end)",
            ));
        }
        let mut worst_cos = 1.0f64;
        let mut worst_maxd = 0.0f64;
        let mut all_ok = true;
        let mut gpu_path_tokens = 0usize;
        for i in 0..ntok {
            // Distinct hidden per token (arbitrary input: both paths route it
            // identically; only the accumulate differs).
            let h = laguna_gate_fill(hs, 0x1A6Eu64.wrapping_add((i as u64).wrapping_mul(0x100000001B3)));
            if g.moe_router_topk(&h, layer_idx) == 10 {
                gpu_path_tokens += 1;
            }
            let gpu = g.moe_token_1cb_accum(&h, layer_idx, true);
            let host = g.moe_token_1cb_accum(&h, layer_idx, false);
            let (c, d, ok) = laguna_cos_maxdiff(&gpu, &host);
            worst_cos = worst_cos.min(c);
            worst_maxd = worst_maxd.max(d);
            all_ok &= ok;
        }
        Ok((worst_cos, worst_maxd, all_ok, gpu_path_tokens))
    }


    /// A/B the CB-BATCH MoE dispatch fold (`VLLM_VULKAN_LAGUNA_CBBATCH`) vs the
    /// per-expert 1-CB path, ISOLATED to `moe_token_1cb`. For `ntok` distinct
    /// post-norm hidden vectors at MoE `layer_idx`, run `moe_token_1cb_full` with
    /// the CB-batch ON (30 expert matvecs → 3 batched dispatches + flat swiglu +
    /// concatenated-down accum) and OFF (per-expert), both with GPU accumulate
    /// ON, and compare per token. The batched matvec reduces in the SAME
    /// BLOCK_SIZE/NUM_ROWS order as the per-expert kernel and the swiglu/accum are
    /// elementwise/slot-order identical, so this is expected BIT-EXACT (every
    /// element to the last bit), not merely argmax-exact. Returns
    /// `(worst_cos, worst_maxdiff, all_argmax_ok, bit_exact_all, gpu_path_tokens)`
    /// — `bit_exact_all` is true iff every token matched to the last bit;
    /// `gpu_path_tokens` counts tokens where the router selected exactly 10
    /// experts (the batched path engaged) and MUST equal `ntok` for a valid gate.
    fn debug_laguna_cbbatch(&mut self, layer_idx: usize, ntok: usize) -> PyResult<(f64, f64, bool, bool, usize)> {
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        if cfg.mlp_only_layers.contains(&layer_idx) {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_cbbatch needs a MoE layer (layer 0 is dense)",
            ));
        }
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if layer_idx < g.pp_start || layer_idx >= g.pp_end {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_cbbatch: layer_idx outside this stage's [pp_start, pp_end)",
            ));
        }
        let mut worst_cos = 1.0f64;
        let mut worst_maxd = 0.0f64;
        let mut all_ok = true;
        let mut bit_exact_all = true;
        let mut gpu_path_tokens = 0usize;
        for i in 0..ntok {
            let h = laguna_gate_fill(hs, 0x1A6Eu64.wrapping_add((i as u64).wrapping_mul(0x100000001B3)));
            if g.moe_router_topk(&h, layer_idx) == 10 {
                gpu_path_tokens += 1;
            }
            let batched = g.moe_token_1cb_full(&h, layer_idx, true, true);
            let perexpert = g.moe_token_1cb_full(&h, layer_idx, true, false);
            let (c, d, ok) = laguna_cos_maxdiff(&batched, &perexpert);
            worst_cos = worst_cos.min(c);
            worst_maxd = worst_maxd.max(d);
            all_ok &= ok;
            bit_exact_all &= batched.iter().zip(&perexpert).all(|(&a, &b)| a.to_bits() == b.to_bits());
        }
        Ok((worst_cos, worst_maxd, all_ok, bit_exact_all, gpu_path_tokens))
    }


    /// Per-token MoE host-time A/B for the CB-batch lever: times `iters` calls of
    /// `moe_token_1cb_full` at MoE `layer_idx` with `cbbatch` ON vs OFF (GPU
    /// accumulate ON both), after `warmup` untimed calls. On the UMA appliance
    /// each `record_to` is a host `update_descriptor_sets`, so this wall-time is a
    /// direct read on the dispatch-recording tax the lever cuts (per-layer
    /// recordings ~45 -> ~9). Returns `(ms_per_call_batched, ms_per_call_perexpert)`.
    fn debug_laguna_moe_time(&mut self, layer_idx: usize, iters: usize, warmup: usize) -> PyResult<(f64, f64)> {
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        if cfg.mlp_only_layers.contains(&layer_idx) {
            return Err(PyRuntimeError::new_err("debug_laguna_moe_time needs a MoE layer"));
        }
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if layer_idx < g.pp_start || layer_idx >= g.pp_end {
            return Err(PyRuntimeError::new_err("debug_laguna_moe_time: layer_idx outside stage"));
        }
        let h = laguna_gate_fill(hs, 0x51A6u64);
        let run = |g: &mut crate::laguna_gpu::LagunaGpuModel, cbbatch: bool| -> f64 {
            for _ in 0..warmup {
                let _ = g.moe_token_1cb_full(&h, layer_idx, true, cbbatch);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = g.moe_token_1cb_full(&h, layer_idx, true, cbbatch);
            }
            t0.elapsed().as_secs_f64() * 1e3 / iters as f64
        };
        let ms_batched = run(g, true);
        let ms_perexpert = run(g, false);
        Ok((ms_batched, ms_perexpert))
    }


    /// LEVER 4 GATE (CPU shared-expert ∥ routed overlap). Sweeps EVERY MoE layer
    /// in this stage's `[pp_start, pp_end)` window × `ntok` synthetic tokens and,
    /// per (layer, token), compares:
    ///   (A) `moe_token_1cb_overlap(_, true)`  — routed CB submitted NON-BLOCKING,
    ///       shared expert on the host rayon pool CONCURRENTLY, sync, combine.
    ///   (B) `moe_token_1cb_overlap(_, false)` — routed CB submitted BLOCKING,
    ///       shared expert SEQUENTIALLY after, combine.
    /// (A) vs (B) must be BIT-EXACT (maxdiff == 0, argmax-exact) — the overlap
    /// changes only WHEN the shared branch computes, never the reduction order, so
    /// any nonzero delta is a sync/race bug. Also reports (A) vs the deployed
    /// `moe_token_1cb_accum(_, true)` (GPU-f16 shared) cosine, which is the same
    /// silu-vs-libm class of delta as the existing 1-CB gate (expected cos≥0.999,
    /// argmax-exact) — this confirms flipping the flag ON is a safe swap.
    ///
    /// Returns `(overlap_vs_seq_cos, overlap_vs_seq_maxdiff, overlap_vs_seq_argmax_ok,
    ///           overlap_vs_accum_cos, overlap_vs_accum_maxdiff, overlap_vs_accum_argmax_ok,
    ///           moe_layers_swept, gpu_path_tokens)`. `gpu_path_tokens` counts
    /// (layer,token) pairs that routed exactly top-10.
    #[allow(clippy::type_complexity)]
    fn debug_laguna_cpuoverlap(
        &mut self,
        ntok: usize,
    ) -> PyResult<(f64, f64, bool, f64, f64, bool, usize, usize)> {
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        // Bit-exact (A) vs (B).
        let mut seq_cos = 1.0f64;
        let mut seq_maxd = 0.0f64;
        let mut seq_ok = true;
        // (A) overlap vs deployed GPU-f16-shared accum path.
        let mut acc_cos = 1.0f64;
        let mut acc_maxd = 0.0f64;
        let mut acc_ok = true;
        let mut moe_layers = 0usize;
        let mut gpu_path_tokens = 0usize;
        for layer_idx in g.pp_start..g.pp_end {
            if cfg.mlp_only_layers.contains(&layer_idx) {
                continue; // dense layer 0 has no MoE block
            }
            moe_layers += 1;
            for i in 0..ntok {
                let h = laguna_gate_fill(
                    hs,
                    0xC0FFEEu64
                        .wrapping_add((layer_idx as u64).wrapping_mul(0x9E3779B1))
                        .wrapping_add((i as u64).wrapping_mul(0x100000001B3)),
                );
                if g.moe_router_topk(&h, layer_idx) == 10 {
                    gpu_path_tokens += 1;
                }
                let overlap = g.moe_token_1cb_overlap(&h, layer_idx, true);
                let seqref = g.moe_token_1cb_overlap(&h, layer_idx, false);
                let accum = g.moe_token_1cb_accum(&h, layer_idx, true);
                let (c1, d1, ok1) = laguna_cos_maxdiff(&overlap, &seqref);
                seq_cos = seq_cos.min(c1);
                seq_maxd = seq_maxd.max(d1);
                seq_ok &= ok1;
                let (c2, d2, ok2) = laguna_cos_maxdiff(&overlap, &accum);
                acc_cos = acc_cos.min(c2);
                acc_maxd = acc_maxd.max(d2);
                acc_ok &= ok2;
            }
        }
        Ok((seq_cos, seq_maxd, seq_ok, acc_cos, acc_maxd, acc_ok, moe_layers, gpu_path_tokens))
    }


    /// LEVER 4 TIMING probe (on-node "is the shared bucket hidden?"). For MoE
    /// `layer_idx`, runs `iters` calls of `moe_token_1cb_overlap(_, true)` (async
    /// routed CB + concurrent host shared) and `iters` of `moe_token_1cb_overlap(
    /// _, false)` (blocking routed CB then sequential host shared) on a fixed
    /// token, plus `moe_token_1cb_accum(_, true)` (deployed GPU-f16 shared) as a
    /// reference, and returns the MEAN per-call wall time in milliseconds:
    /// `(overlap_ms, sequential_ms, accum_ms)`. `sequential_ms − overlap_ms` is
    /// the wall the overlap reclaimed; when the ~0.7ms host shared bucket fully
    /// hides under the routed-expert GPU wall, `overlap_ms ≈ accum_ms` (the GPU
    /// wall) while `sequential_ms ≈ accum_ms + shared_bucket`. A few warm-up
    /// iterations are discarded. Single-node numbers are governor-noisy; read as
    /// a ratio, not an absolute (the cluster A/B is the perf authority).
    fn debug_laguna_cpuoverlap_timing(
        &mut self,
        layer_idx: usize,
        iters: usize,
    ) -> PyResult<(f64, f64, f64)> {
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        if cfg.mlp_only_layers.contains(&layer_idx) {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_cpuoverlap_timing needs a MoE layer (layer 0 is dense)",
            ));
        }
        let g = self.laguna_gpu.as_mut().ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if layer_idx < g.pp_start || layer_idx >= g.pp_end {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_cpuoverlap_timing: layer_idx outside this stage's [pp_start, pp_end)",
            ));
        }
        let h = laguna_gate_fill(hs, 0xBEEF1234u64.wrapping_add(layer_idx as u64));
        let warmup = 3usize;
        let n = iters.max(1) as f64;
        // mode: 0 = overlap, 1 = sequential, 2 = deployed accum.
        let mut timeit = |g: &mut laguna_gpu::LagunaGpuModel, mode: u8| -> f64 {
            let mut call = |g: &mut laguna_gpu::LagunaGpuModel| match mode {
                0 => { g.moe_token_1cb_overlap(&h, layer_idx, true); }
                1 => { g.moe_token_1cb_overlap(&h, layer_idx, false); }
                _ => { g.moe_token_1cb_accum(&h, layer_idx, true); }
            };
            for _ in 0..warmup { call(g); }
            let t0 = std::time::Instant::now();
            for _ in 0..iters { call(g); }
            t0.elapsed().as_secs_f64() * 1e3 / n
        };
        let overlap_ms = timeit(g, 0);
        let sequential_ms = timeit(g, 1);
        let accum_ms = timeit(g, 2);
        Ok((overlap_ms, sequential_ms, accum_ms))
    }


    /// A/B the GPU `laguna_router` (sigmoid + e_score bias + gate matvec +
    /// top-k on device, tiny idx+weight readback) vs the HOST
    /// `nemotron::router_forward` reference, ISOLATED to the router. For `ntok`
    /// distinct post-norm hidden vectors at MoE `layer_idx`, compute the selected
    /// top-10 set + weights BOTH ways and compare. A router flip changes which
    /// experts fire, so the CRITICAL gate is `sets_identical` (same 10 experts,
    /// same slot order). `w_maxdiff` is the worst per-weight abs diff (host
    /// libm vs GPU `exp` — expected ~1e-7). Returns
    /// `(sets_identical, order_identical, w_maxdiff, mismatched_tokens, gpu_topk10)`
    /// — `gpu_topk10` counts tokens where the GPU router selected exactly 10
    /// experts (must equal `ntok`).
    fn debug_laguna_gpu_router(
        &mut self,
        layer_idx: usize,
        ntok: usize,
    ) -> PyResult<(bool, bool, f64, usize, usize)> {
        let (_dir, cfg) = laguna_gpu_dir_cfg(self)?;
        let hs = cfg.hidden_size;
        if cfg.mlp_only_layers.contains(&layer_idx) {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_gpu_router needs a MoE layer (layer 0 is dense)",
            ));
        }
        let g = self
            .laguna_gpu
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("not resident"))?;
        if layer_idx < g.pp_start || layer_idx >= g.pp_end {
            return Err(PyRuntimeError::new_err(
                "debug_laguna_gpu_router: layer_idx outside this stage's [pp_start, pp_end)",
            ));
        }
        let mut sets_identical = true;
        let mut order_identical = true;
        let mut w_maxdiff = 0.0f64;
        let mut mismatched = 0usize;
        let mut gpu_topk10 = 0usize;
        for i in 0..ntok {
            let h = laguna_gate_fill(hs, 0x1A6Eu64.wrapping_add((i as u64).wrapping_mul(0x100000001B3)));
            let (h_idx, h_w) = g.host_router_select(&h, layer_idx);
            let (gp_idx, gp_w) = g.gpu_router_select(&h, layer_idx);
            if gp_idx.len() == 10 {
                gpu_topk10 += 1;
            }
            // Set identity: same experts regardless of order.
            let mut hs_sorted = h_idx.clone();
            let mut gs_sorted = gp_idx.clone();
            hs_sorted.sort_unstable();
            gs_sorted.sort_unstable();
            let set_ok = hs_sorted == gs_sorted;
            let order_ok = h_idx == gp_idx;
            if !set_ok {
                sets_identical = false;
            }
            if !order_ok {
                order_identical = false;
            }
            if !set_ok || !order_ok {
                mismatched += 1;
            }
            // Weight diff compared slot-for-slot only when the order matches
            // (otherwise the slots aren't comparable — the set mismatch already
            // flags it).
            if order_ok {
                for (&a, &b) in gp_w.iter().zip(&h_w) {
                    w_maxdiff = w_maxdiff.max((a as f64 - b as f64).abs());
                }
            }
        }
        Ok((sets_identical, order_identical, w_maxdiff, mismatched, gpu_topk10))
    }


    /// PP-stage forward for the resident Laguna model (`scripts/pp_laguna.py`).
    /// The Python-driven building block: recv `[seq*hidden]` from Python, run this
    /// stage, return `[seq*hidden]` (mid) / `[vocab]` (last) to Python, which does
    /// the vCCL hop. Unlike the qwen/nemotron stateful single-token hop, Laguna is
    /// STATELESS full-sequence recompute (no KV cache), so the whole `[seq, hidden]`
    /// crosses the hop and this call re-runs the full prefix every decode step.
    ///
    ///  - FIRST stage (`pp_first`): `tokens` = the full token sequence so far;
    ///    embeds + runs `[pp_start, pp_end)` → `[seq*hidden]`. `hidden_in`/`seq`
    ///    ignored (`seq == tokens.len()`).
    ///  - MID / LAST stage: `hidden_in` = `[seq*hidden]` from the previous stage;
    ///    `tokens` ignored. Runs the resident layers; the LAST stage
    ///    (`pp_last`) applies final norm + lm_head (last position) → `[vocab]`.
    fn forward_pp_laguna(
        &mut self,
        tokens: Vec<u32>,
        hidden_in: Vec<f32>,
        seq: usize,
    ) -> PyResult<Vec<f32>> {
        let g = self.laguna_gpu.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err(
                "forward_pp_laguna needs a resident Laguna model (set VLLM_VULKAN_LAGUNA_RESIDENT=1)",
            )
        })?;
        if g.pp_first {
            Ok(g.forward(&tokens))
        } else {
            Ok(g.forward_hidden(&hidden_in, seq))
        }
    }


    /// KV-CACHED prefill PP-stage forward (`scripts/pp_laguna.py`, KVCACHE path).
    /// The cache-populating twin of `forward_pp_laguna`: resets this stage's
    /// per-layer K/V caches and runs the full `[seq]` prefix, storing every
    /// layer's post-rope K/V. Return shape is identical (`[seq*hidden]` mid /
    /// `[vocab]` last) and BIT-EXACT with `forward_pp_laguna` — the difference is
    /// only that the caches are now populated for the subsequent single-token
    /// `forward_pp_laguna_decode` calls.
    ///  - FIRST stage: `tokens` = full prefix; `hidden_in`/`seq` ignored.
    ///  - MID/LAST : `hidden_in` = `[seq*hidden]`; `tokens` ignored.
    fn forward_pp_laguna_prefill(
        &mut self,
        tokens: Vec<u32>,
        hidden_in: Vec<f32>,
        seq: usize,
    ) -> PyResult<Vec<f32>> {
        let g = self.laguna_gpu.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err(
                "forward_pp_laguna_prefill needs a resident Laguna model (VLLM_VULKAN_LAGUNA_RESIDENT=1)",
            )
        })?;
        // MUST mirror `forward_pp_laguna_decode`'s `laguna_1cb` branch: the per-op
        // prefill populates the HOST `kv` map, the 1-CB prefill the DEVICE-RESIDENT
        // K/V planes, and the decode reads whichever its OWN `laguna_1cb` branch
        // selects. If prefill ignored the flag (as it did), then with the fold ON
        // the per-op prefill filled the host map while the 1-CB decode read the
        // never-populated resident planes (start_pos=0, empty prefix) → the g1
        // KV-decode collapse. Branch identically so prefill and decode share a cache.
        let fold = crate::flags::flags_global().laguna_1cb;
        if g.pp_first {
            Ok(if fold { g.forward_prefill_tokens_1cb(&tokens) } else { g.forward_prefill_tokens(&tokens) })
        } else {
            Ok(if fold { g.forward_prefill_hidden_1cb(&hidden_in, seq) } else { g.forward_prefill_hidden(&hidden_in, seq) })
        }
    }


    /// KV-CACHED single-token decode PP-stage forward. O(1)/step: projects only
    /// the one new token, appends its K/V to each layer's cache, attends the
    /// cache (full for YaRN layers, last-`sliding_window` for sliding layers).
    /// The PP hop carries only `[1*hidden]`, not `[seq*hidden]`.
    ///  - FIRST stage: `new_tok` = the single new token id; `hidden_in` ignored.
    ///  - MID/LAST : `hidden_in` = `[1*hidden]` from the previous stage;
    ///    `new_tok` ignored. LAST stage returns `[vocab]` logits.
    /// MUST follow a `forward_pp_laguna_prefill` (which populated the caches).
    fn forward_pp_laguna_decode(
        &mut self,
        new_tok: u32,
        hidden_in: Vec<f32>,
    ) -> PyResult<Vec<f32>> {
        let g = self.laguna_gpu.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err(
                "forward_pp_laguna_decode needs a resident Laguna model (VLLM_VULKAN_LAGUNA_RESIDENT=1)",
            )
        })?;
        let fold = crate::flags::flags_global().laguna_1cb;
        if g.pp_first {
            Ok(if fold { g.forward_decode_token_1cb(new_tok) } else { g.forward_decode_token(new_tok) })
        } else {
            Ok(if fold { g.forward_decode_hidden_1cb(&hidden_in) } else { g.forward_decode_hidden(&hidden_in) })
        }
    }


    /// Last-stage twin of `forward_pp_laguna_decode` that argmaxes in Rust and
    /// returns just `(argmax_token, max_logit)` — the full `[vocab=100352]` logit
    /// Vec NEVER crosses the pyo3 boundary. The last rank's `Vec<f32>→PyList`
    /// vocab marshal + the driver's pure-Python `argmax` is per-token host
    /// overhead the GPU-resident decode otherwise pays every step (same rationale
    /// as `forward_pp_gemma_argmax`/`forward_pp_qwen35_argmax`, which Laguna never
    /// got). Runs the identical resident forward, then argmaxes with the same
    /// strict-`>` first-max tie-break as the driver's
    /// `max(range(len(v)), key=lambda i: v[i])` → byte-identical token. ONLY
    /// meaningful on the LAST PP stage (guarded on `pp_last`; mid/first stages
    /// emit a `[hidden]` vector, not logits — argmaxing that is meaningless). The
    /// full-logit `forward_pp_laguna_decode` stays for the DUMP / split-invariance
    /// path.
    fn forward_pp_laguna_decode_argmax(
        &mut self,
        new_tok: u32,
        hidden_in: Vec<f32>,
    ) -> PyResult<(u32, f32)> {
        {
            let g = self.laguna_gpu.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err(
                    "forward_pp_laguna_decode_argmax needs a resident Laguna model (VLLM_VULKAN_LAGUNA_RESIDENT=1)",
                )
            })?;
            if !g.pp_last {
                return Err(PyRuntimeError::new_err(
                    "forward_pp_laguna_decode_argmax is only valid on the last PP stage"));
            }
        }
        let out = self.forward_pp_laguna_decode(new_tok, hidden_in)?;
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in out.iter().enumerate() {
            if v > bv { bv = v; bi = i; }
        }
        Ok((bi as u32, bv))
    }


    /// FUSED native-vCCL PP DECODE step for Laguna (mirrors `pp_step_nemotron`/
    /// `pp_step_kimi`): recv the previous stage's `[H]` hidden (if not first) →
    /// resident single-token decode forward → send onward (native vCCL, no
    /// PyList) OR Rust argmax on the last stage. The `[H]` hidden and `[vocab]`
    /// logits NEVER cross the pyo3 boundary — only the token id in and
    /// `(argmax token, logit)` out on the last stage. Kills the per-hop
    /// `Vec<f32>→PyList`+`comm.send/recv`+`list(recv)` marshal tax the current
    /// `scripts/pp_laguna.py` `run_decode` pays ~5×/token.
    ///
    /// `recv_from < 0` ⇒ first stage (embeds `token_id`; `hidden_in` is empty);
    /// `send_to < 0` ⇒ last stage (returns `Some((tok, logit))`). MUST follow a
    /// cache-populating `forward_pp_laguna_prefill` (the decode reads the K/V
    /// caches). Honors `VLLM_VULKAN_LAGUNA_1CB` for the fold path exactly like
    /// `forward_pp_laguna_decode`. Requires `set_collective_comm` +
    /// `VLLM_VULKAN_NATIVE_COMM!=0`.
    fn pp_step_laguna(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
    ) -> PyResult<Option<(u32, f32)>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_laguna: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let h = self.laguna_gpu.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err(
                "pp_step_laguna needs a resident Laguna model (VLLM_VULKAN_LAGUNA_RESIDENT=1)"))?
            .config.hidden_size;
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let fold = crate::flags::flags_global().laguna_1cb;

        // Pin the persistent [H] PP-hop scratch (recv + send sides) with the RDMA
        // transport ONCE, so vCCL's send/recv skip the per-call `ibv_reg_mr`/dereg
        // temp-MR — the "buffer (H*4 B) not registered with the comm" warning.
        // Mirrors `pp_step_nemotron`; gated by `VLLM_VULKAN_REG_REDUCE` and libvccl
        // exposing `vcclCommRegister`; on failure we fall back to the fresh-Vec
        // recv_f32/send_f32 path (correct, just per-call regMr).
        let want_reg = self.flags.reg_reduce
            && !comm.is_null()
            && vccl_ffi::registration_available();
        if want_reg {
            let g = self.laguna_gpu.as_mut().unwrap();
            if do_recv && g.pp_recv_handle == 0 {
                g.pp_recv_scratch = vec![0.0f32; h];
                let addr = g.pp_recv_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => g.pp_recv_handle = hd,
                    Err(e) => {
                        log::warn!("pp_step_laguna: register recv scratch failed: {e}; per-call regMr");
                        g.pp_recv_scratch.clear();
                    }
                }
            }
            if !is_last && g.pp_send_handle == 0 {
                g.pp_send_scratch = vec![0.0f32; h];
                let addr = g.pp_send_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => g.pp_send_handle = hd,
                    Err(e) => {
                        log::warn!("pp_step_laguna: register send scratch failed: {e}; per-call regMr");
                        g.pp_send_scratch.clear();
                    }
                }
            }
        }

        // 1) recv the previous stage's [H] hidden INTO the registered scratch (fast
        //    pre-pinned MR), or empty on the first stage (it embeds token_id).
        let hidden_in: Vec<f32> = if do_recv {
            let g = self.laguna_gpu.as_mut().unwrap();
            if g.pp_recv_handle != 0 {
                vccl_ffi::recv_f32_into(py, comm, &mut g.pp_recv_scratch, recv_from)
                    .map_err(PyRuntimeError::new_err)?;
                g.pp_recv_scratch.clone()
            } else {
                vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
            }
        } else {
            Vec::new()
        };

        // 2) resident single-token decode forward (holds the GIL — pure compute,
        //    same as forward_pp_laguna_decode). [H] on mid stages, [vocab] on last.
        let g = self.laguna_gpu.as_mut().unwrap();
        let out = if g.pp_first {
            if fold { g.forward_decode_token_1cb(token_id) } else { g.forward_decode_token(token_id) }
        } else if fold {
            g.forward_decode_hidden_1cb(&hidden_in)
        } else {
            g.forward_decode_hidden(&hidden_in)
        };

        // 3) send onward FROM the registered scratch (mid/first stage out is [H]);
        //    fall back to a fresh-Vec send if registration is off or width differs.
        if !is_last {
            let g = self.laguna_gpu.as_mut().unwrap();
            if g.pp_send_handle != 0 && out.len() == g.pp_send_scratch.len() {
                g.pp_send_scratch.copy_from_slice(&out);
                vccl_ffi::send_f32(py, comm, &g.pp_send_scratch, send_to)
                    .map_err(PyRuntimeError::new_err)?;
            } else {
                vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            }
            Ok(None)
        } else {
            // Strict-`>` first-max tie-break, identical to the driver's python argmax.
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in out.iter().enumerate() {
                if v > bv { bv = v; bi = i; }
            }
            Ok(Some((bi as u32, bv)))
        }
    }


    /// SERVING twin of `pp_step_laguna` (Phase-3 distributed OpenAI serving). Same
    /// fused recv→resident-decode→send-onward hop, EXCEPT the last stage does NOT
    /// argmax: it computes the full `[vocab]` and **rings it back to rank0 over
    /// vCCL** (raw f32, `send_f32`/`recv_f32_into` — NO per-hop PyList marshal),
    /// and rank0 (the first stage) recv's the `[vocab]` after sending its `[H]`
    /// hidden onward. Returns `Some([vocab])` on rank0 (the vLLM head, which needs
    /// the full logit vector for its `Sampler`: temperature/top-k/top-p/penalties/
    /// logprobs/structured-output), `None` on every peer stage. This is the
    /// load-bearing serving gap: the argmax fusion (`pp_step_laguna`) is exactly
    /// wrong for vLLM, which cannot sample from just `(tok, logit)`.
    ///
    /// Wiring (driven by rank0):
    ///  - rank0 / FIRST stage (`recv_from<0`, `send_to>=0`): embed `token_id` →
    ///    decode → send `[H]` to `send_to`; then recv `[vocab]` from `last_rank`;
    ///    return `Some([vocab])`.
    ///  - MID stage (`recv_from>=0`, `send_to>=0`): recv `[H]` → decode → send
    ///    `[H]` onward; return `None`.
    ///  - LAST stage (`recv_from>=0`, `send_to<0`): recv `[H]` → decode → `[vocab]`;
    ///    send `[vocab]` to rank0 (peer 0); return `None`.
    ///  - STANDALONE N=1 (`recv_from<0` && `send_to<0`): embed → decode → `[vocab]`;
    ///    return `Some([vocab])` with no wire.
    ///
    /// `last_rank` = the last PP stage's rank (rank0 recv's the vocab from it; peers
    /// ignore it). MUST follow a cache-populating `forward_pp_laguna_prefill`.
    /// Requires `set_collective_comm` + `VLLM_VULKAN_NATIVE_COMM!=0`. The vocab
    /// ring-back scratch is `comm_register`'d ONCE (`pp_vocab_scratch`, 401 KB) so
    /// the per-token hop skips the temp-MR. Bit-exact with
    /// `forward_pp_laguna_decode`'s `[vocab]` (same resident forward; only the
    /// transport of the result differs).
    fn pp_step_laguna_logits(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
    ) -> PyResult<Option<Vec<f32>>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_laguna_logits: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let (h, vocab) = {
            let g = self.laguna_gpu.as_ref().ok_or_else(|| PyRuntimeError::new_err(
                "pp_step_laguna_logits needs a resident Laguna model (VLLM_VULKAN_LAGUNA_RESIDENT=1)"))?;
            (g.config.hidden_size, g.config.vocab_size)
        };
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;
        let fold = crate::flags::flags_global().laguna_1cb;

        // Pin the persistent scratches ONCE (skip per-call `ibv_reg_mr`/dereg):
        //  - `[H]` recv scratch on any stage that recvs a hidden (mid/last),
        //  - `[H]` send scratch on any stage that forwards a hidden (first/mid),
        //  - `[vocab]` scratch on rank0 (recv's the ring-back) and the last stage
        //    (sends the ring-back). Same fall-back-to-fresh-Vec-on-failure contract
        //    as `pp_step_laguna`.
        let want_reg = self.flags.reg_reduce
            && !comm.is_null()
            && vccl_ffi::registration_available();
        if want_reg {
            let g = self.laguna_gpu.as_mut().unwrap();
            if do_recv && g.pp_recv_handle == 0 {
                g.pp_recv_scratch = vec![0.0f32; h];
                let addr = g.pp_recv_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => g.pp_recv_handle = hd,
                    Err(e) => { log::warn!("pp_step_laguna_logits: reg recv scratch failed: {e}; per-call regMr"); g.pp_recv_scratch.clear(); }
                }
            }
            if !is_last && g.pp_send_handle == 0 {
                g.pp_send_scratch = vec![0.0f32; h];
                let addr = g.pp_send_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => g.pp_send_handle = hd,
                    Err(e) => { log::warn!("pp_step_laguna_logits: reg send scratch failed: {e}; per-call regMr"); g.pp_send_scratch.clear(); }
                }
            }
            // vocab scratch: rank0 recv's it, the last stage sends it. (N=1 needs
            // neither — no wire.) Pin on whichever role this rank plays.
            let needs_vocab = (is_first && !is_last) || (is_last && !is_first);
            if needs_vocab && g.pp_vocab_handle == 0 {
                g.pp_vocab_scratch = vec![0.0f32; vocab];
                let addr = g.pp_vocab_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, vocab * std::mem::size_of::<f32>()) {
                    Ok(hd) => g.pp_vocab_handle = hd,
                    Err(e) => { log::warn!("pp_step_laguna_logits: reg vocab scratch failed: {e}; per-call regMr"); g.pp_vocab_scratch.clear(); }
                }
            }
        }

        // 1) recv the previous stage's [H] hidden INTO the registered scratch.
        let hidden_in: Vec<f32> = if do_recv {
            let g = self.laguna_gpu.as_mut().unwrap();
            if g.pp_recv_handle != 0 {
                vccl_ffi::recv_f32_into(py, comm, &mut g.pp_recv_scratch, recv_from)
                    .map_err(PyRuntimeError::new_err)?;
                g.pp_recv_scratch.clone()
            } else {
                vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
            }
        } else {
            Vec::new()
        };

        // 2) resident single-token decode (holds the GIL). [H] mid, [vocab] last.
        let g = self.laguna_gpu.as_mut().unwrap();
        let out = if g.pp_first {
            if fold { g.forward_decode_token_1cb(token_id) } else { g.forward_decode_token(token_id) }
        } else if fold {
            g.forward_decode_hidden_1cb(&hidden_in)
        } else {
            g.forward_decode_hidden(&hidden_in)
        };

        // 3) route the result.
        if is_first && is_last {
            // STANDALONE N=1: rank0 is both first and last; `out` is already [vocab].
            return Ok(Some(out));
        }
        if !is_last {
            // FIRST / MID: forward `[H]` onward, then (rank0 only) recv the ring-back.
            let g = self.laguna_gpu.as_mut().unwrap();
            if g.pp_send_handle != 0 && out.len() == g.pp_send_scratch.len() {
                g.pp_send_scratch.copy_from_slice(&out);
                vccl_ffi::send_f32(py, comm, &g.pp_send_scratch, send_to)
                    .map_err(PyRuntimeError::new_err)?;
            } else {
                vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            }
            if is_first {
                // rank0: ring the [vocab] back from the last stage into vocab scratch.
                let g = self.laguna_gpu.as_mut().unwrap();
                let logits = if g.pp_vocab_handle != 0 {
                    vccl_ffi::recv_f32_into(py, comm, &mut g.pp_vocab_scratch, last_rank)
                        .map_err(PyRuntimeError::new_err)?;
                    g.pp_vocab_scratch.clone()
                } else {
                    vccl_ffi::recv_f32(py, comm, vocab, last_rank).map_err(PyRuntimeError::new_err)?
                };
                Ok(Some(logits))
            } else {
                Ok(None)
            }
        } else {
            // LAST stage: ring the full [vocab] back to rank0 (peer 0). No argmax.
            let g = self.laguna_gpu.as_mut().unwrap();
            if g.pp_vocab_handle != 0 && out.len() == g.pp_vocab_scratch.len() {
                g.pp_vocab_scratch.copy_from_slice(&out);
                vccl_ffi::send_f32(py, comm, &g.pp_vocab_scratch, 0)
                    .map_err(PyRuntimeError::new_err)?;
            } else {
                vccl_ffi::send_f32(py, comm, &out, 0).map_err(PyRuntimeError::new_err)?;
            }
            Ok(None)
        }
    }


    /// FAST-SAMPLER twin of `pp_step_laguna_logits` (Phase-3 distributed OpenAI
    /// serving). Same fused recv→resident-decode→send-onward hop, EXCEPT the last
    /// stage does the **top-`k` selection IN RUST** over its `[vocab]` logits and
    /// rings back only the `k` winning `(logit, index)` pairs to rank0 — NOT the
    /// whole `[vocab]`. This is the load-bearing serve-path speedup: the full
    /// `[vocab=100352]` ring-back (`pp_step_laguna_logits`) forces rank0 to
    /// marshal ~401 KB across the pyo3 boundary AND run a Python full-vocab
    /// argmax/sample EVERY token — the same ~76 ms/tok Python-vocab tax that put
    /// the serve path at ~128 ms/tok vs the offline 51.9 (whose `pp_step_laguna`
    /// argmaxes in Rust and rings 1 token). Ringing `2*k` f32 (k=64 → 512 B)
    /// instead of `vocab` f32 drops the wire ~780× and the pyo3 marshal from
    /// 100 352 floats to `k` tuples, so rank0's per-token host cost collapses to
    /// the offline argmax class. rank0 then hands the `k` candidates to the
    /// sampler (greedy = argmax of the top-k list = the EXACT global argmax, since
    /// the max logit is always in the top-k; temperature/top-p/top-k over the
    /// candidate set is correct whenever the nucleus ⊆ the k candidates). Requests
    /// that need the WHOLE distribution (repetition/presence/frequency penalties,
    /// min_p, full logprobs, logit_bias) must fall back to `pp_step_laguna_logits`
    /// (rank0 routes per-request on the sampling params).
    ///
    /// Wire format of the ring-back: `[k logits][k indices]` (`2*k` f32). The
    /// index is encoded as an f32 — `vocab < 2^24` so an integer token id
    /// round-trips through f32 exactly. Same strict-`>` earliest-index-wins
    /// tie-break as `forward_topk`/`pp_step_laguna` (byte-identical winner).
    ///
    /// Wiring (driven by rank0), identical to `_logits` except the ring-back is
    /// `[2*k]` not `[vocab]`:
    ///  - rank0 / FIRST (`recv_from<0`, `send_to>=0`): embed → decode → send `[H]`
    ///    → recv `[2*k]` from `last_rank` → return `Some(Vec<(tok,logit)>)`.
    ///  - MID (`recv_from>=0`, `send_to>=0`): recv `[H]` → decode → send `[H]`;
    ///    return `None`.
    ///  - LAST (`recv_from>=0`, `send_to<0`): recv `[H]` → decode → `[vocab]` →
    ///    top-k → send `[2*k]` to rank0; return `None`.
    ///  - STANDALONE N=1 (`recv_from<0` && `send_to<0`): embed → decode → top-k;
    ///    return `Some(Vec<(tok,logit)>)` with no wire.
    ///
    /// MUST follow a cache-populating `forward_pp_laguna_prefill`. Requires
    /// `set_collective_comm` + `VLLM_VULKAN_NATIVE_COMM!=0`. Bit-exact with a
    /// `forward_pp_laguna_decode` + Python top-k (same resident forward; only the
    /// transport + the WHERE of the selection differ).
    fn pp_step_laguna_topk(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
        k: usize,
    ) -> PyResult<Option<Vec<(u32, f32)>>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_laguna_topk: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        if k == 0 {
            return Err(PyRuntimeError::new_err("pp_step_laguna_topk: k must be >= 1"));
        }
        let h = {
            let g = self.laguna_gpu.as_ref().ok_or_else(|| PyRuntimeError::new_err(
                "pp_step_laguna_topk needs a resident Laguna model (VLLM_VULKAN_LAGUNA_RESIDENT=1)"))?;
            g.config.hidden_size
        };
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;
        let fold = crate::flags::flags_global().laguna_1cb;
        let topk_len = 2 * k; // [k logits][k indices-as-f32]

        // Pin the persistent scratches ONCE (skip per-call `ibv_reg_mr`/dereg):
        //  - `[H]` recv scratch on any stage that recvs a hidden (mid/last),
        //  - `[H]` send scratch on any stage that forwards a hidden (first/mid),
        //  - `[2*k]` top-k scratch on rank0 (recv's the ring-back) and the last
        //    stage (sends the ring-back). Same fall-back-to-fresh-Vec-on-failure
        //    contract as `pp_step_laguna_logits`. Re-pin if k changed (len differs).
        let want_reg = self.flags.reg_reduce
            && !comm.is_null()
            && vccl_ffi::registration_available();
        if want_reg {
            let g = self.laguna_gpu.as_mut().unwrap();
            if do_recv && g.pp_recv_handle == 0 {
                g.pp_recv_scratch = vec![0.0f32; h];
                let addr = g.pp_recv_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => g.pp_recv_handle = hd,
                    Err(e) => { log::warn!("pp_step_laguna_topk: reg recv scratch failed: {e}; per-call regMr"); g.pp_recv_scratch.clear(); }
                }
            }
            if !is_last && g.pp_send_handle == 0 {
                g.pp_send_scratch = vec![0.0f32; h];
                let addr = g.pp_send_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => g.pp_send_handle = hd,
                    Err(e) => { log::warn!("pp_step_laguna_topk: reg send scratch failed: {e}; per-call regMr"); g.pp_send_scratch.clear(); }
                }
            }
            // top-k scratch: rank0 recv's it, the last stage sends it. (N=1 needs
            // neither — no wire.) Re-pin when k (hence `2*k`) changes.
            let needs_topk = (is_first && !is_last) || (is_last && !is_first);
            if needs_topk && g.pp_topk_scratch.len() != topk_len {
                if g.pp_topk_handle != 0 {
                    let _ = vccl_ffi::comm_deregister(comm, g.pp_topk_handle);
                    g.pp_topk_handle = 0;
                }
                g.pp_topk_scratch = vec![0.0f32; topk_len];
                let addr = g.pp_topk_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, topk_len * std::mem::size_of::<f32>()) {
                    Ok(hd) => g.pp_topk_handle = hd,
                    Err(e) => { log::warn!("pp_step_laguna_topk: reg topk scratch failed: {e}; per-call regMr"); g.pp_topk_scratch.clear(); }
                }
            }
        }

        // 1) recv the previous stage's [H] hidden INTO the registered scratch.
        let hidden_in: Vec<f32> = if do_recv {
            let g = self.laguna_gpu.as_mut().unwrap();
            if g.pp_recv_handle != 0 {
                vccl_ffi::recv_f32_into(py, comm, &mut g.pp_recv_scratch, recv_from)
                    .map_err(PyRuntimeError::new_err)?;
                g.pp_recv_scratch.clone()
            } else {
                vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
            }
        } else {
            Vec::new()
        };

        // 2) resident single-token decode (holds the GIL). [H] mid, [vocab] last.
        let g = self.laguna_gpu.as_mut().unwrap();
        let out = if g.pp_first {
            if fold { g.forward_decode_token_1cb(token_id) } else { g.forward_decode_token(token_id) }
        } else if fold {
            g.forward_decode_hidden_1cb(&hidden_in)
        } else {
            g.forward_decode_hidden(&hidden_in)
        };

        // 3) route the result.
        if is_first && is_last {
            // STANDALONE N=1: rank0 is both first and last; top-k `out` locally.
            return Ok(Some(topk_select(&out, k)));
        }
        if !is_last {
            // FIRST / MID: forward `[H]` onward, then (rank0 only) recv the ring-back.
            let g = self.laguna_gpu.as_mut().unwrap();
            if g.pp_send_handle != 0 && out.len() == g.pp_send_scratch.len() {
                g.pp_send_scratch.copy_from_slice(&out);
                vccl_ffi::send_f32(py, comm, &g.pp_send_scratch, send_to)
                    .map_err(PyRuntimeError::new_err)?;
            } else {
                vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            }
            if is_first {
                // rank0: ring the [2*k] top-k pairs back from the last stage.
                let g = self.laguna_gpu.as_mut().unwrap();
                let packed = if g.pp_topk_handle != 0 && g.pp_topk_scratch.len() == topk_len {
                    vccl_ffi::recv_f32_into(py, comm, &mut g.pp_topk_scratch, last_rank)
                        .map_err(PyRuntimeError::new_err)?;
                    g.pp_topk_scratch.clone()
                } else {
                    vccl_ffi::recv_f32(py, comm, topk_len, last_rank).map_err(PyRuntimeError::new_err)?
                };
                // Unpack [k logits][k indices]; index encoded as exact f32.
                let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
                for i in 0..k {
                    let logit = packed[i];
                    let idx = packed[k + i].round() as u32;
                    top.push((idx, logit));
                }
                Ok(Some(top))
            } else {
                Ok(None)
            }
        } else {
            // LAST stage: top-k select over [vocab], pack [k logits][k indices],
            // ring `[2*k]` to rank0 (peer 0). No full-vocab marshal.
            let top = topk_select(&out, k);
            let mut packed = vec![0.0f32; topk_len];
            for (i, &(idx, logit)) in top.iter().enumerate() {
                packed[i] = logit;
                packed[k + i] = idx as f32;
            }
            let g = self.laguna_gpu.as_mut().unwrap();
            if g.pp_topk_handle != 0 && g.pp_topk_scratch.len() == topk_len {
                g.pp_topk_scratch.copy_from_slice(&packed);
                vccl_ffi::send_f32(py, comm, &g.pp_topk_scratch, 0)
                    .map_err(PyRuntimeError::new_err)?;
            } else {
                vccl_ffi::send_f32(py, comm, &packed, 0).map_err(PyRuntimeError::new_err)?;
            }
            Ok(None)
        }
    }


}
