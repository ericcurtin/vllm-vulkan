// SPDX-License-Identifier: Apache-2.0
//! GPU-resident FORWARD for **Laguna-S-2.1-NVFP4** (`model_type == "laguna"`).
//!
//! The runtime sibling of the pure-CPU reference in [`crate::laguna`]
//! (`LagunaWeights::forward`, HF-verified bit-exact) and of the GPU-resident
//! LOADER in [`crate::laguna_loader`] (`load_laguna_resident`). It mirrors the
//! Nemotron resident forward (`nemotron::NemotronModel::forward_pp_stage` /
//! `forward_pp_range`): a `[pp_start, pp_end)` layer window kept quantized-
//! resident on the GPU, one host vector in / out per stage.
//!
//! ## Correct-by-construction bit-exactness (the same discipline nemotron/qwen
//!    use to gate GPU forwards against a CPU oracle)
//!  - Every MATMUL goes through [`LagunaGpuModel::f16_matvec`] (BF16-native
//!    attn/dense/shared/lm/embed) or [`LagunaGpuModel::expert_matvec`] (NVFP4
//!    routed experts). Both dispatch the resident GPU kernel when an engine +
//!    resident weight are present, else fall back to the EXACT `cpu_matmul` on
//!    the host f32 copy — so with `engine=None` the forward is bit-identical to
//!    the CPU oracle (the property the Mac tests rely on).
//!  - Every NON-matmul op reuses the IDENTICAL crate fn the oracle calls:
//!    `cpu_rms_norm`/`cpu_rms_norm_inplace` (norms + q/k head-norm),
//!    `cpu_rope_yarn` (full-attn YaRN) / `cpu_rope` (sliding plain rope),
//!    `cpu_sdpa` (causal + `Some(512)` sliding-window mask), `softplus` (the
//!    per-head scalar attn gate), `router_forward` (sigmoid + e_score bias
//!    top-k). So the ONLY GPU-specific numerics are the two matvec shaders
//!    (`mul_mat_vec_*_f16`, `mul_mat_vec_nvfp4_e4m3`), both already cluster-
//!    validated (nemotron `nvfp4_e4m3_resident_matches_f32_fold`; the Laguna
//!    resident expert LAYOUT is bit-exact vs `dequantize_nvfp4` on real bytes —
//!    `laguna_loader::resident_expert_layout_matches_oracle_dequant`).
//!
//! ## Dispatch is reimplemented here (not called into nemotron)
//! Nemotron's `nem_rec_mv` / `nem_expert_matvec` / `NemMvKind` are PRIVATE to
//! that module, so this module dispatches the SAME `pub(crate)` push-constant
//! builders directly (`matvec_f16_variant_k`+`matvec_pc13`,
//! `matvec_nvfp4_e4m3_variant`+`matvec_nvfp4_e4m3_pc_off`). Identical shaders,
//! identical push-constants, identical per-expert `packed_off`/`sb_off`
//! offsets ⇒ bitwise-identical output.

use std::collections::HashMap;

use crate::compute::{Buffer, ComputeEngine};
use crate::laguna::{cpu_rope_yarn, LagunaConfig};
use crate::laguna_loader::{load_laguna_resident, LagunaMoeExperts};
use crate::model::{cpu_matmul, cpu_rms_norm, cpu_rms_norm_inplace, cpu_rope, cpu_sdpa, KvCache, SimpleTensor};
use crate::nemotron::{router_forward, NemGpuWeight, NemQuant};
use crate::push_constants::{
    f32_slice_to_bytes, glu_split_pc, laguna_expb_e4m3_pc, laguna_expb_e4m3_variant,
    laguna_expert_repack_flag, laguna_gpu_sdpa_pc, laguna_moe_accum_pc, laguna_router_pc,
    laguna_softplus_gate_pc, matvec_f16_variant_k, matvec_mlx4_pc_off, matvec_nvfp4_e4m3_pc_off,
    matvec_nvfp4_e4m3_variant, matvec_nvfp4_variant_k, matvec_pc13, matvec_q8_0_variant_k,
    nvfp4_repack_shape_ok, read_f32_buf, rmsnorm_pc, rope_neox_pc, rope_neox_yarn_pc,
};

/// Pick the routed-expert e4m3 NVFP4 matvec shader + rows-per-workgroup. When
/// `VLLM_VULKAN_LAGUNA_EXPERT_REPACK` is on AND the shape clears the repack guard
/// (`nvfp4_repack_shape_ok` — Laguna gate/up k=3072→n=1024 and down k=1024→n=3072,
/// gs=16 all pass), route to the address-gen-free REPACK kernel
/// (`mul_mat_vec_nvfp4_e4m3repack_f32_f32_bs64_r4`, bs64/r4 = the mlx4/nvfp4-repack
/// default); else the v1 `mul_mat_vec_nvfp4_e4m3` oracle. The repack shader threads
/// `packed_off`/`sb_off` identically (push block + base4/sbase math), so the
/// per-expert slice offsets pass straight through `matvec_nvfp4_e4m3_pc_off` — NO
/// push-constant change. argmax-exact vs v1 (repack == f32-fold, single IEEE mul).
fn laguna_e4m3_expert_shader(k: usize, n: usize, gs: usize) -> (String, u32) {
    if laguna_expert_repack_flag() && nvfp4_repack_shape_ok(k, n, gs) {
        return ("mul_mat_vec_nvfp4_e4m3repack_f32_f32_bs64_r4".to_string(), 4);
    }
    matvec_nvfp4_e4m3_variant(n)
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
#[inline]
fn softplus(x: f32) -> f32 {
    // torch.nn.functional.softplus, numerically stable (matches laguna.rs).
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Device-resident post-rope K/V planes for ONE attention layer — the arena the
/// GPU span fold consumes (PIECE 3 of 3). On the BC-250's UMA fabric a
/// host-coherent buffer is SIMULTANEOUSLY device-resident and host-mapped, so a
/// plane is a persistent GPU `Buffer` (allocated once at load, lives across every
/// decode step) whose post-rope rows are appended in place each step: the host
/// write to the mapped coherent memory IS the device write (zero-copy), and SDPA
/// reads the same memory with NO round-trip. Both Laguna attn types share this
/// storage — with Phase-0 per-layer KV sizing the **sliding-window** layers now
/// use a `window`-sized RING (`capacity == sliding_window`): absolute position
/// `p` lives at slot `p % capacity`, so the plane holds only the last `capacity`
/// positions — exactly what `cpu_sdpa`'s sliding clamp
/// (`kv_start = seq_len.saturating_sub(window)`) ever reads. Full-attention
/// (YaRN) layers keep FULL storage (`capacity == max_seq`, slot == absolute
/// position, `ring_capacity == 0` on the shader side → byte-for-byte the pre-ring
/// path). Mirrors `Gemma4` `KvCache::new_windowed` / the `paged_attn_decode_f32`
/// `ring_capacity` push-constant. The layout matches `KvCache` exactly:
/// `[capacity, nkv, hd]` f32, row stride `nkv*hd`.
pub struct ResidentKvPlane {
    k: Buffer,
    v: Buffer,
    /// Valid positions appended so far (ABSOLUTE, monotonically grows to
    /// `max_seq`). Unchanged by windowing — still the token count, NOT a ring
    /// slot index (== the host `KvCache::seq_len`).
    pub seq_len: usize,
    /// Logical position bound (overflow guard on `seq_len`). Independent of the
    /// physical `capacity`.
    max_seq: usize,
    /// Physical number of position-slots actually allocated. Equals `max_seq`
    /// for full layers, `sliding_window` for windowed sliding layers. Invariant:
    /// `1 <= capacity <= max_seq`. When `capacity < max_seq` the plane is a ring
    /// (`append`/`windowed_view` resolve slot `p % capacity`).
    capacity: usize,
    /// Elements per token row (`num_kv_heads * head_dim`).
    stride: usize,
}

impl ResidentKvPlane {
    /// Allocate the two persistent host-coherent K/V plane buffers for one layer.
    /// `capacity` is the physical slot count (`max_seq` for a full/global layer,
    /// `window` for a windowed sliding layer — see
    /// `LagunaConfig::layer_kv_capacity`); `max_seq` is the logical position
    /// bound. Requires the engine (the buffers are the GPU-resident arena; a
    /// `None`-engine model uses the host `KvCache` instead).
    fn new(engine: &mut ComputeEngine, max_seq: usize, capacity: usize, nkv: usize, hd: usize) -> Result<Self, String> {
        assert!(capacity >= 1, "resident KV plane capacity must be >= 1");
        assert!(capacity <= max_seq, "resident KV plane capacity {capacity} > max_seq {max_seq}");
        let stride = nkv * hd;
        let bytes = (capacity * stride * 4) as u64;
        let k = engine.alloc_host_coherent_storage(bytes)?;
        let v = engine.alloc_host_coherent_storage(bytes)?;
        Ok(Self { k, v, seq_len: 0, max_seq, capacity, stride })
    }

    /// Reset to a fresh sequence (overwrite-in-place storage → just the counter).
    #[inline]
    fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Shader/reader `ring_capacity`: `capacity` when the plane is a shrunk ring
    /// (`capacity < max_seq`, sliding-window layer), else 0 (full absolute plane —
    /// slot == absolute position, byte-identical legacy addressing).
    #[inline]
    fn ring_capacity(&self) -> usize {
        if self.capacity < self.max_seq { self.capacity } else { 0 }
    }

    /// True once absolute `seq_len` has exceeded `capacity`, i.e. the ring has
    /// wrapped and slot index no longer equals absolute position. Mirrors
    /// `KvCache::has_wrapped`.
    #[inline]
    fn has_wrapped(&self) -> bool {
        self.seq_len > self.capacity
    }

    /// Append one token's post-rope K and V rows into the resident planes. For a
    /// full plane (`capacity == max_seq`) this writes absolute slot `seq_len`
    /// (byte-identical to the pre-ring path). For a windowed sliding plane it
    /// writes ring slot `seq_len % capacity`, overwriting the position that just
    /// fell out of the window. The host write targets the mapped COHERENT buffer
    /// — i.e. the device-resident plane — so on UMA this IS the on-GPU append.
    fn append(&mut self, k_row: &[f32], v_row: &[f32]) {
        assert!(self.seq_len < self.max_seq, "resident KV plane overflow");
        debug_assert_eq!(k_row.len(), self.stride, "resident KV plane: K row stride mismatch");
        debug_assert_eq!(v_row.len(), self.stride, "resident KV plane: V row stride mismatch");
        let slot = self.seq_len % self.capacity;
        let off = slot * self.stride * 4;
        let kb = f32_slice_to_bytes(k_row);
        let vb = f32_slice_to_bytes(v_row);
        // SAFETY: single engine-owning thread; slot < capacity so off+len is
        // within the plane, and the mapped region is the full buffer size.
        unsafe {
            let kp = self.k.mapped_ptr.expect("resident K plane mapped");
            let vp = self.v.mapped_ptr.expect("resident V plane mapped");
            std::ptr::copy_nonoverlapping(kb.as_ptr(), kp.add(off), kb.len());
            std::ptr::copy_nonoverlapping(vb.as_ptr(), vp.add(off), vb.len());
        }
        self.seq_len += 1;
    }

    /// K rows `[0, seq_len)` as a contiguous f32 slice viewed straight from the
    /// resident plane's mapped memory — NO readback (the exact bytes a GPU SDPA
    /// would read). Mirrors `KvCache::k_up_to_now`. Only valid on a NOT-wrapped
    /// plane (full layers, or sliding layers still within the first `capacity`
    /// positions); a wrapped ring must be read via [`windowed_view`].
    fn k_up_to_now(&self) -> &[f32] {
        debug_assert!(!self.has_wrapped(),
            "k_up_to_now() on a wrapped resident ring plane (seq_len {} > capacity {}); use windowed_view()",
            self.seq_len, self.capacity);
        let n = self.seq_len * self.stride;
        // SAFETY: mapped_ptr is a valid, permanently-mapped f32 region of at
        // least capacity*stride elements; n <= that (not wrapped); read-only
        // borrow tied to &self.
        unsafe {
            std::slice::from_raw_parts(
                self.k.mapped_ptr.expect("resident K plane mapped") as *const f32,
                n,
            )
        }
    }

    /// Contiguous, ascending-absolute-position view of the last `window`
    /// positions (or all `seq_len` positions if `seq_len <= window`), copied out
    /// of the ring into `[valid_len, nkv, hd]` row order — ready to feed
    /// `cpu_sdpa`/`cpu_sdpa_gqa` with `sliding_window = None` and
    /// `seq_len = valid_len`. Because attention math depends only on the SET of
    /// K/V rows and their ascending order (never the absolute index), this is
    /// bit-for-bit identical to attending a full-size plane with
    /// `Some(window)` — the correctness contract that makes ring sizing exact
    /// (see `KvCache::windowed_view`). Returns `(k, v, valid_len)`.
    fn windowed_view(&self, window: usize) -> (Vec<f32>, Vec<f32>, usize) {
        let stride = self.stride;
        // SAFETY: the mapped planes hold at least capacity*stride f32; the gather
        // below only reads slots `< capacity`.
        let (kp, vp) = unsafe {
            (
                std::slice::from_raw_parts(self.k.mapped_ptr.expect("resident K plane mapped") as *const f32, self.capacity * stride),
                std::slice::from_raw_parts(self.v.mapped_ptr.expect("resident V plane mapped") as *const f32, self.capacity * stride),
            )
        };
        ring_windowed_gather(kp, vp, self.capacity, stride, self.seq_len, window)
    }

    /// The resident K plane `Buffer` (device-resident, host-coherent). The GPU
    /// decode-SDPA (`laguna_gpu_sdpa`) reads this directly — the exact bytes
    /// `k_up_to_now` views, no readback.
    fn k_buf(&self) -> &Buffer {
        &self.k
    }

    /// The resident V plane `Buffer`. See [`k_buf`].
    fn v_buf(&self) -> &Buffer {
        &self.v
    }

    /// V counterpart of [`k_up_to_now`].
    fn v_up_to_now(&self) -> &[f32] {
        debug_assert!(!self.has_wrapped(),
            "v_up_to_now() on a wrapped resident ring plane (seq_len {} > capacity {}); use windowed_view()",
            self.seq_len, self.capacity);
        let n = self.seq_len * self.stride;
        // SAFETY: see k_up_to_now.
        unsafe {
            std::slice::from_raw_parts(
                self.v.mapped_ptr.expect("resident V plane mapped") as *const f32,
                n,
            )
        }
    }

    /// NAS-prefix tile export: gather head `kv_head`'s `[head_dim]` columns for
    /// the absolute-position rows `[base, base+n_rows)` (ring slot `abs %
    /// capacity`) straight off the mapped planes — a plain memcpy, NO device
    /// readback (host-coherent UMA memory, `laguna_gpu.rs` module doc). Returns
    /// `(k, v)`, each `[n_rows * head_dim]`.
    fn export_head(&self, kv_head: usize, head_dim: usize, base: usize, n_rows: usize) -> (Vec<f32>, Vec<f32>) {
        // SAFETY: mapped_ptr is a valid permanently-mapped f32 region of
        // capacity*stride elements; every slot read below is < capacity.
        let (kp, vp) = unsafe {
            (
                std::slice::from_raw_parts(self.k.mapped_ptr.expect("resident K plane mapped") as *const f32, self.capacity * self.stride),
                std::slice::from_raw_parts(self.v.mapped_ptr.expect("resident V plane mapped") as *const f32, self.capacity * self.stride),
            )
        };
        let mut k = vec![0.0f32; n_rows * head_dim];
        let mut v = vec![0.0f32; n_rows * head_dim];
        for i in 0..n_rows {
            let slot = (base + i) % self.capacity;
            let src = slot * self.stride + kv_head * head_dim;
            k[i * head_dim..(i + 1) * head_dim].copy_from_slice(&kp[src..src + head_dim]);
            v[i * head_dim..(i + 1) * head_dim].copy_from_slice(&vp[src..src + head_dim]);
        }
        (k, v)
    }

    /// NAS-prefix tile import: write head `kv_head`'s `[head_dim]` columns for
    /// the absolute rows `[window_base, window_base+n_rows)` back into the mapped
    /// planes (ring slot `abs % capacity`). The host write to coherent memory IS
    /// the on-GPU write (UMA), so the resident 1-CB path sees the restored KV.
    fn import_head(&mut self, kv_head: usize, head_dim: usize, window_base: usize, k: &[f32], v: &[f32]) {
        let n_rows = k.len() / head_dim;
        // SAFETY: single engine-owning thread; every slot write below is <
        // capacity, within the full mapped buffer.
        let (kp, vp) = unsafe {
            (
                std::slice::from_raw_parts_mut(self.k.mapped_ptr.expect("resident K plane mapped") as *mut f32, self.capacity * self.stride),
                std::slice::from_raw_parts_mut(self.v.mapped_ptr.expect("resident V plane mapped") as *mut f32, self.capacity * self.stride),
            )
        };
        for i in 0..n_rows {
            let slot = (window_base + i) % self.capacity;
            let dst = slot * self.stride + kv_head * head_dim;
            kp[dst..dst + head_dim].copy_from_slice(&k[i * head_dim..(i + 1) * head_dim]);
            vp[dst..dst + head_dim].copy_from_slice(&v[i * head_dim..(i + 1) * head_dim]);
        }
    }
}

/// Core of `ResidentKvPlane::windowed_view` and the `laguna_gpu_sdpa` ring shader:
/// compact the last `window` positions of a ring plane into ascending-absolute-
/// position `[valid_len, stride]` row order. `kbuf`/`vbuf` are the ring planes
/// (`capacity * stride` f32 each); absolute position `p` in `[kv_start, seq_len)`
/// lives at slot `p % capacity` (`kv_start = seq_len.saturating_sub(window)`).
/// Returns `(k, v, valid_len)`. Because attention math depends only on the SET of
/// rows and their ascending order (never the absolute index), feeding this to
/// `cpu_sdpa` with `sliding_window = None, seq_len = valid_len` is bit-for-bit
/// identical to a full absolute plane with `Some(window)` — the Phase-0 ring
/// correctness contract. Pure host math → the offline-testable heart of the ring;
/// the on-node `laguna_gpu_sdpa` shader reads the exact same `p % capacity` slots.
fn ring_windowed_gather(
    kbuf: &[f32],
    vbuf: &[f32],
    capacity: usize,
    stride: usize,
    seq_len: usize,
    window: usize,
) -> (Vec<f32>, Vec<f32>, usize) {
    let kv_start = seq_len.saturating_sub(window);
    let valid_len = seq_len - kv_start;
    let mut k = vec![0.0f32; valid_len * stride];
    let mut v = vec![0.0f32; valid_len * stride];
    for i in 0..valid_len {
        let abs = kv_start + i;
        let slot = abs % capacity;
        k[i * stride..(i + 1) * stride].copy_from_slice(&kbuf[slot * stride..(slot + 1) * stride]);
        v[i * stride..(i + 1) * stride].copy_from_slice(&vbuf[slot * stride..(slot + 1) * stride]);
    }
    (k, v, valid_len)
}

/// Write an f32 slice straight into a host-coherent buffer's mapped memory
/// (UMA: the write IS the on-GPU update). Avoids the `f32_slice_to_bytes` temp
/// `Vec<u8>` the per-op path allocates on every `Buffer::write`.
#[inline]
fn write_f32_mapped(buf: &Buffer, data: &[f32]) {
    // SAFETY: single engine-owning thread; the mapped region is the full buffer
    // and every scratch buffer is sized >= data.len()*4 (see `LagunaScratch`).
    unsafe {
        let p = buf.mapped_ptr.expect("scratch buffer mapped");
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, p, std::mem::size_of_val(data));
    }
}

/// View the first `count` f32s of a host-coherent buffer's mapped memory as a
/// borrowed slice — NO readback / NO temp `Vec` (unlike `read_f32_buf`). The
/// borrow is tied to `buf`; the caller must copy out before the buffer is
/// overwritten by a later dispatch.
#[inline]
fn read_f32_mapped(buf: &Buffer, count: usize) -> &[f32] {
    // SAFETY: mapped_ptr is a valid permanently-mapped f32 region of at least
    // `count` elements (scratch buffers are sized to the max layer shape).
    unsafe { std::slice::from_raw_parts(buf.mapped_ptr.expect("scratch buffer mapped") as *const f32, count) }
}

/// Persistent host-coherent scratch banks for the 1-CB single-token decode path
/// (`VLLM_VULKAN_LAGUNA_SCRATCH`). The per-op `moe_token_1cb`/`attn_cached_1cb`
/// pool-alloc + free ~44 buffers per MoE layer + ~10 per attn layer EVERY decode
/// step; the GPU-timestamp profile pinned that host-coherent alloc/free churn at
/// ~30% of the decode stage. This bank allocates ONE set of buffers, each sized
/// to the MAX layer shape in the PP window, ONCE on the first decode step and
/// reuses them across every subsequent token AND layer — the buffers are
/// overwritten in place (all producers precede all consumers within a command
/// buffer, so no cross-token/-layer hazard). Sized for `new_seq == 1` (decode);
/// the multi-row prefill keeps the per-op path. Pure allocation reuse: the GPU
/// dispatches, push-constants and readbacks are byte-identical to the pool path.
struct LagunaScratch {
    // ── MoE mixer banks (one token; `topk_cap` == num_experts_per_tok) ────────
    inp: Buffer,           // [hs] router input upload
    b_gate: Vec<Buffer>,   // topk_cap × [inter]
    b_up: Vec<Buffer>,     // topk_cap × [inter]
    b_mid: Vec<Buffer>,    // topk_cap × [inter]  (swiglu out)
    b_down: Vec<Buffer>,   // topk_cap × [hs]
    bs_gate: Buffer,       // [shared_inter]
    bs_up: Buffer,         // [shared_inter]
    bs_mid: Buffer,        // [shared_inter]
    bs_down: Buffer,       // [hs]
    moe_out: Buffer,       // [hs] gpu-accum result
    // ── Attention banks (one token; sized to max nq over the window) ──────────
    a_inp: Buffer,         // [hs] q/k/v/g projection input
    a_q: Buffer,           // [max_nq*hd]
    a_k: Buffer,           // [nkv*hd]
    a_v: Buffer,           // [nkv*hd]
    a_g: Buffer,           // [max_nq] gate proj
    a_qb: Buffer,          // [max_nq*hd] gpu-sdpa q upload
    a_ob: Buffer,          // [max_nq*hd] gpu-sdpa out
    a_gb: Buffer,          // [max_nq] softplus gate upload
    a_ab: Buffer,          // [max_nq*hd] attn_out (gated in place)
    a_out: Buffer,         // [hs] o_proj out
    // ── HOSTFOLD host-side working banks (`VLLM_VULKAN_LAGUNA_HOSTFOLD`) ───────
    // Reused across tokens/layers so the qk-norm/rope path allocates NO per-token
    // host Vec. `h_q`/`h_k` hold the owned (in-place mutated) q/k projections;
    // `h_attn_out` the SDPA output. Empty (len 0) unless the flag is on.
    h_q: Vec<f32>,         // [max_nq*hd]
    h_k: Vec<f32>,         // [nkv*hd]
    h_attn_out: Vec<f32>,  // [max_nq*hd]
}

/// HOSTFOLD layer-level host working banks (`VLLM_VULKAN_LAGUNA_HOSTFOLD`).
/// Held in a SEPARATE `Option` from `LagunaScratch` so the layer forward can
/// `.take()` these while `attn_cached_1cb` independently `.take()`s the GPU
/// banks (disjoint fields, no borrow conflict). One set of `[hs]` buffers reused
/// across every token AND layer: the two `cpu_rms_norm` outputs and the
/// post-attn residual `h1`. The final layer-output residual is the returned
/// `Vec` (moved onward), so it keeps its one alloc.
struct LagunaHostScratch {
    normed: Vec<f32>,  // [hs] input_layernorm output
    h1: Vec<f32>,      // [hs] hidden + attn residual
    normed2: Vec<f32>, // [hs] post_attention_layernorm output
}

/// A GPU-resident Laguna model for the PP window `[pp_start, pp_end)`. The
/// analog of `nemotron::NemotronModel` / `qwen35::Qwen35Model` for the resident
/// hybrid: owns the engine + the resident weight maps that `load_laguna_resident`
/// fills, and runs the window forward.
/// GPU MoE accumulate toggle (`VLLM_VULKAN_LAGUNA_GPU_ACCUM`): default ON.
/// When set to `0`, `moe_token_1cb` uses the host weighted-accumulate loop
/// (== commit bb33073) instead of the folded `laguna_moe_accum` dispatch —
/// the A/B baseline for the single-node bit-exact gate.
fn laguna_gpu_accum_on() -> bool {
    std::env::var("VLLM_VULKAN_LAGUNA_GPU_ACCUM")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// MoE CB-batch dispatch fold toggle (`VLLM_VULKAN_LAGUNA_CBBATCH`): default OFF.
/// Reads the process flag (parsed once by `Flags::from_env`). When on, the
/// folded MoE path collapses the 30 per-expert NVFP4 matvecs into 3 batched
/// dispatches (see `moe_token_1cb_full`).
fn laguna_gpu_cbbatch_on() -> bool {
    crate::flags::flags_global().laguna_cbbatch
}

/// GPU MoE router toggle (`VLLM_VULKAN_LAGUNA_GPU_ROUTER`): default OFF.
/// When set, `moe_token_1cb` computes the sigmoid router (gate matvec + sigmoid
/// + e_score_correction_bias + top-k selection) on the GPU (`laguna_router`
/// shader) and reads back only the top-k idx+weights, instead of the HOST
/// `router_forward` matvec over the [num_experts, hidden] gate weight. Default
/// OFF; the single-node bit-exact gate (`debug_laguna_gpu_router`) confirms the
/// selected expert set + weights match the host path before it ever ships ON.
fn laguna_gpu_router_on() -> bool {
    std::env::var("VLLM_VULKAN_LAGUNA_GPU_ROUTER")
        .map(|v| v != "0")
        .unwrap_or(false)
}

/// CPU shared-expert ∥ routed-expert overlap toggle
/// (`VLLM_VULKAN_LAGUNA_CPU_OVERLAP`): default OFF. When set to `1`/`true`,
/// `moe_token_1cb` submits the routed-expert command buffer NON-BLOCKING, runs
/// the data-independent shared expert on the host rayon pool CONCURRENTLY with
/// the GPU, then syncs before the terminal residual add. Bit-exact with the
/// sequential host-shared path by construction (only the timing of the shared
/// branch changes, not the reduction order). Gated by `debug_laguna_cpuoverlap`.
fn laguna_cpu_overlap_on() -> bool {
    std::env::var("VLLM_VULKAN_LAGUNA_CPU_OVERLAP")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

/// Rayon pool for the overlapped shared expert. Sized to 6 threads by default
/// (override `VLLM_VULKAN_LAGUNA_CPU_OVERLAP_THREADS`), NOT the full 12 SMT
/// threads: the GPU driver's host-submit thread needs a core while the routed
/// command buffer is in flight, and the measured 6-thread shared bucket
/// (~0.737ms) already hides under the routed-expert GPU wall. A dedicated pool
/// (vs the global rayon pool) keeps the thread count independent of any other
/// rayon use and avoids re-reading the env on every token.
fn laguna_overlap_pool() -> &'static rayon::ThreadPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("VLLM_VULKAN_LAGUNA_CPU_OVERLAP_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(6);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("laguna-overlap-{i}"))
            .build()
            .expect("build laguna overlap rayon pool")
    })
}

pub struct LagunaGpuModel {
    pub config: LagunaConfig,
    pub engine: Option<ComputeEngine>,
    /// f16-resident BF16-native matmuls: attn q/k/v/o/g, dense (layer-0) MLP,
    /// shared expert, lm_head, and (embed-f16 flag) the embedding table.
    pub gpu_weights: HashMap<String, NemGpuWeight>,
    /// NVFP4 routed experts, one concat holder per MoE layer.
    pub gpu_experts: HashMap<usize, LagunaMoeExperts>,
    /// Small f32 host tensors: input/post layernorm, q/k head-norm, router gate,
    /// e_score bias, final norm, and (no embed-f16) the embedding table.
    pub host: HashMap<String, SimpleTensor>,
    pub pp_start: usize,
    pub pp_end: usize,
    pub pp_first: bool,
    pub pp_last: bool,
    /// Checkpoint dir (lets the bit-exact gate load the CPU oracle layer).
    pub dir: std::path::PathBuf,
    /// Per-layer POST-ROPE K/V cache (one per global layer in `[pp_start,
    /// pp_end)`), keyed by global layer index. Populated by the cache-populating
    /// prefill (`forward_prefill_*`) and appended by the single-token decode
    /// (`forward_decode_*`). Both the full-attn (unbounded) and sliding-window
    /// (last-512 via `cpu_sdpa`'s `kv_start`) layers share the same storage: the
    /// window is a READ-side clamp in SDPA, not a storage ring, so the cache is
    /// bit-exact with the stateless full recompute at every step (see
    /// `attn_cached`). All Laguna layers have the SAME `num_key_value_heads`
    /// (8) and `head_dim`, so the caches are uniform.
    pub kv: HashMap<usize, KvCache>,
    /// DEVICE-RESIDENT post-rope K/V planes (PIECE 3 — the span-fold arena), one
    /// per resident layer, allocated when an engine is present. The 1-CB decode
    /// path (`attn_cached_1cb` / `forward_*_1cb`) appends to and reads from THESE
    /// (bit-identical to the host `kv` map); the per-op path keeps using `kv`.
    /// Empty when `engine=None` (the CPU-fallback / Mac-test path).
    pub kv_res: HashMap<usize, ResidentKvPlane>,
    /// Cache capacity (== the Python `MAXSEQ`); full-attn K/V grows to this.
    pub max_seq: usize,
    /// Persistent 1-CB decode scratch banks (`VLLM_VULKAN_LAGUNA_SCRATCH`).
    /// `None` until the first decode step allocates them (see `ensure_scratch`);
    /// reused for every token/layer thereafter. Empty when the flag is off.
    scratch: Option<LagunaScratch>,
    /// GPU-resident router gate weight + e_score bias per MoE layer, uploaded
    /// LAZILY on the first `VLLM_VULKAN_LAGUNA_GPU_ROUTER` dispatch for the layer
    /// (`(gate [ne*hs] f32, bias [ne] f32)`), so the [num_experts, hidden] gate
    /// no longer streams to host per layer. Empty on the host-router path.
    pub gpu_router: HashMap<usize, (Buffer, Buffer)>,
    /// FUSED native-vCCL PP-hop `[H]` scratch (`VulkanModel::pp_step_laguna`,
    /// `VLLM_VULKAN_LAGUNA_NATIVE_HOP`), mirroring nemotron/kimi's
    /// `pp_{recv,send}_scratch`: the previous stage's hidden is recv'd INTO
    /// `pp_recv_scratch` and this stage's hidden is sent FROM `pp_send_scratch`,
    /// both `comm_register`'d ONCE so vCCL's send/recv skip the per-call
    /// `ibv_reg_mr`/dereg temp-MR (the "buffer (N B) not registered with the
    /// comm" WARN). Empty + `0` handle until `pp_step_laguna` pins them (gated by
    /// `VLLM_VULKAN_REG_REDUCE`); on a registration failure they fall back to the
    /// per-call fresh-Vec `recv_f32`/`send_f32` path (correct, just slower).
    /// Decode carries `[1*H]`; a shape change re-pins (see `pp_step_laguna`).
    pub pp_recv_scratch: Vec<f32>,
    pub pp_recv_handle: usize,
    pub pp_send_scratch: Vec<f32>,
    pub pp_send_handle: usize,
    /// `[vocab]` ring-back scratch for the SERVING logits hop (`pp_step_laguna_logits`,
    /// Phase-3 distributed OpenAI serving). The last PP stage sends its full
    /// `[vocab=100352]` logit vector to rank0 FROM `pp_vocab_scratch`; rank0 (first
    /// stage) recv's it INTO the SAME field. `comm_register`'d ONCE (401 KB), same
    /// pattern as `pp_{recv,send}_scratch`, so the per-token sampler ring-back skips
    /// the per-call temp-MR. Unlike the fused `pp_step_laguna` (which argmaxes on the
    /// last stage → `(tok,logit)`), the serving path must return the WHOLE `[vocab]`
    /// to rank0 for vLLM's `Sampler` (temperature/top-k/top-p/penalties/logprobs).
    pub pp_vocab_scratch: Vec<f32>,
    pub pp_vocab_handle: usize,
    /// `[2*K]` top-K ring-back scratch for the FAST SERVING sampler hop
    /// (`pp_step_laguna_topk`). Instead of ringing the whole `[vocab=100352]`
    /// logit vector to rank0 (the ~401 KB marshal + Python full-vocab argmax/
    /// sample that makes the serve path ~128 ms/tok vs the offline 51.9), the
    /// last PP stage does the top-K selection IN RUST and rings back only the K
    /// winning `(logit, index)` pairs — `2*K` f32 (K logits, then K indices
    /// encoded as f32; vocab < 2^24 so the index round-trips exactly). rank0
    /// recv's them INTO the SAME field and hands the K candidates to the
    /// sampler. `comm_register`'d ONCE, same fall-back-to-fresh-Vec contract as
    /// `pp_vocab_scratch`. Sized `2*K` on first use; re-pinned if K changes.
    /// The full-vocab `pp_step_laguna_logits` stays as the fallback for requests
    /// that need the whole distribution (penalties/min_p/full logprobs).
    pub pp_topk_scratch: Vec<f32>,
    pub pp_topk_handle: usize,
    /// Persistent HOST-side layer scratch (`VLLM_VULKAN_LAGUNA_HOSTFOLD`). Kept
    /// separate from `scratch` so `layer_forward_cached_1cb` can hold it while the
    /// callee `attn_cached_1cb` takes the GPU banks. `None` until first decode.
    host_scratch: Option<LagunaHostScratch>,
}

/// Copy dispatch geometry for one resident f16 weight, gathered before the
/// mutable engine borrow (raw ptr stays valid: `gpu_weights` is never mutated
/// during a forward).
type F16Meta = (*const Buffer, usize, usize); // (buffer, out_features, in_features)

/// Dispatch kind for a resident `f16_matvec`/`rec_mv` weight: plain f16, or the
/// int8-attn lever's Q8_0 weight (symmetric int8, per-32-block scale in-block —
/// no side scale buffer, so the same 3-buffer [w,in,out] dispatch as f16, only
/// the shader/variant differ). Selected from `NemGpuWeight.quant` by `mv_meta`.
#[derive(Clone, Copy)]
enum LagMvKind {
    F16,
    Q8_0,
}

/// Copy dispatch geometry for one routed-expert projection.
type ExpMeta = (
    *const Buffer, // packed
    *const Buffer, // scales
    usize,         // out_features (n)
    usize,         // in_features  (k)
    usize,         // group_size
    bool,          // e4m3
    f32,           // reciprocal global for expert e (e4m3 path)
);

impl LagunaGpuModel {
    /// Load a Laguna checkpoint's `[layer_start, layer_end)` window GPU-resident
    /// (via [`load_laguna_resident`]). `engine` MUST be Some for the resident
    /// GPU path; a None engine keeps the maps empty and the forward degenerates
    /// to the CPU fallback (which needs host weights — only used by tests).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights_path: &std::path::Path,
        cfg: LagunaConfig,
        mut engine: ComputeEngine,
        layer_start: usize,
        layer_end: usize,
        max_seq: usize,
    ) -> Result<Self, String> {
        // `weights_path` is a checkpoint FILE (any shard); `discover_shards`
        // globs its PARENT dir, which is the checkpoint dir the oracle also uses.
        let dir = weights_path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let total = cfg.num_hidden_layers;
        // STANDALONE (gate-only): treat a partial `[0,K)` window as a full model —
        // keep BOTH the embedding (pp_first) and final-norm+lm_head (pp_last) so a
        // single node runs the whole embed→layers→lm_head free-gen loop with NO PP
        // comm. Output coherence is meaningless (layers K..N missing), but the
        // KVCACHE-vs-stateless free-gen DIVERGENCE (the repeat bug) reproduces iff
        // it is kernel-side (embed-decode / lm_head-decode / cache) — which the
        // MID-window `debug_laguna_kvcache` gate cannot exercise.
        let standalone = std::env::var("VLLM_VULKAN_LAGUNA_STANDALONE")
            .map(|v| v != "0").unwrap_or(false);
        let pp_first = standalone || layer_start == 0;
        let pp_last = standalone || layer_end >= total;
        let mut gpu_weights = HashMap::new();
        let mut gpu_experts = HashMap::new();
        let mut host = HashMap::new();
        load_laguna_resident(
            weights_path,
            &cfg,
            &mut engine,
            &mut gpu_weights,
            &mut gpu_experts,
            &mut host,
            layer_start,
            layer_end,
            pp_first,   // keep_embed on the first stage
            pp_last,    // keep_lm + final norm on the last stage
        )?;
        // One POST-ROPE K/V cache per resident layer (uniform nkv/head_dim).
        let mut kv = HashMap::new();
        for li in layer_start..layer_end {
            kv.insert(li, KvCache::new(max_seq, cfg.num_key_value_heads, cfg.head_dim));
        }
        // DEVICE-RESIDENT K/V planes (span-fold arena) — one per layer, persistent
        // GPU buffers appended in place each decode step and consumed by SDPA with
        // no round-trip. Allocated now while `engine` is still borrowable.
        // Phase-0 per-layer KV sizing: with VLLM_VULKAN_LAGUNA_KV_RING each
        // sliding-window layer's plane is a `window`-sized ring (capacity =
        // layer_kv_capacity); full/YaRN layers keep the full `max_seq` plane.
        // Default OFF → capacity == max_seq for every layer (byte-identical alloc).
        let kv_ring = crate::flags::flags_global().laguna_kv_ring;
        let mut kv_res = HashMap::new();
        for li in layer_start..layer_end {
            let capacity = if kv_ring { cfg.layer_kv_capacity(li, max_seq) } else { max_seq };
            kv_res.insert(
                li,
                ResidentKvPlane::new(&mut engine, max_seq, capacity, cfg.num_key_value_heads, cfg.head_dim)?,
            );
        }
        Ok(LagunaGpuModel {
            config: cfg,
            engine: Some(engine),
            gpu_weights,
            gpu_experts,
            host,
            pp_start: layer_start,
            pp_end: layer_end,
            pp_first,
            pp_last,
            dir: dir.to_path_buf(),
            kv,
            kv_res,
            max_seq,
            scratch: None,
            gpu_router: HashMap::new(),
            pp_recv_scratch: Vec::new(),
            pp_recv_handle: 0,
            pp_send_scratch: Vec::new(),
            pp_send_handle: 0,
            pp_vocab_scratch: Vec::new(),
            pp_vocab_handle: 0,
            pp_topk_scratch: Vec::new(),
            pp_topk_handle: 0,
            host_scratch: None,
        })
    }

    /// Allocate the persistent 1-CB decode scratch banks if not already present
    /// (`VLLM_VULKAN_LAGUNA_SCRATCH`). Sized to the MAX layer shape in the PP
    /// window so ONE set of buffers serves every token and layer. Called once on
    /// the first decode step; a no-op thereafter. Requires an engine.
    fn ensure_scratch(&mut self) -> Result<(), String> {
        if self.scratch.is_some() {
            return Ok(());
        }
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nkv = cfg.num_key_value_heads;
        let inter = cfg.moe_intermediate_size;
        let sh = cfg.shared_expert_intermediate_size;
        let topk_cap = cfg.num_experts_per_tok;
        // Max query-head count over the resident window (attn banks are shared
        // across layers so must fit the widest layer).
        let max_nq = (self.pp_start..self.pp_end)
            .map(|li| cfg.num_attention_heads_per_layer[li])
            .max()
            .unwrap_or(cfg.num_attention_heads);
        let eng = self.engine.as_mut().ok_or("ensure_scratch: no engine")?;
        fn a(eng: &mut ComputeEngine, elems: usize) -> Result<Buffer, String> {
            eng.alloc_host_coherent_storage((elems * 4) as u64)
        }
        fn av(eng: &mut ComputeEngine, n: usize, elems: usize) -> Result<Vec<Buffer>, String> {
            (0..n).map(|_| a(eng, elems)).collect()
        }
        let scratch = LagunaScratch {
            inp: a(eng, hs)?,
            b_gate: av(eng, topk_cap, inter)?,
            b_up: av(eng, topk_cap, inter)?,
            b_mid: av(eng, topk_cap, inter)?,
            b_down: av(eng, topk_cap, hs)?,
            bs_gate: a(eng, sh)?,
            bs_up: a(eng, sh)?,
            bs_mid: a(eng, sh)?,
            bs_down: a(eng, hs)?,
            moe_out: a(eng, hs)?,
            a_inp: a(eng, hs)?,
            a_q: a(eng, max_nq * hd)?,
            a_k: a(eng, nkv * hd)?,
            a_v: a(eng, nkv * hd)?,
            a_g: a(eng, max_nq)?,
            a_qb: a(eng, max_nq * hd)?,
            a_ob: a(eng, max_nq * hd)?,
            a_gb: a(eng, max_nq)?,
            a_ab: a(eng, max_nq * hd)?,
            a_out: a(eng, hs)?,
            h_q: vec![0.0f32; max_nq * hd],
            h_k: vec![0.0f32; nkv * hd],
            h_attn_out: vec![0.0f32; max_nq * hd],
        };
        self.scratch = Some(scratch);
        // HOSTFOLD layer-level host banks (allocated regardless of the flag; a few
        // `[hs]` Vecs, negligible — the flag only gates whether they're USED).
        self.host_scratch = Some(LagunaHostScratch {
            normed: vec![0.0f32; hs],
            h1: vec![0.0f32; hs],
            normed2: vec![0.0f32; hs],
        });
        Ok(())
    }

    #[inline]
    fn w(&self, name: &str) -> &[f32] {
        &self.host.get(name).unwrap_or_else(|| panic!("host tensor '{name}' missing")).data
    }

    /// Raw pointer + length of a host weight's f32 data — for the HOSTFOLD path to
    /// BORROW the weight (no per-layer `.to_vec()`) without holding a `self` borrow
    /// across a later `&mut self` call. Safe to reconstruct a slice from: `host` is
    /// never mutated during a forward (same invariant `f16_meta` relies on).
    #[inline]
    fn w_ptr(&self, name: &str) -> (*const f32, usize) {
        let d = &self.host.get(name).unwrap_or_else(|| panic!("host tensor '{name}' missing")).data;
        (d.as_ptr(), d.len())
    }

    // ── matmul dispatch (GPU resident, else exact cpu_matmul) ────────────────

    fn f16_meta(&self, name: &str) -> Option<F16Meta> {
        self.gpu_weights
            .get(name)
            .map(|w| (&w.buffer as *const Buffer, w.out_features, w.in_features))
    }

    /// Like `f16_meta` but also carries the dispatch KIND (plain f16 vs the
    /// int8-attn lever's Q8_0 weight). Raw ptr stays valid: `gpu_weights` is
    /// never mutated during a forward. Returns `(weight_ptr, kind, out, in)`.
    fn mv_meta(&self, name: &str) -> Option<(*const Buffer, LagMvKind, usize, usize)> {
        self.gpu_weights.get(name).map(|w| {
            let kind = match &w.quant {
                NemQuant::Q8_0 => LagMvKind::Q8_0,
                _ => LagMvKind::F16,
            };
            (&w.buffer as *const Buffer, kind, w.out_features, w.in_features)
        })
    }

    /// `out[1,n] = x[1,k] @ W[n,k]^T` for a BF16-native (f16-resident) weight,
    /// OR (int8-attn lever) a Q8_0-resident weight routed through
    /// `mul_mat_vec_q8_0deq_f32_f32`. GPU when resident, else `cpu_matmul` on the
    /// host f32 copy (bit-exact fallback; f16 weights only — a q8 target is always
    /// resident when the flag is on).
    fn f16_matvec(&mut self, name: &str, x: &[f32], k: usize, n: usize) -> Vec<f32> {
        let meta = self.mv_meta(name);
        if let (Some(eng), Some((w_ptr, kind, _out, _in))) = (self.engine.as_mut(), meta) {
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            let inp_p = &inp as *const Buffer;
            let out_p = &out as *const Buffer;
            let (shader, r) = match kind {
                LagMvKind::Q8_0 => matvec_q8_0_variant_k(k, n),
                LagMvKind::F16 => matvec_f16_variant_k(k, n),
            };
            let wg = ((n as u32 + r - 1) / r, 1u32, 1u32);
            let pc = matvec_pc13(k, n);
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, &shader, &[&*w_ptr, &*inp_p, &*out_p], &pc, wg).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            let result = read_f32_buf(&out, n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
            result
        } else {
            cpu_matmul(x, self.w(name), 1, k, n)
        }
    }

    fn expert_meta(&self, layer: usize, proj: u8, e: usize) -> Option<ExpMeta> {
        let ex = self.gpu_experts.get(&layer)?;
        let p = match proj {
            0 => &ex.gate,
            1 => &ex.up,
            _ => &ex.down,
        };
        Some((
            &p.packed as *const Buffer,
            &p.scales as *const Buffer,
            p.out_features,
            p.in_features,
            ex.group_size as usize,
            ex.e4m3,
            p.globals[e],
        ))
    }

    /// One routed-expert NVFP4 matvec `out[1,n] = x[1,k] @ W_e[n,k]^T`, sliced to
    /// expert `e` in the concatenated per-layer buffer via the shader's
    /// `packed_off`/`sb_off` (exactly the nemotron/qwen `nem_rec_expert_mv`
    /// offsets). proj: 0=gate, 1=up, 2=down. GPU when resident, else the CPU
    /// oracle dequant (`OwnedExpertsPacked::dequant`) is NOT available here, so a
    /// resident buffer is required for the GPU path; the host fallback panics
    /// (experts are never kept host-f32 under the resident loader).
    fn expert_matvec(&mut self, layer: usize, proj: u8, e: usize, x: &[f32]) -> Vec<f32> {
        let meta = self
            .expert_meta(layer, proj, e)
            .unwrap_or_else(|| panic!("no resident experts for layer {layer}"));
        let (w_ptr, s_ptr, n, k, gs, e4m3, global) = meta;
        let eng = self.engine.as_mut().expect("engine present for resident experts");
        let xb = f32_slice_to_bytes(x);
        let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
        inp.write(&xb).unwrap();
        let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        let inp_p = &inp as *const Buffer;
        let out_p = &out as *const Buffer;
        // word stride e*out*(in/8); scale stride e*out*(in/group_size) (byte-elem
        // for e4m3, f32-elem for fold — same value, 1 unit/group).
        let packed_off = e * n * (k / 8);
        let sb_off = e * n * (k / gs);
        let cb = eng.begin_batch().unwrap();
        unsafe {
            if e4m3 {
                let (shader, r) = laguna_e4m3_expert_shader(k, n, gs);
                let wg = ((n as u32 + r - 1) / r, 1u32, 1u32);
                let pc = matvec_nvfp4_e4m3_pc_off(k, n, gs, packed_off, sb_off, global);
                eng.record_to(cb, &shader, &[&*w_ptr, &*s_ptr, &*inp_p, &*out_p], &pc, wg)
                    .unwrap();
            } else {
                let (shader, r) = matvec_nvfp4_variant_k(k, n);
                let wg = ((n as u32 + r - 1) / r, 1u32, 1u32);
                let pc = matvec_mlx4_pc_off(k, n, gs, packed_off, sb_off);
                eng.record_to(cb, &shader, &[&*w_ptr, &*s_ptr, &*inp_p, &*out_p], &pc, wg)
                    .unwrap();
            }
        }
        eng.submit_batch(cb).unwrap();
        let result = read_f32_buf(&out, n);
        eng.return_to_pool(inp);
        eng.return_to_pool(out);
        result
    }

    // ── forward blocks (mirror laguna.rs exactly; matmuls GPU-routed) ────────

    /// Gated GQA attention for one layer over `[seq, hidden]`. Byte-for-byte the
    /// oracle `laguna::laguna_attn` control flow with q/k/v/o/g projections
    /// routed through `f16_matvec`.
    pub fn attn(&mut self, hidden_normed: &[f32], seq: usize, layer_idx: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nkv = cfg.num_key_value_heads;
        let nq = cfg.num_attention_heads_per_layer[layer_idx];
        let is_full = cfg.layer_is_full[layer_idx];
        let eps = cfg.rms_norm_eps;
        let p = format!("model.layers.{layer_idx}.self_attn");

        // Full [seq, *] projections, row by row (matches cpu_matmul's output).
        let mut q = self.proj_seq(&format!("{p}.q_proj.weight"), hidden_normed, seq, hs, nq * hd);
        let mut k = self.proj_seq(&format!("{p}.k_proj.weight"), hidden_normed, seq, hs, nkv * hd);
        let v = self.proj_seq(&format!("{p}.v_proj.weight"), hidden_normed, seq, hs, nkv * hd);

        let q_norm = self.w(&format!("{p}.q_norm.weight")).to_vec();
        let k_norm = self.w(&format!("{p}.k_norm.weight")).to_vec();
        for pos in 0..seq {
            {
                let qs = &mut q[pos * nq * hd..(pos + 1) * nq * hd];
                for h in 0..nq {
                    cpu_rms_norm_inplace(&mut qs[h * hd..(h + 1) * hd], &q_norm, eps);
                }
            }
            {
                let ks = &mut k[pos * nkv * hd..(pos + 1) * nkv * hd];
                for h in 0..nkv {
                    cpu_rms_norm_inplace(&mut ks[h * hd..(h + 1) * hd], &k_norm, eps);
                }
            }
            let qs = &mut q[pos * nq * hd..(pos + 1) * nq * hd] as *mut [f32];
            let ks = &mut k[pos * nkv * hd..(pos + 1) * nkv * hd] as *mut [f32];
            // SAFETY: q and k are distinct Vecs; the two slices never overlap.
            unsafe {
                if is_full {
                    cpu_rope_yarn(
                        &mut *qs, &mut *ks, pos, nq, nkv, hd, cfg.full_rotary_dim,
                        &cfg.yarn_inv_freq, cfg.full_attention_factor,
                    );
                } else {
                    cpu_rope(&mut *qs, &mut *ks, pos, nq, nkv, hd, hd, cfg.sliding_rope_theta);
                }
            }
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        let mut attn_out = vec![0.0f32; seq * nq * hd];
        for pos in 0..seq {
            let q_p = &q[pos * nq * hd..(pos + 1) * nq * hd];
            let k_ctx = &k[0..(pos + 1) * nkv * hd];
            let v_ctx = &v[0..(pos + 1) * nkv * hd];
            let o = cpu_sdpa(q_p, k_ctx, v_ctx, nq, nkv, hd, pos + 1, scale, window);
            attn_out[pos * nq * hd..(pos + 1) * nq * hd].copy_from_slice(&o);
        }

        // Per-head scalar softplus gate from the attention INPUT, broadcast
        // across head_dim, applied BEFORE o_proj.
        let g = self.proj_seq(&format!("{p}.g_proj.weight"), hidden_normed, seq, hs, nq);
        for pos in 0..seq {
            for h in 0..nq {
                let gate = softplus(g[pos * nq + h]);
                let head = &mut attn_out[pos * nq * hd + h * hd..pos * nq * hd + (h + 1) * hd];
                for d in head.iter_mut() {
                    *d *= gate;
                }
            }
        }

        self.proj_seq(&format!("{p}.o_proj.weight"), &attn_out, seq, nq * hd, hs)
    }

    /// `[seq, n] = X[seq, k] @ W[n,k]^T` by dispatching the resident matvec once
    /// per row — the multi-row form of `f16_matvec` (matches `cpu_matmul`'s
    /// row-major output).
    fn proj_seq(&mut self, name: &str, x: &[f32], seq: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; seq * n];
        for t in 0..seq {
            let row = self.f16_matvec(name, &x[t * k..(t + 1) * k], k, n);
            out[t * n..(t + 1) * n].copy_from_slice(&row);
        }
        out
    }

    /// Dense SwiGLU MLP (layer 0) over `[seq, hidden]`.
    pub fn dense_mlp(&mut self, h: &[f32], seq: usize, layer_idx: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let p = format!("model.layers.{layer_idx}.mlp");
        let g = self.proj_seq(&format!("{p}.gate_proj.weight"), h, seq, hs, inter);
        let u = self.proj_seq(&format!("{p}.up_proj.weight"), h, seq, hs, inter);
        let act: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu(a) * b).collect();
        self.proj_seq(&format!("{p}.down_proj.weight"), &act, seq, inter, hs)
    }

    /// MoE mixer for a SINGLE token `h` (`[hidden]`, post-norm). Router top-k +
    /// per-expert gated-SiLU (routed weights carry ×routed_scaling_factor) plus
    /// the ungated shared expert. Mirrors `laguna::laguna_moe_token`.
    pub fn moe_token(&mut self, h: &[f32], layer_idx: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        let dims = cfg.router_dims();
        let p = format!("model.layers.{layer_idx}.mlp");

        let router = self.w(&format!("{p}.gate.weight")).to_vec();
        let bias = self.w(&format!("{p}.experts.e_score_correction_bias")).to_vec();
        let (indices, weights) = router_forward(h, &router, &bias, &dims);

        let mut routed = vec![0.0f32; hs];
        for (kth, &e) in indices.iter().enumerate() {
            let gp = self.expert_matvec(layer_idx, 0, e, h); // [inter]
            let up = self.expert_matvec(layer_idx, 1, e, h); // [inter]
            let act: Vec<f32> = gp.iter().zip(&up).map(|(&a, &b)| silu(a) * b).collect();
            let dn = self.expert_matvec(layer_idx, 2, e, &act); // [hidden]
            let wk = weights[kth];
            for (r, &o) in routed.iter_mut().zip(&dn) {
                *r += o * wk;
            }
            let _ = inter; // dim documented; act length == inter by construction
        }

        // Ungated shared expert (gated-SiLU, no extra sigmoid gate).
        let sg = self.f16_matvec(&format!("{p}.shared_expert.gate_proj.weight"), h, hs, shared_inter);
        let su = self.f16_matvec(&format!("{p}.shared_expert.up_proj.weight"), h, hs, shared_inter);
        let sact: Vec<f32> = sg.iter().zip(&su).map(|(&a, &b)| silu(a) * b).collect();
        let sd = self.f16_matvec(&format!("{p}.shared_expert.down_proj.weight"), &sact, shared_inter, hs);

        routed.iter().zip(&sd).map(|(&r, &s)| r + s).collect()
    }

    /// One decoder layer (pre-norm gated attn + residual, pre-norm MLP +
    /// residual). Mirrors `laguna::laguna_layer_forward`. THE per-layer
    /// bit-exact GATE primitive: feed an identical `hidden`, compare vs the
    /// oracle over the same weights.
    pub fn layer_forward(&mut self, hidden: &[f32], seq: usize, layer_idx: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        let in_ln = self.w(&format!("model.layers.{layer_idx}.input_layernorm.weight")).to_vec();
        let normed = cpu_rms_norm(hidden, &in_ln, eps);
        let attn = self.attn(&normed, seq, layer_idx);
        let h1: Vec<f32> = hidden.iter().zip(&attn).map(|(&a, &b)| a + b).collect();

        let post_ln = self.w(&format!("model.layers.{layer_idx}.post_attention_layernorm.weight")).to_vec();
        let normed2 = cpu_rms_norm(&h1, &post_ln, eps);
        let mlp = if cfg.mlp_only_layers.contains(&layer_idx) {
            self.dense_mlp(&normed2, seq, layer_idx)
        } else {
            let mut out = vec![0.0f32; seq * hs];
            for t in 0..seq {
                let m = self.moe_token(&normed2[t * hs..(t + 1) * hs], layer_idx);
                out[t * hs..(t + 1) * hs].copy_from_slice(&m);
            }
            out
        };
        h1.iter().zip(&mlp).map(|(&a, &b)| a + b).collect()
    }

    /// Embedding row for `tok` (first stage). f16-resident (widen) or host f32.
    fn embed_row(&self, tok: u32) -> Vec<f32> {
        let hs = self.config.hidden_size;
        if let Some(w) = self.gpu_weights.get("model.embed_tokens.weight") {
            let ptr = w.buffer.mapped_ptr.expect("embed buffer mapped") as *const u8;
            let mut row = vec![0.0f32; hs];
            unsafe {
                for (i, r) in row.iter_mut().enumerate() {
                    let off = (tok as usize * hs + i) * 2;
                    let bits = u16::from_le_bytes([*ptr.add(off), *ptr.add(off + 1)]);
                    *r = half::f16::from_bits(bits).to_f32();
                }
            }
            row
        } else {
            let e = self.w("model.embed_tokens.weight");
            e[tok as usize * hs..(tok as usize + 1) * hs].to_vec()
        }
    }

    /// Full-window prefill forward over a token-id sequence (the from-embedding
    /// driver, used when this stage owns `[0, num_layers)` or for a first-stage
    /// prefill). Returns `[seq, hidden]` last_hidden_state (no lm_head) OR, on
    /// the last stage, last-position logits `[vocab]`. Mirrors
    /// `LagunaWeights::forward`.
    pub fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let seq = tokens.len();
        let mut hidden = vec![0.0f32; seq * hs];
        if self.pp_first {
            for (t, &tok) in tokens.iter().enumerate() {
                let row = self.embed_row(tok);
                hidden[t * hs..(t + 1) * hs].copy_from_slice(&row);
            }
        }
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward(&hidden, seq, li);
        }
        if self.pp_last {
            let fn_w = self.w("model.norm.weight").to_vec();
            let normed = cpu_rms_norm(&hidden, &fn_w, cfg.rms_norm_eps);
            if self.gpu_weights.contains_key("lm_head.weight")
                || self.host.contains_key("lm_head.weight")
            {
                let last = &normed[(seq - 1) * hs..seq * hs];
                return self.f16_matvec("lm_head.weight", last, hs, cfg.vocab_size);
            }
            return normed;
        }
        hidden
    }

    /// Receive-hidden PP-stage forward (the mid/last-stage twin of [`forward`],
    /// which owns the FIRST stage's embed path). Continues from a `[seq, hidden]`
    /// hidden state received over the PP hop — NO embedding — runs the resident
    /// layers `[pp_start, pp_end)`, and on the last stage applies the final
    /// `model.norm` + `lm_head` (LAST position) → `[vocab]` logits; otherwise
    /// returns the `[seq, hidden]` to ship onward.
    ///
    /// ⚠️ Laguna attention is STATELESS full-sequence recompute (`attn` /
    /// `laguna::laguna_attn` rebuild K/V over `[0, seq)` every call — there is NO
    /// KV cache), so the PP hop carries the WHOLE `[seq * hidden]` hidden, not a
    /// single decode token's `hidden_size` vector the way the stateful
    /// nemotron/qwen3.6 hops do: a mid stage needs every prior position present to
    /// recompute this step's attention. Bit-exact with the `[pp_start, pp_end)`
    /// slice of a single-node [`forward`] over the same tokens (identical ops,
    /// same GPU matvecs) — the PP split is invariant to where the stage boundary
    /// falls because each layer's compute is local to `hidden`.
    pub fn forward_hidden(&mut self, hidden_in: &[f32], seq: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        debug_assert_eq!(
            seq * hs,
            hidden_in.len(),
            "forward_hidden: hidden_in.len() {} != seq {seq} * hidden {hs}",
            hidden_in.len()
        );
        let mut hidden = hidden_in.to_vec();
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward(&hidden, seq, li);
        }
        if self.pp_last {
            let fn_w = self.w("model.norm.weight").to_vec();
            let normed = cpu_rms_norm(&hidden, &fn_w, cfg.rms_norm_eps);
            if self.gpu_weights.contains_key("lm_head.weight")
                || self.host.contains_key("lm_head.weight")
            {
                let last = &normed[(seq - 1) * hs..seq * hs];
                return self.f16_matvec("lm_head.weight", last, hs, cfg.vocab_size);
            }
            return normed;
        }
        hidden
    }

    // ── KV-CACHED decode path (O(1)/step instead of O(seq) recompute) ────────
    //
    // The stateless `attn`/`forward`/`forward_hidden` above rebuild K/V over
    // `[0, seq)` every call — a per-step quadratic. The methods below keep a
    // per-layer POST-ROPE K/V cache (`self.kv`) so a decode step only projects
    // the ONE new token, appends its K/V, and attends the single new query
    // against the cache. Bit-exactness vs the stateless path is by construction:
    // `attn_cached` appends token j's post-rope K/V then attends with
    // `seq_len = cache.seq_len` and the SAME `window` — identical to what the
    // stateless `attn` computes for that position (its `k_ctx = k[0..(pos+1)]`
    // and `cpu_sdpa(seq_len = pos+1, window)`). The sliding-window layers need no
    // ring buffer: `cpu_sdpa` clamps reads to the last `sliding_window` via
    // `kv_start = seq_len.saturating_sub(window)`, so full storage + windowed
    // read is bit-identical (the ring is a pure memory optimization, unneeded
    // for `seq < 512`).

    /// Reset every resident layer's K/V counter to 0 (start of a fresh sequence).
    /// The underlying storage is overwrite-in-place, so this is just the counters.
    pub fn reset_kv(&mut self) {
        for c in self.kv.values_mut() {
            c.truncate(0);
        }
        for p in self.kv_res.values_mut() {
            p.reset();
        }
    }

    /// KV-cached gated GQA attention for `new_seq` NEW tokens appended at the
    /// cache's current tail (`start_pos = cache.seq_len`). `new_seq == seq` for a
    /// cache-populating prefill (`start_pos == 0`), `new_seq == 1` for a decode
    /// step. Byte-for-byte the `attn` control flow, except K/V come from / go to
    /// `self.kv[layer_idx]` and each new query attends the cache instead of a
    /// freshly recomputed prefix.
    pub fn attn_cached(&mut self, hidden_normed: &[f32], new_seq: usize, layer_idx: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nkv = cfg.num_key_value_heads;
        let nq = cfg.num_attention_heads_per_layer[layer_idx];
        let is_full = cfg.layer_is_full[layer_idx];
        let eps = cfg.rms_norm_eps;
        let p = format!("model.layers.{layer_idx}.self_attn");

        let start_pos = self.kv.get(&layer_idx).map(|c| c.seq_len).unwrap_or(0);

        // Project the NEW tokens only (1 matvec/row/proj — O(new_seq), not O(seq)).
        let mut q = self.proj_seq(&format!("{p}.q_proj.weight"), hidden_normed, new_seq, hs, nq * hd);
        let mut k = self.proj_seq(&format!("{p}.k_proj.weight"), hidden_normed, new_seq, hs, nkv * hd);
        let v = self.proj_seq(&format!("{p}.v_proj.weight"), hidden_normed, new_seq, hs, nkv * hd);

        let q_norm = self.w(&format!("{p}.q_norm.weight")).to_vec();
        let k_norm = self.w(&format!("{p}.k_norm.weight")).to_vec();
        for j in 0..new_seq {
            let abs = start_pos + j; // ABSOLUTE position for rope
            {
                let qs = &mut q[j * nq * hd..(j + 1) * nq * hd];
                for h in 0..nq {
                    cpu_rms_norm_inplace(&mut qs[h * hd..(h + 1) * hd], &q_norm, eps);
                }
            }
            {
                let ks = &mut k[j * nkv * hd..(j + 1) * nkv * hd];
                for h in 0..nkv {
                    cpu_rms_norm_inplace(&mut ks[h * hd..(h + 1) * hd], &k_norm, eps);
                }
            }
            let qs = &mut q[j * nq * hd..(j + 1) * nq * hd] as *mut [f32];
            let ks = &mut k[j * nkv * hd..(j + 1) * nkv * hd] as *mut [f32];
            // SAFETY: q and k are distinct Vecs; the two slices never overlap.
            unsafe {
                if is_full {
                    cpu_rope_yarn(
                        &mut *qs, &mut *ks, abs, nq, nkv, hd, cfg.full_rotary_dim,
                        &cfg.yarn_inv_freq, cfg.full_attention_factor,
                    );
                } else {
                    cpu_rope(&mut *qs, &mut *ks, abs, nq, nkv, hd, hd, cfg.sliding_rope_theta);
                }
            }
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        let mut attn_out = vec![0.0f32; new_seq * nq * hd];
        for j in 0..new_seq {
            // Append THIS token's post-rope K/V, then attend the cache (which now
            // includes it) — causal by construction (append-before-attend).
            {
                let cache = self.kv.get_mut(&layer_idx).expect("kv cache for layer");
                cache.append(
                    &k[j * nkv * hd..(j + 1) * nkv * hd],
                    &v[j * nkv * hd..(j + 1) * nkv * hd],
                );
            }
            let cache = self.kv.get(&layer_idx).expect("kv cache for layer");
            let seq_now = cache.seq_len;
            let q_j = &q[j * nq * hd..(j + 1) * nq * hd];
            let o = cpu_sdpa(
                q_j, cache.k_up_to_now(), cache.v_up_to_now(), nq, nkv, hd, seq_now, scale, window,
            );
            attn_out[j * nq * hd..(j + 1) * nq * hd].copy_from_slice(&o);
        }

        // Per-head scalar softplus gate (query-side, unaffected by caching).
        let g = self.proj_seq(&format!("{p}.g_proj.weight"), hidden_normed, new_seq, hs, nq);
        for j in 0..new_seq {
            for h in 0..nq {
                let gate = softplus(g[j * nq + h]);
                let head = &mut attn_out[j * nq * hd + h * hd..j * nq * hd + (h + 1) * hd];
                for d in head.iter_mut() {
                    *d *= gate;
                }
            }
        }

        self.proj_seq(&format!("{p}.o_proj.weight"), &attn_out, new_seq, nq * hd, hs)
    }

    /// One decoder layer over `new_seq` NEW tokens using the KV cache (the
    /// cached twin of `layer_forward` — identical except `attn_cached`).
    pub fn layer_forward_cached(&mut self, hidden: &[f32], new_seq: usize, layer_idx: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        let in_ln = self.w(&format!("model.layers.{layer_idx}.input_layernorm.weight")).to_vec();
        let normed = cpu_rms_norm(hidden, &in_ln, eps);
        let attn = self.attn_cached(&normed, new_seq, layer_idx);
        let h1: Vec<f32> = hidden.iter().zip(&attn).map(|(&a, &b)| a + b).collect();

        let post_ln = self.w(&format!("model.layers.{layer_idx}.post_attention_layernorm.weight")).to_vec();
        let normed2 = cpu_rms_norm(&h1, &post_ln, eps);
        let mlp = if cfg.mlp_only_layers.contains(&layer_idx) {
            self.dense_mlp(&normed2, new_seq, layer_idx)
        } else {
            let mut out = vec![0.0f32; new_seq * hs];
            for t in 0..new_seq {
                let m = self.moe_token(&normed2[t * hs..(t + 1) * hs], layer_idx);
                out[t * hs..(t + 1) * hs].copy_from_slice(&m);
            }
            out
        };
        h1.iter().zip(&mlp).map(|(&a, &b)| a + b).collect()
    }

    /// Finalize the last stage: `model.norm` + `lm_head` on the LAST of `seq`
    /// positions → `[vocab]`; else the un-normed `[seq*hidden]` to ship onward.
    fn finalize(&mut self, hidden: Vec<f32>, seq: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        if self.pp_last {
            let fn_w = self.w("model.norm.weight").to_vec();
            let normed = cpu_rms_norm(&hidden, &fn_w, cfg.rms_norm_eps);
            if self.gpu_weights.contains_key("lm_head.weight")
                || self.host.contains_key("lm_head.weight")
            {
                let last = &normed[(seq - 1) * hs..seq * hs];
                return self.f16_matvec("lm_head.weight", last, hs, cfg.vocab_size);
            }
            return normed;
        }
        hidden
    }

    /// FIRST-stage cache-populating prefill from token ids. Resets the caches,
    /// embeds, runs `[pp_start,pp_end)` with `layer_forward_cached` (which fills
    /// every layer's K/V), returns `[seq*hidden]` (mid) / `[vocab]` (last).
    /// Bit-exact with the stateless `forward` over the same tokens.
    pub fn forward_prefill_tokens(&mut self, tokens: &[u32]) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let seq = tokens.len();
        self.reset_kv();
        let mut hidden = vec![0.0f32; seq * hs];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = self.embed_row(tok);
            hidden[t * hs..(t + 1) * hs].copy_from_slice(&row);
        }
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward_cached(&hidden, seq, li);
        }
        self.finalize(hidden, seq)
    }

    /// MID/LAST-stage cache-populating prefill from a received `[seq*hidden]`.
    /// Bit-exact with the stateless `forward_hidden`.
    pub fn forward_prefill_hidden(&mut self, hidden_in: &[f32], seq: usize) -> Vec<f32> {
        self.reset_kv();
        let mut hidden = hidden_in.to_vec();
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward_cached(&hidden, seq, li);
        }
        self.finalize(hidden, seq)
    }

    /// FIRST-stage single-token decode. Embeds the one new token, runs the
    /// resident layers with `new_seq == 1` against the caches (which advance by
    /// one), returns `[1*hidden]` (mid) / `[vocab]` (last). O(1) weight reads.
    pub fn forward_decode_token(&mut self, tok: u32) -> Vec<f32> {
        let mut hidden = self.embed_row(tok);
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward_cached(&hidden, 1, li);
        }
        self.finalize(hidden, 1)
    }

    /// MID/LAST-stage single-token decode from a received `[1*hidden]`.
    pub fn forward_decode_hidden(&mut self, hidden_in: &[f32]) -> Vec<f32> {
        let mut hidden = hidden_in.to_vec();
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward_cached(&hidden, 1, li);
        }
        self.finalize(hidden, 1)
    }

    // ── RESIDENT 1-CB DECODE FOLD (VLLM_VULKAN_LAGUNA_1CB) ────────────────────
    //
    // Same math as the per-op KV-cache decode above, but the independent matvec
    // DISPATCHES are recorded into shared command buffers and submitted ONCE with
    // a bulk readback, instead of one begin_batch/submit/read_f32_buf per matvec.
    // The per-op path costs ~38 submit+readbacks per MoE layer (5 attn + 10×3
    // expert + 3 shared); the fold costs ~3 (q/k/v/g CB, o_proj CB, and ONE MoE
    // CB that folds all 33 expert/shared dispatches via the GPU `swiglu_f32`
    // glue). Every non-matmul op (qk-norm, YaRN/plain rope, SDPA, softplus gate,
    // router, weighted accumulate, residual, rms_norm) stays on the HOST calling
    // the IDENTICAL crate fns — so the ONLY numeric delta vs the validated per-op
    // decode is `swiglu_f32` computing silu with the GPU `exp` intrinsic instead
    // of libm (last-ulp; cos>=0.999, argmax-exact — see `debug_laguna_1cb`).
    // Requires an engine; with `engine=None` the caller must use the per-op path.

    /// Record ONE BF16-native (f16-resident) matvec `out = in @ W[n,k]^T` into an
    /// open batch `cb` (no submit). `meta` is `(w_ptr, out=n, in=k)` from
    /// `f16_meta`, gathered before the mutable engine borrow.
    #[allow(clippy::too_many_arguments)]
    fn rec_f16_mv(
        eng: &mut ComputeEngine,
        cb: ash::vk::CommandBuffer,
        w_ptr: *const Buffer,
        in_buf: &Buffer,
        out_buf: &Buffer,
        k: usize,
        n: usize,
    ) {
        let (shader, r) = matvec_f16_variant_k(k, n);
        let wg = ((n as u32 + r - 1) / r, 1u32, 1u32);
        let pc = matvec_pc13(k, n);
        unsafe {
            eng.record_to(cb, &shader, &[&*w_ptr, in_buf, out_buf], &pc, wg).unwrap();
        }
    }

    /// Quant-aware sibling of `rec_f16_mv` for the int8-attn lever: records the
    /// plain f16 matvec OR (Q8_0 weight) `mul_mat_vec_q8_0deq_f32_f32` into an
    /// open batch `cb`. Q8_0 needs no side scale buffer (scale is in-block), so
    /// the buffer set and push constants (`matvec_pc13`) are identical to f16 —
    /// only the shader/variant differ. `kind` comes from `mv_meta`.
    #[allow(clippy::too_many_arguments)]
    fn rec_mv(
        eng: &mut ComputeEngine,
        cb: ash::vk::CommandBuffer,
        w_ptr: *const Buffer,
        kind: LagMvKind,
        in_buf: &Buffer,
        out_buf: &Buffer,
        k: usize,
        n: usize,
    ) {
        let (shader, r) = match kind {
            LagMvKind::Q8_0 => matvec_q8_0_variant_k(k, n),
            LagMvKind::F16 => matvec_f16_variant_k(k, n),
        };
        let wg = ((n as u32 + r - 1) / r, 1u32, 1u32);
        let pc = matvec_pc13(k, n);
        unsafe {
            eng.record_to(cb, &shader, &[&*w_ptr, in_buf, out_buf], &pc, wg).unwrap();
        }
    }

    /// GPU YaRN partial-rotary RoPE for one full-attn token's q+k, in-place
    /// (span-fold enabler; `VLLM_VULKAN_LAGUNA_YARN_GPU`). Dispatches the
    /// `rope_neox_f32_f32` yarn_direct path (inv_freq table + mscale) — bit-exact
    /// table vs `laguna::cpu_rope_yarn`, GPU sin/cos last-ulp vs libm. `q` is
    /// `[nq*hd]`, `k` is `[nkv*hd]`; only the low `rotary_dim` dims of each head
    /// rotate, the tail passes through (output seeded from input). One submit.
    #[allow(clippy::too_many_arguments)]
    fn rope_full_yarn_gpu(
        eng: &mut ComputeEngine,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        rotary_dim: usize,
        inv_freq: &[f32],
        mscale: f32,
    ) {
        let posbuf = eng.alloc_host_coherent_storage(4).unwrap();
        posbuf.write(&(pos as i32).to_le_bytes()).unwrap();
        let ff_bytes = f32_slice_to_bytes(inv_freq);
        let ffbuf = eng.alloc_host_coherent_storage(ff_bytes.len() as u64).unwrap();
        ffbuf.write(&ff_bytes).unwrap();
        let idxbuf = eng.alloc_host_coherent_storage(8).unwrap();
        idxbuf.write(&0u64.to_le_bytes()).unwrap();
        let wgy = (((hd / 2) as u32) + 255) / 256;

        let mut rope_one = |eng: &mut ComputeEngine, x: &mut [f32], num_heads: usize| {
            let n = num_heads * hd;
            let inp = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            inp.write(&f32_slice_to_bytes(x)).unwrap();
            let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            out.write(&f32_slice_to_bytes(x)).unwrap(); // seed tail pass-through
            let pc = rope_neox_yarn_pc(num_heads, hd, rotary_dim, mscale);
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, "rope_neox_f32_f32", &[&inp, &posbuf, &ffbuf, &out, &idxbuf], &pc, (num_heads as u32, wgy, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            x.copy_from_slice(&read_f32_buf(&out, n));
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
        };
        rope_one(eng, q, nq);
        rope_one(eng, k, nkv);

        eng.return_to_pool(posbuf);
        eng.return_to_pool(ffbuf);
        eng.return_to_pool(idxbuf);
    }

    /// GPU per-head RMSNorm for `num_heads` contiguous rows of `hd` in `x`
    /// (`VLLM_VULKAN_LAGUNA_GPU_ATTNMATH` qk-norm). One `rms_norm_f32_mul`
    /// dispatch (workgroup-per-head, `do_multiply` weight = `weight[hd]`
    /// broadcast per row) replaces the host `cpu_rms_norm_inplace` loop —
    /// element-wise, so bit-exact modulo the shared-mem tree reduction order vs
    /// the host sequential sum (last-ulp). `x` is `[num_heads*hd]`, normalized in
    /// place. One submit. Batched across ALL new decode rows by the caller (q:
    /// `new_seq*nq` heads, k: `new_seq*nkv` heads).
    fn qk_norm_gpu(eng: &mut ComputeEngine, x: &mut [f32], weight: &[f32], num_heads: usize, hd: usize, eps: f32) {
        let n = num_heads * hd;
        let inp = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        inp.write(&f32_slice_to_bytes(x)).unwrap();
        let wb = eng.alloc_host_coherent_storage((hd * 4) as u64).unwrap();
        wb.write(&f32_slice_to_bytes(weight)).unwrap();
        let pc = rmsnorm_pc(hd, eps);
        let cb = eng.begin_batch().unwrap();
        // in-place: each workgroup reads its whole row (sum) then writes it; rows
        // are disjoint so binding0 == binding2 is race-free (see gemma qk-norm).
        eng.record_to(cb, "rms_norm_f32_mul", &[&inp, &wb, &inp], &pc, (num_heads as u32, 1, 1)).unwrap();
        eng.submit_batch(cb).unwrap();
        x.copy_from_slice(&read_f32_buf(&inp, n));
        eng.return_to_pool(inp);
        eng.return_to_pool(wb);
    }

    /// GPU plain NeoX RoPE for one SLIDING-layer token's q+k, in-place
    /// (`VLLM_VULKAN_LAGUNA_GPU_ATTNMATH` sliding-rope). Full rotary (`rotary_dim
    /// == freq_dim == hd`), θ = `sliding_rope_theta`; dispatches
    /// `rope_neox_f32_f32`'s plain path (`yarn_direct=0`, `has_ff=0`,
    /// `theta_scale = θ^(-2/hd)`) — the exact op sequence of
    /// `model::cpu_rope`/`cpu_rope_with_basis` with `freq_dim==rotary_dim`, GPU
    /// sin/cos last-ulp vs libm. Twin of `rope_full_yarn_gpu` for the sliding
    /// regime. `q` is `[nq*hd]`, `k` is `[nkv*hd]`. One submit per tensor.
    fn rope_sliding_gpu(
        eng: &mut ComputeEngine,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        theta: f32,
    ) {
        let posbuf = eng.alloc_host_coherent_storage(4).unwrap();
        posbuf.write(&(pos as i32).to_le_bytes()).unwrap();
        // ff buffer required by binding 2 but unused (has_ff=0); single 1.0.
        let ffbuf = eng.alloc_host_coherent_storage(4).unwrap();
        ffbuf.write(&1.0f32.to_le_bytes()).unwrap();
        let idxbuf = eng.alloc_host_coherent_storage(8).unwrap();
        idxbuf.write(&0u64.to_le_bytes()).unwrap();
        let wgy = (((hd / 2) as u32) + 255) / 256;

        let mut rope_one = |eng: &mut ComputeEngine, x: &mut [f32], num_heads: usize| {
            let n = num_heads * hd;
            let inp = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            inp.write(&f32_slice_to_bytes(x)).unwrap();
            let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            out.write(&f32_slice_to_bytes(x)).unwrap(); // seed (full-rotary: all dims written)
            let pc = rope_neox_pc(num_heads, hd, hd, hd, theta);
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, "rope_neox_f32_f32", &[&inp, &posbuf, &ffbuf, &out, &idxbuf], &pc, (num_heads as u32, wgy, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            x.copy_from_slice(&read_f32_buf(&out, n));
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
        };
        rope_one(eng, q, nq);
        rope_one(eng, k, nkv);

        eng.return_to_pool(posbuf);
        eng.return_to_pool(ffbuf);
        eng.return_to_pool(idxbuf);
    }

    /// Record ONE routed-expert NVFP4 matvec into an open batch `cb`. `meta` is
    /// the `ExpMeta` from `expert_meta`; `e` selects the expert's slice of the
    /// concatenated per-layer buffer via `packed_off`/`sb_off` (identical to the
    /// per-op `expert_matvec`).
    #[allow(clippy::too_many_arguments)]
    fn rec_expert_mv(
        eng: &mut ComputeEngine,
        cb: ash::vk::CommandBuffer,
        meta: &ExpMeta,
        e: usize,
        in_buf: &Buffer,
        out_buf: &Buffer,
    ) {
        let (w_ptr, s_ptr, n, k, gs, e4m3, global) = *meta;
        let packed_off = e * n * (k / 8);
        let sb_off = e * n * (k / gs);
        unsafe {
            if e4m3 {
                let (shader, r) = laguna_e4m3_expert_shader(k, n, gs);
                let wg = ((n as u32 + r - 1) / r, 1u32, 1u32);
                let pc = matvec_nvfp4_e4m3_pc_off(k, n, gs, packed_off, sb_off, global);
                eng.record_to(cb, &shader, &[&*w_ptr, &*s_ptr, in_buf, out_buf], &pc, wg).unwrap();
            } else {
                let (shader, r) = matvec_nvfp4_variant_k(k, n);
                let wg = ((n as u32 + r - 1) / r, 1u32, 1u32);
                let pc = matvec_mlx4_pc_off(k, n, gs, packed_off, sb_off);
                eng.record_to(cb, &shader, &[&*w_ptr, &*s_ptr, in_buf, out_buf], &pc, wg).unwrap();
            }
        }
    }

    /// FOLDED MoE mixer for a SINGLE token `h` (`[hidden]`, post-norm). Router on
    /// host (as `moe_token`); then the 10 routed experts (gate/up → `swiglu_f32`
    /// → down) AND the ungated shared expert record into ONE command buffer /
    /// ONE submit, reading back only the per-expert `down` + shared `down`, which
    /// the host weight-accumulates exactly like `moe_token`. Bit-exact with
    /// `moe_token` up to `swiglu_f32`'s silu (GPU `exp` vs host libm).
    pub fn moe_token_1cb(&mut self, h: &[f32], layer_idx: usize) -> Vec<f32> {
        // LEVER 4 (VLLM_VULKAN_LAGUNA_CPU_OVERLAP, default OFF): CPU shared ∥
        // routed overlap path (reintroduces a host readback+accum; secondary arm).
        if laguna_cpu_overlap_on() {
            return self.moe_token_1cb_overlap(h, layer_idx, true);
        }
        // LEVER 1 (VLLM_VULKAN_LAGUNA_CBBATCH) folds into moe_token_1cb_full.
        self.moe_token_1cb_full(h, layer_idx, laguna_gpu_accum_on(), laguna_gpu_cbbatch_on())
    }

    /// Number of routed experts selected for `h` at `layer_idx` (== the router
    /// top-k). The gate uses this to CONFIRM the fixed top-10 GPU accumulate
    /// path was exercised (topk == 10) rather than a silent host fallback.
    pub fn moe_router_topk(&self, h: &[f32], layer_idx: usize) -> usize {
        self.host_router_select(h, layer_idx).0.len()
    }

    /// HOST sigmoid-router reference (`nemotron::router_forward` over the host
    /// gate weight + e_score bias) — the numeric oracle the GPU-router gate
    /// (`debug_laguna_gpu_router`) compares against.
    pub fn host_router_select(&self, h: &[f32], layer_idx: usize) -> (Vec<usize>, Vec<f32>) {
        let dims = self.config.router_dims();
        let p = format!("model.layers.{layer_idx}.mlp");
        let router = self.w(&format!("{p}.gate.weight")).to_vec();
        let bias = self.w(&format!("{p}.experts.e_score_correction_bias")).to_vec();
        router_forward(h, &router, &bias, &dims)
    }

    /// GPU sigmoid-router selection: gate matvec + sigmoid + e_score bias +
    /// top-k, all in the `laguna_router` dispatch, reading back only the top-k
    /// `(indices, weights)` (10 idx + 10 weights for Laguna) instead of the host
    /// matvec over the [num_experts, hidden] gate weight. The numeric twin of
    /// `nemotron::router_forward` (Laguna's `n_group == 1` makes the group mask a
    /// no-op → plain top-k). The gate weight + e_score bias upload to GPU lazily
    /// on first use per layer (cached in `gpu_router`). Returns the SAME shape as
    /// `router_forward`; the single-node gate confirms the selected set + weights
    /// match the host path (a router flip would change which experts fire).
    pub fn gpu_router_select(&mut self, h: &[f32], layer_idx: usize) -> (Vec<usize>, Vec<f32>) {
        let dims = self.config.router_dims();
        let ne = dims.n_routed_experts;
        let hs = self.config.hidden_size;
        let top_k = dims.top_k;
        let p = format!("model.layers.{layer_idx}.mlp");
        // Lazy one-time upload of the gate weight + e_score bias for this layer.
        if !self.gpu_router.contains_key(&layer_idx) {
            let gate = self
                .host
                .get(&format!("{p}.gate.weight"))
                .unwrap_or_else(|| panic!("router gate '{p}.gate.weight' missing"))
                .data
                .clone();
            let bias = self
                .host
                .get(&format!("{p}.experts.e_score_correction_bias"))
                .unwrap_or_else(|| panic!("router bias for '{p}' missing"))
                .data
                .clone();
            let eng = self.engine.as_mut().unwrap();
            let gbuf = eng.alloc_host_coherent_storage((gate.len() * 4) as u64).unwrap();
            gbuf.write(&f32_slice_to_bytes(&gate)).unwrap();
            let bbuf = eng.alloc_host_coherent_storage((bias.len() * 4) as u64).unwrap();
            bbuf.write(&f32_slice_to_bytes(&bias)).unwrap();
            self.gpu_router.insert(layer_idx, (gbuf, bbuf));
        }
        let pc = laguna_router_pc(ne, hs, top_k, dims.routed_scaling_factor, dims.norm_topk_prob);
        let eng = self.engine.as_mut().unwrap();
        let inp = eng.alloc_host_coherent_storage((hs * 4) as u64).unwrap();
        inp.write(&f32_slice_to_bytes(h)).unwrap();
        let out = eng.alloc_host_coherent_storage((2 * top_k * 4) as u64).unwrap();
        // `gpu_router` and `engine` are disjoint fields — both borrows are valid.
        let (gbuf, bbuf) = self.gpu_router.get(&layer_idx).unwrap();
        let cb = eng.begin_batch().unwrap();
        eng.record_to(cb, "laguna_router", &[gbuf, bbuf, &inp, &out], &pc, (1, 1, 1))
            .unwrap();
        eng.submit_batch(cb).unwrap();
        let raw = read_f32_buf(&out, 2 * top_k);
        let indices: Vec<usize> = raw[..top_k].iter().map(|&f| f as usize).collect();
        let weights: Vec<f32> = raw[top_k..2 * top_k].to_vec();
        eng.return_to_pool(inp);
        eng.return_to_pool(out);
        (indices, weights)
    }

    /// `moe_token_1cb` with the accumulate-path choice made explicit (the A/B
    /// hook for the single-node bit-exact gate). When `gpu_accum` is true AND
    /// the config is the fixed top-10 Laguna MoE, the score-weighted routed
    /// accumulate + ungated shared add fold into ONE `laguna_moe_accum`
    /// dispatch at the tail of the same command buffer, and only the final
    /// `[hidden]` vector is read back (vs the 10 routed `down` + shared `down`
    /// under the host path). Otherwise the host loop runs (bit-exact baseline
    /// == commit bb33073).
    pub fn moe_token_1cb_accum(&mut self, h: &[f32], layer_idx: usize, gpu_accum: bool) -> Vec<f32> {
        self.moe_token_1cb_full(h, layer_idx, gpu_accum, laguna_gpu_cbbatch_on())
    }

    /// `moe_token_1cb_accum` with BOTH the accumulate-path choice AND the
    /// CB-batch dispatch choice made explicit (the A/B hook for
    /// `debug_laguna_cbbatch`). When `cbbatch` is true AND the fixed top-10 GPU
    /// accumulate path applies (`gpu_accum`, e4m3-resident experts, top_k==10),
    /// the 30 per-expert NVFP4 matvecs collapse to 3 expert-batched dispatches
    /// (`moe_expert_batched`), the per-expert swiglu to one flat dispatch, and
    /// the routed combine to the concatenated-down `laguna_moe_accum_b`. The
    /// batched matvec reduces in the SAME order as the per-expert kernel, so the
    /// batched output is BIT-EXACT with `cbbatch=false`. Router selection (host
    /// `router_forward` or the flagged GPU `laguna_router`, lever 3) runs at the
    /// TOP of both the batched and the per-expert bodies.
    pub fn moe_token_1cb_full(
        &mut self,
        h: &[f32],
        layer_idx: usize,
        gpu_accum: bool,
        cbbatch: bool,
    ) -> Vec<f32> {
        // Fall back to the per-op path if experts aren't resident (no engine).
        if self.engine.is_none() || self.gpu_experts.get(&layer_idx).is_none() {
            return self.moe_token(h, layer_idx);
        }
        // CB-batch path (`VLLM_VULKAN_LAGUNA_CBBATCH`, default OFF) takes
        // precedence when explicitly enabled: requires e4m3-resident experts +
        // the top-10 GPU accumulate; the router (host or GPU-router lever) is
        // re-run inside for the top_k==10 check.
        if cbbatch && gpu_accum {
            if let Some(out) = self.moe_token_1cb_batched(h, layer_idx) {
                return out;
            }
        }
        // Persistent scratch-bank path (`VLLM_VULKAN_LAGUNA_SCRATCH`, default ON):
        // reuse pre-allocated buffers across tokens/layers (bit-identical; see
        // `moe_token_1cb_scratch`).
        if crate::flags::flags_global().laguna_scratch {
            return self.moe_token_1cb_scratch(h, layer_idx, gpu_accum);
        }
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        let dims = cfg.router_dims();
        let p = format!("model.layers.{layer_idx}.mlp");

        // Router selection: GPU (sigmoid + bias + top-k on device, tiny readback)
        // behind the flag (lever 3), else the HOST `router_forward` matvec.
        let (indices, weights) = if laguna_gpu_router_on() {
            self.gpu_router_select(h, layer_idx)
        } else {
            let router = self.w(&format!("{p}.gate.weight")).to_vec();
            let bias = self.w(&format!("{p}.experts.e_score_correction_bias")).to_vec();
            let _t_router = std::time::Instant::now();
            let sel = router_forward(h, &router, &bias, &dims);
            crate::prof_add("lag_host_router", _t_router); // SPIKE instrumentation
            sel
        };
        let topk = indices.len();

        // Gather all weight metas BEFORE the mutable engine borrow (raw ptrs stay
        // valid: gpu_weights/gpu_experts are never mutated during a forward).
        let ex_gate: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 0, e).unwrap()).collect();
        let ex_up: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 1, e).unwrap()).collect();
        let ex_down: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 2, e).unwrap()).collect();
        let (sg_ptr, sg_k, _, _) = self.mv_meta(&format!("{p}.shared_expert.gate_proj.weight")).unwrap();
        let (su_ptr, su_k, _, _) = self.mv_meta(&format!("{p}.shared_expert.up_proj.weight")).unwrap();
        let (sd_ptr, sd_k, _, _) = self.mv_meta(&format!("{p}.shared_expert.down_proj.weight")).unwrap();

        let eng = self.engine.as_mut().unwrap();
        let _t_alloc = std::time::Instant::now(); // profiler: per-token buffer churn (alloc)
        let inp = eng.alloc_host_coherent_storage((hs * 4) as u64).unwrap();
        inp.write(&f32_slice_to_bytes(h)).unwrap();
        // Per-expert scratch banks + shared banks.
        let alloc = |eng: &mut ComputeEngine, m: usize| eng.alloc_host_coherent_storage((m * 4) as u64).unwrap();
        let mut b_gate = Vec::with_capacity(topk);
        let mut b_up = Vec::with_capacity(topk);
        let mut b_mid = Vec::with_capacity(topk);
        let mut b_down = Vec::with_capacity(topk);
        for _ in 0..topk {
            b_gate.push(alloc(eng, inter));
            b_up.push(alloc(eng, inter));
            b_mid.push(alloc(eng, inter));
            b_down.push(alloc(eng, hs));
        }
        let bs_gate = alloc(eng, shared_inter);
        let bs_up = alloc(eng, shared_inter);
        let bs_mid = alloc(eng, shared_inter);
        let bs_down = alloc(eng, hs);
        crate::prof_add("lag_moe_bufalloc", _t_alloc);

        let ts_on = crate::prof_on() && eng.ensure_ts_pool(8); // SPIKE
        let cb = eng.begin_batch().unwrap();
        if ts_on { eng.ts_cmd_reset(cb, 0, 8); eng.ts_cmd_mark(cb, 0, true); } // SPIKE
        // Stage 1: all gate + up projections (routed experts + shared) read INP.
        for kth in 0..topk {
            Self::rec_expert_mv(eng, cb, &ex_gate[kth], indices[kth], &inp, &b_gate[kth]);
            Self::rec_expert_mv(eng, cb, &ex_up[kth], indices[kth], &inp, &b_up[kth]);
        }
        // SPIKE: routed-only gate/up boundary — bottom-of-pipe TS drains the 20
        // routed NVFP4 matvecs BEFORE the 2 shared f16 gate/up matvecs are
        // recorded, so `lag_gpu_moe_gateup_routed` isolates the repack/v1 kernel
        // from the constant shared-f16 cost (the deciding-GB/s target).
        if ts_on { eng.ts_cmd_mark(cb, 5, false); }
        Self::rec_mv(eng, cb, sg_ptr, sg_k, &inp, &bs_gate, hs, shared_inter);
        Self::rec_mv(eng, cb, su_ptr, su_k, &inp, &bs_up, hs, shared_inter);
        eng.record_barrier_to(cb);
        if ts_on { eng.ts_cmd_mark(cb, 1, false); } // SPIKE: after gate/up expert matvecs
        // Stage 2: swiglu = silu(gate)*up for every expert + shared (split mode).
        for kth in 0..topk {
            eng.record_to(cb, "swiglu_f32", &[&b_gate[kth], &b_up[kth], &b_mid[kth]],
                &glu_split_pc(inter), ((inter as u32 + 511) / 512, 1, 1)).unwrap();
        }
        eng.record_to(cb, "swiglu_f32", &[&bs_gate, &bs_up, &bs_mid],
            &glu_split_pc(shared_inter), ((shared_inter as u32 + 511) / 512, 1, 1)).unwrap();
        eng.record_barrier_to(cb);
        if ts_on { eng.ts_cmd_mark(cb, 2, false); } // SPIKE: after swiglu
        // Stage 3: down projections (routed read their b_mid; shared reads bs_mid).
        for kth in 0..topk {
            Self::rec_expert_mv(eng, cb, &ex_down[kth], indices[kth], &b_mid[kth], &b_down[kth]);
        }
        // SPIKE: routed-only down boundary (before the shared f16 down matvec).
        if ts_on { eng.ts_cmd_mark(cb, 6, false); }
        Self::rec_mv(eng, cb, sd_ptr, sd_k, &bs_mid, &bs_down, shared_inter, hs);
        if ts_on { eng.ts_cmd_mark(cb, 3, false); } // SPIKE: after down expert matvecs

        // GPU accumulate: fold the top-10 score-weighted routed combine + the
        // UNGATED shared add into ONE `laguna_moe_accum` dispatch at the tail of
        // the SAME command buffer, so only the final `[hidden]` vector is read
        // back. Only the fixed top-10 Laguna MoE is supported (the shader binds
        // exactly 10 routed `down` buffers); any other `topk` falls back to the
        // host loop below. `weights` already carry norm_topk_prob + ×2.5 scaling.
        let use_gpu_accum = gpu_accum && topk == 10;
        let b_out = if use_gpu_accum {
            let b_out = alloc(eng, hs);
            eng.record_barrier_to(cb);
            let mut binds: Vec<&Buffer> = b_down.iter().collect(); // 10 routed down
            binds.push(&bs_down); // ungated shared down
            binds.push(&b_out); // moe out
            let acc_pc = laguna_moe_accum_pc(hs, &weights);
            eng.record_to(cb, "laguna_moe_accum", &binds, &acc_pc, ((hs as u32 + 255) / 256, 1, 1))
                .unwrap();
            Some(b_out)
        } else {
            None
        };
        if ts_on { eng.ts_cmd_mark(cb, 4, false); } // SPIKE: after accum (before submit)
        eng.submit_batch(cb).unwrap();
        if ts_on { // SPIKE: read GPU-exec ns for each MoE phase
            if let Ok(v) = eng.ts_read_ns(0, 7) {
                crate::prof_add_ns("lag_gpu_moe_gateup", (v[1] - v[0]).max(0.0) as u128);
                crate::prof_add_ns("lag_gpu_moe_swiglu", (v[2] - v[1]).max(0.0) as u128);
                crate::prof_add_ns("lag_gpu_moe_down",   (v[3] - v[2]).max(0.0) as u128);
                crate::prof_add_ns("lag_gpu_moe_accum",  (v[4] - v[3]).max(0.0) as u128);
                // Routed-only (NVFP4 repack/v1) isolated from shared f16:
                crate::prof_add_ns("lag_gpu_moe_gateup_routed", (v[5] - v[0]).max(0.0) as u128);
                crate::prof_add_ns("lag_gpu_moe_down_routed",   (v[6] - v[2]).max(0.0) as u128);
            }
        }

        let _t_rb = std::time::Instant::now(); // SPIKE
        let out = if let Some(ref b_out) = b_out {
            // Single readback of the GPU-accumulated moe output.
            read_f32_buf(b_out, hs)
        } else {
            // Host: weighted accumulate (identical to moe_token) + ungated shared.
            let mut out = vec![0.0f32; hs];
            for kth in 0..topk {
                let dn = read_f32_buf(&b_down[kth], hs);
                let wk = weights[kth];
                for (r, &o) in out.iter_mut().zip(&dn) {
                    *r += o * wk;
                }
            }
            let sd = read_f32_buf(&bs_down, hs);
            for (r, &s) in out.iter_mut().zip(&sd) {
                *r += s;
            }
            out
        };
        crate::prof_add("lag_readback", _t_rb); // SPIKE

        let eng = self.engine.as_mut().unwrap();
        let _t_free = std::time::Instant::now(); // profiler: per-token buffer churn (free)
        for buf in b_gate.into_iter().chain(b_up).chain(b_mid).chain(b_down) {
            eng.return_to_pool(buf);
        }
        for buf in [inp, bs_gate, bs_up, bs_mid, bs_down] {
            eng.return_to_pool(buf);
        }
        if let Some(b_out) = b_out {
            eng.return_to_pool(b_out);
        }
        crate::prof_add("lag_moe_buffree", _t_free);
        out
    }

    /// SCRATCH-BANK twin of `moe_token_1cb_accum` (`VLLM_VULKAN_LAGUNA_SCRATCH`).
    /// Byte-for-byte the same router → gate/up → swiglu → down → accum dispatch
    /// sequence, push-constants and readback, but every GPU buffer comes from the
    /// persistent `LagunaScratch` bank (allocated once, reused across every token
    /// and layer) instead of the per-token pool alloc/free of ~44 buffers; the
    /// router input is written straight into mapped memory (no `f32_slice_to_bytes`
    /// temp Vec) and the router weight is borrowed (no per-token `.to_vec()`).
    /// Bit-identical to the pool path (pure allocation reuse).
    fn moe_token_1cb_scratch(&mut self, h: &[f32], layer_idx: usize, gpu_accum: bool) -> Vec<f32> {
        self.ensure_scratch().expect("laguna scratch alloc");
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        let dims = cfg.router_dims();
        let p = format!("model.layers.{layer_idx}.mlp");

        // Router on host — BORROW the weights (no per-token `.to_vec()`).
        let (indices, weights) = {
            let router = self.w(&format!("{p}.gate.weight"));
            let bias = self.w(&format!("{p}.experts.e_score_correction_bias"));
            router_forward(h, router, bias, &dims)
        };
        let topk = indices.len();

        // Metas gathered before the engine/scratch borrow (raw ptrs stay valid).
        let ex_gate: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 0, e).unwrap()).collect();
        let ex_up: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 1, e).unwrap()).collect();
        let ex_down: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 2, e).unwrap()).collect();
        let (sg_ptr, sg_k, _, _) = self.mv_meta(&format!("{p}.shared_expert.gate_proj.weight")).unwrap();
        let (su_ptr, su_k, _, _) = self.mv_meta(&format!("{p}.shared_expert.up_proj.weight")).unwrap();
        let (sd_ptr, sd_k, _, _) = self.mv_meta(&format!("{p}.shared_expert.down_proj.weight")).unwrap();

        // Own the bank locally so it can coexist with the mutable engine borrow.
        let sc = self.scratch.take().unwrap();
        let eng = self.engine.as_mut().unwrap();
        let _t_alloc = std::time::Instant::now(); // profiler: buffer churn (== ~0, no alloc)
        write_f32_mapped(&sc.inp, h); // direct mapped upload (no temp Vec)
        crate::prof_add("lag_moe_bufalloc", _t_alloc);

        let cb = eng.begin_batch().unwrap();
        // Stage 1: gate + up projections (routed + shared) read INP.
        for kth in 0..topk {
            Self::rec_expert_mv(eng, cb, &ex_gate[kth], indices[kth], &sc.inp, &sc.b_gate[kth]);
            Self::rec_expert_mv(eng, cb, &ex_up[kth], indices[kth], &sc.inp, &sc.b_up[kth]);
        }
        Self::rec_mv(eng, cb, sg_ptr, sg_k, &sc.inp, &sc.bs_gate, hs, shared_inter);
        Self::rec_mv(eng, cb, su_ptr, su_k, &sc.inp, &sc.bs_up, hs, shared_inter);
        eng.record_barrier_to(cb);
        // Stage 2: swiglu = silu(gate)*up.
        for kth in 0..topk {
            eng.record_to(cb, "swiglu_f32", &[&sc.b_gate[kth], &sc.b_up[kth], &sc.b_mid[kth]],
                &glu_split_pc(inter), ((inter as u32 + 511) / 512, 1, 1)).unwrap();
        }
        eng.record_to(cb, "swiglu_f32", &[&sc.bs_gate, &sc.bs_up, &sc.bs_mid],
            &glu_split_pc(shared_inter), ((shared_inter as u32 + 511) / 512, 1, 1)).unwrap();
        eng.record_barrier_to(cb);
        // Stage 3: down projections.
        for kth in 0..topk {
            Self::rec_expert_mv(eng, cb, &ex_down[kth], indices[kth], &sc.b_mid[kth], &sc.b_down[kth]);
        }
        Self::rec_mv(eng, cb, sd_ptr, sd_k, &sc.bs_mid, &sc.bs_down, shared_inter, hs);

        let use_gpu_accum = gpu_accum && topk == 10;
        if use_gpu_accum {
            eng.record_barrier_to(cb);
            let mut binds: Vec<&Buffer> = sc.b_down.iter().take(10).collect();
            binds.push(&sc.bs_down);
            binds.push(&sc.moe_out);
            let acc_pc = laguna_moe_accum_pc(hs, &weights);
            eng.record_to(cb, "laguna_moe_accum", &binds, &acc_pc, ((hs as u32 + 255) / 256, 1, 1))
                .unwrap();
        }
        eng.submit_batch(cb).unwrap();

        let _t_rb = std::time::Instant::now();
        let out = if use_gpu_accum {
            read_f32_mapped(&sc.moe_out, hs).to_vec()
        } else {
            let mut out = vec![0.0f32; hs];
            for kth in 0..topk {
                let dn = read_f32_mapped(&sc.b_down[kth], hs);
                let wk = weights[kth];
                for (r, &o) in out.iter_mut().zip(dn) {
                    *r += o * wk;
                }
            }
            let sd = read_f32_mapped(&sc.bs_down, hs);
            for (r, &s) in out.iter_mut().zip(sd) {
                *r += s;
            }
            out
        };
        crate::prof_add("lag_readback", _t_rb);
        // No free — the bank persists. Restore it (buffree bucket == ~0).
        let _t_free = std::time::Instant::now();
        self.scratch = Some(sc);
        crate::prof_add("lag_moe_buffree", _t_free);
        out
    }

    /// CB-BATCH MoE mixer for a SINGLE token (the `VLLM_VULKAN_LAGUNA_CBBATCH`
    /// fold). Same math as `moe_token_1cb_full`'s per-expert path, but the 10
    /// routed experts' gate/up/down are each computed in ONE expert-batched
    /// dispatch (`mul_mat_vec_laguna_expb_e4m3_f32_f32`, `gl_WorkGroupID.y` =
    /// expert slot) reading the concatenated per-layer packed/scale buffers, the
    /// per-expert swiglu is one flat elementwise dispatch over the concatenated
    /// `[10*inter]` gate/up, and the routed combine is the concatenated-down
    /// `laguna_moe_accum_b`. Per-layer `record_to` recordings drop from ~45 to
    /// ~9 (the dominant short-context host-orchestration cost). Returns `None`
    /// (=> caller falls back to the per-expert path) unless the experts are
    /// e4m3-resident AND the router selected exactly top_k==10 — the fixed shape
    /// the batched shaders bind. BIT-EXACT with the per-expert path (identical
    /// BLOCK_SIZE/NUM_ROWS reduction, slot-order accumulate).
    fn moe_token_1cb_batched(&mut self, h: &[f32], layer_idx: usize) -> Option<Vec<f32>> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        let dims = cfg.router_dims();
        let p = format!("model.layers.{layer_idx}.mlp");

        // Router selection: GPU (lever 3) behind the flag, else HOST router_forward.
        let (indices, weights) = if laguna_gpu_router_on() {
            self.gpu_router_select(h, layer_idx)
        } else {
            let router = self.w(&format!("{p}.gate.weight")).to_vec();
            let bias = self.w(&format!("{p}.experts.e_score_correction_bias")).to_vec();
            router_forward(h, &router, &bias, &dims)
        };
        let topk = indices.len();
        // The batched shaders bind exactly the fixed top-10 Laguna shape.
        if topk != 10 {
            return None;
        }

        // Gather all weight metas BEFORE the mutable engine borrow (raw ptrs stay
        // valid: gpu_weights/gpu_experts are never mutated during a forward).
        let ex_gate: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 0, e).unwrap()).collect();
        let ex_up: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 1, e).unwrap()).collect();
        let ex_down: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 2, e).unwrap()).collect();
        // e4m3-resident experts only (the batched shader is the e4m3 twin).
        if !ex_gate[0].5 {
            return None;
        }
        let gs = ex_gate[0].4;
        // Concatenated per-layer packed/scale buffers are shared across experts;
        // bind slot 0's pointers (identical for all experts of a projection).
        let (gate_pk, gate_sc) = (ex_gate[0].0, ex_gate[0].1);
        let (up_pk, up_sc) = (ex_up[0].0, ex_up[0].1);
        let (down_pk, down_sc) = (ex_down[0].0, ex_down[0].1);
        // Per-slot meta = [eid, floatBitsToUint(global)] × topk. `eid` (=expert
        // index) drives packed_off/sb_off in-shader; `global` is that expert's
        // per-tensor `.weight_scale_2` (ExpMeta.6), read per projection.
        let meta_bytes = |ex: &[ExpMeta]| -> Vec<u8> {
            let mut v = Vec::with_capacity(topk * 8);
            for (kth, &eid) in indices.iter().enumerate() {
                v.extend_from_slice(&(eid as u32).to_le_bytes());
                v.extend_from_slice(&ex[kth].6.to_bits().to_le_bytes());
            }
            v
        };
        let mg = meta_bytes(&ex_gate);
        let mu = meta_bytes(&ex_up);
        let md = meta_bytes(&ex_down);

        // FORMAT-AWARE shared expert (mv_meta → rec_mv), NOT the hardcoded f16
        // path: the int8-attn lever (VLLM_VULKAN_LAGUNA_INT8_ATTN, part of the GO
        // stack) drops the shared-expert gate/up/down weights to Q8_0-resident.
        // `f16_meta` returns the buffer pointer regardless of quant, so recording
        // an f16 matvec over the Q8_0 bytes reads garbage → the CB-batch path
        // degenerated to `[0]×N`. `mv_meta` carries the dispatch KIND so rec_mv
        // picks `mul_mat_vec_q8_0deq_f32_f32` under int8 (bit-identical to the
        // per-op / scratch 1CB shared path). When int8 is OFF, `mv_meta` yields
        // `LagMvKind::F16` and rec_mv records the SAME f16 matvec rec_f16_mv did
        // — the non-int8 path is byte-unchanged. Mirrors the moe_token_1cb_full /
        // moe_token_1cb_scratch shared handling + the cpu_overlap int8 fix.
        let (sg_ptr, sg_kind, _, _) = self.mv_meta(&format!("{p}.shared_expert.gate_proj.weight")).unwrap();
        let (su_ptr, su_kind, _, _) = self.mv_meta(&format!("{p}.shared_expert.up_proj.weight")).unwrap();
        let (sd_ptr, sd_kind, _, _) = self.mv_meta(&format!("{p}.shared_expert.down_proj.weight")).unwrap();

        let eng = self.engine.as_mut().unwrap();
        let inp = eng.alloc_host_coherent_storage((hs * 4) as u64).unwrap();
        inp.write(&f32_slice_to_bytes(h)).unwrap();
        let alloc = |eng: &mut ComputeEngine, m: usize| eng.alloc_host_coherent_storage((m * 4) as u64).unwrap();
        // Concatenated per-expert scratch: [topk * inter] / [topk * hs].
        let b_gate = alloc(eng, topk * inter);
        let b_up = alloc(eng, topk * inter);
        let b_mid = alloc(eng, topk * inter);
        let b_down = alloc(eng, topk * hs);
        // Small meta buffers (uint[2*topk]).
        let mk_meta = |eng: &mut ComputeEngine, bytes: &[u8]| {
            let b = eng.alloc_host_coherent_storage(bytes.len() as u64).unwrap();
            b.write(bytes).unwrap();
            b
        };
        let meta_gate = mk_meta(eng, &mg);
        let meta_up = mk_meta(eng, &mu);
        let meta_down = mk_meta(eng, &md);
        // Shared expert banks (f16 path, unchanged).
        let bs_gate = alloc(eng, shared_inter);
        let bs_up = alloc(eng, shared_inter);
        let bs_mid = alloc(eng, shared_inter);
        let bs_down = alloc(eng, hs);
        let b_out = alloc(eng, hs);

        // Batched-matvec dispatch geometry (mirror the per-op r-tier pick so the
        // reduction order is identical → bit-exact).
        let (gate_shader, gr) = laguna_expb_e4m3_variant(inter);
        let (up_shader, ur) = laguna_expb_e4m3_variant(inter);
        let (down_shader, dr) = laguna_expb_e4m3_variant(hs);
        let gate_wg = ((inter as u32 + gr - 1) / gr, topk as u32, 1);
        let up_wg = ((inter as u32 + ur - 1) / ur, topk as u32, 1);
        let down_wg = ((hs as u32 + dr - 1) / dr, topk as u32, 1);
        // gate/up read the SHARED input (x_stride=0); down reads concatenated
        // b_mid (x_stride=inter).
        let gate_pc = laguna_expb_e4m3_pc(hs, inter, gs, 0);
        let up_pc = laguna_expb_e4m3_pc(hs, inter, gs, 0);
        let down_pc = laguna_expb_e4m3_pc(inter, hs, gs, inter);

        let cb = eng.begin_batch().unwrap();
        // Stage 1: batched gate + up over all experts (2 dispatches) + shared.
        unsafe {
            eng.record_to(cb, &gate_shader, &[&*gate_pk, &*gate_sc, &inp, &b_gate, &meta_gate], &gate_pc, gate_wg).unwrap();
            eng.record_to(cb, &up_shader, &[&*up_pk, &*up_sc, &inp, &b_up, &meta_up], &up_pc, up_wg).unwrap();
        }
        Self::rec_mv(eng, cb, sg_ptr, sg_kind, &inp, &bs_gate, hs, shared_inter);
        Self::rec_mv(eng, cb, su_ptr, su_kind, &inp, &bs_up, hs, shared_inter);
        eng.record_barrier_to(cb);
        // Stage 2: one flat swiglu over the concatenated [topk*inter] gate/up
        // (split mode is elementwise → bit-identical to the per-expert dispatch)
        // + shared swiglu.
        let all = topk * inter;
        eng.record_to(cb, "swiglu_f32", &[&b_gate, &b_up, &b_mid],
            &glu_split_pc(all), ((all as u32 + 511) / 512, 1, 1)).unwrap();
        eng.record_to(cb, "swiglu_f32", &[&bs_gate, &bs_up, &bs_mid],
            &glu_split_pc(shared_inter), ((shared_inter as u32 + 511) / 512, 1, 1)).unwrap();
        eng.record_barrier_to(cb);
        // Stage 3: batched down over all experts (1 dispatch) + shared down.
        unsafe {
            eng.record_to(cb, &down_shader, &[&*down_pk, &*down_sc, &b_mid, &b_down, &meta_down], &down_pc, down_wg).unwrap();
        }
        Self::rec_mv(eng, cb, sd_ptr, sd_kind, &bs_mid, &bs_down, shared_inter, hs);
        eng.record_barrier_to(cb);
        // Tail: concatenated-down accumulate (top-10 weighted routed + ungated
        // shared) → single [hidden] readback.
        let acc_pc = laguna_moe_accum_pc(hs, &weights);
        eng.record_to(cb, "laguna_moe_accum_b", &[&b_down, &bs_down, &b_out], &acc_pc, ((hs as u32 + 255) / 256, 1, 1)).unwrap();
        eng.submit_batch(cb).unwrap();

        let out = read_f32_buf(&b_out, hs);
        for buf in [inp, b_gate, b_up, b_mid, b_down, meta_gate, meta_up, meta_down,
                    bs_gate, bs_up, bs_mid, bs_down, b_out] {
            eng.return_to_pool(buf);
        }
        Some(out)
    }

    /// CPU shared-expert ∥ routed-expert OVERLAP for a SINGLE token `h`
    /// (`[hidden]`, post-norm). LEVER 4: the routed top-k experts (gate/up →
    /// `swiglu_f32` → down) record into ONE command buffer submitted NON-BLOCKING
    /// (`submit_batch_async`); while the GPU runs it, the DATA-INDEPENDENT shared
    /// expert (depends only on `h`, not routed output) computes on the host rayon
    /// pool (`expert_swiglu_par`, 6 threads); then `wait_batch` syncs before the
    /// host reads the routed `down` buffers, weight-accumulates them, and adds the
    /// ungated shared vector.
    ///
    /// `overlap=true` submits async + computes shared concurrently; `overlap=false`
    /// submits BLOCKING then computes shared sequentially. BOTH use the identical
    /// rayon shared math + identical host weighted-accumulate order, so their
    /// outputs are BIT-EXACT (the overlap changes only WHEN the shared branch
    /// runs). That equality is the single-node gate (`debug_laguna_cpuoverlap`);
    /// a nonzero delta would be a sync/race bug. Only the shared branch's numeric
    /// source differs from `moe_token_1cb_accum` (host f32 rayon vs GPU f16), the
    /// same class of delta as the existing 1-CB silu-vs-libm gap (cos≥0.999).
    pub fn moe_token_1cb_overlap(&mut self, h: &[f32], layer_idx: usize, overlap: bool) -> Vec<f32> {
        // LEVER CONFLICT GUARD: the int8-attn lever (VLLM_VULKAN_LAGUNA_INT8_ATTN)
        // drops the shared-expert gate/up/down weights to Q8_0-resident with NO
        // host-f32 copy. Phase B below runs the shared expert on the CPU rayon
        // pool via `self.w(...)`, which reads those host-f32 tensors — under int8
        // they are absent, so it panics ("host tensor '…shared_expert.gate_proj.
        // weight' missing"). When int8 is active there is no host shared path to
        // overlap, so fall back to the int8-aware non-overlap 1cb path, which
        // records the Q8_0 shared matvec on the GPU (mv_meta → rec_mv). The
        // CPU/routed overlap is perf-flat (NO-GO), so nothing is lost; the result
        // is bit-identical to the GO (argmax + native_hop + int8) stack. The
        // default path (int8 OFF, or overlap-without-int8) is byte-unchanged.
        if crate::flags::flags_global().laguna_int8_attn {
            return self.moe_token_1cb_full(h, layer_idx, laguna_gpu_accum_on(), laguna_gpu_cbbatch_on());
        }
        // Fall back to the per-op path if experts aren't resident (no engine).
        if self.engine.is_none() || self.gpu_experts.get(&layer_idx).is_none() {
            return self.moe_token(h, layer_idx);
        }
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        let dims = cfg.router_dims();
        let p = format!("model.layers.{layer_idx}.mlp");

        // Router selection: GPU (lever 3) behind the flag, else HOST router_forward.
        let (indices, weights) = if laguna_gpu_router_on() {
            self.gpu_router_select(h, layer_idx)
        } else {
            let router = self.w(&format!("{p}.gate.weight")).to_vec();
            let bias = self.w(&format!("{p}.experts.e_score_correction_bias")).to_vec();
            router_forward(h, &router, &bias, &dims)
        };
        let topk = indices.len();

        // Gather routed-expert weight metas BEFORE the mutable engine borrow (raw
        // ptrs stay valid: gpu_experts is never mutated during a forward). The
        // shared expert is NOT recorded on the GPU here — it runs on the host.
        let ex_gate: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 0, e).unwrap()).collect();
        let ex_up: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 1, e).unwrap()).collect();
        let ex_down: Vec<ExpMeta> = indices.iter().map(|&e| self.expert_meta(layer_idx, 2, e).unwrap()).collect();

        // ── Phase A: record the ROUTED-ONLY command buffer and submit it. When
        // `overlap`, submit NON-BLOCKING and drop the engine borrow so the host
        // shared expert (which reads `self.host` weights) can run concurrently.
        let inp;
        let mut b_gate: Vec<Buffer> = Vec::with_capacity(topk);
        let mut b_up: Vec<Buffer> = Vec::with_capacity(topk);
        let mut b_mid: Vec<Buffer> = Vec::with_capacity(topk);
        let mut b_down: Vec<Buffer> = Vec::with_capacity(topk);
        {
            let eng = self.engine.as_mut().unwrap();
            inp = eng.alloc_host_coherent_storage((hs * 4) as u64).unwrap();
            inp.write(&f32_slice_to_bytes(h)).unwrap();
            let alloc = |eng: &mut ComputeEngine, m: usize| eng.alloc_host_coherent_storage((m * 4) as u64).unwrap();
            for _ in 0..topk {
                b_gate.push(alloc(eng, inter));
                b_up.push(alloc(eng, inter));
                b_mid.push(alloc(eng, inter));
                b_down.push(alloc(eng, hs));
            }
            let cb = eng.begin_batch().unwrap();
            // Stage 1: gate + up projections (routed experts read INP).
            for kth in 0..topk {
                Self::rec_expert_mv(eng, cb, &ex_gate[kth], indices[kth], &inp, &b_gate[kth]);
                Self::rec_expert_mv(eng, cb, &ex_up[kth], indices[kth], &inp, &b_up[kth]);
            }
            eng.record_barrier_to(cb);
            // Stage 2: swiglu = silu(gate)*up per routed expert (split mode).
            for kth in 0..topk {
                eng.record_to(cb, "swiglu_f32", &[&b_gate[kth], &b_up[kth], &b_mid[kth]],
                    &glu_split_pc(inter), ((inter as u32 + 511) / 512, 1, 1)).unwrap();
            }
            eng.record_barrier_to(cb);
            // Stage 3: down projections (routed read their b_mid).
            for kth in 0..topk {
                Self::rec_expert_mv(eng, cb, &ex_down[kth], indices[kth], &b_mid[kth], &b_down[kth]);
            }
            if overlap {
                eng.submit_batch_async(cb).unwrap();
            } else {
                eng.submit_batch(cb).unwrap();
            }
        } // engine borrow ends here — the async CB is in flight.

        // ── Phase B: shared expert on the HOST rayon pool. Reads only `h` +
        // resident host f32 shared weights (`self.host`), never any GPU output,
        // so it is safe to run while the routed CB is in flight. `install` is
        // synchronous — the borrows of `self.host` live only for its duration.
        let shared = {
            let sg = self.w(&format!("{p}.shared_expert.gate_proj.weight"));
            let su = self.w(&format!("{p}.shared_expert.up_proj.weight"));
            let sd = self.w(&format!("{p}.shared_expert.down_proj.weight"));
            laguna_overlap_pool()
                .install(|| crate::moe::expert_swiglu_par(h, sg, su, sd, hs, shared_inter))
        };

        // ── Phase C: sync (async path only — the blocking submit already waited),
        // then read back the routed down buffers and combine.
        if overlap {
            self.engine.as_mut().unwrap().wait_batch().unwrap();
        }
        // Host weighted-accumulate the routed experts (identical to moe_token:
        // weights carry norm_topk_prob + routed scaling), then add the UNGATED
        // shared vector. Reduction order is fixed and shared-timing-independent.
        let mut out = vec![0.0f32; hs];
        for kth in 0..topk {
            let dn = read_f32_buf(&b_down[kth], hs);
            let wk = weights[kth];
            for (r, &o) in out.iter_mut().zip(&dn) {
                *r += o * wk;
            }
        }
        for (r, &s) in out.iter_mut().zip(&shared) {
            *r += s;
        }

        let eng = self.engine.as_mut().unwrap();
        for buf in b_gate.into_iter().chain(b_up).chain(b_mid).chain(b_down) {
            eng.return_to_pool(buf);
        }
        eng.return_to_pool(inp);
        out
    }

    /// FOLDED dense SwiGLU MLP (layer 0) for `[seq, hidden]`. gate/up →
    /// `swiglu_f32` → down in ONE command buffer per row.
    pub fn dense_mlp_1cb(&mut self, x: &[f32], seq: usize, layer_idx: usize) -> Vec<f32> {
        if self.engine.is_none() {
            return self.dense_mlp(x, seq, layer_idx);
        }
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let p = format!("model.layers.{layer_idx}.mlp");
        let (gp, _, _) = self.f16_meta(&format!("{p}.gate_proj.weight")).unwrap();
        let (up, _, _) = self.f16_meta(&format!("{p}.up_proj.weight")).unwrap();
        let (dp, _, _) = self.f16_meta(&format!("{p}.down_proj.weight")).unwrap();

        let eng = self.engine.as_mut().unwrap();
        let mut out = vec![0.0f32; seq * hs];
        for t in 0..seq {
            let inp = eng.alloc_host_coherent_storage((hs * 4) as u64).unwrap();
            inp.write(&f32_slice_to_bytes(&x[t * hs..(t + 1) * hs])).unwrap();
            let b_gate = eng.alloc_host_coherent_storage((inter * 4) as u64).unwrap();
            let b_up = eng.alloc_host_coherent_storage((inter * 4) as u64).unwrap();
            let b_mid = eng.alloc_host_coherent_storage((inter * 4) as u64).unwrap();
            let b_out = eng.alloc_host_coherent_storage((hs * 4) as u64).unwrap();
            let cb = eng.begin_batch().unwrap();
            Self::rec_f16_mv(eng, cb, gp, &inp, &b_gate, hs, inter);
            Self::rec_f16_mv(eng, cb, up, &inp, &b_up, hs, inter);
            eng.record_barrier_to(cb);
            eng.record_to(cb, "swiglu_f32", &[&b_gate, &b_up, &b_mid],
                &glu_split_pc(inter), ((inter as u32 + 511) / 512, 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            Self::rec_f16_mv(eng, cb, dp, &b_mid, &b_out, inter, hs);
            eng.submit_batch(cb).unwrap();
            out[t * hs..(t + 1) * hs].copy_from_slice(&read_f32_buf(&b_out, hs));
            for buf in [inp, b_gate, b_up, b_mid, b_out] {
                eng.return_to_pool(buf);
            }
        }
        out
    }

    /// FOLDED KV-cached gated GQA attention (decode). q/k/v/g projections batch
    /// into ONE command buffer; qk-norm + YaRN/plain rope + cache append + SDPA +
    /// per-head softplus gate run on the HOST (byte-identical to `attn_cached`);
    /// o_proj is a second CB. `new_seq` rows are recorded into the same CBs.
    pub fn attn_cached_1cb(&mut self, hidden_normed: &[f32], new_seq: usize, layer_idx: usize) -> Vec<f32> {
        if self.engine.is_none() {
            return self.attn_cached(hidden_normed, new_seq, layer_idx);
        }
        // Persistent scratch-bank decode path (`VLLM_VULKAN_LAGUNA_SCRATCH`): only
        // the single-token decode (`new_seq == 1`) — the scratch banks are sized
        // for one row; the multi-row prefill keeps the per-op alloc path.
        if crate::flags::flags_global().laguna_scratch && new_seq == 1 {
            return self.attn_cached_1cb_scratch(hidden_normed, layer_idx);
        }
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nkv = cfg.num_key_value_heads;
        let nq = cfg.num_attention_heads_per_layer[layer_idx];
        let is_full = cfg.layer_is_full[layer_idx];
        let eps = cfg.rms_norm_eps;
        let p = format!("model.layers.{layer_idx}.self_attn");

        // The 1-CB path is backed by the DEVICE-RESIDENT K/V plane (span-fold
        // arena); engine is Some here (checked above) so the plane exists.
        let start_pos = self
            .kv_res
            .get(&layer_idx)
            .map(|c| c.seq_len)
            .or_else(|| self.kv.get(&layer_idx).map(|c| c.seq_len))
            .unwrap_or(0);

        let (qw, qw_k, _, _) = self.mv_meta(&format!("{p}.q_proj.weight")).unwrap();
        let (kw, kw_k, _, _) = self.mv_meta(&format!("{p}.k_proj.weight")).unwrap();
        let (vw, vw_k, _, _) = self.mv_meta(&format!("{p}.v_proj.weight")).unwrap();
        let (gw, gw_k, _, _) = self.mv_meta(&format!("{p}.g_proj.weight")).unwrap();

        // CB1: q/k/v/g projections for all new rows (each reads its own row).
        let eng = self.engine.as_mut().unwrap();
        let ts_on = crate::prof_on() && eng.ensure_ts_pool(8); // SPIKE
        let cb = eng.begin_batch().unwrap();
        if ts_on { eng.ts_cmd_reset(cb, 0, 8); eng.ts_cmd_mark(cb, 0, true); } // SPIKE
        let mut inps = Vec::with_capacity(new_seq);
        let mut bq = Vec::with_capacity(new_seq);
        let mut bk = Vec::with_capacity(new_seq);
        let mut bv = Vec::with_capacity(new_seq);
        let mut bg = Vec::with_capacity(new_seq);
        for j in 0..new_seq {
            let inp = eng.alloc_host_coherent_storage((hs * 4) as u64).unwrap();
            inp.write(&f32_slice_to_bytes(&hidden_normed[j * hs..(j + 1) * hs])).unwrap();
            let q = eng.alloc_host_coherent_storage((nq * hd * 4) as u64).unwrap();
            let k = eng.alloc_host_coherent_storage((nkv * hd * 4) as u64).unwrap();
            let v = eng.alloc_host_coherent_storage((nkv * hd * 4) as u64).unwrap();
            let g = eng.alloc_host_coherent_storage((nq * 4) as u64).unwrap();
            Self::rec_mv(eng, cb, qw, qw_k, &inp, &q, hs, nq * hd);
            Self::rec_mv(eng, cb, kw, kw_k, &inp, &k, hs, nkv * hd);
            Self::rec_mv(eng, cb, vw, vw_k, &inp, &v, hs, nkv * hd);
            Self::rec_mv(eng, cb, gw, gw_k, &inp, &g, hs, nq);
            inps.push(inp);
            bq.push(q);
            bk.push(k);
            bv.push(v);
            bg.push(g);
        }
        if ts_on { eng.ts_cmd_mark(cb, 1, false); } // SPIKE: after qkvg matvecs
        eng.submit_batch(cb).unwrap();
        if ts_on { if let Ok(v) = eng.ts_read_ns(0, 2) { // SPIKE
            crate::prof_add_ns("lag_gpu_attnproj", (v[1] - v[0]).max(0.0) as u128); } }
        let _t_rb = std::time::Instant::now(); // SPIKE
        let mut q: Vec<f32> = Vec::with_capacity(new_seq * nq * hd);
        let mut k: Vec<f32> = Vec::with_capacity(new_seq * nkv * hd);
        let mut v: Vec<f32> = Vec::with_capacity(new_seq * nkv * hd);
        let mut g: Vec<f32> = Vec::with_capacity(new_seq * nq);
        for j in 0..new_seq {
            q.extend_from_slice(&read_f32_buf(&bq[j], nq * hd));
            k.extend_from_slice(&read_f32_buf(&bk[j], nkv * hd));
            v.extend_from_slice(&read_f32_buf(&bv[j], nkv * hd));
            g.extend_from_slice(&read_f32_buf(&bg[j], nq));
        }
        crate::prof_add("lag_readback", _t_rb); // SPIKE
        for buf in inps.into_iter().chain(bq).chain(bk).chain(bv).chain(bg) {
            eng.return_to_pool(buf);
        }

        // qk-norm + rope + cache append + SDPA + softplus gate (IDENTICAL to
        // `attn_cached`).
        let q_norm = self.w(&format!("{p}.q_norm.weight")).to_vec();
        let k_norm = self.w(&format!("{p}.k_norm.weight")).to_vec();
        let _t_qkrope = std::time::Instant::now(); // SPIKE
        let gpu_attnmath = crate::flags::flags_global().laguna_gpu_attnmath;
        if gpu_attnmath {
            // LEVER 2 (VLLM_VULKAN_LAGUNA_GPU_ATTNMATH): the two remaining host
            // attention-math ops on the GPU. qk-norm is batched over ALL new rows
            // (position-independent, so norm-all-then-rope is bit-identical to the
            // per-token interleave below); rope stays per-token (per-position
            // angle). Full-attn uses GPU-YaRN, sliding uses GPU plain rope.
            let eng = self.engine.as_mut().unwrap();
            Self::qk_norm_gpu(eng, &mut q, &q_norm, new_seq * nq, hd, eps);
            Self::qk_norm_gpu(eng, &mut k, &k_norm, new_seq * nkv, hd, eps);
            for j in 0..new_seq {
                let abs = start_pos + j;
                let qs = &mut q[j * nq * hd..(j + 1) * nq * hd] as *mut [f32];
                let ks = &mut k[j * nkv * hd..(j + 1) * nkv * hd] as *mut [f32];
                let eng = self.engine.as_mut().unwrap();
                unsafe {
                    if is_full {
                        Self::rope_full_yarn_gpu(
                            eng, &mut *qs, &mut *ks, abs, nq, nkv, hd,
                            cfg.full_rotary_dim, &cfg.yarn_inv_freq, cfg.full_attention_factor,
                        );
                    } else {
                        Self::rope_sliding_gpu(
                            eng, &mut *qs, &mut *ks, abs, nq, nkv, hd, cfg.sliding_rope_theta,
                        );
                    }
                }
            }
        } else {
        for j in 0..new_seq {
            let abs = start_pos + j;
            {
                let qs = &mut q[j * nq * hd..(j + 1) * nq * hd];
                for h in 0..nq {
                    cpu_rms_norm_inplace(&mut qs[h * hd..(h + 1) * hd], &q_norm, eps);
                }
            }
            {
                let ks = &mut k[j * nkv * hd..(j + 1) * nkv * hd];
                for h in 0..nkv {
                    cpu_rms_norm_inplace(&mut ks[h * hd..(h + 1) * hd], &k_norm, eps);
                }
            }
            let qs = &mut q[j * nq * hd..(j + 1) * nq * hd] as *mut [f32];
            let ks = &mut k[j * nkv * hd..(j + 1) * nkv * hd] as *mut [f32];
            unsafe {
                if is_full {
                    // Full-attn: partial-rotary YaRN. GPU (yarn_direct) when
                    // VLLM_VULKAN_LAGUNA_YARN_GPU is set (span-fold enabler),
                    // else host cpu_rope_yarn. Bit-exact table; GPU sin/cos
                    // last-ulp vs libm.
                    if crate::flags::flags_global().laguna_yarn_gpu {
                        let eng = self.engine.as_mut().unwrap();
                        Self::rope_full_yarn_gpu(
                            eng, &mut *qs, &mut *ks, abs, nq, nkv, hd,
                            cfg.full_rotary_dim, &cfg.yarn_inv_freq, cfg.full_attention_factor,
                        );
                    } else {
                        cpu_rope_yarn(
                            &mut *qs, &mut *ks, abs, nq, nkv, hd, cfg.full_rotary_dim,
                            &cfg.yarn_inv_freq, cfg.full_attention_factor,
                        );
                    }
                } else {
                    cpu_rope(&mut *qs, &mut *ks, abs, nq, nkv, hd, hd, cfg.sliding_rope_theta);
                }
            }
        }
        }

        crate::prof_add("lag_host_qkrope", _t_qkrope); // SPIKE
        let scale = 1.0 / (hd as f32).sqrt();
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        let mut attn_out = vec![0.0f32; new_seq * nq * hd];

        // Phase-0 per-layer KV sizing: a sliding layer whose plane was allocated
        // as a `window`-sized RING (capacity < max_seq, VLLM_VULKAN_LAGUNA_KV_RING)
        // holds only the last `capacity` positions. `ring_capacity` (>0) tells the
        // shader/reader to resolve slot `pos % capacity`; full/YaRN layers keep the
        // full absolute plane (ring_capacity == 0, byte-identical).
        let ring_capacity = self.kv_res.get(&layer_idx).expect("resident KV plane for layer").ring_capacity();
        let is_ring = ring_capacity != 0;
        let use_gpu_sdpa = crate::flags::flags_global().laguna_gpu_sdpa;

        if is_ring {
            // RING (sliding) path — interleave append→attend PER ROW. Because the
            // plane retains only the last `cap` positions, a batched
            // append-all-then-dispatch would overwrite an earlier query row's
            // window before it is read; interleaving guarantees that at row j's
            // attend the ring holds exactly [max(0, pos_j+1-cap), pos_j] = row j's
            // full sliding window, none overwritten. Bit-exact for prefill (any
            // length, many wraps) AND single-token decode. The window read uses
            // `windowed_view` (compacted ascending) with window=None, which is
            // bit-for-bit identical to a full plane read with Some(window) (see
            // ResidentKvPlane::windowed_view). GPU sub-path submits per row (ring
            // appends cannot batch across rows) — DEFERRED to the on-node gate.
            let _t_hs = std::time::Instant::now(); // SPIKE
            for j in 0..new_seq {
                {
                    let plane = self.kv_res.get_mut(&layer_idx).expect("resident KV plane for layer");
                    plane.append(
                        &k[j * nkv * hd..(j + 1) * nkv * hd],
                        &v[j * nkv * hd..(j + 1) * nkv * hd],
                    );
                }
                let seq_now = self.kv_res.get(&layer_idx).unwrap().seq_len;
                let window_start = seq_now.saturating_sub(cfg.sliding_window);
                let q_j_off = j * nq * hd;
                if use_gpu_sdpa {
                    let plane = self.kv_res.get(&layer_idx).unwrap();
                    let kbuf = plane.k_buf() as *const Buffer;
                    let vbuf = plane.v_buf() as *const Buffer;
                    let eng = self.engine.as_mut().unwrap();
                    let cb = eng.begin_batch().unwrap();
                    let qb = eng.alloc_host_coherent_storage((nq * hd * 4) as u64).unwrap();
                    qb.write(&f32_slice_to_bytes(&q[q_j_off..q_j_off + nq * hd])).unwrap();
                    let ob = eng.alloc_host_coherent_storage((nq * hd * 4) as u64).unwrap();
                    let pc = laguna_gpu_sdpa_pc(seq_now, nq, nkv, hd, scale, window_start, ring_capacity);
                    // SAFETY: kbuf/vbuf point into kv_res (not mutated between the
                    // append above and this per-row submit); the mapped ring holds
                    // row j's full window at this point.
                    let kb: &Buffer = unsafe { &*kbuf };
                    let vb: &Buffer = unsafe { &*vbuf };
                    eng.record_to(cb, "laguna_gpu_sdpa", &[&qb, kb, vb, &ob], &pc, (nq as u32, 1, 1))
                        .unwrap();
                    eng.submit_batch(cb).unwrap();
                    attn_out[q_j_off..q_j_off + nq * hd].copy_from_slice(&read_f32_buf(&ob, nq * hd));
                    eng.return_to_pool(qb);
                    eng.return_to_pool(ob);
                } else {
                    let plane = self.kv_res.get(&layer_idx).unwrap();
                    let (kw, vw, vlen) = plane.windowed_view(cfg.sliding_window);
                    let q_j = &q[q_j_off..q_j_off + nq * hd];
                    let o = cpu_sdpa(q_j, &kw, &vw, nq, nkv, hd, vlen, scale, None);
                    attn_out[q_j_off..q_j_off + nq * hd].copy_from_slice(&o);
                }
            }
            crate::prof_add("lag_host_sdpa", _t_hs); // SPIKE
        } else {
        // FULL (absolute) plane — byte-identical to the pre-ring path.
        // Append EVERY token's post-rope K/V into the DEVICE-RESIDENT plane first
        // (on UMA the host write to the mapped coherent buffer IS the on-GPU
        // append). Record each token's post-append seq_len as its causal read
        // bound — token j attends only [.., start_pos+j], never future rows, so a
        // batched dispatch (all appends visible before submit) is still causal.
        let mut seq_at = Vec::with_capacity(new_seq);
        for j in 0..new_seq {
            let plane = self.kv_res.get_mut(&layer_idx).expect("resident KV plane for layer");
            plane.append(
                &k[j * nkv * hd..(j + 1) * nkv * hd],
                &v[j * nkv * hd..(j + 1) * nkv * hd],
            );
            seq_at.push(plane.seq_len);
        }

        if use_gpu_sdpa {
            // GPU decode-SDPA: the `laguna_gpu_sdpa` subgroup kernel reads the
            // resident planes directly (no readback), one wave64 subgroup per q
            // head, in ONE command buffer for all `new_seq` rows. `window_start`
            // replicates cpu_sdpa's absolute-position clamp exactly.
            let plane = self.kv_res.get(&layer_idx).expect("resident KV plane for layer");
            // Raw ptrs: kv_res is NOT mutated between here and submit (appends are
            // done), so the plane buffers are stable across the engine mut-borrow.
            let kbuf = plane.k_buf() as *const Buffer;
            let vbuf = plane.v_buf() as *const Buffer;
            let eng = self.engine.as_mut().unwrap();
            let ts_on = crate::prof_on() && eng.ensure_ts_pool(8); // SPIKE
            let cb = eng.begin_batch().unwrap();
            if ts_on { eng.ts_cmd_reset(cb, 0, 8); eng.ts_cmd_mark(cb, 0, true); } // SPIKE
            let mut qbufs = Vec::with_capacity(new_seq);
            let mut obufs = Vec::with_capacity(new_seq);
            for j in 0..new_seq {
                let seq_now = seq_at[j];
                let window_start = if is_full {
                    0
                } else {
                    seq_now.saturating_sub(cfg.sliding_window)
                };
                let qb = eng.alloc_host_coherent_storage((nq * hd * 4) as u64).unwrap();
                qb.write(&f32_slice_to_bytes(&q[j * nq * hd..(j + 1) * nq * hd])).unwrap();
                let ob = eng.alloc_host_coherent_storage((nq * hd * 4) as u64).unwrap();
                let pc = laguna_gpu_sdpa_pc(seq_now, nq, nkv, hd, scale, window_start, ring_capacity);
                // SAFETY: kbuf/vbuf point into kv_res (unmutated across this loop),
                // the mapped planes hold all appended rows before submit.
                let kb: &Buffer = unsafe { &*kbuf };
                let vb: &Buffer = unsafe { &*vbuf };
                eng.record_to(cb, "laguna_gpu_sdpa", &[&qb, kb, vb, &ob], &pc, (nq as u32, 1, 1))
                    .unwrap();
                qbufs.push(qb);
                obufs.push(ob);
            }
            if ts_on { eng.ts_cmd_mark(cb, 1, false); } // SPIKE
            eng.submit_batch(cb).unwrap();
            if ts_on { if let Ok(v) = eng.ts_read_ns(0, 2) { // SPIKE
                crate::prof_add_ns("lag_gpu_sdpa", (v[1] - v[0]).max(0.0) as u128); } }
            let _t_rb = std::time::Instant::now(); // SPIKE
            for j in 0..new_seq {
                attn_out[j * nq * hd..(j + 1) * nq * hd]
                    .copy_from_slice(&read_f32_buf(&obufs[j], nq * hd));
            }
            crate::prof_add("lag_readback", _t_rb); // SPIKE
            for buf in qbufs.into_iter().chain(obufs) {
                eng.return_to_pool(buf);
            }
        } else {
            // Host cpu_sdpa over the resident plane's mapped memory (no readback,
            // but the reduction runs on the CPU — the span-fold decode wall).
            let _t_hs = std::time::Instant::now(); // SPIKE
            for j in 0..new_seq {
                let plane = self.kv_res.get(&layer_idx).expect("resident KV plane for layer");
                let seq_now = seq_at[j];
                let q_j = &q[j * nq * hd..(j + 1) * nq * hd];
                let o = cpu_sdpa(
                    q_j, plane.k_up_to_now(), plane.v_up_to_now(), nq, nkv, hd, seq_now, scale, window,
                );
                attn_out[j * nq * hd..(j + 1) * nq * hd].copy_from_slice(&o);
            }
            crate::prof_add("lag_host_sdpa", _t_hs); // SPIKE
        }
        }

        // CB2: per-head GPU softplus gate (scales attn_out in place) → o_proj, for
        // all rows, in ONE command buffer — the pre-gate attn_out + gate `g` are
        // uploaded once and o_proj reads the GPU-gated buffer (no gate readback).
        // The softplus dispatch matches the host `softplus` up to the last ulp of
        // the GPU exp/log intrinsics (see `laguna_softplus_gate.comp`).
        let (ow, ow_k, _, _) = self.mv_meta(&format!("{p}.o_proj.weight")).unwrap();
        let eng = self.engine.as_mut().unwrap();
        let ts_on = crate::prof_on() && eng.ensure_ts_pool(8); // SPIKE
        let cb = eng.begin_batch().unwrap();
        if ts_on { eng.ts_cmd_reset(cb, 0, 8); eng.ts_cmd_mark(cb, 0, true); } // SPIKE
        let mut gin = Vec::with_capacity(new_seq); // gate proj g[nq]
        let mut ain = Vec::with_capacity(new_seq); // attn_out[nq*hd] (gated in place)
        let mut oout = Vec::with_capacity(new_seq); // o_proj out[hs]
        let sp_pc = laguna_softplus_gate_pc(nq, hd);
        let sp_wg = (((nq * hd) as u32 + 255) / 256, 1u32, 1u32);
        for j in 0..new_seq {
            let gb = eng.alloc_host_coherent_storage((nq * 4) as u64).unwrap();
            gb.write(&f32_slice_to_bytes(&g[j * nq..(j + 1) * nq])).unwrap();
            let ab = eng.alloc_host_coherent_storage((nq * hd * 4) as u64).unwrap();
            ab.write(&f32_slice_to_bytes(&attn_out[j * nq * hd..(j + 1) * nq * hd])).unwrap();
            eng.record_to(cb, "laguna_softplus_gate", &[&gb, &ab], &sp_pc, sp_wg).unwrap();
            gin.push(gb);
            ain.push(ab);
        }
        // Barrier: the softplus writes (SHADER_WRITE) must be visible to o_proj's
        // reads (SHADER_READ) of the same attn buffers.
        eng.record_barrier_to(cb);
        if ts_on { eng.ts_cmd_mark(cb, 1, false); } // SPIKE: after softplus gate
        for j in 0..new_seq {
            let out = eng.alloc_host_coherent_storage((hs * 4) as u64).unwrap();
            Self::rec_mv(eng, cb, ow, ow_k, &ain[j], &out, nq * hd, hs);
            oout.push(out);
        }
        if ts_on { eng.ts_cmd_mark(cb, 2, false); } // SPIKE: after o_proj
        eng.submit_batch(cb).unwrap();
        if ts_on { if let Ok(v) = eng.ts_read_ns(0, 3) { // SPIKE
            crate::prof_add_ns("lag_gpu_softplus", (v[1] - v[0]).max(0.0) as u128);
            crate::prof_add_ns("lag_gpu_oproj",    (v[2] - v[1]).max(0.0) as u128); } }
        let _t_rb = std::time::Instant::now(); // SPIKE
        let mut result = vec![0.0f32; new_seq * hs];
        for j in 0..new_seq {
            result[j * hs..(j + 1) * hs].copy_from_slice(&read_f32_buf(&oout[j], hs));
        }
        crate::prof_add("lag_readback", _t_rb); // SPIKE
        for buf in gin.into_iter().chain(ain).chain(oout) {
            eng.return_to_pool(buf);
        }
        result
    }

    /// SCRATCH-BANK twin of `attn_cached_1cb` for the SINGLE-TOKEN decode
    /// (`new_seq == 1`, `VLLM_VULKAN_LAGUNA_SCRATCH`). Byte-for-byte the same
    /// q/k/v/g CB1 → host qk-norm/rope/append → SDPA → softplus-gate/o-proj CB2
    /// sequence, push-constants and readbacks, but the GPU I/O buffers all come
    /// from the persistent `LagunaScratch` bank (allocated once, reused across
    /// every token and layer) instead of the ~10 per-step pool alloc/free; inputs
    /// are written straight into mapped memory (no `f32_slice_to_bytes` temp Vec).
    /// Bit-identical to the pool path (pure allocation reuse).
    fn attn_cached_1cb_scratch(&mut self, hidden_normed: &[f32], layer_idx: usize) -> Vec<f32> {
        self.ensure_scratch().expect("laguna scratch alloc");
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nkv = cfg.num_key_value_heads;
        let nq = cfg.num_attention_heads_per_layer[layer_idx];
        let is_full = cfg.layer_is_full[layer_idx];
        let eps = cfg.rms_norm_eps;
        let p = format!("model.layers.{layer_idx}.self_attn");

        let start_pos = self
            .kv_res
            .get(&layer_idx)
            .map(|c| c.seq_len)
            .or_else(|| self.kv.get(&layer_idx).map(|c| c.seq_len))
            .unwrap_or(0);

        let (qw, qw_k, _, _) = self.mv_meta(&format!("{p}.q_proj.weight")).unwrap();
        let (kw, kw_k, _, _) = self.mv_meta(&format!("{p}.k_proj.weight")).unwrap();
        let (vw, vw_k, _, _) = self.mv_meta(&format!("{p}.v_proj.weight")).unwrap();
        let (gw, gw_k, _, _) = self.mv_meta(&format!("{p}.g_proj.weight")).unwrap();
        let (ow, ow_k, _, _) = self.mv_meta(&format!("{p}.o_proj.weight")).unwrap();

        // Own the bank locally so it coexists with the mutable engine borrow.
        let mut sc = self.scratch.take().unwrap();

        // CB1: q/k/v/g projections for the single new row.
        let eng = self.engine.as_mut().unwrap();
        let _t_alloc = std::time::Instant::now(); // profiler: buffer churn (== ~0, no alloc)
        write_f32_mapped(&sc.a_inp, hidden_normed);
        crate::prof_add("lag_attn_bufalloc", _t_alloc);
        let cb = eng.begin_batch().unwrap();
        Self::rec_mv(eng, cb, qw, qw_k, &sc.a_inp, &sc.a_q, hs, nq * hd);
        Self::rec_mv(eng, cb, kw, kw_k, &sc.a_inp, &sc.a_k, hs, nkv * hd);
        Self::rec_mv(eng, cb, vw, vw_k, &sc.a_inp, &sc.a_v, hs, nkv * hd);
        Self::rec_mv(eng, cb, gw, gw_k, &sc.a_inp, &sc.a_g, hs, nq);
        eng.submit_batch(cb).unwrap();
        let nqh = nq * hd;
        let nkvh = nkv * hd;
        // HOSTFOLD copy-elision (`VLLM_VULKAN_LAGUNA_HOSTFOLD`): q/k copy into the
        // persistent host banks (mutated in place by qk-norm/rope; no per-call Vec),
        // v/g borrow the mapped GPU readback directly (read-only; no `.to_vec()`),
        // and q_norm/k_norm borrow the host weight (no per-layer `.to_vec()`). All
        // byte-identical to the OFF path below.
        let hostfold = crate::flags::flags_global().laguna_hostfold;
        // Owned fallbacks for the OFF path (untouched when hostfold).
        let mut q_own: Vec<f32> = Vec::new();
        let mut k_own: Vec<f32> = Vec::new();
        let v_own: Vec<f32>;
        let g_own: Vec<f32>;
        let qn_own: Vec<f32>;
        let kn_own: Vec<f32>;
        let q: &mut [f32];
        let k: &mut [f32];
        let v: &[f32];
        let g: &[f32];
        let q_norm: &[f32];
        let k_norm: &[f32];
        if hostfold {
            sc.h_q[..nqh].copy_from_slice(read_f32_mapped(&sc.a_q, nqh));
            sc.h_k[..nkvh].copy_from_slice(read_f32_mapped(&sc.a_k, nkvh));
            q = &mut sc.h_q[..nqh];
            k = &mut sc.h_k[..nkvh];
            v = read_f32_mapped(&sc.a_v, nkvh);
            g = read_f32_mapped(&sc.a_g, nq);
            let (qp, qnn) = self.w_ptr(&format!("{p}.q_norm.weight"));
            let (kp, knn) = self.w_ptr(&format!("{p}.k_norm.weight"));
            // SAFETY: `host` is not mutated during a forward (see `w_ptr`).
            q_norm = unsafe { std::slice::from_raw_parts(qp, qnn) };
            k_norm = unsafe { std::slice::from_raw_parts(kp, knn) };
        } else {
            // Copy projections into owned host buffers for the in-place qk-norm/rope.
            q_own = read_f32_mapped(&sc.a_q, nqh).to_vec();
            k_own = read_f32_mapped(&sc.a_k, nkvh).to_vec();
            v_own = read_f32_mapped(&sc.a_v, nkvh).to_vec();
            g_own = read_f32_mapped(&sc.a_g, nq).to_vec();
            qn_own = self.w(&format!("{p}.q_norm.weight")).to_vec();
            kn_own = self.w(&format!("{p}.k_norm.weight")).to_vec();
            q = &mut q_own[..];
            k = &mut k_own[..];
            v = &v_own[..];
            g = &g_own[..];
            q_norm = &qn_own[..];
            k_norm = &kn_own[..];
        }
        let abs = start_pos;
        for h in 0..nq {
            cpu_rms_norm_inplace(&mut q[h * hd..(h + 1) * hd], q_norm, eps);
        }
        for h in 0..nkv {
            cpu_rms_norm_inplace(&mut k[h * hd..(h + 1) * hd], k_norm, eps);
        }
        {
            let qs = &mut q[..] as *mut [f32];
            let ks = &mut k[..] as *mut [f32];
            // SAFETY: q and k are distinct buffers; the slices never overlap.
            unsafe {
                if is_full {
                    if crate::flags::flags_global().laguna_yarn_gpu {
                        let eng = self.engine.as_mut().unwrap();
                        Self::rope_full_yarn_gpu(
                            eng, &mut *qs, &mut *ks, abs, nq, nkv, hd,
                            cfg.full_rotary_dim, &cfg.yarn_inv_freq, cfg.full_attention_factor,
                        );
                    } else {
                        cpu_rope_yarn(
                            &mut *qs, &mut *ks, abs, nq, nkv, hd, cfg.full_rotary_dim,
                            &cfg.yarn_inv_freq, cfg.full_attention_factor,
                        );
                    }
                } else {
                    cpu_rope(&mut *qs, &mut *ks, abs, nq, nkv, hd, hd, cfg.sliding_rope_theta);
                }
            }
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        // HOSTFOLD: SDPA output into the persistent bank (no per-call Vec).
        let mut ao_own: Vec<f32> = Vec::new();
        let attn_out: &mut [f32] = if hostfold {
            &mut sc.h_attn_out[..nqh]
        } else {
            ao_own = vec![0.0f32; nqh];
            &mut ao_own[..]
        };

        // Append post-rope K/V into the resident plane. Single-token decode, so a
        // window-sized ring is safe by construction (one append, one attend — no
        // batched-overwrite hazard). `ring_capacity` (>0 for a shrunk sliding
        // plane) selects ring slot resolution in the shader/reader; 0 for full
        // layers (and when VLLM_VULKAN_LAGUNA_KV_RING is off) → byte-identical.
        let (seq_now, ring_capacity) = {
            let plane = self.kv_res.get_mut(&layer_idx).expect("resident KV plane for layer");
            plane.append(&*k, v);
            (plane.seq_len, plane.ring_capacity())
        };

        if crate::flags::flags_global().laguna_gpu_sdpa {
            let plane = self.kv_res.get(&layer_idx).expect("resident KV plane for layer");
            let kbuf = plane.k_buf() as *const Buffer;
            let vbuf = plane.v_buf() as *const Buffer;
            let window_start = if is_full { 0 } else { seq_now.saturating_sub(cfg.sliding_window) };
            let eng = self.engine.as_mut().unwrap();
            write_f32_mapped(&sc.a_qb, &*q);
            let cb = eng.begin_batch().unwrap();
            let pc = laguna_gpu_sdpa_pc(seq_now, nq, nkv, hd, scale, window_start, ring_capacity);
            // SAFETY: kbuf/vbuf point into kv_res (unmutated across this call).
            let kb: &Buffer = unsafe { &*kbuf };
            let vb: &Buffer = unsafe { &*vbuf };
            eng.record_to(cb, "laguna_gpu_sdpa", &[&sc.a_qb, kb, vb, &sc.a_ob], &pc, (nq as u32, 1, 1))
                .unwrap();
            eng.submit_batch(cb).unwrap();
            attn_out.copy_from_slice(read_f32_mapped(&sc.a_ob, nq * hd));
        } else if ring_capacity != 0 {
            // Sliding ring plane: read the compacted window (bit-identical to a
            // full plane with Some(window); see ResidentKvPlane::windowed_view).
            let plane = self.kv_res.get(&layer_idx).expect("resident KV plane for layer");
            let (kw, vw, vlen) = plane.windowed_view(cfg.sliding_window);
            let o = cpu_sdpa(&*q, &kw, &vw, nq, nkv, hd, vlen, scale, None);
            attn_out.copy_from_slice(&o);
        } else {
            let plane = self.kv_res.get(&layer_idx).expect("resident KV plane for layer");
            let o = cpu_sdpa(&*q, plane.k_up_to_now(), plane.v_up_to_now(), nq, nkv, hd, seq_now, scale, window);
            attn_out.copy_from_slice(&o);
        }

        // CB2: per-head GPU softplus gate (scales attn_out in place) → o_proj.
        let eng = self.engine.as_mut().unwrap();
        write_f32_mapped(&sc.a_gb, g);
        write_f32_mapped(&sc.a_ab, &*attn_out);
        let sp_pc = laguna_softplus_gate_pc(nq, hd);
        let sp_wg = (((nq * hd) as u32 + 255) / 256, 1u32, 1u32);
        let cb = eng.begin_batch().unwrap();
        eng.record_to(cb, "laguna_softplus_gate", &[&sc.a_gb, &sc.a_ab], &sp_pc, sp_wg).unwrap();
        eng.record_barrier_to(cb);
        Self::rec_mv(eng, cb, ow, ow_k, &sc.a_ab, &sc.a_out, nq * hd, hs);
        eng.submit_batch(cb).unwrap();
        let _t_rb = std::time::Instant::now();
        let result = read_f32_mapped(&sc.a_out, hs).to_vec();
        crate::prof_add("lag_readback", _t_rb);

        let _t_free = std::time::Instant::now();
        self.scratch = Some(sc);
        crate::prof_add("lag_attn_buffree", _t_free);
        result
    }

    /// FOLDED decoder layer (the 1-CB twin of `layer_forward_cached`). Norms +
    /// residuals stay on host (identical); attention and MLP use the folded
    /// batched-submit paths.
    pub fn layer_forward_cached_1cb(&mut self, hidden: &[f32], new_seq: usize, layer_idx: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        // HOSTFOLD copy-elision path (single-token decode only). Borrows the two
        // layernorm weights (no per-layer `.to_vec()`) and rms/residuals through
        // persistent host banks (no per-layer Vec). Byte-identical GPU dispatches +
        // host math to the OFF path below → bit-exact. `lag_hostresid` buckets ONLY
        // the norm/residual host work so the ON-vs-OFF delta is the residual removed.
        let hostfold = crate::flags::flags_global().laguna_hostfold
            && crate::flags::flags_global().laguna_scratch
            && new_seq == 1;
        if hostfold {
            self.ensure_scratch().expect("laguna scratch alloc");
            let mut hsc = self.host_scratch.take().expect("host scratch");
            let _t_hr = std::time::Instant::now();
            {
                let (p, n) = self.w_ptr(&format!("model.layers.{layer_idx}.input_layernorm.weight"));
                let in_ln = unsafe { std::slice::from_raw_parts(p, n) };
                hsc.normed[..hs].copy_from_slice(hidden);
                cpu_rms_norm_inplace(&mut hsc.normed[..hs], in_ln, eps);
            }
            crate::prof_add("lag_hostresid", _t_hr);
            let _t_attn = std::time::Instant::now();
            let attn = self.attn_cached_1cb(&hsc.normed[..hs], 1, layer_idx);
            crate::prof_add("lag_attn", _t_attn);
            let _t_hr = std::time::Instant::now();
            for i in 0..hs {
                hsc.h1[i] = hidden[i] + attn[i];
            }
            {
                let (p, n) = self.w_ptr(&format!("model.layers.{layer_idx}.post_attention_layernorm.weight"));
                let post_ln = unsafe { std::slice::from_raw_parts(p, n) };
                hsc.normed2[..hs].copy_from_slice(&hsc.h1[..hs]);
                cpu_rms_norm_inplace(&mut hsc.normed2[..hs], post_ln, eps);
            }
            crate::prof_add("lag_hostresid", _t_hr);
            let _t_mlp = std::time::Instant::now();
            let mlp = if cfg.mlp_only_layers.contains(&layer_idx) {
                self.dense_mlp_1cb(&hsc.normed2[..hs], 1, layer_idx)
            } else {
                self.moe_token_1cb(&hsc.normed2[..hs], layer_idx)
            };
            crate::prof_add("lag_mlp", _t_mlp);
            let _t_hr = std::time::Instant::now();
            let mut out = vec![0.0f32; hs];
            for i in 0..hs {
                out[i] = hsc.h1[i] + mlp[i];
            }
            crate::prof_add("lag_hostresid", _t_hr);
            self.host_scratch = Some(hsc);
            return out;
        }

        let _t_hr = std::time::Instant::now();
        let in_ln = self.w(&format!("model.layers.{layer_idx}.input_layernorm.weight")).to_vec();
        let normed = cpu_rms_norm(hidden, &in_ln, eps);
        crate::prof_add("lag_hostresid", _t_hr);
        let _t_attn = std::time::Instant::now();
        let attn = self.attn_cached_1cb(&normed, new_seq, layer_idx);
        crate::prof_add("lag_attn", _t_attn);
        let _t_hr = std::time::Instant::now();
        let h1: Vec<f32> = hidden.iter().zip(&attn).map(|(&a, &b)| a + b).collect();

        let post_ln = self.w(&format!("model.layers.{layer_idx}.post_attention_layernorm.weight")).to_vec();
        let normed2 = cpu_rms_norm(&h1, &post_ln, eps);
        crate::prof_add("lag_hostresid", _t_hr);
        let _t_mlp = std::time::Instant::now();
        let mlp = if cfg.mlp_only_layers.contains(&layer_idx) {
            self.dense_mlp_1cb(&normed2, new_seq, layer_idx)
        } else {
            let mut out = vec![0.0f32; new_seq * hs];
            for t in 0..new_seq {
                let m = self.moe_token_1cb(&normed2[t * hs..(t + 1) * hs], layer_idx);
                out[t * hs..(t + 1) * hs].copy_from_slice(&m);
            }
            out
        };
        crate::prof_add("lag_mlp", _t_mlp);
        let _t_hr = std::time::Instant::now();
        let out = h1.iter().zip(&mlp).map(|(&a, &b)| a + b).collect();
        crate::prof_add("lag_hostresid", _t_hr);
        out
    }

    /// FIRST-stage folded single-token decode (1-CB twin of `forward_decode_token`).
    pub fn forward_decode_token_1cb(&mut self, tok: u32) -> Vec<f32> {
        let _t_dec = std::time::Instant::now();
        let mut hidden = self.embed_row(tok);
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward_cached_1cb(&hidden, 1, li);
        }
        let out = self.finalize(hidden, 1);
        crate::prof_add("lag_decode_total", _t_dec);
        out
    }

    /// MID/LAST-stage folded single-token decode (1-CB twin of
    /// `forward_decode_hidden`).
    pub fn forward_decode_hidden_1cb(&mut self, hidden_in: &[f32]) -> Vec<f32> {
        let _t_dec = std::time::Instant::now();
        let mut hidden = hidden_in.to_vec();
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward_cached_1cb(&hidden, 1, li);
        }
        let out = self.finalize(hidden, 1);
        crate::prof_add("lag_decode_total", _t_dec);
        out
    }

    /// FIRST-stage cache-populating prefill via the 1-CB fold (populates the
    /// DEVICE-RESIDENT K/V planes that the folded decode reads). The resident-arena
    /// twin of `forward_prefill_tokens`; bit-exact with the stateless `forward`
    /// up to `swiglu_f32`/softplus GPU-intrinsic last-ulp. MUST precede
    /// `forward_decode_token_1cb` / `forward_decode_hidden_1cb` so the planes hold
    /// the prefix (the per-op `forward_prefill_*` populate the HOST `kv` map, which
    /// the 1-CB path no longer reads).
    pub fn forward_prefill_tokens_1cb(&mut self, tokens: &[u32]) -> Vec<f32> {
        let cfg = self.config.clone();
        let hs = cfg.hidden_size;
        let seq = tokens.len();
        self.reset_kv();
        let mut hidden = vec![0.0f32; seq * hs];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = self.embed_row(tok);
            hidden[t * hs..(t + 1) * hs].copy_from_slice(&row);
        }
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward_cached_1cb(&hidden, seq, li);
        }
        self.finalize(hidden, seq)
    }

    /// MID/LAST-stage cache-populating prefill via the 1-CB fold (resident-arena
    /// twin of `forward_prefill_hidden`). Populates the resident K/V planes from a
    /// received `[seq*hidden]`.
    pub fn forward_prefill_hidden_1cb(&mut self, hidden_in: &[f32], seq: usize) -> Vec<f32> {
        self.reset_kv();
        let mut hidden = hidden_in.to_vec();
        for li in self.pp_start..self.pp_end {
            hidden = self.layer_forward_cached_1cb(&hidden, seq, li);
        }
        self.finalize(hidden, seq)
    }

    /// Standalone GPU dispatch of the per-head softplus gate for `x` (each `x[i]`
    /// its own 1-wide head), returning `softplus(x)` — the micro-gate that checks
    /// the `laguna_softplus_gate` shader against the host `softplus` in isolation
    /// (edge cases: large ±x, the ln2 crossover). Multiplies into a ones vector so
    /// the readback IS the softplus values. Requires an engine.
    pub fn softplus_gpu(&mut self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let eng = self.engine.as_mut().expect("engine present for softplus_gpu");
        let gb = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        gb.write(&f32_slice_to_bytes(x)).unwrap();
        let ones = vec![1.0f32; n];
        let ab = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        ab.write(&f32_slice_to_bytes(&ones)).unwrap();
        // nq = n heads, hd = 1 → element i uses gate x[i].
        let pc = laguna_softplus_gate_pc(n, 1);
        let wg = ((n as u32 + 255) / 256, 1u32, 1u32);
        let cb = eng.begin_batch().unwrap();
        eng.record_to(cb, "laguna_softplus_gate", &[&gb, &ab], &pc, wg).unwrap();
        eng.submit_batch(cb).unwrap();
        let out = read_f32_buf(&ab, n);
        eng.return_to_pool(gb);
        eng.return_to_pool(ab);
        out
    }

    /// QK-NORM micro-gate: run `num_heads` random rows of `head_dim` through the
    /// GPU `qk_norm_gpu` (`rms_norm_f32_mul`) and the host `cpu_rms_norm_inplace`
    /// (a random per-head weight, so the `do_multiply` weight path is exercised),
    /// and compare. Returns `(cos, maxdiff, bit_exact)`. Requires an engine.
    pub fn qknorm_micro(&mut self, num_heads: usize, head_dim: usize, eps: f32, seed: u64) -> (f64, f64, bool) {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut nf = || -> f32 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..num_heads * head_dim).map(|_| nf()).collect();
        let weight: Vec<f32> = (0..head_dim).map(|_| nf()).collect();
        // Host reference.
        let mut host = x.clone();
        for h in 0..num_heads {
            crate::model::cpu_rms_norm_inplace(&mut host[h * head_dim..(h + 1) * head_dim], &weight, eps);
        }
        // GPU.
        let mut gpu = x.clone();
        let eng = self.engine.as_mut().expect("engine present for qknorm_micro");
        Self::qk_norm_gpu(eng, &mut gpu, &weight, num_heads, head_dim, eps);
        let mut dot = 0f64; let mut ng = 0f64; let mut nc = 0f64; let mut maxd = 0f64;
        for (a, b) in gpu.iter().zip(&host) {
            dot += (*a as f64) * (*b as f64);
            ng += (*a as f64) * (*a as f64);
            nc += (*b as f64) * (*b as f64);
            maxd = maxd.max((*a as f64 - *b as f64).abs());
        }
        let cos = dot / (ng.sqrt() * nc.sqrt() + 1e-12);
        let bit_exact = gpu.iter().zip(&host).all(|(&a, &b)| a.to_bits() == b.to_bits());
        (cos, maxd, bit_exact)
    }
}

// ─── Canonical (layer, kv_head)-tile KV prefix export/import (NAS prefix-cache
//     Phase 1) ────────────────────────────────────────────────────────────────
//
// Laguna attention KV is a plain host-coherent memcpy exactly like gemma — NOT a
// device readback (verified: `ResidentKvPlane` mapped memory, `export_head`
// above). Two Laguna specifics vs gemma:
//   1. Dual storage. The 1-CB resident path lives in `kv_res` (mapped planes);
//      the per-op / CPU-fallback path lives in `kv` (host `KvCache`). Export
//      reads whichever is live (prefer `kv_res` when engine-resident), import
//      writes BOTH so a subsequent token on either path sees the restored state
//      (mirrors `reset_kv`'s dual reset).
//   2. Full/sliding split via `config.layer_is_full[L]`. Uniform head geometry
//      (all layers `num_key_value_heads`, `head_dim`).
//
// ⚠ ON-NODE GATE: the `kv_res` resident-plane branch (export-source-select +
// dual-write) can only be exercised with a GPU engine, so it is compiled but
// host-UNTESTED here — the offline gate below covers the `kv` host path only.
// The resident ring↔tile slicing is consistent WITH the ring BY CONSTRUCTION:
// `export_head`/`import_head` reuse `self.capacity` and the `abs % capacity`
// slot map, identical to `append`/`windowed_view`/`ring_windowed_gather`. Flag
// for the on-node Item-1b/Item-2 gate.
impl crate::kv_prefix::KvPrefixExport for LagunaGpuModel {
    fn kv_content_dims(&self) -> crate::kv_prefix::KvContentDims {
        let cfg = &self.config;
        let layers = (0..cfg.num_hidden_layers)
            .map(|l| crate::kv_prefix::LayerKvGeom {
                kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                is_full: cfg.layer_is_full[l],
                window: cfg.sliding_window,
                k_eq_v: false,
            })
            .collect();
        let mut rope = 0xcbf2_9ce4_8422_2325u64;
        for v in [
            cfg.full_rope_theta.to_bits() as u64,
            cfg.sliding_rope_theta.to_bits() as u64,
            cfg.sliding_window as u64,
            cfg.head_dim as u64,
        ] {
            rope ^= v;
            rope = rope.wrapping_mul(0x0000_0100_0000_01B3);
        }
        crate::kv_prefix::KvContentDims {
            arch_tag: 1,
            num_layers: cfg.num_hidden_layers,
            layers,
            rope_ident: rope,
        }
    }

    fn owned_tiles(&self) -> Vec<crate::kv_prefix::TileSpec> {
        let cfg = &self.config;
        let mut out = Vec::new();
        for l in self.pp_start..self.pp_end {
            let is_full = cfg.layer_is_full[l];
            for h in 0..cfg.num_key_value_heads {
                out.push(crate::kv_prefix::TileSpec {
                    layer: l,
                    kv_head: h,
                    head_dim: cfg.head_dim,
                    is_full,
                    window: cfg.sliding_window,
                    k_eq_v: false,
                });
            }
        }
        out
    }

    fn export_tile(
        &self,
        layer: usize,
        kv_head: usize,
        upto: usize,
        dtype: crate::kv_prefix::KvDtype,
    ) -> Result<Vec<u8>, String> {
        let cfg = &self.config;
        let is_full = cfg.layer_is_full[layer];
        let head_dim = cfg.head_dim;
        let (base, n_rows) = crate::model::tile_row_range(is_full, upto, cfg.sliding_window);
        // Prefer the resident plane when engine-loaded (1-CB path), else the host
        // per-op / CPU-fallback KvCache — the authoritative-source rule (scope
        // §1.2): whichever the live decode path writes.
        let (k, v) = if let Some(plane) = self.kv_res.get(&layer) {
            plane.export_head(kv_head, head_dim, base, n_rows)
        } else if let Some(cache) = self.kv.get(&layer) {
            let stride = cache.num_kv_heads * cache.head_dim;
            let mut k = vec![0.0f32; n_rows * head_dim];
            let mut v = vec![0.0f32; n_rows * head_dim];
            for i in 0..n_rows {
                let slot = (base + i) % cache.capacity;
                let src = slot * stride + kv_head * head_dim;
                k[i * head_dim..(i + 1) * head_dim].copy_from_slice(&cache.k[src..src + head_dim]);
                v[i * head_dim..(i + 1) * head_dim].copy_from_slice(&cache.v[src..src + head_dim]);
            }
            (k, v)
        } else {
            return Err(format!("export_tile: no KV for resident layer {layer}"));
        };
        crate::kv_prefix::write_tile(layer, kv_head, dtype, base, head_dim, false, &k, &v)
    }

    fn import_tile(&mut self, layer: usize, kv_head: usize, blob: &[u8]) -> Result<usize, String> {
        let hdr = crate::kv_prefix::read_tile_header(blob)?;
        if hdr.layer != layer || hdr.kv_head != kv_head {
            return Err(format!(
                "import_tile: blob (L{},h{}) != caller (L{layer},h{kv_head})",
                hdr.layer, hdr.kv_head
            ));
        }
        let (k, v) = crate::kv_prefix::read_tile_body(blob, &hdr)?;
        let head_dim = hdr.head_dim;
        // Dual-write: BOTH the host KvCache (per-op / CPU path) and the resident
        // plane (1-CB path) so either decode path resumes from restored state.
        if let Some(cache) = self.kv.get_mut(&layer) {
            let stride = cache.num_kv_heads * cache.head_dim;
            for i in 0..hdr.n_rows {
                let slot = (hdr.window_base + i) % cache.capacity;
                let dst = slot * stride + kv_head * head_dim;
                cache.k[dst..dst + head_dim].copy_from_slice(&k[i * head_dim..(i + 1) * head_dim]);
                cache.v[dst..dst + head_dim].copy_from_slice(&v[i * head_dim..(i + 1) * head_dim]);
            }
        }
        if let Some(plane) = self.kv_res.get_mut(&layer) {
            plane.import_head(kv_head, head_dim, hdr.window_base, &k, &v);
        }
        Ok(hdr.n_rows)
    }

    fn set_seq_len(&mut self, n: usize) {
        for c in self.kv.values_mut() {
            c.seq_len = n;
        }
        for p in self.kv_res.values_mut() {
            p.seq_len = n;
        }
    }
}

#[cfg(test)]
mod laguna_kv_tile_tests {
    //! Host bit-exact gate for the Laguna canonical `(layer, kv_head)`-tile KV
    //! prefix export/import (NAS prefix-cache Phase 1), CPU-fallback path.
    //!
    //! `ResidentKvPlane` needs a GPU engine (can't build on Mac), so this
    //! exercises the `kv` HashMap<KvCache> path — the per-op / CPU-fallback
    //! authoritative source. It covers a FULL-attn layer (`[0,upto)`) and a
    //! SLIDING layer (`[upto-window, upto)` with a non-zero `window_base`), and
    //! asserts `f32::to_bits` equality on the covered rows. The `kv_res`
    //! dual-write + resident-source-select path is DEFERRED to the on-node GPU
    //! gate (documented in the task report / IMPROVEMENTS).
    use super::*;
    use crate::kv_prefix::{KvDtype, KvPrefixExport};

    fn empty_model(num_layers: usize, max_seq: usize) -> LagunaGpuModel {
        let cfg = crate::laguna::tiny_test_config(num_layers);
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let mut kv = HashMap::new();
        for l in 0..num_layers {
            kv.insert(l, KvCache::new(max_seq, nkv, hd));
        }
        LagunaGpuModel {
            config: cfg,
            engine: None,
            gpu_weights: HashMap::new(),
            gpu_experts: HashMap::new(),
            host: HashMap::new(),
            pp_start: 0,
            pp_end: num_layers,
            pp_first: true,
            pp_last: true,
            dir: std::path::PathBuf::new(),
            kv,
            kv_res: HashMap::new(),
            max_seq,
            scratch: None,
            gpu_router: HashMap::new(),
            pp_recv_scratch: Vec::new(),
            pp_recv_handle: 0,
            pp_send_scratch: Vec::new(),
            pp_send_handle: 0,
            pp_vocab_scratch: Vec::new(),
            pp_vocab_handle: 0,
            pp_topk_scratch: Vec::new(),
            pp_topk_handle: 0,
            host_scratch: None,
        }
    }

    fn seed_kv(m: &mut LagunaGpuModel, upto: usize, seed: u64) {
        let mut s = seed | 1;
        let mut nxt = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5
        };
        for c in m.kv.values_mut() {
            let n = upto * c.num_kv_heads * c.head_dim;
            for i in 0..n {
                c.k[i] = nxt();
                c.v[i] = nxt();
            }
            c.seq_len = upto;
        }
    }

    #[test]
    fn laguna_cpu_tile_roundtrip_bit_exact() {
        let num_layers = 4usize; // layer 0 full, 1..4 sliding (window 4)
        let max_seq = 32usize;
        let upto = 6usize; // > window(4): forces a non-zero window_base on sliding
        let mut src = empty_model(num_layers, max_seq);
        seed_kv(&mut src, upto, 0x1234_ABCD);

        let tiles = src.owned_tiles();
        assert!(tiles.iter().any(|t| t.is_full), "must have a full-attn tile");
        assert!(tiles.iter().any(|t| !t.is_full), "must have a sliding tile");
        // Confirm the sliding split really uses a non-zero base at this upto.
        let sliding = tiles.iter().find(|t| !t.is_full).unwrap();
        let (base, _n) = crate::model::tile_row_range(sliding.is_full, upto, src.config.sliding_window);
        assert!(base > 0, "sliding tile must have window_base > 0 at upto={upto}");

        let blobs: Vec<(usize, usize, Vec<u8>)> = tiles
            .iter()
            .map(|t| (t.layer, t.kv_head, src.export_tile(t.layer, t.kv_head, upto, KvDtype::F32).unwrap()))
            .collect();

        let mut dst = empty_model(num_layers, max_seq);
        for (l, h, blob) in &blobs {
            dst.import_tile(*l, *h, blob).unwrap();
        }
        dst.set_seq_len(upto);
        assert_eq!(dst.kv.get(&0).unwrap().seq_len, upto);

        for t in &tiles {
            let (b, n_rows) = crate::model::tile_row_range(t.is_full, upto, src.config.sliding_window);
            let sc = src.kv.get(&t.layer).unwrap();
            let dc = dst.kv.get(&t.layer).unwrap();
            let stride = sc.num_kv_heads * sc.head_dim;
            for i in 0..n_rows {
                let slot = (b + i) % sc.capacity;
                let off = slot * stride + t.kv_head * t.head_dim;
                for j in 0..t.head_dim {
                    assert_eq!(sc.k[off + j].to_bits(), dc.k[off + j].to_bits(), "K drift L{} h{}", t.layer, t.kv_head);
                    assert_eq!(sc.v[off + j].to_bits(), dc.v[off + j].to_bits(), "V drift L{} h{}", t.layer, t.kv_head);
                }
            }
        }
    }
}

#[cfg(test)]
mod per_layer_kv_ring_tests {
    //! Offline host bit-exact gate for the Laguna Phase-0 per-layer KV ring
    //! (`VLLM_VULKAN_LAGUNA_KV_RING`). `ResidentKvPlane` itself wraps a GPU
    //! `Buffer` (needs an engine, so it can't be built on the Mac), but its ring
    //! math is factored into the free `ring_windowed_gather` — the EXACT function
    //! `ResidentKvPlane::windowed_view` calls over its mapped planes — and the
    //! `laguna_gpu_sdpa` shader's `slot = token_idx % ring_capacity` addressing is
    //! mirrored here byte-for-byte. Both tests drive a stream PAST the window so
    //! the ring wraps many times. The on-node argmax gate exercises the real GPU
    //! shader; these prove the addressing it shares with the host reader is exact.
    use super::*;

    fn prng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed | 1;
        move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        }
    }

    /// `ring_windowed_gather` over a WRAPPED ring plane is byte-identical to a
    /// gather over a full absolute plane, and both drive `cpu_sdpa` to a result
    /// bit-identical to the legacy full-plane `Some(window)` path — at every step
    /// of a Laguna sliding-layer stream far longer than the window (many wraps).
    /// This is the `windowed_view` correctness contract that makes ring sizing
    /// bit-exact (Laguna analog of the gemma `ring_windowed_view_bit_exact` test).
    #[test]
    fn laguna_ring_windowed_gather_bit_exact_vs_full_and_legacy() {
        // Laguna sliding-layer GQA shape (72Q/8KV, scaled down for speed; the
        // gather/attention math is head-count/stride agnostic).
        let (nkv, hd, nq) = (8usize, 16usize, 8usize);
        let stride = nkv * hd;
        let window = 8usize; // Laguna's real window is 512; the wrap behaviour is what matters.
        let max_seq = 80usize;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut rng = prng(0x1A6_11A_u64);

        // Full absolute plane (never wraps: capacity == max_seq) and the ring
        // plane (capacity == window). Append with the SAME modulus append uses:
        // slot = seq_len % capacity.
        let mut full_k = vec![0.0f32; max_seq * stride];
        let mut full_v = vec![0.0f32; max_seq * stride];
        let mut ring_k = vec![0.0f32; window * stride];
        let mut ring_v = vec![0.0f32; window * stride];

        for seq_len_before in 0..max_seq {
            let krow: Vec<f32> = (0..stride).map(|_| rng()).collect();
            let vrow: Vec<f32> = (0..stride).map(|_| rng()).collect();
            // append into both planes at slot = pos % capacity
            let fs = (seq_len_before % max_seq) * stride;
            full_k[fs..fs + stride].copy_from_slice(&krow);
            full_v[fs..fs + stride].copy_from_slice(&vrow);
            let rs = (seq_len_before % window) * stride;
            ring_k[rs..rs + stride].copy_from_slice(&krow);
            ring_v[rs..rs + stride].copy_from_slice(&vrow);

            let seq = seq_len_before + 1;
            let q: Vec<f32> = (0..nq * hd).map(|_| rng()).collect();

            // legacy reference: full absolute [0,seq) + Some(window) mask
            let legacy = cpu_sdpa(
                &q, &full_k[..seq * stride], &full_v[..seq * stride],
                nq, nkv, hd, seq, scale, Some(window),
            );
            // gather via the REAL ring code: full plane (never wraps) and ring.
            let (kf, vf, lf) = ring_windowed_gather(&full_k, &full_v, max_seq, stride, seq, window);
            let (kr, vr, lr) = ring_windowed_gather(&ring_k, &ring_v, window, stride, seq, window);
            let via_full = cpu_sdpa(&q, &kf, &vf, nq, nkv, hd, lf, scale, None);
            let via_ring = cpu_sdpa(&q, &kr, &vr, nq, nkv, hd, lr, scale, None);

            assert_eq!(lf, lr, "valid_len mismatch at seq {seq}");
            assert_eq!(lf, seq.min(window), "valid_len != min(seq,window) at seq {seq}");
            for i in 0..kf.len() {
                assert_eq!(kf[i].to_bits(), kr[i].to_bits(), "ring K byte drift at seq {seq} idx {i}");
                assert_eq!(vf[i].to_bits(), vr[i].to_bits(), "ring V byte drift at seq {seq} idx {i}");
            }
            for i in 0..legacy.len() {
                assert_eq!(legacy[i].to_bits(), via_full[i].to_bits(),
                    "windowed_view != legacy at seq {seq} idx {i}");
                assert_eq!(legacy[i].to_bits(), via_ring[i].to_bits(),
                    "ring windowed_view != legacy at seq {seq} idx {i}");
            }
        }
        // The stream (80) far exceeds the window (8): the ring wrapped ~10×.
        assert!(max_seq > window * 2);
    }

    /// GPU-plane addressing gate (host simulation of the exact byte offsets the
    /// `laguna_gpu_sdpa` shader + `ResidentKvPlane::append` share). Models Laguna's
    /// layout — SEPARATE contiguous K and V planes `[capacity, nkv, hd]`, row
    /// stride `nkv*hd`, token `t`'s `kv_head` at `slot*row_stride + kv_head*hd`
    /// where `slot = t % ring_capacity` (ring) or `t` (full) — with:
    ///   • append at slot `pos % capacity` (== `ResidentKvPlane::append`);
    ///   • shader window read `[seq-window, seq)` at slot `t % ring_capacity` and
    ///     the same `+ kv_head*hd` sub-row offset the shader computes.
    /// Asserts the window rows gathered PER KV-HEAD from the RING planes are
    /// byte-identical to those from a full absolute plane, AND equal the golden
    /// source rows — over a decode that wraps the ring several times. Full/YaRN
    /// layers (`ring_capacity == 0`) are the `slot == t` branch (unchanged).
    #[test]
    fn laguna_gpu_ring_plane_offsets_bit_exact_vs_absolute() {
        let (nkv, hd) = (8usize, 16usize);
        let row_stride = nkv * hd;
        let window = 8usize;      // sliding capacity (ring)
        let max_seq = 60usize;    // full capacity
        let prompt = 5usize;      // seeded prefill positions
        let decode_to = 44usize;  // wraps the ring several times
        let mut rng = prng(0xC0FFEE_u64);

        // Golden absolute source: every position's full K/V token row.
        let mut src_k = vec![0f32; decode_to * row_stride];
        let mut src_v = vec![0f32; decode_to * row_stride];
        for i in 0..decode_to * row_stride { src_k[i] = rng(); src_v[i] = rng(); }

        // Fill a device plane (separate K/V, cap*row_stride each) up to `upto`.
        let make_plane = |cap: usize, ring: bool, upto: usize| -> (Vec<f32>, Vec<f32>) {
            let mut pk = vec![0f32; cap * row_stride];
            let mut pv = vec![0f32; cap * row_stride];
            let slot_of = |p: usize| if ring { p % cap } else { p };
            // Device state = the contiguous run [prompt-cap .. upto) that survives
            // (seed of the last cap prompt rows ∪ in-CB appends [prompt..upto)).
            let start = if ring { prompt.saturating_sub(cap) } else { 0 };
            for p in start..upto {
                let s = slot_of(p);
                pk[s * row_stride..(s + 1) * row_stride]
                    .copy_from_slice(&src_k[p * row_stride..(p + 1) * row_stride]);
                pv[s * row_stride..(s + 1) * row_stride]
                    .copy_from_slice(&src_v[p * row_stride..(p + 1) * row_stride]);
            }
            (pk, pv)
        };

        // Shader-side per-kv-head gather of window [ws,seq): kv_base = slot*row_stride + kv_head*hd.
        let gather = |pk: &[f32], pv: &[f32], cap: usize, ring: bool, seq: usize, ws: usize, kv_head: usize|
            -> (Vec<f32>, Vec<f32>) {
            let (mut gk, mut gv) = (Vec::new(), Vec::new());
            for t in ws..seq {
                let slot = if ring { t % cap } else { t };
                let base = slot * row_stride + kv_head * hd;
                gk.extend_from_slice(&pk[base..base + hd]);
                gv.extend_from_slice(&pv[base..base + hd]);
            }
            (gk, gv)
        };

        for seq in (prompt + 1)..=decode_to {
            let (rk, rv) = make_plane(window, true, seq);
            let (ak, av) = make_plane(max_seq, false, seq);
            let ws = seq.saturating_sub(window);
            for kv_head in 0..nkv {
                let (grk, grv) = gather(&rk, &rv, window, true, seq, ws, kv_head);
                let (gak, gav) = gather(&ak, &av, max_seq, false, seq, ws, kv_head);
                assert_eq!(grk.len(), gak.len(), "gather len mismatch seq {seq} kvh {kv_head}");
                for i in 0..gak.len() {
                    assert_eq!(grk[i].to_bits(), gak[i].to_bits(),
                        "ring K != absolute at seq {seq} kvh {kv_head} idx {i}");
                    assert_eq!(grv[i].to_bits(), gav[i].to_bits(),
                        "ring V != absolute at seq {seq} kvh {kv_head} idx {i}");
                }
                // gathered window == golden source rows [ws,seq) for this kv_head
                for (j, p) in (ws..seq).enumerate() {
                    for d in 0..hd {
                        let want = src_k[p * row_stride + kv_head * hd + d];
                        assert_eq!(grk[j * hd + d].to_bits(), want.to_bits(),
                            "ring K row != golden pos {p} kvh {kv_head} at seq {seq}");
                    }
                }
            }
        }
    }
}
