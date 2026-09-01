// SPDX-License-Identifier: Apache-2.0
//! The UNIFIED GPU decode-layer engine (VLLM_VULKAN_UNIFIED=1): one slot set
//! + one `gpu_layer` dispatch path serving both Qwen3 and Gemma4, described
//! by a config-parameterized `LayerSpec`. Extracted verbatim from lib.rs
//! (M1).

use crate::compute;
use crate::model;
use crate::gpu_error::GpuResult;
use crate::VulkanModel;
use crate::{
    matvec_pc13, matvec_variant, sdpa_pc, ew_unary_pc, ew_mul_pc, rmsnorm_pc,
    rope_neox_pc, f32_slice_to_bytes, read_f32_buf,
};
use crate::{unified_1cb_enabled, attn_decode_kernel, prof_add};


// Slot indices for the UNIFIED GPU layer engine (VLLM_VULKAN_UNIFIED=1). One
// slot set serves BOTH Qwen3 and Gemma4 — a superset of QR_* and GR_*. The
// sandwich-norm "normed output" slots (UR_ON / UR_DOWNN) are unused on the
// qwen PRE-norm path. Sized for the loaded model's max dims in init_ures_bufs.
const UR_HA:    usize = 0;  // hidden A (layer input / ffn-add output)         [h]
const UR_HB:    usize = 1;  // hidden B (attn-add output / ffn residual)       [h]
const UR_X:     usize = 2;  // input_layernorm output (q/k/v input)            [h]
const UR_Q:     usize = 3;  // q proj → q-norm → rope (in place)              [q_dim]
const UR_K:     usize = 4;  // k proj → k-norm → rope (in place)             [kv_dim]
const UR_V:     usize = 5;  // v proj (→ v-norm no-weight, gemma)            [kv_dim]
const UR_ATTN:  usize = 6;  // attention output (host sdpa → uploaded)        [q_dim]
const UR_O:     usize = 7;  // o proj output                                   [h]
const UR_ON:    usize = 8;  // post_attention_layernorm(o_proj) (gemma)        [h]
const UR_FFIN:  usize = 9;  // ffn-input norm output                           [h]
const UR_GATE:  usize = 10; // gate proj output                          [max_inter]
const UR_UP:    usize = 11; // up proj output                            [max_inter]
const UR_ACT:   usize = 12; // silu/gelu(gate) output                    [max_inter]
const UR_MID:   usize = 13; // act(gate)*up                              [max_inter]
const UR_DOWN:  usize = 14; // down proj output                                [h]
const UR_DOWNN: usize = 15; // post_feedforward_layernorm(down) (gemma)        [h]
const UR_POS:   usize = 16; // rope position (1 int)
const UR_FF:    usize = 17; // rope freq-factors dummy (1 f32)
const UR_IDX:   usize = 18; // rope set_rows idx dummy (2 u32)
const UR_DUMMY: usize = 19; // harmless binding-3 buf for add_f32_f32_f32       [h]
const UR_PLE_G: usize = 20; // PLE gate output (gemma)                   [ple_dim]
const UR_PLE_C: usize = 21; // PLE projection contribution (gemma)             [h]
const UR_LOGITS:usize = 22; // final lm_head output                        [vocab]
const UR_COUNT: usize = 23;

/// FFN activation: the only ew-unary that differs between the two arches.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Activation { Silu, Gelu }
impl Activation {
    /// SPIR-V entry point for this activation. The gemma/qwen split is exactly
    /// one kernel wide — everything else in the unified layer is shared — so
    /// picking the wrong one here silently swaps GELU for SiLU and shows up
    /// only as drifted logits, never as an error.
    pub(crate) fn shader(self) -> &'static str {
        match self { Activation::Silu => "silu_f32", Activation::Gelu => "gelu_f32" }
    }
}

/// Gemma per-layer PLE (per-layer-embedding) tail descriptor.
pub(crate) struct PleSpec {
    pub(crate) ple_dim: usize,
    pub(crate) layer_ple: Vec<f32>,   // this layer's slice of the PLE inputs
    pub(crate) layer_scalar: f32,     // model.layers.{i}.layer_scalar
}

/// Config-parameterized description of ONE decoder layer. `gpu_layer(spec)`
/// consumes it and records the whole layer. Qwen is a degenerate Gemma:
///   - residual: PRE-norm (sandwich=false) vs Gemma SANDWICH (sandwich=true)
///   - norms: qwen has only input + post_attention (reused as ffn_in_norm);
///            gemma adds k_norm/v_norm + pre/post-feedforward norms + PLE norm
///   - rope: qwen full-rotary (rotary_dim=head_dim); gemma global=1e6 partial
///            (head_dim/4) / sliding=1e4 full
///   - attention: qwen full (window=None); gemma windowed/KV-shared
///   - activation: Silu (qwen) vs Gelu (gemma)
pub(crate) struct LayerSpec {
    pub(crate) is_qwen: bool,
    pub(crate) hidden: usize,
    pub(crate) head_dim: usize,
    pub(crate) num_q: usize,
    pub(crate) num_kv: usize,
    pub(crate) inter: usize,
    pub(crate) eps: f32,
    pub(crate) attn_scale: f32,
    // norms
    pub(crate) k_norm: bool,
    pub(crate) v_norm: bool,
    /// rms weight name applied to HB before FFN (qwen post_attention_layernorm;
    /// gemma pre_feedforward_layernorm).
    pub(crate) ffn_in_norm: String,
    pub(crate) sandwich: bool,
    pub(crate) post_attn_norm: Option<String>,  // sandwich: applied to o_proj output
    pub(crate) post_ffn_norm: Option<String>,   // sandwich: applied to down output
    // rope
    pub(crate) theta: f32,
    pub(crate) rotary_dim: usize,
    /// RoPE frequency basis dimension (decoupled from `rotary_dim` for gemma
    /// proportional RoPE: full-rotary callers set this == rotary_dim ==
    /// head_dim, which is bit-exact with the pre-decoupling behaviour).
    pub(crate) freq_dim: usize,
    // attention
    pub(crate) window: Option<usize>,
    pub(crate) kv_shared: bool,
    // ffn
    pub(crate) activation: Activation,
    // gemma per-layer PLE
    pub(crate) ple: Option<PleSpec>,
    /// Value-less global (full-attention) gemma layers: no `v_proj` tensor on
    /// disk at all; V is derived from the raw (pre-k_norm) K. Always false
    /// for qwen. Mirrors `cfg.layer_uses_k_eq_v` / the base path.
    pub(crate) uses_k_eq_v: bool,
}

impl LayerSpec {
    /// Build the unified layer descriptor for a Qwen3-dense layer.
    ///
    /// Every layer is identical, hence the ignored `_layer_idx` — kept in the
    /// signature so the two constructors stay interchangeable at the call site.
    /// The qwen shape is uniform in the ways gemma's is not: full rotary, one
    /// RoPE theta, no sliding window, no PLE, no sandwich norms, PRE-norm
    /// (`post_attention_layernorm` normalizes the FFN INPUT here, despite the
    /// name).
    pub(crate) fn qwen(cfg: &model::Qwen3Config, _layer_idx: usize) -> Self {
        let head_dim = cfg.head_dim;
        LayerSpec {
            is_qwen: true,
            hidden: cfg.hidden_size,
            head_dim,
            num_q: cfg.num_attention_heads,
            num_kv: cfg.num_key_value_heads,
            inter: cfg.intermediate_size,
            eps: cfg.rms_norm_eps,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            k_norm: true,
            v_norm: false,
            // qwen PRE-norm: post_attention_layernorm normalizes the FFN input.
            ffn_in_norm: "post_attention_layernorm.weight".to_string(),
            sandwich: false,
            post_attn_norm: None,
            post_ffn_norm: None,
            theta: cfg.rope_theta,
            rotary_dim: head_dim,   // full rotary
            freq_dim: head_dim,
            window: None,
            kv_shared: false,
            activation: Activation::Silu,
            ple: None,
            uses_k_eq_v: false,
        }
    }

    /// Build the unified layer descriptor for a Gemma4 layer.
    ///
    /// STRONGLY per-layer, unlike the qwen twin: full (global) and sliding
    /// layers differ in head dim, kv-head count, RoPE theta (1e6 vs 1e4),
    /// rotary width (PARTIAL `head_dim/4` on global layers, full on sliding)
    /// and window. Reading any of these off the model-level config instead of
    /// `layer_idx` mis-shapes half the network.
    pub(crate) fn gemma(cfg: &model::Gemma4Config, layer_idx: usize,
             layer_ple: Vec<f32>, layer_scalar: f32) -> Self {
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let is_kv_shared = cfg.is_kv_shared(layer_idx);
        // global → theta 1e6, partial rotary head_dim/4; sliding → 1e4, full.
        let (theta, rotary_dim) = if is_full {
            (1_000_000.0f32, head_dim / 4)
        } else {
            (10_000.0f32, head_dim)
        };
        // Per-layer KV head count: global (full-attention) layers on g12b are
        // value-less MQA(1), sliding layers are GQA(8) -- was hardcoded to
        // `cfg.num_key_value_heads` (the sliding count), mis-sizing every
        // global layer's K/V. Mirrors `forward_layer_gpu_matmuls`'s identical
        // fix (gemma_forward.rs) and `cfg.layer_num_kv_heads`.
        let num_kv = cfg.layer_num_kv_heads(layer_idx);
        // Value-less global attention: no v_proj tensor on disk; V is derived
        // from the raw (pre-k_norm) K via weightless rms-norm.
        let uses_k_eq_v = cfg.layer_uses_k_eq_v(layer_idx);
        LayerSpec {
            is_qwen: false,
            hidden: cfg.hidden_size,
            head_dim,
            num_q: cfg.num_attention_heads,
            num_kv,
            inter: cfg.layer_intermediate_size(layer_idx),
            eps: cfg.rms_norm_eps,
            attn_scale: 1.0,   // gemma SDPA uses scale 1.0 (q already scaled)
            k_norm: !is_kv_shared,
            v_norm: !is_kv_shared,
            ffn_in_norm: "pre_feedforward_layernorm.weight".to_string(),
            sandwich: true,
            post_attn_norm: Some("post_attention_layernorm.weight".to_string()),
            post_ffn_norm: Some("post_feedforward_layernorm.weight".to_string()),
            theta,
            rotary_dim,
            freq_dim: head_dim,
            window: if is_full { None } else { Some(cfg.sliding_window) },
            kv_shared: is_kv_shared,
            activation: Activation::Gelu,
            // `layer_scalar` applies on EVERY gemma layer (has_ple() or not
            // -- see gemma_forward.rs's `hidden3 *= layer_scalar` after the
            // `if cfg.has_ple() { ... }` PLE block), so PleSpec itself is
            // still always populated; `unified_ple_tail` is the one that
            // gates the PLE-specific tensor lookups on `ple_dim > 0`.
            ple: Some(PleSpec {
                ple_dim: cfg.hidden_size_per_layer_input,
                layer_ple,
                layer_scalar,
            }),
            uses_k_eq_v,
        }
    }
}

/// Every `gpu_weights` key the unified path will INDEX (`self.gpu_weights[..]`,
/// which panics on a miss — it is not a lookup) while executing `layer_idx`
/// under `spec`.
///
/// Kept beside `LayerSpec` and derived from the SAME spec fields the recording
/// paths branch on, so the pre-flight and the execution cannot drift:
///
///  - q / k / o / gate / up / down: indexed unconditionally by both
///    `gpu_layer_1cb` and `gpu_layer_2cb`.
///  - v: NOT required when `spec.uses_k_eq_v`. Value-less global Gemma layers
///    carry no `v_proj` tensor on disk AT ALL; `is_layer_1cb_eligible` routes
///    them to `gpu_layer_2cb`, which reads `v_proj` only under
///    `if spec.uses_k_eq_v { .. } else { .. }`. Demanding the key here would
///    re-break the very layers an earlier fix un-broke, only from the other
///    side (a spurious refusal instead of a panic).
///  - PLE gate/projection: required only when `ple.ple_dim > 0`.
///    `LayerSpec::gemma` populates `ple: Some(..)` on EVERY gemma layer (the
///    `layer_scalar` multiply is unconditional), so `ple.is_some()` is NOT the
///    condition — `unified_ple_tail` gates the two tensor lookups on
///    `ple_dim > 0`, and g12b/g31b (`hidden_size_per_layer_input == 0`) carry
///    no `per_layer*` tensors on disk.
///
/// Deliberately NOT included: `model.layers.{i}.layer_scalar` and the norm
/// weights. Those come from `self.inner.weights` / `unified_norm_w`, not
/// `gpu_weights`, and the norms already have their own `ensure_unified_norm`
/// readiness pass.
pub(crate) fn unified_layer_weight_keys(spec: &LayerSpec, layer_idx: usize) -> Vec<String> {
    let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
    let mut keys = vec![
        ln("self_attn.q_proj.weight"),
        ln("self_attn.k_proj.weight"),
        ln("self_attn.o_proj.weight"),
        ln("mlp.gate_proj.weight"),
        ln("mlp.up_proj.weight"),
        ln("mlp.down_proj.weight"),
    ];
    if !spec.uses_k_eq_v {
        keys.push(ln("self_attn.v_proj.weight"));
    }
    if spec.ple.as_ref().is_some_and(|p| p.ple_dim > 0) {
        keys.push(ln("per_layer_input_gate.weight"));
        keys.push(ln("per_layer_projection.weight"));
    }
    keys
}

/// First `gpu_weights` key that is missing for ANY executed layer, or `None` if
/// every layer is fully backed.
///
/// `specs` must be exactly the layers the caller will EXECUTE — the qwen path
/// runs `0..num_hidden_layers`, the gemma path runs `pp_start..pp_end`. Passing
/// the whole model on a PP rank that owns a slice would turn this guard into a
/// spurious startup fallback on every non-first stage.
pub(crate) fn first_missing_unified_weight(
    specs: &[(usize, LayerSpec)],
    present: impl Fn(&str) -> bool,
) -> Option<String> {
    for (layer_idx, spec) in specs {
        for key in unified_layer_weight_keys(spec, *layer_idx) {
            if !present(&key) { return Some(key); }
        }
    }
    None
}

impl VulkanModel {
    /// Stable raw pointer to one persistent `UR_*` unified activation buffer.
    ///
    /// Raw, not a reference, so the immutable borrow of `self.ures_bufs` ends
    /// before `self.engine.as_mut()` takes its mutable borrow — one recording
    /// needs both. Valid only while `ures_bufs` is not reallocated:
    /// `init_ures_bufs` fills it once and never resizes it, so no dispatch may
    /// allocate activation buffers mid-recording.
    pub(crate) fn ures_ptr(&self, slot: usize) -> *const compute::Buffer {
        &self.ures_bufs[slot] as *const compute::Buffer
    }
    /// Mutable twin of `ures_ptr` for the slots written host-side before a
    /// recording (rope position, dummies). Same non-reallocation requirement.
    pub(crate) fn ures_ptr_mut(&mut self, slot: usize) -> *mut compute::Buffer {
        &mut self.ures_bufs[slot] as *mut compute::Buffer
    }
    /// Allocate the persistent unified activation buffers once. Sized from the
    /// LOADED model's max dims (a VulkanModel is qwen XOR gemma).
    pub(crate) fn init_ures_bufs(&mut self) -> bool {
        if self.ures_ready { return true; }
        // Dimensions: take the per-element max the loaded arch can produce.
        let (h, q_dim, kv_dim, max_inter, ple_dim, vocab) = if let Some(q) = self.qwen.as_ref() {
            let c = &q.config;
            (c.hidden_size, c.num_attention_heads * c.head_dim,
             c.num_key_value_heads * c.head_dim, c.intermediate_size, 1usize, c.vocab_size)
        } else {
            let c = &self.inner.config;
            (c.hidden_size, c.num_attention_heads * c.global_head_dim,
             c.num_key_value_heads * c.global_head_dim, c.intermediate_size * 2,
             c.hidden_size_per_layer_input, c.vocab_size)
        };
        let sizes: [u64; UR_COUNT] = [
            (h * 4) as u64,          // UR_HA
            (h * 4) as u64,          // UR_HB
            (h * 4) as u64,          // UR_X
            (q_dim * 4) as u64,      // UR_Q
            (kv_dim * 4) as u64,     // UR_K
            (kv_dim * 4) as u64,     // UR_V
            (q_dim * 4) as u64,      // UR_ATTN
            (h * 4) as u64,          // UR_O
            (h * 4) as u64,          // UR_ON
            (h * 4) as u64,          // UR_FFIN
            (max_inter * 4) as u64,  // UR_GATE
            (max_inter * 4) as u64,  // UR_UP
            (max_inter * 4) as u64,  // UR_ACT
            (max_inter * 4) as u64,  // UR_MID
            (h * 4) as u64,          // UR_DOWN
            (h * 4) as u64,          // UR_DOWNN
            4,                       // UR_POS
            4,                       // UR_FF
            8,                       // UR_IDX
            (h * 4) as u64,          // UR_DUMMY
            (ple_dim * 4) as u64,    // UR_PLE_G
            (h * 4) as u64,          // UR_PLE_C
            (vocab * 4) as u64,      // UR_LOGITS
        ];
        let eng = match self.engine.as_mut() { Some(e) => e, None => return false };
        let mut bufs = Vec::with_capacity(UR_COUNT);
        for &sz in &sizes {
            match eng.alloc_host_coherent_storage(sz.max(4)) {
                Ok(b) => bufs.push(b),
                Err(e) => { log::warn!("init_ures_bufs alloc failed: {e}"); return false; }
            }
        }
        bufs[UR_FF].write(&1.0f32.to_le_bytes()).ok();
        bufs[UR_IDX].write(&0u64.to_le_bytes()).ok();
        self.ures_bufs = bufs;
        self.ures_ready = true;
        true
    }
    /// Upload a norm weight into `unified_norm_w` once (stable pointer). Returns
    /// false if absent / too short. `qwen` selects which weight store to read.
    pub(crate) fn ensure_unified_norm(&mut self, name: &str, n: usize, qwen: bool) -> bool {
        if self.unified_norm_w.contains_key(name) { return true; }
        let w: Vec<f32> = if qwen {
            self.qwen.as_ref().unwrap().weights.f32_slice(name).to_vec()
        } else {
            self.inner.weights.f32_slice(name).to_vec()
        };
        if w.len() < n { return false; }
        let eng = match self.engine.as_mut() { Some(e) => e, None => return false };
        let buf = match eng.alloc_host_coherent_storage((n * 4) as u64) {
            Ok(b) => b, Err(_) => return false,
        };
        if buf.write(&f32_slice_to_bytes(&w[..n])).is_err() { return false; }
        self.unified_norm_w.insert(name.to_string(), buf);
        true
    }
    /// Record + execute ONE decoder layer through the unified engine. Hidden
    /// enters in UR_HA and leaves in UR_HA (ping-pong via UR_HB). All norm
    /// weights named in `spec` must already be in `unified_norm_w`, and every
    /// projection weight in `gpu_weights`, so no map inserts happen mid-record
    /// (which would invalidate the raw Buffer pointers).
    ///
    /// Dispatches to the 1-CB resident-KV path (1 submit/layer) when
    /// VLLM_VULKAN_UNIFIED_1CB=1 and the layer type is eligible; otherwise the
    /// proven 2-CB host-split path (2 submits/layer).
    pub(crate) fn gpu_layer(&mut self, spec: &LayerSpec, layer_idx: usize, pos: usize) -> GpuResult<()> {
        if unified_1cb_enabled() && self.is_layer_1cb_eligible(spec, layer_idx) {
            self.gpu_layer_1cb(spec, layer_idx, pos)
        } else {
            self.gpu_layer_2cb(spec, layer_idx, pos)
        }
    }
    /// Is this layer type eligible for the 1-CB resident-KV path?
    ///   qwen   : always (uniform full causal).
    ///   gemma  : global (full causal), sliding (windowed), and KV-shared layers
    ///            are all handled by gpu_layer_1cb. The PLE tail still runs as a
    ///            small separate submit afterward (it is host-glued, not part of
    ///            the attention split this work targets), so a PLE layer is still
    ///            "1-CB for the attention block" — its layer body is one submit.
    /// Returns false only if the resident-KV target buffer can't be guaranteed
    /// (defensive; in practice all gemma/qwen layer types are covered).
    pub(crate) fn is_layer_1cb_eligible(&self, spec: &LayerSpec, _layer_idx: usize) -> bool {
        // Value-less global gemma layers (`uses_k_eq_v`: no v_proj tensor, V
        // derived from raw pre-k_norm K) aren't wired into the fused
        // resident-KV append/SDPA path yet -- fall back to the (fully
        // correct) 2-CB host-split path for those specific layers. All other
        // gemma layer types (sliding, kv-shared) and all qwen layers stay on
        // the 1-CB path.
        !spec.uses_k_eq_v
    }
    /// Ensure the per-layer resident KV buffer exists (lazily, [K-plane][V-plane],
    /// each plane `capacity*num_kv*head_dim` f32). Returns the raw pointer (stable —
    /// gpu_kv buffers are never reallocated once inserted). Must be called BEFORE
    /// any command-buffer recording so the map insert can't move other entries'
    /// pointers mid-record.
    ///
    /// `capacity` is the PHYSICAL slot count of the plane: `max_seq` for full /
    /// global layers (absolute addressing), or the sliding `window` for a
    /// per-layer-sized RING plane (the caller then drives the SDPA with a matching
    /// `ring_capacity` push-constant and appends at slot `pos % capacity`). Full
    /// layers pass `capacity == max_seq`, preserving the historical allocation
    /// byte-for-byte.
    pub(crate) fn ensure_gpu_kv(&mut self, layer: usize, num_kv: usize, head_dim: usize,
                     capacity: usize) -> Option<*const compute::Buffer> {
        let plane = capacity * num_kv * head_dim;
        if !self.gpu_kv.contains_key(&layer) {
            let eng = self.engine.as_mut()?;
            match eng.alloc_host_coherent_storage((2 * plane * 4) as u64) {
                Ok(b) => { self.gpu_kv.insert(layer, b); }
                Err(e) => { log::warn!("ensure_gpu_kv alloc failed: {e}"); return None; }
            }
        }
        self.gpu_kv.get(&layer).map(|b| b as *const compute::Buffer)
    }
    /// Record + execute ONE decoder layer as a SINGLE command buffer (1 submit).
    /// K/V stay GPU-resident: after RoPE, this records a buffer-copy of UR_K/UR_V
    /// into gpu_kv[layer] (K-plane / V-plane), a TRANSFER→COMPUTE barrier, then
    /// paged_attn_decode_f32 reading gpu_kv[layer] + UR_Q → UR_ATTN, then the rest
    /// of the layer — all in one CB. No host readback of q/k/v, no host KV append.
    /// Only the CPU `KvCache.seq_len` bookkeeping is updated (for the SDPA seq_len
    /// push-constant and the sliding-window start). Gated by VLLM_VULKAN_UNIFIED_1CB.
    pub(crate) fn gpu_layer_1cb(&mut self, spec: &LayerSpec, layer_idx: usize, pos: usize) -> GpuResult<()> {
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
        let h = spec.hidden;
        let eps = spec.eps;
        let head_dim = spec.head_dim;
        let num_q = spec.num_q;
        let num_kv = spec.num_kv;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let inter = spec.inter;
        let scale = spec.attn_scale;
        let max_seq = self.max_seq_len;

        // Bound `pos` against the PHYSICAL resident plane BEFORE any offset is
        // computed. `k_dst_off`/`v_dst_off` below are raw byte offsets into the
        // gpu_kv plane, consumed by `vkCmdCopyBuffer` — an out-of-range `pos`
        // is a device-side write past the end of the buffer (memory corruption,
        // not a Rust panic). `pos` reaches here from the Python decode driver,
        // so it is untrusted. Same precedent as `forward_prefill_gemma`
        // (lib.rs: "prompt length {t} exceeds max_seq_len").
        if pos >= max_seq {
            return Err(format!(
                "gpu_layer_1cb: position {pos} exceeds max_seq_len {max_seq} — \
                 construct VulkanModel with a larger max_seq_len").into());
        }

        // ── KV bookkeeping (no host data copy) ───────────────────────────────
        // Determine which resident KV buffer this layer's SDPA reads, the seq_len
        // (positions to attend), and whether this layer appends its own K/V.
        let (kv_layer, append, slen) = if spec.is_qwen {
            // Invariant: spec.is_qwen implies this model was loaded as Qwen3.
            let qm = self.qwen.as_mut().expect("invariant: spec.is_qwen implies self.qwen is Some");
            qm.kv_caches[layer_idx].seq_len = pos + 1;
            (layer_idx, true, pos + 1)
        } else if spec.kv_shared {
            // KV-shared: read the target layer's resident KV; do NOT append.
            let target = self.inner.kv_shared_target(layer_idx);
            let slen = self.inner.kv_caches[target].seq_len.max(pos + 1);
            (target, false, slen)
        } else {
            self.inner.kv_caches[layer_idx].seq_len = pos + 1;
            (layer_idx, true, pos + 1)
        };
        // Sliding-window start (gemma sliding layers); 0 = full causal.
        let window_start = spec.window.map(|w| slen.saturating_sub(w)).unwrap_or(0);

        // Allocate the resident KV buffer for the *target* layer up front (stable
        // pointer; never reallocated). For an appending layer this is its own;
        // for a KV-shared layer it must already exist (target ran earlier) — if it
        // somehow doesn't, allocate it so the record can't panic.
        let kv_ptr = match self.ensure_gpu_kv(kv_layer, num_kv, head_dim, max_seq) {
            Some(p) => p,
            None => { // GPU unavailable — fall back to the 2-CB host path.
                return self.gpu_layer_2cb(spec, layer_idx, pos);
            }
        };
        let plane = max_seq * kv_dim; // elements per K (or V) plane

        unsafe { (*self.ures_ptr_mut(UR_POS)).write(&(pos as i32).to_le_bytes())?; }

        let rope_pc_q = rope_neox_pc(num_q, head_dim, spec.rotary_dim, spec.freq_dim, spec.theta);
        let rope_pc_k = rope_neox_pc(num_kv, head_dim, spec.rotary_dim, spec.freq_dim, spec.theta);
        let rope_wgy = (((head_dim / 2) as u32) + 255) / 256;

        // Gather every buffer pointer BEFORE recording (no map inserts mid-record).
        let ha = self.ures_ptr(UR_HA);
        let hb = self.ures_ptr(UR_HB);
        let xp = self.ures_ptr(UR_X);
        let qp = self.ures_ptr(UR_Q);
        let kp = self.ures_ptr(UR_K);
        let vp = self.ures_ptr(UR_V);
        let posp = self.ures_ptr(UR_POS);
        let ffp = self.ures_ptr(UR_FF);
        let idxp = self.ures_ptr(UR_IDX);
        let btp = self.ures_ptr(UR_IDX);   // block-table [0] (UR_IDX is zero-inited)
        let attnp = self.ures_ptr(UR_ATTN);
        let op = self.ures_ptr(UR_O);
        let onp = self.ures_ptr(UR_ON);
        let ffin = self.ures_ptr(UR_FFIN);
        let gate = self.ures_ptr(UR_GATE);
        let up = self.ures_ptr(UR_UP);
        let act = self.ures_ptr(UR_ACT);
        let mid = self.ures_ptr(UR_MID);
        let down = self.ures_ptr(UR_DOWN);
        let downn = self.ures_ptr(UR_DOWNN);
        let dummy = self.ures_ptr(UR_DUMMY);
        let inln_p  = &self.unified_norm_w[&ln("input_layernorm.weight")] as *const compute::Buffer;
        let qnorm_p = &self.unified_norm_w[&ln("self_attn.q_norm.weight")] as *const compute::Buffer;
        let knorm_p = if spec.k_norm {
            Some(&self.unified_norm_w[&ln("self_attn.k_norm.weight")] as *const compute::Buffer)
        } else { None };
        let inln_after = &self.unified_norm_w[&ln(&spec.ffn_in_norm)] as *const compute::Buffer;
        let pa_p = spec.post_attn_norm.as_ref()
            .map(|n| &self.unified_norm_w[&ln(n)] as *const compute::Buffer);
        let postff_p = spec.post_ffn_norm.as_ref()
            .map(|n| &self.unified_norm_w[&ln(n)] as *const compute::Buffer);
        let qw = &self.gpu_weights[&ln("self_attn.q_proj.weight")].buffer as *const compute::Buffer;
        let kw = &self.gpu_weights[&ln("self_attn.k_proj.weight")].buffer as *const compute::Buffer;
        let vw = &self.gpu_weights[&ln("self_attn.v_proj.weight")].buffer as *const compute::Buffer;
        let ow = &self.gpu_weights[&ln("self_attn.o_proj.weight")].buffer as *const compute::Buffer;
        let gw = &self.gpu_weights[&ln("mlp.gate_proj.weight")].buffer as *const compute::Buffer;
        let uw = &self.gpu_weights[&ln("mlp.up_proj.weight")].buffer as *const compute::Buffer;
        let dw = &self.gpu_weights[&ln("mlp.down_proj.weight")].buffer as *const compute::Buffer;

        // Per-weight-format dispatch (see the identical comment in
        // `gpu_layer_2cb`): gemma4_unified's attn is Mlx4, MLP is Q8_0,
        // unconditionally, regardless of the global `VLLM_VULKAN_QUANT` flag
        // `matvec_variant` reads. `is_layer_1cb_eligible` keeps value-less
        // (uses_k_eq_v) layers off this path entirely, so `v_proj` always
        // exists here.
        let (q_fmt, q_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("self_attn.q_proj.weight")]);
        let (k_fmt, k_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("self_attn.k_proj.weight")]);
        let (v_fmt, v_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("self_attn.v_proj.weight")]);
        let (o_fmt, o_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("self_attn.o_proj.weight")]);
        let (g_fmt, g_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("mlp.gate_proj.weight")]);
        let (u_fmt, u_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("mlp.up_proj.weight")]);
        let (d_fmt, d_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("mlp.down_proj.weight")]);
        let rms_h = rmsnorm_pc(h, eps);
        let rms_hd = rmsnorm_pc(head_dim, eps);
        let rms_h2 = rmsnorm_pc(h, eps);
        let add_pc = ew_mul_pc(h as u32);
        let act_pc = ew_unary_pc(inter as u32);
        let mul_pc = ew_mul_pc(inter as u32);
        let act_shader = spec.activation.shader();
        let elem = inter as u32;
        let sandwich = spec.sandwich;
        // paged_attn push-constants: block_size = max_seq (resident plane size),
        // plane = max_seq*kv_dim, window_start for sliding layers.
        // ring_capacity = 0: the unified resident plane stays full max_seq-sized
        // (absolute addressing). Per-layer ring sizing is wired only in the gemma
        // resident 1-CB path for now.
        let sdpa = sdpa_pc(slen, num_q, num_kv, head_dim, max_seq, plane, scale, window_start, 0);
        // Decode-attention kernel + its dispatch geometry (see attn_decode_kernel):
        //  - paged_attn_decode_f32:     one thread per output element (q_dim/256 wgs)
        //  - _sg:   one wave64 subgroup per q_head  -> num_q workgroups of 64
        //  - _coop: one workgroup per q_head, output tiled by 256 -> (num_q, hd/256)
        let sdpa_kernel = attn_decode_kernel();
        let sdpa_wg = match sdpa_kernel {
            "paged_attn_decode_f32_sg"   => (num_q as u32, 1u32, 1u32),
            "paged_attn_decode_f32_coop" => (num_q as u32, (head_dim as u32 + 255) / 256, 1u32),
            _                            => ((q_dim as u32 + 255) / 256, 1u32, 1u32),
        };
        // Byte offsets of this token's K/V slot inside the resident plane.
        let k_dst_off = (pos * kv_dim * 4) as u64;
        let v_dst_off = ((plane + pos * kv_dim) * 4) as u64;
        let kv_copy_sz = (kv_dim * 4) as u64;

        let eng = self.engine.as_mut().expect("invariant: gpu_layer_1cb only called when self.engine is Some");
        let cb = eng.begin_batch()?;
        unsafe {
            // input_norm → X
            eng.record_to(cb, "rms_norm_f32_mul", &[&*ha, &*inln_p, &*xp], &rms_h, (1, 1, 1))?;
            eng.record_barrier_to(cb);
            // q/k/v proj
            crate::gemma_forward::record_gemma_mv(eng, cb, qw, q_fmt, q_kind, xp, qp, h, q_dim);
            crate::gemma_forward::record_gemma_mv(eng, cb, kw, k_fmt, k_kind, xp, kp, h, kv_dim);
            crate::gemma_forward::record_gemma_mv(eng, cb, vw, v_fmt, v_kind, xp, vp, h, kv_dim);
            eng.record_barrier_to(cb);
            // q/k/v norm
            eng.record_to(cb, "rms_norm_f32_mul", &[&*qp, &*qnorm_p, &*qp], &rms_hd, (num_q as u32, 1, 1))?;
            if let Some(knp) = knorm_p {
                eng.record_to(cb, "rms_norm_f32_mul", &[&*kp, &*knp, &*kp], &rms_hd, (num_kv as u32, 1, 1))?;
            }
            if spec.v_norm {
                eng.record_to(cb, "rms_norm_f32", &[&*vp, &*ffp, &*vp], &rms_hd, (num_kv as u32, 1, 1))?;
            }
            eng.record_barrier_to(cb);
            // RoPE (in place)
            eng.record_to(cb, "rope_neox_f32_f32", &[&*qp, &*posp, &*ffp, &*qp, &*idxp], &rope_pc_q, (num_q as u32, rope_wgy, 1))?;
            eng.record_to(cb, "rope_neox_f32_f32", &[&*kp, &*posp, &*ffp, &*kp, &*idxp], &rope_pc_k, (num_kv as u32, rope_wgy, 1))?;

            // ── Resident-KV append (recorded copy, no host readback) ─────────
            // An appending layer copies this token's RoPE'd K/V into its resident
            // plane. A KV-shared layer skips this (its SDPA reads the target's KV).
            // RoPE'd Q (SHADER_WRITE, COMPUTE) feeds the SDPA dispatch below
            // (SHADER_READ, COMPUTE). This barrier is needed on BOTH branches:
            // the append branch's COMPUTE→TRANSFER + TRANSFER→COMPUTE pair
            // covers only K/V (it makes TRANSFER_WRITEs visible to SHADER_READ,
            // never the COMPUTE SHADER_WRITE of Q), so without it Q's write is
            // never made visible to the SDPA read on the hot path.
            eng.record_barrier_to(cb);
            if append {
                // RoPE wrote UR_K/UR_V (COMPUTE) → the copy reads them (TRANSFER).
                eng.record_compute_to_transfer_barrier(cb);
                eng.record_copy_to(cb, &*kp, &*kv_ptr, 0, k_dst_off, kv_copy_sz);
                eng.record_copy_to(cb, &*vp, &*kv_ptr, 0, v_dst_off, kv_copy_sz);
                // The copy wrote gpu_kv (TRANSFER) → SDPA reads it (COMPUTE).
                eng.record_transfer_to_compute_barrier(cb);
            }

            // ── Resident SDPA: Q (UR_Q) over gpu_kv[kv_layer] → UR_ATTN ───────
            eng.record_to(cb, sdpa_kernel, &[&*qp, &*btp, &*kv_ptr, &*attnp], &sdpa, sdpa_wg)?;
            eng.record_barrier_to(cb);

            // ── o_proj → residual → ffn_in_norm → FFN → residual2 ────────────
            crate::gemma_forward::record_gemma_mv(eng, cb, ow, o_fmt, o_kind, attnp, op, q_dim, h);
            eng.record_barrier_to(cb);
            if sandwich {
                // Invariant: spec.sandwich implies spec.post_attn_norm is Some.
                let pa = pa_p.expect("invariant: sandwich layer must carry post_attn_norm");
                eng.record_to(cb, "rms_norm_f32_mul", &[&*op, &*pa, &*onp], &rms_h2, (1, 1, 1))?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "add_f32_f32_f32", &[&*ha, &*onp, &*hb, &*dummy], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
            } else {
                eng.record_to(cb, "add_f32_f32_f32", &[&*ha, &*op, &*hb, &*dummy], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
            }
            eng.record_barrier_to(cb);
            eng.record_to(cb, "rms_norm_f32_mul", &[&*hb, &*inln_after, &*ffin], &rms_h2, (1, 1, 1))?;
            eng.record_barrier_to(cb);
            crate::gemma_forward::record_gemma_mv(eng, cb, gw, g_fmt, g_kind, ffin, gate, h, inter);
            crate::gemma_forward::record_gemma_mv(eng, cb, uw, u_fmt, u_kind, ffin, up, h, inter);
            eng.record_barrier_to(cb);
            eng.record_to(cb, act_shader, &[&*gate, &*act], &act_pc, ((elem + 511)/512, 1, 1))?;
            eng.record_barrier_to(cb);
            eng.record_to(cb, "mul_f32_f32_f32", &[&*act, &*up, &*mid], &mul_pc, ((elem + 255)/256, 1, 1))?;
            eng.record_barrier_to(cb);
            crate::gemma_forward::record_gemma_mv(eng, cb, dw, d_fmt, d_kind, mid, down, inter, h);
            eng.record_barrier_to(cb);
            if sandwich {
                // Invariant: spec.sandwich implies spec.post_ffn_norm is Some.
                let pf = postff_p.expect("invariant: sandwich layer must carry post_ffn_norm");
                eng.record_to(cb, "rms_norm_f32_mul", &[&*down, &*pf, &*downn], &rms_h2, (1, 1, 1))?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "add_f32_f32_f32", &[&*hb, &*downn, &*ha, &*dummy], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
            } else {
                eng.record_to(cb, "add_f32_f32_f32", &[&*hb, &*down, &*ha, &*dummy], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
            }
        }
        let ts = std::time::Instant::now();
        eng.submit_batch(cb)?;
        prof_add("layer_submit_fence", ts);

        // ── PLE tail (gemma only) — small host-glued separate submit ─────────
        if let Some(ple) = spec.ple.as_ref() {
            self.unified_ple_tail(layer_idx, h, eps, ple)?;
        }
        Ok(())
    }
    /// The proven 2-CB host-split decoder layer (CB1 = norm→qkv→rope, host KV
    /// append + SDPA, CB2 = o_proj→FFN→residual). 2 submits/layer. Used when
    /// VLLM_VULKAN_UNIFIED_1CB is off, or as a defensive fallback.
    pub(crate) fn gpu_layer_2cb(&mut self, spec: &LayerSpec, layer_idx: usize, pos: usize) -> GpuResult<()> {
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
        let h = spec.hidden;
        let eps = spec.eps;
        let head_dim = spec.head_dim;
        let num_q = spec.num_q;
        let num_kv = spec.num_kv;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let inter = spec.inter;
        let scale = spec.attn_scale;

        unsafe { (*self.ures_ptr_mut(UR_POS)).write(&(pos as i32).to_le_bytes())?; }

        let rope_pc_q = rope_neox_pc(num_q, head_dim, spec.rotary_dim, spec.freq_dim, spec.theta);
        let rope_pc_k = rope_neox_pc(num_kv, head_dim, spec.rotary_dim, spec.freq_dim, spec.theta);
        let rope_wgy = (((head_dim / 2) as u32) + 255) / 256;

        // ── CB1: input_norm → q/k/v → q/k/v-norm → RoPE (one submit) ─────────
        let ha = self.ures_ptr(UR_HA);
        let xp = self.ures_ptr(UR_X);
        let qp = self.ures_ptr(UR_Q);
        let kp = self.ures_ptr(UR_K);
        let vp = self.ures_ptr(UR_V);
        let posp = self.ures_ptr(UR_POS);
        let ffp = self.ures_ptr(UR_FF);
        let idxp = self.ures_ptr(UR_IDX);
        let inln_p  = &self.unified_norm_w[&ln("input_layernorm.weight")] as *const compute::Buffer;
        let qnorm_p = &self.unified_norm_w[&ln("self_attn.q_norm.weight")] as *const compute::Buffer;
        let knorm_p = if spec.k_norm {
            Some(&self.unified_norm_w[&ln("self_attn.k_norm.weight")] as *const compute::Buffer)
        } else { None };
        // Dispatch each matvec by the WEIGHT'S OWN recorded format (mirrors
        // `gemma_res_mv_kind`/`record_gemma_mv` in gemma_forward.rs), not the
        // blanket `matvec_variant` (which keys off the GLOBAL
        // `VLLM_VULKAN_QUANT` snapshot). gemma4_unified uploads attn as Mlx4
        // (4-bit affine) and MLP as Q8_0 UNCONDITIONALLY regardless of that
        // flag -- `matvec_variant`'s shader choice silently mismatches the
        // packed bytes (reads Mlx4-packed nibbles as if they were f16/q8_0
        // blocks) and produces NaN. Qwen3's resident weights ARE uniformly
        // the global format, so this is a no-op shader choice for it (same
        // format either way) and stays argmax-exact.
        let (q_fmt, q_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("self_attn.q_proj.weight")]);
        let (k_fmt, k_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("self_attn.k_proj.weight")]);
        let v_meta = if spec.uses_k_eq_v { None } else {
            Some(crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("self_attn.v_proj.weight")]))
        };
        let qw = &self.gpu_weights[&ln("self_attn.q_proj.weight")].buffer as *const compute::Buffer;
        let kw = &self.gpu_weights[&ln("self_attn.k_proj.weight")].buffer as *const compute::Buffer;
        let vw = if spec.uses_k_eq_v { None } else {
            Some(&self.gpu_weights[&ln("self_attn.v_proj.weight")].buffer as *const compute::Buffer)
        };
        let rms_h = rmsnorm_pc(h, eps);
        let rms_hd = rmsnorm_pc(head_dim, eps);

        let eng = self.engine.as_mut().expect("invariant: gpu_layer_2cb only called when self.engine is Some");
        let cb = eng.begin_batch()?;
        unsafe {
            eng.record_to(cb, "rms_norm_f32_mul", &[&*ha, &*inln_p, &*xp], &rms_h, (1, 1, 1))?;
            eng.record_barrier_to(cb);
            crate::gemma_forward::record_gemma_mv(eng, cb, qw, q_fmt, q_kind, xp, qp, h, q_dim);
            crate::gemma_forward::record_gemma_mv(eng, cb, kw, k_fmt, k_kind, xp, kp, h, kv_dim);
            // Value-less global (full-attention) gemma layers have NO v_proj
            // tensor on disk at all (`cfg.layer_uses_k_eq_v`/`spec.uses_k_eq_v`);
            // V is derived from the RAW (pre-k_norm) K below, mirroring the base
            // path's `forward_layer_gpu_matmuls`/`gemma_resident_layer`.
            if let (Some(vw), Some((v_fmt, v_kind))) = (vw, v_meta) {
                crate::gemma_forward::record_gemma_mv(eng, cb, vw, v_fmt, v_kind, xp, vp, h, kv_dim);
            }
            eng.record_barrier_to(cb);
            // q-norm (weighted, per head).
            eng.record_to(cb, "rms_norm_f32_mul", &[&*qp, &*qnorm_p, &*qp], &rms_hd, (num_q as u32, 1, 1))?;
            if spec.uses_k_eq_v {
                // Value-less global: V = v_norm_no_weight(RAW k_proj), computed
                // BEFORE k_norm mutates K in place (mirrors the CPU reference's
                // `v_raw = k_raw.clone()` taken before k_norm). A barrier
                // separates the two dispatches since both read/write the same
                // K buffer and one of them (k_norm) then overwrites it.
                eng.record_to(cb, "rms_norm_f32", &[&*kp, &*ffp, &*vp], &rms_hd, (num_kv as u32, 1, 1))?;
                eng.record_barrier_to(cb);
                if let Some(knp) = knorm_p {
                    eng.record_to(cb, "rms_norm_f32_mul", &[&*kp, &*knp, &*kp], &rms_hd, (num_kv as u32, 1, 1))?;
                }
            } else {
                if let Some(knp) = knorm_p {
                    eng.record_to(cb, "rms_norm_f32_mul", &[&*kp, &*knp, &*kp], &rms_hd, (num_kv as u32, 1, 1))?;
                }
                if spec.v_norm {
                    // v-norm is NO-WEIGHT: rms_norm_f32, output at binding 2
                    // (binding 1 is an ignored dummy for the do_multiply=false path).
                    eng.record_to(cb, "rms_norm_f32", &[&*vp, &*ffp, &*vp], &rms_hd, (num_kv as u32, 1, 1))?;
                }
            }
            eng.record_barrier_to(cb);
            // NeoX RoPE in place (partial-rotary tail passes through; ne00=head_dim).
            eng.record_to(cb, "rope_neox_f32_f32", &[&*qp, &*posp, &*ffp, &*qp, &*idxp], &rope_pc_q, (num_q as u32, rope_wgy, 1))?;
            eng.record_to(cb, "rope_neox_f32_f32", &[&*kp, &*posp, &*ffp, &*kp, &*idxp], &rope_pc_k, (num_kv as u32, rope_wgy, 1))?;
        }
        let ts = std::time::Instant::now();
        eng.submit_batch(cb)?;
        prof_add("layer_submit_fence", ts);

        // ── Host boundary: KV cache append + (windowed) attention ────────────
        let q_host = read_f32_buf(unsafe { &*qp }, q_dim);
        let k_host = read_f32_buf(unsafe { &*kp }, kv_dim);
        let v_host = read_f32_buf(unsafe { &*vp }, kv_dim);
        let attn = self.unified_attention(spec, layer_idx, &q_host, &k_host, &v_host, scale);

        // ── CB2: o → (sandwich post_attn_norm) → +residual → ffn_in_norm →
        //         act-FFN → (sandwich post_ffn_norm) → +residual2 ─────────────
        unsafe { (*self.ures_ptr_mut(UR_ATTN)).write(&f32_slice_to_bytes(&attn))?; }
        let ha = self.ures_ptr(UR_HA);
        let hb = self.ures_ptr(UR_HB);
        let attnp = self.ures_ptr(UR_ATTN);
        let op = self.ures_ptr(UR_O);
        let onp = self.ures_ptr(UR_ON);
        let ffin = self.ures_ptr(UR_FFIN);
        let gate = self.ures_ptr(UR_GATE);
        let up = self.ures_ptr(UR_UP);
        let act = self.ures_ptr(UR_ACT);
        let mid = self.ures_ptr(UR_MID);
        let down = self.ures_ptr(UR_DOWN);
        let downn = self.ures_ptr(UR_DOWNN);
        let dummy = self.ures_ptr(UR_DUMMY);
        let inln_after = &self.unified_norm_w[&ln(&spec.ffn_in_norm)] as *const compute::Buffer;
        let pa_p = spec.post_attn_norm.as_ref()
            .map(|n| &self.unified_norm_w[&ln(n)] as *const compute::Buffer);
        let postff_p = spec.post_ffn_norm.as_ref()
            .map(|n| &self.unified_norm_w[&ln(n)] as *const compute::Buffer);
        let (o_fmt, o_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("self_attn.o_proj.weight")]);
        let (g_fmt, g_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("mlp.gate_proj.weight")]);
        let (u_fmt, u_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("mlp.up_proj.weight")]);
        let (d_fmt, d_kind) = crate::gemma_forward::gemma_res_mv_kind(&self.gpu_weights[&ln("mlp.down_proj.weight")]);
        let ow = &self.gpu_weights[&ln("self_attn.o_proj.weight")].buffer as *const compute::Buffer;
        let gw = &self.gpu_weights[&ln("mlp.gate_proj.weight")].buffer as *const compute::Buffer;
        let uw = &self.gpu_weights[&ln("mlp.up_proj.weight")].buffer as *const compute::Buffer;
        let dw = &self.gpu_weights[&ln("mlp.down_proj.weight")].buffer as *const compute::Buffer;
        let add_pc = ew_mul_pc(h as u32);
        let act_pc = ew_unary_pc(inter as u32);
        let mul_pc = ew_mul_pc(inter as u32);
        let rms_h2 = rmsnorm_pc(h, eps);
        let act_shader = spec.activation.shader();
        let elem = inter as u32;
        let sandwich = spec.sandwich;

        let eng = self.engine.as_mut().expect("invariant: gpu_layer_2cb only called when self.engine is Some");
        let cb = eng.begin_batch()?;
        unsafe {
            // o_proj: ATTN → O
            crate::gemma_forward::record_gemma_mv(eng, cb, ow, o_fmt, o_kind, attnp, op, q_dim, h);
            eng.record_barrier_to(cb);
            // residual after attn. SANDWICH (gemma): post_attn_norm(O) → ON, then
            // HB = HA + ON. PRE-norm (qwen): HB = HA + O straight.
            if sandwich {
                // Invariant: spec.sandwich implies spec.post_attn_norm is Some.
                let pa = pa_p.expect("invariant: sandwich layer must carry post_attn_norm");
                eng.record_to(cb, "rms_norm_f32_mul", &[&*op, &*pa, &*onp], &rms_h2, (1, 1, 1))?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "add_f32_f32_f32", &[&*ha, &*onp, &*hb, &*dummy], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
            } else {
                eng.record_to(cb, "add_f32_f32_f32", &[&*ha, &*op, &*hb, &*dummy], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
            }
            eng.record_barrier_to(cb);
            // ffn-input norm: HB → FFIN (qwen post_attention_layernorm; gemma
            // pre_feedforward_layernorm).
            eng.record_to(cb, "rms_norm_f32_mul", &[&*hb, &*inln_after, &*ffin], &rms_h2, (1, 1, 1))?;
            eng.record_barrier_to(cb);
            // gate/up (independent)
            crate::gemma_forward::record_gemma_mv(eng, cb, gw, g_fmt, g_kind, ffin, gate, h, inter);
            crate::gemma_forward::record_gemma_mv(eng, cb, uw, u_fmt, u_kind, ffin, up, h, inter);
            eng.record_barrier_to(cb);
            // act(gate) → ACT; mid = ACT * up
            eng.record_to(cb, act_shader, &[&*gate, &*act], &act_pc, ((elem + 511)/512, 1, 1))?;
            eng.record_barrier_to(cb);
            eng.record_to(cb, "mul_f32_f32_f32", &[&*act, &*up, &*mid], &mul_pc, ((elem + 255)/256, 1, 1))?;
            eng.record_barrier_to(cb);
            // down_proj: MID → DOWN
            crate::gemma_forward::record_gemma_mv(eng, cb, dw, d_fmt, d_kind, mid, down, inter, h);
            eng.record_barrier_to(cb);
            // residual2. SANDWICH: post_ffn_norm(DOWN) → DOWNN, HA = HB + DOWNN.
            // PRE-norm: HA = HB + DOWN.
            if sandwich {
                // Invariant: spec.sandwich implies spec.post_ffn_norm is Some.
                let pf = postff_p.expect("invariant: sandwich layer must carry post_ffn_norm");
                eng.record_to(cb, "rms_norm_f32_mul", &[&*down, &*pf, &*downn], &rms_h2, (1, 1, 1))?;
                eng.record_barrier_to(cb);
                eng.record_to(cb, "add_f32_f32_f32", &[&*hb, &*downn, &*ha, &*dummy], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
            } else {
                eng.record_to(cb, "add_f32_f32_f32", &[&*hb, &*down, &*ha, &*dummy], &add_pc, ((h as u32 + 255)/256, 1, 1))?;
            }
        }
        let ts2 = std::time::Instant::now();
        eng.submit_batch(cb)?;
        prof_add("layer_submit_fence", ts2);

        // ── PLE tail (gemma only) ────────────────────────────────────────────
        if let Some(ple) = spec.ple.as_ref() {
            self.unified_ple_tail(layer_idx, h, eps, ple)?;
        }
        Ok(())
    }
    /// KV-cache append + SDPA for the unified layer. Routes qwen (full attn) vs
    /// gemma (windowed / KV-shared) by `spec`.
    pub(crate) fn unified_attention(&mut self, spec: &LayerSpec, layer_idx: usize,
                         q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
        let num_q = spec.num_q;
        let num_kv = spec.num_kv;
        let head_dim = spec.head_dim;
        if spec.is_qwen {
            let (ck, cv, slen) = {
                let qm = self.qwen.as_mut().unwrap();
                qm.kv_caches[layer_idx].append(k, v);
                let cache = &qm.kv_caches[layer_idx];
                (cache.k_up_to_now().to_vec(), cache.v_up_to_now().to_vec(), cache.seq_len)
            };
            model::cpu_sdpa(q, &ck, &cv, num_q, num_kv, head_dim, slen, scale, None)
        } else {
            let target_cache_idx = if spec.kv_shared {
                self.inner.kv_shared_target(layer_idx)
            } else {
                self.inner.kv_caches[layer_idx].append(k, v);
                layer_idx
            };
            let cache = &self.inner.kv_caches[target_cache_idx];
            model::cpu_sdpa(q, cache.k_up_to_now(), cache.v_up_to_now(),
                num_q, num_kv, head_dim, cache.seq_len, scale, spec.window)
        }
    }
    /// Gemma per-layer PLE tail: matmuls on GPU (UR_FFIN/UR_PLE_G/UR_PLE_C
    /// scratch), small CPU gelu/mul/rmsnorm glue (mirrors gemma_resident_layer).
    /// Hidden enters and leaves in UR_HA.
    pub(crate) fn unified_ple_tail(&mut self, layer_idx: usize, h: usize, eps: f32, ple: &PleSpec) -> GpuResult<()> {
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
        let ple_dim = ple.ple_dim;
        let mut hidden3 = read_f32_buf(unsafe { &*self.ures_ptr(UR_HA) }, h);

        // PLE (per-layer-embedding) contribution: `per_layer_input_gate` /
        // `per_layer_projection` / `post_per_layer_input_norm` only exist on
        // has_ple() checkpoints (E2B, ple_dim>0). g12b/g31b carry NO
        // per_layer* tensors on disk at all -- skip the whole contribution
        // for them, exactly like the base path's `if cfg.has_ple() { ... }`
        // gate in gemma_forward.rs (the `layer_scalar` multiply below still
        // runs unconditionally on every gemma layer, has_ple() or not).
        if ple_dim > 0 {
            let (ps, prr) = matvec_variant(true, ple_dim);
            let mv_pg = matvec_pc13(h, ple_dim);
            let pgw = &self.gpu_weights[&ln("per_layer_input_gate.weight")].buffer as *const compute::Buffer;
            let ffin_p = self.ures_ptr(UR_FFIN);
            let pg_p = self.ures_ptr(UR_PLE_G);
            unsafe { (*self.ures_ptr_mut(UR_FFIN)).write(&f32_slice_to_bytes(&hidden3))?; }
            let eng = self.engine.as_mut().expect("invariant: unified_ple_tail only called when self.engine is Some");
            let cb = eng.begin_batch()?;
            unsafe {
                eng.record_to(cb, &ps, &[&*pgw, &*ffin_p, &*pg_p], &mv_pg, ((ple_dim as u32 + prr - 1)/prr, 1, 1))?;
            }
            eng.submit_batch(cb)?;
            let gate_ple = read_f32_buf(unsafe { &*pg_p }, ple_dim);
            let gate_ple_act = model::cpu_gelu(&gate_ple);
            let gated: Vec<f32> = gate_ple_act.iter().zip(ple.layer_ple.iter()).map(|(&g, &p)| g * p).collect();

            let (pps, pprr) = matvec_variant(true, h);
            let mv_pp = matvec_pc13(ple_dim, h);
            let ppw = &self.gpu_weights[&ln("per_layer_projection.weight")].buffer as *const compute::Buffer;
            let pg_in = self.ures_ptr(UR_PLE_G);
            let pc_p = self.ures_ptr(UR_PLE_C);
            unsafe { (*self.ures_ptr_mut(UR_PLE_G)).write(&f32_slice_to_bytes(&gated))?; }
            let eng = self.engine.as_mut().expect("invariant: unified_ple_tail only called when self.engine is Some");
            let cb = eng.begin_batch()?;
            unsafe {
                eng.record_to(cb, &pps, &[&*ppw, &*pg_in, &*pc_p], &mv_pp, ((h as u32 + pprr - 1)/pprr, 1, 1))?;
            }
            eng.submit_batch(cb)?;
            let contrib = read_f32_buf(unsafe { &*pc_p }, h);

            let ple_norm_w = self.inner.weights.f32_slice(&ln("post_per_layer_input_norm.weight")).to_vec();
            let contrib_normed = model::cpu_rms_norm(&contrib, &ple_norm_w, eps);
            hidden3.iter_mut().zip(contrib_normed.iter()).for_each(|(hv, &c)| *hv += c);
        }
        hidden3.iter_mut().for_each(|v| *v *= ple.layer_scalar);
        unsafe { (*self.ures_ptr_mut(UR_HA)).write(&f32_slice_to_bytes(&hidden3))?; }
        Ok(())
    }
    /// UNIFIED Qwen3 decode: builds per-layer specs + drives `gpu_layer`. Numerically
    /// equal to forward_qwen_gpu_resident (one shared code path now). Falls back to
    /// the proven forward_qwen_gpu if GPU/weights/buffers aren't ready.
    pub(crate) fn forward_unified_qwen(&mut self, token_id: u32, pos: usize) -> GpuResult<Vec<f32>> {
        let cfg = self.qwen.as_ref().expect("invariant: forward_unified_qwen only called when self.qwen is Some").config.clone();
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let vocab = cfg.vocab_size;
        let eps = cfg.rms_norm_eps;

        // PRE-FLIGHT. `gpu_layer` INDEXES `self.gpu_weights[..]` for every
        // projection of every layer it records — a miss is a panic in the middle
        // of an open command buffer, not a recoverable error. Probing layer 0's
        // q_proj alone enabled the whole path on ONE key, so any other missing
        // projection (a truncated / partially-uploaded checkpoint, a loader that
        // declined to upload one tensor) got past readiness and died deep in the
        // recording loop. Validate EVERY key of EVERY executed layer instead.
        //
        // Executed range = `0..cfg.num_hidden_layers` (the recording loop below),
        // matching the existing norm-staging loop; the qwen unified path is not
        // PP-sliced. The old `contains_key("model.layers.0.self_attn.q_proj.weight")`
        // term is dropped here rather than kept alongside: layer 0 is inside that
        // range and `q_proj` is unconditional, so the scan below is a strict
        // superset of it.
        let lm_name = self.qwen.as_ref().unwrap().lm_head_name.clone();
        let mut ready = self.engine.is_some();
        if ready {
            let specs: Vec<(usize, LayerSpec)> = (0..cfg.num_hidden_layers)
                .map(|li| (li, LayerSpec::qwen(&cfg, li)))
                .collect();
            if let Some(missing) =
                first_missing_unified_weight(&specs, |k| self.gpu_weights.contains_key(k))
            {
                log::warn!("forward_unified_qwen: unified path OFF — gpu_weights has no \
                            '{missing}'; falling back to forward_qwen_gpu");
                ready = false;
            } else if !self.gpu_weights.contains_key(&lm_name) {
                // The LM head is indexed the same way after the layer loop.
                log::warn!("forward_unified_qwen: unified path OFF — gpu_weights has no \
                            lm_head '{lm_name}'; falling back to forward_qwen_gpu");
                ready = false;
            }
        }
        // Buffers are allocated only once the weights are known good (as before:
        // `init_ures_bufs` sat behind the weight probe, not in front of it).
        ready = ready && self.init_ures_bufs();
        if ready {
            // Stage all norm weights once (stable pointers during recording).
            for li in 0..cfg.num_hidden_layers {
                let p = |s: &str| format!("model.layers.{li}.{s}");
                ready &= self.ensure_unified_norm(&p("input_layernorm.weight"), h, true);
                ready &= self.ensure_unified_norm(&p("post_attention_layernorm.weight"), h, true);
                ready &= self.ensure_unified_norm(&p("self_attn.q_norm.weight"), head_dim, true);
                ready &= self.ensure_unified_norm(&p("self_attn.k_norm.weight"), head_dim, true);
                if !ready { break; }
            }
            ready &= self.ensure_unified_norm("model.norm.weight", h, true);
        }
        if !ready { return Ok(self.forward_qwen_gpu(token_id, pos)); }

        // Embedding row → UR_HA (the only host write of hidden).
        {
            let emb = {
                let w = self.qwen.as_ref().unwrap().weights.f32_slice("model.embed_tokens.weight");
                f32_slice_to_bytes(&w[token_id as usize * h..(token_id as usize + 1) * h])
            };
            unsafe { (*self.ures_ptr_mut(UR_HA)).write(&emb)?; }
        }

        for layer_idx in 0..cfg.num_hidden_layers {
            let spec = LayerSpec::qwen(&cfg, layer_idx);
            self.gpu_layer(&spec, layer_idx, pos)?;
        }

        // Final norm + LM head (no softcap). Reuse UR_X / UR_LOGITS.
        // `lm_name` was resolved (and its key validated) in the pre-flight above.
        let ha = self.ures_ptr(UR_HA);
        let xp = self.ures_ptr(UR_X);
        let logitp = self.ures_ptr(UR_LOGITS);
        let norm_p = &self.unified_norm_w["model.norm.weight"] as *const compute::Buffer;
        let lmw = &self.gpu_weights[&lm_name].buffer as *const compute::Buffer;
        let (lms, lmr) = matvec_variant(true, vocab);
        let rms_f = rmsnorm_pc(h, eps);
        let mv_lm = matvec_pc13(h, vocab);
        let eng = self.engine.as_mut().expect("invariant: forward_unified_qwen only called when self.engine is Some");
        let cb = eng.begin_batch()?;
        unsafe {
            eng.record_to(cb, "rms_norm_f32_mul", &[&*ha, &*norm_p, &*xp], &rms_f, (1, 1, 1))?;
            eng.record_barrier_to(cb);
            eng.record_to(cb, &lms, &[&*lmw, &*xp, &*logitp], &mv_lm, ((vocab as u32 + lmr - 1)/lmr, 1, 1))?;
        }
        eng.submit_batch(cb)?;
        Ok(read_f32_buf(unsafe { &*logitp }, vocab))
    }
    /// UNIFIED Gemma4 decode: builds per-layer specs + drives `gpu_layer`.
    /// Numerically equal to forward_gemma_gpu_resident. Falls back to forward_gpu.
    pub(crate) fn forward_unified_gemma(&mut self, token_id: u32, position: usize) -> GpuResult<Vec<f32>> {
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let ple_dim = cfg.hidden_size_per_layer_input;

        // PRE-FLIGHT: see the twin in `forward_unified_qwen`. Executed range is
        // `pp_start..pp_end` (the recording loop below), NOT the whole model —
        // validating layers this rank does not own would refuse every PP stage.
        //
        // The `l0_q` probe is kept as-is on purpose: `gemma_embed_and_ple` and
        // `gemma_final` around the layer loop are whole-model operations, so this
        // entry is only valid on a rank that holds the head of the model. It is a
        // weaker statement than the per-layer scan, not a redundant one.
        let l0_q = "model.layers.0.self_attn.q_proj.weight".to_string();
        let mut ready = self.engine.is_some()
            && self.gpu_weights.contains_key(&l0_q);
        if ready {
            let specs: Vec<(usize, LayerSpec)> = (self.pp_start..self.pp_end)
                // `layer_ple` / `layer_scalar` are execution inputs, not key
                // selectors: only `uses_k_eq_v` and `ple.ple_dim` (both pure
                // functions of `cfg` and `li`) decide which keys are required.
                .map(|li| (li, LayerSpec::gemma(&cfg, li, Vec::new(), 0.0)))
                .collect();
            if let Some(missing) =
                first_missing_unified_weight(&specs, |k| self.gpu_weights.contains_key(k))
            {
                log::warn!("forward_unified_gemma: unified path OFF — gpu_weights has no \
                            '{missing}'; falling back to forward_gpu");
                ready = false;
            }
        }
        ready = ready
            && self.init_ures_bufs()
            // gemma_final reuses ACT_QKV_IN for the LM-head input; ensure those
            // buffers exist (the reference inits them inside forward_layer_gpu_matmuls).
            && self.init_act_bufs();
        if ready {
            for li in self.pp_start..self.pp_end {
                let hd = cfg.layer_head_dim(li);
                let p = |s: &str| format!("model.layers.{li}.{s}");
                ready &= self.ensure_unified_norm(&p("input_layernorm.weight"), h, false);
                ready &= self.ensure_unified_norm(&p("self_attn.q_norm.weight"), hd, false);
                if !cfg.is_kv_shared(li) {
                    ready &= self.ensure_unified_norm(&p("self_attn.k_norm.weight"), hd, false);
                }
                ready &= self.ensure_unified_norm(&p("post_attention_layernorm.weight"), h, false);
                ready &= self.ensure_unified_norm(&p("pre_feedforward_layernorm.weight"), h, false);
                ready &= self.ensure_unified_norm(&p("post_feedforward_layernorm.weight"), h, false);
                if !ready { break; }
            }
        }
        if !ready { return Ok(self.forward_gpu(token_id, position)); }

        let (hidden0, ple_inputs) = self.gemma_embed_and_ple(token_id);
        unsafe { (*self.ures_ptr_mut(UR_HA)).write(&f32_slice_to_bytes(&hidden0))?; }

        for layer_idx in self.pp_start..self.pp_end {
            let layer_ple = ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim].to_vec();
            let layer_scalar = self.inner.weights.f32_slice(&format!("model.layers.{layer_idx}.layer_scalar"))[0];
            let spec = LayerSpec::gemma(&cfg, layer_idx, layer_ple, layer_scalar);
            self.gpu_layer(&spec, layer_idx, position)?;
        }

        let hidden = read_f32_buf(unsafe { &*self.ures_ptr(UR_HA) }, h);
        Ok(self.gemma_final(&hidden))
    }
}

#[cfg(test)]
mod unified_readiness_tests {
    use super::{first_missing_unified_weight, unified_layer_weight_keys, LayerSpec};
    use crate::model::{Gemma4Config, Qwen3Config};
    use std::collections::HashSet;

    fn qwen_cfg(layers: usize) -> Qwen3Config {
        Qwen3Config {
            hidden_size: 128, num_hidden_layers: layers, num_attention_heads: 4,
            num_key_value_heads: 2, head_dim: 32, intermediate_size: 256,
            vocab_size: 512, rms_norm_eps: 1e-6, rope_theta: 1e6,
            tie_word_embeddings: false,
        }
    }

    fn gemma_specs(cfg: &Gemma4Config, range: std::ops::Range<usize>) -> Vec<(usize, LayerSpec)> {
        range.map(|li| (li, LayerSpec::gemma(cfg, li, Vec::new(), 0.0))).collect()
    }

    fn qwen_specs(cfg: &Qwen3Config, range: std::ops::Range<usize>) -> Vec<(usize, LayerSpec)> {
        range.map(|li| (li, LayerSpec::qwen(cfg, li))).collect()
    }

    /// A "checkpoint": every key every listed spec needs, so the scan is clean.
    fn complete_checkpoint(specs: &[(usize, LayerSpec)]) -> HashSet<String> {
        specs.iter()
            .flat_map(|(li, s)| unified_layer_weight_keys(s, *li))
            .collect()
    }

    // ── key selection ────────────────────────────────────────────────────────

    /// Qwen needs all seven projections and never a PLE tensor.
    #[test]
    fn qwen_layer_requires_all_seven_projections_and_no_ple() {
        let cfg = qwen_cfg(2);
        let keys = unified_layer_weight_keys(&LayerSpec::qwen(&cfg, 1), 1);
        for suffix in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                       "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            assert!(keys.contains(&format!("model.layers.1.{suffix}.weight")),
                    "qwen layer must require {suffix}: {keys:?}");
        }
        assert!(!keys.iter().any(|k| k.contains("per_layer")), "{keys:?}");
        assert_eq!(keys.len(), 7);
    }

    /// ANTI-REGRESSION (v_proj). g12b's period-6 global layers are value-less
    /// (`attention_k_eq_v`): there is no `v_proj` tensor on disk at all, and an
    /// earlier fix stopped `gpu_layer_2cb` indexing one. Requiring the key here
    /// would break the same layers from the other side — the model would be
    /// refused for a tensor it is not supposed to have.
    #[test]
    fn gemma_value_less_global_layer_does_not_require_v_proj() {
        let cfg = Gemma4Config::g12b();
        assert!(cfg.layer_uses_k_eq_v(5), "layer 5 must be a value-less global layer");
        let keys = unified_layer_weight_keys(&LayerSpec::gemma(&cfg, 5, Vec::new(), 0.0), 5);
        assert!(!keys.contains(&"model.layers.5.self_attn.v_proj.weight".to_string()),
                "value-less global layer must NOT require v_proj: {keys:?}");
        assert!(keys.contains(&"model.layers.5.self_attn.k_proj.weight".to_string()));

        // And end to end: a real g12b checkpoint (no v_proj on global layers)
        // must scan clean rather than being refused.
        let specs = gemma_specs(&cfg, 0..cfg.num_hidden_layers);
        let present = complete_checkpoint(&specs);
        assert!(!present.iter().any(|k| k.starts_with("model.layers.5.self_attn.v_proj")));
        assert_eq!(first_missing_unified_weight(&specs, |k| present.contains(k)), None);
    }

    /// The conditional cuts BOTH ways: sliding layers do carry `v_proj`, and a
    /// checkpoint missing it there must still be refused.
    #[test]
    fn gemma_sliding_layer_still_requires_v_proj() {
        let cfg = Gemma4Config::g12b();
        assert!(!cfg.layer_uses_k_eq_v(0), "layer 0 must be a sliding layer");
        let specs = gemma_specs(&cfg, 0..cfg.num_hidden_layers);
        let mut present = complete_checkpoint(&specs);
        assert!(present.remove("model.layers.0.self_attn.v_proj.weight"));
        assert_eq!(first_missing_unified_weight(&specs, |k| present.contains(k)),
                   Some("model.layers.0.self_attn.v_proj.weight".to_string()));
    }

    /// ANTI-REGRESSION (PLE). `LayerSpec::gemma` sets `ple: Some(..)` on EVERY
    /// gemma layer (the `layer_scalar` multiply is unconditional), so
    /// `ple.is_some()` is the wrong test. g12b/g31b have
    /// `hidden_size_per_layer_input == 0` and carry no `per_layer*` tensors;
    /// demanding them would refuse both models outright.
    #[test]
    fn no_ple_checkpoint_does_not_require_per_layer_tensors() {
        let cfg = Gemma4Config::g12b();
        assert_eq!(cfg.hidden_size_per_layer_input, 0);
        let spec = LayerSpec::gemma(&cfg, 0, Vec::new(), 0.0);
        assert!(spec.ple.is_some(), "the trap: ple is Some even with ple_dim 0");
        let keys = unified_layer_weight_keys(&spec, 0);
        assert!(!keys.iter().any(|k| k.contains("per_layer")), "{keys:?}");
    }

    /// E2B (`hidden_size_per_layer_input == 256`) does need them.
    #[test]
    fn ple_checkpoint_requires_per_layer_tensors() {
        let cfg = Gemma4Config::e2b();
        assert!(cfg.hidden_size_per_layer_input > 0);
        let keys = unified_layer_weight_keys(&LayerSpec::gemma(&cfg, 3, Vec::new(), 0.0), 3);
        assert!(keys.contains(&"model.layers.3.per_layer_input_gate.weight".to_string()), "{keys:?}");
        assert!(keys.contains(&"model.layers.3.per_layer_projection.weight".to_string()), "{keys:?}");
    }

    // ── the readiness scan ───────────────────────────────────────────────────

    /// THE HEADLINE TEST. A checkpoint whose layer-0 q_proj is present but which
    /// is missing a MID-SLICE weight must be caught by the pre-flight, BEFORE
    /// the unified path engages. The old probe looked only at layer 0's q_proj,
    /// declared itself ready, and then panicked indexing
    /// `self.gpu_weights[&ln("mlp.down_proj.weight")]` in the middle of an open
    /// command buffer at layer 31.
    #[test]
    fn mid_slice_missing_weight_is_caught_before_the_path_engages() {
        let cfg = Gemma4Config::g12b();
        let specs = gemma_specs(&cfg, 0..cfg.num_hidden_layers);
        let mut present = complete_checkpoint(&specs);
        let hole = "model.layers.31.mlp.down_proj.weight".to_string();
        assert!(present.remove(&hole));

        // The OLD readiness rule would have said "go".
        assert!(present.contains("model.layers.0.self_attn.q_proj.weight"),
                "precondition: the layer-0 probe still passes on this checkpoint");

        assert_eq!(first_missing_unified_weight(&specs, |k| present.contains(k)), Some(hole));
    }

    /// Same for qwen, and for a hole in the LAST layer (the scan must not stop
    /// early or check only a prefix of the model).
    #[test]
    fn qwen_missing_weight_in_the_last_layer_is_caught() {
        let cfg = qwen_cfg(8);
        let specs = qwen_specs(&cfg, 0..cfg.num_hidden_layers);
        let mut present = complete_checkpoint(&specs);
        let hole = "model.layers.7.self_attn.o_proj.weight".to_string();
        assert!(present.remove(&hole));
        assert!(present.contains("model.layers.0.self_attn.q_proj.weight"));
        assert_eq!(first_missing_unified_weight(&specs, |k| present.contains(k)), Some(hole));
    }

    /// The executed RANGE is load-bearing. A PP rank owning `[12,24)` holds only
    /// those layers' weights: scanning its own slice must pass, while scanning
    /// the whole model would refuse it — which is why the gemma call site passes
    /// `pp_start..pp_end` and not `0..num_hidden_layers`.
    #[test]
    fn pp_rank_validates_only_the_layers_it_executes() {
        let cfg = Gemma4Config::g12b();
        let owned = gemma_specs(&cfg, 12..24);
        let present = complete_checkpoint(&owned);
        assert_eq!(first_missing_unified_weight(&owned, |k| present.contains(k)), None);

        let whole_model = gemma_specs(&cfg, 0..cfg.num_hidden_layers);
        assert!(first_missing_unified_weight(&whole_model, |k| present.contains(k)).is_some(),
                "scanning unowned layers on a PP rank must not be what the guard does");
    }

    /// A complete checkpoint is not refused (the guard adds no false negatives).
    #[test]
    fn complete_checkpoints_scan_clean() {
        for specs in [
            gemma_specs(&Gemma4Config::g12b(), 0..48),
            gemma_specs(&Gemma4Config::e2b(), 0..35),
            qwen_specs(&qwen_cfg(6), 0..6),
        ] {
            let present = complete_checkpoint(&specs);
            assert_eq!(first_missing_unified_weight(&specs, |k| present.contains(k)), None);
        }
    }
}
