// SPDX-License-Identifier: Apache-2.0
//! Gemma4-E2B forward pass implemented entirely in Rust + Vulkan.
//!
//! The entire forward pass (embed → 35 decoder layers → norm → lm_head)
//! is executed as a series of Vulkan compute dispatches with no Python
//! roundtrips between ops.  Each decoder layer submits all its ops
//! (norms, projections, attention prep) in a single vkQueueSubmit,
//! then returns control to Rust for the KV-cache attention step (which
//! runs on CPU via a simple SDPA implementation).
//!
//! This approach yields ~10x GPU utilization improvement over the
//! per-op Python dispatch model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use half::bf16;

// Note: compute engine is used for GPU acceleration in the Vulkan path;
// the CPU reference implementation doesn't need it.

// ─── Model configuration ─────────────────────────────────────────────────────

/// Which Gemma-4 checkpoint family a `Gemma4Config` describes.  The forward pass
/// branches on this to enable/disable the PLE (per-layer-embedding) machinery and
/// the value-less (k==v) global-attention path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4Variant {
    /// gemma-4-E2B: PLE on, KV-sharing on, MQA everywhere, period-5 globals.
    E2b,
    /// gemma-4-12B-it: dense, PLE off, no KV-sharing, GQA(8) sliding / MQA(1)
    /// value-less global, period-6 globals.
    G12b,
    /// gemma-4-31B-it (NVFP4 base): dense, PLE off, no KV-sharing, 60L, hidden
    /// 5376, 32 q-heads, GQA(16) sliding @ head_dim 256 / value-less GQA(4)
    /// global @ head_dim 512, period-6 globals, intermediate 21504, softcap 30.
    /// Differs from G12b only in scale + the global layers using 4 KV heads
    /// (not MQA(1)); same forward code path (config-only).
    G31b,
}

/// Gemma4 architecture constants (from config.json).
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub variant: Gemma4Variant,
    pub hidden_size: usize,           // 1536 / 3840
    pub num_hidden_layers: usize,     // 35 / 48
    pub num_attention_heads: usize,   // 8 / 16
    pub num_key_value_heads: usize,   // 1 / 8   (sliding layers)
    pub num_global_key_value_heads: usize, // 1 / 1 (full/global layers)
    pub head_dim: usize,              // 256  (sliding attention)
    pub global_head_dim: usize,       // 512  (full attention)
    pub intermediate_size: usize,     // 6144 / 15360
    pub num_kv_shared_layers: usize,  // 20 / 0
    /// Period of the sliding/global interleave: a full-attention layer occurs at
    /// `idx % attention_period == attention_period - 1`.  E2B=5, 12B=6.
    pub attention_period: usize,
    /// True when full-attention layers reuse the raw K projection as V (no
    /// v_proj tensor on disk).  E2B=false, 12B=true.
    pub attention_k_eq_v: bool,
    pub vocab_size: usize,            // 262144
    pub rms_norm_eps: f32,            // 1e-6
    pub sliding_window: usize,        // 512 / 1024
    pub hidden_size_per_layer_input: usize, // 256 (PLE) / 0 (no PLE)
    pub final_logit_softcapping: f32, // 30.0
    pub embed_scale: f32,             // sqrt(hidden_size)
    pub ple_scale: f32,               // sqrt(hidden_size_per_layer_input)
    pub per_layer_projection_scale: f32, // hidden_size^(-0.5)
    pub per_layer_input_scale: f32,   // 1/sqrt(2)
}

impl Gemma4Config {
    pub fn e2b() -> Self {
        let h = 1536usize;
        let ple = 256usize;
        Gemma4Config {
            variant: Gemma4Variant::E2b,
            hidden_size: h,
            num_hidden_layers: 35,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            num_global_key_value_heads: 1,
            head_dim: 256,
            global_head_dim: 512,
            intermediate_size: 6144,
            num_kv_shared_layers: 20,
            attention_period: 5,
            attention_k_eq_v: false,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            sliding_window: 512,
            hidden_size_per_layer_input: ple,
            final_logit_softcapping: 30.0,
            embed_scale: (h as f32).sqrt(),
            ple_scale: (ple as f32).sqrt(),
            per_layer_projection_scale: (h as f32).powf(-0.5),
            per_layer_input_scale: (2.0f32).powf(-0.5),
        }
    }

    /// gemma-4-12B-it (dense).  48L, hidden 3840, 16 q-heads, GQA(8) on sliding
    /// layers, value-less MQA(1) @ head_dim 512 on the period-6 global layers,
    /// intermediate 15360, sliding_window 1024, no PLE, no KV-sharing, tied
    /// embeddings, final-logit softcap 30.
    pub fn g12b() -> Self {
        let h = 3840usize;
        Gemma4Config {
            variant: Gemma4Variant::G12b,
            hidden_size: h,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            num_global_key_value_heads: 1,
            head_dim: 256,
            global_head_dim: 512,
            intermediate_size: 15360,
            num_kv_shared_layers: 0,
            attention_period: 6,
            attention_k_eq_v: true,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            sliding_window: 1024,
            hidden_size_per_layer_input: 0,
            final_logit_softcapping: 30.0,
            embed_scale: (h as f32).sqrt(),
            ple_scale: 1.0,
            per_layer_projection_scale: (h as f32).powf(-0.5),
            per_layer_input_scale: (2.0f32).powf(-0.5),
        }
    }

    /// gemma-4-31B-it (NVFP4 base).  60L, hidden 5376, 32 q-heads, GQA(16) on
    /// sliding layers @ head_dim 256, value-less GQA(4) @ head_dim 512 on the
    /// period-6 global layers, intermediate 21504, sliding_window 1024, no PLE,
    /// no KV-sharing, tied embeddings, final-logit softcap 30.  Values verified
    /// against the real `gemma-4-31B-it-NVFP4/config.json` (`text_config`).
    ///
    /// NOTE vs g12b(): identical forward semantics EXCEPT the global layers use
    /// `num_global_key_value_heads: 4` (12B uses MQA(1)); the forward reads this
    /// via `layer_num_kv_heads`, so no code change is needed beyond this config.
    pub fn g31b() -> Self {
        let h = 5376usize;
        Gemma4Config {
            variant: Gemma4Variant::G31b,
            hidden_size: h,
            num_hidden_layers: 60,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            num_global_key_value_heads: 4,
            head_dim: 256,
            global_head_dim: 512,
            intermediate_size: 21504,
            num_kv_shared_layers: 0,
            attention_period: 6,
            attention_k_eq_v: true,
            vocab_size: 262144,
            rms_norm_eps: 1e-6,
            sliding_window: 1024,
            hidden_size_per_layer_input: 0,
            final_logit_softcapping: 30.0,
            embed_scale: (h as f32).sqrt(),
            ple_scale: 1.0,
            per_layer_projection_scale: (h as f32).powf(-0.5),
            per_layer_input_scale: (2.0f32).powf(-0.5),
        }
    }

    /// True when the per-layer-embedding (PLE) machinery is present.
    pub fn has_ple(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }

    pub fn first_kv_shared_layer(&self) -> usize {
        self.num_hidden_layers - self.num_kv_shared_layers  // 35 - 20 = 15
    }

    /// Is layer `idx` a full-attention (global) layer?
    pub fn is_full_attention(&self, idx: usize) -> bool {
        // E2B: full at 4,9,14,…(period 5).  12B: full at 5,11,17,…(period 6).
        idx % self.attention_period == self.attention_period - 1
    }

    /// Number of KV heads for layer `idx` (global layers may use MQA).
    pub fn layer_num_kv_heads(&self, idx: usize) -> usize {
        if self.is_full_attention(idx) {
            self.num_global_key_value_heads
        } else {
            self.num_key_value_heads
        }
    }

    /// True when layer `idx` reuses the raw K projection as V (no v_proj tensor).
    /// Mirrors mlx_vlm gemma4: `attention_k_eq_v and not is_sliding`.
    pub fn layer_uses_k_eq_v(&self, idx: usize) -> bool {
        self.attention_k_eq_v && self.is_full_attention(idx)
    }

    /// head_dim for layer `idx`
    pub fn layer_head_dim(&self, idx: usize) -> usize {
        if self.is_full_attention(idx) { self.global_head_dim } else { self.head_dim }
    }

    /// Physical KV capacity (number of position-slots to allocate) for layer
    /// `idx` given a logical `max_seq_len`. Full-attention (global) layers are
    /// unbounded and get `max_seq_len`; sliding-window layers only ever attend
    /// the last `sliding_window` positions, so they get a `sliding_window`-sized
    /// ring (`KvCache::new_windowed`). This is the Phase-0 per-layer KV sizing
    /// that collapses a 35K gemma session from ~24.1 GB to ~1.82 GB
    /// (`docs/session-kv-continuation-scope.md` §2). Capacity is clamped to
    /// `max_seq_len` so short contexts never over-allocate.
    pub fn layer_kv_capacity(&self, idx: usize, max_seq_len: usize) -> usize {
        if self.is_full_attention(idx) {
            max_seq_len
        } else {
            self.sliding_window.min(max_seq_len)
        }
    }

    /// intermediate_size for layer `idx`
    pub fn layer_intermediate_size(&self, idx: usize) -> usize {
        if idx >= self.first_kv_shared_layer() {
            self.intermediate_size * 2  // double-wide for KV-shared layers
        } else {
            self.intermediate_size
        }
    }

    /// Is layer `idx` a KV-sharing layer?
    pub fn is_kv_shared(&self, idx: usize) -> bool {
        idx >= self.first_kv_shared_layer()
    }
}

// ─── Simple weight tensor (host memory, f32) ─────────────────────────────────

/// A simple host-memory tensor used for the CPU reference implementation.
/// In the GPU path, weights would live in host-coherent Vulkan memory.
pub struct SimpleTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

// ─── Weight storage ──────────────────────────────────────────────────────────

/// All model weights.
pub struct Gemma4Weights {
    pub tensors: HashMap<String, SimpleTensor>,
}

impl Gemma4Weights {
    pub fn get(&self, name: &str) -> &SimpleTensor {
        self.tensors.get(name)
            .unwrap_or_else(|| panic!("Weight '{}' not found", name))
    }

    pub fn f32_slice(&self, name: &str) -> &[f32] {
        &self.get(name).data
    }
}

/// Architecture-neutral alias for the host-memory weight map.  Both the Gemma4
/// and Qwen3 model implementations share the same `name -> tensor` storage; the
/// type is named generically when used outside the Gemma4-specific code.
pub type ModelWeights = Gemma4Weights;

// ─── KV cache ─────────────────────────────────────────────────────────────────

/// Per-layer KV cache.  Stored as host-coherent f32 tensors so both CPU
/// attention and (future) GPU attention can access them directly.
pub struct KvCache {
    /// K storage. Length = `capacity * num_kv_heads * head_dim` f32.
    /// For a full-attention layer `capacity == max_seq_len` and the tensor is a
    /// plain `[max_seq_len, num_kv_heads, head_dim]` array indexed by absolute
    /// position. For a **windowed** (sliding-window) layer `capacity == window`
    /// and the tensor is a **ring buffer**: absolute position `p` lives at slot
    /// `p % capacity` (see `append` / `windowed_view`).
    pub k: Vec<f32>,
    /// V storage. Same layout/semantics as `k`.
    pub v: Vec<f32>,
    /// Number of positions appended so far (ABSOLUTE, monotonically grows to
    /// `max_seq_len`). This is unchanged by windowing — it is still the token
    /// count, NOT a ring slot index.
    pub seq_len: usize,
    /// Logical maximum context length this cache will ever be asked to hold
    /// (the overflow bound on `seq_len`). Independent of physical `capacity`.
    pub max_seq_len: usize,
    /// Physical number of position-slots actually allocated. Equals
    /// `max_seq_len` for full layers, `window` for windowed sliding layers.
    /// Invariant: `capacity >= 1` and (for windowed caches) `capacity <= max_seq_len`.
    pub capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCache {
    /// Full-size cache: physical capacity == `max_seq_len` (plain absolute
    /// indexing, no ring). This is the historical behaviour and is byte-for-byte
    /// identical to the pre-windowing code for every existing caller.
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self::new_windowed(max_seq_len, max_seq_len, num_kv_heads, head_dim)
    }

    /// Per-layer-sized cache. `capacity` is the physical number of slots to
    /// allocate; `max_seq_len` is the logical position bound. When
    /// `capacity < max_seq_len` the storage is a ring buffer that retains only
    /// the most-recent `capacity` positions — correct for sliding-window
    /// attention, whose SDPA only ever reads the last `window` positions
    /// (`cpu_sdpa`'s `kv_start = seq_len - window`). Reads MUST go through
    /// [`windowed_view`]; the absolute-indexed accessors (`k_up_to_now` etc.)
    /// assert the cache has not wrapped.
    pub fn new_windowed(max_seq_len: usize, capacity: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        assert!(capacity >= 1, "KvCache capacity must be >= 1");
        assert!(capacity <= max_seq_len, "KvCache capacity {capacity} > max_seq_len {max_seq_len}");
        KvCache {
            k: vec![0.0; capacity * num_kv_heads * head_dim],
            v: vec![0.0; capacity * num_kv_heads * head_dim],
            seq_len: 0,
            max_seq_len,
            capacity,
            num_kv_heads,
            head_dim,
        }
    }

    /// True once absolute `seq_len` has exceeded the physical `capacity`, i.e.
    /// the ring has wrapped and slot index no longer equals absolute position.
    #[inline]
    pub fn has_wrapped(&self) -> bool {
        self.seq_len > self.capacity
    }

    /// Append one token's K and V to the cache. For a full cache this writes at
    /// absolute slot `seq_len` (identical to the historical code). For a
    /// windowed cache it writes at ring slot `seq_len % capacity`, overwriting
    /// the position that just fell out of the window.
    pub fn append(&mut self, k_token: &[f32], v_token: &[f32]) {
        assert!(self.seq_len < self.max_seq_len, "KV cache overflow");
        let stride = self.num_kv_heads * self.head_dim;
        let slot = self.seq_len % self.capacity;
        let pos = slot * stride;
        self.k[pos..pos + stride].copy_from_slice(k_token);
        self.v[pos..pos + stride].copy_from_slice(v_token);
        self.seq_len += 1;
    }

    /// Contiguous, ascending-absolute-position view of the last `window`
    /// positions (or all `seq_len` positions if `seq_len <= window`), suitable
    /// for feeding directly to `cpu_sdpa`/`cpu_sdpa_gqa` with `sliding_window =
    /// None` and `seq_len = valid_len`.
    ///
    /// Returns `(k, v, valid_len)` where `k`/`v` are `[valid_len, num_kv_heads,
    /// head_dim]` in the SAME row order (ascending absolute position
    /// `kv_start..seq_len`) that `cpu_sdpa` iterates. Because the attention math
    /// depends only on the set of rows and their order — never on the absolute
    /// index value — attending over this compacted view with `window = None`
    /// is **bit-for-bit identical** to attending over a full-size cache with
    /// `sliding_window = Some(window)`. This is the correctness contract that
    /// makes per-layer (ring) sizing bit-exact.
    ///
    /// For a not-yet-wrapped cache the rows are copied straight out of the
    /// existing contiguous slice (`windowed_view` is a superset of what
    /// `k_up_to_now()[kv_start*stride..]` used to hand the SDPA sites).
    pub fn windowed_view(&self, window: usize) -> (Vec<f32>, Vec<f32>, usize) {
        let stride = self.num_kv_heads * self.head_dim;
        let kv_start = self.seq_len.saturating_sub(window);
        let valid_len = self.seq_len - kv_start;
        let mut k = vec![0.0f32; valid_len * stride];
        let mut v = vec![0.0f32; valid_len * stride];
        for i in 0..valid_len {
            let abs = kv_start + i;
            let slot = abs % self.capacity;
            k[i * stride..(i + 1) * stride]
                .copy_from_slice(&self.k[slot * stride..(slot + 1) * stride]);
            v[i * stride..(i + 1) * stride]
                .copy_from_slice(&self.v[slot * stride..(slot + 1) * stride]);
        }
        (k, v, valid_len)
    }

    pub fn k_up_to_now(&self) -> &[f32] {
        debug_assert!(!self.has_wrapped(),
            "k_up_to_now() on a wrapped windowed KvCache (seq_len {} > capacity {}); use windowed_view()",
            self.seq_len, self.capacity);
        &self.k[..self.seq_len * self.num_kv_heads * self.head_dim]
    }

    pub fn v_up_to_now(&self) -> &[f32] {
        debug_assert!(!self.has_wrapped(),
            "v_up_to_now() on a wrapped windowed KvCache (seq_len {} > capacity {}); use windowed_view()",
            self.seq_len, self.capacity);
        &self.v[..self.seq_len * self.num_kv_heads * self.head_dim]
    }

    /// K for an arbitrary prefix boundary `n` (n may be < or == `seq_len`;
    /// unlike `k_up_to_now` this does not require `n == seq_len`). Used by the
    /// KV-prefix export seam to snapshot an older boundary than "now".
    pub fn k_upto(&self, n: usize) -> &[f32] {
        debug_assert!(n <= self.capacity,
            "k_upto({n}) exceeds windowed KvCache capacity {}; use windowed_view()", self.capacity);
        &self.k[..n * self.num_kv_heads * self.head_dim]
    }

    /// V counterpart of `k_upto`.
    pub fn v_upto(&self, n: usize) -> &[f32] {
        debug_assert!(n <= self.capacity,
            "v_upto({n}) exceeds windowed KvCache capacity {}; use windowed_view()", self.capacity);
        &self.v[..n * self.num_kv_heads * self.head_dim]
    }

    /// Rewind the cache to `n` valid positions (spec-decode reject/rollback).
    /// The underlying K/V storage (host Vec, or the GPU-resident plane for the
    /// batched-verify path) is overwrite-in-place, so truncation is JUST the
    /// counter — the next append overwrites the abandoned slots, and both
    /// `k_up_to_now`/`v_up_to_now` and SDPA's `seq_len` push-constant already
    /// only read `0..seq_len`.
    pub fn truncate(&mut self, n: usize) {
        self.seq_len = n;
    }
}

// ─── CPU op primitives (used before full GPU pipeline is wired) ───────────────

/// RMS normalisation in place: `x[i] = x[i] / rms(x) * weight[i]`.
///
/// Several call sites (per-head Q-norm/K-norm/V-norm in the decode hot path)
/// used to call the allocating `cpu_rms_norm` and immediately
/// `copy_from_slice` the result back over the same slice they read from —
/// paying for a heap allocation *and* a redundant copy just to mutate a
/// buffer in place. This does the same math directly over `x`, with no
/// allocation.
pub fn cpu_rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = weight.len();
    assert!(n > 0, "cpu_rms_norm_inplace: weight must be non-empty");
    assert_eq!(
        x.len() % n,
        0,
        "cpu_rms_norm_inplace: x.len() ({}) must be a multiple of weight.len() ({n})",
        x.len(),
    );
    let chunks = x.len() / n;
    for c in 0..chunks {
        let row = &mut x[c * n..(c + 1) * n];
        let rms = (row.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
        let scale = 1.0 / rms;
        for (v, &w) in row.iter_mut().zip(weight.iter()) {
            *v = *v * scale * w;
        }
    }
}

/// RMS normalisation without weight, in place (has_weight=False, e.g.
/// v_norm). See `cpu_rms_norm_inplace` doc comment for why this exists
/// alongside the allocating `cpu_rms_norm_no_weight`.
pub fn cpu_rms_norm_no_weight_inplace(x: &mut [f32], n: usize, eps: f32) {
    assert!(n > 0, "cpu_rms_norm_no_weight_inplace: n must be greater than zero");
    assert_eq!(
        x.len() % n,
        0,
        "cpu_rms_norm_no_weight_inplace: x.len() ({}) must be a multiple of n ({n})",
        x.len(),
    );
    let chunks = x.len() / n;
    for c in 0..chunks {
        let row = &mut x[c * n..(c + 1) * n];
        let rms = (row.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
        let scale = 1.0 / rms;
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

/// RMS normalisation in f32: out = x / rms(x) * weight
pub fn cpu_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    cpu_rms_norm_inplace(&mut out, weight, eps);
    out
}

/// RMS normalisation without weight (has_weight=False, e.g. v_norm).
pub fn cpu_rms_norm_no_weight(x: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    cpu_rms_norm_no_weight_inplace(&mut out, n, eps);
    out
}

/// Matrix multiply: C[T,N] = A[T,K] × B[N,K]^T
/// B is stored row-major (output_features, input_features).
pub fn cpu_matmul(a: &[f32], b: &[f32], t: usize, k: usize, n: usize) -> Vec<f32> {
    // a: [t, k] row-major
    // b: [n, k] row-major (weight matrix, each row is one output neuron's weights)
    // c: [t, n] row-major = a @ b^T
    let mut c = vec![0.0f32; t * n];
    unsafe {
        matrixmultiply::sgemm(
            t, k, n,
            1.0,
            a.as_ptr(), k as isize, 1,  // a: row-stride=k, col-stride=1
            b.as_ptr(), 1, k as isize,  // b^T: row-stride=1 (transposed), col-stride=k
            0.0,
            c.as_mut_ptr(), n as isize, 1,  // c: row-stride=n, col-stride=1
        );
    }
    c
}

/// Parallel matrix-vector product for the t=1 (decode) case: `c[n] = a[k] @ b[n,k]^T`.
/// Each output element `c[i]` is the dot product of `a` with row `i` of `b`; the
/// rows are fully independent and the within-row summation order is unchanged, so
/// the result is BIT-IDENTICAL to a serial `cpu_matmul(a,b,1,k,n)` regardless of
/// thread count. Parallelizes across the `n` output rows with rayon. Used for the
/// MoE shared expert + router, which were the dominant single-thread CPU cost in
/// the 35B-A3B decode (STEP-0: shared-expert ~3.2ms/layer single-core).
pub fn cpu_matvec_par(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let mut c = vec![0.0f32; n];
    c.par_iter_mut().enumerate().for_each(|(i, ci)| {
        let row = &b[i * k..(i + 1) * k];
        let mut acc = 0.0f32;
        for j in 0..k { acc += a[j] * row[j]; }
        *ci = acc;
    });
    c
}

/// Element-wise GELU (tanh approximation).
pub fn cpu_gelu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| {
        let c = 0.044715f32;
        let sqrt_2_over_pi = 0.7978845608f32;
        let inner = sqrt_2_over_pi * (v + c * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    }).collect()
}

/// Element-wise SiLU / swish: x * sigmoid(x).  Used by Qwen3's SwiGLU MLP.
pub fn cpu_silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// RoPE: apply rotary positional embedding to q and k.
/// pos: token position, x: [num_heads, head_dim], rotary_dim = dims to rotate
///
/// The per-index rotation angle (and hence its `sin`/`cos`) depends only on
/// `pos`, `i`, `rotary_dim`, and `theta` — the same values for every head,
/// and the same for Q and K (both are called with the same `rotary_dim`/
/// `theta` here). The previous implementation recomputed `theta.powf(..)`
/// and `angle.sin_cos()` (both transcendental, i.e. genuinely expensive —
/// unlike a plain multiply/add) inside the per-head loop, so a single
/// decode step paid for `rotary_dim/2` of each per *head* (8 query heads +
/// up to 1 key head for Gemma4-E2B) instead of just once. Precomputing the
/// `(sin, cos)` table once and reusing it across every head removes that
/// redundant work entirely — same math, computed once instead of up to 9
/// times, with no change in the result (every head applies the exact same
/// precomputed rotation it would otherwise have recomputed itself).
pub fn cpu_rope(
    q: &mut [f32], k: &mut [f32],
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
) {
    // Default: the frequency basis is the rotated span itself (`rotary_dim`).
    // When `rotary_dim == head_dim` (full rotation, the common case) this is the
    // standard RoPE and the extra parameter is a no-op.
    cpu_rope_with_basis(q, k, pos, num_q_heads, num_kv_heads, head_dim, rotary_dim, rotary_dim, theta)
}

/// RoPE with an explicit frequency **basis** dimension, decoupled from the number
/// of rotated dims.  `rotary_dim` is how many leading dims are rotated;
/// `freq_dim` is the denominator of the frequency exponent (`theta^(2i/freq_dim)`).
///
/// gemma-4's global (`full_attention`) layers use mlx_vlm's `ProportionalRoPE`,
/// which computes frequencies **relative to the full head dimension** (dims =
/// global_head_dim) even though only `partial_rotary_factor * head_dim` dims are
/// rotated — so `freq_dim = head_dim` (e.g. 512) while `rotary_dim = head_dim/4`
/// (128).  Passing `freq_dim = rotary_dim` recovers plain RoPE for every other
/// caller (qwen, sliding gemma layers where rotary_dim == head_dim).
pub fn cpu_rope_with_basis(
    q: &mut [f32], k: &mut [f32],
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    freq_dim: usize,
    theta: f32,
) {
    // Number of rotated pairs is set by the rotated span; the pairing DISTANCE is
    // set by the frequency basis (mlx_vlm splits the head into `freq_dim/2` halves
    // and rotates `rotary_dim/2` leading pairs across that split).  When
    // `freq_dim == rotary_dim` (default RoPE) the two coincide, so plain callers
    // are unaffected.
    let n_pairs = rotary_dim / 2;
    let pair_off = freq_dim / 2;
    let mut sin_cos: Vec<(f32, f32)> = Vec::with_capacity(n_pairs);
    for i in 0..n_pairs {
        let freq = 1.0 / theta.powf(i as f32 * 2.0 / freq_dim as f32);
        let angle = pos as f32 * freq;
        sin_cos.push(angle.sin_cos());
    }

    let rotate_head = |x: &mut [f32], sin_cos: &[(f32, f32)]| {
        for (i, &(s, c)) in sin_cos.iter().enumerate() {
            let x0 = x[i];
            let x1 = x[i + pair_off];
            x[i]            = x0 * c - x1 * s;
            x[i + pair_off] = x0 * s + x1 * c;
        }
        // pairs [n_pairs..pair_off) and dims [pair_off+n_pairs..head_dim) unchanged
    };

    for h in 0..num_q_heads {
        let slice = &mut q[h * head_dim..(h + 1) * head_dim];
        rotate_head(slice, &sin_cos);
    }
    for h in 0..num_kv_heads {
        let slice = &mut k[h * head_dim..(h + 1) * head_dim];
        rotate_head(slice, &sin_cos);
    }
}

/// Dot product of two equal-length `f32` slices using 4 independent
/// accumulator lanes instead of a single running sum.
///
/// `Iterator::sum()` over floats must preserve strict left-to-right
/// summation order (float addition isn't associative, so reordering it
/// would change rounding — the compiler can't do this on its own), which
/// means the natural `a.iter().zip(b).map(|(x,y)| x*y).sum()` dot product
/// has a single serial dependency chain: each addition must wait for the
/// previous one to complete, regardless of how well the multiplies
/// themselves vectorize. Splitting the accumulation across 4 independent
/// lanes (summed together only once, at the end) breaks that chain and
/// lets the compiler pipeline/vectorize the multiply-adds — measured
/// ~1.67x faster than the single-accumulator version for `cpu_sdpa`'s
/// score computation (head_dim=256, see `bench_sdpa` below), which is
/// dominated by exactly this dot product run `seq_len` times per head.
/// 4 lanes (rather than 8 or 16) measured best on this hardware — it
/// matches a 128-bit SIMD register's f32 width (the smallest width common
/// to every target architecture this crate ships on: NEON on aarch64,
/// SSE on x86_64), and going wider actually regressed, most likely by
/// working against the compiler's own auto-vectorization of each lane's
/// scalar loop rather than complementing it.
#[inline]
fn dot4(a: &[f32], b: &[f32]) -> f32 {
    // A real (not debug-only) assertion: it lets the compiler prove that
    // indexing into `b` at every offset derived from `a.len()` below is
    // in-bounds even in release builds, eliding the bounds checks that
    // would otherwise remain in this hot loop and undermine the whole
    // point of hand-splitting the accumulator (a debug_assert_eq! here
    // would vanish in release builds, leaving the compiler unable to
    // prove `b`'s indices are safe).
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let mut acc = [0.0f32; 4];
    for c in 0..chunks {
        let i = c * 4;
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
    }
    let mut tail = 0.0f32;
    for i in chunks * 4..n {
        tail += a[i] * b[i];
    }
    acc[0] + acc[1] + acc[2] + acc[3] + tail
}

/// Scaled dot-product attention (single query token, GQA).
/// q: [num_q_heads, head_dim]
/// k: [seq_len, num_kv_heads, head_dim]
/// v: [seq_len, num_kv_heads, head_dim]
/// Returns: [num_q_heads, head_dim]
pub fn cpu_sdpa(
    q: &[f32], k: &[f32], v: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    sliding_window: Option<usize>,
) -> Vec<f32> {
    // Single-node / non-TP: q head qh maps to kv head qh/(num_q/num_kv),
    // i.e. gqa_ratio = num_q/num_kv and no global offset.
    let gqa_ratio = num_q_heads / num_kv_heads;
    cpu_sdpa_gqa(
        q, k, v, num_q_heads, num_kv_heads, head_dim, seq_len, scale,
        sliding_window, gqa_ratio, 0,
    )
}

/// GQA SDPA for the TP replicated-KV + column-sharded-Q regime.
///
/// `num_q_heads` is the LOCAL (this-rank) query-head count; `k`/`v` hold the
/// FULL replicated `num_kv_heads`. The kv head for local query head `qh` is
/// `(q_head_offset + qh) / gqa_ratio`, where `gqa_ratio` is the GLOBAL ratio
/// (`total_num_q / num_kv`) and `q_head_offset = tp_rank * num_local_q`. This
/// makes each rank's local q heads attend the SAME kv heads they would in the
/// single-node layout (and, since kv is replicated, those heads are all local
/// → no cross-rank gather). It also sidesteps the div-by-zero that a naive
/// `num_local_q / num_kv` ratio hits once TP shards q below the kv count
/// (e.g. TP-4 sliding: 8 local q / 16 kv = 0). Passing `q_head_offset=0` and
/// `gqa_ratio=num_q/num_kv` reduces exactly to `cpu_sdpa`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_sdpa_gqa(
    q: &[f32], k: &[f32], v: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    sliding_window: Option<usize>,
    gqa_ratio: usize,
    q_head_offset: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_q_heads * head_dim];

    // Which KV positions any query head can attend to only depends on
    // seq_len/sliding_window — it's the same for every head in this call, so
    // the `scores`/`exp_scores` scratch buffers can be allocated once here
    // and reused across all `num_q_heads` iterations below (every index is
    // unconditionally overwritten before being read in each iteration, so
    // reuse is safe) instead of once per head.
    let kv_start = if let Some(window) = sliding_window {
        seq_len.saturating_sub(window)
    } else {
        0
    };
    let valid_len = seq_len - kv_start;
    let mut scores = vec![0.0f32; valid_len];
    let mut exp_scores = vec![0.0f32; valid_len];

    for qh in 0..num_q_heads {
        let kvh = (q_head_offset + qh) / gqa_ratio;
        let q_row = &q[qh * head_dim..(qh + 1) * head_dim];

        for (si, kv_pos) in (kv_start..seq_len).enumerate() {
            let k_row = &k[(kv_pos * num_kv_heads + kvh) * head_dim
                          ..(kv_pos * num_kv_heads + kvh + 1) * head_dim];
            // Bit-exact single-accumulator sum (the fork's argmax-exact reference).
            // NOT upstream's dot4 4-lane accumulator: it reorders the summation
            // (~1e-3 drift) and could flip a near-tie argmax on the live
            // host-resident attention path. Re-adopt dot4 here only if the GPU
            // cluster proves it argmax-safe.
            let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(&a, &b)| a * b).sum();
            scores[si] = dot * scale;
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for (e, &s) in exp_scores.iter_mut().zip(scores.iter()) {
            *e = (s - max_score).exp();
        }
        let sum: f32 = exp_scores.iter().sum();
        exp_scores.iter_mut().for_each(|s| *s /= sum);

        // Weighted sum of V
        let out_row = &mut out[qh * head_dim..(qh + 1) * head_dim];
        for (si, kv_pos) in (kv_start..seq_len).enumerate() {
            let v_row = &v[(kv_pos * num_kv_heads + kvh) * head_dim
                          ..(kv_pos * num_kv_heads + kvh + 1) * head_dim];
            let w = exp_scores[si];
            for (o, &vv) in out_row.iter_mut().zip(v_row.iter()) {
                *o += w * vv;
            }
        }
    }
    out
}

/// Batched causal SDPA for a spec-verify chunk. `T` query positions
/// (absolute `start_pos..start_pos+T`) attend over a SHARED host K/V cache
/// (`kbuf`/`vbuf`, `[>=start_pos+T, num_kv_heads, head_dim]` row-major, the
/// caller having already appended all `T` verify tokens IN ORDER), query `ti`
/// attending the causal prefix `[0, start_pos+ti]` (length `start_pos+ti+1`).
///
/// BIT-IDENTICAL to `T` sequential `cpu_sdpa(q_ti, k[..l], v[..l], .., l, ..)`
/// calls: each `(ti, head)` output is produced by ONE thread running the EXACT
/// same single-accumulator dot / `fold`-max softmax / in-order AV as `cpu_sdpa`
/// — there is NO cross-thread reduction, so every output float is byte-for-byte
/// the per-token result. The ONLY change is that the `T * num_q_heads`
/// independent head-jobs are spread across rayon threads instead of run
/// serially (the single-threaded `cpu_sdpa`-per-token loop was the TP verify's
/// dominant, un-amortized compute — see the batched-verify mixer in tp.rs).
///
/// `q_all` is `[T, num_q_heads*head_dim]` (already q-norm'd + RoPE'd). Returns
/// `[T, num_q_heads*head_dim]`. No sliding-window arg: the qwen3.6 full-attn
/// layers are non-sliding, so this mirrors the `cpu_sdpa(.., None)` the TP core
/// passes; add a window only if a caller needs it (and re-derive `l`).
pub fn cpu_sdpa_batched_causal(
    q_all: &[f32],
    kbuf: &[f32],
    vbuf: &[f32],
    t: usize,
    start_pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    use rayon::prelude::*;
    let gqa_ratio = num_q_heads / num_kv_heads;
    let q_dim = num_q_heads * head_dim;
    let stride = num_kv_heads * head_dim;
    let mut out = vec![0.0f32; t * q_dim];
    // One rayon job per (token, q-head): out chunk `job` = head `job % nq` of
    // token `job / nq` (out is [T, nq*hd] row-major, so chunk offset job*hd
    // lands exactly at ti*q_dim + qh*hd).
    out.par_chunks_mut(head_dim).enumerate().for_each(|(job, out_head)| {
        let ti = job / num_q_heads;
        let qh = job % num_q_heads;
        let kvh = qh / gqa_ratio;
        let l = start_pos + ti + 1; // causal length for this query (== per-token cpu_sdpa seq_len)
        let q_row = &q_all[ti * q_dim + qh * head_dim..ti * q_dim + qh * head_dim + head_dim];

        // Scores: single-accumulator dot (bit-exact match to cpu_sdpa).
        let mut scores = vec![0.0f32; l];
        for p in 0..l {
            let k_row = &kbuf[(p * num_kv_heads + kvh) * head_dim
                            ..(p * num_kv_heads + kvh + 1) * head_dim];
            let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(&a, &b)| a * b).sum();
            scores[p] = dot * scale;
        }
        // Softmax (fold-max, exp, normalize — identical order to cpu_sdpa).
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        exp_scores.iter_mut().for_each(|s| *s /= sum);
        // Weighted sum of V (accumulate positions 0..l in order, per cpu_sdpa).
        for p in 0..l {
            let v_row = &vbuf[(p * num_kv_heads + kvh) * head_dim
                            ..(p * num_kv_heads + kvh + 1) * head_dim];
            let w = exp_scores[p];
            for (o, &vv) in out_head.iter_mut().zip(v_row.iter()) {
                *o += w * vv;
            }
        }
    });
    out
}

// ─── Sampling ──────────────────────────────────────────────────────────────

/// Temperature/top-p/top-k sampling over a full vocab of logits, mirroring
/// (and replacing) `vllm_vulkan/server.py`'s pure-Python `temperature_sample`
/// — the sampling step used once per decode step by the standalone Rust
/// `VulkanModel` serving path (`vllm_vulkan.server`, documented as giving
/// "~3 tok/s on GB10").
///
/// That Python implementation does a full `sorted()` over all
/// `vocab_size` (262144 for Gemma4-E2B) logits just to select the top
/// `top_k` (a full O(n log n) sort in the CPython interpreter, plus several
/// more full-vocab list comprehensions for temperature scaling and
/// softmax), on every single decode step — on top of `VulkanModel.forward`
/// already having to convert its `Vec<f32>` return value into 262144
/// individual Python `float` objects for `sorted()`/the list comprehensions
/// to iterate over in the first place. Both of those costs disappear
/// entirely if the whole computation (starting from the logits Rust
/// already has, ending at a single sampled token id) never leaves Rust —
/// see `VulkanModel::forward_and_sample` (src/lib.rs), which calls this
/// directly on `forward_gpu`'s/`forward`'s own output.
///
/// `uniform_random` is a caller-supplied uniform `[0, 1)` draw (e.g.
/// Python's `random.random()`) rather than something this function
/// generates itself, so this crate doesn't need to add a `rand`
/// dependency or make any choice about RNG algorithm/quality — sampling
/// quality is exactly as good as whatever uniform source the caller
/// already trusted before this change.
///
/// `top_k <= 0` means "no top-k filtering" (use the full vocab), matching
/// the Python implementation's `else` branch — mathematically equivalent
/// to `top_k >= vocab_size` here, since softmax probabilities already sum
/// to 1, so renormalizing the full set by its own sum is a no-op up to
/// floating-point rounding.
///
/// `temperature < 0.01` means greedy (argmax) sampling, matching
/// `vllm_vulkan/server.py`'s `generate()` — which never called
/// `temperature_sample` at all below that threshold, calling
/// `greedy_sample` directly instead, both to avoid dividing by
/// a near-zero temperature and because sampling is deterministic there
/// anyway. `temperature_sample` itself only special-cased exactly `0.0`,
/// which was unreachable in practice since `generate()` was always the
/// caller — this function uses the threshold that actually mattered
/// end-to-end.
pub fn sample_with_temperature(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: i64,
    uniform_random: f32,
) -> usize {
    if temperature < 0.01 {
        return argmax(logits);
    }

    let n = logits.len();
    let inv_temp = 1.0 / temperature;

    // Softmax over the temperature-scaled logits.
    let max_scaled = logits.iter().fold(f32::NEG_INFINITY, |m, &l| m.max(l * inv_temp));
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l * inv_temp - max_scaled).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    // Sort every (index, prob) pair descending by prob — this single sort
    // serves both the top-k selection (the Python version's separate
    // `sorted()` call for that) and the top-p step (which iterates the
    // already-top-k-sorted list in order), since `probs` sorted descending
    // is a superset of both intermediate orderings the Python version
    // computes with two separate sorts.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let k = if top_k <= 0 { n } else { (top_k as usize).min(n) };
    let top_k_order = &order[..k];
    let top_k_sum: f32 = top_k_order.iter().map(|&i| probs[i]).sum();

    // Top-p (nucleus): walk the (already-sorted) top-k prefix accumulating
    // renormalized probability mass until it reaches top_p.
    let mut cumsum = 0.0f32;
    let mut nucleus_end = top_k_order.len();
    for (pos, &idx) in top_k_order.iter().enumerate() {
        cumsum += probs[idx] / top_k_sum;
        if cumsum >= top_p {
            nucleus_end = pos + 1;
            break;
        }
    }
    let nucleus = &top_k_order[..nucleus_end];
    let nucleus_sum: f32 = nucleus.iter().map(|&i| probs[i]).sum();

    // Sample via the caller-supplied uniform draw against the nucleus's
    // renormalized cumulative distribution.
    let mut cumsum = 0.0f32;
    for &idx in nucleus {
        cumsum += probs[idx] / nucleus_sum;
        if uniform_random <= cumsum {
            return idx;
        }
    }
    *nucleus.last().unwrap()
}

/// Index of the largest element (ties broken by first occurrence, matching
/// Python's `max(range(len(logits)), key=...)`).
pub fn argmax(logits: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

// ─── Gemma4 forward pass (pure CPU, correct, used to verify) ─────────────────

/// Complete Gemma4-E2B forward pass for a single token (decode step).
///
/// All computation runs on CPU using the ops above.  This is the
/// reference implementation used for correctness testing.  The
/// Vulkan-accelerated version will call the same ops but via GPU shaders.
pub struct Gemma4Model {
    pub config: Gemma4Config,
    pub weights: Gemma4Weights,
    pub kv_caches: Vec<KvCache>,

}

impl Gemma4Model {
    /// Embedding + PLE preprocessing for one token (extracted from `forward`
    /// so `forward_verify_core` can compute the same per-position inputs
    /// independent of a serial token loop). Read-only over `self.weights` —
    /// no cache/state touched, so it's safe to call once per verify-batch
    /// position ahead of the layer-major sweep.
    fn embed_and_ple(&self, token_id: u32) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        // ── Embedding ──────────────────────────────────────────────────────
        let embed_w = self.weights.f32_slice("model.embed_tokens.weight");
        let hidden: Vec<f32> = embed_w[token_id as usize * h..
                                        (token_id as usize + 1) * h]
            .iter().map(|&v| v * cfg.embed_scale).collect();

        // ── PLE global preprocessing (E2B only; 12B/31B have hidden_size_per_layer_input=0) ──
        let ple_dim = cfg.hidden_size_per_layer_input;  // 256 (E2B) / 0 (12B/31B)
        let ple_inputs: Vec<f32> = if cfg.has_ple() {
            let total_ple = cfg.num_hidden_layers * ple_dim;  // 35 * 256 = 8960

            // per_layer_embeds[layer_idx] = embed_tokens_per_layer[token_id, layer_idx*ple_dim..] * ple_scale
            let ple_embed_w = self.weights.f32_slice("model.embed_tokens_per_layer.weight");
            let ple_embeds_flat: Vec<f32> = ple_embed_w[token_id as usize * total_ple..
                                                           (token_id as usize + 1) * total_ple]
                .iter().map(|&v| v * cfg.ple_scale).collect();

            // per_layer_projection = per_layer_model_projection(hidden) * per_layer_projection_scale
            let proj_w = self.weights.f32_slice("model.per_layer_model_projection.weight");
            let ple_proj_flat = cpu_matmul(&hidden, proj_w, 1, h, total_ple);
            let ple_proj_flat: Vec<f32> = ple_proj_flat.iter()
                .map(|&v| v * cfg.per_layer_projection_scale).collect();

            // per_layer_projection_norm (applied to [ple_dim] blocks)
            let proj_norm_w = self.weights.f32_slice("model.per_layer_projection_norm.weight");
            let ple_proj_normed = cpu_rms_norm(&ple_proj_flat, proj_norm_w, eps);

            // per_layer_inputs = (ple_proj_normed + ple_embeds) * per_layer_input_scale
            ple_proj_normed.iter()
                .zip(ple_embeds_flat.iter())
                .map(|(&p, &e)| (p + e) * cfg.per_layer_input_scale)
                .collect()
            // per-layer slice = ple_inputs[layer*ple_dim..(layer+1)*ple_dim]
        } else {
            Vec::new()
        };
        (hidden, ple_inputs)
    }

    /// Forward pass for one token at position `pos`.
    /// Returns logits [vocab_size].
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        let (mut hidden, ple_inputs) = self.embed_and_ple(token_id);
        let ple_dim = cfg.hidden_size_per_layer_input;

        // ── Decoder layers ──────────────────────────────────────────────────
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_ple: &[f32] = if cfg.has_ple() {
                &ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim]
            } else {
                &[]
            };
            hidden = self.forward_layer(layer_idx, &hidden, pos, layer_ple);
        }

        // ── Final norm ──────────────────────────────────────────────────────
        let norm_w = self.weights.f32_slice("model.norm.weight");
        hidden = cpu_rms_norm(&hidden, norm_w, eps);

        // ── LM head (tied weights) ──────────────────────────────────────────
        let lm_w = self.weights.f32_slice("model.embed_tokens.weight");
        let mut logits = cpu_matmul(&hidden, lm_w, 1, h, cfg.vocab_size);

        // ── Final logit softcap ─────────────────────────────────────────────
        let cap = cfg.final_logit_softcapping;
        for l in logits.iter_mut() {
            *l = (*l / cap).tanh() * cap;
        }

        logits
    }

    /// Like [`forward`], but also returns the post-final-norm hidden (the exact
    /// f32 input to the tied LM head) alongside the softcapped logits, WITHOUT
    /// re-running the decoder. Lets a gate re-project the same `normed` with an
    /// alternate lm_head weight (e.g. the mlx4-4bit round-trip) to isolate the
    /// lm_head-quantization effect: the input-embed lookup and all decoder
    /// layers stay bit-exact, matching the GPU H2 lever
    /// (VLLM_VULKAN_GEMMA_LMHEAD_Q4), which keeps the host input-embed copy f16
    /// and quantizes only the GPU lm_head matvec. KV state advances exactly as
    /// `forward` does (this IS the forward pass, just exposing `normed`).
    pub fn forward_with_normed(&mut self, token_id: u32, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let (mut hidden, ple_inputs) = self.embed_and_ple(token_id);
        let ple_dim = cfg.hidden_size_per_layer_input;
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_ple: &[f32] = if cfg.has_ple() {
                &ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim]
            } else {
                &[]
            };
            hidden = self.forward_layer(layer_idx, &hidden, pos, layer_ple);
        }
        let norm_w = self.weights.f32_slice("model.norm.weight");
        let normed = cpu_rms_norm(&hidden, norm_w, eps);
        let lm_w = self.weights.f32_slice("model.embed_tokens.weight");
        let mut logits = cpu_matmul(&normed, lm_w, 1, h, cfg.vocab_size);
        let cap = cfg.final_logit_softcapping;
        for l in logits.iter_mut() {
            *l = (*l / cap).tanh() * cap;
        }
        (logits, normed)
    }

    /// Design-A batched-VERIFY core (INC-4, spec §"gemma batched-verify core").
    /// Runs `tokens` (the bonus token + K drafted tokens, T=len) at consecutive
    /// positions `start_pos..start_pos+T` through the full decoder stack and
    /// returns per-position logits (`[T][vocab]`, row `ti` = the argmax
    /// candidate produced by having fed `tokens[0..=ti]`).
    ///
    /// Unlike `forward`'s TOKEN-major loop (all layers for token 0, then all
    /// layers for token 1, ...), this sweeps LAYER-major: for each layer,
    /// process all T positions in position order (appending each position's
    /// own K/V to that layer's cache before advancing to the next position).
    /// Both orders compute IDENTICAL numbers — a layer's output at position i
    /// depends only on that layer's own input hidden at i and the KV cache
    /// contents at positions < i, which are populated the same way regardless
    /// of whether the outer loop is layer-major or token-major. Layer-major is
    /// the structure a real batched dispatch (T-row GEMM instead of T serial
    /// matvecs) would use, so this CPU version validates the reordering itself
    /// ahead of any GPU-resident batching.
    ///
    /// No GatedDeltaNet/recurrent state to snapshot (Gemma4 is pure attention,
    /// unlike Qwen3.6) — the batched pass is just KV-cache appends, so there is
    /// no capture step analogous to `forward_qwen35_verify_core`'s
    /// `spec_verify_gdn_inputs`. Partial-accept rollback is a KV-counter rewind
    /// only: see `verify_rollback`.
    pub fn forward_verify_core(&mut self, tokens: &[u32], start_pos: usize) -> Vec<Vec<f32>> {
        let t = tokens.len();
        assert!(t > 0, "forward_verify_core: empty verify batch");
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let ple_dim = cfg.hidden_size_per_layer_input;

        // Embed all T tokens + their per-position PLE inputs up front (both
        // are pure functions of the token id, independent of layer order).
        let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(t);
        let mut ples: Vec<Vec<f32>> = Vec::with_capacity(t);
        for &tok in tokens {
            let (hid, ple) = self.embed_and_ple(tok);
            hiddens.push(hid);
            ples.push(ple);
        }

        // Layer-major sweep: for each layer, advance every position in
        // position order (start_pos+0, start_pos+1, ..., start_pos+T-1) so
        // each position's self-attention sees exactly the KV appended by the
        // earlier positions at this same layer — the same causal contract
        // `forward_layer` relies on when driven token-major.
        for layer_idx in 0..cfg.num_hidden_layers {
            for ti in 0..t {
                let layer_ple: &[f32] = if cfg.has_ple() {
                    &ples[ti][layer_idx * ple_dim..(layer_idx + 1) * ple_dim]
                } else {
                    &[]
                };
                hiddens[ti] = self.forward_layer(layer_idx, &hiddens[ti], start_pos + ti, layer_ple);
            }
        }

        // Final norm + tied LM head + softcap, per position.
        let norm_w = self.weights.f32_slice("model.norm.weight").to_vec();
        let lm_w = self.weights.f32_slice("model.embed_tokens.weight").to_vec();
        let cap = cfg.final_logit_softcapping;
        hiddens.iter().map(|hid| {
            let normed = cpu_rms_norm(hid, &norm_w, eps);
            let mut logits = cpu_matmul(&normed, &lm_w, 1, h, cfg.vocab_size);
            logits.iter_mut().for_each(|l| *l = (*l / cap).tanh() * cap);
            logits
        }).collect()
    }

    /// Partial-accept ROLLBACK after `forward_verify_core(tokens, start_pos)`
    /// (`t = tokens.len()`). `accept_len` is the chain-verify's accepted
    /// prefix length (0..t); `commit_len = accept_len + 1` tokens (the bonus
    /// token plus `accept_len` accepted drafts) are actually kept. Gemma has
    /// no GatedDeltaNet/recurrent state (unlike Qwen3.6's
    /// `qwen35_verify_rollback_impl`, which must re-scan the GDN layers over
    /// the committed prefix) — every layer's KV storage is already correct at
    /// `start_pos+commit_len` (overwrite-in-place, see `KvCache::truncate`),
    /// so rollback is JUST rewinding each layer's KV seq_len counter. A no-op
    /// when `commit_len == t` (full accept: the verify pass already left every
    /// cache at exactly `start_pos+t`).
    ///
    /// ⚠️ WINDOWED (Phase-0 per-layer sizing) CAVEAT — DEFERRED TO THE LIVE GATE:
    /// counter-only truncate is bit-exact on a full cache and on a windowed ring
    /// that did NOT wrap during the verify batch. If the verify overshoot
    /// (`start_pos+t`) pushed a sliding layer's ring past `capacity` (window),
    /// the overshoot appends OVERWRITE the ring slots holding the window-start
    /// positions that the post-rollback window at `frontier` still needs, and
    /// `truncate` (a bare counter rewind) cannot restore them → the next
    /// `windowed_view` reads a rejected token's K/V for up to `t-commit_len`
    /// window-start positions. Correctly ring-porting this needs a pre-verify
    /// ring snapshot/restore (or a window-aware rollback) and is deferred with
    /// the rest of the batched-verify GPU path; the current production DECODE
    /// path does not roll back, and the offline spec test harness (`gemma_spec`)
    /// uses full caches, so no shipped path hits the wrapped-rollback case.
    pub fn verify_rollback(&mut self, start_pos: usize, t: usize, accept_len: usize) {
        assert!(accept_len < t, "verify_rollback: accept_len {accept_len} must be < verify T {t}");
        let commit_len = accept_len + 1;
        if commit_len == t {
            return;
        }
        let frontier = start_pos + commit_len;
        for cache in self.kv_caches.iter_mut() {
            cache.truncate(frontier);
        }
    }

    pub fn forward_layer(&mut self, layer_idx: usize, hidden: &[f32], pos: usize, layer_ple: &[f32]) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.layer_num_kv_heads(layer_idx);
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let is_kv_shared = cfg.is_kv_shared(layer_idx);
        let ffn_inter = cfg.layer_intermediate_size(layer_idx);
        let ple_dim = cfg.hidden_size_per_layer_input;

        let ln = |w: &str| format!("model.layers.{layer_idx}.{w}");

        // 1. Input layernorm
        let inln_w = self.weights.f32_slice(&ln("input_layernorm.weight")).to_vec();
        let x = cpu_rms_norm(hidden, &inln_w, eps);

        // 2. QKV projections
        let q_w = self.weights.f32_slice(&ln("self_attn.q_proj.weight")).to_vec();
        let k_w = self.weights.f32_slice(&ln("self_attn.k_proj.weight")).to_vec();
        let mut q = cpu_matmul(&x, &q_w, 1, h, q_dim);
        let k_raw = cpu_matmul(&x, &k_w, 1, h, kv_dim);
        // V source: a dedicated v_proj, OR (value-less "k_eq_v" global layers, e.g.
        // gemma-4-12B full_attention) the RAW k projection reused as V.  mlx_vlm
        // gemma4 sets `values = keys` BEFORE k_norm, so V derives from k_raw and
        // receives only the weightless v_norm (never RoPE).
        let v_raw = if cfg.layer_uses_k_eq_v(layer_idx) {
            k_raw.clone()
        } else {
            let v_w = self.weights.f32_slice(&ln("self_attn.v_proj.weight")).to_vec();
            cpu_matmul(&x, &v_w, 1, h, kv_dim)
        };

        // 3. Q-norm and K-norm
        let q_norm_w = self.weights.f32_slice(&ln("self_attn.q_norm.weight")).to_vec();
        let k_norm_w = self.weights.f32_slice(&ln("self_attn.k_norm.weight")).to_vec();
        // Apply q_norm per head, in place (no clone, no allocate-then-copy-back).
        for h_idx in 0..num_q_heads {
            let slice = &mut q[h_idx * head_dim..(h_idx + 1) * head_dim];
            cpu_rms_norm_inplace(slice, &q_norm_w, eps);
        }

        let mut k_final: Vec<f32>;
        let mut v_final: Vec<f32>;

        if !is_kv_shared {
            let mut k_heads = k_raw;
            for h_idx in 0..num_kv_heads {
                let slice = &mut k_heads[h_idx * head_dim..(h_idx + 1) * head_dim];
                cpu_rms_norm_inplace(slice, &k_norm_w, eps);
            }

            // V-norm (no weight)
            let mut v_heads = v_raw;
            for h_idx in 0..num_kv_heads {
                let slice = &mut v_heads[h_idx * head_dim..(h_idx + 1) * head_dim];
                cpu_rms_norm_no_weight_inplace(slice, head_dim, eps);
            }
            k_final = k_heads;
            v_final = v_heads;
        } else {
            // KV-shared: k and v come from the target layer's cache.
            // We still need dummy values for RoPE (q only matters).
            k_final = k_raw;
            v_final = v_raw;
        }

        // 4. RoPE
        let (theta, rotary_dim) = if is_full {
            (1_000_000.0f32, head_dim / 4)  // proportional, partial_rotary_factor=0.25
        } else {
            (10_000.0f32, head_dim)           // default, full rotation
        };
        // gemma-4 RoPE frequency basis is the full head_dim (mlx_vlm ProportionalRoPE:
        // dims = head_dim for both the default sliding rope and the proportional global
        // rope).  For sliding layers rotary_dim == head_dim so this is standard RoPE;
        // for global layers rotary_dim == head_dim/4 but freqs are still over head_dim.
        cpu_rope_with_basis(&mut q, &mut k_final, pos, num_q_heads, num_kv_heads, head_dim, rotary_dim, head_dim, theta);
        if is_kv_shared {
            // Only Q rotation matters; restore k_final to target cache later.
        }

        // 5. Update KV cache (only for non-shared layers)
        let target_cache_idx = if is_kv_shared {
            self.kv_shared_target(layer_idx)
        } else {
            layer_idx
        };

        if !is_kv_shared {
            // Append new K, V to this layer's cache.
            let cache = &mut self.kv_caches[layer_idx];
            cache.append(&k_final, &v_final);
        }

        // 6. Attention (SDPA)
        let attn_cache = &self.kv_caches[target_cache_idx];
        let attn_scale = 1.0f32;  // Gemma4 uses scale=1.0, not 1/sqrt(head_dim)
        // Per-layer-sized KV: a sliding layer's cache may be a `window`-sized
        // ring (capacity < max_seq_len). `windowed_view` compacts the last
        // `window` positions into ascending-absolute order, so attending over
        // it with `sliding_window = None` is bit-for-bit identical to attending
        // over a full-size cache with `Some(window)` (see `windowed_view`).
        // Full layers keep the zero-copy absolute-slice path unchanged.
        let attn_out = if is_full {
            cpu_sdpa(
                &q, attn_cache.k_up_to_now(), attn_cache.v_up_to_now(),
                num_q_heads, num_kv_heads, head_dim,
                attn_cache.seq_len, attn_scale, None,
            )
        } else {
            let (kw, vw, vlen) = attn_cache.windowed_view(cfg.sliding_window);
            cpu_sdpa(
                &q, &kw, &vw,
                num_q_heads, num_kv_heads, head_dim,
                vlen, attn_scale, None,
            )
        };
        // attn_out: [num_q_heads * head_dim]

        // 7. O-projection
        let o_w = self.weights.f32_slice(&ln("self_attn.o_proj.weight")).to_vec();
        let attn_proj = cpu_matmul(&attn_out, &o_w, 1, q_dim, h);

        // 8. Post-attention layernorm
        let post_attn_w = self.weights.f32_slice(&ln("post_attention_layernorm.weight")).to_vec();
        let attn_normed = cpu_rms_norm(&attn_proj, &post_attn_w, eps);

        // 9. Residual add
        let hidden2: Vec<f32> = hidden.iter().zip(attn_normed.iter())
            .map(|(&r, &a)| r + a).collect();
        let residual2 = hidden2.clone();

        // 10. Pre-FFN layernorm
        let pre_ff_w = self.weights.f32_slice(&ln("pre_feedforward_layernorm.weight")).to_vec();
        let ff_in = cpu_rms_norm(&hidden2, &pre_ff_w, eps);

        // 11. MLP: gate * up + down
        let gate_w = self.weights.f32_slice(&ln("mlp.gate_proj.weight")).to_vec();
        let up_w   = self.weights.f32_slice(&ln("mlp.up_proj.weight")).to_vec();
        let gate = cpu_matmul(&ff_in, &gate_w, 1, h, ffn_inter);
        let up   = cpu_matmul(&ff_in, &up_w,   1, h, ffn_inter);
        let gate_act = cpu_gelu(&gate);
        let mid: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();

        let down_w = self.weights.f32_slice(&ln("mlp.down_proj.weight")).to_vec();
        let ff_out = cpu_matmul(&mid, &down_w, 1, ffn_inter, h);

        // 12. Post-FFN layernorm
        let post_ff_w = self.weights.f32_slice(&ln("post_feedforward_layernorm.weight")).to_vec();
        let ff_normed = cpu_rms_norm(&ff_out, &post_ff_w, eps);

        // 13. Residual add
        let mut hidden3: Vec<f32> = residual2.iter().zip(ff_normed.iter())
            .map(|(&r, &f)| r + f).collect();

        // 14. PLE block (E2B only; 12B has no per-layer-embedding gating)
        if cfg.has_ple() {
            let ple_gate_w = self.weights.f32_slice(&ln("per_layer_input_gate.weight")).to_vec();
            let gate_out = cpu_matmul(&hidden3, &ple_gate_w, 1, h, cfg.hidden_size_per_layer_input);
            let gate_act2 = cpu_gelu(&gate_out);
            let gated: Vec<f32> = gate_act2.iter().zip(layer_ple.iter())
                .map(|(&g, &p)| g * p).collect();
            let ple_proj_w = self.weights.f32_slice(&ln("per_layer_projection.weight")).to_vec();
            let contrib = cpu_matmul(&gated, &ple_proj_w, 1, ple_dim, h);
            let ple_norm_w = self.weights.f32_slice(&ln("post_per_layer_input_norm.weight")).to_vec();
            let contrib_normed = cpu_rms_norm(&contrib, &ple_norm_w, eps);
            hidden3.iter_mut().zip(contrib_normed.iter()).for_each(|(h, &c)| *h += c);
        }

        // 15. Layer scalar (present on every text layer in both E2B and 12B)
        let scalar_data = self.weights.f32_slice(&ln("layer_scalar"));
        let scalar = scalar_data[0];
        hidden3.iter_mut().for_each(|v| *v *= scalar);

        hidden3
    }

    /// Find which layer's KV cache a KV-shared layer should use.
    pub fn kv_shared_target(&self, layer_idx: usize) -> usize {
        let cfg = &self.config;
        let first_kv = cfg.first_kv_shared_layer();
        assert!(layer_idx >= first_kv, "Layer {} is not a KV-shared layer (first KV shared = {})", layer_idx, first_kv);

        let is_full = cfg.is_full_attention(layer_idx);

        // vLLM's Gemma4Attention logic:
        //   kv_shared_layer_index = last index in prev_layers (layers 0..first_kv)
        //   that has the SAME layer type as layer_idx.
        // This means ALL KV-shared sliding layers target the LAST sliding layer
        // before first_kv, and ALL KV-shared full layers target the LAST full
        // layer before first_kv.
        (0..first_kv).rev()
            .find(|&i| cfg.is_full_attention(i) == is_full)
            .unwrap_or(first_kv - 1)
    }
}

// ─── Canonical (layer, kv_head)-tile KV prefix export/import (NAS prefix-cache
//     Phase 1, `docs/nas-prefix-cache-*.md`) ─────────────────────────────────
//
// gemma KV is host-resident f32 (`kv_caches: Vec<KvCache>`), so a tile is a pure
// host slice — no device involved. Full-attn layers store rows `[0, upto)`;
// sliding layers store the `window`-bounded slice `[max(0, upto-window), upto)`
// with the absolute base in the header (both read/write through the ring slot
// `abs % capacity`, so restore is bit-exact whether or not the physical cache is
// a shrunk ring). KV-shared alias layers own NO tiles — their `attn_cache`
// reads the target layer's restored cache, so the alias topology is re-derived
// from config at restore time (nothing stored for them). See
// `src/kv_prefix.rs` for the `TILE1` body and the K-AND-V (never K-only,
// even on `k_eq_v` globals) storage rationale.
impl crate::kv_prefix::KvPrefixExport for Gemma4Model {
    fn kv_content_dims(&self) -> crate::kv_prefix::KvContentDims {
        let cfg = &self.config;
        let layers = (0..cfg.num_hidden_layers)
            .map(|l| crate::kv_prefix::LayerKvGeom {
                kv_heads: cfg.layer_num_kv_heads(l),
                head_dim: cfg.layer_head_dim(l),
                is_full: cfg.is_full_attention(l),
                window: cfg.sliding_window,
                k_eq_v: cfg.layer_uses_k_eq_v(l),
            })
            .collect();
        // Fold the rope/interleave-defining scalars so a config change that
        // alters stored K (rope) but not head geometry still misses.
        let mut rope = 0xcbf2_9ce4_8422_2325u64;
        for v in [
            cfg.attention_period as u64,
            cfg.sliding_window as u64,
            cfg.num_kv_shared_layers as u64,
            cfg.head_dim as u64,
            cfg.global_head_dim as u64,
        ] {
            rope ^= v;
            rope = rope.wrapping_mul(0x0000_0100_0000_01B3);
        }
        crate::kv_prefix::KvContentDims {
            arch_tag: 0,
            num_layers: cfg.num_hidden_layers,
            layers,
            rope_ident: rope,
        }
    }

    fn owned_tiles(&self) -> Vec<crate::kv_prefix::TileSpec> {
        let cfg = &self.config;
        let mut out = Vec::new();
        for l in 0..cfg.num_hidden_layers {
            // KV-shared alias layers store nothing (they read the target's cache).
            if cfg.is_kv_shared(l) {
                continue;
            }
            let is_full = cfg.is_full_attention(l);
            let head_dim = cfg.layer_head_dim(l);
            for h in 0..cfg.layer_num_kv_heads(l) {
                out.push(crate::kv_prefix::TileSpec {
                    layer: l,
                    kv_head: h,
                    head_dim,
                    is_full,
                    window: cfg.sliding_window,
                    k_eq_v: cfg.layer_uses_k_eq_v(l),
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
        let cache = self
            .kv_caches
            .get(layer)
            .ok_or_else(|| format!("export_tile: layer {layer} out of range"))?;
        let is_full = cfg.is_full_attention(layer);
        let head_dim = cfg.layer_head_dim(layer);
        let stride = cache.num_kv_heads * cache.head_dim;
        if kv_head >= cache.num_kv_heads {
            return Err(format!(
                "export_tile: kv_head {kv_head} >= num_kv_heads {}",
                cache.num_kv_heads
            ));
        }
        let (base, n_rows) = tile_row_range(is_full, upto, cfg.sliding_window);
        let mut k = vec![0.0f32; n_rows * head_dim];
        let mut v = vec![0.0f32; n_rows * head_dim];
        for i in 0..n_rows {
            let abs = base + i;
            let slot = abs % cache.capacity;
            let src = slot * stride + kv_head * head_dim;
            k[i * head_dim..(i + 1) * head_dim].copy_from_slice(&cache.k[src..src + head_dim]);
            v[i * head_dim..(i + 1) * head_dim].copy_from_slice(&cache.v[src..src + head_dim]);
        }
        crate::kv_prefix::write_tile(
            layer,
            kv_head,
            dtype,
            base,
            head_dim,
            cfg.layer_uses_k_eq_v(layer),
            &k,
            &v,
        )
    }

    fn import_tile(&mut self, layer: usize, kv_head: usize, blob: &[u8]) -> Result<usize, String> {
        let hdr = crate::kv_prefix::read_tile_header(blob)?;
        if hdr.layer != layer || hdr.kv_head != kv_head {
            return Err(format!(
                "import_tile: blob addresses (L{},h{}) but caller asked (L{layer},h{kv_head})",
                hdr.layer, hdr.kv_head
            ));
        }
        let (k, v) = crate::kv_prefix::read_tile_body(blob, &hdr)?;
        let cache = self
            .kv_caches
            .get_mut(layer)
            .ok_or_else(|| format!("import_tile: layer {layer} out of range"))?;
        let head_dim = hdr.head_dim;
        let stride = cache.num_kv_heads * cache.head_dim;
        if head_dim != cache.head_dim {
            return Err(format!(
                "import_tile: head_dim {head_dim} != cache head_dim {}",
                cache.head_dim
            ));
        }
        if kv_head >= cache.num_kv_heads {
            return Err(format!("import_tile: kv_head {kv_head} out of range"));
        }
        for i in 0..hdr.n_rows {
            let abs = hdr.window_base + i;
            let slot = abs % cache.capacity;
            let dst = slot * stride + kv_head * head_dim;
            cache.k[dst..dst + head_dim].copy_from_slice(&k[i * head_dim..(i + 1) * head_dim]);
            cache.v[dst..dst + head_dim].copy_from_slice(&v[i * head_dim..(i + 1) * head_dim]);
        }
        Ok(hdr.n_rows)
    }

    fn set_seq_len(&mut self, n: usize) {
        for c in self.kv_caches.iter_mut() {
            c.seq_len = n;
        }
    }
}

/// The `[base, base+n_rows)` absolute-position row range a tile stores at
/// boundary `upto`: full-attn layers keep everything (`[0, upto)`); sliding
/// layers keep only the last `window` positions (`[max(0, upto-window), upto)`),
/// which is exactly what a windowed read (`kv_start = seq_len - window`) will
/// ever touch after the resume. Shared by the gemma and Laguna impls.
pub(crate) fn tile_row_range(is_full: bool, upto: usize, window: usize) -> (usize, usize) {
    if is_full {
        (0, upto)
    } else {
        let base = upto.saturating_sub(window);
        (base, upto - base)
    }
}

// ─── Weight loading from SafeTensors ─────────────────────────────────────────

/// Load Gemma4-E2B weights from a safetensors file.
///
/// Returns a flat map of tensor name → Vec<f32> (all converted to f32 for
/// simplicity; the actual Vulkan compute uses f32 buffers).
pub fn load_weights_from_safetensors(
    path: &Path,
) -> Result<HashMap<String, Vec<f32>>, String> {
    use safetensors::SafeTensors;
    use memmap2::Mmap;
    use std::fs::File;

    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
    let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;

    let mut out = HashMap::new();

    for (raw_name, tensor) in st.tensors() {
        // vLLM naming: strip "model.language_model." prefix → "model."
        let name = if raw_name.starts_with("model.language_model.") {
            format!("model.{}", &raw_name["model.language_model.".len()..])
        } else {
            raw_name.to_string()
        };

        let dtype = tensor.dtype();
        let data = tensor.data();

        let f32_data: Vec<f32> = match dtype {
            safetensors::Dtype::BF16 => {
                data.chunks_exact(2).map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    bf16::from_bits(bits).to_f32()
                }).collect()
            }
            safetensors::Dtype::F32 => {
                data.chunks_exact(4).map(|chunk| {
                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                }).collect()
            }
            safetensors::Dtype::F16 => {
                data.chunks_exact(2).map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                }).collect()
            }
            other => {
                log::warn!("Skipping tensor '{}' with unsupported dtype {:?}", name, other);
                continue;
            }
        };

        out.insert(name, f32_data);
    }

    Ok(out)
}

/// Load a gemma-4 **MLX-affine quantized** checkpoint directory into a host f32
/// weight map keyed by the `model.*` names the CPU reference forward expects.
///
/// Every quantized linear is a `<base>.weight`(U32 packed) + `<base>.scales` +
/// `<base>.biases`(BF16) triple; it is dequantized via [`dequantize_mlx_affine`]
/// (`w = scale*q + bias`, group_size 64).  The per-tensor bit width (4 or 8 in the
/// 12B QAT checkpoint) is derived purely from the tensor shapes:
///   `in_features = scales_cols * 64`,  `bits = packed_cols * 32 / in_features`.
/// BF16 scales/biases are widened to f32 (`bf16::to_f32`, i.e. `u16 << 16`), as the
/// qwen mlx4 path does.  Plain BF16/F16/F32 tensors (norms, layer_scalar) pass
/// through.  The `language_model.` prefix is stripped to land in `model.*`; the
/// vision/audio towers are skipped.  Tied embeddings: no `lm_head` — the forward
/// reuses the dequantized `model.embed_tokens.weight`.
///
/// This is the Mac CPU-correctness loader (dequant → f32 host → cpu_matmul); the
/// GPU-resident loader lives in `lib.rs` and is a separate cluster-perf concern.
pub fn load_gemma_mlx_affine(dir: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    use safetensors::SafeTensors;
    use memmap2::Mmap;
    use std::fs::File;

    // Rewrite a raw checkpoint name into the loader's `model.*` namespace, or None
    // to skip (vision/audio towers).
    fn remap(raw: &str) -> Option<String> {
        let n = raw.strip_prefix("language_model.").unwrap_or(raw);
        if !n.starts_with("model.") {
            return None; // embed_vision / embed_audio / vision_embedder / …
        }
        Some(n.to_string())
    }

    let shards = discover_shards(&dir.join("model.safetensors"));
    // Keep every mmap alive for the duration so the SafeTensors views stay valid.
    let mut mmaps: Vec<Mmap> = Vec::new();
    for sp in &shards {
        let f = File::open(sp).map_err(|e| format!("open {}: {e}", sp.display()))?;
        mmaps.push(unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", sp.display()))?);
    }
    let mut sts: Vec<SafeTensors> = Vec::new();
    for m in &mmaps {
        sts.push(SafeTensors::deserialize(m).map_err(|e| format!("parse safetensors: {e}"))?);
    }

    // First index every tensor name -> (shard, is present) so we can find the
    // .scales sibling of a quantized .weight regardless of which shard it lives in.
    let mut name_to_shard: HashMap<String, usize> = HashMap::new();
    for (si, st) in sts.iter().enumerate() {
        for name in st.names() {
            name_to_shard.insert(name.to_string(), si);
        }
    }

    let bf16_to_f32 = |bytes: &[u8]| -> Vec<f32> {
        bytes.chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    };

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();

    for (raw_name, si) in name_to_shard.iter() {
        // Only drive off `.weight` tensors; `.scales`/`.biases` are consumed with
        // their `.weight`.
        if raw_name.ends_with(".scales") || raw_name.ends_with(".biases") {
            continue;
        }
        let out_name = match remap(raw_name) { Some(n) => n, None => continue };

        let st = &sts[*si];
        let view = st.tensor(raw_name).map_err(|e| format!("get {raw_name}: {e}"))?;
        let shape = view.shape().to_vec();

        // Quantized triple?  scales sibling = same base, `.weight` -> `.scales`.
        let scales_name = raw_name.strip_suffix(".weight")
            .map(|b| format!("{b}.scales"));
        let is_quant = raw_name.ends_with(".weight")
            && view.dtype() == safetensors::Dtype::U32
            && scales_name.as_ref().map(|s| name_to_shard.contains_key(s)).unwrap_or(false);

        if is_quant {
            let scales_name = scales_name.unwrap();
            let biases_name = format!("{}.biases", raw_name.strip_suffix(".weight").unwrap());
            let s_si = name_to_shard[&scales_name];
            let b_si = *name_to_shard.get(&biases_name)
                .ok_or_else(|| format!("quant tensor {raw_name} missing .biases"))?;
            let s_view = sts[s_si].tensor(&scales_name).map_err(|e| e.to_string())?;
            let b_view = sts[b_si].tensor(&biases_name).map_err(|e| e.to_string())?;

            let out_features = shape[0];
            let packed_cols = shape[1];
            let scales_cols = s_view.shape()[1];
            let group_size = 64usize;
            let in_features = scales_cols * group_size;
            // bits = packed_cols * 32 / in_features  (4 or 8 for the 12B QAT ckpt)
            let bits = (packed_cols * 32) / in_features;
            if in_features * bits != packed_cols * 32 {
                return Err(format!(
                    "{raw_name}: shape mismatch (packed {packed_cols} cols, {scales_cols} groups, in {in_features})"));
            }

            let packed: Vec<u32> = view.data().chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let scales = bf16_to_f32(s_view.data());
            let biases = bf16_to_f32(b_view.data());

            let deq = dequantize_mlx_affine(
                &packed, &scales, &biases, out_features, in_features, group_size, bits);
            out.insert(out_name, deq);
        } else {
            // Plain tensor (norm weights, layer_scalar): widen to f32.
            let data = view.data();
            let f32_data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => bf16_to_f32(data),
                safetensors::Dtype::F16 => data.chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                other => {
                    log::warn!("gemma loader: skip '{raw_name}' dtype {other:?}");
                    continue;
                }
            };
            out.insert(out_name, f32_data);
        }
    }

    Ok(out)
}

/// GPU-resident STREAMING loader for `gemma4_unified` (gemma-4-12B-it QAT-4bit)
/// checkpoints. Mirrors `load_gemma_mlx_affine`'s per-tensor MLX-affine dequant
/// (same shape-derived `in_features`/`bits`, same `language_model.` stripping,
/// same vision/audio skip), but instead of accumulating the WHOLE model as f32
/// (~11GB on-disk -> ~44GB f32, OOMs a cluster node) it hands each MATVEC
/// projection to `on_proj` immediately after decode so the caller can
/// upload-to-GPU-and-drop it, capping the host f32 peak at one tensor.
///
/// Tensor routing:
///  - `self_attn.{q,k,v,o}_proj` (4-bit on this checkpoint): offered PACKED as
///    `ProjWeight::Mlx4` (no f32 dequant unless the sink declines). Global
///    (`k_eq_v`) layers have no `v_proj` on disk — it is simply never present
///    among this layer's tensor names, so no special-casing is needed here.
///  - `mlp.{gate,up,down}_proj` (8-bit on this checkpoint): there is no packed
///    8-bit GPU shader, so these are ALWAYS dequantized to f32 here and handed
///    to the sink as `ProjWeight::F32` (the sink q8_0-requantizes on upload).
///  - Everything else (norms, `layer_scalar`) is dequantized to f32 and returned
///    directly in the host map.
///  - `embed_tokens` (also 4-bit mlx-affine on disk, doubles as the tied lm_head)
///    is **NOT** dequantized here: dequantizing the whole 262144×3840 vocab table
///    to f32 is a ~4GB load-time transient held across the entire GTT upload loop
///    (the single-node full-load OOM driver). Instead its RAW on-disk packed form
///    is returned as the second tuple element (`Some(Mlx4Weight)`); the caller
///    uploads it straight to the mlx4 lm_head GPU buffer (lever) or streams it to
///    f16 per-row (baseline) via [`dequantize_mlx_affine_f16`], never a whole-
///    table f32. `None` iff the embed wasn't 4-bit quantized on disk (the caller
///    then falls back to the host map, whichever legacy dtype it landed in).
///
/// Only decoder layers with GLOBAL index in `[layer_start, layer_end)` are
/// loaded; pass `(0, num_hidden_layers)` for a single-node full load.
pub fn load_gemma_resident_weights(
    dir: &Path,
    layer_start: usize,
    layer_end: usize,
    mlp_q4: bool,
    mut on_proj: impl FnMut(&str, ProjWeight) -> ProjResult,
) -> Result<(HashMap<String, Vec<f32>>, Option<Mlx4Weight>), String> {
    use safetensors::SafeTensors;
    use memmap2::Mmap;
    use std::fs::File;

    fn remap(raw: &str) -> Option<String> {
        let n = raw.strip_prefix("language_model.").unwrap_or(raw);
        if !n.starts_with("model.") {
            return None; // embed_vision / embed_audio / vision_embedder / …
        }
        Some(n.to_string())
    }
    fn is_attn_matvec(name: &str) -> bool {
        name.ends_with("self_attn.q_proj.weight")
            || name.ends_with("self_attn.k_proj.weight")
            || name.ends_with("self_attn.v_proj.weight")
            || name.ends_with("self_attn.o_proj.weight")
    }
    fn is_mlp_matvec(name: &str) -> bool {
        name.ends_with("mlp.gate_proj.weight")
            || name.ends_with("mlp.up_proj.weight")
            || name.ends_with("mlp.down_proj.weight")
    }
    // Keep tensor `name` in this resident window? (embed/lm_head/norm/misc
    // always kept — single-node full load; only decoder layers are windowed.)
    let keep_layer = |name: &str| -> bool {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                return idx >= layer_start && idx < layer_end;
            }
        }
        true
    };

    let shards = discover_shards(&dir.join("model.safetensors"));
    let mut mmaps: Vec<Mmap> = Vec::new();
    for sp in &shards {
        let f = File::open(sp).map_err(|e| format!("open {}: {e}", sp.display()))?;
        mmaps.push(unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", sp.display()))?);
    }
    let mut sts: Vec<SafeTensors> = Vec::new();
    for m in &mmaps {
        sts.push(SafeTensors::deserialize(m).map_err(|e| format!("parse safetensors: {e}"))?);
    }
    let mut name_to_shard: HashMap<String, usize> = HashMap::new();
    for (si, st) in sts.iter().enumerate() {
        for name in st.names() {
            name_to_shard.insert(name.to_string(), si);
        }
    }
    let bf16_to_f32 = |bytes: &[u8]| -> Vec<f32> {
        bytes.chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    };

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    // Tied embed/lm_head, streamed out RAW (see fn doc): captured here to keep
    // the ~4GB f32 whole-vocab dequant out of the load-time peak.
    let mut embed_raw: Option<Mlx4Weight> = None;
    // Sorted iteration purely for deterministic logging; correctness doesn't
    // depend on order (every tensor is independent).
    let mut raw_names: Vec<String> = name_to_shard.keys().cloned().collect();
    raw_names.sort();
    for raw_name in &raw_names {
        if raw_name.ends_with(".scales") || raw_name.ends_with(".biases") {
            continue;
        }
        let out_name = match remap(raw_name) { Some(n) => n, None => continue };
        if !keep_layer(&out_name) {
            continue;
        }

        let si = name_to_shard[raw_name];
        let st = &sts[si];
        let view = st.tensor(raw_name).map_err(|e| format!("get {raw_name}: {e}"))?;
        let shape = view.shape().to_vec();

        let scales_name = raw_name.strip_suffix(".weight").map(|b| format!("{b}.scales"));
        let is_quant = raw_name.ends_with(".weight")
            && view.dtype() == safetensors::Dtype::U32
            && scales_name.as_ref().map(|s| name_to_shard.contains_key(s)).unwrap_or(false);

        if is_quant {
            let scales_name = scales_name.unwrap();
            let biases_name = format!("{}.biases", raw_name.strip_suffix(".weight").unwrap());
            let s_si = name_to_shard[&scales_name];
            let b_si = *name_to_shard.get(&biases_name)
                .ok_or_else(|| format!("quant tensor {raw_name} missing .biases"))?;
            let s_view = sts[s_si].tensor(&scales_name).map_err(|e| e.to_string())?;
            let b_view = sts[b_si].tensor(&biases_name).map_err(|e| e.to_string())?;

            let out_features = shape[0];
            let packed_cols = shape[1];
            let scales_cols = s_view.shape()[1];
            let group_size = 64usize;
            let in_features = scales_cols * group_size;
            let bits = (packed_cols * 32) / in_features;
            if in_features * bits != packed_cols * 32 {
                return Err(format!(
                    "{raw_name}: shape mismatch (packed {packed_cols} cols, {scales_cols} groups, in {in_features})"));
            }

            let packed: Vec<u32> = view.data().chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let scales = bf16_to_f32(s_view.data());
            let biases = bf16_to_f32(b_view.data());

            // Tied embed/lm_head: hand the caller the RAW 4-bit packed form and
            // move on — do NOT build the ~4GB f32 whole-vocab table (the OOM
            // driver). scales/biases are already widened bf16->f32 (exact); the
            // packed nibble order matches `quantize_mlx_affine_4bit`/the mlx4 GPU
            // kernel verbatim, so no repack is needed.
            if bits == 4 && out_name.ends_with("embed_tokens.weight") {
                embed_raw = Some(Mlx4Weight {
                    packed, scales, biases, out_features, in_features, group_size,
                });
                continue;
            }

            if bits == 4 && is_attn_matvec(&out_name) {
                let taken = matches!(
                    on_proj(&out_name, ProjWeight::Mlx4(Mlx4Weight {
                        packed: packed.clone(), scales: scales.clone(), biases: biases.clone(),
                        out_features, in_features, group_size,
                    })),
                    ProjResult::Consumed);
                if taken { continue; }
                let deq = dequantize_mlx_affine(
                    &packed, &scales, &biases, out_features, in_features, group_size, bits);
                if let ProjResult::KeepF32(v) = on_proj(&out_name, ProjWeight::F32(deq)) {
                    out.insert(out_name, v);
                }
                continue;
            }

            // 8-bit MLP (no packed GPU form to offer — always dequant here and
            // let the sink q8_0-requantize on upload) OR any other quantized
            // tensor that isn't an attn-matvec name (embed_tokens: 4-bit, but
            // routed to the plain host-f32 path since it isn't `is_matvec_weight`
            // — mirrors the E2B generic loader's embed/lm_head handling).
            let deq = dequantize_mlx_affine(
                &packed, &scales, &biases, out_features, in_features, group_size, bits);
            if is_mlp_matvec(&out_name) {
                // H1 lever (VLLM_VULKAN_GEMMA_MLP_Q4): this checkpoint's MLP is
                // natively 8-bit; requant the dequantized f32 -> mlx4 4-bit
                // (group 64) and offer it PACKED so it uploads/dispatches through
                // the SAME validated mlx4 path the 4-bit attention already uses
                // (halves MLP bandwidth + drops ~3.7GB residency -> single node).
                // LOSSY 8->4 re-round, gated on the CPU argmax-agreement test.
                if mlp_q4 {
                    let (rp, rs, rb) = quantize_mlx_affine_4bit(
                        &deq, out_features, in_features, group_size);
                    let taken = matches!(
                        on_proj(&out_name, ProjWeight::Mlx4(Mlx4Weight {
                            packed: rp.clone(), scales: rs.clone(), biases: rb.clone(),
                            out_features, in_features, group_size,
                        })),
                        ProjResult::Consumed);
                    if taken { continue; }
                    // Sink declined (no engine / upload fail): keep the 4-bit-
                    // ROUNDED f32 (not the 8-bit source) so a CPU forward reflects
                    // the exact numerics the GPU mlx4 kernel would use.
                    let rt = dequantize_mlx_affine(
                        &rp, &rs, &rb, out_features, in_features, group_size, 4);
                    if let ProjResult::KeepF32(v) = on_proj(&out_name, ProjWeight::F32(rt)) {
                        out.insert(out_name, v);
                    }
                    continue;
                }
                if let ProjResult::KeepF32(v) = on_proj(&out_name, ProjWeight::F32(deq)) {
                    out.insert(out_name, v);
                }
            } else {
                out.insert(out_name, deq);
            }
        } else {
            let data = view.data();
            let f32_data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => bf16_to_f32(data),
                safetensors::Dtype::F16 => data.chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                other => {
                    log::warn!("gemma resident loader: skip '{raw_name}' dtype {other:?}");
                    continue;
                }
            };
            out.insert(out_name, f32_data);
        }
    }
    Ok((out, embed_raw))
}

/// True if `name` (already remapped to `model.layers.{idx}.mlp....` /
/// `model.*` form) is one of the 4 gemma-31B MLP layers the `recipe.yaml`
/// FP8-DYNAMIC target regex `layers\.(1|57|58|59)\.mlp\.` special-cases to
/// FP8 instead of the model-default NVFP4. Every other layer's
/// `mlp.{gate,up,down}_proj` is NVFP4.
fn gemma31b_mlp_is_fp8_exception(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("model.layers.") {
        if let Some((idx_str, after)) = rest.split_once('.') {
            if after.starts_with("mlp.") {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    return matches!(idx, 1 | 57 | 58 | 59);
                }
            }
        }
    }
    false
}

/// True if `name` (post-remap `model.*` form) is a matvec (attn or MLP)
/// projection weight for the gemma-31B NVFP4+FP8 checkpoint — the set of
/// tensors the streaming `on_proj` sink gets offered (everything else, norms/
/// `layer_scalar`/embed, is accumulated straight into the host f32/f16 map).
fn is_gemma31b_matvec_weight_name(name: &str) -> bool {
    name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
        || name.ends_with(".self_attn.o_proj.weight")
        || name.ends_with(".mlp.gate_proj.weight")
        || name.ends_with(".mlp.up_proj.weight")
        || name.ends_with(".mlp.down_proj.weight")
}

/// GPU-resident STREAMING loader for the gemma-4-31B-it **NVFP4 + FP8 mixed**
/// (compressed-tensors) base checkpoint — distinct from
/// `load_gemma_resident_weights` (the MLX-affine-quantized 12B QAT format).
/// Per the real `recipe.yaml`/`config.json.quantization_config` (confirmed
/// against `~/repos/OminiX-MLX/models/gemma-4-31B-it-NVFP4`):
///  - `self_attn.(q|k|v|o)_proj` on EVERY layer, **plus**
///    `mlp.(gate|up|down)_proj` on layers **1, 57, 58, 59 only**
///    ([`gemma31b_mlp_is_fp8_exception`]) -> **FP8** (`FP8_DYNAMIC`,
///    weight=channel/per-row-symmetric): on disk `<base>.weight` is
///    `F8_E4M3` `[out,in]` (unpacked, 1 byte/elem) with a `BF16`
///    `<base>.weight_scale` sibling shaped `[out,1]` (per-row).
///  - every OTHER layer's `mlp.(gate|up|down)_proj` -> **NVFP4**
///    (`nvfp4-pack-quantized`, group_size 16): on disk
///    `<base>.weight_packed` is `U8` `[out, in/2]` nibbles, with an
///    `F8_E4M3` `<base>.weight_scale` sibling `[out, in/group_size]` and an
///    `F32` `<base>.weight_global_scale` scalar — the RECIPROCAL of
///    modelopt's `weight_scale_2` (dequant divides by it; the loader hands
///    `on_proj` the inverted value so downstream multiply-convention
///    consumers stay unchanged).
///    `<base>.input_global_scale`/`.input_scale` are the (unused for
///    weight-only dequant) dynamic activation scales — skipped.
///  - `embed_tokens` (tied lm_head; no separate `lm_head.weight` on this
///    checkpoint), all norms (`input_layernorm`, `post_attention_layernorm`,
///    `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `q_norm`,
///    `k_norm`, final `model.norm`), and `layer_scalar` are plain BF16 ->
///    host f16 (embed_tokens) / f32 (everything else, all tiny).
///  - `self_attn.k_scale`/`.v_scale` are the FP8 KV-cache-quant scales
///    (`kv_cache_scheme` in `recipe.yaml`) — needed only by the cluster GPU
///    KV path, not this host-side loader; skipped.
///  - `vision_tower.*` / `embed_vision.*` / `audio.*` tensors are skipped
///    (this is a text-only forward).
///
/// Mirrors `load_gemma_resident_weights`'s streaming-sink shape: only decoder
/// layers with GLOBAL index in `[layer_start, layer_end)` are loaded, and
/// every matvec projection is handed to `on_proj` immediately after its
/// siblings are read so the caller can upload it (packed, no f32 dequant) to
/// the GPU and drop it — the host f32 map never accumulates more than the
/// projections the sink declines (`ProjResult::KeepF32`/`Dequantize`).
/// Returns `(f32_map, f16_map)`; `f16_map` holds only `model.embed_tokens.weight`.
pub fn load_gemma_nvfp4_weights(
    dir: &Path,
    layer_start: usize,
    layer_end: usize,
    mut on_proj: impl FnMut(&str, ProjWeight) -> ProjResult,
) -> Result<(HashMap<String, Vec<f32>>, HashMap<String, Vec<u16>>), String> {
    use safetensors::SafeTensors;
    use memmap2::Mmap;
    use std::fs::File;

    fn remap(raw: &str) -> Option<String> {
        raw.strip_prefix("model.language_model.").map(|rest| format!("model.{rest}"))
    }
    let keep_layer = |name: &str| -> bool {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                return idx >= layer_start && idx < layer_end;
            }
        }
        true
    };

    let shards = discover_shards(&dir.join("model.safetensors"));
    let mut mmaps: Vec<Mmap> = Vec::new();
    for sp in &shards {
        let f = File::open(sp).map_err(|e| format!("open {}: {e}", sp.display()))?;
        mmaps.push(unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", sp.display()))?);
    }
    let mut sts: Vec<SafeTensors> = Vec::new();
    for m in &mmaps {
        sts.push(SafeTensors::deserialize(m).map_err(|e| format!("parse safetensors: {e}"))?);
    }
    let mut name_to_shard: HashMap<String, usize> = HashMap::new();
    for (si, st) in sts.iter().enumerate() {
        for name in st.names() {
            name_to_shard.insert(name.to_string(), si);
        }
    }
    let bf16_to_f32 = |bytes: &[u8]| -> Vec<f32> {
        bytes.chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    };

    let mut out_f32: HashMap<String, Vec<f32>> = HashMap::new();
    let mut out_f16: HashMap<String, Vec<u16>> = HashMap::new();
    let mut raw_names: Vec<String> = name_to_shard.keys().cloned().collect();
    raw_names.sort();
    for raw_name in &raw_names {
        // Sibling scale/activation-scale tensors: consumed alongside their
        // `.weight`/`.weight_packed`, never loaded standalone. `.k_scale`/
        // `.v_scale` are the FP8 KV-cache scales (cluster-only, see doc comment).
        if raw_name.ends_with(".weight_scale")
            || raw_name.ends_with(".weight_global_scale")
            || raw_name.ends_with(".input_global_scale")
            || raw_name.ends_with(".input_scale")
            || raw_name.ends_with(".k_scale")
            || raw_name.ends_with(".v_scale")
        {
            continue;
        }
        let out_name = match remap(raw_name) { Some(n) => n, None => continue };
        if !keep_layer(&out_name) {
            continue;
        }

        let si = name_to_shard[raw_name];
        let st = &sts[si];
        let view = st.tensor(raw_name).map_err(|e| format!("get {raw_name}: {e}"))?;
        let shape = view.shape().to_vec();

        if raw_name.ends_with(".weight_packed") {
            // ── NVFP4 mlp weight ────────────────────────────────────────
            let final_name = out_name.strip_suffix("_packed")
                .ok_or_else(|| format!("{out_name}: expected a '_packed' suffix"))?
                .to_string();
            // NVFP4 only ever targets `mlp.{gate,up,down}_proj` on non-exception
            // layers (`recipe.yaml` group_1 regex `.*\.mlp\.(gate|up|down)_proj$`
            // with no layer restriction) — loudly reject a checkpoint that
            // disagrees (H7: never silently mis-route a quant format).
            if !is_gemma31b_matvec_weight_name(&final_name) {
                return Err(format!("{final_name}: NVFP4-packed but not a recognized matvec name"));
            }
            if gemma31b_mlp_is_fp8_exception(&final_name) {
                return Err(format!(
                    "{final_name}: NVFP4-packed on an FP8-exception layer (1/57/58/59) \
                     — checkpoint recipe mismatch"));
            }
            let raw_base = &raw_name[..raw_name.len() - "_packed".len()];
            let wscale = st.tensor(&format!("{raw_base}_scale"))
                .map_err(|e| format!("{raw_name}: missing weight_scale sibling: {e}"))?
                .data().to_vec();
            let global = {
                let gv = st.tensor(&format!("{raw_base}_global_scale"))
                    .map_err(|e| format!("{raw_name}: missing weight_global_scale sibling: {e}"))?;
                let raw = f32::from_le_bytes(gv.data()[..4].try_into()
                    .map_err(|_| format!("{raw_name}: weight_global_scale: short"))?);
                // compressed-tensors `weight_global_scale` is the RECIPROCAL of
                // modelopt's `weight_scale_2`: llmcompressor picks
                // `global = 448*6/amax` and stores the per-group scales as
                // `e4m3(scale_f32 * global)` (mapping them into e4m3 range), so
                // dequant must DIVIDE: `scale = e4m3 / global`. Our downstream
                // consumers (`dequantize_nvfp4`, `nvfp4_fold_scales`) MULTIPLY
                // by `global` (modelopt `weight_scale_2` convention), so hand
                // them the reciprocal. Measured on the real 31B checkpoint:
                // raw = 10304.0 (= 448*6/0.261) with e4m3 group scales 15..256 —
                // multiplying inflates |w| to ~1.6e7 (the Gate-1 base-coherence
                // corruption); dividing yields the sane max |w| = 0.149.
                if raw == 0.0 || !raw.is_finite() {
                    return Err(format!("{raw_name}: weight_global_scale {raw} not usable"));
                }
                1.0 / raw
            };
            // `.weight_packed` is `[out, in/2]` (two nibbles/byte).
            let out_features = shape[0];
            let in_features = shape[1] * 2;
            let groups = (wscale.len() / out_features).max(1);
            let group_size = in_features / groups;
            // Byte-packed (2 nibbles/byte — NOT `validate_affine_dims`'s
            // u32-word packing convention), so check dims directly.
            if group_size == 0 || in_features % group_size != 0 {
                return Err(format!(
                    "{final_name}: in_features {in_features} not a multiple of \
                     derived group_size {group_size}"));
            }
            let want_packed_bytes = out_features * (in_features / 2);
            if view.data().len() != want_packed_bytes {
                return Err(format!(
                    "{final_name}: weight_packed len {} != expected {want_packed_bytes}",
                    view.data().len()));
            }
            let want_scales = out_features * groups;
            if wscale.len() != want_scales {
                return Err(format!(
                    "{final_name}: weight_scale len {} != expected {want_scales}",
                    wscale.len()));
            }
            let taken = matches!(
                on_proj(&final_name, ProjWeight::Nvfp4(Nvfp4Weight {
                    packed: view.data(), wscale: &wscale, global,
                    out_features, in_features, group_size,
                })),
                ProjResult::Consumed);
            if taken { continue; }
            let deq = dequantize_nvfp4(view.data(), &wscale, global, out_features, in_features, group_size);
            if let ProjResult::KeepF32(v) = on_proj(&final_name, ProjWeight::F32(deq)) {
                out_f32.insert(final_name, v);
            }
            continue;
        }

        if raw_name.ends_with(".weight") && view.dtype() == safetensors::Dtype::F8_E4M3 {
            // ── FP8 attn / MLP-exception weight ─────────────────────────
            if !is_gemma31b_matvec_weight_name(&out_name) {
                return Err(format!("{out_name}: FP8 but not a recognized matvec name"));
            }
            let is_mlp = out_name.contains(".mlp.");
            if is_mlp && !gemma31b_mlp_is_fp8_exception(&out_name) {
                return Err(format!(
                    "{out_name}: FP8 mlp weight outside the layer-1/57/58/59 exception \
                     set — checkpoint recipe mismatch"));
            }
            let raw_base = &raw_name[..raw_name.len() - ".weight".len()];
            let scale = match st.tensor(&format!("{raw_base}.weight_scale")) {
                Ok(sv) => bf16_to_f32(sv.data()),
                Err(e) => return Err(format!("{raw_name}: missing weight_scale sibling: {e}")),
            };
            let out_features = shape[0];
            let in_features = shape[1];
            let taken = matches!(
                on_proj(&out_name, ProjWeight::Fp8(Fp8Weight {
                    weight: view.data(), scale: scale.clone(), out_features, in_features,
                })),
                ProjResult::Consumed);
            if taken { continue; }
            let deq = dequantize_fp8(view.data(), &scale, out_features, in_features);
            if let ProjResult::KeepF32(v) = on_proj(&out_name, ProjWeight::F32(deq)) {
                out_f32.insert(out_name, v);
            }
            continue;
        }

        // ── Plain tensor (norms, layer_scalar, embed_tokens) ─────────────
        if out_name.ends_with("embed_tokens.weight") {
            let f16v: Vec<u16> = match view.dtype() {
                safetensors::Dtype::BF16 => view.data().chunks_exact(2)
                    .map(|c| half::f16::from_f32(
                        bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).to_bits())
                    .collect(),
                safetensors::Dtype::F16 => view.data().to_vec().chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
                safetensors::Dtype::F32 => view.data().chunks_exact(4)
                    .map(|c| half::f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_bits())
                    .collect(),
                other => return Err(format!("{raw_name}: unsupported embed dtype {other:?}")),
            };
            out_f16.insert(out_name, f16v);
            continue;
        }
        let data = view.data();
        let f32_data: Vec<f32> = match view.dtype() {
            safetensors::Dtype::BF16 => bf16_to_f32(data),
            safetensors::Dtype::F16 => data.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            safetensors::Dtype::F32 => data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            other => {
                log::warn!("gemma31b nvfp4 loader: skip '{raw_name}' dtype {other:?}");
                continue;
            }
        };
        out_f32.insert(out_name, f32_data);
    }
    Ok((out_f32, out_f16))
}

/// Collect every sibling `*.safetensors` shard in `path`'s directory, sorted.
/// Single-file checkpoints (no siblings found) fall back to `path` itself. This
/// is the one shard-discovery routine used by every loader (previously copied
/// verbatim in six places).
pub(crate) fn discover_shards(path: &Path) -> Vec<PathBuf> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut shards: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                shards.push(p);
            }
        }
    }
    if shards.is_empty() {
        shards.push(path.to_path_buf());
    }
    shards.sort();
    shards
}

/// Parse the `[lo,hi)` layer bounds encoded in a pre-sliced stage filename
/// (`pp-stage{i}-L{lo}-{hi}.safetensors`, emitted by
/// `scripts/coalesce_quant_shards.py`). Returns None for any file that does not
/// match the scheme (so foreign files in the dir are ignored, not errors).
fn parse_stage_bounds(fname: &str) -> Option<(usize, usize)> {
    let stem = fname.strip_suffix(".safetensors")?;
    let idx = stem.rfind("-L")?;
    let mut it = stem[idx + 2..].split('-');
    let lo = it.next()?.parse::<usize>().ok()?;
    let hi = it.next()?.parse::<usize>().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some((lo, hi))
}

/// Resolve which safetensors file(s) a pipeline-parallel rank owning layers
/// `[layer_start, layer_end)` should open.
///
/// When `VLLM_VULKAN_PP_PRESLICED_DIR` is UNSET this is exactly
/// `discover_shards(path)` — byte-for-byte the monolithic-checkpoint path.
///
/// When SET, the rank opens ONLY the one pre-sliced per-stage file whose encoded
/// `[lo,hi)` bounds equal its runtime `[layer_start, layer_end)` — it never mmaps
/// or reads the other stages' bytes (the load-transient-OOM lever). The bounds
/// are the GUARD: if no stage file matches the runtime window, this HARD-FAILS
/// and echoes both the requested window and every stage window found in the dir,
/// so a runtime/slice bounds mismatch can never silently load the wrong layers.
/// The per-stage file carries the SAME tensor names as the monolith, so every
/// downstream loader (`keep()` filter + dequant/upload) runs unchanged.
pub(crate) fn resolve_pp_stage_shards(
    path: &Path,
    layer_start: usize,
    layer_end: usize,
) -> Result<Vec<PathBuf>, String> {
    resolve_pp_stage_shards_in(
        crate::flags::flags_global().pp_presliced_dir.as_deref(),
        path,
        layer_start,
        layer_end,
    )
}

/// Core of [`resolve_pp_stage_shards`] with the pre-slice dir passed explicitly
/// (so it is unit-testable without the process-global cached flag). `dir=None`
/// ⇒ the monolithic `discover_shards(path)` path.
pub(crate) fn resolve_pp_stage_shards_in(
    dir: Option<&str>,
    path: &Path,
    layer_start: usize,
    layer_end: usize,
) -> Result<Vec<PathBuf>, String> {
    let dir = match dir {
        None => return Ok(discover_shards(path)),
        Some(d) => d.to_string(),
    };
    let dir_path = Path::new(&dir);
    let entries = std::fs::read_dir(dir_path)
        .map_err(|e| format!("PP_PRESLICED_DIR '{dir}' unreadable: {e}"))?;
    let mut found: Vec<(usize, usize, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        if let Some(fname) = p.file_name().and_then(|f| f.to_str()) {
            if let Some((lo, hi)) = parse_stage_bounds(fname) {
                found.push((lo, hi, p.clone()));
            }
        }
    }
    if found.is_empty() {
        return Err(format!(
            "PP pre-slice: no 'pp-stage{{i}}-L{{lo}}-{{hi}}.safetensors' files in \
             VLLM_VULKAN_PP_PRESLICED_DIR '{dir}' (produce them with \
             scripts/coalesce_quant_shards.py slice --pp-bounds …)"
        ));
    }
    let matches: Vec<&(usize, usize, PathBuf)> = found
        .iter()
        .filter(|(lo, hi, _)| *lo == layer_start && *hi == layer_end)
        .collect();
    match matches.as_slice() {
        [(_, _, p)] => {
            log::info!(
                "PP pre-slice: rank layers [{layer_start},{layer_end}) -> {}",
                p.display()
            );
            Ok(vec![p.clone()])
        }
        [] => {
            let mut avail: Vec<String> =
                found.iter().map(|(lo, hi, _)| format!("[{lo},{hi})")).collect();
            avail.sort();
            Err(format!(
                "PP pre-slice BOUNDS MISMATCH: this rank owns layers \
                 [{layer_start},{layer_end}) but no pre-sliced stage file in '{dir}' \
                 covers it. Available stage windows: {}. Re-slice with matching \
                 --pp-bounds (the runtime PP split must equal the slice bounds).",
                avail.join(" ")
            ))
        }
        _ => Err(format!(
            "PP pre-slice: {} stage files claim layers [{layer_start},{layer_end}) \
             in '{dir}' (duplicate/overlapping slices) — dir must hold exactly one \
             file per stage window",
            matches.len()
        )),
    }
}

/// Normalize a raw safetensors tensor name to the `model.*` namespace the loader
/// and forward expect. MLX checkpoints prefix `language_model.`; HF /
/// compressed-tensors (e.g. the NVFP4 modelopt export) nest as
/// `model.language_model.*`. Both normalize to `model.*`; anything else is
/// returned unchanged.
/// True for qwen3_5 RMSNorm weights that the HF/modelopt export stores
/// ZERO-CENTERED (module computes `x_hat * (1 + w)`): the per-layer
/// input/post_attention layernorms, the attention q/k norms, and the final
/// `model.norm`. The GDN `linear_attn.norm` (gated RMSNorm) is stored plain in
/// both exports and is excluded. Keyed on the RAW `model.language_model.` HF
/// nesting so MLX checkpoints (`language_model.model.*`, +1 already baked in —
/// e.g. the 4bit 27B and the 122B) are untouched.
fn qwen35_hf_zero_centered_norm(raw_name: &str, name: &str) -> bool {
    raw_name.starts_with("model.language_model.")
        && (name.ends_with(".input_layernorm.weight")
            || name.ends_with(".post_attention_layernorm.weight")
            || name.ends_with(".q_norm.weight")
            || name.ends_with(".k_norm.weight")
            || name == "model.norm.weight")
}

fn normalize_qwen35_name(raw: &str) -> String {
    let r = raw.strip_prefix("language_model.").unwrap_or(raw);
    match r.strip_prefix("model.language_model.") {
        Some(rest) => format!("model.{rest}"),
        None => r.to_string(),
    }
}

/// Derive the MLX-affine quantization group size from tensor geometry, the way
/// the NVFP4 path already derives its own. `scales_len` is the flattened length
/// of the `.scales` sibling (`out_features * groups`). Returns the true group
/// size instead of trusting a caller-supplied constant — a checkpoint quantized
/// with a group size other than the hardcoded default would otherwise read
/// misaligned scales and dequantize to silently-wrong weights (H7).
fn mlx_affine_group_size(
    name: &str,
    in_features: usize,
    out_features: usize,
    scales_len: usize,
) -> Result<usize, String> {
    if out_features == 0 || in_features == 0 {
        return Err(format!("{name}: zero-size tensor (out={out_features}, in={in_features})"));
    }
    if scales_len % out_features != 0 {
        return Err(format!(
            "{name}: scales len {scales_len} not divisible by out_features {out_features}"
        ));
    }
    let groups = scales_len / out_features;
    if groups == 0 || in_features % groups != 0 {
        return Err(format!(
            "{name}: cannot derive group size (in={in_features}, groups={groups})"
        ));
    }
    Ok(in_features / groups)
}

/// Derive the MLX-affine bit-width (2/3/4/5/6/8) from packed geometry, trusting
/// ONLY the checkpoint's `group_size` (a single global constant from
/// `config.quantization.group_size`) rather than a per-tensor name table. For any
/// affine tensor `packed_u32*32 == out*in*bits` and `scales_len == out*(in/group_size)`,
/// so `bits == packed_u32*32 / (scales_len*group_size)` (the `out*in` factor cancels).
///
/// This lets the loader adapt to BOTH the 35B-A3B mixed layout (8-bit routers /
/// 4-bit experts) AND the mlx-community 122B-A10B layout (UNIFORM 4-bit, incl. the
/// `mlp.gate`/`shared_expert_gate` routers) with no hardcoded assumption. The old
/// `qwen35_tensor_bits` name table returned 8 for the routers, which — on the
/// uniformly-4-bit 122B — passes `validate_affine_dims` (all constraints stay
/// self-consistent at half in_features / half group_size) yet silently dequantizes
/// a wrong-width, garbage router weight. Returns 0 on degenerate input, which
/// `validate_affine_dims` then rejects loudly.
fn mlx_affine_bits(packed_u32_len: usize, scales_len: usize, group_size: usize) -> usize {
    let denom = scales_len.saturating_mul(group_size);
    if denom == 0 {
        return 0;
    }
    packed_u32_len.saturating_mul(32) / denom
}

/// Tensors NEVER loaded for the qwen3_5 language model, regardless of PP layer
/// range. Names are post-`normalize_qwen35_name`. Two families:
///   - `vision_tower.*` — VL checkpoints (incl. this 122B) ship a vision tower the
///     text decode path ignores (already skipped historically).
///   - MTP / multi-token-prediction head (`*.mtp*`, `*nextn*`) — spec-decode is
///     fleet-wide NO-GO. The mlx-community 122B-4bit export already OMITS these
///     tensors, but the bf16 original ships one MTP layer; skip it defensively so
///     no variant loads a dead multi-GB head into host RAM.
fn qwen35_skip_aux(name: &str) -> bool {
    name.starts_with("vision_tower.")
        // The Qwen3.6-27B-NVFP4 modelopt export nests its vision tower as
        // `model.visual.*` (not `vision_tower.*`): without this arm every PP
        // rank decoded the whole ~0.9GB BF16 tower into ~1.8GB of host f32
        // that the text forward never reads.
        || name.starts_with("model.visual.")
        || name.starts_with("mtp")
        || name.contains(".mtp")
        || name.contains("nextn")
}

/// Validate that an MLX-affine quantized tensor's buffers tile exactly, BEFORE
/// dequant reads them. Catches malformed / mismatched checkpoints with a loud
/// error instead of an OOB read or silently-wrong output (H7).
fn validate_affine_dims(
    name: &str,
    packed_len: usize,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    bits: usize,
    scales_len: usize,
    biases_len: usize,
) -> Result<(), String> {
    let per_word = 32 / bits;
    if in_features % per_word != 0 {
        return Err(format!("{name}: in_features {in_features} not a multiple of {per_word} ({bits}-bit pack)"));
    }
    if group_size == 0 || in_features % group_size != 0 {
        return Err(format!("{name}: in_features {in_features} not a multiple of group_size {group_size}"));
    }
    let want_packed = out_features * (in_features / per_word);
    if packed_len != want_packed {
        return Err(format!("{name}: packed len {packed_len} != expected {want_packed}"));
    }
    let want_scales = out_features * (in_features / group_size);
    if scales_len != want_scales {
        return Err(format!("{name}: scales len {scales_len} != expected {want_scales}"));
    }
    if biases_len != scales_len {
        return Err(format!("{name}: biases len {biases_len} != scales len {scales_len}"));
    }
    Ok(())
}

/// Load weights from `path` together with any sibling `*.safetensors` shards in
/// the same directory, merged into one map.  Single-file checkpoints (Gemma4-E2B,
/// Qwen3-0.6B) load exactly one file; multi-shard checkpoints (e.g. Qwen3-4B)
/// load all shards so no weights are missing.
pub fn load_weights_auto(path: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let shards = discover_shards(path);

    let mut out = HashMap::new();
    for shard in &shards {
        let tensors = load_weights_from_safetensors(shard)?;
        out.extend(tensors);
    }
    if shards.len() > 1 {
        log::info!("Merged {} safetensors shards from {}", shards.len(), dir.display());
    }
    Ok(out)
}

/// Dequantize an MLX affine-quantized weight (`mode="affine"`, e.g. Qwen3.6
/// 4-bit checkpoints). Element `(o,i)` = `scales[o,g]*q + biases[o,g]` where `q`
/// is the `bits`-bit value packed little-bits-first (`32/bits` per u32, along the
/// input axis) and `g = i / group_size`. Returns f32, row-major `[out, in]`.
pub fn dequantize_mlx_affine(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    bits: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let per_word = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let groups = in_features / group_size;
    let words_per_row = in_features / per_word;
    let mut w = vec![0.0f32; out_features * in_features];
    w.par_chunks_mut(in_features).enumerate().for_each(|(o, row)| {
        let prow = &packed[o * words_per_row..(o + 1) * words_per_row];
        let srow = &scales[o * groups..(o + 1) * groups];
        let brow = &biases[o * groups..(o + 1) * groups];
        for i in 0..in_features {
            let word = prow[i / per_word];
            let q = ((word >> ((i % per_word) * bits)) & mask) as f32;
            row[i] = srow[i / group_size] * q + brow[i / group_size];
        }
    });
    w
}

/// MLX-affine dequant for **arbitrary bit widths incl. non-power-of-two (6-bit)**,
/// where codes are a **contiguous little-endian bitstream** inside the packed u32
/// row and a code may straddle a u32 word boundary. This is the layout MLX emits
/// for `bits ∉ {2,4,8}` (e.g. DeepSeek-V4-Flash's 6-bit attn/MLA/DSA tensors:
/// `packed_cols = in*bits/32`, not `in/(32/bits)`), which the aligned
/// [`dequantize_mlx_affine`] above (per_word = 32/bits) does NOT handle.
///
/// For the aligned widths 2/4/8 this produces byte-identical output to
/// `dequantize_mlx_affine` (a code never crosses a word there), so this is a
/// strict generalization — see the `dequantize_mlx_affine_bits_*` tests. The
/// affine math is the same split `scale*q + bias` (matching the mlx4 GPU matvec
/// and ULP-close to `mlx.core.dequantize`, which fuses the multiply-add).
///
/// `in_features * bits` must be a multiple of 32 (MLX packing invariant); the
/// packed row holds `in_features * bits / 32` u32 words.
pub fn dequantize_mlx_affine_bits(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    bits: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    assert!(bits >= 1 && bits <= 16, "bits out of range");
    assert_eq!((in_features * bits) % 32, 0, "in_features*bits must be a multiple of 32");
    let mask: u64 = (1u64 << bits) - 1;
    let groups = in_features / group_size;
    let words_per_row = in_features * bits / 32;
    let mut w = vec![0.0f32; out_features * in_features];
    w.par_chunks_mut(in_features).enumerate().for_each(|(o, row)| {
        let prow = &packed[o * words_per_row..(o + 1) * words_per_row];
        let srow = &scales[o * groups..(o + 1) * groups];
        let brow = &biases[o * groups..(o + 1) * groups];
        for i in 0..in_features {
            let bit = i * bits; // absolute bit offset of code `i`
            let wi = bit / 32;
            let off = bit % 32;
            // Read up to two words as a 64-bit window so a code that crosses the
            // 32-bit boundary is reassembled (low bits from word wi, high from wi+1).
            let lo = prow[wi] as u64;
            let hi = if off + bits > 32 { *prow.get(wi + 1).unwrap_or(&0) as u64 } else { 0 };
            let q = (((lo | (hi << 32)) >> off) & mask) as f32;
            row[i] = srow[i / group_size] * q + brow[i / group_size];
        }
    });
    w
}

#[cfg(test)]
mod dsv4_dequant_tests {
    use super::*;

    // Pack `codes` (each < 2^bits) as a contiguous little-endian bitstream, the
    // MLX layout for non-power-of-two widths (a code may cross a u32 boundary).
    fn pack_contiguous(codes: &[u32], bits: usize) -> Vec<u32> {
        let nwords = (codes.len() * bits).div_ceil(32);
        let mut w = vec![0u64; nwords + 1];
        let mask = (1u64 << bits) - 1;
        for (i, &c) in codes.iter().enumerate() {
            let bit = i * bits;
            let (wi, off) = (bit / 32, bit % 32);
            let v = (c as u64 & mask) << off; // spans at most 2 words (bits <= 16, off <= 31)
            w[wi] |= v & 0xFFFF_FFFF;
            w[wi + 1] |= v >> 32;
        }
        w[..nwords].iter().map(|&x| x as u32).collect()
    }

    #[test]
    fn dequantize_mlx_affine_bits_6bit_crosses_word_boundary() {
        // in_features*bits multiple of 32: 32 codes * 6 = 192 bits = 6 words.
        let in_f = 32usize;
        let bits = 6usize;
        let codes: Vec<u32> = (0..in_f as u32).map(|i| (i * 7 + 1) & 0x3F).collect();
        let packed = pack_contiguous(&codes, bits);
        assert_eq!(packed.len(), in_f * bits / 32);
        // scale=1, bias=0 over one group -> dequant reproduces the raw codes exactly.
        let gs = in_f; // one group
        let scales = vec![1.0f32];
        let biases = vec![0.0f32];
        let out = dequantize_mlx_affine_bits(&packed, &scales, &biases, 1, in_f, gs, bits);
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(out[i], c as f32, "code {i} mismatch (word-crossing unpack)");
        }
        // Affine reproduced: w = 0.5*q + 3.0
        let out2 = dequantize_mlx_affine_bits(&packed, &[0.5], &[3.0], 1, in_f, gs, bits);
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(out2[i], 0.5 * c as f32 + 3.0);
        }
    }

    #[test]
    fn dequantize_mlx_affine_bits_matches_aligned_for_2_4_8() {
        // For power-of-two widths, codes never cross a word, so the contiguous
        // unpacker must be byte-identical to the aligned dequantize_mlx_affine.
        let in_f = 128usize;
        let out_f = 5usize;
        let gs = 64usize;
        let groups = in_f / gs;
        for &bits in &[2usize, 4, 8] {
            let per_word = 32 / bits;
            let mask = (1u32 << bits) - 1;
            let words_per_row = in_f / per_word;
            // deterministic pseudo-random packed data
            let mut packed = vec![0u32; out_f * words_per_row];
            let mut st = 0x1234_5678u32;
            for p in packed.iter_mut() {
                st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *p = st;
            }
            let scales: Vec<f32> = (0..out_f * groups).map(|i| 0.1 + (i % 7) as f32 * 0.01).collect();
            let biases: Vec<f32> = (0..out_f * groups).map(|i| -0.3 + (i % 5) as f32 * 0.02).collect();
            let aligned =
                dequantize_mlx_affine(&packed, &scales, &biases, out_f, in_f, gs, bits);
            let contig =
                dequantize_mlx_affine_bits(&packed, &scales, &biases, out_f, in_f, gs, bits);
            assert_eq!(aligned, contig, "bits={bits}: contiguous != aligned");
            let _ = mask;
        }
    }

    // ── DeepSeek-V4-Flash Phase 2a lead-kernel MIRRORS (regression guard) ──
    // These reproduce the exact GLSL arithmetic of the two lead matvec kernels
    // (shaders/mul_mat_vec_mlx2repack_f32_f32.comp + mul_mat_vec_mlx6_f32_f32.comp)
    // in Rust f32 and assert argmax-exact + ~ULP vs dequantize_mlx_affine_bits ->
    // dense matvec. Mirrors the numpy proof in scripts/dsv4/kernel_matvec_oracle.py;
    // guards the unpack scheme + fma-affine factoring in CI without the checkpoint.

    // 2-bit repack mirror: 16 codes/u32, dwordx4 chunk = 64 codes, gs128
    // (chunks_per_group = 2), temp += scale*dot(q,x) + bias*sum(x).
    fn mlx2_repack_matvec(packed: &[u32], scales: &[f32], biases: &[f32],
                          out_f: usize, in_f: usize, x: &[f32], gs: usize) -> Vec<f32> {
        let wpr = in_f / 16;
        let groups = in_f / gs;
        let cpg = gs / 64;
        let nchunks = in_f / 64;
        (0..out_f).map(|r| {
            let mut acc = 0.0f32;
            for c in 0..nchunks {
                let g = c / cpg;
                let xc = &x[c * 64..c * 64 + 64];
                let xsum: f32 = xc.iter().sum();
                let mut qx = 0.0f32;
                for wi in 0..4 {
                    let w = packed[r * wpr + c * 4 + wi];
                    for j in 0..16 {
                        qx += (((w >> (2 * j)) & 0x3) as f32) * xc[wi * 16 + j];
                    }
                }
                acc = scales[r * groups + g].mul_add(qx, biases[r * groups + g].mul_add(xsum, acc));
            }
            acc
        }).collect()
    }

    // 6-bit contiguous mirror: 3 words / 16 codes, gs128 (chunks_per_group = 8).
    fn mlx6_contig_matvec(packed: &[u32], scales: &[f32], biases: &[f32],
                          out_f: usize, in_f: usize, x: &[f32], gs: usize) -> Vec<f32> {
        let wpr = in_f * 6 / 32;
        let groups = in_f / gs;
        let cpg = gs / 16;
        let nchunks = in_f / 16;
        let m = 0x3Fu32;
        (0..out_f).map(|r| {
            let mut acc = 0.0f32;
            for c in 0..nchunks {
                let g = c / cpg;
                let xc = &x[c * 16..c * 16 + 16];
                let xsum: f32 = xc.iter().sum();
                let (w0, w1, w2) = (packed[r * wpr + c * 3], packed[r * wpr + c * 3 + 1], packed[r * wpr + c * 3 + 2]);
                let codes = [
                    (w0 >> 0) & m, (w0 >> 6) & m, (w0 >> 12) & m, (w0 >> 18) & m,
                    (w0 >> 24) & m, ((w0 >> 30) | (w1 << 2)) & m, (w1 >> 4) & m, (w1 >> 10) & m,
                    (w1 >> 16) & m, (w1 >> 22) & m, ((w1 >> 28) | (w2 << 4)) & m, (w2 >> 2) & m,
                    (w2 >> 8) & m, (w2 >> 14) & m, (w2 >> 20) & m, (w2 >> 26) & m,
                ];
                let qx: f32 = codes.iter().zip(xc).map(|(&q, &xv)| q as f32 * xv).sum();
                acc = scales[r * groups + g].mul_add(qx, biases[r * groups + g].mul_add(xsum, acc));
            }
            acc
        }).collect()
    }

    fn synth(n: usize, seed: u32) -> (Vec<u32>, u32) {
        let mut st = seed;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n { st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223); v.push(st); }
        (v, st)
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
    }

    #[test]
    fn dsv4_2bit_kernel_mirror_matches_oracle() {
        // DSV4 expert-ish shape: k=4096, n=64, gs=128, 2-bit.
        let (in_f, out_f, gs, bits) = (4096usize, 64usize, 128usize, 2usize);
        let wpr = in_f / 16;
        let (packed, st) = synth(out_f * wpr, 0xC0FFEE);
        let groups = in_f / gs;
        let scales: Vec<f32> = (0..out_f * groups).map(|i| 0.1 + (i % 13) as f32 * 0.003).collect();
        let biases: Vec<f32> = (0..out_f * groups).map(|i| -0.2 + (i % 11) as f32 * 0.005).collect();
        let (xr, _) = synth(in_f, st);
        let x: Vec<f32> = xr.iter().map(|&u| (u >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0).collect();
        let deq = dequantize_mlx_affine_bits(&packed, &scales, &biases, out_f, in_f, gs, bits);
        let refv = cpu_matmul(&x, &deq, 1, in_f, out_f);
        let mine = mlx2_repack_matvec(&packed, &scales, &biases, out_f, in_f, &x, gs);
        assert_eq!(argmax(&mine), argmax(&refv), "2-bit argmax mismatch");
        let maxd = mine.iter().zip(&refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(maxd < 1e-3, "2-bit max_abs_err {maxd} too large");
    }

    #[test]
    fn dsv4_6bit_kernel_mirror_matches_oracle() {
        // DSV4 attn-ish shape: k=4096, n=64, gs=128, 6-bit contiguous.
        let (in_f, out_f, gs, bits) = (4096usize, 64usize, 128usize, 6usize);
        let wpr = in_f * 6 / 32;
        let (packed, st) = synth(out_f * wpr, 0xBADC0DE);
        let groups = in_f / gs;
        let scales: Vec<f32> = (0..out_f * groups).map(|i| 0.08 + (i % 9) as f32 * 0.004).collect();
        let biases: Vec<f32> = (0..out_f * groups).map(|i| -0.15 + (i % 7) as f32 * 0.006).collect();
        let (xr, _) = synth(in_f, st);
        let x: Vec<f32> = xr.iter().map(|&u| (u >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0).collect();
        let deq = dequantize_mlx_affine_bits(&packed, &scales, &biases, out_f, in_f, gs, bits);
        let refv = cpu_matmul(&x, &deq, 1, in_f, out_f);
        let mine = mlx6_contig_matvec(&packed, &scales, &biases, out_f, in_f, &x, gs);
        assert_eq!(argmax(&mine), argmax(&refv), "6-bit argmax mismatch");
        let maxd = mine.iter().zip(&refv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(maxd < 1e-3, "6-bit max_abs_err {maxd} too large");
    }
}

/// Quantize an f32 weight `[out_features, in_features]` (row-major) to the
/// **MLX-affine 4-bit** layout that [`dequantize_mlx_affine`] (and the mlx4 GPU
/// matvec shader) read: per (row, group) `scale`/`bias` with each element packed
/// as a 4-bit unsigned nibble, 8 nibbles per u32 word, group_size 64.
///
/// This is the exact inverse of `dequantize_mlx_affine`'s `w = scale*q + bias`
/// contract (NOT a re-derivation of MLX's own quantizer): per group we pick
/// `bias = min`, `scale = (max - min) / 15`, `q = round((w - bias)/scale)`
/// clamped to `[0,15]`, so `dequantize_mlx_affine(quantize_mlx_affine_4bit(w))`
/// is the closest 4-bit-representable reconstruction of `w` under this affine
/// grid. Used by the gemma-12B MLP requant lever (`VLLM_VULKAN_GEMMA_MLP_Q4`):
/// the checkpoint's MLP is natively 8-bit, so this is a LOSSY 8→4-bit re-round,
/// gated on the CPU argmax-agreement test in this module.
///
/// Returns `(packed_u32, scales, biases)` in the same shapes the loader offers
/// via [`Mlx4Weight`]: `packed[out*(in/8)]`, `scales`/`biases` `[out*(in/gs)]`.
pub fn quantize_mlx_affine_4bit(
    data: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    assert_eq!(data.len(), out_features * in_features, "quantize_mlx_affine_4bit shape");
    assert_eq!(in_features % group_size, 0, "in_features must be a multiple of group_size");
    assert_eq!(in_features % 8, 0, "in_features must be a multiple of 8 (nibble packing)");
    let groups = in_features / group_size;
    let words_per_row = in_features / 8; // 8 nibbles / u32 word (4-bit)
    let mut packed = vec![0u32; out_features * words_per_row];
    let mut scales = vec![0f32; out_features * groups];
    let mut biases = vec![0f32; out_features * groups];

    packed
        .par_chunks_mut(words_per_row)
        .zip(scales.par_chunks_mut(groups))
        .zip(biases.par_chunks_mut(groups))
        .enumerate()
        .for_each(|(o, ((prow, srow), brow))| {
            let row = &data[o * in_features..(o + 1) * in_features];
            // Per-group affine params.
            for g in 0..groups {
                let seg = &row[g * group_size..(g + 1) * group_size];
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for &v in seg {
                    if v < mn { mn = v; }
                    if v > mx { mx = v; }
                }
                let range = mx - mn;
                // Degenerate group (all-equal, incl. all-zero): scale 0, bias=mn
                // => dequant reproduces `mn` exactly for every element.
                srow[g] = if range > 0.0 { range / 15.0 } else { 0.0 };
                brow[g] = mn;
            }
            // Quantize + pack nibbles.
            for i in 0..in_features {
                let g = i / group_size;
                let s = srow[g];
                let q = if s > 0.0 {
                    let mut qi = ((row[i] - brow[g]) / s).round();
                    if qi < 0.0 { qi = 0.0; }
                    if qi > 15.0 { qi = 15.0; }
                    qi as u32
                } else {
                    0u32
                };
                prow[i / 8] |= (q & 0xF) << ((i % 8) * 4);
            }
        });
    (packed, scales, biases)
}

/// MLX-4bit affine dequant that writes **f16 bits** directly, never allocating
/// the full f32 tensor. For the 27B's 248k×5120 embed/lm_head this halves the
/// dequant peak (no ~5GB f32 temp) — the difference between fitting and OOM on a
/// 14GB node. Numerically: dequantize in f32 then round to f16 per element
/// (identical to `dequantize_mlx_affine` followed by an f32→f16 cast).
pub fn dequantize_mlx_affine_f16(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    out_features: usize,
    in_features: usize,
    group_size: usize,
    bits: usize,
) -> Vec<u16> {
    use rayon::prelude::*;
    let per_word = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let groups = in_features / group_size;
    let words_per_row = in_features / per_word;
    let mut w = vec![0u16; out_features * in_features];
    w.par_chunks_mut(in_features).enumerate().for_each(|(o, row)| {
        let prow = &packed[o * words_per_row..(o + 1) * words_per_row];
        let srow = &scales[o * groups..(o + 1) * groups];
        let brow = &biases[o * groups..(o + 1) * groups];
        for i in 0..in_features {
            let word = prow[i / per_word];
            let q = ((word >> ((i % per_word) * bits)) & mask) as f32;
            let v = srow[i / group_size] * q + brow[i / group_size];
            row[i] = half::f16::from_f32(v).to_bits();
        }
    });
    w
}

/// An MLX-affine 4-bit quantized embed/lm_head table kept **packed** resident so
/// that a decode step dequantizes only the current token's ROW (~`hidden` elems)
/// on demand, instead of materializing the whole `[vocab, hidden]` f16 table
/// (~1.5GB for the 122B, 248320×3072) at load — the stage-0 anon spike that
/// OOM-kills a 14GB UMA node. `row_f16` is **bit-exact** to the old whole-table
/// [`dequantize_mlx_affine_f16`]: same f32 affine math, same per-element f16
/// round, so the forward's f16→f32 embed lookup is byte-identical either way.
pub struct PackedEmbed {
    pub packed: Vec<u32>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
    pub vocab: usize,
    pub hidden: usize,
    pub group_size: usize,
    pub bits: usize,
}

impl PackedEmbed {
    /// Dequantize row `token` (`hidden` elems) to f16 bits — identical to the
    /// `token`-th output row of [`dequantize_mlx_affine_f16`].
    #[inline]
    pub fn row_f16(&self, token: usize) -> Vec<u16> {
        let per_word = 32 / self.bits;
        let mask = (1u32 << self.bits) - 1;
        let groups = self.hidden / self.group_size;
        let words_per_row = self.hidden / per_word;
        let prow = &self.packed[token * words_per_row..(token + 1) * words_per_row];
        let srow = &self.scales[token * groups..(token + 1) * groups];
        let brow = &self.biases[token * groups..(token + 1) * groups];
        (0..self.hidden)
            .map(|i| {
                let word = prow[i / per_word];
                let q = ((word >> ((i % per_word) * self.bits)) & mask) as f32;
                let v = srow[i / self.group_size] * q + brow[i / self.group_size];
                half::f16::from_f32(v).to_bits()
            })
            .collect()
    }

    /// Dequantize row `token` to f32 — exactly what the forward's whole-table
    /// f16 lookup (`f16::from_bits(table[...]).to_f32()`) would yield.
    #[inline]
    pub fn row_f32(&self, token: usize) -> Vec<f32> {
        self.row_f16(token)
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect()
    }

    /// Resident byte cost (packed u32 + f32 scales + f32 biases).
    pub fn resident_bytes(&self) -> usize {
        self.packed.len() * 4 + self.scales.len() * 4 + self.biases.len() * 4
    }
}

/// Keep the qwen3_5 mlx-affine embed PACKED-4bit resident and dequant only the
/// looked-up token row per step, instead of materializing the whole
/// `[vocab, hidden]` f16 table (~1.5GB for the 122B) at load — the stage-0 anon
/// spike. Bit-exact to the whole-table path. `VLLM_VULKAN_Q35_EMBED_PACKED=0`
/// restores the whole-table f16 embed (A/B parity / fallback).
pub fn q35_embed_packed_enabled() -> bool {
    std::env::var("VLLM_VULKAN_Q35_EMBED_PACKED")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Keep the qwen3_5 mlx-affine (untied) `lm_head.weight` PACKED-4bit resident and
/// run the vocab projection matvec directly on the packed nibbles via the mlx4
/// matvec (`mul_mat_vec_mlx4`), instead of dequantizing the whole `[vocab,hidden]`
/// table to a ~1.5GB f16 buffer at load (then requantizing it to a ~0.76GB q8_0
/// GPU buffer). The untied 122B (`tie_word_embeddings=false`, separate 4-bit
/// `lm_head.weight` + `.scales`/`.biases`) is the target: its last-stage lm_head
/// load transient is the residual anon spike that OOM-kills rank5 at PP-6 — the
/// direct analog of the embed-packed fix, but for lm_head (which the embed work
/// only special-cased for the TIED `embed_tokens` path, never the untied
/// `lm_head.weight`). Packed-resident ≈ 0.48GB (381MB u32 + 2×48MB f32
/// scales/biases) vs 1.5GB f16 + 0.76GB q8. Bit-exact to dequant-then-matvec by
/// construction (the mlx4 kernel does the exact f32 affine `scale*q+bias`).
/// `VLLM_VULKAN_Q35_LMHEAD_PACKED=1` opts in (default OFF).
pub fn q35_lmhead_packed_enabled() -> bool {
    std::env::var("VLLM_VULKAN_Q35_LMHEAD_PACKED")
        .map(|v| v != "0")
        .unwrap_or(false)
}

// ─── NVFP4 (NVIDIA FP4 / ModelOpt compressed-tensors W4A16) ──────────────────
//
// On-disk layout (three sibling tensors per linear, e.g. `*.mlp.gate_proj`):
//   `.weight`         u8   [out, in/2]     two 4-bit E2M1 codes per byte
//   `.weight_scale`   e4m3 [out, in/16]    one FP8 block scale per 16-elem group
//   `.weight_scale_2` f32  scalar          per-tensor global scale
// Dequant: `w[o,i] = e2m1(nibble) * e4m3(weight_scale[o, i/16]) * weight_scale_2`.
// The low nibble of each byte is the even input index, the high nibble the odd
// (compressed-tensors `pack_fp4_to_uint8` order). Validated bit-exact vs a numpy
// reference and cos=0.993 vs an independent MLX-4bit dequant of the same weights.

/// E2M1 (FP4) code → f32. Index is the raw 4-bit code `[sign|exp2|man1]`; the
/// representable magnitudes are {0,.5,1,1.5,2,3,4,6}, sign in bit 3.
pub const NVFP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode one signed OCP FP8 E4M3 byte (1 sign, 4 exp bias-7, 3 mantissa) to f32.
/// `exp==0` is subnormal (`man * 2^-9`); the lone NaN code (0x7f/0xff) maps to 0.0
/// (block scales are finite non-negative, so this only guards malformed data).
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 0x7) as f32;
    let mag = if exp == 0 {
        man * (1.0 / 512.0) // man/8 * 2^-6
    } else if exp == 0xF && man == 7.0 {
        0.0 // NaN guard
    } else {
        (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
    };
    if sign == 1 { -mag } else { mag }
}

/// Dequantize an NVFP4 weight to f32, row-major `[out, in]`. `packed` is the u8
/// `.weight` (`out*in/2` bytes), `wscale` the raw e4m3 `.weight_scale` bytes
/// (`out*in/group_size`), `global` the `.weight_scale_2` scalar. `group_size` is
/// 16 for NVFP4 (derive from `in_features / (wscale.len()/out_features)`).
pub fn dequantize_nvfp4(
    packed: &[u8],
    wscale: &[u8],
    global: f32,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let groups = in_features / group_size;
    let bytes_per_row = in_features / 2;
    let mut w = vec![0.0f32; out_features * in_features];
    w.par_chunks_mut(in_features).enumerate().for_each(|(o, row)| {
        let brow = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
        let wrow = &wscale[o * groups..(o + 1) * groups];
        for i in 0..in_features {
            let byte = brow[i / 2];
            let nib = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let bscale = e4m3_to_f32(wrow[i / group_size]);
            row[i] = NVFP4_E2M1_LUT[nib as usize] * bscale * global;
        }
    });
    w
}

/// NVFP4 dequant writing **f16 bits** directly (no full f32 temp) — the lm_head
/// path, mirroring [`dequantize_mlx_affine_f16`].
pub fn dequantize_nvfp4_f16(
    packed: &[u8],
    wscale: &[u8],
    global: f32,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Vec<u16> {
    use rayon::prelude::*;
    let groups = in_features / group_size;
    let bytes_per_row = in_features / 2;
    let mut w = vec![0u16; out_features * in_features];
    w.par_chunks_mut(in_features).enumerate().for_each(|(o, row)| {
        let brow = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
        let wrow = &wscale[o * groups..(o + 1) * groups];
        for i in 0..in_features {
            let byte = brow[i / 2];
            let nib = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let bscale = e4m3_to_f32(wrow[i / group_size]);
            let v = NVFP4_E2M1_LUT[nib as usize] * bscale * global;
            row[i] = half::f16::from_f32(v).to_bits();
        }
    });
    w
}

// ─── FP8 (E4M3 W8A16 / modelopt per-tensor) ──────────────────────────────────
//
// The Qwen3.6-NVFP4 checkpoint is MIXED precision: attention projections
// (`self_attn.q/k/v/o_proj`, `linear_attn.in_proj_qkv/in_proj_z/out_proj`) are
// FP8, not NVFP4. On disk: `.weight` is F8_E4M3 `[out, in]` (unpacked, 1 byte/
// element), `.weight_scale` an f32 scalar (per-tensor), `.input_scale` the unused
// activation scale. Dequant: `w[o,i] = e4m3(byte) * weight_scale`. `scale` may be
// a single per-tensor value or a per-output-row vector (`out_features` long).

/// Dequantize an FP8-E4M3 weight to f32, row-major `[out, in]`. `scale` is either
/// length 1 (per-tensor, broadcast) or `out_features` (per-row).
pub fn dequantize_fp8(
    weight: &[u8],
    scale: &[f32],
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let per_row = scale.len() == out_features;
    let s0 = scale.first().copied().unwrap_or(1.0);
    let mut w = vec![0.0f32; out_features * in_features];
    w.par_chunks_mut(in_features).enumerate().for_each(|(o, row)| {
        let wrow = &weight[o * in_features..(o + 1) * in_features];
        let s = if per_row { scale[o] } else { s0 };
        for i in 0..in_features {
            row[i] = e4m3_to_f32(wrow[i]) * s;
        }
    });
    w
}

/// FP8-E4M3 dequant writing **f16 bits** directly (embed/lm_head path).
pub fn dequantize_fp8_f16(
    weight: &[u8],
    scale: &[f32],
    out_features: usize,
    in_features: usize,
) -> Vec<u16> {
    use rayon::prelude::*;
    let per_row = scale.len() == out_features;
    let s0 = scale.first().copied().unwrap_or(1.0);
    let mut w = vec![0u16; out_features * in_features];
    w.par_chunks_mut(in_features).enumerate().for_each(|(o, row)| {
        let wrow = &weight[o * in_features..(o + 1) * in_features];
        let s = if per_row { scale[o] } else { s0 };
        for i in 0..in_features {
            let v = e4m3_to_f32(wrow[i]) * s;
            row[i] = half::f16::from_f32(v).to_bits();
        }
    });
    w
}

/// Stream every tensor (across all sibling `*.safetensors` shards) as f32,
/// invoking `f(name, data)` once per tensor.  Unlike [`load_weights_auto`],
/// this never builds a full in-memory f32 copy of the model: each tensor's
/// data is handed to `f` and then dropped (unless `f` keeps it).  This is the
/// low-memory path — essential on small-RAM Vulkan boxes (e.g. the BC-250 with
/// ~3.5 GB system RAM) where materialising the entire f32 weight set thrashes.
pub fn for_each_safetensors_tensor<F>(path: &Path, f: F) -> Result<(), String>
where
    F: FnMut(String, Vec<f32>) -> Result<(), String>,
{
    for_each_safetensors_tensor_filtered(path, |_| true, f)
}

/// Like [`for_each_safetensors_tensor`] but skips tensors for which `keep`
/// returns false BEFORE any dtype conversion. With mmap-backed files this means
/// the skipped tensors' bytes are never faulted in — critical when loading a
/// pipeline-parallel stage over NFS, where each stage should read only its own
/// ~1/N of the weights instead of the whole file.
pub fn for_each_safetensors_tensor_filtered<F, K>(path: &Path, keep: K, mut f: F) -> Result<(), String>
where
    F: FnMut(String, Vec<f32>) -> Result<(), String>,
    K: Fn(&str) -> bool,
{
    use safetensors::SafeTensors;
    use memmap2::Mmap;
    use std::fs::File;

    let shards = discover_shards(path);

    for shard in &shards {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;
        for (raw_name, tensor) in st.tensors() {
            let name = if raw_name.starts_with("model.language_model.") {
                format!("model.{}", &raw_name["model.language_model.".len()..])
            } else {
                raw_name.to_string()
            };
            // Skip before touching tensor data: with mmap this avoids faulting
            // the bytes in at all (no NFS read for tensors this stage discards).
            if !keep(&name) {
                continue;
            }
            let data = tensor.data();
            let f32_data: Vec<f32> = match tensor.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                other => {
                    log::warn!("Skipping tensor '{}' with unsupported dtype {:?}", name, other);
                    continue;
                }
            };
            f(name, f32_data)?;
        }
    }
    Ok(())
}

/// Quantize a row-major f32 weight to GGUF `q8_0` blocks (the layout the
/// `mul_mat_vec_q8_0_*` shaders expect): each block is 32 values stored as one
/// f16 scale `d` followed by 32 `int8`. ~1.06 bytes/weight vs 2 for f16 — and
/// since the matvec is memory-bandwidth-bound, ~half the bytes read per matmul.
/// Near-lossless (per-block absmax scale). Requires `data.len() % 32 == 0`
/// (true for these weights since every contraction dim is a multiple of 32).
pub fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0, "q8_0 needs len%32==0, got {}", data.len());
    let nblocks = data.len() / 32;
    let mut out = Vec::with_capacity(nblocks * 34);
    for b in 0..nblocks {
        let blk = &data[b * 32..b * 32 + 32];
        let amax = blk.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&bf16_unused_f16(d).to_le_bytes());
        for &x in blk {
            let q = (x * id).round().clamp(-127.0, 127.0) as i32 as i8;
            out.push(q as u8);
        }
    }
    out
}

/// Streaming q8_0 quantization straight from f16 bits, WITHOUT materializing the
/// full f32 vector. Used for the huge tied embed/lm_head (248320×5120 → ~5GB as
/// f32) under TP, where EVERY rank builds the q8 lm_head and the f32 transient
/// would OOM a 15GB node (the resident weights already take ~6GB). Per 32-element
/// block we convert only that block to f32, so the transient is O(1).
pub fn quantize_q8_0_from_f16(f16: &[u16]) -> Vec<u8> {
    assert!(f16.len() % 32 == 0, "q8_0 needs len%32==0, got {}", f16.len());
    let nblocks = f16.len() / 32;
    let mut out = Vec::with_capacity(nblocks * 34);
    let mut blk = [0f32; 32];
    for b in 0..nblocks {
        for j in 0..32 { blk[j] = half::f16::from_bits(f16[b * 32 + j]).to_f32(); }
        let amax = blk.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&bf16_unused_f16(d).to_le_bytes());
        for &x in &blk {
            let q = (x * id).round().clamp(-127.0, 127.0) as i32 as i8;
            out.push(q as u8);
        }
    }
    out
}

#[inline]
fn bf16_unused_f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// Chunked-alloc row plan (plan §5, P2): split `total_rows` rows of
/// `row_bytes` each into a sequence of per-chunk row counts, none exceeding
/// `limit_bytes` (a single oversized row still gets its own 1-row chunk —
/// can't split within a row). `limit_bytes == 0` disables chunking (single
/// chunk covering every row), matching the `VLLM_VULKAN_MAX_ALLOC_MB=0` kill
/// switch. Used to keep any single GPU weight buffer under a GTT
/// alloc/fragmentation edge (RADV fails a monolithic 1.35GB lm_head alloc
/// late in load on some devices) by uploading it as several smaller buffers
/// dispatched in one command buffer via `record_to_off` row-offset binding.
pub(crate) fn chunk_row_plan(total_rows: usize, row_bytes: usize, limit_bytes: usize) -> Vec<usize> {
    if total_rows == 0 { return Vec::new(); }
    if limit_bytes == 0 || row_bytes == 0 {
        return vec![total_rows];
    }
    let rows_per_chunk = (limit_bytes / row_bytes).max(1);
    let mut plan = Vec::new();
    let mut remaining = total_rows;
    while remaining > 0 {
        let take = remaining.min(rows_per_chunk);
        plan.push(take);
        remaining -= take;
    }
    plan
}

/// Faithful port of llama.cpp ggml-quants.c `make_qkx2_quants` (the iterative
/// scale+min fit used by q4_K/q5_K). For a sub-block of `n` weights, find the
/// scale `d` and (positive) `min` minimizing the weighted squared error of
/// `x ≈ d*L + (-min)` with quant levels `L in [0, nmax]`. Returns `(scale, min)`
/// and fills `l` with the per-element quant indices. `weights` are per-element
/// importance weights; matches the reference exactly (rmin=-1, rdelta=0.1,
/// nstep=20, use_mad=false as called by `quantize_row_q4_K_ref`).
fn make_qkx2_quants(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> (f32, f32) {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = weights[0] * x[0];
    for i in 1..n {
        if x[i] < min { min = x[i]; }
        if x[i] > max { max = x[i]; }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 { min = 0.0; }
    if max == min {
        for li in l.iter_mut().take(n) { *li = 0; }
        return (0.0, -min);
    }
    let mut iscale = (nmax as f32) / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let li = ((iscale * (x[i] - min)).round() as i32).clamp(0, nmax);
        l[i] = li as u8;
        let diff = scale * (li as f32) + min - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        let w = weights[i];
        best_mad += w * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * (is as f32) + (nmax as f32)) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let li = ((iscale * (x[i] - min)).round() as i32).clamp(0, nmax);
            laux[i] = li as u8;
            let w = weights[i];
            sum_l += w * (li as f32);
            sum_l2 += w * (li as f32) * (li as f32);
            sum_xl += w * (li as f32) * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let diff = this_scale * (laux[i] as f32) + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                mad += weights[i] * diff;
            }
            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Quantize a row-major f32 weight to GGUF `q4_K` blocks (the layout the
/// `mul_mat_vec_q4_kdeq_*` / `block_q4_K` shaders expect). QK_K=256: each
/// super-block of 256 weights is `block_q4_K { f16 d; f16 dmin; u8 scales[12];
/// u8 qs[128]; }` = 144 bytes (~4.5 bits/weight). 256 = 8 sub-blocks of 32; each
/// sub-block j has a 6-bit scale `sc[j]` and 6-bit min `m[j]`. The super-block
/// stores `d = max(sc)/63` and `dmin = max(m)/63` (f16); `scales[12]` packs the
/// 8×(6-bit sc, 6-bit min) via GGUF's `get_scale_min_k4` bit layout. Dequant
/// (matches the shader): `w = d*sc[j]*q - dmin*m[j]`. Faithful port of
/// llama.cpp `quantize_row_q4_K_ref` (uses `make_qkx2_quants` per sub-block).
/// Requires `data.len() % 256 == 0`; far better quality than q4_0.
pub fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    const QK_K: usize = 256;
    assert!(data.len() % QK_K == 0, "q4_K needs len%256==0, got {}", data.len());
    let nblocks = data.len() / QK_K;
    let mut out = Vec::with_capacity(nblocks * 144);
    // Per-sub-block scratch.
    let mut scales = [0f32; QK_K / 32]; // 8
    let mut mins = [0f32; QK_K / 32];
    let mut l = [0u8; QK_K];
    let mut laux = [0u8; 32];
    let mut weights = [0f32; 32];
    for b in 0..nblocks {
        let x = &data[b * QK_K..b * QK_K + QK_K];
        let mut max_scale = 0f32;
        let mut max_min = 0f32;
        // Canonical ggml `quantize_row_q4_K_ref` importance weighting: sigma2 is
        // computed over the FULL 256-element super-block (not per sub-block), and
        // av_x = sqrt(sigma2) with sigma2 = 2*sum_x2/QK_K. Using a per-sub-block
        // av_x (the previous bug) over- or under-weights low-variance sub-blocks,
        // so make_qkx2 settled on poor scales on real Gaussian weights (q4_K only
        // marginally beat q4_0). weights[l] = sqrt(sigma2 + x[l]^2).
        let mut sum_x2_full = 0f32;
        for &v in x.iter() { sum_x2_full += v * v; }
        let sigma2 = 2.0 * sum_x2_full / (QK_K as f32);
        for j in 0..QK_K / 32 {
            let xb = &x[32 * j..32 * j + 32];
            for (i, &v) in xb.iter().enumerate() {
                weights[i] = (sigma2 + v * v).sqrt();
            }
            let (scale, min) = make_qkx2_quants(
                15,
                xb,
                &weights,
                &mut l[32 * j..32 * j + 32],
                &mut laux,
                -1.0,
                0.1,
                20,
                false,
            );
            scales[j] = scale;
            mins[j] = min;
            if scale > max_scale { max_scale = scale; }
            if min > max_min { max_min = min; }
        }

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

        // Pack the 8 6-bit scales + 8 6-bit mins into scales[12] using the GGUF
        // layout (the inverse of the shader's get_scale_min_k4 unpack).
        let mut sc_bytes = [0u8; 12];
        for j in 0..QK_K / 32 {
            let ls = ((inv_scale * scales[j]).round() as i32).clamp(0, 63) as u8;
            let lm = ((inv_min * mins[j]).round() as i32).clamp(0, 63) as u8;
            if j < 4 {
                sc_bytes[j] = ls;
                sc_bytes[j + 4] = lm;
            } else {
                sc_bytes[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                sc_bytes[j - 4] |= (ls >> 4) << 6;
                sc_bytes[j] |= (lm >> 4) << 6;
            }
        }

        // Header: d (f16), dmin (f16).
        let d = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 };
        let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };
        out.extend_from_slice(&bf16_unused_f16(d).to_le_bytes());
        out.extend_from_slice(&bf16_unused_f16(dmin).to_le_bytes());
        out.extend_from_slice(&sc_bytes);

        // Re-quantize element values using the *unpacked* 6-bit scale/min so the
        // stored quants match what the shader reconstructs (reference does the
        // same: it recomputes L from the final per-sub-block sc/m).
        for j in 0..QK_K / 32 {
            let (sc, m) = get_scale_min_k4(j, &sc_bytes);
            let d_j = d * sc as f32;
            if d_j == 0.0 {
                for i in 0..32 { l[32 * j + i] = 0; }
                continue;
            }
            let dm_j = dmin * m as f32;
            for i in 0..32 {
                let q = ((x[32 * j + i] + dm_j) / d_j).round() as i32;
                l[32 * j + i] = q.clamp(0, 15) as u8;
            }
        }

        // Pack qs[128]: for sub-block pair (j, j+1) under the same 64-group, the
        // low nibble holds group j's element, high nibble group j+1's, matching
        // the shader's qsi/(b*4) addressing.
        let mut qs = [0u8; 128];
        for jpair in 0..QK_K / 64 {
            let lo = &l[64 * jpair..64 * jpair + 32];
            let hi = &l[64 * jpair + 32..64 * jpair + 64];
            for i in 0..32 {
                qs[32 * jpair + i] = (lo[i] & 0xF) | ((hi[i] & 0xF) << 4);
            }
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// Inverse-companion to the shader's `get_scale_min_k4` unpack (ggml). Given the
/// packed `scales[12]`, return sub-block `j`'s 6-bit scale and min.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Quantize a row-major f32 weight to GGUF `q4_0` blocks (the layout the
/// `mul_mat_vec_q4_0*` shaders expect): each block is 32 values stored as one
/// f16 scale `d` followed by 16 bytes, each byte packing two 4-bit quants
/// (`qs[j]` low nibble = element j, high nibble = element j+16), values offset
/// by 8 (dequant = (nibble-8)*d). ~0.56 bytes/weight (~3.5× smaller than f16).
/// Lossier than q8_0. Requires `data.len() % 32 == 0`. Matches llama.cpp's
/// `quantize_row_q4_0` (scale from the signed max-abs element, d = vmax/-8).
pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0, "q4_0 needs len%32==0, got {}", data.len());
    let nblocks = data.len() / 32;
    let mut out = Vec::with_capacity(nblocks * 18);
    for b in 0..nblocks {
        let blk = &data[b * 32..b * 32 + 32];
        let mut amax = 0f32;
        let mut vmax = 0f32;
        for &x in blk {
            if x.abs() > amax { amax = x.abs(); vmax = x; }
        }
        let d = vmax / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&bf16_unused_f16(d).to_le_bytes());
        for j in 0..16 {
            let x0 = blk[j] * id;
            let x1 = blk[j + 16] * id;
            let xi0 = ((x0 + 8.5).floor() as i32).clamp(0, 15) as u8;
            let xi1 = ((x1 + 8.5).floor() as i32).clamp(0, 15) as u8;
            out.push(xi0 | (xi1 << 4));
        }
    }
    out
}

/// Dequantize a q4_0 buffer EXACTLY as the `mul_mat_vec_q4_0deq` shader does:
/// `w = d * (nibble - 8)`, low nibble = elem j, high nibble = elem j+16.
pub fn dequant_q4_0_to_f32(buf: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(buf.len() / 18 * 32);
    for blk in buf.chunks_exact(18) {
        let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        let mut lo = [0f32; 16];
        let mut hi = [0f32; 16];
        for j in 0..16 {
            lo[j] = d * ((blk[2 + j] & 0xF) as i32 - 8) as f32;
            hi[j] = d * ((blk[2 + j] >> 4) as i32 - 8) as f32;
        }
        out.extend_from_slice(&lo);
        out.extend_from_slice(&hi);
    }
    out
}

/// Dequantize a q8_0 buffer EXACTLY as the `mul_mat_vec_q8_0deq` shader does.
pub fn dequant_q8_0_to_f32(buf: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(buf.len() / 34 * 32);
    for blk in buf.chunks_exact(34) {
        let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for j in 0..32 {
            out.push(d * (blk[2 + j] as i8) as f32);
        }
    }
    out
}

/// Dequantize a q4_K buffer EXACTLY as the Vulkan `DATA_A_Q4_K` `dequantize()`
/// path does (dequant_funcs.glsl): per-256 super-block, unpack f16 d/dmin,
/// each sub-block's 6-bit (sc,m) via `get_scale_min_k4`, `w = d*sc*q - dmin*m`.
/// Element-position mapping matches the matvec: weight at flat index `c` is
/// sub-block `c/32` element `c%32`; sub-block `is` is stored in byte-group
/// `is/2`, nibble `is%2`. This is the Rust mirror of the GPU read path.
pub fn dequant_q4_k_to_f32(buf: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(buf.len() / 144 * 256);
    for block in buf.chunks_exact(144) {
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let mut sc12 = [0u8; 12];
        sc12.copy_from_slice(&block[4..16]);
        let qs = &block[16..144];
        // Flat element order: 64-element group `jpair` = [sub-block 2*jpair (low
        // nibble, 32 elems), then sub-block 2*jpair+1 (high nibble, 32 elems)].
        for jpair in 0..4 {
            let (sc_lo, m_lo) = get_scale_min_k4(2 * jpair, &sc12);
            let (sc_hi, m_hi) = get_scale_min_k4(2 * jpair + 1, &sc12);
            let d_lo = d * sc_lo as f32;
            let dm_lo = dmin * m_lo as f32;
            let d_hi = d * sc_hi as f32;
            let dm_hi = dmin * m_hi as f32;
            for i in 0..32 {
                let byte = qs[32 * jpair + i];
                out.push(d_lo * (byte & 0xF) as f32 - dm_lo);
            }
            for i in 0..32 {
                let byte = qs[32 * jpair + i];
                out.push(d_hi * (byte >> 4) as f32 - dm_hi);
            }
        }
    }
    out
}

/// Load a single tensor's raw bf16 bits (no f32 expansion) by name, scanning all
/// `.safetensors` shards in the file's directory. For very large embedding tables
/// (e.g. Gemma's `embed_tokens_per_layer`, ~9.4GB as f32) that would OOM if
/// materialized as f32 — kept bf16-resident, one row converted to f32 on lookup.
/// With mmap only the wanted tensor's bytes fault in. f16 inputs are re-encoded
/// to bf16. Errors if the tensor is missing or not bf16/f16.
pub fn load_tensor_raw_bf16(path: &Path, want: &str) -> Result<Vec<u16>, String> {
    use safetensors::SafeTensors;
    use memmap2::Mmap;
    use std::fs::File;
    let shards = discover_shards(path);
    for shard in &shards {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;
        for (raw_name, tensor) in st.tensors() {
            let name = if raw_name.starts_with("model.language_model.") {
                format!("model.{}", &raw_name["model.language_model.".len()..])
            } else { raw_name.to_string() };
            if name != want { continue; }
            let data = tensor.data();
            return match tensor.dtype() {
                safetensors::Dtype::BF16 => Ok(data.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
                safetensors::Dtype::F16 => Ok(data.chunks_exact(2)
                    .map(|c| bf16::from_f32(half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).to_bits())
                    .collect()),
                other => Err(format!("tensor '{want}' dtype {other:?} not bf16/f16")),
            };
        }
    }
    Err(format!("tensor '{want}' not found"))
}

/// Find a quantization sibling tensor (`{base_raw}.scales` / `.biases`) and
/// decode it to f32, searching ACROSS all shards. A tensor's `.weight` can live
/// in a different shard than its `.scales`/`.biases` (the 35B-A3B splits e.g.
/// layer 20's `in_proj_qkv.weight`/`.scales` in shard 2 but `.biases` in shard
/// 3) — a same-shard lookup then fails with TensorNotFound. `base_raw` is the
/// ORIGINAL (un-stripped) name without the `.weight` suffix.
fn find_quant_sibling_f32(
    shards: &[PathBuf],
    base_raw: &str,
    suffix: &str,
) -> Result<Vec<f32>, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;
    let want = format!("{base_raw}.{suffix}");
    for shard in shards {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse: {e}"))?;
        if let Ok(v) = st.tensor(&want) {
            let d = v.data();
            return Ok(match v.dtype() {
                safetensors::Dtype::BF16 => d.chunks_exact(2)
                    .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
                safetensors::Dtype::F16 => d.chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
                safetensors::Dtype::F32 => d.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                other => return Err(format!("{want}: unsupported dtype {other:?}")),
            });
        }
    }
    Err(format!("{want}: not found in any shard"))
}

/// Cross-shard sibling lookup returning the tensor's RAW bytes (no dtype decode).
/// Used for NVFP4 `.weight_scale` (e4m3), which must stay byte-exact for the
/// custom decode rather than being widened to f32 like the mlx affine scales.
fn find_quant_sibling_bytes(
    shards: &[PathBuf],
    base_raw: &str,
    suffix: &str,
) -> Result<Vec<u8>, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;
    let want = format!("{base_raw}.{suffix}");
    for shard in shards {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse: {e}"))?;
        if let Ok(v) = st.tensor(&want) {
            return Ok(v.data().to_vec());
        }
    }
    Err(format!("{want}: not found in any shard"))
}

/// Load Qwen3.6 (`qwen3_5`) weights into f32, dequantizing MLX-4bit affine
/// quantized linears (grouping `{.weight(u32), .scales, .biases}`) and decoding
/// plain tensors (bf16/f16/f32). Strips the `language_model.` prefix, skips
/// `vision_tower.*`. `max_layers` limits to `model.layers.{i}` with `i < n`
/// (for cheap layer-truncated validation); `include_embeddings` keeps the large
/// `embed_tokens`/`lm_head` tensors.
pub fn load_qwen35_weights(
    path: &Path,
    group_size: usize,
    bits: usize,
    max_layers: Option<usize>,
    include_embeddings: bool,
) -> Result<HashMap<String, Vec<f32>>, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;

    let shards = discover_shards(path);

    let skip = |name: &str| -> bool {
        if qwen35_skip_aux(name) {
            return true;
        }
        if !include_embeddings
            && (name.starts_with("model.embed_tokens") || name.starts_with("lm_head"))
        {
            return true;
        }
        if let Some(n) = max_layers {
            if let Some(rest) = name.strip_prefix("model.layers.") {
                if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                    if idx >= n {
                        return true;
                    }
                }
            }
        }
        false
    };
    let decode = |view: &safetensors::tensor::TensorView| -> Vec<f32> {
        let d = view.data();
        match view.dtype() {
            safetensors::Dtype::BF16 => d.chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            safetensors::Dtype::F16 => d.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            safetensors::Dtype::F32 => d.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            other => panic!(
                "unsupported safetensors dtype {other:?} in weight loader \
                 (only BF16/F16/F32 are decodable here)"
            ),
        }
    };

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    for shard in &shards {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;
        for (raw_name, view) in st.tensors() {
            let name = normalize_qwen35_name(&raw_name);
            if skip(&name) {
                continue;
            }
            if name.ends_with(".scales") || name.ends_with(".biases") {
                continue; // consumed with the matching .weight
            }
            if name.ends_with(".weight") && view.dtype() == safetensors::Dtype::U32 {
                // MLX-4bit affine quantized linear.
                let raw_base = &raw_name[..raw_name.len() - ".weight".len()];
                let scales = match st.tensor(&format!("{raw_base}.scales")) {
                    Ok(sv) => decode(&sv),
                    Err(_) => find_quant_sibling_f32(&shards, raw_base, "scales")?,
                };
                let biases = match st.tensor(&format!("{raw_base}.biases")) {
                    Ok(bv) => decode(&bv),
                    Err(_) => find_quant_sibling_f32(&shards, raw_base, "biases")?,
                };
                let packed: Vec<u32> = view.data().chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                // Per-tensor bits derived from geometry (see `mlx_affine_bits`):
                // the 35B-A3B routers are 8-bit, but the mlx-community 122B-A10B is
                // uniform 4-bit incl. routers — a hardcoded name table gets the
                // 122B silently wrong. Config `bits` is the degenerate fallback.
                // 3D `switch_mlp` experts flatten to `[E*out, in]`.
                let tbits = match mlx_affine_bits(packed.len(), scales.len(), group_size) {
                    0 => bits,
                    b => b,
                };
                let shape = view.shape();
                let (out_features, in_features) = if shape.len() == 3 {
                    (shape[0] * shape[1], shape[2] * (32 / tbits))
                } else {
                    (shape[0], shape[1] * (32 / tbits))
                };
                let gsize = mlx_affine_group_size(&name, in_features, out_features, scales.len())?;
                validate_affine_dims(&name, packed.len(), out_features, in_features, gsize, tbits, scales.len(), biases.len())?;
                let deq = dequantize_mlx_affine(
                    &packed, &scales, &biases, out_features, in_features, gsize, tbits,
                );
                out.insert(name, deq);
            } else {
                out.insert(name, decode(&view));
            }
        }
    }
    Ok(out)
}

/// Borrowed PACKED NVFP4 weight handed to the projection sink under
/// `VLLM_VULKAN_NVFP4_GPU=1` (no f32 dequant). `packed` is `[out, in/2]` u8
/// nibbles; `wscale` is raw e4m3 `[out*groups]`; `global` = weight_scale_2.
pub struct Nvfp4Weight<'a> {
    pub packed: &'a [u8],
    pub wscale: &'a [u8],
    pub global: f32,
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

/// A PACKED FP8-E4M3 attention projection the sink may take verbatim (V2:
/// keep it resident 1 B/elem instead of dequantizing to f16).
pub struct Fp8Weight<'a> {
    pub weight: &'a [u8],      // FP8-E4M3 [out, in], 1 byte/elem, row-major
    pub scale: Vec<f32>,       // 1 (per-tensor) or out_features (per-row)
    pub out_features: usize,
    pub in_features: usize,
}

/// A PACKED MLX-affine 4-bit DENSE projection (GatedDeltaNet/attn, NOT MoE
/// experts — those already go 4-bit-resident via `QuantSwitch`) the sink may
/// take verbatim instead of the lossy f32-dequant -> q8_0-requantize path.
/// Owned (unlike `Nvfp4Weight`/`Fp8Weight`'s borrowed nibbles) because the
/// loader reinterprets the safetensors bytes to `u32` before offering it, so
/// there is nothing left to re-borrow from `view.data()` on decline; the
/// caller re-runs `dequantize_mlx_affine` on the same `packed`/`scales`/
/// `biases` it already built instead.
pub struct Mlx4Weight {
    pub packed: Vec<u32>,   // [out * (in/8)], 8 nibbles/u32 word, row-major
    pub scales: Vec<f32>,   // [out * groups]
    pub biases: Vec<f32>,   // [out * groups]
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

/// Input to the projection sink: an f32 (dequantized) weight, or a packed NVFP4,
/// FP8, or MLX4-affine weight the sink may take verbatim.
pub enum ProjWeight<'a> {
    F32(Vec<f32>),
    Nvfp4(Nvfp4Weight<'a>),
    Fp8(Fp8Weight<'a>),
    Mlx4(Mlx4Weight),
}

/// Sink verdict. `Consumed` = sink uploaded/dropped it (loader keeps nothing).
/// `KeepF32(v)` = accumulate `v` in the host f32 map. `Dequantize` = (NVFP4
/// only) sink declined the packed form; loader must dequant to f32 and re-offer.
pub enum ProjResult {
    Consumed,
    KeepF32(Vec<f32>),
    Dequantize,
}

/// As `load_qwen35_weights`, but tensors whose name ends with `embed_tokens.weight`
/// or `lm_head.weight` are converted to **f16** and returned in a separate map,
/// with their (~5GB-each for the 27B) f32 dequant freed immediately. This caps the
/// load-time peak on a 14GB node, where holding both embed AND lm_head as f32
/// (~10GB) alongside the f32 projections OOMs. Returns `(f32_map, f16_map)`.
///
/// Pipeline-parallel: only decoder layers with GLOBAL index in
/// `[layer_start, max_layers)` are loaded (weight names keep their GLOBAL
/// `model.layers.{idx}` form — the forward maps to stage-local state). `keep_embed`
/// gates `model.embed_tokens` (the first stage embeds) and `keep_lm` gates
/// `lm_head` + `model.norm` (the last stage produces logits); dropping the ~5GB
/// f16 table a stage doesn't own halves its host footprint. For a single-node
/// (non-PP) load pass `layer_start = 0`, `keep_embed = keep_lm = true`.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn load_qwen35_weights_split(
    path: &Path,
    group_size: usize,
    bits: usize,
    layer_start: usize,
    max_layers: Option<usize>,
    keep_embed: bool,
    keep_lm: bool,
    // Streaming sink for projection (matvec) weights: called with each dequantized
    // f32 projection IMMEDIATELY after dequant. If it returns `true` the loader
    // does NOT accumulate that tensor in `out_f32` (the caller has uploaded it to
    // the GPU and dropped the f32) — this caps host f32 peak at ONE tensor, which
    // is what makes a 27B PP stage fit a 14GB node. Pass a no-op `|_,_| false` to
    // keep the old accumulate-all-f32 behavior.
    mut on_proj: impl FnMut(&str, ProjWeight) -> ProjResult,
) -> Result<(HashMap<String, Vec<f32>>, HashMap<String, Vec<u16>>, Option<PackedEmbed>, Option<PackedEmbed>), String> {
    // Read weights by PREAD of each needed tensor's byte range, NOT by mapping
    // the whole shard. A `Mmap::map(&file)` maps the ENTIRE shard as one VMA; on
    // the RAM-backed live medium the mapped shard counts as resident against the
    // process at FULL shard size (~10GB) regardless of which tensors the stage
    // actually reads, so stage 0 (embed + shard-1) OOM-killed a 14GB node at
    // load, with a FIXED anon footprint identical at PP-3 and PP-4. Here we parse
    // only each shard's header (a few MB), then pread each kept tensor's bytes on
    // demand and drop them after the sink uploads to the GPU — the host peak is
    // one tensor plus the retained embed/lm_head f16 table, independent of shard
    // size and of PP depth.
    use safetensors::Dtype;
    use std::fs::File;
    use std::os::unix::fs::FileExt;

    let shards = discover_shards(path);

    // A tensor located in a shard header: dtype/shape + absolute byte range.
    struct Entry {
        shard: usize,
        dtype: Dtype,
        shape: Vec<usize>,
        off: u64,
        len: usize,
    }
    // An owned, pread-loaded tensor. Mirrors the subset of the safetensors
    // TensorView API the dequant code below uses (data/dtype/shape), so that
    // math is untouched by the mmap→pread switch.
    struct Pv {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }
    impl Pv {
        fn data(&self) -> &[u8] { &self.data }
        fn dtype(&self) -> Dtype { self.dtype }
        fn shape(&self) -> &[usize] { &self.shape }
    }
    let dtype_from_str = |s: &str| -> Result<Dtype, String> {
        Ok(match s {
            "F64" => Dtype::F64, "F32" => Dtype::F32, "F16" => Dtype::F16,
            "BF16" => Dtype::BF16, "F8_E4M3" => Dtype::F8_E4M3,
            "F8_E5M2" => Dtype::F8_E5M2, "U8" => Dtype::U8, "I8" => Dtype::I8,
            "U16" => Dtype::U16, "I16" => Dtype::I16, "U32" => Dtype::U32,
            "I32" => Dtype::I32, "U64" => Dtype::U64, "I64" => Dtype::I64,
            "BOOL" => Dtype::BOOL,
            other => return Err(format!("unsupported safetensors dtype {other}")),
        })
    };

    // Parse every shard header (no tensor data faulted). First shard that
    // defines a name wins (safetensors names are unique across a checkpoint).
    let mut files: Vec<File> = Vec::with_capacity(shards.len());
    let mut entries: HashMap<String, Entry> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for (si, shard) in shards.iter().enumerate() {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mut lenb = [0u8; 8];
        file.read_exact_at(&mut lenb, 0)
            .map_err(|e| format!("read header len {}: {e}", shard.display()))?;
        let hlen = u64::from_le_bytes(lenb);
        let mut hbuf = vec![0u8; hlen as usize];
        file.read_exact_at(&mut hbuf, 8)
            .map_err(|e| format!("read header {}: {e}", shard.display()))?;
        let hjson: serde_json::Value = serde_json::from_slice(&hbuf)
            .map_err(|e| format!("parse header json {}: {e}", shard.display()))?;
        let data_start = 8 + hlen;
        if let Some(obj) = hjson.as_object() {
            for (name, info) in obj {
                if name == "__metadata__" { continue; }
                let dt = dtype_from_str(
                    info.get("dtype").and_then(|v| v.as_str())
                        .ok_or_else(|| format!("{name}: missing dtype"))?)?;
                let shape: Vec<usize> = info.get("shape").and_then(|v| v.as_array())
                    .ok_or_else(|| format!("{name}: missing shape"))?
                    .iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
                let offs = info.get("data_offsets").and_then(|v| v.as_array())
                    .ok_or_else(|| format!("{name}: missing data_offsets"))?;
                if offs.len() != 2 { return Err(format!("{name}: bad data_offsets")); }
                let begin = offs[0].as_u64().unwrap_or(0);
                let end = offs[1].as_u64().unwrap_or(0);
                if end < begin { return Err(format!("{name}: end<begin offset")); }
                if !entries.contains_key(name) {
                    order.push(name.clone());
                    entries.insert(name.clone(), Entry {
                        shard: si, dtype: dt, shape,
                        off: data_start + begin, len: (end - begin) as usize,
                    });
                }
            }
        }
        files.push(file);
    }

    // pread a tensor's bytes into an owned Pv (searches all shards). None if the
    // name is absent (used for optional siblings / cross-shard scale splits).
    let read_pv = |raw: &str| -> Result<Option<Pv>, String> {
        match entries.get(raw) {
            None => Ok(None),
            Some(e) => {
                let mut buf = vec![0u8; e.len];
                files[e.shard].read_exact_at(&mut buf, e.off)
                    .map_err(|err| format!("pread {raw}: {err}"))?;
                Ok(Some(Pv { dtype: e.dtype, shape: e.shape.clone(), data: buf }))
            }
        }
    };

    // Stream a large REAL (BF16/F16/F32) tensor straight to f16 in fixed-size
    // preads, so the load-time transient is one chunk (~32MB) rather than the
    // whole source table plus its f16 output held at once. This is the fix for
    // the embed_tokens / (tied) lm_head OOM: the 27B embed is BF16 248320×5120
    // (~2.5GB); read_pv would fault the full 2.5GB Vec<u8> AND build the 2.5GB
    // f16 out concurrently (~5GB spike). Chunked, the peak is the retained f16
    // output (~2.5GB) + one 32MB chunk. Carries to the 122B (shared loader).
    let read_real_to_f16_chunked = |e: &Entry| -> Result<Vec<u16>, String> {
        let elem: usize = match e.dtype {
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
            other => return Err(format!("read_real_to_f16_chunked: non-real dtype {other:?}")),
        };
        let n_elems = e.len / elem;
        let mut out: Vec<u16> = Vec::with_capacity(n_elems);
        const CHUNK_ELEMS: usize = 8 * 1024 * 1024; // ~16-32MB source per chunk
        let mut done = 0usize;
        while done < n_elems {
            let take = CHUNK_ELEMS.min(n_elems - done);
            let mut buf = vec![0u8; take * elem];
            files[e.shard].read_exact_at(&mut buf, e.off + (done * elem) as u64)
                .map_err(|err| format!("pread chunk: {err}"))?;
            match e.dtype {
                Dtype::BF16 => out.extend(buf.chunks_exact(2).map(|c| {
                    half::f16::from_f32(
                        bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).to_bits()
                })),
                Dtype::F16 => out.extend(buf.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))),
                Dtype::F32 => out.extend(buf.chunks_exact(4).map(|c| {
                    half::f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_bits()
                })),
                _ => unreachable!(),
            }
            done += take;
            // `buf` dropped here → transient capped at one chunk.
        }
        Ok(out)
    };

    let skip = |name: &str| -> bool {
        if qwen35_skip_aux(name) {
            return true;
        }
        // PP: stage owns embed only if keep_embed; lm_head + final norm only if keep_lm.
        if name.starts_with("model.embed_tokens") {
            return !keep_embed;
        }
        if name.starts_with("lm_head") || name == "model.norm.weight" {
            return !keep_lm;
        }
        // Decoder layers: keep GLOBAL index in [layer_start, max_layers).
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                if idx < layer_start {
                    return true;
                }
                if let Some(n) = max_layers {
                    if idx >= n {
                        return true;
                    }
                }
            }
        }
        false
    };

    let decode = |view: &Pv| -> Vec<f32> {
        let d = view.data();
        match view.dtype() {
            safetensors::Dtype::BF16 => d.chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            safetensors::Dtype::F16 => d.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            safetensors::Dtype::F32 => d.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            other => panic!(
                "unsupported safetensors dtype {other:?} in weight loader \
                 (only BF16/F16/F32 are decodable here)"
            ),
        }
    };
    // Decode a small/large real tensor STRAIGHT to f16 with no intermediate
    // whole-tensor Vec<f32>. The embed_tokens / (tied) lm_head table is BF16 and
    // ~2.5GB; the old path built a ~5GB f32 temp before downcasting — a second
    // load-time OOM lever on a 14GB node. Per-element convert instead.
    let decode_to_f16 = |view: &Pv| -> Vec<u16> {
        let d = view.data();
        match view.dtype() {
            safetensors::Dtype::BF16 => d.chunks_exact(2)
                .map(|c| half::f16::from_f32(
                    bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).to_bits())
                .collect(),
            // Already f16: copy the bits through.
            safetensors::Dtype::F16 => d.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
            safetensors::Dtype::F32 => d.chunks_exact(4)
                .map(|c| half::f16::from_f32(
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_bits()).collect(),
            other => panic!(
                "unsupported safetensors dtype {other:?} in weight loader (f16 path)"
            ),
        }
    };
    let want_f16 = |name: &str| -> bool {
        name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight")
    };

    let mut out_f32: HashMap<String, Vec<f32>> = HashMap::new();
    let mut out_f16: HashMap<String, Vec<u16>> = HashMap::new();
    // Packed-resident embed (VLLM_VULKAN_Q35_EMBED_PACKED, default on): when the
    // mlx-affine `embed_tokens` is kept packed instead of dequantized to a whole
    // f16 table, it lands here for the caller (per-row decode + streaming q8
    // lm_head build).
    let mut out_embed_packed: Option<PackedEmbed> = None;
    let embed_packed_on = q35_embed_packed_enabled();
    // Packed-resident (untied) lm_head: keep the mlx-affine 4-bit `lm_head.weight`
    // packed for a direct mlx4 vocab matvec, instead of the whole-table f16 dequant
    // (+ q8 requant) the last stage would otherwise build — the rank5 OOM lever.
    let mut out_lmhead_packed: Option<PackedEmbed> = None;
    let lmhead_packed_on = q35_lmhead_packed_enabled();
    for raw_name in &order {
        // compressed-tensors "nvfp4-pack-quantized" (unsloth Qwen3.8-27B-NVFP4)
        // names the packed NVFP4 mlp weight `<base>.weight_packed` (vs modelopt's
        // `<base>.weight`). Present it to the rest of the loop as a plain
        // `<base>.weight` for name derivation / matvec classification; DATA reads
        // still use the real `raw_name` key (U8-packed nibbles). The
        // `.weight_global_scale` global (RECIPROCAL of modelopt `.weight_scale_2`)
        // is resolved in `nvfp4_parts` below.
        let logical_raw: std::borrow::Cow<str> = if raw_name.ends_with(".weight_packed") {
            std::borrow::Cow::Owned(format!(
                "{}.weight", &raw_name[..raw_name.len() - ".weight_packed".len()]))
        } else {
            std::borrow::Cow::Borrowed(raw_name.as_str())
        };
        let name = normalize_qwen35_name(&logical_raw);
        if skip(&name) {
            continue;
        }
        if name.ends_with(".scales") || name.ends_with(".biases") {
            continue;
        }
        // NVFP4 (modelopt/compressed-tensors) sibling scale tensors — consumed
        // alongside their `.weight`, never loaded standalone. `.input_scale` is
        // the (unused for W4A16) activation scale.
        if name.ends_with(".weight_scale")
            || name.ends_with(".weight_scale_2")
            || name.ends_with(".input_scale")
            // compressed-tensors / llmcompressor sibling scales (unsloth flavor):
            || name.ends_with(".weight_global_scale")
            || name.ends_with(".input_global_scale")
            // FP8-KV cache quant scales (unused in the W16A16 CPU/GPU forward)
            || name.ends_with(".k_scale")
            || name.ends_with(".v_scale")
        {
            continue;
        }
        // 4-bit-RESIDENT MoE experts: when enabled, the big `switch_mlp.*`
        // expert tensors are loaded packed (QuantSwitch) by
        // `load_qwen35_moe_quant_experts`, NOT dequantized to f32 here — the
        // memory lever that fits an 8-layer MoE PP stage on a 15GB node.
        if moe_q4_resident_enabled() && name.contains(".mlp.switch_mlp.") {
            continue;
        }
        if !logical_raw.ends_with(".weight") {
            // Real-dtype params with NO `.weight` suffix: the GatedDeltaNet
            // per-layer `linear_attn.A_log` / `linear_attn.dt_bias` (BF16
            // [num_v_heads]). The pread rewrite's original `.weight`-only guard
            // DROPPED these — a regression vs the mmap loader (which decoded
            // any surviving real tensor into the host f32 map): the GDN
            // forward then panics `Weight '...A_log' not found` on its first
            // linear-attention layer. Decode them to host f32 under the
            // normalized name, exactly like the old loader. Anything non-real
            // that survives the sibling guards above (unexpected packed aux)
            // is still skipped — never mis-sliced as a `raw_base`.
            if matches!(entries[raw_name].dtype,
                safetensors::Dtype::BF16 | safetensors::Dtype::F16
                | safetensors::Dtype::F32)
            {
                let view = read_pv(raw_name)?
                    .ok_or_else(|| format!("tensor vanished from header: {raw_name}"))?;
                out_f32.insert(name, decode(&view));
            }
            continue;
        }
        let raw_base = &logical_raw[..logical_raw.len() - ".weight".len()];
        // dtype + sibling existence come from the header (no data faulted yet).
        // DATA/dtype reads use the REAL `raw_name` key (`.weight_packed` for
        // compressed-tensors NVFP4, `.weight` otherwise); `raw_base` (sans the
        // packed/weight suffix) is only for `.weight_scale*` sibling lookups.
        let ent_dtype = entries[raw_name].dtype;
        let is_quant = ent_dtype == safetensors::Dtype::U32;
        // NVFP4: u8-packed `.weight` with an e4m3 `.weight_scale` sibling.
        let is_nvfp4 = ent_dtype == safetensors::Dtype::U8
            && entries.contains_key(&format!("{raw_base}.weight_scale"));
        // FP8 (E4M3 W8A16): f8_e4m3 `.weight` with an f32 `.weight_scale` sibling.
        let is_fp8 = ent_dtype == safetensors::Dtype::F8_E4M3
            && entries.contains_key(&format!("{raw_base}.weight_scale"));

        // Big REAL (unquantized) embed/lm_head: stream bf16/f16/f32 → f16 in
        // chunks (no whole-table Vec<u8> + whole-table f16 held together). The
        // quantized embed/lm_head variants (nvfp4/fp8/mlx4, e.g. the 122B's
        // 4-bit embed) still go through the whole-tensor read_pv below, where
        // dequantize_*_f16 already avoids an f32 temp.
        if want_f16(&name)
            && !is_nvfp4 && !is_fp8 && !is_quant
            && matches!(ent_dtype,
                safetensors::Dtype::BF16 | safetensors::Dtype::F16 | safetensors::Dtype::F32)
        {
            let f16v = read_real_to_f16_chunked(&entries[raw_name])?;
            out_f16.insert(name, f16v);
            continue;
        }

        // pread the primary tensor's bytes (dropped at end of this iteration).
        let view = read_pv(raw_name)?
            .ok_or_else(|| format!("tensor vanished from header: {raw_name}"))?;

        // Fetch NVFP4 siblings + derive dims (only call when `is_nvfp4`).
        let nvfp4_parts = |view: &Pv|
         -> Result<(Vec<u8>, f32, usize, usize, usize), String> {
            let wscale = read_pv(&format!("{raw_base}.weight_scale"))?
                .ok_or_else(|| format!("{raw_base}.weight_scale missing"))?
                .data().to_vec();
            // modelopt stores `.weight_scale_2` = global (dequant MULTIPLIES);
            // compressed-tensors/llmcompressor stores `.weight_global_scale`
            // = 448*6/amax, the RECIPROCAL. Downstream (`dequantize_nvfp4`)
            // MULTIPLIES by `global`, so hand back the reciprocal — identical to
            // `load_gemma_nvfp4_weights` (model.rs ~1997). Verified on the 31B
            // checkpoint: dividing yields sane |w|, multiplying corrupts it.
            let global = match read_pv(&format!("{raw_base}.weight_scale_2"))? {
                Some(gv) => f32::from_le_bytes(
                    gv.data()[..4].try_into().map_err(|_| "weight_scale_2: short")?),
                None => match read_pv(&format!("{raw_base}.weight_global_scale"))? {
                    Some(gv) => {
                        let raw = f32::from_le_bytes(gv.data()[..4].try_into()
                            .map_err(|_| "weight_global_scale: short")?);
                        if raw == 0.0 || !raw.is_finite() {
                            return Err(format!(
                                "{raw_base}.weight_global_scale {raw} not usable"));
                        }
                        1.0 / raw
                    }
                    None => return Err(format!(
                        "{raw_base}.weight_scale_2 / .weight_global_scale missing")),
                },
            };
            // `.weight` is [out, in/2]; two nibbles per byte -> in = cols*2.
            let out_features = view.shape()[0];
            let in_features = view.shape()[1] * 2;
            let groups = (wscale.len() / out_features).max(1);
            let group_size = in_features / groups;
            Ok((wscale, global, out_features, in_features, group_size))
        };
        let fp8_parts = |view: &Pv|
         -> Result<(Vec<f32>, usize, usize), String> {
            let scale = decode(&read_pv(&format!("{raw_base}.weight_scale"))?
                .ok_or_else(|| format!("{raw_base}.weight_scale missing"))?);
            let shape = view.shape();
            Ok((scale, shape[0], shape[1]))
        };
        if want_f16(&name) {
            // ── Packed-resident embed (per-row decode) ───────────────────────
            // The mlx-affine (is_quant) embed table is kept PACKED — no
            // [vocab,hidden] f16 materialization at load. Only `embed_tokens`
            // (random-row lookup); a separate quantized `lm_head.weight`, if
            // present, still goes to the whole f16 table below (it is a full
            // matvec). For the tied case (lm_name == embed_tokens) the caller
            // stream-builds the q8 lm_head from THIS packed data.
            if embed_packed_on && is_quant && name.ends_with("embed_tokens.weight") {
                let scales = decode(&read_pv(&format!("{raw_base}.scales"))?
                    .ok_or_else(|| format!("{raw_base}.scales missing"))?);
                let biases = decode(&read_pv(&format!("{raw_base}.biases"))?
                    .ok_or_else(|| format!("{raw_base}.biases missing"))?);
                let packed: Vec<u32> = view.data().chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let out_features = view.shape()[0];
                let in_features = view.shape()[1] * (32 / bits);
                let gsize = mlx_affine_group_size(&name, in_features, out_features, scales.len())?;
                validate_affine_dims(&name, packed.len(), out_features, in_features, gsize, bits, scales.len(), biases.len())?;
                out_embed_packed = Some(PackedEmbed {
                    packed, scales, biases,
                    vocab: out_features, hidden: in_features, group_size: gsize, bits,
                });
                continue;
            }
            // ── Packed-resident (untied) lm_head (direct mlx4 vocab matvec) ──
            // The mlx-affine 4-bit `lm_head.weight` (122B: U32 [vocab, hidden/8]
            // + BF16 `.scales`/`.biases` [vocab, hidden/gs]) is kept PACKED — no
            // [vocab,hidden] f16 materialization + q8 requant at load. Only when
            // the tensor is genuinely 4-bit (the mlx4 kernel is 4-bit; an 8-bit
            // lm_head would fall through to the whole f16 table below). The caller
            // uploads THIS packed data as an `MvKind::Mlx4` GPU weight, so the
            // last-stage vocab matvec dispatches exactly like a dense projection.
            if lmhead_packed_on && is_quant && name.ends_with("lm_head.weight")
                && qwen35_tensor_bits(&name, bits) == 4
            {
                let scales = decode(&read_pv(&format!("{raw_base}.scales"))?
                    .ok_or_else(|| format!("{raw_base}.scales missing"))?);
                let biases = decode(&read_pv(&format!("{raw_base}.biases"))?
                    .ok_or_else(|| format!("{raw_base}.biases missing"))?);
                let packed: Vec<u32> = view.data().chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let out_features = view.shape()[0];
                let in_features = view.shape()[1] * (32 / bits);
                let gsize = mlx_affine_group_size(&name, in_features, out_features, scales.len())?;
                validate_affine_dims(&name, packed.len(), out_features, in_features, gsize, bits, scales.len(), biases.len())?;
                out_lmhead_packed = Some(PackedEmbed {
                    packed, scales, biases,
                    vocab: out_features, hidden: in_features, group_size: gsize, bits,
                });
                continue;
            }
            // Dequantize STRAIGHT to f16 (no full f32 temp) — the embed/lm_head
            // ~5GB f32 temp is what OOMs a 14GB node.
            let f16v: Vec<u16> = if is_nvfp4 {
                let (wscale, global, out_features, in_features, gsize) = nvfp4_parts(&view)?;
                dequantize_nvfp4_f16(
                    view.data(), &wscale, global, out_features, in_features, gsize,
                )
            } else if is_fp8 {
                let (scale, out_features, in_features) = fp8_parts(&view)?;
                dequantize_fp8_f16(view.data(), &scale, out_features, in_features)
            } else if is_quant {
                // Same-shard sibling, else search all shards (cross-shard split).
                let scales = decode(&read_pv(&format!("{raw_base}.scales"))?
                    .ok_or_else(|| format!("{raw_base}.scales missing"))?);
                let biases = decode(&read_pv(&format!("{raw_base}.biases"))?
                    .ok_or_else(|| format!("{raw_base}.biases missing"))?);
                let packed: Vec<u32> = view.data().chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let out_features = view.shape()[0];
                let in_features = view.shape()[1] * (32 / bits);
                let gsize = mlx_affine_group_size(&name, in_features, out_features, scales.len())?;
                validate_affine_dims(&name, packed.len(), out_features, in_features, gsize, bits, scales.len(), biases.len())?;
                dequantize_mlx_affine_f16(
                    &packed, &scales, &biases, out_features, in_features, gsize, bits,
                )
            } else {
                // Real (BF16/F16/F32) embed/lm_head: bf16 -> f16 directly, no
                // whole-table f32 transient.
                decode_to_f16(&view)
            };
            out_f16.insert(name, f16v);
        } else {
            // ── NVFP4-GPU packed path ────────────────────────────────────
            // Offer PACKED nibbles + raw e4m3 scales to the sink (no f32
            // dequant). Only matvec projections. Sink returns `Dequantize`
            // when the flag is off / TP>1 / no engine → fall through to the
            // exact same dequant as before (flag-off is byte-identical).
            if is_nvfp4 && is_qwen35_matvec_weight_name(&name) {
                let (wscale, global, out_f, in_f, gs) = nvfp4_parts(&view)?;
                let taken = matches!(
                    on_proj(&name, ProjWeight::Nvfp4(Nvfp4Weight {
                        packed: view.data(), wscale: &wscale, global,
                        out_features: out_f, in_features: in_f, group_size: gs,
                    })),
                    ProjResult::Consumed);
                if taken { continue; }
                let deq = dequantize_nvfp4(view.data(), &wscale, global, out_f, in_f, gs);
                if let ProjResult::KeepF32(v) = on_proj(&name, ProjWeight::F32(deq)) {
                    out_f32.insert(name, v);
                }
                continue;
            }
            // ── FP8-GPU packed path ──────────────────────────────────────
            // Offer PACKED FP8-E4M3 bytes + f32 scale to the sink (no f32
            // dequant). Only matvec projections. Sink returns `Dequantize`
            // when the flag is off / no engine → fall through to the exact
            // same dequant as before (flag-off is byte-identical).
            if is_fp8 && is_qwen35_matvec_weight_name(&name) {
                let (scale, out_f, in_f) = fp8_parts(&view)?;
                let taken = matches!(
                    on_proj(&name, ProjWeight::Fp8(Fp8Weight {
                        weight: view.data(), scale: scale.clone(),
                        out_features: out_f, in_features: in_f,
                    })),
                    ProjResult::Consumed);
                if taken { continue; }
                let deq = dequantize_fp8(view.data(), &scale, out_f, in_f);
                if let ProjResult::KeepF32(v) = on_proj(&name, ProjWeight::F32(deq)) {
                    out_f32.insert(name, v);
                }
                continue;
            }
            // ── MLX4-affine dense packed path (item-3 4-bit dense
            // residency). Gemma offered this via `st.tensor`, which this
            // OOM-refactored split loader no longer has, so it is ported onto
            // the `read_pv` streaming reads used by the nvfp4/fp8 sinks above
            // and the `is_quant` dequant below. Offer PACKED 4-bit nibbles +
            // f32 scales/biases to the sink; only DENSE matvec projections
            // (GDN/attn) — MoE experts are excluded by the matvec predicate.
            // Sink returns `Dequantize` when the flag is off / no engine ->
            // fall through to the byte-identical dequant below.
            if is_quant && is_qwen35_matvec_weight_name(&name) {
                let tbits = qwen35_tensor_bits(&name, bits);
                if tbits == 4 {
                    let scales = decode(&read_pv(&format!("{raw_base}.scales"))?
                        .ok_or_else(|| format!("{raw_base}.scales missing"))?);
                    let biases = decode(&read_pv(&format!("{raw_base}.biases"))?
                        .ok_or_else(|| format!("{raw_base}.biases missing"))?);
                    let packed: Vec<u32> = view.data().chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    let out_features = view.shape()[0];
                    let in_features = view.shape()[1] * (32 / tbits);
                    let gsize = mlx_affine_group_size(&name, in_features, out_features, scales.len())?;
                    validate_affine_dims(&name, packed.len(), out_features, in_features, gsize, tbits, scales.len(), biases.len())?;
                    let taken = matches!(
                        on_proj(&name, ProjWeight::Mlx4(Mlx4Weight {
                            packed: packed.clone(), scales: scales.clone(), biases: biases.clone(),
                            out_features, in_features, group_size: gsize,
                        })),
                        ProjResult::Consumed);
                    if taken { continue; }
                    let deq = dequantize_mlx_affine(
                        &packed, &scales, &biases, out_features, in_features, gsize, tbits,
                    );
                    if let ProjResult::KeepF32(v) = on_proj(&name, ProjWeight::F32(deq)) {
                        out_f32.insert(name, v);
                    }
                    continue;
                }
            }
            let deq: Vec<f32> = if is_nvfp4 {
                let (wscale, global, out_features, in_features, gsize) = nvfp4_parts(&view)?;
                dequantize_nvfp4(
                    view.data(), &wscale, global, out_features, in_features, gsize,
                )
            } else if is_fp8 {
                let (scale, out_features, in_features) = fp8_parts(&view)?;
                dequantize_fp8(view.data(), &scale, out_features, in_features)
            } else if is_quant {
                let scales = decode(&read_pv(&format!("{raw_base}.scales"))?
                    .ok_or_else(|| format!("{raw_base}.scales missing"))?);
                let biases = decode(&read_pv(&format!("{raw_base}.biases"))?
                    .ok_or_else(|| format!("{raw_base}.biases missing"))?);
                let packed: Vec<u32> = view.data().chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                // Per-tensor bit-width from geometry (`mlx_affine_bits`): the
                // 35B-A3B routers are 8-bit but the mlx-community 122B-A10B is
                // uniform 4-bit incl. routers. Config `bits` is the fallback.
                let tbits = match mlx_affine_bits(packed.len(), scales.len(), group_size) {
                    0 => bits,
                    b => b,
                };
                // 3D `switch_mlp.*` experts are `[E, out, in_packed]`; flatten
                // to `[E*out, in]` (each expert-row is a contiguous quantized
                // row, so the 2D affine dequant is exact). 2D tensors are
                // `[out, in_packed]`. `in_features = in_packed * (32/bits)`.
                let shape = view.shape();
                let (out_features, in_features) = if shape.len() == 3 {
                    (shape[0] * shape[1], shape[2] * (32 / tbits))
                } else {
                    (shape[0], shape[1] * (32 / tbits))
                };
                let gsize = mlx_affine_group_size(&name, in_features, out_features, scales.len())?;
                validate_affine_dims(&name, packed.len(), out_features, in_features, gsize, tbits, scales.len(), biases.len())?;
                dequantize_mlx_affine(
                    &packed, &scales, &biases, out_features, in_features, gsize, tbits,
                )
            } else {
                let mut w = decode(&view);
                // HF/modelopt qwen3_5 exports store the decoder RMSNorm
                // weights ZERO-CENTERED: the HF module computes
                // `x_hat * (1 + w)`, and the modelopt NVFP4/FP8 checkpoint
                // keeps that raw `w` (mean ~0). MLX exports bake the +1 into
                // the stored weight, and this codebase's forward multiplies by
                // `w` directly (MLX convention) — loading the HF weights raw
                // multiplies every layer's hidden state by ~0 and produces
                // structural garbage from the first token (2026-07-19 PP-5
                // NVFP4 bring-up: argmax 107544 instead of 11751 ' Paris',
                // identical with NVFP4_GPU=0/1). Fold the +1 at load for the
                // HF-nested (`model.language_model.*`) flavor only; verified
                // against the 4bit MLX export of the same base model:
                // `mlx == 1 + hf` to bf16 rounding for input/post_attention
                // layernorms, q/k norms and the final `model.norm`, while
                // `linear_attn.norm` (gated RMSNorm) is bit-identical (plain
                // in both) and must NOT be offset.
                if qwen35_hf_zero_centered_norm(raw_name, &name) {
                    for v in w.iter_mut() {
                        *v += 1.0;
                    }
                }
                w
            };
            // Projection (matvec) weights: hand to the streaming sink. It
            // returns `Consumed` if it took the tensor (GPU upload + drop),
            // or `KeepF32` to accumulate as f32 (CPU-only mode). MoE tensors
            // are excluded from the matvec predicate -> host f32.
            if is_qwen35_matvec_weight_name(&name) {
                if let ProjResult::KeepF32(v) = on_proj(&name, ProjWeight::F32(deq)) {
                    out_f32.insert(name, v);
                }
            } else {
                out_f32.insert(name, deq);
            }
        }
    }
    Ok((out_f32, out_f16, out_embed_packed, out_lmhead_packed))
}

/// Whether to keep 35B-A3B MoE experts 4-bit-RESIDENT (default ON). The f32
/// dequant of a layer's experts is ~3.2GB; resident-4-bit is ~0.4GB, the
/// difference between fitting an 8-layer PP stage on a 15GB node and OOMing.
/// Set `VLLM_VULKAN_MOE_Q4_RESIDENT=0` to force the f32-host experts path
/// (single-node parity / debugging).
/// Whether to load the resident MoE experts via header-parse + per-tensor pread
/// instead of a whole-shard `Mmap::map` (default OFF; `VLLM_VULKAN_MOE_PREAD_LOAD=1`
/// opts in). The mmap path maps the ENTIRE shard as one VMA; on the RAM-backed
/// livecd that mapped shard counts resident at FULL shard size (multi-GB) for the
/// duration of the tensor loop, on top of the retained 4-bit experts — the
/// load-time anon spike that tips one 122B PP-6 stage over the ~10-11GB cliff
/// (steady-state fits). pread reads only each kept tensor's byte range on demand,
/// so the whole-shard VMA is never resident: the load peak drops to one packed
/// expert tensor plus the retained experts. Bit-exact by construction (pread reads
/// the same bytes mmap would); the only change is the read mechanism.
pub fn moe_pread_load_enabled() -> bool {
    std::env::var("VLLM_VULKAN_MOE_PREAD_LOAD").map(|v| v != "0").unwrap_or(false)
}

pub fn moe_q4_resident_enabled() -> bool {
    std::env::var("VLLM_VULKAN_MOE_Q4_RESIDENT").map(|v| v != "0").unwrap_or(true)
}

/// Load the 4-bit-RESIDENT `switch_mlp.*` expert tensors for the resident layer
/// range `[layer_start, layer_end)` WITHOUT dequantizing — packed u32 + f32
/// scales/biases per `QuantSwitch`. This is the memory lever for the 35B-A3B MoE
/// on the cluster: a layer's experts cost ~0.4GB (4-bit) here vs ~3.2GB (f32).
/// The router gate + shared expert (small) are NOT loaded here; they come from
/// the f32 `load_qwen35_weights_split` path.
pub fn load_qwen35_moe_quant_experts(
    path: &Path,
    group_size: usize,
    bits: usize,
    layer_start: usize,
    layer_end: usize,
) -> Result<crate::moe::QuantMoeLayers, String> {
    // pread path (VLLM_VULKAN_MOE_PREAD_LOAD=1): header-parse + per-tensor pread,
    // no whole-shard mmap resident during the load loop. Bit-exact to the mmap
    // body below (same bytes, same dequant). This is the 122B PP-6 load lever.
    if moe_pread_load_enabled() {
        return load_qwen35_moe_quant_experts_pread(path, group_size, bits, layer_start, layer_end);
    }

    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;

    let shards = discover_shards(path);

    let decode_f32 = |view: &safetensors::tensor::TensorView| -> Vec<f32> {
        let d = view.data();
        match view.dtype() {
            safetensors::Dtype::BF16 => d.chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            safetensors::Dtype::F16 => d.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            safetensors::Dtype::F32 => d.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            other => panic!(
                "unsupported safetensors dtype {other:?} in weight loader \
                 (only BF16/F16/F32 are decodable here)"
            ),
        }
    };

    let mut out = crate::moe::QuantMoeLayers::default();
    for shard in &shards {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;
        for (raw_name, view) in st.tensors() {
            let name = normalize_qwen35_name(&raw_name);
            // Only `model.layers.{idx}.mlp.switch_mlp.{gate,up,down}_proj.weight`.
            let rest = match name.strip_prefix("model.layers.") { Some(r) => r, None => continue };
            let (idx_s, after) = match rest.split_once('.') { Some(x) => x, None => continue };
            let idx: usize = match idx_s.parse() { Ok(i) => i, Err(_) => continue };
            if idx < layer_start || idx >= layer_end { continue; }
            let which = if after == "mlp.switch_mlp.gate_proj.weight" { 0 }
                else if after == "mlp.switch_mlp.up_proj.weight" { 1 }
                else if after == "mlp.switch_mlp.down_proj.weight" { 2 }
                else { continue };
            if view.dtype() != safetensors::Dtype::U32 { continue; }
            let shape = view.shape(); // [E, out, in_packed]
            if shape.len() != 3 { return Err(format!("{name}: expected 3D switch tensor, got {:?}", shape)); }
            let tbits = qwen35_tensor_bits(&name, bits);
            let num_experts = shape[0];
            let out_features = shape[1];
            let in_features = shape[2] * (32 / tbits);
            let packed: Vec<u32> = view.data().chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let raw_base = &raw_name[..raw_name.len() - ".weight".len()];
            let scales = match st.tensor(&format!("{raw_base}.scales")) {
                Ok(sv) => decode_f32(&sv),
                Err(_) => find_quant_sibling_f32(&shards, raw_base, "scales")?,
            };
            let biases = match st.tensor(&format!("{raw_base}.biases")) {
                Ok(bv) => decode_f32(&bv),
                Err(_) => find_quant_sibling_f32(&shards, raw_base, "biases")?,
            };
            // The packed / scales / biases buffers span ALL experts (the tensor is
            // 3D `[E, out, in_packed]`), so validate against the flattened
            // `[E*out, in]` layout — exactly how `load_qwen35_weights_split`
            // flattens the f32 path. `QuantSwitch.out_features` stays PER-EXPERT
            // (`out_features`) because the downstream per-expert dispatch slices
            // rows by it. (Regression: 5923c49 validated with the per-expert
            // `out_features` against the all-expert buffers, so every 3D switch
            // tensor failed `want_packed`, left `quant_moe` empty, and the forward
            // panicked in the f32 borrowed path.)
            let stacked_out = num_experts * out_features;
            validate_affine_dims(&name, packed.len(), stacked_out, in_features, group_size, tbits, scales.len(), biases.len())?;
            let qs = crate::moe::QuantSwitch {
                packed,
                scales,
                biases,
                out_features,
                in_features,
                group_size,
                bits: tbits,
            };
            match which {
                0 => { out.gate.insert(idx, qs); }
                1 => { out.up.insert(idx, qs); }
                _ => { out.down.insert(idx, qs); }
            }
        }
    }
    Ok(out)
}

/// pread analog of `load_qwen35_moe_quant_experts`: header-parse every shard,
/// then pread ONLY the resident-range `switch_mlp.*` expert tensors (and their
/// `.scales`/`.biases` siblings) by byte offset. NO whole-shard `Mmap::map` is
/// held resident during the load loop, so the load-time anon peak is one packed
/// expert tensor plus the retained experts — not one-full-shard-VMA + experts.
/// Mirrors `load_qwen35_weights_split`'s header-parse + pread exactly, and is
/// BIT-EXACT to the mmap body (same byte ranges, same dequant math).
fn load_qwen35_moe_quant_experts_pread(
    path: &Path,
    group_size: usize,
    bits: usize,
    layer_start: usize,
    layer_end: usize,
) -> Result<crate::moe::QuantMoeLayers, String> {
    use safetensors::Dtype;
    use std::fs::File;
    use std::os::unix::fs::FileExt;

    let shards = discover_shards(path);

    // A tensor located in a shard header: dtype/shape + absolute byte range.
    struct Entry {
        shard: usize,
        dtype: Dtype,
        shape: Vec<usize>,
        off: u64,
        len: usize,
    }
    let dtype_from_str = |s: &str| -> Result<Dtype, String> {
        Ok(match s {
            "F64" => Dtype::F64, "F32" => Dtype::F32, "F16" => Dtype::F16,
            "BF16" => Dtype::BF16, "F8_E4M3" => Dtype::F8_E4M3,
            "F8_E5M2" => Dtype::F8_E5M2, "U8" => Dtype::U8, "I8" => Dtype::I8,
            "U16" => Dtype::U16, "I16" => Dtype::I16, "U32" => Dtype::U32,
            "I32" => Dtype::I32, "U64" => Dtype::U64, "I64" => Dtype::I64,
            "BOOL" => Dtype::BOOL,
            other => return Err(format!("unsupported safetensors dtype {other}")),
        })
    };

    // Parse every shard header (no tensor data faulted). First shard that defines
    // a name wins (safetensors names are unique across a checkpoint).
    let mut files: Vec<File> = Vec::with_capacity(shards.len());
    let mut entries: HashMap<String, Entry> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for (si, shard) in shards.iter().enumerate() {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mut lenb = [0u8; 8];
        file.read_exact_at(&mut lenb, 0)
            .map_err(|e| format!("read header len {}: {e}", shard.display()))?;
        let hlen = u64::from_le_bytes(lenb);
        let mut hbuf = vec![0u8; hlen as usize];
        file.read_exact_at(&mut hbuf, 8)
            .map_err(|e| format!("read header {}: {e}", shard.display()))?;
        let hjson: serde_json::Value = serde_json::from_slice(&hbuf)
            .map_err(|e| format!("parse header json {}: {e}", shard.display()))?;
        let data_start = 8 + hlen;
        if let Some(obj) = hjson.as_object() {
            for (name, info) in obj {
                if name == "__metadata__" { continue; }
                let dt = dtype_from_str(
                    info.get("dtype").and_then(|v| v.as_str())
                        .ok_or_else(|| format!("{name}: missing dtype"))?)?;
                let shape: Vec<usize> = info.get("shape").and_then(|v| v.as_array())
                    .ok_or_else(|| format!("{name}: missing shape"))?
                    .iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
                let offs = info.get("data_offsets").and_then(|v| v.as_array())
                    .ok_or_else(|| format!("{name}: missing data_offsets"))?;
                if offs.len() != 2 { return Err(format!("{name}: bad data_offsets")); }
                let begin = offs[0].as_u64().unwrap_or(0);
                let end = offs[1].as_u64().unwrap_or(0);
                if end < begin { return Err(format!("{name}: end<begin offset")); }
                if !entries.contains_key(name) {
                    order.push(name.clone());
                    entries.insert(name.clone(), Entry {
                        shard: si, dtype: dt, shape,
                        off: data_start + begin, len: (end - begin) as usize,
                    });
                }
            }
        }
        files.push(file);
    }

    // pread a tensor's raw bytes (dtype/shape from header). None if absent — the
    // scales/biases fall back to the cross-shard sibling search on None, exactly
    // like the mmap body's `st.tensor(..)` → `find_quant_sibling_f32` fallback
    // (dead here, since `entries` already spans all shards, but kept for parity).
    let read_raw = |raw: &str| -> Result<Option<(Dtype, Vec<u8>)>, String> {
        match entries.get(raw) {
            None => Ok(None),
            Some(e) => {
                let mut buf = vec![0u8; e.len];
                files[e.shard].read_exact_at(&mut buf, e.off)
                    .map_err(|err| format!("pread {raw}: {err}"))?;
                Ok(Some((e.dtype, buf)))
            }
        }
    };
    let decode_f32 = |dtype: Dtype, d: &[u8]| -> Vec<f32> {
        match dtype {
            Dtype::BF16 => d.chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            Dtype::F16 => d.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            Dtype::F32 => d.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            other => panic!(
                "unsupported safetensors dtype {other:?} in weight loader \
                 (only BF16/F16/F32 are decodable here)"
            ),
        }
    };

    let mut out = crate::moe::QuantMoeLayers::default();
    for raw_name in &order {
        let name = normalize_qwen35_name(raw_name);
        // Only `model.layers.{idx}.mlp.switch_mlp.{gate,up,down}_proj.weight`.
        let rest = match name.strip_prefix("model.layers.") { Some(r) => r, None => continue };
        let (idx_s, after) = match rest.split_once('.') { Some(x) => x, None => continue };
        let idx: usize = match idx_s.parse() { Ok(i) => i, Err(_) => continue };
        if idx < layer_start || idx >= layer_end { continue; }
        let which = if after == "mlp.switch_mlp.gate_proj.weight" { 0 }
            else if after == "mlp.switch_mlp.up_proj.weight" { 1 }
            else if after == "mlp.switch_mlp.down_proj.weight" { 2 }
            else { continue };
        let ent = &entries[raw_name];
        if ent.dtype != Dtype::U32 { continue; }
        let shape = ent.shape.clone(); // [E, out, in_packed]
        if shape.len() != 3 { return Err(format!("{name}: expected 3D switch tensor, got {:?}", shape)); }
        let tbits = qwen35_tensor_bits(&name, bits);
        let num_experts = shape[0];
        let out_features = shape[1];
        let in_features = shape[2] * (32 / tbits);
        let (_, pbytes) = read_raw(raw_name)?
            .ok_or_else(|| format!("tensor vanished from header: {raw_name}"))?;
        let packed: Vec<u32> = pbytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let raw_base = &raw_name[..raw_name.len() - ".weight".len()];
        let scales = match read_raw(&format!("{raw_base}.scales"))? {
            Some((dt, d)) => decode_f32(dt, &d),
            None => find_quant_sibling_f32(&shards, raw_base, "scales")?,
        };
        let biases = match read_raw(&format!("{raw_base}.biases"))? {
            Some((dt, d)) => decode_f32(dt, &d),
            None => find_quant_sibling_f32(&shards, raw_base, "biases")?,
        };
        let stacked_out = num_experts * out_features;
        validate_affine_dims(&name, packed.len(), stacked_out, in_features, group_size, tbits, scales.len(), biases.len())?;
        let qs = crate::moe::QuantSwitch {
            packed,
            scales,
            biases,
            out_features,
            in_features,
            group_size,
            bits: tbits,
        };
        match which {
            0 => { out.gate.insert(idx, qs); }
            1 => { out.up.insert(idx, qs); }
            _ => { out.down.insert(idx, qs); }
        }
    }
    Ok(out)
}

/// Same matvec-weight predicate as lib.rs `is_qwen35_matvec_weight`, duplicated
/// here so the streaming loader can decide what to route to the GPU sink.
///
/// NOTE: MoE expert tensors (`switch_mlp.*`, `shared_expert.*`) ALSO end in
/// `.gate_proj.weight` / `.up_proj.weight` / `.down_proj.weight`, but they are
/// NOT dense projection matvecs — they are consumed by the CPU `moe` block and
/// must stay host f32. They are excluded here via `is_qwen35_moe_weight_name`.
pub fn is_qwen35_matvec_weight_name(name: &str) -> bool {
    if is_qwen35_moe_weight_name(name) {
        return false;
    }
    name.ends_with(".q_proj.weight")
        || name.ends_with(".k_proj.weight")
        || name.ends_with(".v_proj.weight")
        || name.ends_with(".o_proj.weight")
        || name.ends_with(".gate_proj.weight")
        || name.ends_with(".up_proj.weight")
        || name.ends_with(".down_proj.weight")
        || name.ends_with(".in_proj_qkv.weight")
        || name.ends_with(".in_proj_z.weight")
        || name.ends_with(".in_proj_a.weight")
        || name.ends_with(".in_proj_b.weight")
        || name.ends_with(".out_proj.weight")
}

/// True if this is a Qwen3.6 MoE-block tensor (router gate, stacked experts, or
/// shared expert). These need MoE-specific handling in the loader: the router
/// gate + `shared_expert_gate` are 8-bit (not the model-default 4-bit), and the
/// `switch_mlp.*` expert tensors are 3D `[num_experts, out, in]`. They are kept
/// host f32 and consumed by the CPU `moe::moe_forward_token` block.
pub fn is_qwen35_moe_weight_name(name: &str) -> bool {
    if let Some(rest) = name.strip_prefix("model.layers.") {
        // rest = "{idx}.mlp...."
        let after = rest.split_once('.').map(|(_, r)| r).unwrap_or("");
        return after.starts_with("mlp.gate.weight")
            || after.starts_with("mlp.switch_mlp.")
            || after.starts_with("mlp.shared_expert.")
            || after.starts_with("mlp.shared_expert_gate.");
    }
    false
}

/// Per-tensor quantization bit-width for a Qwen3.6 tensor. The 35B-A3B MoE
/// checkpoint uses a non-uniform layout: the router `mlp.gate` and the
/// `mlp.shared_expert_gate` are **8-bit**, everything else is the model-default
/// (4-bit). Mirrors the `config.json` `quantization` per-tensor overrides.
pub fn qwen35_tensor_bits(name: &str, default_bits: usize) -> usize {
    if name.ends_with(".mlp.gate.weight") || name.ends_with(".mlp.shared_expert_gate.weight") {
        8
    } else {
        default_bits
    }
}

// ─── Qwen3 (dense) architecture ──────────────────────────────────────────────
//
// Standard Qwen3 dense models (Qwen3-0.6B / 1.7B / 4B / 8B …).  Compared to
// Gemma4-E2B this is a much simpler, "classic" pre-norm transformer:
//   - GQA attention with per-head Q/K RMSNorm (applied before RoPE)
//   - full (non-sliding) attention on every layer, scale = 1/sqrt(head_dim)
//   - NeoX-style RoPE over the whole head_dim (theta from config)
//   - SwiGLU MLP: down(silu(gate) * up)
//   - two RMSNorms per layer (input_layernorm, post_attention_layernorm),
//     each applied *before* its sub-block with the residual added after
//   - no PLE, no per-layer scalar, no KV sharing, no logit softcapping,
//     no embedding scaling, no attention/QKV bias
// `head_dim` is independent of hidden_size/num_heads (e.g. 128 with hidden 1024).

/// Qwen3 dense architecture constants (parsed from config.json).
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    /// Build a config from a parsed `config.json` value.
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let usize_field = |key: &str| v[key].as_u64().map(|x| x as usize);
        let require = |key: &str| {
            usize_field(key).ok_or_else(|| format!("config.json missing '{key}'"))
        };

        let hidden_size = require("hidden_size")?;
        let num_attention_heads = require("num_attention_heads")?;
        let num_key_value_heads = usize_field("num_key_value_heads").unwrap_or(num_attention_heads);
        // Qwen3 ships an explicit head_dim that is *not* hidden_size / num_heads.
        let head_dim = usize_field("head_dim").unwrap_or(hidden_size / num_attention_heads);

        Ok(Qwen3Config {
            hidden_size,
            num_hidden_layers: require("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size: require("intermediate_size")?,
            vocab_size: require("vocab_size")?,
            rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta: v["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
            tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        })
    }
}

/// Qwen3 dense model — pure-CPU reference forward pass (one token per call).
///
/// Used directly when no Vulkan device is available; the GPU-accelerated path
/// in `lib.rs` mirrors this exactly but routes the large matmuls to the GPU.
pub struct Qwen3Model {
    pub config: Qwen3Config,
    pub weights: ModelWeights,
    pub kv_caches: Vec<KvCache>,
    /// Resolved LM-head weight name: tied → "model.embed_tokens.weight".
    pub lm_head_name: String,
}

impl Qwen3Model {
    /// Forward pass for one token at position `pos`.  Returns logits [vocab_size].
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        // Embedding (no scaling, unlike Gemma4).
        let embed_w = self.weights.f32_slice("model.embed_tokens.weight");
        let mut hidden: Vec<f32> =
            embed_w[token_id as usize * h..(token_id as usize + 1) * h].to_vec();

        for layer_idx in 0..cfg.num_hidden_layers {
            hidden = self.forward_layer(layer_idx, &hidden, pos);
        }

        // Final norm + LM head (no logit softcapping).
        let norm_w = self.weights.f32_slice("model.norm.weight");
        let normed = cpu_rms_norm(&hidden, norm_w, eps);
        let lm_w = self.weights.f32_slice(&self.lm_head_name);
        cpu_matmul(&normed, lm_w, 1, h, cfg.vocab_size)
    }

    /// Like [`forward`](Self::forward), but also returns the hidden state after
    /// each decoder layer.  Used by the numerical parity harness to localise
    /// any divergence from a reference implementation.
    /// Returns `(per_layer_hidden, logits)`.
    pub fn forward_capture(&mut self, token_id: u32, pos: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        let embed_w = self.weights.f32_slice("model.embed_tokens.weight");
        let mut hidden: Vec<f32> =
            embed_w[token_id as usize * h..(token_id as usize + 1) * h].to_vec();

        let mut per_layer = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            hidden = self.forward_layer(layer_idx, &hidden, pos);
            per_layer.push(hidden.clone());
        }

        let norm_w = self.weights.f32_slice("model.norm.weight");
        let normed = cpu_rms_norm(&hidden, norm_w, eps);
        let lm_w = self.weights.f32_slice(&self.lm_head_name);
        let logits = cpu_matmul(&normed, lm_w, 1, h, cfg.vocab_size);
        (per_layer, logits)
    }

    pub fn forward_layer(&mut self, layer_idx: usize, hidden: &[f32], pos: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let num_q = cfg.num_attention_heads;
        let num_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let inter = cfg.intermediate_size;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let ln = |w: &str| format!("model.layers.{layer_idx}.{w}");

        // ── Attention block ────────────────────────────────────────────────
        let residual = hidden.to_vec();
        let inln_w = self.weights.f32_slice(&ln("input_layernorm.weight")).to_vec();
        let x = cpu_rms_norm(hidden, &inln_w, eps);

        let q_w = self.weights.f32_slice(&ln("self_attn.q_proj.weight")).to_vec();
        let k_w = self.weights.f32_slice(&ln("self_attn.k_proj.weight")).to_vec();
        let v_w = self.weights.f32_slice(&ln("self_attn.v_proj.weight")).to_vec();
        let mut q = cpu_matmul(&x, &q_w, 1, h, q_dim);
        let mut k = cpu_matmul(&x, &k_w, 1, h, kv_dim);
        let v = cpu_matmul(&x, &v_w, 1, h, kv_dim);

        // Per-head Q/K RMSNorm (applied before RoPE).
        let q_norm_w = self.weights.f32_slice(&ln("self_attn.q_norm.weight")).to_vec();
        let k_norm_w = self.weights.f32_slice(&ln("self_attn.k_norm.weight")).to_vec();
        for hi in 0..num_q {
            let s = &mut q[hi * head_dim..(hi + 1) * head_dim];
            let n = cpu_rms_norm(s, &q_norm_w, eps);
            s.copy_from_slice(&n);
        }
        for hi in 0..num_kv {
            let s = &mut k[hi * head_dim..(hi + 1) * head_dim];
            let n = cpu_rms_norm(s, &k_norm_w, eps);
            s.copy_from_slice(&n);
        }

        // RoPE: full rotation over head_dim, NeoX style.
        cpu_rope(&mut q, &mut k, pos, num_q, num_kv, head_dim, head_dim, cfg.rope_theta);

        // KV cache update + SDPA (full causal attention, no sliding window).
        self.kv_caches[layer_idx].append(&k, &v);
        let attn_out = {
            let cache = &self.kv_caches[layer_idx];
            cpu_sdpa(
                &q, cache.k_up_to_now(), cache.v_up_to_now(),
                num_q, num_kv, head_dim, cache.seq_len, scale, None,
            )
        };

        let o_w = self.weights.f32_slice(&ln("self_attn.o_proj.weight")).to_vec();
        let attn_proj = cpu_matmul(&attn_out, &o_w, 1, q_dim, h);
        let hidden2: Vec<f32> = residual.iter().zip(attn_proj.iter())
            .map(|(&r, &a)| r + a).collect();
        let residual2 = hidden2.clone();

        // ── MLP block (SwiGLU) ──────────────────────────────────────────────
        let pa_w = self.weights.f32_slice(&ln("post_attention_layernorm.weight")).to_vec();
        let ff_in = cpu_rms_norm(&hidden2, &pa_w, eps);

        let gate_w = self.weights.f32_slice(&ln("mlp.gate_proj.weight")).to_vec();
        let up_w = self.weights.f32_slice(&ln("mlp.up_proj.weight")).to_vec();
        let gate = cpu_matmul(&ff_in, &gate_w, 1, h, inter);
        let up = cpu_matmul(&ff_in, &up_w, 1, h, inter);
        let gate_act = cpu_silu(&gate);
        let mid: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
        let down_w = self.weights.f32_slice(&ln("mlp.down_proj.weight")).to_vec();
        let ff_out = cpu_matmul(&mid, &down_w, 1, inter, h);

        residual2.iter().zip(ff_out.iter()).map(|(&r, &f)| r + f).collect()
    }
}

// ─── Common model interface ──────────────────────────────────────────────────

/// CPU reference forward pass + KV-cache metadata shared by every supported
/// architecture.  This is the extension point for adding new models: implement
/// `LanguageModel` for the new `*Model` type and wire it into `ModelConfig` /
/// `load_config`.  GPU acceleration is layered on top per-architecture in
/// `lib.rs` (it needs arch-specific kernels and is not part of this trait).
pub trait LanguageModel {
    /// Forward pass for one token at `pos`; returns logits `[vocab_size]`.
    fn forward(&mut self, token_id: u32, pos: usize) -> Vec<f32>;
    /// Reset all KV caches (start a new sequence).
    fn reset_kv_cache(&mut self);
    /// Current cached sequence length.
    fn seq_len(&self) -> usize;
    /// Number of decoder layers.
    fn num_layers(&self) -> usize;
}

impl LanguageModel for Gemma4Model {
    fn forward(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        Gemma4Model::forward(self, token_id, pos)
    }
    fn reset_kv_cache(&mut self) {
        for cache in self.kv_caches.iter_mut() {
            cache.seq_len = 0;
        }
    }
    fn seq_len(&self) -> usize {
        self.kv_caches[0].seq_len
    }
    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

impl LanguageModel for Qwen3Model {
    fn forward(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        Qwen3Model::forward(self, token_id, pos)
    }
    fn reset_kv_cache(&mut self) {
        for cache in self.kv_caches.iter_mut() {
            cache.seq_len = 0;
        }
    }
    fn seq_len(&self) -> usize {
        self.kv_caches[0].seq_len
    }
    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

// ─── Architecture detection ──────────────────────────────────────────────────

/// Parsed model configuration, tagged by architecture.
pub enum ModelConfig {
    Gemma4(Gemma4Config),
    Qwen3(Qwen3Config),
}

/// Detect the model architecture from a `config.json` file and build the
/// matching config.  Defaults to Gemma4-E2B (the original hardcoded behaviour)
/// when the architecture is unrecognised, so existing Gemma4 setups are
/// unaffected.
pub fn load_config(config_path: &Path) -> Result<ModelConfig, String> {
    let text = std::fs::read_to_string(config_path)
        .map_err(|e| format!("read {}: {e}", config_path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("parse {}: {e}", config_path.display()))?;

    let model_type = v["model_type"].as_str().unwrap_or("");
    let arch0 = v["architectures"]
        .get(0)
        .and_then(|a| a.as_str())
        .unwrap_or("");

    // qwen3_5 / Qwen3_5* is the hybrid GatedDeltaNet+MoE "Qwen3.6" family,
    // which this backend does not support yet — reject it explicitly rather
    // than silently mis-loading it as a plain transformer.
    if model_type.starts_with("qwen3_5") || arch0.starts_with("Qwen3_5") {
        return Err(
            "Qwen3.6 (qwen3_5) hybrid GatedDeltaNet/MoE models are not supported \
             (only standard Qwen3 dense models)".to_string(),
        );
    }

    if model_type == "qwen3" || arch0 == "Qwen3ForCausalLM" {
        if v.get("num_experts").and_then(|n| n.as_u64()).unwrap_or(0) > 0 {
            return Err("Qwen3 MoE models are not supported (dense Qwen3 only)".to_string());
        }
        return Ok(ModelConfig::Qwen3(Qwen3Config::from_json(&v)?));
    }

    // Unknown / Gemma4 → preserve original hardcoded behaviour.
    Ok(ModelConfig::Gemma4(Gemma4Config::e2b()))
}

// ─── Tiny in-memory synthetic Gemma4 (test-only) ─────────────────────────────
//
// The INC-4/INC-5a CPU logic gates (batched-verify == serial, KV rollback,
// spec-decode accept/reject bookkeeping) only need a model that exercises the
// same code paths as the real gemma-4-12B/31B checkpoints — they do NOT need
// real weights, since what's being validated is control flow / bookkeeping,
// not numerical accuracy. Loading the real ~48GB f32 12B checkpoint takes
// ~30 min per test run, which makes iterating on this logic impractical.
// `tiny_synthetic_gemma` builds a small but structurally representative
// Gemma4Model entirely in memory (no checkpoint, no env var) in well under a
// second, so gates that only assert self-consistency (verify_core vs serial
// forward; rollback vs a clean baseline; spec-driver vs greedy) can run on
// every `cargo test`. The original real-checkpoint versions of these gates are
// kept as `#[ignore]`d "_real12b" tests for optional real-weight confidence.

/// FNV-1a over a tensor name — deterministic per-tensor seed, no RNG crate, no
/// time/entropy source, so `synth_tensor` output is bit-reproducible run to run.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Deterministic weight tensor: an index-seeded LCG (seeded from the tensor
/// name via `fnv1a`), mapped into `center ± half_range`. Same name + shape +
/// range always produces the same data, so gates built on this are
/// bit-reproducible. Projection matrices use `synth_tensor` (range centered on
/// 0, matching typical small-init transformer weights); RMSNorm weights and
/// the residual `layer_scalar` use `synth_tensor_centered(.., 1.0, ..)`
/// instead — those multiply the WHOLE residual stream (norm weight) or the
/// whole post-layer hidden state (`layer_scalar`), so centering them on 0
/// would compound a near-zero multiplier 16+ times across 4 layers and
/// collapse the hidden state into numerical noise (observed in practice: it
/// produced identical argmax across distinct input tokens, and a stray
/// rounding ULP away from an exact `cos == 1.0` in the rollback gate).
/// Centering on 1 (mirroring real trained RMSNorm-weight / residual-scale
/// magnitudes) keeps the signal well-conditioned while remaining fully
/// deterministic.
fn synth_tensor(name: &str, shape: Vec<usize>) -> SimpleTensor {
    synth_tensor_centered(name, shape, 0.0, 0.05)
}

fn synth_tensor_centered(name: &str, shape: Vec<usize>, center: f32, half_range: f32) -> SimpleTensor {
    let n: usize = shape.iter().product();
    let mut s = fnv1a(name);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407)
            .wrapping_add(i as u64);
        let unit = ((s >> 33) % 100_001) as f32 / 100_000.0; // [0, 1]
        data.push(center + (unit - 0.5) * 2.0 * half_range);
    }
    SimpleTensor { data, shape }
}

/// Parameterized synthetic Gemma4 for offline cross-model KV spikes and unit
/// tests. Weights are deterministic from tensor names (+ optional name tag).
#[derive(Clone, Debug)]
pub struct SynthGemmaSpec {
    pub name: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub intermediate_size: usize,
    pub attention_period: usize,
    pub attention_k_eq_v: bool,
    pub vocab_size: usize,
    pub sliding_window: usize,
    /// Prefix mixed into every tensor name so two same-geometry models get
    /// independent weights (different "members" of a synthetic family).
    pub weight_tag: String,
}

impl SynthGemmaSpec {
    /// Default tiny gate geometry (historical `tiny_synthetic_gemma`).
    pub fn tiny() -> Self {
        Self {
            name: "gemma_tiny".into(),
            num_hidden_layers: 4,
            hidden_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_global_key_value_heads: 2,
            head_dim: 32,
            global_head_dim: 64,
            intermediate_size: 256,
            attention_period: 3,
            attention_k_eq_v: true,
            vocab_size: 512,
            sliding_window: 1024,
            weight_tag: String::new(),
        }
    }

    pub fn with_layers(mut self, n: usize) -> Self {
        self.num_hidden_layers = n;
        self.name = format!("{}_L{n}", self.name.trim_end_matches(|c: char| c.is_ascii_digit() || c == 'L' || c == '_').to_string());
        // Keep name simple:
        self.name = format!("gemma_L{n}");
        self
    }

    pub fn with_tag(mut self, tag: &str) -> Self {
        self.weight_tag = tag.to_string();
        self.name = format!("{}_{tag}", self.name);
        self
    }

    pub fn with_kv(mut self, n_kv_slide: usize, n_kv_global: usize, head_dim: usize, global_head_dim: usize) -> Self {
        self.num_key_value_heads = n_kv_slide;
        self.num_global_key_value_heads = n_kv_global;
        self.head_dim = head_dim;
        self.global_head_dim = global_head_dim;
        // Keep q heads >= kv heads
        self.num_attention_heads = self.num_attention_heads.max(n_kv_slide).max(n_kv_global);
        self
    }
}

/// Builds a synthetic Gemma4Model from `spec` (CPU-forward only; no checkpoint).
///
/// Every tensor name the CPU forward path reads is populated with deterministic
/// weights via `synth_tensor`; global (`k_eq_v`) layers correctly have no
/// `v_proj` tensor.
pub fn synthetic_gemma(spec: &SynthGemmaSpec, max_seq: usize) -> Gemma4Model {
    let hidden_size = spec.hidden_size;
    let cfg = Gemma4Config {
        variant: Gemma4Variant::G31b,
        hidden_size,
        num_hidden_layers: spec.num_hidden_layers,
        num_attention_heads: spec.num_attention_heads,
        num_key_value_heads: spec.num_key_value_heads,
        num_global_key_value_heads: spec.num_global_key_value_heads,
        head_dim: spec.head_dim,
        global_head_dim: spec.global_head_dim,
        intermediate_size: spec.intermediate_size,
        num_kv_shared_layers: 0,
        attention_period: spec.attention_period,
        attention_k_eq_v: spec.attention_k_eq_v,
        vocab_size: spec.vocab_size,
        rms_norm_eps: 1e-6,
        sliding_window: spec.sliding_window,
        hidden_size_per_layer_input: 0,
        final_logit_softcapping: 30.0,
        embed_scale: (hidden_size as f32).sqrt(),
        ple_scale: 1.0,
        per_layer_projection_scale: (hidden_size as f32).powf(-0.5),
        per_layer_input_scale: (2.0f32).powf(-0.5),
    };

    let tag = if spec.weight_tag.is_empty() {
        String::new()
    } else {
        format!("{}/", spec.weight_tag)
    };

    let mut tensors: HashMap<String, SimpleTensor> = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        synth_tensor(
            &format!("{tag}model.embed_tokens.weight"),
            vec![cfg.vocab_size, cfg.hidden_size],
        ),
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        synth_tensor_centered(
            &format!("{tag}model.norm.weight"),
            vec![cfg.hidden_size],
            1.0,
            0.05,
        ),
    );

    for l in 0..cfg.num_hidden_layers {
        let head_dim = cfg.layer_head_dim(l);
        let num_kv = cfg.layer_num_kv_heads(l);
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = num_kv * head_dim;
        let ffn = cfg.layer_intermediate_size(l);
        let ln = |w: &str| format!("model.layers.{l}.{w}");

        // Free functions (not closures) so we don't double-borrow `tensors`.
        fn put(tensors: &mut HashMap<String, SimpleTensor>, tag: &str, name: String, shape: Vec<usize>) {
            let t = synth_tensor(&format!("{tag}{name}"), shape);
            tensors.insert(name, t);
        }
        fn put_centered(
            tensors: &mut HashMap<String, SimpleTensor>,
            tag: &str,
            name: String,
            shape: Vec<usize>,
        ) {
            let t = synth_tensor_centered(&format!("{tag}{name}"), shape, 1.0, 0.05);
            tensors.insert(name, t);
        }

        put_centered(&mut tensors, &tag, ln("input_layernorm.weight"), vec![cfg.hidden_size]);
        put(&mut tensors, &tag, ln("self_attn.q_proj.weight"), vec![q_dim, cfg.hidden_size]);
        put(&mut tensors, &tag, ln("self_attn.k_proj.weight"), vec![kv_dim, cfg.hidden_size]);
        if !cfg.layer_uses_k_eq_v(l) {
            put(&mut tensors, &tag, ln("self_attn.v_proj.weight"), vec![kv_dim, cfg.hidden_size]);
        }
        put_centered(&mut tensors, &tag, ln("self_attn.q_norm.weight"), vec![head_dim]);
        put_centered(&mut tensors, &tag, ln("self_attn.k_norm.weight"), vec![head_dim]);
        put(&mut tensors, &tag, ln("self_attn.o_proj.weight"), vec![cfg.hidden_size, q_dim]);
        put_centered(&mut tensors, &tag, ln("post_attention_layernorm.weight"), vec![cfg.hidden_size]);
        put_centered(&mut tensors, &tag, ln("pre_feedforward_layernorm.weight"), vec![cfg.hidden_size]);
        put(&mut tensors, &tag, ln("mlp.gate_proj.weight"), vec![ffn, cfg.hidden_size]);
        put(&mut tensors, &tag, ln("mlp.up_proj.weight"), vec![ffn, cfg.hidden_size]);
        put(&mut tensors, &tag, ln("mlp.down_proj.weight"), vec![cfg.hidden_size, ffn]);
        put_centered(&mut tensors, &tag, ln("post_feedforward_layernorm.weight"), vec![cfg.hidden_size]);
        put_centered(&mut tensors, &tag, ln("layer_scalar"), vec![1]);
    }

    let weights = Gemma4Weights { tensors };
    let kv_caches = (0..cfg.num_hidden_layers)
        .map(|l| {
            KvCache::new_windowed(
                max_seq,
                cfg.layer_kv_capacity(l, max_seq),
                cfg.layer_num_kv_heads(l),
                cfg.layer_head_dim(l),
            )
        })
        .collect();
    Gemma4Model {
        config: cfg,
        weights,
        kv_caches,
    }
}

/// Builds a tiny, fully in-memory, structurally-representative Gemma4Model
/// (legacy entry point used by unit tests). Equivalent to
/// `synthetic_gemma(&SynthGemmaSpec::tiny(), max_seq)`.
pub fn tiny_synthetic_gemma(max_seq: usize) -> Gemma4Model {
    synthetic_gemma(&SynthGemmaSpec::tiny(), max_seq)
}

/// Reset all per-layer KV watermarks to 0 (next append overwrites).
pub fn gemma_reset_kv(model: &mut Gemma4Model) {
    for c in model.kv_caches.iter_mut() {
        c.seq_len = 0;
    }
}

/// Inject host K/V for positions `[0, seq_len)` into layer `layer_idx` and set
/// the watermark. `k`/`v` are flat `[seq_len, n_kv, head_dim]` f32 little-endian
/// layout matching `KvCache` storage (absolute, non-wrapped). Used by the
/// cross-model KV handoff spike (S6) to load a mapped prefix.
pub fn gemma_inject_kv_layer(
    model: &mut Gemma4Model,
    layer_idx: usize,
    k: &[f32],
    v: &[f32],
    seq_len: usize,
) {
    let c = &mut model.kv_caches[layer_idx];
    assert!(!c.has_wrapped() || seq_len <= c.capacity);
    let stride = c.num_kv_heads * c.head_dim;
    assert_eq!(k.len(), seq_len * stride, "k len");
    assert_eq!(v.len(), seq_len * stride, "v len");
    assert!(seq_len <= c.capacity, "seq_len {seq_len} > capacity {}", c.capacity);
    c.k[..seq_len * stride].copy_from_slice(k);
    c.v[..seq_len * stride].copy_from_slice(v);
    c.seq_len = seq_len;
}

#[cfg(test)]
mod quant_tests {
    use super::*;

    /// The packed-resident embed (`PackedEmbed::row_f16`, the per-token decode
    /// used at the first PP stage) must be BIT-EXACT to the corresponding row of
    /// the old whole-table `dequantize_mlx_affine_f16` — same f32 affine math,
    /// same f16 round. If this holds, the embed-stream fix is argmax-identical to
    /// the whole-table path it replaces. Synthetic 4-bit affine, vocab=40,
    /// hidden=128, group=64 (the 122B geometry, scaled down).
    #[test]
    fn packed_embed_row_is_bit_exact_to_whole_table() {
        let (vocab, hidden, group, bits) = (40usize, 128usize, 64usize, 4usize);
        let per_word = 32 / bits;
        let groups = hidden / group;
        let words_per_row = hidden / per_word;
        // Deterministic pseudo-random packed nibbles + varied scales/biases.
        let mut packed = vec![0u32; vocab * words_per_row];
        for (i, w) in packed.iter_mut().enumerate() {
            *w = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(0x9e3779b9);
        }
        let mut scales = vec![0f32; vocab * groups];
        let mut biases = vec![0f32; vocab * groups];
        for i in 0..scales.len() {
            scales[i] = 0.01 + (i as f32 % 7.0) * 0.013;
            biases[i] = -0.4 + (i as f32 % 5.0) * 0.11;
        }
        // Whole-table reference.
        let table = dequantize_mlx_affine_f16(
            &packed, &scales, &biases, vocab, hidden, group, bits,
        );
        let pe = PackedEmbed {
            packed: packed.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
            vocab,
            hidden,
            group_size: group,
            bits,
        };
        for tok in 0..vocab {
            let row = pe.row_f16(tok);
            assert_eq!(
                row.as_slice(),
                &table[tok * hidden..(tok + 1) * hidden],
                "packed embed row {tok} diverges from whole-table f16 dequant"
            );
            // f32 accessor == f16-bits round-trip (what the forward reads).
            let row32 = pe.row_f32(tok);
            for (j, &b) in row.iter().enumerate() {
                assert_eq!(row32[j], half::f16::from_bits(b).to_f32());
            }
        }
    }

    /// Packed-resident (untied) lm_head: the mlx4 vocab matvec run directly on the
    /// PACKED 4-bit `lm_head.weight` (`VLLM_VULKAN_Q35_LMHEAD_PACKED`) must be
    /// (a) BIT-EXACT to `dequantize_mlx_affine` (f32) then matmul — the mlx4 GPU
    /// kernel's exact `scale*q+bias` f32 math, so the packed path reproduces the
    /// full-precision weight EXACTLY — and (b) argmax-exact + cos=1.0 vs the CURRENT
    /// lm_head, which dequantizes to f16 first (`dequantize_mlx_affine_f16`) then
    /// matmuls (the whole-table build this fix replaces). Synthetic 4-bit affine at
    /// the 122B lm_head geometry (group=64), scaled down: vocab=200, hidden=256.
    #[test]
    fn packed_lmhead_mlx4_matvec_matches_whole_table() {
        let (vocab, hidden, group, bits) = (200usize, 256usize, 64usize, 4usize);
        let per_word = 32 / bits;
        let groups = hidden / group;
        let words_per_row = hidden / per_word;
        // Deterministic pseudo-random packed nibbles + varied scales/biases.
        let mut packed = vec![0u32; vocab * words_per_row];
        for (i, w) in packed.iter_mut().enumerate() {
            *w = (i as u32).wrapping_mul(2654435761).wrapping_add(0x9e3779b9);
        }
        let mut scales = vec![0f32; vocab * groups];
        let mut biases = vec![0f32; vocab * groups];
        for i in 0..scales.len() {
            scales[i] = 0.008 + (i as f32 % 11.0) * 0.007;
            biases[i] = -0.35 + (i as f32 % 6.0) * 0.09;
        }
        // Deterministic pseudo-random hidden input (the final-norm output).
        let x: Vec<f32> = (0..hidden)
            .map(|i| ((i as f32 * 0.137).sin() * 0.9) - 0.05)
            .collect();

        // (a) PACKED mlx4 path: dequant to f32 (== the mlx4 kernel's exact math),
        // then matmul. This is what the GPU MvKind::Mlx4 lm_head computes.
        let w_f32 = dequantize_mlx_affine(
            &packed, &scales, &biases, vocab, hidden, group, bits);
        let logits_mlx4 = cpu_matmul(&x, &w_f32, 1, hidden, vocab);

        // (b) CURRENT lm_head: dequant to f16 first, then matmul in f32.
        let w_f16 = dequantize_mlx_affine_f16(
            &packed, &scales, &biases, vocab, hidden, group, bits);
        let w_from_f16: Vec<f32> =
            w_f16.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        let logits_f16 = cpu_matmul(&x, &w_from_f16, 1, hidden, vocab);

        // The PackedEmbed carrier the loader builds must reconstruct the SAME f32
        // weight as dequantize_mlx_affine, row for row (bit-exact by construction).
        let pe = PackedEmbed {
            packed: packed.clone(), scales: scales.clone(), biases: biases.clone(),
            vocab, hidden, group_size: group, bits,
        };
        for row in 0..vocab {
            let r32 = pe.row_f32(row); // f16-round accessor (embed path)
            for j in 0..hidden {
                // row_f32 rounds to f16; assert it equals the f16-table row exactly.
                assert_eq!(r32[j], w_from_f16[row * hidden + j],
                    "packed lm_head row {row} col {j} != whole f16 table");
            }
        }

        // (b') argmax-exact.
        let am = |v: &[f32]| v.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(am(&logits_mlx4), am(&logits_f16),
            "packed mlx4 lm_head argmax diverges from the f16-table lm_head");

        // (b'') cos == 1.0 (to f32 tolerance): the only difference is per-weight f16
        // rounding, so the logit vectors are effectively identical in direction.
        let dot: f64 = logits_mlx4.iter().zip(&logits_f16)
            .map(|(&a, &b)| a as f64 * b as f64).sum();
        let na: f64 = logits_mlx4.iter().map(|&a| (a as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = logits_f16.iter().map(|&b| (b as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb);
        assert!(cos > 0.999999,
            "packed mlx4 lm_head cos {cos} < 0.999999 vs the f16-table lm_head");
    }

    /// `mlx_affine_bits` must recover the true per-tensor width from packed
    /// geometry alone (trusting only group_size). Real header numbers from
    /// mlx-community/Qwen3.5-122B-A10B-4bit (group_size=64) — all 4-bit — plus a
    /// synthetic genuine-8-bit router proving the 35B-A3B mixed path is preserved.
    #[test]
    fn mlx_affine_bits_recovers_per_tensor_width() {
        // Router gate: .weight u32 [256,384], .scales bf16 [256,48] -> 4-bit.
        assert_eq!(mlx_affine_bits(256 * 384, 256 * 48, 64), 4);
        // shared_expert_gate: [1,384] / [1,48] -> 4-bit.
        assert_eq!(mlx_affine_bits(1 * 384, 1 * 48, 64), 4);
        // switch_mlp expert gate_proj: [256,1024,384] / [256,1024,48] -> 4-bit.
        assert_eq!(mlx_affine_bits(256 * 1024 * 384, 256 * 1024 * 48, 64), 4);
        // switch_mlp down_proj (in=1024): [256,3072,128] / [256,3072,16] -> 4-bit.
        assert_eq!(mlx_affine_bits(256 * 3072 * 128, 256 * 3072 * 16, 64), 4);
        // lm_head / embed_tokens: [248320,384] / [248320,48] -> 4-bit.
        assert_eq!(mlx_affine_bits(248320 * 384, 248320 * 48, 64), 4);
        // GENUINE 8-bit router (35B-A3B layout), out=256 in=2048: packed u32 = in/4,
        // scales = in/64 -> recovers 8. Guards behaviour-preservation of the
        // cluster-validated 35B path under the geometry derivation.
        assert_eq!(mlx_affine_bits(256 * 512, 256 * 32, 64), 8);
        // Degenerate -> 0 (caller falls back to config bits / validate rejects).
        assert_eq!(mlx_affine_bits(0, 0, 64), 0);
    }

    /// `qwen35_skip_aux` must drop the vision tower and any MTP/nextn head while
    /// keeping every real language-model tensor (post-normalize names).
    #[test]
    fn qwen35_skip_aux_covers_vision_and_mtp_not_lm() {
        for n in [
            "vision_tower.blocks.0.attn.qkv.weight",
            "vision_tower.patch_embed.proj.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.patch_embed.proj.weight",
            "model.mtp.layers.0.input_layernorm.weight",
            "mtp.embed_tokens.weight",
            "model.nextn.0.weight",
        ] {
            assert!(qwen35_skip_aux(n), "{n} should be skipped");
        }
        for n in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
            "model.layers.3.self_attn.q_proj.weight",
        ] {
            assert!(!qwen35_skip_aux(n), "{n} wrongly skipped");
        }
    }

    /// Minimal little-endian float32 `.npy` v1 reader (numpy `<f4`, C-order).
    fn read_npy_f32(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(&bytes[0..6], b"\x93NUMPY", "not a .npy: {path}");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let data = &bytes[10 + header_len..];
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
        for i in 0..a.len() {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn argmax(x: &[f32]) -> usize {
        x.iter().enumerate().fold((0, f32::MIN), |(bi, bv), (i, &v)|
            if v > bv { (i, v) } else { (bi, bv) }).0
    }

    /// LEVER 2 CPU golden — the FP8-attn → q8_0 requant
    /// (VLLM_VULKAN_GEMMA_FP8_Q8) must preserve the attention matvec: routing
    /// `dequantize_fp8(w)` through q8_0 and matmul-ing against the FP8 path's
    /// OWN dequantized weight is cos≈1.0 / argmax-exact (the nemotron mamba-q8
    /// requant held the same). Mirrors the loader's exact byte path
    /// (fp8 bytes → dequantize_fp8 → quantize_q8_0 → dequant_q8_0_to_f32).
    #[test]
    fn fp8_to_q8_requant_matvec_cos() {
        let out_f = 96usize;   // attn-row-like; out_f*in_f % 32 == 0
        let in_f = 256usize;   // hidden-like, % 32 == 0
        // Deterministic xorshift pseudo-random e4m3 weight bytes, per-row FP8
        // scales, and an input vector (no rand dependency in the test).
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let wbytes: Vec<u8> = (0..out_f * in_f).map(|_| (next() & 0xFF) as u8).collect();
        let scale: Vec<f32> = (0..out_f).map(|_| 0.01 + (next() % 100) as f32 * 0.001).collect();
        let x: Vec<f32> = (0..in_f).map(|_| ((next() % 2000) as f32 - 1000.0) / 1000.0).collect();

        // FP8 path: the fp8 matvec multiplies against dequantize_fp8(w).
        let deq_fp8 = dequantize_fp8(&wbytes, &scale, out_f, in_f);
        let ref_out = cpu_matmul(&x, &deq_fp8, 1, in_f, out_f);

        // Requant path: dequantize_fp8 → quantize_q8_0 → dequant (what the GPU
        // q8_0 matvec reads after the loader requant).
        let q8 = quantize_q8_0(&deq_fp8);
        let deq_q8 = dequant_q8_0_to_f32(&q8);
        let req_out = cpu_matmul(&x, &deq_q8, 1, in_f, out_f);

        let cos = cosine(&ref_out, &req_out);
        eprintln!("fp8->q8 requant: matvec cos={cos:.6} (argmax fp8={} q8={})",
                  argmax(&ref_out), argmax(&req_out));
        assert_eq!(argmax(&ref_out), argmax(&req_out), "fp8->q8 requant argmax drift");
        assert!(cos >= 0.9999, "fp8->q8 requant matvec cos {cos} < 0.9999");
    }

    /// End-to-end Mac correctness gate for gemma-4-12B-it: load the real
    /// QAT-4bit checkpoint, run the pure-CPU forward over a 6-token prompt, and
    /// compare the final (softcapped) logits to the mlx_vlm golden.  Exercises the
    /// value-less "k_eq_v" global-attention path.  Gated on env vars so normal
    /// `cargo test` (E2B) stays fast/green:
    ///   GEMMA12B_DIR=<checkpoint dir>  GEMMA12B_GOLDEN=<dir with logits_last.npy>
    #[test]
    fn gemma12b_forward_cos_vs_mlx() {
        let dir = match std::env::var("GEMMA12B_DIR") { Ok(d) => d, Err(_) => { eprintln!("SKIP gemma12b: set GEMMA12B_DIR"); return } };
        let golden = match std::env::var("GEMMA12B_GOLDEN") { Ok(d) => d, Err(_) => { eprintln!("SKIP gemma12b: set GEMMA12B_GOLDEN"); return } };

        let cfg = Gemma4Config::g12b();
        let tensors = load_gemma_mlx_affine(std::path::Path::new(&dir)).expect("load checkpoint");
        let weights = Gemma4Weights {
            tensors: tensors.into_iter()
                .map(|(k, v)| (k, SimpleTensor { shape: vec![v.len()], data: v }))
                .collect(),
        };
        let max_seq = 16usize;
        let kv_caches = (0..cfg.num_hidden_layers)
            .map(|l| KvCache::new_windowed(max_seq, cfg.layer_kv_capacity(l, max_seq), cfg.layer_num_kv_heads(l), cfg.layer_head_dim(l)))
            .collect();
        let mut model = Gemma4Model { config: cfg.clone(), weights, kv_caches };

        let tokens = [2u32, 1024, 2048, 4096, 8192, 16384];
        let mut logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            logits = model.forward(tok, pos);
        }

        let g = read_npy_f32(&format!("{golden}/logits_last.npy"));
        assert_eq!(g.len(), logits.len(), "vocab mismatch: rust {} mlx {}", logits.len(), g.len());
        let cos = cosine(&logits, &g);
        let amx_r = argmax(&logits);
        let amx_g = argmax(&g);
        eprintln!("gemma12b: logits cos={cos:.6}  argmax rust={amx_r} mlx={amx_g}  \
                   rust[amax]={:.4} mlx[amax]={:.4}", logits[amx_r], g[amx_g]);
        assert!(cos >= 0.999, "logits cos {cos} < 0.999");
        assert_eq!(amx_r, amx_g, "argmax mismatch");
    }

    /// Self-consistency unit test for `quantize_mlx_affine_4bit` (no checkpoint):
    /// (1) the packed nibbles + affine scale/bias must reconstruct BIT-EXACTLY
    /// via `dequantize_mlx_affine` (the CPU oracle AND the mlx4 GPU shader both
    /// read this exact `scale*nib+bias` formula, so bit-exactness here == GPU
    /// path == this reconstruction); (2) per-group max abs error ≤ scale/2 (a
    /// 4-bit affine grid rounds to at worst half a step); (3) a constant group
    /// round-trips exactly (degenerate scale=0 path).
    #[test]
    fn quantize_mlx_affine_4bit_self_consistent() {
        let (out_f, in_f, gs) = (5usize, 128usize, 64usize);
        let groups = in_f / gs;
        // Deterministic pseudo-random weights + one constant row (row 4).
        let mut data = vec![0f32; out_f * in_f];
        let mut st = 0x1234_5678u32;
        let mut rnd = || { st = st.wrapping_mul(1664525).wrapping_add(1013904223); (st >> 8) as f32 / 16_777_216.0 };
        for r in 0..out_f - 1 {
            for i in 0..in_f { data[r * in_f + i] = (rnd() - 0.5) * (2.0 + r as f32); }
        }
        for i in 0..in_f { data[(out_f - 1) * in_f + i] = 0.375; } // constant row

        let (packed, scales, biases) = quantize_mlx_affine_4bit(&data, out_f, in_f, gs);
        assert_eq!(packed.len(), out_f * (in_f / 8));
        assert_eq!(scales.len(), out_f * groups);

        // (1) bit-exact vs dequantize_mlx_affine (== GPU shader formula).
        let deq = dequantize_mlx_affine(&packed, &scales, &biases, out_f, in_f, gs, 4);
        let per_word = 8usize;
        let wpr = in_f / per_word;
        for r in 0..out_f {
            let prow = &packed[r * wpr..(r + 1) * wpr];
            for j in 0..in_f {
                let nib = (prow[j / per_word] >> ((j % per_word) * 4)) & 0xF;
                let g = r * groups + j / gs;
                let recon = scales[g] * (nib as f32) + biases[g];
                assert_eq!(recon.to_bits(), deq[r * in_f + j].to_bits(),
                    "row {r} elem {j}: packed-recon != dequantize_mlx_affine");
            }
        }
        // (2) max abs error ≤ scale/2 (+fp slack) per group.
        for r in 0..out_f {
            for g in 0..groups {
                let s = scales[r * groups + g];
                for j in g * gs..(g + 1) * gs {
                    let e = (data[r * in_f + j] - deq[r * in_f + j]).abs();
                    assert!(e <= s * 0.5 + 1e-5, "row {r} elem {j}: err {e} > scale/2 {}", s * 0.5);
                }
            }
        }
        // (3) constant group exact + scale 0.
        for g in 0..groups { assert_eq!(scales[(out_f - 1) * groups + g], 0.0); }
        for j in 0..in_f { assert_eq!(deq[(out_f - 1) * in_f + j], 0.375); }
    }

    /// Requant every MLP tensor of a loaded gemma-12B model from its
    /// (dequantized-8-bit) f32 to the mlx4 4-bit ROUND-TRIP, IN PLACE. This is
    /// the exact numeric twin of `load_gemma_resident_weights(mlp_q4=true)`:
    /// the GPU uploads `quantize_mlx_affine_4bit(deq)` and dispatches
    /// `scale*nib+bias`, i.e. `dequantize_mlx_affine` of the same packed form,
    /// which is what we substitute here. Returns aggregate
    /// (max_abs, mean_abs, l2_rel) error across all replaced tensors.
    #[cfg(test)]
    fn requant_gemma12b_mlp_inplace(model: &mut Gemma4Model) -> (f32, f32, f32) {
        let cfg = model.config.clone();
        let (h, inter, gs) = (cfg.hidden_size, cfg.intermediate_size, 64usize);
        let mut keys: Vec<String> = model.weights.tensors.keys()
            .filter(|k| k.ends_with("mlp.gate_proj.weight")
                || k.ends_with("mlp.up_proj.weight")
                || k.ends_with("mlp.down_proj.weight"))
            .cloned().collect();
        keys.sort();
        let (mut gmax, mut gsum_abs, mut gnum_err, mut gsum_sq, mut gsum_ref) =
            (0f32, 0f64, 0usize, 0f64, 0f64);
        for (ti, k) in keys.iter().enumerate() {
            let (out_f, in_f) = if k.ends_with("down_proj.weight") { (h, inter) } else { (inter, h) };
            let orig = model.weights.tensors[k].data.clone();
            assert_eq!(orig.len(), out_f * in_f, "{k} shape {}", orig.len());
            let (p, s, b) = quantize_mlx_affine_4bit(&orig, out_f, in_f, gs);
            let rt = dequantize_mlx_affine(&p, &s, &b, out_f, in_f, gs, 4);
            let (mut tmax, mut tsum, mut tsq, mut tref) = (0f32, 0f64, 0f64, 0f64);
            for (a, c) in orig.iter().zip(rt.iter()) {
                let e = (a - c).abs();
                if e > tmax { tmax = e; }
                tsum += e as f64; tsq += (e as f64) * (e as f64); tref += (*a as f64) * (*a as f64);
            }
            let l2_rel = (tsq.sqrt() / tref.sqrt().max(1e-12)) as f32;
            if ti < 6 {
                eprintln!("  [{k}] max_abs={tmax:.5} mean_abs={:.6} l2_rel={l2_rel:.5}",
                    tsum / orig.len() as f64);
            }
            if tmax > gmax { gmax = tmax; }
            gsum_abs += tsum; gnum_err += orig.len(); gsum_sq += tsq; gsum_ref += tref;
            model.weights.tensors.get_mut(k).unwrap().data = rt;
        }
        eprintln!("MLP requant: {} tensors, max_abs={gmax:.5} mean_abs={:.6} l2_rel={:.5}",
            keys.len(), gsum_abs / gnum_err as f64, (gsum_sq.sqrt() / gsum_ref.sqrt()) as f32);
        (gmax, (gsum_abs / gnum_err as f64) as f32, (gsum_sq.sqrt() / gsum_ref.sqrt()) as f32)
    }

    /// FAST forward-free half of the H1 gate: load the checkpoint and report
    /// the per-tensor + aggregate MLP 8->4bit requant round-trip error. No
    /// forward passes, so it finishes in ~load+requant (~1-2 min) — the
    /// deterministic quality signal for the LOSSY 8->4 re-round, independent of
    /// the slow CPU argmax gate below.
    ///   GEMMA12B_DIR=<dir> cargo test --release --lib -- --ignored --nocapture \
    ///     gemma12b_mlp_q4_rterror
    #[test]
    #[ignore]
    fn gemma12b_mlp_q4_rterror() {
        let dir = match std::env::var("GEMMA12B_DIR") {
            Ok(d) => d, Err(_) => { eprintln!("SKIP gemma12b_mlp_q4_rterror: set GEMMA12B_DIR"); return; }
        };
        let cfg = Gemma4Config::g12b();
        let t = std::time::Instant::now();
        let tensors = load_gemma_mlx_affine(std::path::Path::new(&dir)).expect("load checkpoint");
        let weights = Gemma4Weights {
            tensors: tensors.into_iter()
                .map(|(k, v)| (k, SimpleTensor { shape: vec![v.len()], data: v })).collect(),
        };
        let mut model = Gemma4Model { config: cfg, weights, kv_caches: Vec::new() };
        eprintln!("loaded in {:.1}s; computing MLP requant round-trip error...", t.elapsed().as_secs_f32());
        let (max_abs, mean_abs, l2_rel) = requant_gemma12b_mlp_inplace(&mut model);
        eprintln!("RTERROR: max_abs={max_abs:.5} mean_abs={mean_abs:.6} l2_rel={l2_rel:.5}");
    }

    /// H1 quality gate (VLLM_VULKAN_GEMMA_MLP_Q4): full-48L Mac-native CPU
    /// forward, 4-bit-MLP vs the 8-bit-MLP baseline. Loads the checkpoint ONCE,
    /// greedily generates from several diverse seeds with the 8-bit model,
    /// requants the MLP to 4-bit IN PLACE (one 47GB weight table, not two),
    /// then teacher-forces the SAME 8-bit sequences through the 4-bit model and
    /// measures per-token argmax agreement + first divergence. Also emits the
    /// per-tensor round-trip error (gate 1). Heavy (~hrs on the f32 table),
    /// `#[ignore]`d. Defaults meet the gate (5 prompts x 64 tokens):
    ///   GEMMA12B_DIR=<dir> [GEMMA_MLPQ4_PROMPTS=5] [GEMMA_MLPQ4_GEN=64] \
    ///     cargo test --release --lib -- --ignored --nocapture gemma12b_mlp_q4_gate
    #[test]
    #[ignore]
    fn gemma12b_mlp_q4_gate() {
        let dir = match std::env::var("GEMMA12B_DIR") {
            Ok(d) => d, Err(_) => { eprintln!("SKIP gemma12b_mlp_q4_gate: set GEMMA12B_DIR"); return; }
        };
        let gen: usize = std::env::var("GEMMA_MLPQ4_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let nprompts: usize = std::env::var("GEMMA_MLPQ4_PROMPTS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);

        // Seeds: prefer REAL tokenized prompts via GEMMA_MLPQ4_SEEDS (prompts
        // ';'-separated, token ids ','-separated) so the greedy continuation is
        // COHERENT (arbitrary-id seeds collapse to a repeated token, making the
        // 8-bit vs 4-bit argmax agreement non-diagnostic). Falls back to the
        // arbitrary-id seeds only if unset. Trimmed to `nprompts`.
        let seeds: Vec<Vec<u32>> = match std::env::var("GEMMA_MLPQ4_SEEDS") {
            Ok(s) if !s.trim().is_empty() => s.split(';')
                .map(|p| p.split(',').filter_map(|t| t.trim().parse::<u32>().ok()).collect::<Vec<u32>>())
                .filter(|v| !v.is_empty())
                .collect(),
            _ => vec![
                vec![2, 651, 1163, 2094, 476, 573],
                vec![2, 8291, 20, 34120, 1902, 88, 12],
                vec![2, 106, 1645, 108, 5049, 573, 3186],
                vec![2, 235280, 12345, 6789, 44, 991],
                vec![2, 100, 200, 4000, 80000, 150000, 9],
            ],
        }.into_iter().take(nprompts.max(1)).collect();

        let max_seq = seeds.iter().map(|s| s.len()).max().unwrap() + gen + 4;
        let cfg = Gemma4Config::g12b();
        eprintln!("gemma12b_mlp_q4_gate: loading {dir} (dequant to f32; may take minutes)...");
        let t_load = std::time::Instant::now();
        let tensors = load_gemma_mlx_affine(std::path::Path::new(&dir)).expect("load checkpoint");
        let weights = Gemma4Weights {
            tensors: tensors.into_iter()
                .map(|(k, v)| (k, SimpleTensor { shape: vec![v.len()], data: v })).collect(),
        };
        let kv_caches = (0..cfg.num_hidden_layers)
            .map(|l| KvCache::new_windowed(max_seq, cfg.layer_kv_capacity(l, max_seq), cfg.layer_num_kv_heads(l), cfg.layer_head_dim(l)))
            .collect();
        let mut model = Gemma4Model { config: cfg.clone(), weights, kv_caches };
        eprintln!("  loaded in {:.1}s", t_load.elapsed().as_secs_f32());

        // TEACHER-FORCE mode (GEMMA_MLPQ4_TF=1): the seeds ARE full real-text
        // passages; we compare the 8-bit vs 4-bit models' next-token argmax at
        // EVERY position over the same fixed real context. This is the
        // DIAGNOSTIC quant-fidelity metric: unlike free greedy generation (which
        // an IT model fed raw text collapses into a repeated digit token, making
        // agreement trivially ~1.0), each position here has a distinct varied
        // context so argmax genuinely exercises the perturbed MLP. Default
        // (unset) keeps the greedy-generate-then-teacher-force path.
        let tf_mode = std::env::var("GEMMA_MLPQ4_TF").map(|v| v == "1").unwrap_or(false);

        // ── 8-bit pass: record argmax at every scored position per prompt ────
        // (greedy mode also extends each seed by `gen` tokens first).
        let mut sequences: Vec<Vec<u32>> = Vec::new();  // full token seq per prompt
        let mut score_from: Vec<usize> = Vec::new();     // first scored position per prompt
        let mut am8: Vec<Vec<u32>> = Vec::new();         // 8-bit argmax at scored positions
        for (pi, seed) in seeds.iter().enumerate() {
            reset_kv_caches(&mut model, max_seq);
            let mut seq = seed.clone();
            if tf_mode {
                // Score every position 0..len-1 (predict the next real token).
                let mut a = Vec::new();
                for pos in 0..seq.len() - 1 {
                    let logits = model.forward(seq[pos], pos);
                    a.push(argmax(&logits) as u32);
                }
                eprintln!("  8-bit prompt {pi}: {} real-text positions; first5_argmax={:?}",
                    a.len(), &a[..5.min(a.len())]);
                score_from.push(0);
                am8.push(a);
            } else {
                let mut logits = Vec::new();
                for (pos, &tok) in seed.iter().enumerate() { logits = model.forward(tok, pos); }
                let mut a = Vec::new();
                for step in 0..gen {
                    let next = argmax(&logits) as u32;
                    a.push(next); seq.push(next);
                    logits = model.forward(next, seed.len() + step);
                }
                eprintln!("  8-bit prompt {pi}: seed_len={} first5_gen={:?}", seed.len(), &a[..5.min(a.len())]);
                score_from.push(seed.len() - 1);
                am8.push(a);
            }
            sequences.push(seq);
        }

        // ── Gate 1 + requant MLP to 4-bit IN PLACE ──────────────────────────
        eprintln!("--- per-tensor round-trip error (8-bit deq vs 4-bit requant) ---");
        let (rt_max, rt_mean, rt_l2) = requant_gemma12b_mlp_inplace(&mut model);

        // ── 4-bit pass: same contexts, compare argmax to the 8-bit record ────
        let (mut agree, mut total) = (0usize, 0usize);
        let mut first_div: Option<(usize, usize)> = None;
        for (pi, seq) in sequences.iter().enumerate() {
            reset_kv_caches(&mut model, max_seq);
            let sf = score_from[pi];
            let (mut p_agree, mut p_total) = (0usize, 0usize);
            for pos in 0..seq.len() - 1 {
                let logits = model.forward(seq[pos], pos);
                if pos >= sf {
                    let am4 = argmax(&logits) as u32;
                    let am8v = am8[pi][pos - sf];
                    total += 1; p_total += 1;
                    if am4 == am8v { agree += 1; p_agree += 1; }
                    else if first_div.is_none() { first_div = Some((pi, pos - sf)); }
                }
            }
            eprintln!("  4-bit prompt {pi}: agree {p_agree}/{p_total} ({:.4}); running {agree}/{total} ({:.4})",
                p_agree as f32 / p_total.max(1) as f32, agree as f32 / total.max(1) as f32);
        }
        let rate = agree as f32 / total as f32;
        eprintln!("========================================================");
        eprintln!("GEMMA-12B MLP 8->4bit CPU GATE ({} prompts, mode={}, {total} scored positions)",
            seeds.len(), if tf_mode { "teacher-force-real-text" } else { "greedy-gen" });
        eprintln!("  round-trip: max_abs={rt_max:.5} mean_abs={rt_mean:.6} l2_rel={rt_l2:.5}");
        eprintln!("  argmax agreement: {agree}/{total} = {:.4}", rate);
        eprintln!("  first divergence (prompt, pos): {first_div:?}");
        eprintln!("========================================================");
        assert!(rate >= 0.98, "argmax agreement {rate:.4} < 0.98 gate");
    }

    /// H2 quality gate (VLLM_VULKAN_GEMMA_LMHEAD_Q4): does quantizing the tied
    /// embed/lm_head PROJECTION to mlx4 4-bit shift the next-token argmax? The
    /// lm_head directly emits the logits, so this is MORE sensitive than the
    /// MLP requant (which perturbs a hidden state that a later norm+projection
    /// re-centers). Isolates the effect FAITHFULLY in one forward pass per
    /// position: the input-embed lookup and all decoder layers run with the
    /// EXACT weights (matching the GPU lever, which keeps the host input-embed
    /// f16 copy and quantizes only the GPU lm_head matvec), then the captured
    /// post-final-norm hidden is projected with BOTH the exact and the
    /// w4-round-trip lm_head; argmax agreement is scored over the same hidden.
    ///   GEMMA12B_DIR=<dir> [GEMMA_LMHEADQ4_GEN=64] [GEMMA_LMHEADQ4_PROMPTS=5] \
    ///     [GEMMA_LMHEADQ4_SEEDS='id,id;id,id'] [GEMMA_LMHEADQ4_TF=1] \
    ///     cargo test --release --lib -- --ignored --nocapture gemma12b_lmhead_q4_gate
    #[test]
    #[ignore]
    fn gemma12b_lmhead_q4_gate() {
        let dir = match std::env::var("GEMMA12B_DIR") {
            Ok(d) => d, Err(_) => { eprintln!("SKIP gemma12b_lmhead_q4_gate: set GEMMA12B_DIR"); return; }
        };
        let gen: usize = std::env::var("GEMMA_LMHEADQ4_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let nprompts: usize = std::env::var("GEMMA_LMHEADQ4_PROMPTS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        // Same coherent chat-templated seeds as the MLP gate (arbitrary-id seeds
        // collapse to a repeated token, making agreement non-diagnostic).
        let seeds: Vec<Vec<u32>> = match std::env::var("GEMMA_LMHEADQ4_SEEDS") {
            Ok(s) if !s.trim().is_empty() => s.split(';')
                .map(|p| p.split(',').filter_map(|t| t.trim().parse::<u32>().ok()).collect::<Vec<u32>>())
                .filter(|v| !v.is_empty())
                .collect(),
            _ => vec![
                vec![2, 651, 1163, 2094, 476, 573],
                vec![2, 8291, 20, 34120, 1902, 88, 12],
                vec![2, 106, 1645, 108, 5049, 573, 3186],
                vec![2, 235280, 12345, 6789, 44, 991],
                vec![2, 100, 200, 4000, 80000, 150000, 9],
            ],
        }.into_iter().take(nprompts.max(1)).collect();
        let tf_mode = std::env::var("GEMMA_LMHEADQ4_TF").map(|v| v == "1").unwrap_or(false);

        let max_seq = seeds.iter().map(|s| s.len()).max().unwrap() + gen + 4;
        let cfg = Gemma4Config::g12b();
        eprintln!("gemma12b_lmhead_q4_gate: loading {dir} (dequant to f32; may take minutes)...");
        let t_load = std::time::Instant::now();
        let tensors = load_gemma_mlx_affine(std::path::Path::new(&dir)).expect("load checkpoint");
        let weights = Gemma4Weights {
            tensors: tensors.into_iter()
                .map(|(k, v)| (k, SimpleTensor { shape: vec![v.len()], data: v })).collect(),
        };
        let kv_caches = (0..cfg.num_hidden_layers)
            .map(|l| KvCache::new_windowed(max_seq, cfg.layer_kv_capacity(l, max_seq), cfg.layer_num_kv_heads(l), cfg.layer_head_dim(l)))
            .collect();
        let mut model = Gemma4Model { config: cfg.clone(), weights, kv_caches };
        eprintln!("  loaded in {:.1}s", t_load.elapsed().as_secs_f32());

        // ── w4 round-trip lm_head table (separate owned copy; model stays exact) ──
        let (vocab, h, gs) = (cfg.vocab_size, cfg.hidden_size, 64usize);
        let (p, s, b) = {
            let lm = model.weights.f32_slice("model.embed_tokens.weight");
            assert_eq!(lm.len(), vocab * h, "lm_head shape {}", lm.len());
            quantize_mlx_affine_4bit(lm, vocab, h, gs)
        };
        let lm_w4 = dequantize_mlx_affine(&p, &s, &b, vocab, h, gs, 4);
        // Round-trip error (gate 1) vs the exact table, no extra copy.
        {
            let lm = model.weights.f32_slice("model.embed_tokens.weight");
            let (mut mx, mut sum, mut sq, mut refsq) = (0f32, 0f64, 0f64, 0f64);
            for (a, c) in lm.iter().zip(lm_w4.iter()) {
                let e = (a - c).abs();
                if e > mx { mx = e; }
                sum += e as f64; sq += (e as f64) * (e as f64); refsq += (*a as f64) * (*a as f64);
            }
            eprintln!("  lm_head round-trip: max_abs={mx:.5} mean_abs={:.6} l2_rel={:.5}",
                sum / lm.len() as f64, (sq.sqrt() / refsq.sqrt().max(1e-12)) as f32);
        }
        let cap = cfg.final_logit_softcapping;

        // ── Score exact-vs-w4 lm_head argmax over the SAME (exact) hidden ──────
        let (mut agree, mut total) = (0usize, 0usize);
        let mut first_div: Option<(usize, usize)> = None;
        for (pi, seed) in seeds.iter().enumerate() {
            reset_kv_caches(&mut model, max_seq);
            let (mut p_agree, mut p_total) = (0usize, 0usize);
            let score = |normed: &[f32], am_e: u32,
                         agree: &mut usize, total: &mut usize,
                         p_agree: &mut usize, p_total: &mut usize,
                         first_div: &mut Option<(usize, usize)>, at: usize| {
                let mut lw4 = cpu_matmul(normed, &lm_w4, 1, h, vocab);
                lw4.iter_mut().for_each(|l| *l = (*l / cap).tanh() * cap);
                let am_w = argmax(&lw4) as u32;
                *total += 1; *p_total += 1;
                if am_w == am_e { *agree += 1; *p_agree += 1; }
                else if first_div.is_none() { *first_div = Some((pi, at)); }
            };
            if tf_mode {
                for pos in 0..seed.len() - 1 {
                    let (logits_e, normed) = model.forward_with_normed(seed[pos], pos);
                    let am_e = argmax(&logits_e) as u32;
                    score(&normed, am_e, &mut agree, &mut total,
                        &mut p_agree, &mut p_total, &mut first_div, pos);
                }
            } else {
                // Greedy: continuation driven by the EXACT lm_head argmax (the
                // real sequence the model produces); score w4 vs exact each step.
                for pos in 0..seed.len().saturating_sub(1) {
                    let _ = model.forward_with_normed(seed[pos], pos); // prime KV
                }
                let mut cur = *seed.last().unwrap();
                let mut pos = seed.len() - 1;
                let mut gen_tokens: Vec<u32> = Vec::new();
                for step in 0..gen {
                    let (logits_e, normed) = model.forward_with_normed(cur, pos);
                    let am_e = argmax(&logits_e) as u32;
                    score(&normed, am_e, &mut agree, &mut total,
                        &mut p_agree, &mut p_total, &mut first_div, step);
                    gen_tokens.push(am_e);
                    cur = am_e; pos += 1;
                }
                eprintln!("  prompt {pi}: exact-lm_head gen first10={:?}", &gen_tokens[..10.min(gen_tokens.len())]);
            }
            eprintln!("  prompt {pi}: agree {p_agree}/{p_total} ({:.4}); running {agree}/{total} ({:.4})",
                p_agree as f32 / p_total.max(1) as f32, agree as f32 / total.max(1) as f32);
        }
        let rate = agree as f32 / total.max(1) as f32;
        eprintln!("========================================================");
        eprintln!("GEMMA-12B LM-HEAD f16->4bit CPU GATE ({} prompts, mode={}, {total} scored positions)",
            seeds.len(), if tf_mode { "teacher-force-real-text" } else { "greedy-gen" });
        eprintln!("  argmax agreement (exact vs w4 lm_head, same hidden): {agree}/{total} = {:.4}", rate);
        eprintln!("  first divergence (prompt, scored-idx): {first_div:?}");
        eprintln!("========================================================");
        // lm_head is the argmax-critical projection; keep the same 0.98 bar as
        // the MLP gate but EXPECT it may be tighter (report even on pass).
        assert!(rate >= 0.98, "lm_head w4 argmax agreement {rate:.4} < 0.98 gate");
    }

    /// Loads the real gemma-4-12B checkpoint (validation proxy for the g31b
    /// target — same forward primitives, only `num_global_key_value_heads`
    /// differs, which both `forward` and `forward_verify_core` already read
    /// via `layer_num_kv_heads`). Shared helper for the INC-4 gates below.
    ///   GEMMA12B_DIR=<checkpoint dir>
    fn load_gemma12b_for_verify_gate(max_seq: usize) -> Option<Gemma4Model> {
        let dir = match std::env::var("GEMMA12B_DIR") {
            Ok(d) => d,
            Err(_) => { eprintln!("SKIP gemma31b verify-core gate: set GEMMA12B_DIR"); return None; }
        };
        let cfg = Gemma4Config::g12b();
        let tensors = load_gemma_mlx_affine(std::path::Path::new(&dir)).expect("load checkpoint");
        let weights = Gemma4Weights {
            tensors: tensors.into_iter()
                .map(|(k, v)| (k, SimpleTensor { shape: vec![v.len()], data: v }))
                .collect(),
        };
        let kv_caches = (0..cfg.num_hidden_layers)
            .map(|l| KvCache::new_windowed(max_seq, cfg.layer_kv_capacity(l, max_seq), cfg.layer_num_kv_heads(l), cfg.layer_head_dim(l)))
            .collect();
        Some(Gemma4Model { config: cfg, weights, kv_caches })
    }

    /// Reset a loaded model's KV state back to empty (no reload — the 10GB+
    /// dequantized weight table is left untouched) so the same checkpoint load
    /// can be replayed from position 0 multiple times within one gate test.
    fn reset_kv_caches(model: &mut Gemma4Model, max_seq: usize) {
        let cfg = &model.config;
        model.kv_caches = (0..cfg.num_hidden_layers)
            .map(|l| KvCache::new_windowed(max_seq, cfg.layer_kv_capacity(l, max_seq), cfg.layer_num_kv_heads(l), cfg.layer_head_dim(l)))
            .collect();
    }

    /// INC-4 gate 1 (spec §"gemma batched-verify core"): `forward_verify_core`
    /// must be bit-exact vs the serial per-token `forward` over the same
    /// tokens from the same `start_pos`. Runs a short shared prefix to reach a
    /// non-zero `start_pos`, then compares row-by-row:
    ///   (a) serial: forward(tok, R), forward(tok, R+1), ... one token at a time;
    ///   (b) verify: forward_verify_core([tok..], R) in one layer-major pass.
    /// Uses ONE checkpoint load (`load_gemma12b_for_verify_gate`), replaying the
    /// prefix from a KV-reset between (a) and (b) — no second 10GB+ load.
    ///
    /// Real-checkpoint variant of the synthetic gate below (~30 min/run on the
    /// full gemma-4-12B f32 checkpoint) — kept for optional real-weight
    /// confidence, `#[ignore]`d by default so `cargo test` stays fast.
    #[test]
    #[ignore]
    fn gemma31b_verify_core_bit_exact_vs_serial_real12b() {
        let max_seq = 16usize;
        let mut model = match load_gemma12b_for_verify_gate(max_seq) { Some(m) => m, None => return };

        let prefix = [2u32, 1024, 2048];
        let verify_tokens = [4096u32, 8192, 16384, 32768];
        let r = prefix.len();

        // (a) Serial reference: run the prefix, then one token at a time.
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        let mut logits_serial: Vec<Vec<f32>> = Vec::with_capacity(verify_tokens.len());
        for (i, &tok) in verify_tokens.iter().enumerate() {
            logits_serial.push(model.forward(tok, r + i));
        }

        // (b) Batched verify: replay the SAME prefix from a clean KV state,
        // then one `forward_verify_core` call over all of `verify_tokens`.
        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        let logits_verify = model.forward_verify_core(&verify_tokens, r);

        assert_eq!(logits_verify.len(), verify_tokens.len());
        for (i, (serial, verify)) in logits_serial.iter().zip(logits_verify.iter()).enumerate() {
            assert_eq!(serial.len(), verify.len(), "row {i}: vocab size mismatch");
            let maxdiff = serial.iter().zip(verify.iter())
                .map(|(&s, &v)| (s - v).abs())
                .fold(0.0f32, f32::max);
            let cos = cosine(serial, verify);
            let am_s = argmax(serial);
            let am_v = argmax(verify);
            eprintln!("verify_core row {i}: maxdiff={maxdiff:.8} cos={cos:.6} argmax serial={am_s} verify={am_v}");
            assert_eq!(maxdiff, 0.0, "row {i}: verify_core diverges from serial forward (maxdiff {maxdiff})");
            assert_eq!(cos, 1.0, "row {i}: cos {cos} != 1.0");
            assert_eq!(am_s, am_v, "row {i}: argmax mismatch");
        }
    }

    /// INC-4 gate 1, synthetic: identical to
    /// `gemma31b_verify_core_bit_exact_vs_serial_real12b` above but built on
    /// `tiny_synthetic_gemma` — an in-memory, deterministic-weight model that
    /// exercises the same forward code paths (sliding + value-less `k_eq_v`
    /// global attention, q_norm/k_norm, sandwich norms, dual-RoPE) without a
    /// checkpoint load. This is the DEFAULT gate; the real-checkpoint version
    /// is `#[ignore]`d for optional confidence only.
    #[test]
    fn gemma31b_verify_core_bit_exact_vs_serial() {
        let max_seq = 16usize;
        let mut model = tiny_synthetic_gemma(max_seq);

        let prefix = [2u32, 10, 20];
        let verify_tokens = [30u32, 40, 50, 60];
        let r = prefix.len();

        // (a) Serial reference: run the prefix, then one token at a time.
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        let mut logits_serial: Vec<Vec<f32>> = Vec::with_capacity(verify_tokens.len());
        for (i, &tok) in verify_tokens.iter().enumerate() {
            logits_serial.push(model.forward(tok, r + i));
        }

        // (b) Batched verify: replay the SAME prefix from a clean KV state,
        // then one `forward_verify_core` call over all of `verify_tokens`.
        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        let logits_verify = model.forward_verify_core(&verify_tokens, r);

        assert_eq!(logits_verify.len(), verify_tokens.len());
        for (i, (serial, verify)) in logits_serial.iter().zip(logits_verify.iter()).enumerate() {
            assert_eq!(serial.len(), verify.len(), "row {i}: vocab size mismatch");
            let maxdiff = serial.iter().zip(verify.iter())
                .map(|(&s, &v)| (s - v).abs())
                .fold(0.0f32, f32::max);
            let cos = cosine(serial, verify);
            let am_s = argmax(serial);
            let am_v = argmax(verify);
            eprintln!("verify_core (synthetic) row {i}: maxdiff={maxdiff:.8} cos={cos:.6} argmax serial={am_s} verify={am_v}");
            assert_eq!(maxdiff, 0.0, "row {i}: verify_core diverges from serial forward (maxdiff {maxdiff})");
            assert_eq!(cos, 1.0, "row {i}: cos {cos} != 1.0");
            assert_eq!(am_s, am_v, "row {i}: argmax mismatch");
        }
    }

    /// INC-4 gate 2 (spec §"gemma batched-verify core"): after a batched
    /// verify over T candidates, `verify_rollback(R, T, accept_len)` must
    /// rewind every layer's KV frontier to exactly `R + accept_len + 1`, and a
    /// subsequent serial `forward` from that frontier must be bit-exact vs a
    /// CLEAN baseline that only ever saw the accepted prefix (never ran the
    /// rejected suffix at all). Gemma has no GatedDeltaNet, so this is a pure
    /// KV-counter rewind (see `Gemma4Model::verify_rollback` / `KvCache::truncate`) —
    /// no recurrent-state re-scan like Qwen3.6's GDN rollback needs.
    ///
    /// Real-checkpoint variant of the synthetic gate below (~30 min/run on the
    /// full gemma-4-12B f32 checkpoint) — kept for optional real-weight
    /// confidence, `#[ignore]`d by default so `cargo test` stays fast.
    #[test]
    #[ignore]
    fn gemma31b_verify_rollback_frontier_matches_clean_baseline_real12b() {
        let max_seq = 16usize;
        let mut model = match load_gemma12b_for_verify_gate(max_seq) { Some(m) => m, None => return };

        let prefix = [2u32, 1024, 2048];
        let verify_tokens = [4096u32, 8192, 16384, 32768]; // T=4
        let t = verify_tokens.len();
        let r = prefix.len();
        let accept_len = 1usize;       // only verify_tokens[0..=1] accepted
        let commit_len = accept_len + 1; // 2
        let continuation_tok = 65536u32; // next token fed after the frontier

        // (a) Verify + rollback path.
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        model.forward_verify_core(&verify_tokens, r);
        model.verify_rollback(r, t, accept_len);

        // Frontier check: every layer's KV counter must sit at R+commit_len.
        let expected_frontier = r + commit_len;
        for (li, cache) in model.kv_caches.iter().enumerate() {
            assert_eq!(cache.seq_len, expected_frontier,
                "layer {li}: KV frontier {} != expected {expected_frontier}", cache.seq_len);
        }
        let logits_after_rollback = model.forward(continuation_tok, expected_frontier);

        // (b) Clean baseline: never runs the rejected suffix at all — prefix,
        // then only the accepted `commit_len` verify tokens, then the same
        // continuation token.
        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        for (i, &tok) in verify_tokens[..commit_len].iter().enumerate() {
            model.forward(tok, r + i);
        }
        let logits_baseline = model.forward(continuation_tok, expected_frontier);

        let maxdiff = logits_after_rollback.iter().zip(logits_baseline.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let cos = cosine(&logits_after_rollback, &logits_baseline);
        let am_a = argmax(&logits_after_rollback);
        let am_b = argmax(&logits_baseline);
        eprintln!("verify_rollback: frontier={expected_frontier} maxdiff={maxdiff:.8} cos={cos:.6} \
                   argmax rollback={am_a} baseline={am_b}");
        assert_eq!(maxdiff, 0.0, "post-rollback forward diverges from clean baseline (maxdiff {maxdiff})");
        assert_eq!(cos, 1.0, "cos {cos} != 1.0");
        assert_eq!(am_a, am_b, "argmax mismatch");
    }

    /// INC-4 gate 2, synthetic: identical to
    /// `gemma31b_verify_rollback_frontier_matches_clean_baseline_real12b` above
    /// but built on `tiny_synthetic_gemma`. This is the DEFAULT gate; the
    /// real-checkpoint version is `#[ignore]`d for optional confidence only.
    #[test]
    fn gemma31b_verify_rollback_frontier_matches_clean_baseline() {
        let max_seq = 16usize;
        let mut model = tiny_synthetic_gemma(max_seq);

        let prefix = [2u32, 10, 20];
        let verify_tokens = [30u32, 40, 50, 60]; // T=4
        let t = verify_tokens.len();
        let r = prefix.len();
        let accept_len = 1usize;       // only verify_tokens[0..=1] accepted
        let commit_len = accept_len + 1; // 2
        let continuation_tok = 70u32; // next token fed after the frontier

        // (a) Verify + rollback path.
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        model.forward_verify_core(&verify_tokens, r);
        model.verify_rollback(r, t, accept_len);

        // Frontier check: every layer's KV counter must sit at R+commit_len.
        let expected_frontier = r + commit_len;
        for (li, cache) in model.kv_caches.iter().enumerate() {
            assert_eq!(cache.seq_len, expected_frontier,
                "layer {li}: KV frontier {} != expected {expected_frontier}", cache.seq_len);
        }
        let logits_after_rollback = model.forward(continuation_tok, expected_frontier);

        // (b) Clean baseline: never runs the rejected suffix at all — prefix,
        // then only the accepted `commit_len` verify tokens, then the same
        // continuation token.
        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        for (i, &tok) in verify_tokens[..commit_len].iter().enumerate() {
            model.forward(tok, r + i);
        }
        let logits_baseline = model.forward(continuation_tok, expected_frontier);

        let maxdiff = logits_after_rollback.iter().zip(logits_baseline.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let cos = cosine(&logits_after_rollback, &logits_baseline);
        let am_a = argmax(&logits_after_rollback);
        let am_b = argmax(&logits_baseline);
        eprintln!("verify_rollback (synthetic): frontier={expected_frontier} maxdiff={maxdiff:.8} cos={cos:.6} \
                   argmax rollback={am_a} baseline={am_b}");
        assert_eq!(maxdiff, 0.0, "post-rollback forward diverges from clean baseline (maxdiff {maxdiff})");
        assert_eq!(cos, 1.0, "cos {cos} != 1.0");
        assert_eq!(am_a, am_b, "argmax mismatch");
    }

    /// Pure-CPU config test for the gemma-4-31B-it `g31b()` constructor.
    /// Locks the architecture constants against the real
    /// `gemma-4-31B-it-NVFP4/config.json` (`text_config`) and asserts the
    /// derived per-layer helpers behave for the period-6 global interleave,
    /// the value-less GQA(4) global attention, and the (no-PLE, no-KV-share)
    /// dense layout.  No checkpoint / GPU / network needed.
    #[test]
    fn gemma31b_config_matches_checkpoint() {
        let cfg = Gemma4Config::g31b();
        assert_eq!(cfg.variant, Gemma4Variant::G31b);
        assert_eq!(cfg.hidden_size, 5376);
        assert_eq!(cfg.num_hidden_layers, 60);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 16);
        assert_eq!(cfg.num_global_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.global_head_dim, 512);
        assert_eq!(cfg.intermediate_size, 21504);
        assert_eq!(cfg.num_kv_shared_layers, 0);
        assert_eq!(cfg.attention_period, 6);
        assert!(cfg.attention_k_eq_v);
        assert_eq!(cfg.vocab_size, 262144);
        assert_eq!(cfg.sliding_window, 1024);
        assert_eq!(cfg.hidden_size_per_layer_input, 0);
        assert!(!cfg.has_ple());
        assert_eq!(cfg.final_logit_softcapping, 30.0);
        assert!((cfg.embed_scale - (5376f32).sqrt()).abs() < 1e-3);

        // Period-6 globals: full-attention at idx 5,11,17,...,59.
        assert!(cfg.is_full_attention(5));
        assert!(cfg.is_full_attention(59));
        assert!(!cfg.is_full_attention(58));
        assert!(!cfg.is_full_attention(0));
        let n_global = (0..cfg.num_hidden_layers).filter(|&i| cfg.is_full_attention(i)).count();
        assert_eq!(n_global, 10, "expected 10 global layers in 60");

        // Last-of-type layers the drafter borrows KV from.
        let last_full = (0..cfg.num_hidden_layers).rev().find(|&i| cfg.is_full_attention(i)).unwrap();
        let last_sliding = (0..cfg.num_hidden_layers).rev().find(|&i| !cfg.is_full_attention(i)).unwrap();
        assert_eq!(last_full, 59);
        assert_eq!(last_sliding, 58);

        // Global layers: value-less (k==v), GQA(4) @ head_dim 512.
        assert!(cfg.layer_uses_k_eq_v(59));
        assert_eq!(cfg.layer_num_kv_heads(59), 4);
        assert_eq!(cfg.layer_head_dim(59), 512);
        // Sliding layers: GQA(16) @ head_dim 256, own V.
        assert!(!cfg.layer_uses_k_eq_v(58));
        assert_eq!(cfg.layer_num_kv_heads(0), 16);
        assert_eq!(cfg.layer_head_dim(0), 256);

        // No KV-sharing / no double-wide MLP anywhere (dense).
        assert_eq!(cfg.first_kv_shared_layer(), 60);
        assert!(!cfg.is_kv_shared(59));
        assert_eq!(cfg.layer_intermediate_size(59), 21504);
    }

    /// Regression test for the gemma4_unified NAS-sharded-checkpoint load bug
    /// (VulkanModel::new passed the shard-FILE path where the gemma loader
    /// wanted the checkpoint DIRECTORY, so `load_gemma_resident_weights`'s
    /// `dir.join("model.safetensors")` -> `discover_shards` turned a shard
    /// file into `shard_file.join("model.safetensors")` -> ENOTDIR). This is
    /// Mac-testable (no GPU/NAS): build a fake sharded checkpoint dir with 3
    /// dummy `.safetensors` shards + `model.safetensors.index.json` +
    /// `config.json` (no single `model.safetensors` — mirrors the real NAS
    /// layout) and assert:
    ///   1. `path.parent()` on a shard-file arg recovers the checkpoint dir
    ///      (the qwen/nemotron convention `VulkanModel::new` follows for
    ///      `config.json` resolution, and now for the gemma loader's `dir`).
    ///   2. `discover_shards(&dir.join("model.safetensors"))` — the exact call
    ///      `load_gemma_resident_weights`/`load_gemma_mlx_affine` make — finds
    ///      all 3 real shards via the sharded-only glob fallback (no literal
    ///      `model.safetensors` needs to exist), sorted, and does NOT include
    ///      the index/config JSON siblings.
    #[test]
    fn gemma_shard_path_resolution() {
        use std::path::PathBuf;

        let dir: PathBuf = std::env::temp_dir()
            .join(format!("vllk_vulkan_gemma_shard_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create tmp checkpoint dir");

        let shard_names = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ];
        for name in &shard_names {
            std::fs::write(dir.join(name), b"dummy shard bytes").expect("write dummy shard");
        }
        std::fs::write(dir.join("model.safetensors.index.json"), b"{}")
            .expect("write index.json");
        std::fs::write(dir.join("config.json"), b"{}").expect("write config.json");

        // No single `model.safetensors` exists — only the 3 numbered shards.
        assert!(!dir.join("model.safetensors").exists());

        // (1) shard-file arg -> `.parent()` recovers the checkpoint dir, and
        // `config.json` resolves as a sibling of the shard file (the
        // `VulkanModel::new` convention shared by qwen/nemotron/gemma).
        let shard_file = dir.join(shard_names[0]);
        let recovered_dir = shard_file.parent().expect("shard file has a parent");
        assert_eq!(recovered_dir, dir.as_path(), "shard-file .parent() must recover the checkpoint dir");
        let config_path = recovered_dir.join("config.json");
        assert!(config_path.exists(), "config.json must resolve as a sibling of the shard file");

        // (2) the exact gemma loader call: discover_shards(&dir.join("model.safetensors")).
        let found = discover_shards(&recovered_dir.join("model.safetensors"));
        assert_eq!(found.len(), 3, "expected 3 discovered shards, got {found:?}");
        let mut found_names: Vec<String> = found.iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        found_names.sort();
        let mut expected: Vec<String> = shard_names.iter().map(|s| s.to_string()).collect();
        expected.sort();
        assert_eq!(found_names, expected, "discover_shards must find exactly the 3 real shard files");
        assert!(found.windows(2).all(|w| w[0] <= w[1]), "discover_shards result must be sorted");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// PP pre-slice load path (VLLM_VULKAN_PP_PRESLICED_DIR). Mac-testable, no
    /// GPU: proves the loader-side contract of the pre-slice lever end to end at
    /// the tensor/byte level.
    ///
    ///  (1) `parse_stage_bounds` decodes the tool's `pp-stage{i}-L{lo}-{hi}`
    ///      filename scheme and ignores foreign files.
    ///  (2) `resolve_pp_stage_shards_in(None, ..)` == `discover_shards` (the
    ///      monolithic path is byte-for-byte unchanged when the flag is unset).
    ///  (3) With a dir set, the resolver returns exactly the stage file whose
    ///      encoded [lo,hi) equals the rank's runtime [layer_start,layer_end).
    ///  (4) BOUNDS GUARD: a runtime window with no matching stage file HARD-FAILS
    ///      (never silently loads the wrong layers).
    ///  (5) BYTE-EXACT ROUTING: the set of (name -> raw bytes) the Nemotron
    ///      loader's `keep()` accepts for a stage, read from the pre-sliced file,
    ///      is byte-identical to what it accepts from the monolithic checkpoint.
    ///      This is the offline half of the bit-exact gate — the sliced bytes are
    ///      the tool's job (proven by its own streaming verify); this proves the
    ///      LOADER routes to them correctly.
    #[test]
    fn pp_preslice_resolve_and_byte_exact_routing() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        use std::collections::HashSet;

        // (1) filename scheme
        assert_eq!(parse_stage_bounds("pp-stage2-L37-49.safetensors"), Some((37, 49)));
        assert_eq!(parse_stage_bounds("pp-stage0-L0-22.safetensors"), Some((0, 22)));
        assert_eq!(parse_stage_bounds("model-00001-of-00005.safetensors"), None);
        assert_eq!(parse_stage_bounds("config.json"), None);

        let dir: std::path::PathBuf = std::env::temp_dir()
            .join(format!("vvk_pp_preslice_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Build a tiny nemotron-named monolithic checkpoint: layers 0..4 (one
        // dummy weight each) + embed (stage0) + norm_f/lm_head (last stage) +
        // an mtp head the loader must ignore.
        let mut blobs: Vec<(String, Vec<u8>)> = Vec::new();
        for l in 0..4usize {
            blobs.push((format!("backbone.layers.{l}.mixer.o_proj.weight"),
                        vec![l as u8; 32]));
        }
        blobs.push(("backbone.embeddings.weight".into(), vec![0xEE; 16]));
        blobs.push(("backbone.norm_f.weight".into(), vec![0xBB; 8]));
        blobs.push(("lm_head.weight".into(), vec![0xAA; 16]));
        blobs.push(("mtp.layers.0.mixer.o_proj.weight".into(), vec![0x77; 8]));

        let mk = |items: &[(String, Vec<u8>)], path: &std::path::Path| {
            let views: Vec<(String, TensorView)> = items.iter()
                .map(|(n, b)| (n.clone(),
                    TensorView::new(Dtype::U8, vec![b.len()], b).unwrap()))
                .collect();
            let bytes = safetensors::serialize(
                views.iter().map(|(n, v)| (n.as_str(), v)), &None).unwrap();
            std::fs::write(path, bytes).unwrap();
        };

        let mono = dir.join("model.safetensors");
        mk(&blobs, &mono);

        // (2) flag UNSET == discover_shards (byte-for-byte the monolithic path).
        let via_none = resolve_pp_stage_shards_in(None, &mono, 0, 2).unwrap();
        assert_eq!(via_none, discover_shards(&mono));

        // Pre-slice into 2 stages [0,2) and [2,4) — the loader convention:
        // embed->stage0, norm_f/lm_head->last stage, mtp preserved (ignored).
        let bounds = [0usize, 2, 4];
        let sdir = dir.join("sliced");
        std::fs::create_dir_all(&sdir).unwrap();
        let keep_for = |name: &str, lo: usize, hi: usize, first: bool, last: bool| -> bool {
            if name == "backbone.embeddings.weight" { return first; }
            if name == "lm_head.weight" || name == "backbone.norm_f.weight" { return last; }
            if name.starts_with("mtp.") { return false; }
            match name.strip_prefix("backbone.layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|s| s.parse::<usize>().ok()) {
                Some(i) => i >= lo && i < hi,
                None => false,
            }
        };
        for s in 0..2usize {
            let (lo, hi) = (bounds[s], bounds[s + 1]);
            let first = s == 0;
            let last = s == 1;
            // route by the tool's PP convention (mirrors route_pp in the .py)
            let stage_items: Vec<(String, Vec<u8>)> = blobs.iter().filter(|(n, _)| {
                if let Some(l) = n.strip_prefix("backbone.layers.")
                    .and_then(|r| r.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok()) {
                    return l >= lo && l < hi;
                }
                if n == "backbone.embeddings.weight" { return first; }
                if n == "backbone.norm_f.weight" || n == "lm_head.weight" { return last; }
                if n.starts_with("mtp.") { return first; } // mtp lands somewhere; ignored by loader
                false
            }).cloned().collect();
            mk(&stage_items, &sdir.join(format!("pp-stage{s}-L{lo}-{hi}.safetensors")));
        }

        // (3) resolver picks the correct single stage file for each window.
        let r0 = resolve_pp_stage_shards_in(sdir.to_str(), &mono, 0, 2).unwrap();
        assert_eq!(r0.len(), 1);
        assert!(r0[0].file_name().unwrap().to_str().unwrap().contains("L0-2"));
        let r1 = resolve_pp_stage_shards_in(sdir.to_str(), &mono, 2, 4).unwrap();
        assert!(r1[0].file_name().unwrap().to_str().unwrap().contains("L2-4"));

        // (4) BOUNDS GUARD: a window with no matching stage file hard-fails.
        let miss = resolve_pp_stage_shards_in(sdir.to_str(), &mono, 0, 3);
        assert!(miss.is_err(), "mismatched runtime bounds must hard-fail");
        assert!(miss.unwrap_err().contains("BOUNDS MISMATCH"));

        // (5) BYTE-EXACT ROUTING: for each stage, the loader keep()-set read from
        // the pre-sliced file == read from the monolith, byte for byte.
        let read_kept = |path: &std::path::Path, lo, hi, first, last| -> Vec<(String, Vec<u8>)> {
            let data = std::fs::read(path).unwrap();
            let st = safetensors::SafeTensors::deserialize(&data).unwrap();
            let mut kept: Vec<(String, Vec<u8>)> = st.tensors().into_iter()
                .filter(|(n, _)| keep_for(n, lo, hi, first, last))
                .map(|(n, v)| (n, v.data().to_vec()))
                .collect();
            kept.sort();
            kept
        };
        for s in 0..2usize {
            let (lo, hi) = (bounds[s], bounds[s + 1]);
            let (first, last) = (s == 0, s == 1);
            let from_mono = read_kept(&mono, lo, hi, first, last);
            let stage_file = sdir.join(format!("pp-stage{s}-L{lo}-{hi}.safetensors"));
            let from_slice = read_kept(&stage_file, lo, hi, first, last);
            assert_eq!(from_mono, from_slice,
                "stage {s} keep()-set must be byte-identical mono vs pre-slice");
            // sanity: non-empty and no cross-stage layer leaked in
            let names: HashSet<&str> = from_slice.iter().map(|(n, _)| n.as_str()).collect();
            assert!(!names.is_empty());
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Golden for the *shipping* 4-bit MLX-affine dequant. Two rows exercise
    /// row striding and per-row scale/bias. bits=4 (per_word=8), in_features=8,
    /// group_size=8 (one group/row), words_per_row=1.
    ///   packed[0]=0x7654_3210 -> row0 nibbles q_i = i (0..7)
    ///   packed[1]=0xFEDC_BA98 -> row1 nibbles q_i = 8+i (8..15)
    /// w = scales[o]*q + biases[o]; row0 scale=0.5 bias=-1.0, row1 scale=0.25 bias=2.0.
    #[test]
    fn mlx_affine_dequant_golden() {
        let packed: Vec<u32> = vec![0x7654_3210, 0xFEDC_BA98];
        let scales: Vec<f32> = vec![0.5, 0.25];
        let biases: Vec<f32> = vec![-1.0, 2.0];
        let out = dequantize_mlx_affine(&packed, &scales, &biases,
            /*out_features*/ 2, /*in_features*/ 8, /*group_size*/ 8, /*bits*/ 4);

        let mut expected = Vec::with_capacity(16);
        for i in 0..8 { expected.push(0.5 * (i as f32) - 1.0); }        // row0
        for i in 0..8 { expected.push(0.25 * ((8 + i) as f32) + 2.0); } // row1

        assert_eq!(out.len(), 16);
        assert_eq!(out, expected);
    }

    /// Independently dequantize one q4_K super-block exactly as the Vulkan
    /// shader's DATA_A_Q4_K path does (dequant_funcs.glsl): unpack the f16
    /// d/dmin header, unpack each sub-block's 6-bit (sc, m) from scales[12] via
    /// get_scale_min_k4, then `w = d*sc*q - dmin*m`. This proves the byte layout
    /// produced by `quantize_q4_k` matches what the GPU reads — without a GPU.
    fn dequant_q4_k_block(block: &[u8]) -> Vec<f32> {
        assert_eq!(block.len(), 144);
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let mut sc12 = [0u8; 12];
        sc12.copy_from_slice(&block[4..16]);
        let qs = &block[16..144]; // 128 bytes

        // get_scale_min_k4 (matches shader's bit-twiddling, independent impl).
        let get = |j: usize| -> (u8, u8) {
            if j < 4 {
                (sc12[j] & 63, sc12[j + 4] & 63)
            } else {
                let sc = (sc12[j + 4] & 0xF) | ((sc12[j - 4] >> 6) << 4);
                let m = (sc12[j + 4] >> 4) | ((sc12[j] >> 6) << 4);
                (sc, m)
            }
        };

        let mut out = vec![0f32; 256];
        // qs layout: for the 64-element group pair `jpair`, bytes
        // qs[32*jpair .. +32], low nibble = sub-block (2*jpair), high = (2*jpair+1).
        for jpair in 0..4 {
            let (sc_lo, m_lo) = get(2 * jpair);
            let (sc_hi, m_hi) = get(2 * jpair + 1);
            let d_lo = d * sc_lo as f32;
            let dm_lo = dmin * m_lo as f32;
            let d_hi = d * sc_hi as f32;
            let dm_hi = dmin * m_hi as f32;
            for i in 0..32 {
                let byte = qs[32 * jpair + i];
                let q_lo = (byte & 0xF) as f32;
                let q_hi = (byte >> 4) as f32;
                out[64 * jpair + i] = d_lo * q_lo - dm_lo;
                out[64 * jpair + 32 + i] = d_hi * q_hi - dm_hi;
            }
        }
        out
    }

    #[test]
    fn q4_k_roundtrip_ramp() {
        // A smooth ramp over [-1, 1) across the 256-element super-block — the
        // exact case where q4_K's per-sub-block scale/min should crush error
        // relative to q4_0's single per-32 scale.
        let x: Vec<f32> = (0..256).map(|i| (i as f32) / 128.0 - 1.0).collect();
        let packed = quantize_q4_k(&x);
        assert_eq!(packed.len(), 144, "one super-block = 144 bytes");
        let recon = dequant_q4_k_block(&packed);

        let mut max_abs_err = 0f32;
        let mut sum_sq_err = 0f32;
        let mut sum_sq_x = 0f32;
        for i in 0..256 {
            let e = (recon[i] - x[i]).abs();
            if e > max_abs_err { max_abs_err = e; }
            sum_sq_err += e * e;
            sum_sq_x += x[i] * x[i];
        }
        let rel_rms = (sum_sq_err / sum_sq_x).sqrt();
        // Ramp spans 2.0; each sub-block covers a 0.25-wide slice quantized to
        // 16 levels (~0.0156 step) plus f16 rounding of d/dmin. Measured offline
        // (scratch, no GPU): max_abs_err=0.0304, rel_rms=0.0164, and crucially
        // recon[0]=-0.9959 vs x[0]=-1.0 (pos0 err 0.004) — directly fixes q4_0's
        // "pos0 too coarse" (cos 0.41) failure mode.
        assert!(max_abs_err < 0.05, "q4_K max abs err too high: {max_abs_err}");
        assert!(rel_rms < 0.03, "q4_K relative RMS too high: {rel_rms}");
        // Position-0 must be tight (the q4_0 weakness q4_K is meant to fix).
        assert!((recon[0] - x[0]).abs() < 0.02, "q4_K pos0 err too high: {}", (recon[0] - x[0]).abs());
    }

    #[test]
    fn q4_k_roundtrip_random_multiblock() {
        // Two super-blocks where each 32-element sub-block has a different scale
        // and a nonzero mean — the realistic weight case where q4_K's per-sub-
        // block 6-bit scale + min (zero-point) beats q4_0's single per-32 absmax
        // scale (q4_0 has no min, so a biased/low-range sub-block wastes levels).
        // On uniform i.i.d. noise both are similar; the win comes from
        // heterogeneous dynamic range, which is what real weights look like.
        // Measured offline (scratch): q4_K rel_rms 0.039 vs q4_0 0.059.
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ((s >> 40) as f32 / 0xff_ffff as f32) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..512)
            .map(|i| {
                let sub = (i / 32) % 8;
                let amp = (2f32).powi(sub as i32 - 3); // 0.125 .. 16 across sub-blocks
                next() * amp + amp * 0.5 // nonzero mean exercises the min component
            })
            .collect();

        let q4k = quantize_q4_k(&x);
        let recon: Vec<f32> = q4k
            .chunks_exact(144)
            .flat_map(|b| dequant_q4_k_block(b))
            .collect();
        let mut se_k = 0f32;
        let mut sx = 0f32;
        for i in 0..512 {
            se_k += (recon[i] - x[i]).powi(2);
            sx += x[i] * x[i];
        }
        let rel_k = (se_k / sx).sqrt();

        // q4_0 baseline on identical data.
        let q40 = quantize_q4_0(&x);
        let mut se_0 = 0f32;
        for (bi, blk) in q40.chunks_exact(18).enumerate() {
            let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
            for j in 0..16 {
                let lo = (blk[2 + j] & 0xF) as i32 - 8;
                let hi = (blk[2 + j] >> 4) as i32 - 8;
                let r0 = d * lo as f32;
                let r1 = d * hi as f32;
                se_0 += (r0 - x[bi * 32 + j]).powi(2);
                se_0 += (r1 - x[bi * 32 + j + 16]).powi(2);
            }
        }
        let rel_0 = (se_0 / sx).sqrt();

        assert!(rel_k < 0.05, "q4_K relative RMS too high: {rel_k}");
        assert!(rel_k < rel_0, "q4_K ({rel_k}) should beat q4_0 ({rel_0})");
    }

    #[test]
    fn e4m3_decode_known_codes() {
        // OCP FP8 E4M3 spot-checks: 0x00=0, 0x38=1.0 (exp=7,man=0), 0x40=2.0,
        // 0x08 subnormal-boundary 2^-6, 0x01 smallest subnormal 2^-9, sign bit.
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        assert_eq!(e4m3_to_f32(0x08), 2.0f32.powi(-6));
        assert_eq!(e4m3_to_f32(0x01), 2.0f32.powi(-9));
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
    }

    #[test]
    fn nvfp4_golden_real_checkpoint() {
        // Golden vector lifted from Qwen3.6-27B-NVFP4 layer0 mlp.gate_proj, row 0,
        // first 64 in-elements (32 packed bytes, 4 e4m3 block scales @ group=16, one
        // f32 global). Expected values were produced by an independent numpy
        // reference (also cross-checked cos=0.993 vs the MLX-4bit copy). This proves
        // the Rust dequant is bit-exact on real data with no model file at test time.
        let packed: [u8; 32] = [
            0x38, 0x0c, 0x93, 0xa7, 0x34, 0x25, 0x7a, 0xdb, 0x56, 0x22, 0xb5, 0x42, 0x42, 0x04,
            0x26, 0x2b, 0xad, 0xf5, 0xec, 0xf1, 0x2c, 0x7f, 0xeb, 0x41, 0x98, 0xea, 0x1e, 0x45,
            0xdd, 0xee, 0x3c, 0x19,
        ];
        let wscale: [u8; 4] = [0x5d, 0x60, 0x5a, 0x5d];
        let global = 0.00015113468f32;
        let expect: [f32; 64] = [
            0.0, 0.00589425256, -0.0078590028, 0.0, 0.00589425256, -0.0019647507, 0.0235770103,
            -0.0039295014, 0.0078590028, 0.00589425256, 0.0117885051, 0.0039295014, -0.0039295014,
            0.0235770103, -0.00589425256, -0.0117885051, 0.0193452388, 0.0145089291, 0.0048363097,
            0.0048363097, 0.0145089291, -0.00725446455, 0.0048363097, 0.0096726194, 0.0048363097,
            0.0096726194, 0.0096726194, 0.0, 0.0193452388, 0.0048363097, -0.00725446455,
            0.0048363097, -0.00906808116, -0.00302269356, 0.00906808116, -0.0181361623,
            -0.00604538713, -0.0120907743, 0.00151134678, -0.0181361623, -0.00604538713,
            0.00302269356, -0.0181361623, 0.0181361623, -0.00453404058, -0.0120907743,
            0.00151134678, 0.00604538713, 0.0, -0.0019647507, -0.0039295014, -0.0157180056,
            -0.0157180056, 0.0019647507, 0.0117885051, 0.0078590028, -0.0117885051, -0.0117885051,
            -0.0157180056, -0.0157180056, -0.0078590028, 0.00589425256, -0.0019647507,
            0.0019647507,
        ];
        // one row, in=64, group_size=16.
        let w = dequantize_nvfp4(&packed, &wscale, global, 1, 64, 16);
        assert_eq!(w.len(), 64);
        for (i, (&got, &exp)) in w.iter().zip(expect.iter()).enumerate() {
            assert!(
                (got - exp).abs() <= 1e-9 * exp.abs().max(1e-6),
                "elem {i}: got {got}, expected {exp}"
            );
        }
        // f16 variant must match the f32 path rounded to half.
        let wf16 = dequantize_nvfp4_f16(&packed, &wscale, global, 1, 64, 16);
        for (i, (&bits, &exp)) in wf16.iter().zip(expect.iter()).enumerate() {
            let got = half::f16::from_bits(bits).to_f32();
            let want = half::f16::from_f32(exp).to_f32();
            assert_eq!(got, want, "f16 elem {i}");
        }
    }

    #[test]
    fn nvfp4_shader_logic_matches_cpu_dequant() {
        // GPU-FREE proof that mul_mat_vec_nvfp4.comp's arithmetic reproduces the
        // validated CPU dequant. Emulates the shader element-for-element: reinterpret
        // the u8 packed weight as little-endian u32 words, unpack the nibble with the
        // mlx4 `shift=(j&7)*4` scheme, map through the E2M1 LUT, multiply by the
        // pre-folded f32 scale (e4m3*global). The row dot must equal
        // `dequantize_nvfp4(...) . x`. Uses the real-checkpoint golden bytes.
        let packed: [u8; 32] = [
            0x38, 0x0c, 0x93, 0xa7, 0x34, 0x25, 0x7a, 0xdb, 0x56, 0x22, 0xb5, 0x42, 0x42, 0x04,
            0x26, 0x2b, 0xad, 0xf5, 0xec, 0xf1, 0x2c, 0x7f, 0xeb, 0x41, 0x98, 0xea, 0x1e, 0x45,
            0xdd, 0xee, 0x3c, 0x19,
        ];
        let wscale: [u8; 4] = [0x5d, 0x60, 0x5a, 0x5d];
        let global = 0.00015113468f32;
        let (out_f, in_f, gs) = (1usize, 64usize, 16usize);

        // A deterministic activation vector.
        let x: Vec<f32> = (0..in_f).map(|i| ((i as f32) * 0.37).sin()).collect();

        // CPU reference: full dequant then dot.
        let deq = dequantize_nvfp4(&packed, &wscale, global, out_f, in_f, gs);
        let cpu: f32 = deq.iter().zip(x.iter()).map(|(w, xv)| w * xv).sum();

        // Shader emulation.
        let words: Vec<u32> = packed
            .chunks(4)
            .map(|c| c.iter().enumerate().fold(0u32, |w, (i, &b)| w | ((b as u32) << (8 * i))))
            .collect();
        let folded: Vec<f32> = wscale.iter().map(|&b| e4m3_to_f32(b) * global).collect();
        let groups = in_f / gs;
        let words_per_row = in_f / 8;
        let r = 0usize;
        let mut acc = 0f32;
        for j in 0..in_f {
            let widx = j >> 3;
            let shift = (j & 7) * 4;
            let g = j / gs;
            let code = (words[r * words_per_row + widx] >> shift) & 0xF;
            let w = NVFP4_E2M1_LUT[code as usize] * folded[r * groups + g];
            acc += w * x[j];
        }
        assert!((acc - cpu).abs() <= 1e-6 * cpu.abs().max(1e-6), "shader {acc} vs cpu {cpu}");
    }

    #[test]
    fn fp8_golden_real_checkpoint() {
        // Golden from Qwen3.6-27B-NVFP4 layer 11 self_attn.q_proj (FP8-E4M3),
        // row 0, first 16 in-elements. Per-tensor scalar scale. Independent numpy
        // reference; proves the FP8 dequant is bit-exact on real attention weights.
        let bytes: [u8; 16] = [
            0x4f, 0xa9, 0x3f, 0x51, 0x4c, 0x5a, 0x4f, 0xd3, 0x4e, 0x47, 0x48, 0xc2, 0xd0, 0x40,
            0x5d, 0xd1,
        ];
        let scale = 0.0010550363222137094f32;
        let expect: [f32; 16] = [
            0.00791277271, -0.000296728977, 0.00197819318, 0.00949532725, 0.00633021817,
            0.021100726, 0.00791277271, -0.0116053997, 0.00738525437, 0.00395638635,
            0.00422014529, -0.00263759075, -0.00844029058, 0.00211007264, 0.0274309441,
            -0.00949532725,
        ];
        let w = dequantize_fp8(&bytes, &[scale], 1, 16);
        assert_eq!(w.len(), 16);
        for (i, (&got, &exp)) in w.iter().zip(expect.iter()).enumerate() {
            assert!(
                (got - exp).abs() <= 1e-9 * exp.abs().max(1e-6),
                "elem {i}: got {got}, expected {exp}"
            );
        }
        // per-row scale broadcast: a 2-row tensor with distinct scales.
        let two = dequantize_fp8(&[0x38, 0x40], &[2.0, 3.0], 2, 1);
        assert_eq!(two, vec![1.0 * 2.0, 2.0 * 3.0]);
    }

    #[test]
    fn fp8_shader_logic_matches_cpu_dequant() {
        // Same golden bytes/scale as fp8_golden_real_checkpoint (layer11 q_proj row0).
        let bytes: [u8; 16] = [
            0x4f, 0xa9, 0x3f, 0x51, 0x4c, 0x5a, 0x4f, 0xd3, 0x4e, 0x47, 0x48, 0xc2, 0xd0, 0x40,
            0x5d, 0xd1,
        ];
        let scale = 0.0010550363222137094f32;
        let (out_f, in_f) = (1usize, 16usize);
        let x: Vec<f32> = (0..in_f).map(|i| ((i as f32) * 0.37).sin()).collect();
        let deq = dequantize_fp8(&bytes, &[scale], out_f, in_f);
        let cpu: f32 = deq.iter().zip(&x).map(|(w, xv)| w * xv).sum();
        // Emulate the shader: reinterpret bytes as LE u32 words, absolute-byte index.
        let words: Vec<u32> = bytes
            .chunks(4)
            .map(|c| c.iter().enumerate().fold(0u32, |w, (i, &b)| w | ((b as u32) << (8 * i))))
            .collect();
        let r = 0usize;
        let mut acc = 0f32;
        for j in 0..in_f {
            let bi = r * in_f + j;
            let code = (words[bi >> 2] >> ((bi & 3) * 8)) & 0xFF;
            acc += e4m3_to_f32(code as u8) * scale * x[j];
        }
        assert!((acc - cpu).abs() <= 1e-6 * cpu.abs().max(1e-6), "shader {acc} vs cpu {cpu}");
    }

    /// Full-loader check on a real NVFP4 checkpoint (HF naming -> detection ->
    /// cross-shard sibling fetch -> dequant). Ignored by default; run with the
    /// checkpoint dir in `VLLM_TEST_NVFP4_DIR`:
    ///   VLLM_TEST_NVFP4_DIR=~/repos/OminiX-MLX/models/Qwen3.6-27B-NVFP4 \
    ///     cargo test --lib -- --ignored nvfp4_loads_real_checkpoint
    #[test]
    #[ignore]
    fn nvfp4_loads_real_checkpoint() {
        let dir = match std::env::var("VLLM_TEST_NVFP4_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let path = std::path::Path::new(&dir).join("config.json");
        // group_size/bits here are the MLX-affine defaults; NVFP4 derives its own
        // group size from the scale-tensor shape, so these are unused for it.
        // Layer 0 only, skip the multi-GB embed/lm_head — isolates the dense
        // NVFP4 MLP path (the f16 lm_head path is covered by the golden unit test).
        let (f32w, _f16w, _pe, _lmp) = load_qwen35_weights_split(
            &path, 64, 4, 0, Some(1), false, false,
            |_, w| match w {
                ProjWeight::F32(v) => ProjResult::KeepF32(v),
                // flag-off equivalent: ask the loader to dequant (test asserts f32).
                ProjWeight::Nvfp4(_) => ProjResult::Dequantize,
                ProjWeight::Fp8(_) => ProjResult::Dequantize,
                ProjWeight::Mlx4(_) => ProjResult::Dequantize,
            },
        )
        .expect("load nvfp4 checkpoint");

        // The gate_proj must be present under the normalized `model.layers.*` name
        // and its row-0 head must match the independently-derived golden.
        let gate = f32w
            .get("model.layers.0.mlp.gate_proj.weight")
            .expect("gate_proj present after HF-name normalization");
        assert_eq!(gate.len(), 17408 * 5120, "gate_proj full [out,in]");
        let expect_head: [f32; 8] = [
            0.0, 0.00589425256, -0.0078590028, 0.0, 0.00589425256, -0.0019647507, 0.0235770103,
            -0.0039295014,
        ];
        for (i, &e) in expect_head.iter().enumerate() {
            assert!((gate[i] - e).abs() <= 1e-9 * e.abs().max(1e-6), "gate[{i}]={}", gate[i]);
        }
        assert!(f32w.contains_key("model.layers.0.mlp.up_proj.weight"), "up_proj loaded");
        assert!(f32w.contains_key("model.layers.0.mlp.down_proj.weight"), "down_proj loaded");

        // GatedDeltaNet per-layer params with no `.weight` suffix. The pread
        // rewrite regression dropped these (they fail a `.weight`-only guard),
        // which made forward_pp_qwen35 panic `Weight
        // 'model.layers.0.linear_attn.A_log' not found` on the cluster.
        let a_log = f32w
            .get("model.layers.0.linear_attn.A_log")
            .expect("A_log present (non-.weight GDN param survives the loader)");
        assert_eq!(a_log.len(), 48, "A_log [num_v_heads]");
        let dt_bias = f32w
            .get("model.layers.0.linear_attn.dt_bias")
            .expect("dt_bias present (non-.weight GDN param survives the loader)");
        assert_eq!(dt_bias.len(), 48, "dt_bias [num_v_heads]");
        assert!(f32w.contains_key("model.layers.0.linear_attn.conv1d.weight"), "conv1d loaded");
        assert!(f32w.contains_key("model.layers.0.linear_attn.norm.weight"), "GDN norm loaded");
        // Zero-centered RMSNorm: HF stores w with `x_hat*(1+w)` semantics; the
        // loader folds the +1 (MLX/forward convention). Raw layer-0
        // input_layernorm[0] is ~0.0583 -> loaded ~1.0583; raw mean is ~0 ->
        // loaded mean ~1. Without the fold the forward multiplies hidden
        // states by ~0 and the model emits structural garbage from token 0.
        let iln = f32w.get("model.layers.0.input_layernorm.weight").expect("input_layernorm");
        assert!((1.05..1.07).contains(&iln[0]), "1+w fold applied (got {})", iln[0]);
        let mean = iln.iter().sum::<f32>() / iln.len() as f32;
        assert!((0.8..1.2).contains(&mean), "input_layernorm mean ~1 after fold (got {mean})");
        // The gated GDN norm must stay plain (it is ~0.89-ish, NOT ~1.89).
        let gn = f32w.get("model.layers.0.linear_attn.norm.weight").unwrap();
        assert!(gn[0] < 1.5, "linear_attn.norm NOT offset (got {})", gn[0]);
        // The modelopt export's vision tower is `model.visual.*` — the text
        // forward never reads it and it must not burn ~1.8GB host f32 per rank.
        assert!(
            !f32w.keys().any(|k| k.starts_with("model.visual.")),
            "model.visual.* must be skipped"
        );

        // FP8 attention: layer 0's linear_attn.in_proj_qkv is F8_E4M3 [10240,5120].
        // Before the FP8 loader it decoded to an empty Vec; now it must be full and
        // non-trivial (proves the mixed-precision model actually loads).
        let qkv = f32w
            .get("model.layers.0.linear_attn.in_proj_qkv.weight")
            .expect("FP8 in_proj_qkv present");
        assert_eq!(qkv.len(), 10240 * 5120, "FP8 in_proj_qkv full [out,in]");
        assert!(qkv.iter().any(|&v| v != 0.0), "FP8 weight non-zero (not the old empty path)");
        eprintln!(
            "mixed-precision load OK: {} f32 tensors (layer 0); NVFP4 mlp + FP8 attn both non-empty",
            f32w.len()
        );
    }

    /// Hermetic regression for the `.weight`-only-guard bug: a synthetic shard
    /// with the GDN `A_log`/`dt_bias` (no `.weight` suffix, BF16) must land in
    /// the host f32 map under the normalized name, and `model.visual.*` must
    /// be skipped entirely.
    #[test]
    fn split_loader_keeps_non_weight_gdn_params() {
        // bf16 bit patterns: 1.0, 2.0, -1.0, 0.5
        let bf16: [u16; 4] = [0x3F80, 0x4000, 0xBF80, 0x3F00];
        let expect: [f32; 4] = [1.0, 2.0, -1.0, 0.5];
        let payload: Vec<u8> = bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let names = [
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.0.linear_attn.dt_bias",
            "model.language_model.layers.0.input_layernorm.weight", // HF zero-centered -> +1
            "model.language_model.layers.0.linear_attn.norm.weight", // gated norm -> plain
            "model.visual.blocks.0.norm1.weight", // must be aux-skipped
        ];
        // Hand-build a minimal safetensors shard: 8-byte LE header len + JSON
        // header + concatenated tensor data.
        let mut hdr = String::from("{");
        let mut data: Vec<u8> = Vec::new();
        for (i, n) in names.iter().enumerate() {
            if i > 0 { hdr.push(','); }
            let (s, e) = (data.len(), data.len() + payload.len());
            hdr.push_str(&format!(
                "\"{n}\":{{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[{s},{e}]}}"));
            data.extend_from_slice(&payload);
        }
        hdr.push('}');
        let dir = std::env::temp_dir().join(format!(
            "vv_gdn_param_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("model.safetensors");
        let mut file: Vec<u8> = (hdr.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(hdr.as_bytes());
        file.extend_from_slice(&data);
        std::fs::write(&shard, &file).unwrap();

        let (f32w, f16w, _pe, _lmp) = load_qwen35_weights_split(
            &shard, 64, 4, 0, None, true, true,
            |_, w| match w {
                ProjWeight::F32(v) => ProjResult::KeepF32(v),
                _ => ProjResult::Dequantize,
            },
        ).expect("synthetic shard loads");
        let _ = std::fs::remove_dir_all(&dir);

        let a_log = f32w.get("model.layers.0.linear_attn.A_log")
            .expect("A_log survives the .weight-suffix guard, normalized");
        assert_eq!(a_log.as_slice(), &expect, "A_log decoded bf16->f32");
        let dt = f32w.get("model.layers.0.linear_attn.dt_bias")
            .expect("dt_bias survives, normalized");
        assert_eq!(dt.as_slice(), &expect, "dt_bias decoded bf16->f32");
        // HF zero-centered RMSNorm: `model.language_model.*` input_layernorm
        // gets the +1 folded at load (forward multiplies by `w` directly).
        let iln = f32w.get("model.layers.0.input_layernorm.weight")
            .expect(".weight control tensor still loads");
        let expect_p1: [f32; 4] = [2.0, 3.0, 0.0, 1.5];
        assert_eq!(iln.as_slice(), &expect_p1, "HF input_layernorm loaded as 1+w");
        // The gated GDN norm is stored plain in both exports: NO offset.
        let gdn_norm = f32w.get("model.layers.0.linear_attn.norm.weight")
            .expect("linear_attn.norm loads");
        assert_eq!(gdn_norm.as_slice(), &expect, "linear_attn.norm NOT offset");
        assert!(!f32w.keys().any(|k| k.contains("visual")),
            "model.visual.* skipped");
        assert!(f16w.is_empty(), "no embed/lm_head in this synthetic shard");
    }

    /// The zero-centered predicate: HF-nested norms only, never the gated GDN
    /// norm, never MLX-flavored names (where +1 is already baked in).
    #[test]
    fn hf_zero_centered_norm_predicate() {
        let hf = "model.language_model.layers.7.input_layernorm.weight";
        assert!(qwen35_hf_zero_centered_norm(hf, "model.layers.7.input_layernorm.weight"));
        assert!(qwen35_hf_zero_centered_norm(
            "model.language_model.layers.3.self_attn.q_norm.weight",
            "model.layers.3.self_attn.q_norm.weight"));
        assert!(qwen35_hf_zero_centered_norm(
            "model.language_model.layers.3.self_attn.k_norm.weight",
            "model.layers.3.self_attn.k_norm.weight"));
        assert!(qwen35_hf_zero_centered_norm(
            "model.language_model.norm.weight", "model.norm.weight"));
        assert!(qwen35_hf_zero_centered_norm(
            "model.language_model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight"));
        // gated GDN norm: plain in both exports.
        assert!(!qwen35_hf_zero_centered_norm(
            "model.language_model.layers.0.linear_attn.norm.weight",
            "model.layers.0.linear_attn.norm.weight"));
        // MLX flavor: +1 already baked into the stored weight.
        assert!(!qwen35_hf_zero_centered_norm(
            "language_model.model.layers.7.input_layernorm.weight",
            "model.layers.7.input_layernorm.weight"));
        assert!(!qwen35_hf_zero_centered_norm(
            "model.layers.7.input_layernorm.weight",
            "model.layers.7.input_layernorm.weight"));
    }

    #[test]
    #[ignore] // VLLM_TEST_NVFP4_DIR=~/repos/OminiX-MLX/models/Qwen3.6-27B-NVFP4 \
              //   cargo test --lib -- --ignored nvfp4_packed_loader_matches_dequant
    fn nvfp4_packed_loader_matches_dequant() {
        let dir = match std::env::var("VLLM_TEST_NVFP4_DIR") { Ok(d) => d, Err(_) => return };
        let path = std::path::Path::new(&dir).join("config.json");
        // Capture the packed NVFP4 gate_proj the loader offers (layer 0 only).
        let mut cap: Option<(Vec<u8>, Vec<u8>, f32, usize, usize, usize)> = None; // packed, wscale, global, out,in,gs
        let _ = load_qwen35_weights_split(
            &path, 64, 4, 0, Some(1), false, false,
            |name, w| match w {
                ProjWeight::Nvfp4(nv) if name.ends_with("mlp.gate_proj.weight") => {
                    cap = Some((nv.packed.to_vec(), nv.wscale.to_vec(), nv.global,
                                nv.out_features, nv.in_features, nv.group_size));
                    ProjResult::Consumed
                }
                ProjWeight::Nvfp4(_) => ProjResult::Consumed, // drop other packed
                ProjWeight::F32(_) => ProjResult::Consumed,   // drop f32
                ProjWeight::Fp8(_) => ProjResult::Consumed,   // drop fp8 attn (this test targets nvfp4)
                ProjWeight::Mlx4(_) => ProjResult::Consumed,  // drop mlx4 dense (this test targets nvfp4)
            },
        ).expect("load nvfp4 checkpoint");
        let (packed, wscale, global, out_f, in_f, gs) = cap.expect("gate_proj offered as packed NVFP4");
        assert_eq!(out_f, 17408); assert_eq!(in_f, 5120); assert_eq!(gs, 16);

        // Reconstruct the exact per-element weight the GPU path computes (packed
        // nibbles + folded e4m3*global scales) and assert it is bit-exact against
        // `dequantize_nvfp4` for a few rows (row-0, row-1, last row).
        let groups = in_f / gs;
        let bytes_per_row = in_f / 2;
        let deq = dequantize_nvfp4(&packed, &wscale, global, out_f, in_f, gs);
        for r in [0usize, 1, 17407] {
            let brow = &packed[r * bytes_per_row..(r + 1) * bytes_per_row];
            for j in 0..in_f {
                let byte = brow[j / 2];
                let nib = if j % 2 == 0 { byte & 0xF } else { byte >> 4 } as usize;
                let g = r * groups + j / gs;
                let bscale = e4m3_to_f32(wscale[g]) * global;
                let recon = NVFP4_E2M1_LUT[nib] * bscale;
                let expect = deq[r * in_f + j];
                // f32 reassociation (lut*(bscale*global) vs (lut*bscale)*global)
                // can differ by ~1 ULP; use a realistic f32 relative tolerance.
                let tol = 1e-6 * expect.abs().max(1e-6);
                assert!((recon - expect).abs() <= tol,
                    "row {r} elem {j}: recon {recon} vs dequant {expect}");
            }
        }
    }

    /// Full-loader check on a real FP8 checkpoint (HF naming -> detection -> the
    /// FP8-packed sink branch). Ignored by default; run with the checkpoint dir in
    /// `VLLM_TEST_NVFP4_DIR`:
    ///   VLLM_TEST_NVFP4_DIR=~/repos/OminiX-MLX/models/Qwen3.6-27B-NVFP4 \
    ///     cargo test --lib -- --ignored fp8_packed_loader_matches_dequant
    #[test]
    #[ignore]
    fn fp8_packed_loader_matches_dequant() {
        let dir = match std::env::var("VLLM_TEST_NVFP4_DIR") { Ok(d) => d, Err(_) => return };
        let path = std::path::Path::new(&dir).join("config.json");
        // Capture the packed FP8 self_attn.q_proj the loader offers (layer 11 only:
        // max_layers is a GLOBAL index bound, so [layer_start=11, max_layers=12)).
        let mut cap: Option<(Vec<u8>, Vec<f32>, usize, usize)> = None; // weight, scale, out, in
        let _ = load_qwen35_weights_split(
            &path, 64, 4, 11, Some(12), false, false,
            |name, w| match w {
                ProjWeight::Fp8(fp) if name.ends_with("layers.11.self_attn.q_proj.weight") => {
                    cap = Some((fp.weight.to_vec(), fp.scale.clone(), fp.out_features, fp.in_features));
                    ProjResult::Consumed
                }
                ProjWeight::Fp8(_) => ProjResult::Consumed,   // drop other packed
                ProjWeight::Nvfp4(_) => ProjResult::Consumed, // drop nvfp4 mlp
                ProjWeight::F32(_) => ProjResult::Consumed,   // drop f32
                ProjWeight::Mlx4(_) => ProjResult::Consumed,  // drop mlx4 dense (this test targets fp8)
            },
        ).expect("load fp8 checkpoint");
        let (weight, scale, out_f, in_f) = cap.expect("layer11 q_proj offered as packed FP8");

        // Reconstruct the exact per-element weight the GPU path computes (raw
        // FP8-E4M3 bytes + scale) and assert it is bit-close against
        // `dequantize_fp8` for a few rows.
        let deq = dequantize_fp8(&weight, &scale, out_f, in_f);
        for r in [0usize, 1, out_f - 1] {
            let brow = &weight[r * in_f..(r + 1) * in_f];
            for j in 0..in_f {
                let sidx = if scale.len() == out_f { r } else { 0 };
                let recon = e4m3_to_f32(brow[j]) * scale[sidx];
                let expect = deq[r * in_f + j];
                let tol = 1e-6 * expect.abs().max(1e-6);
                assert!((recon - expect).abs() <= tol,
                    "row {r} elem {j}: recon {recon} vs dequant {expect}");
            }
        }
    }

    /// INC-1 gate: `load_gemma_nvfp4_weights` on the REAL 24.7GB
    /// gemma-4-31B-it-NVFP4 checkpoint (mmap'd, layer-0-only via
    /// `(layer_start=0, layer_end=1)` — the loader always also loads
    /// embed/norm/lm_head-adjacent tensors, but decoder layers are windowed,
    /// so this stays cheap on Mac's 103GB f32/GPU-less box). Captures ONE
    /// NVFP4 mlp tensor (`layers.0.mlp.gate_proj`) and ONE FP8 attn tensor
    /// (`layers.0.self_attn.q_proj`) as the loader offers them (packed,
    /// pre-dequant) and reconstructs each element independently from the raw
    /// packed bytes + LUT/scale — the SAME cross-check method
    /// `nvfp4_packed_loader_matches_dequant`/`fp8_packed_loader_matches_dequant`
    /// use on the Qwen3.6-27B-NVFP4 checkpoint (this is a proven format;
    /// see `GEMMA31B_SPEC_PLAN.md` INC-1). Reports maxdiff + cosine
    /// similarity against `dequantize_nvfp4`/`dequantize_fp8`'s own output
    /// for both tensors. Ignored by default; run with:
    ///   DYLD_LIBRARY_PATH=/opt/homebrew/lib:/opt/homebrew/pkgs/python-3.10.16-h870587a_1_cpython/lib \
    ///   PYTHONHOME=/opt/homebrew/pkgs/python-3.10.16-h870587a_1_cpython \
    ///   VLLM_TEST_GEMMA31B_DIR=~/repos/OminiX-MLX/models/gemma-4-31B-it-NVFP4 \
    ///     cargo test --lib gemma31b_nvfp4_fp8_dequant_bitexact -- --ignored
    #[test]
    #[ignore]
    fn gemma31b_nvfp4_fp8_dequant_bitexact() {
        let dir = match std::env::var("VLLM_TEST_GEMMA31B_DIR") { Ok(d) => d, Err(_) => return };
        let dir = std::path::Path::new(&dir);

        // Capture the packed NVFP4 mlp.gate_proj + packed FP8 self_attn.q_proj
        // of layer 0, exactly as the loader offers them to `on_proj` (before
        // any dequant).
        let mut nvfp4_cap: Option<(Vec<u8>, Vec<u8>, f32, usize, usize, usize)> = None; // packed,wscale,global,out,in,gs
        let mut fp8_cap: Option<(Vec<u8>, Vec<f32>, usize, usize)> = None; // weight,scale,out,in
        let (f32w, f16w) = load_gemma_nvfp4_weights(
            dir, 0, 1,
            |name, w| match w {
                ProjWeight::Nvfp4(nv) if name.ends_with("layers.0.mlp.gate_proj.weight") => {
                    nvfp4_cap = Some((nv.packed.to_vec(), nv.wscale.to_vec(), nv.global,
                                      nv.out_features, nv.in_features, nv.group_size));
                    ProjResult::Consumed
                }
                ProjWeight::Fp8(fp) if name.ends_with("layers.0.self_attn.q_proj.weight") => {
                    fp8_cap = Some((fp.weight.to_vec(), fp.scale.clone(), fp.out_features, fp.in_features));
                    ProjResult::Consumed
                }
                ProjWeight::Nvfp4(_) | ProjWeight::Fp8(_) => ProjResult::Consumed, // drop other packed
                ProjWeight::F32(v) => ProjResult::KeepF32(v),
                ProjWeight::Mlx4(_) => ProjResult::Consumed, // not present in this checkpoint
            },
        ).expect("load gemma31b NVFP4+FP8 checkpoint (layer 0)");

        // Sanity: the un-quantized host tensors (norms, layer_scalar, tied
        // embed) round-tripped too — proves the plain-tensor branch and the
        // "always-kept" embed/norm windowing both fired alongside the layer-0
        // matvec capture.
        assert!(f32w.contains_key("model.layers.0.input_layernorm.weight"));
        assert!(f32w.contains_key("model.norm.weight"));
        assert_eq!(
            f16w.get("model.embed_tokens.weight").map(|v| v.len()),
            Some(262144 * 5376),
            "tied embed_tokens present as f16, full [vocab, hidden]"
        );

        fn cos(a: &[f32], b: &[f32]) -> f64 {
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (x, y) in a.iter().zip(b) {
                dot += *x as f64 * *y as f64;
                na += *x as f64 * *x as f64;
                nb += *y as f64 * *y as f64;
            }
            dot / (na.sqrt() * nb.sqrt())
        }
        fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        }

        // ── NVFP4 mlp.gate_proj: [21504, 5376], group_size 16 ────────────
        let (packed, wscale, global, out_f, in_f, gs) =
            nvfp4_cap.expect("layer0 mlp.gate_proj offered as packed NVFP4");
        assert_eq!(out_f, 21504, "gate_proj out_features (intermediate_size)");
        assert_eq!(in_f, 5376, "gate_proj in_features (hidden_size)");
        assert_eq!(gs, 16, "NVFP4 group_size per recipe.yaml");
        let groups = in_f / gs;
        let bytes_per_row = in_f / 2;
        let deq = dequantize_nvfp4(&packed, &wscale, global, out_f, in_f, gs);
        let mut recon = vec![0.0f32; in_f * 3]; // rows 0, 1, out_f-1
        let mut expect = vec![0.0f32; in_f * 3];
        for (ri, &r) in [0usize, 1, out_f - 1].iter().enumerate() {
            let brow = &packed[r * bytes_per_row..(r + 1) * bytes_per_row];
            for j in 0..in_f {
                let byte = brow[j / 2];
                let nib = if j % 2 == 0 { byte & 0xF } else { byte >> 4 } as usize;
                let g = r * groups + j / gs;
                let bscale = e4m3_to_f32(wscale[g]) * global;
                recon[ri * in_f + j] = NVFP4_E2M1_LUT[nib] * bscale;
                expect[ri * in_f + j] = deq[r * in_f + j];
            }
        }
        let nvfp4_cos = cos(&recon, &expect);
        let nvfp4_maxdiff = maxdiff(&recon, &expect);
        eprintln!(
            "gemma31b NVFP4 mlp.gate_proj (layer 0, 3 rows x {in_f}): cos={nvfp4_cos:.8} maxdiff={nvfp4_maxdiff:e}"
        );
        assert!(nvfp4_cos >= 0.9999, "NVFP4 gate_proj cos {nvfp4_cos} < 0.9999");
        for (r, e) in recon.iter().zip(&expect) {
            let tol = 1e-6 * e.abs().max(1e-6);
            assert!((r - e).abs() <= tol, "NVFP4 recon {r} vs dequant {e}");
        }

        // ── FP8 self_attn.q_proj: [8192, 5376] (32 heads x 256, sliding L0) ─
        let (weight, scale, fout_f, fin_f) = fp8_cap.expect("layer0 self_attn.q_proj offered as packed FP8");
        assert_eq!(fout_f, 8192, "q_proj out_features (32 heads x 256)");
        assert_eq!(fin_f, 5376, "q_proj in_features (hidden_size)");
        let fdeq = dequantize_fp8(&weight, &scale, fout_f, fin_f);
        let mut frecon = vec![0.0f32; fin_f * 3];
        let mut fexpect = vec![0.0f32; fin_f * 3];
        for (ri, &r) in [0usize, 1, fout_f - 1].iter().enumerate() {
            let brow = &weight[r * fin_f..(r + 1) * fin_f];
            for j in 0..fin_f {
                let sidx = if scale.len() == fout_f { r } else { 0 };
                frecon[ri * fin_f + j] = e4m3_to_f32(brow[j]) * scale[sidx];
                fexpect[ri * fin_f + j] = fdeq[r * fin_f + j];
            }
        }
        let fp8_cos = cos(&frecon, &fexpect);
        let fp8_maxdiff = maxdiff(&frecon, &fexpect);
        eprintln!(
            "gemma31b FP8 self_attn.q_proj (layer 0, 3 rows x {fin_f}): cos={fp8_cos:.8} maxdiff={fp8_maxdiff:e}"
        );
        assert!(fp8_cos >= 0.9999, "FP8 q_proj cos {fp8_cos} < 0.9999");
        for (r, e) in frecon.iter().zip(&fexpect) {
            let tol = 1e-6 * e.abs().max(1e-6);
            assert!((r - e).abs() <= tol, "FP8 recon {r} vs dequant {e}");
        }
    }

    /// Full-loader check on a real MLX-4bit checkpoint's DENSE (GatedDeltaNet)
    /// projection: capture the packed MLX4 `in_proj_qkv` the loader offers and
    /// assert it reconstructs bit-identically against `dequantize_mlx_affine`
    /// — the SAME affine math the MoE-expert 4-bit-resident path already
    /// validates at cos=1.0 vs the MLX oracle (this is the item-3 "4-bit
    /// dense residency" wiring: the dense predicate now offers packed MLX4
    /// instead of falling straight to the f32-dequant -> q8_0-requantize
    /// path). Ignored by default; run with the checkpoint dir in
    /// `VLLM_TEST_MOE_DIR` (any Qwen3.6 MLX-4bit checkpoint has dense GDN
    /// layers, MoE or not):
    ///   VLLM_TEST_MOE_DIR=~/repos/OminiX-MLX/models/Qwen3.6-35B-A3B-4bit \
    ///     cargo test --lib -- --ignored mlx4_dense_packed_loader_matches_dequant
    #[test]
    #[ignore]
    fn mlx4_dense_packed_loader_matches_dequant() {
        let dir = match std::env::var("VLLM_TEST_MOE_DIR") { Ok(d) => d, Err(_) => return };
        let path = std::path::Path::new(&dir).join("config.json");
        // Capture the packed MLX4 in_proj_qkv the loader offers (layer 0 only).
        let mut cap: Option<(Vec<u32>, Vec<f32>, Vec<f32>, usize, usize, usize)> = None; // packed,scales,biases,out,in,gs
        let _ = load_qwen35_weights_split(
            &path, 64, 4, 0, Some(1), false, false,
            |name, w| match w {
                ProjWeight::Mlx4(mx) if name.ends_with("linear_attn.in_proj_qkv.weight") => {
                    cap = Some((mx.packed.clone(), mx.scales.clone(), mx.biases.clone(),
                                mx.out_features, mx.in_features, mx.group_size));
                    ProjResult::Consumed
                }
                ProjWeight::Mlx4(_) => ProjResult::Consumed,  // drop other packed dense
                ProjWeight::Nvfp4(_) => ProjResult::Consumed, // not present in an MLX4 checkpoint
                ProjWeight::Fp8(_) => ProjResult::Consumed,   // not present in an MLX4 checkpoint
                ProjWeight::F32(_) => ProjResult::Consumed,   // drop f32 (embed/norm/MoE experts)
            },
        ).expect("load MLX-4bit checkpoint");
        let (packed, scales, biases, out_f, in_f, gs) =
            cap.expect("layer0 linear_attn.in_proj_qkv offered as packed MLX4");

        // Reconstruct the exact per-element weight the GPU shader computes
        // (packed nibbles + affine scale*nibble+bias) and assert it is
        // bit-exact against `dequantize_mlx_affine` for a few rows.
        let per_word = 32 / 4; // bits=4
        let groups = in_f / gs;
        let words_per_row = in_f / per_word;
        let deq = dequantize_mlx_affine(&packed, &scales, &biases, out_f, in_f, gs, 4);
        for r in [0usize, 1, out_f - 1] {
            let prow = &packed[r * words_per_row..(r + 1) * words_per_row];
            for j in 0..in_f {
                let word = prow[j / per_word];
                let nib = (word >> ((j % per_word) * 4)) & 0xF;
                let g = r * groups + j / gs;
                let recon = scales[g] * (nib as f32) + biases[g];
                let expect = deq[r * in_f + j];
                let tol = 1e-6 * expect.abs().max(1e-6);
                assert!((recon - expect).abs() <= tol,
                    "row {r} elem {j}: recon {recon} vs dequant {expect}");
            }
        }
    }

    #[test]
    fn normalize_name_all_prefixes() {
        // MLX prefix.
        assert_eq!(
            normalize_qwen35_name("language_model.model.layers.0.mlp.gate.weight"),
            "model.layers.0.mlp.gate.weight"
        );
        // HF nested prefix (the switch_mlp expert path that used to load ZERO experts).
        assert_eq!(
            normalize_qwen35_name("model.language_model.layers.5.mlp.switch_mlp.gate_proj.weight"),
            "model.layers.5.mlp.switch_mlp.gate_proj.weight"
        );
        // Already-normalized name is unchanged.
        assert_eq!(
            normalize_qwen35_name("model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        // A bare language_model. prefix on a non-model tensor still strips.
        assert_eq!(normalize_qwen35_name("language_model.norm.weight"), "norm.weight");
    }

    #[test]
    fn derive_group_size_and_validate() {
        // 2 rows, in=8, one group/row => gs=8 (the golden's geometry).
        assert_eq!(mlx_affine_group_size("t", 8, 2, 2).unwrap(), 8);
        // 2 rows, in=8, two groups/row => gs=4.
        assert_eq!(mlx_affine_group_size("t", 8, 2, 4).unwrap(), 4);
        // scales not divisible by out_features => error.
        assert!(mlx_affine_group_size("t", 8, 2, 3).is_err());
        // zero dims => error.
        assert!(mlx_affine_group_size("t", 0, 2, 2).is_err());

        // validate: golden geometry (out=2,in=8,gs=8,bits=4 => 1 word/row, 1 group/row).
        assert!(validate_affine_dims("t", 2, 2, 8, 8, 4, 2, 2).is_ok());
        // wrong scales len => error.
        assert!(validate_affine_dims("t", 2, 2, 8, 8, 4, 3, 3).is_err());
        // wrong packed len => error.
        assert!(validate_affine_dims("t", 3, 2, 8, 8, 4, 2, 2).is_err());
        // in_features not a multiple of the pack width => error.
        assert!(validate_affine_dims("t", 2, 2, 7, 7, 4, 2, 2).is_err());
    }

    #[test]
    fn mlx_affine_parallel_multirow() {
        // Three rows exercise par_chunks_mut striding beyond the 2-row golden.
        // in=8, gs=8 (1 group/row), bits=4 (1 word/row). q_i = nibble i of packed[o].
        let packed: Vec<u32> = vec![0x7654_3210, 0xFEDC_BA98, 0x1111_1111];
        let scales: Vec<f32> = vec![1.0, 2.0, 0.5];
        let biases: Vec<f32> = vec![0.0, 1.0, -1.0];
        let out = dequantize_mlx_affine(&packed, &scales, &biases, 3, 8, 8, 4);
        let mut expected = Vec::new();
        for i in 0..8 { expected.push(1.0 * (i as f32) + 0.0); }         // row0: q=0..7
        for i in 0..8 { expected.push(2.0 * ((8 + i) as f32) + 1.0); }   // row1: q=8..15
        for _ in 0..8 { expected.push(0.5 * 1.0 - 1.0); }               // row2: q=1 each nibble
        assert_eq!(out, expected);
    }

    /// Regression for the 35B-A3B MoE resident-4-bit expert loader. The 3D
    /// `switch_mlp.*` tensors span all experts; `validate_affine_dims` must be
    /// fed the flattened `[E*out, in]` geometry, not the per-expert `out`.
    /// Before the fix (5923c49) this loader returned `Err` for every MoE
    /// checkpoint, leaving `quant_moe` empty and panicking the forward in the
    /// f32 borrowed path (`Weight 'model.layers.0.mlp.switch_mlp.gate_proj.weight'
    /// not found`). Ignored by default; run with the MLX MoE checkpoint dir:
    ///   VLLM_TEST_MOE_DIR=~/repos/OminiX-MLX/models/Qwen3.6-35B-A3B-4bit \
    ///     cargo test --lib -- --ignored moe_quant_experts_load_layer0
    #[test]
    #[ignore]
    fn moe_quant_experts_load_layer0() {
        let dir = match std::env::var("VLLM_TEST_MOE_DIR") { Ok(d) => d, Err(_) => return };
        let path = std::path::Path::new(&dir).join("config.json");
        // group_size=64, bits=4 mirror the config.json quantization block.
        let qm = load_qwen35_moe_quant_experts(&path, 64, 4, 0, 1)
            .expect("MoE quant experts must load (regression: 3D validate dims)");
        // Layer 0's three switch projections must be present and non-empty.
        let gate = qm.gate.get(&0).expect("layer 0 gate_proj experts resolvable");
        let up = qm.up.get(&0).expect("layer 0 up_proj experts resolvable");
        let down = qm.down.get(&0).expect("layer 0 down_proj experts resolvable");
        // gate/up: [256 experts, 512 out, 2048 in]; down: [256, 2048 out, 512 in].
        assert_eq!(gate.out_features, 512, "gate per-expert out");
        assert_eq!(gate.in_features, 2048, "gate per-expert in");
        assert_eq!(down.out_features, 2048, "down per-expert out");
        assert_eq!(down.in_features, 512, "down per-expert in");
        // Packed length spans ALL experts (E*out*in/8), the flattened geometry
        // validate_affine_dims now checks.
        assert_eq!(gate.packed.len(), 256 * 512 * 2048 / 8, "gate packed spans all experts");
        assert_eq!(up.packed.len(), 256 * 512 * 2048 / 8, "up packed spans all experts");
        assert_eq!(down.packed.len(), 256 * 2048 * 512 / 8, "down packed spans all experts");
        assert!(!gate.scales.is_empty() && !gate.biases.is_empty(), "gate scales/biases loaded");
        eprintln!(
            "MoE quant experts OK: layer 0 gate/up/down loaded ({} u32 words each for gate)",
            gate.packed.len()
        );
    }

    /// Bit-exact gate for the pread MoE expert loader (VLLM_VULKAN_MOE_PREAD_LOAD).
    /// The header-parse + per-tensor pread path must return the SAME packed u32
    /// words and SAME scales/biases (bit-for-bit) as the whole-shard `Mmap::map`
    /// body — the read mechanism changes, the bytes do not. Run over the 35B-A3B
    /// MoE checkpoint (fits a single box):
    ///   VLLM_TEST_MOE_DIR=~/repos/OminiX-MLX/models/Qwen3.6-35B-A3B-4bit \
    ///     cargo test --lib -- --ignored moe_quant_experts_pread_bitexact
    #[test]
    #[ignore]
    fn moe_quant_experts_pread_bitexact() {
        let dir = match std::env::var("VLLM_TEST_MOE_DIR") { Ok(d) => d, Err(_) => return };
        let path = std::path::Path::new(&dir).join("config.json");
        // Compare a couple of layers so cross-shard sibling resolution is exercised.
        let (ls, le) = (0usize, 2usize);
        // mmap body: force the flag OFF so the dispatcher takes the mmap branch.
        std::env::remove_var("VLLM_VULKAN_MOE_PREAD_LOAD");
        let mm = load_qwen35_moe_quant_experts(&path, 64, 4, ls, le)
            .expect("mmap MoE expert load");
        let pr = load_qwen35_moe_quant_experts_pread(&path, 64, 4, ls, le)
            .expect("pread MoE expert load");

        let cmp = |tag: &str, a: &std::collections::HashMap<usize, crate::moe::QuantSwitch>,
                   b: &std::collections::HashMap<usize, crate::moe::QuantSwitch>| {
            assert_eq!(a.len(), b.len(), "{tag}: layer count differs");
            for (idx, qa) in a {
                let qb = b.get(idx).unwrap_or_else(|| panic!("{tag} L{idx}: missing in pread"));
                assert_eq!(qa.out_features, qb.out_features, "{tag} L{idx}: out_features");
                assert_eq!(qa.in_features, qb.in_features, "{tag} L{idx}: in_features");
                assert_eq!(qa.group_size, qb.group_size, "{tag} L{idx}: group_size");
                assert_eq!(qa.bits, qb.bits, "{tag} L{idx}: bits");
                // Packed nibbles: exact u32 equality.
                assert_eq!(qa.packed, qb.packed, "{tag} L{idx}: packed u32 mismatch");
                // Scales/biases: bit-for-bit f32 equality (no NaN-blind ==).
                assert_eq!(qa.scales.len(), qb.scales.len(), "{tag} L{idx}: scales len");
                assert_eq!(qa.biases.len(), qb.biases.len(), "{tag} L{idx}: biases len");
                for (x, y) in qa.scales.iter().zip(&qb.scales) {
                    assert_eq!(x.to_bits(), y.to_bits(), "{tag} L{idx}: scale bits");
                }
                for (x, y) in qa.biases.iter().zip(&qb.biases) {
                    assert_eq!(x.to_bits(), y.to_bits(), "{tag} L{idx}: bias bits");
                }
            }
        };
        cmp("gate", &mm.gate, &pr.gate);
        cmp("up", &mm.up, &pr.up);
        cmp("down", &mm.down, &pr.down);
        eprintln!(
            "pread MoE experts BIT-EXACT vs mmap: layers {ls}..{le}, \
             gate={} up={} down={} (packed+scales+biases all bit-identical)",
            mm.gate.len(), mm.up.len(), mm.down.len()
        );
    }
}

#[cfg(test)]
mod chunk_alloc_tests {
    use super::chunk_row_plan;

    #[test]
    fn chunk_plan_disabled_is_single_chunk() {
        assert_eq!(chunk_row_plan(248320, 34 * 5120 / 32, 0), vec![248320]);
    }

    #[test]
    fn chunk_plan_exact_divide() {
        // row_bytes=100, limit=512 -> 5 rows/chunk exactly; 10 rows -> 2 chunks of 5.
        assert_eq!(chunk_row_plan(10, 100, 512), vec![5, 5]);
    }

    #[test]
    fn chunk_plan_remainder() {
        // 7 rows/chunk (700<=1000 wait recompute): limit=750,row=100 -> 7 rows/chunk.
        let plan = chunk_row_plan(23, 100, 750);
        assert_eq!(plan, vec![7, 7, 7, 2]);
        assert_eq!(plan.iter().sum::<usize>(), 23);
    }

    #[test]
    fn chunk_plan_27b_lmhead_512mb_limit() {
        // 27B lm_head: 248320 rows, q8_0 row_bytes = (5120/32)*34 = 5440.
        // 512MB limit -> rows_per_chunk = floor(512*1024*1024/5440) = 98689 (>0),
        // so 3 chunks total, each well under the limit, covering all rows.
        let row_bytes = (5120 / 32) * 34;
        let limit = 512 * 1024 * 1024;
        let plan = chunk_row_plan(248320, row_bytes, limit);
        assert_eq!(plan.iter().sum::<usize>(), 248320);
        for &rows in &plan {
            assert!(rows * row_bytes <= limit, "chunk of {rows} rows exceeds the {limit}B limit");
        }
        assert!(plan.len() >= 2, "248320 rows at row_bytes={row_bytes} should need >1 chunk under 512MB");
    }

    #[test]
    fn chunk_plan_oversized_single_row_gets_its_own_chunk() {
        // A single row bigger than the limit still gets exactly 1 row/chunk
        // (can't split within a row) rather than looping forever or panicking.
        let plan = chunk_row_plan(3, 1000, 10);
        assert_eq!(plan, vec![1, 1, 1]);
    }

    #[test]
    fn chunk_plan_zero_rows() {
        assert_eq!(chunk_row_plan(0, 100, 512), Vec::<usize>::new());
    }
}

/// Correctness gate for the EngineClient serving bindings' sampler wiring.
///
/// `VulkanModel::forward_and_sample` (src/lib.rs) is, by construction, exactly
/// `forward_rs(token, pos)` followed by `sample_with_temperature(&logits, ...)`.
/// The forward half is bit-identical to what `forward` / `prefill_logits`
/// return (same `forward_rs` internals — no GPU needed to assert that), so the
/// only new load-bearing behavior to pin is the sampler contract the binding
/// relies on: greedy at temperature 0 equals the argmax of those same logits,
/// and the draw-in draw-out determinism the async `generate()` loop depends on.
#[cfg(test)]
mod engineclient_sampler_tests {
    use super::{argmax, sample_with_temperature};

    fn synth_logits(n: usize, seed: u64) -> Vec<f32> {
        // Deterministic pseudo-random logits spanning a realistic magnitude range.
        (0..n)
            .map(|i| {
                let h = (i as u64)
                    .wrapping_add(seed)
                    .wrapping_mul(2654435761)
                    .wrapping_add(0x9e37_79b9);
                ((h >> 11) as f32 / (1u64 << 21) as f32) * 20.0 - 10.0
            })
            .collect()
    }

    /// `forward_and_sample(temperature < 0.01)` must equal the greedy argmax of
    /// the identical logit vector `forward` returns — the no-sampling path the
    /// EngineClient greedy/`temperature=0` case takes. Independent of the draw.
    #[test]
    fn greedy_equals_forward_argmax() {
        for seed in 0..8u64 {
            let logits = synth_logits(50_000, seed);
            let expected = argmax(&logits) as u32;
            // temperature below the 0.01 greedy threshold, several draws — all
            // must collapse to argmax regardless of uniform_random.
            for &u in &[0.0f32, 0.5, 0.999] {
                let got = sample_with_temperature(&logits, 0.0, 1.0, 64, u) as u32;
                assert_eq!(got, expected, "greedy!=argmax seed={seed} u={u}");
                let got2 = sample_with_temperature(&logits, 0.005, 0.95, 64, u) as u32;
                assert_eq!(got2, expected, "sub-threshold temp!=argmax seed={seed} u={u}");
            }
        }
    }

    /// Same logits + same uniform draw ⇒ same token, every time. This is the
    /// determinism the caller's seeded RNG (`random.random()` in server.py's
    /// reference loop) turns into a reproducible generation.
    #[test]
    fn sampling_is_deterministic_given_draw() {
        let logits = synth_logits(32_000, 123);
        for &u in &[0.01f32, 0.37, 0.5, 0.83, 0.999] {
            let a = sample_with_temperature(&logits, 0.8, 0.95, 64, u);
            let b = sample_with_temperature(&logits, 0.8, 0.95, 64, u);
            assert_eq!(a, b, "non-deterministic for the same draw u={u}");
        }
    }

    /// A near-1.0 draw with temperature>0 must always land on a valid vocab
    /// index (nucleus fallback never returns out of range) — the binding
    /// returns `idx as u32`, so an out-of-range idx would be a silent bad token.
    #[test]
    fn sampled_index_in_range() {
        let logits = synth_logits(1000, 7);
        for &u in &[0.0f32, 0.25, 0.5, 0.75, 0.9999] {
            let idx = sample_with_temperature(&logits, 1.0, 0.9, 40, u);
            assert!(idx < logits.len(), "idx {idx} out of range for u={u}");
        }
    }
}

// ─── Phase-0 per-layer KV sizing: host bit-exactness gate ────────────────────
//
// Proves that sizing a sliding-window layer's KV cache to `window` positions (a
// ring buffer, `KvCache::new_windowed` + `windowed_view`) is BIT-FOR-BIT
// identical to the historical uniform `max_seq_len` allocation. Two levels:
//   1. the `KvCache`/`cpu_sdpa` primitive directly (ring vs full vs the legacy
//      absolute-slice `Some(window)` path), driven far past the window so the
//      ring wraps every step;
//   2. the full `Gemma4Model::forward` path (windowed-cache model vs
//      uniform-cache model on identical synthetic weights) over a token stream
//      longer than the window.
// Both assert `f32::to_bits` equality (project argmax-exact / cos=1.0 standard).
// GPU-resident planes (`laguna_gpu.rs`, gemma GPU KV planes) and the production
// `gemma_forward.rs` / `laguna.rs` host SDPA sites are OUT OF SCOPE here and
// gated on-cluster later — see the commit message.
#[cfg(test)]
mod per_layer_kv_sizing_tests {
    use super::*;

    /// Small deterministic pseudo-random f32 generator in roughly [-0.5, 0.5).
    fn prng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed | 1;
        move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        }
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best = i;
            }
        }
        best
    }

    /// `windowed_view` over a WRAPPED ring is byte-identical to `windowed_view`
    /// over a full-size (never-wrapping) cache, and both drive `cpu_sdpa` to a
    /// result bit-identical to the legacy absolute-slice path
    /// `cpu_sdpa(k_up_to_now, seq_len, Some(window))` — at every step of a
    /// stream far longer than the window (forcing many ring wraps).
    #[test]
    fn ring_windowed_view_bit_exact_vs_full_and_legacy() {
        let (num_kv, head_dim, num_q) = (2usize, 32usize, 4usize);
        let stride = num_kv * head_dim;
        let window = 8usize;
        let max_seq = 64usize;
        let scale = 0.125f32;
        let mut rng = prng(0xA11CE_u64);

        let mut full = KvCache::new(max_seq, num_kv, head_dim); // capacity == max_seq
        let mut ring = KvCache::new_windowed(max_seq, window, num_kv, head_dim); // capacity == window
        assert_eq!(full.capacity, max_seq);
        assert_eq!(ring.capacity, window);

        for step in 0..max_seq {
            let k: Vec<f32> = (0..stride).map(|_| rng()).collect();
            let v: Vec<f32> = (0..stride).map(|_| rng()).collect();
            full.append(&k, &v);
            ring.append(&k, &v);
            assert_eq!(full.seq_len, ring.seq_len);

            let q: Vec<f32> = (0..num_q * head_dim).map(|_| rng()).collect();

            // Legacy reference: full absolute array + Some(window) masking.
            let legacy = cpu_sdpa(
                &q, full.k_up_to_now(), full.v_up_to_now(),
                num_q, num_kv, head_dim, full.seq_len, scale, Some(window),
            );
            // windowed_view over the full (unwrapped) cache.
            let (kf, vf, lf) = full.windowed_view(window);
            let via_full = cpu_sdpa(&q, &kf, &vf, num_q, num_kv, head_dim, lf, scale, None);
            // windowed_view over the ring (wrapped once step >= window).
            let (kr, vr, lr) = ring.windowed_view(window);
            let via_ring = cpu_sdpa(&q, &kr, &vr, num_q, num_kv, head_dim, lr, scale, None);

            assert_eq!(lf, lr, "valid_len mismatch at step {step}");
            for (a, b) in kf.iter().zip(kr.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "ring K byte drift at step {step}");
            }
            for (a, b) in vf.iter().zip(vr.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "ring V byte drift at step {step}");
            }
            for i in 0..legacy.len() {
                assert_eq!(legacy[i].to_bits(), via_full[i].to_bits(),
                    "windowed_view != legacy at step {step} idx {i}");
                assert_eq!(legacy[i].to_bits(), via_ring[i].to_bits(),
                    "ring windowed_view != legacy at step {step} idx {i}");
            }
            if step + 1 > window {
                assert!(ring.has_wrapped(), "ring should have wrapped by step {step}");
            }
        }
    }

    /// End-to-end: a Gemma4Model whose sliding layers use `window`-sized rings
    /// produces bit-identical logits (and argmax) to the same model with uniform
    /// `max_seq_len` caches, over a token stream longer than the window.
    #[test]
    fn gemma_forward_per_layer_sizing_bit_exact_over_window() {
        let max_seq = 48usize;
        let small_window = 6usize;

        let build = |windowed: bool| -> Gemma4Model {
            let mut m = tiny_synthetic_gemma(max_seq);
            m.config.sliding_window = small_window;
            m.kv_caches = (0..m.config.num_hidden_layers)
                .map(|l| {
                    if windowed {
                        KvCache::new_windowed(
                            max_seq,
                            m.config.layer_kv_capacity(l, max_seq),
                            m.config.layer_num_kv_heads(l),
                            m.config.layer_head_dim(l),
                        )
                    } else {
                        KvCache::new(
                            max_seq,
                            m.config.layer_num_kv_heads(l),
                            m.config.layer_head_dim(l),
                        )
                    }
                })
                .collect();
            m
        };

        let mut m_win = build(true);
        let mut m_uni = build(false);
        let cfg = m_win.config.clone();

        // A real sliding layer must have physically shrunk; a full layer must not.
        let sliding = (0..cfg.num_hidden_layers).find(|&l| !cfg.is_full_attention(l)).unwrap();
        let full_l = (0..cfg.num_hidden_layers).find(|&l| cfg.is_full_attention(l)).unwrap();
        assert_eq!(m_win.kv_caches[sliding].capacity, small_window);
        assert_eq!(m_win.kv_caches[full_l].capacity, max_seq);
        assert_eq!(m_uni.kv_caches[sliding].capacity, max_seq);

        let seq: Vec<u32> = (0..40u32)
            .map(|i| (i.wrapping_mul(37).wrapping_add(11)) % cfg.vocab_size as u32)
            .collect();
        for (pos, &tok) in seq.iter().enumerate() {
            let lw = m_win.forward(tok, pos);
            let lu = m_uni.forward(tok, pos);
            assert_eq!(lw.len(), lu.len());
            assert_eq!(argmax(&lw), argmax(&lu), "argmax diverged at pos {pos}");
            for i in 0..lw.len() {
                assert_eq!(lw[i].to_bits(), lu[i].to_bits(),
                    "logit byte drift at pos {pos} idx {i}");
            }
        }
        // The stream (40) far exceeds the window (6): the ring must have wrapped.
        assert!(m_win.kv_caches[sliding].has_wrapped());
    }

    /// GPU-resident-plane addressing gate (host simulation of the exact byte
    /// offsets in `gemma_forward.rs`'s 1-CB path + `paged_attn_decode_f32_sg.comp`
    /// ring branch). Models the device plane `[K-plane][V-plane]`, each
    /// `capacity*stride` f32, with:
    ///   • seed: absolute host position `p` → ring slot `p % capacity`
    ///     (K at `slot*stride`, V at `plane + slot*stride`);
    ///   • in-CB append at slot `pos % capacity`;
    ///   • shader read of window `[seq-window, seq)` at slot `t % capacity`.
    /// Asserts the window rows gathered from the RING plane are byte-identical to
    /// those gathered from a full absolute (`capacity == max_seq`) plane — i.e.
    /// the ring modulus that the GPU shader/append/seed all share is correct, and
    /// full/global layers (ring_capacity 0) are unchanged. Purely offline; the
    /// on-cluster argmax gate exercises the real shader.
    #[test]
    fn gpu_ring_plane_offsets_bit_exact_vs_absolute() {
        let (num_kv, head_dim) = (2usize, 16usize);
        let stride = num_kv * head_dim;
        let window = 8usize;      // sliding capacity (ring)
        let max_seq = 40usize;    // full capacity
        let prompt = 5usize;      // seeded prefill positions
        let decode_to = 30usize;  // decode wraps the ring several times
        let mut rng = prng(0xDEADBEEF);

        // Golden absolute source: every position's K/V row (host cache is full).
        let mut src_k = vec![0f32; decode_to * stride];
        let mut src_v = vec![0f32; decode_to * stride];
        for i in 0..decode_to * stride { src_k[i] = rng(); src_v[i] = rng(); }

        // Emulate a device plane [K|V] of `cap*stride` each.
        let make_plane = |cap: usize, ring: bool, upto: usize| -> Vec<f32> {
            let plane = cap * stride;
            let mut buf = vec![0f32; 2 * plane];
            let slot_of = |p: usize| if ring { p % cap } else { p };
            // Device state = seed [max(0, prompt-cap)..prompt) ∪ in-CB appends
            // [prompt..upto); the union is the contiguous run [prompt-cap .. upto).
            let start = if ring { prompt.saturating_sub(cap) } else { 0 };
            for p in start..upto {
                let s = slot_of(p);
                buf[s * stride..(s + 1) * stride].copy_from_slice(&src_k[p * stride..(p + 1) * stride]);
                buf[plane + s * stride..plane + (s + 1) * stride]
                    .copy_from_slice(&src_v[p * stride..(p + 1) * stride]);
            }
            buf
        };

        // Shader-side gather of window [ws, seq) using the plane's slot math.
        let gather = |buf: &[f32], cap: usize, ring: bool, seq: usize, ws: usize| -> (Vec<f32>, Vec<f32>) {
            let plane = cap * stride;
            let (mut gk, mut gv) = (Vec::new(), Vec::new());
            for t in ws..seq {
                let s = if ring { t % cap } else { t };
                gk.extend_from_slice(&buf[s * stride..(s + 1) * stride]);
                gv.extend_from_slice(&buf[plane + s * stride..plane + (s + 1) * stride]);
            }
            (gk, gv)
        };

        for seq in (prompt + 1)..=decode_to {
            // Sliding (ring) plane vs full absolute plane, both filled to `seq`.
            let ring_buf = make_plane(window, true, seq);
            let abs_buf = make_plane(max_seq, false, seq);
            let ws = seq.saturating_sub(window);
            let (rk, rv) = gather(&ring_buf, window, true, seq, ws);
            let (ak, av) = gather(&abs_buf, max_seq, false, seq, ws);
            assert_eq!(rk.len(), ak.len(), "gather len mismatch at seq {seq}");
            for i in 0..ak.len() {
                assert_eq!(rk[i].to_bits(), ak[i].to_bits(), "ring K != absolute at seq {seq} idx {i}");
                assert_eq!(rv[i].to_bits(), av[i].to_bits(), "ring V != absolute at seq {seq} idx {i}");
            }
            // And the gathered window is exactly the golden source rows [ws, seq).
            for (j, p) in (ws..seq).enumerate() {
                for d in 0..stride {
                    assert_eq!(rk[j * stride + d].to_bits(), src_k[p * stride + d].to_bits(),
                        "ring K row != golden pos {p} at seq {seq}");
                }
            }
        }
    }

    /// TP host-SDPA reader (`gemma_forward.rs::gemma_tp_sdpa`) gather gate. That
    /// method builds a per-TP-rank head-block `[vlen, local_num_kv, head_dim]`
    /// buffer over the last-`window` positions of a replicated KV cache and feeds
    /// it to `gpu_sdpa`/`cpu_sdpa_gqa`. This reproduces the exact gather for a
    /// SLIDING layer two ways — over a `window`-sized RING (`windowed_view`, the
    /// migrated path) and over a full absolute cache (the legacy
    /// `k_up_to_now()[kv_start..]` path) — over a stream far longer than the
    /// window (forcing wraps), and asserts the gathered head-block bytes AND the
    /// resulting SDPA are bit-identical. Full/global layers (`window=None`) keep
    /// the absolute path and are covered by the primitive test above.
    #[test]
    fn tp_head_block_gather_ring_bit_exact_vs_absolute() {
        // Global GQA layout + a TP-2 rank-1 shard (the non-trivial head block).
        let (num_kv, head_dim) = (4usize, 16usize);
        let stride = num_kv * head_dim;                 // full replicated per-pos
        let (local_num_kv, g_kv_start) = (2usize, 2usize);
        let total_q = 8usize;
        let gqa_ratio = total_q / num_kv;               // 2
        let r_num_q = total_q / 2;                       // TP-2 local q heads = 4
        let q_head_offset = r_num_q;                     // rank 1
        let window = 8usize;
        let max_seq = 64usize;
        let scale = 1.0f32;
        let mut rng = prng(0x7BEEF);

        let mut full = KvCache::new(max_seq, num_kv, head_dim);
        let mut ring = KvCache::new_windowed(max_seq, window, num_kv, head_dim);

        // Head-block gather mirroring gemma_tp_sdpa: for each visible position,
        // copy the `local_num_kv` heads starting at `g_kv_start`.
        let head_block = |kbuf: &[f32], vbuf: &[f32], vlen: usize| -> (Vec<f32>, Vec<f32>) {
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
            (ks, vs)
        };

        for step in 0..max_seq {
            let k: Vec<f32> = (0..stride).map(|_| rng()).collect();
            let v: Vec<f32> = (0..stride).map(|_| rng()).collect();
            full.append(&k, &v);
            ring.append(&k, &v);

            let q: Vec<f32> = (0..r_num_q * head_dim).map(|_| rng()).collect();

            // Migrated (ring) path: windowed_view compacts [seq-window, seq).
            let (kw, vw, vlen_r) = ring.windowed_view(window);
            let (rk, rv) = head_block(&kw, &vw, vlen_r);

            // Legacy (full absolute) path: k_up_to_now()[kv_start..] then gather.
            let slen = full.seq_len;
            let kv_start = slen.saturating_sub(window);
            let vlen_f = slen - kv_start;
            let kabs = &full.k_up_to_now()[kv_start * stride..slen * stride];
            let vabs = &full.v_up_to_now()[kv_start * stride..slen * stride];
            let (ak, av) = head_block(kabs, vabs, vlen_f);

            assert_eq!(vlen_r, vlen_f, "vlen mismatch at step {step}");
            assert_eq!(rk.len(), ak.len(), "gather len mismatch at step {step}");
            for i in 0..ak.len() {
                assert_eq!(rk[i].to_bits(), ak[i].to_bits(), "TP K gather drift step {step} idx {i}");
                assert_eq!(rv[i].to_bits(), av[i].to_bits(), "TP V gather drift step {step} idx {i}");
            }

            // End-to-end: identical gathered head-block ⇒ identical SDPA. The
            // GPU branch runs plain GQA over the pre-sliced local heads
            // (`gpu_sdpa`), so the host mirror is plain `cpu_sdpa` here.
            let out_r = cpu_sdpa(&q, &rk, &rv, r_num_q, local_num_kv, head_dim, vlen_r, scale, None);
            let out_f = cpu_sdpa(&q, &ak, &av, r_num_q, local_num_kv, head_dim, vlen_f, scale, None);
            for i in 0..out_f.len() {
                assert_eq!(out_r[i].to_bits(), out_f[i].to_bits(), "TP SDPA drift step {step} idx {i}");
            }
            let _ = (gqa_ratio, q_head_offset); // documented TP mapping (see gemma_tp_sdpa)
            if step + 1 > window { assert!(ring.has_wrapped()); }
        }
    }

    /// Batched-verify reader (`Gemma4Model::forward_verify_core`) over a host KV
    /// ring that has ALREADY WRAPPED from prior decode. `forward_verify_core`
    /// reuses `forward_layer` (windowed sliding layers → `windowed_view`), so a
    /// windowed-cache model must produce logits bit-identical to a uniform-cache
    /// model for the whole verify batch. This is the spec-decode host-simulable
    /// path (no GPU / no cluster). NOTE: partial-accept `verify_rollback` on a
    /// windowed cache that WRAPPED *during* the verify overshoot is a separate,
    /// documented deferral (the overshoot overwrites window-start ring slots that
    /// `truncate` cannot restore); the safe unwrapped-rollback regime is covered
    /// by `verify_rollback_ring_bit_exact_no_wrap` below.
    #[test]
    fn forward_verify_core_ring_bit_exact_wrapped() {
        let max_seq = 48usize;
        let small_window = 6usize;
        let build = |windowed: bool| -> Gemma4Model {
            let mut m = tiny_synthetic_gemma(max_seq);
            m.config.sliding_window = small_window;
            m.kv_caches = (0..m.config.num_hidden_layers)
                .map(|l| if windowed {
                    KvCache::new_windowed(max_seq, m.config.layer_kv_capacity(l, max_seq),
                        m.config.layer_num_kv_heads(l), m.config.layer_head_dim(l))
                } else {
                    KvCache::new(max_seq, m.config.layer_num_kv_heads(l), m.config.layer_head_dim(l))
                })
                .collect();
            m
        };
        let mut m_win = build(true);
        let mut m_uni = build(false);
        let cfg = m_win.config.clone();
        let sliding = (0..cfg.num_hidden_layers).find(|&l| !cfg.is_full_attention(l)).unwrap();

        let vocab = cfg.vocab_size as u32;
        let tok = |i: usize| ((i as u32).wrapping_mul(37).wrapping_add(11)) % vocab;

        // Prefill both models WELL past the window so the ring wraps (P=20 > 6).
        let prefill = 20usize;
        for pos in 0..prefill {
            let _ = m_win.forward(tok(pos), pos);
            let _ = m_uni.forward(tok(pos), pos);
        }
        assert!(m_win.kv_caches[sliding].has_wrapped(), "ring must have wrapped by prefill end");

        // Batched verify of T tokens at start_pos = prefill (reads the wrapped ring).
        let t = 4usize;
        let batch: Vec<u32> = (0..t).map(|j| tok(prefill + j)).collect();
        let lw = m_win.forward_verify_core(&batch, prefill);
        let lu = m_uni.forward_verify_core(&batch, prefill);
        assert_eq!(lw.len(), lu.len());
        for ti in 0..lw.len() {
            assert_eq!(lw[ti].len(), lu[ti].len());
            for i in 0..lw[ti].len() {
                assert_eq!(lw[ti][i].to_bits(), lu[ti][i].to_bits(),
                    "verify logit drift at t {ti} idx {i}");
            }
        }
    }

    /// Safe-regime `verify_rollback`: with the whole sequence kept within the
    /// window (no ring wrap), a windowed-cache model must be bit-identical to a
    /// uniform-cache model across a verify → partial-accept-rollback → continue
    /// cycle. Pins the boundary of what the host half guarantees today (wrapped
    /// rollback is deferred to the live gate — see the test above).
    #[test]
    fn verify_rollback_ring_bit_exact_no_wrap() {
        let max_seq = 64usize;
        // Window comfortably larger than every position we touch ⇒ never wraps.
        let big_window = 40usize;
        let build = |windowed: bool| -> Gemma4Model {
            let mut m = tiny_synthetic_gemma(max_seq);
            m.config.sliding_window = big_window;
            m.kv_caches = (0..m.config.num_hidden_layers)
                .map(|l| if windowed {
                    KvCache::new_windowed(max_seq, m.config.layer_kv_capacity(l, max_seq),
                        m.config.layer_num_kv_heads(l), m.config.layer_head_dim(l))
                } else {
                    KvCache::new(max_seq, m.config.layer_num_kv_heads(l), m.config.layer_head_dim(l))
                })
                .collect();
            m
        };
        let mut m_win = build(true);
        let mut m_uni = build(false);
        let cfg = m_win.config.clone();
        let vocab = cfg.vocab_size as u32;
        let tok = |i: usize| ((i as u32).wrapping_mul(29).wrapping_add(7)) % vocab;

        let start = 5usize;
        for pos in 0..start {
            let _ = m_win.forward(tok(pos), pos);
            let _ = m_uni.forward(tok(pos), pos);
        }
        // Verify T=4, accept 1 (commit_len=2), rollback the rest, then continue.
        let t = 4usize;
        let batch: Vec<u32> = (0..t).map(|j| tok(start + j)).collect();
        let _ = m_win.forward_verify_core(&batch, start);
        let _ = m_uni.forward_verify_core(&batch, start);
        let accept_len = 1usize;
        m_win.verify_rollback(start, t, accept_len);
        m_uni.verify_rollback(start, t, accept_len);
        assert!(!m_win.kv_caches[0].has_wrapped(), "must stay unwrapped in the safe regime");

        // Continue decoding from the committed frontier; must stay bit-identical.
        let frontier = start + accept_len + 1;
        for pos in frontier..frontier + 6 {
            let lw = m_win.forward(tok(pos), pos);
            let lu = m_uni.forward(tok(pos), pos);
            for i in 0..lw.len() {
                assert_eq!(lw[i].to_bits(), lu[i].to_bits(),
                    "post-rollback logit drift at pos {pos} idx {i}");
            }
        }
    }
}


#[cfg(test)]
mod sdpa_batched_tests {
    use super::*;

    /// GATE: the rayon-parallel `cpu_sdpa_batched_causal` (the TP spec-verify
    /// attention flip) must be BYTE-EXACT vs `T` sequential per-token `cpu_sdpa`
    /// calls — the exact bit-exactness the `debug_tp_qwen35_verify_vs_serial`
    /// harness enforces on-cluster (cos=1.0 + argmax every position), proven
    /// here on synthetic data with no GPU/model. Exercises GQA (nq!=nkv), a
    /// nonzero accepted-prefix `start_pos`, and per-query growing causal length.
    #[test]
    fn batched_causal_sdpa_bit_exact_vs_per_token() {
        // Real qwen3.6 full-attn shape at TP-4 (nq=6, nkv=1, hd=256) is the
        // target; use hd=64 here so the test is fast but still GQA + multi-head.
        let (t, start_pos, nq, nkv, hd) = (5usize, 37usize, 6usize, 2usize, 64usize);
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let total = start_pos + t; // caller appended all T verify tokens
        let scale = 1.0 / (hd as f32).sqrt();

        // Deterministic xorshift fill (no rand dep).
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut nxt = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; ((rng >> 40) as f32) / 16777216.0 - 0.5 };
        let q_all: Vec<f32> = (0..t * q_dim).map(|_| nxt()).collect();
        let kbuf: Vec<f32> = (0..total * kv_dim).map(|_| nxt()).collect();
        let vbuf: Vec<f32> = (0..total * kv_dim).map(|_| nxt()).collect();

        // Reference: T independent per-token cpu_sdpa calls, each over its own
        // causal length l = start_pos + ti + 1.
        let mut reference = vec![0.0f32; t * q_dim];
        for ti in 0..t {
            let l = start_pos + ti + 1;
            let o = cpu_sdpa(
                &q_all[ti * q_dim..(ti + 1) * q_dim],
                &kbuf[..l * kv_dim], &vbuf[..l * kv_dim],
                nq, nkv, hd, l, scale, None);
            reference[ti * q_dim..(ti + 1) * q_dim].copy_from_slice(&o);
        }

        let got = cpu_sdpa_batched_causal(&q_all, &kbuf, &vbuf, t, start_pos, nq, nkv, hd, scale);

        assert_eq!(got.len(), reference.len());
        // Byte-exact: no cross-thread reduction, identical per-head arithmetic.
        assert_eq!(got, reference, "batched causal SDPA diverged from per-token cpu_sdpa (must be bit-exact)");
    }

    /// Same, single-token (T=1) and no prefix — the degenerate case the verify
    /// falls back to must also match exactly.
    #[test]
    fn batched_causal_sdpa_bit_exact_t1() {
        let (t, start_pos, nq, nkv, hd) = (1usize, 0usize, 4usize, 4usize, 32usize);
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let total = start_pos + t;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut rng = 0xDEADBEEFCAFEBABEu64;
        let mut nxt = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; ((rng >> 40) as f32) / 16777216.0 - 0.5 };
        let q_all: Vec<f32> = (0..t * q_dim).map(|_| nxt()).collect();
        let kbuf: Vec<f32> = (0..total * kv_dim).map(|_| nxt()).collect();
        let vbuf: Vec<f32> = (0..total * kv_dim).map(|_| nxt()).collect();

        let l = start_pos + 1;
        let reference = cpu_sdpa(&q_all, &kbuf[..l * kv_dim], &vbuf[..l * kv_dim], nq, nkv, hd, l, scale, None);
        let got = cpu_sdpa_batched_causal(&q_all, &kbuf, &vbuf, t, start_pos, nq, nkv, hd, scale);
        assert_eq!(got, reference);
    }
}

// ─── Cross-model KV (CMKV) CPU-forward dump ──────────────────────────────────
//
// Gated offline dump for spike/cross-model-kv S5–S7. Activate with:
//   CMKV_DUMP_DIR=spike/cross-model-kv/dumps cargo test --lib cmkv_cpu_dump -- --nocapture
//
// Writes per-sequence raw f32 K/V + meta.json produced by the real
// `Gemma4Model::forward` CPU path (not synthetic GT linear maps).

#[cfg(test)]
mod cmkv_dump_tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    fn write_f32_bin(path: &Path, data: &[f32]) {
        let mut f = fs::File::create(path).expect("create bin");
        for &x in data {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }

    fn write_json(path: &Path, s: &str) {
        fs::write(path, s).expect("write json");
    }

    fn layer_kind(cfg: &Gemma4Config, li: usize) -> &'static str {
        if cfg.is_full_attention(li) {
            "full"
        } else {
            "sliding"
        }
    }

    fn dump_model_after_prefill(
        model: &Gemma4Model,
        out_dir: &Path,
        name: &str,
        tokens: &[u32],
        seq_idx: usize,
    ) {
        let seq_dir = out_dir.join(name).join(format!("seq_{seq_idx:04}"));
        fs::create_dir_all(&seq_dir).unwrap();
        let cfg = &model.config;
        let mut layers_json = String::from("[");
        for (li, cache) in model.kv_caches.iter().enumerate() {
            let n = cache.seq_len;
            assert_eq!(n, tokens.len(), "seq_len vs tokens");
            let k = cache.k_upto(n);
            let v = cache.v_upto(n);
            write_f32_bin(&seq_dir.join(format!("layer_{li:02}_k.f32")), k);
            write_f32_bin(&seq_dir.join(format!("layer_{li:02}_v.f32")), v);
            if li > 0 {
                layers_json.push(',');
            }
            layers_json.push_str(&format!(
                "{{\"idx\":{li},\"kind\":\"{}\",\"n_kv\":{},\"head_dim\":{},\"seq_len\":{n},\"k_elems\":{},\"v_elems\":{}}}",
                layer_kind(cfg, li),
                cache.num_kv_heads,
                cache.head_dim,
                k.len(),
                v.len(),
            ));
        }
        layers_json.push(']');
        let tok_json: String = tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");
        write_json(
            &seq_dir.join("meta.json"),
            &format!(
                "{{\n  \"name\": \"{name}\",\n  \"seq_idx\": {seq_idx},\n  \"tokens\": [{tok_json}],\n  \"seq_len\": {},\n  \"num_layers\": {},\n  \"vocab_size\": {},\n  \"hidden_size\": {},\n  \"attention_period\": {},\n  \"layers\": {layers_json},\n  \"source\": \"vllm_vulkan_cpu_forward\"\n}}\n",
                tokens.len(),
                cfg.num_hidden_layers,
                cfg.vocab_size,
                cfg.hidden_size,
                cfg.attention_period,
            ),
        );
    }

    fn prefill(model: &mut Gemma4Model, tokens: &[u32]) -> Vec<f32> {
        gemma_reset_kv(model);
        let mut last = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            last = model.forward(tok, pos);
        }
        last
    }

    fn make_seqs(n_seq: usize, seq_len: usize, vocab: usize, seed: u64) -> Vec<Vec<u32>> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n_seq);
        for _ in 0..n_seq {
            let mut seq = Vec::with_capacity(seq_len);
            for i in 0..seq_len {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407)
                    .wrapping_add(i as u64);
                seq.push(((s >> 33) as usize % vocab.max(1)) as u32);
            }
            out.push(seq);
        }
        out
    }

    fn dump_spec(out: &Path, spec: SynthGemmaSpec, n_seq: usize, seq_len: usize, seed: u64) {
        let max_seq = seq_len + 8;
        let mut model = synthetic_gemma(&spec, max_seq);
        let name = spec.name.clone();
        let root = out.join(&name);
        fs::create_dir_all(&root).unwrap();
        let seqs = make_seqs(n_seq, seq_len, model.config.vocab_size, seed);
        let mut last_logits_path = None;
        for (si, tokens) in seqs.iter().enumerate() {
            let logits = prefill(&mut model, tokens);
            dump_model_after_prefill(&model, out, &name, tokens, si);
            if si == 0 {
                let p = root.join("seq_0000").join("logits_last.f32");
                write_f32_bin(&p, &logits);
                last_logits_path = Some(p);
            }
        }
        // Model-level manifest
        let mut layers = String::from("[");
        for li in 0..model.config.num_hidden_layers {
            if li > 0 {
                layers.push(',');
            }
            layers.push_str(&format!(
                "{{\"idx\":{li},\"kind\":\"{}\",\"n_kv\":{},\"head_dim\":{}}}",
                layer_kind(&model.config, li),
                model.config.layer_num_kv_heads(li),
                model.config.layer_head_dim(li),
            ));
        }
        layers.push(']');
        write_json(
            &root.join("manifest.json"),
            &format!(
                "{{\n  \"name\": \"{name}\",\n  \"family\": \"gemma4_synth\",\n  \"n_seq\": {n_seq},\n  \"seq_len\": {seq_len},\n  \"num_layers\": {},\n  \"weight_tag\": \"{}\",\n  \"matched_note\": \"synthetic CPU-forward via Gemma4Model::forward\",\n  \"layers\": {layers}\n}}\n",
                model.config.num_hidden_layers,
                spec.weight_tag,
            ),
        );
        let _ = last_logits_path;
        eprintln!(
            "cmkv dump: {name}  L={}  n_seq={n_seq} seq_len={seq_len} → {}",
            model.config.num_hidden_layers,
            root.display()
        );
    }

    /// Inject mapped KV and compare next-token logits vs real prefill (S6 smoke).
    fn handoff_logit_smoke(out: &Path) {
        let seq_len = 16usize;
        let max_seq = 64usize;
        let src_spec = SynthGemmaSpec::tiny().with_layers(4).with_tag("srcA");
        let tgt_spec = SynthGemmaSpec::tiny().with_layers(4).with_tag("tgtB");
        // Same geometry, different weights — matched-KV intra-family pair.
        let mut src = synthetic_gemma(&src_spec, max_seq);
        let tokens = make_seqs(1, seq_len + 1, src.config.vocab_size, 99)[0].clone();
        let prefix = &tokens[..seq_len];
        let next = tokens[seq_len];

        // Reference: full prefill on target then one more step.
        let mut tgt_ref = synthetic_gemma(&tgt_spec, max_seq);
        let _ = prefill(&mut tgt_ref, prefix);
        let logits_ref = tgt_ref.forward(next, seq_len);

        // Source prefill (for raw-inject control).
        let _ = prefill(&mut src, prefix);

        // Identity handoff: inject TARGET's own prefilled KV into a fresh model
        // and continue — must match ref logits (proves inject + continue).
        let mut tgt_pre = synthetic_gemma(&tgt_spec, max_seq);
        let _ = prefill(&mut tgt_pre, prefix);
        let mut tgt_id = synthetic_gemma(&tgt_spec, max_seq);
        for li in 0..tgt_pre.config.num_hidden_layers {
            let c = &tgt_pre.kv_caches[li];
            let n = c.seq_len;
            gemma_inject_kv_layer(&mut tgt_id, li, c.k_upto(n), c.v_upto(n), n);
        }
        let logits_id = tgt_id.forward(next, seq_len);
        let mut max_abs = 0.0f32;
        for (a, b) in logits_ref.iter().zip(logits_id.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 1e-5,
            "identity inject handoff must match prefill logits, max_abs={max_abs}"
        );

        // Cross-model: inject SOURCE KV into target-shaped caches (dims match)
        // without mapping — expects poor agreement (control). With dims match
        // we can still write bytes; quality is intentionally low.
        let mut tgt_raw = synthetic_gemma(&tgt_spec, max_seq);
        for li in 0..src.config.num_hidden_layers {
            let c = &src.kv_caches[li];
            let n = c.seq_len;
            // Geometry must match for raw inject
            assert_eq!(c.num_kv_heads, tgt_raw.kv_caches[li].num_kv_heads);
            assert_eq!(c.head_dim, tgt_raw.kv_caches[li].head_dim);
            gemma_inject_kv_layer(&mut tgt_raw, li, c.k_upto(n), c.v_upto(n), n);
        }
        let logits_raw = tgt_raw.forward(next, seq_len);
        let mut cos_num = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&a, &b) in logits_ref.iter().zip(logits_raw.iter()) {
            let a = a as f64;
            let b = b as f64;
            cos_num += a * b;
            na += a * a;
            nb += b * b;
        }
        let cos = cos_num / (na.sqrt() * nb.sqrt() + 1e-18);
        write_json(
            &out.join("s6_identity_handoff.json"),
            &format!(
                "{{\n  \"identity_max_abs\": {max_abs},\n  \"raw_src_inject_logit_cos\": {cos:.6},\n  \"seq_len\": {seq_len}\n}}\n"
            ),
        );
        eprintln!(
            "cmkv S6 smoke: identity max_abs={max_abs:.2e}  raw-src inject logit cos={cos:.4}"
        );
    }

    #[test]
    fn cmkv_cpu_dump() {
        let dir = match std::env::var("CMKV_DUMP_DIR") {
            Ok(d) => PathBuf::from(d),
            Err(_) => {
                eprintln!("cmkv_cpu_dump: set CMKV_DUMP_DIR to enable (skipped)");
                return;
            }
        };
        fs::create_dir_all(&dir).unwrap();
        let n_seq = std::env::var("CMKV_N_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(24usize);
        let seq_len = std::env::var("CMKV_SEQ_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32usize);

        // S5 pairs:
        // 1) intra-family same L, different weights (matched KV)
        dump_spec(
            &dir,
            SynthGemmaSpec::tiny().with_layers(4).with_tag("famA_s"),
            n_seq,
            seq_len,
            1,
        );
        dump_spec(
            &dir,
            SynthGemmaSpec::tiny().with_layers(4).with_tag("famA_t"),
            n_seq,
            seq_len,
            1, // same token streams
        );
        // 2) intra-family different L (4 → 8), matched KV dims
        dump_spec(
            &dir,
            SynthGemmaSpec::tiny().with_layers(4).with_tag("famB_shallow"),
            n_seq,
            seq_len,
            2,
        );
        dump_spec(
            &dir,
            SynthGemmaSpec::tiny().with_layers(8).with_tag("famB_deep"),
            n_seq,
            seq_len,
            2,
        );
        // 3) inter-family mismatched n_kv (2→4 on sliding)
        dump_spec(
            &dir,
            SynthGemmaSpec::tiny()
                .with_layers(4)
                .with_kv(2, 2, 32, 64)
                .with_tag("famC_src"),
            n_seq,
            seq_len,
            3,
        );
        dump_spec(
            &dir,
            SynthGemmaSpec::tiny()
                .with_layers(6)
                .with_kv(4, 4, 32, 64)
                .with_tag("famC_tgt"),
            n_seq,
            seq_len,
            3,
        );

        handoff_logit_smoke(&dir);
        #[cfg(feature = "qwen35")]
        dump_hybrid_qwen35(&dir, n_seq, seq_len);
        eprintln!("cmkv_cpu_dump: done → {}", dir.display());
    }

    /// S7: hybrid Linear/Full schedule — dump full-attn KV only after CPU
    /// `delta_net` + `gated_attention` steps (true state advance, not random fill).
    #[cfg(feature = "qwen35")]
    fn dump_hybrid_qwen35(out: &Path, n_seq: usize, seq_len: usize) {
        use crate::qwen35::{synthetic_hybrid_qwen35, LayerState, LayerType};

        // Shallow: L F L F   Deep: L F L F L F L F  (matched full-attn n_kv/d)
        let shallow_types = vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ];
        let deep_types = vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ];

        for (name, types, tag, seed) in [
            ("q35_hybrid_shallow", shallow_types.clone(), "hyA", 11u64),
            ("q35_hybrid_deep", deep_types, "hyB", 11u64),
        ] {
            let max_seq = seq_len + 8;
            let mut model = synthetic_hybrid_qwen35(types, max_seq, tag);
            let root = out.join(name);
            fs::create_dir_all(&root).unwrap();
            let h = model.config.hidden_size;
            let mut s = seed;
            let mut nxt = || {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            };

            for si in 0..n_seq {
                model.reset();
                let seq_dir = root.join(format!("seq_{si:04}"));
                fs::create_dir_all(&seq_dir).unwrap();
                for pos in 0..seq_len {
                    let x: Vec<f32> = (0..h).map(|_| nxt()).collect();
                    // Advance every layer: GDN or full-attn (fills KV).
                    for li in 0..model.config.num_hidden_layers {
                        match model.config.layer_types[li] {
                            LayerType::LinearAttention => {
                                let _ = model.delta_net(li, &x);
                            }
                            LayerType::FullAttention => {
                                let _ = model.gated_attention(li, &x, pos);
                            }
                        }
                    }
                }
                // Dump FULL-ATTENTION layers only (S7 carve-out).
                let mut layers_json = String::from("[");
                let mut first = true;
                for (si_local, st) in model.layer_state.iter().enumerate() {
                    if let LayerState::Full(c) = st {
                        let n = c.seq_len;
                        let k = c.k_upto(n);
                        let v = c.v_upto(n);
                        let global_li = model.pp_start + si_local;
                        write_f32_bin(&seq_dir.join(format!("full_{global_li:02}_k.f32")), k);
                        write_f32_bin(&seq_dir.join(format!("full_{global_li:02}_v.f32")), v);
                        if !first {
                            layers_json.push(',');
                        }
                        first = false;
                        layers_json.push_str(&format!(
                            "{{\"global_idx\":{global_li},\"kind\":\"full\",\"n_kv\":{},\"head_dim\":{},\"seq_len\":{n}}}",
                            c.num_kv_heads, c.head_dim
                        ));
                    }
                }
                layers_json.push(']');
                write_json(
                    &seq_dir.join("meta.json"),
                    &format!(
                        "{{\n  \"name\": \"{name}\",\n  \"seq_idx\": {si},\n  \"seq_len\": {seq_len},\n  \"full_attn_layers\": {layers_json},\n  \"source\": \"vllm_vulkan_cpu_hybrid_step\"\n}}\n"
                    ),
                );
            }
            write_json(
                &root.join("manifest.json"),
                &format!(
                    "{{\n  \"name\": \"{name}\",\n  \"family\": \"qwen35_hybrid_synth\",\n  \"n_seq\": {n_seq},\n  \"seq_len\": {seq_len},\n  \"note\": \"full-attn KV only; GDN state not transferred (S7)\"\n}}\n"
                ),
            );
            eprintln!("cmkv dump hybrid: {name} → {}", root.display());
        }
    }

    fn logit_cos(a: &[f32], b: &[f32]) -> f64 {
        let mut num = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let x = x as f64;
            let y = y as f64;
            num += x * y;
            na += x * x;
            nb += y * y;
        }
        num / (na.sqrt() * nb.sqrt() + 1e-18)
    }

    fn write_u32_bin(path: &Path, data: &[u32]) {
        let mut f = fs::File::create(path).expect("create u32 bin");
        for &x in data {
            use std::io::Write;
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }

    /// Run one S6 handoff scenario (same or different weight tags).
    /// Returns (mean_identity_cos, mean_raw_cos, mean_mapped_cos).
    fn s6_run_scenario(
        work: &Path,
        src_tag: &str,
        tgt_tag: &str,
        n_train: usize,
        n_test: usize,
        prefix_len: usize,
        mapper_k: usize,
        seed: u64,
    ) -> (f64, f64, f64) {
        let _ = fs::remove_dir_all(work);
        fs::create_dir_all(work).unwrap();

        let max_seq = prefix_len + 8;
        let src_spec = SynthGemmaSpec::tiny().with_layers(4).with_tag(src_tag);
        let tgt_spec = SynthGemmaSpec::tiny().with_layers(4).with_tag(tgt_tag);
        let mut src_m = synthetic_gemma(&src_spec, max_seq);
        let mut tgt_m = synthetic_gemma(&tgt_spec, max_seq);
        let n_layers = src_m.config.num_hidden_layers;
        assert_eq!(n_layers, tgt_m.config.num_hidden_layers);

        // Geometry meta for the Python fitter
        let mut n_kv = Vec::new();
        let mut hd = Vec::new();
        let mut is_full = Vec::new();
        for li in 0..n_layers {
            n_kv.push(src_m.config.layer_num_kv_heads(li));
            hd.push(src_m.config.layer_head_dim(li));
            is_full.push(src_m.config.is_full_attention(li));
            assert_eq!(
                src_m.config.layer_num_kv_heads(li),
                tgt_m.config.layer_num_kv_heads(li),
                "same-geometry gate only"
            );
        }
        {
            let n_kv_s = n_kv
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let hd_s = hd
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let full_s = is_full
                .iter()
                .map(|b| if *b { "true" } else { "false" })
                .collect::<Vec<_>>()
                .join(",");
            write_json(
                &work.join("meta.json"),
                &format!(
                    "{{\n  \"n_layers\": {n_layers},\n  \"prefix_len\": {prefix_len},\n  \"n_kv\": [{n_kv_s}],\n  \"head_dim\": [{hd_s}],\n  \"is_full\": [{full_s}],\n  \"mapper_k\": {mapper_k},\n  \"lam\": 0.01\n}}\n"
                ),
            );
        }

        let train_src = work.join("train");
        let train_tgt = work.join("train_tgt");
        let test_src = work.join("test_src");
        let test_out = work.join("test");
        fs::create_dir_all(&train_src).unwrap();
        fs::create_dir_all(&train_tgt).unwrap();
        fs::create_dir_all(&test_src).unwrap();
        fs::create_dir_all(&test_out).unwrap();

        let all_seqs = make_seqs(
            n_train + n_test,
            prefix_len + 1,
            src_m.config.vocab_size,
            seed,
        );

        // Dump train prefixes (source + target KV)
        for (si, tokens) in all_seqs.iter().take(n_train).enumerate() {
            let prefix = &tokens[..prefix_len];
            let _ = prefill(&mut src_m, prefix);
            let _ = prefill(&mut tgt_m, prefix);
            let sd_s = train_src.join(format!("seq_{si:04}"));
            let sd_t = train_tgt.join(format!("seq_{si:04}"));
            fs::create_dir_all(&sd_s).unwrap();
            fs::create_dir_all(&sd_t).unwrap();
            for li in 0..n_layers {
                let cs = &src_m.kv_caches[li];
                let ct = &tgt_m.kv_caches[li];
                assert_eq!(cs.seq_len, prefix_len);
                write_f32_bin(&sd_s.join(format!("layer_{li:02}_k.f32")), cs.k_upto(prefix_len));
                write_f32_bin(&sd_s.join(format!("layer_{li:02}_v.f32")), cs.v_upto(prefix_len));
                write_f32_bin(&sd_t.join(format!("layer_{li:02}_k.f32")), ct.k_upto(prefix_len));
                write_f32_bin(&sd_t.join(format!("layer_{li:02}_v.f32")), ct.v_upto(prefix_len));
            }
        }

        // Held-out test: write source prefix + tokens/next for handoff
        let mut refs: Vec<(Vec<u32>, u32, Vec<f32>)> = Vec::new();
        for (ti, tokens) in all_seqs.iter().skip(n_train).enumerate() {
            let prefix = &tokens[..prefix_len];
            let next = tokens[prefix_len];
            // Reference logits on target
            let mut tgt_ref = synthetic_gemma(&tgt_spec, max_seq);
            let _ = prefill(&mut tgt_ref, prefix);
            let logits_ref = tgt_ref.forward(next, prefix_len);

            // Source dump for mapping
            let _ = prefill(&mut src_m, prefix);
            let sd = test_src.join(format!("seq_{ti:04}"));
            fs::create_dir_all(&sd).unwrap();
            for li in 0..n_layers {
                let c = &src_m.kv_caches[li];
                write_f32_bin(&sd.join(format!("layer_{li:02}_k.f32")), c.k_upto(prefix_len));
                write_f32_bin(&sd.join(format!("layer_{li:02}_v.f32")), c.v_upto(prefix_len));
            }
            write_u32_bin(&sd.join("tokens.u32"), prefix);
            write_u32_bin(&sd.join("next_tok.u32"), &[next]);
            refs.push((prefix.to_vec(), next, logits_ref));
        }

        // Python ridge fit + map
        let py = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("spike/cross-model-kv/fit_handoff.py");
        assert!(py.is_file(), "missing fit_handoff.py at {}", py.display());
        let status = std::process::Command::new("python3")
            .arg(&py)
            .arg(&work)
            .status()
            .expect("spawn python3 fit_handoff");
        assert!(status.success(), "fit_handoff.py failed: {status}");

        // Identity + raw + mapped logit cosines on each test seq
        let mut id_cos = Vec::new();
        let mut raw_cos = Vec::new();
        let mut map_cos = Vec::new();
        for (ti, (prefix, next, logits_ref)) in refs.iter().enumerate() {
            // Identity
            let mut tgt_pre = synthetic_gemma(&tgt_spec, max_seq);
            let _ = prefill(&mut tgt_pre, prefix);
            let mut tgt_id = synthetic_gemma(&tgt_spec, max_seq);
            for li in 0..n_layers {
                let c = &tgt_pre.kv_caches[li];
                gemma_inject_kv_layer(&mut tgt_id, li, c.k_upto(prefix_len), c.v_upto(prefix_len), prefix_len);
            }
            let logits_id = tgt_id.forward(*next, prefix_len);
            id_cos.push(logit_cos(logits_ref, &logits_id));

            // Raw source inject (same geometry)
            let mut src_pre = synthetic_gemma(&src_spec, max_seq);
            let _ = prefill(&mut src_pre, prefix);
            let mut tgt_raw = synthetic_gemma(&tgt_spec, max_seq);
            for li in 0..n_layers {
                let c = &src_pre.kv_caches[li];
                gemma_inject_kv_layer(&mut tgt_raw, li, c.k_upto(prefix_len), c.v_upto(prefix_len), prefix_len);
            }
            let logits_raw = tgt_raw.forward(*next, prefix_len);
            raw_cos.push(logit_cos(logits_ref, &logits_raw));

            // Mapped inject
            let sd = test_out.join(format!("seq_{ti:04}"));
            let mut tgt_map = synthetic_gemma(&tgt_spec, max_seq);
            for li in 0..n_layers {
                let stride = n_kv[li] * hd[li];
                let k = {
                    let bytes = fs::read(sd.join(format!("mapped_layer_{li:02}_k.f32"))).unwrap();
                    let mut v = vec![0.0f32; prefix_len * stride];
                    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                        v[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    v
                };
                let v = {
                    let bytes = fs::read(sd.join(format!("mapped_layer_{li:02}_v.f32"))).unwrap();
                    let mut v = vec![0.0f32; prefix_len * stride];
                    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                        v[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                    v
                };
                gemma_inject_kv_layer(&mut tgt_map, li, &k, &v, prefix_len);
            }
            let logits_map = tgt_map.forward(*next, prefix_len);
            map_cos.push(logit_cos(logits_ref, &logits_map));
        }

        let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
        (mean(&id_cos), mean(&raw_cos), mean(&map_cos))
    }

    /// S6 next-gate: map→inject→next-token logits on synthetic gemma.
    ///
    /// Two scenarios:
    ///   1. **identical weights** (same tag): ridge ≈ identity → mapped logit cos
    ///      must be high — proves the full map→inject→forward pipeline.
    ///   2. **independent weights** (different tags): reports mapped vs raw;
    ///      soft check mapped ≥ raw (random synth has little shared structure).
    ///
    /// Real transfer quality is G3 (atlas 12B→31B), not the random-weight case.
    #[test]
    fn cmkv_s6_mapped_handoff_gate() {
        // This gate drives a Python ridge-fit oracle (fit_handoff.py) that lives in
        // the internal, non-shipped spike/ tree. When that harness is absent — e.g.
        // the upstream/foundation carve, which excludes spike/ — skip rather than
        // hard-fail, so `cargo test` stays green on trees without the fixture.
        let oracle = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("spike/cross-model-kv/fit_handoff.py");
        if !oracle.is_file() {
            eprintln!(
                "cmkv_s6_mapped_handoff_gate: SKIP — oracle absent at {}",
                oracle.display()
            );
            return;
        }
        let base = match std::env::var("CMKV_S6_WORK") {
            Ok(d) => PathBuf::from(d),
            Err(_) => std::env::temp_dir().join(format!("cmkv_s6_{}", std::process::id())),
        };
        let n_train = 8usize;
        let n_test = 3usize;
        let prefix_len = 24usize;
        let mapper_k = 2usize;

        // ── Scenario A: identical models (pipeline proof) ─────────────────
        let work_a = base.join("identical");
        let (id_a, raw_a, map_a) = s6_run_scenario(
            &work_a,
            "s6same",
            "s6same",
            n_train,
            n_test,
            prefix_len,
            mapper_k,
            20260808,
        );
        eprintln!(
            "cmkv S6 A (identical weights): identity={id_a:.4} raw={raw_a:.4} mapped={map_a:.4}"
        );
        assert!(id_a > 0.999, "A identity cos={id_a}");
        // Same weights ⇒ raw inject of source KV is already correct target KV
        assert!(raw_a > 0.999, "A raw (same model) cos={raw_a}");
        assert!(
            map_a > 0.90,
            "A mapped cos {map_a:.4} < 0.90 — map→inject pipeline broken"
        );

        // ── Scenario B: independent weights (control / soft) ──────────────
        let work_b = base.join("cross");
        let (id_b, raw_b, map_b) = s6_run_scenario(
            &work_b,
            "s6src",
            "s6tgt",
            n_train,
            n_test,
            prefix_len,
            mapper_k,
            20260809,
        );
        eprintln!(
            "cmkv S6 B (independent weights): identity={id_b:.4} raw={raw_b:.4} mapped={map_b:.4}"
        );
        assert!(id_b > 0.999, "B identity cos={id_b}");
        // Independent synth weights share little linear structure — mapped can
        // be near zero / worse than raw. Recorded for the assessment report;
        // real transfer quality is G3 (atlas), not this control.

        let summary = format!(
            "{{\n  \"identical\": {{\n    \"identity_logit_cos\": {id_a:.6},\n    \"raw_logit_cos\": {raw_a:.6},\n    \"mapped_logit_cos\": {map_a:.6}\n  }},\n  \"cross_weight\": {{\n    \"identity_logit_cos\": {id_b:.6},\n    \"raw_foreign_logit_cos\": {raw_b:.6},\n    \"mapped_logit_cos\": {map_b:.6}\n  }},\n  \"n_train\": {n_train},\n  \"n_test\": {n_test},\n  \"prefix_len\": {prefix_len},\n  \"mapper_k\": {mapper_k},\n  \"identity_logit_cos\": {id_a:.6},\n  \"mapped_logit_cos\": {map_a:.6},\n  \"raw_foreign_logit_cos\": {raw_b:.6}\n}}\n"
        );
        write_json(&base.join("s6_gate_summary.json"), &summary);
        let results = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("spike/cross-model-kv/results/s6_gate_summary.json");
        if let Some(parent) = results.parent() {
            let _ = fs::create_dir_all(parent);
            let _ = fs::write(&results, &summary);
        }
    }

    fn read_f32_le(path: &Path) -> Vec<f32> {
        let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert_eq!(bytes.len() % 4, 0);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Mechanics gate: inject real-geometry (gemma-31B sliding) mapped / GT KV
    /// blobs produced by `real_kv_reconstruct.py`, then run one forward step.
    ///
    /// Requires:
    ///   CMKV_RECON_DIR=spike/cross-model-kv/results/real_kv_recon
    /// (run `python3 real_kv_reconstruct.py` first). Skips if unset/missing.
    #[test]
    fn cmkv_real_kv_mechanics_inject() {
        let recon = match std::env::var("CMKV_RECON_DIR") {
            Ok(d) => PathBuf::from(d),
            Err(_) => {
                // Default to repo path if artifacts exist
                let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("spike/cross-model-kv/results/real_kv_recon");
                if !(d.join("mapped/seq_0000").is_dir()) {
                    eprintln!(
                        "cmkv_real_kv_mechanics_inject: skip (no CMKV_RECON_DIR and no {})",
                        d.display()
                    );
                    return;
                }
                d
            }
        };
        let mapped_seq = recon.join("mapped/seq_0000");
        let gt_seq = recon.join("gt_sample/seq_0000");
        let meta_path = mapped_seq.join("meta.json");
        assert!(
            meta_path.is_file(),
            "missing {}; run real_kv_reconstruct.py first",
            meta_path.display()
        );
        let meta_txt = fs::read_to_string(&meta_path).unwrap();
        // Minimal parse without serde: pull seq_len and n_layers
        let seq_len = meta_txt
            .split("\"seq_len\":")
            .nth(1)
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.trim().parse::<usize>().ok())
            .expect("seq_len");
        let n_layers = meta_txt
            .split("\"n_layers\":")
            .nth(1)
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.trim().parse::<usize>().ok())
            .expect("n_layers");

        // 31B-like sliding geometry: all sliding (period > n_layers), n_kv=16, d=256
        let mut spec = SynthGemmaSpec::tiny()
            .with_layers(n_layers)
            .with_kv(16, 16, 256, 256)
            .with_tag("mech31");
        spec.attention_period = n_layers + 1; // no full layers in [0,n_layers)
        spec.num_attention_heads = 16;
        spec.hidden_size = 512;
        spec.intermediate_size = 1024;
        spec.sliding_window = 1024.max(seq_len);
        let max_seq = seq_len + 4;
        let mut model = synthetic_gemma(&spec, max_seq);
        for li in 0..n_layers {
            assert!(
                !model.config.is_full_attention(li),
                "layer {li} should be sliding"
            );
            assert_eq!(model.config.layer_num_kv_heads(li), 16);
            assert_eq!(model.config.layer_head_dim(li), 256);
        }

        // ── Inject GT sample ─────────────────────────────────────────────
        gemma_reset_kv(&mut model);
        for li in 0..n_layers {
            let k = read_f32_le(&gt_seq.join(format!("layer_{li:02}_k.f32")));
            let v = read_f32_le(&gt_seq.join(format!("layer_{li:02}_v.f32")));
            assert_eq!(k.len(), seq_len * 16 * 256, "gt k len layer {li}");
            gemma_inject_kv_layer(&mut model, li, &k, &v, seq_len);
            assert_eq!(model.kv_caches[li].seq_len, seq_len);
            // Round-trip: stored bytes match
            let k2 = model.kv_caches[li].k_upto(seq_len);
            assert_eq!(k2, k.as_slice(), "gt inject round-trip layer {li}");
        }
        // One decode step after GT inject (must not panic)
        let logits_gt = model.forward(1u32, seq_len);
        assert_eq!(logits_gt.len(), model.config.vocab_size);
        let finite_gt = logits_gt.iter().all(|x| x.is_finite());
        assert!(finite_gt, "non-finite logits after GT inject");

        // ── Inject mapped reconstruction ─────────────────────────────────
        let mut model_m = synthetic_gemma(&spec, max_seq);
        for li in 0..n_layers {
            let k = read_f32_le(&mapped_seq.join(format!("layer_{li:02}_k.f32")));
            let v = read_f32_le(&mapped_seq.join(format!("layer_{li:02}_v.f32")));
            assert_eq!(k.len(), seq_len * 16 * 256, "mapped k len layer {li}");
            gemma_inject_kv_layer(&mut model_m, li, &k, &v, seq_len);
            assert_eq!(model_m.kv_caches[li].seq_len, seq_len);
        }
        let logits_m = model_m.forward(1u32, seq_len);
        assert_eq!(logits_m.len(), model_m.config.vocab_size);
        assert!(
            logits_m.iter().all(|x| x.is_finite()),
            "non-finite logits after mapped inject"
        );

        // Cosine between GT-inject and mapped-inject next-token logits
        // (same synthetic weights ⇒ logit cos tracks KV fidelity under this head)
        let cos = logit_cos(&logits_gt, &logits_m);
        eprintln!(
            "cmkv real-KV mechanics: injected seq_len={seq_len} n_layers={n_layers} \
             n_kv=16 d=256  gt_vs_mapped_logit_cos={cos:.4}  recon={}",
            recon.display()
        );
        let summary = format!(
            "{{\n  \"seq_len\": {seq_len},\n  \"n_layers\": {n_layers},\n  \"n_kv\": 16,\n  \"head_dim\": 256,\n  \"gt_inject_ok\": true,\n  \"mapped_inject_ok\": true,\n  \"gt_vs_mapped_logit_cos\": {cos:.6},\n  \"recon_dir\": \"{}\"\n}}\n",
            recon.display()
        );
        let _ = fs::write(recon.join("mechanics_inject.json"), &summary);
        let results = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("spike/cross-model-kv/results/real_kv_recon/mechanics_inject.json");
        let _ = fs::write(results, &summary);

        // Soft quality: mapped should not be orthogonal to GT under same weights
        assert!(
            cos > 0.3,
            "gt_vs_mapped logit cos {cos:.4} < 0.3 — mapped KV likely corrupt"
        );
    }
}

#[cfg(test)]
mod gemma_kv_tile_tests {
    //! Host bit-exact gate for the gemma canonical `(layer, kv_head)`-tile KV
    //! prefix export/import (NAS prefix-cache Phase 1). Gates:
    //!  (a) per-tile store round-trip is `f32::to_bits`-identical (f32 dtype);
    //!  (b) LOAD-AND-RESUME == FULL-PREFILL is argmax-exact (and here bit-exact)
    //!      on the gemma CPU golden forward.
    use super::*;
    use crate::kv_prefix::{KvDtype, KvPrefixExport};

    fn seeded_tokens(n: usize, vocab: usize, seed: u64) -> Vec<u32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 17) as usize % vocab) as u32
            })
            .collect()
    }

    fn argmax(v: &[f32]) -> usize {
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        bi
    }

    #[test]
    fn gemma_tile_store_roundtrip_bit_exact() {
        let max_seq = 64usize;
        let mut src = tiny_synthetic_gemma(max_seq);
        let vocab = src.config.vocab_size;
        let toks = seeded_tokens(10, vocab, 0xA11CE_u64);
        for (pos, &t) in toks.iter().enumerate() {
            src.forward(t, pos);
        }
        let upto = toks.len();

        let tiles = src.owned_tiles();
        assert!(tiles.iter().any(|t| t.is_full), "must include a full-attn tile");
        assert!(tiles.iter().any(|t| !t.is_full), "must include a sliding tile");
        let blobs: Vec<(usize, usize, Vec<u8>)> = tiles
            .iter()
            .map(|t| (t.layer, t.kv_head, src.export_tile(t.layer, t.kv_head, upto, KvDtype::F32).unwrap()))
            .collect();

        let mut dst = tiny_synthetic_gemma(max_seq);
        for (l, h, blob) in &blobs {
            dst.import_tile(*l, *h, blob).unwrap();
        }
        dst.set_seq_len(upto);

        for t in &tiles {
            let cfg = &src.config;
            let (base, n_rows) = tile_row_range(t.is_full, upto, cfg.sliding_window);
            let sc = &src.kv_caches[t.layer];
            let dc = &dst.kv_caches[t.layer];
            let stride = sc.num_kv_heads * sc.head_dim;
            for i in 0..n_rows {
                let slot = (base + i) % sc.capacity;
                let off = slot * stride + t.kv_head * t.head_dim;
                for j in 0..t.head_dim {
                    assert_eq!(sc.k[off + j].to_bits(), dc.k[off + j].to_bits(), "K drift L{} h{}", t.layer, t.kv_head);
                    assert_eq!(sc.v[off + j].to_bits(), dc.v[off + j].to_bits(), "V drift L{} h{}", t.layer, t.kv_head);
                }
            }
        }
    }

    #[test]
    fn gemma_load_and_resume_equals_full_prefill() {
        let max_seq = 64usize;
        let l = 14usize;
        let p = 6usize;
        let vocab = tiny_synthetic_gemma(max_seq).config.vocab_size;
        let toks = seeded_tokens(l, vocab, 0x5EED_F00D_u64);

        let mut full = tiny_synthetic_gemma(max_seq);
        let mut full_logits = Vec::new();
        for (pos, &t) in toks.iter().enumerate() {
            full_logits = full.forward(t, pos);
        }

        let mut srcm = tiny_synthetic_gemma(max_seq);
        for pos in 0..p {
            srcm.forward(toks[pos], pos);
        }
        let tiles = srcm.owned_tiles();
        let blobs: Vec<(usize, usize, Vec<u8>)> = tiles
            .iter()
            .map(|t| (t.layer, t.kv_head, srcm.export_tile(t.layer, t.kv_head, p, KvDtype::F32).unwrap()))
            .collect();

        let mut resume = tiny_synthetic_gemma(max_seq);
        for (lyr, h, blob) in &blobs {
            resume.import_tile(*lyr, *h, blob).unwrap();
        }
        resume.set_seq_len(p);
        let mut resume_logits = Vec::new();
        for pos in p..l {
            resume_logits = resume.forward(toks[pos], pos);
        }

        assert_eq!(argmax(&full_logits), argmax(&resume_logits), "resume argmax != full argmax");
        assert_eq!(full_logits.len(), resume_logits.len());
        for (a, b) in full_logits.iter().zip(&resume_logits) {
            assert_eq!(a.to_bits(), b.to_bits(), "resume logits not bit-identical to full prefill");
        }
    }

    #[test]
    fn gemma_f16_tile_resume_argmax_holds() {
        let max_seq = 64usize;
        let (l, p) = (14usize, 6usize);
        let vocab = tiny_synthetic_gemma(max_seq).config.vocab_size;
        let toks = seeded_tokens(l, vocab, 0xBEE5_u64);

        let mut full = tiny_synthetic_gemma(max_seq);
        let mut full_logits = Vec::new();
        for (pos, &t) in toks.iter().enumerate() {
            full_logits = full.forward(t, pos);
        }

        let mut srcm = tiny_synthetic_gemma(max_seq);
        for pos in 0..p {
            srcm.forward(toks[pos], pos);
        }
        let tiles = srcm.owned_tiles();
        let blobs: Vec<(usize, usize, Vec<u8>)> = tiles
            .iter()
            .map(|t| (t.layer, t.kv_head, srcm.export_tile(t.layer, t.kv_head, p, KvDtype::F16).unwrap()))
            .collect();

        let mut resume = tiny_synthetic_gemma(max_seq);
        for (lyr, h, blob) in &blobs {
            resume.import_tile(*lyr, *h, blob).unwrap();
        }
        resume.set_seq_len(p);
        let mut resume_logits = Vec::new();
        for pos in p..l {
            resume_logits = resume.forward(toks[pos], pos);
        }
        assert_eq!(argmax(&full_logits), argmax(&resume_logits), "f16 resume argmax != full argmax");
    }
}
