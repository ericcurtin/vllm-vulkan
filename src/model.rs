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
use std::path::Path;
use std::sync::Arc;

use half::bf16;

// Note: compute engine is used for GPU acceleration in the Vulkan path;
// the CPU reference implementation doesn't need it.

// ─── Model configuration ─────────────────────────────────────────────────────

/// Gemma4-E2B architecture constants (from config.json).
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub hidden_size: usize,           // 1536
    pub num_hidden_layers: usize,     // 35
    pub num_attention_heads: usize,   // 8
    pub num_key_value_heads: usize,   // 1
    pub head_dim: usize,              // 256  (sliding attention)
    pub global_head_dim: usize,       // 512  (full attention)
    pub intermediate_size: usize,     // 6144
    pub num_kv_shared_layers: usize,  // 20
    pub vocab_size: usize,            // 262144
    pub rms_norm_eps: f32,            // 1e-6
    pub sliding_window: usize,        // 512
    pub hidden_size_per_layer_input: usize, // 256 (PLE)
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
            hidden_size: h,
            num_hidden_layers: 35,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            global_head_dim: 512,
            intermediate_size: 6144,
            num_kv_shared_layers: 20,
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

    pub fn first_kv_shared_layer(&self) -> usize {
        self.num_hidden_layers - self.num_kv_shared_layers  // 35 - 20 = 15
    }

    /// Is layer `idx` a full-attention layer?
    pub fn is_full_attention(&self, idx: usize) -> bool {
        // Layer types from config: full_attention at indices 4,9,14,19,24,29,34
        idx % 5 == 4
    }

    /// head_dim for layer `idx`
    pub fn layer_head_dim(&self, idx: usize) -> usize {
        if self.is_full_attention(idx) { self.global_head_dim } else { self.head_dim }
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

// ─── KV cache ─────────────────────────────────────────────────────────────────

/// Per-layer KV cache.  Stored as host-coherent f32 tensors so both CPU
/// attention and (future) GPU attention can access them directly.
pub struct KvCache {
    /// [max_seq_len, num_kv_heads, head_dim] f32, for K
    pub k: Vec<f32>,
    /// [max_seq_len, num_kv_heads, head_dim] f32, for V
    pub v: Vec<f32>,
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        KvCache {
            k: vec![0.0; max_seq_len * num_kv_heads * head_dim],
            v: vec![0.0; max_seq_len * num_kv_heads * head_dim],
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
        }
    }

    /// Append one token's K and V to the cache.
    pub fn append(&mut self, k_token: &[f32], v_token: &[f32]) {
        assert!(self.seq_len < self.max_seq_len, "KV cache overflow");
        let stride = self.num_kv_heads * self.head_dim;
        let pos = self.seq_len * stride;
        self.k[pos..pos + stride].copy_from_slice(k_token);
        self.v[pos..pos + stride].copy_from_slice(v_token);
        self.seq_len += 1;
    }

    pub fn k_up_to_now(&self) -> &[f32] {
        &self.k[..self.seq_len * self.num_kv_heads * self.head_dim]
    }

    pub fn v_up_to_now(&self) -> &[f32] {
        &self.v[..self.seq_len * self.num_kv_heads * self.head_dim]
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

/// Element-wise GELU (tanh approximation).
pub fn cpu_gelu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| {
        let c = 0.044715f32;
        let sqrt_2_over_pi = 0.7978845608f32;
        let inner = sqrt_2_over_pi * (v + c * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    }).collect()
}

/// RoPE: apply rotary positional embedding to q and k.
/// pos: token position, x: [num_heads, head_dim], rotary_dim = dims to rotate
pub fn cpu_rope(
    q: &mut [f32], k: &mut [f32],
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
) {
    let rotate_head = |x: &mut [f32], pos: usize, head_dim: usize, rotary_dim: usize, theta: f32| {
        let half = rotary_dim / 2;
        for i in 0..half {
            let freq = 1.0 / theta.powf(i as f32 * 2.0 / rotary_dim as f32);
            let angle = pos as f32 * freq;
            let (s, c) = angle.sin_cos();
            let x0 = x[i];
            let x1 = x[i + half];
            x[i]        = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
        // dims [rotary_dim..head_dim] are unchanged
    };

    for h in 0..num_q_heads {
        let slice = &mut q[h * head_dim..(h + 1) * head_dim];
        rotate_head(slice, pos, head_dim, rotary_dim, theta);
    }
    for h in 0..num_kv_heads {
        let slice = &mut k[h * head_dim..(h + 1) * head_dim];
        rotate_head(slice, pos, head_dim, rotary_dim, theta);
    }
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
    let gqa_ratio = num_q_heads / num_kv_heads;
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
        let kvh = qh / gqa_ratio;
        let q_row = &q[qh * head_dim..(qh + 1) * head_dim];

        for (si, kv_pos) in (kv_start..seq_len).enumerate() {
            let k_row = &k[(kv_pos * num_kv_heads + kvh) * head_dim
                          ..(kv_pos * num_kv_heads + kvh + 1) * head_dim];
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
    /// Forward pass for one token at position `pos`.
    /// Returns logits [vocab_size].
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        // ── Embedding ──────────────────────────────────────────────────────
        let embed_w = self.weights.f32_slice("model.embed_tokens.weight");
        let mut hidden: Vec<f32> = embed_w[token_id as usize * h..
                                            (token_id as usize + 1) * h]
            .iter().map(|&v| v * cfg.embed_scale).collect();

        // ── PLE global preprocessing ────────────────────────────────────────
        let ple_dim = cfg.hidden_size_per_layer_input;  // 256
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
        let ple_inputs: Vec<f32> = ple_proj_normed.iter()
            .zip(ple_embeds_flat.iter())
            .map(|(&p, &e)| (p + e) * cfg.per_layer_input_scale)
            .collect();
        // ple_inputs: [total_ple] = [35 * 256], per-layer slice = ple_inputs[layer*ple_dim..(layer+1)*ple_dim]

        // ── 35 decoder layers ───────────────────────────────────────────────
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_ple = &ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim];
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

    pub fn forward_layer(&mut self, layer_idx: usize, hidden: &[f32], pos: usize, layer_ple: &[f32]) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
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
        let v_w = self.weights.f32_slice(&ln("self_attn.v_proj.weight")).to_vec();
        let mut q = cpu_matmul(&x, &q_w, 1, h, q_dim);
        let k_raw = cpu_matmul(&x, &k_w, 1, h, kv_dim);
        let v_raw = cpu_matmul(&x, &v_w, 1, h, kv_dim);

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
        cpu_rope(&mut q, &mut k_final, pos, num_q_heads, num_kv_heads, head_dim, rotary_dim, theta);
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
        let seq_len = attn_cache.seq_len;
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        let attn_scale = 1.0f32;  // Gemma4 uses scale=1.0, not 1/sqrt(head_dim)
        let attn_out = cpu_sdpa(
            &q, attn_cache.k_up_to_now(), attn_cache.v_up_to_now(),
            num_q_heads, num_kv_heads, head_dim,
            seq_len, attn_scale, window,
        );
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

        // 14. PLE block
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

        // 15. Layer scalar
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
