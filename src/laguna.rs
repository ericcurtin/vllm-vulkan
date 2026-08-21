//! CPU-only forward for **Laguna-S-2.1-NVFP4** (`model_type == "laguna"`,
//! `LagunaForCausalLM`).
//!
//! This is the pure-`cpu_matmul` reference path — the same role
//! `nemotron_cpu_stage_oracle` plays for Nemotron-H: an absolute check the GPU
//! forward can be bisected against with `scripts/diff_f32.py`. Every op mirrors
//! the HuggingFace `modeling_laguna.py` (verified against the real
//! `config.json` + one real expert's NVFP4 bytes), so a stage dump that matches
//! this exonerates the CPU wiring and localizes a bug into the shaders.
//!
//! Arch (48 layers, hidden 3072, 8 KV heads, head_dim 128, vocab 100352):
//!   * full_attention layers (idx 0,4,8,…,44): 48 Q heads, **YaRN** RoPE
//!     (θ=5e5, factor 32, orig_max 8192, β_fast 32/β_slow 1,
//!     attention_factor≈1.3466, partial_rotary 0.5 → rotary_dim 64), no window.
//!   * sliding_attention layers: 72 Q heads, **plain** RoPE (θ=1e4, full
//!     rotary 128), causal + sliding window 512.
//!   * per-head q/k RMSNorm over head_dim BEFORE RoPE.
//!   * per-head **scalar softplus gate** `softplus(g_proj(h))` broadcast across
//!     head_dim, applied to the attention output BEFORE o_proj.
//!   * MoE (all layers except 0): sigmoid router + `e_score_correction_bias`,
//!     top-10, `norm_topk_prob`, routed_scaling 2.5; per-expert gated-SiLU on
//!     NVFP4-dequantized `gate/up/down_proj`; plus an **ungated** shared expert.
//!   * layer 0 (mlp_only_layers=[0]): dense SwiGLU, intermediate 12288.
//!
//! NVFP4 experts are kept PACKED in RAM (u8 nibbles + F8_E4M3 group-16 scales +
//! F32 reciprocal global) and only the top-10 selected experts per token are
//! dequantized on demand, reusing the crate-wide pure fns
//! [`crate::model::dequantize_nvfp4`] / `e4m3_to_f32` / `NVFP4_E2M1_LUT` — NOT
//! the gemma-name-gated dispatcher at model.rs:1697.

use crate::model::{cpu_matmul, cpu_rms_norm, cpu_rms_norm_inplace, cpu_rope, cpu_sdpa,
    dequantize_nvfp4};
use crate::nemotron::{router_forward, RouterDims};
use serde_json::Value;

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
#[inline]
fn softplus(x: f32) -> f32 {
    // numerically stable log(1 + exp(x)) == max(x,0) + log1p(exp(-|x|)),
    // matching torch.nn.functional.softplus.
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Parsed `config.json` for a Laguna checkpoint. Uses `num_experts` /
/// `num_experts_per_tok` (NOT the deepseek `n_routed_experts`/`num_experts`
/// pair). Per-layer head counts and layer types come from the explicit arrays.
#[derive(Debug, Clone)]
pub struct LagunaConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    /// dense (layer-0) MLP intermediate.
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub moe_routed_scaling_factor: f32,
    pub moe_router_logit_softcapping: f32,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    /// Layers using a dense MLP instead of MoE (== `[0]`).
    pub mlp_only_layers: Vec<usize>,
    /// `layer_types[i] == "full_attention"`.
    pub layer_is_full: Vec<bool>,
    pub num_attention_heads_per_layer: Vec<usize>,

    // full_attention YaRN rope
    pub full_rope_theta: f32,
    pub full_rope_factor: f32,
    pub full_orig_max_pos: usize,
    pub full_beta_fast: f32,
    pub full_beta_slow: f32,
    pub full_attention_factor: f32,
    pub full_partial_rotary: f32,
    /// number of rotated dims on full layers = `head_dim * partial` (== 64).
    pub full_rotary_dim: usize,
    /// precomputed YaRN inv_freq, length `full_rotary_dim / 2` (== 32).
    pub yarn_inv_freq: Vec<f32>,

    // sliding_attention plain rope
    pub sliding_rope_theta: f32,
    pub sliding_partial_rotary: f32,
}

fn get_f32(v: &Value, k: &str, default: f32) -> f32 {
    v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(default)
}
fn get_usize(v: &Value, k: &str) -> Result<usize, String> {
    v.get(k)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| format!("config: missing/!u64 field '{k}'"))
}

impl LagunaConfig {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let mt = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if mt != "laguna" {
            return Err(format!("LagunaConfig::from_json: model_type '{mt}' != 'laguna'"));
        }
        let hidden_size = get_usize(v, "hidden_size")?;
        let num_hidden_layers = get_usize(v, "num_hidden_layers")?;
        let num_attention_heads = get_usize(v, "num_attention_heads")?;
        let num_key_value_heads = get_usize(v, "num_key_value_heads")?;
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        let vocab_size = get_usize(v, "vocab_size")?;
        let rms_norm_eps = get_f32(v, "rms_norm_eps", 1e-6);
        let intermediate_size = get_usize(v, "intermediate_size")?;
        let moe_intermediate_size = get_usize(v, "moe_intermediate_size")?;
        let shared_expert_intermediate_size = v
            .get("shared_expert_intermediate_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(moe_intermediate_size);
        let num_experts = get_usize(v, "num_experts")?;
        let num_experts_per_tok = get_usize(v, "num_experts_per_tok")?;
        let norm_topk_prob = v.get("norm_topk_prob").and_then(|x| x.as_bool()).unwrap_or(true);
        let moe_routed_scaling_factor = get_f32(v, "moe_routed_scaling_factor", 1.0);
        let moe_router_logit_softcapping = get_f32(v, "moe_router_logit_softcapping", 0.0);
        let sliding_window = get_usize(v, "sliding_window").unwrap_or(512);
        let max_position_embeddings = get_usize(v, "max_position_embeddings").unwrap_or(8192);

        let mlp_only_layers: Vec<usize> = v
            .get("mlp_only_layers")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|e| e.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_default();

        let layer_types = v
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or("config: missing 'layer_types'")?;
        let layer_is_full: Vec<bool> = layer_types
            .iter()
            .map(|t| t.as_str() == Some("full_attention"))
            .collect();
        if layer_is_full.len() != num_hidden_layers {
            return Err(format!(
                "config: layer_types len {} != num_hidden_layers {num_hidden_layers}",
                layer_is_full.len()
            ));
        }

        let num_attention_heads_per_layer: Vec<usize> = v
            .get("num_attention_heads_per_layer")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|e| e.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_else(|| vec![num_attention_heads; num_hidden_layers]);
        if num_attention_heads_per_layer.len() != num_hidden_layers {
            return Err(format!(
                "config: num_attention_heads_per_layer len {} != num_hidden_layers {num_hidden_layers}",
                num_attention_heads_per_layer.len()
            ));
        }

        // rope_parameters.{full,sliding}_attention.*
        let rope = v.get("rope_parameters").ok_or("config: missing 'rope_parameters'")?;
        let full = rope
            .get("full_attention")
            .ok_or("config: missing rope_parameters.full_attention")?;
        let slide = rope
            .get("sliding_attention")
            .ok_or("config: missing rope_parameters.sliding_attention")?;

        let full_rope_theta = get_f32(full, "rope_theta", 500000.0);
        let full_rope_factor = get_f32(full, "factor", 32.0);
        let full_orig_max_pos = full
            .get("original_max_position_embeddings")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(8192);
        let full_beta_fast = get_f32(full, "beta_fast", 32.0);
        let full_beta_slow = get_f32(full, "beta_slow", 1.0);
        let full_partial_rotary = get_f32(full, "partial_rotary_factor", 1.0);
        // attention_factor: explicit in config, else transformers' default
        // 0.1*ln(factor)+1.0.
        let full_attention_factor = full
            .get("attention_factor")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or_else(|| 0.1 * full_rope_factor.ln() + 1.0);
        let full_rotary_dim = (head_dim as f32 * full_partial_rotary) as usize;

        let yarn_inv_freq = compute_yarn_inv_freq(
            full_rope_theta,
            head_dim,
            full_partial_rotary,
            full_rope_factor,
            full_beta_fast,
            full_beta_slow,
            full_orig_max_pos,
        );

        let sliding_rope_theta = get_f32(slide, "rope_theta", 10000.0);
        let sliding_partial_rotary = get_f32(slide, "partial_rotary_factor", 1.0);

        Ok(LagunaConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            intermediate_size,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob,
            moe_routed_scaling_factor,
            moe_router_logit_softcapping,
            sliding_window,
            max_position_embeddings,
            mlp_only_layers,
            layer_is_full,
            num_attention_heads_per_layer,
            full_rope_theta,
            full_rope_factor,
            full_orig_max_pos,
            full_beta_fast,
            full_beta_slow,
            full_attention_factor,
            full_partial_rotary,
            full_rotary_dim,
            yarn_inv_freq,
            sliding_rope_theta,
            sliding_partial_rotary,
        })
    }

    pub fn router_dims(&self) -> RouterDims {
        RouterDims {
            n_routed_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            routed_scaling_factor: self.moe_routed_scaling_factor,
            // Laguna's router is a PLAIN top-k over all experts (no group
            // routing). A single group with topk_group=1 makes the nemotron
            // group-mask a no-op, so `router_forward` reduces exactly to
            // `torch.topk(sigmoid(logits)+bias)`.
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: self.norm_topk_prob,
        }
    }

    /// Physical KV capacity (number of position-slots to allocate) for layer
    /// `idx` given a logical `max_seq`. Full-attention (YaRN, `i % 4 == 0`)
    /// layers are UNBOUNDED and get `max_seq`; sliding-window layers only ever
    /// attend the last `sliding_window` (512) positions, so they get a
    /// `sliding_window`-sized RING (`ResidentKvPlane` with `ring_capacity > 0`).
    /// This is the Laguna analog of `Gemma4Config::layer_kv_capacity` — the
    /// Phase-0 per-layer KV sizing that collapses the 36 sliding layers' KV
    /// planes from `max_seq` to `window`. Capacity is clamped to `max_seq` so
    /// short contexts never over-allocate. `layer_is_full[idx]` is the same
    /// full/sliding split that `laguna_attn` uses for RoPE + the SDPA window.
    pub fn layer_kv_capacity(&self, idx: usize, max_seq: usize) -> usize {
        if self.layer_is_full[idx] {
            max_seq
        } else {
            self.sliding_window.min(max_seq)
        }
    }
}

// ─── YaRN RoPE (host-side; no YaRN existed in the crate) ─────────────────────

/// Faithful port of transformers `_compute_yarn_parameters` inv_freq (the
/// `attention_scaling`/`attention_factor` multiply is applied later in
/// [`cpu_rope_yarn`]). Returns the per-pair inverse frequencies, length
/// `(head_dim*partial)/2`.
pub fn compute_yarn_inv_freq(
    theta: f32,
    head_dim: usize,
    partial: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    orig_max: usize,
) -> Vec<f32> {
    let dim = (head_dim as f32 * partial) as usize; // 64
    let half = dim / 2; // 32
    let base = theta as f64;
    let factor = factor as f64;

    // find_correction_dim(num_rot) — the rotary index at which `num_rot` full
    // rotations fit in `orig_max` positions.
    let find_dim = |num_rot: f64| -> f64 {
        (dim as f64 * ((orig_max as f64) / (num_rot * 2.0 * std::f64::consts::PI)).ln())
            / (2.0 * base.ln())
    };
    // find_correction_range: floor(low), ceil(high), clamped to [0, dim-1].
    let low = find_dim(beta_fast as f64).floor().max(0.0);
    let mut high = find_dim(beta_slow as f64).ceil().min((dim - 1) as f64);
    // linear_ramp min==max guard.
    if (high - low).abs() < f64::EPSILON {
        high += 0.001;
    }

    let mut inv = Vec::with_capacity(half);
    for j in 0..half {
        // pos_freqs[j] = base^((2j)/dim); extrapolation = 1/pos_freqs,
        // interpolation = 1/(factor*pos_freqs).
        let pf = base.powf((2 * j) as f64 / dim as f64);
        let inv_extrap = 1.0 / pf;
        let inv_interp = 1.0 / (factor * pf);
        let ramp = ((j as f64 - low) / (high - low)).clamp(0.0, 1.0);
        // inv_freq_extrapolation_factor = 1 - ramp; mix.
        let extrap_factor = 1.0 - ramp;
        let v = inv_interp * (1.0 - extrap_factor) + inv_extrap * extrap_factor;
        inv.push(v as f32);
    }
    inv
}

/// YaRN partial RoPE for a single position. `inv_freq` has `rotary_dim/2`
/// entries; `mscale` is the `attention_factor` multiplied into cos/sin
/// (transformers multiplies BOTH). Pairing is NeoX `(i, i+rotary_dim/2)`,
/// identical to [`crate::model::cpu_rope_with_basis`] with
/// `freq_dim == rotary_dim`; dims `[rotary_dim..head_dim)` pass through.
#[allow(clippy::too_many_arguments)]
pub fn cpu_rope_yarn(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: &[f32],
    mscale: f32,
) {
    let pair_off = rotary_dim / 2;
    assert_eq!(inv_freq.len(), pair_off, "cpu_rope_yarn: inv_freq len must be rotary_dim/2");
    let mut sin_cos: Vec<(f32, f32)> = Vec::with_capacity(pair_off);
    for &f in inv_freq {
        let angle = pos as f32 * f;
        let (s, c) = angle.sin_cos();
        sin_cos.push((s * mscale, c * mscale));
    }
    let rotate_head = |x: &mut [f32]| {
        for (i, &(s, c)) in sin_cos.iter().enumerate() {
            let x0 = x[i];
            let x1 = x[i + pair_off];
            x[i] = x0 * c - x1 * s;
            x[i + pair_off] = x0 * s + x1 * c;
        }
    };
    for h in 0..num_q_heads {
        rotate_head(&mut q[h * head_dim..(h + 1) * head_dim]);
    }
    for h in 0..num_kv_heads {
        rotate_head(&mut k[h * head_dim..(h + 1) * head_dim]);
    }
}

// ─── Owned weights (one host copy, dequantized/packed as appropriate) ────────

/// Attention projections + per-head q/k norms for one layer. BF16→f32 on load.
pub struct OwnedAttn {
    /// [num_heads*head_dim, hidden]
    pub q_proj: Vec<f32>,
    /// [num_kv_heads*head_dim, hidden]
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    /// [hidden, num_heads*head_dim]
    pub o_proj: Vec<f32>,
    /// per-head gate: [num_heads, hidden]
    pub g_proj: Vec<f32>,
    /// [head_dim]
    pub q_norm: Vec<f32>,
    /// [head_dim]
    pub k_norm: Vec<f32>,
}

/// Dense (layer-0) SwiGLU MLP. All BF16→f32.
pub struct OwnedDense {
    /// [intermediate, hidden]
    pub gate: Vec<f32>,
    /// [intermediate, hidden]
    pub up: Vec<f32>,
    /// [hidden, intermediate]
    pub down: Vec<f32>,
}

/// NVFP4-packed routed experts for one layer: nibbles + F8_E4M3 group-16 scales
/// + F32 RAW globals (the reciprocal is applied at dequant time). Each `*_global`
/// vector has one entry per expert.
pub struct OwnedExpertsPacked {
    pub gate_packed: Vec<u8>,
    pub gate_scale: Vec<u8>,
    pub gate_global: Vec<f32>,
    pub up_packed: Vec<u8>,
    pub up_scale: Vec<u8>,
    pub up_global: Vec<f32>,
    pub down_packed: Vec<u8>,
    pub down_scale: Vec<u8>,
    pub down_global: Vec<f32>,
    pub num_experts: usize,
    /// gate/up out_features == moe_intermediate_size; down out_features == hidden.
    pub inter: usize,
    pub hidden: usize,
    pub group_size: usize,
}

impl OwnedExpertsPacked {
    /// Dequantize one expert's `gate`/`up`/`down` (proj 0/1/2) to f32 `[out,in]`.
    fn dequant(&self, e: usize, proj: u8) -> Vec<f32> {
        let (packed, scale, globals, out_f, in_f) = match proj {
            0 => (&self.gate_packed, &self.gate_scale, &self.gate_global, self.inter, self.hidden),
            1 => (&self.up_packed, &self.up_scale, &self.up_global, self.inter, self.hidden),
            2 => (&self.down_packed, &self.down_scale, &self.down_global, self.hidden, self.inter),
            _ => unreachable!("proj must be 0/1/2"),
        };
        let packed_bytes = out_f * in_f / 2;
        let scale_bytes = out_f * (in_f / self.group_size);
        let p = &packed[e * packed_bytes..(e + 1) * packed_bytes];
        let s = &scale[e * scale_bytes..(e + 1) * scale_bytes];
        // compressed-tensors weight_global_scale is the RECIPROCAL of the
        // modelopt weight_scale_2 our dequant fns MULTIPLY by (model.rs:1728).
        let global = 1.0 / globals[e];
        dequantize_nvfp4(p, s, global, out_f, in_f, self.group_size)
    }
}

/// Router + NVFP4 experts + ungated shared expert for one MoE layer.
pub struct OwnedMoe {
    /// router gate: [num_experts, hidden]
    pub router: Vec<f32>,
    /// e_score_correction_bias: [num_experts]
    pub bias: Vec<f32>,
    pub experts: OwnedExpertsPacked,
    /// shared expert (LagunaMLP, ungated): [shared_inter, hidden]
    pub shared_gate: Vec<f32>,
    pub shared_up: Vec<f32>,
    /// [hidden, shared_inter]
    pub shared_down: Vec<f32>,
}

pub enum OwnedMlp {
    Dense(OwnedDense),
    Moe(OwnedMoe),
}

pub struct OwnedLayer {
    pub input_ln: Vec<f32>,
    pub post_ln: Vec<f32>,
    pub attn: OwnedAttn,
    pub mlp: OwnedMlp,
}

pub struct LagunaWeights {
    /// [vocab, hidden]
    pub embed: Vec<f32>,
    /// model.norm.weight [hidden]
    pub final_norm: Vec<f32>,
    pub layers: Vec<OwnedLayer>,
    /// lm_head.weight [vocab, hidden] (None → return final hidden instead of logits)
    pub lm_head: Option<Vec<f32>>,
}

/// A loaded Laguna model: parsed config + owned host weights for a resident
/// layer window. This is the `lib.rs` model-load product (the `if mt=="laguna"`
/// dispatch stores it as `VulkanModel.laguna`), the Laguna analog of
/// [`crate::nemotron::NemotronModel`] / [`crate::qwen35::Qwen35Model`].
///
/// Phase 1 is CPU-only (the same pure-`cpu_matmul` reference the
/// `laguna_cpu_stage_oracle` gate validates); the GPU-resident expert path
/// (qwen35_moe kernels), 48-layer load and PP-8..10 sharding are the later
/// node-count-blocked phase. The loader currently materializes layers
/// `[0, num_layers)`; a true `[pp_start, pp_end)` window (for PP) is deferred to
/// that phase, so `pp_start` is 0 here and `pp_end == num_layers` loaded.
pub struct LagunaModel {
    pub config: LagunaConfig,
    pub weights: LagunaWeights,
    /// Resident layer window (global indices). Phase 1: `[0, num_layers_loaded)`.
    pub pp_start: usize,
    pub pp_end: usize,
    pub pp_first: bool,
    pub pp_last: bool,
}

impl LagunaModel {
    /// Load a Laguna checkpoint from `dir` (must hold `config.json` +
    /// `model.safetensors.index.json` + the referenced shards) into a CPU
    /// model. Loads layers `[0, pp_end)`; `keep_lm` (== last stage) also pulls
    /// `lm_head.weight`. This calls the SAME [`load_laguna_weights_cpu`] the
    /// Deliverable-A `laguna_cpu_stage_oracle` gate exercises, so a passing gate
    /// certifies this exact load path.
    pub fn load_cpu(
        dir: &std::path::Path,
        cfg: LagunaConfig,
        pp_start: usize,
        pp_end: usize,
        keep_lm: bool,
    ) -> Result<Self, String> {
        let total = cfg.num_hidden_layers;
        let pp_first = pp_start == 0;
        let pp_last = pp_end >= total;
        let weights = load_laguna_weights_cpu(dir, &cfg, pp_end, keep_lm)?;
        Ok(LagunaModel { config: cfg, weights, pp_start, pp_end, pp_first, pp_last })
    }

    /// Full CPU forward over a token-id sequence (see
    /// [`LagunaWeights::forward`]): last_hidden_state `[seq, hidden]` when no
    /// `lm_head`, else last-position logits `[vocab]`.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        self.weights.forward(tokens, &self.config)
    }
}

// ─── Forward blocks ──────────────────────────────────────────────────────────

/// Gated GQA attention for one layer over a `[seq, hidden]` prefill.
/// `hidden_normed` is the post-input_layernorm input (also the gate source).
pub fn laguna_attn(
    hidden_normed: &[f32],
    seq: usize,
    layer_idx: usize,
    w: &OwnedAttn,
    cfg: &LagunaConfig,
) -> Vec<f32> {
    let hs = cfg.hidden_size;
    let hd = cfg.head_dim;
    let nkv = cfg.num_key_value_heads;
    let nq = cfg.num_attention_heads_per_layer[layer_idx];
    let is_full = cfg.layer_is_full[layer_idx];
    let eps = cfg.rms_norm_eps;

    // Projections over all positions.
    let mut q = cpu_matmul(hidden_normed, &w.q_proj, seq, hs, nq * hd); // [seq, nq*hd]
    let mut k = cpu_matmul(hidden_normed, &w.k_proj, seq, hs, nkv * hd); // [seq, nkv*hd]
    let v = cpu_matmul(hidden_normed, &w.v_proj, seq, hs, nkv * hd); // [seq, nkv*hd]

    // Per-head q/k RMSNorm (over head_dim) then RoPE, per position.
    for p in 0..seq {
        {
            let qs = &mut q[p * nq * hd..(p + 1) * nq * hd];
            for h in 0..nq {
                cpu_rms_norm_inplace(&mut qs[h * hd..(h + 1) * hd], &w.q_norm, eps);
            }
        }
        {
            let ks = &mut k[p * nkv * hd..(p + 1) * nkv * hd];
            for h in 0..nkv {
                cpu_rms_norm_inplace(&mut ks[h * hd..(h + 1) * hd], &w.k_norm, eps);
            }
        }
        let qs = &mut q[p * nq * hd..(p + 1) * nq * hd];
        // SAFETY-free: q and k are distinct Vecs, so both mutable borrows below
        // do not overlap.
        let ks = &mut k[p * nkv * hd..(p + 1) * nkv * hd];
        if is_full {
            cpu_rope_yarn(
                qs, ks, p, nq, nkv, hd, cfg.full_rotary_dim, &cfg.yarn_inv_freq,
                cfg.full_attention_factor,
            );
        } else {
            // plain NeoX rope, full rotary over head_dim, θ from sliding config.
            cpu_rope(qs, ks, p, nq, nkv, hd, hd, cfg.sliding_rope_theta);
        }
    }

    // Causal (+ sliding-window) SDPA, per query position, over the growing KV.
    let scale = 1.0 / (hd as f32).sqrt();
    let window = if is_full { None } else { Some(cfg.sliding_window) };
    let mut attn_out = vec![0.0f32; seq * nq * hd];
    for p in 0..seq {
        let q_p = &q[p * nq * hd..(p + 1) * nq * hd];
        let k_ctx = &k[0..(p + 1) * nkv * hd];
        let v_ctx = &v[0..(p + 1) * nkv * hd];
        let o = cpu_sdpa(q_p, k_ctx, v_ctx, nq, nkv, hd, p + 1, scale, window);
        attn_out[p * nq * hd..(p + 1) * nq * hd].copy_from_slice(&o);
    }

    // Per-head scalar softplus gate from the attention input, broadcast across
    // head_dim, applied BEFORE o_proj.
    let g = cpu_matmul(hidden_normed, &w.g_proj, seq, hs, nq); // [seq, nq]
    for p in 0..seq {
        for h in 0..nq {
            let gate = softplus(g[p * nq + h]);
            let head = &mut attn_out[p * nq * hd + h * hd..p * nq * hd + (h + 1) * hd];
            for d in head.iter_mut() {
                *d *= gate;
            }
        }
    }

    // o_proj.
    cpu_matmul(&attn_out, &w.o_proj, seq, nq * hd, hs) // [seq, hidden]
}

/// Dense SwiGLU MLP over `[seq, hidden]`.
pub fn laguna_dense_mlp(h: &[f32], seq: usize, w: &OwnedDense, cfg: &LagunaConfig) -> Vec<f32> {
    let hs = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let g = cpu_matmul(h, &w.gate, seq, hs, inter);
    let u = cpu_matmul(h, &w.up, seq, hs, inter);
    let act: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| silu(a) * b).collect();
    cpu_matmul(&act, &w.down, seq, inter, hs)
}

/// MoE mixer for a SINGLE token `h` (`[hidden]`, post-norm). Router selects the
/// top-k experts, each dequantized on demand from packed NVFP4; the shared
/// expert (ungated gated-SiLU) always runs. Routed weights already carry the
/// ×routed_scaling_factor (baked in by `router_forward`); the shared expert is
/// NOT scaled.
pub fn laguna_moe_token(h: &[f32], w: &OwnedMoe, cfg: &LagunaConfig) -> Vec<f32> {
    let hs = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let shared_inter = cfg.shared_expert_intermediate_size;
    let dims = cfg.router_dims();

    let (indices, weights) = router_forward(h, &w.router, &w.bias, &dims);

    let mut routed = vec![0.0f32; hs];
    for (kth, &e) in indices.iter().enumerate() {
        let gate_w = w.experts.dequant(e, 0); // [inter, hidden]
        let up_w = w.experts.dequant(e, 1); // [inter, hidden]
        let down_w = w.experts.dequant(e, 2); // [hidden, inter]
        let gp = cpu_matmul(h, &gate_w, 1, hs, inter);
        let up = cpu_matmul(h, &up_w, 1, hs, inter);
        let act: Vec<f32> = gp.iter().zip(&up).map(|(&a, &b)| silu(a) * b).collect();
        let dn = cpu_matmul(&act, &down_w, 1, inter, hs); // [hidden]
        let wk = weights[kth];
        for (r, &o) in routed.iter_mut().zip(&dn) {
            *r += o * wk;
        }
    }

    // Ungated shared expert.
    let sg = cpu_matmul(h, &w.shared_gate, 1, hs, shared_inter);
    let su = cpu_matmul(h, &w.shared_up, 1, hs, shared_inter);
    let sact: Vec<f32> = sg.iter().zip(&su).map(|(&a, &b)| silu(a) * b).collect();
    let sd = cpu_matmul(&sact, &w.shared_down, 1, shared_inter, hs); // [hidden]

    routed.iter().zip(&sd).map(|(&r, &s)| r + s).collect()
}

/// One decoder layer: pre-norm gated attention + residual, then pre-norm
/// MoE/dense MLP + residual. `hidden` is `[seq, hidden]`.
pub fn laguna_layer_forward(
    hidden: &[f32],
    seq: usize,
    layer_idx: usize,
    lw: &OwnedLayer,
    cfg: &LagunaConfig,
) -> Vec<f32> {
    let hs = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    let normed = cpu_rms_norm(hidden, &lw.input_ln, eps);
    let attn = laguna_attn(&normed, seq, layer_idx, &lw.attn, cfg);
    let h1: Vec<f32> = hidden.iter().zip(&attn).map(|(&a, &b)| a + b).collect();

    let normed2 = cpu_rms_norm(&h1, &lw.post_ln, eps);
    let mlp = match &lw.mlp {
        OwnedMlp::Dense(d) => laguna_dense_mlp(&normed2, seq, d, cfg),
        OwnedMlp::Moe(m) => {
            let mut out = vec![0.0f32; seq * hs];
            for t in 0..seq {
                let ht = &normed2[t * hs..(t + 1) * hs];
                let m_out = laguna_moe_token(ht, m, cfg);
                out[t * hs..(t + 1) * hs].copy_from_slice(&m_out);
            }
            out
        }
    };
    h1.iter().zip(&mlp).map(|(&a, &b)| a + b).collect()
}

impl LagunaWeights {
    /// Full CPU forward over a token id sequence. Returns per-position
    /// `[seq, hidden]` (last_hidden_state, pre-lm_head) when `lm_head` is None,
    /// else logits for the LAST position only (`[vocab]`).
    pub fn forward(&self, tokens: &[u32], cfg: &LagunaConfig) -> Vec<f32> {
        let hs = cfg.hidden_size;
        let seq = tokens.len();
        let mut hidden = vec![0.0f32; seq * hs];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = &self.embed[tok as usize * hs..(tok as usize + 1) * hs];
            hidden[t * hs..(t + 1) * hs].copy_from_slice(row);
        }
        for (li, lw) in self.layers.iter().enumerate() {
            hidden = laguna_layer_forward(&hidden, seq, li, lw, cfg);
        }
        let normed = cpu_rms_norm(&hidden, &self.final_norm, cfg.rms_norm_eps);
        match &self.lm_head {
            None => normed,
            Some(lm) => {
                let last = &normed[(seq - 1) * hs..seq * hs];
                cpu_matmul(last, lm, 1, hs, cfg.vocab_size)
            }
        }
    }

    /// Forward that dumps each layer's post-residual hidden as raw LE-f32 to
    /// `{dump_path}.r0` when `dump_path` is Some — the `STAGE_HIDDEN_DUMP`
    /// bisect-ladder producer for `scripts/diff_f32.py`. Returns the same as
    /// [`LagunaWeights::forward`] but never applies lm_head (dumps hidden).
    pub fn forward_with_stage_dump(
        &self,
        tokens: &[u32],
        cfg: &LagunaConfig,
        dump_path: Option<&str>,
    ) -> Vec<f32> {
        use std::io::Write;
        let hs = cfg.hidden_size;
        let seq = tokens.len();
        let mut hidden = vec![0.0f32; seq * hs];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = &self.embed[tok as usize * hs..(tok as usize + 1) * hs];
            hidden[t * hs..(t + 1) * hs].copy_from_slice(row);
        }
        let write_stage = |name: &str, data: &[f32]| {
            if let Some(base) = dump_path {
                let path = format!("{base}.{name}.r0");
                let bytes = crate::push_constants::f32_slice_to_bytes(data);
                if let Ok(mut f) = std::fs::File::create(&path) {
                    let _ = f.write_all(&bytes);
                }
            }
        };
        write_stage("embed", &hidden);
        for (li, lw) in self.layers.iter().enumerate() {
            hidden = laguna_layer_forward(&hidden, seq, li, lw, cfg);
            write_stage(&format!("layer{li}"), &hidden);
        }
        let normed = cpu_rms_norm(&hidden, &self.final_norm, cfg.rms_norm_eps);
        write_stage("final_norm", &normed);
        normed
    }
}

// ─── Real-checkpoint loader (compiles now; runs when the shards are present) ──

/// Load `[0, num_layers)` of a Laguna checkpoint to host f32 (BF16 dequant) +
/// packed NVFP4 experts, keyed off `model.safetensors.index.json` in `dir`.
/// `keep_lm` also loads `lm_head.weight`. This is the pure-CPU reference loader
/// used by [`laguna_cpu_stage_oracle`]; it needs shards 1+2 for a 2-layer run
/// (~11GB) so it is exercised only by the ignored oracle test.
pub fn load_laguna_weights_cpu(
    dir: &std::path::Path,
    cfg: &LagunaConfig,
    num_layers: usize,
    keep_lm: bool,
) -> Result<LagunaWeights, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::collections::HashMap;

    // index.json maps tensor name → shard filename.
    let index_path = dir.join("model.safetensors.index.json");
    let index: Value = serde_json::from_str(
        &std::fs::read_to_string(&index_path).map_err(|e| format!("read index: {e}"))?,
    )
    .map_err(|e| format!("parse index: {e}"))?;
    let weight_map = index
        .get("weight_map")
        .and_then(|x| x.as_object())
        .ok_or("index.json: missing weight_map")?;

    // Group the tensor names we need by shard so each shard is mmapped once.
    let hs = cfg.hidden_size;
    let hd = cfg.head_dim;
    let nkv = cfg.num_key_value_heads;
    let inter_moe = cfg.moe_intermediate_size;
    let shared_inter = cfg.shared_expert_intermediate_size;
    let group = 16usize;

    // Which tensor names are wanted → we simply resolve on demand per shard.
    let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
    let want = |name: String, set: &mut HashMap<String, Vec<String>>| {
        if let Some(shard) = weight_map.get(&name).and_then(|x| x.as_str()) {
            set.entry(shard.to_string()).or_default().push(name);
        }
    };
    want("model.embed_tokens.weight".into(), &mut by_shard);
    want("model.norm.weight".into(), &mut by_shard);
    if keep_lm {
        want("lm_head.weight".into(), &mut by_shard);
    }
    for li in 0..num_layers {
        let p = format!("model.layers.{li}");
        for t in [
            format!("{p}.input_layernorm.weight"),
            format!("{p}.post_attention_layernorm.weight"),
            format!("{p}.self_attn.q_proj.weight"),
            format!("{p}.self_attn.k_proj.weight"),
            format!("{p}.self_attn.v_proj.weight"),
            format!("{p}.self_attn.o_proj.weight"),
            format!("{p}.self_attn.g_proj.weight"),
            format!("{p}.self_attn.q_norm.weight"),
            format!("{p}.self_attn.k_norm.weight"),
        ] {
            want(t, &mut by_shard);
        }
        if cfg.mlp_only_layers.contains(&li) {
            for t in [
                format!("{p}.mlp.gate_proj.weight"),
                format!("{p}.mlp.up_proj.weight"),
                format!("{p}.mlp.down_proj.weight"),
            ] {
                want(t, &mut by_shard);
            }
        } else {
            want(format!("{p}.mlp.gate.weight"), &mut by_shard);
            want(format!("{p}.mlp.experts.e_score_correction_bias"), &mut by_shard);
            for t in [
                format!("{p}.mlp.shared_expert.gate_proj.weight"),
                format!("{p}.mlp.shared_expert.up_proj.weight"),
                format!("{p}.mlp.shared_expert.down_proj.weight"),
            ] {
                want(t, &mut by_shard);
            }
            for e in 0..cfg.num_experts {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    let b = format!("{p}.mlp.experts.{e}.{proj}");
                    want(format!("{b}.weight_packed"), &mut by_shard);
                    want(format!("{b}.weight_scale"), &mut by_shard);
                    want(format!("{b}.weight_global_scale"), &mut by_shard);
                }
            }
        }
    }

    // f32 tensor store (plain BF16→f32), u8 store (packed / e4m3 scales),
    // f32-scalar store (globals).
    let mut f32s: HashMap<String, Vec<f32>> = HashMap::new();
    let mut u8s: HashMap<String, Vec<u8>> = HashMap::new();

    for (shard, names) in &by_shard {
        let path = dir.join(shard);
        let file = std::fs::File::open(&path).map_err(|e| format!("open {shard}: {e}"))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap {shard}: {e}"))? };
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse {shard}: {e}"))?;
        for name in names {
            let view = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
            if name.ends_with(".weight_packed") || name.ends_with(".weight_scale") {
                u8s.insert(name.clone(), view.data().to_vec());
            } else if name.ends_with(".weight_global_scale") {
                let g = f32::from_le_bytes(
                    view.data()[..4].try_into().map_err(|_| format!("{name}: short global"))?,
                );
                f32s.insert(name.clone(), vec![g]);
            } else {
                f32s.insert(name.clone(), decode_bf16_f32(&view)?);
            }
        }
    }

    let take_f32 = |m: &mut HashMap<String, Vec<f32>>, n: &str| -> Result<Vec<f32>, String> {
        m.remove(n).ok_or_else(|| format!("missing tensor {n}"))
    };

    let embed = take_f32(&mut f32s, "model.embed_tokens.weight")?;
    let final_norm = take_f32(&mut f32s, "model.norm.weight")?;
    let lm_head = if keep_lm { Some(take_f32(&mut f32s, "lm_head.weight")?) } else { None };

    let mut layers = Vec::with_capacity(num_layers);
    for li in 0..num_layers {
        let p = format!("model.layers.{li}");
        let attn = OwnedAttn {
            q_proj: take_f32(&mut f32s, &format!("{p}.self_attn.q_proj.weight"))?,
            k_proj: take_f32(&mut f32s, &format!("{p}.self_attn.k_proj.weight"))?,
            v_proj: take_f32(&mut f32s, &format!("{p}.self_attn.v_proj.weight"))?,
            o_proj: take_f32(&mut f32s, &format!("{p}.self_attn.o_proj.weight"))?,
            g_proj: take_f32(&mut f32s, &format!("{p}.self_attn.g_proj.weight"))?,
            q_norm: take_f32(&mut f32s, &format!("{p}.self_attn.q_norm.weight"))?,
            k_norm: take_f32(&mut f32s, &format!("{p}.self_attn.k_norm.weight"))?,
        };
        let mlp = if cfg.mlp_only_layers.contains(&li) {
            OwnedMlp::Dense(OwnedDense {
                gate: take_f32(&mut f32s, &format!("{p}.mlp.gate_proj.weight"))?,
                up: take_f32(&mut f32s, &format!("{p}.mlp.up_proj.weight"))?,
                down: take_f32(&mut f32s, &format!("{p}.mlp.down_proj.weight"))?,
            })
        } else {
            // concatenate per-expert packed tensors in expert order.
            let mut ex = OwnedExpertsPacked {
                gate_packed: Vec::new(),
                gate_scale: Vec::new(),
                gate_global: Vec::with_capacity(cfg.num_experts),
                up_packed: Vec::new(),
                up_scale: Vec::new(),
                up_global: Vec::with_capacity(cfg.num_experts),
                down_packed: Vec::new(),
                down_scale: Vec::new(),
                down_global: Vec::with_capacity(cfg.num_experts),
                num_experts: cfg.num_experts,
                inter: inter_moe,
                hidden: hs,
                group_size: group,
            };
            for e in 0..cfg.num_experts {
                let b = format!("{p}.mlp.experts.{e}");
                ex.gate_packed.extend(u8s.remove(&format!("{b}.gate_proj.weight_packed"))
                    .ok_or_else(|| format!("missing {b}.gate_proj.weight_packed"))?);
                ex.gate_scale.extend(u8s.remove(&format!("{b}.gate_proj.weight_scale"))
                    .ok_or_else(|| format!("missing {b}.gate_proj.weight_scale"))?);
                ex.gate_global.push(f32s.remove(&format!("{b}.gate_proj.weight_global_scale"))
                    .ok_or_else(|| format!("missing {b}.gate_proj.weight_global_scale"))?[0]);
                ex.up_packed.extend(u8s.remove(&format!("{b}.up_proj.weight_packed"))
                    .ok_or_else(|| format!("missing {b}.up_proj.weight_packed"))?);
                ex.up_scale.extend(u8s.remove(&format!("{b}.up_proj.weight_scale"))
                    .ok_or_else(|| format!("missing {b}.up_proj.weight_scale"))?);
                ex.up_global.push(f32s.remove(&format!("{b}.up_proj.weight_global_scale"))
                    .ok_or_else(|| format!("missing {b}.up_proj.weight_global_scale"))?[0]);
                ex.down_packed.extend(u8s.remove(&format!("{b}.down_proj.weight_packed"))
                    .ok_or_else(|| format!("missing {b}.down_proj.weight_packed"))?);
                ex.down_scale.extend(u8s.remove(&format!("{b}.down_proj.weight_scale"))
                    .ok_or_else(|| format!("missing {b}.down_proj.weight_scale"))?);
                ex.down_global.push(f32s.remove(&format!("{b}.down_proj.weight_global_scale"))
                    .ok_or_else(|| format!("missing {b}.down_proj.weight_global_scale"))?[0]);
            }
            OwnedMlp::Moe(OwnedMoe {
                router: take_f32(&mut f32s, &format!("{p}.mlp.gate.weight"))?,
                bias: take_f32(&mut f32s, &format!("{p}.mlp.experts.e_score_correction_bias"))?,
                experts: ex,
                shared_gate: take_f32(&mut f32s, &format!("{p}.mlp.shared_expert.gate_proj.weight"))?,
                shared_up: take_f32(&mut f32s, &format!("{p}.mlp.shared_expert.up_proj.weight"))?,
                shared_down: take_f32(&mut f32s, &format!("{p}.mlp.shared_expert.down_proj.weight"))?,
            })
        };
        layers.push(OwnedLayer {
            input_ln: take_f32(&mut f32s, &format!("{p}.input_layernorm.weight"))?,
            post_ln: take_f32(&mut f32s, &format!("{p}.post_attention_layernorm.weight"))?,
            attn,
            mlp,
        });
    }

    // touch a couple of derived dims so an unused-var lint never fires.
    let _ = (hd, nkv, shared_inter);
    Ok(LagunaWeights { embed, final_norm, layers, lm_head })
}

/// Load JUST ONE layer's weights (attn + norms + dense/MoE) to the CPU-oracle
/// `OwnedLayer` format, by GLOBBING the shard files (`discover_shards`) rather
/// than reading `model.safetensors.index.json` — so it works while the index is
/// still staging (the resident loader uses the same glob). Memory ≈ one layer's
/// packed experts (~1.3GB), NOT the whole `[0,L]` prefix `load_cpu` forces. This
/// is the reference producer for the single-node GPU bit-exact GATE
/// (`debug_laguna_layer`): feed an identical `hidden` to
/// [`laguna_layer_forward`] with this `OwnedLayer` and to the resident
/// `LagunaGpuModel::layer_forward`, compare cos/argmax.
pub fn load_owned_layer_cpu(
    dir: &std::path::Path,
    cfg: &LagunaConfig,
    layer_idx: usize,
) -> Result<OwnedLayer, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::collections::HashMap;

    let p = format!("model.layers.{layer_idx}");
    let is_dense = cfg.mlp_only_layers.contains(&layer_idx);
    let mut f32s: HashMap<String, Vec<f32>> = HashMap::new();
    let mut u8s: HashMap<String, Vec<u8>> = HashMap::new();

    let want = |name: &str| -> bool { name.starts_with(&format!("{p}.")) };

    // Glob the checkpoint DIR directly (discover_shards globs a FILE's parent,
    // so it can't be handed a dir). Works while index.json is still staging.
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir {}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    for shard in shards {
        let file = std::fs::File::open(&shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse: {e}"))?;
        for (name, view) in st.tensors() {
            if !want(&name) {
                continue;
            }
            if name.ends_with(".weight_packed") || name.ends_with(".weight_scale") {
                u8s.insert(name.clone(), view.data().to_vec());
            } else if name.ends_with(".weight_global_scale") {
                let g = f32::from_le_bytes(
                    view.data()[..4].try_into().map_err(|_| format!("{name}: short global"))?,
                );
                f32s.insert(name.clone(), vec![g]);
            } else if name.ends_with(".input_scale")
                || name.ends_with(".input_global_scale")
                || name.ends_with(".k_scale")
                || name.ends_with(".v_scale")
            {
                // W4A16 activation scales — ignored (see loader).
            } else {
                f32s.insert(name.clone(), decode_bf16_f32(&view)?);
            }
        }
    }

    let take = |m: &mut HashMap<String, Vec<f32>>, n: &str| -> Result<Vec<f32>, String> {
        m.remove(n).ok_or_else(|| format!("missing tensor {n} (layer {layer_idx} not fully staged?)"))
    };

    let attn = OwnedAttn {
        q_proj: take(&mut f32s, &format!("{p}.self_attn.q_proj.weight"))?,
        k_proj: take(&mut f32s, &format!("{p}.self_attn.k_proj.weight"))?,
        v_proj: take(&mut f32s, &format!("{p}.self_attn.v_proj.weight"))?,
        o_proj: take(&mut f32s, &format!("{p}.self_attn.o_proj.weight"))?,
        g_proj: take(&mut f32s, &format!("{p}.self_attn.g_proj.weight"))?,
        q_norm: take(&mut f32s, &format!("{p}.self_attn.q_norm.weight"))?,
        k_norm: take(&mut f32s, &format!("{p}.self_attn.k_norm.weight"))?,
    };
    let mlp = if is_dense {
        OwnedMlp::Dense(OwnedDense {
            gate: take(&mut f32s, &format!("{p}.mlp.gate_proj.weight"))?,
            up: take(&mut f32s, &format!("{p}.mlp.up_proj.weight"))?,
            down: take(&mut f32s, &format!("{p}.mlp.down_proj.weight"))?,
        })
    } else {
        let mut ex = OwnedExpertsPacked {
            gate_packed: Vec::new(), gate_scale: Vec::new(), gate_global: Vec::with_capacity(cfg.num_experts),
            up_packed: Vec::new(), up_scale: Vec::new(), up_global: Vec::with_capacity(cfg.num_experts),
            down_packed: Vec::new(), down_scale: Vec::new(), down_global: Vec::with_capacity(cfg.num_experts),
            num_experts: cfg.num_experts,
            inter: cfg.moe_intermediate_size,
            hidden: cfg.hidden_size,
            group_size: 16,
        };
        for e in 0..cfg.num_experts {
            let b = format!("{p}.mlp.experts.{e}");
            ex.gate_packed.extend(u8s.remove(&format!("{b}.gate_proj.weight_packed"))
                .ok_or_else(|| format!("missing {b}.gate_proj.weight_packed"))?);
            ex.gate_scale.extend(u8s.remove(&format!("{b}.gate_proj.weight_scale"))
                .ok_or_else(|| format!("missing {b}.gate_proj.weight_scale"))?);
            ex.gate_global.push(f32s.remove(&format!("{b}.gate_proj.weight_global_scale"))
                .ok_or_else(|| format!("missing {b}.gate_proj.weight_global_scale"))?[0]);
            ex.up_packed.extend(u8s.remove(&format!("{b}.up_proj.weight_packed"))
                .ok_or_else(|| format!("missing {b}.up_proj.weight_packed"))?);
            ex.up_scale.extend(u8s.remove(&format!("{b}.up_proj.weight_scale"))
                .ok_or_else(|| format!("missing {b}.up_proj.weight_scale"))?);
            ex.up_global.push(f32s.remove(&format!("{b}.up_proj.weight_global_scale"))
                .ok_or_else(|| format!("missing {b}.up_proj.weight_global_scale"))?[0]);
            ex.down_packed.extend(u8s.remove(&format!("{b}.down_proj.weight_packed"))
                .ok_or_else(|| format!("missing {b}.down_proj.weight_packed"))?);
            ex.down_scale.extend(u8s.remove(&format!("{b}.down_proj.weight_scale"))
                .ok_or_else(|| format!("missing {b}.down_proj.weight_scale"))?);
            ex.down_global.push(f32s.remove(&format!("{b}.down_proj.weight_global_scale"))
                .ok_or_else(|| format!("missing {b}.down_proj.weight_global_scale"))?[0]);
        }
        OwnedMlp::Moe(OwnedMoe {
            router: take(&mut f32s, &format!("{p}.mlp.gate.weight"))?,
            bias: take(&mut f32s, &format!("{p}.mlp.experts.e_score_correction_bias"))?,
            experts: ex,
            shared_gate: take(&mut f32s, &format!("{p}.mlp.shared_expert.gate_proj.weight"))?,
            shared_up: take(&mut f32s, &format!("{p}.mlp.shared_expert.up_proj.weight"))?,
            shared_down: take(&mut f32s, &format!("{p}.mlp.shared_expert.down_proj.weight"))?,
        })
    };
    Ok(OwnedLayer {
        input_ln: take(&mut f32s, &format!("{p}.input_layernorm.weight"))?,
        post_ln: take(&mut f32s, &format!("{p}.post_attention_layernorm.weight"))?,
        attn,
        mlp,
    })
}

fn decode_bf16_f32(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>, String> {
    let d = view.data();
    Ok(match view.dtype() {
        safetensors::Dtype::BF16 => d
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        safetensors::Dtype::F16 => d
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        safetensors::Dtype::F32 => d
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => return Err(format!("unsupported plain dtype {other:?} in Laguna loader")),
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Crate-visible synthetic tiny Laguna config (hidden 32, head_dim 8, 4 KV
/// heads, sliding_window 4, `layer_is_full` = every 4th layer). Test-only —
/// reused by `laguna_gpu`'s KV-tile round-trip gate, which needs a config
/// outside this module's private `tests`.
#[cfg(test)]
pub(crate) fn tiny_test_config(num_layers: usize) -> LagunaConfig {
    {
        // A synthetic tiny Laguna: hidden 32, head_dim 8, 4 KV heads, tiny MoE.
        let hidden = 32;
        let head_dim = 8;
        let full_partial = 0.5f32;
        let full_rotary_dim = (head_dim as f32 * full_partial) as usize; // 4
        let yarn = compute_yarn_inv_freq(500000.0, head_dim, full_partial, 32.0, 32.0, 1.0, 8192);
        LagunaConfig {
            hidden_size: hidden,
            num_hidden_layers: num_layers,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            intermediate_size: 48,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 16,
            num_experts: 8,
            num_experts_per_tok: 3,
            norm_topk_prob: true,
            moe_routed_scaling_factor: 2.5,
            moe_router_logit_softcapping: 0.0,
            sliding_window: 4,
            max_position_embeddings: 8192,
            mlp_only_layers: vec![0],
            layer_is_full: (0..num_layers).map(|i| i % 4 == 0).collect(),
            num_attention_heads_per_layer: (0..num_layers)
                .map(|i| if i % 4 == 0 { 8 } else { 12 })
                .collect(),
            full_rope_theta: 500000.0,
            full_rope_factor: 32.0,
            full_orig_max_pos: 8192,
            full_beta_fast: 32.0,
            full_beta_slow: 1.0,
            full_attention_factor: 1.3465735902799727,
            full_partial_rotary: full_partial,
            full_rotary_dim,
            yarn_inv_freq: yarn,
            sliding_rope_theta: 10000.0,
            sliding_partial_rotary: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg(num_layers: usize) -> LagunaConfig {
        super::tiny_test_config(num_layers)
    }

    // Deterministic pseudo-random f32 fill in [-a, a].
    fn fill(n: usize, seed: u64, a: f32) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
                (u * 2.0 - 1.0) * a
            })
            .collect()
    }

    // ─── YaRN table vs the numpy _compute_yarn_parameters port ───────────────
    #[test]
    fn yarn_inv_freq_matches_transformers() {
        // Reference computed by /tmp/yarn_ref.py (transformers
        // _compute_yarn_parameters) for the real Laguna full_attention params.
        let expect: [f32; 32] = [
            1.0000000000e0, 6.6360127926e-1, 4.4036662579e-1, 2.9222780466e-1,
            1.9392275810e-1, 1.2868738174e-1, 8.5397101939e-2, 5.6669618934e-2,
            3.7606030703e-2, 2.4955408648e-2, 1.4777894132e-2, 8.6237275973e-3,
            4.9377414398e-3, 2.7557816356e-3, 1.4830635628e-3, 7.5477146311e-4,
            3.4864293411e-4, 1.3034358562e-4, 1.9461638658e-5, 1.2914767467e-5,
            8.5702558863e-6, 5.6872322602e-6, 3.7740544485e-6, 2.5044671474e-6,
            1.6619674170e-6, 1.1028836298e-6, 7.3187493399e-7, 4.8567312660e-7,
            3.2229328895e-7, 2.1387423033e-7, 1.4192720243e-7, 9.4183064903e-8,
        ];
        let got = compute_yarn_inv_freq(500000.0, 128, 0.5, 32.0, 32.0, 1.0, 8192);
        assert_eq!(got.len(), 32);
        for (i, (&g, &e)) in got.iter().zip(&expect).enumerate() {
            let rel = (g - e).abs() / e.abs().max(1e-12);
            assert!(rel < 1e-5, "yarn inv_freq[{i}]: got {g:e} expect {e:e} rel {rel:e}");
        }
        // attention_factor default check.
        let af = 0.1f32 * 32.0f32.ln() + 1.0;
        assert!((af - 1.3465735902799727).abs() < 1e-6, "attention_factor {af}");
    }

    // ─── Router: sigmoid + bias top-k vs an explicit numpy-style oracle ──────
    #[test]
    fn router_sigmoid_bias_topk_matches_oracle() {
        let ne = 8usize;
        let top_k = 3usize;
        let hidden = 4usize;
        let h = fill(hidden, 1, 1.0);
        let gate = fill(ne * hidden, 2, 0.7);
        let bias = fill(ne, 3, 0.3);
        let dims = RouterDims {
            n_routed_experts: ne,
            top_k,
            routed_scaling_factor: 2.5,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
        };
        let (idx, wts) = router_forward(&h, &gate, &bias, &dims);

        // Oracle: logits = gate @ h; scores = sigmoid(logits);
        // choice = scores + bias; top-k by choice; weights = scores[idx]
        // renormed then ×2.5.
        let sigmoid = |z: f32| 1.0 / (1.0 + (-z).exp());
        let mut scores = vec![0.0f32; ne];
        let mut choice = vec![0.0f32; ne];
        for e in 0..ne {
            let mut z = 0.0f32;
            for j in 0..hidden {
                z += gate[e * hidden + j] * h[j];
            }
            scores[e] = sigmoid(z);
            choice[e] = scores[e] + bias[e];
        }
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| {
            choice[b].partial_cmp(&choice[a]).unwrap().then(a.cmp(&b))
        });
        let exp_idx: Vec<usize> = order.into_iter().take(top_k).collect();
        assert_eq!(idx, exp_idx, "router top-k index set");
        let mut exp_w: Vec<f32> = exp_idx.iter().map(|&e| scores[e]).collect();
        let denom: f32 = exp_w.iter().sum::<f32>() + 1e-20;
        for w in exp_w.iter_mut() {
            *w = *w / denom * 2.5;
        }
        for (i, (&g, &e)) in wts.iter().zip(&exp_w).enumerate() {
            assert!((g - e).abs() < 1e-5, "router weight[{i}]: got {g} expect {e}");
        }
    }

    // ─── Per-head softplus gate broadcast ────────────────────────────────────
    #[test]
    fn softplus_gate_broadcast() {
        // gate value g applied to every channel of a head.
        let x = 0.75f32;
        let sp = super::softplus(x);
        let np_ref = x.max(0.0) + (-x.abs()).exp().ln_1p();
        assert!((sp - np_ref).abs() < 1e-7);
        // spot vs closed form log(1+e^x)
        assert!((sp - (1.0 + x.exp()).ln()).abs() < 1e-5, "softplus {sp}");
    }

    // ─── RMSNorm building block vs explicit mean/rsqrt ───────────────────────
    #[test]
    fn rms_norm_matches_reference() {
        let n = 8;
        let x = fill(n, 9, 2.0);
        let w = fill(n, 10, 1.0);
        let eps = 1e-6f32;
        let got = cpu_rms_norm(&x, &w, eps);
        let mean = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (mean + eps).sqrt();
        for i in 0..n {
            let e = x[i] * inv * w[i];
            assert!((got[i] - e).abs() < 1e-5, "rmsnorm[{i}] {} vs {e}", got[i]);
        }
    }

    // ─── YaRN rope pairing == cpu_rope_with_basis (freq_dim==rotary_dim) ─────
    #[test]
    fn yarn_rope_pairing_is_neox_partial() {
        // With inv_freq set to the plain 1/theta^(2i/rotary) basis and
        // mscale=1, cpu_rope_yarn must equal cpu_rope_with_basis over the
        // rotated span (dims [rotary..head_dim) untouched by both).
        let head_dim = 8;
        let rotary = 4;
        let nq = 2;
        let nkv = 1;
        let theta = 10000.0f32;
        let inv: Vec<f32> = (0..rotary / 2)
            .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / rotary as f32))
            .collect();
        let base_q = fill(nq * head_dim, 21, 1.0);
        let base_k = fill(nkv * head_dim, 22, 1.0);
        let pos = 5;

        let mut q1 = base_q.clone();
        let mut k1 = base_k.clone();
        cpu_rope_yarn(&mut q1, &mut k1, pos, nq, nkv, head_dim, rotary, &inv, 1.0);

        let mut q2 = base_q.clone();
        let mut k2 = base_k.clone();
        crate::model::cpu_rope_with_basis(
            &mut q2, &mut k2, pos, nq, nkv, head_dim, rotary, rotary, theta,
        );
        for i in 0..nq * head_dim {
            assert!((q1[i] - q2[i]).abs() < 1e-5, "q[{i}] {} vs {}", q1[i], q2[i]);
        }
        for i in 0..nkv * head_dim {
            assert!((k1[i] - k2[i]).abs() < 1e-5, "k[{i}] {} vs {}", k1[i], k2[i]);
        }
    }

    // ─── Synthetic NVFP4 expert: dequant+matvec vs a numpy-order oracle ──────
    #[test]
    fn nvfp4_expert_dequant_matvec_synth() {
        // Build a tiny NVFP4 expert (out=4, in=32, group=16) with known bytes,
        // dequant via OwnedExpertsPacked (reusing dequantize_nvfp4), matvec a
        // synthetic activation, and compare to a straight LUT*scale*(1/global)
        // reference done in the numpy op-order.
        let out_f = 4usize;
        let in_f = 32usize;
        let group = 16usize;
        let packed: Vec<u8> = (0..out_f * in_f / 2).map(|i| (i * 37 + 5) as u8).collect();
        let scale: Vec<u8> = (0..out_f * in_f / group).map(|i| (0x38 + (i % 8)) as u8).collect(); // ~1..2 range e4m3
        let global_raw = 2048.0f32;

        let ex = OwnedExpertsPacked {
            gate_packed: packed.clone(),
            gate_scale: scale.clone(),
            gate_global: vec![global_raw],
            up_packed: packed.clone(),
            up_scale: scale.clone(),
            up_global: vec![global_raw],
            down_packed: packed.clone(),
            down_scale: scale.clone(),
            down_global: vec![global_raw],
            num_experts: 1,
            inter: out_f,
            hidden: in_f,
            group_size: group,
        };
        let w = ex.dequant(0, 0); // [out_f, in_f]
        assert_eq!(w.len(), out_f * in_f);

        // Reference dequant, numpy op-order.
        let inv_global = 1.0f32 / global_raw;
        let mut wref = vec![0.0f32; out_f * in_f];
        for o in 0..out_f {
            for i in 0..in_f {
                let byte = packed[o * (in_f / 2) + i / 2];
                let nib = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let bscale = crate::model::e4m3_to_f32(scale[o * (in_f / group) + i / group]);
                wref[o * in_f + i] =
                    crate::model::NVFP4_E2M1_LUT[nib as usize] * bscale * inv_global;
            }
        }
        for i in 0..out_f * in_f {
            assert_eq!(w[i].to_bits(), wref[i].to_bits(), "dequant bit mismatch at {i}");
        }

        // matvec: y = W @ x  vs manual dot products.
        let x = fill(in_f, 55, 1.0);
        let y = cpu_matmul(&x, &w, 1, in_f, out_f);
        for o in 0..out_f {
            let mut acc = 0.0f32;
            for i in 0..in_f {
                acc += wref[o * in_f + i] * x[i];
            }
            assert!((y[o] - acc).abs() < 1e-4, "matvec[{o}] {} vs {acc}", y[o]);
        }
    }

    // ─── Cross-language bit-exact dump: Rust dequantize_nvfp4 → .bin for the
    //     three real L1E0 projections, so a numpy np.array_equal can assert the
    //     Rust and dequant_ref.py outputs are IDENTICAL f32 bits (A.2 gate). ──
    //     Run: VLLM_LAGUNA_DUMP_DEQUANT=1 cargo test --lib -- --ignored \
    //          --nocapture dump_rust_dequant_real_L1E0
    #[test]
    #[ignore]
    fn dump_rust_dequant_real_L1E0() {
        let home = std::env::var("HOME").expect("HOME");
        let dir = std::path::Path::new(&home).join("laguna_parity");
        let rd = |n: &str| std::fs::read(dir.join(n)).expect("read bin");
        // (proj_name, out_f, in_f)
        let projs = [("gate_proj", 1024usize, 3072usize),
                     ("up_proj", 1024, 3072),
                     ("down_proj", 3072, 1024)];
        for (proj, out_f, in_f) in projs {
            let packed = rd(&format!("1_0_{proj}_weight_packed.bin"));
            let scale = rd(&format!("1_0_{proj}_weight_scale.bin"));
            let gbytes = rd(&format!("1_0_{proj}_weight_global_scale.bin"));
            let global_raw = f32::from_le_bytes(gbytes[..4].try_into().unwrap());
            // Reciprocal, exactly as OwnedExpertsPacked::dequant applies it.
            let global = 1.0f32 / global_raw;
            let w = dequantize_nvfp4(&packed, &scale, global, out_f, in_f, 16);
            assert_eq!(w.len(), out_f * in_f);
            let mut bytes = Vec::with_capacity(w.len() * 4);
            for &v in &w {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let out = dir.join(format!("rust_dequant_1_0_{proj}.bin"));
            std::fs::write(&out, &bytes).expect("write rust dequant");
            eprintln!("wrote {} ({} f32, global_raw={global_raw})", out.display(), w.len());
        }
    }

    // ─── One real expert (L1 E0) from ~/laguna_parity, if present ────────────
    #[test]
    fn nvfp4_real_expert_L1E0_sanity() {
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => return,
        };
        let dir = std::path::Path::new(&home).join("laguna_parity");
        let gate_p = dir.join("1_0_gate_proj_weight_packed.bin");
        if !gate_p.exists() {
            eprintln!("~/laguna_parity not populated — skipping real-expert test");
            return;
        }
        let rd = |n: &str| std::fs::read(dir.join(n)).expect("read bin");
        let packed = rd("1_0_gate_proj_weight_packed.bin"); // U8 [1024,1536]
        let scale = rd("1_0_gate_proj_weight_scale.bin"); // F8_E4M3 [1024,192]
        let gbytes = rd("1_0_gate_proj_weight_global_scale.bin");
        let global_raw = f32::from_le_bytes(gbytes[..4].try_into().unwrap());
        let out_f = 1024usize;
        let in_f = 3072usize;
        let ex = OwnedExpertsPacked {
            gate_packed: packed,
            gate_scale: scale,
            gate_global: vec![global_raw],
            up_packed: Vec::new(),
            up_scale: Vec::new(),
            up_global: Vec::new(),
            down_packed: Vec::new(),
            down_scale: Vec::new(),
            down_global: Vec::new(),
            num_experts: 1,
            inter: out_f,
            hidden: in_f,
            group_size: 16,
        };
        let w = ex.dequant(0, 0);
        let amax = w.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        // dequant_ref.py: global_raw=11520, abs.max≈0.2333.
        assert!((global_raw - 11520.0).abs() < 1.0, "global_raw {global_raw}");
        assert!(amax > 0.01 && amax < 100.0, "abs.max {amax} out of sane NVFP4 range");
        assert!((amax - 0.233333).abs() < 1e-3, "abs.max {amax} != numpy 0.23333");
        // W[0,0]==0.05 per dequant_ref.py sample.
        assert!((w[0] - 0.05).abs() < 1e-4, "W[0,0] {}", w[0]);
    }

    // ─── End-to-end synthetic forward: shape flow, no NaN, 2 layers ──────────
    #[test]
    fn synthetic_two_layer_forward_no_nan() {
        let cfg = tiny_cfg(2);
        let hs = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nkv = cfg.num_key_value_heads;
        let inter_moe = cfg.moe_intermediate_size;

        let make_attn = |li: usize| {
            let nq = cfg.num_attention_heads_per_layer[li];
            OwnedAttn {
                q_proj: fill(nq * hd * hs, 100 + li as u64, 0.1),
                k_proj: fill(nkv * hd * hs, 200 + li as u64, 0.1),
                v_proj: fill(nkv * hd * hs, 300 + li as u64, 0.1),
                o_proj: fill(hs * nq * hd, 400 + li as u64, 0.1),
                g_proj: fill(nq * hs, 500 + li as u64, 0.1),
                q_norm: vec![1.0; hd],
                k_norm: vec![1.0; hd],
            }
        };

        // layer 0: dense
        let l0 = OwnedLayer {
            input_ln: vec![1.0; hs],
            post_ln: vec![1.0; hs],
            attn: make_attn(0),
            mlp: OwnedMlp::Dense(OwnedDense {
                gate: fill(cfg.intermediate_size * hs, 11, 0.1),
                up: fill(cfg.intermediate_size * hs, 12, 0.1),
                down: fill(hs * cfg.intermediate_size, 13, 0.1),
            }),
        };

        // layer 1: MoE with synthetic NVFP4 experts.
        let gp_bytes = cfg.num_experts * inter_moe * hs / 2;
        let gp_scale = cfg.num_experts * inter_moe * (hs / 16);
        let dp_bytes = cfg.num_experts * hs * inter_moe / 2;
        let dp_scale = cfg.num_experts * hs * (inter_moe / 16);
        let byte_fill = |n: usize, seed: usize| -> Vec<u8> {
            (0..n).map(|i| ((i * 31 + seed * 7) % 256) as u8).collect()
        };
        // e4m3 bytes in a modest positive range (exp≈7 → ~1.x).
        let scale_fill = |n: usize| -> Vec<u8> { (0..n).map(|i| (0x38 + (i % 6)) as u8).collect() };
        let ex = OwnedExpertsPacked {
            gate_packed: byte_fill(gp_bytes, 1),
            gate_scale: scale_fill(gp_scale),
            gate_global: vec![2048.0; cfg.num_experts],
            up_packed: byte_fill(gp_bytes, 2),
            up_scale: scale_fill(gp_scale),
            up_global: vec![2048.0; cfg.num_experts],
            down_packed: byte_fill(dp_bytes, 3),
            down_scale: scale_fill(dp_scale),
            down_global: vec![2048.0; cfg.num_experts],
            num_experts: cfg.num_experts,
            inter: inter_moe,
            hidden: hs,
            group_size: 16,
        };
        let l1 = OwnedLayer {
            input_ln: vec![1.0; hs],
            post_ln: vec![1.0; hs],
            attn: make_attn(1),
            mlp: OwnedMlp::Moe(OwnedMoe {
                router: fill(cfg.num_experts * hs, 71, 0.3),
                bias: fill(cfg.num_experts, 72, 0.1),
                experts: ex,
                shared_gate: fill(cfg.shared_expert_intermediate_size * hs, 73, 0.1),
                shared_up: fill(cfg.shared_expert_intermediate_size * hs, 74, 0.1),
                shared_down: fill(hs * cfg.shared_expert_intermediate_size, 75, 0.1),
            }),
        };

        let weights = LagunaWeights {
            embed: fill(cfg.vocab_size * hs, 900, 0.5),
            final_norm: vec![1.0; hs],
            layers: vec![l0, l1],
            lm_head: None,
        };

        let tokens = [1u32, 5, 3, 7, 2];
        let out = weights.forward(&tokens, &cfg);
        assert_eq!(out.len(), tokens.len() * hs, "output shape [seq,hidden]");
        assert!(out.iter().all(|v| v.is_finite()), "forward produced non-finite output");
        let l2 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(l2 > 0.0, "output is all zero");

        // lm_head path shape.
        let mut w2 = weights;
        w2.lm_head = Some(fill(cfg.vocab_size * hs, 950, 0.1));
        let logits = w2.forward(&tokens, &cfg);
        assert_eq!(logits.len(), cfg.vocab_size, "logits shape [vocab]");
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    // ─── from_json against the real Laguna config.json (if present) ──────────
    #[test]
    fn config_from_json_real() {
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => return,
        };
        let path = std::path::Path::new(&home).join("laguna_parity/config.json");
        if !path.exists() {
            eprintln!("no laguna config.json — skipping");
            return;
        }
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let cfg = LagunaConfig::from_json(&v).expect("parse LagunaConfig");
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 100352);
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 10);
        assert_eq!(cfg.moe_intermediate_size, 1024);
        assert_eq!(cfg.mlp_only_layers, vec![0]);
        assert_eq!(cfg.sliding_window, 512);
        assert!((cfg.moe_routed_scaling_factor - 2.5).abs() < 1e-6);
        assert_eq!(cfg.full_rotary_dim, 64);
        assert_eq!(cfg.yarn_inv_freq.len(), 32);
        assert!((cfg.full_attention_factor - 1.3465735902799727).abs() < 1e-6);
        // layer 0 full, layer 1 sliding; every 4th full.
        assert!(cfg.layer_is_full[0]);
        assert!(!cfg.layer_is_full[1]);
        assert!(cfg.layer_is_full[4]);
        assert_eq!(cfg.num_attention_heads_per_layer[0], 48);
        assert_eq!(cfg.num_attention_heads_per_layer[1], 72);
    }

    /// Phase-0 per-layer KV sizing: full (YaRN, i%4==0) layers get `max_seq`;
    /// sliding layers get `sliding_window.min(max_seq)`. Clamped so a short
    /// context never over-allocates. Mirrors `Gemma4Config::layer_kv_capacity`.
    #[test]
    fn layer_kv_capacity_full_vs_sliding() {
        let cfg = tiny_cfg(8); // sliding_window == 4; layer_is_full == (i % 4 == 0)
        let max_seq = 64usize;
        for l in 0..8 {
            let cap = cfg.layer_kv_capacity(l, max_seq);
            if cfg.layer_is_full[l] {
                assert_eq!(cap, max_seq, "full layer {l} must keep max_seq");
            } else {
                assert_eq!(cap, cfg.sliding_window, "sliding layer {l} must shrink to window");
                assert!(cap < max_seq, "sliding capacity must be < max_seq (a real ring)");
            }
        }
        // Layers 0 and 4 are full; 1,2,3,5,6,7 sliding.
        assert_eq!(cfg.layer_kv_capacity(0, max_seq), max_seq);
        assert_eq!(cfg.layer_kv_capacity(4, max_seq), max_seq);
        assert_eq!(cfg.layer_kv_capacity(1, max_seq), 4);
        // Clamp: a context shorter than the window never over-allocates.
        assert_eq!(cfg.layer_kv_capacity(1, 3), 3);
    }

    // ─── Real 2-layer CPU stage oracle (mirrors nemotron_cpu_stage_oracle) ───
    //
    // Needs shards 1+2 (~11GB) + the checkpoint dir. Writes per-stage hidden as
    // raw LE-f32 `{VLLM_LAGUNA_ORACLE_OUT}.{stage}.r0` for scripts/diff_f32.py.
    //
    //   VLLM_TEST_LAGUNA_DIR=/path/to/Laguna-S-2.1-NVFP4 \
    //   VLLM_LAGUNA_ORACLE_LAYERS=2 VLLM_LAGUNA_ORACLE_OUT=/tmp/laguna_cpu \
    //   cargo test --lib -- --ignored --nocapture laguna_cpu_stage_oracle
    #[test]
    #[ignore]
    fn laguna_cpu_stage_oracle() {
        let dir = match std::env::var("VLLM_TEST_LAGUNA_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("VLLM_TEST_LAGUNA_DIR unset — skipping Laguna CPU stage oracle");
                return;
            }
        };
        let num_layers: usize = std::env::var("VLLM_LAGUNA_ORACLE_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let out_base =
            std::env::var("VLLM_LAGUNA_ORACLE_OUT").unwrap_or_else(|_| "/tmp/laguna_cpu".into());
        // Multi-token prompt so RoPE/YaRN + causal (+sliding) attention are
        // actually exercised (a single token at pos 0 makes RoPE the identity
        // and attention trivial). Comma-separated ids; defaults to 8 tokens.
        let tokens: Vec<u32> = std::env::var("VLLM_LAGUNA_ORACLE_TOKENS")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1u32, 2, 3, 5, 8, 13, 21, 34]);

        let d = std::path::Path::new(&dir);
        let cfg_json: Value =
            serde_json::from_str(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
        let cfg = LagunaConfig::from_json(&cfg_json).expect("LagunaConfig");
        eprintln!(
            "Laguna CPU oracle: {num_layers} layers, {} tokens={tokens:?}, hidden={}",
            tokens.len(), cfg.hidden_size
        );

        let t0 = std::time::Instant::now();
        let weights = load_laguna_weights_cpu(d, &cfg, num_layers, false)
            .expect("load_laguna_weights_cpu");
        eprintln!("loaded {num_layers} layers in {:.1}s", t0.elapsed().as_secs_f64());

        let hidden = weights.forward_with_stage_dump(&tokens, &cfg, Some(&out_base));
        assert_eq!(hidden.len(), tokens.len() * cfg.hidden_size);
        assert!(hidden.iter().all(|v| v.is_finite()), "non-finite oracle output");
        let l2 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("final hidden l2={l2:.4} first8={:?}", &hidden[..8]);
        eprintln!(
            "stage dumps: {out_base}.embed.r0 / .layer0.r0 / ... / .final_norm.r0\n  \
             diff vs HF STAGE_HIDDEN_DUMP with scripts/diff_f32.py"
        );
    }

    // ─── Deliverable-B gate: the lib.rs product `LagunaModel::load_cpu` MUST
    //     reproduce the free-function `load_laguna_weights_cpu` forward
    //     bit-for-bit (same load path the `mt=="laguna"` dispatch invokes). ──
    //
    //   VLLM_TEST_LAGUNA_DIR=/path/to/Laguna-S-2.1-NVFP4 \
    //   VLLM_LAGUNA_ORACLE_LAYERS=2 \
    //   cargo test --lib -- --ignored --nocapture laguna_model_load_cpu_bit_exact
    #[test]
    #[ignore]
    fn laguna_model_load_cpu_bit_exact() {
        let dir = match std::env::var("VLLM_TEST_LAGUNA_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("VLLM_TEST_LAGUNA_DIR unset — skipping LagunaModel::load_cpu gate");
                return;
            }
        };
        let num_layers: usize = std::env::var("VLLM_LAGUNA_ORACLE_LAYERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let tokens: Vec<u32> = std::env::var("VLLM_LAGUNA_ORACLE_TOKENS")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1u32, 2, 3, 5, 8, 13, 21, 34]);

        let d = std::path::Path::new(&dir);
        let cfg_json: Value =
            serde_json::from_str(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
        let cfg = LagunaConfig::from_json(&cfg_json).expect("LagunaConfig");

        // Free-function reference (the A-validated path).
        let ref_w = load_laguna_weights_cpu(d, &cfg, num_layers, false)
            .expect("load_laguna_weights_cpu");
        let ref_out = ref_w.forward(&tokens, &cfg);

        // The lib.rs product: LagunaModel::load_cpu([0,num_layers), keep_lm=false).
        let model = LagunaModel::load_cpu(d, cfg.clone(), 0, num_layers, false)
            .expect("LagunaModel::load_cpu");
        assert_eq!(model.pp_start, 0);
        assert_eq!(model.pp_end, num_layers);
        assert!(model.pp_first);
        assert_eq!(model.pp_last, num_layers >= cfg.num_hidden_layers);
        assert_eq!(model.weights.layers.len(), num_layers);
        let out = model.forward(&tokens);

        assert_eq!(out.len(), ref_out.len());
        // Same code, same bytes → require BIT-identical (not just cos).
        for (i, (&a, &b)) in out.iter().zip(&ref_out).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(),
                "LagunaModel::load_cpu diverged from load_laguna_weights_cpu at {i}: {a} vs {b}");
        }
        let l2 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("LagunaModel::load_cpu bit-exact vs free-fn: {} elems, l2={l2:.4}", out.len());
    }
}

#[cfg(test)]
mod shader_guard {
    //! Registry guard for the Laguna-specific compute kernels.
    //!
    //! `scripts/compile_shaders.sh` SKIPS a `compile` entry whose `.comp`
    //! source is absent, rather than failing. That is deliberate — one script
    //! drives every per-feature slice, compiling the shaders present in the
    //! tree and ignoring the rest — but it means a renamed file, a typo'd
    //! compile entry, or a bad carve no longer breaks the build: the entry
    //! compiles nothing, the script exits 0, and the kernel simply vanishes
    //! from the registry.
    //!
    //! A count or self-consistency check cannot catch that: when an entry
    //! disappears, the generated registry and the runtime shader map shrink
    //! together and stay 1:1. So this test NAMES the kernels instead of
    //! counting them — adding a shader never breaks it, losing one always
    //! does. Every name below is dispatched by name from this model's GPU
    //! path, so its absence would be a runtime failure on device, which CI
    //! has no way to reach.

    /// Laguna (GPU decode-SDPA + MoE tail) kernels this model owns.
    const REQUIRED_LAGUNA_KERNELS: &[&str] = &[
        // resident decode SDPA + its softplus output gate
        "laguna_gpu_sdpa",
        "laguna_softplus_gate",
        // MoE router + routed-expert weighted accumulate (plain / batched)
        "laguna_router",
        "laguna_moe_accum",
        "laguna_moe_accum_b",
        // expert-batched e4m3 matvec (Laguna MoE CB-batch lever)
        "mul_mat_vec_laguna_expb_e4m3_f32_f32",
    ];

    #[test]
    fn laguna_kernels_are_registered() {
        let map = crate::include_all_shaders();
        let missing: Vec<&str> = REQUIRED_LAGUNA_KERNELS
            .iter()
            .copied()
            .filter(|n| !map.contains_key(*n))
            .collect();
        assert!(
            missing.is_empty(),
            "{} Laguna shader(s) missing from the registry: {:?}\n\
             The SPIR-V for these was not produced, so any dispatch of them would \
             fail on device. Check that the .comp source exists under shaders/ and \
             that scripts/compile_shaders.sh still has a compile entry for it \
             (a missing source is SKIPPED, not an error, by design).",
            missing.len(),
            missing,
        );
    }

    /// The SPIR-V behind each required kernel must be non-empty and well-formed
    /// enough to be a SPIR-V module: correct magic number and a 5-word header.
    /// Catches a truncated or empty `.spv` surviving into the registry.
    #[test]
    fn laguna_kernel_spirv_is_wellformed() {
        let map = crate::include_all_shaders();
        for name in REQUIRED_LAGUNA_KERNELS {
            let Some(bytes) = map.get(*name) else { continue };
            assert!(
                bytes.len() >= 20 && bytes.len() % 4 == 0,
                "{name}: SPIR-V is {} bytes — too short or not word-aligned",
                bytes.len()
            );
            let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            assert_eq!(
                magic, 0x0723_0203,
                "{name}: bad SPIR-V magic 0x{magic:08x} (expected 0x07230203)"
            );
        }
    }
}
