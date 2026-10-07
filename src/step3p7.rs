//! Step-3.7-Flash-148B (`0xSero/Step-3.7-Flash-148B`) — model type `step3p7`.
//!
//! A REAP-pruned (212/288 experts kept) derivative of Step-3.7-Flash-198B: a
//! sigmoid-router MoE that reuses ~85% of the qwen3_5 MoE stack. This module holds
//! the CPU reference for the *genuinely-new* ops, each gated bit-exact against the
//! repo's own `modeling_step3p7.py` (third-party reference), NOT a self-derived one.
//!
//! Novel / divergent-from-qwen35 pieces validated here:
//!  1. RMSNorm is **GEMMA-style `weight + 1`** (every norm) — NOT qwen35's weight-only.
//!  2. Head-wise attention gate: per-head SCALAR `sigmoid(g_proj(x))` broadcast over
//!     head_dim, applied to attn_out before o_proj (qwen35 gates per-element via a
//!     double-width q_proj; step3p7 has a separate `g_proj: hidden->num_heads`).
//!  3. Bias-corrected sigmoid router (DeepSeek-V3 style): select top-k on
//!     `sigmoid(logits)+router_bias`, weight from UN-biased sigmoid, renorm, `*3.0`.
//!  4. Clamped SwiGLU on layers 43,44 ONLY (`swiglu_limit` 7 experts / 16 shared).
//!  5. Partial RoPE with per-layer theta + llama3 inv_freq scaling; dual per-layer
//!     head count (64 full-attn / 96 sliding); shared expert ADDED un-gated.
//!
//! Quant layout (from the safetensors dtype audit): routed experts are 3D-stacked
//! modelopt NVFP4 (`moe.{gate,up,down}_proj.weight` U8 + `weight_scale` F8_E4M3 g16 +
//! `weight_scale_2` F32 global; `input_scale` ignored, W4A16). Everything else BF16.

use crate::model::{cpu_matmul, cpu_sdpa, dequantize_nvfp4};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnKind {
    Full,
    Sliding,
}

/// Per-decoder-layer attention parameters (they differ by layer type on this model).
#[derive(Clone, Copy, Debug)]
pub struct LayerAttn {
    pub kind: AttnKind,
    pub num_heads: usize,       // 64 for Full, 96 for Sliding
    pub rope_theta: f32,        // 5e6 Full, 1e4 Sliding
    pub partial_rotary: f32,    // 0.5 Full, 1.0 Sliding
    pub use_llama3_scale: bool, // yarn/llama3 applied on this layer's inv_freq
    pub sliding_window: Option<usize>, // Some(512) for Sliding
}

#[derive(Clone, Copy, Debug)]
pub struct Llama3Rope {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_pos: f32,
}

#[derive(Clone, Debug)]
pub struct Step3p7Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize, // 45 (checkpoint has 48 incl 3 MTP heads we drop)
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub head_dim: usize,
    pub num_key_value_heads: usize, // 8 (both layer types)

    // Dense MLP (layers 0..dense_layers)
    pub intermediate_size: usize, // 11264

    // MoE
    pub num_experts: usize,          // 212 (post-REAP)
    pub num_experts_per_tok: usize,  // 8
    pub moe_intermediate_size: usize,// 1280
    pub share_expert_dim: usize,     // 1280
    pub router_scaling_factor: f32,  // 3.0
    pub use_router_bias: bool,       // true
    pub moe_first_layer: usize,      // 3 (layers 0..3 are dense)

    pub layers: Vec<LayerAttn>,      // len == num_hidden_layers
    pub swiglu_limit_expert: Vec<f32>, // per layer, 0 == none
    pub swiglu_limit_shared: Vec<f32>,
    pub llama3: Llama3Rope,
    pub tie_word_embeddings: bool,
}

impl Step3p7Config {
    /// Parse the nested `text_config` of the real step3p7 config.json.
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let tc = root
            .get("text_config")
            .ok_or("step3p7: missing text_config")?;
        let g = |k: &str| tc.get(k);
        let usz = |k: &str| -> Result<usize, String> {
            g(k).and_then(|v| v.as_u64()).map(|x| x as usize).ok_or(format!("step3p7: missing usize {k}"))
        };
        let f = |k: &str, d: f32| g(k).and_then(|v| v.as_f64()).map(|x| x as f32).unwrap_or(d);

        let num_hidden_layers = usz("num_hidden_layers")?;
        let head_dim = usz("head_dim")?;
        let full_heads = usz("num_attention_heads")?; // 64
        // sliding layers override heads via attention_other_setting.num_attention_heads (96)
        let sliding_heads = tc
            .get("attention_other_setting")
            .and_then(|a| a.get("num_attention_heads"))
            .and_then(|v| v.as_u64())
            .map(|x| x as usize)
            .unwrap_or(full_heads);
        let sliding_window = usz("sliding_window").unwrap_or(512);

        let layer_types: Vec<String> = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .ok_or("step3p7: missing layer_types")?
            .iter()
            .map(|v| v.as_str().unwrap_or("full_attention").to_string())
            .collect();
        let prfs: Vec<f32> = tc
            .get("partial_rotary_factors")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(1.0) as f32).collect())
            .unwrap_or_else(|| vec![1.0; layer_types.len()]);
        let thetas: Vec<f32> = tc
            .get("rope_theta")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(10000.0) as f32).collect())
            .unwrap_or_else(|| vec![f("rope_theta", 10000.0); layer_types.len()]);
        let yarn_only: Vec<String> = tc
            .get("yarn_only_types")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            let full = layer_types.get(i).map(|s| s == "full_attention").unwrap_or(true);
            let kind = if full { AttnKind::Full } else { AttnKind::Sliding };
            // GROUND-TRUTH (modeling_step3p7.py Step3p7Attention.__init__, lines
            // 648-653): llama3 inv_freq scaling is GATED on `yarn_only_types`. For a
            // layer whose type is NOT in yarn_only_types, `config.rope_parameters` is
            // set to None, which makes Step3p7RotaryEmbedding fall back to rope_type
            // "default" (compute_default_rope_parameters — NO scaling). yarn_only_types
            // == ["full_attention"], so ONLY full_attention layers get llama3-scaled
            // inv_freq; sliding_attention layers use PLAIN NeoX RoPE with their per-layer
            // theta. (The earlier "scale every layer" reading was wrong; verified against
            // the real modeling file.)
            let ltype = layer_types.get(i).map(String::as_str).unwrap_or("full_attention");
            let use_llama3_scale = if yarn_only.is_empty() {
                // No gating list → the modeling default keeps rope_parameters for all.
                true
            } else {
                yarn_only.iter().any(|t| t == ltype)
            };
            layers.push(LayerAttn {
                kind,
                num_heads: if full { full_heads } else { sliding_heads },
                rope_theta: thetas.get(i).copied().unwrap_or(10000.0),
                partial_rotary: prfs.get(i).copied().unwrap_or(1.0),
                use_llama3_scale,
                sliding_window: if full { None } else { Some(sliding_window) },
            });
        }

        let read_limits = |k: &str| -> Vec<f32> {
            tc.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                .unwrap_or_else(|| vec![0.0; num_hidden_layers])
        };

        let rp = tc.get("rope_parameters").or_else(|| tc.get("rope_scaling"));
        let llama3 = Llama3Rope {
            factor: rp.and_then(|v| v.get("factor")).and_then(|v| v.as_f64()).unwrap_or(2.0) as f32,
            low_freq_factor: rp.and_then(|v| v.get("low_freq_factor")).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            high_freq_factor: rp.and_then(|v| v.get("high_freq_factor")).and_then(|v| v.as_f64()).unwrap_or(32.0) as f32,
            original_max_pos: rp
                .and_then(|v| v.get("original_max_position_embeddings"))
                .and_then(|v| v.as_f64())
                .unwrap_or(131072.0) as f32,
        };

        Ok(Step3p7Config {
            hidden_size: usz("hidden_size")?,
            num_hidden_layers,
            vocab_size: usz("vocab_size")?,
            rms_norm_eps: f("rms_norm_eps", 1e-5),
            head_dim,
            num_key_value_heads: usz("num_attention_groups").or_else(|_| usz("num_key_value_heads"))?,
            intermediate_size: usz("intermediate_size")?,
            num_experts: usz("moe_num_experts")?,
            num_experts_per_tok: usz("moe_top_k")?,
            moe_intermediate_size: usz("moe_intermediate_size")?,
            share_expert_dim: usz("share_expert_dim")?,
            router_scaling_factor: f("moe_router_scaling_factor", 1.0),
            use_router_bias: tc.get("use_moe_router_bias").and_then(|v| v.as_bool()).unwrap_or(false),
            moe_first_layer: tc
                .get("moe_layers_enum")
                .and_then(|v| v.as_str())
                .and_then(|s| s.split(',').next())
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(3),
            layers,
            swiglu_limit_expert: read_limits("swiglu_limits"),
            swiglu_limit_shared: read_limits("swiglu_limits_shared"),
            llama3,
            tie_word_embeddings: root.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    }

    #[inline]
    pub fn is_moe_layer(&self, l: usize) -> bool {
        l >= self.moe_first_layer && l < self.num_hidden_layers
    }
}

// ============================ novel-op CPU reference ============================

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// GEMMA-style RMSNorm: `x/rms(x) * (weight + 1)`. (f64 accumulation, matches ref.)
pub fn rms_norm_plus1(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let inv = 1.0 / ((ss / n as f64) + eps as f64).sqrt();
    (0..n)
        .map(|i| ((x[i] as f64) * inv * (weight[i] as f64 + 1.0)) as f32)
        .collect()
}

/// Head-wise attention gate. `attn_out` is [num_heads*head_dim] for one token,
/// `gate_logits` is [num_heads] = g_proj(x). Returns gated [num_heads*head_dim].
pub fn head_gate(attn_out: &[f32], gate_logits: &[f32], num_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let g = sigmoid(gate_logits[h]);
        for d in 0..head_dim {
            out[h * head_dim + d] = attn_out[h * head_dim + d] * g;
        }
    }
    out
}

/// Clamped SwiGLU per-element product term. `gate_lin`/`up_lin` are the pre-activation
/// linear outputs [inter]. `limit==None` (or 0) → plain SiLU-SwiGLU. Returns silu(gate)*up.
pub fn clamped_swiglu_prod(gate_lin: &[f32], up_lin: &[f32], limit: Option<f32>) -> Vec<f32> {
    let inter = gate_lin.len();
    let mut out = vec![0.0f32; inter];
    for i in 0..inter {
        let mut g = silu(gate_lin[i]);
        let mut u = up_lin[i];
        if let Some(l) = limit {
            if g > l {
                g = l;
            }
            u = u.clamp(-l, l);
        }
        out[i] = g * u;
    }
    out
}

/// Bias-corrected sigmoid router (DeepSeek-V3 style). Returns (indices, weights),
/// each length top_k, ordered by biased-prob descending (matches torch.topk order).
pub fn bias_router(
    logits: &[f32],
    router_bias: &[f32],
    top_k: usize,
    scaling: f32,
) -> (Vec<usize>, Vec<f32>) {
    let e = logits.len();
    // fp32 sigmoid probs (unbiased) — need_fp32_gate handled by caller (logits already fp32).
    let p: Vec<f32> = logits.iter().map(|&z| sigmoid(z)).collect();
    // selection by (p + bias) descending
    let mut order: Vec<usize> = (0..e).collect();
    order.sort_by(|&a, &b| {
        let ka = p[a] + router_bias.get(a).copied().unwrap_or(0.0);
        let kb = p[b] + router_bias.get(b).copied().unwrap_or(0.0);
        kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
    });
    let idx: Vec<usize> = order[..top_k].to_vec();
    // weights from UN-biased p, renormalized, then scaled
    let mut w: Vec<f32> = idx.iter().map(|&i| p[i]).collect();
    let s: f32 = w.iter().sum::<f32>() + 1e-20;
    for wi in w.iter_mut() {
        *wi = *wi / s * scaling;
    }
    (idx, w)
}

/// llama3 inv_freq scaling (verbatim transformers `_compute_llama3_parameters`).
fn llama3_scale(inv_freq: &mut [f64], r: &Llama3Rope) {
    let (factor, lf, hf, octx) = (
        r.factor as f64,
        r.low_freq_factor as f64,
        r.high_freq_factor as f64,
        r.original_max_pos as f64,
    );
    let low_wl = octx / lf;
    let high_wl = octx / hf;
    for f in inv_freq.iter_mut() {
        let wl = 2.0 * std::f64::consts::PI / *f;
        let mut il = if wl > low_wl { *f / factor } else { *f };
        let smooth = (octx / wl - lf) / (hf - lf);
        let smoothed = (1.0 - smooth) * il / factor + smooth * il;
        let is_medium = !(wl < high_wl) && !(wl > low_wl);
        if is_medium {
            il = smoothed;
        }
        *f = il;
    }
}

/// Partial NeoX RoPE for one head vector `v` [head_dim] at absolute `pos`.
/// Rotates the first `int(head_dim*partial_rotary)` dims (paired i, i+half);
/// passes the rest through. Applies llama3 inv_freq scaling when requested.
pub fn partial_rope(v: &[f32], pos: usize, la: &LayerAttn, head_dim: usize, l3: &Llama3Rope) -> Vec<f32> {
    let dim = ((head_dim as f32) * la.partial_rotary) as usize;
    let half = dim / 2;
    let mut inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / (la.rope_theta as f64).powf((2 * i) as f64 / dim as f64))
        .collect();
    if la.use_llama3_scale {
        llama3_scale(&mut inv, l3);
    }
    let mut out = v.to_vec();
    for i in 0..half {
        let ang = pos as f64 * inv[i];
        let (c, s) = (ang.cos(), ang.sin());
        let x1 = v[i] as f64;
        let x2 = v[half + i] as f64;
        out[i] = (x1 * c - x2 * s) as f32;
        out[half + i] = (x2 * c + x1 * s) as f32;
    }
    out
}

/// Decode a single e2m1 fp4 code (0..7 magnitude index) — NVFP4 magnitude table.
pub const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Loader destination + target resident precision for a checkpoint tensor.
/// This is the "brain" of `load_step3p7_weights`: it routes every tensor name to
/// a bucket without needing the 95GB payload, so it is unit-testable on Mac.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRole {
    /// Skip entirely: vision tower / projector / MTP heads (layers 45-47).
    Drop,
    /// 3D-stacked NVFP4 routed-expert weight (U8 packed) — upload packed-resident.
    ExpertNvfp4Weight,
    /// NVFP4 companion scale (`weight_scale` e4m3 g16 / `weight_scale_2` f32 global).
    ExpertNvfp4Scale,
    /// NVFP4 activation `input_scale` — IGNORED (W4A16).
    IgnoreInputScale,
    /// BF16 weight to requant q8_0 resident (attn, shared expert, moe.gate, dense L0-2).
    Bf16ToQ8,
    /// FP8-KV cache scalar (`k_scale`/`v_scale`) — kept as f32 metadata.
    KvScale,
    /// Router bias (f32, keep).
    RouterBias,
    /// Token embedding — keep f16 (charge to stage 0).
    EmbedF16,
    /// Output projection — keep f16 (charge to last stage; may be tied to embed).
    LmHeadF16,
    /// Final norm / layernorm weights — keep (small, +1 RMSNorm at use).
    NormF16,
}

/// Classify a checkpoint tensor name into its loader role. Names are the real
/// `model.language_model.*` / top-level layout from the safetensors index audit.
pub fn classify_tensor(name: &str) -> TensorRole {
    // Drops first (order matters).
    if name.contains("vision_model")
        || name.contains("vit_large_projector")
        || name.contains("projector")
    {
        return TensorRole::Drop;
    }
    // MTP heads live at decoder indices 45/46/47 (and in the separate mtp file we never open).
    for l in [45usize, 46, 47] {
        if name.contains(&format!(".layers.{l}.")) {
            return TensorRole::Drop;
        }
    }
    if name == "lm_head.weight" {
        return TensorRole::LmHeadF16;
    }
    if name.ends_with("embed_tokens.weight") {
        return TensorRole::EmbedF16;
    }
    // MoE expert 3D-stacked tensors.
    let is_expert = name.contains(".moe.")
        && (name.contains(".gate_proj.")
            || name.contains(".up_proj.")
            || name.contains(".down_proj."));
    if is_expert {
        if name.ends_with(".input_scale") {
            return TensorRole::IgnoreInputScale;
        }
        if name.ends_with(".weight_scale") || name.ends_with(".weight_scale_2") {
            return TensorRole::ExpertNvfp4Scale;
        }
        if name.ends_with(".weight") {
            return TensorRole::ExpertNvfp4Weight;
        }
    }
    if name.ends_with(".k_scale") || name.ends_with(".v_scale") {
        return TensorRole::KvScale;
    }
    if name.ends_with(".router_bias") {
        return TensorRole::RouterBias;
    }
    if name.ends_with("norm.weight") || name.contains("layernorm") {
        return TensorRole::NormF16;
    }
    // Everything else with a weight is a bf16 matmul weight → requant q8_0:
    // attn q/k/v/o/g_proj, moe.gate (router), share_expert.*, dense mlp L0-2.
    TensorRole::Bf16ToQ8
}

// ══════════════════════ owned host weights + CPU forward ══════════════════════
//
// The CPU reference forward — the Step-3.7 analog of `laguna::LagunaModel` /
// `nemotron::NemotronModel`. All BF16 matmul weights are dequantized to f32 on
// load; the routed experts are kept PACKED (3D-stacked NVFP4 nibbles + F8_E4M3
// group-16 scales + F32 per-expert global) and only the top-8 selected experts
// per token are dequantized on demand via the crate-wide pure fn
// `crate::model::dequantize_nvfp4`. Every op is one of the bit-exact-validated
// novel ops above (`rms_norm_plus1`, `head_gate`, `clamped_swiglu_prod`,
// `bias_router`, `partial_rope`) + the shared `cpu_matmul`/`cpu_sdpa`.

/// Attention projections + per-head q/k norms for one layer (BF16→f32).
pub struct OwnedAttn {
    /// [num_heads*head_dim, hidden]
    pub q_proj: Vec<f32>,
    /// [num_kv_heads*head_dim, hidden]
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    /// [hidden, num_heads*head_dim]
    pub o_proj: Vec<f32>,
    /// per-head scalar gate proj: [num_heads, hidden]
    pub g_proj: Vec<f32>,
    /// [head_dim]
    pub q_norm: Vec<f32>,
    /// [head_dim]
    pub k_norm: Vec<f32>,
}

/// Dense (layers 0..moe_first_layer) SwiGLU MLP. All BF16→f32.
pub struct OwnedDense {
    /// [intermediate, hidden]
    pub gate: Vec<f32>,
    /// [intermediate, hidden]
    pub up: Vec<f32>,
    /// [hidden, intermediate]
    pub down: Vec<f32>,
}

/// Overflow-stream harness config, read once at load. Default-OFF (dev-only):
/// `VLLM_VULKAN_MOE_STREAM_OVERFLOW=1` enables it, `VLLM_VULKAN_MOE_RESIDENT_BUDGET_GB`
/// sets the per-stage routed-expert resident budget in GB — experts loaded past the
/// budget become NAS-streamed (fetched + evicted on demand). When disabled the loader
/// is byte-identical to the all-resident path. Mirrors the Ling overflow-stream harness
/// (`src/ling.rs`) so the SAME env vars drive both arches. This is what lets Step-3.7
/// — whose PP-6 all-resident expert footprint EXCEEDS a BC-250 node's physical DRAM
/// (~1.88 GiB/MoE-layer × ~7 layers/stage ≈ 13 GiB) — fit and run its whole-model
/// forward on today's 5-6 nodes, decoupled from the N≥10 cluster.
#[derive(Clone, Copy)]
pub struct MoeStreamCfg {
    pub enabled: bool,
    pub budget_bytes: u64,
}
impl MoeStreamCfg {
    pub fn from_env() -> Self {
        let truthy = |v: &str| matches!(v, "1" | "true" | "TRUE" | "yes" | "on");
        let enabled = std::env::var("VLLM_VULKAN_MOE_STREAM_OVERFLOW")
            .map(|v| truthy(&v))
            .unwrap_or(false);
        // Default 4.0 GB routed-expert resident budget: leaves headroom under the
        // ~13 GB GTT floor for attn/dense/shared f32 + edge embed/lm_head (~2 GB) +
        // activations. Only consulted when `enabled`.
        let gb = std::env::var("VLLM_VULKAN_MOE_RESIDENT_BUDGET_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(4.0);
        MoeStreamCfg { enabled, budget_bytes: (gb * 1e9) as u64 }
    }
    /// Resident bytes of one NVFP4 expert linear `[out,in]`: packed `out*in/2` u8
    /// (4-bit nibbles) + scale `out*in/group` u8 (F8_E4M3 block scales). The one-f32
    /// global (`weight_scale_2`) is always resident and not counted.
    #[inline]
    pub fn expert_linear_bytes(out: usize, inn: usize, group: usize) -> u64 {
        (out * inn / 2 + out * inn / group) as u64
    }
}

/// Backing store for one routed-expert NVFP4 linear: either the packed nibbles +
/// e4m3 group scales held **resident** in DRAM, or a **non-resident** NAS descriptor
/// (shard path + 3D tensor names + expert index) streamed on demand and evicted.
pub enum ExpertStore {
    Resident { packed: Vec<u8>, scale: Vec<u8> },
    // packed and scale can live in DIFFERENT shards (they straddle shard boundaries in
    // the real checkpoint), so each carries its own shard path.
    Streamed {
        packed_path: String,
        scale_path: String,
        packed_name: String,
        scale_name: String,
        expert: usize,
    },
}

/// One routed-expert linear (modelopt NVFP4: 4-bit nibbles + F8_E4M3 group-16 block
/// scales + one F32 global), resident **or** overflow-streamed. `dequant()` produces
/// the identical `[out,in]` f32 either way — a streamed expert reuses the exact
/// per-expert byte slice + `dequantize_nvfp4` on the exact same on-disk bytes, so it
/// is **bit-identical to the resident expert BY CONSTRUCTION**; only WHEN/WHERE the
/// buffer is allocated differs.
pub struct Step3p7Expert {
    pub store: ExpertStore,
    /// modelopt `weight_scale_2[e]` — the global we MULTIPLY by (amax/(448*6)); NOT
    /// the compressed-tensors reciprocal (P3 dequant-direction gate confirmed).
    pub global: f32,
    pub out_f: usize,
    pub in_f: usize,
    pub group_size: usize,
}
impl Step3p7Expert {
    /// Per-expert `(packed, scale)` byte spans within a 3D `[E, out, in..]` tensor.
    #[inline]
    fn byte_spans(out_f: usize, in_f: usize, group: usize) -> (usize, usize) {
        (out_f * in_f / 2, out_f * (in_f / group))
    }
    pub fn resident_bytes(&self) -> usize {
        match &self.store {
            ExpertStore::Resident { packed, scale } => packed.len() + scale.len(),
            ExpertStore::Streamed { .. } => 0,
        }
    }
    pub fn is_streamed(&self) -> bool {
        matches!(self.store, ExpertStore::Streamed { .. })
    }

    /// Read a streamed expert's `(packed, scale)` slices from its shard on demand:
    /// mmap + header-parse the 3D `.weight`/`.weight_scale` tensors, take expert
    /// `e`'s contiguous byte range. Pure function of the on-disk bytes ⟹ bit-identical
    /// to the resident slice. Fails LOUD (dev-only harness).
    #[allow(clippy::too_many_arguments)]
    fn read_streamed(
        packed_path: &str,
        scale_path: &str,
        packed_name: &str,
        scale_name: &str,
        e: usize,
        out_f: usize,
        in_f: usize,
        group: usize,
    ) -> Result<(Vec<u8>, Vec<u8>), String> {
        use memmap2::Mmap;
        use safetensors::SafeTensors;
        let (pb, sb) = Self::byte_spans(out_f, in_f, group);
        let fp = std::fs::File::open(packed_path).map_err(|x| format!("stream open {packed_path}: {x}"))?;
        let mp = unsafe { Mmap::map(&fp) }.map_err(|x| format!("stream mmap {packed_path}: {x}"))?;
        let stp = SafeTensors::deserialize(&mp).map_err(|x| format!("stream deser {packed_path}: {x}"))?;
        let packed = stp.tensor(packed_name).map_err(|x| format!("{packed_name}: {x}"))?
            .data()[e * pb..(e + 1) * pb].to_vec();
        let fs = std::fs::File::open(scale_path).map_err(|x| format!("stream open {scale_path}: {x}"))?;
        let ms = unsafe { Mmap::map(&fs) }.map_err(|x| format!("stream mmap {scale_path}: {x}"))?;
        let sts = SafeTensors::deserialize(&ms).map_err(|x| format!("stream deser {scale_path}: {x}"))?;
        let scale = sts.tensor(scale_name).map_err(|x| format!("{scale_name}: {x}"))?
            .data()[e * sb..(e + 1) * sb].to_vec();
        Ok((packed, scale))
    }

    /// Dequantize to f32 `[out_f, in_f]`. Streamed experts fetch-then-evict (the
    /// transient buffers drop at end of call → peak DRAM = one expert, not the
    /// stage's whole overflow set).
    pub fn dequant(&self) -> Vec<f32> {
        match &self.store {
            ExpertStore::Resident { packed, scale } => {
                dequantize_nvfp4(packed, scale, self.global, self.out_f, self.in_f, self.group_size)
            }
            ExpertStore::Streamed { packed_path, scale_path, packed_name, scale_name, expert } => {
                let (packed, scale) = Self::read_streamed(
                    packed_path, scale_path, packed_name, scale_name, *expert,
                    self.out_f, self.in_f, self.group_size,
                )
                .unwrap_or_else(|e| panic!("MOE_STREAM_OVERFLOW: {e}"));
                dequantize_nvfp4(&packed, &scale, self.global, self.out_f, self.in_f, self.group_size)
                // packed/scale drop here → evicted, peak bounded.
            }
        }
    }
}

/// NVFP4 3D-stacked routed experts for one MoE layer, now a Vec of per-expert linears
/// (each resident OR overflow-streamed). Step-3.7 stores all experts in ONE 3D modelopt
/// tensor per proj (`moe.{gate,up,down}_proj.weight` U8 `[E,out,in/2]`, `.weight_scale`
/// F8_E4M3 `[E,out,in/16]`, `.weight_scale_2` F32 `[E]`); the blobs are expert-contiguous
/// row-major so each expert is a clean byte slice. `dequant(e, proj)` keeps the SAME
/// interface the MoE forward uses — only the backing store changed.
pub struct OwnedExpertsPacked {
    pub gate: Vec<Step3p7Expert>, // out=inter, in=hidden
    pub up: Vec<Step3p7Expert>,   // out=inter, in=hidden
    pub down: Vec<Step3p7Expert>, // out=hidden, in=inter
    pub num_experts: usize,
    /// gate/up out_features == moe_intermediate_size; down out_features == hidden.
    pub inter: usize,
    pub hidden: usize,
    pub group_size: usize,
}

impl OwnedExpertsPacked {
    /// Dequantize one expert's `gate`/`up`/`down` (proj 0/1/2) to f32 `[out,in]`.
    fn dequant(&self, e: usize, proj: u8) -> Vec<f32> {
        match proj {
            0 => self.gate[e].dequant(),
            1 => self.up[e].dequant(),
            2 => self.down[e].dequant(),
            _ => unreachable!("proj must be 0/1/2"),
        }
    }
    /// Total resident DRAM charged by this layer's experts (streamed → 0).
    pub fn resident_bytes(&self) -> usize {
        self.gate.iter().chain(&self.up).chain(&self.down).map(|x| x.resident_bytes()).sum()
    }
    /// How many of the 3·E expert linears are overflow-streamed.
    pub fn streamed_count(&self) -> usize {
        self.gate.iter().chain(&self.up).chain(&self.down).filter(|x| x.is_streamed()).count()
    }
}

/// Load one MoE proj's `[E]` experts (resident or overflow-streamed per the budget)
/// by slicing the 3D `{base}.weight`/`.weight_scale` tensors per expert. Keeps a live
/// mmap cache so the big packed tensor is never fully copied to the heap — resident
/// experts copy only their own slice; streamed experts copy nothing (descriptor only)
/// ⟹ peak load DRAM stays ≤ the resident budget.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_experts_proj(
    dir: &std::path::Path,
    weight_map: &serde_json::Map<String, Value>,
    mmaps: &mut std::collections::HashMap<String, memmap2::Mmap>,
    base: &str,
    globals: &[f32],
    num_experts: usize,
    out_f: usize,
    in_f: usize,
    group: usize,
    stream_cfg: MoeStreamCfg,
    resident_bytes: &mut u64,
) -> Result<Vec<Step3p7Expert>, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    let packed_name = format!("{base}.weight");
    let scale_name = format!("{base}.weight_scale");
    // packed and scale can live in DIFFERENT shards — resolve each independently.
    let resolve = |name: &str| -> Result<(String, String), String> {
        let shard = weight_map
            .get(name)
            .and_then(|x| x.as_str())
            .ok_or_else(|| format!("index.json missing {name}"))?
            .to_string();
        let path = dir.join(&shard).to_string_lossy().to_string();
        Ok((shard, path))
    };
    let (packed_shard, packed_path) = resolve(&packed_name)?;
    let (scale_shard, scale_path) = resolve(&scale_name)?;
    for (shard, path) in [(&packed_shard, &packed_path), (&scale_shard, &scale_path)] {
        if !mmaps.contains_key(shard) {
            let f = std::fs::File::open(path).map_err(|e| format!("open {shard}: {e}"))?;
            let m = unsafe { Mmap::map(&f).map_err(|e| format!("mmap {shard}: {e}"))? };
            mmaps.insert(shard.clone(), m);
        }
    }
    let stp = SafeTensors::deserialize(&mmaps[&packed_shard]).map_err(|e| format!("parse {packed_shard}: {e}"))?;
    let sts = SafeTensors::deserialize(&mmaps[&scale_shard]).map_err(|e| format!("parse {scale_shard}: {e}"))?;
    let pv = stp.tensor(&packed_name).map_err(|e| format!("{packed_name}: {e}"))?;
    let sv = sts.tensor(&scale_name).map_err(|e| format!("{scale_name}: {e}"))?;
    let pd = pv.data();
    let sd = sv.data();
    let (pb, sb) = Step3p7Expert::byte_spans(out_f, in_f, group);
    if globals.len() < num_experts {
        return Err(format!("{base}.weight_scale_2 has {} globals < {num_experts}", globals.len()));
    }
    let est = MoeStreamCfg::expert_linear_bytes(out_f, in_f, group);
    let mut out = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let make_resident =
            !stream_cfg.enabled || (*resident_bytes + est <= stream_cfg.budget_bytes);
        let store = if make_resident {
            *resident_bytes += est;
            ExpertStore::Resident {
                packed: pd[e * pb..(e + 1) * pb].to_vec(),
                scale: sd[e * sb..(e + 1) * sb].to_vec(),
            }
        } else {
            ExpertStore::Streamed {
                packed_path: packed_path.clone(),
                scale_path: scale_path.clone(),
                packed_name: packed_name.clone(),
                scale_name: scale_name.clone(),
                expert: e,
            }
        };
        out.push(Step3p7Expert { store, global: globals[e], out_f, in_f, group_size: group });
    }
    Ok(out)
}

/// Router + 3D NVFP4 experts + ungated shared expert for one MoE layer.
pub struct OwnedMoe {
    /// router gate: [num_experts, hidden]
    pub router: Vec<f32>,
    /// router_bias: [num_experts]
    pub bias: Vec<f32>,
    pub experts: OwnedExpertsPacked,
    /// shared expert (Step3p7MLP, plain SwiGLU): [shared_inter, hidden]
    pub shared_gate: Vec<f32>,
    pub shared_up: Vec<f32>,
    /// [hidden, shared_inter]
    pub shared_down: Vec<f32>,
    /// swiglu clamp for the routed experts (L43/44 → Some(7.0)), else None.
    pub expert_limit: Option<f32>,
    /// swiglu clamp for the shared expert (L43/44 → Some(16.0)), else None.
    pub shared_limit: Option<f32>,
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

pub struct Step3p7Weights {
    /// [vocab, hidden]
    pub embed: Vec<f32>,
    /// model.language_model.norm.weight [hidden]
    pub final_norm: Vec<f32>,
    pub layers: Vec<OwnedLayer>,
    /// lm_head.weight [vocab, hidden] (None → return final hidden instead of logits)
    pub lm_head: Option<Vec<f32>>,
}

/// A loaded Step-3.7 model: parsed config + owned host weights for a resident
/// layer window. This is the `lib.rs` model-load product stored as
/// `VulkanModel.step3p7` by the `mt=="step3p7"` dispatch — the Step-3.7 analog of
/// `NemotronModel`/`Qwen35Model`/`LagunaModel`.
///
/// Phase 1 is CPU-only (the pure `cpu_matmul`/`cpu_sdpa` reference this gate
/// validates); GPU-resident experts (qwen35_moe kernels) + full 45-layer +
/// TP-2×PP-5 @ 10 nodes are the node-count-blocked later phase. The loader
/// materializes layers `[0, pp_end)`; a true `[pp_start, pp_end)` window is
/// deferred to that phase, so `pp_start` is 0 here.
pub struct Step3p7Model {
    pub config: Step3p7Config,
    pub weights: Step3p7Weights,
    pub pp_start: usize,
    pub pp_end: usize,
    pub pp_first: bool,
    pub pp_last: bool,
    /// Stateful single-token DECODE state (KV cache per resident layer). `None`
    /// until `reset_decode_state`. The CPU decode path here is the correctness
    /// ORACLE for the GPU-resident `Step3p7GpuStage` decode: it reuses the EXACT
    /// prefill ops, so a chained single-token decode reproduces the stateless
    /// prefill BIT-IDENTICALLY (proven offline by `step3p7_decode_equals_prefill`).
    pub decode: Option<Step3p7DecodeState>,
    /// GPU-resident stage (experts NVFP4 on device + attn/dense/shared f16 + host
    /// glue). `None` in CPU mode; `Some` when `VLLM_VULKAN_STEP3P7_GPU_RESIDENT=1`.
    /// When present, `decode_step` dispatches to it (Ling `gpu`-field pattern).
    pub gpu: Option<crate::step3p7_gpu::Step3p7GpuStage>,
}

/// Per-layer KV cache for stateful single-token decode. `k` holds the RoPE'd keys
/// and `v` the raw values, each appended `[nkv*head_dim]` per accepted token, so
/// slot `t` is `k[t*nkv*hd .. (t+1)*nkv*hd]`. Growing to `len` tokens is exactly
/// the `k_ctx`/`v_ctx = &k[0..(p+1)*nkv*hd]` slice the prefill SDPA consumes at
/// position `p` — which is why chained decode == prefill bit-for-bit.
#[derive(Clone, Default)]
pub struct Step3p7KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
}

/// Whole-window decode state: one KV cache per resident (local) layer + the running
/// absolute position (== every layer's `kv.len`, tracked once).
#[derive(Clone, Default)]
pub struct Step3p7DecodeState {
    pub kv: Vec<Step3p7KvCache>,
    pub pos: usize,
}

impl Step3p7Model {
    /// Load a Step-3.7 checkpoint from `dir` (must hold `config.json` +
    /// `model.safetensors.index.json` + the referenced shards) into a CPU model.
    /// Loads layers `[0, pp_end)`; `keep_lm` (== last stage) also pulls
    /// `lm_head.weight`. Calls the SAME `load_step3p7_weights_cpu` the assembled-
    /// forward gate exercises, so a passing gate certifies this exact load path.
    pub fn load_cpu(
        dir: &std::path::Path,
        cfg: Step3p7Config,
        pp_start: usize,
        pp_end: usize,
        keep_lm: bool,
    ) -> Result<Self, String> {
        let total = cfg.num_hidden_layers;
        let pp_first = pp_start == 0;
        let pp_last = pp_end >= total;
        let weights = load_step3p7_weights_cpu(dir, &cfg, pp_start, pp_end, keep_lm)?;
        Ok(Step3p7Model {
            config: cfg, weights, pp_start, pp_end, pp_first, pp_last,
            decode: None, gpu: None,
        })
    }

    /// Load a GPU-resident stage for this PP window: build the compute engine, upload
    /// experts (NVFP4 3D-stacked) + attn/dense/shared (f16) to device with a per-layer
    /// stream-upload-then-free (dodges the host-double LOAD-OOM), and keep the host CPU
    /// weights ONLY for the small glue the hybrid decode runs on host. `device_idx`
    /// selects the Vulkan device. Gated by `VLLM_VULKAN_STEP3P7_GPU_RESIDENT` in lib.rs.
    pub fn load_gpu_resident(
        dir: &std::path::Path,
        cfg: Step3p7Config,
        pp_start: usize,
        pp_end: usize,
        keep_lm: bool,
        device_idx: usize,
    ) -> Result<Self, String> {
        let total = cfg.num_hidden_layers;
        let pp_first = pp_start == 0;
        let pp_last = pp_end >= total;
        // `keep_edges` must cover BOTH edges: the FIRST stage needs the embedding table
        // and the LAST stage needs final_norm + lm_head. (from_ckpt_streamed then gates
        // embed on `first` and lm_head on `last` internally.) `keep_lm` alone (== last)
        // would starve a first-only PP window of its embed.
        let _ = keep_lm;
        let keep_edges = pp_first || pp_last;
        let gpu = crate::step3p7_gpu::Step3p7GpuStage::from_ckpt_streamed(
            dir, &cfg, pp_start, pp_end, keep_edges, device_idx,
        )?;
        // Host weights kept minimal — the GPU stage owns the resident buffers; the CPU
        // `weights` here is left empty (the resident path never touches it).
        let weights = Step3p7Weights {
            embed: Vec::new(), final_norm: Vec::new(), layers: Vec::new(), lm_head: None,
        };
        Ok(Step3p7Model {
            config: cfg, weights, pp_start, pp_end, pp_first, pp_last,
            decode: None, gpu: Some(gpu),
        })
    }

    #[inline]
    pub fn is_gpu_resident(&self) -> bool {
        self.gpu.is_some()
    }

    /// Reset the stateful decode KV caches (one per resident layer), position 0.
    /// Dispatches to the GPU stage when resident (Ling `reset_decode_state` pattern).
    pub fn reset_decode_state(&mut self) {
        if let Some(g) = self.gpu.as_mut() {
            g.reset_state();
            return;
        }
        let n = self.weights.layers.len();
        self.decode = Some(Step3p7DecodeState {
            kv: vec![Step3p7KvCache::default(); n],
            pos: 0,
        });
    }

    /// Single-token stateful decode step (the DECODE seam). First stage embeds
    /// `token_id` (ignores `hidden_in`); a mid stage consumes the previous stage's
    /// `[hidden]` and returns `[hidden]`; the LAST stage applies final_norm + lm_head
    /// on the token and returns `[vocab]` logits. Mirrors `pp_prefill` per-position but
    /// advances the KV cache IN PLACE — bit-identical to the prefill of the same prefix.
    /// Dispatches to the GPU-resident stage when present.
    pub fn decode_step(&mut self, token_id: u32, hidden_in: &[f32]) -> Result<Vec<f32>, String> {
        if self.gpu.is_some() {
            let out = {
                let g = self.gpu.as_mut().unwrap();
                g.decode_step(token_id, hidden_in)?
            };
            return Ok(out);
        }
        if self.decode.is_none() {
            self.reset_decode_state();
        }
        let cfg = &self.config;
        let h = cfg.hidden_size;
        // Build this token's input row [hidden].
        let mut hidden: Vec<f32> = if self.pp_first {
            let embed = &self.weights.embed;
            if embed.len() < cfg.vocab_size * h {
                return Err("decode_step: first stage missing embed".into());
            }
            embed[token_id as usize * h..(token_id as usize + 1) * h].to_vec()
        } else {
            if hidden_in.len() != h {
                return Err(format!(
                    "decode_step: hidden_in.len()={} != H={h}", hidden_in.len()
                ));
            }
            hidden_in.to_vec()
        };
        let pos = self.decode.as_ref().unwrap().pos;
        for local in 0..self.weights.layers.len() {
            let global = self.pp_start + local;
            hidden = step3p7_decode_layer(
                &hidden, global, local, &self.weights.layers[local], cfg,
                self.decode.as_mut().unwrap(),
            );
        }
        // advance position once per token (all layers advanced their kv.len in lockstep)
        self.decode.as_mut().unwrap().pos = pos + 1;
        if !self.pp_last {
            return Ok(hidden);
        }
        let normed = rms_norm_plus1(&hidden, &self.weights.final_norm, cfg.rms_norm_eps);
        let lm = self.weights.lm_head.as_ref().ok_or("decode_step: last stage missing lm_head")?;
        Ok(cpu_matmul(&normed, lm, 1, h, cfg.vocab_size))
    }

    /// Full CPU forward over a token-id sequence: last_hidden_state `[seq, hidden]`
    /// when no `lm_head`, else last-position logits `[vocab]`.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        self.weights.forward(tokens, &self.config)
    }

    /// PP-window prefill (the fleet fit-to-validate seam). Builds this stage's input
    /// `[seq, hidden]` — embed the tokens on the FIRST stage, else take the previous
    /// stage's hidden — runs the window's layers (global layer index = `pp_start +
    /// local`, so per-layer RoPE/attn-kind/MoE/swiglu stay correct for a window that
    /// does not start at layer 0), and on the LAST stage applies final_norm + lm_head
    /// on the last position → `[vocab]`. Middle stages return `[seq*hidden]` hidden.
    /// This is bit-identical to the single-node `Step3p7Weights::forward` composed
    /// across the PP split (same ops, same order), so a passing multi-node gate
    /// certifies the whole-model forward.
    pub fn pp_prefill(&self, tokens: &[u32], hidden_in: Vec<f32>, seq: usize) -> Result<Vec<f32>, String> {
        let h = self.config.hidden_size;
        if seq == 0 {
            return Err("pp_prefill: empty prompt".into());
        }
        let mut hidden = if self.pp_first {
            let embed = &self.weights.embed;
            if embed.len() < self.config.vocab_size * h {
                return Err("pp_prefill: first stage missing embed".into());
            }
            let mut hv = vec![0f32; seq * h];
            for (t, &tok) in tokens.iter().enumerate().take(seq) {
                hv[t * h..(t + 1) * h]
                    .copy_from_slice(&embed[tok as usize * h..(tok as usize + 1) * h]);
            }
            hv
        } else {
            if hidden_in.len() != seq * h {
                return Err(format!(
                    "pp_prefill: hidden_in.len()={} != seq*H={}",
                    hidden_in.len(), seq * h
                ));
            }
            hidden_in
        };
        for (local, lw) in self.weights.layers.iter().enumerate() {
            let global = self.pp_start + local;
            hidden = step3p7_layer_forward(&hidden, seq, global, lw, &self.config);
        }
        if !self.pp_last {
            return Ok(hidden);
        }
        let normed = rms_norm_rows_plus1(&hidden, &self.weights.final_norm, seq, h, self.config.rms_norm_eps);
        let lm = self.weights.lm_head.as_ref().ok_or("pp_prefill: last stage missing lm_head")?;
        let last = &normed[(seq - 1) * h..seq * h];
        Ok(cpu_matmul(last, lm, 1, h, self.config.vocab_size))
    }
}

// ─── Forward blocks ──────────────────────────────────────────────────────────

/// Gated GQA attention for one layer over a `[seq, hidden]` prefill. `normed` is
/// the post-input_layernorm input (also the gate source). q/k get per-head
/// weight+1 RMSNorm over head_dim BEFORE RoPE; RoPE is llama3-scaled partial on
/// full layers, plain on sliding; the per-head SCALAR sigmoid gate is applied to
/// the attn output BEFORE o_proj.
fn step3p7_attn(
    normed: &[f32],
    seq: usize,
    layer_idx: usize,
    w: &OwnedAttn,
    cfg: &Step3p7Config,
) -> Vec<f32> {
    let hs = cfg.hidden_size;
    let hd = cfg.head_dim;
    let nkv = cfg.num_key_value_heads;
    let la = cfg.layers[layer_idx];
    let nq = la.num_heads;
    let eps = cfg.rms_norm_eps;

    let mut q = cpu_matmul(normed, &w.q_proj, seq, hs, nq * hd); // [seq, nq*hd]
    let mut k = cpu_matmul(normed, &w.k_proj, seq, hs, nkv * hd); // [seq, nkv*hd]
    let v = cpu_matmul(normed, &w.v_proj, seq, hs, nkv * hd); // [seq, nkv*hd]

    // Per-head weight+1 RMSNorm (over head_dim) then partial RoPE, per position.
    for p in 0..seq {
        for h in 0..nq {
            let head = &mut q[p * nq * hd + h * hd..p * nq * hd + (h + 1) * hd];
            let normed_h = rms_norm_plus1(head, &w.q_norm, eps);
            let roped = partial_rope(&normed_h, p, &la, hd, &cfg.llama3);
            head.copy_from_slice(&roped);
        }
        for h in 0..nkv {
            let head = &mut k[p * nkv * hd + h * hd..p * nkv * hd + (h + 1) * hd];
            let normed_h = rms_norm_plus1(head, &w.k_norm, eps);
            let roped = partial_rope(&normed_h, p, &la, hd, &cfg.llama3);
            head.copy_from_slice(&roped);
        }
    }

    // Causal (+ sliding-window) SDPA, per query position over the growing KV.
    let scale = 1.0 / (hd as f32).sqrt();
    let window = la.sliding_window;
    let mut attn_out = vec![0.0f32; seq * nq * hd];
    for p in 0..seq {
        let q_p = &q[p * nq * hd..(p + 1) * nq * hd];
        let k_ctx = &k[0..(p + 1) * nkv * hd];
        let v_ctx = &v[0..(p + 1) * nkv * hd];
        let o = cpu_sdpa(q_p, k_ctx, v_ctx, nq, nkv, hd, p + 1, scale, window);
        attn_out[p * nq * hd..(p + 1) * nq * hd].copy_from_slice(&o);
    }

    // Per-head SCALAR sigmoid gate from the attn input, broadcast over head_dim,
    // applied BEFORE o_proj.
    let g = cpu_matmul(normed, &w.g_proj, seq, hs, nq); // [seq, nq]
    for p in 0..seq {
        let head_out = &mut attn_out[p * nq * hd..(p + 1) * nq * hd];
        let gated = head_gate(head_out, &g[p * nq..(p + 1) * nq], nq, hd);
        head_out.copy_from_slice(&gated);
    }

    cpu_matmul(&attn_out, &w.o_proj, seq, nq * hd, hs) // [seq, hidden]
}

/// Dense SwiGLU MLP over `[seq, hidden]` (layers 0..moe_first_layer, no clamp).
fn step3p7_dense_mlp(h: &[f32], seq: usize, w: &OwnedDense, cfg: &Step3p7Config) -> Vec<f32> {
    let hs = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let g = cpu_matmul(h, &w.gate, seq, hs, inter);
    let u = cpu_matmul(h, &w.up, seq, hs, inter);
    let act: Vec<f32> = clamped_swiglu_prod(&g, &u, None);
    cpu_matmul(&act, &w.down, seq, inter, hs)
}

/// MoE mixer for a SINGLE token `h` (`[hidden]`, post-norm). Bias-corrected
/// sigmoid router selects top-8; each expert dequantized on demand from packed
/// 3D NVFP4; the ungated shared expert always runs. Router weights already carry
/// ×router_scaling_factor (baked in by `bias_router`); shared expert is NOT scaled.
fn step3p7_moe_token(h: &[f32], w: &OwnedMoe, cfg: &Step3p7Config) -> Vec<f32> {
    let hs = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let shared_inter = cfg.share_expert_dim;
    let top_k = cfg.num_experts_per_tok;

    // need_fp32_gate: router logits in f32 (weights already f32 here).
    let logits = cpu_matmul(h, &w.router, 1, hs, cfg.num_experts); // [num_experts]
    let (indices, weights) = bias_router(&logits, &w.bias, top_k, cfg.router_scaling_factor);

    let mut routed = vec![0.0f32; hs];
    for (kth, &e) in indices.iter().enumerate() {
        let gate_w = w.experts.dequant(e, 0); // [inter, hidden]
        let up_w = w.experts.dequant(e, 1); // [inter, hidden]
        let down_w = w.experts.dequant(e, 2); // [hidden, inter]
        let gp = cpu_matmul(h, &gate_w, 1, hs, inter);
        let up = cpu_matmul(h, &up_w, 1, hs, inter);
        let act = clamped_swiglu_prod(&gp, &up, w.expert_limit);
        let dn = cpu_matmul(&act, &down_w, 1, inter, hs); // [hidden]
        let wk = weights[kth];
        for (r, &o) in routed.iter_mut().zip(&dn) {
            *r += o * wk;
        }
    }

    // Ungated shared expert (plain/clamped SwiGLU).
    let sg = cpu_matmul(h, &w.shared_gate, 1, hs, shared_inter);
    let su = cpu_matmul(h, &w.shared_up, 1, hs, shared_inter);
    let sact = clamped_swiglu_prod(&sg, &su, w.shared_limit);
    let sd = cpu_matmul(&sact, &w.shared_down, 1, shared_inter, hs); // [hidden]

    routed.iter().zip(&sd).map(|(&r, &s)| r + s).collect()
}

/// One decoder layer: pre-norm(+1) gated attention + residual, then pre-norm(+1)
/// MoE/dense MLP + residual. `hidden` is `[seq, hidden]`.
pub fn step3p7_layer_forward(
    hidden: &[f32],
    seq: usize,
    layer_idx: usize,
    lw: &OwnedLayer,
    cfg: &Step3p7Config,
) -> Vec<f32> {
    let hs = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;

    // weight+1 RMSNorm (row-wise) — gemma-style.
    let normed = rms_norm_rows_plus1(hidden, &lw.input_ln, seq, hs, eps);
    let attn = step3p7_attn(&normed, seq, layer_idx, &lw.attn, cfg);
    let h1: Vec<f32> = hidden.iter().zip(&attn).map(|(&a, &b)| a + b).collect();

    let normed2 = rms_norm_rows_plus1(&h1, &lw.post_ln, seq, hs, eps);
    let mlp = match &lw.mlp {
        OwnedMlp::Dense(d) => step3p7_dense_mlp(&normed2, seq, d, cfg),
        OwnedMlp::Moe(m) => {
            let mut out = vec![0.0f32; seq * hs];
            for t in 0..seq {
                let ht = &normed2[t * hs..(t + 1) * hs];
                let m_out = step3p7_moe_token(ht, m, cfg);
                out[t * hs..(t + 1) * hs].copy_from_slice(&m_out);
            }
            out
        }
    };
    h1.iter().zip(&mlp).map(|(&a, &b)| a + b).collect()
}

// ─── Stateful single-token DECODE blocks (CPU oracle for the GPU stage) ──────
//
// These reproduce the prefill ops EXACTLY, one token at a time, advancing the KV
// cache in place. Because the per-position math (proj → qk-norm+1 → partial RoPE
// at `pos` → append → causal/sliding SDPA over the grown KV → head-gate → o_proj)
// is byte-for-byte the same as the prefill's per-position path, a chained decode
// over a prompt reproduces the stateless prefill BIT-IDENTICALLY. This is the
// property the offline `step3p7_decode_equals_prefill` gate asserts, and it is
// what certifies the GPU-resident decode STATE machine (the exact place Kimi
// silently drifted: correct prefill, wrong decode state).

/// Gated GQA attention for ONE decode token. `normed` is `[hidden]` (post-input_ln,
/// also the gate source). `kv` is this layer's cache; the new RoPE'd key + raw value
/// are appended, then SDPA runs over the whole grown cache (sliding-window on sliding
/// layers) — identical to prefill position `pos == kv.len` before the append.
fn step3p7_attn_decode(
    normed: &[f32],
    global_idx: usize,
    w: &OwnedAttn,
    cfg: &Step3p7Config,
    kv: &mut Step3p7KvCache,
) -> Vec<f32> {
    let hs = cfg.hidden_size;
    let hd = cfg.head_dim;
    let nkv = cfg.num_key_value_heads;
    let la = cfg.layers[global_idx];
    let nq = la.num_heads;
    let eps = cfg.rms_norm_eps;
    let pos = kv.len;

    let mut q = cpu_matmul(normed, &w.q_proj, 1, hs, nq * hd);
    let mut k = cpu_matmul(normed, &w.k_proj, 1, hs, nkv * hd);
    let v = cpu_matmul(normed, &w.v_proj, 1, hs, nkv * hd);
    for hh in 0..nq {
        let head = &mut q[hh * hd..(hh + 1) * hd];
        let normed_h = rms_norm_plus1(head, &w.q_norm, eps);
        let roped = partial_rope(&normed_h, pos, &la, hd, &cfg.llama3);
        head.copy_from_slice(&roped);
    }
    for hh in 0..nkv {
        let head = &mut k[hh * hd..(hh + 1) * hd];
        let normed_h = rms_norm_plus1(head, &w.k_norm, eps);
        let roped = partial_rope(&normed_h, pos, &la, hd, &cfg.llama3);
        head.copy_from_slice(&roped);
    }
    // Append this token's RoPE'd K and raw V; the cache now spans [0, pos+1).
    kv.k.extend_from_slice(&k);
    kv.v.extend_from_slice(&v);
    kv.len += 1;

    let scale = 1.0 / (hd as f32).sqrt();
    let o = cpu_sdpa(
        &q,
        &kv.k[0..kv.len * nkv * hd],
        &kv.v[0..kv.len * nkv * hd],
        nq, nkv, hd, kv.len, scale, la.sliding_window,
    );
    let g = cpu_matmul(normed, &w.g_proj, 1, hs, nq);
    let gated = head_gate(&o, &g, nq, hd);
    cpu_matmul(&gated, &w.o_proj, 1, nq * hd, hs)
}

/// One decoder layer for a single decode token: pre-norm(+1) gated attention +
/// residual, then pre-norm(+1) MoE/dense MLP + residual. Mirrors
/// `step3p7_layer_forward` with `seq==1` and the stateful KV attention.
pub fn step3p7_decode_layer(
    hidden: &[f32],
    global_idx: usize,
    local_idx: usize,
    lw: &OwnedLayer,
    cfg: &Step3p7Config,
    state: &mut Step3p7DecodeState,
) -> Vec<f32> {
    let eps = cfg.rms_norm_eps;
    let normed = rms_norm_plus1(hidden, &lw.input_ln, eps);
    let attn = step3p7_attn_decode(&normed, global_idx, &lw.attn, cfg, &mut state.kv[local_idx]);
    let h1: Vec<f32> = hidden.iter().zip(&attn).map(|(&a, &b)| a + b).collect();
    let normed2 = rms_norm_plus1(&h1, &lw.post_ln, eps);
    let mlp = match &lw.mlp {
        OwnedMlp::Dense(d) => step3p7_dense_mlp(&normed2, 1, d, cfg),
        OwnedMlp::Moe(m) => step3p7_moe_token(&normed2, m, cfg),
    };
    h1.iter().zip(&mlp).map(|(&a, &b)| a + b).collect()
}

/// Row-wise weight+1 RMSNorm over `[seq, hidden]`.
fn rms_norm_rows_plus1(x: &[f32], w: &[f32], seq: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * hidden];
    for t in 0..seq {
        let row = rms_norm_plus1(&x[t * hidden..(t + 1) * hidden], w, eps);
        out[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }
    out
}

impl Step3p7Weights {
    /// Full CPU forward over a token id sequence. Returns per-position
    /// `[seq, hidden]` (last_hidden_state, pre-lm_head) when `lm_head` is None,
    /// else logits for the LAST position only (`[vocab]`).
    pub fn forward(&self, tokens: &[u32], cfg: &Step3p7Config) -> Vec<f32> {
        let hs = cfg.hidden_size;
        let seq = tokens.len();
        let mut hidden = vec![0.0f32; seq * hs];
        for (t, &tok) in tokens.iter().enumerate() {
            let row = &self.embed[tok as usize * hs..(tok as usize + 1) * hs];
            hidden[t * hs..(t + 1) * hs].copy_from_slice(row);
        }
        for (li, lw) in self.layers.iter().enumerate() {
            hidden = step3p7_layer_forward(&hidden, seq, li, lw, cfg);
        }
        let normed = rms_norm_rows_plus1(&hidden, &self.final_norm, seq, hs, cfg.rms_norm_eps);
        match &self.lm_head {
            None => normed,
            Some(lm) => {
                let last = &normed[(seq - 1) * hs..seq * hs];
                cpu_matmul(last, lm, 1, hs, cfg.vocab_size)
            }
        }
    }

    /// Forward that dumps each layer's post-residual hidden as raw LE-f32 to
    /// `{dump_path}.{stage}.r0` when `dump_path` is Some — the bisect-ladder
    /// producer for the assembled-forward gate. Never applies lm_head.
    pub fn forward_with_stage_dump(
        &self,
        tokens: &[u32],
        cfg: &Step3p7Config,
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
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &v in data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                if let Ok(mut f) = std::fs::File::create(&path) {
                    let _ = f.write_all(&bytes);
                }
            }
        };
        write_stage("embed", &hidden);
        for (li, lw) in self.layers.iter().enumerate() {
            hidden = step3p7_layer_forward(&hidden, seq, li, lw, cfg);
            write_stage(&format!("layer{li}"), &hidden);
        }
        let normed = rms_norm_rows_plus1(&hidden, &self.final_norm, seq, hs, cfg.rms_norm_eps);
        write_stage("final_norm", &normed);
        normed
    }
}

// ─── Real-checkpoint CPU loader ──────────────────────────────────────────────

pub(crate) fn decode_bf16_f32(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>, String> {
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
        other => return Err(format!("unsupported plain dtype {other:?} in step3p7 loader")),
    })
}

/// Load `[0, num_layers)` of a Step-3.7 checkpoint to host f32 (BF16 dequant) +
/// packed 3D NVFP4 experts, keyed off `model.safetensors.index.json` in `dir`
/// (reads whatever shard layout is present — no shard-count assumption). The 3
/// MTP heads (checkpoint layers 45-47) are naturally dropped: `num_layers <=
/// num_hidden_layers == 45`. `keep_lm` also loads `lm_head.weight`.
pub fn load_step3p7_weights_cpu(
    dir: &std::path::Path,
    cfg: &Step3p7Config,
    pp_start: usize,
    pp_end: usize,
    keep_lm: bool,
) -> Result<Step3p7Weights, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::collections::HashMap;

    // True PP window: materialize only layers [pp_start, pp_end). The embedding is
    // needed only on the first stage; final_norm is cheap (always kept); lm_head only
    // on the last stage (keep_lm). This — plus per-expert overflow-streaming below —
    // is what bounds a stage's DRAM ≤ node so PP-6 fits on 5-6 nodes.
    let keep_embed = pp_start == 0;

    let index_path = dir.join("model.safetensors.index.json");
    let index: Value = serde_json::from_str(
        &std::fs::read_to_string(&index_path).map_err(|e| format!("read index: {e}"))?,
    )
    .map_err(|e| format!("parse index: {e}"))?;
    let weight_map = index
        .get("weight_map")
        .and_then(|x| x.as_object())
        .ok_or("index.json: missing weight_map")?;

    let hs = cfg.hidden_size;
    let group = 16usize;
    let lm = "model.language_model";

    let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
    let mut want = |name: String, set: &mut HashMap<String, Vec<String>>| {
        if let Some(shard) = weight_map.get(&name).and_then(|x| x.as_str()) {
            set.entry(shard.to_string()).or_default().push(name);
        }
    };
    if keep_embed {
        want(format!("{lm}.embed_tokens.weight"), &mut by_shard);
    }
    want(format!("{lm}.norm.weight"), &mut by_shard);
    if keep_lm {
        want("lm_head.weight".into(), &mut by_shard);
    }
    for li in pp_start..pp_end {
        let p = format!("{lm}.layers.{li}");
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
        if cfg.is_moe_layer(li) {
            want(format!("{p}.moe.gate.weight"), &mut by_shard);
            want(format!("{p}.moe.router_bias"), &mut by_shard);
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                // Only the tiny F32 per-expert global here; the big packed nibbles +
                // e4m3 group scales are NOT slurped into the heap — they are sliced
                // per-expert (resident) or left as NAS descriptors (streamed) by
                // `load_experts_proj`, which bounds peak load DRAM to the budget.
                want(format!("{p}.moe.{proj}.weight_scale_2"), &mut by_shard);
            }
            for t in [
                format!("{p}.share_expert.gate_proj.weight"),
                format!("{p}.share_expert.up_proj.weight"),
                format!("{p}.share_expert.down_proj.weight"),
            ] {
                want(t, &mut by_shard);
            }
        } else {
            for t in [
                format!("{p}.mlp.gate_proj.weight"),
                format!("{p}.mlp.up_proj.weight"),
                format!("{p}.mlp.down_proj.weight"),
            ] {
                want(t, &mut by_shard);
            }
        }
    }

    let mut f32s: HashMap<String, Vec<f32>> = HashMap::new();
    let mut f32arr: HashMap<String, Vec<f32>> = HashMap::new();

    for (shard, names) in &by_shard {
        let path = dir.join(shard);
        let file = std::fs::File::open(&path).map_err(|e| format!("open {shard}: {e}"))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap {shard}: {e}"))? };
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse {shard}: {e}"))?;
        for name in names {
            let view = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
            if name.ends_with(".weight_scale_2") {
                // F32 per-expert global scales [E] — the ONLY expert tensor slurped to
                // the heap here (tiny). The big packed nibbles + e4m3 group scales are
                // sliced per-expert by `load_experts_proj` (resident) or left as NAS
                // descriptors (streamed), bounding peak DRAM to the resident budget.
                let arr: Vec<f32> = view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                f32arr.insert(name.clone(), arr);
            } else {
                f32s.insert(name.clone(), decode_bf16_f32(&view)?);
            }
        }
    }

    let take_f32 = |m: &mut HashMap<String, Vec<f32>>, n: &str| -> Result<Vec<f32>, String> {
        m.remove(n).ok_or_else(|| format!("missing tensor {n}"))
    };
    let take_arr = |m: &mut HashMap<String, Vec<f32>>, n: &str| -> Result<Vec<f32>, String> {
        m.remove(n).ok_or_else(|| format!("missing tensor {n}"))
    };

    let embed = if keep_embed {
        take_f32(&mut f32s, &format!("{lm}.embed_tokens.weight"))?
    } else {
        Vec::new()
    };
    let final_norm = take_f32(&mut f32s, &format!("{lm}.norm.weight"))?;
    let lm_head = if keep_lm { Some(take_f32(&mut f32s, "lm_head.weight")?) } else { None };

    // Overflow-stream harness (default-OFF; byte-identical when unset). A live mmap
    // cache lets resident experts copy only their own byte slice — the big packed
    // tensor is never fully materialized — so per-stage peak DRAM ≤ the budget.
    let stream_cfg = MoeStreamCfg::from_env();
    let mut expert_mmaps: HashMap<String, Mmap> = HashMap::new();
    let mut resident_expert_bytes: u64 = 0;
    let mut streamed_total: usize = 0;
    let mut moe_layers: usize = 0;

    let mut layers = Vec::with_capacity(pp_end - pp_start);
    for li in pp_start..pp_end {
        let p = format!("{lm}.layers.{li}");
        let attn = OwnedAttn {
            q_proj: take_f32(&mut f32s, &format!("{p}.self_attn.q_proj.weight"))?,
            k_proj: take_f32(&mut f32s, &format!("{p}.self_attn.k_proj.weight"))?,
            v_proj: take_f32(&mut f32s, &format!("{p}.self_attn.v_proj.weight"))?,
            o_proj: take_f32(&mut f32s, &format!("{p}.self_attn.o_proj.weight"))?,
            g_proj: take_f32(&mut f32s, &format!("{p}.self_attn.g_proj.weight"))?,
            q_norm: take_f32(&mut f32s, &format!("{p}.self_attn.q_norm.weight"))?,
            k_norm: take_f32(&mut f32s, &format!("{p}.self_attn.k_norm.weight"))?,
        };
        let mlp = if cfg.is_moe_layer(li) {
            moe_layers += 1;
            let inter_moe = cfg.moe_intermediate_size;
            let gate_g = take_arr(&mut f32arr, &format!("{p}.moe.gate_proj.weight_scale_2"))?;
            let up_g = take_arr(&mut f32arr, &format!("{p}.moe.up_proj.weight_scale_2"))?;
            let down_g = take_arr(&mut f32arr, &format!("{p}.moe.down_proj.weight_scale_2"))?;
            let gate = load_experts_proj(
                dir, weight_map, &mut expert_mmaps, &format!("{p}.moe.gate_proj"),
                &gate_g, cfg.num_experts, inter_moe, hs, group, stream_cfg, &mut resident_expert_bytes,
            )?;
            let up = load_experts_proj(
                dir, weight_map, &mut expert_mmaps, &format!("{p}.moe.up_proj"),
                &up_g, cfg.num_experts, inter_moe, hs, group, stream_cfg, &mut resident_expert_bytes,
            )?;
            let down = load_experts_proj(
                dir, weight_map, &mut expert_mmaps, &format!("{p}.moe.down_proj"),
                &down_g, cfg.num_experts, hs, inter_moe, group, stream_cfg, &mut resident_expert_bytes,
            )?;
            let ex = OwnedExpertsPacked {
                gate, up, down,
                num_experts: cfg.num_experts,
                inter: inter_moe,
                hidden: hs,
                group_size: group,
            };
            streamed_total += ex.streamed_count();
            let lim = |v: &[f32]| -> Option<f32> {
                v.get(li).copied().filter(|&x| x != 0.0)
            };
            OwnedMlp::Moe(OwnedMoe {
                router: take_f32(&mut f32s, &format!("{p}.moe.gate.weight"))?,
                bias: take_f32(&mut f32s, &format!("{p}.moe.router_bias"))?,
                experts: ex,
                shared_gate: take_f32(&mut f32s, &format!("{p}.share_expert.gate_proj.weight"))?,
                shared_up: take_f32(&mut f32s, &format!("{p}.share_expert.up_proj.weight"))?,
                shared_down: take_f32(&mut f32s, &format!("{p}.share_expert.down_proj.weight"))?,
                expert_limit: lim(&cfg.swiglu_limit_expert),
                shared_limit: lim(&cfg.swiglu_limit_shared),
            })
        } else {
            OwnedMlp::Dense(OwnedDense {
                gate: take_f32(&mut f32s, &format!("{p}.mlp.gate_proj.weight"))?,
                up: take_f32(&mut f32s, &format!("{p}.mlp.up_proj.weight"))?,
                down: take_f32(&mut f32s, &format!("{p}.mlp.down_proj.weight"))?,
            })
        };
        layers.push(OwnedLayer {
            input_ln: take_f32(&mut f32s, &format!("{p}.input_layernorm.weight"))?,
            post_ln: take_f32(&mut f32s, &format!("{p}.post_attention_layernorm.weight"))?,
            attn,
            mlp,
        });
    }

    if stream_cfg.enabled {
        let total_linears = moe_layers * 3 * cfg.num_experts;
        log::info!(
            "Step-3.7 overflow-stream [{pp_start},{pp_end}): resident-expert {:.2} GiB \
             (budget {:.2} GiB); {streamed_total}/{total_linears} expert-linears STREAMED \
             ({:.0}%)",
            resident_expert_bytes as f64 / 1e9,
            stream_cfg.budget_bytes as f64 / 1e9,
            if total_linears > 0 { 100.0 * streamed_total as f64 / total_linears as f64 } else { 0.0 },
        );
    }

    Ok(Step3p7Weights { embed, final_norm, layers, lm_head })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: &[f32], b: &[f32], tol: f32) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    // ── Phase-2 offline proof: chained single-token DECODE == stateless PREFILL ──
    // The exact property the GPU-resident decode STATE machine must satisfy (the spot
    // Kimi silently drifted). Runs entirely on the Mac with synthetic tiny weights — no
    // GPU, no checkpoint — so it certifies the KV-append / RoPE-at-pos / sliding-window /
    // causal-mask state logic that `Step3p7GpuStage::decode_step` mirrors 1:1.
    fn lcg(state: &mut u64) -> f32 {
        // deterministic pseudo-random in [-0.1, 0.1]
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = ((*state >> 33) as u32) as f32 / u32::MAX as f32; // [0,1)
        (x - 0.5) * 0.2
    }
    fn rvec(n: usize, s: &mut u64) -> Vec<f32> {
        (0..n).map(|_| lcg(s)).collect()
    }
    fn rbytes(n: usize, s: &mut u64) -> Vec<u8> {
        (0..n).map(|_| { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (*s >> 40) as u8 }).collect()
    }

    fn fake_cfg() -> Step3p7Config {
        let hd = 8usize;
        let layers = vec![
            LayerAttn { kind: AttnKind::Full, num_heads: 4, rope_theta: 5e6, partial_rotary: 0.5, use_llama3_scale: true, sliding_window: None },
            LayerAttn { kind: AttnKind::Sliding, num_heads: 4, rope_theta: 1e4, partial_rotary: 1.0, use_llama3_scale: false, sliding_window: Some(2) },
            LayerAttn { kind: AttnKind::Full, num_heads: 4, rope_theta: 5e6, partial_rotary: 0.5, use_llama3_scale: true, sliding_window: None },
            LayerAttn { kind: AttnKind::Sliding, num_heads: 4, rope_theta: 1e4, partial_rotary: 1.0, use_llama3_scale: false, sliding_window: Some(2) },
        ];
        let n = layers.len();
        Step3p7Config {
            hidden_size: 32, num_hidden_layers: n, vocab_size: 40, rms_norm_eps: 1e-5,
            head_dim: hd, num_key_value_heads: 2, intermediate_size: 24,
            num_experts: 6, num_experts_per_tok: 2, moe_intermediate_size: 16,
            share_expert_dim: 16, router_scaling_factor: 3.0, use_router_bias: true,
            moe_first_layer: 2, layers,
            // clamp on the LAST moe layer only (exercise the clamped-swiglu decode path)
            swiglu_limit_expert: vec![0.0, 0.0, 0.0, 7.0],
            swiglu_limit_shared: vec![0.0, 0.0, 0.0, 16.0],
            llama3: Llama3Rope { factor: 2.0, low_freq_factor: 1.0, high_freq_factor: 32.0, original_max_pos: 131072.0 },
            tie_word_embeddings: false,
        }
    }

    fn fake_expert(out: usize, inn: usize, group: usize, s: &mut u64) -> Step3p7Expert {
        Step3p7Expert {
            store: ExpertStore::Resident {
                packed: rbytes(out * inn / 2, s),
                scale: rbytes(out * (inn / group), s),
            },
            global: 0.02, out_f: out, in_f: inn, group_size: group,
        }
    }
    fn fake_switch(e: usize, out: usize, inn: usize, group: usize, s: &mut u64) -> Vec<Step3p7Expert> {
        (0..e).map(|_| fake_expert(out, inn, group, s)).collect()
    }

    fn fake_model(cfg: &Step3p7Config) -> Step3p7Model {
        let mut s = 0x9E3779B97F4A7C15u64;
        let hs = cfg.hidden_size;
        let hd = cfg.head_dim;
        let nkv = cfg.num_key_value_heads;
        let group = 16usize;
        let mut layers = Vec::new();
        for li in 0..cfg.num_hidden_layers {
            let nq = cfg.layers[li].num_heads;
            let attn = OwnedAttn {
                q_proj: rvec(nq * hd * hs, &mut s),
                k_proj: rvec(nkv * hd * hs, &mut s),
                v_proj: rvec(nkv * hd * hs, &mut s),
                o_proj: rvec(hs * nq * hd, &mut s),
                g_proj: rvec(nq * hs, &mut s),
                q_norm: rvec(hd, &mut s),
                k_norm: rvec(hd, &mut s),
            };
            let mlp = if cfg.is_moe_layer(li) {
                let inter = cfg.moe_intermediate_size;
                let sh = cfg.share_expert_dim;
                let experts = OwnedExpertsPacked {
                    gate: fake_switch(cfg.num_experts, inter, hs, group, &mut s),
                    up: fake_switch(cfg.num_experts, inter, hs, group, &mut s),
                    down: fake_switch(cfg.num_experts, hs, inter, group, &mut s),
                    num_experts: cfg.num_experts, inter, hidden: hs, group_size: group,
                };
                let lim = |v: &[f32]| v.get(li).copied().filter(|&x| x != 0.0);
                OwnedMlp::Moe(OwnedMoe {
                    router: rvec(cfg.num_experts * hs, &mut s),
                    bias: rvec(cfg.num_experts, &mut s),
                    experts,
                    shared_gate: rvec(sh * hs, &mut s),
                    shared_up: rvec(sh * hs, &mut s),
                    shared_down: rvec(hs * sh, &mut s),
                    expert_limit: lim(&cfg.swiglu_limit_expert),
                    shared_limit: lim(&cfg.swiglu_limit_shared),
                })
            } else {
                let inter = cfg.intermediate_size;
                OwnedMlp::Dense(OwnedDense {
                    gate: rvec(inter * hs, &mut s),
                    up: rvec(inter * hs, &mut s),
                    down: rvec(hs * inter, &mut s),
                })
            };
            layers.push(OwnedLayer {
                input_ln: rvec(hs, &mut s), post_ln: rvec(hs, &mut s), attn, mlp,
            });
        }
        let weights = Step3p7Weights {
            embed: rvec(cfg.vocab_size * hs, &mut s),
            final_norm: rvec(hs, &mut s),
            layers,
            lm_head: Some(rvec(cfg.vocab_size * hs, &mut s)),
        };
        Step3p7Model {
            config: cfg.clone(), weights,
            pp_start: 0, pp_end: cfg.num_hidden_layers, pp_first: true, pp_last: true,
            decode: None, gpu: None,
        }
    }

    #[test]
    fn step3p7_decode_equals_prefill() {
        let cfg = fake_cfg();
        let mut model = fake_model(&cfg);
        let tokens: Vec<u32> = vec![7, 1, 30, 4, 22, 3, 15]; // seq 7 > sliding window 2
        let seq = tokens.len();

        // Stateless PREFILL of each prefix → last-position [vocab] logits (the golden).
        let mut prefill_logits: Vec<Vec<f32>> = Vec::with_capacity(seq);
        for l in 1..=seq {
            let logits = model
                .pp_prefill(&tokens[0..l], Vec::new(), l)
                .expect("prefill prefix");
            assert_eq!(logits.len(), cfg.vocab_size);
            prefill_logits.push(logits);
        }

        // Chained stateful DECODE → per-step [vocab] logits, KV advancing in place.
        model.reset_decode_state();
        for t in 0..seq {
            let dec = model.decode_step(tokens[t], &[]).expect("decode step");
            assert_eq!(dec.len(), cfg.vocab_size);
            let want = &prefill_logits[t];
            // BIT-IDENTICAL is the bar (same ops, same order → same bits).
            let mut mism = 0usize;
            for i in 0..dec.len() {
                if dec[i].to_bits() != want[i].to_bits() {
                    mism += 1;
                }
            }
            assert_eq!(
                mism, 0,
                "decode step {t}: {mism}/{} logits differ from prefill(tokens[0..{}]) \
                 (max_abs_err {:.3e})",
                dec.len(), t + 1, close(&dec, want, 0.0)
            );
            // argmax must also match (defensive; implied by bit-identity).
            let da = dec.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0;
            let pa = want.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0;
            assert_eq!(da, pa, "decode step {t}: argmax {da} != prefill argmax {pa}");
        }
        eprintln!(
            "STEP3P7 DECODE==PREFILL BIT-IDENTICAL: {seq} tokens, dense+MoE+sliding+clamp \
             layers, KV/RoPE/sliding-window state logic certified"
        );
    }

    // Goldens below are exported from p2_op_goldens.py / the golden generator, whose
    // reference paths were validated bit-exact (cos=1.0) against modeling_step3p7.py.
    #[test]
    fn t_rms_norm_plus1() {
        let x = [0.5, -1.2, 0.3, 2.0];
        let w = [0.1, -0.2, 0.05, 0.3];
        let want = [4.57538099e-01, -7.98611954e-01, 2.62044547e-01, 2.16290738e+00];
        let got = rms_norm_plus1(&x, &w, 1e-5);
        assert!(close(&got, &want, 1e-5) < 1e-5, "rms+1 {:?}", got);
    }

    #[test]
    fn t_head_gate() {
        let attn = [1.0, 2.0, -1.0, 0.5, -0.5, 3.0];
        let gl = [0.8, -0.4];
        let want = [6.89974481e-01, 1.37994896e+00, -6.89974481e-01, 2.00656170e-01, -2.00656170e-01, 1.20393702e+00];
        let got = head_gate(&attn, &gl, 2, 3);
        assert!(close(&got, &want, 1e-6) < 1e-6, "head_gate {:?}", got);
    }

    #[test]
    fn t_clamped_swiglu() {
        let gl = [0.5, 9.0, -2.0, 8.5];
        let up = [3.0, 10.0, -9.0, 1.0];
        let want = [9.33688997e-01, 4.90000000e+01, 1.66884091e+00, 7.00000000e+00];
        let got = clamped_swiglu_prod(&gl, &up, Some(7.0));
        assert!(close(&got, &want, 1e-4) < 1e-4, "clamped_swiglu {:?}", got);
    }

    #[test]
    fn t_bias_router() {
        let logits = [0.2, -1.0, 0.5, 0.1, -0.3, 0.8];
        let bias = [0.0, 2.0, 0.0, 0.0, 0.0, -1.0];
        let (idx, w) = bias_router(&logits, &bias, 2, 3.0);
        assert_eq!(idx, vec![1, 2], "bias-router selected {:?}", idx);
        let want_w = [9.05119568e-01, 2.09488043e+00];
        assert!(close(&w, &want_w, 1e-5) < 1e-5, "bias-router w {:?}", w);
    }

    #[test]
    fn t_partial_rope_llama3() {
        let vin = [0.3, -0.7, 1.1, 0.2, 0.9, -0.4, 0.6, -1.3];
        let want = [-4.52229758e-01, -7.00169958e-01, -1.04665574e+00, 1.99404186e-01, 0.9, -0.4, 0.6, -1.3];
        let la = LayerAttn {
            kind: AttnKind::Full,
            num_heads: 64,
            rope_theta: 5e6,
            partial_rotary: 0.5,
            use_llama3_scale: true,
            sliding_window: None,
        };
        let l3 = Llama3Rope { factor: 2.0, low_freq_factor: 1.0, high_freq_factor: 32.0, original_max_pos: 131072.0 };
        let got = partial_rope(&vin, 3, &la, 8, &l3);
        assert!(close(&got, &want, 1e-5) < 1e-5, "partial_rope {:?}", got);
    }

    #[test]
    fn t_classify_tensor() {
        use TensorRole::*;
        let cases = [
            ("model.language_model.layers.3.moe.gate_proj.weight", ExpertNvfp4Weight),
            ("model.language_model.layers.3.moe.down_proj.weight_scale", ExpertNvfp4Scale),
            ("model.language_model.layers.3.moe.up_proj.weight_scale_2", ExpertNvfp4Scale),
            ("model.language_model.layers.3.moe.gate_proj.input_scale", IgnoreInputScale),
            ("model.language_model.layers.3.moe.gate.weight", Bf16ToQ8), // router gate matmul
            ("model.language_model.layers.3.moe.router_bias", RouterBias),
            ("model.language_model.layers.3.self_attn.q_proj.weight", Bf16ToQ8),
            ("model.language_model.layers.3.self_attn.g_proj.weight", Bf16ToQ8),
            ("model.language_model.layers.0.self_attn.k_proj.k_scale", KvScale),
            ("model.language_model.layers.0.self_attn.q_norm.weight", NormF16),
            ("model.language_model.layers.0.input_layernorm.weight", NormF16),
            ("model.language_model.layers.3.share_expert.down_proj.weight", Bf16ToQ8),
            ("model.language_model.layers.0.mlp.gate_proj.weight", Bf16ToQ8), // dense L0
            ("model.language_model.embed_tokens.weight", EmbedF16),
            ("lm_head.weight", LmHeadF16),
            ("model.language_model.norm.weight", NormF16),
            ("model.vision_model.conv1.weight", Drop),
            ("model.vit_large_projector.weight", Drop),
            ("model.language_model.layers.45.self_attn.q_proj.weight", Drop), // MTP head
        ];
        for (n, want) in cases {
            assert_eq!(classify_tensor(n), want, "classify({n})");
        }
    }

    #[test]
    fn t_config_parse() {
        // sanity: the parser reads the nested text_config shape we validated.
        let j = serde_json::json!({
            "tie_word_embeddings": false,
            "text_config": {
                "hidden_size": 4096, "num_hidden_layers": 45, "vocab_size": 128896,
                "rms_norm_eps": 1e-5, "head_dim": 128, "num_attention_groups": 8,
                "num_attention_heads": 64, "intermediate_size": 11264,
                "moe_num_experts": 212, "moe_top_k": 8, "moe_intermediate_size": 1280,
                "share_expert_dim": 1280, "moe_router_scaling_factor": 3.0,
                "use_moe_router_bias": true, "moe_layers_enum": "3,4,5",
                "sliding_window": 512,
                "attention_other_setting": {"num_attention_heads": 96},
                "layer_types": ["full_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
                "partial_rotary_factors": [0.5,1.0,1.0,1.0,0.5],
                "rope_theta": [5000000.0,10000.0,10000.0,10000.0,5000000.0],
                "swiglu_limits": [0.0,0.0,0.0,0.0,0.0],
                "swiglu_limits_shared": [0.0,0.0,0.0,0.0,0.0],
                "yarn_only_types": ["full_attention"],
                "rope_parameters": {"factor":2.0,"low_freq_factor":1.0,"high_freq_factor":32.0,"original_max_position_embeddings":131072}
            }
        });
        let mut c = Step3p7Config::from_json(&j).unwrap();
        c.num_hidden_layers = 5; // the toy layer_types is length 5
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_experts, 212);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_first_layer, 3);
        assert_eq!(c.layers[0].kind, AttnKind::Full);
        assert_eq!(c.layers[0].num_heads, 64);
        assert_eq!(c.layers[1].kind, AttnKind::Sliding);
        assert_eq!(c.layers[1].num_heads, 96);
        assert_eq!(c.layers[1].sliding_window, Some(512));
        assert!(!c.is_moe_layer(2) && c.is_moe_layer(3));
    }

    #[test]
    fn t_real_config_parses() {
        // Parse the REAL Step-3.7 config.json (nested text_config, 48-len per-layer
        // arrays with num_hidden_layers=45 → the 3 MTP heads are dropped) if it is
        // available next to the mini shards. Verifies the llama3-scale gate: full
        // layers scaled, sliding layers plain.
        let dir = match std::env::var("VLLM_TEST_STEP37_DIR") {
            Ok(d) => d,
            Err(_) => { eprintln!("VLLM_TEST_STEP37_DIR unset — skipping real-config parse"); return; }
        };
        let txt = std::fs::read_to_string(format!("{dir}/config.json")).expect("read config.json");
        let j: Value = serde_json::from_str(&txt).expect("parse config.json");
        let c = Step3p7Config::from_json(&j).expect("from_json");
        assert_eq!(c.num_hidden_layers, 45, "MTP heads must be dropped (48→45)");
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_experts, 212);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_first_layer, 3);
        assert_eq!(c.router_scaling_factor, 3.0);
        assert!(c.use_router_bias);
        assert_eq!(c.layers.len(), 45);
        // L0 full-attn 64 heads llama3-scaled; L1 sliding 96 heads plain.
        assert_eq!(c.layers[0].kind, AttnKind::Full);
        assert_eq!(c.layers[0].num_heads, 64);
        assert!(c.layers[0].use_llama3_scale, "full layer must be llama3-scaled");
        assert_eq!(c.layers[1].kind, AttnKind::Sliding);
        assert_eq!(c.layers[1].num_heads, 96);
        assert!(!c.layers[1].use_llama3_scale, "sliding layer must be PLAIN rope");
        // clamped SwiGLU on 43/44 only.
        assert_eq!(c.swiglu_limit_expert[43], 7.0);
        assert_eq!(c.swiglu_limit_expert[44], 7.0);
        assert_eq!(c.swiglu_limit_shared[43], 16.0);
        assert_eq!(c.swiglu_limit_expert[42], 0.0);
        eprintln!("real config parse OK: 45L, 212 experts top-8, full/sliding rope gate correct");
    }

    // Cosine + argmax + max_abs vs a raw-LE-f32 oracle dump.
    fn read_r0(path: &str) -> Vec<f32> {
        let b = std::fs::read(path).unwrap_or_else(|_| panic!("read {path}"));
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            dot += x as f64 * y as f64;
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-30)
    }
    fn argmax(a: &[f32]) -> usize {
        a.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0
    }

    /// ASSEMBLED-FORWARD GATE: load the mini subset THROUGH the real lib.rs loader
    /// (`Step3p7Model::load_cpu`), run the multi-layer CPU forward, and compare every
    /// stage's hidden to the torch oracle dumps (`oracle.<stage>.r0`, from
    /// `oracle.py`). Gate: cos>=0.999 (dense L0-2) / >=0.99 (MoE L3-4 + final) +
    /// per-position argmax-exact + report max_abs_err. Set VLLM_TEST_STEP37_DIR to
    /// the mini dir (needs the ~6.6GB range-fetched shards; run `oracle.py` first).
    #[test]
    fn step3p7_assembled_forward_gate() {
        let dir = match std::env::var("VLLM_TEST_STEP37_DIR") {
            Ok(d) => d,
            Err(_) => { eprintln!("VLLM_TEST_STEP37_DIR unset — skipping assembled-forward gate"); return; }
        };
        let path = std::path::Path::new(&dir);
        let j: Value = serde_json::from_str(
            &std::fs::read_to_string(path.join("config.json")).expect("config.json")
        ).expect("parse config");
        let cfg = Step3p7Config::from_json(&j).expect("from_json");
        let num_layers = 5usize; // mini subset = L0-4 (dense 0-2 + MoE 3-4)
        // Load THROUGH the real product path.
        let model = Step3p7Model::load_cpu(path, cfg.clone(), 0, num_layers, false)
            .expect("Step3p7Model::load_cpu");
        assert_eq!(model.weights.layers.len(), num_layers);
        let tokens: Vec<u32> = vec![1, 100, 2000, 5, 42, 128, 9, 7];
        let rust_prefix = format!("{dir}/rust");
        model.weights.forward_with_stage_dump(&tokens, &cfg, Some(&rust_prefix));

        let stages: Vec<(String, f64)> = {
            let mut s = vec![("embed".to_string(), 0.999), ("layer0".into(), 0.999),
                             ("layer1".into(), 0.999), ("layer2".into(), 0.999),
                             ("layer3".into(), 0.99), ("layer4".into(), 0.99),
                             ("final_norm".into(), 0.99)];
            s.sort_by(|a, b| a.0.cmp(&b.0)); s
        };
        let hs = cfg.hidden_size;
        let seq = tokens.len();
        let mut worst_cos = 1.0f64;
        let mut all_argmax_ok = true;
        for (stage, thr) in &stages {
            let oref = read_r0(&format!("{dir}/oracle.{stage}.r0"));
            let rref = read_r0(&format!("{rust_prefix}.{stage}.r0"));
            assert_eq!(oref.len(), rref.len(), "stage {stage} len mismatch {} vs {}", oref.len(), rref.len());
            let c = cos(&oref, &rref);
            let maxad = oref.iter().zip(&rref).map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
            // per-position argmax over hidden rows (proxy for the untied lm_head argmax).
            let mut argmax_ok = true;
            for t in 0..seq {
                let oa = argmax(&oref[t * hs..(t + 1) * hs]);
                let ra = argmax(&rref[t * hs..(t + 1) * hs]);
                if oa != ra { argmax_ok = false; }
            }
            all_argmax_ok &= argmax_ok;
            worst_cos = worst_cos.min(c);
            eprintln!("[gate] {stage:11} cos={c:.6} max_abs_err={maxad:.3e} argmax-exact={argmax_ok} (thr {thr})");
            assert!(c >= *thr, "stage {stage}: cos {c:.6} < {thr}");
        }
        assert!(all_argmax_ok, "some stage had a per-position argmax mismatch");
        eprintln!("STEP3P7 ASSEMBLED-FORWARD GATE PASS: worst cos {worst_cos:.6}, argmax-exact all stages");
    }

    /// REAP-PRUNE COHERENCE SMOKE: on the REAL loaded MoE layers, confirm the router
    /// over the 212 KEPT experts is coherent across many tokens — every selected
    /// index is a valid kept expert (<212), the ×3.0 bias-corrected weighting is a
    /// sane finite distribution, expert usage is spread (not collapsed onto one), and
    /// the MoE output is finite (no NaN/Inf). Validates the REAP prune didn't break
    /// routing. Broader token coverage than the 2-layer gate.
    #[test]
    fn step3p7_reap_prune_smoke() {
        let dir = match std::env::var("VLLM_TEST_STEP37_DIR") {
            Ok(d) => d,
            Err(_) => { eprintln!("VLLM_TEST_STEP37_DIR unset — skipping REAP smoke"); return; }
        };
        let path = std::path::Path::new(&dir);
        let j: Value = serde_json::from_str(
            &std::fs::read_to_string(path.join("config.json")).expect("config.json")
        ).expect("parse config");
        let cfg = Step3p7Config::from_json(&j).expect("from_json");
        let model = Step3p7Model::load_cpu(path, cfg.clone(), 0, 5, false).expect("load_cpu");
        let hs = cfg.hidden_size;
        // Drive many tokens through the real embed → 3 layers to get realistic
        // MoE-input hidden states, then inspect routing on L3 and L4.
        let toks: Vec<u32> = (0..64u32).map(|i| (i * 1997 + 11) % cfg.vocab_size as u32).collect();
        let seq = toks.len();
        let mut hidden = vec![0.0f32; seq * hs];
        for (t, &tk) in toks.iter().enumerate() {
            hidden[t * hs..(t + 1) * hs]
                .copy_from_slice(&model.weights.embed[tk as usize * hs..(tk as usize + 1) * hs]);
        }
        for li in 0..3 { // run dense layers 0,1,2 to get MoE-input distribution
            hidden = step3p7_layer_forward(&hidden, seq, li, &model.weights.layers[li], &cfg);
        }
        for li in [3usize, 4] {
            let lw = &model.weights.layers[li];
            let m = match &lw.mlp { OwnedMlp::Moe(m) => m, _ => panic!("layer {li} not MoE") };
            let normed = rms_norm_rows_plus1(&hidden, &lw.post_ln, seq, hs, cfg.rms_norm_eps);
            let mut usage = vec![0usize; cfg.num_experts];
            let mut any_nan = false;
            let mut wsum_min = f32::INFINITY;
            let mut wsum_max = f32::NEG_INFINITY;
            for t in 0..seq {
                let ht = &normed[t * hs..(t + 1) * hs];
                let logits = cpu_matmul(ht, &m.router, 1, hs, cfg.num_experts);
                let (idx, w) = bias_router(&logits, &m.bias, cfg.num_experts_per_tok, cfg.router_scaling_factor);
                assert_eq!(idx.len(), cfg.num_experts_per_tok);
                for &e in &idx {
                    assert!(e < cfg.num_experts, "L{li}: selected expert {e} >= 212 (prune broke routing)");
                    usage[e] += 1;
                }
                let ws: f32 = w.iter().sum();
                wsum_min = wsum_min.min(ws); wsum_max = wsum_max.max(ws);
                // weights renormalized then ×3.0 → sum ≈ 3.0.
                assert!((ws - cfg.router_scaling_factor).abs() < 1e-2,
                    "L{li}: routed weight sum {ws} != ×{} scaling", cfg.router_scaling_factor);
                let out = step3p7_moe_token(ht, m, &cfg);
                if out.iter().any(|v| !v.is_finite()) { any_nan = true; }
            }
            let used = usage.iter().filter(|&&u| u > 0).count();
            let maxu = *usage.iter().max().unwrap();
            assert!(!any_nan, "L{li}: MoE output had NaN/Inf");
            // Coverage: over 64 tokens × top-8 = 512 selections, expect a healthy
            // spread (not all mass on a handful of experts).
            assert!(used >= 20, "L{li}: only {used}/212 experts ever selected (degenerate routing)");
            assert!(maxu <= seq, "L{li}: an expert selected more than once per token");
            eprintln!("[REAP smoke] L{li}: {used}/212 experts used over {seq} tokens, \
                       max-expert-load={maxu}, weight-sum∈[{wsum_min:.4},{wsum_max:.4}], finite=OK");
        }
        eprintln!("STEP3P7 REAP-PRUNE SMOKE PASS: pruned 212-expert routing coherent (valid idx, ×3.0 weights, spread, finite)");
    }

    /// OVERFLOW-STREAM GATE (offline correctness proof for
    /// `VLLM_VULKAN_MOE_STREAM_OVERFLOW`): a **streamed** NVFP4 expert produces the
    /// **bit-identical** dequant to the **resident** expert. Builds a resident
    /// `Step3p7Expert` from random packed nibbles + e4m3 group scales, writes those
    /// same bytes into a minimal 3D `[1,out,..]` safetensors shard, streams them back
    /// through the real mmap+`SafeTensors` per-expert slice path, and asserts
    /// `to_bits()` equality — proving overflow-streaming is argmax-exact BY
    /// CONSTRUCTION (only WHEN/WHERE the buffer is allocated differs).
    #[test]
    fn step3p7_moe_stream_overflow_bit_exact() {
        use safetensors::tensor::TensorView;
        use safetensors::{serialize, Dtype};
        let (out, inn, gs) = (8usize, 32usize, 16usize);
        let global = 0.0123f32;
        // Deterministic pseudo-random packed nibbles + e4m3 scale bytes.
        let pb: Vec<u8> = (0..out * inn / 2).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
        let sb: Vec<u8> = (0..out * (inn / gs)).map(|i| ((i * 53 + 7) & 0x7F) as u8).collect();

        let resident = Step3p7Expert {
            store: ExpertStore::Resident { packed: pb.clone(), scale: sb.clone() },
            global, out_f: out, in_f: inn, group_size: gs,
        };

        // Minimal 3D `[E=1, out, ..]` safetensors shard carrying the two siblings.
        let base = "model.language_model.layers.3.moe.gate_proj";
        let tensors = vec![
            (format!("{base}.weight"), TensorView::new(Dtype::U8, vec![1, out, inn / 2], &pb).unwrap()),
            (format!("{base}.weight_scale"), TensorView::new(Dtype::U8, vec![1, out, inn / gs], &sb).unwrap()),
        ];
        let blob = serialize(tensors, &None).expect("serialize safetensors");
        let dir = std::env::temp_dir().join(format!("step3p7_stream_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("model-00001.safetensors");
        std::fs::write(&shard, &blob).unwrap();
        let sp = shard.to_string_lossy().into_owned();
        let streamed = Step3p7Expert {
            store: ExpertStore::Streamed {
                packed_path: sp.clone(),
                scale_path: sp,
                packed_name: format!("{base}.weight"),
                scale_name: format!("{base}.weight_scale"),
                expert: 0,
            },
            global, out_f: out, in_f: inn, group_size: gs,
        };

        let rd = resident.dequant();
        let sd = streamed.dequant();
        assert_eq!(rd.len(), out * inn);
        assert_eq!(rd.len(), sd.len(), "streamed/resident length mismatch");
        for i in 0..rd.len() {
            assert_eq!(rd[i].to_bits(), sd[i].to_bits(), "streamed != resident (bit) at {i}");
        }
        assert!(resident.resident_bytes() > 0, "resident expert charges DRAM");
        assert_eq!(streamed.resident_bytes(), 0, "streamed expert charges no DRAM");
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("STEP3P7 OVERFLOW-STREAM BIT-EXACT PASS: streamed expert == resident to_bits() ({} elems)", rd.len());
    }

    /// Budget accounting: `expert_linear_bytes` for the real Step-3.7 shapes and the
    /// per-MoE-layer all-resident footprint that MOTIVATES streaming (exceeds a node).
    #[test]
    fn step3p7_moe_stream_budget_accounting() {
        // gate/up [1280,4096], down [4096,1280], group 16.
        let gu = MoeStreamCfg::expert_linear_bytes(1280, 4096, 16);
        assert_eq!(gu, 1280 * 4096 / 2 + 1280 * 4096 / 16);
        let dn = MoeStreamCfg::expert_linear_bytes(4096, 1280, 16);
        assert_eq!(dn, gu, "down has the same out*in product");
        // A 212-expert MoE layer's routed footprint ≈ 212·(2·gu + dn) ≈ 1.88 GiB.
        let per_layer = 212u64 * (2 * gu + dn);
        assert!((1.8e9..2.0e9).contains(&(per_layer as f64)), "per-layer {per_layer}");
        // ~7 such layers per PP-6 stage ≈ 13 GiB > a BC-250 node ⟹ streaming REQUIRED.
        assert!(7 * per_layer > 12_000_000_000, "PP stage all-resident must exceed a node");
    }
}
