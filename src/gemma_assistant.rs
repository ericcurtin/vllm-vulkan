// SPDX-License-Identifier: Apache-2.0
//! Gemma4 EAGLE drafter (`gemma4_assistant`) — INC-2 (loader) + INC-3 (CPU
//! forward) of the Gemma4-31B spec-decode Mac-side prep
//! (`scripts/GEMMA31B_SPEC_PLAN.md`).
//!
//! The drafter is a 4-layer Q-only recurrent cross-attender over the
//! TARGET's borrowed K/V (last layer of each attention type) — NOT a
//! mini standalone LM. See the plan doc section 1 for the full derivation
//! from the HF `Gemma4AssistantForCausalLM` / mlx_vlm
//! `Gemma4AssistantDraftModel` / OminiX-MLX `assistant.rs` references.
//!
//! Checkpoint: `gemma-4-31B-it-assistant/model.safetensors`, plain bf16
//! (NOT MLX-quantized) — 48 tensors, 939MB. Loaded host-f32 (mirrors the
//! generic `model::load_weights_from_safetensors` bf16->f32 widen path;
//! at 939MB this easily fits Mac's 103GB, so there is no memory pressure
//! motivating the f16-host trick the GPU-resident g12b path uses).

use std::collections::HashMap;
use std::path::Path;

use pyo3::prelude::*;

use crate::model::{self, cpu_gelu, cpu_matmul, cpu_rms_norm, cpu_rope_with_basis, cpu_sdpa};
use crate::compute;
use crate::{f32_slice_to_bytes, f32_to_f16_bytes, matvec_pc13, matvec_variant, read_f32_buf};

/// Attention-type layer kind for the drafter's 4 layers (mirrors
/// `text_config.layer_types` = `[sliding, sliding, sliding, full]` for the
/// 31B pairing: 3 sliding cross-attn layers borrowing the target's LAST
/// sliding layer (58) K/V, 1 full/global layer borrowing the target's LAST
/// full layer (59) K/V).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistantLayerKind {
    Sliding,
    Full,
}

impl AssistantLayerKind {
    /// Map a checkpoint `layer_type` string to the kind. An unknown value is an
    /// ERROR, never a default: sliding and full layers differ in head_dim,
    /// kv-head count and RoPE theta, so guessing would produce a silently
    /// mis-shaped drafter instead of a load failure.
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "sliding_attention" => Ok(Self::Sliding),
            "full_attention" => Ok(Self::Full),
            other => Err(format!("Unknown gemma4_assistant layer_type '{other}'")),
        }
    }
}

/// `text_config` fields needed by the CPU forward, read from the real
/// `gemma-4-31B-it-assistant/config.json`. Pinned as constants here (rather
/// than a JSON-parsed struct) since this drafter shape is fixed to the one
/// checkpoint we pair with the 31B target; see
/// `AssistantConfig::g31b_pair()` for the citation.
#[derive(Debug, Clone)]
pub struct AssistantConfig {
    pub hidden_size: usize,           // 1024
    pub backbone_hidden_size: usize,  // 5376 (target hidden_size)
    pub num_hidden_layers: usize,     // 4
    pub num_attention_heads: usize,   // 32 (every layer, both types)
    pub num_key_value_heads: usize,   // 16 (sliding)
    pub num_global_key_value_heads: usize, // 4 (full/global)
    pub head_dim: usize,              // 256 (sliding)
    pub global_head_dim: usize,       // 512 (full/global)
    pub intermediate_size: usize,     // 8192
    pub sliding_window: usize,        // 1024
    pub rms_norm_eps: f32,            // 1e-6
    pub vocab_size: usize,            // 262144
    pub layer_types: Vec<AssistantLayerKind>, // [Sliding,Sliding,Sliding,Full]
    /// full_attention RoPE: proportional, theta 1e6, partial_rotary_factor 0.25
    /// -> rotary_dim = head_dim/4, freq_dim = head_dim (see
    /// `model::cpu_rope_with_basis` doc for the proportional-RoPE convention
    /// this mirrors, already validated by the g12b/g31b base-model forward).
    pub full_rope_theta: f32,
    /// sliding_attention RoPE: default, theta 10000, full rotation.
    pub sliding_rope_theta: f32,
}

impl AssistantConfig {
    /// The gemma-4-31B-it drafter pairing (`gemma-4-31B-it-assistant/config.json`,
    /// `text_config`). Q-only 4-layer cross-attender: layers 0-2 sliding
    /// (32q x256hd, 16kv x256hd), layer 3 full/global (32q x512hd, 4kv x512hd).
    pub fn g31b_pair() -> Self {
        AssistantConfig {
            hidden_size: 1024,
            backbone_hidden_size: 5376,
            num_hidden_layers: 4,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            num_global_key_value_heads: 4,
            head_dim: 256,
            global_head_dim: 512,
            intermediate_size: 8192,
            sliding_window: 1024,
            rms_norm_eps: 1e-6,
            vocab_size: 262144,
            // Verbatim from `text_config.layer_types` in the real
            // `gemma-4-31B-it-assistant/config.json`.
            layer_types: ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
                .iter()
                .map(|s| AssistantLayerKind::parse(s).unwrap())
                .collect(),
            full_rope_theta: 1_000_000.0,
            sliding_rope_theta: 10_000.0,
        }
    }

    /// True when this layer is FULL attention rather than sliding-window.
    ///
    /// This one predicate drives head_dim, kv-head count and RoPE theta below,
    /// so a per-layer value must never be read from the model-level fields
    /// directly — the two layer kinds do not share them.
    pub fn is_full(&self, layer_idx: usize) -> bool {
        matches!(self.layer_types[layer_idx], AssistantLayerKind::Full)
    }

    /// Head dim FOR THIS LAYER. Full layers use `global_head_dim`, sliding
    /// layers `head_dim`; the two differ, so KV-cache sizing and RoPE must both
    /// go through here rather than the model-level `head_dim`.
    pub fn layer_head_dim(&self, layer_idx: usize) -> usize {
        if self.is_full(layer_idx) { self.global_head_dim } else { self.head_dim }
    }

    /// KV-head count FOR THIS LAYER. Full layers can be MQA/low-GQA while
    /// sliding layers are not, so a fixed `num_key_value_heads` mis-sizes the
    /// full layers' cache — the exact shape mismatch behind the g12b
    /// 4096-vs-512 KV panic.
    pub fn layer_num_kv_heads(&self, layer_idx: usize) -> usize {
        if self.is_full(layer_idx) { self.num_global_key_value_heads } else { self.num_key_value_heads }
    }

    /// Q projection width FOR THIS LAYER. The query HEAD COUNT is uniform
    /// across layers; only the head dim varies, so this is `num_attention_heads
    /// * layer_head_dim` and NOT a per-layer head count times a fixed dim.
    pub fn layer_q_dim(&self, layer_idx: usize) -> usize {
        self.num_attention_heads * self.layer_head_dim(layer_idx)
    }
}

/// Every tensor name + expected shape the INC-2 load-shape gate checks.
/// Mirrors the real checkpoint dump (48 tensors): embed/norm/pre+post
/// projection, and per-layer input/post_attention/pre_feedforward/
/// post_feedforward norms + layer_scalar + mlp gate/up/down + self_attn
/// q_proj/o_proj/q_norm (NO k_proj/v_proj/k_norm anywhere — the drafter is
/// Q-only, K/V is borrowed from the target).
pub fn expected_tensor_shapes(cfg: &AssistantConfig) -> Vec<(String, Vec<usize>)> {
    let h = cfg.hidden_size;
    let mut v = vec![
        ("model.embed_tokens.weight".to_string(), vec![cfg.vocab_size, h]),
        ("model.norm.weight".to_string(), vec![h]),
        ("pre_projection.weight".to_string(), vec![h, 2 * cfg.backbone_hidden_size]),
        ("post_projection.weight".to_string(), vec![cfg.backbone_hidden_size, h]),
    ];
    for li in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{li}");
        let hd = cfg.layer_head_dim(li);
        let q_dim = cfg.layer_q_dim(li);
        v.push((format!("{p}.input_layernorm.weight"), vec![h]));
        v.push((format!("{p}.post_attention_layernorm.weight"), vec![h]));
        v.push((format!("{p}.pre_feedforward_layernorm.weight"), vec![h]));
        v.push((format!("{p}.post_feedforward_layernorm.weight"), vec![h]));
        v.push((format!("{p}.layer_scalar"), vec![1]));
        v.push((format!("{p}.mlp.gate_proj.weight"), vec![cfg.intermediate_size, h]));
        v.push((format!("{p}.mlp.up_proj.weight"), vec![cfg.intermediate_size, h]));
        v.push((format!("{p}.mlp.down_proj.weight"), vec![h, cfg.intermediate_size]));
        v.push((format!("{p}.self_attn.q_proj.weight"), vec![q_dim, h]));
        v.push((format!("{p}.self_attn.o_proj.weight"), vec![h, q_dim]));
        v.push((format!("{p}.self_attn.q_norm.weight"), vec![hd]));
    }
    v
}

/// Tensor names the drafter must NOT have (Q-only — no k/v_proj, no k_norm).
pub fn forbidden_tensor_names(cfg: &AssistantConfig) -> Vec<String> {
    let mut v = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{li}");
        v.push(format!("{p}.self_attn.k_proj.weight"));
        v.push(format!("{p}.self_attn.v_proj.weight"));
        v.push(format!("{p}.self_attn.k_norm.weight"));
    }
    v
}

/// Load the drafter checkpoint's `model.safetensors` (bf16 -> host f32).
/// The tensor names already land in the loader's `model.*` / bare
/// `pre_projection.weight` / `post_projection.weight` namespace verbatim (no
/// `language_model.` prefix on disk), so the generic bf16 widen loader
/// applies unmodified.
pub fn load_assistant_weights(dir: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    let path = dir.join("model.safetensors");
    if !path.exists() {
        return Err(format!("No model.safetensors in {}", dir.display()));
    }
    model::load_weights_from_safetensors(&path)
}

/// Target K/V borrowed by the drafter, one pair per attention TYPE (not per
/// drafter layer) — the last layer of each type in the TARGET. Layout
/// `[kv_len, n_kv_heads, head_dim]` (matches `cpu_sdpa`'s expected K/V
/// layout directly, already RoPE-rotated + normed by the target; the
/// drafter does not re-rotate it).
pub struct AssistantSharedKv {
    pub sliding_k: Vec<f32>,
    pub sliding_v: Vec<f32>,
    pub full_k: Vec<f32>,
    pub full_v: Vec<f32>,
    pub kv_len: usize,
}

/// Output of one drafter forward step.
pub struct AssistantOutput {
    /// `[vocab_size]` sampling logits from the drafter's OWN norm + tied
    /// lm_head. NOT routed through `post_projection` or the target's norm/
    /// lm_head/softcap (drafter `final_logit_softcapping` is null).
    pub logits: Vec<f32>,
    /// `[backbone_hidden_size]` — `post_projection` output, fed back as the
    /// next step's `recurrent_hidden` half of `inputs_embeds`.
    pub post_projection_out: Vec<f32>,
}

/// Build `inputs_embeds = concat([prev_token_embed, recurrent_hidden])`,
/// **embed FIRST** — reversing this order collapses acceptance 0.43->0.01
/// (assistant.rs `build_inputs_embeds` / GEMMA31B_SPEC_PLAN.md risk #5).
/// Both halves are `backbone_hidden_size`-wide (5376); `prev_token_embed` is
/// the TARGET's `embed_tokens(prev_token) * target.embed_scale`, NOT the
/// drafter's own (1024-dim) embedding table.
pub fn build_inputs_embeds(prev_token_embed: &[f32], recurrent_hidden: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(prev_token_embed.len() + recurrent_hidden.len());
    v.extend_from_slice(prev_token_embed);
    v.extend_from_slice(recurrent_hidden);
    v
}

/// GPU-resident matvec store for the EAGLE drafter (`VLLM_VULKAN_ASSISTANT_GPU`,
/// default ON). Owns its OWN `ComputeEngine` + f16 weight buffers, independent
/// of the target `VulkanModel` — the drafter is a standalone rank0-only pyclass
/// (`GemmaAssistant`) with no `VulkanModel` handle, so it cannot borrow the
/// target's engine; a self-contained engine is the drafter analogue of the qwen
/// `mtp_upload_dense_gpu` pattern (which could upload into the target's own
/// `gpu_weights` because the MTP head lives inside `VulkanModel`). Only the
/// drafter's dense matvec projections are uploaded here (pre/post projection,
/// per-layer q/o + gate/up/down, tied lm-head=embed_tokens); the tiny norms,
/// layer_scalar, RoPE, and the Q-only cross-attention over the borrowed target
/// KV stay on the host, byte-for-byte identical to the CPU reference path.
pub struct AssistantGpu {
    engine: compute::ComputeEngine,
    /// weight name -> its f16 GPU buffer, `[n, k]` row-major (the SAME layout
    /// `cpu_matmul(x, w, 1, k, n)` reads, so the GPU matvec is a drop-in).
    weights_f16: HashMap<String, compute::Buffer>,
}

impl AssistantGpu {
    /// The drafter weight names whose matvec moves to the GPU (everything a
    /// draft step feeds through `cpu_matmul` in [`assistant_forward`]).
    fn matvec_weight_names(cfg: &AssistantConfig) -> Vec<String> {
        let mut v = vec![
            "pre_projection.weight".to_string(),
            "post_projection.weight".to_string(),
            "model.embed_tokens.weight".to_string(),
        ];
        for li in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{li}");
            v.push(format!("{p}.self_attn.q_proj.weight"));
            v.push(format!("{p}.self_attn.o_proj.weight"));
            v.push(format!("{p}.mlp.gate_proj.weight"));
            v.push(format!("{p}.mlp.up_proj.weight"));
            v.push(format!("{p}.mlp.down_proj.weight"));
        }
        v
    }

    /// Build the engine on `device_idx` and upload the drafter's matvec weights
    /// f16 ONCE. Returns `None` (→ host fallback) if no Vulkan device / engine is
    /// available (e.g. Mac build) or any upload fails, so the drafter degrades to
    /// the bit-exact CPU path rather than erroring.
    fn build(cfg: &AssistantConfig, weights: &HashMap<String, Vec<f32>>, device_idx: usize) -> Option<Self> {
        let dev = match crate::device::ComputeDevice::create(device_idx) {
            Ok(d) => d,
            Err(e) => { log::warn!("gemma assistant GPU: no Vulkan device {device_idx}: {e}; host-f32 drafter"); return None; }
        };
        let shader_spvs = crate::include_all_shaders();
        let refs: HashMap<&str, &[u8]> = shader_spvs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let mut engine = match compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, dev.caps(), &refs,
        ) {
            Ok(e) => e,
            Err(e) => { log::warn!("gemma assistant GPU: engine init failed: {e}; host-f32 drafter"); return None; }
        };

        let mut weights_f16 = HashMap::new();
        let mut bytes_total = 0u64;
        for name in Self::matvec_weight_names(cfg) {
            let w = weights.get(&name)?;
            let bytes = f32_to_f16_bytes(w);
            let buf = match engine.alloc_host_coherent_storage(bytes.len() as u64) {
                Ok(b) => b,
                Err(e) => { log::warn!("gemma assistant GPU: alloc '{name}' failed: {e}; host-f32 drafter"); return None; }
            };
            if let Err(e) = buf.write(&bytes) {
                log::warn!("gemma assistant GPU: upload '{name}' failed: {e}; host-f32 drafter");
                return None;
            }
            bytes_total += bytes.len() as u64;
            weights_f16.insert(name, buf);
        }
        log::info!(
            "gemma assistant: EAGLE drafter GPU-resident ({} f16 matvec weights, {:.2} GB) — per-step draft matvecs on GPU",
            weights_f16.len(), bytes_total as f64 / 1e9);
        Some(Self { engine, weights_f16 })
    }

    /// One f16 matvec `[1,n] = x[1,k] @ W[n,k]^T` on the GPU — a verbatim mirror
    /// of `VulkanModel::gemma_matvec`'s GPU branch (the SAME `matvec_variant`
    /// f16 shader the target's own forward uses). Returns `None` when the weight
    /// was not uploaded, so the caller falls back to host `cpu_matmul`. NOTE:
    /// GPU f16 ≠ host f32 bit-for-bit; drafts may shift vs the f32-host path, but
    /// the verify step guarantees greedy-exactness (see `VLLM_VULKAN_ASSISTANT_GPU=0`
    /// for the bit-exact host A/B path).
    fn mv(&mut self, name: &str, x: &[f32], k: usize, n: usize) -> Option<Vec<f32>> {
        let w_ptr = self.weights_f16.get(name)? as *const compute::Buffer;
        let eng = &mut self.engine;
        let xb = f32_slice_to_bytes(x);
        let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).ok()?;
        inp.write(&xb).ok()?;
        let out = eng.alloc_host_coherent_storage((n * 4) as u64).ok()?;
        let inp_p = &inp as *const compute::Buffer;
        let out_p = &out as *const compute::Buffer;
        let (shader, r) = matvec_variant(true, n);
        let wg = (n as u32 + r - 1) / r;
        let pc = matvec_pc13(k, n);
        let cb = eng.begin_batch().ok()?;
        unsafe {
            eng.record_to(cb, &shader, &[&*w_ptr, &*inp_p, &*out_p], &pc, (wg, 1, 1)).ok()?;
        }
        eng.submit_batch(cb).ok()?;
        let result = read_f32_buf(&out, n);
        eng.return_to_pool(inp);
        eng.return_to_pool(out);
        Some(result)
    }
}

/// One drafter matvec: GPU (f16-resident) when `gpu` holds the weight, else the
/// bit-exact host `cpu_matmul`. Kept as a free fn (not a closure) so `gpu` is
/// re-borrowed per call rather than moved into the closure for the whole forward.
fn amv(
    gpu: &mut Option<&mut AssistantGpu>,
    weights: &HashMap<String, Vec<f32>>,
    name: &str,
    x: &[f32],
    k: usize,
    n: usize,
) -> Vec<f32> {
    if let Some(g) = gpu.as_deref_mut() {
        if let Some(r) = g.mv(name, x, k, n) {
            return r;
        }
    }
    let w = weights.get(name).unwrap_or_else(|| panic!("assistant weight '{name}' not found"));
    cpu_matmul(x, w, 1, k, n)
}

/// One drafter forward step.
///
/// `gpu`: `Some` → the dense matvec projections dispatch to the GPU
/// (`VLLM_VULKAN_ASSISTANT_GPU`, default ON); `None` → the bit-exact host CPU
/// reference (the golden-test path and the flag-off A/B fallback).
///
/// `inputs_embeds`: `[2*backbone_hidden_size]`, from [`build_inputs_embeds`].
/// `position_offset`: absolute RoPE position of this (single) query token —
/// held CONSTANT across all K draft-loop steps (GEMMA31B_SPEC_PLAN.md
/// section 1.4).  `shared_kv`: the target's borrowed K/V, a FIXED snapshot
/// across the K-step loop (the drafter never appends its own K/V — it has
/// none).
pub fn assistant_forward(
    cfg: &AssistantConfig,
    weights: &HashMap<String, Vec<f32>>,
    inputs_embeds: &[f32],
    position_offset: usize,
    shared_kv: &AssistantSharedKv,
    mut gpu: Option<&mut AssistantGpu>,
) -> AssistantOutput {
    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let w = |name: &str| -> &[f32] {
        weights.get(name).unwrap_or_else(|| panic!("assistant weight '{name}' not found"))
    };

    // pre_projection: [2*backbone_hidden] -> [hidden_size]
    let mut hidden = amv(&mut gpu, weights, "pre_projection.weight", inputs_embeds, 2 * cfg.backbone_hidden_size, h);

    for layer_idx in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{layer_idx}");
        let is_full = cfg.is_full(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let num_q = cfg.num_attention_heads;
        let num_kv = cfg.layer_num_kv_heads(layer_idx);
        let q_dim = num_q * head_dim;

        let residual = hidden.clone();

        // input_layernorm -> q_proj -> q_norm (per head) -> RoPE (own Q only;
        // borrowed K is NOT re-rotated, it's already RoPE'd by the target).
        let x = cpu_rms_norm(&hidden, w(&format!("{p}.input_layernorm.weight")), eps);
        let mut q = amv(&mut gpu, weights, &format!("{p}.self_attn.q_proj.weight"), &x, h, q_dim);

        let q_norm_w = w(&format!("{p}.self_attn.q_norm.weight"));
        for hi in 0..num_q {
            let s = &mut q[hi * head_dim..(hi + 1) * head_dim];
            let n = cpu_rms_norm(s, q_norm_w, eps);
            s.copy_from_slice(&n);
        }

        // Own-Q RoPE only: pass an empty K slice with num_kv_heads=0 so
        // `cpu_rope_with_basis` rotates Q and leaves K entirely untouched
        // (there is no drafter-owned K to rotate — the borrowed K is
        // pre-rotated by the target).
        let (theta, rotary_dim, freq_dim) = if is_full {
            // Proportional RoPE (partial_rotary_factor 0.25): frequency basis
            // is the FULL global head_dim (512) even though only head_dim/4
            // (128) dims rotate — see `cpu_rope_with_basis` doc comment.
            (cfg.full_rope_theta, head_dim / 4, head_dim)
        } else {
            (cfg.sliding_rope_theta, head_dim, head_dim)
        };
        let mut no_k: [f32; 0] = [];
        cpu_rope_with_basis(&mut q, &mut no_k, position_offset, num_q, 0, head_dim, rotary_dim, freq_dim, theta);

        // Bidirectional cross-attention over the borrowed K/V, scale=1.0
        // (the target's QK-norm convention — NOT 1/sqrt(head_dim); see
        // GEMMA31B_SPEC_PLAN.md section 1.2 / assistant.rs L598-608).
        // Sliding layers additionally restrict to the last `sliding_window`
        // KV positions once kv_len exceeds it (`cpu_sdpa`'s `sliding_window`
        // param already implements exactly this truncation).
        let (sk, sv): (&[f32], &[f32]) = if is_full {
            (&shared_kv.full_k, &shared_kv.full_v)
        } else {
            (&shared_kv.sliding_k, &shared_kv.sliding_v)
        };
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        let attn_out = cpu_sdpa(&q, sk, sv, num_q, num_kv, head_dim, shared_kv.kv_len, 1.0, window);

        let o = amv(&mut gpu, weights, &format!("{p}.self_attn.o_proj.weight"), &attn_out, q_dim, h);
        let pa_normed = cpu_rms_norm(&o, w(&format!("{p}.post_attention_layernorm.weight")), eps);
        let mut hidden2: Vec<f32> = residual.iter().zip(pa_normed.iter()).map(|(&r, &a)| r + a).collect();
        let residual2 = hidden2.clone();

        let ff_in = cpu_rms_norm(&hidden2, w(&format!("{p}.pre_feedforward_layernorm.weight")), eps);
        let gate = amv(&mut gpu, weights, &format!("{p}.mlp.gate_proj.weight"), &ff_in, h, cfg.intermediate_size);
        let up = amv(&mut gpu, weights, &format!("{p}.mlp.up_proj.weight"), &ff_in, h, cfg.intermediate_size);
        let gate_act = cpu_gelu(&gate);
        let mid: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
        let down = amv(&mut gpu, weights, &format!("{p}.mlp.down_proj.weight"), &mid, cfg.intermediate_size, h);
        let ff_normed = cpu_rms_norm(&down, w(&format!("{p}.post_feedforward_layernorm.weight")), eps);
        hidden2.iter_mut().zip(ff_normed.iter()).for_each(|(hv, &f)| *hv += f);
        let _ = residual2; // consumed via hidden2's residual add above

        // layer_scalar applied at the very end (matches the target's own
        // DecoderLayer + assistant.rs).
        let layer_scalar = w(&format!("{p}.layer_scalar"))[0];
        hidden2.iter_mut().for_each(|v| *v *= layer_scalar);

        hidden = hidden2;
    }

    let inner = cpu_rms_norm(&hidden, w("model.norm.weight"), eps);

    // post_projection: SEPARATE output (next step's recurrent_hidden), NOT
    // on the logit path.
    let post_projection_out = amv(&mut gpu, weights, "post_projection.weight", &inner, h, cfg.backbone_hidden_size);

    // Tied lm_head over the drafter's OWN embed_tokens — NOT the target's
    // norm/lm_head, NO softcap (drafter final_logit_softcapping is null).
    let logits = amv(&mut gpu, weights, "model.embed_tokens.weight", &inner, h, cfg.vocab_size);

    AssistantOutput { logits, post_projection_out }
}

/// INC-5b piece 3 — pyo3 wrapper around the CPU `assistant_forward` drafter,
/// loaded/held independently of `VulkanModel` (the drafter is tiny — 0.5B,
/// 939MB host-f32 — and per `GEMMA31B_SPEC_PLAN.md` §1.5 runs on rank0 ONLY,
/// so it needs no TP sharding/GPU residency; a standalone pyclass avoids
/// adding drafter fields to every one of `VulkanModel`'s construction sites).
/// The driver (`scripts/tp_gemma_spec.py`) constructs ONE `GemmaAssistant` on
/// rank0, then calls `.forward(...)` once per draft step, feeding it the
/// TARGET's replicated embed/hidden and the all-gathered borrowed-KV
/// snapshot (`VulkanModel.gemma_kv_layer`).
#[pyclass]
pub struct GemmaAssistant {
    cfg: AssistantConfig,
    weights: HashMap<String, Vec<f32>>,
    /// `Some` when the drafter's dense matvecs run on the GPU
    /// (`VLLM_VULKAN_ASSISTANT_GPU`, default ON, engine available); `None` keeps
    /// the bit-exact host-f32 path (flag off, or Mac/engine-less). The host
    /// `weights` copy is always retained as the A/B fallback.
    gpu: Option<AssistantGpu>,
}

#[pymethods]
impl GemmaAssistant {
    /// Load the drafter checkpoint's `model.safetensors` from `dir`
    /// (bf16->f32 widen, 939MB — see `load_assistant_weights`) and pin the
    /// 31B pairing's config (`AssistantConfig::g31b_pair`).
    #[pyo3(signature = (dir, device_idx = 0))]
    #[new]
    fn new(dir: String, device_idx: usize) -> PyResult<Self> {
        let weights = load_assistant_weights(std::path::Path::new(&dir))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let cfg = AssistantConfig::g31b_pair();
        // GPU-resident drafter matvecs, default ON (`VLLM_VULKAN_ASSISTANT_GPU=0`
        // → bit-exact host-f32 A/B path). Mirrors the qwen dense-MTP fix: kills
        // the single-threaded host `cpu_matmul` per draft step that made the
        // gate-3 spec arm drafter-overhead-bound. Engine-less builds (Mac) fall
        // back to host automatically inside `AssistantGpu::build`.
        let want_gpu = crate::flags::flags_global().assistant_gpu;
        let gpu = if want_gpu { AssistantGpu::build(&cfg, &weights, device_idx) } else { None };
        Ok(Self { cfg, weights, gpu })
    }

    /// One drafter forward step (INC-3's `assistant_forward`, CPU reference —
    /// the drafter is 4 tiny layers, trivial compute even on CPU per the
    /// plan's own risk assessment). `prev_token_embed`/`recurrent_hidden` are
    /// both `[backbone_hidden_size]` (5376); `position_offset` is the FIXED
    /// RoPE position held constant across the whole K-step draft loop
    /// (`GEMMA31B_SPEC_PLAN.md` §1.4); `sliding_k`/`sliding_v`/`full_k`/
    /// `full_v` are the all-gathered borrowed-KV snapshot
    /// (`[kv_len, n_kv_heads, head_dim]` flat, already RoPE-rotated+normed by
    /// the target — NOT re-rotated here); `kv_len` is the FIXED snapshot
    /// length for this whole draft cycle (does not grow across K steps).
    /// Returns `(logits, post_projection_out)` — `post_projection_out`
    /// becomes the next step's `recurrent_hidden`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        prev_token_embed: Vec<f32>,
        recurrent_hidden: Vec<f32>,
        position_offset: usize,
        sliding_k: Vec<f32>,
        sliding_v: Vec<f32>,
        full_k: Vec<f32>,
        full_v: Vec<f32>,
        kv_len: usize,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let inputs_embeds = build_inputs_embeds(&prev_token_embed, &recurrent_hidden);
        let shared_kv = AssistantSharedKv { sliding_k, sliding_v, full_k, full_v, kv_len };
        let out = assistant_forward(
            &self.cfg, &self.weights, &inputs_embeds, position_offset, &shared_kv, self.gpu.as_mut());
        Ok((out.logits, out.post_projection_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Checkpoint directory for the 939MB bf16 drafter, from the environment
    /// ONLY — the same env-gated-skip discipline as `laguna_loader.rs`'s
    /// `VLLM_TEST_LAGUNA_SHARD`. There is deliberately NO default path: the
    /// previous `$HOME/repos/OminiX-MLX/models/...` fallback made these tests
    /// real on exactly one developer machine and a silent no-op pass
    /// everywhere else, including CI.
    ///
    ///   VLLM_TEST_GEMMA31B_ASSISTANT_DIR=<checkpoint dir> cargo test ... -- --nocapture
    ///
    /// `GEMMA31B_ASSISTANT_DIR` stays accepted for the existing scripts.
    fn assistant_dir() -> Option<std::path::PathBuf> {
        for var in ["VLLM_TEST_GEMMA31B_ASSISTANT_DIR", "GEMMA31B_ASSISTANT_DIR"] {
            if let Ok(d) = std::env::var(var) {
                let p = std::path::PathBuf::from(d);
                assert!(
                    p.join("model.safetensors").exists(),
                    "{var}={} does not contain model.safetensors — the checkpoint path is \
                     wrong; refusing to silently skip a test that was explicitly requested",
                    p.display());
                return Some(p);
            }
        }
        None
    }

    /// One place to print a SKIP so an un-run test is visible in the log
    /// rather than an indistinguishable green `ok`.
    fn skip(test: &str, why: &str) {
        eprintln!("SKIP {test}: {why} — set VLLM_TEST_GEMMA31B_ASSISTANT_DIR to run it");
    }

    /// INC-2 gate: load the REAL 939MB bf16 drafter checkpoint and assert
    /// every one of the 48 expected tensors is present with the exact shape
    /// the plan's tensor map specifies, and that no k_proj/v_proj/k_norm
    /// tensor exists anywhere (the drafter is Q-only).
    #[test]
    fn gemma31b_assistant_load_shapes() {
        let dir = match assistant_dir() {
            Some(d) => d,
            None => { skip("gemma31b_assistant_load_shapes", "checkpoint dir not configured"); return; }
        };
        let weights = load_assistant_weights(&dir).expect("load assistant checkpoint");
        let cfg = AssistantConfig::g31b_pair();

        let expected = expected_tensor_shapes(&cfg);
        assert_eq!(expected.len(), 48, "expected 48 tensors in the tensor map, got {}", expected.len());

        for (name, shape) in &expected {
            let v = weights.get(name).unwrap_or_else(|| panic!("missing tensor '{name}'"));
            let numel: usize = shape.iter().product();
            assert_eq!(v.len(), numel, "shape mismatch for '{name}': expected {shape:?} ({numel} elems), got {} elems", v.len());
        }

        let forbidden = forbidden_tensor_names(&cfg);
        for name in &forbidden {
            assert!(!weights.contains_key(name), "drafter is Q-only: unexpected tensor '{name}' present");
        }

        // No unexpected extra tensors beyond the 48-tensor map.
        let expected_names: HashSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
        let extra: Vec<&String> = weights.keys().filter(|k| !expected_names.contains(k.as_str())).collect();
        assert!(extra.is_empty(), "unexpected extra tensors in checkpoint: {extra:?}");
    }

    /// Minimal .npy reader for the golden fixtures. Checks the dtype in the
    /// header: the body is decoded as little-endian f32, so a fixture
    /// regenerated as f64/f16/bf16 (or big-endian) would otherwise be read as
    /// garbage and only show up as a mysterious cosine failure — or, worse,
    /// pass by accident on a short array.
    fn read_npy_f32(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert!(bytes.len() >= 10, "truncated .npy: {}", path.display());
        assert_eq!(&bytes[0..6], b"\x93NUMPY", "not a .npy: {}", path.display());
        // v1.x: 2-byte little-endian header length at offset 8; v2.x: 4-byte.
        let (header_len, body) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
            2 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12usize),
            v => panic!("unsupported .npy major version {v}: {}", path.display()),
        };
        let header = std::str::from_utf8(&bytes[body..body + header_len])
            .unwrap_or_else(|e| panic!("non-utf8 .npy header in {}: {e}", path.display()));
        // '<f4' little-endian f32; '|'/'=' would be native-order, which on every
        // target this builds for is also little-endian f32 for a 4-byte float.
        assert!(header.contains("'descr': '<f4'") || header.contains("'descr': '=f4'"),
            "{} is not little-endian float32 — header: {header}", path.display());
        assert!(!header.contains("'fortran_order': True"),
            "{} is Fortran-ordered; this reader assumes C order", path.display());
        let data = &bytes[body + header_len..];
        assert_eq!(data.len() % 4, 0, "{}: body is not a whole number of f32", path.display());
        data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Cosine similarity, accumulated in f64 so the gate's own reduction error
    /// stays well under the tolerance it asserts on.
    ///
    /// Returns NaN on an all-zero input (0/0), and that is deliberate — a
    /// silent 0.0 there would read as "totally dissimilar" for a degenerate
    /// tensor, whereas NaN fails any `>= threshold` comparison loudly. Do not
    /// "fix" it with a zero guard.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
        for i in 0..a.len() {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    /// First-maximum argmax with a STRICT `>` comparison, so ties keep the
    /// LOWEST index — the same tie-break the python driver and
    /// `model::argmax` use. A `>=` here would pick the last of a tied run and
    /// break token-for-token equality with the reference on flat logits.
    fn argmax(x: &[f32]) -> usize {
        x.iter().enumerate().fold((0, f32::MIN), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0
    }

    /// INC-3 GOLDEN GATE: run the CPU drafter forward over a tiny fixed
    /// input (synthetic prev-token-embed/recurrent-hidden/borrowed-KV, REAL
    /// checkpoint weights) and compare against the golden dumped by
    /// `scripts/gen_gemma31b_assistant_golden.py` (upstream mlx_vlm
    /// `Attention`/`DecoderLayer` classes driven the same way
    /// `Gemma4AssistantDraftModel.__call__` does). Locks embed-first concat
    /// order + scale=1.0 + the KV-borrow (no re-rotation) — GEMMA31B_SPEC_PLAN.md
    /// risk #5's silent-failure knobs.
    ///   GEMMA31B_ASSISTANT_DIR=<checkpoint dir>  GEMMA31B_GOLDEN=<fixture dir>
    #[test]
    fn gemma31b_assistant_forward_cos_vs_golden() {
        let dir = match assistant_dir() {
            Some(d) => d,
            None => { skip("gemma31b_assistant_forward_cos_vs_golden", "checkpoint dir not configured"); return; }
        };
        let golden_explicit = std::env::var("GEMMA31B_GOLDEN").ok();
        let golden = if let Some(d) = golden_explicit.as_ref() {
            std::path::PathBuf::from(d)
        } else {
            // In-repo fixture generated by scripts/gen_gemma31b_assistant_golden.py
            // (committed alongside this test — see tests/fixtures/gemma31b_assistant_golden/).
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/gemma31b_assistant_golden")
        };
        if !golden.join("logits.npy").exists() {
            // An EXPLICIT GEMMA31B_GOLDEN that does not resolve is a
            // misconfiguration, not a reason to skip. The in-repo fixture
            // (tests/fixtures/gemma31b_assistant_golden/) is committed, so the
            // default branch reaching here means the checkout is incomplete.
            assert!(golden_explicit.is_none(),
                "GEMMA31B_GOLDEN={} has no logits.npy", golden.display());
            skip("gemma31b_assistant_forward_cos_vs_golden",
                 &format!("golden fixture not found at {}", golden.display()));
            return;
        }

        let weights = load_assistant_weights(&dir).expect("load assistant checkpoint");
        let cfg = AssistantConfig::g31b_pair();

        let prev_token_embed = read_npy_f32(&golden.join("prev_token_embed.npy"));
        let recurrent_hidden = read_npy_f32(&golden.join("recurrent_hidden.npy"));
        let sliding_k = read_npy_f32(&golden.join("sliding_k.npy"));
        let sliding_v = read_npy_f32(&golden.join("sliding_v.npy"));
        let full_k = read_npy_f32(&golden.join("full_k.npy"));
        let full_v = read_npy_f32(&golden.join("full_v.npy"));
        let golden_logits = read_npy_f32(&golden.join("logits.npy"));
        let golden_post_proj = read_npy_f32(&golden.join("post_proj.npy"));

        assert_eq!(prev_token_embed.len(), cfg.backbone_hidden_size);
        assert_eq!(recurrent_hidden.len(), cfg.backbone_hidden_size);
        let kv_len = 4usize;
        assert_eq!(sliding_k.len(), kv_len * cfg.num_key_value_heads * cfg.head_dim);
        assert_eq!(full_k.len(), kv_len * cfg.num_global_key_value_heads * cfg.global_head_dim);

        let shared_kv = AssistantSharedKv { sliding_k, sliding_v, full_k, full_v, kv_len };
        let inputs_embeds = build_inputs_embeds(&prev_token_embed, &recurrent_hidden);
        let position_offset = 7usize; // matches gen_gemma31b_assistant_golden.py's POSITION_OFFSET

        let out = assistant_forward(&cfg, &weights, &inputs_embeds, position_offset, &shared_kv, None);

        assert_eq!(out.logits.len(), golden_logits.len());
        assert_eq!(out.post_projection_out.len(), golden_post_proj.len());

        let logits_cos = cosine(&out.logits, &golden_logits);
        let amx_r = argmax(&out.logits);
        let amx_g = argmax(&golden_logits);
        let post_cos = cosine(&out.post_projection_out, &golden_post_proj);

        eprintln!(
            "gemma31b assistant: logits cos={logits_cos:.6} argmax rust={amx_r} golden={amx_g} \
             rust[amax]={:.4} golden[amax]={:.4}  post_proj cos={post_cos:.6}",
            out.logits[amx_r], golden_logits[amx_g]
        );

        assert!(logits_cos >= 0.999, "logits cos {logits_cos} < 0.999");
        assert_eq!(amx_r, amx_g, "argmax mismatch: rust={amx_r} golden={amx_g}");
        assert!(post_cos >= 0.999, "post_projection cos {post_cos} < 0.999");
    }
}
