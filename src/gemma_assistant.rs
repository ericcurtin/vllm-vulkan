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

    // ─────────────────────────────────────────────────────────────────────
    // SYNTHETIC (checkpoint-free) DRAFTER FIXTURE
    //
    // Everything above this line needs `VLLM_TEST_GEMMA31B_ASSISTANT_DIR` and
    // a 939MB checkpoint, so in CI it prints SKIP and the whole assistant path
    // stays unexercised — green without being tested. The fixture below is the
    // unconditional twin: a TINY deterministic drafter, paired with
    // `model::tiny_synthetic_gemma` as its target, running the REAL
    // `assistant_forward` and the REAL `gemma_spec::run_spec_decode`.
    //
    // It reproduces the pairing `gemma_spec.rs` already uses (real-checkpoint
    // gate `#[ignore]`d / skipped, synthetic twin unconditional), and reuses
    // `model::synth_vec` — the SAME deterministic generator
    // `tiny_synthetic_gemma`'s weights come from — rather than forking a second
    // near-copy of it.
    //
    // WHAT IT CANNOT DO, stated so the gap is on the record: it has no external
    // reference. The real golden gate
    // (`gemma31b_assistant_forward_cos_vs_golden`) compares against numbers
    // dumped from upstream mlx_vlm, which is what pins attention `scale = 1.0`
    // (not `1/sqrt(head_dim)`) and the borrowed-K "already rotated, do not
    // re-rotate" convention to something outside this crate. Nothing synthetic
    // can replace that: a fixture generated by this code cannot prove this code
    // agrees with mlx_vlm. The checkpoint gate keeps its value on a machine
    // that has the weights; what follows covers the invariants that do NOT
    // need an outside oracle.
    // ─────────────────────────────────────────────────────────────────────

    use crate::model::{synth_vec, tiny_synthetic_gemma, Gemma4Config, Gemma4Model};
    use crate::gemma_spec::{run_spec_decode, SpecConfig};

    /// The drafter geometry paired with `tiny_synthetic_gemma`.
    ///
    /// Every field that must AGREE with the target is derived from the target's
    /// own config rather than restated: `backbone_hidden_size` is the target's
    /// hidden size, and the KV-head counts / head dims are read off the
    /// target's LAST layer of each attention type — the two layers the drafter
    /// borrows K/V from (`GEMMA31B_SPEC_PLAN.md` §1.2). Restating them as
    /// literals is how the pairing silently rots when the target fixture
    /// changes, and a mis-sized borrowed KV is the exact class of the g12b
    /// 4096-vs-512 panic `AssistantConfig::layer_num_kv_heads` documents.
    ///
    /// The drafter's OWN sizes (hidden 64, ffn 128) are free and are chosen
    /// tiny; the 4-layer `[sliding, sliding, sliding, full]` layer_types is the
    /// real 31B pairing's shape, kept verbatim so the fixture exercises both
    /// attention kinds and the head_dim/theta switch between them.
    fn tiny_assistant_cfg(t: &Gemma4Config) -> AssistantConfig {
        let last_sliding = (0..t.num_hidden_layers)
            .rev().find(|&l| !t.is_full_attention(l)).expect("target has a sliding layer");
        let last_full = (0..t.num_hidden_layers)
            .rev().find(|&l| t.is_full_attention(l)).expect("target has a full-attention layer");
        AssistantConfig {
            hidden_size: 64,
            backbone_hidden_size: t.hidden_size,
            num_hidden_layers: 4,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.layer_num_kv_heads(last_sliding),
            num_global_key_value_heads: t.layer_num_kv_heads(last_full),
            head_dim: t.layer_head_dim(last_sliding),
            global_head_dim: t.layer_head_dim(last_full),
            intermediate_size: 128,
            sliding_window: t.sliding_window,
            rms_norm_eps: 1e-6,
            // Tied lm_head: the drafter samples token ids that are fed straight
            // back to the TARGET, so its vocab must be the target's.
            vocab_size: t.vocab_size,
            layer_types: ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
                .iter().map(|s| AssistantLayerKind::parse(s).unwrap()).collect(),
            full_rope_theta: 1_000_000.0,
            sliding_rope_theta: 10_000.0,
        }
    }

    /// Deterministic drafter weights, built from `expected_tensor_shapes` and
    /// NOTHING else.
    ///
    /// That is deliberate and is itself a gate: the checkpoint tensor map is
    /// the only source of names and shapes here, so a name `assistant_forward`
    /// reads but the map does not list makes every test below panic with
    /// "assistant weight '<name>' not found" instead of passing. On the real
    /// checkpoint that agreement is only checked when someone has the 939MB
    /// weights.
    ///
    /// Centering follows `model::synthetic_gemma`: RMSNorm weights and the
    /// residual `layer_scalar` multiply the WHOLE stream and are centered on
    /// 1.0 (centering them on 0 compounds a near-zero multiplier across the
    /// layers and collapses the hidden state into noise); projections are
    /// centered on 0.
    fn tiny_assistant_weights(cfg: &AssistantConfig, tag: &str) -> HashMap<String, Vec<f32>> {
        expected_tensor_shapes(cfg)
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                let (center, half) = if name.ends_with("norm.weight") || name.ends_with("layer_scalar") {
                    (1.0f32, 0.05f32)
                } else {
                    (0.0f32, 0.05f32)
                };
                let data = synth_vec(&format!("{tag}{name}"), n, center, half);
                (name, data)
            })
            .collect()
    }

    /// The target's borrowed K/V snapshot: the LAST layer of each attention
    /// type, read straight out of the target's own caches in the same
    /// `[seq_len, n_kv_heads, head_dim]` absolute layout `gemma_kv_layer`
    /// (`pyseam_gemma.rs`) hands the production driver.
    ///
    /// The dimension asserts are the point: they are what fails if the drafter
    /// config and the target's layers ever stop lining up.
    fn borrowed_kv(model: &Gemma4Model, acfg: &AssistantConfig) -> AssistantSharedKv {
        let t = &model.config;
        let last_sliding = (0..t.num_hidden_layers).rev().find(|&l| !t.is_full_attention(l)).unwrap();
        let last_full = (0..t.num_hidden_layers).rev().find(|&l| t.is_full_attention(l)).unwrap();
        let cs = &model.kv_caches[last_sliding];
        let cf = &model.kv_caches[last_full];
        assert!(!cs.has_wrapped() && !cf.has_wrapped(),
                "borrowed_kv needs the absolute-position layout; the ring must not have wrapped");
        assert_eq!(cs.seq_len, cf.seq_len, "both borrowed layers must be at the same frontier");
        assert_eq!(cs.num_kv_heads, acfg.num_key_value_heads,
                   "sliding kv-head count: target layer {last_sliding} vs drafter config");
        assert_eq!(cs.head_dim, acfg.head_dim,
                   "sliding head_dim: target layer {last_sliding} vs drafter config");
        assert_eq!(cf.num_kv_heads, acfg.num_global_key_value_heads,
                   "full kv-head count: target layer {last_full} vs drafter config");
        assert_eq!(cf.head_dim, acfg.global_head_dim,
                   "full head_dim: target layer {last_full} vs drafter config");
        AssistantSharedKv {
            sliding_k: cs.k_up_to_now().to_vec(),
            sliding_v: cs.v_up_to_now().to_vec(),
            full_k: cf.k_up_to_now().to_vec(),
            full_v: cf.v_up_to_now().to_vec(),
            kv_len: cs.seq_len,
        }
    }

    /// A greedy SPEC-OFF baseline that ALSO keeps, for every emitted token, the
    /// target hidden state that produced it.
    ///
    /// `gemma_spec.rs`'s `greedy_baseline` returns ids only, which is all its
    /// stub drafter needs. The real drafter needs more: the plan's step-0
    /// `recurrent_hidden` IS the target's final hidden — the state the bonus
    /// token was sampled from (`GEMMA31B_SPEC_PLAN.md` §1.1) — so a gate that
    /// drives the real drafter has to carry it. `hidden_before[j]` is the
    /// hidden that produced `ids[j]`, i.e. the recurrent seed for a block whose
    /// bonus is `ids[j]`; `hidden_before[0]` is the post-prompt hidden, the
    /// state `start_bonus` would have been sampled from in a real run.
    struct Baseline {
        ids: Vec<u32>,
        hidden_before: Vec<Vec<f32>>,
    }

    fn greedy_baseline_with_hidden(
        model: &mut Gemma4Model,
        prompt_hidden: Vec<f32>,
        start_bonus: u32,
        start_pos: usize,
        n: usize,
    ) -> Baseline {
        let mut ids = Vec::with_capacity(n);
        let mut hidden_before = Vec::with_capacity(n);
        let mut tok = start_bonus;
        let mut prev_hidden = prompt_hidden;
        for i in 0..n {
            ids.push(tok);
            hidden_before.push(prev_hidden);
            let (logits, normed) = model.forward_with_normed(tok, start_pos + i);
            tok = crate::model::argmax(&logits) as u32;
            prev_hidden = normed;
        }
        Baseline { ids, hidden_before }
    }

    /// ONE draft block, implementing the plan's draft loop
    /// (`GEMMA31B_SPEC_PLAN.md` §1.4 pseudocode) verbatim on top of the real
    /// `assistant_forward`:
    ///
    /// ```text
    /// for _ in 0..k:
    ///     tok_embed     = target.embed(tok) * target.embed_scale
    ///     inputs_embeds = concat([tok_embed, h_prev])      # embed FIRST
    ///     h_prev, logits = drafter(inputs_embeds, shared_kv, pos)
    ///     tok = argmax(logits)
    /// ```
    ///
    /// with `position_offset` held CONSTANT across the block and `shared_kv` a
    /// fixed snapshot the drafter never appends to. This is the INC-5b coupling
    /// that has no in-crate caller yet; writing it here is what makes the
    /// drafter and the driver meet at all.
    ///
    /// `forced` teacher-forces the fed token (the drafter still computes its
    /// own argmax, which is returned) — used only when BUILDING the aligned
    /// fixture below, never when gating it.
    ///
    /// Returns `(argmax, logits)` per step.
    #[allow(clippy::too_many_arguments)]
    fn draft_block(
        acfg: &AssistantConfig,
        aw: &HashMap<String, Vec<f32>>,
        target_embed: &[f32],
        embed_scale: f32,
        kv: &AssistantSharedKv,
        bonus: u32,
        seed_hidden: &[f32],
        position_offset: usize,
        k: usize,
        forced: Option<&[u32]>,
    ) -> Vec<(u32, Vec<f32>)> {
        let backbone = acfg.backbone_hidden_size;
        let mut out: Vec<(u32, Vec<f32>)> = Vec::with_capacity(k);
        let mut recurrent = seed_hidden.to_vec();
        let mut tok = bonus;
        for i in 0..k {
            if i > 0 {
                tok = match forced {
                    Some(f) => f[i - 1],
                    None => out[i - 1].0,
                };
            }
            let embed: Vec<f32> = target_embed[tok as usize * backbone..(tok as usize + 1) * backbone]
                .iter().map(|&v| v * embed_scale).collect();
            let inputs_embeds = build_inputs_embeds(&embed, &recurrent);
            let step = assistant_forward(acfg, aw, &inputs_embeds, position_offset, kv, None);
            recurrent = step.post_projection_out;
            out.push((argmax(&step.logits) as u32, step.logits));
        }
        out
    }

    /// The fixed fixture state every gate below starts from: a tiny target with
    /// a replayed prompt, its borrowed-KV snapshot, the drafter config/weights,
    /// and the greedy baseline (+ hidden) the spec-decode driver must reproduce.
    struct Fixture {
        model: Gemma4Model,
        acfg: AssistantConfig,
        aw: HashMap<String, Vec<f32>>,
        kv: AssistantSharedKv,
        target_embed: Vec<f32>,
        embed_scale: f32,
        baseline: Baseline,
        start_pos: usize,
        start_bonus: u32,
        max_seq: usize,
        prompt: Vec<u32>,
    }

    const FIXTURE_MAX_SEQ: usize = 64;
    const FIXTURE_PROMPT: [u32; 6] = [2, 10, 20, 33, 41, 7];
    const FIXTURE_START_BONUS: u32 = 30;

    /// Builds the fixture and leaves the target's KV holding EXACTLY the
    /// prompt, so a caller can drive the spec-decode loop from the same clean
    /// state the baseline was computed from.
    fn build_fixture(baseline_len: usize) -> Fixture {
        let max_seq = FIXTURE_MAX_SEQ;
        let mut model = tiny_synthetic_gemma(max_seq);
        let tcfg = model.config.clone();
        let acfg = tiny_assistant_cfg(&tcfg);
        let aw = tiny_assistant_weights(&acfg, "");
        let prompt = FIXTURE_PROMPT.to_vec();

        let mut prompt_hidden = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            let (_, normed) = model.forward_with_normed(tok, pos);
            prompt_hidden = normed;
        }
        let start_pos = prompt.len();
        let baseline = greedy_baseline_with_hidden(
            &mut model, prompt_hidden, FIXTURE_START_BONUS, start_pos, baseline_len);

        // Rewind to the post-prompt state and snapshot the borrowed KV there.
        crate::model::gemma_reset_kv(&mut model);
        for (pos, &tok) in prompt.iter().enumerate() {
            model.forward(tok, pos);
        }
        let kv = borrowed_kv(&model, &acfg);
        let target_embed = model.weights.f32_slice("model.embed_tokens.weight").to_vec();

        Fixture {
            embed_scale: tcfg.embed_scale,
            target_embed,
            model, acfg, aw, kv, baseline, start_pos,
            start_bonus: FIXTURE_START_BONUS,
            max_seq, prompt,
        }
    }

    impl Fixture {
        /// Rebuild the target's KV caches and replay the prompt, so a second
        /// run starts from the identical state the first did.
        fn rewind(&mut self) {
            let cfg = self.model.config.clone();
            self.model.kv_caches = (0..cfg.num_hidden_layers)
                .map(|l| crate::model::KvCache::new_windowed(
                    self.max_seq,
                    cfg.layer_kv_capacity(l, self.max_seq),
                    cfg.layer_num_kv_heads(l),
                    cfg.layer_head_dim(l)))
                .collect();
            for (pos, &tok) in self.prompt.iter().enumerate() {
                self.model.forward(tok, pos);
            }
        }

        fn draft(&self, bonus: u32, seed_hidden: &[f32], pos: usize, k: usize, forced: Option<&[u32]>)
            -> Vec<(u32, Vec<f32>)>
        {
            draft_block(&self.acfg, &self.aw, &self.target_embed, self.embed_scale,
                        &self.kv, bonus, seed_hidden, pos, k, forced)
        }
    }

    /// Largest RELATIVE difference between two equal-length vectors, scaled by
    /// the larger vector norm — a scale-free "did this change at all" measure
    /// for the wiring gates below.
    fn rel_change(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let maxdiff = a.iter().zip(b).map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, |m, d| if d.is_nan() || d > m { d } else { m });
        let scale = a.iter().chain(b.iter()).map(|v| v.abs())
            .fold(0.0f32, |m, d| if d.is_nan() || d > m { d } else { m });
        if scale == 0.0 { return 0.0; }
        maxdiff / scale
    }

    // ── Gate 1: the fixture IS the checkpoint tensor map ──────────────────

    /// Synthetic twin of `gemma31b_assistant_load_shapes`.
    ///
    /// It cannot check the real checkpoint's contents (that needs the
    /// checkpoint), but it does check the half of that gate which is about
    /// THIS CRATE: that `expected_tensor_shapes` is a complete and exact
    /// description of what `assistant_forward` reads, that the map contains no
    /// forbidden Q-only-violating name, and that the forward's output shapes
    /// are the ones `GemmaAssistant::forward` promises Python. A weight name
    /// the forward reads but the map omits panics here; a map entry with the
    /// wrong shape trips a `cpu_matmul` dimension check.
    #[test]
    fn gemma_assistant_synthetic_fixture_covers_the_checkpoint_tensor_map() {
        let f = build_fixture(2);
        let cfg = &f.acfg;

        // Same 4 + 11-per-layer structure the real 48-tensor map has.
        let expected = expected_tensor_shapes(cfg);
        assert_eq!(expected.len(), 4 + 11 * cfg.num_hidden_layers,
                   "tensor map shape changed: {} entries for {} layers",
                   expected.len(), cfg.num_hidden_layers);
        assert_eq!(expected_tensor_shapes(&AssistantConfig::g31b_pair()).len(), 48,
                   "the real 31B pairing must still be a 48-tensor map");

        // The fixture was built from the map alone, so this also proves the
        // map has no duplicate names.
        assert_eq!(f.aw.len(), expected.len(), "fixture must hold exactly the mapped tensors");
        for (name, shape) in &expected {
            let numel: usize = shape.iter().product();
            assert_eq!(f.aw.get(name).map(|v| v.len()), Some(numel),
                       "fixture tensor '{name}' must be {shape:?} ({numel} elems)");
        }
        for name in forbidden_tensor_names(cfg) {
            assert!(!f.aw.contains_key(&name),
                    "drafter is Q-only: '{name}' must not exist");
        }

        // The forward runs consuming ONLY mapped names (a miss panics inside
        // `amv`/`w`), and returns the shapes the pyclass contract promises.
        let seed = f.baseline.hidden_before[0].clone();
        let steps = f.draft(f.start_bonus, &seed, f.start_pos, 1, None);
        let (tok, logits) = &steps[0];
        assert_eq!(logits.len(), cfg.vocab_size, "logits must be [vocab_size]");
        assert!((*tok as usize) < cfg.vocab_size, "drafted id {tok} outside the target's vocab");
        assert!(logits.iter().all(|v| v.is_finite()), "drafter produced non-finite logits");
    }

    // ── Gate 2: every drafter input is actually wired in ──────────────────

    /// ENGAGEMENT gate: each of the drafter's five inputs must demonstrably
    /// change its output.
    ///
    /// A drafter that ignores an input is the assistant-path form of "the lever
    /// is disengaged but the gate is green": the spec-decode committed stream
    /// is greedy-identical no matter how bad the drafts are (see
    /// `gemma_assistant_zero_acceptance_is_caught_by_the_drafter_gate` below),
    /// so nothing downstream notices. Each check below is a differential, which
    /// is why it needs no external oracle and no platform-sensitive constant.
    ///
    /// The inputs_embeds ORDER check is the one with history: recurrent-first
    /// instead of embed-first collapses real acceptance 0.43 -> 0.01
    /// (`GEMMA31B_SPEC_PLAN.md` risk #5), and it is invisible to any check that
    /// only looks at shapes.
    #[test]
    fn gemma_assistant_synthetic_forward_is_wired_to_every_input() {
        let f = build_fixture(2);
        let cfg = &f.acfg;
        let backbone = cfg.backbone_hidden_size;
        let seed = f.baseline.hidden_before[0].clone();

        // `build_inputs_embeds` puts the token embedding FIRST. Exact, not a
        // differential: this is the concat order itself.
        let a: Vec<f32> = (0..backbone).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..backbone).map(|i| -(i as f32)).collect();
        let ie = build_inputs_embeds(&a, &b);
        assert_eq!(ie.len(), 2 * backbone);
        assert!(ie[..backbone] == a[..],
                "prev_token_embed must be the FIRST half of inputs_embeds (got ie[1]={}, want {})",
                ie[1], a[1]);
        assert!(ie[backbone..] == b[..],
                "recurrent_hidden must be the SECOND half of inputs_embeds (got ie[{}]={}, want {})",
                backbone + 1, ie[backbone + 1], b[1]);

        let tok = f.start_bonus as usize;
        let embed: Vec<f32> = f.target_embed[tok * backbone..(tok + 1) * backbone]
            .iter().map(|&v| v * f.embed_scale).collect();
        let base = assistant_forward(cfg, &f.aw, &build_inputs_embeds(&embed, &seed),
                                     f.start_pos, &f.kv, None);

        // 1. Concat ORDER: halves swapped must change the logits.
        let swapped = assistant_forward(cfg, &f.aw, &build_inputs_embeds(&seed, &embed),
                                        f.start_pos, &f.kv, None);
        let d = rel_change(&base.logits, &swapped.logits);
        assert!(d > 1e-3, "inputs_embeds order is not observable (rel change {d:.3e}) — \
                           embed-first vs recurrent-first must not be interchangeable");

        // 2. RECURRENT half: the rolling target-space hidden must feed the forward.
        let mut rec2 = seed.clone();
        rec2.iter_mut().for_each(|v| *v += 0.25);
        let r = assistant_forward(cfg, &f.aw, &build_inputs_embeds(&embed, &rec2),
                                  f.start_pos, &f.kv, None);
        let d = rel_change(&base.logits, &r.logits);
        assert!(d > 1e-3, "recurrent_hidden is not observable (rel change {d:.3e}) — \
                           the drafter's recurrence is disengaged");

        // 3-6. BORROWED KV: the sliding layers (0..2) read the sliding pair and
        // the global layer (3) reads the full pair. All four tensors must move
        // the logits — if they do not, the cross-attention is not attending to
        // the target's KV at all, which is the whole point of an EAGLE drafter.
        //
        // The perturbation is applied to ONE kv POSITION, not to the whole
        // tensor, and that detail is load-bearing: adding the same constant to
        // every K row shifts every attention logit by the same amount, which
        // softmax cancels exactly — the first draft of this gate did that and
        // measured a 5.0e-7 change on a correctly-wired drafter. A K check has
        // to break the symmetry between positions to be a check at all.
        let bump_first_row = |t: &[f32], stride: usize| -> Vec<f32> {
            let mut v = t.to_vec();
            v[..stride].iter_mut().for_each(|x| *x += 0.5);
            v
        };
        let s_stride = cfg.num_key_value_heads * cfg.head_dim;
        let g_stride = cfg.num_global_key_value_heads * cfg.global_head_dim;
        let variants: [(&str, AssistantSharedKv); 4] = [
            ("SLIDING K", AssistantSharedKv {
                sliding_k: bump_first_row(&f.kv.sliding_k, s_stride), sliding_v: f.kv.sliding_v.clone(),
                full_k: f.kv.full_k.clone(), full_v: f.kv.full_v.clone(), kv_len: f.kv.kv_len }),
            ("SLIDING V", AssistantSharedKv {
                sliding_k: f.kv.sliding_k.clone(), sliding_v: bump_first_row(&f.kv.sliding_v, s_stride),
                full_k: f.kv.full_k.clone(), full_v: f.kv.full_v.clone(), kv_len: f.kv.kv_len }),
            ("FULL K", AssistantSharedKv {
                sliding_k: f.kv.sliding_k.clone(), sliding_v: f.kv.sliding_v.clone(),
                full_k: bump_first_row(&f.kv.full_k, g_stride), full_v: f.kv.full_v.clone(), kv_len: f.kv.kv_len }),
            ("FULL V", AssistantSharedKv {
                sliding_k: f.kv.sliding_k.clone(), sliding_v: f.kv.sliding_v.clone(),
                full_k: f.kv.full_k.clone(), full_v: bump_first_row(&f.kv.full_v, g_stride), kv_len: f.kv.kv_len }),
        ];
        for (what, kv) in &variants {
            let r = assistant_forward(cfg, &f.aw, &build_inputs_embeds(&embed, &seed),
                                      f.start_pos, kv, None);
            let d = rel_change(&base.logits, &r.logits);
            assert!(d > 1e-3, "borrowed {what} is not observable (rel change {d:.3e}) — \
                               the drafter is not cross-attending the target's KV");
        }

        // 5. POSITION: the drafter RoPEs its own Q at `position_offset`.
        let r = assistant_forward(cfg, &f.aw, &build_inputs_embeds(&embed, &seed),
                                  f.start_pos + 1, &f.kv, None);
        let d = rel_change(&base.logits, &r.logits);
        assert!(d > 1e-3, "position_offset is not observable (rel change {d:.3e}) — \
                           the drafter's own-Q RoPE is disengaged");

        // The borrowed K/V must NOT be re-rotated by the drafter, so the
        // drafter's output must be invariant to which ABSOLUTE positions the
        // borrowed rows came from — there is no place to feed them, and the
        // check above already proves `position_offset` reaches only Q. Stated
        // here because a future "helpfully" re-rotating K would still pass
        // every shape check. The real golden gate is what pins the numbers.
        assert_eq!(base.post_projection_out.len(), backbone,
                   "post_projection must produce a backbone-wide recurrent hidden");
    }

    // ── Gates 3 + 4: the drafter driving the real spec-decode loop ────────

    // Block structure of the acceptance gate below, derived from
    // `run_spec_decode`'s own budget rule `step_k = min(k, remaining - 1)` and
    // a drafter constructed to have exactly `SPEC_FORCED_CORRECT` of each
    // block's drafts accepted:
    //
    //   remaining 9 -> width min(4,8)=4, accepts 2, commits 3   (block at off 0)
    //   remaining 6 -> width min(4,5)=4, accepts 2, commits 3   (block at off 3)
    //   remaining 3 -> width min(4,2)=2, accepts 2, commits 3   (block at off 6)
    //
    // so 3 blocks, 10 tokens drafted, 6 accepted, 9 committed. `k = 4` with a
    // budget of 9 is deliberately a non-dividing pair: it forces the last block
    // to be narrowed, the case the driver's overshoot bug lived in. The first
    // two blocks reject a suffix and therefore exercise `verify_rollback`; the
    // third accepts its full width and exercises the rollback NO-OP path.
    const SPEC_K: usize = 4;
    const SPEC_N: usize = 9;
    const SPEC_FORCED_CORRECT: usize = 2;
    const SPEC_WIDTHS: [usize; 3] = [4, 4, 2];
    const SPEC_OFFS: [usize; 3] = [0, 3, 6];
    /// Greedy-baseline length the gates need. The deepest read is the last
    /// block's last draft, `SPEC_OFFS[2] + 1 + (SPEC_WIDTHS[2] - 1) == 8`, and
    /// the committed-stream comparison reads `0..SPEC_N`; one spare keeps a
    /// small change to the block structure from running off the end. Every
    /// extra token here is a full target forward, so it is not padded further.
    const SPEC_BASELINE_LEN: usize = SPEC_N + 1;

    /// The draft ids the constructed drafter must produce, per block: the
    /// target's own greedy continuation for the first `SPEC_FORCED_CORRECT`
    /// slots, then ids guaranteed to be rejected (the true next token bumped by
    /// one). Rejection is decided by the TARGET's argmax, so bumping is enough
    /// to guarantee it.
    fn desired_drafts(baseline: &[u32], vocab: u32) -> Vec<Vec<u32>> {
        SPEC_OFFS.iter().zip(SPEC_WIDTHS.iter()).map(|(&off, &w)| {
            (0..w).map(|i| {
                let truth = baseline[off + 1 + i];
                if i < SPEC_FORCED_CORRECT { truth } else { (truth + 1) % vocab }
            }).collect()
        }).collect()
    }

    /// Inverse of a small dense f64 matrix by Gauss-Jordan with partial
    /// pivoting. Panics on a singular matrix — the caller only ever inverts a
    /// Gram matrix, whose singularity would mean the fixture's hidden states
    /// are linearly dependent and the construction below is impossible; that
    /// deserves a loud failure, not a silently degraded fixture.
    fn invert(mut a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n = a.len();
        let mut inv: Vec<Vec<f64>> = (0..n).map(|i| {
            (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()
        }).collect();
        for col in 0..n {
            let piv = (col..n).max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap()).unwrap();
            assert!(a[piv][col].abs() > 1e-12, "singular Gram matrix at column {col}");
            a.swap(col, piv);
            inv.swap(col, piv);
            let d = a[col][col];
            for j in 0..n { a[col][j] /= d; inv[col][j] /= d; }
            for r in 0..n {
                if r == col { continue; }
                let f = a[r][col];
                if f == 0.0 { continue; }
                for j in 0..n { a[r][j] -= f * a[col][j]; inv[r][j] -= f * inv[col][j]; }
            }
        }
        inv
    }

    /// Dual (biorthogonal) basis of `vs`: vectors `d_i` with
    /// `vs[s] · d[i] == 1` when `s == i` and `0` otherwise.
    ///
    /// `d = G^-1 V` where `G` is the Gram matrix, since
    /// `vs[s] · d[i] = sum_j G^-1[i][j] G[s][j] = (G G^-1)[s][i]`.
    fn dual_basis(vs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let m = vs.len();
        let h = vs[0].len();
        let gram: Vec<Vec<f64>> = (0..m).map(|i| (0..m).map(|j|
            vs[i].iter().zip(vs[j].iter()).map(|(&a, &b)| a as f64 * b as f64).sum()
        ).collect()).collect();
        let gi = invert(gram);
        (0..m).map(|i| {
            let mut d = vec![0.0f64; h];
            for j in 0..m {
                let c = gi[i][j];
                for t in 0..h { d[t] += c * vs[j][t] as f64; }
            }
            d.into_iter().map(|x| x as f32).collect()
        }).collect()
    }

    /// Builds a drafter that, running the REAL `assistant_forward`, drafts
    /// exactly `desired`.
    ///
    /// This is the assistant-path analogue of `gemma_spec.rs`'s stub drafter
    /// ("always draft the target's own argmax") — with the crucial difference
    /// that the drafts here come out of the actual drafter forward instead of
    /// out of a closure that bypasses it. Random weights cannot serve: measured
    /// on this fixture they accept 0 of the 26 tokens they offer (that run is the
    /// negative control below), which would make the acceptance assertion this
    /// gate exists for vacuous.
    ///
    /// How: the drafter's logits are `inner @ E^T` where `E` is the tied
    /// lm_head `model.embed_tokens.weight`, and `E` is read NOWHERE ELSE in
    /// `assistant_forward` — the input side uses the TARGET's embedding table,
    /// and the recurrent feedback is `post_projection`'s output. So `inner`,
    /// and with it the whole drafter trajectory, is independent of `E`, and `E`
    /// can be solved for after the fact:
    ///
    ///   1. run the chain with `E = [I; 0]`, which makes `logits[..hidden]`
    ///      read out `inner` itself, teacher-forcing the tokens `desired`;
    ///   2. set `E[desired[s]] += dual_s`, the dual basis of those `inner`
    ///      vectors. Then at step `s` the desired row scores exactly 1.0, every
    ///      other constructed row exactly 0.0, and every untouched row exactly
    ///      0.0 — an argmax margin of 1.0, immune to any platform float
    ///      difference. Repeated desired tokens are handled by the `+=`: the
    ///      duals of the other steps contribute 0 at step `s` by construction.
    ///
    /// What this does and does not gate: it does NOT re-check the drafter's
    /// numerics (step 1 and the gated run compute `inner` the same way, so a
    /// change to the forward moves both together). It DOES gate that the
    /// drafter is deterministic, that the driver feeds it exactly what the
    /// reference loop feeds it, and — the reason it exists — that the whole
    /// accept/rollback/commit path runs at a NONZERO, exactly-known acceptance
    /// instead of the zero a random drafter yields.
    fn aligned_drafter_weights(f: &Fixture, desired: &[Vec<u32>]) -> HashMap<String, Vec<f32>> {
        let hs = f.acfg.hidden_size;
        let vocab = f.acfg.vocab_size;

        // (1) Probe: E = [I; 0] makes logits[..hs] == inner.
        let mut probe = f.aw.clone();
        let mut eye = vec![0.0f32; vocab * hs];
        for i in 0..hs { eye[i * hs + i] = 1.0; }
        probe.insert("model.embed_tokens.weight".to_string(), eye);

        let mut inners: Vec<Vec<f32>> = Vec::new();
        let mut targets: Vec<u32> = Vec::new();
        for (b, (&off, &w)) in SPEC_OFFS.iter().zip(SPEC_WIDTHS.iter()).enumerate() {
            let steps = draft_block(
                &f.acfg, &probe, &f.target_embed, f.embed_scale, &f.kv,
                f.baseline.ids[off], &f.baseline.hidden_before[off],
                f.start_pos + off, w, Some(&desired[b]));
            for (i, (_, logits)) in steps.iter().enumerate() {
                inners.push(logits[..hs].to_vec());
                targets.push(desired[b][i]);
            }
        }

        // (2) Solve for E.
        let duals = dual_basis(&inners);
        let mut e = vec![0.0f32; vocab * hs];
        for (d, &tok) in duals.iter().zip(targets.iter()) {
            for t in 0..hs { e[tok as usize * hs + t] += d[t]; }
        }
        // The construction must actually separate: the desired row must beat
        // every other row by a wide margin at every step. A near-singular Gram
        // would silently give a fixture that only sometimes drafts what it was
        // built to draft.
        for (s, inner) in inners.iter().enumerate() {
            let scores: Vec<f32> = (0..vocab).map(|w|
                (0..hs).map(|t| inner[t] * e[w * hs + t]).sum()).collect();
            let want = targets[s] as usize;
            let runner_up = (0..vocab).filter(|&w| w != want)
                .fold(f32::MIN, |m, w| if scores[w] > m { scores[w] } else { m });
            assert!(scores[want] - runner_up > 0.5,
                    "aligned fixture step {s}: margin {} is too small (want {} = {}, runner-up {}) — \
                     the Gram matrix of the drafter's hidden states is too ill-conditioned for \
                     this construction",
                    scores[want] - runner_up, want, scores[want], runner_up);
        }

        let mut aligned = f.aw.clone();
        aligned.insert("model.embed_tokens.weight".to_string(), e);
        aligned
    }

    /// Runs `run_spec_decode` with the REAL drafter and returns
    /// `(committed, widths, drafts)`.
    fn run_with_assistant_drafter(
        f: &mut Fixture,
        aw: HashMap<String, Vec<f32>>,
        n: usize,
    ) -> (Vec<u32>, Vec<usize>, Vec<Vec<u32>>) {
        f.rewind();
        let acfg = f.acfg.clone();
        let kv = AssistantSharedKv {
            sliding_k: f.kv.sliding_k.clone(), sliding_v: f.kv.sliding_v.clone(),
            full_k: f.kv.full_k.clone(), full_v: f.kv.full_v.clone(), kv_len: f.kv.kv_len,
        };
        let target_embed = f.target_embed.clone();
        let embed_scale = f.embed_scale;
        let start_pos = f.start_pos;
        // The drafter closure cannot borrow the target (the driver holds it
        // `&mut`), so the per-block recurrent seed — the target hidden that
        // produced this block's bonus token — is taken from the precomputed
        // baseline. That is the SAME value a production driver reads off the
        // target, valid precisely because this gate also asserts the committed
        // stream equals that baseline.
        let hidden_before = f.baseline.hidden_before.clone();

        let mut widths: Vec<usize> = Vec::new();
        let mut drafts: Vec<Vec<u32>> = Vec::new();
        let committed = {
            let draft_fn = |bonus: u32, pos: usize, k: usize| -> Vec<u32> {
                widths.push(k);
                let off = pos - start_pos;
                let steps = draft_block(&acfg, &aw, &target_embed, embed_scale, &kv,
                                        bonus, &hidden_before[off], pos, k, None);
                let ids: Vec<u32> = steps.into_iter().map(|(t, _)| t).collect();
                drafts.push(ids.clone());
                ids
            };
            let cfg = SpecConfig { k: SPEC_K, max_new_tokens: n };
            run_spec_decode(&mut f.model, draft_fn, f.start_bonus, start_pos, &cfg)
        };
        (committed, widths, drafts)
    }

    /// THE GATE the maintainer follow-up asks for: the EAGLE drafter and the
    /// target meeting, unconditionally, with no checkpoint and no env var.
    ///
    /// It runs the real `assistant_forward` as the drafter inside the real
    /// `gemma_spec::run_spec_decode` over `tiny_synthetic_gemma`, wiring them
    /// together exactly as `GEMMA31B_SPEC_PLAN.md` §1.4 specifies (embed-first
    /// `inputs_embeds`, `position_offset` constant across a block, a fixed
    /// borrowed-KV snapshot the drafter never appends to, `post_projection`
    /// fed back as the next step's recurrent hidden). That coupling is INC-5b
    /// and has no in-crate caller, so before this gate nothing in the crate
    /// ever ran the drafter and the driver together.
    ///
    /// It asserts MEASURED quantities, not just that the output looks right.
    /// Committed-stream equality alone is worthless here: it holds even when
    /// every draft is rejected (that is exactly what the negative control below
    /// demonstrates), so a gate resting on it would stay green with the
    /// speculative lever completely disengaged.
    #[test]
    fn gemma_assistant_synthetic_drafter_drives_spec_decode_with_measured_acceptance() {
        let mut f = build_fixture(SPEC_BASELINE_LEN);
        let vocab = f.acfg.vocab_size as u32;
        let desired = desired_drafts(&f.baseline.ids, vocab);
        let aligned = aligned_drafter_weights(&f, &desired);
        let baseline = f.baseline.ids.clone();

        let (committed, widths, drafts) = run_with_assistant_drafter(&mut f, aligned, SPEC_N);

        // 1. Committed-stream equality against the SPEC-off greedy baseline,
        //    and the budget honoured EXACTLY (not `>=`).
        assert_eq!(committed.len(), SPEC_N,
                   "expected EXACTLY {SPEC_N} committed tokens, got {}", committed.len());
        assert_eq!(committed[..], baseline[..SPEC_N],
                   "spec-decode stream diverged from the greedy baseline");

        // 2. ACCEPTANCE, measured. Each block commits its bonus plus its
        //    accepted drafts, so over the run `accepted == n - blocks`.
        let blocks = widths.len();
        let drafted: usize = widths.iter().sum();
        let accepted = SPEC_N - blocks;
        assert_eq!(accepted, SPEC_FORCED_CORRECT * blocks,
                   "expected {SPEC_FORCED_CORRECT} of each block's drafts accepted over {blocks} \
                    blocks, got {accepted} of {drafted} offered (acceptance collapsed / the \
                    drafter is disengaged)");
        assert!(accepted > 0, "a gate that passes at zero acceptance gates nothing");
        assert_eq!(widths, SPEC_WIDTHS.to_vec(),
                   "block structure changed: k={SPEC_K}, n={SPEC_N} must run these draft widths");
        assert_eq!(drafted, SPEC_WIDTHS.iter().sum::<usize>(), "tokens offered");

        // 3. The drafter produced what it was constructed to produce — i.e. the
        //    driver fed it the same bonus/position/recurrent state the
        //    reference loop did, block after block.
        assert_eq!(drafts, desired,
                   "the drafter did not emit its constructed drafts: the driver is feeding it \
                    different state than the reference draft loop does");

        // 4. Rollback: the KV frontier of EVERY layer sits at exactly the
        //    committed length, so the rejected drafts left nothing behind.
        let frontier = f.start_pos + SPEC_N;
        for (li, c) in f.model.kv_caches.iter().enumerate() {
            assert_eq!(c.seq_len, frontier, "layer {li}: KV frontier {} != {frontier}", c.seq_len);
        }

        eprintln!("assistant spec gate (synthetic): {SPEC_N} committed tokens match the greedy \
                   baseline; {blocks} blocks, {accepted}/{drafted} drafted tokens accepted");
    }

    /// NEGATIVE CONTROL for the acceptance assertion above, and the reason it
    /// is not decoration.
    ///
    /// The same fixture with the drafter's PLAIN synthetic weights — a drafter
    /// with no reason to agree with the target — shows that:
    ///
    ///   1. the committed stream is still bit-identical to the greedy baseline
    ///      and the budget is still honoured exactly, so every "the output
    ///      looks right" check passes with speculation contributing nothing;
    ///   2. the measured acceptance is ZERO across the 26 tokens offered, which is
    ///      what the gate above asserts against.
    ///
    /// The block structure differs too: with nothing accepted every block
    /// commits its bonus alone, so the run takes 9 blocks instead of 3.
    #[test]
    fn gemma_assistant_zero_acceptance_is_caught_by_the_drafter_gate() {
        let mut f = build_fixture(SPEC_BASELINE_LEN);
        let baseline = f.baseline.ids.clone();
        let plain = f.aw.clone();

        let (committed, widths, drafts) = run_with_assistant_drafter(&mut f, plain, SPEC_N);

        // (1) Everything the acceptance assertions do NOT cover still passes.
        assert_eq!(committed.len(), SPEC_N, "budget must still be honoured exactly");
        assert_eq!(committed[..], baseline[..SPEC_N],
                   "zero-acceptance stream must still equal the greedy baseline — this is \
                    precisely why token identity alone cannot gate speculation");

        // (2) ...but acceptance collapses, and the block structure with it.
        let blocks = widths.len();
        let drafted: usize = widths.iter().sum();
        let accepted = SPEC_N - blocks;
        assert!(drafted > 0, "the control is only meaningful if drafts were actually offered");
        assert!(drafts.iter().all(|d| !d.is_empty()) || widths.contains(&0),
                "every non-zero-width block must have produced drafts");
        assert_eq!(accepted, 0, "an unaligned drafter must have nothing accepted");
        assert_eq!(blocks, SPEC_N, "with nothing accepted every block commits its bonus alone");
        assert_ne!(widths, SPEC_WIDTHS.to_vec(),
                   "the collapsed block structure must differ from the accepting one");

        eprintln!("assistant negative control: greedy-identical stream, but {accepted}/{drafted} \
                   accepted over {blocks} blocks (the accepting gate runs {:?})", SPEC_WIDTHS);
    }

    /// Number of drafter steps the pinned chain below covers.
    const PINNED_CHAIN_LEN: usize = 8;
    /// The draft ids the fixture's UNALIGNED drafter emits, chained from the
    /// fixture's start state. Regenerate with:
    ///   cargo test --features "multiple-pymethods,gemma" --lib \
    ///     gemma_assistant_synthetic_forward_matches_its_pinned_draft_chain -- --nocapture
    const PINNED_CHAIN: [u32; PINNED_CHAIN_LEN] = [83, 333, 404, 444, 213, 452, 452, 419];

    /// CHARACTERIZATION gate — the one check here that pins the drafter's
    /// NUMBERS rather than its wiring.
    ///
    /// It is needed because the acceptance gate cannot do it: that gate's
    /// lm_head is solved for from the drafter's own hidden states, so a change
    /// to the forward moves the construction and the run together and the gate
    /// stays green. Concretely, changing the cross-attention scale from the
    /// target's QK-norm convention (`1.0`) to the conventional
    /// `1/sqrt(head_dim)` is invisible to every other test in this module that
    /// runs without a checkpoint; it is visible here.
    ///
    /// This pins THIS implementation's output, not upstream mlx_vlm's — it
    /// cannot find a bug that was present when the values were generated. That
    /// remains the real golden gate's job. What it does catch is the change
    /// nobody meant to make.
    ///
    /// Platform note: the assertion is on argmax, and the run below prints the
    /// top-1/top-2 gap and requires it to stay far above float noise, so the
    /// pin cannot become a coin-flip between x86-64 and aarch64 without saying
    /// so first.
    #[test]
    fn gemma_assistant_synthetic_forward_matches_its_pinned_draft_chain() {
        let f = build_fixture(2);
        let seed = f.baseline.hidden_before[0].clone();
        let steps = f.draft(f.start_bonus, &seed, f.start_pos, PINNED_CHAIN_LEN, None);

        let ids: Vec<u32> = steps.iter().map(|(t, _)| *t).collect();
        let mut min_rel_gap = f32::MAX;
        for (i, (tok, logits)) in steps.iter().enumerate() {
            let top = logits[*tok as usize];
            let second = logits.iter().enumerate()
                .filter(|(w, _)| *w != *tok as usize)
                .fold(f32::MIN, |m, (_, &v)| if v > m { v } else { m });
            let rel = (top - second) / top.abs().max(1e-30);
            min_rel_gap = min_rel_gap.min(rel);
            eprintln!("pinned chain step {i}: id={tok} top1={top:.6} top2={second:.6} rel_gap={rel:.3e}");
        }
        eprintln!("pinned chain ids: {ids:?}  min rel gap {min_rel_gap:.3e}");

        // ~1e-2 measured; f32 accumulation error over these dot products is
        // ~1e-6 relative, so the argmax is not a float coin-flip. If this ever
        // trips, the pin below must be replaced by a tolerance-based check —
        // do NOT just widen it.
        assert!(min_rel_gap > 1e-4,
                "argmax margin {min_rel_gap:.3e} is too small for a cross-platform pin");
        assert_eq!(ids, PINNED_CHAIN.to_vec(),
                   "the drafter's output changed. If that was intended, regenerate the pin \
                    (see the const); if not, this is the regression the gate is for");
    }
}
