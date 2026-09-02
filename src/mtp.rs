// SPDX-License-Identifier: Apache-2.0
//! Qwen3.6-35B-A3B MTP (Multi-Token-Prediction) draft head — Phase P2 of the
//! speculative-pipelining plan (`plan-mtp-draft.md` §P2).
//!
//! The head is a DeepSeek-V3-style NextN predictor trained by Qwen as a next-2
//! token drafter. It consumes:
//!   * the embedding of the **last emitted token** `t_next`, and
//!   * the target model's **pre-`model.norm` residual hidden** `h_pre`,
//! each RMSNormed (`pre_fc_norm_embedding` / `pre_fc_norm_hidden`, **embedding
//! FIRST** in the concat), projected by `fc [2048,4096] → [2048]`, run through
//! **exactly one target-geometry gated-full-attention + 256-expert MoE layer**
//! (q_proj 8192 = the `attn_output_gate` doubling; partial rope 0.25, θ=1e7,
//! GQA 16q/2kv, head_dim 256), then `mtp.norm` and the target's **shared
//! lm_head**. `mtp_use_dedicated_embeddings=false` ⇒ embed + lm_head are the
//! main model's tables.
//!
//! Wiring authoritatively recovered in P0 (α=0.889 on the 4-bit target):
//!   * concat order = `[norm_e(emb(t_next)) ; norm_h(h_pre)]` — EMBEDDING FIRST
//!     (reversed → α collapses to 0.005).
//!   * ALL head RMSNorm weights need the **+1.0 shift** (mlx_lm qwen3_5 sanitize
//!     convention; without it α=0.0). This codebase stores norm weights already
//!     shifted (see `qwen35::gated_attention` + the P1 test), so the shift is
//!     baked in at load here.
//!   * hidden source = target pre-`model.norm` residual (DeepSeek-V3 NextN).
//!
//! The head layer math reuses the exact op sequence the target's full-attn+MoE
//! layers already run and validate against the MLX oracle
//! (`qwen35::Qwen35Model::gated_attention` / `moe`). The dense projections
//! (`fc`, `q/k/v/o_proj`) go through a pluggable matvec closure so ONE
//! implementation serves both the engine-less CPU parity path (Mac) and the
//! GPU-resident node path (`VulkanModel::qwen35_matvec` over uploaded buffers).
#![allow(dead_code)]

use crate::model::{cpu_matmul, cpu_rms_norm, cpu_sdpa, cpu_silu, cpu_rope, dequantize_mlx_affine, KvCache, ModelWeights, SimpleTensor};
use crate::moe::{self, MoeDims};
use crate::qwen35::Qwen35Config;
use std::collections::HashMap;
use std::path::Path;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dense SwiGLU MLP weights for the `qwen3_5_mtp` DENSE head (Qwen3.6-27B). The
/// 27B NextN head ships a plain SwiGLU block (`gate/up/down_proj`), NOT the
/// 256-expert MoE the 35B-A3B head carries. Row-major `[out, in]`; the forward
/// is `down · (silu(gate·x) ⊙ (up·x))`, bit-identical to the base 27B's
/// `Qwen35Model::dense_mlp`.
pub struct DenseMlp {
    /// `[inter, hidden]`
    pub gate: Vec<f32>,
    /// `[inter, hidden]`
    pub up: Vec<f32>,
    /// `[hidden, inter]`
    pub down: Vec<f32>,
    pub inter: usize,
}

/// One dense SwiGLU MLP step, `[hidden] -> [hidden]`. Mirrors
/// `qwen35::Qwen35Model::dense_mlp` exactly (silu = x·sigmoid(x)).
pub(crate) fn dense_swiglu(ff_in: &[f32], dm: &DenseMlp, hidden: usize) -> Vec<f32> {
    let gate = cpu_matmul(ff_in, &dm.gate, 1, hidden, dm.inter);
    let up = cpu_matmul(ff_in, &dm.up, 1, hidden, dm.inter);
    // `cpu_silu` (not an inline g·σ(g)) so this is BIT-identical to the base
    // 27B `Qwen35Model::dense_mlp` (same rounding), keeping the head/serial and
    // eventual GPU paths bit-comparable.
    let act = cpu_silu(&gate);
    let mid: Vec<f32> = act.iter().zip(&up).map(|(&a, &u)| a * u).collect();
    cpu_matmul(&mid, &dm.down, 1, dm.inter, hidden)
}

/// Names passed to the dense-matvec op. Each maps to a `[out, in]`
/// row-major weight; the op returns `W · x` (length `out`).
pub const MV_FC: &str = "mtp.fc";
pub const MV_Q: &str = "mtp.q_proj";
pub const MV_K: &str = "mtp.k_proj";
pub const MV_V: &str = "mtp.v_proj";
pub const MV_O: &str = "mtp.o_proj";

/// DENSE-head (`qwen3_5_mtp`, Qwen3.6-27B) SwiGLU-MLP GPU-resident keys. Present
/// in `gpu_weights` only when the head was uploaded (`VLLM_VULKAN_MTP_DENSE_GPU`,
/// default ON on-node); absent ⇒ `NodeOps::dense_mlp` runs the bit-exact host
/// `dense_swiglu`. Each maps to a `[out, in]` row-major f16 weight.
pub const MV_GATE: &str = "mtp.mlp.gate";
pub const MV_UP: &str = "mtp.mlp.up";
pub const MV_DOWN: &str = "mtp.mlp.down";

/// Pluggable backend for the head layer: the dense projections AND the
/// 256-expert MoE block. ONE `head_hidden_with` drives both the engine-less CPU
/// parity path (Mac — [`CpuOps`], all host f32) and the GPU-resident node path
/// (`VulkanModel`'s `NodeOps`: dense via `qwen35_matvec`, MoE via the fused f16
/// `mtp_moe_mlp_gpu` command buffer). A single `&mut dyn MtpOps` trait object
/// carries both operations so the node path can hold one `&mut VulkanModel`
/// across the whole head forward — two separate `FnMut(&mut self)` closures
/// (one for matvec, one for MoE) cannot coexist under the borrow checker.
pub trait MtpOps {
    /// `W · x` for the named dense projection (`MV_FC`/`MV_Q`/…), `[n]`.
    fn matvec(&mut self, name: &str, x: &[f32], k: usize, n: usize) -> Vec<f32>;
    /// Full MoE block output `[hidden]` for the post-attn-norm input `ff_in`
    /// (routed top-k experts + sigmoid-gated shared expert).
    fn moe(&mut self, ff_in: &[f32]) -> Vec<f32>;
    /// DENSE-head (`qwen3_5_mtp`, 27B) SwiGLU MLP `[hidden]->[hidden]`. Default =
    /// the bit-exact host `dense_swiglu` (used by the Mac/CPU parity path and the
    /// `VLLM_VULKAN_MTP_DENSE_GPU=0` fallback); the node path overrides it to run
    /// the three projections through the GPU-resident `qwen35_matvec`.
    fn dense_mlp(&mut self, ff_in: &[f32], dm: &DenseMlp, hidden: usize) -> Vec<f32> {
        dense_swiglu(ff_in, dm, hidden)
    }
}

/// Engine-less (Mac) `MtpOps`: dense matvecs from the head's own host f32
/// weights, MoE via the proven rayon CPU block. THE parity vehicle — its output
/// is bit-comparable to the target's MLX-validated full-attn+MoE ops. Owns the
/// weights (moved out of the head via `std::mem::take`) so it can be used while
/// the head is mutably borrowed for its KV.
pub struct CpuOps {
    pub fc: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub moe_w: moe::MoeWeights,
    pub dims: MoeDims,
}

impl MtpOps for CpuOps {
    fn matvec(&mut self, name: &str, x: &[f32], k: usize, n: usize) -> Vec<f32> {
        let w = match name {
            MV_FC => &self.fc,
            MV_Q => &self.q_proj,
            MV_K => &self.k_proj,
            MV_V => &self.v_proj,
            MV_O => &self.o_proj,
            other => panic!("unknown mtp matvec '{other}'"),
        };
        cpu_matmul(x, w, 1, k, n)
    }
    fn moe(&mut self, ff_in: &[f32]) -> Vec<f32> {
        moe::moe_forward_token_rayon(ff_in, &self.moe_w, self.dims).1
    }
}

/// Head-KV allocation cap (positions). The KV is tiny (nkv=2 × hd=256 × 2
/// planes × 4 B ≈ 4 KB/pos) so a generous cap costs ~16 MB and lets the parity
/// gate replay a full reference sequence. The P3 chain budget (`max_depth`) is a
/// separate, semantic limit.
const MTP_KV_CAP: usize = 4096;

/// The 7 head RMSNorm tensors that take the +1.0 sanitize shift.
const SHIFTED_NORMS: &[&str] = &[
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
];

/// The MTP draft head. Holds its own weights (f32 host, from the bf16 file) and
/// a tiny KV cache for its single attention layer (chainable, depth ≤ 4). embed
/// + lm_head are NOT held here — they are the main model's shared tables, applied
/// by the caller (the CPU parity harness feeds `embed(t_next)` and applies
/// lm_head via the oracle; the node path uses `q35_f16_host` + `qwen35_matvec`).
pub struct MtpHead {
    pub cfg: Qwen35Config,
    /// fc [hidden, 2*hidden] row-major (out=hidden, in=2*hidden). Applied to the
    /// concat `[norm_e ; norm_h]`.
    pub fc: Vec<f32>,
    // RMSNorm weights (all +1.0 shifted at load).
    pub enorm: Vec<f32>,      // pre_fc_norm_embedding
    pub hnorm: Vec<f32>,      // pre_fc_norm_hidden
    pub final_norm: Vec<f32>, // mtp.norm
    pub in_ln: Vec<f32>,      // layers.0.input_layernorm
    pub post_ln: Vec<f32>,    // layers.0.post_attention_layernorm
    pub q_norm: Vec<f32>,     // layers.0.self_attn.q_norm  (head_dim)
    pub k_norm: Vec<f32>,     // layers.0.self_attn.k_norm  (head_dim)
    // Dense projections (host f32; used by the default CPU matvec closure).
    pub q_proj: Vec<f32>, // [nq*2*hd, hidden]
    pub k_proj: Vec<f32>, // [nkv*hd, hidden]
    pub v_proj: Vec<f32>, // [nkv*hd, hidden]
    pub o_proj: Vec<f32>, // [hidden, nq*hd]
    // MoE weights (host f32). `moe_w` mirrors `moe::MoeWeights` layout so the
    // proven `moe::moe_forward_token` runs unchanged. Empty/default for the
    // DENSE (`qwen3_5_mtp`, Qwen3.6-27B) head — see `dense_mlp`.
    pub moe_w: moe::MoeWeights,
    pub dims: MoeDims,
    /// DENSE-head MLP (`qwen3_5_mtp`, Qwen3.6-27B). `Some` ⇒ the head layer's
    /// MLP is a plain SwiGLU (`layer_step` uses it instead of `ops.moe`); `None`
    /// ⇒ the 35B-A3B MoE path (`ops.moe`). Kept on the head (not moved into
    /// `CpuOps`) because the CPU/GPU parity for dense runs inline here.
    pub dense_mlp: Option<DenseMlp>,
    /// Own KV cache for the head's single full-attention layer.
    pub kv: KvCache,
    /// Chain depth budget (KV positions before a reset is required). ≤ 4 per plan.
    pub max_depth: usize,
}

impl MtpHead {
    /// Load the MTP head from `mtp.safetensors` (bf16 → f32) using the target
    /// model's geometry `cfg` (only `num_hidden_layers`/`layer_types` are
    /// overridden internally — the head is one full-attention layer). All 7 head
    /// RMSNorm weights get the +1.0 sanitize shift here.
    pub fn load(mtp_path: &Path, cfg: &Qwen35Config, max_depth: usize) -> Result<Self, String> {
        let raw = crate::model::load_weights_from_safetensors(mtp_path)?;
        Self::from_raw(&raw, cfg, max_depth)
    }

    /// Build from an already-loaded `name → f32` map (keys `mtp.*`). Split out so
    /// unit tests can drive it with a synthetic tensor set.
    pub fn from_raw(
        raw: &HashMap<String, Vec<f32>>,
        cfg: &Qwen35Config,
        max_depth: usize,
    ) -> Result<Self, String> {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let mi = cfg.moe_intermediate_size;
        let e = cfg.num_experts;

        let get = |k: &str| -> Result<Vec<f32>, String> {
            raw.get(k).cloned().ok_or_else(|| format!("MTP tensor '{k}' missing"))
        };
        // A shifted RMSNorm weight: +1.0 on every element (sanitize convention).
        let get_norm = |k: &str| -> Result<Vec<f32>, String> {
            Ok(get(k)?.into_iter().map(|v| v + 1.0).collect())
        };

        // gate_up_proj [E, 2*mi, hidden] → switch gate [E, mi, hidden] + up [E, mi, hidden].
        let gate_up = get("mtp.layers.0.mlp.experts.gate_up_proj")?;
        let e_blk = 2 * mi * h; // per-expert rows in gate_up
        let half = mi * h; // per-expert rows in gate (== up)
        if gate_up.len() != e * e_blk {
            return Err(format!(
                "gate_up_proj len {} != E*2*mi*h {}", gate_up.len(), e * e_blk));
        }
        let mut switch_gate = vec![0f32; e * half];
        let mut switch_up = vec![0f32; e * half];
        for ex in 0..e {
            switch_gate[ex * half..(ex + 1) * half]
                .copy_from_slice(&gate_up[ex * e_blk..ex * e_blk + half]);
            switch_up[ex * half..(ex + 1) * half]
                .copy_from_slice(&gate_up[ex * e_blk + half..ex * e_blk + 2 * half]);
        }

        let moe_w = moe::MoeWeights {
            gate: get("mtp.layers.0.mlp.gate.weight")?,
            switch_gate,
            switch_up,
            switch_down: get("mtp.layers.0.mlp.experts.down_proj")?, // [E, hidden, mi]
            shared_gate: get("mtp.layers.0.mlp.shared_expert.gate_proj.weight")?,
            shared_up: get("mtp.layers.0.mlp.shared_expert.up_proj.weight")?,
            shared_down: get("mtp.layers.0.mlp.shared_expert.down_proj.weight")?,
            shared_expert_gate: get("mtp.layers.0.mlp.shared_expert_gate.weight")?,
        };
        let dims = MoeDims {
            hidden: h,
            num_experts: e,
            top_k: cfg.num_experts_per_tok,
            moe_inter: mi,
            shared_inter: cfg.shared_expert_intermediate_size,
        };

        // head config: identical geometry, forced to one full-attention layer.
        let mut head_cfg = cfg.clone();
        head_cfg.num_hidden_layers = 1;
        head_cfg.layer_types = vec![crate::qwen35::LayerType::FullAttention];

        let head = MtpHead {
            cfg: head_cfg,
            fc: get("mtp.fc.weight")?,
            enorm: get_norm("mtp.pre_fc_norm_embedding.weight")?,
            hnorm: get_norm("mtp.pre_fc_norm_hidden.weight")?,
            final_norm: get_norm("mtp.norm.weight")?,
            in_ln: get_norm("mtp.layers.0.input_layernorm.weight")?,
            post_ln: get_norm("mtp.layers.0.post_attention_layernorm.weight")?,
            q_norm: get_norm("mtp.layers.0.self_attn.q_norm.weight")?,
            k_norm: get_norm("mtp.layers.0.self_attn.k_norm.weight")?,
            q_proj: get("mtp.layers.0.self_attn.q_proj.weight")?,
            k_proj: get("mtp.layers.0.self_attn.k_proj.weight")?,
            v_proj: get("mtp.layers.0.self_attn.v_proj.weight")?,
            o_proj: get("mtp.layers.0.self_attn.o_proj.weight")?,
            moe_w,
            dims,
            dense_mlp: None,
            // KV is allocated generously (independent of `max_depth`): the P2
            // parity gate replays the oracle's full causal pass over a reference
            // sequence, while `max_depth` is only the P3 chain-reset budget.
            kv: KvCache::new(MTP_KV_CAP, nkv, hd),
            max_depth,
        };
        // Shape sanity. These are CHECKPOINT-derived (`load` reads an arbitrary
        // `mtp.safetensors`), so a truncated or mis-paired head file must come
        // back as an `Err` the pyo3 boundary can raise — not a panic that aborts
        // through it. Same treatment the `gate_up_proj` length gets above.
        // `v_proj` is included: `layer_step` runs `ops.matvec(MV_V, ..)` on it,
        // and it had no check at all.
        for (name, got, want) in [
            ("fc", head.fc.len(), h * 2 * h),
            ("q_proj", head.q_proj.len(), nq * 2 * hd * h),
            ("k_proj", head.k_proj.len(), nkv * hd * h),
            ("v_proj", head.v_proj.len(), nkv * hd * h),
            ("o_proj", head.o_proj.len(), h * nq * hd),
        ] {
            if got != want {
                return Err(format!(
                    "MTP tensor '{name}' has {got} elements, expected {want}"));
            }
        }
        Ok(head)
    }

    /// P3 LEAN node path: build a head that holds ONLY the small tensors (the 7
    /// RMSNorm weights + KV + dims), leaving the big dense projections and MoE
    /// experts EMPTY on the host. The node loader (`load_mtp_gpu_lean`) streams
    /// those straight to the GPU (f16), so their multi-GB f32 host copies (and
    /// the bf16→f32 load transient) never materialize — the fix for the 14GB UMA
    /// rank OOM. On-node `NodeOps` always resolves dense via `gpu_weights` and
    /// MoE via `mtp_moe_gpu`, so the empty host fields are never read.
    ///
    /// `norms` maps the raw `mtp.*` norm tensor names → their (UNSHIFTED) f32
    /// weights; the +1.0 sanitize shift is applied here, exactly as `from_raw`.
    pub fn from_norms(
        cfg: &Qwen35Config,
        max_depth: usize,
        norms: &HashMap<String, Vec<f32>>,
    ) -> Result<Self, String> {
        let h = cfg.hidden_size;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let get_norm = |k: &str| -> Result<Vec<f32>, String> {
            norms
                .get(k)
                .map(|v| v.iter().map(|x| x + 1.0).collect())
                .ok_or_else(|| format!("MTP norm '{k}' missing (lean load)"))
        };
        let dims = MoeDims {
            hidden: h,
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            moe_inter: cfg.moe_intermediate_size,
            shared_inter: cfg.shared_expert_intermediate_size,
        };
        let mut head_cfg = cfg.clone();
        head_cfg.num_hidden_layers = 1;
        head_cfg.layer_types = vec![crate::qwen35::LayerType::FullAttention];
        Ok(MtpHead {
            cfg: head_cfg,
            fc: Vec::new(),
            enorm: get_norm("mtp.pre_fc_norm_embedding.weight")?,
            hnorm: get_norm("mtp.pre_fc_norm_hidden.weight")?,
            final_norm: get_norm("mtp.norm.weight")?,
            in_ln: get_norm("mtp.layers.0.input_layernorm.weight")?,
            post_ln: get_norm("mtp.layers.0.post_attention_layernorm.weight")?,
            q_norm: get_norm("mtp.layers.0.self_attn.q_norm.weight")?,
            k_norm: get_norm("mtp.layers.0.self_attn.k_norm.weight")?,
            q_proj: Vec::new(),
            k_proj: Vec::new(),
            v_proj: Vec::new(),
            o_proj: Vec::new(),
            moe_w: moe::MoeWeights::default(),
            dims,
            dense_mlp: None,
            kv: KvCache::new(MTP_KV_CAP, nkv, hd),
            max_depth,
        })
    }

    /// The raw `mtp.*` names of the 7 RMSNorm tensors the lean loader must keep
    /// on the host (small) to build the head. Exposed so the node loader's
    /// stream filter and `from_norms` agree on exactly this set.
    pub fn norm_tensor_names() -> &'static [&'static str] {
        SHIFTED_NORMS
    }

    /// Load the DENSE `qwen3_5_mtp` MTP head (Qwen3.6-27B, `mlx-community/
    /// Qwen3.6-27B-MTP-4bit`) from its single `model.safetensors`. Unlike the
    /// 35B (`from_raw`) head this:
    ///   * reads the head's OWN tensor names — **no `mtp.` prefix** (`fc.*`,
    ///     `pre_fc_norm_{hidden,embedding}.weight`, `norm.weight`, `layers.0.*`);
    ///   * mlx4-dequantizes the 4-bit projections (`weight`:u32 + `scales`/
    ///     `biases`:bf16, group 64) to f32 via `dequantize_mlx_affine` — the same
    ///     path the base 27B loader uses;
    ///   * loads the 7 RMSNorm weights **WITHOUT** the +1.0 shift — this
    ///     checkpoint ships BAKED gammas (verified: norm means ≈1.0), matching
    ///     the base-27B convention, NOT the zero-centered 35B sidecar. Applying
    ///     the shift here would collapse acceptance (the 35B α=0.0 failure mode).
    ///   * builds a DENSE SwiGLU MLP (`gate/up/down_proj`), not a 256-expert MoE.
    ///
    /// Concat order confirmed against OminiX-MLX `qwen3.6-mlx/src/mtp.rs:143`:
    /// `fc([norm_e(emb) ; norm_h(hidden)])` — EMBEDDING FIRST (reused verbatim
    /// from `fc_input`).
    pub fn load_dense(mtp_path: &Path, cfg: &Qwen35Config, max_depth: usize) -> Result<Self, String> {
        let raw = load_dense_mtp_weights(mtp_path)?;
        Self::from_raw_dense(&raw, cfg, max_depth)
    }

    /// Build the DENSE head from an already-dequantized `name → f32 [out,in]`
    /// map (keys = the head's own dense names, norms UNSHIFTED). Split out so a
    /// unit test can drive it with a synthetic tensor set (parity vs the base
    /// dense layer).
    pub fn from_raw_dense(
        raw: &HashMap<String, Vec<f32>>,
        cfg: &Qwen35Config,
        max_depth: usize,
    ) -> Result<Self, String> {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        if inter == 0 {
            return Err("dense MTP head needs cfg.intermediate_size > 0".to_string());
        }
        let get = |k: &str| -> Result<Vec<f32>, String> {
            raw.get(k).cloned().ok_or_else(|| format!("dense MTP tensor '{k}' missing"))
        };
        // NO +1 shift — baked gammas (see `load_dense` doc).
        let dense_mlp = Some(DenseMlp {
            gate: get("layers.0.mlp.gate_proj.weight")?,
            up: get("layers.0.mlp.up_proj.weight")?,
            down: get("layers.0.mlp.down_proj.weight")?,
            inter,
        });
        let dims = MoeDims { hidden: h, num_experts: 0, top_k: 0, moe_inter: 0, shared_inter: 0 };
        let mut head_cfg = cfg.clone();
        head_cfg.num_hidden_layers = 1;
        head_cfg.layer_types = vec![crate::qwen35::LayerType::FullAttention];
        let head = MtpHead {
            cfg: head_cfg,
            fc: get("fc.weight")?,
            enorm: get("pre_fc_norm_embedding.weight")?,
            hnorm: get("pre_fc_norm_hidden.weight")?,
            final_norm: get("norm.weight")?,
            in_ln: get("layers.0.input_layernorm.weight")?,
            post_ln: get("layers.0.post_attention_layernorm.weight")?,
            q_norm: get("layers.0.self_attn.q_norm.weight")?,
            k_norm: get("layers.0.self_attn.k_norm.weight")?,
            q_proj: get("layers.0.self_attn.q_proj.weight")?,
            k_proj: get("layers.0.self_attn.k_proj.weight")?,
            v_proj: get("layers.0.self_attn.v_proj.weight")?,
            o_proj: get("layers.0.self_attn.o_proj.weight")?,
            moe_w: moe::MoeWeights::default(),
            dims,
            dense_mlp,
            kv: KvCache::new(MTP_KV_CAP, nkv, hd),
            max_depth,
        };
        assert_eq!(head.fc.len(), h * 2 * h, "fc [{h},{}]", 2 * h);
        assert_eq!(head.q_proj.len(), nq * 2 * hd * h, "q_proj [{},{h}]", nq * 2 * hd);
        assert_eq!(head.k_proj.len(), nkv * hd * h, "k_proj");
        assert_eq!(head.o_proj.len(), h * nq * hd, "o_proj");
        if let Some(dm) = &head.dense_mlp {
            assert_eq!(dm.gate.len(), inter * h, "gate_proj [{inter},{h}]");
            assert_eq!(dm.up.len(), inter * h, "up_proj [{inter},{h}]");
            assert_eq!(dm.down.len(), h * inter, "down_proj [{h},{inter}]");
        }
        Ok(head)
    }

    /// Reset the head's KV (start a fresh draft chain). Call before drafting from
    /// a newly-accepted true token.
    pub fn reset(&mut self) {
        self.kv.seq_len = 0;
    }

    /// Build the fc input for one position: `fc · [norm_e(emb) ; norm_h(hidden)]`
    /// — EMBEDDING FIRST. `mv` runs the fc matvec (GPU on node / CPU on Mac).
    fn fc_input(
        &self,
        embed_next: &[f32],
        hidden_pre: &[f32],
        ops: &mut dyn MtpOps,
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;
        let ne = cpu_rms_norm(embed_next, &self.enorm, eps);
        let nh = cpu_rms_norm(hidden_pre, &self.hnorm, eps);
        let mut comb = Vec::with_capacity(2 * h);
        // STEP-7 draft bisect: `VLLM_VULKAN_MTP_CONCAT_FLIP=1` swaps to
        // `[norm_h ; norm_e]` so the harness can confirm the documented
        // EMBEDDING-FIRST order (validated at alpha_1=0.855) is the one that
        // accepts. Default = embedding first.
        if crate::flags::flags_global().mtp_concat_flip {
            comb.extend_from_slice(&nh); // hidden first (flipped)
            comb.extend_from_slice(&ne);
        } else {
            comb.extend_from_slice(&ne); // embedding FIRST
            comb.extend_from_slice(&nh);
        }
        ops.matvec(MV_FC, &comb, 2 * h, h)
    }

    /// One head-layer forward step at absolute position `pos`, advancing the
    /// head's own KV. Returns `out` = the layer residual output (pre-`mtp.norm`).
    /// Mirrors `qwen35::Qwen35Model::gated_attention` + MoE exactly, but routes
    /// the four attention projections through `mv`.
    fn layer_step(
        &mut self,
        x: &[f32],
        pos: usize,
        ops: &mut dyn MtpOps,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
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

        // ── Attention sub-block ──────────────────────────────────────────────
        let x_in = cpu_rms_norm(x, &self.in_ln, eps);
        // q_proj is double-width per head: [query(hd) | gate(hd)].
        let q_and_gate = ops.matvec(MV_Q, &x_in, h, nq * 2 * hd);
        let mut q = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for head in 0..nq {
            let base = head * 2 * hd;
            q[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base..base + hd]);
            gate[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base + hd..base + 2 * hd]);
        }
        let mut k = ops.matvec(MV_K, &x_in, h, kv_dim);
        let v = ops.matvec(MV_V, &x_in, h, kv_dim);

        // Per-head Q/K RMSNorm (before RoPE).
        for hi in 0..nq {
            let s = &mut q[hi * hd..(hi + 1) * hd];
            let n = cpu_rms_norm(s, &self.q_norm, eps);
            s.copy_from_slice(&n);
        }
        for hi in 0..nkv {
            let s = &mut k[hi * hd..(hi + 1) * hd];
            let n = cpu_rms_norm(s, &self.k_norm, eps);
            s.copy_from_slice(&n);
        }
        cpu_rope(&mut q, &mut k, pos, nq, nkv, hd, rotary, theta);

        // KV append + GQA causal SDPA over the head's own cache.
        self.kv.append(&k, &v);
        let attn = cpu_sdpa(
            &q, self.kv.k_up_to_now(), self.kv.v_up_to_now(),
            nq, nkv, hd, self.kv.seq_len, scale, None,
        );
        // Output gate then o_proj.
        let gated: Vec<f32> = attn.iter().zip(&gate).map(|(&a, &g)| a * sigmoid(g)).collect();
        let attn_out = ops.matvec(MV_O, &gated, q_dim, h);
        let h1: Vec<f32> = x.iter().zip(&attn_out).map(|(&r, &a)| r + a).collect();

        // ── MoE sub-block ────────────────────────────────────────────────────
        // Delegated to `ops.moe`: the node path (`NodeOps`) runs the fused f16
        // GPU command buffer (`mtp_moe_mlp_gpu`); the Mac/CPU path (`CpuOps`)
        // runs `moe::moe_forward_token_rayon` — bit-identical to the serial
        // `moe_forward_token` the parity tests below use, just parallelized.
        // Timed into its own profiler bucket (`mtp_draft_moe`) so the D
        // breakdown separates dense-matvec cost from MoE cost, regardless of
        // which backend ran it.
        let ff_in = cpu_rms_norm(&h1, &self.post_ln, eps);
        let t_mlp = std::time::Instant::now();
        // DENSE head (`qwen3_5_mtp`, 27B): plain SwiGLU via `ops.dense_mlp` — the
        // node path runs it on the GPU-resident projections (`qwen35_matvec`), the
        // Mac/CPU path runs the bit-exact host `dense_swiglu`. MoE head (35B-A3B):
        // the 256-expert block via `ops.moe`. SEPARATE profiler buckets so a dense
        // head reports `mtp_draft_mlp` and NO `mtp_draft_moe` (no MoE ever runs) —
        // the old shared `mtp_draft_moe` bucket mis-attributed the dense SwiGLU.
        let (moe_out, bucket) = if let Some(dm) = &self.dense_mlp {
            (ops.dense_mlp(&ff_in, dm, h), "mtp_draft_mlp")
        } else {
            (ops.moe(&ff_in), "mtp_draft_moe")
        };
        crate::prof_add_ns(bucket, t_mlp.elapsed().as_nanos());
        h1.iter().zip(&moe_out).map(|(&r, &m)| r + m).collect()
    }

    /// Full head forward for ONE draft position. `embed_next` = `embed(t_next)`
    /// (main table row), `hidden_pre` = target pre-`model.norm` residual, `pos` =
    /// the head-KV position (0 for the first draft off a fresh accept). Returns
    /// the head hidden `mtp.norm(out)` — apply the shared lm_head to it for the
    /// draft logits/argmax. Advances the head KV.
    pub fn head_hidden_with(
        &mut self,
        embed_next: &[f32],
        hidden_pre: &[f32],
        pos: usize,
        ops: &mut dyn MtpOps,
    ) -> Vec<f32> {
        self.head_forward_with(embed_next, hidden_pre, pos, ops).1
    }

    /// Like `head_hidden_with` but ALSO returns the PRE-`mtp.norm` layer residual
    /// `out` alongside the post-norm head hidden. The residual is what a P4
    /// autoregressive chain feeds back as the next draft's `hidden_pre` (plan §1.2
    /// "feed its own output hidden + drafted-token embedding back in for draft
    /// j+1"); the post-norm `head_hidden = mtp.norm(out)` is what the shared
    /// lm_head consumes for this position's draft logits. Returns
    /// `(residual, head_hidden)`. Advances the head KV by one position.
    pub fn head_forward_with(
        &mut self,
        embed_next: &[f32],
        hidden_pre: &[f32],
        pos: usize,
        ops: &mut dyn MtpOps,
    ) -> (Vec<f32>, Vec<f32>) {
        let x = self.fc_input(embed_next, hidden_pre, ops);
        let out = self.layer_step(&x, pos, ops);
        let head_hidden = cpu_rms_norm(&out, &self.final_norm, self.cfg.rms_norm_eps);
        (out, head_hidden)
    }

    /// Rewind the head KV to `len` positions (P4 chain rollback). The KV storage
    /// is overwrite-in-place (`KvCache::truncate`), so this is just the counter —
    /// the next chained draft overwrites the abandoned speculative positions. Used
    /// to drop the mispredicted suffix of a draft chain while KEEPING the confirmed
    /// prefix's KV (and RoPE context), so refill never resets the head to pos 0.
    pub fn kv_rewind(&mut self, len: usize) {
        self.kv.truncate(len.min(self.kv.seq_len));
    }

    /// Current head-KV length (positions consumed). Lets the P4 driver track the
    /// head's confirmed frontier for gap-free continuous refill.
    pub fn kv_len(&self) -> usize {
        self.kv.seq_len
    }

    /// Engine-less CPU convenience: dense matvecs + MoE from the host f32
    /// weights. This is THE parity path (Mac). Bit-comparable to the target's
    /// proven full-attn+MoE ops.
    pub fn head_hidden_cpu(&mut self, embed_next: &[f32], hidden_pre: &[f32], pos: usize) -> Vec<f32> {
        // Snapshot the proj + MoE weights out so the ops object can own them
        // while `self` is mutably borrowed for the KV (take/restore dance).
        let mut ops = CpuOps {
            fc: std::mem::take(&mut self.fc),
            q_proj: std::mem::take(&mut self.q_proj),
            k_proj: std::mem::take(&mut self.k_proj),
            v_proj: std::mem::take(&mut self.v_proj),
            o_proj: std::mem::take(&mut self.o_proj),
            moe_w: std::mem::take(&mut self.moe_w),
            dims: self.dims,
        };
        let out = self.head_hidden_with(embed_next, hidden_pre, pos, &mut ops);
        self.fc = ops.fc;
        self.q_proj = ops.q_proj;
        self.k_proj = ops.k_proj;
        self.v_proj = ops.v_proj;
        self.o_proj = ops.o_proj;
        self.moe_w = ops.moe_w;
        out
    }

    /// Engine-less CPU autoregressive draft chain (plan §P4) — THE Mac chain
    /// vehicle; the node path (`VulkanModel::mtp_draft_chain_impl`) mirrors it with
    /// GPU dense/MoE + the shared lm_head. Consumes `first_embed`/`first_hidden`
    /// (the just-committed token's embedding + the target pre-norm hidden) to draft
    /// d_1, then chains `depth-1` more: each step feeds the head's OWN residual as
    /// the next `hidden_pre` and `embed_of(d_{j})` as the next embedding (the NextN
    /// self-chain). `argmax_of` is the caller's shared lm_head (head hidden → token
    /// id). Returns `(drafts[depth], head_hiddens[depth])` and advances the head KV
    /// by `depth` positions from `start_pos`.
    pub fn head_chain_cpu(
        &mut self,
        first_embed: &[f32],
        first_hidden: &[f32],
        start_pos: usize,
        depth: usize,
        embed_of: &dyn Fn(u32) -> Vec<f32>,
        argmax_of: &dyn Fn(&[f32]) -> u32,
    ) -> (Vec<u32>, Vec<Vec<f32>>) {
        let mut ops = CpuOps {
            fc: std::mem::take(&mut self.fc),
            q_proj: std::mem::take(&mut self.q_proj),
            k_proj: std::mem::take(&mut self.k_proj),
            v_proj: std::mem::take(&mut self.v_proj),
            o_proj: std::mem::take(&mut self.o_proj),
            moe_w: std::mem::take(&mut self.moe_w),
            dims: self.dims,
        };
        let mut drafts = Vec::with_capacity(depth);
        let mut hiddens = Vec::with_capacity(depth);
        let mut embed = first_embed.to_vec();
        let mut hidden = first_hidden.to_vec();
        for j in 0..depth {
            let (residual, head_hidden) =
                self.head_forward_with(&embed, &hidden, start_pos + j, &mut ops);
            let tok = argmax_of(&head_hidden);
            embed = embed_of(tok);
            hidden = residual; // NextN self-chain: feed the head's own residual
            drafts.push(tok);
            hiddens.push(head_hidden);
        }
        self.fc = ops.fc;
        self.q_proj = ops.q_proj;
        self.k_proj = ops.k_proj;
        self.v_proj = ops.v_proj;
        self.o_proj = ops.o_proj;
        self.moe_w = ops.moe_w;
        (drafts, hiddens)
    }
}

/// True when `path`'s safetensors is the DENSE `qwen3_5_mtp` head (Qwen3.6-27B:
/// top-level `fc.weight`, no `mtp.` prefix), false when it is the 35B-A3B MoE
/// sidecar (`mtp.fc.weight`). Lets `load_mtp_gpu` dispatch the right loader from
/// the checkpoint alone (no config plumbing).
pub fn mtp_checkpoint_is_dense(path: &Path) -> Result<bool, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
    let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;
    let has_dense = st.tensor("fc.weight").is_ok();
    let has_moe = st.tensor("mtp.fc.weight").is_ok();
    if has_moe {
        Ok(false)
    } else if has_dense {
        Ok(true)
    } else {
        Err(format!("{}: not an MTP head (no fc.weight / mtp.fc.weight)", path.display()))
    }
}

/// Read the DENSE `qwen3_5_mtp` head's single `model.safetensors` into a
/// `name → f32 [out,in]` map: mlx4 triples (`.weight`:u32 + `.scales` + `.biases`)
/// dequantized to f32 under the base `.weight` name; every other float tensor
/// (RMSNorm weights, bf16/f16/f32) copied as-is (NO +1 shift). Standalone
/// because `model::load_weights_from_safetensors` SKIPS u32 packed tensors.
pub fn load_dense_mtp_weights(path: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;

    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
    let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;
    let tmap: HashMap<String, safetensors::tensor::TensorView> =
        st.tensors().into_iter().collect();

    let read_floats = |t: &safetensors::tensor::TensorView| -> Result<Vec<f32>, String> {
        let d = t.data();
        Ok(match t.dtype() {
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
            other => return Err(format!("unexpected float dtype {other:?}")),
        })
    };
    let read_u32 = |t: &safetensors::tensor::TensorView| -> Vec<u32> {
        t.data()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, t) in &tmap {
        if name.ends_with(".scales") || name.ends_with(".biases") {
            continue; // consumed alongside the sibling `.weight`
        }
        if let Some(base) = name.strip_suffix(".weight") {
            let sk = format!("{base}.scales");
            if let Some(s) = tmap.get(&sk) {
                // mlx4-quantized projection.
                let b = tmap
                    .get(&format!("{base}.biases"))
                    .ok_or_else(|| format!("{base}: .scales present but .biases missing"))?;
                // A sibling `.scales` name is the ONLY evidence so far that this
                // is an mlx4 triple. `read_u32` reinterprets raw bytes with no
                // dtype check, so a BF16/F16 `.weight` would dequantize to
                // garbage that `from_raw_dense`'s length assert still accepts
                // (the element count is unchanged). Check dtype and rank before
                // the `shape()[1]` reads below, which panic on a 1-D tensor.
                if t.dtype() != safetensors::Dtype::U32 {
                    return Err(format!(
                        "{name}: .scales present but the packed weight dtype is {:?}, expected U32",
                        t.dtype()));
                }
                if t.shape().len() != 2 || s.shape().len() != 2 {
                    return Err(format!(
                        "{name}: an mlx4 triple needs 2-D weight and scales; got {:?} / {:?}",
                        t.shape(), s.shape()));
                }
                let out_f = t.shape()[0];
                let in_f = t.shape()[1] * 8; // 8 × 4-bit nibbles per u32 word
                let groups = s.shape()[1];
                if groups == 0 || in_f % groups != 0 {
                    return Err(format!("{name}: bad group geometry in={in_f} groups={groups}"));
                }
                let gsize = in_f / groups;
                let deq = dequantize_mlx_affine(
                    &read_u32(t),
                    &read_floats(s)?,
                    &read_floats(b)?,
                    out_f,
                    in_f,
                    gsize,
                    4,
                );
                out.insert(name.clone(), deq);
                continue;
            }
        }
        // plain float tensor (RMSNorm weight etc.) — as-is, NO shift.
        out.insert(name.clone(), read_floats(t)?);
    }
    Ok(out)
}

/// Build a `ModelWeights` map for the head layer under the target's
/// `model.layers.0.*` names, so a 1-layer `Qwen35Model` reproduces the head
/// exactly (the cross-check reference in tests). Kept here so the reuse story is
/// explicit and testable.
pub fn head_as_layer0_weights(raw: &HashMap<String, Vec<f32>>, cfg: &Qwen35Config) -> ModelWeights {
    let mi = cfg.moe_intermediate_size;
    let h = cfg.hidden_size;
    let e = cfg.num_experts;
    let mut t: HashMap<String, SimpleTensor> = HashMap::new();
    let mut ins = |name: &str, data: Vec<f32>| {
        t.insert(name.to_string(), SimpleTensor { data, shape: vec![] });
    };
    let shift = |k: &str| -> Vec<f32> { raw[k].iter().map(|v| v + 1.0).collect() };
    ins("model.layers.0.input_layernorm.weight", shift("mtp.layers.0.input_layernorm.weight"));
    ins("model.layers.0.post_attention_layernorm.weight", shift("mtp.layers.0.post_attention_layernorm.weight"));
    ins("model.layers.0.self_attn.q_proj.weight", raw["mtp.layers.0.self_attn.q_proj.weight"].clone());
    ins("model.layers.0.self_attn.k_proj.weight", raw["mtp.layers.0.self_attn.k_proj.weight"].clone());
    ins("model.layers.0.self_attn.v_proj.weight", raw["mtp.layers.0.self_attn.v_proj.weight"].clone());
    ins("model.layers.0.self_attn.o_proj.weight", raw["mtp.layers.0.self_attn.o_proj.weight"].clone());
    ins("model.layers.0.self_attn.q_norm.weight", shift("mtp.layers.0.self_attn.q_norm.weight"));
    ins("model.layers.0.self_attn.k_norm.weight", shift("mtp.layers.0.self_attn.k_norm.weight"));
    // MoE
    ins("model.layers.0.mlp.gate.weight", raw["mtp.layers.0.mlp.gate.weight"].clone());
    let gate_up = &raw["mtp.layers.0.mlp.experts.gate_up_proj"];
    let e_blk = 2 * mi * h;
    let half = mi * h;
    let mut sg = vec![0f32; e * half];
    let mut su = vec![0f32; e * half];
    for ex in 0..e {
        sg[ex * half..(ex + 1) * half].copy_from_slice(&gate_up[ex * e_blk..ex * e_blk + half]);
        su[ex * half..(ex + 1) * half].copy_from_slice(&gate_up[ex * e_blk + half..ex * e_blk + 2 * half]);
    }
    ins("model.layers.0.mlp.switch_mlp.gate_proj.weight", sg);
    ins("model.layers.0.mlp.switch_mlp.up_proj.weight", su);
    ins("model.layers.0.mlp.switch_mlp.down_proj.weight", raw["mtp.layers.0.mlp.experts.down_proj"].clone());
    ins("model.layers.0.mlp.shared_expert.gate_proj.weight", raw["mtp.layers.0.mlp.shared_expert.gate_proj.weight"].clone());
    ins("model.layers.0.mlp.shared_expert.up_proj.weight", raw["mtp.layers.0.mlp.shared_expert.up_proj.weight"].clone());
    ins("model.layers.0.mlp.shared_expert.down_proj.weight", raw["mtp.layers.0.mlp.shared_expert.down_proj.weight"].clone());
    ins("model.layers.0.mlp.shared_expert_gate.weight", raw["mtp.layers.0.mlp.shared_expert_gate.weight"].clone());
    ModelWeights { tensors: t }
}

#[cfg(test)]
mod tests {
    //! Mac-runnable (engine-less) P2 head-math gates: shapes, the EMBEDDING-FIRST
    //! concat order, the +1.0 RMSNorm sanitize shift, fc dims, and — the strong
    //! one — bit-exact parity of the inline head layer vs the PROVEN
    //! `Qwen35Model` full-attn+MoE path (which is validated against the MLX
    //! oracle). If `layer_step` diverged from the target geometry in ANY op
    //! (concat order, shift, q_proj gate split, GQA, rope, MoE routing), the
    //! cross-check fails.
    use super::*;
    use crate::qwen35::{LayerType, Qwen35Config, Qwen35Model};

    fn gen(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// Tiny target-geometry config (full-attn + MoE), same shape family as 35B.
    fn tiny_cfg() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 8,
            num_hidden_layers: 1,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            attn_output_gate: true,
            rope_theta: 1e7,
            partial_rotary_factor: 0.5, // rotary_dim = 2
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 1,
            linear_conv_kernel_dim: 1,
            intermediate_size: 0,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 6,
            shared_expert_intermediate_size: 6,
            layer_types: vec![LayerType::FullAttention],
        }
    }

    /// Synthetic `mtp.*` tensor set matching `tiny_cfg`.
    fn tiny_raw(cfg: &Qwen35Config) -> HashMap<String, Vec<f32>> {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let mi = cfg.moe_intermediate_size;
        let si = cfg.shared_expert_intermediate_size;
        let e = cfg.num_experts;
        let mut g = gen(0xABCDEF);
        let mut mk = |n: usize, scale: f32| -> Vec<f32> { (0..n).map(|_| g() * scale).collect() };
        let mut m = HashMap::new();
        m.insert("mtp.fc.weight".into(), mk(h * 2 * h, 0.1));
        for k in [
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.norm.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
        ] {
            m.insert(k.into(), mk(h, 0.1));
        }
        m.insert("mtp.layers.0.self_attn.q_norm.weight".into(), mk(hd, 0.1));
        m.insert("mtp.layers.0.self_attn.k_norm.weight".into(), mk(hd, 0.1));
        m.insert("mtp.layers.0.self_attn.q_proj.weight".into(), mk(nq * 2 * hd * h, 0.1));
        m.insert("mtp.layers.0.self_attn.k_proj.weight".into(), mk(nkv * hd * h, 0.1));
        m.insert("mtp.layers.0.self_attn.v_proj.weight".into(), mk(nkv * hd * h, 0.1));
        m.insert("mtp.layers.0.self_attn.o_proj.weight".into(), mk(h * nq * hd, 0.1));
        m.insert("mtp.layers.0.mlp.gate.weight".into(), mk(e * h, 0.1));
        m.insert("mtp.layers.0.mlp.experts.gate_up_proj".into(), mk(e * 2 * mi * h, 0.1));
        m.insert("mtp.layers.0.mlp.experts.down_proj".into(), mk(e * h * mi, 0.1));
        m.insert("mtp.layers.0.mlp.shared_expert.gate_proj.weight".into(), mk(si * h, 0.1));
        m.insert("mtp.layers.0.mlp.shared_expert.up_proj.weight".into(), mk(si * h, 0.1));
        m.insert("mtp.layers.0.mlp.shared_expert.down_proj.weight".into(), mk(h * si, 0.1));
        m.insert("mtp.layers.0.mlp.shared_expert_gate.weight".into(), mk(h, 0.1));
        m
    }

    /// A checkpoint-shape mismatch must be an `Err`, not a panic.
    ///
    /// `from_raw` is reached from `load`, which reads an ARBITRARY
    /// `mtp.safetensors`, and it returns `Result<_, String>` that the pyo3 seam
    /// turns into a Python exception. A truncated or mis-paired head file is
    /// therefore user input, not a programmer error: an `assert_eq!` here aborts
    /// the process through the boundary instead of raising.
    ///
    /// `v_proj` is in the loop because it had NO check at all, while
    /// `layer_step` runs `ops.matvec(MV_V, ..)` on it — a short `v_proj` used to
    /// reach the matvec and panic there.
    #[test]
    fn from_raw_reports_a_bad_tensor_shape_as_an_error() {
        let cfg = tiny_cfg();
        for name in ["mtp.fc.weight",
                     "mtp.layers.0.self_attn.q_proj.weight",
                     "mtp.layers.0.self_attn.k_proj.weight",
                     "mtp.layers.0.self_attn.v_proj.weight",
                     "mtp.layers.0.self_attn.o_proj.weight"] {
            let mut raw = tiny_raw(&cfg);
            let full = raw[name].len();
            raw.get_mut(name).unwrap().truncate(full - 1); // a truncated shard
            let short = name.rsplit('.').nth(1).unwrap();
            let err = match MtpHead::from_raw(&raw, &cfg, 4) {
                Ok(_) => panic!("{name}: a truncated tensor must not build a head"),
                Err(e) => e,
            };
            assert!(err.contains(short) && err.contains(&(full - 1).to_string())
                        && err.contains(&full.to_string()),
                    "{name}: the error must name the tensor and both counts; got: {err}");
        }
        // ...and the intact set still builds.
        MtpHead::from_raw(&tiny_raw(&cfg), &cfg, 4).expect("the intact fixture must still build");
    }

    /// `load_dense_mtp_weights` treats a tensor as mlx4-packed on the strength of
    /// a sibling `.scales` NAME alone, then reads it with `read_u32`, which
    /// reinterprets raw bytes with no dtype check. Two malformed layouts must be
    /// refused rather than dequantized into silent garbage:
    ///
    ///   * a BF16/F16 `.weight` — `read_u32` would produce wrong words, and
    ///     `from_raw_dense`'s length assert still passes because dequantizing
    ///     does not change the element count, so the model would simply be
    ///     WRONG; and
    ///   * a 1-D `.weight` or `.scales` — the `shape()[1]` reads panic, and for
    ///     `.weight` they happen before `read_floats` could report the dtype.
    #[test]
    fn dense_loader_refuses_a_packed_tensor_of_the_wrong_dtype_or_rank() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let dir = std::env::temp_dir().join(format!(
            "vv-mtp-dense-{}-{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();

        // out=2, in=16 -> 2 packed u32 words per row; 1 group of 16.
        let write = |tag: &str, w: (Dtype, Vec<usize>, Vec<u8>), s_shape: Vec<usize>| {
            let scales = vec![0u8; s_shape.iter().product::<usize>() * 4];
            let biases = scales.clone();
            let wv = TensorView::new(w.0, w.1, &w.2).unwrap();
            let sv = TensorView::new(Dtype::F32, s_shape.clone(), &scales).unwrap();
            let bv = TensorView::new(Dtype::F32, s_shape, &biases).unwrap();
            let bytes = safetensors::serialize(
                [("q.weight", &wv), ("q.scales", &sv), ("q.biases", &bv)], &None).unwrap();
            let path = dir.join(format!("{tag}.safetensors"));
            std::fs::write(&path, bytes).unwrap();
            path
        };

        // Baseline: a WELL-FORMED mlx4 triple must still load, so the two
        // refusals below are the malformation and not the guard over-firing.
        let ok = write("ok", (Dtype::U32, vec![2, 2], vec![0u8; 2 * 2 * 4]), vec![2, 1]);
        let m = super::load_dense_mtp_weights(&ok).expect("a well-formed mlx4 triple must load");
        assert_eq!(m["q.weight"].len(), 2 * 16, "out*in f32 weights");

        // (a) wrong dtype: same byte count, declared BF16.
        let bad_dtype = write("bad_dtype", (Dtype::BF16, vec![2, 8], vec![0u8; 2 * 8 * 2]), vec![2, 1]);
        let e = super::load_dense_mtp_weights(&bad_dtype)
            .expect_err("a non-U32 packed weight must be refused");
        assert!(e.contains("expected U32"), "got: {e}");

        // (b) 1-D packed weight: `shape()[1]` would panic.
        let bad_rank = write("bad_rank", (Dtype::U32, vec![4], vec![0u8; 4 * 4]), vec![2, 1]);
        let e = super::load_dense_mtp_weights(&bad_rank)
            .expect_err("a 1-D packed weight must be refused");
        assert!(e.contains("2-D weight and scales"), "got: {e}");

        // (c) 1-D scales: `s.shape()[1]` would panic.
        let bad_scales = write("bad_scales", (Dtype::U32, vec![2, 2], vec![0u8; 2 * 2 * 4]), vec![2]);
        let e = super::load_dense_mtp_weights(&bad_scales)
            .expect_err("1-D scales must be refused");
        assert!(e.contains("2-D weight and scales"), "got: {e}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shift_applied_to_all_seven_norms() {
        let cfg = tiny_cfg();
        let raw = tiny_raw(&cfg);
        let head = MtpHead::from_raw(&raw, &cfg, 4).unwrap();
        let plus1 = |k: &str| -> Vec<f32> { raw[k].iter().map(|v| v + 1.0).collect() };
        assert_eq!(head.enorm, plus1("mtp.pre_fc_norm_embedding.weight"));
        assert_eq!(head.hnorm, plus1("mtp.pre_fc_norm_hidden.weight"));
        assert_eq!(head.final_norm, plus1("mtp.norm.weight"));
        assert_eq!(head.in_ln, plus1("mtp.layers.0.input_layernorm.weight"));
        assert_eq!(head.post_ln, plus1("mtp.layers.0.post_attention_layernorm.weight"));
        assert_eq!(head.q_norm, plus1("mtp.layers.0.self_attn.q_norm.weight"));
        assert_eq!(head.k_norm, plus1("mtp.layers.0.self_attn.k_norm.weight"));
        // fc / projections are NOT shifted.
        assert_eq!(head.fc, raw["mtp.fc.weight"]);
    }

    #[test]
    fn shapes_and_output_len() {
        let cfg = tiny_cfg();
        let raw = tiny_raw(&cfg);
        let mut head = MtpHead::from_raw(&raw, &cfg, 4).unwrap();
        let h = cfg.hidden_size;
        let mut g = gen(1);
        let e: Vec<f32> = (0..h).map(|_| g()).collect();
        let hp: Vec<f32> = (0..h).map(|_| g()).collect();
        let out = head.head_hidden_cpu(&e, &hp, 0);
        assert_eq!(out.len(), h);
        assert_eq!(head.kv.seq_len, 1, "KV advanced by one position");
    }

    #[test]
    fn concat_order_is_embedding_first_and_matters() {
        // Swapping the two inputs must change the output (enorm≠hnorm, and fc
        // treats the two halves distinctly) — a guard that the concat isn't a
        // symmetric mush and that embed occupies the FIRST half.
        let cfg = tiny_cfg();
        let raw = tiny_raw(&cfg);
        let h = cfg.hidden_size;
        let mut g = gen(2);
        let a: Vec<f32> = (0..h).map(|_| g()).collect();
        let b: Vec<f32> = (0..h).map(|_| g()).collect();
        let mut h1 = MtpHead::from_raw(&raw, &cfg, 4).unwrap();
        let o_ab = h1.head_hidden_cpu(&a, &b, 0);
        let mut h2 = MtpHead::from_raw(&raw, &cfg, 4).unwrap();
        let o_ba = h2.head_hidden_cpu(&b, &a, 0);
        let diff: f32 = o_ab.iter().zip(&o_ba).map(|(&x, &y)| (x - y).abs()).sum();
        assert!(diff > 1e-4, "embed/hidden slots must be distinct (diff={diff})");
    }

    #[test]
    fn inline_head_matches_qwen35model_layer_bitexact() {
        // THE strong gate: the inline `layer_step` must reproduce the proven
        // `Qwen35Model` full-attn+MoE layer (MLX-validated) bit-for-bit, and the
        // fc/concat/norm wrapper must match a hand-built reference — over a
        // multi-position causal chain (so the KV/rope path is exercised).
        let cfg = tiny_cfg();
        let raw = tiny_raw(&cfg);
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        // Reference: a 1-layer Qwen35Model built from the SAME head weights.
        let mut ref_cfg = cfg.clone();
        ref_cfg.num_hidden_layers = 1;
        ref_cfg.layer_types = vec![LayerType::FullAttention];
        let weights = head_as_layer0_weights(&raw, &cfg);
        let mut refm = Qwen35Model::new(ref_cfg, weights, 16, "unused".into());
        let enorm: Vec<f32> = raw["mtp.pre_fc_norm_embedding.weight"].iter().map(|v| v + 1.0).collect();
        let hnorm: Vec<f32> = raw["mtp.pre_fc_norm_hidden.weight"].iter().map(|v| v + 1.0).collect();
        let fnorm: Vec<f32> = raw["mtp.norm.weight"].iter().map(|v| v + 1.0).collect();
        let fc = raw["mtp.fc.weight"].clone();

        let mut head = MtpHead::from_raw(&raw, &cfg, 8).unwrap();
        let mut g = gen(99);
        for pos in 0..4 {
            let e: Vec<f32> = (0..h).map(|_| g()).collect();
            let hp: Vec<f32> = (0..h).map(|_| g()).collect();
            // reference fc input
            let ne = cpu_rms_norm(&e, &enorm, eps);
            let nh = cpu_rms_norm(&hp, &hnorm, eps);
            let mut comb = ne.clone();
            comb.extend_from_slice(&nh);
            let x = cpu_matmul(&comb, &fc, 1, 2 * h, h);
            let out_ref = refm.forward_pp_range(&x, pos, 0, 1);
            let hid_ref = cpu_rms_norm(&out_ref, &fnorm, eps);
            // inline head
            let hid = head.head_hidden_cpu(&e, &hp, pos);
            assert_eq!(hid, hid_ref, "pos {pos}: inline head != Qwen35Model layer");
        }
    }

    // ─────────────────────────── P4 chain tests ───────────────────────────

    #[test]
    fn head_chain_advances_kv_and_rewinds() {
        // `head_chain_cpu` must advance the head KV by exactly `depth`, and
        // `kv_rewind` must roll it back to a confirmed length (KvCache.truncate
        // semantics) so continuous refill keeps the confirmed prefix's RoPE
        // context instead of resetting.
        let cfg = tiny_cfg();
        let raw = tiny_raw(&cfg);
        let mut head = MtpHead::from_raw(&raw, &cfg, 8).unwrap();
        let h = cfg.hidden_size;
        let mut g = gen(7);
        let e0: Vec<f32> = (0..h).map(|_| g()).collect();
        let hp0: Vec<f32> = (0..h).map(|_| g()).collect();
        // fake embed table (16 rows) + fake lm_head argmax (sum-of-hidden bucket).
        let embed_tab: Vec<Vec<f32>> = (0..16)
            .map(|r| {
                let mut gg = gen(100 + r as u64);
                (0..h).map(|_| gg()).collect()
            })
            .collect();
        let embed_of = |t: u32| embed_tab[(t as usize) % 16].clone();
        let argmax_of = |hid: &[f32]| -> u32 {
            let s: f32 = hid.iter().sum();
            ((s.abs() * 1000.0) as u32) % 16
        };
        assert_eq!(head.kv_len(), 0);
        let (drafts, hiddens) = head.head_chain_cpu(&e0, &hp0, 0, 4, &embed_of, &argmax_of);
        assert_eq!(drafts.len(), 4);
        assert_eq!(hiddens.len(), 4);
        assert_eq!(head.kv_len(), 4, "chain of depth 4 advances KV by 4");
        // rewind to the confirmed frontier (say 2 of 4 accepted).
        head.kv_rewind(2);
        assert_eq!(head.kv_len(), 2, "rewind drops the mispredicted suffix");
        // rewind never extends.
        head.kv_rewind(9);
        assert_eq!(head.kv_len(), 2, "rewind is a floor, never grows the KV");
    }

    #[test]
    fn head_chain_uses_history_not_reset() {
        // The chain's KV history must MATTER: drafting position 1 as the 2nd step
        // of a chain (KV holds position 0) differs from drafting it with a fresh
        // KV — proof the head attends over the accumulated draft context (so the
        // continuous-refill KV preservation is meaningful, not decorative).
        let cfg = tiny_cfg();
        let raw = tiny_raw(&cfg);
        let h = cfg.hidden_size;
        let mut g = gen(11);
        let e0: Vec<f32> = (0..h).map(|_| g()).collect();
        let hp0: Vec<f32> = (0..h).map(|_| g()).collect();
        let e1: Vec<f32> = (0..h).map(|_| g()).collect();
        let hp1: Vec<f32> = (0..h).map(|_| g()).collect();

        // Chained: consume pos 0, then consume pos 1 (KV has pos 0).
        let mut chained = MtpHead::from_raw(&raw, &cfg, 8).unwrap();
        let _ = chained.head_hidden_cpu(&e0, &hp0, 0);
        let hid_chained = chained.head_hidden_cpu(&e1, &hp1, 1);

        // Fresh: consume pos 1 with an empty KV.
        let mut fresh = MtpHead::from_raw(&raw, &cfg, 8).unwrap();
        let hid_fresh = fresh.head_hidden_cpu(&e1, &hp1, 1);

        let diff: f32 = hid_chained.iter().zip(&hid_fresh).map(|(&a, &b)| (a - b).abs()).sum();
        assert!(diff > 1e-5, "chain KV history must change the draft (diff={diff})");
    }

    // ───────────────────── DENSE head (qwen3_5_mtp, 27B) ─────────────────────

    /// Tiny DENSE-head config: full-attention + a plain SwiGLU MLP (num_experts
    /// = 0 ⇒ `Qwen35Model` routes to `dense_mlp`). Same attention geometry as the
    /// 27B head (attn_output_gate, partial rope, GQA), just tiny dims.
    fn tiny_dense_cfg() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 8,
            num_hidden_layers: 1,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            attn_output_gate: true,
            rope_theta: 1e7,
            partial_rotary_factor: 0.5, // rotary_dim = 2
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 1,
            linear_conv_kernel_dim: 1,
            intermediate_size: 6, // DENSE SwiGLU width
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            layer_types: vec![LayerType::FullAttention],
        }
    }

    /// Synthetic DENSE-head weight map — the head's OWN names (no `mtp.` prefix),
    /// norms UNSHIFTED (baked, as the real 4-bit checkpoint ships them).
    fn tiny_dense_raw(cfg: &Qwen35Config) -> HashMap<String, Vec<f32>> {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let mut g = gen(0x5151DE);
        let mut mk = |n: usize, scale: f32| -> Vec<f32> { (0..n).map(|_| g() * scale).collect() };
        // Norm gammas drawn around ~1.0 to mimic BAKED weights (mean≈1).
        let mut m = HashMap::new();
        m.insert("fc.weight".into(), mk(h * 2 * h, 0.1));
        for k in [
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
            "norm.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
        ] {
            let nrm: Vec<f32> = mk(h, 0.1).into_iter().map(|v| 1.0 + v).collect();
            m.insert(k.into(), nrm);
        }
        let qn: Vec<f32> = mk(hd, 0.1).into_iter().map(|v| 1.0 + v).collect();
        m.insert("layers.0.self_attn.q_norm.weight".into(), qn);
        let kn: Vec<f32> = mk(hd, 0.1).into_iter().map(|v| 1.0 + v).collect();
        m.insert("layers.0.self_attn.k_norm.weight".into(), kn);
        m.insert("layers.0.self_attn.q_proj.weight".into(), mk(nq * 2 * hd * h, 0.1));
        m.insert("layers.0.self_attn.k_proj.weight".into(), mk(nkv * hd * h, 0.1));
        m.insert("layers.0.self_attn.v_proj.weight".into(), mk(nkv * hd * h, 0.1));
        m.insert("layers.0.self_attn.o_proj.weight".into(), mk(h * nq * hd, 0.1));
        m.insert("layers.0.mlp.gate_proj.weight".into(), mk(inter * h, 0.1));
        m.insert("layers.0.mlp.up_proj.weight".into(), mk(inter * h, 0.1));
        m.insert("layers.0.mlp.down_proj.weight".into(), mk(h * inter, 0.1));
        m
    }

    /// THE strong P1 gate for the DENSE head: the inline dense `layer_step` +
    /// fc/concat/final-norm wrapper must reproduce a 1-layer `Qwen35Model` DENSE
    /// forward (full-attn + SwiGLU, MLX-validated) BIT-FOR-BIT over a causal
    /// chain — proving the dense-MLP branch, the EMBEDDING-FIRST concat, and the
    /// NO-SHIFT norm convention are all wired correctly. Mirrors the MoE
    /// `inline_head_matches_qwen35model_layer_bitexact` gate.
    #[test]
    fn dense_head_matches_qwen35model_dense_bitexact() {
        let cfg = tiny_dense_cfg();
        let raw = tiny_dense_raw(&cfg);
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        // Reference: 1-layer dense Qwen35Model from the SAME (unshifted) weights.
        let mut ref_cfg = cfg.clone();
        ref_cfg.num_hidden_layers = 1;
        ref_cfg.layer_types = vec![LayerType::FullAttention];
        let mut t: HashMap<String, SimpleTensor> = HashMap::new();
        let mut ins = |name: &str, data: Vec<f32>| {
            t.insert(name.to_string(), SimpleTensor { data, shape: vec![] });
        };
        // NO +1 shift — baked norms (unlike `head_as_layer0_weights`).
        ins("model.layers.0.input_layernorm.weight", raw["layers.0.input_layernorm.weight"].clone());
        ins("model.layers.0.post_attention_layernorm.weight", raw["layers.0.post_attention_layernorm.weight"].clone());
        ins("model.layers.0.self_attn.q_proj.weight", raw["layers.0.self_attn.q_proj.weight"].clone());
        ins("model.layers.0.self_attn.k_proj.weight", raw["layers.0.self_attn.k_proj.weight"].clone());
        ins("model.layers.0.self_attn.v_proj.weight", raw["layers.0.self_attn.v_proj.weight"].clone());
        ins("model.layers.0.self_attn.o_proj.weight", raw["layers.0.self_attn.o_proj.weight"].clone());
        ins("model.layers.0.self_attn.q_norm.weight", raw["layers.0.self_attn.q_norm.weight"].clone());
        ins("model.layers.0.self_attn.k_norm.weight", raw["layers.0.self_attn.k_norm.weight"].clone());
        ins("model.layers.0.mlp.gate_proj.weight", raw["layers.0.mlp.gate_proj.weight"].clone());
        ins("model.layers.0.mlp.up_proj.weight", raw["layers.0.mlp.up_proj.weight"].clone());
        ins("model.layers.0.mlp.down_proj.weight", raw["layers.0.mlp.down_proj.weight"].clone());
        let weights = ModelWeights { tensors: t };
        let mut refm = Qwen35Model::new(ref_cfg, weights, 16, "unused".into());

        let enorm = raw["pre_fc_norm_embedding.weight"].clone();
        let hnorm = raw["pre_fc_norm_hidden.weight"].clone();
        let fnorm = raw["norm.weight"].clone();
        let fc = raw["fc.weight"].clone();

        let mut head = MtpHead::from_raw_dense(&raw, &cfg, 8).unwrap();
        assert!(head.dense_mlp.is_some(), "dense head must carry a DenseMlp");
        let mut g = gen(4242);
        for pos in 0..4 {
            let e: Vec<f32> = (0..h).map(|_| g()).collect();
            let hp: Vec<f32> = (0..h).map(|_| g()).collect();
            // reference fc input: EMBEDDING FIRST, no shift.
            let ne = cpu_rms_norm(&e, &enorm, eps);
            let nh = cpu_rms_norm(&hp, &hnorm, eps);
            let mut comb = ne.clone();
            comb.extend_from_slice(&nh);
            let x = cpu_matmul(&comb, &fc, 1, 2 * h, h);
            let out_ref = refm.forward_pp_range(&x, pos, 0, 1);
            let hid_ref = cpu_rms_norm(&out_ref, &fnorm, eps);
            let hid = head.head_hidden_cpu(&e, &hp, pos);
            assert_eq!(hid, hid_ref, "pos {pos}: inline dense head != Qwen35Model dense layer");
        }
    }

    /// Real-checkpoint P1 gate (opt-in): set `MTP_DENSE_CKPT=<dir>` to the
    /// `mlx-community/Qwen3.6-27B-MTP-4bit` dir (must hold `config.json` +
    /// `model.safetensors`). Auto-skips when unset (CI / no NAS mount). Loads the
    /// real 4-bit head via `load_dense`, drafts a depth-4 chain off a synthetic
    /// base hidden, and asserts every drafted hidden is FINITE + full-length +
    /// non-degenerate, and the head KV advanced — the equivalent of the A3B
    /// MTP_DRAFT_DEBUG sanity that caught the format bugs. With a real lm_head
    /// this would also gate argmax-in-vocab; here it proves loader + forward are
    /// sound on the actual weights (dequant, names, NO-shift).
    #[test]
    fn dense_head_loads_real_checkpoint_and_drafts_finite() {
        let dir = match std::env::var("MTP_DENSE_CKPT") {
            Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
            _ => {
                eprintln!("skipping dense_head_loads_real_checkpoint_and_drafts_finite: set MTP_DENSE_CKPT");
                return;
            }
        };
        let cfg_json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
        let cfg_v: serde_json::Value = serde_json::from_str(&cfg_json).expect("parse config.json");
        let cfg = Qwen35Config::from_json(&cfg_v).expect("build Qwen35Config");
        assert_eq!(cfg.hidden_size, 5120, "expected 27B hidden 5120");
        assert_eq!(cfg.intermediate_size, 17408, "expected dense inter 17408");
        assert!(cfg.attn_output_gate, "27B head uses attn_output_gate");

        let mut head =
            MtpHead::load_dense(&dir.join("model.safetensors"), &cfg, 4).expect("load_dense");
        assert!(head.dense_mlp.is_some());
        // Baked (unshifted) norm sanity: mean must be near the raw checkpoint's
        // (≈1) not ≈2 (a double +1 shift would land near +2).
        let fn_mean = head.final_norm.iter().sum::<f32>() / head.final_norm.len() as f32;
        assert!(fn_mean > 0.5 && fn_mean < 3.5, "norm.weight mean {fn_mean} — shift bug?");
        // Lock the mlx4 dequant on the REAL file: fc row0 vs the MLX
        // `mx.dequantize(fc, group_size=64, bits=4)` reference [0,0..4).
        let fc_ref = [0.00464f32, -0.01843, -0.00916, 0.02771];
        for (i, &r) in fc_ref.iter().enumerate() {
            assert!((head.fc[i] - r).abs() < 1e-3, "fc[{i}]={} != mlx ref {r}", head.fc[i]);
        }

        let h = cfg.hidden_size;
        let mut g = gen(1234);
        let embed_tab: Vec<Vec<f32>> = (0..64)
            .map(|r| {
                let mut gg = gen(9000 + r as u64);
                (0..h).map(|_| gg() * 0.05).collect()
            })
            .collect();
        let embed_of = |t: u32| embed_tab[(t as usize) % 64].clone();
        // Proxy lm_head (deterministic): argmax over a fixed random projection —
        // only needs to yield an in-range token to exercise the chain feedback.
        let mut pg = gen(777);
        let proj: Vec<f32> = (0..(64 * h)).map(|_| pg()).collect();
        let argmax_of = |hid: &[f32]| -> u32 {
            let mut best = 0u32;
            let mut bv = f32::NEG_INFINITY;
            for r in 0..64usize {
                let s: f32 = hid.iter().zip(&proj[r * h..(r + 1) * h]).map(|(&a, &b)| a * b).sum();
                if s > bv { bv = s; best = r as u32; }
            }
            best
        };
        let e0: Vec<f32> = (0..h).map(|_| g() * 0.05).collect();
        let hp0: Vec<f32> = (0..h).map(|_| g() * 0.05).collect();
        let (drafts, hiddens) = head.head_chain_cpu(&e0, &hp0, 0, 4, &embed_of, &argmax_of);
        assert_eq!(drafts.len(), 4);
        assert_eq!(head.kv_len(), 4, "chain advanced KV by 4");
        for (j, hid) in hiddens.iter().enumerate() {
            assert_eq!(hid.len(), h, "draft {j} hidden length");
            assert!(hid.iter().all(|v| v.is_finite()), "draft {j} hidden has non-finite");
            let mag: f32 = hid.iter().map(|v| v.abs()).sum();
            assert!(mag > 0.0, "draft {j} hidden is all-zero (degenerate)");
        }
        eprintln!("[dense-mtp] real-checkpoint drafts = {drafts:?} (all finite, KV={})", head.kv_len());
    }
}
