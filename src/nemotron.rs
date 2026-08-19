// SPDX-License-Identifier: Apache-2.0
//! Nemotron-H-Puzzle (`nemotron_h_puzzle`) hybrid architecture — the
//! `nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B` model. Each of the 88 layers
//! is *exactly one* mixer (single-mixer, NOT attn-subblock + MLP-subblock like
//! qwen35): a Mamba2/SSD recurrence, a NoPE GQA attention, or a latent-space
//! MoE, chosen per-layer by `config.block_configs[i].block_type`.
//!
//! This module is the first Rust increment (Mac-only, no cluster, no loader):
//!   * `NemotronConfig::from_json` — parse `config.json` `block_configs` into a
//!     `Vec<BlockSpec>` (with per-MoE-layer `num_experts_per_tok` /
//!     `moe_intermediate_size` overrides).
//!   * three CPU-reference mixer kernels transcribed faithfully from the
//!     validated numpy golden references (`nemotron_ref/reference_mamba2.py`,
//!     `nemotron_ref/reference_latent_moe.py`), fixture-validated in the tests
//!     at the bottom of this file (cos ≥ 0.9999 vs the golden `.npz` outputs).
//!   * a skeleton single-mixer `forward_pp_range` that dispatches by `BlockSpec`
//!     (compile-checked; cannot RUN end-to-end yet — the loader is a later
//!     increment, see the module-level notes on tensor naming / quant map).
//!
//! ## Resolved tensor naming (THIS NVFP4 checkpoint's `model.safetensors.index.json`)
//! `backbone.layers.{i}.mixer.*`, `backbone.layers.{i}.norm.weight`,
//! `backbone.embeddings.weight`, `backbone.norm_f.weight`, `lm_head.weight`,
//! and the MTP head under `mtp.*`. This is **`backbone.*`, not `model.*`** — the
//! numeric-reference agent saw `model.*` in the *BF16* repo, but the shipped
//! NVFP4 checkpoint (the deployment target) uses `backbone.*`. The attention
//! projections live under `.mixer.` too (`mixer.q_proj`/`k_proj`/`v_proj`/
//! `o_proj`), not `.self_attn.`.
#![allow(dead_code)]

use crate::model::{cpu_matmul, cpu_matvec_par, cpu_rms_norm, cpu_sdpa, KvCache, ModelWeights};
use crate::compute;
use crate::nemotron_1cb_enabled;
use crate::{prof_add, prof_add_ns, prof_on};
use crate::push_constants::{
    ew_mul_pc, ew_unary_pc, f32_slice_to_bytes, gemm_pc, matvec_fp8_pc, matvec_fp8_variant_k,
    matvec_mlx4_pc_off, matvec_f16_variant_k, matvec_nvfp4_variant_k, matvec_pc13,
    matvec_nvfp4_e4m3_variant, matvec_nvfp4_e4m3_pc, matvec_nvfp4_e4m3_pc_off,
    matvec_q8_0_variant_k, laguna_expert_repack_flag, nvfp4_repack_shape_ok,
    nem_gated_rmsnorm_pc, nem_moe_accum_pc, nem_ssd_scan_pc, nem_ssm_conv_pc, read_f32_buf,
    rmsnorm_pc,
};

// ─── WS3-style span-resident stage (Increment 4, VLLM_VULKAN_NEMOTRON_1CB) ────
//
// Slot indices for `NemotronModel::nem_res_bufs` — the persistent GPU buffers
// that hold the residual stream across the WHOLE resident layer span (and
// across tokens). Mirrors qwen3.6 WS3's `Q35R_*` (`qwen35_forward.rs`), but
// Nemotron only needs a small fixed set of persistent slots: `NR_H` is the
// only buffer that must survive across layers (single-mixer-per-layer, one
// residual add). Per-layer/per-type intermediates (q/k/v, mamba proj/scan,
// MoE latent/expert buffers) are allocated from the engine's transient pool
// exactly as increments 1-3 did — their sizes vary per call (Mamba's
// `in_proj_out`, and especially MoE's `top_k` which ranges 4-18 across
// layers) so giving them fixed persistent slots (as qwen does, with a
// hardcoded top-8) is not a drop-in fit here; the pool-alloc path is already
// the validated, bitwise-preserving machinery from Increments 1-3. The actual
// Increment-4 lever — hidden never touching the host between layers, RMSNorm
// + residual add moved onto the GPU, and the CB ring overlapping record with
// drain — is fully captured by this scheme.
const NR_H: usize = 0; // resident hidden / residual stream          [hs]
const NR_X: usize = 1; // rms_norm(hidden) output (mixer input)      [hs]
const NR_MIX: usize = 2; // per-layer mixer output, pre-residual     [hs]
// R2 (VLLM_VULKAN_NEMOTRON_GPU_SCAN): model-level scratch for the GPU Mamba2
// SSD scan, sized off `mamba_dims()` (NOT `hs`) — see `init_nem_res_bufs`.
const NR_CONV: usize = 3; // nemotron_ssm_conv_step output            [conv_dim]
const NR_GATED: usize = 4; // nemotron_ssd_scan output (pre-norm)     [intermediate]
// TP=2×PP (lever 2): the TP peer's mixer partial received by `nem_tp_reduce_mix`.
// The pairwise all-reduce finishes ON THE GPU as `NR_MIX += NR_PEER` folded into
// the next layer's leading CB, so the reduce itself is pure comm (no host add).
// Sized `[hs]`; only written/read when `tp_size > 1`.
const NR_PEER: usize = 5; // TP-reduce received peer partial          [hs]
const NR_COUNT: usize = 6;

// ─── GPU-resident quantized weight store (the 75B OOM fix) ────────────────────
//
// The CPU loader (`nemotron_loader::load_nemotron_weights`) dequantizes every
// weight to f32 host (~205GB / ~41GB per PP-5 stage) which OOMs the 14GB BC-250
// nodes at ANY pipeline depth. The resident path
// (`nemotron_loader::load_nemotron_resident`) instead keeps each matmul weight
// in its native quantized form uploaded GPU-resident — NVFP4 packed nibbles +
// folded f32 scales for the routed experts, raw FP8 bytes + scale for
// mamba/attn/shared projections, f16 for the BF16-native attn/latent
// projections — and the forward dequantizes IN THE SHADER via the existing
// `mul_mat_vec_nvfp4` / `mul_mat_vec_fp8` / `mul_mat_vec_*_f16` kernels. This is
// the direct analogue of the qwen3.6 `gpu_weights`/`QuantAux` resident path
// (`lib.rs` + `qwen35_forward::qwen35_matvec`); the two 4-bit/8-bit shaders and
// all the push-constant helpers are reused VERBATIM.

/// Per-weight quantization metadata for a GPU-resident Nemotron matmul weight.
/// Mirrors `crate::QuantAux` but lives on `NemotronModel` (the M6 precedent put
/// the Nemotron GPU machinery on the model, not on `VulkanModel`).
pub enum NemQuant {
    /// NVFP4: `NemGpuWeight.buffer` = packed u8 nibbles (2 E2M1/byte). Two
    /// scale-residency modes (mirrors the shared loader's `QuantAux::Nvfp4`):
    ///  * f32-fold (default, `e4m3 == false`): `scales` = folded f32
    ///    `e4m3(weight_scale)*weight_scale_2`, one per (row, group); `global`
    ///    unused (1.0). Reads `mul_mat_vec_nvfp4_f32_f32`.
    ///  * e4m3-resident (`VLLM_VULKAN_NVFP4_E4M3_SCALES`, `e4m3 == true`):
    ///    `scales` = the RAW on-disk `.weight_scale` e4m3 bytes (1 byte/group,
    ///    4x smaller), `global` = the per-tensor `.weight_scale_2` scalar
    ///    re-applied in-shader by `mul_mat_vec_nvfp4_e4m3_f32_f32`. Bit-exact
    ///    to the fold path (`nvfp4_e4m3_resident_matches_f32_fold`).
    Nvfp4 { scales: compute::Buffer, group_size: u32, e4m3: bool, global: f32 },
    /// FP8-E4M3: `NemGpuWeight.buffer` = raw FP8 bytes; `scale` = f32 (1 elem
    /// per-tensor, or `out_features` per-row → `per_row`).
    Fp8 { scale: compute::Buffer, per_row: bool },
    /// BF16-native weight uploaded as f16 (attn q/k/v/o, latent fc1/fc2, gate,
    /// lm_head): the plain `mul_mat_vec_*_f16` matvec.
    F16,
    /// Q8_0-requantized weight (mamba in_proj/out_proj, requanted at load from
    /// FP8 when `VLLM_VULKAN_NEMOTRON_MAMBA_Q8` is on): the scale lives inside
    /// each 34-byte block, so there is no side scale buffer — just the
    /// `mul_mat_vec_q8_0deq_*` matvec over the raw q8_0 bytes.
    Q8_0,
}

/// One GPU-resident Nemotron matmul weight (a single `nn.Linear`), keyed in
/// `NemotronModel::gpu_weights` by its full tensor name.
pub struct NemGpuWeight {
    pub buffer: compute::Buffer,
    pub quant: NemQuant,
    pub out_features: usize,
    pub in_features: usize,
}

/// One MoE layer's 512 routed experts, resident on the GPU as two concatenated
/// NVFP4 buffers (up + down), sliced per expert at dispatch via the nvfp4
/// shader's `packed_off`/`sb_off` (exactly the qwen `MoeGpuLayer` mlx4 pattern,
/// but NVFP4). Only the `top_k` selected experts are dispatched per token.
///
///  - `up`:   `[n_experts, moe_inter, latent/8]` packed words; `up_scales`
///            `[n_experts, moe_inter, latent/group_size]` folded f32.
///  - `down`: `[n_experts, latent, moe_inter/8]` packed words; `down_scales`
///            `[n_experts, latent, moe_inter/group_size]` folded f32.
/// R2 GPU Mamba2 SSD scan state for ONE resident layer — the GPU-resident
/// analogue of `Mamba2State` plus the per-layer constants the scan/conv/norm
/// shaders need, uploaded once by `attach_gpu_mamba_scan`. Unlike
/// `Mamba2State` (host `Vec<f32>`, re-read/re-written every decode step by
/// the CPU scan), `ssm_state`/`conv_state` stay GTT-resident and are only
/// ever touched by the GPU shaders once `VLLM_VULKAN_NEMOTRON_GPU_SCAN` is
/// on for this layer — see `nem_mamba_resident_layer`'s all-or-nothing gate.
pub struct NemMambaGpu {
    /// `[num_heads, head_dim, ssm_state_size]` row-major, GTT-resident.
    pub ssm_state: compute::Buffer,
    /// `[conv_dim, conv_kernel]` row-major, GTT-resident.
    pub conv_state: compute::Buffer,
    /// `[a_log(num_heads) | D(num_heads) | dt_bias(num_heads)]`, f32.
    pub params: compute::Buffer,
    /// `[conv_dim, conv_kernel]` depthwise conv weight, f32.
    pub conv_w: compute::Buffer,
    /// `[conv_dim]` conv bias, f32.
    pub conv_bias: compute::Buffer,
    /// `[intermediate]` gated-RMSNorm weight, f32.
    pub norm_w: compute::Buffer,
}

pub struct NemMoeExperts {
    pub n_experts: usize,
    pub up: compute::Buffer,
    pub up_scales: compute::Buffer,
    pub up_out: usize, // moe_intermediate_size
    pub up_in: usize,  // moe_latent_size
    pub down: compute::Buffer,
    pub down_scales: compute::Buffer,
    pub down_out: usize, // moe_latent_size
    pub down_in: usize,  // moe_intermediate_size
    pub group_size: u32,
    /// E4M3-RESIDENT scale mode (`VLLM_VULKAN_NVFP4_E4M3_SCALES`). When true,
    /// `up_scales`/`down_scales` hold the RAW on-disk `.weight_scale` e4m3
    /// bytes (1 byte/group, 4x smaller than the folded f32) and the per-expert
    /// `.weight_scale_2` global is applied in-shader via
    /// `mul_mat_vec_nvfp4_e4m3_f32_f32`. When false (default), the scale
    /// buffers hold folded f32 and `up_globals`/`down_globals` are unused.
    pub e4m3: bool,
    /// Per-LOCAL-expert `.weight_scale_2` (per-tensor NVFP4 global), one f32
    /// per owned expert, indexed by the same local expert id the dispatch
    /// `packed_off`/`sb_off` uses. Only consulted when `e4m3` is true; carried
    /// separately (not folded into the block scales) so the block scales can
    /// stay raw e4m3 bytes. `vec![1.0; ne_local]` in the f32-fold path.
    pub up_globals: Vec<f32>,
    pub down_globals: Vec<f32>,
}

/// Copy dispatch descriptor for one routed-expert NVFP4 matvec, gathered from
/// `NemMoeExperts` BEFORE `engine.as_mut()` so the immutable `gpu_experts`
/// borrow ends before the mutable `engine` borrow (same borrow dance as
/// `nem_meta`). Fields: `(w_ptr, s_ptr, k=in_features, n=out_features,
/// group_size, e4m3, globals_ptr)`. `globals_ptr` is a raw `*const Vec<f32>`
/// into the resident `up_globals`/`down_globals` (never mutated during a
/// forward — same invariant as the buffer ptrs); the per-expert
/// `.weight_scale_2` is read as `(*globals_ptr)[e]` only on the e4m3 path.
type ExpertMeta = (
    *const compute::Buffer,
    *const compute::Buffer,
    usize,
    usize,
    usize,
    bool,
    *const Vec<f32>,
);

/// Copy dispatch descriptor (raw pointers gathered BEFORE `engine.as_mut()` so
/// the immutable `gpu_weights` borrow ends before the mutable `engine` borrow —
/// the same borrow dance as `qwen35_forward::MvKind`).
#[derive(Clone, Copy)]
enum NemMvKind {
    F16,
    Q8_0,
    Nvfp4 { s: *const compute::Buffer, gs: u32, e4m3: bool, global: f32 },
    Fp8 { s: *const compute::Buffer, per_row: bool },
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
#[inline]
fn softplus(x: f32) -> f32 {
    // numerically stable log(1 + exp(x)), matching the numpy reference's
    // max(x,0) + log1p(exp(-|x|)) form.
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
#[inline]
fn relu2(x: f32) -> f32 {
    let r = x.max(0.0);
    r * r
}

// ─── Config / BlockSpec ──────────────────────────────────────────────────────

/// Per-layer mixer kind, parsed from `config.block_configs[i]`. MoE layers
/// carry per-layer overrides (`num_experts_per_tok`, `moe_intermediate_size`);
/// mamba/attention layers have no per-layer knobs (all their dims are global).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockSpec {
    Mamba,
    Attention,
    Moe {
        num_experts_per_tok: usize,
        moe_intermediate_size: usize,
    },
}

/// Nemotron-H-Puzzle configuration, parsed from `config.json`.
#[derive(Debug, Clone)]
pub struct NemotronConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub tie_word_embeddings: bool,

    // Mamba2 / SSD dims (global across all mamba layers)
    pub mamba_num_heads: usize,
    pub mamba_head_dim: usize,
    pub ssm_state_size: usize,
    pub n_groups: usize,
    pub conv_kernel: usize,
    pub use_conv_bias: bool,
    pub time_step_min: f32,

    // Attention dims (global across all attention layers) — NoPE GQA
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,

    // MoE dims (global; per-layer top_k / moe_inter live in BlockSpec::Moe)
    pub n_routed_experts: usize,
    pub moe_latent_size: usize,
    pub moe_shared_expert_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub n_group: usize,
    pub topk_group: usize,

    /// Per-layer mixer kind (length == num_hidden_layers).
    pub block_specs: Vec<BlockSpec>,
}

impl NemotronConfig {
    /// Mamba `intermediate_size` = d_inner = num_heads · head_dim (= expand·hidden).
    pub fn mamba_intermediate(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// Mamba2 dims bundle used by the decode-step kernel.
    pub fn mamba_dims(&self) -> Mamba2Dims {
        Mamba2Dims {
            hidden_size: self.hidden_size,
            num_heads: self.mamba_num_heads,
            head_dim: self.mamba_head_dim,
            ssm_state_size: self.ssm_state_size,
            n_groups: self.n_groups,
            conv_kernel: self.conv_kernel,
            time_step_min: self.time_step_min,
            eps: self.norm_eps,
        }
    }

    /// Latent-MoE dims for a given MoE layer (folds in the per-layer overrides).
    /// `top_k`/`moe_inter` come from that layer's `BlockSpec::Moe`.
    pub fn latent_moe_dims(&self, top_k: usize, moe_inter: usize) -> LatentMoeDims {
        LatentMoeDims {
            hidden_size: self.hidden_size,
            moe_latent_size: self.moe_latent_size,
            moe_intermediate_size: moe_inter,
            moe_shared_expert_intermediate_size: self.moe_shared_expert_intermediate_size,
            router: RouterDims {
                n_routed_experts: self.n_routed_experts,
                top_k,
                routed_scaling_factor: self.routed_scaling_factor,
                n_group: self.n_group,
                topk_group: self.topk_group,
                norm_topk_prob: self.norm_topk_prob,
            },
        }
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let u = |key: &str| v[key].as_u64().map(|x| x as usize);
        let req = |key: &str| u(key).ok_or_else(|| format!("config.json missing '{key}'"));
        let f = |key: &str, dflt: f64| v[key].as_f64().unwrap_or(dflt) as f32;

        let block_specs: Vec<BlockSpec> = v["block_configs"]
            .as_array()
            .ok_or("config.json missing 'block_configs'")?
            .iter()
            .map(|b| match b["block_type"].as_str() {
                Some("mamba") => Ok(BlockSpec::Mamba),
                Some("attention") => Ok(BlockSpec::Attention),
                Some("moe") => Ok(BlockSpec::Moe {
                    num_experts_per_tok: b["num_experts_per_tok"]
                        .as_u64()
                        .ok_or("moe block missing 'num_experts_per_tok'")?
                        as usize,
                    moe_intermediate_size: b["moe_intermediate_size"]
                        .as_u64()
                        .ok_or("moe block missing 'moe_intermediate_size'")?
                        as usize,
                }),
                other => Err(format!("unknown block_type {other:?}")),
            })
            .collect::<Result<_, _>>()?;

        // The real Nemotron-H-Puzzle checkpoint's `config.json` does NOT carry
        // a top-level `num_hidden_layers` key (verified against the actual
        // NVFP4 export — its top-level keys are `block_configs`,
        // `layers_block_type`, etc, with no `num_hidden_layers`); derive it
        // from `block_configs.len()` in that case. When the key IS present
        // (e.g. a synthetic test config), it must agree with `block_configs`.
        let num_hidden_layers = match u("num_hidden_layers") {
            Some(n) => {
                if block_specs.len() != n {
                    return Err(format!(
                        "block_configs len {} != num_hidden_layers {n}",
                        block_specs.len()
                    ));
                }
                n
            }
            None => block_specs.len(),
        };

        let num_attention_heads = req("num_attention_heads")?;
        let hidden_size = req("hidden_size")?;

        Ok(NemotronConfig {
            hidden_size,
            num_hidden_layers,
            vocab_size: req("vocab_size")?,
            // config carries both `norm_eps` and `layer_norm_epsilon` (both 1e-5);
            // the gated-RMSNorm uses `layer_norm_epsilon`, the per-layer input
            // norm uses `norm_eps` — identical here, prefer `norm_eps`.
            norm_eps: v["norm_eps"]
                .as_f64()
                .or_else(|| v["layer_norm_epsilon"].as_f64())
                .unwrap_or(1e-5) as f32,
            tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),

            mamba_num_heads: req("mamba_num_heads")?,
            mamba_head_dim: req("mamba_head_dim")?,
            ssm_state_size: req("ssm_state_size")?,
            n_groups: req("n_groups")?,
            conv_kernel: req("conv_kernel")?,
            use_conv_bias: v["use_conv_bias"].as_bool().unwrap_or(true),
            time_step_min: f("time_step_min", 0.001),

            num_attention_heads,
            num_key_value_heads: u("num_key_value_heads").unwrap_or(num_attention_heads),
            head_dim: u("head_dim").unwrap_or(hidden_size / num_attention_heads),
            attention_bias: v["attention_bias"].as_bool().unwrap_or(false),

            n_routed_experts: req("n_routed_experts")?,
            moe_latent_size: req("moe_latent_size")?,
            moe_shared_expert_intermediate_size: req("moe_shared_expert_intermediate_size")?,
            routed_scaling_factor: f("routed_scaling_factor", 1.0),
            norm_topk_prob: v["norm_topk_prob"].as_bool().unwrap_or(true),
            n_group: u("n_group").unwrap_or(1),
            topk_group: u("topk_group").unwrap_or(1),

            block_specs,
        })
    }
}

// ─── Mamba2 / SSD decode-step kernel ─────────────────────────────────────────

/// Dims for one Mamba2/SSD decode step. Mirrors `reference_mamba2.Mamba2Config`.
#[derive(Debug, Clone, Copy)]
pub struct Mamba2Dims {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ssm_state_size: usize,
    pub n_groups: usize,
    pub conv_kernel: usize,
    pub time_step_min: f32,
    pub eps: f32,
}

impl Mamba2Dims {
    /// d_inner = num_heads · head_dim.
    pub fn intermediate(&self) -> usize {
        self.num_heads * self.head_dim
    }
    /// conv1d channel count = intermediate + 2·n_groups·ssm_state (x, B, C).
    pub fn conv_dim(&self) -> usize {
        self.intermediate() + 2 * self.n_groups * self.ssm_state_size
    }
    /// in_proj output width = intermediate(gate) + conv_dim(x|B|C) + num_heads(dt).
    /// (`d_mlp` is always 0 for this model — no merged extra-MLP channels.)
    pub fn in_proj_out(&self) -> usize {
        self.intermediate() + self.conv_dim() + self.num_heads
    }
    /// Gated-RMSNorm group size = intermediate / n_groups.
    pub fn norm_group_size(&self) -> usize {
        self.intermediate() / self.n_groups
    }
}

/// Borrowed Mamba2 weights (all standard `nn.Linear` `[out, in]` orientation).
pub struct Mamba2Weights<'a> {
    /// [in_proj_out, hidden_size]
    pub in_proj: &'a [f32],
    /// [conv_dim, conv_kernel] (depthwise, per-channel kernel)
    pub conv1d_weight: &'a [f32],
    /// [conv_dim] or None
    pub conv1d_bias: Option<&'a [f32]>,
    /// [num_heads]
    pub a_log: &'a [f32],
    /// [num_heads]
    pub d: &'a [f32],
    /// [num_heads]
    pub dt_bias: &'a [f32],
    /// [intermediate] (gated RMSNorm weight)
    pub norm_weight: &'a [f32],
    /// [hidden_size, intermediate]
    pub out_proj: &'a [f32],
}

/// Owned Mamba2 weight buffers for one layer, materialized from `ModelWeights`.
/// `.borrow()` builds the borrowed `Mamba2Weights<'_>` the kernels take.
struct Mamba2WeightBufs {
    in_proj: Vec<f32>,
    conv_w: Vec<f32>,
    conv_b: Option<Vec<f32>>,
    a_log: Vec<f32>,
    d_skip: Vec<f32>,
    dt_bias: Vec<f32>,
    norm_w: Vec<f32>,
    out_proj: Vec<f32>,
}

impl Mamba2WeightBufs {
    fn borrow(&self) -> Mamba2Weights<'_> {
        Mamba2Weights {
            in_proj: &self.in_proj,
            conv1d_weight: &self.conv_w,
            conv1d_bias: self.conv_b.as_deref(),
            a_log: &self.a_log,
            d: &self.d_skip,
            dt_bias: &self.dt_bias,
            norm_weight: &self.norm_w,
            out_proj: &self.out_proj,
        }
    }
}

/// Recurrent decode state for one Mamba2 layer (batch folded away, B = 1).
#[derive(Clone)]
pub struct Mamba2State {
    /// Causal conv window `[conv_dim, conv_kernel]` row-major; most-recent token
    /// in the LAST column of each channel.
    pub conv_state: Vec<f32>,
    /// SSM state `[num_heads, head_dim, ssm_state_size]` row-major.
    pub ssm_state: Vec<f32>,
}

impl Mamba2State {
    pub fn zeros(d: &Mamba2Dims) -> Self {
        Mamba2State {
            conv_state: vec![0.0; d.conv_dim() * d.conv_kernel],
            ssm_state: vec![0.0; d.num_heads * d.head_dim * d.ssm_state_size],
        }
    }
    pub fn reset(&mut self) {
        self.conv_state.iter_mut().for_each(|x| *x = 0.0);
        self.ssm_state.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// Causal depthwise conv1d + SiLU for ONE token, updating the sliding window
/// in place: roll each channel's window left by one, drop the new token into
/// the last column, then dot the (now causal) window against the per-channel
/// kernel. Shared verbatim between the decode step and the prefill scan (the
/// prefill scan calls this once per token, in order, threading the same
/// `conv_state` — bit-identical to calling it via `mamba2_decode_step` T times).
#[inline]
fn conv1d_silu_step(raw_bc: &[f32], state: &mut Mamba2State, w: &Mamba2Weights, d: &Mamba2Dims) -> Vec<f32> {
    use rayon::prelude::*;

    let conv_dim = d.conv_dim();
    let k = d.conv_kernel;
    let mut conv_out = vec![0.0f32; conv_dim];
    conv_out
        .par_iter_mut()
        .zip(state.conv_state.par_chunks_mut(k))
        .enumerate()
        .for_each(|(c, (out_c, win))| {
            // roll left by one, append raw_bc[c] in the last slot
            for t in 0..k - 1 {
                win[t] = win[t + 1];
            }
            win[k - 1] = raw_bc[c];
            let base = c * k;
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += win[t] * w.conv1d_weight[base + t];
            }
            if let Some(b) = w.conv1d_bias {
                acc += b[c];
            }
            *out_c = silu(acc);
        });
    conv_out
}

/// The per-token SSD recurrence body + gated-RMSNorm, shared verbatim between
/// the decode step and the prefill scan: given this token's post-conv
/// `hidden_x`/`b_flat`/`c_flat`/`dt_raw`/`gate` (already split out of a
/// `[in_proj_out]` or `[conv_dim]` row by the caller), advance `state.ssm_state`
/// in place and return the post-gated-norm `scan` output `[intermediate]`
/// (pre-`out_proj`, which the caller batches separately for the prefill path).
#[inline]
fn mamba2_recurrence_and_norm(
    gate: &[f32],
    hidden_x: &[f32],
    b_flat: &[f32],
    c_flat: &[f32],
    dt_raw: &[f32],
    w: &Mamba2Weights,
    state: &mut Mamba2State,
    d: &Mamba2Dims,
) -> Vec<f32> {
    use rayon::prelude::*;

    let inter = d.intermediate();
    let nh = d.num_heads;
    let hd = d.head_dim;
    let ss = d.ssm_state_size;
    let ng = d.n_groups;
    let heads_per_group = nh / ng;

    // y accumulator [num_heads, head_dim]
    let mut y = vec![0.0f32; inter];
    y.par_chunks_mut(hd)
        .zip(state.ssm_state.par_chunks_mut(hd * ss))
        .enumerate()
        .for_each(|(h, (y_h, ssm_h))| {
            let a = -(w.a_log[h].exp()); // A = -exp(A_log), scalar per head
            let mut dt_h = softplus(dt_raw[h] + w.dt_bias[h]);
            if dt_h < d.time_step_min {
                dt_h = d.time_step_min; // time_step_max clamp is disabled upstream too
            }
            let da = (dt_h * a).exp(); // scalar decay, broadcast over head_dim × ssm
            let g = h / heads_per_group; // group-shared B/C
            let bg = &b_flat[g * ss..(g + 1) * ss];
            let cg = &c_flat[g * ss..(g + 1) * ss];
            let hx = &hidden_x[h * hd..(h + 1) * hd];
            for di in 0..hd {
                let x_hd = hx[di];
                let sd = di * ss; // local offset within this head's ssm chunk
                let mut acc = 0.0f32;
                for n in 0..ss {
                    // new_ssm = ssm·dA + dt·B·x ; y = Σ_n new_ssm · C
                    let db_x = dt_h * bg[n] * x_hd;
                    let s = ssm_h[sd + n] * da + db_x;
                    ssm_h[sd + n] = s;
                    acc += s * cg[n];
                }
                // D skip-connection
                y_h[di] = acc + x_hd * w.d[h];
            }
        });

    // Gated RMSNorm (norm_before_gate = false): gate first, then per-group RMS.
    let gs = d.norm_group_size();
    let mut scan = vec![0.0f32; inter];
    scan.par_chunks_mut(gs).enumerate().for_each(|(gi, scan_g)| {
        let y_g = &y[gi * gs..(gi + 1) * gs];
        let gate_g = &gate[gi * gs..(gi + 1) * gs];
        let nw_g = &w.norm_weight[gi * gs..(gi + 1) * gs];
        let mut ss_sum = 0.0f32;
        for j in 0..gs {
            let gated = y_g[j] * silu(gate_g[j]);
            scan_g[j] = gated;
            ss_sum += gated * gated;
        }
        let rstd = 1.0 / (ss_sum / gs as f32 + d.eps).sqrt();
        for j in 0..gs {
            scan_g[j] = scan_g[j] * rstd * nw_g[j];
        }
    });
    scan
}

/// One Mamba2/SSD decode step for a single new token. Faithful port of
/// `reference_mamba2.mamba2_decode_step` (the `has_previous_state` decode branch
/// of `NemotronHMamba2Mixer.torch_forward`).
///
/// `x`: `[hidden_size]` — the pre-normed hidden state (the mixer input). Returns
/// `[hidden_size]`; `state` is advanced in place (conv window + SSM state).
pub fn mamba2_decode_step(
    x: &[f32],
    w: &Mamba2Weights,
    state: &mut Mamba2State,
    d: &Mamba2Dims,
) -> Vec<f32> {
    let inter = d.intermediate();
    let conv_dim = d.conv_dim();
    let ng = d.n_groups;
    let ss = d.ssm_state_size;

    // in_proj: [in_proj_out] = x @ in_proj^T.
    let proj = cpu_matmul(x, w.in_proj, 1, d.hidden_size, d.in_proj_out());
    let gate = &proj[..inter];
    let raw_bc = &proj[inter..inter + conv_dim];
    let dt = &proj[inter + conv_dim..]; // [num_heads]

    let conv_out = conv1d_silu_step(raw_bc, state, w, d);
    let hidden_x = &conv_out[..inter]; // [intermediate]
    let b_flat = &conv_out[inter..inter + ng * ss]; // [n_groups, ssm]
    let c_flat = &conv_out[inter + ng * ss..]; // [n_groups, ssm]

    let scan = mamba2_recurrence_and_norm(gate, hidden_x, b_flat, c_flat, dt, w, state, d);

    // out_proj.
    cpu_matmul(&scan, w.out_proj, 1, inter, d.hidden_size)
}

/// Sequential multi-token Mamba2/SSD prefill scan: the FLOP-dominant
/// projections (`in_proj`, `out_proj`) are batched as GEMMs over all `T`
/// tokens; the (cheap, sequentially-dependent) conv1d and SSD recurrence run
/// token-by-token, reusing the EXACT SAME per-token code
/// (`conv1d_silu_step` / `mamba2_recurrence_and_norm`) as `mamba2_decode_step`.
/// This makes the result **bit-exact by construction** vs calling
/// `mamba2_decode_step` `T` times on the same input from the same starting
/// `state` — see `mamba2_prefill_matches_t_times_decode` below.
///
/// `xs`: `[T, hidden_size]` flat (pre-normed mixer input for T tokens, in
/// order). Returns `[T, hidden_size]` flat; `state` is advanced in place
/// exactly as it would be after T decode steps.
pub fn mamba2_prefill_seq(xs: &[f32], w: &Mamba2Weights, state: &mut Mamba2State, d: &Mamba2Dims) -> Vec<f32> {
    let hs = d.hidden_size;
    let t = xs.len() / hs;
    debug_assert_eq!(t * hs, xs.len(), "mamba2_prefill_seq: xs.len() not a multiple of hidden_size");

    let inter = d.intermediate();
    let in_proj_out = d.in_proj_out();

    // Batched in_proj GEMM over all T rows (the FLOP-dominant pre-pass).
    let proj = cpu_matmul(xs, w.in_proj, t, hs, in_proj_out); // [T, in_proj_out]
    // Sequential per-token conv1d + SSD recurrence (unavoidable dependency chain).
    let scans = mamba2_scan_only(&proj, w, state, d); // [T, inter]
    // Batched out_proj GEMM over all T rows.
    cpu_matmul(&scans, w.out_proj, t, inter, hs)
}

/// The MIDDLE of the Mamba2 prefill: the sequential conv1d + SSD recurrence that
/// maps the `in_proj` output `proj` `[T, in_proj_out]` to the pre-`out_proj`
/// scan output `scans` `[T, intermediate]`, advancing `state` exactly as `T`
/// decode steps would. Split out of `mamba2_prefill_seq` so the GPU prefill path
/// can drive the two FLOP-dominant projections (`in_proj`/`out_proj`) through the
/// batched GEMM while reusing this verbatim per-token recurrence — the result is
/// bit-identical to `mamba2_prefill_seq` when both projections use `cpu_matmul`.
pub fn mamba2_scan_only(proj: &[f32], w: &Mamba2Weights, state: &mut Mamba2State, d: &Mamba2Dims) -> Vec<f32> {
    let inter = d.intermediate();
    let conv_dim = d.conv_dim();
    let ng = d.n_groups;
    let ss = d.ssm_state_size;
    let in_proj_out = d.in_proj_out();
    let t = proj.len() / in_proj_out;
    debug_assert_eq!(t * in_proj_out, proj.len(), "mamba2_scan_only: proj.len() not a multiple of in_proj_out");

    let mut scans = vec![0.0f32; t * inter];
    for ti in 0..t {
        let row = &proj[ti * in_proj_out..(ti + 1) * in_proj_out];
        let gate = &row[..inter];
        let raw_bc = &row[inter..inter + conv_dim];
        let dt = &row[inter + conv_dim..];

        let conv_out = conv1d_silu_step(raw_bc, state, w, d);
        let hidden_x = &conv_out[..inter];
        let b_flat = &conv_out[inter..inter + ng * ss];
        let c_flat = &conv_out[inter + ng * ss..];

        let scan = mamba2_recurrence_and_norm(gate, hidden_x, b_flat, c_flat, dt, w, state, d);
        scans[ti * inter..(ti + 1) * inter].copy_from_slice(&scan);
    }
    scans
}

// ─── Latent-MoE kernel (router + relu² experts + shared expert) ──────────────

/// DeepSeek-style sigmoid router dims.
#[derive(Debug, Clone, Copy)]
pub struct RouterDims {
    pub n_routed_experts: usize,
    pub top_k: usize,
    pub routed_scaling_factor: f32,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
}

/// Latent-MoE dims bundle.
#[derive(Debug, Clone, Copy)]
pub struct LatentMoeDims {
    pub hidden_size: usize,
    pub moe_latent_size: usize,
    pub moe_intermediate_size: usize,
    pub moe_shared_expert_intermediate_size: usize,
    pub router: RouterDims,
}

/// Borrowed latent-MoE weights (all standard `nn.Linear` `[out, in]`).
pub struct LatentMoeWeights<'a> {
    /// [n_routed_experts, hidden_size]
    pub gate_weight: &'a [f32],
    /// [n_routed_experts]
    pub e_score_correction_bias: &'a [f32],
    /// [moe_latent_size, hidden_size]
    pub fc1_latent_proj: &'a [f32],
    /// [hidden_size, moe_latent_size]
    pub fc2_latent_proj: &'a [f32],
    /// per-expert up: [moe_intermediate_size, moe_latent_size] (row-major, N experts)
    pub expert_up: &'a [f32],
    /// per-expert down: [moe_latent_size, moe_intermediate_size] (row-major, N experts)
    pub expert_down: &'a [f32],
    /// shared up: [moe_shared_expert_intermediate_size, hidden_size] (FULL hidden!)
    pub shared_up: &'a [f32],
    /// shared down: [hidden_size, moe_shared_expert_intermediate_size]
    pub shared_down: &'a [f32],
}

/// relu² 2-matrix expert: `down(relu(up(x))²)`. `up: [inter, in]`, `down: [in, inter]`.
///
/// `pub(crate)` (not private): reused verbatim by `nemotron_tp::latent_moe_routed_partial`
/// (Increment 3 EP=2 shard) so the per-expert math is byte-identical between the
/// monolithic and sharded paths — no reimplementation, no divergence risk.
pub(crate) fn mlp_relu2(x: &[f32], up: &[f32], down: &[f32], in_dim: usize, inter: usize) -> Vec<f32> {
    let h = cpu_matmul(x, up, 1, in_dim, inter);
    let act: Vec<f32> = h.iter().map(|&v| relu2(v)).collect();
    cpu_matmul(&act, down, 1, inter, in_dim)
}

/// DeepSeek-style sigmoid router with `e_score_correction_bias` + group top-k,
/// for a single token. Faithful port of `reference_latent_moe.router_forward`.
///
/// Returns `(indices [top_k], weights [top_k])`. Weights are gathered from the
/// RAW sigmoid `scores` (NOT the bias-corrected scores), optionally renormed,
/// then scaled by `routed_scaling_factor`.
pub fn router_forward(
    hidden: &[f32],
    gate_weight: &[f32],
    e_score_correction_bias: &[f32],
    cfg: &RouterDims,
) -> (Vec<usize>, Vec<f32>) {
    let ne = cfg.n_routed_experts;
    let t = std::time::Instant::now();
    // Router gate matvec: [1,hidden]·[ne,hidden]. rayon-parallel over the ne output
    // rows (each row an independent dot product). NOT bit-identical to cpu_matmul on
    // every arch (aarch64 fused-fmla ULP, Gate B), but validated argmax-exact/cos-
    // 0.99999 on the cluster (x86_64) — the same bar the qwen router uses (moe.rs).
    // The top-k expert *selection* is what matters, and it is unaffected.
    let logits = cpu_matvec_par(hidden, gate_weight, hidden.len(), ne); // [ne]
    crate::prof_add("nem_router_matmul", t);
    let scores: Vec<f32> = logits.iter().map(|&z| sigmoid(z)).collect();
    let choice: Vec<f32> = scores
        .iter()
        .zip(e_score_correction_bias)
        .map(|(&s, &b)| s + b)
        .collect();

    // Group top-k: split the `ne` experts into `n_group` contiguous groups,
    // score each group by the sum of its top-2 corrected scores, keep the top
    // `topk_group` groups, and mask out (zero) experts in unselected groups.
    let t_sort = std::time::Instant::now();
    let per_group = ne / cfg.n_group;
    let mut group_score = vec![0.0f32; cfg.n_group];
    for g in 0..cfg.n_group {
        let seg = &choice[g * per_group..(g + 1) * per_group];
        let mut top2 = [f32::NEG_INFINITY, f32::NEG_INFINITY];
        for &s in seg {
            if s > top2[0] {
                top2[1] = top2[0];
                top2[0] = s;
            } else if s > top2[1] {
                top2[1] = s;
            }
        }
        // per_group >= 2 in the real model; guard tiny fixtures anyway.
        group_score[g] = top2[0] + if top2[1].is_finite() { top2[1] } else { 0.0 };
    }
    // top `topk_group` groups.
    let mut group_order: Vec<usize> = (0..cfg.n_group).collect();
    group_order.sort_by(|&a, &b| {
        group_score[b]
            .partial_cmp(&group_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut group_selected = vec![false; cfg.n_group];
    for &g in group_order.iter().take(cfg.topk_group) {
        group_selected[g] = true;
    }

    // top_k over the masked corrected scores.
    let mut order: Vec<usize> = (0..ne).collect();
    let masked = |e: usize| -> f32 {
        if group_selected[e / per_group] {
            choice[e]
        } else {
            0.0
        }
    };
    order.sort_by(|&a, &b| {
        masked(b)
            .partial_cmp(&masked(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let indices: Vec<usize> = order.into_iter().take(cfg.top_k).collect();
    prof_add("nem_router_sort", t_sort);

    // weights gathered from RAW scores, optional renorm, then scaling.
    let mut weights: Vec<f32> = indices.iter().map(|&e| scores[e]).collect();
    if cfg.norm_topk_prob {
        let denom = weights.iter().sum::<f32>() + 1e-20;
        for wv in weights.iter_mut() {
            *wv /= denom;
        }
    }
    for wv in weights.iter_mut() {
        *wv *= cfg.routed_scaling_factor;
    }
    (indices, weights)
}

/// Full latent-MoE mixer forward for a batch of tokens. Faithful port of
/// `reference_latent_moe.latent_moe_forward`.
///
/// `hidden`: `[n_tokens, hidden_size]` (pre-normed mixer input). Returns
/// `[n_tokens, hidden_size]`. The routed path runs in the `moe_latent_size`
/// bottleneck (`fc1` → experts → `fc2`); the shared expert runs in parallel on
/// the FULL `hidden_size` directly on the token's original mixer input (NOT the
/// latent bottleneck — the confirmed correction to the plan).
pub fn latent_moe_forward(
    hidden: &[f32],
    n_tokens: usize,
    w: &LatentMoeWeights,
    d: &LatentMoeDims,
) -> Vec<f32> {
    let hs = d.hidden_size;
    let lat = d.moe_latent_size;
    let inter = d.moe_intermediate_size;
    let shared_inter = d.moe_shared_expert_intermediate_size;
    let expert_up_stride = inter * lat;
    let expert_down_stride = lat * inter;

    let mut out = vec![0.0f32; n_tokens * hs];
    for t in 0..n_tokens {
        let htok = &hidden[t * hs..(t + 1) * hs];

        let (indices, weights) =
            router_forward(htok, w.gate_weight, w.e_score_correction_bias, &d.router);

        // routed path: down-project to latent, run only the selected experts.
        let latent = cpu_matmul(htok, w.fc1_latent_proj, 1, hs, lat); // [lat]
        let mut routed = vec![0.0f32; lat];
        for (k, &e) in indices.iter().enumerate() {
            let up = &w.expert_up[e * expert_up_stride..(e + 1) * expert_up_stride];
            let down = &w.expert_down[e * expert_down_stride..(e + 1) * expert_down_stride];
            let eout = mlp_relu2(&latent, up, down, lat, inter); // [lat]
            let wk = weights[k];
            for (r, &o) in routed.iter_mut().zip(&eout) {
                *r += o * wk;
            }
        }
        let moe_out = cpu_matmul(&routed, w.fc2_latent_proj, 1, lat, hs); // [hs]

        // shared expert: FULL hidden, on the original mixer input.
        let shared = mlp_relu2(htok, w.shared_up, w.shared_down, hs, shared_inter); // [hs]

        for i in 0..hs {
            out[t * hs + i] = moe_out[i] + shared[i];
        }
    }
    out
}

// ─── Per-layer state + model skeleton ────────────────────────────────────────

/// Per-layer mutable decode state — one of the 3 mixer kinds. Generalizes
/// qwen35's 2-variant `LayerState` to 3 (single-mixer per layer).
pub enum LayerState {
    Mamba(Mamba2State),
    Attention(KvCache),
    /// MoE layers are stateless at decode time.
    Moe,
}

/// Nemotron-H-Puzzle model (CPU reference skeleton). Pipeline-parallel aware in
/// the same shape as `Qwen35Model`: `config.block_specs` is always the GLOBAL
/// list, but only layers `[pp_start, pp_end)` are resident (one `layer_state`
/// entry each), indexed by `state_idx(global) = global - pp_start`.
///
/// NOTE: this is a *skeleton* — the 3-way dispatch + the three fixture-validated
/// kernels compile and run, but a full end-to-end `forward` cannot run until the
/// (deferred) NVFP4/FP8 loader materializes the `backbone.*` weight tensors.
pub struct NemotronModel {
    pub config: NemotronConfig,
    pub weights: ModelWeights,
    pub layer_state: Vec<LayerState>,
    pub lm_head_name: String,
    pub pp_start: usize,
    pub pp_end: usize,
    /// Optional Vulkan compute engine for the GPU Mamba-projection path
    /// (`VLLM_VULKAN_NEMOTRON_GPU_MAMBA`). `None` = CPU-only (the default, and
    /// the only mode on a device-less host). Populated at construction in
    /// `lib.rs` after the FP8 in_proj/out_proj weights are uploaded f16.
    pub engine: Option<compute::ComputeEngine>,
    /// GPU-resident f16 Mamba `in_proj`/`out_proj` weights, keyed by full
    /// tensor name (`backbone.layers.{i}.mixer.{in,out}_proj.weight`). Empty
    /// unless the GPU-mamba path is enabled AND the upload succeeded; a missing
    /// key makes `mamba_proj` fall back to `cpu_matmul` for that projection.
    pub gpu_proj: std::collections::HashMap<String, compute::Buffer>,
    /// GPU-resident quantized matmul weights (the 75B OOM fix): attn q/k/v/o,
    /// mamba in/out_proj, moe fc1/fc2/gate/shared, lm_head — keyed by full
    /// tensor name. Empty unless the resident loader ran; a missing key makes
    /// [`NemotronModel::nem_matvec`] fall back to the f32 host `cpu_matmul`.
    pub gpu_weights: std::collections::HashMap<String, NemGpuWeight>,
    /// GPU-resident NVFP4 routed experts, per GLOBAL MoE layer index. Empty
    /// unless the resident loader ran; a missing key makes the MoE mixer fall
    /// back to the concatenated-host CPU path (`latent_moe_forward`).
    pub gpu_experts: std::collections::HashMap<usize, NemMoeExperts>,
    /// Persistent GPU buffers for the WS3-style span-resident stage
    /// (`nem_forward_range_resident`, Increment 4), indexed by `NR_*`. Empty
    /// until `init_nem_res_bufs` succeeds.
    pub nem_res_bufs: Vec<compute::Buffer>,
    /// Whether `nem_res_bufs` has been allocated (idempotency guard for
    /// `init_nem_res_bufs`).
    pub nem_res_ready: bool,
    /// Cached one-time readiness verdict for the span-resident stage
    /// (`ensure_nem_res`). A `Some(false)` is permanent for the process.
    pub nem_res_ok: Option<bool>,
    /// Pre-uploaded per-layer + final RMSNorm weights for the resident stage,
    /// keyed by full tensor name. Populated ONCE by `ensure_nem_norm_weights`
    /// (no map inserts during a forward — raw `*const Buffer` pointers
    /// gathered for a CB must not be invalidated by a map rehash).
    pub gpu_norm_w: std::collections::HashMap<String, compute::Buffer>,
    /// R2 (`VLLM_VULKAN_NEMOTRON_GPU_SCAN`): per-layer GPU-resident Mamba2
    /// SSD scan state + constants, keyed by GLOBAL layer index. Populated
    /// ONCE by `attach_gpu_mamba_scan` for every resident Mamba layer; empty
    /// unless the flag is set AND the upload succeeded for every such layer
    /// (see that function's all-or-nothing doc comment).
    pub gpu_mamba: std::collections::HashMap<usize, NemMambaGpu>,
    /// Phase-1 MTP acceptance sim (`VLLM_VULKAN_NEMOTRON_MTP`): the LAST
    /// stage's pre-`backbone.norm_f` residual hidden from the most recent
    /// `forward_pp_stage` call — the DeepSeek-NextN `h_pre` the MTP draft
    /// head consumes (see `nemotron_mtp::NemMtpHead`). Populated
    /// unconditionally (cheap: `hidden_size` floats) so it's always fresh
    /// when the caller's next `nem_mtp_draft` runs; meaningless on a
    /// non-last stage (this model instance never computes a final hidden
    /// there) but harmless to fill anyway.
    pub last_pre_norm_hidden: Vec<f32>,
    /// Phase-1 DECOUPLED trace-dump (`VLLM_VULKAN_NEMOTRON_MTP_TRACE=<path>`):
    /// an open handle written to on every LAST-stage `forward_pp_stage` call
    /// (one record per decode step — see `attach_mtp_trace`'s doc for the
    /// format). `None` unless the flag was set AND `attach_mtp_trace`
    /// succeeded. Lets a normal PP-5 run of the SHIPPED stack (no MTP head,
    /// so it fits — the head does not need to be co-resident with the base
    /// model at all) record exactly what the head would need, for an
    /// offline single-node replay (`nemotron_mtp::NemMtpHead` + a trace file,
    /// no base model, no PP, no OOM risk).
    pub mtp_trace: Option<std::fs::File>,

    // ── TP=2×PP hybrid (Megatron tensor-parallel within a PP stage-group) ──
    /// This rank's tensor-parallel index within its PP stage-group, and the TP
    /// world size. `tp_size==1` (the default) => no TP: full tensors, no
    /// all-reduce, byte-identical to the shipped PP-only resident path.
    pub tp_rank: usize,
    pub tp_size: usize,
    /// GLOBAL rank of this rank's TP peer (the other rank in the same PP stage)
    /// on the flat vCCL communicator, or `-1` when `tp_size==1`. A TP=2
    /// all-reduce over the pair is ONE pairwise `send_f32`/`recv_f32` exchange
    /// with this peer (deadlock-safe: even `tp_rank` sends first) — the exact
    /// primitive the PP hop already uses, so TP=2 needs NO new collective (see
    /// [[nemotron-tp-pp-scaling]]).
    pub tp_peer: i32,
    /// Raw `vcclComm_t` (as `usize`) for the TP all-reduce — the SAME flat comm
    /// the PP hop uses (`set_collective_comm`). `0` when not wired.
    pub collective_comm: usize,
    /// Lever 1: persistent host f32 exchange scratch for `nem_tp_reduce_mix`,
    /// sized to `hidden_size`, allocated ONCE (lazily on the first reduce) and
    /// reused across all 40 MoE-layer reduces/token and across tokens. The
    /// send buffer holds the local mixer partial read back from `NR_MIX`; the
    /// recv buffer receives the peer's partial in place (no per-call `Vec`
    /// alloc). Both are `comm_register`'d ONCE (handles below) so each reduce
    /// skips the per-call `ibv_reg_mr`/dereg the vCCL FFI flags as the dominant
    /// cost for small reduces. Empty until `ensure_tp_scratch`.
    tp_send_scratch: Vec<f32>,
    tp_recv_scratch: Vec<f32>,
    /// `comm_register` handles for the two scratch buffers (`0` = unregistered:
    /// registration unavailable in this libvccl or it failed — the reduce still
    /// uses the persistent buffers, just pays the per-call reg_mr).
    tp_send_reg: usize,
    tp_recv_reg: usize,
    /// Guards the one-time scratch alloc + registration in `ensure_tp_scratch`.
    tp_reg_ready: bool,
    /// PP forward-hop `[H]` scratch for the FUSED native-vCCL hop
    /// (`VulkanModel::pp_step_nemotron`), mirroring kimi's `pp_{recv,send}_
    /// scratch`: recv the previous stage's hidden INTO `pp_recv_scratch` and
    /// send this stage's hidden FROM `pp_send_scratch`, both `comm_register`'d
    /// ONCE so vCCL skips the per-call temp-MR + the "buffer (H*4 B) not
    /// registered with the comm" warning the fresh per-token `Vec` hit. Empty/`0`
    /// until `pp_step_nemotron` pins them (gated by `VLLM_VULKAN_REG_REDUCE`); on
    /// unavailable/failed registration the fused hop falls back to the fresh-`Vec`
    /// `recv_f32`/`send_f32` path (correct, per-call regMr).
    pub pp_recv_scratch: Vec<f32>,
    pub pp_recv_handle: usize,
    pub pp_send_scratch: Vec<f32>,
    pub pp_send_handle: usize,
}

impl NemotronModel {
    pub fn new(
        config: NemotronConfig,
        weights: ModelWeights,
        max_seq_len: usize,
        lm_head_name: String,
    ) -> Self {
        let end = config.num_hidden_layers;
        Self::new_range(config, weights, max_seq_len, lm_head_name, 0, end)
    }

    pub fn new_range(
        config: NemotronConfig,
        weights: ModelWeights,
        max_seq_len: usize,
        lm_head_name: String,
        pp_start: usize,
        pp_end: usize,
    ) -> Self {
        let mamba_dims = config.mamba_dims();
        let layer_state = (pp_start..pp_end)
            .map(|g| match config.block_specs[g] {
                BlockSpec::Mamba => LayerState::Mamba(Mamba2State::zeros(&mamba_dims)),
                BlockSpec::Attention => LayerState::Attention(KvCache::new(
                    max_seq_len,
                    config.num_key_value_heads,
                    config.head_dim,
                )),
                BlockSpec::Moe { .. } => LayerState::Moe,
            })
            .collect();
        NemotronModel {
            config,
            weights,
            layer_state,
            lm_head_name,
            pp_start,
            pp_end,
            engine: None,
            gpu_proj: std::collections::HashMap::new(),
            gpu_weights: std::collections::HashMap::new(),
            gpu_experts: std::collections::HashMap::new(),
            nem_res_bufs: Vec::new(),
            nem_res_ready: false,
            nem_res_ok: None,
            gpu_norm_w: std::collections::HashMap::new(),
            gpu_mamba: std::collections::HashMap::new(),
            last_pre_norm_hidden: Vec::new(),
            mtp_trace: None,
            tp_rank: 0,
            tp_size: 1,
            tp_peer: -1,
            collective_comm: 0,
            tp_send_scratch: Vec::new(),
            tp_recv_scratch: Vec::new(),
            tp_send_reg: 0,
            tp_recv_reg: 0,
            tp_reg_ready: false,
            pp_recv_scratch: Vec::new(),
            pp_recv_handle: 0,
            pp_send_scratch: Vec::new(),
            pp_send_handle: 0,
        }
    }

    /// Set the Megatron TP grid position for this stage's rank (called from the
    /// model constructor once `VLLM_VULKAN_TP_RANK`/`_SIZE` are read). Shards are
    /// applied at LOAD time (`load_nemotron_resident` with the same rank/size),
    /// so this only records the forward-side reduce parameters.
    pub fn set_tp(&mut self, tp_rank: usize, tp_size: usize) {
        self.tp_size = tp_size.max(1);
        self.tp_rank = tp_rank.min(self.tp_size - 1);
    }

    /// Wire the flat vCCL comm + this rank's TP-peer GLOBAL rank for the
    /// per-layer TP all-reduce. `peer < 0` (or `tp_size==1`) disables the
    /// reduce. The comm handle is the SAME one `set_collective_comm` holds for
    /// the PP hop.
    pub fn set_tp_comm(&mut self, comm: usize, peer: i32) {
        self.collective_comm = comm;
        self.tp_peer = peer;
    }

    /// Lever 1: lazily allocate + `comm_register` the persistent TP-reduce
    /// exchange scratch (`tp_send_scratch`/`tp_recv_scratch`, `[hs]` each),
    /// ONCE. Registration is best-effort: if unavailable/failing the buffers
    /// are still reused (persistent, no per-call `Vec` alloc), just paying the
    /// per-call `ibv_reg_mr`. Timed into the `nem_tp_reduce_regmr` bucket.
    /// `comm_register` has no size cap, so this is safe wherever the reduce runs.
    fn ensure_tp_scratch(&mut self, hs: usize) {
        if self.tp_reg_ready {
            return;
        }
        let t = std::time::Instant::now();
        // Allocated exactly once and never resized → stable heap addresses to
        // register. (Never touched again after this if already sized.)
        self.tp_send_scratch = vec![0.0f32; hs];
        self.tp_recv_scratch = vec![0.0f32; hs];
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        if !comm.is_null() && crate::vccl_ffi::registration_available() {
            let bytes = hs * std::mem::size_of::<f32>();
            match crate::vccl_ffi::comm_register(comm, self.tp_send_scratch.as_ptr() as usize, bytes) {
                Ok(h) => self.tp_send_reg = h,
                Err(e) => log::warn!("TP send scratch comm_register failed \
                    (per-call reg_mr fallback): {e}"),
            }
            match crate::vccl_ffi::comm_register(comm, self.tp_recv_scratch.as_ptr() as usize, bytes) {
                Ok(h) => self.tp_recv_reg = h,
                Err(e) => log::warn!("TP recv scratch comm_register failed \
                    (per-call reg_mr fallback): {e}"),
            }
            log::info!("nemotron TP-reduce scratch registered (send_h={:#x} recv_h={:#x}, {} B each)",
                self.tp_send_reg, self.tp_recv_reg, bytes);
        }
        self.tp_reg_ready = true;
        prof_add("nem_tp_reduce_regmr", t);
    }

    /// TP=2 pairwise exchange of the per-layer mixer partial resident in
    /// `NR_MIX` (the row-parallel projection output —
    /// moe `fc2(routed_partial)+shared_down_partial`). Reads `NR_MIX` back into
    /// the pre-registered send scratch, does ONE pairwise `send`/`recv` with the
    /// TP peer over the flat comm (even `tp_rank` sends first — deadlock-safe),
    /// and writes the peer's partial into `NR_PEER`. The sum that completes the
    /// all-reduce (`NR_MIX += NR_PEER`) and the residual add (`NR_H += NR_MIX`)
    /// are done ON THE GPU, folded into the NEXT layer's leading CB by
    /// [`Self::nem_rec_deferred_resid`] (lever 2) — so this method is now pure
    /// comm, no host arithmetic and no fresh allocation (lever 1). No-op when
    /// `tp_size<=1`. Uses `Python::with_gil` (the calling pymethod already holds
    /// the GIL). Timed into `nem_tp_reduce` (total) split by `_send`/`_recv`.
    ///
    /// TP=2 only: a pair's all-reduce == this single exchange. TP>2 would need a
    /// real sub-communicator (`vcclCommSplit`, absent from the FFI) — asserted.
    fn nem_tp_reduce_mix(&mut self) -> Option<()> {
        if self.tp_size <= 1 {
            return Some(());
        }
        assert_eq!(self.tp_size, 2,
            "nemotron TP all-reduce is TP=2 pairwise only; TP>2 needs a sub-communicator");
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        if comm.is_null() || self.tp_peer < 0 {
            log::error!("nemotron TP reduce: comm/peer not wired (comm={:?} peer={})",
                comm, self.tp_peer);
            return None;
        }
        let hs = self.config.hidden_size;
        let peer = self.tp_peer;
        let send_first = self.tp_rank % 2 == 0;
        let t_total = std::time::Instant::now();

        // Lever 1: persistent, pre-registered exchange scratch.
        self.ensure_tp_scratch(hs);

        // Read the local mixer partial (NR_MIX) back into the pre-registered
        // send scratch, IN PLACE (no per-call Vec alloc, no per-call reg_mr).
        {
            let src = unsafe { (*self.nr_ptr(NR_MIX)).mapped_ptr }.unwrap() as *const f32;
            let dst = self.tp_send_scratch.as_mut_ptr();
            unsafe { std::ptr::copy_nonoverlapping(src, dst, hs); }
        }

        // Disjoint field borrows so the exchange can send from one registered
        // buffer and recv into the other simultaneously.
        let send = &self.tp_send_scratch;
        let recv = &mut self.tp_recv_scratch;
        pyo3::Python::with_gil(|py| -> Result<(), String> {
            // Comm-floor Lever 3: prefer the duplex exchange primitive. ONE
            // `vcclSendRecv` sends our partial to the peer and recvs theirs in a
            // single call; the library picks full-duplex overlap under
            // `VCCL_DUPLEX_OVERLAP=1` (recovers the send-first rank's ~3.7 ms/tok
            // block) vs a deadlock-free ordered half-duplex fallback when unset.
            // Both modes exchange identical bytes → argmax-exact either way; only
            // the send/recv wait order changes. `send`/`recv` are disjoint
            // registered buffers (borrowed from separate fields above).
            if crate::vccl_ffi::send_recv_available() {
                let te = std::time::Instant::now();
                crate::vccl_ffi::send_recv_f32(
                    py, comm, &send[..hs], peer, &mut recv[..hs], peer,
                )?;
                prof_add("nem_tp_reduce_sendrecv", te);
            } else if send_first {
                // Legacy ordered fallback (libvccl without vcclSendRecv): lower
                // rank sends first to stay deadlock-free.
                let ts = std::time::Instant::now();
                crate::vccl_ffi::send_f32(py, comm, &send[..hs], peer)?;
                prof_add("nem_tp_reduce_send", ts);
                let tr = std::time::Instant::now();
                crate::vccl_ffi::recv_f32_into(py, comm, &mut recv[..hs], peer)?;
                prof_add("nem_tp_reduce_recv", tr);
            } else {
                let tr = std::time::Instant::now();
                crate::vccl_ffi::recv_f32_into(py, comm, &mut recv[..hs], peer)?;
                prof_add("nem_tp_reduce_recv", tr);
                let ts = std::time::Instant::now();
                crate::vccl_ffi::send_f32(py, comm, &send[..hs], peer)?;
                prof_add("nem_tp_reduce_send", ts);
            }
            Ok(())
        })
        .map_err(|e| log::error!("nemotron TP reduce exchange failed: {e}"))
        .ok()?;

        // Stage the peer's partial into NR_PEER for the GPU-side finish
        // (NR_MIX += NR_PEER; NR_H += NR_MIX), folded into the next layer's CB.
        let peerp = self.nr_ptr_mut(NR_PEER);
        unsafe { (*peerp).write(&f32_slice_to_bytes(&self.tp_recv_scratch[..hs])).ok()? };
        prof_add("nem_tp_reduce", t_total);
        Some(())
    }

    /// Lever 2 core: record the deferred TP residual add — `NR_MIX += NR_PEER`
    /// (finish the pairwise all-reduce ON the GPU) then `NR_H += NR_MIX` — as
    /// the LEADING dispatches of an already-open, barrier-primed CB, so the add
    /// rides the next layer's existing submit instead of a standalone
    /// begin/submit drain (removes ~1 blocking fence per MoE layer). Records a
    /// trailing barrier so the following `rms_norm(NR_H)` sees post-residual
    /// `NR_H`. Caller passes raw NR pointers (gathered before the `eng` borrow)
    /// + the shared add PC/wg. Timed into `nem_tp_add`.
    ///
    /// # Safety
    /// `hp`/`mixp`/`peerp` must point to the live `NR_H`/`NR_MIX`/`NR_PEER`
    /// buffers, and `cb` must be open with its leading barrier already recorded.
    unsafe fn nem_rec_deferred_resid(
        eng: &mut compute::ComputeEngine,
        cb: ash::vk::CommandBuffer,
        hp: *const compute::Buffer,
        mixp: *const compute::Buffer,
        peerp: *const compute::Buffer,
        add_pc: &[u8],
        wg_add: u32,
    ) -> Option<()> {
        let t = std::time::Instant::now();
        eng.record_to(cb, "add_f32_f32_f32", &[&*mixp, &*peerp, &*mixp, &*mixp], add_pc, (wg_add, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        eng.record_to(cb, "add_f32_f32_f32", &[&*hp, &*mixp, &*hp, &*hp], add_pc, (wg_add, 1, 1)).ok()?;
        eng.record_barrier_to(cb);
        prof_add("nem_tp_add", t);
        Some(())
    }

    /// End-of-span fallback for lever 2: when the LAST layer of a stage is a
    /// TP-reduce MoE layer there is no next layer's CB to fold the deferred add
    /// into, so flush it as a standalone (waited) CB. Same math as the folded
    /// path (`NR_MIX += NR_PEER; NR_H += NR_MIX`). Rare (≤1/token). (The
    /// `tp_size==1` path keeps the add fused in the mixer CB, unchanged.)
    fn nem_record_residual_add(&mut self, add_pc: &[u8], wg_add: u32, use_ring: bool) -> Option<()> {
        let hp = self.nr_ptr(NR_H);
        let mixp = self.nr_ptr(NR_MIX);
        let peerp = self.nr_ptr(NR_PEER);
        let eng = self.engine.as_mut()?;
        // Under the TP_RING lever the ring owns the CB/fence/descriptor state for
        // this token, so record this trailing add on the ring too (a blocking
        // `begin_batch`/`submit_batch` would reset `ds_cursor` and drive the
        // non-ring fence mid-token). It is drained by the caller's end-of-span
        // `wait_batch_pipelined`. The flag-OFF path keeps the blocking CB.
        if use_ring {
            let cb = eng.begin_batch_pipelined().ok()?;
            eng.record_barrier_to(cb);
            unsafe { Self::nem_rec_deferred_resid(eng, cb, hp, mixp, peerp, add_pc, wg_add)?; }
            eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
        } else {
            let cb = eng.begin_batch().ok()?;
            eng.record_barrier_to(cb);
            unsafe { Self::nem_rec_deferred_resid(eng, cb, hp, mixp, peerp, add_pc, wg_add)?; }
            eng.submit_batch(cb).ok()?;
        }
        Some(())
    }

    /// Open `path` (created/truncated) and write the trace header: magic
    /// `b"NMTP"` + `u32` LE `hidden_size`. Every subsequent LAST-stage
    /// `forward_pp_stage` call appends one record: `hidden_size` `f32` LE
    /// (the pre-`norm_f` `h_pre` — `last_pre_norm_hidden`) followed by one
    /// `u32` LE (the base model's own greedy next-token, i.e. `argmax(lm_head
    /// (norm_f(h_pre)))` — computed here from the SAME logits
    /// `forward_pp_stage` already produces, not re-derived by the caller).
    /// This is everything `nemotron_mtp::NemMtpHead::head_chain_cpu` needs
    /// (`first_hidden`/`first_embed`) to replay the DeepSeek-NextN draft
    /// chain completely OFFLINE, without the base model or a PP topology —
    /// see the module-level DECOUPLED-mode doc in `nemotron_mtp.rs`.
    pub fn attach_mtp_trace(&mut self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let mut f = std::fs::File::create(path).map_err(|e| format!("create {path}: {e}"))?;
        f.write_all(b"NMTP").map_err(|e| format!("write header magic: {e}"))?;
        f.write_all(&(self.config.hidden_size as u32).to_le_bytes())
            .map_err(|e| format!("write header hidden_size: {e}"))?;
        self.mtp_trace = Some(f);
        Ok(())
    }

    /// Attach a Vulkan compute engine and dequant+upload the Mamba `in_proj`/
    /// `out_proj` weights for every resident Mamba layer as f16, GPU-resident,
    /// so `mamba_mixer_prefill` can drive those two GEMMs through the batched
    /// `matmul_f16_f32_fp32` tiled kernel. The f32 host copies stay in
    /// `self.weights` as the correctness reference + decode-step path. Returns
    /// the number of projection tensors successfully uploaded. Called only when
    /// `flags.nemotron_gpu_mamba` is set and a device was created; on failure
    /// of any single upload that projection simply keeps its CPU path.
    pub fn attach_gpu_mamba(&mut self, mut engine: compute::ComputeEngine) -> usize {
        let mut uploaded = 0usize;
        let names: Vec<String> = (self.pp_start..self.pp_end)
            .filter(|&g| matches!(self.config.block_specs[g], BlockSpec::Mamba))
            .flat_map(|g| {
                let p = format!("backbone.layers.{g}.mixer");
                [format!("{p}.in_proj.weight"), format!("{p}.out_proj.weight")]
            })
            .collect();
        for name in names {
            // Dequantized f32 host copy (loader already ran FP8->f32); re-cast
            // to f16 bytes — the mul_mm kernel reads the weight as f16 (binding 0).
            let f32w = self.weights.f32_slice(&name);
            let mut bytes = vec![0u8; f32w.len() * 2];
            for (i, &v) in f32w.iter().enumerate() {
                bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
            }
            match engine
                .alloc_host_coherent_storage(bytes.len() as u64)
                .and_then(|buf| buf.write(&bytes).map(|_| buf))
            {
                Ok(buf) => {
                    self.gpu_proj.insert(name, buf);
                    uploaded += 1;
                }
                Err(e) => log::warn!("nemotron gpu-mamba: upload '{name}' failed: {e}; CPU fallback"),
            }
        }
        self.engine = Some(engine);
        uploaded
    }

    /// R2 (`VLLM_VULKAN_NEMOTRON_GPU_SCAN`): upload the per-layer Mamba2 SSD-
    /// scan constants (conv weight/bias, packed `[A_log|D|dt_bias]`,
    /// gated-RMSNorm weight) and allocate+zero the GPU-resident
    /// `ssm_state`/`conv_state` for every resident Mamba layer. Called
    /// lazily from `nem_res_probe` (the WS3-stage "resident loader") exactly
    /// like `ensure_nem_norm_weights`; idempotent — already-uploaded layers
    /// are skipped via `gpu_mamba.contains_key`.
    ///
    /// Returns `true` only if EVERY resident Mamba layer ends up with a
    /// `gpu_mamba` entry — a single bool, not a per-layer partial count —
    /// because `nem_mamba_resident_layer`'s GPU-scan branch must be
    /// all-or-nothing for the WHOLE run (see that function's doc comment): a
    /// `false` here makes `nem_res_probe` fail, which makes `ensure_nem_res`'s
    /// cached verdict permanently `false`, so the WHOLE resident stage (not
    /// just the GPU scan) falls back to the per-layer path in
    /// `forward_pp_range`. This is deliberately conservative — GPU-scan is a
    /// stage-wide invariant of the resident fast path here, not an
    /// independent per-layer toggle, so it can never flip CPU->GPU (or back)
    /// partway through a decode sequence.
    fn attach_gpu_mamba_scan(&mut self) -> bool {
        let dims = self.config.mamba_dims();
        let nh = dims.num_heads;
        let conv_dim = dims.conv_dim();
        let ss_len = dims.num_heads * dims.head_dim * dims.ssm_state_size;
        let conv_state_len = conv_dim * dims.conv_kernel;

        let layers: Vec<usize> = (self.pp_start..self.pp_end)
            .filter(|&g| matches!(self.config.block_specs[g], BlockSpec::Mamba))
            .collect();

        for layer_idx in layers.iter().copied() {
            if self.gpu_mamba.contains_key(&layer_idx) {
                continue;
            }
            let bufs = self.load_mamba_weights(layer_idx);
            if bufs.conv_w.len() != conv_dim * dims.conv_kernel
                || bufs.a_log.len() != nh
                || bufs.d_skip.len() != nh
                || bufs.dt_bias.len() != nh
                || bufs.norm_w.len() != dims.intermediate()
            {
                return false;
            }
            // use_conv_bias=true is a confirmed config invariant for this model.
            let conv_bias = match &bufs.conv_b {
                Some(b) if b.len() == conv_dim => b.clone(),
                _ => return false,
            };
            let mut params = Vec::with_capacity(3 * nh);
            params.extend_from_slice(&bufs.a_log);
            params.extend_from_slice(&bufs.d_skip);
            params.extend_from_slice(&bufs.dt_bias);

            let eng = match self.engine.as_mut() {
                Some(e) => e,
                None => return false,
            };
            let upload = |eng: &mut compute::ComputeEngine, data: &[f32]| -> Option<compute::Buffer> {
                let buf = eng.alloc_host_coherent_storage((data.len() * 4) as u64).ok()?;
                buf.write(&f32_slice_to_bytes(data)).ok()?;
                Some(buf)
            };
            let zeroed = |eng: &mut compute::ComputeEngine, n: usize| -> Option<compute::Buffer> {
                let buf = eng.alloc_host_coherent_storage((n * 4) as u64).ok()?;
                buf.write(&f32_slice_to_bytes(&vec![0.0f32; n])).ok()?;
                Some(buf)
            };

            let conv_w = match upload(eng, &bufs.conv_w) {
                Some(b) => b,
                None => return false,
            };
            let conv_bias_buf = match upload(eng, &conv_bias) {
                Some(b) => b,
                None => return false,
            };
            let norm_w = match upload(eng, &bufs.norm_w) {
                Some(b) => b,
                None => return false,
            };
            let params_buf = match upload(eng, &params) {
                Some(b) => b,
                None => return false,
            };
            let ssm_state = match zeroed(eng, ss_len) {
                Some(b) => b,
                None => return false,
            };
            let conv_state = match zeroed(eng, conv_state_len) {
                Some(b) => b,
                None => return false,
            };

            self.gpu_mamba.insert(
                layer_idx,
                NemMambaGpu {
                    ssm_state,
                    conv_state,
                    params: params_buf,
                    conv_w,
                    conv_bias: conv_bias_buf,
                    norm_w,
                },
            );
        }

        self.gpu_mamba.len() == layers.len()
    }

    #[inline]
    pub fn state_idx(&self, global_layer: usize) -> usize {
        debug_assert!(
            global_layer >= self.pp_start && global_layer < self.pp_end,
            "layer {global_layer} not resident on stage [{},{})",
            self.pp_start,
            self.pp_end
        );
        global_layer - self.pp_start
    }

    pub fn reset(&mut self) {
        for s in self.layer_state.iter_mut() {
            match s {
                LayerState::Mamba(m) => m.reset(),
                LayerState::Attention(c) => c.seq_len = 0,
                LayerState::Moe => {}
            }
        }
        // R2 (VLLM_VULKAN_NEMOTRON_GPU_SCAN): zero the GPU-resident scan
        // state too, so a new sequence can't inherit the previous one's
        // ssm_state/conv_state (mirrors Mamba2State::reset above, but for
        // the GPU-side state of record).
        for g in self.gpu_mamba.values() {
            let zero_ssm = vec![0u8; g.ssm_state.size as usize];
            let _ = g.ssm_state.write(&zero_ssm);
            let zero_conv = vec![0u8; g.conv_state.size as usize];
            let _ = g.conv_state.write(&zero_conv);
        }
    }

    /// Current decode position = number of tokens already committed to the KV
    /// cache = the `seq_len` of this stage's NoPE-attention layers (they advance
    /// in lockstep, one append per decode token; prefill fills them to the prompt
    /// length). Used by `pp_step_nemotron_logits` to derive the decode `pos` the
    /// generic `serve_dist` launcher does NOT pass. A stage whose
    /// `[pp_start,pp_end)` owns only Mamba/MoE layers has no attention `seq_len`
    /// and returns 0 — harmless, since those layers do not consume `pos`.
    pub fn current_decode_pos(&self) -> usize {
        self.layer_state
            .iter()
            .filter_map(|s| match s {
                LayerState::Attention(c) => Some(c.seq_len),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    fn w(&self, name: &str) -> Vec<f32> {
        self.weights.f32_slice(name).to_vec()
    }

    /// Host-f32 weight slice, or an EMPTY vec when the tensor isn't in the host
    /// map (the GPU-resident loader keeps matmul weights on the GPU, not host).
    /// Only for tensors that are either resident-or-host AND whose host copy is
    /// unused when resident (the Mamba in/out_proj — routed via `mamba_proj`).
    fn w_or_empty(&self, name: &str) -> Vec<f32> {
        self.weights
            .tensors
            .get(name)
            .map(|t| t.data.clone())
            .unwrap_or_default()
    }

    /// Single-mixer per-layer forward over the resident range `[start, end)`
    /// (GLOBAL indices). One pre-norm + one mixer + one residual per layer (NOT
    /// qwen35's attn-subblock-then-MLP-subblock). `hidden_in` is a
    /// `[n_tokens, hidden_size]` flat slab — `n_tokens == 1` is the decode call
    /// shape, `n_tokens > 1` is a prompt/prefill slab. `pos` is the sequence
    /// position of the FIRST token in the slab (attention KV-cache append
    /// advances one position per token; NoPE has no rotary table so `pos` is
    /// otherwise unused).
    ///
    /// Mamba layers get the real multi-token unblock: batched projections +
    /// the sequential SSD recurrence (`mamba_mixer_prefill`) for `n_tokens>1`,
    /// same decode step for `n_tokens==1`. Attention/MoE layers are looped
    /// per-token at this level (their own per-token math is unchanged; MoE is
    /// stateless per token and attention's KV cache already accumulates
    /// correctly across repeated calls) — this is orchestration only, not a
    /// new kernel, and keeps a mixed-block-type resident range correct for
    /// n_tokens>1 without touching those two kernels' per-token numerics.
    ///
    /// Skeleton: dispatches by `BlockSpec` and runs the fixture-validated
    /// kernels, but requires the (deferred) loader to have populated the
    /// `backbone.*` weights to actually RUN end-to-end.
    pub fn forward_pp_range(
        &mut self,
        hidden_in: &[f32],
        pos: usize,
        start: usize,
        end: usize,
    ) -> Vec<f32> {
        let cfg = self.config.clone();
        let eps = cfg.norm_eps;
        let hs = cfg.hidden_size;
        let n_tokens = hidden_in.len() / hs;
        debug_assert_eq!(
            n_tokens * hs,
            hidden_in.len(),
            "forward_pp_range: hidden_in.len() not a multiple of hidden_size"
        );

        // Increment 4 (VLLM_VULKAN_NEMOTRON_1CB): the WS3-style span-resident
        // path runs the WHOLE `[start,end)` span with `hidden` GPU-resident,
        // one host write in + one host read out. Only applies when `[start,
        // end)` is this model's OWN resident range (`nem_forward_range_
        // resident` reads `self.pp_start`/`self.pp_end` internally, mirroring
        // qwen3.6 WS3's per-stage bounds) — the fixture tests call
        // `forward_pp_range` with synthetic sub-ranges on an `engine=None`
        // model, which never takes this branch anyway. Falls back to the
        // Increment-1/2/3 per-layer path (below) on any readiness miss.
        if n_tokens == 1
            && start == self.pp_start
            && end == self.pp_end
            && self.engine.is_some()
            && nemotron_1cb_enabled()
        {
            if let Some(out) = self.nem_forward_range_resident(hidden_in, pos) {
                return out;
            }
        }

        let mut hidden = hidden_in.to_vec();
        for layer_idx in start..end {
            let residual = hidden.clone();
            let norm_w = self.w(&format!("backbone.layers.{layer_idx}.norm.weight"));
            let x = cpu_rms_norm(&hidden, &norm_w, eps); // [n_tokens, hidden_size], multi-row aware
            let mixer_out = match cfg.block_specs[layer_idx] {
                BlockSpec::Mamba => {
                    if n_tokens > 1 {
                        self.mamba_mixer_prefill(layer_idx, &x)
                    } else if n_tokens == 1 && self.engine.is_some() && nemotron_1cb_enabled() {
                        match self.mamba_mixer_1cb(layer_idx, &x) {
                            Some(out) => out,
                            None => self.mamba_mixer(layer_idx, &x),
                        }
                    } else {
                        self.mamba_mixer(layer_idx, &x)
                    }
                }
                BlockSpec::Attention => {
                    let mut out = vec![0.0f32; n_tokens * hs];
                    for ti in 0..n_tokens {
                        let xt = &x[ti * hs..(ti + 1) * hs];
                        let ot = if n_tokens == 1 && self.engine.is_some() && nemotron_1cb_enabled() {
                            match self.nope_attention_1cb(layer_idx, xt, pos + ti) {
                                Some(o) => o,
                                None => self.nope_attention(layer_idx, xt, pos + ti),
                            }
                        } else {
                            self.nope_attention(layer_idx, xt, pos + ti)
                        };
                        out[ti * hs..(ti + 1) * hs].copy_from_slice(&ot);
                    }
                    out
                }
                BlockSpec::Moe {
                    num_experts_per_tok,
                    moe_intermediate_size,
                } => {
                    let mut out = vec![0.0f32; n_tokens * hs];
                    for ti in 0..n_tokens {
                        let xt = &x[ti * hs..(ti + 1) * hs];
                        let ot = if n_tokens == 1 && self.engine.is_some() && nemotron_1cb_enabled() {
                            let dims = self.config.latent_moe_dims(num_experts_per_tok, moe_intermediate_size);
                            let p = format!("backbone.layers.{layer_idx}.mixer");
                            match self.latent_moe_1cb(layer_idx, &p, xt, &dims) {
                                Some(o) => o,
                                None => self.latent_moe_mixer(
                                    layer_idx,
                                    xt,
                                    num_experts_per_tok,
                                    moe_intermediate_size,
                                ),
                            }
                        } else {
                            self.latent_moe_mixer(layer_idx, xt, num_experts_per_tok, moe_intermediate_size)
                        };
                        out[ti * hs..(ti + 1) * hs].copy_from_slice(&ot);
                    }
                    out
                }
            };
            hidden = residual
                .iter()
                .zip(&mixer_out)
                .map(|(&r, &m)| r + m)
                .collect();
        }
        hidden
    }

    /// Full pipeline-parallel STAGE forward for ONE decode token: the per-stage
    /// embed / lm_head bookends wrapped around `forward_pp_range`. Mirrors
    /// `Qwen35Model::forward` (single node) / `forward_pp_qwen35_impl` (per
    /// stage) for the CPU-resident hybrid.
    ///
    ///  - FIRST stage (`pp_start == 0`): embed `token_id` from
    ///    `backbone.embeddings.weight` (`hidden_in` ignored — pass `&[]`).
    ///  - MID / LAST stage: continue from the previous stage's `hidden_in`
    ///    (`hidden_size` floats received over the PP hop).
    ///  - Runs the resident layers `[pp_start, pp_end)` via `forward_pp_range`,
    ///    which ADVANCES each resident layer's recurrent state IN PLACE (Mamba
    ///    conv_state + ssm_state, attention KV) — so decode step t+1 resumes
    ///    from step t's state. That per-stage recurrent state is NOT shipped
    ///    across the hop; only the `hidden_size` hidden vector crosses stages
    ///    (each stage owns a contiguous layer range, so layer i's Mamba
    ///    recurrence only ever runs on the one stage that owns layer i).
    ///  - LAST stage (`pp_end >= num_hidden_layers`): final `backbone.norm_f`
    ///    RMSNorm + `lm_head` GEMM → full `[vocab_size]` logits. Otherwise
    ///    returns the `hidden_size` hidden to send onward.
    pub fn forward_pp_stage(&mut self, token_id: u32, hidden_in: &[f32], pos: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let first = self.pp_start == 0;
        let last = self.pp_end >= cfg.num_hidden_layers;

        let hidden0: Vec<f32> = if first {
            let embed = self.weights.f32_slice("backbone.embeddings.weight");
            embed[token_id as usize * h..(token_id as usize + 1) * h].to_vec()
        } else {
            hidden_in.to_vec()
        };

        let hidden = self.forward_pp_range(&hidden0, pos, self.pp_start, self.pp_end);
        // Cache for the Phase-1 MTP acceptance sim (`nem_mtp_draft`) — see
        // the `last_pre_norm_hidden` field doc. Cheap (hidden_size floats);
        // done unconditionally so callers never race a stale value.
        self.last_pre_norm_hidden = hidden.clone();

        if last {
            let norm_w = self.w("backbone.norm_f.weight");
            let normed = cpu_rms_norm(&hidden, &norm_w, cfg.norm_eps);
            // lm_head through the resident f16 matvec (or f32 host `cpu_matmul`).
            let lm_name = self.lm_head_name.clone();
            let logits = self.nem_matvec(&lm_name, &normed, h, cfg.vocab_size);

            // Phase-1 DECOUPLED trace-dump (see `mtp_trace` field doc): one
            // record per decode step, using the SAME `hidden`/`logits` this
            // call already computed (no extra forward, no extra cost besides
            // the argmax + the write).
            if let Some(f) = self.mtp_trace.as_mut() {
                use std::io::Write;
                let mut best_i = 0u32;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best_i = i as u32;
                    }
                }
                let write_ok = (|| -> std::io::Result<()> {
                    for &v in &hidden {
                        f.write_all(&v.to_le_bytes())?;
                    }
                    f.write_all(&best_i.to_le_bytes())
                })();
                if let Err(e) = write_ok {
                    log::warn!("nemotron MTP trace write failed at pos {pos}: {e}");
                }
            }

            logits
        } else {
            hidden
        }
    }

    /// Owned Mamba2 weight buffers for one layer (sidesteps aliasing
    /// `self.weights` with `&mut self.layer_state` at the call site — shared by
    /// the single-token decode step and the multi-token prefill scan).
    fn load_mamba_weights(&self, layer_idx: usize) -> Mamba2WeightBufs {
        let p = format!("backbone.layers.{layer_idx}.mixer");
        Mamba2WeightBufs {
            // in_proj/out_proj are routed through `mamba_proj` (GPU-resident
            // FP8 matvec, or f32-host cpu_matmul); under the resident loader
            // they are NOT in the host map, so tolerate their absence here (the
            // SSD recurrence `mamba2_scan_only` never reads them — only the
            // standalone `mamba2_decode_step`/`mamba2_prefill_seq`, which the
            // model path no longer calls, do).
            in_proj: self.w_or_empty(&format!("{p}.in_proj.weight")),
            conv_w: self.w(&format!("{p}.conv1d.weight")),
            conv_b: if dims_uses_conv_bias(&self.config) {
                Some(self.w(&format!("{p}.conv1d.bias")))
            } else {
                None
            },
            a_log: self.w(&format!("{p}.A_log")),
            d_skip: self.w(&format!("{p}.D")),
            dt_bias: self.w(&format!("{p}.dt_bias")),
            norm_w: self.w(&format!("{p}.norm.weight")),
            out_proj: self.w_or_empty(&format!("{p}.out_proj.weight")),
        }
    }

    /// Mamba2/SSD mixer for one resident layer (single-token decode step,
    /// advances state). Used for T=1 (decode).
    pub fn mamba_mixer(&mut self, layer_idx: usize, x: &[f32]) -> Vec<f32> {
        // Single-token decode is the T=1 case of the prefill scan: it routes the
        // in_proj/out_proj through `mamba_proj` (GPU-resident when uploaded, else
        // `cpu_matmul`) and reuses the verbatim per-token recurrence. Bit-exact
        // vs the standalone `mamba2_decode_step` (see the prefill==T×decode test).
        self.mamba_mixer_prefill(layer_idx, x)
    }

    /// Mamba2/SSD mixer for one resident layer, MULTI-token prefill (T>1):
    /// batched in_proj/out_proj GEMMs + the sequential SSD recurrence
    /// (`mamba2_prefill_seq`), advancing state exactly as T decode steps would.
    ///
    /// `xs`: `[n_tokens, hidden_size]` flat. Returns `[n_tokens, hidden_size]` flat.
    pub fn mamba_mixer_prefill(&mut self, layer_idx: usize, xs: &[f32]) -> Vec<f32> {
        let dims = self.config.mamba_dims();
        let si = self.state_idx(layer_idx);
        let hs = dims.hidden_size;
        let inter = dims.intermediate();
        let in_proj_out = dims.in_proj_out();
        let t = xs.len() / hs;
        let p = format!("backbone.layers.{layer_idx}.mixer");
        let in_name = format!("{p}.in_proj.weight");
        let out_name = format!("{p}.out_proj.weight");

        // (1) in_proj GEMM (GPU when resident, else CPU) — releases the &mut
        // borrow before the scan touches layer_state.
        let proj = self.mamba_proj(&in_name, xs, t, hs, in_proj_out); // [T, in_proj_out]

        // (2) sequential conv1d + SSD recurrence (CPU-only recurrence).
        let bufs = self.load_mamba_weights(layer_idx);
        let scans = {
            let weights = bufs.borrow();
            let st = match &mut self.layer_state[si] {
                LayerState::Mamba(s) => s,
                _ => unreachable!("mamba layer has a Mamba2State"),
            };
            mamba2_scan_only(&proj, &weights, st, &dims) // [T, inter]
        };

        // (3) out_proj GEMM (GPU when resident, else CPU).
        self.mamba_proj(&out_name, &scans, t, inter, hs)
    }

    /// One Mamba projection: `out[t,n] = x[t,k] @ W[n,k]^T`. When the f16 weight
    /// `name` is GPU-resident (`gpu_proj`) and an engine is present, dispatch the
    /// proven batched tiled GEMM (`matmul_f16_f32_fp32`, the same kernel + index
    /// mapping as `VulkanModel::gpu_gemm`); otherwise fall back to the exact CPU
    /// `cpu_matmul` on the f32 host weight. The two `in_proj`/`out_proj` GEMMs are
    /// the FLOP-dominant part of the Mamba prefill mixer; the caller keeps the
    /// intervening SSD scan on CPU.
    fn mamba_proj(&mut self, name: &str, x: &[f32], t: usize, k: usize, n: usize) -> Vec<f32> {
        // Resident (FP8/f16) weight: dispatch the dequant-in-shader matvec once
        // per token row (correct for any T; the batched GEMM is an M6-only f16
        // path). Takes precedence over the M6 f16 GEMM buffer when both exist.
        if self.gpu_weights.contains_key(name) {
            let mut out = vec![0.0f32; t * n];
            for ti in 0..t {
                let row = self.nem_matvec(name, &x[ti * k..(ti + 1) * k], k, n);
                out[ti * n..(ti + 1) * n].copy_from_slice(&row);
            }
            return out;
        }
        const BM: usize = 64;
        const BN: usize = 64;
        // Raw ptr to the resident weight buffer so the immutable `gpu_proj`
        // borrow ends before the mutable `engine` borrow (mirrors gpu_gemm).
        let w_ptr = self.gpu_proj.get(name).map(|b| b as *const compute::Buffer);
        if let (Some(eng), Some(w_ptr)) = (self.engine.as_mut(), w_ptr) {
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((t * k * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let out = eng.alloc_host_coherent_storage((t * n * 4) as u64).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let out_p = &out as *const compute::Buffer;
            let pc = gemm_pc(t, n, k);
            let wg = (((n + BM - 1) / BM) as u32, ((t + BN - 1) / BN) as u32, 1u32);
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, "matmul_f16_f32_fp32", &[&*w_ptr, &*inp_p, &*out_p], &pc, wg).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            let result = read_f32_buf(&out, t * n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
            result
        } else {
            cpu_matmul(x, self.weights.f32_slice(name), t, k, n)
        }
    }

    /// One GPU-resident dequant-matvec `out[1,n] = x[1,k] @ W[n,k]^T`, reading
    /// the quantized weight `name` from `gpu_weights` (NVFP4 / FP8 / f16) and
    /// dequantizing IN the shader — the direct reuse of the qwen3.6
    /// `qwen35_matvec` dispatch (`mul_mat_vec_nvfp4` / `mul_mat_vec_fp8` /
    /// `mul_mat_vec_*_f16`). When the weight is NOT resident (device-less Mac,
    /// or the resident loader didn't run), falls back to the exact f32 host
    /// `cpu_matmul` — so this is bit-identical to the CPU reference whenever the
    /// GPU store is empty (the property the Mac tests rely on).
    fn nem_matvec(&mut self, name: &str, x: &[f32], k: usize, n: usize) -> Vec<f32> {
        // Gather the buffer ptr + Copy dispatch descriptor first so the
        // immutable `gpu_weights` borrow ends before `engine.as_mut()`.
        let meta = self.nem_meta(name);
        if let (Some(eng), Some((w_ptr, kind))) = (self.engine.as_mut(), meta) {
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let out_p = &out as *const compute::Buffer;
            let cb = eng.begin_batch().unwrap();
            Self::nem_rec_mv(eng, cb, (w_ptr, kind), inp_p, out_p, k, n).unwrap();
            eng.submit_batch(cb).unwrap();
            let result = read_f32_buf(&out, n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
            result
        } else {
            cpu_matmul(x, self.weights.f32_slice(name), 1, k, n)
        }
    }

    /// GPU weight meta (buffer + dequant kind) for one resident Nemotron
    /// projection. Raw pointer stays valid through a recording: `gpu_weights`
    /// is never mutated during a forward (same invariant as `nem_matvec`
    /// before this was factored out). Mirrors qwen3.6 WS3's `q35r_meta`
    /// (`qwen35_forward.rs`).
    fn nem_meta(&self, name: &str) -> Option<(*const compute::Buffer, NemMvKind)> {
        self.gpu_weights.get(name).map(|w| {
            (
                &w.buffer as *const compute::Buffer,
                match &w.quant {
                    NemQuant::F16 => NemMvKind::F16,
                    NemQuant::Q8_0 => NemMvKind::Q8_0,
                    NemQuant::Nvfp4 { scales, group_size, e4m3, global } => {
                        NemMvKind::Nvfp4 {
                            s: scales as *const _, gs: *group_size,
                            e4m3: *e4m3, global: *global,
                        }
                    }
                    NemQuant::Fp8 { scale, per_row } => {
                        NemMvKind::Fp8 { s: scale as *const _, per_row: *per_row }
                    }
                },
            )
        })
    }

    /// Pick the routed-expert e4m3 NVFP4 matvec shader + rows for Nemotron. When
    /// `VLLM_VULKAN_LAGUNA_EXPERT_REPACK` is on AND the shape clears the repack
    /// guard (`nvfp4_repack_shape_ok` — k%32==0, k>=1024, n>=1024, gs==16), route
    /// to the address-gen-free REPACK kernel
    /// (`mul_mat_vec_nvfp4_e4m3repack_f32_f32_bs64_r4`) instead of the v1
    /// `mul_mat_vec_nvfp4_e4m3` oracle. Mirrors `laguna_gpu::laguna_e4m3_expert_shader`
    /// and `step3p7_gpu::step3p7_e4m3_expert_shader` EXACTLY: the repack shader threads
    /// `packed_off`/`sb_off` + the per-tensor `global` identically (same push block via
    /// `matvec_nvfp4_e4m3_pc`/`matvec_nvfp4_e4m3_pc_off`, unchanged), so the dequant math
    /// is bit-exact (repack == f32-fold, single IEEE mul ⇒ argmax-exact vs v1). Only fires
    /// under `VLLM_VULKAN_NVFP4_E4M3_SCALES` (the e4m3-resident footprint lever); the
    /// default f32-fold expert path already reaches the nvfp4 repack via
    /// `matvec_nvfp4_variant_k`. Flag off / out-of-gate shape ⇒ v1 (byte-identical).
    fn nem_e4m3_expert_shader(k: usize, n: usize, gs: usize) -> (String, u32) {
        if laguna_expert_repack_flag() && nvfp4_repack_shape_ok(k, n, gs) {
            return ("mul_mat_vec_nvfp4_e4m3repack_f32_f32_bs64_r4".to_string(), 4);
        }
        matvec_nvfp4_e4m3_variant(n)
    }

    /// Record ONE format-routed matvec dispatch into an already-open command
    /// buffer — the shared building block for `nem_matvec` and the WS3-style
    /// resident CB helpers (`mamba_mixer_1cb` and later increments). Pure
    /// refactor of `nem_matvec`'s former inline GPU arm: identical shaders,
    /// identical push-constants, identical dispatch order — bitwise-identical
    /// output. Mirrors qwen3.6 WS3's `q35r_rec_mv` (`qwen35_forward.rs`).
    fn nem_rec_mv(
        eng: &mut compute::ComputeEngine,
        cb: ash::vk::CommandBuffer,
        (w_ptr, kind): (*const compute::Buffer, NemMvKind),
        ip: *const compute::Buffer,
        op: *const compute::Buffer,
        k: usize,
        n: usize,
    ) -> Option<()> {
        unsafe {
            match kind {
                NemMvKind::Nvfp4 { s, gs, e4m3, global } => {
                    // e4m3-resident (VLLM_VULKAN_NVFP4_E4M3_SCALES): raw e4m3
                    // scale bytes + per-tensor global re-applied in-shader —
                    // bit-exact to the f32-fold kernel. Same 4-buffer binding
                    // order (packed, scale, x, dst) either way; only the shader
                    // name + push constants differ. See push_constants::
                    // nvfp4_dispatch + mul_mat_vec_nvfp4_e4m3.comp.
                    if e4m3 {
                        let (shader, r) = Self::nem_e4m3_expert_shader(k, n, gs as usize);
                        let wg = (n as u32 + r - 1) / r;
                        let pc = matvec_nvfp4_e4m3_pc(k, n, gs as usize, global);
                        eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
                    } else {
                        let (shader, r) = matvec_nvfp4_variant_k(k, n);
                        let wg = (n as u32 + r - 1) / r;
                        let pc = matvec_mlx4_pc_off(k, n, gs as usize, 0, 0);
                        eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
                    }
                }
                NemMvKind::Fp8 { s, per_row } => {
                    let (shader, r) = matvec_fp8_variant_k(k, n);
                    let wg = (n as u32 + r - 1) / r;
                    let pc = matvec_fp8_pc(k, n, per_row);
                    eng.record_to(cb, &shader, &[&*w_ptr, &*s, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
                }
                NemMvKind::F16 => {
                    // PINNED f16 base — the weight is genuinely f16 in memory
                    // (NemQuant::F16), so it must NOT be routed through the
                    // VLLM_VULKAN_QUANT-driven matvec_variant (which would
                    // dispatch the q8_0/q4 dequant shader on f16 bytes ->
                    // garbage logits when q8_0 is exported for a co-hosted
                    // model). See matvec_f16_variant.
                    let (shader, r) = matvec_f16_variant_k(k, n);
                    let wg = (n as u32 + r - 1) / r;
                    let pc = matvec_pc13(k, n);
                    eng.record_to(cb, &shader, &[&*w_ptr, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
                }
                NemMvKind::Q8_0 => {
                    let (shader, r) = matvec_q8_0_variant_k(k, n);
                    let wg = (n as u32 + r - 1) / r;
                    let pc = matvec_pc13(k, n);
                    eng.record_to(cb, &shader, &[&*w_ptr, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
                }
            }
        }
        Some(())
    }

    /// WS3-style 2-CB Mamba mixer scaffold (`VLLM_VULKAN_NEMOTRON_1CB`, T=1
    /// decode only). Same two matvec dispatches and the same unchanged CPU
    /// SSD scan as `mamba_mixer_prefill`'s T=1 case — this only collapses
    /// each matvec's own `begin_batch`/`submit_batch` into a pair of named
    /// CBs (CB_A: in_proj, CB_B: out_proj) via `nem_rec_mv`, so it is
    /// bitwise-identical to the current resident Mamba path (batching
    /// submits cannot change kernel outputs). Mamba's two matvecs are NOT
    /// batchable into one CB — the SSD scan is a mandatory host round-trip
    /// between them — so this is deliberately a scaffold, not a speedup (see
    /// the Nemotron 1-CB plan, Increment 1). Returns `None` when `in_proj`/
    /// `out_proj` are not GPU-resident (device-less host, or the resident
    /// loader didn't run) or the engine is absent, so the caller falls back
    /// to `mamba_mixer`.
    fn mamba_mixer_1cb(&mut self, layer_idx: usize, x: &[f32]) -> Option<Vec<f32>> {
        let dims = self.config.mamba_dims();
        let si = self.state_idx(layer_idx);
        let hs = dims.hidden_size;
        let inter = dims.intermediate();
        let in_proj_out = dims.in_proj_out();
        let p = format!("backbone.layers.{layer_idx}.mixer");
        let in_name = format!("{p}.in_proj.weight");
        let out_name = format!("{p}.out_proj.weight");

        // CB_A: in_proj matvec, one fenced submit.
        let in_meta = self.nem_meta(&in_name)?;
        let eng = self.engine.as_mut()?;
        let xb = f32_slice_to_bytes(x);
        let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).ok()?;
        inp.write(&xb).ok()?;
        let out = eng.alloc_host_coherent_storage((in_proj_out * 4) as u64).ok()?;
        let inp_p = &inp as *const compute::Buffer;
        let out_p = &out as *const compute::Buffer;
        let cb = eng.begin_batch().ok()?;
        Self::nem_rec_mv(eng, cb, in_meta, inp_p, out_p, hs, in_proj_out)?;
        eng.submit_batch(cb).ok()?;
        let proj = read_f32_buf(&out, in_proj_out);
        eng.return_to_pool(inp);
        eng.return_to_pool(out);

        // host: conv1d + SSD recurrence — UNCHANGED (mandatory recurrence,
        // not batchable with either matvec).
        let bufs = self.load_mamba_weights(layer_idx);
        let scan = {
            let weights = bufs.borrow();
            let st = match &mut self.layer_state[si] {
                LayerState::Mamba(s) => s,
                _ => unreachable!("mamba layer has a Mamba2State"),
            };
            mamba2_scan_only(&proj, &weights, st, &dims) // [1, inter]
        };

        // CB_B: out_proj matvec, one fenced submit.
        let out_meta = self.nem_meta(&out_name)?;
        let eng = self.engine.as_mut()?;
        let xb = f32_slice_to_bytes(&scan);
        let inp = eng.alloc_host_coherent_storage((scan.len() * 4) as u64).ok()?;
        inp.write(&xb).ok()?;
        let out = eng.alloc_host_coherent_storage((hs * 4) as u64).ok()?;
        let inp_p = &inp as *const compute::Buffer;
        let out_p = &out as *const compute::Buffer;
        let cb = eng.begin_batch().ok()?;
        Self::nem_rec_mv(eng, cb, out_meta, inp_p, out_p, inter, hs)?;
        eng.submit_batch(cb).ok()?;
        let result = read_f32_buf(&out, hs);
        eng.return_to_pool(inp);
        eng.return_to_pool(out);
        Some(result)
    }

    /// One routed-expert NVFP4 matvec into a concatenated per-layer expert
    /// buffer, offset to expert `e` via the nvfp4 shader's `packed_off`/`sb_off`
    /// (word/element strides = `e * out * in/8` and `e * out * (in/group_size)`),
    /// the exact qwen `MoeGpuLayer` slicing pattern. `up == true` selects the
    /// `up_proj` buffer, else `down_proj`. Requires an engine + a resident
    /// expert buffer for `layer` (guaranteed by the caller, which checks
    /// `gpu_experts`).
    fn nem_expert_matvec(&mut self, layer: usize, up: bool, e: usize, x: &[f32]) -> Vec<f32> {
        let meta = self.nem_expert_meta(layer, up).expect("resident experts for layer");
        let n = meta.3;
        let eng = self.engine.as_mut().expect("engine present for resident experts");
        let xb = f32_slice_to_bytes(x);
        let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
        inp.write(&xb).unwrap();
        let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        let inp_p = &inp as *const compute::Buffer;
        let out_p = &out as *const compute::Buffer;
        let cb = eng.begin_batch().unwrap();
        Self::nem_rec_expert_mv(eng, cb, meta, e, inp_p, out_p).unwrap();
        eng.submit_batch(cb).unwrap();
        let result = read_f32_buf(&out, n);
        eng.return_to_pool(inp);
        eng.return_to_pool(out);
        result
    }

    /// Gather one routed-expert's dispatch geometry + raw buffer ptrs (Copy)
    /// so the immutable `gpu_experts` borrow ends before `engine.as_mut()`.
    /// Returns `(w_ptr, s_ptr, k, n, group_size)` — `k`=in_features,
    /// `n`=out_features, matching `nem_meta`'s convention. `None` when `layer`
    /// has no resident expert buffer.
    fn nem_expert_meta(
        &self,
        layer: usize,
        up: bool,
    ) -> Option<ExpertMeta> {
        let ex = self.gpu_experts.get(&layer)?;
        let (w, s, out_f, in_f, globals) = if up {
            (&ex.up, &ex.up_scales, ex.up_out, ex.up_in, &ex.up_globals)
        } else {
            (&ex.down, &ex.down_scales, ex.down_out, ex.down_in, &ex.down_globals)
        };
        Some((
            w as *const compute::Buffer,
            s as *const compute::Buffer,
            in_f,
            out_f,
            ex.group_size as usize,
            ex.e4m3,
            globals as *const Vec<f32>,
        ))
    }

    /// Record ONE routed-expert NVFP4 matvec dispatch into an already-open
    /// command buffer, offset to expert `e` via the nvfp4 shader's
    /// `packed_off`/`sb_off` (word/element strides = `e * out * in/8` and
    /// `e * out * (in/group_size)`) — the exact qwen `MoeGpuLayer` slicing
    /// pattern. Pure refactor of `nem_expert_matvec`'s former inline body:
    /// identical shader, identical push-constants, identical offsets —
    /// bitwise-identical output. Lets `latent_moe_1cb` batch top_k
    /// independent expert dispatches into one CB.
    fn nem_rec_expert_mv(
        eng: &mut compute::ComputeEngine,
        cb: ash::vk::CommandBuffer,
        (w_ptr, s_ptr, k, n, gs, e4m3, globals): ExpertMeta,
        e: usize,
        ip: *const compute::Buffer,
        op: *const compute::Buffer,
    ) -> Option<()> {
        let packed_off = e * n * (k / 8); // words: expert * out_features * (in/8)
        // sb_off is the per-expert base into the scale[] array: for f32-fold it
        // is an f32-ELEMENT offset, for e4m3 a BYTE-ELEMENT offset — numerically
        // identical (`e * out * groups`), since the e4m3 scale is 1 byte/group
        // where the folded scale is 1 f32/group.
        let sb_off = e * n * (k / gs);
        unsafe {
            if e4m3 {
                let global = (&*globals)[e]; // this expert's .weight_scale_2
                let (shader, r) = Self::nem_e4m3_expert_shader(k, n, gs);
                let wg = (n as u32 + r - 1) / r;
                let pc = matvec_nvfp4_e4m3_pc_off(k, n, gs, packed_off, sb_off, global);
                eng.record_to(cb, &shader, &[&*w_ptr, &*s_ptr, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
            } else {
                let (shader, r) = matvec_nvfp4_variant_k(k, n);
                let wg = (n as u32 + r - 1) / r;
                let pc = matvec_mlx4_pc_off(k, n, gs, packed_off, sb_off);
                eng.record_to(cb, &shader, &[&*w_ptr, &*s_ptr, &*ip, &*op], &pc, (wg, 1, 1)).ok()?;
            }
        }
        Some(())
    }

    /// Like `nem_rec_expert_mv`, but the input/output live at byte offsets
    /// into shared multi-expert scratch buffers (`lat_buf` sliced per-expert,
    /// `up_all`/`down_all` concatenated top_k-wide) instead of each expert
    /// getting its own dedicated buffer — used by the MoE-tail collapse
    /// (`VLLM_VULKAN_NEMOTRON_MOE_TAIL`, R1b) so all top_k expert matvecs can
    /// batch into one CB via `record_to_off`. The weight-addressing PC
    /// (`packed_off`/`sb_off` for expert `e`) is identical to
    /// `nem_rec_expert_mv` — only the input/output buffer bindings gain a
    /// byte offset.
    fn nem_rec_expert_mv_off(
        eng: &mut compute::ComputeEngine,
        cb: ash::vk::CommandBuffer,
        (w_ptr, s_ptr, k, n, gs, e4m3, globals): ExpertMeta,
        e: usize,
        ip: *const compute::Buffer,
        in_off: u64,
        op: *const compute::Buffer,
        out_off: u64,
    ) -> Option<()> {
        let packed_off = e * n * (k / 8); // words: expert * out_features * (in/8)
        let sb_off = e * n * (k / gs); // f32-elem (fold) or byte-elem (e4m3): same value
        unsafe {
            let (shader, r, pc) = if e4m3 {
                let global = (&*globals)[e];
                let (shader, r) = Self::nem_e4m3_expert_shader(k, n, gs);
                (shader, r, matvec_nvfp4_e4m3_pc_off(k, n, gs, packed_off, sb_off, global))
            } else {
                let (shader, r) = matvec_nvfp4_variant_k(k, n);
                (shader, r, matvec_mlx4_pc_off(k, n, gs, packed_off, sb_off))
            };
            let wg = (n as u32 + r - 1) / r;
            eng.record_to_off(
                cb,
                &shader,
                &[(&*w_ptr, 0), (&*s_ptr, 0), (&*ip, in_off), (&*op, out_off)],
                &pc,
                (wg, 1, 1),
            ).ok()?;
        }
        Some(())
    }

    /// WS3-style 2-CB NoPE attention (`VLLM_VULKAN_NEMOTRON_1CB`, T=1 decode
    /// only). q/k/v all read the same `x` and write independent outputs, so
    /// they batch into ONE fenced CB (CB_A, 3 dispatches, no inter-dispatch
    /// barrier needed — batching independent dispatches into one submit
    /// cannot change kernel outputs). KV-cache append + `cpu_sdpa` stay on
    /// the host, unchanged (lever #4). `o_proj` is a second fenced CB
    /// (CB_B) since it depends on the host SDPA result. Bitwise-identical to
    /// `nope_attention`'s GPU arm: same four matvec shaders, same order,
    /// only the submit count collapses (4 -> 2). Returns `None` when any of
    /// q/k/v/o_proj is not GPU-resident (or the engine is absent), so the
    /// caller falls back to `nope_attention`.
    fn nope_attention_1cb(&mut self, layer_idx: usize, x: &[f32], _pos: usize) -> Option<Vec<f32>> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let p = format!("backbone.layers.{layer_idx}.mixer");
        let si = self.state_idx(layer_idx);

        // CB_A: q_proj, k_proj, v_proj — independent dispatches over the same
        // `x`, batched into one fenced submit.
        let q_meta = self.nem_meta(&format!("{p}.q_proj.weight"))?;
        let k_meta = self.nem_meta(&format!("{p}.k_proj.weight"))?;
        let v_meta = self.nem_meta(&format!("{p}.v_proj.weight"))?;
        let eng = self.engine.as_mut()?;
        let xb = f32_slice_to_bytes(x);
        let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).ok()?;
        inp.write(&xb).ok()?;
        let q_out = eng.alloc_host_coherent_storage((q_dim * 4) as u64).ok()?;
        let k_out = eng.alloc_host_coherent_storage((kv_dim * 4) as u64).ok()?;
        let v_out = eng.alloc_host_coherent_storage((kv_dim * 4) as u64).ok()?;
        let inp_p = &inp as *const compute::Buffer;
        let q_out_p = &q_out as *const compute::Buffer;
        let k_out_p = &k_out as *const compute::Buffer;
        let v_out_p = &v_out as *const compute::Buffer;
        let cb = eng.begin_batch().ok()?;
        Self::nem_rec_mv(eng, cb, q_meta, inp_p, q_out_p, h, q_dim)?;
        Self::nem_rec_mv(eng, cb, k_meta, inp_p, k_out_p, h, kv_dim)?;
        Self::nem_rec_mv(eng, cb, v_meta, inp_p, v_out_p, h, kv_dim)?;
        eng.submit_batch(cb).ok()?;
        let q = read_f32_buf(&q_out, q_dim);
        let k = read_f32_buf(&k_out, kv_dim);
        let v = read_f32_buf(&v_out, kv_dim);
        eng.return_to_pool(inp);
        eng.return_to_pool(q_out);
        eng.return_to_pool(k_out);
        eng.return_to_pool(v_out);

        // host: KV-cache append + SDPA — UNCHANGED (lever #4).
        let attn = {
            let cache = match &mut self.layer_state[si] {
                LayerState::Attention(c) => c,
                _ => unreachable!("attention layer has a KV cache"),
            };
            cache.append(&k, &v);
            cpu_sdpa(
                &q,
                cache.k_up_to_now(),
                cache.v_up_to_now(),
                nq,
                nkv,
                hd,
                cache.seq_len,
                scale,
                None,
            )
        };

        // CB_B: o_proj matvec, one fenced submit.
        let o_meta = self.nem_meta(&format!("{p}.o_proj.weight"))?;
        let eng = self.engine.as_mut()?;
        let ab = f32_slice_to_bytes(&attn);
        let inp = eng.alloc_host_coherent_storage((attn.len() * 4) as u64).ok()?;
        inp.write(&ab).ok()?;
        let out = eng.alloc_host_coherent_storage((h * 4) as u64).ok()?;
        let inp_p = &inp as *const compute::Buffer;
        let out_p = &out as *const compute::Buffer;
        let cb = eng.begin_batch().ok()?;
        Self::nem_rec_mv(eng, cb, o_meta, inp_p, out_p, q_dim, h)?;
        eng.submit_batch(cb).ok()?;
        let result = read_f32_buf(&out, h);
        eng.return_to_pool(inp);
        eng.return_to_pool(out);
        Some(result)
    }

    /// NoPE GQA attention for one resident layer. Reuses `cpu_sdpa`; no RoPE, no
    /// q/k-norm, no output gate (strictly simpler than qwen35's gated attention).
    pub fn nope_attention(&mut self, layer_idx: usize, x: &[f32], _pos: usize) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let p = format!("backbone.layers.{layer_idx}.mixer");
        let si = self.state_idx(layer_idx);

        // q/k/v projections through the GPU-resident dequant-matvec (f16 for the
        // BF16-native attn weights), or f32 host `cpu_matmul` when not resident.
        let q = self.nem_matvec(&format!("{p}.q_proj.weight"), x, h, q_dim);
        let k = self.nem_matvec(&format!("{p}.k_proj.weight"), x, h, kv_dim);
        let v = self.nem_matvec(&format!("{p}.v_proj.weight"), x, h, kv_dim);

        // no rope, no qk-norm.
        let attn = {
            let cache = match &mut self.layer_state[si] {
                LayerState::Attention(c) => c,
                _ => unreachable!("attention layer has a KV cache"),
            };
            cache.append(&k, &v);
            cpu_sdpa(
                &q,
                cache.k_up_to_now(),
                cache.v_up_to_now(),
                nq,
                nkv,
                hd,
                cache.seq_len,
                scale,
                None,
            )
        };
        self.nem_matvec(&format!("{p}.o_proj.weight"), &attn, q_dim, h)
    }

    /// Latent-MoE mixer for one resident layer (single token).
    pub fn latent_moe_mixer(
        &mut self,
        layer_idx: usize,
        x: &[f32],
        top_k: usize,
        moe_inter: usize,
    ) -> Vec<f32> {
        let dims = self.config.latent_moe_dims(top_k, moe_inter);
        let ne = self.config.n_routed_experts;
        let lat = dims.moe_latent_size;
        let p = format!("backbone.layers.{layer_idx}.mixer");
        // GPU-resident routed experts (NVFP4) + FP8/f16 projections: dispatch
        // only the top_k selected experts, dequant-in-shader. Falls through to
        // the CPU concatenate path below when the layer isn't resident.
        if self.gpu_experts.contains_key(&layer_idx) {
            return self.latent_moe_routed(layer_idx, &p, x, &dims);
        }
        // Concatenate the per-expert tensors into contiguous [ne, ...] buffers.
        let mut expert_up = Vec::with_capacity(ne * moe_inter * lat);
        let mut expert_down = Vec::with_capacity(ne * lat * moe_inter);
        for e in 0..ne {
            expert_up.extend_from_slice(self.weights.f32_slice(&format!("{p}.experts.{e}.up_proj.weight")));
            expert_down
                .extend_from_slice(self.weights.f32_slice(&format!("{p}.experts.{e}.down_proj.weight")));
        }
        let gate_weight = self.w(&format!("{p}.gate.weight"));
        let e_bias = self.w(&format!("{p}.gate.e_score_correction_bias"));
        let fc1 = self.w(&format!("{p}.fc1_latent_proj.weight"));
        let fc2 = self.w(&format!("{p}.fc2_latent_proj.weight"));
        let shared_up = self.w(&format!("{p}.shared_experts.up_proj.weight"));
        let shared_down = self.w(&format!("{p}.shared_experts.down_proj.weight"));
        let weights = LatentMoeWeights {
            gate_weight: &gate_weight,
            e_score_correction_bias: &e_bias,
            fc1_latent_proj: &fc1,
            fc2_latent_proj: &fc2,
            expert_up: &expert_up,
            expert_down: &expert_down,
            shared_up: &shared_up,
            shared_down: &shared_down,
        };
        latent_moe_forward(x, 1, &weights, &dims)
    }

    /// GPU-resident latent-MoE for a single token: the NVFP4 routed experts are
    /// resident (`gpu_experts[layer]`, per-expert nvfp4 matvec on the top_k
    /// selected experts only), and the fc1/fc2/shared projections go through the
    /// FP8/f16 resident `nem_matvec`. Faithful op-for-op mirror of
    /// `latent_moe_forward` for n_tokens=1 (router on the RAW pre-latent hidden;
    /// routed path in the moe_latent bottleneck; shared expert parallel on the
    /// FULL hidden), so it degrades to numerically-equal-modulo-quant results vs
    /// the CPU reference (cluster-validated cos vs single-node CPU-ref).
    fn latent_moe_routed(
        &mut self,
        layer_idx: usize,
        p: &str,
        x: &[f32],
        d: &LatentMoeDims,
    ) -> Vec<f32> {
        let hs = d.hidden_size;
        let lat = d.moe_latent_size;
        let shared_inter = d.moe_shared_expert_intermediate_size;

        // Router: small BF16 gate.weight + e_score_correction_bias stay f32 host.
        let gate_weight = self.weights.f32_slice(&format!("{p}.gate.weight"));
        let e_bias = self.weights.f32_slice(&format!("{p}.gate.e_score_correction_bias"));
        let (indices, weights) = router_forward(x, gate_weight, e_bias, &d.router);

        // Routed path: fc1 down-projects to the latent bottleneck, run only the
        // top_k selected experts (per-expert NVFP4 up→relu²→down), fc2 back up.
        let latent = self.nem_matvec(&format!("{p}.fc1_latent_proj.weight"), x, hs, lat);
        let mut routed = vec![0.0f32; lat];
        for (k, &e) in indices.iter().enumerate() {
            let up = self.nem_expert_matvec(layer_idx, true, e, &latent); // [moe_inter]
            let act: Vec<f32> = up.iter().map(|&v| relu2(v)).collect();
            let down = self.nem_expert_matvec(layer_idx, false, e, &act); // [lat]
            let wk = weights[k];
            for (r, &o) in routed.iter_mut().zip(&down) {
                *r += o * wk;
            }
        }
        let moe_out = self.nem_matvec(&format!("{p}.fc2_latent_proj.weight"), &routed, lat, hs);

        // Shared expert: FP8, FULL hidden, on the original mixer input (parallel).
        let sup = self.nem_matvec(&format!("{p}.shared_experts.up_proj.weight"), x, hs, shared_inter);
        let sact: Vec<f32> = sup.iter().map(|&v| relu2(v)).collect();
        let sdown =
            self.nem_matvec(&format!("{p}.shared_experts.down_proj.weight"), &sact, shared_inter, hs);

        moe_out.iter().zip(&sdown).map(|(&m, &s)| m + s).collect()
    }

    /// WS3-style batched-CB latent-MoE (`VLLM_VULKAN_NEMOTRON_1CB`, T=1 decode
    /// only). Collapses `latent_moe_routed`'s ~4 + 2*top_k serial fenced
    /// matvec submits into ~5 fenced CBs by recording each CB's independent
    /// matvecs together (no inter-dispatch barrier — independent dispatches
    /// into distinct output regions). relu², the weighted latent accumulate,
    /// and `+shared` all stay on the **host**, byte-for-byte as
    /// `latent_moe_routed` (this increment batches submits only — lever #2's
    /// expert-batched kernel, lever #3's matvec geometry, and a GPU
    /// relu²/accumulate kernel are explicit non-goals; see the Nemotron 1-CB
    /// plan, Increment 3). Bitwise-identical to `latent_moe_routed`: same
    /// shaders, same push-constants, same per-expert `packed_off`/`sb_off`
    /// offsets, same host math order.
    ///
    /// CB grouping:
    /// - host: `router_forward` (unchanged).
    /// - CB_1: `fc1_latent_proj` (x->latent) ‖ `shared_experts.up_proj` (x->shared_inter).
    /// - host: relu² on shared up (unchanged).
    /// - CB_2: all top_k expert "up" matvecs (latent->moe_inter), independent.
    /// - host: per-expert relu² (unchanged).
    /// - CB_3: all top_k expert "down" matvecs (act->latent) ‖ `shared_experts.down_proj`.
    /// - host: weighted latent accumulate (unchanged).
    /// - CB_4 (via `nem_matvec`): `fc2_latent_proj` (routed->hs).
    /// - host: `moe_out + sdown` (unchanged).
    ///
    /// Returns `None` when the layer isn't GPU-resident (`gpu_experts`) or any
    /// projection buffer/engine allocation is unavailable, so the caller
    /// falls back to `latent_moe_mixer`.
    fn latent_moe_1cb(
        &mut self,
        layer_idx: usize,
        p: &str,
        x: &[f32],
        d: &LatentMoeDims,
    ) -> Option<Vec<f32>> {
        let hs = d.hidden_size;
        let lat = d.moe_latent_size;
        let shared_inter = d.moe_shared_expert_intermediate_size;

        if !self.gpu_experts.contains_key(&layer_idx) {
            return None;
        }

        // host: router — UNCHANGED (needs `x` on host; tiny).
        let gate_weight = self.weights.f32_slice(&format!("{p}.gate.weight"));
        let e_bias = self.weights.f32_slice(&format!("{p}.gate.e_score_correction_bias"));
        let (indices, weights) = router_forward(x, gate_weight, e_bias, &d.router);
        let top_k = indices.len();

        // CB_1: fc1_latent_proj (x -> latent) ‖ shared_experts.up_proj (x -> shared_inter).
        let fc1_meta = self.nem_meta(&format!("{p}.fc1_latent_proj.weight"))?;
        let sup_meta = self.nem_meta(&format!("{p}.shared_experts.up_proj.weight"))?;
        let eng = self.engine.as_mut()?;
        let xb = f32_slice_to_bytes(x);
        let x_in = eng.alloc_host_coherent_storage((x.len() * 4) as u64).ok()?;
        x_in.write(&xb).ok()?;
        let lat_out = eng.alloc_host_coherent_storage((lat * 4) as u64).ok()?;
        let sup_out = eng.alloc_host_coherent_storage((shared_inter * 4) as u64).ok()?;
        let x_in_p = &x_in as *const compute::Buffer;
        let lat_out_p = &lat_out as *const compute::Buffer;
        let sup_out_p = &sup_out as *const compute::Buffer;
        let cb = eng.begin_batch().ok()?;
        Self::nem_rec_mv(eng, cb, fc1_meta, x_in_p, lat_out_p, hs, lat)?;
        Self::nem_rec_mv(eng, cb, sup_meta, x_in_p, sup_out_p, hs, shared_inter)?;
        eng.submit_batch(cb).ok()?;
        let latent = read_f32_buf(&lat_out, lat);
        let sup = read_f32_buf(&sup_out, shared_inter);
        eng.return_to_pool(x_in);
        eng.return_to_pool(lat_out);
        eng.return_to_pool(sup_out);

        // host: relu² on shared up — UNCHANGED.
        let sact: Vec<f32> = sup.iter().map(|&v| relu2(v)).collect();

        // CB_2: all top_k expert "up" matvecs (latent -> moe_inter), independent.
        let up_meta = self.nem_expert_meta(layer_idx, true)?;
        let eng = self.engine.as_mut()?;
        let latb = f32_slice_to_bytes(&latent);
        let lat_in = eng.alloc_host_coherent_storage((latent.len() * 4) as u64).ok()?;
        lat_in.write(&latb).ok()?;
        let lat_in_p = &lat_in as *const compute::Buffer;
        let up_n = up_meta.3;
        let mut up_outs = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            up_outs.push(eng.alloc_host_coherent_storage((up_n * 4) as u64).ok()?);
        }
        let cb = eng.begin_batch().ok()?;
        for (k, &e) in indices.iter().enumerate() {
            let op = &up_outs[k] as *const compute::Buffer;
            Self::nem_rec_expert_mv(eng, cb, up_meta, e, lat_in_p, op)?;
        }
        eng.submit_batch(cb).ok()?;
        let ups: Vec<Vec<f32>> = up_outs.iter().map(|b| read_f32_buf(b, up_n)).collect();
        eng.return_to_pool(lat_in);
        for b in up_outs {
            eng.return_to_pool(b);
        }

        // host: per-expert relu² — UNCHANGED.
        let acts: Vec<Vec<f32>> =
            ups.iter().map(|up| up.iter().map(|&v| relu2(v)).collect()).collect();

        // CB_3: all top_k expert "down" matvecs (act -> latent) ‖ shared_experts.down_proj.
        let down_meta = self.nem_expert_meta(layer_idx, false)?;
        let sdown_meta = self.nem_meta(&format!("{p}.shared_experts.down_proj.weight"))?;
        let eng = self.engine.as_mut()?;
        let mut act_ins = Vec::with_capacity(top_k);
        for act in &acts {
            let b = eng.alloc_host_coherent_storage((act.len() * 4) as u64).ok()?;
            b.write(&f32_slice_to_bytes(act)).ok()?;
            act_ins.push(b);
        }
        let sact_in = eng.alloc_host_coherent_storage((sact.len() * 4) as u64).ok()?;
        sact_in.write(&f32_slice_to_bytes(&sact)).ok()?;
        let mut down_outs = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            down_outs.push(eng.alloc_host_coherent_storage((lat * 4) as u64).ok()?);
        }
        let sdown_out = eng.alloc_host_coherent_storage((hs * 4) as u64).ok()?;
        let sact_in_p = &sact_in as *const compute::Buffer;
        let sdown_out_p = &sdown_out as *const compute::Buffer;
        let cb = eng.begin_batch().ok()?;
        for (k, &e) in indices.iter().enumerate() {
            let ip = &act_ins[k] as *const compute::Buffer;
            let op = &down_outs[k] as *const compute::Buffer;
            Self::nem_rec_expert_mv(eng, cb, down_meta, e, ip, op)?;
        }
        Self::nem_rec_mv(eng, cb, sdown_meta, sact_in_p, sdown_out_p, shared_inter, hs)?;
        eng.submit_batch(cb).ok()?;
        let downs: Vec<Vec<f32>> = down_outs.iter().map(|b| read_f32_buf(b, lat)).collect();
        let sdown = read_f32_buf(&sdown_out, hs);
        for b in act_ins {
            eng.return_to_pool(b);
        }
        eng.return_to_pool(sact_in);
        for b in down_outs {
            eng.return_to_pool(b);
        }
        eng.return_to_pool(sdown_out);

        // host: weighted latent accumulate — UNCHANGED (same order as
        // `latent_moe_routed`: per-expert-k in-place add, float summation
        // order preserved).
        let mut routed = vec![0.0f32; lat];
        for (k, down) in downs.iter().enumerate() {
            let wk = weights[k];
            for (r, &o) in routed.iter_mut().zip(down) {
                *r += o * wk;
            }
        }

        // CB_4 (via `nem_matvec`): fc2_latent_proj (routed -> hs).
        let moe_out = self.nem_matvec(&format!("{p}.fc2_latent_proj.weight"), &routed, lat, hs);

        // host: moe_out + sdown — UNCHANGED.
        Some(moe_out.iter().zip(&sdown).map(|(&m, &s)| m + s).collect())
    }

    // ── WS3 span-resident stage (Increment 4, VLLM_VULKAN_NEMOTRON_1CB) ────

    fn nr_ptr(&self, slot: usize) -> *const compute::Buffer {
        &self.nem_res_bufs[slot] as *const compute::Buffer
    }
    fn nr_ptr_mut(&mut self, slot: usize) -> *mut compute::Buffer {
        &mut self.nem_res_bufs[slot] as *mut compute::Buffer
    }

    /// Upload the per-layer RMSNorm weights for this stage's resident span
    /// into `gpu_norm_w` ONCE, up front, so no map inserts happen during a
    /// forward (raw `*const Buffer` pointers gathered per-layer must not be
    /// invalidated by a rehash). Mirrors qwen3.6 WS3's
    /// `ensure_qwen35_norm_weights`. Does NOT include `backbone.norm_f.weight`
    /// (the final norm): unlike qwen3.6 WS3, the final-norm+lm_head tail is
    /// NOT folded into the resident path here (see
    /// `nem_forward_range_resident`'s doc comment) — it stays on
    /// `forward_pp_stage`'s existing host-norm + `nem_matvec` tail.
    fn ensure_nem_norm_weights(&mut self) -> bool {
        let hs = self.config.hidden_size;
        let names: Vec<String> = (self.pp_start..self.pp_end)
            .map(|li| format!("backbone.layers.{li}.norm.weight"))
            .collect();
        for name in names {
            if self.gpu_norm_w.contains_key(&name) {
                continue;
            }
            let data = match self.weights.tensors.get(&name) {
                Some(t) if t.data.len() >= hs => f32_slice_to_bytes(&t.data[..hs]),
                _ => return false,
            };
            let eng = match self.engine.as_mut() {
                Some(e) => e,
                None => return false,
            };
            let buf = match eng.alloc_host_coherent_storage((hs * 4) as u64) {
                Ok(b) => b,
                Err(_) => return false,
            };
            if buf.write(&data).is_err() {
                return false;
            }
            self.gpu_norm_w.insert(name, buf);
        }
        true
    }

    /// Allocate the persistent activation buffers for the span-resident stage
    /// (once). Mirrors qwen3.6 WS3's `init_q35res_bufs`.
    fn init_nem_res_bufs(&mut self) -> bool {
        if self.nem_res_ready {
            return true;
        }
        let hs = self.config.hidden_size;
        let dims = self.config.mamba_dims();
        // .max(4): Vulkan buffers can't be zero-sized.
        let f4 = |n: usize| ((n * 4).max(4)) as u64;
        // NR_X (slot 1) is the generic mixer-input scratch (hs-wide for rms_norm output),
        // but the GPU-scan path reuses it for the gated-RMSNorm output which is `intermediate`
        // wide → size it for the larger of the two so the GPU scan can't OOB-write past it.
        let sizes: [u64; NR_COUNT] =
            [f4(hs), f4(hs.max(dims.intermediate())), f4(hs), f4(dims.conv_dim()), f4(dims.intermediate()), f4(hs)];
        let eng = match self.engine.as_mut() {
            Some(e) => e,
            None => return false,
        };
        let mut bufs = Vec::with_capacity(NR_COUNT);
        for &sz in &sizes {
            match eng.alloc_host_coherent_storage(sz) {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    log::warn!("init_nem_res_bufs alloc failed: {e}");
                    return false;
                }
            }
        }
        self.nem_res_bufs = bufs;
        self.nem_res_ready = true;
        true
    }

    /// One-time readiness probe for the span-resident stage, cached in
    /// `nem_res_ok`. A false verdict is permanent for the process (the stage
    /// falls back to the Increment-1/2/3 per-layer submit path). Mirrors
    /// qwen3.6 WS3's `ensure_q35res`/`q35res_probe`.
    pub(crate) fn ensure_nem_res(&mut self) -> bool {
        if let Some(ok) = self.nem_res_ok {
            return ok;
        }
        let ok = self.nem_res_probe();
        if !ok {
            log::warn!(
                "nemotron resident span stage (VLLM_VULKAN_NEMOTRON_1CB) unavailable on \
                 layers [{}, {}); using the per-layer submit path",
                self.pp_start, self.pp_end
            );
        }
        self.nem_res_ok = Some(ok);
        ok
    }

    fn nem_res_probe(&mut self) -> bool {
        if self.engine.is_none() {
            return false;
        }
        for layer_idx in self.pp_start..self.pp_end {
            let p = format!("backbone.layers.{layer_idx}.mixer");
            match self.config.block_specs[layer_idx] {
                BlockSpec::Mamba => {
                    for w in ["in_proj", "out_proj"] {
                        if !self.gpu_weights.contains_key(&format!("{p}.{w}.weight")) {
                            return false;
                        }
                    }
                }
                BlockSpec::Attention => {
                    for w in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                        if !self.gpu_weights.contains_key(&format!("{p}.{w}.weight")) {
                            return false;
                        }
                    }
                }
                BlockSpec::Moe { .. } => {
                    if !self.gpu_experts.contains_key(&layer_idx) {
                        return false;
                    }
                    for w in [
                        "fc1_latent_proj",
                        "fc2_latent_proj",
                        "shared_experts.up_proj",
                        "shared_experts.down_proj",
                    ] {
                        if !self.gpu_weights.contains_key(&format!("{p}.{w}.weight")) {
                            return false;
                        }
                    }
                }
            }
        }
        if !self.ensure_nem_norm_weights() {
            return false;
        }
        if !self.init_nem_res_bufs() {
            return false;
        }
        // R2 (VLLM_VULKAN_NEMOTRON_GPU_SCAN): the GPU-scan branch must be
        // all-or-nothing for the WHOLE run (see `attach_gpu_mamba_scan`'s
        // doc comment) — reuse THIS probe's cached, permanent verdict
        // (`nem_res_ok`) as the GPU-scan gate too, instead of a separate
        // per-step check, so it can never flip CPU<->GPU mid-sequence. If a
        // batched n_tokens>1 nemotron prefill is ever added later, it
        // bypasses this GPU state entirely and would need CPU->GPU seeding
        // of ssm_state/conv_state before its first decode step — out of
        // scope here (T=1 decode only).
        if crate::nemotron_gpu_scan_enabled() && !self.attach_gpu_mamba_scan() {
            return false;
        }
        true
    }

    /// WS3-style span-resident forward (`VLLM_VULKAN_NEMOTRON_1CB`, T=1 decode
    /// only): runs the WHOLE `[pp_start, pp_end)` span with `hidden` GPU-
    /// resident in `NR_H`. Per layer: leading barrier, GPU RMSNorm (NR_H ->
    /// NR_X), the mixer body (mirroring Increments 1-3's per-type helpers but
    /// reading NR_X / writing NR_MIX instead of host `x`/`Vec<f32>`), then a
    /// GPU residual add (NR_H += NR_MIX) — the mid-layer host boundaries
    /// (Mamba scan, attention SDPA, MoE router + relu² + weighted latent
    /// accumulate) still read/write per-call pool-allocated scratch buffers,
    /// never NR_H. Returns ONLY the resident-span hidden `[hs]` — matching
    /// `forward_pp_range`'s exact contract — NOT logits: unlike qwen3.6 WS3
    /// (a top-level per-stage forward that owns its own final-norm+lm_head
    /// tail), Nemotron's final-norm+lm_head is applied by the CALLER,
    /// `forward_pp_stage`, on ANY stage's hidden output (it already routes
    /// through the GPU-resident `nem_matvec` when the weight is resident, so
    /// folding the tail into this function would double-apply it on the last
    /// stage). Cos-exact vs the CPU reference and vs Increments 1-3 (GPU
    /// rsqrt ulp + reordered residual accumulate), NOT bitwise —
    /// cluster-gated (argmax-exact + cos >= 0.99), same posture as qwen3.6
    /// WS3. Returns `None` on any readiness miss or mid-token GPU error, so
    /// the caller falls back to the per-layer path in `forward_pp_range`.
    pub fn nem_forward_range_resident(&mut self, hidden_in: &[f32], pos: usize) -> Option<Vec<f32>> {
        let hs = self.config.hidden_size;
        if hidden_in.len() != hs {
            return None;
        }
        if !self.ensure_nem_res() {
            return None;
        }
        let eps = self.config.norm_eps;

        // Drain the previous token's in-flight ring CBs BEFORE the host
        // writes NR_H, and reset the descriptor cursor once per token.
        // TP=2×PP: force the blocking (non-ring) submit path so each layer's
        // NR_MIX partial is GPU-complete before the loop reads it back for the
        // pairwise all-reduce. The CB-ring's async submit would race the host
        // readback. tp_size==1 keeps the shipped ring path untouched.
        let tp_shard = self.tp_size > 1;
        // TP=2×PP: by default the ring is disabled under TP (the tail CB's async
        // submit would race the reduce's host readback of NR_MIX). The TP_RING
        // lever re-enables it and instead drains the ring (a single
        // `wait_batch_pipelined`) right before each `nem_tp_reduce_mix` — strictly
        // stronger than the race, so bit-exact — letting the non-reduce CBs
        // (mamba/attention out_proj, GPU-scan mamba) pipeline as they do in PP-5.
        let ring_avail = self.engine.as_ref().map_or(false, |e| e.ring_active());
        let use_ring = ring_avail && (!tp_shard || crate::nemotron_tp_ring_enabled());
        if use_ring {
            self.engine.as_mut()?.begin_forward_ring().ok()?;
        }
        // The only host write of hidden; it stays GPU-resident all stage.
        unsafe {
            (*self.nr_ptr_mut(NR_H)).write(&f32_slice_to_bytes(hidden_in)).ok()?;
        }

        let rms_pc = rmsnorm_pc(hs, eps);
        let add_pc = ew_mul_pc(hs as u32);
        let wg_add = (hs as u32 + 255) / 256;

        // Lever 2: the TP residual add for a reduced MoE layer is DEFERRED to
        // the NEXT layer's leading CB (folded in via `nem_rec_deferred_resid`),
        // so it rides an existing submit instead of a standalone fence drain.
        // `pending_resid` carries "the previous MoE layer left a reduced partial
        // in NR_MIX/NR_PEER that this layer's first CB must add into NR_H first".
        let mut pending_resid = false;
        for layer_idx in self.pp_start..self.pp_end {
            // TP=2×PP: only MoE layers are row-parallel under this TP scope
            // (EP routed experts + shared-expert down_proj) → the moe layer fn
            // leaves NR_MIX = fc2(routed_partial)+shared_down_partial and SKIPS
            // the residual add; we exchange that partial across the TP pair (one
            // pairwise send/recv → NR_PEER) then finish the sum + residual add on
            // the GPU, folded into the next layer's CB. Mamba + attention are
            // REPLICATED (full output, in-CB residual add) → no reduce.
            let tp_reduce_layer = tp_shard
                && matches!(self.config.block_specs[layer_idx], BlockSpec::Moe { .. });
            match self.config.block_specs[layer_idx] {
                BlockSpec::Mamba => {
                    self.nem_mamba_resident_layer(layer_idx, use_ring, &rms_pc, &add_pc, wg_add, pending_resid)?;
                }
                BlockSpec::Attention => {
                    self.nem_attn_resident_layer(layer_idx, pos, use_ring, &rms_pc, &add_pc, wg_add, pending_resid)?;
                }
                BlockSpec::Moe { num_experts_per_tok, moe_intermediate_size } => {
                    self.nem_moe_resident_layer(
                        layer_idx,
                        num_experts_per_tok,
                        moe_intermediate_size,
                        use_ring,
                        &rms_pc,
                        &add_pc,
                        wg_add,
                        pending_resid,
                    )?;
                }
            }
            // The layer above consumed any pending deferred add into its CB.
            pending_resid = false;
            if tp_reduce_layer {
                // TP_RING lever: the MoE tail CB was ring-submitted WITHOUT a
                // wait, so drain it here so its NR_MIX partial is host-visible
                // before the reduce reads it back. (Flag-OFF path already blocked
                // in `submit_batch`, so this is a no-op there.)
                if use_ring {
                    self.engine.as_mut()?.wait_batch_pipelined().ok()?;
                }
                // Pure comm: exchange the partial, stage the peer's into NR_PEER.
                self.nem_tp_reduce_mix()?;
                // Defer NR_MIX += NR_PEER; NR_H += NR_MIX to the next layer's CB.
                pending_resid = true;
            }
        }
        // End-of-span fallback: the LAST layer was a reduced MoE layer, so there
        // is no next CB to fold into — flush the add as a standalone CB.
        if pending_resid {
            self.nem_record_residual_add(&add_pc, wg_add, use_ring)?;
        }

        // Drain the last layer's (possibly non-waited, ring-submitted) CB and
        // ship the resident hidden back to the caller (`forward_pp_stage`
        // applies the final-norm+lm_head tail itself when this is the last
        // stage, else forwards it to the next PP stage).
        if use_ring {
            self.engine.as_mut()?.wait_batch_pipelined().ok()?;
        }
        let hidden = read_f32_buf(unsafe { &*self.nr_ptr(NR_H) }, hs);
        Some(hidden)
    }

    /// Resident-span Mamba layer: CB_A (rms_norm NR_H->NR_X, in_proj matvec),
    /// fenced/waited (the SSD scan needs the proj readback on host); host scan
    /// (unchanged); CB_B (out_proj matvec -> NR_MIX, GPU residual add NR_H +=
    /// NR_MIX) — ring-submitted WITHOUT a wait (its pool input buffer is
    /// returned via the ring's deferred-return list, not immediately, since
    /// the GPU may still be reading it when the next layer starts recording).
    ///
    /// R2 (`VLLM_VULKAN_NEMOTRON_GPU_SCAN`): when the GPU-scan branch below is
    /// taken, this whole 2-CB shape collapses to ONE CB (see
    /// `nem_mamba_resident_layer_gpu_scan`) — the host round-trip disappears
    /// because the conv1d/SSD/gated-norm math moves onto the GPU, reading and
    /// writing THIS layer's `gpu_mamba[layer_idx]` state directly instead of
    /// the host `Mamba2State` in `layer_state`.
    fn nem_mamba_resident_layer(
        &mut self,
        layer_idx: usize,
        use_ring: bool,
        rms_pc: &[u8],
        add_pc: &[u8],
        wg_add: u32,
        pending_resid: bool,
    ) -> Option<()> {
        // R2: the GPU-scan branch is gated on `ensure_nem_res`'s cached,
        // permanent readiness verdict (via `attach_gpu_mamba_scan`, called
        // from `nem_res_probe`) — NOT re-checked per step — so it can never
        // fall back to the CPU scan mid-sequence and desync the GPU
        // `ssm_state`/`conv_state` (the state of record once this branch is
        // live) from the host `Mamba2State` (which the flag-OFF path below
        // keeps advancing instead). `gpu_mamba.contains_key` is safe to read
        // here as a per-layer check ONLY because `attach_gpu_mamba_scan`
        // guarantees it is all-or-nothing across every resident Mamba layer
        // in this stage, and is populated once, before any forward call.
        if crate::nemotron_gpu_scan_enabled() && self.gpu_mamba.contains_key(&layer_idx) {
            return self.nem_mamba_resident_layer_gpu_scan(layer_idx, use_ring, rms_pc, add_pc, wg_add, pending_resid);
        }

        let dims = self.config.mamba_dims();
        let si = self.state_idx(layer_idx);
        let hs = dims.hidden_size;
        let inter = dims.intermediate();
        let in_proj_out = dims.in_proj_out();
        let p = format!("backbone.layers.{layer_idx}.mixer");
        let in_meta = self.nem_meta(&format!("{p}.in_proj.weight"))?;
        let out_meta = self.nem_meta(&format!("{p}.out_proj.weight"))?;
        let norm_w =
            &self.gpu_norm_w[&format!("backbone.layers.{layer_idx}.norm.weight")] as *const compute::Buffer;

        // CB_A: rms_norm(NR_H -> NR_X) -> in_proj(NR_X -> proj). Fenced/waited:
        // the SSD scan needs `proj` on the host.
        let hp = self.nr_ptr(NR_H);
        let xp = self.nr_ptr(NR_X);
        let mixp = self.nr_ptr(NR_MIX);
        let peerp = self.nr_ptr(NR_PEER);
        let eng = self.engine.as_mut()?;
        let proj_buf = eng.alloc_host_coherent_storage((in_proj_out * 4) as u64).ok()?;
        let proj_p = &proj_buf as *const compute::Buffer;
        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        eng.record_barrier_to(cb);
        unsafe {
            // Lever 2: fold the previous MoE layer's deferred TP residual add in.
            if pending_resid {
                Self::nem_rec_deferred_resid(eng, cb, hp, mixp, peerp, add_pc, wg_add)?;
            }
            eng.record_to(cb, "rms_norm_f32_mul", &[&*hp, &*norm_w, &*xp], rms_pc, (1, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            Self::nem_rec_mv(eng, cb, in_meta, xp, proj_p, hs, in_proj_out)?;
        }
        let t_fence = std::time::Instant::now();
        if use_ring {
            eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
            eng.wait_batch_pipelined().ok()?;
        } else {
            eng.submit_batch(cb).ok()?;
        }
        prof_add("nem_cb_fence", t_fence);
        let proj = read_f32_buf(&proj_buf, in_proj_out);
        eng.return_to_pool(proj_buf);

        // host: conv1d + SSD recurrence — UNCHANGED.
        let bufs = self.load_mamba_weights(layer_idx);
        let t_scan = std::time::Instant::now();
        let scan = {
            let weights = bufs.borrow();
            let st = match &mut self.layer_state[si] {
                LayerState::Mamba(s) => s,
                _ => unreachable!("mamba layer has a Mamba2State"),
            };
            mamba2_scan_only(&proj, &weights, st, &dims) // [1, inter]
        };
        prof_add("nem_mamba_scan", t_scan);

        // CB_B: out_proj(scan -> NR_MIX) -> residual add (NR_H += NR_MIX).
        let hp = self.nr_ptr(NR_H);
        let mixp = self.nr_ptr(NR_MIX);
        let eng = self.engine.as_mut()?;
        let scan_buf = eng.alloc_host_coherent_storage((scan.len() * 4) as u64).ok()?;
        scan_buf.write(&f32_slice_to_bytes(&scan)).ok()?;
        let scan_p = &scan_buf as *const compute::Buffer;
        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        eng.record_barrier_to(cb);
        unsafe {
            Self::nem_rec_mv(eng, cb, out_meta, scan_p, mixp, inter, hs)?;
            eng.record_barrier_to(cb);
            // Mamba is REPLICATED under TP (output is the full mixer result, not
            // a partial), so the residual add stays in-CB — no all-reduce.
            eng.record_to(cb, "add_f32_f32_f32", &[&*hp, &*mixp, &*hp, &*hp], add_pc, (wg_add, 1, 1)).ok()?;
        }
        if use_ring {
            eng.submit_batch_pipelined(cb, vec![scan_buf]).ok()?;
        } else {
            eng.submit_batch(cb).ok()?;
            eng.return_to_pool(scan_buf);
        }
        Some(())
    }

    /// R2 GPU Mamba2 SSD-scan resident layer (`VLLM_VULKAN_NEMOTRON_GPU_SCAN`):
    /// the WHOLE mixer — rms_norm, in_proj, conv1d+SiLU, per-head SSD
    /// recurrence+gate, gated RMSNorm, out_proj, residual add — records into
    /// ONE command buffer with NO host readback at all (unlike
    /// `nem_mamba_resident_layer`'s CB_A/CB_B split, whose CB_A is a
    /// mandatory fence because the host CPU scan needs `proj`). The scan
    /// dispatches read/write `self.gpu_mamba[&layer_idx]`'s
    /// `ssm_state`/`conv_state` directly — those buffers ARE this layer's
    /// state of record while this branch is live; the host `Mamba2State` in
    /// `layer_state` is simply never touched (and is stale from here on) as
    /// long as `attach_gpu_mamba_scan`'s all-or-nothing gate holds for the
    /// whole process — see `nem_mamba_resident_layer`'s branch comment.
    /// Mirrors `nem_moe_resident_layer_tail`'s collapsed tail CB +
    /// `submit_batch_pipelined` usage.
    ///
    /// NOTE: this is T=1 decode only. If a batched `n_tokens>1` Nemotron
    /// prefill is ever added on top of this GPU-scan path, it would bypass
    /// this GPU state entirely (`mamba_mixer_prefill`'s sequential scan
    /// assumes host state) and would need explicit CPU->GPU seeding of
    /// ssm_state/conv_state before its first decode step — out of scope here.
    fn nem_mamba_resident_layer_gpu_scan(
        &mut self,
        layer_idx: usize,
        use_ring: bool,
        rms_pc: &[u8],
        add_pc: &[u8],
        wg_add: u32,
        pending_resid: bool,
    ) -> Option<()> {
        let dims = self.config.mamba_dims();
        let hs = dims.hidden_size;
        let inter = dims.intermediate();
        let conv_dim = dims.conv_dim();
        let in_proj_out = dims.in_proj_out();
        // NR_X holds the `inter`-wide gated-RMSNorm output here; fail LOUD (release too —
        // debug_assert is compiled out) if it was sized too small, rather than OOB-faulting the GPU.
        assert!(
            self.nem_res_bufs[NR_X].size as usize >= inter * 4,
            "NR_X too small for GPU-scan gated output: {} < {}",
            self.nem_res_bufs[NR_X].size,
            inter * 4
        );
        let nh = dims.num_heads;
        let hd = dims.head_dim;
        let ss = dims.ssm_state_size;
        let n_groups = dims.n_groups;
        let heads_per_group = nh / n_groups;
        let ng_ss = n_groups * ss;
        let group_size = dims.norm_group_size();

        let p = format!("backbone.layers.{layer_idx}.mixer");
        let in_meta = self.nem_meta(&format!("{p}.in_proj.weight"))?;
        let out_meta = self.nem_meta(&format!("{p}.out_proj.weight"))?;
        let norm_w =
            &self.gpu_norm_w[&format!("backbone.layers.{layer_idx}.norm.weight")] as *const compute::Buffer;
        let g = self.gpu_mamba.get(&layer_idx)?;
        let conv_w_p = &g.conv_w as *const compute::Buffer;
        let conv_bias_p = &g.conv_bias as *const compute::Buffer;
        let conv_state_p = &g.conv_state as *const compute::Buffer;
        let params_p = &g.params as *const compute::Buffer;
        let ssm_state_p = &g.ssm_state as *const compute::Buffer;
        let gated_norm_w_p = &g.norm_w as *const compute::Buffer;

        let conv_pc = nem_ssm_conv_pc(conv_dim, dims.conv_kernel, inter);
        let scan_pc =
            nem_ssd_scan_pc(nh, hd, ss, heads_per_group, inter, conv_dim, ng_ss, dims.time_step_min);
        let norm_pc = nem_gated_rmsnorm_pc(group_size, dims.eps, n_groups);

        let hp = self.nr_ptr(NR_H);
        let xp = self.nr_ptr(NR_X);
        let mixp = self.nr_ptr(NR_MIX);
        let peerp = self.nr_ptr(NR_PEER);
        let convp = self.nr_ptr(NR_CONV);
        let gatedp = self.nr_ptr(NR_GATED);

        let eng = self.engine.as_mut()?;
        let proj_buf = eng.alloc_host_coherent_storage((in_proj_out * 4) as u64).ok()?;
        let proj_p = &proj_buf as *const compute::Buffer;

        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        // L1: per-dispatch timestamps to split the mamba CB (the old single
        // nem_ts_mamba bracket includes the in_proj/out_proj matvecs which are
        // ~44x the scan MACs). 8 marks → 7 stage intervals; marks are bottom-of-
        // pipe after each barrier so each dispatch is cleanly bracketed.
        let ts_on = prof_on() && !use_ring && eng.ensure_ts_pool(8);
        if ts_on {
            eng.ts_cmd_reset(cb, 0, 8);
            eng.ts_cmd_mark(cb, 0, true);
        }
        eng.record_barrier_to(cb);
        unsafe {
            // Lever 2: fold the previous MoE layer's deferred TP residual add in
            // (may nudge the nem_ts_m_rms bracket — debug-only, post-MoE layer).
            if pending_resid {
                Self::nem_rec_deferred_resid(eng, cb, hp, mixp, peerp, add_pc, wg_add)?;
            }
            eng.record_to(cb, "rms_norm_f32_mul", &[&*hp, &*norm_w, &*xp], rms_pc, (1, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            if ts_on { eng.ts_cmd_mark(cb, 1, false); }
            Self::nem_rec_mv(eng, cb, in_meta, xp, proj_p, hs, in_proj_out)?;
            eng.record_barrier_to(cb);
            if ts_on { eng.ts_cmd_mark(cb, 2, false); }
            eng.record_to(
                cb,
                "nemotron_ssm_conv_step",
                &[&*conv_w_p, &*proj_p, &*conv_bias_p, &*conv_state_p, &*convp],
                &conv_pc,
                ((conv_dim as u32 + 255) / 256, 1, 1),
            )
            .ok()?;
            eng.record_barrier_to(cb);
            if ts_on { eng.ts_cmd_mark(cb, 3, false); }
            eng.record_to(
                cb,
                "nemotron_ssd_scan",
                &[&*proj_p, &*convp, &*params_p, &*ssm_state_p, &*gatedp],
                &scan_pc,
                (nh as u32, 1, 1),
            )
            .ok()?;
            eng.record_barrier_to(cb);
            if ts_on { eng.ts_cmd_mark(cb, 4, false); }
            eng.record_to(
                cb,
                "nemotron_gated_rmsnorm",
                &[&*gatedp, &*gated_norm_w_p, &*xp],
                &norm_pc,
                (n_groups as u32, 1, 1),
            )
            .ok()?;
            eng.record_barrier_to(cb);
            if ts_on { eng.ts_cmd_mark(cb, 5, false); }
            Self::nem_rec_mv(eng, cb, out_meta, xp, mixp, inter, hs)?;
            eng.record_barrier_to(cb);
            if ts_on { eng.ts_cmd_mark(cb, 6, false); }
            // Mamba is REPLICATED under TP — residual add stays in-CB, no reduce.
            eng.record_to(cb, "add_f32_f32_f32", &[&*hp, &*mixp, &*hp, &*hp], add_pc, (wg_add, 1, 1)).ok()?;
        }
        if ts_on {
            eng.ts_cmd_mark(cb, 7, false);
        }
        if use_ring {
            eng.submit_batch_pipelined(cb, vec![proj_buf]).ok()?;
        } else {
            let t_fence = std::time::Instant::now();
            eng.submit_batch(cb).ok()?;
            prof_add("nem_cb_fence", t_fence);
            if ts_on {
                if let Ok(v) = eng.ts_read_ns(0, 8) {
                    let d = |a: usize, b: usize| (v[b] - v[a]).max(0.0) as u128;
                    prof_add_ns("nem_ts_mamba", d(0, 7)); // total (continuity)
                    prof_add_ns("nem_ts_m_rms", d(0, 1));
                    prof_add_ns("nem_ts_m_inproj", d(1, 2));
                    prof_add_ns("nem_ts_m_conv", d(2, 3));
                    prof_add_ns("nem_ts_m_scan", d(3, 4));
                    prof_add_ns("nem_ts_m_norm", d(4, 5));
                    prof_add_ns("nem_ts_m_outproj", d(5, 6));
                    prof_add_ns("nem_ts_m_add", d(6, 7));
                }
            }
            eng.return_to_pool(proj_buf);
        }
        Some(())
    }

    /// Resident-span NoPE attention layer: CB_A (rms_norm NR_H->NR_X, q/k/v
    /// matvecs), fenced/waited (host SDPA needs q/k/v); host KV-append+SDPA
    /// (unchanged); CB_B (upload attn -> o_proj -> NR_MIX, GPU residual add).
    fn nem_attn_resident_layer(
        &mut self,
        layer_idx: usize,
        pos: usize,
        use_ring: bool,
        rms_pc: &[u8],
        add_pc: &[u8],
        wg_add: u32,
        pending_resid: bool,
    ) -> Option<()> {
        let cfg = &self.config;
        let hs = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let p = format!("backbone.layers.{layer_idx}.mixer");
        let si = self.state_idx(layer_idx);
        let _ = pos; // NoPE: no rotary table, position only advances the KV cache (handled by `cache.append`).

        let q_meta = self.nem_meta(&format!("{p}.q_proj.weight"))?;
        let k_meta = self.nem_meta(&format!("{p}.k_proj.weight"))?;
        let v_meta = self.nem_meta(&format!("{p}.v_proj.weight"))?;
        let o_meta = self.nem_meta(&format!("{p}.o_proj.weight"))?;
        let norm_w =
            &self.gpu_norm_w[&format!("backbone.layers.{layer_idx}.norm.weight")] as *const compute::Buffer;

        // CB_A: rms_norm(NR_H -> NR_X) -> q/k/v matvecs (independent, no
        // inter-dispatch barrier). Fenced/waited: host SDPA needs q/k/v.
        let hp = self.nr_ptr(NR_H);
        let xp = self.nr_ptr(NR_X);
        let mixp = self.nr_ptr(NR_MIX);
        let peerp = self.nr_ptr(NR_PEER);
        let eng = self.engine.as_mut()?;
        let q_out = eng.alloc_host_coherent_storage((q_dim * 4) as u64).ok()?;
        let k_out = eng.alloc_host_coherent_storage((kv_dim * 4) as u64).ok()?;
        let v_out = eng.alloc_host_coherent_storage((kv_dim * 4) as u64).ok()?;
        let q_out_p = &q_out as *const compute::Buffer;
        let k_out_p = &k_out as *const compute::Buffer;
        let v_out_p = &v_out as *const compute::Buffer;
        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        eng.record_barrier_to(cb);
        unsafe {
            // Lever 2: fold the previous MoE layer's deferred TP residual add in.
            if pending_resid {
                Self::nem_rec_deferred_resid(eng, cb, hp, mixp, peerp, add_pc, wg_add)?;
            }
            eng.record_to(cb, "rms_norm_f32_mul", &[&*hp, &*norm_w, &*xp], rms_pc, (1, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            Self::nem_rec_mv(eng, cb, q_meta, xp, q_out_p, hs, q_dim)?;
            Self::nem_rec_mv(eng, cb, k_meta, xp, k_out_p, hs, kv_dim)?;
            Self::nem_rec_mv(eng, cb, v_meta, xp, v_out_p, hs, kv_dim)?;
        }
        let t_fence = std::time::Instant::now();
        if use_ring {
            eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
            eng.wait_batch_pipelined().ok()?;
        } else {
            eng.submit_batch(cb).ok()?;
        }
        prof_add("nem_cb_fence", t_fence);
        let q = read_f32_buf(&q_out, q_dim);
        let k = read_f32_buf(&k_out, kv_dim);
        let v = read_f32_buf(&v_out, kv_dim);
        eng.return_to_pool(q_out);
        eng.return_to_pool(k_out);
        eng.return_to_pool(v_out);

        // host: KV-cache append + SDPA — UNCHANGED.
        let t_sdpa = std::time::Instant::now();
        let attn = {
            let cache = match &mut self.layer_state[si] {
                LayerState::Attention(c) => c,
                _ => unreachable!("attention layer has a KV cache"),
            };
            cache.append(&k, &v);
            cpu_sdpa(&q, cache.k_up_to_now(), cache.v_up_to_now(), nq, nkv, hd, cache.seq_len, scale, None)
        };
        prof_add("nem_sdpa", t_sdpa);

        // CB_B: upload attn -> o_proj(attn -> NR_MIX) -> residual add.
        let hp = self.nr_ptr(NR_H);
        let mixp = self.nr_ptr(NR_MIX);
        let eng = self.engine.as_mut()?;
        let attn_buf = eng.alloc_host_coherent_storage((attn.len() * 4) as u64).ok()?;
        attn_buf.write(&f32_slice_to_bytes(&attn)).ok()?;
        let attn_p = &attn_buf as *const compute::Buffer;
        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        eng.record_barrier_to(cb);
        unsafe {
            Self::nem_rec_mv(eng, cb, o_meta, attn_p, mixp, q_dim, hs)?;
            eng.record_barrier_to(cb);
            // Attention is REPLICATED under TP (q/k/v/o not head-sharded here) —
            // full output, residual add stays in-CB, no all-reduce.
            eng.record_to(cb, "add_f32_f32_f32", &[&*hp, &*mixp, &*hp, &*hp], add_pc, (wg_add, 1, 1)).ok()?;
        }
        if use_ring {
            eng.submit_batch_pipelined(cb, vec![attn_buf]).ok()?;
        } else {
            eng.submit_batch(cb).ok()?;
            eng.return_to_pool(attn_buf);
        }
        Some(())
    }

    /// Resident-span latent-MoE layer. Dispatches to the R1b MoE-tail-collapse
    /// arm (`VLLM_VULKAN_NEMOTRON_MOE_TAIL`, `nem_moe_resident_layer_tail`,
    /// 1 unwaited CB for the whole MoE tail) when the flag is set, else the
    /// existing 4-CB per-layer path (`nem_moe_resident_layer_legacy`) — the
    /// correctness reference, byte-for-byte unchanged.
    fn nem_moe_resident_layer(
        &mut self,
        layer_idx: usize,
        top_k_cfg: usize,
        moe_inter: usize,
        use_ring: bool,
        rms_pc: &[u8],
        add_pc: &[u8],
        wg_add: u32,
        pending_resid: bool,
    ) -> Option<()> {
        if crate::nemotron_moe_tail_enabled() {
            return self.nem_moe_resident_layer_tail(
                layer_idx, top_k_cfg, moe_inter, use_ring, rms_pc, add_pc, wg_add, pending_resid,
            );
        }
        self.nem_moe_resident_layer_legacy(
            layer_idx, top_k_cfg, moe_inter, use_ring, rms_pc, add_pc, wg_add, pending_resid,
        )
    }

    /// Resident-span latent-MoE layer, the 4-CB (3 waited fences) path — the
    /// correctness reference for R1b. Same 4-CB grouping as `latent_moe_1cb`
    /// (Increment 3), but CB_1 reads NR_X directly (no host upload of `x`) and
    /// CB_1's `fc1_latent_proj` output (`latent`) is consumed DIRECTLY by CB_2
    /// from the GPU pool buffer — no host round trip for `latent` (it is never
    /// needed on the host; only the router needs `x`/NR_X). relu², the
    /// weighted latent accumulate, and the router stay on the host, unchanged
    /// (lever #2/#4 non-goals). CB_4 (fc2 + both adds) is the layer's only
    /// non-waited ring submission — no further host read happens this layer.
    fn nem_moe_resident_layer_legacy(
        &mut self,
        layer_idx: usize,
        top_k_cfg: usize,
        moe_inter: usize,
        use_ring: bool,
        rms_pc: &[u8],
        add_pc: &[u8],
        wg_add: u32,
        pending_resid: bool,
    ) -> Option<()> {
        let _ = top_k_cfg; // actual top_k comes from `router_forward`'s `indices.len()`.
        let tp_shard = self.tp_size > 1; // TP=2×PP EP: owned-expert filter + skip in-CB residual add
        let (tp_rank, tp_size) = (self.tp_rank, self.tp_size);
        let d = self.config.latent_moe_dims(top_k_cfg, moe_inter);
        let hs = d.hidden_size;
        let lat = d.moe_latent_size;
        // TP=2×PP: shared-expert up is col-sharded / down row-sharded, so this
        // rank's sup/sact/sdown work over shared_inter/tp_size channels (its
        // sdown output is a hidden-wide PARTIAL, summed by the NR_MIX reduce).
        let shared_inter = if tp_shard {
            d.moe_shared_expert_intermediate_size / tp_size
        } else {
            d.moe_shared_expert_intermediate_size
        };
        let p = format!("backbone.layers.{layer_idx}.mixer");
        let norm_w =
            &self.gpu_norm_w[&format!("backbone.layers.{layer_idx}.norm.weight")] as *const compute::Buffer;

        let fc1_meta = self.nem_meta(&format!("{p}.fc1_latent_proj.weight"))?;
        let sup_meta = self.nem_meta(&format!("{p}.shared_experts.up_proj.weight"))?;
        let fc2_meta = self.nem_meta(&format!("{p}.fc2_latent_proj.weight"))?;
        let sdown_meta = self.nem_meta(&format!("{p}.shared_experts.down_proj.weight"))?;

        // CB_1: rms_norm(NR_H -> NR_X) -> fc1_latent_proj(NR_X -> latent) ‖
        // shared_experts.up_proj(NR_X -> sup). Fenced/waited: the host router
        // needs NR_X, and relu² needs `sup`. `latent` is NOT read back here —
        // it stays in the pool buffer for CB_2 to consume directly.
        let hp = self.nr_ptr(NR_H);
        let xp = self.nr_ptr(NR_X);
        let mixp = self.nr_ptr(NR_MIX);
        let peerp = self.nr_ptr(NR_PEER);
        let eng = self.engine.as_mut()?;
        let lat_buf = eng.alloc_host_coherent_storage((lat * 4) as u64).ok()?;
        let sup_buf = eng.alloc_host_coherent_storage((shared_inter * 4) as u64).ok()?;
        let lat_p = &lat_buf as *const compute::Buffer;
        let sup_p = &sup_buf as *const compute::Buffer;
        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        let ts_on = prof_on() && !use_ring && eng.ensure_ts_pool(2);
        if ts_on {
            eng.ts_cmd_reset(cb, 0, 2);
            eng.ts_cmd_mark(cb, 0, true);
        }
        eng.record_barrier_to(cb);
        unsafe {
            // Lever 2: fold the previous MoE layer's deferred TP residual add in.
            if pending_resid {
                Self::nem_rec_deferred_resid(eng, cb, hp, mixp, peerp, add_pc, wg_add)?;
            }
            eng.record_to(cb, "rms_norm_f32_mul", &[&*hp, &*norm_w, &*xp], rms_pc, (1, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            Self::nem_rec_mv(eng, cb, fc1_meta, xp, lat_p, hs, lat)?;
            Self::nem_rec_mv(eng, cb, sup_meta, xp, sup_p, hs, shared_inter)?;
        }
        if ts_on { eng.ts_cmd_mark(cb, 1, false); }
        let t_fence = std::time::Instant::now();
        if use_ring {
            eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
            eng.wait_batch_pipelined().ok()?;
        } else {
            eng.submit_batch(cb).ok()?;
        }
        prof_add("nem_cb_fence", t_fence);
        if ts_on {
            if let Ok(v) = eng.ts_read_ns(0, 2) {
                prof_add_ns("nem_ts_cb1", (v[1] - v[0]).max(0.0) as u128);
            }
        }
        let x_host = read_f32_buf(unsafe { &*xp }, hs);
        let sup = read_f32_buf(&sup_buf, shared_inter);
        eng.return_to_pool(sup_buf);

        // host: router (needs x) + relu² on shared up — UNCHANGED.
        let t_gate = std::time::Instant::now();
        let gate_weight = self.weights.f32_slice(&format!("{p}.gate.weight"));
        let e_bias = self.weights.f32_slice(&format!("{p}.gate.e_score_correction_bias"));
        prof_add("nem_gate_copy", t_gate);
        let (indices, weights) = router_forward(&x_host, gate_weight, e_bias, &d.router);
        // TP=2×PP EP owned-expert filter (see nem_moe_resident_layer_tail for the
        // full rationale): same replicated top-k, run only owned experts, remap
        // to local id, keep the gate weight, partial routed → NR_MIX all-reduce.
        let (indices, weights) = if tp_shard {
            let (owned_lo, owned_cnt) =
                crate::nemotron_tp::expert_owned_range(d.router.n_routed_experts, tp_rank, tp_size);
            let mut idx = Vec::new();
            let mut w = Vec::new();
            for (k, &e) in indices.iter().enumerate() {
                if e >= owned_lo && e < owned_lo + owned_cnt {
                    idx.push(e - owned_lo);
                    w.push(weights[k]);
                }
            }
            (idx, w)
        } else {
            (indices, weights)
        };
        let top_k = indices.len();
        let sact: Vec<f32> = sup.iter().map(|&v| relu2(v)).collect();

        // CB_2: top_k expert "up" matvecs (lat_buf, still GPU-resident from
        // CB_1, -> moe_inter), independent. Fenced/waited: relu² needs it.
        let up_meta = self.nem_expert_meta(layer_idx, true)?;
        let eng = self.engine.as_mut()?;
        let up_n = up_meta.3;
        let mut up_outs = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            up_outs.push(eng.alloc_host_coherent_storage((up_n * 4) as u64).ok()?);
        }
        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        let ts_on = prof_on() && !use_ring && eng.ensure_ts_pool(2);
        if ts_on {
            eng.ts_cmd_reset(cb, 0, 2);
            eng.ts_cmd_mark(cb, 0, true);
        }
        eng.record_barrier_to(cb);
        for (k, &e) in indices.iter().enumerate() {
            let op = &up_outs[k] as *const compute::Buffer;
            Self::nem_rec_expert_mv(eng, cb, up_meta, e, lat_p, op)?;
        }
        if ts_on { eng.ts_cmd_mark(cb, 1, false); }
        let t_fence = std::time::Instant::now();
        if use_ring {
            eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
            eng.wait_batch_pipelined().ok()?;
        } else {
            eng.submit_batch(cb).ok()?;
        }
        prof_add("nem_cb_fence", t_fence);
        if ts_on {
            if let Ok(v) = eng.ts_read_ns(0, 2) {
                prof_add_ns("nem_ts_cb2", (v[1] - v[0]).max(0.0) as u128);
            }
        }
        let ups: Vec<Vec<f32>> = up_outs.iter().map(|b| read_f32_buf(b, up_n)).collect();
        eng.return_to_pool(lat_buf);
        for b in up_outs {
            eng.return_to_pool(b);
        }

        // host: per-expert relu² — UNCHANGED.
        let acts: Vec<Vec<f32>> = ups.iter().map(|up| up.iter().map(|&v| relu2(v)).collect()).collect();

        // CB_3: upload acts -> top_k expert "down" matvecs (-> latent) ‖
        // shared_experts.down_proj(sact -> hs). Fenced/waited: the host
        // weighted accumulate needs the downs.
        let down_meta = self.nem_expert_meta(layer_idx, false)?;
        let eng = self.engine.as_mut()?;
        let mut act_ins = Vec::with_capacity(top_k);
        for act in &acts {
            let b = eng.alloc_host_coherent_storage((act.len() * 4) as u64).ok()?;
            b.write(&f32_slice_to_bytes(act)).ok()?;
            act_ins.push(b);
        }
        let sact_in = eng.alloc_host_coherent_storage((sact.len() * 4) as u64).ok()?;
        sact_in.write(&f32_slice_to_bytes(&sact)).ok()?;
        let mut down_outs = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            down_outs.push(eng.alloc_host_coherent_storage((lat * 4) as u64).ok()?);
        }
        let sdown_out = eng.alloc_host_coherent_storage((hs * 4) as u64).ok()?;
        let sact_in_p = &sact_in as *const compute::Buffer;
        let sdown_out_p = &sdown_out as *const compute::Buffer;
        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        let ts_on = prof_on() && !use_ring && eng.ensure_ts_pool(2);
        if ts_on {
            eng.ts_cmd_reset(cb, 0, 2);
            eng.ts_cmd_mark(cb, 0, true);
        }
        eng.record_barrier_to(cb);
        for (k, &e) in indices.iter().enumerate() {
            let ip = &act_ins[k] as *const compute::Buffer;
            let op = &down_outs[k] as *const compute::Buffer;
            Self::nem_rec_expert_mv(eng, cb, down_meta, e, ip, op)?;
        }
        Self::nem_rec_mv(eng, cb, sdown_meta, sact_in_p, sdown_out_p, shared_inter, hs)?;
        if ts_on { eng.ts_cmd_mark(cb, 1, false); }
        let t_fence = std::time::Instant::now();
        if use_ring {
            eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
            eng.wait_batch_pipelined().ok()?;
        } else {
            eng.submit_batch(cb).ok()?;
        }
        prof_add("nem_cb_fence", t_fence);
        if ts_on {
            if let Ok(v) = eng.ts_read_ns(0, 2) {
                prof_add_ns("nem_ts_cb3", (v[1] - v[0]).max(0.0) as u128);
            }
        }
        // `sdown_out` is NOT read back — CB_4 consumes it directly from the
        // GPU pool buffer (`sdown_p`), unlike Increment 3 which needed it on
        // the host for the final `moe_out + sdown` add.
        let downs: Vec<Vec<f32>> = down_outs.iter().map(|b| read_f32_buf(b, lat)).collect();
        for b in act_ins {
            eng.return_to_pool(b);
        }
        eng.return_to_pool(sact_in);
        for b in down_outs {
            eng.return_to_pool(b);
        }

        // host: weighted latent accumulate — UNCHANGED (same order as
        // `latent_moe_routed`/`latent_moe_1cb`: per-expert-k in-place add).
        let mut routed = vec![0.0f32; lat];
        for (k, down) in downs.iter().enumerate() {
            let wk = weights[k];
            for (r, &o) in routed.iter_mut().zip(down) {
                *r += o * wk;
            }
        }

        // CB_4: upload routed -> fc2_latent_proj(routed -> moe_out) ->
        // (moe_out + sdown -> NR_MIX) -> residual add (NR_H += NR_MIX). No
        // further host read happens this layer, so this is the layer's only
        // non-waited ring submission; its ephemeral input buffers (routed,
        // moe_out, sdown) are deferred-returned via the ring's `deferred`
        // list instead of being returned to the pool immediately.
        let hp = self.nr_ptr(NR_H);
        let mixp = self.nr_ptr(NR_MIX);
        let eng = self.engine.as_mut()?;
        let routed_buf = eng.alloc_host_coherent_storage((routed.len() * 4) as u64).ok()?;
        routed_buf.write(&f32_slice_to_bytes(&routed)).ok()?;
        let routed_p = &routed_buf as *const compute::Buffer;
        let moe_out_buf = eng.alloc_host_coherent_storage((hs * 4) as u64).ok()?;
        let moe_out_p = &moe_out_buf as *const compute::Buffer;
        let sdown_p = &sdown_out as *const compute::Buffer;
        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        eng.record_barrier_to(cb);
        unsafe {
            Self::nem_rec_mv(eng, cb, fc2_meta, routed_p, moe_out_p, lat, hs)?;
            eng.record_barrier_to(cb);
            eng.record_to(cb, "add_f32_f32_f32", &[&*moe_out_p, &*sdown_p, &*mixp, &*mixp], add_pc, (wg_add, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            if !tp_shard { // TP=2×PP: residual add deferred to the loop's post-reduce step
            eng.record_to(cb, "add_f32_f32_f32", &[&*hp, &*mixp, &*hp, &*hp], add_pc, (wg_add, 1, 1)).ok()?;
            }
        }
        if use_ring {
            eng.submit_batch_pipelined(cb, vec![routed_buf, moe_out_buf, sdown_out]).ok()?;
        } else {
            eng.submit_batch(cb).ok()?;
            eng.return_to_pool(routed_buf);
            eng.return_to_pool(moe_out_buf);
            eng.return_to_pool(sdown_out);
        }
        Some(())
    }

    /// R1b MoE-tail collapse (`VLLM_VULKAN_NEMOTRON_MOE_TAIL`): same CB_1 as
    /// `nem_moe_resident_layer_legacy`, but relu² and the top_k weighted
    /// accumulate move onto the GPU (`relu2_f32` + `nemotron_moe_accum`), so
    /// the entire MoE tail after the host router (up matvecs, both relu²s,
    /// down matvecs + shared down, the routed accumulate, fc2, and both
    /// residual adds) records into ONE command buffer instead of 3 separately
    /// fenced CBs (CB_2/CB_3/CB_4). On the ring this single CB is the layer's
    /// only non-waited submission — same as CB_4 in the legacy path — so this
    /// collapses the layer's fence count 3 -> 1 (CB_1 stays waited: the host
    /// router needs `x_host`). Reference: qwen3.6 WS3's one-CB MoE tail
    /// (`qwen35_forward.rs`, the `q35r_cbb` block).
    fn nem_moe_resident_layer_tail(
        &mut self,
        layer_idx: usize,
        top_k_cfg: usize,
        moe_inter: usize,
        use_ring: bool,
        rms_pc: &[u8],
        add_pc: &[u8],
        wg_add: u32,
        pending_resid: bool,
    ) -> Option<()> {
        let tp_shard = self.tp_size > 1; // TP=2×PP EP: owned-expert filter + skip in-CB residual add
        let (tp_rank, tp_size) = (self.tp_rank, self.tp_size);
        let d = self.config.latent_moe_dims(top_k_cfg, moe_inter);
        let hs = d.hidden_size;
        let lat = d.moe_latent_size;
        // TP=2×PP: shared-expert up col-sharded / down row-sharded → sup/sact
        // buffers + matvec dims are shared_inter/tp_size on this rank; sdown is a
        // hidden-wide PARTIAL that the per-MoE-layer NR_MIX all-reduce completes.
        let shared_inter = if tp_shard {
            d.moe_shared_expert_intermediate_size / tp_size
        } else {
            d.moe_shared_expert_intermediate_size
        };

        // R2: `record_to_off`'s per-expert byte offsets into the concatenated
        // multi-expert scratch buffers (up_all/act_all at k*moe_inter*4,
        // down_all at k*lat*4) must land on a storage-buffer-friendly
        // alignment. Violating a device's minStorageBufferOffsetAlignment is
        // a validation error / UB, not merely a perf hit — degrade safely to
        // the per-layer (flag-OFF) path instead of risking it in prod.
        debug_assert!(
            (lat * 4) % 64 == 0 && (moe_inter * 4) % 64 == 0,
            "nem_moe_resident_layer_tail: lat*4/moe_inter*4 must be 64B-aligned for record_to_off"
        );
        if (lat * 4) % 64 != 0 || (moe_inter * 4) % 64 != 0 {
            return self.nem_moe_resident_layer_legacy(
                layer_idx, top_k_cfg, moe_inter, use_ring, rms_pc, add_pc, wg_add, pending_resid,
            );
        }

        let p = format!("backbone.layers.{layer_idx}.mixer");
        let norm_w =
            &self.gpu_norm_w[&format!("backbone.layers.{layer_idx}.norm.weight")] as *const compute::Buffer;

        let fc1_meta = self.nem_meta(&format!("{p}.fc1_latent_proj.weight"))?;
        let sup_meta = self.nem_meta(&format!("{p}.shared_experts.up_proj.weight"))?;
        let fc2_meta = self.nem_meta(&format!("{p}.fc2_latent_proj.weight"))?;
        let sdown_meta = self.nem_meta(&format!("{p}.shared_experts.down_proj.weight"))?;
        let up_meta = self.nem_expert_meta(layer_idx, true)?;
        let down_meta = self.nem_expert_meta(layer_idx, false)?;
        let moe_inter = up_meta.3; // out_features of the up matvec

        // CB_1 (still waited): rms_norm(NR_H -> NR_X) -> fc1_latent_proj(NR_X
        // -> latent) ‖ shared_experts.up_proj(NR_X -> sup). Only `x_host` is
        // read back — `sup` is NOT read back (relu² on it moves to the GPU
        // below) and neither `lat_buf` nor `sup_buf` is returned to the pool
        // here; both handles are carried forward into the collapsed CB.
        let hp = self.nr_ptr(NR_H);
        let xp = self.nr_ptr(NR_X);
        let mixp = self.nr_ptr(NR_MIX);
        let peerp = self.nr_ptr(NR_PEER);
        let eng = self.engine.as_mut()?;
        let lat_buf = eng.alloc_host_coherent_storage((lat * 4) as u64).ok()?;
        let sup_buf = eng.alloc_host_coherent_storage((shared_inter * 4) as u64).ok()?;
        let lat_p = &lat_buf as *const compute::Buffer;
        let sup_p = &sup_buf as *const compute::Buffer;
        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        let ts_on = prof_on() && !use_ring && eng.ensure_ts_pool(2);
        if ts_on {
            eng.ts_cmd_reset(cb, 0, 2);
            eng.ts_cmd_mark(cb, 0, true);
        }
        eng.record_barrier_to(cb);
        unsafe {
            // Lever 2: fold the previous MoE layer's deferred TP residual add in.
            if pending_resid {
                Self::nem_rec_deferred_resid(eng, cb, hp, mixp, peerp, add_pc, wg_add)?;
            }
            eng.record_to(cb, "rms_norm_f32_mul", &[&*hp, &*norm_w, &*xp], rms_pc, (1, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            Self::nem_rec_mv(eng, cb, fc1_meta, xp, lat_p, hs, lat)?;
            Self::nem_rec_mv(eng, cb, sup_meta, xp, sup_p, hs, shared_inter)?;
        }
        if ts_on { eng.ts_cmd_mark(cb, 1, false); }
        let t_fence = std::time::Instant::now();
        if use_ring {
            eng.submit_batch_pipelined(cb, Vec::new()).ok()?;
            eng.wait_batch_pipelined().ok()?;
        } else {
            eng.submit_batch(cb).ok()?;
        }
        prof_add("nem_cb_fence", t_fence);
        if ts_on {
            if let Ok(v) = eng.ts_read_ns(0, 2) {
                prof_add_ns("nem_ts_cb1", (v[1] - v[0]).max(0.0) as u128);
            }
        }
        let x_host = read_f32_buf(unsafe { &*xp }, hs);

        // host: router only — relu² moved to the GPU, so the former host
        // `sact = relu2(sup)` line is gone.
        let t_gate = std::time::Instant::now();
        let gate_weight = self.weights.f32_slice(&format!("{p}.gate.weight"));
        let e_bias = self.weights.f32_slice(&format!("{p}.gate.e_score_correction_bias"));
        prof_add("nem_gate_copy", t_gate);
        let (indices, weights) = router_forward(&x_host, gate_weight, e_bias, &d.router);
        // TP=2×PP EP: the router gate is REPLICATED, so both ranks draw the SAME
        // top-k global expert ids; this rank runs ONLY the experts it owns
        // (`expert_owned_range`), remapping each to its LOCAL id in the half-sized
        // resident expert buffer, and keeps the SAME per-expert gate weight (NO
        // re-normalization within the owned half — the flagged trap). The routed
        // accumulator is thus a PARTIAL; fc2 is replicated+linear so
        // fc2(Σ_ranks routed_partial) = Σ_ranks fc2(routed_partial), and the
        // single NR_MIX all-reduce in the forward loop completes it.
        let (indices, weights) = if tp_shard {
            let (owned_lo, owned_cnt) =
                crate::nemotron_tp::expert_owned_range(d.router.n_routed_experts, tp_rank, tp_size);
            let mut idx = Vec::new();
            let mut w = Vec::new();
            for (k, &e) in indices.iter().enumerate() {
                if e >= owned_lo && e < owned_lo + owned_cnt {
                    idx.push(e - owned_lo);
                    w.push(weights[k]);
                }
            }
            (idx, w)
        } else {
            (indices, weights)
        };
        let top_k = indices.len();

        // Collapsed CB: top_k up matvecs -> relu2(up_all)/relu2(sup) -> top_k
        // down matvecs ‖ shared down -> nemotron_moe_accum (routed weighted
        // accumulate) -> fc2 -> (moe_out + sdown -> NR_MIX) -> residual add
        // (NR_H += NR_MIX). One dispatch per phase-boundary barrier; no host
        // readback at all until the NEXT layer's CB_1 needs NR_H/NR_X. This
        // is the layer's only non-waited ring submission (mirrors CB_4 in
        // the legacy path).
        let eng = self.engine.as_mut()?;
        // Under EP this rank may own ZERO of the token's selected experts (all
        // top-k fell to the peer half); clamp the top_k-sized concat buffers to
        // >=1 elem so the alloc never zero-sizes, and pre-zero `routed` so the
        // routed contribution is 0 (only the replicated shared expert then feeds
        // NR_MIX — still correct after the all-reduce). The peer contributes the
        // routed terms this rank skipped.
        let kbuf = top_k.max(1);
        let up_all = eng.alloc_host_coherent_storage((kbuf * moe_inter * 4) as u64).ok()?;
        let act_all = eng.alloc_host_coherent_storage((kbuf * moe_inter * 4) as u64).ok()?;
        let down_all = eng.alloc_host_coherent_storage((kbuf * lat * 4) as u64).ok()?;
        let sact = eng.alloc_host_coherent_storage((shared_inter * 4) as u64).ok()?;
        let sdown = eng.alloc_host_coherent_storage((hs * 4) as u64).ok()?;
        let routed = eng.alloc_host_coherent_storage((lat * 4) as u64).ok()?;
        if tp_shard {
            routed.write(&vec![0u8; lat * 4]).ok()?; // routed=0 baseline for the top_k==0 case
        }
        let moe_out = eng.alloc_host_coherent_storage((hs * 4) as u64).ok()?;
        let w_buf = eng.alloc_host_coherent_storage((kbuf * 4) as u64).ok()?;
        w_buf.write(&f32_slice_to_bytes(&weights)).ok()?;

        let up_all_p = &up_all as *const compute::Buffer;
        let act_all_p = &act_all as *const compute::Buffer;
        let down_all_p = &down_all as *const compute::Buffer;
        let sact_p = &sact as *const compute::Buffer;
        let sdown_p = &sdown as *const compute::Buffer;
        let routed_p = &routed as *const compute::Buffer;
        let moe_out_p = &moe_out as *const compute::Buffer;
        let w_buf_p = &w_buf as *const compute::Buffer;
        let mixp = self.nr_ptr(NR_MIX);
        let hp = self.nr_ptr(NR_H);
        let eng = self.engine.as_mut()?;

        let up_all_kx = ew_unary_pc((top_k * moe_inter) as u32);
        let sup_kx = ew_unary_pc(shared_inter as u32);
        let accum_pc = nem_moe_accum_pc(lat, top_k);

        let cb = if use_ring { eng.begin_batch_pipelined().ok()? } else { eng.begin_batch().ok()? };
        let ts_on = prof_on() && !use_ring && eng.ensure_ts_pool(2);
        if ts_on {
            eng.ts_cmd_reset(cb, 0, 2);
            eng.ts_cmd_mark(cb, 0, true);
        }
        eng.record_barrier_to(cb);
        unsafe {
            // 1. top_k up matvecs: lat_buf (whole, in_off=0) -> up_all @ k*moe_inter*4.
            for (k, &e) in indices.iter().enumerate() {
                Self::nem_rec_expert_mv_off(
                    eng, cb, up_meta, e, lat_p, 0, up_all_p, (k * moe_inter * 4) as u64,
                )?;
            }
            eng.record_barrier_to(cb);
            // 2. relu2 over the whole concatenated up_all, and over sup.
            eng.record_to(cb, "relu2_f32", &[&*up_all_p, &*act_all_p], &up_all_kx,
                (((top_k * moe_inter) as u32 + 511) / 512, 1, 1)).ok()?;
            eng.record_to(cb, "relu2_f32", &[&*sup_p, &*sact_p], &sup_kx,
                ((shared_inter as u32 + 511) / 512, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            // 3. top_k down matvecs: act_all @ k*moe_inter*4 -> down_all @ k*lat*4 ‖
            //    shared down (sact -> sdown).
            for (k, &e) in indices.iter().enumerate() {
                Self::nem_rec_expert_mv_off(
                    eng, cb, down_meta, e, act_all_p, (k * moe_inter * 4) as u64,
                    down_all_p, (k * lat * 4) as u64,
                )?;
            }
            Self::nem_rec_mv(eng, cb, sdown_meta, sact_p, sdown_p, shared_inter, hs)?;
            eng.record_barrier_to(cb);
            // 4. weighted routed accumulate (variable top_k, unlike q35_moe_accum's fixed 8).
            //    Skipped when this EP rank owns none of the selected experts —
            //    `routed` keeps its pre-zeroed value (accum writes, not adds).
            if top_k > 0 {
                eng.record_to(cb, "nemotron_moe_accum", &[&*down_all_p, &*w_buf_p, &*routed_p],
                    &accum_pc, ((lat as u32 + 255) / 256, 1, 1)).ok()?;
            }
            eng.record_barrier_to(cb);
            // 5. fc2_latent_proj(routed -> moe_out).
            Self::nem_rec_mv(eng, cb, fc2_meta, routed_p, moe_out_p, lat, hs)?;
            eng.record_barrier_to(cb);
            // 6. (moe_out + sdown -> NR_MIX) -> residual add (NR_H += NR_MIX).
            eng.record_to(cb, "add_f32_f32_f32", &[&*moe_out_p, &*sdown_p, &*mixp, &*mixp], add_pc, (wg_add, 1, 1)).ok()?;
            eng.record_barrier_to(cb);
            if !tp_shard { // TP=2×PP: residual add deferred to the loop's post-reduce step
            eng.record_to(cb, "add_f32_f32_f32", &[&*hp, &*mixp, &*hp, &*hp], add_pc, (wg_add, 1, 1)).ok()?;
            }
        }
        if ts_on { eng.ts_cmd_mark(cb, 1, false); }
        if use_ring {
            eng.submit_batch_pipelined(
                cb,
                vec![lat_buf, sup_buf, up_all, act_all, down_all, sact, sdown, routed, moe_out, w_buf],
            ).ok()?;
        } else {
            let t_fence = std::time::Instant::now();
            eng.submit_batch(cb).ok()?;
            prof_add("nem_cb_fence", t_fence);
            if ts_on {
                if let Ok(v) = eng.ts_read_ns(0, 2) {
                    prof_add_ns("nem_ts_moe_tail", (v[1] - v[0]).max(0.0) as u128);
                }
            }
            eng.return_to_pool(lat_buf);
            eng.return_to_pool(sup_buf);
            eng.return_to_pool(up_all);
            eng.return_to_pool(act_all);
            eng.return_to_pool(down_all);
            eng.return_to_pool(sact);
            eng.return_to_pool(sdown);
            eng.return_to_pool(routed);
            eng.return_to_pool(moe_out);
            eng.return_to_pool(w_buf);
        }
        Some(())
    }
}

#[inline]
fn dims_uses_conv_bias(cfg: &NemotronConfig) -> bool {
    cfg.use_conv_bias
}

// ─── Tests: numeric gate vs the golden fixtures ──────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    // Golden fixtures (small-dim seeded input+weights+expected output, produced
    // by nemotron_ref/validate_numerics.py against the real HF module at cos=1.0).
    // Converted npz→json alongside the .npz (see the fixtures dir).
    const MAMBA_FIXTURE: &str =
        include_str!("../nemotron_ref/fixtures/mamba2_decode_step.json");
    const MOE_FIXTURE: &str =
        include_str!("../nemotron_ref/fixtures/latent_moe_forward.json");
    const MAMBA_PREFILL_FIXTURE: &str =
        include_str!("../nemotron_ref/fixtures/mamba2_prefill_scan.json");

    /// PORT GATE (VLLM_VULKAN_NVFP4_E4M3_SCALES on the nemotron MoE expert
    /// path): the e4m3-resident expert dispatch must be BIT-EXACT to the
    /// f32-fold dispatch. The shared-loader `nvfp4_e4m3_resident_matches_f32_fold`
    /// proves the KERNEL arithmetic; this proves the nemotron-specific plumbing
    /// the port adds — the CONCAT-buffer per-expert slice offset (`sb_off`) and
    /// the PER-EXPERT `.weight_scale_2` global lookup — reconstruct the same
    /// matvec for MULTIPLE experts each with a DIFFERENT global. Emulates both
    /// data paths exactly as the loader writes them + the shaders read them.
    #[test]
    fn nemotron_expert_e4m3_matches_f32_fold() {
        use crate::model::{e4m3_to_f32, NVFP4_E2M1_LUT};
        use crate::push_constants::nvfp4_fold_scales;
        // 3 experts, up_proj shape [out=6, in=32, group=16] → groups=2/row.
        let (ne, out_f, in_f, gs) = (3usize, 6usize, 32usize, 16usize);
        let (bpr, groups) = (in_f / 2, in_f / gs);
        // Per-expert distinct globals (real modelopt: weight_scale_2 is per
        // expert tensor) — the exact thing folding hides and the e4m3 path must
        // reapply per dispatch.
        let globals = [0.00013f32, 0.00021f32, 0.00009f32];
        // Packed nibbles + raw e4m3 block-scale bytes for every expert.
        let packed: Vec<u8> = (0..ne * out_f * bpr)
            .map(|i| (i.wrapping_mul(37).wrapping_add(11)) as u8).collect();
        let wscale: Vec<u8> = (0..ne * out_f * groups)
            .map(|i| (0x50 + (i % 20)) as u8).collect();
        let x: Vec<f32> = (0..in_f).map(|i| ((i as f32) * 0.37).sin()).collect();

        // ── f32-fold CONCAT buffer (what the loader writes when e4m3=false):
        // per expert, fold(wscale_slice, globals[e]) → f32/group, concatenated.
        let mut fold_concat = vec![0f32; ne * out_f * groups];
        for e in 0..ne {
            let ws = &wscale[e * out_f * groups..(e + 1) * out_f * groups];
            let f = nvfp4_fold_scales(ws, globals[e]);
            fold_concat[e * out_f * groups..(e + 1) * out_f * groups].copy_from_slice(&f);
        }
        // ── e4m3 CONCAT buffer (what the loader writes when e4m3=true): raw
        // wscale bytes verbatim (== wscale already, since it IS the concat).
        let e4m3_concat: Vec<u8> = wscale.clone();
        // reinterpret e4m3 bytes as LE u32 words for the shader's byte-extract.
        let swords: Vec<u32> = e4m3_concat
            .chunks(4)
            .map(|c| c.iter().enumerate().fold(0u32, |w, (i, &b)| w | ((b as u32) << (8 * i))))
            .collect();

        for e in 0..ne {
            // Dispatch offsets exactly as nem_rec_expert_mv computes them.
            let packed_off = e * out_f * (in_f / 8); // words (in/8 per row)
            let sb_off = e * out_f * (in_f / gs);     // f32-elem (fold) / byte-elem (e4m3)
            for r in 0..out_f {
                let (mut acc_fold, mut acc_e4m3) = (0f32, 0f32);
                for j in 0..in_f {
                    // packed nibble (word-addressed like the shader: 8 codes/u32).
                    let wi = packed_off + r * (in_f / 8) + j / 8;
                    let word = {
                        let base = wi * 4;
                        (0..4).fold(0u32, |w, b| w | ((packed[base + b] as u32) << (8 * b)))
                    };
                    let code = ((word >> ((j % 8) * 4)) & 0xF) as usize;
                    let g = j / gs;
                    // f32-fold path: E2M1 * folded_concat[elem].
                    let fe = sb_off + r * groups + g;
                    acc_fold += NVFP4_E2M1_LUT[code] * fold_concat[fe] * x[j];
                    // e4m3 path: E2M1 * (e4m3(byte) * global[e]); byte from the
                    // concat via the shader's absolute-byte extract at sb_off+..
                    let sidx = sb_off + r * groups + g;
                    let sbyte = ((swords[sidx >> 2] >> ((sidx & 3) * 8)) & 0xFF) as u8;
                    let bscale = e4m3_to_f32(sbyte) * globals[e];
                    acc_e4m3 += NVFP4_E2M1_LUT[code] * bscale * x[j];
                }
                assert_eq!(acc_e4m3.to_bits(), acc_fold.to_bits(),
                    "expert {e} row {r}: e4m3 {acc_e4m3} vs fold {acc_fold} not bit-exact");
            }
        }
        // e4m3 scale buffer is 4x smaller than the folded f32 buffer (the GTT
        // deliverable): fold bytes = elems*4, e4m3 bytes = elems*1.
        let fold_bytes = fold_concat.len() * 4;
        let e4m3_bytes = e4m3_concat.len();
        assert_eq!(fold_bytes / e4m3_bytes, 4, "e4m3 scale buffer must be 4x smaller");
    }

    fn arr(v: &Value, key: &str) -> Vec<f32> {
        v[key]["data"]
            .as_array()
            .unwrap_or_else(|| panic!("fixture key '{key}' has no data array"))
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }
    fn iarr(v: &Value, key: &str) -> Vec<usize> {
        v[key]["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as usize)
            .collect()
    }
    fn su(v: &Value, key: &str) -> usize {
        v[key].as_u64().unwrap() as usize
    }
    fn sf(v: &Value, key: &str) -> f32 {
        v[key].as_f64().unwrap() as f32
    }
    fn sb(v: &Value, key: &str) -> bool {
        v[key].as_bool().unwrap()
    }

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn mamba2_decode_matches_golden_fixture() {
        let v: Value = serde_json::from_str(MAMBA_FIXTURE).unwrap();
        let dims = Mamba2Dims {
            hidden_size: su(&v, "hidden_size"),
            num_heads: su(&v, "num_heads"),
            head_dim: su(&v, "head_dim"),
            ssm_state_size: su(&v, "ssm_state_size"),
            n_groups: su(&v, "n_groups"),
            conv_kernel: su(&v, "conv_kernel_size"),
            time_step_min: sf(&v, "time_step_min"),
            eps: sf(&v, "layer_norm_epsilon"),
        };

        let x = arr(&v, "x");
        let in_proj = arr(&v, "in_proj_weight");
        let conv_w = arr(&v, "conv1d_weight");
        let conv_b = arr(&v, "conv1d_bias");
        let a_log = arr(&v, "A_log");
        let d_skip = arr(&v, "D");
        let dt_bias = arr(&v, "dt_bias");
        let norm_w = arr(&v, "norm_weight");
        let out_proj = arr(&v, "out_proj_weight");

        let weights = Mamba2Weights {
            in_proj: &in_proj,
            conv1d_weight: &conv_w,
            conv1d_bias: Some(&conv_b),
            a_log: &a_log,
            d: &d_skip,
            dt_bias: &dt_bias,
            norm_weight: &norm_w,
            out_proj: &out_proj,
        };
        let mut state = Mamba2State {
            conv_state: arr(&v, "conv_state_in"),
            ssm_state: arr(&v, "ssm_state_in"),
        };

        let out = mamba2_decode_step(&x, &weights, &mut state, &dims);

        let expected_out = arr(&v, "expected_out");
        let expected_ssm = arr(&v, "expected_ssm_state_out");

        let out_cos = cos(&out, &expected_out);
        let out_mad = max_abs_diff(&out, &expected_out);
        let ssm_cos = cos(&state.ssm_state, &expected_ssm);
        let ssm_mad = max_abs_diff(&state.ssm_state, &expected_ssm);
        eprintln!(
            "mamba2 decode: out cos={out_cos:.7} max_abs_diff={out_mad:.3e}; \
             ssm cos={ssm_cos:.7} max_abs_diff={ssm_mad:.3e}"
        );
        assert!(out_cos >= 0.9999, "mamba out cos {out_cos} < 0.9999");
        assert!(ssm_cos >= 0.9999, "mamba ssm cos {ssm_cos} < 0.9999");
    }

    /// P1a gate 1: `mamba2_prefill_seq` on a T-token slab must be BIT-EXACT
    /// (not just cos-close) vs calling `mamba2_decode_step` T times on the same
    /// tokens from the same starting state — by construction, since both paths
    /// share the exact same per-token conv1d/recurrence/norm code
    /// (`conv1d_silu_step` / `mamba2_recurrence_and_norm`) and GEMMs compute
    /// each output row independently of the other rows in the batch.
    #[test]
    fn mamba2_prefill_seq_matches_t_times_decode_bit_exact() {
        // Reuse the (small, real-ish) decode fixture's dims/weights so this is
        // a clean unit test independent of the prefill fixture's own dims.
        let v: Value = serde_json::from_str(MAMBA_FIXTURE).unwrap();
        let dims = Mamba2Dims {
            hidden_size: su(&v, "hidden_size"),
            num_heads: su(&v, "num_heads"),
            head_dim: su(&v, "head_dim"),
            ssm_state_size: su(&v, "ssm_state_size"),
            n_groups: su(&v, "n_groups"),
            conv_kernel: su(&v, "conv_kernel_size"),
            time_step_min: sf(&v, "time_step_min"),
            eps: sf(&v, "layer_norm_epsilon"),
        };
        let in_proj = arr(&v, "in_proj_weight");
        let conv_w = arr(&v, "conv1d_weight");
        let conv_b = arr(&v, "conv1d_bias");
        let a_log = arr(&v, "A_log");
        let d_skip = arr(&v, "D");
        let dt_bias = arr(&v, "dt_bias");
        let norm_w = arr(&v, "norm_weight");
        let out_proj = arr(&v, "out_proj_weight");
        let weights = Mamba2Weights {
            in_proj: &in_proj,
            conv1d_weight: &conv_w,
            conv1d_bias: Some(&conv_b),
            a_log: &a_log,
            d: &d_skip,
            dt_bias: &dt_bias,
            norm_weight: &norm_w,
            out_proj: &out_proj,
        };

        // Synthesize a T=7 token slab (deterministic pseudo-random via a tiny
        // LCG so this test has no extra dependency) from the fixture's own
        // hidden_size, all starting from a ZERO state (fresh prefill) as well
        // as a NONZERO state (mid-sequence continuation) to exercise both.
        let hs = dims.hidden_size;
        let t = 7usize;
        let mut lcg: u32 = 0xC0FFEE;
        let mut next = || {
            lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
            ((lcg >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.4
        };
        let xs: Vec<f32> = (0..t * hs).map(|_| next()).collect();

        for nonzero_start in [false, true] {
            let mut state_seq = Mamba2State::zeros(&dims);
            let mut state_decode = Mamba2State::zeros(&dims);
            if nonzero_start {
                for v in state_seq.conv_state.iter_mut() {
                    *v = next() * 0.1;
                }
                for v in state_seq.ssm_state.iter_mut() {
                    *v = next() * 0.1;
                }
                state_decode.conv_state.copy_from_slice(&state_seq.conv_state);
                state_decode.ssm_state.copy_from_slice(&state_seq.ssm_state);
            }

            let seq_out = mamba2_prefill_seq(&xs, &weights, &mut state_seq, &dims);

            let mut decode_out = vec![0.0f32; t * hs];
            for ti in 0..t {
                let step_out = mamba2_decode_step(&xs[ti * hs..(ti + 1) * hs], &weights, &mut state_decode, &dims);
                decode_out[ti * hs..(ti + 1) * hs].copy_from_slice(&step_out);
            }

            assert_eq!(
                seq_out, decode_out,
                "mamba2_prefill_seq output not bit-exact vs T*mamba2_decode_step (nonzero_start={nonzero_start})"
            );
            assert_eq!(
                state_seq.ssm_state, state_decode.ssm_state,
                "final ssm_state not bit-exact vs T*decode (nonzero_start={nonzero_start})"
            );
            assert_eq!(
                state_seq.conv_state, state_decode.conv_state,
                "final conv_state not bit-exact vs T*decode (nonzero_start={nonzero_start})"
            );
        }
    }

    /// P1a gate 2: `mamba2_prefill_seq` on a FRESH sequence (T=20, crossing
    /// chunk_size=8 boundaries 3x with padding) vs the HF-derived chunked-SSD
    /// golden fixture (`mamba2_prefill_scan.json`, generated by
    /// `nemotron_ref/validate_numerics.py::validate_mamba2_prefill`). This is
    /// the sequential scan vs the DIFFERENT (chunked) scan algorithm, so the
    /// bar is cos >= 0.9999 (the FP-reassociation gap the plan documents), not
    /// bit-exact.
    #[test]
    fn mamba2_prefill_seq_matches_chunked_golden_fixture() {
        let v: Value = serde_json::from_str(MAMBA_PREFILL_FIXTURE).unwrap();
        let dims = Mamba2Dims {
            hidden_size: su(&v, "hidden_size"),
            num_heads: su(&v, "num_heads"),
            head_dim: su(&v, "head_dim"),
            ssm_state_size: su(&v, "ssm_state_size"),
            n_groups: su(&v, "n_groups"),
            conv_kernel: su(&v, "conv_kernel_size"),
            time_step_min: sf(&v, "time_step_min"),
            eps: sf(&v, "layer_norm_epsilon"),
        };

        let x = arr(&v, "x");
        let in_proj = arr(&v, "in_proj_weight");
        let conv_w = arr(&v, "conv1d_weight");
        let conv_b = arr(&v, "conv1d_bias");
        let a_log = arr(&v, "A_log");
        let d_skip = arr(&v, "D");
        let dt_bias = arr(&v, "dt_bias");
        let norm_w = arr(&v, "norm_weight");
        let out_proj = arr(&v, "out_proj_weight");
        let weights = Mamba2Weights {
            in_proj: &in_proj,
            conv1d_weight: &conv_w,
            conv1d_bias: Some(&conv_b),
            a_log: &a_log,
            d: &d_skip,
            dt_bias: &dt_bias,
            norm_weight: &norm_w,
            out_proj: &out_proj,
        };

        // Fresh prefill: zero starting state (has_previous_state=False on the
        // HF side, matching `mamba2_prefill_scan`'s golden-fixture generation).
        let mut state = Mamba2State::zeros(&dims);
        let out = mamba2_prefill_seq(&x, &weights, &mut state, &dims);

        let expected_out = arr(&v, "expected_out");
        let expected_ssm = arr(&v, "expected_ssm_state_out");
        let expected_conv = arr(&v, "expected_conv_state_out");

        let out_cos = cos(&out, &expected_out);
        let out_mad = max_abs_diff(&out, &expected_out);
        let ssm_cos = cos(&state.ssm_state, &expected_ssm);
        let ssm_mad = max_abs_diff(&state.ssm_state, &expected_ssm);
        let conv_cos = cos(&state.conv_state, &expected_conv);
        let conv_mad = max_abs_diff(&state.conv_state, &expected_conv);
        eprintln!(
            "mamba2 prefill (sequential scan vs chunked-SSD golden fixture): \
             out cos={out_cos:.7} max_abs_diff={out_mad:.3e}; \
             ssm cos={ssm_cos:.7} max_abs_diff={ssm_mad:.3e}; \
             conv cos={conv_cos:.7} max_abs_diff={conv_mad:.3e}"
        );
        assert!(out_cos >= 0.9999, "mamba prefill out cos {out_cos} < 0.9999");
        assert!(ssm_cos >= 0.9999, "mamba prefill ssm_state cos {ssm_cos} < 0.9999");
        assert!(conv_cos >= 0.9999, "mamba prefill conv_state cos {conv_cos} < 0.9999");
    }

    #[test]
    fn latent_moe_matches_golden_fixture() {
        let v: Value = serde_json::from_str(MOE_FIXTURE).unwrap();
        let hs = su(&v, "hidden_size");
        let dims = LatentMoeDims {
            hidden_size: hs,
            moe_latent_size: su(&v, "moe_latent_size"),
            moe_intermediate_size: su(&v, "moe_intermediate_size"),
            moe_shared_expert_intermediate_size: su(&v, "moe_shared_expert_intermediate_size"),
            router: RouterDims {
                n_routed_experts: su(&v, "n_routed_experts"),
                top_k: su(&v, "num_experts_per_tok"),
                routed_scaling_factor: sf(&v, "routed_scaling_factor"),
                n_group: su(&v, "n_group"),
                topk_group: su(&v, "topk_group"),
                norm_topk_prob: sb(&v, "norm_topk_prob"),
            },
        };

        let x = arr(&v, "x");
        let n_tokens = x.len() / hs;
        let gate_weight = arr(&v, "gate_weight");
        let e_bias = arr(&v, "e_score_correction_bias");
        let fc1 = arr(&v, "fc1_latent_proj");
        let fc2 = arr(&v, "fc2_latent_proj");
        let expert_up = arr(&v, "expert_up_proj");
        let expert_down = arr(&v, "expert_down_proj");
        let shared_up = arr(&v, "shared_up_proj");
        let shared_down = arr(&v, "shared_down_proj");

        let weights = LatentMoeWeights {
            gate_weight: &gate_weight,
            e_score_correction_bias: &e_bias,
            fc1_latent_proj: &fc1,
            fc2_latent_proj: &fc2,
            expert_up: &expert_up,
            expert_down: &expert_down,
            shared_up: &shared_up,
            shared_down: &shared_down,
        };

        // Router index check (per token) against the golden selection.
        let expected_idx = iarr(&v, "expected_topk_indices");
        let top_k = dims.router.top_k;
        for t in 0..n_tokens {
            let (idx, _w) = router_forward(
                &x[t * hs..(t + 1) * hs],
                &gate_weight,
                &e_bias,
                &dims.router,
            );
            let mut got = idx.clone();
            let mut exp = expected_idx[t * top_k..(t + 1) * top_k].to_vec();
            got.sort_unstable();
            exp.sort_unstable();
            assert_eq!(got, exp, "token {t} routed-expert set mismatch");
        }

        let out = latent_moe_forward(&x, n_tokens, &weights, &dims);
        let expected_out = arr(&v, "expected_out");
        let out_cos = cos(&out, &expected_out);
        let out_mad = max_abs_diff(&out, &expected_out);
        eprintln!("latent-moe: out cos={out_cos:.7} max_abs_diff={out_mad:.3e}");
        assert!(out_cos >= 0.9999, "latent-moe out cos {out_cos} < 0.9999");
    }

    /// GATE B (NEW-1b): `cpu_matvec_par`'s per-row rayon dot-product must be
    /// BIT-FOR-BIT identical to `cpu_matmul(a,b,1,k,n)`'s `matrixmultiply::sgemm`
    /// path before router_forward's matmul is allowed to swap to it — the two
    /// implementations use different accumulation strategies (naive serial sum
    /// vs. a blocked/vectorized BLAS-style kernel) and floating-point addition is
    /// not associative, so bit-identity is NOT guaranteed a priori and must be
    /// checked empirically at the actual router dims on this target (aarch64).
    ///
    /// RESULT (aarch64/Mac, this run): RED — `matrixmultiply::sgemm`'s blocked
    /// accumulation order does NOT match `cpu_matvec_par`'s naive per-row serial
    /// sum bit-for-bit (differences appear at the ~1e-7 ULP level, e.g.
    /// 0.06353909 vs 0.06353917). So NEW-1b (swapping router_forward's
    /// cpu_matmul for cpu_matvec_par) is NOT shipped. `#[ignore]`d rather than
    /// deleted so the gate can be re-run by hand if matrixmultiply/rayon/rustc
    /// versions change.
    #[test]
    #[ignore = "RED on aarch64: cpu_matvec_par is not bit-identical to cpu_matmul (see doc comment); NEW-1b not shipped"]
    fn cpu_matvec_par_bit_identical() {
        // Simple xorshift-style PRNG for reproducible pseudo-random f32s without
        // pulling in a `rand` dev-dependency.
        fn next(seed: &mut u64) -> f32 {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as i32 as f32) / (i32::MAX as f32)
        }
        let k = 4096usize; // hidden_size
        let n = 512usize; // n_routed_experts
        let mut seed = 0x9E3779B97F4A7C15u64;
        let a: Vec<f32> = (0..k).map(|_| next(&mut seed)).collect();
        let b: Vec<f32> = (0..n * k).map(|_| next(&mut seed)).collect();

        let par = cpu_matvec_par(&a, &b, k, n);
        let serial = cpu_matmul(&a, &b, 1, k, n);
        assert_eq!(par, serial, "cpu_matvec_par is not bit-identical to cpu_matmul at router dims (k={k}, n={n})");
    }

    #[test]
    fn config_parses_block_specs() {
        // Minimal synthetic config exercising all 3 block types + per-MoE overrides.
        let j = serde_json::json!({
            "hidden_size": 4096,
            "num_hidden_layers": 4,
            "vocab_size": 131072,
            "norm_eps": 1e-5,
            "tie_word_embeddings": false,
            "mamba_num_heads": 128, "mamba_head_dim": 64, "ssm_state_size": 96,
            "n_groups": 8, "conv_kernel": 4, "use_conv_bias": true, "time_step_min": 0.001,
            "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
            "attention_bias": false,
            "n_routed_experts": 512, "moe_latent_size": 1024,
            "moe_shared_expert_intermediate_size": 5376,
            "routed_scaling_factor": 5.0, "norm_topk_prob": true,
            "n_group": 1, "topk_group": 1,
            "block_configs": [
                {"block_type": "mamba"},
                {"block_type": "moe", "num_experts_per_tok": 4, "moe_intermediate_size": 1280},
                {"block_type": "attention"},
                {"block_type": "moe", "num_experts_per_tok": 18, "moe_intermediate_size": 2688}
            ]
        });
        let cfg = NemotronConfig::from_json(&j).unwrap();
        assert_eq!(cfg.block_specs.len(), 4);
        assert_eq!(cfg.block_specs[0], BlockSpec::Mamba);
        assert_eq!(
            cfg.block_specs[1],
            BlockSpec::Moe { num_experts_per_tok: 4, moe_intermediate_size: 1280 }
        );
        assert_eq!(cfg.block_specs[2], BlockSpec::Attention);
        assert_eq!(
            cfg.block_specs[3],
            BlockSpec::Moe { num_experts_per_tok: 18, moe_intermediate_size: 2688 }
        );
        // Derived mamba dims (real-checkpoint values).
        let md = cfg.mamba_dims();
        assert_eq!(md.intermediate(), 8192);
        assert_eq!(md.conv_dim(), 8192 + 2 * 8 * 96); // 9728
        assert_eq!(md.in_proj_out(), 8192 + 9728 + 128); // 18048
        assert_eq!(md.norm_group_size(), 8192 / 8); // 1024
    }

    /// Deterministic pseudo-random f32 stream (LCG) so the GEMM tolerance test
    /// needs no `rand` dep and is bit-reproducible across runs/hosts.
    fn lcg_fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * scale
            })
            .collect()
    }

    /// CPU emulation of the `matmul_f16_f32_fp32` mul_mm kernel that the GPU
    /// Mamba-projection path (`NemotronModel::mamba_proj`) dispatches: the weight
    /// is read as f16, the input stays f32, and the dot product accumulates in
    /// f32 — exactly `out[t,n] = sum_k x[t,k] * f16(W[n,k])`. This isolates the
    /// ONE numerical difference vs the CPU `cpu_matmul` reference (f16 weight
    /// rounding); the device-execution cos is the cluster gate.
    fn gpu_gemm_f16_emulated(x: &[f32], w: &[f32], t: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; t * n];
        for ti in 0..t {
            for nn in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let wf16 = half::f16::from_f32(w[nn * k + kk]).to_f32();
                    acc += x[ti * k + kk] * wf16;
                }
                out[ti * n + nn] = acc;
            }
        }
        out
    }

    /// The GPU Mamba-projection path routes `in_proj`/`out_proj` through the
    /// f16-aligned batched GEMM. On a device-less host the actual dispatch can't
    /// run, so this validates the only new numerical risk — casting the
    /// FP8-dequantized f32 weight to f16 for the GPU-resident copy — matches the
    /// CPU `cpu_matmul` reference at cos >= 0.999, using the real projection
    /// contraction depths (`in_proj` k=hidden=4096, `out_proj` k=intermediate
    /// =8192). Weight magnitudes (~0.05) are representative of FP8-E4M3 dequant.
    #[test]
    fn mamba_projection_f16_gemm_matches_cpu_cos() {
        // in_proj: out[t, n] = x[t, hidden] @ W_in[n, hidden]^T
        for &(k, n, label) in &[(4096usize, 2048usize, "in_proj"), (8192usize, 1024usize, "out_proj")] {
            let t = 8usize;
            let x = lcg_fill(t * k, 0xA11CE ^ k as u64, 1.0);
            let w = lcg_fill(n * k, 0xB0B ^ n as u64, 0.05);
            let cpu = cpu_matmul(&x, &w, t, k, n);
            let gpu = gpu_gemm_f16_emulated(&x, &w, t, k, n);
            let c = cos(&cpu, &gpu);
            eprintln!(
                "mamba {label} f16-GEMM vs cpu_matmul (k={k},n={n},t={t}): \
                 cos={c:.7} max_abs_diff={:.3e}",
                max_abs_diff(&cpu, &gpu)
            );
            assert!(
                c >= 0.999,
                "{label} f16-GEMM cos {c} < 0.999 (max_abs_diff {})",
                max_abs_diff(&cpu, &gpu)
            );
        }
    }

    /// The prefill refactor split the SSD scan out of `mamba2_prefill_seq` into
    /// `mamba2_scan_only` so the GPU path can drive the two projections
    /// separately. Guard that the CPU composition (in_proj cpu_matmul ->
    /// scan_only -> out_proj cpu_matmul) is still BIT-EXACT to the fixture
    /// prefill, i.e. the split introduced no arithmetic change on the CPU path
    /// that remains the correctness fallback when the GPU path is off.
    #[test]
    fn mamba2_scan_split_is_bit_exact_vs_prefill_seq() {
        let v: Value = serde_json::from_str(MAMBA_PREFILL_FIXTURE).unwrap();
        let dims = Mamba2Dims {
            hidden_size: su(&v, "hidden_size"),
            num_heads: su(&v, "num_heads"),
            head_dim: su(&v, "head_dim"),
            ssm_state_size: su(&v, "ssm_state_size"),
            n_groups: su(&v, "n_groups"),
            conv_kernel: su(&v, "conv_kernel_size"),
            time_step_min: sf(&v, "time_step_min"),
            eps: sf(&v, "layer_norm_epsilon"),
        };
        let x = arr(&v, "x");
        let in_proj = arr(&v, "in_proj_weight");
        let conv_w = arr(&v, "conv1d_weight");
        let conv_b = arr(&v, "conv1d_bias");
        let a_log = arr(&v, "A_log");
        let d_skip = arr(&v, "D");
        let dt_bias = arr(&v, "dt_bias");
        let norm_w = arr(&v, "norm_weight");
        let out_proj = arr(&v, "out_proj_weight");
        let w = Mamba2Weights {
            in_proj: &in_proj,
            conv1d_weight: &conv_w,
            conv1d_bias: Some(&conv_b),
            a_log: &a_log,
            d: &d_skip,
            dt_bias: &dt_bias,
            norm_weight: &norm_w,
            out_proj: &out_proj,
        };
        let t = x.len() / dims.hidden_size;

        // reference: the whole fused prefill
        let mut st_a = Mamba2State::zeros(&dims);
        let fused = mamba2_prefill_seq(&x, &w, &mut st_a, &dims);
        // manual: in_proj -> scan_only -> out_proj, mirroring mamba_proj's CPU path
        let mut st_b = Mamba2State::zeros(&dims);
        let proj = cpu_matmul(&x, w.in_proj, t, dims.hidden_size, dims.in_proj_out());
        let scans = mamba2_scan_only(&proj, &w, &mut st_b, &dims);
        let manual = cpu_matmul(&scans, w.out_proj, t, dims.intermediate(), dims.hidden_size);
        assert_eq!(fused, manual, "scan-split composition diverged from fused prefill");
    }

    // ─── Generate-loop wiring: multi-step decode + state carry ────────────────

    /// Tiny all-3-block-type config (deterministic dims) for the generate-loop
    /// / PP-split wiring tests. `blocks` picks the per-layer mixer kind.
    fn tiny_nem_config(blocks: &[&str]) -> NemotronConfig {
        let block_configs: Vec<Value> = blocks
            .iter()
            .map(|b| match *b {
                "moe" => serde_json::json!(
                    {"block_type":"moe","num_experts_per_tok":2,"moe_intermediate_size":12}),
                other => serde_json::json!({"block_type": other}),
            })
            .collect();
        let j = serde_json::json!({
            "hidden_size": 32,
            "num_hidden_layers": blocks.len(),
            "vocab_size": 16,
            "norm_eps": 1e-5,
            "tie_word_embeddings": false,
            "mamba_num_heads": 2, "mamba_head_dim": 8, "ssm_state_size": 4,
            "n_groups": 2, "conv_kernel": 4, "use_conv_bias": true, "time_step_min": 0.001,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "attention_bias": false,
            "n_routed_experts": 4, "moe_latent_size": 8,
            "moe_shared_expert_intermediate_size": 10,
            "routed_scaling_factor": 1.0, "norm_topk_prob": true,
            "n_group": 1, "topk_group": 1,
            "block_configs": block_configs,
        });
        NemotronConfig::from_json(&j).unwrap()
    }

    /// Populate EVERY weight tensor `forward_pp_range` + the embed/lm_head
    /// bookends read, with deterministic LCG values (so two calls yield
    /// identical weights → two independent models are bit-comparable). Fills all
    /// layers regardless of stage split; each stage only reads its own range.
    fn build_fixture_weights(cfg: &NemotronConfig) -> ModelWeights {
        use crate::model::SimpleTensor;
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let md = cfg.mamba_dims();
        let inter = md.intermediate();
        let conv_dim = md.conv_dim();
        let in_proj_out = md.in_proj_out();
        let ck = cfg.conv_kernel;
        let nh = cfg.mamba_num_heads;
        let (nq, nkv, hd) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let (q_dim, kv_dim) = (nq * hd, nkv * hd);
        let ne = cfg.n_routed_experts;
        let lat = cfg.moe_latent_size;
        let shared_inter = cfg.moe_shared_expert_intermediate_size;

        let mut tensors: std::collections::HashMap<String, SimpleTensor> =
            std::collections::HashMap::new();
        {
            let mut seed = 0x9E3779B97F4A7C15u64;
            let mut put = |name: String, shape: Vec<usize>| {
                let n: usize = shape.iter().product();
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                tensors.insert(name, SimpleTensor { data: lcg_fill(n, seed, 0.1), shape });
            };
            put("backbone.embeddings.weight".into(), vec![vocab, h]);
            put("backbone.norm_f.weight".into(), vec![h]);
            put("lm_head.weight".into(), vec![vocab, h]);
            for (l, spec) in cfg.block_specs.iter().enumerate() {
                let p = format!("backbone.layers.{l}");
                put(format!("{p}.norm.weight"), vec![h]);
                let m = format!("{p}.mixer");
                match *spec {
                    BlockSpec::Mamba => {
                        put(format!("{m}.in_proj.weight"), vec![in_proj_out, h]);
                        put(format!("{m}.conv1d.weight"), vec![conv_dim, ck]);
                        put(format!("{m}.conv1d.bias"), vec![conv_dim]);
                        put(format!("{m}.A_log"), vec![nh]);
                        put(format!("{m}.D"), vec![nh]);
                        put(format!("{m}.dt_bias"), vec![nh]);
                        put(format!("{m}.norm.weight"), vec![inter]);
                        put(format!("{m}.out_proj.weight"), vec![h, inter]);
                    }
                    BlockSpec::Attention => {
                        put(format!("{m}.q_proj.weight"), vec![q_dim, h]);
                        put(format!("{m}.k_proj.weight"), vec![kv_dim, h]);
                        put(format!("{m}.v_proj.weight"), vec![kv_dim, h]);
                        put(format!("{m}.o_proj.weight"), vec![h, q_dim]);
                    }
                    BlockSpec::Moe { moe_intermediate_size, .. } => {
                        put(format!("{m}.gate.weight"), vec![ne, h]);
                        put(format!("{m}.gate.e_score_correction_bias"), vec![ne]);
                        put(format!("{m}.fc1_latent_proj.weight"), vec![lat, h]);
                        put(format!("{m}.fc2_latent_proj.weight"), vec![h, lat]);
                        for e in 0..ne {
                            put(format!("{m}.experts.{e}.up_proj.weight"),
                                vec![moe_intermediate_size, lat]);
                            put(format!("{m}.experts.{e}.down_proj.weight"),
                                vec![lat, moe_intermediate_size]);
                        }
                        put(format!("{m}.shared_experts.up_proj.weight"), vec![shared_inter, h]);
                        put(format!("{m}.shared_experts.down_proj.weight"), vec![h, shared_inter]);
                    }
                }
            }
        }
        ModelWeights { tensors }
    }

    fn fixture_model(cfg: &NemotronConfig, start: usize, end: usize) -> NemotronModel {
        NemotronModel::new_range(
            cfg.clone(),
            build_fixture_weights(cfg),
            64,
            "lm_head.weight".to_string(),
            start,
            end,
        )
    }

    /// THE generate-loop test: N single-token decode steps through the WIRED
    /// `forward_pp_stage` (embed → resident layers advance conv/ssm + KV state in
    /// place → final norm + lm_head) must reproduce, BIT-EXACT at the final step,
    /// an independently-built reference that feeds the same N embedded tokens as
    /// one slab through `forward_pp_range` (multi-token unblock: prefill scan /
    /// per-token attn) + the same final norm+lm_head on the last row. This is
    /// what distinguishes a wired generate loop (step t+1 resumes from step t's
    /// Mamba/KV state) from calling forward_pp_range once — if the state did NOT
    /// carry, the sequential decode would diverge from the sequential slab.
    #[test]
    fn nemotron_generate_state_carry_bit_exact() {
        let cfg = tiny_nem_config(&["mamba", "attention", "mamba"]);
        let n = cfg.num_hidden_layers;
        let h = cfg.hidden_size;
        let tokens: [u32; 5] = [3, 7, 1, 5, 2];

        // Wired autoregressive decode: one token per forward_pp_stage call.
        let mut m = fixture_model(&cfg, 0, n);
        let mut wired: Vec<Vec<f32>> = Vec::new();
        for (t, &tok) in tokens.iter().enumerate() {
            let lg = m.forward_pp_stage(tok, &[], t);
            assert_eq!(lg.len(), cfg.vocab_size);
            assert!(lg.iter().all(|v| v.is_finite()), "step {t} logits not finite");
            wired.push(lg);
        }

        // Reference: embed all tokens → [N,H] slab → forward_pp_range once →
        // final norm+lm_head on the LAST row (state carried across the slab).
        let mut mref = fixture_model(&cfg, 0, n);
        let embed = mref.weights.f32_slice("backbone.embeddings.weight").to_vec();
        let mut slab = vec![0.0f32; tokens.len() * h];
        for (t, &tok) in tokens.iter().enumerate() {
            slab[t * h..(t + 1) * h]
                .copy_from_slice(&embed[tok as usize * h..(tok as usize + 1) * h]);
        }
        let hidden_ref = mref.forward_pp_range(&slab, 0, 0, n);
        let last_row = &hidden_ref[(tokens.len() - 1) * h..];
        let normed = cpu_rms_norm(last_row, &mref.w("backbone.norm_f.weight"), cfg.norm_eps);
        let logits_ref = cpu_matmul(&normed, &mref.w("lm_head.weight"), 1, h, cfg.vocab_size);

        let wired_last = wired.last().unwrap();
        let c = cos(wired_last, &logits_ref);
        eprintln!(
            "nemotron generate state-carry: final-step cos={c:.7} max_abs_diff={:.3e}",
            max_abs_diff(wired_last, &logits_ref)
        );
        assert_eq!(
            wired_last, &logits_ref,
            "wired sequential decode final-step logits not bit-exact vs slab reference \
             (Mamba conv/ssm or attention KV state did NOT carry across decode steps)"
        );

        // Guard against a vacuously-passing test: the final step MUST depend on
        // history — a fresh model fed ONLY the last token differs.
        let mut mfresh = fixture_model(&cfg, 0, n);
        let solo = mfresh.forward_pp_stage(*tokens.last().unwrap(), &[], 0);
        assert_ne!(
            wired_last, &solo,
            "final-step logits identical with vs without history — state carry not exercised"
        );
    }

    /// Phase-1 DECOUPLED trace-dump: `attach_mtp_trace` + `forward_pp_stage`'s
    /// write must produce EXACTLY the header + per-step record format the
    /// offline replay (`nem_mtp_accept_offline.py`) parses: magic `b"NMTP"` +
    /// `u32` LE `hidden_size`, then repeated `[hidden_size f32 LE][u32 LE
    /// token]`. Verifies byte-for-byte against `last_pre_norm_hidden` (the
    /// SAME cache the inline `nem_mtp_draft` path reads) and an
    /// independently-recomputed argmax of the step's own logits.
    #[test]
    fn mtp_trace_dump_round_trips_hidden_and_greedy_token() {
        use std::io::Read as _;
        let cfg = tiny_nem_config(&["mamba", "attention", "moe"]);
        let n = cfg.num_hidden_layers;
        let h = cfg.hidden_size;
        let tokens: [u32; 3] = [2, 5, 1];

        let tmp = std::env::temp_dir().join(format!(
            "nem_mtp_trace_test_{}.bin",
            std::process::id()
        ));
        let mut m = fixture_model(&cfg, 0, n);
        m.attach_mtp_trace(tmp.to_str().unwrap()).expect("attach_mtp_trace");

        let mut expected_hidden: Vec<Vec<f32>> = Vec::new();
        let mut expected_tok: Vec<u32> = Vec::new();
        for (t, &tok) in tokens.iter().enumerate() {
            let logits = m.forward_pp_stage(tok, &[], t);
            expected_hidden.push(m.last_pre_norm_hidden.clone());
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > bv { bv = v; bi = i; }
            }
            expected_tok.push(bi as u32);
        }
        drop(m); // flush/close the file

        let mut buf = Vec::new();
        std::fs::File::open(&tmp).unwrap().read_to_end(&mut buf).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert_eq!(&buf[0..4], b"NMTP", "trace header magic");
        let hidden_size = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        assert_eq!(hidden_size, h, "trace header hidden_size");

        let record_len = h * 4 + 4;
        let body = &buf[8..];
        assert_eq!(body.len(), tokens.len() * record_len, "record count/size");

        for (t, _) in tokens.iter().enumerate() {
            let rec = &body[t * record_len..(t + 1) * record_len];
            let hidden: Vec<f32> = rec[..h * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let tok = u32::from_le_bytes(rec[h * 4..h * 4 + 4].try_into().unwrap());
            assert_eq!(hidden, expected_hidden[t], "step {t}: traced hidden mismatch");
            assert_eq!(tok, expected_tok[t], "step {t}: traced greedy token mismatch");
        }
    }

    /// PP correctness proxy for the multi-node run (single node): splitting the
    /// layer range across two STAGES and shipping only the `hidden_size` vector
    /// between them (state resident per stage) must reproduce the monolithic
    /// single-stage decode BIT-EXACT, over multiple steps. The only untested
    /// delta vs the real PP-N run is the byte-transparent vCCL wire transport.
    #[test]
    fn nemotron_pp_stage_split_matches_monolithic_bit_exact() {
        let cfg = tiny_nem_config(&["mamba", "moe", "attention", "mamba"]);
        let n = cfg.num_hidden_layers;
        let k = 2; // stage boundary: stage0=[0,2), stage1=[2,4)
        let tokens: [u32; 6] = [4, 0, 9, 2, 7, 1];

        let mut mono = fixture_model(&cfg, 0, n);
        let mut s0 = fixture_model(&cfg, 0, k); // first, not last
        let mut s1 = fixture_model(&cfg, k, n); // not first, last

        for (t, &tok) in tokens.iter().enumerate() {
            let mono_lg = mono.forward_pp_stage(tok, &[], t);
            // stage 0 embeds + runs [0,k) → hidden (this is what ships over vCCL).
            let hidden = s0.forward_pp_stage(tok, &[], t);
            assert_eq!(hidden.len(), cfg.hidden_size, "inter-stage message must be hidden_size");
            // stage 1 consumes the hidden, runs [k,n) + norm+lm_head → logits.
            let split_lg = s1.forward_pp_stage(u32::MAX /*ignored (not first)*/, &hidden, t);
            assert_eq!(
                mono_lg, split_lg,
                "PP split diverged from monolithic at step {t} (hidden hop / resident state)"
            );
        }
    }

    /// Coherence smoke over a mixed-block-type model driven as a generate loop:
    /// several decode steps, every logit vector finite and full-vocab width.
    #[test]
    fn nemotron_generate_loop_finite_all_block_types() {
        let cfg = tiny_nem_config(&["mamba", "moe", "attention", "moe", "mamba"]);
        let n = cfg.num_hidden_layers;
        let mut m = fixture_model(&cfg, 0, n);
        let mut tok = 5u32;
        for t in 0..8 {
            let lg = m.forward_pp_stage(tok, &[], t);
            assert_eq!(lg.len(), cfg.vocab_size);
            assert!(lg.iter().all(|v| v.is_finite()), "step {t}: non-finite logit");
            // greedy feedback → exercises the loop like real generation.
            let (bi, _) = lg.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });
            tok = bi as u32;
        }
    }

    /// CPU STAGE ORACLE for the GPU-resident forward bisect (Mac, no Vulkan).
    ///
    /// Loads layers `[0, BOUND)` of the REAL checkpoint to f32 host (the exact
    /// `cpu_matmul` reference — NVFP4/FP8 dequantized at load, engine=None so
    /// every `nem_matvec`/`mamba_proj`/`latent_moe_mixer` takes its CPU fallback)
    /// and runs `forward_pp_stage(TOKEN, &[], 0)` — i.e. embed `TOKEN` + run the
    /// resident layers — producing the SAME `hidden_size` vector that PP stage 0
    /// ships across the first vCCL hop. Writes it f32-binary to
    /// `VLLM_NEM_ORACLE_OUT` so the cluster run's dumped stage-0 hidden
    /// (`STAGE_HIDDEN_DUMP` in pp_nemotron.py) can be diffed against it
    /// (scripts/diff_f32.py). A cos < ~0.99 localizes the bug INTO stage 0's GPU
    /// forward (layers `[0,BOUND)`); a match exonerates it and moves the bisect
    /// downstream. This is the ABSOLUTE check the split-invariance gate cannot
    /// give (a uniformly-wrong shader is split-invariant yet wrong).
    ///
    /// BOUND defaults to 22 (the PP-5 footprint split's stage-0 boundary
    /// [0,22,37,49,69,88]); TOKEN defaults to 9707 (the observed run's prompt).
    /// [0,22) dequants to ~51GB f32 host — fits this Mac (103GB), NOT a BC-250.
    ///
    ///   VLLM_TEST_NEMOTRON_DIR=/Volumes/Shared_Drive/models/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 \
    ///   VLLM_NEM_ORACLE_BOUND=22 VLLM_NEM_ORACLE_TOKEN=9707 \
    ///   VLLM_NEM_ORACLE_OUT=/tmp/nem_cpu_oracle.r0.hidden \
    ///   cargo test --lib -- --ignored --nocapture nemotron_cpu_stage_oracle
    #[test]
    #[ignore]
    fn nemotron_cpu_stage_oracle() {
        use std::io::Write;
        let dir = match std::env::var("VLLM_TEST_NEMOTRON_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("VLLM_TEST_NEMOTRON_DIR unset — skipping CPU stage oracle");
                return;
            }
        };
        let bound: usize = std::env::var("VLLM_NEM_ORACLE_BOUND")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(22);
        let token: u32 = std::env::var("VLLM_NEM_ORACLE_TOKEN")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(9707);
        let out_path = std::env::var("VLLM_NEM_ORACLE_OUT")
            .unwrap_or_else(|_| "/tmp/nem_cpu_oracle.r0.hidden".to_string());

        let d = std::path::Path::new(&dir);
        let cfg_json: Value = serde_json::from_str(
            &std::fs::read_to_string(d.join("config.json")).expect("read config.json"),
        ).expect("parse config.json");
        let config = NemotronConfig::from_json(&cfg_json).expect("parse NemotronConfig");
        let total = config.num_hidden_layers;
        assert!(bound <= total, "BOUND {bound} > num_hidden_layers {total}");
        let last = bound >= total;

        eprintln!(
            "CPU stage oracle: layers [0,{bound}) of {total}, token={token}, \
             block0={:?} (this dequants ~{:.0}GB f32 host)",
            config.block_specs[0],
            bound as f64 / total as f64 * 205.0,
        );

        let idx = d.join("model.safetensors.index.json");
        let t0 = std::time::Instant::now();
        let (weights, stats) = crate::nemotron_loader::load_nemotron_weights(
            &idx, &config, 0, bound, /*keep_embed*/ true, /*keep_lm*/ last,
        ).expect("load_nemotron_weights [0,bound)");
        eprintln!(
            "loaded {} tensors ({} fp8, {} nvfp4), {:.2}GB f32 in {:.1}s",
            stats.tensors_loaded, stats.fp8_tensors, stats.nvfp4_tensors,
            stats.bytes_f32 as f64 / 1e9, t0.elapsed().as_secs_f64(),
        );

        let mut m = NemotronModel::new_range(
            config.clone(), weights, 64, "lm_head.weight".to_string(), 0, bound,
        );
        // engine=None → every matvec is the f32 host cpu_matmul reference.
        assert!(m.engine.is_none() && m.gpu_weights.is_empty() && m.gpu_experts.is_empty(),
            "oracle must run the pure-CPU reference path (no resident GPU weights)");

        let ts = std::time::Instant::now();
        let hidden = m.forward_pp_stage(token, &[], 0);
        eprintln!("forward_pp_stage [0,{bound}) in {:.2}s → {} f32",
            ts.elapsed().as_secs_f64(), hidden.len());

        let expect_len = if last { config.vocab_size } else { config.hidden_size };
        assert_eq!(hidden.len(), expect_len, "oracle output width");
        let finite = hidden.iter().filter(|v| v.is_finite()).count();
        let l2 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
        let (mn, mx) = hidden.iter().fold((f32::INFINITY, f32::NEG_INFINITY),
            |(a, b), &v| (a.min(v), b.max(v)));
        eprintln!("hidden: finite={finite}/{} l2={l2:.4} min={mn:.4} max={mx:.4}", hidden.len());
        eprintln!("first8={:?}", &hidden[..8.min(hidden.len())]);
        assert_eq!(finite, hidden.len(), "CPU oracle produced non-finite output \
            — bug is in the CPU forward WIRING, not the GPU shaders");

        let bytes = crate::push_constants::f32_slice_to_bytes(&hidden);
        std::fs::File::create(&out_path)
            .and_then(|mut f| f.write_all(&bytes))
            .expect("write oracle output");
        eprintln!("wrote {} f32 → {out_path}\n  DIFF vs cluster stage-0 dump with: \
            python3 scripts/diff_f32.py {out_path} <STAGE_HIDDEN_DUMP.r0>", hidden.len());
    }
}
