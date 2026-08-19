// SPDX-License-Identifier: Apache-2.0
//! Block-type-aware NVFP4/FP8/BF16 weight loader for Nemotron-H-Puzzle
//! (`nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B`).
//!
//! Reuses the qwen3.6 NVFP4/FP8 CPU dequant machinery in `model.rs`
//! VERBATIM — `dequantize_nvfp4`, `dequantize_fp8`, and the shard-discovery
//! helper `discover_shards` are called as-is with no modification. They are
//! generic byte-slice-in/f32-out functions parameterized only by
//! (out_features, in_features, group_size) and carry no qwen-specific
//! naming or shape assumptions, so they apply unchanged to Nemotron's
//! differently-named, differently-shaped tensors. Everything in this file is
//! new: tensor-name detection/dispatch and the `NemotronConfig::block_specs`
//! driven per-layer keep/validate logic.
//!
//! ## Quant map
//! Derived from the checkpoint's own `config.json` →
//! `quantization_config.config_groups` (`quant_method: "modelopt"`) and
//! cross-checked against the real
//! `model.safetensors.index.json` at
//! `/Volumes/Shared_Drive/models/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4`:
//!
//!   - **group_0 — FP8-E4M3, per-tensor** (`num_bits: 8`): mamba
//!     `mixer.in_proj` / `mixer.out_proj`, and moe
//!     `mixer.shared_experts.{up,down}_proj`. On disk: `.weight` is
//!     `F8_E4M3 [out, in]`, `.weight_scale` is an `F32` SCALAR (shape `[]`,
//!     confirmed on the real checkpoint), `.input_scale` is an unused
//!     (W8A16) `F32` scalar. Feeds `dequantize_fp8` unchanged.
//!   - **group_1 — NVFP4, group_size 16** (`num_bits: 4`): moe routed
//!     `mixer.experts.{e}.{up,down}_proj` (40 MoE layers × up to 512 experts
//!     each — verified `n_routed_experts=512` on the real config, expert
//!     counts confirmed present per-layer in the index). On disk: `.weight`
//!     is `U8 [out, in/2]` (2 packed E2M1 nibbles/byte), `.weight_scale` is
//!     `F8_E4M3 [out, in/16]`, `.weight_scale_2` is an `F32` scalar (global
//!     double-scale). Feeds `dequantize_nvfp4` unchanged (verified
//!     `group_size == 16` from real shapes, e.g. `experts.0.up_proj.weight`
//!     `[1280,512]` (in=1024) with `weight_scale` `[1280,64]` → group_size
//!     `1024/64 == 16`).
//!   - **rest — BF16, passthrough**: attn `q/k/v/o_proj`, mamba
//!     `conv1d/A_log/D/dt_bias/norm`, moe
//!     `gate/e_score_correction_bias/fc1_latent_proj/fc2_latent_proj`,
//!     per-layer `norm.weight`, `backbone.embeddings.weight`,
//!     `backbone.norm_f.weight`, `lm_head.weight`. KV-cache scales
//!     `mixer.k_proj.k_scale` / `mixer.v_proj.v_scale` are standalone `F32`
//!     scalars (not siblings of a `.weight` tensor) — they pass straight
//!     through the plain decode path like any other non-quantized tensor;
//!     the current CPU forward doesn't consume them yet (fp16 KV cache is
//!     not implemented), so they just ride along in the weight map.
//!
//! `conv1d.weight` is `[conv_dim, 1, kernel]` on disk (confirmed
//! `[9728, 1, 4]` for a mamba layer's `mixer.conv1d.weight` on the real
//! checkpoint, matching `NemotronConfig::mamba_dims().conv_dim() == 9728`);
//! the middle dim is always 1, so squeezing it is a shape-only no-op on the
//! flattened `Vec<f32>` — [`squeeze_conv1d_middle_dim`] only asserts the
//! expected shape rather than doing any data movement.
//!
//! Every `.weight`/`.weight_scale`/`.weight_scale_2` sibling triple in the
//! real checkpoint's index lives in the SAME shard file (checked across all
//! 41,120 quantized tensors in `model.safetensors.index.json` — zero
//! cross-shard splits), so unlike the qwen3.6 loader's
//! `find_quant_sibling_*` fallback, this loader can look up siblings via a
//! plain in-shard `SafeTensors::tensor()` call and treat a missing sibling
//! as a hard error (an actual corrupt/incomplete checkpoint) rather than
//! needing a cross-shard search.

use std::collections::HashMap;
use std::path::Path;

use crate::compute::{Buffer, ComputeEngine};
use crate::model::{dequantize_fp8, dequantize_nvfp4, ModelWeights, SimpleTensor};
use crate::nemotron::{BlockSpec, NemGpuWeight, NemMoeExperts, NemQuant, NemotronConfig};
use crate::push_constants::nvfp4_fold_scales;

/// NVFP4 block size for the Nemotron routed experts (`group_size` from the
/// checkpoint's `quantization_config`, cross-checked in the loader module docs
/// against the real `experts.0.up_proj` shapes). Used to pre-size the resident
/// expert scale buffers; each streamed tensor's derived group size is asserted
/// to match.
pub const NVFP4_MOE_GROUP_SIZE: usize = 16;

/// Decode a plain (non-quantized) safetensors view to f32. Covers every
/// Nemotron tensor outside the FP8/NVFP4 groups: BF16 (the overwhelming
/// majority), F32 (e.g. the `k_scale`/`v_scale`/`e_score_correction_bias`
/// scalars and vectors), and F16 (not present in this checkpoint but
/// harmless to support).
fn decode_plain(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>, String> {
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
        other => return Err(format!("unsupported plain dtype {other:?} in Nemotron loader")),
    })
}

/// `conv1d.weight` is `[conv_dim, 1, kernel]` on disk; the Mamba2 kernel
/// wants a flat `[conv_dim, kernel]` (see `Mamba2Weights::conv1d_weight`).
/// Since the middle dim is 1, the flattened byte/element order is IDENTICAL
/// for `[conv_dim,1,kernel]` and `[conv_dim,kernel]` — this only validates
/// the expected shape (catching a checkpoint surprise) and passes the data
/// through unchanged.
fn squeeze_conv1d_middle_dim(shape: &[usize], data: Vec<f32>) -> Result<Vec<f32>, String> {
    match shape {
        [_conv_dim, 1, _kernel] => Ok(data),
        [_conv_dim, _kernel] => Ok(data), // already 2D (defensive)
        other => Err(format!(
            "conv1d.weight: expected [conv_dim,1,kernel] or [conv_dim,kernel], got {other:?}"
        )),
    }
}

/// GLOBAL layer index for a `backbone.layers.{i}.*` tensor name, else `None`.
fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("backbone.layers.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|s| s.parse::<usize>().ok())
}

/// Validate that a `backbone.layers.{i}.mixer.*` tensor name is consistent
/// with that layer's declared `BlockSpec` — e.g. a Mamba layer must never
/// carry `.mixer.q_proj` (that's Attention-only), an Attention layer must
/// never carry `.mixer.in_proj` (Mamba-only) or `.mixer.experts.*` (Moe
/// -only). This is NOT how the loader decides what to load (the checkpoint
/// simply doesn't contain tensors a layer's mixer doesn't have — the loader
/// only ever processes tensors that exist) — it's a defensive early-error
/// against a `config.json`/checkpoint mismatch (wrong `block_configs`,
/// mismatched files, etc). Layer-level tensors shared by all block kinds
/// (`backbone.layers.{i}.norm.weight`, i.e. no `.mixer.` in the name) always
/// pass.
fn check_tensor_matches_block_spec(name: &str, spec: BlockSpec) -> Result<(), String> {
    let mixer_suffix = match name.split_once(".mixer.") {
        Some((_, rest)) => rest,
        None => return Ok(()),
    };
    let ok = match spec {
        BlockSpec::Mamba => {
            mixer_suffix.starts_with("in_proj")
                || mixer_suffix.starts_with("out_proj")
                || mixer_suffix.starts_with("conv1d")
                || mixer_suffix == "A_log"
                || mixer_suffix == "D"
                || mixer_suffix == "dt_bias"
                || mixer_suffix.starts_with("norm.")
        }
        BlockSpec::Attention => {
            mixer_suffix.starts_with("q_proj")
                || mixer_suffix.starts_with("k_proj")
                || mixer_suffix.starts_with("v_proj")
                || mixer_suffix.starts_with("o_proj")
        }
        BlockSpec::Moe { .. } => {
            mixer_suffix.starts_with("experts.")
                || mixer_suffix.starts_with("shared_experts.")
                || mixer_suffix.starts_with("gate.")
                || mixer_suffix.starts_with("fc1_latent_proj")
                || mixer_suffix.starts_with("fc2_latent_proj")
        }
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "tensor '{name}' (.mixer.{mixer_suffix}) doesn't match its layer's declared \
             block_type {spec:?} — config.json/checkpoint mismatch?"
        ))
    }
}

/// Load-time counters, surfaced for logging/validation (how many tensors
/// took each dequant path).
#[derive(Debug, Default, Clone, Copy)]
pub struct NemotronLoadStats {
    pub tensors_loaded: usize,
    pub fp8_tensors: usize,
    pub nvfp4_tensors: usize,
    pub bytes_f32: u64,
}

/// Stream every relevant tensor from the Nemotron-H-Puzzle checkpoint at
/// `path`, dequantizing per the group_0 (FP8) / group_1 (NVFP4) / rest
/// (BF16 passthrough) map above, and return a `ModelWeights` ready for
/// `NemotronModel::new_range`/`new`.
///
/// Only tensors whose GLOBAL layer index falls in `[layer_start, layer_end)`
/// are loaded (pipeline-parallel-ready, mirroring
/// `model::load_qwen35_weights_split`); non-layer tensors
/// (`backbone.embeddings.weight`, `backbone.norm_f.weight`,
/// `lm_head.weight`) are gated by `keep_embed`/`keep_lm`. The MTP head
/// (`mtp.*`) is out of scope for this loader (deferred — the base model has
/// no forward-path use for it yet) and is always skipped.
///
/// Block-type-aware: `config.block_specs[layer_idx]` drives
/// [`check_tensor_matches_block_spec`], which errors out early on a
/// checkpoint/config mismatch. A layer's ABSENT tensor kinds (e.g. no
/// `q_proj` on a Mamba layer) are never an error — the loader only reacts to
/// tensors that are actually present in the checkpoint; nothing enumerates
/// an expected tensor set up front.
pub fn load_nemotron_weights(
    path: &Path,
    config: &NemotronConfig,
    layer_start: usize,
    layer_end: usize,
    keep_embed: bool,
    keep_lm: bool,
) -> Result<(ModelWeights, NemotronLoadStats), String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;

    // PP pre-slice load lever (VLLM_VULKAN_PP_PRESLICED_DIR): when set, open ONLY
    // this stage's pre-sliced file (bounds-guarded) instead of the whole
    // multi-shard checkpoint; unset ⇒ identical to discover_shards(path).
    let shards = crate::model::resolve_pp_stage_shards(path, layer_start, layer_end)?;

    let keep = |name: &str| -> bool {
        if name == "backbone.embeddings.weight" {
            return keep_embed;
        }
        if name == "lm_head.weight" || name == "backbone.norm_f.weight" {
            return keep_lm;
        }
        if name.starts_with("mtp.") {
            return false;
        }
        match layer_of(name) {
            Some(idx) => idx >= layer_start && idx < layer_end,
            None => false, // unrecognized top-level tensor: not part of this model
        }
    };

    let mut out: HashMap<String, SimpleTensor> = HashMap::new();
    let mut stats = NemotronLoadStats::default();

    for shard in &shards {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;

        for (name, view) in st.tensors() {
            if !keep(&name) {
                continue;
            }
            // Sibling scale/activation tensors are consumed alongside their
            // `.weight`, never loaded standalone.
            if name.ends_with(".weight_scale")
                || name.ends_with(".weight_scale_2")
                || name.ends_with(".input_scale")
            {
                continue;
            }

            let is_weight = name.ends_with(".weight");
            let base = if is_weight {
                Some(&name[..name.len() - ".weight".len()])
            } else {
                None
            };

            let is_nvfp4 = is_weight
                && view.dtype() == safetensors::Dtype::U8
                && st.tensor(&format!("{}.weight_scale", base.unwrap())).is_ok();
            let is_fp8 = is_weight
                && view.dtype() == safetensors::Dtype::F8_E4M3
                && st.tensor(&format!("{}.weight_scale", base.unwrap())).is_ok();

            let mut data = if is_nvfp4 {
                let base = base.unwrap();
                let wscale_view = st
                    .tensor(&format!("{base}.weight_scale"))
                    .map_err(|e| format!("{name}: missing weight_scale sibling: {e}"))?;
                let global_view = st
                    .tensor(&format!("{base}.weight_scale_2"))
                    .map_err(|e| format!("{name}: missing weight_scale_2 sibling: {e}"))?;
                let global = f32::from_le_bytes(
                    global_view.data()[..4]
                        .try_into()
                        .map_err(|_| format!("{name}: weight_scale_2 too short"))?,
                );
                let out_features = view.shape()[0];
                let in_features = view.shape()[1] * 2; // 2 nibbles/byte
                let groups = (wscale_view.data().len() / out_features).max(1);
                let group_size = in_features / groups;
                stats.nvfp4_tensors += 1;
                dequantize_nvfp4(
                    view.data(),
                    wscale_view.data(),
                    global,
                    out_features,
                    in_features,
                    group_size,
                )
            } else if is_fp8 {
                let base = base.unwrap();
                let sview = st
                    .tensor(&format!("{base}.weight_scale"))
                    .map_err(|e| format!("{name}: missing weight_scale sibling: {e}"))?;
                let scale = decode_plain(&sview)?;
                let out_features = view.shape()[0];
                let in_features = view.shape()[1];
                stats.fp8_tensors += 1;
                dequantize_fp8(view.data(), &scale, out_features, in_features)
            } else {
                decode_plain(&view)?
            };

            if name.ends_with("mixer.conv1d.weight") {
                data = squeeze_conv1d_middle_dim(view.shape(), data)?;
            }

            if let Some(idx) = layer_of(&name) {
                if idx < config.block_specs.len() {
                    check_tensor_matches_block_spec(&name, config.block_specs[idx])?;
                }
            }

            stats.bytes_f32 += (data.len() * 4) as u64;
            stats.tensors_loaded += 1;
            out.insert(name, SimpleTensor { data, shape: vec![] });
        }
    }

    Ok((ModelWeights { tensors: out }, stats))
}

/// Projected GPU-resident + host footprint for a layer range `[start, end)`,
/// in bytes, split by dequant path. This is the memory-fit model that decides
/// whether a PP stage fits the 13.3GB BC-250 GTT budget: the resident loader
/// keeps weights QUANTIZED (4-bit NVFP4 experts, 8-bit FP8 mamba/attn/shared,
/// f16 for the BF16-native attn/latent projections) instead of dequantizing to
/// f32 host, so `gpu_resident_bytes` is ~the on-disk quantized size (plus the
/// f32-folded NVFP4 scale expansion) rather than the ~41GB/PP-5-stage f32-host
/// blow-up that OOM-killed the baseline.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResidentFootprint {
    /// NVFP4 routed experts: packed nibbles (in/2 bytes/row) + folded f32
    /// scales (in/group_size * 4 bytes/row).
    pub nvfp4_expert_bytes: u64,
    /// FP8 mamba in/out_proj + moe shared_experts up/down: raw fp8 bytes + a
    /// scalar f32 scale.
    pub fp8_bytes: u64,
    /// f16-resident BF16-native matmul weights: attn q/k/v/o, moe fc1/fc2, and
    /// (last stage) lm_head.
    pub f16_bytes: u64,
    /// f32 host tensors (norms, conv1d, A_log, D, dt_bias, gate/router, kv
    /// scales, and — first stage — the embedding table for the row lookup).
    pub host_f32_bytes: u64,
}

impl ResidentFootprint {
    /// Total GPU-resident (GTT) bytes — the quantity gated against the node's
    /// GTT budget.
    pub fn gpu_resident_bytes(&self) -> u64 {
        self.nvfp4_expert_bytes + self.fp8_bytes + self.f16_bytes
    }
    /// Resident + host (the anon-rss the loader materializes on the node).
    pub fn total_bytes(&self) -> u64 {
        self.gpu_resident_bytes() + self.host_f32_bytes
    }
}

/// Project the resident footprint for `[start, end)` from `config` dims alone
/// (no checkpoint read) — the sizing model behind the PP-depth decision. Mirrors
/// exactly what [`load_nemotron_resident`] uploads/keeps per weight.
pub fn resident_footprint(
    config: &NemotronConfig,
    start: usize,
    end: usize,
    keep_embed: bool,
    keep_lm: bool,
) -> ResidentFootprint {
    let gs = NVFP4_MOE_GROUP_SIZE as u64;
    let hidden = config.hidden_size as u64;
    let ne = config.n_routed_experts as u64;
    let latent = config.moe_latent_size as u64;
    let mamba = config.mamba_dims();
    let in_proj_out = mamba.in_proj_out() as u64;
    let inter = mamba.intermediate() as u64;
    let conv_dim = mamba.conv_dim() as u64;
    let nh_mamba = config.mamba_num_heads as u64;
    let q_dim = (config.num_attention_heads * config.head_dim) as u64;
    let kv_dim = (config.num_key_value_heads * config.head_dim) as u64;

    // NVFP4 packed = params/2 bytes; folded scales = (params/group_size)*4 bytes.
    let nvfp4 = |out: u64, in_: u64| out * (in_ / 2) + out * (in_ / gs) * 4;
    // FP8 = params bytes + one f32 scalar scale.
    let fp8 = |out: u64, in_: u64| out * in_ + 4;
    // f16 = params*2 bytes.
    let f16 = |out: u64, in_: u64| out * in_ * 2;

    let mut fp = ResidentFootprint::default();
    for g in start..end {
        match config.block_specs[g] {
            BlockSpec::Mamba => {
                // in_proj [in_proj_out, hidden] + out_proj [hidden, inter] FP8.
                fp.fp8_bytes += fp8(in_proj_out, hidden) + fp8(hidden, inter);
                // conv1d [conv_dim, kernel] + A_log/D/dt_bias [nh] + norm [inter]
                // + layer norm [hidden] → f32 host.
                fp.host_f32_bytes += (conv_dim * config.conv_kernel as u64
                    + 3 * nh_mamba
                    + inter
                    + hidden)
                    * 4;
            }
            BlockSpec::Attention => {
                // q/k/v/o BF16 → f16 resident.
                fp.f16_bytes += f16(q_dim, hidden) + 2 * f16(kv_dim, hidden) + f16(hidden, q_dim);
                // k_scale/v_scale scalars + layer norm → f32 host.
                fp.host_f32_bytes += (2 + hidden) * 4;
            }
            BlockSpec::Moe { moe_intermediate_size, .. } => {
                let mi = moe_intermediate_size as u64;
                let shared_inter = config.moe_shared_expert_intermediate_size as u64;
                // Routed experts up [mi, latent] + down [latent, mi] × ne, NVFP4.
                fp.nvfp4_expert_bytes += ne * (nvfp4(mi, latent) + nvfp4(latent, mi));
                // Shared experts up [shared_inter, hidden] + down [hidden, shared_inter] FP8.
                fp.fp8_bytes += fp8(shared_inter, hidden) + fp8(hidden, shared_inter);
                // fc1 [latent, hidden] + fc2 [hidden, latent] BF16 → f16.
                fp.f16_bytes += f16(latent, hidden) + f16(hidden, latent);
                // gate [ne, hidden] + e_score_bias [ne] + layer norm [hidden] → f32 host.
                fp.host_f32_bytes += (ne * hidden + ne + hidden) * 4;
            }
        }
    }
    if keep_embed {
        // Embedding table stays f32 host (row lookup on the first stage).
        fp.host_f32_bytes += config.vocab_size as u64 * hidden * 4;
    }
    if keep_lm {
        // lm_head BF16 → f16 resident + norm_f f32 host.
        fp.f16_bytes += f16(config.vocab_size as u64, hidden);
        fp.host_f32_bytes += hidden * 4;
    }
    fp
}

/// Load-time counters for the resident path (mirrors `NemotronLoadStats` but
/// counts GPU-resident bytes actually uploaded).
#[derive(Debug, Default, Clone, Copy)]
pub struct NemotronResidentStats {
    pub nvfp4_expert_tensors: usize,
    pub fp8_tensors: usize,
    pub f16_tensors: usize,
    pub host_tensors: usize,
    pub gpu_resident_bytes: u64,
    pub host_bytes: u64,
}

/// True for the BF16-native `nn.Linear` weights that are uploaded as f16-
/// resident (routed through `nem_matvec`'s plain f16 matvec): attention
/// q/k/v/o, the latent-MoE fc1/fc2 projections, and lm_head. Everything else
/// BF16 (conv1d, A_log, D, dt_bias, norms, router gate, e_score bias,
/// embeddings, k/v scales) stays f32 host.
fn is_bf16_resident_matmul(name: &str) -> bool {
    if name == "lm_head.weight" {
        return true;
    }
    match name.split_once(".mixer.") {
        Some((_, s)) => {
            s == "q_proj.weight"
                || s == "k_proj.weight"
                || s == "v_proj.weight"
                || s == "o_proj.weight"
                || s == "fc1_latent_proj.weight"
                || s == "fc2_latent_proj.weight"
        }
        None => false,
    }
}

/// Parse `backbone.layers.{i}.mixer.experts.{e}.{up|down}_proj.weight` into
/// `(layer, expert, is_up)`, else `None`.
fn parse_expert(name: &str) -> Option<(usize, usize, bool)> {
    let layer = layer_of(name)?;
    let s = name.split_once(".mixer.experts.")?.1;
    let (e_str, rest) = s.split_once('.')?;
    let e: usize = e_str.parse().ok()?;
    let is_up = if rest == "up_proj.weight" {
        true
    } else if rest == "down_proj.weight" {
        false
    } else {
        return None;
    };
    Some((layer, e, is_up))
}

/// Alloc a host-coherent GPU buffer and copy `bytes` into it. Host-coherent
/// (not device-local) because the large concatenated expert / lm_head buffers
/// exceed GFX1013's device-local carveout, and on the UMA BC-250 host-coherent
/// GTT is GPU-readable at no DMA penalty (see the qwen lm_head note).
fn upload(engine: &mut ComputeEngine, bytes: &[u8]) -> Result<Buffer, String> {
    let buf = engine.alloc_host_coherent_storage(bytes.len() as u64)?;
    buf.write(bytes)?;
    Ok(buf)
}

/// Copy `bytes` into an already-allocated host-coherent buffer at `byte_off`
/// (used to fill a per-layer concatenated expert buffer one expert at a time).
fn write_at(buf: &Buffer, byte_off: usize, bytes: &[u8]) {
    let ptr = buf.mapped_ptr.expect("host-coherent buffer is mapped") as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(byte_off), bytes.len());
    }
}

/// GPU-RESIDENT streaming loader (the 75B OOM fix). Keeps every matmul weight
/// in its native quantized form uploaded GPU-resident and never materializes
/// the f32-expanded weight set on the host: NVFP4 routed experts (packed
/// nibbles + folded f32 scales, concatenated per MoE layer for per-expert
/// dispatch), FP8 mamba in/out_proj + moe shared_experts (raw fp8 + scalar
/// scale), f16 for the BF16-native attn q/k/v/o + latent fc1/fc2 + lm_head, and
/// f32 host only for the small recurrence/router/norm tensors (+ the first
/// stage's embedding table). Fills `nem`'s `gpu_weights` / `gpu_experts` /
/// `weights` (host) in place; `nem.engine` must already hold the passed engine.
///
/// Peak host RAM ≈ the resident footprint + one streamed tensor (never the f32
/// blow-up), so a PP stage fits the node GTT — see [`resident_footprint`].
pub fn load_nemotron_resident(
    path: &Path,
    config: &NemotronConfig,
    engine: &mut ComputeEngine,
    gpu_weights: &mut HashMap<String, NemGpuWeight>,
    gpu_experts: &mut HashMap<usize, NemMoeExperts>,
    host: &mut HashMap<String, SimpleTensor>,
    layer_start: usize,
    layer_end: usize,
    keep_embed: bool,
    keep_lm: bool,
    tp_rank: usize,
    tp_size: usize,
) -> Result<NemotronResidentStats, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;

    let mut stats = NemotronResidentStats::default();
    let gs = NVFP4_MOE_GROUP_SIZE;
    let latent = config.moe_latent_size;
    let ne = config.n_routed_experts;
    // TP=2×PP: this rank's EP owned-expert range (whole-expert partition of the
    // routed experts — the dominant footprint) + local expert count. tp_size==1
    // => own all experts (owned_lo=0, ne_local=ne), byte-identical to the ship
    // PP-only path.
    let (owned_lo, ne_local) = if tp_size > 1 {
        crate::nemotron_tp::expert_owned_range(ne, tp_rank, tp_size)
    } else {
        (0, ne)
    };
    let tp_shard = tp_size > 1;

    // Opt-in per-(pp,tp) per-tensor-class LOAD PROGRESS trace
    // (`VLLM_VULKAN_NEMOTRON_LOAD_PROGRESS=1`). Emits a FLUSHED stderr line at
    // every large resident/host allocation so that, if a rank SIGSEGVs mid-load
    // (e.g. a host-coherent GTT overcommit faulting on first write when a stage
    // exceeds the node's UMA/GTT budget — the failure mode on the vocab-tensor
    // stages 0/2), the LAST printed line pins the exact allocation. Uses
    // `eprintln!` (unbuffered) rather than `log::*` (may buffer past the fault).
    // Default-off ⇒ byte-for-byte identical to the shipped path when unset.
    let progress = std::env::var("VLLM_VULKAN_NEMOTRON_LOAD_PROGRESS").ok().as_deref() == Some("1");
    macro_rules! plog {
        ($($a:tt)*) => {
            if progress {
                eprintln!("[nem-load pp={layer_start}..{layer_end} tp={tp_rank}/{tp_size}] {}", format!($($a)*));
            }
        };
    }
    plog!("START keep_embed={keep_embed} keep_lm={keep_lm} ne={ne} ne_local={ne_local} owned_lo={owned_lo}");

    // E4M3-RESIDENT NVFP4 expert scales (VLLM_VULKAN_NVFP4_E4M3_SCALES): keep the
    // raw on-disk `.weight_scale` e4m3 bytes resident (1 byte/group) instead of
    // folding e4m3*weight_scale_2 into an f32/group — 4x smaller scale buffers,
    // the per-expert `.weight_scale_2` global carried separately and re-applied
    // in-shader by `mul_mat_vec_nvfp4_e4m3_f32_f32`. Default OFF (f32-fold stays
    // the oracle); bit-exact to fold (nvfp4_e4m3_resident_matches_f32_fold). This
    // is the ≥150B/nemotron-tier port of the shared-loader e4m3 branch.
    let nvfp4_e4m3 = crate::flags::flags_global().nvfp4_e4m3_scales;

    // Pre-allocate the concatenated per-MoE-layer expert buffers so each expert
    // tensor can be written at its `packed_off`/`sb_off` slice during streaming.
    // Under EP only THIS rank's `ne_local` owned experts are allocated/held.
    for g in layer_start..layer_end {
        if let BlockSpec::Moe { moe_intermediate_size: mi, .. } = config.block_specs[g] {
            // up:   [ne_local, mi, latent/2] packed u8 + [ne_local, mi, latent/gs] scale
            // down: [ne_local, latent, mi/2] packed u8 + [ne_local, latent, mi/gs] scale
            // scale is 1 byte/group (e4m3-resident) or 4 bytes/group (f32-fold).
            let sbpg = if nvfp4_e4m3 { 1 } else { 4 };
            let up_packed = ne_local * mi * (latent / 2);
            let up_scale = ne_local * mi * (latent / gs) * sbpg;
            let down_packed = ne_local * latent * (mi / 2);
            let down_scale = ne_local * latent * (mi / gs) * sbpg;
            let up = engine.alloc_host_coherent_storage(up_packed as u64)?;
            let up_scales = engine.alloc_host_coherent_storage(up_scale as u64)?;
            let down = engine.alloc_host_coherent_storage(down_packed as u64)?;
            let down_scales = engine.alloc_host_coherent_storage(down_scale as u64)?;
            stats.gpu_resident_bytes +=
                (up_packed + up_scale + down_packed + down_scale) as u64;
            plog!(
                "prealloc MoE L{g}: +{} MB experts (cum GTT {} MB, e4m3={})",
                (up_packed + up_scale + down_packed + down_scale) >> 20,
                stats.gpu_resident_bytes >> 20,
                nvfp4_e4m3
            );
            gpu_experts.insert(
                g,
                NemMoeExperts {
                    n_experts: ne_local,
                    up,
                    up_scales,
                    up_out: mi,
                    up_in: latent,
                    down,
                    down_scales,
                    down_out: latent,
                    down_in: mi,
                    group_size: gs as u32,
                    e4m3: nvfp4_e4m3,
                    up_globals: vec![1.0f32; ne_local],
                    down_globals: vec![1.0f32; ne_local],
                },
            );
        }
    }

    // PP pre-slice load lever (VLLM_VULKAN_PP_PRESLICED_DIR): open ONLY this
    // stage's pre-sliced file when set (bounds-guarded); unset ⇒ discover_shards.
    let shards = crate::model::resolve_pp_stage_shards(path, layer_start, layer_end)?;
    let keep = |name: &str| -> bool {
        if name == "backbone.embeddings.weight" {
            return keep_embed;
        }
        if name == "lm_head.weight" || name == "backbone.norm_f.weight" {
            return keep_lm;
        }
        if name.starts_with("mtp.") {
            return false;
        }
        match layer_of(name) {
            Some(idx) => idx >= layer_start && idx < layer_end,
            None => false,
        }
    };

    for shard in &shards {
        let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;

        for (name, view) in st.tensors() {
            if !keep(&name) {
                continue;
            }
            if name.ends_with(".weight_scale")
                || name.ends_with(".weight_scale_2")
                || name.ends_with(".input_scale")
            {
                continue;
            }
            if let Some(idx) = layer_of(&name) {
                if idx < config.block_specs.len() {
                    check_tensor_matches_block_spec(&name, config.block_specs[idx])?;
                }
            }

            let is_weight = name.ends_with(".weight");
            let base = if is_weight { Some(&name[..name.len() - ".weight".len()]) } else { None };
            let is_nvfp4 = is_weight
                && view.dtype() == safetensors::Dtype::U8
                && st.tensor(&format!("{}.weight_scale", base.unwrap())).is_ok();
            let is_fp8 = is_weight
                && view.dtype() == safetensors::Dtype::F8_E4M3
                && st.tensor(&format!("{}.weight_scale", base.unwrap())).is_ok();

            if is_nvfp4 {
                // Routed expert → write packed nibbles + folded scales into the
                // pre-allocated per-layer concat buffer at this expert's offset.
                let (layer, e, is_up) = parse_expert(&name).ok_or_else(|| {
                    format!("NVFP4 tensor '{name}' is not a routed expert up/down_proj")
                })?;
                // EP=2: skip experts owned by the peer; write owned ones at the
                // LOCAL index into this rank's half-sized concat buffer.
                if tp_shard && (e < owned_lo || e >= owned_lo + ne_local) {
                    continue;
                }
                let e = e - owned_lo; // local expert id (owned_lo=0 when tp_size==1)
                let base = base.unwrap();
                let wscale = st
                    .tensor(&format!("{base}.weight_scale"))
                    .map_err(|e| format!("{name}: missing weight_scale: {e}"))?;
                let global = f32::from_le_bytes(
                    st.tensor(&format!("{base}.weight_scale_2"))
                        .map_err(|e| format!("{name}: missing weight_scale_2: {e}"))?
                        .data()[..4]
                        .try_into()
                        .map_err(|_| format!("{name}: weight_scale_2 too short"))?,
                );
                let out_f = view.shape()[0];
                let in_f = view.shape()[1] * 2; // 2 nibbles/byte
                let groups = (wscale.data().len() / out_f).max(1);
                if in_f / groups != gs {
                    return Err(format!(
                        "{name}: NVFP4 group_size {} != expected {gs}",
                        in_f / groups
                    ));
                }
                let ex = gpu_experts
                    .get(&layer)
                    .ok_or_else(|| format!("{name}: no pre-allocated experts for layer {layer}"))?;
                let (wbuf, sbuf, off_out, off_in) = if is_up {
                    (&ex.up, &ex.up_scales, ex.up_out, ex.up_in)
                } else {
                    (&ex.down, &ex.down_scales, ex.down_out, ex.down_in)
                };
                debug_assert_eq!((out_f, in_f), (off_out, off_in), "{name}: expert shape");
                let packed_byte_off = e * out_f * (in_f / 2);
                write_at(wbuf, packed_byte_off, view.data());
                // Scale residency: e4m3-resident stores the RAW on-disk e4m3
                // `.weight_scale` bytes VERBATIM (1 byte/group) + carries the
                // per-tensor `.weight_scale_2` global separately; f32-fold folds
                // e4m3*global into an f32/group. Byte-exact per-expert slice into
                // the concat scale buffer either way (byte-elem == f32-elem index
                // since e4m3 is 1 byte where fold is 1 f32).
                if nvfp4_e4m3 {
                    let scale_byte_off = e * out_f * (in_f / gs); // 1 byte/group
                    write_at(sbuf, scale_byte_off, wscale.data());
                } else {
                    let folded = nvfp4_fold_scales(wscale.data(), global); // [out*groups] f32
                    let scale_byte_off = e * out_f * (in_f / gs) * 4;
                    write_at(sbuf, scale_byte_off, &crate::push_constants::f32_slice_to_bytes(&folded));
                }
                // Record this expert's per-tensor global (used only by the e4m3
                // dispatch; harmless 1.0-overwrite on the fold path). Separate
                // mutable borrow AFTER the immutable buffer-write borrow ends.
                if nvfp4_e4m3 {
                    let exm = gpu_experts
                        .get_mut(&layer)
                        .ok_or_else(|| format!("{name}: no experts for layer {layer}"))?;
                    if is_up { exm.up_globals[e] = global } else { exm.down_globals[e] = global }
                }
                stats.nvfp4_expert_tensors += 1;
            } else if is_fp8 {
                let base = base.unwrap();
                let scale = decode_plain(
                    &st.tensor(&format!("{base}.weight_scale"))
                        .map_err(|e| format!("{name}: missing weight_scale: {e}"))?,
                )?;
                let out_f = view.shape()[0];
                let in_f = view.shape()[1];
                let is_mamba_proj = name.ends_with(".mixer.in_proj.weight")
                    || name.ends_with(".mixer.out_proj.weight");
                let is_shared_proj = name.ends_with(".shared_experts.up_proj.weight")
                    || name.ends_with(".shared_experts.down_proj.weight");
                // TP=2×PP: force fp8->q8 requant for the sharded SHARED-expert
                // projections so they route through the f32 shard staging
                // (nem_tp_shard_full, col up / row down). Mamba proj is
                // REPLICATED under this TP scope (nem_tp_shard_full returns it
                // unchanged), so it is NOT force-requanted here — the MAMBA_Q8
                // flag alone governs it, exactly as in the PP-only ship stack.
                let requant_q8 = (crate::flags::flags_global().nemotron_mamba_q8 && is_mamba_proj)
                    || (crate::flags::flags_global().nemotron_shared_q8 && is_shared_proj)
                    || (tp_shard && is_shared_proj);
                if requant_q8 {
                    // Requant FP8 -> q8_0 at load: dequant to f32 (existing
                    // per-row/per-tensor scale rules), re-quantize per-row into
                    // GGUF q8_0 blocks (scale lives in-block, so no side buffer),
                    // upload raw q8_0 bytes. `flat_map_iter` over `par_chunks`
                    // preserves row order (blocks are independent, each row is
                    // whole 32-element blocks since in_f%32==0: mamba proj
                    // in_f=hidden, and the shared_experts proj shapes are
                    // up in_f=4096 (=128x32) / down in_f=5376 (=168x32)), so
                    // the parallel output is byte-identical to serial.
                    let deq = crate::model::dequantize_fp8(view.data(), &scale, out_f, in_f);
                    // TP=2×PP: shard the DEQUANTIZED f32 (mamba in_proj 5-seg
                    // col-shard / out_proj row-shard / shared up col / down row),
                    // then requant per LOCAL row. Row-parallel weights shrink the
                    // contraction dim → assert it stays q8_0-block(32)-aligned.
                    let (deq, out_l, in_l) = if tp_shard {
                        let s = crate::nemotron_tp::nem_tp_shard_full(&name, deq, config, tp_rank, tp_size);
                        let (ol, il) = crate::nemotron_tp::nem_tp_local_shape(&name, out_f, in_f, config, tp_size);
                        assert_eq!(il % 32, 0,
                            "{name}: TP row-shard in_features {il} not q8_0-block(32)-aligned");
                        (s, ol, il)
                    } else {
                        (deq, out_f, in_f)
                    };
                    use rayon::prelude::*;
                    let q8: Vec<u8> = deq
                        .par_chunks(in_l)
                        .flat_map_iter(|row| crate::model::quantize_q8_0(row))
                        .collect();
                    let wbuf = upload(engine, &q8)?;
                    stats.gpu_resident_bytes += q8.len() as u64;
                    stats.fp8_tensors += 1;
                    if q8.len() >= (16 << 20) {
                        plog!("q8 {name}: +{} MB (cum GTT {} MB)",
                            q8.len() >> 20, stats.gpu_resident_bytes >> 20);
                    }
                    gpu_weights.insert(
                        name.clone(),
                        NemGpuWeight {
                            buffer: wbuf,
                            quant: NemQuant::Q8_0,
                            out_features: out_l,
                            in_features: in_l,
                        },
                    );
                    continue;
                }
                let wbuf = upload(engine, view.data())?;
                let sbuf = upload(engine, &crate::push_constants::f32_slice_to_bytes(&scale))?;
                let per_row = scale.len() > 1;
                stats.gpu_resident_bytes += (view.data().len() + scale.len() * 4) as u64;
                stats.fp8_tensors += 1;
                gpu_weights.insert(
                    name.clone(),
                    NemGpuWeight {
                        buffer: wbuf,
                        quant: NemQuant::Fp8 { scale: sbuf, per_row },
                        out_features: out_f,
                        in_features: in_f,
                    },
                );
            } else if is_weight && is_bf16_resident_matmul(&name) {
                // BF16 → f16 resident (plain matvec: attn q/k/v/o, fc1/fc2, and —
                // last stage — lm_head [vocab, hidden], the single largest f16).
                let f32w = decode_plain(&view)?;
                let out_f0 = view.shape()[0];
                let in_f0 = *view.shape().get(1).unwrap_or(&1);
                // TP=2×PP: shard the f32 (attn q/k/v col by head + o row; fc1/fc2
                // + lm_head replicated) BEFORE the f16 re-encode. Column-parallel
                // shrinks out, row-parallel shrinks in; nem_tp_local_shape mirrors
                // it. lm_head/fc1/fc2 return unchanged (replicated).
                let (f32w, out_l, in_l) = if tp_shard {
                    let s = crate::nemotron_tp::nem_tp_shard_full(&name, f32w, config, tp_rank, tp_size);
                    let (ol, il) = crate::nemotron_tp::nem_tp_local_shape(&name, out_f0, in_f0, config, tp_size);
                    (s, ol, il)
                } else {
                    (f32w, out_f0, in_f0)
                };
                // Encode f16 DIRECTLY into the resident GPU buffer's mapped
                // memory (host-coherent = plain UMA RAM on the BC-250), then drop
                // the f32 staging. This avoids the previous THIRD large transient
                // (a separate `Vec<u8>` of the same size as the buffer): for the
                // last stage's lm_head that removed a ~1GB peak that stacked on
                // top of the 2.1GB f32 staging + the 1GB GPU buffer during upload
                // — the exact transient that pushes an at-budget vocab stage over
                // its UMA/GTT envelope. Byte-identical to the old encode+upload.
                let nbytes = f32w.len() * 2;
                let buf = engine.alloc_host_coherent_storage(nbytes as u64)?;
                {
                    let dst = buf.mapped_ptr.expect("host-coherent buffer is mapped") as *mut u8;
                    for (i, &v) in f32w.iter().enumerate() {
                        let b = half::f16::from_f32(v).to_bits().to_le_bytes();
                        unsafe {
                            *dst.add(i * 2) = b[0];
                            *dst.add(i * 2 + 1) = b[1];
                        }
                    }
                }
                drop(f32w); // free the f32 staging before recording the weight
                stats.gpu_resident_bytes += nbytes as u64;
                stats.f16_tensors += 1;
                if nbytes >= (16 << 20) {
                    plog!("f16 {name}: +{} MB (cum GTT {} MB)",
                        nbytes >> 20, stats.gpu_resident_bytes >> 20);
                }
                gpu_weights.insert(
                    name.clone(),
                    NemGpuWeight {
                        buffer: buf,
                        quant: NemQuant::F16,
                        out_features: out_l,
                        in_features: in_l,
                    },
                );
            } else {
                // Small host f32 (norms, conv1d, A_log/D/dt_bias, router gate,
                // e_score bias, kv scales, embeddings, norm_f).
                let mut data = decode_plain(&view)?;
                if name.ends_with("mixer.conv1d.weight") {
                    data = squeeze_conv1d_middle_dim(view.shape(), data)?;
                }
                stats.host_bytes += (data.len() * 4) as u64;
                stats.host_tensors += 1;
                // The embedding table (first stage, ~2.1GB f32) and norm_f (last
                // stage) are the non-layer host tensors unique to the two edge
                // stages — mark them explicitly, plus any other large host alloc.
                if name == "backbone.embeddings.weight"
                    || name == "backbone.norm_f.weight"
                    || data.len() * 4 >= (16 << 20)
                {
                    plog!("host {name}: +{} MB f32 (cum host {} MB)",
                        (data.len() * 4) >> 20, stats.host_bytes >> 20);
                }
                host.insert(name.clone(), SimpleTensor { data, shape: vec![] });
            }
        }
    }
    plog!(
        "DONE resident GTT {} MB ({} nvfp4-expert + {} fp8 + {} f16), host {} MB ({} tensors)",
        stats.gpu_resident_bytes >> 20, stats.nvfp4_expert_tensors, stats.fp8_tensors,
        stats.f16_tensors, stats.host_bytes >> 20, stats.host_tensors
    );
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nemotron::BlockSpec;

    fn tiny_cfg() -> NemotronConfig {
        NemotronConfig {
            hidden_size: 32,
            num_hidden_layers: 3,
            vocab_size: 32,
            norm_eps: 1e-5,
            tie_word_embeddings: false,
            mamba_num_heads: 2,
            mamba_head_dim: 8,
            ssm_state_size: 4,
            n_groups: 2,
            conv_kernel: 4,
            use_conv_bias: true,
            time_step_min: 0.001,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            attention_bias: false,
            n_routed_experts: 3,
            moe_latent_size: 16,
            moe_shared_expert_intermediate_size: 32,
            routed_scaling_factor: 5.0,
            norm_topk_prob: true,
            n_group: 1,
            topk_group: 1,
            block_specs: vec![
                BlockSpec::Mamba,
                BlockSpec::Attention,
                BlockSpec::Moe { num_experts_per_tok: 2, moe_intermediate_size: 32 },
            ],
        }
    }

    /// The resident-footprint formula, checked byte-exact on a tiny config
    /// (every category hand-derived) so the PP-fit projection in the report is
    /// trustworthy. Also asserts the compression that makes the 75B fit: the
    /// GPU-resident bytes are a small fraction of what the f32-host loader
    /// (params×4) would materialize — the OOM lever.
    #[test]
    fn resident_footprint_is_byte_exact_and_compresses() {
        let cfg = tiny_cfg();
        let fp = resident_footprint(&cfg, 0, 3, true, true);
        assert_eq!(fp.nvfp4_expert_bytes, 2304, "nvfp4 experts");
        assert_eq!(fp.fp8_bytes, 4176, "fp8 mamba+shared");
        assert_eq!(fp.f16_bytes, 7168, "f16 attn+fc+lm");
        assert_eq!(fp.host_f32_bytes, 5612, "host f32 (incl embed+norm_f)");
        assert_eq!(fp.gpu_resident_bytes(), 13648);
        assert_eq!(fp.total_bytes(), 19260);

        // f32-host baseline = every resident matmul weight ×4 (what the OOM'd
        // loader stored). Resident keeps them 4-bit/8-bit/f16 → well under half.
        // nvfp4 experts: 0.75B vs 4B/param → the dominant win.
        let f32_matmul = {
            // params: nvfp4 0.75B/param, fp8 ~1B/param, f16 2B/param → invert.
            let nvfp4_params = (fp.nvfp4_expert_bytes as f64) / 0.75;
            let fp8_params = fp.fp8_bytes as f64; // ≈ params (scalar scale negligible)
            let f16_params = (fp.f16_bytes as f64) / 2.0;
            (nvfp4_params + fp8_params + f16_params) * 4.0
        };
        assert!(
            (fp.gpu_resident_bytes() as f64) < 0.45 * f32_matmul,
            "resident {} not < 0.45× f32 baseline {f32_matmul}",
            fp.gpu_resident_bytes()
        );
    }

    /// The per-expert MoE dispatch crux (GPU-free): the byte layout the resident
    /// loader writes into the concatenated expert buffer must line up EXACTLY
    /// with the `packed_off`/`sb_off` slice offsets `NemotronModel::
    /// nem_expert_matvec` dispatches with. Proven by building the concat exactly
    /// as `load_nemotron_resident` does (`e*out*(in/2)` packed bytes, `e*out*(in/
    /// group_size)*4` scale bytes) and asserting (a) the shader's WORD/ELEM
    /// offsets reduce to those same byte offsets and (b) each expert's slice
    /// equals its standalone packed nibbles + folded scales. The nvfp4 shader
    /// itself is already GPU-validated on the standalone (dense) case, so
    /// matching bytes at the right offset ⇒ correct per-expert dispatch.
    #[test]
    fn nvfp4_expert_concat_layout_matches_dispatch_offsets() {
        let (out, in_, gs, ne) = (2usize, 64usize, 16usize, 3usize);
        let per_packed = out * (in_ / 2); // bytes/expert
        let per_scale = out * (in_ / gs); // folded f32 elems/expert

        // Distinct per-expert packed nibbles + folded scales.
        let experts: Vec<(Vec<u8>, Vec<f32>)> = (0..ne)
            .map(|e| {
                let packed: Vec<u8> =
                    (0..per_packed).map(|i| ((i * 7 + e * 131) & 0xFF) as u8).collect();
                let wscale: Vec<u8> =
                    (0..per_scale).map(|i| (0x50 + ((i + e) & 0x0F)) as u8).collect();
                let folded = nvfp4_fold_scales(&wscale, 0.00013 + e as f32 * 1e-5);
                (packed, folded)
            })
            .collect();

        // Concatenate exactly as the loader (byte offsets e*per_packed / scales
        // e*per_scale*4 into contiguous buffers).
        let mut cat_packed = vec![0u8; ne * per_packed];
        let mut cat_scales_f32 = vec![0f32; ne * per_scale];
        for (e, (p, s)) in experts.iter().enumerate() {
            cat_packed[e * per_packed..(e + 1) * per_packed].copy_from_slice(p);
            cat_scales_f32[e * per_scale..(e + 1) * per_scale].copy_from_slice(s);
        }

        for e in 0..ne {
            // Offsets as nem_expert_matvec computes them (WORDS / ELEMS).
            let packed_word_off = e * out * (in_ / 8);
            let scale_elem_off = e * out * (in_ / gs);
            // (a) reduce to the loader's byte offsets.
            assert_eq!(packed_word_off * 4, e * per_packed, "packed byte offset e={e}");
            assert_eq!(scale_elem_off, e * per_scale, "scale elem offset e={e}");
            // (b) the slice at that offset == the standalone expert.
            let pslice = &cat_packed[packed_word_off * 4..packed_word_off * 4 + per_packed];
            let sslice = &cat_scales_f32[scale_elem_off..scale_elem_off + per_scale];
            assert_eq!(pslice, experts[e].0.as_slice(), "packed slice e={e}");
            assert_eq!(sslice, experts[e].1.as_slice(), "scale slice e={e}");
        }
    }

    /// Full-loader smoke test against the real NVFP4 checkpoint, restricted
    /// to a handful of representative tensors via a tiny `layer_start..
    /// layer_end` window (this is NOT a full-model load — the 75B checkpoint
    /// does not fit in RAM on this Mac). Set
    /// `VLLM_TEST_NEMOTRON_DIR=/Volumes/Shared_Drive/models/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4`
    /// to run:
    ///   VLLM_TEST_NEMOTRON_DIR=... cargo test --lib -- --ignored nemotron_loader_real_checkpoint
    #[test]
    #[ignore]
    fn nemotron_loader_real_checkpoint() {
        let dir = match std::env::var("VLLM_TEST_NEMOTRON_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg_path = Path::new(&dir).join("config.json");
        let cfg_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).expect("read config.json"))
                .expect("parse config.json");
        let config = NemotronConfig::from_json(&cfg_json).expect("parse NemotronConfig");

        // Layer 0 = mamba (in_proj/out_proj FP8), layer 1 = moe (routed
        // experts NVFP4 + shared_experts FP8 + BF16 gate/fc1/fc2), layer 7 =
        // the first attention layer (BF16 q/k/v/o_proj + F32 k_scale/
        // v_scale). Covers every group in one narrow window.
        assert_eq!(config.block_specs[0], BlockSpec::Mamba);
        assert!(matches!(config.block_specs[1], BlockSpec::Moe { .. }));
        assert_eq!(config.block_specs[7], BlockSpec::Attention);

        let weights_path = Path::new(&dir).join("model.safetensors.index.json");
        let (weights, stats) =
            load_nemotron_weights(&weights_path, &config, 0, 8, false, false)
                .expect("load nemotron layers [0,8)");

        eprintln!(
            "nemotron loader: {} tensors ({} fp8, {} nvfp4), {:.3} GB f32",
            stats.tensors_loaded,
            stats.fp8_tensors,
            stats.nvfp4_tensors,
            stats.bytes_f32 as f64 / 1e9
        );
        assert!(stats.fp8_tensors > 0, "expected FP8 tensors in layers [0,8)");
        assert!(stats.nvfp4_tensors > 0, "expected NVFP4 tensors in layers [0,8)");

        // Spot-check specific real tensors by name + shape.
        let in_proj = weights.f32_slice("backbone.layers.0.mixer.in_proj.weight");
        assert_eq!(in_proj.len(), 18048 * 4096, "mamba in_proj dequant shape");

        let expert_up = weights.f32_slice("backbone.layers.1.mixer.experts.0.up_proj.weight");
        assert_eq!(expert_up.len(), 1280 * 1024, "nvfp4 expert up_proj dequant shape");

        let q_proj = weights.f32_slice("backbone.layers.7.mixer.q_proj.weight");
        assert_eq!(q_proj.len(), 4096 * 4096, "attn q_proj (bf16 passthrough) shape");

        let conv1d = weights.f32_slice("backbone.layers.0.mixer.conv1d.weight");
        assert_eq!(conv1d.len(), 9728 * 4, "mamba conv1d squeeze shape (conv_dim*kernel)");

        // No NaN/inf in any dequantized tensor (a common silent-corruption
        // signature for a scale/shape mismatch).
        for (name, t) in [
            ("in_proj", in_proj),
            ("expert_up", expert_up),
            ("q_proj", q_proj),
        ] {
            assert!(
                t.iter().all(|v| v.is_finite()),
                "{name}: non-finite values after dequant"
            );
        }
    }

    /// Accuracy pre-screen for the VLLM_VULKAN_NEMOTRON_MAMBA_Q8 /
    /// VLLM_VULKAN_NEMOTRON_SHARED_Q8 requant levers (pure Rust, no GPU):
    /// dequant a random FP8-E4M3 weight to f32 (the reference,
    /// `crate::model::dequantize_fp8`), re-quantize it per-row to q8_0
    /// exactly as the loader's `requant_q8` branch does, dequant the q8_0
    /// bytes back EXACTLY as the `mul_mat_vec_q8_0deq` shader reads them
    /// (`crate::model::dequant_q8_0_to_f32`), then matvec a random activation
    /// against both weight matrices and compare cosine similarity. This is a
    /// requant-fidelity check (FP8 -> q8_0), not a load-path integration test.
    /// Shapes: a small mamba-proj-shaped row layout, plus the two real
    /// shared_experts shapes (up_proj out=5376/in=4096, down_proj
    /// out=4096/in=5376) — both in_f are %32==0 (4096=128x32, 5376=168x32) so
    /// every row is whole q8_0 blocks.
    #[test]
    fn fp8_to_q8_0_requant_matches_fp8_reference_cos() {
        use crate::model::{dequant_q8_0_to_f32, dequantize_fp8, quantize_q8_0};

        // (out_features, in_features): mamba-proj-shaped, then the two real
        // shared_experts shapes. Mamba's out_f is shrunk from the real
        // hidden-size row count for test speed; the cos property is
        // per-row, so it is shape-of-row (in_f) sensitive, not out_f.
        let shapes: [(usize, usize); 3] = [(64, 4096), (5376, 4096), (4096, 5376)];

        // Deterministic PRNG (xorshift) so the test is reproducible without
        // pulling in a `rand` dev-dependency.
        fn next_u32(state: &mut u64) -> u32 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state >> 32) as u32
        }
        fn next_f32(state: &mut u64) -> f32 {
            ((next_u32(state) as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        }
        let mut state: u64 = 0x9E3779B97F4A7C15;

        for (out_f, in_f) in shapes {
            // Random FP8-E4M3 weight bytes (any byte pattern is a valid e4m3
            // value) + per-row scale.
            let fp8_bytes: Vec<u8> = (0..out_f * in_f).map(|_| (next_u32(&mut state) & 0xFF) as u8).collect();
            let scale: Vec<f32> = (0..out_f).map(|_| 0.01 + next_f32(&mut state).abs() * 0.05).collect();
            let x: Vec<f32> = (0..in_f).map(|_| next_f32(&mut state)).collect();

            let w_ref = dequantize_fp8(&fp8_bytes, &scale, out_f, in_f);

            let q8: Vec<u8> = w_ref
                .chunks(in_f)
                .flat_map(|row| quantize_q8_0(row))
                .collect();
            let w_q8 = dequant_q8_0_to_f32(&q8);
            assert_eq!(w_q8.len(), w_ref.len());

            let matvec = |w: &[f32]| -> Vec<f32> {
                w.chunks(in_f)
                    .map(|row| row.iter().zip(&x).map(|(&a, &b)| a * b).sum())
                    .collect()
            };
            let y_ref = matvec(&w_ref);
            let y_q8 = matvec(&w_q8);

            let dot: f32 = y_ref.iter().zip(&y_q8).map(|(&a, &b)| a * b).sum();
            let na: f32 = y_ref.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let nb: f32 = y_q8.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let cos = dot / (na * nb);
            assert!(cos >= 0.999, "fp8->q8_0 requant matvec cos too low for shape ({out_f},{in_f}): {cos}");
            eprintln!("fp8_to_q8_0_requant_matches_fp8_reference_cos: shape=({out_f},{in_f}) cos={cos}");
        }
    }
}
