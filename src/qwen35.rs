// SPDX-License-Identifier: Apache-2.0
//! Qwen3.6 (`qwen3_5`) hybrid architecture — GatedDeltaNet (linear attention)
//! + gated full attention, dense (27B) or MoE (35B).
//!
//! This module is the in-progress port; see `docs/qwen35-port.md` for the full
//! spec extracted from the qwen3.6-mlx reference. Config + state scaffolding is
//! in place; the CPU reference forward passes are implemented and validated
//! against the MLX oracle in Milestone A (see the checklist in that doc).
#![allow(dead_code)]

use crate::model::{
    cpu_matmul, cpu_rms_norm, cpu_rms_norm_no_weight, cpu_rope, cpu_sdpa, cpu_silu, KvCache,
    ModelWeights,
};
use crate::moe;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
fn softplus(x: f32) -> f32 {
    // numerically stable log(1 + exp(x))
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

// --- GatedDeltaNet recurrent-matrix update primitives -----------------------
//
// The per-value-head recurrence is `state = decay*state + k⊗delta`, applied
// as two in-place f32 passes with NO round-back: `gdn_apply_decay` then (after
// a `gdn_kv_mem` read of the freshly-decayed state, used by the forward path
// to derive `delta`) `gdn_apply_delta`. These are split out of `delta_net` as
// `pub(crate)` fns so a tape-replay path can call the identical decay/delta
// passes with a FROZEN `delta` (produced once during a forward pass and
// stored per-position) while skipping the `kv_mem` re-derivation — the
// bit-exactness of that substitution is proven by `gdn_tape_replay_tests`
// below, which de-risks the delta-tape rollback design surveyed in
// `audit-gdn-rollback-vs-tape-replay.md` (design lineage: MTPLX
// `mtplx_linear_gated_delta_from_conv_tape_v1` / `..._replay_v1`,
// Apache-2.0, `github.com/mtplx/mtplx` — reimplemented here from the public
// algorithm description, no code copied).

/// Pass 1: `state[k,v] *= decay`, in place, over one head's `kd*vd` state.
#[inline]
pub(crate) fn gdn_apply_decay(head_state: &mut [f32], decay: f32) {
    for e in head_state.iter_mut() {
        *e *= decay;
    }
}

/// `kv_mem[v] = sum_k state[k,v] * k_j[k]` — a READ of `head_state` (does not
/// mutate it). Called on the already-decayed state by the forward path to
/// derive `delta`; the tape-replay path skips this and uses a frozen `delta`.
#[inline]
pub(crate) fn gdn_kv_mem(head_state: &[f32], kd: usize, vd: usize, k_j: &[f32]) -> Vec<f32> {
    let mut kv_mem = vec![0.0f32; vd];
    for kk in 0..kd {
        let kv = k_j[kk];
        for vv in 0..vd {
            kv_mem[vv] += head_state[kk * vd + vv] * kv;
        }
    }
    kv_mem
}

/// Pass 2: `state[k,v] += k_j[k] * delta[v]`, in place, over one head's
/// `kd*vd` state. Identical whether `delta` was just derived (forward) or
/// read off a tape (replay).
#[inline]
pub(crate) fn gdn_apply_delta(head_state: &mut [f32], kd: usize, vd: usize, k_j: &[f32], delta: &[f32]) {
    for kk in 0..kd {
        let kv = k_j[kk];
        for vv in 0..vd {
            head_state[kk * vd + vv] += kv * delta[vv];
        }
    }
}

/// Per-layer attention kind, from `config.layer_types`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Gated full attention (KV-cache SDPA + output gate + partial RoPE).
    FullAttention,
    /// GatedDeltaNet linear attention (depthwise conv + delta-rule recurrence).
    LinearAttention,
}

/// Qwen3.6 (`qwen3_5`) configuration, parsed from `config.json` (`text_config`).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,

    // Full-attention dims
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attn_output_gate: bool,

    // RoPE
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,

    // Linear-attention (GatedDeltaNet) dims
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,

    // Dense MLP
    pub intermediate_size: usize,

    // MoE (None / 0 for dense models)
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    /// Per-layer attention kind (length == num_hidden_layers).
    pub layer_types: Vec<LayerType>,
}

impl Qwen35Config {
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0 && self.num_experts_per_tok > 0
    }

    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }
    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
    /// conv1d channel count = key_dim*2 + value_dim (q, k, v concatenated).
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
    /// RoPE rotary dimension = head_dim * partial_rotary_factor.
    pub fn rotary_dim(&self) -> usize {
        ((self.head_dim as f32) * self.partial_rotary_factor).round() as usize
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        // `qwen3_5` ships dims under `text_config` (VL/text-only both); fall back
        // to top level for flattened configs.
        let tc = v.get("text_config").unwrap_or(v);
        let u = |key: &str| tc[key].as_u64().map(|x| x as usize);
        let req = |key: &str| u(key).ok_or_else(|| format!("config.json missing '{key}'"));

        let num_attention_heads = req("num_attention_heads")?;
        let hidden_size = req("hidden_size")?;
        let head_dim = u("head_dim").unwrap_or(hidden_size / num_attention_heads);

        let layer_types: Vec<LayerType> = tc["layer_types"]
            .as_array()
            .ok_or("config.json missing 'layer_types'")?
            .iter()
            .map(|s| match s.as_str() {
                Some("full_attention") => Ok(LayerType::FullAttention),
                Some("linear_attention") => Ok(LayerType::LinearAttention),
                other => Err(format!("unknown layer_type {other:?}")),
            })
            .collect::<Result<_, _>>()?;

        let rope = tc.get("rope_parameters").unwrap_or(tc);
        let rope_theta = rope["rope_theta"].as_f64().unwrap_or(10_000_000.0) as f32;
        let partial_rotary_factor =
            rope["partial_rotary_factor"].as_f64().unwrap_or(0.25) as f32;

        Ok(Qwen35Config {
            hidden_size,
            num_hidden_layers: req("num_hidden_layers")?,
            vocab_size: req("vocab_size")?,
            rms_norm_eps: tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            tie_word_embeddings: v["tie_word_embeddings"]
                .as_bool()
                .or_else(|| tc["tie_word_embeddings"].as_bool())
                .unwrap_or(false),
            num_attention_heads,
            num_key_value_heads: u("num_key_value_heads").unwrap_or(num_attention_heads),
            head_dim,
            attn_output_gate: tc["attn_output_gate"].as_bool().unwrap_or(false),
            rope_theta,
            partial_rotary_factor,
            linear_num_key_heads: req("linear_num_key_heads")?,
            linear_num_value_heads: req("linear_num_value_heads")?,
            linear_key_head_dim: req("linear_key_head_dim")?,
            linear_value_head_dim: req("linear_value_head_dim")?,
            linear_conv_kernel_dim: req("linear_conv_kernel_dim")?,
            intermediate_size: u("intermediate_size").unwrap_or(0),
            num_experts: u("num_experts").unwrap_or(0),
            num_experts_per_tok: u("num_experts_per_tok").unwrap_or(0),
            moe_intermediate_size: u("moe_intermediate_size").unwrap_or(0),
            shared_expert_intermediate_size: u("shared_expert_intermediate_size").unwrap_or(0),
            layer_types,
        })
    }
}

/// Recurrent state for one GatedDeltaNet layer (decode), B = 1.
pub struct DeltaNetState {
    /// Sliding conv window: `[conv_dim, kernel-1]` row-major.
    pub conv_state: Vec<f32>,
    /// Delta-rule state matrix: `[num_v_heads, key_head_dim, value_head_dim]`.
    pub state: Vec<f32>,
}

impl DeltaNetState {
    pub fn new(cfg: &Qwen35Config) -> Self {
        let conv_window = cfg.linear_conv_kernel_dim.saturating_sub(1);
        DeltaNetState {
            conv_state: vec![0.0; cfg.conv_dim() * conv_window],
            state: vec![
                0.0;
                cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim
            ],
        }
    }
    pub fn reset(&mut self) {
        self.conv_state.iter_mut().for_each(|x| *x = 0.0);
        self.state.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Tensor-parallel state: this rank owns `key_dim/tp` + `value_dim/tp`
    /// channels and `num_value_heads/tp` recurrence heads. The conv window holds
    /// the rank's sharded `conv_dim/tp` channels (q|k|v local segments), and the
    /// delta-rule state holds the rank's `nv/tp` value heads. Channel/head
    /// independence makes this exact vs the single-node state restricted to this
    /// rank's heads.
    pub fn new_tp(cfg: &Qwen35Config, tp: usize) -> Self {
        let conv_window = cfg.linear_conv_kernel_dim.saturating_sub(1);
        let conv_dim_r = cfg.conv_dim() / tp;
        let state_r = (cfg.linear_num_value_heads / tp)
            * cfg.linear_key_head_dim * cfg.linear_value_head_dim;
        DeltaNetState {
            conv_state: vec![0.0; conv_dim_r * conv_window],
            state: vec![0.0; state_r],
        }
    }
}

/// Per-layer mutable state: a KV cache (full attention) or a DeltaNet recurrent
/// state (linear attention).
pub enum LayerState {
    Full(KvCache),
    Linear(DeltaNetState),
}

/// Build a small hybrid Qwen3.5 model (Linear + Full interleaved) with
/// deterministic synthetic weights, large enough to drive `delta_net` +
/// `gated_attention` on the CPU path for cross-model KV spikes (S7).
///
/// `layer_types` selects the schedule; full-attention layers share one
/// (n_kv, head_dim) geometry so they are matched-KV across two models that
/// only differ in depth / weight tags.
pub fn synthetic_hybrid_qwen35(
    layer_types: Vec<LayerType>,
    max_seq: usize,
    weight_tag: &str,
) -> Qwen35Model {
    use crate::model::{ModelWeights, SimpleTensor};
    use std::collections::HashMap;

    let h = 32usize;
    let (nq, nkv, hd) = (4usize, 2usize, 8usize);
    let (nk, nv, kd, vd, kern) = (1usize, 2usize, 8usize, 8usize, 4usize);
    let n = layer_types.len();
    let cfg = Qwen35Config {
        hidden_size: h,
        num_hidden_layers: n,
        vocab_size: 64,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        num_attention_heads: nq,
        num_key_value_heads: nkv,
        head_dim: hd,
        attn_output_gate: true,
        rope_theta: 1e7,
        partial_rotary_factor: 0.5,
        linear_num_key_heads: nk,
        linear_num_value_heads: nv,
        linear_key_head_dim: kd,
        linear_value_head_dim: vd,
        linear_conv_kernel_dim: kern,
        intermediate_size: 16,
        num_experts: 0,
        num_experts_per_tok: 0,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        layer_types: layer_types.clone(),
    };

    // Deterministic LCG seeded from tag + index
    let mut seed = 0xC0FFEE_u64;
    for b in weight_tag.bytes() {
        seed = seed.wrapping_mul(0x100_0000_01B3).wrapping_add(b as u64);
    }
    let mut g = move || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    let mut mk = |rows: usize, cols: usize, scale: f32| -> Vec<f32> {
        (0..rows * cols).map(|_| g() * scale).collect()
    };

    let key_dim = nk * kd;
    let value_dim = nv * vd;
    let conv_dim = 2 * key_dim + value_dim;
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;

    let mut tensors: HashMap<String, SimpleTensor> = HashMap::new();
    let mut ins = |name: String, data: Vec<f32>| {
        tensors.insert(name, SimpleTensor { data, shape: vec![] });
    };

    for (li, kind) in layer_types.iter().enumerate() {
        match kind {
            LayerType::LinearAttention => {
                let p = format!("model.layers.{li}.linear_attn");
                ins(format!("{p}.in_proj_qkv.weight"), mk(conv_dim, h, 0.05));
                ins(format!("{p}.in_proj_z.weight"), mk(value_dim, h, 0.05));
                ins(format!("{p}.in_proj_a.weight"), mk(nv, h, 0.05));
                ins(format!("{p}.in_proj_b.weight"), mk(nv, h, 0.05));
                ins(format!("{p}.conv1d.weight"), mk(conv_dim, kern, 0.5));
                ins(format!("{p}.A_log"), mk(nv, 1, 1.0));
                ins(format!("{p}.dt_bias"), mk(nv, 1, 0.5));
                ins(
                    format!("{p}.norm.weight"),
                    mk(vd, 1, 0.1).iter().map(|&v| 1.0 + v).collect(),
                );
                ins(format!("{p}.out_proj.weight"), mk(h, value_dim, 0.05));
            }
            LayerType::FullAttention => {
                let p = format!("model.layers.{li}.self_attn");
                ins(format!("{p}.q_proj.weight"), mk(q_dim * 2, h, 0.05));
                ins(format!("{p}.k_proj.weight"), mk(kv_dim, h, 0.05));
                ins(format!("{p}.v_proj.weight"), mk(kv_dim, h, 0.05));
                ins(format!("{p}.o_proj.weight"), mk(h, q_dim, 0.05));
                ins(
                    format!("{p}.q_norm.weight"),
                    mk(hd, 1, 0.1).iter().map(|&v| 1.0 + v).collect(),
                );
                ins(
                    format!("{p}.k_norm.weight"),
                    mk(hd, 1, 0.1).iter().map(|&v| 1.0 + v).collect(),
                );
            }
        }
        // Minimal layernorm / MLP so forward_pp_range can run if needed later.
        let base = format!("model.layers.{li}");
        ins(
            format!("{base}.input_layernorm.weight"),
            mk(h, 1, 0.05).iter().map(|&v| 1.0 + v).collect(),
        );
        ins(
            format!("{base}.post_attention_layernorm.weight"),
            mk(h, 1, 0.05).iter().map(|&v| 1.0 + v).collect(),
        );
        ins(format!("{base}.mlp.gate_proj.weight"), mk(16, h, 0.05));
        ins(format!("{base}.mlp.up_proj.weight"), mk(16, h, 0.05));
        ins(format!("{base}.mlp.down_proj.weight"), mk(h, 16, 0.05));
    }

    let weights = ModelWeights { tensors };
    Qwen35Model::new(cfg, weights, max_seq, "unused".to_string())
}

/// Bit-exact rollback snapshot of one PP stage's HOST-authoritative per-token
/// decode state (P1 speculative-pipelining rollback). Captures, keyed by
/// stage-local `state_idx` (matches `layer_state`):
///   - every linear (GatedDeltaNet) layer's `DeltaNetState` (sliding conv window
///     + delta-rule matrix). Authoritative on the CPU path (`DN_GPU` off /
///     engine-less Mac tests). When the GPU-resident `DnGpuLayer` buffers are
///     authoritative (`DN_GPU` on) the caller passes that layer's `state_idx` in
///     `skip_linear_si` so its stale host copy is NOT captured (the GPU buffers
///     are snapshotted device-side instead — see `VulkanModel::spec_snapshot`).
///   - every full-attention layer's KV `seq_len` counter. The K/V planes are
///     overwrite-in-place, so rewinding the counter is a COMPLETE rollback: the
///     next append reuses the abandoned slots and SDPA only ever reads
///     `0..seq_len`, so nothing downstream can observe the stale tail.
// `HostStateSnapshot` moved to the foundation module `crate::host_state` (Phase
// A2) so the generic `SpecSlot` can reference it with no model feature enabled.
// Re-exported here so `qwen35::HostStateSnapshot` and the qwen35-internal
// spec_snapshot/spec_restore call sites keep resolving unchanged.
pub use crate::host_state::HostStateSnapshot;

/// FNV-1a fingerprint of the model config + resident layer range that a KV
/// prefix blob was captured against. Import compares this to the loading
/// model's own fingerprint and refuses to load on mismatch — the
/// correctness gate the plan's §3.4/§6 calls a "config/layout fingerprint".
/// Covers every dimension that changes the byte layout of a prefix section
/// (head counts/dims, conv kernel, RoPE params) plus the GLOBAL layer-type
/// map and the resident `[pp_start, pp_end)` range (so a PP-3 stage-1 blob
/// can never be loaded into a PP-5 stage-2, even with identical dims).
pub(crate) fn prefix_fingerprint(cfg: &Qwen35Config, pp_start: usize, pp_end: usize) -> u64 {
    fn fnv(acc: &mut u64, bytes: &[u8]) {
        for &b in bytes {
            *acc ^= b as u64;
            *acc = acc.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    let mut acc = 0xcbf2_9ce4_8422_2325u64;
    fnv(&mut acc, &(cfg.hidden_size as u64).to_le_bytes());
    fnv(&mut acc, &(cfg.num_hidden_layers as u64).to_le_bytes());
    fnv(&mut acc, &(cfg.vocab_size as u64).to_le_bytes());
    fnv(&mut acc, &(cfg.num_attention_heads as u64).to_le_bytes());
    fnv(&mut acc, &(cfg.num_key_value_heads as u64).to_le_bytes());
    fnv(&mut acc, &(cfg.head_dim as u64).to_le_bytes());
    fnv(&mut acc, &cfg.rope_theta.to_bits().to_le_bytes());
    fnv(&mut acc, &cfg.partial_rotary_factor.to_bits().to_le_bytes());
    fnv(&mut acc, &(cfg.linear_num_key_heads as u64).to_le_bytes());
    fnv(&mut acc, &(cfg.linear_num_value_heads as u64).to_le_bytes());
    fnv(&mut acc, &(cfg.linear_key_head_dim as u64).to_le_bytes());
    fnv(&mut acc, &(cfg.linear_value_head_dim as u64).to_le_bytes());
    fnv(&mut acc, &(cfg.linear_conv_kernel_dim as u64).to_le_bytes());
    for lt in &cfg.layer_types {
        let tag = match lt {
            LayerType::FullAttention => 0u8,
            LayerType::LinearAttention => 1u8,
        };
        fnv(&mut acc, &[tag]);
    }
    fnv(&mut acc, &(pp_start as u64).to_le_bytes());
    fnv(&mut acc, &(pp_end as u64).to_le_bytes());
    acc
}

/// Append a length-prefixed f32 (LE) section: `u64 byte-length` then the
/// float bytes themselves.
fn write_f32_section(out: &mut Vec<u8>, data: &[f32]) {
    out.extend_from_slice(&((data.len() * 4) as u64).to_le_bytes());
    for x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > buf.len() {
        return Err("prefix blob truncated (u32)".to_string());
    }
    let v = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > buf.len() {
        return Err("prefix blob truncated (u64)".to_string());
    }
    let v = u64::from_le_bytes(buf[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

/// Read one length-prefixed f32 (LE) section written by `write_f32_section`.
fn read_f32_section(buf: &[u8], pos: &mut usize) -> Result<Vec<f32>, String> {
    let len = read_u64(buf, pos)? as usize;
    if *pos + len > buf.len() {
        return Err("prefix blob truncated (section body)".to_string());
    }
    if len % 4 != 0 {
        return Err(format!("prefix blob section length {len} not a multiple of 4"));
    }
    let n = len / 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = *pos + i * 4;
        v.push(f32::from_le_bytes(buf[o..o + 4].try_into().unwrap()));
    }
    *pos += len;
    Ok(v)
}

/// Qwen3.6 dense model (CPU reference). MoE (35B) is Milestone C.
///
/// Pipeline-parallel aware: `config.layer_types` is always the GLOBAL list
/// (length == the model's full `num_hidden_layers`), but only the layers in
/// `[pp_start, pp_end)` are resident — their weights are loaded and one
/// `layer_state` entry exists per resident layer. State for a global layer
/// index `g` lives at `layer_state[g - pp_start]` (`state_idx`). On a
/// single-node (non-PP) model `pp_start == 0` and `pp_end == num_hidden_layers`,
/// so the indexing collapses to the plain 0-based form.
pub struct Qwen35Model {
    pub config: Qwen35Config,
    pub weights: ModelWeights,
    /// One entry per RESIDENT layer (`pp_end - pp_start`), indexed by
    /// `global_layer - pp_start`. Use `state_idx()` to map.
    pub layer_state: Vec<LayerState>,
    pub lm_head_name: String,
    /// First resident GLOBAL layer index (0 on stage 0 / non-PP).
    pub pp_start: usize,
    /// One-past-last resident GLOBAL layer index.
    pub pp_end: usize,
    /// 4-bit-RESIDENT MoE experts (35B-A3B). When non-empty, `moe_mlp`
    /// dequantizes only the routed experts per token instead of reading f32
    /// experts from `weights` (the ~3.2GB/layer -> ~0.4GB/layer footprint cut
    /// that makes an 8-layer MoE PP stage fit a 15GB node). Empty for dense
    /// (27B) and for the f32-host MoE parity path.
    pub quant_moe: moe::QuantMoeLayers,
    /// MLX-affine 4-bit `embed_tokens` kept PACKED resident (default; gated by
    /// `VLLM_VULKAN_Q35_EMBED_PACKED`). When `Some`, the embed lookup decodes
    /// only the current token's row on demand instead of holding the whole
    /// `[vocab,hidden]` f16 table (~1.5GB for the 122B) — the stage-0 load anon
    /// spike. `None` on a middle/last-only PP stage (no embed) or when the flag
    /// is off (whole f16 table lives in `VulkanModel.q35_f16_host` instead).
    pub embed_packed: Option<crate::model::PackedEmbed>,
    /// KV-offload chunk-boundary snapshot: `boundary` (multiple of
    /// `kvstore::CHUNK`) -> per resident-linear-layer `state_idx` ->
    /// `(conv_state, state)` AT that boundary, i.e. the DeltaNet recurrent
    /// state after exactly `boundary` tokens (`[0, boundary)`). GatedDeltaNet
    /// is a one-way recurrence that cannot rewind, so `export_prefix` cannot
    /// reconstruct a past boundary's state from the live (post-prompt) state
    /// — it must be captured in-flight, here, at the moment the boundary is
    /// crossed. Retain-latest: only the most-recently-crossed boundary is
    /// kept (constant ~13.2MB/stage regardless of prefix length); see
    /// `export_prefix`'s `Linear` arm for the read side.
    pub gdn_boundary:
        std::collections::HashMap<usize, std::collections::HashMap<usize, (Vec<f32>, Vec<f32>)>>,
}

impl Qwen35Model {
    /// Single-node / full model: resident range is `[0, num_hidden_layers)`.
    pub fn new(config: Qwen35Config, weights: ModelWeights, max_seq_len: usize, lm_head_name: String) -> Self {
        let end = config.num_hidden_layers;
        Self::new_range(config, weights, max_seq_len, lm_head_name, 0, end)
    }

    /// Pipeline-parallel stage: holds only global layers `[pp_start, pp_end)`.
    /// `config.layer_types` MUST still be the full global list so that
    /// `layer_types[global_idx]` is correct for the resident range. One
    /// `layer_state` entry is built per resident layer (using the global
    /// layer_type at that index).
    pub fn new_range(
        config: Qwen35Config,
        weights: ModelWeights,
        max_seq_len: usize,
        lm_head_name: String,
        pp_start: usize,
        pp_end: usize,
    ) -> Self {
        let layer_state = (pp_start..pp_end)
            .map(|g| match config.layer_types[g] {
                LayerType::FullAttention => LayerState::Full(KvCache::new(
                    max_seq_len,
                    config.num_key_value_heads,
                    config.head_dim,
                )),
                LayerType::LinearAttention => LayerState::Linear(DeltaNetState::new(&config)),
            })
            .collect();
        Qwen35Model {
            config, weights, layer_state, lm_head_name, pp_start, pp_end,
            quant_moe: moe::QuantMoeLayers::default(),
            embed_packed: None,
            gdn_boundary: std::collections::HashMap::new(),
        }
    }

    /// Tensor-parallel constructor: full resident layer range `[0, num_layers)`
    /// but every per-layer state is sized for this rank's 1/tp head shard (KV
    /// cache `num_key_value_heads/tp` heads; DeltaNet `num_value_heads/tp` heads
    /// + `conv_dim/tp` channels). `cfg` stays the FULL global config; only the
    /// state is sharded.
    pub fn new_range_tp(
        config: Qwen35Config,
        weights: ModelWeights,
        max_seq_len: usize,
        lm_head_name: String,
        pp_start: usize,
        pp_end: usize,
        tp: usize,
    ) -> Self {
        let layer_state = (pp_start..pp_end)
            .map(|g| match config.layer_types[g] {
                LayerType::FullAttention => LayerState::Full(KvCache::new(
                    max_seq_len,
                    config.num_key_value_heads / tp,
                    config.head_dim,
                )),
                LayerType::LinearAttention => LayerState::Linear(DeltaNetState::new_tp(&config, tp)),
            })
            .collect();
        Qwen35Model {
            config, weights, layer_state, lm_head_name, pp_start, pp_end,
            quant_moe: moe::QuantMoeLayers::default(),
            embed_packed: None,
            gdn_boundary: std::collections::HashMap::new(),
        }
    }

    /// Map a GLOBAL layer index to its `layer_state` slot. Panics if the layer
    /// is not resident on this stage (a PP-routing bug).
    #[inline]
    pub fn state_idx(&self, global_layer: usize) -> usize {
        debug_assert!(
            global_layer >= self.pp_start && global_layer < self.pp_end,
            "layer {global_layer} not resident on stage [{},{})", self.pp_start, self.pp_end
        );
        global_layer - self.pp_start
    }

    pub fn reset(&mut self) {
        for s in self.layer_state.iter_mut() {
            match s {
                LayerState::Full(c) => c.seq_len = 0,
                LayerState::Linear(d) => d.reset(),
            }
        }
        self.gdn_boundary.clear();
    }

    /// Current decode position = number of tokens already committed to the KV
    /// cache = the `seq_len` of this stage's full-attention layers (they all
    /// advance in lockstep, one append per decode token; prefill fills them to
    /// the prompt length). Used by `pp_step_qwen35_logits` to derive the decode
    /// `pos` the generic `serve_dist` launcher does NOT pass (mirrors how the
    /// resident Laguna decode reads its own cache length). A stage whose
    /// `[pp_start,pp_end)` owns only DeltaNet (linear) layers has no KV `seq_len`
    /// and returns 0 — harmless, since those layers are recurrent (position-free)
    /// and never consume `pos`.
    pub fn current_decode_pos(&self) -> usize {
        self.layer_state
            .iter()
            .filter_map(|s| match s {
                LayerState::Full(c) => Some(c.seq_len),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// Snapshot this stage's HOST-authoritative per-token decode state for a
    /// speculative rollback (P1). `skip_linear_si` names linear layers whose
    /// DeltaNet state is authoritative on the GPU (captured device-side by the
    /// caller) — their stale host copy is skipped to keep the snapshot cheap
    /// (on the resident node path this leaves only the tiny KV `seq_len`
    /// counters). Pass an empty set on the pure-CPU / engine-less path to
    /// capture the full DeltaNet state.
    pub fn spec_snapshot_host(
        &self,
        skip_linear_si: &std::collections::HashSet<usize>,
    ) -> HostStateSnapshot {
        let mut snap = HostStateSnapshot::default();
        for (si, s) in self.layer_state.iter().enumerate() {
            match s {
                LayerState::Linear(d) if !skip_linear_si.contains(&si) => {
                    snap.dn.push((si, d.conv_state.clone(), d.state.clone()));
                }
                LayerState::Linear(_) => {}
                LayerState::Full(c) => snap.kv.push((si, c.seq_len)),
            }
        }
        snap
    }

    /// Restore host-authoritative state captured by `spec_snapshot_host`. Rewinds
    /// each full-attention KV counter (overwrite-in-place storage ⇒ a counter
    /// rewind is a complete rollback) and writes back each captured DeltaNet
    /// state. Layers omitted from the snapshot (GPU-authoritative deltanet) are
    /// untouched — the caller restores those device-side.
    pub fn spec_restore_host(&mut self, snap: &HostStateSnapshot) {
        for (si, conv, state) in &snap.dn {
            if let LayerState::Linear(d) = &mut self.layer_state[*si] {
                d.conv_state.copy_from_slice(conv);
                d.state.copy_from_slice(state);
            }
        }
        for (si, seq_len) in &snap.kv {
            if let LayerState::Full(c) = &mut self.layer_state[*si] {
                c.truncate(*seq_len);
            }
        }
    }

    /// Design-A batched-verify rollback helper: set EVERY resident full-attention
    /// layer's KV counter to `n`. `KvCache` storage is overwrite-in-place, so a
    /// verify pass that appended `[s_R, d_1..d_D]` at positions `R..R+T` left the
    /// committed prefix's K/V physically valid at `R..R+k+1`; after a
    /// `spec_restore` rewound the counter to `R`, this re-exposes exactly the
    /// committed prefix by advancing the counter to `R+k+1` (no recompute — the
    /// counterpart to GDN's re-scan). `n` must be `<= max_seq_len` and only bytes
    /// actually written by the preceding verify are valid.
    pub fn set_full_attn_seq_len(&mut self, n: usize) {
        for s in self.layer_state.iter_mut() {
            if let LayerState::Full(c) = s {
                debug_assert!(n <= c.max_seq_len, "set_full_attn_seq_len {n} > max {}", c.max_seq_len);
                c.seq_len = n;
            }
        }
    }

    // ─── KV-prefix export/import (LMCache-NAS plan, Step 1) ────────────────
    //
    // A "resumable prefix" snapshot: for every layer resident on THIS stage,
    // capture whatever is needed to resume decoding at boundary `n` without
    // recomputing tokens `[0, n)`:
    //   - full-attention layers: K/V bytes for positions `[0, n)` — an
    //     arbitrary-boundary slice (`KvCache::k_upto`/`v_upto`), not just
    //     "up to now".
    //   - linear (GatedDeltaNet) layers: the recurrent state at that
    //     boundary. This is FIXED-SIZE regardless of `n` (see the plan's
    //     §1) — the same data `HostStateSnapshot` captures for spec-decode
    //     rollback, serialized to bytes here instead of cloned into a
    //     `Vec<f32>`.
    //
    // DEVIATION (documented, RESOLVED at the `VulkanModel` pymethod layer):
    // this core operates on `Qwen35Model::layer_state`, which is the HOST/CPU
    // copy of DeltaNet state. Under `VLLM_VULKAN_DN_GPU=1` (including the
    // `Q35_1CB` resident stage, which reads/writes the same `dn_gpu` buffers
    // directly) the GPU-resident `DnGpuLayer` buffers are authoritative and
    // this host copy alone goes stale — exactly the same regime
    // `spec_snapshot_host`/`spec_restore_host` already document and handle via
    // a `skip_linear_si` set (see `qwen35_forward.rs`'s `dn_gpu_skip_set`).
    // Rather than teach this wire format about device residency, the
    // `VulkanModel` pymethod wrappers (`kv_export_prefix`/`kv_import_prefix`
    // in `lib.rs`) bracket these two calls with `dn_gpu_sync_to_host`/
    // `dn_gpu_sync_from_host` (`qwen35_forward.rs`) — a direct memcpy against
    // the host-coherent `dn_gpu` buffers, the same access pattern
    // `spec_state_fingerprint_impl` uses — so by the time `export_prefix`
    // runs (and immediately after `import_prefix` returns) this host copy is
    // the authoritative one. Full-attention KV is host-resident in BOTH
    // regimes (`cpu_sdpa` over the host `KvCache`), so it needs no such sync.
    //
    // Wire format (v1, all integers/floats little-endian):
    //
    //   magic        [4]u8   = b"PFX1"
    //   version      u32     = 1
    //   fingerprint  u64     FNV-1a over config dims + layer_types + pp range
    //                        (`prefix_fingerprint`) — import rejects on
    //                        mismatch so a blob built for one model/config/
    //                        layout can never be silently loaded into another.
    //   seq_len      u64     prefix boundary this blob captures
    //   pp_start     u64     first resident GLOBAL layer index
    //   pp_end       u64     one-past-last resident GLOBAL layer index
    //   num_layers   u64     = pp_end - pp_start = number of layer sections
    //   layer_types  [num_layers]u8   0 = Full, 1 = Linear (state_idx order)
    //   -- per layer, in state_idx order --
    //     Full:   k_len u64, k_bytes[k_len], v_len u64, v_bytes[v_len]
    //     Linear: conv_len u64, conv_bytes[conv_len], state_len u64, state_bytes[state_len]
    //
    // Every section is independently length-prefixed (not solely derived
    // from the header dims) so a truncated/corrupt blob fails with a clear
    // error instead of reading out of bounds.

    /// Export a resumable prefix at boundary `upto_seq_len`: this stage's
    /// full-attn K/V for `[0, upto_seq_len)` plus every linear layer's
    /// (fixed-size) DeltaNet recurrent state, into the versioned binary
    /// layout documented above. Errors if `upto_seq_len` exceeds any
    /// resident full-attn layer's `max_seq_len`.
    pub fn export_prefix(&self, upto_seq_len: usize) -> Result<Vec<u8>, String> {
        for s in &self.layer_state {
            if let LayerState::Full(c) = s {
                if upto_seq_len > c.max_seq_len {
                    return Err(format!(
                        "export_prefix: boundary {upto_seq_len} exceeds max_seq_len {}",
                        c.max_seq_len
                    ));
                }
            }
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"PFX1");
        out.extend_from_slice(&1u32.to_le_bytes());
        let fp = prefix_fingerprint(&self.config, self.pp_start, self.pp_end);
        out.extend_from_slice(&fp.to_le_bytes());
        out.extend_from_slice(&(upto_seq_len as u64).to_le_bytes());
        out.extend_from_slice(&(self.pp_start as u64).to_le_bytes());
        out.extend_from_slice(&(self.pp_end as u64).to_le_bytes());
        out.extend_from_slice(&(self.layer_state.len() as u64).to_le_bytes());
        for s in &self.layer_state {
            out.push(match s {
                LayerState::Full(_) => 0u8,
                LayerState::Linear(_) => 1u8,
            });
        }
        for (si, s) in self.layer_state.iter().enumerate() {
            match s {
                LayerState::Full(c) => {
                    write_f32_section(&mut out, c.k_upto(upto_seq_len));
                    write_f32_section(&mut out, c.v_upto(upto_seq_len));
                }
                LayerState::Linear(d) => {
                    // GatedDeltaNet cannot rewind: the LIVE (d.conv_state,
                    // d.state) is only correct when `upto_seq_len` IS the
                    // live position. When exporting a PAST chunk boundary
                    // (`kv_cache_store`'s align-down), prefer the snapshot
                    // captured in-flight at that boundary (see
                    // `gdn_boundary`'s doc comment); fall back to the live
                    // state when no snapshot exists (e.g. `kv_export_prefix`
                    // at the live length — unchanged/backward-compatible).
                    match self.gdn_boundary.get(&upto_seq_len).and_then(|m| m.get(&si)) {
                        Some((conv, state)) => {
                            write_f32_section(&mut out, conv);
                            write_f32_section(&mut out, state);
                        }
                        None => {
                            write_f32_section(&mut out, &d.conv_state);
                            write_f32_section(&mut out, &d.state);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Import a resumable prefix produced by `export_prefix` into THIS
    /// model's resident layers: validates the fingerprint/layer-type/pp-range
    /// against this model (rejecting a mismatched blob with a clear `Err`,
    /// never a panic or silent wrong state), memcpys K/V into each full-attn
    /// layer's `[0, seq_len)` range, restores each linear layer's DeltaNet
    /// snapshot, and sets every resident layer's `seq_len` to the boundary.
    /// Returns the loaded prefix length on success.
    ///
    /// Restores this HOST/CPU copy of state only — see the DN_GPU deviation
    /// note in the module-level doc above this impl block. On the
    /// `VLLM_VULKAN_DN_GPU=1` path, callers must also push this restored
    /// state to the live `dn_gpu` device buffers afterwards (the
    /// `VulkanModel::kv_import_prefix` pymethod wrapper does this via
    /// `dn_gpu_sync_from_host`) — otherwise the next resident-path token
    /// reads the stale pre-import device state instead.
    pub fn import_prefix(&mut self, blob: &[u8]) -> Result<usize, String> {
        let mut pos = 0usize;
        if blob.len() < 4 || &blob[0..4] != b"PFX1" {
            return Err("import_prefix: bad magic (not a PFX1 blob)".to_string());
        }
        pos += 4;
        let version = read_u32(blob, &mut pos)?;
        if version != 1 {
            return Err(format!("import_prefix: unsupported blob version {version}"));
        }
        let fp = read_u64(blob, &mut pos)?;
        let expect_fp = prefix_fingerprint(&self.config, self.pp_start, self.pp_end);
        if fp != expect_fp {
            return Err(format!(
                "import_prefix: fingerprint mismatch (blob=0x{fp:016x}, model=0x{expect_fp:016x}) \
                 — cache built for a different model/config/layout, refusing to load"
            ));
        }
        let seq_len = read_u64(blob, &mut pos)? as usize;
        let pp_start = read_u64(blob, &mut pos)? as usize;
        let pp_end = read_u64(blob, &mut pos)? as usize;
        if pp_start != self.pp_start || pp_end != self.pp_end {
            return Err(format!(
                "import_prefix: pp range mismatch (blob=[{pp_start},{pp_end}), model=[{},{}))",
                self.pp_start, self.pp_end
            ));
        }
        let num_layers = read_u64(blob, &mut pos)? as usize;
        if num_layers != self.layer_state.len() {
            return Err(format!(
                "import_prefix: layer count mismatch (blob={num_layers}, model={})",
                self.layer_state.len()
            ));
        }
        if pos + num_layers > blob.len() {
            return Err("import_prefix: truncated layer-type map".to_string());
        }
        let layer_types = blob[pos..pos + num_layers].to_vec();
        pos += num_layers;
        for (si, s) in self.layer_state.iter().enumerate() {
            let expect = match s {
                LayerState::Full(_) => 0u8,
                LayerState::Linear(_) => 1u8,
            };
            if layer_types[si] != expect {
                return Err(format!(
                    "import_prefix: layer {si} type mismatch (blob={}, model={expect})",
                    layer_types[si]
                ));
            }
        }
        for s in self.layer_state.iter_mut() {
            match s {
                LayerState::Full(c) => {
                    let k = read_f32_section(blob, &mut pos)?;
                    let v = read_f32_section(blob, &mut pos)?;
                    if k.len() != v.len() {
                        return Err("import_prefix: K/V section length mismatch".to_string());
                    }
                    let stride = c.num_kv_heads * c.head_dim;
                    if stride == 0 || k.len() % stride != 0 {
                        return Err("import_prefix: K section length not a multiple of stride".to_string());
                    }
                    let n = k.len() / stride;
                    if n != seq_len {
                        return Err(format!(
                            "import_prefix: full-attn section length ({n} tok) != header seq_len ({seq_len})"
                        ));
                    }
                    if n > c.max_seq_len {
                        return Err(format!(
                            "import_prefix: prefix length {n} exceeds max_seq_len {}",
                            c.max_seq_len
                        ));
                    }
                    c.k[..k.len()].copy_from_slice(&k);
                    c.v[..v.len()].copy_from_slice(&v);
                    c.seq_len = n;
                }
                LayerState::Linear(d) => {
                    let conv = read_f32_section(blob, &mut pos)?;
                    let state = read_f32_section(blob, &mut pos)?;
                    if conv.len() != d.conv_state.len() || state.len() != d.state.len() {
                        return Err("import_prefix: DeltaNet state size mismatch".to_string());
                    }
                    d.conv_state.copy_from_slice(&conv);
                    d.state.copy_from_slice(&state);
                }
            }
        }
        Ok(seq_len)
    }

    fn w(&self, name: &str) -> Vec<f32> {
        self.weights.f32_slice(name).to_vec()
    }

    /// Run decoder layers `0..n` starting from a given hidden state (no embed,
    /// no final norm / lm_head). Used for layer-truncated parity vs MLX (B3).
    /// (Equivalent to `forward_pp_range(hidden, pos, 0, n)` on a stage-0 model.)
    pub fn forward_layers_from_hidden(&mut self, hidden_in: &[f32], pos: usize, n: usize) -> Vec<f32> {
        self.forward_pp_range(hidden_in, pos, 0, n)
    }

    /// Run the resident decoder layers `[start, end)` (GLOBAL indices) from a
    /// hidden state. No embed, no final norm / lm_head — the PP building block.
    /// Per-layer state is read/written via `state_idx`, so each stage advances
    /// only its own DeltaNet/KV state. `start`/`end` must lie within
    /// `[pp_start, pp_end)`.
    pub fn forward_pp_range(&mut self, hidden_in: &[f32], pos: usize, start: usize, end: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let eps = cfg.rms_norm_eps;
        // KV-offload chunk-boundary capture (CPU reference path): BEFORE
        // token `pos` mutates any state, the host `layer_state` reflects
        // exactly `[0, pos)`. At a chunk boundary (`pos>0 && pos%CHUNK==0`)
        // that IS the state `export_prefix` needs for `kv_cache_store`'s
        // aligned-down boundary — snapshot every resident Linear layer's
        // (conv_state, state) now, since GatedDeltaNet can't rewind to
        // reconstruct it later from the post-prompt live state. See
        // `gdn_boundary`'s doc comment and `export_prefix`'s Linear arm.
        if pos > 0 && pos % crate::kvstore::CHUNK == 0 {
            let boundary = pos;
            let mut snap = std::collections::HashMap::new();
            for layer_idx in start..end {
                if cfg.layer_types[layer_idx] == LayerType::LinearAttention {
                    let si = self.state_idx(layer_idx);
                    if let LayerState::Linear(d) = &self.layer_state[si] {
                        snap.insert(si, (d.conv_state.clone(), d.state.clone()));
                    }
                }
            }
            // Retain-latest: drop any older boundary before merging this
            // stage's slice of the new one in (constant memory; a multi-PP
            // model may call forward_pp_range once per stage per token, each
            // contributing its own layer_state slice for the same boundary).
            self.gdn_boundary.retain(|&b, _| b == boundary);
            self.gdn_boundary.entry(boundary).or_default().extend(snap);
        }
        let mut hidden = hidden_in.to_vec();
        for layer_idx in start..end {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
            // Attention sub-block.
            let residual = hidden.clone();
            let in_ln = self.w(&ln("input_layernorm.weight"));
            let x = cpu_rms_norm(&hidden, &in_ln, eps);
            let attn_out = match cfg.layer_types[layer_idx] {
                LayerType::FullAttention => self.gated_attention(layer_idx, &x, pos),
                LayerType::LinearAttention => self.delta_net(layer_idx, &x),
            };
            let h1: Vec<f32> = residual.iter().zip(&attn_out).map(|(&r, &a)| r + a).collect();

            // MLP sub-block: dense SwiGLU (27B) or MoE (35B-A3B).
            let residual2 = h1.clone();
            let post_ln = self.w(&ln("post_attention_layernorm.weight"));
            let ff_in = cpu_rms_norm(&h1, &post_ln, eps);
            let mlp_out = if cfg.is_moe() {
                self.moe_mlp(layer_idx, &ff_in)
            } else {
                self.dense_mlp(layer_idx, &ff_in)
            };
            hidden = residual2.iter().zip(&mlp_out).map(|(&r, &m)| r + m).collect();
        }
        hidden
    }

    /// Batched-prefill CPU reference (P2, engine-less fallback — see
    /// `plan-batched-prefill.md`): run decoder layers `[start, end)` over
    /// `t_count` tokens from `hidden_in` (`[t_count * hidden_size]` row-major),
    /// reproducing `t_count` sequential `forward_pp_range` calls EXACTLY.
    ///
    /// Per layer this is batched over T (all tokens through the SAME layer
    /// before moving to the next layer) rather than token-major (all layers
    /// for one token before the next) — the two orders are bit-exact for a
    /// causal decoder stack: layer `L`'s per-token computation only depends on
    /// (a) layer `L-1`'s already-fully-computed output for that token, and (b)
    /// layer `L`'s OWN recurrent state advanced through tokens `0..t-1`, which
    /// this method's inner `t` loops (and `delta_net_scan`'s explicit `for t`
    /// loop) advance in the same increasing order `forward_pp_range` would
    /// have, called once per token in sequence. So each layer's per-token
    /// results are identical regardless of which order surrounds it.
    ///
    /// `LinearAttention` layers collapse their T serial `delta_net` calls into
    /// ONE `delta_net_scan` call (proven bit-exact-by-construction, see that
    /// method's doc comment + `delta_net_scan_bit_exact_vs_serial`).
    /// `FullAttention`/MLP sub-blocks stay literally T serial per-token calls
    /// (`gated_attention`, `moe_mlp`/`dense_mlp`) — trivially exact since
    /// that's exactly what serial decode already does.
    pub fn forward_pp_range_batched(
        &mut self,
        hidden_in: &[f32],
        start_pos: usize,
        t_count: usize,
        start: usize,
        end: usize,
    ) -> Vec<f32> {
        self.forward_pp_range_batched_capture(hidden_in, start_pos, t_count, start, end, None)
    }

    /// `forward_pp_range_batched` with an optional Design-A batched-verify GDN
    /// capture sink: when `gdn_capture` is `Some`, each `LinearAttention`
    /// layer's batched input `x` ([t_count,h]) is pushed as `(global_layer_idx,
    /// x)` so a partial-accept rollback can re-scan the committed prefix through
    /// the GDN layers only (option-B, no attn/MoE recompute). `None` preserves
    /// the exact prefill behaviour (and its bit-exact test).
    pub fn forward_pp_range_batched_capture(
        &mut self,
        hidden_in: &[f32],
        start_pos: usize,
        t_count: usize,
        start: usize,
        end: usize,
        mut gdn_capture: Option<&mut Vec<(usize, Vec<f32>)>>,
    ) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let mut hidden = hidden_in.to_vec();
        for layer_idx in start..end {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
            let residual = hidden.clone();
            let in_ln = self.w(&ln("input_layernorm.weight"));
            let x = cpu_rms_norm(&hidden, &in_ln, eps); // batched [T,h]
            let attn_out: Vec<f32> = match cfg.layer_types[layer_idx] {
                LayerType::LinearAttention => {
                    if let Some(cap) = gdn_capture.as_deref_mut() {
                        cap.push((layer_idx, x.clone()));
                    }
                    self.delta_net_scan(layer_idx, &x, t_count)
                }
                LayerType::FullAttention => {
                    let mut out = vec![0.0f32; t_count * h];
                    for ti in 0..t_count {
                        let xt = &x[ti * h..(ti + 1) * h];
                        let o = self.gated_attention(layer_idx, xt, start_pos + ti);
                        out[ti * h..(ti + 1) * h].copy_from_slice(&o);
                    }
                    out
                }
            };
            let h1: Vec<f32> = residual.iter().zip(&attn_out).map(|(&r, &a)| r + a).collect();

            let residual2 = h1.clone();
            let post_ln = self.w(&ln("post_attention_layernorm.weight"));
            let ff_in = cpu_rms_norm(&h1, &post_ln, eps);
            let mlp_out: Vec<f32> = {
                let mut out = vec![0.0f32; t_count * h];
                for ti in 0..t_count {
                    let fi = &ff_in[ti * h..(ti + 1) * h];
                    let o = if cfg.is_moe() { self.moe_mlp(layer_idx, fi) } else { self.dense_mlp(layer_idx, fi) };
                    out[ti * h..(ti + 1) * h].copy_from_slice(&o);
                }
                out
            };
            hidden = residual2.iter().zip(&mlp_out).map(|(&r, &m)| r + m).collect();
            // KvStore 256-token-boundary snapshot hook: NOT WIRED (HELD,
            // pending a KV device-capture fix) — a per-256-boundary snapshot
            // would fire here, once per boundary crossed in
            // [start_pos, start_pos+t_count) (see the batched-prefill plan §5).
        }
        hidden
    }

    /// CPU reference forward for one token. Implements the spec in
    /// docs/qwen35-port.md. PENDING op-by-op validation vs the qwen3.6-mlx
    /// oracle (Milestone A5) before being wired into VulkanModel.
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        // Embedding (no scaling).
        let embed = self.weights.f32_slice("model.embed_tokens.weight");
        let hidden0: Vec<f32> = embed[token_id as usize * h..(token_id as usize + 1) * h].to_vec();

        let hidden = self.forward_layers_from_hidden(&hidden0, pos, cfg.num_hidden_layers);

        let norm_w = self.weights.f32_slice("model.norm.weight");
        let normed = cpu_rms_norm(&hidden, norm_w, eps);
        let lm_w = self.weights.f32_slice(&self.lm_head_name);
        cpu_matmul(&normed, lm_w, 1, h, cfg.vocab_size)
    }

    /// Gated full attention (full_attention layers).
    pub fn gated_attention(&mut self, layer_idx: usize, x: &[f32], pos: usize) -> Vec<f32> {
        let cfg = &self.config;
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
        let ln = |s: &str| format!("model.layers.{layer_idx}.self_attn.{s}");

        // Split-borrow: `self.weights` and `self.layer_state` are disjoint
        // fields, so a live `&self.weights` slice borrow can coexist with a
        // later `&mut self.layer_state[si]` — no clone needed.
        let si = self.state_idx(layer_idx);
        let w = &self.weights;
        let qw = w.f32_slice(&ln("q_proj.weight"));
        let kw = w.f32_slice(&ln("k_proj.weight"));
        let vw = w.f32_slice(&ln("v_proj.weight"));
        let ow = w.f32_slice(&ln("o_proj.weight"));
        let qn = w.f32_slice(&ln("q_norm.weight"));
        let kn = w.f32_slice(&ln("k_norm.weight"));

        // q_proj is double-width: per head [query(hd) | gate(hd)].
        let q_and_gate = cpu_matmul(x, qw, 1, h, nq * hd * 2);
        let mut q = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for head in 0..nq {
            let base = head * 2 * hd;
            q[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base..base + hd]);
            gate[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base + hd..base + 2 * hd]);
        }
        let mut k = cpu_matmul(x, kw, 1, h, kv_dim);
        let v = cpu_matmul(x, vw, 1, h, kv_dim);

        // Per-head Q/K RMSNorm (before RoPE).
        for hi in 0..nq {
            let s = &mut q[hi * hd..(hi + 1) * hd];
            let n = cpu_rms_norm(s, qn, eps);
            s.copy_from_slice(&n);
        }
        for hi in 0..nkv {
            let s = &mut k[hi * hd..(hi + 1) * hd];
            let n = cpu_rms_norm(s, kn, eps);
            s.copy_from_slice(&n);
        }

        // Partial RoPE.
        cpu_rope(&mut q, &mut k, pos, nq, nkv, hd, rotary, theta);

        // KV cache + GQA causal SDPA. State is indexed by stage-local slot
        // (si computed above, before the `w` borrow).
        let attn_out = {
            let cache = match &mut self.layer_state[si] {
                LayerState::Full(c) => c,
                _ => unreachable!("full_attention layer has a KV cache"),
            };
            cache.append(&k, &v);
            cpu_sdpa(&q, cache.k_up_to_now(), cache.v_up_to_now(), nq, nkv, hd, cache.seq_len, scale, None)
        };

        // Output gate then o_proj.
        let gated: Vec<f32> = attn_out.iter().zip(&gate).map(|(&a, &g)| a * sigmoid(g)).collect();
        cpu_matmul(&gated, ow, 1, q_dim, h)
    }

    /// GatedDeltaNet linear attention (linear_attention layers), decode step.
    ///
    /// The recurrent-matrix update is `state = decay*state + k⊗delta`, applied
    /// as two in-place passes (`gdn_apply_decay` then `gdn_apply_delta`) with a
    /// `kv_mem` read (`gdn_kv_mem`) of the ALREADY-decayed state sandwiched in
    /// between to derive `delta` from `(v - kv_mem)*beta`. Those three fns are
    /// `pub(crate)` (not inlined here) so a tape-replay path can reuse the exact
    /// same op sequence with a FROZEN `delta` (skipping the `kv_mem` read) and
    /// stay bit-exact with this forward path — see `gdn_tape_replay_tests`.
    pub fn delta_net(&mut self, layer_idx: usize, x: &[f32]) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let kern = cfg.linear_conv_kernel_dim;
        let ratio = nv / nk;
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");

        // Split-borrow: compute the state slot index before binding the
        // weights borrow (see gated_attention for the same pattern).
        let si = self.state_idx(layer_idx);
        let w = &self.weights;
        let qkv_w = w.f32_slice(&ln("in_proj_qkv.weight"));
        let z_w = w.f32_slice(&ln("in_proj_z.weight"));
        let a_w = w.f32_slice(&ln("in_proj_a.weight"));
        let b_w = w.f32_slice(&ln("in_proj_b.weight"));
        let conv_w = w.f32_slice(&ln("conv1d.weight")); // [conv_dim, 1, kern] row-major
        let a_log = w.f32_slice(&ln("A_log"));
        let dt_bias = w.f32_slice(&ln("dt_bias"));
        let norm_w = w.f32_slice(&ln("norm.weight"));
        let out_w = w.f32_slice(&ln("out_proj.weight"));

        // Projections.
        let qkv = cpu_matmul(x, qkv_w, 1, h, conv_dim);
        let z = cpu_matmul(x, z_w, 1, h, value_dim);
        let a = cpu_matmul(x, a_w, 1, h, nv);
        let b = cpu_matmul(x, b_w, 1, h, nv);

        // Causal depthwise conv1d + SiLU, updating the sliding window state.
        // (si computed above, before the `w` borrow.)
        let win = kern - 1;
        let mut conv_out = vec![0.0f32; conv_dim];
        {
            let st = match &mut self.layer_state[si] {
                LayerState::Linear(d) => d,
                _ => unreachable!("linear_attention layer has a DeltaNet state"),
            };
            for c in 0..conv_dim {
                // window: [conv_state[c, 0..win], qkv[c]]  (length kern)
                let mut acc = 0.0f32;
                for t in 0..win {
                    acc += st.conv_state[c * win + t] * conv_w[c * kern + t];
                }
                acc += qkv[c] * conv_w[c * kern + win];
                conv_out[c] = acc / (1.0 + (-acc).exp()); // silu
                // shift window: drop oldest, append qkv[c]
                if win > 0 {
                    for t in 0..win - 1 {
                        st.conv_state[c * win + t] = st.conv_state[c * win + t + 1];
                    }
                    st.conv_state[c * win + (win - 1)] = qkv[c];
                }
            }
        }

        // Split + per-head RMSNorm(no weight) with the inv_scale / inv_scale^2 scaling.
        let inv = 1.0 / (kd as f32).sqrt();
        let q_flat = &conv_out[..key_dim];
        let k_flat = &conv_out[key_dim..2 * key_dim];
        let v_flat = &conv_out[2 * key_dim..];
        let mut q = vec![0.0f32; key_dim];
        let mut k = vec![0.0f32; key_dim];
        for hi in 0..nk {
            let qn = cpu_rms_norm_no_weight(&q_flat[hi * kd..(hi + 1) * kd], kd, 1e-6);
            let kn = cpu_rms_norm_no_weight(&k_flat[hi * kd..(hi + 1) * kd], kd, 1e-6);
            for j in 0..kd {
                q[hi * kd + j] = qn[j] * inv * inv;
                k[hi * kd + j] = kn[j] * inv;
            }
        }

        // Recurrent delta rule per v-head, gated norm, collect.
        let mut gated = vec![0.0f32; value_dim];
        {
            let st = match &mut self.layer_state[si] {
                LayerState::Linear(d) => d,
                _ => unreachable!(),
            };
            for j in 0..nv {
                let kh = j / ratio; // repeat_interleave: v-head j uses k-head j/ratio
                let q_j = &q[kh * kd..(kh + 1) * kd];
                let k_j = &k[kh * kd..(kh + 1) * kd];
                let v_j = &v_flat[j * vd..(j + 1) * vd];
                let g = -(a_log[j].exp()) * softplus(a[j] + dt_bias[j]);
                let decay = g.exp();
                let beta = sigmoid(b[j]);
                let sb = j * kd * vd; // state base for head j: state[k*vd + v]
                let head_state = &mut st.state[sb..sb + kd * vd];

                // state *= decay (pass 1)
                gdn_apply_decay(head_state, decay);
                // kv_mem[v] = sum_k state[k,v] * k_j[k] (read of the decayed state)
                let kv_mem = gdn_kv_mem(head_state, kd, vd, k_j);
                // delta[v] = (v - kv_mem) * beta
                let mut delta = vec![0.0f32; vd];
                for vv in 0..vd {
                    delta[vv] = (v_j[vv] - kv_mem[vv]) * beta;
                }
                // state[k,v] += k_j[k] * delta[v] (pass 2)
                gdn_apply_delta(head_state, kd, vd, k_j, &delta);
                // output[v] = sum_k state[k,v] * q_j[k]
                let mut out_j = vec![0.0f32; vd];
                for kk in 0..kd {
                    let qv = q_j[kk];
                    for vv in 0..vd {
                        out_j[vv] += st.state[sb + kk * vd + vv] * qv;
                    }
                }
                // gated norm: RMSNorm(out_j, norm_w) * silu(z_j)
                let normed = cpu_rms_norm(&out_j, norm_w, eps);
                for vv in 0..vd {
                    gated[j * vd + vv] = normed[vv] * cpu_silu(&[z[j * vd + vv]])[0];
                }
            }
        }

        cpu_matmul(&gated, out_w, 1, value_dim, h)
    }

    /// Batched pre-pass for `delta_net_scan` (P0): projections + causal
    /// depthwise conv1d/SiLU + per-head qk-norm over `t_count` tokens,
    /// carrying `conv_state` across the token loop. Factored out of
    /// `delta_net_scan` so `debug_gdn_scan` (P1a) can feed the exact same
    /// per-token `(q,k,conv_out,a,b,z)` tensors into the GPU `q35_gdn_scan`
    /// shader that the CPU recurrence below consumes — isolating the
    /// shader-vs-CPU comparison to just the new scan kernel (the pre-pass
    /// projection/conv/qknorm kernels are already validated by
    /// `debug_qwen35_gdn_gpu`).
    ///
    /// Returns `(q, k, conv_out, a, b, z)`, all `[t_count * stride]` row-major
    /// (`key_dim` / `key_dim` / `conv_dim` / `nv` / `nv` / `value_dim` strides).
    #[allow(clippy::type_complexity)]
    pub(crate) fn gdn_scan_prepass(
        &mut self,
        layer_idx: usize,
        xs: &[f32],
        t_count: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let nv = cfg.linear_num_value_heads;
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");

        // NOTE: reads the projection weights straight from the host f32 store
        // (`self.weights`). This is ONLY correct when that store actually
        // holds them — true for the CPU-only forward path (this function's
        // sole caller today, `delta_net_scan`), where the loader always keeps
        // a full f32 copy (no GPU engine -> `on_proj` never returns
        // `Consumed`). The GPU-resident batched-prefill path
        // (`qwen35_forward::qwen35_linear_prefill_gpu`) does NOT call this —
        // it does its own GPU-resident `in_proj_*` projections (matching how
        // `qwen35_delta_net_gpu`/decode reads them) and calls
        // `gdn_scan_conv_qknorm` below with the result, since `in_proj_*` are
        // matvec weights the split-loader streams to the GPU and DROPS from
        // this f32 store on a real lean load.
        let w = &self.weights;
        let qkv_w = w.f32_slice(&ln("in_proj_qkv.weight"));
        let z_w = w.f32_slice(&ln("in_proj_z.weight"));
        let a_w = w.f32_slice(&ln("in_proj_a.weight"));
        let b_w = w.f32_slice(&ln("in_proj_b.weight"));

        // Batched projections: [T,h] @ [out,h]^T -> [T,out]. Each output row is
        // independent of every other row's presence, so this is bit-exact vs T
        // separate `cpu_matmul(.., 1, h, out)` calls (see doc comment).
        let qkv = cpu_matmul(xs, qkv_w, t_count, h, conv_dim); // [T, conv_dim]
        let z = cpu_matmul(xs, z_w, t_count, h, value_dim); // [T, value_dim]
        let a = cpu_matmul(xs, a_w, t_count, h, nv); // [T, nv]
        let b = cpu_matmul(xs, b_w, t_count, h, nv); // [T, nv]

        let (q, k, conv_out) = self.gdn_scan_conv_qknorm(layer_idx, &qkv, t_count);
        (q, k, conv_out, a, b, z)
    }

    /// The conv1d/SiLU + qk-norm tail of `gdn_scan_prepass`, taking an
    /// ALREADY-COMPUTED `qkv` projection (`[T, conv_dim]`) instead of reading
    /// `in_proj_qkv.weight` itself. `conv1d.weight` is NOT a matvec weight
    /// (never GPU-streamed/dropped by the split loader — `is_qwen35_matvec_weight_name`
    /// excludes it), so it's always safe to read from the host f32 store here,
    /// on both the CPU-only and GPU-resident prefill paths. `a`/`b`/`z` need no
    /// further CPU processing (the caller returns/uses them as-is), so they
    /// aren't touched by this half of the split.
    pub(crate) fn gdn_scan_conv_qknorm(
        &mut self,
        layer_idx: usize,
        qkv: &[f32],
        t_count: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let cfg = self.config.clone();
        let nk = cfg.linear_num_key_heads;
        let kd = cfg.linear_key_head_dim;
        let key_dim = cfg.key_dim();
        let conv_dim = cfg.conv_dim();
        let kern = cfg.linear_conv_kernel_dim;
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");

        let si = self.state_idx(layer_idx);
        let conv_w = self.weights.f32_slice(&ln("conv1d.weight")); // [conv_dim, 1, kern] row-major

        // Batched causal depthwise conv1d + SiLU, carrying conv_state across T.
        // Channel `c`'s window state evolves independently of every other
        // channel, so nesting `c` outer / `t` inner (vs. `delta_net`'s implicit
        // `t` outer / `c` inner across separate calls) does not change any
        // single output element's accumulation order.
        let win = kern - 1;
        let mut conv_out = vec![0.0f32; t_count * conv_dim];
        {
            let st = match &mut self.layer_state[si] {
                LayerState::Linear(d) => d,
                _ => unreachable!("linear_attention layer has a DeltaNet state"),
            };
            for c in 0..conv_dim {
                for t in 0..t_count {
                    let mut acc = 0.0f32;
                    for wi in 0..win {
                        acc += st.conv_state[c * win + wi] * conv_w[c * kern + wi];
                    }
                    let cur = qkv[t * conv_dim + c];
                    acc += cur * conv_w[c * kern + win];
                    conv_out[t * conv_dim + c] = acc / (1.0 + (-acc).exp()); // silu
                    // shift window: drop oldest, append cur
                    if win > 0 {
                        for wi in 0..win - 1 {
                            st.conv_state[c * win + wi] = st.conv_state[c * win + wi + 1];
                        }
                        st.conv_state[c * win + (win - 1)] = cur;
                    }
                }
            }
        }

        // Split + per-head RMSNorm(no weight) with the inv_scale / inv_scale^2
        // scaling, batched over [T, nk] (same per-head math as `delta_net`).
        let inv = 1.0 / (kd as f32).sqrt();
        let mut q = vec![0.0f32; t_count * key_dim];
        let mut k = vec![0.0f32; t_count * key_dim];
        for t in 0..t_count {
            let base = t * conv_dim;
            let q_flat = &conv_out[base..base + key_dim];
            let k_flat = &conv_out[base + key_dim..base + 2 * key_dim];
            for hi in 0..nk {
                let qn = cpu_rms_norm_no_weight(&q_flat[hi * kd..(hi + 1) * kd], kd, 1e-6);
                let kn = cpu_rms_norm_no_weight(&k_flat[hi * kd..(hi + 1) * kd], kd, 1e-6);
                for j in 0..kd {
                    q[t * key_dim + hi * kd + j] = qn[j] * inv * inv;
                    k[t * key_dim + hi * kd + j] = kn[j] * inv;
                }
            }
        }

        (q, k, conv_out)
    }

    /// GatedDeltaNet linear attention, MULTI-TOKEN scan (P0, batched prefill
    /// foundation — see `plan-batched-prefill.md` §1d). Processes `t_count`
    /// tokens in ONE call, carrying `(conv_state, state)` across the inner
    /// token loop.
    ///
    /// Construction note: the projections (`qkv/z/a/b`), the causal depthwise
    /// conv1d+SiLU, and the per-head qk-norm are batched pre-passes
    /// (`gdn_scan_prepass`, each output element's accumulation independent of
    /// `t_count` — same channel/head math, just looped over T instead of
    /// called T times). The delta-rule recurrence itself is NOT batchable
    /// (read-before-write on `state`) and is kept as an explicit `for t in
    /// 0..t_count { for j in 0..nv { ... } }` loop with EXACTLY the same
    /// per-step body + accumulation order as `delta_net`'s single-token path,
    /// so this is bit-exact by construction vs `t_count` sequential
    /// `delta_net` calls (see `delta_net_scan_bit_exact_vs_serial` below).
    ///
    /// `xs`: `[t_count * hidden_size]` row-major. Returns `[t_count *
    /// hidden_size]` row-major (post `out_proj`).
    pub fn delta_net_scan(&mut self, layer_idx: usize, xs: &[f32], t_count: usize) -> Vec<f32> {
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim = cfg.key_dim();
        let value_dim = cfg.value_dim();
        let conv_dim = cfg.conv_dim();
        let ratio = nv / cfg.linear_num_key_heads;
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");

        let si = self.state_idx(layer_idx);
        let a_log = self.weights.f32_slice(&ln("A_log")).to_vec();
        let dt_bias = self.weights.f32_slice(&ln("dt_bias")).to_vec();
        let norm_w = self.weights.f32_slice(&ln("norm.weight")).to_vec();
        let out_w = self.weights.f32_slice(&ln("out_proj.weight")).to_vec();

        let (q, k, conv_out, a, b, z) = self.gdn_scan_prepass(layer_idx, xs, t_count);

        // Sequential recurrent delta rule: `for t { for j in 0..nv { ... } }`,
        // matching the exact per-step body + accumulation order of `delta_net`
        // called once per token in sequence (state carried across the outer
        // `t` loop).
        let mut gated = vec![0.0f32; t_count * value_dim];
        {
            let st = match &mut self.layer_state[si] {
                LayerState::Linear(d) => d,
                _ => unreachable!(),
            };
            for t in 0..t_count {
                let a_t = &a[t * nv..(t + 1) * nv];
                let b_t = &b[t * nv..(t + 1) * nv];
                let q_t = &q[t * key_dim..(t + 1) * key_dim];
                let k_t = &k[t * key_dim..(t + 1) * key_dim];
                let v_t = &conv_out[t * conv_dim + 2 * key_dim..t * conv_dim + conv_dim];
                let z_t = &z[t * value_dim..(t + 1) * value_dim];
                for j in 0..nv {
                    let kh = j / ratio; // repeat_interleave: v-head j uses k-head j/ratio
                    let q_j = &q_t[kh * kd..(kh + 1) * kd];
                    let k_j = &k_t[kh * kd..(kh + 1) * kd];
                    let v_j = &v_t[j * vd..(j + 1) * vd];
                    let g = -(a_log[j].exp()) * softplus(a_t[j] + dt_bias[j]);
                    let decay = g.exp();
                    let beta = sigmoid(b_t[j]);
                    let sb = j * kd * vd; // state base for head j: state[k*vd + v]

                    // state *= decay
                    for e in 0..kd * vd {
                        st.state[sb + e] *= decay;
                    }
                    // kv_mem[v] = sum_k state[k,v] * k_j[k]
                    let mut kv_mem = vec![0.0f32; vd];
                    for kk in 0..kd {
                        let kv = k_j[kk];
                        for vv in 0..vd {
                            kv_mem[vv] += st.state[sb + kk * vd + vv] * kv;
                        }
                    }
                    // delta[v] = (v - kv_mem) * beta ; state[k,v] += k_j[k] * delta[v]
                    let mut delta = vec![0.0f32; vd];
                    for vv in 0..vd {
                        delta[vv] = (v_j[vv] - kv_mem[vv]) * beta;
                    }
                    for kk in 0..kd {
                        let kv = k_j[kk];
                        for vv in 0..vd {
                            st.state[sb + kk * vd + vv] += kv * delta[vv];
                        }
                    }
                    // output[v] = sum_k state[k,v] * q_j[k]
                    let mut out_j = vec![0.0f32; vd];
                    for kk in 0..kd {
                        let qv = q_j[kk];
                        for vv in 0..vd {
                            out_j[vv] += st.state[sb + kk * vd + vv] * qv;
                        }
                    }
                    // gated norm: RMSNorm(out_j, norm_w) * silu(z_j)
                    let normed = cpu_rms_norm(&out_j, &norm_w, eps);
                    for vv in 0..vd {
                        gated[t * value_dim + j * vd + vv] =
                            normed[vv] * cpu_silu(&[z_t[j * vd + vv]])[0];
                    }
                }
            }
        }

        cpu_matmul(&gated, &out_w, t_count, value_dim, h)
    }

    /// Public wrapper for `dense_mlp` (TP-invariance validator uses it on a
    /// sharded-config model to get this rank's row-sharded down_proj partial).
    pub fn dense_mlp_pub(&mut self, layer_idx: usize, ff_in: &[f32]) -> Vec<f32> {
        self.dense_mlp(layer_idx, ff_in)
    }

    /// Dense SwiGLU MLP.
    fn dense_mlp(&self, layer_idx: usize, ff_in: &[f32]) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let ln = |s: &str| format!("model.layers.{layer_idx}.mlp.{s}");
        let w = &self.weights;
        let gate_w = w.f32_slice(&ln("gate_proj.weight"));
        let up_w = w.f32_slice(&ln("up_proj.weight"));
        let down_w = w.f32_slice(&ln("down_proj.weight"));
        let gate = cpu_matmul(ff_in, gate_w, 1, h, inter);
        let up = cpu_matmul(ff_in, up_w, 1, h, inter);
        let act = cpu_silu(&gate);
        let mid: Vec<f32> = act.iter().zip(&up).map(|(&g, &u)| g * u).collect();
        cpu_matmul(&mid, down_w, 1, inter, h)
    }

    /// MoE MLP (35B-A3B): top-k routed experts + shared expert. Borrows the
    /// dequantized expert tensors from the weight store (no per-token copy).
    fn moe_mlp(&mut self, layer_idx: usize, ff_in: &[f32]) -> Vec<f32> {
        let cfg = &self.config;
        let dims = crate::moe::MoeDims {
            hidden: cfg.hidden_size,
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            moe_inter: cfg.moe_intermediate_size,
            shared_inter: cfg.shared_expert_intermediate_size,
        };
        let (_routing, out) = if self.quant_moe.gate.contains_key(&layer_idx) {
            // 4-bit-resident experts: dequant only the routed ones on the fly.
            moe::moe_forward_token_quant(ff_in, &self.weights, &self.quant_moe, layer_idx, dims)
        } else {
            // f32-host experts (parity / single-node).
            moe::moe_forward_token_borrowed(ff_in, &self.weights, layer_idx, dims)
        };
        out
    }
}

#[cfg(test)]
mod spec_rollback_tests {
    //! P1 speculative-rollback host-state gate (Mac-runnable, engine-less).
    //! Mirrors THE GATE on the pure-CPU path (host DeltaNetState + KV counter
    //! are authoritative when no Vulkan engine is present): run t1 → snapshot →
    //! advance t2..t4 → restore → re-run t2..t4 → assert every emitted vector
    //! AND all mutable state is bit-identical to the never-speculated run.
    use super::*;
    use crate::model::{ModelWeights, SimpleTensor};
    use std::collections::HashMap;

    /// Deterministic xorshift weight/input generator (same shape as the debug
    /// probes) so the whole test is reproducible without an RNG dependency.
    fn gen(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// A tiny hybrid stage: layer 0 = GatedDeltaNet (linear), layer 1 = gated
    /// full attention. Just enough weights to drive `delta_net` + `gated_attention`
    /// (the two ops that mutate per-token state); MLP/layernorms are not needed
    /// because the test calls those sub-blocks directly.
    fn build_model() -> Qwen35Model {
        let h = 32usize;
        let (nq, nkv, hd) = (4usize, 2usize, 8usize);
        let (nk, nv, kd, vd, kern) = (1usize, 2usize, 8usize, 8usize, 4usize);
        let cfg = Qwen35Config {
            hidden_size: h,
            num_hidden_layers: 2,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            num_attention_heads: nq,
            num_key_value_heads: nkv,
            head_dim: hd,
            attn_output_gate: true,
            rope_theta: 1e7,
            partial_rotary_factor: 0.5, // rotary_dim = 4 (even)
            linear_num_key_heads: nk,
            linear_num_value_heads: nv,
            linear_key_head_dim: kd,
            linear_value_head_dim: vd,
            linear_conv_kernel_dim: kern,
            intermediate_size: 16,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        };
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = 2 * key_dim + value_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let mut g = gen(0xC0FFEE);
        let mut mk = |rows: usize, cols: usize, scale: f32| -> Vec<f32> {
            (0..rows * cols).map(|_| g() * scale).collect()
        };
        let mut tensors: HashMap<String, SimpleTensor> = HashMap::new();
        let mut ins = |name: String, data: Vec<f32>| {
            tensors.insert(name, SimpleTensor { data, shape: vec![] });
        };
        // Layer 0: linear_attn.
        let p0 = "model.layers.0.linear_attn";
        ins(format!("{p0}.in_proj_qkv.weight"), mk(conv_dim, h, 0.05));
        ins(format!("{p0}.in_proj_z.weight"), mk(value_dim, h, 0.05));
        ins(format!("{p0}.in_proj_a.weight"), mk(nv, h, 0.05));
        ins(format!("{p0}.in_proj_b.weight"), mk(nv, h, 0.05));
        ins(format!("{p0}.conv1d.weight"), mk(conv_dim, kern, 0.5));
        ins(format!("{p0}.A_log"), mk(nv, 1, 1.0));
        ins(format!("{p0}.dt_bias"), mk(nv, 1, 0.5));
        ins(format!("{p0}.norm.weight"), mk(vd, 1, 0.1).iter().map(|&v| 1.0 + v).collect());
        ins(format!("{p0}.out_proj.weight"), mk(h, value_dim, 0.05));
        // Layer 1: self_attn (gated full attention).
        let p1 = "model.layers.1.self_attn";
        ins(format!("{p1}.q_proj.weight"), mk(q_dim * 2, h, 0.05));
        ins(format!("{p1}.k_proj.weight"), mk(kv_dim, h, 0.05));
        ins(format!("{p1}.v_proj.weight"), mk(kv_dim, h, 0.05));
        ins(format!("{p1}.o_proj.weight"), mk(h, q_dim, 0.05));
        ins(format!("{p1}.q_norm.weight"), mk(hd, 1, 0.1).iter().map(|&v| 1.0 + v).collect());
        ins(format!("{p1}.k_norm.weight"), mk(hd, 1, 0.1).iter().map(|&v| 1.0 + v).collect());
        let weights = ModelWeights { tensors };
        Qwen35Model::new(cfg, weights, 64, "unused".to_string())
    }

    /// One decode step over the tiny stage: advance both stateful ops and return
    /// their concatenated output (the "hidden-like" observable for the gate).
    fn step(m: &mut Qwen35Model, x: &[f32], pos: usize) -> Vec<f32> {
        let mut out = m.delta_net(0, x);
        out.extend(m.gated_attention(1, x, pos));
        out
    }

    /// Fingerprint ALL mutable state (bit-exact): both layers' DeltaNet conv +
    /// delta matrices and the full-attn KV counter + live K/V bytes.
    fn state_bits(m: &Qwen35Model) -> (Vec<f32>, Vec<f32>, usize, Vec<f32>, Vec<f32>) {
        let dn = match &m.layer_state[0] {
            LayerState::Linear(d) => (d.conv_state.clone(), d.state.clone()),
            _ => unreachable!(),
        };
        let kv = match &m.layer_state[1] {
            LayerState::Full(c) => (c.seq_len, c.k_up_to_now().to_vec(), c.v_up_to_now().to_vec()),
            _ => unreachable!(),
        };
        (dn.0, dn.1, kv.0, kv.1, kv.2)
    }

    #[test]
    fn snapshot_restore_is_bit_exact() {
        let mut m = build_model();
        let mut g = gen(7);
        let h = m.config.hidden_size;
        let xs: Vec<Vec<f32>> = (0..4).map(|_| (0..h).map(|_| g()).collect()).collect();

        // ── Reference run: t1..t4 with no speculation. ──────────────────────
        let mut m_ref = build_model();
        let mut ref_out = Vec::new();
        for (pos, x) in xs.iter().enumerate() {
            ref_out.push(step(&mut m_ref, x, pos));
        }
        let ref_state_after4 = state_bits(&m_ref);

        // ── Speculative run: t1 → snapshot → t2..t4 → restore → re-run. ─────
        let empty = std::collections::HashSet::new();
        let o1 = step(&mut m, &xs[0], 0);
        assert_eq!(o1, ref_out[0], "t1 must match reference");
        let snap = m.spec_snapshot_host(&empty);
        let state_at_snap = state_bits(&m);

        // Advance the speculative tokens t2..t4.
        for pos in 1..4 {
            let o = step(&mut m, &xs[pos], pos);
            assert_eq!(o, ref_out[pos], "speculative t{} must match reference", pos + 1);
        }

        // Roll back to the snapshot moment.
        m.spec_restore_host(&snap);
        let state_after_restore = state_bits(&m);
        // Live state (deltanet + KV counter + live K/V range) is bit-identical
        // to the snapshot moment.
        assert_eq!(state_after_restore.0, state_at_snap.0, "conv_state restored");
        assert_eq!(state_after_restore.1, state_at_snap.1, "delta state restored");
        assert_eq!(state_after_restore.2, state_at_snap.2, "kv seq_len rewound");
        assert_eq!(state_after_restore.3, state_at_snap.3, "kv K live-range restored");
        assert_eq!(state_after_restore.4, state_at_snap.4, "kv V live-range restored");

        // Re-run t2..t4: outputs AND final state bit-identical to the reference.
        for pos in 1..4 {
            let o = step(&mut m, &xs[pos], pos);
            assert_eq!(o, ref_out[pos], "re-run t{} bit-identical", pos + 1);
        }
        let rerun_state = state_bits(&m);
        assert_eq!(rerun_state.0, ref_state_after4.0);
        assert_eq!(rerun_state.1, ref_state_after4.1);
        assert_eq!(rerun_state.2, ref_state_after4.2);
        assert_eq!(rerun_state.3, ref_state_after4.3);
        assert_eq!(rerun_state.4, ref_state_after4.4);
    }

    #[test]
    fn snapshot_skips_gpu_authoritative_linear() {
        // When a linear layer's state_idx is in skip set, its DeltaNet state is
        // NOT captured (the resident node path snapshots the GPU buffer instead),
        // but the KV counter still is.
        let mut m = build_model();
        let mut g = gen(11);
        let h = m.config.hidden_size;
        for pos in 0..3 {
            let x: Vec<f32> = (0..h).map(|_| g()).collect();
            step(&mut m, &x, pos);
        }
        let mut skip = std::collections::HashSet::new();
        skip.insert(0usize); // linear layer's state_idx
        let snap = m.spec_snapshot_host(&skip);
        assert!(snap.dn.is_empty(), "GPU-authoritative linear layer skipped");
        assert_eq!(snap.kv.len(), 1, "full-attn KV counter still captured");
        assert_eq!(snap.kv[0], (1usize, 3usize));
    }

    /// Phase-0 gate: the generic `qwen3_5` config parser reads ALL 122B-A10B dims
    /// from the real config.json (dims under `text_config`, rope under
    /// `text_config.rope_parameters`) with nothing hardcoded to the 35B geometry.
    #[test]
    fn parses_qwen35_122b_a10b_config() {
        let ltypes = std::iter::repeat(
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        )
        .take(12)
        .flatten()
        .map(|s| format!("\"{s}\""))
        .collect::<Vec<_>>()
        .join(",");
        let json = format!(
            r#"{{
              "architectures":["Qwen3_5MoeForConditionalGeneration"],
              "model_type":"qwen3_5_moe",
              "tie_word_embeddings":false,
              "quantization":{{"group_size":64,"bits":4,"mode":"affine"}},
              "text_config":{{
                "hidden_size":3072,"num_hidden_layers":48,"vocab_size":248320,
                "num_attention_heads":32,"num_key_value_heads":2,"head_dim":256,
                "attn_output_gate":true,"rms_norm_eps":1e-6,
                "linear_conv_kernel_dim":4,"linear_key_head_dim":128,
                "linear_num_key_heads":16,"linear_num_value_heads":64,
                "linear_value_head_dim":128,
                "moe_intermediate_size":1024,"shared_expert_intermediate_size":1024,
                "num_experts":256,"num_experts_per_tok":8,
                "rope_parameters":{{"rope_theta":10000000,"partial_rotary_factor":0.25}},
                "layer_types":[{ltypes}]
              }}
            }}"#
        );
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let c = Qwen35Config::from_json(&v).expect("122B-A10B config parses");

        assert_eq!(c.hidden_size, 3072);
        assert_eq!(c.num_hidden_layers, 48);
        assert_eq!(c.layer_types.len(), 48);
        assert_eq!(c.vocab_size, 248320);
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.num_key_value_heads, 2);
        assert_eq!(c.head_dim, 256);
        assert!(c.attn_output_gate);
        assert!(!c.tie_word_embeddings);
        assert_eq!(c.rope_theta, 10_000_000.0);
        assert_eq!(c.rotary_dim(), 64); // 256 * 0.25 partial RoPE

        // MoE (256e / top-8 / shared, moe_inter 1024).
        assert!(c.is_moe());
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_intermediate_size, 1024);
        assert_eq!(c.shared_expert_intermediate_size, 1024);

        // GDN dims (64 value heads — larger than the 35B's 32).
        assert_eq!(c.linear_num_key_heads, 16);
        assert_eq!(c.linear_num_value_heads, 64);
        assert_eq!(c.linear_key_head_dim, 128);
        assert_eq!(c.linear_value_head_dim, 128);
        assert_eq!(c.conv_dim(), 12288); // 16*128*2 + 64*128 == in_proj_qkv rows

        // 3:1 hybrid: 12 full-attn (interval 4), 36 linear.
        let full = c.layer_types.iter().filter(|t| **t == LayerType::FullAttention).count();
        assert_eq!(full, 12);
        assert_eq!(c.layer_types[3], LayerType::FullAttention);
        assert_eq!(c.layer_types[0], LayerType::LinearAttention);

        // DeltaNet state must size from the 122B's 64 value heads, not a literal.
        let st = DeltaNetState::new(&c);
        assert_eq!(st.state.len(), 64 * 128 * 128);
        assert_eq!(st.conv_state.len(), 12288 * 3); // conv_dim * (kernel-1)
    }

    /// P0 (batched-prefill foundation): `delta_net_scan(T)` must be BIT-EXACT
    /// vs T sequential `delta_net` calls on the same seeded inputs — the
    /// guardrail for the plan's "gated-decay folding is transcription
    /// sensitive" flag (`plan-batched-prefill.md` §7 P0, §10).
    #[test]
    fn delta_net_scan_bit_exact_vs_serial() {
        const T: usize = 300;

        // Reference: T sequential delta_net(...) calls.
        let mut m_ref = build_model();
        let h = m_ref.config.hidden_size;
        let mut g = gen(0xBADC0FFEE);
        let xs: Vec<Vec<f32>> = (0..T).map(|_| (0..h).map(|_| g()).collect()).collect();
        let mut ref_out = Vec::with_capacity(T * h);
        for x in &xs {
            ref_out.extend(m_ref.delta_net(0, x));
        }
        let ref_dn = match &m_ref.layer_state[0] {
            LayerState::Linear(d) => (d.conv_state.clone(), d.state.clone()),
            _ => unreachable!(),
        };

        // Scan: one call over all T tokens, flattened row-major input.
        let mut m_scan = build_model();
        let xs_flat: Vec<f32> = xs.iter().flatten().copied().collect();
        let scan_out = m_scan.delta_net_scan(0, &xs_flat, T);
        let scan_dn = match &m_scan.layer_state[0] {
            LayerState::Linear(d) => (d.conv_state.clone(), d.state.clone()),
            _ => unreachable!(),
        };

        assert_eq!(scan_out.len(), ref_out.len());
        assert_eq!(scan_out.len(), T * h);

        // Byte-exact (f32::to_bits) comparison of every emitted output token
        // and the final recurrent state.
        let mut mismatches = 0usize;
        for (i, (&a, &b)) in scan_out.iter().zip(ref_out.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!("output mismatch at flat idx {i}: scan={a:?} ({:#x}) ref={b:?} ({:#x})", a.to_bits(), b.to_bits());
                }
            }
        }
        assert_eq!(mismatches, 0, "delta_net_scan output must be byte-exact vs serial delta_net (T={T}, len={})", scan_out.len());

        assert_eq!(scan_dn.0.len(), ref_dn.0.len());
        assert_eq!(scan_dn.1.len(), ref_dn.1.len());
        for (i, (&a, &b)) in scan_dn.0.iter().zip(ref_dn.0.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "final conv_state[{i}] byte-exact");
        }
        for (i, (&a, &b)) in scan_dn.1.iter().zip(ref_dn.1.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "final delta state[{i}] byte-exact");
        }

        // Sanity: not a degenerate all-zero pass (guards against a construction
        // bug that silently produces an empty/zero comparison).
        assert!(ref_out.iter().any(|&v| v != 0.0));
    }
}

/// Bit-exactness gate for the KV-prefix export/import seam (LMCache-NAS
/// plan, Step 1). Mirrors the `spec_rollback_tests` methodology: populate
/// state with seeded pseudo-random data, export, scribble garbage into the
/// live model (proving import isn't a no-op), import, and assert every K/V
/// and DeltaNet element is `f32::to_bits`-identical to the pre-export value.
#[cfg(test)]
pub(crate) mod kv_prefix_tests {
    use super::*;
    use crate::model::{ModelWeights, SimpleTensor};
    use std::collections::HashMap;

    /// Deterministic xorshift generator (same recipe as the other test
    /// modules in this file) — reproducible without an RNG dependency.
    fn gen(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// A weight-less hybrid stage with the given `layer_types` — sufficient
    /// for export/import tests, which only touch `layer_state` (never
    /// `weights`). Same head geometry as `spec_rollback_tests::build_model`.
    /// `pub(crate)`: reused by `kvstore`'s file-round-trip test (LMCache-NAS
    /// plan Step 1) so it exercises the SAME synthetic-model recipe as this
    /// module's own bit-exactness gate instead of a second copy.
    pub(crate) fn build_hybrid(layer_types: Vec<LayerType>) -> Qwen35Model {
        let h = 32usize;
        let (nq, nkv, hd) = (4usize, 2usize, 8usize);
        let (nk, nv, kd, vd, kern) = (1usize, 2usize, 8usize, 8usize, 4usize);
        let n = layer_types.len();
        let cfg = Qwen35Config {
            hidden_size: h,
            num_hidden_layers: n,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            num_attention_heads: nq,
            num_key_value_heads: nkv,
            head_dim: hd,
            attn_output_gate: true,
            rope_theta: 1e7,
            partial_rotary_factor: 0.5,
            linear_num_key_heads: nk,
            linear_num_value_heads: nv,
            linear_key_head_dim: kd,
            linear_value_head_dim: vd,
            linear_conv_kernel_dim: kern,
            intermediate_size: 16,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            layer_types,
        };
        let weights = ModelWeights { tensors: HashMap::new() };
        Qwen35Model::new(cfg, weights, 64, "unused".to_string())
    }

    /// Fill every resident layer with seeded pseudo-random data: full-attn
    /// layers get `full_seq_len` tokens of K/V (and `seq_len` set to it);
    /// linear layers get a fully random conv/delta state (their state is
    /// fixed-size, independent of any sequence position).
    pub(crate) fn populate(m: &mut Qwen35Model, full_seq_len: usize, seed: u64) {
        let mut g = gen(seed);
        for s in m.layer_state.iter_mut() {
            match s {
                LayerState::Full(c) => {
                    let n = full_seq_len * c.num_kv_heads * c.head_dim;
                    for i in 0..n {
                        c.k[i] = g();
                        c.v[i] = g();
                    }
                    c.seq_len = full_seq_len;
                }
                LayerState::Linear(d) => {
                    for x in d.conv_state.iter_mut() {
                        *x = g();
                    }
                    for x in d.state.iter_mut() {
                        *x = g();
                    }
                }
            }
        }
    }

    /// Scribble different garbage over every resident layer's live arrays —
    /// used between export and import to prove import actually overwrites
    /// state rather than the test passing vacuously because nothing changed.
    fn scribble(m: &mut Qwen35Model, seed: u64) {
        let mut g = gen(seed);
        for s in m.layer_state.iter_mut() {
            match s {
                LayerState::Full(c) => {
                    for x in c.k.iter_mut() {
                        *x = g();
                    }
                    for x in c.v.iter_mut() {
                        *x = g();
                    }
                    c.seq_len = 0;
                }
                LayerState::Linear(d) => {
                    for x in d.conv_state.iter_mut() {
                        *x = g();
                    }
                    for x in d.state.iter_mut() {
                        *x = g();
                    }
                }
            }
        }
    }

    enum Expected {
        Full { k: Vec<f32>, v: Vec<f32> },
        Linear { conv: Vec<f32>, state: Vec<f32> },
    }

    fn capture_expected(m: &Qwen35Model, boundary: usize) -> Vec<Expected> {
        m.layer_state
            .iter()
            .map(|s| match s {
                LayerState::Full(c) => Expected::Full {
                    k: c.k_upto(boundary).to_vec(),
                    v: c.v_upto(boundary).to_vec(),
                },
                LayerState::Linear(d) => Expected::Linear {
                    conv: d.conv_state.clone(),
                    state: d.state.clone(),
                },
            })
            .collect()
    }

    fn assert_bits_eq(a: &[f32], b: &[f32], msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{msg}: element {i} not bit-exact");
        }
    }

    /// Build, populate, export at `boundary`, scribble garbage, import, and
    /// assert every element is bit-exact to the pre-export snapshot.
    fn roundtrip_case(layer_types: Vec<LayerType>, full_seq_len: usize, boundary: usize, seed: u64) {
        let mut m = build_hybrid(layer_types);
        populate(&mut m, full_seq_len, seed);
        let expected = capture_expected(&m, boundary);

        let blob = m.export_prefix(boundary).expect("export_prefix must succeed");
        scribble(&mut m, seed.wrapping_add(0xA5A5));

        let loaded = m.import_prefix(&blob).expect("import_prefix must succeed");
        assert_eq!(loaded, boundary, "import_prefix must report the boundary as the loaded length");

        for (s, exp) in m.layer_state.iter().zip(expected.iter()) {
            match (s, exp) {
                (LayerState::Full(c), Expected::Full { k, v }) => {
                    assert_eq!(c.seq_len, boundary, "seq_len must be set to the boundary");
                    assert_bits_eq(c.k_upto(boundary), k, "K");
                    assert_bits_eq(c.v_upto(boundary), v, "V");
                }
                (LayerState::Linear(d), Expected::Linear { conv, state }) => {
                    assert_bits_eq(&d.conv_state, conv, "conv_state");
                    assert_bits_eq(&d.state, state, "delta state");
                }
                _ => unreachable!("layer-type mismatch between model and captured expectation"),
            }
        }
    }

    #[test]
    fn export_import_roundtrip_bitexact() {
        let two_layer = || vec![LayerType::LinearAttention, LayerType::FullAttention];

        // boundary == seq_len (full prefix).
        roundtrip_case(two_layer(), 5, 5, 1);
        // boundary < seq_len (partial prefix).
        roundtrip_case(two_layer(), 5, 2, 2);
        // boundary == 0 (empty prefix — only the fixed-size DeltaNet state
        // is meaningful; full-attn K/V sections are zero-length).
        roundtrip_case(two_layer(), 5, 0, 3);
        // Multi-layer mix: interleaved linear/full, 5 layers, partial boundary.
        roundtrip_case(
            vec![
                LayerType::LinearAttention,
                LayerType::FullAttention,
                LayerType::LinearAttention,
                LayerType::FullAttention,
                LayerType::LinearAttention,
            ],
            6,
            4,
            4,
        );
    }

    #[test]
    fn import_rejects_fingerprint_mismatch() {
        let layer_types = vec![LayerType::LinearAttention, LayerType::FullAttention];

        // (a) Byte-level corruption of the fingerprint field (bytes [8..16),
        // right after the 4-byte magic + 4-byte version).
        let mut m = build_hybrid(layer_types.clone());
        populate(&mut m, 3, 5);
        let mut blob = m.export_prefix(3).expect("export_prefix must succeed");
        blob[8] ^= 0xFF;
        let err = m.import_prefix(&blob).expect_err("corrupted fingerprint must be rejected");
        assert!(err.contains("fingerprint mismatch"), "unexpected error: {err}");
        // The model's live state must be untouched by the rejected import.
        assert_eq!(m.layer_state.len(), 2);

        // (b) A structurally different model (different full-attn head_dim)
        // rejects a blob exported from the model above — a real mismatch,
        // not just synthetic byte corruption.
        let mut m_a = build_hybrid(layer_types.clone());
        populate(&mut m_a, 3, 6);
        let blob_a = m_a.export_prefix(3).expect("export_prefix must succeed");
        let mut m_b = build_hybrid(layer_types);
        m_b.config.head_dim = m_b.config.head_dim + 4; // diverge the fingerprint
        let err = m_b
            .import_prefix(&blob_a)
            .expect_err("blob from a differently-configured model must be rejected");
        assert!(err.contains("fingerprint mismatch"), "unexpected error: {err}");

        // (c) Bad magic is rejected without panicking.
        let mut bad_magic = blob_a.clone();
        bad_magic[0] = b'X';
        let err = m_a.import_prefix(&bad_magic).expect_err("bad magic must be rejected");
        assert!(err.contains("bad magic"), "unexpected error: {err}");
    }

    /// A 2-layer hybrid stage WITH real weights, sufficient to drive
    /// `delta_net`/`gated_attention` directly (same recipe as
    /// `spec_rollback_tests::build_model`) — needed only by the
    /// forward-survival gate below.
    fn build_fwd_model() -> Qwen35Model {
        let h = 32usize;
        let (nq, nkv, hd) = (4usize, 2usize, 8usize);
        let (nk, nv, kd, vd, kern) = (1usize, 2usize, 8usize, 8usize, 4usize);
        let cfg = Qwen35Config {
            hidden_size: h,
            num_hidden_layers: 2,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            num_attention_heads: nq,
            num_key_value_heads: nkv,
            head_dim: hd,
            attn_output_gate: true,
            rope_theta: 1e7,
            partial_rotary_factor: 0.5,
            linear_num_key_heads: nk,
            linear_num_value_heads: nv,
            linear_key_head_dim: kd,
            linear_value_head_dim: vd,
            linear_conv_kernel_dim: kern,
            intermediate_size: 16,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        };
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = 2 * key_dim + value_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let mut g = gen(0xC0FFEE);
        let mut mk = |rows: usize, cols: usize, scale: f32| -> Vec<f32> {
            (0..rows * cols).map(|_| g() * scale).collect()
        };
        let mut tensors: HashMap<String, SimpleTensor> = HashMap::new();
        let mut ins = |name: String, data: Vec<f32>| {
            tensors.insert(name, SimpleTensor { data, shape: vec![] });
        };
        let p0 = "model.layers.0.linear_attn";
        ins(format!("{p0}.in_proj_qkv.weight"), mk(conv_dim, h, 0.05));
        ins(format!("{p0}.in_proj_z.weight"), mk(value_dim, h, 0.05));
        ins(format!("{p0}.in_proj_a.weight"), mk(nv, h, 0.05));
        ins(format!("{p0}.in_proj_b.weight"), mk(nv, h, 0.05));
        ins(format!("{p0}.conv1d.weight"), mk(conv_dim, kern, 0.5));
        ins(format!("{p0}.A_log"), mk(nv, 1, 1.0));
        ins(format!("{p0}.dt_bias"), mk(nv, 1, 0.5));
        ins(format!("{p0}.norm.weight"), mk(vd, 1, 0.1).iter().map(|&v| 1.0 + v).collect());
        ins(format!("{p0}.out_proj.weight"), mk(h, value_dim, 0.05));
        let p1 = "model.layers.1.self_attn";
        ins(format!("{p1}.q_proj.weight"), mk(q_dim * 2, h, 0.05));
        ins(format!("{p1}.k_proj.weight"), mk(kv_dim, h, 0.05));
        ins(format!("{p1}.v_proj.weight"), mk(kv_dim, h, 0.05));
        ins(format!("{p1}.o_proj.weight"), mk(h, q_dim, 0.05));
        ins(format!("{p1}.q_norm.weight"), mk(hd, 1, 0.1).iter().map(|&v| 1.0 + v).collect());
        ins(format!("{p1}.k_norm.weight"), mk(hd, 1, 0.1).iter().map(|&v| 1.0 + v).collect());
        let weights = ModelWeights { tensors };
        Qwen35Model::new(cfg, weights, 64, "unused".to_string())
    }

    fn fwd_step(m: &mut Qwen35Model, x: &[f32], pos: usize) -> Vec<f32> {
        let mut out = m.delta_net(0, x);
        out.extend(m.gated_attention(1, x, pos));
        out
    }

    #[test]
    fn export_import_survives_a_forward() {
        // Reference: full decode t0..t3, no export/import in the loop.
        let mut m_ref = build_fwd_model();
        let mut g = gen(42);
        let h = m_ref.config.hidden_size;
        let xs: Vec<Vec<f32>> = (0..4).map(|_| (0..h).map(|_| g()).collect()).collect();
        let ref_out: Vec<Vec<f32>> = xs
            .iter()
            .enumerate()
            .map(|(pos, x)| fwd_step(&mut m_ref, x, pos))
            .collect();

        // Decode t0..t1 for real, export the prefix at boundary=2.
        let mut m = build_fwd_model();
        for (pos, x) in xs.iter().enumerate().take(2) {
            let o = fwd_step(&mut m, x, pos);
            assert_eq!(o, ref_out[pos], "t{pos} must match reference before export");
        }
        let blob = m.export_prefix(2).expect("export_prefix must succeed");

        // A FRESH model instance ("cold resume" from the NAS blob, nothing
        // carried over except the weights) imports the prefix and continues
        // decoding t2..t3 — output must be bit-identical to the reference,
        // proving the imported state is a faithful resume point.
        let mut m2 = build_fwd_model();
        let loaded = m2.import_prefix(&blob).expect("import_prefix must succeed");
        assert_eq!(loaded, 2);
        for pos in 2..4 {
            let o = fwd_step(&mut m2, &xs[pos], pos);
            assert_eq!(o, ref_out[pos], "resumed decode t{pos} must match reference bit-exactly");
        }
    }

    /// ITEM 3 gate (session-KV continuation): turn-2 CONTINUATION — keep the KV
    /// alive from turn 1 and feed ONLY the appended tail at `start_pos = L` with
    /// NO reset — is bit-identical to turn-2 FULL REPREFILL (feed the whole
    /// `[0, L+tail)` from a clean cache). This is the load-bearing invariant the
    /// serve-path `SessionKvManager` (`vllm_vulkan/session_kv.py`) relies on to
    /// skip the reset+full-reprefill every turn: continuation is bit-exact by
    /// construction because `forward`/`prefill_logits` are position-addressed and
    /// deterministic (K stored post-RoPE at absolute positions; the linear
    /// layer's recurrent state carries), so resident `[0, L)` + tail-at-`L` == a
    /// clean feed of `[0, L+tail)`. Mirrors `export_import_survives_a_forward`
    /// (the prefix-cache round-trip gate) but for the RESIDENT (no export/import)
    /// continuation path.
    #[test]
    fn session_continuation_matches_full_reprefill() {
        let l = 5usize; // turn-1 resident context length
        let tail = 3usize; // turn-2 appended tokens
        let total = l + tail;
        let mut g = gen(0xABCDEF);
        let h = build_fwd_model().config.hidden_size;
        let xs: Vec<Vec<f32>> = (0..total).map(|_| (0..h).map(|_| g()).collect()).collect();

        // Control: FULL REPREFILL of [0, L+tail) from a clean cache.
        let mut ctrl = build_fwd_model();
        let ctrl_out: Vec<Vec<f32>> = xs
            .iter()
            .enumerate()
            .map(|(pos, x)| fwd_step(&mut ctrl, x, pos))
            .collect();

        // CONTINUATION: feed turn-1 [0, L) to establish residency, then feed ONLY
        // the tail at its absolute positions [L, L+tail) — no reset in between.
        let mut cont = build_fwd_model();
        for (pos, x) in xs.iter().enumerate().take(l) {
            fwd_step(&mut cont, x, pos);
        }
        let cont_tail: Vec<Vec<f32>> = (l..total)
            .map(|pos| fwd_step(&mut cont, &xs[pos], pos))
            .collect();

        // The continuation tail must be bit-exact to the full-reprefill tail —
        // i.e. skipping the re-feed of [0, L) changes NOTHING (argmax-identical
        // trivially follows: identical logits => identical argmax).
        for (i, pos) in (l..total).enumerate() {
            assert_eq!(
                cont_tail[i], ctrl_out[pos],
                "continuation tail at position {pos} must match full reprefill bit-exactly"
            );
        }

        // Negative control (proves residency is load-bearing, so the gate above
        // is not vacuous): a model fed ONLY the tail with NO [0, L) history MUST
        // diverge from the full-reprefill tail.
        let mut cold = build_fwd_model();
        let cold_first = fwd_step(&mut cold, &xs[l], l);
        assert!(
            cold_first != ctrl_out[l],
            "a tail fed without the [0, L) resident history MUST diverge from the \
             full reprefill — otherwise the continuation gate proves nothing"
        );
    }
}

/// Gate for the KV-offload GatedDeltaNet boundary-snapshot fix
/// (`gdn_boundary` + `export_prefix`'s Linear-arm override): proves the
/// state captured in-flight at a `kvstore::CHUNK` boundary is BYTE-EXACT to
/// a fresh, independent forward of exactly that many tokens — i.e. the
/// capture is the true `[0, boundary)` state, uncontaminated by any token
/// past the boundary. This is the property `export_prefix`'s old
/// live-state-only Linear arm violated (it exported the POST-PROMPT state
/// instead of the state at the aligned-down boundary `kv_cache_store` asks
/// for). Does not touch the tape-replay machinery in
/// `gdn_tape_replay_tests` (kept green, unrelated: that's the P1
/// spec-rollback ADAPT target, a different granularity — see
/// `docs`/the KV-offload boundary-fix plan for the A-vs-B tradeoff).
#[cfg(test)]
mod kv_boundary_snapshot_tests {
    use super::*;
    use crate::model::{ModelWeights, SimpleTensor};
    use std::collections::HashMap;

    fn gen(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn assert_bits_eq(a: &[f32], b: &[f32], msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{msg}: element {i} not bit-exact");
        }
    }

    /// Single-layer (GatedDeltaNet-only) model with real MLP weights too, so
    /// `forward_pp_range` can be driven end-to-end (attn sub-block + dense
    /// SwiGLU MLP sub-block) — unlike `kv_prefix_tests::build_fwd_model`,
    /// which only wires enough for the per-op `delta_net`/`gated_attention`
    /// helpers, not the full-layer `forward_pp_range` path this fix's
    /// capture hook lives in.
    fn build_linear_only_model() -> Qwen35Model {
        let h = 32usize;
        let (nk, nv, kd, vd, kern) = (1usize, 2usize, 8usize, 8usize, 4usize);
        let inter = 16usize;
        let cfg = Qwen35Config {
            hidden_size: h,
            num_hidden_layers: 1,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            attn_output_gate: true,
            rope_theta: 1e7,
            partial_rotary_factor: 0.5,
            linear_num_key_heads: nk,
            linear_num_value_heads: nv,
            linear_key_head_dim: kd,
            linear_value_head_dim: vd,
            linear_conv_kernel_dim: kern,
            intermediate_size: inter,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            layer_types: vec![LayerType::LinearAttention],
        };
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = 2 * key_dim + value_dim;
        let mut g = gen(0xC0FFEE);
        let mut mk = |rows: usize, cols: usize, scale: f32| -> Vec<f32> {
            (0..rows * cols).map(|_| g() * scale).collect()
        };
        let mut tensors: HashMap<String, SimpleTensor> = HashMap::new();
        let mut ins = |name: String, data: Vec<f32>| {
            tensors.insert(name, SimpleTensor { data, shape: vec![] });
        };
        let p0 = "model.layers.0";
        ins(format!("{p0}.input_layernorm.weight"), mk(h, 1, 0.1).iter().map(|&v| 1.0 + v).collect());
        ins(format!("{p0}.post_attention_layernorm.weight"), mk(h, 1, 0.1).iter().map(|&v| 1.0 + v).collect());
        let la = format!("{p0}.linear_attn");
        ins(format!("{la}.in_proj_qkv.weight"), mk(conv_dim, h, 0.05));
        ins(format!("{la}.in_proj_z.weight"), mk(value_dim, h, 0.05));
        ins(format!("{la}.in_proj_a.weight"), mk(nv, h, 0.05));
        ins(format!("{la}.in_proj_b.weight"), mk(nv, h, 0.05));
        ins(format!("{la}.conv1d.weight"), mk(conv_dim, kern, 0.5));
        ins(format!("{la}.A_log"), mk(nv, 1, 1.0));
        ins(format!("{la}.dt_bias"), mk(nv, 1, 0.5));
        ins(format!("{la}.norm.weight"), mk(vd, 1, 0.1).iter().map(|&v| 1.0 + v).collect());
        ins(format!("{la}.out_proj.weight"), mk(h, value_dim, 0.05));
        ins(format!("{p0}.mlp.gate_proj.weight"), mk(inter, h, 0.05));
        ins(format!("{p0}.mlp.up_proj.weight"), mk(inter, h, 0.05));
        ins(format!("{p0}.mlp.down_proj.weight"), mk(h, inter, 0.05));
        let weights = ModelWeights { tensors };
        Qwen35Model::new(cfg, weights, 64, "unused".to_string())
    }

    #[test]
    fn boundary_snapshot_is_bitexact_to_a_fresh_forward() {
        const CHUNK: usize = crate::kvstore::CHUNK;
        let n_extra = 44;
        let n_total = CHUNK + n_extra;

        // Same token-embedding inputs for both models over [0, CHUNK) — a
        // shared seed drives one xs vector long enough for the N-token run;
        // the 256-token run consumes its prefix.
        let mut g = gen(42);
        let h = 32usize;
        let xs: Vec<Vec<f32>> = (0..n_total).map(|_| (0..h).map(|_| g()).collect()).collect();

        // (A) Run N = CHUNK + 44 tokens; the capture hook in
        // `forward_pp_range` snapshots `gdn_boundary[CHUNK]` in-flight, BEFORE
        // token CHUNK mutates state (so it holds exactly [0, CHUNK)`), even
        // though the model keeps decoding another 44 tokens past it.
        let mut m_long = build_linear_only_model();
        for pos in 0..n_total {
            let _ = m_long.forward_pp_range(&xs[pos], pos, 0, 1);
        }
        assert!(
            m_long.gdn_boundary.contains_key(&CHUNK),
            "expected a captured snapshot at boundary {CHUNK}"
        );
        let (cap_conv, cap_state) = m_long.gdn_boundary[&CHUNK][&0].clone();

        // (B) Independent fresh model, run EXACTLY CHUNK tokens (same first
        // CHUNK inputs) — its live state after the loop IS `[0, CHUNK)`, by
        // construction, with no contamination possible (nothing ran past it).
        let mut m_fresh = build_linear_only_model();
        for pos in 0..CHUNK {
            let _ = m_fresh.forward_pp_range(&xs[pos], pos, 0, 1);
        }
        let (fresh_conv, fresh_state) = match &m_fresh.layer_state[0] {
            LayerState::Linear(d) => (d.conv_state.clone(), d.state.clone()),
            _ => unreachable!(),
        };

        // The core gate: captured-at-boundary-mid-run == fresh-256-token-run,
        // byte for byte.
        assert_bits_eq(&cap_conv, &fresh_conv, "conv_state @ boundary");
        assert_bits_eq(&cap_state, &fresh_state, "delta state @ boundary");

        // Blob-level: export_prefix(CHUNK) on the long run must match
        // export_prefix(CHUNK) on the fresh run (this is exactly what
        // `kv_cache_store`'s align-down does after a full prefill).
        let blob_long = m_long.export_prefix(CHUNK).expect("export_prefix(CHUNK) on long run");
        let blob_fresh = m_fresh.export_prefix(CHUNK).expect("export_prefix(CHUNK) on fresh run");
        assert_eq!(blob_long, blob_fresh, "export_prefix(CHUNK) blobs must match byte-for-byte");

        // And importing the long run's boundary export into a THIRD fresh
        // model reproduces the fresh-256 state byte-exact end to end.
        let mut m_import = build_linear_only_model();
        let loaded = m_import.import_prefix(&blob_long).expect("import_prefix must succeed");
        assert_eq!(loaded, CHUNK);
        match &m_import.layer_state[0] {
            LayerState::Linear(d) => {
                assert_bits_eq(&d.conv_state, &fresh_conv, "imported conv_state @ boundary");
                assert_bits_eq(&d.state, &fresh_state, "imported delta state @ boundary");
            }
            _ => unreachable!(),
        }
    }

    /// Without the fix, `export_prefix` at a boundary strictly less than the
    /// live position would export the LIVE (post-prompt) state instead —
    /// this asserts that would-be-buggy state is in fact DIFFERENT from the
    /// correct boundary snapshot, so the previous test's equality isn't
    /// vacuously true (e.g. because the state stopped changing after
    /// CHUNK tokens).
    #[test]
    fn live_state_past_the_boundary_actually_differs() {
        const CHUNK: usize = crate::kvstore::CHUNK;
        let n_total = CHUNK + 44;
        let mut g = gen(7);
        let h = 32usize;
        let xs: Vec<Vec<f32>> = (0..n_total).map(|_| (0..h).map(|_| g()).collect()).collect();

        let mut m = build_linear_only_model();
        for pos in 0..n_total {
            let _ = m.forward_pp_range(&xs[pos], pos, 0, 1);
        }
        let (live_conv, live_state) = match &m.layer_state[0] {
            LayerState::Linear(d) => (d.conv_state.clone(), d.state.clone()),
            _ => unreachable!(),
        };
        let (cap_conv, cap_state) = m.gdn_boundary[&CHUNK][&0].clone();
        assert_ne!(
            live_conv.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            cap_conv.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "live conv_state must have moved on past the boundary"
        );
        assert_ne!(
            live_state.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            cap_state.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "live delta state must have moved on past the boundary"
        );
    }
}

/// De-risking gate for the PARKED delta-tape rollback design (see
/// `audit-gdn-rollback-vs-tape-replay.md`, §Q1). Design lineage: the tape
/// decomposition (freeze `delta` per position, replay `state = decay*state +
/// k⊗delta` from a base snapshot) is the same idea as MTPLX's
/// `mtplx_linear_gated_delta_from_conv_tape_v1` / `..._replay_v1`
/// (`mtplx/gdn_capture.py`, Apache-2.0, `github.com/mtplx/mtplx`) — the
/// algorithm is reimplemented here from the public description against our
/// own f32 CPU reference (`gdn_apply_decay` / `gdn_kv_mem` / `gdn_apply_delta`
/// in this module); no MTPLX code is copied.
///
/// Q1's claim: replaying the recurrent matrix from a frozen
/// `{k_normed, decay, delta}` tape, in the IDENTICAL f32 op order as the
/// forward pass, reproduces the forward state trajectory bit-for-bit — the
/// conv FIFO is explicitly out of tape scope (per-position snapshot instead,
/// per the audit) so this module tests only the recurrent-matrix state.
#[cfg(test)]
mod gdn_tape_replay_tests {
    use super::*;

    /// Real 35B GDN per-layer head geometry (memory-note anchors: key_dim
    /// 2048 = nk(16)*kd(128), value_dim 4096 = nv(32)*vd(128)). This test
    /// exercises the recurrent-matrix state update directly at head
    /// granularity (post `repeat_interleave`), so `nk` does not enter — only
    /// `nv`, `kd`, `vd` matter, and they are the real values.
    const NV: usize = 32;
    const KD: usize = 128;
    const VD: usize = 128;

    /// Deterministic xorshift generator (same recipe as `spec_rollback_tests::gen`)
    /// so the synthetic tape is reproducible without an RNG dependency.
    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// One head's frozen tape entry for one position: `k_normed`, `decay`,
    /// and `delta` — exactly the three quantities the audit says must be
    /// frozen (§Q1) because they are the only ones the replay path needs and
    /// the only one at risk of a recompute-divergence (`k_normed`, via a
    /// separate RMSNorm reduction on the GPU path; not a risk here on CPU,
    /// but frozen anyway to test the tape contract itself).
    #[derive(Clone)]
    struct TapeHead {
        k: Vec<f32>,
        decay: f32,
        delta: Vec<f32>,
    }

    /// One position's tape across all `NV` heads.
    #[derive(Clone)]
    struct TapePos {
        heads: Vec<TapeHead>,
    }

    /// Per-position synthetic inputs that are NOT on the tape (they mirror
    /// `k_normed`'s upstream siblings `v` and `beta`, which the forward path
    /// re-derives `delta` from via a fresh `kv_mem` read each time — the tape
    /// freezes only the OUTPUT of that derivation).
    struct PosInputs {
        k: Vec<Vec<f32>>,     // [head][kd] -- frozen on the tape too (k_normed)
        v: Vec<Vec<f32>>,     // [head][vd]
        decay: Vec<f32>,      // [head]
        beta: Vec<f32>,       // [head]
    }

    fn gen_pos_inputs(g: &mut impl FnMut() -> f32) -> PosInputs {
        let mut k = Vec::with_capacity(NV);
        let mut v = Vec::with_capacity(NV);
        let mut decay = Vec::with_capacity(NV);
        let mut beta = Vec::with_capacity(NV);
        for _ in 0..NV {
            k.push((0..KD).map(|_| g() * 0.3).collect());
            v.push((0..VD).map(|_| g() * 0.3).collect());
            // Keep decay/beta away from the 0/1 edges (both directions matter:
            // decay=1 would hide a decay/delta-order bug; decay=0 would hide
            // a stale-state-read bug).
            decay.push(0.85 + 0.10 * ((g() + 1.0) * 0.5));
            beta.push(0.3 + 0.4 * ((g() + 1.0) * 0.5));
        }
        PosInputs { k, v, decay, beta }
    }

    /// THE forward path: identical op sequence to `Qwen35Model::delta_net`'s
    /// recurrent-matrix block (lines around `gdn_apply_decay` / `gdn_kv_mem` /
    /// `gdn_apply_delta` in this file) — calls those exact `pub(crate)` fns,
    /// not a reimplementation. Mutates `state` in place and returns the tape
    /// entry (`k`, `decay`, `delta`) for this position.
    fn forward_step(state: &mut [f32], inputs: &PosInputs) -> TapePos {
        let mut heads = Vec::with_capacity(NV);
        for j in 0..NV {
            let sb = j * KD * VD;
            let head_state = &mut state[sb..sb + KD * VD];
            gdn_apply_decay(head_state, inputs.decay[j]);
            let kv_mem = gdn_kv_mem(head_state, KD, VD, &inputs.k[j]);
            let delta: Vec<f32> = (0..VD)
                .map(|vv| (inputs.v[j][vv] - kv_mem[vv]) * inputs.beta[j])
                .collect();
            gdn_apply_delta(head_state, KD, VD, &inputs.k[j], &delta);
            heads.push(TapeHead { k: inputs.k[j].clone(), decay: inputs.decay[j], delta });
        }
        TapePos { heads }
    }

    /// The tape-replay path (§Q1's `mtplx_..._replay_v1` analogue): uses
    /// ONLY the frozen tape quantities, in the IDENTICAL op order as the
    /// forward pass (`gdn_apply_decay` then `gdn_apply_delta`) — no `kv_mem`
    /// re-derivation.
    fn replay_step(state: &mut [f32], tape: &TapePos) {
        for j in 0..NV {
            let sb = j * KD * VD;
            let head_state = &mut state[sb..sb + KD * VD];
            let th = &tape.heads[j];
            gdn_apply_decay(head_state, th.decay);
            gdn_apply_delta(head_state, KD, VD, &th.k, &th.delta);
        }
    }

    /// Negative-control replay: deliberately WRONG op order (delta applied
    /// before decay, so the decay multiply also scales the just-added delta
    /// contribution — a real bug an implementer could introduce). Used only
    /// by the negative-control test to prove bit-comparison has teeth.
    fn replay_step_wrong_order(state: &mut [f32], tape: &TapePos) {
        for j in 0..NV {
            let sb = j * KD * VD;
            let head_state = &mut state[sb..sb + KD * VD];
            let th = &tape.heads[j];
            gdn_apply_delta(head_state, KD, VD, &th.k, &th.delta);
            gdn_apply_decay(head_state, th.decay);
        }
    }

    fn zero_state() -> Vec<f32> {
        vec![0.0f32; NV * KD * VD]
    }

    /// Bit-exact compare; on mismatch, reports the first diverging (head,
    /// element) and both raw bit patterns, then fails.
    fn assert_bits_eq(a: &[f32], b: &[f32], ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (idx, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            if av.to_bits() != bv.to_bits() {
                let head = idx / (KD * VD);
                let elem = idx % (KD * VD);
                panic!(
                    "{ctx}: first bit divergence at head {head}, elem {elem} \
                     (flat idx {idx}): a={av} (0x{:08x}) vs b={bv} (0x{:08x})",
                    av.to_bits(),
                    bv.to_bits(),
                );
            }
        }
    }

    /// Builds T positions of deterministic synthetic inputs and, from a
    /// zeroed base state, runs the real forward path once to produce the
    /// tape. Returns (base_state, per-position inputs, per-position tape).
    fn build_tape(seed: u64, t: usize) -> (Vec<f32>, Vec<PosInputs>, Vec<TapePos>) {
        let mut g = lcg(seed);
        let inputs: Vec<PosInputs> = (0..t).map(|_| gen_pos_inputs(&mut g)).collect();
        let base_state = zero_state();
        let mut ref_state = base_state.clone();
        let tape: Vec<TapePos> = inputs.iter().map(|pi| forward_step(&mut ref_state, pi)).collect();
        (base_state, inputs, tape)
    }

    #[test]
    fn tape_replay_reproduces_forward_state_bitexact() {
        const T: usize = 8;
        let (base_state, inputs, tape) = build_tape(0xDEADBEEFCAFEu64, T);

        // For EVERY keep index k in 0..=T, a tape-replay of k steps from the
        // base must be bit-identical to a completely fresh forward run of k
        // steps (recomputing kv_mem/delta from scratch, not reusing the
        // tape's delta) using the same {k, v, decay, beta} inputs.
        for k in 0..=T {
            let mut replay_state = base_state.clone();
            for tp in tape.iter().take(k) {
                replay_step(&mut replay_state, tp);
            }

            let mut fresh_state = base_state.clone();
            for pi in inputs.iter().take(k) {
                forward_step(&mut fresh_state, pi);
            }

            assert_bits_eq(&replay_state, &fresh_state, &format!("keep k={k}"));
        }
    }

    #[test]
    fn tape_replay_detects_op_order_sensitivity() {
        // Negative control: proves the bit-comparison in the test above
        // actually discriminates op order, rather than passing vacuously
        // (e.g. because delta ends up additive-only or decay is ~1.0).
        const T: usize = 8;
        let (base_state, _inputs, tape) = build_tape(0xDEADBEEFCAFEu64, T);

        let mut correct_state = base_state.clone();
        for tp in &tape {
            replay_step(&mut correct_state, tp);
        }

        let mut wrong_state = base_state.clone();
        for tp in &tape {
            replay_step_wrong_order(&mut wrong_state, tp);
        }

        let mismatch = correct_state
            .iter()
            .zip(wrong_state.iter())
            .any(|(&a, &b)| a.to_bits() != b.to_bits());
        assert!(
            mismatch,
            "reordered replay (delta before decay) must NOT match the \
             correctly-ordered replay -- if it does, the bit-comparison has \
             no discriminating power"
        );
    }
}

#[cfg(test)]
mod shader_guard {
    //! Registry guard for the Qwen3.5-specific compute kernels.
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

    /// Qwen3.5 (GatedDeltaNet + MoE) kernels this model owns.
    const REQUIRED_QWEN35_KERNELS: &[&str] = &[
        // GatedDeltaNet decode step (short conv, q/k norm, recurrence)
        "q35_dn_conv_step",
        "q35_gdn_qknorm",
        "q35_gdn_step",
        // GatedDeltaNet prefill scan
        "q35_gdn_scan",
        // MoE routed-expert weighted accumulate (plain / column-batched)
        "q35_moe_accum",
        "q35_moe_accum_batched",
    ];

    #[test]
    fn qwen35_kernels_are_registered() {
        let map = crate::include_all_shaders();
        let missing: Vec<&str> = REQUIRED_QWEN35_KERNELS
            .iter()
            .copied()
            .filter(|n| !map.contains_key(*n))
            .collect();
        assert!(
            missing.is_empty(),
            "{} Qwen3.5 shader(s) missing from the registry: {:?}\n\
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
    fn qwen35_kernel_spirv_is_wellformed() {
        let map = crate::include_all_shaders();
        for name in REQUIRED_QWEN35_KERNELS {
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
