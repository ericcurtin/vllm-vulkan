// SPDX-License-Identifier: Apache-2.0
//! GPU-resident, footprint-first weight loader for **Laguna-S-2.1-NVFP4**
//! (`model_type == "laguna"`, `poolside/Laguna-S-2.1-NVFP4`).
//!
//! This is the GPU-resident sibling of the pure-CPU reference in [`crate::laguna`]
//! (`load_laguna_weights_cpu`, HF-verified bit-exact). It ADAPTS the Nemotron
//! resident machinery ([`crate::nemotron_loader::load_nemotron_resident`] /
//! `resident_footprint`) — the streaming, never-materialize-f32,
//! quantized-GPU-resident load that fixed the 75B PP-stage OOM — to Laguna's
//! architecture, keeping every matmul weight in its native quantized form and
//! never expanding the full weight set to f32 host.
//!
//! ## Arch deltas from Nemotron (why this is a new module, not a call into
//!    `nemotron_loader`)
//!  1. **3-projection gated-SiLU experts** (`gate_proj` + `up_proj` +
//!     `down_proj`), NVFP4, on FULL hidden — not Nemotron's 2-matrix latent
//!     (`up`/`down`) MoE. So the resident expert holder is [`LagunaMoeExperts`]
//!     (three [`LagunaExpertProj`]), not `NemMoeExperts` (two).
//!  2. **On-disk tensor name is `.weight_packed`** (compressed-tensors),
//!     NOT `.weight`+U8 like Nemotron's modelopt export. Detection keys on the
//!     `.weight_packed` suffix; the sibling scale is `.weight_scale`
//!     (`F8_E4M3 [out, in/16]`) and the per-tensor global is
//!     `.weight_global_scale` (an F32 scalar).
//!  3. **`weight_global_scale` is the RECIPROCAL** of Nemotron's modelopt
//!     `weight_scale_2`: Laguna's compressed-tensors global is `1/scale`, so the
//!     value fed to `dequantize_nvfp4` / `nvfp4_fold_scales` is `1.0 / raw`
//!     (see [`crate::laguna::OwnedExpertsPacked::dequant`], the HF-verified
//!     oracle: `global = 1.0 / globals[e]`). Getting this wrong yields ~1e7
//!     weights (the f669d1e gemma reciprocal bug).
//!  4. **No FP8, no Mamba.** Everything that is BF16 on disk (attn q/k/v/o/g,
//!     dense layer-0 MLP, shared expert gate/up/down, lm_head, embed) is
//!     f16-resident (plain f16 matvec). Norms, q/k head-norms, router gate,
//!     e_score bias stay small f32 host.
//!  5. **Per-layer attention head count varies** (48 on full-attn layers, 72 on
//!     sliding), so q/o/g projection sizes are read from
//!     `num_attention_heads_per_layer[g]`.
//!
//! ## Footprint techniques designed in from the start (all default-OFF; each is
//!    a footprint-vs-speed knob, on when memory-forced)
//!  - **e4m3-resident expert scales** (`VLLM_VULKAN_NVFP4_E4M3_SCALES`, reused
//!    from the shared flag): keep the raw on-disk `.weight_scale` e4m3 bytes
//!    resident (1 byte/group) instead of folding to f32/group (4 bytes/group) —
//!    4x smaller scale buffers, the per-expert reciprocal global carried
//!    separately and re-applied in-shader. On Laguna this is LOAD-BEARING for
//!    the memory fit: experts are ~64GB e4m3 vs ~85GB folded (the whole
//!    difference between a ~8-node and a ~10-node PP split). +25% decode/step
//!    (the step-3 e4m3 finding), so it is a fit lever, not a speed lever.
//!  - **embed-f16** (`VLLM_VULKAN_LAGUNA_EMBED_F16`): store the
//!    `model.embed_tokens.weight` table f16-resident (vocab×hidden: 1.23GB f32 →
//!    0.62GB f16) on the embed-owning (first) stage; the row lookup reads the
//!    mapped f16 buffer + widens. f16 holds every in-range bf16 value exactly.
//!  - **pread-per-tensor** (`VLLM_VULKAN_LAGUNA_PREAD_LOAD`): header-parse +
//!    per-tensor `pread` instead of whole-shard `Mmap` — bounds peak RSS to one
//!    streamed tensor on nodes that cannot afford a 5GB page-cache spike per
//!    shard. (Design mirrors the Nemotron pread source; the mmap source is the
//!    default and the only one exercised in this module's on-Mac tests.)
//!
//! ## Validation status (see module tests + the bring-up plan)
//!  - `laguna_resident_footprint` is byte-exact vs hand math on a tiny config
//!    (no GPU, no checkpoint) — `footprint_is_byte_exact`.
//!  - The resident expert concat LAYOUT is bit-exact vs `dequantize_nvfp4` (the
//!    exact fn the HF-verified CPU oracle calls) on REAL checkpoint expert bytes
//!    — `resident_expert_layout_matches_oracle_dequant` (runs on the Mac with
//!    `VLLM_TEST_LAGUNA_DIR`, NO GPU: it exercises the shared offset/fold/e4m3
//!    helpers the engine loader uses).
//!  - The engine path (`load_laguna_resident`) compiles and mirrors the
//!    Nemotron resident loader; its on-NODE GTT-measurement small-window load
//!    and the GPU forward are the ≥8-node / GPU-blocked remainder (bring-up
//!    plan).

use std::collections::HashMap;
use std::path::Path;

use crate::compute::{Buffer, ComputeEngine};
use crate::laguna::LagunaConfig;
use crate::model::{discover_shards, SimpleTensor};
use crate::nemotron::{NemGpuWeight, NemQuant};
use crate::push_constants::nvfp4_fold_scales;

/// NVFP4 group size for the Laguna routed experts (`group_size: 16` from the
/// checkpoint `quantization_config`, cross-checked against real shapes:
/// `gate_proj.weight_packed [1024,1536]` with `weight_scale [1024,192]` →
/// in=3072, groups=192, 3072/192 == 16).
pub const LAGUNA_MOE_GROUP_SIZE: usize = 16;

// ─── Pure byte-layout helpers (shared by the engine loader AND the tests) ────

/// Byte offset of expert `e`'s packed-nibble block inside the per-layer
/// concatenated `[n_experts, out, in/2]` buffer.
#[inline]
pub fn expert_packed_off(e: usize, out_features: usize, in_features: usize) -> usize {
    e * out_features * (in_features / 2)
}

/// Byte offset of expert `e`'s scale block inside the per-layer concatenated
/// scale buffer. e4m3-resident scales are 1 byte/group; folded scales are 4
/// bytes/group (one f32). Numerically the same element index `e*out*groups`.
#[inline]
pub fn expert_scale_off(
    e: usize,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    e4m3: bool,
) -> usize {
    let bytes_per_group = if e4m3 { 1 } else { 4 };
    e * out_features * (in_features / group_size) * bytes_per_group
}

/// The scale bytes to store resident for one expert projection, given the raw
/// on-disk e4m3 `.weight_scale` bytes and the RECIPROCAL global
/// (`1.0 / weight_global_scale`). e4m3 path stores the raw e4m3 bytes verbatim
/// (global carried separately, re-applied in-shader); fold path folds
/// `e4m3(scale) * global` into f32/group. Both reconstruct the SAME dequant
/// (`nvfp4_fold_scales_reconstructs_dequant` / `dequantize_nvfp4`).
pub fn expert_scale_bytes(wscale_e4m3: &[u8], global_recip: f32, e4m3: bool) -> Vec<u8> {
    if e4m3 {
        wscale_e4m3.to_vec()
    } else {
        crate::push_constants::f32_slice_to_bytes(&nvfp4_fold_scales(wscale_e4m3, global_recip))
    }
}

/// True for a BF16-on-disk `nn.Linear` `.weight` that is uploaded f16-resident
/// (plain f16 matvec): attention `q/k/v/o/g_proj`, the dense (layer-0) MLP
/// `gate/up/down_proj`, the shared-expert `gate/up/down_proj`, and `lm_head`.
/// Everything else BF16 (norms, q/k head-norms, router `gate`, e_score bias,
/// embed) stays f32 host (or f16-resident embed under the embed-f16 flag).
pub fn is_bf16_resident_matmul(name: &str) -> bool {
    if name == "lm_head.weight" {
        return true;
    }
    // model.layers.{i}.self_attn.{q,k,v,o,g}_proj.weight
    if let Some(s) = name.rsplit(".self_attn.").next() {
        if s != name {
            return matches!(
                s,
                "q_proj.weight"
                    | "k_proj.weight"
                    | "v_proj.weight"
                    | "o_proj.weight"
                    | "g_proj.weight"
            );
        }
    }
    // dense layer-0 MLP: model.layers.0.mlp.{gate,up,down}_proj.weight
    // shared expert:     model.layers.{i}.mlp.shared_expert.{gate,up,down}_proj.weight
    if let Some(s) = name.rsplit(".mlp.").next() {
        if s != name {
            return matches!(
                s,
                "gate_proj.weight" | "up_proj.weight" | "down_proj.weight"
            ) || matches!(
                s,
                "shared_expert.gate_proj.weight"
                    | "shared_expert.up_proj.weight"
                    | "shared_expert.down_proj.weight"
            );
        }
    }
    false
}

/// True for the f16-resident weights the Laguna int8-attn lever
/// (`VLLM_VULKAN_LAGUNA_INT8_ATTN`) stores Q8_0-resident instead: the attention
/// `self_attn.{q,k,v,o,g}_proj` and the shared-expert
/// `shared_expert.{gate,up,down}_proj` — the ~78%-of-traffic BW slice. A STRICT
/// subset of [`is_bf16_resident_matmul`]; the dense (layer-0) MLP and `lm_head`
/// deliberately stay f16 (accuracy-critical / small share). All in-features here
/// are multiples of 32 (the q8_0 per-row block(32) requirement), guarded again
/// at the call site.
pub fn is_laguna_int8_attn_target(name: &str) -> bool {
    if let Some(s) = name.rsplit(".self_attn.").next() {
        if s != name {
            return matches!(
                s,
                "q_proj.weight"
                    | "k_proj.weight"
                    | "v_proj.weight"
                    | "o_proj.weight"
                    | "g_proj.weight"
            );
        }
    }
    is_shared_expert_proj(name)
}

/// True for the three shared-expert projection weights
/// (`...mlp.shared_expert.{gate,up,down}_proj.weight`). These are normally
/// f16-resident-only; LEVER 4's CPU shared-expert ∥ routed overlap reads them as
/// host f32, so the loader ALSO keeps an f32 host copy of these (and only these)
/// when `VLLM_VULKAN_LAGUNA_CPU_OVERLAP` is on. Also the shared-expert half of
/// the int8-attn lever's Q8_0 target set (see `is_laguna_int8_attn_target`).
fn is_shared_expert_proj(name: &str) -> bool {
    if let Some(s) = name.rsplit(".mlp.").next() {
        if s != name {
            return matches!(
                s,
                "shared_expert.gate_proj.weight"
                    | "shared_expert.up_proj.weight"
                    | "shared_expert.down_proj.weight"
            );
        }
    }
    false
}

/// Whether the loader should retain a host f32 copy of `name` for the CPU
/// shared-expert overlap lever: only the shared-expert projections, and only
/// when `VLLM_VULKAN_LAGUNA_CPU_OVERLAP` is enabled (default OFF → zero cost).
fn laguna_cpu_overlap_wants_host(name: &str) -> bool {
    is_shared_expert_proj(name)
        && std::env::var("VLLM_VULKAN_LAGUNA_CPU_OVERLAP")
            .map(|v| v == "1" || v == "true")
            .unwrap_or(false)
}

/// Parse `model.layers.{L}.mlp.experts.{e}.{gate|up|down}_proj.weight_packed`
/// into `(layer, expert, proj)` where proj is 0=gate, 1=up, 2=down; else `None`.
pub fn parse_expert_packed(name: &str) -> Option<(usize, usize, u8)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (l_str, rest) = rest.split_once('.')?;
    let layer: usize = l_str.parse().ok()?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (e_str, rest) = rest.split_once('.')?;
    let e: usize = e_str.parse().ok()?;
    let proj = match rest {
        "gate_proj.weight_packed" => 0,
        "up_proj.weight_packed" => 1,
        "down_proj.weight_packed" => 2,
        _ => return None,
    };
    Some((layer, e, proj))
}

/// GLOBAL layer index of a `model.layers.{i}.*` tensor, else `None`.
fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("model.layers.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|s| s.parse::<usize>().ok())
}

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
        other => return Err(format!("unsupported plain dtype {other:?} in Laguna loader")),
    })
}

// ─── pread source (VLLM_VULKAN_LAGUNA_PREAD_LOAD) ─────────────────────────────
// Header-parse + per-tensor `pread` instead of whole-shard `Mmap::map`. Two
// reasons this is the on-node path: (1) NFS mmap fails `ENODEV` on the BC-250
// nodes (the checkpoint lives on the read-only NFS mount, too big to stage per
// node), and (2) even where mmap works, its full-shard VMA is a load-time RSS
// transient a 14GB UMA node cannot afford next to a ~12GB resident stage. The
// reader parses ONLY each shard's safetensors header (a few MB) and preads each
// KEPT tensor's exact byte range on demand — bit-identical to
// `SafeTensors::deserialize(mmap).tensor(name).data()` (same absolute range,
// same decode math). Ported from the Nemotron pread source (perf/nemotron-embed
// -pread `parse_shard_headers`/`pread_entry`).

/// One tensor located in a shard header: dtype/shape + absolute byte range.
struct PreadEntry {
    shard: usize,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    off: u64,
    len: usize,
}

/// Parse ONLY each shard's safetensors header (8-byte little-endian length +
/// JSON) → open files + a name→`PreadEntry` index + header tensor order. No
/// tensor data is faulted. First shard defining a name wins (names are unique
/// across a checkpoint).
fn parse_shard_headers(
    shards: &[std::path::PathBuf],
) -> Result<(Vec<std::fs::File>, HashMap<String, PreadEntry>, Vec<String>), String> {
    use safetensors::Dtype;
    use std::fs::File;
    use std::os::unix::fs::FileExt;
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
    let mut files: Vec<File> = Vec::with_capacity(shards.len());
    let mut entries: HashMap<String, PreadEntry> = HashMap::new();
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
                    entries.insert(name.clone(), PreadEntry {
                        shard: si, dtype: dt, shape,
                        off: data_start + begin, len: (end - begin) as usize,
                    });
                }
            }
        }
        files.push(file);
    }
    Ok((files, entries, order))
}

/// `pread` one indexed tensor's raw bytes (no whole-shard mmap). Byte-identical
/// to `SafeTensors::deserialize(mmap).tensor(name).data()`.
fn pread_entry(files: &[std::fs::File], e: &PreadEntry) -> Result<Vec<u8>, String> {
    use std::os::unix::fs::FileExt;
    let mut buf = vec![0u8; e.len];
    files[e.shard]
        .read_exact_at(&mut buf, e.off)
        .map_err(|err| format!("pread: {err}"))?;
    Ok(buf)
}

/// The per-tensor upload/keep dispatch, shared by the whole-shard `Mmap` source
/// and the `pread` source so the two are bit-identical by construction (same
/// `view`, same sibling bytes, same decode/encode calls). `get_sibling` fetches
/// a co-tensor's RAW bytes (the NVFP4 `weight_scale` / `weight_global_scale`),
/// from the same `SafeTensors` (mmap) or via `pread` (pread source).
#[allow(clippy::too_many_arguments)]
fn handle_laguna_tensor(
    name: &str,
    view: &safetensors::tensor::TensorView,
    get_sibling: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>,
    engine: &mut ComputeEngine,
    gpu_weights: &mut HashMap<String, NemGpuWeight>,
    gpu_experts: &mut HashMap<usize, LagunaMoeExperts>,
    host: &mut HashMap<String, SimpleTensor>,
    stats: &mut LagunaResidentStats,
    e4m3: bool,
    embed_f16: bool,
    gs: usize,
) -> Result<(), String> {
    if name.ends_with(".weight_packed") {
        // NVFP4 routed expert → write packed + fold/e4m3 scale into the
        // pre-allocated per-layer concat buffer at this expert's offset.
        let (layer, e, proj) = parse_expert_packed(name)
            .ok_or_else(|| format!("NVFP4 tensor '{name}' is not a routed expert proj"))?;
        let base = &name[..name.len() - ".weight_packed".len()];
        let wscale = get_sibling(&format!("{base}.weight_scale"))?;
        let gbytes = get_sibling(&format!("{base}.weight_global_scale"))?;
        let raw_global = f32::from_le_bytes(
            gbytes[..4].try_into().map_err(|_| format!("{name}: weight_global_scale too short"))?,
        );
        // Laguna compressed-tensors global is the RECIPROCAL (see module doc).
        let global = 1.0 / raw_global;
        let out_f = view.shape()[0];
        let in_f = view.shape()[1] * 2; // 2 nibbles/byte
        let groups = (wscale.len() / out_f).max(1);
        if in_f / groups != gs {
            return Err(format!("{name}: NVFP4 group_size {} != {gs}", in_f / groups));
        }
        let ex = gpu_experts
            .get(&layer)
            .ok_or_else(|| format!("{name}: no pre-allocated experts for layer {layer}"))?;
        let projp = match proj {
            0 => &ex.gate,
            1 => &ex.up,
            _ => &ex.down,
        };
        debug_assert_eq!((out_f, in_f), (projp.out_features, projp.in_features), "{name}: expert shape");
        write_at(&projp.packed, expert_packed_off(e, out_f, in_f), view.data());
        let sbytes = expert_scale_bytes(&wscale, global, e4m3);
        write_at(&projp.scales, expert_scale_off(e, out_f, in_f, gs, e4m3), &sbytes);
        if e4m3 {
            let exm = gpu_experts.get_mut(&layer).unwrap();
            let projm = match proj {
                0 => &mut exm.gate,
                1 => &mut exm.up,
                _ => &mut exm.down,
            };
            projm.globals[e] = global;
        }
        stats.nvfp4_expert_tensors += 1;
    } else if name == "model.embed_tokens.weight" && embed_f16 {
        // f16-resident embedding table (VLLM_VULKAN_LAGUNA_EMBED_F16).
        let f32w = decode_plain(view)?;
        let out_f = view.shape()[0];
        let in_f = *view.shape().get(1).unwrap_or(&1);
        let buf = encode_f16_resident(engine, &f32w)?;
        stats.gpu_resident_bytes += (f32w.len() * 2) as u64;
        stats.f16_tensors += 1;
        gpu_weights.insert(
            name.to_string(),
            NemGpuWeight { buffer: buf, quant: NemQuant::F16, out_features: out_f, in_features: in_f },
        );
    } else if name.ends_with(".weight")
        && crate::flags::flags_global().laguna_int8_attn
        && is_laguna_int8_attn_target(name)
        && (*view.shape().get(1).unwrap_or(&1)) % 32 == 0
    {
        // int8-attn lever (VLLM_VULKAN_LAGUNA_INT8_ATTN): BF16 → Q8_0 resident
        // (symmetric int8, per-32-block f16 scale in-block — GGUF q8_0), ~half the
        // bytes of the f16 form. Quantized per-row (each row is whole 32-blocks
        // since in_features % 32 == 0, guarded above), so the rayon output is
        // byte-identical to serial. Routed through `mul_mat_vec_q8_0deq_f32_f32`
        // at dispatch (see laguna_gpu::mv_meta / rec_mv). The int8 sibling of the
        // E4M3 f8-attn lever (banked NO-GO); int8's cheap dequant (v_cvt + block
        // scale, no LUT) is the perf bet, its uniform levels the accuracy bet.
        let f32w = decode_plain(view)?;
        let out_f = view.shape()[0];
        let in_f = *view.shape().get(1).unwrap_or(&1);
        use rayon::prelude::*;
        let q8: Vec<u8> = f32w
            .par_chunks(in_f)
            .flat_map_iter(|row| crate::model::quantize_q8_0(row))
            .collect();
        let buf = encode_bytes_resident(engine, &q8)?;
        stats.gpu_resident_bytes += q8.len() as u64;
        stats.int8_tensors += 1;
        gpu_weights.insert(
            name.to_string(),
            NemGpuWeight {
                buffer: buf,
                quant: NemQuant::Q8_0,
                out_features: out_f,
                in_features: in_f,
            },
        );
    } else if name.ends_with(".weight") && is_bf16_resident_matmul(name) {
        // BF16 → f16 resident (plain matvec).
        let f32w = decode_plain(view)?;
        let out_f = view.shape()[0];
        let in_f = *view.shape().get(1).unwrap_or(&1);
        let nbytes = f32w.len() * 2;
        // LEVER 4 (VLLM_VULKAN_LAGUNA_CPU_OVERLAP): the CPU shared-expert ∥ routed
        // overlap runs the shared expert (gate/up/down_proj) on the host rayon
        // pool, which reads its weights as host f32 via `self.w()`. Those are
        // normally f16-resident-only (dropped from host for footprint), so ALSO
        // keep an f32 host copy of JUST the three small shared-expert projections
        // when the flag is on (~tens of MB total, gated so it's zero-cost off).
        if laguna_cpu_overlap_wants_host(name) {
            host.insert(name.to_string(), SimpleTensor { data: f32w.clone(), shape: vec![] });
            stats.host_bytes += (f32w.len() * 4) as u64;
            stats.host_tensors += 1;
        }
        let buf = encode_f16_resident(engine, &f32w)?;
        stats.gpu_resident_bytes += nbytes as u64;
        stats.f16_tensors += 1;
        gpu_weights.insert(
            name.to_string(),
            NemGpuWeight { buffer: buf, quant: NemQuant::F16, out_features: out_f, in_features: in_f },
        );
    } else {
        // Small host f32: norms, q/k head-norms, router gate, e_score bias,
        // final norm, and (no embed-f16) the embed table.
        let data = decode_plain(view)?;
        stats.host_bytes += (data.len() * 4) as u64;
        stats.host_tensors += 1;
        host.insert(name.to_string(), SimpleTensor { data, shape: vec![] });
    }
    Ok(())
}

/// True if this tensor is a routed NVFP4 auxiliary scale (consumed as a sibling
/// of its `.weight_packed`, never handled on its own).
fn is_laguna_aux_scale(name: &str) -> bool {
    name.ends_with(".weight_scale")
        || name.ends_with(".weight_global_scale")
        || name.ends_with(".weight_scale_2")
        || name.ends_with(".input_global_scale")
        || name.ends_with(".input_scale")
        || name.ends_with(".k_scale")
        || name.ends_with(".v_scale")
}

// ─── Footprint projection (pure; the PP-depth / GTT-fit model) ───────────────

/// Projected GPU-resident + host footprint for a Laguna layer range
/// `[start, end)`, split by dequant path. `gpu_resident_bytes()` is the quantity
/// gated against the ~13.3GB BC-250 GTT budget per PP stage.
#[derive(Debug, Default, Clone, Copy)]
pub struct LagunaResidentFootprint {
    /// NVFP4 routed experts: packed nibbles (out*in/2) + scales
    /// (out*(in/gs)*{1 e4m3|4 fold}), summed over gate+up+down × n_experts.
    pub nvfp4_expert_bytes: u64,
    /// f16-resident BF16-native matmuls: attn q/k/v/o/g, dense MLP, shared
    /// expert, lm_head, and (embed-f16 flag) the embedding table.
    pub f16_bytes: u64,
    /// f32 host tensors: input/post layernorm, q/k head-norms, router gate,
    /// e_score bias, final norm, and (no embed-f16) the embedding table.
    pub host_f32_bytes: u64,
}

impl LagunaResidentFootprint {
    pub fn gpu_resident_bytes(&self) -> u64 {
        self.nvfp4_expert_bytes + self.f16_bytes
    }
    pub fn total_bytes(&self) -> u64 {
        self.gpu_resident_bytes() + self.host_f32_bytes
    }
}

/// Project the resident footprint for `[start, end)` from config dims alone (no
/// checkpoint read) — the sizing model behind the PP-depth decision. Mirrors
/// exactly what [`load_laguna_resident`] uploads/keeps per weight.
pub fn laguna_resident_footprint(
    cfg: &LagunaConfig,
    start: usize,
    end: usize,
    keep_embed: bool,
    keep_lm: bool,
    e4m3: bool,
    embed_f16: bool,
) -> LagunaResidentFootprint {
    let gs = LAGUNA_MOE_GROUP_SIZE as u64;
    let sbpg: u64 = if e4m3 { 1 } else { 4 };
    let hidden = cfg.hidden_size as u64;
    let hd = cfg.head_dim as u64;
    let nkv = cfg.num_key_value_heads as u64;
    let vocab = cfg.vocab_size as u64;
    let ne = cfg.num_experts as u64;
    let moe_inter = cfg.moe_intermediate_size as u64;
    let shared_inter = cfg.shared_expert_intermediate_size as u64;

    let nvfp4 = |out: u64, in_: u64| out * (in_ / 2) + out * (in_ / gs) * sbpg;
    let f16 = |out: u64, in_: u64| out * in_ * 2;

    let mut fp = LagunaResidentFootprint::default();
    for g in start..end {
        // Attention (per-layer head count: 48 full / 72 sliding).
        let nq = *cfg.num_attention_heads_per_layer.get(g).unwrap_or(&cfg.num_attention_heads) as u64;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        fp.f16_bytes += f16(q_dim, hidden)       // q_proj
            + 2 * f16(kv_dim, hidden)            // k_proj, v_proj
            + f16(hidden, q_dim)                 // o_proj
            + f16(nq, hidden);                   // g_proj (per-head softplus gate)
        // q_norm/k_norm [head_dim] + input/post layernorm [hidden] → f32 host.
        fp.host_f32_bytes += (2 * hd + 2 * hidden) * 4;

        // MLP.
        if cfg.mlp_only_layers.contains(&g) {
            let inter = cfg.intermediate_size as u64;
            fp.f16_bytes += 2 * f16(inter, hidden) + f16(hidden, inter);
        } else {
            // Routed experts: gate/up [moe_inter, hidden], down [hidden, moe_inter], NVFP4.
            fp.nvfp4_expert_bytes +=
                ne * (2 * nvfp4(moe_inter, hidden) + nvfp4(hidden, moe_inter));
            // Shared expert: gate/up [shared_inter, hidden], down [hidden, shared_inter], f16.
            fp.f16_bytes += 2 * f16(shared_inter, hidden) + f16(hidden, shared_inter);
            // router gate [ne, hidden] + e_score bias [ne] → f32 host.
            fp.host_f32_bytes += (ne * hidden + ne) * 4;
        }
    }
    if keep_embed {
        if embed_f16 {
            fp.f16_bytes += vocab * hidden * 2;
        } else {
            fp.host_f32_bytes += vocab * hidden * 4;
        }
    }
    if keep_lm {
        fp.f16_bytes += f16(vocab, hidden);
        fp.host_f32_bytes += hidden * 4; // final norm
    }
    fp
}

// ─── Resident GPU structures ─────────────────────────────────────────────────

/// One NVFP4 routed-expert projection (gate/up OR down) for a whole MoE layer,
/// resident as two concatenated GPU buffers sliced per expert at dispatch (the
/// qwen35_moe / NemMoeExperts pattern). `globals` holds each expert's RECIPROCAL
/// `weight_global_scale` (only consulted on the e4m3 path; `vec![1.0; ne]` on
/// the fold path).
pub struct LagunaExpertProj {
    pub packed: Buffer,
    pub scales: Buffer,
    /// out_features (gate/up = moe_intermediate_size; down = hidden_size).
    pub out_features: usize,
    /// in_features (gate/up = hidden_size; down = moe_intermediate_size).
    pub in_features: usize,
    pub globals: Vec<f32>,
}

/// GPU-resident routed experts for one Laguna MoE layer: gated-SiLU 3-matrix
/// (`gate`, `up`, `down`), NVFP4. The 3-projection analogue of `NemMoeExperts`
/// (which is 2-matrix latent). `top_k` (10) of `n_experts` (256) dispatched per
/// token; the ungated shared expert runs from `gpu_weights` (f16).
pub struct LagunaMoeExperts {
    pub n_experts: usize,
    pub gate: LagunaExpertProj,
    pub up: LagunaExpertProj,
    pub down: LagunaExpertProj,
    pub group_size: u32,
    pub e4m3: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LagunaResidentStats {
    pub nvfp4_expert_tensors: usize,
    pub f16_tensors: usize,
    /// Q8_0-resident attn+shared weights under the int8-attn lever
    /// (`VLLM_VULKAN_LAGUNA_INT8_ATTN`). Zero when the flag is off.
    pub int8_tensors: usize,
    pub host_tensors: usize,
    pub gpu_resident_bytes: u64,
    pub host_bytes: u64,
}

fn write_at(buf: &Buffer, byte_off: usize, bytes: &[u8]) {
    let ptr = buf.mapped_ptr.expect("host-coherent buffer is mapped") as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(byte_off), bytes.len());
    }
}

fn alloc_proj(
    engine: &mut ComputeEngine,
    ne: usize,
    out_f: usize,
    in_f: usize,
    e4m3: bool,
) -> Result<LagunaExpertProj, String> {
    let packed = engine.alloc_host_coherent_storage((ne * out_f * (in_f / 2)) as u64)?;
    let sbpg = if e4m3 { 1 } else { 4 };
    let scales =
        engine.alloc_host_coherent_storage((ne * out_f * (in_f / LAGUNA_MOE_GROUP_SIZE) * sbpg) as u64)?;
    Ok(LagunaExpertProj {
        packed,
        scales,
        out_features: out_f,
        in_features: in_f,
        globals: vec![1.0f32; ne],
    })
}

/// GPU-RESIDENT streaming loader for Laguna. Keeps every matmul weight in its
/// native quantized form uploaded GPU-resident (NVFP4 routed experts concatenated
/// per MoE layer for per-expert dispatch; f16 for the BF16-native attn/dense/
/// shared/lm_head; f32 host only for norms/router/bias + the first stage's embed)
/// and never materializes the f32-expanded weight set. Fills `gpu_weights` /
/// `gpu_experts` / `host` in place for the PP window `[layer_start, layer_end)`.
///
/// PP-only (whole-model-per-stage experts). EP/TP expert sharding is a documented
/// future extension (the Nemotron `expert_owned_range` pattern drops straight in).
#[allow(clippy::too_many_arguments)]
pub fn load_laguna_resident(
    path: &Path,
    cfg: &LagunaConfig,
    engine: &mut ComputeEngine,
    gpu_weights: &mut HashMap<String, NemGpuWeight>,
    gpu_experts: &mut HashMap<usize, LagunaMoeExperts>,
    host: &mut HashMap<String, SimpleTensor>,
    layer_start: usize,
    layer_end: usize,
    keep_embed: bool,
    keep_lm: bool,
) -> Result<LagunaResidentStats, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::fs::File;

    let mut stats = LagunaResidentStats::default();
    let gs = LAGUNA_MOE_GROUP_SIZE;
    let ne = cfg.num_experts;
    let hidden = cfg.hidden_size;
    let moe_inter = cfg.moe_intermediate_size;

    let e4m3 = crate::flags::flags_global().nvfp4_e4m3_scales;
    let embed_f16 = crate::flags::flags_global().laguna_embed_f16;
    // pread-per-tensor source (VLLM_VULKAN_LAGUNA_PREAD_LOAD) is the on-node
    // peak-RSS lever; the header-parse + `pread` reader is a documented follow-on
    // (mirrors the Nemotron pread source). Until it lands, honor the request by
    // logging it and falling through to the whole-shard `Mmap` source below.
    let pread = crate::flags::flags_global().laguna_pread_load;
    let progress = std::env::var("VLLM_VULKAN_LAGUNA_LOAD_PROGRESS").ok().as_deref() == Some("1");
    macro_rules! plog {
        ($($a:tt)*) => {
            if progress {
                eprintln!("[laguna-load pp={layer_start}..{layer_end}] {}", format!($($a)*));
            }
        };
    }
    plog!("START keep_embed={keep_embed} keep_lm={keep_lm} ne={ne} e4m3={e4m3} embed_f16={embed_f16}");
    plog!("source = {}", if pread { "pread-per-tensor" } else { "whole-shard Mmap" });

    // Pre-allocate the concatenated per-MoE-layer expert buffers so each expert
    // tensor can be written at its offset during streaming.
    for g in layer_start..layer_end {
        if cfg.mlp_only_layers.contains(&g) {
            continue; // dense layer, no routed experts
        }
        let gate = alloc_proj(engine, ne, moe_inter, hidden, e4m3)?;
        let up = alloc_proj(engine, ne, moe_inter, hidden, e4m3)?;
        let down = alloc_proj(engine, ne, hidden, moe_inter, e4m3)?;
        let bytes = [&gate, &up, &down]
            .iter()
            .map(|p| {
                let sbpg = if e4m3 { 1 } else { 4 };
                (ne * p.out_features * (p.in_features / 2)
                    + ne * p.out_features * (p.in_features / gs) * sbpg) as u64
            })
            .sum::<u64>();
        stats.gpu_resident_bytes += bytes;
        plog!("prealloc MoE L{g}: +{} MB experts (cum GTT {} MB)", bytes >> 20, stats.gpu_resident_bytes >> 20);
        gpu_experts.insert(
            g,
            LagunaMoeExperts { n_experts: ne, gate, up, down, group_size: gs as u32, e4m3 },
        );
    }

    let shards = discover_shards(path);
    let keep = |name: &str| -> bool {
        if name == "model.embed_tokens.weight" {
            return keep_embed;
        }
        if name == "lm_head.weight" || name == "model.norm.weight" {
            return keep_lm;
        }
        match layer_of(name) {
            Some(idx) => idx >= layer_start && idx < layer_end,
            None => false,
        }
    };

    if pread {
        // pread source: parse headers once, then pread each KEPT tensor's exact
        // byte range on demand (siblings via the global index). No whole-shard
        // VMA — the NFS-mmap-ENODEV / load-RSS fix. Bit-identical to the mmap
        // arm (same `handle_laguna_tensor`, same byte ranges).
        let (files, entries, order) = parse_shard_headers(&shards)?;
        for name in &order {
            if !keep(name) || is_laguna_aux_scale(name) {
                continue;
            }
            let entry = entries.get(name).unwrap(); // name came from `order`
            let bytes = pread_entry(&files, entry)?;
            let view = safetensors::tensor::TensorView::new(entry.dtype, entry.shape.clone(), &bytes)
                .map_err(|e| format!("{name}: TensorView::new: {e}"))?;
            let mut get_sibling = |sib: &str| -> Result<Vec<u8>, String> {
                let se = entries.get(sib).ok_or_else(|| format!("{name}: sibling {sib} not in index"))?;
                pread_entry(&files, se)
            };
            handle_laguna_tensor(
                name, &view, &mut get_sibling, engine, gpu_weights, gpu_experts, host,
                &mut stats, e4m3, embed_f16, gs,
            )?;
        }
    } else {
        for shard in &shards {
            let file = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
            let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;
            for (name, view) in st.tensors() {
                if !keep(&name) || is_laguna_aux_scale(&name) {
                    continue;
                }
                let mut get_sibling = |sib: &str| -> Result<Vec<u8>, String> {
                    st.tensor(sib).map(|t| t.data().to_vec())
                        .map_err(|e| format!("{name}: sibling {sib}: {e}"))
                };
                handle_laguna_tensor(
                    &name, &view, &mut get_sibling, engine, gpu_weights, gpu_experts, host,
                    &mut stats, e4m3, embed_f16, gs,
                )?;
            }
        }
    }
    plog!(
        "DONE resident GTT {} MB ({} nvfp4-expert + {} f16 + {} int8/q8), host {} MB ({} tensors)",
        stats.gpu_resident_bytes >> 20, stats.nvfp4_expert_tensors, stats.f16_tensors,
        stats.int8_tensors, stats.host_bytes >> 20, stats.host_tensors
    );
    Ok(stats)
}

/// Encode an f32 weight DIRECTLY into a resident host-coherent GPU buffer's
/// mapped memory as f16 (no third large transient), then drop the f32 staging —
/// the Nemotron f16-resident encode. f16 holds every in-range bf16 value exactly.
/// Upload raw pre-quantized bytes (e.g. q8_0 blocks) to a host-coherent GPU
/// buffer, no reinterpretation. Used by the int8-attn lever's q8_0 weights.
fn encode_bytes_resident(engine: &mut ComputeEngine, bytes: &[u8]) -> Result<Buffer, String> {
    let buf = engine.alloc_host_coherent_storage(bytes.len() as u64)?;
    buf.write(bytes)?;
    Ok(buf)
}

fn encode_f16_resident(engine: &mut ComputeEngine, f32w: &[f32]) -> Result<Buffer, String> {
    let buf = engine.alloc_host_coherent_storage((f32w.len() * 2) as u64)?;
    let dst = buf.mapped_ptr.expect("host-coherent buffer is mapped") as *mut u8;
    for (i, &v) in f32w.iter().enumerate() {
        let b = half::f16::from_f32(v).to_bits().to_le_bytes();
        unsafe {
            *dst.add(i * 2) = b[0];
            *dst.add(i * 2 + 1) = b[1];
        }
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{dequantize_nvfp4, NVFP4_E2M1_LUT};

    fn tiny_cfg() -> LagunaConfig {
        // Minimal config for the footprint math (dims only; no checkpoint).
        let json = serde_json::json!({
            "model_type": "laguna",
            "hidden_size": 3072,
            "num_hidden_layers": 48,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 100352,
            "intermediate_size": 12288,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "num_experts": 256,
            "num_experts_per_tok": 10,
            "norm_topk_prob": true,
            "moe_routed_scaling_factor": 2.5,
            "sliding_window": 512,
            "max_position_embeddings": 262144,
            "mlp_only_layers": [0],
            "layer_types": (0..48).map(|i| if i % 4 == 0 { "full_attention" } else { "sliding_attention" }).collect::<Vec<_>>(),
            "num_attention_heads_per_layer": (0..48).map(|i| if i % 4 == 0 { 48 } else { 72 }).collect::<Vec<_>>(),
            "rope_parameters": {
                "full_attention": {"rope_theta": 500000.0, "rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192, "beta_slow": 1.0, "beta_fast": 32.0, "attention_factor": 1.3465735902799727, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}
            }
        });
        LagunaConfig::from_json(&json).expect("tiny cfg")
    }

    /// The footprint formula is byte-exact vs hand-rolled per-tensor sizing on a
    /// single MoE layer + the embed/lm edges — and the resident set is a small
    /// fraction of the f32-host blow-up that would OOM a PP stage.
    #[test]
    fn footprint_is_byte_exact() {
        let cfg = tiny_cfg();
        // One sliding MoE layer (layer 1: 72 heads), fold scales, no edges.
        let fp = laguna_resident_footprint(&cfg, 1, 2, false, false, false, false);
        let gs = LAGUNA_MOE_GROUP_SIZE as u64;
        let (h, hd, nkv, ne, mi, si) = (3072u64, 128u64, 8u64, 256u64, 1024u64, 1024u64);
        let nq = 72u64;
        let nvfp4 = |o: u64, i: u64| o * (i / 2) + o * (i / gs) * 4;
        let f16 = |o: u64, i: u64| o * i * 2;
        let exp_nvfp4 = ne * (2 * nvfp4(mi, h) + nvfp4(h, mi));
        let exp_f16 = f16(nq * hd, h) + 2 * f16(nkv * hd, h) + f16(h, nq * hd) + f16(nq, h)
            + 2 * f16(si, h) + f16(h, si);
        let exp_host = (2 * hd + 2 * h) * 4 + (ne * h + ne) * 4;
        assert_eq!(fp.nvfp4_expert_bytes, exp_nvfp4, "nvfp4 expert bytes");
        assert_eq!(fp.f16_bytes, exp_f16, "f16 bytes");
        assert_eq!(fp.host_f32_bytes, exp_host, "host f32 bytes");

        // e4m3 scales quarter the scale term (1 vs 4 bytes/group).
        let fp_e4m3 = laguna_resident_footprint(&cfg, 1, 2, false, false, true, false);
        let scale_fold = ne * (2 * (mi * (h / gs) * 4) + h * (mi / gs) * 4);
        let scale_e4m3 = ne * (2 * (mi * (h / gs)) + h * (mi / gs));
        assert_eq!(fp.nvfp4_expert_bytes - fp_e4m3.nvfp4_expert_bytes, scale_fold - scale_e4m3);

        // embed-f16 halves the embed contribution and moves it resident.
        let fp_ef = laguna_resident_footprint(&cfg, 0, 0, true, false, false, true);
        let fp_e0 = laguna_resident_footprint(&cfg, 0, 0, true, false, false, false);
        assert_eq!(fp_ef.f16_bytes, cfg.vocab_size as u64 * h * 2);
        assert_eq!(fp_e0.host_f32_bytes, cfg.vocab_size as u64 * h * 4);
    }

    /// Full-model PP-stage sizing sanity: report the total resident set and the
    /// per-stage footprint at PP-8 (6 layers/stage) so the GTT-fit is visible.
    #[test]
    fn footprint_pp_stage_report() {
        let cfg = tiny_cfg();
        let total_fold = laguna_resident_footprint(&cfg, 0, 48, true, true, false, false);
        let total_e4m3 = laguna_resident_footprint(&cfg, 0, 48, true, true, true, false);
        eprintln!(
            "LAGUNA total resident: fold {:.1} GB (experts {:.1}), e4m3 {:.1} GB (experts {:.1})",
            total_fold.gpu_resident_bytes() as f64 / 1e9,
            total_fold.nvfp4_expert_bytes as f64 / 1e9,
            total_e4m3.gpu_resident_bytes() as f64 / 1e9,
            total_e4m3.nvfp4_expert_bytes as f64 / 1e9,
        );
        // Heaviest PP-8 stage (6 contiguous MoE layers, no vocab edge): stages
        // 1..7 carry 6 MoE layers each; report the max resident.
        for (label, e4m3) in [("fold", false), ("e4m3", true)] {
            let mut max_gb = 0.0f64;
            for s in 0..8 {
                let (lo, hi) = (s * 6, (s + 1) * 6);
                let fp = laguna_resident_footprint(&cfg, lo, hi, s == 0, s == 7, e4m3, false);
                let gb = fp.gpu_resident_bytes() as f64 / 1e9;
                if gb > max_gb { max_gb = gb; }
            }
            eprintln!("LAGUNA PP-8 heaviest stage ({label}): {max_gb:.2} GB resident");
        }
        assert!(total_e4m3.gpu_resident_bytes() < total_fold.gpu_resident_bytes());
    }

    /// REAL-checkpoint, GPU-FREE bit-exactness of the resident expert LAYOUT.
    /// Reads one fully-present MoE layer's experts from a shard, builds the
    /// per-layer concat buffers with the SAME offset/fold/e4m3 helpers the engine
    /// loader uses, and asserts that dequantizing an expert's gate/up/down slice
    /// FROM the concat buffer is byte-identical to `dequantize_nvfp4` (the exact
    /// fn the HF-verified CPU oracle `OwnedExpertsPacked::dequant` calls) applied
    /// to that expert's on-disk bytes with the reciprocal global. Also checks
    /// that e4m3-resident scales reconstruct the same weights as folded scales,
    /// and that the weights are in a sane range (reciprocal global applied).
    ///
    ///   VLLM_TEST_LAGUNA_SHARD=/path/model-00004-of-00015.safetensors \
    ///   VLLM_TEST_LAGUNA_LAYER=7 cargo test -p ... resident_expert_layout -- --nocapture
    #[test]
    fn resident_expert_layout_matches_oracle_dequant() {
        use memmap2::Mmap;
        use safetensors::SafeTensors;
        let shard = match std::env::var("VLLM_TEST_LAGUNA_SHARD") {
            Ok(s) => s,
            Err(_) => {
                eprintln!("VLLM_TEST_LAGUNA_SHARD unset — skipping resident layout bit-exact test");
                return;
            }
        };
        let layer: usize = std::env::var("VLLM_TEST_LAGUNA_LAYER").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(7);
        let file = std::fs::File::open(&shard).expect("open shard");
        let mmap = unsafe { Mmap::map(&file) }.expect("mmap shard");
        let st = SafeTensors::deserialize(&mmap).expect("parse shard");

        let gs = LAGUNA_MOE_GROUP_SIZE;
        // Discover expert count present for this layer.
        let mut ne = 0usize;
        while st.tensor(&format!("model.layers.{layer}.mlp.experts.{ne}.gate_proj.weight_packed")).is_ok() {
            ne += 1;
        }
        assert!(ne > 0, "no experts for layer {layer} in {shard}");
        eprintln!("layer {layer}: {ne} experts present");

        // Build the gate concat (fold path) + a parallel e4m3-scale concat.
        let (out_f, in_f) = {
            let v = st.tensor(&format!("model.layers.{layer}.mlp.experts.0.gate_proj.weight_packed")).unwrap();
            (v.shape()[0], v.shape()[1] * 2)
        };
        assert_eq!(in_f / (st.tensor(&format!("model.layers.{layer}.mlp.experts.0.gate_proj.weight_scale")).unwrap().data().len() / out_f), gs, "group_size");

        let mut packed_concat = vec![0u8; ne * out_f * (in_f / 2)];
        let mut fold_concat = vec![0u8; ne * out_f * (in_f / gs) * 4];
        let mut e4m3_concat = vec![0u8; ne * out_f * (in_f / gs)];
        let mut globals = vec![0f32; ne];
        for e in 0..ne {
            let b = format!("model.layers.{layer}.mlp.experts.{e}.gate_proj");
            let packed = st.tensor(&format!("{b}.weight_packed")).unwrap();
            let wscale = st.tensor(&format!("{b}.weight_scale")).unwrap();
            let raw_global = f32::from_le_bytes(
                st.tensor(&format!("{b}.weight_global_scale")).unwrap().data()[..4].try_into().unwrap());
            let global = 1.0 / raw_global;
            globals[e] = global;
            let po = expert_packed_off(e, out_f, in_f);
            packed_concat[po..po + packed.data().len()].copy_from_slice(packed.data());
            let fo = expert_scale_off(e, out_f, in_f, gs, false);
            let fb = expert_scale_bytes(wscale.data(), global, false);
            fold_concat[fo..fo + fb.len()].copy_from_slice(&fb);
            let eo = expert_scale_off(e, out_f, in_f, gs, true);
            e4m3_concat[eo..eo + wscale.data().len()].copy_from_slice(wscale.data());
        }

        // Validate a spread of experts.
        let mut max_abs = 0f32;
        for &e in &[0usize, 1, ne / 2, ne - 1] {
            let b = format!("model.layers.{layer}.mlp.experts.{e}.gate_proj");
            let disk_packed = st.tensor(&format!("{b}.weight_packed")).unwrap();
            let disk_wscale = st.tensor(&format!("{b}.weight_scale")).unwrap();
            // ORACLE: dequant directly from on-disk bytes with reciprocal global.
            let oracle = dequantize_nvfp4(disk_packed.data(), disk_wscale.data(), globals[e], out_f, in_f, gs);

            // RESIDENT (fold): dequant from the concat slice. The GPU shader reads
            // folded f32 and computes LUT*folded; reconstruct that here.
            let po = expert_packed_off(e, out_f, in_f);
            let packed_slice = &packed_concat[po..po + out_f * (in_f / 2)];
            let fo = expert_scale_off(e, out_f, in_f, gs, false);
            let fold_slice = &fold_concat[fo..fo + out_f * (in_f / gs) * 4];
            let fold_f32: Vec<f32> = fold_slice.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let groups = in_f / gs;
            let mut max_rel = 0f32;
            for o in 0..out_f {
                for i in 0..in_f {
                    let byte = packed_slice[o * (in_f / 2) + i / 2];
                    let nib = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let recon = NVFP4_E2M1_LUT[nib as usize] * fold_f32[o * groups + i / gs];
                    let ov = oracle[o * in_f + i];
                    // The GPU fold shader computes LUT*(e4m3*global); the CPU oracle
                    // computes (LUT*e4m3)*global. f32 reassociation differs by ~1 ULP,
                    // so the fold path is ULP-close (NOT bit-exact) to the oracle —
                    // same tolerance as `nvfp4_fold_scales_reconstructs_dequant`.
                    let tol = 1e-6 * ov.abs().max(1e-6);
                    assert!((recon - ov).abs() <= tol,
                        "expert {e} elem [{o},{i}]: fold {recon} vs oracle {ov}");
                    if ov.abs() > 1e-6 { max_rel = max_rel.max((recon - ov).abs() / ov.abs()); }
                }
            }
            eprintln!("expert {e}: fold-path max relative dev from oracle = {max_rel:.2e}");

            // RESIDENT (e4m3): raw e4m3 scale + reciprocal global reapplied.
            let eo = expert_scale_off(e, out_f, in_f, gs, true);
            let e4m3_slice = &e4m3_concat[eo..eo + out_f * (in_f / gs)];
            let e4m3_dq = dequantize_nvfp4(packed_slice, e4m3_slice, globals[e], out_f, in_f, gs);
            assert_eq!(e4m3_dq, oracle, "expert {e}: e4m3 resident != oracle dequant");

            for &v in &oracle { max_abs = max_abs.max(v.abs()); }
        }
        eprintln!("layer {layer} gate max|W| = {max_abs:.4} (sane O(0.1-10); ~1e7 => reciprocal dropped)");
        assert!(max_abs < 100.0, "gate weights insane ({max_abs}) — reciprocal global likely wrong");
        assert!(max_abs > 1e-3, "gate weights all ~0 — dequant broken");

        // down_proj has the OTHER shape ([hidden, moe_inter] = [3072,1024]); check
        // the shape-dependent offsets + e4m3 bit-exactness for one expert.
        {
            let b = format!("model.layers.{layer}.mlp.experts.0.down_proj");
            let dp = st.tensor(&format!("{b}.weight_packed")).unwrap();
            let ds = st.tensor(&format!("{b}.weight_scale")).unwrap();
            let (d_out, d_in) = (dp.shape()[0], dp.shape()[1] * 2);
            let dg = 1.0 / f32::from_le_bytes(
                st.tensor(&format!("{b}.weight_global_scale")).unwrap().data()[..4].try_into().unwrap());
            // Concat expert 0 at its offset in a 2-expert buffer, read it back.
            let mut dpc = vec![0u8; 2 * d_out * (d_in / 2)];
            let po = expert_packed_off(1, d_out, d_in); // put at index 1 to test non-zero offset
            dpc[po..po + dp.data().len()].copy_from_slice(dp.data());
            let slice = &dpc[po..po + d_out * (d_in / 2)];
            let oracle = dequantize_nvfp4(dp.data(), ds.data(), dg, d_out, d_in, gs);
            let resident = dequantize_nvfp4(slice, ds.data(), dg, d_out, d_in, gs);
            assert_eq!(resident, oracle, "down_proj: concat-slice offset produced wrong bytes");
            eprintln!("down_proj [{d_out},{d_in}] concat-offset e4m3 dequant == oracle: OK");
        }
    }
}
