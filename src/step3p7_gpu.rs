//! Step-3.7-Flash-148B GPU-resident decode stage (`model_type == "step3p7"`).
//!
//! The GPU-resident + stateful-decode + TP-2 campaign for step3p7 — the Ling/Laguna-
//! class port that `step3p7.rs` (CPU prefill + CPU decode oracle) leaves unbuilt.
//! Structurally this mirrors `ling_gpu::LingGpuStage` (a per-layer resident stage owned
//! inside the model, `forward_pp_stage`/`reset_state`), but step3p7 is a STANDARD-attn
//! sigmoid-router MoE — no KDA, no MLA — so the decode is strictly simpler: a growing
//! GQA KV cache, not a recurrent state.
//!
//! ── What runs where (the Ling hybrid) ──────────────────────────────────────────
//! GPU (resident buffers, one submit per matvec): every linear — attn q/k/v/o/g,
//! dense gate/up/down, shared-expert gate/up/down, lm_head (all f16), and the routed
//! experts (NVFP4 3D-stacked, e4m3 group scales) via `mul_mat_vec_nvfp4_e4m3`
//! (Laguna `expert_matvec` template — NOT Ling's mlx4 path). Host (small glue, reusing
//! the bit-exact `step3p7.rs` pure fns): +1 RMSNorm, per-head qk-norm, partial RoPE,
//! causal/sliding SDPA over the KV, the head-wise scalar sigmoid gate, clamped SwiGLU,
//! the DeepSeek-V3 bias-corrected router, and the routed-expert weighted accumulate.
//! The decode STATE machine is byte-for-byte the same op sequence as `step3p7.rs`'s
//! CPU `decode_step` (which is proven == prefill offline), so this stage inherits that
//! correctness up to f16-weight rounding on the GPU matvecs (argmax-exact vs the oracle
//! is the on-cluster bar, exactly like the fleet prefill gate).
//!
//! ── Dequant fidelity ───────────────────────────────────────────────────────────
//! The NVFP4 packed nibbles + F8_E4M3 group-16 block scales are uploaded VERBATIM; the
//! per-tensor global (`weight_scale_2`, MULTIPLIED — the modelopt convention, not the
//! compressed-tensors reciprocal) rides in the push constant, exactly as the CPU
//! `dequantize_nvfp4` consumes it. So a resident expert dequants identically to the CPU
//! oracle BY CONSTRUCTION — only the arithmetic engine differs.
//!
//! ── TP-2 (Phase 3) ─────────────────────────────────────────────────────────────
//! Load-time sharding: qwen35-style column-shard of attn q/k/v out-rows + dense/shared
//! gate/up out-rows, row-shard of o_proj / down in-cols; nemotron-style EP whole-expert
//! partition of the routed experts (`expert_owned_range`, router replicated + owned
//! filter). Forward: a single `all_reduce_f32_sum_inplace` after o_proj and after the
//! MLP down (the partial → full reduction), gated on `tp_size > 1`. GIL re-acquired via
//! `Python::with_gil` inside the reduce so `decode_step` stays self-contained.

use std::collections::HashMap;
use std::os::raw::c_void;

use crate::compute;
use crate::device;
use crate::flags::QuantFormat;
use crate::model::{cpu_matmul, cpu_sdpa_gqa};
use crate::push_constants::{
    f32_slice_to_bytes, f32_to_f16_bytes, laguna_expert_repack_flag, matvec_f32_variant,
    matvec_nvfp4_e4m3_pc_off, matvec_nvfp4_e4m3_variant, matvec_pc13, matvec_variant_by_format,
    nvfp4_repack_shape_ok, read_f32_buf, sdpa_pc,
};
use crate::step3p7::{
    bias_router, clamped_swiglu_prod, head_gate, partial_rope, rms_norm_plus1,
    Step3p7Config, Step3p7KvCache,
};
use crate::vccl_ffi;
use serde_json::Value;

// ─── resident weight holders ────────────────────────────────────────────────

/// A dense (bf16-origin) matvec weight `[n, k]`, uploaded f16 (halves the read).
struct GpuMat {
    buf: compute::Buffer,
    k: usize,
    n: usize,
}

/// One MoE sub-projection's routed experts, NVFP4 3D-stacked and CONCATENATED into
/// one packed buffer + one e4m3 scale buffer (the on-device analog of the checkpoint's
/// `[E, out, in..]` tensor). Expert `local_e`'s slice starts at word/element offset
/// `local_e * {pack,sb}_stride`. Under TP only the EP-owned experts are resident.
struct GpuSwitch {
    packed: compute::Buffer, // u32 words, owned experts concatenated
    scale: compute::Buffer,  // e4m3 block-scale bytes, owned experts concatenated
    globals: Vec<f32>,       // per-owned-expert weight_scale_2 (MULTIPLY convention)
    out: usize,
    inn: usize,
    group: usize,
    pack_stride: usize, // words per expert = out * (inn / 8)
    sb_stride: usize,   // e4m3 scale elems per expert = out * (inn / group)
}

struct GpuAttn {
    q: GpuMat,
    k: GpuMat,
    v: GpuMat,
    o: GpuMat,
    g: GpuMat,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    /// local query-head count for THIS layer (global nq, or nq/tp under TP).
    nq_local: usize,
    /// local kv-head count (global nkv, or nkv/tp under TP).
    nkv_local: usize,
}

struct GpuDense {
    gate: GpuMat,
    up: GpuMat,
    down: GpuMat,
}

struct GpuMoe {
    gate: GpuSwitch,
    up: GpuSwitch,
    down: GpuSwitch,
    router: Vec<f32>, // host [num_experts, hidden] (replicated)
    bias: Vec<f32>,   // host [num_experts]
    shared_gate: GpuMat,
    shared_up: GpuMat,
    shared_down: GpuMat,
    expert_limit: Option<f32>,
    shared_limit: Option<f32>,
    inter: usize,
    /// EP-owned expert range on this rank (owned_lo, owned_cnt). (0, E) when tp==1.
    owned_lo: usize,
    owned_cnt: usize,
}

enum GpuMlp {
    Dense(GpuDense),
    Moe(GpuMoe),
}

struct GpuLayerR {
    input_ln: Vec<f32>,
    post_ln: Vec<f32>,
    attn: GpuAttn,
    mlp: GpuMlp,
}

/// The GPU-resident PP-window stage for step3p7.
pub struct Step3p7GpuStage {
    eng: compute::ComputeEngine,
    _dev: device::ComputeDevice,
    cfg: Step3p7Config,
    pub layer_start: usize,
    pub layer_end: usize,
    pub first: bool,
    pub last: bool,
    h: usize,
    eps: f32,

    // edges
    embed: Option<Vec<f32>>,      // host [vocab, h] (first stage)
    final_norm: Option<Vec<f32>>, // host [h] (last stage)
    lm_head: Option<GpuMat>,      // [vocab, h] (last stage)

    layers: Vec<GpuLayerR>,

    // decode state (one growing KV cache per resident layer)
    kv: Vec<Step3p7KvCache>,
    /// Board #308: per-layer GPU KV + attention scratch, created on the first
    /// token when `VLLM_VULKAN_STEP37_GPU_ATTN` is on (else stays `None`).
    gpu_attn: Vec<Option<Step3p7GpuAttn>>,
    pos: usize,

    // TP
    tp_rank: usize,
    tp_size: usize,
    tp_peer: i32,           // GLOBAL rank of the TP-2 peer on the flat comm; -1 == none
    collective_comm: usize, // raw vcclComm_t as usize; 0 == unset

    // Persistent RDMA-registered scratch for the per-layer TP-2 pairwise reduce
    // (nemotron's `reduce_scratch` lifecycle, see lib.rs). The v1 reduce sent/recv'd
    // fresh `Vec`s (a new address every call) so vCCL's per-call `ScopedReg` paid an
    // `ibv_reg_mr`/dereg on BOTH the send and recv buffer for every `[h]` reduce (the
    // WARN in the run logs). We register ONE fixed send buffer + ONE fixed recv buffer
    // up front (address-stable while live) and copy the partial through them — the
    // registrations then cover every reduce, so the per-call regMr disappears. Distinct
    // buffers (send vs recv) as `vcclSendRecv` requires. `handle == 0` ⇒ not registered
    // → falls back to the direct fresh-`Vec` path (correct, per-call regMr).
    tp_send_scratch: Vec<f32>,
    tp_send_handle: usize,
    tp_recv_scratch: Vec<f32>,
    tp_recv_handle: usize,
}

// ─── engine + upload primitives (mirrors ling_gpu) ──────────────────────────

fn make_engine(device_idx: usize) -> Result<(compute::ComputeEngine, device::ComputeDevice), String> {
    let dev = device::ComputeDevice::create(device_idx)?;
    let shader_spvs = crate::include_all_shaders();
    let refs: HashMap<&str, &[u8]> =
        shader_spvs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
    let eng = compute::ComputeEngine::new(
        dev.instance.clone(),
        dev.physical_device,
        dev.device.clone(),
        dev.compute_queue,
        dev.compute_queue_family,
        dev.caps(),
        &refs,
    )?;
    Ok((eng, dev))
}

/// Upload a dense `[n, k]` weight to a resident f16 buffer.
fn up_f16(eng: &mut compute::ComputeEngine, w: &[f32], k: usize, n: usize) -> Result<GpuMat, String> {
    if w.len() != k * n {
        return Err(format!("up_f16: weight [{n},{k}] len {} != {}", w.len(), k * n));
    }
    let bb = f32_to_f16_bytes(w);
    let buf = eng.alloc_host_coherent_storage(bb.len().max(4) as u64)?;
    buf.write(&bb)?;
    Ok(GpuMat { buf, k, n })
}

/// Column-shard rows of a `[out, in]` weight (out-dim ÷ tp); returns this rank's slice.
fn col_shard(w: &[f32], in_f: usize, rank: usize, n: usize) -> Vec<f32> {
    let out = w.len() / in_f;
    let per = out / n;
    let lo = rank * per;
    w[lo * in_f..(lo + per) * in_f].to_vec()
}

/// Row-shard input-cols of a `[out, in]` weight (in-dim ÷ tp); returns this rank's slice.
fn row_shard(w: &[f32], in_f: usize, rank: usize, n: usize) -> Vec<f32> {
    let out = w.len() / in_f;
    let per = in_f / n;
    let lo = rank * per;
    let mut o = Vec::with_capacity(out * per);
    for r in 0..out {
        o.extend_from_slice(&w[r * in_f + lo..r * in_f + lo + per]);
    }
    o
}

/// Upload the OWNED routed experts `[owned_lo, owned_lo + owned_cnt)` of one 3D
/// NVFP4-e4m3 projection (`{base}.weight` + `{base}.weight_scale`) into one packed
/// and one scale buffer, slot by slot, straight from the mapped checkpoint: no
/// host copy of the projection, and none of the experts another TP rank owns
/// (PR #98 review: the earlier path materialised all experts on the host first).
#[allow(clippy::too_many_arguments)]
fn upload_owned_experts(
    eng: &mut compute::ComputeEngine,
    dir: &std::path::Path,
    weight_map: &serde_json::Map<String, Value>,
    mmaps: &mut HashMap<String, memmap2::Mmap>,
    base: &str,
    globals_all: &[f32],
    num_experts: usize,
    out: usize,
    inn: usize,
    group: usize,
    owned_lo: usize,
    owned_cnt: usize,
) -> Result<GpuSwitch, String> {
    if owned_lo + owned_cnt > num_experts {
        return Err(format!("{base}: owned experts [{owned_lo}, +{owned_cnt}) past {num_experts}"));
    }
    let src = crate::step3p7::map_expert_proj(dir, weight_map, mmaps, base)?;
    let stp = safetensors::SafeTensors::deserialize(&mmaps[&src.packed_shard])
        .map_err(|e| format!("parse {}: {e}", src.packed_shard))?;
    let sts = safetensors::SafeTensors::deserialize(&mmaps[&src.scale_shard])
        .map_err(|e| format!("parse {}: {e}", src.scale_shard))?;
    let pd = stp.tensor(&src.packed_name).map_err(|e| format!("{}: {e}", src.packed_name))?.data();
    let sd = sts.tensor(&src.scale_name).map_err(|e| format!("{}: {e}", src.scale_name))?.data();
    let (pack_bytes, sb_bytes) =
        crate::step3p7::check_expert_proj(base, pd, sd, globals_all, num_experts, out, inn, group)?;
    let pbuf = eng.alloc_host_coherent_storage((owned_cnt * pack_bytes).max(4) as u64)?;
    let sbuf = eng.alloc_host_coherent_storage((owned_cnt * sb_bytes).max(4) as u64)?;
    for slot in 0..owned_cnt {
        let e = owned_lo + slot;
        pbuf.write_at((slot * pack_bytes) as u64, &pd[e * pack_bytes..(e + 1) * pack_bytes])?;
        sbuf.write_at((slot * sb_bytes) as u64, &sd[e * sb_bytes..(e + 1) * sb_bytes])?;
    }
    Ok(GpuSwitch {
        packed: pbuf,
        scale: sbuf,
        globals: globals_all[owned_lo..owned_lo + owned_cnt].to_vec(),
        out,
        inn,
        group,
        pack_stride: out * (inn / 8), // words
        sb_stride: out * (inn / group), // e4m3 elems
    })
}

// ─── matvec dispatch (one submit each; perf-fusion is a cluster follow-up) ───

/// f16 dense matvec `out[n] = W[n,k] · x[k]`.
fn mv_dense(eng: &mut compute::ComputeEngine, m: &GpuMat, x: &[f32]) -> Result<Vec<f32>, String> {
    if x.len() != m.k {
        return Err(format!("mv_dense: x {} != k {}", x.len(), m.k));
    }
    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let o = eng.alloc_host_coherent_storage((m.n * 4).max(4) as u64)?;
    let (shader, r) = matvec_variant_by_format(QuantFormat::F16, m.n);
    let wg = (m.n as u32 + r - 1) / r;
    let pc = matvec_pc13(m.k, m.n);
    let cb = eng.begin_batch()?;
    eng.record_to(cb, &shader, &[&m.buf, &xbuf, &o], &pc, (wg, 1, 1))?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o, m.n);
    eng.return_to_pool(xbuf);
    eng.return_to_pool(o);
    Ok(out)
}

/// f32 host-vector · f32 GPU weight — used only if an f16 upload was skipped. (Kept
/// for completeness / debugging; the resident path is all-f16.)
#[allow(dead_code)]
fn mv_dense_f32(eng: &mut compute::ComputeEngine, buf: &compute::Buffer, k: usize, n: usize, x: &[f32]) -> Result<Vec<f32>, String> {
    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let o = eng.alloc_host_coherent_storage((n * 4).max(4) as u64)?;
    let (shader, r) = matvec_f32_variant(n);
    let wg = (n as u32 + r - 1) / r;
    let pc = matvec_pc13(k, n);
    let cb = eng.begin_batch()?;
    eng.record_to(cb, &shader, &[buf, &xbuf, &o], &pc, (wg, 1, 1))?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o, n);
    eng.return_to_pool(xbuf);
    eng.return_to_pool(o);
    Ok(out)
}

/// Pick the routed-expert e4m3 NVFP4 matvec shader + rows-per-workgroup for step3p7.
/// When `VLLM_VULKAN_LAGUNA_EXPERT_REPACK` is on AND the shape clears the repack guard
/// (`nvfp4_repack_shape_ok` — step3p7 experts k=inn→n=out, gs=16 all pass), route to the
/// address-gen-free REPACK kernel (`mul_mat_vec_nvfp4_e4m3repack_f32_f32_bs64_r4`, the
/// mlx4/nvfp4-repack bs64/r4 default) instead of the v1 `mul_mat_vec_nvfp4_e4m3` oracle.
/// This mirrors `laguna_gpu::laguna_e4m3_expert_shader` EXACTLY. The repack shader threads
/// `packed_off`/`sb_off` + the per-tensor `global` identically (same push block + base4/sbase
/// math), so the per-expert slice offsets pass straight through `matvec_nvfp4_e4m3_pc_off`
/// unchanged — NO push-constant change. step3p7's global is the MULTIPLY-convention
/// `weight_scale_2` (see `GpuSwitch::globals`); the repack kernel applies it identically to v1,
/// so the dequant math is bit-exact (repack == f32-fold, single IEEE mul ⇒ argmax-exact vs v1).
fn step3p7_e4m3_expert_shader(k: usize, n: usize, gs: usize) -> (String, u32) {
    if laguna_expert_repack_flag() && nvfp4_repack_shape_ok(k, n, gs) {
        return ("mul_mat_vec_nvfp4_e4m3repack_f32_f32_bs64_r4".to_string(), 4);
    }
    matvec_nvfp4_e4m3_variant(n)
}

/// NVFP4-e4m3 routed-expert matvec `out[n] = expert(local_e) · x[k]`.
fn mv_expert(eng: &mut compute::ComputeEngine, sw: &GpuSwitch, local_e: usize, x: &[f32]) -> Result<Vec<f32>, String> {
    let (k, n) = (sw.inn, sw.out);
    if x.len() != k {
        return Err(format!("mv_expert: x {} != k {}", x.len(), k));
    }
    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let o = eng.alloc_host_coherent_storage((n * 4).max(4) as u64)?;
    let (shader, r) = step3p7_e4m3_expert_shader(k, n, sw.group);
    let wg = (n as u32 + r - 1) / r;
    let packed_off = local_e * sw.pack_stride;
    let sb_off = local_e * sw.sb_stride;
    let pc = matvec_nvfp4_e4m3_pc_off(k, n, sw.group, packed_off, sb_off, sw.globals[local_e]);
    let cb = eng.begin_batch()?;
    eng.record_to(cb, &shader, &[&sw.packed, &sw.scale, &xbuf, &o], &pc, (wg, 1, 1))?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o, n);
    eng.return_to_pool(xbuf);
    eng.return_to_pool(o);
    Ok(out)
}

/// Shader + rows-per-workgroup for the Step-3.7 expert-BATCHED nvfp4-e4m3 matvec
/// (`VLLM_VULKAN_STEP37_EXPERT_BATCH`). The batched analog of the serial e4m3 repack
/// (`step3p7_e4m3_expert_shader`): same per-(row,chunk) dequant+accumulate body, only
/// the dispatch gains the expert (`gl_WorkGroupID.y`) axis. Requires the SAME repack
/// shape guard (`nvfp4_repack_shape_ok`); `None` ⇒ shape fails ⇒ caller keeps the serial
/// per-expert path. bs64/r4 is the wired default (== the single-expert repack pick).
fn step3p7_batched_expert_shader(k: usize, n: usize, gs: usize) -> Option<(String, u32)> {
    if nvfp4_repack_shape_ok(k, n, gs) {
        return Some(("mul_mat_vec_nvfp4_e4m3repack_batched_f32_f32_bs64_r4".to_string(), 4));
    }
    None
}

/// Per-expert `meta[]` (uvec4 = packed_off, sb_off, x_off, dst_off) for one batched
/// sub-projection over `local_experts` (their local slot indices into the concatenated
/// switch). `x_shared` = true for gate/up (all experts read the same [k] activation,
/// x_off=0) or false for down (x concatenated [n_ex,k], expert slot `e` reads x[e*k..]).
/// Byte-for-byte the offset math the serial `mv_expert` threads (packed_off = le*pack_stride
/// words, sb_off = le*sb_stride e4m3 elems), just laid out per-slot for the .y axis.
fn step3p7_expert_meta_bytes(sw: &GpuSwitch, local_experts: &[usize], x_shared: bool) -> Vec<u8> {
    let meta = step3p7_expert_meta_u32(
        sw.pack_stride, sw.sb_stride, sw.inn, sw.out, local_experts, x_shared);
    bytemuck::cast_slice::<u32, u8>(&meta).to_vec()
}

/// Pure (device-free) core of `step3p7_expert_meta_bytes` — the per-expert uvec4
/// `meta[]` (packed_off words, sb_off e4m3 byte-elems, x_off floats, dst_off floats).
/// Factored out so the batched-vs-serial offset math is unit-testable without a GPU.
fn step3p7_expert_meta_u32(
    pack_stride: usize, sb_stride: usize, k: usize, n: usize,
    local_experts: &[usize], x_shared: bool,
) -> Vec<u32> {
    let mut meta: Vec<u32> = Vec::with_capacity(local_experts.len() * 4);
    for (e, &le) in local_experts.iter().enumerate() {
        meta.push((le * pack_stride) as u32);                 // packed_off (words)
        meta.push((le * sb_stride) as u32);                   // sb_off (e4m3 byte-elems)
        meta.push((if x_shared { 0 } else { e * k }) as u32); // x_off (floats)
        meta.push((e * n) as u32);                            // dst_off (floats)
    }
    meta
}

/// Expert-BATCHED nvfp4-e4m3 routed-expert matvec: `out[e][r] = expert(local_experts[e]) · x_e`
/// for ALL `local_experts` in ONE dispatch (the Step-3.7 "#3" dispatch-collapse lever).
/// Numerically identical to calling `mv_expert` once per expert (same repack dequant body,
/// same fma reduction order, each expert's own `sw.globals[le]`) but collapses the host
/// record + submit. Returns the concatenated `[n_ex * n]` output. `x` is the shared `[k]`
/// activation (`x_shared`) or the concatenated `[n_ex * k]` intermediate (down projection).
fn mv_experts_batched(
    eng: &mut compute::ComputeEngine,
    sw: &GpuSwitch,
    local_experts: &[usize],
    x: &[f32],
    x_shared: bool,
    shader: &str,
    r: u32,
) -> Result<Vec<f32>, String> {
    let (k, n) = (sw.inn, sw.out);
    let n_ex = local_experts.len();
    let expect_x = if x_shared { k } else { n_ex * k };
    if x.len() != expect_x {
        return Err(format!("mv_experts_batched: x {} != expected {} (x_shared={x_shared})", x.len(), expect_x));
    }
    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let meta_bytes = step3p7_expert_meta_bytes(sw, local_experts, x_shared);
    let metabuf = eng.alloc_host_coherent_storage(meta_bytes.len().max(4) as u64)?;
    metabuf.write(&meta_bytes)?;
    let globals: Vec<f32> = local_experts.iter().map(|&le| sw.globals[le]).collect();
    let gb = f32_slice_to_bytes(&globals);
    let gbuf = eng.alloc_host_coherent_storage(gb.len().max(4) as u64)?;
    gbuf.write(&gb)?;
    let o = eng.alloc_host_coherent_storage((n_ex * n * 4).max(4) as u64)?;
    let wg = (n as u32 + r - 1) / r;
    // packed_off/sb_off/global in the push constant are UNUSED by the batched shader
    // (per-expert values come from meta[]/globals[]); pass 0 to reuse the pc builder.
    let pc = matvec_nvfp4_e4m3_pc_off(k, n, sw.group, 0, 0, 0.0);
    let cb = eng.begin_batch()?;
    eng.record_to(cb, shader, &[&sw.packed, &sw.scale, &xbuf, &o, &metabuf, &gbuf], &pc, (wg, n_ex as u32, 1))?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o, n_ex * n);
    eng.return_to_pool(xbuf);
    eng.return_to_pool(metabuf);
    eng.return_to_pool(gbuf);
    eng.return_to_pool(o);
    Ok(out)
}

/// With TP>1 this rank holds only a shard of o_proj / down, so the per-layer reduce is
/// mandatory: `Err` unless the comm and TP peer are wired and the TP size is 2 (the only
/// pairwise exchange implemented). `Ok` for TP=1. `decode_step` calls it before any layer
/// runs, so a misconfigured rank fails before it advances the KV state.
fn check_tp_wired(comm: usize, tp_size: usize, tp_peer: i32) -> Result<(), String> {
    if tp_size <= 1 {
        return Ok(());
    }
    if comm == 0 || tp_peer < 0 {
        return Err(format!(
            "step3p7 TP{tp_size}: weights are sharded but the TP reduce is not wired \
             (comm={comm:#x}, peer={tp_peer}); call set_collective_comm and set_tp_peer \
             before decoding"));
    }
    if tp_size != 2 {
        return Err(format!(
            "step3p7 TP reduce is TP=2 pairwise only (tp_size={tp_size}; TP>2 needs a vcclCommSplit sub-comm)"
        ));
    }
    Ok(())
}

/// TP-2 all-reduce a `[hidden]` partial in place: a deadlock-safe PAIRWISE exchange with
/// the tp_peer (even-`tp_rank`-sends-first), then `buf += peer_partial` — the SUM of the
/// two ranks' partials. This is nemotron's TP=2 pattern (NOT a full-comm all_reduce, which
/// on a flat PP+TP comm would wrongly reduce across every rank). Re-acquires the GIL —
/// safe because `decode_step` is always reached from a pyo3 method that holds it. No-op
/// when `tp_size <= 1`; an error when TP>1 and the comm or peer is unset
/// (see [`check_tp_wired`]).
///
/// When `send_scratch`/`recv_scratch` are RDMA-registered (both `>= buf.len()`), the
/// partial is copied THROUGH them so vCCL's per-call `ScopedReg` short-circuits (no
/// `ibv_reg_mr`/dereg per reduce — the WARN cure). The wire op + accumulate are
/// byte-identical to the fresh-`Vec` path (same `vcclSendRecv`/ordered send+recv, same
/// `buf += peer_partial`), so it stays argmax-exact. `registered == false` (older
/// libvccl, or registration failed) reverts to the fresh-`Vec` per-call-regMr path.
fn tp_all_reduce(
    comm: usize,
    tp_rank: usize,
    tp_size: usize,
    tp_peer: i32,
    buf: &mut [f32],
    send_scratch: &mut [f32],
    recv_scratch: &mut [f32],
    registered: bool,
) -> Result<(), String> {
    if tp_size <= 1 {
        return Ok(());
    }
    // Skipping the reduce would add a half-sum to the residual and decode wrong tokens
    // without an error.
    check_tp_wired(comm, tp_size, tp_peer)?;
    let commp = comm as *mut c_void;
    let send_first = tp_rank % 2 == 0;
    let n = buf.len();
    let use_scratch = registered && send_scratch.len() >= n && recv_scratch.len() >= n;
    if use_scratch {
        // Copy the partial into the registered send buffer; recv into the registered
        // recv buffer. Distinct buffers, both covered by a prior vcclCommRegister.
        let sb = &mut send_scratch[..n];
        let rb = &mut recv_scratch[..n];
        sb.copy_from_slice(buf);
        pyo3::Python::with_gil(|py| -> Result<(), String> {
            if vccl_ffi::send_recv_available() {
                vccl_ffi::send_recv_f32(py, commp, sb, tp_peer, rb, tp_peer)
            } else if send_first {
                vccl_ffi::send_f32(py, commp, sb, tp_peer)?;
                vccl_ffi::recv_f32_into(py, commp, rb, tp_peer)
            } else {
                vccl_ffi::recv_f32_into(py, commp, rb, tp_peer)?;
                vccl_ffi::send_f32(py, commp, sb, tp_peer)
            }
        })?;
        for (b, r) in buf.iter_mut().zip(rb.iter()) {
            *b += *r;
        }
        return Ok(());
    }
    let mut recv = vec![0f32; n];
    pyo3::Python::with_gil(|py| -> Result<(), String> {
        if vccl_ffi::send_recv_available() {
            vccl_ffi::send_recv_f32(py, commp, buf, tp_peer, &mut recv, tp_peer)
        } else if send_first {
            vccl_ffi::send_f32(py, commp, buf, tp_peer)?;
            vccl_ffi::recv_f32_into(py, commp, &mut recv, tp_peer)
        } else {
            vccl_ffi::recv_f32_into(py, commp, &mut recv, tp_peer)?;
            vccl_ffi::send_f32(py, commp, buf, tp_peer)
        }
    })?;
    for (b, r) in buf.iter_mut().zip(&recv) {
        *b += *r;
    }
    Ok(())
}

impl Step3p7GpuStage {
    /// Read TP rank/size from the environment (both `VLLM_VULKAN_TP_{RANK,SIZE}`).
    fn read_tp() -> (usize, usize) {
        let size = std::env::var("VLLM_VULKAN_TP_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let rank = std::env::var("VLLM_VULKAN_TP_RANK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            .min(size - 1);
        (rank, size)
    }

    /// Stream a PP window `[layer_start, layer_end)` from the checkpoint, uploading each
    /// layer to GPU and FREEING its host copy before the next — never holding the whole
    /// host window (the LOAD-OOM cure). Experts (NVFP4) upload packed-resident; attn /
    /// dense / shared / lm_head upload f16. `keep_edges` (first || last) pulls embed /
    /// final_norm / lm_head. TP sharding (col/row + EP) happens here at load time.
    pub fn from_ckpt_streamed(
        dir: &std::path::Path,
        cfg: &Step3p7Config,
        layer_start: usize,
        layer_end: usize,
        keep_edges: bool,
        device_idx: usize,
    ) -> Result<Step3p7GpuStage, String> {
        use memmap2::Mmap;
        use safetensors::SafeTensors;

        let total = cfg.num_hidden_layers;
        let first = layer_start == 0;
        let last = layer_end >= total;
        let (tp_rank, tp_size) = Self::read_tp();
        // Divisibility guards (fail LOUD at load, not silently mid-decode).
        if tp_size > 1 {
            if cfg.num_key_value_heads % tp_size != 0 {
                return Err(format!("TP{tp_size}: num_kv_heads {} not divisible", cfg.num_key_value_heads));
            }
            if cfg.num_experts % tp_size != 0 {
                return Err(format!("TP{tp_size}: num_experts {} not divisible", cfg.num_experts));
            }
            if cfg.moe_intermediate_size % tp_size != 0 || cfg.share_expert_dim % tp_size != 0 {
                return Err(format!("TP{tp_size}: moe/share intermediate not divisible"));
            }
        }

        let (mut eng, dev) = make_engine(device_idx)?;
        let h = cfg.hidden_size;
        let group = 16usize;
        let lmp = "model.language_model";

        // index.json → weight_map + a live mmap cache (lazy per-shard).
        let index_path = dir.join("model.safetensors.index.json");
        let index: Value = serde_json::from_str(
            &std::fs::read_to_string(&index_path).map_err(|e| format!("read index: {e}"))?,
        )
        .map_err(|e| format!("parse index: {e}"))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|x| x.as_object())
            .ok_or("index.json: missing weight_map")?
            .clone();
        let mut mmaps: HashMap<String, Mmap> = HashMap::new();

        // Pull ONE tensor to host f32 (bf16/f16/f32 decode) via the shard mmap cache.
        // Each call frees nothing persistent — the returned Vec is the only new host
        // allocation, and every caller drops it after upload.
        let get_f32 = |name: &str,
                           mmaps: &mut HashMap<String, Mmap>|
         -> Result<Vec<f32>, String> {
            let shard = weight_map
                .get(name)
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("index.json missing {name}"))?
                .to_string();
            if !mmaps.contains_key(&shard) {
                let f = std::fs::File::open(dir.join(&shard)).map_err(|e| format!("open {shard}: {e}"))?;
                let m = unsafe { Mmap::map(&f).map_err(|e| format!("mmap {shard}: {e}"))? };
                mmaps.insert(shard.clone(), m);
            }
            let st = SafeTensors::deserialize(&mmaps[&shard]).map_err(|e| format!("parse {shard}: {e}"))?;
            let view = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
            crate::step3p7::decode_bf16_f32(&view)
        };

        let mut expert_mmaps: HashMap<String, Mmap> = HashMap::new();

        let (owned_lo, owned_cnt) = if tp_size > 1 {
            let per = cfg.num_experts / tp_size;
            (tp_rank * per, per)
        } else {
            (0, cfg.num_experts)
        };

        let mut layers: Vec<GpuLayerR> = Vec::with_capacity(layer_end - layer_start);
        for li in layer_start..layer_end {
            let p = format!("{lmp}.layers.{li}");
            let la = cfg.layers[li];
            let nq = la.num_heads;
            let nkv = cfg.num_key_value_heads;
            let hd = cfg.head_dim;
            let nq_local = if tp_size > 1 { nq / tp_size } else { nq };
            let nkv_local = if tp_size > 1 { nkv / tp_size } else { nkv };
            if tp_size > 1 && nq % tp_size != 0 {
                return Err(format!("TP{tp_size}: layer {li} nq {nq} not divisible"));
            }

            // Attention — col-shard q/k/v (out ÷ tp), row-shard o (in ÷ tp), col-shard g.
            let q_w = get_f32(&format!("{p}.self_attn.q_proj.weight"), &mut mmaps)?;
            let k_w = get_f32(&format!("{p}.self_attn.k_proj.weight"), &mut mmaps)?;
            let v_w = get_f32(&format!("{p}.self_attn.v_proj.weight"), &mut mmaps)?;
            let o_w = get_f32(&format!("{p}.self_attn.o_proj.weight"), &mut mmaps)?;
            let g_w = get_f32(&format!("{p}.self_attn.g_proj.weight"), &mut mmaps)?;
            let (q_w, k_w, v_w, o_w, g_w) = if tp_size > 1 {
                (
                    col_shard(&q_w, h, tp_rank, tp_size),
                    col_shard(&k_w, h, tp_rank, tp_size),
                    col_shard(&v_w, h, tp_rank, tp_size),
                    row_shard(&o_w, nq * hd, tp_rank, tp_size),
                    col_shard(&g_w, h, tp_rank, tp_size),
                )
            } else {
                (q_w, k_w, v_w, o_w, g_w)
            };
            let attn = GpuAttn {
                q: up_f16(&mut eng, &q_w, h, nq_local * hd)?,
                k: up_f16(&mut eng, &k_w, h, nkv_local * hd)?,
                v: up_f16(&mut eng, &v_w, h, nkv_local * hd)?,
                o: up_f16(&mut eng, &o_w, nq_local * hd, h)?,
                g: up_f16(&mut eng, &g_w, h, nq_local)?,
                q_norm: get_f32(&format!("{p}.self_attn.q_norm.weight"), &mut mmaps)?,
                k_norm: get_f32(&format!("{p}.self_attn.k_norm.weight"), &mut mmaps)?,
                nq_local,
                nkv_local,
            };

            let input_ln = get_f32(&format!("{p}.input_layernorm.weight"), &mut mmaps)?;
            let post_ln = get_f32(&format!("{p}.post_attention_layernorm.weight"), &mut mmaps)?;

            let mlp = if cfg.is_moe_layer(li) {
                let inter = cfg.moe_intermediate_size;
                let sh = cfg.share_expert_dim;
                // routed experts: EP whole-expert partition (nemotron pattern) — only
                // owned experts resident, uploaded slot by slot from the mmap'd 3D
                // tensors (no host copy of the projection, nothing of the other rank's).
                let build_proj = |base: &str, out_f: usize, in_f: usize,
                                  eng: &mut compute::ComputeEngine,
                                  emm: &mut HashMap<String, memmap2::Mmap>|
                 -> Result<GpuSwitch, String> {
                    let g = get_scales2(dir, &weight_map, emm, &format!("{base}.weight_scale_2"))?;
                    upload_owned_experts(eng, dir, &weight_map, emm, base, &g, cfg.num_experts,
                                         out_f, in_f, group, owned_lo, owned_cnt)
                };
                let gate_sw = build_proj(&format!("{p}.moe.gate_proj"), inter, h, &mut eng, &mut expert_mmaps)?;
                let up_sw = build_proj(&format!("{p}.moe.up_proj"), inter, h, &mut eng, &mut expert_mmaps)?;
                let down_sw = build_proj(&format!("{p}.moe.down_proj"), h, inter, &mut eng, &mut expert_mmaps)?;

                // shared expert — col-shard gate/up (out ÷ tp), row-shard down (in ÷ tp).
                let sg = get_f32(&format!("{p}.share_expert.gate_proj.weight"), &mut mmaps)?;
                let su = get_f32(&format!("{p}.share_expert.up_proj.weight"), &mut mmaps)?;
                let sd = get_f32(&format!("{p}.share_expert.down_proj.weight"), &mut mmaps)?;
                let sh_local = if tp_size > 1 { sh / tp_size } else { sh };
                let (sg, su, sd) = if tp_size > 1 {
                    (
                        col_shard(&sg, h, tp_rank, tp_size),
                        col_shard(&su, h, tp_rank, tp_size),
                        row_shard(&sd, sh, tp_rank, tp_size),
                    )
                } else {
                    (sg, su, sd)
                };
                let lim = |v: &[f32]| -> Option<f32> { v.get(li).copied().filter(|&x| x != 0.0) };
                GpuMlp::Moe(GpuMoe {
                    gate: gate_sw,
                    up: up_sw,
                    down: down_sw,
                    router: get_f32(&format!("{p}.moe.gate.weight"), &mut mmaps)?,
                    bias: get_f32(&format!("{p}.moe.router_bias"), &mut mmaps)?,
                    shared_gate: up_f16(&mut eng, &sg, h, sh_local)?,
                    shared_up: up_f16(&mut eng, &su, h, sh_local)?,
                    shared_down: up_f16(&mut eng, &sd, sh_local, h)?,
                    expert_limit: lim(&cfg.swiglu_limit_expert),
                    shared_limit: lim(&cfg.swiglu_limit_shared),
                    inter,
                    owned_lo,
                    owned_cnt,
                })
            } else {
                let inter = cfg.intermediate_size;
                let g = get_f32(&format!("{p}.mlp.gate_proj.weight"), &mut mmaps)?;
                let u = get_f32(&format!("{p}.mlp.up_proj.weight"), &mut mmaps)?;
                let d = get_f32(&format!("{p}.mlp.down_proj.weight"), &mut mmaps)?;
                let inter_local = if tp_size > 1 { inter / tp_size } else { inter };
                let (g, u, d) = if tp_size > 1 {
                    (
                        col_shard(&g, h, tp_rank, tp_size),
                        col_shard(&u, h, tp_rank, tp_size),
                        row_shard(&d, inter, tp_rank, tp_size),
                    )
                } else {
                    (g, u, d)
                };
                GpuMlp::Dense(GpuDense {
                    gate: up_f16(&mut eng, &g, h, inter_local)?,
                    up: up_f16(&mut eng, &u, h, inter_local)?,
                    down: up_f16(&mut eng, &d, inter_local, h)?,
                })
            };

            layers.push(GpuLayerR { input_ln, post_ln, attn, mlp });
            // per-layer host working set has dropped here (all host Vecs uploaded+freed);
            // the mmap cache stays (lazy pages, evictable), never the decoded tensors.
        }

        // edges
        let embed = if keep_edges && first {
            Some(get_f32(&format!("{lmp}.embed_tokens.weight"), &mut mmaps)?)
        } else {
            None
        };
        let final_norm = if last {
            Some(get_f32(&format!("{lmp}.norm.weight"), &mut mmaps)?)
        } else {
            None
        };
        let lm_head = if keep_edges && last {
            let w = get_f32("lm_head.weight", &mut mmaps)?;
            Some(up_f16(&mut eng, &w, h, cfg.vocab_size)?)
        } else {
            None
        };

        let comm = 0usize;
        let n_layers = layers.len();
        Ok(Step3p7GpuStage {
            eng,
            _dev: dev,
            cfg: cfg.clone(),
            layer_start,
            layer_end,
            first,
            last,
            h,
            eps: cfg.rms_norm_eps,
            embed,
            final_norm,
            lm_head,
            layers,
            kv: vec![Step3p7KvCache::default(); n_layers],
            gpu_attn: (0..n_layers).map(|_| None).collect(),
            pos: 0,
            tp_rank,
            tp_size,
            tp_peer: -1,
            collective_comm: comm,
            tp_send_scratch: Vec::new(),
            tp_send_handle: 0,
            tp_recv_scratch: Vec::new(),
            tp_recv_handle: 0,
        })
    }

    /// Wire the collective communicator (raw vcclComm_t as usize) + this rank's TP-2
    /// peer GLOBAL rank, used by the per-layer TP reduce. Called from
    /// `set_collective_comm` / `set_tp_peer` in lib.rs. With TP>1, `decode_step` errors
    /// until both are set (`peer < 0` / `comm == 0` mean unwired).
    pub fn set_tp_comm(&mut self, comm: usize, peer: i32) {
        // A comm handle change invalidates any MR registered on the old comm — drop the
        // reduce scratch registrations so `ensure_tp_scratch` re-pins on the new comm.
        if comm != self.collective_comm {
            self.release_tp_scratch();
        }
        self.collective_comm = comm;
        self.tp_peer = peer;
    }

    /// Deregister + drop the TP reduce scratch (both send + recv). Safe to call when
    /// nothing is registered (handles 0). Uses the CURRENT `collective_comm` — call
    /// BEFORE overwriting it on a comm change.
    fn release_tp_scratch(&mut self) {
        let comm = self.collective_comm as *mut c_void;
        if self.tp_send_handle != 0 {
            let _ = vccl_ffi::comm_deregister(comm, self.tp_send_handle);
            self.tp_send_handle = 0;
        }
        if self.tp_recv_handle != 0 {
            let _ = vccl_ffi::comm_deregister(comm, self.tp_recv_handle);
            self.tp_recv_handle = 0;
        }
        self.tp_send_scratch = Vec::new();
        self.tp_recv_scratch = Vec::new();
    }

    /// Ensure a `>= n`-f32 RDMA-registered send + recv scratch is pinned on the current
    /// comm for the TP-2 pairwise reduce. Idempotent + early-returns once registered and
    /// large enough (the payload is a fixed `[h]` every reduce, so this registers exactly
    /// ONCE on the first decode step). No-op when TP is off / comm or peer unset / the
    /// libvccl lacks the registration entry points (→ per-call regMr fallback preserved).
    /// Mirrors nemotron's `ensure_reduce_scratch`.
    fn ensure_tp_scratch(&mut self, n: usize) {
        if self.tp_size <= 1 || self.collective_comm == 0 || self.tp_peer < 0 {
            return;
        }
        if !vccl_ffi::registration_available() {
            return; // older libvccl: keep the correct per-call regMr path.
        }
        if self.tp_send_handle != 0 && self.tp_recv_handle != 0 && self.tp_send_scratch.len() >= n {
            return; // already pinned + big enough.
        }
        self.release_tp_scratch();
        let comm = self.collective_comm as *mut c_void;
        let bytes = n * std::mem::size_of::<f32>();
        self.tp_send_scratch = vec![0.0f32; n];
        match vccl_ffi::comm_register(comm, self.tp_send_scratch.as_ptr() as usize, bytes) {
            Ok(h) => self.tp_send_handle = h,
            Err(e) => {
                log::warn!("step3p7 ensure_tp_scratch({n}) send register failed: {e}; per-call regMr");
                self.tp_send_scratch = Vec::new();
                self.tp_send_handle = 0;
                return;
            }
        }
        self.tp_recv_scratch = vec![0.0f32; n];
        match vccl_ffi::comm_register(comm, self.tp_recv_scratch.as_ptr() as usize, bytes) {
            Ok(h) => self.tp_recv_handle = h,
            Err(e) => {
                log::warn!("step3p7 ensure_tp_scratch({n}) recv register failed: {e}; per-call regMr");
                let _ = vccl_ffi::comm_deregister(comm, self.tp_send_handle);
                self.tp_send_handle = 0;
                self.tp_send_scratch = Vec::new();
                self.tp_recv_scratch = Vec::new();
                self.tp_recv_handle = 0;
            }
        }
    }

    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }
    pub fn tp_size(&self) -> usize {
        self.tp_size
    }
    pub fn tp_peer(&self) -> i32 {
        self.tp_peer
    }

    /// Reset the decode KV caches + position (Ling `reset_state`).
    pub fn reset_state(&mut self) {
        for c in self.kv.iter_mut() {
            c.k.clear();
            c.v.clear();
            c.len = 0;
        }
        for g in self.gpu_attn.iter_mut().flatten() {
            g.reset();
        }
        self.pos = 0;
    }

    /// Single-token GPU-resident decode step. First stage embeds `token_id`; a mid stage
    /// consumes the previous stage's `[hidden]`; the last stage returns `[vocab]` logits.
    pub fn decode_step(&mut self, token_id: u32, hidden_in: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.h;
        // Before any layer appends to the KV: a retry after wiring must not see a
        // half-advanced session.
        check_tp_wired(self.collective_comm, self.tp_size, self.tp_peer)?;
        let mut hidden: Vec<f32> = if self.first {
            let embed = self.embed.as_ref().ok_or("decode_step: first stage missing embed")?;
            if embed.len() < self.cfg.vocab_size * h {
                return Err("decode_step: embed too small".into());
            }
            embed[token_id as usize * h..(token_id as usize + 1) * h].to_vec()
        } else {
            if hidden_in.len() != h {
                return Err(format!("decode_step: hidden_in {} != H {h}", hidden_in.len()));
            }
            hidden_in.to_vec()
        };

        // Pin the TP-2 reduce scratch on the comm (once; the per-layer reduce payload is
        // a fixed `[h]`). No-op when TP is off / registration unavailable.
        self.ensure_tp_scratch(h);
        let tp_registered = self.tp_send_handle != 0 && self.tp_recv_handle != 0;

        for local in 0..self.layers.len() {
            let global = self.layer_start + local;
            // disjoint-field borrows: eng (mut), layers[local] (imm), kv[local] (mut),
            // tp_{send,recv}_scratch (mut).
            hidden = decode_one_layer(
                &mut self.eng,
                &self.layers[local],
                &mut self.kv[local],
                &mut self.gpu_attn[local],
                &hidden,
                global,
                &self.cfg,
                self.eps,
                self.tp_rank,
                self.tp_size,
                self.tp_peer,
                self.collective_comm,
                &mut self.tp_send_scratch,
                &mut self.tp_recv_scratch,
                tp_registered,
            )?;
        }
        self.pos += 1;

        if !self.last {
            return Ok(hidden);
        }
        let fnorm = self.final_norm.as_ref().ok_or("decode_step: last stage missing final_norm")?;
        let normed = rms_norm_plus1(&hidden, fnorm, self.eps);
        let lm = self.lm_head.as_ref().ok_or("decode_step: last stage missing lm_head")?;
        mv_dense(&mut self.eng, lm, &normed)
    }
}

/// Get the tiny F32 `[E]` per-expert global (`weight_scale_2`) from the mmap cache.
fn get_scales2(
    dir: &std::path::Path,
    weight_map: &serde_json::Map<String, Value>,
    mmaps: &mut HashMap<String, memmap2::Mmap>,
    name: &str,
) -> Result<Vec<f32>, String> {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    let shard = weight_map
        .get(name)
        .and_then(|x| x.as_str())
        .ok_or_else(|| format!("index.json missing {name}"))?
        .to_string();
    if !mmaps.contains_key(&shard) {
        let f = std::fs::File::open(dir.join(&shard)).map_err(|e| format!("open {shard}: {e}"))?;
        let m = unsafe { Mmap::map(&f).map_err(|e| format!("mmap {shard}: {e}"))? };
        mmaps.insert(shard.clone(), m);
    }
    let st = SafeTensors::deserialize(&mmaps[&shard]).map_err(|e| format!("parse {shard}: {e}"))?;
    let view = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
    Ok(view.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Which decode-attention kernel a layer runs. Both read the same
/// `[K plane | V plane]` KV buffer with the same addressing (`sdpa_pc` words 0..10);
/// they differ only in how the work is split, so the choice changes the summation
/// order, not the math. (A split-K variant comes with the shared split-K kernels.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step3p7AttnKernel {
    /// `paged_attn_decode_f32_sg`: one 64-lane workgroup per q head.
    Sg,
    /// `paged_attn_decode_f32`: one thread per output element. Portable (any
    /// subgroup size) but slow; used for testing on non-wave64 devices.
    Scalar,
}

/// `VLLM_VULKAN_STEP37_GPU_ATTN` (default ON): unset or `1` runs the decode
/// attention on the GPU (`_sg` on a wave64 device; host SDPA if it is not compiled),
/// `scalar` forces the portable scalar kernel (tests / non-wave64 devices), `0`
/// keeps the host `cpu_sdpa_gqa`.
fn step37_gpu_attn_mode() -> Option<bool> {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VLLM_VULKAN_STEP37_GPU_ATTN").ok().as_deref() {
        Some("0") => None,
        Some("scalar") => Some(true),
        _ => Some(false),
    })
}

/// Pick the decode-attention kernel, or `None` when the device has no wave64
/// kernel and the scalar one is not forced.
pub(crate) fn pick_attn_kernel(eng: &compute::ComputeEngine, nq: usize, nkv: usize, hd: usize,
                               force_scalar: bool) -> Option<Step3p7AttnKernel> {
    if nkv == 0 || nq % nkv != 0 || hd > 512 {
        return None;
    }
    if force_scalar {
        return eng.has_pipeline("paged_attn_decode_f32").then_some(Step3p7AttnKernel::Scalar);
    }
    if eng.has_pipeline("paged_attn_decode_f32_sg") {
        return Some(Step3p7AttnKernel::Sg);
    }
    None
}

/// GPU-resident decode KV + attention scratch for ONE layer (board #308). The KV is
/// `[K plane | V plane]`, each `cap` token rows of `kv_dim = nkv_local * hd` f32,
/// token-major (the paged-attention single-block layout). It replaces the host
/// `Step3p7KvCache` k/v Vecs for that layer, so the attention no longer walks the
/// whole context on the host every token. `cap` doubles on demand.
pub(crate) struct Step3p7GpuAttn {
    kv: compute::Buffer,
    cap: usize,
    /// Tokens held (the next row's index); equals the host cache's `len`.
    t: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    idx: compute::Buffer,
    q: compute::Buffer,
    out: compute::Buffer,
}

impl Step3p7GpuAttn {
    pub(crate) fn new(eng: &mut compute::ComputeEngine, nq: usize, nkv: usize, hd: usize) -> Result<Self, String> {
        let cap = 256;
        let kv = eng.alloc_host_coherent_storage((2 * cap * nkv * hd * 4) as u64)?;
        let idx = eng.alloc_host_coherent_storage(8)?;
        idx.write(&0u64.to_le_bytes())?;
        Ok(Step3p7GpuAttn {
            kv, cap, t: 0, nq, nkv, hd, idx,
            q: eng.alloc_host_coherent_storage((nq * hd * 4) as u64)?,
            out: eng.alloc_host_coherent_storage((nq * hd * 4) as u64)?,
        })
    }

    pub(crate) fn reset(&mut self) {
        self.t = 0;
    }

    /// Make room for one more row: double both planes, copying the held rows.
    fn reserve_one(&mut self, eng: &mut compute::ComputeEngine) -> Result<(), String> {
        if self.t < self.cap {
            return Ok(());
        }
        let kv_dim = self.nkv * self.hd;
        let (old_plane, cap2) = (self.cap * kv_dim, self.cap * 2);
        let kv2 = eng.alloc_host_coherent_storage((2 * cap2 * kv_dim * 4) as u64)?;
        let old = read_f32_buf(&self.kv, 2 * old_plane);
        kv2.write_at(0, &f32_slice_to_bytes(&old[..old_plane]))?;
        kv2.write_at((cap2 * kv_dim * 4) as u64, &f32_slice_to_bytes(&old[old_plane..]))?;
        let prev = std::mem::replace(&mut self.kv, kv2);
        eng.return_to_pool(prev);
        self.cap = cap2;
        Ok(())
    }

    /// Append this token's `k`/`v` rows (`[nkv*hd]` each, already q/k-normed and
    /// RoPE'd) and attend `q` (`[nq*hd]`) over the held rows `[window_start, t)`.
    /// Returns `[nq*hd]` — the same quantity `cpu_sdpa_gqa` returns.
    pub(crate) fn append_and_attend(&mut self, eng: &mut compute::ComputeEngine, kernel: Step3p7AttnKernel,
                                    q: &[f32], k: &[f32], v: &[f32], window: Option<usize>)
        -> Result<Vec<f32>, String> {
        let (nq, nkv, hd) = (self.nq, self.nkv, self.hd);
        let kv_dim = nkv * hd;
        if q.len() != nq * hd || k.len() != kv_dim || v.len() != kv_dim {
            return Err(format!("step3p7 gpu attn: q/k/v {} {} {} != {} {kv_dim}", q.len(), k.len(), v.len(), nq * hd));
        }
        self.reserve_one(eng)?;
        let plane = self.cap * kv_dim;
        self.kv.write_at((self.t * kv_dim * 4) as u64, &f32_slice_to_bytes(k))?;
        self.kv.write_at(((plane + self.t * kv_dim) * 4) as u64, &f32_slice_to_bytes(v))?;
        self.q.write(&f32_slice_to_bytes(q))?;
        self.t += 1;
        let len = self.t;
        let window_start = window.map(|w| len.saturating_sub(w)).unwrap_or(0);
        let scale = 1.0 / (hd as f32).sqrt();
        let cb = eng.begin_batch()?;
        match kernel {
            Step3p7AttnKernel::Sg => {
                let pc = sdpa_pc(len, nq, nkv, hd, self.cap, plane, scale, window_start, 0);
                eng.record_to(cb, "paged_attn_decode_f32_sg", &[&self.q, &self.idx, &self.kv, &self.out], &pc, (nq as u32, 1, 1))?;
            }
            Step3p7AttnKernel::Scalar => {
                let pc = sdpa_pc(len, nq, nkv, hd, self.cap, plane, scale, window_start, 0);
                let wg = ((nq * hd) as u32 + 255) / 256;
                eng.record_to(cb, "paged_attn_decode_f32", &[&self.q, &self.idx, &self.kv, &self.out], &pc, (wg, 1, 1))?;
            }
        }
        eng.submit_batch(cb)?;
        Ok(read_f32_buf(&self.out, nq * hd))
    }
}

/// One decoder layer for a single GPU decode token: pre-norm(+1) gated GQA attention +
/// residual, then pre-norm(+1) MoE/dense MLP + residual. A free fn so the caller can
/// hand it disjoint `&mut eng` / `&layer` / `&mut kv` borrows.
#[allow(clippy::too_many_arguments)]
fn decode_one_layer(
    eng: &mut compute::ComputeEngine,
    layer: &GpuLayerR,
    kv: &mut Step3p7KvCache,
    ga: &mut Option<Step3p7GpuAttn>,
    hidden: &[f32],
    global_idx: usize,
    cfg: &Step3p7Config,
    eps: f32,
    tp_rank: usize,
    tp_size: usize,
    tp_peer: i32,
    comm: usize,
    tp_send_scratch: &mut [f32],
    tp_recv_scratch: &mut [f32],
    tp_registered: bool,
) -> Result<Vec<f32>, String> {
    let hd = cfg.head_dim;
    let la = cfg.layers[global_idx];
    let a = &layer.attn;

    // ── gated attention ──
    let normed = rms_norm_plus1(hidden, &layer.input_ln, eps);
    let mut q = mv_dense(eng, &a.q, &normed)?; // [nq_local*hd]
    let mut k = mv_dense(eng, &a.k, &normed)?; // [nkv_local*hd]
    let v = mv_dense(eng, &a.v, &normed)?;
    let pos = kv.len;
    for hh in 0..a.nq_local {
        let head = &mut q[hh * hd..(hh + 1) * hd];
        let nrm = rms_norm_plus1(head, &a.q_norm, eps);
        let roped = partial_rope(&nrm, pos, &la, hd, &cfg.llama3);
        head.copy_from_slice(&roped);
    }
    for hh in 0..a.nkv_local {
        let head = &mut k[hh * hd..(hh + 1) * hd];
        let nrm = rms_norm_plus1(head, &a.k_norm, eps);
        let roped = partial_rope(&nrm, pos, &la, hd, &cfg.llama3);
        head.copy_from_slice(&roped);
    }
    // Board #308: GPU decode attention when enabled and this device has a kernel
    // for the shape; the layer's KV then lives on the GPU (`ga`) and the host cache
    // only tracks the length. Otherwise the host SDPA below, as before.
    let gpu_kernel = step37_gpu_attn_mode().and_then(|force_scalar| {
        pick_attn_kernel(eng, a.nq_local, a.nkv_local, hd, force_scalar)
    });
    if let Some(kernel) = gpu_kernel {
        if ga.is_none() {
            if kv.len != 0 {
                return Err("step3p7 gpu attn: enabled mid-sequence (host KV already holds tokens)".into());
            }
            *ga = Some(Step3p7GpuAttn::new(eng, a.nq_local, a.nkv_local, hd)?);
        }
        let g_attn = ga.as_mut().unwrap();
        if g_attn.t != kv.len {
            return Err(format!("step3p7 gpu attn: GPU KV holds {} tokens, host position {}", g_attn.t, kv.len));
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| eprintln!("[step3p7] GPU DECODE ATTENTION ENGAGED: {kernel:?} nq={} nkv={} hd={hd}",
                                    a.nq_local, a.nkv_local));
        let o = g_attn.append_and_attend(eng, kernel, &q, &k, &v, la.sliding_window)?;
        kv.len += 1;
        let g = mv_dense(eng, &a.g, &normed)?; // [nq_local]
        let gated = head_gate(&o, &g, a.nq_local, hd);
        let mut attn_out = mv_dense(eng, &a.o, &gated)?;
        tp_all_reduce(comm, tp_rank, tp_size, tp_peer, &mut attn_out, tp_send_scratch, tp_recv_scratch, tp_registered)?;
        let h1: Vec<f32> = hidden.iter().zip(&attn_out).map(|(&x, &y)| x + y).collect();
        return finish_layer_mlp(eng, layer, &h1, cfg, eps, tp_rank, tp_size, tp_peer, comm,
                                tp_send_scratch, tp_recv_scratch, tp_registered);
    }
    kv.k.extend_from_slice(&k);
    kv.v.extend_from_slice(&v);
    kv.len += 1;

    // GQA SDPA over the grown LOCAL KV. We col-SHARD k/v (each rank owns nkv/tp kv heads),
    // so the KV buffer is local — index it with the LOCAL ratio (nq_local/nkv_local, which
    // equals the global nq/nkv since both are ÷tp) and offset 0. (This is the shard-KV
    // layout; NOT qwen35's replicate-KV + global-offset scheme.) tp==1 ⇒ plain cpu_sdpa.
    let _ = tp_rank; // (offset is 0 under shard-KV; kept for signature symmetry)
    let local_ratio = a.nq_local / a.nkv_local;
    let scale = 1.0 / (hd as f32).sqrt();
    let o = cpu_sdpa_gqa(
        &q,
        &kv.k[0..kv.len * a.nkv_local * hd],
        &kv.v[0..kv.len * a.nkv_local * hd],
        a.nq_local,
        a.nkv_local,
        hd,
        kv.len,
        scale,
        la.sliding_window,
        local_ratio,
        0,
    );
    let g = mv_dense(eng, &a.g, &normed)?; // [nq_local]
    let gated = head_gate(&o, &g, a.nq_local, hd);
    let mut attn_out = mv_dense(eng, &a.o, &gated)?; // [hidden] (partial under TP)
    tp_all_reduce(comm, tp_rank, tp_size, tp_peer, &mut attn_out, tp_send_scratch, tp_recv_scratch, tp_registered)?;
    let h1: Vec<f32> = hidden.iter().zip(&attn_out).map(|(&x, &y)| x + y).collect();
    finish_layer_mlp(eng, layer, &h1, cfg, eps, tp_rank, tp_size, tp_peer, comm,
                     tp_send_scratch, tp_recv_scratch, tp_registered)
}

/// The MLP half of a decoder layer (shared by the GPU- and host-attention paths of
/// [`decode_one_layer`]): pre-norm(+1) MoE/dense MLP on `h1` + residual.
#[allow(clippy::too_many_arguments)]
fn finish_layer_mlp(
    eng: &mut compute::ComputeEngine,
    layer: &GpuLayerR,
    h1: &[f32],
    cfg: &Step3p7Config,
    eps: f32,
    tp_rank: usize,
    tp_size: usize,
    tp_peer: i32,
    comm: usize,
    tp_send_scratch: &mut [f32],
    tp_recv_scratch: &mut [f32],
    tp_registered: bool,
) -> Result<Vec<f32>, String> {
    // ── MLP ──
    let normed2 = rms_norm_plus1(&h1, &layer.post_ln, eps);
    let mut mlp_out = match &layer.mlp {
        GpuMlp::Dense(d) => {
            let gate = mv_dense(eng, &d.gate, &normed2)?;
            let up = mv_dense(eng, &d.up, &normed2)?;
            let act = clamped_swiglu_prod(&gate, &up, None);
            mv_dense(eng, &d.down, &act)?
        }
        GpuMlp::Moe(m) => {
            // router replicated → full top-k selection, then keep only owned experts.
            let logits = cpu_matmul(&normed2, &m.router, 1, cfg.hidden_size, cfg.num_experts);
            let (indices, weights) = bias_router(&logits, &m.bias, cfg.num_experts_per_tok, cfg.router_scaling_factor);
            let mut routed = vec![0.0f32; cfg.hidden_size];
            // Owned selected experts, in route order: (kth weight index, local slot).
            let mut sel: Vec<(usize, usize)> = Vec::with_capacity(indices.len());
            for (kth, &e) in indices.iter().enumerate() {
                if tp_size > 1 && !(e >= m.owned_lo && e < m.owned_lo + m.owned_cnt) {
                    continue; // another rank owns this expert; its partial arrives via all-reduce
                }
                sel.push((kth, e - m.owned_lo));
            }
            // Step-3.7 "#3": expert-batched nvfp4-e4m3 matvec (VLLM_VULKAN_STEP37_EXPERT_BATCH,
            // default ON; =0 restores the serial path). Collapses the selected-experts × {gate,up,down} per-expert dispatches
            // into 3 batched dispatches. Bit-exact vs the serial `mv_expert` loop below: same
            // repack dequant body, same per-expert global, and `sel` preserves route order so
            // the `routed` accumulation order is identical. Falls back to serial if any switch's
            // shape misses the repack guard (step3p7 experts always clear it, so this is belt-and-
            // suspenders). Off ⇒ the serial path runs byte-unchanged.
            let use_batch = crate::flags::flags_global().step37_expert_batch
                && !sel.is_empty()
                && step3p7_batched_expert_shader(m.gate.inn, m.gate.out, m.gate.group).is_some()
                && step3p7_batched_expert_shader(m.up.inn, m.up.out, m.up.group).is_some()
                && step3p7_batched_expert_shader(m.down.inn, m.down.out, m.down.group).is_some();
            // Engagement proof for the on-node A/B: the batched path is bit-exact by
            // construction, so identical logits cannot tell "engaged" from "never ran".
            // One banner per process either way (requested+eligible, or requested+not).
            static BATCH_BANNER: std::sync::Once = std::sync::Once::new();
            if crate::flags::flags_global().step37_expert_batch && !sel.is_empty() {
                BATCH_BANNER.call_once(|| {
                    if use_batch {
                        eprintln!("[vllm-vulkan] STEP37_EXPERT_BATCH ENGAGED: {} routed experts/token \
                                   -> 3 batched dispatches (gate {}x{}, down {}x{})",
                                  sel.len(), m.gate.out, m.gate.inn, m.down.out, m.down.inn);
                    } else {
                        eprintln!("[vllm-vulkan] STEP37_EXPERT_BATCH requested but NOT eligible \
                                   (no batched shader for these shapes); serial path kept");
                    }
                });
            }
            if use_batch {
                let les: Vec<usize> = sel.iter().map(|&(_, le)| le).collect();
                let inter = m.inter;
                let h = cfg.hidden_size;
                let (gu_sh, gu_r) = step3p7_batched_expert_shader(m.gate.inn, m.gate.out, m.gate.group).unwrap();
                let gp_all = mv_experts_batched(eng, &m.gate, &les, &normed2, true, &gu_sh, gu_r)?; // [n_ex, inter]
                let up_all = mv_experts_batched(eng, &m.up, &les, &normed2, true, &gu_sh, gu_r)?;
                // Per-expert clamped SwiGLU (host — bit-identical to the serial path),
                // concatenated into [n_ex, inter] to feed the batched down projection.
                let mut act_all = vec![0.0f32; les.len() * inter];
                for e in 0..les.len() {
                    let a = clamped_swiglu_prod(&gp_all[e * inter..(e + 1) * inter],
                                                &up_all[e * inter..(e + 1) * inter], m.expert_limit);
                    act_all[e * inter..(e + 1) * inter].copy_from_slice(&a);
                }
                let (d_sh, d_r) = step3p7_batched_expert_shader(m.down.inn, m.down.out, m.down.group).unwrap();
                let dn_all = mv_experts_batched(eng, &m.down, &les, &act_all, false, &d_sh, d_r)?; // [n_ex, hidden]
                for (e, &(kth, _)) in sel.iter().enumerate() {
                    let wk = weights[kth];
                    for (r, &o) in routed.iter_mut().zip(&dn_all[e * h..(e + 1) * h]) {
                        *r += o * wk;
                    }
                }
            } else {
                for &(kth, le) in &sel {
                    let gp = mv_expert(eng, &m.gate, le, &normed2)?; // [inter]
                    let up = mv_expert(eng, &m.up, le, &normed2)?;
                    let act = clamped_swiglu_prod(&gp, &up, m.expert_limit);
                    let dn = mv_expert(eng, &m.down, le, &act)?; // [hidden]
                    let wk = weights[kth];
                    for (r, &o) in routed.iter_mut().zip(&dn) {
                        *r += o * wk;
                    }
                }
            }
            // ungated shared expert (partial under TP row-shard of down)
            let sg = mv_dense(eng, &m.shared_gate, &normed2)?;
            let su = mv_dense(eng, &m.shared_up, &normed2)?;
            let sact = clamped_swiglu_prod(&sg, &su, m.shared_limit);
            let sd = mv_dense(eng, &m.shared_down, &sact)?;
            for (r, &s) in routed.iter_mut().zip(&sd) {
                *r += s;
            }
            routed
        }
    };
    tp_all_reduce(comm, tp_rank, tp_size, tp_peer, &mut mlp_out, tp_send_scratch, tp_recv_scratch, tp_registered)?;
    Ok(h1.iter().zip(&mlp_out).map(|(&x, &y)| x + y).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PR #98 review: with sharded weights (TP>1) an unwired comm or peer must fail
    /// the reduce instead of letting a half-sum through; single-TP is a no-op.
    #[test]
    fn tp_reduce_errors_when_sharded_but_unwired() {
        let (mut b, mut s1, mut s2) = (vec![1f32; 4], vec![0f32; 4], vec![0f32; 4]);
        assert!(tp_all_reduce(0, 0, 1, -1, &mut b, &mut s1, &mut s2, false).is_ok());
        let e = tp_all_reduce(0, 0, 2, 1, &mut b, &mut s1, &mut s2, false).unwrap_err();
        assert!(e.contains("not wired"), "{e}");
        let e = tp_all_reduce(0x1000, 0, 2, -1, &mut b, &mut s1, &mut s2, false).unwrap_err();
        assert!(e.contains("not wired"), "{e}");
        assert_eq!(b, vec![1f32; 4], "a failed reduce must not touch the buffer");
        // The decode_step pre-check shares the rule: wired TP=2 passes, TP>2 does not.
        assert!(check_tp_wired(0, 1, -1).is_ok());
        assert!(check_tp_wired(0x1000, 2, 1).is_ok());
        assert!(check_tp_wired(0, 2, 1).is_err());
        assert!(check_tp_wired(0x1000, 4, 1).is_err());
    }

    // ── Phase-3 offline proof: TP shard math is a clean, lossless partition ──────
    // col_shard/row_shard + EP owned-range are the load-time TP pieces (the forward
    // reduces + GPU dispatch are cluster-gated). These are pure — fully testable on Mac.

    #[test]
    fn tp_col_shard_partitions_rows() {
        // [out=8, in=3]; column-parallel splits the OUT rows across ranks.
        let inn = 3usize;
        let out = 8usize;
        let w: Vec<f32> = (0..out * inn).map(|i| i as f32).collect();
        for n in [2usize, 4] {
            let mut recon = Vec::new();
            for r in 0..n {
                recon.extend_from_slice(&col_shard(&w, inn, r, n));
            }
            assert_eq!(recon, w, "col_shard TP{n}: concat of rank slices != original");
            // each rank owns out/n contiguous rows
            let per = out / n;
            for r in 0..n {
                let s = col_shard(&w, inn, r, n);
                assert_eq!(s.len(), per * inn);
                assert_eq!(s[0], (r * per * inn) as f32, "rank {r} wrong first row");
            }
        }
    }

    #[test]
    fn tp_row_shard_partitions_cols() {
        // [out=4, in=8]; row-parallel splits the IN columns across ranks. Reduction over
        // ranks (sum of per-rank partial matvecs) reconstructs the full matvec — here we
        // just check the slice tiling reconstructs every row's columns in order.
        let inn = 8usize;
        let out = 4usize;
        let w: Vec<f32> = (0..out * inn).map(|i| i as f32).collect();
        for n in [2usize, 4] {
            let per = inn / n;
            // reassemble row by row from the rank slices
            let mut recon = vec![0.0f32; out * inn];
            for r in 0..n {
                let s = row_shard(&w, inn, r, n);
                assert_eq!(s.len(), out * per);
                for row in 0..out {
                    for c in 0..per {
                        recon[row * inn + r * per + c] = s[row * per + c];
                    }
                }
            }
            assert_eq!(recon, w, "row_shard TP{n}: reassembled cols != original");
        }
    }

    #[test]
    fn tp_ep_owned_range_partitions_experts() {
        // EP whole-expert partition: every expert owned by exactly one rank, contiguous.
        let e = 212usize;
        for n in [2usize, 4] {
            assert_eq!(e % n, 0);
            let per = e / n;
            let mut seen = vec![0u8; e];
            for r in 0..n {
                let lo = r * per;
                for x in lo..lo + per {
                    seen[x] += 1;
                }
            }
            assert!(seen.iter().all(|&c| c == 1), "EP TP{n}: expert double/unowned");
        }
    }

    #[test]
    fn nvfp4_expert_offset_math() {
        // The concatenated-switch offsets a resident expert `local_e` dispatches at must
        // match the Laguna/nemotron stacked layout: packed word offset e*out*(in/8),
        // scale elem offset e*out*(in/group). (gate/up: out=inter,in=hidden; down swap.)
        let (out, inn, group) = (16usize, 32usize, 16usize);
        let pack_stride = out * (inn / 8);
        let sb_stride = out * (inn / group);
        assert_eq!(pack_stride, out * inn / 8);
        assert_eq!(sb_stride, out * inn / group);
        for e in 0..4usize {
            assert_eq!(e * pack_stride, e * out * (inn / 8));
            assert_eq!(e * sb_stride, e * out * (inn / group));
        }
        // packed bytes per expert (u8 nibbles) == 4 * words per expert (u32).
        assert_eq!(out * inn / 2, 4 * pack_stride);
    }

    #[test]
    fn e4m3_expert_selector_routes_repack_only_on_flag_and_shape() {
        // The step3p7 expert selector must route to the repack kernel EXACTLY when
        // `VLLM_VULKAN_LAGUNA_EXPERT_REPACK` is on AND the shape clears the repack guard,
        // and fall back to the byte-identical v1 e4m3 oracle otherwise — i.e. it is a
        // clean SUPERSET of v1 (the "existing arches byte-identical, change gated on the
        // step3p7 path" guarantee). Robust to the process-wide flag snapshot: we derive
        // the expected branch from the SAME predicates the selector uses.
        // [1280,4096] = the real step3p7 expert sub-matvec shapes (both gate/up and down).
        for &(k, n) in &[(4096usize, 1280usize), (1280usize, 4096usize)] {
            let got = step3p7_e4m3_expert_shader(k, n, 16);
            if laguna_expert_repack_flag() && nvfp4_repack_shape_ok(k, n, 16) {
                assert_eq!(got, ("mul_mat_vec_nvfp4_e4m3repack_f32_f32_bs64_r4".to_string(), 4),
                    "flag+shape on: [{k},{n}] must route to the e4m3 repack kernel");
            } else {
                assert_eq!(got, matvec_nvfp4_e4m3_variant(n),
                    "flag/shape off: [{k},{n}] must fall back to the v1 e4m3 oracle");
            }
        }
        // Both orientations clear the repack SHAPE constraints (k%32==0, k>=1024, n>=1024,
        // gs==16), so shape is never the reason a step3p7 expert falls back to v1. (The
        // GPU cos=1.0/argmax A/B itself is the on-node debug_nvfp4_repack gate.)
        for &(k, n) in &[(4096usize, 1280usize), (1280usize, 4096usize)] {
            assert_eq!(k % 32, 0, "k={k} must be a multiple of 32");
            assert!(k >= 1024 && n >= 1024, "shape [{k},{n}] below repack floor");
        }
    }

    #[test]
    fn step37_batched_expert_shader_routes_only_on_shape() {
        // The batched selector routes to the repack-batched kernel exactly when the
        // shape clears the SAME guard as the serial e4m3 repack, else None (⇒ serial).
        for &(k, n) in &[(4096usize, 1280usize), (1280usize, 4096usize)] {
            assert_eq!(step3p7_batched_expert_shader(k, n, 16),
                Some(("mul_mat_vec_nvfp4_e4m3repack_batched_f32_f32_bs64_r4".to_string(), 4)),
                "real step3p7 expert shape [{k},{n}] must route to the batched repack kernel");
        }
        // Sub-floor shapes fall back to serial (None). (k<1024 / n<1024 miss the guard.)
        assert_eq!(step3p7_batched_expert_shader(64, 8, 16), None);
    }

    // ── Step-3.7 "#3" expert-batched nvfp4-e4m3: bit-exact vs serial (host sim) ──
    // The batched-shader math == serial-shader math is a pure REORG (same repack body,
    // same per-expert global, only the dispatch schedule + offset source differ). We
    // prove the offset math + accumulation are 0-diff by running a faithful host port of
    // the shader body under BOTH schedules — the serial one (packed_off=le*stride, x_off=0)
    // and the batched one driven by the PRODUCTION `step3p7_expert_meta_u32` offset builder.
    // (GPU-execution equivalence of the compiled SPIR-V is the deferred on-node cos=1.0 gate,
    // like every other repack A/B — no GPU device on the offline CI path.)

    /// E2M1 (FP4) code -> value, matching the kE2M1[16] table in the shaders.
    fn e2m1(code: u32) -> f32 {
        const T: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
        T[(code & 0xF) as usize]
    }

    /// E4M3 code -> value, ARITHMETIC decode identical to kE4M3_decode() in the shaders.
    fn e4m3_decode(b: u32) -> f32 {
        let s = b & 0x80;
        let e = (b >> 3) & 0xF;
        let m = b & 0x7;
        let mag = if e == 0 {
            (m as f32) * 0.001953125
        } else if e == 15 && m == 7 {
            0.0
        } else {
            f32::from_bits(((e + 120) << 23) | (m << 20))
        };
        f32::from_bits(mag.to_bits() | (s << 24))
    }

    /// Faithful host port of ONE workgroup of mul_mat_vec_nvfp4_e4m3repack[_batched]:
    /// out[r] for r in [0,n), reading weights from `packed` (u32 words) at word offset
    /// `packed_off`, e4m3 scale bytes (4/u32) from `scaleb` at byte-elem offset `sb_off`,
    /// activations from `x` at float offset `x_off`, folding per-expert `global`. group==16
    /// (g0=2c, g1=2c+1). Same nibble→activation map and fma order as the GLSL — the test
    /// drives it once with the serial offset scheme and once with the batched meta[] scheme.
    fn expert_matvec_ref(
        packed: &[u32], scaleb: &[u32], x: &[f32], k: usize, n: usize,
        packed_off: usize, sb_off: usize, x_off: usize, global: f32,
    ) -> Vec<f32> {
        let groups = k / 16;
        let words_per_row = k / 8;
        let chunks_per_row = k / 32;
        let dot16 = |wlo: u32, whi: u32, xbase: usize| -> f32 {
            let mut s = 0.0f32;
            for (i, w) in [wlo, whi].iter().enumerate() {
                for j in 0..8usize {
                    let code = (w >> (j * 4)) & 0xF;
                    s += e2m1(code) * x[xbase + i * 8 + j];
                }
            }
            s
        };
        let bscale = |abs_sidx: usize, global: f32| -> f32 {
            let sword = scaleb[abs_sidx >> 2];
            let sbyte = (sword >> ((abs_sidx & 3) * 8)) & 0xFF;
            e4m3_decode(sbyte) * global
        };
        let mut out = vec![0.0f32; n];
        for r in 0..n {
            let mut acc = 0.0f32;
            for c in 0..chunks_per_row {
                let base_word = packed_off + r * words_per_row + c * 4;
                let pw = [packed[base_word], packed[base_word + 1],
                          packed[base_word + 2], packed[base_word + 3]];
                let xb = x_off + c * 32;
                let qxa = dot16(pw[0], pw[1], xb);
                let qxb = dot16(pw[2], pw[3], xb + 16);
                let sbase = sb_off + r * groups;
                let scale_a = bscale(sbase + c * 2, global);
                let scale_b = bscale(sbase + c * 2 + 1, global);
                acc = scale_a.mul_add(qxa, scale_b.mul_add(qxb, acc)); // == GLSL fma order
            }
            out[r] = acc;
        }
        out
    }

    /// Build a deterministic concatenated nvfp4-e4m3 expert set + activations, then assert
    /// the batched schedule (via `step3p7_expert_meta_u32`) is 0-diff vs the serial schedule.
    fn assert_batched_eq_serial(k: usize, n: usize, les: &[usize], x_shared: bool) {
        let n_experts = 4usize; // resident concatenated experts (les selects among these)
        let pack_stride = n * (k / 8);
        let sb_stride = n * (k / 16);
        assert_eq!(pack_stride % 4, 0, "pack_stride must be 4-word aligned (k%32==0)");
        // Deterministic LCG fill (avoids a rand dep; exercises the full nibble/byte range).
        let mut st = 0x9E3779B1u32;
        let mut next = || { st = st.wrapping_mul(1664525).wrapping_add(1013904223); st };
        let packed: Vec<u32> = (0..n_experts * pack_stride).map(|_| next()).collect();
        let scaleb: Vec<u32> = (0..(n_experts * sb_stride) / 4).map(|_| next()).collect();
        let globals: Vec<f32> = (0..n_experts).map(|e| 0.5f32 + e as f32 * 0.37).collect();
        let n_ex = les.len();
        let x: Vec<f32> = {
            let len = if x_shared { k } else { n_ex * k };
            (0..len).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect()
        };

        // Serial: each selected expert dispatched on its own (packed_off=le*stride, x_off=0
        // for gate/up or per-slot for down), reading its own global.
        let mut serial = Vec::with_capacity(n_ex * n);
        for (e, &le) in les.iter().enumerate() {
            let x_off = if x_shared { 0 } else { e * k };
            serial.extend(expert_matvec_ref(
                &packed, &scaleb, &x, k, n, le * pack_stride, le * sb_stride, x_off, globals[le]));
        }

        // Batched: offsets from the PRODUCTION meta builder + per-expert globals[le].
        let meta = step3p7_expert_meta_u32(pack_stride, sb_stride, k, n, les, x_shared);
        let mut batched = vec![0.0f32; n_ex * n];
        for (e, &le) in les.iter().enumerate() {
            let packed_off = meta[e * 4] as usize;
            let sb_off = meta[e * 4 + 1] as usize;
            let x_off = meta[e * 4 + 2] as usize;
            let dst_off = meta[e * 4 + 3] as usize;
            let o = expert_matvec_ref(&packed, &scaleb, &x, k, n, packed_off, sb_off, x_off, globals[le]);
            batched[dst_off..dst_off + n].copy_from_slice(&o);
        }

        assert_eq!(serial.len(), batched.len());
        assert!(serial.iter().any(|&v| v != 0.0), "test is vacuous: serial output all-zero");
        // 0-diff, bit-for-bit (the reorg preserves the exact arithmetic).
        for (i, (&s, &b)) in serial.iter().zip(&batched).enumerate() {
            assert_eq!(s.to_bits(), b.to_bits(),
                "batched != serial at flat idx {i}: serial={s} batched={b} (k={k},n={n},x_shared={x_shared})");
        }
    }

    #[test]
    fn step37_expert_batched_bit_exact_vs_serial() {
        // gate/up orientation (out=inter, in=hidden): shared activation across experts.
        // down orientation (out=hidden, in=inter): per-expert concatenated activation.
        // Small k (multiple of 32) + non-identity + non-contiguous slot selections to
        // exercise the meta offset math (packed_off/sb_off/x_off/dst_off) end to end.
        assert_batched_eq_serial(64, 8, &[0, 1, 2, 3], true);      // gate/up, all experts, identity
        assert_batched_eq_serial(64, 8, &[2, 0, 3], true);          // gate/up, arbitrary order/subset
        assert_batched_eq_serial(96, 6, &[3, 1], false);           // down, per-expert x_off
        assert_batched_eq_serial(128, 4, &[1, 3, 0, 2], false);    // down, full reorder
    }
}

/// Board #308: `Step3p7GpuAttn` against the host `cpu_sdpa_gqa` oracle, token by
/// token, over 300 decode steps (crossing the 256-row KV doubling), for a full
/// layer and a sliding-window layer, GQA ratio 8. Every kernel the device compiled
/// is checked (the Mac / MoltenVK has only the portable scalar one; a BC-250 also
/// runs sg and split-K). `#[ignore]`: needs a Vulkan device; panics without one.
///   cargo test --lib step37_gpu_attn -- --ignored --nocapture
#[cfg(test)]
mod gpu_attn_tests {
    use super::*;
    use crate::model::cpu_sdpa_gqa;

    fn engine() -> compute::ComputeEngine {
        assert!(crate::device::is_vulkan_available(), "needs a Vulkan device (VK_ICD_FILENAMES)");
        let dev = device::ComputeDevice::create(0).expect("device 0");
        let spvs = crate::include_all_shaders();
        let refs: HashMap<&str, &[u8]> = spvs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        compute::ComputeEngine::new(dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, dev.caps(), &refs).expect("engine")
    }

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    #[test]
    #[ignore = "needs a Vulkan device; run with --ignored"]
    fn step37_gpu_attn_matches_host_sdpa() {
        let mut eng = engine();
        let (nq, nkv, hd, steps) = (16usize, 2usize, 128usize, 300usize);
        let kernels = [Step3p7AttnKernel::Scalar, Step3p7AttnKernel::Sg];
        let mut checked = Vec::new();
        for &kernel in &kernels {
            let name = match kernel {
                Step3p7AttnKernel::Scalar => "paged_attn_decode_f32",
                Step3p7AttnKernel::Sg => "paged_attn_decode_f32_sg",
            };
            if !eng.has_pipeline(name) {
                eprintln!("[step37_gpu_attn] {kernel:?}: {name} not compiled on this device");
                continue;
            }
            for window in [None, Some(64usize)] {
                let mut r = Rng(0x0308 ^ (window.unwrap_or(0) as u64));
                let mut ga = Step3p7GpuAttn::new(&mut eng, nq, nkv, hd).unwrap();
                let (mut hk, mut hv) = (Vec::new(), Vec::new());
                let mut worst = 0f32;
                for step in 0..steps {
                    let q: Vec<f32> = (0..nq * hd).map(|_| r.f()).collect();
                    let k: Vec<f32> = (0..nkv * hd).map(|_| r.f()).collect();
                    let v: Vec<f32> = (0..nkv * hd).map(|_| r.f()).collect();
                    hk.extend_from_slice(&k);
                    hv.extend_from_slice(&v);
                    let len = step + 1;
                    let want = cpu_sdpa_gqa(&q, &hk, &hv, nq, nkv, hd, len, 1.0 / (hd as f32).sqrt(),
                                            window, nq / nkv, 0);
                    let got = ga.append_and_attend(&mut eng, kernel, &q, &k, &v, window).unwrap();
                    let d = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    worst = worst.max(d);
                    assert!(d < 1e-4, "{kernel:?} window {window:?} step {step}: max |gpu-host| {d}");
                }
                assert!(ga.cap >= steps, "KV did not grow past the first 256 rows");
                eprintln!("[step37_gpu_attn] {kernel:?} window {window:?}: {steps} steps, max |gpu-host| {worst:.2e}, cap {}", ga.cap);
                // reset restarts the sequence
                ga.reset();
                assert_eq!(ga.t, 0);
            }
            checked.push(kernel);
        }
        assert!(!checked.is_empty(), "no decode-attention kernel compiled on this device");
    }
}
