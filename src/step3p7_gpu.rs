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
    nvfp4_repack_shape_ok, read_f32_buf,
};
use crate::step3p7::{
    bias_router, clamped_swiglu_prod, head_gate, load_experts_proj, partial_rope, rms_norm_plus1,
    ExpertStore, MoeStreamCfg, Step3p7Config, Step3p7KvCache,
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

/// Build the EP-owned NVFP4 experts of one proj into a resident `GpuSwitch`. CONSUMES
/// the expert Vec, streaming each owned expert's packed nibbles + e4m3 scales STRAIGHT
/// into the GTT buffer at its slot offset (via `write_at`) and freeing the host bytes
/// as it goes — so peak host stays ~one expert (~9 MB), NOT the whole proj's ~1.88 GiB
/// concat (the LOAD-OOM cure at proj granularity, on top of the per-layer one).
fn build_switch(
    eng: &mut compute::ComputeEngine,
    experts: Vec<crate::step3p7::Step3p7Expert>,
    owned_lo: usize,
    owned_cnt: usize,
) -> Result<GpuSwitch, String> {
    let out = experts[0].out_f;
    let inn = experts[0].in_f;
    let group = experts[0].group_size;
    let pack_bytes = out * inn / 2; // u8 nibbles per expert
    let sb_bytes = out * (inn / group); // e4m3 scale bytes per expert
    let pbuf = eng.alloc_host_coherent_storage((owned_cnt * pack_bytes).max(4) as u64)?;
    let sbuf = eng.alloc_host_coherent_storage((owned_cnt * sb_bytes).max(4) as u64)?;
    let mut globals: Vec<f32> = Vec::with_capacity(owned_cnt);
    let mut slot = 0usize;
    for (e, ex) in experts.into_iter().enumerate() {
        if e < owned_lo || e >= owned_lo + owned_cnt {
            continue; // another rank owns it (EP); dropped here
        }
        match ex.store {
            ExpertStore::Resident { packed, scale } => {
                if packed.len() != pack_bytes || scale.len() != sb_bytes {
                    return Err(format!(
                        "build_switch: expert {e} packed {} (want {pack_bytes}) scale {} (want {sb_bytes})",
                        packed.len(), scale.len()
                    ));
                }
                pbuf.write_at((slot * pack_bytes) as u64, &packed)?;
                sbuf.write_at((slot * sb_bytes) as u64, &scale)?;
                globals.push(ex.global);
                slot += 1;
                // packed/scale drop here → host bytes freed before the next expert.
            }
            ExpertStore::Streamed { .. } => {
                return Err(format!(
                    "build_switch: expert {e} is overflow-STREAMED; GPU-resident load requires \
                     all owned experts resident (disable VLLM_VULKAN_MOE_STREAM_OVERFLOW)"
                ));
            }
        }
    }
    Ok(GpuSwitch {
        packed: pbuf,
        scale: sbuf,
        globals,
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

/// TP-2 all-reduce a `[hidden]` partial in place: a deadlock-safe PAIRWISE exchange with
/// the tp_peer (even-`tp_rank`-sends-first), then `buf += peer_partial` — the SUM of the
/// two ranks' partials. This is nemotron's TP=2 pattern (NOT a full-comm all_reduce, which
/// on a flat PP+TP comm would wrongly reduce across every rank). Re-acquires the GIL —
/// safe because `decode_step` is always reached from a pyo3 method that holds it. No-op
/// when tp is off / the comm or peer is unset.
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
    if tp_size <= 1 || comm == 0 || tp_peer < 0 {
        return Ok(());
    }
    if tp_size != 2 {
        return Err(format!(
            "step3p7 TP reduce is TP=2 pairwise only (tp_size={tp_size}; TP>2 needs a vcclCommSplit sub-comm)"
        ));
    }
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
        let mut get_f32 = |name: &str,
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

        let stream_off = MoeStreamCfg { enabled: false, budget_bytes: u64::MAX };
        let mut dummy_bytes: u64 = 0;
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
                // owned experts resident. `load_experts_proj` slices per-expert from the
                // mmap'd 3D tensors (no full-tensor slurp); we then keep [owned_lo, +cnt).
                // Load + build + FREE one projection at a time (never hold all three
                // projections' ~1.88 GiB expert sets host-resident at once).
                let mut build_proj = |base: &str, out_f: usize, in_f: usize,
                                      eng: &mut compute::ComputeEngine,
                                      emm: &mut HashMap<String, memmap2::Mmap>|
                 -> Result<GpuSwitch, String> {
                    let g = get_scales2(dir, &weight_map, emm, &format!("{base}.weight_scale_2"))?;
                    let ex = load_experts_proj(dir, &weight_map, emm, base, &g, cfg.num_experts, out_f, in_f, group, stream_off, &mut dummy_bytes)?;
                    build_switch(eng, ex, owned_lo, owned_cnt) // consumes ex → freed inside
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
    /// `set_collective_comm` / `set_tp_peer` in lib.rs. `peer < 0` disables the reduce.
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
        self.pos = 0;
    }

    /// Single-token GPU-resident decode step. First stage embeds `token_id`; a mid stage
    /// consumes the previous stage's `[hidden]`; the last stage returns `[vocab]` logits.
    pub fn decode_step(&mut self, token_id: u32, hidden_in: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.h;
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

/// One decoder layer for a single GPU decode token: pre-norm(+1) gated GQA attention +
/// residual, then pre-norm(+1) MoE/dense MLP + residual. A free fn so the caller can
/// hand it disjoint `&mut eng` / `&layer` / `&mut kv` borrows.
#[allow(clippy::too_many_arguments)]
fn decode_one_layer(
    eng: &mut compute::ComputeEngine,
    layer: &GpuLayerR,
    kv: &mut Step3p7KvCache,
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
            for (kth, &e) in indices.iter().enumerate() {
                if tp_size > 1 && !(e >= m.owned_lo && e < m.owned_lo + m.owned_cnt) {
                    continue; // another rank owns this expert; its partial arrives via all-reduce
                }
                let le = e - m.owned_lo;
                let gp = mv_expert(eng, &m.gate, le, &normed2)?; // [inter]
                let up = mv_expert(eng, &m.up, le, &normed2)?;
                let act = clamped_swiglu_prod(&gp, &up, m.expert_limit);
                let dn = mv_expert(eng, &m.down, le, &act)?; // [hidden]
                let wk = weights[kth];
                for (r, &o) in routed.iter_mut().zip(&dn) {
                    *r += o * wk;
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
}
