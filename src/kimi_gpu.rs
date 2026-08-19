// SPDX-License-Identifier: Apache-2.0
//! Kimi-Linear-48B-A3B GPU QUANT-RESIDENT per-PP-stage decode.
//!
//! Phase-C build: the resident analog of the Gate-1 streaming harness
//! (`debug_kimi_decode_gpu` in debug_api.rs). A `KimiGpuStage` owns a
//! `ComputeEngine` and uploads its PP window `[layer_start, layer_end)`'s PACKED
//! 4-bit KDA/MoE buffers ONCE (never `return_to_pool`'d — held resident), plus the
//! per-layer persistent GPU recurrence state (`kda_gdn_step` buffer + host conv
//! window + host MLA KV). The per-token step is the EXACT math of
//! `kimi_kda_decode_step_gpu` + `kimi_moe_gpu_combine`; the only change is reading
//! HELD buffers instead of re-streaming per layer per token.
//!
//! Lifetime note: `ComputeDevice` has no Drop and nothing calls `destroy_device`
//! (the VkDevice lives for the process), so resident `Buffer`s (which free their
//! own memory via their held `ash::Device` handle) have no drop-order hazard.
//!
//! Bit-exact oracle = the CPU resident decode (`kimi::KimiModel::decode_step`) and
//! the streaming Gate-1 (`debug_kimi_decode_gpu`): resident-held == streaming.

use crate::compute;
use crate::device;
use crate::model;
use crate::moe;
use crate::push_constants::*;
use crate::kimi::{self, KimiConfig, KimiLayerKind};

use ash::vk;

use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

const GROUP_SIZE: usize = 64;

/// A resident packed mlx4 matvec weight (held on the GPU across decode steps).
struct GpuMatR {
    p: compute::Buffer,
    s: compute::Buffer,
    b: compute::Buffer,
    k: usize,
    n: usize,
}

/// A resident packed 3D switch-expert tensor (all E experts, one MoE sub-proj).
struct GpuSwitchR {
    p: compute::Buffer,
    s: compute::Buffer,
    b: compute::Buffer,
    out_features: usize,
    in_features: usize,
}

/// Resident KDA attention (9 packed projections + host glue + persistent state).
struct KdaGpuR {
    q: GpuMatR,
    k: GpuMatR,
    v: GpuMatR,
    fa: GpuMatR,
    fb: GpuMatR,
    b: GpuMatR,
    ga: GpuMatR,
    gb: GpuMatR,
    o: GpuMatR,
    q_conv: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<f32>,
    nh: usize,
    hd: usize,
    kern: usize,
    /// Persistent recurrence state [nv,kd,vd], advanced in place by `kda_gdn_step`.
    g_state: compute::Buffer,
    /// Persistent host conv sliding-windows (q/k/v). Used by the OFF (host-seam)
    /// path; the fused path advances GPU-resident windows (`conv_state_*`).
    conv: kimi::kda::KdaState,
    // --- Lever #5 (fused-KDA) resident glue buffers. Held GPU-resident so the
    // depthwise conv / qk-RMSNorm / per-channel decay record into the KDA
    // command buffer instead of round-tripping to host. Built regardless of the
    // flag (tiny: ~0.35 MB/KDA-layer) so `reset_state` can zero them and the
    // flag can be honoured at call time.
    /// Depthwise conv taps [dim, kern] (== the host `*_conv` Vec, uploaded once).
    q_conv_buf: compute::Buffer,
    k_conv_buf: compute::Buffer,
    v_conv_buf: compute::Buffer,
    /// Persistent GPU conv sliding-windows [dim, kern-1] (channel-major; the
    /// GPU analog of `conv.conv_*`, advanced in place by `q35_dn_conv_step`).
    conv_state_q: compute::Buffer,
    conv_state_k: compute::Buffer,
    conv_state_v: compute::Buffer,
    /// Per-head A_log [nh] and per-channel dt_bias [proj] for the `kda_decay`
    /// shader (== the host `a_log`/`dt_bias` Vecs, uploaded once).
    a_log_buf: compute::Buffer,
    dt_bias_buf: compute::Buffer,
    /// `kda_gdn_step` params [2*nv | o_norm(vd)] (the leading 2*nv is unused —
    /// A_log/dt_bias are folded into the decay buffer). Built once.
    g_params: compute::Buffer,
}

/// Resident MLA attention (lever #2): the 4 projections held GPU-resident as
/// packed 4-bit `GpuMatR` + the host softmax-SDPA seam. `q_proj`
/// `[nh*(nope+pe), h]`, `kv_a_proj` (kv_a_proj_with_mqa) `[r+pe, h]`, `o_proj`
/// `[h, nh*v]` map 1:1 to `matvec_mlx4`. `kv_b_proj` is held in its NATURAL
/// on-disk `[nh*(nope+v), r]` layout: one matvec against the layer-normed KV
/// latent produces `[nh*(nope+v)]`, whose per-head `head_dim=nope+v` block
/// splits into `k_nope` (first `nope`) || `v` (last `v`) — the SAME decompress
/// the CPU `embed_q`/`unembed_out` loops do, WITHOUT the loader's `[r,nope]`
/// transpose (so no row-permute repack). `kv_a_layernorm` stays host f32.
struct MlaGpuR {
    q_proj: GpuMatR,
    kv_a_proj: GpuMatR,
    kv_b_proj: GpuMatR,
    o_proj: GpuMatR,
    kv_a_layernorm: Vec<f32>,
    h: usize,
    nh: usize,
    nope: usize,
    pe: usize,
    v: usize,
    r: usize,
    eps: f32,
}

/// MLA projection backend: GPU-resident (lever #2, flag ON) or the host-f32
/// `matmul_wt` oracle (`kimi::mla::decode_step`, flag OFF / bit-exact fallback).
enum MlaImpl {
    Gpu(MlaGpuR),
    Host(kimi::mla::MlaWeights),
}

enum KAttnR {
    Kda(KdaGpuR),
    /// MLA keeps the softmax-SDPA seam + host KV cache (per Block-4). Lever #2
    /// moves ONLY the projections onto the GPU (`MlaImpl::Gpu`); `MlaImpl::Host`
    /// is the legacy all-host oracle.
    Mla(MlaImpl, kimi::mla::MlaCache),
}

/// Resident MoE (packed switch experts held; router/shared kept host f32).
struct MoeGpuR {
    gate: GpuSwitchR,
    up: GpuSwitchR,
    down: GpuSwitchR,
    router_gate: Vec<f32>, // [E, h]
    bias: Vec<f32>,        // [E]
    // Shared expert (ungated). Lever #1: when GPU-resident, the packed 4-bit
    // gate/up/down are held ONCE as `GpuMatR` (like the routed experts) and the
    // per-token combine dispatches `matvec_mlx4`; the `Vec<f32>` mirrors stay
    // empty. When the resident flag is OFF they carry the dequantized f32
    // weights for the legacy `up_f32`+`matvec_f32` per-token path (the A/B and
    // bit-exact fallback), and the `Option`s are `None`.
    sh_gate_gpu: Option<GpuMatR>,
    sh_up_gpu: Option<GpuMatR>,
    sh_down_gpu: Option<GpuMatR>,
    sh_gate: Vec<f32>,
    sh_up: Vec<f32>,
    sh_down: Vec<f32>,
    sh_inter: usize,
    inter: usize,
    e: usize,
    scale: f32, // routed_scaling_factor
}

/// Resident dense SwiGLU MLP (lever #3): the layer-0 dense MLP's gate/up/down held
/// GPU-resident as packed 4-bit `GpuMatR` (uploaded ONCE, like the routed/shared
/// experts, MLA projections and lm_head). The per-token decode dispatches
/// `matvec_mlx4` + the existing silu/mul kernels against the held buffers, instead
/// of `kimi::dense_forward` streaming ~255 MB of host-f32 dequantized gate/up/down
/// per token. `gate`/`up` are `[n=inter, k=h]`, `down` is `[n=h, k=inter]`.
struct DenseGpuR {
    gate: GpuMatR,
    up: GpuMatR,
    down: GpuMatR,
    h: usize,
    inter: usize,
}

/// Dense MLP backend: GPU-resident (lever #3, flag ON) or the host-f32
/// `kimi::dense_forward` oracle (flag OFF / bit-exact fallback).
enum DenseImpl {
    Gpu(DenseGpuR),
    Host(kimi::DenseMlp),
}

enum KMlpR {
    /// Layer-0 dense MLP. Lever #3 moves the projections onto the GPU
    /// (`DenseImpl::Gpu`); `DenseImpl::Host` is the legacy all-host oracle.
    Dense(DenseImpl),
    Moe(MoeGpuR),
}

struct KLayerR {
    idx: usize,
    input_ln: Vec<f32>,
    post_ln: Vec<f32>,
    attn: KAttnR,
    mlp: KMlpR,
}

/// One PP window of Kimi held GPU-resident, decoded a token at a time.
pub struct KimiGpuStage {
    // NOTE: buffers-holding fields must be declared BEFORE `eng`/`_dev` only
    // matters if the device were destroyed; it is not. Order kept intuitive.
    layers: Vec<KLayerR>,
    embed: Option<Vec<f32>>,
    final_norm: Option<Vec<f32>>,
    /// Untied lm_head held GPU-RESIDENT as a packed 4-bit `GpuMatR` (k=H, n=vocab).
    /// The last-stage logits matvec runs on the GPU (matvec_mlx4 / repack) exactly
    /// like every KDA/MoE projection — NOT a 1.5GB host-f32 dequant + single-core
    /// CPU matmul (which cost ~467ms/token, 66% of the Kimi PP-3 step).
    lm_head_gpu: Option<GpuMatR>,
    eng: compute::ComputeEngine,
    _dev: device::ComputeDevice,
    cfg: KimiConfig,
    pub layer_start: usize,
    pub layer_end: usize,
    pub first: bool,
    pub last: bool,
    h: usize,
    eps: f32,
}

// ---- little-endian tensor readers (mirror the harness) ----
fn to_u32(d: &[u8]) -> Vec<u32> {
    d.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn bf16(d: &[u8]) -> Vec<f32> {
    d.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
}

impl KimiGpuStage {
    /// Build the resident stage for window `[layer_start, layer_end)`. Uploads all
    /// packed KDA/MoE buffers ONCE. `load_edges` pulls embed (first stage) and
    /// final_norm/lm_head (last stage) to host.
    pub fn new(
        model_dir: &str,
        cfg: &KimiConfig,
        layer_start: usize,
        layer_end: usize,
        load_edges: bool,
        device_idx: usize,
    ) -> Result<KimiGpuStage, String> {
        if layer_start >= layer_end || layer_end > cfg.num_hidden_layers {
            return Err(format!("bad window [{layer_start},{layer_end}) for {} layers", cfg.num_hidden_layers));
        }
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let n_layers = cfg.num_hidden_layers;
        let first = layer_start == 0;
        let last = layer_end == n_layers;

        // --- engine ---
        let dev = device::ComputeDevice::create(device_idx)?;
        let shader_spvs = crate::include_all_shaders();
        let refs: HashMap<&str, &[u8]> = shader_spvs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let mut eng = compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, dev.caps(), &refs,
        )?;

        // --- shard index + mmaps ---
        let ckpt_file = Path::new(model_dir).join("model.safetensors.index.json");
        let index: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&ckpt_file).map_err(|e| format!("index: {e}"))?,
        ).map_err(|e| format!("index: {e}"))?;
        let wm = index["weight_map"].as_object().ok_or("no weight_map")?;
        let shard_of = |name: &str| -> Result<String, String> {
            wm.get(name).and_then(|x| x.as_str()).map(|s| s.to_string())
                .ok_or_else(|| format!("{name} not in weight_map"))
        };
        let mut shard_set: BTreeSet<String> = Default::default();
        for v in wm.values() { if let Some(s) = v.as_str() { shard_set.insert(s.to_string()); } }
        let mut mmaps: HashMap<String, Mmap> = Default::default();
        for sp in &shard_set {
            let f = std::fs::File::open(format!("{model_dir}/{sp}")).map_err(|e| format!("open {sp}: {e}"))?;
            mmaps.insert(sp.clone(), unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {sp}: {e}"))?);
        }
        let deq = |base: &str| -> Result<Vec<f32>, String> {
            let wname = format!("{base}.weight");
            let sp = shard_of(&wname)?;
            let st = SafeTensors::deserialize(&mmaps[&sp]).map_err(|e| format!("deser {sp}: {e}"))?;
            let wv = st.tensor(&wname).map_err(|e| format!("{wname}: {e}"))?;
            let sv = st.tensor(&format!("{base}.scales")).map_err(|e| format!("{base}.scales: {e}"))?;
            let bv = st.tensor(&format!("{base}.biases")).map_err(|e| format!("{base}.biases: {e}"))?;
            let wshape = wv.shape();
            let in_features = sv.shape().last().unwrap() * GROUP_SIZE;
            let out_total: usize = wshape[..wshape.len() - 1].iter().product();
            let bits = (*wshape.last().unwrap() * 32) / in_features;
            Ok(model::dequantize_mlx_affine(&to_u32(wv.data()), &bf16(sv.data()), &bf16(bv.data()), out_total, in_features, GROUP_SIZE, bits))
        };
        let raw = |name: &str| -> Result<Vec<f32>, String> {
            let sp = shard_of(name)?;
            let st = SafeTensors::deserialize(&mmaps[&sp]).map_err(|e| format!("deser {sp}: {e}"))?;
            let tv = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
            Ok(match tv.dtype() {
                safetensors::Dtype::BF16 => bf16(tv.data()),
                safetensors::Dtype::F32 => tv.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                d => return Err(format!("{name}: dtype {d:?}")),
            })
        };
        // resident packed matvec uploader
        let mut up_mat = |eng: &mut compute::ComputeEngine, base: &str| -> Result<GpuMatR, String> {
            let wname = format!("{base}.weight");
            let sp = shard_of(&wname)?;
            let st = SafeTensors::deserialize(&mmaps[&sp]).map_err(|e| format!("deser {sp}: {e}"))?;
            let wv = st.tensor(&wname).map_err(|e| format!("{wname}: {e}"))?;
            let sv = st.tensor(&format!("{base}.scales")).map_err(|e| format!("{base}.scales: {e}"))?;
            let bv = st.tensor(&format!("{base}.biases")).map_err(|e| format!("{base}.biases: {e}"))?;
            let wshape = wv.shape();
            let inn = sv.shape().last().unwrap() * GROUP_SIZE;
            let out: usize = wshape[..wshape.len() - 1].iter().product();
            let packed = to_u32(wv.data());
            let scales = bf16(sv.data());
            let biases = bf16(bv.data());
            let pb = bytemuck::cast_slice::<u32, u8>(&packed).to_vec();
            let p = eng.alloc_host_coherent_storage(pb.len().max(4) as u64)?;
            p.write(&pb)?;
            let sbuf = eng.alloc_host_coherent_storage((f32_slice_to_bytes(&scales).len()).max(4) as u64)?;
            sbuf.write(&f32_slice_to_bytes(&scales))?;
            let bbuf = eng.alloc_host_coherent_storage((f32_slice_to_bytes(&biases).len()).max(4) as u64)?;
            bbuf.write(&f32_slice_to_bytes(&biases))?;
            Ok(GpuMatR { p, s: sbuf, b: bbuf, k: inn, n: out })
        };
        let up_switch = |eng: &mut compute::ComputeEngine, qs: &moe::QuantSwitch| -> Result<GpuSwitchR, String> {
            let pb = bytemuck::cast_slice::<u32, u8>(&qs.packed).to_vec();
            let p = eng.alloc_host_coherent_storage(pb.len().max(4) as u64)?;
            p.write(&pb)?;
            let s = eng.alloc_host_coherent_storage((f32_slice_to_bytes(&qs.scales).len()).max(4) as u64)?;
            s.write(&f32_slice_to_bytes(&qs.scales))?;
            let b = eng.alloc_host_coherent_storage((f32_slice_to_bytes(&qs.biases).len()).max(4) as u64)?;
            b.write(&f32_slice_to_bytes(&qs.biases))?;
            Ok(GpuSwitchR { p, s, b, out_features: qs.out_features, in_features: qs.in_features })
        };

        // --- build each layer resident ---
        let mut layers: Vec<KLayerR> = Vec::with_capacity(layer_end - layer_start);
        for l in layer_start..layer_end {
            let p = format!("model.layers.{l}");
            let ap = format!("{p}.self_attn");
            let input_ln = raw(&format!("{p}.input_layernorm.weight"))?;
            let post_ln = raw(&format!("{p}.post_attention_layernorm.weight"))?;

            let attn = match cfg.layer_schedule[l] {
                KimiLayerKind::Kda => {
                    let (nh, hd, kern) = (cfg.kda_num_heads, cfg.kda_head_dim, cfg.kda_conv_kernel);
                    let q = up_mat(&mut eng, &format!("{ap}.q_proj"))?;
                    let k = up_mat(&mut eng, &format!("{ap}.k_proj"))?;
                    let v = up_mat(&mut eng, &format!("{ap}.v_proj"))?;
                    let fa = up_mat(&mut eng, &format!("{ap}.f_a_proj"))?;
                    let fb = up_mat(&mut eng, &format!("{ap}.f_b_proj"))?;
                    let b = up_mat(&mut eng, &format!("{ap}.b_proj"))?;
                    let ga = up_mat(&mut eng, &format!("{ap}.g_a_proj"))?;
                    let gb = up_mat(&mut eng, &format!("{ap}.g_b_proj"))?;
                    let o = up_mat(&mut eng, &format!("{ap}.o_proj"))?;
                    let zb = f32_slice_to_bytes(&vec![0f32; nh * hd * hd]);
                    let g_state = eng.alloc_host_coherent_storage(zb.len() as u64)?;
                    g_state.write(&zb)?;
                    // --- Lever #5 resident glue buffers ---
                    let q_conv = raw(&format!("{ap}.q_conv.conv.weight"))?;
                    let k_conv = raw(&format!("{ap}.k_conv.conv.weight"))?;
                    let v_conv = raw(&format!("{ap}.v_conv.conv.weight"))?;
                    let a_log = raw(&format!("{ap}.A_log"))?;
                    let dt_bias = raw(&format!("{ap}.dt_bias"))?;
                    let o_norm = raw(&format!("{ap}.o_norm.weight"))?;
                    let up_f32_buf =
                        |eng: &mut compute::ComputeEngine, w: &[f32]| -> Result<compute::Buffer, String> {
                            let bb = f32_slice_to_bytes(w);
                            let buf = eng.alloc_host_coherent_storage(bb.len().max(4) as u64)?;
                            buf.write(&bb)?;
                            Ok(buf)
                        };
                    let proj = nh * hd;
                    let win = kern.saturating_sub(1);
                    let key_dim = proj; // KDA: nk == nh, kd == hd
                    let q_conv_buf = up_f32_buf(&mut eng, &q_conv)?;
                    let k_conv_buf = up_f32_buf(&mut eng, &k_conv)?;
                    let v_conv_buf = up_f32_buf(&mut eng, &v_conv)?;
                    let conv_state_q = up_f32_buf(&mut eng, &vec![0f32; key_dim * win])?;
                    let conv_state_k = up_f32_buf(&mut eng, &vec![0f32; key_dim * win])?;
                    let conv_state_v = up_f32_buf(&mut eng, &vec![0f32; proj * win])?;
                    let a_log_buf = up_f32_buf(&mut eng, &a_log)?;
                    let dt_bias_buf = up_f32_buf(&mut eng, &dt_bias)?;
                    // params = [0 * 2*nv | o_norm(hd)] (leading tail unused by kda_gdn_step).
                    let mut params = vec![0f32; 2 * nh];
                    params.extend_from_slice(&o_norm);
                    let g_params = up_f32_buf(&mut eng, &params)?;
                    KAttnR::Kda(KdaGpuR {
                        q, k, v, fa, fb, b, ga, gb, o,
                        q_conv, k_conv, v_conv, a_log, dt_bias, o_norm,
                        nh, hd, kern,
                        g_state,
                        conv: kimi::kda::KdaState::new(nh, hd, kern),
                        q_conv_buf, k_conv_buf, v_conv_buf,
                        conv_state_q, conv_state_k, conv_state_v,
                        a_log_buf, dt_bias_buf, g_params,
                    })
                }
                KimiLayerKind::Mla => {
                    let (nh, nope, v, r) = (cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
                    let pe = cfg.qk_rope_head_dim;
                    // Lever #2: hold the 4 MLA projections GPU-resident (packed
                    // 4-bit, uploaded ONCE) when VLLM_VULKAN_KIMI_MLA_RESIDENT is
                    // ON — the per-token decode dispatches `matvec_mlx4` against the
                    // held buffers instead of streaming ~116MB of host-f32
                    // dequantized weights per MLA layer per token. OFF (=0)
                    // restores the host-f32 `matmul_wt` oracle (`decode_step`).
                    let mla = if crate::flags::flags_global().kimi_mla_resident {
                        // kv_b_proj held in its NATURAL on-disk [nh*(nope+v), r]
                        // layout — NO embed_q/unembed_out transpose; the per-head
                        // head_dim=(nope+v) matvec output splits into k_nope || v.
                        MlaImpl::Gpu(MlaGpuR {
                            q_proj: up_mat(&mut eng, &format!("{ap}.q_proj"))?,
                            kv_a_proj: up_mat(&mut eng, &format!("{ap}.kv_a_proj_with_mqa"))?,
                            kv_b_proj: up_mat(&mut eng, &format!("{ap}.kv_b_proj"))?,
                            o_proj: up_mat(&mut eng, &format!("{ap}.o_proj"))?,
                            kv_a_layernorm: raw(&format!("{ap}.kv_a_layernorm.weight"))?,
                            h, nh, nope, pe, v, r, eps,
                        })
                    } else {
                        let head_dim = nope + v;
                        let kvb = deq(&format!("{ap}.kv_b_proj"))?;
                        let mut embed_q = vec![0f32; nh * r * nope];
                        let mut unembed_out = vec![0f32; nh * v * r];
                        for hh in 0..nh {
                            let vb = &kvb[hh * head_dim * r..(hh + 1) * head_dim * r];
                            for n in 0..nope { for rr in 0..r { embed_q[(hh * r + rr) * nope + n] = vb[n * r + rr]; } }
                            for d in 0..v { for rr in 0..r { unembed_out[(hh * v + d) * r + rr] = vb[(nope + d) * r + rr]; } }
                        }
                        MlaImpl::Host(kimi::mla::MlaWeights {
                            h, nh, nope, pe, v, r, eps,
                            q_proj: deq(&format!("{ap}.q_proj"))?,
                            kv_a_proj: deq(&format!("{ap}.kv_a_proj_with_mqa"))?,
                            kv_a_layernorm: raw(&format!("{ap}.kv_a_layernorm.weight"))?,
                            embed_q, unembed_out, o_proj: deq(&format!("{ap}.o_proj"))?,
                        })
                    };
                    KAttnR::Mla(mla, kimi::mla::MlaCache::new())
                }
            };

            let mp = format!("{p}.mlp");
            let mlp = if cfg.is_dense_mlp(l) {
                let inter = cfg.intermediate_size;
                // Lever #3: hold the layer-0 dense MLP GPU-resident (packed 4-bit,
                // uploaded ONCE) when VLLM_VULKAN_KIMI_DENSE_RESIDENT is ON — the
                // per-token forward dispatches `matvec_mlx4` + silu/mul against the
                // held buffers instead of streaming ~255MB of host-f32 dequantized
                // gate/up/down per token. OFF (=0) restores the host-f32
                // `kimi::dense_forward` oracle.
                let dense = if crate::flags::flags_global().kimi_dense_resident {
                    DenseImpl::Gpu(DenseGpuR {
                        gate: up_mat(&mut eng, &format!("{mp}.gate_proj"))?,
                        up: up_mat(&mut eng, &format!("{mp}.up_proj"))?,
                        down: up_mat(&mut eng, &format!("{mp}.down_proj"))?,
                        h, inter,
                    })
                } else {
                    DenseImpl::Host(kimi::DenseMlp {
                        h, inter,
                        gate: deq(&format!("{mp}.gate_proj"))?, up: deq(&format!("{mp}.up_proj"))?,
                        down: deq(&format!("{mp}.down_proj"))?,
                    })
                };
                KMlpR::Dense(dense)
            } else {
                let qm = model::load_qwen35_moe_quant_experts(&ckpt_file, GROUP_SIZE, 4, l, l + 1)?;
                let qg = qm.gate.get(&l).ok_or("no switch experts")?;
                let qu = qm.up.get(&l).unwrap();
                let qd = qm.down.get(&l).unwrap();
                let inter = qg.out_features;
                let (router_gate, e) = { let g = deq(&format!("{mp}.gate"))?; let e = g.len() / h; (g, e) };
                let bias = raw(&format!("{mp}.e_score_correction_bias"))?;
                // Lever #1: hold the ungated shared expert GPU-resident (packed
                // 4-bit, uploaded ONCE) when VLLM_VULKAN_KIMI_SHARED_RESIDENT is
                // ON — the per-token combine then dispatches `matvec_mlx4`
                // against the held buffers instead of re-uploading ~28MB of host
                // f32 (`up_f32`) per MoE layer per token. OFF = legacy host-f32.
                let shared_resident = crate::flags::flags_global().kimi_shared_resident;
                let sg = format!("{mp}.shared_experts.gate_proj");
                let su = format!("{mp}.shared_experts.up_proj");
                let sd = format!("{mp}.shared_experts.down_proj");
                let (sh_gate_gpu, sh_up_gpu, sh_down_gpu, sh_gate, sh_up, sh_down, sh_inter) =
                    if shared_resident {
                        let g = up_mat(&mut eng, &sg)?;
                        let u = up_mat(&mut eng, &su)?;
                        let d = up_mat(&mut eng, &sd)?;
                        let si = g.n; // out_features == shared intermediate size
                        (Some(g), Some(u), Some(d), Vec::new(), Vec::new(), Vec::new(), si)
                    } else {
                        let g = deq(&sg)?;
                        let u = deq(&su)?;
                        let d = deq(&sd)?;
                        let si = g.len() / h;
                        (None, None, None, g, u, d, si)
                    };
                let gate = up_switch(&mut eng, qg)?;
                let up = up_switch(&mut eng, qu)?;
                let down = up_switch(&mut eng, qd)?;
                KMlpR::Moe(MoeGpuR { gate, up, down, router_gate, bias, sh_gate_gpu, sh_up_gpu, sh_down_gpu, sh_gate, sh_up, sh_down, sh_inter, inter, e, scale: cfg.routed_scaling_factor })
            };

            layers.push(KLayerR { idx: l, input_ln, post_ln, attn, mlp });
        }

        // --- edges ---
        let (mut embed, mut final_norm, mut lm_head_gpu) = (None, None, None);
        if load_edges && first {
            // embed_tokens is mlx4-quantized (dtype U32 packed) — dequantize.
            embed = Some(deq("model.embed_tokens")?);
        }
        if load_edges && last {
            final_norm = Some(raw("model.norm.weight")?);
            // lm_head is untied + mlx4-quantized: hold it packed 4-bit GPU-resident
            // (~189MB) and run the logits matvec on the GPU, instead of dequantizing
            // to a 1.5GB host f32 blob + a single-core CPU matmul (the 467ms tax).
            lm_head_gpu = Some(up_mat(&mut eng, "lm_head")?);
        }

        Ok(KimiGpuStage {
            layers, embed, final_norm, lm_head_gpu, eng, _dev: dev,
            cfg: cfg.clone(), layer_start, layer_end, first, last, h, eps,
        })
    }

    /// Reset every layer's persistent recurrence/KV state (fresh decode session).
    pub fn reset_state(&mut self) -> Result<(), String> {
        for layer in self.layers.iter_mut() {
            match &mut layer.attn {
                KAttnR::Kda(kda) => {
                    let zb = f32_slice_to_bytes(&vec![0f32; kda.nh * kda.hd * kda.hd]);
                    kda.g_state.write(&zb)?;
                    kda.conv = kimi::kda::KdaState::new(kda.nh, kda.hd, kda.kern);
                    // Lever #5: reset the GPU-resident conv sliding-windows too.
                    let win = kda.kern.saturating_sub(1);
                    let zq = f32_slice_to_bytes(&vec![0f32; kda.nh * kda.hd * win]);
                    kda.conv_state_q.write(&zq)?;
                    kda.conv_state_k.write(&zq)?;
                    kda.conv_state_v.write(&zq)?;
                }
                KAttnR::Mla(_, c) => { *c = kimi::mla::MlaCache::new(); }
            }
        }
        Ok(())
    }

    /// One PP-stage single-token decode step (the GPU-resident `forward_pp_stage`).
    /// First stage embeds `token_id`; else consumes `hidden_in[H]`. Last stage
    /// returns `[vocab]` logits; else the `[H]` hidden to ship onward.
    pub fn forward_pp_stage(&mut self, token_id: u32, hidden_in: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.h;
        let eps = self.eps;
        let mut x = if self.first {
            let emb = self.embed.as_ref().ok_or("stage 0 requires embed")?;
            let row = token_id as usize * h;
            emb[row..row + h].to_vec()
        } else {
            if hidden_in.len() != h { return Err(format!("PP hidden_in {} != {h}", hidden_in.len())); }
            hidden_in.to_vec()
        };

        // disjoint field borrows: eng (mut) + layers (mut) + cfg (shared)
        let eng = &mut self.eng;
        for layer in self.layers.iter_mut() {
            let xn = kimi::rmsnorm(&x, 1, h, &layer.input_ln, eps);
            let attn = match &mut layer.attn {
                KAttnR::Kda(kda) => {
                    if crate::flags::flags_global().kimi_kda_fused {
                        kda_step_resident_fused(eng, kda, &xn, eps)?
                    } else {
                        kda_step_resident(eng, kda, &xn, eps)?
                    }
                }
                KAttnR::Mla(imp, c) => match imp {
                    MlaImpl::Gpu(g) => mla_step_resident(eng, g, &xn, c)?,
                    MlaImpl::Host(w) => kimi::mla::decode_step(w, &xn, c),
                },
            };
            let mut hres = vec![0f32; h];
            for i in 0..h { hres[i] = x[i] + attn[i]; }
            let hn = kimi::rmsnorm(&hres, 1, h, &layer.post_ln, eps);
            let mlp = match &layer.mlp {
                KMlpR::Dense(d) => match d {
                    DenseImpl::Gpu(g) => dense_step_resident(eng, g, &hn)?,
                    DenseImpl::Host(w) => kimi::dense_forward(w, &hn, 1),
                },
                KMlpR::Moe(m) => moe_combine_resident(eng, m, &hn, h)?,
            };
            let mut out = vec![0f32; h];
            for i in 0..h { out[i] = hres[i] + mlp[i]; }
            x = out;
        }

        if self.last {
            let fnorm = self.final_norm.as_ref().ok_or("tail stage requires final_norm")?;
            let normed = kimi::rmsnorm(&x, 1, h, fnorm, eps);
            // GPU-resident logits matvec (matvec_mlx4 / repack) — disjoint field
            // borrows: &self.lm_head_gpu (shared) + &mut self.eng.
            let lm = self.lm_head_gpu.as_ref().ok_or("tail stage requires lm_head_gpu")?;
            lm_head_logits(&mut self.eng, lm, &normed)
        } else {
            Ok(x)
        }
    }
}

/// Record a resident mlx4 matvec into an OPEN command buffer WITHOUT submitting:
/// the packed weight `m` is held on the GPU; `xin` (a GPU buffer of >=k f32) is
/// the activation, `out` (>=n f32) receives the result. The caller owns the
/// begin/submit/fence and must insert a `record_barrier_to` before any dispatch
/// that reads `out`. This is the CB-batching primitive: many projections that
/// only depend on the same held input collapse into ONE submit+fence instead of
/// one fence apiece (the Kimi resident fence-tax lever). When
/// `VLLM_VULKAN_MLX4_REPACK=1`, admitted (k%32==0, k>=1024, n>=1024) shapes route
/// to the repack kernel here exactly as the standalone path did — Kimi KDA
/// projections and MoE experts inherit the 1.5-6x per-op win transparently.
fn record_matvec_r(
    eng: &mut compute::ComputeEngine,
    cb: vk::CommandBuffer,
    m: &GpuMatR,
    xin: &compute::Buffer,
    out: &compute::Buffer,
) -> Result<(), String> {
    let (shader, r) = matvec_mlx4_variant_k(m.k, m.n);
    let wg = (m.n as u32 + r - 1) / r;
    let pc = matvec_mlx4_pc_off(m.k, m.n, GROUP_SIZE, 0, 0);
    eng.record_to(cb, &shader, &[&m.p, &m.s, &m.b, xin, out], &pc, (wg, 1, 1))
}

/// Last-stage logits: `normed[H]` @ held packed lm_head `[vocab, H]` on the GPU
/// (one matvec_mlx4/repack dispatch, one submit+fence), returning `[vocab]` f32.
/// Replaces the 1.5GB host-f32 dequant + single-core CPU matmul (~467ms/token).
/// The caller (pp_step_kimi) argmaxes the returned Vec in Rust, so the full vocab
/// never crosses the pyo3 boundary — only the readback (vocab*4 B) leaves the GPU.
fn lm_head_logits(
    eng: &mut compute::ComputeEngine,
    lm: &GpuMatR,
    normed: &[f32],
) -> Result<Vec<f32>, String> {
    if normed.len() != lm.k {
        return Err(format!("lm_head normed {} != k {}", normed.len(), lm.k));
    }
    let xb = f32_slice_to_bytes(normed);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let o = eng.alloc_host_coherent_storage((lm.n.max(1) * 4) as u64)?;
    let cb = eng.begin_batch()?;
    record_matvec_r(eng, cb, lm, &xbuf, &o)?;
    eng.submit_batch(cb)?;
    let logits = read_f32_buf(&o, lm.n);
    eng.return_to_pool(xbuf);
    eng.return_to_pool(o);
    Ok(logits)
}

/// The resident KDA decode step: identical math to `kimi_kda_decode_step_gpu`, but
/// projections read HELD `GpuMatR`s and the recurrence advances the HELD state.
fn kda_step_resident(
    eng: &mut compute::ComputeEngine,
    kda: &mut KdaGpuR,
    x: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    let (nh, hd, kern) = (kda.nh, kda.hd, kda.kern);
    let proj = nh * hd;
    let inv = (hd as f32).powf(-0.5);

    // ---- SEGMENT 1: all 8 KDA projections in ONE command buffer / ONE fence ----
    // The 6 direct projections (q/k/v/fa/b/ga) read the held input `x`; the 2
    // low-rank tails (fb reads fa, gb reads ga) chain off GPU-resident outputs via
    // a single barrier — no host round-trip between them. Was 8 submits+fences.
    if x.len() != kda.q.k { return Err(format!("kda x {} != k {}", x.len(), kda.q.k)); }
    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let a1 = |eng: &mut compute::ComputeEngine, n: usize| eng.alloc_host_coherent_storage((n.max(1) * 4) as u64);
    let o_q = a1(eng, kda.q.n)?;
    let o_k = a1(eng, kda.k.n)?;
    let o_v = a1(eng, kda.v.n)?;
    let o_fa = a1(eng, kda.fa.n)?;
    let o_b = a1(eng, kda.b.n)?;
    let o_ga = a1(eng, kda.ga.n)?;
    let o_alog = a1(eng, kda.fb.n)?;
    let o_gate = a1(eng, kda.gb.n)?;
    let cb = eng.begin_batch()?;
    record_matvec_r(eng, cb, &kda.q, &xbuf, &o_q)?;
    record_matvec_r(eng, cb, &kda.k, &xbuf, &o_k)?;
    record_matvec_r(eng, cb, &kda.v, &xbuf, &o_v)?;
    record_matvec_r(eng, cb, &kda.fa, &xbuf, &o_fa)?;
    record_matvec_r(eng, cb, &kda.b, &xbuf, &o_b)?;
    record_matvec_r(eng, cb, &kda.ga, &xbuf, &o_ga)?;
    eng.record_barrier_to(cb); // fa, ga written -> feed the low-rank tails
    record_matvec_r(eng, cb, &kda.fb, &o_fa, &o_alog)?;
    record_matvec_r(eng, cb, &kda.gb, &o_ga, &o_gate)?;
    eng.submit_batch(cb)?;
    let qc0 = read_f32_buf(&o_q, kda.q.n);
    let kc0 = read_f32_buf(&o_k, kda.k.n);
    let vc0 = read_f32_buf(&o_v, kda.v.n);
    let a_log_in = read_f32_buf(&o_alog, kda.fb.n);
    let b_in = read_f32_buf(&o_b, kda.b.n);
    let gate = read_f32_buf(&o_gate, kda.gb.n);
    for buf in [xbuf, o_q, o_k, o_v, o_fa, o_b, o_ga, o_alog, o_gate] {
        eng.return_to_pool(buf);
    }

    // host conv/qk-norm/decay glue (verbatim from kda::decode_step / the harness)
    let mut q = kimi::kda::conv_step(&mut kda.conv.conv_q, &qc0, proj, &kda.q_conv, kern);
    let mut k = kimi::kda::conv_step(&mut kda.conv.conv_k, &kc0, proj, &kda.k_conv, kern);
    let v = kimi::kda::conv_step(&mut kda.conv.conv_v, &vc0, proj, &kda.v_conv, kern);
    kimi::kda::rms_no_weight(&mut q, nh, hd, 1e-6);
    kimi::kda::rms_no_weight(&mut k, nh, hd, 1e-6);
    for z in q.iter_mut() { *z *= inv * inv; }
    for z in k.iter_mut() { *z *= inv; }
    let mut decay = vec![0f32; proj];
    for hh in 0..nh {
        let neg_exp_a = -(kda.a_log[hh].exp());
        for dk in 0..hd {
            let al = a_log_in[hh * hd + dk] + kda.dt_bias[hh * hd + dk];
            decay[hh * hd + dk] = (neg_exp_a * kimi::kda::softplus(al)).exp();
        }
    }

    let (nv, kd, vd) = (nh, hd, hd);
    let (key_dim, value_dim) = (proj, proj);
    let conv_dim = 2 * key_dim + value_dim;
    let v_off = 2 * key_dim;
    let mut params = vec![0f32; 2 * nv];
    params.extend_from_slice(&kda.o_norm);
    let g_params = {
        let pb = f32_slice_to_bytes(&params);
        let bf = eng.alloc_host_coherent_storage(pb.len() as u64)?;
        bf.write(&pb)?; bf
    };
    let alloc = |eng: &mut compute::ComputeEngine, n: usize| -> Result<compute::Buffer, String> {
        eng.alloc_host_coherent_storage((n.max(1) * 4) as u64)
    };
    let b_q = alloc(eng, key_dim)?;
    let b_k = alloc(eng, key_dim)?;
    let b_conv = alloc(eng, conv_dim)?;
    let b_decay = alloc(eng, nv * kd)?;
    let b_b = alloc(eng, nv)?;
    let b_gate = alloc(eng, value_dim)?;
    let b_gated = alloc(eng, value_dim)?;
    let pc = q35_gdn_pc(kd, vd, 1, v_off, eps, nv);
    let mut conv_scratch = vec![0f32; conv_dim];
    conv_scratch[v_off..v_off + value_dim].copy_from_slice(&v);
    b_q.write(&f32_slice_to_bytes(&q))?;
    b_k.write(&f32_slice_to_bytes(&k))?;
    b_conv.write(&f32_slice_to_bytes(&conv_scratch))?;
    b_decay.write(&f32_slice_to_bytes(&decay))?;
    b_b.write(&f32_slice_to_bytes(&b_in))?;
    b_gate.write(&f32_slice_to_bytes(&gate))?;
    // ---- SEGMENT 2: kda_gdn_step + o_proj in ONE command buffer / ONE fence ----
    // The recurrence step advances the HELD state in place and writes `b_gated`;
    // o_proj reads `b_gated` GPU-resident (barrier, no host round-trip). Was 2
    // submits+fences (gdn_step, then matvec_r for o_proj).
    let o_out = alloc(eng, kda.o.n)?;
    let cb = eng.begin_batch()?;
    eng.record_to(cb, "kda_gdn_step",
        &[&b_q, &b_k, &b_conv, &b_decay, &b_b, &b_gate, &g_params, &kda.g_state, &b_gated],
        &pc, (nv as u32, 1, 1))?;
    eng.record_barrier_to(cb); // b_gated written -> o_proj input
    record_matvec_r(eng, cb, &kda.o, &b_gated, &o_out)?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o_out, kda.o.n);

    for buf in [g_params, b_q, b_k, b_conv, b_decay, b_b, b_gate, b_gated, o_out] {
        eng.return_to_pool(buf);
    }
    Ok(out)
}

/// Lever #5: the FULLY FUSED resident KDA decode step — the same math as
/// `kda_step_resident`, but the depthwise conv / qk-RMSNorm / per-channel decay
/// glue records ONTO THE GPU (no host round-trip), so the whole
/// projections -> conv -> qknorm -> decay -> gdn_step -> o_proj chain is ONE
/// command buffer / ONE fence (vs the 2 fenced submits + host seam of the OFF
/// path). Only `x` (input_ln'd) is uploaded and only `[H]` is read back.
///
/// Buffer layout mirrors the proven qwen3.6 GDN decode CB (`debug_qwen35_gdn_gpu`):
/// the three separate q/k/v conv outputs are placed into ONE combined `b_conv`
/// `[conv_dim = 2*key_dim + value_dim]` (q at `[0,key_dim)`, k at
/// `[key_dim,2*key_dim)`, v at `[2*key_dim,conv_dim)`) via `record_to_off`
/// output offsets, exactly the layout `q35_gdn_qknorm` (reads q/k) and
/// `kda_gdn_step` (reads v at `v_off`) expect. Reused shaders are BIT-IDENTICAL
/// to their host-reference accumulation order; the moved-to-GPU conv-silu /
/// qknorm-sqrt / decay-exp differ only in the last ulp (GPU intrinsic vs libm)
/// → argmax-exact / cos≈1.0, the accepted KimiGpuStage decode tolerance.
fn kda_step_resident_fused(
    eng: &mut compute::ComputeEngine,
    kda: &mut KdaGpuR,
    x: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    let (nh, hd, kern) = (kda.nh, kda.hd, kda.kern);
    let proj = nh * hd;
    let key_dim = proj; // KDA: nk == nh, kd == hd
    let value_dim = proj; // nv == nh, vd == hd
    let conv_dim = 2 * key_dim + value_dim;
    let v_off = 2 * key_dim;
    let inv = (hd as f32).powf(-0.5);
    if x.len() != kda.q.k {
        return Err(format!("kda x {} != k {}", x.len(), kda.q.k));
    }

    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let a1 = |eng: &mut compute::ComputeEngine, n: usize| eng.alloc_host_coherent_storage((n.max(1) * 4) as u64);
    // projection outputs
    let o_q = a1(eng, kda.q.n)?;
    let o_k = a1(eng, kda.k.n)?;
    let o_v = a1(eng, kda.v.n)?;
    let o_fa = a1(eng, kda.fa.n)?;
    let o_b = a1(eng, kda.b.n)?;
    let o_ga = a1(eng, kda.ga.n)?;
    let o_alog = a1(eng, kda.fb.n)?; // f_b(f_a(x)) = a_log_in [proj]
    let o_gate = a1(eng, kda.gb.n)?; // g_b(g_a(x)) [value_dim] (pre-sigmoid)
    // fused intermediates
    let b_conv = a1(eng, conv_dim)?;
    let b_decay = a1(eng, proj)?;
    let b_q = a1(eng, key_dim)?;
    let b_k = a1(eng, key_dim)?;
    let b_gated = a1(eng, value_dim)?;
    let o_out = a1(eng, kda.o.n)?;

    let conv_pc_qk = q35_conv_pc(key_dim, kern);
    let conv_pc_v = q35_conv_pc(value_dim, kern);
    let qk_pc = q35_qknorm_pc(nh, hd, key_dim, 1e-6, inv);
    let decay_pc = kda_decay_pc(nh, hd);
    let gdn_pc = q35_gdn_pc(hd, hd, 1, v_off, eps, nh);
    let conv_wg_qk = ((key_dim as u32) + 255) / 256;
    let conv_wg_v = ((value_dim as u32) + 255) / 256;
    let decay_wg = ((proj as u32) + 255) / 256;

    // ---- ENTIRE KDA layer in ONE command buffer / ONE fence ----
    let cb = eng.begin_batch()?;
    // stage A: 6 direct projections (read x).
    record_matvec_r(eng, cb, &kda.q, &xbuf, &o_q)?;
    record_matvec_r(eng, cb, &kda.k, &xbuf, &o_k)?;
    record_matvec_r(eng, cb, &kda.v, &xbuf, &o_v)?;
    record_matvec_r(eng, cb, &kda.fa, &xbuf, &o_fa)?;
    record_matvec_r(eng, cb, &kda.b, &xbuf, &o_b)?;
    record_matvec_r(eng, cb, &kda.ga, &xbuf, &o_ga)?;
    eng.record_barrier_to(cb); // fa, ga written -> low-rank tails
    // stage B: 2 low-rank tails (fb reads fa, gb reads ga).
    record_matvec_r(eng, cb, &kda.fb, &o_fa, &o_alog)?;
    record_matvec_r(eng, cb, &kda.gb, &o_ga, &o_gate)?;
    eng.record_barrier_to(cb); // all projections written -> conv + decay
    // stage C: depthwise conv+silu q/k/v into the combined b_conv (offset
    // outputs) + per-channel decay (independent, reads o_alog). The conv
    // advances the RESIDENT per-layer windows in place.
    eng.record_to_off(cb, "q35_dn_conv_step",
        &[(&kda.q_conv_buf, 0), (&o_q, 0), (&kda.conv_state_q, 0), (&b_conv, 0)],
        &conv_pc_qk, (conv_wg_qk, 1, 1))?;
    eng.record_to_off(cb, "q35_dn_conv_step",
        &[(&kda.k_conv_buf, 0), (&o_k, 0), (&kda.conv_state_k, 0), (&b_conv, (key_dim * 4) as u64)],
        &conv_pc_qk, (conv_wg_qk, 1, 1))?;
    eng.record_to_off(cb, "q35_dn_conv_step",
        &[(&kda.v_conv_buf, 0), (&o_v, 0), (&kda.conv_state_v, 0), (&b_conv, (v_off * 4) as u64)],
        &conv_pc_v, (conv_wg_v, 1, 1))?;
    eng.record_to(cb, "kda_decay",
        &[&o_alog, &kda.dt_bias_buf, &kda.a_log_buf, &b_decay],
        &decay_pc, (decay_wg, 1, 1))?;
    eng.record_barrier_to(cb); // conv (q/k) + decay written -> qknorm
    // stage D: qk-RMSNorm (no weight, eps 1e-6) + inv-scale, reading q/k from
    // b_conv, writing b_q/b_k.
    eng.record_to(cb, "q35_gdn_qknorm",
        &[&b_conv, &b_q, &b_k],
        &qk_pc, (2 * nh as u32, 1, 1))?;
    eng.record_barrier_to(cb); // b_q/b_k written -> gdn_step
    // stage E: KDA delta-rule recurrence + gated o_norm (advances g_state in
    // place; v read from b_conv at v_off; beta from o_b; gate from o_gate).
    eng.record_to(cb, "kda_gdn_step",
        &[&b_q, &b_k, &b_conv, &b_decay, &o_b, &o_gate, &kda.g_params, &kda.g_state, &b_gated],
        &gdn_pc, (nh as u32, 1, 1))?;
    eng.record_barrier_to(cb); // b_gated written -> o_proj
    // stage F: o_proj.
    record_matvec_r(eng, cb, &kda.o, &b_gated, &o_out)?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o_out, kda.o.n);

    for buf in [
        xbuf, o_q, o_k, o_v, o_fa, o_b, o_ga, o_alog, o_gate,
        b_conv, b_decay, b_q, b_k, b_gated, o_out,
    ] {
        eng.return_to_pool(buf);
    }
    Ok(out)
}

/// The resident MLA decode step (lever #2): the 4 MLA projections dispatch
/// `matvec_mlx4` against HELD `GpuMatR`s; the softmax-SDPA seam
/// (`kv_a_layernorm`, KV append, causal softmax, value combine) stays on the
/// host via `kimi::mla::{kv_a_layernorm_apply, attend_append}` — the SAME code
/// the CPU oracle `kimi::mla::decode_step` runs. The two paths therefore differ
/// ONLY in the projection accumulation order (GPU subgroup-reduce vs host
/// sequential dot) → argmax-exact / cos≈1.0, NOT byte-identity.
///
/// Three fenced submits, split at the two host seams: {q_proj, kv_a_proj} both
/// read `x`; then host layernorm produces the latent → {kv_b_proj} decompress;
/// then host softmax produces the attention output → {o_proj}. kv_b_proj is
/// held in its natural on-disk `[nh*(nope+v), r]` layout so ONE matvec against
/// the latent yields `[nh*(nope+v)]`, per-head split into `k_nope || v` — the
/// exact decompress the CPU `embed_q`/`unembed_out` loops do, no transpose.
fn mla_step_resident(
    eng: &mut compute::ComputeEngine,
    w: &MlaGpuR,
    x: &[f32],
    c: &mut kimi::mla::MlaCache,
) -> Result<Vec<f32>, String> {
    let (h, nh, nope, pe, v, r) = (w.h, w.nh, w.nope, w.pe, w.v, w.r);
    let qhd = nope + pe;
    let head_dim = nope + v;
    let scale = (qhd as f32).powf(-0.5);
    if x.len() != h {
        return Err(format!("mla x {} != h {h}", x.len()));
    }

    // ---- SEGMENT 1: q_proj + kv_a_proj in ONE command buffer (both read x) ----
    let xb = f32_slice_to_bytes(x);
    let xbuf = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    xbuf.write(&xb)?;
    let o_q = eng.alloc_host_coherent_storage((w.q_proj.n.max(1) * 4) as u64)?;
    let o_kva = eng.alloc_host_coherent_storage((w.kv_a_proj.n.max(1) * 4) as u64)?;
    let cb = eng.begin_batch()?;
    record_matvec_r(eng, cb, &w.q_proj, &xbuf, &o_q)?;
    record_matvec_r(eng, cb, &w.kv_a_proj, &xbuf, &o_kva)?;
    eng.submit_batch(cb)?;
    let q = read_f32_buf(&o_q, w.q_proj.n); // [nh*qhd]
    let kva = read_f32_buf(&o_kva, w.kv_a_proj.n); // [r+pe]
    for buf in [xbuf, o_q, o_kva] {
        eng.return_to_pool(buf);
    }

    // host glue (verbatim from decode_step): split q into per-head nope/pe.
    let mut q_nope = vec![0f32; nh * nope];
    let mut q_pe = vec![0f32; nh * pe];
    for hh in 0..nh {
        let base = hh * qhd;
        q_nope[hh * nope..(hh + 1) * nope].copy_from_slice(&q[base..base + nope]);
        q_pe[hh * pe..(hh + 1) * pe].copy_from_slice(&q[base + nope..base + qhd]);
    }
    // split kva into KV latent + shared pe key; kv_a_layernorm on the latent.
    let mut c_kv = kva[..r].to_vec();
    let kpe_new = kva[r..r + pe].to_vec();
    kimi::mla::kv_a_layernorm_apply(&mut c_kv, &w.kv_a_layernorm, w.eps);

    // ---- SEGMENT 2: kv_b decompress — one matvec against the layer-normed
    // latent. out[nh*head_dim]; per head, [0..nope]=k_nope, [nope..head_dim]=v.
    let ckb = f32_slice_to_bytes(&c_kv);
    let cbuf = eng.alloc_host_coherent_storage(ckb.len().max(4) as u64)?;
    cbuf.write(&ckb)?;
    let o_kvb = eng.alloc_host_coherent_storage((w.kv_b_proj.n.max(1) * 4) as u64)?;
    let cb = eng.begin_batch()?;
    record_matvec_r(eng, cb, &w.kv_b_proj, &cbuf, &o_kvb)?;
    eng.submit_batch(cb)?;
    let kvb_out = read_f32_buf(&o_kvb, w.kv_b_proj.n); // [nh*head_dim]
    for buf in [cbuf, o_kvb] {
        eng.return_to_pool(buf);
    }
    let mut kn_new = vec![0f32; nh * nope];
    let mut v_new = vec![0f32; nh * v];
    for hh in 0..nh {
        let base = hh * head_dim;
        kn_new[hh * nope..(hh + 1) * nope].copy_from_slice(&kvb_out[base..base + nope]);
        v_new[hh * v..(hh + 1) * v].copy_from_slice(&kvb_out[base + nope..base + head_dim]);
    }

    // host softmax-SDPA seam (append + causal softmax + value combine) — the
    // bit-exact code shared with the CPU oracle `decode_step`.
    let out_attn = kimi::mla::attend_append(
        c, &q_nope, &q_pe, &kn_new, &kpe_new, &v_new, nh, nope, pe, v, scale,
    );

    // ---- SEGMENT 3: o_proj matvec against the held weight ----
    if out_attn.len() != w.o_proj.k {
        return Err(format!("mla o_proj in {} != k {}", out_attn.len(), w.o_proj.k));
    }
    let ob = f32_slice_to_bytes(&out_attn);
    let obuf = eng.alloc_host_coherent_storage(ob.len().max(4) as u64)?;
    obuf.write(&ob)?;
    let o_out = eng.alloc_host_coherent_storage((w.o_proj.n.max(1) * 4) as u64)?;
    let cb = eng.begin_batch()?;
    record_matvec_r(eng, cb, &w.o_proj, &obuf, &o_out)?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&o_out, w.o_proj.n);
    for buf in [obuf, o_out] {
        eng.return_to_pool(buf);
    }
    Ok(out)
}

/// Resident MoE combine: identical math to `kimi_moe_gpu_combine`, but the packed
/// switch experts read HELD buffers (only x/router/shared uploaded per token).
/// Resident dense SwiGLU MLP decode (lever #3): `silu(gate·x) ⊙ (up·x) → down`, all
/// on the GPU via `matvec_mlx4` (against HELD packed 4-bit `GpuMatR`) + the existing
/// `silu_f32` / `mul_f32_f32_f32` kernels, in ONE command buffer / ONE fence. This
/// is exactly the math `kimi::dense_forward(d, hn, 1)` computes; only the in-kernel
/// mlx4 dequant reorders the projection dot accumulation vs the host f32 path (the
/// accepted KimiGpuStage argmax-exact / cos≈1.0 tolerance). The silu/mul shaders &
/// workgroup tiling mirror the shared-expert arm of `moe_combine_resident` (silu:
/// 512 threads/wg, mul: 256 threads/wg; the length rides in the push constant).
fn dense_step_resident(
    eng: &mut compute::ComputeEngine,
    d: &DenseGpuR,
    hn: &[f32],
) -> Result<Vec<f32>, String> {
    let (h, inter) = (d.h, d.inter);
    if hn.len() != h {
        return Err(format!("dense hn {} != h {h}", hn.len()));
    }
    let alloc = |eng: &mut compute::ComputeEngine, n: usize| -> Result<compute::Buffer, String> {
        eng.alloc_host_coherent_storage((n.max(1) * 4) as u64)
    };
    let xb = f32_slice_to_bytes(hn);
    let inp = eng.alloc_host_coherent_storage(xb.len().max(4) as u64)?;
    inp.write(&xb)?;
    let b_g = alloc(eng, inter)?; // gate·x
    let b_u = alloc(eng, inter)?; // up·x
    let b_a = alloc(eng, inter)?; // silu(gate·x)
    let b_m = alloc(eng, inter)?; // silu(gate)⊙up
    let b_out = alloc(eng, h)?;   // down·mid
    let silu_pc = ew_unary_pc(inter as u32);
    let mul_pc = ew_mul_pc(inter as u32);
    let silu_wg = (inter as u32 + 511) / 512;
    let mul_wg = (inter as u32 + 255) / 256;

    // ---- entire dense MLP in ONE command buffer / ONE fence ----
    let cb = eng.begin_batch()?;
    // stage A: gate + up (both read the same activation `inp`)
    record_matvec_r(eng, cb, &d.gate, &inp, &b_g)?;
    record_matvec_r(eng, cb, &d.up, &inp, &b_u)?;
    eng.record_barrier_to(cb);
    // stage B: silu(gate)
    eng.record_to(cb, "silu_f32", &[&b_g, &b_a], &silu_pc, (silu_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage C: mul(silu, up)
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_a, &b_u, &b_m], &mul_pc, (mul_wg, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage D: down
    record_matvec_r(eng, cb, &d.down, &b_m, &b_out)?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&b_out, h);

    for buf in [inp, b_g, b_u, b_a, b_m, b_out] {
        eng.return_to_pool(buf);
    }
    Ok(out)
}

fn moe_combine_resident(
    eng: &mut compute::ComputeEngine,
    m: &MoeGpuR,
    hn: &[f32],
    h: usize,
) -> Result<Vec<f32>, String> {
    // host router selection (sigmoid + bias + scale + top-k renorm)
    let lc = model::cpu_matmul(hn, &m.router_gate, 1, h, m.e);
    let (inds, weights) = kimi::moe::route(&lc, &m.bias, 8, m.scale, true);
    let top_k = inds.len();
    debug_assert_eq!(top_k, 8);
    let inter = m.inter;

    let up_f32 = |eng: &mut compute::ComputeEngine, w: &[f32]| -> Result<compute::Buffer, String> {
        let b = f32_slice_to_bytes(w);
        let buf = eng.alloc_host_coherent_storage(b.len().max(4) as u64)?;
        buf.write(&b)?; Ok(buf)
    };
    let alloc = |eng: &mut compute::ComputeEngine, n: usize| -> Result<compute::Buffer, String> {
        eng.alloc_host_coherent_storage((n.max(1) * 4) as u64)
    };

    let inp = up_f32(eng, hn)?;
    // Lever #1: when the shared expert is GPU-resident (packed 4-bit GpuMatR held
    // since `new`), there is NOTHING to upload per token — the ~1.47 GB/token host
    // memcpy (3 * ~28MB f32 shared weights * 26 MoE layers) is gone. OFF (=0)
    // restores the legacy per-token `up_f32` of the dequantized host f32 weights.
    let shared_resident = m.sh_gate_gpu.is_some();
    let (sgw, suw, sdw) = if shared_resident {
        (None, None, None)
    } else {
        (Some(up_f32(eng, &m.sh_gate)?), Some(up_f32(eng, &m.sh_up)?), Some(up_f32(eng, &m.sh_down)?))
    };

    let g_pack_stride = m.gate.out_features * (m.gate.in_features / 8);
    let g_sb_stride = m.gate.out_features * (m.gate.in_features / GROUP_SIZE);
    let d_pack_stride = m.down.out_features * (m.down.in_features / 8);
    let d_sb_stride = m.down.out_features * (m.down.in_features / GROUP_SIZE);

    let (gu_shader, gu_r) = matvec_mlx4_variant_k(h, inter);
    let wg_inter = (inter as u32 + gu_r - 1) / gu_r;
    let (down_shader, down_r) = matvec_mlx4_variant_k(inter, h);
    let wg_h = (h as u32 + down_r - 1) / down_r;
    let silu_pc = ew_unary_pc(inter as u32);
    let mul_pc = ew_mul_pc(inter as u32);

    // shared-expert shaders/pcs (f32, ungated) — set up before the CB so both the
    // routed experts AND the shared expert record into ONE command buffer.
    let sh_inter = m.sh_inter;
    let (sgu_shader, sgu_r) = matvec_f32_variant(sh_inter);
    let s_wg_i = (sh_inter as u32 + sgu_r - 1) / sgu_r;
    let (sd_shader, sd_r) = matvec_f32_variant(h);
    let s_wg_h = (h as u32 + sd_r - 1) / sd_r;
    let pc_sgu = matvec_pc13(h, sh_inter);
    let pc_sd = matvec_pc13(sh_inter, h);
    let ss_pc = ew_unary_pc(sh_inter as u32);
    let sm_pc = ew_mul_pc(sh_inter as u32);
    let silu_wg_i = (inter as u32 + 511) / 512;
    let mul_wg_i = (inter as u32 + 255) / 256;
    let silu_wg_s = (sh_inter as u32 + 511) / 512;
    let mul_wg_s = (sh_inter as u32 + 255) / 256;

    // Per-slot intermediate buffers: with all 8 experts + the shared expert in ONE
    // command buffer (no host sync between them), the gate/up/act/mid scratch CANNOT
    // be reused across slots (the sequential-submit code could) — each slot needs
    // its own, so the experts run concurrently on the GPU instead of serialized.
    let mut b_g: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    let mut b_u: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    let mut b_act: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    let mut b_mid: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    let mut dwn: Vec<compute::Buffer> = Vec::with_capacity(top_k);
    for _ in 0..top_k {
        b_g.push(alloc(eng, inter)?);
        b_u.push(alloc(eng, inter)?);
        b_act.push(alloc(eng, inter)?);
        b_mid.push(alloc(eng, inter)?);
        dwn.push(alloc(eng, h)?);
    }
    let b_sg = alloc(eng, sh_inter)?; let b_su = alloc(eng, sh_inter)?;
    let b_sa = alloc(eng, sh_inter)?; let b_sm = alloc(eng, sh_inter)?;
    let b_sop = alloc(eng, h)?;
    // ungated shared: sigmoid(30) == 1.0 in f32; residual h1 = 0 so out == acc.
    let slp = up_f32(eng, &[30.0])?;
    let h1z = up_f32(eng, &vec![0f32; h])?;
    let b_out = alloc(eng, h)?;
    let acc_pc = q35_moe_accum_pc(h, &weights);

    // ---- ENTIRE MoE combine in ONE command buffer / ONE fence ----
    // Interleaved by stage (all gate/up, barrier, all silu, barrier, all mul,
    // barrier, all down, barrier, accum) so a handful of GLOBAL barriers replace
    // the per-slot ones AND the experts overlap. Was 10 submits+fences
    // (8 experts + shared + accum); math identical (same shaders/pcs/inputs).
    let cb = eng.begin_batch()?;
    // stage A: gate + up (routed experts + shared)
    for slot in 0..top_k {
        let ex = inds[slot];
        let pc_gu = matvec_mlx4_pc_off(h, inter, GROUP_SIZE, ex * g_pack_stride, ex * g_sb_stride);
        eng.record_to(cb, &gu_shader, &[&m.gate.p, &m.gate.s, &m.gate.b, &inp, &b_g[slot]], &pc_gu, (wg_inter, 1, 1))?;
        eng.record_to(cb, &gu_shader, &[&m.up.p, &m.up.s, &m.up.b, &inp, &b_u[slot]], &pc_gu, (wg_inter, 1, 1))?;
    }
    if shared_resident {
        // held packed 4-bit shared gate/up -> matvec_mlx4 (same in-kernel dequant
        // as the routed experts); reads the same f32 `inp` activation.
        record_matvec_r(eng, cb, m.sh_gate_gpu.as_ref().unwrap(), &inp, &b_sg)?;
        record_matvec_r(eng, cb, m.sh_up_gpu.as_ref().unwrap(), &inp, &b_su)?;
    } else {
        eng.record_to(cb, &sgu_shader, &[sgw.as_ref().unwrap(), &inp, &b_sg], &pc_sgu, (s_wg_i, 1, 1))?;
        eng.record_to(cb, &sgu_shader, &[suw.as_ref().unwrap(), &inp, &b_su], &pc_sgu, (s_wg_i, 1, 1))?;
    }
    eng.record_barrier_to(cb);
    // stage B: silu(gate)
    for slot in 0..top_k {
        eng.record_to(cb, "silu_f32", &[&b_g[slot], &b_act[slot]], &silu_pc, (silu_wg_i, 1, 1))?;
    }
    eng.record_to(cb, "silu_f32", &[&b_sg, &b_sa], &ss_pc, (silu_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage C: mul(silu, up)
    for slot in 0..top_k {
        eng.record_to(cb, "mul_f32_f32_f32", &[&b_act[slot], &b_u[slot], &b_mid[slot]], &mul_pc, (mul_wg_i, 1, 1))?;
    }
    eng.record_to(cb, "mul_f32_f32_f32", &[&b_sa, &b_su, &b_sm], &sm_pc, (mul_wg_s, 1, 1))?;
    eng.record_barrier_to(cb);
    // stage D: down
    for slot in 0..top_k {
        let ex = inds[slot];
        let pc_d = matvec_mlx4_pc_off(inter, h, GROUP_SIZE, ex * d_pack_stride, ex * d_sb_stride);
        eng.record_to(cb, &down_shader, &[&m.down.p, &m.down.s, &m.down.b, &b_mid[slot], &dwn[slot]], &pc_d, (wg_h, 1, 1))?;
    }
    if shared_resident {
        record_matvec_r(eng, cb, m.sh_down_gpu.as_ref().unwrap(), &b_sm, &b_sop)?;
    } else {
        eng.record_to(cb, &sd_shader, &[sdw.as_ref().unwrap(), &b_sm, &b_sop], &pc_sd, (s_wg_h, 1, 1))?;
    }
    eng.record_barrier_to(cb);
    // stage E: score-weighted accumulate + ungated shared + residual
    let binds: Vec<&compute::Buffer> = vec![
        &dwn[0], &dwn[1], &dwn[2], &dwn[3], &dwn[4], &dwn[5], &dwn[6], &dwn[7],
        &b_sop, &slp, &h1z, &b_out,
    ];
    eng.record_to(cb, "q35_moe_accum", &binds, &acc_pc, ((h as u32 + 255) / 256, 1, 1))?;
    eng.submit_batch(cb)?;
    let out = read_f32_buf(&b_out, h);

    for buf in [inp, b_sg, b_su, b_sa, b_sm, b_sop, slp, h1z, b_out] {
        eng.return_to_pool(buf);
    }
    // sgw/suw/sdw only exist on the legacy (non-resident) path.
    for buf in [sgw, suw, sdw].into_iter().flatten() {
        eng.return_to_pool(buf);
    }
    for buf in b_g { eng.return_to_pool(buf); }
    for buf in b_u { eng.return_to_pool(buf); }
    for buf in b_act { eng.return_to_pool(buf); }
    for buf in b_mid { eng.return_to_pool(buf); }
    for buf in dwn { eng.return_to_pool(buf); }
    Ok(out)
}
