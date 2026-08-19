// SPDX-License-Identifier: Apache-2.0
//! DeepSeek-V4-Flash — GPU quant-resident decode (M2), the perf port of the
//! bit-exact CPU forward [`crate::dsv4_forward::dsv4_forward`] (the ARGMAX ORACLE:
//! real-ckpt argmax 11111 ` Paris`).
//!
//! ## The matvec-backend seam
//!
//! The whole difficulty of a GPU-resident DSV4 forward is that the compute is
//! deeply fused (MLA fuses 5 projections with softmax/rope/sink; the mHC block and
//! the DSA compressor fuse matmuls with Sinkhorn/pooling glue). Rather than
//! duplicate that op-DAG, [`forward_mv`] re-expresses it ONCE over a [`Mv`] trait:
//! every quantized `cpu_matmul` site becomes `mv.mm(name, …)`, while every non-
//! matmul op (RMSNorm, interleaved-RoPE, the eager-attention softmax+sink, the mHC
//! Sinkhorn, the MoE router, the SwiGLU accumulation, HyperHead) is the VERBATIM
//! bit-exact host helper from `dsv4.rs` / `dsv4_dsa.rs` / `dsv4_moe.rs`.
//!
//!   * [`CpuMv`] — matmuls via `cpu_matmul` over the dequantized `Dsv4Src`. Then
//!     `forward_mv(cfg, ids, &mut CpuMv{src})` reproduces `dsv4_forward` and is
//!     validated OFFLINE (no GPU) against the tiny transformers fixture — proving
//!     the re-expressed DAG (incl. the fused MLA/compressor bodies) is faithful.
//!   * [`Dsv4GpuStage`] — matmuls via the shipped `mul_mat_vec_mlx{6,8,2repack}`
//!     kernels over GPU-RESIDENT packed weights (the same buffers PP-10 needs),
//!     streamed in per-layer (`from_ckpt_streamed`, the LOAD-OOM cure mirroring
//!     `ling_gpu::from_ckpt_streamed`). On node this reproduces argmax 11111.
//!
//! ## GPU vs host split (this pass)
//!
//! GPU-resident matvecs: MLA `wq_a`/`wq_b`/`wkv`/`wo_b` (6-bit mlx6), routed experts
//! gate/up/down (2-bit mlx2repack), shared-expert gate/up/down + `lm_head` (8-bit
//! mlx8). Host (bit-exact) seams: every non-matmul op above, plus the DSA CSA/HCA
//! compressor + Lightning-Indexer internals, the grouped block-diagonal `wo_a`
//! o-lora projection, the plain-bf16 router gate, and the 8-bit embedding row-gather
//! — all documented remaining GPU levers (see the report). Correctness is preserved
//! throughout: the host seams run the identical oracle math.

use std::collections::HashMap;

use crate::compute;
use crate::device;
use crate::dsv4::{
    hc_residual_mix, rmsnorm_rows, unweighted_rmsnorm_rows,
    apply_interleaved_rope_inplace, Dsv4Config, LayerType, MlpType,
};
use crate::dsv4_dsa::{hca_compressor, rope_cos_sin, IndexerProj, IndexerWeights};
use crate::dsv4_forward::Dsv4Src;
use crate::dsv4_loader::Dsv4RealSrc;
use crate::dsv4_moe::{hash_router, sqrtsoftplus, topk_router};
use crate::push_constants::{
    dsv4_hc_residual_mix_pc, dsv4_swiglu_clamp_pc, f32_slice_to_bytes, matvec_mlx4_pc,
    q35_moe_accum_batched_pc, read_f32_buf,
};

// ============================================================================
// The matvec backend seam
// ============================================================================

/// Matvec backend for [`forward_mv`]. `mm`/`mm_expert` are the swappable seam
/// (CPU `cpu_matmul` vs GPU resident kernel); the rest are host-tensor fetches that
/// are identical in both backends (dequantized/plain, bit-exact to the oracle).
pub trait Mv {
    /// `out[s,out_f] = x[s,in_f] @ W[out_f,in_f]^T` for a quantized/plain linear.
    fn mm(&mut self, name: &str, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32>;
    /// Same, but `W` is the `e`-th expert slice of a 3D `[E,out,in]` switch tensor.
    fn mm_expert(&mut self, name: &str, e: usize, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32>;
    /// Dequantized/plain weight fetch for the HOST seams (compressor, grouped `wo_a`,
    /// router gate) — bit-identical to the CPU oracle's `Dsv4Src::linear`.
    fn dq_linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32>;
    /// LEVER: the pre-dequantized f32 MoE-router gate weight for `name`, if resident
    /// (`VLLM_VULKAN_DSV4_RESIDENT_ROUTER`). DEFAULT = `None` → callers fall back to the
    /// per-token `dq_linear` dequant (byte-identical). Only [`Dsv4GpuStage`] overrides it.
    fn resident_router_gate(&self, _name: &str) -> Option<&[f32]> {
        None
    }
    /// Plain dense tensor (norms / hc fn|base|scale / sinks / position_bias).
    fn dense(&self, name: &str) -> Vec<f32>;
    /// The hash-router `tid2eid` `[vocab, top_k]` i64 table.
    fn dense_i64(&self, name: &str) -> Vec<i64>;
    /// Embedding rows for `ids` → `[S, H]` (8-bit gs64 row-gather, host).
    fn embed_rows(&self, ids: &[u32], vocab: usize, h: usize) -> Vec<f32>;

    /// Manifold Hyper-Connection block → `(post [S,hc], comb [S,hc,hc], collapsed
    /// [S,h])`. This is the seam for the shipped `dsv4_hyper_connection` kernel: the
    /// DEFAULT is the bit-exact host oracle [`crate::dsv4::hc_block`] (so `CpuMv`/
    /// `CachedCpuMv` are byte-identical to before — the argmax-11111 oracle is
    /// preserved), while [`Dsv4GpuStage`] overrides it to run the two-float mHC
    /// kernel GPU-resident (cos=1.0, `debug_dsv4_hc`). HC is the highest-frequency
    /// host op in decode — 2 sites × 43 layers = 86 calls/token, each a wide
    /// `flat @ fn` reduction + 4×4 Sinkhorn — so collapsing it onto the device
    /// removes 86 host round-trips + the single-threaded f64 mix per token.
    #[allow(clippy::too_many_arguments)]
    fn hc_block(
        &mut self, streams: &[f32], seq: usize, hc: usize, h: usize,
        fn_w: &[f32], base: &[f32], scale: &[f32], iters: usize, hc_eps: f32, rms_eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        crate::dsv4::hc_block(streams, seq, hc, h, fn_w, base, scale, iters, hc_eps, rms_eps)
    }

    /// CSA compressor + Lightning-Indexer → `(compressed_kv [T,hd], block_bias_vis
    /// [S,T])`. Seam for the DSA GPU trio (`dsv4_dsa_compress` + `dsv4_dsa_index_score`
    /// + `dsv4_dsa_topk`). DEFAULT = the bit-exact host oracle
    /// [`crate::dsv4_dsa::csa_compressor`] (CpuMv/CachedCpuMv unchanged → oracle
    /// preserved); [`Dsv4GpuStage`] overrides it to run the trio GPU-resident behind
    /// `VLLM_VULKAN_DSV4_DSA_GPU=1` (the matvecs stay host — compressor weights are
    /// not resident — while pool+RoPE, index-score and top-512 move to the device).
    #[allow(clippy::too_many_arguments)]
    fn csa_compressor(
        &mut self, hs: &[f32], q_residual: &[f32], s: usize, h: usize, hd: usize, m: usize,
        positions: &[usize], kv_proj: &[f32], gate_proj: &[f32], position_bias: &[f32],
        kv_norm: &[f32], eps: f32, inv_freq: &[f32], scaling: f32, q_lora: usize,
        ix_nh: usize, ix_hd: usize, index_topk: usize, ix: &IndexerWeights,
    ) -> (Vec<f32>, Vec<i32>) {
        crate::dsv4_dsa::csa_compressor(
            hs, q_residual, s, h, hd, m, positions, kv_proj, gate_proj, position_bias,
            kv_norm, eps, inv_freq, scaling, q_lora, ix_nh, ix_hd, index_topk, ix,
        )
    }

    /// DECODE-path CSA compressor + Lightning-Indexer over the ALREADY-PROJECTED
    /// inputs (`kv`/`gate` outer `[s, 2*hd]`, indexer projections in `ixp`) → the
    /// resident-projection oracle [`crate::dsv4_dsa::csa_compressor_pre`]. This is the
    /// seam the rolling decode consumes (the outer/indexer matvecs already ran through
    /// the resident `mm` seam). DEFAULT = the bit-exact host oracle (CpuMv/CachedCpuMv
    /// preserved → argmax 11111 never at risk); [`Dsv4GpuStage`] overrides it to fold
    /// pool+RoPE (`dsv4_dsa_compress` ×2) → index-score (`dsv4_dsa_index_score`) →
    /// causal top-512 (`dsv4_dsa_topk`) into ONE command buffer (the resident DSA
    /// trio, no host readback between the four dispatches) behind
    /// `VLLM_VULKAN_DSV4_DSA_GPU=1`.
    #[allow(clippy::too_many_arguments)]
    fn csa_compressor_pre(
        &mut self, kv: &[f32], gate: &[f32], s: usize, hd: usize, m: usize,
        positions: &[usize], position_bias: &[f32], kv_norm: &[f32], eps: f32,
        inv_freq: &[f32], scaling: f32, ix_nh: usize, ix_hd: usize, index_topk: usize,
        ixp: &IndexerProj,
    ) -> (Vec<f32>, Vec<i32>) {
        crate::dsv4_dsa::csa_compressor_pre(
            kv, gate, s, hd, m, positions, position_bias, kv_norm, eps, inv_freq,
            scaling, ix_nh, ix_hd, index_topk, ixp,
        )
    }

    /// MoE / MLP block seam (`moe_layer_mv`). DEFAULT = the bit-exact host oracle
    /// (`CpuMv`/`CachedCpuMv` byte-identical → argmax-11111 oracle preserved).
    /// [`Dsv4GpuStage`] overrides it to run the routed+shared experts GPU-RESIDENT
    /// in ONE command buffer (M1: gate/up matvec → `dsv4_swiglu_clamp` →
    /// down matvec → `q35_moe_accum_batched`, router stays host), behind
    /// `VLLM_VULKAN_DSV4_1CB=1`. This is the biggest per-layer submit sink
    /// ((num_experts_per_tok*3 + 3) serial matvecs + host SwiGLU/token → one CB).
    /// Any dispatch error falls back to the host oracle (correctness never at risk).
    fn moe_block(
        &mut self, cfg: &Dsv4Config, li: usize, mt: MlpType, x: &[f32], ids: &[u32],
    ) -> Vec<f32>
    where
        Self: Sized,
    {
        moe_layer_mv(cfg, li, mt, x, ids, self)
    }

    /// M2b seam: the DECODE MLA attention TAIL — the per-head eager softmax (sink +
    /// sliding-window + compressed-KV `block_bias`), the output-rope conjugate, the
    /// grouped block-diagonal `wo_a` and the `wo_b` projection — over the resident
    /// weights. This is exactly the block that stays HOST today (the 64-head softmax
    /// + the 33M-MAC grouped `wo_a`, per token × 43 layers). DEFAULT = the bit-exact
    /// host path ([`attn_tail_host_proj`] + `self.mm(wo_b)`), so `CpuMv`/`CachedCpuMv`
    /// are byte-identical (argmax-11111 oracle preserved). [`Dsv4GpuStage`] overrides
    /// it to record `dsv4_mla_softmax` → grouped `wo_a` matvecs → `wo_b` matvec into
    /// ONE command buffer (single readback) behind `VLLM_VULKAN_DSV4_1CB=1`.
    /// `q` = `[nh*hd]` post-q-rope, `kv_sliding` = `[t1*hd]`, `compressed_kv` =
    /// `[t_comp*hd]`, `block_bias_last` = `[t_comp]` additive, `cos`/`sin` = the
    /// output-rope `[rope_dim/2]` at this token's position.
    #[allow(clippy::too_many_arguments)]
    fn attn_tail(
        &mut self, p: &str, q: &[f32], kv_sliding: &[f32], compressed_kv: &[f32],
        block_bias_last: &[f32], sinks: &[f32], cos: &[f32], sin: &[f32],
        nh: usize, hd: usize, h: usize, g: usize, olr: usize, sw: usize, t1: usize, rope_dim: usize,
    ) -> Vec<f32>
    where
        Self: Sized,
    {
        let per_g = (nh * hd) / g;
        let w_o_a = self.dq_linear(&format!("{p}.attn.wo_a"), g * olr, per_g);
        let proj = attn_tail_host_proj(
            q, kv_sliding, compressed_kv, block_bias_last, sinks, cos, sin,
            nh, hd, g, olr, sw, t1, rope_dim, &w_o_a,
        );
        self.mm(&format!("{p}.attn.wo_b"), &proj, 1, g * olr, h)
    }

    /// Whether this backend runs the resident HC-site span (the attn-site
    /// `hc_residual_mix` + the ffn-site mHC recorded into the resident attn-tail CB,
    /// riding its submit). DEFAULT = false → the plain host mHC sequence. Only
    /// [`Dsv4GpuStage`] returns true, and only under `VLLM_VULKAN_DSV4_HC_RESIDENT=1`
    /// + `VLLM_VULKAN_DSV4_1CB=1`.
    fn hc_resident_fused(&self) -> bool {
        false
    }

    /// The DECODE attention TAIL followed by the trailing attn-site `hc_residual_mix`
    /// and the ffn-site mHC, as ONE step (see [`dsv4_hc_resident_enabled`]). Returns
    /// `(streams', ffn_post, ffn_comb, ffn_collapsed)` — the caller applies the
    /// (cheap, 1-row) rmsnorm to `ffn_collapsed`. The DEFAULT reproduces the host
    /// decode sequence op-for-op (`attn_tail` → `hc_residual_mix` → `hc_block`), so
    /// `CpuMv`/`CachedCpuMv` and the flag-OFF path stay BYTE-IDENTICAL to the current
    /// [`decoder_layer_decode`]. [`Dsv4GpuStage`] overrides it to record the two
    /// trailing HC ops into the resident attn-tail command buffer.
    #[allow(clippy::too_many_arguments)]
    fn attn_tail_hc(
        &mut self, p: &str, q: &[f32], kv_sliding: &[f32], compressed_kv: &[f32],
        block_bias_last: &[f32], sinks: &[f32], cos: &[f32], sin: &[f32],
        nh: usize, hd: usize, h: usize, g: usize, olr: usize, sw: usize, t1: usize, rope_dim: usize,
        post_a: &[f32], comb_a: &[f32], streams: &[f32],
        ffn_fn: &[f32], ffn_base: &[f32], ffn_scale: &[f32],
        hc: usize, iters: usize, hc_eps: f32, rms_eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)
    where
        Self: Sized,
    {
        let attn_out = self.attn_tail(
            p, q, kv_sliding, compressed_kv, block_bias_last, sinks, cos, sin,
            nh, hd, h, g, olr, sw, t1, rope_dim,
        );
        let streams2 = hc_residual_mix(post_a, &attn_out, comb_a, streams, 1, hc, h);
        let (post_f, comb_f, coll_f) =
            self.hc_block(&streams2, 1, hc, h, ffn_fn, ffn_base, ffn_scale, iters, hc_eps, rms_eps);
        (streams2, post_f, comb_f, coll_f)
    }
}

/// CPU backend: matmuls via `cpu_matmul` over a dequantized [`Dsv4Src`]. Makes
/// `forward_mv` reproduce `dsv4_forward` (validated offline vs the tiny fixture).
pub struct CpuMv<'a, S: Dsv4Src> {
    pub src: &'a S,
}

impl<'a, S: Dsv4Src> Mv for CpuMv<'a, S> {
    fn mm(&mut self, name: &str, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        crate::model::cpu_matmul(x, &self.src.linear(name, out_f, in_f), s, in_f, out_f)
    }
    fn mm_expert(&mut self, name: &str, e: usize, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        crate::model::cpu_matmul(x, &self.src.expert(name, e, out_f, in_f), s, in_f, out_f)
    }
    fn dq_linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32> {
        self.src.linear(name, out_f, in_f)
    }
    fn dense(&self, name: &str) -> Vec<f32> {
        self.src.dense(name)
    }
    fn dense_i64(&self, name: &str) -> Vec<i64> {
        self.src.dense_i64(name)
    }
    fn embed_rows(&self, ids: &[u32], vocab: usize, h: usize) -> Vec<f32> {
        self.src.embed_rows(ids, vocab, h)
    }
}

/// CPU backend with a per-layer expert/linear dequant cache. Bit-identical to
/// [`CpuMv`] (same `cpu_matmul` over the same dequant bytes) but memoizes each
/// unique dequantized weight for the current layer — the "expert-UNION dedup" M1's
/// `dsv4_forward::moe_layer` does — so a real-ckpt `forward_mv` runs at M1 speed
/// instead of re-dequantizing the ~96MB experts once per token. Caches evict when
/// the layer index in the tensor name advances (peak ≈ one layer's hit experts),
/// so DRAM stays bounded over the 43-layer forward.
pub struct CachedCpuMv<'a, S: Dsv4Src> {
    pub src: &'a S,
    exp: std::cell::RefCell<HashMap<String, Vec<f32>>>,
    lin: std::cell::RefCell<HashMap<String, Vec<f32>>>,
    cur_layer: std::cell::RefCell<Option<usize>>,
}

impl<'a, S: Dsv4Src> CachedCpuMv<'a, S> {
    pub fn new(src: &'a S) -> Self {
        CachedCpuMv {
            src,
            exp: std::cell::RefCell::new(HashMap::new()),
            lin: std::cell::RefCell::new(HashMap::new()),
            cur_layer: std::cell::RefCell::new(None),
        }
    }
    /// Parse the `li` in `model.layers.{li}.…`; `None` for non-layer tensors
    /// (`lm_head`, `model.norm`, embed) which never trigger eviction.
    fn layer_of(name: &str) -> Option<usize> {
        let rest = name.strip_prefix("model.layers.")?;
        let end = rest.find('.')?;
        rest[..end].parse::<usize>().ok()
    }
    /// Evict both caches when the layer advances (keeps DRAM bounded to one layer).
    fn touch(&self, name: &str) {
        if let Some(li) = Self::layer_of(name) {
            let mut cur = self.cur_layer.borrow_mut();
            if *cur != Some(li) {
                self.exp.borrow_mut().clear();
                self.lin.borrow_mut().clear();
                *cur = Some(li);
            }
        }
    }
    fn cached_linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32> {
        self.touch(name);
        if let Some(v) = self.lin.borrow().get(name) {
            return v.clone();
        }
        let v = self.src.linear(name, out_f, in_f);
        self.lin.borrow_mut().insert(name.to_string(), v.clone());
        v
    }
    fn cached_expert(&self, name: &str, e: usize, out_f: usize, in_f: usize) -> Vec<f32> {
        self.touch(name);
        let key = format!("{name}#{e}");
        if let Some(v) = self.exp.borrow().get(&key) {
            return v.clone();
        }
        let v = self.src.expert(name, e, out_f, in_f);
        self.exp.borrow_mut().insert(key, v.clone());
        v
    }
}

impl<'a, S: Dsv4Src> Mv for CachedCpuMv<'a, S> {
    fn mm(&mut self, name: &str, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        crate::model::cpu_matmul(x, &self.cached_linear(name, out_f, in_f), s, in_f, out_f)
    }
    fn mm_expert(&mut self, name: &str, e: usize, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        crate::model::cpu_matmul(x, &self.cached_expert(name, e, out_f, in_f), s, in_f, out_f)
    }
    fn dq_linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32> {
        self.cached_linear(name, out_f, in_f)
    }
    fn dense(&self, name: &str) -> Vec<f32> {
        self.src.dense(name)
    }
    fn dense_i64(&self, name: &str) -> Vec<i64> {
        self.src.dense_i64(name)
    }
    fn embed_rows(&self, ids: &[u32], vocab: usize, h: usize) -> Vec<f32> {
        self.src.embed_rows(ids, vocab, h)
    }
}

// ============================================================================
// GPU-resident stage
// ============================================================================

/// A resident packed MLX-affine linear on GPU (`mul_mat_vec_mlx{2,6,8}` bindings:
/// packed u32 words, f32 scales, f32 biases). `bits` picks the base shader.
struct Qbuf {
    p: compute::Buffer,
    s: compute::Buffer,
    b: compute::Buffer,
    out: usize,
    inn: usize,
    bits: usize,
    gs: usize,
}

fn base_shader(bits: usize) -> &'static str {
    match bits {
        2 => "mul_mat_vec_mlx2repack_f32_f32",
        6 => "mul_mat_vec_mlx6_f32_f32",
        8 => "mul_mat_vec_mlx8_f32_f32",
        _ => panic!("dsv4_gpu: unsupported bits {bits}"),
    }
}

/// GPU matvec dispatch parameters shared by every resident linear (bs=64, rows=2 —
/// the geometry `debug_dsv4_verify` gates cos=1.0 across mlx2/6/8).
const BS: u32 = 64;
const ROWS: u32 = 2;

/// One PP window of DSV4 held GPU quant-resident, decoded a token at a time. Owns
/// its own `ComputeEngine`; the resident maps hold the layer window's packed
/// weights, and `src` (mmapped, lazy) serves the host-seam tensors + the small
/// host-dequant weights (compressor / grouped wo_a / router gate / embed).
pub struct Dsv4GpuStage {
    /// Resident 2D linears keyed by checkpoint name (`{p}.attn.wq_a`, `lm_head`, …).
    lin: HashMap<String, Qbuf>,
    /// Resident routed experts keyed `"{name}#{e}"`.
    exp: HashMap<String, Qbuf>,
    eng: compute::ComputeEngine,
    _dev: device::ComputeDevice,
    /// Kept alive so `compile_variant_timeout` can (re)build a matvec variant.
    _shader_spvs: HashMap<String, Vec<u8>>,
    src: Dsv4RealSrc,
    cfg: Dsv4Config,
    pub layer_start: usize,
    pub layer_end: usize,
    pub first: bool,
    pub last: bool,
    /// Persistent rolling decode cache for this stage's layer window (lazily built
    /// on the first `decode_step_stage`; reset between sequences).
    decode_cache: Option<Dsv4DecodeCache>,
    /// LEVER: resident f32 MoE-router gate weights keyed by `{p}.ffn.gate.weight`,
    /// pre-dequantized ONCE at load (`VLLM_VULKAN_DSV4_RESIDENT_ROUTER`, default-OFF)
    /// so the per-token router stops re-dequantizing the bf16 gate from mmap. Empty
    /// (and thus byte-identical to the host dequant) when the flag is off.
    router_gate_f32: HashMap<String, Vec<f32>>,
}

impl Dsv4GpuStage {
    /// Memory-lean loader: stream the checkpoint layer window, uploading each
    /// layer's quantized matvec weights GPU-resident (packed) then dropping the host
    /// copy — mirroring `ling_gpu::LingGpuStage::from_ckpt_streamed`, the LOAD-OOM
    /// cure. `load_edges` uploads `lm_head` (last stage) resident; `embed` stays a
    /// host row-gather. Peak DRAM ≈ resident packed GTT + mmap pages (never a full
    /// dequant materialization).
    pub fn from_ckpt_streamed(
        ckpt_dir: &str,
        cfg: &Dsv4Config,
        layer_start: usize,
        layer_end: usize,
        load_edges: bool,
        device_idx: usize,
    ) -> Result<Dsv4GpuStage, String> {
        let src = Dsv4RealSrc::open(ckpt_dir)?;
        let (mut eng, dev, shader_spvs) = make_engine(device_idx)?;
        // Pre-compile the three resident matvec variants once.
        for bits in [2usize, 6, 8] {
            ensure_variant(&mut eng, &shader_spvs, base_shader(bits))?;
        }

        let h = cfg.mla.hidden_size;
        let nh = cfg.mla.num_attention_heads;
        let hd = cfg.mla.head_dim;
        let ql = cfg.mla.q_lora_rank;
        let olr = cfg.mla.o_lora_rank;
        let g = cfg.mla.o_groups;
        let ii = cfg.moe_intermediate_size;
        let ne = cfg.num_local_experts;

        let mut lin: HashMap<String, Qbuf> = HashMap::new();
        let mut exp: HashMap<String, Qbuf> = HashMap::new();
        let mut router_gate_f32: HashMap<String, Vec<f32>> = HashMap::new();

        for li in layer_start..layer_end {
            let p = format!("model.layers.{li}");
            // MLA 6-bit projections that are plain matvecs (grouped wo_a stays host).
            up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.wq_a"), ql, h)?;
            up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.wq_b"), nh * hd, ql)?;
            up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.wkv"), hd, h)?;
            up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.wo_b"), h, g * olr)?;
            // M2b: grouped block-diagonal wo_a → g per-group resident matvec
            // weights `wo_a#g{gg}` ([olr, per_g], per_g = nh*hd/g), for the resident
            // attention tail. Output-row slicing is packing-safe (see raw_row_block).
            {
                let wo_a_name = format!("{p}.attn.wo_a");
                if src.is_quant(&wo_a_name) {
                    let per_g = (nh * hd) / g;
                    let rq = src.raw_linear(&wo_a_name, g * olr, per_g);
                    for gg in 0..g {
                        let sub = raw_row_block(&rq, gg * olr, (gg + 1) * olr);
                        lin.insert(format!("{wo_a_name}#g{gg}"), alloc_qbuf(&mut eng, sub)?);
                    }
                }
            }
            // LEVER #2: resident compressor/indexer projection weights (per layer
            // type), so decode stops re-dequantizing them from mmap every token.
            // Flag-gated (default-OFF); when off these stay non-resident and the
            // compressor `mm` falls back to the byte-identical host dequant path.
            if dsv4_resident_compressor_enabled() {
                let ihd = cfg.index_head_dim;
                let inh = cfg.index_n_heads;
                match cfg.layer_types[li] {
                    LayerType::HeavilyCompressed => {
                        up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.compressor.wkv"), hd, h)?;
                        up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.compressor.wgate"), hd, h)?;
                    }
                    LayerType::CompressedSparse => {
                        up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.compressor.wkv"), 2 * hd, h)?;
                        up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.compressor.wgate"), 2 * hd, h)?;
                        up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.indexer.compressor.wkv"), 2 * ihd, h)?;
                        up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.indexer.compressor.wgate"), 2 * ihd, h)?;
                        up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.indexer.wq_b"), inh * ihd, ql)?;
                        up_lin(&mut eng, &src, &mut lin, &format!("{p}.attn.indexer.weights_proj"), inh, h)?;
                    }
                    LayerType::Sliding => {}
                }
            }
            // shared expert (8-bit).
            up_lin(&mut eng, &src, &mut lin, &format!("{p}.ffn.shared_experts.gate_proj"), ii, h)?;
            up_lin(&mut eng, &src, &mut lin, &format!("{p}.ffn.shared_experts.up_proj"), ii, h)?;
            up_lin(&mut eng, &src, &mut lin, &format!("{p}.ffn.shared_experts.down_proj"), h, ii)?;
            // routed experts (2-bit), all E resident (routing is dynamic per token).
            for e in 0..ne {
                up_exp(&mut eng, &src, &mut exp, &format!("{p}.ffn.switch_mlp.gate_proj"), e, ii, h)?;
                up_exp(&mut eng, &src, &mut exp, &format!("{p}.ffn.switch_mlp.up_proj"), e, ii, h)?;
                up_exp(&mut eng, &src, &mut exp, &format!("{p}.ffn.switch_mlp.down_proj"), e, h, ii)?;
            }
            // LEVER: pre-dequant the bf16 MoE-router gate ONCE at load (every layer is
            // MoE/HashMoe → has a `ffn.gate.weight`). Tiny host DRAM ([ne,h]×4B, NOT GTT);
            // the per-token router then reads it resident instead of re-dequantizing.
            if dsv4_resident_router_enabled() {
                let gname = format!("{p}.ffn.gate.weight");
                let gw = src.linear(&gname, ne, h);
                router_gate_f32.insert(gname, gw);
            }
        }

        let last = layer_end == cfg.num_hidden_layers;
        if load_edges && last {
            up_lin(&mut eng, &src, &mut lin, "lm_head", cfg.vocab_size, h)?;
        }

        Ok(Dsv4GpuStage {
            lin, exp, eng, _dev: dev, _shader_spvs: shader_spvs, src, cfg: cfg.clone(),
            layer_start, layer_end,
            first: layer_start == 0, last,
            decode_cache: None,
            router_gate_f32,
        })
    }

    /// Single-node (all 43 layers) GPU-resident forward → `[S, vocab]` logits.
    /// GATE 2a: last-position argmax must reproduce the golden 11111 (` Paris`).
    pub fn forward(&mut self, input_ids: &[u32]) -> Vec<f32> {
        let cfg = self.cfg.clone();
        forward_mv(&cfg, input_ids, self)
    }

    /// GATE 2a helper: last-position argmax + its logit.
    pub fn argmax_last(&mut self, input_ids: &[u32]) -> (u32, f32) {
        let logits = self.forward(input_ids);
        let vocab = self.cfg.vocab_size;
        argmax_last(&logits, input_ids.len(), vocab)
    }

    /// One PP-window PREFILL forward over this stage's resident layer window.
    /// First stage (`layer_start==0`) embeds `input_ids` and ignores `streams_in`;
    /// mid/last stages consume the `[seq, hc*h]` stream payload hopped from the
    /// previous stage. Returns `[seq, hc*h]` streams (mid) or `[seq, vocab]` logits
    /// (last stage). This is the distributed-prefill primitive `pp_dsv4.py` rings
    /// stage-to-stage; the matching decode primitive is [`Dsv4GpuStage::decode_step_stage`]
    /// (rolling KV + prefix-stable compressor window over the same resident weights).
    pub fn forward_pp_stage_prefill(
        &mut self,
        input_ids: &[u32],
        streams_in: Option<Vec<f32>>,
    ) -> WindowOut {
        let cfg = self.cfg.clone();
        let (ls, le) = (self.layer_start, self.layer_end);
        forward_mv_window(&cfg, input_ids, streams_in, ls, le, self)
    }

    /// One PP-window DECODE step over this stage's resident layer window, advancing
    /// the persistent rolling cache. First stage ingests `id` (embed); mid/last
    /// stages consume the `[1, hc*h]` streams hopped from the previous stage. Returns
    /// `[1, hc*h]` streams (mid) or `[vocab]` logits (last stage). This is the
    /// distributed-decode primitive `pp_dsv4.py` rings stage-to-stage over vCCL —
    /// the GPU twin of [`decode_step_window`], reusing the (cos=1.0) mlx `mm`
    /// kernels with ZERO new kernel code (decode is generic over the [`Mv`] seam).
    pub fn decode_step_stage(&mut self, id: u32, streams_in: Option<Vec<f32>>) -> WindowOut {
        // M0 seam: attempt the resident 1-CB span (flag-gated, DEFAULT-OFF). The
        // stub returns None today → byte-identical host fallthrough below.
        if let Some(out) = self.try_decode_step_resident(id, &streams_in) {
            return out;
        }
        let cfg = self.cfg.clone();
        let (ls, le) = (self.layer_start, self.layer_end);
        let mut cache = self
            .decode_cache
            .take()
            .unwrap_or_else(|| Dsv4DecodeCache::new_window(&cfg, ls, le));
        let out = decode_step_window(&cfg, id, streams_in, &mut cache, self);
        self.decode_cache = Some(cache);
        out
    }

    /// Drop the rolling decode cache (call between independent sequences).
    pub fn reset_decode_cache(&mut self) {
        self.decode_cache = None;
    }

    /// Tokens ingested by this stage's decode cache so far (== next position).
    pub fn decode_pos(&self) -> usize {
        self.decode_cache.as_ref().map(|c| c.len()).unwrap_or(0)
    }
}

impl Mv for Dsv4GpuStage {
    fn mm(&mut self, name: &str, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        if let Some(q) = self.lin.get(name) {
            debug_assert_eq!((q.inn, q.out), (in_f, out_f), "resident {name} dims");
            return gpu_matvec_rows(&mut self.eng, q, x, s).expect("gpu matvec");
        }
        // Non-resident (e.g. plain bf16) → host dequant + cpu_matmul.
        crate::model::cpu_matmul(x, &self.src.linear(name, out_f, in_f), s, in_f, out_f)
    }
    fn mm_expert(&mut self, name: &str, e: usize, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
        let key = format!("{name}#{e}");
        if let Some(q) = self.exp.get(&key) {
            debug_assert_eq!((q.inn, q.out), (in_f, out_f), "resident expert {key} dims");
            return gpu_matvec_rows(&mut self.eng, q, x, s).expect("gpu matvec expert");
        }
        crate::model::cpu_matmul(x, &self.src.expert(name, e, out_f, in_f), s, in_f, out_f)
    }
    fn dq_linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32> {
        self.src.linear(name, out_f, in_f)
    }
    fn resident_router_gate(&self, name: &str) -> Option<&[f32]> {
        self.router_gate_f32.get(name).map(|v| v.as_slice())
    }
    fn dense(&self, name: &str) -> Vec<f32> {
        self.src.dense(name)
    }
    fn dense_i64(&self, name: &str) -> Vec<i64> {
        self.src.dense_i64(name)
    }
    fn embed_rows(&self, ids: &[u32], vocab: usize, h: usize) -> Vec<f32> {
        self.src.embed_rows(ids, vocab, h)
    }
    /// GPU-resident mHC: dispatch `dsv4_hyper_connection` (the two-float kernel,
    /// cos=1.0 per `debug_dsv4_hc`) in place of the host `hc_block`. OPT-IN via
    /// `VLLM_VULKAN_DSV4_HC_GPU=1`; the DEFAULT is the bit-exact host oracle (see
    /// `hc_gpu_enabled` for the ITEM A NO-GO rationale — on a host-orchestrated
    /// forward this only ADDS 86 submit/fence ops/token with no resident CB to fuse
    /// into). Any dispatch error also falls back to the oracle so correctness is
    /// never at risk.
    fn hc_block(
        &mut self, streams: &[f32], seq: usize, hc: usize, h: usize,
        fn_w: &[f32], base: &[f32], scale: &[f32], iters: usize, hc_eps: f32, rms_eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        if !hc_gpu_enabled() {
            return crate::dsv4::hc_block(streams, seq, hc, h, fn_w, base, scale, iters, hc_eps, rms_eps);
        }
        match gpu_hc_block(
            &mut self.eng, &self._shader_spvs, streams, seq, hc, h, fn_w, base, scale, iters, hc_eps, rms_eps,
        ) {
            Ok(t) => t,
            Err(_) => crate::dsv4::hc_block(streams, seq, hc, h, fn_w, base, scale, iters, hc_eps, rms_eps),
        }
    }

    /// GPU-resident CSA compressor + Lightning-Indexer (behind
    /// `VLLM_VULKAN_DSV4_DSA_GPU=1`). Replicates [`crate::dsv4_dsa::csa_compressor`]'s
    /// dataflow but runs pool+RoPE (`dsv4_dsa_compress`), index-score
    /// (`dsv4_dsa_index_score`) and causal top-512 (`dsv4_dsa_topk`) on the device.
    /// The compressor/indexer matvecs stay host (`cpu_matmul` over dequantized
    /// weights — those tensors are not GPU-resident). Any dispatch error falls back
    /// to the host oracle; disabled → the exact host default (correctness never at
    /// risk). NOTE: only the CSA layers route here; HCA is single-series (the
    /// compress kernel is Ca/Cb two-series only) and stays host.
    #[allow(clippy::too_many_arguments)]
    fn csa_compressor(
        &mut self, hs: &[f32], q_residual: &[f32], s: usize, h: usize, hd: usize, m: usize,
        positions: &[usize], kv_proj: &[f32], gate_proj: &[f32], position_bias: &[f32],
        kv_norm: &[f32], eps: f32, inv_freq: &[f32], scaling: f32, q_lora: usize,
        ix_nh: usize, ix_hd: usize, index_topk: usize, ix: &IndexerWeights,
    ) -> (Vec<f32>, Vec<i32>) {
        let host = || crate::dsv4_dsa::csa_compressor(
            hs, q_residual, s, h, hd, m, positions, kv_proj, gate_proj, position_bias,
            kv_norm, eps, inv_freq, scaling, q_lora, ix_nh, ix_hd, index_topk, ix,
        );
        if !dsa_gpu_enabled() {
            return host();
        }
        match self.gpu_csa_compressor(
            hs, q_residual, s, h, hd, m, positions, kv_proj, gate_proj, position_bias,
            kv_norm, eps, inv_freq, scaling, q_lora, ix_nh, ix_hd, index_topk, ix,
        ) {
            Ok(v) => {
                // M3 instrument: per-CSA-layer top-512-set-vs-oracle A/B (does NOT
                // alter the returned GPU selection).
                if dsa_debug_enabled() {
                    let (ckv_h, vis_h) = host();
                    dsa_dump_ab(&v.0, &v.1, &ckv_h, &vis_h, s);
                }
                v
            }
            Err(_) => host(),
        }
    }

    /// DECODE-path CSA compressor over resident-projected inputs. Records the DSA
    /// trio (`dsv4_dsa_compress` ×2 → `index_score` → `topk`) into ONE command
    /// buffer behind `VLLM_VULKAN_DSV4_DSA_GPU=1` (the compress/index/top-512 MATH
    /// that otherwise runs HOST in [`crate::dsv4_dsa::csa_compressor_pre`], the
    /// ~compressor-bucket sink). Byte-identical dataflow to [`Self::gpu_csa_compressor`]
    /// (validated ckv=1.0 / argmax on-node); disabled or dispatch-error → the exact
    /// host oracle (correctness never at risk).
    #[allow(clippy::too_many_arguments)]
    fn csa_compressor_pre(
        &mut self, kv: &[f32], gate: &[f32], s: usize, hd: usize, m: usize,
        positions: &[usize], position_bias: &[f32], kv_norm: &[f32], eps: f32,
        inv_freq: &[f32], scaling: f32, ix_nh: usize, ix_hd: usize, index_topk: usize,
        ixp: &IndexerProj,
    ) -> (Vec<f32>, Vec<i32>) {
        let host = || crate::dsv4_dsa::csa_compressor_pre(
            kv, gate, s, hd, m, positions, position_bias, kv_norm, eps, inv_freq,
            scaling, ix_nh, ix_hd, index_topk, ixp,
        );
        if !dsa_gpu_enabled() {
            return host();
        }
        match self.gpu_csa_compressor_pre(
            kv, gate, s, hd, m, positions, position_bias, kv_norm, eps, inv_freq,
            scaling, ix_nh, ix_hd, index_topk, ixp,
        ) {
            Ok(v) => {
                if dsa_debug_enabled() {
                    let (ckv_h, vis_h) = host();
                    dsa_dump_ab(&v.0, &v.1, &ckv_h, &vis_h, s);
                }
                v
            }
            Err(_) => host(),
        }
    }

    /// M1: GPU-resident MoE span (behind `VLLM_VULKAN_DSV4_1CB=1`). Router stays
    /// host (tiny E-logit top-k); the routed+shared expert compute — the biggest
    /// per-layer submit sink — is recorded into ONE command buffer per token. Any
    /// dispatch error falls back to the bit-exact host `moe_layer_mv` (argmax
    /// 11111 never at risk); DEFAULT-OFF pins the host oracle.
    fn moe_block(
        &mut self, cfg: &Dsv4Config, li: usize, mt: MlpType, x: &[f32], ids: &[u32],
    ) -> Vec<f32> {
        if !dsv4_1cb_enabled() {
            return moe_layer_mv(cfg, li, mt, x, ids, self);
        }
        match self.moe_resident(cfg, li, mt, x, ids) {
            Ok(v) => v,
            Err(_) => moe_layer_mv(cfg, li, mt, x, ids, self),
        }
    }

    /// M2b: GPU-resident decode MLA attention tail (behind `VLLM_VULKAN_DSV4_1CB=1`).
    /// Records `dsv4_mla_softmax` (per-head eager softmax + output-rope) → grouped
    /// block-diagonal `wo_a` matvecs → `wo_b` matvec into ONE command buffer, single
    /// readback. Any dispatch/lookup error falls back to the bit-exact host tail
    /// ([`attn_tail_host_proj`] + `self.mm(wo_b)`); DEFAULT-OFF pins the host path.
    #[allow(clippy::too_many_arguments)]
    fn attn_tail(
        &mut self, p: &str, q: &[f32], kv_sliding: &[f32], compressed_kv: &[f32],
        block_bias_last: &[f32], sinks: &[f32], cos: &[f32], sin: &[f32],
        nh: usize, hd: usize, h: usize, g: usize, olr: usize, sw: usize, t1: usize, rope_dim: usize,
    ) -> Vec<f32> {
        if dsv4_1cb_enabled() {
            if let Ok(v) = self.attn_tail_resident(
                p, q, kv_sliding, compressed_kv, block_bias_last, sinks, cos, sin,
                nh, hd, h, g, olr, sw, t1, rope_dim,
            ) {
                return v;
            }
        }
        let per_g = (nh * hd) / g;
        let w_o_a = self.src.linear(&format!("{p}.attn.wo_a"), g * olr, per_g);
        let proj = attn_tail_host_proj(
            q, kv_sliding, compressed_kv, block_bias_last, sinks, cos, sin,
            nh, hd, g, olr, sw, t1, rope_dim, &w_o_a,
        );
        self.mm(&format!("{p}.attn.wo_b"), &proj, 1, g * olr, h)
    }

    fn hc_resident_fused(&self) -> bool {
        dsv4_1cb_enabled() && dsv4_hc_resident_enabled()
    }

    /// Resident HC-site override: record the attn-site `hc_residual_mix` + the
    /// ffn-site mHC into the resident attn-tail command buffer (they ride its
    /// submit). Any dispatch/lookup error falls back to the byte-exact host sequence
    /// (`attn_tail` → `hc_residual_mix` → `hc_block`), so correctness is never at
    /// risk. Flag-OFF pins the host path via [`Mv::attn_tail_hc`]'s default.
    #[allow(clippy::too_many_arguments)]
    fn attn_tail_hc(
        &mut self, p: &str, q: &[f32], kv_sliding: &[f32], compressed_kv: &[f32],
        block_bias_last: &[f32], sinks: &[f32], cos: &[f32], sin: &[f32],
        nh: usize, hd: usize, h: usize, g: usize, olr: usize, sw: usize, t1: usize, rope_dim: usize,
        post_a: &[f32], comb_a: &[f32], streams: &[f32],
        ffn_fn: &[f32], ffn_base: &[f32], ffn_scale: &[f32],
        hc: usize, iters: usize, hc_eps: f32, rms_eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        if dsv4_1cb_enabled() && dsv4_hc_resident_enabled() {
            if let Ok(r) = self.attn_tail_resident_hc(
                p, q, kv_sliding, compressed_kv, block_bias_last, sinks, cos, sin,
                nh, hd, h, g, olr, sw, t1, rope_dim,
                post_a, comb_a, streams, ffn_fn, ffn_base, ffn_scale, hc, iters, hc_eps, rms_eps,
            ) {
                return r;
            }
        }
        // Fallback = the host sequence (== the trait default): tail → residual-mix → mHC.
        let attn_out = self.attn_tail(
            p, q, kv_sliding, compressed_kv, block_bias_last, sinks, cos, sin,
            nh, hd, h, g, olr, sw, t1, rope_dim,
        );
        let streams2 = hc_residual_mix(post_a, &attn_out, comb_a, streams, 1, hc, h);
        let (post_f, comb_f, coll_f) =
            self.hc_block(&streams2, 1, hc, h, ffn_fn, ffn_base, ffn_scale, iters, hc_eps, rms_eps);
        (streams2, post_f, comb_f, coll_f)
    }
}

impl Dsv4GpuStage {
    /// M1 resident MoE span body (see the `moe_block` override). Router = host
    /// (bit-exact to `moe_layer_mv`); the routed+shared expert compute is recorded
    /// into ONE command buffer per token: gate/up matvecs (`{base}_bs64_r2`) →
    /// `dsv4_swiglu_clamp` (routed: `swiglu_limit`; shared: +inf = plain SwiGLU) →
    /// down matvecs → `q35_moe_accum_batched` (routed-weighted + ungated shared via
    /// logit=30). Replaces `moe_layer_mv`'s `(top_k*3 + 3)` per-op submits/token
    /// with one submit. Errors bubble to the host fallback. NOTE: the SwiGLU is
    /// computed in f32 (vs the oracle's f64 `silu`), so per-element it is cos≈1 not
    /// bit-exact — the assembled-decode argmax gate (on-node) is authoritative.
    fn moe_resident(
        &mut self, cfg: &Dsv4Config, li: usize, mt: MlpType, x: &[f32], ids: &[u32],
    ) -> Result<Vec<f32>, String> {
        let h = cfg.mla.hidden_size;
        let ii = cfg.moe_intermediate_size;
        let ne = cfg.num_local_experts;
        let tk = cfg.num_experts_per_tok;
        let seq = ids.len();
        let limit = cfg.swiglu_limit;
        let p = format!("model.layers.{li}");

        // ---- router: HOST (bit-exact to moe_layer_mv) ----
        let _router_t = dsv4_prof::start();
        // LEVER: resident f32 gate (VLLM_VULKAN_DSV4_RESIDENT_ROUTER) skips the per-token
        // bf16→f32 dequant; flag-OFF falls back to the byte-identical per-token dequant.
        let gate_name = format!("{p}.ffn.gate.weight");
        let gate_owned;
        let gate_w: &[f32] = match self.router_gate_f32.get(&gate_name) {
            Some(w) => w,
            None => {
                gate_owned = self.src.linear(&gate_name, ne, h);
                &gate_owned
            }
        };
        let (idx, wts) = match mt {
            MlpType::Moe => {
                let corr = self.src.dense(&format!("{p}.ffn.gate.e_score_correction_bias"));
                topk_router(x, gate_w, &corr, seq, h, ne, tk, cfg.routed_scaling_factor, cfg.norm_topk_prob)
            }
            MlpType::HashMoe => {
                let tid2eid = self.src.dense_i64(&format!("{p}.ffn.gate.tid2eid"));
                hash_router(x, gate_w, &tid2eid, ids, seq, h, ne, tk, cfg.routed_scaling_factor, cfg.norm_topk_prob)
            }
        };
        dsv4_prof::stop(dsv4_prof::B::Router, 0, _router_t);
        let _moe_t = dsv4_prof::start();

        let gate_name = format!("{p}.ffn.switch_mlp.gate_proj");
        let up_name = format!("{p}.ffn.switch_mlp.up_proj");
        let down_name = format!("{p}.ffn.switch_mlp.down_proj");
        let sh_gate = format!("{p}.ffn.shared_experts.gate_proj");
        let sh_up = format!("{p}.ffn.shared_experts.up_proj");
        let sh_down = format!("{p}.ffn.shared_experts.down_proj");

        // ---- resident scratch (allocated once, reused per token) ----
        let a = |eng: &mut compute::ComputeEngine, n: usize| eng.alloc_host_coherent_storage((n * 4).max(4) as u64);
        let xbuf = a(&mut self.eng, h)?;
        let gate_all = a(&mut self.eng, tk * ii)?;
        let up_all = a(&mut self.eng, tk * ii)?;
        let hid_all = a(&mut self.eng, tk * ii)?;
        let down_all = a(&mut self.eng, tk * h)?;
        let sg = a(&mut self.eng, ii)?;
        let su = a(&mut self.eng, ii)?;
        let sh_hid = a(&mut self.eng, ii)?;
        let shared_out = a(&mut self.eng, h)?;
        let scores = a(&mut self.eng, tk)?;
        let logits = a(&mut self.eng, 1)?;
        let outbuf = a(&mut self.eng, h)?;
        logits.write(&f32_slice_to_bytes(&[30.0f32]))?; // sigmoid(30)==1 → ungated shared

        let clamp_pc = dsv4_swiglu_clamp_pc(ii, limit);
        let inf_pc = dsv4_swiglu_clamp_pc(ii, f32::INFINITY);
        let acc_pc = q35_moe_accum_batched_pc(1, h, tk);
        let swg_wg = ((ii as u32) + 511) / 512;
        let acc_wg = ((h as u32) + 255) / 256;

        let mut out = vec![0f32; seq * h];
        for ti in 0..seq {
            let sc: Vec<f32> = (0..tk).map(|j| wts[ti * tk + j]).collect();
            scores.write(&f32_slice_to_bytes(&sc))?;
            xbuf.write(&f32_slice_to_bytes(&x[ti * h..(ti + 1) * h]))?;

            let cb = self.eng.begin_batch()?;
            // Stage A — gate/up matvecs (routed + shared), independent → no barriers.
            for j in 0..tk {
                let e = idx[ti * tk + j];
                let gq = self.exp.get(&format!("{gate_name}#{e}")).ok_or_else(|| format!("resident gate {gate_name}#{e}"))?;
                dsv4r_rec_mv(&mut self.eng, cb, gq, &xbuf, 0, &gate_all, (j * ii * 4) as u64)?;
                let uq = self.exp.get(&format!("{up_name}#{e}")).ok_or_else(|| format!("resident up {up_name}#{e}"))?;
                dsv4r_rec_mv(&mut self.eng, cb, uq, &xbuf, 0, &up_all, (j * ii * 4) as u64)?;
            }
            let sgq = self.lin.get(&sh_gate).ok_or("resident shared gate")?;
            dsv4r_rec_mv(&mut self.eng, cb, sgq, &xbuf, 0, &sg, 0)?;
            let suq = self.lin.get(&sh_up).ok_or("resident shared up")?;
            dsv4r_rec_mv(&mut self.eng, cb, suq, &xbuf, 0, &su, 0)?;
            self.eng.record_barrier_to(cb);

            // Stage B — SwiGLU-clamp (routed: swiglu_limit; shared: +inf = plain).
            for j in 0..tk {
                let off = (j * ii * 4) as u64;
                self.eng.record_to_off(
                    cb, "dsv4_swiglu_clamp",
                    &[(&gate_all, off), (&up_all, off), (&hid_all, off)],
                    &clamp_pc, (swg_wg, 1, 1),
                )?;
            }
            self.eng.record_to(cb, "dsv4_swiglu_clamp", &[&sg, &su, &sh_hid], &inf_pc, (swg_wg, 1, 1))?;
            self.eng.record_barrier_to(cb);

            // Stage C — down matvecs → down_all[j*h] / shared_out.
            for j in 0..tk {
                let e = idx[ti * tk + j];
                let dq = self.exp.get(&format!("{down_name}#{e}")).ok_or_else(|| format!("resident down {down_name}#{e}"))?;
                dsv4r_rec_mv(&mut self.eng, cb, dq, &hid_all, (j * ii * 4) as u64, &down_all, (j * h * 4) as u64)?;
            }
            let sdq = self.lin.get(&sh_down).ok_or("resident shared down")?;
            dsv4r_rec_mv(&mut self.eng, cb, sdq, &sh_hid, 0, &shared_out, 0)?;
            self.eng.record_barrier_to(cb);

            // Stage D — routed-weighted accum + ungated shared.
            self.eng.record_to(
                cb, "q35_moe_accum_batched",
                &[&down_all, &scores, &shared_out, &logits, &outbuf],
                &acc_pc, (acc_wg, 1, 1),
            )?;
            self.eng.submit_batch(cb)?;
            out[ti * h..(ti + 1) * h].copy_from_slice(&read_f32_buf(&outbuf, h));
        }

        for b in [xbuf, gate_all, up_all, hid_all, down_all, sg, su, sh_hid, shared_out, scores, logits, outbuf] {
            self.eng.return_to_pool(b);
        }
        dsv4_prof::stop(dsv4_prof::B::Moe, seq as u64, _moe_t);
        Ok(out)
    }

    /// M2b resident attention-tail span (see the `attn_tail` override). Records the
    /// per-head MLA eager softmax (`dsv4_mla_softmax`) → grouped block-diagonal
    /// `wo_a` (g resident matvecs) → `wo_b` (resident matvec) into ONE command
    /// buffer, reading the hidden back ONCE. Errors bubble to the host fallback.
    #[allow(clippy::too_many_arguments)]
    fn attn_tail_resident(
        &mut self, p: &str, q: &[f32], kv_sliding: &[f32], compressed_kv: &[f32],
        block_bias_last: &[f32], sinks: &[f32], cos: &[f32], sin: &[f32],
        nh: usize, hd: usize, h: usize, g: usize, olr: usize, sw: usize, t1: usize, rope_dim: usize,
    ) -> Result<Vec<f32>, String> {
        let per_g = (nh * hd) / g;
        let t_comp = compressed_kv.len() / hd;
        // Verify the resident weights are present BEFORE recording (so a miss falls
        // back cleanly without a half-recorded CB).
        for gg in 0..g {
            if !self.lin.contains_key(&format!("{p}.attn.wo_a#g{gg}")) {
                return Err(format!("resident wo_a#g{gg} missing"));
            }
        }
        if !self.lin.contains_key(&format!("{p}.attn.wo_b")) {
            return Err("resident wo_b missing".into());
        }

        let a = |eng: &mut compute::ComputeEngine, n: usize| eng.alloc_host_coherent_storage((n * 4).max(4) as u64);
        let qbuf = a(&mut self.eng, nh * hd)?;
        let kvbuf = a(&mut self.eng, (t1 * hd).max(1))?;
        let ckvbuf = a(&mut self.eng, (t_comp * hd).max(1))?;
        let bbbuf = a(&mut self.eng, t_comp.max(1))?;
        let sinkbuf = a(&mut self.eng, nh)?;
        let cosbuf = a(&mut self.eng, cos.len().max(1))?;
        let sinbuf = a(&mut self.eng, sin.len().max(1))?;
        let aobuf = a(&mut self.eng, nh * hd)?;
        let projbuf = a(&mut self.eng, g * olr)?;
        let outbuf = a(&mut self.eng, h)?;

        qbuf.write(&f32_slice_to_bytes(q))?;
        if t1 > 0 { kvbuf.write(&f32_slice_to_bytes(kv_sliding))?; }
        if t_comp > 0 {
            ckvbuf.write(&f32_slice_to_bytes(compressed_kv))?;
            bbbuf.write(&f32_slice_to_bytes(block_bias_last))?;
        }
        sinkbuf.write(&f32_slice_to_bytes(sinks))?;
        cosbuf.write(&f32_slice_to_bytes(cos))?;
        sinbuf.write(&f32_slice_to_bytes(sin))?;

        let cb = self.eng.begin_batch()?;
        // Stage A — per-head eager softmax + output-rope → ao [nh*hd].
        dsv4r_rec_mla_softmax(
            &mut self.eng, cb, &qbuf, &kvbuf, &ckvbuf, &bbbuf, &sinkbuf, &cosbuf, &sinbuf, &aobuf,
            nh, hd, t1, t_comp, sw, rope_dim,
        )?;
        self.eng.record_barrier_to(cb);
        // Stage B — grouped block-diagonal wo_a (g independent matvecs) → proj [g*olr].
        let groups: Vec<&Qbuf> = (0..g)
            .map(|gg| self.lin.get(&format!("{p}.attn.wo_a#g{gg}")).ok_or_else(|| format!("resident wo_a#g{gg}")))
            .collect::<Result<_, _>>()?;
        dsv4r_rec_wo_a_grouped(&mut self.eng, cb, &groups, &aobuf, &projbuf, per_g, olr)?;
        self.eng.record_barrier_to(cb);
        // Stage C — wo_b: proj → out [h].
        let wob = self.lin.get(&format!("{p}.attn.wo_b")).ok_or("resident wo_b")?;
        dsv4r_rec_mv(&mut self.eng, cb, wob, &projbuf, 0, &outbuf, 0)?;
        self.eng.submit_batch(cb)?;
        let out = read_f32_buf(&outbuf, h);

        for b in [qbuf, kvbuf, ckvbuf, bbbuf, sinkbuf, cosbuf, sinbuf, aobuf, projbuf, outbuf] {
            self.eng.return_to_pool(b);
        }
        Ok(out)
    }

    /// Resident attention-tail span WITH the trailing HC-site ops appended into the
    /// SAME command buffer (the [`Mv::attn_tail_hc`] GPU override): after the tail's
    /// `wo_b` produces the branch output `attn_out [h]` (device), record
    ///   `dsv4_hc_residual_mix`  (attn-site: post_a, attn_out, comb_a, streams → streams')
    ///   `dsv4_hyper_connection` (ffn-site mHC/Sinkhorn: streams' → ffn_post/comb/collapsed)
    /// then submit ONCE and read `(streams', ffn_post, ffn_comb, ffn_collapsed)` back
    /// in that single readback. The mHC therefore rides the tail's existing submit —
    /// no per-mHC round-trip (the `hc_gpu_enabled` NO-GO cure). Bindings/PC mirror
    /// [`dsv4r_rec_hc_residual_mix`]/[`dsv4r_rec_hc_block`] (offline-validated by
    /// `dsv4_hc_site_resident_mirror_matches_oracle`). Errors bubble to the host
    /// fallback (a half-recorded CB is never submitted — the resident-weight checks
    /// and the pipeline-ensure run before `begin_batch`).
    #[allow(clippy::too_many_arguments)]
    fn attn_tail_resident_hc(
        &mut self, p: &str, q: &[f32], kv_sliding: &[f32], compressed_kv: &[f32],
        block_bias_last: &[f32], sinks: &[f32], cos: &[f32], sin: &[f32],
        nh: usize, hd: usize, h: usize, g: usize, olr: usize, sw: usize, t1: usize, rope_dim: usize,
        post_a: &[f32], comb_a: &[f32], streams: &[f32],
        ffn_fn: &[f32], ffn_base: &[f32], ffn_scale: &[f32],
        hc: usize, iters: usize, hc_eps: f32, rms_eps: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let per_g = (nh * hd) / g;
        let t_comp = compressed_kv.len() / hd;
        // Verify the resident tail weights BEFORE recording (clean fallback, no
        // half-recorded CB).
        for gg in 0..g {
            if !self.lin.contains_key(&format!("{p}.attn.wo_a#g{gg}")) {
                return Err(format!("resident wo_a#g{gg} missing"));
            }
        }
        if !self.lin.contains_key(&format!("{p}.attn.wo_b")) {
            return Err("resident wo_b missing".into());
        }
        // Ensure the decode-shaped two-float mHC pipeline (spec BLOCK_SIZE = dsv4_hc_bs()).
        let hc_variant = dsv4_hc_variant();
        if !self.eng.has_pipeline(&hc_variant) {
            let spv = self._shader_spvs.get("dsv4_hyper_connection").map(|v| v.as_slice())
                .ok_or_else(|| "dsv4_hyper_connection SPIR-V missing".to_string())?;
            if !self.eng.compile_variant_timeout(&hc_variant, spv, &[(0, dsv4_hc_bs())], 2000)? {
                return Err("hc pipeline timeout".into());
            }
        }

        let a = |eng: &mut compute::ComputeEngine, n: usize| eng.alloc_host_coherent_storage((n * 4).max(4) as u64);
        // Tail scratch (== attn_tail_resident).
        let qbuf = a(&mut self.eng, nh * hd)?;
        let kvbuf = a(&mut self.eng, (t1 * hd).max(1))?;
        let ckvbuf = a(&mut self.eng, (t_comp * hd).max(1))?;
        let bbbuf = a(&mut self.eng, t_comp.max(1))?;
        let sinkbuf = a(&mut self.eng, nh)?;
        let cosbuf = a(&mut self.eng, cos.len().max(1))?;
        let sinbuf = a(&mut self.eng, sin.len().max(1))?;
        let aobuf = a(&mut self.eng, nh * hd)?;
        let projbuf = a(&mut self.eng, g * olr)?;
        let outbuf = a(&mut self.eng, h)?; // attn_out [h] (the `sub` of hc_residual_mix)
        // Trailing HC scratch.
        let postbuf = a(&mut self.eng, hc)?;
        let combbuf = a(&mut self.eng, hc * hc)?;
        let streamsbuf = a(&mut self.eng, hc * h)?;
        let fnbuf = a(&mut self.eng, ffn_fn.len())?;
        let basebuf = a(&mut self.eng, ffn_base.len())?;
        let scalebuf = a(&mut self.eng, ffn_scale.len())?;
        let streams2buf = a(&mut self.eng, hc * h)?; // hc_residual_mix output (mHC input)
        let npostbuf = a(&mut self.eng, hc)?;
        let ncombbuf = a(&mut self.eng, hc * hc)?;
        let ncollbuf = a(&mut self.eng, h)?;

        qbuf.write(&f32_slice_to_bytes(q))?;
        if t1 > 0 { kvbuf.write(&f32_slice_to_bytes(kv_sliding))?; }
        if t_comp > 0 {
            ckvbuf.write(&f32_slice_to_bytes(compressed_kv))?;
            bbbuf.write(&f32_slice_to_bytes(block_bias_last))?;
        }
        sinkbuf.write(&f32_slice_to_bytes(sinks))?;
        cosbuf.write(&f32_slice_to_bytes(cos))?;
        sinbuf.write(&f32_slice_to_bytes(sin))?;
        postbuf.write(&f32_slice_to_bytes(post_a))?;
        combbuf.write(&f32_slice_to_bytes(comb_a))?;
        streamsbuf.write(&f32_slice_to_bytes(streams))?;
        fnbuf.write(&f32_slice_to_bytes(ffn_fn))?;
        basebuf.write(&f32_slice_to_bytes(ffn_base))?;
        scalebuf.write(&f32_slice_to_bytes(ffn_scale))?;

        let cb = self.eng.begin_batch()?;
        // Stage A — per-head eager softmax + output-rope → ao [nh*hd].
        dsv4r_rec_mla_softmax(
            &mut self.eng, cb, &qbuf, &kvbuf, &ckvbuf, &bbbuf, &sinkbuf, &cosbuf, &sinbuf, &aobuf,
            nh, hd, t1, t_comp, sw, rope_dim,
        )?;
        self.eng.record_barrier_to(cb);
        // Stage B — grouped block-diagonal wo_a → proj [g*olr].
        let groups: Vec<&Qbuf> = (0..g)
            .map(|gg| self.lin.get(&format!("{p}.attn.wo_a#g{gg}")).ok_or_else(|| format!("resident wo_a#g{gg}")))
            .collect::<Result<_, _>>()?;
        dsv4r_rec_wo_a_grouped(&mut self.eng, cb, &groups, &aobuf, &projbuf, per_g, olr)?;
        self.eng.record_barrier_to(cb);
        // Stage C — wo_b: proj → attn_out [h].
        let wob = self.lin.get(&format!("{p}.attn.wo_b")).ok_or("resident wo_b")?;
        dsv4r_rec_mv(&mut self.eng, cb, wob, &projbuf, 0, &outbuf, 0)?;
        self.eng.record_barrier_to(cb);
        // Stage D — trailing attn-site hc_residual_mix: (post_a, attn_out, comb_a, streams) → streams'.
        dsv4r_rec_hc_residual_mix(
            &mut self.eng, cb, &postbuf, &outbuf, &combbuf, &streamsbuf, &streams2buf, 1, hc, h,
        )?;
        self.eng.record_barrier_to(cb);
        // Stage E — ffn-site mHC (Sinkhorn) over streams' → ffn_post/comb/collapsed.
        dsv4r_rec_hc_block(
            &mut self.eng, cb, &streams2buf, &fnbuf, &basebuf, &scalebuf,
            &npostbuf, &ncombbuf, &ncollbuf, 1, hc, h, iters, hc_eps, rms_eps,
        )?;
        self.eng.submit_batch(cb)?;
        let streams2 = read_f32_buf(&streams2buf, hc * h);
        let npost = read_f32_buf(&npostbuf, hc);
        let ncomb = read_f32_buf(&ncombbuf, hc * hc);
        let ncoll = read_f32_buf(&ncollbuf, h);

        for b in [
            qbuf, kvbuf, ckvbuf, bbbuf, sinkbuf, cosbuf, sinbuf, aobuf, projbuf, outbuf,
            postbuf, combbuf, streamsbuf, fnbuf, basebuf, scalebuf, streams2buf, npostbuf, ncombbuf, ncollbuf,
        ] {
            self.eng.return_to_pool(b);
        }
        Ok((streams2, npost, ncomb, ncoll))
    }

    /// The GPU-trio CSA compressor body (see the `csa_compressor` override). Errors
    /// bubble to the host fallback. `positions[0]` is the base offset for the causal
    /// top-k threshold (0 in this frozen-prefix window — the compressor always sees
    /// positions starting at 0, matching the host oracle's `positions`).
    #[allow(clippy::too_many_arguments)]
    /// DECODE variant of [`Self::gpu_csa_compressor`]: the outer + indexer
    /// projections already ran through the resident `mm` seam (`ixp` + `kv`/`gate`),
    /// so we skip the host `cpu_matmul` and go straight to the trio prep (indexer q
    /// RoPE + weight scale, exactly as the host `indexer_topk_pre`/`gpu_csa_compressor`
    /// do) → `dsa_trio_onecb` (ONE submit) → block-bias visibility. Dataflow into the
    /// trio is byte-identical to [`Self::gpu_csa_compressor`].
    #[allow(clippy::too_many_arguments)]
    fn gpu_csa_compressor_pre(
        &mut self, kv: &[f32], gate: &[f32], s: usize, hd: usize, m: usize,
        positions: &[usize], position_bias: &[f32], kv_norm: &[f32], eps: f32,
        inv_freq: &[f32], scaling: f32, ix_nh: usize, ix_hd: usize, index_topk: usize,
        ixp: &IndexerProj,
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        let n_win = s / m;
        if n_win == 0 {
            return Ok((Vec::new(), vec![0i32; 0]));
        }
        // Indexer q RoPE (same closure as indexer_topk_pre / gpu_csa_compressor).
        let rope_dim = 2 * inv_freq.len();
        let mut q = vec![0.0f32; s * ix_nh * ix_hd];
        q.copy_from_slice(ixp.q_flat);
        let (cos_q, sin_q) = rope_cos_sin(positions, inv_freq, scaling);
        apply_interleaved_rope_inplace(&mut q, s * ix_nh, ix_hd, rope_dim, &|r| r / ix_nh, &cos_q, &sin_q);
        let w_scale = (ix_nh as f64).powf(-0.5) as f32;
        let wgt: Vec<f32> = ixp.wgt.iter().map(|v| v * w_scale).collect();
        let softmax_scale = (ix_hd as f64).powf(-0.5) as f32;
        let pos0 = positions.first().copied().unwrap_or(0);
        let (compressed, sel) = self.dsa_trio_onecb(
            kv, gate, position_bias, kv_norm, ixp.kv, ixp.gate, ixp.position_bias, ixp.kv_norm,
            &q, &wgt, inv_freq, s, m, hd, ix_hd, ix_nh, n_win, index_topk, eps, softmax_scale, pos0,
        )?;
        let mut vis = vec![0i32; s * n_win];
        for si in 0..s {
            for kk in 0..index_topk {
                let idx = sel[si * index_topk + kk];
                if idx >= 0 {
                    vis[si * n_win + idx as usize] = 1;
                }
            }
        }
        Ok((compressed, vis))
    }

    fn gpu_csa_compressor(
        &mut self, hs: &[f32], q_residual: &[f32], s: usize, h: usize, hd: usize, m: usize,
        positions: &[usize], kv_proj: &[f32], gate_proj: &[f32], position_bias: &[f32],
        kv_norm: &[f32], eps: f32, inv_freq: &[f32], scaling: f32, q_lora: usize,
        ix_nh: usize, ix_hd: usize, index_topk: usize, ix: &IndexerWeights,
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        let n_win = s / m;
        if n_win == 0 {
            return Ok((Vec::new(), vec![0i32; 0]));
        }
        let cpu_mm = crate::model::cpu_matmul;
        // ---- host matvecs (compressor weights are not resident) + indexer q RoPE ----
        let kv = cpu_mm(hs, kv_proj, s, h, 2 * hd);
        let gate = cpu_mm(hs, gate_proj, s, h, 2 * hd);
        let ix_kv = cpu_mm(hs, ix.kv_proj, s, h, 2 * ix_hd);
        let ix_gate = cpu_mm(hs, ix.gate_proj, s, h, 2 * ix_hd);
        let mut q = cpu_mm(q_residual, ix.q_b_proj, s, q_lora, ix_nh * ix_hd);
        let rope_dim = 2 * inv_freq.len();
        let (cos_q, sin_q) = rope_cos_sin(positions, inv_freq, scaling);
        apply_interleaved_rope_inplace(&mut q, s * ix_nh, ix_hd, rope_dim, &|r| r / ix_nh, &cos_q, &sin_q);
        let wgt0 = cpu_mm(hs, ix.weights_proj, s, h, ix_nh);
        let w_scale = (ix_nh as f64).powf(-0.5) as f32;
        let wgt: Vec<f32> = wgt0.iter().map(|v| v * w_scale).collect();
        let softmax_scale = (ix_hd as f64).powf(-0.5) as f32;
        let pos0 = positions.first().copied().unwrap_or(0);

        // ---- FOLDED DSA trio: compress(outer) ∥ compress(indexer) → index_score →
        // topk in ONE command buffer. The indexer `ck` + the `scores` are resident
        // intermediates (ck feeds index_score; scores feeds topk) — no host readback
        // + re-upload between the three kernels (the prior 3-submit orchestration).
        // Only `compressed` (outer, for the MLA KV concat) and `sel` (top-512) read
        // back. Correctness is unchanged (identical kernels + dataflow, see the
        // dsa_gpu_orchestration mirror + the ckv=1.0 on-node gate).
        let (compressed, sel) = self.dsa_trio_onecb(
            &kv, &gate, position_bias, kv_norm, &ix_kv, &ix_gate, ix.position_bias, ix.kv_norm,
            &q, &wgt, inv_freq, s, m, hd, ix_hd, ix_nh, n_win, index_topk, eps, softmax_scale, pos0,
        )?;
        // ---- block_bias visibility from the (set-valid) top-k picks ----
        let mut vis = vec![0i32; s * n_win];
        for si in 0..s {
            for kk in 0..index_topk {
                let idx = sel[si * index_topk + kk];
                if idx >= 0 {
                    vis[si * n_win + idx as usize] = 1;
                }
            }
        }
        Ok((compressed, vis))
    }

    /// The FOLDED DSA trio (`dsv4_dsa_compress` ×2 ∥ → `dsv4_dsa_index_score` →
    /// `dsv4_dsa_topk`) recorded into ONE command buffer. The two compress
    /// dispatches (outer `head_dim` + indexer `index_head_dim`) are independent →
    /// no barrier between them; the indexer `ck` and the `scores` stay RESIDENT
    /// (ck → index_score, scores → topk) with a producer→consumer barrier each, so
    /// only `compressed` (outer, for the MLA KV concat) and `sel` (top-512) read
    /// back. Collapses the prior 3-submit + 2-intermediate-readback orchestration to
    /// one submit; the kernels + dataflow are byte-identical (correctness proven by
    /// the dsa_gpu_orchestration mirror + the on-node ckv=1.0 / argmax gate).
    #[allow(clippy::too_many_arguments)]
    fn dsa_trio_onecb(
        &mut self,
        kv: &[f32], gate: &[f32], pbias: &[f32], knorm: &[f32],
        ix_kv: &[f32], ix_gate: &[f32], ix_pbias: &[f32], ix_knorm: &[f32],
        q: &[f32], wgt: &[f32], ifreq: &[f32],
        s: usize, m: usize, hd: usize, ix_hd: usize, ix_nh: usize, n_win: usize,
        index_topk: usize, rms_eps: f32, softmax_scale: f32, pos0: usize,
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        // Ensure the three trio pipelines (spec BLOCK_SIZE=64) before opening the CB.
        for base in ["dsv4_dsa_compress", "dsv4_dsa_index_score", "dsv4_dsa_topk"] {
            let shader = format!("{base}_bs64");
            if !self.eng.has_pipeline(&shader) {
                let spv = self._shader_spvs.get(base).map(|v| v.as_slice())
                    .ok_or_else(|| format!("{base} SPIR-V missing"))?;
                if !self.eng.compile_variant_timeout(&shader, spv, &[(0, 64)], 2000)? {
                    return Err(format!("{shader} pipeline timeout"));
                }
            }
        }
        let rope_dim = 2 * ifreq.len();
        let compress_pc = |hdv: usize| {
            let mut pc = Vec::with_capacity(24);
            pc.extend_from_slice(&(s as u32).to_le_bytes());
            pc.extend_from_slice(&(m as u32).to_le_bytes());
            pc.extend_from_slice(&(hdv as u32).to_le_bytes());
            pc.extend_from_slice(&(rope_dim as u32).to_le_bytes());
            pc.extend_from_slice(&(n_win as u32).to_le_bytes());
            pc.extend_from_slice(&rms_eps.to_le_bytes());
            pc
        };
        let mut score_pc = Vec::with_capacity(20);
        score_pc.extend_from_slice(&(s as u32).to_le_bytes());
        score_pc.extend_from_slice(&(ix_nh as u32).to_le_bytes());
        score_pc.extend_from_slice(&(ix_hd as u32).to_le_bytes());
        score_pc.extend_from_slice(&(n_win as u32).to_le_bytes());
        score_pc.extend_from_slice(&softmax_scale.to_le_bytes());
        let mut topk_pc = Vec::with_capacity(20);
        topk_pc.extend_from_slice(&(s as u32).to_le_bytes());
        topk_pc.extend_from_slice(&(n_win as u32).to_le_bytes());
        topk_pc.extend_from_slice(&(index_topk as u32).to_le_bytes());
        topk_pc.extend_from_slice(&(m as u32).to_le_bytes());
        topk_pc.extend_from_slice(&(pos0 as u32).to_le_bytes());

        let a = |eng: &mut compute::ComputeEngine, v: &[f32]| -> Result<compute::Buffer, String> {
            let b = eng.alloc_host_coherent_storage((v.len() * 4).max(4) as u64)?;
            b.write(&f32_slice_to_bytes(v))?;
            Ok(b)
        };
        let kvbuf = a(&mut self.eng, kv)?;
        let gbuf = a(&mut self.eng, gate)?;
        let pbuf = a(&mut self.eng, pbias)?;
        let knbuf = a(&mut self.eng, knorm)?;
        let ixkvbuf = a(&mut self.eng, ix_kv)?;
        let ixgbuf = a(&mut self.eng, ix_gate)?;
        let ixpbuf = a(&mut self.eng, ix_pbias)?;
        let ixknbuf = a(&mut self.eng, ix_knorm)?;
        let ifbuf = a(&mut self.eng, ifreq)?;
        let qbuf = a(&mut self.eng, q)?;
        let wbuf = a(&mut self.eng, wgt)?;
        let compbuf = self.eng.alloc_host_coherent_storage((n_win * hd * 4).max(4) as u64)?;
        let ckbuf = self.eng.alloc_host_coherent_storage((n_win * ix_hd * 4).max(4) as u64)?;
        let scoresbuf = self.eng.alloc_host_coherent_storage((s * n_win * 4).max(4) as u64)?;
        let selbuf = self.eng.alloc_host_coherent_storage((s * index_topk * 4).max(4) as u64)?;

        let cb = self.eng.begin_batch()?;
        // outer + indexer compress: independent (distinct out-buffers) → no barrier.
        self.eng.record_to(cb, "dsv4_dsa_compress_bs64", &[&kvbuf, &gbuf, &pbuf, &knbuf, &ifbuf, &compbuf], &compress_pc(hd), (n_win as u32, 1, 1))?;
        self.eng.record_to(cb, "dsv4_dsa_compress_bs64", &[&ixkvbuf, &ixgbuf, &ixpbuf, &ixknbuf, &ifbuf, &ckbuf], &compress_pc(ix_hd), (n_win as u32, 1, 1))?;
        self.eng.record_barrier_to(cb); // ck → index_score
        self.eng.record_to(cb, "dsv4_dsa_index_score_bs64", &[&qbuf, &ckbuf, &wbuf, &scoresbuf], &score_pc, (s as u32, n_win as u32, 1))?;
        self.eng.record_barrier_to(cb); // scores → topk
        self.eng.record_to(cb, "dsv4_dsa_topk_bs64", &[&scoresbuf, &selbuf], &topk_pc, (s as u32, 1, 1))?;
        self.eng.submit_batch(cb)?;

        let compressed = read_f32_buf(&compbuf, n_win * hd);
        let sel_raw = read_f32_buf(&selbuf, s * index_topk);
        let sel: Vec<i32> = sel_raw.iter().map(|f| f.to_bits() as i32).collect();
        for b in [kvbuf, gbuf, pbuf, knbuf, ixkvbuf, ixgbuf, ixpbuf, ixknbuf, ifbuf, qbuf, wbuf, compbuf, ckbuf, scoresbuf, selbuf] {
            self.eng.return_to_pool(b);
        }
        Ok((compressed, sel))
    }
}

/// mHC backend selector. DEFAULT = the bit-exact HOST oracle; `VLLM_VULKAN_DSV4_HC_GPU=1`
/// opts into the `dsv4_hyper_connection` GPU kernel.
///
/// ── ITEM A gate call (mHC-GPU submit-fusion): NO-GO for default-ON ────────────
/// The GPU mHC is CORRECT (argmax 11111, per-op cos=0.99999) but ~5% SLOWER on the
/// PP-10 decode (3996.5 → 4200.7 ms/tok): it adds 86 DISCRETE dispatch+fence+readback
/// ops/token (2 sites × 43 layers) whose submit tax exceeds the pure-CPU host mHC it
/// replaces. The intended cure — record the mHC dispatch into ONE resident command
/// buffer alongside the surrounding decode ops (the qwen35 `q35r` WS3 1-CB pattern:
/// `record_to_off` + `record_barrier_to`, whole span in one submit) — is UNREACHABLE
/// here: the DSV4 decode is HOST-ORCHESTRATED. Every `Mv::mm`/`hc_block` returns a
/// host `Vec<f32>` and the non-matmul glue (RMSNorm, RoPE, MLA softmax, MoE router,
/// SwiGLU, residual-mix) runs host-side, so there is NO resident intermediate buffer
/// and NO CB ring to fuse into. The mHC is sandwiched between two HOST ops (prev
/// `hc_residual_mix` → mHC → `rmsnorm_rows`) with NO adjacent GPU dispatch to co-record.
/// True fusion therefore requires first building the resident 1-CB decode forward
/// (all glue as resident GPU kernels reading/writing resident buffers — the M2
/// project), not a wiring change. On a host-orchestrated forward, moving a cheap CPU
/// op onto the GPU can only ADD submit tax, so mHC-GPU stays behind an opt-in flag
/// until that resident forward exists. (This is the same dispatch/fence-tax wall the
/// direct-submit + fence-tax spikes hit; do NOT reintroduce a libdrm direct-submit
/// path — it was net-negative here via the GCR_FULL barrier.)
fn hc_gpu_enabled() -> bool {
    std::env::var("VLLM_VULKAN_DSV4_HC_GPU").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// Resident HC-site span (`VLLM_VULKAN_DSV4_HC_RESIDENT=1`, default-OFF). This is
/// the CURE for the `hc_gpu_enabled` NO-GO: instead of dispatching the mHC as a
/// STANDALONE submit (sandwiched between two host ops → per-call round-trip), the
/// attn-site `hc_residual_mix` + the ffn-site mHC (Sinkhorn) are recorded into the
/// END of the resident attn-tail command buffer (M2b). The mHC thus RIDES the
/// attn-tail's existing single submit — the branch output is already device-
/// resident, and the mHC output (`streams'`, next post/comb, collapsed) reads back
/// in the same single readback the tail already did → ZERO added submits, and the
/// 43 ffn-site Sinkhorns leave the host critical path. Requires the resident 1-CB
/// tail (`VLLM_VULKAN_DSV4_1CB=1`); disabled → the exact host mHC sequence
/// (byte-identical), so correctness is never at risk. On-node argmax/gen gate is
/// authoritative (the two-float Sinkhorn is cos>0.9999, not bit-identical to the
/// host f64 oracle — same numeric path as `hc_gpu_enabled`, argmax-11111 there).
///
/// ── ON-NODE GATE (2026-08-16, PP-10 @1850, real 86GB ckpt, GEN=64) ────────────
/// CORRECTNESS: PASS. Gen sequence BIT-IDENTICAL to the host oracle (HC_RESIDENT=0)
/// at GEN=8 AND GEN=64; argmax 11111 (' Paris'). The span assembles cleanly — the
/// structural hypothesis is CONFIRMED and the submit-tax cure WORKS: the hc bucket
/// COLLAPSED 38.3 → 19.4 ms/tok (the 43 ffn-site Sinkhorns left `B::Hc`, riding the
/// attn-tail submit — zero added round-trips).
/// PERF: NET-NEGATIVE — do NOT default-ON. 268.8 → 376.8 ms/tok (+40%). Killing the
/// submit tax EXPOSED that the mHC KERNEL itself is compute-slow at decode:
/// `dsv4_hyper_connection_bs64` dispatches ONE workgroup per token (grid.x=seq=1),
/// so at seq=1 a single wave64 runs the whole `flat@fn` projection (fn_w
/// [(2+hc)*hc, hc*h] = [24, 8192] ≈ 196k MACs) + 20-iter Sinkhorn SERIALLY on 1 of
/// 40 CUs — ~4.3 ms/call vs the ~0.44 ms host f64 Sinkhorn it replaced. The
/// attn-tail bucket ballooned 15.5 → 200.4 ms/tok (+185), dwarfing the 19 ms hc
/// saving. So decode is NOT submit-tax-bound after this fix; it is mHC-kernel-
/// OCCUPANCY-bound.
///
/// ── DECODE-SHAPED KERNEL RE-GATE (2026-08-16b, PP-10 @1850, real 86GB, GEN=64) ─
/// `dsv4_hyper_connection` was restructured to a workgroup LDS TREE-reduce at
/// BLOCK_SIZE=512 (8 wave64 subgroups; [`dsv4_hc_bs`]) — microbench (HD=16384,
/// n_tokens=1) 4.3ms → 0.49ms/call (per-submit, incl fence tax), cos vs the f64
/// CPU oracle still 1.0 (ARGMAX-SAFE). Re-gate: CORRECTNESS PASS — argmax 11111,
/// gen BIT-IDENTICAL to the host oracle at GEN=8 AND GEN=64. The submit-tax cure
/// holds (hc bucket 4.35 → 2.18 ms/stage; the ffn-site Sinkhorns leave `B::Hc`).
/// PERF: STILL NET-NEGATIVE — 289.5 → 308.5 ms/tok (+6.6%, down from +40%). The
/// resident ffn-site mHC now rides attn_tail at +4.43 ms/stage while it only frees
/// −2.17 ms/stage from hc: the seq=1 GPU mHC (one workgroup streams ~1.5MB fnw/call)
/// is ~2× the CACHE-RESIDENT host f64 Sinkhorn it replaces — a fundamental GFX1013
/// seq=1 occupancy/BW characteristic (one workgroup = one CU can't beat an in-cache
/// f64 reduction). The ONLY remaining flip is lever #2 — spread each mHC reduction
/// across MANY workgroups (grid.x>1) to engage the other ~39 CUs — but that is a
/// fiddly 3-stage split (max → scaled sumsq/dots → finalize, overflow-safe two-float
/// across global scratch) for a SMALL e2e ceiling: even a FREE resident mHC only
/// reaches ~275 ms/tok (~5%) on this comm-dominated 289 ms PP-10 decode. So the
/// decode-shaped kernel SHIPS (argmax-exact, 5-9× faster, foundation for any batched
/// HC path) but this flag stays DEFAULT-OFF; do NOT flip.
fn dsv4_hc_resident_enabled() -> bool {
    std::env::var("VLLM_VULKAN_DSV4_HC_RESIDENT").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// mHC workgroup size = the `dsv4_hyper_connection` spec constant 0 (`BLOCK_SIZE`).
/// The DECODE-SHAPED default is 512 (8 wave64 subgroups) so the single seq=1
/// workgroup has enough waves in flight to HIDE the ~1.5MB fnw streaming latency
/// that starved the old single-wave BLOCK_SIZE=64 build. On-node sweep (n53 @1850,
/// HD=16384, decode n_tokens=1, per-submit microbench): bs64=2435µs bs128=1335
/// bs256=776 bs512=493 bs1024=859 — 512 is the sweet spot (1024 over-subscribes the
/// CU + deepens the reduction tree). Override via `DSV4_HC_BS`; clamped to a power
/// of two in [64,1024].
pub(crate) fn dsv4_hc_bs() -> u32 {
    let bs = std::env::var("DSV4_HC_BS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(512);
    match bs { 64 | 128 | 256 | 512 | 1024 => bs, _ => 512 }
}

/// Variant name for the mHC pipeline at the current [`dsv4_hc_bs`] (one compiled
/// pipeline per BLOCK_SIZE so a sweep never collides in the pipeline cache).
pub(crate) fn dsv4_hc_variant() -> String {
    format!("dsv4_hyper_connection_bs{}", dsv4_hc_bs())
}

/// Dispatch the shipped `dsv4_hyper_connection` kernel for one HC site over `seq`
/// tokens (one workgroup/token). Bindings + push-constant layout mirror
/// `debug_dsv4_hc` (cos=1.0 gate). `logit_clamp` is the validated 1e4 no-op.
/// Returns `(post [seq,hc], comb [seq,hc,hc], collapsed [seq,h])` — the same tuple
/// shape as [`crate::dsv4::hc_block`].
#[allow(clippy::too_many_arguments)]
fn gpu_hc_block(
    eng: &mut compute::ComputeEngine,
    shader_spvs: &HashMap<String, Vec<u8>>,
    streams: &[f32], seq: usize, hc: usize, h: usize,
    fn_w: &[f32], base: &[f32], scale: &[f32], iters: usize, hc_eps: f32, rms_eps: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let variant = dsv4_hc_variant();
    let shader = variant.as_str();
    if !eng.has_pipeline(shader) {
        let spv = shader_spvs.get("dsv4_hyper_connection").map(|v| v.as_slice())
            .ok_or_else(|| "dsv4_hyper_connection SPIR-V missing".to_string())?;
        if !eng.compile_variant_timeout(shader, spv, &[(0, dsv4_hc_bs())], 2000)? {
            return Err("hc pipeline timeout".into());
        }
    }
    // push constants: hc_mult(u32) hidden(u32) iters(u32) hc_eps(f32) rms_eps(f32) logit_clamp(f32)
    let mut pc = Vec::with_capacity(24);
    pc.extend_from_slice(&(hc as u32).to_le_bytes());
    pc.extend_from_slice(&(h as u32).to_le_bytes());
    pc.extend_from_slice(&(iters as u32).to_le_bytes());
    pc.extend_from_slice(&hc_eps.to_le_bytes());
    pc.extend_from_slice(&rms_eps.to_le_bytes());
    pc.extend_from_slice(&1e4f32.to_le_bytes());

    let sbuf = eng.alloc_host_coherent_storage((streams.len() * 4).max(4) as u64)?;
    sbuf.write(&f32_slice_to_bytes(streams))?;
    let fbuf = eng.alloc_host_coherent_storage((fn_w.len() * 4).max(4) as u64)?;
    fbuf.write(&f32_slice_to_bytes(fn_w))?;
    let bbuf = eng.alloc_host_coherent_storage((base.len() * 4).max(4) as u64)?;
    bbuf.write(&f32_slice_to_bytes(base))?;
    let scbuf = eng.alloc_host_coherent_storage((scale.len() * 4).max(4) as u64)?;
    scbuf.write(&f32_slice_to_bytes(scale))?;
    let pbuf = eng.alloc_host_coherent_storage((seq * hc * 4).max(4) as u64)?;
    let cbuf = eng.alloc_host_coherent_storage((seq * hc * hc * 4).max(4) as u64)?;
    let lbuf = eng.alloc_host_coherent_storage((seq * h * 4).max(4) as u64)?;
    let cb = eng.begin_batch()?;
    eng.record_to(cb, shader, &[&sbuf, &fbuf, &bbuf, &scbuf, &pbuf, &cbuf, &lbuf], &pc, (seq as u32, 1, 1))?;
    eng.submit_batch(cb)?;
    let post = read_f32_buf(&pbuf, seq * hc);
    let comb = read_f32_buf(&cbuf, seq * hc * hc);
    let coll = read_f32_buf(&lbuf, seq * h);
    for b in [sbuf, fbuf, bbuf, scbuf, pbuf, cbuf, lbuf] { eng.return_to_pool(b); }
    Ok((post, comb, coll))
}

/// `VLLM_VULKAN_DSV4_DSA_GPU=1` opts the CSA compressor into the GPU trio
/// (`dsv4_dsa_compress` + `dsv4_dsa_index_score` + `dsv4_dsa_topk`). Default OFF:
/// the CSA compressor stays on the host oracle until the on-node per-op cos gate
/// (`debug_dsv4_dsa_*`) + the composed argmax-11111 gate are re-run (cluster down).
fn dsa_gpu_enabled() -> bool {
    std::env::var("VLLM_VULKAN_DSV4_DSA_GPU").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// M3 instrument. `DSV4_DSA_DEBUG=1` (with `VLLM_VULKAN_DSV4_DSA_GPU=1`) dumps, per
/// CSA-compressor call (= one CSA layer for the current token), the GPU DSA trio's
/// output vs the host oracle: `compressed_kv` cos + max_abs, and the top-512
/// visibility SET diff per query row (the discrete `dsv4_dsa_topk` selection that
/// feeds `block_bias`). This is the real per-layer A/B the DSA divergence needs —
/// the referenced hook never existed. Index-score is already RULED OUT (the
/// two-float fix had zero effect), so the localizer surfaces `dsv4_dsa_topk`
/// tie-break + the host `block_bias`/`vis_to_additive` assembly first: a call whose
/// `compressed_kv` cos≈1 but whose vis SET differs points at the argsort/assembly,
/// not the compressor. Instrument only (does NOT change the selection or fix DSA).
static DSA_DBG_CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn dsa_debug_enabled() -> bool {
    std::env::var("DSV4_DSA_DEBUG").map(|v| v == "1" || v == "true").unwrap_or(false)
}

/// Per-call GPU-vs-oracle A/B dump for the CSA compressor. `(ckv_g, vis_g)` = GPU
/// trio output, `(ckv_h, vis_h)` = host oracle; `s` = query rows. `vis_*` are the
/// `[s, n_win]` 0/1 visibility planes (the top-512 selection). Prints to stderr.
fn dsa_dump_ab(ckv_g: &[f32], vis_g: &[i32], ckv_h: &[f32], vis_h: &[i32], s: usize) {
    let call = DSA_DBG_CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // compressed_kv cos + max_abs.
    let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
    for (a, b) in ckv_g.iter().zip(ckv_h.iter()) {
        dot += *a as f64 * *b as f64; na += *a as f64 * *a as f64; nb += *b as f64 * *b as f64;
        mx = mx.max((*a as f64 - *b as f64).abs());
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    if vis_g.len() != vis_h.len() || s == 0 {
        eprintln!("DSV4_DSA_DEBUG[call {call}]: ckv cos={cos:.9} max_abs={mx:.3e} \
                   vis LEN MISMATCH g={} h={}", vis_g.len(), vis_h.len());
        return;
    }
    let n_win = vis_h.len() / s;
    let (mut agree, mut first_bad, mut worst_sym) = (0usize, None::<usize>, 0usize);
    for si in 0..s {
        let g: std::collections::HashSet<usize> = (0..n_win).filter(|&w| vis_g[si * n_win + w] != 0).collect();
        let h: std::collections::HashSet<usize> = (0..n_win).filter(|&w| vis_h[si * n_win + w] != 0).collect();
        let sym = g.symmetric_difference(&h).count();
        if sym == 0 { agree += 1; } else {
            if first_bad.is_none() { first_bad = Some(si); }
            worst_sym = worst_sym.max(sym);
        }
    }
    let tag = match first_bad {
        None => "SET-MATCH".to_string(),
        Some(row) => {
            let g: std::collections::HashSet<usize> = (0..n_win).filter(|&w| vis_g[row * n_win + w] != 0).collect();
            let h: std::collections::HashSet<usize> = (0..n_win).filter(|&w| vis_h[row * n_win + w] != 0).collect();
            let mut only_g: Vec<usize> = g.difference(&h).copied().collect(); only_g.sort_unstable();
            let mut only_h: Vec<usize> = h.difference(&g).copied().collect(); only_h.sort_unstable();
            only_g.truncate(8); only_h.truncate(8);
            format!("DIVERGE row={row} only_gpu={only_g:?} only_oracle={only_h:?} worst_sym={worst_sym}")
        }
    };
    eprintln!("DSV4_DSA_DEBUG[call {call}]: ckv cos={cos:.9} max_abs={mx:.3e} \
               vis {agree}/{s} rows agree; {tag}");
}


// ---------------- GPU plumbing ----------------

fn make_engine(
    device_idx: usize,
) -> Result<(compute::ComputeEngine, device::ComputeDevice, HashMap<String, Vec<u8>>), String> {
    let dev = device::ComputeDevice::create(device_idx)?;
    let shader_spvs = crate::include_all_shaders();
    let refs: HashMap<&str, &[u8]> =
        shader_spvs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
    let eng = compute::ComputeEngine::new(
        dev.instance.clone(), dev.physical_device, dev.device.clone(),
        dev.compute_queue, dev.compute_queue_family, dev.caps(), &refs,
    )?;
    Ok((eng, dev, shader_spvs))
}

/// Ensure the `{base}_bs{BS}_r{ROWS}` matvec variant pipeline exists (mirrors
/// `debug_dsv4_verify`'s lazy compile).
fn ensure_variant(
    eng: &mut compute::ComputeEngine,
    shader_spvs: &HashMap<String, Vec<u8>>,
    base: &str,
) -> Result<(), String> {
    let shader = format!("{base}_bs{BS}_r{ROWS}");
    if eng.has_pipeline(&shader) {
        return Ok(());
    }
    let spv = shader_spvs.get(base).map(|v| v.as_slice())
        .ok_or_else(|| format!("{base} SPIR-V missing"))?;
    match eng.compile_variant_timeout(&shader, spv, &[(0, BS), (1, ROWS), (2, 1)], 2000)? {
        true => Ok(()),
        false => Err(format!("{shader}: pipeline creation timed out")),
    }
}

/// Upload a raw quantized 2D linear GPU-resident (skips non-quant tensors — those
/// stay host).
fn up_lin(
    eng: &mut compute::ComputeEngine,
    src: &Dsv4RealSrc,
    map: &mut HashMap<String, Qbuf>,
    name: &str,
    out_f: usize,
    in_f: usize,
) -> Result<(), String> {
    if !src.is_quant(name) {
        return Ok(());
    }
    let rq = src.raw_linear(name, out_f, in_f);
    map.insert(name.to_string(), alloc_qbuf(eng, rq)?);
    Ok(())
}

fn up_exp(
    eng: &mut compute::ComputeEngine,
    src: &Dsv4RealSrc,
    map: &mut HashMap<String, Qbuf>,
    name: &str,
    e: usize,
    out_f: usize,
    in_f: usize,
) -> Result<(), String> {
    let rq = src.raw_expert(name, e, out_f, in_f);
    map.insert(format!("{name}#{e}"), alloc_qbuf(eng, rq)?);
    Ok(())
}

fn alloc_qbuf(eng: &mut compute::ComputeEngine, rq: crate::dsv4_loader::RawQ) -> Result<Qbuf, String> {
    let pb = bytemuck::cast_slice::<u32, u8>(&rq.packed).to_vec();
    let p = eng.alloc_host_coherent_storage(pb.len().max(4) as u64)?;
    p.write(&pb)?;
    let sb = f32_slice_to_bytes(&rq.scales);
    let s = eng.alloc_host_coherent_storage(sb.len().max(4) as u64)?;
    s.write(&sb)?;
    let bb = f32_slice_to_bytes(&rq.biases);
    let b = eng.alloc_host_coherent_storage(bb.len().max(4) as u64)?;
    b.write(&bb)?;
    Ok(Qbuf { p, s, b, out: rq.out, inn: rq.inn, bits: rq.bits, gs: rq.gs })
}

/// `out[s,out] = x[s,in] @ W^T` over a resident quantized `W`, one matvec per row
/// (the shipped kernels are single-vector). Reuses the resident p/s/b buffers.
fn gpu_matvec_rows(
    eng: &mut compute::ComputeEngine,
    q: &Qbuf,
    x: &[f32],
    s: usize,
) -> Result<Vec<f32>, String> {
    let shader = format!("{}_bs{BS}_r{ROWS}", base_shader(q.bits));
    let pc = matvec_mlx4_pc(q.inn, q.out, q.gs);
    let wg = (q.out as u32 + ROWS - 1) / ROWS;
    let mut out = vec![0f32; s * q.out];
    if s == 0 {
        return Ok(out);
    }
    // Record ALL s row-matvecs into ONE command buffer (single submit + single
    // readback) over a wide x/out scratch. Each row reads/writes a DISJOINT slice
    // (byte-offset bound), so the dispatches are independent — no barriers, and the
    // math is bit-identical to the former submit-per-row loop. Collapses the s
    // synchronous round-trips (the compressor's history-length projections under
    // LEVER #2 would otherwise be t1 submits/weight) into one.
    let xbuf = eng.alloc_host_coherent_storage((s * q.inn * 4).max(4) as u64)?;
    let obuf = eng.alloc_host_coherent_storage((s * q.out * 4).max(4) as u64)?;
    xbuf.write(&f32_slice_to_bytes(&x[..s * q.inn]))?;
    let cb = eng.begin_batch()?;
    for si in 0..s {
        eng.record_to_off(
            cb, &shader,
            &[(&q.p, 0), (&q.s, 0), (&q.b, 0), (&xbuf, (si * q.inn * 4) as u64), (&obuf, (si * q.out * 4) as u64)],
            &pc, (wg, 1, 1),
        )?;
    }
    eng.submit_batch(cb)?;
    out.copy_from_slice(&read_f32_buf(&obuf, s * q.out));
    eng.return_to_pool(xbuf);
    eng.return_to_pool(obuf);
    Ok(out)
}

/// Record ONE resident quantized matvec (`out[out_f] = x[in_f] @ W^T`) into an
/// OPEN command buffer over resident off-buffers — the resident twin of
/// [`gpu_matvec_rows`] with NO per-op alloc/submit/readback. `x`/`out` are bound
/// at BYTE offsets so a single wide scratch buffer can hold per-expert slices
/// (`out_all[j*out_f]`, `hid_all[j*in_f]`). Reuses the `{base}_bs64_r2` variant
/// (pre-compiled for bits 2/6/8 by `from_ckpt_streamed`). The caller inserts the
/// producer→consumer `record_barrier_to` between dependent stages.
fn dsv4r_rec_mv(
    eng: &mut compute::ComputeEngine,
    cb: ash::vk::CommandBuffer,
    q: &Qbuf,
    xbuf: &compute::Buffer,
    xoff: u64,
    out: &compute::Buffer,
    outoff: u64,
) -> Result<(), String> {
    let shader = format!("{}_bs{BS}_r{ROWS}", base_shader(q.bits));
    let pc = matvec_mlx4_pc(q.inn, q.out, q.gs);
    let wg = (q.out as u32 + ROWS - 1) / ROWS;
    eng.record_to_off(
        cb, &shader,
        &[(&q.p, 0), (&q.s, 0), (&q.b, 0), (xbuf, xoff), (out, outoff)],
        &pc, (wg, 1, 1),
    )
}

/// M2a: record `dsv4_hc_residual_mix` (the manifold HC residual mix, oracle
/// `dsv4::hc_residual_mix`) into an OPEN command buffer over resident buffers.
/// `post [seq*hc]`, `sub [seq*hidden]` (branch output), `comb [seq*hc*hc]`,
/// `streams [seq*hc*hidden]` → `out [seq*hc*hidden]`. Built + mirror-validated
/// (dsv4_hc_residual_mix_mirror_matches_oracle); wired by the M2b full-span path.
#[allow(dead_code, clippy::too_many_arguments)] // M2a primitive — wired by M2b.
fn dsv4r_rec_hc_residual_mix(
    eng: &mut compute::ComputeEngine,
    cb: ash::vk::CommandBuffer,
    post: &compute::Buffer,
    sub: &compute::Buffer,
    comb: &compute::Buffer,
    streams: &compute::Buffer,
    out: &compute::Buffer,
    seq: usize,
    hc: usize,
    hidden: usize,
) -> Result<(), String> {
    let pc = dsv4_hc_residual_mix_pc(seq, hc, hidden);
    let wg = ((seq * hc * hidden) as u32 + 255) / 256;
    eng.record_to(cb, "dsv4_hc_residual_mix", &[post, sub, comb, streams, out], &pc, (wg, 1, 1))
}

/// Slice a raw MLX-affine quantized linear to its OUTPUT-row block `[r0, r1)`.
/// `RawQ` is row-major (`packed [out, packed_cols]`, `scales`/`biases [out,
/// groups]`), so output-row slicing is a plain contiguous cut — always safe for
/// any bit-width/group-size (rows are independent). Used to upload the grouped
/// block-diagonal `wo_a` as `g` per-group resident matvec weights (M2b).
fn raw_row_block(rq: &crate::dsv4_loader::RawQ, r0: usize, r1: usize) -> crate::dsv4_loader::RawQ {
    let pc = rq.packed.len() / rq.out;
    let g = rq.scales.len() / rq.out;
    crate::dsv4_loader::RawQ {
        packed: rq.packed[r0 * pc..r1 * pc].to_vec(),
        scales: rq.scales[r0 * g..r1 * g].to_vec(),
        biases: rq.biases[r0 * g..r1 * g].to_vec(),
        out: r1 - r0,
        inn: rq.inn,
        bits: rq.bits,
        gs: rq.gs,
    }
}

/// M2b: record `dsv4_mla_softmax` (resident single-token MLA eager attention with
/// sink + sliding window + compressed-KV block_bias + output-rope conjugate) into
/// an OPEN command buffer. Bindings mirror the shader; one wave64 workgroup per
/// head (`nh` groups). `q`/`kvs`/`ckv`/`bb`/`sinks`/`cos`/`sin` are resident input
/// buffers; `out` receives `ao [nh*hd]` (post output-rope). Oracle: the softmax
/// core of [`attention_layer_decode`].
#[allow(clippy::too_many_arguments)]
fn dsv4r_rec_mla_softmax(
    eng: &mut compute::ComputeEngine,
    cb: ash::vk::CommandBuffer,
    q: &compute::Buffer,
    kvs: &compute::Buffer,
    ckv: &compute::Buffer,
    bb: &compute::Buffer,
    sinks: &compute::Buffer,
    cos: &compute::Buffer,
    sin: &compute::Buffer,
    out: &compute::Buffer,
    nh: usize,
    hd: usize,
    t1: usize,
    t_comp: usize,
    sliding_window: usize,
    rope_dim: usize,
) -> Result<(), String> {
    let scaling = (hd as f64).powf(-0.5) as f32;
    let pc = crate::push_constants::dsv4_mla_softmax_pc(nh, hd, t1, t_comp, sliding_window, rope_dim, scaling);
    eng.record_to(cb, "dsv4_mla_softmax", &[q, kvs, ckv, bb, sinks, cos, sin, out], &pc, (nh as u32, 1, 1))
}

/// M2b: record the grouped block-diagonal `wo_a` output projection as `g` resident
/// mlx6 matvecs into an OPEN command buffer. Group `gg` reads the `ao` slice
/// `[gg*per_g, (gg+1)*per_g)` (contiguous, `per_g = nh*hd/g` == `nh/g` heads) and
/// its per-group weight `wo_a#g{gg}` (`[olr, per_g]`) → `proj[gg*olr..]`.
/// Arithmetically identical to the host block-diagonal loop in
/// [`attention_layer_decode`] (re-associated as `g` independent matvecs), so the
/// on-node-validated matvec kernel needs NO new packed-offset math. The `g`
/// matvecs are independent → no inter-group barrier.
fn dsv4r_rec_wo_a_grouped(
    eng: &mut compute::ComputeEngine,
    cb: ash::vk::CommandBuffer,
    wo_a_groups: &[&Qbuf],
    ao: &compute::Buffer,
    proj: &compute::Buffer,
    per_g: usize,
    olr: usize,
) -> Result<(), String> {
    for (gg, q) in wo_a_groups.iter().enumerate() {
        dsv4r_rec_mv(eng, cb, q, ao, (gg * per_g * 4) as u64, proj, (gg * olr * 4) as u64)?;
    }
    Ok(())
}

/// Resident-Sinkhorn (M2/M2b): record the FULL manifold hyper-connection block
/// (`dsv4_hyper_connection` — max-factored RMSNorm + two-float `flat@fn` mix +
/// the 20-iter doubly-stochastic Sinkhorn + stream collapse, cos>0.9999 on-node)
/// into an OPEN command buffer over resident buffers. `streams [seq*hc*h]` →
/// `post [seq*hc]`, `comb [seq*hc*hc]`, `coll [seq*h]`. The resident twin of
/// [`gpu_hc_block`] with NO per-op alloc/submit/readback — this is the heavier
/// Sinkhorn+projection half that M2a's `hc_residual_mix` left host. Wired by the
/// resident-Sinkhorn full-HC-site span (staged); requires the caller to have
/// `compile_variant_timeout`'d `dsv4_hyper_connection_bs64` (spec BLOCK_SIZE=64).
#[allow(dead_code, clippy::too_many_arguments)] // resident-Sinkhorn primitive — wired by the full HC-site span.
fn dsv4r_rec_hc_block(
    eng: &mut compute::ComputeEngine,
    cb: ash::vk::CommandBuffer,
    streams: &compute::Buffer,
    fn_w: &compute::Buffer,
    base: &compute::Buffer,
    scale: &compute::Buffer,
    post: &compute::Buffer,
    comb: &compute::Buffer,
    coll: &compute::Buffer,
    seq: usize,
    hc: usize,
    h: usize,
    iters: usize,
    hc_eps: f32,
    rms_eps: f32,
) -> Result<(), String> {
    // push constants: hc_mult(u32) hidden(u32) iters(u32) hc_eps(f32) rms_eps(f32) logit_clamp(f32)
    let mut pc = Vec::with_capacity(24);
    pc.extend_from_slice(&(hc as u32).to_le_bytes());
    pc.extend_from_slice(&(h as u32).to_le_bytes());
    pc.extend_from_slice(&(iters as u32).to_le_bytes());
    pc.extend_from_slice(&hc_eps.to_le_bytes());
    pc.extend_from_slice(&rms_eps.to_le_bytes());
    pc.extend_from_slice(&1e4f32.to_le_bytes());
    eng.record_to(cb, &dsv4_hc_variant(), &[streams, fn_w, base, scale, post, comb, coll], &pc, (seq as u32, 1, 1))
}

// ============================================================================
// Backend-generic forward (mirrors dsv4_forward::dsv4_forward op-for-op)
// ============================================================================

/// Sliding-window causal additive mask `[S,S]` (== `dsv4_forward::sliding_mask`).
fn sliding_mask(s: usize, sliding_window: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if j > i || (i - j) >= sliding_window {
                m[i * s + j] = f32::NEG_INFINITY;
            }
        }
    }
    m
}

fn vis_to_additive(vis: &[i32]) -> Vec<f32> {
    vis.iter().map(|&v| if v != 0 { 0.0 } else { f32::NEG_INFINITY }).collect()
}

/// The full model forward over a matvec backend → `[S, vocab]` logits.
pub fn forward_mv<M: Mv>(cfg: &Dsv4Config, input_ids: &[u32], mv: &mut M) -> Vec<f32> {
    match forward_mv_window(cfg, input_ids, None, 0, cfg.num_hidden_layers, mv) {
        WindowOut::Logits(l) => l,
        WindowOut::Streams(_) => unreachable!("full window must produce logits"),
    }
}

/// Output of one PP window: mid-stages ship the `[S, hc*H]` hyper-connection
/// streams onward; the last stage (owns `num_hidden_layers`) additionally runs
/// HyperHead + `model.norm` + `lm_head` → `[S, vocab]` logits.
pub enum WindowOut {
    /// `[seq, hc*h]` hyper-connection streams (mid-stage hop payload).
    Streams(Vec<f32>),
    /// `[seq, vocab]` logits (last stage).
    Logits(Vec<f32>),
}

/// One PP window `[layer_start, layer_end)` of the forward over a matvec backend.
///
/// * First stage (`layer_start == 0`): `streams_in` is ignored; the window embeds
///   `input_ids` and expands to the `hc` parallel streams.
/// * Mid/last stages: `streams_in` is the `[seq, hc*h]` payload received from the
///   previous stage.
/// * Last stage (`layer_end == num_hidden_layers`): collapses via HyperHead +
///   `model.norm` + `lm_head` → [`WindowOut::Logits`]; otherwise ships
///   [`WindowOut::Streams`].
///
/// Chaining the windows `[0,b1) [b1,b2) … [bk,L)` reproduces the monolithic
/// [`forward_mv`] bit-for-bit (the stream payload is the ONLY cross-stage state at
/// prefill) — this is the correctness decomposition PP-10 rides on.
pub fn forward_mv_window<M: Mv>(
    cfg: &Dsv4Config,
    input_ids: &[u32],
    streams_in: Option<Vec<f32>>,
    layer_start: usize,
    layer_end: usize,
    mv: &mut M,
) -> WindowOut {
    let h = cfg.mla.hidden_size;
    let hc = cfg.hc_mult;
    let vocab = cfg.vocab_size;
    let seq = input_ids.len();
    let eps = cfg.mla.rms_norm_eps;
    let positions: Vec<usize> = (0..seq).collect();
    let inv_main = cfg.inv_freq_main();
    let inv_comp = cfg.inv_freq_compress();
    let mask = sliding_mask(seq, cfg.mla.sliding_window);
    let hcd = hc * h;

    // First stage embeds + expands to hc streams; later stages inherit the hop.
    let mut streams = if layer_start == 0 {
        let emb = mv.embed_rows(input_ids, vocab, h);
        let mut s = vec![0.0f32; seq * hcd];
        for si in 0..seq {
            for k in 0..hc {
                s[si * hcd + k * h..si * hcd + k * h + h]
                    .copy_from_slice(&emb[si * h..(si + 1) * h]);
            }
        }
        s
    } else {
        let s = streams_in.expect("mid/last PP window needs streams_in");
        assert_eq!(s.len(), seq * hcd, "streams_in shape [seq, hc*h]");
        s
    };

    for li in layer_start..layer_end {
        streams = decoder_layer_mv(cfg, li, streams, input_ids, &positions, &mask, &inv_main, &inv_comp, mv);
    }

    if layer_end == cfg.num_hidden_layers {
        let collapsed = hyper_head_mv(&streams, cfg, mv, seq);
        let normed = rmsnorm_rows(&collapsed, &mv.dense("model.norm.weight"), seq, h, eps);
        WindowOut::Logits(mv.mm("lm_head", &normed, seq, h, vocab))
    } else {
        WindowOut::Streams(streams)
    }
}

/// Argmax of last-position logits.
pub fn argmax_last(logits: &[f32], seq: usize, vocab: usize) -> (u32, f32) {
    let last = &logits[(seq - 1) * vocab..];
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in last.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    (bi as u32, bv)
}

// ============================================================================
// Stateful DECODE (M2 GATE 2b) — rolling per-layer KV + compressor window
// ============================================================================
//
// ★ Prefix-stability (report Step 0, verified against transformers-5.8.1
// `deepseek_v4`): the CSA/HCA compressors are APPEND-ONLY. Each closed window of
// `compress_rate` source tokens emits exactly one compressed entry
// (`update_compressor_states` = `torch.cat`), the within-window softmax
// normalises over the window's slots ONLY (never across the whole sequence), and
// the CSA overlap reaches back at most one (already-closed) window's Ca slice.
// So a completed compressed entry is a function of past tokens alone and is NEVER
// recomputed when new tokens arrive — a rolling-window cache is BIT-EXACT.
//
// This decode keeps, per layer:
//   * `x_hist[li]` — the running `[t, h]` attention-input rows (the compressor's
//     `hidden_states`). The (already argmax-validated) stateless
//     `hca_compressor`/`csa_compressor` are re-run over this each step; because
//     completed entries are frozen, the result equals the batched-prefill
//     compressor bit-for-bit. Only the last query row's visibility is consumed.
//   * `kv_hist[li]` — the running `[t, hd]` RoPE'd sliding KV rows (K==V, MQA),
//     appended one per step; the single decode query attends the sliding window
//     over these.
// Everything else (mHC streams, MoE, grouped `wo_a`, HyperHead) is per-token with
// no cross-token state, reused verbatim from the prefill path. The single decode
// query at absolute position `pos` therefore reproduces the batched-prefill row
// `pos` op-for-op (proven by the `chained-decode == prefill` gate).

/// Rolling decode cache for one DSV4 PP window `[layer_start, layer_end)`.
pub struct Dsv4DecodeCache {
    /// `[num_layers][t*h]` attention-input history (compressor `hidden_states`).
    x_hist: Vec<Vec<f32>>,
    /// `[num_layers][t*hd]` RoPE'd sliding KV rows (K==V).
    kv_hist: Vec<Vec<f32>>,
    /// LEVER (VLLM_VULKAN_DSV4_COMPRESSOR_CACHE): per-layer RoPE'd compressed-KV
    /// plane for the CLOSED windows (`[n_closed, hd]`). Append-only / prefix-stable;
    /// grown one window at a time as `t1` crosses `k*m`. Empty when the flag is OFF.
    ckv_hist: Vec<Vec<f32>>,
    /// Number of closed compressed windows already committed to `ckv_hist[li]`.
    ckv_nwin: Vec<usize>,
    /// Tokens ingested so far == the NEXT token's absolute position.
    pos: usize,
    layer_start: usize,
    layer_end: usize,
}

impl Dsv4DecodeCache {
    /// Whole-model decode cache (single node).
    pub fn new(cfg: &Dsv4Config) -> Self {
        Self::new_window(cfg, 0, cfg.num_hidden_layers)
    }
    /// Per-PP-stage decode cache holding only `[ls, le)` (other slots stay empty).
    pub fn new_window(cfg: &Dsv4Config, ls: usize, le: usize) -> Self {
        let n = cfg.num_hidden_layers;
        Dsv4DecodeCache {
            x_hist: vec![Vec::new(); n],
            kv_hist: vec![Vec::new(); n],
            ckv_hist: vec![Vec::new(); n],
            ckv_nwin: vec![0usize; n],
            pos: 0,
            layer_start: ls,
            layer_end: le,
        }
    }
    /// Number of tokens ingested (== next token's position).
    pub fn len(&self) -> usize {
        self.pos
    }
    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }
}

/// Coarse per-token/per-stage phase profiler for the DSV4 decode path, gated by
/// `VLLM_VULKAN_DSV4_PROFILE=1` (default-OFF, behavior-neutral — no math changes,
/// no allocations on the OFF path). Accumulates wall-time per phase bucket across
/// the layers of ONE `decode_step_window` call (== one token on this PP stage) and
/// dumps a compact per-stage line. Buckets mirror the host-orchestration sinks
/// called out in the decode-perf review: HC/Sinkhorn (host mHC), MLA input
/// projections (wq_a/wq_b/wkv/lm_head standalone blocking `mm`), host compressor/
/// indexer re-dequant, resident attention tail, MoE router (host), MoE expert
/// compute, plus an orchestration-visible synchronous submit+readback round-trip
/// count. Thread-local (decode is single-threaded per stage process); the flag is
/// read once via `OnceLock`.
mod dsv4_prof {
    use std::cell::RefCell;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    #[derive(Clone, Copy)]
    pub enum B {
        Hc,
        MlaProj,
        Compressor,
        AttnTail,
        Router,
        Moe,
    }

    #[derive(Default)]
    struct Acc {
        hc: Duration,
        mla_proj: Duration,
        compressor: Duration,
        attn_tail: Duration,
        router: Duration,
        moe: Duration,
        roundtrips: u64,
    }

    thread_local! {
        static ACC: RefCell<Acc> = RefCell::new(Acc::default());
    }
    static ON: OnceLock<bool> = OnceLock::new();

    #[inline]
    pub fn enabled() -> bool {
        *ON.get_or_init(|| {
            std::env::var("VLLM_VULKAN_DSV4_PROFILE")
                .map(|v| v != "0" && v != "false")
                .unwrap_or(false)
        })
    }

    #[inline]
    fn add(a: &mut Acc, b: B, d: Duration) {
        match b {
            B::Hc => a.hc += d,
            B::MlaProj => a.mla_proj += d,
            B::Compressor => a.compressor += d,
            B::AttnTail => a.attn_tail += d,
            B::Router => a.router += d,
            B::Moe => a.moe += d,
        }
    }

    /// Time `f`, adding its wall duration to bucket `b` and `rt` round-trips. On the
    /// OFF path this is a straight call to `f` (zero overhead).
    #[inline]
    pub fn timed<T>(b: B, rt: u64, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let t = Instant::now();
        let r = f();
        let d = t.elapsed();
        ACC.with(|a| {
            let mut a = a.borrow_mut();
            add(&mut a, b, d);
            a.roundtrips += rt;
        });
        r
    }

    /// Manual span start (for sections that don't factor cleanly into a closure).
    /// Returns `None` on the OFF path.
    #[inline]
    pub fn start() -> Option<Instant> {
        if enabled() {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Close a manual span opened by [`start`], attributing it to bucket `b`.
    #[inline]
    pub fn stop(b: B, rt: u64, t: Option<Instant>) {
        if let Some(t) = t {
            let d = t.elapsed();
            ACC.with(|a| {
                let mut a = a.borrow_mut();
                add(&mut a, b, d);
                a.roundtrips += rt;
            });
        }
    }

    /// Zero the per-token accumulator (called at the top of `decode_step_window`).
    #[inline]
    pub fn reset() {
        if !enabled() {
            return;
        }
        ACC.with(|a| *a.borrow_mut() = Acc::default());
    }

    /// Emit the per-token, per-stage bucket split (ms) + round-trip count.
    pub fn dump(ls: usize, le: usize) {
        if !enabled() {
            return;
        }
        ACC.with(|a| {
            let a = a.borrow();
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            let total = ms(a.hc) + ms(a.mla_proj) + ms(a.compressor) + ms(a.attn_tail) + ms(a.router) + ms(a.moe);
            println!(
                "[dsv4-prof L{ls}..{le}] hc={:.2} mla_proj={:.2} compressor={:.2} attn_tail={:.2} router={:.2} moe={:.2} accounted={:.2} roundtrips={} (ms/tok, this stage)",
                ms(a.hc), ms(a.mla_proj), ms(a.compressor), ms(a.attn_tail), ms(a.router), ms(a.moe), total, a.roundtrips
            );
        });
    }
}

/// One decode step over a PP window `[layer_start, layer_end)`: ingest a single
/// token (`id` for the first stage; `streams_in` `[1, hc*h]` for mid/last stages),
/// advance the rolling cache, and return the next-stage streams (mid) or `[vocab]`
/// logits (last stage). Chaining these over a sequence reproduces
/// [`forward_mv`]'s per-position logits bit-for-bit (prefix-stable — see above).
pub fn decode_step_window<M: Mv>(
    cfg: &Dsv4Config,
    id: u32,
    streams_in: Option<Vec<f32>>,
    cache: &mut Dsv4DecodeCache,
    mv: &mut M,
) -> WindowOut {
    let h = cfg.mla.hidden_size;
    let hc = cfg.hc_mult;
    let vocab = cfg.vocab_size;
    let eps = cfg.mla.rms_norm_eps;
    let hcd = hc * h;
    let pos = cache.pos;
    let ls = cache.layer_start;
    let le = cache.layer_end;
    let inv_main = cfg.inv_freq_main();
    let inv_comp = cfg.inv_freq_compress();
    dsv4_prof::reset();

    // First stage embeds + expands to hc streams; later stages inherit the hop.
    let mut streams = if ls == 0 {
        let emb = mv.embed_rows(&[id], vocab, h);
        let mut s = vec![0.0f32; hcd];
        for k in 0..hc {
            s[k * h..k * h + h].copy_from_slice(&emb[..h]);
        }
        s
    } else {
        let s = streams_in.expect("mid/last decode window needs streams_in");
        assert_eq!(s.len(), hcd, "streams_in shape [1, hc*h]");
        s
    };

    for li in ls..le {
        streams = decoder_layer_decode(cfg, li, streams, id, pos, &inv_main, &inv_comp, cache, mv);
    }
    cache.pos += 1;

    if le == cfg.num_hidden_layers {
        let collapsed = hyper_head_mv(&streams, cfg, mv, 1);
        let normed = rmsnorm_rows(&collapsed, &mv.dense("model.norm.weight"), 1, h, eps);
        let logits = dsv4_prof::timed(dsv4_prof::B::MlaProj, 1, || mv.mm("lm_head", &normed, 1, h, vocab));
        dsv4_prof::dump(ls, le);
        WindowOut::Logits(logits)
    } else {
        dsv4_prof::dump(ls, le);
        WindowOut::Streams(streams)
    }
}

/// Full-model single-token decode → `[vocab]` logits (single-node convenience).
pub fn decode_step<M: Mv>(cfg: &Dsv4Config, id: u32, cache: &mut Dsv4DecodeCache, mv: &mut M) -> Vec<f32> {
    match decode_step_window(cfg, id, None, cache, mv) {
        WindowOut::Logits(l) => l,
        WindowOut::Streams(_) => unreachable!("full decode window must produce logits"),
    }
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer_decode<M: Mv>(
    cfg: &Dsv4Config,
    li: usize,
    mut streams: Vec<f32>,
    id: u32,
    pos: usize,
    inv_main: &[f32],
    inv_comp: &[f32],
    cache: &mut Dsv4DecodeCache,
    mv: &mut M,
) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let hc = cfg.hc_mult;
    let eps = cfg.mla.rms_norm_eps;
    let iters = cfg.hc_sinkhorn_iters;
    let hc_eps = cfg.hc_eps;
    let p = format!("model.layers.{li}");
    let lt = cfg.layer_types[li];
    let mt = cfg.mlp_layer_types[li];

    // ---- attention site: mHC (attn) → rmsnorm → attention ----
    let attn_fn = mv.dense(&format!("{p}.attn_hc.fn"));
    let attn_base = mv.dense(&format!("{p}.attn_hc.base"));
    let attn_scale = mv.dense(&format!("{p}.attn_hc.scale"));
    let (post, comb, collapsed) = dsv4_prof::timed(dsv4_prof::B::Hc, 0, || {
        mv.hc_block(&streams, 1, hc, h, &attn_fn, &attn_base, &attn_scale, iters, hc_eps, eps)
    });
    let in_ln = mv.dense(&format!("{p}.attn_norm.weight"));
    let x_in = rmsnorm_rows(&collapsed, &in_ln, 1, h, eps);

    // FFN-site mHC weights + ffn_norm are needed early for the resident HC-site span
    // (they ride the attn-tail CB). Cheap dense reads — fetch up front either way so
    // the flag-OFF path stays arithmetically identical to the original ordering.
    let ffn_fn = mv.dense(&format!("{p}.ffn_hc.fn"));
    let ffn_base = mv.dense(&format!("{p}.ffn_hc.base"));
    let ffn_scale = mv.dense(&format!("{p}.ffn_hc.scale"));
    let post_ln = mv.dense(&format!("{p}.ffn_norm.weight"));

    // (post_f, comb_f) = the ffn-site mHC gates; x_f = rmsnorm of its collapsed
    // stream (the MoE input). Produced either by the fused tail span (GPU) or the
    // host sequence.
    let (post_f, comb_f, x_f): (Vec<f32>, Vec<f32>, Vec<f32>);
    if mv.hc_resident_fused() {
        // Resident HC-site span: the attn-site hc_residual_mix + the ffn-site mHC
        // ride the attn-tail submit. `streams'` + the ffn mHC gates/collapsed all
        // return in the tail's single readback — the 43 ffn-site Sinkhorns leave the
        // host critical path (see dsv4_hc_resident_enabled).
        let trail = HcTrailIn {
            post_a: &post, comb_a: &comb, streams: &streams,
            ffn_fn: &ffn_fn, ffn_base: &ffn_base, ffn_scale: &ffn_scale,
            hc, iters, hc_eps,
        };
        match attention_layer_decode(cfg, li, lt, &x_in, pos, inv_main, inv_comp, cache, mv, Some(trail)) {
            AttnDecodeOut::Fused { streams2, post_f: pf, comb_f: cf, coll_f } => {
                streams = streams2;
                x_f = rmsnorm_rows(&coll_f, &post_ln, 1, h, eps);
                post_f = pf;
                comb_f = cf;
            }
            AttnDecodeOut::Plain(_) => unreachable!("fused trail must yield AttnDecodeOut::Fused"),
        }
    } else {
        // Host path (byte-identical to the original): attn_out → attn-site
        // hc_residual_mix → ffn-site mHC (host f64) → rmsnorm.
        let attn_out = match attention_layer_decode(cfg, li, lt, &x_in, pos, inv_main, inv_comp, cache, mv, None) {
            AttnDecodeOut::Plain(v) => v,
            AttnDecodeOut::Fused { .. } => unreachable!("no trail must yield AttnDecodeOut::Plain"),
        };
        streams = hc_residual_mix(&post, &attn_out, &comb, &streams, 1, hc, h);
        let (pf, cf, coll_f) = dsv4_prof::timed(dsv4_prof::B::Hc, 0, || {
            mv.hc_block(&streams, 1, hc, h, &ffn_fn, &ffn_base, &ffn_scale, iters, hc_eps, eps)
        });
        x_f = rmsnorm_rows(&coll_f, &post_ln, 1, h, eps);
        post_f = pf;
        comb_f = cf;
    }

    // ---- ffn site: MoE + trailing ffn-site hc_residual_mix (host) ----
    // M1 seam: MoE goes GPU-resident (1-CB) when VLLM_VULKAN_DSV4_1CB=1 on the
    // Dsv4GpuStage backend; DEFAULT-OFF → the bit-exact host `moe_layer_mv`.
    let mlp_out = mv.moe_block(cfg, li, mt, &x_f, &[id]);
    streams = hc_residual_mix(&post_f, &mlp_out, &comb_f, &streams, 1, hc, h);
    streams
}

/// Single-token attention over the rolling cache. Recomputes the (frozen-prefix)
/// compressor over `x_hist[li]`, appends this token's sliding KV row, and runs the
/// eager MLA for the single query at absolute position `pos`. Bit-identical to
/// [`attention_layer_mv`]'s row-`pos` output.
#[allow(clippy::too_many_arguments)]
/// Trailing HC-site ops folded onto the attn-tail submit (resident HC-site span).
/// `post_a`/`comb_a` are THIS layer's attn-site mHC gates; `streams` the pre-attn
/// residual stack; `ffn_*` the ffn-site mHC weights (see [`Mv::attn_tail_hc`]).
struct HcTrailIn<'a> {
    post_a: &'a [f32],
    comb_a: &'a [f32],
    streams: &'a [f32],
    ffn_fn: &'a [f32],
    ffn_base: &'a [f32],
    ffn_scale: &'a [f32],
    hc: usize,
    iters: usize,
    hc_eps: f32,
}

/// Result of [`attention_layer_decode`]: `Plain` = just the branch output
/// `attn_out [h]` (no trail); `Fused` = the trail rode the tail CB, so the attn-site
/// `hc_residual_mix` output `streams'` + the ffn-site mHC (`post_f`/`comb_f`/
/// `coll_f`) already came back in the tail's single readback.
enum AttnDecodeOut {
    Plain(Vec<f32>),
    Fused { streams2: Vec<f32>, post_f: Vec<f32>, comb_f: Vec<f32>, coll_f: Vec<f32> },
}

#[allow(clippy::too_many_arguments)]
fn attention_layer_decode<M: Mv>(
    cfg: &Dsv4Config,
    li: usize,
    lt: LayerType,
    x_cur: &[f32],
    pos: usize,
    inv_main: &[f32],
    inv_comp: &[f32],
    cache: &mut Dsv4DecodeCache,
    mv: &mut M,
    hc_trail: Option<HcTrailIn>,
) -> AttnDecodeOut {
    let h = cfg.mla.hidden_size;
    let nh = cfg.mla.num_attention_heads;
    let hd = cfg.mla.head_dim;
    let ql = cfg.mla.q_lora_rank;
    let olr = cfg.mla.o_lora_rank;
    let g = cfg.mla.o_groups;
    let eps = cfg.mla.rms_norm_eps;
    let sw = cfg.mla.sliding_window;
    let p = format!("model.layers.{li}");

    // Append this token's compressor input to the rolling history.
    cache.x_hist[li].extend_from_slice(x_cur);
    let t1 = cache.x_hist[li].len() / h; // == pos + 1
    let positions: Vec<usize> = (0..t1).collect();

    let (inv, scaling_rope) = if lt == LayerType::Sliding {
        (inv_main, 1.0f32)
    } else {
        (inv_comp, cfg.rope.compress_scaling)
    };
    let (cos, sin) = rope_cos_sin(&[pos], inv, scaling_rope);
    let rope_dim = 2 * cos.len(); // single position → cos.len() == inv.len()

    let w_q_a_norm = mv.dense(&format!("{p}.attn.q_norm.weight"));
    let w_kv_norm = mv.dense(&format!("{p}.attn.kv_norm.weight"));
    let sinks = mv.dense(&format!("{p}.attn.attn_sink"));

    // Compressor over the FULL history (completed entries frozen → matches
    // batched prefill). We consume only the LAST query row's visibility.
    // The full-history clone is ONLY needed by the non-incremental (recompute)
    // arms; when the incremental cache (VLLM_VULKAN_DSV4_COMPRESSOR_CACHE) covers
    // this layer it reads `cache.x_hist[li]` directly, so we skip the O(T) memcpy.
    let cache_on = dsv4_compressor_cache_enabled();
    let need_x_hist = match lt {
        LayerType::Sliding => false, // sliding attention never touches the compressor
        LayerType::HeavilyCompressed => !cache_on,
        LayerType::CompressedSparse => {
            let n_win_pred = t1 / cfg.compress_rate_csa;
            let short_circuit = dsv4_csa_shortcircuit_enabled() && n_win_pred <= cfg.index_topk;
            !(cache_on && short_circuit)
        }
    };
    let x_hist = if need_x_hist { cache.x_hist[li].clone() } else { Vec::new() };
    let _comp_t = dsv4_prof::start();
    let (compressed_kv, block_bias_last): (Vec<f32>, Vec<f32>) = match lt {
        LayerType::Sliding => (Vec::new(), Vec::new()),
        LayerType::HeavilyCompressed => {
            let m = cfg.compress_rate_hca;
            let ape = mv.dense(&format!("{p}.attn.compressor.ape"));
            let norm = mv.dense(&format!("{p}.attn.compressor.norm.weight"));
            if dsv4_compressor_cache_enabled() {
                // INCREMENTAL: pool only newly-closed windows; the last query's HCA
                // visibility is provably all-visible (threshold == n_win) so
                // block_bias_last is all-zeros. Byte-identical to the full path.
                let n_win = t1 / m;
                for w in cache.ckv_nwin[li]..n_win {
                    let src = &cache.x_hist[li][w * m * h..(w + 1) * m * h];
                    let kv = mv.mm(&format!("{p}.attn.compressor.wkv"), src, m, h, hd);
                    let gate = mv.mm(&format!("{p}.attn.compressor.wgate"), src, m, h, hd);
                    let row = crate::dsv4_dsa::hca_compress_window_incr(
                        &kv, &gate, hd, m, w, &ape, &norm, eps, inv, scaling_rope,
                    );
                    cache.ckv_hist[li].extend_from_slice(&row);
                }
                cache.ckv_nwin[li] = n_win;
                (cache.ckv_hist[li].clone(), vec![0.0f32; n_win])
            } else {
                // LEVER #2: run the compressor projections through the resident matvec
                // seam (`mm`). NON-resident → byte-identical to the former
                // `cpu_matmul(x, dq_linear(w))`; resident (flag ON) → on-device, killing
                // the per-token 6-bit re-dequant. Pool/rope/bias math unchanged (`_pre`).
                let kv = mv.mm(&format!("{p}.attn.compressor.wkv"), &x_hist, t1, h, hd);
                let gate = mv.mm(&format!("{p}.attn.compressor.wgate"), &x_hist, t1, h, hd);
                let (ckv, vis) = crate::dsv4_dsa::hca_compressor_pre(
                    &kv, &gate, t1, hd, m, &positions, &ape, &norm, eps, inv, scaling_rope,
                );
                let n_win = ckv.len() / hd;
                let last = if n_win == 0 { Vec::new() } else { vis[(t1 - 1) * n_win..t1 * n_win].to_vec() };
                (ckv, vis_to_additive(&last))
            }
        }
        LayerType::CompressedSparse => {
            let m = cfg.compress_rate_csa;
            let ihd = cfg.index_head_dim;
            let inh = cfg.index_n_heads;
            // Window count is deterministic (== `window_pool`'s `s/m`), so the
            // short-circuit regime is known BEFORE any projection runs.
            let n_win_pred = t1 / m;
            // CSA INDEXER SHORT-CIRCUIT (VLLM_VULKAN_DSV4_CSA_SHORTCIRCUIT, default-ON):
            // for the consumed LAST query row, whenever `n_win <= index_topk` the
            // Lightning-Indexer's top-k is `top_k = min(index_topk, n_win) = n_win`
            // and every window `w < threshold = t1/m = n_win` is admitted
            // (`indexer_topk_pre` :469/:494-500) — so `vis[last]` is all-visible and
            // `block_bias_last` is all-zeros REGARDLESS of scores. The entire indexer
            // (history `wq_a`, the four `ix_*` projections, ck pool/RoPE, scoring,
            // top-k) is therefore dead work. We keep ONLY the outer compressor
            // (`compressor.wkv`/`wgate` → pooled `compressed_kv`, still consumed by
            // the attention tail) and emit the all-zeros bias directly. Above 2048
            // ctx (`n_win > index_topk`) this is FALSE → full indexer path (unchanged).
            if dsv4_csa_shortcircuit_enabled() && n_win_pred <= cfg.index_topk {
                let c_ape = mv.dense(&format!("{p}.attn.compressor.ape"));
                let c_norm = mv.dense(&format!("{p}.attn.compressor.norm.weight"));
                if dsv4_compressor_cache_enabled() {
                    // INCREMENTAL: pool only newly-closed windows from the last one/two
                    // windows' projected rows (Ca=prev window, Cb=this window). The
                    // consumed last query is all-visible in the short-circuit regime, so
                    // block_bias_last is all-zeros. Byte-identical to the full outer path.
                    let n_win = n_win_pred;
                    for w in cache.ckv_nwin[li]..n_win {
                        let (lo, s) = if w == 0 { (0usize, m) } else { ((w - 1) * m, 2 * m) };
                        let src = &cache.x_hist[li][lo * h..(lo + s) * h];
                        let c_kv = mv.mm(&format!("{p}.attn.compressor.wkv"), src, s, h, 2 * hd);
                        let c_gate = mv.mm(&format!("{p}.attn.compressor.wgate"), src, s, h, 2 * hd);
                        let row = crate::dsv4_dsa::csa_compress_window_incr(
                            &c_kv, &c_gate, s, hd, m, w, &c_ape, &c_norm, eps, inv, scaling_rope,
                        );
                        cache.ckv_hist[li].extend_from_slice(&row);
                    }
                    cache.ckv_nwin[li] = n_win;
                    (cache.ckv_hist[li].clone(), vec![0.0f32; n_win])
                } else {
                    let c_kv = mv.mm(&format!("{p}.attn.compressor.wkv"), &x_hist, t1, h, 2 * hd);
                    let c_gate = mv.mm(&format!("{p}.attn.compressor.wgate"), &x_hist, t1, h, 2 * hd);
                    // Byte-identical to the outer-compressor portion of
                    // `csa_compressor_pre` (shared `csa_compressor_outer_pre`).
                    let (ckv, n_win) = crate::dsv4_dsa::csa_compressor_outer_pre(
                        &c_kv, &c_gate, t1, hd, m, &c_ape, &c_norm, eps, inv, scaling_rope,
                    );
                    // vis_to_additive(all-visible) == all zeros.
                    (ckv, vec![0.0f32; n_win])
                }
            } else {
                let q_a = rmsnorm_rows(&mv.mm(&format!("{p}.attn.wq_a"), &x_hist, t1, h, ql), &w_q_a_norm, t1, ql, eps);
                // LEVER #2: outer-compressor + indexer projections via the resident
                // matvec seam (see the HCA note). Dense (ape/norm) stay host.
                let c_kv = mv.mm(&format!("{p}.attn.compressor.wkv"), &x_hist, t1, h, 2 * hd);
                let c_gate = mv.mm(&format!("{p}.attn.compressor.wgate"), &x_hist, t1, h, 2 * hd);
                let c_ape = mv.dense(&format!("{p}.attn.compressor.ape"));
                let c_norm = mv.dense(&format!("{p}.attn.compressor.norm.weight"));
                let ix_kv = mv.mm(&format!("{p}.attn.indexer.compressor.wkv"), &x_hist, t1, h, 2 * ihd);
                let ix_gate = mv.mm(&format!("{p}.attn.indexer.compressor.wgate"), &x_hist, t1, h, 2 * ihd);
                let ix_qflat = mv.mm(&format!("{p}.attn.indexer.wq_b"), &q_a, t1, ql, inh * ihd);
                let ix_wgt = mv.mm(&format!("{p}.attn.indexer.weights_proj"), &x_hist, t1, h, inh);
                let ix_pb = mv.dense(&format!("{p}.attn.indexer.compressor.ape"));
                let ix_norm = mv.dense(&format!("{p}.attn.indexer.compressor.norm.weight"));
                let ixp = crate::dsv4_dsa::IndexerProj {
                    kv: &ix_kv, gate: &ix_gate, q_flat: &ix_qflat, wgt: &ix_wgt,
                    position_bias: &ix_pb, kv_norm: &ix_norm,
                };
                let (ckv, vis) = mv.csa_compressor_pre(
                    &c_kv, &c_gate, t1, hd, m, &positions, &c_ape, &c_norm,
                    eps, inv, scaling_rope, inh, ihd, cfg.index_topk, &ixp,
                );
                let n_win = ckv.len() / hd;
                let last = if n_win == 0 { Vec::new() } else { vis[(t1 - 1) * n_win..t1 * n_win].to_vec() };
                (ckv, vis_to_additive(&last))
            }
        }
    };
    // compressor/indexer projections (resident under LEVER #2, else per-token mmap
    // re-dequant) + windowed compress; CSA layers also run wq_a `mm` over history.
    let comp_rt = if lt == LayerType::CompressedSparse { 1 } else { 0 };
    dsv4_prof::stop(dsv4_prof::B::Compressor, comp_rt, _comp_t);

    // Q path for the CURRENT token only.
    let wqa = dsv4_prof::timed(dsv4_prof::B::MlaProj, 1, || mv.mm(&format!("{p}.attn.wq_a"), x_cur, 1, h, ql));
    let q_a = rmsnorm_rows(&wqa, &w_q_a_norm, 1, ql, eps);
    let q_flat = dsv4_prof::timed(dsv4_prof::B::MlaProj, 1, || mv.mm(&format!("{p}.attn.wq_b"), &q_a, 1, ql, nh * hd)); // [nh, hd], si=0
    let mut q = unweighted_rmsnorm_rows(&q_flat, nh, hd, eps);
    apply_interleaved_rope_inplace(&mut q, nh, hd, rope_dim, &|_r| 0, &cos, &sin);

    // KV for the current token → RoPE at `pos` → append to the rolling sliding cache.
    let wkv = dsv4_prof::timed(dsv4_prof::B::MlaProj, 1, || mv.mm(&format!("{p}.attn.wkv"), x_cur, 1, h, hd));
    let mut kv_cur = rmsnorm_rows(&wkv, &w_kv_norm, 1, hd, eps);
    apply_interleaved_rope_inplace(&mut kv_cur, 1, hd, rope_dim, &|_r| 0, &cos, &sin);
    cache.kv_hist[li].extend_from_slice(&kv_cur);
    let kv_sliding = &cache.kv_hist[li]; // [t1, hd]

    // Resident-capable attention TAIL: per-head eager softmax (sink + sliding +
    // compressed-KV block_bias) → output-rope conjugate → grouped block-diagonal
    // wo_a → wo_b. HOST by default ([`attn_tail_host_proj`]); Dsv4GpuStage records
    // dsv4_mla_softmax + grouped wo_a + wo_b into ONE command buffer when
    // VLLM_VULKAN_DSV4_1CB=1 (M2b). The [nh*hd] q is head-major (== ao_sd for
    // seq=1 decode), so group gg reads the contiguous ao slice [gg*per_g..].
    match hc_trail {
        None => AttnDecodeOut::Plain(dsv4_prof::timed(dsv4_prof::B::AttnTail, 1, || {
            mv.attn_tail(
                &p, &q, kv_sliding, &compressed_kv, &block_bias_last, &sinks, &cos, &sin,
                nh, hd, h, g, olr, sw, t1, rope_dim,
            )
        })),
        Some(tr) => {
            let (streams2, post_f, comb_f, coll_f) = dsv4_prof::timed(dsv4_prof::B::AttnTail, 1, || {
                mv.attn_tail_hc(
                    &p, &q, kv_sliding, &compressed_kv, &block_bias_last, &sinks, &cos, &sin,
                    nh, hd, h, g, olr, sw, t1, rope_dim,
                    tr.post_a, tr.comb_a, tr.streams, tr.ffn_fn, tr.ffn_base, tr.ffn_scale,
                    tr.hc, tr.iters, tr.hc_eps, eps,
                )
            });
            AttnDecodeOut::Fused { streams2, post_f, comb_f, coll_f }
        }
    }
}

/// Host implementation of the decode MLA attention TAIL → the grouped `wo_a`
/// projection `proj [g*olr]` (softmax + output-rope + block-diagonal `wo_a`),
/// bit-exact to the inline oracle formerly in [`attention_layer_decode`]. Shared
/// by the [`Mv::attn_tail`] default and the [`Dsv4GpuStage`] GPU-path fallback.
/// f64 softmax/accumulation, `block_bias_last` additive (0 visible / -inf hidden).
#[allow(clippy::too_many_arguments)]
fn attn_tail_host_proj(
    q: &[f32], kv_sliding: &[f32], compressed_kv: &[f32], block_bias_last: &[f32],
    sinks: &[f32], cos: &[f32], sin: &[f32],
    nh: usize, hd: usize, g: usize, olr: usize, sw: usize, t1: usize, rope_dim: usize,
    w_o_a: &[f32],
) -> Vec<f32> {
    let t_comp = compressed_kv.len() / hd;
    let total = t1 + t_comp;
    let scaling = (hd as f64).powf(-0.5);
    let qpos = t1 - 1;
    let mut ao = vec![0.0f32; nh * hd];
    for hh in 0..nh {
        let sink = sinks[hh] as f64;
        let qrow = &q[hh * hd..hh * hd + hd];
        let mut logits = vec![f64::NEG_INFINITY; total];
        let mut mx = sink;
        for ti in 0..total {
            let m = if ti < t1 {
                if qpos - ti >= sw { f64::NEG_INFINITY } else { 0.0 }
            } else {
                block_bias_last[ti - t1] as f64
            };
            if m == f64::NEG_INFINITY {
                continue;
            }
            let krow = if ti < t1 {
                &kv_sliding[ti * hd..ti * hd + hd]
            } else {
                &compressed_kv[(ti - t1) * hd..(ti - t1) * hd + hd]
            };
            let mut dot = 0.0f64;
            for d in 0..hd {
                dot += qrow[d] as f64 * krow[d] as f64;
            }
            let l = dot * scaling + m;
            logits[ti] = l;
            if l > mx {
                mx = l;
            }
        }
        let mut denom = (sink - mx).exp();
        for ti in 0..total {
            if logits[ti] != f64::NEG_INFINITY {
                logits[ti] = (logits[ti] - mx).exp();
                denom += logits[ti];
            } else {
                logits[ti] = 0.0;
            }
        }
        let orow = &mut ao[hh * hd..hh * hd + hd];
        for ti in 0..total {
            let pr = logits[ti] / denom;
            if pr == 0.0 {
                continue;
            }
            let vrow = if ti < t1 {
                &kv_sliding[ti * hd..ti * hd + hd]
            } else {
                &compressed_kv[(ti - t1) * hd..(ti - t1) * hd + hd]
            };
            for d in 0..hd {
                orow[d] += (pr * vrow[d] as f64) as f32;
            }
        }
    }
    // output-rope conjugate (-sin) at `pos`.
    let neg_sin: Vec<f32> = sin.iter().map(|v| -v).collect();
    apply_interleaved_rope_inplace(&mut ao, nh, hd, rope_dim, &|_r| 0, cos, &neg_sin);
    // grouped block-diagonal wo_a: ao head-major == ao_sd for seq=1.
    let per_g = (nh * hd) / g;
    let mut proj = vec![0.0f32; g * olr];
    for gg in 0..g {
        let xin = &ao[gg * per_g..gg * per_g + per_g];
        for o in 0..olr {
            let wrow = &w_o_a[(gg * olr + o) * per_g..(gg * olr + o) * per_g + per_g];
            let mut acc = 0.0f64;
            for k in 0..per_g {
                acc += xin[k] as f64 * wrow[k] as f64;
            }
            proj[gg * olr + o] = acc as f32;
        }
    }
    proj
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer_mv<M: Mv>(
    cfg: &Dsv4Config,
    li: usize,
    mut streams: Vec<f32>,
    input_ids: &[u32],
    positions: &[usize],
    mask: &[f32],
    inv_main: &[f32],
    inv_comp: &[f32],
    mv: &mut M,
) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let hc = cfg.hc_mult;
    let seq = input_ids.len();
    let eps = cfg.mla.rms_norm_eps;
    let iters = cfg.hc_sinkhorn_iters;
    let hc_eps = cfg.hc_eps;
    let p = format!("model.layers.{li}");
    let lt = cfg.layer_types[li];
    let mt = cfg.mlp_layer_types[li];

    // ---- attention site ----
    let attn_fn = mv.dense(&format!("{p}.attn_hc.fn"));
    let attn_base = mv.dense(&format!("{p}.attn_hc.base"));
    let attn_scale = mv.dense(&format!("{p}.attn_hc.scale"));
    let (post, comb, collapsed) =
        mv.hc_block(&streams, seq, hc, h, &attn_fn, &attn_base, &attn_scale, iters, hc_eps, eps);
    let in_ln = mv.dense(&format!("{p}.attn_norm.weight"));
    let x_in = rmsnorm_rows(&collapsed, &in_ln, seq, h, eps);
    let attn_out = attention_layer_mv(cfg, li, lt, &x_in, seq, positions, mask, inv_main, inv_comp, mv);
    streams = hc_residual_mix(&post, &attn_out, &comb, &streams, seq, hc, h);

    // ---- ffn site ----
    let ffn_fn = mv.dense(&format!("{p}.ffn_hc.fn"));
    let ffn_base = mv.dense(&format!("{p}.ffn_hc.base"));
    let ffn_scale = mv.dense(&format!("{p}.ffn_hc.scale"));
    let (post, comb, collapsed) =
        mv.hc_block(&streams, seq, hc, h, &ffn_fn, &ffn_base, &ffn_scale, iters, hc_eps, eps);
    let post_ln = mv.dense(&format!("{p}.ffn_norm.weight"));
    let x_in = rmsnorm_rows(&collapsed, &post_ln, seq, h, eps);
    let mlp_out = moe_layer_mv(cfg, li, mt, &x_in, input_ids, mv);
    streams = hc_residual_mix(&post, &mlp_out, &comb, &streams, seq, hc, h);
    streams
}

/// One attention sub-layer over the backend. The 5 MLA projections that are plain
/// matvecs (`wq_a`/`wq_b`/`wkv`/`wo_b`) go through `mv.mm`; the DSA compressor +
/// Lightning-Indexer + grouped block-diagonal `wo_a` stay host (bit-exact). All
/// non-matmul ops (RMSNorm/RoPE/softmax+sink) are verbatim from `mla_core_ext`.
#[allow(clippy::too_many_arguments)]
fn attention_layer_mv<M: Mv>(
    cfg: &Dsv4Config,
    li: usize,
    lt: LayerType,
    x: &[f32],
    seq: usize,
    positions: &[usize],
    mask: &[f32],
    inv_main: &[f32],
    inv_comp: &[f32],
    mv: &mut M,
) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let nh = cfg.mla.num_attention_heads;
    let hd = cfg.mla.head_dim;
    let ql = cfg.mla.q_lora_rank;
    let olr = cfg.mla.o_lora_rank;
    let g = cfg.mla.o_groups;
    let eps = cfg.mla.rms_norm_eps;
    let p = format!("model.layers.{li}");

    let (inv, scaling_rope) = if lt == LayerType::Sliding {
        (inv_main, 1.0f32)
    } else {
        (inv_comp, cfg.rope.compress_scaling)
    };
    let (cos, sin) = rope_cos_sin(positions, inv, scaling_rope);

    // Weights fetched host-side for the compressor + grouped wo_a (bit-exact).
    let w_q_a_norm = mv.dense(&format!("{p}.attn.q_norm.weight"));
    let w_kv_norm = mv.dense(&format!("{p}.attn.kv_norm.weight"));
    let w_o_a = mv.dq_linear(&format!("{p}.attn.wo_a"), g * olr, nh * hd / g);
    let sinks = mv.dense(&format!("{p}.attn.attn_sink"));

    // CSA/HCA compressed KV + additive block_bias (empty for sliding).
    let (compressed_kv, block_bias): (Vec<f32>, Vec<f32>) = match lt {
        LayerType::Sliding => (Vec::new(), Vec::new()),
        LayerType::HeavilyCompressed => {
            let m = cfg.compress_rate_hca;
            let (ckv, vis) = hca_compressor(
                x, seq, h, hd, m, positions,
                &mv.dq_linear(&format!("{p}.attn.compressor.wkv"), hd, h),
                &mv.dq_linear(&format!("{p}.attn.compressor.wgate"), hd, h),
                &mv.dense(&format!("{p}.attn.compressor.ape")),
                &mv.dense(&format!("{p}.attn.compressor.norm.weight")),
                eps, inv, scaling_rope,
            );
            (ckv, vis_to_additive(&vis))
        }
        LayerType::CompressedSparse => {
            let m = cfg.compress_rate_csa;
            let ihd = cfg.index_head_dim;
            let inh = cfg.index_n_heads;
            // q_residual = q_a_norm(q_a_proj(x)) — via the backend matvec.
            let q_a = rmsnorm_rows(&mv.mm(&format!("{p}.attn.wq_a"), x, seq, h, ql), &w_q_a_norm, seq, ql, eps);
            let ix_kv = mv.dq_linear(&format!("{p}.attn.indexer.compressor.wkv"), 2 * ihd, h);
            let ix_gate = mv.dq_linear(&format!("{p}.attn.indexer.compressor.wgate"), 2 * ihd, h);
            let ix_pb = mv.dense(&format!("{p}.attn.indexer.compressor.ape"));
            let ix_norm = mv.dense(&format!("{p}.attn.indexer.compressor.norm.weight"));
            let ix_qb = mv.dq_linear(&format!("{p}.attn.indexer.wq_b"), inh * ihd, ql);
            let ix_wp = mv.dq_linear(&format!("{p}.attn.indexer.weights_proj"), inh, h);
            let ix = IndexerWeights {
                kv_proj: &ix_kv, gate_proj: &ix_gate, position_bias: &ix_pb, kv_norm: &ix_norm,
                q_b_proj: &ix_qb, weights_proj: &ix_wp,
            };
            // Hoist the compressor weights out of the call so no `mv` borrow is live
            // across the `&mut mv` seam method (the GPU override needs `&mut self`).
            let c_wkv = mv.dq_linear(&format!("{p}.attn.compressor.wkv"), 2 * hd, h);
            let c_wgate = mv.dq_linear(&format!("{p}.attn.compressor.wgate"), 2 * hd, h);
            let c_ape = mv.dense(&format!("{p}.attn.compressor.ape"));
            let c_norm = mv.dense(&format!("{p}.attn.compressor.norm.weight"));
            let (ckv, vis) = mv.csa_compressor(
                x, &q_a, seq, h, hd, m, positions, &c_wkv, &c_wgate, &c_ape, &c_norm,
                eps, inv, scaling_rope, ql, inh, ihd, cfg.index_topk, &ix,
            );
            (ckv, vis_to_additive(&vis))
        }
    };

    // ---- inlined mla_core_ext with mv.mm on the 4 plain projections ----
    let s = seq;
    let rope_dim = 2 * (cos.len() / s);
    let scaling = (hd as f64).powf(-0.5);
    let t_comp = compressed_kv.len() / hd;
    let total = s + t_comp;

    // Q path
    let q_a = mv.mm(&format!("{p}.attn.wq_a"), x, s, h, ql);
    let q_a = rmsnorm_rows(&q_a, &w_q_a_norm, s, ql, eps);
    let q_flat = mv.mm(&format!("{p}.attn.wq_b"), &q_a, s, ql, nh * hd);
    let mut q = vec![0.0f32; nh * s * hd];
    for si in 0..s {
        for hh in 0..nh {
            let srcq = &q_flat[si * nh * hd + hh * hd..si * nh * hd + hh * hd + hd];
            q[(hh * s + si) * hd..(hh * s + si) * hd + hd].copy_from_slice(srcq);
        }
    }
    let mut q = unweighted_rmsnorm_rows(&q, nh * s, hd, eps);
    apply_interleaved_rope_inplace(&mut q, nh * s, hd, rope_dim, &|r| r % s, &cos, &sin);

    // KV path
    let kv = mv.mm(&format!("{p}.attn.wkv"), x, s, h, hd);
    let kv = rmsnorm_rows(&kv, &w_kv_norm, s, hd, eps);
    let mut kv = kv;
    apply_interleaved_rope_inplace(&mut kv, s, hd, rope_dim, &|r| r, &cos, &sin);
    let mut kv_full = vec![0.0f32; total * hd];
    kv_full[..s * hd].copy_from_slice(&kv);
    if t_comp > 0 {
        kv_full[s * hd..].copy_from_slice(&compressed_kv);
    }

    // eager attention: sink + extended mask; K==V single head broadcast.
    let mut ao = vec![0.0f32; nh * s * hd];
    for hh in 0..nh {
        let sink = sinks[hh] as f64;
        for si in 0..s {
            let qrow = &q[(hh * s + si) * hd..(hh * s + si) * hd + hd];
            let mut logits = vec![f64::NEG_INFINITY; total];
            let mut mx = sink;
            for ti in 0..total {
                let m = if ti < s {
                    mask[si * s + ti] as f64
                } else {
                    block_bias[si * t_comp + (ti - s)] as f64
                };
                if m == f64::NEG_INFINITY {
                    continue;
                }
                let krow = &kv_full[ti * hd..ti * hd + hd];
                let mut dot = 0.0f64;
                for d in 0..hd {
                    dot += qrow[d] as f64 * krow[d] as f64;
                }
                let l = dot * scaling + m;
                logits[ti] = l;
                if l > mx {
                    mx = l;
                }
            }
            let mut denom = (sink - mx).exp();
            for ti in 0..total {
                if logits[ti] != f64::NEG_INFINITY {
                    logits[ti] = (logits[ti] - mx).exp();
                    denom += logits[ti];
                } else {
                    logits[ti] = 0.0;
                }
            }
            let orow = &mut ao[(hh * s + si) * hd..(hh * s + si) * hd + hd];
            for ti in 0..total {
                let pr = logits[ti] / denom;
                if pr == 0.0 {
                    continue;
                }
                let vrow = &kv_full[ti * hd..ti * hd + hd];
                for d in 0..hd {
                    orow[d] += (pr * vrow[d] as f64) as f32;
                }
            }
        }
    }

    // output-rope conjugate (-sin)
    let neg_sin: Vec<f32> = sin.iter().map(|v| -v).collect();
    apply_interleaved_rope_inplace(&mut ao, nh * s, hd, rope_dim, &|r| r % s, &cos, &neg_sin);

    // reshape head-major → [S, nh*hd]
    let mut ao_sd = vec![0.0f32; s * nh * hd];
    for hh in 0..nh {
        for si in 0..s {
            let srco = &ao[(hh * s + si) * hd..(hh * s + si) * hd + hd];
            ao_sd[si * nh * hd + hh * hd..si * nh * hd + hh * hd + hd].copy_from_slice(srco);
        }
    }

    // grouped o_lora (host, block-diagonal wo_a)
    let per_g = (nh * hd) / g;
    let mut proj = vec![0.0f32; s * g * olr];
    for si in 0..s {
        for gg in 0..g {
            let xin = &ao_sd[si * nh * hd + gg * per_g..si * nh * hd + gg * per_g + per_g];
            for o in 0..olr {
                let wrow = &w_o_a[(gg * olr + o) * per_g..(gg * olr + o) * per_g + per_g];
                let mut acc = 0.0f64;
                for k in 0..per_g {
                    acc += xin[k] as f64 * wrow[k] as f64;
                }
                proj[si * g * olr + gg * olr + o] = acc as f32;
            }
        }
    }
    mv.mm(&format!("{p}.attn.wo_b"), &proj, s, g * olr, h)
}

/// One MoE sub-layer over the backend. Router is host (plain bf16 gate); each
/// routed/shared expert's gate/up/down go through `mv.mm_expert`/`mv.mm` (GPU-
/// resident), with the SwiGLU accumulation verbatim from `dsv4_moe`.
fn moe_layer_mv<M: Mv>(
    cfg: &Dsv4Config,
    li: usize,
    mt: MlpType,
    x: &[f32],
    input_ids: &[u32],
    mv: &mut M,
) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let ii = cfg.moe_intermediate_size;
    let ne = cfg.num_local_experts;
    let tk = cfg.num_experts_per_tok;
    let seq = input_ids.len();
    let limit = cfg.swiglu_limit as f64;
    let p = format!("model.layers.{li}");

    let _router_t = dsv4_prof::start();
    // LEVER: resident f32 gate (VLLM_VULKAN_DSV4_RESIDENT_ROUTER) skips the per-token
    // bf16→f32 dequant; DEFAULT (None) falls back to the byte-identical per-token dequant.
    let gate_name = format!("{p}.ffn.gate.weight");
    let gate_owned;
    let gate_w: &[f32] = match mv.resident_router_gate(&gate_name) {
        Some(w) => w,
        None => {
            gate_owned = mv.dq_linear(&gate_name, ne, h);
            &gate_owned
        }
    };
    let (idx, wts) = match mt {
        MlpType::Moe => {
            let corr = mv.dense(&format!("{p}.ffn.gate.e_score_correction_bias"));
            topk_router(x, gate_w, &corr, seq, h, ne, tk, cfg.routed_scaling_factor, cfg.norm_topk_prob)
        }
        MlpType::HashMoe => {
            let tid2eid = mv.dense_i64(&format!("{p}.ffn.gate.tid2eid"));
            hash_router(x, gate_w, &tid2eid, input_ids, seq, h, ne, tk, cfg.routed_scaling_factor, cfg.norm_topk_prob)
        }
    };
    dsv4_prof::stop(dsv4_prof::B::Router, 0, _router_t);
    // avoid an unused-import warning path: router already scored via sqrtsoftplus.
    let _ = sqrtsoftplus;
    let _moe_t = dsv4_prof::start();

    let gate_name = format!("{p}.ffn.switch_mlp.gate_proj");
    let up_name = format!("{p}.ffn.switch_mlp.up_proj");
    let down_name = format!("{p}.ffn.switch_mlp.down_proj");
    let sh_gate = format!("{p}.ffn.shared_experts.gate_proj");
    let sh_up = format!("{p}.ffn.shared_experts.up_proj");
    let sh_down = format!("{p}.ffn.shared_experts.down_proj");

    let mut out = vec![0.0f32; seq * h];
    for ti in 0..seq {
        let xrow = x[ti * h..(ti + 1) * h].to_vec();
        let base = ti * h;
        // routed experts (swiglu_limit clamp)
        for j in 0..tk {
            let e = idx[ti * tk + j];
            let w = wts[ti * tk + j];
            let gate = mv.mm_expert(&gate_name, e, &xrow, 1, h, ii);
            let up = mv.mm_expert(&up_name, e, &xrow, 1, h, ii);
            let mut hid = vec![0.0f32; ii];
            for k in 0..ii {
                let gv = (gate[k] as f64).min(limit);
                let uv = (up[k] as f64).clamp(-limit, limit);
                hid[k] = (silu_f64(gv) * uv) as f32;
            }
            let contrib = mv.mm_expert(&down_name, e, &hid, 1, ii, h);
            for c in 0..h {
                out[base + c] += contrib[c] * w;
            }
        }
        // shared expert (no clamp, weight 1.0)
        let sg = mv.mm(&sh_gate, &xrow, 1, h, ii);
        let su = mv.mm(&sh_up, &xrow, 1, h, ii);
        let mut hid = vec![0.0f32; ii];
        for k in 0..ii {
            hid[k] = (silu_f64(sg[k] as f64) * su[k] as f64) as f32;
        }
        let sh = mv.mm(&sh_down, &hid, 1, ii, h);
        for c in 0..h {
            out[base + c] += sh[c];
        }
    }
    dsv4_prof::stop(dsv4_prof::B::Moe, (seq * (tk * 3 + 3)) as u64, _moe_t);
    out
}

#[inline]
fn silu_f64(z: f64) -> f64 {
    z * (1.0 / (1.0 + (-z).exp()))
}

/// HyperHead final stream collapse (verbatim from `dsv4_forward::hyper_head`).
fn hyper_head_mv<M: Mv>(streams: &[f32], cfg: &Dsv4Config, mv: &M, seq: usize) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let hc = cfg.hc_mult;
    let hcd = hc * h;
    let eps = cfg.mla.rms_norm_eps;
    let hc_fn = mv.dense("model.hc_head.fn");
    let hc_base = mv.dense("model.hc_head.base");
    let hc_scale = mv.dense("model.hc_head.scale");
    let flat = unweighted_rmsnorm_rows(streams, seq, hcd, eps);
    let sc = hc_scale[0] as f64;
    let mut out = vec![0.0f32; seq * h];
    for si in 0..seq {
        let frow = &flat[si * hcd..(si + 1) * hcd];
        let mut pre = vec![0.0f64; hc];
        for k in 0..hc {
            let wrow = &hc_fn[k * hcd..(k + 1) * hcd];
            let mut acc = 0.0f64;
            for d in 0..hcd {
                acc += frow[d] as f64 * wrow[d] as f64;
            }
            pre[k] = 1.0 / (1.0 + (-(acc * sc + hc_base[k] as f64)).exp()) + cfg.hc_eps as f64;
        }
        let orow = &mut out[si * h..(si + 1) * h];
        for k in 0..hc {
            let strow = &streams[si * hcd + k * h..si * hcd + k * h + h];
            for d in 0..h {
                orow[d] += (pre[k] * strow[d] as f64) as f32;
            }
        }
    }
    out
}

// ============================================================================
// M0 — Resident 1-CB decode infra foundation (mirrors the qwen35 `q35r` WS3
// quintet: init_*res_bufs / ensure_*res / *r_meta / *r_rec_mv /
// forward_*_span_resident). See docs/dsv4-resident-1cb-plan.md.
//
// STATUS: M0 scaffolding only. The flag is DEFAULT-OFF and the span entry
// returns `None` (host fallthrough), so with the flag unset the decode is
// BYTE-IDENTICAL to the shipped host-orchestrated path (argmax 11111 preserved
// by construction). The resident buffer layout below is sized from cfg and is
// the concrete slot table M1+ will record into. Do NOT flip the default until
// the assembled-decode on-node gate (argmax 11111 + ms/tok A/B) passes — the
// cluster was OFFLINE when this landed (see the plan doc §7).
// ============================================================================

/// DSV4 decode resident 1-CB span path (M1 resident MoE + M2b resident MLA
/// attention tail). DEFAULT-ON as of the M2b GEN=64 gate (PP-10 @1850: prompt-final
/// argmax 11111 ` Paris`, 64/64-token gen BIT-IDENTICAL to the host oracle, 2.4–2.9×
/// faster — 3977→1357 ms/tok @GEN=8, 4751→1982 @GEN=64). `VLLM_VULKAN_DSV4_1CB=0`
/// (or `false`) still FORCES the bit-exact host oracle path (the OFF override kept
/// for A/B + fallback); unset or any other value → the resident path.
fn dsv4_1cb_enabled() -> bool {
    std::env::var("VLLM_VULKAN_DSV4_1CB").map(|v| v != "0" && v != "false").unwrap_or(true)
}

/// LEVER #2: upload the 6-bit compressor/indexer projection weights as RESIDENT GPU
/// buffers at load, so the per-token compressor stops re-dequantizing them from mmap
/// (the measured ~81%-of-decode host-orchestration sink). `VLLM_VULKAN_DSV4_RESIDENT_COMPRESSOR`,
/// DEFAULT-ON (5.45× decode, 1350.7→247.6 ms/tok, argmax 11111 + gen bit-identical @GEN=8 AND GEN=64).
/// Set `=0`/`=false` to force the host dequant path — byte-identical to the pre-lever compressor.
/// GTT-aware: 6-bit packed residency (~1/5 of the f32 materialization, ~+35MB/stage), NOT a host cache.
fn dsv4_resident_compressor_enabled() -> bool {
    std::env::var("VLLM_VULKAN_DSV4_RESIDENT_COMPRESSOR").map(|v| v != "0" && v != "false").unwrap_or(true)
}

/// LEVER: pre-dequantize the bf16 MoE-router gate weight ONCE at load and keep the f32
/// resident (host DRAM, `[ne,h]`×4B ≈ 4MB/layer — tiny, NOT GTT), so the per-token router
/// (both the resident 1-CB `moe_resident` path and the host `moe_layer_mv` fallback) stops
/// re-running the serial bf16→f32 dequant every token — the residual left after LEVER B
/// (rayon) parallelized the matmul but not the dequant. `VLLM_VULKAN_DSV4_RESIDENT_ROUTER`,
/// DEFAULT-ON (206.3→164.1 ms/tok −20.4%, router bucket 54.35→11.41, argmax 11111 + gen
/// bit-identical @GEN=8 AND GEN=64). Set `=0`/`=false` for the byte-identical per-token-dequant path.
fn dsv4_resident_router_enabled() -> bool {
    std::env::var("VLLM_VULKAN_DSV4_RESIDENT_ROUTER").map(|v| v != "0" && v != "false").unwrap_or(true)
}

/// CSA indexer short-circuit (`VLLM_VULKAN_DSV4_CSA_SHORTCIRCUIT`, DEFAULT-ON).
/// Below 2048 ctx the CSA Lightning-Indexer provably emits an all-visible
/// (all-zeros) `block_bias_last` for the consumed last query — top_k collapses to
/// `n_win` and every causal window is admitted regardless of score. So the whole
/// indexer (history `wq_a` + the four `ix_*` projections + ck pool/RoPE + scoring +
/// top-k) is dead work; we keep only the outer compressor. BIT-EXACT (proven by the
/// dsv4 chained-decode-vs-prefill oracle: flag ON vs OFF max_abs_diff == 0). Set
/// `=0`/`=false` to force the full per-token indexer path (byte-identical output).
fn dsv4_csa_shortcircuit_enabled() -> bool {
    std::env::var("VLLM_VULKAN_DSV4_CSA_SHORTCIRCUIT").map(|v| v != "0" && v != "false").unwrap_or(true)
}

/// Incremental (prefix-stable) compressor cache (`VLLM_VULKAN_DSV4_COMPRESSOR_CACHE`,
/// DEFAULT-ON). The decode compressor recomputes the CSA/HCA outer projection +
/// window-pool + RoPE over the FULL `x_hist` every token even though a completed
/// window is frozen (report P0). With this flag ON, each layer keeps the RoPE'd
/// compressed-KV plane for its CLOSED windows in [`Dsv4DecodeCache::ckv_hist`] and,
/// per token, only pools the window(s) that just closed (project ≤2m source rows,
/// pool, RoPE, append) — turning the O(T)/O(T²) per-token compressor into O(1)
/// amortized. Scoped to the CSA short-circuit regime (`n_win <= index_topk`, i.e.
/// below 2048 ctx, where the indexer is provably all-visible) and to HCA (pure
/// causal, last-query all-visible). Above 2048 ctx the full indexer path runs
/// unchanged. BIT-EXACT to the full recompute (chained-decode-vs-prefill oracle,
/// cache ON vs OFF max_abs_diff == 0; per-window helpers gated random-input in
/// `dsv4_dsa`).
///
/// DEFAULT-ON (gated 2026-08-18): PP-10 real-ckpt A/B (10 nodes @1850, BOUNDS
/// 0,5,10,15,19,23,27,31,35,39,43) — prompt-final argmax 11111 (' Paris') AND the
/// generated sequence BIT-IDENTICAL cache-ON vs cache-OFF across 650 decode tokens
/// (ctx 5..655; exercises HCA m=128 window-closes at 128/256/384/512/640 and CSA
/// m=4 throughout). Per-ctx wall OFF→ON: 488.4→228.6 ms @256 (2.1x), 998.4→247.6
/// @640 (4.0x); OFF grows O(t) (~1.5 ms/tok·ctx) while ON is nearly flat → the
/// speedup rises with context (extrapolates ~10x @1850). `=0`/`=false` forces the
/// full per-token recompute (byte-identical output).
fn dsv4_compressor_cache_enabled() -> bool {
    std::env::var("VLLM_VULKAN_DSV4_COMPRESSOR_CACHE").map(|v| v != "0" && v != "false").unwrap_or(true)
}

/// Slot indices into [`Dsv4ResBufs::bufs`] — the fixed, named resident decode
/// activations (the DSV4 analog of q35r's `Q35R_*` slots). One token (seq=1)
/// decode. Attention-branch and MoE-branch scratch reuse across the two HC sites
/// within a layer. Extend as M1/M2 wire more of the span resident.
#[allow(dead_code)]
mod dsv4r {
    /// `[hc*h]` — the manifold hyper-connection stream stack (the resident hidden
    /// carried layer→layer; written once at span top, read once at span end).
    pub const STREAMS: usize = 0;
    /// `[h]` — RMSNorm(collapsed) attention/ffn input.
    pub const XIN: usize = 1;
    /// `[h]` — mHC collapsed stream (per HC site).
    pub const COLLAPSED: usize = 2;
    /// `[hc]` — mHC post gates (per HC site).
    pub const POST: usize = 3;
    /// `[hc*hc]` — mHC combine matrix (per HC site).
    pub const COMB: usize = 4;
    /// `[h]` — attention or MoE branch output, before `hc_residual_mix`.
    pub const BRANCH_OUT: usize = 5;
    /// `[moe_intermediate]` — routed/shared expert SwiGLU hidden scratch.
    pub const FFN_HID: usize = 6;
    /// `[moe_intermediate]` — expert gate matvec out.
    pub const FFN_GATE: usize = 7;
    /// `[moe_intermediate]` — expert up matvec out.
    pub const FFN_UP: usize = 8;
    /// `[vocab]` — final logits (last stage).
    pub const VLOG: usize = 9;
    pub const COUNT: usize = 10;
}

/// The fixed slot-indexed resident-activation buffer table for the DSV4 1-CB
/// decode span. Allocated ONCE (host-coherent storage, one `compute::Buffer` per
/// slot) and reused every token; intermediates never round-trip to a host
/// `Vec<f32>` between ops. Mirrors q35r's `q35res_bufs` (`qwen35_forward.rs`).
#[allow(dead_code)] // M0 scaffolding — constructed by M1's ensure_dsv4res.
struct Dsv4ResBufs {
    bufs: Vec<compute::Buffer>,
    ready: bool,
}

impl Dsv4ResBufs {
    /// Size every slot from cfg and allocate resident storage (min 4 bytes, like
    /// q35r). Called once (M1 `ensure_dsv4res`). Sizing is correct-by-construction
    /// from the decode shapes in `decoder_layer_decode`/`moe_layer_mv`.
    #[allow(dead_code)] // M0 scaffolding — called by M1's ensure_dsv4res.
    fn init(eng: &mut compute::ComputeEngine, cfg: &Dsv4Config) -> Result<Dsv4ResBufs, String> {
        let h = cfg.mla.hidden_size;
        let hc = cfg.hc_mult;
        let ii = cfg.moe_intermediate_size;
        let vocab = cfg.vocab_size;
        let mut sz = vec![4usize; dsv4r::COUNT];
        sz[dsv4r::STREAMS] = (hc * h).max(1);
        sz[dsv4r::XIN] = h.max(1);
        sz[dsv4r::COLLAPSED] = h.max(1);
        sz[dsv4r::POST] = hc.max(1);
        sz[dsv4r::COMB] = (hc * hc).max(1);
        sz[dsv4r::BRANCH_OUT] = h.max(1);
        sz[dsv4r::FFN_HID] = ii.max(1);
        sz[dsv4r::FFN_GATE] = ii.max(1);
        sz[dsv4r::FFN_UP] = ii.max(1);
        sz[dsv4r::VLOG] = vocab.max(1);
        let mut bufs = Vec::with_capacity(dsv4r::COUNT);
        for n in sz {
            bufs.push(eng.alloc_host_coherent_storage((n * 4).max(4) as u64)?);
        }
        Ok(Dsv4ResBufs { bufs, ready: true })
    }
}

impl Dsv4GpuStage {
    /// M0 seam: try the resident 1-CB decode span for one token. Returns `None`
    /// when the flag is OFF (the default) OR the span is not yet wired for this
    /// stage/layer-type, so the caller (`decode_step_stage`) falls through to the
    /// bit-exact host-orchestrated path. When `Some(out)` is returned it MUST be
    /// bit-identical to that path (argmax 11111). M1+ fills in the body per
    /// docs/dsv4-resident-1cb-plan.md; today it is an always-`None` stub, so the
    /// decode is byte-identical whether or not the flag is set.
    fn try_decode_step_resident(
        &mut self,
        _id: u32,
        _streams_in: &Option<Vec<f32>>,
    ) -> Option<WindowOut> {
        if !dsv4_1cb_enabled() {
            return None;
        }
        // M1 TODO: ensure_dsv4res() → begin_forward_ring → per-layer resident CBs
        // (MoE span first) → single final readback. Until wired, fall through to
        // the host oracle so correctness is never at risk.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// ★ M2b flag semantics (offline, no GPU): the resident-1CB flag is DEFAULT-ON
    /// (GEN=64 gate passed — argmax-exact + gen bit-identical + 2.4–2.9×), and
    /// `VLLM_VULKAN_DSV4_1CB=0`/`false` still FORCES the bit-exact host oracle path
    /// (the OFF override kept for A/B + fallback). Also pins the slot table's
    /// internal consistency (distinct indices < COUNT).
    #[test]
    fn m0_resident_span_defaults_off_and_slots_consistent() {
        // Default-ON when the env var is unset; explicit "0"/"false" forces host.
        if std::env::var("VLLM_VULKAN_DSV4_1CB").is_err() {
            assert!(dsv4_1cb_enabled(), "DSV4 1CB must default ON after the M2b GEN=64 gate");
        }
        // Slot table: every named slot is a distinct index within COUNT.
        let slots = [
            dsv4r::STREAMS, dsv4r::XIN, dsv4r::COLLAPSED, dsv4r::POST, dsv4r::COMB,
            dsv4r::BRANCH_OUT, dsv4r::FFN_HID, dsv4r::FFN_GATE, dsv4r::FFN_UP, dsv4r::VLOG,
        ];
        assert_eq!(slots.len(), dsv4r::COUNT, "slot count matches COUNT");
        let mut seen = slots.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), dsv4r::COUNT, "slots are distinct");
        assert!(slots.iter().all(|&s| s < dsv4r::COUNT), "slots in range");
    }
    use std::collections::HashMap;

    /// In-memory `Dsv4Src` (== `dsv4_forward::tests::DictSrc`), for the offline
    /// CPU-backend equivalence gate.
    struct DictSrc {
        d: HashMap<String, Vec<f32>>,
        i: HashMap<String, Vec<i64>>,
    }
    impl Dsv4Src for DictSrc {
        fn linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32> {
            let key = if self.d.contains_key(name) { name.to_string() } else { format!("{name}.weight") };
            let v = self.d.get(&key).unwrap_or_else(|| panic!("missing linear {name}"));
            assert_eq!(v.len(), out_f * in_f, "linear {name} shape");
            v.clone()
        }
        fn dense(&self, name: &str) -> Vec<f32> {
            self.d.get(name).unwrap_or_else(|| panic!("missing dense {name}")).clone()
        }
        fn expert(&self, name: &str, e: usize, out_f: usize, in_f: usize) -> Vec<f32> {
            let v = self.d.get(name).unwrap_or_else(|| panic!("missing expert {name}"));
            v[e * out_f * in_f..(e + 1) * out_f * in_f].to_vec()
        }
        fn dense_i64(&self, name: &str) -> Vec<i64> {
            self.i.get(name).unwrap_or_else(|| panic!("missing i64 {name}")).clone()
        }
    }

    fn f32v(v: &Value) -> Vec<f32> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
    }

    /// CPU mirror of `dsv4_swiglu_clamp.comp` — the EXACT f32 shader math
    /// (`silu(min(g,limit)) * clamp(u,±limit)`, alpha=1). `limit = INFINITY` → the
    /// shared-expert plain SwiGLU (min/clamp identity). Used to validate the M1
    /// resident-MoE ORCHESTRATION offline (no GPU): substitute this for the GPU
    /// dispatch and the composed MoE must reproduce the transformers golden.
    fn mirror_swiglu_clamp(gate: &[f32], up: &[f32], limit: f32) -> Vec<f32> {
        gate.iter().zip(up.iter()).map(|(&g, &u)| {
            let gv = g.min(limit);
            let uv = u.clamp(-limit, limit);
            let silu = gv / (1.0f32 + (-gv).exp());
            silu * uv
        }).collect()
    }

    /// CPU mirror of `dsv4_hc_residual_mix.comp` — replicates the shader's FLAT
    /// index decode (i → si,k,dd) + f32 accumulation EXACTLY, so a wrong stride in
    /// the GLSL would diverge from the oracle. `out[si,k,dd] = post[si,k]*sub[si,dd]
    /// + Σ_j comb[si,j,k]*streams[si,j,dd]`.
    fn mirror_hc_residual_mix(post: &[f32], sub: &[f32], comb: &[f32], streams: &[f32],
                              seq: usize, hc: usize, hidden: usize) -> Vec<f32> {
        let hcd = hc * hidden;
        let mut out = vec![0f32; seq * hcd];
        for i in 0..seq * hcd {
            let si = i / hcd;
            let rem = i - si * hcd;
            let k = rem / hidden;
            let dd = rem - k * hidden;
            let mut acc = post[si * hc + k] * sub[si * hidden + dd];
            for j in 0..hc {
                acc += comb[si * hc * hc + j * hc + k] * streams[si * hcd + j * hidden + dd];
            }
            out[i] = acc;
        }
        out
    }

    /// ★ M2a GATE (offline, no GPU): the `dsv4_hc_residual_mix` shader's CPU mirror
    /// (flat-index decode + f32 accum) reproduces the host oracle
    /// `dsv4::hc_residual_mix` on deterministic synthetic streams. Validates the
    /// kernel's index math + formula before the on-node span gate. cos≈1 (mirror
    /// f32 vs oracle f64 accumulation; the Sinkhorn combine coeffs are O(1)).
    #[test]
    fn dsv4_hc_residual_mix_mirror_matches_oracle() {
        let (seq, hc, hidden) = (3usize, 4usize, 16usize);
        // Deterministic pseudo-random inputs.
        let g = |a: usize, b: usize| ((((a * 2654435761) ^ (b * 40503)) % 1000) as f32 / 500.0) - 1.0;
        let post: Vec<f32> = (0..seq * hc).map(|i| g(i, 1)).collect();
        let sub: Vec<f32> = (0..seq * hidden).map(|i| g(i, 2)).collect();
        let comb: Vec<f32> = (0..seq * hc * hc).map(|i| g(i, 3) * 0.25).collect();
        let streams: Vec<f32> = (0..seq * hc * hidden).map(|i| g(i, 4)).collect();

        let oracle = crate::dsv4::hc_residual_mix(&post, &sub, &comb, &streams, seq, hc, hidden);
        let mirror = mirror_hc_residual_mix(&post, &sub, &comb, &streams, seq, hc, hidden);
        assert_eq!(oracle.len(), mirror.len(), "hc_residual_mix len");
        let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
        for (a, b) in mirror.iter().zip(oracle.iter()) {
            dot += *a as f64 * *b as f64; na += *a as f64 * *a as f64; nb += *b as f64 * *b as f64;
            mx = mx.max((*a as f64 - *b as f64).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("M2a hc_residual_mix mirror: cos={cos:.9} max_abs={mx:.3e}");
        assert!(cos > 0.9999999, "hc_residual_mix mirror cos {cos} too low");
    }

    /// ★ M1 GATE (offline, no GPU): drive the resident-MoE dataflow with the CPU
    /// kernel-mirrors (== the shader math: f32 matvec + `dsv4_swiglu_clamp` +
    /// routed-weighted accum + ungated shared) on the real `moe_block.json` fixture
    /// (transformers-5.8.1 cross-checked golden `moe_out`), and assert the composed
    /// output reproduces the golden. This validates the marshalling `moe_resident`
    /// wrote — the gate_up split, the swiglu_limit clamp placement (routed clamps,
    /// shared does NOT), the down projection, and the `q35_moe_accum_batched`
    /// routed-weight/shared fold — everything on the resident path except the
    /// kernels themselves (separately cos=1.0 on-node). f32 SwiGLU vs the golden's
    /// f64 → cos≈1 (not bit-exact); the assembled-decode argmax gate is on-node.
    #[test]
    fn moe_resident_mirror_matches_oracle() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/moe_block.json");
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => { eprintln!("SKIP: moe_block.json fixture absent"); return; }
        };
        let j: Value = serde_json::from_str(&raw).unwrap();
        let t = j["T"].as_u64().unwrap() as usize;
        let hh = j["H"].as_u64().unwrap() as usize;
        let ii = j["I"].as_u64().unwrap() as usize;
        let tk = j["top_k"].as_u64().unwrap() as usize;
        let limit = j["limit"].as_f64().unwrap() as f32;
        let hs = f32v(&j["hs"]);
        let gate_up = f32v(&j["gate_up"]); // [E, 2I, H]
        let down = f32v(&j["down"]);       // [E, H, I]
        let sh_gate = f32v(&j["shared_gate"]); // [I, H]
        let sh_up = f32v(&j["shared_up"]);
        let sh_down = f32v(&j["shared_down"]); // [H, I]
        let idx: Vec<usize> = j["idx_topk"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let w = f32v(&j["w_topk"]);
        let golden = f32v(&j["moe_out"]); // [T, H] — routed + shared, transformers-checked
        let mm = crate::model::cpu_matmul;
        let per_e = 2 * ii * hh;

        // Mirror == moe_resident's op-DAG in host arithmetic.
        let mut mir = vec![0f32; t * hh];
        for ti in 0..t {
            let xrow = &hs[ti * hh..(ti + 1) * hh];
            let acc = &mut mir[ti * hh..(ti + 1) * hh];
            for jj in 0..tk {
                let e = idx[ti * tk + jj];
                let wj = w[ti * tk + jj];
                let gate = mm(xrow, &gate_up[e * per_e..e * per_e + ii * hh], 1, hh, ii);
                let up = mm(xrow, &gate_up[e * per_e + ii * hh..e * per_e + 2 * ii * hh], 1, hh, ii);
                let hid = mirror_swiglu_clamp(&gate, &up, limit);
                let contrib = mm(&hid, &down[e * hh * ii..(e + 1) * hh * ii], 1, ii, hh);
                for c in 0..hh { acc[c] += wj * contrib[c]; }
            }
            // shared expert (no clamp → limit = +inf).
            let sg = mm(xrow, &sh_gate, 1, hh, ii);
            let su = mm(xrow, &sh_up, 1, hh, ii);
            let sh = mirror_swiglu_clamp(&sg, &su, f32::INFINITY);
            let sc = mm(&sh, &sh_down, 1, ii, hh);
            for c in 0..hh { acc[c] += sc[c]; }
        }

        assert_eq!(mir.len(), golden.len(), "moe_out len");
        let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
        for (a, b) in mir.iter().zip(golden.iter()) {
            dot += *a as f64 * *b as f64; na += *a as f64 * *a as f64; nb += *b as f64 * *b as f64;
            mx = mx.max((*a as f64 - *b as f64).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("M1 resident-MoE mirror: moe_out cos={cos:.9} max_abs={mx:.3e}");
        assert!(cos > 0.999999, "resident-MoE mirror cos {cos} too low vs golden");
    }

    /// CPU mirror of `dsv4_mla_softmax.comp` — the EXACT online (flash-style) f32
    /// softmax the kernel runs, per head: sink as the initial virtual key, sliding
    /// causal mask (`qpos-ti < sw`), compressed-KV `block_bias` (-inf hidden), then
    /// the interleaved output-rope conjugate (-sin). Returns `ao [nh*hd]`. Substitute
    /// this for the GPU dispatch to validate the M2b resident-attention ORCHESTRATION
    /// offline (the composed tail must reproduce the f64 oracle `attn_tail_host_proj`).
    #[allow(clippy::too_many_arguments)]
    fn mirror_mla_softmax(
        q: &[f32], kv_sliding: &[f32], compressed_kv: &[f32], block_bias: &[f32],
        sinks: &[f32], cos: &[f32], sin: &[f32], nh: usize, hd: usize, sw: usize, t1: usize, rope_dim: usize,
    ) -> Vec<f32> {
        let t_comp = compressed_kv.len() / hd;
        let total = t1 + t_comp;
        let qpos = t1 - 1;
        let scaling = (hd as f64).powf(-0.5) as f32;
        let mut ao = vec![0f32; nh * hd];
        for hh in 0..nh {
            let mut run_max = sinks[hh];
            let mut run_den = 1.0f32; // exp(sink - sink)
            let mut acc = vec![0f32; hd];
            for ti in 0..total {
                let visible = if ti < t1 {
                    (qpos - ti) < sw
                } else {
                    block_bias[ti - t1] > -1.0e30
                };
                if !visible { continue; }
                let kbase = if ti < t1 { ti * hd } else { (ti - t1) * hd };
                let ksrc = if ti < t1 { kv_sliding } else { compressed_kv };
                let mut dot = 0f32;
                for d in 0..hd { dot += q[hh * hd + d] * ksrc[kbase + d]; }
                let l = dot * scaling; // visible additive mask is exactly 0
                let new_max = run_max.max(l);
                let corr = (run_max - new_max).exp();
                let w = (l - new_max).exp();
                run_den = run_den * corr + w;
                for d in 0..hd { acc[d] = acc[d] * corr + w * ksrc[kbase + d]; }
                run_max = new_max;
            }
            let invden = 1.0 / run_den;
            for d in 0..hd { ao[hh * hd + d] = acc[d] * invden; }
        }
        // output-rope conjugate (-sin), interleaved pairs (nope+2i, nope+2i+1).
        let nhalf = rope_dim / 2;
        let nope = hd - rope_dim;
        for hh in 0..nh {
            for i in 0..nhalf {
                let c = cos[i];
                let sn = -sin[i];
                let a = ao[hh * hd + nope + 2 * i];
                let b = ao[hh * hd + nope + 2 * i + 1];
                ao[hh * hd + nope + 2 * i] = a * c - b * sn;
                ao[hh * hd + nope + 2 * i + 1] = b * c + a * sn;
            }
        }
        ao
    }

    /// ★ M2b GATE (offline, no GPU): the resident MLA attention TAIL orchestration
    /// (`dsv4_mla_softmax` mirror → grouped block-diagonal `wo_a` matvecs) reproduces
    /// the f64 host oracle `attn_tail_host_proj` on deterministic synthetic MLA
    /// inputs — sink + sliding-window + compressed-KV `block_bias` + output-rope +
    /// the g-way group decomposition. cos≈1 (kernel online-f32 vs oracle two-pass
    /// f64); the assembled-decode argmax gate is on-node authoritative.
    #[test]
    fn dsv4_mla_softmax_tail_mirror_matches_oracle() {
        let (nh, hd, t1, t_comp, sw, g, olr) = (4usize, 64usize, 6usize, 2usize, 4usize, 2usize, 8usize);
        let rope_dim = 16usize;
        let per_g = (nh * hd) / g;
        let gg = |a: usize, b: usize| ((((a * 2654435761) ^ (b * 40503)) % 2000) as f32 / 1000.0) - 1.0;
        let q: Vec<f32> = (0..nh * hd).map(|i| gg(i, 1) * 0.3).collect();
        let kv_sliding: Vec<f32> = (0..t1 * hd).map(|i| gg(i, 2) * 0.3).collect();
        let compressed_kv: Vec<f32> = (0..t_comp * hd).map(|i| gg(i, 3) * 0.3).collect();
        // block_bias: window 0 visible (0.0), window 1 hidden (-inf).
        let block_bias: Vec<f32> = vec![0.0, f32::NEG_INFINITY];
        let sinks: Vec<f32> = (0..nh).map(|i| gg(i, 4) * 0.5).collect();
        let cos: Vec<f32> = (0..rope_dim / 2).map(|i| (0.11 * i as f32).cos()).collect();
        let sin: Vec<f32> = (0..rope_dim / 2).map(|i| (0.11 * i as f32).sin()).collect();
        let w_o_a: Vec<f32> = (0..g * olr * per_g).map(|i| gg(i, 5) * 0.1).collect();

        // Oracle: the f64 host tail.
        let oracle = attn_tail_host_proj(
            &q, &kv_sliding, &compressed_kv, &block_bias, &sinks, &cos, &sin,
            nh, hd, g, olr, sw, t1, rope_dim, &w_o_a,
        );
        // Mirror: GPU online-f32 softmax → grouped wo_a (g per-group f32 matvecs).
        let ao = mirror_mla_softmax(&q, &kv_sliding, &compressed_kv, &block_bias, &sinks, &cos, &sin, nh, hd, sw, t1, rope_dim);
        let mm = crate::model::cpu_matmul;
        let mut mir = vec![0f32; g * olr];
        for grp in 0..g {
            let xin = &ao[grp * per_g..(grp + 1) * per_g];
            let wblk = &w_o_a[grp * olr * per_g..(grp + 1) * olr * per_g];
            let out = mm(xin, wblk, 1, per_g, olr);
            mir[grp * olr..(grp + 1) * olr].copy_from_slice(&out);
        }
        assert_eq!(mir.len(), oracle.len(), "proj len");
        let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
        for (a, b) in mir.iter().zip(oracle.iter()) {
            dot += *a as f64 * *b as f64; na += *a as f64 * *a as f64; nb += *b as f64 * *b as f64;
            mx = mx.max((*a as f64 - *b as f64).abs());
        }
        let cos_sim = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("M2b MLA-tail mirror: proj cos={cos_sim:.9} max_abs={mx:.3e}");
        assert!(cos_sim > 0.999999, "MLA-tail mirror cos {cos_sim} too low");
    }

    /// ★ M2b GATE (offline, no GPU): the grouped block-diagonal `wo_a` DECOMPOSITION
    /// — `g` independent per-group matvecs over the resident output-row-sliced
    /// weights (`dsv4r_rec_wo_a_grouped` orchestration) — is ARITHMETICALLY IDENTICAL
    /// to the monolithic block-diagonal loop in `attn_tail_host_proj`. Pins the
    /// row-slice + group-stride marshalling (the on-node matvec kernel is already
    /// cos=1.0), independent of the softmax path.
    #[test]
    fn dsv4_wo_a_grouped_decomposition_is_exact() {
        let (nh, hd, g, olr) = (4usize, 64usize, 2usize, 8usize);
        let per_g = (nh * hd) / g;
        let gg = |a: usize, b: usize| ((((a * 2654435761) ^ (b * 40503)) % 2000) as f32 / 1000.0) - 1.0;
        let ao: Vec<f32> = (0..nh * hd).map(|i| gg(i, 7)).collect();
        let w_o_a: Vec<f32> = (0..g * olr * per_g).map(|i| gg(i, 8) * 0.1).collect();
        // Monolithic block-diagonal via cpu_matmul-per-row (the oracle formula).
        let mm = crate::model::cpu_matmul;
        let mut mono = vec![0f32; g * olr];
        for grp in 0..g {
            let xin = &ao[grp * per_g..grp * per_g + per_g];
            for o in 0..olr {
                let wrow = &w_o_a[(grp * olr + o) * per_g..(grp * olr + o) * per_g + per_g];
                mono[grp * olr + o] = mm(xin, wrow, 1, per_g, 1)[0];
            }
        }
        // Group decomposition via g cpu_matmuls over the row-sliced weight blocks —
        // the exact `dsv4r_rec_wo_a_grouped` orchestration (g independent matvecs).
        let mut dec = vec![0f32; g * olr];
        for grp in 0..g {
            let xin = &ao[grp * per_g..(grp + 1) * per_g];
            let wblk = &w_o_a[grp * olr * per_g..(grp + 1) * olr * per_g];
            let out = mm(xin, wblk, 1, per_g, olr);
            dec[grp * olr..(grp + 1) * olr].copy_from_slice(&out);
        }
        // Same primitive (cpu_matmul) row-for-row → bit-identical group decomposition.
        assert_eq!(mono, dec, "grouped wo_a decomposition must be bit-identical to the monolithic per-row matvec");
    }

    /// ★ resident-Sinkhorn GATE (offline, no GPU): a full HC SITE run resident —
    /// the `dsv4_hyper_connection` kernel math (`hc_stable_mirror`: max-factored
    /// RMSNorm + two-float `flat@fn` mix + 20-iter Sinkhorn + collapse) feeding the
    /// `dsv4_hc_residual_mix` kernel math (`mirror_hc_residual_mix`) — reproduces the
    /// host oracle chain (`dsv4::hc_block` → `dsv4::hc_residual_mix`) on deterministic
    /// streams. Validates wiring BOTH resident HC halves op→op (Sinkhorn+projection,
    /// then residual mix) with no host round-trip. cos≈1.
    #[test]
    fn dsv4_hc_site_resident_mirror_matches_oracle() {
        let (seq, hc, hidden, iters) = (1usize, 4usize, 32usize, 20usize);
        let (hc_eps, rms_eps) = (1e-6f32, 1e-6f32);
        let hcd = hc * hidden;
        let mix = (2 + hc) * hc;
        let g = |a: usize, b: usize| ((((a * 2654435761) ^ (b * 40503)) % 2000) as f32 / 1000.0) - 1.0;
        let streams: Vec<f32> = (0..seq * hcd).map(|i| g(i, 1) * 2.0).collect();
        let fn_w: Vec<f32> = (0..mix * hcd).map(|i| g(i, 2) * 0.05).collect();
        let base: Vec<f32> = (0..mix).map(|i| g(i, 3) * 0.1).collect();
        let scale: Vec<f32> = vec![1.0, 1.0, 1.0];
        let sublayer: Vec<f32> = (0..seq * hidden).map(|i| g(i, 4)).collect();

        // Host oracle chain.
        let (post_o, comb_o, coll_o) = crate::dsv4::hc_block(&streams, seq, hc, hidden, &fn_w, &base, &scale, iters, hc_eps, rms_eps);
        let mixed_o = crate::dsv4::hc_residual_mix(&post_o, &sublayer, &comb_o, &streams, seq, hc, hidden);

        // Resident mirror chain: hc kernel math → residual-mix kernel math.
        let (post_m, comb_m, _coll_m) = hc_stable_mirror(&streams, seq, hc, hidden, &fn_w, &base, &scale, iters, hc_eps, rms_eps);
        let mixed_m = mirror_hc_residual_mix(&post_m, &sublayer, &comb_m, &streams, seq, hc, hidden);

        assert_eq!(mixed_m.len(), mixed_o.len(), "hc site mixed len");
        let cosf = |a: &[f32], b: &[f32]| -> (f64, f64) {
            let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
            for (x, y) in a.iter().zip(b.iter()) {
                dot += *x as f64 * *y as f64; na += *x as f64 * *x as f64; nb += *y as f64 * *y as f64;
                mx = mx.max((*x as f64 - *y as f64).abs());
            }
            (dot / (na.sqrt() * nb.sqrt() + 1e-30), mx)
        };
        let (cpost, _) = cosf(&post_m, &post_o);
        let (ccomb, _) = cosf(&comb_m, &comb_o);
        let (cmix, mxmix) = cosf(&mixed_m, &mixed_o);
        eprintln!("resident-Sinkhorn HC site mirror: post cos={cpost:.9} comb cos={ccomb:.9} mixed cos={cmix:.9} max_abs={mxmix:.3e}");
        assert!(cpost > 0.9999 && ccomb > 0.999 && cmix > 0.9999, "HC-site resident mirror cos too low");
    }

    // ── CPU mirrors of the three DSA kernels (the EXACT shader math, == the mirrors
    //    in debug_dsv4_dsa_*). Used ONLY to validate the `gpu_csa_compressor`
    //    ORCHESTRATION offline (no GPU): substitute these for the GPU dispatches and
    //    the composed compressor must reproduce the host `csa_compressor` oracle. On
    //    node the same orchestration calls the real kernels (cos=1.0, debug gates).
    fn mirror_compress(kv: &[f32], gate: &[f32], pbias: &[f32], knorm: &[f32], ifreq: &[f32],
                       s: usize, m: usize, hd: usize, eps: f32) -> Vec<f32> {
        let twohd = 2 * hd;
        let n_win = s / m;
        let rope_dim = 2 * ifreq.len();
        let nope = hd - rope_dim;
        let neg = -3.4e38f32;
        let mut out = vec![0f32; n_win * hd];
        for win in 0..n_win {
            let mut pooled = vec![0f32; hd];
            for d in 0..hd {
                let mut g = vec![neg; 2 * m];
                let mut v = vec![0f32; 2 * m];
                let mut gmax = neg;
                for t in 0..m {
                    if win >= 1 {
                        let tok = (win - 1) * m + t;
                        g[t] = gate[tok * twohd + d] + pbias[t * twohd + d];
                        v[t] = kv[tok * twohd + d];
                    }
                    let tokc = win * m + t;
                    g[m + t] = gate[tokc * twohd + hd + d] + pbias[t * twohd + hd + d];
                    v[m + t] = kv[tokc * twohd + hd + d];
                    gmax = gmax.max(g[t]).max(g[m + t]);
                }
                let (mut denom, mut acc) = (0f32, 0f32);
                for k in 0..2 * m { let e = (g[k] - gmax).exp(); denom += e; acc += e * v[k]; }
                pooled[d] = acc / denom;
            }
            let ss: f32 = pooled.iter().map(|&x| x * x).sum();
            let inv = 1.0 / (ss / hd as f32 + eps).sqrt();
            let normed: Vec<f32> = (0..hd).map(|d| pooled[d] * inv * knorm[d]).collect();
            let pos = (win * m) as f32;
            for d in 0..hd {
                if d < nope { out[win * hd + d] = normed[d]; }
                else {
                    let r = d - nope; let j = r / 2;
                    let ang = pos * ifreq[j];
                    let (c, sn) = (ang.cos(), ang.sin());
                    let pb = nope + j * 2;
                    let (x1, x2) = (normed[pb], normed[pb + 1]);
                    out[win * hd + d] = if r % 2 == 0 { x1 * c - x2 * sn } else { x2 * c + x1 * sn };
                }
            }
        }
        out
    }
    fn mirror_index_score(q: &[f32], ckv: &[f32], wgt: &[f32], s: usize, nh: usize, hd: usize,
                          n_win: usize, softmax_scale: f32) -> Vec<f32> {
        // f64 accumulation == what the two-float (double-single) `dsv4_dsa_index_score`
        // kernel now computes (ITEM B): the hd-dot and head-sum carry ~fp64 precision so
        // the discrete top-512 argsort matches the fp64 `indexer_topk` oracle across an
        // assembled multi-step decode (fp32 rounding here flipped near-tied windows).
        let mut out = vec![0f32; s * n_win];
        for si in 0..s {
            for w in 0..n_win {
                let mut acc = 0f64;
                for head in 0..nh {
                    let qb = (si * nh + head) * hd;
                    let cb = w * hd;
                    let mut dot = 0f64;
                    for dd in 0..hd { dot += q[qb + dd] as f64 * ckv[cb + dd] as f64; }
                    acc += wgt[si * nh + head] as f64 * dot.max(0.0);
                }
                out[si * n_win + w] = (acc * softmax_scale as f64) as f32;
            }
        }
        out
    }
    fn mirror_topk(scores: &[f32], s: usize, n_win: usize, top_k: usize, m: usize, pos0: usize) -> Vec<i32> {
        // Strict-rank scatter with the index tie-break — the exact `dsv4_dsa_topk`
        // logic (set-identical to argsort on continuous scores).
        let mut out = vec![-1i32; s * top_k];
        for si in 0..s {
            let thr = (pos0 + si + 1) / m;
            let vthr = thr.min(n_win);
            let k = top_k.min(n_win);
            for w in 0..vthr {
                let sw = scores[si * n_win + w];
                let mut rank = 0usize;
                for w2 in 0..vthr {
                    if w2 == w { continue; }
                    let s2 = scores[si * n_win + w2];
                    if s2 > sw || (s2 == sw && w2 < w) { rank += 1; if rank >= k { break; } }
                }
                if rank < k { out[si * top_k + rank] = w as i32; }
            }
        }
        out
    }

    /// CPU mirror of the `dsv4_hyper_connection` kernel math (== `hc_stable_cpu` in
    /// debug_api): sanitize + max-factored RMSNorm, f64 mix dot (mirrors the shader's
    /// two-float carry), fp32 pre/post/comb + exact linear Sinkhorn. The clamp is the
    /// validated 1e4 no-op. This is what the GPU kernel computes; the oracle
    /// `dsv4::hc_block` does the same in full f64 (cos=1.0, not bit-identical).
    #[allow(clippy::too_many_arguments)]
    fn hc_stable_mirror(streams: &[f32], seq: usize, hc: usize, h: usize, fn_w: &[f32],
                        base: &[f32], scale: &[f32], iters: usize, hc_eps: f32, rms_eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let san = |v: f32| if v.is_finite() { v } else { 0.0f32 };
        let sig = |x: f32| 1.0f32 / (1.0 + (-x).exp());
        let cl = |x: f32| x.clamp(-1e4, 1e4);
        let (d, hd, mix) = (h, hc * h, (2 + hc) * hc);
        let (mut post, mut comb, mut coll) = (vec![0f32; seq * hc], vec![0f32; seq * hc * hc], vec![0f32; seq * d]);
        for t in 0..seq {
            let sb = t * hd;
            let mm = (0..hd).fold(0f32, |a, i| a.max(san(streams[sb + i]).abs()));
            let mm = if mm > 0.0 { mm } else { 1.0 };
            let invm = 1.0 / mm;
            let ms = (0..hd).fold(0f32, |a, i| { let xs = san(streams[sb + i]) * invm; a + xs * xs }) / hd as f32;
            let inv = 1.0 / (ms + rms_eps * invm * invm).sqrt();
            let mixed: Vec<f32> = (0..mix).map(|o| {
                let fr = o * hd;
                let dot = (0..hd).fold(0f64, |a, i| a + ((san(streams[sb + i]) * invm) as f64) * (fn_w[fr + i] as f64));
                inv * (dot as f32)
            }).collect();
            let mut pre = vec![0f32; hc];
            for hh in 0..hc {
                pre[hh] = sig(cl(mixed[hh] * scale[0] + base[hh])) + hc_eps;
                post[t * hc + hh] = 2.0 * sig(cl(mixed[hc + hh] * scale[1] + base[hc + hh]));
            }
            let mut cm = vec![0f32; hc * hc];
            for i in 0..hc {
                let mut mx = f32::NEG_INFINITY;
                for jj in 0..hc { let lg = cl(mixed[2 * hc + i * hc + jj] * scale[2] + base[2 * hc + i * hc + jj]); cm[i * hc + jj] = lg; mx = mx.max(lg); }
                let mut sm = 0f32;
                for jj in 0..hc { cm[i * hc + jj] = (cm[i * hc + jj] - mx).exp(); sm += cm[i * hc + jj]; }
                for jj in 0..hc { cm[i * hc + jj] = cm[i * hc + jj] / sm + hc_eps; }
            }
            for jj in 0..hc { let mut cs = 0f32; for i in 0..hc { cs += cm[i * hc + jj]; } cs += hc_eps; for i in 0..hc { cm[i * hc + jj] /= cs; } }
            for _ in 1..iters {
                for i in 0..hc { let mut rs = 0f32; for jj in 0..hc { rs += cm[i * hc + jj]; } rs += hc_eps; for jj in 0..hc { cm[i * hc + jj] /= rs; } }
                for jj in 0..hc { let mut cs = 0f32; for i in 0..hc { cs += cm[i * hc + jj]; } cs += hc_eps; for i in 0..hc { cm[i * hc + jj] /= cs; } }
            }
            for i in 0..hc * hc { comb[t * hc * hc + i] = cm[i]; }
            for dd in 0..d {
                let mut acc = 0f32;
                for hh in 0..hc { acc += pre[hh] * san(streams[sb + hh * d + dd]); }
                coll[t * d + dd] = acc;
            }
        }
        (post, comb, coll)
    }

    /// Backend that mirrors `CpuMv` but overrides `hc_block` with the GPU kernel's
    /// stable-HC math — so `forward_mv(HcMirrorMv)` is `forward_mv(CpuMv)` with the
    /// mHC swapped for exactly what `Dsv4GpuStage::hc_block` runs on the device.
    struct HcMirrorMv<'a, S: Dsv4Src> { src: &'a S }
    impl<'a, S: Dsv4Src> Mv for HcMirrorMv<'a, S> {
        fn mm(&mut self, name: &str, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
            crate::model::cpu_matmul(x, &self.src.linear(name, out_f, in_f), s, in_f, out_f)
        }
        fn mm_expert(&mut self, name: &str, e: usize, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
            crate::model::cpu_matmul(x, &self.src.expert(name, e, out_f, in_f), s, in_f, out_f)
        }
        fn dq_linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32> { self.src.linear(name, out_f, in_f) }
        fn dense(&self, name: &str) -> Vec<f32> { self.src.dense(name) }
        fn dense_i64(&self, name: &str) -> Vec<i64> { self.src.dense_i64(name) }
        fn embed_rows(&self, ids: &[u32], vocab: usize, h: usize) -> Vec<f32> { self.src.embed_rows(ids, vocab, h) }
        fn hc_block(&mut self, streams: &[f32], seq: usize, hc: usize, h: usize, fn_w: &[f32],
                    base: &[f32], scale: &[f32], iters: usize, hc_eps: f32, rms_eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            hc_stable_mirror(streams, seq, hc, h, fn_w, base, scale, iters, hc_eps, rms_eps)
        }
    }

    /// ★ OFFLINE HC-GPU MATH-SWAP GATE (no GPU): `forward_mv` with the mHC replaced by
    /// the GPU kernel's stable-HC math must reproduce the CPU-oracle argmax (and stay
    /// logit-cos≈1) on the tiny fixture. Proves that moving mHC onto the device does
    /// not perturb the composed op-DAG's decision — the kernel itself is separately
    /// cos=1.0 (`debug_dsv4_hc`).
    #[test]
    fn hc_gpu_math_swap_preserves_argmax() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/selftest.json");
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => { eprintln!("SKIP: selftest.json fixture absent"); return; }
        };
        let j: Value = serde_json::from_str(&raw).unwrap();
        let cfg = Dsv4Config::from_json(&j["config"]).unwrap();
        let input_ids: Vec<u32> = j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut d = HashMap::new();
        let mut i = HashMap::new();
        for (k, v) in j["weights"].as_object().unwrap() { d.insert(k.clone(), f32v(v)); }
        for (k, v) in j["weights_i64"].as_object().unwrap() {
            i.insert(k.clone(), v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect());
        }
        let src = DictSrc { d, i };
        let vocab = cfg.vocab_size;
        let n = input_ids.len();
        let mut oracle = CpuMv { src: &src };
        let ref_logits = forward_mv(&cfg, &input_ids, &mut oracle);
        let mut mirror = HcMirrorMv { src: &src };
        let mir_logits = forward_mv(&cfg, &input_ids, &mut mirror);
        let (ra, _) = argmax_last(&ref_logits, n, vocab);
        let (ma, _) = argmax_last(&mir_logits, n, vocab);
        let last_ref = &ref_logits[(n - 1) * vocab..];
        let last_mir = &mir_logits[(n - 1) * vocab..];
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, b) in last_mir.iter().zip(last_ref.iter()) { dot += *a as f64 * *b as f64; na += *a as f64 * *a as f64; nb += *b as f64 * *b as f64; }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("HC-GPU math-swap: argmax mirror={ma} oracle={ra}; last-logit cos={cos:.9}");
        assert_eq!(ma, ra, "HC math-swap changed the argmax");
        assert!(cos > 0.9999, "HC math-swap logit cos {cos} too low");
    }

    /// ★ OFFLINE DSA-GPU ORCHESTRATION GATE (no GPU): drive the `gpu_csa_compressor`
    /// dataflow with the three CPU kernel-mirrors (== the shader math) on the REAL
    /// `dsa_csa.json` reference fixture, and assert the composed output reproduces the
    /// host `csa_compressor` oracle: compressed_kv cos≈1 + block_bias visibility
    /// SET-MATCH. This validates the marshalling I wrote (matvec layout, w_scale fold,
    /// softmax_scale placement, causal threshold/pos0, vis assembly with the top_k
    /// stride) — everything on the DSA-GPU path except the kernels themselves, which
    /// are separately cos=1.0 (`debug_dsv4_dsa_{compress,score,topk}`, on-node).
    #[test]
    fn dsa_gpu_orchestration_mirror_matches_oracle() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/dsa_csa.json");
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => { eprintln!("SKIP: dsa_csa.json fixture absent"); return; }
        };
        let j: Value = serde_json::from_str(&raw).unwrap();
        let s = j["seq_len"].as_u64().unwrap() as usize;
        let h = j["hidden_size"].as_u64().unwrap() as usize;
        let hd = j["head_dim"].as_u64().unwrap() as usize;
        let m = j["compress_rate"].as_u64().unwrap() as usize;
        let eps = j["rms_norm_eps"].as_f64().unwrap() as f32;
        let positions: Vec<usize> = j["positions"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let inv_freq = f32v(&j["compress_inv_freq"]);
        let scaling = j["compress_scaling"].as_f64().unwrap() as f32;
        let q_lora = f32v(&j["q_residual"]).len() / s;
        let ix_nh = j["index_n_heads"].as_u64().unwrap() as usize;
        let ix_hd = j["index_head_dim"].as_u64().unwrap() as usize;
        let topk = j["index_topk"].as_u64().unwrap() as usize;
        let hs = f32v(&j["hs"]);
        let q_res = f32v(&j["q_residual"]);
        let w = &j["weights"];
        let iw = &j["indexer"];
        let (kvp, gp, pb, kn) = (f32v(&w["kv_proj"]), f32v(&w["gate_proj"]), f32v(&w["position_bias"]), f32v(&w["kv_norm"]));
        let (ikvp, igp, ipb, ikn, iqb, iwp) = (
            f32v(&iw["kv_proj"]), f32v(&iw["gate_proj"]), f32v(&iw["position_bias"]),
            f32v(&iw["kv_norm"]), f32v(&iw["q_b_proj"]), f32v(&iw["weights_proj"]),
        );
        let ix = IndexerWeights { kv_proj: &ikvp, gate_proj: &igp, position_bias: &ipb, kv_norm: &ikn, q_b_proj: &iqb, weights_proj: &iwp };

        // Host oracle.
        let (ckv_ref, vis_ref) = crate::dsv4_dsa::csa_compressor(
            &hs, &q_res, s, h, hd, m, &positions, &kvp, &gp, &pb, &kn, eps, &inv_freq, scaling,
            q_lora, ix_nh, ix_hd, topk, &ix,
        );

        // Mirror orchestration == gpu_csa_compressor with CPU kernel-mirrors.
        let n_win = s / m;
        let cpu_mm = crate::model::cpu_matmul;
        let kv = cpu_mm(&hs, &kvp, s, h, 2 * hd);
        let gate = cpu_mm(&hs, &gp, s, h, 2 * hd);
        let compressed = mirror_compress(&kv, &gate, &pb, &kn, &inv_freq, s, m, hd, eps);
        let ix_kv = cpu_mm(&hs, &ikvp, s, h, 2 * ix_hd);
        let ix_gate = cpu_mm(&hs, &igp, s, h, 2 * ix_hd);
        let ck = mirror_compress(&ix_kv, &ix_gate, &ipb, &ikn, &inv_freq, s, m, ix_hd, eps);
        let mut q = cpu_mm(&q_res, &iqb, s, q_lora, ix_nh * ix_hd);
        let rope_dim = 2 * inv_freq.len();
        let (cos_q, sin_q) = rope_cos_sin(&positions, &inv_freq, scaling);
        apply_interleaved_rope_inplace(&mut q, s * ix_nh, ix_hd, rope_dim, &|r| r / ix_nh, &cos_q, &sin_q);
        let wgt0 = cpu_mm(&hs, &iwp, s, h, ix_nh);
        let w_scale = (ix_nh as f64).powf(-0.5) as f32;
        let wgt: Vec<f32> = wgt0.iter().map(|v| v * w_scale).collect();
        let softmax_scale = (ix_hd as f64).powf(-0.5) as f32;
        let scores = mirror_index_score(&q, &ck, &wgt, s, ix_nh, ix_hd, n_win, softmax_scale);
        let pos0 = positions.first().copied().unwrap_or(0);
        let sel = mirror_topk(&scores, s, n_win, topk, m, pos0);
        let mut vis = vec![0i32; s * n_win];
        for si in 0..s {
            for kk in 0..topk {
                let idx = sel[si * topk + kk];
                if idx >= 0 { vis[si * n_win + idx as usize] = 1; }
            }
        }

        // compressed_kv cos≈1 vs oracle.
        assert_eq!(compressed.len(), ckv_ref.len(), "compressed_kv len");
        let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f64);
        for (a, b) in compressed.iter().zip(ckv_ref.iter()) {
            dot += *a as f64 * *b as f64; na += *a as f64 * *a as f64; nb += *b as f64 * *b as f64;
            mx = mx.max((*a as f64 - *b as f64).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("DSA-GPU orchestration: compressed_kv cos={cos:.9} max_abs={mx:.3e}");
        assert!(cos > 0.999999, "compressed_kv cos {cos} too low");
        // block_bias visibility SET-MATCH vs oracle (per query row).
        assert_eq!(vis.len(), vis_ref.len(), "vis len");
        let mut agree_rows = 0usize;
        for si in 0..s {
            let a: std::collections::HashSet<usize> = (0..n_win).filter(|&w| vis[si * n_win + w] != 0).collect();
            let b: std::collections::HashSet<usize> = (0..n_win).filter(|&w| vis_ref[si * n_win + w] != 0).collect();
            if a == b { agree_rows += 1; }
        }
        eprintln!("DSA-GPU orchestration: vis set-match {agree_rows}/{s} rows");
        assert_eq!(agree_rows, s, "block_bias visibility set-match failed");
    }

    /// ★ ITEM B MECHANISM (offline, deterministic): the DSA index-score reduction
    /// precision is a *selection*-changing decision, not just a cosine. A single
    /// window whose true (fp64) dot is a large-magnitude cancellation
    /// (`+1e8 + 1 + 1 - 1e8 == 2`) collapses to `0` under fp32 left-to-right
    /// accumulation. With a rival plain-dot window at `1.0`, the fp32 top-1 pick
    /// FLIPS (picks the rival) while the fp64 path (== what the two-float
    /// `dsv4_dsa_index_score` kernel now computes, and == the fp64 `indexer_topk`
    /// oracle) picks the true winner. This is exactly the mHC-style low-order-bit
    /// recovery that keeps the assembled multi-step decode's block_bias — hence its
    /// argmax sequence — bit-locked to the oracle. `mirror_index_score` is now fp64.
    #[test]
    fn dsa_index_score_precision_flips_topk() {
        // s=1 query, nh=1 head, hd=4, n_win=2 windows, m=1, top_k=1.
        let (s, nh, hd, n_win, m, top_k) = (1usize, 1usize, 4usize, 2usize, 1usize, 1usize);
        // q for the single head (shared across windows): the cancellation pattern.
        let q = vec![1.0e8f32, 1.0, 1.0, -1.0e8];
        // window 0 (ckv[0..4]) = all ones -> true dot = 1e8+1+1-1e8 = 2.0 (fp32 -> 0.0).
        // window 1 (ckv[4..8]) zeros the ±1e8 channels -> plain dot 0.5+0.5 = 1.0 in
        // BOTH fp32 and fp64 (no cancellation, so the rival is precision-invariant).
        let ckv = vec![1.0f32, 1.0, 1.0, 1.0, /* w1 */ 0.0, 0.5, 0.5, 0.0];
        let wgt = vec![1.0f32]; // [s*nh]
        let softmax_scale = 1.0f32;

        // fp64 path (the new kernel / oracle).
        let scores64 = mirror_index_score(&q, &ckv, &wgt, s, nh, hd, n_win, softmax_scale);
        // reference fp32 left-to-right accumulator (the OLD kernel math).
        let mut scores32 = vec![0f32; n_win];
        for w in 0..n_win {
            let mut dot = 0f32;
            for dd in 0..hd { dot += q[dd] * ckv[w * hd + dd]; }
            scores32[w] = wgt[0] * dot.max(0.0) * softmax_scale;
        }
        // Window 0's true score (2.0) beats window 1 (~1.0); fp32 collapses w0 to 0.0.
        assert!((scores64[0] - 2.0).abs() < 1e-3, "fp64 w0 score {}", scores64[0]);
        assert!(scores32[0] < 0.5, "fp32 w0 should collapse, got {}", scores32[0]);

        let sel64 = mirror_topk(&scores64, s, n_win, top_k, m, /*pos0*/ 10);
        let sel32 = mirror_topk(&scores32, s, n_win, top_k, m, 10);
        eprintln!("ITEM B: fp64 top-1 = {:?}  vs  fp32 top-1 = {:?}", sel64, sel32);
        assert_eq!(sel64[0], 0, "fp64 must pick the true winner window 0");
        assert_eq!(sel32[0], 1, "fp32 flips to the rival window 1");
        assert_ne!(sel64[0], sel32[0], "precision must change the selection (the drift source)");
    }

    /// ★ GATE 2a (offline structural): `forward_mv` over the CPU backend reproduces
    /// the transformers reference argmax on the tiny model — proving the re-expressed
    /// GPU op-DAG (fused MLA / DSA-compressor / mHC / MoE bodies) is faithful. The
    /// on-node GPU-backend twin (`Dsv4GpuStage`) then only swaps `mm` → the shipped
    /// mlx kernels. Same fixture as `dsv4_forward::tests::full_forward_matches_reference_tiny`.
    #[test]
    fn forward_mv_cpu_matches_reference_tiny() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/selftest.json");
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("SKIP: selftest.json fixture absent");
                return;
            }
        };
        let j: Value = serde_json::from_str(&raw).unwrap();
        let cfg = Dsv4Config::from_json(&j["config"]).unwrap();
        let input_ids: Vec<u32> =
            j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut d = HashMap::new();
        let mut i = HashMap::new();
        for (k, v) in j["weights"].as_object().unwrap() {
            d.insert(k.clone(), f32v(v));
        }
        for (k, v) in j["weights_i64"].as_object().unwrap() {
            i.insert(k.clone(), v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect());
        }
        let src = DictSrc { d, i };
        let mut mv = CpuMv { src: &src };

        let logits = forward_mv(&cfg, &input_ids, &mut mv);
        let vocab = cfg.vocab_size;
        let last = &logits[(input_ids.len() - 1) * vocab..];
        let (mine, _) = argmax_last(&logits, input_ids.len(), vocab);
        let ref_argmax = j["ref_argmax"].as_u64().unwrap() as u32;
        let ref_logits = f32v(&j["ref_last_logits"]);
        let mae = last.iter().zip(ref_logits.iter()).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("forward_mv(CpuMv): argmax mine={mine} ref={ref_argmax} logits_mae={mae:.3e}");
        assert_eq!(mine, ref_argmax, "forward_mv argmax mismatch (mae={mae:.3e})");
        assert!(mae < 5e-2, "forward_mv logits max_abs_err {mae:.3e} too large");

        // ★ PP-decomposition proof: chaining per-layer windows (streams hopped
        // stage-to-stage) reproduces the monolithic logits BYTE-FOR-BYTE. This is
        // the correctness core of PP-10 minus the GPU kernels + the wire.
        let mut mvc = CachedCpuMv::new(&src);
        let mut streams: Option<Vec<f32>> = None;
        let l = cfg.num_hidden_layers;
        let chained = loop {
            let mut acc: Option<Vec<f32>> = None;
            for li in 0..l {
                match forward_mv_window(&cfg, &input_ids, streams.take(), li, li + 1, &mut mvc) {
                    WindowOut::Streams(s) => streams = Some(s),
                    WindowOut::Logits(lg) => acc = Some(lg),
                }
            }
            break acc.expect("last window logits");
        };
        assert_eq!(chained.len(), logits.len(), "window-chain logits len");
        let cmae = chained.iter().zip(logits.iter()).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("window-chain vs monolithic: max_abs_diff={cmae:.3e} (expect 0)");
        assert_eq!(cmae, 0.0, "PP window chain must be byte-identical to monolithic forward_mv");
    }

    /// ★ DECODE-CORRECTNESS GATE (M2, the decisive one): chained one-token-at-a-time
    /// [`decode_step`] over the rolling KV + compressor cache reproduces the batched
    /// [`forward_mv`] prefill logits at EVERY position, bit-for-bit. This proves the
    /// stateful decode is bit-exact (prefix-stable compressor — see the module's
    /// Step-0 note) and subsumes "prefill-then-step": any prefix warm followed by
    /// steps is just a partition of the same chain. Tiny all-layer-types fixture
    /// (sliding + CSA + HCA, hash_moe + moe), <1s on Mac.
    #[test]
    fn decode_chain_matches_prefill_tiny() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/selftest.json");
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("SKIP: selftest.json fixture absent");
                return;
            }
        };
        let j: Value = serde_json::from_str(&raw).unwrap();
        let cfg = Dsv4Config::from_json(&j["config"]).unwrap();
        let input_ids: Vec<u32> =
            j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut d = HashMap::new();
        let mut i = HashMap::new();
        for (k, v) in j["weights"].as_object().unwrap() {
            d.insert(k.clone(), f32v(v));
        }
        for (k, v) in j["weights_i64"].as_object().unwrap() {
            i.insert(k.clone(), v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect());
        }
        let src = DictSrc { d, i };
        let vocab = cfg.vocab_size;
        let n = input_ids.len();

        // Batched prefill reference (all positions).
        let mut mv = CpuMv { src: &src };
        let prefill = forward_mv(&cfg, &input_ids, &mut mv);

        // Chained decode: ingest one token per step, compare each step's logits to
        // the batched prefill row at that position.
        let mut cache = Dsv4DecodeCache::new(&cfg);
        let mut worst = 0.0f32;
        for (t, &id) in input_ids.iter().enumerate() {
            let step = decode_step(&cfg, id, &mut cache, &mut mv);
            assert_eq!(step.len(), vocab, "decode step logits len");
            let refrow = &prefill[t * vocab..(t + 1) * vocab];
            let diff = step.iter().zip(refrow.iter()).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            worst = worst.max(diff);
            assert_eq!(diff, 0.0, "decode step {t} (pos {t}) diverges from batched prefill (max_abs={diff:.3e})");
        }
        // Argmax of the last decode step matches the batched prefill argmax.
        let mut cache2 = Dsv4DecodeCache::new(&cfg);
        let mut last = Vec::new();
        for &id in &input_ids {
            last = decode_step(&cfg, id, &mut cache2, &mut mv);
        }
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (ix, &v) in last.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = ix;
            }
        }
        let (ref_arg, _) = argmax_last(&prefill, n, vocab);
        eprintln!("decode-chain vs prefill: worst max_abs_diff={worst:.3e} (expect 0); argmax dec={bi} ref={ref_arg}");
        assert_eq!(bi as u32, ref_arg, "decode last-step argmax != prefill argmax");
    }

    /// ★ CSA INDEXER SHORT-CIRCUIT BIT-EXACT GATE (offline, no GPU). Drives the
    /// decode chain over the tiny fixture (CSA layers 2 & 4, `m=4`, `index_topk=4`,
    /// so every decode step has `n_win = t1/m <= index_topk` → the short-circuit
    /// regime). Proves THREE things with `max_abs_diff == 0`:
    ///   (a) flag OFF  == batched-prefill full-indexer oracle (else-branch unchanged);
    ///   (b) flag ON   == batched-prefill full-indexer oracle (the skip is exact — the
    ///       indexer really does admit exactly the causal windows here);
    ///   (c) flag ON   == flag OFF, step for step (pure-additive guard, A/B-clean).
    /// (b) is the decisive claim: an INDEPENDENT full-indexer path (prefill) confirms
    /// the skipped indexer's last-row visibility is all-visible, not merely that ON
    /// reproduces itself.
    #[test]
    fn csa_shortcircuit_bit_identical_tiny() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/selftest.json");
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => { eprintln!("SKIP: selftest.json fixture absent"); return; }
        };
        let j: Value = serde_json::from_str(&raw).unwrap();
        let cfg = Dsv4Config::from_json(&j["config"]).unwrap();
        let input_ids: Vec<u32> =
            j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut d = HashMap::new();
        let mut i = HashMap::new();
        for (k, v) in j["weights"].as_object().unwrap() { d.insert(k.clone(), f32v(v)); }
        for (k, v) in j["weights_i64"].as_object().unwrap() {
            i.insert(k.clone(), v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect());
        }
        let src = DictSrc { d, i };
        let vocab = cfg.vocab_size;

        // Confirm the tiny fixture actually exercises the short-circuit regime:
        // at least one CSA layer + max n_win over the chain <= index_topk.
        let has_csa = cfg.layer_types.iter().any(|&lt| lt == LayerType::CompressedSparse);
        let max_nwin = input_ids.len() / cfg.compress_rate_csa; // t1/m at the last step
        assert!(has_csa, "fixture has no CSA layer — test would not exercise the short-circuit");
        assert!(max_nwin <= cfg.index_topk,
            "fixture max n_win {max_nwin} > index_topk {} — not the short-circuit regime", cfg.index_topk);

        // Independent oracle: batched prefill runs the FULL indexer (flag-agnostic).
        let mut mv = CpuMv { src: &src };
        let prefill = forward_mv(&cfg, &input_ids, &mut mv);

        let run_chain = |mv: &mut CpuMv<DictSrc>| -> Vec<Vec<f32>> {
            let mut cache = Dsv4DecodeCache::new(&cfg);
            input_ids.iter().map(|&id| decode_step(&cfg, id, &mut cache, mv)).collect()
        };

        // (a) flag OFF: full-indexer decode path.
        std::env::set_var("VLLM_VULKAN_DSV4_CSA_SHORTCIRCUIT", "0");
        let off = run_chain(&mut mv);
        // (b+c) flag ON: short-circuit decode path.
        std::env::set_var("VLLM_VULKAN_DSV4_CSA_SHORTCIRCUIT", "1");
        let on = run_chain(&mut mv);
        std::env::remove_var("VLLM_VULKAN_DSV4_CSA_SHORTCIRCUIT");

        let mut worst_on_pref = 0.0f32;
        let mut worst_off_pref = 0.0f32;
        let mut worst_on_off = 0.0f32;
        for (t, (on_t, off_t)) in on.iter().zip(off.iter()).enumerate() {
            assert_eq!(on_t.len(), vocab);
            let refrow = &prefill[t * vocab..(t + 1) * vocab];
            let d_on = on_t.iter().zip(refrow).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            let d_off = off_t.iter().zip(refrow).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            let d_onoff = on_t.iter().zip(off_t).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            worst_on_pref = worst_on_pref.max(d_on);
            worst_off_pref = worst_off_pref.max(d_off);
            worst_on_off = worst_on_off.max(d_onoff);
            assert_eq!(d_off, 0.0, "flag-OFF step {t} diverges from prefill (max_abs={d_off:.3e})");
            assert_eq!(d_on, 0.0, "flag-ON short-circuit step {t} diverges from prefill (max_abs={d_on:.3e})");
            assert_eq!(d_onoff, 0.0, "flag ON vs OFF step {t} not bit-identical (max_abs={d_onoff:.3e})");
        }
        eprintln!(
            "CSA short-circuit: steps={} max_nwin={max_nwin} index_topk={} | ON-vs-prefill={worst_on_pref:.3e} OFF-vs-prefill={worst_off_pref:.3e} ON-vs-OFF={worst_on_off:.3e} (all expect 0)",
            on.len(), cfg.index_topk,
        );
    }

    /// ★ INCREMENTAL COMPRESSOR CACHE BIT-EXACT GATE (offline, no GPU). The
    /// `VLLM_VULKAN_DSV4_COMPRESSOR_CACHE` decode lever keeps a per-layer append-only
    /// compressed-KV plane and pools only the window(s) that just closed, instead of
    /// re-running the outer compressor over the full `x_hist` every token. This drives
    /// the whole chained decode with the cache ON and proves `max_abs_diff == 0` vs:
    ///   (a) the independent batched-prefill oracle (`forward_mv`), and
    ///   (b) the cache-OFF chained decode, step for step (pure-additive guard).
    /// The tiny fixture exercises CSA (m=4, 2 windows close over 10 steps); the HCA
    /// window-close decomposition is proven separately at scale by the `dsv4_dsa`
    /// unit gate `incremental_window_matches_full_outer_csa_and_hca` (HCA m=128 never
    /// closes here). Both together fully gate the lever offline.
    #[test]
    fn compressor_cache_bit_identical_tiny() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/selftest.json");
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => { eprintln!("SKIP: selftest.json fixture absent"); return; }
        };
        let j: Value = serde_json::from_str(&raw).unwrap();
        let cfg = Dsv4Config::from_json(&j["config"]).unwrap();
        let input_ids: Vec<u32> =
            j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut d = HashMap::new();
        let mut i = HashMap::new();
        for (k, v) in j["weights"].as_object().unwrap() { d.insert(k.clone(), f32v(v)); }
        for (k, v) in j["weights_i64"].as_object().unwrap() {
            i.insert(k.clone(), v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect());
        }
        let src = DictSrc { d, i };
        let vocab = cfg.vocab_size;

        let mut mv = CpuMv { src: &src };
        let prefill = forward_mv(&cfg, &input_ids, &mut mv);
        let run_chain = |mv: &mut CpuMv<DictSrc>| -> Vec<Vec<f32>> {
            let mut cache = Dsv4DecodeCache::new(&cfg);
            input_ids.iter().map(|&id| decode_step(&cfg, id, &mut cache, mv)).collect()
        };

        // cache OFF (full recompute path) and cache ON (incremental). Short-circuit
        // stays default-ON in both so the ON path exercises the CSA incremental arm.
        // The flag is DEFAULT-ON now, so OFF must be set explicitly.
        std::env::set_var("VLLM_VULKAN_DSV4_COMPRESSOR_CACHE", "0");
        let off = run_chain(&mut mv);
        std::env::set_var("VLLM_VULKAN_DSV4_COMPRESSOR_CACHE", "1");
        let on = run_chain(&mut mv);
        std::env::remove_var("VLLM_VULKAN_DSV4_COMPRESSOR_CACHE");

        let mut worst_on_pref = 0.0f32;
        let mut worst_on_off = 0.0f32;
        for (t, (on_t, off_t)) in on.iter().zip(off.iter()).enumerate() {
            assert_eq!(on_t.len(), vocab);
            let refrow = &prefill[t * vocab..(t + 1) * vocab];
            let d_on = on_t.iter().zip(refrow).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            let d_onoff = on_t.iter().zip(off_t).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            worst_on_pref = worst_on_pref.max(d_on);
            worst_on_off = worst_on_off.max(d_onoff);
            assert_eq!(d_on, 0.0, "cache-ON step {t} diverges from prefill (max_abs={d_on:.3e})");
            assert_eq!(d_onoff, 0.0, "cache ON vs OFF step {t} not bit-identical (max_abs={d_onoff:.3e})");
        }
        eprintln!(
            "compressor-cache: steps={} | ON-vs-prefill={worst_on_pref:.3e} ON-vs-OFF={worst_on_off:.3e} (all expect 0)",
            on.len(),
        );
    }

    /// Backend == [`CpuMv`] EXCEPT `hc_resident_fused()` returns true, so the DECODE
    /// path takes the resident HC-site branch of [`decoder_layer_decode`] and calls
    /// `attn_tail_hc`. With the DEFAULT (host) `attn_tail_hc` the fused branch is
    /// arithmetically identical to the non-fused one — it just reorders the SAME host
    /// ops (attn_tail → hc_residual_mix → hc_block). This backend proves the fused
    /// ORCHESTRATION (state threading: streams', post_f/comb_f, x_f) is a byte-exact
    /// re-expression, independent of the GPU. On node the same branch calls the real
    /// resident recorders (two-float Sinkhorn cos>0.9999 → argmax gate authoritative).
    struct FusedCpuMv<'a, S: Dsv4Src> { src: &'a S }
    impl<'a, S: Dsv4Src> Mv for FusedCpuMv<'a, S> {
        fn mm(&mut self, name: &str, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
            crate::model::cpu_matmul(x, &self.src.linear(name, out_f, in_f), s, in_f, out_f)
        }
        fn mm_expert(&mut self, name: &str, e: usize, x: &[f32], s: usize, in_f: usize, out_f: usize) -> Vec<f32> {
            crate::model::cpu_matmul(x, &self.src.expert(name, e, out_f, in_f), s, in_f, out_f)
        }
        fn dq_linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32> {
            self.src.linear(name, out_f, in_f)
        }
        fn dense(&self, name: &str) -> Vec<f32> { self.src.dense(name) }
        fn dense_i64(&self, name: &str) -> Vec<i64> { self.src.dense_i64(name) }
        fn embed_rows(&self, ids: &[u32], vocab: usize, h: usize) -> Vec<f32> {
            self.src.embed_rows(ids, vocab, h)
        }
        fn hc_resident_fused(&self) -> bool { true }
    }

    /// ★ FUSED HC-SITE ORCHESTRATION GATE (offline, no GPU): the decode chain over
    /// [`FusedCpuMv`] (fused branch, host math) is BYTE-IDENTICAL to the [`CpuMv`]
    /// chain. Validates that the resident HC-site restructuring of
    /// [`decoder_layer_decode`] threads state exactly (nothing lost/reordered), so
    /// any on-node divergence is purely the two-float Sinkhorn — not a wiring bug.
    #[test]
    fn fused_hc_orchestration_matches_host_decode_tiny() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/selftest.json");
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => { eprintln!("SKIP: selftest.json fixture absent"); return; }
        };
        let j: Value = serde_json::from_str(&raw).unwrap();
        let cfg = Dsv4Config::from_json(&j["config"]).unwrap();
        let input_ids: Vec<u32> =
            j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut d = HashMap::new();
        let mut i = HashMap::new();
        for (k, v) in j["weights"].as_object().unwrap() { d.insert(k.clone(), f32v(v)); }
        for (k, v) in j["weights_i64"].as_object().unwrap() {
            i.insert(k.clone(), v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect());
        }
        let src = DictSrc { d, i };
        let vocab = cfg.vocab_size;

        let mut host = CpuMv { src: &src };
        let mut fused = FusedCpuMv { src: &src };
        let mut ch = Dsv4DecodeCache::new(&cfg);
        let mut cf = Dsv4DecodeCache::new(&cfg);
        let mut worst = 0.0f32;
        for &id in &input_ids {
            let a = decode_step(&cfg, id, &mut ch, &mut host);
            let b = decode_step(&cfg, id, &mut cf, &mut fused);
            assert_eq!(a.len(), vocab);
            let diff = a.iter().zip(b.iter()).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            worst = worst.max(diff);
            assert_eq!(diff, 0.0, "fused HC-site orchestration diverges from host decode (max_abs={diff:.3e})");
        }
        eprintln!("fused-HC-site orchestration vs host decode: worst max_abs_diff={worst:.3e} (expect 0)");
    }

    /// ★ PP-window decode decomposition: chaining per-STAGE [`decode_step_window`]
    /// calls (one `Dsv4DecodeCache` per PP stage, streams hopped stage-to-stage —
    /// exactly what `pp_dsv4.py` rings over vCCL) reproduces the single-node
    /// [`decode_step`] chain bit-for-bit. Isolates the PP decode dataflow from the
    /// GPU kernels + the wire. 2-stage split on the tiny fixture.
    #[test]
    fn decode_pp_window_chain_matches_single_node_tiny() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/selftest.json");
        let raw = match std::fs::read_to_string(path) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("SKIP: selftest.json fixture absent");
                return;
            }
        };
        let j: Value = serde_json::from_str(&raw).unwrap();
        let cfg = Dsv4Config::from_json(&j["config"]).unwrap();
        let input_ids: Vec<u32> =
            j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let mut d = HashMap::new();
        let mut i = HashMap::new();
        for (k, v) in j["weights"].as_object().unwrap() {
            d.insert(k.clone(), f32v(v));
        }
        for (k, v) in j["weights_i64"].as_object().unwrap() {
            i.insert(k.clone(), v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect());
        }
        let src = DictSrc { d, i };
        let vocab = cfg.vocab_size;
        let l = cfg.num_hidden_layers;
        let bounds = pp_bounds(l, 3); // 3-stage PP over 6 layers

        // single-node chain
        let mut mv = CpuMv { src: &src };
        let mut c_single = Dsv4DecodeCache::new(&cfg);
        let mut single_last = Vec::new();

        // PP-staged chain: one cache per stage.
        let n_stages = bounds.len() - 1;
        let mut mv2 = CpuMv { src: &src };
        let mut stage_caches: Vec<Dsv4DecodeCache> =
            (0..n_stages).map(|s| Dsv4DecodeCache::new_window(&cfg, bounds[s], bounds[s + 1])).collect();
        let mut pp_last = Vec::new();

        for &id in &input_ids {
            single_last = decode_step(&cfg, id, &mut c_single, &mut mv);

            let mut streams: Option<Vec<f32>> = None;
            for s in 0..n_stages {
                match decode_step_window(&cfg, id, streams.take(), &mut stage_caches[s], &mut mv2) {
                    WindowOut::Streams(st) => streams = Some(st),
                    WindowOut::Logits(lg) => pp_last = lg,
                }
            }
        }
        assert_eq!(single_last.len(), vocab);
        assert_eq!(pp_last.len(), vocab);
        let diff = single_last.iter().zip(pp_last.iter()).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("PP-decode chain vs single-node decode: max_abs_diff={diff:.3e} (expect 0); bounds={bounds:?}");
        assert_eq!(diff, 0.0, "PP-staged decode must be byte-identical to single-node decode");
    }

    const DIR: &str = "/Volumes/Shared_Drive/models/DeepSeek-V4-Flash-0731-2.4bit-mixed";

    /// Checkpoint dir: `$VLLM_VULKAN_DSV4_DIR` if set (on-node: usually
    /// `/mnt/nas/models/DeepSeek-V4-Flash-0731-2.4bit-mixed`), else the Mac SMB mount.
    fn ckpt_dir() -> String {
        std::env::var("VLLM_VULKAN_DSV4_DIR").unwrap_or_else(|_| DIR.to_string())
    }

    /// ★ GATE 2a (real-ckpt DAG proof, CPU backend, no GPU): `forward_mv` — the
    /// EXACT op-DAG the GPU stage runs, only with `mm` = cpu_matmul instead of the
    /// mlx kernels — reproduces the golden argmax 11111 (` Paris`) on the real 86GB
    /// checkpoint. This proves the re-expressed forward is faithful on the REAL model
    /// (not just the tiny fixture); the on-node GPU twin then swaps only the (already
    /// cos=1.0) kernels. `#[ignore]` (heavy — full 43-layer dequant forward):
    ///   cargo test --lib dsv4_gpu::tests::gate2a_forward_mv_real_argmax_11111 -- --ignored --nocapture
    ///
    /// ★ CONFIRMED (pass-5, 2026-08-15): `finite=true argmax=11111 val=25.799`,
    /// wall=2572.5s — the CachedCpuMv union-dedup made the real-86GB CPU-DAG forward
    /// tractable (single-threaded 2-bit dequant over the SMB-mounted ckpt). The DAG is
    /// LOCKED on the real model.
    #[test]
    #[ignore]
    fn gate2a_forward_mv_real_argmax_11111() {
        use crate::dsv4_loader::{is_dsv4_dir, Dsv4RealSrc};
        if !is_dsv4_dir(&ckpt_dir()) {
            eprintln!("SKIP: checkpoint not present");
            return;
        }
        let cfg_j: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/real_config.json")).unwrap(),
        ).unwrap();
        let cfg = Dsv4Config::from_json(&cfg_j).unwrap();
        let input_ids: Vec<u32> = vec![671, 6102, 294, 8760, 344];
        let t0 = std::time::Instant::now();
        let src = Dsv4RealSrc::open(&ckpt_dir()).unwrap();
        let mut mv = CachedCpuMv::new(&src);
        let logits = forward_mv(&cfg, &input_ids, &mut mv);
        let (argmax, val) = argmax_last(&logits, input_ids.len(), cfg.vocab_size);
        let finite = logits[(input_ids.len() - 1) * cfg.vocab_size..].iter().all(|x| x.is_finite());
        eprintln!(
            "GATE 2a (forward_mv CpuMv, real ckpt): finite={finite} argmax={argmax} (expect 11111) val={val:.3} wall={:.1}s",
            t0.elapsed().as_secs_f32()
        );
        assert!(finite, "logits not finite");
        assert_eq!(argmax, 11111, "forward_mv real argmax {argmax} != golden 11111");
    }

    /// Minimax layer BOUNDS splitting `n_layers` across `n_stages` PP windows as
    /// evenly as possible (the first `n_layers % n_stages` stages get one extra) —
    /// the `pp_dsv4.py` split. Returns `n_stages+1` boundaries `[0, …, n_layers]`.
    fn pp_bounds(n_layers: usize, n_stages: usize) -> Vec<usize> {
        let base = n_layers / n_stages;
        let extra = n_layers % n_stages;
        let mut b = vec![0usize];
        for s in 0..n_stages {
            let w = base + if s < extra { 1 } else { 0 };
            b.push(b[s] + w);
        }
        b
    }

    /// Dump the CPU-oracle `[0, W)` window PREFILL streams to `$DSV4_WINDOW_OUT`
    /// (json `{"W":W,"streams":[..]}`) — the golden the on-node first-GPU-run script
    /// (`gpu_window_onnode.py`, env `DSV4_GOLDEN`) cross-checks the GPU window streams
    /// against (cos vs the exact CPU `forward_mv_window`). Runs on the Mac (SMB ckpt +
    /// RAM). `#[ignore]`:
    ///   DSV4_WINDOW_OUT=/tmp/dsv4_win0_5.json W=5 cargo test --lib \
    ///     dsv4_gpu::tests::dump_cpu_window_golden -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_cpu_window_golden() {
        use crate::dsv4_loader::{is_dsv4_dir, Dsv4RealSrc};
        if !is_dsv4_dir(&ckpt_dir()) {
            eprintln!("SKIP: checkpoint not present");
            return;
        }
        let w: usize = std::env::var("W").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
        let out = std::env::var("DSV4_WINDOW_OUT").unwrap_or_else(|_| "/tmp/dsv4_win.json".into());
        let cfg_j: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/real_config.json")).unwrap(),
        ).unwrap();
        let cfg = Dsv4Config::from_json(&cfg_j).unwrap();
        let input_ids: Vec<u32> = vec![671, 6102, 294, 8760, 344];
        let t0 = std::time::Instant::now();
        let src = Dsv4RealSrc::open(&ckpt_dir()).unwrap();
        let mut mv = CachedCpuMv::new(&src);
        let streams = match forward_mv_window(&cfg, &input_ids, None, 0, w, &mut mv) {
            WindowOut::Streams(s) => s,
            WindowOut::Logits(_) => panic!("window [0,{w}) unexpectedly produced logits (== full depth?)"),
        };
        let finite = streams.iter().all(|x| x.is_finite());
        let mx = streams.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        eprintln!(
            "dump_cpu_window_golden: W={w} len={} finite={finite} max={mx:.4e} wall={:.1}s -> {out}",
            streams.len(), t0.elapsed().as_secs_f32()
        );
        std::fs::write(&out, serde_json::to_string(&serde_json::json!({"W": w, "streams": streams})).unwrap()).unwrap();
    }

    /// ★ GATE 2b (CPU proof of the PP decomposition, real ckpt): split the 43 layers
    /// into 10 PP windows (the PP-10 minimax bounds) and chain them stage-to-stage,
    /// passing ONLY the `[seq, hc*h]` stream payload across boundaries — exactly what
    /// `pp_dsv4.py` rings over vCCL. The chained last-stage argmax must still be the
    /// golden 11111 (` Paris`). This isolates the PP dataflow correctness from the GPU
    /// kernels + the wire: on-node GATE 2b then only swaps `mm` → the (cos=1.0) mlx
    /// kernels and the in-process stream hop → the vCCL hop. `#[ignore]` (heavy):
    ///   cargo test --lib dsv4_gpu::tests::gate2b_pp10_window_chain_cpu_argmax_11111 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gate2b_pp10_window_chain_cpu_argmax_11111() {
        use crate::dsv4_loader::{is_dsv4_dir, Dsv4RealSrc};
        if !is_dsv4_dir(&ckpt_dir()) {
            eprintln!("SKIP: checkpoint not present");
            return;
        }
        let cfg_j: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/real_config.json")).unwrap(),
        ).unwrap();
        let cfg = Dsv4Config::from_json(&cfg_j).unwrap();
        let input_ids: Vec<u32> = vec![671, 6102, 294, 8760, 344];
        let bounds = pp_bounds(cfg.num_hidden_layers, 10);
        eprintln!("PP-10 bounds: {bounds:?}");
        let t0 = std::time::Instant::now();
        let src = Dsv4RealSrc::open(&ckpt_dir()).unwrap();
        let mut mv = CachedCpuMv::new(&src);
        let mut streams: Option<Vec<f32>> = None;
        let mut argmax = u32::MAX;
        let mut val = 0.0f32;
        for w in 0..10 {
            let (ls, le) = (bounds[w], bounds[w + 1]);
            match forward_mv_window(&cfg, &input_ids, streams.take(), ls, le, &mut mv) {
                WindowOut::Streams(s) => streams = Some(s),
                WindowOut::Logits(lg) => {
                    let (a, v) = argmax_last(&lg, input_ids.len(), cfg.vocab_size);
                    argmax = a;
                    val = v;
                }
            }
        }
        eprintln!(
            "GATE 2b (PP-10 window chain, CPU): argmax={argmax} (expect 11111) val={val:.3} wall={:.1}s",
            t0.elapsed().as_secs_f32()
        );
        assert_eq!(argmax, 11111, "PP-10 window-chain argmax {argmax} != golden 11111");
    }

    /// ★ GATE 2a (on-node): the GPU quant-resident forward reproduces the golden
    /// argmax 11111 (` Paris`) on the real 86GB checkpoint. `#[ignore]` — run on a
    /// RADV GFX1013 node with the checkpoint present:
    ///   cargo test --lib dsv4_gpu::tests::gate2a_gpu_resident_argmax_11111 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gate2a_gpu_resident_argmax_11111() {
        use crate::dsv4_loader::is_dsv4_dir;
        if !is_dsv4_dir(&ckpt_dir()) {
            eprintln!("SKIP: checkpoint not present");
            return;
        }
        let cfg_j: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/real_config.json")).unwrap(),
        ).unwrap();
        let cfg = Dsv4Config::from_json(&cfg_j).unwrap();
        let input_ids: Vec<u32> = vec![671, 6102, 294, 8760, 344];
        let t0 = std::time::Instant::now();
        let mut stage =
            Dsv4GpuStage::from_ckpt_streamed(&ckpt_dir(), &cfg, 0, cfg.num_hidden_layers, true, 0).unwrap();
        let load = t0.elapsed().as_secs_f32();
        let (argmax, val) = stage.argmax_last(&input_ids);
        eprintln!(
            "GATE 2a (GPU-resident): argmax={argmax} (expect 11111) val={val:.3} load={load:.1}s wall={:.1}s",
            t0.elapsed().as_secs_f32()
        );
        assert_eq!(argmax, 11111, "GPU-resident argmax {argmax} != golden 11111");
    }

    /// ★ GATE 2b (on-node, single-GPU DECODE self-consistency): the GPU-resident
    /// rolling decode chain must reproduce the GPU-resident batched-prefill logits
    /// at every position bit-for-bit (same `Mv` kernels, only the cache differs).
    /// Proves the stateful decode is correct on the REAL checkpoint over the GPU
    /// backend — the single-node prerequisite for the PP-10 fleet decode.
    /// `#[ignore]` — run on a RADV GFX1013 node with the checkpoint present:
    ///   cargo test --lib dsv4_gpu::tests::gate2b_gpu_decode_matches_prefill -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gate2b_gpu_decode_matches_prefill() {
        use crate::dsv4_loader::is_dsv4_dir;
        if !is_dsv4_dir(&ckpt_dir()) {
            eprintln!("SKIP: checkpoint not present");
            return;
        }
        let cfg_j: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/real_config.json")).unwrap(),
        ).unwrap();
        let cfg = Dsv4Config::from_json(&cfg_j).unwrap();
        let input_ids: Vec<u32> = vec![671, 6102, 294, 8760, 344];
        let vocab = cfg.vocab_size;
        let mut stage =
            Dsv4GpuStage::from_ckpt_streamed(&ckpt_dir(), &cfg, 0, cfg.num_hidden_layers, true, 0).unwrap();

        // Batched prefill over the GPU backend.
        let prefill = stage.forward(&input_ids);

        // Chained decode over the same GPU backend; compare each step to prefill.
        stage.reset_decode_cache();
        let mut worst = 0.0f32;
        let mut last = Vec::new();
        for (t, &id) in input_ids.iter().enumerate() {
            last = match stage.decode_step_stage(id, None) {
                WindowOut::Logits(l) => l,
                WindowOut::Streams(_) => unreachable!(),
            };
            let refrow = &prefill[t * vocab..(t + 1) * vocab];
            let diff = last.iter().zip(refrow.iter()).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            worst = worst.max(diff);
        }
        let (dec_arg, _) = {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in last.iter().enumerate() {
                if v > bv { bv = v; bi = i; }
            }
            (bi as u32, bv)
        };
        let (ref_arg, _) = argmax_last(&prefill, input_ids.len(), vocab);
        eprintln!("GATE 2b (GPU decode vs prefill): worst_max_abs={worst:.3e} dec_argmax={dec_arg} ref_argmax={ref_arg} (expect 11111)");
        assert_eq!(dec_arg, ref_arg, "GPU decode argmax != prefill argmax");
        assert_eq!(dec_arg, 11111, "GPU decode argmax {dec_arg} != golden 11111");
    }

    /// ★ STAGED GATE 2c (on-node, the composed GPU-wired vs CPU-oracle cross-check):
    /// the GPU-resident forward with the mHC + CSA-DSA ops WIRED onto the device must
    /// reproduce the CPU-oracle `forward_mv(CachedCpuMv)` decision (argmax 11111) and
    /// stay logit-cos≈1. This is the direct `forward_mv(GpuStage) vs forward_mv(CpuMv)`
    /// hook the perf re-measure gates on — it proves that collapsing the 86 mHC host
    /// round-trips + (with `VLLM_VULKAN_DSV4_DSA_GPU=1`) the 21 CSA compressor chains
    /// onto the GPU did not move the composed op-DAG. Loads the 86GB ckpt twice (GPU +
    /// CPU dequant), so `#[ignore]` — run on a RADV GFX1013 node with the ckpt:
    ///   cargo test --lib dsv4_gpu::tests::gate2c_gpu_wired_matches_cpu_oracle -- --ignored --nocapture
    /// A/B the wire with `VLLM_VULKAN_DSV4_HC_GPU=1` / `VLLM_VULKAN_DSV4_DSA_GPU=1`.
    #[test]
    #[ignore]
    fn gate2c_gpu_wired_matches_cpu_oracle() {
        use crate::dsv4_loader::{is_dsv4_dir, Dsv4RealSrc};
        if !is_dsv4_dir(&ckpt_dir()) {
            eprintln!("SKIP: checkpoint not present");
            return;
        }
        let cfg_j: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/real_config.json")).unwrap(),
        ).unwrap();
        let cfg = Dsv4Config::from_json(&cfg_j).unwrap();
        let input_ids: Vec<u32> = vec![671, 6102, 294, 8760, 344];
        let vocab = cfg.vocab_size;

        // GPU-wired forward (host mHC by default; mHC-GPU per VLLM_VULKAN_DSV4_HC_GPU,
        // DSA per VLLM_VULKAN_DSV4_DSA_GPU — the resident matvecs are always GPU).
        let mut stage =
            Dsv4GpuStage::from_ckpt_streamed(&ckpt_dir(), &cfg, 0, cfg.num_hidden_layers, true, 0).unwrap();
        let gpu_logits = stage.forward(&input_ids);
        let (ga, gv) = argmax_last(&gpu_logits, input_ids.len(), vocab);

        // CPU oracle forward (host mHC + host CSA).
        let src = Dsv4RealSrc::open(&ckpt_dir()).unwrap();
        let mut mv = CachedCpuMv::new(&src);
        let cpu_logits = forward_mv(&cfg, &input_ids, &mut mv);
        let (ca, _) = argmax_last(&cpu_logits, input_ids.len(), vocab);

        // last-row logit cosine (GPU-wired vs CPU-oracle).
        let n = input_ids.len();
        let (lr_g, lr_c) = (&gpu_logits[(n - 1) * vocab..], &cpu_logits[(n - 1) * vocab..]);
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (a, b) in lr_g.iter().zip(lr_c.iter()) { dot += *a as f64 * *b as f64; na += *a as f64 * *a as f64; nb += *b as f64 * *b as f64; }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
        eprintln!("GATE 2c (GPU-wired vs CPU-oracle): gpu_argmax={ga} (val {gv:.3}) cpu_argmax={ca} last-logit cos={cos:.9} (expect 11111, cos≈1)");
        assert_eq!(ga, ca, "GPU-wired argmax != CPU-oracle argmax");
        assert_eq!(ga, 11111, "GPU-wired argmax {ga} != golden 11111");
        assert!(cos > 0.9999, "GPU-wired vs oracle logit cos {cos} too low");
    }
}
