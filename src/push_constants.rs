// SPDX-License-Identifier: Apache-2.0
//! Push-constant builders + matvec/nvfp4/fp8 shader-variant selection.
//!
//! Extracted verbatim from lib.rs (M1). No logic changes — see git history
//! for the original context/derivation of each push-constant layout.

use crate::compute;
use crate::flags::{self, QuantFormat};
use crate::model;

/// Returns true if this weight tensor should be uploaded to GPU as f16.
/// Norm weights, scalars, and embeddings stay as f32 for precision.
pub(crate) fn is_matvec_weight(name: &str) -> bool {
    // Projection weights: q/k/v/o/gate/up/down, PLE gate, PLE projection
    name.ends_with("_proj.weight")          // e.g. q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        || name.ends_with("_gate.weight")   // e.g. per_layer_input_gate.weight
        || name.ends_with("_projection.weight")  // e.g. per_layer_projection.weight
        // embed_tokens.weight stays f32 (LM head precision), layernorm/scalar stay f32
}

pub(crate) fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    // Both targets are little-endian; bytemuck reinterprets, then one memcpy.
    bytemuck::cast_slice::<f32, u8>(data).to_vec()
}

/// Select the `mul_mat_vec` shader variant.  Defaults to the scalar (non-
/// subgroup) shader, which is correct on all devices; the `_subgroup` variants
/// are faster but assume a subgroup size that is wrong on some GPUs (e.g. RADV
/// GFX1013 / AMD wave64), producing incorrect results.  Opt into them with
/// VLLM_VULKAN_USE_SUBGROUP=1 on hardware where they are verified correct.
/// Pick the matvec pipeline variant + the rows-per-workgroup (NUM_ROWS) for an
/// output of `n` rows. With NUM_ROWS=1 a matvec launches one workgroup per
/// output row — for the LM head (n≈151936) that is ~150k tiny workgroups and the
/// kernel runs far below memory bandwidth (~26 GB/s measured). The `_r4` variant
/// (NUM_ROWS=4, compiled by PipelineCache::compile_matvec) makes each workgroup
/// emit 4 rows, amortizing launch/occupancy overhead. Only the scalar variants
/// have `_r*` siblings, and only large outputs benefit, so gate on both.
/// Cached snapshot of VLLM_VULKAN_QUANT (GPU weight residency format). Read once;
/// the chosen matvec kernel must match the format weights were quantized into at
/// load time, so this must not change mid-process.
pub(crate) fn quant_format() -> QuantFormat {
    flags::flags_global().quant
}

/// Cached snapshot of VLLM_VULKAN_MATVEC_ROWS (rows-per-workgroup override).
pub(crate) fn matvec_rows_override() -> Option<u32> {
    flags::flags_global().matvec_rows
}

/// Cached snapshot of VLLM_VULKAN_USE_SUBGROUP (opt into the `_subgroup` matvec
/// variants). Read once — the pipeline set is fixed at construction.
pub(crate) fn use_subgroup_flag() -> bool {
    flags::flags_global().use_subgroup
}

/// Cached snapshot of VLLM_VULKAN_MLX4_W8 (kill switch for the mlx4w8sg MoE
/// down-dispatch wiring; see `matvec_mlx4_variant_k`). Default ON.
pub(crate) fn mlx4_w8sg_down_flag() -> bool {
    flags::flags_global().mlx4_w8sg_down
}

/// Cached snapshot of VLLM_VULKAN_MLX4_REPACK (route the dense decode mlx4 4-bit
/// matvec through the `mul_mat_vec_mlx4repack_f32_f32` VALU-bound repack refactor
/// — see `matvec_mlx4_variant_k`). Default OFF; v1 stays the oracle.
pub(crate) fn mlx4_repack_flag() -> bool {
    flags::flags_global().mlx4_repack
}

/// Cached snapshot of VLLM_VULKAN_MLX4_REPACK_R8 (r8 geometry lever on the repack
/// — see `matvec_mlx4_variant_k` and the flag field doc). Default ON (2026-07-30
/// fleet A/B GO); =0 restores the r4 geometry baseline.
pub(crate) fn mlx4_repack_r8_flag() -> bool {
    flags::flags_global().mlx4_repack_r8
}

/// L1: cached snapshot of VLLM_VULKAN_MLX4_RGU_REPACK (route the SMALL-n (n==512)
/// MoE gate/up mlx4 4-bit matvec — the Qwen3.6-35B-A3B shape (k=2048,n=512) the
/// shipped `mlx4_repack` n>=1024 clause excludes — through the SAME
/// `mul_mat_vec_mlx4repack_f32_f32_bs64_r4` shader; see `matvec_mlx4_variant_k`).
/// Default ON (2026-07-30 PP-5 fleet A/B GO); =0 reverts to the v1 kernel oracle.
/// Kept separate from `mlx4_repack` so the shipped n>=1024 branch is never relaxed.
pub(crate) fn mlx4_rgu_repack_flag() -> bool {
    flags::flags_global().mlx4_rgu_repack
}

/// Cached snapshot of VLLM_VULKAN_MOE_F16_SCALES (store the routed-expert affine
/// scales+biases f16 in the resident MoE GTT buffer + read them via the
/// `mul_mat_vec_mlx4_f16scale_f32_f32` matvec variant — the 122B PP-6 fit-enabler,
/// see `ensure_moe_gpu_layer` / `matvec_mlx4_moe_variant_k`). Default OFF.
pub(crate) fn moe_f16_scales_flag() -> bool {
    flags::flags_global().moe_f16_scales
}

/// Cached snapshot of VLLM_VULKAN_NVFP4_REPACK (route the NVFP4 4-bit matvec
/// through the `mul_mat_vec_nvfp4repack_f32_f32` VALU-bound repack refactor — or
/// its e4m3-resident twin when VLLM_VULKAN_NVFP4_E4M3_SCALES is also on — see
/// `nvfp4_dispatch` / `matvec_nvfp4_variant_k`). Default OFF; v1 stays the
/// oracle. This is the win that LANDS E2E on the PP-resident NVFP4 fleet.
pub(crate) fn nvfp4_repack_flag() -> bool {
    flags::flags_global().nvfp4_repack
}

/// Cached snapshot of VLLM_VULKAN_LAGUNA_EXPERT_REPACK (route Laguna's routed-
/// expert NVFP4 e4m3 matvec through the `mul_mat_vec_nvfp4_e4m3repack_f32_f32`
/// repack kernel instead of the v1 `mul_mat_vec_nvfp4_e4m3` it dispatches
/// directly — see `laguna_gpu::expert_matvec`). Default ON since 2026-07-30
/// (productionized); =0 reverts to the v1 argmax-exact oracle for a clean
/// single-node A/B. Primary Laguna decode lever (repack is address-gen-free; v1
/// is address-gen-bound at ~75 GB/s).
pub(crate) fn laguna_expert_repack_flag() -> bool {
    flags::flags_global().laguna_expert_repack
}

/// Cached snapshot of VLLM_VULKAN_FP8_FAST (opt into the arithmetic-decode +
/// vec4-load fp8 matvec, `mul_mat_vec_fp8_fast.comp`, instead of the const-LUT
/// `mul_mat_vec_fp8.comp`). Default ON (see flags.rs `fp8_fast` = bdef1): the
/// arithmetic-decode fast path is live; `=0` reverts to the const-LUT oracle.
pub(crate) fn fp8_fast_flag() -> bool {
    flags::flags_global().fp8_fast
}

/// Cached snapshot of VLLM_VULKAN_FP8_REPACK (route the fp8_fast matvec through
/// the `mul_mat_vec_fp8repack_f32_f32` subgroup-reduction twin — see
/// `matvec_fp8_variant_k`). Default OFF; fp8_fast stays the oracle. NOT a load
/// repack (fp8 is not address-gen-bound — s4rig ISA rig); it swaps only the LDS
/// barrier-tree reduction for a subgroupAdd reduction.
pub(crate) fn fp8_repack_flag() -> bool {
    flags::flags_global().fp8_repack
}

/// Base fp8 matvec shader stem, gated by `fp8_fast_flag`.
fn fp8_base() -> &'static str {
    if fp8_fast_flag() { "mul_mat_vec_fp8fast_f32_f32" } else { "mul_mat_vec_fp8_f32_f32" }
}

/// Cached snapshot of VLLM_VULKAN_GEMM_F16ALIGNED (kill switch for the GEMM
/// campaign Phase 1 wiring; see `gemm_variant_k`). Default ON.
pub(crate) fn gemm_f16aligned_flag() -> bool {
    flags::flags_global().gemm_f16aligned
}

/// Cached snapshot of VLLM_VULKAN_GEMM_QUANT (kill switch for the Phase A
/// quant-batched-GEMM wiring; see `qwen35_forward.rs::qwen35_gemm` and
/// `gemm_variant_quant_k`). Default OFF, unlike `gemm_f16aligned_flag` — this
/// is a brand-new, cluster-unvalidated dispatch path (no on-node cos/argmax
/// gate has run yet; see plan-quant-batched-matmul.md's cluster validation
/// gate). OFF preserves today's serial-`qwen35_matvec`-per-token fallback as
/// the clean A/B baseline; flip to `=1` only after that gate passes.
pub(crate) fn gemm_quant_flag() -> bool {
    flags::flags_global().gemm_quant
}

// ─── GEMM (mul_mm) campaign Phase 1 — sweep-winner picker ───────────────────
//
// (BM, BN, WARP, TM, TN) tile geometry for the `matmul_f16_f32_f16_aligned`
// SPIR-V (f16 arithmetic + ALIGNED vec8 loads; GEMM campaign Phase 0,
// scripts/compile_shaders.sh). WM=WN=32, WMITER=2 are held fixed — the
// `debug_gemm_geometry` sweep tool's own fixed axes (see plan-gemm-campaign.md
// §3) — so a 5-tuple fully determines the compiled pipeline.
pub(crate) type GemmGeom = (u32, u32, u32, u32, u32); // (BM, BN, WARP, TM, TN)

/// Sweep-derived winner geometry keyed by the EXACT (k, n) dispatch shape, or
/// `None` for any shape outside the swept set (byte-identical fallback to the
/// live `matmul_f16_f32_fp32` BM=BN=64 kernel — G3 no-regression by
/// construction). Source: cluster sweep 2026-07-10, pinned 1850MHz,
/// `gemm_sweep_phase2a.csv`, T=1 (decode/spec-verify regime) rows, both
/// correctness gates passing (cos>=0.999999 structural, plus the f16-arith
/// looser cos>=0.999 bar — see plan-gemm-campaign.md §4). Speedup vs the live
/// kernel at this geometry: dn-qkv 1.76x, dn-out 2.33x, gate-up 2.27x,
/// router 2.79x, lm_head 2.13x.
///
/// HONEST CAVEATS (do not drop when editing this table):
///   - iters=10 per sweep row (noisy timing) — the RELATIVE ranking is robust
///     but the absolute speedup numbers above are provisional pending a
///     cluster re-measure at iters=200 (queued).
///   - f16-arith is NOT bit-exact vs the f32-accumulate live kernel (cos
///     ~0.999, ~1e-3 max-abs-diff) — an argmax-exact re-validation against the
///     f16f32 baseline is queued on the cluster. If it flips a near-tie
///     argmax, the safe universal fallback is `matmul_f16_f32_fp32_aligned`
///     (f32-arith, ALIGNED loads, BIT-EXACT — cos>=0.999999), which won
///     dn-qkv at 1.76x on its own with no precision tradeoff; that variant
///     is compiled (Phase 0) but not yet wired into this picker.
pub(crate) fn gemm_pick(k: usize, n: usize) -> Option<GemmGeom> {
    match (k, n) {
        (2048, 8192)    => Some((64, 32, 32, 2, 4)), // dn-qkv   (DeltaNet in->qkv)
        (4096, 2048)    => Some((32, 32, 64, 4, 2)), // dn-out   (DeltaNet/attn out)
        (2048, 512)     => Some((32, 64, 64, 4, 2)), // gate-up  (MoE expert / FFN gate-up)
        (2048, 256)     => Some((32, 32, 64, 4, 2)), // router   (MoE router)
        (2048, 151936)  => Some((32, 32, 32, 4, 4)), // lm_head

        // ── Gemma4-E2B (M1 flip): forward_prefill_gemma's gpu_gemm projection
        // shapes (hidden=1536, ple_dim=256, head_dim in {256,512} sliding/full,
        // q_dim=8*head_dim, kv_dim=1*head_dim, ffn_inter in {6144,12288}
        // non-shared/kv-shared layers). No cluster sweep exists yet for these
        // exact shapes (only the QWEN geometries above are sweep-derived), so
        // — per the M1 plan — each is assigned the COMPILED combo (one of the
        // 4 GEMM_GEOM_COMBOS siblings above) whose qwen shape is the closest
        // analog by role, not a fresh sweep winner:
        //   - k/v_proj (n=256 / n=512) match the qwen router/gate-up shapes
        //     EXACTLY on n (256/512) — same GQA-narrow-output role, reuse
        //     their winners verbatim.
        //   - q_proj / o_proj ("square-ish", moderate n<=2048 or k=4096
        //     exactly matching dn-out's k) reuse dn-out's combo.
        //   - gate_proj/up_proj (wide fan-out, n=6144/12288 >> k=1536) reuse
        //     dn-qkv's wide-fan-out combo.
        //   - down_proj / per_layer_projection (contracting to hidden=1536,
        //     large-to-huge k) reuse dn-out's combo (same contracting role).
        // TODO(cluster gate): re-sweep these on .53 and replace with
        // shape-exact winners once gemma_sweep_phase2a-equivalent data exists.
        (1536, 2048)    => Some((32, 32, 64, 4, 2)), // q_proj (sliding, head_dim=256)
        (1536, 4096)    => Some((64, 32, 32, 2, 4)), // q_proj (full-attn, head_dim=512)
        (1536, 256)     => Some((32, 32, 64, 4, 2)), // k/v_proj (sliding) + per_layer_input_gate
        (1536, 512)     => Some((32, 64, 64, 4, 2)), // k/v_proj (full-attn, head_dim=512)
        (2048, 1536)    => Some((32, 32, 64, 4, 2)), // o_proj (sliding, q_dim=2048)
        (4096, 1536)    => Some((32, 32, 64, 4, 2)), // o_proj (full-attn, q_dim=4096)
        (1536, 6144)    => Some((64, 32, 32, 2, 4)), // gate/up_proj (non-kv-shared layers)
        (1536, 12288)   => Some((64, 32, 32, 2, 4)), // gate/up_proj (kv-shared layers, 2x inter)
        (6144, 1536)    => Some((32, 32, 64, 4, 2)), // down_proj (non-kv-shared layers)
        (12288, 1536)   => Some((32, 32, 64, 4, 2)), // down_proj (kv-shared layers, 2x inter)
        (256, 1536)     => Some((32, 32, 64, 4, 2)), // per_layer_projection

        _ => None,
    }
}

/// The distinct geometry siblings compiled at startup for
/// `matmul_f16_f32_f16_aligned` (see `PipelineCache::compile_mul_mm_geom`).
/// Every `gemm_pick` winner must be one of these combos or the dispatch would
/// name a missing pipeline (asserted by `gemm_pick_combos_compiled`).
pub(crate) const GEMM_GEOM_COMBOS: &[GemmGeom] = &[
    (64, 32, 32, 2, 4),
    (32, 32, 64, 4, 2),
    (32, 64, 64, 4, 2),
    (32, 32, 32, 4, 4),
];

/// `<base>_bm{BM}_bn{BN}_w{WARP}_tm{TM}_tn{TN}` name for a compiled geometry
/// sibling (mirrors `geom_name`'s `_bs{bs}_r{r}` convention for matvec).
pub(crate) fn gemm_geom_name(base: &str, g: GemmGeom) -> String {
    let (bm, bn, warp, tm, tn) = g;
    format!("{base}_bm{bm}_bn{bn}_w{warp}_tm{tm}_tn{tn}")
}

/// Shape-keyed GEMM picker: returns `(pipeline_name, BM, BN)` for the wg-size
/// math at the call site. Inside the swept win region (and
/// VLLM_VULKAN_GEMM_F16ALIGNED != "0") this is the `matmul_f16_f32_f16_aligned`
/// geometry sibling; otherwise (or with the kill switch set) it is the
/// byte-identical legacy `matmul_f16_f32_fp32` BM=BN=64 dispatch — the same
/// no-regression-by-construction pattern as `matvec_mlx4_variant_k`.
pub(crate) fn gemm_variant_k(k: usize, n: usize) -> (String, u32, u32) {
    const LEGACY: &str = "matmul_f16_f32_fp32";
    const LEGACY_BM: u32 = 64;
    const LEGACY_BN: u32 = 64;
    if gemm_f16aligned_flag() {
        if let Some(g) = gemm_pick(k, n) {
            let (bm, bn, _warp, _tm, _tn) = g;
            return (gemm_geom_name("matmul_f16_f32_f16_aligned", g), bm, bn);
        }
    }
    (LEGACY.to_string(), LEGACY_BM, LEGACY_BN)
}

/// Pure core of `matvec_shader`: pick the scalar/subgroup, f16/f32 base shader.
pub(crate) fn matvec_shader_core(f16_weight: bool, use_sg: bool) -> &'static str {
    match (f16_weight, use_sg) {
        (true, true) => "mul_mat_vec_f16_f32_f32_subgroup",
        (true, false) => "mul_mat_vec_f16_f32_f32",
        (false, true) => "mul_mat_vec_f32_f32_f32_subgroup",
        (false, false) => "mul_mat_vec_f32_f32_f32",
    }
}

pub(crate) fn matvec_shader(f16_weight: bool) -> &'static str {
    matvec_shader_core(f16_weight, use_subgroup_flag())
}

/// Pure core of `matvec_variant`: choose the base shader (from the quant format)
/// and the rows-per-workgroup variant for an `n`-row output. Depends only on its
/// arguments so it is unit-testable without a GPU or process env.
///
/// q8_0-resident weights (VLLM_VULKAN_QUANT=q8_0): dispatch the q8_0 matvec.
/// Input/output stay f32; only the weight buffer is quantized.
///
/// Use the DEQUANTIZE kernel (mul_mat_vec_q8_0deq_*, from mul_mat_vec.comp),
/// NOT the MMQ kernel (mul_mat_vec_q8_0_*, from mul_mat_vecq.comp). The MMQ
/// kernel reads the activation vector on binding 1 as a q8_1 block buffer
/// (block_q8_1_x4); our dispatch uploads raw f32 activations there, so MMQ
/// reinterprets f32 bytes as q8_1 and produces all-zero/garbage logits. The
/// dequantize kernel reads f32 B directly and dequantizes the q8_0 weight.
///
/// Rows-per-workgroup. With r1 a matvec launches one workgroup per output
/// row, which on GFX1013 leaves the kernel well below memory bandwidth.
/// Measured (0.6B, single node): r1 27.2ms → r4 22.1ms → r8 21.2ms/tok,
/// argmax identical. r8 wins even for the small projections (n=1024 still
/// saturates via in-workgroup parallelism). Override with
/// VLLM_VULKAN_MATVEC_ROWS; only {1,2,4,8} are compiled (see compile_matvec).
/// r8 for any sizeable output (the proven max on GFX1013 — r16/r32 hang
/// pipeline creation, so they're not compiled).
pub(crate) fn matvec_variant_core(
    quant: QuantFormat, use_sg: bool, f16_weight: bool, rows_override: Option<u32>, n: usize,
) -> (String, u32) {
    let base = match quant {
        QuantFormat::Q8_0 => "mul_mat_vec_q8_0deq_f32_f32",
        QuantFormat::Q4_0 => "mul_mat_vec_q4_0deq_f32_f32",
        QuantFormat::Q4_K => "mul_mat_vec_q4_kdeq_f32_f32",
        QuantFormat::Bf16 => "mul_mat_vec_bf16_f32_f32",
        _ => matvec_shader_core(f16_weight, use_sg),
    };
    // Subgroup variants have no _r* siblings.
    if base.ends_with("_subgroup") {
        return (base.to_string(), 1);
    }
    let mut r: u32 = if n >= 1024 { 8 } else { 1 };
    if let Some(v) = rows_override {
        r = v;
    }
    if !matches!(r, 2 | 4 | 8) {
        return (base.to_string(), 1);
    }
    (format!("{base}_r{r}"), r)
}

pub(crate) fn matvec_variant(f16_weight: bool, n: usize) -> (String, u32) {
    matvec_variant_core(quant_format(), use_subgroup_flag(), f16_weight, matvec_rows_override(), n)
}

/// Matvec variant keyed on the WEIGHT's ACTUAL stored `format`, NOT the global
/// `VLLM_VULKAN_QUANT` flag. A resident weight can be a different format than the
/// global quant — most importantly the MTP draft head's dense projections
/// (`mtp.fc`/`q/k/v/o`), which `load_mtp_gpu` ALWAYS uploads as F16 regardless of
/// the flag. Under `VLLM_VULKAN_QUANT=q8_0` the global-keyed `matvec_variant`
/// would pick the q8_0 dequant shader and read those F16 bytes as q8_0 blocks →
/// garbage draft logits (acc_rate=0). Dispatching by the weight's own format
/// fixes it and is a no-op for the main model (its weights ARE the global format).
pub(crate) fn matvec_variant_by_format(fmt: QuantFormat, n: usize) -> (String, u32) {
    let f16 = matches!(fmt, QuantFormat::F16);
    matvec_variant_core(fmt, use_subgroup_flag(), f16, matvec_rows_override(), n)
}

/// Batched matvec variant for forward_batched: (shader, rows) for `t` columns.
/// Reads the resident weight format (VLLM_VULKAN_QUANT) so a q8_0/q4-resident
/// batched verify streams+dequantizes each weight ONCE across all T columns
/// (Design C, the quantized batched-matmul flip) instead of T serial matvecs.
/// See `matvec_cols_variant_core`.
pub(crate) fn matvec_cols_variant(n: usize, t: usize) -> (String, u32) {
    matvec_cols_variant_core(quant_format(), n, t)
}

/// Pure core of `matvec_cols_variant` (unit-testable, no GPU/env).
///
/// t==1 => the exact per-token decode variant (`matvec_variant_core`), byte-
/// identical to a serial decode by construction.
///
/// t>1 => a SCALAR `_r{rows}_c{t}` base with NUM_COLS=t and rows=max(1,8/t)
/// (rows*t<=8). Only scalar `_r*_c*` pipelines are compiled (compile_matvec_cols;
/// the `_subgroup` reduction has no cols siblings). The base is chosen from the
/// weight codec:
///   - q8_0/q4_0/q4_K resident -> the DEQUANT matvec base (mul_mat_vec.comp,
///     shared with the decode kernel) so each weight is read+dequantized once
///     per workgroup and MAC'd across all T columns. pipeline.rs compiles these
///     `_r*_c*` siblings from the same SPIR-V.
///   - anything else (f16/bf16/…) -> the f16 scalar base (the pre-flip behavior;
///     f16 weights already amortize the read across columns with no dequant).
/// Bit-exact per column vs the serial decode matvec: same kernel, same
/// accumulation order; the shader's dequant-hoist reorder is bit-neutral.
pub(crate) fn matvec_cols_variant_core(quant: QuantFormat, n: usize, t: usize) -> (String, u32) {
    if t <= 1 {
        return matvec_variant_core(quant, use_subgroup_flag(), true, matvec_rows_override(), n);
    }
    let base = match quant {
        QuantFormat::Q8_0 => "mul_mat_vec_q8_0deq_f32_f32",
        QuantFormat::Q4_0 => "mul_mat_vec_q4_0deq_f32_f32",
        QuantFormat::Q4_K => "mul_mat_vec_q4_kdeq_f32_f32",
        _ => matvec_shader_core(true, false),
    };
    let rows = (8 / t as u32).max(1);
    (format!("{base}_r{rows}_c{t}"), rows)
}

/// Push constants for the standalone single-stream multi-column matvec kernels
/// (`mul_mat_vec_f16_cols` / `mul_mat_vec_q8_0_cols` — see those shaders):
/// `{ uint ncols(=k); uint nrows(=n); }`. Distinct from `matvec_pc13` (the ggml
/// base.glsl layout) — these kernels index the weight/activation/output
/// directly from k,n, so only two words are needed.
pub(crate) fn matvec_cols_pc2(k: usize, n: usize) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(2 * 4);
    v.write_all(&(k as u32).to_le_bytes()).unwrap();
    v.write_all(&(n as u32).to_le_bytes()).unwrap();
    v
}

pub(crate) fn matvec_pc13(k: usize, n: usize) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(13 * 4);
    for x in [
        k as u32, k as u32, k as u32, n as u32,
        (k * n) as u32, k as u32, n as u32,
        0u32, 0u32, 1u32, 1u32, 1u32, 1u32,
    ] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v
}

/// Push constants for the MLX-affine 4-bit matvec (`mul_mat_vec_mlx4_f32_f32`).
/// Layout: { uint ncols(=k); uint nrows(=n); uint group_size; } — see
/// shaders/mul_mat_vec_mlx4.comp.
pub(crate) fn matvec_mlx4_pc(k: usize, n: usize, group_size: usize) -> Vec<u8> {
    matvec_mlx4_pc_off(k, n, group_size, 0, 0)
}

/// Push constants for the MLX4 matvec WITH per-expert base offsets. `packed_off`
/// is the WORD offset into the packed[] buffer where this expert's slice begins
/// (= expert * out_features * (in_features/8)); `sb_off` is the ELEMENT offset
/// into scales[]/biases[] (= expert * out_features * groups). For a 2D weight
/// (non-MoE) both offsets are 0 — see `matvec_mlx4_pc`.
pub(crate) fn matvec_mlx4_pc_off(k: usize, n: usize, group_size: usize, packed_off: usize, sb_off: usize) -> Vec<u8> {
    // The 4-bit mlx4/nvfp4 matvec shaders compute words_per_row = k/8; a non-
    // multiple-of-8 k truncates and shifts every row after the first (silent
    // garbage). All k values are known host-side, so assert here.
    assert_eq!(k % 8, 0,
        "mlx4/nvfp4 matvec requires k (in_features) divisible by 8; got k={k}");
    use std::io::Write;
    let mut v = Vec::with_capacity(5 * 4);
    for x in [k as u32, n as u32, group_size as u32, packed_off as u32, sb_off as u32] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v
}

/// Rows-per-workgroup variant selector for the MLX4 matvec. Mirrors the r-tier
/// logic in `matvec_variant`: r8 for sizeable outputs, r1 for small ones. Only
/// {1,2,4,8} are compiled (compile_matvec); r16/r32 hang GFX1013 pipeline
/// creation. Overridable via VLLM_VULKAN_MATVEC_ROWS.
pub(crate) fn matvec_mlx4_variant(n: usize) -> (String, u32) {
    let base = "mul_mat_vec_mlx4_f32_f32";
    let mut r: u32 = if n >= 1024 { 8 } else { 1 };
    if let Some(v) = matvec_rows_override() {
        r = v;
    }
    if !matches!(r, 2 | 4 | 8) {
        return (base.to_string(), 1);
    }
    (format!("{base}_r{r}"), r)
}

// ─── Kernel-geometry lever (BLOCK_SIZE by dispatch shape) ────────────────────
//
// The legacy matvec geometry is BLOCK_SIZE=512 for every shape. For the small
// contraction dims of the Qwen3.6-35B MoE/deltanet dispatches each thread does
// only ceil(k/512) = 1..4 MACs and then pays a log2(512) = 9-round barrier-
// separated shared-memory tree reduction — the reduction dwarfs the loads
// (work:sync 4:9 for gate/up k=2048, 1:9 for down k=512). Smaller BLOCK_SIZE
// raises loads/thread and cuts reduction rounds. The winners below are the
// min-µs cos==1.0 rows of the on-node debug_mlx4_geometry/debug_matvec_geometry
// sweep (GFX1013, real 35B-A3B-4bit weights, 2026-07-04 geom_sweep.csv):
//
//   mlx4 gate/up (k=2048,n=512): bs512_r1 40.5µs → bs128_r2 30.3µs  (−25%)
//   mlx4 down    (k=512,n=2048): bs512_r8 45.8µs → bs128_r2 31.0µs  (−32%)
//   bf16 dn qkv  (k=2048,n=8192): bs512_r8 110.8µs → bs256_r2 88.5µs (−20%)
//   bf16 dn z    (k=2048,n=4096): bs512_r8 62.7µs → bs256_r2 46.4µs  (−26%)
//   bf16 dn out  (k=4096,n=2048): bs512_r8 58.6µs → bs256_r2 47.2µs  (−19%)
//   f32 shared-down (k=512,n=2048): bs512_r8 19.3µs → bs64_r1 11.3µs (−41%)
//   f32 router   (k=2048,n=256): bs512_r1 90.4µs → bs32_r2 59.5µs    (−34%)
//   q8_0deq dn-qkv/attn-q (k=2048,n=8192): bs512_r8 82.1µs → bs32_r2 48.1µs (−41%)
//   q8_0deq dn-z (k=2048,n=4096): bs512_r8 43.9µs → bs64_r1 25.2µs   (−43%)
//   q8_0deq attn-k/v (k=2048,n=512): bs512_r1 54.6µs → bs32_r4 47.5µs (−13%)
//   q8_0deq dn-out/attn-o (k=4096,n=2048): bs512_r8 33.2µs → bs32_r4 25.9µs (−23%)
//   (dn a/b k=2048,n=32 [both bases] and f32 shared-gate/up k=2048,n=512:
//    legacy already wins / noise-level — no entry, legacy path. The q8_0deq
//    rows were re-swept on an idle node and are stable across two runs.)
//
// The table is keyed by EXACT (base, k, n) — the tightest possible win region.
// Any other shape (lm_head, dense q/k/v/o, dense FFN, TP shards) returns None
// and keeps the byte-identical legacy BS=512 dispatch (G3 no-regression by
// construction). Only the MoE/deltanet dispatch sites call the `_k` pickers.

/// Sweep-derived (BLOCK_SIZE, NUM_ROWS) winner for one dispatch shape, or None
/// for the legacy BS=512 geometry. Keep entries in sync with the compiled
/// combos (`geom_combos_for` — asserted by `geom_pick_combos_compiled`).
pub(crate) fn geom_pick(base: &str, k: usize, n: usize) -> Option<(u32, u32)> {
    match (base, k, n) {
        ("mul_mat_vec_mlx4_f32_f32", 2048, 512) => Some((128, 2)),  // MoE gate/up
        ("mul_mat_vec_mlx4_f32_f32", 512, 2048) => Some((128, 2)),  // MoE down
        ("mul_mat_vec_bf16_f32_f32", 2048, 8192) => Some((256, 2)), // dn in_proj_qkv
        ("mul_mat_vec_bf16_f32_f32", 2048, 4096) => Some((256, 2)), // dn in_proj_z
        ("mul_mat_vec_bf16_f32_f32", 4096, 2048) => Some((256, 2)), // dn out_proj
        ("mul_mat_vec_f32_f32_f32", 512, 2048) => Some((64, 1)),    // shared down
        ("mul_mat_vec_f32_f32_f32", 2048, 256) => Some((32, 2)),    // MoE router
        // q8_0-resident deltanet/attention projections (VLLM_VULKAN_QUANT=q8_0,
        // the production 35B PP config). lm_head (2048, vocab=248320) shares
        // the base but no table entry -> legacy by construction.
        ("mul_mat_vec_q8_0deq_f32_f32", 2048, 8192) => Some((32, 2)), // dn qkv / attn q
        ("mul_mat_vec_q8_0deq_f32_f32", 2048, 4096) => Some((64, 1)), // dn z
        ("mul_mat_vec_q8_0deq_f32_f32", 2048, 512) => Some((32, 4)),  // attn k/v
        ("mul_mat_vec_q8_0deq_f32_f32", 4096, 2048) => Some((32, 4)), // dn out / attn o
        // nvfp4/fp8/f16 winners: cluster geometry sweep (cos=1.0, no pipeline
        // hang), NEW-2 Phase 2 (2026-07-14).
        ("mul_mat_vec_nvfp4_f32_f32", 1024, 1280) => Some((256, 2)), // MoE expert up (2.61x)
        ("mul_mat_vec_nvfp4_f32_f32", 1280, 1024) => Some((128, 1)), // MoE expert down
        ("mul_mat_vec_fp8_f32_f32", 4096, 5376) => Some((512, 4)),  // shared up
        ("mul_mat_vec_fp8_f32_f32", 5376, 4096) => Some((256, 1)),  // shared down
        ("mul_mat_vec_f16_f32_f32", 4096, 1024) => Some((512, 2)),  // fc1
        ("mul_mat_vec_f16_f32_f32", 4096, 4096) => Some((256, 2)),  // q_proj / o_proj
        _ => None,
    }
}

/// Qwen3.6-27B TP-4 per-rank q8_0 projection shapes (h=5120; heads sharded /4).
/// Kept SEPARATE from `geom_pick` so it is reached ONLY via the flag-gated qwen35
/// TP matvec sites (`matvec_variant_q35geom`, VLLM_VULKAN_Q35_GEOM=1) — the
/// unconditional resident-stage path (`matvec_variant_geom` at q35r_rec_mv) and
/// the single-node reference stay byte-identical when the flag is off.
///
/// These (bs,rows) are HEURISTIC: mapped by k/n role from the swept 35B winners
/// in `geom_pick`, reusing ONLY the already-compiled + GFX1013-vetted q8_0deq
/// combos {(32,2),(64,1),(32,4)}, so no new pipeline is created (no startup-hang
/// risk, no missing dispatch) and the dequant math is unchanged → argmax-
/// identical. A 27B on-node `debug_matvec_geometry` micro-sweep (incl. bs128/256)
/// would confirm/tune them; until then this is a safe first A/B geometry.
pub(crate) fn geom_pick_q35tp(base: &str, k: usize, n: usize) -> Option<(u32, u32)> {
    match (base, k, n) {
        ("mul_mat_vec_q8_0deq_f32_f32", 5120, 4352) => Some((32, 2)), // mlp gate/up (dominant)
        ("mul_mat_vec_q8_0deq_f32_f32", 5120, 3072) => Some((32, 2)), // attn q_proj (query|gate)
        ("mul_mat_vec_q8_0deq_f32_f32", 5120, 2560) => Some((32, 2)), // dn in_proj_qkv
        ("mul_mat_vec_q8_0deq_f32_f32", 5120, 1536) => Some((64, 1)), // dn in_proj_z
        ("mul_mat_vec_q8_0deq_f32_f32", 5120, 256)  => Some((32, 4)), // attn k/v_proj
        ("mul_mat_vec_q8_0deq_f32_f32", 1536, 5120) => Some((32, 4)), // attn o_proj / dn out_proj
        ("mul_mat_vec_q8_0deq_f32_f32", 4352, 5120) => Some((32, 4)), // mlp down_proj
        _ => None,
    }
}

/// The `(bs, rows)` geometry siblings compiled at startup for `base` (see
/// `PipelineCache::compile_matvec_geom`). Every `geom_pick` winner must be in
/// this list or the dispatch would name a missing pipeline. All combos were
/// creation-vetted on GFX1013 by the on-node sweep (no `_r16`-style hangs).
pub(crate) fn geom_combos_for(base: &str) -> &'static [(u32, u32)] {
    match base {
        "mul_mat_vec_mlx4_f32_f32" => &[(128, 2)],
        "mul_mat_vec_bf16_f32_f32" => &[(256, 2)],
        "mul_mat_vec_f32_f32_f32" => &[(64, 1), (32, 2)],
        "mul_mat_vec_q8_0deq_f32_f32" => &[(32, 2), (64, 1), (32, 4)],
        // 1850MHz-pinned re-sweep winner for the MoE down (k=512,n=2048) shape
        // only (perf/matvec-batch-dispatch); see matvec_mlx4_variant_k.
        "mul_mat_vec_mlx4w8sg_f32_f32" => &[(64, 2)],
        // VALU-bound repack refactor (VLLM_VULKAN_MLX4_REPACK). A small combo set
        // for the on-node dense-decode micro-sweep; BLOCK<=64 is single-subgroup
        // (no LDS combine). All bs<=128 (creation-vetted class, no _r16 hang).
        // (64,8)/(128,8) are the VLLM_VULKAN_MLX4_REPACK_R8 per-shape winners
        // (short-k->bs64/r8, long-k/down->bs128/r8); creation-vetted on GFX1013 by
        // the on-node debug_mlx4_geometry sweep (ok=True, cos=1.0). r8 == the v1
        // kernel's row count, so no new register/LDS-hang class.
        "mul_mat_vec_mlx4repack_f32_f32" => &[(32, 2), (64, 2), (128, 2), (64, 4), (128, 4), (64, 8), (128, 8)],
        // f16-scale REPACK twin (VLLM_VULKAN_MLX4_REPACK + VLLM_VULKAN_MOE_F16_SCALES):
        // the SCALE_F16 sibling routed by matvec_mlx4_moe_variant_k for the 122B-A10B
        // MoE routed experts (gate/up k=3072 n=1024, down k=1024 n=3072). bs64/r4 is
        // the wired default (== the f32 repack MoE-expert pick); the rest are for the
        // on-node micro-sweep. Same single-subgroup class (bs<=128, creation-vetted).
        "mul_mat_vec_mlx4repack_f16scale_f32_f32" => &[(32, 2), (64, 2), (128, 2), (64, 4), (128, 4)],
        // repack EXPERT-BATCHED (VLLM_VULKAN_LING_MOE_BATCH). Same single-subgroup
        // combos as the repack parent (bs<=128, creation-vetted class); bs64/r4 is
        // the wired default for Ling's small MoE experts (n=768/2560), the rest for
        // the on-node micro-sweep. Per-workgroup geometry is identical to the
        // single-expert repack — only the dispatch gains the expert (.y) axis.
        "mul_mat_vec_mlx4repack_batched_f32_f32" => &[(32, 2), (64, 2), (128, 2), (64, 4), (128, 4)],
        // NVFP4 VALU-bound repack refactor (VLLM_VULKAN_NVFP4_REPACK). Same combo
        // set as the mlx4 repack (bs64/r4 is the wired default; the rest are for
        // the on-node micro-sweep). BLOCK<=64 single-subgroup; all bs<=128
        // (creation-vetted class). The e4m3-resident twin reuses the same combos.
        "mul_mat_vec_nvfp4repack_f32_f32" => &[(32, 2), (64, 2), (128, 2), (64, 4), (128, 4)],
        "mul_mat_vec_nvfp4_e4m3repack_f32_f32" => &[(32, 2), (64, 2), (128, 2), (64, 4), (128, 4)],
        // FP8 subgroup-reduction twin (VLLM_VULKAN_FP8_REPACK). Reduction-epilogue
        // swap only (fp8 is not address-gen-bound). Same combo set as the mlx4/
        // nvfp4 repack twins; bs64/r4 is the wired default, the rest for the
        // on-node micro-sweep. BLOCK<=64 single-subgroup; all bs<=128.
        "mul_mat_vec_fp8repack_f32_f32" => &[(32, 2), (64, 2), (128, 2), (64, 4), (128, 4)],
        // nvfp4/fp8/f16: NEW-2 Phase 2 cluster-swept winners (cos=1.0, no
        // pipeline hang on GFX1013). Each combo is creation-vetted; do NOT
        // add an unswept `bs` here -- compile_matvec_geom is non-watchdog and
        // an unvetted combo can hang vkCreateComputePipelines on GFX1013.
        "mul_mat_vec_nvfp4_f32_f32" => &[(256, 2), (128, 1)],
        "mul_mat_vec_fp8_f32_f32" => &[(512, 4), (256, 1)],
        "mul_mat_vec_f16_f32_f32" => &[(512, 2), (256, 2)],
        _ => &[],
    }
}

/// `<base>_bs{bs}_r{rows}` name for a swept-winner shape, or None ⇒ legacy.
/// A VLLM_VULKAN_MATVEC_ROWS override wins (the legacy `_r{v}` A/B path).
fn geom_name(base: &str, k: usize, n: usize, rows_override: Option<u32>) -> Option<(String, u32)> {
    if rows_override.is_some() {
        return None;
    }
    let (bs, r) = geom_pick(base, k, n)?;
    debug_assert!(matches!(r, 1 | 2 | 4 | 8) && r * bs * 4 <= 16384);
    Some((format!("{base}_bs{bs}_r{r}"), r))
}

/// Shape-aware MLX4 matvec picker for the MoE dispatch sites: swept-winner
/// geometry inside the win region, byte-identical `matvec_mlx4_variant`
/// otherwise.
pub(crate) fn matvec_mlx4_variant_k(k: usize, n: usize) -> (String, u32) {
    // 1850MHz-pinned clock re-sweep (perf/matvec-batch-dispatch): the
    // word-granular-w8-load + subgroupAdd-reduction shader beat the wired v1
    // winner by 49-81% at EVERY (bs,rows) combo on the MoE down (k=512,n=2048)
    // shape specifically — best-of-sweep 6.47µs (bs64,rows2) vs v1's 28.81µs.
    // Gate/up (k=2048,n=512) stayed under the 25% decision bar (best 21.7%)
    // and is untouched — only this exact (k,n) is redirected, and only when
    // rows isn't manually overridden (VLLM_VULKAN_MATVEC_ROWS A/B path wins).
    if k == 512 && n == 2048 && matvec_rows_override().is_none() && mlx4_w8sg_down_flag() {
        return ("mul_mat_vec_mlx4w8sg_f32_f32_bs64_r2".to_string(), 2);
    }
    // VALU-bound repack refactor (VLLM_VULKAN_MLX4_REPACK, default OFF): the
    // DENSE decode mlx4 4-bit shapes AND the LARGER MoE-expert shapes (Kimi
    // gate/up k=2304/n=1024, down k=1024/n=2304; gemma-12b attn k=3840-4096).
    // The k%32==0 gate is the dwordx4 (uvec4) requirement AND — because it makes
    // words_per_row=k/8 a multiple of 4 — it GUARANTEES every per-expert base
    // (packed_off = e * n * (k/8)) is itself a multiple of 4, i.e. 16B-aligned
    // for the `buffer_load_dwordx4`, for ANY expert e and ANY n. So no per-expert
    // alignment check is needed: k%32==0 is necessary AND sufficient (see
    // shaders/mul_mat_vec_mlx4repack_f32_f32.comp REQUIRES clause and
    // debug_mlx4_repack_moe). The k>=1024 / n>=1024 bounds EXCLUDE the Qwen3.6-35B
    // MoE experts (down k=512, gate/up n=512) — those keep v1/w8sg — while
    // admitting the larger Kimi/gemma experts. group_size=64 (mlx affine, always)
    // has gs%32==0 so each 32-elem chunk lies wholly within one affine group.
    // rows-override A/B path still wins (returns None into the legacy `_r{v}`).
    // L1 (VLLM_VULKAN_MLX4_RGU_REPACK, default OFF): the SMALL-n MoE gate/up
    // sibling of the clause below. The shipped n>=1024 bound EXCLUDES the
    // Qwen3.6-35B-A3B MoE gate/up (k=2048, n=512) because the on-node dense sweep
    // only covered n>=1024. But the s4rig offline ISA classify shows the v1 kernel
    // gate/up currently dispatches is address-gen-bound (addr/unpack=20.11, the
    // exact pathology the repack collapses to 1.16) — and k/n are RUNTIME push
    // constants so the histogram is shape-invariant. k%32==0 makes words_per_row=
    // k/8 a mult of 4, so every per-expert base packed_off=e*n*(k/8) is 16B-aligned
    // for the dwordx4 load for ANY e and ANY n (identical alignment guarantee as
    // the n>=1024 clause). n==512 self-scopes to qwen35 gate/up (down is k=512/
    // n=2048 → the w8sg branch above). SEPARATE flag: the shipped default-ON
    // branch is never relaxed. Same bs64/r4 shader (already compiled), same
    // fma-factored/subgroupAdd rounding → argmax-exact (cos=1.0, mae~ULP).
    if mlx4_rgu_repack_flag() && matvec_rows_override().is_none()
        && k % 32 == 0 && n == 512 {
        return ("mul_mat_vec_mlx4repack_f32_f32_bs64_r4".to_string(), 4);
    }
    if mlx4_repack_flag() && matvec_rows_override().is_none()
        && k % 32 == 0 && k >= 1024 && n >= 1024 {
        // bs64/r4 = on-node .81 TP-shard sweep winner: down (k4352,n5120) 389.7->
        // 64.1us (6.1x, at the ~62us weight-read BW floor), gate/up (k5120,n4352)
        // 186.5->67.3us (2.8x). argmax-exact, cos=1.0 vs v1. (bs128/r4 marginally
        // faster on gate/up; bs64/r4 is the near-optimal single pick for both.)
        //
        // R8 LEVER (VLLM_VULKAN_MLX4_REPACK_R8, default OFF): the .53 GPU-tick
        // geometry re-sweep (debug_mlx4_geometry, real qwen27B-4bit weights,
        // cos=1.0 argmax-exact) shows the `rows` axis is the dominant knob
        // (r4>>r2>>r1, each ~1.5x — more independent dwordx4 weight loads in
        // flight per chunk closes the MLP-latency gap to the 263 GB/s unpack-ALU
        // ceiling). r8 extends it past the r4 cap: gate/up (k5120,n17408) bs64/r8
        // 245 GB/s (was 221, 1.11x), lm_head (k5120,n248320) bs64/r8 283 (was
        // 237, 1.19x), down (k17408,n5120) bs128/r8 240 (was 216, 1.11x). The
        // per-shape split is real: down (k>n) wants bs128/r8 (2 subgroups) while
        // short-k (k<=n) peaks at bs64/r8 (single subgroup). (64,8)/(128,8) are
        // in geom_combos_for so both pipelines exist at startup.
        if mlx4_repack_r8_flag() {
            return if k > n {
                ("mul_mat_vec_mlx4repack_f32_f32_bs128_r8".to_string(), 8)
            } else {
                ("mul_mat_vec_mlx4repack_f32_f32_bs64_r8".to_string(), 8)
            };
        }
        return ("mul_mat_vec_mlx4repack_f32_f32_bs64_r4".to_string(), 4);
    }
    if let Some(v) = geom_name("mul_mat_vec_mlx4_f32_f32", k, n, matvec_rows_override()) {
        return v;
    }
    matvec_mlx4_variant(n)
}

/// MoE routed-expert matvec picker. When VLLM_VULKAN_MOE_F16_SCALES is ON the
/// routed-expert scales/biases are resident f16, so this returns the
/// `mul_mat_vec_mlx4_f16scale_f32_f32[_r{r}]` variant (base mlx4 geometry: the
/// f16-scale sibling has no repack/w8sg/geom twins — fit-first for 122B, a perf
/// reconciliation with the repack path is noted for when both land). Rows are
/// picked exactly like the f32 base (`matvec_mlx4_variant(n)`). Flag OFF ⇒
/// byte-identical to `matvec_mlx4_variant_k`. The upload format in
/// `ensure_moe_gpu_layer` reads the SAME flag, so buffer and shader always agree.
pub(crate) fn matvec_mlx4_moe_variant_k(k: usize, n: usize) -> (String, u32) {
    if moe_f16_scales_flag() {
        // f16scale REPACK twin (VLLM_VULKAN_MLX4_REPACK, default ON): the routed-
        // expert shapes that qualify BY SHAPE (k%32==0, k>=1024, n>=1024) route to
        // the SCALE_F16 sibling of the VALU-bound repack refactor instead of the
        // address-gen-bound v1 f16scale kernel (mul_mat_vec_mlx4_f16scale_f32_f32,
        // bs512). The 122B-A10B experts all clear: gate/up k=3072 n=1024, down
        // k=1024 n=3072. Before this, the moe_f16_scales_flag() branch returned the
        // v1 f16scale variant UNCONDITIONALLY — so an expert that qualified for the
        // repack by shape never reached the repack branch of matvec_mlx4_variant_k
        // (MOE_F16_SCALES is mandatory-to-fit; f32 scales OOM the 122B). k%32==0
        // makes words_per_row=k/8 a mult of 4, so every per-expert base packed_off=
        // e*n*(k/8) is 16B-aligned for the dwordx4 load (same guarantee as the f32
        // twin — see matvec_mlx4_variant_k / debug_mlx4_repack_moe). f16 holds bf16
        // scales EXACTLY in range, so the only rounding is the repack's fma/subgroup
        // reassociation -> argmax-exact, cos=1.0/mae~ULP vs BOTH v1 f16scale and the
        // CPU oracle. v1 f16scale stays the oracle/fallback for out-of-gate shapes
        // (e.g. the Qwen3.6-35B-A3B experts: down k=512, gate/up n=512). rows-
        // override A/B path still wins (returns the legacy v1 f16scale `_r{v}`).
        if mlx4_repack_flag() && matvec_rows_override().is_none()
            && k % 32 == 0 && k >= 1024 && n >= 1024 {
            return ("mul_mat_vec_mlx4repack_f16scale_f32_f32_bs64_r4".to_string(), 4);
        }
        let (nm, r) = matvec_mlx4_variant(n);
        return (
            nm.replace(
                "mul_mat_vec_mlx4_f32_f32",
                "mul_mat_vec_mlx4_f16scale_f32_f32",
            ),
            r,
        );
    }
    matvec_mlx4_variant_k(k, n)
}

/// Convert f32 affine-quant scales/biases to f16 bytes for the resident MoE
/// buffer. The scales are bf16 on disk (widened to f32 in `QuantSwitch`); f16 has
/// 10 mantissa bits vs bf16's 7, so every value in f16's NORMAL exponent range is
/// bit-identical. Real MoE affine scales are ~O(1e-3..1e0) — normal-range, exact —
/// BUT measured on Qwen3.5-122B-A10B the switch-expert tensors carry a small tail
/// of tiny subnormal values (down to ~3e-7). Those round to the nearest f16
/// subnormal with NEGLIGIBLE absolute error (<~1e-6 on weights O(1e-2..1e0) →
/// argmax-exact / cos≈1.0), so we ACCEPT them — rejecting instead is what OOMed
/// the 122B load (nearly every switch tensor has the tail, so the strict
/// bit-exact guard fell back the ENTIRE model to f32/CPU). We return None ONLY for
/// a value that cannot be represented at all: a non-finite input, or a magnitude
/// overflowing f16's max (~65504 → ±Inf), which WOULD corrupt the dequant.
/// Measured 122B max|x|≈0.6, so no rejections in practice — the guard is a safety
/// net for a pathological future checkpoint. NOT bit-exact in general (the
/// subnormal tail rounds); the cluster run's argmax coherence is the final gate.
pub(crate) fn f32_scales_to_f16_bytes_safe(data: &[f32]) -> Option<Vec<u8>> {
    let mut bytes = vec![0u8; data.len() * 2];
    for (i, &v) in data.iter().enumerate() {
        // Reject non-finite inputs (a finite affine scale is never Inf/NaN; a
        // corrupt source must not be silently stored).
        if !v.is_finite() {
            return None;
        }
        let h = half::f16::from_f32(v);
        // Reject only overflow-to-Inf (|v| > f16 max ~65504) — that would corrupt
        // the dequant. Tiny subnormals round to finite near-values (negligible),
        // and are accepted.
        if !h.to_f32().is_finite() {
            return None;
        }
        bytes[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
    }
    Some(bytes)
}

/// Shape-aware f32 matvec picker for the shared-expert/router dispatch sites.
pub(crate) fn matvec_f32_variant_k(k: usize, n: usize) -> (String, u32) {
    if let Some(v) = geom_name("mul_mat_vec_f32_f32_f32", k, n, matvec_rows_override()) {
        return v;
    }
    matvec_f32_variant(n)
}

/// Shape-aware wrapper over `matvec_variant` for the deltanet/attention
/// projections (`q35r_rec_mv`, MvKind::Plain). Geometry siblings exist only
/// for the swept bf16 and q8_0deq bases, so the swept name is returned only
/// when the legacy pick resolves to one of them AND the exact shape is in the
/// win table; every other quant format / shape is untouched.
pub(crate) fn matvec_variant_geom(f16_weight: bool, k: usize, n: usize) -> (String, u32) {
    let legacy = matvec_variant(f16_weight, n);
    for base in ["mul_mat_vec_bf16_f32_f32", "mul_mat_vec_q8_0deq_f32_f32"] {
        if legacy.0.starts_with(base) {
            if let Some(v) = geom_name(base, k, n, matvec_rows_override()) {
                return v;
            }
        }
    }
    legacy
}

/// Flag-gated (VLLM_VULKAN_Q35_GEOM) shape-aware wrapper for the qwen3.6 TP
/// matvec sites (`qwen35_matvec`/`qwen35_matvec_multi`). Adds the 27B TP-4
/// per-rank q8_0 win table (`geom_pick_q35tp`) on top of the base swept table,
/// so the TP path picks the tuned tiling for its sharded projection shapes.
/// A VLLM_VULKAN_MATVEC_ROWS override still forces the legacy `_r{v}` A/B path.
/// Every non-tabled shape returns the exact `matvec_variant` legacy string.
pub(crate) fn matvec_variant_q35geom(k: usize, n: usize) -> (String, u32) {
    let legacy = matvec_variant(true, n);
    if matvec_rows_override().is_none() {
        for base in ["mul_mat_vec_bf16_f32_f32", "mul_mat_vec_q8_0deq_f32_f32"] {
            if legacy.0.starts_with(base) {
                // 27B TP-4 shapes first (this flag's target), then the base
                // swept (35B) shapes for completeness.
                if let Some((bs, r)) = geom_pick_q35tp(base, k, n) {
                    return (format!("{base}_bs{bs}_r{r}"), r);
                }
                if let Some(v) = geom_name(base, k, n, None) {
                    return v;
                }
            }
        }
    }
    legacy
}

/// The shape guard for the NVFP4 repack refactor, mirroring the mlx4 repack:
/// dwordx4 needs k%32==0; k>=1024 && n>=1024 keeps it to the large mlp/expert
/// shapes (the ~62/223µs-BW-floor class) and excludes tiny router/small shapes.
/// group_size==16 is the NVFP4 invariant (each 32-elem dwordx4 chunk = 2 blocks);
/// asserted structurally here so a non-16 gs never reaches the 2-groups-per-chunk
/// kernel. A VLLM_VULKAN_MATVEC_ROWS override still forces the legacy `_r{v}`.
pub(crate) fn nvfp4_repack_shape_ok(k: usize, n: usize, gs: usize) -> bool {
    nvfp4_repack_flag()
        && matvec_rows_override().is_none()
        && gs == 16
        && k % 32 == 0
        && k >= 1024
        && n >= 1024
}

/// Shape-aware NVFP4 matvec picker. Routes to the VALU-bound repack refactor
/// (`mul_mat_vec_nvfp4repack_f32_f32`, VLLM_VULKAN_NVFP4_REPACK) for the large
/// mlp/expert shapes, else the swept-winner geometry inside the win region, else
/// byte-identical `matvec_nvfp4_variant`. This is the f32-fold (default scale)
/// path — the e4m3-resident repack is picked in `nvfp4_dispatch`. Used by the
/// Nemotron NVFP4 expert dispatch (`nem_rec_mv`, NemMvKind::Nvfp4, gs=16).
pub(crate) fn matvec_nvfp4_variant_k(k: usize, n: usize) -> (String, u32) {
    if nvfp4_repack_shape_ok(k, n, 16) {
        // bs64/r4 = the mlx4-repack-class default (the nvfp4 twin has the SAME
        // dwordx4/subgroupAdd structure); on-node micro-sweep may re-pick among
        // the geom_combos_for combos. argmax-exact + ~223µs-BW-floor target.
        return ("mul_mat_vec_nvfp4repack_f32_f32_bs64_r4".to_string(), 4);
    }
    if let Some(v) = geom_name("mul_mat_vec_nvfp4_f32_f32", k, n, matvec_rows_override()) {
        return v;
    }
    matvec_nvfp4_variant(n)
}

/// Shape-aware FP8 matvec picker. Same contract as `matvec_nvfp4_variant_k`.
pub(crate) fn matvec_fp8_variant_k(k: usize, n: usize) -> (String, u32) {
    // FP8 subgroup-reduction twin (VLLM_VULKAN_FP8_REPACK, default OFF): swaps
    // fp8_fast's LDS log2(BLOCK) barrier-tree reduction for a subgroupAdd
    // reduction. NO load/address-gen change — the s4rig offline ISA rig confirms
    // fp8 is NOT address-gen-bound (addr_gen/unpack 1.76 vs mlx4-v1's 20.11; the
    // uint-word load already amortizes address-gen), so the epilogue is the only
    // lever. Built ON fp8_fast (same arithmetic-decode hot loop, byte-identical
    // math), so it's gated on fp8_fast too; k%4==0 (word load, always true for
    // fp8 dispatches — matvec_fp8_pc asserts it); rows-override A/B still wins.
    // bs64/r4 = the mlx4/nvfp4-repack default (BLOCK<=64 = single wave64
    // subgroup, no LDS combine); an on-node micro-sweep can retune.
    if fp8_repack_flag() && fp8_fast_flag() && matvec_rows_override().is_none() && k % 4 == 0 {
        return ("mul_mat_vec_fp8repack_f32_f32_bs64_r4".to_string(), 4);
    }
    if let Some(v) = geom_name(fp8_base(), k, n, matvec_rows_override()) {
        return v;
    }
    matvec_fp8_variant(n)
}

/// Shape-aware f16 matvec picker. Geometry applies ONLY when the legacy pick is the
/// plain scalar f16 base (the `_subgroup` variant is a single-subgroup reduction with
/// no bs-siblings), matching `matvec_variant_geom`'s base guard.
pub(crate) fn matvec_f16_variant_k(k: usize, n: usize) -> (String, u32) {
    let legacy = matvec_f16_variant(n);
    if !legacy.0.ends_with("_subgroup") {
        if let Some(v) = geom_name("mul_mat_vec_f16_f32_f32", k, n, matvec_rows_override()) {
            return v;
        }
    }
    legacy
}

/// Shape-aware q8_0-resident matvec picker for the Nemotron mamba in/out_proj
/// requant route (`NemMvKind::Q8_0`). Swept-winner geometry inside the win
/// region (see `geom_pick`'s `mul_mat_vec_q8_0deq_f32_f32` entries), the
/// legacy PINNED q8_0deq base/r-tier otherwise — the weight is ALWAYS
/// resident as q8_0 on this route regardless of the process-wide
/// VLLM_VULKAN_QUANT, so unlike `matvec_variant_geom` this bypasses
/// `quant_format()`/`matvec_variant` entirely (mirrors the fp8/nvfp4 pins).
pub(crate) fn matvec_q8_0_variant_k(k: usize, n: usize) -> (String, u32) {
    if let Some(v) = geom_name("mul_mat_vec_q8_0deq_f32_f32", k, n, matvec_rows_override()) {
        return v;
    }
    matvec_variant_core(QuantFormat::Q8_0, false, false, matvec_rows_override(), n)
}

/// Rows-per-workgroup variant selector for the f32-weight matvec, PINNED to the
/// scalar f32 base regardless of the process-wide VLLM_VULKAN_QUANT format.
/// Used by the WS2 GPU MoE glue (shared expert / router weights are uploaded as
/// raw f32, never re-quantized), so the shader must always read f32 rows even
/// when the projection weights are resident in another format. Same r-tier
/// logic as `matvec_mlx4_variant`; push constants = `matvec_pc13`.
pub(crate) fn matvec_f32_variant(n: usize) -> (String, u32) {
    let base = "mul_mat_vec_f32_f32_f32";
    let mut r: u32 = if n >= 1024 { 8 } else { 1 };
    if let Some(v) = matvec_rows_override() {
        r = v;
    }
    if !matches!(r, 2 | 4 | 8) {
        return (base.to_string(), 1);
    }
    (format!("{base}_r{r}"), r)
}

/// Rows-per-workgroup variant selector for the NVFP4 matvec — identical r-tier
/// logic to `matvec_mlx4_variant`, different shader base. The push-constant layout
/// is the same 5×u32, so `matvec_mlx4_pc`/`matvec_mlx4_pc_off` are reused.
pub(crate) fn matvec_nvfp4_variant(n: usize) -> (String, u32) {
    let base = "mul_mat_vec_nvfp4_f32_f32";
    let mut r: u32 = if n >= 1024 { 8 } else { 1 };
    if let Some(v) = matvec_rows_override() {
        r = v;
    }
    if !matches!(r, 2 | 4 | 8) {
        return (base.to_string(), 1);
    }
    (format!("{base}_r{r}"), r)
}

/// Rows-per-workgroup selector for the E4M3-RESIDENT NVFP4 matvec
/// (`VLLM_VULKAN_NVFP4_E4M3_SCALES`). Identical r-tier logic + spec-constant
/// layout (BLOCK_SIZE/NUM_ROWS) to `matvec_nvfp4_variant`, different shader base
/// (`mul_mat_vec_nvfp4_e4m3_f32_f32`, which reads the raw e4m3 scale byte buffer
/// + the per-tensor global from the push constant). build.rs classes it Matvec
/// (mul_mat_vec_ prefix) so the `_r2/_r4/_r8` siblings compile automatically.
pub(crate) fn matvec_nvfp4_e4m3_variant(n: usize) -> (String, u32) {
    let base = "mul_mat_vec_nvfp4_e4m3_f32_f32";
    let mut r: u32 = if n >= 1024 { 8 } else { 1 };
    if let Some(v) = matvec_rows_override() {
        r = v;
    }
    if !matches!(r, 2 | 4 | 8) {
        return (base.to_string(), 1);
    }
    (format!("{base}_r{r}"), r)
}

/// Rows-per-workgroup selector for the EXPERT-BATCHED e4m3 NVFP4 matvec
/// (`mul_mat_vec_laguna_expb_e4m3_f32_f32`, the Laguna MoE CB-batch lever). MUST
/// mirror `matvec_nvfp4_e4m3_variant`'s r-tier logic EXACTLY (same BLOCK_SIZE=128
/// Matvec class, same NUM_ROWS pick) so the batched dispatch reduces in the same
/// order as the per-expert dispatch it replaces — that identity is what makes the
/// batched path BIT-EXACT. Same `mul_mat_vec_` prefix, so build.rs classes it
/// Matvec and pipeline.rs auto-compiles the `_r2/_r4/_r8` siblings.
pub(crate) fn laguna_expb_e4m3_variant(n: usize) -> (String, u32) {
    let base = "mul_mat_vec_laguna_expb_e4m3_f32_f32";
    let mut r: u32 = if n >= 1024 { 8 } else { 1 };
    if let Some(v) = matvec_rows_override() {
        r = v;
    }
    if !matches!(r, 2 | 4 | 8) {
        return (base.to_string(), 1);
    }
    (format!("{base}_r{r}"), r)
}

/// Push constants for the expert-batched e4m3 NVFP4 matvec:
/// `{ uint ncols(=k); uint nrows(=n); uint group_size(=16); uint x_stride; }`.
/// `x_stride` is the per-expert-slot activation stride: 0 when all experts share
/// one input (gate/up read the same `[k]` hidden), or `k` when the input is
/// concatenated per slot (down reads slot e's `[k]` at `e*k`). Per-slice weight
/// offsets + the per-tensor global come from the `meta` buffer (binding 4), NOT
/// push constants, so this block is fixed at 16 bytes regardless of top_k.
pub(crate) fn laguna_expb_e4m3_pc(k: usize, n: usize, group_size: usize, x_stride: usize) -> Vec<u8> {
    assert_eq!(k % 8, 0,
        "laguna expb e4m3 matvec requires k (in_features) divisible by 8; got k={k}");
    use std::io::Write;
    let mut v = Vec::with_capacity(4 * 4);
    for x in [k as u32, n as u32, group_size as u32, x_stride as u32] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v
}

/// Push constants for the E4M3-RESIDENT NVFP4 matvec. Same 5×u32 head as
/// `matvec_mlx4_pc_off` ({ncols, nrows, group_size, packed_off, sb_off}) plus a
/// trailing `float global` (the per-tensor `.weight_scale_2`) — 24 bytes,
/// mirroring `shaders/mul_mat_vec_nvfp4_e4m3.comp`'s push block. `sb_off` is a
/// BYTE-ELEMENT offset into the raw e4m3 scale array (0 for a plain dense
/// weight). See `mul_mat_vec_nvfp4_e4m3.comp` for the bit-exactness contract.
pub(crate) fn matvec_nvfp4_e4m3_pc(k: usize, n: usize, group_size: usize, global: f32) -> Vec<u8> {
    matvec_nvfp4_e4m3_pc_off(k, n, group_size, 0, 0, global)
}

/// `matvec_nvfp4_e4m3_pc` WITH per-slice base offsets (reserved for a future
/// per-expert MoE dispatch, mirroring `matvec_mlx4_pc_off`).
pub(crate) fn matvec_nvfp4_e4m3_pc_off(
    k: usize, n: usize, group_size: usize, packed_off: usize, sb_off: usize, global: f32,
) -> Vec<u8> {
    assert_eq!(k % 8, 0,
        "nvfp4 e4m3 matvec requires k (in_features) divisible by 8; got k={k}");
    use std::io::Write;
    let mut v = Vec::with_capacity(5 * 4 + 4);
    for x in [k as u32, n as u32, group_size as u32, packed_off as u32, sb_off as u32] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v.write_all(&global.to_le_bytes()).unwrap();
    v
}

/// Format-route an NVFP4 matvec dispatch to either the E4M3-resident kernel
/// (raw e4m3 scale bytes + `global` push constant) or the default f32-fold
/// kernel, returning `(shader, rows_per_workgroup, push_constants)`. Both
/// kernels share the SAME 4-buffer binding order (packed, scale, x, dst), so
/// every `MvKind::Nvfp4` dispatch site only differs in shader name + push
/// constants — centralized here so the flag flips one place.
pub(crate) fn nvfp4_dispatch(
    k: usize, n: usize, gs: u32, e4m3: bool, global: f32,
) -> (String, u32, Vec<u8>) {
    // VALU-bound repack refactor (VLLM_VULKAN_NVFP4_REPACK, default OFF): the
    // large mlp/expert shapes route to the dwordx4 + subgroupAdd kernel. The
    // repack REUSES the exact same 4-buffer bindings + push-constant layout as
    // its v1 (packed, scale, x, dst; 5×u32 [+ f32 global for e4m3]), so only the
    // shader NAME + rows change — the pc builders are untouched. Default OFF keeps
    // v1 byte-identical (the oracle). bs64/r4 = the mlx4-repack-class default.
    if nvfp4_repack_shape_ok(k, n, gs as usize) {
        if e4m3 {
            return ("mul_mat_vec_nvfp4_e4m3repack_f32_f32_bs64_r4".to_string(), 4,
                    matvec_nvfp4_e4m3_pc(k, n, gs as usize, global));
        }
        return ("mul_mat_vec_nvfp4repack_f32_f32_bs64_r4".to_string(), 4,
                matvec_mlx4_pc(k, n, gs as usize));
    }
    if e4m3 {
        let (shader, r) = matvec_nvfp4_e4m3_variant(n);
        (shader, r, matvec_nvfp4_e4m3_pc(k, n, gs as usize, global))
    } else {
        let (shader, r) = matvec_nvfp4_variant(n);
        (shader, r, matvec_mlx4_pc(k, n, gs as usize))
    }
}

/// Fold the NVFP4 two-level scale into the single per-(row,group) f32 the GPU
/// shader reads: `scale[r,g] = e4m3(weight_scale[r,g]) * weight_scale_2`. Input
/// `wscale` is the raw e4m3 bytes `[out*groups]`; returns f32 `[out*groups]`.
/// This is what makes the shader bias-free and e4m3/global-free — all the
/// two-level arithmetic happens once here, matching `model::dequantize_nvfp4`.
pub(crate) fn nvfp4_fold_scales(wscale: &[u8], global: f32) -> Vec<f32> {
    wscale.iter().map(|&b| model::e4m3_to_f32(b) * global).collect()
}

/// Rows-per-workgroup selector for a genuinely-f16-in-memory weight, PINNED to
/// the f16 base regardless of the process-wide VLLM_VULKAN_QUANT (mirrors
/// `matvec_f32_variant`'s pin, and the fp8/nvfp4 pins below). The Nemotron
/// resident attn q/k/v/o, latent fc1/fc2 and lm_head are uploaded as f16
/// (`NemQuant::F16`) and MUST dispatch the f16 matvec even when
/// VLLM_VULKAN_QUANT=q8_0 is exported (the qwen production default). Routing
/// them through the quant-format-driven `matvec_variant` would feed f16 bytes to
/// the q8_0/q4 dequant shader and produce garbage logits. Honors the subgroup
/// flag + rows override; push constants = `matvec_pc13`, same as the f16 arm of
/// `matvec_variant`.
pub(crate) fn matvec_f16_variant(n: usize) -> (String, u32) {
    matvec_variant_core(QuantFormat::F16, use_subgroup_flag(), true, matvec_rows_override(), n)
}

/// Rows-per-workgroup variant selector for the FP8 matvec (same r-tier as nvfp4).
pub(crate) fn matvec_fp8_variant(n: usize) -> (String, u32) {
    let base = fp8_base();
    let mut r: u32 = if n >= 1024 { 8 } else { 1 };
    if let Some(v) = matvec_rows_override() { r = v; }
    if !matches!(r, 2 | 4 | 8) { return (base.to_string(), 1); }
    (format!("{base}_r{r}"), r)
}

/// Push constants for the FP8 matvec: { k, n, scale_per_row(0|1), packed_off=0, sb_off=0 }.
/// Absolute-byte addressing needs each row word-aligned -> k%4==0 (asserted; all
/// attn in-dims and their TP shards satisfy it).
pub(crate) fn matvec_fp8_pc(k: usize, n: usize, per_row: bool) -> Vec<u8> {
    use std::io::Write;
    assert_eq!(k % 4, 0, "fp8 matvec requires k (in_features) divisible by 4; got k={k}");
    let mut v = Vec::with_capacity(5 * 4);
    for x in [k as u32, n as u32, per_row as u32, 0u32, 0u32] { v.write_all(&x.to_le_bytes()).unwrap(); }
    v
}

/// Push constants for the `mul_mm` tiled GEMM. The shader computes
/// D[m,nn] = sum_k A[m,k]*B[nn,k] and writes element (m,nn) at
/// data_d[nn*stride_d + m]. We bind A = the f16 weight and B = the f32 input,
/// so M = n (weight rows / out cols), N = t (input rows / tokens), K = k.
/// With stride_d = n, element (col, token) lands at token*n + col — i.e.
/// out is row-major [t, n], exactly cpu_matmul's layout.
///
/// CRITICAL: this is the **non-MUL_MAT_ID** parameter struct, which has only
/// 16 u32 — NOT 20. The `#ifdef MUL_MAT_ID` branch in mul_mm.comp swaps in
/// {nei0,nei1,nbi1,ne11}; the `#else` branch (what we compile) has instead
/// {base_work_group_z,num_batches,k_split,ne02,ne12,broadcast2,broadcast3}.
/// An earlier version wrote 4 phantom nei* zeros before base_work_group_z,
/// which shifted k_split into a zero slot → k_split=0 → end_k=min(K,0)=0 →
/// the K-accumulation loop never ran → all-zero output. Keep this at 16 u32.
pub(crate) fn gemm_pc(t: usize, n: usize, k: usize) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(16 * 4);
    for x in [
        n as u32, t as u32, k as u32,        // M=n (A=weight rows), N=t (B=input rows), K
        k as u32, k as u32, n as u32,        // stride_a (weight row=k), stride_b (input row=k), stride_d (=n)
        (n * k) as u32, (t * k) as u32, (t * n) as u32, // batch_stride a(weight),b(input),d(out)
        0u32,                                // base_work_group_z
        1u32,                                // num_batches
        k as u32,                            // k_split = K (no split-k): inner loop
                                             // runs block in [0, min(K, k_split)) = [0, K).
        1u32, 1u32,                          // ne02, ne12
        1u32, 1u32,                          // broadcast2, broadcast3
    ] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v
}

/// Push constants for the Phase A quant `mul_mm` GEMM (DATA_A_Q8_0 or
/// DATA_A_MLX4). Identical 16-u32 prefix to `gemm_pc` (same non-MUL_MAT_ID
/// layout/derivation — see that doc-comment), PLUS 3 more u32 that only the
/// `#if defined(DATA_A_MLX4)` branch of `mul_mm.comp`'s push-constant struct
/// declares: `mlx4_group_size`, `mlx4_packed_off`, `mlx4_sb_off`. Q8_0 has no
/// aux buffers (its scale is inline per-block), so `gemm_pc` alone suffices
/// for that variant — only mlx4 needs this 19-u32 form. `packed_off`/`sb_off`
/// are 0 for a plain dense (non-MoE) weight, mirroring `matvec_mlx4_pc_off`'s
/// per-tensor-base convention.
pub(crate) fn gemm_pc_mlx4(t: usize, n: usize, k: usize, group_size: usize, packed_off: usize, sb_off: usize) -> Vec<u8> {
    let mut v = gemm_pc(t, n, k);
    use std::io::Write;
    for x in [group_size as u32, packed_off as u32, sb_off as u32] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v
}

/// Push constants for the Phase B GROUPED-expert quant GEMM
/// (`matmul_mlx4_id_f32_fp32`, MUL_MAT_ID + DATA_A_MLX4). The MUL_MAT_ID arm of
/// `mul_mm.comp`'s push-constant struct is DIFFERENT from the dense (`gemm_pc`/
/// `gemm_pc_mlx4`) tail: the 9-u32 common prefix
/// `{M,N,K, stride_a,stride_b,stride_d, batch_stride_a,batch_stride_b,batch_stride_d}`
/// is followed by `{nei0,nei1,nbi1,ne11}` then the mlx4 sub-arm
/// `{mlx4_group_size, mlx4_pack_stride, mlx4_sb_stride}` = EXACTLY 16 u32. Do
/// NOT reuse `gemm_pc_mlx4` (19 u32, non-id middle {base_work_group_z,
/// num_batches,k_split,ne02,ne12,broadcast2,broadcast3}) — the middle differs.
///
/// The head IGNORES `batch_stride_a` (it recomputes the A base from
/// `gl_WorkGroupID.z * mlx4_pack_stride`, see mul_mm_funcs.glsl's DATA_A_MLX4
/// branch), so it is written 0. `N = nei0*nei1 = top_k*t` is not load-bearing
/// in the MUL_MAT_ID store (the per-expert `_ne1` from the scan bounds the
/// columns) but is set for completeness. Contract (decoded from the shader):
///   - `nei1 = t` is the TOKEN dim (row_idx.y), `nei0 = top_k` the expert-slot
///     dim (row_idx.x); `nbi1 = top_k` is the ids row stride.
///   - B addressing = `row_idx.y*batch_stride_b + (row_idx.x % ne11)*stride_b`:
///     `ne11 = 1` broadcasts x across all top_k slots of a token (gate/up);
///     `ne11 = top_k` reads a per-slot B row (down).
///   - D output = `row_idx.y*batch_stride_d + row_idx.x*stride_d + m`
///     → already un-permuted [token][slot][M].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_pc_mlx4_id(
    m: usize, k: usize, top_k: usize, t: usize, ne11: usize,
    stride_b: usize, batch_stride_b: usize, stride_d: usize, batch_stride_d: usize,
    group_size: usize, pack_stride: usize, sb_stride: usize,
) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(16 * 4);
    for x in [
        m as u32, (top_k * t) as u32, k as u32,   // M=out_features, N=nei0*nei1, K
        k as u32, stride_b as u32, stride_d as u32, // stride_a (weight row=k), stride_b, stride_d
        0u32, batch_stride_b as u32, batch_stride_d as u32, // batch_stride a(ignored),b,d
        top_k as u32, t as u32, top_k as u32, ne11 as u32,  // nei0, nei1, nbi1, ne11
        group_size as u32, pack_stride as u32, sb_stride as u32, // mlx4 group_size, pack_stride, sb_stride
    ] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v
}

/// Push constants for the EPILOGUE-FUSED gate+up grouped GEMM
/// (`matmul_mlx4_id_gateup_silu_f32_fp32`, plan-epilogue-fused-moe-gemm.md
/// §2.1). Identical 16-u32 prefix to `gemm_pc_mlx4_id` (same field order/
/// derivation — see that doc-comment) PLUS 2 more u32 the shader's
/// `MLX4_ID_GATEUP` push-constant branch declares: `up_pack_stride`,
/// `up_sb_stride` — the UP weight's own per-expert strides (the gate's
/// `mlx4_group_size`/`mlx4_pack_stride`/`mlx4_sb_stride` at indices 13/14/15
/// belong to gate; `group_size` is shared with up, so no separate
/// `up_group_size` field). Total 18 u32 — do NOT reuse `gemm_pc_mlx4_id`
/// (16 u32) for this variant, the shader's push-constant struct is 2 words
/// longer under `#if defined(MLX4_ID_GATEUP)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_pc_mlx4_id_gateup(
    m: usize, k: usize, top_k: usize, t: usize, ne11: usize,
    stride_b: usize, batch_stride_b: usize, stride_d: usize, batch_stride_d: usize,
    group_size: usize, gate_pack_stride: usize, gate_sb_stride: usize,
    up_pack_stride: usize, up_sb_stride: usize,
) -> Vec<u8> {
    let mut v = gemm_pc_mlx4_id(
        m, k, top_k, t, ne11, stride_b, batch_stride_b, stride_d, batch_stride_d,
        group_size, gate_pack_stride, gate_sb_stride,
    );
    use std::io::Write;
    for x in [up_pack_stride as u32, up_sb_stride as u32] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v
}

/// Shape-keyed quant-GEMM picker: returns `(pipeline_name, BM, BN)` for the
/// wg-size math at the call site, mirroring `gemm_variant_k` exactly but for
/// the Phase A quant `mul_mm` variants (`base` = "matmul_mlx4_f32_fp32" or
/// "matmul_q8_0_f32_fp32"). Inside the swept win region (`gemm_pick`) this is
/// the geometry sibling compiled by `compile_mul_mm_quant_geom`; otherwise it
/// is `base` itself unsuffixed — which IS the BM=BN=64 default geometry
/// `compile_mul_mm_quant` compiles under the bare name (same
/// no-missing-pipeline-by-construction pattern as `gemm_variant_k`'s LEGACY
/// fallback). Does NOT gate on `gemm_f16aligned_flag` — the quant path has
/// its own independent kill switch (`gemm_quant_flag`, checked by the caller
/// before this is ever reached).
pub(crate) fn gemm_variant_quant_k(base: &str, k: usize, n: usize) -> (String, u32, u32) {
    const DEFAULT_BM: u32 = 64;
    const DEFAULT_BN: u32 = 64;
    if let Some(g) = gemm_pick(k, n) {
        let (bm, bn, _warp, _tm, _tn) = g;
        return (gemm_geom_name(base, g), bm, bn);
    }
    (base.to_string(), DEFAULT_BM, DEFAULT_BN)
}

/// Phase 2 sparse-BN picker (plan-epilogue-fused-moe-gemm.md §3): returns
/// `(pipeline_name, BM, BN)` for a MUL_MAT_ID grouped-expert quant GEMM
/// (`base` = "matmul_mlx4_id_f32_fp32" or
/// "matmul_mlx4_id_gateup_silu_f32_fp32"), keyed on `avg_tokens_per_expert`
/// (NOT the (k,n) shape `gemm_variant_quant_k` keys on — this axis is the
/// per-expert N-dim occupancy, orthogonal to the weight shape). The dense
/// BM=BN=64 default over-provisions BN at low T (≤64 = ~25% occupancy at
/// the T=512 example the plan works through) — a smaller BN sibling gives
/// the token (N) dim full column occupancy and more resident workgroups
/// (better latency hiding on this memory-bound kernel; see §3/§4). Compiled
/// by `compile_mul_mm_quant_id_bn` under `<base>_bn16`/`<base>_bn32`; falls
/// back to the plain `base` (BM=BN=64) once tokens/expert clears BN=32.
/// Gated by the SAME `VLLM_VULKAN_MOE_GEMM_FUSED` kill switch as the rest of
/// this plan at the call site — not cluster-validated as of this change.
pub(crate) fn gemm_variant_quant_id_bn(base: &str, avg_tokens_per_expert: usize) -> (String, u32, u32) {
    const BM: u32 = 64;
    if avg_tokens_per_expert <= 16 {
        (format!("{base}_bn16"), BM, 16)
    } else if avg_tokens_per_expert <= 32 {
        (format!("{base}_bn32"), BM, 32)
    } else {
        (base.to_string(), BM, 64)
    }
}

/// Push constants for `paged_attn_decode_f32` (single contiguous block): seq_len,
/// num_q_heads, num_kv_heads, head_size, block_size(=seq_len), layer_base(=0),
/// plane_elements_per_block(=seq*num_kv*hd), block_elements(=2*plane), scale(f32),
/// window_start(=first token_idx to attend; 0 = full causal, >0 = sliding window),
/// ring_capacity(=0 for the legacy absolute/block-table plane — byte-identical;
/// >0 for a window-sized RING plane where absolute pos `t` lives at slot
/// `t % ring_capacity`, used by per-layer-sized sliding-window KV. In ring mode
/// `plane` must be `ring_capacity*num_kv*head_size`).
pub(crate) fn sdpa_pc(seq_len: usize, num_q: usize, num_kv: usize, head_size: usize, block_size: usize, plane: usize, scale: f32, window_start: usize, ring_capacity: usize) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(11 * 4);
    for x in [seq_len as u32, num_q as u32, num_kv as u32, head_size as u32,
              block_size as u32, 0u32, plane as u32, (2 * plane) as u32] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v.write_all(&scale.to_le_bytes()).unwrap();
    v.write_all(&(window_start as u32).to_le_bytes()).unwrap();
    v.write_all(&(ring_capacity as u32).to_le_bytes()).unwrap();
    v
}

/// Push constants for `flash_attn_*` (flash_attn_base.glsl `parameter`, ~30 u32).
///
/// Single batch (ne3=1), gqa_ratio param left at 1 so each head writes its own
/// contiguous output. Strides (nb*) are in ELEMENTS (ggml convention, the shader
/// divides by the vec width itself). Layouts:
///   Q [head][query][head_dim]: nb01=head_dim (per query), nb02=N*head_dim (per head)
///   K [pos][num_kv][head_dim]: nb11=num_kv*head_dim (per pos), nb12=head_dim (per head)
///   V same as K (nb21/nb22). ne1=N (queries), ne2=num_q (heads).
/// GQA broadcast via neq2=num_q, nek2=nev2=num_kv -> rk2=gqa_ratio, ik2=h/gqa_ratio.
pub(crate) fn flash_pc(n_queries: usize, kv_len: usize, num_q: usize, num_kv: usize,
            head_dim: usize, scale: f32) -> Vec<u8> {
    use std::io::Write;
    let n = n_queries as u32;
    let hd = head_dim as u32;
    let mut v = Vec::with_capacity(32 * 4);
    let u = |v: &mut Vec<u8>, x: u32| v.write_all(&x.to_le_bytes()).unwrap();
    u(&mut v, n);                       // N (#queries)
    u(&mut v, kv_len as u32);           // KV (#keys)
    // ne1 is the OUTPUT per-query (per-position) stride in the store:
    //   o element = iq2*HSV + n*ne1*HSV + dim. We want output [pos n][head h][hd],
    //   so consecutive positions are num_q*hd apart -> ne1 = num_q (NOT N).
    u(&mut v, num_q as u32);            // ne1 (output query stride = num_q heads)
    u(&mut v, num_q as u32);            // ne2 (= num_q heads)
    u(&mut v, 1);                       // ne3 (batch)
    u(&mut v, num_q as u32);            // neq2
    u(&mut v, 1);                       // neq3
    u(&mut v, num_kv as u32);           // nek2
    u(&mut v, 1);                       // nek3
    u(&mut v, num_kv as u32);           // nev2
    u(&mut v, 1);                       // nev3
    u(&mut v, n);                       // nem1 (mask rows = N)
    u(&mut v, 1);                       // nem2
    u(&mut v, 1);                       // nem3
    // CRITICAL stride units: the shader uses nb01/nb11/nb21 (per-row strides)
    // DIRECTLY as element strides, but the per-HEAD/BATCH strides go through an
    // extra divide inside q_offset/k_offset/v_offset: Q's head term is
    // (iq2*nb02)/4 and is then *4 again in the vec4 index → net element stride is
    // nb02/4, so nb02 must carry a ×4. K/V head term is (ik2*nb12)/2 → net nb12/2,
    // so nb12/nb22 carry a ×2. (ggml byte-stride convention surfacing here.)
    u(&mut v, hd);                          // nb01 (Q per-query stride, elements)
    u(&mut v, 4 * n * hd);                  // nb02 (Q per-head: ×4, net N*hd)
    u(&mut v, 4 * num_q as u32 * n * hd);   // nb03 (Q per-batch: ×4)
    u(&mut v, num_kv as u32 * hd);          // nb11 (K per-pos stride, elements)
    u(&mut v, 2 * hd);                      // nb12 (K per-head: ×2, net hd)
    u(&mut v, 2 * kv_len as u32 * num_kv as u32 * hd); // nb13 (K per-batch: ×2)
    u(&mut v, num_kv as u32 * hd);          // nb21 (V per-pos stride, elements)
    u(&mut v, 2 * hd);                      // nb22 (V per-head: ×2, net hd)
    u(&mut v, 2 * kv_len as u32 * num_kv as u32 * hd); // nb23 (V per-batch: ×2)
    v.write_all(&scale.to_le_bytes()).unwrap(); // scale (f32)
    v.write_all(&0f32.to_le_bytes()).unwrap();  // max_bias (no ALiBi)
    v.write_all(&0f32.to_le_bytes()).unwrap();  // logit_softcap
    u(&mut v, 0);                       // mask_n_head_log2 (no sink/ALiBi)
    v.write_all(&0f32.to_le_bytes()).unwrap();  // m0
    v.write_all(&0f32.to_le_bytes()).unwrap();  // m1
    u(&mut v, 1);                       // gqa_ratio (=1: per-head non-interleaved store)
    u(&mut v, kv_len as u32);           // split_kv (= KV, single split)
    u(&mut v, 1);                       // k_num (no split-k)
    v
}

/// Push constants for a unary elementwise shader (silu/gelu, `generic_head`):
/// KX = element count, KY = 1, then 4 unused param slots.
pub(crate) fn ew_unary_pc(kx: u32) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(6 * 4);
    v.write_all(&kx.to_le_bytes()).unwrap();
    v.write_all(&1u32.to_le_bytes()).unwrap();
    for _ in 0..4 { v.write_all(&0u32.to_le_bytes()).unwrap(); }
    v
}

/// Push constants for `mul_f32_f32_f32` (generic_binary_head) over a flat [n]
/// tensor: ne + ne00-03/nb00-03 (src0), then src1, then dst, then misalign+params.
/// nb (strides) are in ELEMENTS (ggml convention).
pub(crate) fn ew_mul_pc(n: u32) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(29 * 4);
    for &x in &[n, n, 1u32, 1, 1, 1u32, n, n, n] { v.write_all(&x.to_le_bytes()).unwrap(); }
    for &x in &[n, 1u32, 1, 1, 1u32, n, n, n] { v.write_all(&x.to_le_bytes()).unwrap(); }
    for &x in &[n, 1u32, 1, 1, 1u32, n, n, n] { v.write_all(&x.to_le_bytes()).unwrap(); }
    for &x in &[0u32, 0u32, 0u32, 0u32] { v.write_all(&x.to_le_bytes()).unwrap(); }
    v
}

/// Push constants for `rms_norm_f32_mul` (generic_binary_head `parameter`, 29 u32).
/// One workgroup per row (dispatch (n_rows,1,1)); ncols=ne00=n_cols. The weight
/// (binding 1) is broadcast across rows: ne10=n_cols, nb10=1, nb11=0, so every
/// row reads weight[col]. d_offset = row*ncols → row-major [n_rows,n_cols] out.
/// param1 = eps. Matches model::cpu_rms_norm exactly.
pub(crate) fn rmsnorm_pc(n_cols: usize, eps: f32) -> Vec<u8> {
    use std::io::Write;
    let c = n_cols as u32;
    let mut v = Vec::with_capacity(29 * 4);
    let u = |v: &mut Vec<u8>, x: u32| v.write_all(&x.to_le_bytes()).unwrap();
    u(&mut v, c);                                   // ne (unused here)
    // src0 (input): ne00..ne03, nb00..nb03. row stride nb01 = n_cols.
    u(&mut v, c); u(&mut v, 1); u(&mut v, 1); u(&mut v, 1);   // ne00,ne01,ne02,ne03
    u(&mut v, 1); u(&mut v, c); u(&mut v, c); u(&mut v, c);   // nb00,nb01,nb02,nb03
    // src1 (weight): broadcast — ne10=n_cols, nb10=1, nb11=0 (same weight/row).
    u(&mut v, c); u(&mut v, 1); u(&mut v, 1); u(&mut v, 1);   // ne10,ne11,ne12,ne13
    u(&mut v, 1); u(&mut v, 0); u(&mut v, 0); u(&mut v, 0);   // nb10,nb11,nb12,nb13
    // dst: unused by the shader's d_offset formula, keep sane.
    u(&mut v, c); u(&mut v, 1); u(&mut v, 1); u(&mut v, 1);   // ne20..ne23
    u(&mut v, 1); u(&mut v, c); u(&mut v, c); u(&mut v, c);   // nb20..nb23
    u(&mut v, 0);                                   // misalign_offsets
    v.write_all(&eps.to_le_bytes()).unwrap();       // param1 = eps (f32)
    v.write_all(&0f32.to_le_bytes()).unwrap();      // param2
    u(&mut v, 0);                                   // param3 (int)
    v
}

/// Push constants for `rope_neox_f32_f32` (rope_params struct). NeoX rope over
/// `num_heads` rows of `head_dim`, rotating `rotary_dim` dims with a frequency
/// basis of `freq_dim` (decoupled from `rotary_dim` for gemma proportional
/// RoPE: the pair-partner offset and frequency denominator are set by
/// `freq_dim`, while `rotary_dim` only bounds how many pairs actually rotate).
/// theta_scale = theta^(-2/freq_dim) so theta_base(j) = pos * theta_scale^j =
/// pos * theta^(-2j/freq_dim), matching model::cpu_rope_with_basis's NeoX
/// freqs. Full-rotary callers pass freq_dim == rotary_dim == head_dim, making
/// this bit-exact with the pre-decoupling behaviour. nb01=nb11=head_dim
/// (per-head row stride, elements). Dispatch covers i0 in [0,head_dim).
pub(crate) fn rope_neox_pc(num_heads: usize, head_dim: usize, rotary_dim: usize, freq_dim: usize, theta: f32) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(40 * 4);
    let u = |v: &mut Vec<u8>, x: u32| v.write_all(&x.to_le_bytes()).unwrap();
    let f = |v: &mut Vec<u8>, x: f32| v.write_all(&x.to_le_bytes()).unwrap();
    let theta_scale = theta.powf(-2.0f32 / freq_dim as f32);
    u(&mut v, 2);                       // rope_mode = GGML_ROPE_TYPE_NEOX
    u(&mut v, num_heads as u32);        // nrows (row bound)
    u(&mut v, freq_dim as u32);         // n_dims (shader partner offset = n_dims/2 = freq_dim/2)
    f(&mut v, 1.0);                     // freq_scale
    f(&mut v, theta);                   // freq_base (unused; theta_scale drives it)
    f(&mut v, 0.0);                     // ext_factor (no YaRN)
    f(&mut v, 1.0);                     // attn_factor
    f(&mut v, 0.0); f(&mut v, 0.0);     // corr_dims[2]
    f(&mut v, theta_scale);             // theta_scale
    u(&mut v, 0);                       // has_ff = 0
    u(&mut v, 0); u(&mut v, 0); u(&mut v, 0); u(&mut v, 0); // sections[4]
    u(&mut v, 0);                       // is_imrope
    u(&mut v, 0);                       // is_back
    u(&mut v, 0);                       // set_rows_stride
    u(&mut v, rotary_dim as u32);       // ne00 (rotation bound: i0 < ne00; also
                                        // kills invocations beyond the rotated
                                        // span, suppressing the shader's
                                        // pass-through write to avoid an
                                        // in-place race)
    u(&mut v, num_heads as u32);        // ne01 (#rows = heads)
    u(&mut v, 1);                       // ne02
    u(&mut v, head_dim as u32);         // nb01 (input per-head stride, elements)
    u(&mut v, 0);                       // nb02
    u(&mut v, 0);                       // nb03
    u(&mut v, head_dim as u32);         // nb11 (output per-head stride, elements)
    u(&mut v, 0);                       // nb12
    u(&mut v, 0);                       // nb13
    u(&mut v, 0);                       // yarn_direct = 0 (plain/NTK path)
    v
}

/// Push constants for `rope_neox_f32_f32` in Laguna's full-attn **YaRN direct**
/// mode (piece 2 of the GPU-resident span fold). The `rope_data_ff` buffer must
/// hold the precomputed transformers-YaRN inv_freq table (length
/// `rotary_dim/2`, i.e. [`crate::laguna::compute_yarn_inv_freq`] /
/// `LagunaConfig::yarn_inv_freq`); the shader forms `angle = pos*inv_freq[j]`
/// (single multiply, matching `laguna::cpu_rope_yarn`) and scales cos/sin by
/// `mscale` (== `full_attention_factor`). NeoX partial pairing: `n_dims =
/// rotary_dim` (partner offset `rotary_dim/2`), dispatch bound `ne00 =
/// rotary_dim` so only the low `rotary_dim` dims rotate and the high
/// `[rotary_dim..head_dim)` dims pass through in-place (seed output = input).
/// Row/head strides identical to `rope_neox_pc`.
pub(crate) fn rope_neox_yarn_pc(num_heads: usize, head_dim: usize, rotary_dim: usize, mscale: f32) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(40 * 4);
    let u = |v: &mut Vec<u8>, x: u32| v.write_all(&x.to_le_bytes()).unwrap();
    let f = |v: &mut Vec<u8>, x: f32| v.write_all(&x.to_le_bytes()).unwrap();
    u(&mut v, 2);                       // rope_mode = GGML_ROPE_TYPE_NEOX
    u(&mut v, num_heads as u32);        // nrows (row bound)
    u(&mut v, rotary_dim as u32);       // n_dims (partner offset = rotary_dim/2)
    f(&mut v, 1.0);                     // freq_scale (unused in direct path)
    f(&mut v, 0.0);                     // freq_base (unused)
    f(&mut v, 0.0);                     // ext_factor (unused in direct path)
    f(&mut v, mscale);                  // attn_factor = full_attention_factor mscale
    f(&mut v, 0.0); f(&mut v, 0.0);     // corr_dims[2] (unused)
    f(&mut v, 0.0);                     // theta_scale (unused in direct path)
    u(&mut v, 1);                       // has_ff = 1 (rope_data_ff = inv_freq table)
    u(&mut v, 0); u(&mut v, 0); u(&mut v, 0); u(&mut v, 0); // sections[4]
    u(&mut v, 0);                       // is_imrope
    u(&mut v, 0);                       // is_back
    u(&mut v, 0);                       // set_rows_stride
    u(&mut v, rotary_dim as u32);       // ne00 (rotation bound: i0 < ne00)
    u(&mut v, num_heads as u32);        // ne01 (#rows = heads)
    u(&mut v, 1);                       // ne02
    u(&mut v, head_dim as u32);         // nb01 (input per-head stride, elements)
    u(&mut v, 0);                       // nb02
    u(&mut v, 0);                       // nb03
    u(&mut v, head_dim as u32);         // nb11 (output per-head stride, elements)
    u(&mut v, 0);                       // nb12
    u(&mut v, 0);                       // nb13
    u(&mut v, 1);                       // yarn_direct = 1
    v
}

/// Push constants for the `swiglu_f32` / `geglu_f32` GLU kernels in SPLIT mode
/// (mode=2): `data_d[i] = op(data_a[i], data_b[i])` for a single token of width
/// `n` (binding0 = gate, binding1 = up, binding2 = out). The kernel's push
/// struct is `{ uint N,ne00,ne20,mode; float alpha,limit; uint nb01,nb02,nb03,
/// ne01,ne02,nb11,nb12,nb13,ne11,ne12; }` — for one token (row 0) every row
/// index i1/i2/i3 is 0, so the row strides are inert; we only need N=ne20=n,
/// mode=2, and the row-count divisors (ne01/ne02/ne11/ne12) nonzero (=1) so the
/// index math doesn't divide by zero. Dispatch ((n+511)/512, 1, 1).
pub(crate) fn glu_split_pc(n: usize) -> Vec<u8> {
    use std::io::Write;
    let n = n as u32;
    let mut v = Vec::with_capacity(16 * 4);
    v.write_all(&n.to_le_bytes()).unwrap();          // N
    v.write_all(&0u32.to_le_bytes()).unwrap();       // ne00 (unused in split mode)
    v.write_all(&n.to_le_bytes()).unwrap();          // ne20 (cols/row -> row=i/ne20)
    v.write_all(&2u32.to_le_bytes()).unwrap();       // mode = 2 (Split: op(a,b))
    v.write_all(&0.0f32.to_le_bytes()).unwrap();     // alpha (unused)
    v.write_all(&0.0f32.to_le_bytes()).unwrap();     // limit (unused)
    v.write_all(&n.to_le_bytes()).unwrap();          // nb01 (row stride a; *0)
    v.write_all(&n.to_le_bytes()).unwrap();          // nb02 (*0)
    v.write_all(&n.to_le_bytes()).unwrap();          // nb03 (*0)
    v.write_all(&1u32.to_le_bytes()).unwrap();       // ne01 (rows/plane divisor)
    v.write_all(&1u32.to_le_bytes()).unwrap();       // ne02
    v.write_all(&n.to_le_bytes()).unwrap();          // nb11 (row stride d; *0)
    v.write_all(&n.to_le_bytes()).unwrap();          // nb12 (*0)
    v.write_all(&n.to_le_bytes()).unwrap();          // nb13 (*0)
    v.write_all(&1u32.to_le_bytes()).unwrap();       // ne11 (dst rows/plane divisor)
    v.write_all(&1u32.to_le_bytes()).unwrap();       // ne12
    v
}

/// Push constants for `dsv4_swiglu_clamp` (DSV4 routed-expert SwiGLU with the
/// `swiglu_limit` clamp): identical `glu_head` layout to [`glu_split_pc`] (Split
/// mode 2, separate gate/up buffers) but with `alpha=1.0` and `limit` set to the
/// config `swiglu_limit`. Pass `limit = f32::INFINITY` for the shared expert
/// (min/clamp become identity => plain SwiGLU, matching "shared has NO
/// swiglu_limit"). `n` = intermediate width (elements this dispatch writes).
pub(crate) fn dsv4_swiglu_clamp_pc(n: usize, limit: f32) -> Vec<u8> {
    use std::io::Write;
    let n = n as u32;
    let mut v = Vec::with_capacity(16 * 4);
    v.write_all(&n.to_le_bytes()).unwrap();          // N
    v.write_all(&0u32.to_le_bytes()).unwrap();       // ne00 (unused in split mode)
    v.write_all(&n.to_le_bytes()).unwrap();          // ne20 (cols/row -> row=i/ne20)
    v.write_all(&2u32.to_le_bytes()).unwrap();       // mode = 2 (Split: op(a,b))
    v.write_all(&1.0f32.to_le_bytes()).unwrap();     // alpha = 1 (plain silu sigmoid)
    v.write_all(&limit.to_le_bytes()).unwrap();      // limit = swiglu_limit (or +inf)
    v.write_all(&n.to_le_bytes()).unwrap();          // nb01
    v.write_all(&n.to_le_bytes()).unwrap();          // nb02
    v.write_all(&n.to_le_bytes()).unwrap();          // nb03
    v.write_all(&1u32.to_le_bytes()).unwrap();       // ne01
    v.write_all(&1u32.to_le_bytes()).unwrap();       // ne02
    v.write_all(&n.to_le_bytes()).unwrap();          // nb11
    v.write_all(&n.to_le_bytes()).unwrap();          // nb12
    v.write_all(&n.to_le_bytes()).unwrap();          // nb13
    v.write_all(&1u32.to_le_bytes()).unwrap();       // ne11
    v.write_all(&1u32.to_le_bytes()).unwrap();       // ne12
    v
}

/// Push constants for `dsv4_hc_residual_mix`: `{ uint seq; uint hc; uint hidden; }`
/// — the manifold hyper-connection residual mix over the `[seq, hc, hidden]`
/// stream stack (one invocation per output element).
pub(crate) fn dsv4_hc_residual_mix_pc(seq: usize, hc: usize, hidden: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(12);
    v.extend_from_slice(&(seq as u32).to_le_bytes());
    v.extend_from_slice(&(hc as u32).to_le_bytes());
    v.extend_from_slice(&(hidden as u32).to_le_bytes());
    v
}

/// Push constants for `dsv4_mla_softmax` (resident single-token MLA eager
/// attention): `{ uint num_heads; uint head_dim; uint t1; uint t_comp;
/// uint sliding_window; uint rope_dim; float scaling; }`. `t1` = sliding KV rows
/// (== pos+1), `t_comp` = concatenated compressed KV rows (0 for sliding layers),
/// `scaling` = `(head_dim)^-0.5`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsv4_mla_softmax_pc(
    num_heads: usize, head_dim: usize, t1: usize, t_comp: usize,
    sliding_window: usize, rope_dim: usize, scaling: f32,
) -> Vec<u8> {
    let mut v = Vec::with_capacity(28);
    v.extend_from_slice(&(num_heads as u32).to_le_bytes());
    v.extend_from_slice(&(head_dim as u32).to_le_bytes());
    v.extend_from_slice(&(t1 as u32).to_le_bytes());
    v.extend_from_slice(&(t_comp as u32).to_le_bytes());
    v.extend_from_slice(&(sliding_window as u32).to_le_bytes());
    v.extend_from_slice(&(rope_dim as u32).to_le_bytes());
    v.extend_from_slice(&scaling.to_le_bytes());
    v
}

/// Push constants for `laguna_softplus_gate`: `{ uint nq; uint hd; }` — the
/// per-head query count and `head_dim` for the broadcast per-head softplus attn
/// gate (binding0 = g[nq], binding1 = attn[nq*hd] scaled in place).
pub(crate) fn laguna_softplus_gate_pc(nq: usize, hd: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&(nq as u32).to_le_bytes());
    v.extend_from_slice(&(hd as u32).to_le_bytes());
    v
}

/// Push constants for `laguna_gpu_sdpa` (GPU decode-SDPA over the resident K/V
/// planes): { uint seq_len; uint num_q_heads; uint num_kv_heads; uint head_size;
/// float scale; uint window_start; uint ring_capacity; }. `seq_len` is the causal
/// read bound for THIS query (== plane.seq_len after the token's append),
/// `window_start` the sliding clamp (0 for full-attn, `seq_len - min(seq_len,
/// sliding_window)` for sliding). `ring_capacity` = 0 for the full absolute plane
/// (byte-identical legacy path) or `window` for a sliding-layer RING plane where
/// absolute pos `t` lives at slot `t % ring_capacity` (Phase-0 per-layer KV
/// sizing). Dispatch (num_q_heads, 1, 1), 64 threads/workgroup.
pub(crate) fn laguna_gpu_sdpa_pc(
    seq_len: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    scale: f32,
    window_start: usize,
    ring_capacity: usize,
) -> Vec<u8> {
    let mut v = Vec::with_capacity(7 * 4);
    v.extend_from_slice(&(seq_len as u32).to_le_bytes());
    v.extend_from_slice(&(nq as u32).to_le_bytes());
    v.extend_from_slice(&(nkv as u32).to_le_bytes());
    v.extend_from_slice(&(hd as u32).to_le_bytes());
    v.extend_from_slice(&scale.to_le_bytes());
    v.extend_from_slice(&(window_start as u32).to_le_bytes());
    v.extend_from_slice(&(ring_capacity as u32).to_le_bytes());
    v
}

/// Push constants for `q35_dn_conv_step` (WS1b GatedDeltaNet conv1d decode
/// step): { uint conv_dim; uint kern; }.
pub(crate) fn q35_conv_pc(conv_dim: usize, kern: usize) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(2 * 4);
    v.write_all(&(conv_dim as u32).to_le_bytes()).unwrap();
    v.write_all(&(kern as u32).to_le_bytes()).unwrap();
    v
}

/// Push constants for `q35_gdn_qknorm` (WS1b per-head Q/K RMSNorm + inv-scale):
/// { uint nk; uint kd; uint key_dim; float eps; float inv; }. Dispatch
/// (2*nk, 1, 1); eps is the head-norm epsilon (1e-6, NOT rms_norm_eps — it is
/// hardcoded in the CPU reference), inv = 1/sqrt(kd).
pub(crate) fn q35_qknorm_pc(nk: usize, kd: usize, key_dim: usize, eps: f32, inv: f32) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(5 * 4);
    v.write_all(&(nk as u32).to_le_bytes()).unwrap();
    v.write_all(&(kd as u32).to_le_bytes()).unwrap();
    v.write_all(&(key_dim as u32).to_le_bytes()).unwrap();
    v.write_all(&eps.to_le_bytes()).unwrap();
    v.write_all(&inv.to_le_bytes()).unwrap();
    v
}

/// Push constants for `q35_gdn_step` (WS1b delta-rule recurrence + gated norm):
/// { uint kd; uint vd; uint ratio; uint v_off; float eps; uint nv; }. Dispatch
/// (nv, 1, 1); ratio = nv/nk, v_off = 2*key_dim (v_flat offset into conv_out),
/// eps = cfg.rms_norm_eps (the gated-norm epsilon).
pub(crate) fn q35_gdn_pc(kd: usize, vd: usize, ratio: usize, v_off: usize, eps: f32, nv: usize) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(6 * 4);
    v.write_all(&(kd as u32).to_le_bytes()).unwrap();
    v.write_all(&(vd as u32).to_le_bytes()).unwrap();
    v.write_all(&(ratio as u32).to_le_bytes()).unwrap();
    v.write_all(&(v_off as u32).to_le_bytes()).unwrap();
    v.write_all(&eps.to_le_bytes()).unwrap();
    v.write_all(&(nv as u32).to_le_bytes()).unwrap();
    v
}

/// Push constants for `kda_decay` (Kimi lever #5 per-key-channel decay
/// precompute): { uint nh; uint kd; }. Dispatch ((nh*kd + 255)/256, 1, 1); one
/// thread per key-channel of `nh*kd` (== proj == key_dim).
pub(crate) fn kda_decay_pc(nh: usize, kd: usize) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(2 * 4);
    v.write_all(&(nh as u32).to_le_bytes()).unwrap();
    v.write_all(&(kd as u32).to_le_bytes()).unwrap();
    v
}

/// Push constants for `ling_kda_decay` (Ling fused-KDA safe_gate per-key-channel
/// decay precompute): { uint nh; uint kd; float lower_bound; }. Dispatch
/// ((nh*kd + 255)/256, 1, 1). Distinct from `kda_decay_pc` by the `lower_bound`
/// field — Ling's decay is `exp(lower_bound·sigmoid(exp(A_log)·(f+dt_bias)))`,
/// NOT Kimi's `exp(-exp(A_log)·softplus(·))`.
pub(crate) fn ling_kda_decay_pc(nh: usize, kd: usize, lower_bound: f32) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(3 * 4);
    v.write_all(&(nh as u32).to_le_bytes()).unwrap();
    v.write_all(&(kd as u32).to_le_bytes()).unwrap();
    v.write_all(&lower_bound.to_le_bytes()).unwrap();
    v
}

/// Push constants for `q35_gdn_scan` (P1a sequential multi-token GatedDeltaNet
/// scan): { uint kd; uint vd; uint ratio; uint v_off; float eps; uint nv;
/// uint n_tokens; uint conv_dim; uint key_dim; uint value_dim; }. Dispatch
/// (nv, 1, 1); same layout as `q35_gdn_pc` with the T-scan row strides
/// appended (conv_dim/key_dim/value_dim are the per-token buffer row
/// strides — see the shader's buffer-layout doc comment).
#[allow(clippy::too_many_arguments)]
pub(crate) fn q35_gdn_scan_pc(
    kd: usize, vd: usize, ratio: usize, v_off: usize, eps: f32, nv: usize,
    n_tokens: usize, conv_dim: usize, key_dim: usize, value_dim: usize,
) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(10 * 4);
    v.write_all(&(kd as u32).to_le_bytes()).unwrap();
    v.write_all(&(vd as u32).to_le_bytes()).unwrap();
    v.write_all(&(ratio as u32).to_le_bytes()).unwrap();
    v.write_all(&(v_off as u32).to_le_bytes()).unwrap();
    v.write_all(&eps.to_le_bytes()).unwrap();
    v.write_all(&(nv as u32).to_le_bytes()).unwrap();
    v.write_all(&(n_tokens as u32).to_le_bytes()).unwrap();
    v.write_all(&(conv_dim as u32).to_le_bytes()).unwrap();
    v.write_all(&(key_dim as u32).to_le_bytes()).unwrap();
    v.write_all(&(value_dim as u32).to_le_bytes()).unwrap();
    v
}

/// Push constants for `q35_moe_accum` (WS3 MoE tail on GPU): { uint n;
/// float s0..s7 }. Dispatch ((n+255)/256, 1, 1); n = hidden size, s0..s7 the
/// routed-expert scores in slot order (fixed top_k = 8).
pub(crate) fn q35_moe_accum_pc(n: usize, scores: &[f32]) -> Vec<u8> {
    use std::io::Write;
    debug_assert_eq!(scores.len(), 8, "q35_moe_accum is a fixed top-8 kernel");
    let mut v = Vec::with_capacity(9 * 4);
    v.write_all(&(n as u32).to_le_bytes()).unwrap();
    for &s in &scores[..8] {
        v.write_all(&s.to_le_bytes()).unwrap();
    }
    v
}

/// Push constants for `laguna_moe_accum` (Laguna MoE tail on GPU): { uint n;
/// float w0..w9 }. Dispatch ((n+255)/256, 1, 1); n = hidden size, w0..w9 the
/// routed-expert weights in slot order (fixed top_k = 10; the weights already
/// carry norm_topk_prob + ×2.5 routed_scaling, baked by `router_forward`).
pub(crate) fn laguna_moe_accum_pc(n: usize, weights: &[f32]) -> Vec<u8> {
    use std::io::Write;
    debug_assert_eq!(weights.len(), 10, "laguna_moe_accum is a fixed top-10 kernel");
    let mut v = Vec::with_capacity(11 * 4);
    v.write_all(&(n as u32).to_le_bytes()).unwrap();
    for &w in &weights[..10] {
        v.write_all(&w.to_le_bytes()).unwrap();
    }
    v
}

/// Push constants for `laguna_router` (Laguna MoE router on GPU): { uint ne;
/// uint hs; uint top_k; float routed_scaling; uint norm_topk }. One workgroup
/// (dispatch (1,1,1)); `ne` <= 256 (one thread per expert). `norm_topk` = 1 to
/// renormalize the selected weights to sum 1 before the `routed_scaling` (×2.5)
/// multiply, matching `nemotron::router_forward`.
pub(crate) fn laguna_router_pc(
    ne: usize,
    hs: usize,
    top_k: usize,
    routed_scaling: f32,
    norm_topk: bool,
) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(5 * 4);
    v.write_all(&(ne as u32).to_le_bytes()).unwrap();
    v.write_all(&(hs as u32).to_le_bytes()).unwrap();
    v.write_all(&(top_k as u32).to_le_bytes()).unwrap();
    v.write_all(&routed_scaling.to_le_bytes()).unwrap();
    v.write_all(&(norm_topk as u32).to_le_bytes()).unwrap();
    v
}

/// Push constants for `ling_moe_router` (Phase-1 GPU grouped-topk SELECTION over
/// pre-computed logits): { uint ne; uint top_k; uint n_group; uint topk_group;
///   float routed_scaling; uint norm_topk }. One workgroup (1,1,1). The router
/// matvec (logits) is done separately by the fast `mul_mat_vec` kernel.
pub(crate) fn ling_moe_router_pc(
    ne: usize,
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling: f32,
    norm_topk: bool,
) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(6 * 4);
    v.write_all(&(ne as u32).to_le_bytes()).unwrap();
    v.write_all(&(top_k as u32).to_le_bytes()).unwrap();
    v.write_all(&(n_group as u32).to_le_bytes()).unwrap();
    v.write_all(&(topk_group as u32).to_le_bytes()).unwrap();
    v.write_all(&routed_scaling.to_le_bytes()).unwrap();
    v.write_all(&(norm_topk as u32).to_le_bytes()).unwrap();
    v
}

/// Push constants for `ling_moe_meta` (GPU meta-builder): { uint top_k;
/// uint out_gu; uint packstride_gu; uint sbstride_gu; uint out_dn;
/// uint packstride_dn; uint sbstride_dn; uint xstride_dn }. Dispatch (1,1,1)
/// with top_k threads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ling_moe_meta_pc(
    top_k: usize,
    out_gu: usize,
    packstride_gu: usize,
    sbstride_gu: usize,
    out_dn: usize,
    packstride_dn: usize,
    sbstride_dn: usize,
    xstride_dn: usize,
) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 * 4);
    for x in [top_k, out_gu, packstride_gu, sbstride_gu, out_dn, packstride_dn, sbstride_dn, xstride_dn] {
        v.extend_from_slice(&(x as u32).to_le_bytes());
    }
    v
}

/// Push constants for `nemotron_moe_accum` (Nemotron MoE-tail collapse,
/// R1b): { uint n; uint top_k }. Dispatch ((n+255)/256, 1, 1); n = latent
/// size, top_k = number of routed experts (variable, unlike q35_moe_accum's
/// fixed top-8).
pub(crate) fn nem_moe_accum_pc(n: usize, top_k: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&(n as u32).to_le_bytes());
    v.extend_from_slice(&(top_k as u32).to_le_bytes());
    v
}

/// Push constants for `nemotron_ssm_conv_step` (R2 GPU SSD scan, Mamba2
/// causal depthwise conv1d + SiLU): { uint conv_dim; uint kern; uint x_off; }.
/// Dispatch ((conv_dim+255)/256, 1, 1); x_off = intermediate (the raw_bc
/// offset into the in_proj output row `proj`).
pub(crate) fn nem_ssm_conv_pc(conv_dim: usize, kern: usize, x_off: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(12);
    v.extend_from_slice(&(conv_dim as u32).to_le_bytes());
    v.extend_from_slice(&(kern as u32).to_le_bytes());
    v.extend_from_slice(&(x_off as u32).to_le_bytes());
    v
}

/// Push constants for `nemotron_ssd_scan` (R2 GPU SSD scan, per-head SSD
/// recurrence + gate): { uint nh; uint hd; uint ss; uint heads_per_group;
/// uint inter; uint conv_dim; uint ng_ss; float time_step_min; }. Dispatch
/// (nh, 1, 1); ng_ss = n_groups * ssm_state_size (C's group-block offset
/// into `conv_out`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn nem_ssd_scan_pc(
    nh: usize,
    hd: usize,
    ss: usize,
    heads_per_group: usize,
    inter: usize,
    conv_dim: usize,
    ng_ss: usize,
    time_step_min: f32,
) -> Vec<u8> {
    let mut v = Vec::with_capacity(32);
    v.extend_from_slice(&(nh as u32).to_le_bytes());
    v.extend_from_slice(&(hd as u32).to_le_bytes());
    v.extend_from_slice(&(ss as u32).to_le_bytes());
    v.extend_from_slice(&(heads_per_group as u32).to_le_bytes());
    v.extend_from_slice(&(inter as u32).to_le_bytes());
    v.extend_from_slice(&(conv_dim as u32).to_le_bytes());
    v.extend_from_slice(&(ng_ss as u32).to_le_bytes());
    v.extend_from_slice(&time_step_min.to_le_bytes());
    v
}

/// Push constants for `nemotron_gated_rmsnorm` (R2 GPU SSD scan, gated
/// RMSNorm scale + weight): { uint group_size; float eps; uint n_groups; }.
/// Dispatch (n_groups, 1, 1).
pub(crate) fn nem_gated_rmsnorm_pc(group_size: usize, eps: f32, n_groups: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(12);
    v.extend_from_slice(&(group_size as u32).to_le_bytes());
    v.extend_from_slice(&eps.to_le_bytes());
    v.extend_from_slice(&(n_groups as u32).to_le_bytes());
    v
}

/// Push constants for `q35_moe_accum_batched` (Phase 3 grouped-prefill MoE
/// tail, plan-epilogue-fused-moe-gemm.md §5): { uint n_out; uint h; uint top_k }.
/// `n_out` = T*h is the thread bound; dispatch ((n_out+255)/256, 1, 1). The
/// score-weighted routed combine reduces the T*top_k contiguous `down_out` rows
/// for each token in fixed slot-ascending order (== the host combine's order,
/// so cos=1.0), then adds the sigmoid-gated shared expert. Runtime `top_k`
/// (NOT the top_k=8-hardcoded decode `q35_moe_accum_pc`).
pub(crate) fn q35_moe_accum_batched_pc(t: usize, h: usize, top_k: usize) -> Vec<u8> {
    use std::io::Write;
    let mut v = Vec::with_capacity(3 * 4);
    for x in [(t * h) as u32, h as u32, top_k as u32] {
        v.write_all(&x.to_le_bytes()).unwrap();
    }
    v
}

/// Convert f32 weights to f16 bytes for GPU upload.
/// f16 halves memory bandwidth which is the main bottleneck for matvec ops.
pub(crate) fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; data.len() * 2];
    for (i, &v) in data.iter().enumerate() {
        let h = half::f16::from_f32(v);
        bytes[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
    }
    bytes
}

pub(crate) fn read_f32_buf(buf: &compute::Buffer, count: usize) -> Vec<f32> {
    let ptr = buf.mapped_ptr.unwrap() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
}

#[cfg(all(test, feature = "qwen35"))]
mod dispatch_tests {
    use super::*;
    use crate::qwen35;
    use crate::tp::{q35_tp_shard, nvfp4_shard_rows, nvfp4_shard_cols};

    fn u32s(bytes: &[u8]) -> Vec<u32> {
        bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[test]
    fn matvec_variant_quant_and_rows() {
        // q8_0 large output -> dequant base, r8.
        assert_eq!(matvec_variant_core(QuantFormat::Q8_0, false, true, None, 2048),
                   ("mul_mat_vec_q8_0deq_f32_f32_r8".to_string(), 8));
        // Default (f16 weight), small output n<1024 -> r1 (no _r suffix).
        assert_eq!(matvec_variant_core(QuantFormat::F16, false, true, None, 512),
                   ("mul_mat_vec_f16_f32_f32".to_string(), 1));
        // Default f32 weight, large output -> r8.
        assert_eq!(matvec_variant_core(QuantFormat::F16, false, false, None, 4096),
                   ("mul_mat_vec_f32_f32_f32_r8".to_string(), 8));
        // bf16 quant -> bf16 base.
        assert_eq!(matvec_variant_core(QuantFormat::Bf16, false, true, None, 2048),
                   ("mul_mat_vec_bf16_f32_f32_r8".to_string(), 8));
        // Explicit rows override to a valid value.
        assert_eq!(matvec_variant_core(QuantFormat::F16, false, true, Some(4), 2048),
                   ("mul_mat_vec_f16_f32_f32_r4".to_string(), 4));
        // Invalid override (3) -> falls back to r1, base only.
        assert_eq!(matvec_variant_core(QuantFormat::F16, false, true, Some(3), 2048),
                   ("mul_mat_vec_f16_f32_f32".to_string(), 1));
        // Subgroup base has no _r siblings.
        assert_eq!(matvec_variant_core(QuantFormat::F16, true, true, None, 2048),
                   ("mul_mat_vec_f16_f32_f32_subgroup".to_string(), 1));
    }

    #[test]
    fn matvec_cols_variant_quant_aware() {
        // Design C: q8_0-resident batched verify at t>1 -> the DEQUANT cols base
        // (weight read+dequantized once, MAC'd across T columns), rows=8/t.
        assert_eq!(matvec_cols_variant_core(QuantFormat::Q8_0, 2048, 4),
                   ("mul_mat_vec_q8_0deq_f32_f32_r2_c4".to_string(), 2));
        assert_eq!(matvec_cols_variant_core(QuantFormat::Q8_0, 2048, 8),
                   ("mul_mat_vec_q8_0deq_f32_f32_r1_c8".to_string(), 1));
        assert_eq!(matvec_cols_variant_core(QuantFormat::Q8_0, 512, 2),
                   ("mul_mat_vec_q8_0deq_f32_f32_r4_c2".to_string(), 4));
        // q4_0 / q4_K ride the same flip.
        assert_eq!(matvec_cols_variant_core(QuantFormat::Q4_0, 2048, 5),
                   ("mul_mat_vec_q4_0deq_f32_f32_r1_c5".to_string(), 1));
        assert_eq!(matvec_cols_variant_core(QuantFormat::Q4_K, 2048, 3),
                   ("mul_mat_vec_q4_kdeq_f32_f32_r2_c3".to_string(), 2));
        // f16 / other codecs keep the pre-flip f16 scalar cols base.
        assert_eq!(matvec_cols_variant_core(QuantFormat::F16, 2048, 4),
                   ("mul_mat_vec_f16_f32_f32_r2_c4".to_string(), 2));
        // t<=1 collapses to the exact per-token decode variant, quant-aware.
        assert_eq!(matvec_cols_variant_core(QuantFormat::Q8_0, 2048, 1),
                   ("mul_mat_vec_q8_0deq_f32_f32_r8".to_string(), 8));
        assert_eq!(matvec_cols_variant_core(QuantFormat::F16, 512, 1),
                   ("mul_mat_vec_f16_f32_f32".to_string(), 1));
    }

    #[test]
    fn q35_dn_pc_layouts() {
        // Live 35B-A3B deltanet dims: h=2048, nk=16, nv=32, kd=vd=128, kern=4.
        let conv_dim = 2 * 16 * 128 + 32 * 128; // 2*key_dim + value_dim = 8192
        let pc = q35_conv_pc(conv_dim, 4);
        assert_eq!(pc.len(), 2 * 4);
        assert_eq!(u32s(&pc), vec![8192, 4]);

        let inv = 1.0f32 / (128f32).sqrt();
        let pc = q35_qknorm_pc(16, 128, 2048, 1e-6, inv);
        assert_eq!(pc.len(), 5 * 4);
        assert_eq!(&u32s(&pc)[..3], &[16, 128, 2048]);
        assert_eq!(f32::from_le_bytes(pc[12..16].try_into().unwrap()), 1e-6);
        assert_eq!(f32::from_le_bytes(pc[16..20].try_into().unwrap()), inv);

        let pc = q35_gdn_pc(128, 128, 2, 4096, 1e-6, 32);
        assert_eq!(pc.len(), 6 * 4);
        assert_eq!(&u32s(&pc)[..4], &[128, 128, 2, 4096]);
        assert_eq!(f32::from_le_bytes(pc[16..20].try_into().unwrap()), 1e-6);
        assert_eq!(u32s(&pc)[5], 32);
    }

    #[test]
    fn matvec_pc13_layout() {
        // k=5120, n=1024. 13 u32s: [k,k,k,n, k*n, k,n, 0,0,1,1,1,1].
        let pc = matvec_pc13(5120, 1024);
        assert_eq!(pc.len(), 13 * 4);
        assert_eq!(u32s(&pc),
            vec![5120, 5120, 5120, 1024, 5120 * 1024, 5120, 1024, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn q35_moe_accum_pc_layout() {
        // { uint n; float s0..s7 } — 9 words, scores in slot order.
        let scores = [0.5f32, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625];
        let pc = q35_moe_accum_pc(2048, &scores);
        assert_eq!(pc.len(), 9 * 4);
        assert_eq!(u32s(&pc)[0], 2048);
        for (i, &s) in scores.iter().enumerate() {
            let off = 4 + i * 4;
            assert_eq!(f32::from_le_bytes(pc[off..off + 4].try_into().unwrap()), s);
        }
    }

    #[test]
    fn gemm_pc_16_u32_invariant() {
        // The documented all-zero-output bug was a 16-vs-20 u32 mismatch; assert
        // the exact 16-u32 layout. t=2 (B rows), n=1024 (weight rows=M), k=5120.
        let pc = gemm_pc(2, 1024, 5120);
        assert_eq!(pc.len(), 16 * 4, "gemm_pc MUST emit exactly 16 u32");
        let w = u32s(&pc);
        assert_eq!(w[0], 1024, "M = n (weight rows)");
        assert_eq!(w[1], 2, "N = t (input rows)");
        assert_eq!(w[2], 5120, "K");
        assert_eq!(w[10], 1, "num_batches = 1");
        assert_eq!(w[11], 5120, "k_split = K (no split-k)");
    }

    #[test]
    fn gemm_pc_mlx4_id_16_u32_invariant() {
        // Phase B grouped-expert PC: EXACTLY 16 u32 with the MUL_MAT_ID +
        // DATA_A_MLX4 field order (NOT the 19-u32 dense gemm_pc_mlx4). Model
        // the gate/up dispatch: M=mi=768, K=h=2048, top_k=8, t=32, ne11=1
        // (broadcast x), stride_b=h, batch_stride_b=h, stride_d=mi,
        // batch_stride_d=top_k*mi, group_size=64, pack_stride=mi*h/8,
        // sb_stride=mi*(h/64).
        let (m, k, top_k, t, ne11) = (768usize, 2048usize, 8usize, 32usize, 1usize);
        let (stride_b, bsb, stride_d, bsd) = (k, k, m, top_k * m);
        let (gs, pack_stride, sb_stride) = (64usize, m * k / 8, m * (k / 64));
        let pc = gemm_pc_mlx4_id(m, k, top_k, t, ne11, stride_b, bsb, stride_d, bsd, gs, pack_stride, sb_stride);
        assert_eq!(pc.len(), 16 * 4, "gemm_pc_mlx4_id MUST emit exactly 16 u32");
        let w = u32s(&pc);
        assert_eq!(w[0], m as u32, "M = out_features");
        assert_eq!(w[1], (top_k * t) as u32, "N = nei0*nei1 = top_k*t");
        assert_eq!(w[2], k as u32, "K");
        assert_eq!(w[3], k as u32, "stride_a = K");
        assert_eq!(w[4], stride_b as u32, "stride_b");
        assert_eq!(w[5], stride_d as u32, "stride_d");
        assert_eq!(w[6], 0, "batch_stride_a ignored by mlx4 head → 0");
        assert_eq!(w[7], bsb as u32, "batch_stride_b");
        assert_eq!(w[8], bsd as u32, "batch_stride_d");
        assert_eq!(w[9], top_k as u32, "nei0 = top_k");
        assert_eq!(w[10], t as u32, "nei1 = t");
        assert_eq!(w[11], top_k as u32, "nbi1 = top_k");
        assert_eq!(w[12], ne11 as u32, "ne11");
        assert_eq!(w[13], gs as u32, "mlx4_group_size");
        assert_eq!(w[14], pack_stride as u32, "mlx4_pack_stride");
        assert_eq!(w[15], sb_stride as u32, "mlx4_sb_stride");
    }

    #[test]
    fn gemm_pc_mlx4_id_gateup_18_u32_invariant() {
        // Epilogue-fused gate+up PC: the gemm_pc_mlx4_id 16-u32 prefix
        // UNCHANGED, plus 2 more u32 for the up weight's own strides.
        let (m, k, top_k, t, ne11) = (512usize, 2048usize, 8usize, 32usize, 1usize);
        let (stride_b, bsb, stride_d, bsd) = (k, k, m, top_k * m);
        let (gs, g_pack, g_sb) = (64usize, m * k / 8, m * (k / 64));
        let (u_pack, u_sb) = (m * k / 8 + 7, m * (k / 64) + 3); // distinct from gate's, to catch field swaps
        let pc = gemm_pc_mlx4_id_gateup(
            m, k, top_k, t, ne11, stride_b, bsb, stride_d, bsd, gs, g_pack, g_sb, u_pack, u_sb,
        );
        assert_eq!(pc.len(), 18 * 4, "gemm_pc_mlx4_id_gateup MUST emit exactly 18 u32");
        let w = u32s(&pc);
        // Prefix identical to gemm_pc_mlx4_id (spot-check the load-bearing fields).
        assert_eq!(w[0], m as u32, "M = out_features");
        assert_eq!(w[13], gs as u32, "mlx4_group_size (shared gate/up)");
        assert_eq!(w[14], g_pack as u32, "gate mlx4_pack_stride");
        assert_eq!(w[15], g_sb as u32, "gate mlx4_sb_stride");
        // New tail: the up weight's own strides.
        assert_eq!(w[16], u_pack as u32, "up_pack_stride");
        assert_eq!(w[17], u_sb as u32, "up_sb_stride");
    }

    #[test]
    fn gemm_variant_quant_id_bn_thresholds() {
        let base = "matmul_mlx4_id_gateup_silu_f32_fp32";
        // <=16 tokens/expert -> the BN=16 sibling.
        assert_eq!(gemm_variant_quant_id_bn(base, 1), (format!("{base}_bn16"), 64, 16));
        assert_eq!(gemm_variant_quant_id_bn(base, 16), (format!("{base}_bn16"), 64, 16));
        // 17..=32 -> BN=32.
        assert_eq!(gemm_variant_quant_id_bn(base, 17), (format!("{base}_bn32"), 64, 32));
        assert_eq!(gemm_variant_quant_id_bn(base, 32), (format!("{base}_bn32"), 64, 32));
        // >32 -> falls back to the plain BM=BN=64 base (no sibling).
        assert_eq!(gemm_variant_quant_id_bn(base, 33), (base.to_string(), 64, 64));
        assert_eq!(gemm_variant_quant_id_bn(base, 512), (base.to_string(), 64, 64));
    }

    #[test]
    fn nvfp4_fold_scales_reconstructs_dequant() {
        use crate::model::{dequantize_nvfp4, NVFP4_E2M1_LUT};
        // layer0 mlp.gate_proj row0, in=64, group=16 (same golden as model.rs).
        let packed: [u8; 32] = [
            0x38,0x0c,0x93,0xa7,0x34,0x25,0x7a,0xdb,0x56,0x22,0xb5,0x42,0x42,0x04,0x26,0x2b,
            0xad,0xf5,0xec,0xf1,0x2c,0x7f,0xeb,0x41,0x98,0xea,0x1e,0x45,0xdd,0xee,0x3c,0x19];
        let wscale: [u8; 4] = [0x5d,0x60,0x5a,0x5d];
        let global = 0.00015113468f32;
        let (out_f, in_f, gs) = (1usize, 64usize, 16usize);
        let groups = in_f / gs;
        let folded = nvfp4_fold_scales(&wscale, global); // Vec<f32> [out*groups]
        let deq = dequantize_nvfp4(&packed, &wscale, global, out_f, in_f, gs);
        for j in 0..in_f {
            let byte = packed[j / 2];
            let nib = if j % 2 == 0 { byte & 0xF } else { byte >> 4 } as usize;
            let recon = NVFP4_E2M1_LUT[nib] * folded[j / gs];
            // f32 reassociation (lut*(bscale*global) vs (lut*bscale)*global) can
            // differ by ~1 ULP; use a realistic f32 relative tolerance, not a
            // fixed absolute one (which is unrealistic near the value's own scale).
            let tol = 1e-6 * deq[j].abs().max(1e-6);
            assert!((recon - deq[j]).abs() <= tol,
                "elem {j}: fold {recon} vs dequant {}", deq[j]);
        }
        assert_eq!(folded.len(), out_f * groups);
    }

    #[test]
    fn nvfp4_variant_selects_r8_for_wide() {
        assert_eq!(matvec_nvfp4_variant(17408), ("mul_mat_vec_nvfp4_f32_f32_r8".to_string(), 8));
        assert_eq!(matvec_nvfp4_variant(512),   ("mul_mat_vec_nvfp4_f32_f32".to_string(), 1));
    }

    #[test]
    fn nvfp4_e4m3_variant_selects_r8_for_wide() {
        assert_eq!(matvec_nvfp4_e4m3_variant(17408),
            ("mul_mat_vec_nvfp4_e4m3_f32_f32_r8".to_string(), 8));
        assert_eq!(matvec_nvfp4_e4m3_variant(512),
            ("mul_mat_vec_nvfp4_e4m3_f32_f32".to_string(), 1));
    }

    #[test]
    fn nvfp4_e4m3_pc_layout() {
        // 5×u32 head (matvec_mlx4_pc_off layout) + trailing f32 global = 24 bytes.
        let pc = matvec_nvfp4_e4m3_pc(64, 8, 16, 0.00015113468f32);
        assert_eq!(pc.len(), 24);
        assert_eq!(u32::from_le_bytes(pc[0..4].try_into().unwrap()), 64);   // k
        assert_eq!(u32::from_le_bytes(pc[4..8].try_into().unwrap()), 8);    // n
        assert_eq!(u32::from_le_bytes(pc[8..12].try_into().unwrap()), 16);  // group_size
        assert_eq!(u32::from_le_bytes(pc[12..16].try_into().unwrap()), 0);  // packed_off
        assert_eq!(u32::from_le_bytes(pc[16..20].try_into().unwrap()), 0);  // sb_off
        assert_eq!(f32::from_le_bytes(pc[20..24].try_into().unwrap()), 0.00015113468f32);
    }

    #[test]
    fn nvfp4_dispatch_routes_by_flag() {
        // OFF -> f32-fold kernel + 20-byte mlx4 pc; ON -> e4m3 kernel + 24-byte pc.
        let (s_off, r_off, pc_off) = nvfp4_dispatch(2048, 512, 16, false, 1.0);
        assert_eq!(s_off, "mul_mat_vec_nvfp4_f32_f32");
        assert_eq!(r_off, 1);
        assert_eq!(pc_off.len(), 20);
        let (s_on, r_on, pc_on) = nvfp4_dispatch(2048, 2048, 16, true, 0.25);
        assert_eq!(s_on, "mul_mat_vec_nvfp4_e4m3_f32_f32_r8");
        assert_eq!(r_on, 8);
        assert_eq!(pc_on.len(), 24);
        assert_eq!(f32::from_le_bytes(pc_on[20..24].try_into().unwrap()), 0.25);
    }

    /// CPU BIT-EXACT GATE (mandatory): the E4M3-resident matvec path must be
    /// argmax-exact / cos=1.0 vs the f32-fold path on a synthetic NVFP4 weight.
    /// This emulates the two ACTUAL data paths:
    ///   * f32-fold: load-time `nvfp4_fold_scales(wscale, global)` -> f32 scale;
    ///     shader does `kE2M1[code] * folded[r,g]`.
    ///   * e4m3-resident: raw `wscale` byte stays resident (read via the same
    ///     uint[]-reinterpret byte-extract the shader uses); shader does
    ///     `kE2M1[code] * (e4m3(byte) * global)`.
    /// Because `e4m3_to_f32` (== the shader's kE4M3 LUT, per
    /// `fp8_shader_lut_matches_e4m3`) and `NVFP4_E2M1_LUT` (== kE2M1) are
    /// bit-equal to the GPU LUTs, and the parenthesization matches the shader,
    /// the two paths are BIT-EXACT (not merely cos=1.0). Asserted per-row with
    /// `to_bits()` equality — stronger than the required argmax-exactness. This
    /// is the correctness story: a scale-direction bug (the class that bit the
    /// gemma inverted-global-scale) would break the byte<->global split here.
    #[test]
    fn nvfp4_e4m3_resident_matches_f32_fold() {
        use crate::model::{e4m3_to_f32, NVFP4_E2M1_LUT};
        // Multi-row synthetic NVFP4 weight [out=8, in=64, group=16] so the
        // argmax over the 8-row output is meaningful. Real global magnitude.
        let (out_f, in_f, gs) = (8usize, 64usize, 16usize);
        let (bpr, groups) = (in_f / 2, in_f / gs);
        let global = 0.00015113468f32;
        let packed: Vec<u8> = (0..out_f * bpr).map(|i| (i.wrapping_mul(37).wrapping_add(11)) as u8).collect();
        // e4m3 scale bytes in the 0x50..0x63 range (finite positive block scales).
        let wscale: Vec<u8> = (0..out_f * groups).map(|i| (0x50 + (i % 20)) as u8).collect();
        let x: Vec<f32> = (0..in_f).map(|i| ((i as f32) * 0.37).sin()).collect();

        // ── f32-fold data path ──────────────────────────────────────────────
        let folded = nvfp4_fold_scales(&wscale, global); // exactly the load-time fold
        let mut out_fold = vec![0f32; out_f];
        for r in 0..out_f {
            let mut acc = 0f32;
            for j in 0..in_f {
                let byte = packed[r * bpr + j / 2];
                let code = if j % 2 == 0 { byte & 0xF } else { byte >> 4 } as usize;
                let w = NVFP4_E2M1_LUT[code] * folded[r * groups + j / gs];
                acc += w * x[j];
            }
            out_fold[r] = acc;
        }

        // ── e4m3-resident data path ─────────────────────────────────────────
        // Reinterpret the raw wscale bytes as LE u32 words + absolute-byte
        // extract, mirroring the shader's `scaleb[sidx>>2] >> ((sidx&3)*8)`.
        let swords: Vec<u32> = wscale
            .chunks(4)
            .map(|c| c.iter().enumerate().fold(0u32, |w, (i, &b)| w | ((b as u32) << (8 * i))))
            .collect();
        let mut out_e4m3 = vec![0f32; out_f];
        for r in 0..out_f {
            let mut acc = 0f32;
            for j in 0..in_f {
                let byte = packed[r * bpr + j / 2];
                let code = if j % 2 == 0 { byte & 0xF } else { byte >> 4 } as usize;
                let sidx = r * groups + j / gs;
                let sbyte = ((swords[sidx >> 2] >> ((sidx & 3) * 8)) & 0xFF) as u8;
                // Parenthesized exactly as the shader: bscale first, then E2M1.
                let bscale = e4m3_to_f32(sbyte) * global;
                let w = NVFP4_E2M1_LUT[code] * bscale;
                acc += w * x[j];
            }
            out_e4m3[r] = acc;
        }

        // BIT-EXACT per row (strictly stronger than argmax-exact / cos=1.0).
        for r in 0..out_f {
            assert_eq!(out_e4m3[r].to_bits(), out_fold[r].to_bits(),
                "row {r}: e4m3 {} vs f32-fold {} not bit-exact", out_e4m3[r], out_fold[r]);
        }
        // And explicit argmax equality (the decode-time correctness gate).
        let argmax = |v: &[f32]| v.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        assert_eq!(argmax(&out_e4m3), argmax(&out_fold), "argmax differs");
    }

    /// CPU PRE-VALIDATION GATE (Mac, no GPU) for the NVFP4 REPACK refactor: the
    /// factored per-chunk form the `mul_mat_vec_nvfp4repack_f32_f32` shader
    /// computes must be ARGMAX-EXACT (and cos≈1.0, mae~ULP) vs the naive
    /// per-element `dequantize_nvfp4` + `cpu_matmul` oracle. This is the offline
    /// twin of the on-node gate: it emulates the shader's exact arithmetic —
    /// dwordx4 chunk of 32 nibbles = 2 groups of 16, each 16-elem group's E2M1
    /// dot factored out with the folded f32 scale applied ONCE, accumulated
    /// `fma(scaleA, qxA, fma(scaleB, qxB, temp))` per chunk — and shows the
    /// fma+reassociation vs the per-element `(E2M1*scale)*x` sum stays
    /// argmax-exact. Mirrors the mlx4 Mac pre-validation that preceded its
    /// on-node gate. The e4m3-resident repack yields the IDENTICAL factored
    /// value (folded == e4m3(byte)*global, parenthesized — see
    /// nvfp4_e4m3_resident_matches_f32_fold), so this covers both kernels.
    #[test]
    fn nvfp4_repack_factored_argmax_exact_vs_dequant() {
        use crate::model::{cpu_matmul, dequantize_nvfp4, NVFP4_E2M1_LUT};
        // (nvfp4_fold_scales is a push_constants fn, in scope via `use super::*`.)
        // Realistic decode shape: k%32==0, group_size=16, multi-row output.
        let (n, k, gs) = (256usize, 512usize, 16usize);
        let (bpr, groups) = (k / 2, k / gs);
        let global = 0.00015113468f32;
        let packed: Vec<u8> = (0..n * bpr)
            .map(|i| (i.wrapping_mul(131).wrapping_add(17)) as u8).collect();
        let wscale: Vec<u8> = (0..n * groups)
            .map(|i| (0x48 + (i * 7 % 40)) as u8).collect(); // finite positive e4m3
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.191).sin() * 0.9).collect();

        // Oracle: full dequant -> matmul (the v1 kernel's per-element math).
        let deq = dequantize_nvfp4(&packed, &wscale, global, n, k, gs);
        let cpu = cpu_matmul(&x, &deq, 1, k, n);

        // Factored repack emulation (the new shader's exact arithmetic).
        let folded = nvfp4_fold_scales(&wscale, global);
        let chunks = k / 32;
        let mut rep = vec![0f32; n];
        for r in 0..n {
            let mut temp = 0f32;
            for c in 0..chunks {
                // qxA = first 16 elems (group 2c), qxB = next 16 (group 2c+1),
                // each as 4 in-order dot4 partials summed (matches the shader).
                let mut half_dot = |j0: usize| -> f32 {
                    let mut acc = 0f32;
                    for v in 0..4 {
                        let mut d = 0f32;
                        for t in 0..4 {
                            let j = j0 + 4 * v + t;
                            let byte = packed[r * bpr + j / 2];
                            let code = if j % 2 == 0 { byte & 0xF } else { byte >> 4 } as usize;
                            d += NVFP4_E2M1_LUT[code] * x[j];
                        }
                        acc += d;
                    }
                    acc
                };
                let qx_a = half_dot(32 * c);
                let qx_b = half_dot(32 * c + 16);
                let scale_a = folded[r * groups + 2 * c];
                let scale_b = folded[r * groups + 2 * c + 1];
                // fma(scaleA, qxA, fma(scaleB, qxB, temp))
                temp = scale_a.mul_add(qx_a, scale_b.mul_add(qx_b, temp));
            }
            rep[r] = temp;
        }

        // Gate: argmax-exact, cos≈1.0, mae ~ULP-scale.
        let argmax = |v: &[f32]| v.iter().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &val)|
                if val > bv { (i, val) } else { (bi, bv) }).0;
        assert_eq!(argmax(&rep), argmax(&cpu), "repack argmax != dequant argmax");
        let (mut dot, mut ng, mut nc, mut mae) = (0f64, 0f64, 0f64, 0f64);
        for (a, b) in rep.iter().zip(cpu.iter()) {
            dot += (*a as f64) * (*b as f64);
            ng += (*a as f64) * (*a as f64);
            nc += (*b as f64) * (*b as f64);
            mae = mae.max((*a as f64 - *b as f64).abs());
        }
        let cos = dot / (ng.sqrt() * nc.sqrt() + 1e-12);
        assert!(cos > 0.99999999, "repack cos {cos} < 1-1e-8");
        // Relative mae: the reassociation error scales with the output magnitude.
        let scale = cpu.iter().fold(0f64, |m, &v| m.max((v as f64).abs())).max(1e-6);
        assert!(mae <= 1e-4 * scale, "repack mae {mae} too large (scale {scale})");
    }

    #[test]
    fn nvfp4_repack_routes_only_on_flag_and_shape() {
        // Flag default-OFF in the test process (env unset) -> v1 everywhere.
        assert!(!nvfp4_repack_flag(), "VLLM_VULKAN_NVFP4_REPACK must be unset in tests");
        // Shape guard is false without the flag regardless of shape.
        assert!(!nvfp4_repack_shape_ok(17408, 5120, 16));
        // f32-fold dispatch stays v1 (byte-identical default).
        let (s, _r, pc) = nvfp4_dispatch(17408, 5120, 16, false, 1.0);
        assert_eq!(s, "mul_mat_vec_nvfp4_f32_f32_r8");
        assert_eq!(pc.len(), 20);
        // e4m3 dispatch stays v1 too.
        let (se, _re, pce) = nvfp4_dispatch(17408, 5120, 16, true, 0.25);
        assert_eq!(se, "mul_mat_vec_nvfp4_e4m3_f32_f32_r8");
        assert_eq!(pce.len(), 24);
        // nemotron path (matvec_nvfp4_variant_k) stays v1.
        assert_eq!(matvec_nvfp4_variant_k(17408, 5120), matvec_nvfp4_variant(5120));
        // The repack combos are compiled at init (geom_combos_for non-empty).
        assert_eq!(geom_combos_for("mul_mat_vec_nvfp4repack_f32_f32").len(), 5);
        assert_eq!(geom_combos_for("mul_mat_vec_nvfp4_e4m3repack_f32_f32").len(), 5);
    }

    // Reconstruct per-element weight the GPU shader computes from (packed, folded).
    fn nvfp4_recon(packed: &[u8], folded: &[f32], out: usize, in_: usize, gs: usize) -> Vec<f32> {
        let (bpr, groups) = (in_/2, in_/gs);
        let mut w = vec![0f32; out*in_];
        for r in 0..out { for j in 0..in_ {
            let byte = packed[r*bpr + j/2];
            let nib = if j%2==0 { byte & 0xF } else { byte >> 4 } as usize;
            w[r*in_ + j] = crate::model::NVFP4_E2M1_LUT[nib] * folded[r*groups + j/gs];
        }}
        w
    }

    #[test]
    fn nvfp4_col_shard_equals_dequant_then_shard() {
        // [out=8, in=32, gs=16] synthetic. col-shard N=2,r=1 keeps rows [4,8).
        let (out, in_, gs, n, r) = (8usize, 32usize, 16usize, 2usize, 1usize);
        let packed: Vec<u8> = (0..out*in_/2).map(|i| (i*37 + 11) as u8).collect();
        let folded: Vec<f32> = (0..out*(in_/gs)).map(|i| 0.1 + i as f32 * 0.03).collect();
        let full = nvfp4_recon(&packed, &folded, out, in_, gs);
        let (p, s, o2, i2) = nvfp4_shard_rows(&packed, &folded, in_, gs, r*(out/n), out/n);
        assert_eq!((o2, i2), (out/n, in_));
        let got = nvfp4_recon(&p, &s, o2, i2, gs);
        // rows [4,8) of full == all rows of got
        for rr in 0..o2 { for j in 0..in_ {
            assert_eq!(got[rr*in_ + j].to_bits(), full[(rr + r*(out/n))*in_ + j].to_bits());
        }}
    }

    #[test]
    fn nvfp4_row_shard_equals_dequant_then_shard() {
        // down_proj-like [out=4, in=32, gs=16], row-shard N=2 keeps cols [16,32) (rank1).
        let (out, in_, gs, n, r) = (4usize, 32usize, 16usize, 2usize, 1usize);
        let packed: Vec<u8> = (0..out*in_/2).map(|i| (i*53 + 7) as u8).collect();
        let folded: Vec<f32> = (0..out*(in_/gs)).map(|i| 0.2 + i as f32 * 0.05).collect();
        let full = nvfp4_recon(&packed, &folded, out, in_, gs);
        let (cper, clo) = (in_/n, r*(in_/n));
        let (p, s, o2, i2) = nvfp4_shard_cols(&packed, &folded, out, in_, gs, clo, cper);
        assert_eq!((o2, i2), (out, cper));
        let got = nvfp4_recon(&p, &s, o2, i2, gs);
        for rr in 0..out { for j in 0..cper {
            assert_eq!(got[rr*cper + j].to_bits(), full[rr*in_ + (clo + j)].to_bits());
        }}
    }

    #[test]
    fn fp8_shader_lut_matches_e4m3() {
        let src = include_str!("../shaders/mul_mat_vec_fp8.comp");
        let s = src.find("kE4M3[256] = float[256](").expect("LUT decl");
        let start = src[s..].find('(').unwrap() + s + 1;
        let end = src[start..].find(')').expect("LUT close") + start;
        let vals: Vec<f32> = src[start..end].split(',').map(|t| t.trim())
            .filter(|t| !t.is_empty()).map(|t| t.parse::<f32>().expect("parse LUT")).collect();
        assert_eq!(vals.len(), 256, "LUT must have 256 entries");
        for code in 0..256u32 {
            let e = crate::model::e4m3_to_f32(code as u8);
            assert_eq!(vals[code as usize].to_bits(), e.to_bits(),
                "LUT[{code}]={} != e4m3_to_f32={e}", vals[code as usize]);
        }
    }

    #[test]
    fn fp8_arith_decode_matches_e4m3() {
        // Port of shaders/mul_mat_vec_fp8_fast.comp's e4m3_decode: identical
        // integer ops to the GLSL (bit-shift decode + bias-120 re-encode into
        // an f32 exponent field), asserted bit-exact vs model::e4m3_to_f32 for
        // all 256 codes.
        fn e4m3_decode(code: u8) -> f32 {
            let s = ((code >> 7) & 1) as u32;
            let e = ((code >> 3) & 0xF) as u32;
            let m = (code & 0x7) as u32;
            let mag: f32 = if e == 0 {
                (m as f32) / 512.0
            } else if e == 0xF && m == 7 {
                0.0
            } else {
                f32::from_bits(((e + 120) << 23) | (m << 20))
            };
            f32::from_bits(mag.to_bits() | (s << 31))
        }
        for code in 0..=255u8 {
            let want = crate::model::e4m3_to_f32(code);
            let got = e4m3_decode(code);
            assert_eq!(got.to_bits(), want.to_bits(),
                "code={code:#04x} got={got} want={want}");
        }
    }

    #[test]
    fn repack_moe_expert_base_is_16b_aligned_by_construction() {
        // The repack gate is `k % 32 == 0`. This test locks in WHY that single
        // condition is sufficient to admit MoE-expert shapes (non-zero per-expert
        // packed_off) without a per-expert alignment check: k%32==0 ⟹ k/8 is a
        // multiple of 4 ⟹ pack_stride = n*(k/8) is a multiple of 4 for ANY n ⟹
        // every expert base e*pack_stride is a multiple of 4 (16B-aligned for the
        // buffer_load_dwordx4). Kimi gate/up + down and gemma-12b attn shapes.
        let per_word = 8usize; // 4-bit
        for (k, n) in [(2304usize, 1024usize), (1024, 2304),   // Kimi gate/up, down
                       (3840, 4096), (3840, 2048), (4096, 3840)] { // gemma-12b attn
            assert_eq!(k % 32, 0, "k={k} must pass the repack gate");
            let pack_stride = n * (k / per_word);
            for e in [0usize, 1, 7, 255] {
                assert_eq!((e * pack_stride) % 4, 0,
                    "expert {e} base for (k={k},n={n}) is not 16B-aligned");
            }
        }
    }

    #[test]
    fn fp8_variant_selects_r8_for_wide() {
        // VLLM_VULKAN_FP8_FAST is now default-ON (address-gen cure), so the base
        // stem resolves to the arithmetic-decode fast kernel; the r-tier selection
        // (r8 for n>=1024, r1 otherwise) is unchanged. The =0 kill-switch path
        // (LUT oracle "mul_mat_vec_fp8_f32_f32") is covered by fp8_base()'s branch.
        assert_eq!(matvec_fp8_variant(6144), ("mul_mat_vec_fp8fast_f32_f32_r8".to_string(), 8));
        assert_eq!(matvec_fp8_variant(512),  ("mul_mat_vec_fp8fast_f32_f32".to_string(), 1));
    }

    #[test]
    fn geom_pickers_return_swept_winners() {
        // The sweep-derived win region (geom_sweep.csv, GFX1013, 2026-07-04).
        assert_eq!(matvec_mlx4_variant_k(2048, 512),
                   ("mul_mat_vec_mlx4_f32_f32_bs128_r2".to_string(), 2)); // MoE gate/up
        // MoE down: 1850MHz-pinned re-sweep rewired this shape onto the w8sg
        // base (VLLM_VULKAN_MLX4_W8 default ON — see matvec_mlx4_variant_k).
        assert_eq!(matvec_mlx4_variant_k(512, 2048),
                   ("mul_mat_vec_mlx4w8sg_f32_f32_bs64_r2".to_string(), 2)); // MoE down
        assert_eq!(matvec_f32_variant_k(512, 2048),
                   ("mul_mat_vec_f32_f32_f32_bs64_r1".to_string(), 1));   // shared down
        assert_eq!(matvec_f32_variant_k(2048, 256),
                   ("mul_mat_vec_f32_f32_f32_bs32_r2".to_string(), 2));   // router
        assert_eq!(geom_pick("mul_mat_vec_bf16_f32_f32", 2048, 8192), Some((256, 2)));
        assert_eq!(geom_pick("mul_mat_vec_bf16_f32_f32", 2048, 4096), Some((256, 2)));
        assert_eq!(geom_pick("mul_mat_vec_bf16_f32_f32", 4096, 2048), Some((256, 2)));
        assert_eq!(geom_pick("mul_mat_vec_q8_0deq_f32_f32", 2048, 8192), Some((32, 2)));
        assert_eq!(geom_pick("mul_mat_vec_q8_0deq_f32_f32", 2048, 4096), Some((64, 1)));
        assert_eq!(geom_pick("mul_mat_vec_q8_0deq_f32_f32", 2048, 512), Some((32, 4)));
        assert_eq!(geom_pick("mul_mat_vec_q8_0deq_f32_f32", 4096, 2048), Some((32, 4)));
        // Qwen3.6-27B TP-4 per-rank q8 shapes (VLLM_VULKAN_Q35_GEOM path;
        // geom_pick_q35tp — deliberately absent from the unconditional geom_pick).
        assert_eq!(geom_pick("mul_mat_vec_q8_0deq_f32_f32", 5120, 4352), None);
        assert_eq!(geom_pick_q35tp("mul_mat_vec_q8_0deq_f32_f32", 5120, 4352), Some((32, 2)));
        assert_eq!(geom_pick_q35tp("mul_mat_vec_q8_0deq_f32_f32", 5120, 3072), Some((32, 2)));
        assert_eq!(geom_pick_q35tp("mul_mat_vec_q8_0deq_f32_f32", 5120, 2560), Some((32, 2)));
        assert_eq!(geom_pick_q35tp("mul_mat_vec_q8_0deq_f32_f32", 5120, 1536), Some((64, 1)));
        assert_eq!(geom_pick_q35tp("mul_mat_vec_q8_0deq_f32_f32", 5120, 256), Some((32, 4)));
        assert_eq!(geom_pick_q35tp("mul_mat_vec_q8_0deq_f32_f32", 1536, 5120), Some((32, 4)));
        assert_eq!(geom_pick_q35tp("mul_mat_vec_q8_0deq_f32_f32", 4352, 5120), Some((32, 4)));
        // Non-tabled shape (lm_head) and wrong base → no 27B win.
        assert_eq!(geom_pick_q35tp("mul_mat_vec_q8_0deq_f32_f32", 5120, 248320), None);
        assert_eq!(geom_pick_q35tp("mul_mat_vec_bf16_f32_f32", 5120, 4352), None);
        // The q8_0deq geom names are reachable through matvec_variant_geom's core
        // (matvec_variant_core with quant=Q8_0 resolves to the q8_0deq base).
        assert_eq!(matvec_variant_core(QuantFormat::Q8_0, false, true, None, 8192).0,
                   "mul_mat_vec_q8_0deq_f32_f32_r8");
    }

    #[test]
    fn geom_pickers_fall_back_to_legacy_outside_win_region() {
        // G3 no-regression: everything outside the swept shapes is the exact
        // legacy string (byte-identical dispatch). lm_head-shaped (k=2048,
        // n=vocab), dense projections, TP shards, shared gate/up (legacy won
        // the sweep), dn a/b (noise-level win — kept legacy).
        assert_eq!(matvec_mlx4_variant_k(2048, 151936), matvec_mlx4_variant(151936));
        assert_eq!(matvec_mlx4_variant_k(2048, 1024), matvec_mlx4_variant(1024));
        assert_eq!(matvec_mlx4_variant_k(256, 2048), matvec_mlx4_variant(2048));
        assert_eq!(matvec_f32_variant_k(2048, 512), matvec_f32_variant(512)); // shared gate/up
        assert_eq!(matvec_f32_variant_k(2048, 1), matvec_f32_variant(1));
        assert_eq!(geom_pick("mul_mat_vec_bf16_f32_f32", 2048, 32), None);    // dn a/b
        assert_eq!(geom_pick("mul_mat_vec_q8_0deq_f32_f32", 2048, 32), None); // dn a/b (q8)
        assert_eq!(geom_pick("mul_mat_vec_q8_0deq_f32_f32", 2048, 248320), None); // lm_head
        assert_eq!(geom_pick("mul_mat_vec_nvfp4_f32_f32", 2048, 512), None);  // unswept base
        // A VLLM_VULKAN_MATVEC_ROWS override forces the legacy `_r{v}` path.
        assert_eq!(geom_name("mul_mat_vec_mlx4_f32_f32", 2048, 512, Some(4)), None);
        // matvec_variant_geom leaves non-bf16 quant formats untouched.
        assert_eq!(matvec_variant_geom(true, 2048, 4096), matvec_variant(true, 4096));
    }

    #[test]
    fn geom_pick_combos_compiled() {
        // Every geom_pick winner must be a compiled combo, within the
        // GFX1013-safe envelope (rows<=8, LDS rows*bs*4 <= 16 KB), and only
        // the swept bases (mlx4/bf16/f32/q8_0deq) may have combos at all.
        let shapes: &[(&str, usize, usize)] = &[
            ("mul_mat_vec_mlx4_f32_f32", 2048, 512),
            ("mul_mat_vec_mlx4_f32_f32", 512, 2048),
            ("mul_mat_vec_bf16_f32_f32", 2048, 8192),
            ("mul_mat_vec_bf16_f32_f32", 2048, 4096),
            ("mul_mat_vec_bf16_f32_f32", 4096, 2048),
            ("mul_mat_vec_f32_f32_f32", 512, 2048),
            ("mul_mat_vec_f32_f32_f32", 2048, 256),
            ("mul_mat_vec_q8_0deq_f32_f32", 2048, 8192),
            ("mul_mat_vec_q8_0deq_f32_f32", 2048, 4096),
            ("mul_mat_vec_q8_0deq_f32_f32", 2048, 512),
            ("mul_mat_vec_q8_0deq_f32_f32", 4096, 2048),
            ("mul_mat_vec_nvfp4_f32_f32", 1024, 1280),
            ("mul_mat_vec_nvfp4_f32_f32", 1280, 1024),
            ("mul_mat_vec_fp8_f32_f32", 4096, 5376),
            ("mul_mat_vec_fp8_f32_f32", 5376, 4096),
            ("mul_mat_vec_f16_f32_f32", 4096, 1024),
            ("mul_mat_vec_f16_f32_f32", 4096, 4096),
        ];
        // Qwen3.6-27B TP-4 per-rank q8 shapes (flag-gated geom_pick_q35tp) — must
        // ALSO reuse only compiled q8_0deq combos so no new pipeline is named.
        let q35tp_shapes: &[(&str, usize, usize)] = &[
            ("mul_mat_vec_q8_0deq_f32_f32", 5120, 4352),
            ("mul_mat_vec_q8_0deq_f32_f32", 5120, 3072),
            ("mul_mat_vec_q8_0deq_f32_f32", 5120, 2560),
            ("mul_mat_vec_q8_0deq_f32_f32", 5120, 1536),
            ("mul_mat_vec_q8_0deq_f32_f32", 5120, 256),
            ("mul_mat_vec_q8_0deq_f32_f32", 1536, 5120),
            ("mul_mat_vec_q8_0deq_f32_f32", 4352, 5120),
        ];
        for &(base, k, n) in shapes {
            let (bs, r) = geom_pick(base, k, n).expect("swept shape must have a winner");
            assert!(geom_combos_for(base).contains(&(bs, r)),
                "{base} ({k},{n}) winner ({bs},{r}) not in compiled combos");
            assert!(matches!(r, 1 | 2 | 4 | 8) && r * bs * 4 <= 16384 && bs.is_power_of_two());
        }
        for &(base, k, n) in q35tp_shapes {
            let (bs, r) = geom_pick_q35tp(base, k, n).expect("27B TP shape must have a winner");
            assert!(geom_combos_for(base).contains(&(bs, r)),
                "{base} ({k},{n}) 27B-TP winner ({bs},{r}) not in compiled combos");
            assert!(matches!(r, 1 | 2 | 4 | 8) && r * bs * 4 <= 16384 && bs.is_power_of_two());
        }
        assert_eq!(geom_combos_for("mul_mat_vec_nvfp4_f32_f32"), &[(256, 2), (128, 1)]);
        assert_eq!(geom_combos_for("mul_mat_vec_fp8_f32_f32"), &[(512, 4), (256, 1)]);
        assert_eq!(geom_combos_for("mul_mat_vec_f16_f32_f32"), &[(512, 2), (256, 2)]);
    }

    #[test]
    fn nvfp4_fp8_f16_k_pickers_are_byte_identical_to_legacy() {
        // Outside the swept win region, every `_k` wrapper must resolve to the
        // exact legacy `matvec_*_variant(n)` dispatch (name + rows) for
        // representative Nemotron shapes.
        assert_eq!(matvec_nvfp4_variant_k(2048, 512), matvec_nvfp4_variant(512));
        assert_eq!(matvec_nvfp4_variant_k(4096, 2048), matvec_nvfp4_variant(2048));
        assert_eq!(matvec_fp8_variant_k(2048, 8192), matvec_fp8_variant(8192));
        assert_eq!(matvec_f16_variant_k(2048, 4096), matvec_f16_variant(4096));
    }

    #[test]
    fn nvfp4_fp8_f16_k_pickers_return_swept_winners() {
        // NEW-2 Phase 2 cluster-sweep winners (cos=1.0, no pipeline hang).
        assert_eq!(matvec_nvfp4_variant_k(1024, 1280),
                   ("mul_mat_vec_nvfp4_f32_f32_bs256_r2".to_string(), 2)); // MoE expert up
        assert_eq!(matvec_nvfp4_variant_k(1280, 1024),
                   ("mul_mat_vec_nvfp4_f32_f32_bs128_r1".to_string(), 1)); // MoE expert down
        assert_eq!(matvec_fp8_variant_k(4096, 5376),
                   ("mul_mat_vec_fp8_f32_f32_bs512_r4".to_string(), 4)); // shared up
        assert_eq!(matvec_fp8_variant_k(5376, 4096),
                   ("mul_mat_vec_fp8_f32_f32_bs256_r1".to_string(), 1)); // shared down
        assert_eq!(matvec_f16_variant_k(4096, 1024),
                   ("mul_mat_vec_f16_f32_f32_bs512_r2".to_string(), 2)); // fc1
        assert_eq!(matvec_f16_variant_k(4096, 4096),
                   ("mul_mat_vec_f16_f32_f32_bs256_r2".to_string(), 2)); // q_proj / o_proj
    }

    #[test]
    fn q8_0_variant_k_mamba_shapes_hit_legacy_r8() {
        // The mamba in_proj/out_proj shapes (Nemotron q8_0 requant route) are
        // NOT in the geom_pick win table, so both must fall back to the
        // legacy PINNED q8_0deq base at r8 (n>=1024 in both cases).
        assert_eq!(matvec_q8_0_variant_k(4096, 18048),
                   ("mul_mat_vec_q8_0deq_f32_f32_r8".to_string(), 8)); // mamba in_proj
        assert_eq!(matvec_q8_0_variant_k(8192, 4096),
                   ("mul_mat_vec_q8_0deq_f32_f32_r8".to_string(), 8)); // mamba out_proj
    }

    #[test]
    fn gemm_picker_returns_swept_winner_for_priority_shapes() {
        // GEMM campaign Phase 1 (cluster sweep 2026-07-10, pinned 1850MHz).
        assert_eq!(gemm_variant_k(2048, 8192),
                   ("matmul_f16_f32_f16_aligned_bm64_bn32_w32_tm2_tn4".to_string(), 64, 32)); // dn-qkv
        assert_eq!(gemm_variant_k(4096, 2048),
                   ("matmul_f16_f32_f16_aligned_bm32_bn32_w64_tm4_tn2".to_string(), 32, 32)); // dn-out
        assert_eq!(gemm_variant_k(2048, 512),
                   ("matmul_f16_f32_f16_aligned_bm32_bn64_w64_tm4_tn2".to_string(), 32, 64)); // gate-up
        assert_eq!(gemm_variant_k(2048, 256),
                   ("matmul_f16_f32_f16_aligned_bm32_bn32_w64_tm4_tn2".to_string(), 32, 32)); // router
        assert_eq!(gemm_variant_k(2048, 151936),
                   ("matmul_f16_f32_f16_aligned_bm32_bn32_w32_tm4_tn4".to_string(), 32, 32)); // lm_head
    }

    #[test]
    fn gemm_picker_reaches_f16_aligned_for_gemma_projection_shapes() {
        // M1 flip: forward_prefill_gemma's gpu_gemm shapes must hit the
        // f16-aligned GEMM (a compiled combo), not the LEGACY fallback,
        // now that gemm_pick has gemma entries. Each expected geometry is
        // reused verbatim from an existing (qwen-sweep-derived) combo — see
        // the gemm_pick doc comment for the shape-class reasoning.
        let aligned = |g: GemmGeom| {
            let (bm, bn, _w, _tm, _tn) = g;
            (gemm_geom_name("matmul_f16_f32_f16_aligned", g), bm, bn)
        };
        assert_eq!(gemm_variant_k(1536, 2048), aligned((32, 32, 64, 4, 2))); // q_proj (sliding)
        assert_eq!(gemm_variant_k(1536, 4096), aligned((64, 32, 32, 2, 4))); // q_proj (full)
        assert_eq!(gemm_variant_k(1536, 256), aligned((32, 32, 64, 4, 2)));  // k/v_proj (sliding) / ple gate
        assert_eq!(gemm_variant_k(1536, 512), aligned((32, 64, 64, 4, 2)));  // k/v_proj (full)
        assert_eq!(gemm_variant_k(2048, 1536), aligned((32, 32, 64, 4, 2))); // o_proj (sliding)
        assert_eq!(gemm_variant_k(4096, 1536), aligned((32, 32, 64, 4, 2))); // o_proj (full)
        assert_eq!(gemm_variant_k(1536, 6144), aligned((64, 32, 32, 2, 4))); // gate/up_proj
        assert_eq!(gemm_variant_k(1536, 12288), aligned((64, 32, 32, 2, 4))); // gate/up_proj (kv-shared)
        assert_eq!(gemm_variant_k(6144, 1536), aligned((32, 32, 64, 4, 2))); // down_proj
        assert_eq!(gemm_variant_k(12288, 1536), aligned((32, 32, 64, 4, 2))); // down_proj (kv-shared)
        assert_eq!(gemm_variant_k(256, 1536), aligned((32, 32, 64, 4, 2)));  // per_layer_projection
        // None of these are the LEGACY fallback.
        for &(k, n) in &[(1536, 2048), (1536, 4096), (1536, 256), (1536, 512),
                         (2048, 1536), (4096, 1536), (1536, 6144), (1536, 12288),
                         (6144, 1536), (12288, 1536), (256, 1536)] {
            let (variant, _, _) = gemm_variant_k(k, n);
            assert_ne!(variant, "matmul_f16_f32_fp32",
                "gemma shape ({k},{n}) fell back to LEGACY — M1 flip did not reach it");
        }
    }

    #[test]
    fn gemm_picker_falls_back_to_legacy_outside_win_region() {
        // G3 no-regression: any shape outside the swept set is the exact
        // byte-identical legacy dispatch (matmul_f16_f32_fp32, BM=BN=64).
        assert_eq!(gemm_pick(2048, 1024), None);
        assert_eq!(gemm_pick(5120, 5120), None); // 27B dense hidden (not yet swept)
        assert_eq!(gemm_variant_k(2048, 1024),
                   ("matmul_f16_f32_fp32".to_string(), 64, 64));
    }

    #[test]
    fn gemm_picker_respects_kill_switch() {
        // VLLM_VULKAN_GEMM_F16ALIGNED=0 must revert every swept shape to the
        // legacy kernel unconditionally. `gemm_f16aligned_flag` delegates to
        // the process-wide `flags_global()` OnceLock (read once, like every
        // other flag here), so exercise `Flags::from_env()` directly instead
        // of the cached accessor — mirrors the flags.rs field doc, not a live
        // toggle of the singleton.
        std::env::set_var("VLLM_VULKAN_GEMM_F16ALIGNED", "0");
        assert!(!flags::Flags::from_env().gemm_f16aligned);
        std::env::remove_var("VLLM_VULKAN_GEMM_F16ALIGNED");
        assert!(flags::Flags::from_env().gemm_f16aligned); // default ON
    }

    #[test]
    fn gemm_pick_combos_compiled() {
        // Every gemm_pick winner must be a compiled GEMM_GEOM_COMBOS sibling,
        // and every combo must be BLOCK_SIZE<=1024 / integer-WNITER internally
        // consistent (WM=WN=32, WMITER=2 fixed — see compile_mul_mm_geom).
        let shapes: &[(usize, usize)] = &[
            (2048, 8192), (4096, 2048), (2048, 512), (2048, 256), (2048, 151936),
            // Gemma4-E2B (M1 flip) additions — see gemm_pick doc comment.
            (1536, 2048), (1536, 4096), (1536, 256), (1536, 512),
            (2048, 1536), (4096, 1536), (1536, 6144), (1536, 12288),
            (6144, 1536), (12288, 1536), (256, 1536),
        ];
        for &(k, n) in shapes {
            let g = gemm_pick(k, n).expect("swept shape must have a winner");
            assert!(GEMM_GEOM_COMBOS.contains(&g),
                "({k},{n}) winner {g:?} not in compiled GEMM_GEOM_COMBOS");
        }
        const WM: u32 = 32;
        const WN: u32 = 32;
        const WMITER: u32 = 2;
        for &(bm, bn, warp, tm, tn) in GEMM_GEOM_COMBOS {
            let block_size = (bm / WM) * (bn / WN) * warp;
            assert!(block_size <= 1024, "combo {:?} exceeds BLOCK_SIZE<=1024", (bm, bn, warp, tm, tn));
            assert_eq!((WM * WN) % (warp * tm * tn * WMITER), 0,
                "combo {:?} gives non-integer WNITER", (bm, bn, warp, tm, tn));
        }
    }

    #[test]
    fn fp8_tp_byte_shard_matches_f32_shard_indices() {
        // Cfg-free equivalence: q35_tp_shard::<u8> and q35_tp_shard::<f32> must pick
        // the SAME index ranges for the same weight name — verified by shading an
        // index-array (value == flat index) through both instantiations and
        // checking the returned indices line up element-for-element. Covers both
        // the col-shard (gate_proj) and row-shard (down_proj) arms used by FP8
        // attention's generic byte sharder.
        use qwen35::LayerType::{FullAttention, LinearAttention};
        let n = 2usize;
        let cfg = qwen35::Qwen35Config {
            hidden_size: 8, num_hidden_layers: 1, vocab_size: 16, rms_norm_eps: 1e-6,
            tie_word_embeddings: true, num_attention_heads: 2 * n, num_key_value_heads: 1 * n,
            head_dim: 4, attn_output_gate: true, rope_theta: 1e7, partial_rotary_factor: 0.25,
            linear_num_key_heads: 2 * n, linear_num_value_heads: 4 * n,
            linear_key_head_dim: 4, linear_value_head_dim: 4, linear_conv_kernel_dim: 4,
            intermediate_size: 16 * n, num_experts: 0, num_experts_per_tok: 0,
            moe_intermediate_size: 0, shared_expert_intermediate_size: 0,
            layer_types: vec![LinearAttention, FullAttention],
        };
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        // col-shard: mlp.gate_proj [inter, h] (col-shards output rows).
        let gate_u8: Vec<u8> = (0..inter*h).map(|i| (i % 256) as u8).collect();
        let gate_f32: Vec<f32> = (0..inter*h).map(|i| i as f32).collect();
        for r in 0..n {
            let su8 = q35_tp_shard::<u8>("model.layers.0.mlp.gate_proj.weight", gate_u8.clone(), &cfg, r, n);
            let sf32 = q35_tp_shard::<f32>("model.layers.0.mlp.gate_proj.weight", gate_f32.clone(), &cfg, r, n);
            assert_eq!(su8.len(), sf32.len());
            for (a, b) in su8.iter().zip(sf32.iter()) {
                assert_eq!(*a as f32 % 256.0, *b % 256.0, "col-shard index mismatch rank {r}");
            }
        }

        // row-shard: mlp.down_proj [h, inter] (row-shards input cols).
        let down_u8: Vec<u8> = (0..h*inter).map(|i| (i % 256) as u8).collect();
        let down_f32: Vec<f32> = (0..h*inter).map(|i| i as f32).collect();
        for r in 0..n {
            let su8 = q35_tp_shard::<u8>("model.layers.0.mlp.down_proj.weight", down_u8.clone(), &cfg, r, n);
            let sf32 = q35_tp_shard::<f32>("model.layers.0.mlp.down_proj.weight", down_f32.clone(), &cfg, r, n);
            assert_eq!(su8.len(), sf32.len());
            for (a, b) in su8.iter().zip(sf32.iter()) {
                assert_eq!(*a as f32 % 256.0, *b % 256.0, "row-shard index mismatch rank {r}");
            }
        }
    }

    // ── VLLM_VULKAN_MOE_F16_SCALES: overflow guard + tiny-tail acceptance +
    //    CPU bit-exactness on the normal-range values ──

    #[test]
    fn moe_f16_scales_roundtrip_and_overflow_guard() {
        // Normal-range affine scales are bf16 on disk (7 mantissa bits); f16 has 10,
        // so every normal-range bf16 value survives f32→f16 bit-for-bit.
        let bf = |x: f32| half::bf16::from_f32(x).to_f32();
        let scales: Vec<f32> = [0.0123f32, -0.5, 1.0, 63.5, 0.001953125, -12.0]
            .iter().map(|&v| bf(v)).collect();
        let bytes = f32_scales_to_f16_bytes_safe(&scales)
            .expect("normal-range bf16 scales must convert exactly");
        assert_eq!(bytes.len(), scales.len() * 2);
        for (i, &v) in scales.iter().enumerate() {
            let h = half::f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
            assert_eq!(h.to_f32().to_bits(), v.to_bits(), "scale {i} not bit-exact in f16");
        }
        // Tiny subnormal tail (measured on 122B switch tensors, ~3e-7): ACCEPTED,
        // rounds to a finite near-value with negligible absolute error — NOT
        // rejected (rejecting the tail is what OOMed the 122B load).
        let tiny = 3.0e-7f32;
        let tb = f32_scales_to_f16_bytes_safe(&[tiny]).expect("tiny subnormal must be accepted");
        let tr = half::f16::from_le_bytes([tb[0], tb[1]]).to_f32();
        assert!(tr.is_finite() && (tr - tiny).abs() < 1.0e-7, "tiny subnormal rounds negligibly, got {tr}");
        // Underflow-to-zero (1e-9 < min subnormal) is also accepted (→0, negligible).
        assert!(f32_scales_to_f16_bytes_safe(&[1.0e-9]).is_some(), "underflow-to-0 accepted (negligible)");
        // Overflow guard: |x|>65504 → Inf → REJECT (would corrupt); NaN/Inf → REJECT.
        assert!(f32_scales_to_f16_bytes_safe(&[70000.0]).is_none(), "overflow must be rejected");
        assert!(f32_scales_to_f16_bytes_safe(&[f32::INFINITY]).is_none());
        assert!(f32_scales_to_f16_bytes_safe(&[f32::NAN]).is_none());
    }

    #[test]
    fn moe_f16_scales_dequant_bit_exact() {
        // Full CPU-path bit-exactness on NORMAL-range values: dequantizing a routed
        // expert with f16-stored scales/biases yields IDENTICAL f32 weights to the
        // f32 oracle — the numerical claim the resident f16 buffer relies on for
        // the bulk (O(1e-2..1e0)) of the affine scales.
        use crate::moe::QuantSwitch;
        let (e, out, inf, gs, bits) = (2usize, 4usize, 128usize, 64usize, 4usize);
        let groups = inf / gs;
        let per_word = 32 / bits;
        let words_per_row = inf / per_word;
        let mut rng = 0x1234_5678u32;
        let mut next = || { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; rng };
        let packed: Vec<u32> = (0..e * out * words_per_row).map(|_| next()).collect();
        let bf = |x: f32| half::bf16::from_f32(x).to_f32();
        let scales: Vec<f32> = (0..e * out * groups)
            .map(|_| bf((next() as f32 / u32::MAX as f32) * 0.1)).collect();
        let biases: Vec<f32> = (0..e * out * groups)
            .map(|_| bf(((next() as f32 / u32::MAX as f32) - 0.5) * 0.02)).collect();
        let f16rt = |v: &[f32]| -> Vec<f32> {
            let b = f32_scales_to_f16_bytes_safe(v).expect("safe for bf16-sourced scales");
            b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
        };
        let qs_f32 = QuantSwitch {
            packed: packed.clone(), scales: scales.clone(), biases: biases.clone(),
            out_features: out, in_features: inf, group_size: gs, bits,
        };
        let qs_f16 = QuantSwitch {
            packed, scales: f16rt(&scales), biases: f16rt(&biases),
            out_features: out, in_features: inf, group_size: gs, bits,
        };
        for expert in 0..e {
            assert_eq!(
                qs_f32.dequant_expert(expert), qs_f16.dequant_expert(expert),
                "f16-scale dequant must be bit-identical to f32 (expert {expert})"
            );
        }
    }
}
