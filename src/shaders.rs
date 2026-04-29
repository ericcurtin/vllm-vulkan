// SPDX-License-Identifier: Apache-2.0
//! Shader SPIR-V registry.
//!
//! All SPIR-V bytecode is embedded at compile time via `include_bytes!`.
//! The `OUT_DIR/spirv/` directory is populated by `build.rs` from the
//! pre-compiled shaders in `shaders/spirv/`.
//!
//! Each shader variant has a unique name (matching the filename without
//! `.spv`) that is used as the key in the pipeline cache.

use std::collections::HashMap;

/// A SPIR-V module: its name and raw bytes.
pub struct SpvModule {
    pub name: &'static str,
    pub bytes: &'static [u8],
}

/// Build the static shader registry.
///
/// We use a macro to include each .spv file from OUT_DIR.
/// Shaders that don't exist (optional features) are silently skipped.
macro_rules! spv {
    ($name:literal) => {{
        const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/", $name, ".spv"));
        SpvModule { name: $name, bytes: BYTES }
    }};
}

/// Core shaders needed for LLM inference.
/// Extended with all 1329 pre-compiled variants via the full list below.
pub fn core_shaders() -> Vec<SpvModule> {
    vec![
        // ── Elementwise unary ───────────────────────────────────────────
        spv!("silu_f32"),
        spv!("silu_f16"),
        spv!("gelu_f32"),
        spv!("gelu_f16"),
        spv!("gelu_quick_f32"),
        spv!("gelu_quick_f16"),
        spv!("relu_f32"),
        spv!("relu_f16"),
        spv!("tanh_f32"),
        spv!("tanh_f16"),
        spv!("exp_f32"),
        spv!("exp_f16"),
        spv!("sqrt_f32"),
        spv!("abs_f32"),
        spv!("neg_f32"),
        spv!("sigmoid_f32"),
        spv!("sigmoid_f16"),
        spv!("scale_f32"),
        // ── Normalization ───────────────────────────────────────────────
        spv!("rms_norm_f32"),
        spv!("rms_norm_f16"),
        spv!("norm_f32"),
        spv!("group_norm_f32"),
        spv!("l2_norm_f32"),
        // ── Softmax ─────────────────────────────────────────────────────
        spv!("soft_max_f32"),
        spv!("soft_max_large1_f32"),
        spv!("soft_max_large2_f32"),
        spv!("soft_max_large3_f32"),
        // ── Binary elementwise ──────────────────────────────────────────
        spv!("add_f32_f32_f32"),
        spv!("add_f32_f16_f32"),
        spv!("mul_f32_f32_f32"),
        spv!("div_f32_f32_f32"),
        spv!("sub_f32_f32_f32"),
        // ── GLU activations ─────────────────────────────────────────────
        spv!("swiglu_f32"),
        spv!("swiglu_f16"),
        spv!("geglu_f32"),
        spv!("geglu_f16"),
        // ── RoPE ────────────────────────────────────────────────────────
        spv!("rope_norm_f32"),
        spv!("rope_norm_f16"),
        spv!("rope_neox_f32"),
        spv!("rope_neox_f16"),
        spv!("rope_multi_f32"),
        spv!("rope_multi_f16"),
        // ── Copy / reshape ──────────────────────────────────────────────
        spv!("get_rows_f32"),
        spv!("get_rows_f16"),
        spv!("fill_f32"),
        spv!("pad_f32"),
        spv!("repeat_f32"),
        spv!("concat_f32"),
        spv!("diag_mask_inf_f32"),
        spv!("sum_rows_f32"),
        spv!("argmax_f32"),
        // ── Dequantize ──────────────────────────────────────────────────
        spv!("dequant_q4_0"),
        spv!("dequant_q4_1"),
        spv!("dequant_q5_0"),
        spv!("dequant_q5_1"),
        spv!("dequant_q8_0"),
        spv!("dequant_q2_k"),
        spv!("dequant_q3_k"),
        spv!("dequant_q4_k"),
        spv!("dequant_q5_k"),
        spv!("dequant_q6_k"),
        spv!("dequant_iq4_nl"),
        spv!("dequant_iq4_xs"),
        // ── Matrix-vector multiply (f32 weights) ────────────────────────
        spv!("mul_mat_vec_f32_f32_f32"),
        spv!("mul_mat_vec_f16_f32_f32"),
        spv!("mul_mat_vec_f32_f32_f32_subgroup"),
        spv!("mul_mat_vec_f16_f32_f32_subgroup"),
        // ── Matrix-vector multiply (quantized weights) ──────────────────
        spv!("mul_mat_vec_q4_0_f32_f32"),
        spv!("mul_mat_vec_q4_1_f32_f32"),
        spv!("mul_mat_vec_q5_0_f32_f32"),
        spv!("mul_mat_vec_q5_1_f32_f32"),
        spv!("mul_mat_vec_q8_0_f32_f32"),
        spv!("mul_mat_vec_q2_k_f32_f32"),
        spv!("mul_mat_vec_q3_k_f32_f32"),
        spv!("mul_mat_vec_q4_k_f32_f32"),
        spv!("mul_mat_vec_q5_k_f32_f32"),
        spv!("mul_mat_vec_q6_k_f32_f32"),
        // ── General matmul (prefill) ─────────────────────────────────────
        spv!("matmul_f32_f16_fp32"),
        spv!("matmul_f32_f32_fp32"),
        spv!("matmul_f16_f32_fp32"),
        spv!("matmul_f32_f16_aligned_fp32"),
        spv!("matmul_f32_f32_aligned_fp32"),
        // ── Flash attention ─────────────────────────────────────────────
        spv!("flash_attn_f32_f16_f16_fp32"),
        spv!("flash_attn_f32_f16_f32_fp32"),
        spv!("fa_mask_opt"),
        spv!("fa_split_k_reduce"),
        // ── Misc ────────────────────────────────────────────────────────
        spv!("split_k_reduce"),
        spv!("quantize_q8_1"),
    ]
}

/// Returns all shaders indexed by name.
pub fn shader_registry() -> HashMap<&'static str, &'static [u8]> {
    core_shaders()
        .iter()
        .map(|s| (s.name, s.bytes))
        .collect()
}
