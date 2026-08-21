#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# shellcheck disable=SC2086  # intentional word-splitting of define strings
#
# Compile all GLSL compute shaders to SPIR-V.
#
# Usage: compile_shaders.sh [output_dir]
#   output_dir defaults to shaders/spirv/
#
# Requires glslangValidator (from the glslang package).
# Ubuntu:  sudo apt-get install -y glslang-tools
# macOS:   brew install glslang

set -eu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHADER_DIR="${REPO_DIR}/shaders"
OUT_DIR="${1:-${SHADER_DIR}/spirv}"

mkdir -p "${OUT_DIR}"

# ── Find compiler ──────────────────────────────────────────────────────────────

find_compiler() {
  for cmd in glslangValidator \
             /usr/bin/glslangValidator \
             /usr/local/bin/glslangValidator \
             /opt/homebrew/bin/glslangValidator; do
    if command -v "$cmd" &>/dev/null 2>&1; then
      echo "$cmd"; return 0
    fi
  done
  return 1
}

GLSLANG="$(find_compiler)" || {
  echo "Error: glslangValidator not found." >&2
  echo "  Ubuntu/Debian: sudo apt-get install -y glslang-tools" >&2
  echo "  macOS:         brew install glslang" >&2
  exit 1
}

GLSLANG_VERSION="$(${GLSLANG} --version 2>&1 | head -1)"
echo "Using compiler: ${GLSLANG} (${GLSLANG_VERSION})"

# Detect if glslang knows about GL_EXT_integer_dot_product.
# It was added around v14.0; Ubuntu 24.04's system package (v15.1.0 from
# Ubuntu repos) is compiled without it. We detect this and pre-patch shaders
# that use it by rewriting 'require' to 'enable' in a temp file.
NEEDS_DOT_PRODUCT_PATCH=0
_probe="$(mktemp /tmp/probe-XXXXXX.comp)"
printf '#version 450\n#extension GL_EXT_integer_dot_product : require\nvoid main() {}\n' \
  > "${_probe}"
if ! "${GLSLANG}" --target-env vulkan1.3 -S comp -o /dev/null "${_probe}" &>/dev/null; then
  NEEDS_DOT_PRODUCT_PATCH=1
  echo "  (note: GL_EXT_integer_dot_product/dotPacked4x8EXT not supported; emulating)"
fi
rm -f "${_probe}"

# ── Compile helper ─────────────────────────────────────────────────────────────
#
# compile <output_stem> <source.comp> [KEY=VALUE ...]
#
# The -P preamble enables GL_GOOGLE_include_directive so #include works.

FAILED=0
TOTAL=0
SKIPPED=0

compile() {
  local out_stem="$1"; shift
  local src="$1";      shift
  local out="${OUT_DIR}/${out_stem}.spv"
  local actual_src="${SHADER_DIR}/${src}"

  # Skip cleanly when the source shader isn't present. The crate is built as
  # per-feature slices (foundation + one model at a time): each slice ships only
  # the shaders it uses, so this one script can drive any slice — it compiles
  # what's present and skips the rest (build.rs derives the registry from the
  # compiled output, so a skipped shader is simply absent from the registry).
  if [ ! -f "${actual_src}" ]; then
    echo "skip (source absent): ${src}"
    SKIPPED=$(( SKIPPED + 1 ))
    return 0
  fi

  TOTAL=$(( TOTAL + 1 ))

  # If glslang doesn't support GL_EXT_integer_dot_product as 'require',
  # create a patched temp file with 'enable' instead.
  local tmp_src=""
  if [ "${NEEDS_DOT_PRODUCT_PATCH}" -eq 1 ] && \
     grep -q "GL_EXT_integer_dot_product" "${actual_src}" 2>/dev/null; then
    # Glslang lacks GL_EXT_integer_dot_product support. Patch the shader:
    # 1. Remove the unsupported extension directive.
    # 2. Prepend a software emulation of dotPacked4x8EXT.
    tmp_src="$(mktemp /tmp/shader-XXXXXX.comp)"
    # Extract the #version line from the original file
    local ver_line
    ver_line="$(head -1 "${actual_src}")"
    {
      printf '%s\n' "${ver_line}"
      # Software emulation of dotPacked4x8EXT as a GLSL macro.
      # Only injected when the compiler lacks GL_EXT_integer_dot_product support.
      # Using a macro (not a function) avoids conflicts if the compiler knows
      # dotPacked4x8EXT as a built-in.
      cat <<'GLSL_COMPAT'
// Emulate GL_EXT_integer_dot_product for compilers that lack native support.
// dotPacked4x8EXT(a,b) = dot product of 4 signed 8-bit integers packed in uint.
#define dotPacked4x8EXT(a,b) ( \
    int((a)         << 24 >> 24) * int((b)         << 24 >> 24) + \
    int(((a) >>  8u) << 24 >> 24) * int(((b) >>  8u) << 24 >> 24) + \
    int(((a) >> 16u) << 24 >> 24) * int(((b) >> 16u) << 24 >> 24) + \
    int(((a) >> 24u) << 24 >> 24) * int(((b) >> 24u) << 24 >> 24) )
GLSL_COMPAT
      # Skip the first line (#version) and remove the problematic extension require
      tail -n +2 "${actual_src}" \
        | sed 's/#extension GL_EXT_integer_dot_product.*//g'
    } > "${tmp_src}"
    actual_src="${tmp_src}"
  fi

  local args=(
    --target-env vulkan1.3
    -S comp
    "-I${SHADER_DIR}"
    "-P#extension GL_GOOGLE_include_directive : enable"
  )
  for def in "$@"; do
    args+=( "-D${def}" )
  done
  args+=( -o "${out}" "${actual_src}" )

  if "${GLSLANG}" "${args[@]}" 2>/dev/null; then
    echo "  OK   ${out_stem}"
  else
    echo "  FAIL ${out_stem}" >&2
    "${GLSLANG}" "${args[@]}" 2>&1 | sed 's/^/        /' || true
    FAILED=$(( FAILED + 1 ))
  fi

  if [ -n "${tmp_src}" ]; then rm -f "${tmp_src}"; fi
}

# ── Common define sets ─────────────────────────────────────────────────────────
#
# D_TYPE, A_TYPE, B_TYPE, FLOAT_TYPE, FLOAT_TYPEV2/V4, ACC_TYPE/V2 are purely
# external — none of the GLSL files define them; they must be injected here.
#
# Key findings from reading the shader sources:
#  - types.glsl defines A_TYPE from DATA_A_* but only when processed after
#    DATA_A_* is set. When shaders include generic_binary_head.glsl BEFORE
#    types.glsl, A_TYPE is undefined at that point. Always pass A_TYPE=<type>
#    explicitly alongside DATA_A_*=1.
#  - RMS_NORM_ROPE_FUSION=0 must always be set when not doing rope fusion
#    (generic_binary_head.glsl uses #if !RMS_NORM_ROPE_FUSION, not #ifndef).
#  - TEMP_TYPE: used in get_rows.comp for intermediate value; use float always
#    (even for f16 output) to avoid local float16_t requiring explicit_arithmetic
#    extension that isn't enabled in included files.
#  - FLOAT_TYPEV2 for k-quant matvec: use vec2 (not f16vec2) for local vars
#    to avoid needing GL_EXT_shader_explicit_arithmetic_types_float16 for
#    local variables (f16vec2 as local var needs that extension).

# Scalar unary (generic_head.glsl path — no B_TYPE needed)
U_F32="DATA_A_F32=1 A_TYPE=float D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0"
# Binary f32×f32→f32 (generic_binary_head.glsl path)
BIN_F32="DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0"
# Tiled matmul with f32 accum (no FLOAT16 path). Do NOT include A_TYPE here —
# types.glsl defines it from DATA_A_* and overriding it causes a redefinition error.
MM_F32="FLOAT_TYPE=float FLOAT_TYPEV2=vec2 FLOAT_TYPEV4=vec4 ACC_TYPE=float ACC_TYPEV2=vec2 D_TYPE=float"
# Tiled matmul with f16 internal FLOAT_TYPE (FLOAT16=1).
MM_F16="FLOAT16=1 FLOAT_TYPE=float16_t FLOAT_TYPEV2=f16vec2 FLOAT_TYPEV4=f16vec4 ACC_TYPE=float ACC_TYPEV2=vec2 D_TYPE=float"
# Flash-attn f32 accum
FA_F32="FLOAT_TYPE=float FLOAT_TYPEV2=vec2 FLOAT_TYPEV4=vec4 ACC_TYPE=float ACC_TYPEV2=vec2 ACC_TYPEV4=vec4 D_TYPEV4=vec4"

# ── Elementwise unary ──────────────────────────────────────────────────────────
echo ""
echo "=== Elementwise unary (f32→f32) ==="
# silu/gelu etc. use generic_head.glsl — A_TYPE is defined by types.glsl which
# is included after, so it resolves correctly without explicit A_TYPE. But pass
# it anyway for safety on strict parsers.
for op in silu gelu gelu_quick relu relu2 tanh exp sigmoid abs neg ceil; do
  compile "${op}_f32" "${op}.comp" ${U_F32}
done
compile "gelu_inplace_f32" "gelu.comp" ${U_F32} INPLACE=1

# ── Elementwise binary ─────────────────────────────────────────────────────────
echo ""
echo "=== Elementwise binary ==="
# NB: the _f32_f32_f32 binary variants are referenced by src/shaders.rs (spv!)
# but were previously not compiled here — a full shader rebuild would then fail
# the include_bytes! for the missing .spv. They are compiled below. add_f32_f32_f32
# is also the residual-add kernel used by the fused GPU-resident Qwen layer.
# (${BIN_F32} expands to the same flags upstream spelled out explicitly.)
compile "add_f32_f32_f32" "add.comp"  ${BIN_F32}
compile "add_f32_f32_f16" "add.comp"  DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "add_f32_f16_f32" "add.comp"  DATA_A_F32=1 A_TYPE=float DATA_B_F16=1 B_TYPE=float16_t D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "add_rms_f32_f32_f32" "multi_add.comp" ${BIN_F32}
compile "add_rms_f32_f32_f16" "multi_add.comp" DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "mul_f32_f32_f32" "mul.comp" ${BIN_F32}
compile "mul_f32_f32_f16" "mul.comp" DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "div_f32_f32_f32" "div.comp" ${BIN_F32}
compile "div_f32_f32_f16" "div.comp" DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "sub_f32_f32_f32" "sub.comp" ${BIN_F32}
compile "sub_f32_f32_f16" "sub.comp" DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0

# ── Normalization ──────────────────────────────────────────────────────────────
echo ""
echo "=== Normalization ==="
compile "rms_norm_f32" "rms_norm.comp" \
  DATA_A_F32=1 A_TYPE=float B_TYPE=float D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "rms_norm_mul_rope_f32_f32" "rms_norm.comp" \
  DATA_A_F32=1 A_TYPE=float B_TYPE=float D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=1 ROPE_D_TYPE=float
compile "rms_norm_mul_rope_f32_f16" "rms_norm.comp" \
  DATA_A_F32=1 A_TYPE=float B_TYPE=float D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=1 ROPE_D_TYPE=float16_t

# ── Softmax ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Softmax ==="
for suffix in "" "_large1" "_large2" "_large3"; do
  compile "soft_max${suffix}_f32_f16" "soft_max${suffix}.comp" \
    DATA_A_F32=1 A_TYPE=float DATA_B_F16=1 B_TYPE=float16_t D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
done

# ── GLU activations ────────────────────────────────────────────────────────────
echo ""
echo "=== GLU activations ==="
# glu_head.glsl uses A_TYPE for both inputs; D_TYPE for output. No B_TYPE.
compile "swiglu_f32" "swiglu.comp" DATA_A_F32=1 A_TYPE=float D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "geglu_f32"  "geglu.comp"  DATA_A_F32=1 A_TYPE=float D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
# DSV4 routed-expert SwiGLU with swiglu_limit clamp (silu(min(g,lim))*clamp(u,±lim),
# no GPT-OSS +1). Class Plain; Split mode 2 (separate gate/up buffers).
compile "dsv4_swiglu_clamp" "dsv4_swiglu_clamp.comp" DATA_A_F32=1 A_TYPE=float D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
# DSV4 manifold hyper-connection residual mix (self-contained; class Plain):
# out[k] = post[k]*sublayer + Σ_j comb[j,k]*streams[j].
compile "dsv4_hc_residual_mix" "dsv4_hc_residual_mix.comp"
# DSV4 M2b resident single-token MLA eager-attention softmax (sink + sliding
# window + compressed-KV block_bias + output-rope conjugate). Class Plain; one
# wave64 workgroup per head. Oracle dsv4_gpu::attention_layer_decode softmax core.
compile "dsv4_mla_softmax" "dsv4_mla_softmax.comp"

# Laguna per-head softplus attention gate (self-contained; class Plain).
compile "laguna_softplus_gate" "laguna_softplus_gate.comp"

# Laguna GPU decode-SDPA over the resident K/V planes (subgroup, wave64-only;
# skipped on non-64 devices in pipeline.rs, like paged_attn_decode_f32_sg).
compile "laguna_gpu_sdpa" "laguna_gpu_sdpa.comp"

# Laguna MoE router scoring + top-k on GPU (sigmoid + e_score_correction_bias +
# gate matvec + serial top-k); reads back only the top-k idx+weights. Self-
# contained; class Plain.
compile "laguna_router" "laguna_router.comp"

# ── RoPE ───────────────────────────────────────────────────────────────────────
echo ""
echo "=== RoPE ==="
compile "rope_norm_f32_f16" "rope_norm.comp" DATA_A_F32=1 A_TYPE=float  ROPE_D_TYPE=float16_t D_TYPE=float      FLOAT_TYPE=float      RMS_NORM_ROPE_FUSION=0
compile "rope_norm_f16"     "rope_norm.comp" DATA_A_F16=1 A_TYPE=float16_t ROPE_D_TYPE=float16_t D_TYPE=float16_t FLOAT_TYPE=float16_t RMS_NORM_ROPE_FUSION=0
compile "rope_neox_f32_f16" "rope_neox.comp" DATA_A_F32=1 A_TYPE=float  ROPE_D_TYPE=float16_t D_TYPE=float      FLOAT_TYPE=float      RMS_NORM_ROPE_FUSION=0
compile "rope_neox_f32_f32" "rope_neox.comp" DATA_A_F32=1 A_TYPE=float  ROPE_D_TYPE=float      D_TYPE=float      FLOAT_TYPE=float      RMS_NORM_ROPE_FUSION=0
compile "rope_neox_f16"     "rope_neox.comp" DATA_A_F16=1 A_TYPE=float16_t ROPE_D_TYPE=float16_t D_TYPE=float16_t FLOAT_TYPE=float16_t RMS_NORM_ROPE_FUSION=0
compile "rope_multi_f32_f16" "rope_multi.comp" DATA_A_F32=1 A_TYPE=float  ROPE_D_TYPE=float16_t D_TYPE=float      FLOAT_TYPE=float      RMS_NORM_ROPE_FUSION=0
compile "rope_multi_f16"     "rope_multi.comp" DATA_A_F16=1 A_TYPE=float16_t ROPE_D_TYPE=float16_t D_TYPE=float16_t FLOAT_TYPE=float16_t RMS_NORM_ROPE_FUSION=0

# ── Copy / reshape ─────────────────────────────────────────────────────────────
echo ""
echo "=== Copy / reshape ==="
# get_rows: generic_binary_head.glsl needs A_TYPE and B_TYPE (index type).
# TEMP_TYPE is used as the local intermediate — use float always (even for
# f16 output) to avoid needing GL_EXT_shader_explicit_arithmetic_types_float16
# for local variables in shaders that don't explicitly enable it.
compile "get_rows_f32_f32" "get_rows.comp" \
  DATA_A_F32=1 A_TYPE=float TEMP_TYPE=float B_TYPE=int D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "get_rows_f16" "get_rows.comp" \
  DATA_A_F16=1 A_TYPE=float16_t TEMP_TYPE=float B_TYPE=int D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
# fill only writes D; no A/B type needed
compile "fill_f16" "fill.comp" DATA_A_F16=1 A_TYPE=float16_t D_TYPE=float16_t FLOAT_TYPE=float16_t RMS_NORM_ROPE_FUSION=0
# concat: A_TYPE=f16, B_TYPE=f16, D_TYPE=f16
compile "concat_f16" "concat.comp" \
  DATA_A_F16=1 A_TYPE=float16_t B_TYPE=float16_t D_TYPE=float16_t FLOAT_TYPE=float16_t RMS_NORM_ROPE_FUSION=0
compile "contig_cpy_f32_f16" "contig_copy.comp" \
  DATA_A_F32=1 A_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "contig_cpy_f16_f32" "contig_copy.comp" \
  DATA_A_F16=1 A_TYPE=float16_t D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0

# ── Matrix-vector multiply (decode) ───────────────────────────────────────────
echo ""
echo "=== Matrix-vector multiply (decode) ==="
# F32/F16 — mul_mat_vec.comp; D_TYPE always float for matvec output
MMVEC_BASE="D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0"
compile "mul_mat_vec_f32_f32_f32" "mul_mat_vec.comp" \
  DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float ${MMVEC_BASE}
compile "mul_mat_vec_f32_f32_f32_subgroup" "mul_mat_vec.comp" \
  DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float ${MMVEC_BASE} USE_SUBGROUP_ADD_NO_SHMEM=1
compile "mul_mat_vec_f16_f32_f32" "mul_mat_vec.comp" \
  DATA_A_F16=1 A_TYPE=float16_t DATA_B_F32=1 B_TYPE=float ${MMVEC_BASE}
# bf16 weight matvec (VLLM_VULKAN_QUANT=bf16): weights loaded raw bf16 (native
# precision, no f32→f16 conversion churn on load). 2 bytes/weight like f16.
compile "mul_mat_vec_bf16_f32_f32" "mul_mat_vec.comp" \
  DATA_A_BF16=1 A_TYPE=uint16_t DATA_B_F32=1 B_TYPE=float ${MMVEC_BASE}
compile "mul_mat_vec_f16_f32_f32_subgroup" "mul_mat_vec.comp" \
  DATA_A_F16=1 A_TYPE=float16_t DATA_B_F32=1 B_TYPE=float ${MMVEC_BASE} USE_SUBGROUP_ADD_NO_SHMEM=1

# mul_mat_vecq.comp defines B_TYPE=block_q8_1_x4 internally — do NOT pass B_TYPE.
# Q4_1/Q5_1 use FLOAT_TYPEV2 for the packed dm field via mul_mat_vecq_funcs.glsl.
MMVECQ_BASE="DATA_B_F32=1 D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 RMS_NORM_ROPE_FUSION=0"
compile "mul_mat_vec_q4_0_f32_f32"          "mul_mat_vecq.comp" DATA_A_Q4_0=1   ${MMVECQ_BASE}
compile "mul_mat_vec_q4_0_f32_f32_subgroup"  "mul_mat_vecq.comp" DATA_A_Q4_0=1  ${MMVECQ_BASE} USE_SUBGROUP_ADD_NO_SHMEM=1
compile "mul_mat_vec_q4_1_f32_f32"          "mul_mat_vecq.comp" DATA_A_Q4_1=1   ${MMVECQ_BASE}
compile "mul_mat_vec_q5_0_f32_f32"          "mul_mat_vecq.comp" DATA_A_Q5_0=1   ${MMVECQ_BASE}
compile "mul_mat_vec_q5_1_f32_f32"          "mul_mat_vecq.comp" DATA_A_Q5_1=1   ${MMVECQ_BASE}
compile "mul_mat_vec_q8_0_f32_f32"          "mul_mat_vecq.comp" DATA_A_Q8_0=1   ${MMVECQ_BASE}
compile "mul_mat_vec_q8_0_f32_f32_subgroup"  "mul_mat_vecq.comp" DATA_A_Q8_0=1  ${MMVECQ_BASE} USE_SUBGROUP_ADD_NO_SHMEM=1
# iq4_nl is NOT a QUANT_LEGACY type — it needs mul_mat_vec.comp with B_TYPEV4
compile "mul_mat_vec_iq4_nl_f32_f32" "mul_mat_vec.comp" DATA_A_IQ4_NL=1 \
  DATA_B_F32=1 B_TYPE=float B_TYPEV4=vec4 \
  D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 RMS_NORM_ROPE_FUSION=0
# q8_0 DEQUANTIZE path (mul_mat_vec.comp, not the MMQ mul_mat_vecq.comp). The MMQ
# variant above expects the activation vector pre-quantized to q8_1 (block_q8_1_x4)
# on binding 1; our matvec dispatch uploads raw f32 activations, so we use the
# dequantize kernel instead (reads f32 B directly, dequantizes the q8_0 weight via
# get_dm/dequantize4). Needs B_TYPEV4 for the K_PER_ITER==8 vec4 B loads.
compile "mul_mat_vec_q8_0deq_f32_f32" "mul_mat_vec.comp" DATA_A_Q8_0=1 \
  DATA_B_F32=1 B_TYPE=float B_TYPEV4=vec4 \
  D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 RMS_NORM_ROPE_FUSION=0

# q4_0 dequantize kernel (VLLM_VULKAN_QUANT=q4_0): 4-bit weights, f32 activations.
# ~4x less weight bandwidth/memory vs f16. Same dequant-kernel approach as q8_0deq.
compile "mul_mat_vec_q4_0deq_f32_f32" "mul_mat_vec.comp" DATA_A_Q4_0=1 \
  DATA_B_F32=1 B_TYPE=float B_TYPEV4=vec4 \
  D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 RMS_NORM_ROPE_FUSION=0

# q4_K dequantize kernel (VLLM_VULKAN_QUANT=q4_k): per-sub-block 6-bit scale+min
# K-quant, much higher quality than q4_0 at the same ~4 bits. Uses the DEQUANT
# kernel from mul_mat_vec.comp (reads f32 B, dequantizes the q4_K weight via the
# DATA_A_Q4_K block in dequant_funcs.glsl: w = d*sc*q - dmin*m). NOT the
# mul_mat_vec_q4_k.comp MMQ kernel (that wants quantized activations → all-zero).
compile "mul_mat_vec_q4_kdeq_f32_f32" "mul_mat_vec.comp" DATA_A_Q4_K=1 \
  DATA_B_F32=1 B_TYPE=float B_TYPEV4=vec4 \
  D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 RMS_NORM_ROPE_FUSION=0

# MLX-affine 4-bit dequantize matvec (VLLM_VULKAN_MLX_GPU=1). Dedicated shader
# (mul_mat_vec_mlx4.comp), NOT mul_mat_vec.comp: MLX stores scales+biases in two
# separate buffers (not GGUF blocks), so it can't express the affine layout. The
# packed 4-bit weight (~1/8 of f32) stays resident on the GPU and is dequantized
# per-matvec, letting the 27B fit one 16GB UMA node. Mirrors
# model::dequantize_mlx_affine exactly. Uses the BLOCK_SIZE/NUM_ROWS spec
# constants (compile_matvec), so no -D type defines are needed.
compile "mul_mat_vec_mlx4_f32_f32" "mul_mat_vec_mlx4.comp"

# f16-resident affine scales/biases variant (VLLM_VULKAN_MOE_F16_SCALES, default
# OFF): identical kernel, but scales+biases are read as float16_t from a HALF-size
# resident MoE buffer. This is the 122B-A10B PP-6 fit-enabler (−~1.2GB/stage of
# expert-scale GTT) and a general MoE load-footprint win. f16 holds every in-range
# bf16-sourced affine scale EXACTLY (10 vs 7 mantissa bits) -> bit-identical to the
# f32 oracle; the host upload's exact round-trip guard rejects any out-of-range
# tensor (keeps f32). Gets the same _r2/_r4/_r8 spec-constant siblings as the base.
compile "mul_mat_vec_mlx4_f16scale_f32_f32" "mul_mat_vec_mlx4.comp" SCALE_F16=1

# mlx4 word-granular affine-factored siblings (perf/matvec-batch-dispatch):
# each thread owns whole 4-bit-packed word(s) instead of striding nibble-by-
# nibble, cutting VMEM issue 8x/16x and factoring the affine (q*s+b)*x sum into
# s*dot(q,x) + b*sum(x) so scale/bias load once per word(-pair) not per nibble.
# See shaders/mul_mat_vec_mlx4_w8.comp / _w16.comp for the math writeup.
compile "mul_mat_vec_mlx4w8_f32_f32"  "mul_mat_vec_mlx4_w8.comp"
compile "mul_mat_vec_mlx4w16_f32_f32" "mul_mat_vec_mlx4_w16.comp"

# w8sg: word-granular w8 load/dequant + subgroupAdd reduction (no LDS tree) for
# single-subgroup workgroups (BLOCK_SIZE<=64 on wave64 GFX1013). 1850MHz-clock
# re-sweep probe (registered but wired only where it clears the 25% decision
# bar vs the wired v1 winner per dispatch shape — see IMPROVEMENTS.md history).
compile "mul_mat_vec_mlx4w8sg_f32_f32" "mul_mat_vec_mlx4w8sg_f32_f32.comp"

# repack: VALU-bound repack refactor (VLLM_VULKAN_MLX4_REPACK, default OFF) for
# the DENSE decode mlx4 path (k%32==0). dwordx4 (uvec4) word-granular load = 32
# nibbles/load, scale/bias ONCE per 32-elem chunk, fma-factored affine +
# subgroupAdd reduction. Cuts the ~190-addr-VALU/16-unpack address-gen the
# gfx1013 ISA probe found in v1 (mul_mat_vec_mlx4.comp bs512/r8). Packed layout
# unchanged (contiguous-4-word-per-thread => coalescing order == existing order).
compile "mul_mat_vec_mlx4repack_f32_f32" "mul_mat_vec_mlx4repack_f32_f32.comp"

# repack + f16-RESIDENT scale (VLLM_VULKAN_MLX4_REPACK=1 AND VLLM_VULKAN_MOE_F16_SCALES=1).
# The SCALE_F16 sibling of the repack twin above: same dwordx4/subgroupAdd repack
# body, but the affine scales/biases are read float16_t (half the resident GTT —
# the 122B-A10B PP-8 fit-enabler). Routes the MoE routed-expert f16scale branch of
# matvec_mlx4_moe_variant_k OFF the address-gen-bound v1 f16scale kernel
# (mul_mat_vec_mlx4_f16scale_f32_f32, bs512) onto the repack roofline for the
# k>=1024/n>=1024 experts (122B gate/up k=3072 n=1024, down k=1024 n=3072). f16
# holds bf16 scales EXACTLY in range -> argmax-exact vs the f32 twin + v1 oracle.
compile "mul_mat_vec_mlx4repack_f16scale_f32_f32" "mul_mat_vec_mlx4repack_f32_f32.comp" SCALE_F16=1

# repack EXPERT-BATCHED (VLLM_VULKAN_LING_MOE_BATCH, default OFF): the Ling MoE
# decode dispatch-collapse lever. Same per-byte repack body, plus a per-expert
# base lookup from a meta[] buffer (binding 5) + a 2D dispatch (gl_WorkGroupID.y
# = expert). Collapses the 8-experts x {gate,up,down} = 24 tiny per-expert matvec
# dispatches into 3 batched dispatches while keeping per-expert throughput at the
# BW floor (the guardrail: batch THROUGH the repack kernel, not the v1 tree).
compile "mul_mat_vec_mlx4repack_batched_f32_f32" "mul_mat_vec_mlx4repack_batched_f32_f32.comp"

# DeepSeek-V4-Flash Phase 2a — 2-bit affine repack (routed experts, gs128, 84% of
# model). The 2-bit twin of mlx4repack: dwordx4 (uvec4) = 64 codes/load, 16 codes
# per u32 (word-aligned, byte-identical to model::dequantize_mlx_affine per_word=16),
# scale/bias ONCE per 64-elem chunk (2 chunks/gs128 group), fma-affine + subgroupAdd.
# k%64==0, packed base 16B-aligned (wpr=k/16 mult 4). Gate ARGMAX-EXACT vs the oracle.
compile "mul_mat_vec_mlx2repack_f32_f32" "mul_mat_vec_mlx2repack_f32_f32.comp"

# 2-bit EXPERT-BATCHED (Ling MoE dispatch-collapse primitive at 2-bit width): same
# repack body + per-expert meta[] base lookup + 2D dispatch (WorkGroupID.y=expert).
compile "mul_mat_vec_mlx2repack_batched_f32_f32" "mul_mat_vec_mlx2repack_batched_f32_f32.comp"

# DeepSeek-V4-Flash Phase 2a — 6-bit CONTIGUOUS affine matvec (attn/MLA/DSA
# indexer+compressor, gs128). 6-bit codes are a contiguous little-endian bitstream
# crossing u32 boundaries (packed_cols = in*6/32). "3 words per 16 codes" scheme:
# each thread unpacks 16 codes from 3 consecutive words with COMPILE-TIME-CONSTANT
# shifts (only codes 5/10 straddle) => register-resident, no scratch. Ports the
# model::dequantize_mlx_affine_bits 6-bit path to GLSL; gate ARGMAX-EXACT vs it.
compile "mul_mat_vec_mlx6_f32_f32" "mul_mat_vec_mlx6_f32_f32.comp"

# DeepSeek-V4-Flash Phase 2b — 8-bit gs64 affine matvec (shared_experts /
# embed_tokens / lm_head). Aligned 4 codes/u32 (8 divides 32, no boundary cross);
# dwordx4 = 16 codes/load, fma-factored affine, gs64 => chunks_per_group=4. First
# REAL 8-bit MLX-affine matvec (mul_mat_vec_mlx4_w8 is a misnamed 4-bit). Gate
# ARGMAX-EXACT vs model::dequantize_mlx_affine_bits (bits=8).
compile "mul_mat_vec_mlx8_f32_f32" "mul_mat_vec_mlx8_f32_f32.comp"

# DeepSeek-V4-Flash Phase 2b — Manifold Hyper-Connection (mHC) with the fp32
# stability treatment the reference lacks (sanitize + max-factored RMSNorm + logit
# clamp; exact linear Sinkhorn). One workgroup/token; proven in
# scripts/dsv4/hc_stability.py (equivalence cos=1.0 + depth-finiteness).
compile "dsv4_hyper_connection" "dsv4_hyper_connection.comp"

# DeepSeek-V4-Flash Phase 2b — DSA Lightning-Indexer SCORE kernel (ReLU
# head-weighted q·compressed_kv -> index_scores; the sparse-attention gate).
# Ports the reference indexer score math; validated vs hc_dsa_oracle.py.
compile "dsv4_dsa_index_score" "dsv4_dsa_index_score.comp"

# DeepSeek-V4-Flash Phase 2d — DSA compressor-POOL + dual-RoPE(yarn θ=160000).
# Ca/Cb two-series windowed lookback -> per-channel softmax-gate pool over the 2m
# axis -> kv_norm RMSNorm -> interleaved dual-RoPE. Emits the post-RoPE
# compressed_kv consumed by dsv4_dsa_index_score; validated vs hc_dsa_oracle.py
# (DSA.compressed) through debug_dsv4_dsa_compress.
compile "dsv4_dsa_compress" "dsv4_dsa_compress.comp"

# DeepSeek-V4-Flash Phase 2e — DSA causal TOP-512 SELECT. Consumes the
# index_scores from dsv4_dsa_index_score and emits the causal top-k(512) window
# indices with the -1 sentinel (the sparse-attention candidate set). Rank-scatter
# selection; set-match vs the host dsa_topk_causal mirror through
# debug_dsv4_dsa_topk. Completes the DSA GPU trio (compress -> score -> select).
compile "dsv4_dsa_topk" "dsv4_dsa_topk.comp"

# NVFP4 (NVIDIA FP4 / modelopt W4A16) dequantize matvec (VLLM_VULKAN_NVFP4_GPU=1).
# Like mlx4 but the 4-bit codes map through the E2M1 LUT (not a raw int) and the
# per-block e4m3 scale * per-tensor f32 global are pre-folded into one f32 scale
# buffer on upload — so no bias buffer and no in-shader e4m3/global. Packed 4-bit
# weight stays resident (~1/4 of bf16). Mirrors model::dequantize_nvfp4.
compile "mul_mat_vec_nvfp4_f32_f32" "mul_mat_vec_nvfp4.comp"

# NVFP4 VALU-bound repack refactor (VLLM_VULKAN_NVFP4_REPACK, default OFF) for the
# NVFP4 mlp/expert path (k%32==0, group_size=16). The nvfp4 twin of the mlx4
# repack: dwordx4 (uvec4) word-granular load = 32 nibbles/load, folded f32 scale
# ONCE per 16-elem group (2 groups/chunk), E2M1-LUT dot + subgroupAdd reduction.
# Cuts the same ~190-addr-VALU/16-unpack address-gen the gfx1013 ISA probe found
# in v1 (mul_mat_vec_nvfp4.comp). Packed layout unchanged (contiguous-4-word-per-
# thread == existing row-major pack order). Lands e2e on the PP-resident NVFP4
# fleet (nemotron-75B PP-5, gemma-31B, Laguna, Step-3.7, qwen-27B-NVFP4).
compile "mul_mat_vec_nvfp4repack_f32_f32" "mul_mat_vec_nvfp4repack_f32_f32.comp"

# NVFP4 repack + E4M3-RESIDENT scale variant (VLLM_VULKAN_NVFP4_REPACK=1 AND
# VLLM_VULKAN_NVFP4_E4M3_SCALES=1). Same dwordx4/subgroupAdd repack, but the
# per-group scale is the raw e4m3 byte (0.5 bits/param) decoded * f32 global
# hoisted ONCE per chunk — stacks the >=150B footprint lever on the address-gen
# win. Bit-exact vs the f32-fold repack (parenthesized global fold).
compile "mul_mat_vec_nvfp4_e4m3repack_f32_f32" "mul_mat_vec_nvfp4_e4m3repack_f32_f32.comp"

# NVFP4 E4M3-RESIDENT scale variant (VLLM_VULKAN_NVFP4_E4M3_SCALES=1). Keeps the
# raw per-group e4m3 block scale resident (1 byte) + the per-tensor f32 global as
# a push-constant, re-adding the e4m3 decode in-loop (0.5 bits/param vs the
# f32-fold's 2.0). The >=150B NVFP4 OOM-vs-fit lever (6.0 -> 4.5 bits/param).
# Same mul_mat_vec_ prefix -> build.rs classes it Matvec -> auto _r2/_r4/_r8.
compile "mul_mat_vec_nvfp4_e4m3_f32_f32" "mul_mat_vec_nvfp4_e4m3.comp"

# EXPERT-BATCHED e4m3 NVFP4 matvec (Laguna MoE CB-batch lever,
# VLLM_VULKAN_LAGUNA_CBBATCH): computes one projection (gate/up/down) for ALL
# routed experts in ONE dispatch (gl_WorkGroupID.y = expert slot; per-slice
# offsets + per-tensor global read from a small meta buffer). BIT-EXACT with the
# per-expert mul_mat_vec_nvfp4_e4m3 dispatches it replaces (identical
# BLOCK_SIZE/NUM_ROWS spec constants + dequant + reduction). Same mul_mat_vec_
# prefix -> build.rs classes it Matvec -> auto _r2/_r4/_r8.
compile "mul_mat_vec_laguna_expb_e4m3_f32_f32" "mul_mat_vec_laguna_expb_e4m3.comp"

# Genuine single-stream multi-column (NUM_COLS) matvec kernels for the
# spec-decode Phase-0 batched-verify amortization microbench
# (debug_matvec_cols_timing). Unlike the generic mul_mat_vec.comp base.glsl
# _r*_c* path — which re-streams the weight per token-column — these load/
# dequantize each weight element ONCE and reuse it across all NUM_COLS columns
# (the f16/q8_0 siblings of mul_mat_vec_nvfp4.comp). Same BLOCK_SIZE/NUM_ROWS/
# NUM_COLS spec constants; self-contained, no -D type defines. 2D workgroup
# grid so n can exceed the 65535 one-dimension limit (lm_head n=131072).
compile "mul_mat_vec_f16_cols"  "mul_mat_vec_f16_cols.comp"
compile "mul_mat_vec_q8_0_cols" "mul_mat_vec_q8_0_cols.comp"
# mlx4 4-bit DEQUANT-cols batched matvec (on-node ACO gate for the fleet's REAL
# 4-bit weight format). Same single-stream design as q8_0_cols: unpack+affine-
# dequantize each nibble ONCE and reuse the scalar across NUM_COLS token columns
# (columns = inner loop), amortizing the weight read AND the nibble-unpack+affine
# ALU across the batch. Minimal live state (only NUM_ROWS*NUM_COLS accumulators
# cross the column loop) to stay LDS-bound and avoid the repack VGPR trap.
# NUM_COLS=1 is byte-identical to mul_mat_vec_mlx4_f32_f32. Bindings match
# mul_mat_vec_mlx4.comp: [packed, scales, biases, x, out].
compile "mul_mat_vec_mlx4_cols" "mul_mat_vec_mlx4_cols.comp"

# FP8-E4M3 (modelopt W8A16) dequantize matvec (VLLM_VULKAN_NVFP4_GPU=1, attention).
# 1 byte/weight, per-tensor (or per-row) f32 scale kept in a scale buffer; the
# 256-entry E4M3 LUT is baked in (proven == model::e4m3_to_f32). Same BLOCK_SIZE/
# NUM_ROWS spec constants as nvfp4/mlx4, so no -D type defines.
compile "mul_mat_vec_fp8_f32_f32" "mul_mat_vec_fp8.comp"

# FP8-E4M3 fast variant (VLLM_VULKAN_FP8_FAST): arithmetic decode + vec4 loads
# instead of the const-LUT gather + scalar byte loads above. Default OFF.
compile "mul_mat_vec_fp8fast_f32_f32" "mul_mat_vec_fp8_fast.comp"

# FP8-E4M3 subgroup-reduction twin (VLLM_VULKAN_FP8_REPACK): fp8_fast hot loop +
# subgroupAdd reduction (no LDS barrier tree). Reduction-epilogue-only lever
# (fp8 is not address-gen-bound per the s4rig ISA rig). Default OFF.
compile "mul_mat_vec_fp8repack_f32_f32" "mul_mat_vec_fp8repack_f32_f32.comp"

# ── Qwen3.6 GatedDeltaNet decode kernels (WS1b, VLLM_VULKAN_DN_GPU) ───────────
# Plain-class shaders (fixed local_size, no spec constants, no -D defines):
# depthwise conv1d step + SiLU with sliding-window state update; per-head Q/K
# RMSNorm(no-weight) + inv-scale; per-value-head delta-rule recurrence + gated
# RMSNorm. The conv/delta state stays GTT-resident between decode steps and the
# whole deltanet block records into ONE command buffer
# (qwen35_delta_net_gpu_fused).
compile "q35_dn_conv_step" "q35_dn_conv_step.comp"
compile "q35_gdn_qknorm"   "q35_gdn_qknorm.comp"
compile "q35_gdn_step"     "q35_gdn_step.comp"
# Kimi Phase-B / Block-2: KDA decode-step recurrence — q35_gdn_step with
# per-KEY-CHANNEL decay (binding 3 = decay_in[nk*kd]) + sigmoid output gate.
compile "kda_gdn_step"     "kda_gdn_step.comp"
# Kimi decode lever #5: per-key-channel decay precompute (softplus/exp), moved
# off the host into the fused KDA command buffer.
compile "kda_decay"        "kda_decay.comp"
# Ling decode Phase-3 (fused 1-CB KDA): the NEW-MATH glue variants for Ling's
# safe_gate decay (lower_bound*sigmoid(exp(A_log)*fb)) and L2-norm qknorm
# (scale-on-q-only), distinct from the Kimi kda_decay / q35_gdn_qknorm forms.
compile "ling_kda_decay"   "ling_kda_decay.comp"
compile "ling_kda_l2norm"  "ling_kda_l2norm.comp"
# Phase-1 GPU MoE router (VLLM_VULKAN_LING_MOE_INDIRECT): grouped-topk scoring +
# selection in one dispatch (adapted from laguna_router for Ling n_group=8/
# topk_group=4). Emits top_k indices + weights; only the top-k is read back.
compile "ling_moe_router"  "ling_moe_router.comp"
# GPU meta-builder: consumes ling_moe_router top-k -> per-expert gather descriptors
# (meta[] for gate/up/down) + routed scores, so the route->meta->matvec chain
# records into one CB with no host index readback (VLLM_VULKAN_LING_MOE_INDIRECT).
compile "ling_moe_meta"    "ling_moe_meta.comp"
# WS3: MoE tail (score-weighted routed accumulate + sigmoid-gated shared add +
# residual) in one dispatch, so the fused MoE CB writes the next hidden on-GPU.
compile "q35_moe_accum"    "q35_moe_accum.comp"
# Nemotron MoE-tail collapse (R1b): same routed-accumulate role as
# q35_moe_accum but variable top_k (loop in-shader) instead of fixed top-8,
# and latent-space output (fc2 projects to hidden separately).
compile "nemotron_moe_accum" "nemotron_moe_accum.comp"
# Laguna-S-2.1 MoE tail (VLLM_VULKAN_LAGUNA_GPU_ACCUM): top-10 score-weighted
# routed accumulate + UNGATED shared add (NO sigmoid gate, unlike q35_moe_accum's
# fixed top-8 sigmoid-gated shared) in one dispatch, so moe_token_1cb reads back
# ONE [hidden] vector instead of the 10 routed down + shared down.
compile "laguna_moe_accum" "laguna_moe_accum.comp"
# CB-batch twin (VLLM_VULKAN_LAGUNA_CBBATCH): same top-10 accumulate but reads
# the 10 routed down outputs from ONE concatenated [10*n] buffer (d[e*n+i])
# instead of 10 bindings, matching the batched down projection. BIT-EXACT.
compile "laguna_moe_accum_b" "laguna_moe_accum_b.comp"

# Nemotron-H Mamba2 GPU SSD decode-scan (R2, VLLM_VULKAN_NEMOTRON_GPU_SCAN):
# depthwise conv1d+SiLU (full-kern state row, unlike q35's split state) +
# per-head SSD recurrence/gate + gated RMSNorm, ported verbatim from the CPU
# reference in src/nemotron.rs. Plain-class (fixed local_size, no spec
# constants), same geometry posture as the q35 GDN decode chain above.
compile "nemotron_ssm_conv_step"   "nemotron_ssm_conv_step.comp"
compile "nemotron_ssd_scan"        "nemotron_ssd_scan.comp"
compile "nemotron_gated_rmsnorm"   "nemotron_gated_rmsnorm.comp"
# Phase 3 (plan-epilogue-fused-moe-gemm.md §5): BATCHED, runtime-top_k variant
# of the MoE tail for the grouped PREFILL path -- score-weighted routed combine
# + sigmoid-gated shared add over all T tokens in one dispatch, so `down_out`
# never leaves VRAM (readback 16384->2048 f/tok) and the host combine is gone.
# Gather-reduce (no scatter) -> atomic-free + deterministic (cos=1.0).
compile "q35_moe_accum_batched" "q35_moe_accum_batched.comp"
# P1a (batched-prefill foundation): q35_gdn_step + an inner `for t in
# 0..n_tokens` loop, state column register-resident across the loop — ONE
# dispatch per linear layer instead of one per token. See
# plan-batched-prefill.md §1c/§7 and debug_gdn_scan for the CPU-ref cos gate.
compile "q35_gdn_scan"     "q35_gdn_scan.comp"

# K-quant dedicated shaders. B_TYPEV2/V4 needed for packed buffer views.
# FLOAT_TYPEV2 for local dm values: use vec2 (f32) not f16vec2, to avoid
# requiring GL_EXT_shader_explicit_arithmetic_types_float16 for local vars.
compile "mul_mat_vec_q2_k_f16_f32" "mul_mat_vec_q2_k.comp" DATA_A_Q2_K=1 \
  DATA_B_F16=1 B_TYPE=float16_t B_TYPEV2=f16vec2 \
  D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 RMS_NORM_ROPE_FUSION=0

compile "mul_mat_vec_q3_k_f16_f32" "mul_mat_vec_q3_k.comp" DATA_A_Q3_K=1 \
  DATA_B_F16=1 B_TYPE=float16_t B_TYPEV2=f16vec2 \
  D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 RMS_NORM_ROPE_FUSION=0

compile "mul_mat_vec_q4_k_f32_f32_subgroup" "mul_mat_vec_q4_k.comp" DATA_A_Q4_K=1 \
  DATA_B_F32=1 B_TYPE=float B_TYPEV4=vec4 \
  D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 USE_SUBGROUP_ADD_NO_SHMEM=1 RMS_NORM_ROPE_FUSION=0

compile "mul_mat_vec_q5_k_f16_f32" "mul_mat_vec_q5_k.comp" DATA_A_Q5_K=1 \
  DATA_B_F16=1 B_TYPE=float16_t B_TYPEV2=f16vec2 \
  D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 RMS_NORM_ROPE_FUSION=0

compile "mul_mat_vec_q6_k_f32_f32_subgroup" "mul_mat_vec_q6_k.comp" DATA_A_Q6_K=1 \
  DATA_B_F32=1 B_TYPE=float B_TYPEV4=vec4 \
  D_TYPE=float FLOAT_TYPE=float FLOAT_TYPEV2=vec2 USE_SUBGROUP_ADD_NO_SHMEM=1 RMS_NORM_ROPE_FUSION=0

# ── General matmul (prefill) ───────────────────────────────────────────────────
echo ""
echo "=== General matmul (prefill) ==="
compile "matmul_f32_f16"         "mul_mm.comp" DATA_A_F32=1 DATA_B_F16=1 B_TYPE=float16_t ${MM_F16} RMS_NORM_ROPE_FUSION=0
compile "matmul_f32_f16_aligned" "mul_mm.comp" DATA_A_F32=1 DATA_B_F16=1 B_TYPE=float16_t ${MM_F16} ALIGNED=1 RMS_NORM_ROPE_FUSION=0
compile "matmul_f32_f32"         "mul_mm.comp" DATA_A_F32=1 DATA_B_F32=1 B_TYPE=float     ${MM_F32} RMS_NORM_ROPE_FUSION=0
compile "matmul_f32_f32_fp32"    "mul_mm.comp" DATA_A_F32=1 DATA_B_F32=1 B_TYPE=float     ${MM_F32} ACC_F32=1 RMS_NORM_ROPE_FUSION=0
compile "matmul_f16_f32_fp32"    "mul_mm.comp" DATA_A_F16=1 DATA_B_F32=1 B_TYPE=float     ${MM_F32} ACC_F32=1 RMS_NORM_ROPE_FUSION=0

# ── Design A: quantized tiled GEMM (PREFILL / M6 Mamba2) ────────────────────────
# mul_mm.comp already dequantizes GGUF blocks into the f16 LDS `buf_a` tile ONCE
# per streamed block (mul_mm_funcs.glsl::load_a_to_shmem: DATA_A_Q8_0 :119,
# Q4_K :190, Q6_K :265), then the register-blocked MAC loop reuses that tile
# across all BN output columns — the "dequant-to-LDS then f16-GEMM" that
# amortizes weight read+dequant across N. It WINS at prefill N (many BN tiles
# fill the machine); it is the WRONG tool for spec-decode N=T<=8 (occupancy-
# starved ~18 GB/s — use Design C's column-batched matvec there instead).
#
# LOAD_VEC_A=4 is LOAD-BEARING (proven on perf/quant-batched-cols): it makes the
# loader's per-thread `row` range over BK/LOAD_VEC_A=8 values, matching the
# 8-value-per-thread `idx = pos_a + col*stride_a/LOAD_VEC_A + row` addressing in
# mul_mm_funcs.glsl's DATA_A_Q8_0 branch. A_TYPE / A_TYPE_PACKED16 are supplied
# by types.glsl from the DATA_A_* macro; the packed16 path needs no explicit
# A_TYPE.
#
# ⚠ SCAFFOLD ONLY — these .spv are compiled + verified but INTENTIONALLY
# UNREGISTERED (build.rs SKIP). The quant mul_mm path needs BK=32 (spec
# constant_id 3, default 16 = "assumed 32 for a quant"), which the current f16
# compile_mul_mm (src/pipeline.rs) does NOT set, and a quant-aware prefill
# dispatch swap. Registering them through the f16 MulMm class would create a
# BK=16 pipeline = wrong. Wiring (compile_mul_mm_quant + dispatch) is the
# remaining GPU-gated work — see quant-batched-matmul-impl.md.
compile "matmul_q8_0_f32_fp32" "mul_mm.comp" DATA_A_Q8_0=1 DATA_B_F32=1 B_TYPE=float \
  ${MM_F32} ACC_F32=1 LOAD_VEC_A=4 RMS_NORM_ROPE_FUSION=0
compile "matmul_q4_k_f32_fp32" "mul_mm.comp" DATA_A_Q4_K=1 DATA_B_F32=1 B_TYPE=float \
  ${MM_F32} ACC_F32=1 LOAD_VEC_A=4 RMS_NORM_ROPE_FUSION=0
compile "matmul_q6_k_f32_fp32" "mul_mm.comp" DATA_A_Q6_K=1 DATA_B_F32=1 B_TYPE=float \
  ${MM_F32} ACC_F32=1 LOAD_VEC_A=4 RMS_NORM_ROPE_FUSION=0
# GEMM campaign Phase 0 (mul_mm sweep-able variants; see IMPROVEMENTS.md /
# the gemm-campaign plan). All three keep A=f16 (DATA_A_F16), B=f32
# (DATA_B_F32) like the live matmul_f16_f32_fp32 -- only the arithmetic
# (MM_F32 vs MM_F16) and the load path (ALIGNED) differ. BM/BN/WM/WN/WMITER/
# TM/TN/WARP are all spec constants (mul_mm.comp constant_id 1/2/4/5/6/7/8/10),
# so the sweep tool (debug_gemm_geometry) tries the whole tile/warp grid on
# THESE SAME 4 SPIR-V without recompiling.
#
# _aligned fix (load-bearing, not just a flag flip): ALIGNED=1 alone leaves
# LOAD_VEC_A/B at their #ifndef default of 1, which falls through to the
# SAME unaligned 2x-scalar branch as the non-aligned build -- this is why the
# in-tree matmul_f32_f16_aligned/matmul_f32_f32_fp32 aligned variants were
# dead/unvalidated (never dispatched, and would have been no-ops even if
# wired). Setting LOAD_VEC_A=8 LOAD_VEC_B=8 selects the true vec8-coalesced
# branch in mul_mm_funcs.glsl -- but that branch also references a
# `FLOAT_TYPEV8` macro that is NEVER DEFINED anywhere in this tree (only
# FLOAT_TYPEV2/V4 are wired via MM_F32/MM_F16); it is dead upstream code that
# would fail to compile as-is. Defining FLOAT_TYPEV8 here (mirroring
# FLOAT_TYPEV4's f32/f16 split: mat2x4 / f16mat2x4) is what actually makes the
# vec8 path buildable. B_TYPE must ALSO become mat2x4 (not float) for
# LOAD_VEC_B=8: mul_mm.comp declares `buffer B { B_TYPE data_b[]; }` with no
# separate vec4/vec8 view binding (unlike the matvec kernels' B_TYPEV4
# convention), so the vec8 load `FLOAT_TYPEV8(data_b[idx])` requires the
# buffer's own element type to already be a 2x4 matrix. Requires K % 8 == 0
# (all live shapes are K multiples of 256 -- OK; enforced in the sweep tool).
compile "matmul_f16_f32_fp32_aligned" "mul_mm.comp" DATA_A_F16=1 DATA_B_F32=1 \
  B_TYPE=mat2x4 ${MM_F32} ACC_F32=1 ALIGNED=1 LOAD_VEC_A=8 LOAD_VEC_B=8 \
  FLOAT_TYPEV8=mat2x4 RMS_NORM_ROPE_FUSION=0
# f16 arithmetic (MM_F16 already sets ACC_TYPE=float -- f32 accumulate per the
# plan), unaligned load path (2x-scalar, same as the live kernel).
compile "matmul_f16_f32_f16" "mul_mm.comp" DATA_A_F16=1 DATA_B_F32=1 B_TYPE=float \
  ${MM_F16} RMS_NORM_ROPE_FUSION=0
# f16 arithmetic + ALIGNED vec8 loads (the "everything on" corner from the plan).
compile "matmul_f16_f32_f16_aligned" "mul_mm.comp" DATA_A_F16=1 DATA_B_F32=1 \
  B_TYPE=mat2x4 ${MM_F16} ALIGNED=1 LOAD_VEC_A=8 LOAD_VEC_B=8 \
  FLOAT_TYPEV8=f16mat2x4 RMS_NORM_ROPE_FUSION=0

# GEMM campaign Phase A (quantized batched matmul; see plan-quant-batched-
# matmul.md): dequant-to-shmem `mul_mm` heads for quantized A already existed
# source-side (mul_mm_funcs.glsl) but were compiled by NOTHING -- these are
# the first quant `mul_mm` variants ever built. ACC_F32 for the tight
# correctness gate (cos>=0.999). BK=32 is set via a spec constant at pipeline
# creation (pipeline.rs::compile_mul_mm_quant*), not a compile-time -D, so it
# is NOT listed here.
#
# A1 -- MLX-affine 4-bit dense (the production 4-bit path). No ggml block
# struct backs this format (scale/bias live in separate Scales/Biases SSBOs,
# bindings 3/4 -- see mul_mm.comp's DATA_A_MLX4 guard), so A_TYPE=uint (raw
# packed nibble words) MUST be passed explicitly -- types.glsl has no
# DATA_A_MLX4 branch to derive it. Same LOAD_VEC_A=4 addressing convention as
# Q8_0 (see mul_mm_funcs.glsl's DATA_A_MLX4 branch doc-comment).
compile "matmul_mlx4_f32_fp32" "mul_mm.comp" DATA_A_MLX4=1 A_TYPE=uint DATA_B_F32=1 B_TYPE=float \
  ${MM_F32} ACC_F32=1 LOAD_VEC_A=4 RMS_NORM_ROPE_FUSION=0
# B (Phase B) -- MLX-affine 4-bit GROUPED-expert GEMM (MUL_MAT_ID). Same dequant
# head + LOAD_VEC_A=4 addressing as the dense mlx4 line above, but the
# MUL_MAT_ID machinery (IDS/Counts bindings 3/4, per-workgroup expert scan,
# in-shader B-gather + output scatter) batches the Qwen3.6 MoE routed experts:
# one dispatch per gate/up/down over ALL routed (token,slot) pairs instead of
# T*top_k per-token matvecs. Scales/Biases move to bindings 5/6 (3/4 are IDS/
# Counts under MUL_MAT_ID). No MUL_MAT_ID_USE_SUBGROUPS -- the scalar data_ids
# scan works on wave64; the subgroup-ballot variant is a deferred micro-opt.
# Do NOT pass LOAD_VEC_B (batch-2 B path, matching the dense mlx4 line).
compile "matmul_mlx4_id_f32_fp32" "mul_mm.comp" DATA_A_MLX4=1 MUL_MAT_ID=1 A_TYPE=uint DATA_B_F32=1 B_TYPE=float \
  ${MM_F32} ACC_F32=1 LOAD_VEC_A=4 RMS_NORM_ROPE_FUSION=0
# C (epilogue-fused MoE GEMM, plan-epilogue-fused-moe-gemm.md) -- gate+up
# fused into ONE dispatch: a SECOND A-binding (up weight, bindings 7/8/9) +
# a silu(gate)*up epilogue at the store (mul_mm.comp's MLX4_ID_GATEUP branch)
# replace the separate gu_gate/gu_up buffers + the silu_f32/mul_f32_f32_f32
# dispatches entirely. Same dequant head/LOAD_VEC_A=4 addressing as the base
# matmul_mlx4_id_f32_fp32 line above; VLLM_VULKAN_MOE_GEMM_FUSED gates
# whether qwen35_forward.rs ever selects this pipeline (default OFF).
compile "matmul_mlx4_id_gateup_silu_f32_fp32" "mul_mm.comp" DATA_A_MLX4=1 MUL_MAT_ID=1 MLX4_ID_GATEUP=1 A_TYPE=uint DATA_B_F32=1 B_TYPE=float \
  ${MM_F32} ACC_F32=1 LOAD_VEC_A=4 RMS_NORM_ROPE_FUSION=0

# ── Flash attention ────────────────────────────────────────────────────────────
echo ""
echo "=== Flash attention ==="
# Q=f32 (hardcoded), KV=f16, D=f32, f32 accum
compile "flash_attn_f32_f16_f32" "flash_attn.comp" \
  DATA_A_F16=1 D_TYPE=float ${FA_F32}

# Q=f32, KV=f16, D=f16, f32 accum (FLOAT16=1 enables f16 extension)
compile "flash_attn_f32_f16_f16" "flash_attn.comp" \
  DATA_A_F16=1 FLOAT16=1 \
  FLOAT_TYPE=float16_t FLOAT_TYPEV2=f16vec2 FLOAT_TYPEV4=f16vec4 \
  ACC_TYPE=float ACC_TYPEV2=vec2 ACC_TYPEV4=vec4 \
  D_TYPE=float16_t D_TYPEV4=f16vec4

# f16 accumulator variants
compile "flash_attn_f32_f16_f32_f16acc" "flash_attn.comp" \
  DATA_A_F16=1 FLOAT16=1 ACC_F16=1 \
  FLOAT_TYPE=float16_t FLOAT_TYPEV2=f16vec2 FLOAT_TYPEV4=f16vec4 \
  ACC_TYPE=float16_t ACC_TYPEV2=f16vec2 ACC_TYPEV4=f16vec4 \
  D_TYPE=float D_TYPEV4=vec4

compile "flash_attn_f32_f16_f16_f16acc" "flash_attn.comp" \
  DATA_A_F16=1 FLOAT16=1 ACC_F16=1 \
  FLOAT_TYPE=float16_t FLOAT_TYPEV2=f16vec2 FLOAT_TYPEV4=f16vec4 \
  ACC_TYPE=float16_t ACC_TYPEV2=f16vec2 ACC_TYPEV4=f16vec4 \
  D_TYPE=float16_t D_TYPEV4=f16vec4

# Explicit fp32 accumulator path
compile "flash_attn_f32_f16_f32_fp32" "flash_attn.comp" \
  DATA_A_F16=1 ACC_F32=1 D_TYPE=float ${FA_F32}

# ── Quantization utils ─────────────────────────────────────────────────────────
echo ""
echo "=== Quantization utils ==="
compile "quantize_q8_1_x4" "quantize_q8_1.comp" \
  DATA_A_F32=1 A_TYPE=float D_TYPE=float FLOAT_TYPE=float QBLOCK_X4=1 RMS_NORM_ROPE_FUSION=0

# ── Paged KV cache ────────────────────────────────────────────────────────────
echo ""
echo "=== Paged KV cache ==="
compile "paged_kv_write_f16" "paged_kv_write_f16.comp"
compile "paged_kv_write_f32" "paged_kv_write_f32.comp"
compile "paged_attn_decode_f16" "paged_attn_decode_f16.comp"
compile "paged_attn_decode_f16_coop" "paged_attn_decode_f16_coop.comp"
compile "paged_attn_decode_f32" "paged_attn_decode_f32.comp"
compile "paged_attn_decode_f32_coop" "paged_attn_decode_f32_coop.comp"
# Fork's GFX1013 wave64 subgroup decode kernel (not in upstream); still
# referenced by the fork's attn-kernel dispatch — keep it.
compile "paged_attn_decode_f32_sg" "paged_attn_decode_f32_sg.comp"
# item-4b: f16-KV counterpart of the _sg kernel above (the existing
# paged_attn_decode_f16/_coop variants lack `window_start` and are the slow
# scalar/tree-reduce dispatch shapes, so they can't back the resident
# 1-CB/decode seam under VLLM_VULKAN_Q35_KV_F16).
compile "paged_attn_decode_f16_sg" "paged_attn_decode_f16_sg.comp"
# NB: upstream's BLOCK_SIZE=512 `_coop_512` decode variants (#56) are
# perf-opts to the replaced Python paged-attention decode path; the fork's
# Rust engine never dispatches them (no reference in src/), and its
# `cutover_guard_registry_len` intentionally pins the lean shader set — so
# they are deliberately NOT compiled here.

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
if [ "${FAILED}" -gt 0 ]; then
  echo "ERROR: ${FAILED}/${TOTAL} shaders failed to compile." >&2
  exit 1
else
  if [ "${SKIPPED}" -gt 0 ]; then
    # Not an error: a feature slice ships only the shaders it uses, and the
    # rest are skipped by design. Reported so a shader that goes missing by
    # ACCIDENT (renamed file, typo'd entry) is visible in the build log rather
    # than silently absent from the registry.
    echo "OK: ${TOTAL} shaders compiled to ${OUT_DIR} (${SKIPPED} skipped: source not in this slice)"
  else
    echo "OK: all ${TOTAL} shaders compiled to ${OUT_DIR}"
  fi
fi
