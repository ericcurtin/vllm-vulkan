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

compile() {
  local out_stem="$1"; shift
  local src="$1";      shift
  local out="${OUT_DIR}/${out_stem}.spv"
  local actual_src="${SHADER_DIR}/${src}"

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
for op in silu gelu gelu_quick relu tanh exp sigmoid abs neg ceil; do
  compile "${op}_f32" "${op}.comp" ${U_F32}
done
compile "gelu_inplace_f32" "gelu.comp" ${U_F32} INPLACE=1

# ── Elementwise binary ─────────────────────────────────────────────────────────
echo ""
echo "=== Elementwise binary ==="
compile "add_f32_f32_f32" "add.comp"  DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "add_f32_f32_f16" "add.comp"  DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "add_f32_f16_f32" "add.comp"  DATA_A_F32=1 A_TYPE=float DATA_B_F16=1 B_TYPE=float16_t D_TYPE=float FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "add_rms_f32_f32_f32" "multi_add.comp" ${BIN_F32}
compile "add_rms_f32_f32_f16" "multi_add.comp" DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "mul_f32_f32_f32" "mul.comp" ${BIN_F32}
compile "mul_f32_f32_f16" "mul.comp" DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
compile "div_f32_f32_f16" "div.comp" DATA_A_F32=1 A_TYPE=float DATA_B_F32=1 B_TYPE=float D_TYPE=float16_t FLOAT_TYPE=float RMS_NORM_ROPE_FUSION=0
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

# ── RoPE ───────────────────────────────────────────────────────────────────────
echo ""
echo "=== RoPE ==="
compile "rope_norm_f32_f16" "rope_norm.comp" DATA_A_F32=1 A_TYPE=float  ROPE_D_TYPE=float16_t D_TYPE=float      FLOAT_TYPE=float      RMS_NORM_ROPE_FUSION=0
compile "rope_norm_f16"     "rope_norm.comp" DATA_A_F16=1 A_TYPE=float16_t ROPE_D_TYPE=float16_t D_TYPE=float16_t FLOAT_TYPE=float16_t RMS_NORM_ROPE_FUSION=0
compile "rope_neox_f32_f16" "rope_neox.comp" DATA_A_F32=1 A_TYPE=float  ROPE_D_TYPE=float16_t D_TYPE=float      FLOAT_TYPE=float      RMS_NORM_ROPE_FUSION=0
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
compile "paged_attn_decode_f16_coop" "paged_attn_decode_f16_coop.comp" BLOCK_SIZE=256
compile "paged_attn_decode_f16_coop_512" "paged_attn_decode_f16_coop.comp" BLOCK_SIZE=512
compile "paged_attn_decode_f32" "paged_attn_decode_f32.comp"
compile "paged_attn_decode_f32_coop" "paged_attn_decode_f32_coop.comp" BLOCK_SIZE=256
compile "paged_attn_decode_f32_coop_512" "paged_attn_decode_f32_coop.comp" BLOCK_SIZE=512

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
if [ "${FAILED}" -gt 0 ]; then
  echo "ERROR: ${FAILED}/${TOTAL} shaders failed to compile." >&2
  exit 1
else
  echo "OK: all ${TOTAL} shaders compiled to ${OUT_DIR}"
fi
