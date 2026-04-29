/*
 * SPDX-License-Identifier: Apache-2.0
 * Compile all vllm-vulkan GLSL shaders to SPIR-V using libshaderc.
 *
 * Usage: compile_shaders <shader_dir> <output_dir>
 *
 * This is the shader compilation tool used by build.rs.
 * libshaderc handles #include directives natively.
 */

#include <shaderc/shaderc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

static char* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);
    char* buf = malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    if (out_len) *out_len = (size_t)len;
    return buf;
}

/* shaderc includer: resolve #include relative to the shader directory */
typedef struct {
    char shader_dir[4096];
} includer_data;

static shaderc_include_result* resolve_include(
    void* user_data,
    const char* requested_source,
    int type,
    const char* requesting_source,
    size_t include_depth
) {
    (void)type; (void)requesting_source; (void)include_depth;
    includer_data* d = (includer_data*)user_data;

    char path[8192];
    snprintf(path, sizeof(path), "%s/%s", d->shader_dir, requested_source);

    size_t src_len;
    char* src = read_file(path, &src_len);

    shaderc_include_result* result = calloc(1, sizeof(shaderc_include_result));
    if (!src) {
        result->source_name = "";
        result->source_name_length = 0;
        result->content = "file not found";
        result->content_length = 14;
    } else {
        char* name = strdup(path);
        result->source_name = name;
        result->source_name_length = strlen(name);
        result->content = src;
        result->content_length = src_len;
        result->user_data = src;  /* store for free */
    }
    return result;
}

static void release_include(void* user_data, shaderc_include_result* result) {
    (void)user_data;
    free((void*)result->user_data);  /* the file content */
    free((void*)result->source_name);
    free(result);
}

static int ends_with(const char* s, const char* suffix) {
    size_t sl = strlen(s), xl = strlen(suffix);
    return sl >= xl && strcmp(s + sl - xl, suffix) == 0;
}

static int compile_shader(
    shaderc_compiler_t compiler,
    shaderc_compile_options_t opts,
    const char* src_path,
    const char* out_path,
    const char* tag  /* e.g. "DATA_A_F32" */
) {
    size_t src_len;
    char* src = read_file(src_path, &src_len);
    if (!src) {
        fprintf(stderr, "Cannot read %s\n", src_path);
        return 1;
    }

    shaderc_compilation_result_t result = shaderc_compile_into_spv(
        compiler,
        src, src_len,
        shaderc_glsl_compute_shader,
        src_path,
        "main",
        opts
    );
    free(src);

    if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
        fprintf(stderr, "FAILED [%s] %s:\n%s\n",
            tag, src_path,
            shaderc_result_get_error_message(result));
        shaderc_result_release(result);
        return 1;
    }

    size_t spv_len = shaderc_result_get_length(result);
    const char* spv = shaderc_result_get_bytes(result);

    FILE* out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "Cannot write %s: %s\n", out_path, strerror(errno));
        shaderc_result_release(result);
        return 1;
    }
    fwrite(spv, 1, spv_len, out);
    fclose(out);
    shaderc_result_release(result);
    return 0;
}

/* ─── shader variant table ──────────────────────────────────────────────── */

typedef struct {
    const char* src;     /* source .comp filename */
    const char* out;     /* output .spv filename (relative to out_dir) */
    const char* defines; /* semicolon-separated KEY=VAL pairs, or "" */
} ShaderVariant;

/* Types used for matmul A matrix */
#define A_TYPES \
    "f32", "f16", \
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", \
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", \
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", \
    "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs", \
    "mxfp4", "nvfp4", "q1_0"

static const char* A_TYPE_NAMES[] = {
    "f32", "f16",
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s",
    "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs",
    "mxfp4", "nvfp4", "q1_0",
    NULL
};

/* uppercase variant for define name */
static void to_upper(const char* in, char* out, size_t max) {
    size_t i;
    for (i = 0; in[i] && i + 1 < max; i++) {
        char c = in[i];
        out[i] = (c >= 'a' && c <= 'z') ? c - 32 : c;
    }
    out[i] = '\0';
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: compile_shaders <shader_dir> <output_dir>\n");
        return 1;
    }
    const char* shader_dir = argv[1];
    const char* out_dir = argv[2];

    /* Create output dir */
    mkdir(out_dir, 0755);

    shaderc_compiler_t compiler = shaderc_compiler_initialize();

    /* Set up includer */
    includer_data inc_data;
    snprintf(inc_data.shader_dir, sizeof(inc_data.shader_dir), "%s", shader_dir);

    int total = 0, failed = 0;

    /* ── Helper macro to compile one variant ─────────────────────────── */
#define COMPILE(src_file, out_name, ...) do { \
    shaderc_compile_options_t opts = shaderc_compile_options_initialize(); \
    shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2); \
    shaderc_compile_options_set_target_spirv(opts, shaderc_spirv_version_1_5); \
    shaderc_compile_options_set_include_callbacks(opts, resolve_include, release_include, &inc_data); \
    shaderc_compile_options_set_optimization_level(opts, shaderc_optimization_level_performance); \
    shaderc_compile_options_set_forced_version_profile(opts, 450, shaderc_profile_none); \
    /* parse defines: "KEY=VAL;KEY2=VAL2" */ \
    const char* _defs[] = {__VA_ARGS__, NULL}; \
    for (int _i = 0; _defs[_i]; _i++) { \
        char _dk[256], _dv[256]; \
        const char* _eq = strchr(_defs[_i], '='); \
        if (_eq) { \
            strncpy(_dk, _defs[_i], _eq - _defs[_i]); _dk[_eq-_defs[_i]] = '\0'; \
            strncpy(_dv, _eq + 1, sizeof(_dv) - 1); _dv[sizeof(_dv)-1] = '\0'; \
        } else { \
            strncpy(_dk, _defs[_i], sizeof(_dk)-1); _dk[sizeof(_dk)-1] = '\0'; \
            _dv[0] = '\0'; \
        } \
        shaderc_compile_options_add_macro_definition(opts, _dk, strlen(_dk), \
            _dv[0] ? _dv : NULL, _dv[0] ? strlen(_dv) : 0); \
    } \
    char _src_path[4096], _out_path[4096]; \
    snprintf(_src_path, sizeof(_src_path), "%s/%s", shader_dir, src_file); \
    snprintf(_out_path, sizeof(_out_path), "%s/%s", out_dir, out_name); \
    printf("  %-60s -> %s\n", _src_path, out_name); \
    fflush(stdout); \
    total++; \
    if (compile_shader(compiler, opts, _src_path, _out_path, out_name) != 0) failed++; \
    shaderc_compile_options_release(opts); \
} while(0)

    /* ── Elementwise unary ops (no DATA_A needed) ──────────────────── */
    printf("== Elementwise unary ==\n");
    COMPILE("silu.comp",       "silu.spv",       "DATA_A_F32=1");
    COMPILE("gelu.comp",       "gelu.spv",       "DATA_A_F32=1");
    COMPILE("gelu_quick.comp", "gelu_quick.spv", "DATA_A_F32=1");
    COMPILE("relu.comp",       "relu.spv",       "DATA_A_F32=1");
    COMPILE("tanh.comp",       "tanh.spv",       "DATA_A_F32=1");
    COMPILE("exp.comp",        "exp.spv",        "DATA_A_F32=1");
    COMPILE("log.comp",        "log.spv",        "DATA_A_F32=1");
    COMPILE("sqrt.comp",       "sqrt.spv",       "DATA_A_F32=1");
    COMPILE("abs.comp",        "abs.spv",        "DATA_A_F32=1");
    COMPILE("neg.comp",        "neg.spv",        "DATA_A_F32=1");
    COMPILE("sigmoid.comp",    "sigmoid.spv",    "DATA_A_F32=1");
    COMPILE("square.comp",     "square.spv",     "DATA_A_F32=1");
    COMPILE("scale.comp",      "scale.spv",      "DATA_A_F32=1");

    printf("== Elementwise binary ==\n");
    COMPILE("add.comp",  "add_f32_f32.spv", "DATA_A_F32=1", "DATA_B_F32=1");
    COMPILE("add.comp",  "add_f32_f16.spv", "DATA_A_F32=1", "DATA_B_F16=1");
    COMPILE("add.comp",  "add_f16_f32.spv", "DATA_A_F16=1", "DATA_B_F32=1");
    COMPILE("mul.comp",  "mul_f32_f32.spv", "DATA_A_F32=1", "DATA_B_F32=1");
    COMPILE("div.comp",  "div_f32_f32.spv", "DATA_A_F32=1", "DATA_B_F32=1");
    COMPILE("sub.comp",  "sub_f32_f32.spv", "DATA_A_F32=1", "DATA_B_F32=1");
    COMPILE("acc.comp",  "acc_f32.spv",     "DATA_A_F32=1");

    printf("== Normalization ==\n");
    COMPILE("rms_norm.comp",  "rms_norm_f32.spv",   "DATA_A_F32=1");
    COMPILE("rms_norm.comp",  "rms_norm_f16.spv",   "DATA_A_F16=1");
    COMPILE("norm.comp",      "norm_f32.spv",        "DATA_A_F32=1");
    COMPILE("group_norm.comp","group_norm_f32.spv",  "DATA_A_F32=1");
    COMPILE("l2_norm.comp",   "l2_norm_f32.spv",     "DATA_A_F32=1");

    printf("== Softmax ==\n");
    COMPILE("soft_max.comp",        "soft_max_f32.spv",  "DATA_A_F32=1");
    COMPILE("soft_max_large1.comp", "soft_max_large1.spv","DATA_A_F32=1");
    COMPILE("soft_max_large2.comp", "soft_max_large2.spv","DATA_A_F32=1");
    COMPILE("soft_max_large3.comp", "soft_max_large3.spv","DATA_A_F32=1");

    printf("== RoPE ==\n");
    COMPILE("rope_norm.comp",  "rope_norm_f32.spv",  "DATA_A_F32=1");
    COMPILE("rope_norm.comp",  "rope_norm_f16.spv",  "DATA_A_F16=1");
    COMPILE("rope_neox.comp",  "rope_neox_f32.spv",  "DATA_A_F32=1");
    COMPILE("rope_neox.comp",  "rope_neox_f16.spv",  "DATA_A_F16=1");
    COMPILE("rope_multi.comp", "rope_multi_f32.spv", "DATA_A_F32=1");
    COMPILE("rope_multi.comp", "rope_multi_f16.spv", "DATA_A_F16=1");

    printf("== Copy / reshape ==\n");
    COMPILE("copy.comp",         "copy_f32_f32.spv",   "DATA_A_F32=1", "DATA_B_F32=1");
    COMPILE("copy.comp",         "copy_f16_f32.spv",   "DATA_A_F16=1", "DATA_B_F32=1");
    COMPILE("copy.comp",         "copy_f32_f16.spv",   "DATA_A_F32=1", "DATA_B_F16=1");
    COMPILE("contig_copy.comp",  "contig_copy_f32.spv","DATA_A_F32=1");
    COMPILE("get_rows.comp",     "get_rows_f32.spv",   "DATA_A_F32=1");
    COMPILE("get_rows.comp",     "get_rows_f16.spv",   "DATA_A_F16=1");
    COMPILE("pad.comp",          "pad_f32.spv",        "DATA_A_F32=1");
    COMPILE("repeat.comp",       "repeat_f32.spv",     "DATA_A_F32=1");
    COMPILE("concat.comp",       "concat_f32.spv",     "DATA_A_F32=1");
    COMPILE("fill.comp",         "fill_f32.spv",       "DATA_A_F32=1");
    COMPILE("diag_mask_inf.comp","diag_mask_inf_f32.spv","DATA_A_F32=1");
    COMPILE("sum_rows.comp",     "sum_rows_f32.spv",   "DATA_A_F32=1");

    printf("== Dequantize ==\n");
    COMPILE("dequant_q4_0.comp",    "dequant_q4_0.spv",    "DATA_A_Q4_0=1");
    COMPILE("dequant_q4_1.comp",    "dequant_q4_1.spv",    "DATA_A_Q4_1=1");
    COMPILE("dequant_q5_0.comp",    "dequant_q5_0.spv",    "DATA_A_Q5_0=1");
    COMPILE("dequant_q5_1.comp",    "dequant_q5_1.spv",    "DATA_A_Q5_1=1");
    COMPILE("dequant_q8_0.comp",    "dequant_q8_0.spv",    "DATA_A_Q8_0=1");
    COMPILE("dequant_q2_k.comp",    "dequant_q2_k.spv",    "DATA_A_Q2_K=1");
    COMPILE("dequant_q3_k.comp",    "dequant_q3_k.spv",    "DATA_A_Q3_K=1");
    COMPILE("dequant_q4_k.comp",    "dequant_q4_k.spv",    "DATA_A_Q4_K=1");
    COMPILE("dequant_q5_k.comp",    "dequant_q5_k.spv",    "DATA_A_Q5_K=1");
    COMPILE("dequant_q6_k.comp",    "dequant_q6_k.spv",    "DATA_A_Q6_K=1");
    COMPILE("dequant_iq4_nl.comp",  "dequant_iq4_nl.spv",  "DATA_A_IQ4_NL=1");
    COMPILE("dequant_iq4_xs.comp",  "dequant_iq4_xs.spv",  "DATA_A_IQ4_XS=1");

    printf("== Matrix-vector multiply ==\n");
    /* F32/F16 inputs (activations always f32, weights vary) */
    COMPILE("mul_mat_vec.comp",  "mul_mat_vec_f32_f32.spv", "DATA_A_F32=1", "DATA_B_F32=1");
    COMPILE("mul_mat_vec.comp",  "mul_mat_vec_f16_f32.spv", "DATA_A_F16=1", "DATA_B_F32=1");
    COMPILE("mul_mat_vec.comp",  "mul_mat_vec_f32_f16.spv", "DATA_A_F32=1", "DATA_B_F16=1");
    COMPILE("mul_mat_vec.comp",  "mul_mat_vec_f16_f16.spv", "DATA_A_F16=1", "DATA_B_F16=1");

    /* Quantized weight variants */
    COMPILE("mul_mat_vecq.comp", "mul_mat_vec_q4_0_f32.spv", "DATA_A_Q4_0=1",   "DATA_B_F32=1");
    COMPILE("mul_mat_vecq.comp", "mul_mat_vec_q4_1_f32.spv", "DATA_A_Q4_1=1",   "DATA_B_F32=1");
    COMPILE("mul_mat_vecq.comp", "mul_mat_vec_q5_0_f32.spv", "DATA_A_Q5_0=1",   "DATA_B_F32=1");
    COMPILE("mul_mat_vecq.comp", "mul_mat_vec_q5_1_f32.spv", "DATA_A_Q5_1=1",   "DATA_B_F32=1");
    COMPILE("mul_mat_vecq.comp", "mul_mat_vec_q8_0_f32.spv", "DATA_A_Q8_0=1",   "DATA_B_F32=1");
    COMPILE("mul_mat_vec_q2_k.comp", "mul_mat_vec_q2_k_f32.spv", "DATA_A_Q2_K=1", "DATA_B_F32=1");
    COMPILE("mul_mat_vec_q3_k.comp", "mul_mat_vec_q3_k_f32.spv", "DATA_A_Q3_K=1", "DATA_B_F32=1");
    COMPILE("mul_mat_vec_q4_k.comp", "mul_mat_vec_q4_k_f32.spv", "DATA_A_Q4_K=1", "DATA_B_F32=1");
    COMPILE("mul_mat_vec_q5_k.comp", "mul_mat_vec_q5_k_f32.spv", "DATA_A_Q5_K=1", "DATA_B_F32=1");
    COMPILE("mul_mat_vec_q6_k.comp", "mul_mat_vec_q6_k_f32.spv", "DATA_A_Q6_K=1", "DATA_B_F32=1");
    COMPILE("mul_mat_vec_iq4_nl.comp","mul_mat_vec_iq4_nl_f32.spv","DATA_A_IQ4_NL=1","DATA_B_F32=1");
    COMPILE("mul_mat_vec_iq4_xs.comp","mul_mat_vec_iq4_xs_f32.spv","DATA_A_IQ4_XS=1","DATA_B_F32=1");

    printf("== General matrix-matrix multiply ==\n");
    COMPILE("mul_mm.comp", "mul_mm_f32_f32.spv", "DATA_A_F32=1", "DATA_B_F32=1");
    COMPILE("mul_mm.comp", "mul_mm_f16_f32.spv", "DATA_A_F16=1", "DATA_B_F32=1");
    COMPILE("mul_mm.comp", "mul_mm_f32_f16.spv", "DATA_A_F32=1", "DATA_B_F16=1");
    COMPILE("mul_mm.comp", "mul_mm_q4_k_f32.spv","DATA_A_Q4_K=1","DATA_B_F32=1");

    printf("== Flash attention ==\n");
    COMPILE("flash_attn.comp", "flash_attn_f16.spv",  "DATA_A_F16=1", "BLOCK_SIZE=32");
    COMPILE("flash_attn.comp", "flash_attn_f32.spv",  "DATA_A_F32=1", "BLOCK_SIZE=32");
    COMPILE("flash_attn_split_k_reduce.comp", "flash_attn_reduce.spv", "DATA_A_F32=1");

    printf("== Misc ==\n");
    COMPILE("arange.comp",     "arange_f32.spv",      "DATA_A_F32=1");
    COMPILE("timestep_embedding.comp","timestep_embedding_f32.spv","DATA_A_F32=1");
    COMPILE("swiglu.comp",     "swiglu_f32.spv",      "DATA_A_F32=1");
    COMPILE("geglu.comp",      "geglu_f32.spv",       "DATA_A_F32=1");
    COMPILE("argmax.comp",     "argmax_f32.spv",      "DATA_A_F32=1");
    COMPILE("argsort.comp",    "argsort_f32.spv",     "DATA_A_F32=1");
    COMPILE("pool2d.comp",     "pool2d_f32.spv",      "DATA_A_F32=1");
    COMPILE("im2col.comp",     "im2col_f32.spv",      "DATA_A_F32=1");
    COMPILE("conv2d_mm.comp",  "conv2d_mm_f32.spv",   "DATA_A_F32=1");
    COMPILE("upscale.comp",    "upscale_f32.spv",     "DATA_A_F32=1");
    COMPILE("opt_step_adamw.comp","opt_step_adamw.spv","DATA_A_F32=1");
    COMPILE("quantize_q8_1.comp","quantize_q8_1.spv", "DATA_A_F32=1");

    shaderc_compiler_release(compiler);

    printf("\nCompiled %d shaders, %d failed.\n", total, failed);
    return failed > 0 ? 1 : 0;
}
