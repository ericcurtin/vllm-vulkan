// SPDX-License-Identifier: Apache-2.0
//! Snapshot of all VLLM_VULKAN_* runtime flags, read once at model construction.
//! Reading env at construction (not per dispatch) removes the global-lock storm
//! and the load-vs-dispatch mutation hazard (a flag change mid-process can no
//! longer desync weight layout from kernel selection).

// Variant names mirror the on-disk/CLI format tags (VLLM_VULKAN_QUANT values);
// Q4_K's underscore-digit-underscore shape trips the camel-case lint despite
// being the clearest name for this domain.
#[allow(non_camel_case_types)]
// F32/Mlx4/Nvfp4 are part of the complete format enum (documented residency
// formats used elsewhere in the loader — mlx4/nvfp4 have their own dedicated
// dispatch, not yet unified into GpuWeight); not all are constructed via
// `from_env_str` today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum QuantFormat { F32, F16, Bf16, Q8_0, Q4_0, Q4_K, Mlx4, Nvfp4, Fp8 }

impl QuantFormat {
    pub fn from_env_str(s: &str) -> Self {
        match s {
            "q8_0" => Self::Q8_0, "q4_0" => Self::Q4_0, "q4_k" => Self::Q4_K,
            "bf16" => Self::Bf16, _ => Self::F16, // default weight residency = f16
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnKernel { Scalar, Sg, Coop }

#[derive(Clone, Debug)]
pub struct Flags {
    pub quant: QuantFormat,
    pub matvec_rows: Option<u32>,
    pub use_subgroup: bool,
    pub unified: bool,
    // `unified_1cb_enabled()` in lib.rs deliberately re-reads the env live
    // (per-layer, not per-process) to let one process validate ref/2-CB/1-CB
    // without a reload; this snapshot field is kept for struct completeness.
    #[allow(dead_code)]
    pub unified_1cb: bool,
    pub resident: bool,
    pub gemma_resident: bool,
    /// VLLM_VULKAN_GEMMA_RESIDENT_1CB — the single-node Gemma4 decode fold that
    /// collapses `gemma_resident_layer`'s TWO command buffers (CB1 = norms/qkv/
    /// qk-norm/RoPE, host SDPA round-trip, CB2 = o_proj/FFN) into ONE CB/layer.
    /// K/V stay GPU-resident (in-CB buffer-copy of GR_K/GR_V into `gpu_kv[layer]`)
    /// and attention runs IN-CB via the `_sg` decode kernel over that resident KV
    /// (sliding-window via `window_start`, per-layer kv-heads/head_dim, value-less
    /// global V from GR_V) — eliminating the per-layer host attn round-trip and
    /// the 2nd fenced submit that boundary forced. Only host work left is the tiny
    /// `layer_scalar` [h] readback (same as `gemma_resident_layer`'s tail; g12b/
    /// g31b have no PLE). Requires `VLLM_VULKAN_GEMMA_RESIDENT` semantics (same
    /// `gres_bufs`/norm-weight setup); engaged from `forward_gemma_gpu_resident`
    /// when set AND `!cfg.has_ple()`. Default OFF: gated on argmax-exact + cos≈1.0
    /// vs the 2-CB resident path over a multi-token decode.
    pub gemma_resident_1cb: bool,
    /// VLLM_VULKAN_KV_RING_DISABLE — gate/debug knob for Phase-0 per-layer KV
    /// sizing. When set, the gemma resident 1-CB path forces every layer's
    /// GPU-resident KV plane back to a full `max_seq` allocation with absolute
    /// (ring_capacity=0) addressing — i.e. the pre-ring behaviour. Used to A/B
    /// the window-sized ring planes (default) against the uniform-alloc build on
    /// the IDENTICAL `_sg` shader so the comparison isolates purely the ring
    /// addressing (must be argmax-exact + cos=1.0). Default OFF (rings ON).
    pub kv_ring_disable: bool,
    pub gpu_sdpa: bool,
    pub prefill_sdpa: bool,
    pub gemma_gpu_attn: bool,
    /// LEVER 2 (VLLM_VULKAN_GEMMA_1CB): the Gemma4 TP-path analog of qwen's
    /// `q35_1cb` / WS3 resident restructure. When on, `gemma_attn_tp` /
    /// `gemma_mlp_tp` fold their per-projection submits + the CPU
    /// q/k/v-norm + RoPE (attn) and GELU/mul glue (mlp) into the
    /// PROVEN `gemma_resident_layer` fused-CB dispatch sequences over the
    /// persistent `gres_bufs`, sharded to this rank (r_num_q / r_inter).
    /// This removes the un-fused per-projection host round-trips and the
    /// single-threaded host norm/rope work — the structural lever behind the
    /// 3.5→~1.5 ms/layer gap. Default OFF: reuses the resident path's GPU
    /// norm/rope/gelu math (cos≈1.0 vs CPU, argmax-safe per that path's gate),
    /// but authored blind (no Mac Vulkan device) → the cluster COHERENT +
    /// argmax-exact ("Paris.", 236773) gate is required before shipping.
    /// Independent of `gpu_sdpa`: the SDPA kernel choice inside the 1cb attn
    /// path still honors `VLLM_VULKAN_GPU_SDPA` via the shared `gemma_tp_sdpa`.
    pub gemma_1cb: bool,
    /// LEVER 1 (VLLM_VULKAN_GEMMA_1CB_FULL) — COMPLETE the TP fusion the partial
    /// `gemma_1cb` leaves half-done. `gemma_1cb` folds the attn/mlp PROJECTIONS
    /// into resident fused CBs but leaves the 4 per-layer norms (input /
    /// post-attn / pre-ffn / post-ffn), the 2 residual adds and layer_scalar on
    /// the single-core CPU (the `gemma_host` bucket) with their own submit/fence
    /// churn. When this is on, `forward_tp_gemma` runs the fully-fused per-layer
    /// path: hidden stays GPU-resident in the `gres_bufs` HA/HB ping-pong across
    /// the WHOLE layer, every norm/residual runs as a GPU dispatch (mirroring the
    /// proven single-node `gemma_resident_layer` op sequence), and the only host
    /// boundaries left are the 3 unavoidable ones — attention SDPA + KV append,
    /// and the two TP all-reduces. Submits drop to 4/layer (attn-front, o_proj,
    /// post-attn-norm+mlp, post-ffn-norm+residual tail) with ZERO host norm
    /// compute. Supersedes `gemma_1cb` when set (checked first). layer_scalar
    /// (and E2B PLE) stay on the host via a tiny [h] readback, exactly like
    /// `gemma_resident_layer`'s host PLE/scalar tail. Default OFF: authored blind
    /// (no Mac Vulkan device) → gated on the cluster COHERENT + argmax-exact
    /// ("Paris.", 236773) A/B vs partial `gemma_1cb`.
    pub gemma_1cb_full: bool,
    /// LEVER 2 (VLLM_VULKAN_GEMMA_FP8_Q8) — requantize the FP8 attention
    /// projections (q/k/v/o_proj) to q8_0 AT LOAD (dequantize_fp8 →
    /// quantize_q8_0), so the decode-hot attn matvec dispatches the cheaper
    /// `mul_mat_vec_q8_0deq_f32_f32` kernel instead of the ALU-heavy e4m3 FP8
    /// dequant-matmul (`gemma_attn_mv` = 53.5ms/tok, ALU-bound on the
    /// matrix-core-less GFX1013). Same nemotron mamba-q8 requant pattern
    /// (1.41x there); q8_0's per-block int8 dequant is cheaper VALU than e4m3,
    /// and q8_0 ≈ fp8 footprint so no GTT increase. Requant runs on the already
    /// TP-sharded per-rank slice, gated to `self_attn.*` weights whose
    /// (out*in) % 32 == 0 (falls back to the FP8 upload otherwise). Default OFF:
    /// q8_0 is lossy vs fp8 → gated on argmax-exact / cos≈0.9999 vs the FP8
    /// baseline (CPU golden `fp8_to_q8_requant_matvec_cos` asserts cos≈1.0).
    pub gemma_fp8_q8: bool,
    /// H1 lever (VLLM_VULKAN_GEMMA_MLP_Q4) — requantize the gemma-4-12B MLP
    /// projections (gate/up/down_proj, all layers) from the checkpoint's native
    /// mlx-affine **8-bit** down to mlx4 **4-bit** (group 64) AT LOAD, in
    /// `load_gemma_resident_weights`. The 8-bit MLP is ~8.5B of the 12B params
    /// and ~2x Q4_K_M's bytes; 4-bit halves MLP matvec bandwidth AND drops MLP
    /// residency ~9.0GB -> ~5.3GB, letting the full 48L fit ONE node (killing
    /// the PP-2 hop llama.cpp avoids). Reuses the validated 4-bit attention
    /// mlx4 upload + `record_gemma_mv` dispatch verbatim (the MLP tensors just
    /// arrive as `ProjWeight::Mlx4` instead of `F32`). LOSSY (8->4 re-round on
    /// a QAT-4bit checkpoint whose MLP was trained at 8-bit) -> default OFF,
    /// gated on the CPU per-tensor round-trip error + full-48L argmax-agreement
    /// tests (`gemma12b_mlp_q4_*`); single-node-4bit vs PP-2-8bit is a clean A/B.
    pub gemma_mlp_q4: bool,
    /// H2 lever (VLLM_VULKAN_GEMMA_LMHEAD_Q4) — requantize the gemma-4-12B tied
    /// embed/lm_head table (`model.embed_tokens.weight`, 262144×3840) from the
    /// GPU-resident **f16** copy down to mlx4 **4-bit** (group 64) AT LOAD, in
    /// the `gemma4_unified` embed-upload block. The lm_head matvec streams the
    /// full ~2GB f16 vocab table EVERY token (~10ms in one fence, the standout
    /// gemma-12B decode cost); 4-bit cuts that stream to ~0.5GB (packed+scale+
    /// bias) → ~7.5ms saved, same class as `GEMMA_MLP_Q4`. Only the GPU lm_head
    /// PROJECTION copy is quantized — the host f16 input-embed LOOKUP copy
    /// (`q35_f16_host`) stays f16-exact, so the input embedding is unaffected;
    /// only the SENSITIVE logits projection is 4-bit. LOSSY → default OFF, gated
    /// on the CPU argmax-agreement gate (`gemma12b_lmhead_q4_gate`). Reuses the
    /// mlx4 upload + `gemma_res_mv_kind`/`record_gemma_mv` dispatch; `gemma_final`
    /// branches on the embed weight's own `format` (Mlx4 → mlx4 matvec, else the
    /// legacy f16 matvec).
    pub gemma_lmhead_q4: bool,
    pub qwen35_gpu: bool,
    /// item-4a: route the Qwen3.6 full-attention decode seam's SDPA through
    /// the resident `_sg` GPU kernel (`gpu_kv_append` + `gpu_sdpa_resident`)
    /// instead of `model::cpu_sdpa` over the host `KvCache`. Requires
    /// `VLLM_VULKAN_ATTN=sg` (falls back to `cpu_sdpa` otherwise, or if the
    /// GPU call returns `None`). Default OFF: ships alongside the existing
    /// host-SDPA path unchanged so the cluster A/B baseline is untouched;
    /// the host `KvCache` is still appended-to either way (KvStore export
    /// stays host-f32-only under this flag — see item 4b for the KV-RAM win).
    pub q35_gpu_attn: bool,
    /// item-4b (RESERVED, NOT YET WIRED): f16-resident Qwen3.6 full-attn KV
    /// (paged_attn_decode_f16_sg + an f16 gpu_kv plane), for the KV-RAM
    /// halving win (~17.2GB -> ~8.6GB @128k, 27B geometry). The `_f16_sg`
    /// shader is built and registered (see `paged_attn_decode_f16_sg.comp`),
    /// but the gpu_kv f16 plane/append, the dispatch wiring, and — the part
    /// that actually frees the RAM — retiring the host f32 `KvCache` in
    /// favor of it are deliberately deferred: dropping the host copy forces
    /// `qwen35::export_prefix`/`import_prefix` (the just-landed KvStore NAS
    /// export path) to become device+dtype-aware (PFX2 + a dtype byte folded
    /// into `prefix_fingerprint`), which is real surgery on a feature that
    /// just landed and deserves its own change + its own cluster gate
    /// (32k-context argmax-exact accuracy check). Flag is read but currently
    /// a no-op; flip it on only once that follow-up lands.
    #[allow(dead_code)]
    pub q35_kv_f16: bool,
    // Vestigial since WS2 (shared expert + router moved onto the GPU): the
    // rayon CPU-glue path this gated no longer exists on the GPU-resident MoE
    // path. Snapshot kept for struct completeness / env-contract stability.
    #[allow(dead_code)]
    pub qwen35_resident: bool,
    pub q35_parrec: bool,
    pub dn_gpu: bool,
    /// WS3: the qwen3.6 resident per-layer CB consolidation (norms/residuals/
    /// MoE tail on GPU, hidden GPU-resident, 1 fenced CB per linear layer).
    pub q35_1cb: bool,
    /// VLLM_VULKAN_Q35_TP_FUSED (default ON as of 2026-07-25; set =0 to force the
    /// host-orchestrated oracle): collapse the per-reduce-segment host-orchestrated
    /// qwen3_5 TP-4 compute into ONE fused command buffer per segment, halving the
    /// submit tax on the dense TP-4 workhorse. Flipped default-ON after a broad-prompt
    /// A/B (code/prose/multilingual/numeric/650-tok long-context, 128 decode tok each)
    /// was argmax-exact + gen byte-identical vs the oracle on every prompt, step0
    /// logits max_abs_err ~1e-6 (sub-ULP silu-exp only). When on,
    /// `qwen35_dense_mlp_tp` runs gate/up -> `swiglu_f32` -> down in a single CB
    /// (was: gate/up submit, host silu+mul, down submit = 2 submits -> 1), and
    /// `qwen35_delta_net_tp` runs in_proj*4 -> conv -> qknorm -> gdn_step ->
    /// out_proj in a single CB via the rank-sharded GDN kernels (2 submits -> 1).
    /// The all-reduce boundaries are UNCHANGED (still two reduces/layer); only
    /// the intra-segment submits collapse. The host-orchestrated path stays the
    /// default correctness oracle. Full-attn layers keep the host path (GPU-
    /// resident per-rank KV is a separate lift). Reuses the single-node-validated
    /// `swiglu_f32` / `q35_dn_conv_step` / `q35_gdn_qknorm` / `q35_gdn_step`
    /// kernels with per-rank (1/N) head/channel dims.
    pub q35_tp_fused: bool,
    pub moe_gpu: bool,
    pub moe_host_free: bool,
    // Mirrors `model::moe_q4_resident_enabled()` (a separate cached accessor
    // in the model module); kept here for struct completeness, not yet wired
    // as the single source of truth for that decision.
    #[allow(dead_code)]
    pub moe_q4_resident: bool,
    /// Phase B grouped-expert MoE GEMM (VLLM_VULKAN_MOE_GEMM; default ON, set
    /// `=0` to force the per-token baseline; see plan-phaseB-moe-mulmatid.md).
    /// PREFILL-ONLY: reached solely from `forward_qwen35_prefill_impl` via
    /// `qwen35_moe_mlp_prefill_gpu` — decode (`qwen35_moe_mlp_gpu_resident`) is
    /// never touched, so this cannot regress decode ms/tok. `=1` routes
    /// prefill's routed-expert matmuls through the grouped
    /// `matmul_mlx4_id_f32_fp32` GEMM (one dispatch per gate/up/down over ALL
    /// routed (token,slot) pairs), falling through to the per-token loop if the
    /// grouped path returns None. Cluster-validated 2026-07-11 (GFX1013, 1850):
    /// cos == 1.0 vs the per-token CPU reference (`debug_moe_gemm_correctness`,
    /// t=8/512), and a MoE-block win of 1.62x @T=512 / 1.41x @T=2048 over the
    /// per-token GPU-resident path (per-phase bucket timing). The win compresses
    /// with T because the grouped GPU GEMM itself scales slightly super-linearly
    /// (memory-bound on the O(T*top_k*mi) intermediates); the host tail
    /// (route/combine) is rayon-parallel and no longer a factor.
    pub moe_gemm: bool,
    /// Epilogue-fused grouped MoE GEMM (VLLM_VULKAN_MOE_GEMM_FUSED; default
    /// OFF — see plan-epilogue-fused-moe-gemm.md). Only consulted when
    /// `moe_gemm` is already ON. `=1` replaces the routed gate GEMM + up GEMM
    /// + silu_f32 + mul_f32_f32_f32 (4 dispatches, 2 barriers) inside
    /// `qwen35_moe_mlp_prefill_gpu_grouped` with ONE dispatch of
    /// `matmul_mlx4_id_gateup_silu_f32_fp32` that computes silu(gate)*up in
    /// its store epilogue, writing `mid_r` directly (gu_gate/gu_up/act_r are
    /// never allocated). Cluster-unvalidated as of this change — cos=1.0 vs
    /// the grouped path is REQUIRED before flipping the default (see
    /// `debug_moe_gemm_fused_correctness`).
    pub moe_gemm_fused: bool,
    /// Phase 3 down-combine fusion (VLLM_VULKAN_MOE_GEMM_COMBINE; default OFF —
    /// plan-epilogue-fused-moe-gemm.md §5). Only consulted from
    /// `qwen35_moe_mlp_prefill_gpu_grouped` (grouped prefill path). `=1`
    /// replaces the host rayon score-weighted combine + the 16384-f/token
    /// `down_out` host readback with ONE batched `q35_moe_accum_batched`
    /// dispatch that reduces the routed [T,top_k,h] down output + sigmoid-gated
    /// shared expert into out[T,h] entirely on-GPU (readback drops to 2048
    /// f/token). Independent of `moe_gemm_fused` so the cluster can A/B
    /// Phase-3-combine vs Phase-1+2 (gate/up fusion) vs the plain grouped
    /// baseline separately. Gather-reduce (no scatter) => atomic-free +
    /// deterministic; cos=1.0 vs the host combine is REQUIRED before flipping
    /// the default (see `debug_moe_gemm_fused_correctness`, `test_combine=true`).
    /// The [T,top_k,h] `down_out` is still MATERIALIZED in VRAM (the down GEMM
    /// still writes it) — true in-epilogue combine would need cross-workgroup
    /// float atomics (a token's top_k slots land in top_k different z-experts),
    /// unverified on RADV GFX1013; this pass kills the host combine + shrinks
    /// the readback without them.
    pub moe_gemm_combine: bool,
    pub native_comm: bool,
    pub reg_reduce: bool,
    /// FUSED native-vCCL PP hop for Gemma (`VLLM_VULKAN_GEMMA_NATIVE_HOP`,
    /// default OFF). When on (and PP + a comm handle is set), the PP driver
    /// calls `pp_step_gemma` — recv the composite `[hidden+ple+tkv]` message
    /// natively into a `comm_register`'d scratch, run the SAME `forward_pp_gemma`
    /// compute body (KV-share / `gemma_hidden_ring` preserved bit-for-bit), then
    /// native-send onward or Rust-argmax on the last stage. Kills the per-hop
    /// `Vec<f32>→PyList` marshal `forward_pp_gemma` otherwise pays. Gemma-12B is
    /// PP-2 (one hop), so this is a small (~2 ms/tok) lever; correctness/parity
    /// dominates. No effect unless PP + native comm. Silent no-op otherwise.
    pub gemma_native_hop: bool,
    pub tp_bf16_reduce: bool,
    pub nvfp4_gpu: bool,
    /// E4M3-RESIDENT NVFP4 scales (VLLM_VULKAN_NVFP4_E4M3_SCALES=1, default OFF).
    /// The OOM-vs-fit footprint lever for the >=150B NVFP4 tier. When OFF (the
    /// current, cluster-validated path) the loader folds the two-level NVFP4
    /// scale to one f32/(row,group) at upload (`nvfp4_fold_scales`) — 32 bits/16
    /// elems = 2.0 bits/param on top of the 4-bit nibble => NVFP4 = 6.0
    /// bits/param, LARGER than mlx4-affine (5.0). When ON, the RAW per-group
    /// e4m3 block scale (`.weight_scale`, 1 byte) stays resident and the
    /// per-tensor f32 global (`.weight_scale_2`) rides in the push constant:
    /// 8 bits/16 = 0.5 bits/param => NVFP4 = 4.5 bits/param (saves ~1.5
    /// bits/param, ~36GB on Step-3.7-198B — the difference between OOM and fit
    /// at 10 nodes), AND cuts the group=16 scale-buffer traffic 4x (1 byte vs 4).
    /// Cost: the e4m3 decode moves back into the matvec inner loop, but NVFP4
    /// only feeds the big BW-bound mlp/expert shapes where the ALU hides under
    /// bandwidth, so it is ~free there. Option (a) — true e4m3-resident (raw
    /// block scale + separate global), the LOSSLESS path (nothing is re-encoded;
    /// the on-disk e4m3 bytes are stored verbatim). The
    /// `mul_mat_vec_nvfp4_e4m3_f32_f32` shader decodes e4m3 via the SAME
    /// arithmetic `kE4M3_decode` (bit-equal to `model::e4m3_to_f32`; the fp8
    /// const-LUT is only the default-OFF oracle) then applies global,
    /// so it is BIT-EXACT vs the f32-fold kernel (proven by
    /// `nvfp4_e4m3_resident_matches_f32_fold`). Only consulted when `nvfp4_gpu`
    /// is already ON. Default OFF so it is a safe carry; flip on for the ≥150B
    /// NVFP4 tier (cluster A/B: gemma-31B-NVFP4 argmax-exact "Paris." + GTT/rank
    /// drop is the cheap check; Step-3.7-198B fitting at 10 nodes is the payoff).
    pub nvfp4_e4m3_scales: bool,
    /// MLX4-affine 4-bit DENSE (GatedDeltaNet/attn) projection residency
    /// (VLLM_VULKAN_DENSE_Q4_RESIDENT=1). Off by default: ships alongside the
    /// existing f32-dequant -> q8_0-requantize path unchanged so the cluster
    /// A/B baseline is the current q8 behavior; flip on to measure the
    /// bandwidth win (mirrors the MoE 4-bit-resident kill switch, but that one
    /// (`moe_q4_resident_enabled()` in model.rs) is default-ON because MoE
    /// residency is required to fit on a 15GB node — dense residency is a pure
    /// perf/precision win with no memory-fit requirement, so it stays opt-in
    /// until the 27B-dense TP-4 A/B gate passes).
    pub dense_q4_resident: bool,
    pub attn: AttnKernel,
    pub spin: bool,
    pub profile: bool,
    pub gil_release: bool,
    pub cb_ring: bool,
    /// TP vocab-sharded lm_head kill switch (VLLM_VULKAN_TP_SHARD_LMHEAD=0 to
    /// revert to the replicated lm_head). Default ON: sharding removes the
    /// >1GB single-alloc GTT edge failure at 27B-class vocab sizes and is
    /// value-bit-identical per logit vs replicated (same rows, same input),
    /// so there's no accuracy argument for opt-in. Actual gating also
    /// requires `tp_size > 1` AND native comm + `vcclAllGather` availability
    /// (checked at the loader call site, not here — this snapshot only holds
    /// the user's explicit override).
    pub tp_shard_lmhead: bool,
    /// MLX4 MoE-down w8sg kill switch (VLLM_VULKAN_MLX4_W8=0 reverts to the
    /// legacy `mul_mat_vec_mlx4_f32_f32` geometry winner). Default ON: the
    /// 1850MHz-pinned re-sweep (perf/matvec-batch-dispatch) showed the
    /// word-granular w8-load + subgroupAdd-reduction shader beating the wired
    /// v1 winner by 49-81% across every (bs,rows) combo on the down
    /// (k=512,n=2048) MoE dispatch shape specifically — gate/up (k=2048,n=512)
    /// stayed below the 25% bar and is NOT gated by this flag (still legacy).
    pub mlx4_w8sg_down: bool,
    /// MLX4 VALU-bound repack refactor for the DENSE decode path
    /// (VLLM_VULKAN_MLX4_REPACK, default ON since 2026-07-26; =0 kill-switch).
    /// When ON, dense mlx4 4-bit
    /// matvecs with k%32==0 && k>=1024 && n>=1024 route to
    /// `mul_mat_vec_mlx4repack_f32_f32` (dwordx4 word-granular load + scale/bias
    /// once-per-chunk + fma-factored affine + subgroupAdd reduction) instead of
    /// the v1 `mul_mat_vec_mlx4_f32_f32` bs512/r8 scalar-per-nibble kernel. The
    /// fma factoring + subgroup reorder change f32 rounding (~ULP) -> re-gated
    /// argmax-exact everywhere (cos=1.0, mae~ULP), NOT byte-identity. The routing
    /// shape-guard structurally excludes unswept small shapes; no known regression
    /// regime. Default ON (qwen-27B mlx4 TP-4 ~1.06x); =0 restores the v1 oracle.
    pub mlx4_repack: bool,
    /// Geometry lever on top of `mlx4_repack` (VLLM_VULKAN_MLX4_REPACK_R8,
    /// default OFF). Lifts the wired repack decode geometry from bs64/r4 to the
    /// on-node GPU-tick-swept per-shape r8 winner: short-k (k<=n, gate/up/lm_head)
    /// -> bs64/r8, long-k (k>n, down_proj) -> bs128/r8. Same shader (r8 is a
    /// NUM_ROWS spec-const of `mul_mat_vec_mlx4repack_f32_f32`, already creation-
    /// vetted on GFX1013), so argmax-exact/cos=1.0 vs the r4 default — the r8
    /// puts 8 independent dwordx4 weight loads in flight per chunk, closing the
    /// last MLP-latency gap to the 263 GB/s unpack-ALU ceiling. Node microbench
    /// (qwen27B-4bit, GPU-tick): gate 221->245, down 216->240, lm_head 237->283
    /// GB/s (~1.11-1.19x/op). Default ON since 2026-07-30 (fleet e2e A/B, clock-
    /// pinned 1850: gemma-4-12B single-node 50.46->47.33 ms/tok 1.066x argmax
    /// byte-identical; qwen3.6-35B-A3B PP-5 24.27->24.26 flat, argmax identical
    /// [n>=1024 clause misses the n=512 experts]; 27B-dense TP-4 argmax byte-
    /// identical [argmax=2614 oracle-exact] + ms/tok flat within noise [2x2
    /// STEPS=48 interleaved: R8=0 71.3/68.1 vs R8=1 67.2/70.7 mean, arms cross
    /// over, min ~64 both — decode is comm-Amdahl-capped]. No mlx4 model
    /// regresses). =0 restores the r4 geometry baseline.
    pub mlx4_repack_r8: bool,
    /// f16-resident MoE expert scales/biases (VLLM_VULKAN_MOE_F16_SCALES, default
    /// OFF). When ON, `ensure_moe_gpu_layer` uploads each routed-expert's affine
    /// scales+biases as f16 (half the GTT of f32) and the routed-expert matvec
    /// dispatch reads them via the `mul_mat_vec_mlx4_f16scale_f32_f32` variant.
    /// The scales are bf16 on disk; f16 holds every in-range bf16 value EXACTLY
    /// (10 vs 7 mantissa bits), so this is bit-identical to the f32 oracle — the
    /// upload's exact round-trip guard rejects (falls back, layer stays host/CPU)
    /// any tensor with an out-of-f16-range value. This is the 122B-A10B PP-6
    /// fit-enabler: −~1.2GB/stage of expert-scale GTT, ~14.8→13.6GB/node. Also a
    /// general MoE load-footprint win (helps 35B-A3B at PP-2). Prefill's grouped
    /// MUL_MAT_ID GEMM (`matmul_mlx4_id`) still reads f32 scales, so when this is
    /// ON prefill falls back to the per-token resident matvec path (also f16).
    pub moe_f16_scales: bool,
    /// Kimi decode lever #1: hold the MoE SHARED-EXPERT weights GPU-resident
    /// (VLLM_VULKAN_KIMI_SHARED_RESIDENT, default ON). When ON, `KimiGpuStage`
    /// uploads the packed 4-bit shared-expert gate/up/down ONCE (as `GpuMatR`,
    /// exactly like the routed experts / lm_head) and the per-token MoE combine
    /// dispatches `matvec_mlx4` against the held buffers. When OFF (=0), it
    /// restores the legacy path: dequantize the shared expert to host f32 at
    /// load and `up_f32`+`matvec_f32` it every layer every token — the ~1.47
    /// GB/token single-core host memcpy this lever kills. The in-kernel mlx4
    /// dequant is the same affine math as the f32 path but reorders the dot
    /// accumulation → argmax-exact / cos≈1.0 (the already-accepted KimiGpuStage
    /// decode tolerance), NOT byte-identity. Default ON: the memcpy is the #1
    /// dominant Kimi PP decode term; OFF is the A/B and bit-exact fallback.
    pub kimi_shared_resident: bool,
    /// Kimi decode lever #2: hold the MLA projection weights GPU-resident
    /// (VLLM_VULKAN_KIMI_MLA_RESIDENT, default ON). When ON, `KimiGpuStage`
    /// uploads the packed 4-bit q_proj / kv_a_proj_with_mqa / kv_b_proj /
    /// o_proj of the 7 MLA layers ONCE (as `GpuMatR`, like the KDA/MoE
    /// projections) and the per-token MLA decode dispatches `matvec_mlx4`
    /// against the held buffers. The softmax-SDPA seam stays on the host
    /// (`kv_a_layernorm`, the scaled-dot softmax over cached K, the V combine
    /// `cpu_sdpa_mla`, the KV append) — the bit-exact attention seam is
    /// UNCHANGED; only the projections move to the GPU. When OFF (=0) it
    /// restores the legacy path: dequantize the MLA weights to host f32 at load
    /// and run the naive single-core `matmul_wt` (`kimi::mla::decode_step`)
    /// every layer every token — the ~116 MB/layer/token host weight-stream
    /// this lever kills. The kv_b_proj decompress is held in its NATURAL
    /// on-disk `[nh*(nope+v), r]` layout (a single matvec whose per-head output
    /// splits into k_nope||v) — NOT the CPU loader's `[r,nope]`-transposed
    /// embed_q/unembed_out — so no row-permute repack is needed. The in-kernel
    /// mlx4 dequant is the same affine math as the f32 path but reorders the
    /// dot accumulation → argmax-exact / cos≈1.0 (the accepted KimiGpuStage
    /// decode tolerance), NOT byte-identity. Default ON; OFF is the A/B and
    /// bit-exact fallback (the CPU `decode_step` oracle).
    pub kimi_mla_resident: bool,
    /// Kimi decode lever #3: hold the layer-0 DENSE MLP weights GPU-resident
    /// (VLLM_VULKAN_KIMI_DENSE_RESIDENT, default ON). Kimi layer 0 is the only
    /// dense (SwiGLU) MLP (`first_k_dense_replace=1`, inter 9216); it lives on
    /// stage-0 rank0 — the slowest PP stage / the critical path. When ON,
    /// `KimiGpuStage` uploads the packed 4-bit gate/up/down ONCE (as `GpuMatR`,
    /// exactly like the routed/shared experts, MLA projections and lm_head) and
    /// the per-token dense forward dispatches `matvec_mlx4` + the existing
    /// silu/mul kernels against the held buffers in ONE command buffer. When OFF
    /// (=0) it restores the legacy path: dequantize gate/up/down to host f32 at
    /// load and run the naive single-core `kimi::dense_forward` every token —
    /// the ~255 MB/token host weight-stream this lever kills (the last
    /// host-streamed weight matmul left on stage 0). Preserves SwiGLU semantics
    /// exactly (silu(gate·x)⊙(up·x) → down). The in-kernel mlx4 dequant is the
    /// same affine math as the f32 path but reorders the projection dot
    /// accumulation → argmax-exact / cos≈1.0 (the accepted KimiGpuStage decode
    /// tolerance), NOT byte-identity. Default ON; OFF is the A/B and bit-exact
    /// fallback (the CPU `dense_forward` oracle).
    pub kimi_dense_resident: bool,
    /// Kimi decode lever #5: collapse each KDA attention layer to ONE command
    /// buffer / ONE fence (VLLM_VULKAN_KIMI_KDA_FUSED, default ON). The current
    /// `kda_step_resident` issues TWO fenced submits with a host seam between
    /// them — the projections (SEGMENT 1) read back to host for the depthwise
    /// conv(k=4) + qk-RMSNorm(eps 1e-6) + per-channel decay(softplus/exp) glue,
    /// which is then re-uploaded for the recurrence (SEGMENT 2). When ON, that
    /// glue moves onto the GPU (reusing `q35_dn_conv_step` / `q35_gdn_qknorm` /
    /// `kda_gdn_step` verbatim + the new `kda_decay` shader over GPU-resident
    /// conv-window / taps / decay-param buffers) so the whole
    /// projections->conv->qknorm->decay->gdn_step->o_proj chain records into ONE
    /// submit — no per-layer host round-trip. The projections + recurrence +
    /// o_proj are the SAME kernels as OFF (bit-identical); only the moved-to-GPU
    /// conv-silu / qknorm-sqrt / decay-exp differ at the last ulp (GPU intrinsic
    /// vs libm, accumulation order bit-identical) → argmax-exact / cos≈1.0, the
    /// accepted KimiGpuStage decode tolerance. OFF (=0) is the 2-submit
    /// host-seam path (the exact `kda::decode_step` oracle / bit-exact fallback).
    pub kimi_kda_fused: bool,
    /// NVFP4 VALU-bound repack refactor
    /// (VLLM_VULKAN_NVFP4_REPACK, default ON since 2026-07-26; =0 kill-switch).
    /// The nvfp4 twin of `mlx4_repack`, and the win that LANDS E2E: NVFP4 runs
    /// PP-resident / per-stage compute-bound (nemotron-75B PP-5, gemma-31B,
    /// Laguna, Step-3.7, qwen-27B-NVFP4), NOT the TP-4 comm wall. When ON, NVFP4
    /// matvecs with k%32==0 && k>=1024 && n>=1024 route to
    /// `mul_mat_vec_nvfp4repack_f32_f32` (or the e4m3-resident twin when
    /// `nvfp4_e4m3_scales` is also ON): dwordx4 word-granular load + folded scale
    /// once-per-16-group + E2M1-LUT dot + subgroupAdd, instead of the v1
    /// `mul_mat_vec_nvfp4_f32_f32` bs512 scalar-per-nibble kernel. fma factoring +
    /// subgroup reorder change f32 rounding (~ULP) -> re-gated argmax-exact
    /// (cos=1.0, mae~ULP), NOT byte-identity. Same shape-guard exclusion as
    /// mlx4_repack; no known regression regime. Default ON (nemotron nvfp4 PP-5
    /// ~1.11x); =0 restores the v1 oracle. NOTE: independent of
    /// `nvfp4_e4m3_scales`, which stays default-OFF (a footprint lever, not speed).
    pub nvfp4_repack: bool,
    /// The fp8 subgroup-reduction twin of `mlx4_repack` (VLLM_VULKAN_FP8_REPACK,
    /// default OFF). fp8 is NOT address-gen-bound (s4rig offline ISA rig:
    /// addr_gen/unpack 1.76 vs mlx4-v1's 20.11 — the uint-word load already
    /// amortizes address-gen 4x), so unlike mlx4/nvfp4 there is NO load repack;
    /// this only swaps fp8_fast's LDS log2(BLOCK) barrier-tree reduction for a
    /// subgroupAdd reduction (no LDS/barrier for BLOCK<=64 wave64) → ~1.2-1.4x/op
    /// on the compute-bound nemotron FP8 attn matvecs. Requires fp8_fast. OFF
    /// until an on-node argmax-exact + perf A/B (cluster-gated).
    pub fp8_repack: bool,
    /// L1 decode lever — the SMALL-n (n==512) MoE gate/up sibling of `mlx4_repack`
    /// (VLLM_VULKAN_MLX4_RGU_REPACK, default OFF). The shipped `mlx4_repack`
    /// n>=1024 clause EXCLUDES the Qwen3.6-35B-A3B MoE gate/up shape (k=2048,
    /// n=512) because the on-node dense sweep only covered n>=1024 shapes. But the
    /// s4rig offline ISA classify shows the v1 kernel that gate/up currently
    /// dispatches is address-gen-bound (addr/unpack=20.11, same pathology the
    /// repack fixes to 1.16) — k/n are runtime push-constants so the histogram is
    /// shape-invariant. When ON, mlx4 4-bit matvecs with n==512 && k%32==0 route to
    /// the SAME `mul_mat_vec_mlx4repack_f32_f32_bs64_r4` shader (no new shader; the
    /// bs64/r4 geometry is already compiled). k%32==0 guarantees words_per_row=k/8
    /// is a mult of 4 → every per-expert base is 16B-aligned for the dwordx4 load
    /// (identical alignment argument as the shipped clause). n==512 self-scopes to
    /// the qwen35 gate/up; kept SEPARATE from the default-ON n>=1024 branch so the
    /// shipped path is untouched. Same fma-factored/subgroupAdd rounding → re-gated
    /// argmax-exact (cos=1.0, mae~ULP), NOT byte-identity. Default ON since
    /// 2026-07-30 (qwen3.6-35B-A3B PP-5 fleet A/B: 24.27->20.29 ms/tok, -3.98,
    /// argmax byte-identical; the n==512 self-scope leaves gemma-12B + 27B-dense
    /// argmax-untouched, confirmed on-node); =0 reverts to the v1 kernel oracle.
    pub mlx4_rgu_repack: bool,
    /// Swept geometry-tuned q8_0 matvec for the Qwen3.6 TP forward
    /// (VLLM_VULKAN_Q35_GEOM=1, default OFF). When ON, the qwen3.6 TP q8 matvec
    /// dispatch sites (`qwen35_matvec`/`qwen35_matvec_multi`, MvKind::Plain)
    /// route through `matvec_variant_geom` so the swept-winner (BLOCK_SIZE,rows)
    /// geometry in `geom_pick` fires for the TP-4 sharded 27B projection shapes.
    /// Same dequant math, different tiling → argmax-identical (cos≈1.0). OFF =
    /// byte-identical to the legacy `matvec_variant` r1/r8 dispatch.
    pub q35_geom: bool,
    /// Chunked-alloc mitigation (plan §5, P2): split any single GPU weight
    /// buffer over this many MB into row-boundary chunks (one command buffer,
    /// several `record_to_off` dispatches) instead of one monolithic alloc.
    /// `0` = OFF. NOTE: the plan's default is 512 (always-on safety net); this
    /// ships default-OFF instead — chunking changes the dispatch path for
    /// every existing large-buffer load (including today's working PP/single-
    /// node flagship configs) and could not be validated against a live GPU
    /// in this change (Mac-only implementation window, cluster offline).
    /// Opt in with VLLM_VULKAN_MAX_ALLOC_MB=<n> once validated on-node.
    pub max_alloc_mb: u32,
    /// Nemotron-H-Puzzle: route the Mamba2 mixer `in_proj`/`out_proj` GEMMs
    /// (the FLOP-dominant, ~52.8%-of-prefill projections) through the proven
    /// f16-aligned batched tiled GEMM (`matmul_f16_f32_fp32`) instead of the
    /// CPU `cpu_matmul`. The FP8 weights are dequantized to f32 by the loader,
    /// re-cast to f16, and uploaded GPU-resident at construction. The
    /// sequential conv1d+SSD scan between the two projections stays on CPU
    /// (unavoidable recurrence). Default OFF; the CPU path is the correctness
    /// reference. GPU dispatch requires a live Vulkan device — on a
    /// device-less host (Mac) the projection transparently falls back to
    /// `cpu_matmul`, so this is measured/validated on the cluster.
    pub nemotron_gpu_mamba: bool,
    /// Nemotron-H-Puzzle: load the whole model GPU-resident in QUANTIZED form
    /// (NVFP4 routed experts, FP8 mamba/attn/shared, f16 for the BF16-native
    /// attn/latent projections) and dequant-in-shader through `nem_matvec`,
    /// instead of the CPU loader's dequant-to-f32-host (~41GB/PP-5-stage, which
    /// OOM-kills the 14GB nodes). This is the memory-fit fix that lets a 75B
    /// PP-5 stage fit the ~13.3GB GTT budget. Requires a live Vulkan device
    /// (hard-errors if absent — there is no host fallback, that is the whole
    /// point). Default OFF; mirrors qwen's VLLM_VULKAN_DEVICE_WEIGHTS residency.
    pub nemotron_resident: bool,
    /// Nemotron-H-Puzzle: WS3-style resident-CB decode path (collapses the
    /// per-matvec begin_batch/submit_batch/readback storm into fenced
    /// per-layer command buffers). Default OFF; cluster-gated.
    pub nemotron_1cb: bool,
    /// Nemotron-H-Puzzle R1b: collapse the resident MoE layer's 3 waited
    /// fences (CB_1/CB_2/CB_3) down to 1 by moving relu² and the top_k
    /// routed accumulate onto the GPU (relu2_f32 + nemotron_moe_accum
    /// shaders), so the tail becomes a single unwaited CB on the ring.
    /// Default OFF; cluster-gated. Flag OFF leaves the existing per-CB path
    /// (and CPU correctness reference) untouched.
    pub nemotron_moe_tail: bool,
    /// Nemotron-H-Puzzle R2: fold the Mamba2 SSD decode-scan (depthwise
    /// conv1d+SiLU, per-head recurrence+gate, gated RMSNorm) onto the GPU via
    /// `nemotron_ssm_conv_step`/`nemotron_ssd_scan`/`nemotron_gated_rmsnorm`,
    /// so the resident Mamba layer's mandatory host round-trip (the scan)
    /// disappears and the whole layer becomes ONE unwaited ring CB. The GPU
    /// `ssm_state`/`conv_state` become the state of record for a resident
    /// Mamba layer once this flag is on — see
    /// `nemotron::NemotronModel::attach_gpu_mamba_scan` for the all-or-
    /// nothing readiness gating (no per-step CPU/GPU fallback, to avoid a
    /// state desync). Default OFF; cluster-gated. Flag OFF leaves the
    /// existing 2-CB host-scan path (the CPU correctness reference)
    /// byte-identical.
    pub nemotron_gpu_scan: bool,

    /// TP=2×PP lever: re-enable the pipelined CB ring on the TP-sharded resident
    /// span. Without it (`use_ring = ring_active && !tp_shard`) EVERY CB under TP
    /// is a blocking `submit_batch`, so the GPU execution of the non-waited CBs
    /// PP-5 hides on the ring (mamba out_proj CB_B, the whole GPU-scan mamba CB,
    /// attention o_proj CB_B) is fully exposed on the host critical path. With
    /// the flag ON those CBs pipeline exactly as in PP-5; the ONLY extra sync is
    /// a single `wait_batch_pipelined` drain immediately before each MoE layer's
    /// `nem_tp_reduce_mix` (so the tail CB's `NR_MIX` partial is host-visible for
    /// the wire exchange) — a strictly stronger sync than a race, so the result
    /// stays bit-exact. Default OFF; cluster-gated (argmax A/B vs PP-5).
    pub nemotron_tp_ring: bool,

    /// Requant the resident Nemotron mamba `in_proj`/`out_proj` weights from
    /// FP8 to q8_0 at load time (see `nemotron_loader::load_...`'s `is_fp8`
    /// branch). q8_0's block-scalar layout gets a ~3.5x faster matvec on
    /// GFX1013 than the FP8 dequant shader for these two projections. Default
    /// OFF; cluster-gated (accuracy pre-screened by a Mac unit test, but the
    /// end-to-end argmax A/B is cluster-only).
    pub nemotron_mamba_q8: bool,
    /// VLLM_VULKAN_FP8_FAST: dispatch mul_mat_vec_fp8_fast.comp (arithmetic
    /// E4M3 decode + vec4 word loads) instead of the const-LUT + per-element
    /// scalar-byte fp8 matvec (`mul_mat_vec_fp8.comp`). The base kernel is
    /// address-gen-bound: a per-element divergent index into `kE4M3[256]` lowers
    /// to a ~255-wide v_cndmask SELECT CASCADE on ACO/GFX1013 (NOT a load — same
    /// pathology the nvfp4-e4m3 repack fix identified), plus it re-loads a whole
    /// u32 word per element to use one byte. The fast kernel loads the word once,
    /// decodes 4 codes arithmetically (~8 int ops, bit-exact vs the LUT — proven
    /// by fp8_arith_decode_matches_e4m3 + fp8_shader_lut_matches_e4m3), and dot4s
    /// against a vec4 activation load. Default ON (=0 kill-switch reverts to the
    /// LUT oracle for A/B). Decode change is ~ULP-reassociative (dot4 vs scalar
    /// accumulation) -> argmax-exact; on-node perf gate on a free node/staged A/B.
    pub fp8_fast: bool,
    /// Requant resident Nemotron shared_experts.{up,down}_proj FP8->q8_0 at load
    /// (same is_fp8 loader branch as nemotron_mamba_q8; always-on dense MLP added
    /// to every token's routed output per MoE layer). Default OFF; cluster-gated.
    /// Independent of nemotron_mamba_q8 so the two requants A/B separately.
    pub nemotron_shared_q8: bool,
    /// Store the resident Laguna `model.embed_tokens.weight` f16-resident (a
    /// host-coherent GTT buffer, from the BF16-on-disk value) instead of the f32
    /// host table — halves the embed footprint (vocab×hidden: 1.23GB f32 →
    /// 0.62GB f16) on the embed-owning (first) PP stage. f16 holds every in-range
    /// bf16 value exactly. Default OFF; footprint/load-OOM lever, cluster-gated.
    /// See `laguna_loader::load_laguna_resident`.
    pub laguna_embed_f16: bool,
    /// Header-parse + per-tensor `pread` load for the Laguna resident loader
    /// (`VLLM_VULKAN_LAGUNA_PREAD_LOAD`) instead of whole-shard `Mmap` — bounds
    /// peak RSS to one streamed tensor per shard. Default OFF; footprint/load-OOM
    /// lever, cluster-gated. (Design mirrors the Nemotron pread source.)
    pub laguna_pread_load: bool,
    /// PP pre-slice load directory (`VLLM_VULKAN_PP_PRESLICED_DIR`). When set,
    /// a pipeline-parallel rank loads its stage's tensors from a per-stage
    /// pre-sliced safetensors file in this dir (produced offline by
    /// `scripts/coalesce_quant_shards.py slice --pp-bounds …`) instead of
    /// mmapping the monolithic multi-shard checkpoint. The rank opens ONLY its
    /// stage file — it never mmaps/reads the other stages' bytes — cutting the
    /// LOAD-TRANSIENT host-memory peak (whole-shard VMA / pread buffer) and the
    /// per-node staging footprint. The resolver (`model::resolve_pp_stage_shards`)
    /// picks the file whose encoded `[lo,hi)` layer bounds equal the rank's
    /// runtime `[layer_start,layer_end)`, and HARD-FAILS on any mismatch (echoing
    /// both). Default None ⇒ byte-for-byte identical to the monolithic path
    /// (`discover_shards`). Complementary to the `pread` load levers (pread bounds
    /// per-tensor RSS *within* a file; pre-slice removes the other stages' bytes
    /// from the file entirely — the two compose). Load-transient lever only; does
    /// NOT address GPU-resident eager-upload ceilings (e.g. 122B PP-8).
    pub pp_presliced_dir: Option<String>,
    /// GPU-RESIDENT Laguna forward (`VLLM_VULKAN_LAGUNA_RESIDENT`): the
    /// `mt=="laguna"` dispatch loads a `LagunaGpuModel` (quantized-resident
    /// `[start,end)` window, GPU matvec forward) instead of the CPU
    /// `LagunaModel`. Requires a Vulkan device. Default OFF. See
    /// `laguna_gpu::LagunaGpuModel`.
    pub laguna_resident: bool,
    /// RESIDENT 1-CB single-token DECODE fold for the Laguna resident forward
    /// (`VLLM_VULKAN_LAGUNA_1CB`, default OFF). When on, `forward_decode_*`
    /// dispatch the batched-submit fold (`layer_forward_cached_1cb`): each MoE
    /// layer's 10×(gate/up)+shared gate/up matvecs + on-GPU `swiglu_f32` +
    /// 10×down+shared down record into ONE command buffer / ONE submit (was 33),
    /// and attention batches q/k/v/g into one CB (host qk-norm/rope/SDPA/softplus)
    /// + o_proj in a second (was 5). Collapses ~38 per-op submit+readbacks/layer
    /// to ~3. Bit-exact vs the per-op KV-cache decode within f32-reduction noise
    /// (only new numeric = GPU `swiglu_f32` silu vs host libm; cos>=0.999,
    /// argmax-exact — gated by `debug_laguna_1cb`). Decode-only; prefill keeps the
    /// per-op path. No effect unless `laguna_resident`.
    pub laguna_1cb: bool,
    /// GPU YaRN RoPE for Laguna full-attn layers (`VLLM_VULKAN_LAGUNA_YARN_GPU`,
    /// default OFF). When on (and `laguna_1cb`), `attn_cached_1cb` computes the
    /// full-attn partial-rotary YaRN rope on the GPU via `rope_neox_f32_f32`'s
    /// yarn_direct path (inv_freq table = `LagunaConfig::yarn_inv_freq`, mscale =
    /// `full_attention_factor`) instead of host `cpu_rope_yarn`, removing the
    /// host per-step rope for full-attn layers. Sliding layers keep host plain
    /// rope. Bit-exact table; GPU sin/cos is last-ulp vs libm (cos≈1.0,
    /// argmax-exact — gated by `debug_laguna_yarn_rope_gpu` + `debug_laguna_1cb`).
    /// Enabler for the GPU-resident attention span fold. No effect unless
    /// `laguna_1cb`.
    pub laguna_yarn_gpu: bool,
    /// GPU decode-SDPA over the resident K/V planes (`VLLM_VULKAN_LAGUNA_GPU_SDPA`,
    /// default OFF). When on (and `laguna_1cb`), `attn_cached_1cb` reads the
    /// device-resident post-rope K/V planes with the `laguna_gpu_sdpa` subgroup
    /// kernel (q·Kᵀ → scale → online softmax → ·V, one wave64 subgroup per q head)
    /// instead of the host `cpu_sdpa` — no readback. Both attn regimes: full-attn
    /// layers read [0, seq); sliding layers pass the absolute-position clamp
    /// `window_start = seq - min(seq, sliding_window)` (== cpu_sdpa's kv_start).
    /// The span-fold final piece (removes the host SDPA decode wall). Bit-exact vs
    /// host softmax up to GPU exp/subgroupAdd last-ulp (cos≈1.0, argmax-exact —
    /// gated by `debug_laguna_gpusdpa`). No effect unless `laguna_1cb`.
    pub laguna_gpu_sdpa: bool,
    /// Phase-0 per-layer KV sizing for Laguna (`VLLM_VULKAN_LAGUNA_KV_RING`,
    /// default ON — the on-node gate PASSED bit-exact and it is a pure footprint
    /// win; set `=0` to revert to full `max_seq` absolute planes). When set
    /// (and `laguna_1cb`), each SLIDING-window layer's device-resident K/V plane
    /// (`ResidentKvPlane`) is allocated as a `sliding_window`-sized RING instead
    /// of a full `max_seq` plane (36 of 48 layers shrink); absolute position `p`
    /// lives at ring slot `p % capacity`, and the `laguna_gpu_sdpa` shader reads
    /// slot `token_idx % ring_capacity`. Full-attention (YaRN) layers keep the
    /// full plane (`ring_capacity == 0`, byte-identical). Mirrors the gemma
    /// `VLLM_VULKAN_KV_RING_DISABLE` A/B knob but INVERTED (Laguna rings are
    /// opt-in until the on-node gate lands, since the GPU-shader ring read cannot
    /// be validated offline). To keep multi-row prefill bit-exact the sliding
    /// ring layers interleave append→attend per row (each query row reads its
    /// full window before the next append can overwrite a ring slot). With `=0`
    /// the resident plane is a full `max_seq` absolute plane exactly as before.
    pub laguna_kv_ring: bool,
    /// Persistent host-coherent scratch banks for the 1-CB decode path
    /// (`VLLM_VULKAN_LAGUNA_SCRATCH`, default ON since 2026-07-30; =0 kill-switch).
    /// When on (and `laguna_1cb`),
    /// `moe_token_1cb`/`attn_cached_1cb` reuse ONE set of pre-allocated GPU
    /// buffers (sized to the max layer shape, allocated once on first decode)
    /// across every token and layer instead of pool-alloc/free-ing ~44 buffers
    /// per MoE layer + ~10 per attn layer per step — plus direct mapped-memory
    /// writes/reads (no `f32_slice_to_bytes`/`read_f32_buf` temp Vecs) and the
    /// router-weight borrow (no per-token `.to_vec()`). Pure allocation reuse:
    /// bit-identical to OFF (argmax-exact, cos=1.0 — gated by the same
    /// `debug_laguna_1cb`). Decode-only (`new_seq==1`); prefill keeps the per-op
    /// path. No effect unless `laguna_1cb`.
    pub laguna_scratch: bool,
    /// Route Laguna's routed-expert NVFP4 e4m3 matvec through the address-gen-free
    /// REPACK kernel (`mul_mat_vec_nvfp4_e4m3repack_f32_f32_bs64_r4`) instead of the
    /// v1 `mul_mat_vec_nvfp4_e4m3` it dispatches directly (which BYPASSES
    /// `nvfp4_dispatch` — every other model's NVFP4 matvec gets the repack, Laguna
    /// never did). The repack shader already threads `packed_off`/`sb_off`, so the
    /// per-expert slice offsets pass straight through `matvec_nvfp4_e4m3_pc_off`.
    /// Gated on `nvfp4_repack_shape_ok(k,n,gs)` (Laguna gate/up k=3072→n=1024, down
    /// k=1024→n=3072, gs=16 all pass). argmax-exact vs the v1 f32-fold oracle.
    /// Default ON since 2026-07-30 (productionized); =0 reverts to the v1 oracle
    /// for a clean single-node A/B. The primary Laguna decode lever. Silently
    /// no-ops unless the experts are e4m3-resident (`VLLM_VULKAN_NVFP4_E4M3_SCALES`).
    /// See `laguna_gpu::expert_matvec`. (`VLLM_VULKAN_LAGUNA_EXPERT_REPACK`)
    pub laguna_expert_repack: bool,
    /// MoE CB-batch dispatch fold (`VLLM_VULKAN_LAGUNA_CBBATCH`, default OFF).
    /// When on (and `laguna_1cb` + e4m3-resident experts + top_k==10),
    /// `moe_token_1cb` collapses the 10-experts×3-projection = 30 separate
    /// NVFP4 matvec dispatches into 3 expert-batched dispatches (one per
    /// projection, `gl_WorkGroupID.y` = expert slot), the per-expert swiglu into
    /// one flat dispatch, and the routed combine into the concatenated-down
    /// `laguna_moe_accum_b` — cutting per-layer `record_to` /
    /// `update_descriptor_sets` recordings from ~45 to ~9. The batched matvec
    /// reduces in the SAME order as the per-expert kernel, so the result is
    /// BIT-EXACT (gated by `debug_laguna_cbbatch`). No effect unless `laguna_1cb`.
    pub laguna_cbbatch: bool,
    /// GPU per-layer attention MATH for Laguna decode
    /// (`VLLM_VULKAN_LAGUNA_GPU_ATTNMATH`, default OFF). When on (and
    /// `laguna_1cb`), `attn_cached_1cb` moves the two remaining host per-layer
    /// attention-math ops onto the GPU: (1) per-head q/k RMSNorm BEFORE rope
    /// (`rms_norm_f32_mul`, batched over all new rows, bit-exact vs
    /// `cpu_rms_norm_inplace`), and (2) the SLIDING-layer plain NeoX rope
    /// (`rope_neox_f32_f32` plain path, full rotary `head_dim`, θ =
    /// `sliding_rope_theta`) instead of host `cpu_rope`. Full-attn layers use the
    /// GPU-YaRN `rope_neox_f32_f32` yarn_direct path (same as `laguna_yarn_gpu`),
    /// so with this flag ALL attn math is on-GPU. Bit-exact qk-norm; GPU sin/cos
    /// last-ulp vs libm for rope (cos≈1.0, argmax-exact — gated by
    /// `debug_laguna_attnmath` + `debug_laguna_qknorm` + the existing
    /// `debug_laguna_yarn_rope_gpu` sliding split). No effect unless `laguna_1cb`.
    pub laguna_gpu_attnmath: bool,
    /// Fuse the last-stage argmax into Rust for the Laguna PP decode step so the
    /// full `[vocab=100352]` logit Vec never crosses the pyo3 boundary as a
    /// `Vec<f32>→PyList` marshal followed by a pure-Python argmax. When on, the
    /// driver calls `forward_pp_laguna_decode_argmax` on the LAST rank and gets
    /// just `(token_id, logit)` back (same strict-`>` first-max tie-break as the
    /// driver's `max(range(len(v)), key=...)`, so the token is argmax-identical).
    /// The full-logit `forward_pp_laguna_decode` stays for the DUMP /
    /// split-invariance path. Default OFF for a clean single-node A/B.
    /// (`VLLM_VULKAN_LAGUNA_RUST_ARGMAX`)
    pub laguna_rust_argmax: bool,
    /// FUSED native-vCCL PP hop for Laguna (`VLLM_VULKAN_LAGUNA_NATIVE_HOP`,
    /// default OFF). When on, `scripts/pp_laguna.py` calls the single
    /// `VulkanModel::pp_step_laguna` pymethod per stage (recv → stage forward →
    /// send, all in Rust on a registered `[H]` scratch) instead of marshalling
    /// the hidden through Python as a `Vec<f32>→PyList`, `comm.send/recv`, then
    /// `list(recv)` — the ~7ms/collective PyList+GIL callback tax
    /// [[vccl-comm-python-callback-bottleneck]] applied ~5×/token on the PP hop.
    /// Mirrors nemotron/kimi's `pp_step_*`; needs `set_collective_comm` +
    /// `VLLM_VULKAN_NATIVE_COMM!=0`. Bit-exact (same bytes, different transport).
    /// Registration of the hop scratch is under `VLLM_VULKAN_REG_REDUCE`.
    pub laguna_native_hop: bool,
    /// Store the Laguna attention (`self_attn.{q,k,v,o,g}_proj`) + shared-expert
    /// (`shared_expert.{gate,up,down}_proj`) weights Q8_0-RESIDENT (symmetric
    /// int8, per-32-block scale in-block) instead of f16, and route their
    /// `f16_matvec`/`rec_mv` through `mul_mat_vec_q8_0deq_f32_f32`. Halves the
    /// bytes of the ~78%-of-traffic f16 attention/shared slice (the last decode
    /// lever once it is at the BW roofline). The int8/q8 sibling of the f8-attn
    /// lever (`VLLM_VULKAN_LAGUNA_F8_ATTN`, VALIDATED NO-GO on both accuracy and
    /// perf): int8 is uniform-affine (more accurate than E4M3 in the weight
    /// range) with a CHEAP dequant (int8→f32 `v_cvt` + block-scale mul, no LUT),
    /// so it may pass BOTH gates where f8 failed both. Default OFF — the
    /// single-node gate is an accuracy A/B (argmax agreement + logit cos vs the
    /// f16 baseline) + a decode-perf A/B. Routed NVFP4 experts, lm_head, dense
    /// (layer-0) MLP, embed unchanged. (`VLLM_VULKAN_LAGUNA_INT8_ATTN`)
    pub laguna_int8_attn: bool,
    /// Host-residual copy-elision for the 1-CB single-token decode path
    /// (`VLLM_VULKAN_LAGUNA_HOSTFOLD`, default OFF). Composes on top of
    /// `laguna_scratch`: extends the persistent-buffer reuse from the GPU banks to
    /// the remaining HOST-side per-token work in `layer_forward_cached_1cb` /
    /// `attn_cached_1cb_scratch` — the two `cpu_rms_norm` scratch Vecs + two
    /// residual-add temps per layer, the `input/post_attention_layernorm` +
    /// `q/k_norm` weight `.to_vec()` copies (now borrowed, no per-layer copy), and
    /// the q/k/v/g readback `.to_vec()`s (host banks / borrowed mapped slices).
    /// Pure allocation reuse: byte-identical GPU dispatches + host math → bit-exact
    /// (cos=1.0, argmax-exact) vs OFF. Decode-only (`new_seq==1`); no effect unless
    /// `laguna_1cb` + `laguna_scratch`. (`VLLM_VULKAN_LAGUNA_HOSTFOLD`)
    pub laguna_hostfold: bool,
    /// Nemotron-75B MTP Phase 1 (spec-decode acceptance sim): load the
    /// shipped `mtp.*` draft head (host f32/BF16-bits, CPU-only — see
    /// `nemotron_mtp` module doc) alongside the LAST PP stage and expose the
    /// `nem_mtp_draft` pyfn. Default OFF; large host addition (~7GB, see
    /// module doc), only meaningful when this rank owns the last layer range.
    pub nemotron_mtp: bool,
    /// Nemotron-75B MTP Phase 1 DECOUPLED trace-dump: `VLLM_VULKAN_NEMOTRON_MTP_TRACE=<path>`.
    /// Writes, on the LAST PP stage of a NORMAL (no head loaded — independent
    /// of `nemotron_mtp`) run, one record per decode step: the pre-`norm_f`
    /// hidden + the base model's own greedy next-token. Lets the shipped
    /// stack run at its normal PP-5 footprint (no ~7GB head addition, no
    /// OOM) while still capturing everything an OFFLINE, single-node
    /// head-only replay needs — see `nemotron::NemotronModel::
    /// attach_mtp_trace` / `nemotron_mtp` module doc. `None` (default) = no
    /// trace file, zero overhead.
    pub nemotron_mtp_trace: Option<String>,
    /// GEMM campaign Phase 1 kill switch (VLLM_VULKAN_GEMM_F16ALIGNED=0
    /// reverts to the live `matmul_f16_f32_fp32` BM=BN=64 kernel unconditionally).
    /// Default ON: the cluster sweep (2026-07-10, pinned 1850MHz) found
    /// `matmul_f16_f32_f16_aligned` (f16-arith + ALIGNED vec8 loads) 1.76-2.79x
    /// faster than the live kernel on all 5 priority shapes, cos>=0.999 (the
    /// f16-arith correctness gate — NOT bit-exact, see gemm_pick doc-comment).
    /// PROVISIONAL: the sweep ran iters=10 (noisy timing) and only checked cos,
    /// not argmax; the argmax-exact re-validation + a higher-iters re-measure
    /// are queued on the cluster (push_constants::gemm_pick doc-comment).
    pub gemm_f16aligned: bool,
    /// Phase A quant-batched-GEMM kill switch (VLLM_VULKAN_GEMM_QUANT=1 to
    /// enable; see plan-quant-batched-matmul.md). Default OFF — unlike
    /// `gemm_f16aligned` (default ON, cluster-swept), this dispatch path has
    /// no on-node cos/argmax gate yet. OFF = today's serial
    /// `qwen35_matvec`-per-token fallback for quant/aux weights in
    /// `qwen35_gemm` (the current, working A/B baseline); `=1` routes
    /// T>=2 mlx4/q8_0-resident weights through the new quant `mul_mm` GEMM.
    pub gemm_quant: bool,
    /// Batched-prefill cols kill switch (VLLM_VULKAN_QWEN35_PREFILL_COLS; default
    /// ON, set =0 to revert to serial prefill). When ON, the qwen3.6 batched
    /// prefill (`forward_qwen35_prefill_impl`) routes its attn q/k/v/o + GDN
    /// in/out_proj + DENSE-MLP gate/up/down weight projections through the
    /// single-stream `mul_mat_vec_{mlx4,q8_0,f16}_cols` kernels — the weight is
    /// streamed ONCE per <=8-column tile instead of the T serial per-token
    /// matvecs the prefill projections fell through to. Column-tiled inside
    /// `qwen35_gemm` so a prompt of any length is processed in <=8-token
    /// projection tiles (the on-node T=8 sweet spot; tile size overridable via
    /// VLLM_VULKAN_QWEN35_COLS_TILE, default 8). MoE experts keep their grouped
    /// MUL_MAT_ID GEMM (flag left OFF around the MoE block).
    ///
    /// FLIPPED default-ON after the live on-node gate (n55, 2026-08-02, mlx4
    /// 4-bit dense-27B, DENSE_Q4_RESIDENT, sclk 1850, 32-layer resident subset —
    /// single node cannot hold all 64 layers of the TP-4 27B):
    ///   * CORRECTNESS: argmax-EXACT at every position vs the proven serial
    ///     per-token oracle (`debug_qwen35_verify_vs_serial`) across tiles
    ///     {4,6,8} × t {2,4,6,8}, cos=1.0 / maxd ~f16-ULP (5e-6..8e-6); the
    ///     flag-gated prefill hidden is bit-exact (cos=1.0, maxd<=1e-5) vs the
    ///     serial prefill at prompt lengths {64,256,512,1024} for all tiles.
    ///   * PERF: end-to-end prefill 1.78x–2.25x faster (tile=8 best) — L64 2.19x,
    ///     L256 2.25x, L512 2.00x, L1024 1.78x (the win compresses as O(L^2)
    ///     attention + the GDN scan + CPU norms/rope/residuals, all unchanged by
    ///     this flag, grow to dominate prefill at long context). Decode is
    ///     UNCHANGED (8.90 vs 8.89 tok/s) — the flag is prefill-scoped.
    /// OFF remains byte-identical to serial prefill (kill switch).
    pub qwen35_prefill_cols: bool,
    /// Gemma4 batched-prefill cols lever (VLLM_VULKAN_GEMMA_PREFILL_COLS; default
    /// ON as of the on-node GO — see gate note below; set =0 to revert to serial
    /// per-row prefill). When ON, `gemma_prefill_matmul` routes each prefill weight projection
    /// (attn q/k/v/o + MLP gate/up/down; and PLE on E2B) through the SAME
    /// single-stream `mul_mat_vec_{mlx4,q8_0,f16}_cols` kernels the validated
    /// qwen35 cols prefill uses (`qwen35_matvec_cols_tiled`) — the resident weight
    /// is streamed ONCE per <=8-column tile instead of the T per-row matvecs the
    /// per-row `gemma_prefill_matmul` loop issues today. Stacks INDEPENDENTLY with
    /// the gemma windowed-ring prefill (ring = KV footprint; cols = projection
    /// compute). Gemma's mlx4-affine attn + q8_0 MLP resident weights are directly
    /// eligible (same GpuWeight format/aux the cols kernels read); Nvfp4/Fp8/
    /// f16-CPU weights decline (the cols helper returns None) → per-row fallback,
    /// byte-identical. Per-layer KV-head/head-dim variation (global MQA/GQA vs
    /// sliding GQA) and the value-less global layers are handled transparently:
    /// the caller passes the already-correct per-layer (k,n) and simply omits the
    /// v_proj call on value-less globals, so the cols path never sees them.
    /// OFF (=0) is byte-identical to the per-row prefill (the flag skips the cols
    /// delegation entirely; kill switch). ON reorders the f32 reduction per column
    /// exactly like the qwen35 cols path (argmax-exact / cos=1.0, mlx4-cols ACO
    /// gate).
    ///
    /// ON-NODE GATE (n75, 2026-08-03, gemma-4-12B-it-qat-4bit, LAYER_END=24
    /// hidden-signature self-consistency since the full 48L single-node load OOMs
    /// → gemma ships PP-2): cols(=1) vs per-row(=0) at prompt lengths
    /// {64,256,512,1024} is ARGMAX-EXACT at every position (4/4 lengths + 64/64
    /// swept prefix positions, 0 mismatches), cos = 0.99999998, maxd 0.004–0.013
    /// (~1 f16-ULP on the 24L hidden). Prefill NET-POSITIVE at every length —
    /// ring-ON 1.25/1.15/1.11/1.09× (L=64/256/512/1024), ring-OFF
    /// 1.08/1.08/1.12/1.11×. Delta is HONEST-MODEST (~1.1× median, well below the
    /// qwen35 1.8–2.25×) and compresses as O(L²) attention grows: this
    /// forward_prefill_gemma path is attention/host-orchestration-bound (L=1024
    /// prefill ≈29–31s @24L), so the cols projection win only moves the ~10–20%
    /// spent in projections. Decode UNCHANGED (46.9 vs 47.9 ms/tok; flag is
    /// prefill-scoped). STACKS with the windowed KV ring (speedup present in both
    /// ring states; argmax identical across ring-ON/OFF → ring=KV-footprint and
    /// cols=projection-compute are orthogonal). Argmax-exact AND net-positive →
    /// DEFAULT-ON (like qwen35_prefill_cols); =0 reverts to serial prefill.
    pub gemma_prefill_cols: bool,
}

impl Flags {
    pub fn from_env() -> Self {
        let b1 = |k: &str| std::env::var(k).map(|v| v == "1" || v == "true").unwrap_or(false);
        let bdef1 = |k: &str| std::env::var(k).map(|v| v != "0").unwrap_or(true); // default-on
        Flags {
            quant: QuantFormat::from_env_str(
                &std::env::var("VLLM_VULKAN_QUANT").unwrap_or_default()),
            matvec_rows: std::env::var("VLLM_VULKAN_MATVEC_ROWS").ok().and_then(|v| v.parse().ok()),
            use_subgroup: b1("VLLM_VULKAN_USE_SUBGROUP"),
            unified: b1("VLLM_VULKAN_UNIFIED"),
            unified_1cb: b1("VLLM_VULKAN_UNIFIED_1CB"),
            resident: b1("VLLM_VULKAN_RESIDENT"),
            gemma_resident: b1("VLLM_VULKAN_GEMMA_RESIDENT"),
            gemma_resident_1cb: b1("VLLM_VULKAN_GEMMA_RESIDENT_1CB"), // default OFF; single-node 1-CB decode fold
            kv_ring_disable: b1("VLLM_VULKAN_KV_RING_DISABLE"), // default OFF; forces uniform max_seq KV planes (ring A/B ref)
            gpu_sdpa: b1("VLLM_VULKAN_GPU_SDPA"),
            prefill_sdpa: b1("VLLM_VULKAN_PREFILL_SDPA"),
            gemma_gpu_attn: b1("VLLM_VULKAN_GEMMA_GPU_ATTN"),
            gemma_1cb: b1("VLLM_VULKAN_GEMMA_1CB"), // default OFF; see field doc (Lever 2)
            gemma_1cb_full: b1("VLLM_VULKAN_GEMMA_1CB_FULL"), // default OFF; LEVER 1 (full fusion)
            gemma_fp8_q8: b1("VLLM_VULKAN_GEMMA_FP8_Q8"), // default OFF; LEVER 2 (fp8->q8_0 attn requant)
            gemma_mlp_q4: b1("VLLM_VULKAN_GEMMA_MLP_Q4"), // default OFF; H1 (mlp 8->4bit requant); see field doc
            gemma_lmhead_q4: b1("VLLM_VULKAN_GEMMA_LMHEAD_Q4"), // default OFF; H2 (lm_head f16->4bit requant); see field doc
            gemma_prefill_cols: bdef1("VLLM_VULKAN_GEMMA_PREFILL_COLS"), // default ON (on-node GO n75 2026-08-03: argmax-exact + net-positive prefill, stacks w/ ring); =0 reverts to serial per-row prefill. See field doc.
            qwen35_gpu: b1("VLLM_VULKAN_QWEN35_GPU"),
            q35_gpu_attn: b1("VLLM_VULKAN_Q35_GPU_ATTN"), // default OFF; see field doc
            q35_kv_f16: b1("VLLM_VULKAN_Q35_KV_F16"), // default OFF; RESERVED, not yet wired — see field doc
            qwen35_resident: b1("VLLM_VULKAN_QWEN35_RESIDENT"),
            q35_parrec: bdef1("VLLM_VULKAN_Q35_PARREC"),
            dn_gpu: bdef1("VLLM_VULKAN_DN_GPU"),
            q35_1cb: bdef1("VLLM_VULKAN_Q35_1CB"),
            q35_tp_fused: bdef1("VLLM_VULKAN_Q35_TP_FUSED"), // default ON (2026-07-25 broad-prompt A/B GO); set =0 for host oracle
            moe_gpu: bdef1("VLLM_VULKAN_MOE_GPU"),
            // NOTE: VLLM_VULKAN_MOE_OVERLAP was retired by WS2 — the shared
            // expert is recorded into the routed-expert command buffer, so the
            // GPU overlaps it with no host thread machinery. The env var is
            // ignored if set.
            moe_host_free: bdef1("VLLM_VULKAN_MOE_HOST_FREE"),
            moe_q4_resident: b1("VLLM_VULKAN_MOE_Q4_RESIDENT"),
            moe_gemm: bdef1("VLLM_VULKAN_MOE_GEMM"), // default ON (prefill-only, cos=1.0); see field doc
            moe_gemm_fused: b1("VLLM_VULKAN_MOE_GEMM_FUSED"), // default OFF; see field doc
            moe_gemm_combine: b1("VLLM_VULKAN_MOE_GEMM_COMBINE"), // default OFF; see field doc
            native_comm: std::env::var("VLLM_VULKAN_NATIVE_COMM").map(|v| v != "0").unwrap_or(true),
            reg_reduce: std::env::var("VLLM_VULKAN_REG_REDUCE").map(|v| v != "0").unwrap_or(true),
            gemma_native_hop: b1("VLLM_VULKAN_GEMMA_NATIVE_HOP"), // default OFF; fused native PP hop, see field doc
            tp_bf16_reduce: b1("VLLM_VULKAN_TP_BF16_REDUCE"),
            nvfp4_gpu: b1("VLLM_VULKAN_NVFP4_GPU"),
            nvfp4_e4m3_scales: b1("VLLM_VULKAN_NVFP4_E4M3_SCALES"), // default OFF; ≥150B footprint lever, see field doc
            dense_q4_resident: b1("VLLM_VULKAN_DENSE_Q4_RESIDENT"), // default OFF; see field doc
            attn: {
                let a = match std::env::var("VLLM_VULKAN_ATTN").unwrap_or_default().as_str() {
                    "sg" => AttnKernel::Sg, "coop" => AttnKernel::Coop, _ => AttnKernel::Scalar,
                };
                // sg-safety: the gemma 1-CB fold (GEMMA_RESIDENT_1CB) runs attention
                // IN-CB and REQUIRES the `_sg` decode kernel — with the default scalar
                // paged_attn_decode_f32 it is a ~2.2x decode REGRESSION (127 vs 48 ms/tok,
                // measured 2026-07-29). Auto-imply sg so enabling the fold can't silently
                // pathologize decode.
                if b1("VLLM_VULKAN_GEMMA_RESIDENT_1CB") && !matches!(a, AttnKernel::Sg) {
                    eprintln!("[vllm-vulkan] GEMMA_RESIDENT_1CB set: forcing VLLM_VULKAN_ATTN=sg (scalar/coop attn is a ~2.2x decode regression with the 1-CB fold)");
                    AttnKernel::Sg
                } else { a }
            },
            spin: b1("VLLM_VULKAN_SPIN"),
            profile: b1("VLLM_VULKAN_PROFILE"),
            gil_release: std::env::var("VLLM_VULKAN_GIL_RELEASE").map(|v| v != "0").unwrap_or(true),
            cb_ring: bdef1("VLLM_VULKAN_CB_RING"), // default ON; validated -4.1% decode on cluster; =0 to disable
            tp_shard_lmhead: bdef1("VLLM_VULKAN_TP_SHARD_LMHEAD"), // default ON; =0 reverts to replicated
            mlx4_w8sg_down: bdef1("VLLM_VULKAN_MLX4_W8"), // default ON; =0 reverts to v1 winners
            mlx4_repack: bdef1("VLLM_VULKAN_MLX4_REPACK"), // default ON (2026-07-26 fleet-regression GO); =0 reverts to the v1 mul_mat_vec_mlx4 oracle
            mlx4_repack_r8: bdef1("VLLM_VULKAN_MLX4_REPACK_R8"), // default ON (2026-07-30 fleet A/B GO: gemma-12B 1.066x, qwen35/27B argmax-exact); =0 restores the r4 geometry baseline
            moe_f16_scales: b1("VLLM_VULKAN_MOE_F16_SCALES"), // default OFF; 122B PP-6 fit-enabler, see field doc
            kimi_shared_resident: bdef1("VLLM_VULKAN_KIMI_SHARED_RESIDENT"), // default ON; =0 reverts to per-token host f32 shared expert
            kimi_mla_resident: bdef1("VLLM_VULKAN_KIMI_MLA_RESIDENT"), // default ON; =0 reverts to host-f32 MLA projections (kimi::mla::decode_step)
            kimi_dense_resident: bdef1("VLLM_VULKAN_KIMI_DENSE_RESIDENT"), // default ON; =0 reverts to host-f32 layer-0 dense MLP (kimi::dense_forward)
            kimi_kda_fused: bdef1("VLLM_VULKAN_KIMI_KDA_FUSED"), // default ON; =0 reverts to the 2-submit host-seam kda_step_resident (bit-exact oracle)
            nvfp4_repack: bdef1("VLLM_VULKAN_NVFP4_REPACK"), // default ON (2026-07-26 fleet-regression GO); =0 reverts to the v1 mul_mat_vec_nvfp4 oracle
            fp8_repack: b1("VLLM_VULKAN_FP8_REPACK"), // default OFF; cluster-gated (argmax-exact + perf A/B), reduction-epilogue-only twin of fp8_fast
            mlx4_rgu_repack: bdef1("VLLM_VULKAN_MLX4_RGU_REPACK"), // L1 default ON (2026-07-30 PP-5 fleet A/B GO: 24.27->20.29 ms/tok -3.98, argmax-exact; n==512 self-scopes, gemma/27B untouched); =0 reverts to the v1 kernel
            q35_geom: b1("VLLM_VULKAN_Q35_GEOM"), // default OFF; =1 routes TP q8 matvecs through matvec_variant_geom
            max_alloc_mb: std::env::var("VLLM_VULKAN_MAX_ALLOC_MB").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(0), // default OFF (see field doc)
            nemotron_gpu_mamba: b1("VLLM_VULKAN_NEMOTRON_GPU_MAMBA"), // default OFF; cluster-gated
            nemotron_resident: b1("VLLM_VULKAN_NEMOTRON_RESIDENT"), // default OFF; cluster-gated
            nemotron_1cb: b1("VLLM_VULKAN_NEMOTRON_1CB"), // default OFF; cluster-gated
            nemotron_moe_tail: b1("VLLM_VULKAN_NEMOTRON_MOE_TAIL"), // default OFF; cluster-gated
            nemotron_gpu_scan: b1("VLLM_VULKAN_NEMOTRON_GPU_SCAN"), // default OFF; cluster-gated
            nemotron_tp_ring: b1("VLLM_VULKAN_NEMOTRON_TP_RING"), // default OFF; cluster-gated
            nemotron_mamba_q8: b1("VLLM_VULKAN_NEMOTRON_MAMBA_Q8"), // default OFF; cluster-gated
            fp8_fast: bdef1("VLLM_VULKAN_FP8_FAST"), // default ON (address-gen cure: arithmetic E4M3 decode + vec4 word loads replace the const-LUT cascade + per-element word reloads; bit-exact decode, ~ULP dot4 reassoc); =0 reverts to the mul_mat_vec_fp8 LUT oracle
            nemotron_shared_q8: b1("VLLM_VULKAN_NEMOTRON_SHARED_Q8"), // default OFF; cluster-gated
            laguna_embed_f16: b1("VLLM_VULKAN_LAGUNA_EMBED_F16"), // default OFF; footprint/load-OOM lever, cluster-gated
            laguna_pread_load: b1("VLLM_VULKAN_LAGUNA_PREAD_LOAD"), // default OFF; footprint/load-OOM lever, cluster-gated
            pp_presliced_dir: std::env::var("VLLM_VULKAN_PP_PRESLICED_DIR").ok()
                .filter(|s| !s.is_empty()), // default None; per-stage pre-sliced load dir (load-transient lever)
            laguna_resident: b1("VLLM_VULKAN_LAGUNA_RESIDENT"), // default OFF; GPU-resident forward, needs a device
            laguna_1cb: b1("VLLM_VULKAN_LAGUNA_1CB"), // default OFF; resident 1-CB single-token decode fold
            laguna_yarn_gpu: b1("VLLM_VULKAN_LAGUNA_YARN_GPU"), // default OFF; GPU YaRN rope for full-attn layers (span-fold enabler)
            laguna_gpu_sdpa: b1("VLLM_VULKAN_LAGUNA_GPU_SDPA"), // default OFF; GPU decode-SDPA over resident K/V planes (span-fold final piece)
            laguna_kv_ring: bdef1("VLLM_VULKAN_LAGUNA_KV_RING"), // default ON (2026-08-02 gate PASSED bit-exact, pure footprint win); =0 reverts to full max_seq absolute planes. window-sized ring planes for sliding layers
            laguna_scratch: bdef1("VLLM_VULKAN_LAGUNA_SCRATCH"), // default ON (2026-07-30 productionized: cluster PP-6 bit-exact 1.99x, zero footprint); =0 reverts to per-op alloc/free. Silent no-op unless laguna_1cb.
            laguna_expert_repack: bdef1("VLLM_VULKAN_LAGUNA_EXPERT_REPACK"), // default ON (2026-07-30 productionized: cluster PP-6 argmax-exact, 3.6-3.75x/op); =0 reverts to the v1 mul_mat_vec_nvfp4_e4m3 oracle. Silent no-op unless e4m3-resident experts (needs NVFP4_E4M3_SCALES).
            laguna_cbbatch: b1("VLLM_VULKAN_LAGUNA_CBBATCH"), // default OFF; MoE CB-batch dispatch fold (30 expert matvecs -> 3 batched)
            laguna_gpu_attnmath: b1("VLLM_VULKAN_LAGUNA_GPU_ATTNMATH"), // default OFF; GPU qk-norm + GPU sliding-rope in attn_cached_1cb
            laguna_rust_argmax: b1("VLLM_VULKAN_LAGUNA_RUST_ARGMAX"), // default OFF; last-stage Rust argmax fusion (kills the [vocab] Vec<f32>→PyList marshal + py-argmax). Only affects the last PP rank.
            laguna_native_hop: b1("VLLM_VULKAN_LAGUNA_NATIVE_HOP"), // default OFF; fused native-vCCL PP hop (pp_step_laguna) vs the PyList marshal. Cluster PP-6 A/B pending.
            laguna_int8_attn: b1("VLLM_VULKAN_LAGUNA_INT8_ATTN"), // default OFF; q8_0-resident attn+shared weights (int8 sibling of f8-attn; cheap dequant, accuracy+perf-gated)
            laguna_hostfold: b1("VLLM_VULKAN_LAGUNA_HOSTFOLD"), // default OFF; host-residual copy-elision on the 1-CB decode path (composes on laguna_scratch)
            nemotron_mtp: b1("VLLM_VULKAN_NEMOTRON_MTP"), // default OFF; Phase-1 acceptance sim only
            nemotron_mtp_trace: std::env::var("VLLM_VULKAN_NEMOTRON_MTP_TRACE")
                .ok().filter(|s| !s.is_empty()), // default None; Phase-1 DECOUPLED trace-dump
            gemm_f16aligned: bdef1("VLLM_VULKAN_GEMM_F16ALIGNED"), // default ON; =0 reverts to live f16f32
            gemm_quant: b1("VLLM_VULKAN_GEMM_QUANT"), // default OFF; see field doc
            qwen35_prefill_cols: bdef1("VLLM_VULKAN_QWEN35_PREFILL_COLS"), // default ON (on-node GO n55 2026-08-02); =0 reverts to serial prefill
        }
    }
}

/// Global snapshot the free-fn accessors delegate to, so there is exactly one
/// env read per flag for the whole process (matches `self.flags` on
/// `VulkanModel`, which is populated from the same `Flags::from_env()` call
/// site pattern at each construction).
pub fn flags_global() -> &'static Flags {
    static F: std::sync::OnceLock<Flags> = std::sync::OnceLock::new();
    F.get_or_init(Flags::from_env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_format_from_env_str() {
        assert_eq!(QuantFormat::from_env_str("q8_0"), QuantFormat::Q8_0);
        assert_eq!(QuantFormat::from_env_str("q4_0"), QuantFormat::Q4_0);
        assert_eq!(QuantFormat::from_env_str("q4_k"), QuantFormat::Q4_K);
        assert_eq!(QuantFormat::from_env_str("bf16"), QuantFormat::Bf16);
        assert_eq!(QuantFormat::from_env_str(""), QuantFormat::F16);
        assert_eq!(QuantFormat::from_env_str("garbage"), QuantFormat::F16);
    }
}
