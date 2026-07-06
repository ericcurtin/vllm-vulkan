# SPDX-License-Identifier: Apache-2.0
"""Vulkan-accelerated tensor operations — batched dispatch path.

The primary performance improvement over the naive implementation:
  OLD: one vkQueueSubmit per op → ~150µs driver overhead each
  NEW: batch all ops for a full transformer layer into one vkQueueSubmit

Key API:
  - rms_norm(x, weight, eps) — uses execute_batch internally
  - linear(x, weight, bias)  — uses execute_batch internally
  - set_context / get_context

For the highest-throughput path, callers can build op lists and submit
them via ctx.execute_batch() directly.
"""

from __future__ import annotations

import logging
import struct
import weakref
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm_vulkan._rs import VulkanContext

logger = logging.getLogger(__name__)

# ─── Global context ──────────────────────────────────────────────────────────

_ctx: VulkanContext | None = None
# Cache for `available_shaders()` — see `_cached_available_shaders`'s doc
# comment. Reset to None (invalidated, recomputed lazily on first use)
# whenever `set_context` installs a (possibly different) context.
_available_shaders_cache: frozenset[str] | None = None


def set_context(ctx: VulkanContext) -> None:
    global _ctx, _available_shaders_cache, _weight_f16_decision_cache
    _ctx = ctx
    _available_shaders_cache = None
    # `_weight_f16_decision_cache` (see `_weight_uses_f16`'s doc comment)
    # caches a per-weight decision that depends on the *context* too (via
    # `_cached_available_shaders(ctx)`), not just the weight's own values
    # — reset it here for exactly the same reason `_available_shaders_cache`
    # is reset above: without this, a weight tensor object queried against
    # two different `VulkanContext` instances with different compiled
    # shader sets (e.g. sequential tests, or a real caller that swaps
    # devices) could silently keep reading a decision computed for the
    # *previous* context instead of the current one.
    _weight_f16_decision_cache = _WeightCache()

    # Detect software renderer and disable GPU dispatch.
    try:
        from vllm_vulkan._rs import enumerate_devices  # noqa: PLC0415

        devs = enumerate_devices()
        if devs:
            name = devs[0].get("name", "").lower()
            dtype = devs[0].get("device_type", "")
            if dtype == "cpu" or any(
                s in name for s in ("llvmpipe", "lavapipe", "swiftshader", "software")
            ):
                logger.info(
                    "Vulkan device '%s' is a software renderer; GPU dispatch disabled.",
                    devs[0].get("name", ""),
                )
                _ctx = None
    except Exception:
        pass


def _cached_available_shaders(ctx: VulkanContext) -> frozenset[str]:
    """Returns `ctx.available_shaders()`, computed once and cached instead
    of re-queried (and re-marshalled from Rust into a fresh Python list on
    every call) on every single invocation.

    `available_shaders()` is a pure function of the context — the set of
    compiled shaders is fixed for the whole lifetime of a `VulkanContext`
    (determined once, at pipeline-cache construction time) — but
    `linear()` called it on *every* decode-shaped call (checking whether
    `"mul_mat_vec_f32_f32_f32"` is in the returned list, then discarding
    the whole list), and that's ~5-6 Linear-family modules x 35 Gemma4-E2B
    decoder layers = ~175-210 calls/decode-step. Measured directly on
    this hardware: `ctx.available_shaders()` (a `Vec<String>` rebuilt and
    marshalled through PyO3 into a fresh Python list on every call) costs
    ~1.1us/call — modest per call, but purely repeated, avoidable work
    across every one of those calls, every decode step.

    Cached as a `frozenset` (not the raw list) so every `in` membership
    check this module makes (`"shader_name" in ...`) is O(1) instead of
    O(shader_count) — a second, smaller improvement layered on top of
    just avoiding the repeated PyO3 round-trip.

    Invalidated (see `set_context`) whenever a new context is installed,
    so tests that construct multiple `VulkanContext` instances (e.g. the
    `vulkan_ctx` fixture in tests/python/test_vulkan_ops.py) never read a
    stale cache from a previous context.
    """
    global _available_shaders_cache
    if _available_shaders_cache is None:
        _available_shaders_cache = frozenset(ctx.available_shaders())
    return _available_shaders_cache


def get_context() -> VulkanContext:
    if _ctx is None:
        raise RuntimeError("VulkanContext not initialised.")
    return _ctx


def is_ready() -> bool:
    return _ctx is not None


# ─── Weight cache (persistent GPU buffers + CPU float32 shadow copies) ───────


class _WeightCache:
    """Weak-ref keyed cache: storage object id → arbitrary cached value.

    Used both for persistent GPU-resident weight buffers (_weight_cache)
    and for CPU-side float32 shadow copies of bf16/fp16 weights
    (_cpu_float32_cache) - same assumption either way: the same nn.Module's
    persistent .weight Parameter object (same storage) is passed in on
    every forward call, so a conversion that only depends on the weight
    itself only needs to be redone once, not on every single call.

    put() registers a cleanup callback on the storage's weakref (rather
    than just storing (weakref, value) and relying on a future get() call
    to notice the weakref is dead and evict it) - without that callback,
    a weight whose storage is garbage collected but whose id() key is
    never looked up again (e.g. the model is replaced/reloaded, or the
    module is discarded) would leave its entry - and the large cached
    value it holds (a GPU-resident buffer or a full float32 CPU copy of
    the weight matrix) - in self._data forever, since a plain dict never
    removes entries on its own. The callback closes over a *weak*
    reference to self (not self directly) so the cache instance itself
    isn't kept alive by its own cleanup callbacks.

    The weakref object returned by weakref.ref(storage, _cleanup) is
    stored as part of the cached entry (not discarded) - a weakref's
    callback only fires while the weakref object itself is still alive;
    if nothing keeps a reference to it, CPython is free to collect the
    weakref object itself before `storage` is ever collected, silently
    disabling the callback (verified empirically while writing this).
    """

    def __init__(self) -> None:
        self._data: dict[int, tuple] = {}  # key → (weakref, cached value)

    def get(self, w: torch.Tensor) -> object | None:
        entry = self._data.get(id(w.untyped_storage()))
        return entry[1] if entry is not None else None

    def put(self, w: torch.Tensor, value: object) -> None:
        storage = w.untyped_storage()
        key = id(storage)
        self_ref = weakref.ref(self)

        def _cleanup(_storage_ref: object) -> None:
            cache = self_ref()
            if cache is not None:
                cache._data.pop(key, None)

        self._data[key] = (weakref.ref(storage, _cleanup), value)

    def __len__(self) -> int:
        return len(self._data)


_weight_cache = _WeightCache()
_cpu_float32_cache = _WeightCache()
# Per-weight decision cache for `_weight_uses_f16` — see its doc comment.
_weight_f16_decision_cache = _WeightCache()


def _weight_safe_for_f16(weight: torch.Tensor) -> bool:
    """True if every element of `weight` round-trips through float16
    without overflowing to +/-inf.

    bf16 (the dtype real checkpoints load projection weights as — see
    module docstring) has the same 8-bit exponent as float32, so its
    representable range is ~3.4e38; float16 has only a 5-bit exponent,
    range ~6.5e4. A value that's perfectly finite in bf16/float32 can
    therefore silently become +/-inf in float16 — not an error, just a
    wrong number quietly propagating through every downstream matvec/
    matmul that reads it, exactly the kind of silent-wrong-output failure
    `mul_mat_vec_f32_f32_f32_subgroup`'s postmortem (src/lib.rs's
    `subgroup_matvec_correctness_tests`) established this codebase must
    check for directly rather than assume away. Real transformer weights
    are essentially always small (normalized to roughly unit scale by
    training), so this is expected to return True in practice — but
    "expected" is exactly the assumption that shader's bug hid behind.
    """
    return bool(torch.isfinite(weight.half()).all())


def _weight_uses_f16(ctx: VulkanContext, weight: torch.Tensor) -> bool:
    """Whether `weight`'s persistent GPU buffer should be (and, once
    `_get_or_upload_weight` has run for it, was) uploaded as float16
    instead of float32.

    A pure, cached function of `weight`'s identity + values (same
    weak-ref-keyed convention as `_weight_cache` itself — see
    `_WeightCache`'s docstring): both `_vulkan_matvec` and `_vulkan_matmul`
    need this exact same decision twice (once to pick which shader
    variant to dispatch, once inside `_get_or_upload_weight` to pick which
    dtype to upload), and `_weight_safe_for_f16` costs a full weight-sized
    `.half()` conversion + `isfinite` scan — real, avoidable, repeated
    work if redone on every single decode-step/prefill-chunk call instead
    of once per distinct weight tensor, same rationale as `_weight_cache`
    avoiding a re-upload on every call.

    Gated on both f16-weight shader variants
    (`mul_mat_vec_f16_f32_f32`/`matmul_f16_f32_fp32`) actually being
    compiled — unconditionally true for this backend's fixed shader set
    today (see scripts/compile_shaders.sh), but checked the same
    defensive way every other shader dispatch in this file gates on
    `_cached_available_shaders`, rather than assuming a future build
    always has them.
    """
    cached = _weight_f16_decision_cache.get(weight)
    if cached is not None:
        return cast("bool", cached)
    available_shaders = _cached_available_shaders(ctx)
    decision = (
        "mul_mat_vec_f16_f32_f32" in available_shaders
        and "matmul_f16_f32_fp32" in available_shaders
        and _weight_safe_for_f16(weight)
    )
    _weight_f16_decision_cache.put(weight, decision)
    return decision


def _get_or_upload_weight(
    ctx: VulkanContext, weight: torch.Tensor, prefer_f16: bool = False
) -> object | None:
    """Return a persistent GpuTensor for weight, uploading once on first use.

    `prefer_f16`: if True AND `_weight_uses_f16` agrees (safe range, f16
    shaders available), upload/cache a float16 buffer — HALF the bytes
    of the usual float32 upload, and (more importantly, since this same
    buffer is re-read from GPU memory on every matvec/matmul dispatch
    against it, not just once here) half the bytes every dispatch has to
    move for this weight. This mirrors the standalone Rust `VulkanModel`
    path's own weight upload (src/lib.rs's `VulkanModel::new`, "Projection
    weights are uploaded as f16 to halve memory bandwidth") — this
    (separate, `VulkanContext`-based) code path never had that applied
    even though the matching f16 shaders were compiled and available for
    it too, per this project's established "compiled but never
    dispatched" pattern (see e.g. #82's `matmul_f32_f32`/#77's fused
    decode dispatch).

    Only `_vulkan_matvec`/`_vulkan_matmul` pass `prefer_f16=True` — every
    other caller (RMSNorm's `norm_weight` via `rms_norm_then_linear`,
    which has no matching f16-weight shader wired up here) keeps this at
    its default and gets the exact same float32-always behaviour as
    before this change.

    Callers needing to know whether f16 was actually used (to pick a
    matching shader variant) must call `_weight_uses_f16` themselves —
    the same cached decision this function uses internally, so the two
    can never disagree for a given weight.
    """
    try:
        cached = _weight_cache.get(weight)
        if cached is not None:
            return cached
        if prefer_f16 and _weight_uses_f16(ctx, weight):
            gpu = ctx.upload_tensor(_to_bytes(weight.half()))
        else:
            w_f32 = weight.float() if weight.dtype != torch.float32 else weight
            gpu = ctx.upload_tensor(_to_bytes(w_f32))
        _weight_cache.put(weight, gpu)
        return gpu
    except Exception as exc:
        logger.debug("Failed to upload weight: %s", exc)
        return None


def _get_or_convert_to_float32_cpu(weight: torch.Tensor) -> torch.Tensor:
    """Return a CPU float32 copy of weight, converting once on first use.

    Used by linear()'s CPU (prefill) fallback path, which - unlike the GPU
    matvec decode path's _get_or_upload_weight - has no cache at all before
    this: weight.float() allocated a brand-new float32 copy of the entire
    weight matrix on every single prefill call, for every wrapped Linear
    module, even though the weight (and hence its correct float32 copy)
    never changes between calls. Measured on this hardware, for a
    realistic Gemma4-E2B FFN weight shape (k=1536, n=6144, bf16): the
    conversion alone costs ~1.26ms out of ~8.8ms for the whole prefill
    linear() call (~14%) - real, repeated, entirely avoidable work.
    """
    if weight.dtype == torch.float32:
        return weight
    cached = _cpu_float32_cache.get(weight)
    if cached is not None:
        return cast("torch.Tensor", cached)
    w_f32 = weight.float()
    _cpu_float32_cache.put(weight, w_f32)
    return w_f32


# ─── Tensor ↔ bytes helpers ──────────────────────────────────────────────────


def _to_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.numpy().tobytes()


def _from_bytes(
    data: bytes | bytearray, shape: tuple, dtype: torch.dtype
) -> torch.Tensor:
    """Reconstructs a tensor from `ctx.execute_batch`/`execute_chained`'s
    raw output.

    Those Rust methods return a `bytearray` (not `bytes`) specifically so
    `np.frombuffer` here produces a *writable* array — skipping the
    `.copy()` a read-only (`bytes`-backed) buffer would otherwise force
    on every single call before `torch.from_numpy` can use it. Measured
    directly against real GPU-produced output (not a synthetic buffer,
    which understates this): ~31.9us/call with the copy vs ~27.2us/call
    without, on this hardware — a real, if modest next to the ~400-900us
    GPU dispatch this follows, saving that adds up across the ~150+ GPU-
    dispatched Linear calls per decode step (see kv_ops.py's
    paged_kv_write_and_decode_batch_f32 docstring for that count). If
    ever called with a genuine (read-only) `bytes` object instead, this
    still works correctly — just re-pays the writability warning/copy
    PyTorch itself falls back to internally in that case.
    """
    if dtype == torch.bfloat16:
        arr = np.frombuffer(data, dtype=np.int16).reshape(shape)
        return torch.from_numpy(arr).view(torch.bfloat16)
    np_dtype = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }.get(dtype, np.float32)
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    return torch.from_numpy(arr)


# ─── Push-constant builders ──────────────────────────────────────────────────


@lru_cache(maxsize=256)
def _rms_norm_pc(nrows: int, ncols: int, eps: float) -> bytes:
    """Pack push constants for rms_norm_f32 / rms_norm_f32_mul.

    `@lru_cache`d: this is a pure function of (nrows, ncols, eps), and the
    decode path (by far the hottest caller — see `rms_norm`, called once
    per RMSNorm module, per decoder layer, per generated token) always
    passes the exact same arguments for a given module (nrows=1 for
    single-token decode; ncols/eps are fixed properties of that module's
    weight/config, never varying between calls). Before this, every one
    of those repeated calls re-ran 4 `struct.pack` calls and several
    `bytes` concatenations from scratch — measured at ~0.7us/call on this
    hardware; with ~5 RMSNorm modules x 35 Gemma4-E2B decoder layers
    (~175 calls) invoked once per decode step, that's ~125us/decode-step
    of pure, avoidable repacking, now paid once per distinct
    (nrows, ncols, eps) triple instead of once per call. `maxsize=256` is
    far more than the handful of distinct shapes any real model actually
    uses (bounded by the model's own hidden_size/per-layer-input sizes),
    so no meaningful eviction thrashing is expected even across prefill
    calls with varying `nrows` (prompt length).
    """
    nb00, nb01, nb02 = 1, ncols, nrows * ncols
    ne10, nb10, nb11 = ncols, 1, ncols
    nb20, nb21 = 1, ncols
    pc = struct.pack("9I", nrows * ncols, ncols, nrows, 1, 1, nb00, nb01, nb02, 1)
    pc += struct.pack("8I", ne10, 1, 1, 1, nb10, nb11, 1, 1)
    pc += struct.pack("8I", ncols, nrows, 1, 1, nb20, nb21, 1, 1)
    pc += struct.pack("I f f i", 0, eps, 0.0, 0)
    return pc


@lru_cache(maxsize=256)
def _matvec_pc(T: int, K: int, N: int) -> bytes:  # noqa: N803
    """Pack push constants for mul_mat_vec_f32_f32_f32.

    `@lru_cache`d for the same reason as `_rms_norm_pc` above: a pure
    function of (T, K, N), called once per Linear-family module per
    decoder layer per token on the decode fast path (`_vulkan_matvec`) —
    for Gemma4-E2B, T is always 1 for single-token decode, and K/N are
    fixed per module (in/out feature counts never change), so this is
    the exact same "recomputed every call, but always the same result
    for a given caller" waste, just for the matvec push-constant layout
    instead of RMSNorm's.
    """
    return struct.pack(
        "13I",
        K,
        K,
        K,
        N,  # ncols, stride_a, stride_b, stride_d
        K * N,
        K,
        N,  # batch_stride_a, batch_stride_b, batch_stride_d
        0,  # fusion_flags
        0,  # base_work_group_y
        1,
        T,
        T,
        1,  # ne02, ne12, broadcast2, broadcast3
    )


@lru_cache(maxsize=256)
def _matmul_pc(m: int, t: int, k: int) -> bytes:  # noqa: N803
    """Pack push constants for the tiled-matmul shaders (`matmul_f32_f32`
    et al, from shaders/mul_mm.comp) — 16 uint32 fields, exactly matching
    that shader's own `push_constant` block (mul_mm.comp:74-101), NOT
    llama.cpp upstream's newer 17-field struct (see the `padded_N` note
    at this shader's registration site in src/lib.rs). Verified against
    a direct CPU reference before use — see `tiled_matmul_dispatch_tests`
    in src/lib.rs.

    For a single, unbatched Linear call computing
    `out[T, out_features] = x[T, in_features] @ weight[out_features,
    in_features]^T`: `m`=out_features (weight's row count), `t`=T
    (activation row count), `k`=in_features (shared dimension). Both
    `weight` and `x` must be row-major-contiguous with row stride
    exactly `k` — true for any freshly-`.contiguous()`'d/uploaded
    tensor, which `_to_bytes`/`_get_or_upload_weight` already guarantee.

    `@lru_cache`d for the same reason as `_matvec_pc` above: a pure
    function of (m, t, k), and while T varies per prefill call (unlike
    decode's always-T=1), the number of *distinct* (m, t, k) triples in
    any real chunked-prefill workload is still small and bounded (a
    handful of Linear-module shapes x a handful of chunk sizes), so this
    still avoids repeating the same `struct.pack` work across identical
    calls.
    """
    return struct.pack(
        "16I",
        m,
        t,
        k,  # M, N, K
        k,
        k,
        m,  # stride_a, stride_b, stride_d
        m * k,
        k * t,
        m * t,  # batch_stride_a, batch_stride_b, batch_stride_d
        0,  # base_work_group_z
        1,  # num_batches
        k,  # k_split = K (skip the multi-pass K-split reduce path — see
        # `compile_matmul`'s doc comment in src/pipeline.rs for why this
        # is always safe for realistic transformer K)
        1,
        1,
        1,
        1,  # ne02, ne12, broadcast2, broadcast3
    )


def _matmul_workgroups(m: int, t: int) -> tuple[int, int, int]:
    """Workgroup dispatch count for the tiled-matmul shaders:
    `(ceil(M/BM), ceil(N/BN), num_batches)`, using the BM=BN=64 these
    shaders are actually compiled with (`compile_matmul` in
    src/pipeline.rs — NOT llama.cpp's own named "l"/"m"/"s" tiling
    presets, a different, unused-here parameter set).
    """
    bm = bn = 64
    return ((m + bm - 1) // bm, (t + bn - 1) // bn, 1)


# ─── RMS Norm ────────────────────────────────────────────────────────────────

_MATVEC_THRESHOLD = 4  # T < this → matvec shader; T >= this → tiled matmul/CPU
# Weight matrices with fewer than this many elements (in_features *
# out_features) always use the CPU path, regardless of T — see
# `linear`'s doc comment for the measurement behind this cutoff.
_MATVEC_MIN_WEIGHT_ELEMENTS = 1_000_000

# Do NOT "fix" prefill performance by simply raising _MATVEC_THRESHOLD so
# larger T also dispatches to mul_mat_vec_f32_f32_f32 — measured directly
# on this hardware, at Gemma4-E2B's real q_proj shape (in_features=1536,
# out_features=2048), this makes prefill dramatically SLOWER, not faster:
#
#   T      CPU (torch.nn.functional.linear)   GPU (mul_mat_vec_f32_f32_f32)
#   1        873.2us  (873.21us/token)           388.5us  (388.48us/token)
#   4        815.4us  (203.86us/token)            980.1us  (245.01us/token)
#  16        988.0us   (61.75us/token)           3387.8us  (211.74us/token)
#  64       2304.4us   (36.01us/token)          12767.6us  (199.49us/token)
# 128       3917.5us   (30.61us/token)          25378.4us  (198.27us/token)
# 256       7370.5us   (28.79us/token)          50219.2us  (196.17us/token)
# 512      14736.1us   (28.78us/token)          99819.7us  (194.96us/token)
#
# GPU is already ~6.5x slower than CPU by T=128, and the gap keeps
# widening. Root cause: mul_mat_vec's per-token dispatch (workgroups =
# (N, T, 1), see `_vulkan_matvec`) is a genuine batched matrix-VECTOR
# multiply — each of the T "batch" workgroup-rows independently re-reads
# the *entire* weight matrix from GPU memory (see mul_mat_vec_base.glsl's
# `get_offsets()`: every batch_idx in [0, T) resolves to the same
# `a_offset` since the shader's ne02/ne12/broadcast2/broadcast3 push
# constants are set up for "same weight, T independent inputs"
# broadcasting, not weight-tile reuse across inputs). That's fine at
# T<4 (decode) where the fixed per-dispatch driver overhead dominates
# and re-reading a ~3-12MB weight matrix a handful of times is cheap
# relative to it, but at prefill-scale T it means GPU bandwidth cost
# scales as O(T x in_features x out_features) — the same order as CPU's
# cost, with none of a real tiled matmul's weight-tile-reuse advantage,
# so GPU just pays that cost at a *worse* effective rate than CPU's own
# BLAS routines here. `linear()` therefore never raises this threshold
# for matvec — see `test_matvec_batching_does_not_help_at_prefill_scale_t`
# (tests/python/test_vulkan_ops.py) for the regression guard.
#
# The REAL prefill speedup this analysis pointed to — the tiled matmul
# shaders (`matmul_f32_f16`/`matmul_f32_f16_aligned`/`matmul_f32_f32`/
# `matmul_f32_f32_fp32`/`matmul_f16_f32_fp32`, from shaders/mul_mm.comp)
# — is now wired up (`_vulkan_matmul` below), after two real Vulkan
# pipeline-compilation bugs blocking it were found and fixed (see
# `pipeline::PipelineCache::compile_matmul`'s doc comment in
# src/pipeline.rs: an infinite-GPU-loop hang, then a silent half-tile-
# of-zeros correctness bug — both invisible until this shader was
# actually dispatched and checked against a direct CPU reference, not
# just read). Unlike matvec, the tiled matmul shader genuinely reuses
# weight tiles across the T dimension the way a real GEMM should, so it
# *wins* at prefill scale, and the margin widens with T instead of
# narrowing — measured directly at the same q_proj shape as the table
# above (`matmul_f32_f32`, best-of-N trials):
#
#   T      CPU (torch.nn.functional.linear)   GPU (matmul_f32_f32)
#   4        790.6us  (197.65us/token)           475.4us  (118.85us/token)
#  16       1137.5us   (71.09us/token)           466.7us   (29.17us/token)
#  64       3499.6us   (54.68us/token)          1090.8us   (17.04us/token)
# 128       6540.5us   (51.10us/token)          1396.6us   (10.91us/token)
# 512      24631.8us   (48.11us/token)          2088.9us    (4.08us/token)
#
# GPU is faster at every T tested (1.66x-11.79x here; 1.4x-17.3x across
# all 3 real large-weight Linear shapes measured — see
# `tiled_matmul_beats_cpu_at_prefill_scale_for_large_weights` in
# src/lib.rs). For the *small*-weight case (k_proj/v_proj,
# `< _MATVEC_MIN_WEIGHT_ELEMENTS`, same gate `_vulkan_matmul` uses), GPU
# only pulls ahead once T is much larger than typical (measured
# crossover between T=64 and T=128 — see
# `tiled_matmul_small_weight_crossover_point`), so `linear()`
# deliberately keeps that shape on the CPU-only path at every T, exactly
# mirroring the decode matvec threshold's existing size gate rather than
# adding a second, narrower one.


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """RMS normalisation — always runs on CPU.

    Unlike `linear()` (whose decode path genuinely benefits from GPU
    dispatch — see `_MATVEC_THRESHOLD`), RMSNorm has no matmul-sized
    compute to amortize a Vulkan submission's fixed driver overhead
    against: it's a lightweight elementwise reduction (sum of squares,
    rsqrt, multiply) over just `ncols` (1536 or 256 for Gemma4-E2B)
    elements per row. Measured directly on this hardware, dispatching
    `rms_norm_f32`/`rms_norm_f32_mul` via `ctx.execute_batch` is
    *consistently slower* than `_cpu_rms_norm` (plain PyTorch ops) at
    every batch size tested, from single-token decode (nrows=1: ~166-220us
    GPU vs ~24-30us CPU, a 5.6-7.3x CPU win) all the way through full-length
    prefill and beyond (nrows=32768: ~78.6ms GPU vs ~31.9ms CPU, still a
    2.5x CPU win) — the ratio never favors GPU, unlike `linear()` where the
    crossover genuinely depends on T. Given `rms_norm` is called once per
    RMSNorm module, per decoder layer, per token (~175 calls/decode-step
    for Gemma4-E2B's 35 layers), this was a substantial, previously
    unaddressed cost: switching every one of those calls from GPU dispatch
    to direct CPU computation saves on the order of 150-190us *per call*
    at the (by far most common) decode shape alone.

    See `TestRmsNormAlwaysUsesCpu` (tests/python/test_vulkan_ops.py) for
    the measurement this doc comment summarizes, and this function's git
    history for the previous GPU-dispatching implementation should a
    future hardware/driver combination ever change this trade-off.
    """
    return _cpu_rms_norm(x, weight, eps)


def _cpu_rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out


# ─── Linear (matmul) ─────────────────────────────────────────────────────────


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear layer: decode uses Vulkan matvec, prefill uses Vulkan tiled
    matmul — both only for large-enough weights; small weights always use
    CPU.

    `_MATVEC_THRESHOLD` alone (T < 4 → decode GPU path) isn't the whole
    story: the right dispatch decision also depends on the weight
    matrix's total size (`in_features * out_features`), not just `T`.
    Measured directly on this hardware at Gemma4-E2B's real Linear-module
    shapes: for `k_proj`/`v_proj` (small, since `num_key_value_heads=1`
    gives an `out_features` of only 256/512) the CPU path is faster at
    *every* T tested — including the all-important T=1 decode case,
    where GPU dispatch measured ~1.1-2.0x *slower* than CPU there (e.g.
    ~164-209us GPU vs ~81-187us CPU) — while for `q_proj`/`o_proj`/
    `gate_proj`/`up_proj`/`down_proj` (all >=3.15M elements), GPU
    dispatch is a clear win at T=1 (0.24x-0.45x of CPU's time) exactly as
    `_MATVEC_THRESHOLD` already assumed. `_MATVEC_MIN_WEIGHT_ELEMENTS`
    (1,000,000) sits between the largest measured small-weight case
    (512*1536 = 0.79M, CPU-favoring even at T=1) and the smallest
    measured large-weight case (2048*1536 = 3.15M, GPU-favoring at
    T=1-2) — comfortably clear of either boundary rather than sitting
    right at a fragile crossover point. The same size gate is reused for
    the T>=_MATVEC_THRESHOLD (prefill) tiled-matmul path below — see
    `_MATVEC_THRESHOLD`'s doc comment for the prefill-scale measurements
    justifying that reuse rather than a second, separately-tuned
    threshold.

    See `TestLinearMatvecDispatchThreshold` (tests/python/test_vulkan_ops.py)
    for the measurements this doc comment summarizes.
    """
    ctx = get_context()

    orig_shape = x.shape
    in_feat = weight.shape[1]
    out_feat = weight.shape[0]
    x_2d = x.float().reshape(-1, in_feat)
    T = x_2d.shape[0]  # noqa: N806

    weight_elements = in_feat * out_feat
    available_shaders = _cached_available_shaders(ctx)
    if (
        weight_elements >= _MATVEC_MIN_WEIGHT_ELEMENTS
        and T < _MATVEC_THRESHOLD
        and "mul_mat_vec_f32_f32_f32" in available_shaders
    ):
        result = _vulkan_matvec(ctx, x_2d, weight)
    elif (
        weight_elements >= _MATVEC_MIN_WEIGHT_ELEMENTS
        and T >= _MATVEC_THRESHOLD
        and "matmul_f32_f32" in available_shaders
    ):
        result = _vulkan_matmul(ctx, x_2d, weight)
    else:
        result = torch.nn.functional.linear(
            x_2d, _get_or_convert_to_float32_cpu(weight)
        )

    if bias is not None:
        result = result + bias.float()

    return result.reshape(*orig_shape[:-1], out_feat)


def _vulkan_matvec(
    ctx: VulkanContext,
    x: torch.Tensor,  # [T, K] float32
    weight: torch.Tensor,  # [N, K] any dtype
) -> torch.Tensor:
    """Dispatch mul_mat_vec_{f16,f32}_f32_f32 for decode (T<4).

    NOT mul_mat_vec_f32_f32_f32_subgroup: that variant was found to
    silently diverge from the correct result for ncols>=~256 (i.e.
    essentially every real hidden/intermediate size) -- see
    subgroup_matvec_correctness_tests in src/lib.rs for the measured
    divergence and root-cause discussion. mul_mat_vec_f32_f32_f32 (the
    plain, non-subgroup variant) has an identical binding/push-constant/
    workgroup-dispatch convention and was confirmed correct at every
    size tested.

    Uploads (and dispatches against) `weight` as float16 instead of
    float32 whenever `_weight_uses_f16` says it's safe to — halving the
    bytes this (large, per-call, re-read-from-GPU-memory) weight buffer
    costs to move. `mul_mat_vec_f16_f32_f32` shares `mul_mat_vec_base.glsl`
    with the f32 variant and only differs in `A_TYPE`
    (scripts/compile_shaders.sh), converting each element to float on
    load before the identical dot-product math either variant runs — see
    `f16_weight_dispatch_tests::mul_mat_vec_f16_f32_f32_matches_cpu_reference_at_realistic_k`
    (src/lib.rs) for the direct correctness verification against a CPU
    reference computed from the same f16-rounded weight values.
    """
    # Cache key on original weight (before .float()/.half()) - and that
    # conversion itself only happens lazily, in the `w_gpu is None`
    # fallback branch below, since on the (overwhelmingly common) cache-hit
    # path the GPU-resident buffer already holds the converted weight and
    # recomputing it here would just be a wasted full weight-matrix copy,
    # every single call, for a value nothing else uses.
    use_f16 = _weight_uses_f16(ctx, weight)
    w_gpu = _get_or_upload_weight(ctx, weight, prefer_f16=True)

    T, K = x.shape  # noqa: N806
    N = weight.shape[0]  # noqa: N806
    pc = _matvec_pc(T, K, N)
    x_bytes = _to_bytes(x)
    out_size = T * N * 4

    if w_gpu is not None:
        w_binding = w_gpu
    else:
        w_binding = _to_bytes(weight.half()) if use_f16 else _to_bytes(weight.float())

    shader = "mul_mat_vec_f16_f32_f32" if use_f16 else "mul_mat_vec_f32_f32_f32"
    results = ctx.execute_batch(
        [
            (
                shader,
                [w_binding, x_bytes],
                [out_size],
                pc,
                (N, T, 1),
                False,
            ),
        ]
    )
    return _from_bytes(results[0][0], (T, N), torch.float32)


def _vulkan_matmul(
    ctx: VulkanContext,
    x: torch.Tensor,  # [T, K] float32
    weight: torch.Tensor,  # [M, K] any dtype
) -> torch.Tensor:
    """Dispatch `matmul_{f16,f32}_f32` (the tiled matmul shader,
    mul_mm.comp) for prefill (T >= _MATVEC_THRESHOLD, large-enough
    weight — see `linear`'s dispatch decision).

    Only the plain (non-`_aligned`) variants are dispatched: the
    `_aligned` variants assume `K` is already a multiple of the tile
    size and skip bounds-checking loads, silently producing wrong
    results otherwise. `matmul_f32_f32`/`matmul_f16_f32_fp32` handle any
    `M`/`T`/`K` shape safely — verified directly, including shapes
    deliberately NOT aligned to this shader's BM=BN=64/BK=32 tile sizes,
    against a CPU reference (see `tiled_matmul_dispatch_tests` and
    `f16_weight_dispatch_tests` in src/lib.rs) — so no alignment check is
    needed here.

    Shares `_get_or_upload_weight`'s GPU weight cache with
    `_vulkan_matvec`: both shaders require the exact same buffer layout
    (row-major `[out_features, in_features]`, same dtype as
    `_weight_uses_f16` decided for this weight), so the same persistent
    upload serves either dispatch path for a given weight, whichever T a
    given call happens to use — `_weight_uses_f16`'s own cache guarantees
    both call sites agree on that dtype for the same weight tensor.
    """
    use_f16 = _weight_uses_f16(ctx, weight)
    w_gpu = _get_or_upload_weight(ctx, weight, prefer_f16=True)

    T, K = x.shape  # noqa: N806
    M = weight.shape[0]  # noqa: N806
    pc = _matmul_pc(M, T, K)
    x_bytes = _to_bytes(x)
    out_size = T * M * 4

    if w_gpu is not None:
        w_binding = w_gpu
    else:
        w_binding = _to_bytes(weight.half()) if use_f16 else _to_bytes(weight.float())

    shader = "matmul_f16_f32_fp32" if use_f16 else "matmul_f32_f32"
    results = ctx.execute_batch(
        [
            (
                shader,
                [w_binding, x_bytes],
                [out_size],
                pc,
                _matmul_workgroups(M, T),
                False,
            ),
        ]
    )
    return _from_bytes(results[0][0], (T, M), torch.float32)


# ─── Fused RMSNorm + Linear batch dispatch ───────────────────────────────────


def rms_norm_then_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    linear_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm → Linear in ONE vkQueueSubmit using execute_chained.

    2 ops, 1 fence wait. The intermediate (normed) activation stays
    on the GPU between ops — no CPU readback of the norm output.

    Old: 2 × (submit+wait) = ~300µs overhead
    New: 1 × (submit+wait) = ~150µs overhead
    """
    ctx = get_context()

    orig_shape = x.shape
    ncols = orig_shape[-1]
    x_flat = x.float().reshape(-1, ncols)
    nrows = x_flat.shape[0]
    T = nrows  # noqa: N806

    in_feat = linear_weight.shape[1]
    out_feat = linear_weight.shape[0]

    # Fused single-submit path only for decode (T < threshold); prefill
    # falls back to separate rms_norm() + linear() calls — rms_norm()
    # always runs on CPU (see its own doc comment) and linear() now
    # dispatches the GPU tiled matmul for large-enough weights at
    # prefill scale too (see `_vulkan_matmul`), so this fallback is not
    # a full CPU path, just an un-fused one (2 Python-level calls instead
    # of 1 chained GPU submit — fusing RMSNorm into the same submit as
    # `_vulkan_matmul` is a possible future improvement, not attempted
    # here: rms_norm() itself is CPU-only regardless, so there is no GPU
    # intermediate buffer to keep on-device between the two steps the
    # way there is for the decode matvec case below).
    available_shaders = _cached_available_shaders(ctx)
    if (
        T >= _MATVEC_THRESHOLD
        or "rms_norm_f32_mul" not in available_shaders
        or "mul_mat_vec_f32_f32_f32" not in available_shaders
    ):
        normed = rms_norm(x, norm_weight, eps)
        return linear(normed, linear_weight, bias)

    norm_pc = _rms_norm_pc(nrows, ncols, eps)
    matvec_pc = _matvec_pc(T, in_feat, out_feat)

    x_bytes = _to_bytes(x_flat)
    norm_out_size = nrows * ncols * 4  # intermediate (stays on GPU)
    matvec_out_size = T * out_feat * 4

    w_norm_gpu = _get_or_upload_weight(ctx, norm_weight)
    w_lin_gpu = _get_or_upload_weight(ctx, linear_weight)

    w_norm_binding = w_norm_gpu if w_norm_gpu is not None else bytes(ncols * 4)
    w_lin_binding = (
        w_lin_gpu if w_lin_gpu is not None else _to_bytes(linear_weight.float())
    )

    # execute_chained: Op0 = RMSNorm, Op1 = MatVec
    # Op0 bindings: [input_x, weight_norm]          output → inter_buf
    # Op1 bindings: [weight_linear] + inter_buf(auto) output → out_buf
    # inter_buf is passed as the LAST binding of Op1 automatically.
    _norm_out_bytes, matvec_bytes = ctx.execute_chained(
        "rms_norm_f32_mul",
        [x_bytes, w_norm_binding],
        norm_out_size,
        norm_pc,
        (nrows, 1, 1),
        "mul_mat_vec_f32_f32_f32",
        [w_lin_binding],  # Op1 gets [weight, inter_buf(auto), out_buf(auto)]
        matvec_out_size,
        matvec_pc,
        (out_feat, T, 1),
    )
    result = _from_bytes(matvec_bytes, (T, out_feat), torch.float32)
    if bias is not None:
        result = result + bias.float()
    return result.reshape(*orig_shape[:-1], out_feat)


# ─── Fused SwiGLU MLP (gate_up_proj → silu(gate)*up → down_proj) ─────────────


@lru_cache(maxsize=256)
def _swiglu_pc(t: int, ne00: int, ne20: int) -> bytes:  # noqa: N803
    """Pack push constants for `swiglu_f32` (shaders/swiglu.comp, via
    `glu_head.glsl`'s 16-field struct), computing
    `silu(src[..., :ne20]) * src[..., ne20:]` for a `[t, ne00]` (`ne00 ==
    2*ne20`) row-major source into a `[t, ne20]` row-major destination —
    exactly `vllm.model_executor.layers.activation.SiluAndMul.forward_native`'s
    `silu(x[..., :d]) * x[..., d:]`.

    `mode=0` ("Default"): reads `data_a[idx]` (gate) and
    `data_a[idx + ne00/2]` (up) from the SAME buffer — matches a *merged*
    `gate_up_proj` weight's single `[T, 2*intermediate]` output tensor
    (Qwen2MLP/Qwen3MLP's `gate_up_proj`), gate and up interleaved per row.
    `alpha`/`limit` are unused by this shader's `op()` (SiLU has no extra
    parameters) — always 0.

    `nb01`/`nb11` (row strides, in elements) are `ne00`/`ne20`
    respectively: both source and destination are freshly-produced,
    contiguous GPU buffers (this function's only caller,
    `fused_swiglu_mlp`, always passes a chain-ref to another op's fresh
    output — see `execute_batch`'s doc comment), so there is never a
    non-default stride to account for. `ne01`/`ne02`/`ne11`/`ne12`
    (batch dims) are `t`/`1` — no head/batch dimension beyond the token
    axis for the same reason `_matvec_pc`'s `ne02`/`ne12`/broadcast fields
    are inert here (see that function's own doc comment): this is always
    a single unbatched `[T, ...]` 2D tensor. `nb02`/`nb03`/`nb12`/`nb13`
    are consequently never read by `glu_main.glsl`'s indexing math (its
    `i3`/`i2` always resolve to 0 when `ne02`/`ne12` == 1) but are set to
    consistent (not garbage) values regardless, matching this file's
    existing convention for similarly-inert fields elsewhere (see
    `_matvec_pc`'s `ne02`/`ne12`/broadcast2/broadcast3 doc comment).
    """
    return struct.pack(
        "4I2f10I",
        t * ne20,
        ne00,
        ne20,
        0,  # N, ne00, ne20, mode
        0.0,
        0.0,  # alpha, limit
        ne00,
        ne00 * t,
        ne00 * t,  # nb01, nb02, nb03
        t,
        1,  # ne01, ne02
        ne20,
        ne20 * t,
        ne20 * t,  # nb11, nb12, nb13
        t,
        1,  # ne11, ne12
    )


def _wg512(n: int) -> int:
    """Workgroup count for `swiglu_f32`'s `local_size_x=512` (glu_head.glsl)."""
    return (n + 511) // 512


class FusedMlpUnavailableError(Exception):
    """Raised by `fused_swiglu_mlp` when this weight/shape/shader
    combination can't be dispatched as a single fused GPU submit (e.g. a
    weight too small to be GPU-eligible at all, or `swiglu_f32` not
    compiled). Callers should catch this and fall back to the separate
    `gate_up_proj` → activation → `down_proj` calls.
    """


def fused_swiglu_mlp(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU MLP block (`gate_up_proj` → `silu(gate)*up` → `down_proj`)
    in ONE `vkQueueSubmit`, using `execute_batch`'s chain-ref bindings
    (see its doc comment) to keep both intermediate activations
    (`gate_up_proj`'s `[T, 2*intermediate]` output and the SwiGLU
    activation's `[T, intermediate]` output) on the GPU between the three
    dispatches — only the block's final `[T, hidden]` output ever
    round-trips back to Python/CPU.

    This is the "fuse consecutively-dispatched matvecs/matmuls into one
    submit" optimization the per-module `vk_forward` hooks in
    model_runner.py cannot do on their own: `gate_up_proj` and
    `down_proj` are independent `nn.Module`s, each independently
    `execute_batch`-ing a single op today (see `linear()`) — there is no
    way to fuse across two separate Python-level module calls without
    hooking the *containing* MLP block's `forward()` instead of its
    constituent Linear modules, which is what `model_runner.py`'s
    `_wrap_qwen_mlp` (the caller of this function) does.

    Raises `FusedMlpUnavailableError` if either weight is too small to be
    GPU-eligible (mirrors `linear()`'s own `_MATVEC_MIN_WEIGHT_ELEMENTS`
    gate) or the required shaders aren't compiled — callers should catch
    this and fall back to separate `gate_up_proj(x)` /
    `SiluAndMul()` / `down_proj(...)` calls, exactly as if this function
    didn't exist.
    """
    ctx = get_context()

    orig_shape = x.shape
    hidden = gate_up_weight.shape[1]
    x_2d = x.float().reshape(-1, hidden)
    T = x_2d.shape[0]  # noqa: N806
    two_intermediate = gate_up_weight.shape[0]
    intermediate = two_intermediate // 2
    out_features = down_weight.shape[0]

    if (
        hidden * two_intermediate < _MATVEC_MIN_WEIGHT_ELEMENTS
        or intermediate * out_features < _MATVEC_MIN_WEIGHT_ELEMENTS
    ):
        raise FusedMlpUnavailableError("weight(s) too small to be GPU-eligible")

    available_shaders = _cached_available_shaders(ctx)
    if "swiglu_f32" not in available_shaders:
        raise FusedMlpUnavailableError("swiglu_f32 not compiled")

    use_f16_gate_up = _weight_uses_f16(ctx, gate_up_weight)
    use_f16_down = _weight_uses_f16(ctx, down_weight)

    if T < _MATVEC_THRESHOLD:
        shader_gate_up = (
            "mul_mat_vec_f16_f32_f32" if use_f16_gate_up else "mul_mat_vec_f32_f32_f32"
        )
        shader_down = (
            "mul_mat_vec_f16_f32_f32" if use_f16_down else "mul_mat_vec_f32_f32_f32"
        )
        pc_gate_up = _matvec_pc(T, hidden, two_intermediate)
        wg_gate_up = (two_intermediate, T, 1)
        pc_down = _matvec_pc(T, intermediate, out_features)
        wg_down = (out_features, T, 1)
    else:
        shader_gate_up = "matmul_f16_f32_fp32" if use_f16_gate_up else "matmul_f32_f32"
        shader_down = "matmul_f16_f32_fp32" if use_f16_down else "matmul_f32_f32"
        pc_gate_up = _matmul_pc(two_intermediate, T, hidden)
        wg_gate_up = _matmul_workgroups(two_intermediate, T)
        pc_down = _matmul_pc(out_features, T, intermediate)
        wg_down = _matmul_workgroups(out_features, T)

    if shader_gate_up not in available_shaders or shader_down not in available_shaders:
        raise FusedMlpUnavailableError(
            f"required matmul/matvec shader(s) not compiled: {shader_gate_up}, {shader_down}"
        )

    w_gate_up_gpu = _get_or_upload_weight(ctx, gate_up_weight, prefer_f16=True)
    w_down_gpu = _get_or_upload_weight(ctx, down_weight, prefer_f16=True)

    w_gate_up_binding = (
        w_gate_up_gpu
        if w_gate_up_gpu is not None
        else (
            _to_bytes(gate_up_weight.half())
            if use_f16_gate_up
            else _to_bytes(gate_up_weight.float())
        )
    )
    w_down_binding = (
        w_down_gpu
        if w_down_gpu is not None
        else (
            _to_bytes(down_weight.half())
            if use_f16_down
            else _to_bytes(down_weight.float())
        )
    )

    x_bytes = _to_bytes(x_2d)
    gate_up_out_size = T * two_intermediate * 4
    swiglu_out_size = T * intermediate * 4
    down_out_size = T * out_features * 4
    swiglu_pc = _swiglu_pc(T, two_intermediate, intermediate)

    # Op 0: gate_up_proj matmul/matvec. Op 1 (SwiGLU) needs its output, so
    # barrier_after=True.
    # Op 1: SwiGLU — bindings are [A, B] per glu_head.glsl; mode=0 never
    # reads B, but the shader still declares (and Vulkan still requires a
    # bound buffer for) that binding, so both point at Op 0's output.
    # Op 2 (down_proj) needs Op 1's output, so barrier_after=True.
    # Op 2: down_proj matmul/matvec — this block's real, final output, so
    # no barrier needed after it (nothing else in this batch reads it).
    ops = [
        (
            shader_gate_up,
            [w_gate_up_binding, x_bytes],
            [gate_up_out_size],
            pc_gate_up,
            wg_gate_up,
            True,
        ),
        (
            "swiglu_f32",
            [(0, 0), (0, 0)],
            [swiglu_out_size],
            swiglu_pc,
            (_wg512(T * intermediate), 1, 1),
            True,
        ),
        (
            shader_down,
            [w_down_binding, (1, 0)],
            [down_out_size],
            pc_down,
            wg_down,
            False,
        ),
    ]
    results = ctx.execute_batch(ops)
    final_bytes = results[2][0]
    result = _from_bytes(final_bytes, (T, out_features), torch.float32)
    return result.reshape(*orig_shape[:-1], out_features)


# ─── Attention (SDPA fallback) ───────────────────────────────────────────────


def _sdpa_per_request(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Per-request SDPA using PyTorch (attention stays on CPU)."""
    q_t = q.transpose(0, 1).unsqueeze(0).float()
    k_t = k.transpose(0, 1).unsqueeze(0).float()
    v_t = v.transpose(0, 1).unsqueeze(0).float()
    if num_kv_heads < num_heads:
        ratio = num_heads // num_kv_heads
        k_t = k_t.repeat_interleave(ratio, dim=1)
        v_t = v_t.repeat_interleave(ratio, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)
