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


def set_context(ctx: VulkanContext) -> None:
    global _ctx
    _ctx = ctx

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


def _get_or_upload_weight(ctx: VulkanContext, weight: torch.Tensor) -> object | None:
    """Return a persistent GpuTensor for weight, uploading once on first use."""
    try:
        cached = _weight_cache.get(weight)
        if cached is not None:
            return cached
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


def _from_bytes(data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bfloat16:
        arr = np.frombuffer(data, dtype=np.int16).reshape(shape)
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    np_dtype = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }.get(dtype, np.float32)
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    return torch.from_numpy(arr.copy())


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


# ─── RMS Norm ────────────────────────────────────────────────────────────────

_MATVEC_THRESHOLD = 4  # T < this → matvec shader; T >= this → CPU


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
    """Linear layer: decode path uses Vulkan matvec, prefill uses CPU."""
    ctx = get_context()

    orig_shape = x.shape
    in_feat = weight.shape[1]
    out_feat = weight.shape[0]
    x_2d = x.float().reshape(-1, in_feat)
    T = x_2d.shape[0]  # noqa: N806

    if T < _MATVEC_THRESHOLD and "mul_mat_vec_f32_f32_f32" in ctx.available_shaders():
        result = _vulkan_matvec(ctx, x_2d, weight)
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
    """Dispatch mul_mat_vec_f32_f32_f32 for decode (T<4).

    NOT mul_mat_vec_f32_f32_f32_subgroup: that variant was found to
    silently diverge from the correct result for ncols>=~256 (i.e.
    essentially every real hidden/intermediate size) -- see
    subgroup_matvec_correctness_tests in src/lib.rs for the measured
    divergence and root-cause discussion. mul_mat_vec_f32_f32_f32 (the
    plain, non-subgroup variant) has an identical binding/push-constant/
    workgroup-dispatch convention and was confirmed correct at every
    size tested.
    """
    # Cache key on original weight (before .float()) - and the .float()
    # conversion itself only happens lazily, in the `w_gpu is None`
    # fallback branch below, since on the (overwhelmingly common) cache-hit
    # path the GPU-resident buffer already holds the converted weight and
    # recomputing weight.float() here would just be a wasted full
    # weight-matrix copy, every single call, for a value nothing else uses.
    w_gpu = _get_or_upload_weight(ctx, weight)

    T, K = x.shape  # noqa: N806
    N = weight.shape[0]  # noqa: N806
    pc = _matvec_pc(T, K, N)
    x_bytes = _to_bytes(x)
    out_size = T * N * 4

    w_binding = w_gpu if w_gpu is not None else _to_bytes(weight.float())

    results = ctx.execute_batch(
        [
            (
                "mul_mat_vec_f32_f32_f32",
                [w_binding, x_bytes],
                [out_size],
                pc,
                (N, T, 1),
                False,
            ),
        ]
    )
    return _from_bytes(results[0][0], (T, N), torch.float32)


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

    # Decode path only (T < threshold); prefill falls back to CPU for linear.
    if (
        T >= _MATVEC_THRESHOLD
        or "rms_norm_f32_mul" not in ctx.available_shaders()
        or "mul_mat_vec_f32_f32_f32" not in ctx.available_shaders()
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
