# SPDX-License-Identifier: Apache-2.0
"""Python-facing Vulkan paged KV-cache operations."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_vulkan.kv_layout import KVCacheLayerSpec, VulkanPagedKVLayout

if TYPE_CHECKING:
    from vllm_vulkan._rs import GpuTensor, VulkanContext

_PAGED_KV_WRITE_SHADER = "paged_kv_write_f32"
_PAGED_ATTN_DECODE_SHADER = "paged_attn_decode_f32"
_PAGED_KV_WRITE_F16_SHADER = "paged_kv_write_f16"
_PAGED_ATTN_DECODE_F16_SHADER = "paged_attn_decode_f16"
_PAGED_ATTN_DECODE_COOP_SHADER = "paged_attn_decode_f32_coop"
_PAGED_ATTN_DECODE_F16_COOP_SHADER = "paged_attn_decode_f16_coop"
# "_512" variants: same shader source, compiled with BLOCK_SIZE=512
# instead of the default 256 (see paged_attn_decode_f32_coop.comp's
# BLOCK_SIZE comment) — preferred over the 256 variant when head_size is
# large enough (>=512, e.g. Gemma4-E2B's full-attention layers) to avoid
# wasted/idle threads in the cooperative dot-product reduction. Measured
# ~5-9% faster than BLOCK_SIZE=256 at head_size=512, while BLOCK_SIZE=256
# remains ~9-14% faster at head_size=256 — matching BLOCK_SIZE to
# head_size in both directions, not just unconditionally preferring the
# larger one.
_PAGED_ATTN_DECODE_COOP_512_SHADER = "paged_attn_decode_f32_coop_512"
_PAGED_ATTN_DECODE_F16_COOP_512_SHADER = "paged_attn_decode_f16_coop_512"
_PAGED_KV_WRITE_WORKGROUP_SIZE = 256
_PAGED_ATTN_DECODE_WORKGROUP_SIZE = 256
_PAGED_ATTN_DECODE_WORKGROUP_SIZE_LARGE = 512
_F32_NBYTES = np.dtype(np.float32).itemsize


def paged_kv_write_f16(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor | list[int] | tuple[int, ...],
) -> None:
    """Write f16 K/V tensors into a paged Vulkan KV-cache buffer."""
    _paged_kv_write(
        ctx=ctx,
        layout=layout,
        cache=cache,
        layer_index=layer_index,
        k=k,
        v=v,
        slot_mapping=slot_mapping,
        shader_name=_PAGED_KV_WRITE_F16_SHADER,
        dtype=torch.float16,
        dtype_size=2,
    )


def paged_kv_write_f32(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor | list[int] | tuple[int, ...],
) -> None:
    """Write f32 K/V tensors into a paged Vulkan KV-cache buffer.

    Args:
        ctx: Live Vulkan context.
        layout: Paged KV layout contract.
        cache: Persistent cache buffer allocated with at least
            ``layout.total_bytes`` bytes.
        layer_index: Layer whose K/V planes should be updated.
        k: ``[num_tokens, num_kv_heads, head_size]`` float tensor.
        v: ``[num_tokens, num_kv_heads, head_size]`` float tensor.
        slot_mapping: vLLM slots, where
            ``slot = physical_block_id * block_size + token_offset_in_block``.
    """
    _paged_kv_write(
        ctx=ctx,
        layout=layout,
        cache=cache,
        layer_index=layer_index,
        k=k,
        v=v,
        slot_mapping=slot_mapping,
        shader_name=_PAGED_KV_WRITE_SHADER,
        dtype=torch.float32,
        dtype_size=4,
    )


def _paged_kv_write(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor | list[int] | tuple[int, ...],
    shader_name: str,
    dtype: torch.dtype,
    dtype_size: int,
) -> None:
    spec = layout.layer_spec(layer_index)
    _require_shader(ctx, shader_name)
    _validate_cache_buffer(cache, layout)
    _validate_layout_dtype(spec.dtype_size, dtype_size, shader_name)

    k_cpu = k.detach().to(dtype=dtype, device="cpu").contiguous()
    v_cpu = v.detach().to(dtype=dtype, device="cpu").contiguous()
    if k_cpu.shape != v_cpu.shape:
        raise ValueError(
            f"K and V shapes must match, got {k_cpu.shape} and {v_cpu.shape}"
        )
    if k_cpu.ndim != 3:
        raise ValueError(f"K/V must be rank-3, got rank {k_cpu.ndim}")

    num_tokens = k_cpu.shape[0]
    if num_tokens <= 0:
        raise ValueError("K/V must contain at least one token")
    expected_shape = (num_tokens, spec.num_kv_heads, spec.head_size)
    if tuple(k_cpu.shape) != expected_shape:
        raise ValueError(
            f"K/V shape {tuple(k_cpu.shape)} does not match {expected_shape}"
        )

    slots = _slot_mapping_to_u32(
        slot_mapping, num_tokens, layout.capacity_tokens_per_layer
    )
    pc = _paged_kv_write_pc(layout, spec.layer_index, num_tokens)
    total_elements = num_tokens * spec.num_kv_heads * spec.head_size
    workgroups = (
        math.ceil(total_elements / _PAGED_KV_WRITE_WORKGROUP_SIZE),
        1,
        1,
    )

    ctx.execute_batch(
        [
            (
                shader_name,
                [
                    _tensor_to_bytes(k_cpu),
                    _tensor_to_bytes(v_cpu),
                    slots.tobytes(),
                    cache,
                ],
                [],
                pc,
                workgroups,
                False,
            )
        ]
    )


def paged_attn_decode_f16(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    q: torch.Tensor,
    block_table: torch.Tensor | list[int] | tuple[int, ...],
    seq_len: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode one sequence with f16 KV cache and f32 accumulation/output."""
    return _paged_attn_decode(
        ctx=ctx,
        layout=layout,
        cache=cache,
        layer_index=layer_index,
        q=q,
        block_table=block_table,
        seq_len=seq_len,
        scale=scale,
        shader_name=_PAGED_ATTN_DECODE_F16_SHADER,
        coop_shader_name=_PAGED_ATTN_DECODE_F16_COOP_SHADER,
        coop_512_shader_name=_PAGED_ATTN_DECODE_F16_COOP_512_SHADER,
        dtype_size=2,
    )


def paged_attn_decode_f32(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    q: torch.Tensor,
    block_table: torch.Tensor | list[int] | tuple[int, ...],
    seq_len: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode one sequence with f32 paged attention on Vulkan.

    This is the first correctness-oriented paged attention primitive.  It
    handles one query token and one sequence, reads K/V from the paged cache via
    ``block_table``, and returns ``[num_q_heads, head_size]``.  Batched variable
    length serving and optimized FlashAttention tiling are follow-up work.
    """
    return _paged_attn_decode(
        ctx=ctx,
        layout=layout,
        cache=cache,
        layer_index=layer_index,
        q=q,
        block_table=block_table,
        seq_len=seq_len,
        scale=scale,
        shader_name=_PAGED_ATTN_DECODE_SHADER,
        coop_shader_name=_PAGED_ATTN_DECODE_COOP_SHADER,
        coop_512_shader_name=_PAGED_ATTN_DECODE_COOP_512_SHADER,
        dtype_size=4,
    )


def paged_attn_decode_batch_f16(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    queries: list[torch.Tensor],
    block_tables: list[torch.Tensor | list[int] | tuple[int, ...]],
    seq_lens: list[int],
    scale: float | None = None,
) -> list[torch.Tensor]:
    """Batched form of paged_attn_decode_f16: decode every (query,
    block_table, seq_len) triple in the batch as a single vkQueueSubmit
    instead of one submit per token. See _paged_attn_decode_batch.
    """
    return _paged_attn_decode_batch(
        ctx=ctx,
        layout=layout,
        cache=cache,
        layer_index=layer_index,
        queries=queries,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        shader_name=_PAGED_ATTN_DECODE_F16_SHADER,
        coop_shader_name=_PAGED_ATTN_DECODE_F16_COOP_SHADER,
        coop_512_shader_name=_PAGED_ATTN_DECODE_F16_COOP_512_SHADER,
        dtype_size=2,
    )


def paged_attn_decode_batch_f32(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    queries: list[torch.Tensor],
    block_tables: list[torch.Tensor | list[int] | tuple[int, ...]],
    seq_lens: list[int],
    scale: float | None = None,
) -> list[torch.Tensor]:
    """Batched form of paged_attn_decode_f32: decode every (query,
    block_table, seq_len) triple in the batch as a single vkQueueSubmit
    instead of one submit per token. See _paged_attn_decode_batch.
    """
    return _paged_attn_decode_batch(
        ctx=ctx,
        layout=layout,
        cache=cache,
        layer_index=layer_index,
        queries=queries,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        shader_name=_PAGED_ATTN_DECODE_SHADER,
        coop_shader_name=_PAGED_ATTN_DECODE_COOP_SHADER,
        coop_512_shader_name=_PAGED_ATTN_DECODE_COOP_512_SHADER,
        dtype_size=4,
    )


def _resolve_paged_attn_decode_dispatch(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    shader_name: str,
    coop_shader_name: str,
    coop_512_shader_name: str,
    dtype_size: int,
) -> tuple[str, int, KVCacheLayerSpec]:
    """Resolve everything about a decode dispatch that's constant across
    every token in a batch - which shader variant to use, and the shared
    cache buffer/layout dtype validation - exactly once.

    _select_decode_shader queries the Vulkan context for available shaders
    (an FFI call into Rust), and _validate_cache_buffer/_validate_layout_dtype
    don't depend on any per-token value (q, block_table, seq_len) at all;
    doing all three once per batch instead of once per token is what makes
    _paged_attn_decode_batch's single execute_batch call actually cheap
    per-token, not just "one submit instead of N" with the same per-token
    Python/FFI overhead still paid beforehand.

    Returns (dispatch_shader_name, coop_workgroup_size, spec):
    coop_workgroup_size is 0 when the plain (non-coop) shader was
    selected (its workgroup formula doesn't depend on this value at
    all — see _build_paged_attn_decode_op), otherwise it's the
    BLOCK_SIZE (256 or 512) the selected `_coop`/`_coop_512` shader was
    compiled with.
    """
    spec = layout.layer_spec(layer_index)
    dispatch_shader_name, coop_workgroup_size = _select_decode_shader(
        ctx, shader_name, coop_shader_name, coop_512_shader_name, spec.head_size
    )
    _validate_cache_buffer(cache, layout)
    _validate_layout_dtype(spec.dtype_size, dtype_size, shader_name)
    return dispatch_shader_name, coop_workgroup_size, spec


def _build_paged_attn_decode_op(
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    spec: KVCacheLayerSpec,
    dispatch_shader_name: str,
    coop_workgroup_size: int,
    q: torch.Tensor,
    block_table: torch.Tensor | list[int] | tuple[int, ...],
    seq_len: int,
    scale: float | None,
) -> tuple[tuple[str, list, list[int], bytes, tuple[int, int, int], bool], int, int]:
    """Validate per-token inputs and build one execute_batch op-tuple for a
    single (query, block_table, seq_len) decode step, without submitting
    it. Everything shared across a whole batch (shader selection, cache/
    dtype validation) must already be resolved by
    _resolve_paged_attn_decode_dispatch and passed in via
    dispatch_shader_name/coop_workgroup_size/spec - this function only
    does work that genuinely varies per token.

    Returns (op_tuple, num_q_heads, head_size) so callers can either submit
    it alone (_paged_attn_decode, one token) or collect several from
    different tokens/sequences and submit them all as a single
    ctx.execute_batch call (_paged_attn_decode_batch) - one vkQueueSubmit
    and one fence wait for the whole batch instead of one per token.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if seq_len > layout.capacity_tokens_per_layer:
        raise ValueError(
            f"seq_len {seq_len} exceeds capacity {layout.capacity_tokens_per_layer}"
        )

    q_f32 = q.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if q_f32.ndim != 2:
        raise ValueError(f"Q must be rank-2, got rank {q_f32.ndim}")

    num_q_heads = q_f32.shape[0]
    if q_f32.shape[1] != spec.head_size:
        raise ValueError(
            f"Q head size {q_f32.shape[1]} does not match {spec.head_size}"
        )
    if num_q_heads <= 0:
        raise ValueError("Q must contain at least one query head")
    if num_q_heads % spec.num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads {num_q_heads} must be divisible by "
            f"num_kv_heads {spec.num_kv_heads}"
        )

    needed_blocks = math.ceil(seq_len / spec.block_size)
    blocks = _block_table_to_u32(block_table, needed_blocks, layout.num_blocks)
    pc = _paged_attn_decode_pc(
        layout=layout,
        layer_index=layer_index,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        scale=scale if scale is not None else 1.0 / math.sqrt(spec.head_size),
    )
    total_elements = num_q_heads * spec.head_size
    if coop_workgroup_size > 0:
        workgroups = (
            num_q_heads,
            math.ceil(spec.head_size / coop_workgroup_size),
            1,
        )
    else:
        workgroups = (
            math.ceil(total_elements / _PAGED_ATTN_DECODE_WORKGROUP_SIZE),
            1,
            1,
        )
    output_nbytes = total_elements * _F32_NBYTES

    op = (
        dispatch_shader_name,
        [
            _tensor_to_bytes(q_f32),
            blocks.tobytes(),
            cache,
        ],
        [output_nbytes],
        pc,
        workgroups,
        False,
    )
    return op, num_q_heads, spec.head_size


def _paged_attn_decode(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    q: torch.Tensor,
    block_table: torch.Tensor | list[int] | tuple[int, ...],
    seq_len: int,
    scale: float | None,
    shader_name: str,
    coop_shader_name: str,
    coop_512_shader_name: str,
    dtype_size: int,
) -> torch.Tensor:
    dispatch_shader_name, coop_workgroup_size, spec = (
        _resolve_paged_attn_decode_dispatch(
            ctx=ctx,
            layout=layout,
            cache=cache,
            layer_index=layer_index,
            shader_name=shader_name,
            coop_shader_name=coop_shader_name,
            coop_512_shader_name=coop_512_shader_name,
            dtype_size=dtype_size,
        )
    )
    op, num_q_heads, head_size = _build_paged_attn_decode_op(
        layout=layout,
        cache=cache,
        layer_index=layer_index,
        spec=spec,
        dispatch_shader_name=dispatch_shader_name,
        coop_workgroup_size=coop_workgroup_size,
        q=q,
        block_table=block_table,
        seq_len=seq_len,
        scale=scale,
    )
    results = ctx.execute_batch([op])
    output = np.frombuffer(results[0][0], dtype=np.float32).copy()
    return torch.from_numpy(output.reshape(num_q_heads, head_size))


def _paged_attn_decode_batch(
    ctx: VulkanContext,
    layout: VulkanPagedKVLayout,
    cache: GpuTensor,
    layer_index: int,
    queries: list[torch.Tensor],
    block_tables: list[torch.Tensor | list[int] | tuple[int, ...]],
    seq_lens: list[int],
    scale: float | None,
    shader_name: str,
    coop_shader_name: str,
    coop_512_shader_name: str,
    dtype_size: int,
) -> list[torch.Tensor]:
    """Decode a whole batch of (query, block_table, seq_len) triples - one
    per token/sequence - as a SINGLE ctx.execute_batch call: one
    vkQueueSubmit and one fence wait for the entire batch, instead of one
    per token (see attention.py's _try_vulkan_decode, which used to call
    the single-token _paged_attn_decode in a Python loop).
    """
    if not (len(queries) == len(block_tables) == len(seq_lens)):
        raise ValueError(
            "queries, block_tables, and seq_lens must have the same length "
            f"(got {len(queries)}, {len(block_tables)}, {len(seq_lens)})"
        )
    if not queries:
        return []

    dispatch_shader_name, coop_workgroup_size, spec = (
        _resolve_paged_attn_decode_dispatch(
            ctx=ctx,
            layout=layout,
            cache=cache,
            layer_index=layer_index,
            shader_name=shader_name,
            coop_shader_name=coop_shader_name,
            coop_512_shader_name=coop_512_shader_name,
            dtype_size=dtype_size,
        )
    )

    ops = []
    shapes = []
    for q, block_table, seq_len in zip(queries, block_tables, seq_lens, strict=True):
        op, num_q_heads, head_size = _build_paged_attn_decode_op(
            layout=layout,
            cache=cache,
            layer_index=layer_index,
            spec=spec,
            dispatch_shader_name=dispatch_shader_name,
            coop_workgroup_size=coop_workgroup_size,
            q=q,
            block_table=block_table,
            seq_len=seq_len,
            scale=scale,
        )
        ops.append(op)
        shapes.append((num_q_heads, head_size))

    results = ctx.execute_batch(ops)
    outputs = []
    for (num_q_heads, head_size), result in zip(shapes, results, strict=True):
        output = np.frombuffer(result[0], dtype=np.float32).copy()
        outputs.append(torch.from_numpy(output.reshape(num_q_heads, head_size)))
    return outputs


def _require_shader(ctx: VulkanContext, shader_name: str) -> None:
    if shader_name not in ctx.available_shaders():
        raise RuntimeError(f"{shader_name} shader is not available")


def _select_decode_shader(
    ctx: VulkanContext,
    shader_name: str,
    coop_shader_name: str,
    coop_512_shader_name: str,
    head_size: int,
) -> tuple[str, int]:
    """Returns (dispatch_shader_name, coop_workgroup_size).

    Prefers whichever `_coop`/`_coop_512` variant's BLOCK_SIZE best
    matches `head_size` (see paged_attn_decode_f32_coop.comp's BLOCK_SIZE
    comment for the measured rationale) — large enough head_size prefers
    the 512-wide variant, otherwise the default 256-wide one, each falling
    back to whichever coop variant IS available if the preferred one
    isn't (e.g. an older build), and finally to the plain non-coop shader
    if neither coop variant is available at all. `coop_workgroup_size` is
    0 when the plain shader was selected (see _build_paged_attn_decode_op,
    whose workgroup formula doesn't use this value in that case).
    """
    available_shaders = ctx.available_shaders()
    prefer_512 = head_size >= _PAGED_ATTN_DECODE_WORKGROUP_SIZE_LARGE
    if prefer_512 and coop_512_shader_name in available_shaders:
        return coop_512_shader_name, _PAGED_ATTN_DECODE_WORKGROUP_SIZE_LARGE
    if coop_shader_name in available_shaders:
        return coop_shader_name, _PAGED_ATTN_DECODE_WORKGROUP_SIZE
    if coop_512_shader_name in available_shaders:
        return coop_512_shader_name, _PAGED_ATTN_DECODE_WORKGROUP_SIZE_LARGE
    if shader_name in available_shaders:
        return shader_name, 0
    raise RuntimeError(f"{shader_name} shader is not available")


def _validate_cache_buffer(cache: GpuTensor, layout: VulkanPagedKVLayout) -> None:
    if cache.nbytes < layout.total_bytes:
        raise ValueError(
            f"cache buffer has {cache.nbytes} bytes, "
            f"expected at least {layout.total_bytes}"
        )


def _validate_layout_dtype(
    actual_dtype_size: int, expected_dtype_size: int, shader_name: str
) -> None:
    if actual_dtype_size != expected_dtype_size:
        raise ValueError(
            f"{shader_name} requires a {expected_dtype_size}-byte KV-cache layout"
        )


def _paged_kv_write_pc(
    layout: VulkanPagedKVLayout,
    layer_index: int,
    num_tokens: int,
) -> bytes:
    spec = layout.layer_spec(layer_index)
    return struct.pack(
        "<7I",
        num_tokens,
        spec.num_kv_heads,
        spec.head_size,
        spec.block_size,
        layout.layer_base_offset(layer_index) // spec.dtype_size,
        spec.plane_bytes_per_block // spec.dtype_size,
        spec.bytes_per_block // spec.dtype_size,
    )


def _paged_attn_decode_pc(
    layout: VulkanPagedKVLayout,
    layer_index: int,
    seq_len: int,
    num_q_heads: int,
    scale: float,
) -> bytes:
    spec = layout.layer_spec(layer_index)
    return struct.pack(
        "<8If",
        seq_len,
        num_q_heads,
        spec.num_kv_heads,
        spec.head_size,
        spec.block_size,
        layout.layer_base_offset(layer_index) // spec.dtype_size,
        spec.plane_bytes_per_block // spec.dtype_size,
        spec.bytes_per_block // spec.dtype_size,
        scale,
    )


def _slot_mapping_to_u32(
    slot_mapping: torch.Tensor | list[int] | tuple[int, ...],
    num_tokens: int,
    capacity_tokens: int,
) -> np.ndarray:
    if isinstance(slot_mapping, torch.Tensor):
        slots = slot_mapping.detach().to(device="cpu", dtype=torch.int64).contiguous()
        slot_values = slots.numpy()
    else:
        slot_values = np.asarray(slot_mapping, dtype=np.int64)

    if slot_values.shape != (num_tokens,):
        raise ValueError(
            f"slot_mapping shape {slot_values.shape} does not match ({num_tokens},)"
        )
    if np.any(slot_values < 0):
        raise ValueError("slot_mapping must not contain negative slots")
    if np.any(slot_values >= capacity_tokens):
        raise ValueError(
            f"slot_mapping contains a slot outside capacity {capacity_tokens}"
        )
    return slot_values.astype(np.uint32, copy=False)


def _block_table_to_u32(
    block_table: torch.Tensor | list[int] | tuple[int, ...],
    needed_blocks: int,
    num_blocks: int,
) -> np.ndarray:
    if isinstance(block_table, torch.Tensor):
        blocks = block_table.detach().to(device="cpu", dtype=torch.int64).contiguous()
        block_values = blocks.numpy()
    else:
        block_values = np.asarray(block_table, dtype=np.int64)

    if block_values.ndim != 1:
        raise ValueError(f"block_table must be rank-1, got shape {block_values.shape}")
    if block_values.shape[0] < needed_blocks:
        raise ValueError(
            f"block_table has {block_values.shape[0]} block(s), "
            f"expected at least {needed_blocks}"
        )
    active_blocks = block_values[:needed_blocks]
    if np.any(active_blocks < 0):
        raise ValueError("block_table must not contain negative block ids")
    if np.any(active_blocks >= num_blocks):
        raise ValueError(
            f"block_table contains a block id outside {num_blocks} block(s)"
        )
    return active_blocks.astype(np.uint32, copy=False)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.numpy().tobytes()
