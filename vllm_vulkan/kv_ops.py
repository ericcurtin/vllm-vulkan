# SPDX-License-Identifier: Apache-2.0
"""Python-facing Vulkan paged KV-cache operations."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm_vulkan.kv_layout import VulkanPagedKVLayout

if TYPE_CHECKING:
    from vllm_vulkan._rs import GpuTensor, VulkanContext

_PAGED_KV_WRITE_SHADER = "paged_kv_write_f32"
_PAGED_ATTN_DECODE_SHADER = "paged_attn_decode_f32"
_PAGED_KV_WRITE_F16_SHADER = "paged_kv_write_f16"
_PAGED_ATTN_DECODE_F16_SHADER = "paged_attn_decode_f16"
_PAGED_KV_WRITE_WORKGROUP_SIZE = 256
_PAGED_ATTN_DECODE_WORKGROUP_SIZE = 256


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
    if shader_name not in ctx.available_shaders():
        raise RuntimeError(f"{shader_name} shader is not available")

    spec = layout.layer_spec(layer_index)
    if spec.dtype_size != dtype_size:
        raise ValueError(
            f"{shader_name} requires a {dtype_size}-byte KV-cache layout"
        )
    if cache.nbytes < layout.total_bytes:
        raise ValueError(
            f"cache buffer has {cache.nbytes} bytes, "
            f"expected at least {layout.total_bytes}"
        )

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
        dtype_size=4,
    )


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
    dtype_size: int,
) -> torch.Tensor:
    if shader_name not in ctx.available_shaders():
        raise RuntimeError(f"{shader_name} shader is not available")

    spec = layout.layer_spec(layer_index)
    if spec.dtype_size != dtype_size:
        raise ValueError(
            f"{shader_name} requires a {dtype_size}-byte KV-cache layout"
        )
    if cache.nbytes < layout.total_bytes:
        raise ValueError(
            f"cache buffer has {cache.nbytes} bytes, "
            f"expected at least {layout.total_bytes}"
        )
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
    expected_shape = (num_q_heads, spec.head_size)
    if tuple(q_f32.shape) != expected_shape:
        raise ValueError(
            f"Q shape {tuple(q_f32.shape)} does not match {expected_shape}"
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
    workgroups = (
        math.ceil(total_elements / _PAGED_ATTN_DECODE_WORKGROUP_SIZE),
        1,
        1,
    )
    output_nbytes = total_elements * np.dtype(np.float32).itemsize

    results = ctx.execute_batch(
        [
            (
                shader_name,
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
        ]
    )
    output = np.frombuffer(results[0][0], dtype=np.float32).copy()
    return torch.from_numpy(output.reshape(num_q_heads, spec.head_size))


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
