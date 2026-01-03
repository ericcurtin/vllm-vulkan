"""
KV Cache Operations

This module provides cache management operations for the Vulkan backend.
"""

from typing import Any, Dict, List, Optional, Tuple

import vllm_vulkan


def reshape_and_cache(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
    kv_cache_dtype: str = "auto",
    kv_scale: float = 1.0,
) -> None:
    """
    Reshape key and value tensors and store them in the KV cache.

    This operation takes the key and value tensors from the current
    forward pass and stores them in the appropriate cache slots.

    Args:
        key: Key tensor from current forward pass
        value: Value tensor from current forward pass
        key_cache: Key cache to store into
        value_cache: Value cache to store into
        slot_mapping: Mapping from tokens to cache slots
        kv_cache_dtype: Data type for KV cache
        kv_scale: Scale factor for quantized KV cache
    """
    if not vllm_vulkan._RUST_AVAILABLE:
        raise RuntimeError("Vulkan backend not available")

    # Convert slot_mapping to list
    if hasattr(slot_mapping, "tolist"):
        slot_mapping_list = slot_mapping.tolist()
    else:
        slot_mapping_list = list(slot_mapping)

    # Call Rust implementation
    vllm_vulkan.reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping_list,
    )


def reshape_and_cache_flash(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
    kv_cache_dtype: str = "auto",
    kv_scale: float = 1.0,
) -> None:
    """
    Reshape and cache for flash attention format.

    Flash attention uses a different cache layout for better memory access.

    Args:
        key: Key tensor
        value: Value tensor
        key_cache: Key cache
        value_cache: Value cache
        slot_mapping: Slot mapping
        kv_cache_dtype: KV cache dtype
        kv_scale: KV scale
    """
    # For now, delegate to standard reshape_and_cache
    # Real implementation would use flash-specific layout
    reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype=kv_cache_dtype,
        kv_scale=kv_scale,
    )


def copy_blocks(
    key_caches: List[Any],
    value_caches: List[Any],
    block_mapping: Dict[int, List[int]],
) -> None:
    """
    Copy cache blocks (for forking sequences).

    When a sequence is forked (e.g., for beam search), the KV cache
    blocks need to be copied to new locations.

    Args:
        key_caches: List of key cache tensors (one per layer)
        value_caches: List of value cache tensors (one per layer)
        block_mapping: Mapping from source block to destination blocks
    """
    if not vllm_vulkan._RUST_AVAILABLE:
        raise RuntimeError("Vulkan backend not available")

    # Convert block_mapping to list of tuples
    block_mapping_list = [
        (src, dst)
        for src, dsts in block_mapping.items()
        for dst in dsts
    ]

    # Call Rust implementation
    vllm_vulkan.copy_blocks(
        block_mapping=block_mapping_list,
        key_caches=key_caches,
        value_caches=value_caches,
    )


def swap_blocks(
    src_key_cache: Any,
    src_value_cache: Any,
    dst_key_cache: Any,
    dst_value_cache: Any,
    block_mapping: Dict[int, int],
) -> None:
    """
    Swap cache blocks between GPU and CPU.

    This operation is used for block swapping during preemption
    or when the KV cache needs to be moved between devices.

    Args:
        src_key_cache: Source key cache (GPU or CPU)
        src_value_cache: Source value cache
        dst_key_cache: Destination key cache
        dst_value_cache: Destination value cache
        block_mapping: Mapping from source block to destination block
    """
    if not vllm_vulkan._RUST_AVAILABLE:
        raise RuntimeError("Vulkan backend not available")

    # Convert block_mapping to list of tuples
    block_mapping_list = list(block_mapping.items())

    # Call Rust implementation
    vllm_vulkan.swap_blocks(
        src_key_cache=src_key_cache,
        src_value_cache=src_value_cache,
        dst_key_cache=dst_key_cache,
        dst_value_cache=dst_value_cache,
        block_mapping=block_mapping_list,
    )


def allocate_kv_cache(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    device_idx: int = 0,
    dtype: str = "float16",
) -> Tuple[Any, Any]:
    """
    Allocate KV cache tensors.

    Args:
        num_blocks: Number of cache blocks to allocate
        block_size: Number of tokens per block
        num_layers: Number of model layers
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        device_idx: Device index
        dtype: Data type for the cache

    Returns:
        Tuple of (key_cache, value_cache) tensors
    """
    if not vllm_vulkan._RUST_AVAILABLE:
        raise RuntimeError("Vulkan backend not available")

    # Cache shape: [num_blocks, block_size, num_heads, head_dim]
    cache_shape = [num_blocks, block_size, num_heads, head_dim]

    key_cache = vllm_vulkan.VulkanTensor(
        shape=cache_shape,
        dtype=dtype,
        device_idx=device_idx,
    )

    value_cache = vllm_vulkan.VulkanTensor(
        shape=cache_shape,
        dtype=dtype,
        device_idx=device_idx,
    )

    return key_cache, value_cache


def get_cache_block_size(
    block_size: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Calculate the size of a cache block in bytes.

    Args:
        block_size: Number of tokens per block
        num_heads: Number of attention heads
        head_dim: Head dimension
        dtype_bytes: Bytes per element

    Returns:
        Block size in bytes
    """
    # Each block stores K and V
    return 2 * block_size * num_heads * head_dim * dtype_bytes
