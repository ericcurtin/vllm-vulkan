"""
Attention Backend Selector

This module provides logic for selecting the appropriate attention backend.
"""

from typing import Optional, Type

import vllm_vulkan
from vllm_vulkan.attention.backend import (
    VulkanAttentionBackend,
    VulkanFlashAttentionBackend,
)


def get_attn_backend(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int] = None,
    dtype: str = "float16",
    block_size: int = 16,
    use_flash_attn: bool = True,
) -> Type[VulkanAttentionBackend]:
    """
    Select the appropriate attention backend.

    Args:
        num_heads: Number of attention heads
        head_size: Size of each attention head
        num_kv_heads: Number of key-value heads (for GQA)
        sliding_window: Sliding window size (if any)
        dtype: Data type for computation
        block_size: KV cache block size
        use_flash_attn: Whether to prefer flash attention

    Returns:
        Attention backend class
    """
    # Check if Vulkan is available
    if not vllm_vulkan.is_available():
        raise RuntimeError("Vulkan is not available on this system")

    # Check head size support
    supported_sizes = VulkanAttentionBackend.get_supported_head_sizes()
    if head_size not in supported_sizes:
        raise ValueError(
            f"Head size {head_size} not supported. "
            f"Supported sizes: {supported_sizes}"
        )

    # Try flash attention first if requested
    if use_flash_attn and VulkanFlashAttentionBackend.is_available():
        return VulkanFlashAttentionBackend

    # Fall back to standard attention
    return VulkanAttentionBackend


def which_attn_to_use(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: Optional[int] = None,
    dtype: str = "float16",
) -> str:
    """
    Determine which attention implementation to use.

    Returns a string identifier for the attention type.

    Args:
        num_heads: Number of attention heads
        head_size: Size of each attention head
        num_kv_heads: Number of key-value heads
        sliding_window: Sliding window size
        dtype: Data type

    Returns:
        String identifier for the attention type
    """
    if not vllm_vulkan.is_available():
        return "none"

    if VulkanFlashAttentionBackend.is_available():
        return "vulkan_flash"

    return "vulkan"
