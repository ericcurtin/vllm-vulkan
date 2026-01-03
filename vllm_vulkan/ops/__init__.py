"""
vLLM-Vulkan Operations Module

This module provides operation wrappers for the Vulkan backend.
"""

from vllm_vulkan.ops.attention import (
    flash_attention,
    paged_attention_v1,
    paged_attention_v2,
)
from vllm_vulkan.ops.cache import (
    reshape_and_cache,
    copy_blocks,
    swap_blocks,
)
from vllm_vulkan.ops.quantization import (
    quantize_tensor,
    dequantize_tensor,
)

__all__ = [
    # Attention
    "flash_attention",
    "paged_attention_v1",
    "paged_attention_v2",
    # Cache
    "reshape_and_cache",
    "copy_blocks",
    "swap_blocks",
    # Quantization
    "quantize_tensor",
    "dequantize_tensor",
]
