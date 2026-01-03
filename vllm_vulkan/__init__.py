"""
vLLM-Vulkan: A vLLM hardware plugin for Vulkan-based GPU acceleration.

This module provides the entry point for vLLM's plugin system and exposes
the main functionality of the Vulkan backend.
"""

__version__ = "0.1.0"

# Try to import the Rust extension, fall back to stubs if not available
try:
    from vllm_vulkan._vllm_vulkan_rs import (
        PagedKVCache,
        VulkanBackend,
        VulkanBuffer,
        VulkanCommunicator,
        VulkanDevice,
        VulkanGraph,
        VulkanTensor,
        __vulkan_available__,
        enumerate_devices,
        get_device_count,
        get_device_info,
        get_memory_info,
        is_available,
        synchronize,
    )
    from vllm_vulkan._vllm_vulkan_rs import (
        copy_blocks_py as copy_blocks,
    )
    from vllm_vulkan._vllm_vulkan_rs import (
        flash_attention_py as flash_attention,
    )
    from vllm_vulkan._vllm_vulkan_rs import (
        paged_attention_py as paged_attention,
    )
    from vllm_vulkan._vllm_vulkan_rs import (
        reshape_and_cache_py as reshape_and_cache,
    )
    from vllm_vulkan._vllm_vulkan_rs import (
        swap_blocks_py as swap_blocks,
    )

    _RUST_AVAILABLE = True
except ImportError:
    # Rust extension not available - provide stub implementations
    _RUST_AVAILABLE = False
    __vulkan_available__ = False

    def is_available() -> bool:
        """Check if Vulkan is available."""
        return False

    def get_device_count() -> int:
        """Get the number of Vulkan devices."""
        return 0

    def enumerate_devices() -> list:
        """Enumerate available Vulkan devices."""
        return []

    def get_device_info(device_idx: int) -> dict:
        """Get device information."""
        raise RuntimeError("Vulkan not available")

    def synchronize() -> None:
        """Synchronize all Vulkan operations."""
        pass

    def get_memory_info(device_idx: int) -> tuple:
        """Get memory info for a device."""
        raise RuntimeError("Vulkan not available")


def register() -> str | None:
    """
    Register the Vulkan platform plugin with vLLM.

    This function is called by vLLM's plugin discovery system.
    It returns the fully qualified path to the VulkanPlatform class.

    Returns:
        str: The path to the VulkanPlatform class, or None if not available.
    """
    from vllm_vulkan.platform import VulkanPlatform

    if VulkanPlatform.is_available():
        return "vllm_vulkan.platform.VulkanPlatform"
    return None


# Public API
__all__ = [
    # Module info
    "__version__",
    "register",
    # Device functions
    "is_available",
    "get_device_count",
    "enumerate_devices",
    "get_device_info",
    "synchronize",
    "get_memory_info",
    # Classes (only available if Rust extension is loaded)
    "VulkanDevice",
    "VulkanBuffer",
    "VulkanBackend",
    "VulkanTensor",
    "VulkanGraph",
    "PagedKVCache",
    "VulkanCommunicator",
    # Operations
    "flash_attention",
    "paged_attention",
    "reshape_and_cache",
    "copy_blocks",
    "swap_blocks",
]
