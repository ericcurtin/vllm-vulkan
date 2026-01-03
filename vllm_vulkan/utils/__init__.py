"""
vLLM-Vulkan Utilities Module

This module provides utility functions for the Vulkan backend.
"""

from vllm_vulkan.utils.device import (
    get_device_properties,
    get_device_memory_info,
    get_device_compute_capability,
    is_device_available,
    synchronize_device,
)

__all__ = [
    "get_device_properties",
    "get_device_memory_info",
    "get_device_compute_capability",
    "is_device_available",
    "synchronize_device",
]
