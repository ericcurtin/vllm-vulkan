"""
vLLM-Vulkan Distributed Module

This module provides distributed communication primitives for multi-GPU inference.
"""

from vllm_vulkan.distributed.communicator import (
    VulkanDistributedCommunicator,
    barrier,
    get_rank,
    get_world_size,
    init_distributed,
    is_initialized,
)

__all__ = [
    "VulkanDistributedCommunicator",
    "init_distributed",
    "is_initialized",
    "get_world_size",
    "get_rank",
    "barrier",
]
