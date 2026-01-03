"""
vLLM-Vulkan Distributed Module

This module provides distributed communication primitives for multi-GPU inference.
"""

from vllm_vulkan.distributed.communicator import (
    VulkanDistributedCommunicator,
    init_distributed,
    is_initialized,
    get_world_size,
    get_rank,
    barrier,
)

__all__ = [
    "VulkanDistributedCommunicator",
    "init_distributed",
    "is_initialized",
    "get_world_size",
    "get_rank",
    "barrier",
]
