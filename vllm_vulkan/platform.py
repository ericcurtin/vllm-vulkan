"""
VulkanPlatform - Main platform class for vLLM Vulkan backend.

This module implements the Platform interface required by vLLM for hardware backends.
"""

import logging
from typing import TYPE_CHECKING

import torch
from vllm.attention.selector import AttentionBackendEnum, AttentionSelectorConfig
from vllm.platforms.interface import Platform, PlatformEnum

import vllm_vulkan

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


class VulkanPlatform(Platform):
    """
    vLLM Platform implementation for Vulkan backend.

    This class provides the interface between vLLM and the Vulkan-based
    GPU acceleration using ggml-vulkan.
    """

    # Use CPU for PyTorch compatibility (like MetalPlatform does)
    _enum = PlatformEnum.OOT
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of a Vulkan device."""
        try:
            info = vllm_vulkan.get_device_info(device_id)
            return str(info.get("name", f"Vulkan Device {device_id}"))
        except Exception:
            return f"Vulkan Device {device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total memory for a device in bytes."""
        try:
            _, total = vllm_vulkan.get_memory_info(device_id)
            return int(total)
        except Exception:
            return 0

    @classmethod
    def get_device_available_memory(cls, device_id: int = 0) -> int:
        """Get available memory for a device in bytes."""
        try:
            used, total = vllm_vulkan.get_memory_info(device_id)
            return int(total) - int(used)
        except Exception:
            return 0

    @classmethod
    def is_available(cls) -> bool:
        """Check if Vulkan is available."""
        return bool(vllm_vulkan.is_available())

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> tuple[int, int]:
        """Get device compute capability (fake CUDA capability for compatibility)."""
        return (8, 0)

    @classmethod
    def get_device_count(cls) -> int:
        """Get the number of Vulkan devices."""
        return int(vllm_vulkan.get_device_count())

    @classmethod
    def set_device(cls, device_id: int) -> None:
        """Set the current device."""
        pass  # Vulkan handles device selection per-operation

    @classmethod
    def current_device(cls) -> int:
        """Get the current device index."""
        return 0

    @classmethod
    def synchronize(cls, device_id: int = 0) -> None:
        """Synchronize all Vulkan operations."""
        vllm_vulkan.synchronize()

    @classmethod
    def get_torch_device(cls, device_id: int = 0) -> torch.device:
        """Get the PyTorch device for this platform."""
        return torch.device("cpu")

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Check and update vLLM configuration for Vulkan compatibility."""
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        # Set worker class for Vulkan (use vLLM's CPU worker)
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.cpu_worker.CPUWorker"

        # Set executor backend (use uniproc for single device)
        if parallel_config.distributed_executor_backend in ("auto", None):
            parallel_config.distributed_executor_backend = "uni"

        # Disable features not supported on Vulkan
        parallel_config.disable_custom_all_reduce = True

        # Configure cache block size (required for CPU worker)
        if cache_config.block_size is None:
            cache_config.block_size = 16

        # Disable cascade attention (not supported)
        if model_config is not None:
            model_config.disable_cascade_attn = True

        # Log memory configuration
        total_mem = cls.get_device_total_memory()
        available_mem = cls.get_device_available_memory()
        logger.info(
            f"Vulkan memory: {total_mem / 1e9:.1f}GB total, "
            f"{available_mem / 1e9:.1f}GB available"
        )

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Verify quantization is supported."""
        supported = ["none", "fp16", "bfloat16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"]
        if quant not in supported:
            raise ValueError(
                f"Quantization '{quant}' not supported on Vulkan. "
                f"Supported: {supported}"
            )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: AttentionSelectorConfig,
    ) -> str:
        """Get the attention backend class for Vulkan."""
        if selected_backend and selected_backend != AttentionBackendEnum.CPU_ATTN:
            logger.info(f"Cannot use {selected_backend} backend on Vulkan.")
        if attn_selector_config.use_mla:
            raise NotImplementedError("MLA is not supported on Vulkan.")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on Vulkan.")
        return AttentionBackendEnum.CPU_ATTN.get_path()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Check if pin memory is available."""
        return False  # PyTorch tensors are on CPU

    def __repr__(self) -> str:
        device_count = self.get_device_count()
        return f"VulkanPlatform(devices={device_count})"
