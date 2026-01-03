"""
VulkanPlatform - Main platform class for vLLM Vulkan backend.

This module implements the Platform interface required by vLLM for hardware backends.
"""

from typing import TYPE_CHECKING, Any, Optional, Tuple, Type

import vllm_vulkan

if TYPE_CHECKING:
    import torch

# Module path constants for component classes
_ATTENTION_BACKEND_PATH = "vllm_vulkan.attention.backend:VulkanAttentionBackend"
_FLASH_ATTENTION_BACKEND_PATH = "vllm_vulkan.attention.backend:VulkanFlashAttentionBackend"
_MODEL_RUNNER_PATH = "vllm_vulkan.model_runner:VulkanModelRunner"
_WORKER_PATH = "vllm_vulkan.worker:VulkanWorker"
_EXECUTOR_PATH = "vllm_vulkan.executor:VulkanExecutor"


class VulkanPlatform:
    """
    vLLM Platform implementation for Vulkan backend.

    This class provides the interface between vLLM and the Vulkan-based
    GPU acceleration using ggml-vulkan.
    """

    device_name: str = "vulkan"
    device_type: str = "vulkan"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of a Vulkan device."""
        try:
            info = vllm_vulkan.get_device_info(device_id)
            return info.get("name", f"Vulkan Device {device_id}")
        except Exception:
            return f"Vulkan Device {device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total memory for a device in bytes."""
        try:
            _, total = vllm_vulkan.get_memory_info(device_id)
            return total
        except Exception:
            return 0

    @classmethod
    def is_available(cls) -> bool:
        """Check if Vulkan is available."""
        return vllm_vulkan.is_available()

    @classmethod
    def get_device_count(cls) -> int:
        """Get the number of Vulkan devices."""
        return vllm_vulkan.get_device_count()

    @classmethod
    def get_current_memory_usage(cls, device_id: int = 0) -> Tuple[int, int]:
        """
        Get current memory usage for a device.

        Returns:
            Tuple of (used_bytes, total_bytes)
        """
        try:
            return vllm_vulkan.get_memory_info(device_id)
        except Exception:
            return (0, 0)

    @classmethod
    def get_default_attn_backend(cls, selected_backend: Optional[str] = None) -> str:
        """
        Get the default attention backend for this platform.

        Args:
            selected_backend: User-selected backend (if any)

        Returns:
            String path to the attention backend class
        """
        if selected_backend and selected_backend.lower() == "flash_attn":
            return _FLASH_ATTENTION_BACKEND_PATH
        return _ATTENTION_BACKEND_PATH

    @classmethod
    def get_model_runner(cls, model_config: Any, **kwargs) -> str:
        """
        Get the model runner class for this platform.

        Returns:
            String path to the model runner class
        """
        return _MODEL_RUNNER_PATH

    @classmethod
    def get_worker(cls, **kwargs) -> str:
        """
        Get the worker class for this platform.

        Returns:
            String path to the worker class
        """
        return _WORKER_PATH

    @classmethod
    def get_executor(cls, **kwargs) -> str:
        """
        Get the executor class for this platform.

        Returns:
            String path to the executor class
        """
        return _EXECUTOR_PATH

    @classmethod
    def synchronize(cls) -> None:
        """Synchronize all Vulkan operations."""
        vllm_vulkan.synchronize()

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> Optional[Tuple[int, int]]:
        """
        Get device compute capability.

        For Vulkan, we return the Vulkan API version as a tuple.

        Returns:
            Tuple of (major, minor) version or None if not available
        """
        try:
            info = vllm_vulkan.get_device_info(device_id)
            api_version = info.get("api_version", "1.0.0")
            parts = api_version.split(".")
            if len(parts) >= 2:
                return (int(parts[0]), int(parts[1]))
        except Exception:
            pass
        return None

    @classmethod
    def supports_dtype(cls, dtype: "torch.dtype") -> bool:
        """
        Check if the platform supports a given dtype.

        Args:
            dtype: PyTorch dtype to check

        Returns:
            True if supported, False otherwise
        """
        import torch

        # Vulkan via ggml supports these dtypes
        supported = {
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int16,
            torch.int8,
        }
        return dtype in supported

    @classmethod
    def get_supported_quantizations(cls) -> list:
        """
        Get list of supported quantization methods.

        Returns:
            List of supported quantization method names
        """
        return [
            "none",
            "q4_0",
            "q4_1",
            "q5_0",
            "q5_1",
            "q8_0",
            "q8_1",
        ]

    @classmethod
    def verify_model_support(cls, model_config: Any) -> bool:
        """
        Verify that a model is supported by this platform.

        Args:
            model_config: The model configuration

        Returns:
            True if supported, raises exception otherwise
        """
        # Check if Vulkan is available
        if not cls.is_available():
            raise RuntimeError("Vulkan is not available on this system")

        # Check device count
        if cls.get_device_count() == 0:
            raise RuntimeError("No Vulkan devices found")

        return True

    @classmethod
    def empty_cache(cls) -> None:
        """Clear cached memory on all devices."""
        # In real implementation, this would clear GPU caches
        pass

    @classmethod
    def get_device_properties(cls, device_id: int = 0) -> dict:
        """
        Get all properties for a device.

        Args:
            device_id: The device index

        Returns:
            Dictionary of device properties
        """
        try:
            return vllm_vulkan.get_device_info(device_id)
        except Exception:
            return {}

    def __repr__(self) -> str:
        device_count = self.get_device_count()
        return f"VulkanPlatform(devices={device_count})"
