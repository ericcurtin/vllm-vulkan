# SPDX-License-Identifier: Apache-2.0
"""VulkanPlatform — vLLM Platform interface for the Vulkan backend.

On macOS, Vulkan is provided by KosmicKrisp (a community MoltenVK fork that
translates Vulkan API calls to Metal). On Linux, the system Vulkan loader
exposes native GPU drivers.

Availability matrix:
  macOS  aarch64  — KosmicKrisp required (brew install kosmic-krisp / molten-vk)
  macOS  x86_64   — KosmicKrisp required (less common)
  Linux  x86_64   — native Vulkan driver required
  Linux  aarch64  — native Vulkan driver required
"""

import logging
import os
import platform as py_platform
from typing import TYPE_CHECKING

import psutil
import torch

from vllm_vulkan.config import get_config

# vllm is an optional runtime dependency when using the plugin standalone
# (e.g. running unit tests without a full vllm install).  Import the base
# classes at module level only when available; otherwise define lightweight
# stubs so the module can still be imported.
try:
    from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    _VLLM_AVAILABLE = True
except ModuleNotFoundError:
    _VLLM_AVAILABLE = False

    class Platform:  # type: ignore[no-redef]
        pass

    class PlatformEnum:  # type: ignore[no-redef]
        OOT = "OOT"

    class DeviceCapability:  # type: ignore[no-redef]
        def __init__(self, major: int = 0, minor: int = 0) -> None:
            self.major = major
            self.minor = minor

    class AttentionBackendEnum:  # type: ignore[no-redef]
        CPU_ATTN = None

        @staticmethod
        def get_path() -> str:
            return ""


if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = logging.getLogger(__name__)


class VulkanPlatform(Platform):
    """vLLM Platform implementation for Vulkan (KosmicKrisp on macOS, native on Linux)."""

    _enum: PlatformEnum = PlatformEnum.OOT
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    # ─── Device queries ──────────────────────────────────────────────────────

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        try:
            from vllm_vulkan._rs import get_device_info

            info = get_device_info(device_id)
            return str(info.get("name", f"Vulkan Device {device_id}"))
        except Exception:
            return f"Vulkan Device {device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Total memory budget for KV cache, in bytes.

        We allocate KV cache on CPU (the worker uses vLLM's CPU worker path),
        so we report system RAM scaled by the configured memory fraction.
        """
        config = get_config()
        total = psutil.virtual_memory().total
        return int(total * config.memory_fraction)

    @classmethod
    def get_device_available_memory(cls, device_id: int = 0) -> int:
        config = get_config()
        available = psutil.virtual_memory().available
        return int(available * config.memory_fraction)

    @classmethod
    def is_available(cls) -> bool:
        """Return True when at least one Vulkan device is present.

        On macOS this requires KosmicKrisp to be installed so that the Vulkan
        loader can enumerate Metal-backed devices.  The Rust extension performs
        the actual ``vkEnumeratePhysicalDevices`` call; if the shared library
        cannot be loaded (e.g. libvulkan not found) the import falls back to
        ``False``.
        """
        # Platform filter: only macOS aarch64 / Linux x86_64+aarch64.
        machine = py_platform.machine()
        system = py_platform.system()
        if system == "Darwin" and machine not in ("arm64", "x86_64"):
            return False
        if system not in ("Darwin", "Linux"):
            return False

        try:
            from vllm_vulkan._rs import is_available as _is_available

            return bool(_is_available())
        except (ImportError, OSError):
            return False

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        return DeviceCapability(major=8, minor=0)

    @classmethod
    def get_device_count(cls) -> int:
        try:
            from vllm_vulkan._rs import get_device_count

            return int(get_device_count())
        except (ImportError, OSError):
            return 0

    # ─── Device management ───────────────────────────────────────────────────

    @classmethod
    def set_device(cls, device_id: int) -> None:
        # Vulkan handles device selection per-operation via VkDevice.
        pass

    @classmethod
    def current_device(cls) -> int:
        return get_config().device_index

    @classmethod
    def synchronize(cls, device_id: int = 0) -> None:
        try:
            from vllm_vulkan._rs import synchronize

            synchronize()
        except (ImportError, OSError):
            pass

    @classmethod
    def get_torch_device(cls, device_id: int = 0) -> torch.device:
        return torch.device("cpu")

    # ─── vLLM config integration ─────────────────────────────────────────────

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Patch vLLM configuration for Vulkan / CPU-worker compatibility."""
        config = get_config()
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        if config.debug:
            logger.info("Vulkan config: %s", config)

        # Use vLLM's built-in CPU worker — it handles tokenisation, sampling
        # and KV-cache management.  Vulkan compute is invoked for attention
        # kernels through the ops layer.
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.cpu_worker.CPUWorker"

        if parallel_config.distributed_executor_backend in ("auto", None):
            parallel_config.distributed_executor_backend = "uni"

        parallel_config.disable_custom_all_reduce = True

        # Block size minimum — CPU attention requires a multiple of 16.
        if (
            cache_config.block_size is None
            or cache_config.block_size < config.block_size
        ):
            cache_config.block_size = config.block_size

        # CPU KV cache space (required by CPUWorker).
        if cache_config.cpu_kvcache_space_bytes is None:
            kv_gb = float(os.environ.get("VLLM_CPU_KVCACHE_SPACE", "4"))
            cache_config.cpu_kvcache_space_bytes = int(kv_gb * 1024**3)

        if model_config is not None:
            model_config.disable_cascade_attn = True

        total_mem = cls.get_device_total_memory()
        avail_mem = cls.get_device_available_memory()
        logger.info(
            "Vulkan memory budget: %.1fGB total, %.1fGB available",
            total_mem / 1e9,
            avail_mem / 1e9,
        )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
    ) -> str:
        if selected_backend and selected_backend != AttentionBackendEnum.CPU_ATTN:
            logger.info(
                "Cannot use %s backend on Vulkan; falling back to CPU_ATTN.",
                selected_backend,
            )
        if attn_selector_config.use_mla:
            raise NotImplementedError("MLA attention is not supported on Vulkan.")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse attention is not supported on Vulkan.")
        return AttentionBackendEnum.CPU_ATTN.get_path()

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        # Let the model implementation decide; we pass through all quant types.
        pass

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # PyTorch tensors live on CPU; pin_memory is not needed.
        return False

    def __repr__(self) -> str:
        return f"VulkanPlatform(devices={self.get_device_count()})"
