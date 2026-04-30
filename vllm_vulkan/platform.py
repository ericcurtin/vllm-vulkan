# SPDX-License-Identifier: Apache-2.0
"""VulkanPlatform — vLLM Platform interface for the Vulkan backend.

On macOS, Vulkan is provided by KosmicKrisp (Mesa/Zink software Vulkan driver
that translates Vulkan API calls to Metal). On Linux, the system Vulkan loader
exposes native GPU drivers.

Availability matrix:
  macOS  aarch64  — KosmicKrisp required (installed by install.sh)
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
from vllm_vulkan.kv_layout import infer_kv_layer_specs

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

        # Use VulkanWorker (a CPUWorker subclass that gracefully handles the
        # absence of the vllm._C compiled extension for thread-affinity binding,
        # and installs VulkanModelRunner for GPU-accelerated compute).
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_vulkan.worker.VulkanWorker"

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
        # We reserve memory for:
        #   - Model weights in PyTorch (fp16/bf16):  ~model_params * 2 bytes
        #   - Vulkan weight copies (fp32):            ~model_params * 4 bytes
        #   - OS + Python runtime overhead:           ~4 GB
        # KV cache gets the remainder × safety_fraction.
        if cache_config.cpu_kvcache_space_bytes is None:
            kv_env = os.environ.get("VLLM_CPU_KVCACHE_SPACE")
            if kv_env is not None:
                kv_gb = float(kv_env)
            else:
                import psutil as _psutil  # noqa: PLC0415

                total_ram_gb = _psutil.virtual_memory().total / (1024**3)

                # Estimate model weight memory:
                # bf16 weights (already loaded) + float32 Vulkan copies
                _model_params_b = sum(
                    p.numel()
                    for p in getattr(vllm_config, "model_config", None) and [] or []
                )
                # Rough estimate: 6 bytes/param (2 bf16 + 4 f32)
                # For E2B ~2B params: 12 GB; for 31B: ~186 GB
                # Use a conservative 35% of total RAM for KV cache
                # Reserve memory for: PyTorch weights (~4GB), Vulkan weight
                # copies (~10GB), OS (~4GB). KV cache gets the rest up to cap.
                # 115GB total - 4 - 10 - 4 = ~97GB, but use conservative 8GB
                kv_gb = max(4.0, min(total_ram_gb * 0.07, 8.0))  # cap at 8 GB
            cache_config.cpu_kvcache_space_bytes = int(kv_gb * 1024**3)
            logger.info("Vulkan/CPU KV cache space: %.1f GB", kv_gb)

        if model_config is not None:
            model_config.disable_cascade_attn = True

            # Auto-cap max_model_len when the model's native context window
            # would require more KV cache than is available.  This avoids the
            # "220 GiB KV cache needed" error on machines with limited RAM.
            # Users can always pass --max-model-len explicitly to override.
            if (
                cache_config.cpu_kvcache_space_bytes is not None
                and model_config.max_model_len is not None
            ):
                # Estimate KV cache bytes per token using the same paged KV
                # layout contract that Vulkan attention kernels will consume.
                try:
                    specs = infer_kv_layer_specs(
                        model_config.hf_config,
                        block_size=cache_config.block_size or config.block_size,
                        dtype=model_config.dtype,
                    )
                    bytes_per_token = sum(
                        spec.bytes_per_token for spec in specs
                    )
                    max_tokens_in_kv = int(
                        cache_config.cpu_kvcache_space_bytes / bytes_per_token
                    )
                    if max_tokens_in_kv < model_config.max_model_len:
                        # Leave 20% headroom and cap at a multiple of block_size
                        safe_max = int(max_tokens_in_kv * 0.9)
                        bs = cache_config.block_size or config.block_size
                        safe_max = (safe_max // bs) * bs
                        if safe_max > 0 and safe_max < model_config.max_model_len:
                            logger.info(
                                "Vulkan/CPU: capping max_model_len from %d to %d "
                                "based on %.1f GB KV cache budget.",
                                model_config.max_model_len,
                                safe_max,
                                cache_config.cpu_kvcache_space_bytes / 1e9,
                            )
                            model_config.max_model_len = safe_max
                except Exception as exc:
                    logger.debug("Could not estimate max_model_len cap: %s", exc)

        # Async scheduling requires CUDA streams (torch.cuda.current_stream) and
        # is not compatible with the CPU/Vulkan backend.  Disable it explicitly.
        scheduler_config = vllm_config.scheduler_config
        if (
            scheduler_config is not None
            and getattr(scheduler_config, "async_scheduling", None) is not False
        ):
            scheduler_config.async_scheduling = False

        # The Vulkan/CPU backend uses eager (non-compiled) execution by default.
        # torch.compile with the vLLM inductor backend requires CUDA-specific
        # custom ops and is not supported on the CPU path.  Eager execution is
        # functionally equivalent and avoids compilation failures.
        # Users may opt into compilation by setting VLLM_VULKAN_ALLOW_COMPILE=1.
        allow_compile = os.environ.get("VLLM_VULKAN_ALLOW_COMPILE", "0") == "1"
        compilation_config = vllm_config.compilation_config
        if not allow_compile:
            from vllm.config.compilation import (  # noqa: PLC0415
                CompilationMode,
                CUDAGraphMode,
            )

            if compilation_config.mode != CompilationMode.NONE:
                logger.info(
                    "Vulkan/CPU platform: disabling torch.compile (eager mode). "
                    "Set VLLM_VULKAN_ALLOW_COMPILE=1 to enable compilation."
                )
                compilation_config.mode = CompilationMode.NONE
                compilation_config.cudagraph_mode = CUDAGraphMode.NONE

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
        **kwargs: object,
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
    def _vllm_c_available(cls) -> bool:
        """Return True if the vllm._C compiled extension is loadable."""
        try:
            import vllm._C  # noqa: F401

            return True
        except (ImportError, ModuleNotFoundError):
            return False

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
