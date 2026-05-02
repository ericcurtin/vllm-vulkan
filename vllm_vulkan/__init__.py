# SPDX-License-Identifier: Apache-2.0
"""vLLM Vulkan Plugin — cross-platform GPU inference via Vulkan.

On macOS (aarch64), Vulkan calls are translated to Metal by KosmicKrisp.
On Linux (x86_64 / aarch64), native Vulkan is used directly.

This module is discovered by vLLM via the ``vllm.platform_plugins`` entry
point and must expose a ``register()`` function.
"""

import logging
import os
import sys

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Mirror vLLM log level to the vllm_vulkan logger."""
    try:
        from vllm.envs import VLLM_LOGGING_LEVEL

        vllm_logger = logging.getLogger("vllm")
        vulkan_logger = logging.getLogger("vllm_vulkan")
        vulkan_logger.setLevel(logging.getLevelName(VLLM_LOGGING_LEVEL))

        if vllm_logger.handlers and not vulkan_logger.handlers:
            for handler in vllm_logger.handlers:
                vulkan_logger.addHandler(handler)
            vulkan_logger.propagate = False
    except Exception:
        pass


def _apply_macos_defaults() -> None:
    """Apply safe process-start defaults for macOS.

    vLLM's v1 engine forks a worker process; on macOS the Objective-C runtime
    is not fork-safe. Defaulting to ``spawn`` avoids this.
    """
    if sys.platform != "darwin":
        return
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") is not None:
        return
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    logger.debug(
        "macOS detected + Vulkan plugin active: defaulting "
        "VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
        "Set explicitly to override."
    )


# ─── Lazy imports ────────────────────────────────────────────────────────────


def __getattr__(name: str):  # noqa: ANN001, ANN201
    """Lazy import module components to avoid loading vLLM at import time."""
    if name == "VulkanConfig":
        from vllm_vulkan.config import VulkanConfig

        return VulkanConfig
    if name == "KVCacheLayerSpec":
        from vllm_vulkan.kv_layout import KVCacheLayerSpec

        return KVCacheLayerSpec
    if name == "VulkanPagedKVLayout":
        from vllm_vulkan.kv_layout import VulkanPagedKVLayout

        return VulkanPagedKVLayout
    if name == "get_config":
        from vllm_vulkan.config import get_config

        return get_config
    if name == "reset_config":
        from vllm_vulkan.config import reset_config

        return reset_config
    if name == "VulkanPlatform":
        from vllm_vulkan.platform import VulkanPlatform

        return VulkanPlatform
    if name == "register":
        return _register
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VulkanConfig",
    "KVCacheLayerSpec",
    "VulkanPlatform",
    "VulkanPagedKVLayout",
    "get_config",
    "reset_config",
    "register",
]


# ─── Plugin entry point ───────────────────────────────────────────────────────


def _register() -> str | None:
    """Register the Vulkan platform plugin with vLLM.

    Called by vLLM's plugin discovery system.

    Returns:
        Fully-qualified class name if the platform is available, else None.
    """
    _configure_logging()
    _apply_macos_defaults()

    # Register our env vars with vLLM's registry so validate_environ()
    # does not warn about unknown VLLM_VULKAN_* variables.
    try:
        import vllm.envs

        from vllm_vulkan.envs import environment_variables as vulkan_env_vars

        vllm.envs.environment_variables.update(vulkan_env_vars)
    except Exception:
        pass

    # Apply pure-Python fallbacks for vllm._C operations that are unavailable
    # when vLLM is used from its Python source tree without C extensions.
    try:
        from vllm_vulkan.patches import apply_patches

        apply_patches()
    except Exception:
        pass

    from vllm_vulkan.platform import VulkanPlatform

    if VulkanPlatform.is_available():
        return "vllm_vulkan.platform.VulkanPlatform"
    return None
