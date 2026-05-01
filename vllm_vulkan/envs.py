# SPDX-License-Identifier: Apache-2.0
"""Environment variable definitions for the vLLM Vulkan plugin.

Single source of truth for all ``VLLM_VULKAN_*`` variables.  Mirrors the
lazy-evaluation pattern used by ``vllm/envs.py``: each variable is read from
``os.environ`` on access via ``__getattr__``, so values are never stale and
``monkeypatch.setenv`` works in tests without extra resets.

During plugin registration (``vllm_vulkan._register``), the
``environment_variables`` dict is merged into
``vllm.envs.environment_variables`` so that ``validate_environ()``
recognises our variables and does not emit spurious "Unknown vLLM environment
variable" warnings.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VLLM_VULKAN_MEMORY_FRACTION: float = 0.9
    VLLM_VULKAN_BLOCK_SIZE: str = "16"
    VLLM_VULKAN_DEBUG: bool = False
    VLLM_VULKAN_DEVICE_INDEX: int = 0
    VLLM_VULKAN_DISABLE_ATTN: bool = False
    VLLM_VULKAN_RUST_MODEL: bool = True

environment_variables: dict[str, Callable[[], Any]] = {
    # Fraction of device memory to use for KV cache (0, 1].
    "VLLM_VULKAN_MEMORY_FRACTION": lambda: float(
        os.getenv("VLLM_VULKAN_MEMORY_FRACTION", "0.9")
    ),
    # Tokens per KV-cache block (default 16).
    "VLLM_VULKAN_BLOCK_SIZE": lambda: os.getenv("VLLM_VULKAN_BLOCK_SIZE", "16"),
    # Enable verbose debug logging (default False).
    "VLLM_VULKAN_DEBUG": lambda: os.getenv("VLLM_VULKAN_DEBUG", "0") == "1",
    # Which Vulkan physical device index to use (default 0).
    "VLLM_VULKAN_DEVICE_INDEX": lambda: int(os.getenv("VLLM_VULKAN_DEVICE_INDEX", "0")),
    # Disable the experimental Vulkan attention decode path.
    "VLLM_VULKAN_DISABLE_ATTN": lambda: (
        os.getenv("VLLM_VULKAN_DISABLE_ATTN", "0") == "1"
    ),
    # Use the Rust VulkanModel / Vulkan module hooks during model execution.
    "VLLM_VULKAN_RUST_MODEL": lambda: os.getenv("VLLM_VULKAN_RUST_MODEL", "1") == "1",
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())
