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

_VLLM_VULKAN_MEMORY_FRACTION_DEFAULT = 0.9
_VLLM_VULKAN_BLOCK_SIZE_DEFAULT = "16"
_VLLM_VULKAN_DEBUG_DEFAULT = False
_VLLM_VULKAN_DEVICE_INDEX_DEFAULT = 0
_VLLM_VULKAN_DISABLE_ATTN_DEFAULT = False
_VLLM_VULKAN_RUST_MODEL_DEFAULT = True


def _bool_env(name: str, default: bool) -> Callable[[], bool]:
    default_value = "1" if default else "0"
    return lambda: os.getenv(name, default_value) == "1"


def _float_env(name: str, default: float) -> Callable[[], float]:
    return lambda: float(os.getenv(name, str(default)))


def _int_env(name: str, default: int) -> Callable[[], int]:
    return lambda: int(os.getenv(name, str(default)))


def _str_env(name: str, default: str) -> Callable[[], str]:
    return lambda: os.getenv(name, default)


if TYPE_CHECKING:
    VLLM_VULKAN_MEMORY_FRACTION: float = _VLLM_VULKAN_MEMORY_FRACTION_DEFAULT
    VLLM_VULKAN_BLOCK_SIZE: str = _VLLM_VULKAN_BLOCK_SIZE_DEFAULT
    VLLM_VULKAN_DEBUG: bool = _VLLM_VULKAN_DEBUG_DEFAULT
    VLLM_VULKAN_DEVICE_INDEX: int = _VLLM_VULKAN_DEVICE_INDEX_DEFAULT
    VLLM_VULKAN_DISABLE_ATTN: bool = _VLLM_VULKAN_DISABLE_ATTN_DEFAULT
    VLLM_VULKAN_RUST_MODEL: bool = _VLLM_VULKAN_RUST_MODEL_DEFAULT

environment_variables: dict[str, Callable[[], Any]] = {
    # Fraction of device memory to use for KV cache (0, 1].
    "VLLM_VULKAN_MEMORY_FRACTION": _float_env(
        "VLLM_VULKAN_MEMORY_FRACTION", _VLLM_VULKAN_MEMORY_FRACTION_DEFAULT
    ),
    # Tokens per KV-cache block (default 16).
    "VLLM_VULKAN_BLOCK_SIZE": _str_env(
        "VLLM_VULKAN_BLOCK_SIZE", _VLLM_VULKAN_BLOCK_SIZE_DEFAULT
    ),
    # Enable verbose debug logging (default False).
    "VLLM_VULKAN_DEBUG": _bool_env("VLLM_VULKAN_DEBUG", _VLLM_VULKAN_DEBUG_DEFAULT),
    # Which Vulkan physical device index to use (default 0).
    "VLLM_VULKAN_DEVICE_INDEX": _int_env(
        "VLLM_VULKAN_DEVICE_INDEX", _VLLM_VULKAN_DEVICE_INDEX_DEFAULT
    ),
    # Disable the experimental Vulkan attention decode path.
    "VLLM_VULKAN_DISABLE_ATTN": _bool_env(
        "VLLM_VULKAN_DISABLE_ATTN", _VLLM_VULKAN_DISABLE_ATTN_DEFAULT
    ),
    # Use the Rust VulkanModel / Vulkan module hooks during model execution.
    "VLLM_VULKAN_RUST_MODEL": _bool_env(
        "VLLM_VULKAN_RUST_MODEL", _VLLM_VULKAN_RUST_MODEL_DEFAULT
    ),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())
