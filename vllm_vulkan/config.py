# SPDX-License-Identifier: Apache-2.0
"""Configuration for vLLM Vulkan plugin via environment variables."""

from dataclasses import dataclass

import vllm_vulkan.envs as envs


@dataclass
class VulkanConfig:
    """Configuration for the vLLM Vulkan plugin."""

    memory_fraction: float  # (0, 1] — fraction of device memory for KV cache
    block_size: int  # tokens per KV-cache block
    debug: bool
    device_index: int  # Vulkan physical device index

    def __post_init__(self) -> None:
        if not (0 < self.memory_fraction <= 1):
            raise ValueError(
                f"Invalid VLLM_VULKAN_MEMORY_FRACTION={self.memory_fraction}. "
                "Must be in (0, 1]."
            )
        if self.block_size <= 0:
            raise ValueError(
                f"Invalid VLLM_VULKAN_BLOCK_SIZE={self.block_size}. "
                "Must be a positive integer."
            )
        if self.device_index < 0:
            raise ValueError(
                f"Invalid VLLM_VULKAN_DEVICE_INDEX={self.device_index}. Must be >= 0."
            )

    @classmethod
    def from_env(cls) -> "VulkanConfig":
        block_size_str = envs.VLLM_VULKAN_BLOCK_SIZE
        try:
            block_size = int(block_size_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid VLLM_VULKAN_BLOCK_SIZE={block_size_str!r}. "
                "Must be a positive integer."
            ) from e

        return cls(
            memory_fraction=envs.VLLM_VULKAN_MEMORY_FRACTION,
            block_size=block_size,
            debug=envs.VLLM_VULKAN_DEBUG,
            device_index=envs.VLLM_VULKAN_DEVICE_INDEX,
        )


_config: VulkanConfig | None = None


def get_config() -> VulkanConfig:
    """Return the global VulkanConfig, creating it from env on first call."""
    global _config  # noqa: PLW0603
    if _config is None:
        _config = VulkanConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config  # noqa: PLW0603
    _config = None
