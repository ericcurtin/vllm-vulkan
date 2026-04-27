# SPDX-License-Identifier: Apache-2.0
"""Tests for VulkanConfig."""

import pytest

from vllm_vulkan.config import VulkanConfig, get_config, reset_config


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    yield
    reset_config()


def test_default_config_from_env(monkeypatch):
    monkeypatch.delenv("VLLM_VULKAN_MEMORY_FRACTION", raising=False)
    monkeypatch.delenv("VLLM_VULKAN_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("VLLM_VULKAN_DEBUG", raising=False)
    monkeypatch.delenv("VLLM_VULKAN_DEVICE_INDEX", raising=False)

    cfg = VulkanConfig.from_env()
    assert 0 < cfg.memory_fraction <= 1
    assert cfg.block_size > 0
    assert isinstance(cfg.debug, bool)
    assert cfg.device_index >= 0


def test_memory_fraction_validation():
    with pytest.raises(ValueError, match="MEMORY_FRACTION"):
        VulkanConfig(memory_fraction=0.0, block_size=16, debug=False, device_index=0)

    with pytest.raises(ValueError, match="MEMORY_FRACTION"):
        VulkanConfig(memory_fraction=1.5, block_size=16, debug=False, device_index=0)

    # Boundary values
    VulkanConfig(memory_fraction=0.01, block_size=16, debug=False, device_index=0)
    VulkanConfig(memory_fraction=1.0, block_size=16, debug=False, device_index=0)


def test_block_size_validation():
    with pytest.raises(ValueError, match="BLOCK_SIZE"):
        VulkanConfig(memory_fraction=0.9, block_size=0, debug=False, device_index=0)

    with pytest.raises(ValueError, match="BLOCK_SIZE"):
        VulkanConfig(memory_fraction=0.9, block_size=-1, debug=False, device_index=0)


def test_device_index_validation():
    with pytest.raises(ValueError, match="DEVICE_INDEX"):
        VulkanConfig(memory_fraction=0.9, block_size=16, debug=False, device_index=-1)


def test_get_config_singleton(monkeypatch):
    monkeypatch.setenv("VLLM_VULKAN_MEMORY_FRACTION", "0.5")
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2
    assert cfg1.memory_fraction == pytest.approx(0.5)


def test_reset_config(monkeypatch):
    monkeypatch.setenv("VLLM_VULKAN_MEMORY_FRACTION", "0.7")
    cfg1 = get_config()
    assert cfg1.memory_fraction == pytest.approx(0.7)

    reset_config()
    monkeypatch.setenv("VLLM_VULKAN_MEMORY_FRACTION", "0.3")
    cfg2 = get_config()
    assert cfg2.memory_fraction == pytest.approx(0.3)
    assert cfg1 is not cfg2


def test_env_debug_flag(monkeypatch):
    monkeypatch.setenv("VLLM_VULKAN_DEBUG", "1")
    cfg = VulkanConfig.from_env()
    assert cfg.debug is True

    reset_config()
    monkeypatch.setenv("VLLM_VULKAN_DEBUG", "0")
    cfg2 = VulkanConfig.from_env()
    assert cfg2.debug is False


def test_env_device_index(monkeypatch):
    monkeypatch.setenv("VLLM_VULKAN_DEVICE_INDEX", "2")
    cfg = VulkanConfig.from_env()
    assert cfg.device_index == 2
