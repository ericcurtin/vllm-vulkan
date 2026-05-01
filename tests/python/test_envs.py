# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_vulkan.envs lazy environment variables."""

import pytest

import vllm_vulkan.envs as envs


def test_memory_fraction_default(monkeypatch):
    monkeypatch.delenv("VLLM_VULKAN_MEMORY_FRACTION", raising=False)
    val = envs.VLLM_VULKAN_MEMORY_FRACTION
    assert 0 < val <= 1


def test_memory_fraction_override(monkeypatch):
    monkeypatch.setenv("VLLM_VULKAN_MEMORY_FRACTION", "0.42")
    val = envs.VLLM_VULKAN_MEMORY_FRACTION
    assert val == pytest.approx(0.42)


def test_block_size_default(monkeypatch):
    monkeypatch.delenv("VLLM_VULKAN_BLOCK_SIZE", raising=False)
    val = envs.VLLM_VULKAN_BLOCK_SIZE
    assert isinstance(val, str)
    assert int(val) > 0


def test_debug_default(monkeypatch):
    monkeypatch.delenv("VLLM_VULKAN_DEBUG", raising=False)
    assert envs.VLLM_VULKAN_DEBUG is False


def test_debug_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_VULKAN_DEBUG", "1")
    assert envs.VLLM_VULKAN_DEBUG is True


def test_device_index_default(monkeypatch):
    monkeypatch.delenv("VLLM_VULKAN_DEVICE_INDEX", raising=False)
    val = envs.VLLM_VULKAN_DEVICE_INDEX
    assert val == 0


def test_rust_model_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_VULKAN_RUST_MODEL", "0")
    assert envs.VLLM_VULKAN_RUST_MODEL is False


def test_unknown_attr_raises():
    with pytest.raises(AttributeError, match="no attribute"):
        _ = envs.VLLM_VULKAN_DOES_NOT_EXIST  # type: ignore[attr-defined]


def test_environment_variables_dict_populated():
    assert "VLLM_VULKAN_MEMORY_FRACTION" in envs.environment_variables
    assert "VLLM_VULKAN_BLOCK_SIZE" in envs.environment_variables
    assert "VLLM_VULKAN_DEBUG" in envs.environment_variables
    assert "VLLM_VULKAN_DEVICE_INDEX" in envs.environment_variables
    assert "VLLM_VULKAN_DISABLE_ATTN" in envs.environment_variables
    assert "VLLM_VULKAN_RUST_MODEL" in envs.environment_variables
