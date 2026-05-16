# SPDX-License-Identifier: Apache-2.0
"""Tests for VulkanPlatform."""

import platform as py_platform
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="vllm not installed; skipping platform tests")

from vllm_vulkan.config import reset_config  # noqa: E402
from vllm_vulkan.platform import VulkanPlatform  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    yield
    reset_config()


def test_is_available_returns_bool():
    result = VulkanPlatform.is_available()
    assert isinstance(result, bool)


def test_get_device_count_non_negative():
    count = VulkanPlatform.get_device_count()
    assert count >= 0


def test_get_device_name_returns_string():
    if VulkanPlatform.get_device_count() > 0:
        name = VulkanPlatform.get_device_name(0)
        assert isinstance(name, str)
        assert len(name) > 0


def test_get_device_total_memory_positive(monkeypatch):
    monkeypatch.setenv("VLLM_VULKAN_MEMORY_FRACTION", "0.5")
    reset_config()
    total = VulkanPlatform.get_device_total_memory()
    assert total > 0


def test_get_device_available_memory_non_negative(monkeypatch):
    monkeypatch.setenv("VLLM_VULKAN_MEMORY_FRACTION", "0.5")
    reset_config()
    available = VulkanPlatform.get_device_available_memory()
    assert available >= 0


def test_get_device_capability():
    cap = VulkanPlatform.get_device_capability()
    assert cap.major >= 0
    assert cap.minor >= 0


def test_current_device():
    device = VulkanPlatform.current_device()
    assert device >= 0


def test_is_pin_memory_available():
    assert VulkanPlatform.is_pin_memory_available() is False


def test_get_torch_device():
    import torch

    device = VulkanPlatform.get_torch_device()
    assert device == torch.device("cpu")


def test_unsupported_platform_not_available(monkeypatch):
    """Verify the platform returns False on unsupported OSes."""
    monkeypatch.setattr(py_platform, "system", lambda: "Windows")
    # Force re-check — is_available probes the platform each call.
    result = VulkanPlatform.is_available()
    assert result is False


def test_register_returns_class_path_or_none():
    import vllm_vulkan

    result = vllm_vulkan._register()
    assert result is None or result == "vllm_vulkan.platform.VulkanPlatform"


def test_check_and_update_config_sets_kv_cache_memory_bytes(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_KVCACHE_SPACE", "5")
    monkeypatch.setenv("VLLM_VULKAN_ALLOW_COMPILE", "1")

    cache_config = SimpleNamespace(
        block_size=None,
        kv_cache_memory_bytes=None,
    )
    parallel_config = SimpleNamespace(
        worker_cls="auto",
        distributed_executor_backend="auto",
        disable_custom_all_reduce=False,
    )
    scheduler_config = SimpleNamespace(async_scheduling=True)
    compilation_config = SimpleNamespace(mode=None)
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=cache_config,
        model_config=None,
        scheduler_config=scheduler_config,
        compilation_config=compilation_config,
    )

    VulkanPlatform.check_and_update_config(vllm_config)

    assert cache_config.kv_cache_memory_bytes == 5 * 1024**3
    assert cache_config.block_size == 16
    assert parallel_config.worker_cls == "vllm_vulkan.worker.VulkanWorker"
    assert parallel_config.distributed_executor_backend == "uni"
    assert parallel_config.disable_custom_all_reduce is True
    assert scheduler_config.async_scheduling is False
