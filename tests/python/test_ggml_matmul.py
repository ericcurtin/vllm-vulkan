# SPDX-License-Identifier: Apache-2.0
"""Tests for the first real ggml Vulkan compute path."""

import os
import sys

import numpy as np
import pytest

from vllm_vulkan import _rs


@pytest.fixture(scope="module")
def vulkan_runtime():
    # On macOS, the Vulkan loader may need an explicit ICD path. Without it,
    # ggml can abort inside the C backend before Rust can turn the failure into
    # a Python exception. Keep default test runs safe, and run this test with
    # VK_ICD_FILENAMES set when validating MoltenVK locally.
    if (
        sys.platform == "darwin"
        and not os.environ.get("VK_ICD_FILENAMES")
        and not _rs.is_available()
    ):
        pytest.skip("macOS Vulkan ICD is not configured")


@pytest.fixture(scope="module")
def log_vulkan_devices(vulkan_runtime):
    try:
        devices = _rs.enumerate_devices()
    except RuntimeError as exc:
        print(f"Vulkan device enumeration unavailable: {exc}")
    else:
        print(f"Vulkan devices visible to test: {devices}")


def run_vulkan_matmul_or_skip(left, right):
    try:
        return _rs.vulkan_matmul_with_backend(left, right)
    except RuntimeError as exc:
        unavailable_markers = (
            "ggml Vulkan backend found no Vulkan devices",
            "Vulkan device index",
            "ggml_backend_vk_init failed",
        )
        if any(marker in str(exc) for marker in unavailable_markers):
            pytest.skip(f"ggml Vulkan backend unavailable: {exc}")
        raise


@pytest.mark.parametrize(
    ("m", "k", "n", "rtol", "atol"),
    [
        (2, 3, 2, 1e-4, 1e-4),
        (1, 4, 3, 1e-4, 1e-4),
        (4, 5, 1, 1e-4, 1e-4),
        (8, 7, 6, 1e-4, 1e-4),
        # ggml's Vulkan matmul can differ from NumPy's CPU BLAS reference due
        # to backend-specific FP32 reduction order / precision choices. Keep a
        # larger shape in the smoke test, but use a tolerance observed on real
        # MoltenVK output instead of weakening the small exactness checks.
        (32, 64, 16, 1e-3, 1e-2),
    ],
)
def test_vulkan_matmul_matches_numpy(log_vulkan_devices, m, k, n, rtol, atol):
    rng = np.random.default_rng(seed=m * 100 + k * 10 + n)
    left = rng.normal(size=(m, k)).astype(np.float32)
    right = rng.normal(size=(k, n)).astype(np.float32)

    output, backend_name = run_vulkan_matmul_or_skip(left, right)
    actual = np.array(output, dtype=np.float32)

    print(f"ggml matmul backend: {backend_name}")
    assert "Vulkan" in backend_name, f"matmul ran on {backend_name}, not Vulkan"
    np.testing.assert_allclose(actual, left @ right, rtol=rtol, atol=atol)


def test_vulkan_matmul_rejects_bad_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        _rs.vulkan_matmul([[1.0, 2.0]], [[1.0, 2.0]])
