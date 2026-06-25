"""Vulkan Linear dispatch correctness for small batches.

The Vulkan matmul kernel tiles 4 rows and returns incorrect results for M < 4
(the M=1 decode step), which produced garbage tokens during generation.
``_wrap_linear`` pads small batches up to 4 rows and slices back; these tests
verify the wrapped Linear matches a plain ``nn.Linear`` across M=1..8.
"""

import pytest
import torch
from torch import nn

_rs = pytest.importorskip("vllm_vulkan._rs", exc_type=ImportError)


def _require_device():
    if not _rs.is_available():
        pytest.skip("no Vulkan device available")


def _ready_ops():
    from vllm_vulkan import vulkan_ops
    from vllm_vulkan._rs import VulkanContext

    if not vulkan_ops.is_ready():
        vulkan_ops.set_context(VulkanContext(0))
    return vulkan_ops


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4, 8, 17])
@pytest.mark.parametrize("bias", [False, True])
def test_wrapped_linear_matches_torch_for_small_batch(m, bias):
    _require_device()
    _ready_ops()
    from vllm_vulkan.model_runner import _wrap_linear

    torch.manual_seed(0)
    k, n = 896, 896
    lin = nn.Linear(k, n, bias=bias)
    ref = nn.Linear(k, n, bias=bias)
    ref.load_state_dict(lin.state_dict())

    _wrap_linear(lin)  # dispatch lin.forward to Vulkan

    x = torch.randn(m, k)
    got = lin(x)
    exp = ref(x)
    assert torch.allclose(got, exp, atol=1e-2, rtol=1e-2), f"M={m} bias={bias} mismatch"


def test_raw_kernel_exact_for_m_at_least_4():
    """Guard the working path: the matmul kernel is exact for M>=4. (The M<4
    path is wrong, which is what _wrap_linear pads around; a future shader fix
    should make this hold for all M and let the workaround be removed.)
    """
    _require_device()
    ops = _ready_ops()
    torch.manual_seed(0)
    k, n = 896, 896
    w = torch.randn(n, k)
    for m in (4, 8, 32):
        x = torch.randn(m, k)
        assert torch.allclose(
            ops.linear(x, w, None), torch.nn.functional.linear(x, w), atol=1e-2, rtol=1e-2
        ), f"M={m}"
