# SPDX-License-Identifier: Apache-2.0
"""Tests for the Vulkan attention backend shim."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm", reason="vllm not installed; skipping attention tests")
_rs = pytest.importorskip("vllm_vulkan._rs", exc_type=ImportError)

from vllm.v1.attention.backend import AttentionType  # noqa: E402
from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata  # noqa: E402

from vllm_vulkan.attention import VulkanAttentionBackendImpl  # noqa: E402


def _require_vulkan_context():
    if not _rs.is_available():
        pytest.skip("no Vulkan device available")
    try:
        return _rs.VulkanContext(0)
    except RuntimeError as exc:
        pytest.skip(f"VulkanContext unavailable: {exc}")


def test_vulkan_attention_backend_uses_paged_decode_for_single_token_batch():
    _require_vulkan_context()

    num_heads = 4
    num_kv_heads = 2
    head_size = 32
    block_size = 16
    num_blocks = 4
    num_reqs = 2
    scale = head_size**-0.5

    query = torch.randn(num_reqs, num_heads, head_size, dtype=torch.float32)
    key = torch.randn(num_reqs, num_kv_heads, head_size, dtype=torch.float16)
    value = torch.randn(num_reqs, num_kv_heads, head_size, dtype=torch.float16)
    kv_cache = torch.zeros(
        (2, num_blocks, num_kv_heads, block_size, head_size),
        dtype=torch.float16,
    )
    output = torch.empty((num_reqs, num_heads, head_size), dtype=torch.float32)

    metadata = CPUAttentionMetadata(
        isa="vec16",
        num_actual_tokens=num_reqs,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, block_size], dtype=torch.int64),
        scheduler_metadata=None,
        causal=True,
    )
    impl = VulkanAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )

    result = impl.forward(
        layer=None,  # type: ignore[arg-type]
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )

    # With seq_len=1 per request, softmax is exactly 1 for the current token, so
    # decode output is the request's V head repeated through GQA.
    expected = []
    gqa_ratio = num_heads // num_kv_heads
    for req_idx in range(num_reqs):
        rows = []
        for q_head in range(num_heads):
            rows.append(value[req_idx, q_head // gqa_ratio].float())
        expected.append(torch.stack(rows))
    expected = torch.stack(expected)

    assert impl._last_vulkan_decode_used is True
    assert result is output

    torch.testing.assert_close(output, expected, rtol=5e-3, atol=5e-3)
