# SPDX-License-Identifier: Apache-2.0
"""Tests for Vulkan paged KV-cache operations."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_rs = pytest.importorskip("vllm_vulkan._rs", exc_type=ImportError)

from vllm_vulkan.kv_layout import KVCacheLayerSpec, VulkanPagedKVLayout  # noqa: E402
from vllm_vulkan.kv_ops import (  # noqa: E402
    paged_attn_decode_f16,
    paged_attn_decode_f32,
    paged_kv_write_f16,
    paged_kv_write_f32,
)


def _require_vulkan_context():
    if not _rs.is_available():
        pytest.skip("no Vulkan device available")
    try:
        return _rs.VulkanContext(0)
    except RuntimeError as exc:
        pytest.skip(f"VulkanContext unavailable: {exc}")


def test_paged_kv_write_f32_round_trips_gpu_cache_slots():
    ctx = _require_vulkan_context()
    assert "paged_kv_write_f32" in ctx.available_shaders()

    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=2,
        head_size=4,
        block_size=4,
        dtype_size=4,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=3)
    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))

    k = torch.arange(3 * spec.num_kv_heads * spec.head_size, dtype=torch.float32)
    k = k.reshape(3, spec.num_kv_heads, spec.head_size)
    v = k + 1000
    slot_mapping = torch.tensor([0, 5, 10], dtype=torch.int64)

    paged_kv_write_f32(ctx, layout, cache, 0, k, v, slot_mapping)

    raw = ctx.read_activation(cache)
    cache_view = np.frombuffer(raw, dtype=np.float32)
    for token_index, slot in enumerate(slot_mapping.tolist()):
        for kv_head in range(spec.num_kv_heads):
            for head_element in range(spec.head_size):
                k_offset = layout.slot_offset(0, slot, kv_head, head_element, "k") // 4
                v_offset = layout.slot_offset(0, slot, kv_head, head_element, "v") // 4
                assert cache_view[k_offset] == pytest.approx(
                    float(k[token_index, kv_head, head_element])
                )
                assert cache_view[v_offset] == pytest.approx(
                    float(v[token_index, kv_head, head_element])
                )

    untouched_slot = 1
    assert cache_view[layout.slot_offset(0, untouched_slot, 0, 0, "k") // 4] == 0.0
    assert cache_view[layout.slot_offset(0, untouched_slot, 0, 0, "v") // 4] == 0.0


def test_paged_kv_write_f16_round_trips_gpu_cache_slots():
    ctx = _require_vulkan_context()
    if "paged_kv_write_f16" not in ctx.available_shaders():
        pytest.skip("paged_kv_write_f16 shader is unavailable")

    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=2,
        head_size=4,
        block_size=4,
        dtype_size=2,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=3)
    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))

    k = torch.arange(3 * spec.num_kv_heads * spec.head_size, dtype=torch.float16)
    k = k.reshape(3, spec.num_kv_heads, spec.head_size)
    v = k + 1000
    slot_mapping = torch.tensor([0, 5, 10], dtype=torch.int64)

    paged_kv_write_f16(ctx, layout, cache, 0, k, v, slot_mapping)

    raw = ctx.read_activation(cache)
    cache_view = np.frombuffer(raw, dtype=np.float16)
    for token_index, slot in enumerate(slot_mapping.tolist()):
        for kv_head in range(spec.num_kv_heads):
            for head_element in range(spec.head_size):
                k_offset = layout.slot_offset(0, slot, kv_head, head_element, "k") // 2
                v_offset = layout.slot_offset(0, slot, kv_head, head_element, "v") // 2
                assert cache_view[k_offset] == pytest.approx(
                    float(k[token_index, kv_head, head_element])
                )
                assert cache_view[v_offset] == pytest.approx(
                    float(v[token_index, kv_head, head_element])
                )


def test_paged_kv_write_f32_validates_slot_capacity():
    ctx = _require_vulkan_context()
    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=1,
        head_size=2,
        block_size=4,
        dtype_size=4,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=1)
    cache = ctx.alloc_activation(layout.total_bytes)
    k = torch.zeros((1, spec.num_kv_heads, spec.head_size), dtype=torch.float32)
    v = torch.zeros_like(k)

    with pytest.raises(ValueError, match="outside capacity"):
        paged_kv_write_f32(ctx, layout, cache, 0, k, v, [layout.capacity_tokens_per_layer])


def test_paged_kv_write_f32_rejects_empty_tokens():
    ctx = _require_vulkan_context()
    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=1,
        head_size=2,
        block_size=4,
        dtype_size=4,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=1)
    cache = ctx.alloc_activation(layout.total_bytes)
    k = torch.zeros((0, spec.num_kv_heads, spec.head_size), dtype=torch.float32)
    v = torch.zeros_like(k)

    with pytest.raises(ValueError, match="at least one token"):
        paged_kv_write_f32(ctx, layout, cache, 0, k, v, [])


def test_paged_attn_decode_f32_matches_torch_reference():
    ctx = _require_vulkan_context()
    assert "paged_kv_write_f32" in ctx.available_shaders()
    assert "paged_attn_decode_f32" in ctx.available_shaders()

    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=2,
        head_size=8,
        block_size=4,
        dtype_size=4,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=4)
    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))

    seq_len = 6
    block_table = torch.tensor([2, 0], dtype=torch.int64)
    slot_mapping = torch.tensor(
        [
            int(block_table[token_idx // spec.block_size]) * spec.block_size
            + token_idx % spec.block_size
            for token_idx in range(seq_len)
        ],
        dtype=torch.int64,
    )

    torch.manual_seed(0)
    k = torch.randn(seq_len, spec.num_kv_heads, spec.head_size, dtype=torch.float32)
    v = torch.randn_like(k)
    q = torch.randn(4, spec.head_size, dtype=torch.float32)
    scale = spec.head_size**-0.5

    paged_kv_write_f32(ctx, layout, cache, 0, k, v, slot_mapping)
    out = paged_attn_decode_f32(ctx, layout, cache, 0, q, block_table, seq_len, scale)

    expected_rows = []
    gqa_ratio = q.shape[0] // spec.num_kv_heads
    for q_head in range(q.shape[0]):
        kv_head = q_head // gqa_ratio
        scores = (k[:, kv_head, :] * q[q_head]).sum(dim=-1) * scale
        weights = torch.softmax(scores, dim=-1)
        expected_rows.append((weights[:, None] * v[:, kv_head, :]).sum(dim=0))
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_paged_attn_decode_f16_matches_torch_reference():
    ctx = _require_vulkan_context()
    if "paged_kv_write_f16" not in ctx.available_shaders():
        pytest.skip("paged_kv_write_f16 shader is unavailable")
    if "paged_attn_decode_f16" not in ctx.available_shaders():
        pytest.skip("paged_attn_decode_f16 shader is unavailable")

    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=2,
        head_size=8,
        block_size=4,
        dtype_size=2,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=4)
    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))

    seq_len = 6
    block_table = torch.tensor([2, 0], dtype=torch.int64)
    slot_mapping = torch.tensor(
        [
            int(block_table[token_idx // spec.block_size]) * spec.block_size
            + token_idx % spec.block_size
            for token_idx in range(seq_len)
        ],
        dtype=torch.int64,
    )

    torch.manual_seed(1)
    k = torch.randn(seq_len, spec.num_kv_heads, spec.head_size, dtype=torch.float16)
    v = torch.randn_like(k)
    q = torch.randn(4, spec.head_size, dtype=torch.float32)
    scale = spec.head_size**-0.5

    paged_kv_write_f16(ctx, layout, cache, 0, k, v, slot_mapping)
    out = paged_attn_decode_f16(ctx, layout, cache, 0, q, block_table, seq_len, scale)

    k_ref = k.float()
    v_ref = v.float()
    expected_rows = []
    gqa_ratio = q.shape[0] // spec.num_kv_heads
    for q_head in range(q.shape[0]):
        kv_head = q_head // gqa_ratio
        scores = (k_ref[:, kv_head, :] * q[q_head]).sum(dim=-1) * scale
        weights = torch.softmax(scores, dim=-1)
        expected_rows.append((weights[:, None] * v_ref[:, kv_head, :]).sum(dim=0))
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(out, expected, rtol=5e-3, atol=5e-3)


def test_paged_attn_decode_f32_validates_block_table():
    ctx = _require_vulkan_context()
    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=1,
        head_size=4,
        block_size=4,
        dtype_size=4,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=1)
    cache = ctx.alloc_activation(layout.total_bytes)
    q = torch.zeros((1, spec.head_size), dtype=torch.float32)

    with pytest.raises(ValueError, match="expected at least"):
        paged_attn_decode_f32(ctx, layout, cache, 0, q, [], seq_len=1)

    with pytest.raises(ValueError, match="outside"):
        paged_attn_decode_f32(ctx, layout, cache, 0, q, [layout.num_blocks], seq_len=1)
