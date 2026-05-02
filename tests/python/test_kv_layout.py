# SPDX-License-Identifier: Apache-2.0
"""Tests for the Vulkan paged KV-cache layout contract."""

from __future__ import annotations

import pytest

from vllm_vulkan.kv_layout import (
    KVCacheLayerSpec,
    VulkanPagedKVLayout,
    dtype_size_bytes,
    infer_kv_layer_specs,
    layout_from_hf_config,
)


class _TextConfig:
    num_hidden_layers = 3
    num_attention_heads = 8
    num_key_value_heads = 2
    head_dim = 64
    layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
    global_head_dim = 128
    num_global_key_value_heads = 1


class _WrappedConfig:
    text_config = _TextConfig()


def test_dtype_size_bytes_accepts_common_names():
    assert dtype_size_bytes("bfloat16") == 2
    assert dtype_size_bytes("torch.float16") == 2
    assert dtype_size_bytes("float32") == 4
    assert dtype_size_bytes(1) == 1

    with pytest.raises(ValueError, match="Unsupported"):
        dtype_size_bytes("not-a-dtype")


def test_paged_kv_layout_offsets_are_block_token_head_major():
    specs = tuple(
        KVCacheLayerSpec(
            layer_index=i,
            num_kv_heads=2,
            head_size=4,
            block_size=16,
            dtype_size=2,
        )
        for i in range(2)
    )
    layout = VulkanPagedKVLayout(specs, num_blocks=3)

    assert layout.block_size == 16
    assert layout.capacity_tokens_per_layer == 48
    assert layout.bytes_per_token == 64
    assert layout.total_bytes == 2 * 3 * 512

    assert layout.layer_base_offset(0) == 0
    assert layout.layer_base_offset(1) == 3 * 512

    block_base = layout.block_base_offset(layer_index=1, physical_block_id=2)
    assert block_base == 3 * 512 + 2 * 512
    assert layout.plane_base_offset(1, 2, "k") == block_base
    assert layout.plane_base_offset(1, 2, "v") == block_base + 256

    # token 5, KV head 1, element 3 in V:
    # scalar index = ((5 * 2 + 1) * 4 + 3) = 47
    assert layout.token_offset(1, 2, 5, 1, 3, "v") == block_base + 256 + 47 * 2

    slot = 2 * layout.block_size + 5
    assert layout.slot_offset(1, slot, 1, 3, "v") == layout.token_offset(
        1, 2, 5, 1, 3, "v"
    )


def test_paged_kv_layout_exposes_attention_load_strides():
    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=2,
        head_size=4,
        block_size=16,
        dtype_size=4,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=2)

    k_strides = layout.k_plane_strides_for_attn_load(0, 1)

    assert k_strides.base_offset == spec.bytes_per_block
    assert k_strides.token_stride == spec.bytes_per_token_per_plane
    assert k_strides.kv_head_stride == spec.head_size * spec.dtype_size
    assert k_strides.head_element_stride == spec.dtype_size

    v_strides = layout.v_plane_strides_for_attn_load(0, 1)

    assert v_strides.base_offset == spec.bytes_per_block + spec.plane_bytes_per_block
    assert v_strides.token_stride == k_strides.token_stride
    assert v_strides.kv_head_stride == k_strides.kv_head_stride
    assert v_strides.head_element_stride == k_strides.head_element_stride


def test_layout_rejects_invalid_indices():
    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=2,
        head_size=4,
        block_size=16,
        dtype_size=2,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=1)

    with pytest.raises(ValueError, match="physical_block_id"):
        layout.block_base_offset(0, 1)

    with pytest.raises(ValueError, match="token_offset_in_block"):
        layout.token_offset(0, 0, 16, 0, 0, "k")

    with pytest.raises(ValueError, match="kv_head"):
        layout.token_offset(0, 0, 0, 2, 0, "k")

    with pytest.raises(ValueError, match="head_element"):
        layout.token_offset(0, 0, 0, 0, 4, "k")


def test_infer_kv_layer_specs_supports_heterogeneous_attention_layers():
    specs = infer_kv_layer_specs(
        _WrappedConfig(),
        block_size=16,
        dtype="bfloat16",
    )

    assert len(specs) == 3
    assert specs[0].num_kv_heads == 2
    assert specs[0].head_size == 64
    assert specs[1].num_kv_heads == 1
    assert specs[1].head_size == 128
    assert specs[2].num_kv_heads == 2
    assert specs[2].head_size == 64
    assert all(spec.dtype_size == 2 for spec in specs)


def test_layout_from_hf_config_computes_total_bytes():
    layout = layout_from_hf_config(
        _WrappedConfig(),
        block_size=16,
        num_blocks=4,
        dtype="float16",
    )

    # All three specs are 2 * 16 * 128 * 2 = 8192 bytes/block.
    assert layout.total_bytes == 3 * 4 * 8192
    assert layout.bytes_per_token == 3 * 2 * 128 * 2
