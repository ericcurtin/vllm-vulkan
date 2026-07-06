# SPDX-License-Identifier: Apache-2.0
"""GPU correctness tests for Vulkan paged KV-cache operations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_rs = pytest.importorskip("vllm_vulkan._rs", exc_type=ImportError)

from vllm_vulkan.kv_layout import KVCacheLayerSpec, VulkanPagedKVLayout  # noqa: E402
from vllm_vulkan.kv_ops import (  # noqa: E402
    _select_decode_shader,
    paged_attn_decode_batch_f16,
    paged_attn_decode_batch_f32,
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


def _require_shader(ctx, *shader_names: str) -> None:
    available = set(ctx.available_shaders())
    if not any(name in available for name in shader_names):
        pytest.skip(f"none of {shader_names!r} are available")


def _make_layout(dtype_size: int) -> tuple[KVCacheLayerSpec, VulkanPagedKVLayout]:
    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=2,
        head_size=8,
        block_size=4,
        dtype_size=dtype_size,
    )
    return spec, VulkanPagedKVLayout((spec,), num_blocks=4)


def _slot_mapping_from_block_table(
    block_table: torch.Tensor, seq_len: int, block_size: int
) -> torch.Tensor:
    slots = [
        int(block_table[token_idx // block_size]) * block_size + token_idx % block_size
        for token_idx in range(seq_len)
    ]
    return torch.tensor(slots, dtype=torch.int64)


def _attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    k_ref = k.float()
    v_ref = v.float()
    gqa_ratio = q.shape[0] // k_ref.shape[1]
    rows = []
    for q_head in range(q.shape[0]):
        kv_head = q_head // gqa_ratio
        scores = (k_ref[:, kv_head, :] * q[q_head]).sum(dim=-1) * scale
        weights = torch.softmax(scores, dim=-1)
        rows.append((weights[:, None] * v_ref[:, kv_head, :]).sum(dim=0))
    return torch.stack(rows)


@pytest.mark.parametrize(
    ("torch_dtype", "numpy_dtype", "dtype_size", "write_fn", "shader_name"),
    [
        (torch.float32, np.float32, 4, paged_kv_write_f32, "paged_kv_write_f32"),
        (torch.float16, np.float16, 2, paged_kv_write_f16, "paged_kv_write_f16"),
    ],
)
def test_paged_kv_write_round_trips_gpu_cache_slots(
    torch_dtype: torch.dtype,
    numpy_dtype: np.dtype,
    dtype_size: int,
    write_fn: Callable,
    shader_name: str,
):
    ctx = _require_vulkan_context()
    _require_shader(ctx, shader_name)

    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=2,
        head_size=4,
        block_size=4,
        dtype_size=dtype_size,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=3)
    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))

    k = torch.arange(
        3 * spec.num_kv_heads * spec.head_size,
        dtype=torch_dtype,
    ).reshape(3, spec.num_kv_heads, spec.head_size)
    v = k + 1000
    slot_mapping = torch.tensor([0, 5, 10], dtype=torch.int64)

    write_fn(ctx, layout, cache, 0, k, v, slot_mapping)

    cache_view = np.frombuffer(ctx.read_activation(cache), dtype=numpy_dtype)
    for token_index, slot in enumerate(slot_mapping.tolist()):
        for kv_head in range(spec.num_kv_heads):
            for head_element in range(spec.head_size):
                k_offset = (
                    layout.slot_offset(0, slot, kv_head, head_element, "k")
                    // dtype_size
                )
                v_offset = (
                    layout.slot_offset(0, slot, kv_head, head_element, "v")
                    // dtype_size
                )

                assert cache_view[k_offset] == pytest.approx(
                    float(k[token_index, kv_head, head_element])
                )
                assert cache_view[v_offset] == pytest.approx(
                    float(v[token_index, kv_head, head_element])
                )

    untouched_slot = 1
    assert (
        cache_view[layout.slot_offset(0, untouched_slot, 0, 0, "k") // dtype_size]
        == 0.0
    )
    assert (
        cache_view[layout.slot_offset(0, untouched_slot, 0, 0, "v") // dtype_size]
        == 0.0
    )


def test_paged_kv_write_validates_slots_and_empty_inputs():
    ctx = _require_vulkan_context()
    _require_shader(ctx, "paged_kv_write_f32")

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
        paged_kv_write_f32(
            ctx, layout, cache, 0, k, v, [layout.capacity_tokens_per_layer]
        )

    empty_k = torch.zeros((0, spec.num_kv_heads, spec.head_size), dtype=torch.float32)
    with pytest.raises(ValueError, match="at least one token"):
        paged_kv_write_f32(
            ctx, layout, cache, 0, empty_k, torch.zeros_like(empty_k), []
        )


@pytest.mark.parametrize(
    (
        "torch_dtype",
        "dtype_size",
        "write_fn",
        "decode_fn",
        "write_shader",
        "decode_shaders",
        "seed",
        "rtol",
        "atol",
    ),
    [
        (
            torch.float32,
            4,
            paged_kv_write_f32,
            paged_attn_decode_f32,
            "paged_kv_write_f32",
            ("paged_attn_decode_f32", "paged_attn_decode_f32_coop"),
            0,
            1e-4,
            1e-4,
        ),
        (
            torch.float16,
            2,
            paged_kv_write_f16,
            paged_attn_decode_f16,
            "paged_kv_write_f16",
            ("paged_attn_decode_f16", "paged_attn_decode_f16_coop"),
            1,
            5e-3,
            5e-3,
        ),
    ],
)
def test_paged_attn_decode_matches_torch_reference(
    torch_dtype: torch.dtype,
    dtype_size: int,
    write_fn: Callable,
    decode_fn: Callable,
    write_shader: str,
    decode_shaders: tuple[str, ...],
    seed: int,
    rtol: float,
    atol: float,
):
    ctx = _require_vulkan_context()
    _require_shader(ctx, write_shader)
    _require_shader(ctx, *decode_shaders)

    spec, layout = _make_layout(dtype_size)
    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))

    seq_len = 6
    block_table = torch.tensor([2, 0], dtype=torch.int64)
    slot_mapping = _slot_mapping_from_block_table(block_table, seq_len, spec.block_size)

    torch.manual_seed(seed)
    k = torch.randn(seq_len, spec.num_kv_heads, spec.head_size, dtype=torch_dtype)
    v = torch.randn_like(k)
    q = torch.randn(4, spec.head_size, dtype=torch.float32)
    scale = spec.head_size**-0.5

    write_fn(ctx, layout, cache, 0, k, v, slot_mapping)
    out = decode_fn(ctx, layout, cache, 0, q, block_table, seq_len, scale)

    expected = _attention_reference(q, k, v, scale)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def test_paged_attn_decode_validates_block_table():
    ctx = _require_vulkan_context()
    _require_shader(ctx, "paged_attn_decode_f32", "paged_attn_decode_f32_coop")

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


@pytest.mark.parametrize(
    (
        "torch_dtype",
        "dtype_size",
        "write_fn",
        "decode_fn",
        "decode_batch_fn",
        "write_shader",
        "decode_shaders",
        "seed",
        "rtol",
        "atol",
    ),
    [
        (
            torch.float32,
            4,
            paged_kv_write_f32,
            paged_attn_decode_f32,
            paged_attn_decode_batch_f32,
            "paged_kv_write_f32",
            ("paged_attn_decode_f32", "paged_attn_decode_f32_coop"),
            0,
            1e-4,
            1e-4,
        ),
        (
            torch.float16,
            2,
            paged_kv_write_f16,
            paged_attn_decode_f16,
            paged_attn_decode_batch_f16,
            "paged_kv_write_f16",
            ("paged_attn_decode_f16", "paged_attn_decode_f16_coop"),
            1,
            5e-3,
            5e-3,
        ),
    ],
)
def test_paged_attn_decode_batch_matches_per_token_calls_and_torch_reference(
    torch_dtype: torch.dtype,
    dtype_size: int,
    write_fn: Callable,
    decode_fn: Callable,
    decode_batch_fn: Callable,
    write_shader: str,
    decode_shaders: tuple[str, ...],
    seed: int,
    rtol: float,
    atol: float,
):
    """paged_attn_decode_batch_{f16,f32} - one vkQueueSubmit for the whole
    batch instead of one per token (see attention.py's _try_vulkan_decode)
    - must produce exactly the same results as calling the single-token
    paged_attn_decode_{f16,f32} once per (query, block_table, seq_len)
    triple, for a batch of *independent* sequences (different K/V data,
    different block tables, different sequence lengths) sharing one cache.
    """
    ctx = _require_vulkan_context()
    _require_shader(ctx, write_shader)
    _require_shader(ctx, *decode_shaders)

    spec, layout = _make_layout(dtype_size)
    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))

    torch.manual_seed(seed)

    # Two independent "requests" in the batch: different seq_lens, block
    # tables, and K/V/Q data, writing into disjoint block ranges of the
    # same shared cache - exactly like multiple concurrent decode requests.
    seq_lens = [6, 3]
    block_tables = [
        torch.tensor([2, 0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    ]
    scale = spec.head_size**-0.5

    ks, vs, qs = [], [], []
    for seq_len, block_table in zip(seq_lens, block_tables, strict=True):
        k = torch.randn(seq_len, spec.num_kv_heads, spec.head_size, dtype=torch_dtype)
        v = torch.randn_like(k)
        q = torch.randn(4, spec.head_size, dtype=torch.float32)
        slot_mapping = _slot_mapping_from_block_table(
            block_table, seq_len, spec.block_size
        )
        write_fn(ctx, layout, cache, 0, k, v, slot_mapping)
        ks.append(k)
        vs.append(v)
        qs.append(q)

    # Reference: per-token single calls (the old behaviour).
    per_token_outs = [
        decode_fn(ctx, layout, cache, 0, qs[i], block_tables[i], seq_lens[i], scale)
        for i in range(len(seq_lens))
    ]

    # Under test: one batched call.
    batch_outs = decode_batch_fn(
        ctx, layout, cache, 0, qs, block_tables, seq_lens, scale
    )

    assert len(batch_outs) == len(per_token_outs)
    for i, (batch_out, per_token_out) in enumerate(
        zip(batch_outs, per_token_outs, strict=True)
    ):
        torch.testing.assert_close(
            batch_out,
            per_token_out,
            rtol=rtol,
            atol=atol,
            msg=f"batch output {i} diverged from the per-token call",
        )

    # Also cross-check the batch outputs directly against the pure-PyTorch
    # attention reference (not just "agrees with the other Vulkan path"),
    # for the same end-to-end correctness guarantee the per-token test above
    # already has.
    for i in range(len(seq_lens)):
        expected = _attention_reference(qs[i], ks[i], vs[i], scale)
        torch.testing.assert_close(batch_outs[i], expected, rtol=rtol, atol=atol)


def test_paged_attn_decode_batch_validates_mismatched_list_lengths():
    ctx = _require_vulkan_context()
    _require_shader(ctx, "paged_attn_decode_f32", "paged_attn_decode_f32_coop")

    spec, layout = _make_layout(dtype_size=4)
    cache = ctx.alloc_activation(layout.total_bytes)
    q = torch.zeros((1, spec.head_size), dtype=torch.float32)

    with pytest.raises(ValueError, match="same length"):
        paged_attn_decode_batch_f32(
            ctx,
            layout,
            cache,
            0,
            queries=[q, q],
            block_tables=[[0]],
            seq_lens=[1, 1],
        )


def test_paged_attn_decode_batch_empty_returns_empty_list():
    ctx = _require_vulkan_context()
    _require_shader(ctx, "paged_attn_decode_f32", "paged_attn_decode_f32_coop")

    spec, layout = _make_layout(dtype_size=4)
    cache = ctx.alloc_activation(layout.total_bytes)

    assert (
        paged_attn_decode_batch_f32(
            ctx, layout, cache, 0, queries=[], block_tables=[], seq_lens=[]
        )
        == []
    )


@pytest.mark.parametrize(
    (
        "torch_dtype",
        "dtype_size",
        "write_fn",
        "decode_fn",
        "write_shader",
        "decode_shaders",
        "seed",
        "rtol",
        "atol",
    ),
    [
        (
            torch.float32,
            4,
            paged_kv_write_f32,
            paged_attn_decode_f32,
            "paged_kv_write_f32",
            ("paged_attn_decode_f32", "paged_attn_decode_f32_coop_512"),
            2,
            1e-4,
            1e-4,
        ),
        (
            torch.float16,
            2,
            paged_kv_write_f16,
            paged_attn_decode_f16,
            "paged_kv_write_f16",
            ("paged_attn_decode_f16", "paged_attn_decode_f16_coop_512"),
            3,
            5e-3,
            5e-3,
        ),
    ],
)
def test_paged_attn_decode_matches_torch_reference_at_head_size_512(
    torch_dtype: torch.dtype,
    dtype_size: int,
    write_fn: Callable,
    decode_fn: Callable,
    write_shader: str,
    decode_shaders: tuple[str, ...],
    seed: int,
    rtol: float,
    atol: float,
):
    """Exercises the BLOCK_SIZE=512 `_coop_512` shader variant specifically
    (head_size=512 matches Gemma4-E2B's full-attention layers, and
    _select_decode_shader prefers the 512-wide variant at this size — see
    test_select_decode_shader_prefers_block_size_matching_head_size below)
    against the same pure-PyTorch attention reference used at the smaller
    default head_size=8 shape above, to confirm the 512-wide cooperative
    reduction is numerically correct, not just "faster in isolation".
    """
    ctx = _require_vulkan_context()
    _require_shader(ctx, write_shader)
    _require_shader(ctx, *decode_shaders)

    spec = KVCacheLayerSpec(
        layer_index=0,
        num_kv_heads=1,
        head_size=512,
        block_size=16,
        dtype_size=dtype_size,
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=8)
    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))

    seq_len = 40
    block_table = torch.tensor([3, 5, 0], dtype=torch.int64)
    slot_mapping = _slot_mapping_from_block_table(block_table, seq_len, spec.block_size)

    torch.manual_seed(seed)
    k = torch.randn(seq_len, spec.num_kv_heads, spec.head_size, dtype=torch_dtype)
    v = torch.randn_like(k)
    q = torch.randn(8, spec.head_size, dtype=torch.float32)  # 8 q_heads, GQA ratio 8
    scale = spec.head_size**-0.5

    write_fn(ctx, layout, cache, 0, k, v, slot_mapping)
    out = decode_fn(ctx, layout, cache, 0, q, block_table, seq_len, scale)

    expected = _attention_reference(q, k, v, scale)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def test_select_decode_shader_prefers_block_size_matching_head_size():
    """_select_decode_shader should prefer the `_coop_512` variant when
    head_size>=512 (Gemma4-E2B's full-attention layers) and the plain
    `_coop` (BLOCK_SIZE=256) variant otherwise (e.g. head_size=256, its
    sliding-window layers) — matching BLOCK_SIZE to head_size avoids
    wasted/idle threads in the cooperative dot-product reduction (see
    paged_attn_decode_f32_coop.comp's BLOCK_SIZE comment for the measured
    rationale). Both are real, always-available shaders in this codebase
    (not a hypothetical), so this also implicitly confirms both compiled
    successfully.
    """
    ctx = _require_vulkan_context()
    _require_shader(ctx, "paged_attn_decode_f32_coop", "paged_attn_decode_f32_coop_512")

    name_256, wg_256 = _select_decode_shader(
        ctx,
        "paged_attn_decode_f32",
        "paged_attn_decode_f32_coop",
        "paged_attn_decode_f32_coop_512",
        head_size=256,
    )
    assert name_256 == "paged_attn_decode_f32_coop"
    assert wg_256 == 256

    name_512, wg_512 = _select_decode_shader(
        ctx,
        "paged_attn_decode_f32",
        "paged_attn_decode_f32_coop",
        "paged_attn_decode_f32_coop_512",
        head_size=512,
    )
    assert name_512 == "paged_attn_decode_f32_coop_512"
    assert wg_512 == 512
