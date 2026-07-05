# SPDX-License-Identifier: Apache-2.0
"""Tests for the Vulkan attention backend shim."""

from __future__ import annotations

import time
from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm", reason="vllm not installed; skipping attention tests")
_rs = pytest.importorskip("vllm_vulkan._rs", exc_type=ImportError)

from vllm.v1.attention.backend import AttentionType  # noqa: E402
from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata  # noqa: E402

import vllm_vulkan.attention as attention_mod  # noqa: E402
from vllm_vulkan.attention import VulkanAttentionBackendImpl  # noqa: E402
from vllm_vulkan.kv_layout import KVCacheLayerSpec, VulkanPagedKVLayout  # noqa: E402


def _require_vulkan_context():
    if not _rs.is_available():
        pytest.skip("no Vulkan device available")
    try:
        return _rs.VulkanContext(0)
    except RuntimeError as exc:
        pytest.skip(f"VulkanContext unavailable: {exc}")


def test_vulkan_attention_backend_uses_paged_decode_for_single_token_batch(monkeypatch):
    _require_vulkan_context()
    logged_messages = []
    monkeypatch.setattr(
        attention_mod.logger,
        "debug",
        lambda msg, *args, **kwargs: logged_messages.append(msg % args),
    )

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
    assert any("Vulkan attention decode used" in msg for msg in logged_messages)

    torch.testing.assert_close(output, expected, rtol=5e-3, atol=5e-3)


def test_vulkan_attention_backend_mirrors_prefill_kv_before_decode():
    _require_vulkan_context()

    num_heads = 4
    num_kv_heads = 2
    head_size = 32
    block_size = 16
    num_blocks = 2
    seq_len = 3
    scale = head_size**-0.5

    torch.manual_seed(0)
    query = torch.randn(1, num_heads, head_size, dtype=torch.float32)
    prefill_k = torch.randn(seq_len - 1, num_kv_heads, head_size, dtype=torch.float16)
    prefill_v = torch.randn_like(prefill_k)
    decode_k = torch.randn(1, num_kv_heads, head_size, dtype=torch.float16)
    decode_v = torch.randn_like(decode_k)

    kv_cache = torch.zeros(
        (2, num_blocks, num_kv_heads, block_size, head_size),
        dtype=torch.float16,
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

    prefill_metadata = CPUAttentionMetadata(
        isa="vec16",
        num_actual_tokens=seq_len - 1,
        max_query_len=seq_len - 1,
        query_start_loc=torch.tensor([0, seq_len - 1], dtype=torch.int32),
        max_seq_len=seq_len - 1,
        seq_lens=torch.tensor([seq_len - 1], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.arange(seq_len - 1, dtype=torch.int64),
        scheduler_metadata=None,
        causal=True,
        use_sdpa_prefill=True,
        num_decode_tokens=0,
        sdpa_start_loc=torch.tensor([0, seq_len - 1], dtype=torch.int32),
    )
    impl.forward(
        layer=None,  # type: ignore[arg-type]
        query=torch.randn(seq_len - 1, num_heads, head_size, dtype=torch.float16),
        key=prefill_k,
        value=prefill_v,
        kv_cache=kv_cache,
        attn_metadata=prefill_metadata,
        output=torch.empty(seq_len - 1, num_heads, head_size, dtype=torch.float16),
    )
    assert impl._last_vulkan_decode_used is False

    output = torch.empty((1, num_heads, head_size), dtype=torch.float32)
    metadata = CPUAttentionMetadata(
        isa="vec16",
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=seq_len,
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([seq_len - 1], dtype=torch.int64),
        scheduler_metadata=None,
        causal=True,
    )
    result = impl.forward(
        layer=None,  # type: ignore[arg-type]
        query=query,
        key=decode_k,
        value=decode_v,
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )

    k_ref = torch.cat([prefill_k, decode_k], dim=0).float()
    v_ref = torch.cat([prefill_v, decode_v], dim=0).float()
    expected_rows = []
    gqa_ratio = num_heads // num_kv_heads
    for q_head in range(num_heads):
        kv_head = q_head // gqa_ratio
        scores = (k_ref[:, kv_head, :] * query[0, q_head]).sum(dim=-1) * scale
        weights = torch.softmax(scores, dim=-1)
        expected_rows.append((weights[:, None] * v_ref[:, kv_head, :]).sum(dim=0))
    expected = torch.stack(expected_rows).unsqueeze(0)

    assert impl._last_vulkan_decode_used is True
    assert result is output
    torch.testing.assert_close(output, expected, rtol=5e-3, atol=5e-3)


def test_vulkan_attention_backend_rejects_unsupported_decode_features():
    num_heads = 2
    num_kv_heads = 1
    head_size = 8
    block_size = 4

    impl = VulkanAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=block_size,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )
    metadata = CPUAttentionMetadata(
        isa="vec16",
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        scheduler_metadata=None,
        causal=True,
    )

    assert not impl._supports_vulkan_decode(
        key=torch.zeros(1, num_kv_heads, head_size, dtype=torch.float16),
        value=torch.zeros(1, num_kv_heads, head_size, dtype=torch.float16),
        kv_cache=torch.zeros(
            (2, 1, num_kv_heads, block_size, head_size),
            dtype=torch.float16,
        ),
        attn_metadata=metadata,
        output=torch.empty(1, num_heads, head_size),
        num_actual_tokens=1,
    )


# ---------------------------------------------------------------------------
# _vulkan_cache_has_sequences / _mark_vulkan_cache_slots_written / _active_block_ids
#
# These used to convert every tensor element (per request, per block, per
# written slot) to a Python int one at a time via `int(tensor_scalar)` -
# individually cheap, but this runs once per attention layer per decode
# step, once per concurrent request in the batch, so the per-element
# tensor->Python conversion overhead scaled with (num_layers x batch_size)
# on every single call. Vectorized via `.tolist()` (one C-level call per
# tensor instead of N per-element conversions) - see attention.py's
# _vulkan_cache_has_sequences doc comment for the measured speedup.
# ---------------------------------------------------------------------------


def _reference_active_block_ids(row, needed_blocks, num_blocks):
    """Verbatim copy of the original tensor-indexing implementation, kept
    as an independent reference (not derived from the new list-based one)."""
    if row.numel() < needed_blocks:
        return None
    block_ids = tuple(int(block_id) for block_id in row[:needed_blocks])
    if any(block_id < 0 or block_id >= num_blocks for block_id in block_ids):
        return None
    return block_ids


def _reference_vulkan_cache_has_sequences(entry, block_table, seq_lens):
    """Verbatim copy of the original per-element implementation."""
    layout = entry.layout
    spec = layout.layer_spec(0)
    for req_idx, seq_len_tensor in enumerate(seq_lens):
        seq_len = int(seq_len_tensor)
        row = block_table[req_idx]
        needed_blocks = (seq_len + spec.block_size - 1) // spec.block_size
        active_blocks = _reference_active_block_ids(
            row, needed_blocks, layout.num_blocks
        )
        if active_blocks is None:
            return False
        start_pos = min(entry.verified_prefix_by_blocks.get(active_blocks, 0), seq_len)
        for token_pos in range(start_pos, seq_len):
            logical_block_id = token_pos // spec.block_size
            token_offset = token_pos % spec.block_size
            physical_block_id = active_blocks[logical_block_id]
            slot = physical_block_id * spec.block_size + token_offset
            if not entry.written_slots[slot]:
                return False
        entry.verified_prefix_by_blocks[active_blocks] = seq_len
    return True


def _reference_mark_vulkan_cache_slots_written(entry, slots):
    """Verbatim copy of the original per-element implementation."""
    capacity = len(entry.written_slots)
    for slot_tensor in slots:
        slot = int(slot_tensor)
        if 0 <= slot < capacity:
            entry.written_slots[slot] = 1


def _make_entry(num_blocks: int, block_size: int, capacity_blocks: int):
    spec = KVCacheLayerSpec(
        layer_index=0, num_kv_heads=1, head_size=8, block_size=block_size, dtype_size=4
    )
    layout = VulkanPagedKVLayout((spec,), num_blocks=capacity_blocks)
    return attention_mod._VulkanKVCacheEntry(
        storage_key=0,
        layout=layout,
        cache=None,
        shape=(),
        dtype=torch.float32,
        written_slots=bytearray(layout.capacity_tokens_per_layer),
        verified_prefix_by_blocks=OrderedDict(),
    )


def test_active_block_ids_matches_reference():
    row = [5, 2, 9, 1]
    assert attention_mod._active_block_ids(row, 3, num_blocks=100) == (5, 2, 9)
    # out of range block id -> None
    assert attention_mod._active_block_ids(row, 4, num_blocks=5) is None
    # not enough blocks -> None
    assert attention_mod._active_block_ids(row, 5, num_blocks=100) is None

    row_tensor = torch.tensor(row, dtype=torch.int64)
    assert attention_mod._active_block_ids(
        row_tensor.tolist(), 3, num_blocks=100
    ) == _reference_active_block_ids(row_tensor, 3, num_blocks=100)


def test_vulkan_cache_has_sequences_matches_reference():
    block_size = 4
    num_reqs = 6
    seq_lens_list = [1, 4, 5, 0, 8, 3]

    entry_new = _make_entry(num_blocks=200, block_size=block_size, capacity_blocks=200)
    entry_ref = _make_entry(num_blocks=200, block_size=block_size, capacity_blocks=200)

    block_table = torch.stack(
        [torch.arange(r * 10, r * 10 + 8, dtype=torch.int64) for r in range(num_reqs)]
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64)

    # Mark every slot these sequences could touch as already written, so
    # both the reference and the new implementation take the same
    # "everything verified" path deterministically.
    for entry in (entry_new, entry_ref):
        for i in range(len(entry.written_slots)):
            entry.written_slots[i] = 1

    new_result = attention_mod._vulkan_cache_has_sequences(
        entry_new, block_table, seq_lens
    )
    ref_result = _reference_vulkan_cache_has_sequences(entry_ref, block_table, seq_lens)

    assert new_result == ref_result is True
    assert dict(entry_new.verified_prefix_by_blocks) == dict(
        entry_ref.verified_prefix_by_blocks
    )


def test_vulkan_cache_has_sequences_detects_unwritten_slot():
    block_size = 4
    entry = _make_entry(num_blocks=200, block_size=block_size, capacity_blocks=200)
    block_table = torch.tensor([[0, 1]], dtype=torch.int64)
    seq_lens = torch.tensor([5], dtype=torch.int64)
    # written_slots left all-zero -> should report False (not fully written).
    assert not attention_mod._vulkan_cache_has_sequences(entry, block_table, seq_lens)


def test_mark_vulkan_cache_slots_written_matches_reference():
    entry_new = _make_entry(num_blocks=50, block_size=4, capacity_blocks=50)
    entry_ref = _make_entry(num_blocks=50, block_size=4, capacity_blocks=50)
    slots = torch.tensor([0, 5, 10, 199, -1, 10_000], dtype=torch.int64)

    attention_mod._mark_vulkan_cache_slots_written(entry_new, slots)
    _reference_mark_vulkan_cache_slots_written(entry_ref, slots)

    assert entry_new.written_slots == entry_ref.written_slots


def test_vulkan_cache_has_sequences_is_faster_at_a_realistic_batch_size():
    block_size = 16
    num_reqs = 64
    seq_len = 512
    blocks_per_req = (seq_len + block_size - 1) // block_size

    entry_new = _make_entry(
        num_blocks=8192, block_size=block_size, capacity_blocks=8192
    )
    entry_ref = _make_entry(
        num_blocks=8192, block_size=block_size, capacity_blocks=8192
    )
    for entry in (entry_new, entry_ref):
        for i in range(len(entry.written_slots)):
            entry.written_slots[i] = 1

    block_table = torch.stack(
        [
            torch.arange(r * 100, r * 100 + blocks_per_req, dtype=torch.int64)
            for r in range(num_reqs)
        ]
    )
    seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int64)

    # Prime steady state (verified-prefix cache populated) for both.
    attention_mod._vulkan_cache_has_sequences(entry_new, block_table, seq_lens)
    _reference_vulkan_cache_has_sequences(entry_ref, block_table, seq_lens)

    def time_best_of(trials, iters, fn):
        best = float("inf")
        for _ in range(trials):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            best = min(best, (time.perf_counter() - t0) / iters)
        return best

    old_s = time_best_of(
        5,
        50,
        lambda: _reference_vulkan_cache_has_sequences(entry_ref, block_table, seq_lens),
    )
    new_s = time_best_of(
        5,
        50,
        lambda: attention_mod._vulkan_cache_has_sequences(
            entry_new, block_table, seq_lens
        ),
    )

    print(
        f"\n_vulkan_cache_has_sequences ({num_reqs} reqs, steady state): "
        f"old {old_s * 1e6:.1f} us/call   new {new_s * 1e6:.1f} us/call   "
        f"speedup {old_s / new_s:.2f}x"
    )
    assert new_s < old_s, (
        f"new implementation ({new_s * 1e6:.1f}us) was not faster than the "
        f"reference ({old_s * 1e6:.1f}us)"
    )
