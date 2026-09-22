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
from vllm.v1.attention.backends.cpu_attn import (  # noqa: E402
    CPUAttentionBackendImpl,
    CPUAttentionMetadata,
)

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


@pytest.fixture
def cpu_ops_stubbed(monkeypatch):
    """Stub out the two `CPUAttentionBackendImpl` methods that call into
    vLLM's C++ CPU attention ops, leaving the Vulkan path fully real.

    vLLM 0.29 writes the KV cache through `do_kv_cache_update`
    (`unified_kv_cache_update`), separately from `forward`. The plugin's
    override writes vLLM's CPU cache via `super()` and then mirrors the same
    tokens into the Vulkan cache; `forward` falls back to `super()` whenever
    Vulkan decode isn't applicable. Both `super()` halves are
    `ops.cpu_attn_reshape_and_cache` / `ops.cpu_attention_with_kv_cache`,
    which exist only in a CPU-built vLLM -- the PyPI wheel is the CUDA build
    and its `_C` has neither -- so stubbing them is what lets these tests run
    outside a CPU build at all.

    What they assert is untouched by this: the Vulkan mirror, the Vulkan
    decode, `_last_vulkan_decode_used`, and the comparison against a torch
    reference are all still real. A test whose Vulkan decode silently stopped
    working would fall through to the stub and fail on both the flag and the
    output.
    """
    monkeypatch.setattr(
        CPUAttentionBackendImpl, "do_kv_cache_update", lambda *a, **k: None
    )
    monkeypatch.setattr(
        CPUAttentionBackendImpl,
        "forward",
        lambda self, layer, query, key, value, kv_cache, attn_metadata, output, *a: (
            output
        ),
    )


def _write_kv(impl, key, value, kv_cache, slot_mapping):
    """What vLLM's `unified_kv_cache_update` op does before each forward."""
    impl.do_kv_cache_update(
        None,  # layer: only reached by the stubbed-out CPU write
        key,
        value,
        kv_cache,
        slot_mapping,
    )


def test_vulkan_attention_backend_uses_paged_decode_for_single_token_batch(
    monkeypatch, cpu_ops_stubbed
):
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
        (num_blocks, num_kv_heads, block_size, 2 * head_size),
        dtype=torch.float16,
    )
    output = torch.empty((num_reqs, num_heads, head_size), dtype=torch.float32)

    metadata = CPUAttentionMetadata(
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

    _write_kv(impl, key, value, kv_cache, metadata.slot_mapping)
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


def test_vulkan_attention_backend_uses_paged_decode_for_single_token_batch_f32(
    monkeypatch, cpu_ops_stubbed
):
    """Same scenario as
    `test_vulkan_attention_backend_uses_paged_decode_for_single_token_batch`
    above, but with a float32 KV cache — every other test in this file
    uses float16, so this is the only coverage exercising the
    `paged_attn_decode_batch_f32`/`paged_kv_write_f32` branch of
    `_try_vulkan_decode`/`_try_write_tokens_to_vulkan_cache`'s dtype
    selection (now a plain ternary hoisted to module-level imports,
    rather than a conditional local import — see attention.py's
    `_get_vulkan_context` doc comment)."""
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
    key = torch.randn(num_reqs, num_kv_heads, head_size, dtype=torch.float32)
    value = torch.randn(num_reqs, num_kv_heads, head_size, dtype=torch.float32)
    kv_cache = torch.zeros(
        (num_blocks, num_kv_heads, block_size, 2 * head_size),
        dtype=torch.float32,
    )
    output = torch.empty((num_reqs, num_heads, head_size), dtype=torch.float32)

    metadata = CPUAttentionMetadata(
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

    _write_kv(impl, key, value, kv_cache, metadata.slot_mapping)
    result = impl.forward(
        layer=None,  # type: ignore[arg-type]
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )

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


def test_vulkan_attention_backend_mirrors_prefill_kv_before_decode(
    cpu_ops_stubbed,
):
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
        (num_blocks, num_kv_heads, block_size, 2 * head_size),
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
    _write_kv(impl, prefill_k, prefill_v, kv_cache, prefill_metadata.slot_mapping)
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
    _write_kv(impl, decode_k, decode_v, kv_cache, metadata.slot_mapping)
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
            (1, num_kv_heads, block_size, 2 * head_size),
            dtype=torch.float16,
        ),
        attn_metadata=metadata,
        output=torch.empty(1, num_heads, head_size),
        num_actual_tokens=1,
    )


def test_forward_skips_vulkan_decode_for_zero_tokens():
    """num_actual_tokens == 0 (a warmup/profiling call with no real tokens)
    must skip the Vulkan decode attempt entirely rather than doing the
    context lookup, KV-cache-entry resolution and cache scan for nothing:
    `_supports_vulkan_decode`'s query_lens_all_ones check passes trivially
    on an empty batch, so without the explicit guard in `forward` it would
    proceed into all that setup work with nothing to decode.
    """
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
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )
    metadata = CPUAttentionMetadata(
        num_actual_tokens=0,
        max_query_len=1,
        query_start_loc=torch.tensor([0], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([], dtype=torch.int32),
        block_table=torch.zeros((0, 1), dtype=torch.int32),
        slot_mapping=torch.tensor([], dtype=torch.int64),
        scheduler_metadata=None,
        causal=True,
    )
    kv_cache = torch.zeros(
        (1, num_kv_heads, block_size, 2 * head_size), dtype=torch.float16
    )
    output = torch.empty(0, num_heads, head_size, dtype=torch.float32)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Vulkan decode attempted on an empty batch")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(VulkanAttentionBackendImpl, "_try_vulkan_decode", fail_if_called)
        mp.setattr(CPUAttentionBackendImpl, "forward", lambda *a, **k: output)
        impl.forward(
            layer=None,  # type: ignore[arg-type]
            query=torch.zeros(0, num_heads, head_size, dtype=torch.float32),
            key=torch.zeros(0, num_kv_heads, head_size, dtype=torch.float16),
            value=torch.zeros(0, num_kv_heads, head_size, dtype=torch.float16),
            kv_cache=kv_cache,
            attn_metadata=metadata,
            output=output,
        )

    assert impl._last_vulkan_decode_used is False
    # The Vulkan KV-cache-entry cache must never have been populated.
    assert impl._cached_kv_cache_entry is None


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


def _make_decode_metadata(num_reqs: int = 2, seq_len: int = 1) -> CPUAttentionMetadata:
    return CPUAttentionMetadata(
        num_actual_tokens=num_reqs,
        max_query_len=1,
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        max_seq_len=seq_len,
        seq_lens=torch.full((num_reqs,), seq_len, dtype=torch.int32),
        block_table=torch.arange(num_reqs, dtype=torch.int32).unsqueeze(1),
        slot_mapping=torch.arange(num_reqs, dtype=torch.int64),
        scheduler_metadata=None,
        causal=True,
    )


def test_cached_decode_support_data_matches_uncached_reference():
    """`_cached_decode_support_data` (see its own doc comment in
    attention.py) must produce the exact same
    `(query_lens_all_ones, seq_lens_cpu, block_table_cpu)` tuple a
    from-scratch computation would -- pure Python/torch, no Vulkan
    device needed (this function never touches the GPU itself)."""
    metadata = _make_decode_metadata(num_reqs=3, seq_len=5)
    num_actual_tokens = 3

    query_lens_all_ones, seq_lens_cpu, block_table_cpu = (
        attention_mod._cached_decode_support_data(metadata, num_actual_tokens)
    )

    query_lens_ref = (metadata.query_start_loc[1:] - metadata.query_start_loc[:-1]).to(
        device="cpu"
    )
    expected_all_ones = query_lens_ref.numel() == num_actual_tokens and bool(
        torch.all(query_lens_ref[:num_actual_tokens] == 1).item()
    )
    expected_seq_lens = metadata.seq_lens[:num_actual_tokens].to("cpu")
    # int64, not metadata.block_table's original int32: _cached_decode_support_data
    # also normalizes the dtype (matching what kv_ops._block_table_to_u32 would
    # otherwise convert independently, per row, on every per-token call) -- see
    # its own doc comment.
    expected_block_table = metadata.block_table[:num_actual_tokens].to(
        device="cpu", dtype=torch.int64
    )

    assert query_lens_all_ones == expected_all_ones
    assert block_table_cpu.dtype == torch.int64
    torch.testing.assert_close(seq_lens_cpu, expected_seq_lens)
    torch.testing.assert_close(block_table_cpu, expected_block_table)


def test_cached_decode_support_data_detects_non_decode_query_lens():
    """A query_start_loc implying a query length > 1 for some request
    (i.e. a prefill/chunked-prefill step, not pure decode) must still
    correctly report `query_lens_all_ones=False` through the cached
    path, exactly as the original uncached check did."""
    metadata = _make_decode_metadata(num_reqs=2, seq_len=5)
    metadata.query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32)  # lens [1, 3]

    query_lens_all_ones, _, _ = attention_mod._cached_decode_support_data(metadata, 2)
    assert query_lens_all_ones is False


def test_cached_decode_support_data_hits_cache_for_same_metadata_object():
    """Repeated calls with the *same* `attn_metadata` object and
    `num_actual_tokens` must return the identical (not just equal)
    cached tuple, not recompute from scratch every time -- the whole
    point of this cache (see its own doc comment: real vLLM shares one
    `attn_metadata` object across every attention layer within a
    KV-cache-group, confirmed by reading
    `vllm/v1/worker/gpu_model_runner.py`'s `_build_attn_group_metadata`
    directly)."""
    metadata = _make_decode_metadata(num_reqs=2, seq_len=3)

    result1 = attention_mod._cached_decode_support_data(metadata, 2)
    result2 = attention_mod._cached_decode_support_data(metadata, 2)
    assert result1 is result2, (
        "repeated calls with the same (attn_metadata, num_actual_tokens) must "
        "hit the cache, returning the identical cached tuple"
    )


def test_cached_decode_support_data_recomputes_for_different_metadata_object():
    """A second, distinct `attn_metadata` object (even with identical
    field values) must not silently reuse the first object's cached
    result -- the cache compares `attn_metadata` identity with `is`,
    never `==` or a value-derived key (the same reasoning already
    documented for `kv_ops._cached_available_shaders`/
    `vulkan_ops._cached_available_shaders`)."""
    metadata1 = _make_decode_metadata(num_reqs=2, seq_len=3)
    metadata2 = _make_decode_metadata(num_reqs=2, seq_len=3)

    result1 = attention_mod._cached_decode_support_data(metadata1, 2)
    result2 = attention_mod._cached_decode_support_data(metadata2, 2)
    # Equal values (both metadata objects have identical fields), but
    # must not be the literal same cached tuple -- confirms the cache
    # actually recomputed for metadata2 instead of returning metadata1's
    # stale cached result.
    assert result1[0] == result2[0]
    torch.testing.assert_close(result1[1], result2[1])
    torch.testing.assert_close(result1[2], result2[2])
    assert result1 is not result2

    # Re-querying metadata1 after metadata2 was cached must recompute
    # for metadata1 again (single-slot cache, not a full dict) rather
    # than incorrectly returning metadata2's cached value.
    result1_again = attention_mod._cached_decode_support_data(metadata1, 2)
    assert result1_again is not result1
    assert result1_again[0] == result1[0]
    torch.testing.assert_close(result1_again[1], result1[1])
    torch.testing.assert_close(result1_again[2], result1[2])


def test_multiple_layers_sharing_attn_metadata_produce_correct_results(
    cpu_ops_stubbed,
):
    """End-to-end regression test for the real scenario this caching
    change targets: multiple `VulkanAttentionBackendImpl` instances
    (simulating multiple decoder layers within the same KV-cache-group,
    as real vLLM constructs them -- see `_cached_decode_support_data`'s
    doc comment) processing *different* query/key/value data but
    sharing the exact same `attn_metadata` object, in sequence, within
    one decode step. Each layer's own output must be correct and
    independent -- the shared cache must never leak one layer's
    KV-cache/query data into another's result.
    """
    _require_vulkan_context()

    num_heads = 4
    num_kv_heads = 2
    head_size = 32
    block_size = 16
    num_blocks = 4
    num_reqs = 2
    scale = head_size**-0.5

    metadata = CPUAttentionMetadata(
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

    def make_impl() -> VulkanAttentionBackendImpl:
        return VulkanAttentionBackendImpl(
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

    gqa_ratio = num_heads // num_kv_heads
    torch.manual_seed(42)
    for _layer_idx in range(3):
        query = torch.randn(num_reqs, num_heads, head_size, dtype=torch.float32)
        key = torch.randn(num_reqs, num_kv_heads, head_size, dtype=torch.float16)
        value = torch.randn(num_reqs, num_kv_heads, head_size, dtype=torch.float16)
        kv_cache = torch.zeros(
            (num_blocks, num_kv_heads, block_size, 2 * head_size),
            dtype=torch.float16,
        )
        output = torch.empty((num_reqs, num_heads, head_size), dtype=torch.float32)

        impl = make_impl()
        _write_kv(impl, key, value, kv_cache, metadata.slot_mapping)
        result = impl.forward(
            layer=None,  # type: ignore[arg-type]
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=metadata,  # the *same* object, every "layer"
            output=output,
        )

        expected = []
        for req_idx in range(num_reqs):
            rows = []
            for q_head in range(num_heads):
                rows.append(value[req_idx, q_head // gqa_ratio].float())
            expected.append(torch.stack(rows))
        expected = torch.stack(expected)

        assert impl._last_vulkan_decode_used is True
        assert result is output
        torch.testing.assert_close(output, expected, rtol=5e-3, atol=5e-3)


def test_get_kv_cache_entry_hits_instance_cache_for_same_tensor():
    """`VulkanAttentionBackendImpl._get_kv_cache_entry` (see its own doc
    comment) must return the identical (not just equal) cached entry on
    repeated calls with the same `kv_cache` tensor object -- the whole
    point of this per-layer instance cache (real vLLM binds a layer's
    `kv_cache` tensor exactly once, at `initialize_kv_cache()` time, and
    passes that same object on every decode step thereafter).
    """
    _require_vulkan_context()

    impl = VulkanAttentionBackendImpl(
        num_heads=2,
        head_size=8,
        scale=8**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )
    ctx = _require_vulkan_context()
    kv_cache = torch.zeros((4, 1, 16, 16), dtype=torch.float16)

    entry1 = impl._get_kv_cache_entry(ctx, kv_cache)
    entry2 = impl._get_kv_cache_entry(ctx, kv_cache)
    assert entry1 is entry2, (
        "repeated calls with the same kv_cache tensor object must hit the "
        "instance cache, returning the identical entry"
    )


def test_get_kv_cache_entry_recomputes_for_a_different_tensor():
    """A second, distinct `kv_cache` tensor (even with identical shape/
    dtype) must not silently reuse the first tensor's cached entry --
    the per-instance cache compares tensor identity with `is`, matching
    the same reasoning already documented for
    `attention._cached_decode_support_data`/
    `kv_ops._cached_available_shaders`/`vulkan_ops._cached_available_shaders`.
    """
    _require_vulkan_context()

    impl = VulkanAttentionBackendImpl(
        num_heads=2,
        head_size=8,
        scale=8**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    )
    ctx = _require_vulkan_context()
    kv_cache_1 = torch.zeros((4, 1, 16, 16), dtype=torch.float16)
    kv_cache_2 = torch.zeros((4, 1, 16, 16), dtype=torch.float16)

    entry1 = impl._get_kv_cache_entry(ctx, kv_cache_1)
    entry2 = impl._get_kv_cache_entry(ctx, kv_cache_2)
    assert entry1 is not entry2, (
        "a different kv_cache tensor object must not reuse the previous "
        "tensor's cached entry"
    )

    # Re-querying kv_cache_1 after kv_cache_2 was cached must recompute
    # for kv_cache_1 again (single-slot instance cache, not a full
    # dict) rather than incorrectly returning kv_cache_2's cached entry.
    entry1_again = impl._get_kv_cache_entry(ctx, kv_cache_1)
    assert entry1_again is entry1


def test_multiple_decode_steps_reuse_kv_cache_entry_and_stay_correct(
    cpu_ops_stubbed,
):
    """End-to-end regression test for the real scenario this caching
    change targets: several consecutive decode steps (as would happen
    across a real serving session) through the *same*
    `VulkanAttentionBackendImpl` instance with the *same* `kv_cache`
    tensor object each time. The cached KV-cache entry must be reused
    (not just correct) across those calls, and each step's own output
    must remain numerically correct.
    """
    _require_vulkan_context()

    num_heads = 4
    num_kv_heads = 2
    head_size = 32
    block_size = 16
    num_blocks = 4
    scale = head_size**-0.5

    kv_cache = torch.zeros(
        (num_blocks, num_kv_heads, block_size, 2 * head_size),
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

    gqa_ratio = num_heads // num_kv_heads
    torch.manual_seed(7)
    cached_entry = None
    for step in range(3):
        num_reqs = 1
        query = torch.randn(num_reqs, num_heads, head_size, dtype=torch.float32)
        key = torch.randn(num_reqs, num_kv_heads, head_size, dtype=torch.float16)
        value = torch.randn(num_reqs, num_kv_heads, head_size, dtype=torch.float16)
        output = torch.empty((num_reqs, num_heads, head_size), dtype=torch.float32)

        metadata = CPUAttentionMetadata(
            num_actual_tokens=num_reqs,
            max_query_len=1,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            max_seq_len=1,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            block_table=torch.tensor([[step % num_blocks]], dtype=torch.int32),
            # This is the very first (and only) token of a fresh,
            # independent single-token sequence each step (seq_lens=[1]
            # below), so it belongs at offset 0 of whichever block
            # block_table assigns it - slot = block_id * block_size + 0.
            # (A previous version of this test computed slot_mapping as
            # `step % (num_blocks * block_size)`, which only happened to
            # coincide with the correct slot at step=0; for step>=1 it
            # named a slot inside a *different* block than block_table
            # pointed at, so decode read back zeros instead of this
            # step's just-written key/value - a real, if latent, bug in
            # this test's own setup that a merged Vulkan write+decode
            # path (see kv_ops.py's paged_kv_write_and_decode_batch_f32)
            # surfaced by actually depending on write-then-read
            # consistency within a single GPU submit.)
            slot_mapping=torch.tensor(
                [(step % num_blocks) * block_size], dtype=torch.int64
            ),
            scheduler_metadata=None,
            causal=True,
        )

        _write_kv(impl, key, value, kv_cache, metadata.slot_mapping)
        result = impl.forward(
            layer=None,  # type: ignore[arg-type]
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,  # the *same* tensor object, every step
            attn_metadata=metadata,
            output=output,
        )

        expected = torch.stack(
            [value[0, q_head // gqa_ratio].float() for q_head in range(num_heads)]
        )
        assert impl._last_vulkan_decode_used is True
        assert result is output
        torch.testing.assert_close(output[0], expected, rtol=5e-3, atol=5e-3)

        if cached_entry is None:
            cached_entry = impl._cached_kv_cache_entry
        else:
            assert impl._cached_kv_cache_entry is cached_entry, (
                "the same kv_cache tensor across decode steps must keep "
                "reusing the identical cached entry, not re-resolve it"
            )


def test_layers_sharing_one_buffer_get_distinct_vulkan_mirrors():
    """vLLM allocates one buffer per `KVCacheTensor` and gives every layer
    sharing it a view at its own offset, so all those layers report the same
    `untyped_storage().data_ptr()` and differ only in `.data_ptr()`.

    Keying the Vulkan mirror on the storage base therefore collapsed every
    such layer onto the first one's mirror: each layer decoded against
    another layer's K/V. Nothing raised and every kernel stayed individually
    correct -- generation just came out as word salad. Measured on vLLM 0.29
    with Qwen3-0.6B: 28 layers, 1 distinct storage base, 28 distinct
    `.data_ptr()`s.
    """
    ctx = _require_vulkan_context()
    num_blocks, num_kv_heads, block_size, head_size = 4, 2, 16, 8
    layer_numel = num_blocks * num_kv_heads * block_size * 2 * head_size

    # One buffer, two layer-sized views into it, exactly as vLLM lays them out.
    buffer = torch.zeros(2 * layer_numel, dtype=torch.float16)
    layer_0, layer_1 = (
        buffer[i * layer_numel : (i + 1) * layer_numel].view(
            num_blocks, num_kv_heads, block_size, 2 * head_size
        )
        for i in range(2)
    )
    assert (
        layer_0.untyped_storage().data_ptr() == layer_1.untyped_storage().data_ptr()
    ), "fixture must reproduce vLLM's shared-storage layout"

    entry_0 = attention_mod._get_or_create_vulkan_kv_cache(ctx, layer_0)
    entry_1 = attention_mod._get_or_create_vulkan_kv_cache(ctx, layer_1)

    assert entry_0 is not entry_1, (
        "layers sharing one buffer must not share a Vulkan mirror"
    )
    # And the first layer's entry must still be its own afterwards, rather
    # than having been evicted by the second.
    assert attention_mod._get_or_create_vulkan_kv_cache(ctx, layer_0) is entry_0


class TestKVCacheShape:
    """vLLM's per-layer KV cache is HND:
    ``(num_blocks, num_kv_heads, block_size, 2 * head_size)``. The Vulkan
    mirror is a separate GPU allocation written from the `key`/`value` token
    tensors, so `_parse_kv_cache_shape` reading `head_size` out of the doubled
    last dimension is the whole of what vLLM's layout contributes to it.
    """

    def test_head_size_comes_from_the_doubled_last_dimension(self):
        assert attention_mod._parse_kv_cache_shape((4, 2, 16, 16)) == (4, 2, 16, 8)

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 2, 16),  # too few dimensions
            (2, 4, 2, 16, 8),  # the pre-0.23 [K, V] pair layout
            (4, 2, 16, 7),  # odd (un-splittable) last dimension
        ],
    )
    def test_unrecognized_shapes_are_rejected(self, shape):
        with pytest.raises(ValueError, match="expected KV cache shape"):
            attention_mod._parse_kv_cache_shape(shape)

    def test_vulkan_layout_matches_the_model_dimensions(self):
        """The mirror's layout must describe the same model the HND cache
        does, with `head_size` halved out of the last dimension rather than
        taken from it whole -- getting that wrong would size every block in
        the Vulkan cache at double, and silently corrupt every paged read.
        """
        ctx = _require_vulkan_context()
        num_blocks, num_kv_heads, block_size, head_size = 4, 2, 16, 8

        entry = attention_mod._get_or_create_vulkan_kv_cache(
            ctx,
            torch.zeros(
                (num_blocks, num_kv_heads, block_size, 2 * head_size),
                dtype=torch.float16,
            ),
        )

        spec = entry.layout.layer_spec(attention_mod._PER_LAYER_KV_CACHE_INDEX)
        assert entry.layout.num_blocks == num_blocks
        assert spec.num_kv_heads == num_kv_heads
        assert spec.block_size == block_size
        assert spec.head_size == head_size
