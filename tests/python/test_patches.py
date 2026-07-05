# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_vulkan.patches's pure-Python compute_slot_mapping fallback.

_pure_python_compute_slot_mapping (patches.py) is only installed when
vllm._C (the compiled C++ extension) isn't available - a fallback for
running vLLM from its Python source tree without building the C++
extensions. It used to loop over range(num_reqs) in Python, calling
.item() twice per request (forcing a CPU/tensor sync each time) purely to
slice positions/slot_mapping and index one block_table row at a time - work
now done in one vectorized pass instead.

These tests compare the vectorized implementation against a verbatim copy
of the original per-request-loop version (kept here as an independent
reference, not a copy of the new implementation), across several batch
shapes including empty requests, and measure the speedup at a more
realistic concurrent-batch size. No Vulkan device or vllm install needed -
this is pure PyTorch/Python logic.
"""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from vllm_vulkan.patches import _pure_python_compute_slot_mapping  # noqa: E402


def _reference_compute_slot_mapping_loop(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    slot_mapping: torch.Tensor,
) -> None:
    """Verbatim copy of _pure_python_compute_slot_mapping's original
    per-request Python loop, kept as an independent reference (not derived
    from the vectorized implementation under test)."""
    num_reqs = query_start_loc.shape[0] - 1

    for req_idx in range(num_reqs):
        start = int(query_start_loc[req_idx].item())
        end = int(query_start_loc[req_idx + 1].item())
        if start >= end:
            continue

        pos_slice = positions[start:end]
        block_indices = (pos_slice // block_size).long()
        row = block_table[req_idx]
        block_ids = row[block_indices].long()
        offsets = (pos_slice % block_size).long()
        slots = block_ids * block_size + offsets
        slot_mapping[start:end] = slots


def _make_batch(
    seed: int, req_lens: list[int], block_size: int, max_blocks_per_req: int
):
    torch.manual_seed(seed)
    num_tokens = sum(req_lens)
    num_reqs = len(req_lens)

    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(req_lens, dtype=torch.int32).cumsum(0)

    # Random positions within a plausible sequence length range, per request.
    positions = torch.cat(
        [torch.randperm(max(1, ln * 3))[:ln].sort().values.long() for ln in req_lens]
        if num_tokens > 0
        else [torch.zeros(0, dtype=torch.int64)]
    )
    assert positions.shape[0] == num_tokens

    block_table = torch.randint(
        0, 1000, (num_reqs, max_blocks_per_req), dtype=torch.int32
    )
    return num_tokens, query_start_loc, positions, block_table


@pytest.mark.parametrize(
    ("req_lens", "block_size", "max_blocks_per_req"),
    [
        ([5], 4, 8),
        ([3, 4, 2], 4, 8),
        ([0, 5, 0, 3], 4, 8),  # some requests have zero tokens this step
        ([1, 1, 1, 1, 1], 2, 4),
        ([17, 0, 9, 0, 0, 23], 16, 16),
        ([], 4, 8),  # no requests at all
    ],
)
def test_vectorized_matches_reference_loop(
    req_lens: list[int], block_size: int, max_blocks_per_req: int
):
    num_tokens, query_start_loc, positions, block_table = _make_batch(
        0, req_lens, block_size, max_blocks_per_req
    )
    max_num_tokens = max(num_tokens, 1) * 2

    slot_mapping_ref = torch.full((max_num_tokens,), -1, dtype=torch.int64)
    slot_mapping_new = torch.full((max_num_tokens,), -1, dtype=torch.int64)

    _reference_compute_slot_mapping_loop(
        num_tokens,
        max_num_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table.shape[1],
        block_size,
        slot_mapping_ref,
    )
    _pure_python_compute_slot_mapping(
        num_tokens,
        max_num_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table.shape[1],
        block_size,
        slot_mapping_new,
    )

    torch.testing.assert_close(slot_mapping_new, slot_mapping_ref)


def test_vectorized_is_faster_at_a_realistic_batch_size():
    # A moderately large concurrent-decode-style batch: many requests, one
    # token each (the shape this fallback would see once per decode step).
    req_lens = [1] * 128
    num_tokens, query_start_loc, positions, block_table = _make_batch(
        1, req_lens, block_size=16, max_blocks_per_req=64
    )
    max_num_tokens = num_tokens

    def time_best_of(trials: int, iters: int, fn) -> float:
        best = float("inf")
        for _ in range(trials):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            dt = (time.perf_counter() - t0) / iters
            best = min(best, dt)
        return best

    slot_mapping = torch.zeros(max_num_tokens, dtype=torch.int64)

    old_s = time_best_of(
        5,
        50,
        lambda: _reference_compute_slot_mapping_loop(
            num_tokens,
            max_num_tokens,
            query_start_loc,
            positions,
            block_table,
            block_table.shape[1],
            16,
            slot_mapping,
        ),
    )
    new_s = time_best_of(
        5,
        50,
        lambda: _pure_python_compute_slot_mapping(
            num_tokens,
            max_num_tokens,
            query_start_loc,
            positions,
            block_table,
            block_table.shape[1],
            16,
            slot_mapping,
        ),
    )

    print(
        f"\ncompute_slot_mapping ({len(req_lens)} reqs, 1 token each): "
        f"loop {old_s * 1e6:.1f} us/call   vectorized {new_s * 1e6:.1f} us/call   "
        f"speedup {old_s / new_s:.2f}x"
    )
    assert new_s < old_s, (
        f"vectorized ({new_s * 1e6:.1f}us) was not faster than the "
        f"per-request loop ({old_s * 1e6:.1f}us)"
    )
