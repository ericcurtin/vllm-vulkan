# SPDX-License-Identifier: Apache-2.0
"""Deferred Vulkan dispatch context.

Instead of submitting one vkQueueSubmit per op, we accumulate ops into a
pending list and flush them all in one batch when:
  1. A non-Vulkan op is requested (e.g. attention, which runs on CPU).
  2. The caller explicitly calls flush().
  3. The pending list reaches a size limit.

This gives us near-zero per-op submit overhead for sequences of compatible ops.
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

import torch
import numpy as np

if TYPE_CHECKING:
    from vllm_vulkan._rs import VulkanContext

logger = logging.getLogger(__name__)

# Maximum pending ops before auto-flush (prevents unbounded memory use).
MAX_PENDING = 128


class BatchContext:
    """Accumulates Vulkan ops and flushes them in one vkQueueSubmit."""

    def __init__(self, ctx: "VulkanContext") -> None:
        self._ctx = ctx
        # Each entry: (shader, bindings, output_size, pc, workgroups, barrier_after)
        # output_sizes is a list for multi-output shaders
        self._ops: list = []
        # Maps op index → list of future handles for retrieving outputs
        self._futures: list[list[_OutputFuture]] = []
        self._n_outputs = 0

    def add_op(
        self,
        shader: str,
        bindings: list,
        output_sizes: list[int],
        push_constants: bytes,
        workgroups: tuple[int, int, int],
        barrier_after: bool = False,
    ) -> "list[_OutputFuture]":
        """Add an op to the pending batch. Returns futures for each output."""
        self._ops.append((shader, bindings, output_sizes, push_constants, workgroups, barrier_after))
        futures = [_OutputFuture(self, len(self._ops) - 1, i) for i in range(len(output_sizes))]
        self._futures.append(futures)
        if len(self._ops) >= MAX_PENDING:
            self.flush()
        return futures

    def flush(self) -> None:
        """Execute all pending ops in a single vkQueueSubmit."""
        if not self._ops:
            return
        results = self._ctx.execute_batch(self._ops)
        # Deliver results to futures.
        for op_idx, (op_results, futures) in enumerate(zip(results, self._futures)):
            for out_idx, (data_bytes, future) in enumerate(zip(op_results, futures)):
                future._data = data_bytes
        self._ops.clear()
        self._futures.clear()

    def is_empty(self) -> bool:
        return len(self._ops) == 0


class _OutputFuture:
    """Handle to an output buffer that will be filled after flush()."""
    __slots__ = ("_ctx", "_op_idx", "_out_idx", "_data")

    def __init__(self, ctx: BatchContext, op_idx: int, out_idx: int) -> None:
        self._data: bytes | None = None

    def get(self) -> bytes:
        """Return output data, flushing the batch if not yet ready."""
        return self._data  # type: ignore[return-value]


# ─── Global batch context (one per thread / worker) ──────────────────────────

_batch_ctx: "BatchContext | None" = None


def get_batch_ctx() -> "BatchContext | None":
    return _batch_ctx


def set_batch_ctx(ctx: "BatchContext") -> None:
    global _batch_ctx
    _batch_ctx = ctx


def flush() -> None:
    """Flush all pending Vulkan ops."""
    if _batch_ctx is not None:
        _batch_ctx.flush()
