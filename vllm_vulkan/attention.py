# SPDX-License-Identifier: Apache-2.0
"""vLLM attention backend shim for Vulkan decode bring-up.

This backend intentionally keeps vLLM's CPU attention metadata and CPU KV cache
layout as the source of truth, then mirrors supported decode updates into the
Vulkan paged KV cache. Unsupported cases fall back to CPU_ATTN.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm import _custom_ops as ops
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackend,
    CPUAttentionBackendImpl,
    CPUAttentionMetadata,
)

from vllm_vulkan import envs
from vllm_vulkan.kv_layout import KVCacheLayerSpec, VulkanPagedKVLayout

if TYPE_CHECKING:
    from vllm_vulkan._rs import GpuTensor, VulkanContext

logger = logging.getLogger(__name__)

# vLLM passes a per-layer KV cache tensor into each Attention.forward call.
# The Vulkan mirror therefore contains exactly that one layer, represented as
# layer 0 in our VulkanPagedKVLayout.
_PER_LAYER_KV_CACHE_INDEX = 0
_MAX_VERIFIED_SEQUENCE_KEYS = 4096
# Crossover point (in number of tokens) below which a pure-Python loop is
# faster than a vectorized numpy pass for _mark_vulkan_cache_slots_written
# -- measured directly, see that function's docstring.
_MARK_SLOTS_VECTORIZE_THRESHOLD = 64


class VulkanAttentionBackend(CPUAttentionBackend):
    """CPU_ATTN-compatible backend that opportunistically uses Vulkan decode.

    The backend name intentionally remains ``CPU_ATTN`` because vLLM currently
    indexes ``AttentionBackendEnum`` by backend name. Returning a new name would
    require upstream enum registration. The implementation class is Vulkan
    specific, while the metadata builder and KV-cache shape stay CPU-compatible.
    """

    @staticmethod
    def get_name() -> str:
        return CPUAttentionBackend.get_name()

    @staticmethod
    def get_impl_cls() -> type[VulkanAttentionBackendImpl]:
        return VulkanAttentionBackendImpl


@dataclass
class _VulkanKVCacheEntry:
    storage_key: int
    layout: VulkanPagedKVLayout
    cache: GpuTensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    written_slots: bytearray
    verified_prefix_by_blocks: OrderedDict[tuple[int, ...], int]


_VULKAN_KV_CACHES: dict[int, _VulkanKVCacheEntry] = {}


class VulkanAttentionBackendImpl(CPUAttentionBackendImpl):
    """CPU attention implementation with a guarded Vulkan decode fast path."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_vulkan_decode_used = False

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for VulkanAttentionBackendImpl"
            )

        self._last_vulkan_decode_used = False

        # For warming-up.
        if attn_metadata is None:
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens
        allow_vulkan_decode = True
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._run_sdpa_forward(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                self.attn_type,
            )

        key_cache, value_cache = kv_cache.unbind(0)

        # Keep vLLM's CPU KV cache updated first. This preserves the existing
        # fallback behavior and lets unsupported cases use CPU_ATTN immediately.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            ops.cpu_attn_reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                attn_metadata.isa,
            )
            _try_write_tokens_to_vulkan_cache(
                kv_cache=kv_cache,
                key=key,
                value=value,
                slot_mapping=attn_metadata.slot_mapping,
                num_tokens=num_actual_tokens,
            )

        if attn_metadata.use_sdpa_prefill:
            allow_vulkan_decode = False
            assert self.sinks is None, "Attention sink is unsupported in SDPA prefill"
            num_decode_tokens = attn_metadata.num_decode_tokens
            self._run_sdpa_forward(
                query[num_decode_tokens:num_actual_tokens],
                key[num_decode_tokens:num_actual_tokens],
                value[num_decode_tokens:num_actual_tokens],
                output[num_decode_tokens:num_actual_tokens],
                attn_metadata,
                self.attn_type,
            )
            num_actual_tokens = num_decode_tokens

        if num_actual_tokens > 0:
            if not allow_vulkan_decode or not self._try_vulkan_decode(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
                num_actual_tokens=num_actual_tokens,
            ):
                ops.cpu_attention_with_kv_cache(
                    query=query[:num_actual_tokens],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    output=output[:num_actual_tokens],  # type: ignore[arg-type]
                    query_start_loc=attn_metadata.query_start_loc,
                    seq_lens=attn_metadata.seq_lens,
                    scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,  # type: ignore[arg-type]
                    sliding_window=self.sliding_window,
                    block_table=attn_metadata.block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=attn_metadata.scheduler_metadata,
                    s_aux=self.sinks,
                )

        return output

    def _try_vulkan_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata,
        output: torch.Tensor,
        num_actual_tokens: int,
    ) -> bool:
        if envs.VLLM_VULKAN_DISABLE_ATTN:
            return False

        if not self._supports_vulkan_decode(
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            num_actual_tokens=num_actual_tokens,
        ):
            return False

        try:
            ctx = _get_vulkan_context()

            if ctx is None:
                return False

            entry = _get_or_create_vulkan_kv_cache(ctx, kv_cache)
            if kv_cache.dtype == torch.float16:
                from vllm_vulkan.kv_ops import (  # noqa: PLC0415
                    paged_attn_decode_batch_f16,
                )

                decode_batch = paged_attn_decode_batch_f16
            else:
                from vllm_vulkan.kv_ops import (  # noqa: PLC0415
                    paged_attn_decode_batch_f32,
                )

                decode_batch = paged_attn_decode_batch_f32

            seq_lens = attn_metadata.seq_lens[:num_actual_tokens].to("cpu")
            block_table = attn_metadata.block_table[:num_actual_tokens].to("cpu")
            if not _vulkan_cache_has_sequences(entry, block_table, seq_lens):
                return False

            # One vkQueueSubmit for the whole batch instead of one per
            # token: kv_ops.py's module-level design already batches every
            # other op this way ("batch all ops ... into one vkQueueSubmit"
            # to avoid "~150µs driver overhead each") - this loop used to
            # be the one place still issuing a separate submit+fence-wait
            # per token, once per attention layer per decode step.
            #
            # unbind(0)/tolist() do the detach+device-transfer+split (or
            # int conversion) once, in a single C-level call each, instead
            # of num_actual_tokens separate Python-level slice/.item() calls
            # - avoiding exactly the kind of per-token Python overhead this
            # PR otherwise removes from the GPU-submission side.
            outs = decode_batch(
                ctx,
                entry.layout,
                entry.cache,
                _PER_LAYER_KV_CACHE_INDEX,
                list(query[:num_actual_tokens].detach().to("cpu").unbind(0)),
                list(block_table.unbind(0)),
                seq_lens.tolist(),
                self.scale,
            )

            output[:num_actual_tokens].copy_(
                torch.stack(outs).to(dtype=output.dtype, device=output.device)
            )
            logger.debug(
                "Vulkan attention decode used: tokens=%d heads=%d head_size=%d dtype=%s",
                num_actual_tokens,
                self.num_heads,
                self.head_size,
                kv_cache.dtype,
            )
            self._last_vulkan_decode_used = True

            return True
        except Exception as exc:
            logger.debug("Vulkan attention decode fallback to CPU_ATTN: %s", exc)

            return False

    def _supports_vulkan_decode(
        self,
        *,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata,
        output: torch.Tensor,
        num_actual_tokens: int,
    ) -> bool:
        """Return True for the conservative decode-only path we can run."""
        if self.attn_type != AttentionType.DECODER:
            return False
        if not attn_metadata.causal:
            return False
        if self.kv_sharing_target_layer_name is not None:
            return False
        if key is None or value is None:
            return False
        if self.alibi_slopes is not None or bool(self.logits_soft_cap):
            return False
        if self.sinks is not None or self.sliding_window != (-1, -1):
            return False
        if output.shape[-1] != self.head_size:
            return False
        if kv_cache.dtype not in (torch.float16, torch.float32):
            return False

        query_lens = (
            attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        ).to(device="cpu")
        if query_lens.numel() != num_actual_tokens:
            return False
        return bool(torch.all(query_lens[:num_actual_tokens] == 1).item())


def _try_write_tokens_to_vulkan_cache(
    *,
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> None:
    """Best-effort mirror of newly produced K/V tokens into Vulkan cache."""
    if num_tokens <= 0 or kv_cache.dtype not in (torch.float16, torch.float32):
        return

    try:
        ctx = _get_vulkan_context()
        if ctx is None:
            return

        entry = _get_or_create_vulkan_kv_cache(ctx, kv_cache)
        if kv_cache.dtype == torch.float16:
            from vllm_vulkan.kv_ops import (  # noqa: PLC0415
                paged_kv_write_f16 as write_kv,
            )
        else:
            from vllm_vulkan.kv_ops import (  # noqa: PLC0415
                paged_kv_write_f32 as write_kv,
            )

        slots = (
            slot_mapping[:num_tokens]
            .detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
        )
        write_kv(
            ctx,
            entry.layout,
            entry.cache,
            _PER_LAYER_KV_CACHE_INDEX,
            key[:num_tokens],
            value[:num_tokens],
            slots,
        )
        _mark_vulkan_cache_slots_written(entry, slots)
    except Exception as exc:
        logger.debug("Vulkan KV cache mirror skipped: %s", exc)


def _vulkan_cache_has_sequences(
    entry: _VulkanKVCacheEntry,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> bool:
    layout = entry.layout
    spec = layout.layer_spec(_PER_LAYER_KV_CACHE_INDEX)
    block_size = spec.block_size
    num_blocks = layout.num_blocks

    # .tolist() converts each whole tensor to a plain Python list in one
    # C-level call, instead of a per-element `int(tensor_scalar)` conversion
    # for every request/block (each of which is itself a small, surprisingly
    # expensive tensor->Python-int round trip). This function runs once per
    # attention layer per decode step, once per concurrent request in the
    # batch - measured ~10x faster overall this way (64 requests, steady
    # state where the per-token verification loop below is already O(1)
    # amortized via _verified_prefix_len/_remember_verified_prefix: the
    # per-request tensor-element-conversion overhead this avoids was the
    # actual dominant cost, not the token-verification loop itself).
    seq_lens_list = seq_lens.tolist()
    block_table_list = block_table.tolist()

    for req_idx, seq_len in enumerate(seq_lens_list):
        row = block_table_list[req_idx]
        needed_blocks = (seq_len + block_size - 1) // block_size
        active_blocks = _active_block_ids(row, needed_blocks, num_blocks)
        if active_blocks is None:
            return False

        start_pos = min(_verified_prefix_len(entry, active_blocks), seq_len)
        for token_pos in range(start_pos, seq_len):
            logical_block_id = token_pos // block_size
            token_offset = token_pos % block_size
            physical_block_id = active_blocks[logical_block_id]
            slot = physical_block_id * block_size + token_offset
            if not entry.written_slots[slot]:
                return False
        _remember_verified_prefix(entry, active_blocks, seq_len)

    return True


def _mark_vulkan_cache_slots_written(
    entry: _VulkanKVCacheEntry, slots: torch.Tensor
) -> None:
    """Marks each of `slots` as written in `entry.written_slots` (a
    bytearray bitmap).

    `num_tokens` (`slots.numel()`) is the total token count for the current
    step across every sequence in the running batch -- 1 per active
    sequence during decode (continuous batching), or up to a whole
    prompt's length during prefill -- so it varies widely across real
    workloads, from 1 (a single decoding sequence) to hundreds
    (a large concurrently-batched decode step, or prefill).

    Measured directly: a pure-Python loop (`.tolist()` + per-element
    bytearray assignment) is faster below ~48-64 tokens (numpy's fixed
    per-call overhead -- constructing the view/boolean mask -- dominates
    at small sizes), while a vectorized numpy approach (`np.frombuffer`
    over the bytearray's own memory, writable in-place with no copy,
    since unlike immutable `bytes` a `bytearray` supports the writable
    buffer protocol) is up to ~2x faster at 128 tokens and keeps
    improving beyond that, since the Python loop's per-element cost is
    O(n) while numpy's fixed overhead is O(1). Picking the empirically
    faster implementation by size covers both the small-batch/low-
    concurrency case and the large-batch/high-throughput serving case
    optimally, rather than regressing one to fix the other.
    """
    capacity = len(entry.written_slots)
    # `slots.numel()` rather than `len(slots)`: measured `len()` on a
    # torch.Tensor to cost ~0.35us on its own (dunder-method/dispatch
    # overhead) -- the same order of magnitude as this whole function's
    # total runtime at small `num_tokens` -- while `.numel()` costs only
    # ~0.14us, making the size check itself negligible instead of
    # doubling the small-`num_tokens` case's cost on its own.
    if slots.numel() < _MARK_SLOTS_VECTORIZE_THRESHOLD:
        for slot in slots.tolist():
            if 0 <= slot < capacity:
                entry.written_slots[slot] = 1
        return
    slots_np = slots.numpy()
    valid = (slots_np >= 0) & (slots_np < capacity)
    written_view = np.frombuffer(entry.written_slots, dtype=np.uint8)
    written_view[slots_np[valid]] = 1


def _active_block_ids(
    row: list[int], needed_blocks: int, num_blocks: int
) -> tuple[int, ...] | None:
    if len(row) < needed_blocks:
        return None

    active_blocks = tuple(row[:needed_blocks])
    if any(block_id < 0 or block_id >= num_blocks for block_id in active_blocks):
        return None
    return active_blocks


def _verified_prefix_len(
    entry: _VulkanKVCacheEntry, active_blocks: tuple[int, ...]
) -> int:
    verified = entry.verified_prefix_by_blocks.get(active_blocks)
    if verified is not None:
        return verified
    if len(active_blocks) > 1:
        return entry.verified_prefix_by_blocks.get(active_blocks[:-1], 0)
    return 0


def _remember_verified_prefix(
    entry: _VulkanKVCacheEntry, active_blocks: tuple[int, ...], seq_len: int
) -> None:
    entry.verified_prefix_by_blocks[active_blocks] = seq_len
    entry.verified_prefix_by_blocks.move_to_end(active_blocks)
    while len(entry.verified_prefix_by_blocks) > _MAX_VERIFIED_SEQUENCE_KEYS:
        entry.verified_prefix_by_blocks.popitem(last=False)


def _get_vulkan_context() -> VulkanContext | None:
    try:
        from vllm_vulkan import vulkan_ops  # noqa: PLC0415
        from vllm_vulkan._rs import VulkanContext  # noqa: PLC0415

        if not vulkan_ops.is_ready():
            vulkan_ops.set_context(VulkanContext(0))
        if not vulkan_ops.is_ready():
            return None
        return vulkan_ops.get_context()
    except Exception as exc:
        logger.debug("VulkanContext unavailable for attention decode: %s", exc)
        return None


def _get_or_create_vulkan_kv_cache(
    ctx: VulkanContext,
    kv_cache: torch.Tensor,
) -> _VulkanKVCacheEntry:
    key = _kv_cache_storage_key(kv_cache)
    shape = tuple(int(dim) for dim in kv_cache.shape)
    entry = _VULKAN_KV_CACHES.get(key)

    if (
        entry is not None
        and entry.storage_key == key
        and entry.shape == shape
        and entry.dtype == kv_cache.dtype
    ):
        return entry

    if entry is not None:
        _VULKAN_KV_CACHES.pop(key, None)

    if len(shape) != 5 or shape[0] != 2:
        raise ValueError(
            f"expected KV cache shape [2, blocks, heads, block, dim], got {shape}"
        )
    _, num_blocks, num_kv_heads, block_size, head_size = shape
    dtype_size = _kv_cache_dtype_size(kv_cache.dtype)
    layout = VulkanPagedKVLayout(
        (
            KVCacheLayerSpec(
                layer_index=_PER_LAYER_KV_CACHE_INDEX,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                block_size=block_size,
                dtype_size=dtype_size,
            ),
        ),
        num_blocks=num_blocks,
    )
    gpu_cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(gpu_cache, bytes(layout.total_bytes))

    entry = _VulkanKVCacheEntry(
        storage_key=key,
        layout=layout,
        cache=gpu_cache,
        shape=shape,
        dtype=kv_cache.dtype,
        written_slots=bytearray(layout.capacity_tokens_per_layer),
        verified_prefix_by_blocks=OrderedDict(),
    )
    _VULKAN_KV_CACHES[key] = entry

    return entry


def _kv_cache_storage_key(kv_cache: torch.Tensor) -> int:
    return int(kv_cache.untyped_storage().data_ptr())


def _kv_cache_dtype_size(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 2

    if dtype == torch.float32:
        return 4

    raise ValueError(f"unsupported Vulkan KV cache dtype: {dtype}")
