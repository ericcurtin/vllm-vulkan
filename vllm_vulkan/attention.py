# SPDX-License-Identifier: Apache-2.0
"""vLLM attention backend shim for Vulkan decode bring-up.

This backend intentionally keeps vLLM's CPU attention metadata and CPU KV cache
layout as the source of truth, then mirrors supported decode updates into the
Vulkan paged KV cache. Unsupported cases fall back to CPU_ATTN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    written_slots: set[int]


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
            if not self._try_vulkan_decode(
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

            layout, gpu_cache = _get_or_create_vulkan_kv_cache(ctx, kv_cache)
            if kv_cache.dtype == torch.float16:
                from vllm_vulkan.kv_ops import (  # noqa: PLC0415
                    paged_attn_decode_f16,
                )

                decode = paged_attn_decode_f16
            else:
                from vllm_vulkan.kv_ops import (  # noqa: PLC0415
                    paged_attn_decode_f32,
                )

                decode = paged_attn_decode_f32

            seq_lens = attn_metadata.seq_lens[:num_actual_tokens].to("cpu")
            block_table = attn_metadata.block_table[:num_actual_tokens].to("cpu")
            if not _vulkan_cache_has_sequences(layout, kv_cache, block_table, seq_lens):
                return False

            outs = []
            for token_idx in range(num_actual_tokens):
                outs.append(
                    decode(
                        ctx,
                        layout,
                        gpu_cache,
                        0,
                        query[token_idx].detach().to("cpu"),
                        block_table[token_idx],
                        int(seq_lens[token_idx]),
                        self.scale,
                    )
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
        if attn_metadata.use_sdpa_prefill or not attn_metadata.causal:
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

        layout, gpu_cache = _get_or_create_vulkan_kv_cache(ctx, kv_cache)
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
            layout,
            gpu_cache,
            0,
            key[:num_tokens],
            value[:num_tokens],
            slots,
        )
        _vulkan_cache_written_slots(kv_cache).update(int(slot) for slot in slots)
    except Exception as exc:
        logger.debug("Vulkan KV cache mirror skipped: %s", exc)


def _vulkan_cache_has_sequences(
    layout: VulkanPagedKVLayout,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> bool:
    written_slots = _vulkan_cache_written_slots(kv_cache)
    spec = layout.layer_spec(0)
    for req_idx, seq_len_tensor in enumerate(seq_lens):
        seq_len = int(seq_len_tensor)
        row = block_table[req_idx]
        for token_pos in range(seq_len):
            logical_block_id = token_pos // spec.block_size
            if logical_block_id >= row.numel():
                return False
            token_offset = token_pos % spec.block_size
            physical_block_id = int(row[logical_block_id])
            if not 0 <= physical_block_id < layout.num_blocks:
                return False
            slot = physical_block_id * spec.block_size + token_offset
            if slot not in written_slots:
                return False
    return True


def _vulkan_cache_written_slots(kv_cache: torch.Tensor) -> set[int]:
    entry = _VULKAN_KV_CACHES.get(_kv_cache_storage_key(kv_cache))
    if entry is None:
        return set()
    return entry.written_slots


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
) -> tuple[VulkanPagedKVLayout, GpuTensor]:
    key = _kv_cache_storage_key(kv_cache)
    shape = tuple(int(dim) for dim in kv_cache.shape)
    entry = _VULKAN_KV_CACHES.get(key)

    if (
        entry is not None
        and entry.storage_key == key
        and entry.shape == shape
        and entry.dtype == kv_cache.dtype
    ):
        return entry.layout, entry.cache

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
                layer_index=0,
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

    _VULKAN_KV_CACHES[key] = _VulkanKVCacheEntry(
        storage_key=key,
        layout=layout,
        cache=gpu_cache,
        shape=shape,
        dtype=kv_cache.dtype,
        written_slots=set(),
    )

    return layout, gpu_cache


def _kv_cache_storage_key(kv_cache: torch.Tensor) -> int:
    return int(kv_cache.untyped_storage().data_ptr())


def _kv_cache_dtype_size(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 2

    if dtype == torch.float32:
        return 4

    raise ValueError(f"unsupported Vulkan KV cache dtype: {dtype}")
