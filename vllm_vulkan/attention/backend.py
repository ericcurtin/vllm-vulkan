"""
VulkanAttentionBackend - Attention implementation for Vulkan backend.

This module provides the attention backend implementation for vLLM.
"""

from dataclasses import dataclass, field
from typing import Any

import vllm_vulkan


@dataclass
class VulkanAttentionMetadata:
    """
    Metadata for Vulkan attention computation.

    This class holds per-batch metadata needed for attention computation,
    including sequence lengths, block tables, and slot mappings.
    """

    # Number of prefill tokens in this batch
    num_prefill_tokens: int = 0

    # Number of decode tokens in this batch
    num_decode_tokens: int = 0

    # Number of prefill sequences
    num_prefills: int = 0

    # Sequence lengths for each sequence
    seq_lens: list[int] = field(default_factory=list)

    # Context lengths for decode
    context_lens: list[int] = field(default_factory=list)

    # Block tables for paged attention
    # Shape: [num_seqs, max_blocks_per_seq]
    block_tables: Any | None = None

    # Slot mapping for cache storage
    # Maps tokens to cache slots
    slot_mapping: Any | None = None

    # Maximum sequence length in this batch
    max_seq_len: int = 0

    # Maximum prefill sequence length
    max_prefill_seq_len: int = 0

    # Maximum decode sequence length
    max_decode_seq_len: int = 0

    # Query start locations (for variable length sequences)
    query_start_loc: Any | None = None

    # Sequence start locations
    seq_start_loc: Any | None = None

    # Use flash attention
    use_flash_attn: bool = True

    @property
    def is_prefill(self) -> bool:
        """Check if this batch contains prefill tokens."""
        return self.num_prefill_tokens > 0

    @property
    def is_decode(self) -> bool:
        """Check if this batch contains decode tokens."""
        return self.num_decode_tokens > 0

    @property
    def total_tokens(self) -> int:
        """Get total number of tokens in this batch."""
        return self.num_prefill_tokens + self.num_decode_tokens


class VulkanAttentionMetadataBuilder:
    """
    Builder for VulkanAttentionMetadata.

    Constructs attention metadata from sequence group metadata.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        block_size: int,
        device_idx: int = 0,
    ):
        """
        Initialize the metadata builder.

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            num_kv_heads: Number of key-value heads (for GQA)
            block_size: KV cache block size
            device_idx: Device index
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.device_idx = device_idx

        # Accumulated data
        self._seq_lens: list[int] = []
        self._context_lens: list[int] = []
        self._block_tables: list[list[int]] = []
        self._slot_mapping: list[int] = []
        self._num_prefill_tokens = 0
        self._num_decode_tokens = 0
        self._num_prefills = 0

    def add_sequence(
        self,
        seq_len: int,
        context_len: int,
        is_prefill: bool,
        block_table: list[int] | None = None,
        slot_mapping: list[int] | None = None,
    ) -> None:
        """
        Add a sequence to the batch.

        Args:
            seq_len: Sequence length
            context_len: Context length
            is_prefill: Whether this is a prefill sequence
            block_table: Block table for this sequence
            slot_mapping: Slot mapping for this sequence
        """
        self._seq_lens.append(seq_len)
        self._context_lens.append(context_len)

        if block_table is not None:
            self._block_tables.append(block_table)

        if slot_mapping is not None:
            self._slot_mapping.extend(slot_mapping)

        if is_prefill:
            self._num_prefill_tokens += seq_len
            self._num_prefills += 1
        else:
            self._num_decode_tokens += 1

    def build(self) -> VulkanAttentionMetadata:
        """
        Build the attention metadata.

        Returns:
            VulkanAttentionMetadata instance
        """
        max_seq_len = max(self._seq_lens) if self._seq_lens else 0
        max_prefill_seq_len = 0
        max_decode_seq_len = 0

        # Calculate max lengths
        prefill_idx = 0
        for seq_len in self._seq_lens:
            if prefill_idx < self._num_prefills:
                max_prefill_seq_len = max(max_prefill_seq_len, seq_len)
                prefill_idx += 1
            else:
                max_decode_seq_len = max(max_decode_seq_len, seq_len)

        return VulkanAttentionMetadata(
            num_prefill_tokens=self._num_prefill_tokens,
            num_decode_tokens=self._num_decode_tokens,
            num_prefills=self._num_prefills,
            seq_lens=self._seq_lens.copy(),
            context_lens=self._context_lens.copy(),
            block_tables=self._block_tables.copy() if self._block_tables else None,
            slot_mapping=self._slot_mapping.copy() if self._slot_mapping else None,
            max_seq_len=max_seq_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
        )

    def reset(self) -> None:
        """Reset the builder for a new batch."""
        self._seq_lens.clear()
        self._context_lens.clear()
        self._block_tables.clear()
        self._slot_mapping.clear()
        self._num_prefill_tokens = 0
        self._num_decode_tokens = 0
        self._num_prefills = 0


class VulkanAttentionImpl:
    """
    Vulkan attention implementation.

    Core attention computation using the Vulkan backend.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
    ):
        """
        Initialize the attention implementation.

        Args:
            num_heads: Number of query heads
            head_dim: Dimension of each head
            num_kv_heads: Number of key-value heads
            scale: Attention scale factor
            alibi_slopes: ALiBi slopes (if using ALiBi)
            sliding_window: Sliding window size (if using sliding window attention)
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window

        # Number of queries per KV head (for GQA)
        self.num_queries_per_kv = num_heads // num_kv_heads

    def forward(
        self,
        query: Any,
        key: Any,
        value: Any,
        kv_cache: tuple[Any, Any] | None,
        attn_metadata: VulkanAttentionMetadata,
    ) -> Any:
        """
        Compute attention.

        Args:
            query: Query tensor [num_tokens, num_heads * head_dim]
            key: Key tensor [num_tokens, num_kv_heads * head_dim]
            value: Value tensor [num_tokens, num_kv_heads * head_dim]
            kv_cache: Tuple of (key_cache, value_cache) or None
            attn_metadata: Attention metadata

        Returns:
            Attention output tensor [num_tokens, num_heads * head_dim]
        """
        # Check if Rust extension is available
        if not vllm_vulkan._RUST_AVAILABLE:
            raise RuntimeError("Vulkan backend not available")

        # Convert to VulkanTensor if needed
        if not isinstance(query, vllm_vulkan.VulkanTensor):
            # Assume numpy array, convert
            query_tensor = vllm_vulkan.VulkanTensor.from_numpy(query)
        else:
            query_tensor = query

        if attn_metadata.is_prefill:
            # Prefill: compute full attention
            return self._prefill_attention(
                query_tensor, key, value, kv_cache, attn_metadata
            )
        else:
            # Decode: use paged attention
            return self._decode_attention(
                query_tensor, kv_cache, attn_metadata
            )

    def _prefill_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        kv_cache: tuple[Any, Any] | None,
        attn_metadata: VulkanAttentionMetadata,
    ) -> Any:
        """Compute prefill attention."""
        # Use flash attention for prefill
        output = vllm_vulkan.flash_attention(
            query=query,
            key=key,
            value=value,
            mask=None,  # Causal mask applied internally
            scale=self.scale,
        )

        # Store KV in cache if provided
        if kv_cache is not None and attn_metadata.slot_mapping is not None:
            key_cache, value_cache = kv_cache
            vllm_vulkan.reshape_and_cache(
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=attn_metadata.slot_mapping,
            )

        return output

    def _decode_attention(
        self,
        query: Any,
        kv_cache: tuple[Any, Any] | None,
        attn_metadata: VulkanAttentionMetadata,
    ) -> Any:
        """Compute decode attention using paged KV cache."""
        if kv_cache is None:
            raise ValueError("KV cache required for decode attention")

        key_cache, value_cache = kv_cache

        # Flatten block tables for paged attention
        if attn_metadata.block_tables is not None:
            block_tables_flat = [
                item
                for sublist in attn_metadata.block_tables
                for item in sublist
            ]
        else:
            block_tables_flat = []

        output = vllm_vulkan.paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables_flat,
            context_lens=attn_metadata.context_lens,
            scale=self.scale,
            block_size=16,  # Default block size
        )

        return output


class VulkanAttentionBackend:
    """
    Vulkan attention backend for vLLM.

    This class provides the interface expected by vLLM for attention backends.
    """

    name: str = "vulkan"

    @staticmethod
    def get_impl_cls() -> type[VulkanAttentionImpl]:
        """Get the attention implementation class."""
        return VulkanAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[VulkanAttentionMetadata]:
        """Get the metadata class."""
        return VulkanAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[VulkanAttentionMetadataBuilder]:
        """Get the metadata builder class."""
        return VulkanAttentionMetadataBuilder

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        """Get supported head dimensions."""
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def is_available() -> bool:
        """Check if the backend is available."""
        return vllm_vulkan.is_available()


class VulkanFlashAttentionBackend(VulkanAttentionBackend):
    """
    Flash attention variant of the Vulkan backend.

    This uses optimized flash attention kernels when available.
    """

    name: str = "vulkan_flash"

    @staticmethod
    def is_available() -> bool:
        """Check if flash attention is available."""
        if not vllm_vulkan.is_available():
            return False

        # Check for device capabilities
        try:
            device_info = vllm_vulkan.get_device_info(0)
            return device_info.get("supports_fp16", False)
        except Exception:
            return False
