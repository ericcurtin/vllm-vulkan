# SPDX-License-Identifier: Apache-2.0
"""Paged KV-cache layout contract for the Vulkan backend.

The first Vulkan attention kernels need a stable byte layout before they can
read or write paged KV blocks.  This module defines that contract in one place
and keeps it aligned with vLLM's block-table/slot mapping:

    slot = physical_block_id * block_size + token_offset_in_block

Each layer owns a contiguous region.  Inside a layer, each physical block stores
one K plane followed by one V plane:

    [layer][physical block][K tokens][V tokens]

Within each K/V plane, data is token-major:

    [token_offset_in_block][kv_head][head_element]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

KVPlane = Literal["k", "v"]

_DTYPE_SIZE_BYTES = {
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "float": 4,
    "fp32": 4,
    "float64": 8,
    "double": 8,
    "fp8": 1,
    "float8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}


def dtype_size_bytes(dtype: Any) -> int:
    """Return the byte width used for KV cache dtype names or dtype objects."""
    if isinstance(dtype, int):
        if dtype <= 0:
            raise ValueError("dtype byte width must be positive")
        return dtype

    name = str(dtype).lower()
    if name.startswith("torch."):
        name = name.removeprefix("torch.")

    if name in _DTYPE_SIZE_BYTES:
        return _DTYPE_SIZE_BYTES[name]

    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int) and itemsize > 0:
        return itemsize

    raise ValueError(f"Unsupported KV cache dtype {dtype!r}")


@dataclass(frozen=True)
class KVCacheLayerSpec:
    """Shape of one layer's paged KV-cache blocks."""

    layer_index: int
    num_kv_heads: int
    head_size: int
    block_size: int
    dtype_size: int

    def __post_init__(self) -> None:
        for field_name in (
            "layer_index",
            "num_kv_heads",
            "head_size",
            "block_size",
            "dtype_size",
        ):
            value = getattr(self, field_name)
            if field_name == "layer_index":
                if value < 0:
                    raise ValueError("layer_index must be >= 0")
            elif value <= 0:
                raise ValueError(f"{field_name} must be > 0")

    @property
    def elements_per_token(self) -> int:
        return self.num_kv_heads * self.head_size

    @property
    def bytes_per_token_per_plane(self) -> int:
        return self.elements_per_token * self.dtype_size

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.bytes_per_token_per_plane

    @property
    def plane_bytes_per_block(self) -> int:
        return self.block_size * self.bytes_per_token_per_plane

    @property
    def bytes_per_block(self) -> int:
        return 2 * self.plane_bytes_per_block


@dataclass(frozen=True)
class KVAttentionPlaneStrides:
    """Byte strides an attention shader uses to scan one K or V block."""

    base_offset: int
    token_stride: int
    kv_head_stride: int
    head_element_stride: int


@dataclass(frozen=True)
class VulkanPagedKVLayout:
    """Byte-addressable paged KV-cache layout for Vulkan attention kernels."""

    layer_specs: tuple[KVCacheLayerSpec, ...]
    num_blocks: int

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if not self.layer_specs:
            raise ValueError("at least one layer spec is required")

        ordered = tuple(sorted(self.layer_specs, key=lambda spec: spec.layer_index))
        for expected, spec in enumerate(ordered):
            if spec.layer_index != expected:
                raise ValueError(
                    "layer specs must be contiguous and start at 0; "
                    f"expected layer {expected}, got {spec.layer_index}"
                )

        block_size = ordered[0].block_size
        if any(spec.block_size != block_size for spec in ordered):
            raise ValueError("all layer specs must use the same block_size")

        offsets: list[int] = []
        cursor = 0
        for spec in ordered:
            offsets.append(cursor)
            cursor += self.num_blocks * spec.bytes_per_block

        object.__setattr__(self, "layer_specs", ordered)
        object.__setattr__(self, "_layer_base_offsets", tuple(offsets))
        object.__setattr__(self, "_total_bytes", cursor)

    @property
    def num_layers(self) -> int:
        return len(self.layer_specs)

    @property
    def block_size(self) -> int:
        return self.layer_specs[0].block_size

    @property
    def capacity_tokens_per_layer(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def bytes_per_token(self) -> int:
        return sum(spec.bytes_per_token for spec in self.layer_specs)

    def layer_spec(self, layer_index: int) -> KVCacheLayerSpec:
        try:
            return self.layer_specs[layer_index]
        except IndexError as exc:
            raise ValueError(f"layer_index {layer_index} out of range") from exc

    def layer_base_offset(self, layer_index: int) -> int:
        self.layer_spec(layer_index)
        return self._layer_base_offsets[layer_index]

    def block_base_offset(self, layer_index: int, physical_block_id: int) -> int:
        spec = self.layer_spec(layer_index)
        if not 0 <= physical_block_id < self.num_blocks:
            raise ValueError(
                f"physical_block_id {physical_block_id} out of range "
                f"for {self.num_blocks} block(s)"
            )
        return (
            self.layer_base_offset(layer_index)
            + physical_block_id * spec.bytes_per_block
        )

    def plane_base_offset(
        self, layer_index: int, physical_block_id: int, plane: KVPlane
    ) -> int:
        spec = self.layer_spec(layer_index)
        block_base = self.block_base_offset(layer_index, physical_block_id)
        if plane == "k":
            return block_base
        if plane == "v":
            return block_base + spec.plane_bytes_per_block
        raise ValueError(f"unknown KV plane {plane!r}")

    def token_offset(
        self,
        layer_index: int,
        physical_block_id: int,
        token_offset_in_block: int,
        kv_head: int,
        head_element: int,
        plane: KVPlane,
    ) -> int:
        """Return byte offset for one scalar in the paged KV cache."""
        spec = self.layer_spec(layer_index)
        if not 0 <= token_offset_in_block < spec.block_size:
            raise ValueError(
                f"token_offset_in_block {token_offset_in_block} out of range "
                f"for block_size {spec.block_size}"
            )
        if not 0 <= kv_head < spec.num_kv_heads:
            raise ValueError(
                f"kv_head {kv_head} out of range for {spec.num_kv_heads} head(s)"
            )
        if not 0 <= head_element < spec.head_size:
            raise ValueError(
                f"head_element {head_element} out of range for head_size "
                f"{spec.head_size}"
            )

        scalar_index = (
            token_offset_in_block * spec.num_kv_heads + kv_head
        ) * spec.head_size + head_element
        return (
            self.plane_base_offset(layer_index, physical_block_id, plane)
            + scalar_index * spec.dtype_size
        )

    def slot_offset(
        self,
        layer_index: int,
        slot: int,
        kv_head: int,
        head_element: int,
        plane: KVPlane,
    ) -> int:
        """Return byte offset from vLLM slot mapping output."""
        if slot < 0:
            raise ValueError("slot must be >= 0")
        physical_block_id, token_offset_in_block = divmod(slot, self.block_size)
        return self.token_offset(
            layer_index,
            physical_block_id,
            token_offset_in_block,
            kv_head,
            head_element,
            plane,
        )

    def plane_strides_for_attn_load(
        self,
        layer_index: int,
        physical_block_id: int,
        plane: KVPlane,
    ) -> KVAttentionPlaneStrides:
        """Return byte strides for K/V[token, kv_head, head_element] loads."""
        spec = self.layer_spec(layer_index)
        return KVAttentionPlaneStrides(
            base_offset=self.plane_base_offset(layer_index, physical_block_id, plane),
            token_stride=spec.bytes_per_token_per_plane,
            kv_head_stride=spec.head_size * spec.dtype_size,
            head_element_stride=spec.dtype_size,
        )

    def k_plane_strides_for_attn_load(
        self, layer_index: int, physical_block_id: int
    ) -> KVAttentionPlaneStrides:
        """Return byte strides for attention K loads from one physical block."""
        return self.plane_strides_for_attn_load(layer_index, physical_block_id, "k")

    def v_plane_strides_for_attn_load(
        self, layer_index: int, physical_block_id: int
    ) -> KVAttentionPlaneStrides:
        """Return byte strides for attention V loads from one physical block."""
        return self.plane_strides_for_attn_load(layer_index, physical_block_id, "v")

    @classmethod
    def from_layer_specs(
        cls,
        layer_specs: list[KVCacheLayerSpec] | tuple[KVCacheLayerSpec, ...],
        num_blocks: int,
    ) -> VulkanPagedKVLayout:
        return cls(tuple(layer_specs), num_blocks)


def infer_kv_layer_specs(
    hf_config: Any, block_size: int, dtype: Any
) -> tuple[KVCacheLayerSpec, ...]:
    """Infer per-layer KV-cache block shapes from a Hugging Face config."""
    text_cfg = _cfg_get(hf_config, "text_config") or hf_config
    num_layers = int(_cfg_get(text_cfg, "num_hidden_layers", 1))
    num_attention_heads = int(_cfg_get(text_cfg, "num_attention_heads", 1))
    num_kv_heads = int(_cfg_get(text_cfg, "num_key_value_heads", num_attention_heads))
    hidden_size = _cfg_get(text_cfg, "hidden_size")
    head_size = _cfg_get(text_cfg, "head_dim")
    if head_size is None:
        if hidden_size is None:
            raise ValueError("cannot infer KV head size from model config")
        head_size = int(hidden_size) // num_attention_heads
    else:
        head_size = int(head_size)

    global_head_size = _cfg_get(text_cfg, "global_head_dim")
    global_num_kv_heads = _cfg_get(text_cfg, "num_global_key_value_heads")
    layer_types = _cfg_get(text_cfg, "layer_types")
    dtype_size = dtype_size_bytes(dtype)

    specs: list[KVCacheLayerSpec] = []
    for layer_index in range(num_layers):
        layer_type = ""
        if layer_types is not None and layer_index < len(layer_types):
            layer_type = str(layer_types[layer_index]).lower()

        use_global = (
            "full" in layer_type
            and global_head_size is not None
            and global_num_kv_heads is not None
        )
        specs.append(
            KVCacheLayerSpec(
                layer_index=layer_index,
                num_kv_heads=int(global_num_kv_heads if use_global else num_kv_heads),
                head_size=int(global_head_size if use_global else head_size),
                block_size=block_size,
                dtype_size=dtype_size,
            )
        )
    return tuple(specs)


def layout_from_hf_config(
    hf_config: Any, block_size: int, num_blocks: int, dtype: Any
) -> VulkanPagedKVLayout:
    """Build a paged KV layout directly from model config and block count."""
    return VulkanPagedKVLayout(
        infer_kv_layer_specs(hf_config, block_size=block_size, dtype=dtype),
        num_blocks=num_blocks,
    )


def _cfg_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
