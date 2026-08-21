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
    paged_kv_write_and_decode_batch_f16,
    paged_kv_write_and_decode_batch_f32,
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


def test_paged_kv_write_and_decode_batch_empty_returns_empty_list_without_writing():
    """An empty decode batch must be a legitimate no-op (matching
    paged_attn_decode_batch_f32's own empty-batch handling above), not an
    error - calling _build_paged_kv_write_op with zero tokens would raise
    ValueError("K/V must contain at least one token") otherwise. Also
    passes empty (0-token) k/v/slot_mapping tensors, since that's the
    shape attention.py's _try_write_and_decode_vulkan would actually
    build when num_actual_tokens == 0.
    """
    ctx = _require_vulkan_context()
    _require_shader(ctx, "paged_kv_write_f32")
    _require_shader(ctx, "paged_attn_decode_f32", "paged_attn_decode_f32_coop")

    spec, layout = _make_layout(dtype_size=4)
    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))

    empty_k = torch.zeros((0, spec.num_kv_heads, spec.head_size), dtype=torch.float32)
    empty_v = torch.zeros_like(empty_k)
    empty_slot_mapping = torch.zeros((0,), dtype=torch.int64)

    assert (
        paged_kv_write_and_decode_batch_f32(
            ctx,
            layout,
            cache,
            0,
            empty_k,
            empty_v,
            empty_slot_mapping,
            queries=[],
            block_tables=[],
            seq_lens=[],
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


@pytest.mark.slow  # wall-clock comparison; unreliable on shared CI runners
def test_cached_available_shaders_matches_uncached_reference_and_is_faster():
    """`kv_ops._cached_available_shaders` must return the exact same set
    of shader names as calling `ctx.available_shaders()` directly, just
    computed once per distinct context object instead of on every call
    (`_select_decode_shader`/`_require_shader` both use it now) — see its
    own doc comment in kv_ops.py, and
    `vulkan_ops._cached_available_shaders` (same fix, applied first to
    `linear()`'s much hotter call site) for the measurement this is based
    on.

    Also measures the actual speedup here directly, taking the minimum
    elapsed time across several independent trials rather than a single
    timed loop — see vulkan_ops.py's `TestLinearMatvecDispatchThreshold`
    for why (measurement noise under full-suite load).
    """
    import time

    from vllm_vulkan import kv_ops

    ctx = _require_vulkan_context()

    cached = kv_ops._cached_available_shaders(ctx)
    assert cached == frozenset(ctx.available_shaders())
    # Must actually be cached, not recomputed on every call: repeated
    # calls return the identical (not just equal) frozenset object.
    assert kv_ops._cached_available_shaders(ctx) is cached

    iters = 2000
    trials = 5

    def timed(fn) -> float:
        best = float("inf")
        for _ in range(trials):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            best = min(best, time.perf_counter() - t0)
        return best

    uncached_elapsed = timed(ctx.available_shaders)
    cached_elapsed = timed(lambda: kv_ops._cached_available_shaders(ctx))

    print(
        f"\nkv_ops available_shaders(), best-of-{trials}: "
        f"uncached {uncached_elapsed / iters * 1e6:.3f}us/call, "
        f"cached {cached_elapsed / iters * 1e6:.3f}us/call, "
        f"speedup {uncached_elapsed / cached_elapsed:.1f}x"
    )
    assert cached_elapsed < uncached_elapsed, (
        f"cached available_shaders ({cached_elapsed:.4f}s/{iters}) was not "
        f"faster than the uncached reference ({uncached_elapsed:.4f}s/{iters})"
    )


def test_cached_available_shaders_uses_identity_not_equality():
    """Guards against a regression back to a naive `id(ctx)`-integer (or
    `==`-based) cache key: two distinct objects that happen to compare
    `==` to each other (as two garbage-collected-then-reallocated
    objects at the same memory address effectively would under a bare
    `id()` comparison) must still be treated as different contexts —
    only a strict `is` identity check on a live, held reference gets
    this right in every case. Pure Python; doesn't need a real Vulkan
    device (`_cached_available_shaders` only ever calls
    `ctx.available_shaders()`, so a minimal duck-typed stand-in works).
    """
    from vllm_vulkan import kv_ops

    class _FakeCtx:
        def __init__(self, shaders: list[str]) -> None:
            self._shaders = shaders

        def available_shaders(self) -> list[str]:
            return list(self._shaders)

        def __eq__(self, other: object) -> bool:
            return True  # deliberately "equal" to any other _FakeCtx

        def __hash__(self) -> int:
            return 0

    prev_ctx = kv_ops._available_shaders_cache_ctx
    prev_cache = kv_ops._available_shaders_cache
    try:
        ctx_a = _FakeCtx(["shader_a"])
        ctx_b = _FakeCtx(["shader_b"])

        result_a = kv_ops._cached_available_shaders(ctx_a)  # type: ignore[arg-type]
        assert result_a == frozenset({"shader_a"})

        result_b = kv_ops._cached_available_shaders(ctx_b)  # type: ignore[arg-type]
        assert result_b == frozenset({"shader_b"}), (
            "a distinct (but == equal) context object must not incorrectly "
            "reuse the previous context's cached shader set"
        )
    finally:
        kv_ops._available_shaders_cache_ctx = prev_ctx
        kv_ops._available_shaders_cache = prev_cache


def test_cached_available_shaders_recomputes_for_a_different_context():
    """A second, distinct `VulkanContext` object must not silently reuse
    the first context's cached shader set — the single-slot cache
    compares against a stored strong reference to the cached-for context
    with `is` specifically to catch this (see
    `kv_ops._cached_available_shaders`'s doc comment for why a bare
    `id(ctx)` integer comparison would be unsafe here)."""
    from vllm_vulkan import kv_ops

    ctx1 = _require_vulkan_context()
    ctx2 = _require_vulkan_context()

    shaders1 = kv_ops._cached_available_shaders(ctx1)
    shaders2 = kv_ops._cached_available_shaders(ctx2)
    # Different context objects (even from the same device/build) must
    # each get their own freshly-computed (if equal-valued) result, not
    # silently reuse a cache entry meant for a different context object.
    assert shaders1 == shaders2
    # Re-querying ctx1 after ctx2 was cached must recompute for ctx1
    # again (single-slot cache, not a full dict) rather than incorrectly
    # returning ctx2's cached value.
    assert kv_ops._cached_available_shaders(ctx1) == shaders1


def _old_paged_kv_write_pc_reference(
    layout: VulkanPagedKVLayout, layer_index: int, num_tokens: int
) -> bytes:
    """Reconstructs the OLD `_paged_kv_write_pc` (before this change) for
    direct comparison — re-derives `spec`/`layer_base_offset` from
    `layout`/`layer_index` from scratch on every call, exactly as the
    function itself used to."""
    import struct

    spec = layout.layer_spec(layer_index)
    return struct.pack(
        "<7I",
        num_tokens,
        spec.num_kv_heads,
        spec.head_size,
        spec.block_size,
        layout.layer_base_offset(layer_index) // spec.dtype_size,
        spec.plane_bytes_per_block // spec.dtype_size,
        spec.bytes_per_block // spec.dtype_size,
    )


def _old_paged_attn_decode_pc_reference(
    layout: VulkanPagedKVLayout,
    layer_index: int,
    seq_len: int,
    num_q_heads: int,
    scale: float,
) -> bytes:
    """Reconstructs the OLD `_paged_attn_decode_pc` (before this change),
    same rationale as `_old_paged_kv_write_pc_reference` above."""
    import struct

    spec = layout.layer_spec(layer_index)
    return struct.pack(
        "<8If",
        seq_len,
        num_q_heads,
        spec.num_kv_heads,
        spec.head_size,
        spec.block_size,
        layout.layer_base_offset(layer_index) // spec.dtype_size,
        spec.plane_bytes_per_block // spec.dtype_size,
        spec.bytes_per_block // spec.dtype_size,
        scale,
    )


def test_paged_kv_write_pc_matches_old_layout_based_reference():
    """`_paged_kv_write_pc` now takes the already-resolved `spec`/
    `layer_base_offset` directly instead of re-deriving them from
    `layout`/`layer_index` internally — must still produce byte-identical
    push constants to the old (slower, redundant) implementation. Pure
    Python; `VulkanPagedKVLayout`/`KVCacheLayerSpec` are plain frozen
    dataclasses, no Vulkan device needed.
    """
    from vllm_vulkan import kv_ops

    spec, layout = _make_layout(dtype_size=4)
    layer_base_offset = layout.layer_base_offset(spec.layer_index)

    new_pc = kv_ops._paged_kv_write_pc(spec, layer_base_offset, num_tokens=3)
    old_pc = _old_paged_kv_write_pc_reference(layout, spec.layer_index, num_tokens=3)
    assert new_pc == old_pc


def test_paged_attn_decode_pc_matches_old_layout_based_reference():
    """Same check as `test_paged_kv_write_pc_matches_old_layout_based_reference`,
    for `_paged_attn_decode_pc`."""
    from vllm_vulkan import kv_ops

    spec, layout = _make_layout(dtype_size=4)
    layer_base_offset = layout.layer_base_offset(spec.layer_index)

    new_pc = kv_ops._paged_attn_decode_pc(
        spec, layer_base_offset, seq_len=12, num_q_heads=4, scale=0.125
    )
    old_pc = _old_paged_attn_decode_pc_reference(
        layout, spec.layer_index, seq_len=12, num_q_heads=4, scale=0.125
    )
    assert new_pc == old_pc


@pytest.mark.slow  # wall-clock comparison; unreliable on shared CI runners
def test_paged_attn_decode_pc_resolved_spec_is_faster_than_relayout_lookup():
    """Measures the actual speedup: passing an already-resolved `spec`/
    `layer_base_offset` directly should be faster than re-deriving them
    from `layout`/`layer_index` on every call (the old behavior) — the
    exact class of redundant-lookup overhead `_resolve_paged_attn_decode_dispatch`
    resolving both once per *batch* (not once per token) now fully
    eliminates from this function specifically.

    Takes the minimum elapsed time across several independent trials —
    see `test_cached_available_shaders_matches_uncached_reference_and_is_faster`
    (same file) for why (measurement noise under full-suite load).
    """
    import time

    from vllm_vulkan import kv_ops

    spec, layout = _make_layout(dtype_size=4)
    layer_base_offset = layout.layer_base_offset(spec.layer_index)

    iters = 5000
    trials = 5

    def timed(fn) -> float:
        best = float("inf")
        for _ in range(trials):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            best = min(best, time.perf_counter() - t0)
        return best

    old_elapsed = timed(
        lambda: _old_paged_attn_decode_pc_reference(
            layout, spec.layer_index, seq_len=12, num_q_heads=4, scale=0.125
        )
    )
    new_elapsed = timed(
        lambda: kv_ops._paged_attn_decode_pc(
            spec, layer_base_offset, seq_len=12, num_q_heads=4, scale=0.125
        )
    )

    print(
        f"\n_paged_attn_decode_pc, best-of-{trials}: "
        f"old (layout+layer_index) {old_elapsed / iters * 1e6:.3f}us/call, "
        f"new (resolved spec) {new_elapsed / iters * 1e6:.3f}us/call, "
        f"speedup {old_elapsed / new_elapsed:.2f}x"
    )
    assert new_elapsed < old_elapsed, (
        f"resolved-spec _paged_attn_decode_pc ({new_elapsed:.4f}s/{iters}) was "
        f"not faster than the old layout+layer_index reference "
        f"({old_elapsed:.4f}s/{iters})"
    )


def test_block_table_to_u32_matches_reference_for_int32_and_int64_input():
    """`_block_table_to_u32` must produce the same result whether given
    an int32 block-table row (vLLM's own on-device dtype -- see
    `vllm/v1/worker/block_table.py`) or an already-int64 one (what
    `attention._cached_decode_support_data` now hands it, having
    normalized the whole 2-D table once before splitting into per-row
    views -- see that function's own doc comment). Pure Python/numpy;
    no Vulkan device needed.
    """
    from vllm_vulkan import kv_ops

    row_int32 = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    row_int64 = row_int32.to(torch.int64)

    result_int32 = kv_ops._block_table_to_u32(row_int32, needed_blocks=3, num_blocks=8)
    result_int64 = kv_ops._block_table_to_u32(row_int64, needed_blocks=3, num_blocks=8)

    np.testing.assert_array_equal(result_int32, result_int64)
    np.testing.assert_array_equal(result_int32, np.array([2, 0, 3], dtype=np.uint32))


def test_block_table_to_u32_skips_dtype_conversion_for_already_int64_input(
    monkeypatch,
):
    """When given an already-int64, already-CPU, already-contiguous
    tensor, `_block_table_to_u32`'s internal `.to(device="cpu",
    dtype=torch.int64)` call must be a genuine no-op (return `self`,
    no new tensor/copy) -- confirmed by checking the returned tensor's
    `data_ptr()` is unchanged, not just its dtype.
    """
    from vllm_vulkan import kv_ops

    row = torch.tensor([2, 0, 3, 1], dtype=torch.int64).contiguous()
    detached = row.detach()
    original_data_ptr = detached.data_ptr()

    converted = detached.to(device="cpu", dtype=torch.int64)
    assert converted.data_ptr() == original_data_ptr, (
        "Tensor.to() with a matching device/dtype should be a true no-op "
        "(same underlying storage), not a fresh copy"
    )

    # Sanity: _block_table_to_u32 itself still produces correct output
    # end-to-end for this already-normalized input.
    result = kv_ops._block_table_to_u32(row, needed_blocks=3, num_blocks=8)
    np.testing.assert_array_equal(result, np.array([2, 0, 3], dtype=np.uint32))


@pytest.mark.slow  # wall-clock comparison; unreliable on shared CI runners
def test_block_table_whole_batch_int64_conversion_is_faster_than_per_row():
    """Measures the actual speedup: converting a whole (B, blocks_per_row)
    int32 block table to int64 once, before splitting into per-row
    views, is faster than converting each row independently after
    splitting -- the exact pattern `attention._cached_decode_support_data`
    now uses (converting the whole table once, shared across every
    attention layer using the same `attn_metadata`) versus what
    `_paged_attn_decode_batch`'s per-token loop used to force via
    `_block_table_to_u32`'s own internal conversion on each
    already-split row.

    Takes the minimum elapsed time across several independent trials --
    see `test_paged_attn_decode_pc_resolved_spec_is_faster_than_relayout_lookup`
    (same file) for why (measurement noise under full-suite load).
    """
    import time

    from vllm_vulkan import kv_ops

    batch_size = 16
    blocks_per_row = 64
    table_int32 = torch.randint(0, 100, (batch_size, blocks_per_row), dtype=torch.int32)

    iters = 300
    trials = 5

    def timed(fn) -> float:
        best = float("inf")
        for _ in range(trials):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            best = min(best, time.perf_counter() - t0)
        return best

    def per_row_convert() -> None:
        for row in table_int32.unbind(0):
            kv_ops._block_table_to_u32(
                row, needed_blocks=blocks_per_row, num_blocks=100
            )

    def whole_batch_then_split() -> None:
        table_int64 = table_int32.to(device="cpu", dtype=torch.int64).contiguous()
        for row in table_int64.unbind(0):
            kv_ops._block_table_to_u32(
                row, needed_blocks=blocks_per_row, num_blocks=100
            )

    per_row_elapsed = timed(per_row_convert)
    whole_batch_elapsed = timed(whole_batch_then_split)

    print(
        f"\nblock_table dtype conversion (B={batch_size}), best-of-{trials}: "
        f"per-row {per_row_elapsed / iters * 1e6:.1f}us/batch, "
        f"whole-batch-then-split {whole_batch_elapsed / iters * 1e6:.1f}us/batch, "
        f"speedup {per_row_elapsed / whole_batch_elapsed:.2f}x"
    )
    assert whole_batch_elapsed < per_row_elapsed, (
        f"whole-batch conversion ({whole_batch_elapsed:.4f}s/{iters}) was not "
        f"faster than per-row conversion ({per_row_elapsed:.4f}s/{iters})"
    )


@pytest.mark.parametrize(
    (
        "torch_dtype",
        "dtype_size",
        "write_fn",
        "decode_batch_fn",
        "write_and_decode_batch_fn",
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
            paged_attn_decode_batch_f32,
            paged_kv_write_and_decode_batch_f32,
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
            paged_attn_decode_batch_f16,
            paged_kv_write_and_decode_batch_f16,
            "paged_kv_write_f16",
            ("paged_attn_decode_f16", "paged_attn_decode_f16_coop"),
            1,
            5e-3,
            5e-3,
        ),
    ],
)
def test_paged_kv_write_and_decode_batch_matches_separate_calls_and_torch_reference(
    torch_dtype: torch.dtype,
    dtype_size: int,
    write_fn: Callable,
    decode_batch_fn: Callable,
    write_and_decode_batch_fn: Callable,
    write_shader: str,
    decode_shaders: tuple[str, ...],
    seed: int,
    rtol: float,
    atol: float,
):
    """paged_kv_write_and_decode_batch_{f16,f32} - one ctx.execute_batch
    call (one vkQueueSubmit, one fence wait) that writes this step's new
    K/V tokens AND decodes the whole batch against them - must produce
    exactly the same results as the old two-call sequence attention.py
    used before this: paged_kv_write_{f16,f32} (writing the new tokens)
    followed by paged_attn_decode_batch_{f16,f32} (a separate
    ctx.execute_batch call).

    Mirrors a realistic incremental decode step: each sequence already
    has some prior history written to the cache (from earlier steps, via
    the plain per-step write path), and this step contributes exactly one
    new token per sequence, which must be visible to the decode read that
    immediately follows it in the same submit.
    """
    ctx = _require_vulkan_context()
    _require_shader(ctx, write_shader)
    _require_shader(ctx, *decode_shaders)

    spec, layout = _make_layout(dtype_size)

    torch.manual_seed(seed)

    # Two independent "requests", each with some already-written history
    # (seq_len - 1 tokens) plus one new token this step (at seq_len - 1),
    # matching real incremental decode.
    seq_lens = [6, 3]
    block_tables = [
        torch.tensor([2, 0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    ]
    scale = spec.head_size**-0.5

    ks, vs, qs, slot_mappings, new_slots = [], [], [], [], []
    for seq_len, block_table in zip(seq_lens, block_tables, strict=True):
        k = torch.randn(seq_len, spec.num_kv_heads, spec.head_size, dtype=torch_dtype)
        v = torch.randn_like(k)
        q = torch.randn(4, spec.head_size, dtype=torch.float32)
        slot_mapping = _slot_mapping_from_block_table(
            block_table, seq_len, spec.block_size
        )
        ks.append(k)
        vs.append(v)
        qs.append(q)
        slot_mappings.append(slot_mapping)
        new_slots.append(slot_mapping[-1:])

    def populate_history(cache) -> None:
        """Write every token except the last (this step's new one) for
        both sequences, establishing prior decode-step history."""
        for k, v, slot_mapping in zip(ks, vs, slot_mappings, strict=True):
            write_fn(ctx, layout, cache, 0, k[:-1], v[:-1], slot_mapping[:-1])

    # This step's new tokens: one per sequence, combined the way vLLM's
    # own per-step write batches every sequence's new token in one call.
    new_k = torch.stack([k[-1] for k in ks])
    new_v = torch.stack([v[-1] for v in vs])
    new_slot_mapping = torch.cat(new_slots)

    # Reference: old behaviour, two separate ctx.execute_batch calls.
    ref_cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(ref_cache, bytes(layout.total_bytes))
    populate_history(ref_cache)
    write_fn(ctx, layout, ref_cache, 0, new_k, new_v, new_slot_mapping)
    ref_outs = decode_batch_fn(
        ctx, layout, ref_cache, 0, qs, block_tables, seq_lens, scale
    )

    # Under test: one merged ctx.execute_batch call, against a fresh cache
    # with the identical pre-populated history.
    merged_cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(merged_cache, bytes(layout.total_bytes))
    populate_history(merged_cache)
    merged_outs = write_and_decode_batch_fn(
        ctx,
        layout,
        merged_cache,
        0,
        new_k,
        new_v,
        new_slot_mapping,
        qs,
        block_tables,
        seq_lens,
        scale,
    )

    assert len(merged_outs) == len(ref_outs)
    for i, (merged_out, ref_out) in enumerate(zip(merged_outs, ref_outs, strict=True)):
        torch.testing.assert_close(
            merged_out,
            ref_out,
            rtol=rtol,
            atol=atol,
            msg=f"merged write+decode output {i} diverged from the separate-call reference",
        )

    # Cross-check directly against the pure-PyTorch attention reference too.
    for i in range(len(seq_lens)):
        expected = _attention_reference(qs[i], ks[i], vs[i], scale)
        torch.testing.assert_close(merged_outs[i], expected, rtol=rtol, atol=atol)


def test_paged_kv_write_and_decode_batch_is_faster_than_two_separate_submits():
    """Measures the actual round-trip savings: one ctx.execute_batch call
    doing both the write and the decode batch (paged_kv_write_and_decode_
    batch_f32) must be faster wall-clock than the old sequence of two
    separate ctx.execute_batch calls (paged_kv_write_f32 then
    paged_attn_decode_batch_f32) it replaces in attention.py -- each
    ctx.execute_batch call pays a real vkQueueSubmit + fence-wait round
    trip that, measured directly on real hardware, costs several hundred
    microseconds by itself (see src/lib.rs's buffer_pool_capacity_tests
    module and this session's other execute_batch-phase measurements) --
    dwarfing the actual compute work for these small shapes, so removing
    one whole round trip should be clearly measurable even at this
    microbenchmark's small batch size.

    Uses interleaved, alternating-order timing (one call of each variant
    per repetition, flipping which one goes first every other repetition)
    rather than this file's usual "run N iterations of A, then N of B,
    keep the faster block" pattern (see e.g.
    test_block_table_whole_batch_int64_conversion_is_faster_than_per_row).
    That sequential-block pattern was tried first here and gave an
    inverted, misleading result: on this GPU (Apple M1 via Mesa's
    Honeykrisp Vulkan driver), whichever variant's block of N back-to-
    back iterations ran INSIDE a tight loop reached a warmer/higher clock
    state than a variant tested in a separate block afterward - a
    systematic, directional bias (not the kind of symmetric noise
    "keep the minimum of several trials" is designed to filter), and it
    happened to favor whichever function was measured with more frequent,
    smaller submits. Alternating A/B on every repetition means both
    variants experience the same sequence of GPU clock states, so the
    comparison isolates the actual per-call cost difference instead of
    which one got the warmer GPU.
    """
    import time

    ctx = _require_vulkan_context()
    _require_shader(ctx, "paged_kv_write_f32")
    _require_shader(ctx, "paged_attn_decode_f32", "paged_attn_decode_f32_coop")

    spec, layout = _make_layout(dtype_size=4)

    torch.manual_seed(0)
    seq_lens = [6, 3]
    block_tables = [
        torch.tensor([2, 0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    ]
    scale = spec.head_size**-0.5

    ks, vs, qs, slot_mappings = [], [], [], []
    for seq_len, block_table in zip(seq_lens, block_tables, strict=True):
        k = torch.randn(seq_len, spec.num_kv_heads, spec.head_size, dtype=torch.float32)
        v = torch.randn_like(k)
        q = torch.randn(4, spec.head_size, dtype=torch.float32)
        slot_mapping = _slot_mapping_from_block_table(
            block_table, seq_len, spec.block_size
        )
        ks.append(k)
        vs.append(v)
        qs.append(q)
        slot_mappings.append(slot_mapping)

    new_k = torch.stack([k[-1] for k in ks])
    new_v = torch.stack([v[-1] for v in vs])
    new_slot_mapping = torch.cat([s[-1:] for s in slot_mappings])

    cache = ctx.alloc_activation(layout.total_bytes)
    ctx.update_activation(cache, bytes(layout.total_bytes))
    for k, v, slot_mapping in zip(ks, vs, slot_mappings, strict=True):
        paged_kv_write_f32(ctx, layout, cache, 0, k[:-1], v[:-1], slot_mapping[:-1])

    def two_separate_submits() -> None:
        paged_kv_write_f32(ctx, layout, cache, 0, new_k, new_v, new_slot_mapping)
        paged_attn_decode_batch_f32(
            ctx, layout, cache, 0, qs, block_tables, seq_lens, scale
        )

    def one_merged_submit() -> None:
        paged_kv_write_and_decode_batch_f32(
            ctx,
            layout,
            cache,
            0,
            new_k,
            new_v,
            new_slot_mapping,
            qs,
            block_tables,
            seq_lens,
            scale,
        )

    # Warm up both paths together before measuring.
    for _ in range(150):
        two_separate_submits()
        one_merged_submit()

    reps = 400
    two_submits_total = 0.0
    merged_total = 0.0
    for i in range(reps):
        if i % 2 == 0:
            t0 = time.perf_counter()
            two_separate_submits()
            two_submits_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            one_merged_submit()
            merged_total += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            one_merged_submit()
            merged_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            two_separate_submits()
            two_submits_total += time.perf_counter() - t0

    two_submits_mean = two_submits_total / reps
    merged_mean = merged_total / reps

    print(
        f"\npaged kv-write + decode, mean of {reps} alternating-order reps: "
        f"two separate execute_batch calls {two_submits_mean * 1e6:.1f}us/step, "
        f"one merged execute_batch call {merged_mean * 1e6:.1f}us/step, "
        f"speedup {two_submits_mean / merged_mean:.2f}x"
    )
    assert merged_mean < two_submits_mean, (
        f"merged write+decode call ({merged_mean * 1e6:.1f}us/step) was not "
        f"faster than two separate execute_batch calls "
        f"({two_submits_mean * 1e6:.1f}us/step)"
    )
