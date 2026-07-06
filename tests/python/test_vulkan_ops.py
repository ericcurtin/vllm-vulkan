# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm_vulkan.vulkan_ops's persistent GPU weight cache.

_WeightCache (vulkan_ops.py) keys cached GPU weight uploads on
`id(weight.untyped_storage())`, on the assumption that the same nn.Module's
`.weight` Parameter object (same storage) is passed in on every forward
call, so the (potentially large) weight matrix only needs to be uploaded to
the GPU once, not on every single token.

That assumption is silently broken if a caller converts the weight tensor
(e.g. via `.float()`) *before* passing it into vulkan_ops — `Tensor.float()`
allocates a brand-new tensor with brand-new storage whenever the source
isn't already float32 (true for any bf16/fp16-loaded model, the common
case), so the cache key is different on every call and every call re-does
the full host->device upload. See model_runner.py's `_wrap_linear` (fixed
alongside these tests) for the real-world case this reproduces.

These tests count actual uploads via `_WeightCache.put` rather than
inspecting `len(_weight_cache)` at the end: the cache is weak-ref keyed on
`id(storage)`, and CPython can reuse a freed tensor's memory address for a
later allocation, so a *fresh* upload every call can still coincidentally
leave the dict at a stable size — a real (if rare) source of test flakiness
that counting actual `put()` calls avoids entirely.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
_rs = pytest.importorskip("vllm_vulkan._rs", exc_type=ImportError)

from vllm_vulkan import vulkan_ops  # noqa: E402


def _require_vulkan_context():
    if not _rs.is_available():
        pytest.skip("no Vulkan device available")
    try:
        ctx = _rs.VulkanContext(0)
    except RuntimeError as exc:
        pytest.skip(f"VulkanContext unavailable: {exc}")
    if "mul_mat_vec_f32_f32_f32" not in ctx.available_shaders():
        pytest.skip("mul_mat_vec_f32_f32_f32 shader unavailable")
    return ctx


@pytest.fixture
def vulkan_ctx():
    ctx = _require_vulkan_context()
    prev_ctx = vulkan_ops._ctx
    vulkan_ops.set_context(ctx)
    if vulkan_ops._ctx is None:
        pytest.skip("Vulkan device is a software renderer; GPU dispatch disabled")
    prev_cache = vulkan_ops._weight_cache
    prev_cpu_cache = vulkan_ops._cpu_float32_cache
    vulkan_ops._weight_cache = vulkan_ops._WeightCache()
    vulkan_ops._cpu_float32_cache = vulkan_ops._WeightCache()
    try:
        yield ctx
    finally:
        vulkan_ops._ctx = prev_ctx
        vulkan_ops._weight_cache = prev_cache
        vulkan_ops._cpu_float32_cache = prev_cpu_cache


@pytest.fixture
def upload_counter(monkeypatch):
    """Count real GPU uploads (_WeightCache.put calls) deterministically,
    independent of the weak-ref cache's final size (see module docstring
    for why the latter can be misleading)."""
    counts = {"n": 0}
    orig_put = vulkan_ops._WeightCache.put

    def counting_put(self, w, gpu):
        counts["n"] += 1
        return orig_put(self, w, gpu)

    monkeypatch.setattr(vulkan_ops._WeightCache, "put", counting_put)
    return counts


def test_linear_matches_torch_reference_at_realistic_hidden_size(vulkan_ctx):
    """Direct numerical correctness check for vulkan_ops.linear()'s GPU
    decode path (T < _MATVEC_THRESHOLD) against `torch.nn.functional.linear`
    as ground truth, at a realistic transformer hidden size (1536, matching
    Gemma4-E2B) -- not just the tiny in_features=8 shapes the other tests in
    this file use to exercise the weight cache.

    This specific gap (no test comparing the GPU decode-matvec path's
    OUTPUT against a real reference at a realistic K) is what let a real
    correctness bug in mul_mat_vec_f32_f32_f32_subgroup go unnoticed: that
    shader silently diverged from the correct result for any K>=~256 while
    happening to still be correct at the small K (8) every other test here
    uses -- see subgroup_matvec_correctness_tests in src/lib.rs for the
    measured divergence, and this file's `_require_vulkan_context` /
    vulkan_ops.py's `linear`/`_vulkan_matvec` for the fix (now dispatching
    mul_mat_vec_f32_f32_f32, the plain non-subgroup variant, instead).
    """
    out_features, in_features = 512, 1536
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    x = torch.randn(1, in_features, dtype=torch.float32)  # T=1 < _MATVEC_THRESHOLD

    result = vulkan_ops.linear(x, weight, None)
    expected = torch.nn.functional.linear(x, weight, None)
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


def test_get_or_upload_weight_hits_cache_for_same_tensor_object(
    vulkan_ctx, upload_counter
):
    """Baseline sanity check: calling _get_or_upload_weight twice with the
    exact same tensor object must only upload once."""
    weight = torch.randn(8, 4, dtype=torch.float32)

    gpu1 = vulkan_ops._get_or_upload_weight(vulkan_ctx, weight)
    assert gpu1 is not None
    assert upload_counter["n"] == 1

    gpu2 = vulkan_ops._get_or_upload_weight(vulkan_ctx, weight)
    assert gpu2 is gpu1, "second call with the same tensor object should hit the cache"
    assert upload_counter["n"] == 1, "cache hit must not trigger another upload"


def test_linear_reuses_weight_cache_across_calls_with_bf16_weight(
    vulkan_ctx, upload_counter
):
    """The actual bug: a bf16 weight (the common case for LLM inference)
    passed into vulkan_ops.linear() repeatedly - as model_runner.py's
    persistent nn.Module.weight Parameter would be, across every decode
    step - must hit the GPU weight cache on every call after the first,
    rather than re-uploading the whole matrix every time.

    This only passes if vulkan_ops.linear()/_vulkan_matvec() never
    convert `weight` to float32 before it reaches _get_or_upload_weight
    (the conversion must happen lazily, only on a cache miss).
    """
    out_features, in_features = 16, 8
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    x = torch.randn(1, in_features, dtype=torch.float32)  # T=1 < _MATVEC_THRESHOLD

    vulkan_ops.linear(x, weight, None)
    assert upload_counter["n"] == 1, "first call should upload once"

    for _ in range(5):
        vulkan_ops.linear(x, weight, None)
    assert upload_counter["n"] == 1, (
        "repeated calls with the same weight object must hit the cache, "
        "not re-upload the weight matrix on every call"
    )


def test_wrap_linear_reuses_weight_cache_across_repeated_forward_calls(
    vulkan_ctx, upload_counter
):
    """End-to-end regression test for the model_runner.py bug: a wrapped
    Linear module's forward(), called repeatedly (as it would be once per
    decode step), must not defeat the weight cache by pre-converting
    `weight` with `.float()` before calling vulkan_ops.linear() - that
    would give every call a fresh tensor identity and always miss.
    """
    from vllm_vulkan.model_runner import _wrap_linear  # noqa: PLC0415

    # requires_grad=False matches real inference weights (vLLM always runs
    # inference-only, never training) - without it, weight.float() produces
    # a tensor that still requires grad, and the later .numpy() call inside
    # _to_bytes fails, an unrelated test-construction pitfall rather than
    # anything about the cache behaviour actually under test here.
    module = torch.nn.Linear(8, 16, bias=True)
    module.weight = torch.nn.Parameter(
        module.weight.detach().to(torch.bfloat16), requires_grad=False
    )
    module.bias = torch.nn.Parameter(
        module.bias.detach().to(torch.bfloat16), requires_grad=False
    )

    _wrap_linear(module)

    x = torch.randn(1, 8, dtype=torch.float32)  # T=1: decode-shaped input
    module.forward(x)
    assert upload_counter["n"] == 1, "first forward() should upload once"

    for _ in range(5):
        module.forward(x)
    assert upload_counter["n"] == 1, (
        "repeated forward() calls (as happen once per decode step in real "
        "usage) must hit the weight cache, not re-upload the weight matrix "
        "on every single call"
    )


@pytest.fixture
def cpu_float32_conversion_counter(monkeypatch):
    """Count real CPU float32 conversions (_cpu_float32_cache.put calls)
    deterministically, same rationale as upload_counter above."""
    counts = {"n": 0}
    orig_put = vulkan_ops._WeightCache.put

    def counting_put(self, w, value):
        if self is vulkan_ops._cpu_float32_cache:
            counts["n"] += 1
        return orig_put(self, w, value)

    monkeypatch.setattr(vulkan_ops._WeightCache, "put", counting_put)
    return counts


def test_linear_prefill_path_reuses_cpu_float32_cache_across_calls(
    vulkan_ctx, cpu_float32_conversion_counter
):
    """linear()'s CPU (prefill, T >= _MATVEC_THRESHOLD) fallback path used
    to call weight.float() unconditionally on every call, with no caching
    at all - unlike the GPU decode path's _get_or_upload_weight. Repeated
    prefill calls with the same persistent bf16 weight object (as a real
    nn.Module.weight Parameter would be, across every prefill request)
    must convert to float32 once, not on every single call.
    """
    out_features, in_features = 16, 8
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    x = torch.randn(8, in_features, dtype=torch.float32)  # T=8 >= _MATVEC_THRESHOLD

    out1 = vulkan_ops.linear(x, weight, None)
    assert cpu_float32_conversion_counter["n"] == 1, "first call should convert once"

    for _ in range(5):
        out_n = vulkan_ops.linear(x, weight, None)
        torch.testing.assert_close(out_n, out1)
    assert cpu_float32_conversion_counter["n"] == 1, (
        "repeated prefill calls with the same weight object must reuse the "
        "cached float32 conversion, not redo it on every call"
    )


def test_linear_prefill_path_float32_weight_bypasses_cache(
    vulkan_ctx, cpu_float32_conversion_counter
):
    """A weight that's already float32 needs no conversion or caching at
    all - _get_or_convert_to_float32_cpu should return it unchanged and
    never touch _cpu_float32_cache."""
    out_features, in_features = 16, 8
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    x = torch.randn(8, in_features, dtype=torch.float32)

    vulkan_ops.linear(x, weight, None)
    vulkan_ops.linear(x, weight, None)
    assert cpu_float32_conversion_counter["n"] == 0
    assert len(vulkan_ops._cpu_float32_cache) == 0


def test_weight_cache_entry_is_freed_when_weight_storage_is_garbage_collected():
    """Regression test for a real memory leak: _WeightCache.put() used to
    store (weakref.ref(storage), value) with no cleanup callback, so a
    weight whose storage was garbage collected - but whose id() key was
    never looked up again - left its entry (and the large cached value it
    holds: a GPU-resident buffer or a full float32 CPU copy of the weight
    matrix) in the cache's dict forever, since a plain dict never removes
    entries on its own. This doesn't need a real Vulkan device - it's pure
    Python/GC behaviour on _WeightCache directly.
    """
    import gc

    cache = vulkan_ops._WeightCache()

    def put_a_short_lived_weight():
        weight = torch.randn(4, 4, dtype=torch.float32)
        cache.put(weight, object())
        assert len(cache) == 1
        # weight (and its storage) goes out of scope when this function
        # returns, with nothing else referencing it.

    put_a_short_lived_weight()
    gc.collect()
    assert len(cache) == 0, (
        "cache entry for a garbage-collected weight must be removed "
        "automatically, not leak forever"
    )
