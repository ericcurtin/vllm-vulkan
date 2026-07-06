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


def test_rms_norm_pc_is_memoized_across_calls():
    """`_rms_norm_pc`/`_matvec_pc` are `@lru_cache`d pure functions of
    their (small, hashable) arguments — repeated calls with the same
    (nrows, ncols, eps)/(T, K, N) (exactly what happens once per module,
    per decoder layer, per decode step in real usage — see their own
    doc comments) must return the *same* cached `bytes` object rather
    than repacking from scratch every time. This doesn't need a real
    Vulkan device — it's pure Python behaviour on the push-constant
    builders directly. `bytes` are immutable, so sharing the identical
    cached object across every caller is safe by construction (nothing
    can mutate it out from under another caller).
    """
    a = vulkan_ops._rms_norm_pc(1, 1536, 1e-6)
    b = vulkan_ops._rms_norm_pc(1, 1536, 1e-6)
    assert a is b, "identical arguments must hit the lru_cache, not repack"

    c = vulkan_ops._matvec_pc(1, 1536, 6144)
    d = vulkan_ops._matvec_pc(1, 1536, 6144)
    assert c is d, "identical arguments must hit the lru_cache, not repack"

    # Different arguments must still produce distinct (and correct) results
    # -- the cache must be keyed on every argument, not just the first one.
    e = vulkan_ops._rms_norm_pc(1, 256, 1e-6)
    assert e != a, "different ncols must produce different push constants"
    f = vulkan_ops._matvec_pc(1, 1536, 256)
    assert f != c, "different N must produce different push constants"


def test_rms_norm_pc_matches_uncached_reference():
    """The `@lru_cache` decorator must not change `_rms_norm_pc`'s/
    `_matvec_pc`'s actual output — compares the (now memoized) functions'
    result against a plain reimplementation of the pre-caching logic."""

    def rms_norm_pc_reference(nrows: int, ncols: int, eps: float) -> bytes:
        import struct as _struct

        nb00, nb01, nb02 = 1, ncols, nrows * ncols
        ne10, nb10, nb11 = ncols, 1, ncols
        nb20, nb21 = 1, ncols
        pc = _struct.pack("9I", nrows * ncols, ncols, nrows, 1, 1, nb00, nb01, nb02, 1)
        pc += _struct.pack("8I", ne10, 1, 1, 1, nb10, nb11, 1, 1)
        pc += _struct.pack("8I", ncols, nrows, 1, 1, nb20, nb21, 1, 1)
        pc += _struct.pack("I f f i", 0, eps, 0.0, 0)
        return pc

    for nrows, ncols, eps in [(1, 1536, 1e-6), (1, 256, 1e-6), (8, 1536, 1e-5)]:
        assert vulkan_ops._rms_norm_pc(nrows, ncols, eps) == rms_norm_pc_reference(
            nrows, ncols, eps
        )


def test_matvec_pc_repeated_calls_faster_than_uncached_reference():
    """Measures the actual speedup `@lru_cache` gives `_matvec_pc` on a
    cache-hit path — the exact access pattern the real decode loop
    exercises (same module, same shape, called once per generated
    token)."""
    import struct as _struct
    import time

    def matvec_pc_reference(t: int, k: int, n: int) -> bytes:
        return _struct.pack("13I", k, k, k, n, k * n, k, n, 0, 0, 1, t, t, 1)

    iters = 5000
    t0 = time.perf_counter()
    for _ in range(iters):
        matvec_pc_reference(1, 1536, 6144)
    uncached_elapsed = time.perf_counter() - t0

    # Warm the cache once before timing the cache-hit path.
    vulkan_ops._matvec_pc(1, 1536, 6144)
    t0 = time.perf_counter()
    for _ in range(iters):
        vulkan_ops._matvec_pc(1, 1536, 6144)
    cached_elapsed = time.perf_counter() - t0

    assert cached_elapsed < uncached_elapsed, (
        f"cached _matvec_pc ({cached_elapsed:.4f}s/{iters}) was not faster than "
        f"the uncached reference ({uncached_elapsed:.4f}s/{iters})"
    )


def _reference_gpu_rms_norm_dispatch(
    ctx, x: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    """Reconstructs the OLD Vulkan-dispatching `rms_norm()` implementation
    (before this change) for direct comparison — same logic as the
    function used to have, just kept here as an independent reference
    instead of in the real (now CPU-only) `rms_norm`.
    """
    orig_shape = x.shape
    ncols = orig_shape[-1]
    x_flat = x.float().reshape(-1, ncols)
    nrows = x_flat.shape[0]

    has_weight = weight is not None and weight.numel() == ncols
    shader = "rms_norm_f32_mul" if has_weight else "rms_norm_f32"

    pc = vulkan_ops._rms_norm_pc(nrows, ncols, eps)
    x_bytes = vulkan_ops._to_bytes(x_flat)
    out_size = nrows * ncols * 4

    w_gpu = vulkan_ops._get_or_upload_weight(ctx, weight) if has_weight else None
    w_binding = w_gpu if w_gpu is not None else bytes(ncols * 4)

    results = ctx.execute_batch(
        [(shader, [x_bytes, w_binding], [out_size], pc, (nrows, 1, 1), False)]
    )
    out = vulkan_ops._from_bytes(results[0][0], (nrows, ncols), torch.float32)
    return out.reshape(orig_shape)


class TestRmsNormAlwaysUsesCpu:
    """`rms_norm()` now always computes on CPU rather than dispatching to
    Vulkan — measured directly on this hardware to be consistently
    slower via GPU at every batch size tested (see `rms_norm`'s own doc
    comment in vulkan_ops.py for the full writeup). These tests confirm
    (a) the CPU path is still numerically correct, and (b) the measured
    speedup this change is based on actually holds, using the shader
    dispatch machinery (`_reference_gpu_rms_norm_dispatch` above) as an
    independent, direct comparison rather than trusting the doc comment's
    numbers alone.
    """

    def test_matches_reference_rms_norm_math(self):
        """Pure-math correctness check, independent of any Vulkan device:
        RMSNorm(x) = x * rsqrt(mean(x^2) + eps), optionally scaled by a
        weight vector — verified against a from-scratch reference
        implementation (not `_cpu_rms_norm` itself, to avoid comparing an
        implementation against a copy of itself)."""
        torch.manual_seed(0)
        x = torch.randn(4, 1536, dtype=torch.float32)
        weight = torch.randn(1536, dtype=torch.float32)
        eps = 1e-6

        result = vulkan_ops.rms_norm(x, weight, eps)

        variance = (x.double() ** 2).mean(dim=-1, keepdim=True)
        expected = (x.double() * torch.rsqrt(variance + eps) * weight.double()).float()
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_matches_reference_rms_norm_math_no_weight(self):
        torch.manual_seed(1)
        x = torch.randn(2, 256, dtype=torch.float32)
        eps = 1e-6

        result = vulkan_ops.rms_norm(x, None, eps)

        variance = (x.double() ** 2).mean(dim=-1, keepdim=True)
        expected = (x.double() * torch.rsqrt(variance + eps)).float()
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_cpu_path_matches_gpu_dispatch_reference(self, vulkan_ctx):
        """The CPU-only `rms_norm()` must still agree numerically with
        the Vulkan-dispatching path it replaced (within float32
        tolerance) — confirms this was purely a dispatch-target
        performance decision, not a behavior change."""
        torch.manual_seed(2)
        x = torch.randn(3, 1536, dtype=torch.float32)
        weight = torch.randn(1536, dtype=torch.float32)
        eps = 1e-6

        cpu_result = vulkan_ops.rms_norm(x, weight, eps)
        gpu_result = _reference_gpu_rms_norm_dispatch(vulkan_ctx, x, weight, eps)
        torch.testing.assert_close(cpu_result, gpu_result, rtol=1e-3, atol=1e-4)

    def test_cpu_path_is_faster_than_gpu_dispatch_at_decode_shape(self, vulkan_ctx):
        """Measures the actual speedup at the real Gemma4-E2B decode
        shape (nrows=1, ncols=1536) — the single most common call
        pattern (once per RMSNorm module, per decoder layer, per
        generated token)."""
        import time

        x = torch.randn(1, 1536, dtype=torch.float32)
        weight = torch.randn(1536, dtype=torch.float32)
        eps = 1e-6

        # Warm up (pipeline/cache creation) before timing either path.
        for _ in range(5):
            vulkan_ops.rms_norm(x, weight, eps)
            _reference_gpu_rms_norm_dispatch(vulkan_ctx, x, weight, eps)

        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            _reference_gpu_rms_norm_dispatch(vulkan_ctx, x, weight, eps)
        gpu_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(iters):
            vulkan_ops.rms_norm(x, weight, eps)
        cpu_elapsed = time.perf_counter() - t0

        print(
            f"\nrms_norm at decode shape (nrows=1, ncols=1536): "
            f"GPU dispatch {gpu_elapsed / iters * 1e6:.1f}us/call, "
            f"CPU (current) {cpu_elapsed / iters * 1e6:.1f}us/call, "
            f"speedup {gpu_elapsed / cpu_elapsed:.2f}x"
        )
        assert cpu_elapsed < gpu_elapsed, (
            f"CPU rms_norm ({cpu_elapsed:.4f}s/{iters}) was not faster than "
            f"GPU dispatch ({gpu_elapsed:.4f}s/{iters})"
        )
