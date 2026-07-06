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
    # `set_context` above already resets `_weight_f16_decision_cache` to a
    # fresh `_WeightCache()` (see its doc comment) since `ctx` is a newly
    # constructed `VulkanContext` every call — saved/restored here anyway,
    # for the same explicit symmetry with `_weight_cache`/
    # `_cpu_float32_cache` this fixture already keeps.
    prev_f16_decision_cache = vulkan_ops._weight_f16_decision_cache
    vulkan_ops._weight_cache = vulkan_ops._WeightCache()
    vulkan_ops._cpu_float32_cache = vulkan_ops._WeightCache()
    try:
        yield ctx
    finally:
        vulkan_ops._ctx = prev_ctx
        vulkan_ops._weight_cache = prev_cache
        vulkan_ops._cpu_float32_cache = prev_cpu_cache
        vulkan_ops._weight_f16_decision_cache = prev_f16_decision_cache


@pytest.fixture
def upload_counter(monkeypatch):
    """Count real GPU-weight/CPU-float32-cache-populating `_WeightCache.put`
    calls deterministically, independent of the weak-ref cache's final size
    (see module docstring for why the latter can be misleading).

    Used by callers that assert on either `_weight_cache` (a real GPU
    upload) or `_cpu_float32_cache` (a real host-side float32 conversion)
    being populated exactly once — historically the same thing, since
    counting *any* `_WeightCache.put` call was equivalent to counting
    "real, potentially-expensive conversion/upload work done" when those
    were the only two `_WeightCache` instances in this module.

    Excludes `_weight_f16_decision_cache` specifically (added in #83):
    `_weight_uses_f16` caches its own per-weight decision there — a cheap
    cached bool, not a GPU upload or CPU conversion — so counting its
    `put` calls alongside the other two would double-count "1 real
    upload" as 2 (one decision-cache `put`, one actual weight-cache
    `put`) for any weight that goes through
    `_get_or_upload_weight(..., prefer_f16=True)`, exactly the weights
    several of this fixture's callers (>= _MATVEC_MIN_WEIGHT_ELEMENTS)
    use.
    """
    counts = {"n": 0}
    orig_put = vulkan_ops._WeightCache.put

    def counting_put(self, w, gpu):
        if self is not vulkan_ops._weight_f16_decision_cache:
            counts["n"] += 1
        return orig_put(self, w, gpu)

    monkeypatch.setattr(vulkan_ops._WeightCache, "put", counting_put)
    return counts


def test_linear_matches_torch_reference_at_realistic_hidden_size(vulkan_ctx):
    """Direct numerical correctness check for vulkan_ops.linear()'s GPU
    decode path (T < _MATVEC_THRESHOLD, weight_elements >=
    _MATVEC_MIN_WEIGHT_ELEMENTS) against `torch.nn.functional.linear` as
    ground truth, at a realistic transformer shape (2048x1536, matching
    Gemma4-E2B's q_proj) -- not just the tiny in_features=8 shapes the
    other tests in this file use to exercise the weight cache.

    This specific gap (no test comparing the GPU decode-matvec path's
    OUTPUT against a real reference at a realistic K) is what let a real
    correctness bug in mul_mat_vec_f32_f32_f32_subgroup go unnoticed: that
    shader silently diverged from the correct result for any K>=~256 while
    happening to still be correct at the small K (8) every other test here
    uses -- see subgroup_matvec_correctness_tests in src/lib.rs for the
    measured divergence, and this file's `_require_vulkan_context` /
    vulkan_ops.py's `linear`/`_vulkan_matvec` for the fix (now dispatching
    mul_mat_vec_f32_f32_f32, the plain non-subgroup variant, instead).

    out_features=2048 (not 512, used before `_MATVEC_MIN_WEIGHT_ELEMENTS`
    was introduced): 512*1536=786432 elements would now fall *below* that
    threshold and take the CPU path instead, defeating this test's whole
    purpose of exercising the GPU matvec dispatch specifically. 2048*1536
    = 3,145,728 stays clearly above the threshold (see
    TestLinearMatvecDispatchThreshold for the measurements behind it).
    """
    out_features, in_features = 2048, 1536
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    x = torch.randn(1, in_features, dtype=torch.float32)  # T=1 < _MATVEC_THRESHOLD

    result = vulkan_ops.linear(x, weight, None)
    expected = torch.nn.functional.linear(x, weight, None)
    # rtol=1e-2/atol=5e-2 (not 1e-3/1e-3): since #83, `linear()`'s GPU decode
    # path uploads/dispatches this (>= _MATVEC_MIN_WEIGHT_ELEMENTS) weight as
    # float16 whenever `_weight_uses_f16` says it's safe to (see
    # `_get_or_upload_weight`'s doc comment) — a deliberate, expected
    # precision/bandwidth tradeoff, not a correctness bug: the result matches
    # a CPU reference computed from the *same* f16-rounded weight values to
    # within ~2e-4 (well inside f16's own ~1e-3 relative precision), while
    # diverging from this fp32-weight `torch.nn.functional.linear` reference
    # by up to ~0.032 at this shape/dtype across repeated runs on real
    # (non-mock) Vulkan hardware — the two are simply no longer expected to
    # agree at 1e-3/1e-3 now that the weight itself is stored more coarsely
    # on the GPU. rtol=1e-2/atol=5e-2 comfortably covers the observed ~0.03
    # max abs error (including near-zero-`expected` elements, where relative
    # error alone spikes) with margin, without being so loose it'd miss an
    # actual dispatch/binding/stride bug reintroducing errors an order of
    # magnitude larger than f16 rounding alone explains.
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=5e-2)


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

    out_features/in_features=2048/1536 (not a tiny shape like 16/8): must
    stay above `_MATVEC_MIN_WEIGHT_ELEMENTS` so this test actually
    exercises the GPU weight cache (`_get_or_upload_weight`) it claims to
    -- a smaller shape would now take the CPU float32-cache path instead
    (still passing, since `upload_counter` patches the shared
    `_WeightCache.put` used by both caches, but silently testing the
    wrong cache).
    """
    out_features, in_features = 2048, 1536
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

    Linear(1536, 2048) (not a tiny shape like Linear(8, 16)): must stay
    above `_MATVEC_MIN_WEIGHT_ELEMENTS` so this test actually exercises
    the GPU weight cache via the real GPU matvec dispatch path, matching
    `test_linear_reuses_weight_cache_across_calls_with_bf16_weight`'s
    same reasoning above.
    """
    from vllm_vulkan.model_runner import _wrap_linear  # noqa: PLC0415

    # requires_grad=False matches real inference weights (vLLM always runs
    # inference-only, never training) - without it, weight.float() produces
    # a tensor that still requires grad, and the later .numpy() call inside
    # _to_bytes fails, an unrelated test-construction pitfall rather than
    # anything about the cache behaviour actually under test here.
    module = torch.nn.Linear(1536, 2048, bias=True)
    module.weight = torch.nn.Parameter(
        module.weight.detach().to(torch.bfloat16), requires_grad=False
    )
    module.bias = torch.nn.Parameter(
        module.bias.detach().to(torch.bfloat16), requires_grad=False
    )

    _wrap_linear(module)

    x = torch.randn(1, 1536, dtype=torch.float32)  # T=1: decode-shaped input
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


def test_cached_available_shaders_matches_uncached_reference(vulkan_ctx):
    """`_cached_available_shaders` must return the exact same set of
    shader names as calling `ctx.available_shaders()` directly, just
    computed once instead of on every call — see its own doc comment in
    vulkan_ops.py for the measured per-call overhead this avoids
    (`linear()` used to call `ctx.available_shaders()` on every
    invocation, ~175-210 times/decode-step across Gemma4-E2B's decoder
    layers)."""
    cached = vulkan_ops._cached_available_shaders(vulkan_ctx)
    assert cached == frozenset(vulkan_ctx.available_shaders())

    # Must actually be cached, not recomputed on every call: repeated
    # calls return the identical (not just equal) frozenset object.
    assert vulkan_ops._cached_available_shaders(vulkan_ctx) is cached


def test_set_context_invalidates_the_shader_cache():
    """A fresh `set_context` call must invalidate any previously cached
    `available_shaders()` result — otherwise a test (or a real caller)
    that swaps in a different `VulkanContext` could silently keep
    reading a stale cache computed for the *previous* context. Pure
    Python/state behaviour on the module-level cache directly — doesn't
    need a real Vulkan device."""
    prev_ctx = vulkan_ops._ctx
    prev_cache = vulkan_ops._available_shaders_cache
    try:
        vulkan_ops._available_shaders_cache = frozenset({"stale_entry"})
        vulkan_ops.set_context(object())  # type: ignore[arg-type]
        assert vulkan_ops._available_shaders_cache is None, (
            "set_context must invalidate (reset to None) any previously "
            "cached available_shaders() result"
        )
    finally:
        vulkan_ops._ctx = prev_ctx
        vulkan_ops._available_shaders_cache = prev_cache


def test_set_context_invalidates_the_f16_decision_cache():
    """Same regression this file already guards for
    `_available_shaders_cache` (`test_set_context_invalidates_the_shader_cache`
    above), for `_weight_f16_decision_cache`: `_weight_uses_f16`'s per-weight
    decision depends on the *context* too (via `available_shaders()`), not
    just the weight's own values, so a fresh `set_context` call must
    invalidate it — otherwise a weight tensor queried against two
    different `VulkanContext` instances with different compiled shader
    sets (a real risk this project's review process caught before merge)
    could silently keep reading a decision computed for the *previous*
    context. Pure Python/state behaviour on the module-level cache
    directly — doesn't need a real Vulkan device."""
    prev_ctx = vulkan_ops._ctx
    prev_cache = vulkan_ops._weight_f16_decision_cache
    try:
        stale_cache = vulkan_ops._WeightCache()
        vulkan_ops._weight_f16_decision_cache = stale_cache
        vulkan_ops.set_context(object())  # type: ignore[arg-type]
        assert vulkan_ops._weight_f16_decision_cache is not stale_cache, (
            "set_context must invalidate (replace with a fresh _WeightCache) "
            "any previously cached _weight_uses_f16 decisions"
        )
    finally:
        vulkan_ops._ctx = prev_ctx
        vulkan_ops._weight_f16_decision_cache = prev_cache


def test_cached_available_shaders_is_faster_than_uncached_repeated_calls(
    vulkan_ctx,
):
    """Measures the actual speedup: `_cached_available_shaders` should be
    dramatically faster than calling `ctx.available_shaders()` fresh on
    every call, since the latter re-marshals a `Vec<String>` from Rust
    into a brand-new Python list via PyO3 every single time.

    Takes the minimum elapsed time across several independent trials —
    see `TestLinearMatvecDispatchThreshold`'s similarly-timed test for
    why (measurement noise under full-suite load).
    """
    import time

    # Warm the cache once before timing the cached path.
    vulkan_ops._cached_available_shaders(vulkan_ctx)

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

    uncached_elapsed = timed(vulkan_ctx.available_shaders)
    cached_elapsed = timed(lambda: vulkan_ops._cached_available_shaders(vulkan_ctx))

    print(
        f"\navailable_shaders(), best-of-{trials}: "
        f"uncached {uncached_elapsed / iters * 1e6:.3f}us/call, "
        f"cached {cached_elapsed / iters * 1e6:.3f}us/call, "
        f"speedup {uncached_elapsed / cached_elapsed:.1f}x"
    )
    assert cached_elapsed < uncached_elapsed, (
        f"cached available_shaders ({cached_elapsed:.4f}s/{iters}) was not "
        f"faster than the uncached reference ({uncached_elapsed:.4f}s/{iters})"
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
        generated token).

        Takes the *minimum* elapsed time across several independent
        trials rather than a single timed loop — see
        `TestLinearMatvecDispatchThreshold.test_small_weight_cpu_path_is_faster_than_gpu_dispatch_at_decode_shape`'s
        doc comment for why (measurement noise from running as part of
        the full suite, not from anything about this specific
        comparison — this margin is large enough that it's unlikely to
        flip, but the fix is cheap and keeps both tests consistent).
        """
        import time

        x = torch.randn(1, 1536, dtype=torch.float32)
        weight = torch.randn(1536, dtype=torch.float32)
        eps = 1e-6

        # Warm up (pipeline/cache creation) before timing either path.
        for _ in range(5):
            vulkan_ops.rms_norm(x, weight, eps)
            _reference_gpu_rms_norm_dispatch(vulkan_ctx, x, weight, eps)

        iters = 200
        trials = 5

        def timed(fn) -> float:
            best = float("inf")
            for _ in range(trials):
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                best = min(best, time.perf_counter() - t0)
            return best

        gpu_elapsed = timed(
            lambda: _reference_gpu_rms_norm_dispatch(vulkan_ctx, x, weight, eps)
        )
        cpu_elapsed = timed(lambda: vulkan_ops.rms_norm(x, weight, eps))

        print(
            f"\nrms_norm at decode shape (nrows=1, ncols=1536), best-of-{trials}: "
            f"GPU dispatch {gpu_elapsed / iters * 1e6:.1f}us/call, "
            f"CPU (current) {cpu_elapsed / iters * 1e6:.1f}us/call, "
            f"speedup {gpu_elapsed / cpu_elapsed:.2f}x"
        )
        assert cpu_elapsed < gpu_elapsed, (
            f"CPU rms_norm ({cpu_elapsed:.4f}s/{iters}) was not faster than "
            f"GPU dispatch ({gpu_elapsed:.4f}s/{iters})"
        )


class TestLinearMatvecDispatchThreshold:
    """`linear()`'s GPU-vs-CPU dispatch decision now also checks
    `weight_elements >= _MATVEC_MIN_WEIGHT_ELEMENTS`, not just
    `T < _MATVEC_THRESHOLD` — see `linear`'s doc comment in
    vulkan_ops.py for the measurements this threshold is based on:
    Gemma4-E2B's k_proj/v_proj (small, since `num_key_value_heads=1`
    gives an out_features of only 256/512) measured CPU-faster at every
    T tested, including T=1 decode, while q_proj/o_proj/gate_proj/
    up_proj/down_proj (all >=3.15M elements) measured GPU-faster at
    T=1, exactly as `_MATVEC_THRESHOLD` already assumed.
    """

    def test_small_weight_matches_torch_reference(self, vulkan_ctx):
        """Correctness check for the (new) small-weight CPU-only path, at
        Gemma4-E2B's real k_proj/v_proj shape (num_key_value_heads=1 *
        head_dim=256 out_features, hidden_size=1536 in_features)."""
        out_features, in_features = 256, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        x = torch.randn(1, in_features, dtype=torch.float32)

        result = vulkan_ops.linear(x, weight, None)
        expected = torch.nn.functional.linear(x, weight, None)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_small_weight_uses_cpu_path_not_gpu_weight_cache(
        self, vulkan_ctx, upload_counter
    ):
        """A weight below `_MATVEC_MIN_WEIGHT_ELEMENTS` (k_proj/v_proj's
        real shape: 256*1536 = 393216 elements) must take the CPU path
        (`_get_or_convert_to_float32_cpu`) rather than the GPU weight
        cache (`_get_or_upload_weight`) — verified directly by checking
        the weight never appears in `_weight_cache` (the GPU-specific
        cache), even though `linear()` was called at T=1 (which alone
        would have triggered the old T-only threshold's GPU path).

        bf16 (not float32): `_get_or_convert_to_float32_cpu` only
        populates `_cpu_float32_cache` when an actual conversion happens
        (skipped entirely for already-float32 weights — see
        `test_linear_prefill_path_float32_weight_bypasses_cache` above),
        so a bf16 weight is needed here to actually exercise (and count)
        that cache being populated.
        """
        out_features, in_features = 256, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        x = torch.randn(1, in_features, dtype=torch.float32)

        vulkan_ops.linear(x, weight, None)
        assert vulkan_ops._weight_cache.get(weight) is None, (
            "a small weight (below _MATVEC_MIN_WEIGHT_ELEMENTS) must never "
            "reach the GPU weight cache"
        )
        assert upload_counter["n"] == 1, (
            "the CPU float32 conversion cache should still be populated exactly once"
        )

    def test_small_weight_cpu_path_is_faster_than_gpu_dispatch_at_decode_shape(
        self, vulkan_ctx
    ):
        """Measures the actual speedup at Gemma4-E2B's real k_proj shape
        (256x1536) and T=1 (decode) — the shape/size this threshold fix
        targets.

        Takes the *minimum* elapsed time across several independent
        trials (rather than a single timed loop) for each path: when run
        as part of the full test suite (as opposed to in isolation),
        this measurement is noisy enough — from other tests' torch/numpy
        activity, GC pauses, OS scheduling, etc. — that a single trial
        occasionally shows an inflated CPU-path time large enough to
        flip the comparison, even though the true underlying costs are
        not close (isolated runs consistently show ~1.5-3x, not a
        marginal ~1.0x). Taking the minimum of several trials is the
        standard fix for exactly this kind of measurement noise (see
        e.g. `matvec_r4_tests` in src/lib.rs for the same technique
        applied to a Rust benchmark that had the same flakiness).
        """
        import time

        out_features, in_features = 256, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        x = torch.randn(1, in_features, dtype=torch.float32)

        for _ in range(5):
            vulkan_ops.linear(x, weight, None)
            vulkan_ops._vulkan_matvec(vulkan_ctx, x, weight)

        iters = 200
        trials = 5

        def timed(fn) -> float:
            best = float("inf")
            for _ in range(trials):
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                best = min(best, time.perf_counter() - t0)
            return best

        gpu_elapsed = timed(lambda: vulkan_ops._vulkan_matvec(vulkan_ctx, x, weight))
        cpu_elapsed = timed(lambda: vulkan_ops.linear(x, weight, None))

        print(
            f"\nlinear() at k_proj shape (256x1536, T=1), best-of-{trials}: "
            f"GPU matvec {gpu_elapsed / iters * 1e6:.1f}us/call, "
            f"linear() (now CPU) {cpu_elapsed / iters * 1e6:.1f}us/call, "
            f"speedup {gpu_elapsed / cpu_elapsed:.2f}x"
        )
        assert cpu_elapsed < gpu_elapsed, (
            f"linear()'s CPU path ({cpu_elapsed:.4f}s/{iters}) was not faster "
            f"than GPU matvec dispatch ({gpu_elapsed:.4f}s/{iters})"
        )

    def test_large_weight_still_uses_gpu_path_at_decode_shape(
        self, vulkan_ctx, upload_counter
    ):
        """A weight at or above `_MATVEC_MIN_WEIGHT_ELEMENTS` (q_proj's
        real shape: 2048*1536 = 3,145,728 elements) must still take the
        GPU path at T=1 (decode) — this threshold fix must not regress
        the already-correct, already-fast large-weight case."""
        out_features, in_features = 2048, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        x = torch.randn(1, in_features, dtype=torch.float32)

        vulkan_ops.linear(x, weight, None)
        assert vulkan_ops._weight_cache.get(weight) is not None, (
            "a large weight (>= _MATVEC_MIN_WEIGHT_ELEMENTS) at T=1 must "
            "still use the GPU weight cache / matvec dispatch path"
        )
        assert upload_counter["n"] == 1

    def test_matvec_batching_does_not_help_at_prefill_scale_t(self, vulkan_ctx):
        """Regression guard for `_MATVEC_THRESHOLD=4`: do NOT raise this
        threshold to also dispatch larger (prefill-scale) T to the GPU
        matvec shader - see `_MATVEC_THRESHOLD`'s doc comment in
        vulkan_ops.py for the full measurement and root-cause
        explanation (mul_mat_vec_f32_f32_f32's per-token dispatch
        re-reads the entire weight matrix from GPU memory for every one
        of the T "batch" rows, with none of a real tiled matmul's
        weight-tile-reuse - bandwidth cost scales as O(T x in_features x
        out_features), the same order as CPU's cost but at a worse
        effective rate here).

        Measured at Gemma4-E2B's real q_proj shape (2048x1536) and
        T=128 (a representative prefill chunk size, well above
        `_MATVEC_THRESHOLD`): CPU must remain clearly faster. Uses this
        file's own min-of-N-trials pattern (see
        test_small_weight_cpu_path_is_faster_than_gpu_dispatch_at_decode_shape
        just above) - safe here since the underlying effect (measured
        independently at ~6.5x and widening with T) is far larger than
        any plausible measurement noise, unlike cases where a similar
        pattern was found to be misleading for near-tied comparisons
        (see kv_ops.py's paged_kv_write_and_decode_batch test in this
        session's history for that finding).
        """
        import time

        out_features, in_features = 2048, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        t_prefill = 128
        x = torch.randn(t_prefill, in_features, dtype=torch.float32)

        for _ in range(3):
            vulkan_ops._vulkan_matvec(vulkan_ctx, x, weight)
            torch.nn.functional.linear(x, weight, None)

        iters = 10
        trials = 3

        def timed(fn) -> float:
            best = float("inf")
            for _ in range(trials):
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                best = min(best, time.perf_counter() - t0)
            return best

        gpu_elapsed = timed(lambda: vulkan_ops._vulkan_matvec(vulkan_ctx, x, weight))
        cpu_elapsed = timed(lambda: torch.nn.functional.linear(x, weight, None))

        print(
            f"\nmatvec batching at prefill scale (T={t_prefill}, 2048x1536), "
            f"best-of-{trials}: GPU batched-matvec {gpu_elapsed / iters * 1e6:.1f}us/call, "
            f"CPU linear {cpu_elapsed / iters * 1e6:.1f}us/call, "
            f"CPU speedup {gpu_elapsed / cpu_elapsed:.2f}x"
        )
        assert cpu_elapsed < gpu_elapsed, (
            f"CPU ({cpu_elapsed:.4f}s/{iters}) was not faster than GPU batched-matvec "
            f"({gpu_elapsed:.4f}s/{iters}) at prefill-scale T={t_prefill} - if this "
            f"genuinely changed (e.g. a future driver/hardware improvement), see "
            f"_MATVEC_THRESHOLD's doc comment before raising it based on this alone."
        )


class TestLinearTiledMatmulPrefillDispatch:
    """`linear()` now dispatches the tiled matmul shader (`matmul_f32_f32`,
    from shaders/mul_mm.comp — see `_vulkan_matmul`) for prefill-scale T
    on large-enough weights, instead of always falling back to CPU — see
    `_MATVEC_THRESHOLD`'s doc comment in vulkan_ops.py for the
    measurements this is based on (GPU wins at every T from 4 up, unlike
    the matvec case `TestLinearMatvecDispatchThreshold` above guards
    against reusing at prefill scale).
    """

    def test_large_weight_matches_torch_reference_at_prefill_shapes(self, vulkan_ctx):
        """Correctness at Gemma4-E2B's real q_proj shape, across several
        representative prefill chunk sizes — including T=4 (right at
        `_MATVEC_THRESHOLD`'s boundary) and shapes that straddle this
        shader's BM=BN=64/BK=32 tile boundaries (T=63, not a multiple of
        64)."""
        out_features, in_features = 2048, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        for t in (4, 16, 63, 128, 512):
            x = torch.randn(t, in_features, dtype=torch.float32)
            result = vulkan_ops.linear(x, weight, None)
            expected = torch.nn.functional.linear(x, weight, None)
            # rtol=1e-2/atol=5e-2 (not 1e-3/1e-2): same rationale as
            # `test_linear_matches_torch_reference_at_realistic_hidden_size`
            # above — since #83, this weight (>= _MATVEC_MIN_WEIGHT_ELEMENTS)
            # is uploaded/dispatched as float16 by `_vulkan_matmul` too, so
            # this fp32-weight `torch.nn.functional.linear` reference is no
            # longer expected to match at 1e-2 absolute; observed max abs
            # error across these T values on real Vulkan hardware is ~0.045.
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=5e-2)

    def test_large_weight_prefill_uses_gpu_weight_cache_not_cpu_path(
        self, vulkan_ctx, upload_counter
    ):
        """A weight at or above `_MATVEC_MIN_WEIGHT_ELEMENTS` at
        prefill-scale T (128, well above `_MATVEC_THRESHOLD`) must take
        the GPU tiled-matmul path (`_vulkan_matmul`, sharing
        `_get_or_upload_weight`'s cache with the decode matvec path) —
        not silently keep falling back to
        `_get_or_convert_to_float32_cpu`."""
        out_features, in_features = 2048, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        x = torch.randn(128, in_features, dtype=torch.float32)

        vulkan_ops.linear(x, weight, None)
        assert vulkan_ops._weight_cache.get(weight) is not None, (
            "a large weight at prefill-scale T must use the GPU tiled-matmul "
            "path's weight cache"
        )
        assert upload_counter["n"] == 1
        assert vulkan_ops._cpu_float32_cache.get(weight) is None, (
            "the CPU float32 conversion cache must NOT be populated when the "
            "GPU tiled-matmul path is taken"
        )

    def test_small_weight_still_uses_cpu_at_prefill_scale(
        self, vulkan_ctx, upload_counter
    ):
        """A weight below `_MATVEC_MIN_WEIGHT_ELEMENTS` (k_proj/v_proj's
        real shape) must still take the CPU path at prefill-scale T,
        exactly as it already does at decode-scale T (see
        `TestLinearMatvecDispatchThreshold.
        test_small_weight_uses_cpu_path_not_gpu_weight_cache`) — the
        tiled-matmul dispatch decision reuses the same size gate, not a
        separate one, per `_MATVEC_THRESHOLD`'s doc comment."""
        out_features, in_features = 256, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        x = torch.randn(128, in_features, dtype=torch.float32)

        vulkan_ops.linear(x, weight, None)
        assert vulkan_ops._weight_cache.get(weight) is None, (
            "a small weight (below _MATVEC_MIN_WEIGHT_ELEMENTS) must never "
            "reach the GPU weight cache, even at prefill-scale T"
        )
        assert upload_counter["n"] == 1

    def test_tiled_matmul_is_faster_than_cpu_at_prefill_scale(self, vulkan_ctx):
        """Measures the actual speedup at Gemma4-E2B's real q_proj shape
        (2048x1536) and T=128 (a representative prefill chunk size) —
        the mirror image of
        `TestLinearMatvecDispatchThreshold.test_matvec_batching_does_not_help_at_prefill_scale_t`
        above: unlike matvec, the tiled matmul shader genuinely reuses
        weight tiles across T, so GPU must win here, not lose. Uses this
        file's own min-of-N-trials pattern (see that test's doc comment
        for why it's safe here too: the underlying effect, measured
        independently in src/lib.rs's `tiled_matmul_beats_cpu_at_prefill_scale_for_large_weights`,
        is 1.5x-17x across shapes and T, far larger than plausible
        measurement noise).
        """
        import time

        out_features, in_features = 2048, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        t_prefill = 128
        x = torch.randn(t_prefill, in_features, dtype=torch.float32)

        for _ in range(3):
            vulkan_ops._vulkan_matmul(vulkan_ctx, x, weight)
            torch.nn.functional.linear(x, weight, None)

        iters = 10
        trials = 3

        def timed(fn) -> float:
            best = float("inf")
            for _ in range(trials):
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                best = min(best, time.perf_counter() - t0)
            return best

        gpu_elapsed = timed(lambda: vulkan_ops._vulkan_matmul(vulkan_ctx, x, weight))
        cpu_elapsed = timed(lambda: torch.nn.functional.linear(x, weight, None))

        print(
            f"\ntiled matmul at prefill scale (T={t_prefill}, 2048x1536), "
            f"best-of-{trials}: GPU matmul {gpu_elapsed / iters * 1e6:.1f}us/call, "
            f"CPU linear {cpu_elapsed / iters * 1e6:.1f}us/call, "
            f"GPU speedup {cpu_elapsed / gpu_elapsed:.2f}x"
        )
        assert gpu_elapsed < cpu_elapsed, (
            f"GPU tiled matmul ({gpu_elapsed:.4f}s/{iters}) was not faster than "
            f"CPU ({cpu_elapsed:.4f}s/{iters}) at prefill-scale T={t_prefill} - "
            f"see _MATVEC_THRESHOLD's doc comment before changing this dispatch "
            f"decision based on this alone."
        )


class TestVulkanF16WeightUpload:
    """`_get_or_upload_weight(..., prefer_f16=True)` (now used by both
    `_vulkan_matvec` and `_vulkan_matmul`) uploads a large weight as
    float16 instead of float32 whenever `_weight_uses_f16` judges it safe
    to — halving both the one-time upload cost and, more importantly
    since the same persistent buffer is re-read from GPU memory on every
    matvec/matmul dispatch against it, the bytes every one of those calls
    has to move. This mirrors the standalone Rust `VulkanModel` path's
    own weight upload (src/lib.rs's `VulkanModel::new`, "Projection
    weights are uploaded as f16 to halve memory bandwidth") — this
    (separate) `VulkanContext`-based path never had that applied even
    though the matching f16 shaders (`mul_mat_vec_f16_f32_f32`,
    `matmul_f16_f32_fp32`) were already compiled and available for it,
    per this project's established "compiled but never dispatched"
    pattern (see e.g. #82).

    These tests use `_require_vulkan_context()` directly rather than the
    `vulkan_ctx` fixture: the fixture deliberately disables GPU dispatch
    entirely on a software Vulkan renderer (see `set_context`'s doc
    comment) as a safety net for the *real* serving path, but that would
    also skip these tests everywhere a discrete GPU isn't available for
    CI/local testing — while the underlying dispatch/byte-layout
    correctness this file verifies is exactly as meaningful on a
    software renderer as a discrete GPU (same Vulkan pipeline, same
    SPIR-V, same push-constant contract), unlike a timing comparison.
    """

    def test_weight_safe_for_f16_true_for_realistic_model_weights(self):
        """Real transformer weights (bf16-loaded, roughly unit-scale after
        training normalization) must be judged safe for f16 — the
        overwhelmingly common case this optimization exists for."""
        weight = torch.randn(2048, 1536, dtype=torch.bfloat16) * 0.1
        assert vulkan_ops._weight_safe_for_f16(weight) is True

    def test_weight_safe_for_f16_false_for_values_exceeding_f16_range(self):
        """float16's exponent range (5 bits, max ~6.5e4) is much smaller
        than bf16/float32's (8 bits, max ~3.4e38 — see
        `_weight_safe_for_f16`'s doc comment): a weight containing even
        one value beyond that range must be rejected, not silently cast
        to +/-inf."""
        weight = torch.randn(16, 8, dtype=torch.float32)
        weight[3, 2] = 1e6  # finite in float32, overflows float16's range
        assert vulkan_ops._weight_safe_for_f16(weight) is False

    def test_weight_safe_for_f16_false_for_existing_nan_or_inf(self):
        """Not this function's job to fix (a NaN/inf weight is already
        broken before it reaches here), but it must not crash and must
        not claim such a weight is "safe" — `torch.isfinite` is already
        False for either."""
        weight = torch.randn(16, 8, dtype=torch.float32)
        weight[0, 0] = float("inf")
        assert vulkan_ops._weight_safe_for_f16(weight) is False

    def test_get_or_upload_weight_prefer_f16_halves_buffer_size_for_safe_weight(self):
        """Direct, hardware-independent verification that this actually
        reduces GPU memory footprint: a safe-for-f16 weight's persistent
        buffer must be exactly half the bytes of the same-shaped weight
        uploaded as float32 — not a timing measurement (this project's
        own history shows those can be misleading on a software Vulkan
        renderer, see e.g. `tiled_matmul_beats_cpu_at_prefill_scale_for_large_weights`'s
        doc comment in src/lib.rs), but a deterministic byte-count
        comparison that holds identically on any Vulkan device.
        """
        ctx = _require_vulkan_context()
        if "mul_mat_vec_f16_f32_f32" not in ctx.available_shaders():
            pytest.skip("mul_mat_vec_f16_f32_f32 shader unavailable")

        out_features, in_features = 2048, 1536
        weight_f16 = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        weight_f32 = torch.randn(out_features, in_features, dtype=torch.bfloat16)

        gpu_f16 = vulkan_ops._get_or_upload_weight(ctx, weight_f16, prefer_f16=True)
        gpu_f32 = vulkan_ops._get_or_upload_weight(ctx, weight_f32, prefer_f16=False)

        assert gpu_f16 is not None
        assert gpu_f32 is not None
        assert vulkan_ops._weight_uses_f16(ctx, weight_f16) is True
        assert gpu_f16.nbytes == out_features * in_features * 2
        assert gpu_f32.nbytes == out_features * in_features * 4
        assert gpu_f16.nbytes == gpu_f32.nbytes // 2

    def test_get_or_upload_weight_falls_back_to_f32_for_overflowing_weight(self):
        """A weight containing an out-of-f16-range value must still
        upload as float32 even when `prefer_f16=True` is passed — the
        safety fallback, not a crash or silent wrong data."""
        ctx = _require_vulkan_context()
        if "mul_mat_vec_f16_f32_f32" not in ctx.available_shaders():
            pytest.skip("mul_mat_vec_f16_f32_f32 shader unavailable")

        out_features, in_features = 16, 8
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        weight[0, 0] = 1e6

        gpu = vulkan_ops._get_or_upload_weight(ctx, weight, prefer_f16=True)
        assert gpu is not None
        assert vulkan_ops._weight_uses_f16(ctx, weight) is False
        assert gpu.nbytes == out_features * in_features * 4

    def test_vulkan_matvec_f16_weight_matches_cpu_reference(self):
        """End-to-end correctness: `_vulkan_matvec` dispatching the
        f16-weight shader (`mul_mat_vec_f16_f32_f32`) must match
        `torch.nn.functional.linear`'s float32 reference within the
        precision loss f16 weight storage alone accounts for."""
        ctx = _require_vulkan_context()
        if "mul_mat_vec_f16_f32_f32" not in ctx.available_shaders():
            pytest.skip("mul_mat_vec_f16_f32_f32 shader unavailable")

        out_features, in_features = 2048, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.1
        x = torch.randn(1, in_features, dtype=torch.float32)

        result = vulkan_ops._vulkan_matvec(ctx, x, weight)
        expected = torch.nn.functional.linear(x, weight, None)
        torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)

    def test_vulkan_matmul_f16_weight_matches_cpu_reference(self):
        """Same correctness check, for the prefill tiled-matmul path
        (`_vulkan_matmul` dispatching `matmul_f16_f32_fp32`)."""
        ctx = _require_vulkan_context()
        if "matmul_f16_f32_fp32" not in ctx.available_shaders():
            pytest.skip("matmul_f16_f32_fp32 shader unavailable")

        out_features, in_features = 2048, 1536
        weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.1
        x = torch.randn(128, in_features, dtype=torch.float32)

        result = vulkan_ops._vulkan_matmul(ctx, x, weight)
        expected = torch.nn.functional.linear(x, weight, None)
        torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)

    def test_rms_norm_then_linear_still_uses_f32_weight_unchanged(
        self, vulkan_ctx, upload_counter
    ):
        """`rms_norm_then_linear` (the fused decode-only RMSNorm+Linear
        path, unlike the plain `_vulkan_matvec`/`_vulkan_matmul` above)
        deliberately keeps calling `_get_or_upload_weight` without
        `prefer_f16=True` — it hardcodes dispatching
        `mul_mat_vec_f32_f32_f32`, not the f16-weight variant, so its
        weight upload must keep matching that shader's expected byte
        layout exactly, unchanged by this optimization."""
        norm_weight = torch.randn(8, dtype=torch.float32)
        linear_weight = torch.randn(16, 8, dtype=torch.float32)
        x = torch.randn(1, 8, dtype=torch.float32)

        vulkan_ops.rms_norm_then_linear(x, norm_weight, 1e-6, linear_weight, None)
        gpu = vulkan_ops._weight_cache.get(linear_weight)
        assert gpu is not None
        assert gpu.nbytes == 16 * 8 * 4, (
            "rms_norm_then_linear's linear_weight must still upload as "
            "float32 (matching the mul_mat_vec_f32_f32_f32 it hardcodes), "
            "not float16"
        )


class TestWrapRmsNormHoistsStaticLookups:
    """`_wrap_rms_norm` now captures `weight`/`eps`/the disable-ops env
    var once, at hook-install time, instead of re-deriving them (via
    `getattr`/`os.environ.get`) on every single forward call — see its
    doc comment in model_runner.py for the measured per-call overhead
    this avoids. These tests confirm the hoisting doesn't change
    behavior, and measure the actual speedup.
    """

    def test_wrapped_forward_matches_unwrapped_reference(self, vulkan_ctx):
        """Correctness: a wrapped RMSNorm module's forward() must produce
        the same result as computing RMSNorm directly, both with and
        without a residual — exercising every branch the hoisted values
        (`weight`, `eps`) are used in."""
        from vllm_vulkan.model_runner import _wrap_rms_norm

        class FakeRMSNorm(torch.nn.Module):
            def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(hidden_size))
                self.variance_epsilon = eps

            def forward(self, x, residual=None):
                if residual is not None:
                    x = x + residual
                xf = x.float()
                out = (
                    xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
                ) * self.weight.float()
                out = out.to(x.dtype)
                if residual is not None:
                    return out, x
                return out

            @property
            def eps(self) -> float:
                return self.variance_epsilon

        module = FakeRMSNorm(1536)
        x = torch.randn(1, 1536, dtype=torch.float32)

        expected_no_residual = module(x)
        _wrap_rms_norm(module)
        result_no_residual = module.forward(x)
        torch.testing.assert_close(
            result_no_residual, expected_no_residual, rtol=1e-3, atol=1e-4
        )

        module2 = FakeRMSNorm(1536)
        residual = torch.randn(1, 1536, dtype=torch.float32)
        expected_out, expected_residual_out = module2(x, residual)
        _wrap_rms_norm(module2)
        result_out, result_residual_out = module2.forward(x, residual)
        torch.testing.assert_close(result_out, expected_out, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(result_residual_out, expected_residual_out)

    def test_disable_ops_env_var_is_read_once_at_install_time(
        self, vulkan_ctx, monkeypatch
    ):
        """`VLLM_VULKAN_DISABLE_OPS` must be captured once when
        `_wrap_rms_norm` installs the hook, not re-read on every forward
        call — setting it *after* installation must have no effect on an
        already-wrapped module (matches `_returns_tuple`'s existing
        hoisted-at-install-time contract for `_wrap_linear`)."""
        from vllm_vulkan.model_runner import _wrap_rms_norm

        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.randn(8))
        module.variance_epsilon = 1e-6
        module.forward = lambda x, residual=None: ("orig-called", x, residual)

        monkeypatch.delenv("VLLM_VULKAN_DISABLE_OPS", raising=False)
        _wrap_rms_norm(module)

        monkeypatch.setenv("VLLM_VULKAN_DISABLE_OPS", "1")
        x = torch.randn(1, 8, dtype=torch.float32)
        result = module.forward(x)
        assert not (isinstance(result, tuple) and result[0] == "orig-called"), (
            "toggling VLLM_VULKAN_DISABLE_OPS after hook installation must not "
            "affect an already-wrapped module (the env var is captured once, "
            "at install time)"
        )

    def test_weight_float_conversion_happens_once_not_per_call(
        self, vulkan_ctx, monkeypatch
    ):
        """`vulkan_ops.rms_norm` -> `_cpu_rms_norm` calls `weight.float()`
        internally on every invocation — `_wrap_rms_norm` must convert
        `weight` to float32 once, at hook-install time (`weight_f32`),
        so that this internal `.float()` call becomes a no-op (already
        float32) instead of a fresh allocation+copy on every forward
        call. Counts real `Tensor.float()` calls on the *original* bf16
        weight object directly (via `monkeypatch`) rather than relying
        on timing, for a deterministic (not just probabilistic) check.
        """
        from vllm_vulkan.model_runner import _wrap_rms_norm

        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(
            torch.randn(1536, dtype=torch.bfloat16), requires_grad=False
        )
        module.variance_epsilon = 1e-6
        module.forward = lambda x, residual=None: x

        counts = {"n": 0}
        orig_float = torch.Tensor.float

        def counting_float(self):
            if self is module.weight:
                counts["n"] += 1
            return orig_float(self)

        monkeypatch.setattr(torch.Tensor, "float", counting_float)
        _wrap_rms_norm(module)
        assert counts["n"] == 1, (
            "weight.float() must be called exactly once, at hook-install "
            "time, not deferred to forward-call time"
        )

        x = torch.randn(1, 1536, dtype=torch.float32)
        for _ in range(5):
            module.forward(x)
        assert counts["n"] == 1, (
            "repeated forward() calls must not trigger additional "
            "weight.float() conversions of the original bf16 weight"
        )


class TestWrapLinearHoistsStaticLookups:
    """`_wrap_linear` now captures `weight`/`bias`/`skip_bias_add`/
    `tp_size`/`reduce_results`/`gather_output`/the disable-ops-or-linear
    env vars once, at hook-install time, instead of re-deriving them on
    every single forward call — see its doc comment in model_runner.py
    for the measured per-call overhead this avoids (and the also-removed
    redundant `x.float()` call, since `vulkan_ops.linear()` already does
    that conversion internally).
    """

    def test_wrapped_forward_matches_unwrapped_reference(self, vulkan_ctx):
        """Correctness: a wrapped Linear module's forward() must produce
        the same result as `torch.nn.functional.linear` directly — this
        also exercises the removed-redundant-`x.float()` change, since a
        bf16 `x` input now reaches `vulkan_ops.linear()` unconverted."""
        from vllm_vulkan.model_runner import _wrap_linear

        module = torch.nn.Linear(1536, 2048, bias=True)
        module.weight = torch.nn.Parameter(
            module.weight.detach().to(torch.bfloat16), requires_grad=False
        )
        module.bias = torch.nn.Parameter(
            module.bias.detach().to(torch.bfloat16), requires_grad=False
        )

        x = torch.randn(1, 1536, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(
            x.float(), module.weight.float(), module.bias.float()
        ).to(x.dtype)

        _wrap_linear(module)
        result = module.forward(x)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_disable_linear_env_vars_are_read_once_at_install_time(
        self, vulkan_ctx, monkeypatch
    ):
        """Same contract as `TestWrapRmsNormHoistsStaticLookups`'s
        equivalent test, for `_wrap_linear`'s two disable env vars."""
        from vllm_vulkan.model_runner import _wrap_linear

        module = torch.nn.Linear(1536, 2048, bias=True)

        monkeypatch.delenv("VLLM_VULKAN_DISABLE_OPS", raising=False)
        monkeypatch.delenv("VLLM_VULKAN_DISABLE_LINEAR", raising=False)
        _wrap_linear(module)

        monkeypatch.setenv("VLLM_VULKAN_DISABLE_LINEAR", "1")
        x = torch.randn(1, 1536, dtype=torch.float32)
        # Must still dispatch through vulkan_ops.linear() (i.e. behave as
        # if the env var were unset), since it was captured before this
        # setenv call.
        result = module.forward(x)
        expected = torch.nn.functional.linear(x, module.weight.float(), module.bias)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    def test_bias_float_conversion_happens_once_not_per_call(
        self, vulkan_ctx, monkeypatch
    ):
        """`vulkan_ops.linear` calls `bias.float()` internally on every
        invocation when a bias is present — `_wrap_linear` must convert
        `bias` to float32 once, at hook-install time (`bias_f32`), so
        this internal `.float()` call becomes a no-op instead of a fresh
        allocation+copy on every forward call. Same deterministic
        call-counting approach as
        `TestWrapRmsNormHoistsStaticLookups.test_weight_float_conversion_happens_once_not_per_call`,
        applied to `bias` here. Uses a large-enough weight
        (`_MATVEC_MIN_WEIGHT_ELEMENTS`-sized) so this exercises the real
        GPU dispatch path where `vulkan_ops.linear()`'s own `bias.float()`
        call actually runs (T=1, weight_elements >= threshold).
        """
        from vllm_vulkan.model_runner import _wrap_linear

        module = torch.nn.Linear(1536, 2048, bias=True)
        module.weight = torch.nn.Parameter(
            module.weight.detach().to(torch.bfloat16), requires_grad=False
        )
        module.bias = torch.nn.Parameter(
            module.bias.detach().to(torch.bfloat16), requires_grad=False
        )

        counts = {"n": 0}
        orig_float = torch.Tensor.float

        def counting_float(self):
            if self is module.bias:
                counts["n"] += 1
            return orig_float(self)

        monkeypatch.setattr(torch.Tensor, "float", counting_float)
        _wrap_linear(module)
        assert counts["n"] == 1, (
            "bias.float() must be called exactly once, at hook-install time, "
            "not deferred to forward-call time"
        )

        x = torch.randn(1, 1536, dtype=torch.float32)
        for _ in range(5):
            module.forward(x)
        assert counts["n"] == 1, (
            "repeated forward() calls must not trigger additional "
            "bias.float() conversions of the original bf16 bias"
        )

    def test_hoisted_lookups_are_faster_than_per_call_getattr_and_environ(self):
        """Measures the actual speedup of closure-captured static values
        vs. re-deriving them via `getattr`/`os.environ.get` on every
        call — the exact class of overhead this change removes from
        `_wrap_rms_norm`/`_wrap_linear`. Pure Python; doesn't need a real
        Vulkan device or even a real nn.Module (a minimal stand-in with
        the same attributes suffices, since only attribute-lookup cost
        is being measured here, not any Vulkan dispatch).

        Takes the minimum elapsed time across several independent trials
        — see `TestLinearMatvecDispatchThreshold`'s similarly-timed test
        for why (measurement noise under full-suite load).
        """
        import os
        import time

        class FakeModule:
            def __init__(self) -> None:
                self.weight = object()
                self.bias = object()
                self.skip_bias_add = False
                self.tp_size = 1
                self.reduce_results = False
                self.gather_output = False

        module = FakeModule()
        iters = 5000
        trials = 5

        def per_call_lookups() -> None:
            os.environ.get("VLLM_VULKAN_DISABLE_OPS")
            os.environ.get("VLLM_VULKAN_DISABLE_LINEAR")
            getattr(module, "weight", None)
            getattr(module, "bias", None)
            getattr(module, "skip_bias_add", False)
            tp_size = getattr(module, "tp_size", 1)
            getattr(module, "reduce_results", False) and tp_size > 1
            getattr(module, "gather_output", False) and tp_size > 1

        # Hoisted equivalents, captured once (mirrors what _wrap_linear
        # now does at install time).
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        skip_bias = getattr(module, "skip_bias_add", False)
        tp_size = getattr(module, "tp_size", 1)
        row_reduce = getattr(module, "reduce_results", False) and tp_size > 1
        gather_output = getattr(module, "gather_output", False) and tp_size > 1
        disable_linear = bool(
            os.environ.get("VLLM_VULKAN_DISABLE_OPS")
            or os.environ.get("VLLM_VULKAN_DISABLE_LINEAR")
        )

        def hoisted_lookups() -> None:
            # Just references the already-captured closure variables --
            # this is what the actual vk_forward closure body does now.
            _ = (
                weight,
                bias,
                skip_bias,
                tp_size,
                row_reduce,
                gather_output,
                disable_linear,
            )

        def timed(fn) -> float:
            best = float("inf")
            for _ in range(trials):
                t0 = time.perf_counter()
                for _ in range(iters):
                    fn()
                best = min(best, time.perf_counter() - t0)
            return best

        per_call_elapsed = timed(per_call_lookups)
        hoisted_elapsed = timed(hoisted_lookups)

        print(
            f"\nmodel_runner hook lookups, best-of-{trials}: "
            f"per-call {per_call_elapsed / iters * 1e6:.3f}us/call, "
            f"hoisted {hoisted_elapsed / iters * 1e6:.3f}us/call, "
            f"speedup {per_call_elapsed / hoisted_elapsed:.1f}x"
        )
        assert hoisted_elapsed < per_call_elapsed, (
            f"hoisted lookups ({hoisted_elapsed:.4f}s/{iters}) were not faster "
            f"than per-call getattr/os.environ.get ({per_call_elapsed:.4f}s/{iters})"
        )


class TestExecuteBatchChainRef:
    """`VulkanContext.execute_batch`'s `(op_index, output_index)` chain-ref
    binding: lets a later op in the same batch read an EARLIER op's output
    directly, without it ever round-tripping through Python/CPU, so a
    multi-op sequence still costs only one `vkQueueSubmit`. See its own doc
    comment in src/lib.rs for the full binding-resolution contract this
    exercises.

    The mechanism's actual numerical correctness is validated thoroughly
    by `TestFusedSwigluMlp` below (every one of those test cases exercises
    two chain-refs per call: gate_up matmul -> SwiGLU, and SwiGLU -> down
    matmul, checked against a real CPU/torch reference across 7 shapes,
    bf16, and edge cases) -- this class only covers the input-validation
    contract that doesn't need a full, real GPU dispatch (and therefore
    doesn't need to get an unrelated shader's own push-constant struct
    layout right) to exercise.
    """

    def test_chain_ref_to_a_later_op_is_rejected(self, vulkan_ctx):
        """`op_index` must be strictly earlier than the referencing op --
        its output buffer doesn't exist yet otherwise (see execute_batch's
        Phase 1/Phase 2 doc comments in src/lib.rs: buffers are allocated
        up-front, in op order, so a forward/self reference would read an
        unrelated or not-yet-written buffer rather than erroring, if this
        check weren't there)."""
        ctx = vulkan_ctx
        n = 64
        pc = bytes(64)  # exact content is irrelevant: rejected before dispatch
        with pytest.raises(RuntimeError, match="must reference an EARLIER op"):
            ctx.execute_batch(
                [
                    (
                        "swiglu_f32",
                        [(0, 0), (0, 0)],  # op 0 referencing itself
                        [n * 4],
                        pc,
                        (1, 1, 1),
                        False,
                    ),
                ]
            )


class TestFusedSwigluMlp:
    """`vulkan_ops.fused_swiglu_mlp`: `gate_up_proj -> silu(gate)*up ->
    down_proj` (Qwen2MLP/Qwen3MLP's exact structure) in ONE `vkQueueSubmit`
    via `execute_batch`'s chain-ref bindings, instead of `model_runner.py`'s
    `_wrap_linear` hooks independently dispatching `gate_up_proj` and
    `down_proj` as two separate submits with a CPU-side activation in
    between. See `model_runner.py`'s `_wrap_swiglu_mlp` for the real
    integration point.
    """

    @staticmethod
    def _reference(x, gate_up_weight, down_weight):
        gate_up = torch.nn.functional.linear(x.float(), gate_up_weight.float())
        d = gate_up.shape[-1] // 2
        gate, up = gate_up[..., :d], gate_up[..., d:]
        act = torch.nn.functional.silu(gate) * up
        return torch.nn.functional.linear(act, down_weight.float())

    @pytest.mark.parametrize("t", [1, 2, 3, 4, 5, 16, 128])
    def test_matches_torch_reference_across_decode_and_prefill_shapes(
        self, vulkan_ctx, t
    ):
        torch.manual_seed(0)
        hidden, intermediate = 1024, 3072
        gate_up_weight = (
            torch.randn(2 * intermediate, hidden, dtype=torch.float32) * 0.02
        )
        down_weight = torch.randn(hidden, intermediate, dtype=torch.float32) * 0.02
        x = torch.randn(t, hidden, dtype=torch.float32)

        result = vulkan_ops.fused_swiglu_mlp(x, gate_up_weight, down_weight)
        expected = self._reference(x, gate_up_weight, down_weight)

        assert result.shape == expected.shape
        # rtol/atol matching this file's other f16-weight-upload tolerances
        # (see test_linear_matches_torch_reference_at_realistic_hidden_size's
        # doc comment) -- fused_swiglu_mlp uploads both weights as f16 via
        # the same _get_or_upload_weight/prefer_f16=True path linear() uses.
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=5e-2)

    def test_matches_torch_reference_with_bf16_weights_and_input(self, vulkan_ctx):
        torch.manual_seed(1)
        hidden, intermediate = 1024, 3072
        gate_up_weight = (
            torch.randn(2 * intermediate, hidden, dtype=torch.float32) * 0.02
        ).to(torch.bfloat16)
        down_weight = (
            torch.randn(hidden, intermediate, dtype=torch.float32) * 0.02
        ).to(torch.bfloat16)
        x = torch.randn(1, hidden, dtype=torch.bfloat16)

        result = vulkan_ops.fused_swiglu_mlp(x, gate_up_weight, down_weight)
        expected = self._reference(x, gate_up_weight, down_weight)

        # Confirmed regression guard: fused_swiglu_mlp itself always
        # computes/returns float32 (GPU compute happens in float32/float16,
        # never bf16) -- model_runner.py's _wrap_swiglu_mlp's vk_mlp_forward
        # is responsible for the `.to(x.dtype)` conversion back to the
        # caller's dtype, NOT this function. A caller that forgot that
        # conversion previously crashed one full layer downstream inside
        # the CPU attention kernel ("RuntimeError: expected scalar type
        # Float but found BFloat16"), not here -- so this test only checks
        # numerical correctness in float32, matching this function's real
        # contract.
        assert result.dtype == torch.float32
        torch.testing.assert_close(result, expected.float(), rtol=1e-2, atol=5e-2)

    def test_raises_fused_mlp_unavailable_for_small_weights(self, vulkan_ctx):
        small_gate_up = torch.randn(64, 32, dtype=torch.float32)
        small_down = torch.randn(32, 32, dtype=torch.float32)
        x = torch.randn(1, 32, dtype=torch.float32)
        with pytest.raises(vulkan_ops.FusedMlpUnavailableError):
            vulkan_ops.fused_swiglu_mlp(x, small_gate_up, small_down)

    def test_weight_cache_is_reused_across_repeated_calls(
        self, vulkan_ctx, upload_counter
    ):
        hidden, intermediate = 1024, 3072
        gate_up_weight = torch.randn(2 * intermediate, hidden, dtype=torch.float32)
        down_weight = torch.randn(hidden, intermediate, dtype=torch.float32)

        vulkan_ops.fused_swiglu_mlp(torch.randn(1, hidden), gate_up_weight, down_weight)
        assert upload_counter["n"] == 2, "first call should upload both weights once"

        vulkan_ops.fused_swiglu_mlp(torch.randn(1, hidden), gate_up_weight, down_weight)
        assert upload_counter["n"] == 2, (
            "second call (same weight objects) must not re-upload either weight"
        )
