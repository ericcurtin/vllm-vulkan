# SPDX-License-Identifier: Apache-2.0
"""VulkanModelRunner — GPU-accelerated model runner.

``VLLM_VULKAN_RUST_MODEL`` gates whether Vulkan dispatch hooks (RMSNorm,
Linear -> Rust ``vulkan_ops`` via ``vllm_vulkan._rs``) are installed on the
loaded PyTorch model:

  1. VLLM_VULKAN_RUST_MODEL=1 (default): patch individual PyTorch modules
     (RMSNorm, Linear) to dispatch to Vulkan.

  2. VLLM_VULKAN_RUST_MODEL=0: no Vulkan dispatch; the model runs unmodified
     on the CPU backend. Useful for isolating CPU-only behaviour when
     debugging numerical differences.

Note: the standalone Rust ``VulkanModel`` (a from-scratch, fully-fused
Gemma4 forward pass, see ``vllm_vulkan._rs.VulkanModel`` / ``src/lib.rs``)
is a separate, complete GPU decode engine — it is not
wired into vLLM's scheduler/paged-KV-cache here (that would require bypassing
vLLM's attention backend entirely). It is exercised directly via
``scripts/bench_vulkan_model.py``. An earlier version of this module loaded
that Rust model on every startup and then discarded it after logging its
metadata (nothing here ever called its ``forward()``), which cost ~25-30s of
extra startup latency and ~7GB of duplicate GPU-resident weight memory for
zero benefit. That dead load has been removed.
"""

import logging
import os

import torch
import torch.nn as nn

from vllm_vulkan import envs

logger = logging.getLogger(__name__)


def _use_rust_model() -> bool:
    return envs.VLLM_VULKAN_RUST_MODEL


class _VulkanCPUModelRunner:
    """CPUModelRunner that patches the loaded model for Vulkan dispatch.

    Wraps CPUModelRunner and intercepts load_model to additionally patch the
    PyTorch model's RMSNorm/Linear modules to dispatch to Vulkan (via
    ``vllm_vulkan._rs``) instead of running on CPU.
    """

    def __init__(self, vllm_config, device):
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner  # noqa: PLC0415

        self._runner = CPUModelRunner(vllm_config, device)
        self._vllm_config = vllm_config

    def __getattr__(self, name: str):
        return getattr(self._runner, name)

    def load_model(self, **kwargs) -> None:
        """Load the PyTorch model, then patch it for Vulkan dispatch."""
        self._runner.load_model(**kwargs)

        if not _use_rust_model():
            return

        try:
            _apply_module_hooks(self._runner.model)
        except Exception as exc:
            logger.warning("Failed to install Vulkan dispatch hooks: %s", exc)

    # Delegate all other methods to the underlying runner.
    def warming_up_model(self) -> None:
        self._runner.warming_up_model()

    def execute_model(self, *args, **kwargs):
        return self._runner.execute_model(*args, **kwargs)

    def initialize_kv_cache(self, *args, **kwargs):
        self._patch_kv_sharing_in_runner()
        return self._runner.initialize_kv_cache(*args, **kwargs)

    def _patch_kv_sharing_in_runner(self) -> None:
        """Monkey-patch maybe_add_kv_sharing_layers_to_kv_cache_groups on the
        inner runner to also populate UniformTypeKVCacheSpecs.kv_cache_specs
        for KV-sharing layers.

        Background: vLLM's initialize_attn_backend looks up every layer name
        from kv_cache_group.layer_names in UniformTypeKVCacheSpecs.kv_cache_specs.
        For KV-sharing layers (e.g. Gemma4 layers 15-34), vLLM's
        maybe_add_kv_sharing_layers_to_kv_cache_groups appends them to
        layer_names but does NOT add them to kv_cache_specs, causing a KeyError.
        This patch fixes that by copying the target layer's spec.
        """
        try:
            from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs  # noqa: PLC0415, I001
        except ImportError:
            return

        runner = self._runner
        orig_method = runner.__class__.maybe_add_kv_sharing_layers_to_kv_cache_groups

        def _patched_maybe_add_kv_sharing_layers_to_kv_cache_groups(
            self_inner, kv_cache_config
        ):
            # Call the original implementation first (appends to layer_names).
            orig_method(self_inner, kv_cache_config)

            # Then fix up UniformTypeKVCacheSpecs.kv_cache_specs to include
            # the KV-sharing layers, using the target layer's spec.
            shared = getattr(self_inner, "shared_kv_cache_layers", {})
            if not shared:
                return
            for group in kv_cache_config.kv_cache_groups:
                spec = group.kv_cache_spec
                if not isinstance(spec, UniformTypeKVCacheSpecs):
                    continue
                for layer_name, target_name in shared.items():
                    if (
                        layer_name not in spec.kv_cache_specs
                        and target_name in spec.kv_cache_specs
                    ):
                        spec.kv_cache_specs[layer_name] = spec.kv_cache_specs[
                            target_name
                        ]
                        logger.debug(
                            "KV-sharing fix: added kv_cache_specs[%s] = spec of %s",
                            layer_name,
                            target_name,
                        )

        import types  # noqa: PLC0415

        runner.maybe_add_kv_sharing_layers_to_kv_cache_groups = types.MethodType(
            _patched_maybe_add_kv_sharing_layers_to_kv_cache_groups, runner
        )

    def get_kv_cache_spec(self, *args, **kwargs):
        return self._runner.get_kv_cache_spec(*args, **kwargs)

    def determine_available_memory(self, *args, **kwargs):
        return self._runner.determine_available_memory(*args, **kwargs)


def _apply_module_hooks(model: nn.Module) -> None:
    """Apply Vulkan dispatch hooks to RMSNorm and Linear modules.

    Note: there is a separate, complete Rust decode engine
    (``vllm_vulkan._rs.VulkanModel``, see ``src/lib.rs``) that fuses an
    entire Gemma4 forward pass into a handful of GPU submits per token. It is
    not used here — wiring it in would mean bypassing vLLM's own paged
    KV-cache/attention backend for this model, which this hook-based
    per-module approach deliberately avoids so it keeps working with vLLM's
    scheduler, sampler, and KV-cache management unmodified. See
    ``scripts/bench_vulkan_model.py`` to exercise that engine directly.
    """
    counts = {"rms_norm": 0, "linear": 0}
    for _name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name == "RMSNorm":
            _wrap_rms_norm(module)
            counts["rms_norm"] += 1
        elif cls_name in (
            "Linear",
            "RowParallelLinear",
            "ColumnParallelLinear",
            "QKVParallelLinear",
            "MergedColumnParallelLinear",
            "ReplicatedLinear",
        ):
            _wrap_linear(module)
            counts["linear"] += 1
    logger.info(
        "Patched: %d RMSNorm, %d Linear modules for Vulkan dispatch.",
        counts["rms_norm"],
        counts["linear"],
    )

    # Pre-upload all weights to GPU.
    try:
        from vllm_vulkan import vulkan_ops  # noqa: PLC0415
        from vllm_vulkan._rs import VulkanContext  # noqa: PLC0415

        if not vulkan_ops.is_ready():
            ctx = VulkanContext(0)
            vulkan_ops.set_context(ctx)

        ctx = vulkan_ops.get_context()
        total_bytes = 0
        for _name, param in model.named_parameters():
            w = param.data
            if w.numel() > 0:
                vulkan_ops._get_or_upload_weight(ctx, w)
                total_bytes += w.nbytes
        logger.info("Pre-uploaded %.1fGB of weights to Vulkan GPU.", total_bytes / 1e9)
    except Exception as exc:
        logger.warning("Failed to pre-upload weights: %s", exc)


def _wrap_rms_norm(module: nn.Module) -> None:
    orig = module.forward
    # Captured once, at hook-install time, rather than re-derived on every
    # forward call: `_apply_module_hooks` (this function's only caller,
    # via `_VulkanCPUModelRunner.load_model`) runs strictly after
    # `self._runner.load_model(**kwargs)` has fully populated every
    # module's `.weight` Parameter and static config, so none of these
    # ever change again for the lifetime of this wrapped module -- the
    # same reasoning `_wrap_linear`'s `_returns_tuple` below already
    # relies on for its own hoisted value. `os.environ.get(...)` also
    # can't meaningfully change mid-process (env vars are read once at
    # process/worker startup in every realistic deployment), so it's
    # hoisted for the same reason. Measured on this hardware: each of
    # these was a small (~0.2-0.5us) but purely repeated cost, paid once
    # per RMSNorm module, per decoder layer, per generated token
    # (~175 calls/decode-step for Gemma4-E2B's 35 layers) -- see
    # `TestWrapRmsNormHoistsStaticLookups`/`TestWrapLinearHoistsStaticLookups`
    # (tests/python/test_vulkan_ops.py) for the measurement and the
    # equivalent fix applied to `_wrap_linear` just below.
    weight = getattr(module, "weight", None)
    # `vulkan_ops.rms_norm` (always CPU now, see its own doc comment) ->
    # `_cpu_rms_norm` does `weight.float()` internally on every call --
    # for a bf16/fp16-loaded model (the common case), that's a fresh
    # [hidden_size]-sized allocation + copy every single time. Since
    # `weight` itself is already known to be stable for this module's
    # lifetime (see above), so is its float32 conversion -- computed
    # once here instead, and passed straight through `_cpu_rms_norm`'s
    # `weight.float()` as a no-op (`Tensor.float()` on an already-float32
    # tensor returns `self`, no new allocation). Measured ~0.9-1.1us/call
    # for Gemma4-E2B's real RMSNorm weight sizes (256/1536 elements).
    weight_f32 = weight.float() if weight is not None else None
    eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
    disable_ops = bool(os.environ.get("VLLM_VULKAN_DISABLE_OPS"))

    def vk_forward(x: torch.Tensor, residual=None):
        from vllm_vulkan import vulkan_ops  # noqa: PLC0415

        if not vulkan_ops.is_ready() or disable_ops:
            return orig(x, residual)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return orig(x, residual)

        if residual is not None:
            x = x + residual
            residual_out = x

        try:
            result = vulkan_ops.rms_norm(x, weight_f32, eps)
            result = result.to(x.dtype)
        except Exception as exc:
            logger.debug("Vulkan rms_norm failed (%s)", exc)
            result = orig(x, None)
            if isinstance(result, tuple):
                result = result[0]

        if residual is not None:
            return result, residual_out
        return result

    module.forward = vk_forward


def _wrap_linear(module: nn.Module) -> None:
    orig = module.forward
    _returns_tuple = getattr(module, "return_bias", False) or getattr(
        module, "skip_bias_add", False
    )
    # Captured once, at hook-install time, rather than re-derived on every
    # forward call — see `_wrap_rms_norm`'s doc comment just above for
    # why this is safe (hooks install only after `load_model()` has
    # fully populated every module's weight/bias Parameters and
    # parallelism config, and env vars don't change mid-process).
    # Measured on this hardware: `os.environ.get()` costs ~0.3-0.5us/call
    # and `getattr()` costs ~0.2us/call; with ~175-210 Linear-family
    # module calls/decode-step across Gemma4-E2B's 35 layers (2 env
    # checks + up to 4 getattrs each, previously), that's on the order
    # of 150-250us/decode-step of pure, repeated, avoidable lookups now
    # paid once at hook-install time instead.
    weight = getattr(module, "weight", None)
    weight_dtype_ok = weight is not None and weight.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    )
    bias = getattr(module, "bias", None)
    # vulkan_ops.linear() does `bias.float()` internally on every call
    # for a non-None bias -- same class of repeated-per-call allocation
    # as `_wrap_rms_norm`'s `weight.float()` above (measured
    # ~0.9-1.7us/call for Gemma4-E2B's real Linear-module bias sizes),
    # avoided the same way: converted once here instead. `matmul_bias`
    # (passed to vulkan_ops.linear()) uses the float32 copy; the
    # row_reduce path below deliberately keeps using the original
    # (possibly non-float32) `bias`, since that addition happens after
    # `result` has already been cast back to `x.dtype` and doesn't go
    # through vulkan_ops.linear()'s own `.float()` call at all.
    bias_f32 = bias.float() if bias is not None else None
    skip_bias = getattr(module, "skip_bias_add", False)
    tp_size = getattr(module, "tp_size", 1)
    # RowParallelLinear sums partial results across tensor-parallel ranks and
    # applies bias after the reduction, not inside the (sharded) matmul. The
    # bare matmul below produces only this rank's partial sum, so without the
    # all-reduce the model emits garbage under tensor parallelism.
    row_reduce = getattr(module, "reduce_results", False) and tp_size > 1
    matmul_bias = None if (row_reduce or skip_bias) else bias_f32
    gather_output = getattr(module, "gather_output", False) and tp_size > 1
    disable_linear = bool(
        os.environ.get("VLLM_VULKAN_DISABLE_OPS")
        or os.environ.get("VLLM_VULKAN_DISABLE_LINEAR")
    )

    def vk_forward(x: torch.Tensor, *args, **kwargs):
        from vllm_vulkan import vulkan_ops  # noqa: PLC0415

        if not vulkan_ops.is_ready() or args or kwargs:
            return orig(x, *args, **kwargs)
        if disable_linear:
            return orig(x, *args, **kwargs)
        if not weight_dtype_ok:
            return orig(x, *args, **kwargs)

        try:
            # weight is passed through as-is (NOT weight.float()) so its
            # tensor identity - and hence vulkan_ops's persistent GPU
            # weight cache, keyed on id(weight.untyped_storage()) - stays
            # stable across every call. weight.float() would allocate a
            # brand-new tensor (and storage) whenever weight isn't already
            # float32 (true for any bf16/fp16-loaded model, the common
            # case), silently defeating the cache and forcing a full
            # weight-matrix re-upload to the GPU on every single forward
            # call instead of once. vulkan_ops.linear()/_vulkan_matvec()
            # do their own float32 conversion internally, lazily, only on
            # a cache miss (see _get_or_upload_weight).
            #
            # `x` is passed through as-is too (NOT x.float()): linear()
            # already does that conversion internally on its own first
            # line, so converting here first was pure duplicated work
            # (measured ~0.34us/call -- small individually, but the same
            # class of avoidable per-call cost as everything else hoisted
            # in this function).
            result = vulkan_ops.linear(x, weight, matmul_bias)
            result = result.to(x.dtype)
        except Exception as exc:
            logger.debug("Vulkan linear failed (%s)", exc)
            return orig(x, *args, **kwargs)

        # Collectives run outside the try/except, and only on the success path. A
        # rank whose matmul failed has already returned orig(), which runs the
        # same collective, so every rank performs exactly one collective and stays
        # in lockstep. Catching a collective here and re-running it via orig()
        # would issue it twice on that rank and deadlock the group.
        if row_reduce:
            from vllm.distributed import (  # noqa: PLC0415
                tensor_model_parallel_all_reduce,
            )

            result = tensor_model_parallel_all_reduce(result)
            if bias is not None and not skip_bias:
                result = result + bias
        elif gather_output:
            from vllm.distributed import (  # noqa: PLC0415
                tensor_model_parallel_all_gather,
            )

            result = tensor_model_parallel_all_gather(result)

        if _returns_tuple:
            output_bias = module.bias if skip_bias else None
            return result, output_bias
        return result

    module.forward = vk_forward
