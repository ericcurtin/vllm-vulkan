# SPDX-License-Identifier: Apache-2.0
"""Pure-Python / PyTorch fallbacks for vllm._C compiled operations.

When vLLM is used from its Python source tree (without building the C++
extensions), certain CPU-specific operations that are normally implemented in
``vllm._C`` are unavailable.  This module provides drop-in replacements that
are semantically equivalent to the C++ originals.

Call :func:`apply_patches` once at startup (from the platform plugin) to
monkey-patch the relevant vLLM modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_patches_applied = False

_TEMPLATES_DIR = Path(__file__).parent

# Keeps torch.library.Library fragments alive for the process lifetime; a
# Library object unregisters all of its ops when garbage-collected, so a
# module-level reference is required for _patch_cpu_memory_env's fallback op
# to remain callable.
_KEEPALIVE_LIBS: list[torch.library.Library] = []


def apply_patches() -> None:
    """Apply all patches.  Safe to call multiple times (idempotent)."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    try:
        _patch_cpu_memory_env()
    except Exception as exc:
        logger.warning("Failed to apply vllm._C.init_cpu_memory_env patch: %s", exc)

    try:
        _patch_cpu_triton_utils()
    except Exception as exc:
        logger.warning("Failed to apply vllm._C patches: %s", exc)

    try:
        _patch_topk_topp_triton()
    except Exception as exc:
        logger.warning("Failed to apply top-k/top-p Triton-CUDA guard: %s", exc)

    try:
        _register_gemma4_chat_template()
    except Exception as exc:
        logger.warning("Failed to register gemma4 chat template fallback: %s", exc)

    try:
        _patch_accelerator_synchronize()
    except Exception as exc:
        logger.warning("Failed to apply torch.accelerator.synchronize guard: %s", exc)


# ---------------------------------------------------------------------------
# Patch: torch.accelerator.synchronize() on platforms with no torch accelerator
# ---------------------------------------------------------------------------


def _patch_accelerator_synchronize() -> None:
    """Make ``torch.accelerator.synchronize()``/``empty_cache()`` no-ops when
    no torch accelerator (CUDA/XPU/etc.) is registered.

    ``vllm.v1.worker.gpu_model_runner.GPUModelRunner._cleanup_profiling_kv_cache``
    (reused by ``_VulkanCPUModelRunner`` via ``CPUModelRunner``) unconditionally
    calls both during worker shutdown. Their docstrings say they're a no-op
    if the current accelerator is not initialized, but the implementation
    doesn't actually honor that for a device_type="cpu" platform like this
    one (no torch accelerator is ever registered): each raises
        RuntimeError: Cannot access accelerator device when none is available.
    This doesn't corrupt or lose any already-generated output (it only fires
    during worker teardown, after results have been returned), but it turns
    every clean shutdown -- including a normal ``LLM()`` object being garbage
    collected, or ``vllm serve`` receiving Ctrl-C -- into a scary, confusing
    traceback in the logs. Guard both so shutdown is quiet when there truly
    is no accelerator.
    """
    if not hasattr(torch, "accelerator"):
        return

    for name in ("synchronize", "empty_cache"):
        orig_func = getattr(torch.accelerator, name, None)
        if orig_func is None:
            continue

        def _make_safe(orig):  # noqa: ANN001
            def _safe(*args, **kwargs):
                if not torch.accelerator.is_available():
                    return None
                return orig(*args, **kwargs)

            return _safe

        setattr(torch.accelerator, name, _make_safe(orig_func))

    logger.debug(
        "Guarded torch.accelerator.synchronize()/empty_cache() to no-op "
        "when no torch accelerator is registered (Vulkan/CPU platform)."
    )


# ---------------------------------------------------------------------------
# Patch: vllm.v1.sample.ops.topk_topp_sampler's Triton fast path
# ---------------------------------------------------------------------------


def _patch_topk_topp_triton() -> None:
    """Force the PyTorch top-k/top-p sampling fallback on this platform.

    ``vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p`` unconditionally
    dispatches to ``apply_top_k_top_p_triton`` whenever ``vllm.triton_utils
    .HAS_TRITON`` is true and the batch size is >= 8 -- with no platform
    check. That Triton kernel is CUDA-only (it launches through
    ``triton.backends.nvidia.driver``), so it crashes as soon as
    ``triton`` happens to be importable in the environment, which is common:
    it is a normal transitive dependency of vLLM (e.g. via structured-output
    backends) even for CPU-only installs, so a Vulkan-plugin user need not
    have installed it deliberately. Since this platform's tensors always sit
    on ``torch.device("cpu")`` (compute is offloaded to the GPU only inside
    the model's own forward pass, not vLLM's post-hoc sampler), the kernel's
    ``ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu
    tensor?)`` fires on essentially every request with a batch size >= 8.

    We patch ``HAS_TRITON`` to ``False`` inside the already-imported
    ``topk_topp_sampler`` module only (not globally), so the module's
    ``apply_top_k_top_p`` falls back to ``apply_top_k_top_p_pytorch``, which
    is a plain, device-agnostic PyTorch sort-based implementation.
    """
    # NOTE: this is deliberately *not* only called from apply_patches() at
    # plugin-registration time. That runs extremely early (from inside
    # vLLM's own lazy ``current_platform`` resolution, itself triggered
    # while ``vllm.config`` and friends may still be mid-import), so
    # importing ``vllm.v1.sample.ops.topk_topp_sampler`` at that point
    # reliably raises:
    #   ImportError: cannot import name 'CUDAGraphMode' from partially
    #   initialized module 'vllm.config' (most likely due to a circular
    #   import)
    # which is swallowed here and by apply_patches()'s caller, silently
    # making the patch a no-op. ``VulkanWorker.init_device()`` also calls
    # this function directly, at a point where vLLM's module graph is
    # already fully imported, which is where this patch actually takes
    # effect in practice.
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as _sampler_mod  # noqa: PLC0415
    except ImportError:
        return

    if not getattr(_sampler_mod, "HAS_TRITON", False):
        return  # Already false; e.g. triton genuinely not installed.

    _sampler_mod.HAS_TRITON = False
    logger.info(
        "Disabled vLLM's CUDA-only Triton top-k/top-p sampling kernel for "
        "the Vulkan/CPU platform; falling back to the PyTorch "
        "implementation (vllm.v1.sample.ops.topk_topp_sampler."
        "apply_top_k_top_p_pytorch)."
    )


# ---------------------------------------------------------------------------
# Patch: vllm.utils.cpu_triton_utils.compute_slot_mapping_kernel
# ---------------------------------------------------------------------------


def _pure_python_compute_slot_mapping(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,  # [num_reqs + 1], int32
    positions: torch.Tensor,  # [num_tokens], int64
    block_table: torch.Tensor,  # [max_num_reqs, max_num_blocks_per_req], int32
    block_table_stride: int,
    block_size: int,
    slot_mapping: torch.Tensor,  # [max_num_tokens], int64
    TOTAL_CP_WORLD_SIZE: int = 1,  # noqa: N803
    TOTAL_CP_RANK: int = 0,  # noqa: N803
    CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,  # noqa: N803
    PAD_ID: int = -1,  # noqa: N803
    BLOCK_SIZE: int = 1024,  # noqa: N803
) -> None:
    """Pure-PyTorch implementation of ``vllm._C.compute_slot_mapping_kernel_impl``.

    Maps each token to its KV-cache slot.  For each request r and each token
    at position p in that request::

        block_id   = block_table[r, p // block_size]
        slot       = block_id * block_size + p % block_size

    Context Parallelism (TOTAL_CP_WORLD_SIZE > 1) is not supported on CPU,
    consistent with the original C++ implementation.

    Vectorized across every request in one pass instead of a Python loop
    over range(num_reqs): the original version called `.item()` twice per
    request (forcing a CPU/tensor sync each time) purely to slice
    `positions`/`slot_mapping` and index into `block_table` one request row
    at a time - work that `torch.repeat_interleave` (to build a per-token
    "which request do I belong to" index from `query_start_loc` without a
    loop) plus a single fancy-indexed gather from `block_table` does in one
    shot for the whole batch. This runs once per forward pass (prefill and
    decode) whenever `vllm._C` isn't available, so the per-request Python/
    tensor-sync overhead scaled with concurrent batch size on every call.
    """
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on CPU."

    num_reqs = query_start_loc.shape[0] - 1
    if num_reqs <= 0 or num_tokens <= 0:
        return

    positions = positions[:num_tokens]

    # Per-token request index, built without a Python loop: repeat request
    # index r exactly (query_start_loc[r+1] - query_start_loc[r]) times -
    # requests with zero tokens this step (start == end) are naturally
    # skipped, matching the original `if start >= end: continue`.
    #
    # query_start_loc is frequently a CPU tensor (often pinned memory) even
    # when positions/block_table live on the active accelerator device, so
    # req_ids_per_token (used to index block_table alongside block_indices,
    # which is derived from positions) must be built on positions.device,
    # not query_start_loc.device - indexing with tensors on mismatched
    # devices raises a RuntimeError.
    req_lens = (
        (query_start_loc[1:] - query_start_loc[:-1])
        .clamp(min=0)
        .to(device=positions.device, dtype=torch.int64)
    )
    req_ids_per_token = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int64, device=positions.device),
        req_lens,
    )

    block_indices = (positions // block_size).long()
    block_ids = block_table[req_ids_per_token, block_indices].long()
    offsets = (positions % block_size).long()
    slots = block_ids * block_size + offsets

    slot_mapping[:num_tokens] = slots


class _FuncWrapper:
    """Mimics the ``_FuncWrapper`` used in vllm.utils.cpu_triton_utils."""

    def __init__(self, func) -> None:
        self.func = func

    def __getitem__(self, *args, **kwargs):
        return self.func


# ---------------------------------------------------------------------------
# Patch: vllm._C.init_cpu_memory_env
# ---------------------------------------------------------------------------


def _patch_cpu_memory_env() -> None:
    """Register a no-op fallback for ``vllm._C.init_cpu_memory_env``.

    ``CPUWorker.__init__`` (vLLM's base class for our ``VulkanWorker``) calls
    ``torch.ops._C.init_cpu_memory_env(...)`` unconditionally, before
    ``VulkanWorker.init_device()`` even runs, to bind the process to a NUMA
    memory node for the CPU backend's AVX-optimized kernels.

    Some vLLM CPU wheels (observed on aarch64 builds, where the heavy
    AVX-512 CPU kernel sources are not compiled) ship a ``vllm._C`` module
    that imports successfully but registers *zero* ops under the ``_C``
    torch.library namespace. ``_patch_cpu_triton_utils`` below already
    handles the case where ``import vllm._C`` fails outright, but it cannot
    detect this "importable but empty" case, so ``torch.ops._C.init_cpu_memory_env``
    raises an uncaught ``AttributeError`` and engine startup crashes:

        AttributeError: '_OpNamespace' '_C' object has no attribute
        'init_cpu_memory_env'

    The Vulkan plugin runs on ``device_type=cpu`` purely as a host/scheduling
    shim -- all matmul/norm/attention compute is dispatched to the GPU via
    Vulkan -- so NUMA-local memory binding is irrelevant here and safe to
    skip entirely, exactly like the existing ``init_cpu_threads_env``
    try/except in ``VulkanWorker.init_device``.
    """
    try:
        import vllm._C  # noqa: PLC0415, F401
    except (ImportError, ModuleNotFoundError):
        # vllm._C missing entirely; CPUWorker.__init__ will still fail the
        # same way, but that is a separate, pre-existing limitation (running
        # vLLM without any compiled extension at all) outside this patch's
        # scope.
        return

    if hasattr(torch.ops._C, "init_cpu_memory_env"):
        return  # Real op is present and registered; nothing to do.

    lib = torch.library.Library("_C", "FRAGMENT")
    lib.define("init_cpu_memory_env(SymInt[] node_ids) -> ()")
    lib.impl(
        "init_cpu_memory_env",
        lambda node_ids: None,
        "CompositeExplicitAutograd",
    )
    # Prevent the Library (and therefore the op registration) from being
    # garbage-collected once this function returns.
    _KEEPALIVE_LIBS.append(lib)

    logger.info(
        "Registered no-op fallback for vllm._C.init_cpu_memory_env "
        "(op not present in this vLLM CPU build); NUMA-local memory "
        "binding is skipped. Compute is dispatched to the GPU via Vulkan "
        "regardless, so this has no effect on functionality."
    )


def _patch_cpu_triton_utils() -> None:
    """Monkey-patch ``vllm.utils.cpu_triton_utils`` if ``vllm._C`` is absent.

    Checking only ``import vllm._C`` succeeding is not sufficient: some vLLM
    CPU wheels (observed on aarch64 builds) ship a ``vllm._C`` module that
    imports fine but registers *zero* ops under the ``_C`` torch.library
    namespace (see ``_patch_cpu_memory_env`` above for the same situation
    with a different op). ``vllm.utils.cpu_triton_utils.compute_slot_mapping_kernel``
    unconditionally calls ``torch.ops._C.compute_slot_mapping_kernel_impl``
    with no fallback of its own, so in that "importable but empty" case this
    crashes on every forward pass with:
        AttributeError: '_OpNamespace' '_C' object has no attribute
        'compute_slot_mapping_kernel_impl'
    Check for the specific op instead of just the module's importability.
    """
    try:
        import vllm._C  # noqa: PLC0415, F401

        if hasattr(torch.ops._C, "compute_slot_mapping_kernel_impl"):
            # The real op IS available — no patch needed.
            return
    except (ImportError, ModuleNotFoundError):
        pass  # Extension absent; apply the pure-Python fallback.

    try:
        import vllm.utils.cpu_triton_utils as _cpu_utils  # noqa: PLC0415
    except ImportError:
        logger.debug("vllm.utils.cpu_triton_utils not found; skip patch.")
        return

    _cpu_utils.compute_slot_mapping_kernel = _FuncWrapper(
        _pure_python_compute_slot_mapping
    )
    logger.info(
        "Applied pure-Python fallback for compute_slot_mapping_kernel "
        "(vllm._C not available)."
    )

    # Also patch the reference inside block_table if already imported
    try:
        import vllm.v1.worker.block_table as _bt  # noqa: PLC0415

        if hasattr(_bt, "_compute_slot_mapping_kernel"):
            _bt._compute_slot_mapping_kernel = _cpu_utils.compute_slot_mapping_kernel
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Patch: register gemma4 chat template fallback
# ---------------------------------------------------------------------------


def _register_gemma4_chat_template() -> None:
    """Register a chat template fallback for gemma4 base models.

    Gemma4 base models (e.g. google/gemma-4-E2B) do not include a chat
    template in their tokenizer_config.json.  As of transformers v4.44,
    vLLM requires a chat template for the /v1/chat/completions endpoint.

    This registers the standard Gemma4 instruction-tuned chat template as a
    fallback so that base models can be used with the chat completions API.
    The template is identical to the one shipped with google/gemma-4-E2B-it.
    """
    try:
        from vllm.transformers_utils.chat_templates import (  # noqa: PLC0415
            get_chat_template_fallback_path,
        )
        from vllm.transformers_utils.chat_templates.registry import (  # noqa: PLC0415
            register_chat_template_fallback_path,
        )
    except ImportError:
        logger.debug("vllm chat template registry not available; skip gemma4 patch.")
        return

    # Only register if gemma4 does not already have a fallback.
    if get_chat_template_fallback_path("gemma4", "") is not None:
        return

    template_path = _TEMPLATES_DIR / "template_gemma4.jinja"
    if not template_path.exists():
        logger.warning(
            "gemma4 chat template file not found at %s; skip registration.",
            template_path,
        )
        return

    register_chat_template_fallback_path("gemma4", template_path)
    logger.info(
        "Registered gemma4 chat template fallback for base models "
        "(e.g. google/gemma-4-E2B) from %s.",
        template_path,
    )
