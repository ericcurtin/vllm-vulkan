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


def apply_patches() -> None:
    """Apply all patches.  Safe to call multiple times (idempotent)."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    try:
        _patch_cpu_triton_utils()
    except Exception as exc:
        logger.warning("Failed to apply vllm._C patches: %s", exc)

    try:
        _register_gemma4_chat_template()
    except Exception as exc:
        logger.warning("Failed to register gemma4 chat template fallback: %s", exc)


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


def _patch_cpu_triton_utils() -> None:
    """Monkey-patch ``vllm.utils.cpu_triton_utils`` if ``vllm._C`` is absent."""
    try:
        import vllm._C  # noqa: PLC0415, F401

        # If we reach here, the C extension IS available — no patch needed.
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
