# SPDX-License-Identifier: Apache-2.0
"""VulkanWorker — vLLM worker for the Vulkan backend."""

import logging

from vllm.v1.worker.cpu_worker import CPUWorker

logger = logging.getLogger(__name__)


class VulkanWorker(CPUWorker):
    """CPUWorker that runs the model through Vulkan dispatch.

    ``init_device`` re-applies two patches whose plugin-registration-time
    attempt can silently no-op, then wraps whichever model runner
    ``CPUWorker`` built in ``_VulkanCPUModelRunner``.
    """

    def init_device(self) -> None:  # type: ignore[override]
        # Re-apply the top-k/top-p Triton-CUDA guard here. It is also
        # attempted from vllm_vulkan's plugin-registration entrypoint
        # (vllm_vulkan.patches.apply_patches, called extremely early via
        # vLLM's lazy `current_platform` resolution), but that is too early
        # for `import vllm.v1.sample.ops.topk_topp_sampler` to succeed
        # (vllm.config is still mid-import at that point), so it silently
        # no-ops there. By the time init_device() runs, vLLM's module graph
        # is fully loaded, so this is where the patch actually takes effect.
        try:
            from vllm_vulkan.patches import _patch_topk_topp_triton  # noqa: PLC0415

            _patch_topk_topp_triton()
        except Exception:
            logger.warning(
                "Failed to apply top-k/top-p Triton-CUDA guard from init_device().",
                exc_info=True,
            )

        # Same rationale as the topk/topp re-apply just above: this patch
        # must be in place before `load_model()` -> `process_weights_after_
        # loading()` runs (which happens later in this same worker's
        # startup sequence, well after init_device()), so re-attempting it
        # here -- in case the plugin-registration-time attempt no-op'd due
        # to circular-import timing -- is cheap, safe insurance.
        try:
            from vllm_vulkan.patches import _patch_cpu_weight_removal  # noqa: PLC0415

            _patch_cpu_weight_removal()
        except Exception:
            logger.warning(
                "Failed to apply CPU weight-removal guard from init_device().",
                exc_info=True,
            )

        super().init_device()

        # Wrap, rather than replace, what CPUWorker just built: it picks
        # between the V1 and V2 CPU model runners (`use_v2_model_runner`),
        # and rebuilding one here would both duplicate that choice and pay
        # for a second runner's buffers only to discard the first.
        from vllm_vulkan.model_runner import _VulkanCPUModelRunner  # noqa: PLC0415

        self.model_runner = _VulkanCPUModelRunner(self.model_runner)
