# SPDX-License-Identifier: Apache-2.0
"""VulkanWorker — vLLM worker for the Vulkan backend.

Inherits from CPUWorker but overrides ``init_device`` to work without the
optional ``vllm._C`` compiled extension.  The extension normally provides
``torch.ops._C.init_cpu_threads_env`` for CPU thread-affinity binding.
When absent we log a notice and skip the binding; functionality is unchanged,
only the optional NUMA-pinned thread-affinity is omitted.
"""

import logging
import os
import platform
import sys

import torch
from vllm.v1.worker.cpu_worker import CPUWorker

logger = logging.getLogger(__name__)


class VulkanWorker(CPUWorker):
    """CPUWorker subclass that gracefully handles a missing ``vllm._C``.

    The sole change from the parent class is in ``init_device``: the call to
    ``torch.ops._C.init_cpu_threads_env`` is wrapped in a ``try/except`` so
    that it degrades gracefully when the compiled extension is unavailable
    (e.g. in a pure-Python vLLM install or when running from source without
    the C extension being built).
    """

    def init_device(self) -> None:  # type: ignore[override]
        from vllm import envs
        from vllm.platforms import CpuArchEnum, current_platform
        from vllm.utils.torch_utils import set_random_seed
        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

        # Re-apply the top-k/top-p Triton-CUDA guard here. It is also
        # attempted from vllm_vulkan's plugin-registration entrypoint
        # (vllm_vulkan.patches.apply_patches, called extremely early via
        # vLLM's lazy `current_platform` resolution), but that is too early
        # for `import vllm.v1.sample.ops.topk_topp_sampler` to succeed
        # (vllm.config is still mid-import at that point), so it silently
        # no-ops there. By the time init_device() runs, vLLM's module graph
        # is fully loaded, so this is where the patch actually takes effect.
        try:
            from vllm_vulkan.patches import _patch_topk_topp_triton

            _patch_topk_topp_triton()
        except Exception:
            logger.warning(
                "Failed to apply top-k/top-p Triton-CUDA guard from init_device().",
                exc_info=True,
            )

        # ── library presence checks ───────────────────────────────────────
        def check_preloaded_libs(name: str) -> None:
            ld_preload_list = os.environ.get("LD_PRELOAD", "")
            if name not in ld_preload_list:
                logger.warning(
                    "%s is not found in LD_PRELOAD. "
                    "For best performance, please follow the section "
                    "`set LD_PRELOAD` in "
                    "https://docs.vllm.ai/en/latest/getting_started/installation/cpu/ "
                    "to setup required pre-loaded libraries.",
                    name,
                )

        if sys.platform.startswith("linux"):
            check_preloaded_libs("libtcmalloc")
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                check_preloaded_libs("libiomp")

        # ── OpenMP thread-affinity setup ──────────────────────────────────
        omp_cpuids = envs.VLLM_CPU_OMP_THREADS_BIND
        if omp_cpuids == "auto" and platform.system() == "Linux":
            cpu_arch = current_platform.get_cpu_architecture()
            if cpu_arch in (CpuArchEnum.POWERPC, CpuArchEnum.S390X):
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4]
                )
            elif cpu_arch == CpuArchEnum.X86:
                self.local_omp_cpuid = self._get_autobind_cpu_ids(
                    lambda cpus: cpus[-1:]
                )
            elif cpu_arch == CpuArchEnum.ARM:
                self.local_omp_cpuid = self._get_autobind_cpu_ids(lambda cpus: cpus)
            else:
                self.local_omp_cpuid = "nobind"
        elif omp_cpuids == "nobind":
            self.local_omp_cpuid = "nobind"
        else:
            local_dp_rank = self.parallel_config.data_parallel_rank_local
            omp_cpuids_list = omp_cpuids.split("|")
            if local_dp_rank is not None:
                world_size = self.parallel_config.world_size
                omp_cpuids_list = omp_cpuids_list[
                    local_dp_rank * world_size : (local_dp_rank + 1) * world_size
                ]
            self.local_omp_cpuid = omp_cpuids_list[self.rank]

        if self.local_omp_cpuid != "nobind":
            try:
                ret = torch.ops._C.init_cpu_threads_env(self.local_omp_cpuid)
                if ret:
                    logger.info(ret)
            except AttributeError:
                # vllm._C is a compiled extension that is only available when
                # vLLM has been built from source with C extensions.  When it
                # is absent we skip NUMA-pinned thread-affinity binding.  This
                # has no effect on correctness, only on potential performance.
                logger.info(
                    "vllm._C not available; skipping CPU thread-affinity "
                    "binding (cpu_ids=%s). "
                    "Build vLLM from source for optimal CPU performance.",
                    self.local_omp_cpuid,
                )

        # Prevent downstream code from changing the thread count after binding.
        def skip_set_num_threads(x: int) -> None:
            logger.warning(
                "CPU backend doesn't allow to use "
                "`torch.set_num_threads` after the thread binding, skip it."
            )

        torch.set_num_threads = skip_set_num_threads

        # Unique identifier for creating allreduce shared memory.
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]

        # Initialise the distributed environment.
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner.
        # _VulkanCPUModelRunner wraps CPUModelRunner and loads the Rust VulkanModel.
        from vllm_vulkan.model_runner import _VulkanCPUModelRunner  # noqa: PLC0415

        self.model_runner = _VulkanCPUModelRunner(self.vllm_config, torch.device("cpu"))
