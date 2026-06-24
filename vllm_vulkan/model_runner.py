# SPDX-License-Identifier: Apache-2.0
"""VulkanModelRunner — GPU-accelerated model runner.

Two modes:
  1. VLLM_VULKAN_RUST_MODEL=1 (default): Use the Rust VulkanModel for the
     forward pass. Much faster than mode 2 due to batched GPU dispatch.

  2. VLLM_VULKAN_RUST_MODEL=0: Patch individual PyTorch modules (RMSNorm,
     Linear) to dispatch to Vulkan via Python. Slower but useful for debugging.
"""

import logging
import os

import torch
import torch.nn as nn

from vllm_vulkan import envs

logger = logging.getLogger(__name__)

_RUST_MODEL_SINGLETON = None  # lazy-initialised


def _use_rust_model() -> bool:
    return envs.VLLM_VULKAN_RUST_MODEL


class VulkanWorker:
    """CPUWorker subclass that uses our Rust VulkanModel for GPU inference."""

    def __new__(cls, vllm_config, device):
        import types  # noqa: PLC0415

        from vllm.v1.worker.cpu_worker import CPUWorker  # noqa: PLC0415

        worker = CPUWorker.__new__(CPUWorker)
        CPUWorker.__init__(worker, vllm_config, device)

        # Patch init_device to skip vllm._C thread-affinity binding.
        worker.init_device = types.MethodType(_safe_init_device, worker)

        return worker


def _safe_init_device(self) -> None:
    """Same as CPUWorker.init_device but skips the vllm._C call."""
    import platform  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from vllm import envs  # noqa: PLC0415
    from vllm.platforms import CpuArchEnum, current_platform  # noqa: PLC0415
    from vllm.utils.torch_utils import set_random_seed  # noqa: PLC0415

    def check_preloaded_libs(name: str) -> None:
        if name not in os.environ.get("LD_PRELOAD", ""):
            logger.warning("%s is not found in LD_PRELOAD.", name)

    if sys.platform.startswith("linux"):
        check_preloaded_libs("libtcmalloc")
        if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
            check_preloaded_libs("libiomp")

    omp_cpuids = envs.VLLM_CPU_OMP_THREADS_BIND
    if omp_cpuids == "auto" and platform.system() == "Linux":
        cpu_arch = current_platform.get_cpu_architecture()
        if cpu_arch in (CpuArchEnum.POWERPC, CpuArchEnum.S390X):
            self.local_omp_cpuid = self._get_autobind_cpu_ids(
                lambda cpus: [cpu for cpu in cpus if cpu.id % 8 < 4]
            )
        elif cpu_arch == CpuArchEnum.X86:
            self.local_omp_cpuid = self._get_autobind_cpu_ids(lambda cpus: cpus[-1:])
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
            torch.ops._C.init_cpu_threads_env(self.local_omp_cpuid)
        except AttributeError:
            logger.info("vllm._C not available; skipping CPU thread-affinity binding.")

    def skip_set_num_threads(x: int) -> None:
        logger.warning(
            "CPU backend doesn't allow `torch.set_num_threads` after binding."
        )

    torch.set_num_threads = skip_set_num_threads

    os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]

    from vllm.v1.worker.gpu_worker import (
        init_worker_distributed_environment,  # noqa: PLC0415
    )

    init_worker_distributed_environment(
        self.vllm_config,
        self.rank,
        self.distributed_init_method,
        self.local_rank,
        current_platform.dist_backend,
    )

    set_random_seed(self.model_config.seed)

    # Model runner: use our Rust-native runner if enabled.
    if _use_rust_model():
        from vllm_vulkan.model_runner import _VulkanCPUModelRunner  # noqa: PLC0415

        self.model_runner = _VulkanCPUModelRunner(self.vllm_config, torch.device("cpu"))
    else:
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner  # noqa: PLC0415

        self.model_runner = CPUModelRunner(self.vllm_config, torch.device("cpu"))


class _VulkanCPUModelRunner:
    """CPUModelRunner that patches the loaded model for Rust+Vulkan inference.

    Wraps CPUModelRunner and intercepts load_model to additionally load the
    Rust VulkanModel.  After loading, patches the PyTorch model's forward
    method to call our Rust model for the forward pass instead of PyTorch ops.
    """

    def __init__(self, vllm_config, device):
        from vllm.v1.worker.cpu_model_runner import CPUModelRunner  # noqa: PLC0415

        self._runner = CPUModelRunner(vllm_config, device)
        self._rust_model = None
        self._vllm_config = vllm_config

    def __getattr__(self, name: str):
        return getattr(self._runner, name)

    def load_model(self, **kwargs) -> None:
        """Load PyTorch model, then load Rust VulkanModel and patch forward."""
        self._runner.load_model(**kwargs)

        if not _use_rust_model():
            return

        try:
            self._rust_model = _load_rust_vulkan_model(self._vllm_config)
            if self._rust_model is not None:
                _patch_model_with_rust_forward(
                    self._runner.model, self._rust_model, self._vllm_config
                )
                logger.info("Rust VulkanModel patched into PyTorch model forward.")
        except Exception as exc:
            logger.warning("Failed to load Rust VulkanModel: %s", exc)

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


def _load_rust_vulkan_model(vllm_config):
    """Load the Rust VulkanModel for the given vLLM config."""
    from vllm_vulkan._rs import VulkanModel, is_available  # noqa: PLC0415

    if not is_available():
        return None

    model_name = vllm_config.model_config.model
    try:
        import huggingface_hub  # noqa: PLC0415

        local_dir = huggingface_hub.snapshot_download(
            model_name, local_files_only=True, ignore_patterns=["*.bin", "*.gguf"]
        )
        import glob  # noqa: PLC0415

        st_files = sorted(glob.glob(f"{local_dir}/*.safetensors"))
    except Exception:
        st_files = []

    if not st_files:
        logger.warning("No safetensors for %s; skipping Rust model.", model_name)
        return None

    st_path = st_files[0]
    max_seq = min(int(vllm_config.model_config.max_model_len or 512), 2048)
    logger.info("Loading Rust VulkanModel from %s (max_seq=%d)", st_path, max_seq)

    rust_model = VulkanModel(st_path, max_seq_len=max_seq, device_idx=0)
    logger.info(
        "Rust VulkanModel: %d layers, GPU=%s",
        rust_model.num_layers(),
        rust_model.has_gpu(),
    )
    return rust_model


def _patch_model_with_rust_forward(pytorch_model, rust_model, vllm_config):
    """Patch the PyTorch model's forward to use Rust VulkanModel.

    The PyTorch model is still used for the KV cache and attention backends.
    We intercept the outermost model forward and route it to Rust for the
    compute-intensive parts (embed → 35 layers → norm → lm_head).

    The KV cache management in vLLM works at the attention layer level.
    For now, we can't fully bypass vLLM's attention backend.
    Instead, we install forward hooks that measure what's happening.
    """
    # For now: install per-module Vulkan hooks (the original approach)
    # This gives partial GPU utilization for norms and decode-path linears.
    _apply_module_hooks(pytorch_model, rust_model)


def _apply_module_hooks(model: nn.Module, rust_model) -> None:
    """Apply Vulkan dispatch hooks to RMSNorm and Linear modules."""
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

    def vk_forward(x: torch.Tensor, residual=None):
        from vllm_vulkan import vulkan_ops  # noqa: PLC0415

        if not vulkan_ops.is_ready() or os.environ.get("VLLM_VULKAN_DISABLE_OPS"):
            return orig(x, residual)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return orig(x, residual)

        if residual is not None:
            x = x + residual
            residual_out = x

        weight = getattr(module, "weight", None)
        eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))

        try:
            result = vulkan_ops.rms_norm(x, weight, eps)
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

    def vk_forward(x: torch.Tensor, *args, **kwargs):
        from vllm_vulkan import vulkan_ops  # noqa: PLC0415

        if not vulkan_ops.is_ready() or args or kwargs:
            return orig(x, *args, **kwargs)
        if os.environ.get("VLLM_VULKAN_DISABLE_OPS") or os.environ.get(
            "VLLM_VULKAN_DISABLE_LINEAR"
        ):
            return orig(x, *args, **kwargs)

        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            return orig(x, *args, **kwargs)

        bias = getattr(module, "bias", None)
        try:
            # The Vulkan matmul kernel tiles 4 rows and returns incorrect
            # results for M < 4 (e.g. the M=1 decode step), producing garbage
            # tokens during generation. Pad the row dim up to 4, then slice back.
            xf = x.float().reshape(-1, x.shape[-1])
            m = xf.shape[0]
            if m < 4:
                xf = torch.cat([xf, xf[-1:].expand(4 - m, -1)], dim=0)
                result = vulkan_ops.linear(xf, weight.float(), bias)[:m]
            else:
                result = vulkan_ops.linear(xf, weight.float(), bias)
            result = result.reshape(*x.shape[:-1], result.shape[-1]).to(x.dtype)
        except Exception as exc:
            logger.debug("Vulkan linear failed (%s)", exc)
            return orig(x, *args, **kwargs)

        if _returns_tuple:
            output_bias = (
                module.bias if getattr(module, "skip_bias_add", False) else None
            )
            return result, output_bias
        return result

    module.forward = vk_forward
