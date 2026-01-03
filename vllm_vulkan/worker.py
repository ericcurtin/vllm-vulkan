"""
VulkanWorker - Per-device worker for vLLM Vulkan backend.

This module implements the worker that manages model execution on a single Vulkan device.
"""

from typing import Any, Dict, List, Optional, Tuple

import vllm_vulkan
from vllm_vulkan.platform import VulkanPlatform


class VulkanWorker:
    """
    Worker class for managing model execution on a Vulkan device.

    Each worker is responsible for a single GPU and handles:
    - Model loading and weight transfer
    - Memory profiling
    - KV cache allocation
    - Forward pass execution
    """

    def __init__(
        self,
        model_config: Any,
        parallel_config: Any,
        scheduler_config: Any,
        device_config: Any,
        cache_config: Any,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: Optional[str] = None,
    ):
        """
        Initialize the Vulkan worker.

        Args:
            model_config: Model configuration
            parallel_config: Parallelism configuration
            scheduler_config: Scheduler configuration
            device_config: Device configuration
            cache_config: KV cache configuration
            local_rank: Local rank for multi-GPU
            rank: Global rank for distributed
            distributed_init_method: Method for distributed initialization
        """
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Initialize device
        self.device_idx = local_rank
        self._init_device()

        # Model and cache will be initialized later
        self.model = None
        self.model_runner = None
        self.kv_cache = None

    def _init_device(self) -> None:
        """Initialize the Vulkan device."""
        device_count = vllm_vulkan.get_device_count()
        if self.device_idx >= device_count:
            raise RuntimeError(
                f"Device index {self.device_idx} out of range "
                f"(available: 0-{device_count - 1})"
            )

        # Create device and backend
        if vllm_vulkan._RUST_AVAILABLE:
            self.device = vllm_vulkan.VulkanDevice(self.device_idx)
            self.backend = vllm_vulkan.VulkanBackend(self.device_idx)
        else:
            self.device = None
            self.backend = None

    def init_model(self) -> None:
        """Initialize the model on this device."""
        from vllm_vulkan.model_runner import VulkanModelRunner

        self.model_runner = VulkanModelRunner(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            device_idx=self.device_idx,
        )

    def load_model(self) -> None:
        """Load model weights to the device."""
        if self.model_runner is not None:
            self.model_runner.load_model()

    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float = 0.9,
    ) -> Tuple[int, int]:
        """
        Profile memory to determine available KV cache blocks.

        Args:
            block_size: Size of each KV cache block
            gpu_memory_utilization: Target GPU memory utilization

        Returns:
            Tuple of (num_gpu_blocks, num_cpu_blocks)
        """
        # Get device memory info
        used, total = vllm_vulkan.get_memory_info(self.device_idx)

        # Calculate available memory
        available = int(total * gpu_memory_utilization) - used

        # Calculate block memory requirements
        # Each block stores K and V for all layers
        num_heads = getattr(self.model_config, "num_attention_heads", 32)
        head_dim = getattr(self.model_config, "head_dim", 128)
        num_layers = getattr(self.model_config, "num_hidden_layers", 32)
        dtype_bytes = 2  # Assume float16

        block_memory = (
            2 *  # K and V
            num_layers *
            block_size *
            num_heads *
            head_dim *
            dtype_bytes
        )

        num_gpu_blocks = available // block_memory if block_memory > 0 else 0
        num_cpu_blocks = 0  # CPU cache not implemented yet

        return (num_gpu_blocks, num_cpu_blocks)

    def init_cache_engine(self, cache_config: Any) -> None:
        """
        Initialize the KV cache engine.

        Args:
            cache_config: Cache configuration
        """
        if not vllm_vulkan._RUST_AVAILABLE:
            return

        num_blocks = cache_config.num_gpu_blocks
        block_size = cache_config.block_size
        num_heads = getattr(self.model_config, "num_attention_heads", 32)
        head_dim = getattr(self.model_config, "head_dim", 128)
        num_layers = getattr(self.model_config, "num_hidden_layers", 32)

        self.kv_cache = vllm_vulkan.PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            device_idx=self.device_idx,
            dtype="f16",
        )

    def execute_model(
        self,
        seq_group_metadata_list: List[Any],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> Optional[List[Any]]:
        """
        Execute a batch of sequences.

        Args:
            seq_group_metadata_list: Metadata for each sequence group
            blocks_to_swap_in: Blocks to swap from CPU to GPU
            blocks_to_swap_out: Blocks to swap from GPU to CPU
            blocks_to_copy: Blocks to copy (for forking)

        Returns:
            List of outputs for each sequence
        """
        if self.model_runner is None:
            raise RuntimeError("Model not initialized. Call init_model() first.")

        # Handle block swaps and copies
        self._execute_cache_ops(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # Execute forward pass
        return self.model_runner.execute_model(seq_group_metadata_list)

    def _execute_cache_ops(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        """Execute cache operations (swap and copy)."""
        if self.kv_cache is None:
            return

        # Swap blocks from CPU to GPU
        for src, dst in blocks_to_swap_in.items():
            self.kv_cache.swap_blocks(src, dst)

        # Swap blocks from GPU to CPU
        for src, dst in blocks_to_swap_out.items():
            self.kv_cache.swap_blocks(src, dst)

        # Copy blocks
        for src, dsts in blocks_to_copy.items():
            for dst in dsts:
                self.kv_cache.copy_blocks(src, dst)

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a KV cache block in bytes."""
        if self.kv_cache is not None:
            num_blocks = self.kv_cache.num_total_blocks()
            if num_blocks > 0:
                return self.kv_cache.total_capacity() // num_blocks
        return 0

    def synchronize(self) -> None:
        """Synchronize all operations on this device."""
        vllm_vulkan.synchronize()

    def __repr__(self) -> str:
        device_name = "Unknown"
        if self.device is not None:
            device_name = self.device.name
        return f"VulkanWorker(rank={self.rank}, device={device_name})"
