"""
Distributed Communicator

This module provides distributed communication primitives for multi-GPU
inference using Vulkan.
"""

from typing import Any

import vllm_vulkan

# Global distributed state
_DISTRIBUTED_INITIALIZED = False
_WORLD_SIZE = 1
_RANK = 0
_LOCAL_RANK = 0
_COMMUNICATOR = None


class VulkanDistributedCommunicator:
    """
    Distributed communicator for Vulkan multi-GPU operations.

    This class provides collective operations like all-reduce, broadcast,
    and scatter/gather for tensor parallelism.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int | None = None,
        device_idx: int | None = None,
    ):
        """
        Initialize the communicator.

        Args:
            world_size: Total number of processes
            rank: Global rank of this process
            local_rank: Local rank on this node
            device_idx: Vulkan device index to use
        """
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank if local_rank is not None else rank
        self.device_idx = device_idx if device_idx is not None else self.local_rank

        # Initialize Rust communicator if available
        if vllm_vulkan._RUST_AVAILABLE:
            self._comm = vllm_vulkan.VulkanCommunicator(
                world_size=world_size,
                rank=rank,
                local_rank=self.local_rank,
                device_idx=self.device_idx,
            )
        else:
            self._comm = None

    def all_reduce(
        self,
        tensor: Any,
        op: str = "sum",
    ) -> Any:
        """
        All-reduce operation across all ranks.

        Args:
            tensor: Input tensor
            op: Reduction operation (sum, prod, min, max, avg)

        Returns:
            Reduced tensor (same on all ranks)
        """
        if self._comm is None:
            return tensor

        return self._comm.all_reduce(tensor, op)

    def all_gather(
        self,
        tensor: Any,
    ) -> Any:
        """
        All-gather operation across all ranks.

        Args:
            tensor: Input tensor

        Returns:
            Gathered tensor from all ranks
        """
        if self._comm is None:
            return tensor

        return self._comm.all_gather(tensor)

    def reduce_scatter(
        self,
        tensor: Any,
        op: str = "sum",
    ) -> Any:
        """
        Reduce-scatter operation.

        Args:
            tensor: Input tensor
            op: Reduction operation

        Returns:
            Scattered tensor (each rank gets a portion)
        """
        if self._comm is None:
            return tensor

        return self._comm.reduce_scatter(tensor, op)

    def broadcast(
        self,
        tensor: Any,
        src: int = 0,
    ) -> Any:
        """
        Broadcast tensor from source rank to all ranks.

        Args:
            tensor: Input tensor
            src: Source rank

        Returns:
            Broadcasted tensor
        """
        if self._comm is None:
            return tensor

        return self._comm.broadcast(tensor, src)

    def send(
        self,
        tensor: Any,
        dst: int,
    ) -> None:
        """
        Send tensor to destination rank.

        Args:
            tensor: Tensor to send
            dst: Destination rank
        """
        if self._comm is not None:
            self._comm.send(tensor, dst)

    def recv(
        self,
        shape: list[int],
        src: int,
        dtype: str = "f32",
    ) -> Any:
        """
        Receive tensor from source rank.

        Args:
            shape: Expected tensor shape
            src: Source rank
            dtype: Data type

        Returns:
            Received tensor
        """
        if self._comm is None:
            raise RuntimeError("Communicator not initialized")

        return self._comm.recv(src, shape, dtype)

    def barrier(self) -> None:
        """
        Synchronization barrier across all ranks.
        """
        if self._comm is not None:
            self._comm.barrier()


def init_distributed(
    world_size: int = 1,
    rank: int = 0,
    local_rank: int | None = None,
    backend: str = "vulkan",
) -> VulkanDistributedCommunicator:
    """
    Initialize distributed environment.

    Args:
        world_size: Total number of processes
        rank: Global rank
        local_rank: Local rank on this node
        backend: Backend to use (only "vulkan" supported)

    Returns:
        Initialized communicator
    """
    global _DISTRIBUTED_INITIALIZED, _WORLD_SIZE, _RANK, _LOCAL_RANK, _COMMUNICATOR

    if backend != "vulkan":
        raise ValueError(f"Unsupported backend: {backend}")

    _WORLD_SIZE = world_size
    _RANK = rank
    _LOCAL_RANK = local_rank if local_rank is not None else rank
    _COMMUNICATOR = VulkanDistributedCommunicator(
        world_size=world_size,
        rank=rank,
        local_rank=_LOCAL_RANK,
    )
    _DISTRIBUTED_INITIALIZED = True

    return _COMMUNICATOR


def is_initialized() -> bool:
    """Check if distributed environment is initialized."""
    return _DISTRIBUTED_INITIALIZED


def get_world_size() -> int:
    """Get world size."""
    return _WORLD_SIZE


def get_rank() -> int:
    """Get global rank."""
    return _RANK


def get_local_rank() -> int:
    """Get local rank."""
    return _LOCAL_RANK


def get_communicator() -> VulkanDistributedCommunicator | None:
    """Get the global communicator."""
    return _COMMUNICATOR


def barrier() -> None:
    """Execute a barrier across all ranks."""
    if _COMMUNICATOR is not None:
        _COMMUNICATOR.barrier()


def all_reduce(
    tensor: Any,
    op: str = "sum",
) -> Any:
    """
    All-reduce using the global communicator.

    Args:
        tensor: Input tensor
        op: Reduction operation

    Returns:
        Reduced tensor
    """
    if _COMMUNICATOR is None:
        return tensor
    return _COMMUNICATOR.all_reduce(tensor, op)


def broadcast(
    tensor: Any,
    src: int = 0,
) -> Any:
    """
    Broadcast using the global communicator.

    Args:
        tensor: Input tensor
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if _COMMUNICATOR is None:
        return tensor
    return _COMMUNICATOR.broadcast(tensor, src)
