"""
Distributed Tests

Tests for the Vulkan distributed communication module.
"""



class TestCommunicator:
    """Tests for VulkanDistributedCommunicator."""

    def test_import(self):
        """Test that communicator can be imported."""
        from vllm_vulkan.distributed import VulkanDistributedCommunicator

        assert VulkanDistributedCommunicator is not None

    def test_creation(self):
        """Test communicator creation."""
        from vllm_vulkan.distributed import VulkanDistributedCommunicator

        comm = VulkanDistributedCommunicator(
            world_size=2,
            rank=0,
        )

        assert comm.world_size == 2
        assert comm.rank == 0
        assert comm.local_rank == 0

    def test_custom_local_rank(self):
        """Test custom local rank."""
        from vllm_vulkan.distributed import VulkanDistributedCommunicator

        comm = VulkanDistributedCommunicator(
            world_size=4,
            rank=2,
            local_rank=0,  # Different node
            device_idx=0,
        )

        assert comm.world_size == 4
        assert comm.rank == 2
        assert comm.local_rank == 0
        assert comm.device_idx == 0


class TestDistributedModule:
    """Tests for distributed module functions."""

    def test_init_distributed(self):
        """Test init_distributed function."""
        from vllm_vulkan.distributed import init_distributed, is_initialized

        comm = init_distributed(world_size=1, rank=0)

        assert comm is not None
        assert is_initialized()

    def test_get_world_size(self):
        """Test get_world_size function."""
        from vllm_vulkan.distributed import (
            get_world_size,
            init_distributed,
        )

        init_distributed(world_size=4, rank=0)
        assert get_world_size() == 4

    def test_get_rank(self):
        """Test get_rank function."""
        from vllm_vulkan.distributed import get_rank, init_distributed

        init_distributed(world_size=4, rank=2)
        assert get_rank() == 2

    def test_barrier(self):
        """Test barrier function."""
        from vllm_vulkan.distributed import barrier, init_distributed

        init_distributed(world_size=1, rank=0)
        # Should not raise
        barrier()


class TestCollectiveOps:
    """Tests for collective operations."""

    def test_all_reduce_import(self):
        """Test all_reduce function import."""
        from vllm_vulkan.distributed.communicator import all_reduce

        assert all_reduce is not None

    def test_broadcast_import(self):
        """Test broadcast function import."""
        from vllm_vulkan.distributed.communicator import broadcast

        assert broadcast is not None


class TestDeviceUtils:
    """Tests for device utilities."""

    def test_get_device_properties(self):
        """Test get_device_properties function."""
        from vllm_vulkan.utils import get_device_properties

        props = get_device_properties(0)
        assert isinstance(props, dict)
        assert "name" in props

    def test_get_device_memory_info(self):
        """Test get_device_memory_info function."""
        from vllm_vulkan.utils import get_device_memory_info

        used, total = get_device_memory_info(0)
        assert isinstance(used, int)
        assert isinstance(total, int)
        assert used >= 0
        assert total >= 0

    def test_is_device_available(self):
        """Test is_device_available function."""
        from vllm_vulkan.utils import is_device_available

        result = is_device_available(0)
        assert isinstance(result, bool)

    def test_synchronize_device(self):
        """Test synchronize_device function."""
        from vllm_vulkan.utils import synchronize_device

        # Should not raise
        synchronize_device()
