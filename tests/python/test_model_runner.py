"""
Model Runner Tests

Tests for the VulkanModelRunner class.
"""

import pytest


class MockModelConfig:
    """Mock model configuration for testing."""

    def __init__(self):
        self.num_attention_heads = 32
        self.head_dim = 128
        self.num_hidden_layers = 32
        self.hidden_size = 4096
        self.vocab_size = 32000


class MockParallelConfig:
    """Mock parallel configuration."""

    def __init__(self):
        self.world_size = 1
        self.tensor_parallel_size = 1
        self.pipeline_parallel_size = 1


class MockSchedulerConfig:
    """Mock scheduler configuration."""

    def __init__(self):
        self.max_num_batched_tokens = 2048
        self.max_num_seqs = 256


class MockDeviceConfig:
    """Mock device configuration."""

    def __init__(self):
        self.device = "vulkan"
        self.device_id = 0


class MockCacheConfig:
    """Mock cache configuration."""

    def __init__(self):
        self.block_size = 16
        self.num_gpu_blocks = 1000
        self.num_cpu_blocks = 0
        self.gpu_memory_utilization = 0.9


class TestModelRunner:
    """Tests for VulkanModelRunner."""

    def test_import(self):
        """Test that model runner can be imported."""
        from vllm_vulkan.model_runner import VulkanModelRunner

        assert VulkanModelRunner is not None

    def test_creation(self):
        """Test model runner creation."""
        from vllm_vulkan.model_runner import VulkanModelRunner

        runner = VulkanModelRunner(
            model_config=MockModelConfig(),
            parallel_config=MockParallelConfig(),
            scheduler_config=MockSchedulerConfig(),
            device_config=MockDeviceConfig(),
            cache_config=MockCacheConfig(),
            device_idx=0,
        )

        assert runner is not None
        assert runner.device_idx == 0
        assert not runner.model_loaded

    def test_load_model(self):
        """Test model loading."""
        from vllm_vulkan.model_runner import VulkanModelRunner

        runner = VulkanModelRunner(
            model_config=MockModelConfig(),
            parallel_config=MockParallelConfig(),
            scheduler_config=MockSchedulerConfig(),
            device_config=MockDeviceConfig(),
            cache_config=MockCacheConfig(),
            device_idx=0,
        )

        runner.load_model()
        assert runner.model_loaded

    def test_repr(self):
        """Test repr output."""
        from vllm_vulkan.model_runner import VulkanModelRunner

        runner = VulkanModelRunner(
            model_config=MockModelConfig(),
            parallel_config=MockParallelConfig(),
            scheduler_config=MockSchedulerConfig(),
            device_config=MockDeviceConfig(),
            cache_config=MockCacheConfig(),
            device_idx=0,
        )

        repr_str = repr(runner)
        assert "VulkanModelRunner" in repr_str
        assert "device=0" in repr_str


class TestWorker:
    """Tests for VulkanWorker."""

    def test_import(self):
        """Test that worker can be imported."""
        from vllm_vulkan.worker import VulkanWorker

        assert VulkanWorker is not None


class TestExecutor:
    """Tests for VulkanExecutor."""

    def test_import(self):
        """Test that executor can be imported."""
        from vllm_vulkan.executor import VulkanExecutor

        assert VulkanExecutor is not None

    def test_repr(self):
        """Test repr output."""
        import vllm_vulkan
        from vllm_vulkan.executor import VulkanExecutor

        # Skip test if no Vulkan devices available
        if vllm_vulkan.get_device_count() == 0:
            pytest.skip("No Vulkan devices available")

        executor = VulkanExecutor(
            model_config=MockModelConfig(),
            cache_config=MockCacheConfig(),
            parallel_config=MockParallelConfig(),
            scheduler_config=MockSchedulerConfig(),
            device_config=MockDeviceConfig(),
        )

        repr_str = repr(executor)
        assert "VulkanExecutor" in repr_str


class TestInputPreparation:
    """Tests for input preparation."""

    def test_prepare_inputs_empty(self):
        """Test preparing empty inputs."""
        from vllm_vulkan.model_runner import VulkanModelRunner

        runner = VulkanModelRunner(
            model_config=MockModelConfig(),
            parallel_config=MockParallelConfig(),
            scheduler_config=MockSchedulerConfig(),
            device_config=MockDeviceConfig(),
            cache_config=MockCacheConfig(),
            device_idx=0,
        )

        inputs = runner.prepare_inputs([])

        assert "input_tokens" in inputs
        assert "input_positions" in inputs
        assert len(inputs["input_tokens"]) == 0
