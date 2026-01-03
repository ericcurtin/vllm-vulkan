"""
Platform Tests

Tests for the VulkanPlatform class.
"""


class TestVulkanPlatform:
    """Tests for VulkanPlatform."""

    def test_import(self):
        """Test that the platform can be imported."""
        from vllm_vulkan.platform import VulkanPlatform

        assert VulkanPlatform is not None

    def test_device_name(self):
        """Test device name attribute (uses cpu for PyTorch compatibility)."""
        from vllm_vulkan.platform import VulkanPlatform

        # Like MetalPlatform, we use CPU for PyTorch compatibility
        assert VulkanPlatform.device_name == "cpu"
        assert VulkanPlatform.device_type == "cpu"
        assert VulkanPlatform.dispatch_key == "CPU"

    def test_is_available(self):
        """Test is_available method."""
        from vllm_vulkan.platform import VulkanPlatform

        # Should return a boolean
        result = VulkanPlatform.is_available()
        assert isinstance(result, bool)

    def test_get_device_count(self):
        """Test get_device_count method."""
        from vllm_vulkan.platform import VulkanPlatform

        count = VulkanPlatform.get_device_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_verify_quantization(self):
        """Test verify_quantization method."""
        from vllm_vulkan.platform import VulkanPlatform

        # These should not raise
        VulkanPlatform.verify_quantization("none")
        VulkanPlatform.verify_quantization("q4_0")
        VulkanPlatform.verify_quantization("fp16")

        # This should raise
        try:
            VulkanPlatform.verify_quantization("unsupported_quant")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not supported" in str(e)

    def test_get_torch_device(self):
        """Test get_torch_device method."""
        import torch

        from vllm_vulkan.platform import VulkanPlatform

        device = VulkanPlatform.get_torch_device()
        assert device == torch.device("cpu")

    def test_is_pin_memory_available(self):
        """Test is_pin_memory_available method."""
        from vllm_vulkan.platform import VulkanPlatform

        result = VulkanPlatform.is_pin_memory_available()
        assert result is False


class TestPluginRegistration:
    """Tests for plugin registration."""

    def test_register_function(self):
        """Test that register() returns correct path."""
        import vllm_vulkan

        result = vllm_vulkan.register()
        # Returns dot notation path or None if not available
        assert result is None or result == "vllm_vulkan.platform.VulkanPlatform"

    def test_entry_point_format(self):
        """Test that the entry point format is valid."""
        import vllm_vulkan

        result = vllm_vulkan.register()
        if result is not None:
            # Should be in format "module.path.ClassName" (dot notation)
            assert "." in result
            parts = result.rsplit(".", 1)
            assert len(parts) == 2
            assert parts[1] == "VulkanPlatform"


class TestModuleExports:
    """Tests for module exports."""

    def test_version(self):
        """Test version is exported."""
        import vllm_vulkan

        assert hasattr(vllm_vulkan, "__version__")
        assert isinstance(vllm_vulkan.__version__, str)

    def test_is_available_exported(self):
        """Test is_available is exported."""
        import vllm_vulkan

        assert hasattr(vllm_vulkan, "is_available")
        assert callable(vllm_vulkan.is_available)

    def test_get_device_count_exported(self):
        """Test get_device_count is exported."""
        import vllm_vulkan

        assert hasattr(vllm_vulkan, "get_device_count")
        assert callable(vllm_vulkan.get_device_count)

    def test_enumerate_devices_exported(self):
        """Test enumerate_devices is exported."""
        import vllm_vulkan

        assert hasattr(vllm_vulkan, "enumerate_devices")
        assert callable(vllm_vulkan.enumerate_devices)
