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
        """Test device name attribute."""
        from vllm_vulkan.platform import VulkanPlatform

        assert VulkanPlatform.device_name == "vulkan"
        assert VulkanPlatform.device_type == "vulkan"

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

    def test_get_default_attn_backend(self):
        """Test get_default_attn_backend method."""
        from vllm_vulkan.platform import VulkanPlatform

        backend = VulkanPlatform.get_default_attn_backend()
        assert isinstance(backend, str)
        assert "vulkan" in backend.lower()

    def test_get_model_runner(self):
        """Test get_model_runner method."""
        from vllm_vulkan.platform import VulkanPlatform

        runner = VulkanPlatform.get_model_runner(None)
        assert isinstance(runner, str)
        assert "VulkanModelRunner" in runner

    def test_get_worker(self):
        """Test get_worker method."""
        from vllm_vulkan.platform import VulkanPlatform

        worker = VulkanPlatform.get_worker()
        assert isinstance(worker, str)
        assert "VulkanWorker" in worker

    def test_get_executor(self):
        """Test get_executor method."""
        from vllm_vulkan.platform import VulkanPlatform

        executor = VulkanPlatform.get_executor()
        assert isinstance(executor, str)
        assert "VulkanExecutor" in executor

    def test_get_supported_quantizations(self):
        """Test get_supported_quantizations method."""
        from vllm_vulkan.platform import VulkanPlatform

        quants = VulkanPlatform.get_supported_quantizations()
        assert isinstance(quants, list)
        assert "none" in quants
        assert "q4_0" in quants


class TestPluginRegistration:
    """Tests for plugin registration."""

    def test_register_function(self):
        """Test that register() returns correct path."""
        import vllm_vulkan

        result = vllm_vulkan.register()
        assert result == "vllm_vulkan.platform:VulkanPlatform"

    def test_entry_point_format(self):
        """Test that the entry point format is valid."""
        import vllm_vulkan

        result = vllm_vulkan.register()
        # Should be in format "module.path:ClassName"
        assert ":" in result
        module_path, class_name = result.rsplit(":", 1)
        assert "." in module_path
        assert class_name == "VulkanPlatform"


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
