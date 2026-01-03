"""
Attention Backend Tests

Tests for the Vulkan attention backend.
"""



class TestAttentionMetadata:
    """Tests for VulkanAttentionMetadata."""

    def test_import(self):
        """Test that attention metadata can be imported."""
        from vllm_vulkan.attention import VulkanAttentionMetadata

        assert VulkanAttentionMetadata is not None

    def test_default_values(self):
        """Test default metadata values."""
        from vllm_vulkan.attention import VulkanAttentionMetadata

        meta = VulkanAttentionMetadata()

        assert meta.num_prefill_tokens == 0
        assert meta.num_decode_tokens == 0
        assert meta.num_prefills == 0
        assert meta.seq_lens == []
        assert meta.context_lens == []
        assert meta.max_seq_len == 0

    def test_is_prefill(self):
        """Test is_prefill property."""
        from vllm_vulkan.attention import VulkanAttentionMetadata

        meta = VulkanAttentionMetadata()
        assert not meta.is_prefill

        meta.num_prefill_tokens = 100
        assert meta.is_prefill

    def test_is_decode(self):
        """Test is_decode property."""
        from vllm_vulkan.attention import VulkanAttentionMetadata

        meta = VulkanAttentionMetadata()
        assert not meta.is_decode

        meta.num_decode_tokens = 10
        assert meta.is_decode

    def test_total_tokens(self):
        """Test total_tokens property."""
        from vllm_vulkan.attention import VulkanAttentionMetadata

        meta = VulkanAttentionMetadata()
        meta.num_prefill_tokens = 100
        meta.num_decode_tokens = 10

        assert meta.total_tokens == 110


class TestAttentionMetadataBuilder:
    """Tests for VulkanAttentionMetadataBuilder."""

    def test_import(self):
        """Test that builder can be imported."""
        from vllm_vulkan.attention import VulkanAttentionMetadataBuilder

        assert VulkanAttentionMetadataBuilder is not None

    def test_builder_creation(self):
        """Test builder creation."""
        from vllm_vulkan.attention import VulkanAttentionMetadataBuilder

        builder = VulkanAttentionMetadataBuilder(
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            block_size=16,
        )

        assert builder.num_heads == 32
        assert builder.head_dim == 128
        assert builder.num_kv_heads == 8
        assert builder.block_size == 16

    def test_add_sequence(self):
        """Test adding sequences to builder."""
        from vllm_vulkan.attention import VulkanAttentionMetadataBuilder

        builder = VulkanAttentionMetadataBuilder(
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            block_size=16,
        )

        builder.add_sequence(
            seq_len=100,
            context_len=100,
            is_prefill=True,
            block_table=[0, 1, 2, 3, 4, 5, 6],
        )

        meta = builder.build()

        assert meta.num_prefill_tokens == 100
        assert meta.num_prefills == 1
        assert len(meta.seq_lens) == 1
        assert meta.seq_lens[0] == 100

    def test_build_multiple_sequences(self):
        """Test building with multiple sequences."""
        from vllm_vulkan.attention import VulkanAttentionMetadataBuilder

        builder = VulkanAttentionMetadataBuilder(
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            block_size=16,
        )

        # Add prefill sequence
        builder.add_sequence(
            seq_len=100,
            context_len=100,
            is_prefill=True,
        )

        # Add decode sequence
        builder.add_sequence(
            seq_len=50,
            context_len=50,
            is_prefill=False,
        )

        meta = builder.build()

        assert meta.num_prefill_tokens == 100
        assert meta.num_decode_tokens == 1  # Decode adds 1 token
        assert len(meta.seq_lens) == 2

    def test_builder_reset(self):
        """Test builder reset."""
        from vllm_vulkan.attention import VulkanAttentionMetadataBuilder

        builder = VulkanAttentionMetadataBuilder(
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            block_size=16,
        )

        builder.add_sequence(
            seq_len=100,
            context_len=100,
            is_prefill=True,
        )

        builder.reset()

        meta = builder.build()
        assert meta.num_prefill_tokens == 0
        assert len(meta.seq_lens) == 0


class TestAttentionBackend:
    """Tests for VulkanAttentionBackend."""

    def test_import(self):
        """Test that backend can be imported."""
        from vllm_vulkan.attention import VulkanAttentionBackend

        assert VulkanAttentionBackend is not None

    def test_backend_name(self):
        """Test backend name."""
        from vllm_vulkan.attention import VulkanAttentionBackend

        assert VulkanAttentionBackend.name == "vulkan"

    def test_get_impl_cls(self):
        """Test get_impl_cls method."""
        from vllm_vulkan.attention import VulkanAttentionBackend

        impl_cls = VulkanAttentionBackend.get_impl_cls()
        assert impl_cls is not None

    def test_get_metadata_cls(self):
        """Test get_metadata_cls method."""
        from vllm_vulkan.attention import VulkanAttentionBackend

        meta_cls = VulkanAttentionBackend.get_metadata_cls()
        assert meta_cls is not None

    def test_get_builder_cls(self):
        """Test get_builder_cls method."""
        from vllm_vulkan.attention import VulkanAttentionBackend

        builder_cls = VulkanAttentionBackend.get_builder_cls()
        assert builder_cls is not None

    def test_supported_head_sizes(self):
        """Test supported head sizes."""
        from vllm_vulkan.attention import VulkanAttentionBackend

        sizes = VulkanAttentionBackend.get_supported_head_sizes()
        assert isinstance(sizes, list)
        assert 64 in sizes
        assert 128 in sizes


class TestAttentionImpl:
    """Tests for VulkanAttentionImpl."""

    def test_import(self):
        """Test that impl can be imported."""
        from vllm_vulkan.attention.backend import VulkanAttentionImpl

        assert VulkanAttentionImpl is not None

    def test_impl_creation(self):
        """Test impl creation."""
        from vllm_vulkan.attention.backend import VulkanAttentionImpl

        impl = VulkanAttentionImpl(
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            scale=0.088,
        )

        assert impl.num_heads == 32
        assert impl.head_dim == 128
        assert impl.num_kv_heads == 8
        assert impl.scale == 0.088
        assert impl.num_queries_per_kv == 4  # 32 / 8


class TestBackendSelector:
    """Tests for attention backend selector."""

    def test_import(self):
        """Test that selector can be imported."""
        from vllm_vulkan.attention.selector import get_attn_backend

        assert get_attn_backend is not None

    def test_which_attn_to_use(self):
        """Test which_attn_to_use function."""
        from vllm_vulkan.attention.selector import which_attn_to_use

        result = which_attn_to_use(
            num_heads=32,
            head_size=128,
            num_kv_heads=8,
        )
        assert isinstance(result, str)
