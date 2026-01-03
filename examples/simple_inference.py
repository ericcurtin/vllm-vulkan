#!/usr/bin/env python3
"""
Simple Inference Example

Demonstrates basic usage of vLLM-Vulkan for text generation.
"""

import vllm_vulkan


def main():
    print("vLLM-Vulkan Simple Inference Example")
    print("=" * 50)

    # Check Vulkan availability
    print(f"\nVulkan available: {vllm_vulkan.is_available()}")
    print(f"Device count: {vllm_vulkan.get_device_count()}")

    # List available devices
    print("\nAvailable Vulkan devices:")
    devices = vllm_vulkan.enumerate_devices()
    for i, device in enumerate(devices):
        print(f"  [{i}] {device.get('name', 'Unknown')}")
        print(f"      Vendor: {device.get('vendor', 'Unknown')}")
        print(f"      Memory: {device.get('memory_mb', 0)} MB")
        print(f"      API Version: {device.get('api_version', 'Unknown')}")

    if not devices:
        print("  No Vulkan devices found!")
        return

    # Example: Using vLLM with Vulkan backend
    # Note: This requires the full vLLM installation
    print("\n" + "=" * 50)
    print("Example vLLM usage (requires vLLM installation):")
    print("=" * 50)
    print("""
from vllm import LLM, SamplingParams

# vLLM automatically discovers the Vulkan plugin
# when CUDA is not available

# Initialize the model
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    device="vulkan",  # Explicitly use Vulkan
    dtype="float16",
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

# Generate text
prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
]

outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
""")

    # Demonstrate direct API usage
    print("=" * 50)
    print("Direct API Example:")
    print("=" * 50)

    if vllm_vulkan._RUST_AVAILABLE:
        # Create a Vulkan tensor
        print("\nCreating VulkanTensor...")
        tensor = vllm_vulkan.VulkanTensor(
            shape=[2, 4, 8],
            dtype="f32",
            device_idx=0,
        )
        print(f"  Shape: {tensor.shape()}")
        print(f"  Dtype: {tensor.dtype()}")
        print(f"  Device: {tensor.device_idx()}")
        print(f"  Elements: {tensor.numel()}")

        # Create backend
        print("\nCreating VulkanBackend...")
        backend = vllm_vulkan.VulkanBackend(0)
        print(f"  Initialized: {backend.is_initialized}")
        print(f"  Max batch size: {backend.max_batch_size()}")
        print(f"  Max sequence length: {backend.max_sequence_length()}")

        # Create KV cache
        print("\nCreating PagedKVCache...")
        cache = vllm_vulkan.PagedKVCache(
            num_blocks=100,
            block_size=16,
            num_heads=32,
            head_dim=128,
            num_layers=32,
            device_idx=0,
            dtype="f16",
        )
        print(f"  Total blocks: {cache.num_total_blocks()}")
        print(f"  Free blocks: {cache.num_free_blocks()}")
        print(f"  Block size: {cache.block_size()}")

        # Allocate blocks for a sequence
        blocks = cache.allocate_blocks(seq_id=1, num_blocks=5)
        print(f"  Allocated blocks: {blocks}")
        print(f"  Free blocks after allocation: {cache.num_free_blocks()}")

    else:
        print("\nNote: Rust extension not available.")
        print("Build with `maturin develop` to enable full functionality.")


if __name__ == "__main__":
    main()
