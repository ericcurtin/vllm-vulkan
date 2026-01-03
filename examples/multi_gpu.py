#!/usr/bin/env python3
"""
Multi-GPU Inference Example

Demonstrates tensor parallel and pipeline parallel inference
using multiple Vulkan GPUs.
"""

import vllm_vulkan
from vllm_vulkan.distributed import (
    VulkanDistributedCommunicator,
    init_distributed,
)


def main():
    print("vLLM-Vulkan Multi-GPU Inference Example")
    print("=" * 50)

    # Check device count
    device_count = vllm_vulkan.get_device_count()
    print(f"\nAvailable Vulkan devices: {device_count}")

    if device_count < 2:
        print("\nNote: Multi-GPU examples require at least 2 Vulkan devices.")
        print("Running with single device demonstration instead.\n")

    # List all devices
    print("\nDevice Information:")
    print("-" * 50)
    devices = vllm_vulkan.enumerate_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device.get('name', 'Unknown')}")
        print(f"  Memory: {device.get('memory_mb', 0)} MB")
        print(f"  Type: {device.get('device_type', 'unknown')}")
        print()

    # Example: Initialize distributed environment
    print("=" * 50)
    print("Distributed Setup Example")
    print("=" * 50)

    world_size = min(device_count, 2) if device_count >= 2 else 1

    print(f"\nInitializing distributed with world_size={world_size}")

    comm = init_distributed(
        world_size=world_size,
        rank=0,  # This process is rank 0
        local_rank=0,
    )

    print(f"  World size: {comm.world_size}")
    print(f"  Rank: {comm.rank}")
    print(f"  Local rank: {comm.local_rank}")

    # Example: Tensor parallelism configuration
    print("\n" + "=" * 50)
    print("Example vLLM Tensor Parallel Usage")
    print("=" * 50)
    print("""
from vllm import LLM, SamplingParams

# Initialize with tensor parallelism across 2 GPUs
llm = LLM(
    model="meta-llama/Llama-2-70b-chat-hf",
    device="vulkan",
    tensor_parallel_size=2,  # Split model across 2 GPUs
    dtype="float16",
)

# Generate text
prompts = ["Explain the theory of relativity:"]
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
)

outputs = llm.generate(prompts, sampling_params)
print(outputs[0].outputs[0].text)
""")

    # Example: Pipeline parallelism configuration
    print("=" * 50)
    print("Example vLLM Pipeline Parallel Usage")
    print("=" * 50)
    print("""
from vllm import LLM, SamplingParams

# Initialize with pipeline parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-chat-hf",
    device="vulkan",
    tensor_parallel_size=1,
    pipeline_parallel_size=2,  # Split layers across 2 GPUs
    dtype="float16",
)

# Generate text
prompts = ["Write a short poem about AI:"]
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=128,
)

outputs = llm.generate(prompts, sampling_params)
print(outputs[0].outputs[0].text)
""")

    # Demonstrate distributed operations
    if vllm_vulkan._RUST_AVAILABLE and device_count >= 1:
        print("=" * 50)
        print("Distributed Operations Demo")
        print("=" * 50)

        # Create communicator
        communicator = VulkanDistributedCommunicator(
            world_size=1,
            rank=0,
            local_rank=0,
            device_idx=0,
        )

        print("\nCommunicator created:")
        print(f"  World size: {communicator.world_size}")
        print(f"  Rank: {communicator.rank}")
        print(f"  Device: {communicator.device_idx}")

        # Create a test tensor
        tensor = vllm_vulkan.VulkanTensor(
            shape=[4, 4],
            dtype="f32",
            device_idx=0,
        )

        print(f"\nTest tensor: {tensor.shape()}")

        # Demo all-reduce (no-op with single rank)
        result = communicator.all_reduce(tensor, op="sum")
        print(f"All-reduce result shape: {result.shape()}")

        # Demo broadcast (no-op with single rank)
        result = communicator.broadcast(tensor, src=0)
        print(f"Broadcast result shape: {result.shape()}")

        # Demo barrier
        communicator.barrier()
        print("Barrier synchronized")

    print("\n" + "=" * 50)
    print("Multi-GPU Best Practices")
    print("=" * 50)
    print("""
1. Tensor Parallelism:
   - Best for large models that don't fit on single GPU
   - Splits each layer across GPUs
   - Lower latency but higher communication overhead
   - Use for: Large batch inference

2. Pipeline Parallelism:
   - Splits model layers across GPUs
   - Higher throughput for batched requests
   - Use for: High-throughput serving

3. Hybrid Parallelism:
   - Combine tensor and pipeline parallelism
   - tensor_parallel_size * pipeline_parallel_size = total GPUs
   - Use for: Very large models on many GPUs

4. Memory Optimization:
   - Use quantization (Q4_0, Q4_K, Q8_0)
   - Reduces memory footprint significantly
   - Allows larger models on same hardware
""")


if __name__ == "__main__":
    main()
