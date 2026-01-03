#!/usr/bin/env python3
"""
Throughput Benchmark

Benchmark token generation throughput on Vulkan backend.
"""

import argparse
import time
from typing import List, Optional

import vllm_vulkan


def simulate_prefill(
    batch_size: int,
    prompt_len: int,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    device_idx: int = 0,
) -> float:
    """
    Simulate prefill phase and return time taken.

    Returns:
        Time in seconds
    """
    if not vllm_vulkan._RUST_AVAILABLE:
        return 0.0

    # Create tensors
    hidden_size = num_heads * head_dim
    shape = [batch_size, prompt_len, num_heads, head_dim]

    query = vllm_vulkan.VulkanTensor(shape, "f32", device_idx)
    key = vllm_vulkan.VulkanTensor(shape, "f32", device_idx)
    value = vllm_vulkan.VulkanTensor(shape, "f32", device_idx)

    start = time.perf_counter()

    # Simulate num_layers of attention
    for _ in range(num_layers):
        _ = vllm_vulkan.flash_attention(query, key, value)

    vllm_vulkan.synchronize()
    end = time.perf_counter()

    return end - start


def simulate_decode_step(
    batch_size: int,
    context_len: int,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    kv_cache,
    device_idx: int = 0,
) -> float:
    """
    Simulate one decode step and return time taken.

    Returns:
        Time in seconds
    """
    if not vllm_vulkan._RUST_AVAILABLE:
        return 0.0

    # Decode: 1 token per sequence
    query = vllm_vulkan.VulkanTensor(
        [batch_size, num_heads, head_dim], "f32", device_idx
    )

    block_size = 16
    blocks_per_seq = (context_len + block_size - 1) // block_size
    block_tables = list(range(batch_size * blocks_per_seq))
    context_lens = [context_len] * batch_size

    start = time.perf_counter()

    # Simulate num_layers of paged attention
    for _ in range(num_layers):
        _ = vllm_vulkan.paged_attention(
            query,
            kv_cache[0],  # key cache
            kv_cache[1],  # value cache
            block_tables,
            context_lens,
        )

    vllm_vulkan.synchronize()
    end = time.perf_counter()

    return end - start


def benchmark_throughput(
    batch_size: int,
    prompt_len: int,
    generation_len: int,
    num_heads: int = 32,
    head_dim: int = 128,
    num_layers: int = 32,
    block_size: int = 16,
    device_idx: int = 0,
) -> dict:
    """
    Benchmark end-to-end token generation throughput.

    Returns:
        Dictionary with benchmark results
    """
    if not vllm_vulkan._RUST_AVAILABLE:
        return {"error": "Vulkan backend not available"}

    # Allocate KV cache
    max_context = prompt_len + generation_len
    num_blocks = (batch_size * max_context + block_size - 1) // block_size + 100
    cache_shape = [num_blocks, block_size, num_heads, head_dim]

    key_cache = vllm_vulkan.VulkanTensor(cache_shape, "f32", device_idx)
    value_cache = vllm_vulkan.VulkanTensor(cache_shape, "f32", device_idx)
    kv_cache = (key_cache, value_cache)

    # Warmup
    _ = simulate_prefill(
        batch_size, prompt_len, num_heads, head_dim, num_layers, device_idx
    )

    # Prefill phase
    prefill_time = simulate_prefill(
        batch_size, prompt_len, num_heads, head_dim, num_layers, device_idx
    )

    # Decode phase
    decode_times = []
    for step in range(generation_len):
        current_context = prompt_len + step
        decode_time = simulate_decode_step(
            batch_size,
            current_context,
            num_heads,
            head_dim,
            num_layers,
            kv_cache,
            device_idx,
        )
        decode_times.append(decode_time)

    total_decode_time = sum(decode_times)
    total_time = prefill_time + total_decode_time

    # Calculate metrics
    total_prompt_tokens = batch_size * prompt_len
    total_generated_tokens = batch_size * generation_len

    prefill_tokens_per_second = total_prompt_tokens / prefill_time if prefill_time > 0 else 0
    decode_tokens_per_second = total_generated_tokens / total_decode_time if total_decode_time > 0 else 0
    overall_tokens_per_second = (total_prompt_tokens + total_generated_tokens) / total_time if total_time > 0 else 0

    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "generation_len": generation_len,
        "num_layers": num_layers,
        "prefill_time_ms": prefill_time * 1000,
        "decode_time_ms": total_decode_time * 1000,
        "total_time_ms": total_time * 1000,
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
        "overall_tokens_per_second": overall_tokens_per_second,
        "avg_decode_step_ms": (total_decode_time / generation_len) * 1000 if generation_len > 0 else 0,
    }


def run_benchmarks(
    batch_sizes: List[int],
    prompt_lens: List[int],
    generation_len: int,
    num_heads: int = 32,
    head_dim: int = 128,
    num_layers: int = 32,
    device_idx: int = 0,
) -> None:
    """Run throughput benchmarks."""
    print("=" * 70)
    print("Vulkan Throughput Benchmarks")
    print("=" * 70)

    # Get device info
    try:
        device_info = vllm_vulkan.get_device_info(device_idx)
        print(f"Device: {device_info.get('name', 'Unknown')}")
        print(f"Memory: {device_info.get('memory_mb', 0)} MB")
    except Exception:
        print("Device: Unknown (Vulkan not available)")
        return

    print(f"Model config: heads={num_heads}, head_dim={head_dim}, layers={num_layers}")
    print(f"Generation length: {generation_len}")
    print()

    print("-" * 70)
    print(f"{'Batch':>6} {'Prompt':>8} {'Prefill':>12} {'Decode':>12} {'Overall':>12}")
    print(f"{'':>6} {'':>8} {'(tok/s)':>12} {'(tok/s)':>12} {'(tok/s)':>12}")
    print("-" * 70)

    for batch_size in batch_sizes:
        for prompt_len in prompt_lens:
            try:
                result = benchmark_throughput(
                    batch_size=batch_size,
                    prompt_len=prompt_len,
                    generation_len=generation_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_layers=num_layers,
                    device_idx=device_idx,
                )

                if "error" in result:
                    print(f"{batch_size:>6} {prompt_len:>8} Error: {result['error']}")
                else:
                    print(
                        f"{result['batch_size']:>6} "
                        f"{result['prompt_len']:>8} "
                        f"{result['prefill_tokens_per_second']:>12.0f} "
                        f"{result['decode_tokens_per_second']:>12.0f} "
                        f"{result['overall_tokens_per_second']:>12.0f}"
                    )
            except Exception as e:
                print(f"{batch_size:>6} {prompt_len:>8} Error: {e}")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--prompt-lens",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
        help="Prompt lengths to benchmark",
    )
    parser.add_argument(
        "--generation-len",
        type=int,
        default=128,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of attention heads"
    )
    parser.add_argument(
        "--head-dim", type=int, default=128, help="Head dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=32, help="Number of transformer layers"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Device index to use"
    )

    args = parser.parse_args()

    run_benchmarks(
        batch_sizes=args.batch_sizes,
        prompt_lens=args.prompt_lens,
        generation_len=args.generation_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        device_idx=args.device,
    )


if __name__ == "__main__":
    main()
