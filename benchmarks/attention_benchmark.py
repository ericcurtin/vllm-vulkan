#!/usr/bin/env python3
"""
Attention Benchmark

Benchmark attention performance on Vulkan backend.
"""

import argparse
import time

import vllm_vulkan


def create_attention_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device_idx: int = 0,
) -> tuple[any, any, any]:
    """Create input tensors for attention benchmark."""
    if not vllm_vulkan._RUST_AVAILABLE:
        raise RuntimeError("Vulkan backend not available")

    shape = [batch_size, seq_len, num_heads, head_dim]

    query = vllm_vulkan.VulkanTensor(shape, "f32", device_idx)
    key = vllm_vulkan.VulkanTensor(shape, "f32", device_idx)
    value = vllm_vulkan.VulkanTensor(shape, "f32", device_idx)

    return query, key, value


def benchmark_flash_attention(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device_idx: int = 0,
) -> dict:
    """
    Benchmark flash attention.

    Returns:
        Dictionary with benchmark results
    """
    # Create inputs
    query, key, value = create_attention_inputs(
        batch_size, seq_len, num_heads, head_dim, device_idx
    )

    # Warmup
    for _ in range(warmup_iterations):
        _ = vllm_vulkan.flash_attention(query, key, value)
        vllm_vulkan.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = vllm_vulkan.flash_attention(query, key, value)
        vllm_vulkan.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate throughput
    total_tokens = batch_size * seq_len
    tokens_per_second = total_tokens / avg_time

    # Calculate FLOPS (approximate)
    # Flash attention: O(n^2 * d) per head
    flops_per_head = 2 * seq_len * seq_len * head_dim
    total_flops = batch_size * num_heads * flops_per_head
    tflops = total_flops / avg_time / 1e12

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "tokens_per_second": tokens_per_second,
        "tflops": tflops,
    }


def benchmark_paged_attention(
    batch_size: int,
    context_len: int,
    num_heads: int,
    head_dim: int,
    block_size: int = 16,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device_idx: int = 0,
) -> dict:
    """
    Benchmark paged attention (decode phase).

    Returns:
        Dictionary with benchmark results
    """
    if not vllm_vulkan._RUST_AVAILABLE:
        raise RuntimeError("Vulkan backend not available")

    # Create query (decode: 1 token per sequence)
    query = vllm_vulkan.VulkanTensor(
        [batch_size, num_heads, head_dim], "f32", device_idx
    )

    # Create KV cache
    num_blocks = (batch_size * context_len + block_size - 1) // block_size + 100
    cache_shape = [num_blocks, block_size, num_heads, head_dim]

    key_cache = vllm_vulkan.VulkanTensor(cache_shape, "f32", device_idx)
    value_cache = vllm_vulkan.VulkanTensor(cache_shape, "f32", device_idx)

    # Create block tables and context lens
    blocks_per_seq = (context_len + block_size - 1) // block_size
    block_tables = list(range(batch_size * blocks_per_seq))
    context_lens = [context_len] * batch_size

    # Warmup
    for _ in range(warmup_iterations):
        _ = vllm_vulkan.paged_attention(
            query, key_cache, value_cache, block_tables, context_lens
        )
        vllm_vulkan.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = vllm_vulkan.paged_attention(
            query, key_cache, value_cache, block_tables, context_lens
        )
        vllm_vulkan.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate throughput
    tokens_per_second = batch_size / avg_time

    return {
        "batch_size": batch_size,
        "context_len": context_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "block_size": block_size,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "tokens_per_second": tokens_per_second,
    }


def run_benchmarks(
    batch_sizes: list[int],
    seq_lens: list[int],
    context_lens: list[int],
    num_heads: int = 32,
    head_dim: int = 128,
    block_size: int = 16,
    device_idx: int = 0,
) -> None:
    """Run all attention benchmarks."""
    print("=" * 60)
    print("Vulkan Attention Benchmarks")
    print("=" * 60)

    # Get device info
    try:
        device_info = vllm_vulkan.get_device_info(device_idx)
        print(f"Device: {device_info.get('name', 'Unknown')}")
        print(f"Memory: {device_info.get('memory_mb', 0)} MB")
    except Exception:
        print("Device: Unknown (Vulkan not available)")
        return

    print()

    # Flash attention benchmarks
    print("-" * 60)
    print("Flash Attention Benchmarks (Prefill)")
    print("-" * 60)
    print(f"{'Batch':>6} {'SeqLen':>8} {'Time(ms)':>10} {'TPS':>12} {'TFLOPS':>8}")
    print("-" * 60)

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            try:
                result = benchmark_flash_attention(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    device_idx=device_idx,
                )
                print(
                    f"{result['batch_size']:>6} "
                    f"{result['seq_len']:>8} "
                    f"{result['avg_time_ms']:>10.3f} "
                    f"{result['tokens_per_second']:>12.0f} "
                    f"{result['tflops']:>8.2f}"
                )
            except Exception as e:
                print(f"{batch_size:>6} {seq_len:>8} Error: {e}")

    print()

    # Paged attention benchmarks
    print("-" * 60)
    print("Paged Attention Benchmarks (Decode)")
    print("-" * 60)
    print(f"{'Batch':>6} {'Context':>8} {'Time(ms)':>10} {'TPS':>12}")
    print("-" * 60)

    for batch_size in batch_sizes:
        for context_len in context_lens:
            try:
                result = benchmark_paged_attention(
                    batch_size=batch_size,
                    context_len=context_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    device_idx=device_idx,
                )
                print(
                    f"{result['batch_size']:>6} "
                    f"{result['context_len']:>8} "
                    f"{result['avg_time_ms']:>10.3f} "
                    f"{result['tokens_per_second']:>12.0f}"
                )
            except Exception as e:
                print(f"{batch_size:>6} {context_len:>8} Error: {e}")

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Attention benchmark")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 512, 2048],
        help="Sequence lengths to benchmark (for flash attention)",
    )
    parser.add_argument(
        "--context-lens",
        type=int,
        nargs="+",
        default=[512, 2048, 8192],
        help="Context lengths to benchmark (for paged attention)",
    )
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--block-size", type=int, default=16, help="KV cache block size"
    )
    parser.add_argument("--device", type=int, default=0, help="Device index to use")

    args = parser.parse_args()

    run_benchmarks(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        context_lens=args.context_lens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        device_idx=args.device,
    )


if __name__ == "__main__":
    main()
