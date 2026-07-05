#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark for the Python-facing ``vulkan_ops`` dispatch path.

This exercises the exact same ``VulkanContext`` / ``ComputeEngine`` (Rust)
used by ``model_runner.py``'s per-module hook path (``VLLM_VULKAN_RUST_MODEL=0``,
and as a fallback within the default mode too) — i.e. the code that fires on
every ``nn.Linear`` / ``RMSNorm`` call during vLLM decode. It isolates the
Rust dispatch engine from vLLM's scheduler/KV-cache so the effect of engine
level changes (e.g. command-buffer reuse) can be measured directly against
representative google/gemma-4-E2B tensor shapes, without needing a full vLLM
server.
"""

from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1536, help="gemma-4-E2B hidden_size")
    ap.add_argument(
        "--out-features", type=int, default=2048, help="e.g. q_proj out dim"
    )
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device-idx", type=int, default=0)
    args = ap.parse_args()

    from vllm_vulkan._rs import VulkanContext, is_available

    from vllm_vulkan import vulkan_ops

    if not is_available():
        raise SystemExit("Vulkan is not available on this system.")

    ctx = VulkanContext(args.device_idx)
    vulkan_ops.set_context(ctx)
    if not vulkan_ops.is_ready():
        raise SystemExit("Vulkan context not ready (software renderer detected?).")

    torch.manual_seed(0)
    x = torch.randn(1, args.hidden, dtype=torch.float32)
    w = torch.randn(args.out_features, args.hidden, dtype=torch.float32)

    for _ in range(args.warmup):
        vulkan_ops.linear(x, w)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        vulkan_ops.linear(x, w)
    dt = time.perf_counter() - t0

    per_call_us = dt / args.iters * 1e6
    print(
        f"vulkan_ops.linear: [1,{args.hidden}] x [{args.out_features},{args.hidden}]^T"
    )
    print(
        f"{args.iters} calls in {dt:.4f}s  ->  {per_call_us:.1f}us/call  "
        f"({args.iters / dt:.1f} calls/s)"
    )


if __name__ == "__main__":
    main()
