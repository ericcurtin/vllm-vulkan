#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tokens/sec benchmark for the Rust VulkanModel decode path.

Loads the safetensors checkpoint for a local HF model (e.g.
``google/gemma-4-E2B``) directly into ``vllm_vulkan._rs.VulkanModel`` and
drives greedy autoregressive decoding, bypassing vLLM's scheduler entirely.
This isolates the GPU compute engine (KosmicKrisp / Vulkan) so its raw
tokens/sec can be measured and compared across optimizations without noise
from unrelated vLLM engine-orchestration overhead.

Usage:
    python scripts/bench_vulkan_model.py \
        --model google/gemma-4-E2B \
        --prompt "The capital of France is" \
        --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import glob
import time

import huggingface_hub


def find_safetensors(model: str) -> str:
    local_dir = huggingface_hub.snapshot_download(
        model, local_files_only=True, ignore_patterns=["*.bin", "*.gguf"]
    )
    files = sorted(glob.glob(f"{local_dir}/*.safetensors"))
    if not files:
        raise SystemExit(
            f"No safetensors file found for {model} (looked in {local_dir})"
        )
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--device-idx", type=int, default=0)
    ap.add_argument(
        "--repeat", type=int, default=1, help="repeat the decode loop N times"
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm_vulkan._rs import VulkanModel, is_available

    if not is_available():
        raise SystemExit("Vulkan is not available on this system.")

    st_path = find_safetensors(args.model)
    print(f"Loading weights from {st_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tokenizer.encode(args.prompt)

    t0 = time.perf_counter()
    model = VulkanModel(
        st_path, max_seq_len=args.max_seq_len, device_idx=args.device_idx
    )
    load_s = time.perf_counter() - t0
    print(
        f"Model load: {load_s:.2f}s, GPU={model.has_gpu()}, layers={model.num_layers()}"
    )

    all_prefill = []
    all_decode = []
    text_out = ""

    for _rep in range(args.repeat):
        model.reset_kv_cache()

        # ── Prefill: feed prompt tokens one at a time (API is per-token). ──
        t0 = time.perf_counter()
        logits = None
        for pos, tok in enumerate(prompt_ids):
            logits = model.forward(tok, pos)
        prefill_s = time.perf_counter() - t0
        all_prefill.append(prefill_s)

        # ── Decode: greedy sampling for max_new_tokens steps. ──
        generated = []
        pos = len(prompt_ids)
        next_tok = max(range(len(logits)), key=lambda i: logits[i])
        generated.append(next_tok)

        t0 = time.perf_counter()
        for _step in range(args.max_new_tokens - 1):
            logits = model.forward(next_tok, pos)
            pos += 1
            next_tok = max(range(len(logits)), key=lambda i: logits[i])
            generated.append(next_tok)
        decode_s = time.perf_counter() - t0
        all_decode.append(decode_s)

        text_out = tokenizer.decode(generated)

    n_decode_tokens = args.max_new_tokens - 1
    best_decode = min(all_decode)
    tok_per_s = n_decode_tokens / best_decode if best_decode > 0 else float("inf")
    all_tok_s = sorted(n_decode_tokens / d for d in all_decode)
    median_tok_s = all_tok_s[len(all_tok_s) // 2]

    print()
    print(f"Prompt:     {args.prompt!r}")
    print(f"Generated:  {text_out!r}")
    print()
    print(
        f"Prefill:    {len(prompt_ids)} tokens in {min(all_prefill):.4f}s "
        f"({len(prompt_ids) / min(all_prefill):.2f} tok/s)"
    )
    print(
        f"Decode:     {n_decode_tokens} tokens in {best_decode:.4f}s "
        f"({tok_per_s:.2f} tok/s)  [best of {args.repeat}]"
    )
    print(
        f"Decode all-runs tok/s: {[round(v, 2) for v in all_tok_s]}  median={median_tok_s:.2f}"
    )


if __name__ == "__main__":
    main()
