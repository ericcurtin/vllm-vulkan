#!/usr/bin/env python3
"""Greedy single-sequence generation with the native Vulkan LFM2 engine."""

import argparse
from collections.abc import Mapping

import numpy as np
from transformers import AutoTokenizer
from vllm_vulkan._rs import Lfm2Model


def main():
    """Tokenize a chat prompt and greedily decode one sequence on the selected GPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="local LFM2 safetensors checkpoint directory")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--quant", default="auto", help="native auto or explicit residency format"
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--max-buffer-bytes", type=int)
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    model = Lfm2Model(
        args.model,
        args.device,
        len(tokens) + args.max_new_tokens,
        quant=args.quant,
        group_size=args.group_size,
        max_buffer_bytes=args.max_buffer_bytes,
    )
    logits = model.forward_prefill(tokens)
    generated = []
    for i in range(args.max_new_tokens):
        token = int(np.argmax(logits))
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        if i + 1 < args.max_new_tokens:
            logits = model.forward(token)
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
