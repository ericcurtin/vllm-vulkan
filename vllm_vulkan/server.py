# SPDX-License-Identifier: Apache-2.0
"""Standalone OpenAI-compatible API server backed by the Rust VulkanModel.

This server bypasses vLLM's internal model runner entirely and uses our
Rust VulkanModel directly for generation.  This gives ~3 tok/s on GB10
vs 1.7 tok/s from vLLM's CPU backend.

Usage:
    python -m vllm_vulkan.server google/gemma-4-E2B-it --port 8000

API:
    POST /v1/chat/completions  (OpenAI-compatible)
    GET  /v1/models
    GET  /health
"""

import argparse
import glob
import logging
import os
import time
import uuid

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def find_safetensors(model_name_or_path: str) -> str:
    """Find the safetensors file for a model."""
    # Try local directory first
    if os.path.isdir(model_name_or_path):
        files = glob.glob(f"{model_name_or_path}/*.safetensors")
        if files:
            return sorted(files)[0]

    # Try HuggingFace cache
    try:
        import huggingface_hub

        local_dir = huggingface_hub.snapshot_download(
            model_name_or_path,
            local_files_only=True,
            ignore_patterns=["*.bin", "*.gguf", "*.pt"],
        )
        files = glob.glob(f"{local_dir}/*.safetensors")
        if files:
            return sorted(files)[0]
    except Exception as e:
        logger.warning("Could not find model in HF cache: %s", e)

    raise FileNotFoundError(f"No safetensors file found for {model_name_or_path}")


def greedy_sample(logits: list[float]) -> int:
    """Return the token with the highest logit."""
    return max(range(len(logits)), key=lambda i: logits[i])


def temperature_sample(
    logits: list[float], temperature: float = 1.0, top_p: float = 1.0, top_k: int = 64
) -> int:
    """Sample from logits with temperature, top-p, and top-k filtering."""
    import math
    import random

    if temperature == 0.0:
        return greedy_sample(logits)

    # Apply temperature
    scaled = [v / temperature for v in logits]

    # Softmax
    max_l = max(scaled)
    exp_l = [math.exp(x - max_l) for x in scaled]
    total = sum(exp_l)
    probs = [x / total for x in exp_l]

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, len(probs))
        top_k_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[
            :top_k
        ]
        top_k_probs = [probs[i] for i in top_k_indices]
        total_k = sum(top_k_probs)
        top_k_probs = [p / total_k for p in top_k_probs]
    else:
        top_k_indices = list(range(len(probs)))
        top_k_probs = probs

    # Top-p (nucleus) filtering
    sorted_indices = sorted(
        range(len(top_k_probs)), key=lambda i: top_k_probs[i], reverse=True
    )
    cumsum = 0.0
    nucleus = []
    for idx in sorted_indices:
        cumsum += top_k_probs[idx]
        nucleus.append(idx)
        if cumsum >= top_p:
            break

    nucleus_probs = [top_k_probs[i] for i in nucleus]
    total_n = sum(nucleus_probs)
    nucleus_probs = [p / total_n for p in nucleus_probs]

    # Sample
    r = random.random()
    cumsum = 0.0
    for i, p in zip(nucleus, nucleus_probs, strict=False):
        cumsum += p
        if r <= cumsum:
            return top_k_indices[i]
    return top_k_indices[nucleus[-1]]


def generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 64,
) -> tuple[str, int, int]:
    """Generate a response using the Rust VulkanModel.

    Returns: (generated_text, num_prompt_tokens, num_completion_tokens)
    """
    # Format prompt using the tokenizer's chat template.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()

    # Reset and prefill KV cache.
    model.reset_kv_cache()

    # Prefill: run forward for each prompt token.
    for pos, token_id in enumerate(input_ids):
        logits = model.forward(token_id, pos)

    # Get next token from last prefill step.
    if temperature == 0.0 or (temperature < 0.01):
        next_token = greedy_sample(logits)
    else:
        next_token = temperature_sample(logits, temperature, top_p, top_k)

    # Decode: generate new tokens.
    generated_ids: list[int] = []
    pos = len(input_ids)
    eos_token_id = tokenizer.eos_token_id
    stop_tokens = {eos_token_id, 106}  # 106 = <end_of_turn> in Gemma

    while len(generated_ids) < max_new_tokens:
        generated_ids.append(next_token)
        if next_token in stop_tokens:
            break

        logits = model.forward(next_token, pos)
        pos += 1

        if temperature == 0.0 or (temperature < 0.01):
            next_token = greedy_sample(logits)
        else:
            next_token = temperature_sample(logits, temperature, top_p, top_k)

    # Remove trailing EOS/end-of-turn tokens.
    while generated_ids and generated_ids[-1] in stop_tokens:
        generated_ids.pop()

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, len(input_ids), len(generated_ids)


def make_app(model_name: str, model, tokenizer):
    """Create the FastAPI application."""
    import asyncio

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="vllm-vulkan API server")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm-vulkan",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        try:
            messages = request.get("messages", [])
            max_tokens = request.get("max_tokens", 200)
            temperature = request.get("temperature", 1.0)
            top_p = request.get("top_p", 0.95)
            top_k = request.get("top_k", 64)

            t0 = time.perf_counter()
            text, n_prompt, n_gen = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate(
                    model, tokenizer, messages, max_tokens, temperature, top_p, top_k
                ),
            )
            elapsed = time.perf_counter() - t0
            tok_per_sec = n_gen / elapsed if elapsed > 0 else 0

            logger.info(
                "Generated %d tokens in %.1fs = %.1f tok/s", n_gen, elapsed, tok_per_sec
            )

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": n_prompt,
                    "completion_tokens": n_gen,
                    "total_tokens": n_prompt + n_gen,
                },
            }
        except Exception as e:
            logger.exception("Error in chat completion")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "type": "InternalServerError"}},
            )

    return app


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="vllm-vulkan standalone server")
    parser.add_argument("model", help="Model name or path (e.g. google/gemma-4-E2B-it)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device-idx", type=int, default=0)
    args = parser.parse_args()

    # Load the model.
    logger.info("Loading tokenizer for %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logger.info("Finding safetensors file...")
    st_path = find_safetensors(args.model)
    logger.info("Loading VulkanModel from %s...", st_path)

    from vllm_vulkan._rs import VulkanModel

    t0 = time.perf_counter()
    model = VulkanModel(
        st_path, max_seq_len=args.max_seq_len, device_idx=args.device_idx
    )
    elapsed = time.perf_counter() - t0

    logger.info(
        "Model loaded in %.1fs: %d layers, GPU=%s",
        elapsed,
        model.num_layers(),
        model.has_gpu(),
    )

    # Quick test.
    logger.info("Running test forward pass...")
    t1 = time.perf_counter()
    model.forward(1, 0)
    model.reset_kv_cache()
    logger.info("Test forward: %.0fms", (time.perf_counter() - t1) * 1000)

    # Start server.
    import uvicorn

    app = make_app(args.model, model, tokenizer)
    logger.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
