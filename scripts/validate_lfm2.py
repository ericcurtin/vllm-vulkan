#!/usr/bin/env python3
"""Qualify full checkpoint logits and cache lifecycle against Transformers."""

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import transformers
from lfm2_reference import reference_model
from transformers import AutoTokenizer
from vllm_vulkan._rs import Lfm2Model, get_device_info

from vllm_vulkan import _rs


def main():
    """Compare GPU logits and cache lifecycle against an independent FP32 oracle."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--quant", default="auto")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--max-buffer-bytes", type=int)
    ap.add_argument(
        "--reference-model", help="exported residency bytes for explicit --quant"
    )
    args = ap.parse_args()
    if args.length < 8:
        ap.error("--length must be >=8")
    if args.quant != "auto" and not args.reference_model:
        ap.error(
            "explicit --quant requires --reference-model exported by convert_lfm2.py"
        )
    torch.set_num_threads(4)
    directory = Path(args.model)
    reference_directory = Path(args.reference_model or args.model)

    def create_gpu(capacity):
        """Create an engine with the requested residency and sequence capacity."""
        return Lfm2Model(
            str(directory),
            args.device,
            capacity,
            quant=args.quant,
            group_size=args.group_size,
            max_buffer_bytes=args.max_buffer_bytes,
        )

    tokenizer = AutoTokenizer.from_pretrained(directory)
    ids = tokenizer.encode(
        "The capital of France is Paris. A short convolution remembers recent inputs. "
        * 30
    )[: args.length]
    model = reference_model(reference_directory)
    started = time.perf_counter()
    with torch.inference_mode():
        oracle = model(torch.tensor([ids]), use_cache=False).logits[0].numpy().copy()
    print(f"Transformers FP32 oracle: {time.perf_counter() - started:.2f}s", flush=True)
    del model
    gpu = create_gpu(len(ids) + 8)
    formats = gpu.weight_formats()
    checks = []

    def check(label, actual, expected):
        """Require finite, matching full-vocabulary logits and record error metrics."""
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        cosine = float(
            np.dot(actual, expected)
            / (np.linalg.norm(actual) * np.linalg.norm(expected))
        )
        maximum = float(np.max(np.abs(actual - expected)))
        assert np.isfinite(actual).all(), label
        assert relative < 5e-5, (label, relative, maximum)
        assert cosine > 0.999999, (label, cosine)
        assert actual.argmax() == expected.argmax(), (
            label,
            actual.argmax(),
            expected.argmax(),
        )
        checks.append(
            {
                "label": label,
                "relative_l2": relative,
                "cosine": cosine,
                "max_abs": maximum,
                "greedy": int(actual.argmax()),
            }
        )

    started = time.perf_counter()
    for position, token in enumerate(ids):
        check(f"serial/{position}", gpu.forward(token, position), oracle[position])
    serial_seconds = time.perf_counter() - started
    gpu.reset_kv_cache()
    for end in [3, 7, len(ids)]:
        check(
            f"chunk/{end}",
            gpu.forward_prefill(ids[gpu.position : end]),
            oracle[end - 1],
        )
    gpu.reset_kv_cache()
    check("reset/first", gpu.forward(ids[0]), oracle[0])
    for payload, position in [
        ([128000], None),
        ([ids[0]], 0),
        ([], None),
        ([ids[1], 128000], None),
        ([ids[0]] * (len(ids) + 8), None),
    ]:
        try:
            if position is None:
                gpu.forward_prefill(payload)
            else:
                gpu.forward(payload[0], position)
        except RuntimeError:
            assert gpu.position == 1
        else:
            raise AssertionError("invalid request accepted")
    check("after_rejections", gpu.forward_prefill(ids[1:]), oracle[-1])
    # Capacity boundary and recovery after an entirely different cached sequence.
    gpu.reset_kv_cache()
    gpu.forward_prefill(list(reversed(ids)))
    gpu.reset_kv_cache()
    check("reset/other_sequence", gpu.forward_prefill(ids), oracle[-1])
    del gpu
    print(f"Raw prompt/cache checks passed: {len(checks)}", flush=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(prompt, Mapping):
        prompt = prompt["input_ids"]
    chat = create_gpu(len(prompt) + 48)
    logits = chat.forward_prefill(prompt)
    generated = []
    started = time.perf_counter()
    for _ in range(32):
        token = int(np.argmax(logits))
        generated.append(token)
        logits = chat.forward(token)
    decode_seconds = time.perf_counter() - started
    del chat
    # Every generated token also gets an independent teacher-forced oracle.
    model = reference_model(reference_directory)
    with torch.inference_mode():
        expected = (
            model(torch.tensor([prompt + generated]), use_cache=False).logits[0].numpy()
        )
    verifier = create_gpu(len(prompt) + len(generated))
    actual = verifier.forward_prefill(prompt)
    check("chat/prompt", actual, expected[len(prompt) - 1])
    for i, token in enumerate(generated):
        actual = verifier.forward(token)
        check(f"chat/decode/{i}", actual, expected[len(prompt) + i])

    def checkpoint_hashes(path):
        """Hash checkpoint configuration, index and shards for reproducible reports."""
        names = ["config.json"]
        index = path / "model.safetensors.index.json"
        if index.exists():
            names += [index.name]
            names += sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            names += sorted(p.name for p in path.glob("*.safetensors"))
        result = {}
        for name in names:
            with (path / name).open("rb") as stream:
                result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return result

    source_files = checkpoint_hashes(directory)
    reference_files = (
        source_files
        if directory == reference_directory
        else checkpoint_hashes(reference_directory)
    )
    with open(_rs.__file__, "rb") as stream:
        binary_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    root = Path(__file__).resolve().parents[1]
    source_names = [
        "Cargo.toml",
        "src/lfm2.rs",
        "src/lfm2_quant.rs",
        "src/compute.rs",
        "src/lib.rs",
        "scripts/compile_shaders.sh",
        "shaders/mul_mat_vec_mlx4.comp",
        "shaders/lfm2_weights.glsl",
    ]
    source_names += [
        f"shaders/lfm2_{name}.comp"
        for name in ["embed", "matvec", "norm", "elementwise", "rope", "attention"]
    ]
    source_names += [
        "scripts/lfm2_formats.py",
        "scripts/lfm2_reference.py",
        "scripts/validate_lfm2.py",
    ]
    source_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in source_names
    }
    result = {
        "model_directory": str(directory.resolve()),
        "reference_directory": str(reference_directory.resolve()),
        "checkpoint_files_sha256": source_files,
        "reference_files_sha256": reference_files,
        "residency": {
            "quant": args.quant,
            "group_size": args.group_size,
            "max_buffer_bytes": args.max_buffer_bytes,
            "weight_formats": formats,
        },
        "binary_sha256": binary_digest,
        "source_sha256": source_hashes,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": get_device_info(args.device),
        "checks": checks,
        "serial_tokens": len(ids),
        "serial_input_ids": ids,
        "chat_prompt_ids": prompt,
        "serial_seconds": serial_seconds,
        "decode_tokens": len(generated),
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": len(generated) / decode_seconds,
        "generated_ids": generated,
        "text": tokenizer.decode(generated),
        "reference": "Transformers Lfm2ForCausalLM, eager attention, explicit FP32 dequantization",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checks": len(checks),
                "max_relative_l2": max(c["relative_l2"] for c in checks),
                "decode_tokens_per_second": result["decode_tokens_per_second"],
                "text": result["text"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
