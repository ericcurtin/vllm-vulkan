"""Independent Transformers oracle over explicit FP32 MLX INT4 dequantization.

The full checkpoint is streamed a matrix chunk at a time, avoiding a 10 GiB
expanded model. This code is validation-only and is never used by the runtime.
"""

import json
from pathlib import Path

import torch
from lfm2_formats import MatrixReader, fixture_matrix
from safetensors.torch import save_file
from transformers import Lfm2Config, Lfm2ForCausalLM
from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding


def dequant(reader, key, start, end):
    """Decode a matrix row range through the validation-only checkpoint reader."""
    return reader.dequant(key, start, end)


class QuantLinear(torch.nn.Module):
    def __init__(self, reader, key):
        """Bind a logical weight matrix without materializing its dense expansion."""
        super().__init__()
        self.reader, self.key = reader, key
        self.rows = reader.get_slice(key + ".weight").get_shape()[0]

    def forward(self, x):
        """Apply linear projections in row chunks to bound CPU reference memory."""
        return torch.cat(
            [
                torch.nn.functional.linear(
                    x,
                    dequant(self.reader, self.key, start, min(start + 2048, self.rows)),
                )
                for start in range(0, self.rows, 2048)
            ],
            dim=-1,
        )


class QuantEmbedding(torch.nn.Module):
    def __init__(self, reader, key):
        """Bind a packed embedding table for decoding only requested token rows."""
        super().__init__()
        self.reader, self.key = reader, key

    def forward(self, ids):
        """Decode token embeddings while preserving the input token-array shape."""
        rows = [
            dequant(self.reader, self.key, int(i), int(i) + 1)[0] for i in ids.flatten()
        ]
        return torch.stack(rows).reshape(*ids.shape, -1)


def reference_model(directory):
    """Build an eager Transformers oracle using independently decoded checkpoint weights."""
    directory = Path(directory)
    cfg = Lfm2Config.from_dict(json.loads((directory / "config.json").read_text()))
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = Lfm2ForCausalLM(cfg)
    reader = MatrixReader(directory)
    prefix = (
        "language_model."
        if "language_model.model.embed_tokens.weight" in reader.keys()
        else ""
    )
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            key = prefix + ("model.embed_tokens" if name == "lm_head" else name)
            model.set_submodule(name, QuantLinear(reader, key))
        elif isinstance(module, torch.nn.Embedding):
            model.set_submodule(name, QuantEmbedding(reader, prefix + name))
    for name, _ in list(model.named_parameters()):
        value = reader.get_tensor(prefix + name).float()
        if ".conv.conv.weight" in name and tuple(value.shape[1:]) == (
            cfg.conv_L_cache,
            1,
        ):
            value = value.transpose(1, 2).contiguous()
        parent, _, attr = name.rpartition(".")
        setattr(
            model.get_submodule(parent),
            attr,
            torch.nn.Parameter(value, requires_grad=False),
        )
    model.model.rotary_emb = Lfm2RotaryEmbedding(cfg)
    return model.eval()


def tiny_checkpoint(
    directory,
    *,
    hidden=128,
    heads=4,
    kvheads=2,
    length=3,
    coeff_dtype=torch.bfloat16,
    quant="mlx4",
    group_size=64,
    intermediate=192,
    mixed=False,
    sharded=False,
    hf_names=False,
):
    """Quantize a seeded HF model, then validate the GPU against its dequantized weights."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(127)
    cfg = Lfm2Config(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=5,
        num_attention_heads=heads,
        num_key_value_heads=kvheads,
        layer_types=["conv", "full_attention", "conv", "full_attention", "conv"],
        conv_L_cache=length,
        conv_bias=False,
        vocab_size=128,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        block_auto_adjust_ff_dim=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
    )
    cfg._attn_implementation = "eager"
    model = Lfm2ForCausalLM(cfg).float().eval()
    matrix_names = {
        name + ".weight"
        for name, m in model.named_modules()
        if isinstance(m, (torch.nn.Linear, torch.nn.Embedding))
    }
    tensors = {}
    formats = {}
    matrix_index = 0
    prefix = "" if hf_names else "language_model."
    for name, value in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        if name in matrix_names:
            key = prefix + name.removesuffix(".weight")
            kind = (
                [
                    "mlx2",
                    "mlx4",
                    "mlx6",
                    "mlx8",
                    "f32",
                    "f16",
                    "bf16",
                    "q4_0",
                    "q4_k",
                    "q8_0",
                    "fp8_tensor",
                    "fp8",
                    "nvfp4",
                ][matrix_index % 13]
                if mixed
                else quant
            )
            values, info = fixture_matrix(value, kind, group_size, coeff_dtype)
            tensors.update(
                {key + "." + suffix: t.contiguous() for suffix, t in values.items()}
            )
            formats[key] = info
            matrix_index += 1
        else:
            if ".conv.conv.weight" in name:
                value = value.transpose(1, 2)
            tensors[prefix + name] = value.to(coeff_dtype).contiguous()
    config = cfg.to_dict()
    if quant == "mlx4" and group_size == 64 and not mixed:
        config["quantization"] = {"bits": 4, "group_size": 64, "mode": "affine"}
    else:
        config["quantization_config"] = {
            "quant_method": "vllm_vulkan",
            "tensor_formats": formats,
        }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if sharded:
        mapping = {}
        for i in range(3):
            values = {k: v for j, (k, v) in enumerate(tensors.items()) if j % 3 == i}
            filename = f"model-{i:05}.safetensors"
            save_file(values, directory / filename)
            mapping.update(dict.fromkeys(values, filename))
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
    else:
        save_file(tensors, directory / "model.safetensors")
    return directory
