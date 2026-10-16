"""Validation-only safetensors readers and independent packed-weight decoders."""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


class MatrixReader:
    def __init__(self, directory):
        """Index checkpoint tensors and defer opening shards until they are needed."""
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        index = self.directory / "model.safetensors.index.json"
        self.handles = {}
        if index.exists():
            self.locations = json.loads(index.read_text())["weight_map"]
        else:
            self.locations = {}
            files = [self.directory / "model.safetensors"]
            if not files[0].exists():
                files = sorted(self.directory.glob("*.safetensors"))
            for file in files:
                for key in self.open(file.name).keys():
                    if key in self.locations:
                        raise ValueError("duplicate tensor")
                    self.locations[key] = file.name

    def open(self, name):
        """Cache a read-only safetensors handle for one checkpoint shard."""
        if name not in self.handles:
            self.handles[name] = safe_open(
                self.directory / name, framework="pt", device="cpu"
            )
        return self.handles[name]

    def keys(self):
        """List logical tensor names across every indexed shard."""
        return self.locations.keys()

    def get_tensor(self, name):
        """Read a complete tensor from the shard named by the checkpoint index."""
        return self.open(self.locations[name]).get_tensor(name)

    def get_slice(self, name):
        """Expose lazy tensor slicing without expanding an entire weight matrix."""
        return self.open(self.locations[name]).get_slice(name)

    def metadata(self, key):
        """Resolve per-matrix quantization metadata across supported name prefixes."""
        q = self.config.get("quantization", self.config.get("quantization_config", {}))
        relative = key.removeprefix("language_model.").removeprefix("model.")
        for name in [key, "model." + relative, relative]:
            entry = q.get("tensor_formats", {}).get(name, q.get(name))
            if isinstance(entry, dict):
                return entry
        return q

    def dequant(self, key, start, end):
        """Decode selected matrix rows to FP32 using independent reference codecs."""
        t = self.get_slice(key + ".weight")[start:end]
        if t.dtype in (torch.float32, torch.float16, torch.bfloat16):
            return t.float()
        meta = self.metadata(key)
        if t.dtype == torch.uint32:
            bits, group = meta["bits"], meta["group_size"]
            packed = t.to(torch.int64)
            width = packed.shape[1] * 32 // bits
            if 32 % bits == 0:
                shifts = torch.arange(32 // bits) * bits
                codes = ((packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).flatten(
                    1
                )
            else:
                bit = torch.arange(width, dtype=torch.int64) * bits
                word, shift = bit // 32, bit % 32
                codes = packed[:, word] >> shift
                high = packed[:, (word + 1).clamp_max(packed.shape[1] - 1)] << (
                    32 - shift
                )
                codes = (codes | torch.where(shift + bits > 32, high, 0)) & (
                    (1 << bits) - 1
                )
            scales = (
                self.get_slice(key + ".scales")[start:end]
                .float()
                .repeat_interleave(group, -1)
            )
            biases = (
                self.get_slice(key + ".biases")[start:end]
                .float()
                .repeat_interleave(group, -1)
            )
            return codes.float() * scales + biases
        kind = meta.get("format")
        if kind in ("q4_0", "q4_k", "q8_0"):
            return torch.from_numpy(decode_ggml(t.numpy(), kind))
        if key + ".weight_scale_2" in self.locations:
            codes = torch.stack((t & 15, t >> 4), -1).flatten(1).long()
            lut = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            )
            scales = self.get_slice(key + ".weight_scale")[start:end].float()
            global_scale = self.get_tensor(key + ".weight_scale_2").float().item()
            return lut[codes] * (scales * global_scale).repeat_interleave(16, -1)
        if t.dtype == torch.uint8:
            t = t.view(torch.float8_e4m3fn)
        values = torch.nan_to_num(t.float(), nan=0.0)
        scale_slice = self.get_slice(key + ".weight_scale")
        size = int(np.prod(scale_slice.get_shape()))
        scale = (
            (
                scale_slice[start:end]
                if size > 1
                else self.get_tensor(key + ".weight_scale")
            )
            .float()
            .reshape(-1, 1)
        )
        return values * scale


def decode_ggml(data, kind):
    """Decode Q4_0, Q4_K or Q8_0 byte blocks into row-major FP32 weights."""
    rows = data.shape[0]
    block_bytes = {"q4_0": 18, "q8_0": 34, "q4_k": 144}[kind]
    blocks = data.reshape(rows, -1, block_bytes)
    d = blocks[..., :2].copy().view("<f2").astype(np.float32)
    if kind == "q8_0":
        return (d * blocks[..., 2:].copy().view(np.int8).astype(np.float32)).reshape(
            rows, -1
        )
    if kind == "q4_0":
        q = blocks[..., 2:]
        values = np.concatenate((q & 15, q >> 4), axis=-1).astype(np.float32) - 8
        return (d * values).reshape(rows, -1)
    dm = blocks[..., 2:4].copy().view("<f2").astype(np.float32)
    scales = blocks[..., 4:16]
    output = np.empty((*blocks.shape[:2], 256), dtype=np.float32)
    for sub in range(8):
        if sub < 4:
            sc, minimum = scales[..., sub] & 63, scales[..., sub + 4] & 63
        else:
            sc = (scales[..., sub + 4] & 15) | ((scales[..., sub - 4] >> 6) << 4)
            minimum = (scales[..., sub + 4] >> 4) | ((scales[..., sub] >> 6) << 4)
        q = (
            blocks[..., 16 + (sub // 2) * 32 : 48 + (sub // 2) * 32] >> ((sub % 2) * 4)
        ) & 15
        output[..., sub * 32 : (sub + 1) * 32] = (
            d * sc[..., None].astype(np.float32)
        ) * q - dm * minimum[..., None].astype(np.float32)
    return output.reshape(rows, -1)


def fixture_matrix(value, kind, group, coeff_dtype):
    """Build legal encodings independently of the Rust/GPU implementation."""
    n, width = value.shape
    if kind in ("f32", "f16", "bf16"):
        dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[
            kind
        ]
        return {"weight": value.to(dtype).contiguous()}, {"format": kind}
    if kind.startswith("mlx"):
        bits = int(kind[3:])
        x = value.reshape(n, -1, group)
        lo = x.amin(-1).to(coeff_dtype).float()
        scale = (
            ((x.amax(-1) - x.amin(-1)) / ((1 << bits) - 1))
            .clamp_min(1e-8)
            .to(coeff_dtype)
            .float()
        )
        codes = (
            ((x - lo.unsqueeze(-1)) / scale.unsqueeze(-1))
            .round()
            .clamp(0, (1 << bits) - 1)
            .reshape(n, width)
            .long()
        )
        bit = torch.arange(width) * bits
        words, shifts = bit // 32, bit % 32
        packed = torch.zeros((n, width * bits // 32), dtype=torch.int64)
        packed.scatter_add_(1, words.expand(n, -1), (codes << shifts) & 0xFFFFFFFF)
        high = torch.where(shifts + bits > 32, codes >> (32 - shifts), 0)
        packed.scatter_add_(
            1, (words + 1).clamp_max(packed.shape[1] - 1).expand(n, -1), high
        )
        return {
            "weight": packed.to(torch.uint32),
            "scales": scale.to(coeff_dtype),
            "biases": lo.to(coeff_dtype),
        }, {"format": "mlx", "mode": "affine", "bits": bits, "group_size": group}
    if kind in ("q4_0", "q8_0"):
        x = value.reshape(n, -1, 32)
        if kind == "q8_0":
            d = x.abs().amax(-1) / 127
            q = (
                (x / d.unsqueeze(-1))
                .round()
                .clamp(-127, 127)
                .to(torch.int8)
                .view(torch.uint8)
            )
        else:
            maximum = x.gather(-1, x.abs().argmax(-1, keepdim=True)).squeeze(-1)
            d = maximum / -8
            codes = (x / d.unsqueeze(-1) + 8.5).floor().clamp(0, 15).to(torch.uint8)
            q = codes[..., :16] | (codes[..., 16:] << 4)
        raw = torch.cat(
            (d.half().contiguous().view(torch.uint8).reshape(n, -1, 2), q), -1
        )
        return {"weight": raw.reshape(n, -1).contiguous()}, {"format": kind}
    if kind == "q4_k":
        x = value.reshape(n, -1, 8, 32)
        maximum = x.abs().amax((-1, -2)).clamp_min(1e-8)
        d = (maximum * (2 / (15 * 63))).half().float()
        dm = (maximum / 63).half().float()
        sc = torch.tensor([63, 33, 17, 45, 58, 24, 39, 51], dtype=torch.uint8)
        minimum = torch.tensor([47, 63, 21, 34, 52, 61, 12, 42], dtype=torch.uint8)
        q = (
            (
                (x + (dm.unsqueeze(-1) * minimum).unsqueeze(-1))
                / (d.unsqueeze(-1) * sc).unsqueeze(-1)
            )
            .round()
            .clamp(0, 15)
            .to(torch.uint8)
        )
        sb = torch.cat(
            (
                sc[:4] | ((sc[4:] >> 4) << 6),
                minimum[:4] | ((minimum[4:] >> 4) << 6),
                (sc[4:] & 15) | ((minimum[4:] & 15) << 4),
            )
        )
        packed = q[..., 0::2, :] | (q[..., 1::2, :] << 4)
        raw = torch.cat(
            (
                d.half().contiguous().view(torch.uint8).reshape(n, -1, 2),
                dm.half().contiguous().view(torch.uint8).reshape(n, -1, 2),
                sb.expand(n, x.shape[1], 12),
                packed.flatten(-2),
            ),
            -1,
        )
        return {"weight": raw.reshape(n, -1).contiguous()}, {"format": kind}
    if kind.startswith("fp8"):
        scale = (value.abs().amax(-1) / 400).clamp_min(1e-8)
        if kind == "fp8_tensor":
            scale = scale.amax().reshape(1)
        codes = (value / scale.reshape(-1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        return {"weight": codes, "weight_scale": scale.to(coeff_dtype)}, {
            "format": "fp8"
        }
    if kind == "nvfp4":
        x = value.reshape(n, -1, 16)
        global_scale = 0.375
        scale = (
            (x.abs().amax(-1) / (6 * global_scale))
            .clamp_min(1 / 512)
            .to(torch.float8_e4m3fn)
        )
        normalized = x / (scale.float() * global_scale).unsqueeze(-1)
        lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        codes = (normalized.abs().unsqueeze(-1) - lut).abs().argmin(-1).to(torch.uint8)
        codes |= (normalized < 0).to(torch.uint8) * 8
        codes = codes.reshape(n, width)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return {
            "weight": packed,
            "weight_scale": scale,
            "weight_scale_2": torch.tensor([global_scale]),
        }, {"format": "nvfp4"}
    raise ValueError(kind)
