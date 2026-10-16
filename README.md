# vllm-vulkan

An out-of-tree vLLM hardware plugin that enables cross-platform GPU inference
via Vulkan.

| Platform | Architecture | Vulkan provider |
|---|---|---|
| macOS | aarch64 | KosmicKrisp (Vulkan → Metal translation) |
| Linux | x86_64  | Native Vulkan driver |
| Linux | aarch64 | Native Vulkan driver |

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/ericcurtin/vllm-vulkan/main/install.sh | bash
```


## Running on a GPU

The plugin runs on `device_type=cpu` and offloads compute to whatever Vulkan
device it selects, so vLLM is launched normally:

```bash
VLLM_CPU_OMP_THREADS_BIND=nobind vllm serve <model> --enforce-eager
```

It picks the first discrete Vulkan device, so a software ICD (`llvmpipe`) sitting
alongside a real GPU is skipped automatically. Check what Vulkan sees with
`vulkaninfo --summary`.

Validated on AMD RX 7900 XTX (gfx1100), RX 6900 XT (gfx1030), and NVIDIA GB10
(Blackwell, aarch64).

### Native LFM2.5 forward

`vllm_vulkan._rs.Lfm2Model` runs LFM2 checkpoints, including LFM2.5-2.6B,
with floating, MLX affine, GGML Q4/Q8, FP8 and NVFP4 weights through a
standalone, single-sequence Vulkan forward engine. Activations, KV cache and
convolution state stay on the GPU; FP16 arithmetic and subgroup extensions
are not required.

The generation script requires `transformers` for `AutoTokenizer`, included
in the package dependencies.

```bash
PYO3_PYTHON=python3 bash scripts/build_lfm2.sh
PYTHONPATH=. python3 scripts/run_lfm2.py /path/to/checkpoint --device 1
```

```python
from vllm_vulkan._rs import Lfm2Model

model = Lfm2Model("/path/to/checkpoint", device_idx=1, max_seq_len=512)
logits = model.forward_prefill([1, 9, 17])
logits = model.forward(31, position=3)
model.reset_kv_cache()
```

Enumerate devices before selecting an index. Each instance owns one sequence
and stays on its creating thread. Empty chunks, invalid token IDs, nonconsecutive
positions and capacity overflow fail before cache mutation. Recreate the model
after a GPU recording/submission failure.

`quant="auto"` preserves native F32/F16/BF16, MLX affine 2/4/6/8-bit (groups
32/64/128), GGML Q4_0/Q4_K/Q8_0, FP8 E4M3 or NVFP4 storage. Explicit `quant`
accepts `f32`, `f16`, `bf16`, `mlx2`, `mlx4`, `mlx6`, `mlx8`, `q4_0`, `q4_k`
and `q8_0`; `group_size` selects the MLX group size. Checkpoints may use mixed
formats and indexed safetensors shards. `weight_formats()` reports residency,
and `scripts/convert_lfm2.py` exports packed weights for reuse. Q4_K CPU
conversion can be slow; export once and reload with `auto`.

The loader requires tied embeddings, bias-free convolutions and default
unscaled RoPE. GGUF loading, scheduler integration and batched/tiled prefill
are outside this API. Full F32/F16/BF16 2.6B weights exceed the RX 560X's 4 GiB.
`tests/python/test_lfm2.py` and `scripts/validate_lfm2.py` compare full logits
against independent Transformers forwards; validation reports are local outputs.

### In a container

**AMD (RADV).** Pass the render node and the device groups:

```bash
docker run --device /dev/dri/renderD128 --group-add video --group-add render ...
```

**NVIDIA.** `--gpus all` alone is not enough for Vulkan. The container runtime
injects the driver libraries but not the userland they link against, and the
driver's Vulkan ICD only initializes against a matching libc:

- **Match the base image to the host distro** (its glibc). A driver built for
  Ubuntu 24.04 (glibc 2.39) returns `VK_ERROR_INITIALIZATION_FAILED` under a
  22.04 (glibc 2.35) image, even though CUDA / `nvidia-smi` work. The failure is
  silent, at ICD negotiation.
- **Install the X/GL userland** `libGLX_nvidia` pulls in:
  `libx11-6 libxext6 libglvnd0 libgl1 libegl1 libvulkan1 libxcb1 libxau6 libxdmcp6`.
- Run with `-e NVIDIA_DRIVER_CAPABILITIES=all` and `--device /dev/dri/renderD<N>`.

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --device /dev/dri/renderD128 ...
```
