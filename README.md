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
