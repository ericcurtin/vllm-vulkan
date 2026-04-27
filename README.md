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

