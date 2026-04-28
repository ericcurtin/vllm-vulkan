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

## Development

The Rust extension links against ggml's Vulkan backend for compute bring-up.
The local installer initializes the ggml submodule and installs the Vulkan
build dependencies:

```bash
./install.sh
```

For manual setup, clone with submodules:

```bash
git clone --recurse-submodules https://github.com/ericcurtin/vllm-vulkan.git
```

or initialize them after cloning:

```bash
git submodule update --init --recursive
```

On macOS, ggml builds against the Homebrew Vulkan headers/loader, uses
`shaderc` for `glslc` shader compilation, and needs `spirv-headers` for SPIR-V
headers. The runtime Vulkan provider remains KosmicKrisp.

```bash
brew install vulkan-headers vulkan-loader shaderc spirv-headers
```

On Linux, install the Vulkan loader, headers, shader compiler, and SPIR-V
headers from your distribution:

```bash
# Debian / Ubuntu
sudo apt-get install -y build-essential cmake libvulkan-dev glslc spirv-headers

# Fedora / RHEL
sudo dnf install -y cmake gcc-c++ make vulkan-loader-devel glslc spirv-headers
```
