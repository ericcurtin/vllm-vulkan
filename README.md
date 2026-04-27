# vllm-vulkan

An out-of-tree vLLM hardware plugin that enables cross-platform GPU inference
via Vulkan.

| Platform | Architecture | Vulkan provider |
|---|---|---|
| macOS | aarch64 | KosmicKrisp (Vulkan → Metal translation) |
| macOS | x86_64  | KosmicKrisp / MoltenVK |
| Linux | x86_64  | Native Vulkan driver |
| Linux | aarch64 | Native Vulkan driver |

## Installation

```bash
# macOS — install KosmicKrisp / MoltenVK first
brew install molten-vk

# Linux — install Vulkan loader
sudo apt-get install -y libvulkan-dev   # Debian/Ubuntu
sudo dnf install -y vulkan-loader-devel # Fedora/RHEL

# Install the plugin
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-vulkan/main/install.sh | bash
```

## Usage

```bash
source ~/.venv-vllm-vulkan/bin/activate
vllm serve Qwen/Qwen3-0.6B --max-model-len 4096
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `VLLM_VULKAN_MEMORY_FRACTION` | `0.9` | Fraction of RAM for KV cache |
| `VLLM_VULKAN_BLOCK_SIZE` | `16` | Tokens per KV-cache block |
| `VLLM_VULKAN_DEBUG` | `0` | Enable verbose debug logging |
| `VLLM_VULKAN_DEVICE_INDEX` | `0` | Vulkan physical device index |

## How it works

The plugin registers `VulkanPlatform` with vLLM via the
`vllm.platform_plugins` entry point.  vLLM's CPU worker handles tokenisation,
sampling and KV-cache management.  A Rust extension (`_rs`) built with PyO3
enumerates Vulkan physical devices at startup to confirm availability.

On **macOS**, Vulkan API calls are translated to Metal by KosmicKrisp, a
community-maintained fork of MoltenVK.  This means the same Vulkan-based
operator library runs on Apple Silicon without writing Metal-specific code.

On **Linux**, the system Vulkan loader dispatches calls directly to the GPU
driver.

## Development

```bash
# Install dev dependencies
uv venv .venv-vllm-vulkan --python 3.12
source .venv-vllm-vulkan/bin/activate
uv pip install -e ".[dev]"

# Lint
scripts/lint.sh

# Test
pytest tests/python/ -v
```

## License

Apache-2.0
