# vLLM-Vulkan

A vLLM hardware plugin that enables GPU acceleration via Vulkan using ggml-vulkan backend.

## Overview

vLLM-Vulkan provides a full-featured vLLM hardware plugin architecture:
- **Python layer**: vLLM plugin interface (Platform, Worker, ModelRunner, AttentionBackend)
- **Rust layer**: PyO3 bindings wrapping ggml-vulkan for high-performance compute
- **C layer**: ggml-vulkan backend (linked as dependency)

## Prerequisites

- **Vulkan SDK**: Install from [LunarG](https://vulkan.lunarg.com/)
- **Rust toolchain**: Install via [rustup](https://rustup.rs/)
- **CMake 3.16+**: Required for building ggml
- **Python 3.9+**: Required for vLLM

### Ubuntu/Debian
```bash
sudo apt-get install -y vulkan-tools libvulkan-dev vulkan-validationlayers
```

### macOS (MoltenVK)
```bash
brew install molten-vk vulkan-headers vulkan-loader
```

### Windows
Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows).

## Installation

### Clone with submodules
```bash
git clone --recursive https://github.com/ericcurtin/vllm-vulkan
cd vllm-vulkan
```

### Install with pip
```bash
pip install -e .
```

### Verify installation
```python
import vllm_vulkan
print(f"Vulkan devices: {vllm_vulkan.get_device_count()}")
```

## Usage

### Basic Usage with vLLM

vLLM will automatically discover the Vulkan plugin:

```python
from vllm import LLM

# Use Vulkan backend when CUDA is unavailable
llm = LLM(model="meta-llama/Llama-2-7b", device="vulkan")
output = llm.generate("Hello, ")
print(output)
```

### Direct API Usage

```python
import vllm_vulkan

# Enumerate Vulkan devices
devices = vllm_vulkan.enumerate_devices()
for i, device in enumerate(devices):
    print(f"Device {i}: {device['name']}")
    print(f"  Memory: {device['memory_mb']} MB")
    print(f"  Vendor: {device['vendor']}")

# Check if Vulkan is available
if vllm_vulkan.is_available():
    print("Vulkan backend is available!")
```

## Architecture

### Project Structure

```
vllm-vulkan/
├── vllm_vulkan/          # Python vLLM plugin
│   ├── __init__.py       # Plugin registration
│   ├── platform.py       # VulkanPlatform class
│   ├── worker.py         # VulkanWorker class
│   ├── executor.py       # VulkanExecutor class
│   ├── model_runner.py   # VulkanModelRunner class
│   ├── attention/        # Attention backend
│   ├── ops/              # Operations (attention, cache, quantization)
│   ├── distributed/      # Multi-GPU communication
│   └── utils/            # Device utilities
├── src/                  # Rust PyO3 extension
│   ├── lib.rs            # Module entry point
│   ├── device.rs         # Vulkan device management
│   ├── buffer.rs         # Buffer/memory management
│   ├── backend.rs        # ggml backend wrapper
│   ├── tensor.rs         # Tensor operations
│   ├── attention.rs      # Attention kernel dispatch
│   ├── cache.rs          # Paged KV cache
│   ├── graph.rs          # Compute graph execution
│   └── distributed.rs    # Multi-GPU communication
├── ggml/                 # ggml submodule
├── tests/                # Test suites
├── benchmarks/           # Performance benchmarks
└── examples/             # Usage examples
```

### Key Components

1. **VulkanPlatform**: Main platform class implementing vLLM's Platform interface
2. **VulkanWorker**: Per-device worker managing model execution
3. **VulkanModelRunner**: Orchestrates model forward passes
4. **VulkanAttentionBackend**: Implements paged attention using Vulkan

## Supported Models

Initial support includes:
- LLaMA 2/3
- Mistral
- Qwen

More models will be added based on ggml support.

## Quantization Support

vLLM-Vulkan supports ggml quantization formats:
- F32, F16 (full precision)
- Q4_0, Q4_1, Q4_K (4-bit)
- Q5_0, Q5_1, Q5_K (5-bit)
- Q8_0, Q8_1, Q8_K (8-bit)

## Multi-GPU Support

vLLM-Vulkan supports multi-GPU inference with:
- **Tensor parallelism**: Distributing model layers across GPUs
- **Pipeline parallelism**: Sequential layer processing across GPUs

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b",
    device="vulkan",
    tensor_parallel_size=4,
)
```

## Development

### Setup development environment
```bash
pip install -e ".[dev]"
```

### Run tests
```bash
pytest tests/python/
```

### Run benchmarks
```bash
python benchmarks/attention_benchmark.py
```

### Build documentation
```bash
# Coming soon
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

Apache License 2.0

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [ggml](https://github.com/ggerganov/ggml) - Tensor library with Vulkan backend
- [PyO3](https://pyo3.rs/) - Rust bindings for Python
