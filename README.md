# vLLM-Vulkan

A vLLM hardware plugin that enables GPU acceleration via Vulkan.

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

### Key Components

1. **VulkanPlatform**: Main platform class implementing vLLM's Platform interface
2. **VulkanWorker**: Per-device worker managing model execution
3. **VulkanModelRunner**: Orchestrates model forward passes
4. **VulkanAttentionBackend**: Implements paged attention using Vulkan

## Multi-GPU Support

vllm-vulkan supports multi-GPU inference with:
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

## Acknowledgments

- [ggml](https://github.com/ggerganov/ggml) - Tensor library with Vulkan backend
