# vLLM-Vulkan

A vLLM hardware plugin that enables GPU acceleration via Vulkan.

## Architecture

1. **VulkanPlatform**: Main platform class implementing vLLM's Platform interface
2. **VulkanWorker**: Per-device worker managing model execution
3. **VulkanModelRunner**: Orchestrates model forward passes
4. **VulkanAttentionBackend**: Implements paged attention using Vulkan

## Acknowledgments

- [ggml](https://github.com/ggerganov/ggml) - Tensor library with Vulkan backend
