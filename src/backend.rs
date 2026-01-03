//! ggml Backend Wrapper
//!
//! This module wraps the ggml backend for Vulkan operations.

#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::atomic::{AtomicBool, Ordering};

/// Backend state
static BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Vulkan backend wrapper
#[pyclass]
pub struct VulkanBackend {
    device_idx: usize,
    #[pyo3(get)]
    is_initialized: bool,
}

#[pymethods]
impl VulkanBackend {
    /// Initialize the Vulkan backend for a specific device
    #[new]
    #[pyo3(signature = (device_idx=0))]
    pub fn new(device_idx: usize) -> PyResult<Self> {
        // Check device count
        let device_count = crate::device::get_device_count();
        if device_idx >= device_count {
            return Err(PyRuntimeError::new_err(format!(
                "Device index {} out of range (0-{})",
                device_idx,
                device_count.saturating_sub(1)
            )));
        }

        // Initialize backend
        // In real implementation, this would call ggml_backend_vk_init(device_idx)
        BACKEND_INITIALIZED.store(true, Ordering::SeqCst);

        log::info!("Initialized Vulkan backend for device {}", device_idx);

        Ok(Self {
            device_idx,
            is_initialized: true,
        })
    }

    /// Get the device index this backend is associated with
    pub fn device_index(&self) -> usize {
        self.device_idx
    }

    /// Synchronize the backend (wait for all operations to complete)
    pub fn synchronize(&self) -> PyResult<()> {
        if !self.is_initialized {
            return Err(PyRuntimeError::new_err("Backend not initialized"));
        }
        // In real implementation, this would call ggml_backend_synchronize
        Ok(())
    }

    /// Get the default buffer type for this backend
    pub fn get_default_buffer_type(&self) -> String {
        "vulkan_device".to_string()
    }

    /// Check if the backend supports a specific operation
    pub fn supports_op(&self, op_name: &str) -> bool {
        // List of supported operations
        let supported = [
            "matmul", "add", "mul", "div", "softmax", "layer_norm",
            "rms_norm", "rope", "attention", "conv_1d", "conv_2d",
            "pool_2d", "flash_attention", "paged_attention",
        ];
        supported.contains(&op_name.to_lowercase().as_str())
    }

    /// Get the maximum batch size supported
    pub fn max_batch_size(&self) -> usize {
        // This would be determined by device memory and capabilities
        2048
    }

    /// Get the maximum sequence length supported
    pub fn max_sequence_length(&self) -> usize {
        // This would be determined by device memory
        32768
    }

    /// Get memory alignment requirement
    pub fn memory_alignment(&self) -> usize {
        256 // Typical Vulkan alignment requirement
    }

    fn __repr__(&self) -> String {
        format!(
            "VulkanBackend(device={}, initialized={})",
            self.device_idx, self.is_initialized
        )
    }
}

impl VulkanBackend {
    /// Check if backend is initialized (internal use)
    pub fn initialized(&self) -> bool {
        self.is_initialized && BACKEND_INITIALIZED.load(Ordering::SeqCst)
    }
}

/// Check if the Vulkan backend is available
#[pyfunction]
pub fn backend_available() -> bool {
    crate::device::is_vulkan_available()
}

/// Get the number of available backends (one per device)
#[pyfunction]
pub fn backend_count() -> usize {
    crate::device::get_device_count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = VulkanBackend::new(0).unwrap();
        assert!(backend.is_initialized);
        assert_eq!(backend.device_idx, 0);
    }

    #[test]
    fn test_backend_ops() {
        let backend = VulkanBackend::new(0).unwrap();
        assert!(backend.supports_op("matmul"));
        assert!(backend.supports_op("attention"));
        assert!(!backend.supports_op("unknown_op"));
    }

    #[test]
    fn test_backend_sync() {
        let backend = VulkanBackend::new(0).unwrap();
        assert!(backend.synchronize().is_ok());
    }
}
