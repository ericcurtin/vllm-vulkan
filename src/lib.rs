//! vLLM-Vulkan Rust Extension
//!
//! This module provides Python bindings for Vulkan-based GPU acceleration
//! using ggml-vulkan as the backend.

mod backend;
mod buffer;
mod cache;
mod device;
mod graph;
mod tensor;
mod attention;
mod distributed;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

// Re-export key types for internal use
pub use backend::VulkanBackend;
pub use buffer::VulkanBuffer;
pub use cache::PagedKVCache;
pub use device::VulkanDevice;
pub use graph::VulkanGraph;
pub use tensor::VulkanTensor;
pub use attention::{flash_attention, paged_attention};
pub use distributed::VulkanCommunicator;

/// Check if Vulkan is available on this system
#[pyfunction]
fn is_available() -> bool {
    device::is_vulkan_available()
}

/// Get the number of Vulkan-capable devices
#[pyfunction]
fn get_device_count() -> usize {
    device::get_device_count()
}

/// Enumerate all Vulkan-capable devices
#[pyfunction]
fn enumerate_devices() -> PyResult<Vec<pyo3::Py<pyo3::types::PyDict>>> {
    Python::with_gil(|py| {
        let devices = device::enumerate_devices();
        let mut result = Vec::new();

        for info in devices {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("name", info.name)?;
            dict.set_item("vendor", info.vendor)?;
            dict.set_item("memory_mb", info.memory_mb)?;
            dict.set_item("device_type", info.device_type)?;
            dict.set_item("api_version", info.api_version)?;
            dict.set_item("driver_version", info.driver_version)?;
            dict.set_item("supports_fp16", info.supports_fp16)?;
            dict.set_item("supports_int8", info.supports_int8)?;
            dict.set_item("max_compute_work_group_count", info.max_compute_work_group_count)?;
            dict.set_item("max_compute_work_group_size", info.max_compute_work_group_size)?;
            result.push(dict.unbind());
        }

        Ok(result)
    })
}

/// Get device info for a specific device index
#[pyfunction]
fn get_device_info(device_idx: usize) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    let devices = device::enumerate_devices();
    if device_idx >= devices.len() {
        return Err(PyRuntimeError::new_err(format!(
            "Device index {} out of range (0-{})",
            device_idx,
            devices.len().saturating_sub(1)
        )));
    }

    Python::with_gil(|py| {
        let info = &devices[device_idx];
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("name", &info.name)?;
        dict.set_item("vendor", &info.vendor)?;
        dict.set_item("memory_mb", info.memory_mb)?;
        dict.set_item("device_type", &info.device_type)?;
        dict.set_item("api_version", &info.api_version)?;
        dict.set_item("driver_version", &info.driver_version)?;
        dict.set_item("supports_fp16", info.supports_fp16)?;
        dict.set_item("supports_int8", info.supports_int8)?;
        dict.set_item("max_compute_work_group_count", info.max_compute_work_group_count)?;
        dict.set_item("max_compute_work_group_size", info.max_compute_work_group_size)?;
        Ok(dict.unbind())
    })
}

/// Synchronize all Vulkan operations
#[pyfunction]
fn synchronize() -> PyResult<()> {
    device::synchronize_all()
        .map_err(|e| PyRuntimeError::new_err(format!("Synchronization failed: {}", e)))
}

/// Get current memory usage for a device
#[pyfunction]
fn get_memory_info(device_idx: usize) -> PyResult<(u64, u64)> {
    device::get_memory_info(device_idx)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get memory info: {}", e)))
}

/// Python module definition
#[pymodule]
fn _vllm_vulkan_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging
    env_logger::try_init().ok();

    // Module functions
    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_devices, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(get_memory_info, m)?)?;

    // Classes
    m.add_class::<VulkanDevice>()?;
    m.add_class::<VulkanBuffer>()?;
    m.add_class::<VulkanBackend>()?;
    m.add_class::<VulkanTensor>()?;
    m.add_class::<VulkanGraph>()?;
    m.add_class::<PagedKVCache>()?;
    m.add_class::<VulkanCommunicator>()?;

    // Attention functions
    m.add_function(wrap_pyfunction!(attention::flash_attention_py, m)?)?;
    m.add_function(wrap_pyfunction!(attention::paged_attention_py, m)?)?;

    // Cache functions
    m.add_function(wrap_pyfunction!(cache::reshape_and_cache_py, m)?)?;
    m.add_function(wrap_pyfunction!(cache::copy_blocks_py, m)?)?;
    m.add_function(wrap_pyfunction!(cache::swap_blocks_py, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__vulkan_available__", device::is_vulkan_available())?;

    Ok(())
}
