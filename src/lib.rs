// SPDX-License-Identifier: Apache-2.0
//! vLLM-Vulkan Rust extension.
//!
//! On macOS, Vulkan calls are translated to Metal by KosmicKrisp
//! (Mesa/Zink software Vulkan driver).  On Linux x86_64 and aarch64,
//! native Vulkan is used directly.
//!
//! The PyO3 module `_rs` is the bridge between this crate and the Python
//! package `vllm_vulkan`.

mod device;
mod ggml;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub use device::VulkanDevice;

// ─── Module-level helper functions ──────────────────────────────────────────

/// Return True if at least one Vulkan-capable device is present.
/// On macOS this requires KosmicKrisp to be installed.
#[pyfunction]
fn is_available() -> bool {
    device::is_vulkan_available()
}

/// Number of Vulkan-capable physical devices.
#[pyfunction]
fn get_device_count() -> usize {
    device::device_count()
}

/// Enumerate all Vulkan physical devices as a list of dicts.
#[pyfunction]
fn enumerate_devices(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    device::enumerate_devices()
        .into_iter()
        .map(|info| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("name", &info.name)?;
            dict.set_item("vendor_id", info.vendor_id)?;
            dict.set_item("device_type", &info.device_type)?;
            dict.set_item("api_version", &info.api_version)?;
            dict.set_item("driver_version", info.driver_version)?;
            dict.set_item("total_memory_bytes", info.total_memory_bytes)?;
            Ok(dict.into())
        })
        .collect()
}

/// Return device info dict for `device_idx`.
#[pyfunction]
fn get_device_info(py: Python<'_>, device_idx: usize) -> PyResult<PyObject> {
    let devs = device::enumerate_devices();
    let info = devs
        .get(device_idx)
        .ok_or_else(|| PyRuntimeError::new_err(format!("no device at index {device_idx}")))?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("name", &info.name)?;
    dict.set_item("vendor_id", info.vendor_id)?;
    dict.set_item("device_type", &info.device_type)?;
    dict.set_item("api_version", &info.api_version)?;
    dict.set_item("driver_version", info.driver_version)?;
    dict.set_item("total_memory_bytes", info.total_memory_bytes)?;
    Ok(dict.into())
}

/// Synchronise all pending Vulkan work across all devices.
#[pyfunction]
fn synchronize() -> PyResult<()> {
    device::synchronize_all().map_err(PyRuntimeError::new_err)
}

/// Return `(used_bytes, total_bytes)` for `device_idx`.
#[pyfunction]
fn get_memory_info(device_idx: usize) -> PyResult<(u64, u64)> {
    device::memory_info(device_idx).map_err(PyRuntimeError::new_err)
}

/// Run a small FP32 matrix multiplication through ggml's Vulkan backend.
///
/// Inputs are Python nested sequences with standard shapes:
///   a: (m, k)
///   b: (k, n)
/// The return value is a nested list with shape (m, n).
#[pyfunction]
#[pyo3(signature = (a, b, device_idx = 0))]
fn vulkan_matmul(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device_idx: usize) -> PyResult<Vec<Vec<f32>>> {
    let (a_flat, b_transposed, m, k, n) =
        flatten_matmul_inputs(&a, &b).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let result = ggml::vulkan_matmul_f32(&a_flat, &b_transposed, m, k, n, device_idx)
        .map_err(PyRuntimeError::new_err)?;

    Ok(result.output.chunks(n).map(|row| row.to_vec()).collect())
}

/// Run Vulkan matmul and return `(output, backend_name)`.
///
/// This is a diagnostic API for tests and backend bring-up. It lets tests
/// assert that ggml assigned the matmul result to the Vulkan backend instead
/// of silently falling back to CPU.
#[pyfunction]
#[pyo3(signature = (a, b, device_idx = 0))]
fn vulkan_matmul_with_backend(
    a: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    device_idx: usize,
) -> PyResult<(Vec<Vec<f32>>, String)> {
    let (a_flat, b_transposed, m, k, n) =
        flatten_matmul_inputs(&a, &b).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let result = ggml::vulkan_matmul_f32(&a_flat, &b_transposed, m, k, n, device_idx)
        .map_err(PyRuntimeError::new_err)?;
    let output = result.output.chunks(n).map(|row| row.to_vec()).collect();

    Ok((output, result.backend_name))
}

fn flatten_matmul_inputs(
    a: &[Vec<f32>],
    b: &[Vec<f32>],
) -> Result<(Vec<f32>, Vec<f32>, usize, usize, usize), String> {
    let m = a.len();
    let k = a
        .first()
        .map(Vec::len)
        .ok_or_else(|| "left matrix must have at least one row".to_string())?;

    if k == 0 {
        return Err("left matrix must have at least one column".to_string());
    }

    if a.iter().any(|row| row.len() != k) {
        return Err("left matrix rows must all have the same length".to_string());
    }

    if b.len() != k {
        return Err(format!(
            "shape mismatch: left matrix is ({m}, {k}) but right matrix has {} row(s)",
            b.len()
        ));
    }

    let n = b
        .first()
        .map(Vec::len)
        .ok_or_else(|| "right matrix must have at least one row".to_string())?;

    if n == 0 {
        return Err("right matrix must have at least one column".to_string());
    }

    if b.iter().any(|row| row.len() != n) {
        return Err("right matrix rows must all have the same length".to_string());
    }

    let a_flat = a.iter().flat_map(|row| row.iter().copied()).collect();
    let mut b_transposed = vec![0.0; n * k];
    for row in 0..k {
        for col in 0..n {
            b_transposed[col * k + row] = b[row][col];
        }
    }

    Ok((a_flat, b_transposed, m, k, n))
}

// ─── PyO3 module ────────────────────────────────────────────────────────────

/// Python-visible module `vllm_vulkan._rs`.
#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::try_init().ok();

    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_devices, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(get_memory_info, m)?)?;
    m.add_function(wrap_pyfunction!(vulkan_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(vulkan_matmul_with_backend, m)?)?;

    m.add_class::<VulkanDevice>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__vulkan_available__", device::is_vulkan_available())?;

    Ok(())
}
