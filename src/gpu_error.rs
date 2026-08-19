// SPDX-License-Identifier: Apache-2.0
//! Error type for the GPU decode vertical (gpu_layer/forward_qwen_gpu_resident).
//!
//! Most `compute::ComputeEngine` methods already return `Result<_, String>`, so
//! `?` + `From<String>` threads cleanly through the vertical. This replaces
//! `.unwrap()` on a transient Vulkan failure (alloc/device-loss) with a
//! Python-catchable `RuntimeError` instead of a process-aborting panic. Panics
//! are kept only for documented compile-time/layout invariants (slot indices,
//! `LayerSpec` shape assumptions) that a caller cannot recover from anyway.

#[derive(Debug)]
pub enum GpuError {
    Vulkan(String),
    ShaderMissing(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Vulkan(s) => write!(f, "GPU error: {s}"),
            GpuError::ShaderMissing(s) => write!(f, "GPU shader missing: {s}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<String> for GpuError {
    fn from(s: String) -> Self {
        GpuError::Vulkan(s)
    }
}

impl From<GpuError> for pyo3::PyErr {
    fn from(e: GpuError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

pub type GpuResult<T> = Result<T, GpuError>;
