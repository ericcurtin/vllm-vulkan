//! Buffer/Memory Management
//!
//! This module provides Vulkan buffer management wrapping ggml's buffer abstraction.

#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use std::sync::Arc;
use parking_lot::RwLock;

/// Internal buffer data
struct BufferData {
    data: Vec<u8>,
    size: usize,
    device_idx: usize,
    is_device_local: bool,
}

/// Vulkan buffer wrapper
#[pyclass]
pub struct VulkanBuffer {
    inner: Arc<RwLock<BufferData>>,
    #[pyo3(get)]
    size: usize,
    #[pyo3(get)]
    device_idx: usize,
    #[pyo3(get)]
    is_device_local: bool,
}

#[pymethods]
impl VulkanBuffer {
    /// Allocate a new buffer
    #[new]
    #[pyo3(signature = (size, device_idx=0, device_local=true))]
    pub fn new(size: usize, device_idx: usize, device_local: bool) -> PyResult<Self> {
        if size == 0 {
            return Err(PyRuntimeError::new_err("Buffer size must be > 0"));
        }

        let data = BufferData {
            data: vec![0u8; size],
            size,
            device_idx,
            is_device_local: device_local,
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(data)),
            size,
            device_idx,
            is_device_local: device_local,
        })
    }

    /// Create a buffer from a numpy array
    #[staticmethod]
    #[pyo3(signature = (arr, device_idx=0))]
    pub fn from_numpy(arr: PyReadonlyArray1<f32>, device_idx: usize) -> PyResult<Self> {
        let slice = arr.as_slice()?;
        let size = slice.len() * std::mem::size_of::<f32>();

        let mut data = vec![0u8; size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                slice.as_ptr() as *const u8,
                data.as_mut_ptr(),
                size,
            );
        }

        let buffer_data = BufferData {
            data,
            size,
            device_idx,
            is_device_local: true,
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(buffer_data)),
            size,
            device_idx,
            is_device_local: true,
        })
    }

    /// Convert buffer to numpy array (f32)
    pub fn to_numpy_f32<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let inner = self.inner.read();
        let num_elements = inner.size / std::mem::size_of::<f32>();

        let arr = PyArray1::<f32>::zeros_bound(py, num_elements, false);
        {
            let mut arr_rw = unsafe { arr.as_array_mut() };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    inner.data.as_ptr() as *const f32,
                    arr_rw.as_mut_ptr(),
                    num_elements,
                );
            }
        }
        Ok(arr)
    }

    /// Copy data from another buffer
    pub fn copy_from(&self, src: &VulkanBuffer) -> PyResult<()> {
        let src_inner = src.inner.read();
        let mut dst_inner = self.inner.write();

        if src_inner.size != dst_inner.size {
            return Err(PyRuntimeError::new_err(format!(
                "Buffer size mismatch: src={}, dst={}",
                src_inner.size, dst_inner.size
            )));
        }

        dst_inner.data.copy_from_slice(&src_inner.data);
        Ok(())
    }

    /// Copy data to another buffer
    pub fn copy_to(&self, dst: &VulkanBuffer) -> PyResult<()> {
        dst.copy_from(self)
    }

    /// Fill buffer with a value
    pub fn fill(&self, value: u8) -> PyResult<()> {
        let mut inner = self.inner.write();
        inner.data.fill(value);
        Ok(())
    }

    /// Get a slice of the buffer as bytes
    pub fn get_bytes(&self, offset: usize, length: usize) -> PyResult<Vec<u8>> {
        let inner = self.inner.read();
        if offset + length > inner.size {
            return Err(PyRuntimeError::new_err("Slice out of bounds"));
        }
        Ok(inner.data[offset..offset + length].to_vec())
    }

    /// Set bytes in the buffer
    pub fn set_bytes(&self, offset: usize, data: Vec<u8>) -> PyResult<()> {
        let mut inner = self.inner.write();
        if offset + data.len() > inner.size {
            return Err(PyRuntimeError::new_err("Write out of bounds"));
        }
        inner.data[offset..offset + data.len()].copy_from_slice(&data);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "VulkanBuffer(size={}, device={}, device_local={})",
            self.size, self.device_idx, self.is_device_local
        )
    }

    fn __len__(&self) -> usize {
        self.size
    }
}

impl VulkanBuffer {
    /// Get raw pointer to data (internal use only)
    pub fn as_ptr(&self) -> *const u8 {
        self.inner.read().data.as_ptr()
    }

    /// Get mutable raw pointer to data (internal use only)
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.inner.write().data.as_mut_ptr()
    }

    /// Get a reference to the internal data
    pub fn data(&self) -> Vec<u8> {
        self.inner.read().data.clone()
    }
}

/// Allocate a buffer (standalone function for Python)
#[pyfunction]
#[pyo3(signature = (size, device_idx=0, device_local=true))]
pub fn allocate_buffer(size: usize, device_idx: usize, device_local: bool) -> PyResult<VulkanBuffer> {
    VulkanBuffer::new(size, device_idx, device_local)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buf = VulkanBuffer::new(1024, 0, true).unwrap();
        assert_eq!(buf.size, 1024);
        assert_eq!(buf.device_idx, 0);
        assert!(buf.is_device_local);
    }

    #[test]
    fn test_buffer_copy() {
        let buf1 = VulkanBuffer::new(1024, 0, true).unwrap();
        let buf2 = VulkanBuffer::new(1024, 0, true).unwrap();

        buf1.fill(42).unwrap();
        buf2.copy_from(&buf1).unwrap();

        let bytes = buf2.get_bytes(0, 10).unwrap();
        assert!(bytes.iter().all(|&b| b == 42));
    }

    #[test]
    fn test_buffer_bytes() {
        let buf = VulkanBuffer::new(1024, 0, true).unwrap();
        let data = vec![1, 2, 3, 4, 5];
        buf.set_bytes(0, data.clone()).unwrap();

        let read_data = buf.get_bytes(0, 5).unwrap();
        assert_eq!(read_data, data);
    }
}
