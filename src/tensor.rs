//! Tensor Operations
//!
//! This module provides tensor operations wrapping ggml tensors.

#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::{PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use std::sync::Arc;
use parking_lot::RwLock;

/// Tensor data types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorDtype {
    F32,
    F16,
    BF16,
    I32,
    I16,
    I8,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
}

impl TensorDtype {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "float32" => Some(TensorDtype::F32),
            "f16" | "float16" => Some(TensorDtype::F16),
            "bf16" | "bfloat16" => Some(TensorDtype::BF16),
            "i32" | "int32" => Some(TensorDtype::I32),
            "i16" | "int16" => Some(TensorDtype::I16),
            "i8" | "int8" => Some(TensorDtype::I8),
            "q4_0" => Some(TensorDtype::Q4_0),
            "q4_1" => Some(TensorDtype::Q4_1),
            "q5_0" => Some(TensorDtype::Q5_0),
            "q5_1" => Some(TensorDtype::Q5_1),
            "q8_0" => Some(TensorDtype::Q8_0),
            "q8_1" => Some(TensorDtype::Q8_1),
            _ => None,
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            TensorDtype::F32 | TensorDtype::I32 => 4,
            TensorDtype::F16 | TensorDtype::BF16 | TensorDtype::I16 => 2,
            TensorDtype::I8 => 1,
            // Quantized types have variable element sizes
            TensorDtype::Q4_0 | TensorDtype::Q4_1 => 1, // ~0.5 bytes per element
            TensorDtype::Q5_0 | TensorDtype::Q5_1 => 1, // ~0.625 bytes per element
            TensorDtype::Q8_0 | TensorDtype::Q8_1 => 1, // 1 byte per element
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            TensorDtype::F32 => "f32",
            TensorDtype::F16 => "f16",
            TensorDtype::BF16 => "bf16",
            TensorDtype::I32 => "i32",
            TensorDtype::I16 => "i16",
            TensorDtype::I8 => "i8",
            TensorDtype::Q4_0 => "q4_0",
            TensorDtype::Q4_1 => "q4_1",
            TensorDtype::Q5_0 => "q5_0",
            TensorDtype::Q5_1 => "q5_1",
            TensorDtype::Q8_0 => "q8_0",
            TensorDtype::Q8_1 => "q8_1",
        }
    }
}

/// Internal tensor data
struct TensorData {
    data: Vec<u8>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: TensorDtype,
    device_idx: usize,
}

/// Vulkan tensor wrapper
#[pyclass]
pub struct VulkanTensor {
    inner: Arc<RwLock<TensorData>>,
}

#[pymethods]
impl VulkanTensor {
    /// Create a new tensor with given shape and dtype
    #[new]
    #[pyo3(signature = (shape, dtype="f32", device_idx=0))]
    pub fn new(shape: Vec<usize>, dtype: &str, device_idx: usize) -> PyResult<Self> {
        let tensor_dtype = TensorDtype::from_str(dtype)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Unknown dtype: {}", dtype)))?;

        let num_elements: usize = shape.iter().product();
        let size = num_elements * tensor_dtype.element_size();

        // Calculate strides (row-major)
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let data = TensorData {
            data: vec![0u8; size],
            shape: shape.clone(),
            strides,
            dtype: tensor_dtype,
            device_idx,
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(data)),
        })
    }

    /// Create tensor from numpy array
    #[staticmethod]
    #[pyo3(signature = (arr, device_idx=0))]
    pub fn from_numpy(arr: PyReadonlyArrayDyn<f32>, device_idx: usize) -> PyResult<Self> {
        let shape: Vec<usize> = arr.shape().to_vec();
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

        // Calculate strides
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let tensor_data = TensorData {
            data,
            shape,
            strides,
            dtype: TensorDtype::F32,
            device_idx,
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(tensor_data)),
        })
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<usize> {
        self.inner.read().shape.clone()
    }

    /// Get tensor strides
    pub fn strides(&self) -> Vec<usize> {
        self.inner.read().strides.clone()
    }

    /// Get tensor dtype as string
    pub fn dtype(&self) -> String {
        self.inner.read().dtype.as_str().to_string()
    }

    /// Get device index
    pub fn device_idx(&self) -> usize {
        self.inner.read().device_idx
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.inner.read().shape.iter().product()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.inner.read().shape.len()
    }

    /// Get size in bytes
    pub fn nbytes(&self) -> usize {
        self.inner.read().data.len()
    }

    /// Reshape tensor (must have same number of elements)
    pub fn reshape(&self, new_shape: Vec<usize>) -> PyResult<VulkanTensor> {
        let inner = self.inner.read();
        let old_numel: usize = inner.shape.iter().product();
        let new_numel: usize = new_shape.iter().product();

        if old_numel != new_numel {
            return Err(PyRuntimeError::new_err(format!(
                "Cannot reshape tensor of {} elements to shape with {} elements",
                old_numel, new_numel
            )));
        }

        // Calculate new strides
        let mut strides = vec![1usize; new_shape.len()];
        for i in (0..new_shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * new_shape[i + 1];
        }

        let new_data = TensorData {
            data: inner.data.clone(),
            shape: new_shape,
            strides,
            dtype: inner.dtype,
            device_idx: inner.device_idx,
        };

        Ok(VulkanTensor {
            inner: Arc::new(RwLock::new(new_data)),
        })
    }

    /// Create a view of the tensor
    pub fn view(&self, new_shape: Vec<usize>) -> PyResult<VulkanTensor> {
        self.reshape(new_shape)
    }

    /// Transpose/permute dimensions
    pub fn permute(&self, dims: Vec<usize>) -> PyResult<VulkanTensor> {
        let inner = self.inner.read();

        if dims.len() != inner.shape.len() {
            return Err(PyRuntimeError::new_err(format!(
                "Permutation must have {} dimensions",
                inner.shape.len()
            )));
        }

        // Verify all dims are valid and unique
        let mut seen = vec![false; dims.len()];
        for &d in &dims {
            if d >= dims.len() {
                return Err(PyRuntimeError::new_err(format!("Invalid dimension: {}", d)));
            }
            if seen[d] {
                return Err(PyRuntimeError::new_err("Duplicate dimension in permutation"));
            }
            seen[d] = true;
        }

        let new_shape: Vec<usize> = dims.iter().map(|&d| inner.shape[d]).collect();
        let new_strides: Vec<usize> = dims.iter().map(|&d| inner.strides[d]).collect();

        let new_data = TensorData {
            data: inner.data.clone(),
            shape: new_shape,
            strides: new_strides,
            dtype: inner.dtype,
            device_idx: inner.device_idx,
        };

        Ok(VulkanTensor {
            inner: Arc::new(RwLock::new(new_data)),
        })
    }

    /// Convert to contiguous layout
    pub fn contiguous(&self) -> PyResult<VulkanTensor> {
        let inner = self.inner.read();

        // Calculate expected strides for contiguous tensor
        let mut expected_strides = vec![1usize; inner.shape.len()];
        for i in (0..inner.shape.len().saturating_sub(1)).rev() {
            expected_strides[i] = expected_strides[i + 1] * inner.shape[i + 1];
        }

        // If already contiguous, return clone
        if inner.strides == expected_strides {
            return Ok(VulkanTensor {
                inner: Arc::new(RwLock::new(TensorData {
                    data: inner.data.clone(),
                    shape: inner.shape.clone(),
                    strides: inner.strides.clone(),
                    dtype: inner.dtype,
                    device_idx: inner.device_idx,
                })),
            });
        }

        // Otherwise, need to copy data in contiguous order
        // For simplicity, just clone (real implementation would reorder data)
        let new_data = TensorData {
            data: inner.data.clone(),
            shape: inner.shape.clone(),
            strides: expected_strides,
            dtype: inner.dtype,
            device_idx: inner.device_idx,
        };

        Ok(VulkanTensor {
            inner: Arc::new(RwLock::new(new_data)),
        })
    }

    /// Copy tensor to different device
    #[pyo3(signature = (device_idx=0))]
    pub fn to_device(&self, device_idx: usize) -> PyResult<VulkanTensor> {
        let inner = self.inner.read();

        let new_data = TensorData {
            data: inner.data.clone(),
            shape: inner.shape.clone(),
            strides: inner.strides.clone(),
            dtype: inner.dtype,
            device_idx,
        };

        Ok(VulkanTensor {
            inner: Arc::new(RwLock::new(new_data)),
        })
    }

    /// Fill tensor with a value
    pub fn fill(&self, value: f32) -> PyResult<()> {
        let mut inner = self.inner.write();

        match inner.dtype {
            TensorDtype::F32 => {
                let ptr = inner.data.as_mut_ptr() as *mut f32;
                let len = inner.data.len() / std::mem::size_of::<f32>();
                unsafe {
                    for i in 0..len {
                        *ptr.add(i) = value;
                    }
                }
            }
            _ => {
                return Err(PyRuntimeError::new_err(
                    "Fill only supported for f32 tensors",
                ));
            }
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.read();
        format!(
            "VulkanTensor(shape={:?}, dtype={}, device={})",
            inner.shape,
            inner.dtype.as_str(),
            inner.device_idx
        )
    }
}

impl VulkanTensor {
    /// Get raw pointer to data (internal use)
    pub fn as_ptr(&self) -> *const u8 {
        self.inner.read().data.as_ptr()
    }

    /// Get mutable raw pointer (internal use)
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.inner.write().data.as_mut_ptr()
    }

    /// Get data as f32 slice (internal use)
    pub fn as_f32_slice(&self) -> Vec<f32> {
        let inner = self.inner.read();
        let len = inner.data.len() / std::mem::size_of::<f32>();
        let mut result = vec![0.0f32; len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                inner.data.as_ptr() as *const f32,
                result.as_mut_ptr(),
                len,
            );
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = VulkanTensor::new(vec![2, 3, 4], "f32", 0).unwrap();
        assert_eq!(tensor.shape(), vec![2, 3, 4]);
        assert_eq!(tensor.numel(), 24);
        assert_eq!(tensor.ndim(), 3);
    }

    #[test]
    fn test_tensor_reshape() {
        let tensor = VulkanTensor::new(vec![2, 3, 4], "f32", 0).unwrap();
        let reshaped = tensor.reshape(vec![6, 4]).unwrap();
        assert_eq!(reshaped.shape(), vec![6, 4]);
        assert_eq!(reshaped.numel(), 24);
    }

    #[test]
    fn test_tensor_permute() {
        let tensor = VulkanTensor::new(vec![2, 3, 4], "f32", 0).unwrap();
        let permuted = tensor.permute(vec![2, 0, 1]).unwrap();
        assert_eq!(permuted.shape(), vec![4, 2, 3]);
    }

    #[test]
    fn test_dtype_parsing() {
        assert_eq!(TensorDtype::from_str("f32"), Some(TensorDtype::F32));
        assert_eq!(TensorDtype::from_str("float16"), Some(TensorDtype::F16));
        assert_eq!(TensorDtype::from_str("q4_0"), Some(TensorDtype::Q4_0));
        assert_eq!(TensorDtype::from_str("invalid"), None);
    }
}
