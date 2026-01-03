//! Distributed Communication
//!
//! This module provides multi-GPU communication primitives for distributed inference.

#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::Arc;
use parking_lot::RwLock;

use crate::tensor::VulkanTensor;

/// Reduction operation types
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    Sum,
    Prod,
    Min,
    Max,
    Avg,
}

impl ReduceOp {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sum" => Some(ReduceOp::Sum),
            "prod" | "product" => Some(ReduceOp::Prod),
            "min" => Some(ReduceOp::Min),
            "max" => Some(ReduceOp::Max),
            "avg" | "average" | "mean" => Some(ReduceOp::Avg),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Prod => "prod",
            ReduceOp::Min => "min",
            ReduceOp::Max => "max",
            ReduceOp::Avg => "avg",
        }
    }
}

/// Communicator state
struct CommunicatorData {
    world_size: usize,
    rank: usize,
    local_rank: usize,
    device_idx: usize,
    is_initialized: bool,
}

/// Vulkan communicator for multi-GPU operations
#[pyclass]
pub struct VulkanCommunicator {
    inner: Arc<RwLock<CommunicatorData>>,
}

#[pymethods]
impl VulkanCommunicator {
    /// Create a new communicator
    #[new]
    #[pyo3(signature = (world_size, rank, local_rank=None, device_idx=None))]
    pub fn new(
        world_size: usize,
        rank: usize,
        local_rank: Option<usize>,
        device_idx: Option<usize>,
    ) -> PyResult<Self> {
        if rank >= world_size {
            return Err(PyRuntimeError::new_err(format!(
                "Rank {} must be < world_size {}",
                rank, world_size
            )));
        }

        let local_rank = local_rank.unwrap_or(rank);
        let device_idx = device_idx.unwrap_or(local_rank);

        let data = CommunicatorData {
            world_size,
            rank,
            local_rank,
            device_idx,
            is_initialized: true,
        };

        log::info!(
            "Created VulkanCommunicator: rank={}/{} device={}",
            rank,
            world_size,
            device_idx
        );

        Ok(Self {
            inner: Arc::new(RwLock::new(data)),
        })
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.inner.read().world_size
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.inner.read().rank
    }

    /// Get local rank
    pub fn local_rank(&self) -> usize {
        self.inner.read().local_rank
    }

    /// Get device index
    pub fn device_idx(&self) -> usize {
        self.inner.read().device_idx
    }

    /// All-reduce operation
    #[pyo3(signature = (tensor, op="sum"))]
    pub fn all_reduce(&self, tensor: &VulkanTensor, op: &str) -> PyResult<VulkanTensor> {
        let _inner = self.inner.read();

        let reduce_op = ReduceOp::from_str(op)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Unknown reduce op: {}", op)))?;

        // In real implementation, this would:
        // 1. Copy tensor to shared memory or use Vulkan external memory
        // 2. Synchronize with other ranks via semaphores
        // 3. Perform reduction operation
        // 4. Return reduced tensor

        log::debug!(
            "All-reduce: {:?} op={} world_size={}",
            tensor.shape(),
            reduce_op.as_str(),
            _inner.world_size
        );

        // Return a copy for now (real implementation would modify in place or return result)
        let result = VulkanTensor::new(tensor.shape(), &tensor.dtype(), tensor.device_idx())?;
        Ok(result)
    }

    /// Broadcast tensor from source rank
    #[pyo3(signature = (tensor, src=0))]
    pub fn broadcast(&self, tensor: &VulkanTensor, src: usize) -> PyResult<VulkanTensor> {
        let inner = self.inner.read();

        if src >= inner.world_size {
            return Err(PyRuntimeError::new_err(format!(
                "Source rank {} >= world_size {}",
                src, inner.world_size
            )));
        }

        // In real implementation, this would broadcast from src to all ranks

        log::debug!(
            "Broadcast: {:?} src={} world_size={}",
            tensor.shape(),
            src,
            inner.world_size
        );

        let result = VulkanTensor::new(tensor.shape(), &tensor.dtype(), tensor.device_idx())?;
        Ok(result)
    }

    /// Send tensor to destination rank
    pub fn send(&self, tensor: &VulkanTensor, dst: usize) -> PyResult<()> {
        let inner = self.inner.read();

        if dst >= inner.world_size {
            return Err(PyRuntimeError::new_err(format!(
                "Destination rank {} >= world_size {}",
                dst, inner.world_size
            )));
        }

        // In real implementation, this would send tensor to dst rank

        log::debug!(
            "Send: {:?} from {} to {}",
            tensor.shape(),
            inner.rank,
            dst
        );

        Ok(())
    }

    /// Receive tensor from source rank
    pub fn recv(&self, src: usize, shape: Vec<usize>, dtype: &str) -> PyResult<VulkanTensor> {
        let inner = self.inner.read();

        if src >= inner.world_size {
            return Err(PyRuntimeError::new_err(format!(
                "Source rank {} >= world_size {}",
                src, inner.world_size
            )));
        }

        // In real implementation, this would receive tensor from src rank

        log::debug!("Recv: {:?} from {} to {}", shape, src, inner.rank);

        let result = VulkanTensor::new(shape, dtype, inner.device_idx)?;
        Ok(result)
    }

    /// All-gather operation
    pub fn all_gather(&self, tensor: &VulkanTensor) -> PyResult<VulkanTensor> {
        let inner = self.inner.read();

        // Output shape has world_size as first dimension
        let mut output_shape = vec![inner.world_size];
        output_shape.extend(tensor.shape());

        log::debug!(
            "All-gather: {:?} -> {:?}",
            tensor.shape(),
            output_shape
        );

        let result = VulkanTensor::new(output_shape, &tensor.dtype(), tensor.device_idx())?;
        Ok(result)
    }

    /// Reduce-scatter operation
    pub fn reduce_scatter(&self, tensor: &VulkanTensor, op: &str) -> PyResult<VulkanTensor> {
        let inner = self.inner.read();

        let reduce_op = ReduceOp::from_str(op)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Unknown reduce op: {}", op)))?;

        // Output shape has first dimension divided by world_size
        let mut output_shape = tensor.shape();
        if !output_shape.is_empty() {
            output_shape[0] /= inner.world_size;
        }

        log::debug!(
            "Reduce-scatter: {:?} -> {:?} op={}",
            tensor.shape(),
            output_shape,
            reduce_op.as_str()
        );

        let result = VulkanTensor::new(output_shape, &tensor.dtype(), tensor.device_idx())?;
        Ok(result)
    }

    /// Barrier synchronization
    pub fn barrier(&self) -> PyResult<()> {
        let inner = self.inner.read();

        // In real implementation, this would synchronize all ranks

        log::debug!("Barrier: rank={}/{}", inner.rank, inner.world_size);

        Ok(())
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.read();
        format!(
            "VulkanCommunicator(rank={}/{}, device={})",
            inner.rank, inner.world_size, inner.device_idx
        )
    }
}

/// Create communicator group
#[pyfunction]
#[pyo3(signature = (world_size, rank, local_rank=None, device_idx=None))]
pub fn create_communicator(
    world_size: usize,
    rank: usize,
    local_rank: Option<usize>,
    device_idx: Option<usize>,
) -> PyResult<VulkanCommunicator> {
    VulkanCommunicator::new(world_size, rank, local_rank, device_idx)
}

/// Initialize distributed environment
#[pyfunction]
pub fn init_distributed() -> PyResult<()> {
    // In real implementation, this would initialize distributed backend

    log::info!("Initialized distributed environment");

    Ok(())
}

/// Check if distributed is initialized
#[pyfunction]
pub fn is_distributed_initialized() -> bool {
    // In real implementation, this would check if distributed is set up
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_communicator_creation() {
        let comm = VulkanCommunicator::new(4, 0, None, None).unwrap();
        assert_eq!(comm.world_size(), 4);
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.local_rank(), 0);
    }

    #[test]
    fn test_invalid_rank() {
        let result = VulkanCommunicator::new(4, 5, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_op_parsing() {
        assert!(matches!(ReduceOp::from_str("sum"), Some(ReduceOp::Sum)));
        assert!(matches!(ReduceOp::from_str("max"), Some(ReduceOp::Max)));
        assert!(matches!(ReduceOp::from_str("avg"), Some(ReduceOp::Avg)));
        assert!(ReduceOp::from_str("invalid").is_none());
    }

    #[test]
    fn test_all_reduce() {
        let comm = VulkanCommunicator::new(2, 0, None, None).unwrap();
        let tensor = VulkanTensor::new(vec![4, 4], "f32", 0).unwrap();

        let result = comm.all_reduce(&tensor, "sum").unwrap();
        assert_eq!(result.shape(), vec![4, 4]);
    }

    #[test]
    fn test_broadcast() {
        let comm = VulkanCommunicator::new(2, 0, None, None).unwrap();
        let tensor = VulkanTensor::new(vec![4, 4], "f32", 0).unwrap();

        let result = comm.broadcast(&tensor, 0).unwrap();
        assert_eq!(result.shape(), vec![4, 4]);
    }

    #[test]
    fn test_all_gather() {
        let comm = VulkanCommunicator::new(2, 0, None, None).unwrap();
        let tensor = VulkanTensor::new(vec![4, 4], "f32", 0).unwrap();

        let result = comm.all_gather(&tensor).unwrap();
        assert_eq!(result.shape(), vec![2, 4, 4]);
    }
}
