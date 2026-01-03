//! Compute Graph Execution
//!
//! This module provides compute graph construction and execution wrapping ggml graphs.

#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::Arc;
use parking_lot::RwLock;

use crate::tensor::VulkanTensor;
use crate::backend::VulkanBackend;

/// Graph operation types
#[derive(Clone, Debug)]
pub enum GraphOp {
    Add,
    Mul,
    MatMul,
    Softmax,
    LayerNorm,
    RMSNorm,
    RoPE,
    Attention,
    FlashAttention,
    PagedAttention,
    Reshape,
    Permute,
    Concat,
    Split,
    Custom(String),
}

impl GraphOp {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "add" => Some(GraphOp::Add),
            "mul" => Some(GraphOp::Mul),
            "matmul" | "mm" => Some(GraphOp::MatMul),
            "softmax" => Some(GraphOp::Softmax),
            "layernorm" | "layer_norm" => Some(GraphOp::LayerNorm),
            "rmsnorm" | "rms_norm" => Some(GraphOp::RMSNorm),
            "rope" => Some(GraphOp::RoPE),
            "attention" => Some(GraphOp::Attention),
            "flash_attention" => Some(GraphOp::FlashAttention),
            "paged_attention" => Some(GraphOp::PagedAttention),
            "reshape" => Some(GraphOp::Reshape),
            "permute" => Some(GraphOp::Permute),
            "concat" => Some(GraphOp::Concat),
            "split" => Some(GraphOp::Split),
            _ => Some(GraphOp::Custom(s.to_string())),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            GraphOp::Add => "add",
            GraphOp::Mul => "mul",
            GraphOp::MatMul => "matmul",
            GraphOp::Softmax => "softmax",
            GraphOp::LayerNorm => "layer_norm",
            GraphOp::RMSNorm => "rms_norm",
            GraphOp::RoPE => "rope",
            GraphOp::Attention => "attention",
            GraphOp::FlashAttention => "flash_attention",
            GraphOp::PagedAttention => "paged_attention",
            GraphOp::Reshape => "reshape",
            GraphOp::Permute => "permute",
            GraphOp::Concat => "concat",
            GraphOp::Split => "split",
            GraphOp::Custom(name) => name,
        }
    }
}

/// Graph node
#[derive(Clone)]
struct GraphNode {
    op: GraphOp,
    inputs: Vec<usize>,
    output: usize,
    params: Vec<i64>,
}

/// Tensor metadata for graph (simplified, not storing actual tensor)
#[derive(Clone)]
struct TensorMeta {
    shape: Vec<usize>,
    dtype: String,
    device_idx: usize,
}

/// Internal graph data
struct GraphData {
    nodes: Vec<GraphNode>,
    tensors: Vec<TensorMeta>,
    is_built: bool,
}

/// Vulkan compute graph
#[pyclass]
pub struct VulkanGraph {
    inner: Arc<RwLock<GraphData>>,
    #[pyo3(get)]
    device_idx: usize,
}

#[pymethods]
impl VulkanGraph {
    /// Create a new compute graph
    #[new]
    #[pyo3(signature = (device_idx=0))]
    pub fn new(device_idx: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(GraphData {
                nodes: Vec::new(),
                tensors: Vec::new(),
                is_built: false,
            })),
            device_idx,
        }
    }

    /// Add a tensor to the graph, returns tensor index
    #[pyo3(signature = (shape, dtype="f32"))]
    pub fn add_tensor(&self, shape: Vec<usize>, dtype: &str) -> usize {
        let mut inner = self.inner.write();
        let idx = inner.tensors.len();
        inner.tensors.push(TensorMeta {
            shape,
            dtype: dtype.to_string(),
            device_idx: self.device_idx,
        });
        idx
    }

    /// Add an operation node to the graph
    #[pyo3(signature = (op, inputs, output, params=vec![]))]
    pub fn add_node(
        &self,
        op: &str,
        inputs: Vec<usize>,
        output: usize,
        params: Vec<i64>,
    ) -> PyResult<usize> {
        let graph_op = GraphOp::from_str(op)
            .ok_or_else(|| PyRuntimeError::new_err(format!("Unknown operation: {}", op)))?;

        let mut inner = self.inner.write();

        // Validate tensor indices
        for &idx in &inputs {
            if idx >= inner.tensors.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "Input tensor index {} out of range",
                    idx
                )));
            }
        }
        if output >= inner.tensors.len() {
            return Err(PyRuntimeError::new_err(format!(
                "Output tensor index {} out of range",
                output
            )));
        }

        let node = GraphNode {
            op: graph_op,
            inputs,
            output,
            params,
        };

        let node_idx = inner.nodes.len();
        inner.nodes.push(node);

        Ok(node_idx)
    }

    /// Build the graph (prepare for execution)
    pub fn build(&self) -> PyResult<()> {
        let mut inner = self.inner.write();

        if inner.nodes.is_empty() {
            return Err(PyRuntimeError::new_err("Cannot build empty graph"));
        }

        // In real implementation, this would:
        // 1. Validate graph structure
        // 2. Optimize node ordering
        // 3. Allocate intermediate buffers
        // 4. Compile Vulkan command buffers

        inner.is_built = true;
        Ok(())
    }

    /// Execute the graph on the specified backend
    pub fn compute(&self, backend: &VulkanBackend) -> PyResult<()> {
        let inner = self.inner.read();

        if !inner.is_built {
            return Err(PyRuntimeError::new_err("Graph not built. Call build() first."));
        }

        if !backend.initialized() {
            return Err(PyRuntimeError::new_err("Backend not initialized"));
        }

        // In real implementation, this would:
        // 1. Submit Vulkan command buffer
        // 2. Execute all graph operations
        // 3. Handle synchronization

        log::debug!("Executing graph with {} nodes", inner.nodes.len());

        for (i, node) in inner.nodes.iter().enumerate() {
            log::trace!(
                "Node {}: {} inputs={:?} output={}",
                i,
                node.op.as_str(),
                node.inputs,
                node.output
            );
        }

        Ok(())
    }

    /// Get number of nodes in graph
    pub fn num_nodes(&self) -> usize {
        self.inner.read().nodes.len()
    }

    /// Get number of tensors in graph
    pub fn num_tensors(&self) -> usize {
        self.inner.read().tensors.len()
    }

    /// Check if graph is built
    pub fn is_built(&self) -> bool {
        self.inner.read().is_built
    }

    /// Reset the graph (clear all nodes and tensors)
    pub fn reset(&self) {
        let mut inner = self.inner.write();
        inner.nodes.clear();
        inner.tensors.clear();
        inner.is_built = false;
    }

    /// Get tensor by index
    pub fn get_tensor(&self, idx: usize) -> PyResult<VulkanTensor> {
        let inner = self.inner.read();
        if idx >= inner.tensors.len() {
            return Err(PyRuntimeError::new_err(format!(
                "Tensor index {} out of range (0-{})",
                idx,
                inner.tensors.len().saturating_sub(1)
            )));
        }
        // Create a new tensor with the same metadata
        let meta = &inner.tensors[idx];
        VulkanTensor::new(
            meta.shape.clone(),
            &meta.dtype,
            meta.device_idx,
        )
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.read();
        format!(
            "VulkanGraph(nodes={}, tensors={}, built={})",
            inner.nodes.len(),
            inner.tensors.len(),
            inner.is_built
        )
    }
}

impl VulkanGraph {
    /// Get nodes for internal use
    pub fn nodes(&self) -> Vec<(String, Vec<usize>, usize)> {
        let inner = self.inner.read();
        inner
            .nodes
            .iter()
            .map(|n| (n.op.as_str().to_string(), n.inputs.clone(), n.output))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = VulkanGraph::new(0);
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_tensors(), 0);
        assert!(!graph.is_built());
    }

    #[test]
    fn test_graph_add_tensor() {
        let graph = VulkanGraph::new(0);
        let idx = graph.add_tensor(vec![2, 3], "f32");
        assert_eq!(idx, 0);
        assert_eq!(graph.num_tensors(), 1);
    }

    #[test]
    fn test_graph_add_node() {
        let graph = VulkanGraph::new(0);

        let idx1 = graph.add_tensor(vec![2, 3], "f32");
        let idx2 = graph.add_tensor(vec![2, 3], "f32");
        let idx3 = graph.add_tensor(vec![2, 3], "f32");

        let node_idx = graph.add_node("add", vec![idx1, idx2], idx3, vec![]).unwrap();
        assert_eq!(node_idx, 0);
        assert_eq!(graph.num_nodes(), 1);
    }

    #[test]
    fn test_graph_build() {
        let graph = VulkanGraph::new(0);

        let idx1 = graph.add_tensor(vec![2, 3], "f32");
        let idx2 = graph.add_tensor(vec![2, 3], "f32");
        let idx3 = graph.add_tensor(vec![2, 3], "f32");

        graph.add_node("add", vec![idx1, idx2], idx3, vec![]).unwrap();
        graph.build().unwrap();

        assert!(graph.is_built());
    }

    #[test]
    fn test_graph_op_parsing() {
        assert!(matches!(GraphOp::from_str("add"), Some(GraphOp::Add)));
        assert!(matches!(GraphOp::from_str("matmul"), Some(GraphOp::MatMul)));
        assert!(matches!(GraphOp::from_str("flash_attention"), Some(GraphOp::FlashAttention)));
    }
}
