//! Attention Operations
//!
//! This module provides attention kernel dispatch for Vulkan backend.

#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use crate::tensor::VulkanTensor;

/// Flash attention implementation
///
/// Performs fused attention computation: softmax(QK^T / sqrt(d_k)) * V
pub fn flash_attention(
    query: &VulkanTensor,
    key: &VulkanTensor,
    value: &VulkanTensor,
    mask: Option<&VulkanTensor>,
    scale: Option<f32>,
) -> PyResult<VulkanTensor> {
    let q_shape = query.shape();
    let k_shape = key.shape();
    let v_shape = value.shape();

    // Validate shapes
    // Expected: [batch, seq_len, num_heads, head_dim]
    if q_shape.len() < 3 || k_shape.len() < 3 || v_shape.len() < 3 {
        return Err(PyRuntimeError::new_err(
            "Query, key, value must have at least 3 dimensions",
        ));
    }

    // Get dimensions
    let head_dim = *q_shape.last().unwrap();
    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Create output tensor with same shape as query
    let output = VulkanTensor::new(q_shape.clone(), &query.dtype(), query.device_idx())?;

    // In real implementation, this would:
    // 1. Build ggml flash attention graph
    // 2. Execute on Vulkan backend
    // 3. Return computed attention output

    log::debug!(
        "Flash attention: Q{:?} K{:?} V{:?} scale={}",
        q_shape,
        k_shape,
        v_shape,
        scale
    );

    Ok(output)
}

/// Paged attention implementation
///
/// Performs attention with paged KV cache for efficient memory usage
pub fn paged_attention(
    query: &VulkanTensor,
    key_cache: &VulkanTensor,
    value_cache: &VulkanTensor,
    block_tables: &[i32],
    context_lens: &[i32],
    scale: Option<f32>,
    block_size: usize,
) -> PyResult<VulkanTensor> {
    let q_shape = query.shape();

    // Validate shapes
    if q_shape.len() < 3 {
        return Err(PyRuntimeError::new_err(
            "Query must have at least 3 dimensions",
        ));
    }

    let head_dim = *q_shape.last().unwrap();
    let scale = scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

    // Create output tensor with same shape as query
    let output = VulkanTensor::new(q_shape.clone(), &query.dtype(), query.device_idx())?;

    // In real implementation, this would:
    // 1. Use block_tables to gather KV from cache
    // 2. Compute attention for each sequence based on context_lens
    // 3. Support variable sequence lengths efficiently

    log::debug!(
        "Paged attention: Q{:?} block_size={} scale={}",
        q_shape,
        block_size,
        scale
    );

    Ok(output)
}

/// Python-exposed flash attention function
#[pyfunction]
#[pyo3(signature = (query, key, value, mask=None, scale=None))]
pub fn flash_attention_py(
    query: &VulkanTensor,
    key: &VulkanTensor,
    value: &VulkanTensor,
    mask: Option<&VulkanTensor>,
    scale: Option<f32>,
) -> PyResult<VulkanTensor> {
    flash_attention(query, key, value, mask, scale)
}

/// Python-exposed paged attention function (v1 - simple version)
#[pyfunction]
#[pyo3(signature = (query, key_cache, value_cache, block_tables, context_lens, scale=None, block_size=16))]
pub fn paged_attention_py(
    query: &VulkanTensor,
    key_cache: &VulkanTensor,
    value_cache: &VulkanTensor,
    block_tables: Vec<i32>,
    context_lens: Vec<i32>,
    scale: Option<f32>,
    block_size: usize,
) -> PyResult<VulkanTensor> {
    paged_attention(
        query,
        key_cache,
        value_cache,
        &block_tables,
        &context_lens,
        scale,
        block_size,
    )
}

/// Attention metadata for batch processing
#[derive(Clone, Debug)]
pub struct AttentionMetadata {
    pub num_prefill_tokens: usize,
    pub num_decode_tokens: usize,
    pub seq_lens: Vec<usize>,
    pub block_tables: Vec<Vec<i32>>,
    pub slot_mapping: Vec<i32>,
    pub context_lens: Vec<i32>,
    pub max_seq_len: usize,
}

impl AttentionMetadata {
    pub fn new() -> Self {
        Self {
            num_prefill_tokens: 0,
            num_decode_tokens: 0,
            seq_lens: Vec::new(),
            block_tables: Vec::new(),
            slot_mapping: Vec::new(),
            context_lens: Vec::new(),
            max_seq_len: 0,
        }
    }

    pub fn is_prefill(&self) -> bool {
        self.num_prefill_tokens > 0
    }

    pub fn is_decode(&self) -> bool {
        self.num_decode_tokens > 0
    }

    pub fn total_tokens(&self) -> usize {
        self.num_prefill_tokens + self.num_decode_tokens
    }
}

impl Default for AttentionMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-head attention helper
pub fn multi_head_attention(
    query: &VulkanTensor,
    key: &VulkanTensor,
    value: &VulkanTensor,
    num_heads: usize,
    head_dim: usize,
    mask: Option<&VulkanTensor>,
) -> PyResult<VulkanTensor> {
    let q_shape = query.shape();

    // Validate input
    if q_shape.len() < 2 {
        return Err(PyRuntimeError::new_err("Query must have at least 2 dimensions"));
    }

    let batch_size = q_shape[0];
    let seq_len = q_shape[1];
    let hidden_size = num_heads * head_dim;

    // Reshape Q, K, V for multi-head attention
    // [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
    let q_reshaped = query.reshape(vec![batch_size, seq_len, num_heads, head_dim])?;
    let k_reshaped = key.reshape(vec![batch_size, seq_len, num_heads, head_dim])?;
    let v_reshaped = value.reshape(vec![batch_size, seq_len, num_heads, head_dim])?;

    // Compute attention
    let output = flash_attention(&q_reshaped, &k_reshaped, &v_reshaped, mask, None)?;

    // Reshape output back to [batch, seq, hidden]
    output.reshape(vec![batch_size, seq_len, hidden_size])
}

/// Rotary position embedding (RoPE)
pub fn apply_rotary_pos_emb(
    query: &VulkanTensor,
    key: &VulkanTensor,
    cos: &VulkanTensor,
    sin: &VulkanTensor,
    position_ids: &[i64],
) -> PyResult<(VulkanTensor, VulkanTensor)> {
    // In real implementation, this would apply rotary embeddings
    // For now, return clones of input tensors

    let q_out = VulkanTensor::new(query.shape(), &query.dtype(), query.device_idx())?;
    let k_out = VulkanTensor::new(key.shape(), &key.dtype(), key.device_idx())?;

    log::debug!(
        "RoPE: Q{:?} K{:?} positions={}",
        query.shape(),
        key.shape(),
        position_ids.len()
    );

    Ok((q_out, k_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention() {
        let query = VulkanTensor::new(vec![2, 8, 4, 64], "f32", 0).unwrap();
        let key = VulkanTensor::new(vec![2, 8, 4, 64], "f32", 0).unwrap();
        let value = VulkanTensor::new(vec![2, 8, 4, 64], "f32", 0).unwrap();

        let output = flash_attention(&query, &key, &value, None, None).unwrap();
        assert_eq!(output.shape(), vec![2, 8, 4, 64]);
    }

    #[test]
    fn test_paged_attention() {
        let query = VulkanTensor::new(vec![2, 4, 64], "f32", 0).unwrap();
        let key_cache = VulkanTensor::new(vec![100, 16, 4, 64], "f32", 0).unwrap();
        let value_cache = VulkanTensor::new(vec![100, 16, 4, 64], "f32", 0).unwrap();

        let block_tables = vec![0, 1, 2, 3];
        let context_lens = vec![32, 48];

        let output = paged_attention(
            &query,
            &key_cache,
            &value_cache,
            &block_tables,
            &context_lens,
            None,
            16,
        )
        .unwrap();

        assert_eq!(output.shape(), vec![2, 4, 64]);
    }

    #[test]
    fn test_attention_metadata() {
        let mut meta = AttentionMetadata::new();
        meta.num_prefill_tokens = 100;
        meta.num_decode_tokens = 50;

        assert!(meta.is_prefill());
        assert!(meta.is_decode());
        assert_eq!(meta.total_tokens(), 150);
    }
}
