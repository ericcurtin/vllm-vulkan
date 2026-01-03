//! Paged KV Cache Management
//!
//! This module provides paged key-value cache for efficient memory management
//! in attention computation.

#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

use crate::tensor::VulkanTensor;

/// A single cache block (metadata only, actual tensors managed separately)
struct CacheBlock {
    is_allocated: bool,
    layer_idx: usize,
}

/// Internal cache data
struct CacheData {
    num_blocks: usize,
    block_size: usize,
    num_heads: usize,
    head_dim: usize,
    num_layers: usize,
    blocks: Vec<CacheBlock>,
    free_blocks: Vec<usize>,
    sequence_block_tables: HashMap<u64, Vec<usize>>,
}

/// Paged KV cache for attention
#[pyclass]
pub struct PagedKVCache {
    inner: Arc<RwLock<CacheData>>,
    #[pyo3(get)]
    device_idx: usize,
    #[pyo3(get)]
    dtype: String,
}

#[pymethods]
impl PagedKVCache {
    /// Create a new paged KV cache
    #[new]
    #[pyo3(signature = (num_blocks, block_size, num_heads, head_dim, num_layers, device_idx=0, dtype="f16"))]
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
        device_idx: usize,
        dtype: &str,
    ) -> PyResult<Self> {
        if num_blocks == 0 {
            return Err(PyRuntimeError::new_err("num_blocks must be > 0"));
        }
        if block_size == 0 {
            return Err(PyRuntimeError::new_err("block_size must be > 0"));
        }

        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_blocks = Vec::with_capacity(num_blocks);

        // Pre-allocate all cache block metadata
        for i in 0..num_blocks {
            blocks.push(CacheBlock {
                is_allocated: false,
                layer_idx: 0,
            });
            free_blocks.push(i);
        }

        let data = CacheData {
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            num_layers,
            blocks,
            free_blocks,
            sequence_block_tables: HashMap::new(),
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(data)),
            device_idx,
            dtype: dtype.to_string(),
        })
    }

    /// Get number of free blocks
    pub fn num_free_blocks(&self) -> usize {
        self.inner.read().free_blocks.len()
    }

    /// Get total number of blocks
    pub fn num_total_blocks(&self) -> usize {
        self.inner.read().num_blocks
    }

    /// Get number of used blocks
    pub fn num_used_blocks(&self) -> usize {
        let inner = self.inner.read();
        inner.num_blocks - inner.free_blocks.len()
    }

    /// Get block size
    pub fn block_size(&self) -> usize {
        self.inner.read().block_size
    }

    /// Allocate blocks for a sequence
    pub fn allocate_blocks(&self, seq_id: u64, num_blocks: usize) -> PyResult<Vec<usize>> {
        let mut inner = self.inner.write();

        if num_blocks > inner.free_blocks.len() {
            return Err(PyRuntimeError::new_err(format!(
                "Cannot allocate {} blocks, only {} free",
                num_blocks,
                inner.free_blocks.len()
            )));
        }

        // Allocate blocks from free list
        let allocated: Vec<usize> = inner.free_blocks.drain(..num_blocks).collect();

        // Mark blocks as allocated
        for &block_idx in &allocated {
            inner.blocks[block_idx].is_allocated = true;
        }

        // Store in sequence block table
        inner.sequence_block_tables.insert(seq_id, allocated.clone());

        Ok(allocated)
    }

    /// Free blocks for a sequence
    pub fn free_blocks(&self, seq_id: u64) -> PyResult<()> {
        let mut inner = self.inner.write();

        let blocks = inner.sequence_block_tables.remove(&seq_id);
        if let Some(blocks) = blocks {
            for block_idx in blocks {
                inner.blocks[block_idx].is_allocated = false;
                inner.free_blocks.push(block_idx);
            }
        }

        Ok(())
    }

    /// Get block table for a sequence
    pub fn get_block_table(&self, seq_id: u64) -> PyResult<Vec<usize>> {
        let inner = self.inner.read();
        inner
            .sequence_block_tables
            .get(&seq_id)
            .cloned()
            .ok_or_else(|| PyRuntimeError::new_err(format!("Sequence {} not found", seq_id)))
    }

    /// Swap blocks between two block indices
    pub fn swap_blocks(&self, src_block: usize, dst_block: usize) -> PyResult<()> {
        let mut inner = self.inner.write();

        if src_block >= inner.num_blocks || dst_block >= inner.num_blocks {
            return Err(PyRuntimeError::new_err("Block index out of range"));
        }

        // Swap the blocks
        inner.blocks.swap(src_block, dst_block);

        // Update sequence block tables
        for (_, block_table) in inner.sequence_block_tables.iter_mut() {
            for block_idx in block_table.iter_mut() {
                if *block_idx == src_block {
                    *block_idx = dst_block;
                } else if *block_idx == dst_block {
                    *block_idx = src_block;
                }
            }
        }

        Ok(())
    }

    /// Copy blocks (for forking sequences)
    pub fn copy_blocks(&self, src_block: usize, dst_block: usize) -> PyResult<()> {
        let inner = self.inner.read();

        if src_block >= inner.num_blocks || dst_block >= inner.num_blocks {
            return Err(PyRuntimeError::new_err("Block index out of range"));
        }

        // In real implementation, this would copy tensor data
        log::debug!("Copying block {} to {}", src_block, dst_block);

        Ok(())
    }

    /// Append a sequence's block table with a new block
    pub fn append_block(&self, seq_id: u64) -> PyResult<usize> {
        let mut inner = self.inner.write();

        if inner.free_blocks.is_empty() {
            return Err(PyRuntimeError::new_err("No free blocks available"));
        }

        let new_block = inner.free_blocks.pop().unwrap();
        inner.blocks[new_block].is_allocated = true;

        if let Some(block_table) = inner.sequence_block_tables.get_mut(&seq_id) {
            block_table.push(new_block);
        } else {
            inner.sequence_block_tables.insert(seq_id, vec![new_block]);
        }

        Ok(new_block)
    }

    /// Get cache memory usage in bytes
    pub fn memory_usage(&self) -> u64 {
        let inner = self.inner.read();
        let block_memory = inner.block_size * inner.num_heads * inner.head_dim * 2; // 2 bytes for f16
        let total_memory = block_memory * inner.num_layers * 2; // K and V
        (inner.num_used_blocks() * total_memory) as u64
    }

    /// Get total cache capacity in bytes
    pub fn total_capacity(&self) -> u64 {
        let inner = self.inner.read();
        let block_memory = inner.block_size * inner.num_heads * inner.head_dim * 2;
        let total_memory = block_memory * inner.num_layers * 2;
        (inner.num_blocks * total_memory) as u64
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.read();
        format!(
            "PagedKVCache(blocks={}/{}, block_size={}, dtype={}, device={})",
            inner.num_blocks - inner.free_blocks.len(),
            inner.num_blocks,
            inner.block_size,
            self.dtype,
            self.device_idx
        )
    }
}

impl CacheData {
    fn num_used_blocks(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }
}

/// Reshape and cache KV tensors
#[pyfunction]
#[pyo3(signature = (key, value, key_cache, value_cache, slot_mapping))]
pub fn reshape_and_cache_py(
    key: &VulkanTensor,
    value: &VulkanTensor,
    key_cache: &VulkanTensor,
    value_cache: &VulkanTensor,
    slot_mapping: Vec<i64>,
) -> PyResult<()> {
    // In real implementation, this would:
    // 1. Reshape key/value tensors
    // 2. Store them in the cache at the specified slots

    log::debug!(
        "Reshape and cache: K{:?} V{:?} slots={}",
        key.shape(),
        value.shape(),
        slot_mapping.len()
    );

    Ok(())
}

/// Copy blocks between caches
#[pyfunction]
#[pyo3(signature = (block_mapping, key_caches, value_caches))]
pub fn copy_blocks_py(
    block_mapping: Vec<(i64, i64)>,
    key_caches: Vec<PyRef<VulkanTensor>>,
    value_caches: Vec<PyRef<VulkanTensor>>,
) -> PyResult<()> {
    // In real implementation, this would copy block data

    log::debug!("Copy blocks: {} mappings", block_mapping.len());

    Ok(())
}

/// Swap blocks between GPU and CPU
#[pyfunction]
#[pyo3(signature = (src_key_cache, src_value_cache, dst_key_cache, dst_value_cache, block_mapping))]
pub fn swap_blocks_py(
    src_key_cache: &VulkanTensor,
    src_value_cache: &VulkanTensor,
    dst_key_cache: &VulkanTensor,
    dst_value_cache: &VulkanTensor,
    block_mapping: Vec<(i64, i64)>,
) -> PyResult<()> {
    // In real implementation, this would swap block data

    log::debug!("Swap blocks: {} mappings", block_mapping.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = PagedKVCache::new(100, 16, 8, 64, 32, 0, "f16").unwrap();
        assert_eq!(cache.num_total_blocks(), 100);
        assert_eq!(cache.num_free_blocks(), 100);
        assert_eq!(cache.num_used_blocks(), 0);
        assert_eq!(cache.block_size(), 16);
    }

    #[test]
    fn test_block_allocation() {
        let cache = PagedKVCache::new(100, 16, 8, 64, 32, 0, "f16").unwrap();

        let blocks = cache.allocate_blocks(1, 5).unwrap();
        assert_eq!(blocks.len(), 5);
        assert_eq!(cache.num_free_blocks(), 95);
        assert_eq!(cache.num_used_blocks(), 5);

        let block_table = cache.get_block_table(1).unwrap();
        assert_eq!(block_table.len(), 5);
    }

    #[test]
    fn test_block_free() {
        let cache = PagedKVCache::new(100, 16, 8, 64, 32, 0, "f16").unwrap();

        cache.allocate_blocks(1, 5).unwrap();
        assert_eq!(cache.num_free_blocks(), 95);

        cache.free_blocks(1).unwrap();
        assert_eq!(cache.num_free_blocks(), 100);
    }

    #[test]
    fn test_append_block() {
        let cache = PagedKVCache::new(100, 16, 8, 64, 32, 0, "f16").unwrap();

        cache.allocate_blocks(1, 2).unwrap();
        let new_block = cache.append_block(1).unwrap();

        let block_table = cache.get_block_table(1).unwrap();
        assert_eq!(block_table.len(), 3);
        assert_eq!(block_table.last(), Some(&new_block));
    }

    #[test]
    fn test_allocation_failure() {
        let cache = PagedKVCache::new(10, 16, 8, 64, 32, 0, "f16").unwrap();

        let result = cache.allocate_blocks(1, 20);
        assert!(result.is_err());
    }
}
