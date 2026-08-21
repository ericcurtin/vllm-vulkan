// SPDX-License-Identifier: Apache-2.0
//! Foundation-level host decode-state snapshot.
//!
//! `HostStateSnapshot` records the HOST-authoritative per-token decode state
//! that speculative-decode rollback rewinds: per-layer GatedDeltaNet
//! (`conv_state` + delta-rule matrix) clones and per-attention-layer KV
//! `seq_len` counters. It is a plain data container (only `Vec`s of primitives)
//! with NO model-specific type dependencies, so it lives in the foundation and
//! is re-exported by the qwen35 module (`qwen35::HostStateSnapshot`) for the
//! code that populates it. Keeping the TYPE in the foundation lets the generic
//! `SpecSlot` (in `lib.rs`) reference it even when no model feature is enabled.
//!
//! (Phase A2 upstream-carve: this type used to live in `qwen35.rs`, coupling
//! the foundation `SpecSlot` to the qwen35 module. Moved here so the foundation
//! compiles with zero models.)

/// Bit-exact rollback snapshot of one PP stage's HOST-authoritative per-token
/// decode state (P1 speculative-pipelining rollback). Captures, keyed by
/// stage-local `state_idx`:
///   - every linear (GatedDeltaNet) layer's `DeltaNetState` (sliding conv window
///     + delta-rule matrix).
///   - every full-attention layer's KV `seq_len` counter.
#[derive(Clone, Default)]
pub struct HostStateSnapshot {
    /// (state_idx, conv_state clone, delta-state clone) per captured linear layer.
    pub dn: Vec<(usize, Vec<f32>, Vec<f32>)>,
    /// (state_idx, seq_len) per full-attention layer.
    pub kv: Vec<(usize, usize)>,
}
