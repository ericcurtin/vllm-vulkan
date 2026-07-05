// SPDX-License-Identifier: Apache-2.0
//! Micro-benchmark for the activation-tensor CPU→GPU marshalling path.
//!
//! `forward_layer_gpu_matmuls` (src/lib.rs) writes an f32 activation tensor
//! into a host-coherent Vulkan buffer before every GPU dispatch (QKV input,
//! o_proj input, FFN input, PLE input) — up to ~6 times per decoder layer,
//! 35 layers per decode step. Before this change, each write went through
//! `f32_slice_to_bytes`, which allocated a fresh `Vec<u8>` and copied the
//! data one `f32` at a time via `to_le_bytes()` + `copy_from_slice()`,
//! before a *second* full copy into the mapped Vulkan buffer in
//! `Buffer::write`. On little-endian targets (x86_64, aarch64 — the only
//! platforms this crate supports) an `f32` slice's in-memory representation
//! already *is* its little-endian byte representation, so that first pass
//! is redundant: `bytemuck::cast_slice` reinterprets the same bytes with no
//! allocation and no copy, leaving a single bulk `memcpy` into the mapped
//! buffer (done once, in `Buffer::write`) as the only remaining cost.
//!
//! This harness reproduces both code paths standalone (no Vulkan device
//! required) and times them across the tensor widths that actually occur
//! in the Gemma4-E2B decode loop, so the improvement is measurable without
//! needing model weights, a GPU, or network access.
//!
//! Run with:
//!     cargo run --release --example bench_activation_write

use std::time::Instant;

/// The old implementation (verbatim copy of the removed `f32_slice_to_bytes`
/// / `float_to_bytes` helpers), kept here only as a benchmark baseline.
fn f32_slice_to_bytes_old(data: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; data.len() * 4];
    for (i, &v) in data.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// The new implementation: a zero-copy reinterpret cast.
fn f32_slice_to_bytes_new(data: &[f32]) -> &[u8] {
    bytemuck::cast_slice(data)
}

/// Simulates writing into a mapped, host-coherent Vulkan buffer (this is
/// what `Buffer::write` does with `std::ptr::copy_nonoverlapping`).
fn write_into(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
}

fn bench_one(label: &str, hidden: usize, iters: usize) {
    let data: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.5).collect();
    let mut mapped = vec![0u8; hidden * 4];

    // Warm up + correctness check: both paths must produce identical bytes.
    let old_bytes = f32_slice_to_bytes_old(&data);
    let new_bytes = f32_slice_to_bytes_new(&data);
    assert_eq!(old_bytes, new_bytes, "byte mismatch for {label}");

    let t0 = Instant::now();
    for _ in 0..iters {
        let bytes = f32_slice_to_bytes_old(std::hint::black_box(&data));
        write_into(&mut mapped, &bytes);
        std::hint::black_box(&mapped);
    }
    let old_elapsed = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        let bytes = f32_slice_to_bytes_new(std::hint::black_box(&data));
        write_into(&mut mapped, bytes);
        std::hint::black_box(&mapped);
    }
    let new_elapsed = t0.elapsed();

    let old_ns = old_elapsed.as_nanos() as f64 / iters as f64;
    let new_ns = new_elapsed.as_nanos() as f64 / iters as f64;
    let speedup = old_ns / new_ns;

    println!(
        "{label:<28} ({hidden:>6} f32) old: {old_ns:>8.1} ns/op   new: {new_ns:>8.1} ns/op   speedup: {speedup:>5.2}x"
    );
}

fn main() {
    println!("Activation write marshalling: old (per-element loop + alloc) vs new (bytemuck::cast_slice)\n");

    // Representative Gemma4-E2B decode-step tensor widths (src/model.rs):
    //   hidden_size = 1536, q_dim = 8*512 = 4096, kv_dim = 1*512 = 512,
    //   intermediate_size = 6144 (12288 for KV-shared double-wide layers),
    //   ple_dim = 256.
    let iters = 20_000;
    bench_one("ple_dim", 256, iters);
    bench_one("kv_dim", 512, iters);
    bench_one("hidden_size", 1536, iters);
    bench_one("q_dim", 4096, iters);
    bench_one("intermediate_size", 6144, iters);
    bench_one("intermediate_size (KV-shared)", 12288, iters);

    println!(
        "\nEach decoder layer issues up to 6 of these writes (QKV, o_proj, FFN, PLE inputs); \
Gemma4-E2B has 35 layers per decode step, so this is on the per-token hot path."
    );
}
