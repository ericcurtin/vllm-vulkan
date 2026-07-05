// SPDX-License-Identifier: Apache-2.0
//! Micro-benchmark for per-head RMSNorm in the decode hot path.
//!
//! `forward_layer_gpu_matmuls` (src/lib.rs) and `Gemma4Model::forward_layer`
//! (src/model.rs, the pure-CPU fallback used when no Vulkan device is
//! available) both normalise Q/K/V per attention head with the same
//! pattern:
//!
//! ```ignore
//! let s = &mut q[hi * head_dim..(hi + 1) * head_dim];
//! let n = model::cpu_rms_norm(s, &q_norm_w, eps);  // allocates a new Vec<f32>
//! s.copy_from_slice(&n);                            // ...then copies it back
//! ```
//!
//! `cpu_rms_norm` always allocates a fresh output `Vec<f32>`, even though
//! every one of these call sites immediately copies the result back over
//! the exact slice it read from — paying for a heap allocation *and* a
//! redundant copy just to mutate a buffer in place. `cpu_rms_norm_inplace`
//! does the same math directly over `&mut [f32]`, with neither.
//!
//! Gemma4-E2B calls this up to 10 times per decoder layer (8 query heads +
//! up to 1 key head + 1 value head, when K/V aren't KV-shared), 35 layers
//! per decode step — so this is squarely on the per-token hot path.
//!
//! This harness reproduces both approaches standalone (no GPU/model
//! checkpoint required) and times them at the two head widths Gemma4-E2B
//! actually uses (`head_dim` = 256 for sliding-window layers, 512 for
//! full-attention layers).
//!
//! Run with:
//!     cargo run --release --example bench_rms_norm_inplace

use std::time::Instant;

/// Old behaviour: allocate a normalised copy, then copy it back over `x`.
fn rms_norm_alloc_then_copy_back(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = weight.len();
    let mut out = vec![0.0f32; x.len()];
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let scale = 1.0 / rms;
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
    x.copy_from_slice(&out);
}

/// New behaviour: `cpu_rms_norm_inplace` — normalise `x` directly, no
/// allocation.
fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = weight.len();
    let rms = (x.iter().map(|&v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let scale = 1.0 / rms;
    for (v, &w) in x.iter_mut().zip(weight.iter()) {
        *v = *v * scale * w;
    }
}

fn bench_one(label: &str, head_dim: usize, iters: usize) {
    let weight: Vec<f32> = (0..head_dim).map(|i| 1.0 + i as f32 * 0.001).collect();
    let base: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.37).sin()).collect();
    let eps = 1e-6f32;

    // Correctness check: both paths must produce the same result.
    let mut a = base.clone();
    let mut b = base.clone();
    rms_norm_alloc_then_copy_back(&mut a, &weight, eps);
    rms_norm_inplace(&mut b, &weight, eps);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-6, "mismatch for {label}: {x} vs {y}");
    }

    let mut buf = base.clone();
    let t0 = Instant::now();
    for _ in 0..iters {
        rms_norm_alloc_then_copy_back(std::hint::black_box(&mut buf), &weight, eps);
    }
    let old_elapsed = t0.elapsed();

    let mut buf = base.clone();
    let t0 = Instant::now();
    for _ in 0..iters {
        rms_norm_inplace(std::hint::black_box(&mut buf), &weight, eps);
    }
    let new_elapsed = t0.elapsed();

    let old_ns = old_elapsed.as_nanos() as f64 / iters as f64;
    let new_ns = new_elapsed.as_nanos() as f64 / iters as f64;
    let speedup = old_ns / new_ns;

    println!(
        "{label:<28} (head_dim={head_dim:>4}) old: {old_ns:>7.1} ns/op   new: {new_ns:>7.1} ns/op   speedup: {speedup:>5.2}x"
    );
}

fn main() {
    println!("Per-head RMSNorm: old (alloc + copy-back) vs new (cpu_rms_norm_inplace)\n");

    let iters = 200_000;
    bench_one("sliding-window layer", 256, iters);
    bench_one("full-attention layer", 512, iters);

    println!(
        "\nQ-norm alone runs this up to 8 times per decoder layer (one per query \
head), plus up to 2 more for K-norm/V-norm when not KV-shared; Gemma4-E2B \
has 35 layers per decode step."
    );
}
