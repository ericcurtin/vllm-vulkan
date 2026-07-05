// SPDX-License-Identifier: Apache-2.0
//! Micro-benchmark for hoisting the per-index `(sin, cos)` table out of
//! `cpu_rope`'s per-head loop.
//!
//! `cpu_rope` (src/model.rs) applies rotary positional embedding to every
//! query head (8 for Gemma4-E2B) and key head (1, since Gemma4-E2B uses
//! multi-query attention) in the current decoder layer. Its rotation angle
//! for index `i` depends only on `pos`, `i`, `rotary_dim`, and `theta` —
//! the same four inputs for every head in a given call (Q and K alike,
//! since `forward_layer_gpu_matmuls`/`Gemma4Model::forward_layer` both call
//! `cpu_rope` with one shared `rotary_dim`/`theta` for the whole call). The
//! previous implementation recomputed `theta.powf(..)` and
//! `angle.sin_cos()` — both genuinely expensive transcendental functions,
//! unlike a plain multiply/add — inside the per-head loop, so a single
//! decode step paid for `rotary_dim/2` of each per *head* instead of once
//! per call. Precomputing the `(sin, cos)` table once and reusing it
//! across every head removes that redundant transcendental work entirely,
//! with no change in the result (every head applies the exact same
//! rotation it would otherwise have recomputed itself).
//!
//! This harness reproduces both versions standalone (no GPU/model
//! checkpoint required) and times them at both RoPE configurations
//! Gemma4-E2B actually uses (sliding-window layers: rotary_dim=head_dim=256,
//! theta=10000; full-attention layers: rotary_dim=head_dim/4=128,
//! theta=1000000), with num_q_heads=8, num_kv_heads=1 (the real GQA ratio
//! for this model), so the improvement is measurable without needing model
//! weights, a GPU, or network access.
//!
//! Run with:
//!     cargo run --release --example bench_rope

use std::time::Instant;

const NUM_Q_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 1;

/// Old behaviour: recompute `freq`/`angle`/`sin_cos` inside the per-head
/// loop (verbatim copy of `cpu_rope` before this change).
fn rope_recompute_per_head(
    q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, rotary_dim: usize, theta: f32,
) {
    let rotate_head = |x: &mut [f32], pos: usize, rotary_dim: usize, theta: f32| {
        let half = rotary_dim / 2;
        for i in 0..half {
            let freq = 1.0 / theta.powf(i as f32 * 2.0 / rotary_dim as f32);
            let angle = pos as f32 * freq;
            let (s, c) = angle.sin_cos();
            let x0 = x[i];
            let x1 = x[i + half];
            x[i]        = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
    };

    for h in 0..NUM_Q_HEADS {
        let slice = &mut q[h * head_dim..(h + 1) * head_dim];
        rotate_head(slice, pos, rotary_dim, theta);
    }
    for h in 0..NUM_KV_HEADS {
        let slice = &mut k[h * head_dim..(h + 1) * head_dim];
        rotate_head(slice, pos, rotary_dim, theta);
    }
}

/// New behaviour: precompute the `(sin, cos)` table once per call, reuse
/// it across every head (what `cpu_rope` does now).
fn rope_precomputed_table(
    q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, rotary_dim: usize, theta: f32,
) {
    let half = rotary_dim / 2;
    let mut sin_cos: Vec<(f32, f32)> = Vec::with_capacity(half);
    for i in 0..half {
        let freq = 1.0 / theta.powf(i as f32 * 2.0 / rotary_dim as f32);
        let angle = pos as f32 * freq;
        sin_cos.push(angle.sin_cos());
    }

    let rotate_head = |x: &mut [f32], sin_cos: &[(f32, f32)]| {
        for (i, &(s, c)) in sin_cos.iter().enumerate() {
            let x0 = x[i];
            let x1 = x[i + half];
            x[i]        = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
    };

    for h in 0..NUM_Q_HEADS {
        let slice = &mut q[h * head_dim..(h + 1) * head_dim];
        rotate_head(slice, &sin_cos);
    }
    for h in 0..NUM_KV_HEADS {
        let slice = &mut k[h * head_dim..(h + 1) * head_dim];
        rotate_head(slice, &sin_cos);
    }
}

/// Time `f` over `iters` calls, repeated `trials` times, returning the best
/// (minimum) per-call time in nanoseconds — see bench_sdpa.rs's doc comment
/// for why "best of N reps" is used here.
fn time_best_of(trials: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..trials {
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        let ns_per_iter = t0.elapsed().as_nanos() as f64 / iters as f64;
        best = best.min(ns_per_iter);
    }
    best
}

fn bench_one(label: &str, head_dim: usize, rotary_dim: usize, theta: f32, pos: usize, iters: usize) {
    let q_init: Vec<f32> = (0..NUM_Q_HEADS * head_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let k_init: Vec<f32> = (0..NUM_KV_HEADS * head_dim).map(|i| (i as f32 * 0.013).cos()).collect();

    // Correctness check: both paths must agree.
    let mut q_a = q_init.clone();
    let mut k_a = k_init.clone();
    rope_recompute_per_head(&mut q_a, &mut k_a, pos, head_dim, rotary_dim, theta);
    let mut q_b = q_init.clone();
    let mut k_b = k_init.clone();
    rope_precomputed_table(&mut q_b, &mut k_b, pos, head_dim, rotary_dim, theta);
    for (x, y) in q_a.iter().zip(q_b.iter()) {
        assert!((x - y).abs() < 1e-6, "Q mismatch for {label}: {x} vs {y}");
    }
    for (x, y) in k_a.iter().zip(k_b.iter()) {
        assert!((x - y).abs() < 1e-6, "K mismatch for {label}: {x} vs {y}");
    }

    let trials = 7;
    let old_ns = time_best_of(trials, iters, || {
        let mut q = q_init.clone();
        let mut k = k_init.clone();
        rope_recompute_per_head(
            std::hint::black_box(&mut q), std::hint::black_box(&mut k),
            pos, head_dim, rotary_dim, theta,
        );
    });
    let new_ns = time_best_of(trials, iters, || {
        let mut q = q_init.clone();
        let mut k = k_init.clone();
        rope_precomputed_table(
            std::hint::black_box(&mut q), std::hint::black_box(&mut k),
            pos, head_dim, rotary_dim, theta,
        );
    });

    let old_us = old_ns / 1000.0;
    let new_us = new_ns / 1000.0;
    let speedup = old_us / new_us;

    println!(
        "{label:<28} (rotary_dim={rotary_dim:>3}) old: {old_us:>7.3} us/op   new: {new_us:>7.3} us/op   speedup: {speedup:>5.2}x  [best of {trials}]"
    );
}

fn main() {
    println!("cpu_rope: old (sin/cos/powf recomputed per head) vs new (precomputed once per call)\n");

    let iters = 20_000;
    // Sliding-window layers: rotary_dim == head_dim (full rotation).
    bench_one("sliding-window layer", 256, 256, 10_000.0, 128, iters);
    // Full-attention layers: rotary_dim == head_dim/4 (partial_rotary_factor=0.25).
    bench_one("full-attention layer", 512, 128, 1_000_000.0, 128, iters);

    println!(
        "\ncpu_rope runs once per decoder layer regardless of GPU availability — RoPE stays \
on the CPU even when linear projections are GPU-accelerated. Each call previously \
recomputed rotary_dim/2 sin_cos+powf pairs once per head (8 query heads + 1 key head \
for Gemma4-E2B) even though they don't depend on which head is being rotated; \
precomputing them once per call removes 8x (sliding-window) redundant transcendental \
function calls, 35 layers per decode step."
    );
}
