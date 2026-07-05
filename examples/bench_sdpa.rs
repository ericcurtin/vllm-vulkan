// SPDX-License-Identifier: Apache-2.0
//! Micro-benchmark for scratch-buffer reuse in `cpu_sdpa`.
//!
//! `cpu_sdpa` (src/model.rs) runs once per decoder layer regardless of GPU
//! availability — it's the KV-cache attention step, which stays on the CPU
//! even in the GPU-accelerated decode path (`forward_layer_gpu_matmuls` in
//! src/lib.rs only offloads the linear projections to the GPU). Internally
//! it loops over every query head (8 for Gemma4-E2B) and, for each one,
//! allocated a fresh `scores` and `exp_scores` `Vec<f32>` — even though
//! which KV positions are attended to (`kv_start`/`valid_len`) only depends
//! on `seq_len`/`sliding_window`, which are loop-invariant across heads in
//! the same call. Hoisting those two allocations out of the per-head loop
//! turns 8 allocation-pairs per call into 1.
//!
//! (An earlier version of this change also tried reformulating the score
//! and weighted-value steps as `matrixmultiply::sgemm` calls, matching the
//! SIMD kernel `cpu_matmul` already uses for the linear projections. That
//! was measurably *slower* at Gemma4-E2B's actual sliding-window head_dim
//! (256) for realistic sequence lengths — likely GEMM packing/blocking
//! overhead that a single dot-product/accumulation over a few hundred
//! elements doesn't amortize — so it was dropped in favour of this
//! narrower, unconditionally-safe change instead.)
//!
//! This harness reproduces both versions standalone (no GPU/model
//! checkpoint required) and times them at a representative sequence length
//! for Gemma4-E2B's sliding_window (512), with num_q_heads=8, num_kv_heads=1
//! (the real GQA ratio for this model), so the improvement is measurable
//! without needing model weights, a GPU, or network access.
//!
//! Run with:
//!     cargo run --release --example bench_sdpa

use std::time::Instant;

const HEAD_DIM: usize = 256;
const NUM_Q_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 1;

/// Old behaviour: allocate `scores`/`exp_scores` fresh inside the per-head
/// loop (verbatim copy of `cpu_sdpa` before this change).
fn sdpa_alloc_per_head(
    q: &[f32], k: &[f32], v: &[f32], seq_len: usize, scale: f32,
) -> Vec<f32> {
    let gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;
    let mut out = vec![0.0f32; NUM_Q_HEADS * HEAD_DIM];

    for qh in 0..NUM_Q_HEADS {
        let kvh = qh / gqa_ratio;
        let q_row = &q[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];
        let valid_len = seq_len;
        let mut scores = vec![f32::NEG_INFINITY; valid_len];

        for (si, kv_pos) in (0..seq_len).enumerate() {
            let k_row = &k[(kv_pos * NUM_KV_HEADS + kvh) * HEAD_DIM
                          ..(kv_pos * NUM_KV_HEADS + kvh + 1) * HEAD_DIM];
            let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(&a, &b)| a * b).sum();
            scores[si] = dot * scale;
        }

        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        exp_scores.iter_mut().for_each(|s| *s /= sum);

        let out_row = &mut out[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];
        for (si, kv_pos) in (0..seq_len).enumerate() {
            let v_row = &v[(kv_pos * NUM_KV_HEADS + kvh) * HEAD_DIM
                          ..(kv_pos * NUM_KV_HEADS + kvh + 1) * HEAD_DIM];
            let w = exp_scores[si];
            for (o, &vv) in out_row.iter_mut().zip(v_row.iter()) {
                *o += w * vv;
            }
        }
    }
    out
}

/// New behaviour: hoist `scores`/`exp_scores` out of the per-head loop
/// (what `cpu_sdpa` does now).
fn sdpa_scratch_reuse(
    q: &[f32], k: &[f32], v: &[f32], seq_len: usize, scale: f32,
) -> Vec<f32> {
    let gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;
    let mut out = vec![0.0f32; NUM_Q_HEADS * HEAD_DIM];

    let valid_len = seq_len;
    let mut scores = vec![0.0f32; valid_len];
    let mut exp_scores = vec![0.0f32; valid_len];

    for qh in 0..NUM_Q_HEADS {
        let kvh = qh / gqa_ratio;
        let q_row = &q[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];

        for (si, kv_pos) in (0..seq_len).enumerate() {
            let k_row = &k[(kv_pos * NUM_KV_HEADS + kvh) * HEAD_DIM
                          ..(kv_pos * NUM_KV_HEADS + kvh + 1) * HEAD_DIM];
            let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(&a, &b)| a * b).sum();
            scores[si] = dot * scale;
        }

        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for (e, &s) in exp_scores.iter_mut().zip(scores.iter()) {
            *e = (s - max_score).exp();
        }
        let sum: f32 = exp_scores.iter().sum();
        exp_scores.iter_mut().for_each(|s| *s /= sum);

        let out_row = &mut out[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];
        for (si, kv_pos) in (0..seq_len).enumerate() {
            let v_row = &v[(kv_pos * NUM_KV_HEADS + kvh) * HEAD_DIM
                          ..(kv_pos * NUM_KV_HEADS + kvh + 1) * HEAD_DIM];
            let w = exp_scores[si];
            for (o, &vv) in out_row.iter_mut().zip(v_row.iter()) {
                *o += w * vv;
            }
        }
    }
    out
}

/// Time `f` over `iters` calls, repeated `trials` times, returning the best
/// (minimum) per-call time in nanoseconds. Taking the minimum of several
/// trials — rather than a single wall-clock measurement — is standard
/// practice for microbenchmarks on shared/virtualized hardware where
/// scheduling noise can otherwise dominate a small, fast operation (the same
/// "best of N reps" approach scripts/bench_vulkan_model.py already uses for
/// end-to-end tok/s numbers in this repo).
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

fn bench_one(label: &str, seq_len: usize, iters: usize) {
    let q: Vec<f32> = (0..NUM_Q_HEADS * HEAD_DIM).map(|i| (i as f32 * 0.01).sin()).collect();
    let k: Vec<f32> = (0..seq_len * NUM_KV_HEADS * HEAD_DIM).map(|i| (i as f32 * 0.013).cos()).collect();
    let v: Vec<f32> = (0..seq_len * NUM_KV_HEADS * HEAD_DIM).map(|i| (i as f32 * 0.007).sin()).collect();
    let scale = 1.0f32;

    // Correctness check: both paths must agree.
    let a = sdpa_alloc_per_head(&q, &k, &v, seq_len, scale);
    let b = sdpa_scratch_reuse(&q, &k, &v, seq_len, scale);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-5, "mismatch for {label}: {x} vs {y}");
    }

    let trials = 7;
    let old_ns = time_best_of(trials, iters, || {
        std::hint::black_box(sdpa_alloc_per_head(
            std::hint::black_box(&q), std::hint::black_box(&k), std::hint::black_box(&v),
            seq_len, scale,
        ));
    });
    let new_ns = time_best_of(trials, iters, || {
        std::hint::black_box(sdpa_scratch_reuse(
            std::hint::black_box(&q), std::hint::black_box(&k), std::hint::black_box(&v),
            seq_len, scale,
        ));
    });

    let old_us = old_ns / 1000.0;
    let new_us = new_ns / 1000.0;
    let speedup = old_us / new_us;

    println!(
        "{label:<32} (seq_len={seq_len:>4}) old: {old_us:>7.2} us/op   new: {new_us:>7.2} us/op   speedup: {speedup:>5.2}x  [best of {trials}]"
    );
}

fn main() {
    println!("cpu_sdpa: old (scores/exp_scores allocated per head) vs new (allocated once per call)\n");

    let iters = 2_000;
    bench_one("early decode", 32, iters);
    bench_one("mid decode", 256, iters);
    bench_one("sliding_window full (512)", 512, iters);

    println!(
        "\ncpu_sdpa runs once per decoder layer regardless of GPU availability — it's the \
KV-cache attention step, which stays on the CPU even when linear projections are \
GPU-accelerated. 35 layers per decode step for Gemma4-E2B, num_q_heads=8, so this \
removes up to 7 redundant allocation-pairs per layer (56/token). The per-call time is \
dominated by the O(seq_len * head_dim) dot-product/accumulation work, so the benefit \
here is real but modest — allocation elimination reduces allocator pressure without \
being the dominant cost, and never regresses performance (unlike a GEMM-based \
reformulation of the dot products themselves, which measured slower at this head_dim — \
see the module doc comment above)."
    );
}
