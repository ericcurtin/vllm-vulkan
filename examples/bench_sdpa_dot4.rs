// SPDX-License-Identifier: Apache-2.0
//! Micro-benchmark for `cpu_sdpa`'s 4-lane-accumulator dot product.
//!
//! `cpu_sdpa` (src/model.rs) computes its attention scores via a dot
//! product between each query head and every cached key row:
//! `q_row.iter().zip(k_row).map(|(a,b)| a*b).sum()`. `Iterator::sum()` over
//! floats must preserve strict left-to-right summation order — float
//! addition isn't associative, so the compiler can't reorder it without
//! risking a different (if usually negligibly different) rounding result —
//! which means that dot product has a single serial addition chain: each
//! accumulation must wait for the previous one, regardless of how well the
//! multiplies themselves vectorize.
//!
//! Splitting the accumulation into 4 independent lanes (summed together
//! only once, at the very end) breaks that dependency chain and lets the
//! compiler pipeline/vectorize the multiply-adds across lanes. 4 lanes
//! (rather than 8 or 16, which both measured worse) matches a 128-bit SIMD
//! register's f32 width — the smallest width common to every architecture
//! this crate targets (NEON on aarch64, SSE on x86_64) — going wider
//! appears to work against the compiler's own auto-vectorization of each
//! lane's scalar loop rather than complementing it.
//!
//! This harness reproduces both dot-product versions standalone (no
//! GPU/model checkpoint required) and runs them inside a full `cpu_sdpa`-
//! shaped score computation loop at Gemma4-E2B's real head_dim (256) and
//! sliding-window length (512), so the improvement is measurable without
//! needing model weights, a GPU, or network access. See also
//! `model::cpu_dot4_tests` (src/model.rs) for the corresponding
//! correctness tests against the real `cpu_sdpa`.
//!
//! Run with:
//!     cargo run --release --example bench_sdpa_dot4

use std::time::Instant;

const HEAD_DIM: usize = 256;
const NUM_Q_HEADS: usize = 8;
const NUM_KV_HEADS: usize = 1;

/// Old behaviour: single running accumulator via `Iterator::sum()`.
fn dot_naive(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// New behaviour: 4 independent accumulator lanes (what `cpu_sdpa`'s
/// `dot4` helper does now).
fn dot4(a: &[f32], b: &[f32]) -> f32 {
    // Real (not debug-only) assertion — lets the compiler prove `b`'s
    // indices below are in-bounds even in release builds, matching
    // src/model.rs's dot4 (see its doc comment for why this matters for
    // the benchmark to actually measure what it claims to).
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let mut acc = [0.0f32; 4];
    for c in 0..chunks {
        let i = c * 4;
        acc[0] += a[i] * b[i];
        acc[1] += a[i + 1] * b[i + 1];
        acc[2] += a[i + 2] * b[i + 2];
        acc[3] += a[i + 3] * b[i + 3];
    }
    let mut tail = 0.0f32;
    for i in chunks * 4..n {
        tail += a[i] * b[i];
    }
    acc[0] + acc[1] + acc[2] + acc[3] + tail
}

/// Just the score-computation half of `cpu_sdpa` (softmax + weighted-V-sum
/// are unaffected by this change, so they're omitted here to isolate what
/// actually changed).
fn compute_scores(q: &[f32], k: &[f32], seq_len: usize, scale: f32, dotf: fn(&[f32], &[f32]) -> f32) -> Vec<f32> {
    let gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;
    let mut all_scores = vec![0.0f32; NUM_Q_HEADS * seq_len];
    for qh in 0..NUM_Q_HEADS {
        let kvh = qh / gqa_ratio;
        let q_row = &q[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];
        let scores = &mut all_scores[qh * seq_len..(qh + 1) * seq_len];
        for (si, kv_pos) in (0..seq_len).enumerate() {
            let k_row = &k[(kv_pos * NUM_KV_HEADS + kvh) * HEAD_DIM
                          ..(kv_pos * NUM_KV_HEADS + kvh + 1) * HEAD_DIM];
            scores[si] = dotf(q_row, k_row) * scale;
        }
    }
    all_scores
}

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
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    // Correctness: both dot products must agree (up to fp reassociation noise).
    let a = compute_scores(&q, &k, seq_len, scale, dot_naive);
    let b = compute_scores(&q, &k, seq_len, scale, dot4);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-3, "mismatch for {label}: {x} vs {y}");
    }

    let trials = 7;
    let old_ns = time_best_of(trials, iters, || {
        std::hint::black_box(compute_scores(std::hint::black_box(&q), std::hint::black_box(&k), seq_len, scale, dot_naive));
    });
    let new_ns = time_best_of(trials, iters, || {
        std::hint::black_box(compute_scores(std::hint::black_box(&q), std::hint::black_box(&k), seq_len, scale, dot4));
    });

    let old_us = old_ns / 1000.0;
    let new_us = new_ns / 1000.0;
    let speedup = old_us / new_us;

    println!(
        "{label:<32} (seq_len={seq_len:>4}) single-acc: {old_us:>7.2} us/op   4-lane: {new_us:>7.2} us/op   speedup: {speedup:>5.2}x  [best of {trials}]"
    );
}

fn main() {
    println!("cpu_sdpa score computation: single-accumulator dot product vs 4-lane dot4\n");

    let iters = 2_000;
    bench_one("early decode", 32, iters);
    bench_one("mid decode", 256, iters);
    bench_one("sliding_window full (512)", 512, iters);

    println!(
        "\ncpu_sdpa's score computation runs once per decoder layer regardless of GPU \
availability — it's the KV-cache attention step, which stays on the CPU even when \
linear projections are GPU-accelerated (35 layers per decode step for Gemma4-E2B). \
Iterator::sum() over floats can't be auto-vectorized by the compiler because float \
addition isn't associative, so the single-accumulator version pays for a fully serial \
dependency chain across head_dim=256 multiply-adds per (query head, KV position) pair. \
Splitting into 4 independent accumulator lanes removes that chain, matching this \
hardware's 128-bit SIMD register width."
    );
}
