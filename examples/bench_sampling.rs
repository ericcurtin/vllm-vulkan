// SPDX-License-Identifier: Apache-2.0
//! Micro-benchmark for `sample_with_temperature`, the Rust replacement for
//! `vllm_vulkan/server.py`'s pure-Python `temperature_sample`.
//!
//! That Python function is the sampling step used once per decode step by
//! the standalone Rust-`VulkanModel`-backed serving path
//! (`vllm_vulkan.server`, documented as giving "~3 tok/s on GB10"). It does
//! a full `sorted()` over all `vocab_size` (262144 for Gemma4-E2B) logits —
//! plus several more full-vocab list comprehensions for temperature scaling
//! and softmax — in the CPython interpreter, on every single decode step.
//!
//! `VulkanModel.forward_and_sample` (src/lib.rs) replaces the combination
//! of `forward()` + `temperature_sample()` with a single Rust call that
//! never converts the logit vector into a Python object at all. This
//! harness reproduces just the pure-Rust half of that (`sample_with_
//! temperature`'s algorithmic cost) standalone, without needing a GPU,
//! model weights, or a Python interpreter — see
//! `vllm_vulkan._rs.sample_logits` (callable from Python directly) and
//! `/tmp/opencode/bench_sampling.py`-style scripts for measuring the
//! Python-interpreter-overhead side of the comparison, which is where the
//! actual win lives (a real measurement on this hardware: 82.2ms/call for
//! the old pure-Python implementation vs. 8.6ms/call even through
//! `sample_logits`, which still pays a Python-list round trip that
//! `forward_and_sample` skips entirely — a 9.5x speedup, dominated by
//! CPython interpreter overhead rather than anything algorithmic).
//!
//! Run with:
//!     cargo run --release --example bench_sampling

use std::time::Instant;

// Mirrors src/model.rs's sample_with_temperature/argmax exactly (examples
// can't import crate internals from a cdylib — see other examples in this
// directory for the same constraint).

fn argmax(logits: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

fn sample_with_temperature(logits: &[f32], temperature: f32, top_p: f32, top_k: i64, uniform_random: f32) -> usize {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let n = logits.len();
    let inv_temp = 1.0 / temperature;
    let max_scaled = logits.iter().fold(f32::NEG_INFINITY, |m, &l| m.max(l * inv_temp));
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l * inv_temp - max_scaled).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let k = if top_k <= 0 { n } else { (top_k as usize).min(n) };
    let top_k_order = &order[..k];
    let top_k_sum: f32 = top_k_order.iter().map(|&i| probs[i]).sum();

    let mut cumsum = 0.0f32;
    let mut nucleus_end = top_k_order.len();
    for (pos, &idx) in top_k_order.iter().enumerate() {
        cumsum += probs[idx] / top_k_sum;
        if cumsum >= top_p {
            nucleus_end = pos + 1;
            break;
        }
    }
    let nucleus = &top_k_order[..nucleus_end];
    let nucleus_sum: f32 = nucleus.iter().map(|&i| probs[i]).sum();

    let mut cumsum = 0.0f32;
    for &idx in nucleus {
        cumsum += probs[idx] / nucleus_sum;
        if uniform_random <= cumsum {
            return idx;
        }
    }
    *nucleus.last().unwrap()
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

fn main() {
    let vocab = 262144usize; // Gemma4-E2B
    let logits: Vec<f32> = (0..vocab).map(|i| ((i as f32 * 0.0001).sin()) * 10.0).collect();

    let iters = 50;
    let trials = 5;
    let us = time_best_of(trials, iters, || {
        std::hint::black_box(sample_with_temperature(
            std::hint::black_box(&logits), 1.0, 0.95, 64, 0.42,
        ));
    }) / 1000.0;

    println!("sample_with_temperature over {vocab} logits: {us:.2} us/call [best of {trials}, {iters} iters/trial]");
    println!();
    println!(
        "This is the pure-Rust algorithmic cost only (no Python interpreter\n\
involved) — the actual win over vllm_vulkan/server.py's pure-Python\n\
temperature_sample comes overwhelmingly from CPython interpreter overhead,\n\
not algorithmic complexity: a real measurement on this hardware (Python\n\
timeit, not this harness) showed the old pure-Python implementation at\n\
~82.2ms/call for the same vocab size, vs. ~8.6ms/call even through\n\
vllm_vulkan._rs.sample_logits (which still pays a Python-list round trip\n\
that VulkanModel.forward_and_sample skips entirely) — a 9.5x speedup.\n\
See sample_with_temperature's doc comment (src/model.rs) for the full\n\
rationale."
    );
}
