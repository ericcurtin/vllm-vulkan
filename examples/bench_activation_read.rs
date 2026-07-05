// SPDX-License-Identifier: Apache-2.0
//! Micro-benchmark for the GPU→CPU activation-tensor read-back path.
//!
//! `forward_layer_gpu_matmuls` (src/lib.rs) reads a GPU dispatch's output
//! out of a persistent, host-coherent, permanently-mapped Vulkan buffer
//! after every submit. Three of these reads per decoder layer (o_proj
//! output, FFN down_proj output, PLE contribution output) are only ever
//! *read* afterwards — passed straight into `cpu_rms_norm(&x, ...)`, never
//! mutated in place — yet the old code still went through `read_f32_buf`,
//! which allocates a fresh `Vec<f32>` and copies every element out of the
//! mapped buffer via `.to_vec()`, purely so its type would match the
//! CPU-fallback branch's owned `Vec<f32>` return value.
//!
//! Since the buffer is already mapped, permanent, host-coherent memory —
//! and outlives the local scope that reads it — that copy is unnecessary
//! for the read-only cases: an `FBuf` (owned-or-borrowed enum, see
//! `src/lib.rs`) lets the GPU branch hand back a zero-copy `RawSlice` view
//! directly into the mapped buffer instead, while the CPU-fallback branch
//! still returns an owned `Vec<f32>` — both unified behind one `Deref<Target
//! = [f32]>` type so the call site doesn't need to know which one it got.
//!
//! This harness reproduces both approaches standalone (no Vulkan device
//! required — a `Vec<u8>` stands in for the mapped buffer) and times them
//! at the tensor width actually used by all three read-only sites in the
//! Gemma4-E2B decode loop (`hidden_size` = 1536), so the improvement is
//! measurable without needing model weights, a GPU, or network access.
//!
//! Run with:
//!     cargo run --release --example bench_activation_read

use std::time::Instant;

/// Stand-in for a permanently-mapped, host-coherent Vulkan buffer that a GPU
/// dispatch has just written its output into.
///
/// Real `compute::Buffer::mapped_ptr` is a `*mut u8` from `vkMapMemory`,
/// which the Vulkan spec guarantees is suitably aligned for the mapped
/// range's usage. A plain `Vec<u8>`'s allocation is only guaranteed aligned
/// to `align_of::<u8>() == 1`, so reinterpreting *that* as `*const f32`
/// would be unsound (the allocator may happen to hand back a 4-byte-aligned
/// pointer in practice, but nothing guarantees it). Storing `Vec<f32>`
/// directly sidesteps the question entirely — its allocation is guaranteed
/// aligned to `align_of::<f32>()` by construction — while still exercising
/// exactly the pointer-based read paths being compared.
struct FakeMappedBuffer {
    data: Vec<f32>,
}

impl FakeMappedBuffer {
    fn new(count: usize) -> Self {
        let data: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();
        FakeMappedBuffer { data }
    }
}

/// Old behaviour: `read_f32_buf` — allocate a fresh `Vec<f32>` and copy the
/// mapped buffer's contents into it.
fn read_f32_buf_old(buf: &FakeMappedBuffer, count: usize) -> Vec<f32> {
    let ptr = buf.data.as_ptr();
    unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
}

/// New behaviour: a zero-copy borrowed view directly into the mapped buffer
/// (what `FBuf::Borrowed(RawSlice { .. })` does in src/lib.rs).
fn read_f32_buf_new(buf: &FakeMappedBuffer, count: usize) -> &[f32] {
    let ptr = buf.data.as_ptr();
    unsafe { std::slice::from_raw_parts(ptr, count) }
}

/// What the read result is actually used for at all three call sites this
/// change applies to: `cpu_rms_norm(&x, weight, eps)`, a read-only pass over
/// the slice. Kept trivial here since we're isolating the read-back cost,
/// not RMSNorm itself (unchanged by this optimization).
fn consume(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |acc, &v| acc + v)
}

fn bench_one(label: &str, count: usize, iters: usize) {
    let buf = FakeMappedBuffer::new(count);

    // Correctness check: both paths must read identical data.
    assert_eq!(read_f32_buf_old(&buf, count), read_f32_buf_new(&buf, count));

    let t0 = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iters {
        let v = read_f32_buf_old(std::hint::black_box(&buf), count);
        acc += consume(&v);
    }
    std::hint::black_box(acc);
    let old_elapsed = t0.elapsed();

    let t0 = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iters {
        let v = read_f32_buf_new(std::hint::black_box(&buf), count);
        acc += consume(v);
    }
    std::hint::black_box(acc);
    let new_elapsed = t0.elapsed();

    let old_ns = old_elapsed.as_nanos() as f64 / iters as f64;
    let new_ns = new_elapsed.as_nanos() as f64 / iters as f64;
    let speedup = old_ns / new_ns;

    println!(
        "{label:<32} ({count:>6} f32) old: {old_ns:>8.1} ns/op   new: {new_ns:>8.1} ns/op   speedup: {speedup:>5.2}x"
    );
}

fn main() {
    println!("Activation read-back: old (read_f32_buf alloc+copy) vs new (FBuf zero-copy borrow)\n");

    let iters = 50_000;
    // hidden_size = 1536 — the width used by all three read-only-after-read
    // sites this change applies to (o_proj output, FFN down_proj output, PLE
    // contribution output). Also bench a couple of neighbouring widths from
    // the same model (src/model.rs) for context.
    bench_one("ple_dim", 256, iters);
    bench_one("hidden_size (o_proj/down/PLE-out)", 1536, iters);
    bench_one("intermediate_size", 6144, iters);

    println!(
        "\no_proj, FFN down_proj, and PLE contribution outputs are all read exactly \
once (into cpu_rms_norm) and never mutated afterwards, so the GPU-dispatch \
branch can hand back a borrowed view straight into the persistent mapped \
buffer instead of paying for read_f32_buf's Vec<f32> allocation + copy. \
That's 3 of these per decoder layer, 35 layers per decode step."
    );
}
