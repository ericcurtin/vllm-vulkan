// SPDX-License-Identifier: Apache-2.0
//! Numerical conformance harness for the quantized matvec kernels.
//!
//! `registry_tests` (lib.rs) proves the 23 required kernels are *present* and
//! that their SPIR-V is well-formed. That is a packaging check: it says nothing
//! about whether a kernel computes the right numbers. This module closes that
//! gap — it runs the REAL compiled SPIR-V on whatever Vulkan device the machine
//! exposes and compares every dispatch against the CPU quantizer/dequantizer
//! reference that already lives in `crate::model`.
//!
//! Design constraints (all load-bearing for CI):
//!
//! * **No device → no-op PASS.** The default CI job has no ICD, so
//!   `device::is_vulkan_available()` returning false must leave the suite green.
//!   A device only ever *adds* checks; it never turns a green run red for
//!   environmental reasons.
//! * **Device-shaped skips, not failures.** A kernel the `PipelineCache`
//!   declined to compile (e.g. the wave64-baked `paged_attn_decode_*_sg` on a
//!   subgroup!=64 device) is reported as SKIPPED. Same for the f16-typed
//!   kernels when the device lacks `storage_buffer_16_bit_access &&
//!   shader_float16`.
//! * **Tiny + deterministic.** n=16 rows, k=128 contraction, a fixed LCG for
//!   data. No RNG crate, no disk, no network, no checkpoints. The whole sweep
//!   is a few hundred microseconds of GPU work.
//!
//! ─── Tolerance ──────────────────────────────────────────────────────────────
//!
//! Dequantization itself is EXACT on both sides (the same integer codes and the
//! same f32 scale/bias values), so the only legitimate divergence is the ORDER
//! and ASSOCIATIVITY of the f32 accumulation: the CPU reference sums a row
//! left-to-right, while each kernel splits k across BLOCK_SIZE threads and
//! recombines through an LDS tree or a `subgroupAdd`, several of them folding
//! the affine term with `fma` (`scale*Σqx + bias*Σx`) instead of per element.
//! Bit-equality is therefore the WRONG assertion. We use the standard forward
//! error bound for a dot product, normalizing each row's error by that row's
//! sum of absolute products:
//!
//!     rel_err = max_r |gpu[r] - ref[r]| / Σ_j |w[r,j] · x[j]|
//!
//! For k=128 in f32 (eps = 2^-24 ≈ 5.96e-8) any summation order is bounded by
//! roughly k·eps ≈ 7.6e-6 of that denominator; an fma-based reassociation only
//! lowers it. `TOL = 1e-5` sits just above that worst case and is still ~3
//! orders of magnitude tighter than any real kernel bug (a wrong nibble shift,
//! a transposed index, a dropped bias) which shows up at O(1) relative error.
//! Observed values on real devices are ~1e-7, i.e. ~100x inside the bound.
//!
//! Measured sensitivity at k=128: corrupting ONE dequantized weight element by
//! 1% is caught (rel_err 6.9e-5); 0.05% on one element is not (3.5e-6, inside
//! the bound). That is the intended trade — this harness is aimed at structural
//! kernel faults (indexing, unpacking, missing terms, unwritten rows), not at
//! last-ulp drift, which is device-specific and not a correctness property.
//!
//! `harness_detects_a_corrupted_reference` is the falsifiability check: it runs
//! a real dispatch against a deliberately perturbed reference and asserts the
//! comparator REJECTS it, so this file can never degrade into a test that only
//! knows how to pass.

use crate::compute::{Buffer, ComputeEngine};
use crate::device;
use crate::include_all_shaders;
use crate::model;

/// Contraction dimension. 128 satisfies every shipped kernel's blocking rule at
/// once: q8_0's 32-element blocks, mlx2's 64-code uvec4 chunks, mlx6's 16-code
/// 3-word chunks, mlx4-repack's 32-nibble uvec4 chunks, and the `vec4`-typed
/// activation bindings (k%4==0).
const K: usize = 128;
/// Output rows. Small, and >1 so a row-indexing bug cannot hide.
const N: usize = 16;

/// Relative-error bound. See the module docs for the derivation.
const TOL: f64 = 1e-5;

// ─── deterministic data ──────────────────────────────────────────────────────

/// Numerical Recipes LCG. Deterministic across platforms and runs; deliberately
/// not an RNG crate (this file must add no dependency to the CI slice).
struct Lcg(u32);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    /// Uniform in [-1, 1).
    fn f32(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) / 8_388_608.0 - 1.0
    }
    /// Uniform integer in [0, m).
    fn below(&mut self, m: u32) -> u32 {
        self.next_u32() % m
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Reinterpret a byte stream as little-endian u32 words (the `uint[]` /
/// `uvec4[]` bindings that the fp8 and nvfp4-e4m3 kernels index by absolute
/// byte). `bytes.len()` must be a multiple of 4.
fn bytes_as_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ─── plan ────────────────────────────────────────────────────────────────────

/// One kernel dispatch plus the CPU answer it must reproduce.
struct Plan {
    /// Storage buffers in binding order. `None` marks the writeonly output slot
    /// (which is NOT always last — the batched MoE kernels bind `meta` after
    /// `dst`).
    bindings: Vec<Option<Vec<u8>>>,
    /// f32 elements in the output buffer.
    out_len: usize,
    push_constants: Vec<u8>,
    workgroups: (u32, u32, u32),
    /// Expected output, accumulated in f64 (an order-independent stand-in for
    /// "the exact value of the dequantized f32 matvec").
    reference: Vec<f64>,
    /// Per-output Σ_j |w·x|, the forward-error denominator.
    abs_scale: Vec<f64>,
}

/// Reference matvec over an already-dequantized row-major `[n, k]` weight.
/// Returns (dot products, Σ|products|), both f64.
fn ref_matvec(deq: &[f32], x: &[f32], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut dots = Vec::with_capacity(n);
    let mut abs = Vec::with_capacity(n);
    for r in 0..n {
        let (mut d, mut a) = (0.0f64, 0.0f64);
        for j in 0..k {
            let p = deq[r * k + j] as f64 * x[j] as f64;
            d += p;
            a += p.abs();
        }
        dots.push(d);
        abs.push(a);
    }
    (dots, abs)
}

// ─── packing helpers (layouts defined by the CPU dequantizers) ───────────────

/// Pack `codes` as a contiguous little-endian bitstream of `bits`-wide fields —
/// the MLX layout that `model::dequantize_mlx_affine_bits` decodes and that the
/// mlx2/mlx6/mlx8 shaders unpack. A code may straddle a u32 boundary (6-bit).
fn pack_contiguous(codes: &[u32], bits: usize) -> Vec<u32> {
    let nwords = (codes.len() * bits).div_ceil(32);
    let mut out = vec![0u32; nwords];
    for (i, &c) in codes.iter().enumerate() {
        let bit = i * bits;
        let (wi, off) = (bit / 32, bit % 32);
        out[wi] |= ((c as u64) << off) as u32;
        if off + bits > 32 {
            out[wi + 1] |= (((c as u64) << off) >> 32) as u32;
        }
    }
    out
}

/// Two 4-bit codes per byte, low nibble = even index (compressed-tensors
/// `pack_fp4_to_uint8`, which `model::dequantize_nvfp4` decodes).
fn pack_nibbles(codes: &[u32]) -> Vec<u8> {
    codes
        .chunks_exact(2)
        .map(|c| (c[0] as u8 & 0xF) | ((c[1] as u8 & 0xF) << 4))
        .collect()
}

/// A positive OCP-E4M3 byte with exponent in [5, 9] — magnitudes 0.25..8.0, the
/// range real NVFP4 block scales live in, and safely clear of the NaN code.
fn e4m3_positive(rng: &mut Lcg) -> u8 {
    let e = 5 + rng.below(5);
    let m = rng.below(8);
    ((e << 3) | m) as u8
}

// ─── plan builders ───────────────────────────────────────────────────────────

/// The mlx4 affine 4-bit family: `packed | scales | biases | x | dst`, push
/// constants `{k, n, group_size, packed_off, sb_off}`. Shared verbatim by the
/// scalar, `_cols`, `w8`, `w16`, `w8sg` and `repack` kernels — they differ only
/// in vectorization and reduction, never in layout or math.
fn plan_mlx4(seed: u32, gs: usize) -> Plan {
    let mut rng = Lcg(seed);
    let w: Vec<f32> = (0..N * K).map(|_| rng.f32()).collect();
    let x: Vec<f32> = (0..K).map(|_| rng.f32()).collect();

    let (packed, scales, biases) = model::quantize_mlx_affine_4bit(&w, N, K, gs);
    let deq = model::dequantize_mlx_affine(&packed, &scales, &biases, N, K, gs, 4);
    let (reference, abs_scale) = ref_matvec(&deq, &x, N, K);

    Plan {
        bindings: vec![
            Some(u32_bytes(&packed)),
            Some(f32_bytes(&scales)),
            Some(f32_bytes(&biases)),
            Some(f32_bytes(&x)),
            None,
        ],
        out_len: N,
        push_constants: u32_bytes(&[K as u32, N as u32, gs as u32, 0, 0]),
        workgroups: (N as u32, 1, 1),
        reference,
        abs_scale,
    }
}

/// The generic MLX affine bit-width family (2 / 6 / 8 bit): same five bindings
/// and push constants as mlx4, but the codes are a contiguous bitstream and
/// there is no CPU *quantizer* for these widths — the packing is defined by
/// `model::dequantize_mlx_affine_bits`, so we synthesize codes + affine params
/// and use that function as the reference.
fn plan_mlx_bits(seed: u32, bits: usize, gs: usize) -> Plan {
    let mut rng = Lcg(seed);
    let groups = K / gs;
    let hi = 1u32 << bits;
    let codes: Vec<u32> = (0..N * K).map(|_| rng.below(hi)).collect();
    // Per-row contiguous packing: each row is its own bitstream of K*bits bits.
    let mut packed = Vec::with_capacity(N * K * bits / 32);
    for r in 0..N {
        packed.extend(pack_contiguous(&codes[r * K..(r + 1) * K], bits));
    }
    // Scales scaled by 1/hi so dequantized weights land in a normal O(1) range.
    let scales: Vec<f32> = (0..N * groups)
        .map(|_| (rng.f32().abs() + 0.05) / hi as f32)
        .collect();
    let biases: Vec<f32> = (0..N * groups).map(|_| rng.f32() * 0.5).collect();
    let x: Vec<f32> = (0..K).map(|_| rng.f32()).collect();

    let deq = model::dequantize_mlx_affine_bits(&packed, &scales, &biases, N, K, gs, bits);
    let (reference, abs_scale) = ref_matvec(&deq, &x, N, K);

    Plan {
        bindings: vec![
            Some(u32_bytes(&packed)),
            Some(f32_bytes(&scales)),
            Some(f32_bytes(&biases)),
            Some(f32_bytes(&x)),
            None,
        ],
        out_len: N,
        push_constants: u32_bytes(&[K as u32, N as u32, gs as u32, 0, 0]),
        workgroups: (N as u32, 1, 1),
        reference,
        abs_scale,
    }
}

/// NVFP4 (E2M1 codes + per-group e4m3 block scale + per-tensor global).
///
/// `folded == true`  → `packed | scales(f32 = e4m3*global) | x | dst`, 5 uints.
/// `folded == false` → `packed | scaleb(raw e4m3 bytes) | x | dst`, 5 uints plus
///                     the f32 `global`, which the kernel folds itself.
/// Both are checked against the single CPU reference `model::dequantize_nvfp4`.
fn plan_nvfp4(seed: u32, folded: bool) -> Plan {
    const GS: usize = 16;
    let mut rng = Lcg(seed);
    let groups = K / GS;
    let codes: Vec<u32> = (0..N * K).map(|_| rng.below(16)).collect();
    let packed_bytes = pack_nibbles(&codes);
    let wscale: Vec<u8> = (0..N * groups).map(|_| e4m3_positive(&mut rng)).collect();
    let global = 0.375f32;
    let x: Vec<f32> = (0..K).map(|_| rng.f32()).collect();

    let deq = model::dequantize_nvfp4(&packed_bytes, &wscale, global, N, K, GS);
    let (reference, abs_scale) = ref_matvec(&deq, &x, N, K);

    let (scale_binding, push_constants) = if folded {
        let folded_scales: Vec<f32> = wscale
            .iter()
            .map(|&b| model::e4m3_to_f32(b) * global)
            .collect();
        (
            f32_bytes(&folded_scales),
            u32_bytes(&[K as u32, N as u32, GS as u32, 0, 0]),
        )
    } else {
        let mut pc = u32_bytes(&[K as u32, N as u32, GS as u32, 0, 0]);
        pc.extend_from_slice(&global.to_le_bytes());
        (u32_bytes(&bytes_as_words(&wscale)), pc)
    };

    Plan {
        bindings: vec![
            Some(u32_bytes(&bytes_as_words(&packed_bytes))),
            Some(scale_binding),
            Some(f32_bytes(&x)),
            None,
        ],
        out_len: N,
        push_constants,
        workgroups: (N as u32, 1, 1),
        reference,
        abs_scale,
    }
}

/// FP8 E4M3 W8A16: one byte per weight, per-OUTPUT-ROW f32 scale
/// (`scale_per_row = 1`, the harder of the two modes — a broadcast bug in the
/// scalar mode would still read row 0). Bindings `packed | scale | x | dst`.
/// Every one of the 256 e4m3 codes is exercised, NaN guard included.
fn plan_fp8(seed: u32) -> Plan {
    let mut rng = Lcg(seed);
    let wbytes: Vec<u8> = (0..N * K).map(|_| rng.below(256) as u8).collect();
    let scales: Vec<f32> = (0..N).map(|_| (rng.f32().abs() + 0.05) * 0.01).collect();
    let x: Vec<f32> = (0..K).map(|_| rng.f32()).collect();

    let deq = model::dequantize_fp8(&wbytes, &scales, N, K);
    let (reference, abs_scale) = ref_matvec(&deq, &x, N, K);

    Plan {
        bindings: vec![
            Some(u32_bytes(&bytes_as_words(&wbytes))),
            Some(f32_bytes(&scales)),
            Some(f32_bytes(&x)),
            None,
        ],
        out_len: N,
        // {ncols, nrows, scale_per_row, packed_off, sb_off}
        push_constants: u32_bytes(&[K as u32, N as u32, 1, 0, 0]),
        workgroups: (N as u32, 1, 1),
        reference,
        abs_scale,
    }
}

/// GGUF q8_0 (f16 scale + 32 int8 per block), column-batched kernel at
/// NUM_COLS=1. Bindings `data_a | x | dst`, push constants `{ncols, nrows}`.
fn plan_q8_0(seed: u32) -> Plan {
    let mut rng = Lcg(seed);
    let w: Vec<f32> = (0..N * K).map(|_| rng.f32()).collect();
    let x: Vec<f32> = (0..K).map(|_| rng.f32()).collect();

    let q = model::quantize_q8_0(&w);
    let deq = model::dequant_q8_0_to_f32(&q);
    let (reference, abs_scale) = ref_matvec(&deq, &x, N, K);

    Plan {
        bindings: vec![Some(q), Some(f32_bytes(&x)), None],
        out_len: N,
        push_constants: u32_bytes(&[K as u32, N as u32]),
        workgroups: (N as u32, 1, 1),
        reference,
        abs_scale,
    }
}

/// f16 weights, f32 activation. Bindings `w | x | dst`, `{ncols, nrows}`.
fn plan_f16(seed: u32) -> Plan {
    let mut rng = Lcg(seed);
    let w: Vec<half::f16> = (0..N * K).map(|_| half::f16::from_f32(rng.f32())).collect();
    let x: Vec<f32> = (0..K).map(|_| rng.f32()).collect();
    let deq: Vec<f32> = w.iter().map(|h| h.to_f32()).collect();
    let (reference, abs_scale) = ref_matvec(&deq, &x, N, K);

    Plan {
        bindings: vec![
            Some(w.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect()),
            Some(f32_bytes(&x)),
            None,
        ],
        out_len: N,
        push_constants: u32_bytes(&[K as u32, N as u32]),
        workgroups: (N as u32, 1, 1),
        reference,
        abs_scale,
    }
}

/// The batched MoE twins (`*repack_batched`): E independent expert slices in one
/// dispatch, addressed through a `meta[]` uvec4 per expert
/// `(packed_off_words, sb_off_elems, x_off_floats, dst_off_floats)` with the
/// expert on the grid's Y axis. Bindings are
/// `packed | scales | biases | x | dst | meta` — note `dst` is binding 4, i.e.
/// the output is NOT the last slot, which is exactly why `Plan::bindings` uses
/// an explicit `None` hole rather than appending outputs at the end.
///
/// `bits == 4` builds the mlx4 twin (via the real `quantize_mlx_affine_4bit`),
/// `bits == 2` the mlx2 twin (synthesized codes + `dequantize_mlx_affine_bits`).
fn plan_batched(seed: u32, bits: usize, gs: usize) -> Plan {
    const E: usize = 2;
    let mut rng = Lcg(seed);
    let groups = K / gs;
    let words_per_row = K * bits / 32;

    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut x = Vec::new();
    let mut reference = Vec::new();
    let mut abs_scale = Vec::new();
    let mut meta: Vec<u32> = Vec::new();

    for e in 0..E {
        let deq: Vec<f32> = if bits == 4 {
            let w: Vec<f32> = (0..N * K).map(|_| rng.f32()).collect();
            let (p, s, b) = model::quantize_mlx_affine_4bit(&w, N, K, gs);
            let d = model::dequantize_mlx_affine(&p, &s, &b, N, K, gs, 4);
            packed.extend_from_slice(&p);
            scales.extend_from_slice(&s);
            biases.extend_from_slice(&b);
            d
        } else {
            let hi = 1u32 << bits;
            let codes: Vec<u32> = (0..N * K).map(|_| rng.below(hi)).collect();
            let mut p = Vec::new();
            for r in 0..N {
                p.extend(pack_contiguous(&codes[r * K..(r + 1) * K], bits));
            }
            let s: Vec<f32> = (0..N * groups)
                .map(|_| (rng.f32().abs() + 0.05) / hi as f32)
                .collect();
            let b: Vec<f32> = (0..N * groups).map(|_| rng.f32() * 0.5).collect();
            let d = model::dequantize_mlx_affine_bits(&p, &s, &b, N, K, gs, bits);
            packed.extend_from_slice(&p);
            scales.extend_from_slice(&s);
            biases.extend_from_slice(&b);
            d
        };
        let xe: Vec<f32> = (0..K).map(|_| rng.f32()).collect();
        let (dots, abs) = ref_matvec(&deq, &xe, N, K);
        x.extend_from_slice(&xe);
        reference.extend_from_slice(&dots);
        abs_scale.extend_from_slice(&abs);
        meta.extend_from_slice(&[
            (e * N * words_per_row) as u32, // packed_off (words)
            (e * N * groups) as u32,        // sb_off (elements)
            (e * K) as u32,                 // x_off (floats)
            (e * N) as u32,                 // dst_off (floats)
        ]);
    }

    Plan {
        bindings: vec![
            Some(u32_bytes(&packed)),
            Some(f32_bytes(&scales)),
            Some(f32_bytes(&biases)),
            Some(f32_bytes(&x)),
            None,
            Some(u32_bytes(&meta)),
        ],
        out_len: E * N,
        push_constants: u32_bytes(&[K as u32, N as u32, gs as u32, 0, 0]),
        workgroups: (N as u32, E as u32, 1),
        reference,
        abs_scale,
    }
}

// ─── dispatch + compare ──────────────────────────────────────────────────────

/// Upload the plan's buffers, run the real pipeline, read `dst` back, and
/// return the max relative error (see module docs) — or `Err` describing the
/// first output that exceeds `TOL`.
fn run_plan(engine: &mut ComputeEngine, name: &str, plan: &Plan) -> Result<f64, String> {
    let mut bufs: Vec<Buffer> = Vec::with_capacity(plan.bindings.len());
    for slot in &plan.bindings {
        match slot {
            Some(bytes) => {
                let b = engine.alloc_host_coherent_storage(bytes.len() as u64)?;
                b.write(bytes)?;
                bufs.push(b);
            }
            None => {
                let b = engine.alloc_host_coherent_storage((plan.out_len * 4) as u64)?;
                // Poison the output so a kernel that never writes a row is
                // caught as a mismatch rather than passing on a lucky zero.
                b.write(&f32_bytes(&vec![f32::NAN; plan.out_len]))?;
                bufs.push(b);
            }
        }
    }

    let refs: Vec<&Buffer> = bufs.iter().collect();
    engine.dispatch(name, &refs, &plan.push_constants, plan.workgroups)?;

    let out_idx = plan
        .bindings
        .iter()
        .position(|s| s.is_none())
        .ok_or_else(|| format!("{name}: plan has no output slot"))?;
    let mut raw = vec![0u8; plan.out_len * 4];
    bufs[out_idx].read(&mut raw)?;
    let got: Vec<f32> = bytes_as_words(&raw)
        .into_iter()
        .map(f32::from_bits)
        .collect();

    for b in bufs {
        engine.return_to_pool(b);
    }

    let mut worst = 0.0f64;
    let mut worst_at = 0usize;
    for r in 0..plan.out_len {
        if !got[r].is_finite() {
            return Err(format!(
                "{name}: output[{r}] = {} (non-finite; row never written or NaN-producing)",
                got[r]
            ));
        }
        // A degenerate all-zero row would divide by zero; floor the denominator
        // at the reference magnitude so the metric stays meaningful.
        let denom = plan.abs_scale[r].max(plan.reference[r].abs()).max(1e-30);
        let rel = (got[r] as f64 - plan.reference[r]).abs() / denom;
        if rel > worst {
            worst = rel;
            worst_at = r;
        }
    }
    if worst > TOL {
        return Err(format!(
            "{name}: rel_err {worst:.3e} > {TOL:.0e} at output[{worst_at}] \
             (gpu={:.9e} cpu={:.9e}, |w·x| scale={:.3e})",
            got[worst_at], plan.reference[worst_at], plan.abs_scale[worst_at]
        ));
    }
    Ok(worst)
}

// ─── the sweep ───────────────────────────────────────────────────────────────

/// Kernel name, whether it needs device fp16, and how to build its plan.
/// The names are the same registry names `registry_tests::REQUIRED_QUANT_KERNELS`
/// asserts are present — this file checks that they also compute the right
/// numbers. Kernels in that list with no entry here are accounted for in
/// `every_matvec_kernel_is_covered_or_excused`.
type PlanFn = fn() -> Plan;

const CASES: &[(&str, bool, PlanFn)] = &[
    // ── mlx4 affine 4-bit: five vectorizations of one layout ───────────────
    ("mul_mat_vec_mlx4_f32_f32", false, || plan_mlx4(0x1234_5678, 64)),
    ("mul_mat_vec_mlx4_cols", false, || plan_mlx4(0x1234_5678, 64)),
    ("mul_mat_vec_mlx4w8_f32_f32", false, || plan_mlx4(0x1234_5678, 64)),
    ("mul_mat_vec_mlx4w16_f32_f32", false, || plan_mlx4(0x1234_5678, 64)),
    ("mul_mat_vec_mlx4w8sg_f32_f32", false, || plan_mlx4(0x1234_5678, 64)),
    ("mul_mat_vec_mlx4repack_f32_f32", false, || plan_mlx4(0x1234_5678, 64)),
    // ── other MLX affine widths ────────────────────────────────────────────
    ("mul_mat_vec_mlx2repack_f32_f32", false, || plan_mlx_bits(0x0BAD_F00D, 2, 128)),
    ("mul_mat_vec_mlx6_f32_f32", false, || plan_mlx_bits(0x0BAD_F00D, 6, 128)),
    ("mul_mat_vec_mlx8_f32_f32", false, || plan_mlx_bits(0x0BAD_F00D, 8, 64)),
    // ── batched MoE twins (expert on grid.y, meta[] bases) ─────────────────
    ("mul_mat_vec_mlx4repack_batched_f32_f32", false, || plan_batched(0x00C0_FFEE, 4, 64)),
    ("mul_mat_vec_mlx2repack_batched_f32_f32", false, || plan_batched(0x00C0_FFEE, 2, 128)),
    // ── nvfp4 ──────────────────────────────────────────────────────────────
    ("mul_mat_vec_nvfp4_f32_f32", false, || plan_nvfp4(0x5EED_1111, true)),
    ("mul_mat_vec_nvfp4repack_f32_f32", false, || plan_nvfp4(0x5EED_1111, true)),
    ("mul_mat_vec_nvfp4_e4m3_f32_f32", false, || plan_nvfp4(0x5EED_1111, false)),
    ("mul_mat_vec_nvfp4_e4m3repack_f32_f32", false, || plan_nvfp4(0x5EED_1111, false)),
    // ── fp8 ────────────────────────────────────────────────────────────────
    ("mul_mat_vec_fp8_f32_f32", false, || plan_fp8(0x2222_ABCD)),
    ("mul_mat_vec_fp8fast_f32_f32", false, || plan_fp8(0x2222_ABCD)),
    ("mul_mat_vec_fp8repack_f32_f32", false, || plan_fp8(0x2222_ABCD)),
    // ── column-batched dequant matvec (f16-typed bindings) ─────────────────
    ("mul_mat_vec_q8_0_cols", true, || plan_q8_0(0x3333_5555)),
    ("mul_mat_vec_f16_cols", true, || plan_f16(0x3333_5555)),
];

/// Kernels in `REQUIRED_QUANT_KERNELS` that this harness deliberately does not
/// cover, with the reason. Keeping the list here (rather than silently) means
/// `every_matvec_kernel_is_covered_or_excused` fails the moment a new kernel
/// lands without either a plan or an explicit excuse.
const EXCUSED: &[(&str, &str)] = &[
    (
        "paged_attn_decode_f32_sg",
        "paged attention, not a matvec: needs a populated KV page table + \
         block tables, and is wave64-baked (pipeline.rs skips it on subgroup!=64)",
    ),
    (
        "paged_attn_decode_f16_sg",
        "as paged_attn_decode_f32_sg, plus f16 KV storage",
    ),
    (
        "relu2_f32",
        "elementwise activation, not a quantized matvec; no dequant reference",
    ),
];

struct Report {
    ran: Vec<(String, f64)>,
    skipped: Vec<(String, String)>,
}

fn sweep(engine: &mut ComputeEngine, fp16: bool) -> Report {
    let mut ran = Vec::new();
    let mut skipped = Vec::new();
    for &(name, needs_fp16, build) in CASES {
        if !engine.has_pipeline(name) {
            skipped.push((
                name.to_string(),
                "pipeline not compiled on this device (see PipelineCache::new)".to_string(),
            ));
            continue;
        }
        if needs_fp16 && !fp16 {
            skipped.push((
                name.to_string(),
                "device lacks storage_buffer_16_bit_access && shader_float16".to_string(),
            ));
            continue;
        }
        let plan = build();
        match run_plan(engine, name, &plan) {
            Ok(err) => ran.push((name.to_string(), err)),
            Err(e) => panic!("GPU conformance FAILED — {e}"),
        }
    }
    Report { ran, skipped }
}

// ─── tests ───────────────────────────────────────────────────────────────────

/// Build a `ComputeEngine` on device 0, or `None` when the machine has no
/// Vulkan ICD (the default CI job). This is the single place the whole module
/// no-ops through.
fn engine_or_skip() -> Option<(ComputeEngine, bool, u32)> {
    if !device::is_vulkan_available() {
        eprintln!(
            "device_conformance: no Vulkan device (no ICD) — GPU conformance SKIPPED. \
             Point VK_ICD_FILENAMES at a driver (e.g. Mesa lavapipe \
             /usr/share/vulkan/icd.d/lvp_icd.x86_64.json) to run it."
        );
        return None;
    }
    let dev = device::ComputeDevice::create(0).expect("Vulkan reported available but device 0 failed to create");
    let (fp16, subgroup) = (dev.fp16, dev.subgroup_size);
    let spvs = include_all_shaders();
    let refs: std::collections::HashMap<&str, &[u8]> =
        spvs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
    let engine = ComputeEngine::new(
        dev.instance.clone(),
        dev.physical_device,
        dev.device.clone(),
        dev.compute_queue,
        dev.compute_queue_family,
        dev.caps(),
        &refs,
    )
    .expect("ComputeEngine::new failed on an available device");
    Some((engine, fp16, subgroup))
}

/// Every quant matvec kernel this branch ships, run for real against the CPU
/// dequant reference. No-ops (PASS) with no device.
#[test]
fn quant_matvec_kernels_match_cpu_reference() {
    let Some((mut engine, fp16, subgroup)) = engine_or_skip() else { return };
    eprintln!("device_conformance: subgroup_size={subgroup} fp16={fp16} k={K} n={N} tol={TOL:e}");

    let report = sweep(&mut engine, fp16);

    for (name, err) in &report.ran {
        eprintln!("  RAN  {name:<42} max_rel_err={err:.3e}");
    }
    for (name, why) in &report.skipped {
        eprintln!("  SKIP {name:<42} {why}");
    }
    for (name, why) in EXCUSED {
        eprintln!("  N/A  {name:<42} {why}");
    }
    eprintln!(
        "device_conformance: {} kernel(s) ran, {} skipped, {} out of scope",
        report.ran.len(),
        report.skipped.len(),
        EXCUSED.len()
    );

    assert!(
        !report.ran.is_empty(),
        "a Vulkan device was present but not one quant matvec kernel ran — \
         every pipeline was missing. That is a packaging/compile regression, \
         not a device limitation."
    );
}

/// Falsifiability: the comparator must REJECT a wrong answer.
///
/// We run a genuine mlx4 dispatch but hand `run_plan` a reference whose group-0
/// scale has been perturbed by 1e-3 relative — a change far smaller than any
/// realistic kernel bug, and far larger than TOL. If this ever returns `Ok`,
/// every other assertion in this file is worthless.
#[test]
fn harness_detects_a_corrupted_reference() {
    let Some((mut engine, _fp16, _sg)) = engine_or_skip() else { return };
    const NAME: &str = "mul_mat_vec_mlx4_f32_f32";
    if !engine.has_pipeline(NAME) {
        eprintln!("device_conformance: {NAME} not compiled here — fail-injection SKIPPED");
        return;
    }

    // Sanity: the untouched plan must pass, so a rejection below is attributable
    // to the perturbation and not to a broken kernel or a mis-sized buffer.
    let clean = plan_mlx4(0x1234_5678, 64);
    let clean_err = run_plan(&mut engine, NAME, &clean).expect("clean mlx4 plan should pass");

    // Rebuild with one perturbed affine scale and recompute the reference the
    // same way the honest builder does.
    let mut rng = Lcg(0x1234_5678);
    let w: Vec<f32> = (0..N * K).map(|_| rng.f32()).collect();
    let x: Vec<f32> = (0..K).map(|_| rng.f32()).collect();
    let (packed, mut scales, biases) = model::quantize_mlx_affine_4bit(&w, N, K, 64);
    scales[0] *= 1.0 + 1e-3; // <- the injected fault (reference only; GPU sees the true bytes)
    let bad_deq = model::dequantize_mlx_affine(&packed, &scales, &biases, N, K, 64, 4);
    let (reference, abs_scale) = ref_matvec(&bad_deq, &x, N, K);
    let corrupted = Plan { reference, abs_scale, ..clean };

    let err = run_plan(&mut engine, NAME, &corrupted)
        .expect_err("harness accepted a knowingly-wrong reference — the comparator is inert");
    eprintln!(
        "device_conformance: fail-injection OK (clean rel_err={clean_err:.3e}); rejected with: {err}"
    );
}

/// Structural guard: every kernel in `REQUIRED_QUANT_KERNELS` must either have a
/// conformance plan or an explicit written excuse. Pure bookkeeping — runs with
/// or without a device, so a new kernel landing without numerical coverage is
/// caught in the ICD-less CI job too.
#[test]
fn every_matvec_kernel_is_covered_or_excused() {
    let uncovered: Vec<&str> = crate::registry_tests::REQUIRED_QUANT_KERNELS
        .iter()
        .copied()
        .filter(|n| {
            !CASES.iter().any(|(c, _, _)| c == n) && !EXCUSED.iter().any(|(e, _)| e == n)
        })
        .collect();
    assert!(
        uncovered.is_empty(),
        "{} shipped kernel(s) have neither a device_conformance plan nor an \
         entry in EXCUSED: {uncovered:?}. Add a Plan builder, or an EXCUSED \
         line saying why a numerical check is not possible.",
        uncovered.len()
    );
}
