//! DeepSeek-V4-Flash — DSA (Deep Sparse Attention) producers + RoPE derivation.
//!
//! This is M1 item 1: the CPU mirror of the compressor / Lightning-Indexer that
//! produces `compressed_kv` + `block_bias` for [`crate::dsv4::mla_core_ext`] (the
//! CSA/HCA layers). Mirrors the transformers-5.8.1 `DeepseekV4CSACompressor`,
//! `DeepseekV4HCACompressor` and `DeepseekV4Indexer` reference modules for the
//! stateless (`past_key_values is None`) prefill path — the case the M1 single-shot
//! full-forward gate runs. Validated bit-exact vs fixtures captured DIRECTLY from
//! those reference modules (`scripts/dsv4/dump_dsa_fixture.py` → `dsa_{csa,hca}.json`),
//! and the RoPE derivation vs `rope.json`.
//!
//! Conventions: everything `[rows, dim]` row-major, B=1. Attention/o helpers and the
//! interleaved partial-RoPE live in [`crate::dsv4`]; this module reuses them.

use crate::dsv4::{apply_interleaved_rope_inplace, rmsnorm_rows};
use crate::model::cpu_matmul;

// ============================ RoPE derivation ============================

/// "main" (plain) partial-RoPE inverse frequencies. `dim = int(head_dim*partial)`,
/// `inv_freq[i] = base^-(2i/dim)` for `i in 0..dim/2`. (sliding_attention layers.)
pub fn inv_freq_main(head_dim: usize, partial: f64, theta: f64) -> Vec<f32> {
    let dim = (head_dim as f64 * partial) as usize;
    (0..dim / 2)
        .map(|i| (1.0 / theta.powf((2 * i) as f64 / dim as f64)) as f32)
        .collect()
}

/// YaRN inverse frequencies for the "compress" rope (CSA/HCA layers + their
/// compressors). Clean-room port of the transformers `_compute_yarn_parameters`
/// interpolation (validated cos=1.0 vs the reference buffer). `dim` is the rope
/// dimension `int(head_dim*partial)` (== `qk_rope_head_dim`).
#[allow(clippy::too_many_arguments)]
pub fn inv_freq_yarn(
    head_dim: usize,
    partial: f64,
    theta: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    orig_max: f64,
) -> Vec<f32> {
    let dim = (head_dim as f64 * partial) as usize;
    let n = dim / 2;
    // correction range in dim-index units
    let corr_dim = |num_rot: f64| -> f64 {
        dim as f64 * (orig_max / (num_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln())
    };
    let mut low = corr_dim(beta_fast).floor();
    let mut high = corr_dim(beta_slow).ceil();
    low = low.max(0.0);
    high = high.min((dim - 1) as f64);
    if (low - high).abs() < f64::EPSILON {
        high += 0.001;
    }
    (0..n)
        .map(|i| {
            let pos_freq = theta.powf((2 * i) as f64 / dim as f64);
            let inv_extra = 1.0 / pos_freq;
            let inv_interp = 1.0 / (factor * pos_freq);
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let mask = 1.0 - ramp; // inv_freq_mask
            (inv_interp * (1.0 - mask) + inv_extra * mask) as f32
        })
        .collect()
}

/// cos/sin `[P, half]` row-major for the given positions: `cos[p,i] =
/// cos(positions[p]*inv_freq[i]) * scaling`. Math in f64 (reference `.float()`).
pub fn rope_cos_sin(positions: &[usize], inv_freq: &[f32], scaling: f32) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos = vec![0.0f32; positions.len() * half];
    let mut sin = vec![0.0f32; positions.len() * half];
    for (p, &pos) in positions.iter().enumerate() {
        for i in 0..half {
            let f = pos as f64 * inv_freq[i] as f64;
            cos[p * half + i] = (f.cos() * scaling as f64) as f32;
            sin[p * half + i] = (f.sin() * scaling as f64) as f32;
        }
    }
    (cos, sin)
}

// ============================ softmax helper ============================

/// In-place softmax over `len` values that may contain `NEG_INFINITY` (→ weight 0).
/// f64 accumulation, max-subtracted. All-`-inf` rows yield all-zeros.
fn softmax_masked(v: &mut [f64]) {
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !mx.is_finite() {
        for x in v.iter_mut() {
            *x = 0.0;
        }
        return;
    }
    let mut den = 0.0f64;
    for x in v.iter_mut() {
        *x = (*x - mx).exp();
        den += *x;
    }
    for x in v.iter_mut() {
        *x /= den;
    }
}

// ============================ Ca/Cb window pooling ============================

/// Shared Ca/Cb overlapping-window pooling used by the CSA compressor (at
/// `head_dim`) and the CSA indexer (at `index_head_dim`), and — with `two_series=
/// false` — the simple non-overlapping HCA pooling.
///
/// Inputs: `kv`/`gate` are `[S, proj]` where `proj = (if two_series {2} else {1})*hd`.
/// `position_bias` is `[m, proj]`. Produces the pre-RoPE compressed entries
/// `[n_win, hd]` (weighted-RMSNorm applied). `n_win = (S/m)`.
#[allow(clippy::too_many_arguments)]
fn window_pool(
    kv: &[f32],
    gate: &[f32],
    s: usize,
    proj: usize,
    hd: usize,
    m: usize,
    position_bias: &[f32],
    kv_norm: &[f32],
    eps: f32,
    two_series: bool,
) -> Vec<f32> {
    let usable = (s / m) * m;
    let n_win = usable / m;
    if n_win == 0 {
        return Vec::new();
    }
    // new_kv[n_win, slots, hd], new_gate[.. , -inf default]; slots = 2m (Ca/Cb) or m.
    let slots = if two_series { 2 * m } else { m };
    // Accumulate the weighted sum per (window, hd) directly.
    let mut pooled = vec![0.0f32; n_win * hd]; // pre-norm
    for w in 0..n_win {
        // Build [slots, hd] kv + gate for this window.
        let mut nk = vec![0.0f32; slots * hd];
        let mut ng = vec![f64::NEG_INFINITY; slots * hd];
        if two_series {
            // second half (Cb) = current window's [..., hd:]
            for j in 0..m {
                let src = (w * m + j) * proj;
                for d in 0..hd {
                    nk[(m + j) * hd + d] = kv[src + hd + d];
                    ng[(m + j) * hd + d] =
                        (gate[src + hd + d] + position_bias[j * proj + hd + d]) as f64;
                }
            }
            // first half (Ca) = previous window's [..., :hd]; window 0 stays 0/-inf.
            if w >= 1 {
                for j in 0..m {
                    let src = ((w - 1) * m + j) * proj;
                    for d in 0..hd {
                        nk[j * hd + d] = kv[src + d];
                        ng[j * hd + d] = (gate[src + d] + position_bias[j * proj + d]) as f64;
                    }
                }
            }
        } else {
            // simple: one series over the m window tokens.
            for j in 0..m {
                let src = (w * m + j) * proj;
                for d in 0..hd {
                    nk[j * hd + d] = kv[src + d];
                    ng[j * hd + d] = (gate[src + d] + position_bias[j * proj + d]) as f64;
                }
            }
        }
        // per-channel softmax over the slots axis, then weighted sum.
        for d in 0..hd {
            let mut col: Vec<f64> = (0..slots).map(|sl| ng[sl * hd + d]).collect();
            softmax_masked(&mut col);
            let mut acc = 0.0f64;
            for sl in 0..slots {
                acc += col[sl] * nk[sl * hd + d] as f64;
            }
            pooled[w * hd + d] = acc as f32;
        }
    }
    // weighted RMSNorm over hd.
    rmsnorm_rows(&pooled, kv_norm, n_win, hd, eps)
}

// ============================ HCA compressor ============================

/// HCA (heavily_compressed_attention) compressor — stateless prefill.
/// Returns `(compressed_kv [T,hd], block_bias_vis [S,T])` where `block_bias_vis`
/// is 1=visible / 0=masked (caller maps to 0.0 / -inf additive). `T = S/m`.
#[allow(clippy::too_many_arguments)]
pub fn hca_compressor(
    hs: &[f32],
    s: usize,
    h: usize,
    hd: usize,
    m: usize,
    positions: &[usize],
    kv_proj: &[f32],   // [hd, H]
    gate_proj: &[f32], // [hd, H]
    position_bias: &[f32], // [m, hd]
    kv_norm: &[f32],   // [hd]
    eps: f32,
    inv_freq: &[f32],
    scaling: f32,
) -> (Vec<f32>, Vec<i32>) {
    let kv = cpu_matmul(hs, kv_proj, s, h, hd); // [S, hd]
    let gate = cpu_matmul(hs, gate_proj, s, h, hd); // [S, hd]
    let pooled = window_pool(&kv, &gate, s, hd, hd, m, position_bias, kv_norm, eps, false);
    let n_win = pooled.len() / hd;
    let mut compressed = pooled;
    if n_win > 0 {
        let win_pos: Vec<usize> = (0..n_win).map(|w| w * m).collect();
        let (cos, sin) = rope_cos_sin(&win_pos, inv_freq, scaling);
        let rope_dim = 2 * inv_freq.len();
        apply_interleaved_rope_inplace(&mut compressed, n_win, hd, rope_dim, &|w| w, &cos, &sin);
    }
    let bias = causal_block_bias(s, n_win, m, positions);
    (compressed, bias)
}

/// Causal-only visibility `[S, T]`: query `s` sees compressed entry `w` iff
/// `w < (positions[s]+1)/m`. (HCA has no indexer.)
fn causal_block_bias(s: usize, n_win: usize, m: usize, positions: &[usize]) -> Vec<i32> {
    let mut vis = vec![0i32; s * n_win];
    for si in 0..s {
        let threshold = (positions[si] + 1) / m;
        for w in 0..n_win {
            vis[si * n_win + w] = i32::from(w < threshold);
        }
    }
    vis
}

// ============================ CSA compressor + indexer ============================

/// Weights for the CSA Lightning-Indexer (index_head_dim path).
pub struct IndexerWeights<'a> {
    pub kv_proj: &'a [f32],      // [2*ihd, H]
    pub gate_proj: &'a [f32],    // [2*ihd, H]
    pub position_bias: &'a [f32],// [m, 2*ihd]
    pub kv_norm: &'a [f32],      // [ihd]
    pub q_b_proj: &'a [f32],     // [nh_ix*ihd, q_lora]
    pub weights_proj: &'a [f32], // [nh_ix, H]
}

/// CSA (compressed_sparse_attention) compressor + Lightning-Indexer — stateless
/// prefill. Returns `(compressed_kv [T,hd], block_bias_vis [S,T])`. `T = S/m`.
/// `block_bias_vis[s,w] = 1` iff query `s` selected compressed entry `w` (a
/// causally-valid indexer top-k pick), else 0.
#[allow(clippy::too_many_arguments)]
pub fn csa_compressor(
    hs: &[f32],
    q_residual: &[f32],
    s: usize,
    h: usize,
    hd: usize,
    m: usize,
    positions: &[usize],
    kv_proj: &[f32],       // [2*hd, H]
    gate_proj: &[f32],     // [2*hd, H]
    position_bias: &[f32], // [m, 2*hd]
    kv_norm: &[f32],       // [hd]
    eps: f32,
    inv_freq: &[f32],
    scaling: f32,
    q_lora: usize,
    ix_nh: usize,
    ix_hd: usize,
    index_topk: usize,
    ix: &IndexerWeights,
) -> (Vec<f32>, Vec<i32>) {
    // ---- outer compressor (head_dim, Ca/Cb) ----
    let kv = cpu_matmul(hs, kv_proj, s, h, 2 * hd); // [S, 2*hd]
    let gate = cpu_matmul(hs, gate_proj, s, h, 2 * hd);
    let pooled = window_pool(&kv, &gate, s, 2 * hd, hd, m, position_bias, kv_norm, eps, true);
    let n_win = pooled.len() / hd;
    let mut compressed = pooled;
    let win_pos: Vec<usize> = (0..n_win).map(|w| w * m).collect();
    if n_win > 0 {
        let (cos, sin) = rope_cos_sin(&win_pos, inv_freq, scaling);
        let rope_dim = 2 * inv_freq.len();
        apply_interleaved_rope_inplace(&mut compressed, n_win, hd, rope_dim, &|w| w, &cos, &sin);
    }

    // ---- indexer: compressed keys at index_head_dim ----
    let top_idx = if n_win == 0 {
        vec![]
    } else {
        indexer_topk(
            hs, q_residual, s, h, m, positions, &win_pos, n_win, q_lora, ix_nh, ix_hd,
            index_topk, eps, inv_freq, scaling, ix,
        )
    };

    // ---- block_bias from top-k picks (valid picks visible) ----
    let mut vis = vec![0i32; s * n_win];
    let top_k = index_topk.min(n_win);
    for si in 0..s {
        for kk in 0..top_k {
            let idx = top_idx[si * top_k + kk];
            if idx >= 0 {
                vis[si * n_win + idx as usize] = 1;
            }
        }
    }
    (compressed, vis)
}

/// Lightning-Indexer top-k selection → `[S, top_k]` indices (`-1` = invalid).
#[allow(clippy::too_many_arguments)]
fn indexer_topk(
    hs: &[f32],
    q_residual: &[f32],
    s: usize,
    h: usize,
    m: usize,
    positions: &[usize],
    win_pos: &[usize],
    n_win: usize,
    q_lora: usize,
    ix_nh: usize,
    ix_hd: usize,
    index_topk: usize,
    eps: f32,
    inv_freq: &[f32],
    scaling: f32,
    ix: &IndexerWeights,
) -> Vec<i32> {
    let rope_dim = 2 * inv_freq.len();
    // compressed keys [n_win, ix_hd]
    let kv = cpu_matmul(hs, ix.kv_proj, s, h, 2 * ix_hd);
    let gate = cpu_matmul(hs, ix.gate_proj, s, h, 2 * ix_hd);
    let pooled = window_pool(&kv, &gate, s, 2 * ix_hd, ix_hd, m, ix.position_bias, ix.kv_norm, eps, true);
    let mut ck = pooled; // [n_win, ix_hd]
    let (cos_w, sin_w) = rope_cos_sin(win_pos, inv_freq, scaling);
    apply_interleaved_rope_inplace(&mut ck, n_win, ix_hd, rope_dim, &|w| w, &cos_w, &sin_w);

    // queries [S, ix_nh, ix_hd] then rope at token positions.
    let q_flat = cpu_matmul(q_residual, ix.q_b_proj, s, q_lora, ix_nh * ix_hd); // [S, nh*hd]
    // layout rows [S*nh, hd], row r=(si*nh+hh), pos = positions[si]
    let mut q = vec![0.0f32; s * ix_nh * ix_hd];
    q.copy_from_slice(&q_flat); // already [S, nh*hd] == [S*nh, hd] row-major
    let (cos_q, sin_q) = rope_cos_sin(positions, inv_freq, scaling);
    apply_interleaved_rope_inplace(&mut q, s * ix_nh, ix_hd, rope_dim, &|r| r / ix_nh, &cos_q, &sin_q);

    // per-head weights = (hs @ weights_proj^T) * nh^-0.5   [S, nh]
    let wgt = cpu_matmul(hs, ix.weights_proj, s, h, ix_nh); // [S, nh]
    let w_scale = (ix_nh as f64).powf(-0.5);
    let softmax_scale = (ix_hd as f64).powf(-0.5);

    // index_scores[s, w] = sum_h relu(q[s,h]·ck[w]) * softmax_scale * (wgt[s,h]*w_scale)
    let top_k = index_topk.min(n_win);
    let mut out = vec![-1i32; s * top_k];
    for si in 0..s {
        let threshold = (positions[si] + 1) / m;
        let mut scores = vec![f64::NEG_INFINITY; n_win];
        for w in 0..n_win {
            if w >= threshold {
                continue; // future_mask → -inf
            }
            let mut acc = 0.0f64;
            for hh in 0..ix_nh {
                let qrow = &q[(si * ix_nh + hh) * ix_hd..(si * ix_nh + hh) * ix_hd + ix_hd];
                let krow = &ck[w * ix_hd..w * ix_hd + ix_hd];
                let mut dot = 0.0f64;
                for d in 0..ix_hd {
                    dot += qrow[d] as f64 * krow[d] as f64;
                }
                let relu = if dot > 0.0 { dot } else { 0.0 } * softmax_scale;
                let wv = wgt[si * ix_nh + hh] as f64 * w_scale;
                acc += relu * wv;
            }
            scores[w] = acc;
        }
        // top-k by score (descending). Ties: torch.topk is unstable but the block_bias
        // is a SET so order among equal picks does not matter for the mask.
        let mut order: Vec<usize> = (0..n_win).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
        for kk in 0..top_k {
            let w = order[kk];
            // invalid if score is -inf (never selectable) or pick >= threshold.
            if scores[w].is_finite() && w < threshold {
                out[si * top_k + kk] = w as i32;
            } else {
                out[si * top_k + kk] = -1;
            }
        }
    }
    out
}

// ==================== LEVER #2: resident-projection variants ====================
// Twins of `hca_compressor` / `csa_compressor` / `indexer_topk` that take the
// projection RESULTS (`kv`/`gate`/`q_flat`/`wgt` = x @ W^T) precomputed by the
// caller — so the caller can run those matvecs on RESIDENT GPU buffers instead of
// re-dequantizing the 6-bit compressor/indexer weights from mmap every token (the
// measured 81%-of-decode host-orchestration sink). The pool/rope/topk/scoring math
// below is byte-identical to the originals; only the source of kv/gate/q/wgt moves
// from `cpu_matmul(hs, dequant(W))` to the caller's resident matvec.

/// Resident-projection HCA compressor. `kv`/`gate` are `[s, hd]` (== `hs @ W^T`).
#[allow(clippy::too_many_arguments)]
pub fn hca_compressor_pre(
    kv: &[f32],
    gate: &[f32],
    s: usize,
    hd: usize,
    m: usize,
    positions: &[usize],
    position_bias: &[f32], // [m, hd]
    kv_norm: &[f32],       // [hd]
    eps: f32,
    inv_freq: &[f32],
    scaling: f32,
) -> (Vec<f32>, Vec<i32>) {
    let pooled = window_pool(kv, gate, s, hd, hd, m, position_bias, kv_norm, eps, false);
    let n_win = pooled.len() / hd;
    let mut compressed = pooled;
    if n_win > 0 {
        let win_pos: Vec<usize> = (0..n_win).map(|w| w * m).collect();
        let (cos, sin) = rope_cos_sin(&win_pos, inv_freq, scaling);
        let rope_dim = 2 * inv_freq.len();
        apply_interleaved_rope_inplace(&mut compressed, n_win, hd, rope_dim, &|w| w, &cos, &sin);
    }
    let bias = causal_block_bias(s, n_win, m, positions);
    (compressed, bias)
}

/// Precomputed indexer projections (`= x @ W^T`) for [`indexer_topk_pre`].
pub struct IndexerProj<'a> {
    pub kv: &'a [f32],       // [s, 2*ix_hd]  (hs @ ix.kv_proj^T)
    pub gate: &'a [f32],     // [s, 2*ix_hd]  (hs @ ix.gate_proj^T)
    pub q_flat: &'a [f32],   // [s, ix_nh*ix_hd]  (q_residual @ ix.q_b_proj^T)
    pub wgt: &'a [f32],      // [s, ix_nh]  (hs @ ix.weights_proj^T)
    pub position_bias: &'a [f32], // [m, 2*ix_hd]
    pub kv_norm: &'a [f32],  // [ix_hd]
}

/// Resident-projection Lightning-Indexer top-k. Scoring identical to `indexer_topk`.
#[allow(clippy::too_many_arguments)]
fn indexer_topk_pre(
    s: usize,
    m: usize,
    positions: &[usize],
    win_pos: &[usize],
    n_win: usize,
    ix_nh: usize,
    ix_hd: usize,
    index_topk: usize,
    eps: f32,
    inv_freq: &[f32],
    scaling: f32,
    p: &IndexerProj,
) -> Vec<i32> {
    let rope_dim = 2 * inv_freq.len();
    let pooled = window_pool(p.kv, p.gate, s, 2 * ix_hd, ix_hd, m, p.position_bias, p.kv_norm, eps, true);
    let mut ck = pooled; // [n_win, ix_hd]
    let (cos_w, sin_w) = rope_cos_sin(win_pos, inv_freq, scaling);
    apply_interleaved_rope_inplace(&mut ck, n_win, ix_hd, rope_dim, &|w| w, &cos_w, &sin_w);

    let mut q = vec![0.0f32; s * ix_nh * ix_hd];
    q.copy_from_slice(p.q_flat);
    let (cos_q, sin_q) = rope_cos_sin(positions, inv_freq, scaling);
    apply_interleaved_rope_inplace(&mut q, s * ix_nh, ix_hd, rope_dim, &|r| r / ix_nh, &cos_q, &sin_q);

    let w_scale = (ix_nh as f64).powf(-0.5);
    let softmax_scale = (ix_hd as f64).powf(-0.5);
    let top_k = index_topk.min(n_win);
    let mut out = vec![-1i32; s * top_k];
    for si in 0..s {
        let threshold = (positions[si] + 1) / m;
        let mut scores = vec![f64::NEG_INFINITY; n_win];
        for w in 0..n_win {
            if w >= threshold {
                continue;
            }
            let mut acc = 0.0f64;
            for hh in 0..ix_nh {
                let qrow = &q[(si * ix_nh + hh) * ix_hd..(si * ix_nh + hh) * ix_hd + ix_hd];
                let krow = &ck[w * ix_hd..w * ix_hd + ix_hd];
                let mut dot = 0.0f64;
                for d in 0..ix_hd {
                    dot += qrow[d] as f64 * krow[d] as f64;
                }
                let relu = if dot > 0.0 { dot } else { 0.0 } * softmax_scale;
                let wv = p.wgt[si * ix_nh + hh] as f64 * w_scale;
                acc += relu * wv;
            }
            scores[w] = acc;
        }
        let mut order: Vec<usize> = (0..n_win).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
        for kk in 0..top_k {
            let w = order[kk];
            if scores[w].is_finite() && w < threshold {
                out[si * top_k + kk] = w as i32;
            } else {
                out[si * top_k + kk] = -1;
            }
        }
    }
    out
}

/// Outer CSA compressor ONLY — the pooled + RoPE'd `compressed_kv` (`[n_win, hd]`)
/// and `n_win`, factored out of [`csa_compressor_pre`] so the CSA short-circuit
/// (`VLLM_VULKAN_DSV4_CSA_SHORTCIRCUIT`, decode `n_win <= index_topk`) produces a
/// BYTE-IDENTICAL `compressed_kv` while skipping the whole indexer. When the last
/// query's `n_win <= index_topk` the Lightning-Indexer provably admits EXACTLY the
/// causal windows (all-visible), so its `block_bias_last` is all-zeros regardless
/// of scores — see [`indexer_topk_pre`]'s `top_k == n_win` + `w < threshold` guard.
#[allow(clippy::too_many_arguments)]
pub fn csa_compressor_outer_pre(
    kv: &[f32],
    gate: &[f32],
    s: usize,
    hd: usize,
    m: usize,
    position_bias: &[f32], // [m, 2*hd]
    kv_norm: &[f32],       // [hd]
    eps: f32,
    inv_freq: &[f32],
    scaling: f32,
) -> (Vec<f32>, usize) {
    let pooled = window_pool(kv, gate, s, 2 * hd, hd, m, position_bias, kv_norm, eps, true);
    let n_win = pooled.len() / hd;
    let mut compressed = pooled;
    let win_pos: Vec<usize> = (0..n_win).map(|w| w * m).collect();
    if n_win > 0 {
        let (cos, sin) = rope_cos_sin(&win_pos, inv_freq, scaling);
        let rope_dim = 2 * inv_freq.len();
        apply_interleaved_rope_inplace(&mut compressed, n_win, hd, rope_dim, &|w| w, &cos, &sin);
    }
    (compressed, n_win)
}

// ==================== INCREMENTAL single-window compressor ====================
// Prefix-stable decode cache (report P0 / Phase-2): a completed compressed window
// is a pure function of past tokens (CSA overlaps back at most one closed window;
// HCA is disjoint), so it NEVER needs recomputation. These helpers pool + RoPE
// exactly ONE closed window from the projected rows of the last one/two windows —
// BYTE-IDENTICAL to row `w` of [`csa_compressor_outer_pre`] / [`hca_compressor_pre`]
// over the full history, but O(m) instead of O(T). See the unit gate
// `incremental_window_matches_full_outer_*` (random-input, 8 windows, cos-0.0).

/// One CSA (two-series Ca/Cb) outer-compressor window. `kv`/`gate` are the
/// projections (`x @ W^T`, `[s, 2*hd]`) of the source rows `[(w-1)*m .. (w+1)*m)`
/// for `w >= 1` (so `s == 2m`) or `[0 .. m)` for `w == 0` (so `s == m`). Returns
/// the single RoPE'd compressed row `[hd]`. The pool reads window `w`'s Cb (its own
/// m tokens) and Ca (the previous window's m tokens); passing exactly those rows
/// reproduces the full-history pool for that window because `window_pool` is
/// independent across windows except for the one-window Ca lookback (here the
/// leading local window), and window 0's Ca is zeroed by `window_pool` itself.
#[allow(clippy::too_many_arguments)]
pub fn csa_compress_window_incr(
    kv: &[f32],
    gate: &[f32],
    s: usize,
    hd: usize,
    m: usize,
    w: usize,
    position_bias: &[f32], // [m, 2*hd]
    kv_norm: &[f32],       // [hd]
    eps: f32,
    inv_freq: &[f32],
    scaling: f32,
) -> Vec<f32> {
    let pooled = window_pool(kv, gate, s, 2 * hd, hd, m, position_bias, kv_norm, eps, true);
    let n_local = pooled.len() / hd; // 1 (w==0) or 2 (w>=1); the TARGET is the last one
    let row = (n_local - 1) * hd;
    let mut compressed = pooled[row..row + hd].to_vec();
    let (cos, sin) = rope_cos_sin(&[w * m], inv_freq, scaling);
    let rope_dim = 2 * inv_freq.len();
    apply_interleaved_rope_inplace(&mut compressed, 1, hd, rope_dim, &|_| 0, &cos, &sin);
    compressed
}

/// One HCA (single-series, disjoint) outer-compressor window. `kv`/`gate` are the
/// projections (`x @ W^T`, `[m, hd]`) of the source rows `[w*m .. (w+1)*m)`. Returns
/// the single RoPE'd compressed row `[hd]`, byte-identical to row `w` of
/// [`hca_compressor_pre`] over the full history.
#[allow(clippy::too_many_arguments)]
pub fn hca_compress_window_incr(
    kv: &[f32],
    gate: &[f32],
    hd: usize,
    m: usize,
    w: usize,
    position_bias: &[f32], // [m, hd]
    kv_norm: &[f32],       // [hd]
    eps: f32,
    inv_freq: &[f32],
    scaling: f32,
) -> Vec<f32> {
    let pooled = window_pool(kv, gate, m, hd, hd, m, position_bias, kv_norm, eps, false);
    let mut compressed = pooled[0..hd].to_vec();
    let (cos, sin) = rope_cos_sin(&[w * m], inv_freq, scaling);
    let rope_dim = 2 * inv_freq.len();
    apply_interleaved_rope_inplace(&mut compressed, 1, hd, rope_dim, &|_| 0, &cos, &sin);
    compressed
}

/// Resident-projection CSA compressor. Outer `kv`/`gate` are `[s, 2*hd]`; the
/// indexer projections arrive precomputed in `ixp`.
#[allow(clippy::too_many_arguments)]
pub fn csa_compressor_pre(
    kv: &[f32],
    gate: &[f32],
    s: usize,
    hd: usize,
    m: usize,
    positions: &[usize],
    position_bias: &[f32], // [m, 2*hd]
    kv_norm: &[f32],       // [hd]
    eps: f32,
    inv_freq: &[f32],
    scaling: f32,
    ix_nh: usize,
    ix_hd: usize,
    index_topk: usize,
    ixp: &IndexerProj,
) -> (Vec<f32>, Vec<i32>) {
    let (compressed, n_win) =
        csa_compressor_outer_pre(kv, gate, s, hd, m, position_bias, kv_norm, eps, inv_freq, scaling);
    let win_pos: Vec<usize> = (0..n_win).map(|w| w * m).collect();
    let top_idx = if n_win == 0 {
        vec![]
    } else {
        indexer_topk_pre(s, m, positions, &win_pos, n_win, ix_nh, ix_hd, index_topk, eps, inv_freq, scaling, ixp)
    };
    let mut vis = vec![0i32; s * n_win];
    let top_k = index_topk.min(n_win);
    for si in 0..s {
        for kk in 0..top_k {
            let idx = top_idx[si * top_k + kk];
            if idx >= 0 {
                vis[si * n_win + idx as usize] = 1;
            }
        }
    }
    (compressed, vis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn f32v(v: &Value) -> Vec<f32> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
    }
    fn usv(v: &Value) -> Vec<usize> {
        v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
    }
    fn load(name: &str) -> Value {
        let path = format!("{}/tests/fixtures/dsv4/{}", env!("CARGO_MANIFEST_DIR"), name);
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{name} (run dump_dsa_fixture.py)"))).unwrap()
    }
    fn maxerr(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// RoPE main + compress-yarn derivation matches the reference rotary buffers.
    #[test]
    fn rope_derivation_matches_reference() {
        let j = load("rope.json");
        let head_dim = j["head_dim"].as_u64().unwrap() as usize;
        let positions = usv(&j["positions"]);
        // main: theta 10000, partial 0.125
        let ifm = inv_freq_main(head_dim, 0.125, 10000.0);
        assert!(maxerr(&ifm, &f32v(&j["main"]["inv_freq"])) < 1e-6, "main inv_freq");
        let (cm, sm) = rope_cos_sin(&positions, &ifm, j["main"]["attention_scaling"].as_f64().unwrap() as f32);
        assert!(maxerr(&cm, &f32v(&j["main"]["cos"])) < 1e-5, "main cos");
        assert!(maxerr(&sm, &f32v(&j["main"]["sin"])) < 1e-5, "main sin");
        // compress: yarn theta 160000 factor 16 beta_fast 32 beta_slow 1 orig 65536
        let ify = inv_freq_yarn(head_dim, 0.125, 160000.0, 16.0, 32.0, 1.0, 65536.0);
        assert!(maxerr(&ify, &f32v(&j["compress"]["inv_freq"])) < 1e-6, "yarn inv_freq");
        let sc = j["compress"]["attention_scaling"].as_f64().unwrap() as f32;
        let (cc, sic) = rope_cos_sin(&positions, &ify, sc);
        assert!(maxerr(&cc, &f32v(&j["compress"]["cos"])) < 1e-5, "compress cos");
        assert!(maxerr(&sic, &f32v(&j["compress"]["sin"])) < 1e-5, "compress sin");
    }

    #[test]
    fn hca_compressor_matches_reference() {
        let j = load("dsa_hca.json");
        let s = j["seq_len"].as_u64().unwrap() as usize;
        let h = j["hidden_size"].as_u64().unwrap() as usize;
        let hd = j["head_dim"].as_u64().unwrap() as usize;
        let m = j["compress_rate"].as_u64().unwrap() as usize;
        let eps = j["rms_norm_eps"].as_f64().unwrap() as f32;
        let positions = usv(&j["positions"]);
        let inv_freq = f32v(&j["compress_inv_freq"]);
        let scaling = j["compress_scaling"].as_f64().unwrap() as f32;
        let w = &j["weights"];
        let (ckv, vis) = hca_compressor(
            &f32v(&j["hs"]), s, h, hd, m, &positions,
            &f32v(&w["kv_proj"]), &f32v(&w["gate_proj"]), &f32v(&w["position_bias"]),
            &f32v(&w["kv_norm"]), eps, &inv_freq, scaling,
        );
        let t = j["compressed_len"].as_u64().unwrap() as usize;
        assert_eq!(ckv.len(), t * hd);
        assert!(maxerr(&ckv, &f32v(&j["compressed_kv"])) < 1e-3, "HCA compressed_kv");
        let exp_vis: Vec<i32> = j["block_bias_visible"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        assert_eq!(vis, exp_vis, "HCA block_bias visibility");
    }

    #[test]
    fn csa_compressor_matches_reference() {
        let j = load("dsa_csa.json");
        let s = j["seq_len"].as_u64().unwrap() as usize;
        let h = j["hidden_size"].as_u64().unwrap() as usize;
        let hd = j["head_dim"].as_u64().unwrap() as usize;
        let m = j["compress_rate"].as_u64().unwrap() as usize;
        let eps = j["rms_norm_eps"].as_f64().unwrap() as f32;
        let positions = usv(&j["positions"]);
        let inv_freq = f32v(&j["compress_inv_freq"]);
        let scaling = j["compress_scaling"].as_f64().unwrap() as f32;
        let q_lora = f32v(&j["q_residual"]).len() / s;
        let ix_nh = j["index_n_heads"].as_u64().unwrap() as usize;
        let ix_hd = j["index_head_dim"].as_u64().unwrap() as usize;
        let topk = j["index_topk"].as_u64().unwrap() as usize;
        let w = &j["weights"];
        let iw = &j["indexer"];
        let (kvp, gp, pb, kn, qb, wp) = (
            f32v(&iw["kv_proj"]), f32v(&iw["gate_proj"]), f32v(&iw["position_bias"]),
            f32v(&iw["kv_norm"]), f32v(&iw["q_b_proj"]), f32v(&iw["weights_proj"]),
        );
        let ix = IndexerWeights {
            kv_proj: &kvp, gate_proj: &gp, position_bias: &pb, kv_norm: &kn,
            q_b_proj: &qb, weights_proj: &wp,
        };
        let (ckv, vis) = csa_compressor(
            &f32v(&j["hs"]), &f32v(&j["q_residual"]), s, h, hd, m, &positions,
            &f32v(&w["kv_proj"]), &f32v(&w["gate_proj"]), &f32v(&w["position_bias"]),
            &f32v(&w["kv_norm"]), eps, &inv_freq, scaling, q_lora, ix_nh, ix_hd, topk, &ix,
        );
        let t = j["compressed_len"].as_u64().unwrap() as usize;
        assert_eq!(ckv.len(), t * hd);
        assert!(maxerr(&ckv, &f32v(&j["compressed_kv"])) < 1e-3, "CSA compressed_kv");
        let exp_vis: Vec<i32> = j["block_bias_visible"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        assert_eq!(vis, exp_vis, "CSA block_bias visibility (indexer top-k)");
    }

    // Deterministic reproducible pseudo-randoms (no rand dep): splitmix64 → f32 in
    // [-1,1). Enough entropy to catch any index/pooling/RoPE-position mistake in the
    // incremental decomposition.
    fn prng(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                x = x.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// ★ INCREMENTAL COMPRESSOR DECOMPOSITION GATE (offline, no fixture, no GPU).
    /// Assembling the compressed-KV plane one CLOSED window at a time via
    /// [`csa_compress_window_incr`] / [`hca_compress_window_incr`] — each pooling only
    /// the last one/two windows' projected rows — reproduces the full-history
    /// [`csa_compressor_outer_pre`] / [`hca_compressor_pre`] plane BYTE-FOR-BYTE
    /// (max_abs_diff == 0), across many windows. This is the algebraic bit-exactness
    /// proof for the `VLLM_VULKAN_DSV4_COMPRESSOR_CACHE` decode lever — independent of
    /// the tiny fixture (which never closes an HCA window). Covers CSA two-series
    /// (Ca prev-window overlap) AND HCA disjoint single-series.
    #[test]
    fn incremental_window_matches_full_outer_csa_and_hca() {
        let (hd, m, n_win, eps, scaling) = (32usize, 4usize, 8usize, 1e-6f32, 1.0f32);
        let s = n_win * m;
        // rope inv_freq over rope_dim = hd (half = hd/2).
        let inv: Vec<f32> = (0..hd / 2).map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / hd as f32)).collect();
        let kv_norm = prng(1, hd).iter().map(|v| v.abs() + 0.5).collect::<Vec<_>>();

        // ---- CSA (two_series, kv/gate are [s, 2*hd]) ----
        let kv = prng(10, s * 2 * hd);
        let gate = prng(11, s * 2 * hd);
        let pb_csa = prng(12, m * 2 * hd);
        let (full_csa, nw) = csa_compressor_outer_pre(&kv, &gate, s, hd, m, &pb_csa, &kv_norm, eps, &inv, scaling);
        assert_eq!(nw, n_win);
        let mut incr_csa = Vec::new();
        for w in 0..n_win {
            let (lo, sl) = if w == 0 { (0usize, m) } else { ((w - 1) * m, 2 * m) };
            let ks = &kv[lo * 2 * hd..(lo + sl) * 2 * hd];
            let gs = &gate[lo * 2 * hd..(lo + sl) * 2 * hd];
            let row = csa_compress_window_incr(ks, gs, sl, hd, m, w, &pb_csa, &kv_norm, eps, &inv, scaling);
            incr_csa.extend_from_slice(&row);
        }
        assert_eq!(maxerr(&incr_csa, &full_csa), 0.0, "CSA incremental != full outer (byte-exact)");

        // ---- HCA (single-series, kv/gate are [s, hd]) ----
        let kvh = prng(20, s * hd);
        let gateh = prng(21, s * hd);
        let pb_hca = prng(22, m * hd);
        let positions: Vec<usize> = (0..s).collect();
        let (full_hca, _bias) =
            hca_compressor_pre(&kvh, &gateh, s, hd, m, &positions, &pb_hca, &kv_norm, eps, &inv, scaling);
        let mut incr_hca = Vec::new();
        for w in 0..n_win {
            let ks = &kvh[w * m * hd..(w + 1) * m * hd];
            let gs = &gateh[w * m * hd..(w + 1) * m * hd];
            let row = hca_compress_window_incr(ks, gs, hd, m, w, &pb_hca, &kv_norm, eps, &inv, scaling);
            incr_hca.extend_from_slice(&row);
        }
        assert_eq!(maxerr(&incr_hca, &full_hca), 0.0, "HCA incremental != full outer (byte-exact)");
    }
}
