// SPDX-License-Identifier: Apache-2.0
//! Kimi-Linear-48B-A3B (`kimi_linear`, Moonshot) bring-up — config + layer
//! schedule (P1a), plus KDA / MLA / MoE-router hooks landed in later phases.
//!
//! Architecture (from the shipped `config.json`, verified in Phase-0 against the
//! mlx-lm `kimi_linear` oracle — see `docs/kimi-linear-48b-bringup-plan.md`):
//!   - 27 layers, hidden 2304, vocab 163840, untied lm_head, rms_norm_eps 1e-5.
//!   - Heterogeneous attention schedule: 7 **MLA** layers (0-idx
//!     `[3,7,11,15,19,23,26]`), 20 **KDA** (GatedDeltaNet-family, per-CHANNEL
//!     decay) layers (the rest, incl. layer 0).
//!   - `first_k_dense_replace = 1` → layer 0 MLP is **dense** (inter 9216); every
//!     other layer is **MoE** (256 experts, top-8, +1 shared).
//!
//! The schedule is **not** a `layer_types` array (unlike qwen3.5); it is built
//! from `linear_attn_config.full_attn_layers` / `kda_layers`, which are
//! **1-indexed** on disk. The 1→0 conversion is the #1 loader trap and is pinned
//! by `kimi_schedule_offbyone` below.
#![allow(dead_code)]

use std::collections::HashMap;

/// Per-layer attention kind in the Kimi-Linear heterogeneous schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiLayerKind {
    /// Kimi Delta Attention — GatedDeltaNet recurrence with **per-key-channel**
    /// decay (vs qwen3.5's per-head scalar), sigmoid output gate, 3 depthwise
    /// convs. Reuses the qwen3.5 GDN state machine with `decay[kk]` indexing.
    Kda,
    /// Multi-head Latent Attention — compressed KV latent (512) + un-rotated
    /// 64-dim MQA-shared pe key, qk_dim=192 ≠ v_dim=128, scale=192^-0.5.
    Mla,
}

/// MoE router activation. Kimi uses **sigmoid** (DeepSeek-V3 style) with a
/// per-expert `e_score_correction_bias` on selection only, NOT qwen's softmax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterActivation {
    Sigmoid,
    Softmax,
}

/// Kimi-Linear configuration parsed from `config.json`.
#[derive(Debug, Clone)]
pub struct KimiConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,

    /// Layers `[0, first_k_dense_replace)` use a **dense** MLP; the rest are MoE.
    pub first_k_dense_replace: usize,
    pub intermediate_size: usize, // dense MLP inter (layer 0): 9216

    // ---- MLA (full-attention) dims ----
    pub num_attention_heads: usize, // 32
    pub kv_lora_rank: usize,        // 512
    /// `q_lora_rank` is `null` on disk → Q is **uncompressed** (direct q_proj).
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize, // 128
    pub qk_rope_head_dim: usize, // 64  (un-rotated, MQA-shared, contributes to scores)
    pub v_head_dim: usize,       // 128
    pub mla_use_nope: bool,      // true

    // ---- KDA (linear-attention) dims ----
    pub kda_num_heads: usize,     // 32
    pub kda_head_dim: usize,      // 128
    pub kda_conv_kernel: usize,   // 4

    // ---- MoE dims ----
    pub num_experts: usize,           // 256
    pub num_experts_per_token: usize, // 8
    pub num_shared_experts: usize,    // 1
    pub moe_intermediate_size: usize, // 1024
    pub routed_scaling_factor: f32,   // 2.446
    pub router_activation: RouterActivation,
    pub moe_renormalize: bool, // true
    pub num_expert_group: usize, // 1  (grouped-topk degenerates to plain top-8)
    pub topk_group: usize,       // 1

    /// Per-layer attention kind, length == num_hidden_layers (0-indexed).
    pub layer_schedule: Vec<KimiLayerKind>,
}

impl KimiConfig {
    /// q head_dim = nope (128) + pe (64) = 192. Attention scale = 192^-0.5.
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// True iff layer `i`'s MLP is dense (only layers `< first_k_dense_replace`).
    pub fn is_dense_mlp(&self, layer_idx: usize) -> bool {
        layer_idx < self.first_k_dense_replace
    }
    /// KDA conv1d channel count = 3 * (num_heads * head_dim) (q, k, v depthwise).
    pub fn kda_conv_dim(&self) -> usize {
        3 * self.kda_num_heads * self.kda_head_dim
    }

    /// Build the per-layer schedule from `linear_attn_config`, converting the
    /// **1-indexed** on-disk `full_attn_layers` (→ MLA) and `kda_layers` (→ KDA)
    /// to 0-indexed slots. Validates that every layer `[0, num_hidden_layers)` is
    /// assigned **exactly once** (no gap, no overlap) — the off-by-one trap.
    pub fn build_schedule(
        num_hidden_layers: usize,
        full_attn_layers_1idx: &[usize],
        kda_layers_1idx: &[usize],
    ) -> Result<Vec<KimiLayerKind>, String> {
        let mut slots: Vec<Option<KimiLayerKind>> = vec![None; num_hidden_layers];
        let mut place = |one_idx: usize, kind: KimiLayerKind| -> Result<(), String> {
            if one_idx == 0 {
                return Err(format!(
                    "layer index {one_idx} is not 1-indexed (on-disk lists are 1-based)"
                ));
            }
            let z = one_idx - 1; // 1-idx -> 0-idx
            if z >= num_hidden_layers {
                return Err(format!(
                    "layer 1-idx {one_idx} (0-idx {z}) out of range for {num_hidden_layers} layers"
                ));
            }
            if slots[z].is_some() {
                return Err(format!("layer 0-idx {z} assigned twice"));
            }
            slots[z] = Some(kind);
            Ok(())
        };
        for &l in full_attn_layers_1idx {
            place(l, KimiLayerKind::Mla)?;
        }
        for &l in kda_layers_1idx {
            place(l, KimiLayerKind::Kda)?;
        }
        slots
            .into_iter()
            .enumerate()
            .map(|(z, s)| s.ok_or_else(|| format!("layer 0-idx {z} unassigned (schedule gap)")))
            .collect()
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let u = |key: &str| v[key].as_u64().map(|x| x as usize);
        let req = |key: &str| u(key).ok_or_else(|| format!("config.json missing '{key}'"));

        let num_hidden_layers = req("num_hidden_layers")?;

        let lac = v
            .get("linear_attn_config")
            .ok_or("config.json missing 'linear_attn_config'")?;
        let idx_list = |key: &str| -> Result<Vec<usize>, String> {
            lac[key]
                .as_array()
                .ok_or_else(|| format!("linear_attn_config missing '{key}'"))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|n| n as usize)
                        .ok_or_else(|| format!("linear_attn_config.{key} non-integer entry"))
                })
                .collect()
        };
        let full_attn_layers = idx_list("full_attn_layers")?;
        let kda_layers = idx_list("kda_layers")?;
        let layer_schedule =
            Self::build_schedule(num_hidden_layers, &full_attn_layers, &kda_layers)?;

        let router_activation = match v["moe_router_activation_func"].as_str() {
            Some("sigmoid") | None => RouterActivation::Sigmoid,
            Some("softmax") => RouterActivation::Softmax,
            Some(o) => return Err(format!("unknown moe_router_activation_func {o:?}")),
        };

        // top-k key is `num_experts_per_token` on disk (NOT `num_experts_per_tok`).
        let num_experts_per_token = u("num_experts_per_token")
            .or_else(|| u("num_experts_per_tok"))
            .ok_or("config.json missing 'num_experts_per_token'")?;

        Ok(KimiConfig {
            hidden_size: req("hidden_size")?,
            num_hidden_layers,
            vocab_size: req("vocab_size")?,
            rms_norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
            first_k_dense_replace: u("first_k_dense_replace").unwrap_or(0),
            intermediate_size: u("intermediate_size").unwrap_or(0),
            num_attention_heads: req("num_attention_heads")?,
            kv_lora_rank: req("kv_lora_rank")?,
            q_lora_rank: v["q_lora_rank"].as_u64().map(|x| x as usize),
            qk_nope_head_dim: req("qk_nope_head_dim")?,
            qk_rope_head_dim: req("qk_rope_head_dim")?,
            v_head_dim: req("v_head_dim")?,
            mla_use_nope: v["mla_use_nope"].as_bool().unwrap_or(false),
            kda_num_heads: lac["num_heads"]
                .as_u64()
                .map(|x| x as usize)
                .ok_or("linear_attn_config missing 'num_heads'")?,
            kda_head_dim: lac["head_dim"]
                .as_u64()
                .map(|x| x as usize)
                .ok_or("linear_attn_config missing 'head_dim'")?,
            kda_conv_kernel: lac["short_conv_kernel_size"]
                .as_u64()
                .map(|x| x as usize)
                .ok_or("linear_attn_config missing 'short_conv_kernel_size'")?,
            num_experts: u("num_experts").unwrap_or(0),
            num_experts_per_token,
            num_shared_experts: u("num_shared_experts").unwrap_or(0),
            moe_intermediate_size: u("moe_intermediate_size").unwrap_or(0),
            routed_scaling_factor: v["routed_scaling_factor"].as_f64().unwrap_or(1.0) as f32,
            router_activation,
            moe_renormalize: v["moe_renormalize"].as_bool().unwrap_or(true),
            num_expert_group: u("num_expert_group").unwrap_or(1),
            topk_group: u("topk_group").unwrap_or(1),
            layer_schedule,
        })
    }
}

/// Kimi Delta Attention (KDA) — CPU reference. GatedDeltaNet recurrence with
/// **per-key-channel** decay (low-rank `f_a`/`f_b`) + per-head `A_log` + per-channel
/// `dt_bias`, **sigmoid** output gate (low-rank `g_a`/`g_b`, replacing the qwen3.5
/// silu-z tail), 3 depthwise causal convs (silu), qk RMSNorm (eps **1e-6**), and a
/// **sequential** scan (no chunked WY/UT — that reorders rank-1 accums and fails
/// argmax). Reproduces the mlx-lm `KimiDeltaAttention` oracle bit-exact.
///
/// The single-layer decay edit vs qwen3.5 GDN is `state[.. + kk] *= decay[kk]`
/// (per-key-channel vector) at the 3 sites qwen35.rs:1038 (decode), :1281
/// (scan-prefill), q35_gdn_step.comp:67 (GPU); here it is `st[dv][dk] *= g[dk]`.
pub mod kda {
    #[inline]
    pub(crate) fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    #[inline]
    pub(crate) fn softplus(x: f32) -> f32 {
        // log1p(exp(-|x|)) + max(x,0) — matches the oracle's stable softplus
        (-x.abs()).exp().ln_1p() + x.max(0.0)
    }

    /// row-major `x[rows, in] @ w[out, in]^T -> [rows, out]`.
    fn matmul_wt(x: &[f32], rows: usize, inn: usize, w: &[f32], out: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * out];
        for r in 0..rows {
            let xr = &x[r * inn..(r + 1) * inn];
            for o in 0..out {
                let wr = &w[o * inn..(o + 1) * inn];
                let mut acc = 0f32;
                for i in 0..inn {
                    acc += xr[i] * wr[i];
                }
                y[r * out + o] = acc;
            }
        }
        y
    }

    /// Causal depthwise conv (kernel `kern`, `kern-1` zero left-pad) + silu, per
    /// channel. `seq[L, C]`, `taps[C, kern]`; returns `[L, C]`.
    pub(crate) fn depthwise_silu(seq: &[f32], l: usize, c: usize, taps: &[f32], kern: usize) -> Vec<f32> {
        let mut out = vec![0f32; l * c];
        for t in 0..l {
            for ch in 0..c {
                let mut acc = 0f32;
                for kk in 0..kern {
                    // window position (t - (kern-1) + kk); <0 => zero pad
                    let src = t as isize - (kern as isize - 1) + kk as isize;
                    if src >= 0 {
                        acc += seq[src as usize * c + ch] * taps[ch * kern + kk];
                    }
                }
                out[t * c + ch] = acc * sigmoid(acc); // silu
            }
        }
        out
    }

    /// RMSNorm without weight over the last `d` dims (eps as given).
    pub(crate) fn rms_no_weight(v: &mut [f32], rows: usize, d: usize, eps: f32) {
        for r in 0..rows {
            let s = &mut v[r * d..(r + 1) * d];
            let ms: f32 = s.iter().map(|z| z * z).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for z in s.iter_mut() {
                *z *= inv;
            }
        }
    }

    pub struct KdaWeights {
        pub h: usize,
        pub nh: usize,
        pub hd: usize,
        pub kern: usize,
        pub eps: f32, // o_norm eps (rms_norm_eps)
        pub q_proj: Vec<f32>,
        pub k_proj: Vec<f32>,
        pub v_proj: Vec<f32>,
        pub q_conv: Vec<f32>, // [proj, kern]
        pub k_conv: Vec<f32>,
        pub v_conv: Vec<f32>,
        pub f_a: Vec<f32>, // [hd, h]
        pub f_b: Vec<f32>, // [proj, hd]
        pub b_proj: Vec<f32>, // [nh, h]
        pub g_a: Vec<f32>, // [hd, h]
        pub g_b: Vec<f32>, // [proj, hd]
        pub a_log: Vec<f32>,  // [nh]
        pub dt_bias: Vec<f32>, // [proj]
        pub o_norm: Vec<f32>, // [hd]
        pub o_proj: Vec<f32>, // [h, proj]
    }

    /// Per-token transients the GPU `kda_gdn_step` kernel consumes (the exact
    /// tensors fed to the recurrence, so a GPU-vs-CPU gate isolates the new
    /// per-channel-decay kernel from the projection/conv/qknorm math). Each is
    /// flattened `[L, proj]` except `b_in` which is `[L, nh]` (pre-sigmoid).
    pub struct KdaCapture {
        pub q: Vec<f32>,     // [L, proj] normed + scaled
        pub k: Vec<f32>,     // [L, proj] normed + scaled
        pub v: Vec<f32>,     // [L, proj] post silu-conv
        pub decay: Vec<f32>, // [L, proj] per key-channel exp(-exp(A_log)*softplus(..))
        pub b_in: Vec<f32>,  // [L, nh] pre-sigmoid beta
        pub gate: Vec<f32>,  // [L, proj] g_b(g_a(x)) (pre-sigmoid)
    }

    /// Full sequential KDA forward. `x[L, H]` -> `[L, H]`. `state` (if given) is
    /// `[nh*hd*hd]` (`[nh][Dv][Dk]`) and is advanced in place; conv history is
    /// carried by prepending it to the sequence by the caller (here we start from
    /// zero conv state, matching the oracle prefill).
    pub fn forward(w: &KdaWeights, x: &[f32], l: usize, state: &mut [f32]) -> Vec<f32> {
        forward_capture(w, x, l, state).0
    }

    /// Same math as `forward`, additionally returning the per-token GPU-injection
    /// transients (`KdaCapture`). `forward` delegates here so the recurrence is
    /// single-sourced. The decay vector is hoisted out of the recurrence loop
    /// (precomputed `[L, proj]`) — value-identical, just made capturable.
    pub fn forward_capture(
        w: &KdaWeights,
        x: &[f32],
        l: usize,
        state: &mut [f32],
    ) -> (Vec<f32>, KdaCapture) {
        let (h, nh, hd, kern) = (w.h, w.nh, w.hd, w.kern);
        let proj = nh * hd;
        let inv = (hd as f32).powf(-0.5);

        let qc0 = matmul_wt(x, l, h, &w.q_proj, proj);
        let kc0 = matmul_wt(x, l, h, &w.k_proj, proj);
        let vc0 = matmul_wt(x, l, h, &w.v_proj, proj);
        let mut q = depthwise_silu(&qc0, l, proj, &w.q_conv, kern);
        let mut k = depthwise_silu(&kc0, l, proj, &w.k_conv, kern);
        let v = depthwise_silu(&vc0, l, proj, &w.v_conv, kern);

        // qk RMSNorm (no weight, eps 1e-6), then scale by inv^2 / inv.
        rms_no_weight(&mut q, l * nh, hd, 1e-6);
        rms_no_weight(&mut k, l * nh, hd, 1e-6);
        for z in q.iter_mut() {
            *z *= inv * inv;
        }
        for z in k.iter_mut() {
            *z *= inv;
        }

        // low-rank per-channel decay logits + per-head beta
        let fa = matmul_wt(x, l, h, &w.f_a, hd); // [L, hd]
        let a_log_in = matmul_wt(&fa, l, hd, &w.f_b, proj); // [L, proj]
        let b_in = matmul_wt(x, l, h, &w.b_proj, nh); // [L, nh]

        // Hoisted per-channel decay: decay[t, hh, dk] = exp(-exp(A_log[hh]) *
        // softplus(a_log_in + dt_bias)). Value-identical to the in-loop form.
        let mut decay_all = vec![0f32; l * proj];
        for t in 0..l {
            for hh in 0..nh {
                let neg_exp_a = -(w.a_log[hh].exp());
                for dk in 0..hd {
                    let al = a_log_in[t * proj + hh * hd + dk] + w.dt_bias[hh * hd + dk];
                    decay_all[t * proj + hh * hd + dk] = (neg_exp_a * softplus(al)).exp();
                }
            }
        }

        let mut y = vec![0f32; l * proj];
        for t in 0..l {
            for hh in 0..nh {
                let st = &mut state[hh * hd * hd..(hh + 1) * hd * hd]; // [Dv, Dk]
                let kt = &k[(t * nh + hh) * hd..(t * nh + hh + 1) * hd];
                let vt = &v[(t * nh + hh) * hd..(t * nh + hh + 1) * hd];
                let qt = &q[(t * nh + hh) * hd..(t * nh + hh + 1) * hd];
                let beta = sigmoid(b_in[t * nh + hh]);
                let decay = &decay_all[(t * nh + hh) * hd..(t * nh + hh + 1) * hd];
                // st *= decay (per key-channel, last axis Dk)
                for dv in 0..hd {
                    for dk in 0..hd {
                        st[dv * hd + dk] *= decay[dk];
                    }
                }
                // kv_mem[dv] = sum_dk st[dv,dk]*k[dk]
                let mut delta = vec![0f32; hd];
                for dv in 0..hd {
                    let mut m = 0f32;
                    for dk in 0..hd {
                        m += st[dv * hd + dk] * kt[dk];
                    }
                    delta[dv] = (vt[dv] - m) * beta;
                }
                // st += k (x) delta  (rank-1: st[dv,dk] += k[dk]*delta[dv])
                for dv in 0..hd {
                    for dk in 0..hd {
                        st[dv * hd + dk] += kt[dk] * delta[dv];
                    }
                }
                // Y[dv] = sum_dk st[dv,dk]*q[dk]
                let yo = &mut y[(t * nh + hh) * hd..(t * nh + hh + 1) * hd];
                for dv in 0..hd {
                    let mut o = 0f32;
                    for dk in 0..hd {
                        o += st[dv * hd + dk] * qt[dk];
                    }
                    yo[dv] = o;
                }
            }
        }

        // output gate: o_norm(Y) * sigmoid(g_b(g_a(x))), then o_proj
        let ga = matmul_wt(x, l, h, &w.g_a, hd);
        let gate = matmul_wt(&ga, l, hd, &w.g_b, proj);
        // weighted RMSNorm over hd (eps = rms_norm_eps)
        for r in 0..l * nh {
            let s = &mut y[r * hd..(r + 1) * hd];
            let ms: f32 = s.iter().map(|z| z * z).sum::<f32>() / hd as f32;
            let scale = 1.0 / (ms + w.eps).sqrt();
            for (i, z) in s.iter_mut().enumerate() {
                *z = *z * scale * w.o_norm[i];
            }
        }
        for i in 0..l * proj {
            y[i] *= sigmoid(gate[i]);
        }
        let out = matmul_wt(&y, l, proj, &w.o_proj, h);
        let cap = KdaCapture {
            q,
            k,
            v,
            decay: decay_all,
            b_in,
            gate,
        };
        (out, cap)
    }

    // ------------------------- resident single-token DECODE -------------------------
    // The decode path keeps the KDA recurrence matrix AND the depthwise-conv sliding
    // window resident across tokens, so each step is O(1) in sequence length (vs the
    // stateless fresh-prefill `forward`). Bit-identical to row `t` of `forward` on the
    // same input stream: the conv window reproduces the oracle's zero-left-pad causal
    // conv (zero-init history == "src<0 => skip"), the qk-norm/decay/gate are per-token,
    // and the recurrence advances exactly as the `kimi_kda_scan_bit_exact_vs_serial`
    // gate proves the scan does.

    /// Resident KDA decode state: recurrence matrix `[nh][Dv][Dk]` + the last
    /// `kern-1` **pre-conv** q/k/v projection rows (the conv sliding window,
    /// oldest-first). Zero-init = the oracle's zero-left-pad conv start.
    pub struct KdaState {
        pub recur: Vec<f32>, // [nh*hd*hd]
        pub(crate) conv_q: Vec<f32>,    // [(kern-1)*proj]
        pub(crate) conv_k: Vec<f32>,
        pub(crate) conv_v: Vec<f32>,
    }
    impl KdaState {
        pub fn new(nh: usize, hd: usize, kern: usize) -> Self {
            let proj = nh * hd;
            let hist = kern.saturating_sub(1) * proj;
            KdaState {
                recur: vec![0f32; nh * hd * hd],
                conv_q: vec![0f32; hist],
                conv_k: vec![0f32; hist],
                conv_v: vec![0f32; hist],
            }
        }
    }

    /// One causal depthwise-conv+silu output for the CURRENT token, given the
    /// `(kern-1)`-row history `hist` and the current pre-conv row `cur` (`[proj]`).
    /// Then slides the window (drop oldest row, append `cur`). Bit-identical to
    /// `depthwise_silu`'s row `t` (same tap order `kk=0..kern`, same zero-pad).
    pub(crate) fn conv_step(hist: &mut [f32], cur: &[f32], proj: usize, taps: &[f32], kern: usize) -> Vec<f32> {
        let mut out = vec![0f32; proj];
        for ch in 0..proj {
            let mut acc = 0f32;
            for kk in 0..kern {
                // window slot kk: kk<kern-1 -> history row kk; kk==kern-1 -> current.
                let val = if kk == kern - 1 {
                    cur[ch]
                } else {
                    hist[kk * proj + ch]
                };
                acc += val * taps[ch * kern + kk];
            }
            out[ch] = acc * sigmoid(acc); // silu
        }
        if kern > 1 {
            hist.copy_within(proj.., 0); // shift rows [1..] down to [0..]
            let n = hist.len();
            hist[n - proj..].copy_from_slice(cur); // newest row = current
        }
        out
    }

    /// Single-token KDA decode. `x[H]` -> `[H]`, advancing `st` in place. Mirrors
    /// `forward_capture`'s per-token body exactly (order-preserving => bit-identical).
    pub fn decode_step(w: &KdaWeights, x: &[f32], st: &mut KdaState) -> Vec<f32> {
        let (h, nh, hd, kern) = (w.h, w.nh, w.hd, w.kern);
        let proj = nh * hd;
        let inv = (hd as f32).powf(-0.5);

        let qc0 = matmul_wt(x, 1, h, &w.q_proj, proj);
        let kc0 = matmul_wt(x, 1, h, &w.k_proj, proj);
        let vc0 = matmul_wt(x, 1, h, &w.v_proj, proj);
        let mut q = conv_step(&mut st.conv_q, &qc0, proj, &w.q_conv, kern);
        let mut k = conv_step(&mut st.conv_k, &kc0, proj, &w.k_conv, kern);
        let v = conv_step(&mut st.conv_v, &vc0, proj, &w.v_conv, kern);

        rms_no_weight(&mut q, nh, hd, 1e-6);
        rms_no_weight(&mut k, nh, hd, 1e-6);
        for z in q.iter_mut() {
            *z *= inv * inv;
        }
        for z in k.iter_mut() {
            *z *= inv;
        }

        let fa = matmul_wt(x, 1, h, &w.f_a, hd);
        let a_log_in = matmul_wt(&fa, 1, hd, &w.f_b, proj);
        let b_in = matmul_wt(x, 1, h, &w.b_proj, nh);

        let mut decay = vec![0f32; proj];
        for hh in 0..nh {
            let neg_exp_a = -(w.a_log[hh].exp());
            for dk in 0..hd {
                let al = a_log_in[hh * hd + dk] + w.dt_bias[hh * hd + dk];
                decay[hh * hd + dk] = (neg_exp_a * softplus(al)).exp();
            }
        }

        let mut y = vec![0f32; proj];
        for hh in 0..nh {
            let stm = &mut st.recur[hh * hd * hd..(hh + 1) * hd * hd]; // [Dv, Dk]
            let kt = &k[hh * hd..(hh + 1) * hd];
            let vt = &v[hh * hd..(hh + 1) * hd];
            let qt = &q[hh * hd..(hh + 1) * hd];
            let beta = sigmoid(b_in[hh]);
            let dc = &decay[hh * hd..(hh + 1) * hd];
            for dv in 0..hd {
                for dk in 0..hd {
                    stm[dv * hd + dk] *= dc[dk];
                }
            }
            let mut delta = vec![0f32; hd];
            for dv in 0..hd {
                let mut m = 0f32;
                for dk in 0..hd {
                    m += stm[dv * hd + dk] * kt[dk];
                }
                delta[dv] = (vt[dv] - m) * beta;
            }
            for dv in 0..hd {
                for dk in 0..hd {
                    stm[dv * hd + dk] += kt[dk] * delta[dv];
                }
            }
            let yo = &mut y[hh * hd..(hh + 1) * hd];
            for dv in 0..hd {
                let mut o = 0f32;
                for dk in 0..hd {
                    o += stm[dv * hd + dk] * qt[dk];
                }
                yo[dv] = o;
            }
        }

        let ga = matmul_wt(x, 1, h, &w.g_a, hd);
        let gate = matmul_wt(&ga, 1, hd, &w.g_b, proj);
        for hh in 0..nh {
            let s = &mut y[hh * hd..(hh + 1) * hd];
            let ms: f32 = s.iter().map(|z| z * z).sum::<f32>() / hd as f32;
            let scale = 1.0 / (ms + w.eps).sqrt();
            for (i, z) in s.iter_mut().enumerate() {
                *z = *z * scale * w.o_norm[i];
            }
        }
        for i in 0..proj {
            y[i] *= sigmoid(gate[i]);
        }
        matmul_wt(&y, 1, proj, &w.o_proj, h)
    }
}

/// Multi-head Latent Attention (MLA) — CPU reference, **materialized-MHA** path
/// (bring-up). Decompresses the 512 KV latent to full per-head K/V; assembles the
/// per-head key as `[k_nope(128) || k_pe(64, MQA-shared, un-rotated)]`; runs
/// `cpu_sdpa_mla` with **qk_dim=192 != v_dim=128** and **scale=192^-0.5**. No RoPE
/// (pe is un-rotated), no output gate, no o_norm. `kv_a_layernorm` (RMSNorm with
/// weight, eps=rms_norm_eps) sits on the 512 latent. Reproduces the mlx-lm
/// `KimiMLAAttention` prefill branch bit-exact.
pub mod mla {
    fn matmul_wt(x: &[f32], rows: usize, inn: usize, w: &[f32], out: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * out];
        for r in 0..rows {
            let xr = &x[r * inn..(r + 1) * inn];
            for o in 0..out {
                let wr = &w[o * inn..(o + 1) * inn];
                let mut acc = 0f32;
                for i in 0..inn {
                    acc += xr[i] * wr[i];
                }
                y[r * out + o] = acc;
            }
        }
        y
    }

    pub struct MlaWeights {
        pub h: usize,
        pub nh: usize,
        pub nope: usize,
        pub pe: usize,
        pub v: usize,
        pub r: usize, // kv_lora_rank
        pub eps: f32,
        pub q_proj: Vec<f32>,        // [nh*(nope+pe), h]
        pub kv_a_proj: Vec<f32>,     // [r+pe, h]
        pub kv_a_layernorm: Vec<f32>, // [r]
        pub embed_q: Vec<f32>,       // [nh, r, nope]
        pub unembed_out: Vec<f32>,   // [nh, v, r]
        pub o_proj: Vec<f32>,        // [h, nh*v]
    }

    /// `cpu_sdpa_mla` — softmax attention with **two head-dims**: score dim
    /// `qk_dim = nope + pe` (Q·K over k_nope plus the shared pe), value dim `v`,
    /// scale applied to the full qk. Causal. This is the ~40-line variant the plan
    /// calls out (`cpu_sdpa` hard-codes one head_dim).
    #[allow(clippy::too_many_arguments)]
    fn cpu_sdpa_mla(
        q_nope: &[f32], // [L, nh, nope]
        q_pe: &[f32],   // [L, nh, pe]
        k_nope: &[f32], // [L, nh, nope]
        k_pe: &[f32],   // [L, pe]  (shared across heads)
        vmat: &[f32],   // [L, nh, v]
        l: usize,
        nh: usize,
        nope: usize,
        pe: usize,
        v: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; l * nh * v];
        let mut scores = vec![0f32; l];
        for h in 0..nh {
            for i in 0..l {
                let qn = &q_nope[(i * nh + h) * nope..(i * nh + h + 1) * nope];
                let qp = &q_pe[(i * nh + h) * pe..(i * nh + h + 1) * pe];
                let mut maxs = f32::NEG_INFINITY;
                for j in 0..=i {
                    let kn = &k_nope[(j * nh + h) * nope..(j * nh + h + 1) * nope];
                    let kp = &k_pe[j * pe..(j + 1) * pe];
                    let mut s = 0f32;
                    for d in 0..nope {
                        s += qn[d] * kn[d];
                    }
                    for d in 0..pe {
                        s += qp[d] * kp[d];
                    }
                    s *= scale;
                    scores[j] = s;
                    if s > maxs {
                        maxs = s;
                    }
                }
                let mut denom = 0f32;
                for j in 0..=i {
                    scores[j] = (scores[j] - maxs).exp();
                    denom += scores[j];
                }
                let ov = &mut out[(i * nh + h) * v..(i * nh + h + 1) * v];
                for j in 0..=i {
                    let wj = scores[j] / denom;
                    let vj = &vmat[(j * nh + h) * v..(j * nh + h + 1) * v];
                    for d in 0..v {
                        ov[d] += wj * vj[d];
                    }
                }
            }
        }
        out
    }

    /// Materialized MLA forward. `x[L, H]` -> `[L, H]`.
    pub fn forward(w: &MlaWeights, x: &[f32], l: usize) -> Vec<f32> {
        let (h, nh, nope, pe, v, r) = (w.h, w.nh, w.nope, w.pe, w.v, w.r);
        let qhd = nope + pe;
        let scale = (qhd as f32).powf(-0.5);

        // Q (uncompressed): [L, nh*qhd] -> split nope/pe
        let q = matmul_wt(x, l, h, &w.q_proj, nh * qhd);
        let mut q_nope = vec![0f32; l * nh * nope];
        let mut q_pe = vec![0f32; l * nh * pe];
        for i in 0..l {
            for hh in 0..nh {
                let base = (i * nh + hh) * qhd;
                q_nope[(i * nh + hh) * nope..(i * nh + hh + 1) * nope]
                    .copy_from_slice(&q[base..base + nope]);
                q_pe[(i * nh + hh) * pe..(i * nh + hh + 1) * pe]
                    .copy_from_slice(&q[base + nope..base + qhd]);
            }
        }

        // KV latent + shared pe key
        let kva = matmul_wt(x, l, h, &w.kv_a_proj, r + pe);
        let mut c_kv = vec![0f32; l * r];
        let mut k_pe = vec![0f32; l * pe];
        for i in 0..l {
            c_kv[i * r..(i + 1) * r].copy_from_slice(&kva[i * (r + pe)..i * (r + pe) + r]);
            k_pe[i * pe..(i + 1) * pe]
                .copy_from_slice(&kva[i * (r + pe) + r..i * (r + pe) + r + pe]);
        }
        // kv_a_layernorm (RMSNorm with weight) on the 512 latent
        for i in 0..l {
            let s = &mut c_kv[i * r..(i + 1) * r];
            let ms: f32 = s.iter().map(|z| z * z).sum::<f32>() / r as f32;
            let inv = 1.0 / (ms + w.eps).sqrt();
            for (d, z) in s.iter_mut().enumerate() {
                *z = *z * inv * w.kv_a_layernorm[d];
            }
        }

        // decompress: K_nope[l,h,n] = sum_r c_kv[l,r]*embed_q[h,r,n]
        //             V[l,h,vv]    = sum_r c_kv[l,r]*unembed_out[h,vv,r]
        let mut k_nope = vec![0f32; l * nh * nope];
        let mut vmat = vec![0f32; l * nh * v];
        for i in 0..l {
            let cl = &c_kv[i * r..(i + 1) * r];
            for hh in 0..nh {
                let eqh = &w.embed_q[hh * r * nope..(hh + 1) * r * nope]; // [r, nope]
                let kn = &mut k_nope[(i * nh + hh) * nope..(i * nh + hh + 1) * nope];
                for n in 0..nope {
                    let mut acc = 0f32;
                    for rr in 0..r {
                        acc += cl[rr] * eqh[rr * nope + n];
                    }
                    kn[n] = acc;
                }
                let uoh = &w.unembed_out[hh * v * r..(hh + 1) * v * r]; // [v, r]
                let vv = &mut vmat[(i * nh + hh) * v..(i * nh + hh + 1) * v];
                for d in 0..v {
                    let mut acc = 0f32;
                    for rr in 0..r {
                        acc += cl[rr] * uoh[d * r + rr];
                    }
                    vv[d] = acc;
                }
            }
        }

        let attn = cpu_sdpa_mla(&q_nope, &q_pe, &k_nope, &k_pe, &vmat, l, nh, nope, pe, v, scale);
        matmul_wt(&attn, l, nh * v, &w.o_proj, h)
    }

    // ------------------------- resident single-token DECODE -------------------------
    // Materialized-MHA KV cache: append this token's decompressed per-head k_nope /
    // shared k_pe / v, then attend over the whole cache. Bit-identical to row `t` of
    // `forward` (the same causal `cpu_sdpa_mla` inner loop `j=0..=t`, same latent
    // decompress + `kv_a_layernorm`), so `decode_step` reproduces prefill exactly.

    /// Resident MLA KV cache: per-head `k_nope` `[T*nh*nope]`, shared `k_pe`
    /// `[T*pe]`, per-head `v` `[T*nh*v]`, plus the token count.
    pub struct MlaCache {
        k_nope: Vec<f32>,
        k_pe: Vec<f32>,
        vmat: Vec<f32>,
        t: usize,
    }
    impl MlaCache {
        pub fn new() -> Self {
            MlaCache { k_nope: Vec::new(), k_pe: Vec::new(), vmat: Vec::new(), t: 0 }
        }
        pub fn len(&self) -> usize {
            self.t
        }
        pub fn is_empty(&self) -> bool {
            self.t == 0
        }
    }
    impl Default for MlaCache {
        fn default() -> Self {
            Self::new()
        }
    }

    /// `kv_a_layernorm` (RMSNorm-with-weight, eps=`rms_norm_eps`) applied in place
    /// to the 512-wide KV latent. Extracted so the CPU oracle (`decode_step`) and
    /// the GPU-resident MLA path apply it byte-identically (same sequential
    /// sum-of-squares, same `1/sqrt(ms+eps)` scaling) — this is part of the host
    /// SDPA seam that must not drift between the two projection paths.
    pub fn kv_a_layernorm_apply(c_kv: &mut [f32], weight: &[f32], eps: f32) {
        let r = c_kv.len();
        let ms: f32 = c_kv.iter().map(|z| z * z).sum::<f32>() / r as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (d, z) in c_kv.iter_mut().enumerate() {
            *z = *z * inv * weight[d];
        }
    }

    /// The host softmax-SDPA seam: append this token's decompressed per-head
    /// `k_nope` `[nh*nope]`, shared `k_pe` `[pe]`, and per-head `v` `[nh*v]` to the
    /// cache, then attend over `0..=t` (== `cpu_sdpa_mla`'s query row `t`) and
    /// return the pre-`o_proj` attention output `[nh*v]`. This is the bit-exact
    /// attention seam shared by the CPU oracle (`decode_step`) and the GPU-resident
    /// MLA path: the ONLY difference between those two paths is HOW `q` / `kva` /
    /// the `k_nope`,`v` decompress are computed (host `matmul_wt` vs GPU
    /// `matvec_mlx4`, an accum-order-only delta) — everything from the KV append
    /// through the softmax to the value combine is this identical host code.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_append(
        c: &mut MlaCache,
        q_nope: &[f32],  // [nh*nope]
        q_pe: &[f32],    // [nh*pe]
        kn_new: &[f32],  // [nh*nope]
        kpe_new: &[f32], // [pe]
        v_new: &[f32],   // [nh*v]
        nh: usize,
        nope: usize,
        pe: usize,
        v: usize,
        scale: f32,
    ) -> Vec<f32> {
        c.k_nope.extend_from_slice(kn_new);
        c.k_pe.extend_from_slice(kpe_new);
        c.vmat.extend_from_slice(v_new);
        c.t += 1;
        let ti = c.t - 1; // current query index

        let mut out_attn = vec![0f32; nh * v];
        let mut scores = vec![0f32; c.t];
        for hh in 0..nh {
            let qn = &q_nope[hh * nope..(hh + 1) * nope];
            let qp = &q_pe[hh * pe..(hh + 1) * pe];
            let mut maxs = f32::NEG_INFINITY;
            for j in 0..=ti {
                let kn = &c.k_nope[(j * nh + hh) * nope..(j * nh + hh + 1) * nope];
                let kp = &c.k_pe[j * pe..(j + 1) * pe];
                let mut s = 0f32;
                for d in 0..nope {
                    s += qn[d] * kn[d];
                }
                for d in 0..pe {
                    s += qp[d] * kp[d];
                }
                s *= scale;
                scores[j] = s;
                if s > maxs {
                    maxs = s;
                }
            }
            let mut denom = 0f32;
            for j in 0..=ti {
                scores[j] = (scores[j] - maxs).exp();
                denom += scores[j];
            }
            let ov = &mut out_attn[hh * v..(hh + 1) * v];
            for j in 0..=ti {
                let wj = scores[j] / denom;
                let vj = &c.vmat[(j * nh + hh) * v..(j * nh + hh + 1) * v];
                for d in 0..v {
                    ov[d] += wj * vj[d];
                }
            }
        }
        out_attn
    }

    /// Single-token MLA decode. `x[H]` -> `[H]`, appending to `c` and attending over
    /// it. Mirrors `forward` / `cpu_sdpa_mla` for query index `t = c.t` exactly.
    pub fn decode_step(w: &MlaWeights, x: &[f32], c: &mut MlaCache) -> Vec<f32> {
        let (h, nh, nope, pe, v, r) = (w.h, w.nh, w.nope, w.pe, w.v, w.r);
        let qhd = nope + pe;
        let scale = (qhd as f32).powf(-0.5);

        // Q (uncompressed) split nope/pe.
        let q = matmul_wt(x, 1, h, &w.q_proj, nh * qhd);
        let mut q_nope = vec![0f32; nh * nope];
        let mut q_pe = vec![0f32; nh * pe];
        for hh in 0..nh {
            let base = hh * qhd;
            q_nope[hh * nope..(hh + 1) * nope].copy_from_slice(&q[base..base + nope]);
            q_pe[hh * pe..(hh + 1) * pe].copy_from_slice(&q[base + nope..base + qhd]);
        }

        // KV latent + shared pe key, kv_a_layernorm on the latent.
        let kva = matmul_wt(x, 1, h, &w.kv_a_proj, r + pe);
        let mut c_kv = kva[..r].to_vec();
        let kpe_new = kva[r..r + pe].to_vec();
        kv_a_layernorm_apply(&mut c_kv, &w.kv_a_layernorm, w.eps);

        // decompress this token's per-head k_nope / v, append to the cache.
        let mut kn_new = vec![0f32; nh * nope];
        let mut v_new = vec![0f32; nh * v];
        for hh in 0..nh {
            let eqh = &w.embed_q[hh * r * nope..(hh + 1) * r * nope];
            for n in 0..nope {
                let mut acc = 0f32;
                for rr in 0..r {
                    acc += c_kv[rr] * eqh[rr * nope + n];
                }
                kn_new[hh * nope + n] = acc;
            }
            let uoh = &w.unembed_out[hh * v * r..(hh + 1) * v * r];
            for d in 0..v {
                let mut acc = 0f32;
                for rr in 0..r {
                    acc += c_kv[rr] * uoh[d * r + rr];
                }
                v_new[hh * v + d] = acc;
            }
        }
        // append + attend over 0..=ti (== cpu_sdpa_mla's query row ti) — the host
        // SDPA seam shared bit-exactly with the GPU-resident MLA path.
        let out_attn =
            attend_append(c, &q_nope, &q_pe, &kn_new, &kpe_new, &v_new, nh, nope, pe, v, scale);
        matmul_wt(&out_attn, 1, nh * v, &w.o_proj, h)
    }
}

/// Kimi MoE router + block — CPU reference. **sigmoid** router (DeepSeek-V3 style,
/// NOT qwen's softmax): selection by `sigmoid(logits) + e_score_correction_bias`
/// (bias affects SELECTION ONLY), combine weights are the **un-biased** sigmoid
/// scores of the chosen top-k, renormalized `/(sum+1e-20)`, ×`routed_scaling_factor`.
/// Shared expert is **UNGATED** (added straight in; no qwen `shared_expert_gate`).
/// `num_expert_group=topk_group=1` → grouped-topk degenerates to plain top-k.
pub mod moe {
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    #[inline]
    fn silu(x: f32) -> f32 {
        x * sigmoid(x)
    }

    /// Router: returns (selected expert indices, combine weights) for one token.
    /// `inds` are returned sorted-descending by selection score (deterministic);
    /// the caller treats them as a set / index→weight map.
    pub fn route(
        logits: &[f32],
        bias: &[f32],
        top_k: usize,
        routed_scaling_factor: f32,
        renormalize: bool,
    ) -> (Vec<usize>, Vec<f32>) {
        let e = logits.len();
        let scores: Vec<f32> = logits.iter().map(|z| sigmoid(*z)).collect();
        let biased: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
        // top_k by biased score (selection); tie-break lower index first.
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| {
            biased[b]
                .partial_cmp(&biased[a])
                .unwrap()
                .then(a.cmp(&b))
        });
        let inds: Vec<usize> = order[..top_k].to_vec();
        // combine weights = UN-biased sigmoid scores at selected inds.
        let mut w: Vec<f32> = inds.iter().map(|&i| scores[i]).collect();
        if top_k > 1 && renormalize {
            let denom: f32 = w.iter().sum::<f32>() + 1e-20;
            for x in w.iter_mut() {
                *x /= denom;
            }
        }
        for x in w.iter_mut() {
            *x *= routed_scaling_factor;
        }
        (inds, w)
    }

    /// `y[o] = sum_i w[o, i] * x[i]` for `w[out, in]`.
    fn mv(w: &[f32], x: &[f32], out: usize, inn: usize) -> Vec<f32> {
        let mut y = vec![0f32; out];
        for o in 0..out {
            let wr = &w[o * inn..(o + 1) * inn];
            let mut acc = 0f32;
            for i in 0..inn {
                acc += wr[i] * x[i];
            }
            y[o] = acc;
        }
        y
    }

    /// GLU MLP: `down(silu(gate(x)) * up(x))`. gate/up `[inter,h]`, down `[h,inter]`.
    fn glu(x: &[f32], gate: &[f32], up: &[f32], down: &[f32], h: usize, inter: usize) -> Vec<f32> {
        let g = mv(gate, x, inter, h);
        let u = mv(up, x, inter, h);
        let mut hid = vec![0f32; inter];
        for i in 0..inter {
            hid[i] = silu(g[i]) * u[i];
        }
        mv(down, &hid, h, inter)
    }

    #[allow(clippy::too_many_arguments)]
    pub struct MoeWeights {
        pub h: usize,
        pub e: usize,
        pub top_k: usize,
        pub inter: usize,
        pub scale: f32,
        pub gate: Vec<f32>, // router [e, h]
        pub bias: Vec<f32>, // [e]
        pub sw_gate: Vec<f32>, // [e, inter, h]
        pub sw_up: Vec<f32>,   // [e, inter, h]
        pub sw_down: Vec<f32>, // [e, h, inter]
        pub sh_gate: Vec<f32>, // shared [inter, h]
        pub sh_up: Vec<f32>,
        pub sh_down: Vec<f32>, // [h, inter]
    }

    /// Full MoE block forward for one row `x[h]` -> `[h]` (returns the selected
    /// inds/weights too, for the router gate).
    pub fn block(w: &MoeWeights, x: &[f32]) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
        let (h, inter) = (w.h, w.inter);
        let logits = mv(&w.gate, x, w.e, h);
        let (inds, weights) = route(&logits, &w.bias, w.top_k, w.scale, true);
        let mut acc = vec![0f32; h];
        for (k, &e) in inds.iter().enumerate() {
            let g = &w.sw_gate[e * inter * h..(e + 1) * inter * h];
            let u = &w.sw_up[e * inter * h..(e + 1) * inter * h];
            let d = &w.sw_down[e * h * inter..(e + 1) * h * inter];
            let ye = glu(x, g, u, d, h, inter);
            for i in 0..h {
                acc[i] += weights[k] * ye[i];
            }
        }
        // ungated shared expert
        let sh = glu(x, &w.sh_gate, &w.sh_up, &w.sh_down, h, inter);
        for i in 0..h {
            acc[i] += sh[i];
        }
        (acc, inds, weights)
    }
}

// ======================= P4½ integration + P5 offline prep =======================
// `KimiModel` product type (the lib.rs-facing type, sibling to `NemotronModel`)
// = a PP window `[layer_start, layer_end)` of resident CPU weights + the
// **27-layer heterogeneous forward assembly** that dispatches KDA / MLA / dense /
// MoE per the `KimiConfig` schedule, feeding each validated block (`kda::forward`,
// `mla::forward`, `moe::block`) the right input. Plus the **`[start,end)` PP-window
// loader** (real mlx4 dequant through OUR `dequantize_mlx_affine`, opening only the
// shards that host the window's layers) and the **footprint-minimax PP-split
// calculator** (config-only, offline — like `scripts/qwen35_pp_split.py`, but
// heterogeneous: KDA/MLA/MoE per-layer bytes differ + the untied vocab is charged
// to the edge stages).

/// Dense (SwiGLU) MLP — the layer-0 MLP only (`first_k_dense_replace=1`, inter 9216).
pub struct DenseMlp {
    pub h: usize,
    pub inter: usize,
    pub gate: Vec<f32>, // [inter, h]
    pub up: Vec<f32>,   // [inter, h]
    pub down: Vec<f32>, // [h, inter]
}

/// The attention block a layer dispatches to (per the schedule).
pub enum KimiAttn {
    Kda(kda::KdaWeights),
    Mla(mla::MlaWeights),
}
/// The MLP block a layer dispatches to (dense on layer 0, MoE on the rest).
pub enum KimiMlp {
    Dense(DenseMlp),
    Moe(moe::MoeWeights),
}

/// One decoder layer's resident weights. `input_ln`/`post_ln` are the RMSNorm
/// weights around the attention/MLP sub-blocks; `attn`/`mlp` are dispatched by the
/// `KimiConfig` schedule, exactly reproducing the mlx-lm `KimiDecoderLayer`:
/// `h = x + attn(input_ln(x)); out = h + mlp(post_ln(h))`.
pub struct KimiLayer {
    pub idx: usize, // GLOBAL layer index (window-independent)
    pub kind: KimiLayerKind,
    pub input_ln: Vec<f32>, // [H]
    pub post_ln: Vec<f32>,  // [H]
    pub attn: KimiAttn,
    pub mlp: KimiMlp,
}

/// Kimi-Linear-48B-A3B, one PP window `[layer_start, layer_end)` resident on CPU.
/// The lib.rs-facing product type (sibling to `nemotron::NemotronModel`): built by
/// `load_cpu` from the on-disk mlx4 checkpoint, run by `forward`.
pub struct KimiModel {
    pub cfg: KimiConfig,
    pub layer_start: usize,
    pub layer_end: usize,
    pub layers: Vec<KimiLayer>, // len == layer_end - layer_start
    // Edge tensors, present only when the window owns that edge AND `load_edges`:
    pub embed: Option<Vec<f32>>, // [vocab, H]  (layer_start == 0)
    pub final_norm: Option<Vec<f32>>, // [H]     (layer_end == num_hidden_layers)
    pub lm_head: Option<Vec<f32>>, // [vocab, H] untied (layer_end == num_hidden_layers)
    /// Resident per-layer decode state (len == layers.len()), advancing in place
    /// across `forward_pp_stage` calls. Empty until a decode session starts (or
    /// re-init via `reset_decode_state`).
    pub states: Vec<KimiLayerState>,
    /// GPU QUANT-RESIDENT decode stage (Phase C). When `Some`, `forward_pp_stage`
    /// dispatches to it (packed 4-bit experts held on-GPU, top-8 dequant/token) —
    /// the path that fits a 14GB node. Built by `load_gpu_resident`; when `None`
    /// the CPU resident decode runs (fine for small windows / bit-exact gates).
    pub gpu: Option<crate::kimi_gpu::KimiGpuStage>,
    /// Persistent RDMA-registered PP-hop scratch (`[H]` f32 each), pinned ONCE
    /// with `vcclCommRegister` on the first `pp_step_kimi`. The native fused hop
    /// recvs into `pp_recv_scratch` and sends from `pp_send_scratch`, so vCCL
    /// skips the per-call `ibv_reg_mr`/dereg temp-MR (the "buffer not registered"
    /// warning + the ~700 ms/tok Kimi PP-3 comm floor). `*_handle == 0` ⇒ not
    /// registered (falls back to the fresh-Vec recv_f32/send_f32 path). Wired
    /// from `lib.rs::pp_step_kimi` (which owns the comm handle); addresses stay
    /// stable because the Vecs are allocated once and never resized.
    pub pp_recv_scratch: Vec<f32>,
    pub pp_recv_handle: usize,
    pub pp_send_scratch: Vec<f32>,
    pub pp_send_handle: usize,
}

// ------------------------------- math helpers -------------------------------

#[inline]
fn kimi_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Weighted RMSNorm over the last `h` dims: `x[rows,h] -> [rows,h]`.
pub(crate) fn rmsnorm(x: &[f32], rows: usize, h: usize, w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; rows * h];
    for r in 0..rows {
        let s = &x[r * h..(r + 1) * h];
        let ms: f32 = s.iter().map(|z| z * z).sum::<f32>() / h as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        let o = &mut y[r * h..(r + 1) * h];
        for i in 0..h {
            o[i] = s[i] * inv * w[i];
        }
    }
    y
}

/// Dense SwiGLU MLP forward: `x[l,h] -> [l,h]`.
pub(crate) fn dense_forward(d: &DenseMlp, x: &[f32], l: usize) -> Vec<f32> {
    let (h, inter) = (d.h, d.inter);
    let mut out = vec![0f32; l * h];
    for t in 0..l {
        let xt = &x[t * h..(t + 1) * h];
        // gate/up: [inter,h] @ x
        let mut hid = vec![0f32; inter];
        for o in 0..inter {
            let gr = &d.gate[o * h..(o + 1) * h];
            let ur = &d.up[o * h..(o + 1) * h];
            let (mut g, mut u) = (0f32, 0f32);
            for i in 0..h {
                g += gr[i] * xt[i];
                u += ur[i] * xt[i];
            }
            hid[o] = (g * kimi_sigmoid(g)) * u; // silu(gate)*up
        }
        // down: [h,inter] @ hid
        let ot = &mut out[t * h..(t + 1) * h];
        for o in 0..h {
            let dr = &d.down[o * inter..(o + 1) * inter];
            let mut acc = 0f32;
            for i in 0..inter {
                acc += dr[i] * hid[i];
            }
            ot[o] = acc;
        }
    }
    out
}

impl KimiModel {
    /// Apply one decoder layer (dispatch attention + MLP by the schedule). This is
    /// the assembly's per-layer primitive: `h = x + attn(input_ln(x)); h + mlp(post_ln(h))`.
    /// `x[l,H] -> [l,H]`.
    pub fn layer_forward(&self, layer: &KimiLayer, x: &[f32], l: usize) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;

        // --- attention sub-block ---
        let xn = rmsnorm(x, l, h, &layer.input_ln, eps);
        let attn = match &layer.attn {
            KimiAttn::Kda(w) => {
                let mut state = vec![0f32; w.nh * w.hd * w.hd];
                kda::forward(w, &xn, l, &mut state)
            }
            KimiAttn::Mla(w) => mla::forward(w, &xn, l),
        };
        let mut hres = vec![0f32; l * h];
        for i in 0..l * h {
            hres[i] = x[i] + attn[i];
        }
        if let Ok(dir) = std::env::var("KIMI_P5_LAYERDUMP") {
            let bytes: Vec<u8> = hres.iter().flat_map(|z| z.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/rust_attn_{}.f32", layer.idx), &bytes);
        }

        // --- MLP sub-block ---
        let hn = rmsnorm(&hres, l, h, &layer.post_ln, eps);
        let mlp = match &layer.mlp {
            KimiMlp::Dense(d) => dense_forward(d, &hn, l),
            KimiMlp::Moe(w) => {
                let mut out = vec![0f32; l * h];
                for t in 0..l {
                    let (o, _, _) = moe::block(w, &hn[t * h..(t + 1) * h]);
                    out[t * h..(t + 1) * h].copy_from_slice(&o);
                }
                out
            }
        };
        let mut out = vec![0f32; l * h];
        for i in 0..l * h {
            out[i] = hres[i] + mlp[i];
        }
        out
    }

    /// Full PP-window forward over hidden states `h[l,H]`. Runs every layer in the
    /// window in schedule order. (Embedding lookup and the tail `final_norm`/`lm_head`
    /// projection are the caller's responsibility / a P5 tail-stage follow-on; this
    /// is the heterogeneous layer-stack assembly the gate exercises.)
    pub fn forward(&self, mut h: Vec<f32>, l: usize) -> Vec<f32> {
        for layer in &self.layers {
            h = self.layer_forward(layer, &h, l);
        }
        h
    }
}

// ------------------------- resident single-token DECODE (PP-stage core) -------------------------
// The substantial new machinery: a resident per-layer decode state that advances IN
// PLACE across single-token steps (KDA recurrence + conv sliding-window; MLA KV
// cache), so decode is O(1) in sequence length. This is the shared core between a
// shipping single-token decode method AND `forward_pp_stage` (the nemotron
// `LayerState`/`forward_pp_range` analog). Bit-identical to the stateless
// fresh-prefill `forward` on the same input stream (gated below): each layer's decode
// step reproduces the corresponding row of `layer_forward`.

/// Per-layer resident decode state (dispatched by the layer's attention kind).
/// MLP sub-blocks (dense / MoE) are stateless per token.
pub enum KimiLayerState {
    Kda(kda::KdaState),
    Mla(mla::MlaCache),
}

impl KimiModel {
    /// Allocate zero-initialized decode state for every layer in this window.
    pub fn init_decode_state(&self) -> Vec<KimiLayerState> {
        self.layers
            .iter()
            .map(|ly| match &ly.attn {
                KimiAttn::Kda(w) => KimiLayerState::Kda(kda::KdaState::new(w.nh, w.hd, w.kern)),
                KimiAttn::Mla(_) => KimiLayerState::Mla(mla::MlaCache::new()),
            })
            .collect()
    }

    /// One decoder layer's single-token decode step, advancing `st` in place.
    /// `x[H] -> [H]`. Reproduces `layer_forward`'s row for this token exactly:
    /// `h = x + attn_step(input_ln(x)); out = h + mlp(post_ln(h))`.
    pub fn layer_decode_step(
        &self,
        layer: &KimiLayer,
        st: &mut KimiLayerState,
        x: &[f32],
    ) -> Vec<f32> {
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;

        let xn = rmsnorm(x, 1, h, &layer.input_ln, eps);
        let attn = match (&layer.attn, st) {
            (KimiAttn::Kda(w), KimiLayerState::Kda(s)) => kda::decode_step(w, &xn, s),
            (KimiAttn::Mla(w), KimiLayerState::Mla(c)) => mla::decode_step(w, &xn, c),
            _ => panic!("kimi decode: layer {} attn kind vs state mismatch", layer.idx),
        };
        let mut hres = vec![0f32; h];
        for i in 0..h {
            hres[i] = x[i] + attn[i];
        }

        let hn = rmsnorm(&hres, 1, h, &layer.post_ln, eps);
        let mlp = match &layer.mlp {
            KimiMlp::Dense(d) => dense_forward(d, &hn, 1),
            KimiMlp::Moe(w) => moe::block(w, &hn).0,
        };
        let mut out = vec![0f32; h];
        for i in 0..h {
            out[i] = hres[i] + mlp[i];
        }
        out
    }

    /// Single-token resident decode through the whole window `[layer_start, layer_end)`.
    /// `x[H] -> [H]`, advancing every layer's state in place. Chaining this over a
    /// token stream is bit-identical to `forward` over the same stream (each layer's
    /// per-token decode == the matching row of the fresh-prefill scan).
    pub fn decode_step(&self, states: &mut [KimiLayerState], mut x: Vec<f32>) -> Vec<f32> {
        assert_eq!(states.len(), self.layers.len(), "decode state len != window layers");
        for (layer, st) in self.layers.iter().zip(states.iter_mut()) {
            x = self.layer_decode_step(layer, st, &x);
        }
        x
    }

    /// Re-zero the resident decode state (start a fresh decode session).
    pub fn reset_decode_state(&mut self) {
        self.states = self.init_decode_state();
    }

    /// One PP-stage single-token decode step (the nemotron `forward_pp_stage`
    /// analog), advancing this window's resident `self.states` IN PLACE.
    /// - First stage (`layer_start == 0`): embeds `token_id`; `hidden_in` ignored.
    ///   Else: consumes `hidden_in[H]` (the previous stage's output).
    /// - Runs the window's resident decode (`[layer_start, layer_end)`).
    /// - Last stage (`layer_end == num_hidden_layers`): final RMSNorm + untied
    ///   `lm_head` → `[vocab]` logits. Else returns the `[H]` hidden to ship onward.
    ///
    /// Only the `[H]` hidden (or the tail `[vocab]` logits) crosses a PP hop; the
    /// recurrence/KV state lives entirely on its owning stage.
    pub fn forward_pp_stage(&mut self, token_id: u32, hidden_in: &[f32], _pos: usize) -> Vec<f32> {
        // Phase C: dispatch to the GPU quant-resident stage when present.
        if self.gpu.is_some() {
            return self
                .gpu
                .as_mut()
                .unwrap()
                .forward_pp_stage(token_id, hidden_in)
                .expect("kimi gpu forward_pp_stage");
        }
        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;
        let first = self.layer_start == 0;
        let last = self.layer_end == self.cfg.num_hidden_layers;
        if self.states.len() != self.layers.len() {
            self.states = self.init_decode_state();
        }

        // stage input hidden
        let mut x = if first {
            let emb = self.embed.as_ref().expect("stage 0 requires embed (load_edges)");
            let row = token_id as usize * h;
            emb[row..row + h].to_vec()
        } else {
            assert_eq!(hidden_in.len(), h, "PP hidden_in wrong size");
            hidden_in.to_vec()
        };

        // resident window decode (disjoint borrows: &self.cfg/&self.layers[i] +
        // &mut self.states[i] via a taken-out states vec)
        let mut states = std::mem::take(&mut self.states);
        for (layer, st) in self.layers.iter().zip(states.iter_mut()) {
            x = self.layer_decode_step(layer, st, &x);
        }
        self.states = states;

        if last {
            let fnorm = self.final_norm.as_ref().expect("tail stage requires final_norm");
            let lm = self.lm_head.as_ref().expect("tail stage requires lm_head");
            let normed = rmsnorm(&x, 1, h, fnorm, eps);
            let vocab = lm.len() / h;
            let mut logits = vec![0f32; vocab];
            for o in 0..vocab {
                let wr = &lm[o * h..(o + 1) * h];
                let mut acc = 0f32;
                for i in 0..h {
                    acc += wr[i] * normed[i];
                }
                logits[o] = acc;
            }
            logits
        } else {
            x
        }
    }
}

/// Per-layer resident footprint (bytes) for the PP-split calculator. Config-only.
mod pp_footprint {
    use super::{KimiConfig, KimiLayerKind};

    /// mlx-affine group-64 quant footprint: `bits/8` weight bytes/param + the bf16
    /// scales+biases surcharge (2 bf16 per `group_size` params = 4 bytes/64 = 0.0625).
    fn quant_bytes(numel: usize, bits: usize) -> f64 {
        numel as f64 * (bits as f64 / 8.0 + 4.0 / 64.0)
    }
    /// Raw (un-quantized) tensor bytes, bf16 (2B) unless noted.
    fn raw_bytes(numel: usize, bytes_each: f64) -> f64 {
        numel as f64 * bytes_each
    }

    fn kda_bytes(c: &KimiConfig) -> f64 {
        let (h, nh, hd) = (c.hidden_size, c.kda_num_heads, c.kda_head_dim);
        let proj = nh * hd;
        let mut b = 0.0;
        // q/k/v_proj [proj,h] 4-bit
        b += 3.0 * quant_bytes(proj * h, 4);
        // convs [proj,kern] raw bf16
        b += 3.0 * raw_bytes(proj * c.kda_conv_kernel, 2.0);
        // f_a[hd,h] f_b[proj,hd] b_proj[nh,h] g_a[hd,h] g_b[proj,hd] 4-bit
        b += quant_bytes(hd * h, 4) + quant_bytes(proj * hd, 4) + quant_bytes(nh * h, 4);
        b += quant_bytes(hd * h, 4) + quant_bytes(proj * hd, 4);
        // A_log[nh] f32, dt_bias[proj] f32, o_norm[hd] bf16
        b += raw_bytes(nh, 4.0) + raw_bytes(proj, 4.0) + raw_bytes(hd, 2.0);
        // o_proj [h,proj] 4-bit
        b += quant_bytes(h * proj, 4);
        b
    }

    fn mla_bytes(c: &KimiConfig) -> f64 {
        let (h, nh, r) = (c.hidden_size, c.num_attention_heads, c.kv_lora_rank);
        let qhd = c.q_head_dim();
        let (nope, v, pe) = (c.qk_nope_head_dim, c.v_head_dim, c.qk_rope_head_dim);
        let mut b = 0.0;
        b += quant_bytes(nh * qhd * h, 4); // q_proj
        b += quant_bytes((r + pe) * h, 4); // kv_a_proj_with_mqa
        b += raw_bytes(r, 2.0); // kv_a_layernorm bf16
        b += quant_bytes(nh * (nope + v) * r, 4); // kv_b_proj (-> embed_q+unembed_out)
        b += quant_bytes(h * (nh * v), 4); // o_proj
        b
    }

    fn moe_bytes(c: &KimiConfig) -> f64 {
        let (h, e, inter) = (c.hidden_size, c.num_experts, c.moe_intermediate_size);
        let mut b = 0.0;
        b += quant_bytes(e * h, 8); // router gate 8-bit
        b += raw_bytes(e, 2.0); // e_score_correction_bias bf16
        b += 2.0 * quant_bytes(e * inter * h, 4); // switch gate+up
        b += quant_bytes(e * h * inter, 4); // switch down
        b += 3.0 * quant_bytes(inter * h, 4); // shared gate+up+down
        b
    }

    fn dense_bytes(c: &KimiConfig) -> f64 {
        let (h, inter) = (c.hidden_size, c.intermediate_size);
        3.0 * quant_bytes(inter * h, 4) // gate+up+down
    }

    /// Per-layer resident bytes for GLOBAL layer `i` (attn per schedule, MLP dense
    /// on layer 0 else MoE, + the two RMSNorm weights).
    pub fn layer_bytes(c: &KimiConfig, i: usize) -> f64 {
        let attn = match c.layer_schedule[i] {
            KimiLayerKind::Kda => kda_bytes(c),
            KimiLayerKind::Mla => mla_bytes(c),
        };
        let mlp = if c.is_dense_mlp(i) {
            dense_bytes(c)
        } else {
            moe_bytes(c)
        };
        attn + mlp + raw_bytes(2 * c.hidden_size, 2.0) // input_ln + post_ln bf16
    }

    /// Untied vocab edge tensors: embed (stage 0) and lm_head (last stage), both
    /// `[vocab, hidden]` 4-bit. Charged to their edge stages (nemotron lesson).
    pub fn vocab_bytes(c: &KimiConfig) -> f64 {
        quant_bytes(c.vocab_size * c.hidden_size, 4)
    }
}

impl KimiConfig {
    /// Footprint-minimax PP split: contiguous partition of `[0, num_hidden_layers)`
    /// into `n_stages` that minimizes the per-stage resident bytes LEXICOGRAPHICALLY
    /// (min the max, then the 2nd-largest, ...) — minimax-optimal AND balanced.
    /// Stage 0 also carries `embed`; the last stage also carries the untied
    /// `lm_head` + `final_norm`. Heterogeneous: KDA/MLA/MoE/dense per-layer bytes
    /// differ. Config-only, offline (no weights). Returns `(bounds, loads_gb)` where
    /// `bounds` is the `n_stages+1` GLOBAL layer indices `[0, .., num_hidden_layers]`.
    pub fn pp_split(&self, n_stages: usize) -> (Vec<usize>, Vec<f64>) {
        let n = self.num_hidden_layers;
        let per: Vec<f64> = (0..n).map(|i| pp_footprint::layer_bytes(self, i)).collect();
        let vocab = pp_footprint::vocab_bytes(self);
        // prefix sums for O(1) range cost
        let mut pre = vec![0f64; n + 1];
        for i in 0..n {
            pre[i + 1] = pre[i] + per[i];
        }
        let range = |a: usize, b: usize| pre[b] - pre[a];

        // DP: best(start, stages_left, is_first) -> (sorted-desc load vector, take)
        // memoized over (start, stages_left, is_first).
        let mut memo: HashMap<(usize, usize, bool), (Vec<f64>, usize)> = HashMap::new();
        fn best(
            start: usize,
            stages_left: usize,
            is_first: bool,
            n: usize,
            range: &dyn Fn(usize, usize) -> f64,
            vocab: f64,
            memo: &mut HashMap<(usize, usize, bool), (Vec<f64>, usize)>,
        ) -> (Vec<f64>, usize) {
            if let Some(v) = memo.get(&(start, stages_left, is_first)) {
                return v.clone();
            }
            let remaining = n - start;
            let res = if stages_left == 1 {
                // last stage: all remaining + lm_head (+ embed if also first).
                let load = range(start, n) + vocab + if is_first { vocab } else { 0.0 };
                (vec![load], remaining)
            } else {
                let mut best_cost: Option<Vec<f64>> = None;
                let mut best_take = 0usize;
                // leave >=1 layer per remaining stage
                for take in 1..=(remaining - (stages_left - 1)) {
                    let head = range(start, start + take) + if is_first { vocab } else { 0.0 };
                    let (tail, _) = best(start + take, stages_left - 1, false, n, range, vocab, memo);
                    let mut cost = tail.clone();
                    cost.push(head);
                    cost.sort_by(|a, b| b.partial_cmp(a).unwrap()); // desc
                    let better = match &best_cost {
                        None => true,
                        Some(bc) => cost < *bc, // lexicographic on desc-sorted vecs
                    };
                    if better {
                        best_cost = Some(cost);
                        best_take = take;
                    }
                }
                (best_cost.unwrap(), best_take)
            };
            memo.insert((start, stages_left, is_first), res.clone());
            res
        }

        let mut bounds = vec![0usize];
        let (mut start, mut is_first) = (0usize, true);
        for s in (1..=n_stages).rev() {
            let (_, take) = best(start, s, is_first, n, &range, vocab, &mut memo);
            start += take;
            bounds.push(start);
            is_first = false;
        }
        // per-stage GB loads
        let loads: Vec<f64> = (0..n_stages)
            .map(|i| {
                let mut load = range(bounds[i], bounds[i + 1]);
                if i == 0 {
                    load += vocab;
                }
                if i == n_stages - 1 {
                    load += vocab;
                }
                load / 1e9
            })
            .collect();
        (bounds, loads)
    }
}

// ------------------------------- loader (P5) -------------------------------

/// Read a little-endian bf16 tensor's bytes into f32.
fn bytes_bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    use half::bf16;
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}
/// Read a little-endian f32 tensor's bytes.
fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn bytes_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl KimiModel {
    /// Load a PP window `[layer_start, layer_end)` from the on-disk mlx4 checkpoint
    /// at `ckpt_dir` (reads `model.safetensors.index.json`, opens ONLY the shards
    /// that host the window's layers — the real `[start,end)` window loader P5
    /// needs). Every quantized tensor is dequantized through OUR
    /// `dequantize_mlx_affine` (the real loader seam). `load_edges` also materializes
    /// the untied `embed`/`lm_head`/`final_norm` when the window owns that edge.
    pub fn load_cpu(
        ckpt_dir: &str,
        cfg: &KimiConfig,
        layer_start: usize,
        layer_end: usize,
        load_edges: bool,
    ) -> Result<KimiModel, String> {
        use memmap2::Mmap;
        use safetensors::SafeTensors;
        use std::fs::File;

        if layer_start >= layer_end || layer_end > cfg.num_hidden_layers {
            return Err(format!(
                "bad window [{layer_start},{layer_end}) for {} layers",
                cfg.num_hidden_layers
            ));
        }

        // --- read the index (tensor name -> shard file) ---
        let index_path = format!("{ckpt_dir}/model.safetensors.index.json");
        let raw = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("read {index_path}: {e}"))?;
        let index: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| format!("parse index.json: {e}"))?;
        let wm = index["weight_map"]
            .as_object()
            .ok_or("index.json missing weight_map")?;
        let shard_of = |name: &str| -> Result<String, String> {
            wm.get(name)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| format!("tensor '{name}' not in weight_map"))
        };

        // --- open only the shards this window touches (keep mmaps alive) ---
        // A boundary layer may split across two shards, so we index by shard name
        // and (re)deserialize per access — the header parse is cheap; tensor DATA
        // is mmapped, not copied.
        let mut shard_paths: std::collections::BTreeSet<String> = Default::default();
        {
            // collect every tensor name we will read, to know which shards to open.
            let mut want: Vec<String> = Vec::new();
            for l in layer_start..layer_end {
                for b in layer_tensor_names(cfg, l) {
                    want.push(b);
                }
            }
            if load_edges {
                if layer_start == 0 {
                    want.push("model.embed_tokens.weight".to_string());
                }
                if layer_end == cfg.num_hidden_layers {
                    want.push("lm_head.weight".to_string());
                    want.push("model.norm.weight".to_string());
                }
            }
            for w in &want {
                shard_paths.insert(shard_of(w)?);
            }
        }
        let mut mmaps: HashMap<String, Mmap> = HashMap::new();
        for sp in &shard_paths {
            let f = File::open(format!("{ckpt_dir}/{sp}"))
                .map_err(|e| format!("open shard {sp}: {e}"))?;
            let m = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {sp}: {e}"))?;
            mmaps.insert(sp.clone(), m);
        }

        // Dequantize a quantized tensor `{base}.{weight,scales,biases}` -> flat
        // f32 [out_total, in_features] (bits derived from shapes, group 64).
        let deq = |base: &str| -> Result<Vec<f32>, String> {
            let wname = format!("{base}.weight");
            let sp = shard_of(&wname)?;
            let st = SafeTensors::deserialize(&mmaps[&sp])
                .map_err(|e| format!("deserialize {sp}: {e}"))?;
            let wv = st.tensor(&wname).map_err(|e| format!("{wname}: {e}"))?;
            let sv = st
                .tensor(&format!("{base}.scales"))
                .map_err(|e| format!("{base}.scales: {e}"))?;
            let bv = st
                .tensor(&format!("{base}.biases"))
                .map_err(|e| format!("{base}.biases: {e}"))?;
            let wshape = wv.shape();
            let sshape = sv.shape();
            let packed_last = *wshape.last().unwrap();
            let group_size = 64usize;
            let in_features = sshape.last().unwrap() * group_size;
            let out_total: usize = wshape[..wshape.len() - 1].iter().product();
            let bits = (packed_last * 32) / in_features;
            let w = bytes_u32(wv.data());
            let s = bytes_bf16_to_f32(sv.data());
            let b = bytes_bf16_to_f32(bv.data());
            Ok(crate::model::dequantize_mlx_affine(
                &w, &s, &b, out_total, in_features, group_size, bits,
            ))
        };
        // Read a raw (non-quantized) tensor as f32 (bf16 or f32 on disk).
        let raw = |name: &str| -> Result<Vec<f32>, String> {
            let sp = shard_of(name)?;
            let st = SafeTensors::deserialize(&mmaps[&sp])
                .map_err(|e| format!("deserialize {sp}: {e}"))?;
            let tv = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
            Ok(match tv.dtype() {
                safetensors::Dtype::BF16 => bytes_bf16_to_f32(tv.data()),
                safetensors::Dtype::F32 => bytes_f32(tv.data()),
                d => return Err(format!("{name}: unexpected raw dtype {d:?}")),
            })
        };

        let mut layers = Vec::with_capacity(layer_end - layer_start);
        for l in layer_start..layer_end {
            let p = format!("model.layers.{l}");
            let input_ln = raw(&format!("{p}.input_layernorm.weight"))?;
            let post_ln = raw(&format!("{p}.post_attention_layernorm.weight"))?;

            // --- attention ---
            let attn = match cfg.layer_schedule[l] {
                KimiLayerKind::Kda => {
                    let ap = format!("{p}.self_attn");
                    KimiAttn::Kda(kda::KdaWeights {
                        h: cfg.hidden_size,
                        nh: cfg.kda_num_heads,
                        hd: cfg.kda_head_dim,
                        kern: cfg.kda_conv_kernel,
                        eps: cfg.rms_norm_eps,
                        q_proj: deq(&format!("{ap}.q_proj"))?,
                        k_proj: deq(&format!("{ap}.k_proj"))?,
                        v_proj: deq(&format!("{ap}.v_proj"))?,
                        q_conv: raw(&format!("{ap}.q_conv.conv.weight"))?,
                        k_conv: raw(&format!("{ap}.k_conv.conv.weight"))?,
                        v_conv: raw(&format!("{ap}.v_conv.conv.weight"))?,
                        f_a: deq(&format!("{ap}.f_a_proj"))?,
                        f_b: deq(&format!("{ap}.f_b_proj"))?,
                        b_proj: deq(&format!("{ap}.b_proj"))?,
                        g_a: deq(&format!("{ap}.g_a_proj"))?,
                        g_b: deq(&format!("{ap}.g_b_proj"))?,
                        a_log: raw(&format!("{ap}.A_log"))?,
                        dt_bias: raw(&format!("{ap}.dt_bias"))?,
                        o_norm: raw(&format!("{ap}.o_norm.weight"))?,
                        o_proj: deq(&format!("{ap}.o_proj"))?,
                    })
                }
                KimiLayerKind::Mla => {
                    let ap = format!("{p}.self_attn");
                    let (nh, nope, v, r) =
                        (cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
                    // kv_b_proj [nh*(nope+v), r] -> embed_q[nh,r,nope] + unembed_out[nh,v,r]
                    let head_dim = nope + v;
                    let kvb = deq(&format!("{ap}.kv_b_proj"))?; // [nh*head_dim, r]
                    let mut embed_q = vec![0f32; nh * r * nope];
                    let mut unembed_out = vec![0f32; nh * v * r];
                    for hh in 0..nh {
                        let vb = &kvb[hh * head_dim * r..(hh + 1) * head_dim * r];
                        for n in 0..nope {
                            for rr in 0..r {
                                embed_q[(hh * r + rr) * nope + n] = vb[n * r + rr];
                            }
                        }
                        for d in 0..v {
                            for rr in 0..r {
                                unembed_out[(hh * v + d) * r + rr] = vb[(nope + d) * r + rr];
                            }
                        }
                    }
                    KimiAttn::Mla(mla::MlaWeights {
                        h: cfg.hidden_size,
                        nh,
                        nope,
                        pe: cfg.qk_rope_head_dim,
                        v,
                        r,
                        eps: cfg.rms_norm_eps,
                        q_proj: deq(&format!("{ap}.q_proj"))?,
                        kv_a_proj: deq(&format!("{ap}.kv_a_proj_with_mqa"))?,
                        kv_a_layernorm: raw(&format!("{ap}.kv_a_layernorm.weight"))?,
                        embed_q,
                        unembed_out,
                        o_proj: deq(&format!("{ap}.o_proj"))?,
                    })
                }
            };

            // --- MLP ---
            let mlp = if cfg.is_dense_mlp(l) {
                let mp = format!("{p}.mlp");
                KimiMlp::Dense(DenseMlp {
                    h: cfg.hidden_size,
                    inter: cfg.intermediate_size,
                    gate: deq(&format!("{mp}.gate_proj"))?,
                    up: deq(&format!("{mp}.up_proj"))?,
                    down: deq(&format!("{mp}.down_proj"))?,
                })
            } else {
                let mp = format!("{p}.mlp");
                KimiMlp::Moe(moe::MoeWeights {
                    h: cfg.hidden_size,
                    e: cfg.num_experts,
                    top_k: cfg.num_experts_per_token,
                    inter: cfg.moe_intermediate_size,
                    scale: cfg.routed_scaling_factor,
                    gate: deq(&format!("{mp}.gate"))?,
                    bias: raw(&format!("{mp}.e_score_correction_bias"))?,
                    sw_gate: deq(&format!("{mp}.switch_mlp.gate_proj"))?,
                    sw_up: deq(&format!("{mp}.switch_mlp.up_proj"))?,
                    sw_down: deq(&format!("{mp}.switch_mlp.down_proj"))?,
                    sh_gate: deq(&format!("{mp}.shared_experts.gate_proj"))?,
                    sh_up: deq(&format!("{mp}.shared_experts.up_proj"))?,
                    sh_down: deq(&format!("{mp}.shared_experts.down_proj"))?,
                })
            };

            layers.push(KimiLayer {
                idx: l,
                kind: cfg.layer_schedule[l],
                input_ln,
                post_ln,
                attn,
                mlp,
            });
        }

        let (mut embed, mut final_norm, mut lm_head) = (None, None, None);
        if load_edges {
            if layer_start == 0 {
                embed = Some(deq("model.embed_tokens")?);
            }
            if layer_end == cfg.num_hidden_layers {
                final_norm = Some(raw("model.norm.weight")?);
                lm_head = Some(deq("lm_head")?);
            }
        }

        let mut m = KimiModel {
            cfg: cfg.clone(),
            layer_start,
            layer_end,
            layers,
            embed,
            final_norm,
            lm_head,
            states: Vec::new(),
            gpu: None,
            pp_recv_scratch: Vec::new(),
            pp_recv_handle: 0,
            pp_send_scratch: Vec::new(),
            pp_send_handle: 0,
        };
        m.states = m.init_decode_state();
        Ok(m)
    }

    /// Build a GPU QUANT-RESIDENT stage for window `[layer_start, layer_end)` —
    /// the Phase-C path. Skips `load_cpu`'s bulk MoE-expert dequant (which OOMs a
    /// 14GB node at ~7GB f32/MoE layer); instead the packed 4-bit experts are held
    /// on-GPU and only the top-8 are dequantized per token. `layers`/`states` stay
    /// empty (unused) — `forward_pp_stage` dispatches straight to the GPU stage.
    pub fn load_gpu_resident(
        ckpt_dir: &str,
        cfg: &KimiConfig,
        layer_start: usize,
        layer_end: usize,
        load_edges: bool,
        device_idx: usize,
    ) -> Result<KimiModel, String> {
        let gpu = crate::kimi_gpu::KimiGpuStage::new(
            ckpt_dir, cfg, layer_start, layer_end, load_edges, device_idx,
        )?;
        Ok(KimiModel {
            cfg: cfg.clone(),
            layer_start,
            layer_end,
            layers: Vec::new(),
            embed: None,
            final_norm: None,
            lm_head: None,
            states: Vec::new(),
            gpu: Some(gpu),
            pp_recv_scratch: Vec::new(),
            pp_recv_handle: 0,
            pp_send_scratch: Vec::new(),
            pp_send_handle: 0,
        })
    }
}

/// The set of quantized-base + raw tensor names a GLOBAL layer `l` owns (used only
/// to decide which shards a window touches). Bases here are the `.weight` names
/// (quantized ones also imply `.scales`/`.biases`); raw ones are full names.
fn layer_tensor_names(cfg: &KimiConfig, l: usize) -> Vec<String> {
    let p = format!("model.layers.{l}");
    let mut v = vec![
        format!("{p}.input_layernorm.weight"),
        format!("{p}.post_attention_layernorm.weight"),
    ];
    match cfg.layer_schedule[l] {
        KimiLayerKind::Kda => {
            let ap = format!("{p}.self_attn");
            for n in [
                "q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "b_proj", "g_a_proj",
                "g_b_proj", "o_proj",
            ] {
                v.push(format!("{ap}.{n}.weight"));
            }
            for n in ["q_conv.conv", "k_conv.conv", "v_conv.conv"] {
                v.push(format!("{ap}.{n}.weight"));
            }
            v.push(format!("{ap}.A_log"));
            v.push(format!("{ap}.dt_bias"));
            v.push(format!("{ap}.o_norm.weight"));
        }
        KimiLayerKind::Mla => {
            let ap = format!("{p}.self_attn");
            for n in ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"] {
                v.push(format!("{ap}.{n}.weight"));
            }
            v.push(format!("{ap}.kv_a_layernorm.weight"));
        }
    }
    let mp = format!("{p}.mlp");
    if cfg.is_dense_mlp(l) {
        for n in ["gate_proj", "up_proj", "down_proj"] {
            v.push(format!("{mp}.{n}.weight"));
        }
    } else {
        v.push(format!("{mp}.gate.weight"));
        v.push(format!("{mp}.e_score_correction_bias"));
        for n in [
            "switch_mlp.gate_proj",
            "switch_mlp.up_proj",
            "switch_mlp.down_proj",
            "shared_experts.gate_proj",
            "shared_experts.up_proj",
            "shared_experts.down_proj",
        ] {
            v.push(format!("{mp}.{n}.weight"));
        }
    }
    v
}

#[cfg(test)]
mod p4_tests {
    use super::moe::*;
    use std::collections::HashMap;

    fn read_f32(p: &str) -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn read_i32(p: &str) -> Vec<i32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn kimi_moe_router_and_block_vs_oracle() {
        let dir = match std::env::var("KIMI_P4_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("P4 SKIP: set KIMI_P4_DIR (run kimi_p4_dump_moe.py first)");
                return;
            }
        };
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap())
                .unwrap();
        let mut d = HashMap::new();
        for (k, v) in m["dims"].as_object().unwrap() {
            d.insert(k.clone(), v.as_f64().unwrap());
        }
        let f = |n: &str| read_f32(&format!("{dir}/{n}.f32"));
        let (h, e, top_k, inter, t) = (
            d["H"] as usize, d["E"] as usize, d["TOPK"] as usize,
            d["INTER"] as usize, d["T"] as usize,
        );
        let w = MoeWeights {
            h, e, top_k, inter, scale: d["scale"] as f32,
            gate: f("gate"), bias: f("bias"),
            sw_gate: f("sw_gate"), sw_up: f("sw_up"), sw_down: f("sw_down"),
            sh_gate: f("sh_gate"), sh_up: f("sh_up"), sh_down: f("sh_down"),
        };
        let x = f("x");
        let golden = f("golden");
        let oracle_inds = read_i32(&format!("{dir}/router_inds.i32")); // [T, top_k]
        let oracle_w = f("router_weights"); // [T, top_k]

        let mut worst_mae = 0f64;
        let mut all_am = true;
        for tok in 0..t {
            let xt = &x[tok * h..(tok + 1) * h];
            let (out, inds, weights) = block(&w, xt);

            // ---- router selection SET bit-match ----
            let mut mine_set: Vec<i32> = inds.iter().map(|&i| i as i32).collect();
            mine_set.sort();
            let mut orc_set: Vec<i32> =
                oracle_inds[tok * top_k..(tok + 1) * top_k].to_vec();
            orc_set.sort();
            assert_eq!(mine_set, orc_set, "tok {tok} selection set mismatch");

            // ---- combine weights bit-match (as expert->weight map) ----
            let mut orc_map: HashMap<i32, f32> = HashMap::new();
            for k in 0..top_k {
                orc_map.insert(oracle_inds[tok * top_k + k], oracle_w[tok * top_k + k]);
            }
            let mut w_mae = 0f64;
            for (k, &ei) in inds.iter().enumerate() {
                let ow = orc_map[&(ei as i32)];
                w_mae = w_mae.max((weights[k] - ow).abs() as f64);
            }

            // ---- block output cos/argmax/max_abs_err ----
            let g = &golden[tok * h..(tok + 1) * h];
            let (mut dot, mut na, mut nb, mut mae) = (0f64, 0f64, 0f64, 0f64);
            for (a, b) in g.iter().zip(out.iter()) {
                let (a, b) = (*a as f64, *b as f64);
                dot += a * b; na += a * a; nb += b * b; mae = mae.max((a - b).abs());
            }
            let cos = dot / (na.sqrt() * nb.sqrt());
            let amg = g.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let amm = out.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            worst_mae = worst_mae.max(mae);
            all_am &= amg == amm;
            println!("[MoE tok={tok}] router_w_max_err={w_mae:.3e} block cos={cos:.10} max_abs_err={mae:.3e} argmax g={amg} m={amm} {}", if amg == amm { "OK" } else { "MISMATCH" });
            assert!(w_mae < 1e-6, "tok {tok} combine weight mismatch {w_mae}");
            assert!(cos > 0.999_999_99, "tok {tok} block cos {cos}");
        }
        assert!(all_am, "block argmax mismatch");
        assert!(worst_mae < 1e-4, "worst block max_abs_err {worst_mae}");
    }
}

#[cfg(test)]
mod p3_tests {
    use super::mla::*;
    use std::collections::HashMap;

    fn read_f32(path: &str) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn kimi_mla_vs_oracle() {
        let dir = match std::env::var("KIMI_P3_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("P3 SKIP: set KIMI_P3_DIR (run kimi_p3_dump_mla.py first)");
                return;
            }
        };
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap())
                .unwrap();
        let mut dims = HashMap::new();
        for (k, v) in m["dims"].as_object().unwrap() {
            dims.insert(k.clone(), v.as_f64().unwrap() as usize);
        }
        let f = |n: &str| read_f32(&format!("{dir}/{n}.f32"));
        let w = MlaWeights {
            h: dims["H"], nh: dims["NH"], nope: dims["NOPE"], pe: dims["PE"],
            v: dims["V"], r: dims["R"], eps: m["dims"]["eps"].as_f64().unwrap() as f32,
            q_proj: f("q_proj"), kv_a_proj: f("kv_a_proj"),
            kv_a_layernorm: f("kv_a_layernorm"), embed_q: f("embed_q"),
            unembed_out: f("unembed_out"), o_proj: f("o_proj"),
        };
        let x = f("x");
        let golden = f("golden");
        let l = dims["L"];
        let mine = forward(&w, &x, l);

        // per-token cos/argmax/max_abs_err
        let (h, mut worst_mae, mut all_am) = (dims["H"], 0f64, true);
        for t in 0..l {
            let (g, mm) = (&golden[t * h..(t + 1) * h], &mine[t * h..(t + 1) * h]);
            let (mut dot, mut na, mut nb, mut mae) = (0f64, 0f64, 0f64, 0f64);
            for (a, b) in g.iter().zip(mm.iter()) {
                let (a, b) = (*a as f64, *b as f64);
                dot += a * b; na += a * a; nb += b * b; mae = mae.max((a - b).abs());
            }
            let cos = dot / (na.sqrt() * nb.sqrt());
            let amg = g.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let amm = mm.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            worst_mae = worst_mae.max(mae);
            all_am &= amg == amm;
            println!("[MLA t={t}] cos={cos:.10} max_abs_err={mae:.3e} argmax g={amg} m={amm} {}", if amg == amm { "OK" } else { "MISMATCH" });
            assert!(cos > 0.999_999_99, "t{t} cos {cos}");
        }
        assert!(all_am, "argmax mismatch");
        assert!(worst_mae < 1e-4, "worst max_abs_err {worst_mae} above mlx4 floor");
    }

    /// P3 real-weight loader gate — reproduce the mlx `sanitize` split of the REAL
    /// on-disk `kv_b_proj` (layer 3) into embed_q / unembed_out.
    #[test]
    fn kimi_mla_kvb_split_real_weights() {
        let dir = match std::env::var("KIMI_P3_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap())
                .unwrap();
        let split = match m.get("split") {
            Some(s) => s,
            None => {
                eprintln!("P3 split SKIP: dump run without ckpt dir");
                return;
            }
        };
        let g = |k: &str| split[k].as_u64().unwrap() as usize;
        let (nope, v, r, nh, head_dim) = (g("NOPE"), g("V"), g("R"), g("NH"), g("head_dim"));
        let f = |n: &str| read_f32(&format!("{dir}/{n}.f32"));
        let deq = f("real_kv_b_proj_deq"); // [nh*head_dim, r] = [nh, head_dim, r]
        let embed_q_ref = f("real_embed_q"); // [nh, r, nope]
        let unembed_ref = f("real_unembed_out"); // [nh, v, r]

        // our split: v3 = deq.reshape(nh, head_dim, r);
        //   embed_q[h,rr,n]   = v3[h, n, rr]           (v[:, :nope, :].swapaxes(-1,-2))
        //   unembed_out[h,d,rr] = v3[h, nope+d, rr]    (v[:, nope:, :])
        let mut eq = vec![0f32; nh * r * nope];
        let mut uo = vec![0f32; nh * v * r];
        for h in 0..nh {
            let vb = &deq[h * head_dim * r..(h + 1) * head_dim * r];
            for n in 0..nope {
                for rr in 0..r {
                    eq[(h * r + rr) * nope + n] = vb[n * r + rr];
                }
            }
            for d in 0..v {
                for rr in 0..r {
                    uo[(h * v + d) * r + rr] = vb[(nope + d) * r + rr];
                }
            }
        }
        let mae = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let e_mae = mae(&eq, &embed_q_ref);
        let u_mae = mae(&uo, &unembed_ref);
        println!("[MLA kv_b split real] embed_q max_abs_err={e_mae:.3e} unembed_out max_abs_err={u_mae:.3e}");
        assert!(e_mae == 0.0 && u_mae == 0.0, "kv_b_proj split not bit-identical to mlx sanitize");
    }
}

#[cfg(test)]
mod p2_tests {
    use super::kda::*;
    use std::collections::HashMap;

    struct Dump {
        dims: HashMap<String, f64>,
        t: HashMap<String, Vec<f32>>,
    }
    fn load_dump(dir: &str) -> Dump {
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap())
                .unwrap();
        let mut dims = HashMap::new();
        for (k, v) in m["dims"].as_object().unwrap() {
            dims.insert(k.clone(), v.as_f64().unwrap());
        }
        let mut t = HashMap::new();
        for name in m["shapes"].as_object().unwrap().keys() {
            let b = std::fs::read(format!("{dir}/{name}.f32")).unwrap();
            let v: Vec<f32> = b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            t.insert(name.clone(), v);
        }
        Dump { dims, t }
    }

    fn weights_from(d: &Dump) -> (KdaWeights, Vec<f32>, Vec<f32>, usize) {
        let g = |k: &str| d.dims[k] as usize;
        let (h, nh, hd, kern, l) = (g("H"), g("NH"), g("HD"), g("KERN"), g("L"));
        let w = KdaWeights {
            h, nh, hd, kern,
            eps: d.dims["eps"] as f32,
            q_proj: d.t["q_proj"].clone(),
            k_proj: d.t["k_proj"].clone(),
            v_proj: d.t["v_proj"].clone(),
            q_conv: d.t["q_conv"].clone(),
            k_conv: d.t["k_conv"].clone(),
            v_conv: d.t["v_conv"].clone(),
            f_a: d.t["f_a"].clone(),
            f_b: d.t["f_b"].clone(),
            b_proj: d.t["b_proj"].clone(),
            g_a: d.t["g_a"].clone(),
            g_b: d.t["g_b"].clone(),
            a_log: d.t["A_log"].clone(),
            dt_bias: d.t["dt_bias"].clone(),
            o_norm: d.t["o_norm"].clone(),
            o_proj: d.t["o_proj"].clone(),
        };
        (w, d.t["x"].clone(), d.t["golden"].clone(), l)
    }

    fn report(name: &str, golden: &[f32], mine: &[f32]) -> (f64, f64, bool) {
        let (mut dot, mut na, mut nb, mut mae) = (0f64, 0f64, 0f64, 0f64);
        for (a, b) in golden.iter().zip(mine.iter()) {
            let (a, b) = (*a as f64, *b as f64);
            dot += a * b;
            na += a * a;
            nb += b * b;
            mae = mae.max((a - b).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        let am_g = golden
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let am_m = mine
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        println!("[{name}] cos={cos:.10}  max_abs_err={mae:.3e}  argmax g={am_g} m={am_m} {}", if am_g == am_m { "OK" } else { "MISMATCH" });
        (cos, mae, am_g == am_m)
    }

    /// P2 gate — Rust KDA forward vs the mlx-lm KimiDeltaAttention oracle.
    #[test]
    fn kimi_kda_vs_oracle() {
        let dir = match std::env::var("KIMI_P2_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("P2 SKIP: set KIMI_P2_DIR (run kimi_p2_dump_kda.py first)");
                return;
            }
        };
        let d = load_dump(&dir);
        let (w, x, golden, l) = weights_from(&d);
        let mut state = vec![0f32; w.nh * w.hd * w.hd];
        let mine = forward(&w, &x, l, &mut state);
        let (cos, mae, am) = report("KDA vs oracle", &golden, &mine);
        assert!(cos > 0.999_999_99, "cos {cos}");
        assert!(am, "argmax mismatch");
        assert!(mae < 1e-4, "max_abs_err {mae} above mlx4 floor");
    }

    /// P2 structural gate — sequential-scan over L == L single-token decode steps
    /// (bit-identical output AND carried state). Guards against a future chunked/
    /// batched refactor that reorders the rank-1 accumulations.
    #[test]
    fn kimi_kda_scan_bit_exact_vs_serial() {
        let dir = match std::env::var("KIMI_P2_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("P2 SKIP: set KIMI_P2_DIR");
                return;
            }
        };
        let d = load_dump(&dir);
        let (w, x, _golden, l) = weights_from(&d);

        // (a) full-sequence scan
        let mut st_scan = vec![0f32; w.nh * w.hd * w.hd];
        let y_scan = forward(&w, &x, l, &mut st_scan);

        // (b) token-by-token: feed the growing prefix [0..=t] and take the last
        // row. Conv is causal with zero history, so the prefix forward reproduces
        // token t's output exactly — the sequential contract.
        let mut y_serial = vec![0f32; l * w.h];
        for t in 0..l {
            let mut st = vec![0f32; w.nh * w.hd * w.hd];
            let pref = &x[..(t + 1) * w.h];
            let yt = forward(&w, pref, t + 1, &mut st);
            y_serial[t * w.h..(t + 1) * w.h]
                .copy_from_slice(&yt[t * w.h..(t + 1) * w.h]);
            if t == l - 1 {
                // final-step state must equal the scan's carried state, bit-for-bit
                assert_eq!(st, st_scan, "carried state diverged at final step");
            }
        }
        let mut worst = 0f64;
        for (a, b) in y_scan.iter().zip(y_serial.iter()) {
            worst = worst.max((*a as f64 - *b as f64).abs());
        }
        println!("[KDA scan==serial] max_abs_err={worst:.3e}");
        assert_eq!(y_scan, y_serial, "scan vs serial not bit-identical");
    }
}

#[cfg(test)]
mod p1b_tests {
    //! P1b — FIRST REAL-WEIGHT gate. Retires residual #1: OUR mlx4 loader/dequant
    //! seam on REAL Kimi bytes (4-bit + 8-bit router + 3D switch_mlp expert slice
    //! + untied vocab) vs the mlx `mx.dequantize` golden.
    //!
    //! Gated on `KIMI_P1B_MANIFEST` (points at the golden manifest.json produced by
    //! `scripts/kimi_phase0/kimi_p1b_dump_golden.py`). Skips cleanly when unset so
    //! checkpoint-less CI stays green.
    use crate::model::dequantize_mlx_affine;
    use half::bf16;

    fn bf16_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()
    }
    fn u32_slice(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn kimi_p1b_real_weight_mlx4_dequant() {
        let manifest_path = match std::env::var("KIMI_P1B_MANIFEST") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("P1b SKIP: set KIMI_P1B_MANIFEST to run the real-weight gate");
                return;
            }
        };
        use memmap2::Mmap;
        use safetensors::SafeTensors;
        use std::fs::File;

        let mraw = std::fs::read_to_string(&manifest_path).expect("read manifest");
        let manifest: serde_json::Value = serde_json::from_str(&mraw).expect("parse manifest");
        let shard_path = manifest["shard"].as_str().expect("shard path");
        let f = File::open(shard_path).expect("open shard");
        let mmap = unsafe { Mmap::map(&f) }.expect("mmap shard");
        let st = SafeTensors::deserialize(&mmap).expect("parse safetensors");

        let mut worst = 0f64;
        let mut n_checked = 0;
        for t in manifest["tensors"].as_array().expect("tensors[]") {
            let name = t["name"].as_str().unwrap();
            let expert = t["expert"].as_u64().map(|x| x as usize);
            let row_start = t["row_start"].as_u64().unwrap() as usize;
            let row_count = t["row_count"].as_u64().unwrap() as usize;
            let in_features = t["in_features"].as_u64().unwrap() as usize;
            let group_size = t["group_size"].as_u64().unwrap() as usize;
            let bits = t["bits"].as_u64().unwrap() as usize;
            let golden_path = t["golden"].as_str().unwrap();

            let wv = st.tensor(&format!("{name}.weight")).expect("weight view");
            let sv = st.tensor(&format!("{name}.scales")).expect("scales view");
            let bv = st.tensor(&format!("{name}.biases")).expect("biases view");
            let wshape = wv.shape();
            let packed_cols = *wshape.last().unwrap();
            let out_full = wshape[wshape.len() - 2];
            let scales_cols = *sv.shape().last().unwrap();
            // OUR loader's group/bits derivation must reproduce the manifest.
            assert_eq!(scales_cols * group_size, in_features, "{name} in_features");
            assert_eq!(
                (packed_cols * 32) / in_features,
                bits,
                "{name} derived bits"
            );

            // base row index into the (flattened) [.., out, cols] tensor.
            let base_row = expert.map(|e| e * out_full).unwrap_or(0) + row_start;
            let w_all = u32_slice(wv.data());
            let s_all = bf16_slice_to_f32(sv.data());
            let b_all = bf16_slice_to_f32(bv.data());
            let w = &w_all[base_row * packed_cols..(base_row + row_count) * packed_cols];
            let s = &s_all[base_row * scales_cols..(base_row + row_count) * scales_cols];
            let b = &b_all[base_row * scales_cols..(base_row + row_count) * scales_cols];

            let deq = dequantize_mlx_affine(w, s, b, row_count, in_features, group_size, bits);

            let gbytes = std::fs::read(golden_path).expect("read golden");
            let golden: Vec<f32> = gbytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(deq.len(), golden.len(), "{name} numel");

            let mut mae = 0f64;
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (x, y) in deq.iter().zip(golden.iter()) {
                let (x, y) = (*x as f64, *y as f64);
                mae = mae.max((x - y).abs());
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            let cos = dot / (na.sqrt() * nb.sqrt());
            worst = worst.max(mae);
            n_checked += 1;
            println!(
                "[P1b {name}{}] out={row_count} in={in_features} bits={bits}  cos={cos:.10}  max_abs_err={mae:.3e}",
                expert.map(|e| format!(" e{e}")).unwrap_or_default()
            );
            // mlx dequant is exact affine arithmetic; our fp32 path reproduces it
            // to fp32 rounding. bf16-golden round-trip floor ~1e-2 in absolute
            // magnitude terms is NOT acceptable — require bit-tight agreement.
            assert!(
                cos > 0.999_999_99,
                "{name}: cos {cos} below 1.0 — dequant/layout mismatch"
            );
            assert!(
                mae < 1e-3,
                "{name}: max_abs_err {mae} above fp32-vs-mlx floor"
            );
        }
        println!("[P1b] {n_checked} real Kimi tensors dequant-bit-exact, worst max_abs_err={worst:.3e}");
        assert!(n_checked >= 5, "expected >=5 tensors in manifest");
    }
}

#[cfg(test)]
mod p1a_tests {
    use super::*;

    fn load_fixture_config() -> KimiConfig {
        let raw = std::fs::read_to_string("tests/fixtures/kimi/config.json")
            .expect("fixture config.json (run from crate root)");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("parse config.json");
        KimiConfig::from_json(&v).expect("KimiConfig::from_json")
    }

    /// P1a — the schedule off-by-one (1-indexed on disk → 0-indexed slots) is the
    /// #1 loader trap. Pin the exact MLA / KDA 0-idx sets + dense layer.
    #[test]
    fn kimi_schedule_offbyone() {
        let c = load_fixture_config();
        assert_eq!(c.num_hidden_layers, 27);
        assert_eq!(c.layer_schedule.len(), 27, "schedule length == num layers");

        let mla: Vec<usize> = c
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == KimiLayerKind::Mla)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            mla,
            vec![3, 7, 11, 15, 19, 23, 26],
            "MLA 0-idx set (full_attn 1-idx [4,8,12,16,20,24,27] minus 1)"
        );

        // KDA = everything else (20 layers), incl layer 0.
        let kda: Vec<usize> = c
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == KimiLayerKind::Kda)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(kda.len(), 20);
        assert_eq!(c.layer_schedule[0], KimiLayerKind::Kda, "layer 0 is KDA");

        // Layer 0 (only) is dense; every other layer is MoE.
        assert!(c.is_dense_mlp(0));
        assert!(!c.is_dense_mlp(1));
        assert_eq!(c.first_k_dense_replace, 1);
    }

    /// P1a — structural dims match the pinned config (build-to-these table).
    #[test]
    fn kimi_config_dims() {
        let c = load_fixture_config();
        assert_eq!(c.hidden_size, 2304);
        assert_eq!(c.vocab_size, 163840);
        assert_eq!(c.rms_norm_eps, 1e-5);
        assert!(!c.tie_word_embeddings, "untied lm_head");
        assert_eq!(c.intermediate_size, 9216, "dense MLP inter (layer 0)");

        // MLA
        assert_eq!(c.num_attention_heads, 32);
        assert_eq!(c.kv_lora_rank, 512);
        assert_eq!(c.q_lora_rank, None, "Q uncompressed (q_lora_rank null)");
        assert_eq!(c.qk_nope_head_dim, 128);
        assert_eq!(c.qk_rope_head_dim, 64);
        assert_eq!(c.v_head_dim, 128);
        assert_eq!(c.q_head_dim(), 192, "qk_dim = nope+pe = 192");
        assert!(c.mla_use_nope);

        // KDA
        assert_eq!(c.kda_num_heads, 32);
        assert_eq!(c.kda_head_dim, 128);
        assert_eq!(c.kda_conv_kernel, 4);
        assert_eq!(c.kda_conv_dim(), 3 * 32 * 128, "3 depthwise convs concat");

        // MoE
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_token, 8);
        assert_eq!(c.num_shared_experts, 1);
        assert_eq!(c.moe_intermediate_size, 1024);
        assert_eq!(c.routed_scaling_factor, 2.446_f32);
        assert_eq!(c.router_activation, RouterActivation::Sigmoid);
        assert!(c.moe_renormalize);
        assert_eq!(c.num_expert_group, 1, "grouped-topk degenerates to top-8");
        assert_eq!(c.topk_group, 1);
    }

    /// Guard the validator: a duplicated / gapped schedule must be rejected.
    #[test]
    fn kimi_schedule_rejects_bad_lists() {
        // overlap: layer 4 in both lists
        assert!(KimiConfig::build_schedule(4, &[4], &[1, 2, 3, 4]).is_err());
        // gap: layer 3 (0-idx 2) unassigned
        assert!(KimiConfig::build_schedule(4, &[4], &[1, 2]).is_err());
        // 0 is not a valid 1-indexed entry
        assert!(KimiConfig::build_schedule(4, &[], &[0, 1, 2, 3, 4]).is_err());
        // complete + disjoint passes
        assert!(KimiConfig::build_schedule(4, &[4], &[1, 2, 3]).is_ok());
    }
}

#[cfg(test)]
mod p4half_tests {
    //! P4½ — the ASSEMBLY gate. Proves `KimiModel` (loaded through the real
    //! `[start,end)` window loader on shard 1) dispatches every block type
    //! correctly: for each representative layer the assembled per-layer forward
    //! (`layer_forward`: input_ln → attn-dispatch → residual → post_ln →
    //! mlp-dispatch → residual) reproduces the mlx-lm `KimiDecoderLayer` oracle at
    //! FULL dims on REAL weights. Plus the P5-offline-prep gates: the
    //! footprint-minimax PP-3/PP-4 split (config-only) and the window-loader shard
    //! selection (fixture index.json, no weights).
    use super::*;

    fn read_f32(p: &str) -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn fixture_config() -> KimiConfig {
        let raw = std::fs::read_to_string("tests/fixtures/kimi/config.json")
            .expect("fixture config.json");
        KimiConfig::from_json(&serde_json::from_str(&raw).unwrap()).unwrap()
    }

    /// P4½ ASSEMBLY GATE (real weights). Gated on `KIMI_CKPT` (checkpoint dir with
    /// shard 1) + `KIMI_P4HALF_DIR` (oracle dump from `kimi_p4half_dump_layers.py`).
    #[test]
    fn kimi_assembly_dispatch_vs_oracle() {
        let (ckpt, dump) = match (std::env::var("KIMI_CKPT"), std::env::var("KIMI_P4HALF_DIR")) {
            (Ok(a), Ok(b)) => (a, b),
            _ => {
                eprintln!("P4½ SKIP: set KIMI_CKPT + KIMI_P4HALF_DIR");
                return;
            }
        };
        let cfg = fixture_config();
        // Window [0,5): all fully in shard 1 (KDA-dense L0, MLA L3, KDA-MoE L1..).
        let model = KimiModel::load_cpu(&ckpt, &cfg, 0, 5, false).expect("load_cpu");
        assert_eq!(model.layers.len(), 5);

        let man: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dump}/manifest.json")).unwrap())
                .unwrap();
        let l = man["L"].as_u64().unwrap() as usize;
        let h = man["H"].as_u64().unwrap() as usize;
        assert_eq!(h, cfg.hidden_size);

        let mut worst_mae = 0f64;
        let mut all_am = true;
        for &lidx in &[0usize, 3, 1] {
            let x = read_f32(&format!("{dump}/x_{lidx}.f32"));
            let golden = read_f32(&format!("{dump}/golden_{lidx}.f32"));
            let layer = model.layers.iter().find(|ly| ly.idx == lidx).unwrap();
            let mine = model.layer_forward(layer, &x, l);
            assert_eq!(mine.len(), golden.len());

            // per-token cos / argmax / max_abs_err
            let (mut worst_cos, mut layer_am) = (1f64, true);
            for t in 0..l {
                let (g, m) = (&golden[t * h..(t + 1) * h], &mine[t * h..(t + 1) * h]);
                let (mut dot, mut na, mut nb, mut mae) = (0f64, 0f64, 0f64, 0f64);
                for (a, b) in g.iter().zip(m.iter()) {
                    let (a, b) = (*a as f64, *b as f64);
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                    mae = mae.max((a - b).abs());
                }
                let cos = dot / (na.sqrt() * nb.sqrt());
                let amg = g.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                let amm = m.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                worst_cos = worst_cos.min(cos);
                worst_mae = worst_mae.max(mae);
                layer_am &= amg == amm;
            }
            let (kind, mlp) = (
                man["layers"][lidx.to_string()]["kind"].as_str().unwrap(),
                man["layers"][lidx.to_string()]["mlp"].as_str().unwrap(),
            );
            println!(
                "[P4½ L{lidx} {kind}+{mlp}] worst_cos={worst_cos:.10} max_abs_err={worst_mae:.3e} argmax {}",
                if layer_am { "EXACT" } else { "MISMATCH" }
            );
            all_am &= layer_am;
            assert!(worst_cos > 0.999_999_9, "L{lidx} cos {worst_cos} — dispatch wired wrong");
        }
        println!("[P4½ ASSEMBLY] all block types dispatched; worst max_abs_err={worst_mae:.3e}");
        assert!(all_am, "assembly argmax mismatch — a block type is mis-wired");
        assert!(worst_mae < 1e-2, "worst max_abs_err {worst_mae} above mlx4/fp32 floor");
    }

    /// P5 offline prep — footprint-minimax PP-3/PP-4 BOUNDS (config-only, no
    /// weights). Prints the BOUNDS + per-stage GB and pins structural invariants.
    #[test]
    fn kimi_pp_split_bounds() {
        let cfg = fixture_config();
        for n_stages in [3usize, 4] {
            let (bounds, loads) = cfg.pp_split(n_stages);
            let per: Vec<usize> = (0..n_stages).map(|i| bounds[i + 1] - bounds[i]).collect();
            let maxgb = loads.iter().cloned().fold(0f64, f64::max);
            println!(
                "[PP-{n_stages}] BOUNDS={bounds:?}  per-stage-layers={per:?}  \
                 per-stage-GB={:?}  max={maxgb:.2}GB",
                loads.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>()
            );
            // structural invariants
            assert_eq!(bounds.len(), n_stages + 1);
            assert_eq!(bounds[0], 0);
            assert_eq!(*bounds.last().unwrap(), cfg.num_hidden_layers);
            for w in bounds.windows(2) {
                assert!(w[1] > w[0], "each stage owns >=1 layer");
            }
            // fits the ~13.3GB GTT budget (plan §5: PP-3 ~9GB, PP-4 ~6.75GB).
            assert!(maxgb <= 13.3, "PP-{n_stages} max stage {maxgb:.2}GB over 13.3GB budget");
        }
    }

    /// P5 offline prep — the `[start,end)` window loader's SHARD SELECTION (fixture
    /// index.json; no weights loaded). Proves the window opens ONLY the shards that
    /// host its layers — the structural change P5 sharding needs.
    #[test]
    fn kimi_window_shard_selection() {
        let cfg = fixture_config();
        let idx: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("tests/fixtures/kimi/model.safetensors.index.json").unwrap(),
        )
        .unwrap();
        let wm = idx["weight_map"].as_object().unwrap();
        let shards_for = |start: usize, end: usize| -> std::collections::BTreeSet<String> {
            let mut s = std::collections::BTreeSet::new();
            for l in start..end {
                for name in super::layer_tensor_names(&cfg, l) {
                    if let Some(sh) = wm.get(&name).and_then(|v| v.as_str()) {
                        s.insert(sh.to_string());
                    }
                }
            }
            s
        };
        // Window [0,5) — the assembly-gate window — must be shard 1 ONLY.
        let s05 = shards_for(0, 5);
        println!("[window [0,5)] shards={s05:?}");
        assert_eq!(
            s05,
            ["model-00001-of-00006.safetensors".to_string()].into_iter().collect(),
            "window [0,5) must load from shard 1 only"
        );
        // Tail window [26,27) must NOT pull shard 1 (proves selectivity).
        let s_tail = shards_for(26, 27);
        println!("[window [26,27)] shards={s_tail:?}");
        assert!(
            !s_tail.contains("model-00001-of-00006.safetensors"),
            "tail window must not open shard 1"
        );
        assert_eq!(s_tail.len(), 1, "tail layer resides in a single shard");
    }
}

#[cfg(test)]
mod p5_tests {
    //! P5 Phase-A — the FULL-27-LAYER CPU gate (real weights). Runs the whole
    //! heterogeneous model (embed -> 27 layers -> final RMSNorm -> untied lm_head)
    //! over the mlx-lm greedy token stream and asserts **argmax-exact** per position
    //! (the depth-chain validation the shard-1 assembly gate could not do), plus a
    //! `max_abs_err` spot-check on the pre-lm_head normed hidden vs the oracle.
    //!
    //! RAM: dequantizing all 26 MoE layers to f32 is ~188GB, so the sweep loads the
    //! model in fine-grained LAYER WINDOWS (`KIMI_P5_STRIDE`, default 3) via the real
    //! `[start,end)` window loader, threading `[seq,H]` hidden states between windows
    //! and dropping each window before loading the next. This is mathematically
    //! identical to a single `[0,27)` forward (each layer scans the full sequence from
    //! zero state; windowing is across LAYERS, not tokens) and also exercises the
    //! P5 window loader across shard boundaries end-to-end.
    use super::*;

    fn read_f32(p: &str) -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn read_i32(p: &str) -> Vec<i32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn kimi_p5_full_model_gate() {
        let (ckpt, dump) = match (std::env::var("KIMI_CKPT"), std::env::var("KIMI_P5_DIR")) {
            (Ok(a), Ok(b)) => (a, b),
            _ => {
                eprintln!("P5 SKIP: set KIMI_CKPT (full checkpoint) + KIMI_P5_DIR (oracle dump)");
                return;
            }
        };
        let stride: usize = std::env::var("KIMI_P5_STRIDE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);

        // --- config from the real checkpoint ---
        let cfg_raw = std::fs::read_to_string(format!("{ckpt}/config.json")).expect("config.json");
        let cfg = KimiConfig::from_json(&serde_json::from_str(&cfg_raw).unwrap()).unwrap();
        let (h, n_layers) = (cfg.hidden_size, cfg.num_hidden_layers);
        let eps = cfg.rms_norm_eps;

        // --- oracle dump ---
        let toks: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dump}/tokens.json")).unwrap())
                .unwrap();
        let full_ids: Vec<usize> = toks["full_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let prompt_len = toks["prompt_len"].as_u64().unwrap() as usize;
        let seq = full_ids.len();
        assert_eq!(toks["H"].as_u64().unwrap() as usize, h, "H mismatch config vs oracle");
        let oracle_argmax = read_i32(&format!("{dump}/oracle_argmax.i32"));
        let oracle_normed = read_f32(&format!("{dump}/normed_hidden.f32"));
        assert_eq!(oracle_argmax.len(), seq);
        assert_eq!(oracle_normed.len(), seq * h);

        // --- windowed sequential CPU sweep over [0, n_layers) ---
        let mut bounds = vec![0usize];
        while *bounds.last().unwrap() < n_layers {
            bounds.push((bounds.last().unwrap() + stride).min(n_layers));
        }
        eprintln!(
            "[P5] full-model sweep: {} layers, stride {}, windows {:?}, seq {}",
            n_layers, stride, bounds, seq
        );

        let mut hidden: Vec<f32> = Vec::new(); // [seq*H], filled by the stage-0 embed
        let mut my_argmax = vec![0i32; seq];
        for w in bounds.windows(2) {
            let (s, e) = (w[0], w[1]);
            let t_load = std::time::Instant::now();
            let model = KimiModel::load_cpu(&ckpt, &cfg, s, e, true).expect("load_cpu window");
            eprintln!("[P5]  window [{s},{e}) loaded in {:?}", t_load.elapsed());

            if s == 0 {
                // embedding lookup: hidden[t] = embed[full_ids[t]]
                let embed = model.embed.as_ref().expect("embed at stage 0");
                hidden = vec![0f32; seq * h];
                for t in 0..seq {
                    let row = full_ids[t] * h;
                    hidden[t * h..(t + 1) * h].copy_from_slice(&embed[row..row + h]);
                }
                // debug hook: inject an external input hidden (e.g., oracle h_embed)
                // to remove the embed-diff confound when bisecting.
                if let Ok(inp) = std::env::var("KIMI_P5_INPUT") {
                    hidden = read_f32(&inp);
                    assert_eq!(hidden.len(), seq * h, "injected input wrong size");
                    eprintln!("[P5] injected external input from {inp}");
                }
                if let Ok(dir) = std::env::var("KIMI_P5_LAYERDUMP") {
                    let bytes: Vec<u8> = hidden.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(format!("{dir}/rust_embed.f32"), &bytes).unwrap();
                    eprintln!("[P5] dumped rust_embed.f32");
                }
            }

            let t_fwd = std::time::Instant::now();
            // Per-layer forward (so an optional per-layer dump can bisect divergence).
            let layerdump = std::env::var("KIMI_P5_LAYERDUMP").ok();
            for layer in &model.layers {
                hidden = model.layer_forward(layer, &hidden, seq);
                if let Some(dir) = &layerdump {
                    let bytes: Vec<u8> =
                        hidden.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(format!("{dir}/rust_after_{}.f32", layer.idx), &bytes).unwrap();
                }
            }
            eprintln!("[P5]  window [{s},{e}) forward in {:?}", t_fwd.elapsed());

            if e == n_layers {
                // final RMSNorm + untied lm_head -> per-position argmax
                let fnorm = model.final_norm.as_ref().expect("final_norm at tail");
                let lm = model.lm_head.as_ref().expect("lm_head at tail");
                let vocab = lm.len() / h;
                let normed = rmsnorm(&hidden, seq, h, fnorm, eps);

                // depth-chain spot-check: max_abs_err + worst cos on the normed hidden
                let (mut worst_mae, mut worst_cos) = (0f64, 1f64);
                for t in 0..seq {
                    let (a, b) = (
                        &oracle_normed[t * h..(t + 1) * h],
                        &normed[t * h..(t + 1) * h],
                    );
                    let (mut dot, mut na, mut nb, mut mae) = (0f64, 0f64, 0f64, 0f64);
                    for (x, y) in a.iter().zip(b.iter()) {
                        let (x, y) = (*x as f64, *y as f64);
                        dot += x * y;
                        na += x * x;
                        nb += y * y;
                        mae = mae.max((x - y).abs());
                    }
                    worst_mae = worst_mae.max(mae);
                    worst_cos = worst_cos.min(dot / (na.sqrt() * nb.sqrt()));
                }
                eprintln!(
                    "[P5] normed-hidden spot-check: worst_cos={worst_cos:.10} max_abs_err={worst_mae:.3e}"
                );
                // Diagnostic threshold: vs a dense-f32 oracle the 27-layer mlx4-dequant
                // chain stays cos>0.999 (mlx4 floor accumulated through the recurrence);
                // the MANDATORY gate is argmax-exact below. (A bf16-native or
                // quantized-matmul oracle instead lands ~0.96/0.994 — a precision-mode
                // artifact, not a math error; use the dense-f32 oracle for this gate.)
                assert!(
                    worst_cos > 0.999,
                    "normed hidden cos {worst_cos} below the mlx4-f32 floor — likely a real \
                     divergence OR the oracle was not dumped in dense f32"
                );

                // per-position argmax over the vocab (stream; only argmax kept)
                for t in 0..seq {
                    let xt = &normed[t * h..(t + 1) * h];
                    let (mut best_i, mut best_v) = (0usize, f32::NEG_INFINITY);
                    for o in 0..vocab {
                        let wr = &lm[o * h..(o + 1) * h];
                        let mut acc = 0f32;
                        for i in 0..h {
                            acc += wr[i] * xt[i];
                        }
                        if acc > best_v {
                            best_v = acc;
                            best_i = o;
                        }
                    }
                    my_argmax[t] = best_i as i32;
                }
            }
            drop(model); // free the window before loading the next
        }

        // --- gate: argmax-exact vs the oracle's FRESH full-sequence prefill ---
        // Apples-to-apples: both Rust and the oracle run the identical fresh
        // full-sequence forward over `full_ids`, so the depth-chain gate is
        // `my_argmax[t] == oracle_argmax[t]` at EVERY position. (The `full_ids[t+1]`
        // greedy stream came from the oracle's CACHED decode; oracle fresh-prefill
        // vs oracle cache flip argmax at a few near-tied positions at the ~1e-6
        // decode floor — plan §4 — so `full_ids[t+1]` is reported informationally,
        // NOT gated.)
        let mut all_match = true;
        let mut first_mismatch: Option<usize> = None;
        for t in (prompt_len - 1)..(seq - 1) {
            let (mine, orc, nxt) = (my_argmax[t], oracle_argmax[t], full_ids[t + 1] as i32);
            let step = t - (prompt_len - 1);
            let agree = mine == orc;
            eprintln!(
                "[P5] gen[{step:>2}] pos {t}: mine={mine} oracle_prefill={orc} {} | gen_stream_next={nxt}{}",
                if agree { "MATCH" } else { "MISMATCH" },
                if orc != nxt { " (oracle cache!=prefill here)" } else { "" }
            );
            if !agree {
                all_match = false;
                first_mismatch.get_or_insert(t);
            }
        }
        // full-sequence (incl. prompt region) argmax agreement — the strongest form.
        let full_match = (0..seq).filter(|&t| my_argmax[t] == oracle_argmax[t]).count();
        eprintln!(
            "[P5] full-sequence Rust-vs-oracle argmax agreement: {}/{} positions",
            full_match, seq
        );
        assert_eq!(
            full_match, seq,
            "P5 argmax-exact FAILED vs oracle fresh-prefill: first gen-region mismatch at pos {:?}",
            first_mismatch
        );
        assert!(all_match, "gen-region argmax-exact failed");
        eprintln!(
            "[P5 PASS] full-27-layer CPU gate: Rust argmax-exact vs oracle over all {} positions \
             ({} generated tokens teacher-forced)",
            seq,
            seq - prompt_len
        );
    }
}

#[cfg(test)]
mod p_decode_tests {
    //! Resident-DECODE gate (the substantial new machinery). Chaining the
    //! single-token `decode_step` over a token stream must reproduce the stateless
    //! fresh-prefill `forward` **BIT-IDENTICALLY** — the strongest possible form.
    //! This exercises: the KDA conv sliding-window + recurrence carry, the MLA KV
    //! cache, and the per-token MoE/dense MLP, threaded across all layers in a PP
    //! window. Gated on `KIMI_CKPT` (shard-1 window `[0,5)` hosts all four block
    //! types: KDA+dense L0, KDA+MoE L1/L2/L4, MLA+MoE L3) — no 27GB download, no GPU.
    //! The resident decode state advancing IN PLACE must equal a full re-prefill.
    use super::*;

    fn fixture_config() -> KimiConfig {
        let raw = std::fs::read_to_string("tests/fixtures/kimi/config.json")
            .expect("fixture config.json (run from crate root)");
        KimiConfig::from_json(&serde_json::from_str(&raw).unwrap()).unwrap()
    }

    #[test]
    fn kimi_decode_eq_prefill_bit_exact() {
        let ckpt = match std::env::var("KIMI_CKPT") {
            Ok(c) => c,
            Err(_) => {
                eprintln!("decode SKIP: set KIMI_CKPT (checkpoint dir w/ shard 1 for window [0,5))");
                return;
            }
        };
        let cfg = fixture_config();
        let h = cfg.hidden_size;
        let l = 6usize; // token-stream length (each step is one decode token)
        let model = KimiModel::load_cpu(&ckpt, &cfg, 0, 5, false).expect("load_cpu [0,5)");
        assert_eq!(model.layers.len(), 5);

        // Deterministic seeded input stream xs[l, h] (splitmix64-ish LCG, ~[-0.5,0.5]).
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut xs = vec![0f32; l * h];
        for v in xs.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5); // ~[-0.5,0.5]
        }

        // (a) fresh full-sequence prefill through the window.
        let prefill = model.forward(xs.clone(), l);

        // (b) resident decode: one token at a time, state advancing in place.
        let mut states = model.init_decode_state();
        let mut decode = vec![0f32; l * h];
        for t in 0..l {
            let out = model.decode_step(&mut states, xs[t * h..(t + 1) * h].to_vec());
            decode[t * h..(t + 1) * h].copy_from_slice(&out);
        }

        // per-token report + argmax + bit-identical assert.
        let mut worst_mae = 0f64;
        let mut all_am = true;
        for t in 0..l {
            let (a, b) = (&prefill[t * h..(t + 1) * h], &decode[t * h..(t + 1) * h]);
            let mut mae = 0f64;
            for (x, y) in a.iter().zip(b.iter()) {
                mae = mae.max((*x as f64 - *y as f64).abs());
            }
            let amg = a.iter().enumerate().max_by(|p, q| p.1.partial_cmp(q.1).unwrap()).unwrap().0;
            let amd = b.iter().enumerate().max_by(|p, q| p.1.partial_cmp(q.1).unwrap()).unwrap().0;
            worst_mae = worst_mae.max(mae);
            all_am &= amg == amd;
            println!(
                "[decode t={t}] max_abs_err={mae:.3e} argmax prefill={amg} decode={amd} {}",
                if amg == amd { "OK" } else { "MISMATCH" }
            );
        }
        println!("[decode==prefill] window [0,5) worst max_abs_err={worst_mae:.3e}");
        assert!(all_am, "resident-decode argmax != fresh-prefill");
        assert!(prefill == decode, "resident-decode not BIT-IDENTICAL to fresh-prefill");
    }

    /// PP-decomposition gate: splitting the decode into two PP stages `[0,2)` +
    /// `[2,5)`, each holding its OWN resident state, with the `[H]` hidden crossing
    /// the stage boundary, must reproduce the single-window `[0,5)` decode
    /// BIT-IDENTICALLY. This validates `forward_pp_stage`'s inter-stage handoff +
    /// per-stage state carry (the cluster PP machinery) on CPU, single-node. Also
    /// smoke-checks `forward_pp_stage`'s embed (first-stage) + non-last return path.
    #[test]
    fn kimi_pp_stage_decompose_bit_exact() {
        let ckpt = match std::env::var("KIMI_CKPT") {
            Ok(c) => c,
            Err(_) => {
                eprintln!("PP-decompose SKIP: set KIMI_CKPT (shard-1 windows [0,2)+[2,5))");
                return;
            }
        };
        let cfg = fixture_config();
        let h = cfg.hidden_size;
        let l = 6usize;

        let full = KimiModel::load_cpu(&ckpt, &cfg, 0, 5, false).expect("[0,5)");
        let m_a = KimiModel::load_cpu(&ckpt, &cfg, 0, 2, false).expect("[0,2)");
        let m_b = KimiModel::load_cpu(&ckpt, &cfg, 2, 5, false).expect("[2,5)");

        let mut s: u64 = 0xD1B5_4A32_D192_ED03;
        let mut xs = vec![0f32; l * h];
        for v in xs.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        }

        let mut sf = full.init_decode_state();
        let mut sa = m_a.init_decode_state();
        let mut sb = m_b.init_decode_state();
        let mut worst_mae = 0f64;
        for t in 0..l {
            let xt = xs[t * h..(t + 1) * h].to_vec();
            let full_out = full.decode_step(&mut sf, xt.clone());
            // PP: stage A decode -> hidden crosses hop -> stage B decode.
            let mid = m_a.decode_step(&mut sa, xt);
            assert_eq!(mid.len(), h, "inter-stage hidden must be [H]");
            let split_out = m_b.decode_step(&mut sb, mid);
            let mae = full_out
                .iter()
                .zip(&split_out)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0f64, f64::max);
            worst_mae = worst_mae.max(mae);
            assert!(full_out == split_out, "PP-split decode != single-window at t={t}");
        }
        println!("[PP-decompose] [0,2)+[2,5) == [0,5) decode, worst max_abs_err={worst_mae:.3e}");

        // smoke: forward_pp_stage embed (first) path == decode_step over embed row.
        let mut m_edge = KimiModel::load_cpu(&ckpt, &cfg, 0, 2, true).expect("[0,2) edges");
        let tok = 12345u32;
        let stage_out = m_edge.forward_pp_stage(tok, &[], 0);
        assert_eq!(stage_out.len(), h, "non-last stage returns [H] hidden");
        let embed = m_edge.embed.as_ref().unwrap();
        let emb_row = embed[tok as usize * h..(tok as usize + 1) * h].to_vec();
        let mut s2 = m_edge.init_decode_state();
        let ref_out = m_edge.decode_step(&mut s2, emb_row);
        assert!(stage_out == ref_out, "forward_pp_stage embed path != decode_step(embed row)");
        println!("[PP-stage smoke] forward_pp_stage embed+decode OK");
    }
}
