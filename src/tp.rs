// SPDX-License-Identifier: Apache-2.0
//! Qwen3.5/3.6 (qwen35) tensor-parallel forward methods + weight-shard
//! helpers. Extracted verbatim from lib.rs (M1).

use crate::model;
#[cfg(feature = "qwen35")]
use crate::qwen35;
use crate::{VulkanModel, par_deltanet, LmShard};

#[cfg(feature = "qwen35")]
impl VulkanModel {
    /// TP GatedDeltaNet: this rank's 1/n head shard. Identical math to
    /// `qwen35_delta_net_gpu` but with per-rank head counts (nk/n key heads,
    /// nv/n value heads, conv_dim/n channels) on already-sharded weights/state.
    /// Returns the PARTIAL out_proj (row-sharded over this rank's value_dim slice)
    /// — the caller all-reduces it to the full hidden contribution.
    pub(crate) fn qwen35_delta_net_tp(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, x: &[f32], n: usize) -> Vec<f32> {
        let nk = cfg.linear_num_key_heads / n;       // this rank's key heads (4)
        let nv = cfg.linear_num_value_heads / n;     // this rank's value heads (12)
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim = nk * kd;                        // rank-local key_dim
        let value_dim = nv * vd;                      // rank-local value_dim
        let conv_dim = key_dim * 2 + value_dim;       // rank-local conv_dim (= full/n)
        let h = cfg.hidden_size;
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");

        // Projections (GPU, sharded weights → per-rank output dims). qkv/z/a/b
        // share input x → ONE submit.
        let (qkv_n, z_n, a_n, b_n) = (
            ln("in_proj_qkv.weight"), ln("in_proj_z.weight"),
            ln("in_proj_a.weight"), ln("in_proj_b.weight"),
        );
        let mut proj = self.qwen35_matvec_multi(x, h, &[
            (&qkv_n, conv_dim), (&z_n, value_dim), (&a_n, nv), (&b_n, nv),
        ]);
        let b = proj.pop().unwrap();
        let a = proj.pop().unwrap();
        let z = proj.pop().unwrap();
        let qkv = proj.pop().unwrap();

        // Recurrent core (conv1d+SiLU + delta rule + gated norm) → this rank's
        // gated value_dim slice, then out_proj row-shard → PARTIAL.
        let gated = self.qwen35_delta_net_tp_core(cfg, layer_idx, &qkv, &z, &a, &b, n);
        // out_proj row-sharded over this rank's value_dim → PARTIAL (caller reduces).
        self.qwen35_matvec(&ln("out_proj.weight"), &gated, value_dim, h)
    }

    /// Recurrent CORE of `qwen35_delta_net_tp`, factored out so the batched
    /// verify (`qwen35_delta_net_tp_batched`) can run the SAME per-token conv1d +
    /// delta-rule + gated-norm on projections it computed with the single-stream
    /// cols kernel — bit-identical math to the per-token decode mixer, only the
    /// projection DISPATCH differs. Consumes this rank's `qkv`(conv_dim),
    /// `z`(value_dim), `a`(nv), `b`(nv) projection outputs and advances the
    /// resident recurrent state (`conv_state`/`state`) for ONE token. Returns the
    /// gated value_dim slice (pre-out_proj).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen35_delta_net_tp_core(
        &mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize,
        qkv: &[f32], z: &[f32], a: &[f32], b: &[f32], n: usize,
    ) -> Vec<f32> {
        let eps = cfg.rms_norm_eps;
        let nk = cfg.linear_num_key_heads / n;       // this rank's key heads
        let nv = cfg.linear_num_value_heads / n;     // this rank's value heads
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim = nk * kd;                        // rank-local key_dim
        let value_dim = nv * vd;                      // rank-local value_dim
        let conv_dim = key_dim * 2 + value_dim;       // rank-local conv_dim (= full/n)
        let kern = cfg.linear_conv_kernel_dim;
        let ratio = nv / nk;                          // GVA ratio (unchanged)
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");
        // Recurrence host weights (already sharded by head in the loader).
        let conv_w = self.qwen35_w(&ln("conv1d.weight")); // [conv_dim/n, kern]
        let a_log = self.qwen35_w(&ln("A_log"));          // [nv]
        let dt_bias = self.qwen35_w(&ln("dt_bias"));      // [nv]
        let norm_w = self.qwen35_w(&ln("norm.weight"));   // [vd] (replicated)

        // Causal depthwise conv1d + SiLU (per-channel, parallel over cores).
        let win = kern - 1;
        let mut conv_out = vec![0.0f32; conv_dim];
        {
            use rayon::prelude::*;
            let qm = self.qwen35.as_mut().unwrap();
            let si = qm.state_idx(layer_idx);
            let st = match &mut qm.layer_state[si] {
                qwen35::LayerState::Linear(d) => d,
                _ => unreachable!("linear_attention layer has a DeltaNet state"),
            };
            let chan = |c: usize, cs: &mut [f32], out_c: &mut f32| {
                let mut acc = 0.0f32;
                for t in 0..win { acc += cs[t] * conv_w[c * kern + t]; }
                acc += qkv[c] * conv_w[c * kern + win];
                *out_c = acc / (1.0 + (-acc).exp());
                if win > 0 {
                    for t in 0..win - 1 { cs[t] = cs[t + 1]; }
                    cs[win - 1] = qkv[c];
                }
            };
            if win == 0 {
                conv_out.iter_mut().enumerate().for_each(|(c, o)| {
                    *o = { let a = qkv[c] * conv_w[c * kern]; a / (1.0 + (-a).exp()) };
                });
            } else if par_deltanet() {
                st.conv_state.par_chunks_mut(win).zip(conv_out.par_iter_mut())
                    .enumerate().for_each(|(c, (cs, o))| chan(c, cs, o));
            } else {
                st.conv_state.chunks_mut(win).zip(conv_out.iter_mut())
                    .enumerate().for_each(|(c, (cs, o))| chan(c, cs, o));
            }
        }

        // Split + per-head RMSNorm(no weight), inv_scale / inv_scale^2.
        let inv = 1.0 / (kd as f32).sqrt();
        let q_flat = &conv_out[..key_dim];
        let k_flat = &conv_out[key_dim..2 * key_dim];
        let v_flat = &conv_out[2 * key_dim..];
        let mut q = vec![0.0f32; key_dim];
        let mut k = vec![0.0f32; key_dim];
        for hi in 0..nk {
            let qn = model::cpu_rms_norm_no_weight(&q_flat[hi * kd..(hi + 1) * kd], kd, 1e-6);
            let kn = model::cpu_rms_norm_no_weight(&k_flat[hi * kd..(hi + 1) * kd], kd, 1e-6);
            for j in 0..kd {
                q[hi * kd + j] = qn[j] * inv * inv;
                k[hi * kd + j] = kn[j] * inv;
            }
        }

        // Recurrent delta rule per (local) v-head + gated norm.
        let mut gated = vec![0.0f32; value_dim];
        {
            use rayon::prelude::*;
            let qm = self.qwen35.as_mut().unwrap();
            let si = qm.state_idx(layer_idx);
            let st = match &mut qm.layer_state[si] {
                qwen35::LayerState::Linear(d) => d,
                _ => unreachable!(),
            };
            let head = |j: usize, state_j: &mut [f32], gated_j: &mut [f32]| {
                let kh = j / ratio;
                let q_j = &q[kh * kd..(kh + 1) * kd];
                let k_j = &k[kh * kd..(kh + 1) * kd];
                let v_j = &v_flat[j * vd..(j + 1) * vd];
                let g = -(a_log[j].exp()) * {
                    let xx = a[j] + dt_bias[j];
                    if xx > 20.0 { xx } else { (1.0 + xx.exp()).ln() }
                };
                let decay = g.exp();
                let beta = 1.0 / (1.0 + (-b[j]).exp());
                for e in 0..kd * vd { state_j[e] *= decay; }
                let mut kv_mem = vec![0.0f32; vd];
                for kk in 0..kd {
                    let kv = k_j[kk];
                    for vv in 0..vd { kv_mem[vv] += state_j[kk * vd + vv] * kv; }
                }
                let mut delta = vec![0.0f32; vd];
                for vv in 0..vd { delta[vv] = (v_j[vv] - kv_mem[vv]) * beta; }
                for kk in 0..kd {
                    let kv = k_j[kk];
                    for vv in 0..vd { state_j[kk * vd + vv] += kv * delta[vv]; }
                }
                let mut out_j = vec![0.0f32; vd];
                for kk in 0..kd {
                    let qv = q_j[kk];
                    for vv in 0..vd { out_j[vv] += state_j[kk * vd + vv] * qv; }
                }
                let normed = model::cpu_rms_norm(&out_j, &norm_w, eps);
                for vv in 0..vd {
                    let zz = z[j * vd + vv];
                    gated_j[vv] = normed[vv] * (zz / (1.0 + (-zz).exp()));
                }
            };
            if par_deltanet() {
                st.state.par_chunks_mut(kd * vd).zip(gated.par_chunks_mut(vd))
                    .enumerate().for_each(|(j, (state_j, gated_j))| head(j, state_j, gated_j));
            } else {
                st.state.chunks_mut(kd * vd).zip(gated.chunks_mut(vd))
                    .enumerate().for_each(|(j, (state_j, gated_j))| head(j, state_j, gated_j));
            }
        }

        gated
    }

    /// TP GatedAttention: this rank's 1/n head shard (nq/n q-heads, nkv/n KV
    /// heads; GQA ratio preserved). Identical math to `qwen35_gated_attention_gpu`.
    /// Returns the PARTIAL o_proj (row-sharded over this rank's q_dim) — the
    /// caller all-reduces it to the full hidden contribution.
    pub(crate) fn qwen35_gated_attention_tp(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, x: &[f32], pos: usize, n: usize) -> Vec<f32> {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads / n;     // this rank's q heads (6)
        let nkv = cfg.num_key_value_heads / n;    // this rank's KV heads (1)
        let hd = cfg.head_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let ln = |s: &str| format!("model.layers.{layer_idx}.self_attn.{s}");

        // q_proj double-width [query|gate] per head; k/v normal. Sharded weights
        // → per-rank output dims. All three share x → ONE submit.
        let (qn_w, kn_w, vn_w) = (ln("q_proj.weight"), ln("k_proj.weight"), ln("v_proj.weight"));
        let mut qkv = self.qwen35_matvec_multi(x, h, &[
            (&qn_w, nq * hd * 2), (&kn_w, kv_dim), (&vn_w, kv_dim),
        ]);
        let v = qkv.pop().unwrap();
        let k = qkv.pop().unwrap();
        let q_and_gate = qkv.pop().unwrap();

        // Per-token attention core (q/gate split, qk-norm, RoPE, KV append,
        // causal SDPA, output gate) → this rank's gated q_dim slice.
        let gated = self.qwen35_gated_attention_tp_core(cfg, layer_idx, &q_and_gate, &k, &v, pos, n);
        // o_proj row-sharded over q_dim → PARTIAL (caller reduces).
        self.qwen35_matvec(&ln("o_proj.weight"), &gated, q_dim, h)
    }

    /// Per-token CORE of `qwen35_gated_attention_tp`, factored out so the batched
    /// verify (`qwen35_gated_attention_tp_batched`) reuses the EXACT same
    /// q/gate-split + qk-norm + RoPE + KV-append + causal SDPA + output-gate math
    /// on projections it computed with the single-stream cols kernel — only the
    /// projection DISPATCH differs. Consumes this rank's `q_and_gate`(nq*hd*2),
    /// `k_in`(kv_dim), `v`(kv_dim) projection outputs, appends K/V to the resident
    /// cache for absolute position `pos`, and returns the gated q_dim slice
    /// (pre-o_proj).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen35_gated_attention_tp_core(
        &mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize,
        q_and_gate: &[f32], k_in: &[f32], v: &[f32], pos: usize, n: usize,
    ) -> Vec<f32> {
        let eps = cfg.rms_norm_eps;
        let nq = cfg.num_attention_heads / n;     // this rank's q heads
        let nkv = cfg.num_key_value_heads / n;    // this rank's KV heads
        let hd = cfg.head_dim;
        let q_dim = nq * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let rotary = cfg.rotary_dim();
        let theta = cfg.rope_theta;
        let ln = |s: &str| format!("model.layers.{layer_idx}.self_attn.{s}");

        let mut q = vec![0.0f32; q_dim];
        let mut gate = vec![0.0f32; q_dim];
        for head in 0..nq {
            let base = head * 2 * hd;
            q[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base..base + hd]);
            gate[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[base + hd..base + 2 * hd]);
        }
        let mut k = k_in.to_vec();

        // Per-head Q/K RMSNorm (replicated weights; per head_dim).
        let qn = self.qwen35_w(&ln("q_norm.weight"));
        let kn = self.qwen35_w(&ln("k_norm.weight"));
        for hi in 0..nq {
            let s = &mut q[hi * hd..(hi + 1) * hd];
            let nn = model::cpu_rms_norm(s, &qn, eps);
            s.copy_from_slice(&nn);
        }
        for hi in 0..nkv {
            let s = &mut k[hi * hd..(hi + 1) * hd];
            let nn = model::cpu_rms_norm(s, &kn, eps);
            s.copy_from_slice(&nn);
        }

        // Partial RoPE (position-only → per-rank safe).
        model::cpu_rope(&mut q, &mut k, pos, nq, nkv, hd, rotary, theta);

        // KV cache (per-rank, sized nkv heads) + GQA causal SDPA over local heads.
        let attn_out = {
            let qm = self.qwen35.as_mut().unwrap();
            let si = qm.state_idx(layer_idx);
            let cache = match &mut qm.layer_state[si] {
                qwen35::LayerState::Full(c) => c,
                _ => unreachable!("full_attention layer has a KV cache"),
            };
            cache.append(&k, v);
            model::cpu_sdpa(&q, cache.k_up_to_now(), cache.v_up_to_now(), nq, nkv, hd, cache.seq_len, scale, None)
        };

        // Output gate (sigmoid) → gated q_dim slice.
        attn_out.iter().zip(&gate)
            .map(|(&aa, &g)| aa * (1.0 / (1.0 + (-g).exp()))).collect()
    }

    /// TP dense SwiGLU MLP: gate/up column-sharded (r_inter wide), down
    /// row-sharded → PARTIAL (caller all-reduces). Mirrors `qwen35_dense_mlp_gpu`.
    pub(crate) fn qwen35_dense_mlp_tp(&mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize, ff_in: &[f32], n: usize) -> Vec<f32> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size / n;   // rank-local intermediate
        let ln = |s: &str| format!("model.layers.{layer_idx}.mlp.{s}");
        let (gn, un) = (ln("gate_proj.weight"), ln("up_proj.weight"));
        let mut gu = self.qwen35_matvec_multi(ff_in, h, &[(&gn, inter), (&un, inter)]);
        let up = gu.pop().unwrap();
        let gate = gu.pop().unwrap();
        let act = model::cpu_silu(&gate);
        let mid: Vec<f32> = act.iter().zip(&up).map(|(&g, &u)| g * u).collect();
        self.qwen35_matvec(&ln("down_proj.weight"), &mid, inter, h)
    }

    // ── Design-A batched-verify TP mixers (perf/quant-batched-cols) ──────────
    //
    // The TP batched verify (`forward_tp_qwen35_verify_impl`) previously ran the
    // per-token mixers above `t` (=DEPTH+1) times SERIALLY — each call re-streamed
    // this rank's q8_0/f16 projection weights through `qwen35_matvec[_multi]`, so
    // only the all-reduce COUNT amortized, NOT the weight-streaming (the banked
    // [[gemm-batching-misses-quantized-hotpath]] NO-GO on bandwidth-bound
    // gfx1013). These batched mixers mirror the PP STEP-6 projection swap
    // (`qwen35_{full_attn,linear,dense_mlp}_prefill_gpu`): every projection goes
    // through `qwen35_gemm` under `spec_verify_cols`, so the single-stream cols
    // kernel (`mul_mat_vec_{f16,q8_0}_cols`) reads each weight element ONCE and
    // MACs it against all `t` activation columns → `t` outputs. The recurrent /
    // attention CORE (conv1d+delta-rule / qk-norm+RoPE+KV-append+SDPA) still runs
    // per token, sequentially, through the SAME `_core` helpers the serial mixers
    // use — so the resident KV/GDN state advances identically and the projection
    // MATH is unchanged; only the dispatch batches. Used ONLY by the TP verify
    // (t>=2); single-token decode/rollback keep the per-token mixers untouched.

    /// GatedAttention for all `t` verify tokens; q/k/v/o projections streamed once
    /// across the `t` columns (cols kernel). `x_all` = [t,h] row-major normed
    /// inputs at absolute positions `start_pos..start_pos+t`; returns [t,h]
    /// row-major PARTIAL o_proj (caller all-reduces).
    pub(crate) fn qwen35_gated_attention_tp_batched(
        &mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize,
        x_all: &[f32], start_pos: usize, t: usize, n: usize,
    ) -> Vec<f32> {
        let eps = cfg.rms_norm_eps;
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads / n;
        let nkv = cfg.num_key_value_heads / n;
        let hd = cfg.head_dim;
        let q_dim = nq * hd;
        let kv_dim = nkv * hd;
        let qg = nq * hd * 2;                         // double-width q_proj [query|gate]
        let scale = 1.0 / (hd as f32).sqrt();
        let rotary = cfg.rotary_dim();
        let theta = cfg.rope_theta;
        let ln = |s: &str| format!("model.layers.{layer_idx}.self_attn.{s}");

        // Batched input projections: weight streamed ONCE for all `t` columns.
        self.spec_verify_cols = true;
        let q_and_gate = self.qwen35_gemm(&ln("q_proj.weight"), x_all, t, h, qg);
        let k_raw = self.qwen35_gemm(&ln("k_proj.weight"), x_all, t, h, kv_dim);
        let v_all = self.qwen35_gemm(&ln("v_proj.weight"), x_all, t, h, kv_dim);
        self.spec_verify_cols = false;

        // ── Phase 1: q/gate split + per-head qk-norm + RoPE, and APPEND K/V to
        // the resident cache IN ORDER (bit-identical to the per-token `_core`
        // preamble, tp.rs qwen35_gated_attention_tp_core, only unrolled over T).
        // Cheap, per-token elementwise/append — kept serial (append order is
        // load-bearing; the expensive SDPA below is what we parallelize).
        let qn = self.qwen35_w(&ln("q_norm.weight"));
        let kn = self.qwen35_w(&ln("k_norm.weight"));
        let mut q_all = vec![0.0f32; t * q_dim];       // q-norm'd + RoPE'd queries
        let mut gate_all = vec![0.0f32; t * q_dim];    // raw output-gate logits
        for ti in 0..t {
            let base = ti * qg;
            let mut q = vec![0.0f32; q_dim];
            let mut gate = vec![0.0f32; q_dim];
            for head in 0..nq {
                let b = base + head * 2 * hd;
                q[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[b..b + hd]);
                gate[head * hd..(head + 1) * hd].copy_from_slice(&q_and_gate[b + hd..b + 2 * hd]);
            }
            let mut k = k_raw[ti * kv_dim..(ti + 1) * kv_dim].to_vec();
            for hi in 0..nq {
                let s = &mut q[hi * hd..(hi + 1) * hd];
                let nnv = model::cpu_rms_norm(s, &qn, eps);
                s.copy_from_slice(&nnv);
            }
            for hi in 0..nkv {
                let s = &mut k[hi * hd..(hi + 1) * hd];
                let nnv = model::cpu_rms_norm(s, &kn, eps);
                s.copy_from_slice(&nnv);
            }
            model::cpu_rope(&mut q, &mut k, start_pos + ti, nq, nkv, hd, rotary, theta);
            {
                let qm = self.qwen35.as_mut().unwrap();
                let si = qm.state_idx(layer_idx);
                let cache = match &mut qm.layer_state[si] {
                    qwen35::LayerState::Full(c) => c,
                    _ => unreachable!("full_attention layer has a KV cache"),
                };
                cache.append(&k, &v_all[ti * kv_dim..(ti + 1) * kv_dim]);
            }
            q_all[ti * q_dim..(ti + 1) * q_dim].copy_from_slice(&q);
            gate_all[ti * q_dim..(ti + 1) * q_dim].copy_from_slice(&gate);
        }

        // ── Phase 2: PARALLEL causal SDPA over the T×nq head-jobs, reading the
        // now-complete cache ONCE. Byte-exact vs the per-token `cpu_sdpa` the
        // `_core` ran serially (same single-accumulator dot/softmax/AV, no
        // cross-thread reduction) — this is the flip: the TP verify's dominant,
        // previously single-threaded attention compute now uses every core.
        let out_all = {
            let qm = self.qwen35.as_ref().unwrap();
            let si = qm.state_idx(layer_idx);
            let cache = match &qm.layer_state[si] {
                qwen35::LayerState::Full(c) => c,
                _ => unreachable!("full_attention layer has a KV cache"),
            };
            model::cpu_sdpa_batched_causal(
                &q_all, &cache.k, &cache.v, t, start_pos, nq, nkv, hd, scale)
        };

        // ── Phase 3: output gate (sigmoid), then batched o_proj → [t,h] PARTIAL.
        let gated_all: Vec<f32> = out_all.iter().zip(&gate_all)
            .map(|(&a, &g)| a * (1.0 / (1.0 + (-g).exp()))).collect();
        self.spec_verify_cols = true;
        let out = self.qwen35_gemm(&ln("o_proj.weight"), &gated_all, t, q_dim, h);
        self.spec_verify_cols = false;
        out
    }

    /// GatedDeltaNet for all `t` verify tokens; in_proj_qkv/z/a/b + out_proj
    /// streamed once across the `t` columns (cols kernel). Host-authoritative
    /// recurrence (conv1d + delta rule) runs per token, sequentially, via
    /// `qwen35_delta_net_tp_core`. `x_all` = [t,h]; returns [t,h] PARTIAL.
    pub(crate) fn qwen35_delta_net_tp_batched(
        &mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize,
        x_all: &[f32], t: usize, n: usize,
    ) -> Vec<f32> {
        let nk = cfg.linear_num_key_heads / n;
        let nv = cfg.linear_num_value_heads / n;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let h = cfg.hidden_size;
        let ln = |s: &str| format!("model.layers.{layer_idx}.linear_attn.{s}");

        // Batched input projections (weight streamed ONCE for all `t` columns).
        self.spec_verify_cols = true;
        let qkv = self.qwen35_gemm(&ln("in_proj_qkv.weight"), x_all, t, h, conv_dim);
        let z = self.qwen35_gemm(&ln("in_proj_z.weight"), x_all, t, h, value_dim);
        let a = self.qwen35_gemm(&ln("in_proj_a.weight"), x_all, t, h, nv);
        let b = self.qwen35_gemm(&ln("in_proj_b.weight"), x_all, t, h, nv);

        // Per-token recurrent core (SEQUENTIAL: conv/state advance in order).
        let mut gated_all = vec![0.0f32; t * value_dim];
        for ti in 0..t {
            let g = self.qwen35_delta_net_tp_core(
                cfg, layer_idx,
                &qkv[ti * conv_dim..(ti + 1) * conv_dim],
                &z[ti * value_dim..(ti + 1) * value_dim],
                &a[ti * nv..(ti + 1) * nv],
                &b[ti * nv..(ti + 1) * nv], n);
            gated_all[ti * value_dim..(ti + 1) * value_dim].copy_from_slice(&g);
        }

        // Batched out_proj → [t,h] PARTIAL.
        let out = self.qwen35_gemm(&ln("out_proj.weight"), &gated_all, t, value_dim, h);
        self.spec_verify_cols = false;
        out
    }

    /// Dense SwiGLU MLP for all `t` verify tokens; gate/up/down streamed once
    /// across the `t` columns (cols kernel). SwiGLU is elementwise → the whole
    /// [t,inter] batch is bit-identical to the per-token path. `ff_all` = [t,h];
    /// returns [t,h] PARTIAL.
    pub(crate) fn qwen35_dense_mlp_tp_batched(
        &mut self, cfg: &qwen35::Qwen35Config, layer_idx: usize,
        ff_all: &[f32], t: usize, n: usize,
    ) -> Vec<f32> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size / n;
        let ln = |s: &str| format!("model.layers.{layer_idx}.mlp.{s}");
        self.spec_verify_cols = true;
        let gate = self.qwen35_gemm(&ln("gate_proj.weight"), ff_all, t, h, inter);
        let up = self.qwen35_gemm(&ln("up_proj.weight"), ff_all, t, h, inter);
        let act = model::cpu_silu(&gate);
        let mid: Vec<f32> = act.iter().zip(&up).map(|(&g, &u)| g * u).collect();
        let out = self.qwen35_gemm(&ln("down_proj.weight"), &mid, t, inter, h);
        self.spec_verify_cols = false;
        out
    }

}

/// Projection weights for the qwen3_5 (Qwen3.6) hybrid that go to the GPU as
/// f16. GatedDeltaNet: in_proj_qkv/z/a/b + out_proj; GatedAttention:
/// q/k/v/o_proj; dense MLP: gate/up/down_proj. The conv1d.weight, A_log,
/// dt_bias, gated-norm.weight and every layernorm stay f32 on the host.
#[cfg(feature = "qwen35")]
pub(crate) fn is_qwen35_matvec_weight(name: &str) -> bool {
    name.ends_with(".q_proj.weight")
        || name.ends_with(".k_proj.weight")
        || name.ends_with(".v_proj.weight")
        || name.ends_with(".o_proj.weight")
        || name.ends_with(".gate_proj.weight")
        || name.ends_with(".up_proj.weight")
        || name.ends_with(".down_proj.weight")
        || name.ends_with(".in_proj_qkv.weight")
        || name.ends_with(".in_proj_z.weight")
        || name.ends_with(".in_proj_a.weight")
        || name.ends_with(".in_proj_b.weight")
        || name.ends_with(".out_proj.weight")
}


/// Megatron tensor-parallel sharding of a single qwen3_5 (Qwen3.6) weight for
/// rank `r` of `n` (row-major `[out, in]`). Returns the sliced f32 data, or the
/// original (cloned) when the weight is REPLICATED (norms, layernorms, embed,
/// lm_head, gated-norm). Called in the loader (projections in `on_proj`,
/// recurrence host weights in the `raw` loop) so every per-rank buffer is
/// already 1/N and the `forward_tp_qwen35` matvec dims line up.
///
/// Sharding (this rank owns heads `[r*H/n,(r+1)*H/n)` of every head set):
/// - GatedAttention: `q_proj` [nq*2*hd, h] column-shard by attn head (each head
///   is a contiguous `2*hd` block: [query|gate]); `k/v_proj` [nkv*hd, h] by KV
///   head; `o_proj` [h, nq*hd] row-shard by attn head's `hd` input slice.
/// - GatedDeltaNet: `in_proj_qkv` [conv_dim, h] is q(key_dim)|k(key_dim)|v(value_dim)
///   concatenated → 3-segment column-shard (this rank's key heads of q, of k,
///   then value heads of v); `conv1d.weight` [conv_dim,1,kern] matches that same
///   3-segment split (per channel); `in_proj_z` [value_dim, h] by value head;
///   `in_proj_a`/`b` [nv, h] (1 row/v-head) by value head; `A_log`/`dt_bias`
///   [nv] by value head; `out_proj` [h, value_dim] row-shard by value head.
/// - REPLICATED: q_norm/k_norm/gated-norm (per head_dim), all layernorms,
///   embed_tokens, lm_head, model.norm, MoE router/experts (TP=4 dense only).
#[cfg(feature = "qwen35")]
pub(crate) fn q35_tp_shard<T: Copy>(name: &str, data: Vec<T>, cfg: &qwen35::Qwen35Config, r: usize, n: usize) -> Vec<T> {
    if n <= 1 { return data; }
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let nq = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let nk = cfg.linear_num_key_heads;
    let nv = cfg.linear_num_value_heads;
    let kd = cfg.linear_key_head_dim;
    let vd = cfg.linear_value_head_dim;
    let key_dim = cfg.key_dim();      // nk*kd
    let value_dim = cfg.value_dim();  // nv*vd
    let kern = cfg.linear_conv_kernel_dim;

    // column-shard: keep output rows [r*out/n,(r+1)*out/n) of a [out,in] weight.
    let col = |data: Vec<T>, in_feat: usize| -> Vec<T> {
        let out = data.len() / in_feat;
        let per = out / n;
        let lo = r * per;
        data[lo * in_feat..(lo + per) * in_feat].to_vec()
    };
    // row-shard: keep input cols [r*in/n,(r+1)*in/n) of a [out,in] weight.
    let row = |data: Vec<T>, in_feat: usize| -> Vec<T> {
        let out = data.len() / in_feat;
        let per = in_feat / n;
        let lo = r * per;
        let mut o = Vec::with_capacity(out * per);
        for rr in 0..out { o.extend_from_slice(&data[rr * in_feat + lo..rr * in_feat + lo + per]); }
        o
    };
    // 3-segment column-shard for in_proj_qkv [conv_dim, in]: q(key_dim) | k(key_dim)
    // | v(value_dim). Keep this rank's nk/n key-heads of q & k, and nv/n
    // value-heads of v, in that concatenated order (so the sharded conv_dim layout
    // is exactly q_local | k_local | v_local, matching the sharded conv1d below).
    let qkv_seg = |data: Vec<T>, in_feat: usize| -> Vec<T> {
        let rkd = key_dim / n;            // this rank's q/k channel width
        let rvd = value_dim / n;          // this rank's v channel width
        let qoff = 0usize;
        let koff = key_dim;
        let voff = 2 * key_dim;
        let mut o = Vec::with_capacity((2 * rkd + rvd) * in_feat);
        for &(base, lo, len) in &[
            (qoff, r * rkd, rkd),
            (koff, r * rkd, rkd),
            (voff, r * rvd, rvd),
        ] {
            let start = base + lo;
            o.extend_from_slice(&data[start * in_feat..(start + len) * in_feat]);
        }
        o
    };

    // ── GatedAttention ──────────────────────────────────────────────────────
    if name.ends_with(".self_attn.q_proj.weight") { return col(data, h); }   // [nq*2*hd, h]
    if name.ends_with(".self_attn.k_proj.weight") { return col(data, h); }   // [nkv*hd, h]
    if name.ends_with(".self_attn.v_proj.weight") { return col(data, h); }
    if name.ends_with(".self_attn.o_proj.weight") { return row(data, nq * hd); } // [h, nq*hd]

    // ── GatedDeltaNet ───────────────────────────────────────────────────────
    if name.ends_with(".linear_attn.in_proj_qkv.weight") { return qkv_seg(data, h); }
    if name.ends_with(".linear_attn.in_proj_z.weight") { return col(data, h); }   // [value_dim, h]
    if name.ends_with(".linear_attn.in_proj_a.weight") { return col(data, h); }   // [nv, h]
    if name.ends_with(".linear_attn.in_proj_b.weight") { return col(data, h); }   // [nv, h]
    if name.ends_with(".linear_attn.out_proj.weight") { return row(data, value_dim); } // [h, value_dim]
    if name.ends_with(".linear_attn.conv1d.weight") {
        // [conv_dim, 1, kern] flat = [conv_dim, kern]; per-channel rows, same
        // q|k|v 3-segment split as in_proj_qkv. in_feat = kern.
        let _ = (nk, nv, kd, vd);
        return qkv_seg(data, kern);
    }
    if name.ends_with(".linear_attn.A_log") { return col(data, 1); }    // [nv]
    if name.ends_with(".linear_attn.dt_bias") { return col(data, 1); }  // [nv]

    // ── MLP (dense) ─────────────────────────────────────────────────────────
    if name.ends_with(".mlp.gate_proj.weight") { return col(data, h); }
    if name.ends_with(".mlp.up_proj.weight") { return col(data, h); }
    if name.ends_with(".mlp.down_proj.weight") {
        let inter = data.len() / h;  // [h, inter]
        return row(data, inter);
    }

    // Everything else (q_norm/k_norm, linear_attn.norm, all layernorms,
    // embed_tokens, lm_head, model.norm) is REPLICATED.
    data
}


/// Column-shard a PACKED NVFP4 weight: keep OUTPUT rows [lo, lo+per). Packed rows
/// (bpr bytes each) and scale groups (groups each) are both row-contiguous, so this
/// is two contiguous slices. Returns (packed', folded', out'=per, in'=in unchanged).
pub(crate) fn nvfp4_shard_rows<T: Copy>(packed: &[u8], scale: &[T], in_features: usize, gs: usize,
                    lo: usize, per: usize) -> (Vec<u8>, Vec<T>, usize, usize) {
    // Generic over the scale element type so the SAME row-contiguous slice
    // serves both the default f32-fold path (`scale: &[f32]`) and the
    // e4m3-resident path (`scale: &[u8]`, the raw block-scale bytes) —
    // per-(row,group) slicing is identical for either element type.
    let bpr = in_features / 2;
    let groups = in_features / gs;
    (packed[lo*bpr..(lo+per)*bpr].to_vec(),
     scale[lo*groups..(lo+per)*groups].to_vec(),
     per, in_features)
}


/// Row-shard a PACKED NVFP4 weight: keep INPUT cols [clo, clo+cper) within every
/// output row. `cper` MUST be even (nibble→byte) and divisible by `gs` (group
/// boundary); asserted. Returns (packed', folded', out'=out unchanged, in'=cper).
pub(crate) fn nvfp4_shard_cols<T: Copy>(packed: &[u8], scale: &[T], out_features: usize, in_features: usize,
                    gs: usize, clo: usize, cper: usize) -> (Vec<u8>, Vec<T>, usize, usize) {
    // Generic over the scale element type — see `nvfp4_shard_rows`. The scale
    // is sliced per-row by group index, identical for f32-fold or e4m3 bytes.
    assert_eq!(clo % 2, 0, "nvfp4 row-shard col start {clo} not byte-aligned");
    assert_eq!(cper % 2, 0, "nvfp4 row-shard width {cper} not byte-aligned");
    assert_eq!(clo % gs, 0, "nvfp4 row-shard col start {clo} not group({gs})-aligned");
    assert_eq!(cper % gs, 0, "nvfp4 row-shard width {cper} not group({gs})-aligned");
    let bpr = in_features / 2;
    let groups = in_features / gs;
    let (bclo, bcper) = (clo/2, cper/2);
    let (gclo, gcper) = (clo/gs, cper/gs);
    let mut p = Vec::with_capacity(out_features * bcper);
    let mut s = Vec::with_capacity(out_features * gcper);
    for rr in 0..out_features {
        p.extend_from_slice(&packed[rr*bpr + bclo .. rr*bpr + bclo + bcper]);
        s.extend_from_slice(&scale[rr*groups + gclo .. rr*groups + gclo + gcper]);
    }
    (p, s, out_features, cper)
}


/// Full name-dispatched TP shard of a PACKED NVFP4 matvec weight, mirroring
/// `q35_tp_shard`'s col/row/qkv_seg mapping but on (packed u8, folded f32). Only
/// the mlp gate/up (col) and down (row) arms are exercised by the 27B checkpoint
/// (attention is FP8); the attention arms are included for a future NVFP4-attn
/// checkpoint. Returns (packed', folded', out', in').
#[cfg(feature = "qwen35")]
pub(crate) fn nvfp4_tp_shard<T: Copy>(name: &str, packed: &[u8], scale: &[T],
                  out_features: usize, in_features: usize, gs: usize,
                  cfg: &qwen35::Qwen35Config, r: usize, n: usize)
                  -> (Vec<u8>, Vec<T>, usize, usize) {
    // Generic over the scale element type: `scale: &[f32]` for the default
    // f32-fold path, `scale: &[u8]` for the e4m3-resident path. The name-driven
    // col/row/qkv_seg geometry is identical either way.
    if n <= 1 { return (packed.to_vec(), scale.to_vec(), out_features, in_features); }
    let col = |lo, per| nvfp4_shard_rows(packed, scale, in_features, gs, lo, per);
    let row = |clo, cper| nvfp4_shard_cols(packed, scale, out_features, in_features, gs, clo, cper);
    let per_out = out_features / n;
    let per_in  = in_features / n;
    // column-shard: output-row slice
    if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight")
        || name.ends_with(".self_attn.q_proj.weight") || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight") || name.ends_with(".linear_attn.in_proj_z.weight") {
        return col(r*per_out, per_out);
    }
    // row-shard: input-col slice
    if name.ends_with(".mlp.down_proj.weight") || name.ends_with(".self_attn.o_proj.weight")
        || name.ends_with(".linear_attn.out_proj.weight") {
        return row(r*per_in, per_in);
    }
    // in_proj_qkv: 3-segment output-row col-shard (q | k | v), mirroring q35_tp_shard::qkv_seg
    if name.ends_with(".linear_attn.in_proj_qkv.weight") {
        let bpr = in_features / 2;
        let groups = in_features / gs;
        let rkd = cfg.key_dim() / n;
        let rvd = cfg.value_dim() / n;
        let segs = [(0usize, r*rkd, rkd), (cfg.key_dim(), r*rkd, rkd), (2*cfg.key_dim(), r*rvd, rvd)];
        let (mut p, mut s, mut orows) = (Vec::new(), Vec::new(), 0usize);
        for (base, lo, len) in segs {
            let lo = base + lo;
            p.extend_from_slice(&packed[lo*bpr..(lo+len)*bpr]);
            s.extend_from_slice(&scale[lo*groups..(lo+len)*groups]);
            orows += len;
        }
        return (p, s, orows, in_features);
    }
    (packed.to_vec(), scale.to_vec(), out_features, in_features) // replicate (shouldn't hit for matvec)
}


/// Column-shard a PACKED MLX4-affine weight: keep OUTPUT rows [lo, lo+per).
/// Packed rows (wpr words each, 8 nibbles/word) and scale/bias groups (groups
/// each) are all row-contiguous, so this is three contiguous slices. Returns
/// (packed', scales', biases', out'=per, in'=in unchanged).
pub(crate) fn mlx4_shard_rows(packed: &[u32], scales: &[f32], biases: &[f32], in_features: usize, gs: usize,
                    lo: usize, per: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>, usize, usize) {
    let wpr = in_features / 8;
    let groups = in_features / gs;
    (packed[lo*wpr..(lo+per)*wpr].to_vec(),
     scales[lo*groups..(lo+per)*groups].to_vec(),
     biases[lo*groups..(lo+per)*groups].to_vec(),
     per, in_features)
}


/// Row-shard a PACKED MLX4-affine weight: keep INPUT cols [clo, clo+cper)
/// within every output row. `cper` MUST be word-aligned (8 nibbles/word) and
/// divisible by `gs` (group boundary); asserted. Returns (packed', scales',
/// biases', out'=out unchanged, in'=cper).
pub(crate) fn mlx4_shard_cols(packed: &[u32], scales: &[f32], biases: &[f32], out_features: usize, in_features: usize,
                    gs: usize, clo: usize, cper: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>, usize, usize) {
    assert_eq!(clo % 8, 0, "mlx4 row-shard col start {clo} not word(8-nibble)-aligned");
    assert_eq!(cper % 8, 0, "mlx4 row-shard width {cper} not word(8-nibble)-aligned");
    assert_eq!(clo % gs, 0, "mlx4 row-shard col start {clo} not group({gs})-aligned");
    assert_eq!(cper % gs, 0, "mlx4 row-shard width {cper} not group({gs})-aligned");
    let wpr = in_features / 8;
    let groups = in_features / gs;
    let (wclo, wcper) = (clo/8, cper/8);
    let (gclo, gcper) = (clo/gs, cper/gs);
    let mut p = Vec::with_capacity(out_features * wcper);
    let mut s = Vec::with_capacity(out_features * gcper);
    let mut b = Vec::with_capacity(out_features * gcper);
    for rr in 0..out_features {
        p.extend_from_slice(&packed[rr*wpr + wclo .. rr*wpr + wclo + wcper]);
        s.extend_from_slice(&scales[rr*groups + gclo .. rr*groups + gclo + gcper]);
        b.extend_from_slice(&biases[rr*groups + gclo .. rr*groups + gclo + gcper]);
    }
    (p, s, b, out_features, cper)
}


/// Full name-dispatched TP shard of a PACKED MLX4-affine DENSE matvec weight
/// (GatedDeltaNet + full-attn projections), mirroring `nvfp4_tp_shard`'s col/
/// row/qkv_seg mapping but on (packed u32, scales f32, biases f32) and
/// covering the GDN `in_proj_a`/`in_proj_b` gates too (the MLX4-bit 27B-dense
/// checkpoint quantizes ALL dense matvec weights this way, unlike the 27B-
/// NVFP4 checkpoint where attention/GDN are FP8 and only MLP is nvfp4 — see
/// `nvfp4_tp_shard`'s doc comment). Needed for the 27B-dense TP-4 A/B (the
/// 35B-A3B PP stage owns whole layers -> `q35_tp_size==1` -> no-op
/// passthrough). Returns (packed', scales', biases', out', in').
#[cfg(feature = "qwen35")]
pub(crate) fn mlx4_tp_shard(name: &str, packed: &[u32], scales: &[f32], biases: &[f32],
                  out_features: usize, in_features: usize, gs: usize,
                  cfg: &qwen35::Qwen35Config, r: usize, n: usize)
                  -> (Vec<u32>, Vec<f32>, Vec<f32>, usize, usize) {
    if n <= 1 { return (packed.to_vec(), scales.to_vec(), biases.to_vec(), out_features, in_features); }
    let col = |lo, per| mlx4_shard_rows(packed, scales, biases, in_features, gs, lo, per);
    let row = |clo, cper| mlx4_shard_cols(packed, scales, biases, out_features, in_features, gs, clo, cper);
    let per_out = out_features / n;
    let per_in  = in_features / n;
    // column-shard: output-row slice
    if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight")
        || name.ends_with(".self_attn.q_proj.weight") || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight") || name.ends_with(".linear_attn.in_proj_z.weight")
        || name.ends_with(".linear_attn.in_proj_a.weight") || name.ends_with(".linear_attn.in_proj_b.weight") {
        return col(r*per_out, per_out);
    }
    // row-shard: input-col slice
    if name.ends_with(".mlp.down_proj.weight") || name.ends_with(".self_attn.o_proj.weight")
        || name.ends_with(".linear_attn.out_proj.weight") {
        return row(r*per_in, per_in);
    }
    // in_proj_qkv: 3-segment output-row col-shard (q | k | v), mirroring q35_tp_shard::qkv_seg
    if name.ends_with(".linear_attn.in_proj_qkv.weight") {
        let wpr = in_features / 8;
        let groups = in_features / gs;
        let rkd = cfg.key_dim() / n;
        let rvd = cfg.value_dim() / n;
        let segs = [(0usize, r*rkd, rkd), (cfg.key_dim(), r*rkd, rkd), (2*cfg.key_dim(), r*rvd, rvd)];
        let (mut p, mut s, mut b, mut orows) = (Vec::new(), Vec::new(), Vec::new(), 0usize);
        for (base, lo, len) in segs {
            let lo = base + lo;
            p.extend_from_slice(&packed[lo*wpr..(lo+len)*wpr]);
            s.extend_from_slice(&scales[lo*groups..(lo+len)*groups]);
            b.extend_from_slice(&biases[lo*groups..(lo+len)*groups]);
            orows += len;
        }
        return (p, s, b, orows, in_features);
    }
    (packed.to_vec(), scales.to_vec(), biases.to_vec(), out_features, in_features) // replicate (shouldn't hit for matvec)
}


// ─── Gemma-31B (gemma4 G31b) head-aware NVFP4/FP8 TP shard ──────────────────
//
// INC-1b: Megatron-style TP-4 shard of the g31b NVFP4+FP8 mixed checkpoint
// (`model::load_gemma_nvfp4_weights`). Unlike qwen3_5's uniform head layout,
// gemma's per-layer head/dim geometry ALTERNATES: sliding layers use
// `num_key_value_heads` (16) @ `head_dim` (256); the period-6 GLOBAL layers
// use `num_global_key_value_heads` (4) @ `global_head_dim` (512) — so every
// shard decision here is layer-index-aware via `Gemma4Config::layer_*`
// (`gemma_layer_idx` parses the layer index straight out of the tensor name,
// mirroring `gemma31b_mlp_is_fp8_exception`'s convention in model.rs).
//
// Checkpoint quant layout (see `load_gemma_nvfp4_weights`'s doc comment):
// `self_attn.{q,k,v,o}_proj` are ALWAYS FP8 (never NVFP4); `mlp.{gate,up,
// down}_proj` are NVFP4 EXCEPT on layers 1/57/58/59 (FP8). So the NVFP4 shard
// helper below only ever sees mlp tensors; the FP8 shard helper sees both
// attn (every layer) and mlp (the 4 exception layers).
//
// Sharding rule (rank `r` of `n`; this rank owns heads `[r*H/n,(r+1)*H/n)`):
// - `q_proj`: column-shard (output rows) by Q head — `num_attention_heads/n`
//   heads/rank, each `layer_head_dim(idx)` wide (32 heads → 8/rank @ TP-4).
// - `k_proj`/`v_proj`: column-shard by KV head — `layer_num_kv_heads(idx)/n`
//   heads/rank (sliding 16/n=4/rank @256; global 4/n=1/rank @512). `v_proj`
//   doesn't exist on value-less global layers (`layer_uses_k_eq_v`) — the
//   loader never offers it there, so this arm only fires for sliding layers
//   in practice, but the math is head-dim-correct for either.
// - `o_proj`: row-shard (input cols) by the SAME q-head split as `q_proj`.
// - `mlp.gate_proj`/`up_proj`: column-shard (output rows) over the
//   intermediate dim (21504 dense in g31b, no double-wide layers) — `/n`
//   per rank (→ 5376/rank @ TP-4, 336 NVFP4 groups of 16).
// - `mlp.down_proj`: row-shard (input cols), same `/n` split of intermediate.
// - Everything else (embed_tokens, all norms, `layer_scalar`) is REPLICATED —
//   these never reach the packed-weight sink at all (they flow through
//   `ProjWeight::F32`/the host f16 embed map untouched), so there is no
//   shard arm for them here.

/// Parse the global decoder-layer index out of a `model.layers.{idx}....`
/// tensor name (gemma's remapped `model.*` namespace — mirrors
/// `gemma31b_mlp_is_fp8_exception`'s convention in model.rs).
#[cfg(feature = "gemma")]
fn gemma_layer_idx(name: &str) -> Option<usize> {
    name.strip_prefix("model.layers.")
        .and_then(|rest| rest.split('.').next())
        .and_then(|s| s.parse::<usize>().ok())
}

/// Column-shard an unpacked FP8-E4M3 weight (1 byte/elem, row-major `[out,
/// in]`) + its scale (length 1 = per-tensor broadcast, or `out_features` =
/// per-row): keep OUTPUT rows `[lo, lo+per)`. Both the weight rows and (when
/// per-row) the matching scale entries are contiguous, so this is two
/// straight slices. Returns `(weight', scale', out'=per, in'=in unchanged)`.
pub(crate) fn fp8_shard_rows(weight: &[u8], scale: &[f32], in_features: usize,
                  lo: usize, per: usize) -> (Vec<u8>, Vec<f32>, usize, usize) {
    let per_row = scale.len() > 1;
    let w = weight[lo * in_features..(lo + per) * in_features].to_vec();
    let s = if per_row { scale[lo..lo + per].to_vec() } else { scale.to_vec() };
    (w, s, per, in_features)
}

/// Row-shard an unpacked FP8-E4M3 weight: keep INPUT cols `[clo, clo+cper)`
/// within every output row. The scale (per-tensor or per-OUTPUT-row) doesn't
/// depend on the input dim, so it passes through unchanged. Returns
/// `(weight', scale' (unchanged), out'=out unchanged, in'=cper)`.
pub(crate) fn fp8_shard_cols(weight: &[u8], scale: &[f32], out_features: usize, in_features: usize,
                  clo: usize, cper: usize) -> (Vec<u8>, Vec<f32>, usize, usize) {
    let mut w = Vec::with_capacity(out_features * cper);
    for rr in 0..out_features {
        w.extend_from_slice(&weight[rr * in_features + clo..rr * in_features + clo + cper]);
    }
    (w, scale.to_vec(), out_features, cper)
}

/// Head-aware TP shard of a PACKED FP8 gemma-31B weight (attn q/k/v/o on
/// every layer, plus mlp gate/up/down on the 4 FP8-exception layers). `name`
/// must be the post-remap `model.layers.{idx}...` form (`gemma_layer_idx`
/// parses the layer index for the per-layer-type head split — sliding vs
/// global differ in both `num_kv_heads` and `head_dim`). Returns
/// `(weight', scale', out', in')`.
#[cfg(feature = "gemma")]
pub(crate) fn gemma_fp8_tp_shard(name: &str, weight: &[u8], scale: &[f32],
                  out_features: usize, in_features: usize,
                  cfg: &model::Gemma4Config, r: usize, n: usize)
                  -> (Vec<u8>, Vec<f32>, usize, usize) {
    if n <= 1 { return (weight.to_vec(), scale.to_vec(), out_features, in_features); }
    let idx = gemma_layer_idx(name)
        .unwrap_or_else(|| panic!("gemma_fp8_tp_shard: no layer index in '{name}'"));
    let hd = cfg.layer_head_dim(idx);

    if name.ends_with(".self_attn.q_proj.weight") {
        let nq = cfg.num_attention_heads;
        assert_eq!(nq % n, 0, "gemma q-heads {nq} not divisible by TP {n} (layer {idx})");
        let per = (nq / n) * hd;
        return fp8_shard_rows(weight, scale, in_features, r * per, per);
    }
    if name.ends_with(".self_attn.k_proj.weight") || name.ends_with(".self_attn.v_proj.weight") {
        // REPLICATED (not sharded): `gemma_attn_tp` reads the full per-layer
        // K/V on every rank (KV cache is allocated full at KvCache::new, RoPE/
        // K-V-norm/SDPA all iterate the full `layer_num_kv_heads`). Sharding
        // k/v here would leave the forward dispatching a full-kv_dim matvec
        // against a 1/n-row buffer → the same OOB→robustBufferAccess(0) garbage
        // the global-layer const-num_kv bug produced, but on ALL layers at
        // TP≥2. Mirrors the qwen replicated-KV path (lib.rs is_gemma_tp arm).
        return (weight.to_vec(), scale.to_vec(), out_features, in_features);
    }
    if name.ends_with(".self_attn.o_proj.weight") {
        let nq = cfg.num_attention_heads;
        assert_eq!(nq % n, 0, "gemma q-heads {nq} not divisible by TP {n} (layer {idx})");
        let per = (nq / n) * hd; // same q-head input slice o_proj consumes ([h, nq*hd])
        return fp8_shard_cols(weight, scale, out_features, in_features, r * per, per);
    }
    if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
        assert_eq!(out_features % n, 0, "gemma mlp intermediate {out_features} not divisible by TP {n}");
        let per = out_features / n;
        return fp8_shard_rows(weight, scale, in_features, r * per, per);
    }
    if name.ends_with(".mlp.down_proj.weight") {
        assert_eq!(in_features % n, 0, "gemma mlp intermediate {in_features} not divisible by TP {n}");
        let per = in_features / n;
        return fp8_shard_cols(weight, scale, out_features, in_features, r * per, per);
    }
    (weight.to_vec(), scale.to_vec(), out_features, in_features) // replicate (shouldn't hit for matvec)
}

/// Head-aware TP shard of a PACKED NVFP4 gemma-31B `mlp.{gate,up,down}_proj`
/// weight (the only tensors this checkpoint ever quantizes NVFP4 — attn is
/// always FP8, see `gemma_fp8_tp_shard`). Reuses the byte/group-aligned
/// `nvfp4_shard_rows`/`nvfp4_shard_cols` primitives already proven by
/// `q35_tp_shard`'s NVFP4 arm; every split here lands on a whole
/// `intermediate_size/n` boundary, which for g31b (21504/4=5376) is a
/// multiple of `group_size` (16, 336 groups/rank) — `nvfp4_shard_cols`
/// asserts this at the byte/group level. Returns `(packed', folded', out',
/// in')`.
#[cfg(feature = "gemma")]
pub(crate) fn gemma_nvfp4_tp_shard<T: Copy>(name: &str, packed: &[u8], scale: &[T],
                  out_features: usize, in_features: usize, gs: usize,
                  r: usize, n: usize) -> (Vec<u8>, Vec<T>, usize, usize) {
    // Generic over the scale element type (f32-fold or e4m3 bytes) — see
    // `nvfp4_tp_shard`. gemma-31B NVFP4 is mlp-only (attn is FP8).
    if n <= 1 { return (packed.to_vec(), scale.to_vec(), out_features, in_features); }
    if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
        assert_eq!(out_features % n, 0, "gemma mlp intermediate {out_features} not divisible by TP {n}");
        let per = out_features / n;
        return nvfp4_shard_rows(packed, scale, in_features, gs, r * per, per);
    }
    if name.ends_with(".mlp.down_proj.weight") {
        assert_eq!(in_features % n, 0, "gemma mlp intermediate {in_features} not divisible by TP {n}");
        let per = in_features / n;
        return nvfp4_shard_cols(packed, scale, out_features, in_features, gs, r * per, per);
    }
    (packed.to_vec(), scale.to_vec(), out_features, in_features) // replicate (shouldn't hit for matvec)
}

/// Head-aware TP shard of an already-DEQUANTIZED (f32) gemma-31B matvec
/// weight — the fallback path exercised when `VLLM_VULKAN_NVFP4_GPU` is off
/// (or no GPU device is available), where NVFP4/FP8 decline the packed form
/// and the loader hands back a plain `[out, in]` row-major f32 buffer.
/// Mirrors `gemma_fp8_tp_shard`'s name/layer dispatch (generic over `T` like
/// `q35_tp_shard`, since the same row/col slicing works for f32 or f16).
/// Tensors with no layer index (embed_tokens, model.norm, ...) are always
/// REPLICATED (returned unchanged) — `gemma_layer_idx` returns `None` for
/// them, same as every other non-matvec tensor.
#[cfg(feature = "gemma")]
pub(crate) fn gemma_tp_shard_f32<T: Copy>(name: &str, data: Vec<T>, cfg: &model::Gemma4Config, r: usize, n: usize) -> Vec<T> {
    if n <= 1 { return data; }
    let idx = match gemma_layer_idx(name) {
        Some(i) => i,
        None => return data, // replicated (no layer index -> embed/norm/etc.)
    };
    let hd = cfg.layer_head_dim(idx);
    // column-shard: keep output rows [r*per,(r+1)*per) of a [out, in_feat] weight.
    let col = |data: Vec<T>, in_feat: usize, per: usize| -> Vec<T> {
        let lo = r * per;
        data[lo * in_feat..(lo + per) * in_feat].to_vec()
    };
    // row-shard: keep input cols [r*per,(r+1)*per) of a [out, in_feat] weight.
    let row = |data: Vec<T>, in_feat: usize, per: usize| -> Vec<T> {
        let out = data.len() / in_feat;
        let lo = r * per;
        let mut o = Vec::with_capacity(out * per);
        for rr in 0..out { o.extend_from_slice(&data[rr * in_feat + lo..rr * in_feat + lo + per]); }
        o
    };
    if name.ends_with(".self_attn.q_proj.weight") {
        let nq = cfg.num_attention_heads;
        return col(data, cfg.hidden_size, (nq / n) * hd);
    }
    if name.ends_with(".self_attn.k_proj.weight") || name.ends_with(".self_attn.v_proj.weight") {
        // REPLICATED — see `gemma_fp8_tp_shard`'s k/v arm: `gemma_attn_tp`
        // consumes the full per-layer K/V on every rank, so the f32 fallback
        // must keep k_proj/v_proj whole too.
        return data;
    }
    if name.ends_with(".self_attn.o_proj.weight") {
        let nq = cfg.num_attention_heads;
        return row(data, nq * hd, (nq / n) * hd);
    }
    if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
        let inter = cfg.layer_intermediate_size(idx);
        return col(data, cfg.hidden_size, inter / n);
    }
    if name.ends_with(".mlp.down_proj.weight") {
        let inter = cfg.layer_intermediate_size(idx);
        return row(data, inter, inter / n);
    }
    data // replicate (layernorms/q_norm/k_norm/layer_scalar — shouldn't hit for matvec)
}

// ── Generic TP vocab-sharded lm_head (any model family) ─────────────────────

/// (lo, per) of rank `r`'s vocab slice out of `n` ranks over vocab size `v`.
/// `base=v/n, rem=v%n`; the first `rem` ranks get one extra row. Slices are
/// contiguous, ascending in `r`, and exactly cover `[0, v)` with max skew 1
/// row (handles non-divisible `v`; no power-of-two requirement).
pub(crate) fn tp_vocab_shard_range(v: usize, r: usize, n: usize) -> (usize, usize) {
    let (base, rem) = (v / n, v % n);
    let lo = r * base + r.min(rem);
    (lo, base + usize::from(r < rem))
}

/// Merge per-rank local argmax results into the global argmax, reproducing
/// single-vector "strict `>` scan from index 0" tie-break semantics (lowest
/// global index among ties wins). `locals[r] = (local_max_val, local_idx)` for
/// rank `r` (as produced by an independent strict-`>` scan of that rank's
/// vocab slice). Ranks are iterated in order `0..n`; since slices are
/// contiguous and ascending in `r`, and both the per-rank scan and this merge
/// use strict `>`, the composed selection is exactly the single first-max scan
/// over the concatenated full vector.
pub(crate) fn tp_argmax_merge(locals: &[(f32, u32)], v: usize, n: usize) -> (u32, f32) {
    let (mut best_idx, mut best_val) = (0u32, f32::NEG_INFINITY);
    for r in 0..n {
        let (val, local_idx) = locals[r];
        if val > best_val {
            let (lo, _per) = tp_vocab_shard_range(v, r, n);
            best_val = val;
            best_idx = (lo as u32) + local_idx;
        }
    }
    (best_idx, best_val)
}

/// Merge `n` ranks' local top-k candidate lists (global-indexed already, i.e.
/// each rank added its own `tp_vocab_shard_range` offset before returning
/// here) into the global top-`k`, reproducing `forward_topk`'s keep-k
/// semantics: sort all candidates by `(value desc, global_idx asc)` and take
/// the first `k`. The `global_idx asc` tie-break is what makes this equivalent
/// to `forward_topk`'s "replace only if strictly greater" earliest-index-wins
/// rule (a value can never evict an equal value already resident, so among
/// duplicates the lowest index always survives to the cut).
pub(crate) fn tp_topk_merge(mut candidates: Vec<(u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    candidates.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0))
    });
    candidates.truncate(k);
    candidates
}

/// Strict `>`-from-index-0 argmax of a slice (the shared tie-break semantics
/// used by `forward_argmax`/`pp_step_qwen35`/the TP merge — lowest index among
/// ties wins). Returns (max_val, local_idx).
pub(crate) fn strict_argmax(slice: &[f32]) -> (f32, u32) {
    let (mut bi, mut bv) = (0u32, f32::NEG_INFINITY);
    for (i, &v) in slice.iter().enumerate() {
        if v > bv { bv = v; bi = i as u32; }
    }
    (bv, bi)
}

/// `forward_topk`'s exact keep-k algorithm (replace only if strictly greater)
/// run over `slice`, with `lo` added to every returned index so callers can
/// run it independently per rank and hand the (already-global-indexed) result
/// straight to `tp_topk_merge`.
pub(crate) fn strict_topk_local(slice: &[f32], lo: usize, k: usize) -> Vec<(u32, f32)> {
    let k = k.min(slice.len());
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    let mut min_in_top = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if top.len() < k {
            top.push((i as u32, v));
            if top.len() == k {
                min_in_top = top.iter().map(|t| t.1).fold(f32::INFINITY, f32::min);
            }
        } else if v > min_in_top {
            let (mi, _) = top.iter().enumerate()
                .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap()).unwrap();
            top[mi] = (i as u32, v);
            min_in_top = top.iter().map(|t| t.1).fold(f32::INFINITY, f32::min);
        }
    }
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    top.into_iter().map(|(i, v)| (i + lo as u32, v)).collect()
}

// Only called from the qwen35 pyseam (its `qwen35_matvec` body reads the
// qwen3_5 weight store) → gate qwen35.
#[cfg(feature = "qwen35")]
impl VulkanModel {
    /// This rank's local lm_head logit slice: matvec over the sharded (or, if
    /// `lm_shard` is `None`, full-replicated) lm_head rows. Returns
    /// `(local_logits, lo)` where `lo` is this rank's global vocab offset (0
    /// when unsharded). Model-family-agnostic — takes the resolved weight name
    /// and hidden size so `forward_tp_qwen35`/`forward_tp` (qwen3) share it.
    pub(crate) fn tp_lmhead_local(&mut self, lm_name: &str, normed: &[f32], h: usize, vocab: usize) -> (Vec<f32>, usize) {
        match self.lm_shard {
            Some(LmShard { lo, per, v }) => {
                debug_assert_eq!(v, vocab, "lm_shard vocab {v} != model vocab {vocab}");
                (self.qwen35_matvec(lm_name, normed, h, per), lo)
            }
            None => (self.qwen35_matvec(lm_name, normed, h, vocab), 0),
        }
    }
}

#[cfg(test)]
mod vocab_shard_tests {
    use super::*;

    #[test]
    fn shard_range_v10_n4() {
        // V=10, n=4 -> base=2 rem=2 -> ranks 0,1 get 3; ranks 2,3 get 2.
        assert_eq!(tp_vocab_shard_range(10, 0, 4), (0, 3));
        assert_eq!(tp_vocab_shard_range(10, 1, 4), (3, 3));
        assert_eq!(tp_vocab_shard_range(10, 2, 4), (6, 2));
        assert_eq!(tp_vocab_shard_range(10, 3, 4), (8, 2));
    }

    #[test]
    fn shard_range_covers_and_ascending_and_balanced() {
        for v in [1usize, 2, 3, 7, 10, 31, 248320] {
            for n in [1usize, 2, 3, 4, 5, 7] {
                if v < n { continue; }
                let mut covered = 0usize;
                let mut prev_hi = 0usize;
                let mut pers = Vec::with_capacity(n);
                for r in 0..n {
                    let (lo, per) = tp_vocab_shard_range(v, r, n);
                    assert_eq!(lo, prev_hi, "V={v} n={n} r={r}: gap/overlap");
                    prev_hi = lo + per;
                    covered += per;
                    pers.push(per);
                }
                assert_eq!(covered, v, "V={v} n={n}: coverage mismatch");
                assert_eq!(prev_hi, v, "V={v} n={n}: last range doesn't end at V");
                let (min_p, max_p) = (pers.iter().min().unwrap(), pers.iter().max().unwrap());
                assert!(max_p - min_p <= 1, "V={v} n={n}: skew {} > 1", max_p - min_p);
            }
        }
    }

    /// n=10 scale: both real checkpoint vocabs (27B/35B share 248320; the
    /// smaller qwen3-dense family uses 151936) at n=10 (10-node fabric, item 8
    /// in plan-tp-ep-levers.md §6b), plus n=8 (2-ranks-per-node PP/TP emulation)
    /// and a deliberately non-divisible n=3-with-remainder case at 10-scale
    /// vocab sizes. Same invariants as `shard_range_covers_and_ascending_and_balanced`:
    /// balanced split (skew <= 1), full coverage, no gap/overlap.
    #[test]
    fn shard_range_n10_scale_vocabs() {
        for v in [248320usize, 151936usize] {
            for n in [8usize, 10, 3] {
                let mut covered = 0usize;
                let mut prev_hi = 0usize;
                let mut pers = Vec::with_capacity(n);
                for r in 0..n {
                    let (lo, per) = tp_vocab_shard_range(v, r, n);
                    assert_eq!(lo, prev_hi, "V={v} n={n} r={r}: gap/overlap");
                    prev_hi = lo + per;
                    covered += per;
                    pers.push(per);
                }
                assert_eq!(covered, v, "V={v} n={n}: coverage mismatch");
                assert_eq!(prev_hi, v, "V={v} n={n}: last range doesn't end at V");
                let (min_p, max_p) = (pers.iter().min().unwrap(), pers.iter().max().unwrap());
                assert!(max_p - min_p <= 1, "V={v} n={n}: skew {} > 1", max_p - min_p);
                // Remainder distribution: exactly `v % n` ranks (ranks 0..rem)
                // get the +1 row; the rest get the floor share.
                let (base, rem) = (v / n, v % n);
                for r in 0..n {
                    let (_, per) = tp_vocab_shard_range(v, r, n);
                    let expected = base + usize::from(r < rem);
                    assert_eq!(per, expected, "V={v} n={n} r={r}: remainder distribution wrong");
                }
            }
        }
    }

    #[test]
    fn shard_range_27b_exact_divide() {
        // 27B: 248320 / 4 -> per=62080 exactly, rem=0.
        for r in 0..4 {
            assert_eq!(tp_vocab_shard_range(248320, r, 4), (r * 62080, 62080));
        }
    }

    /// Slice-vs-full: concatenating each rank's CPU matvec over its lm_head
    /// row-slice reproduces the full-table matvec bit-identically (same
    /// f32 ops, no reordering within a row — only which rows are computed).
    #[test]
    fn slice_matvec_matches_full_bit_identical() {
        let mut state = 0x1234_5678_u64;
        let mut rng = move || {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let (v, h) = (37usize, 16usize); // non-divisible by n below
        let table: Vec<f32> = (0..v * h).map(|_| rng()).collect();
        let x: Vec<f32> = (0..h).map(|_| rng()).collect();
        let full = crate::model::cpu_matmul(&x, &table, 1, h, v);
        for n in [1usize, 2, 3, 4, 5] {
            let mut assembled = Vec::with_capacity(v);
            for r in 0..n {
                let (lo, per) = tp_vocab_shard_range(v, r, n);
                if per == 0 { continue; }
                let slice = &table[lo * h..(lo + per) * h];
                let partial = crate::model::cpu_matmul(&x, slice, 1, h, per);
                assembled.extend_from_slice(&partial);
            }
            assert_eq!(assembled.len(), full.len());
            for i in 0..full.len() {
                assert_eq!(assembled[i].to_bits(), full[i].to_bits(),
                    "n={n} i={i}: {} != {} (not bit-identical)", assembled[i], full[i]);
            }
        }
    }

    /// Reference single-vector first-max scan (strict `>` from index 0) —
    /// the ground truth `tp_argmax_merge` must reproduce given per-rank locals.
    fn full_argmax(v: &[f32]) -> (u32, f32) {
        let (mut bi, mut bv) = (0u32, f32::NEG_INFINITY);
        for (i, &val) in v.iter().enumerate() {
            if val > bv { bv = val; bi = i as u32; }
        }
        (bi, bv)
    }

    fn local_argmax(slice: &[f32]) -> (f32, u32) {
        let (mut bi, mut bv) = (0u32, f32::NEG_INFINITY);
        for (i, &val) in slice.iter().enumerate() {
            if val > bv { bv = val; bi = i as u32; }
        }
        (bv, bi)
    }

    /// Engineered ties: equal global max split across ranks (0 & 2), ties
    /// inside a single slice, and a max sitting exactly at a slice boundary
    /// (lo and lo+per-1). Merge must match the single-vector first-max scan
    /// in every case (lowest global index among ties wins).
    #[test]
    fn argmax_merge_determinism_engineered_ties() {
        let n = 4usize;
        let cases: Vec<Vec<f32>> = vec![
            // tie across ranks 0 and 2 (v=12 -> per=3 each)
            vec![5.0, 1.0, 2.0,   0.0, 0.0, 0.0,   5.0, -1.0, -2.0,   4.9, 4.9, 4.9],
            // tie inside one slice (rank 1 has the max, repeated within it)
            vec![0.0, -1.0, -2.0,  9.0, 9.0, 1.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
            // max exactly at a slice boundary (lo of rank 2, and lo+per-1 of rank 3)
            vec![0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  7.0, 0.0, 0.0,  0.0, 0.0, 7.0],
        ];
        for full in cases {
            let v = full.len();
            let expected = full_argmax(&full);
            let mut locals = Vec::with_capacity(n);
            for r in 0..n {
                let (lo, per) = tp_vocab_shard_range(v, r, n);
                locals.push(local_argmax(&full[lo..lo + per]));
            }
            let got = tp_argmax_merge(&locals, v, n);
            assert_eq!(got, expected, "case {full:?}: merge {got:?} != full {expected:?}");
        }
    }

    /// Same engineered-tie determinism check as
    /// `argmax_merge_determinism_engineered_ties`, but across a full n=10
    /// rank fabric (item 8: 10-node divisibility/emulation prep) with ties
    /// deliberately placed AT rank-slice boundaries (v=100 -> per=10 each, so
    /// boundary indices are 9/10, 19/20, ... 89/90) to exercise the
    /// rank-ascending strict-`>` merge across every adjacent-slice seam.
    #[test]
    fn argmax_merge_determinism_ties_n10() {
        let n = 10usize;
        let v = 100usize; // per=10 exactly, boundaries at multiples of 10
        let mut cases: Vec<Vec<f32>> = Vec::new();

        // Case 1: tie at every rank boundary (last elt of rank r == first elt
        // of rank r+1 == a shared global max), rest filled with a lower value.
        {
            let mut full = vec![1.0f32; v];
            full[9] = 9.0; // last of rank 0
            full[10] = 9.0; // first of rank 1 (tie with the above)
            cases.push(full);
        }
        // Case 2: tie between the FIRST and LAST rank slices (rank 0 lo=0,
        // rank 9 lo=90..99) - the widest possible span for the merge to get
        // "lowest global index wins" right.
        {
            let mut full = vec![0.0f32; v];
            full[0] = 7.0;
            full[99] = 7.0;
            cases.push(full);
        }
        // Case 3: tie inside a single interior slice (rank 5, indices 50-59)
        // plus a decoy equal value in an earlier untied rank.
        {
            let mut full = vec![-1.0f32; v];
            full[52] = 4.0;
            full[57] = 4.0;
            full[3] = 3.9; // close but strictly less -> must not win
            cases.push(full);
        }
        // Case 4: every rank's slice max is identical (global tie across all
        // 10 ranks) -> rank 0 (lowest global index, position 0) must win.
        {
            let mut full = vec![0.0f32; v];
            for r in 0..n {
                let (lo, _) = tp_vocab_shard_range(v, r, n);
                full[lo] = 2.0;
            }
            cases.push(full);
        }

        for full in cases {
            let expected = full_argmax(&full);
            let mut locals = Vec::with_capacity(n);
            for r in 0..n {
                let (lo, per) = tp_vocab_shard_range(v, r, n);
                locals.push(local_argmax(&full[lo..lo + per]));
            }
            let got = tp_argmax_merge(&locals, v, n);
            assert_eq!(got, expected, "n=10 case {full:?}: merge {got:?} != full {expected:?}");
        }
    }

    /// Property-style seeded fuzz: random vectors with deliberately clamped
    /// value range (few distinct levels) to force ties, over several (v, n).
    #[test]
    fn argmax_merge_property_seeded() {
        let mut state = 0xDEAD_BEEF_u64;
        let mut rng = move || {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            (state >> 48) as u32 % 5 // 5 distinct levels -> frequent ties
        };
        for &(v, n) in &[(20usize, 4usize), (21, 4), (17, 3), (50, 6)] {
            for _trial in 0..20 {
                let full: Vec<f32> = (0..v).map(|_| rng() as f32).collect();
                let expected = full_argmax(&full);
                let mut locals = Vec::with_capacity(n);
                for r in 0..n {
                    let (lo, per) = tp_vocab_shard_range(v, r, n);
                    locals.push(local_argmax(&full[lo..lo + per]));
                }
                let got = tp_argmax_merge(&locals, v, n);
                assert_eq!(got, expected, "v={v} n={n} full={full:?}");
            }
        }
    }

    /// Reference top-k matching `forward_topk`'s exact algorithm (keep-k,
    /// replace only if strictly greater) run over the FULL vector.
    fn full_topk_reference(vals: &[f32], k: usize) -> Vec<(u32, f32)> {
        let k = k.min(vals.len());
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        let mut min_in_top = f32::NEG_INFINITY;
        for (i, &v) in vals.iter().enumerate() {
            if top.len() < k {
                top.push((i as u32, v));
                if top.len() == k {
                    min_in_top = top.iter().map(|t| t.1).fold(f32::INFINITY, f32::min);
                }
            } else if v > min_in_top {
                let (mi, _) = top.iter().enumerate()
                    .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap()).unwrap();
                top[mi] = (i as u32, v);
                min_in_top = top.iter().map(|t| t.1).fold(f32::INFINITY, f32::min);
            }
        }
        top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        top
    }

    fn local_topk(slice_vals: &[f32], lo: usize, k: usize) -> Vec<(u32, f32)> {
        // Per-rank local top-k using the SAME algorithm, indices offset to global.
        let local = full_topk_reference(slice_vals, k);
        local.into_iter().map(|(i, v)| (i + lo as u32, v)).collect()
    }

    /// Topk merge parity vs `forward_topk`'s semantics on the full vector,
    /// including duplicated values across slices (forces the idx-asc
    /// tie-break to matter).
    #[test]
    fn topk_merge_matches_forward_topk_reference() {
        let cases: Vec<(Vec<f32>, usize, usize)> = vec![
            // duplicated max across two slices (n=4, v=12 -> per=3)
            (vec![5.0, 1.0, 2.0, 0.0, 0.0, 0.0, 5.0, -1.0, -2.0, 4.9, 4.9, 4.9], 4, 3),
            // duplicated values throughout, k larger
            (vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 4, 5),
            // ties right at the cut boundary
            (vec![9.0, 8.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0], 4, 4),
        ];
        for (full, n, k) in cases {
            let v = full.len();
            let expected = full_topk_reference(&full, k);
            let mut candidates = Vec::new();
            for r in 0..n {
                let (lo, per) = tp_vocab_shard_range(v, r, n);
                if per == 0 { continue; }
                candidates.extend(local_topk(&full[lo..lo + per], lo, k));
            }
            let got = tp_topk_merge(candidates, k);
            // Compare as sets of (idx, bits) since order within equal values is
            // deterministic (idx asc) in both, but assert value-for-value too.
            assert_eq!(got.len(), expected.len(), "v={v} n={n} k={k}");
            for (g, e) in got.iter().zip(expected.iter()) {
                assert_eq!(g.0, e.0, "v={v} n={n} k={k}: idx mismatch {got:?} != {expected:?}");
                assert_eq!(g.1.to_bits(), e.1.to_bits(), "v={v} n={n} k={k}: val mismatch");
            }
        }
    }

    /// Topk merge parity at n=10 (item 8's 10-node fabric): a non-divisible
    /// vocab (v=103, n=10 -> rem=3, so 3 ranks carry an extra row) and a
    /// case where the top-k values cluster right at a rank-slice boundary
    /// (idx 29/30, the rank-2/rank-3 seam), both with strictly distinct
    /// values so `full_topk_reference`'s slot-overwrite order is unambiguous
    /// (a genuine tie where BOTH duplicates survive eviction is excluded —
    /// `forward_topk`'s keep-k slot order for that case is itself
    /// position-dependent, not a fixed idx-asc rule, so it isn't a valid
    /// cross-check target; the smaller-n test above already covers the safe
    /// duplicated-value patterns).
    #[test]
    fn topk_merge_matches_forward_topk_reference_n10() {
        let n = 10usize;
        let cases: Vec<(Vec<f32>, usize)> = vec![
            {
                // v=103, n=10 -> rem=3. Strictly descending -> no ties at all,
                // so this pins down remainder-shard-boundary handling cleanly.
                let full: Vec<f32> = (0..103).map(|i| (103 - i) as f32).collect();
                (full, 8)
            },
            {
                // v=100 (per=10 exactly): the top-4 values sit right at the
                // rank-2/rank-3 slice boundary (idx 28,29,30,31), all
                // strictly distinct, forcing the merge to assemble the
                // global top-k from candidates straddling two adjacent ranks.
                let mut full: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
                full[28] = 20.0;
                full[29] = 21.0;
                full[30] = 22.0;
                full[31] = 23.0;
                (full, 4)
            },
        ];
        for (full, k) in cases {
            let v = full.len();
            let expected = full_topk_reference(&full, k);
            let mut candidates = Vec::new();
            for r in 0..n {
                let (lo, per) = tp_vocab_shard_range(v, r, n);
                if per == 0 { continue; }
                candidates.extend(local_topk(&full[lo..lo + per], lo, k));
            }
            let got = tp_topk_merge(candidates, k);
            assert_eq!(got.len(), expected.len(), "n=10 v={v} k={k}");
            for (g, e) in got.iter().zip(expected.iter()) {
                assert_eq!(g.0, e.0, "n=10 v={v} k={k}: idx mismatch {got:?} != {expected:?}");
                assert_eq!(g.1.to_bits(), e.1.to_bits(), "n=10 v={v} k={k}: val mismatch");
            }
        }
    }
}

/// INC-1b gate: CPU reassembly bit-exact check of the gemma-31B head-aware
/// TP-4 shard (`gemma_fp8_tp_shard`/`gemma_nvfp4_tp_shard`) against the full
/// unsharded dequant, on a representative tensor set from the REAL 24.7GB
/// `gemma-4-31B-it-NVFP4` checkpoint. This is the whole point of INC-1b: prove
/// the shard functions are a faithful partition of the real weight (every
/// element lands in exactly one rank, reassembly reproduces the original
/// exactly) — NOT merely close, and NOT exercising GPU/multi-rank execution
/// (that's cluster-deferred; see the module doc comment above and
/// `GEMMA31B_SPEC_PLAN.md`).
#[cfg(all(test, feature = "gemma"))]
mod gemma_shard_tests {
    use super::*;
    use crate::model::{
        Gemma4Config, ProjWeight, ProjResult, load_gemma_nvfp4_weights,
        dequantize_fp8, NVFP4_E2M1_LUT,
    };
    use crate::push_constants::nvfp4_fold_scales;

    /// Reconstruct one row-major `[out, in]` NVFP4 tensor from its packed
    /// nibbles + ALREADY-FOLDED per-(row,group) scale (`e4m3(weight_scale) *
    /// weight_scale_2`, i.e. `nvfp4_fold_scales`'s output — the exact form
    /// `gemma_nvfp4_tp_shard` shards and the real GPU path consumes). Using
    /// this single-multiply form (instead of `model::dequantize_nvfp4`'s raw
    /// `wscale`-bytes-plus-separate-`global` two-multiply form) for BOTH the
    /// full tensor and every shard is what makes the reassembly bit-exact:
    /// `nvfp4_fold_scales_reconstructs_dequant` (push_constants.rs) already
    /// shows the two multiply orderings themselves differ by ~1 ULP under
    /// float reassociation, which is irrelevant to what this gate is
    /// checking (the shard's partition correctness, not the fold's own
    /// numerics) — comparing golden-vs-shard through the SAME formula is what
    /// isolates that.
    fn dequant_nvfp4_folded(packed: &[u8], folded: &[f32], out_f: usize, in_f: usize, gs: usize) -> Vec<f32> {
        let groups = in_f / gs;
        let bytes_per_row = in_f / 2;
        let mut w = vec![0.0f32; out_f * in_f];
        for o in 0..out_f {
            let brow = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
            let frow = &folded[o * groups..(o + 1) * groups];
            for i in 0..in_f {
                let byte = brow[i / 2];
                let nib = if i % 2 == 0 { byte & 0xF } else { byte >> 4 } as usize;
                w[o * in_f + i] = NVFP4_E2M1_LUT[nib] * frow[i / gs];
            }
        }
        w
    }

    /// Reassemble NR column-parallel (output-row-sharded) dequantized
    /// f32 shards, in ascending rank order, into the full `[out, in]` tensor.
    /// Column-shard splits keep whole rows contiguous per rank in
    /// rank-ascending order covering `[0, out)` exactly (divisible case), so
    /// this is a straight concat.
    fn reassemble_col(shards: &[Vec<f32>]) -> Vec<f32> {
        shards.concat()
    }

    /// Reassemble NR row-parallel (input-col-sharded) dequantized f32 shards
    /// into the full `[out, in]` tensor: for every output row, concatenate
    /// this row's slice from each rank in ascending order.
    fn reassemble_row(shards: &[Vec<f32>], out_features: usize) -> Vec<f32> {
        let per: Vec<usize> = shards.iter().map(|s| s.len() / out_features).collect();
        let total_in: usize = per.iter().sum();
        let mut out = vec![0.0f32; out_features * total_in];
        for o in 0..out_features {
            let mut col_off = 0usize;
            for (r, s) in shards.iter().enumerate() {
                let pin = per[r];
                out[o * total_in + col_off..o * total_in + col_off + pin]
                    .copy_from_slice(&s[o * pin..(o + 1) * pin]);
                col_off += pin;
            }
        }
        out
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// Ignored by default (needs the real 24.7GB checkpoint + libvulkan on
    /// the DYLD path for `cargo test` to even link on Mac). Run with:
    ///   DYLD_LIBRARY_PATH=/opt/homebrew/lib:/opt/homebrew/pkgs/python-3.10.16-h870587a_1_cpython/lib \
    ///   PYTHONHOME=/opt/homebrew/pkgs/python-3.10.16-h870587a_1_cpython \
    ///   VLLM_TEST_GEMMA31B_DIR=~/repos/OminiX-MLX/models/gemma-4-31B-it-NVFP4 \
    ///     cargo test --lib gemma31b_tp4_shard_reassembly_bitexact -- --ignored
    #[test]
    #[ignore]
    fn gemma31b_tp4_shard_reassembly_bitexact() {
        let dir = match std::env::var("VLLM_TEST_GEMMA31B_DIR") { Ok(d) => d, Err(_) => return };
        let dir = std::path::Path::new(&dir);
        let cfg = Gemma4Config::g31b();
        const NR: usize = 4;

        // Capture the packed tensors exactly as the loader offers them
        // (pre-dequant), for layers 0 (sliding) and 5 (global; period-6 ->
        // full-attention at idx%6==5).
        type Fp8Cap = (Vec<u8>, Vec<f32>, usize, usize); // weight, scale, out, in
        type Nvfp4Cap = (Vec<u8>, Vec<u8>, f32, usize, usize, usize); // packed, wscale, global, out, in, gs
        let mut q0_cap: Option<Fp8Cap> = None;   // layer0 (sliding) q_proj
        let mut q5_cap: Option<Fp8Cap> = None;   // layer5 (global)  q_proj
        let mut o0_cap: Option<Fp8Cap> = None;   // layer0 o_proj (row-parallel FP8)
        let mut gate0_cap: Option<Nvfp4Cap> = None; // layer0 mlp.gate_proj (col-parallel NVFP4)
        let mut down0_cap: Option<Nvfp4Cap> = None; // layer0 mlp.down_proj (row-parallel NVFP4)

        let _ = load_gemma_nvfp4_weights(dir, 0, 6, |name, w| match w {
            ProjWeight::Fp8(fp) if name.ends_with("layers.0.self_attn.q_proj.weight") => {
                q0_cap = Some((fp.weight.to_vec(), fp.scale.clone(), fp.out_features, fp.in_features));
                ProjResult::Consumed
            }
            ProjWeight::Fp8(fp) if name.ends_with("layers.5.self_attn.q_proj.weight") => {
                q5_cap = Some((fp.weight.to_vec(), fp.scale.clone(), fp.out_features, fp.in_features));
                ProjResult::Consumed
            }
            ProjWeight::Fp8(fp) if name.ends_with("layers.0.self_attn.o_proj.weight") => {
                o0_cap = Some((fp.weight.to_vec(), fp.scale.clone(), fp.out_features, fp.in_features));
                ProjResult::Consumed
            }
            ProjWeight::Nvfp4(nv) if name.ends_with("layers.0.mlp.gate_proj.weight") => {
                gate0_cap = Some((nv.packed.to_vec(), nv.wscale.to_vec(), nv.global,
                                  nv.out_features, nv.in_features, nv.group_size));
                ProjResult::Consumed
            }
            ProjWeight::Nvfp4(nv) if name.ends_with("layers.0.mlp.down_proj.weight") => {
                down0_cap = Some((nv.packed.to_vec(), nv.wscale.to_vec(), nv.global,
                                  nv.out_features, nv.in_features, nv.group_size));
                ProjResult::Consumed
            }
            ProjWeight::Nvfp4(_) | ProjWeight::Fp8(_) => ProjResult::Consumed, // drop other packed
            ProjWeight::F32(v) => ProjResult::KeepF32(v),
            ProjWeight::Mlx4(_) => ProjResult::Consumed, // not present in this checkpoint
        }).expect("load gemma31b NVFP4+FP8 checkpoint (layers 0..6)");

        // ── q_proj, layer 0 (SLIDING: head_dim 256, col-parallel by q-head) ──
        {
            let (weight, scale, out_f, in_f) = q0_cap.expect("layer0 self_attn.q_proj offered as packed FP8");
            assert_eq!((out_f, in_f), (32 * 256, 5376), "layer0 q_proj shape (sliding head_dim=256)");
            let name = "model.layers.0.self_attn.q_proj.weight";
            let golden = dequantize_fp8(&weight, &scale, out_f, in_f);
            let shards: Vec<Vec<f32>> = (0..NR).map(|r| {
                let (w, s, of, inf) = gemma_fp8_tp_shard(name, &weight, &scale, out_f, in_f, &cfg, r, NR);
                assert_eq!(of, 8 * 256, "layer0 q_proj rank {r}: 8 heads/rank @ TP-4");
                dequantize_fp8(&w, &s, of, inf)
            }).collect();
            let reassembled = reassemble_col(&shards);
            let md = maxdiff(&reassembled, &golden);
            eprintln!("gemma31b TP-4 shard reassembly: q_proj layer0 (sliding) maxdiff={md:e}");
            assert_eq!(md, 0.0, "q_proj layer0 (sliding) reassembly not bit-exact vs full dequant");
        }

        // ── q_proj, layer 5 (GLOBAL: head_dim 512, col-parallel by q-head) ──
        {
            let (weight, scale, out_f, in_f) = q5_cap.expect("layer5 self_attn.q_proj offered as packed FP8");
            assert_eq!((out_f, in_f), (32 * 512, 5376), "layer5 q_proj shape (global head_dim=512)");
            assert!(cfg.is_full_attention(5), "layer 5 must be the global/full-attention type this case exercises");
            let name = "model.layers.5.self_attn.q_proj.weight";
            let golden = dequantize_fp8(&weight, &scale, out_f, in_f);
            let shards: Vec<Vec<f32>> = (0..NR).map(|r| {
                let (w, s, of, inf) = gemma_fp8_tp_shard(name, &weight, &scale, out_f, in_f, &cfg, r, NR);
                assert_eq!(of, 8 * 512, "layer5 q_proj rank {r}: 8 heads/rank @ TP-4 (global head_dim=512)");
                dequantize_fp8(&w, &s, of, inf)
            }).collect();
            let reassembled = reassemble_col(&shards);
            let md = maxdiff(&reassembled, &golden);
            eprintln!("gemma31b TP-4 shard reassembly: q_proj layer5 (global) maxdiff={md:e}");
            assert_eq!(md, 0.0, "q_proj layer5 (global) reassembly not bit-exact vs full dequant");
        }

        // ── o_proj, layer 0 (FP8, ROW-parallel by the same q-head split) ────
        {
            let (weight, scale, out_f, in_f) = o0_cap.expect("layer0 self_attn.o_proj offered as packed FP8");
            assert_eq!((out_f, in_f), (5376, 32 * 256), "layer0 o_proj shape");
            let name = "model.layers.0.self_attn.o_proj.weight";
            let golden = dequantize_fp8(&weight, &scale, out_f, in_f);
            let shards: Vec<Vec<f32>> = (0..NR).map(|r| {
                let (w, s, of, inf) = gemma_fp8_tp_shard(name, &weight, &scale, out_f, in_f, &cfg, r, NR);
                assert_eq!(of, 5376, "o_proj rank {r}: out_features unchanged (row-parallel)");
                assert_eq!(inf, 8 * 256, "o_proj rank {r}: 8 heads/rank input slice @ TP-4");
                dequantize_fp8(&w, &s, of, inf)
            }).collect();
            let reassembled = reassemble_row(&shards, out_f);
            let md = maxdiff(&reassembled, &golden);
            eprintln!("gemma31b TP-4 shard reassembly: o_proj layer0 maxdiff={md:e}");
            assert_eq!(md, 0.0, "o_proj layer0 reassembly not bit-exact vs full dequant");
        }

        // ── mlp.gate_proj, layer 0 (NVFP4, COLUMN-parallel over intermediate) ─
        {
            let (packed, wscale, global, out_f, in_f, gs) =
                gate0_cap.expect("layer0 mlp.gate_proj offered as packed NVFP4");
            assert_eq!((out_f, in_f, gs), (21504, 5376, 16), "layer0 gate_proj shape/group_size");
            let name = "model.layers.0.mlp.gate_proj.weight";
            let folded = nvfp4_fold_scales(&wscale, global);
            let golden = dequant_nvfp4_folded(&packed, &folded, out_f, in_f, gs);
            let shards: Vec<Vec<f32>> = (0..NR).map(|r| {
                let (p, f, of, inf) = gemma_nvfp4_tp_shard(name, &packed, &folded, out_f, in_f, gs, r, NR);
                assert_eq!(of, 21504 / NR, "gate_proj rank {r}: intermediate/TP-4 output rows");
                dequant_nvfp4_folded(&p, &f, of, inf, gs)
            }).collect();
            let reassembled = reassemble_col(&shards);
            let md = maxdiff(&reassembled, &golden);
            eprintln!("gemma31b TP-4 shard reassembly: mlp.gate_proj layer0 (NVFP4) maxdiff={md:e}");
            assert_eq!(md, 0.0, "mlp.gate_proj layer0 reassembly not bit-exact vs full dequant");
        }

        // ── mlp.down_proj, layer 0 (NVFP4, ROW-parallel over intermediate) ──
        {
            let (packed, wscale, global, out_f, in_f, gs) =
                down0_cap.expect("layer0 mlp.down_proj offered as packed NVFP4");
            assert_eq!((out_f, in_f, gs), (5376, 21504, 16), "layer0 down_proj shape/group_size");
            let name = "model.layers.0.mlp.down_proj.weight";
            let folded = nvfp4_fold_scales(&wscale, global);
            let golden = dequant_nvfp4_folded(&packed, &folded, out_f, in_f, gs);
            let shards: Vec<Vec<f32>> = (0..NR).map(|r| {
                let (p, f, of, inf) = gemma_nvfp4_tp_shard(name, &packed, &folded, out_f, in_f, gs, r, NR);
                assert_eq!(of, 5376, "down_proj rank {r}: out_features unchanged (row-parallel)");
                assert_eq!(inf, 21504 / NR, "down_proj rank {r}: intermediate/TP-4 input cols (336 groups)");
                dequant_nvfp4_folded(&p, &f, of, inf, gs)
            }).collect();
            let reassembled = reassemble_row(&shards, out_f);
            let md = maxdiff(&reassembled, &golden);
            eprintln!("gemma31b TP-4 shard reassembly: mlp.down_proj layer0 (NVFP4) maxdiff={md:e}");
            assert_eq!(md, 0.0, "mlp.down_proj layer0 reassembly not bit-exact vs full dequant");
        }
    }
}

/// perf/quant-batched-cols: the algorithmic invariant the TP batched-verify
/// mixers (`qwen35_*_tp_batched`) and the `mul_mat_vec_q8_0_cols` shader both
/// rely on — batching T activation columns against a q8_0 weight streamed ONCE
/// yields, for each column, EXACTLY the single-token matvec of that column.
/// This is what keeps the batched verify argmax-exact vs the T serial per-token
/// matvecs it replaces (while streaming the weight 1× instead of T×). The GPU
/// bit-exactness of the cols shader vs the single-column matvec shader (a
/// reduction-order question) is separately cluster-validated; here we prove the
/// SCALAR-math equivalence in exact f32 using the real q8_0 quant/dequant.
#[cfg(test)]
mod cols_batched_layout_tests {
    use crate::model;

    /// Scalar reference of `shaders/mul_mat_vec_q8_0_cols.comp`: dequantize each
    /// weight row ONCE, MAC it against all T activation columns → [T,n]
    /// row-major (dst[c*n + r]). The weight is read once; columns are the inner
    /// reuse loop — exactly the shader's streaming.
    fn cols_ref(deq_w: &[f32], x: &[f32], t: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; t * n];
        for r in 0..n {
            let wrow = &deq_w[r * k..(r + 1) * k];
            for c in 0..t {
                let xc = &x[c * k..(c + 1) * k];
                let mut acc = 0.0f32;
                for j in 0..k { acc += wrow[j] * xc[j]; }
                out[c * n + r] = acc;
            }
        }
        out
    }

    /// Single-column matvec (the T-serial per-token path this replaces):
    /// dequant row, dot with the one activation column. Identical scalar
    /// accumulation order (sum over k in index order) as `cols_ref`.
    fn serial_ref(deq_w: &[f32], x1: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for r in 0..n {
            let wrow = &deq_w[r * k..(r + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k { acc += wrow[j] * x1[j]; }
            out[r] = acc;
        }
        out
    }

    #[test]
    fn q8_0_cols_row_equals_per_token_matvec() {
        let mut s = 0x00C0FFEE_u64;
        let mut rng = move || {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let k = 160usize;   // multiple of 32 (q8_0 block size)
        let n = 24usize;
        let w_raw: Vec<f32> = (0..n * k).map(|_| rng()).collect();
        // The EXACT bytes/values the cols shader reads (model::quantize_q8_0's
        // ggml block layout == the shader's block_q8_0).
        let packed = model::quantize_q8_0(&w_raw);
        let deq = model::dequant_q8_0_to_f32(&packed);
        for t in [2usize, 3, 4, 5, 6, 7, 8] {
            let x: Vec<f32> = (0..t * k).map(|_| rng()).collect();
            let batched = cols_ref(&deq, &x, t, k, n);
            for c in 0..t {
                let serial = serial_ref(&deq, &x[c * k..(c + 1) * k], k, n);
                for r in 0..n {
                    assert_eq!(
                        batched[c * n + r].to_bits(), serial[r].to_bits(),
                        "t={t} c={c} r={r}: cols row != per-token matvec (layout/amortization bug)");
                }
            }
        }
    }
}
