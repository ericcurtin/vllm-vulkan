//! DeepSeek-V4-Flash — full per-layer decoder forward + head (M1 item 3).
//!
//! Composes the validated Rust primitives into the complete 43-layer forward,
//! mirroring `transformers-5.8.1` `DeepseekV4DecoderLayer.forward` +
//! `DeepseekV4Model.forward` + `lm_head` (and `golden.py::run_forward` /
//! `build_layer`) op-for-op:
//!   attn site:  attn_hc [hc_block] → input_layernorm → attention → hc_residual_mix
//!   ffn  site:  ffn_hc  [hc_block] → post_attention_layernorm → MoE → hc_residual_mix
//!   attention = [`crate::dsv4::mla_core_ext`], with CSA/HCA compressed_kv+block_bias
//!               from [`crate::dsv4_dsa`]; MoE = [`crate::dsv4_moe`].
//!   head:       hc_head (HyperHead) → model.norm → lm_head.
//!
//! Weights are supplied as already-dequantized f32 by a [`Dsv4Src`] (checkpoint
//! naming per `DSV4_NAME_MAP_DOC` / `golden.py`), so the SAME forward serves the
//! tiny-model composition self-test (in-memory dict) and the real-checkpoint M1
//! gate (streaming per-layer dequant). The stable (max-factored/f64) hc_block is
//! an identity on finite input, so this matches the reference argmax.

use crate::dsv4::{
    hc_block, hc_residual_mix, mla_core_ext, rmsnorm_rows, unweighted_rmsnorm_rows, Dsv4Config,
    LayerType, MlpType,
};
use crate::dsv4_dsa::{csa_compressor, hca_compressor, rope_cos_sin, IndexerWeights};
use crate::dsv4_moe::{hash_router, moe_forward, topk_router};
use crate::model::cpu_matmul;

/// Supplies already-dequantized f32 weights by checkpoint name (see
/// `golden.py::RealSource` / `DictSource`). All returned tensors are row-major.
pub trait Dsv4Src {
    /// A quantized/plain linear weight `[out_f, in_f]` (name has no `.weight`).
    fn linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32>;
    /// A plain dense tensor (norms, hc fn/base/scale, sinks, position_bias, corr_bias).
    fn dense(&self, name: &str) -> Vec<f32>;
    /// Expert `e` row-slice of a 3D `[E, out_f, in_f]` tensor.
    fn expert(&self, name: &str, e: usize, out_f: usize, in_f: usize) -> Vec<f32>;
    /// The hash-router `tid2eid` table `[vocab, top_k]` as i64.
    fn dense_i64(&self, name: &str) -> Vec<i64>;
    /// Embedding rows for `ids` → `[S, H]`. Default: full-table dequant + gather
    /// (real streaming sources override to read only the needed rows).
    fn embed_rows(&self, ids: &[u32], vocab: usize, h: usize) -> Vec<f32> {
        let w = self.linear("model.embed_tokens", vocab, h);
        let mut out = vec![0.0f32; ids.len() * h];
        for (i, &t) in ids.iter().enumerate() {
            out[i * h..(i + 1) * h].copy_from_slice(&w[t as usize * h..(t as usize + 1) * h]);
        }
        out
    }
}

/// Sliding-window causal additive mask `[S,S]` (0 visible / -inf masked): query `i`
/// sees key `j` iff `j <= i && (i-j) < sliding_window`.
fn sliding_mask(s: usize, sliding_window: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if j > i || (i - j) >= sliding_window {
                m[i * s + j] = f32::NEG_INFINITY;
            }
        }
    }
    m
}

/// One attention sub-layer (post input_layernorm) → `[S,H]`. `x` = `[S,H]`.
#[allow(clippy::too_many_arguments)]
fn attention_layer<S: Dsv4Src>(
    cfg: &Dsv4Config,
    li: usize,
    lt: LayerType,
    x: &[f32],
    seq: usize,
    positions: &[usize],
    mask: &[f32],
    inv_main: &[f32],
    inv_comp: &[f32],
    src: &S,
) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let nh = cfg.mla.num_attention_heads;
    let hd = cfg.mla.head_dim;
    let ql = cfg.mla.q_lora_rank;
    let olr = cfg.mla.o_lora_rank;
    let g = cfg.mla.o_groups;
    let eps = cfg.mla.rms_norm_eps;
    let p = format!("model.layers.{li}");

    let (inv, scaling) = if lt == LayerType::Sliding {
        (inv_main, 1.0f32)
    } else {
        (inv_comp, cfg.rope.compress_scaling)
    };
    let (cos, sin) = rope_cos_sin(positions, inv, scaling);

    let w_q_a = src.linear(&format!("{p}.attn.wq_a"), ql, h);
    let w_q_a_norm = src.dense(&format!("{p}.attn.q_norm.weight"));
    let w_q_b = src.linear(&format!("{p}.attn.wq_b"), nh * hd, ql);
    let w_kv = src.linear(&format!("{p}.attn.wkv"), hd, h);
    let w_kv_norm = src.dense(&format!("{p}.attn.kv_norm.weight"));
    let w_o_a = src.linear(&format!("{p}.attn.wo_a"), g * olr, nh * hd / g);
    let w_o_b = src.linear(&format!("{p}.attn.wo_b"), h, g * olr);
    let sinks = src.dense(&format!("{p}.attn.attn_sink"));

    // CSA/HCA compressed KV + additive block_bias (empty for sliding / degenerate).
    let (compressed_kv, block_bias): (Vec<f32>, Vec<f32>) = match lt {
        LayerType::Sliding => (Vec::new(), Vec::new()),
        LayerType::HeavilyCompressed => {
            let m = cfg.compress_rate_hca;
            let (ckv, vis) = hca_compressor(
                x, seq, h, hd, m, positions,
                &src.linear(&format!("{p}.attn.compressor.wkv"), hd, h),
                &src.linear(&format!("{p}.attn.compressor.wgate"), hd, h),
                &src.dense(&format!("{p}.attn.compressor.ape")),
                &src.dense(&format!("{p}.attn.compressor.norm.weight")),
                eps, inv, scaling,
            );
            (ckv, vis_to_additive(&vis))
        }
        LayerType::CompressedSparse => {
            let m = cfg.compress_rate_csa;
            let ihd = cfg.index_head_dim;
            let inh = cfg.index_n_heads;
            // q_residual = q_a_norm(q_a_proj(x)), the compressor's query input.
            let q_a = rmsnorm_rows(&cpu_matmul(x, &w_q_a, seq, h, ql), &w_q_a_norm, seq, ql, eps);
            let ix_kv = src.linear(&format!("{p}.attn.indexer.compressor.wkv"), 2 * ihd, h);
            let ix_gate = src.linear(&format!("{p}.attn.indexer.compressor.wgate"), 2 * ihd, h);
            let ix_pb = src.dense(&format!("{p}.attn.indexer.compressor.ape"));
            let ix_norm = src.dense(&format!("{p}.attn.indexer.compressor.norm.weight"));
            let ix_qb = src.linear(&format!("{p}.attn.indexer.wq_b"), inh * ihd, ql);
            let ix_wp = src.linear(&format!("{p}.attn.indexer.weights_proj"), inh, h);
            let ix = IndexerWeights {
                kv_proj: &ix_kv, gate_proj: &ix_gate, position_bias: &ix_pb, kv_norm: &ix_norm,
                q_b_proj: &ix_qb, weights_proj: &ix_wp,
            };
            let (ckv, vis) = csa_compressor(
                x, &q_a, seq, h, hd, m, positions,
                &src.linear(&format!("{p}.attn.compressor.wkv"), 2 * hd, h),
                &src.linear(&format!("{p}.attn.compressor.wgate"), 2 * hd, h),
                &src.dense(&format!("{p}.attn.compressor.ape")),
                &src.dense(&format!("{p}.attn.compressor.norm.weight")),
                eps, inv, scaling, ql, inh, ihd, cfg.index_topk, &ix,
            );
            (ckv, vis_to_additive(&vis))
        }
    };

    mla_core_ext(
        &cfg.mla, x, seq, &w_q_a, &w_q_a_norm, &w_q_b, &w_kv, &w_kv_norm, &w_o_a, &w_o_b,
        &sinks, &cos, &sin, mask, &compressed_kv, &block_bias,
    )
}

/// 1/0 visibility ints → additive mask (0.0 / -inf).
fn vis_to_additive(vis: &[i32]) -> Vec<f32> {
    vis.iter().map(|&v| if v != 0 { 0.0 } else { f32::NEG_INFINITY }).collect()
}

/// One MoE sub-layer (post post_attention_layernorm) → `[S,H]`.
fn moe_layer<S: Dsv4Src>(
    cfg: &Dsv4Config,
    li: usize,
    mt: MlpType,
    x: &[f32],
    input_ids: &[u32],
    src: &S,
) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let ii = cfg.moe_intermediate_size;
    let ne = cfg.num_local_experts;
    let tk = cfg.num_experts_per_tok;
    let seq = input_ids.len();
    let p = format!("model.layers.{li}");

    let gate_w = src.linear(&format!("{p}.ffn.gate.weight"), ne, h);
    let (idx, wts) = match mt {
        MlpType::Moe => {
            let corr = src.dense(&format!("{p}.ffn.gate.e_score_correction_bias"));
            topk_router(x, &gate_w, &corr, seq, h, ne, tk, cfg.routed_scaling_factor, cfg.norm_topk_prob)
        }
        MlpType::HashMoe => {
            let tid2eid = src.dense_i64(&format!("{p}.ffn.gate.tid2eid"));
            hash_router(x, &gate_w, &tid2eid, input_ids, seq, h, ne, tk, cfg.routed_scaling_factor, cfg.norm_topk_prob)
        }
    };

    // Pre-dequant the hit experts (union over tokens), gate_up = cat[gate; up] [2I,H].
    let mut hits: Vec<usize> = idx.clone();
    hits.sort_unstable();
    hits.dedup();
    let mut store: std::collections::HashMap<usize, (Vec<f32>, Vec<f32>)> = std::collections::HashMap::new();
    for &e in &hits {
        let gate = src.expert(&format!("{p}.ffn.switch_mlp.gate_proj"), e, ii, h);
        let up = src.expert(&format!("{p}.ffn.switch_mlp.up_proj"), e, ii, h);
        let mut gu = Vec::with_capacity(2 * ii * h);
        gu.extend_from_slice(&gate);
        gu.extend_from_slice(&up);
        let down = src.expert(&format!("{p}.ffn.switch_mlp.down_proj"), e, h, ii);
        store.insert(e, (gu, down));
    }

    let shared_gate = src.linear(&format!("{p}.ffn.shared_experts.gate_proj"), ii, h);
    let shared_up = src.linear(&format!("{p}.ffn.shared_experts.up_proj"), ii, h);
    let shared_down = src.linear(&format!("{p}.ffn.shared_experts.down_proj"), h, ii);

    let fetch = |e: usize| -> (&[f32], &[f32]) {
        let (gu, dn) = &store[&e];
        (gu.as_slice(), dn.as_slice())
    };
    moe_forward(
        x, seq, h, ii, &idx, &wts, tk, cfg.swiglu_limit, fetch,
        &shared_gate, &shared_up, &shared_down,
    )
}

/// One full decoder layer: streams `[S, hc, H]` → `[S, hc, H]`.
#[allow(clippy::too_many_arguments)]
fn decoder_layer<S: Dsv4Src>(
    cfg: &Dsv4Config,
    li: usize,
    mut streams: Vec<f32>,
    input_ids: &[u32],
    positions: &[usize],
    mask: &[f32],
    inv_main: &[f32],
    inv_comp: &[f32],
    src: &S,
) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let hc = cfg.hc_mult;
    let seq = input_ids.len();
    let eps = cfg.mla.rms_norm_eps;
    let iters = cfg.hc_sinkhorn_iters;
    let hc_eps = cfg.hc_eps;
    let p = format!("model.layers.{li}");
    let lt = cfg.layer_types[li];
    let mt = cfg.mlp_layer_types[li];

    // ---- attention site ----
    let attn_fn = src.dense(&format!("{p}.attn_hc.fn"));
    let attn_base = src.dense(&format!("{p}.attn_hc.base"));
    let attn_scale = src.dense(&format!("{p}.attn_hc.scale"));
    let (post, comb, collapsed) =
        hc_block(&streams, seq, hc, h, &attn_fn, &attn_base, &attn_scale, iters, hc_eps, eps);
    let in_ln = src.dense(&format!("{p}.attn_norm.weight"));
    let x_in = rmsnorm_rows(&collapsed, &in_ln, seq, h, eps);
    let attn_out = attention_layer(cfg, li, lt, &x_in, seq, positions, mask, inv_main, inv_comp, src);
    streams = hc_residual_mix(&post, &attn_out, &comb, &streams, seq, hc, h);

    // ---- ffn site ----
    let ffn_fn = src.dense(&format!("{p}.ffn_hc.fn"));
    let ffn_base = src.dense(&format!("{p}.ffn_hc.base"));
    let ffn_scale = src.dense(&format!("{p}.ffn_hc.scale"));
    let (post, comb, collapsed) =
        hc_block(&streams, seq, hc, h, &ffn_fn, &ffn_base, &ffn_scale, iters, hc_eps, eps);
    let post_ln = src.dense(&format!("{p}.ffn_norm.weight"));
    let x_in = rmsnorm_rows(&collapsed, &post_ln, seq, h, eps);
    let mlp_out = moe_layer(cfg, li, mt, &x_in, input_ids, src);
    streams = hc_residual_mix(&post, &mlp_out, &comb, &streams, seq, hc, h);
    streams
}

/// HyperHead final stream collapse `[S, hc, H]` → `[S, H]` (mirrors
/// `DeepseekV4HyperHead.forward`).
fn hyper_head<S: Dsv4Src>(streams: &[f32], cfg: &Dsv4Config, src: &S, seq: usize) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let hc = cfg.hc_mult;
    let hcd = hc * h;
    let eps = cfg.mla.rms_norm_eps;
    let hc_fn = src.dense("model.hc_head.fn"); // [hc, hc*H]
    let hc_base = src.dense("model.hc_head.base"); // [hc]
    let hc_scale = src.dense("model.hc_head.scale"); // [1]
    let flat = unweighted_rmsnorm_rows(streams, seq, hcd, eps); // [S, hc*H]
    let sc = hc_scale[0] as f64;
    let mut out = vec![0.0f32; seq * h];
    for si in 0..seq {
        let frow = &flat[si * hcd..(si + 1) * hcd];
        // pre[k] = sigmoid(sum_d frow[d]*hc_fn[k,d] * scale + base[k]) + eps
        let mut pre = vec![0.0f64; hc];
        for k in 0..hc {
            let wrow = &hc_fn[k * hcd..(k + 1) * hcd];
            let mut acc = 0.0f64;
            for d in 0..hcd {
                acc += frow[d] as f64 * wrow[d] as f64;
            }
            pre[k] = 1.0 / (1.0 + (-(acc * sc + hc_base[k] as f64)).exp()) + cfg.hc_eps as f64;
        }
        let orow = &mut out[si * h..(si + 1) * h];
        for k in 0..hc {
            let strow = &streams[si * hcd + k * h..si * hcd + k * h + h];
            for d in 0..h {
                orow[d] += (pre[k] * strow[d] as f64) as f32;
            }
        }
    }
    out
}

/// Full model forward → last-position logits are `logits[(S-1)*vocab..]`. Returns
/// the full `[S, vocab]` logit matrix (f32). Mirrors `golden.py::run_forward`.
pub fn dsv4_forward<S: Dsv4Src>(cfg: &Dsv4Config, input_ids: &[u32], src: &S) -> Vec<f32> {
    let h = cfg.mla.hidden_size;
    let hc = cfg.hc_mult;
    let vocab = cfg.vocab_size;
    let seq = input_ids.len();
    let eps = cfg.mla.rms_norm_eps;
    let positions: Vec<usize> = (0..seq).collect();
    let inv_main = cfg.inv_freq_main();
    let inv_comp = cfg.inv_freq_compress();
    let mask = sliding_mask(seq, cfg.mla.sliding_window);

    // embed → expand to hc parallel streams [S, hc, H].
    let emb = src.embed_rows(input_ids, vocab, h);
    let hcd = hc * h;
    let mut streams = vec![0.0f32; seq * hcd];
    for si in 0..seq {
        for k in 0..hc {
            streams[si * hcd + k * h..si * hcd + k * h + h]
                .copy_from_slice(&emb[si * h..(si + 1) * h]);
        }
    }

    for li in 0..cfg.num_hidden_layers {
        streams = decoder_layer(cfg, li, streams, input_ids, &positions, &mask, &inv_main, &inv_comp, src);
    }

    let collapsed = hyper_head(&streams, cfg, src, seq);
    let normed = rmsnorm_rows(&collapsed, &src.dense("model.norm.weight"), seq, h, eps);
    let lm = src.linear("lm_head", vocab, h);
    cpu_matmul(&normed, &lm, seq, h, vocab)
}

/// Argmax of the last-position logits (the golden ` Paris` == 11111 anchor).
pub fn dsv4_argmax_last<S: Dsv4Src>(cfg: &Dsv4Config, input_ids: &[u32], src: &S) -> (u32, f32) {
    let logits = dsv4_forward(cfg, input_ids, src);
    let vocab = cfg.vocab_size;
    let last = &logits[(input_ids.len() - 1) * vocab..];
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in last.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    (bi as u32, bv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::collections::HashMap;

    /// In-memory `Dsv4Src` from a checkpoint-named f32 dict (the tiny self-test's
    /// `DictSource`). 3D expert tensors are stored flat `[E*out*in]`.
    struct DictSrc {
        d: HashMap<String, Vec<f32>>,
        i: HashMap<String, Vec<i64>>,
    }
    impl Dsv4Src for DictSrc {
        fn linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32> {
            let key = if self.d.contains_key(name) { name.to_string() } else { format!("{name}.weight") };
            let v = self.d.get(&key).unwrap_or_else(|| panic!("missing linear {name}"));
            assert_eq!(v.len(), out_f * in_f, "linear {name} shape");
            v.clone()
        }
        fn dense(&self, name: &str) -> Vec<f32> {
            self.d.get(name).unwrap_or_else(|| panic!("missing dense {name}")).clone()
        }
        fn expert(&self, name: &str, e: usize, out_f: usize, in_f: usize) -> Vec<f32> {
            let v = self.d.get(name).unwrap_or_else(|| panic!("missing expert {name}"));
            v[e * out_f * in_f..(e + 1) * out_f * in_f].to_vec()
        }
        fn dense_i64(&self, name: &str) -> Vec<i64> {
            self.i.get(name).unwrap_or_else(|| panic!("missing i64 {name}")).clone()
        }
    }

    fn f32v(v: &Value) -> Vec<f32> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
    }

    /// Full-composition self-test: the Rust forward on a tiny non-quantized DSV4
    /// model reproduces the transformers reference argmax (and logits within tol),
    /// exercising ALL layer/mlp types (sliding/CSA/HCA, hash/topk). Fixture from
    /// `golden.py --dump-selftest`.
    #[test]
    fn full_forward_matches_reference_tiny() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/selftest.json");
        let raw = std::fs::read_to_string(path)
            .expect("selftest.json (run: python3 scripts/dsv4/golden.py --dump-selftest tests/fixtures/dsv4/selftest.json)");
        let j: Value = serde_json::from_str(&raw).unwrap();
        let cfg = Dsv4Config::from_json(&j["config"]).unwrap();
        let input_ids: Vec<u32> =
            j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();

        let mut d = HashMap::new();
        let mut i = HashMap::new();
        for (k, v) in j["weights"].as_object().unwrap() {
            d.insert(k.clone(), f32v(v));
        }
        for (k, v) in j["weights_i64"].as_object().unwrap() {
            i.insert(k.clone(), v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect());
        }
        let src = DictSrc { d, i };

        let logits = dsv4_forward(&cfg, &input_ids, &src);
        let vocab = cfg.vocab_size;
        let last = &logits[(input_ids.len() - 1) * vocab..];
        let mine_argmax = last.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0;
        let ref_argmax = j["ref_argmax"].as_u64().unwrap() as usize;
        let ref_logits = f32v(&j["ref_last_logits"]);
        let mae = last.iter().zip(ref_logits.iter()).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("tiny full-forward: argmax mine={mine_argmax} ref={ref_argmax} logits_mae={mae:.3e}");
        assert_eq!(mine_argmax, ref_argmax, "argmax mismatch (logits_mae={mae:.3e})");
        assert!(mae < 5e-2, "logits max_abs_err {mae:.3e} too large");
    }
}
