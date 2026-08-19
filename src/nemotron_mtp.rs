// SPDX-License-Identifier: Apache-2.0
//! Nemotron-75B MTP (Multi-Token-Prediction) draft head — Phase 1 of the
//! spec-decode plan (memory `nemotron-mtp-spec-decode-review.md`: Phase 0
//! GO'd the batched-verify kernel amortization; Phase 1 is THIS file, a
//! CPU-only greedy-vs-greedy acceptance sim, gating whether the shipped head
//! is accept-healthy before any GPU kernel work is justified).
//!
//! ## Checkpoint census (read directly off the real NVFP4 checkpoint's
//! `model.safetensors.index.json` + `mtp.safetensors` + `config.json`
//! `mtp_block_configs`/`num_nextn_predict_layers`, NOT assumed):
//!   * ALL 1040 `mtp.*` tensors are plain BF16 (F32 only for
//!     `mtp.layers.1.mixer.gate.e_score_correction_bias`) — the memory note's
//!     "NVFP4 512 experts" description was wrong; the checkpoint's NVFP4
//!     `config_groups` quant map applies only to the BASE model's routed
//!     experts, not the MTP head's. This removes the NVFP4-dequant-reuse
//!     requirement entirely — good news for Phase 1 (no quant math needed)
//!     but the reason the head is 11.8GB as naive f32 (see below), not the
//!     ~1.7GB the memory note estimated.
//!   * `mtp.layers.0` = a NoPE-attention mixer: `eh_proj` [4096,8192] bootstrap
//!     + `enorm`/`hnorm` [4096] RMSNorms + `norm` [4096] (pre-attn input norm)
//!     + `mixer.{q,k,v,o}_proj` — IDENTICAL shapes to the base model's
//!     `nope_attention` (q/o [4096,4096], k/v [256,4096], nq=32 nkv=2 hd=128)
//!     and, confirmed by the tensor census, NO q_norm/k_norm/rope/gate (same
//!     as base `nope_attention` — simpler than the module doc in the task
//!     brief implied).
//!   * `mtp.layers.1` = a latent-MoE mixer, IDENTICAL shape family to the
//!     base model's latent-MoE layers: `norm` (pre-moe input norm),
//!     `mixer.{fc1_latent_proj,fc2_latent_proj,gate,shared_experts.*}` +
//!     512 routed `mixer.experts.{e}.{up,down}_proj`, `final_layernorm`
//!     (the head's own final norm, analogous to `backbone.norm_f`).
//!     `mtp_block_configs[1]` gives this layer's own `top_k=22` /
//!     `moe_intermediate_size=2688` (independent of the base model's per-MoE-
//!     layer overrides).
//!   * lm_head + embed_tokens are the BASE model's shared tables
//!     (`num_nextn_predict_layers=1`, no dedicated-embedding flag in this
//!     checkpoint) — reused by reference, not duplicated, EXCEPT the
//!     embedding table, which must be reachable from whichever PP stage owns
//!     the head (see `embed_bits` below).
//!
//! ## Memory: the 11.8GB naive-f32 problem (flagged, not hidden)
//! 512 experts × (up `[2688,1024]` + down `[1024,2688]`) × 4 B/f32 ≈ 11.28 GB,
//! plus ~0.5GB for the rest ≈ 11.8GB measured (`python3 -c` census against the
//! real checkpoint, see the Phase-1 deliverable report). Eagerly dequantizing
//! ALL 512 experts to f32 host — what `nemotron::latent_moe_forward`'s CPU
//! fallback does, and what it must do since it has no lazy path — would both
//! blow that budget AND (worse) reallocate+recopy the whole 11.8GB on EVERY
//! head-forward call (that function concatenates all `ne` experts from
//! `ModelWeights` fresh each call; fine for its actual use — a correctness
//! fallback, called rarely — fatal for a hot Phase-1 loop calling it up to 4×
//! per decode step). So this module does NOT reuse `latent_moe_forward`
//! directly for the head's MoE sub-layer: routed experts are kept
//! HOST-RESIDENT AS RAW BF16 BITS ONCE at load (5.6GB, half the f32 size,
//! loaded once) and dequantized LAZILY — only the router's selected
//! `top_k=22` of 512 experts, per call — reusing the identical
//! `half::bf16::from_bits(..).to_f32()` decode arithmetic every BF16 tensor
//! in this codebase already uses (`nemotron_loader::decode_plain`'s BF16
//! branch). The router selection (`nemotron::router_forward`) and the
//! per-expert relu² MLP (`nemotron::mlp_relu2`) ARE reused verbatim — only the
//! outer per-token glue (selecting/dequantizing the routed experts, the
//! fc1/fc2 bottleneck, and the shared-expert combine) is new, and it mirrors
//! `latent_moe_forward`'s single-token body op-for-op.
//!
//! Even at 5.6GB (bf16-resident experts) + ~0.5GB (small tensors) + ~1.1GB
//! (bf16-resident embed-table copy, see below) ≈ 7.2GB, this is a LARGE
//! addition to co-locate with the already-heaviest PP-5 last stage (per
//! `nemotron-decode-perf-review` memory, ~11-12.7GB on a 14GB BC-250 node).
//! **This module does not attempt to fix that** — Phase 1's job is the
//! accept-health number, not the memory-fit; the Phase-1 deliverable report
//! flags PP-6 (or a layer-limited base-model slice) as the likely-required
//! mitigation for actually RUNNING the sim on the 5-node cluster.
//!
//! ## Embedding-table duplication
//! The draft chain needs `embed(token)` for every drafted token after the
//! first (the self-chain feeds the head its own previous draft's embedding).
//! In a PP>1 split the embedding table normally lives ONLY on stage 0
//! (`nemotron_loader::load_nemotron_weights`'s `keep_embed` gate), but the
//! head is co-located with the LAST stage (where `lm_head` lives, since it
//! reuses that table for the draft logits). So this loader independently
//! re-reads `backbone.embeddings.weight` from the base checkpoint's shards
//! (`model::discover_shards`) as its own BF16-bits copy — a ~1.1GB addition,
//! unavoidable without cross-stage plumbing that is out of scope for Phase 1.
#![allow(dead_code)]

use crate::model::{cpu_matmul, cpu_matvec_par, cpu_rms_norm, cpu_sdpa, discover_shards, KvCache};
use crate::nemotron::{mlp_relu2, router_forward, NemotronConfig, RouterDims};
use half::bf16;
use std::path::Path;

/// `mtp_block_configs[1]` overrides (read from the real checkpoint's
/// `config.json`; NOT the same as any base-model MoE layer's top_k/inter).
const MTP_MOE_TOP_K: usize = 22;
const MTP_MOE_INTERMEDIATE: usize = 2688;

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

fn decode_bf16_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn decode_bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn dequant_bf16_slice(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| bf16_to_f32(b)).collect()
}

/// The Nemotron-75B MTP draft head (Phase 1, CPU/host-only — see module doc
/// for why: no NVFP4 in this checkpoint's `mtp.*`, and the resident-GPU path
/// is out of scope for this phase). Holds its own small tensors as f32, the
/// 512 routed experts as raw BF16 bits (lazy dequant, see module doc), a
/// duplicated BF16-bits copy of the shared embedding table, and a tiny KV
/// cache for its one attention layer (reset every fresh draft cycle — Phase 1
/// measures PER-CYCLE acceptance, not the continuous multi-cycle KV-refill
/// scheme a later phase would need for a real speculative decode loop).
pub struct NemMtpHead {
    // dims
    pub h: usize,
    pub nq: usize,
    pub nkv: usize,
    pub hd: usize,
    pub ne: usize,
    pub lat: usize,
    pub inter: usize,
    pub shared_inter: usize,
    pub vocab: usize,
    pub eps: f32,
    pub router: RouterDims,

    // bootstrap: eh_proj([norm_h(hidden_pre) ; norm_e(embed_next)]) -> x0.
    // HIDDEN FIRST in the concat — the checkpoint's `eh_proj` name order
    // (embedding-hidden) is the tensor's NAME, not necessarily the concat
    // order; this is set to match the DeepSeek-NextN convention this
    // architecture family uses elsewhere in this codebase's docs (hidden
    // first). Flagged: unlike the Qwen MTP head (P0-recovered, embed-FIRST),
    // this order has NOT been numerically recovered against an MLX oracle —
    // there is no such oracle for Nemotron's MTP head yet. If Phase 1's
    // acceptance number comes back unhealthy, swapping this order is the
    // first thing to try before concluding the head itself is unhealthy.
    pub eh_proj: Vec<f32>, // [h, 2h]
    pub enorm: Vec<f32>,   // mtp.layers.0.enorm, applied to the embedding half
    pub hnorm: Vec<f32>,   // mtp.layers.0.hnorm, applied to the hidden half

    // layer 0: NoPE attention (mirrors nemotron::NemotronModel::nope_attention
    // op-for-op: no rope, no qk-norm, no output gate).
    pub attn_norm: Vec<f32>, // mtp.layers.0.norm (pre-attn input norm)
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,

    // layer 1: latent-MoE (mirrors nemotron::latent_moe_forward's per-token
    // body; router_forward/mlp_relu2 reused verbatim, see module doc).
    pub moe_norm: Vec<f32>, // mtp.layers.1.norm (pre-moe input norm)
    pub gate_weight: Vec<f32>,
    pub e_score_correction_bias: Vec<f32>,
    pub fc1_latent_proj: Vec<f32>,
    pub fc2_latent_proj: Vec<f32>,
    pub shared_up: Vec<f32>,
    pub shared_down: Vec<f32>,
    /// [ne * inter * lat] raw BF16 bits, per-expert-contiguous (NOT eagerly
    /// dequantized — see module doc).
    pub expert_up_bits: Vec<u16>,
    /// [ne * lat * inter] raw BF16 bits, per-expert-contiguous.
    pub expert_down_bits: Vec<u16>,

    pub final_norm: Vec<f32>, // mtp.layers.1.final_layernorm

    /// Duplicated BF16-bits copy of `backbone.embeddings.weight`
    /// (`[vocab, h]`), re-read independently of the base model's own
    /// (stage-0-only) embedding table — see module doc.
    pub embed_bits: Vec<u16>,

    /// The head's own KV for its single attention layer. Reset before every
    /// fresh draft cycle by the caller (`reset()`).
    pub kv: KvCache,
}

impl NemMtpHead {
    /// Load the head from `<base_dir>/mtp.safetensors` (co-located with the
    /// base model's shards) using the base model's `NemotronConfig` for
    /// shared dims (`hidden_size`, attention geometry, `moe_latent_size`,
    /// router knobs). `base_dir` is also rescanned (`discover_shards`) for
    /// `backbone.embeddings.weight` (see module doc on the duplication).
    pub fn load(base_dir: &Path, cfg: &NemotronConfig) -> Result<Self, String> {
        use memmap2::Mmap;
        use safetensors::SafeTensors;
        use std::fs::File;

        let mtp_path = base_dir.join("mtp.safetensors");
        let file = File::open(&mtp_path)
            .map_err(|e| format!("open {}: {e}", mtp_path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", mtp_path.display()))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("parse {}: {e}", mtp_path.display()))?;

        let get_f32 = |name: &str| -> Result<Vec<f32>, String> {
            let v = st.tensor(name).map_err(|e| format!("mtp tensor '{name}': {e}"))?;
            match v.dtype() {
                safetensors::Dtype::BF16 => Ok(decode_bf16_f32(v.data())),
                safetensors::Dtype::F32 => Ok(v
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()),
                other => Err(format!("mtp tensor '{name}': unexpected dtype {other:?} (expected BF16/F32)")),
            }
        };
        let get_bits = |name: &str| -> Result<Vec<u16>, String> {
            let v = st.tensor(name).map_err(|e| format!("mtp tensor '{name}': {e}"))?;
            if v.dtype() != safetensors::Dtype::BF16 {
                return Err(format!("mtp tensor '{name}': expected BF16, got {:?}", v.dtype()));
            }
            Ok(decode_bf16_bits(v.data()))
        };

        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let ne = cfg.n_routed_experts;
        let lat = cfg.moe_latent_size;
        let shared_inter = cfg.moe_shared_expert_intermediate_size;
        let top_k = MTP_MOE_TOP_K;
        let inter = MTP_MOE_INTERMEDIATE;

        let up_stride = inter * lat;
        let down_stride = lat * inter;
        let mut expert_up_bits = Vec::with_capacity(ne * up_stride);
        let mut expert_down_bits = Vec::with_capacity(ne * down_stride);
        for e in 0..ne {
            let up = get_bits(&format!("mtp.layers.1.mixer.experts.{e}.up_proj.weight"))?;
            let down = get_bits(&format!("mtp.layers.1.mixer.experts.{e}.down_proj.weight"))?;
            if up.len() != up_stride {
                return Err(format!(
                    "mtp expert {e} up_proj: {} elems, expected inter*lat={up_stride}", up.len()));
            }
            if down.len() != down_stride {
                return Err(format!(
                    "mtp expert {e} down_proj: {} elems, expected lat*inter={down_stride}", down.len()));
            }
            expert_up_bits.extend(up);
            expert_down_bits.extend(down);
        }

        // Re-scan the base model's shards for the shared embedding table
        // (see module doc — the head co-locates with the LAST stage, which
        // does not otherwise hold `backbone.embeddings.weight`).
        let embed_bits = {
            let probe = base_dir.join("model.safetensors.index.json");
            let shards = discover_shards(&probe);
            let mut found: Option<Vec<u16>> = None;
            for shard in &shards {
                let f = File::open(shard).map_err(|e| format!("open {}: {e}", shard.display()))?;
                let mm = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", shard.display()))?;
                let sst = SafeTensors::deserialize(&mm).map_err(|e| format!("parse {}: {e}", shard.display()))?;
                if let Ok(v) = sst.tensor("backbone.embeddings.weight") {
                    if v.dtype() != safetensors::Dtype::BF16 {
                        return Err(format!(
                            "backbone.embeddings.weight: expected BF16, got {:?}", v.dtype()));
                    }
                    found = Some(decode_bf16_bits(v.data()));
                    break;
                }
            }
            found.ok_or_else(|| "backbone.embeddings.weight not found in any shard".to_string())?
        };
        if embed_bits.len() != cfg.vocab_size * h {
            return Err(format!(
                "embed table: {} elems, expected vocab*h={}", embed_bits.len(), cfg.vocab_size * h));
        }

        let head = NemMtpHead {
            h, nq, nkv, hd, ne, lat, inter, shared_inter,
            vocab: cfg.vocab_size,
            eps: cfg.norm_eps,
            router: RouterDims {
                n_routed_experts: ne,
                top_k,
                routed_scaling_factor: cfg.routed_scaling_factor,
                n_group: cfg.n_group,
                topk_group: cfg.topk_group,
                norm_topk_prob: cfg.norm_topk_prob,
            },
            eh_proj: get_f32("mtp.layers.0.eh_proj.weight")?,
            enorm: get_f32("mtp.layers.0.enorm.weight")?,
            hnorm: get_f32("mtp.layers.0.hnorm.weight")?,
            attn_norm: get_f32("mtp.layers.0.norm.weight")?,
            q_proj: get_f32("mtp.layers.0.mixer.q_proj.weight")?,
            k_proj: get_f32("mtp.layers.0.mixer.k_proj.weight")?,
            v_proj: get_f32("mtp.layers.0.mixer.v_proj.weight")?,
            o_proj: get_f32("mtp.layers.0.mixer.o_proj.weight")?,
            moe_norm: get_f32("mtp.layers.1.norm.weight")?,
            gate_weight: get_f32("mtp.layers.1.mixer.gate.weight")?,
            e_score_correction_bias: get_f32("mtp.layers.1.mixer.gate.e_score_correction_bias")?,
            fc1_latent_proj: get_f32("mtp.layers.1.mixer.fc1_latent_proj.weight")?,
            fc2_latent_proj: get_f32("mtp.layers.1.mixer.fc2_latent_proj.weight")?,
            shared_up: get_f32("mtp.layers.1.mixer.shared_experts.up_proj.weight")?,
            shared_down: get_f32("mtp.layers.1.mixer.shared_experts.down_proj.weight")?,
            expert_up_bits,
            expert_down_bits,
            final_norm: get_f32("mtp.layers.1.final_layernorm.weight")?,
            embed_bits,
            kv: KvCache::new(8, nkv, hd), // depth<=4, reset every cycle; 8 is a generous cap
        };
        // shape sanity (catches an eh_proj/attn transposition early).
        if head.eh_proj.len() != h * 2 * h {
            return Err(format!("eh_proj: {} elems, expected h*2h={}", head.eh_proj.len(), h * 2 * h));
        }
        if head.q_proj.len() != nq * hd * h {
            return Err(format!("mtp q_proj: {} elems, expected nq*hd*h={}", head.q_proj.len(), nq * hd * h));
        }
        Ok(head)
    }

    /// Reset the head's KV (start a fresh draft chain from a newly-committed
    /// real token). Call before every `head_chain_cpu`.
    pub fn reset(&mut self) {
        self.kv.seq_len = 0;
    }

    /// `embed(token)` from the duplicated BF16-bits table (lazy dequant of
    /// one `[h]` row — cheap, called once per drafted token).
    pub fn embed(&self, token: u32) -> Vec<f32> {
        let off = token as usize * self.h;
        dequant_bf16_slice(&self.embed_bits[off..off + self.h])
    }

    /// Draft-logits argmax through the BASE model's shared lm_head (borrowed,
    /// NOT duplicated here — `lm_head_weight` is `[vocab, h]` f32, owned by
    /// the caller's `NemotronModel::weights`). Uses `cpu_matvec_par` (rayon,
    /// bit-identical to a serial matvec — see its doc in `nemotron.rs`) since
    /// a 4096×131072 matvec is the single most expensive op in the chain.
    pub fn argmax_with_lm_head(&self, head_hidden: &[f32], lm_head_weight: &[f32]) -> u32 {
        let logits = cpu_matvec_par(head_hidden, lm_head_weight, self.h, self.vocab);
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        best_i as u32
    }

    /// eh_proj bootstrap: `x0 = eh_proj( [norm_h(hidden_pre) ; norm_e(embed_next)] )`.
    fn bootstrap(&self, embed_next: &[f32], hidden_pre: &[f32]) -> Vec<f32> {
        let nh = cpu_rms_norm(hidden_pre, &self.hnorm, self.eps);
        let ne_ = cpu_rms_norm(embed_next, &self.enorm, self.eps);
        let mut comb = Vec::with_capacity(2 * self.h);
        comb.extend_from_slice(&nh);
        comb.extend_from_slice(&ne_);
        cpu_matmul(&comb, &self.eh_proj, 1, 2 * self.h, self.h)
    }

    /// Layer-0 NoPE attention sub-block. Op-for-op mirror of
    /// `nemotron::NemotronModel::nope_attention` (no rope, no qk-norm, no
    /// output gate) against the head's own small f32 weights + its own KV
    /// (the head never goes GPU-resident — this is `cpu_matmul`, not the
    /// base model's `nem_matvec` GPU/CPU dispatch).
    fn attn_layer(&mut self, x: &[f32]) -> Vec<f32> {
        let residual = x.to_vec();
        let xn = cpu_rms_norm(x, &self.attn_norm, self.eps);
        let q_dim = self.nq * self.hd;
        let kv_dim = self.nkv * self.hd;
        let q = cpu_matmul(&xn, &self.q_proj, 1, self.h, q_dim);
        let k = cpu_matmul(&xn, &self.k_proj, 1, self.h, kv_dim);
        let v = cpu_matmul(&xn, &self.v_proj, 1, self.h, kv_dim);
        self.kv.append(&k, &v);
        let scale = 1.0 / (self.hd as f32).sqrt();
        let attn = cpu_sdpa(
            &q, self.kv.k_up_to_now(), self.kv.v_up_to_now(),
            self.nq, self.nkv, self.hd, self.kv.seq_len, scale, None,
        );
        let attn_out = cpu_matmul(&attn, &self.o_proj, 1, q_dim, self.h);
        residual.iter().zip(&attn_out).map(|(&r, &a)| r + a).collect()
    }

    /// Layer-1 latent-MoE sub-block. `router_forward`/`mlp_relu2` are the
    /// EXACT functions the base model's MoE layers use (reused verbatim);
    /// only the per-call expert lookup is new (lazy BF16 dequant of the
    /// `top_k` selected experts — see module doc on why this does NOT call
    /// `nemotron::latent_moe_forward` directly).
    fn moe_layer(&self, x: &[f32]) -> Vec<f32> {
        let residual = x.to_vec();
        let xn = cpu_rms_norm(x, &self.moe_norm, self.eps);
        let (indices, weights) =
            router_forward(&xn, &self.gate_weight, &self.e_score_correction_bias, &self.router);
        let latent = cpu_matmul(&xn, &self.fc1_latent_proj, 1, self.h, self.lat);
        let up_stride = self.inter * self.lat;
        let down_stride = self.lat * self.inter;
        let mut routed = vec![0.0f32; self.lat];
        for (k, &e) in indices.iter().enumerate() {
            let up_bits = &self.expert_up_bits[e * up_stride..(e + 1) * up_stride];
            let down_bits = &self.expert_down_bits[e * down_stride..(e + 1) * down_stride];
            let up = dequant_bf16_slice(up_bits);
            let down = dequant_bf16_slice(down_bits);
            let eout = mlp_relu2(&latent, &up, &down, self.lat, self.inter);
            let wk = weights[k];
            for (r, &o) in routed.iter_mut().zip(&eout) {
                *r += o * wk;
            }
        }
        let moe_out = cpu_matmul(&routed, &self.fc2_latent_proj, 1, self.lat, self.h);
        let shared = mlp_relu2(&xn, &self.shared_up, &self.shared_down, self.h, self.shared_inter);
        residual
            .iter()
            .zip(moe_out.iter().zip(&shared))
            .map(|(&r, (&m, &s))| r + m + s)
            .collect()
    }

    /// One head-forward step, advancing the head's own KV by one position.
    /// Returns `(residual, head_hidden)`: `residual` (pre-`final_norm`) is
    /// what a chained NEXT draft step feeds back in as `hidden_pre` (the
    /// DeepSeek-NextN self-chain, mirroring the Qwen head's
    /// `head_forward_with`); `head_hidden` (post-`final_norm`) is what the
    /// shared lm_head consumes for THIS position's draft logits.
    pub fn head_forward(&mut self, embed_next: &[f32], hidden_pre: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let x0 = self.bootstrap(embed_next, hidden_pre);
        let x1 = self.attn_layer(&x0);
        let x2 = self.moe_layer(&x1);
        let head_hidden = cpu_rms_norm(&x2, &self.final_norm, self.eps);
        (x2, head_hidden)
    }

    /// Greedy autoregressive draft chain of `depth` tokens (argmax throughout
    /// — this measures the head's agreement with the base model's own greedy
    /// decode, the standard spec-decode acceptance metric). `first_embed` =
    /// `embed(the just-committed real token)`, `first_hidden` = the base
    /// model's pre-`norm_f` residual from THAT SAME decode step. Caller must
    /// `reset()` the head KV first (Phase 1 measures per-cycle acceptance,
    /// not continuous multi-cycle refill).
    pub fn head_chain_cpu(
        &mut self,
        first_embed: &[f32],
        first_hidden: &[f32],
        depth: usize,
        lm_head_weight: &[f32],
    ) -> Vec<u32> {
        let mut drafts = Vec::with_capacity(depth);
        let mut embed = first_embed.to_vec();
        let mut hidden = first_hidden.to_vec();
        for _ in 0..depth {
            let (residual, head_hidden) = self.head_forward(&embed, &hidden);
            let tok = self.argmax_with_lm_head(&head_hidden, lm_head_weight);
            embed = self.embed(tok);
            hidden = residual;
            drafts.push(tok);
        }
        drafts
    }
}

#[cfg(test)]
mod tests {
    //! Mac-runnable (engine-less) shape/wiring gates. There is no MLX oracle
    //! for the Nemotron MTP head (unlike Qwen's, which had a P0 numeric
    //! recovery) — these tests check internal consistency (shapes, KV
    //! advance/reset, chain determinism) against a SYNTHETIC tensor set, not
    //! bit-exactness against a reference. The concat order (`bootstrap`) and
    //! any other architectural assumption stated in the module doc as
    //! "unverified" stays unverified until Phase 1 runs against the real
    //! checkpoint on the cluster and the acceptance number is judged sane.
    use super::*;
    use crate::nemotron::BlockSpec;
    use std::collections::HashMap;

    fn gen(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn tiny_cfg() -> NemotronConfig {
        NemotronConfig {
            hidden_size: 8,
            num_hidden_layers: 2,
            vocab_size: 16,
            norm_eps: 1e-6,
            tie_word_embeddings: false,
            mamba_num_heads: 1,
            mamba_head_dim: 1,
            ssm_state_size: 1,
            n_groups: 1,
            conv_kernel: 1,
            use_conv_bias: false,
            time_step_min: 0.0,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 2,
            attention_bias: false,
            n_routed_experts: 4,
            moe_latent_size: 6,
            moe_shared_expert_intermediate_size: 6,
            routed_scaling_factor: 1.0,
            norm_topk_prob: true,
            n_group: 1,
            topk_group: 1,
            block_specs: vec![BlockSpec::Attention, BlockSpec::Moe { num_experts_per_tok: 2, moe_intermediate_size: 5 }],
        }
    }

    /// Build a head directly (bypassing `load`'s safetensors I/O) from
    /// synthetic weights matching `tiny_cfg`'s shapes, so the wiring can be
    /// exercised without a checkpoint on disk.
    fn tiny_head(cfg: &NemotronConfig) -> NemMtpHead {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let ne = cfg.n_routed_experts;
        let lat = cfg.moe_latent_size;
        let inter = 5usize; // matches tiny_cfg's Moe override
        let si = cfg.moe_shared_expert_intermediate_size;
        let mut g = gen(42);
        fn mk(g: &mut impl FnMut() -> f32, n: usize) -> Vec<f32> {
            (0..n).map(|_| g() * 0.1).collect()
        }
        fn mk_bits(g: &mut impl FnMut() -> f32, n: usize) -> Vec<u16> {
            (0..n).map(|_| bf16::from_f32(g() * 0.1).to_bits()).collect()
        }
        NemMtpHead {
            h, nq, nkv, hd, ne, lat, inter, shared_inter: si,
            vocab: cfg.vocab_size,
            eps: cfg.norm_eps,
            router: RouterDims {
                n_routed_experts: ne, top_k: 2, routed_scaling_factor: 1.0,
                n_group: 1, topk_group: 1, norm_topk_prob: true,
            },
            eh_proj: mk(&mut g, h * 2 * h),
            enorm: mk(&mut g, h),
            hnorm: mk(&mut g, h),
            attn_norm: mk(&mut g, h),
            q_proj: mk(&mut g, nq * hd * h),
            k_proj: mk(&mut g, nkv * hd * h),
            v_proj: mk(&mut g, nkv * hd * h),
            o_proj: mk(&mut g, h * nq * hd),
            moe_norm: mk(&mut g, h),
            gate_weight: mk(&mut g, ne * h),
            e_score_correction_bias: mk(&mut g, ne),
            fc1_latent_proj: mk(&mut g, lat * h),
            fc2_latent_proj: mk(&mut g, h * lat),
            shared_up: mk(&mut g, si * h),
            shared_down: mk(&mut g, h * si),
            expert_up_bits: mk_bits(&mut g, ne * inter * lat),
            expert_down_bits: mk_bits(&mut g, ne * lat * inter),
            final_norm: mk(&mut g, h),
            embed_bits: mk_bits(&mut g, cfg.vocab_size * h),
            kv: KvCache::new(8, nkv, hd),
        }
    }

    #[test]
    fn head_forward_advances_kv_and_shapes() {
        let cfg = tiny_cfg();
        let mut head = tiny_head(&cfg);
        let h = cfg.hidden_size;
        let mut g = gen(1);
        let e: Vec<f32> = (0..h).map(|_| g()).collect();
        let hp: Vec<f32> = (0..h).map(|_| g()).collect();
        assert_eq!(head.kv.seq_len, 0);
        let (residual, head_hidden) = head.head_forward(&e, &hp);
        assert_eq!(residual.len(), h);
        assert_eq!(head_hidden.len(), h);
        assert_eq!(head.kv.seq_len, 1, "one head_forward call advances KV by one position");
    }

    #[test]
    fn reset_zeros_kv() {
        let cfg = tiny_cfg();
        let mut head = tiny_head(&cfg);
        let h = cfg.hidden_size;
        let mut g = gen(2);
        let e: Vec<f32> = (0..h).map(|_| g()).collect();
        let hp: Vec<f32> = (0..h).map(|_| g()).collect();
        head.head_forward(&e, &hp);
        head.head_forward(&e, &hp);
        assert_eq!(head.kv.seq_len, 2);
        head.reset();
        assert_eq!(head.kv.seq_len, 0);
    }

    #[test]
    fn chain_advances_kv_by_depth_and_is_deterministic() {
        let cfg = tiny_cfg();
        let h = cfg.hidden_size;
        let mut g = gen(3);
        let e0: Vec<f32> = (0..h).map(|_| g()).collect();
        let hp0: Vec<f32> = (0..h).map(|_| g()).collect();
        let lm_head: Vec<f32> = {
            let mut gg = gen(4);
            (0..cfg.vocab_size * h).map(|_| gg() * 0.1).collect()
        };

        let mut head1 = tiny_head(&cfg);
        let d1 = head1.head_chain_cpu(&e0, &hp0, 4, &lm_head);
        assert_eq!(d1.len(), 4);
        assert_eq!(head1.kv.seq_len, 4, "chain of depth 4 advances KV by 4");

        // greedy/argmax throughout -> re-running the identical chain (fresh
        // head, same weights, KV reset by construction) must reproduce the
        // exact same draft sequence.
        let mut head2 = tiny_head(&cfg);
        let d2 = head2.head_chain_cpu(&e0, &hp0, 4, &lm_head);
        assert_eq!(d1, d2, "greedy chain must be deterministic");
    }

    #[test]
    fn embed_lookup_matches_manual_dequant() {
        let cfg = tiny_cfg();
        let head = tiny_head(&cfg);
        let h = cfg.hidden_size;
        let tok = 3u32;
        let got = head.embed(tok);
        let want: Vec<f32> = head.embed_bits[tok as usize * h..(tok as usize + 1) * h]
            .iter()
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect();
        assert_eq!(got, want);
    }
}
