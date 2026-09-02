// SPDX-License-Identifier: Apache-2.0
//! INC-5b piece 4 — the PRODUCTION CALLER for the EAGLE spec-decode driver.
//!
//! `gemma_spec.rs` landed the accept/reject/rollback math and `gemma_assistant.rs`
//! landed the drafter; until this module neither had an in-crate caller outside
//! `#[cfg(test)]`, which is the maintainer follow-up on PR #89 ("wire
//! `spec_step`/`run_spec_decode` to a real caller — test-only today"). This is
//! that caller: it owns the drafter, the seed the drafter needs, the env flag
//! that turns it on, and the acceptance numbers a caller reads back.
//!
//! ## Which seam, and why this one
//!
//! Three candidates were on the table:
//!
//! * **`forward_rs` dispatch** — WRONG SHAPE. `forward_rs` is a ONE-token
//!   contract (`token, position -> logits`); a spec block emits between 1 and
//!   `k+1` tokens per call and rewinds KV underneath. Speculation cannot hide
//!   behind a per-token forward without lying to every existing caller about
//!   how far the position cursor moved.
//! * **The serving layer (`vllm_vulkan/`)** — TOO HIGH. It would have to
//!   reimplement the accept/reject loop in Python to drive the existing
//!   per-token pymethods, which is how the driver ends up as a second,
//!   ungated copy of the math this crate already gates.
//! * **The pyo3 seam (this module)** — the fit. Speculation is a
//!   MULTI-token generate call; the pyo3 boundary is where a multi-token
//!   generate call already belongs (`prefill` is the precedent), and it is the
//!   only place that can hold the target, the drafter and the loop at once.
//!
//! So the entry point is `VulkanModel::gemma_spec_generate`, a new pymethod, and
//! everything above the pyo3 marshalling lives in [`spec_decode_gemma`] so the
//! CI gate exercises the production body rather than a re-implementation of it.
//!
//! ## Scope: single-rank, CPU-reference target
//!
//! `spec_step` verifies through `Gemma4Model::forward_verify_core` — the
//! single-node CPU reference verify. The TP verify
//! (`forward_tp_gemma_verify_impl`) is a different backend needing a `Python`
//! token and an `all_reduce` callable, and the drafter's borrowed K/V is the one
//! thing TP shards (plan §1.5: it must be all-gathered). So this pymethod
//! REFUSES `tp_size > 1` with a message naming exactly what is missing, rather
//! than running a CPU verify over a sharded weight set and returning plausible
//! garbage. Generalising the driver over a verify backend is a follow-up; the
//! loop and the drafter coupling here are backend-independent already.

use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::gemma_assistant::{
    draft_block, expected_tensor_shapes, load_assistant_weights, AssistantConfig,
    AssistantSharedKv,
};
use crate::gemma_spec::{run_spec_decode_coupled, SpecConfig, SpecRun};
use crate::model::Gemma4Model;
use crate::VulkanModel;

/// Everything the drafter needs that does NOT change block to block: its
/// config and weights, the TARGET's embedding table (the drafter's input side
/// is the target's embedding, not its own — plan §1.1), and the borrowed-K/V
/// snapshot its Q-only cross-attention reads.
///
/// The K/V is a SNAPSHOT taken once, before generation, and never appended to
/// (plan §1.4). That is not an optimisation: the drafter attends over a fixed
/// context while the target's own cache advances underneath it, and re-reading
/// the target's live cache mid-block would change the drafter's answer
/// depending on how many drafts the previous block happened to have accepted.
pub struct SpecDrafter {
    pub cfg: AssistantConfig,
    pub weights: HashMap<String, Vec<f32>>,
    /// TARGET `model.embed_tokens.weight`, `[vocab, backbone_hidden_size]`.
    pub target_embed: Vec<f32>,
    pub embed_scale: f32,
    pub kv: AssistantSharedKv,
}

impl SpecDrafter {
    /// Take the borrowed-K/V snapshot from a target's live caches: the LAST
    /// layer of each attention type, in the `[seq_len, n_kv_heads, head_dim]`
    /// absolute layout `cpu_sdpa` reads directly.
    ///
    /// Every dimension check here is load-bearing. The drafter config and the
    /// target's layer geometry are pinned independently (`AssistantConfig::
    /// g31b_pair` vs the target checkpoint), so a mismatched pairing would
    /// otherwise surface as a `cpu_matmul` panic deep inside a draft step, or
    /// worse, as a numerically-fine forward over the wrong heads.
    pub fn borrow_kv(target: &Gemma4Model, acfg: &AssistantConfig) -> Result<AssistantSharedKv, String> {
        let t = &target.config;
        let last_sliding = (0..t.num_hidden_layers).rev().find(|&l| !t.is_full_attention(l))
            .ok_or_else(|| "spec drafter: target has no sliding-attention layer to borrow".to_string())?;
        let last_full = (0..t.num_hidden_layers).rev().find(|&l| t.is_full_attention(l))
            .ok_or_else(|| "spec drafter: target has no full-attention layer to borrow".to_string())?;
        let cs = &target.kv_caches[last_sliding];
        let cf = &target.kv_caches[last_full];
        if cs.has_wrapped() || cf.has_wrapped() {
            return Err(format!(
                "spec drafter: borrowed-KV needs the absolute-position layout, but the ring has \
                 wrapped (sliding layer {last_sliding}: seq_len {} > capacity {}; full layer \
                 {last_full}: seq_len {} > capacity {}). Set VLLM_VULKAN_KV_RING_DISABLE=1 to keep \
                 full-size caches.",
                cs.seq_len, cs.capacity, cf.seq_len, cf.capacity));
        }
        if cs.seq_len != cf.seq_len {
            return Err(format!(
                "spec drafter: borrowed layers are at different frontiers (sliding {} vs full {})",
                cs.seq_len, cf.seq_len));
        }
        let want = [
            ("sliding kv-heads", cs.num_kv_heads, acfg.num_key_value_heads),
            ("sliding head_dim", cs.head_dim, acfg.head_dim),
            ("full kv-heads", cf.num_kv_heads, acfg.num_global_key_value_heads),
            ("full head_dim", cf.head_dim, acfg.global_head_dim),
        ];
        for (what, got, expect) in want {
            if got != expect {
                return Err(format!(
                    "spec drafter: {what} mismatch — target has {got}, drafter config expects \
                     {expect}. The drafter checkpoint is not paired with this target."));
            }
        }
        Ok(AssistantSharedKv {
            sliding_k: cs.k_up_to_now().to_vec(),
            sliding_v: cs.v_up_to_now().to_vec(),
            full_k: cf.k_up_to_now().to_vec(),
            full_v: cf.v_up_to_now().to_vec(),
            kv_len: cs.seq_len,
        })
    }

    /// Load the drafter checkpoint from `dir` and snapshot the target's
    /// borrowed K/V at its CURRENT frontier (i.e. call this after the prompt
    /// has been prefilled, not before).
    pub fn load(dir: &str, target: &Gemma4Model) -> Result<Self, String> {
        let cfg = AssistantConfig::g31b_pair();
        let weights = load_assistant_weights(std::path::Path::new(dir))?;
        Self::from_parts(cfg, weights, target)
    }

    /// The loader-free constructor: the caller supplies the drafter config and
    /// weights (a checkpoint, or the synthetic CI fixture), this fills in
    /// everything read off the target.
    pub fn from_parts(
        cfg: AssistantConfig,
        weights: HashMap<String, Vec<f32>>,
        target: &Gemma4Model,
    ) -> Result<Self, String> {
        const EMBED: &str = "model.embed_tokens.weight";
        if !target.weights.contains(EMBED) {
            return Err(format!(
                "spec drafter: the target's host-f32 '{EMBED}' is not loaded, so the drafter's \
                 input embedding (the TARGET's table, plan §1.1) cannot be read. The g31b \
                 nvfp4/f16 loader keeps embed f16-host only; that path needs an f16 embed \
                 accessor, which this seam does not have yet."));
        }
        let backbone = cfg.backbone_hidden_size;
        if backbone != target.config.hidden_size {
            return Err(format!(
                "spec drafter: backbone_hidden_size {backbone} != target hidden_size {} — the \
                 drafter checkpoint is not paired with this target.",
                target.config.hidden_size));
        }
        if cfg.vocab_size != target.config.vocab_size {
            // The drafter samples ids from its OWN tied `lm_head` over
            // `cfg.vocab_size`, and `draft_block` then indexes the TARGET's
            // embedding table with them. A wider drafter vocab therefore fires
            // `draft_block`'s "token {t} is outside the target embed table"
            // assert MID-GENERATION, inside the pymethod, instead of failing
            // here at load with a pairing message.
            return Err(format!(
                "spec drafter: vocab_size {} != target vocab_size {} — the drafter samples ids \
                 that index the TARGET's embedding table, so the two vocabularies must be the \
                 same. The drafter checkpoint is not paired with this target.",
                cfg.vocab_size, target.config.vocab_size));
        }
        // The drafter's OWN tensors. `load_assistant_weights` is a plain
        // safetensors read with no shape opinion, and `assistant_forward`'s `w`
        // / `amv` PANIC on a name they cannot find — through the pyo3 boundary,
        // mid-`gemma_spec_generate`, several blocks into a generation. A
        // wrong-sized tensor is worse: it reaches `cpu_matmul`'s dimension
        // check (a panic) or, on the GPU path, a matvec that reads whatever is
        // resident. `expected_tensor_shapes` is the checkpoint tensor map and
        // is gated as COMPLETE and EXACT for what the forward reads
        // (`gemma_assistant::tests::gemma_assistant_synthetic_fixture_covers_
        // the_checkpoint_tensor_map`), so checking it here converts every one of
        // those into a load-time `Err` that NAMES the tensor.
        for (name, shape) in expected_tensor_shapes(&cfg) {
            let want: usize = shape.iter().product();
            match weights.get(&name) {
                None => return Err(format!(
                    "spec drafter: required tensor '{name}' (shape {shape:?}) is missing from the \
                     drafter checkpoint. The drafter is incomplete or not paired with this \
                     target.")),
                Some(v) if v.len() != want => return Err(format!(
                    "spec drafter: tensor '{name}' has {} elements, expected {want} (shape \
                     {shape:?}). The drafter checkpoint is not paired with this target.",
                    v.len())),
                Some(_) => {}
            }
        }
        let kv = Self::borrow_kv(target, &cfg)?;
        Ok(SpecDrafter {
            target_embed: target.weights.f32_slice(EMBED).to_vec(),
            embed_scale: target.config.embed_scale,
            cfg, weights, kv,
        })
    }
}

/// What a production spec-decode call returns. The token stream plus the
/// MEASURED acceptance — see [`SpecRun`] for why the stream alone is not
/// enough to tell a working speculative decoder from a disengaged one.
#[derive(Debug, Clone)]
pub struct SpecDecodeReport {
    pub tokens: Vec<u32>,
    pub blocks: usize,
    pub drafted: usize,
    pub accepted: usize,
}

impl SpecDecodeReport {
    pub fn accept_rate(&self) -> Option<f32> {
        if self.drafted == 0 { None } else { Some(self.accepted as f32 / self.drafted as f32) }
    }
    /// True when speculation actually contributed. `false` with `drafted > 0`
    /// means the lever ran and bought nothing.
    pub fn engaged(&self) -> bool {
        self.accepted > 0
    }
}

/// Where the drafter's step-0 `recurrent_hidden` comes from (plan §1.1: the
/// target hidden that produced this block's bonus token).
pub enum SeedSource<'a> {
    /// RECOMPUTE it from the target, which is what makes this seam work with no
    /// change to `forward_verify_core`.
    ///
    /// The seed for a block whose bonus sits at position `pos` is the hidden at
    /// position `pos - 1`. The batched verify computes exactly that hidden
    /// internally and returns only logits, so it cannot be read back — and the
    /// TP path's `gemma_hidden_ring` is not populated by the CPU verify (and
    /// stashes the PRE-final-norm hidden, a different vector from the one the
    /// validated drafter fixture is built against; see the module report).
    ///
    /// The recompute is STATE-NEUTRAL and costs ONE single-token target forward
    /// per BLOCK (not per token): rewind every KV cache to `pos - 1`, re-run
    /// `forward_with_normed(prev_token, pos - 1)`, and the append lands in the
    /// same ring slot it already occupied, with the same token at the same
    /// position over the same prior cache — so the bytes written are the bytes
    /// that were there, and the frontier ends where it started.
    Recompute,
    /// The caller already HAS the producing hidden (a driver that kept it from
    /// its own prefill/verify) and hands it over by absolute position. Also how
    /// the CI gate feeds the seeds its greedy baseline recorded, which is what
    /// proves `Recompute` reproduces them.
    ByPosition(&'a dyn Fn(usize) -> Result<Vec<f32>, String>),
}

/// Recompute the target hidden at absolute position `pos`, leaving the target's
/// KV caches EXACTLY as they were found.
///
/// Precondition, CHECKED rather than assumed: every cache frontier is at
/// `pos + 1` (the position was just committed) and `token` is the token at
/// `pos`. Both hold at a block boundary. It is checked because `start_pos`
/// reaches this from Python: a caller whose position cursor has drifted from
/// the KV frontier would otherwise silently rewind the cache to the WRONG place
/// and keep generating plausible tokens from a corrupted context.
pub fn recompute_seed_hidden(
    target: &mut Gemma4Model,
    token: u32,
    pos: usize,
) -> Result<Vec<f32>, String> {
    let before: Vec<usize> = target.kv_caches.iter().map(|c| c.seq_len).collect();
    if let Some((li, &n)) = before.iter().enumerate().find(|(_, &n)| n != pos + 1) {
        return Err(format!(
            "recompute_seed_hidden: layer {li}'s KV frontier is {n}, expected {} (position \
             {pos} just committed). The caller's position cursor and the target's KV have \
             drifted apart.", pos + 1));
    }
    for cache in target.kv_caches.iter_mut() {
        cache.truncate(pos);
    }
    let (_, normed) = target.forward_with_normed(token, pos);
    debug_assert!(target.kv_caches.iter().map(|c| c.seq_len).eq(before.iter().copied()),
                  "recompute_seed_hidden must leave every KV frontier where it found it");
    Ok(normed)
}

/// THE PRODUCTION DRIVER. Generates exactly `cfg.max_new_tokens` tokens from
/// `start_bonus` at `start_pos`, drafting with the real EAGLE drafter and
/// verifying with the target's batched verify.
///
/// `prompt_last_token` is the token at `start_pos - 1` — the one whose hidden
/// seeds the FIRST block. It is a parameter rather than something read off the
/// model because the target's KV holds K/V, not token ids.
///
/// Output-identical to greedy at temperature 0 BY CONSTRUCTION: every committed
/// token is either the target's own argmax (the bonus, and the new bonus read
/// off `logits[accept_len]`) or a drafted token that MATCHED the target's
/// argmax at that position. Speculation changes how many target forwards it
/// takes to produce the stream, never the stream.
pub fn spec_decode_gemma(
    target: &mut Gemma4Model,
    drafter: &SpecDrafter,
    seed: SeedSource<'_>,
    prompt_last_token: u32,
    start_bonus: u32,
    start_pos: usize,
    cfg: &SpecConfig,
) -> Result<SpecDecodeReport, String> {
    if start_pos == 0 {
        return Err("spec_decode_gemma: start_pos must be >= 1 (the first block's drafter seed is \
                    the hidden at start_pos-1, which needs a prefilled prompt)".to_string());
    }
    if cfg.max_new_tokens == 0 {
        return Ok(SpecDecodeReport { tokens: Vec::new(), blocks: 0, drafted: 0, accepted: 0 });
    }

    // ── The seed precondition, checked HERE rather than discovered inside the
    //    loop ────────────────────────────────────────────────────────────────
    //
    // `recompute_seed_hidden` has exactly ONE failure mode: the caller's
    // position cursor has drifted from the target's KV frontier. It refuses
    // BEFORE it truncates anything, so the refusal itself is state-neutral —
    // but the closure below cannot stop `run_spec_decode_coupled`, which then
    // runs every REMAINING block on filler drafts and ADVANCES the target's KV
    // and frontier before `first_error` is re-raised. A caller that catches the
    // error and retries would then be decoding from a context that moved.
    //
    // Hoisting the check makes that unreachable for the production seed source:
    //   * it holds (or does not) at the FIRST block, and
    //   * `spec_step_coupled` ends every block with `verify_rollback` leaving
    //     the frontier at `pos + committed.len()`, which is exactly the next
    //     block's `pos` — so if the invariant holds at block 0 it holds at
    //     every block.
    // Failing here returns with the target EXACTLY as the caller handed it over.
    //
    // `SeedSource::ByPosition` is caller-supplied and can fail at any block;
    // that is a Rust-API-only path (the pymethod hardcodes `Recompute`) and it
    // cannot be stopped mid-loop without giving `draft_fn` a `Result` return,
    // which is the INC-5a contract every stub-drafter call site is written
    // against. Its error message below says the state advanced instead.
    if matches!(&seed, SeedSource::Recompute) {
        if let Some((li, n)) = target.kv_caches.iter().map(|c| c.seq_len).enumerate()
            .find(|&(_, n)| n != start_pos)
        {
            return Err(format!(
                "spec_decode_gemma: layer {li}'s KV frontier is {n}, expected start_pos \
                 {start_pos}. The caller's position cursor and the target's KV have drifted \
                 apart; nothing was generated and the target is untouched."));
        }
    }

    // Errors raised inside the drafter closure cannot propagate through
    // `run_spec_decode_coupled` (its `draft_fn` returns draft ids, and giving it
    // a Result would change the INC-5a contract for every stub-drafter call
    // site). They are stashed here and re-raised after the loop instead.
    //
    // The failure draft must still be EXACTLY `k` wide. The block width is the
    // DRIVER's choice (`min(k, remaining - 1)`), not the closure's, and
    // `spec_step_coupled` asserts `draft.len() == k` — so returning an empty
    // draft for a `k > 0` block PANICS through the pyo3 boundary before
    // `first_error` can be turned into a `PyRuntimeError`. `bonus` is a valid
    // token id (the target's own argmax), and the whole run is discarded by the
    // `first_error` check below, so a filler draft is inert: it only keeps the
    // loop legal long enough to reach that check.
    let mut first_error: Option<String> = None;

    let run: SpecRun = run_spec_decode_coupled(
        target,
        |model, emitted, bonus, pos, k| {
            if k == 0 {
                return Vec::new();
            }
            if first_error.is_some() {
                return vec![bonus; k];
            }
            // The token at `pos - 1`: the previous block's last committed token,
            // or the prompt's last token on the first block.
            let prev_token = emitted.last().copied().unwrap_or(prompt_last_token);
            let seed_hidden = match &seed {
                SeedSource::Recompute => recompute_seed_hidden(model, prev_token, pos - 1),
                SeedSource::ByPosition(f) => f(pos - 1),
            };
            let seed_hidden = match seed_hidden {
                Ok(h) => h,
                Err(e) => { first_error = Some(e); return vec![bonus; k]; }
            };
            draft_block(
                &drafter.cfg, &drafter.weights, &drafter.target_embed, drafter.embed_scale,
                &drafter.kv, bonus, &seed_hidden, pos, k, None, None,
            ).into_iter().map(|(t, _)| t).collect()
        },
        start_bonus, start_pos, cfg,
    );

    if let Some(e) = first_error {
        // Only `SeedSource::ByPosition` can get here — the `Recompute`
        // precondition is checked above, before any block runs. The loop could
        // not be stopped at the failing block, so the target's KV and frontier
        // HAVE advanced past `start_pos` on filler drafts. Say so: the tokens
        // are discarded, but the model is not where the caller left it and must
        // be re-prefilled, not retried in place.
        return Err(format!(
            "spec_decode_gemma: drafter seed unavailable: {e}. The target's KV and position \
             frontier ADVANCED before this could be raised — discard the target's generation \
             state and re-prefill; do not retry in place."));
    }
    let (blocks, drafted, accepted) = (run.blocks(), run.drafted(), run.accepted());
    Ok(SpecDecodeReport { tokens: run.tokens, blocks, drafted, accepted })
}

#[pymethods]
impl VulkanModel {
    /// EAGLE speculative decode, `VLLM_VULKAN_GEMMA_SPEC=1` (DEFAULT OFF).
    ///
    /// Generates exactly `max_new_tokens` tokens starting from `start_token` at
    /// `start_pos`, and returns `(tokens, blocks, drafted, accepted)`.
    ///
    /// `accepted` is the point. At temperature 0 the token stream is identical
    /// whether every draft was accepted or none was, so a caller that only
    /// reads `tokens` cannot tell whether it bought anything — this project has
    /// shipped a lever that measured as a LOSS because it was inert, and a
    /// spec-decode gate that stayed green at zero acceptance. `accepted == 0`
    /// with `drafted > 0` means the drafter and target disagree on everything
    /// and this call was SLOWER than plain greedy decode.
    ///
    /// Refusals, all deliberate and all loud — the flag being off must never
    /// mean "quietly ran greedy instead":
    ///   * flag off,
    ///   * no `VLLM_VULKAN_GEMMA_SPEC_ASSISTANT_DIR`,
    ///   * `tp_size > 1` (see the module doc: sharded verify + un-gathered
    ///     borrowed K/V),
    ///   * `start_pos == 0` (no prompt to seed the first block from).
    #[pyo3(signature = (prompt_last_token, start_token, start_pos, max_new_tokens))]
    fn gemma_spec_generate(
        &mut self,
        prompt_last_token: u32,
        start_token: u32,
        start_pos: usize,
        max_new_tokens: usize,
    ) -> PyResult<(Vec<u32>, usize, usize, usize)> {
        let flags = crate::flags::flags_global();
        if !flags.gemma_spec {
            return Err(PyRuntimeError::new_err(
                "gemma_spec_generate: speculative decode is OFF. Set VLLM_VULKAN_GEMMA_SPEC=1 to \
                 enable it. This call refuses rather than falling back to greedy decode: the two \
                 produce the SAME tokens, so a silent fallback is indistinguishable from a working \
                 speculative decoder."));
        }
        let dir = flags.gemma_spec_assistant_dir.clone().ok_or_else(|| PyRuntimeError::new_err(
            "gemma_spec_generate: set VLLM_VULKAN_GEMMA_SPEC_ASSISTANT_DIR to the drafter \
             checkpoint directory (holding model.safetensors). There is no default path."))?;
        if self.tp_size > 1 {
            return Err(PyRuntimeError::new_err(format!(
                "gemma_spec_generate: tp_size={} is not supported yet. This seam verifies through \
                 the single-node Gemma4Model::forward_verify_core, which is wrong over a sharded \
                 weight set, and the drafter's borrowed K/V is the one thing TP shards (plan §1.5 \
                 requires an all-gather). Run single-rank, or drive the TP verify pymethods \
                 (forward_tp_gemma_verify_argmax + gemma_tp_verify_rollback) from the driver.",
                self.tp_size)));
        }

        let drafter = SpecDrafter::load(&dir, &self.inner).map_err(PyRuntimeError::new_err)?;
        let cfg = SpecConfig { k: flags.gemma_spec_k, max_new_tokens };
        let report = spec_decode_gemma(
            &mut self.inner, &drafter, SeedSource::Recompute,
            prompt_last_token, start_token, start_pos, &cfg,
        ).map_err(PyRuntimeError::new_err)?;

        eprintln!(
            "[VLLM_VULKAN_GEMMA_SPEC] k={} n={} blocks={} drafted={} accepted={} accept_rate={}",
            cfg.k, report.tokens.len(), report.blocks, report.drafted, report.accepted,
            match report.accept_rate() { Some(r) => format!("{r:.3}"), None => "n/a".to_string() });

        Ok((report.tokens, report.blocks, report.drafted, report.accepted))
    }
}
