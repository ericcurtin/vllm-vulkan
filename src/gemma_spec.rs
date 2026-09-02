//! EAGLE spec-decode DRIVER logic (INC-5a, spec §"EAGLE spec-decode driver").
//!
//! This is the CPU accept/reject/rollback BOOKKEEPING only — it drives
//! `Gemma4Model::forward_verify_core` / `verify_rollback` (INC-4) with a
//! caller-supplied drafter, but does no drafting, GPU dispatch, or TP/cluster
//! plumbing itself. The real EAGLE 0.5B-drafter-to-31B-target coupling (KV
//! borrow, hidden-state hookup) is INC-5b, a separate follow-up; here the
//! drafter is injected as a plain closure so the SAME loop drives both a CPU
//! stub (this file's gate) and, later, the real drafter / cluster TP path
//! without any change to the accept/reject math.
//!
//! Design-A loop (plan pseudocode, `GEMMA31B_SPEC_PLAN.md` §INC-5):
//! ```text
//! loop over generation:
//!   draft[0..K]  = drafter.draft_block(bonus, target_hidden, shared_kv_snapshot, pos)
//!   logits[0..K] = forward_verify_core([bonus, draft[0..K-1]], R)   # target, batched
//!   accept_len   = longest prefix where argmax(logits[i]) == draft[i]
//!   new_bonus    = argmax(logits[accept_len])
//!   verify_rollback(R, K, accept_len)                                # KV rewind
//!   emit committed tokens; R += accept_len + 1
//! ```
//! `K` here is the number of DRAFTED tokens per block (block size = K+1,
//! counting the bonus token); `forward_verify_core` is called with exactly
//! `K+1` tokens (bonus + K drafts), matching its existing `[T][vocab]`
//! contract (INC-4).
//!
//! THE PRODUCTION CALLER IS `gemma_spec_wire`. This module was test-only
//! through INC-5a (the accept/reject math, landed and gated on its own with a
//! CPU stub drafter); `gemma_spec_wire::spec_decode_gemma` is INC-5b, wiring
//! the real EAGLE drafter (`gemma_assistant.rs`) into this loop behind
//! `VLLM_VULKAN_GEMMA_SPEC` (default OFF).
//!
//! Two entry points, and the difference matters:
//!   * [`spec_step`] / [`run_spec_decode`] take a `FnMut(u32, usize, usize)`
//!     drafter — enough for a stub, and what the INC-5a gates drive.
//!   * [`spec_step_coupled`] / [`run_spec_decode_coupled`] hand the drafter the
//!     TARGET and the committed stream, and return per-block acceptance. The
//!     real drafter needs the target (its seed and its borrowed K/V come from
//!     it) and the caller needs the acceptance (the token stream is identical
//!     whether speculation worked or not). The first pair are adapters over
//!     the second.

use crate::model::{argmax, Gemma4Model};

/// Per-block driver config: `k` drafted tokens per block, generate EXACTLY
/// `max_new_tokens` committed tokens.
///
/// `max_new_tokens` is a hard cap, not a floor. A block commits between 1 and
/// `k + 1` tokens (the bonus plus the accepted drafts), so a `k` that does not
/// divide the budget would overshoot if the driver always drafted the full
/// width — `run_spec_decode` narrows the last block's draft width instead. The
/// returned stream never needs truncating by the caller.
#[derive(Debug, Clone, Copy)]
pub struct SpecConfig {
    pub k: usize,
    pub max_new_tokens: usize,
}

/// Outcome of one draft -> batched-verify -> accept/rollback block.
#[derive(Debug, Clone)]
pub struct SpecStepResult {
    /// Number of drafted tokens accepted (0..=k).
    pub accept_len: usize,
    /// Tokens actually committed this block: `bonus` followed by
    /// `draft[0..accept_len]` (length `accept_len + 1`).
    pub committed: Vec<u32>,
    /// Greedy-argmax continuation token to feed as next block's bonus.
    pub new_bonus: u32,
}

/// Runs ONE draft/verify/accept/rollback block starting from `bonus` at
/// position `pos`. `draft_fn(bonus, pos, k)` must return exactly `k` drafted
/// token ids — it is the abstracted drafter (stub in the CPU gate below,
/// EAGLE 0.5B model in INC-5b).
///
/// Mirrors the plan pseudocode exactly: batched-verify over
/// `[bonus, draft[0..k-1]]` (T = k+1 tokens fed at `pos..pos+k`), longest
/// matching prefix -> `accept_len`, `new_bonus` read off `logits[accept_len]`
/// (valid since `forward_verify_core` returns `k+1` rows, indices `0..=k`),
/// then `verify_rollback` rewinds every layer's KV frontier to
/// `pos + accept_len + 1` (a no-op on full accept, since `accept_len < k+1 =
/// t` always holds — see `verify_rollback`'s own assert/no-op contract).
///
/// `k == 0` is LEGAL and is what `run_spec_decode` passes when only one token
/// of budget is left: no drafts, `t == 1`, verify the bonus alone, commit it,
/// and `verify_rollback(pos, 1, 0)` is its own no-op case. The block still
/// makes one token of progress, so the driver loop cannot spin.
pub fn spec_step<F>(
    model: &mut Gemma4Model,
    mut draft_fn: F,
    bonus: u32,
    pos: usize,
    k: usize,
) -> SpecStepResult
where
    F: FnMut(u32, usize, usize) -> Vec<u32>,
{
    spec_step_coupled(model, |_m, _emitted, b, p, kk| draft_fn(b, p, kk), &[], bonus, pos, k)
}

/// [`spec_step`] with the TARGET handed to the drafter.
///
/// The extra `&mut Gemma4Model` and `emitted` arguments are what a REAL drafter
/// needs and the closure-only form cannot express. The EAGLE drafter is not a
/// standalone LM: per `GEMMA31B_SPEC_PLAN.md` §1.1 its step-0 `recurrent_hidden`
/// is *the target's own hidden state that produced this block's bonus token*,
/// and its cross-attention reads *the target's* borrowed K/V. The test stub
/// drafters need none of that, which is exactly why the `FnMut(u32, usize,
/// usize)` shape was enough to land INC-5a and NOT enough to wire a production
/// caller: a closure that captures the target cannot coexist with the `&mut
/// Gemma4Model` this function already holds.
///
/// So this — not `spec_step` — is the primary; `spec_step` is the adapter that
/// discards both extra arguments, keeping every existing stub-drafter call site
/// (and the whole INC-5a gate) compiling and behaving identically.
///
/// `emitted` is the committed stream SO FAR (empty on the first block). A
/// production drafter needs it to name the token at `pos - 1`, the one whose
/// hidden state seeds this block; see `gemma_spec_wire::recompute_seed_hidden`.
pub fn spec_step_coupled<F>(
    model: &mut Gemma4Model,
    mut draft_fn: F,
    emitted: &[u32],
    bonus: u32,
    pos: usize,
    k: usize,
) -> SpecStepResult
where
    F: FnMut(&mut Gemma4Model, &[u32], u32, usize, usize) -> Vec<u32>,
{
    let draft = draft_fn(model, emitted, bonus, pos, k);
    assert_eq!(draft.len(), k, "draft_fn must return exactly k={k} draft ids");

    let mut tokens = Vec::with_capacity(k + 1);
    tokens.push(bonus);
    tokens.extend_from_slice(&draft);
    let t = tokens.len(); // k + 1

    let logits = model.forward_verify_core(&tokens, pos);
    assert_eq!(logits.len(), t, "forward_verify_core must return t={t} rows");

    // Longest prefix where the target's own greedy argmax at row i agrees
    // with the drafted token at position i+1 (row i predicts tokens[i+1]).
    let mut accept_len = 0usize;
    while accept_len < k && argmax(&logits[accept_len]) as u32 == draft[accept_len] {
        accept_len += 1;
    }
    let new_bonus = argmax(&logits[accept_len]) as u32;

    // KV-counter rewind to the accepted frontier (no-op on full accept).
    model.verify_rollback(pos, t, accept_len);

    let mut committed = Vec::with_capacity(accept_len + 1);
    committed.push(bonus);
    committed.extend_from_slice(&draft[..accept_len]);

    SpecStepResult { accept_len, committed, new_bonus }
}

/// Runs the full accept/reject generation loop, block after block, until
/// EXACTLY `cfg.max_new_tokens` tokens have been committed. Returns the
/// concatenated committed token stream (bonus + accepted drafts of every
/// block) — this is what should equal a SPEC-off greedy run's token stream
/// when the drafter always drafts the target's own argmax (the identity
/// gate), and it still must equal that same greedy stream when the drafter
/// mis-drafts a suffix of a block (the partial-accept gate), since the
/// committed prefix is by construction the same tokens the greedy baseline
/// would have produced.
///
/// THE BUDGET INVARIANT: a block commits `accept_len + 1` tokens, up to
/// `k + 1` on a full accept — the drafted width plus the BONUS. So the draft
/// width of each block is narrowed to `min(k, remaining - 1)`, one less than
/// the remaining budget, and the loop cannot overshoot. Drafting `cfg.k`
/// unconditionally returned up to `k` tokens too many (`k = 4`,
/// `max_new_tokens = 8` gave 10), which for a real caller is generation past
/// the requested length, not a harmless overshoot.
///
/// `remaining == 1` narrows to a width-0 block. That is a legal, PROGRESSING
/// block, not a stall: `spec_step` still verifies the single bonus token,
/// commits it, and the loop ends. `remaining` is >= 1 at the top of every
/// iteration, so `remaining - 1` never underflows.
pub fn run_spec_decode<F>(
    model: &mut Gemma4Model,
    mut draft_fn: F,
    start_bonus: u32,
    start_pos: usize,
    cfg: &SpecConfig,
) -> Vec<u32>
where
    F: FnMut(u32, usize, usize) -> Vec<u32>,
{
    run_spec_decode_coupled(
        model,
        |_m, _emitted, b, p, k| draft_fn(b, p, k),
        start_bonus, start_pos, cfg,
    ).tokens
}

/// What one `run_spec_decode_coupled` run actually DID, not just what it
/// emitted.
///
/// `run_spec_decode` returns the token stream alone, and that stream is
/// IDENTICAL whether every draft was accepted or none was — the accept/reject
/// math guarantees it. So a caller holding only the stream cannot tell a
/// working speculative decoder from one that is fully disengaged, and neither
/// can a test: `gemma_assistant`'s negative control is a run whose stream is
/// bit-identical to the greedy baseline at ZERO acceptance. The measured
/// quantities below are the only thing that separates the two, which is why
/// they are part of the return value rather than something a test reconstructs
/// by snooping block widths from inside its own `draft_fn` closure.
#[derive(Debug, Clone, Default)]
pub struct SpecRun {
    /// The committed token stream — exactly what `run_spec_decode` returns.
    pub tokens: Vec<u32>,
    /// Draft width offered per block (`min(k, remaining - 1)`; the last block
    /// may be narrowed, and a width-0 block is legal).
    pub block_widths: Vec<usize>,
    /// Accepted draft count per block (`0..=block_widths[b]`).
    pub block_accepts: Vec<usize>,
}

impl SpecRun {
    /// Number of draft/verify/accept blocks the run took.
    pub fn blocks(&self) -> usize {
        self.block_widths.len()
    }
    /// Total drafted tokens OFFERED to the target across the run.
    pub fn drafted(&self) -> usize {
        self.block_widths.iter().sum()
    }
    /// Total drafted tokens ACCEPTED. Zero means speculation contributed
    /// nothing: every block fell back to committing its bonus token alone,
    /// which is slower than plain greedy decode, not faster.
    pub fn accepted(&self) -> usize {
        self.block_accepts.iter().sum()
    }
    /// Accepted / offered. `None` when nothing was offered (`k == 0`), which is
    /// deliberately NOT 0.0 — "no drafts offered" and "every draft rejected"
    /// are different failures and must not read the same on a dashboard.
    pub fn accept_rate(&self) -> Option<f32> {
        let d = self.drafted();
        if d == 0 { None } else { Some(self.accepted() as f32 / d as f32) }
    }
    /// True when speculation actually did something this run. A caller that
    /// enabled the lever and gets `false` has a silently inert lever.
    pub fn engaged(&self) -> bool {
        self.accepted() > 0
    }
}

/// [`run_spec_decode`] with the target handed to the drafter (see
/// [`spec_step_coupled`]) and the per-block acceptance SURFACED.
///
/// This is the primary implementation; `run_spec_decode` is the adapter that
/// drops both. It is what a production caller drives — `gemma_spec_wire`.
pub fn run_spec_decode_coupled<F>(
    model: &mut Gemma4Model,
    mut draft_fn: F,
    start_bonus: u32,
    start_pos: usize,
    cfg: &SpecConfig,
) -> SpecRun
where
    F: FnMut(&mut Gemma4Model, &[u32], u32, usize, usize) -> Vec<u32>,
{
    let mut pos = start_pos;
    let mut bonus = start_bonus;
    let mut run = SpecRun::default();
    while run.tokens.len() < cfg.max_new_tokens {
        let remaining = cfg.max_new_tokens - run.tokens.len(); // >= 1 here
        // One less than the budget: the block also commits the bonus token.
        let step_k = cfg.k.min(remaining - 1);
        // `draft_fn` needs the stream so far; the run's own `tokens` vec is it.
        let emitted = std::mem::take(&mut run.tokens);
        let step = spec_step_coupled(model, &mut draft_fn, &emitted, bonus, pos, step_k);
        run.tokens = emitted;
        debug_assert!(!step.committed.is_empty(), "every block must commit the bonus token");
        debug_assert!(step.committed.len() <= remaining,
                      "block committed {} > remaining budget {remaining}", step.committed.len());
        debug_assert_eq!(step.committed.len(), step.accept_len + 1,
                      "a block commits its bonus plus its accepted drafts");
        run.block_widths.push(step_k);
        run.block_accepts.push(step.accept_len);
        pos += step.committed.len();
        run.tokens.extend(step.committed);
        bonus = step.new_bonus;
    }
    run
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{load_gemma_mlx_affine, tiny_synthetic_gemma, Gemma4Config, Gemma4Weights, KvCache, SimpleTensor};

    /// Cosine similarity between two logit rows.
    ///
    /// Returns 0.0 — NOT NaN — when either vector has zero norm, so a
    /// degenerate row reads as "no similarity" and fails a `>= threshold`
    /// gate rather than propagating NaN through it.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { return 0.0; }
        dot / (na * nb)
    }

    /// Loads the real gemma-4-12B checkpoint as a proxy for the g31b target
    /// (same forward primitives; see `model.rs`'s
    /// `load_gemma12b_for_verify_gate` for the INC-4 precedent this mirrors).
    /// The 12B's own last-layer KV dims don't line up with the 0.5B EAGLE
    /// drafter's borrowed-KV expectation (that coupling is INC-5b), so this
    /// gate exercises the driver with a STUB drafter instead of the real one,
    /// per the plan.
    ///   GEMMA12B_DIR=<checkpoint dir>
    ///
    /// Two DIFFERENT outcomes, deliberately. No `GEMMA12B_DIR` at all means the
    /// gate was not requested: print a visible SKIP and return `None`, so an
    /// un-run test is distinguishable from a green `ok` in the log. But
    /// `GEMMA12B_DIR` SET and pointing somewhere with no checkpoint means the
    /// gate WAS requested and cannot run — that FAILS, loudly. Skipping there
    /// would report success for a test nobody actually executed, which is the
    /// same defect as a silent no-op pass, just wearing the skip's clothes.
    /// Same rule as `gemma_assistant.rs`'s `assistant_dir`, but the presence
    /// check is "any `*.safetensors` in the directory", NOT the literal
    /// `model.safetensors` that helper looks for: the 12B is a SHARDED
    /// checkpoint, and `load_gemma_mlx_affine` reaches it through
    /// `discover_shards`, which globs every `*.safetensors` sibling and never
    /// opens `model.safetensors` itself. Asserting on that one filename would
    /// reject a perfectly good `model-0000N-of-...safetensors` checkpoint.
    fn load_gemma12b(max_seq: usize) -> Option<Gemma4Model> {
        let dir = match std::env::var("GEMMA12B_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("SKIP gemma_spec driver gate: set GEMMA12B_DIR");
                return None;
            }
        };
        let has_shard = std::fs::read_dir(&dir)
            .map(|entries| entries.flatten().any(|e|
                e.path().extension().and_then(|x| x.to_str()) == Some("safetensors")))
            .unwrap_or(false);
        assert!(
            has_shard,
            "GEMMA12B_DIR={dir} holds no *.safetensors shard — the checkpoint path is wrong; \
             refusing to silently skip a test that was explicitly requested");
        let cfg = Gemma4Config::g12b();
        let tensors = load_gemma_mlx_affine(std::path::Path::new(&dir)).expect("load checkpoint");
        let weights = Gemma4Weights {
            tensors: tensors
                .into_iter()
                .map(|(k, v)| (k, SimpleTensor { shape: vec![v.len()], data: v }))
                .collect(),
        };
        let kv_caches = (0..cfg.num_hidden_layers)
            .map(|l| KvCache::new(max_seq, cfg.layer_num_kv_heads(l), cfg.layer_head_dim(l)))
            .collect();
        Some(Gemma4Model { config: cfg, weights, kv_caches })
    }

    /// Drop and REBUILD every layer's KV cache, so a gate can replay the same
    /// prompt from a genuinely clean state.
    ///
    /// A rebuild, not `truncate(0)`: truncation only rewinds the counter and
    /// leaves the old K/V bytes in place, which is correct for spec-decode
    /// rollback but would let stale rows leak into a comparison run if any
    /// path ever read past `seq_len`. Rebuilding also restores each layer's
    /// own `layer_num_kv_heads`/`layer_head_dim` sizing, which differs between
    /// sliding and full layers.
    fn reset_kv_caches(model: &mut Gemma4Model, max_seq: usize) {
        let cfg = &model.config;
        model.kv_caches = (0..cfg.num_hidden_layers)
            .map(|l| KvCache::new(max_seq, cfg.layer_num_kv_heads(l), cfg.layer_head_dim(l)))
            .collect();
    }

    /// The exact sequence of draft WIDTHS `run_spec_decode` must request when
    /// every drafted token is accepted — i.e. the block structure of a
    /// full-accept run, derived from the driver's own budget invariant
    /// (`step_k = min(k, remaining - 1)`, a full-accept block commits
    /// `step_k + 1`).
    ///
    /// This exists so the identity gates can assert ACCEPTANCE, not just token
    /// identity. `run_spec_decode` returns only the committed stream and does
    /// not report `accept_len` to its caller (`spec_step` does, but the loop
    /// consumes it). Rather than widen the driver's production signature just
    /// to make a test easier, the gates observe acceptance through the drafter
    /// closure, which the driver calls exactly ONCE per block and hands that
    /// block's width: recording the widths yields both the block count and the
    /// number of tokens offered, and with a stream of known length `n` that is
    /// enough to pin the accepted count exactly. Each block commits one bonus
    /// token plus its accepted drafts, so over the whole run
    /// `accepted == n - block_count` — and a full-accept drafter must have
    /// `accepted == sum(widths)`.
    fn expected_full_accept_widths(k: usize, max_new_tokens: usize) -> Vec<usize> {
        let mut widths = Vec::new();
        let mut remaining = max_new_tokens;
        while remaining > 0 {
            let w = k.min(remaining - 1);
            widths.push(w);
            remaining -= w + 1; // a full-accept block commits width + bonus
        }
        widths
    }

    /// Precomputes a SPEC-off serial greedy baseline of `n` tokens starting
    /// from `start_bonus` at `start_pos`, driving the model with plain
    /// `forward` calls only (no verify/rollback machinery at all) — this is
    /// the ground truth both gates below compare the spec-decode driver
    /// against.
    fn greedy_baseline(model: &mut Gemma4Model, start_bonus: u32, start_pos: usize, n: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity(n);
        let mut tok = start_bonus;
        let mut pos = start_pos;
        for _ in 0..n {
            ids.push(tok);
            let logits = model.forward(tok, pos);
            tok = argmax(&logits) as u32;
            pos += 1;
        }
        ids
    }

    /// INC-5a gate 1 ("identity gate"): a STUB drafter that always drafts
    /// exactly what the greedy baseline would have produced next (forced ==
    /// target argmax) must make the spec-decode driver's committed token
    /// stream identical to the SPEC-off greedy stream over >=32 tokens, with
    /// every block's `accept_len` maximal (== k, full acceptance).
    ///
    /// ONE checkpoint load only (the ~44GB host-f32 dequant of the 12B
    /// checkpoint dominates wall time — see `load_gemma_mlx_affine`'s doc
    /// comment on `load_gemma_resident_weights` — so, exactly like the INC-4
    /// gates, the greedy baseline is computed on the SAME loaded model, then
    /// the KV state is reset and the short prompt prefix replayed before
    /// driving the spec-decode loop from a clean, comparable starting point.
    ///
    /// Real-checkpoint variant of the synthetic gate below (~30 min/run on the
    /// full gemma-4-12B f32 checkpoint) — kept for optional real-weight
    /// confidence, `#[ignore]`d by default so `cargo test` stays fast.
    #[test]
    #[ignore]
    fn gemma31b_spec_driver_identity_matches_greedy_real12b() {
        let max_seq = 64usize;
        let k = 4usize;
        let n = 32usize; // >= 32 committed tokens required by the gate
        let mut model = match load_gemma12b(max_seq) { Some(m) => m, None => return };

        let prefix = [2u32, 1024, 2048];
        let r0 = prefix.len();
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        // `bonus` is a NEW token that has not yet been forwarded (its KV gets
        // appended by the first `forward_verify_core` call, at `start_pos`) —
        // it plays the same role `verify_tokens[0]` plays in the INC-4 gate,
        // distinct from the already-forwarded `prefix`.
        let start_bonus = 4096u32;
        let start_pos = r0;

        // Ground truth: greedy baseline over a small margin beyond `n` (the
        // stub drafter reads ahead by up to `k` tokens per block, and blocks
        // can overshoot `n` by up to `k`, so one block's worth of slack is
        // enough). Computed on `model` directly — no second checkpoint load.
        let baseline = greedy_baseline(&mut model, start_bonus, start_pos, n + k + 2);

        // Reset KV state and replay the same short prefix so the spec-decode
        // driver starts from the identical clean state the baseline did.
        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }

        // Stub drafter: forces draft[i] = baseline[pos_offset+1+i], i.e.
        // exactly the greedy continuation of whatever bonus token led into
        // this block. `baseline` is indexed relative to `start_pos`.
        // `widths` records the width the driver asked for on each block — one
        // push per block. See `expected_full_accept_widths` for why acceptance
        // is observed here rather than through the driver's return type.
        let mut widths: Vec<usize> = Vec::new();
        let draft_fn = |_bonus: u32, pos: usize, k: usize| -> Vec<u32> {
            widths.push(k);
            let off = pos - start_pos;
            (0..k).map(|i| baseline[off + 1 + i]).collect()
        };

        let cfg = SpecConfig { k, max_new_tokens: n };
        let committed = run_spec_decode(&mut model, draft_fn, start_bonus, start_pos, &cfg);

        // EXACTLY n, not `>= n`: a full-accept block commits `accept_len + 1`,
        // so the driver narrows each block's width to `min(k, remaining - 1)`
        // and the returned stream is never longer than the budget. `>= n` was
        // the loose bound that let the overshoot bug live.
        assert_eq!(committed.len(), n, "expected EXACTLY {n} committed tokens, got {}", committed.len());
        for i in 0..n {
            assert_eq!(committed[i], baseline[i], "token {i}: spec-on {} != greedy baseline {}", committed[i], baseline[i]);
        }

        // ACCEPTANCE, not just token identity. The checks above are blind to
        // whether speculation happened at all: a drafter whose every token is
        // REJECTED commits only its bonus per block, and that stream is still
        // the greedy stream — so the gate would stay green with the entire
        // speculative lever disengaged. These two assertions are what make it
        // observe the acceptance actually achieved.
        let drafted: usize = widths.iter().sum();
        let accepted = n - widths.len(); // n committed = accepted drafts + 1 bonus per block
        assert_eq!(
            accepted, drafted,
            "full-accept drafter: expected all {drafted} drafted tokens accepted over \
             {} blocks, but only {accepted} were (acceptance collapsed / speculation disengaged)",
            widths.len());
        assert_eq!(
            widths, expected_full_accept_widths(k, n),
            "full-accept block structure changed: k={k}, n={n} must run these draft widths");
        eprintln!(
            "identity gate: {n} tokens matched greedy baseline exactly; \
             {} blocks, {accepted}/{drafted} drafted tokens accepted", widths.len());
    }

    /// INC-5a gate 1, synthetic: identical to
    /// `gemma31b_spec_driver_identity_matches_greedy_real12b` above but driven
    /// on `tiny_synthetic_gemma` — an in-memory, deterministic-weight model
    /// (see `model.rs`) that exercises the same `forward_verify_core` /
    /// `verify_rollback` code paths without a checkpoint load. A short 8-token
    /// run is plenty to exercise 2 full spec blocks (k=4) end to end; this is
    /// the DEFAULT gate, the real-checkpoint version is `#[ignore]`d for
    /// optional confidence only.
    #[test]
    fn gemma31b_spec_driver_identity_matches_greedy() {
        let max_seq = 32usize;
        let k = 4usize;
        let n = 8usize; // 8-12 generated tokens is plenty to prove the logic
        let mut model = tiny_synthetic_gemma(max_seq);

        let prefix = [2u32, 10, 20];
        let r0 = prefix.len();
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        // `bonus` is a NEW token that has not yet been forwarded (see the
        // real-checkpoint gate's doc comment above for why).
        let start_bonus = 30u32;
        let start_pos = r0;

        // Ground truth: greedy baseline over a small margin beyond `n` (the
        // stub drafter reads ahead by up to `k` tokens per block, and blocks
        // can overshoot `n` by up to `k`, so one block's worth of slack is
        // enough).
        let baseline = greedy_baseline(&mut model, start_bonus, start_pos, n + k + 2);

        // Reset KV state and replay the same short prefix so the spec-decode
        // driver starts from the identical clean state the baseline did.
        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }

        // Stub drafter: forces draft[i] = baseline[pos_offset+1+i], i.e.
        // exactly the greedy continuation of whatever bonus token led into
        // this block. `baseline` is indexed relative to `start_pos`.
        // `widths` records the width the driver asked for on each block — one
        // push per block. See `expected_full_accept_widths` for why acceptance
        // is observed here rather than through the driver's return type.
        let mut widths: Vec<usize> = Vec::new();
        let draft_fn = |_bonus: u32, pos: usize, k: usize| -> Vec<u32> {
            widths.push(k);
            let off = pos - start_pos;
            (0..k).map(|i| baseline[off + 1 + i]).collect()
        };

        let cfg = SpecConfig { k, max_new_tokens: n };
        let committed = run_spec_decode(&mut model, draft_fn, start_bonus, start_pos, &cfg);

        // EXACTLY n, not `>= n` — see the real-checkpoint twin above.
        assert_eq!(committed.len(), n, "expected EXACTLY {n} committed tokens, got {}", committed.len());
        for i in 0..n {
            assert_eq!(committed[i], baseline[i], "token {i}: spec-on {} != greedy baseline {}", committed[i], baseline[i]);
        }

        // ACCEPTANCE, not just token identity — the checks above stay green
        // with speculation fully disengaged. See the real-checkpoint twin.
        // `spec_decode_zero_acceptance_is_caught_by_the_identity_gate` below is
        // the negative control that proves these two assertions have teeth.
        let drafted: usize = widths.iter().sum();
        let accepted = n - widths.len(); // n committed = accepted drafts + 1 bonus per block
        assert_eq!(
            accepted, drafted,
            "full-accept drafter: expected all {drafted} drafted tokens accepted over \
             {} blocks, but only {accepted} were (acceptance collapsed / speculation disengaged)",
            widths.len());
        assert_eq!(
            widths, expected_full_accept_widths(k, n),
            "full-accept block structure changed: k={k}, n={n} must run these draft widths");
        eprintln!(
            "identity gate (synthetic): {n} tokens matched greedy baseline exactly; \
             {} blocks, {accepted}/{drafted} drafted tokens accepted", widths.len());
    }

    /// NEGATIVE CONTROL for the identity gates' acceptance assertions.
    ///
    /// Pins the failure mode those assertions exist to catch: a drafter whose
    /// every token is rejected. It shows, on the same model and the same
    /// starting state the synthetic identity gate uses, that
    ///
    ///   1. the committed stream is STILL bit-identical to the greedy baseline
    ///      — so the pre-existing token-identity checks pass unchanged, and
    ///      cannot distinguish speculation from no speculation at all; and
    ///   2. the measured acceptance is exactly ZERO while the drafter offered
    ///      `sum(widths)` tokens — which is what the identity gates now assert
    ///      against, and what makes them fail on a disengaged lever.
    ///
    /// The block structure differs too, and deliberately so: with nothing
    /// accepted every block commits only its bonus, so the run takes `n`
    /// blocks instead of the full-accept structure's few.
    #[test]
    fn spec_decode_zero_acceptance_is_caught_by_the_identity_gate() {
        let max_seq = 32usize;
        let k = 4usize;
        let n = 8usize;
        let mut model = tiny_synthetic_gemma(max_seq);

        let prefix = [2u32, 10, 20];
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        let start_bonus = 30u32;
        let start_pos = prefix.len();

        let baseline = greedy_baseline(&mut model, start_bonus, start_pos, n + k + 2);
        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }

        // Mis-drafter: every id is the true next token bumped by one, so it is
        // guaranteed to differ from the target's argmax and NOTHING is ever
        // accepted. The bonus token still comes from the target's own argmax,
        // which is why the committed stream stays greedy-identical.
        let vocab = model.config.vocab_size as u32;
        let mut widths: Vec<usize> = Vec::new();
        let draft_fn = |_bonus: u32, pos: usize, kk: usize| -> Vec<u32> {
            widths.push(kk);
            let off = pos - start_pos;
            (0..kk).map(|i| (baseline[off + 1 + i] + 1) % vocab).collect()
        };

        let cfg = SpecConfig { k, max_new_tokens: n };
        let committed = run_spec_decode(&mut model, draft_fn, start_bonus, start_pos, &cfg);

        // (1) The OLD gate's checks still pass with the lever fully disengaged.
        assert_eq!(committed.len(), n, "budget must still be honoured exactly");
        assert_eq!(committed[..], baseline[..n],
                   "zero-acceptance stream must still equal the greedy baseline — this is \
                    precisely why token identity alone cannot gate speculation");

        // (2) ...but the acceptance measurement the identity gates assert on
        // collapses to zero, so they fail.
        let drafted: usize = widths.iter().sum();
        let accepted = n - widths.len();
        assert_eq!(accepted, 0, "nothing may be accepted from a deliberately wrong drafter");
        assert!(drafted > 0, "the control is only meaningful if drafts were actually offered");
        assert_eq!(widths.len(), n, "with nothing accepted every block commits its bonus alone");
        assert_ne!(widths, expected_full_accept_widths(k, n),
                   "the collapsed block structure must differ from the full-accept one");
        eprintln!(
            "negative control: greedy-identical stream, but {accepted}/{drafted} accepted over \
             {} blocks (full-accept would be {:?})", widths.len(), expected_full_accept_widths(k, n));
    }

    /// `max_new_tokens` is a HARD CAP, not a floor.
    ///
    /// Every `k` here is chosen NOT to divide its budget, because a `k` that
    /// divides it hides the defect: with `k + 1` committed per full-accept
    /// block, `k = 3` / budget 8 lands on 4, 8 and looks correct even when the
    /// last block drafts its full width. The cases below all need the last
    /// block narrowed, and the pre-fix driver overshot each of them — `k = 4`
    /// with a budget of 8 returned 10 tokens (the reviewer's own example, the
    /// second case here).
    ///
    /// `budget = 1` is the boundary: the width narrows to 0, which must still
    /// commit the bonus token and terminate rather than loop forever on a
    /// zero-length block.
    #[test]
    fn spec_decode_respects_max_new_tokens_exactly() {
        let max_seq = 64usize;
        for (k, budget) in [(3usize, 7usize), (4, 8), (4, 5), (2, 9), (5, 1), (1, 1), (4, 3)] {
            let mut model = tiny_synthetic_gemma(max_seq);
            let prefix = [2u32, 10, 20];
            for (pos, &tok) in prefix.iter().enumerate() {
                model.forward(tok, pos);
            }
            let start_bonus = 30u32;
            let start_pos = prefix.len();

            // Full-accept drafter: drafts the target's own greedy continuation,
            // so every block commits its maximum `width + 1` tokens. This is
            // the WORST case for the cap — a partially-rejecting drafter
            // commits fewer and could mask an over-wide draft.
            let baseline = greedy_baseline(&mut model, start_bonus, start_pos, budget + k + 2);
            reset_kv_caches(&mut model, max_seq);
            for (pos, &tok) in prefix.iter().enumerate() {
                model.forward(tok, pos);
            }
            let draft_fn = |_bonus: u32, pos: usize, width: usize| -> Vec<u32> {
                let off = pos - start_pos;
                (0..width).map(|i| baseline[off + 1 + i]).collect()
            };

            let cfg = SpecConfig { k, max_new_tokens: budget };
            let committed = run_spec_decode(&mut model, draft_fn, start_bonus, start_pos, &cfg);

            assert_eq!(committed.len(), budget,
                       "k={k}, max_new_tokens={budget}: expected EXACTLY {budget} committed \
                        tokens, got {}", committed.len());
            // Capping must not change WHICH tokens come out, only how many.
            assert_eq!(committed[..], baseline[..budget],
                       "k={k}, max_new_tokens={budget}: capped stream must still equal the \
                        greedy baseline prefix");
        }
    }

    /// INC-5a gate 2 ("partial-accept + rollback correctness"): a STUB
    /// drafter that drafts a correct prefix then a deliberately WRONG token
    /// (and arbitrary garbage after it, never reachable) must produce
    /// `accept_len` == the correct-prefix length on the mismatching block,
    /// the driver's emitted ids must still equal the greedy baseline (the
    /// wrong drafted suffix is never committed), and the KV frontier after
    /// `verify_rollback` must match a clean baseline run that never executed
    /// the rejected suffix at all (reusing the INC-4
    /// `gemma31b_verify_rollback_frontier_matches_clean_baseline` idea,
    /// driven through the driver instead of calling verify/rollback by hand).
    ///
    /// ONE checkpoint load only (see the identity gate's doc comment above
    /// for why) — the greedy continuation used to derive the "correct
    /// prefix" and the clean-baseline replay are both driven off the SAME
    /// loaded model, with KV resets/prefix replays in between.
    ///
    /// Real-checkpoint variant of the synthetic gate below (~30 min/run on the
    /// full gemma-4-12B f32 checkpoint) — kept for optional real-weight
    /// confidence, `#[ignore]`d by default so `cargo test` stays fast.
    #[test]
    #[ignore]
    fn gemma31b_spec_driver_partial_accept_rollback_matches_baseline_real12b() {
        let max_seq = 32usize;
        let k = 4usize;
        let mut model = match load_gemma12b(max_seq) { Some(m) => m, None => return };

        let prefix = [2u32, 1024, 2048];
        let r0 = prefix.len();
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        // `bonus` is a NEW token not yet forwarded (see identity gate above
        // for why it must be distinct from `prefix`).
        let start_bonus = 4096u32;
        let start_pos = r0;

        // Ground truth greedy continuation (only need k+1 tokens: bonus + k
        // true next tokens) to know the correct prefix and the "wrong" id.
        // Computed on `model` directly, then KV is reset and the prefix
        // replayed so the driver call below starts from the same clean state.
        let baseline = greedy_baseline(&mut model, start_bonus, start_pos, k + 2);
        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }

        // Force a mismatch after 2 correct drafts (assumes k >= 3 so there's
        // a rejected remainder to exercise rollback on).
        let forced_correct = 2usize;
        assert!(forced_correct < k, "test needs k > forced_correct to exercise a rejection");
        // A token id guaranteed different from the true next token: bump by
        // 1 mod vocab (vocab is large; wrap is astronomically unlikely to
        // coincide, and even if it did the test would just be less strict,
        // never wrong — accept_len is checked via argmax equality by the
        // driver itself, not id equality).
        let true_next = baseline[forced_correct + 1];
        let vocab = model.config.vocab_size as u32; // usize -> u32, vocab_size (262144) fits
        let wrong_tok = (true_next + 1) % vocab;

        let baseline_for_draft = baseline.clone();
        let draft = move |_bonus: u32, pos: usize, kk: usize| -> Vec<u32> {
            assert_eq!(pos, start_pos, "single-block test: only the first block should draft");
            let mut d = Vec::with_capacity(kk);
            for i in 0..kk {
                if i < forced_correct {
                    d.push(baseline_for_draft[i + 1]);
                } else if i == forced_correct {
                    d.push(wrong_tok);
                } else {
                    d.push(999_999_999u32 % vocab); // unreachable garbage
                }
            }
            d
        };

        let step = spec_step(&mut model, draft, start_bonus, start_pos, k);

        assert_eq!(step.accept_len, forced_correct, "accept_len should equal the correct-prefix length");
        let expected_committed: Vec<u32> = std::iter::once(start_bonus)
            .chain(baseline[1..1 + forced_correct].iter().copied())
            .collect();
        assert_eq!(step.committed, expected_committed, "committed ids must equal the greedy baseline prefix");

        // Frontier check: every layer's KV counter must sit at
        // start_pos + accept_len + 1, matching the INC-4 rollback contract.
        let expected_frontier = start_pos + step.accept_len + 1;
        for (li, cache) in model.kv_caches.iter().enumerate() {
            assert_eq!(cache.seq_len, expected_frontier, "layer {li}: KV frontier {} != expected {expected_frontier}", cache.seq_len);
        }

        // Clean baseline: never runs the rejected suffix at all — prefix,
        // then only the accepted `accept_len+1` tokens, then a shared
        // continuation token, must be bit-exact vs. the driver's post-
        // rollback state continuing with the same token.
        let continuation_tok = wrong_tok; // arbitrary; just needs to match on both sides
        let logits_after_rollback = model.forward(continuation_tok, expected_frontier);

        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        for (i, &tok) in expected_committed.iter().enumerate() {
            model.forward(tok, start_pos + i);
        }
        let logits_baseline = model.forward(continuation_tok, expected_frontier);

        let maxdiff = logits_after_rollback
            .iter()
            .zip(logits_baseline.iter())
            .map(|(&a, &b)| (a - b).abs())
            // NOT `f32::max`: it returns the non-NaN operand, so a NaN pair
            // would fold away to 0.0 and the assertion below would pass on
            // garbage. Propagate NaN instead, so `maxdiff == 0.0` is a real
            // proof of bit-identity — which is what lets the redundant
            // cosine assertion go (see below).
            .fold(0.0f32, |m, d| if d.is_nan() || d > m { d } else { m });
        let cos = cosine(&logits_after_rollback, &logits_baseline);
        eprintln!(
            "partial-accept gate: accept_len={} frontier={expected_frontier} maxdiff={maxdiff:.8} cos={cos:.6}",
            step.accept_len
        );
        assert_eq!(maxdiff, 0.0, "post-rollback forward diverges from clean baseline (maxdiff {maxdiff})");
        // NO assertion on `cos`. The `maxdiff == 0.0` check above already
        // proves the two vectors are bit-identical, which is strictly
        // stronger than any cosine check, so this one could only ever add
        // false failures. It did: `cosine` computes `dot / (na * nb)`, and on
        // identical inputs that is `s / (sqrt(s) * sqrt(s))`, where
        // `sqrt(s) * sqrt(s)` is NOT guaranteed to round back to `s`. Whether
        // it does depends on the target's sqrt rounding and FMA contraction —
        // so `assert_eq!(cos, 1.0)` passed on aarch64 and failed on x86-64 CI
        // with cos = 0.99999994, on code that was correct. `cos` stays in the
        // eprintln above as a diagnostic.
    }

    /// INC-5a gate 2, synthetic: identical to
    /// `gemma31b_spec_driver_partial_accept_rollback_matches_baseline_real12b`
    /// above but driven on `tiny_synthetic_gemma`. This is the DEFAULT gate;
    /// the real-checkpoint version is `#[ignore]`d for optional confidence
    /// only.
    #[test]
    fn gemma31b_spec_driver_partial_accept_rollback_matches_baseline() {
        let max_seq = 32usize;
        let k = 4usize;
        let mut model = tiny_synthetic_gemma(max_seq);

        let prefix = [2u32, 10, 20];
        let r0 = prefix.len();
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        // `bonus` is a NEW token not yet forwarded (see identity gate above
        // for why it must be distinct from `prefix`).
        let start_bonus = 30u32;
        let start_pos = r0;

        // Ground truth greedy continuation (only need k+1 tokens: bonus + k
        // true next tokens) to know the correct prefix and the "wrong" id.
        // Computed on `model` directly, then KV is reset and the prefix
        // replayed so the driver call below starts from the same clean state.
        let baseline = greedy_baseline(&mut model, start_bonus, start_pos, k + 2);
        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }

        // Force a mismatch after 2 correct drafts (assumes k >= 3 so there's
        // a rejected remainder to exercise rollback on).
        let forced_correct = 2usize;
        assert!(forced_correct < k, "test needs k > forced_correct to exercise a rejection");
        // A token id guaranteed different from the true next token: bump by
        // 1 mod vocab. accept_len is checked via argmax equality by the
        // driver itself, not id equality, so this is never wrong even in the
        // (astronomically unlikely, and harmless if it happened) case of wrap.
        let true_next = baseline[forced_correct + 1];
        let vocab = model.config.vocab_size as u32; // tiny synthetic vocab (512)
        let wrong_tok = (true_next + 1) % vocab;

        let baseline_for_draft = baseline.clone();
        let draft = move |_bonus: u32, pos: usize, kk: usize| -> Vec<u32> {
            assert_eq!(pos, start_pos, "single-block test: only the first block should draft");
            let mut d = Vec::with_capacity(kk);
            for i in 0..kk {
                if i < forced_correct {
                    d.push(baseline_for_draft[i + 1]);
                } else if i == forced_correct {
                    d.push(wrong_tok);
                } else {
                    d.push(999_999_999u32 % vocab); // unreachable garbage
                }
            }
            d
        };

        let step = spec_step(&mut model, draft, start_bonus, start_pos, k);

        assert_eq!(step.accept_len, forced_correct, "accept_len should equal the correct-prefix length");
        let expected_committed: Vec<u32> = std::iter::once(start_bonus)
            .chain(baseline[1..1 + forced_correct].iter().copied())
            .collect();
        assert_eq!(step.committed, expected_committed, "committed ids must equal the greedy baseline prefix");

        // Frontier check: every layer's KV counter must sit at
        // start_pos + accept_len + 1, matching the INC-4 rollback contract.
        let expected_frontier = start_pos + step.accept_len + 1;
        for (li, cache) in model.kv_caches.iter().enumerate() {
            assert_eq!(cache.seq_len, expected_frontier, "layer {li}: KV frontier {} != expected {expected_frontier}", cache.seq_len);
        }

        // Clean baseline: never runs the rejected suffix at all — prefix,
        // then only the accepted `accept_len+1` tokens, then a shared
        // continuation token, must be bit-exact vs. the driver's post-
        // rollback state continuing with the same token.
        let continuation_tok = wrong_tok; // arbitrary; just needs to match on both sides
        let logits_after_rollback = model.forward(continuation_tok, expected_frontier);

        reset_kv_caches(&mut model, max_seq);
        for (pos, &tok) in prefix.iter().enumerate() {
            model.forward(tok, pos);
        }
        for (i, &tok) in expected_committed.iter().enumerate() {
            model.forward(tok, start_pos + i);
        }
        let logits_baseline = model.forward(continuation_tok, expected_frontier);

        let maxdiff = logits_after_rollback
            .iter()
            .zip(logits_baseline.iter())
            .map(|(&a, &b)| (a - b).abs())
            // NOT `f32::max`: it returns the non-NaN operand, so a NaN pair
            // would fold away to 0.0 and the assertion below would pass on
            // garbage. Propagate NaN instead, so `maxdiff == 0.0` is a real
            // proof of bit-identity — which is what lets the redundant
            // cosine assertion go (see below).
            .fold(0.0f32, |m, d| if d.is_nan() || d > m { d } else { m });
        let cos = cosine(&logits_after_rollback, &logits_baseline);
        eprintln!(
            "partial-accept gate (synthetic): accept_len={} frontier={expected_frontier} maxdiff={maxdiff:.8} cos={cos:.6}",
            step.accept_len
        );
        assert_eq!(maxdiff, 0.0, "post-rollback forward diverges from clean baseline (maxdiff {maxdiff})");
        // NO assertion on `cos`. The `maxdiff == 0.0` check above already
        // proves the two vectors are bit-identical, which is strictly
        // stronger than any cosine check, so this one could only ever add
        // false failures. It did: `cosine` computes `dot / (na * nb)`, and on
        // identical inputs that is `s / (sqrt(s) * sqrt(s))`, where
        // `sqrt(s) * sqrt(s)` is NOT guaranteed to round back to `s`. Whether
        // it does depends on the target's sqrt rounding and FMA contraction —
        // so `assert_eq!(cos, 1.0)` passed on aarch64 and failed on x86-64 CI
        // with cos = 0.99999994, on code that was correct. `cos` stays in the
        // eprintln above as a diagnostic.
    }
}
