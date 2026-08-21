// SPDX-License-Identifier: Apache-2.0
//! P3 depth-1 speculative-pipelining bookkeeping — the rank4 accept/reject
//! state machine, factored as PURE LOGIC (no GPU, no vCCL) so the greedy-exact
//! identity contract is Mac-testable with a scripted fake target + drafter.
//!
//! The distributed driver (`scripts/pp_qwen35_spec.py`, deployed from
//! `~/repos/bc250-cluster`) transcribes THIS algorithm line-for-line; the tests
//! below are the P3 Mac gate for the bookkeeping. Keep the two in sync.
//!
//! ## Protocol (see plan §3.3 Design B, depth 1)
//!
//! Token stream `s_0, s_1, …` with the greedy relation `s_{i+1} = step(s_i, i)`
//! (`step` = the target's argmax). The MTP head, given the pre-`model.norm`
//! hidden `h_m` at position `m` and `embed(s_{m+1})`, drafts a guess for
//! `s_{m+2}` (a next-2 predictor) — call it `draft(m)`.
//!
//! Each CYCLE injects two single-token passes through the 5-stage pipeline:
//!   * **pass A** (real): input `cur = s_pos` at position `pos`. Because `cur`
//!     is confirmed, pass A's argmax `a_out = s_{pos+1}` is the TRUE next token.
//!   * **pass B** (spec): input `draft` (last cycle's `draft(pos-1)`, a guess
//!     for `s_{pos+1}`) at position `pos+1`. Valid iff `draft == a_out`.
//!
//! Verification is intrinsic on rank4: `accept = (draft == a_out)`.
//!   * **accept**: pass B fed the correct token, so its argmax `b_out = s_{pos+2}`
//!     is ALSO confirmed. Two tokens committed this cycle; advance `cur = b_out`,
//!     `pos += 2`. Next draft = `draft(pos+1)` off pass B's hidden.
//!   * **reject**: pass B's state advance is garbage → every stage restores its
//!     pre-spec snapshot. One token committed; re-issue `cur = a_out`,
//!     `pos += 1`. Next draft = `draft(pos)` off pass A's hidden.
//!
//! In BOTH cases the emitted sequence is exactly `s_pos, s_{pos+1}, …` — the
//! draft only changes how many tokens land per cycle, never WHICH tokens, so
//! the output is greedy-exact regardless of accept rate. That invariant is what
//! the tests assert.
//!
//! ## MTP head KV (gap-free, never rolled back)
//!
//! The head runs exactly one forward per COMMITTED token, in increasing
//! position order, so its own KV only ever holds confirmed positions and never
//! needs rollback (garbage pass B never reaches the head). Per cycle: always one
//! head forward at `head_pos = pos` (off pass A's hidden `h_pos`, embedding
//! `a_out`); on accept, additionally one at `head_pos = pos+1` (off pass B's
//! `h_{pos+1}`, embedding `b_out`). `draft(pos)` doubles as the reject-draft;
//! `draft(pos+1)` is the accept-draft.
#![allow(dead_code)]

/// The next real token to inject as pass A, plus its position and whether the
/// receiving stages must first restore the pre-spec snapshot (previous cycle
/// was a reject).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inject {
    pub real: u32,
    pub pos: usize,
    /// The draft to inject as pass B (guess for `s_{pos+1}`).
    pub draft: u32,
    /// Restore the pre-spec snapshot on every stage before pass A of this cycle.
    pub rollback: bool,
}

/// Result of resolving one cycle on rank4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub accept: bool,
    /// Tokens committed this cycle, in order (1 on reject, 2 on accept).
    pub emitted: Vec<u32>,
    /// The injection descriptor for the NEXT cycle.
    pub next: Inject,
}

/// Resolve one rank4 cycle from its observable quantities.
///
/// * `inj`      — this cycle's injection (what pass A/B were fed).
/// * `a_out`    — pass A argmax (the true next token `s_{pos+1}`).
/// * `b_out`    — pass B argmax (`s_{pos+2}`, meaningful only when accepted).
/// * `draft_at_pos` — the head's draft at `head_pos = pos` (guess for
///   `s_{pos+2}`); becomes the next draft on reject.
/// * `draft_at_pos1` — the head's draft at `head_pos = pos+1` (guess for
///   `s_{pos+3}`); becomes the next draft on accept. Ignored on reject (the
///   caller need not even run that head forward when it already knows reject).
///
/// This is the single source of truth the Python driver mirrors.
pub fn resolve_cycle(
    inj: Inject,
    a_out: u32,
    b_out: u32,
    draft_at_pos: u32,
    draft_at_pos1: u32,
) -> Resolved {
    let accept = inj.draft == a_out;
    if accept {
        Resolved {
            accept,
            emitted: vec![inj.real, a_out],
            next: Inject { real: b_out, pos: inj.pos + 2, draft: draft_at_pos1, rollback: false },
        }
    } else {
        Resolved {
            accept,
            emitted: vec![inj.real],
            next: Inject { real: a_out, pos: inj.pos + 1, draft: draft_at_pos, rollback: true },
        }
    }
}

// ───────────────────────────── P4: chain-depth resolve ─────────────────────────
// ## Chain-`D` speculative pipelining (plan §P4, Design B continuous refill)
//
// Depth-1 injects two passes per cycle; depth-`D` stacks a WINDOW of `D+1`
// passes one slot apart behind the real token: pass A (real `s_R`) plus `D`
// speculative passes `B_1..B_D` fed the chained drafts `d_1..d_D`, where `d_j` is
// the head's guess for `s_{R+j}` (`d_j` drafted autoregressively off `d_{j-1}` —
// the head's own KV chains, plan §1.2).
//
// Verification walks the chain on the last rank. Let `out(i) = a_out` for `i=0`,
// else `b_i_out` = argmax of pass `B_i`:
//   * `out(0)=a_out` is the TRUE `s_{R+1}` (pass A fed the confirmed `s_R`), so it
//     verifies `d_1`.
//   * inductively, if `d_1..d_i` all matched then pass `B_i` was fed the correct
//     `s_{R+i}`, so `out(i)=b_i_out` is the TRUE `s_{R+i+1}` and verifies `d_{i+1}`.
// The accept-prefix length `k ∈ 0..=D` is the first mismatch (`d_k != out(k)`), or
// `D` if the whole chain holds. The cycle commits `s_R..s_{R+k}` (`k+1` tokens)
// and carries `out(k)` as the next real — always the TRUE sequence, so the greedy
// output is bit-exact regardless of accept rate (the invariant the tests assert).
// On `k<D` the `D-k` deepest passes were fed wrong tokens: every stage restores
// its slot-`k` snapshot (the state right after pass `B_k`); on `k=D` nothing is
// squashed.
//
// ## Snapshot ring (one slot per outstanding draft depth)
//
// Each stage snapshots into ring slot `j` right AFTER pass `B_j` (slot 0 = after
// pass A) for `j = 0..D-1` — exactly `D` slots (`SPEC_RING_DEPTH`). A reject at
// prefix `k` restores slot `k`; `k=D` needs no slot beyond `D-1`. Slots are
// overwritten every cycle (fixed ring, never grows past `D`). Depth-1 uses one
// slot and reduces to `resolve_cycle` exactly (asserted by a cross-check test).

/// Result of resolving one chain-`D` cycle on the last rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainResolved {
    /// Accept-prefix length `k ∈ 0..=D` (drafts `d_1..d_k` confirmed).
    pub accept_len: usize,
    /// Tokens committed this cycle, in order: `[s_R, s_{R+1}, …, s_{R+k}]`
    /// (`k+1` tokens). Always the true greedy sequence.
    pub emitted: Vec<u32>,
    /// The next real token to inject as pass A of the next cycle (`= out(k)`).
    pub new_real: u32,
    /// The next real token's absolute position (`= pos + k + 1`).
    pub new_pos: usize,
    /// Ring slot every stage must restore before the next pass A, or `None` when
    /// the whole chain was accepted (nothing squashed). `Some(k)` on a reject.
    pub restore_slot: Option<usize>,
    /// Number of speculative passes discarded (`D - k`) — accounting only.
    pub squashed: usize,
}

/// Resolve one chain-`D` cycle from its observable quantities. `drafts = d_1..d_D`
/// (the guesses injected as passes `B_1..B_D`), `real`/`pos` the pass-A input and
/// its position, `a_out` = pass-A argmax, `b_outs = b_1_out..b_D_out` = the `D`
/// speculative-pass argmaxes (one per draft). `b_outs[k]` is meaningful only when
/// `d_1..d_k` were accepted; the resolve reads it exactly then and no further.
///
/// This is the single source of truth the Python driver's last-rank branch
/// mirrors. `resolve_cycle` is its `D=1` special case (kept as the P3 regression
/// guard; `resolve_chain_matches_resolve_cycle_at_depth1` cross-checks them).
pub fn resolve_chain(
    drafts: &[u32],
    real: u32,
    pos: usize,
    a_out: u32,
    b_outs: &[u32],
) -> ChainResolved {
    let depth = drafts.len();
    assert_eq!(b_outs.len(), depth, "b_outs must have one argmax per draft");
    // out(i): the verifier for draft i (0-indexed). out(0)=a_out; out(i)=b_i_out.
    // Valid for i in 0..=depth; out(depth)=b_outs[depth-1] is the full-accept carry.
    let out = |i: usize| -> u32 { if i == 0 { a_out } else { b_outs[i - 1] } };
    let mut k = 0usize;
    while k < depth && drafts[k] == out(k) {
        k += 1;
    }
    let mut emitted = Vec::with_capacity(k + 1);
    emitted.push(real);
    for i in 0..k {
        emitted.push(out(i));
    }
    ChainResolved {
        accept_len: k,
        emitted,
        new_real: out(k),
        new_pos: pos + k + 1,
        restore_slot: if k < depth { Some(k) } else { None },
        squashed: depth - k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random greedy world: `step(tok, pos)` is a pure
    /// function, so a plain greedy loop defines the reference sequence and pass
    /// A (always fed a confirmed token) reproduces it exactly.
    fn step(tok: u32, pos: usize) -> u32 {
        let mut x = (tok as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (pos as u64).wrapping_mul(0xD1B54A32D192ED03);
        x ^= x >> 33; x = x.wrapping_mul(0xff51afd7ed558ccd); x ^= x >> 33;
        (x % 50_000) as u32 // vocab-ish
    }

    /// Reference greedy sequence of `n` emitted tokens starting from `t0` at
    /// `pos0` (t0 is the first generated token = argmax of the prompt tail).
    fn greedy(t0: u32, pos0: usize, n: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(n);
        let (mut cur, mut pos) = (t0, pos0);
        while out.len() < n {
            out.push(cur);
            cur = step(cur, pos);
            pos += 1;
        }
        out
    }

    /// The head's draft at `head_pos = m` predicts `s_{m+2}`. `hit(m)` scripts
    /// whether the drafter nails it; a miss returns a value guaranteed to differ
    /// from the truth (so `accept` is exactly `hit`).
    fn draft_at(seq_from_pos0: &dyn Fn(usize) -> u32, m: usize, hit: bool) -> u32 {
        let truth = seq_from_pos0(m + 2); // s_{m+2}
        if hit { truth } else { truth.wrapping_add(1).wrapping_mul(7).wrapping_add(13) ^ 0x5555 }
    }

    /// Run the FULL depth-1 cycle machine against the fake world with a boolean
    /// accept script `hit(head_pos)`, returning the emitted sequence and the
    /// realized accept count. Mirrors `scripts/pp_qwen35_spec.py` exactly.
    fn run(t0: u32, pos0: usize, gen: usize, hit: &dyn Fn(usize) -> bool) -> (Vec<u32>, usize, usize) {
        // s_i as an absolute-position lookup, memoized off the greedy chain.
        // s_{pos0} = t0; s_{pos0+k} = greedy chain. Provide s_i for i >= pos0.
        let chain = greedy(t0, pos0, gen + 8); // a bit of headroom for lookahead drafts
        let s = move |i: usize| -> u32 {
            assert!(i >= pos0, "position {i} below pos0 {pos0}");
            chain[i - pos0]
        };

        let mut emitted: Vec<u32> = Vec::new();
        let mut accepts = 0usize;
        let mut cycles = 0usize;

        // Bootstrap draft (computed during prefill): head_pos = pos0-1, guessing
        // s_{pos0+1}. draft_at(m) predicts s_{m+2}; m = pos0-1 => s_{pos0+1}. ✔
        let boot_hit = hit(pos0.saturating_sub(1));
        let first_draft = draft_at(&s, pos0.saturating_sub(1), boot_hit);
        let mut inj = Inject { real: t0, pos: pos0, draft: first_draft, rollback: false };

        while emitted.len() < gen {
            cycles += 1;
            let pos = inj.pos;
            // pass A: input `inj.real` (= s_pos) at pos -> true next s_{pos+1}.
            let a_out = step(inj.real, pos);
            // pass B: input `inj.draft` at pos+1. Its argmax is the true next of
            // WHATEVER was fed; only meaningful when the fed token == a_out.
            let b_out = step(inj.draft, pos + 1);
            // Head drafts at head_pos=pos (reject-draft, predicts s_{pos+2}) and
            // head_pos=pos+1 (accept-draft, predicts s_{pos+3}).
            let d_pos = draft_at(&s, pos, hit(pos));
            let d_pos1 = draft_at(&s, pos + 1, hit(pos + 1));

            let r = resolve_cycle(inj, a_out, b_out, d_pos, d_pos1);
            if r.accept { accepts += 1; }
            emitted.extend_from_slice(&r.emitted);
            inj = r.next;
        }
        emitted.truncate(gen);
        (emitted, accepts, cycles)
    }

    #[test]
    fn identity_all_reject() {
        // Worst case: every draft wrong. One token per cycle, still greedy-exact.
        let t0 = step(9707, 0);
        let (emitted, accepts, cycles) = run(t0, 1, 200, &|_| false);
        assert_eq!(emitted, greedy(t0, 1, 200), "all-reject must stay greedy-exact");
        assert_eq!(accepts, 0);
        assert_eq!(cycles, 200, "all-reject: 1 token/cycle");
    }

    #[test]
    fn identity_all_accept() {
        // Best case: every draft right. Two tokens per cycle, greedy-exact.
        let t0 = step(9707, 0);
        let (emitted, accepts, cycles) = run(t0, 1, 200, &|_| true);
        assert_eq!(emitted, greedy(t0, 1, 200), "all-accept must stay greedy-exact");
        // 200 tokens at 2/cycle => ~100 cycles, all accepted.
        assert_eq!(accepts, cycles);
        assert!(cycles <= 101 && cycles >= 100, "all-accept ~100 cycles, got {cycles}");
    }

    #[test]
    fn identity_alternating() {
        let t0 = step(1234, 5);
        let (emitted, _a, _c) = run(t0, 6, 200, &|m| m % 2 == 0);
        assert_eq!(emitted, greedy(t0, 6, 200), "alternating script must stay greedy-exact");
    }

    #[test]
    fn identity_pseudorandom_scripts() {
        // Many pseudo-random accept scripts, several prompt offsets and lengths —
        // the identity contract must hold for ALL of them.
        for seed in 0u64..64 {
            let t0 = step((seed as u32).wrapping_mul(2654435761), (seed as usize) & 7);
            let pos0 = 1 + (seed as usize % 5);
            let gen = 37 + (seed as usize % 90);
            let hit = move |m: usize| -> bool {
                let mut x = (m as u64).wrapping_add(seed.wrapping_mul(0x100000001b3));
                x ^= x >> 17; x = x.wrapping_mul(0x9E3779B97F4A7C15); x ^= x >> 31;
                (x & 3) != 0 // ~75% hit
            };
            let (emitted, accepts, cycles) = run(t0, pos0, gen, &hit);
            assert_eq!(emitted, greedy(t0, pos0, gen),
                "seed {seed}: pos0={pos0} gen={gen} must stay greedy-exact");
            // Sanity: accepts within [0, cycles], and 2*accepts + (cycles-accepts)
            // >= gen (each accept yields 2 tokens, each reject 1).
            assert!(accepts <= cycles);
            assert!(2 * accepts + (cycles - accepts) >= gen);
        }
    }

    // ───────────────────────── P4 chain-depth tests ─────────────────────────

    /// Fake depth-`D` drafter: `d_j` (j=1..depth) guesses `s_{r+j}`; `hit(r,j)`
    /// scripts whether it nails it. Per-depth scripting is independent — the
    /// resolve stops at the first miss, so beyond-miss values are don't-cares.
    fn draft_chain(
        s: &dyn Fn(usize) -> u32,
        r: usize,
        depth: usize,
        hit: &dyn Fn(usize, usize) -> bool,
    ) -> Vec<u32> {
        (1..=depth)
            .map(|j| {
                let truth = s(r + j);
                if hit(r, j) {
                    truth
                } else {
                    truth.wrapping_add(1).wrapping_mul(7).wrapping_add(13) ^ 0x5555
                }
            })
            .collect()
    }

    /// Full chain-`D` cycle machine against the fake world — mirrors the
    /// last-rank resolve + continuous refill in `pp_qwen35_spec.py`. Returns
    /// (emitted, accept-length histogram `[0..=depth]`, cycles).
    fn run_chain(
        t0: u32,
        pos0: usize,
        gen: usize,
        depth: usize,
        hit: &dyn Fn(usize, usize) -> bool,
    ) -> (Vec<u32>, Vec<usize>, usize) {
        // Headroom: a full-accept last cycle overshoots emitted by up to `depth`,
        // and the refill then drafts `depth` further ahead ⇒ need `2*depth` slack.
        let chain = greedy(t0, pos0, gen + 2 * depth + 4);
        let s = move |i: usize| -> u32 {
            assert!(i >= pos0, "position {i} below pos0 {pos0}");
            chain[i - pos0]
        };
        let mut emitted: Vec<u32> = Vec::new();
        let mut hist = vec![0usize; depth + 1];
        let mut cycles = 0usize;
        let (mut real, mut pos) = (t0, pos0);
        // Bootstrap chain off the first real token (computed during prefill).
        let mut drafts = draft_chain(&s, pos0, depth, hit);
        while emitted.len() < gen {
            cycles += 1;
            // pass A feeds the confirmed real at `pos`; pass B_j feeds d_j at pos+j.
            let a_out = step(real, pos);
            let b_outs: Vec<u32> = (1..=depth).map(|j| step(drafts[j - 1], pos + j)).collect();
            let r = resolve_chain(&drafts, real, pos, a_out, &b_outs);
            hist[r.accept_len] += 1;
            emitted.extend_from_slice(&r.emitted);
            // Continuous refill: draft a fresh chain off the new real token.
            drafts = draft_chain(&s, r.new_pos, depth, hit);
            real = r.new_real;
            pos = r.new_pos;
        }
        emitted.truncate(gen);
        (emitted, hist, cycles)
    }

    #[test]
    fn chain_identity_all_accept() {
        // Whole chain accepted every cycle: depth+1 tokens/cycle, greedy-exact.
        for depth in 1..=4 {
            let t0 = step(9707, 0);
            let (emitted, hist, cycles) = run_chain(t0, 1, 200, depth, &|_, _| true);
            assert_eq!(emitted, greedy(t0, 1, 200), "depth {depth} all-accept greedy-exact");
            assert_eq!(hist[depth], cycles, "depth {depth}: every cycle full-accept");
            let expect_cyc = 200usize.div_ceil(depth + 1);
            assert!(
                cycles <= expect_cyc + 1 && cycles + 1 >= expect_cyc,
                "depth {depth}: ~{expect_cyc} cycles at {}/cycle, got {cycles}",
                depth + 1
            );
        }
    }

    #[test]
    fn chain_identity_all_reject() {
        // Every draft wrong: k=0 every cycle, 1 token/cycle, greedy-exact — and
        // identical to depth-1 all-reject (the chain adds nothing when nothing
        // accepts, the worst-case floor).
        for depth in 1..=4 {
            let t0 = step(9707, 0);
            let (emitted, hist, cycles) = run_chain(t0, 1, 200, depth, &|_, _| false);
            assert_eq!(emitted, greedy(t0, 1, 200), "depth {depth} all-reject greedy-exact");
            assert_eq!(hist[0], cycles, "depth {depth}: every cycle k=0");
            assert_eq!(cycles, 200, "depth {depth} all-reject: 1 token/cycle");
        }
    }

    #[test]
    fn chain_identity_every_prefix_length() {
        // For each fixed accept-prefix K in 0..=depth: script hit(r,j)=j<=K so the
        // chain accepts exactly the first K drafts and misses at K+1. Output stays
        // greedy-exact and every cycle realizes accept_len == min(K, depth).
        let depth = 4usize;
        for kk in 0..=depth {
            let t0 = step(31337 + kk as u32, kk & 3);
            let hit = move |_r: usize, j: usize| -> bool { j <= kk };
            let (emitted, hist, cycles) = run_chain(t0, 2, 180, depth, &hit);
            assert_eq!(emitted, greedy(t0, 2, 180), "prefix K={kk}: greedy-exact");
            let realized = kk.min(depth);
            assert_eq!(hist[realized], cycles, "prefix K={kk}: every cycle accept_len={realized}");
            // tokens/cycle == realized+1 ⇒ cycles ~ ceil(180/(realized+1)).
            let want = 180usize.div_ceil(realized + 1);
            assert!(cycles <= want + 1, "K={kk}: ~{want} cycles, got {cycles}");
        }
    }

    #[test]
    fn chain_identity_perdepth_decay_random() {
        // Many pseudo-random accept scripts with a realistic per-DEPTH decay
        // (deeper drafts less likely to hit — mirrors the P0 chain-α decay). The
        // greedy identity must hold for ALL of them, at several offsets/depths.
        for seed in 0u64..96 {
            let depth = 1 + (seed as usize % 4); // 1..=4
            let t0 = step((seed as u32).wrapping_mul(2654435761), (seed as usize) & 7);
            let pos0 = 1 + (seed as usize % 5);
            let gen = 41 + (seed as usize % 80);
            let hit = move |r: usize, j: usize| -> bool {
                // base ~0.9 at depth 1, decaying ~0.14/depth, jittered per (r,j).
                let mut x = (r as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add((j as u64).wrapping_mul(0xD1B54A32D192ED03))
                    .wrapping_add(seed.wrapping_mul(0x100000001b3));
                x ^= x >> 29;
                x = x.wrapping_mul(0xff51afd7ed558ccd);
                x ^= x >> 32;
                let roll = (x % 1000) as i64;
                let thresh = 900 - 140 * (j as i64 - 1); // depth-decayed accept prob
                roll < thresh.max(0)
            };
            let (emitted, hist, cycles) = run_chain(t0, pos0, gen, depth, &hit);
            assert_eq!(
                emitted,
                greedy(t0, pos0, gen),
                "seed {seed}: depth={depth} pos0={pos0} gen={gen} greedy-exact"
            );
            // sanity: histogram sums to cycles; each accept_len in 0..=depth.
            assert_eq!(hist.iter().sum::<usize>(), cycles);
            assert_eq!(hist.len(), depth + 1);
        }
    }

    #[test]
    fn resolve_chain_matches_resolve_cycle_at_depth1() {
        // The depth-1 chain resolve must reproduce the P3 `resolve_cycle` exactly
        // (accept AND reject), so PP_SPEC=1 depth=1 is a bit-for-bit regression of
        // the proven P3 v2 loop.
        for &(d1, a_out, b_out) in &[(20u32, 20u32, 30u32), (21, 20, 30), (7, 7, 99), (5, 6, 8)] {
            let (real, pos) = (10u32, 4usize);
            let legacy = resolve_cycle(
                Inject { real, pos, draft: d1, rollback: false },
                a_out,
                b_out,
                /*d_pos*/ 99,
                /*d_pos1*/ 40,
            );
            let ch = resolve_chain(&[d1], real, pos, a_out, &[b_out]);
            assert_eq!(ch.emitted, legacy.emitted, "d1={d1}: emitted");
            assert_eq!(ch.accept_len == 1, legacy.accept, "d1={d1}: accept flag");
            assert_eq!(ch.new_real, legacy.next.real, "d1={d1}: next real");
            assert_eq!(ch.new_pos, legacy.next.pos, "d1={d1}: next pos");
            assert_eq!(ch.restore_slot.is_some(), legacy.next.rollback, "d1={d1}: rollback");
        }
    }

    #[test]
    fn resolve_chain_shapes_depth4() {
        // Hand-checked depth-4 example. drafts d1..d4 = [11,12,13,14]; outs =
        // a_out=11, b_outs=[12,13,99,14]. d1==11,d2==12,d3==13 match; d4==14 but
        // verifier out(3)=b_outs[2]=99 ⇒ MISMATCH at k=3.
        let r = resolve_chain(&[11, 12, 13, 14], /*real*/ 5, /*pos*/ 7, /*a_out*/ 11, &[12, 13, 99, 14]);
        assert_eq!(r.accept_len, 3);
        assert_eq!(r.emitted, vec![5, 11, 12, 13]); // real + out(0..3)
        assert_eq!(r.new_real, 99); // out(3) = the true s_{R+4}
        assert_eq!(r.new_pos, 7 + 4);
        assert_eq!(r.restore_slot, Some(3)); // restore state after pass B_3
        assert_eq!(r.squashed, 1);

        // Full accept: all four match ⇒ k=4, carry = out(4)=b_outs[3].
        let r = resolve_chain(&[11, 12, 13, 14], 5, 7, 11, &[12, 13, 14, 77]);
        assert_eq!(r.accept_len, 4);
        assert_eq!(r.emitted, vec![5, 11, 12, 13, 14]);
        assert_eq!(r.new_real, 77);
        assert_eq!(r.new_pos, 7 + 5);
        assert_eq!(r.restore_slot, None);
        assert_eq!(r.squashed, 0);

        // Immediate miss: d1 wrong ⇒ k=0, restore slot 0, carry = a_out.
        let r = resolve_chain(&[99, 12, 13, 14], 5, 7, 11, &[12, 13, 14, 77]);
        assert_eq!(r.accept_len, 0);
        assert_eq!(r.emitted, vec![5]);
        assert_eq!(r.new_real, 11);
        assert_eq!(r.new_pos, 8);
        assert_eq!(r.restore_slot, Some(0));
        assert_eq!(r.squashed, 4);
    }

    #[test]
    fn chain_snapshot_slot_reuse() {
        // Model ONE PP stage as an append-only processed-position log with a
        // seq_len counter — exactly `KvCache`: snapshot slot = seq_len, restore =
        // truncate(seq_len). The driver snapshots after passes A,B_1..B_{D-1} into
        // ring slots 0..D-1 and, on accept-prefix k<D, restores slot k before the
        // next cycle. Assert: (a) the ring is a FIXED D-slot array reused every
        // cycle, (b) restoring slot k lands on the confirmed frontier pos+k, and
        // (c) the committed-position stream is gap-free & monotone (greedy-exact
        // positions) for every accept pattern.
        let depth = 4usize;
        let ks = [0usize, 4, 1, 3, 2, 4, 0, 2, 3, 1, 4, 0]; // exercise every k incl full-accept
        let mut log: Vec<usize> = Vec::new();
        let mut ring: Vec<Option<usize>> = vec![None; depth]; // fixed ring: seq_len/slot
        let mut pos = 1usize;
        let mut committed: Vec<usize> = Vec::new();
        for &k in &ks {
            assert!(k <= depth);
            // pass A feeds pos; passes B_1..B_D feed pos+1..pos+depth.
            for j in 0..=depth {
                log.push(pos + j);
                if j < depth {
                    ring[j] = Some(log.len()); // snapshot after A (j=0) and B_1..B_{D-1}
                }
            }
            for c in 0..=k {
                committed.push(pos + c); // commit the confirmed prefix
            }
            if k < depth {
                let target = ring[k].expect("slot k populated this cycle");
                log.truncate(target); // KvCache.truncate rollback
                assert_eq!(*log.last().unwrap(), pos + k, "restore slot {k} → confirmed frontier");
            } else {
                assert_eq!(*log.last().unwrap(), pos + depth, "full accept → carry at tail");
            }
            pos += k + 1;
        }
        assert_eq!(ring.len(), depth, "ring is a fixed {depth}-slot array (never grows)");
        let expected: Vec<usize> = (1..=*committed.last().unwrap()).collect();
        assert_eq!(committed, expected, "committed positions must be contiguous/greedy-exact");
    }

    #[test]
    fn resolve_cycle_accept_and_reject_shapes() {
        // Accept: emit [real, a_out], advance pos by 2, use accept-draft, no rollback.
        let inj = Inject { real: 10, pos: 4, draft: 20, rollback: false };
        let r = resolve_cycle(inj, /*a_out*/ 20, /*b_out*/ 30, /*d_pos*/ 99, /*d_pos1*/ 40);
        assert!(r.accept);
        assert_eq!(r.emitted, vec![10, 20]);
        assert_eq!(r.next, Inject { real: 30, pos: 6, draft: 40, rollback: false });

        // Reject: emit [real], advance pos by 1, use reject-draft, rollback set.
        let inj = Inject { real: 10, pos: 4, draft: 21, rollback: false };
        let r = resolve_cycle(inj, /*a_out*/ 20, /*b_out*/ 30, /*d_pos*/ 99, /*d_pos1*/ 40);
        assert!(!r.accept);
        assert_eq!(r.emitted, vec![10]);
        assert_eq!(r.next, Inject { real: 20, pos: 5, draft: 99, rollback: true });
    }
}
