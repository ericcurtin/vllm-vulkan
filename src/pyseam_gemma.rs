// SPDX-License-Identifier: Apache-2.0
//! Per-model pyo3 seam for `gemma` — moved verbatim out of the monolithic
//! `VulkanModel` `#[pymethods]` block in `lib.rs` (Phase A upstream refactor).
//! Behavior-preserving code motion: method bodies are byte-for-byte identical.
//! Kept as separate `#[pymethods] impl VulkanModel` block(s) via pyo3's
//! `multiple-pymethods` feature so a per-model upstream PR can carve this file.
#![allow(clippy::all)]

use crate::*;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;


#[pymethods]
impl VulkanModel {

    /// Megatron tensor-parallel forward for one Gemma4 token (decode).
    ///
    /// Mirrors the validated single-node `forward_layer_gpu_matmuls` per layer,
    /// but each rank runs only its 1/N slice of the attention/MLP projections and
    /// the two per-layer partial sums (after o_proj, after down_proj) are summed
    /// with an all-reduce(SUM) via the Python `all_reduce` callback (vCCL).
    ///
    /// Sharding (done in the loader): q/gate/up column-shard (this rank's heads /
    /// inter slice); o/down row-shard (this rank's input-col partial); k/v are
    /// REPLICATED (num_kv=1 cannot shard, so every rank computes the single KV
    /// head); embed / PLE / all 5 norms / layer_scalar / final norm / lm_head /
    /// softcap are REPLICATED — they run identically per rank because the residual
    /// stream is full/replicated after each all-reduce (standard Megatron TP).
    ///
    /// KV-sharing (layers >= first_kv_shared) reads an earlier layer's K/V; since
    /// the KV head is replicated, every rank already holds the target layer's KV
    /// locally → no cross-rank gather. Validate: TP argmax == single-node argmax.
    fn forward_tp_gemma(&mut self, py: Python<'_>, token_id: u32, pos: usize,
                        all_reduce: PyObject) -> PyResult<Vec<f32>> {
        let cfg = self.inner.config.clone();
        let n = self.tp_size.max(1);
        let _r = self.tp_rank;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let num_q = cfg.num_attention_heads;
        let ple_dim = cfg.hidden_size_per_layer_input;
        let vocab = cfg.vocab_size;
        if num_q % n != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "forward_tp_gemma: num_q={num_q} not divisible by tp_size={n}")));
        }

        // Native vCCL FFI (default) vs Python callback (VLLM_VULKAN_NATIVE_COMM=0).
        let native = self.native_comm_enabled();
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (scratch_addr, scratch_len) = if self.reduce_scratch_handle != 0 {
            (self.reduce_scratch.as_ptr() as usize, self.reduce_scratch.len())
        } else { (0usize, 0usize) };
        // all-reduce a partial (sum across ranks). NATIVE: reduce via the pinned
        // scratch (no per-call regMr) or in-place. CALLBACK: launcher callback.
        // PROFILE (VLLM_VULKAN_PROFILE=1): gemma_allreduce is the 2/layer TP
        // collective total wall; gemma_allr_wire / _copyin / _copyout subdivide
        // it (NESTED, from the vCCL Phases split) so the reduce WIRE is separated
        // from the host copy-in/out marshalling around it. The GPU fence-drain
        // that produces the partial is upstream, inside gemma_attn_mv/gemma_mlp_mv.
        let reduce = |py: Python<'_>, mut partial: Vec<f32>| -> PyResult<Vec<f32>> {
            if native {
                let _t_ar = std::time::Instant::now();
                let ph = if scratch_addr != 0 && partial.len() <= scratch_len {
                    unsafe {
                        vccl_ffi::all_reduce_via_scratch(
                            py, comm, scratch_addr, scratch_len, &mut partial)
                    }.map_err(PyRuntimeError::new_err)?
                } else {
                    vccl_ffi::all_reduce_f32_sum_inplace(py, comm, &mut partial)
                        .map_err(PyRuntimeError::new_err)?
                };
                prof_add("gemma_allreduce", _t_ar);
                prof_add_ns("gemma_allr_copyin", ph.copy_in_ns);
                prof_add_ns("gemma_allr_wire", ph.wire_ns);
                prof_add_ns("gemma_allr_copyout", ph.copy_out_ns);
                Ok(partial)
            } else {
                let _t_ar = std::time::Instant::now();
                let out = all_reduce.call1(py, (partial,))?;
                prof_add("gemma_allreduce", _t_ar);
                out.extract::<Vec<f32>>(py)
            }
        };

        // ── Embedding + PLE (replicated; identical on every rank) ───────────
        let _t_embed = std::time::Instant::now();
        let (mut hidden, ple_inputs) = self.gemma_embed_and_ple(token_id);
        prof_add("gemma_host", _t_embed);

        // LEVER 1 (VLLM_VULKAN_GEMMA_1CB_FULL): the fully-fused per-layer path.
        // Supersedes the partial `gemma_1cb` (checked first). Decided ONCE — pure
        // TP has every layer resident on this rank, so readiness is all-or-
        // nothing (matches `gemma_resident_layer`'s single-node layer-0 gate).
        // Hidden stays GPU-resident in GR_HA/HB across the layer; the only host
        // boundaries are the SDPA + the two all-reduces.
        let full = self.flags.gemma_1cb_full && self.gemma_1cb_ready(0);
        if full {
            self.gemma_full_seed_hidden(&hidden);
        }

        for layer_idx in 0..cfg.num_hidden_layers {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
            let ffn_inter = cfg.layer_intermediate_size(layer_idx);
            if ffn_inter % n != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "forward_tp_gemma: inter={ffn_inter} not divisible by tp_size={n}")));
            }
            let layer_ple = &ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim];

            if full {
                // ── FULLY-FUSED layer: 4 submits, ZERO host norm/residual ─────
                // CB-A (input_norm+qkv+norm+rope) + SDPA + CB-B (o_proj) → partial
                let attn_partial = self.gemma_attn_tp_full_front(&cfg, layer_idx, pos, n);
                // all-reduce #1 → full o_proj, back onto GR_O for the fused mixer.
                let o_proj = reduce(py, attn_partial)?;
                self.gemma_full_set_oproj(&o_proj);
                // CB-CD (post_attn_norm+residual+pre_ffn_norm+gate/up+gelu/mul+down)
                let ff_partial = self.gemma_mlp_tp_full(&cfg, layer_idx, n);
                // all-reduce #2 → full down output, back onto GR_DOWN for the tail.
                let ff_out = reduce(py, ff_partial)?;
                self.gemma_full_set_ffout(&ff_out);
                // CB-E (post_ffn_norm + residual2) → pre-scalar hidden in GR_HA.
                let mut hidden3 = self.gemma_layer_tail_full(&cfg, layer_idx);
                // PLE (E2B only) + layer_scalar on host (tiny), then hidden back
                // to GR_HA for the next layer's front (mirrors the resident tail).
                let _t_h3 = std::time::Instant::now();
                self.gemma_ple_add_tp(&cfg, layer_idx, &mut hidden3, layer_ple);
                prof_add("gemma_host", _t_h3);
                self.gemma_full_seed_hidden(&hidden3);
                hidden = hidden3;
                continue;
            }

            // ── ATTENTION (shared with forward_tp_gemma_verify) ──────────────
            // gemma_host (d): input_layernorm.
            let _t_h1 = std::time::Instant::now();
            let residual = hidden.clone();
            let inln_w = self.gemma_w(&ln("input_layernorm.weight"));
            let x = model::cpu_rms_norm(&hidden, &inln_w, eps);
            prof_add("gemma_host", _t_h1);
            let attn_partial = self.gemma_attn_tp(&cfg, layer_idx, &x, pos, n);
            // all-reduce #1: sum the o_proj partials → full attention output.
            let o_proj = reduce(py, attn_partial)?;

            // post_attn_norm + residual (Gemma2 norm-then-add).
            // gemma_host (d): post_attn_norm + residual + pre_ffn_norm.
            let _t_h2 = std::time::Instant::now();
            let pa_w = self.gemma_w(&ln("post_attention_layernorm.weight"));
            let pa_normed = model::cpu_rms_norm(&o_proj, &pa_w, eps);
            let hidden2: Vec<f32> = residual.iter().zip(pa_normed.iter())
                .map(|(&r, &a)| r + a).collect();
            let residual2 = hidden2.clone();

            // ── MLP (shared with forward_tp_gemma_verify) ────────────────────
            let pf_w = self.gemma_w(&ln("pre_feedforward_layernorm.weight"));
            let ff_in = model::cpu_rms_norm(&hidden2, &pf_w, eps);
            prof_add("gemma_host", _t_h2);
            let ff_partial = self.gemma_mlp_tp(&cfg, layer_idx, &ff_in, n);
            // all-reduce #2: sum the down_proj partials → full MLP output.
            let ff_out = reduce(py, ff_partial)?;

            // post_ffn_norm + residual.
            // gemma_host (d): post_ffn_norm + residual + PLE add + layer_scalar.
            let _t_h3 = std::time::Instant::now();
            let postff_w = self.gemma_w(&ln("post_feedforward_layernorm.weight"));
            let ff_normed = model::cpu_rms_norm(&ff_out, &postff_w, eps);
            let mut hidden3: Vec<f32> = residual2.iter().zip(ff_normed.iter())
                .map(|(&r, &f)| r + f).collect();

            // ── PLE add + layer_scalar (replicated; identical per rank) ───────
            self.gemma_ple_add_tp(&cfg, layer_idx, &mut hidden3, layer_ple);
            prof_add("gemma_host", _t_h3);

            hidden = hidden3;
        }

        // Stash this position's pre-final-norm hidden (INC-5b piece 3 drafter
        // seed) keyed by absolute position, so the driver can fetch the EXACT
        // producing hidden for whichever token ends up the new bonus.
        self.stash_gemma_hidden(pos, &hidden);

        // ── Final norm + LM head + softcap ───────────────────────────────────
        // The tied LM head is `model.embed_tokens.weight`, which the g31b loader
        // keeps ONLY host-side as f16 in `q35_f16_host` (see
        // `load_gemma_nvfp4_weights`: embed goes to `out_f16`, never `host_f32`
        // nor `gpu_weights`), REPLICATED on every rank (not in the TP shard set).
        // `gemma_tp_lmhead_logits` widens ONLY this rank's vocab/n slice
        // (VLLM_VULKAN_TP_SHARD_LMHEAD, default ON) — ~1.4GB transient at TP-4 vs
        // the ~5.6GB full-table widen that OOM-killed the node — and all-gathers
        // the per-rank slices; tp_size==1 / flag-off falls back to the byte-
        // identical full widen. NO embed_scale on the output projection.
        // gemma_lmhead: final RMS norm + TP-sharded LM head + softcap + allgather
        // (once/token tail, not per-layer).
        let _t_lm = std::time::Instant::now();
        let norm_w = self.gemma_w("model.norm.weight");
        let normed = model::cpu_rms_norm(&hidden, &norm_w, eps);
        let cap = cfg.final_logit_softcapping;
        let logits = self.gemma_tp_lmhead_logits(py, &normed, h, vocab, n, cap);
        prof_add("gemma_lmhead", _t_lm);
        logits
    }


    /// INC-5b piece 2 — Design-A batched TENSOR-PARALLEL verify for the
    /// Gemma4-31B TP-4 EAGLE spec-decode arm (the Gemma analog of qwen's
    /// `forward_tp_qwen35_verify`, adapted per `scripts/GEMMA31B_SPEC_PLAN.md`
    /// INC-5). Embeds `tokens` (`[bonus, draft_0..draft_{K-1}]`, T=K+1) at
    /// consecutive positions `start_pos..start_pos+T` and runs every layer on
    /// this rank's 1/N shard, all-reducing the two per-layer `[T*h]` partials
    /// ONCE each — the comm tax is paid ONCE for all T tokens instead of T
    /// single-token `forward_tp_gemma` calls' T reduces (same amortization
    /// that makes the qwen27b-TP4 dense arm and this arm both comm-bound
    /// wins). Returns the FULL replicated `[T*vocab]` logits (every rank
    /// identical), so the driver (`scripts/tp_gemma_spec.py`) does LOCAL
    /// per-rank `resolve_chain`/argmax — no gather.
    ///
    /// Gemma is SIMPLER than qwen here: no GatedDeltaNet, so there is no
    /// recurrent/conv state to capture for rollback (unlike qwen's
    /// `spec_verify_gdn_inputs`) — the per-layer KV just advances token-by-
    /// token exactly as T single-token `forward_tp_gemma` calls would
    /// (`gemma_attn_tp` appends each position's K/V in order), and partial-
    /// accept rollback (`gemma_tp_verify_rollback`) is JUST a KV write-counter
    /// rewind (delegates to `Gemma4Model::verify_rollback`, already landed in
    /// INC-4). Layer-major T-loop mirrors `Gemma4Model::forward_verify_core`'s
    /// reordering proof (a layer's output at position i depends only on that
    /// layer's own input at i and its OWN cache contents at positions < i, so
    /// layer-major and token-major compute IDENTICAL numbers).
    ///
    /// GPU/multi-rank correctness (TP logits == single-node
    /// `forward_verify_core`, argmax-exact) is CLUSTER-DEFERRED — this is
    /// authored against the proven single-token TP mixers
    /// (`gemma_attn_tp`/`gemma_mlp_tp`/`gemma_ple_add_tp`) but has no Mac
    /// Vulkan device to GPU-validate against (mirrors the qwen TP-4 precedent
    /// and INC-1b's "load-time shard only, GPU/multi-rank EXECUTION is
    /// cluster-deferred" note).
    fn forward_tp_gemma_verify(&mut self, py: Python<'_>, tokens: Vec<u32>, start_pos: usize,
                               all_reduce: PyObject) -> PyResult<Vec<f32>> {
        self.forward_tp_gemma_verify_impl(py, tokens, start_pos, all_reduce)
    }


    /// INC-5b piece 2 — TP batched verify, argmax form: runs
    /// `forward_tp_gemma_verify` and argmaxes EACH of the T replicated-logit
    /// rows in Rust, returning `[out(0)..out(T-1)]` (the exact per-position
    /// candidates the driver's `resolve_chain` needs) without marshalling
    /// `T*vocab` floats to Python — mirrors
    /// `forward_tp_qwen35_verify_argmax`/`forward_qwen35_verify_argmax`.
    fn forward_tp_gemma_verify_argmax(&mut self, py: Python<'_>, tokens: Vec<u32>, start_pos: usize,
                                      all_reduce: PyObject) -> PyResult<Vec<u32>> {
        let logits = self.forward_tp_gemma_verify_impl(py, tokens, start_pos, all_reduce)?;
        let vocab = self.inner.config.vocab_size;
        if vocab == 0 || logits.len() % vocab != 0 {
            return Err(PyRuntimeError::new_err("forward_tp_gemma_verify_argmax: bad [T*vocab] result"));
        }
        let t = logits.len() / vocab;
        let mut outs = Vec::with_capacity(t);
        for ti in 0..t {
            let row = &logits[ti * vocab..(ti + 1) * vocab];
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in row.iter().enumerate() {
                if v > bv { bv = v; bi = i; }
            }
            outs.push(bi as u32);
        }
        Ok(outs)
    }


    /// INC-5b piece 2 — TP partial-accept ROLLBACK, the TP analog of the
    /// single-node `verify_rollback` (INC-4). After a
    /// `forward_tp_gemma_verify` + LOCAL `resolve_chain` giving
    /// `accept_len` (0..T-1), rewind every layer's KV write-counter to
    /// `start_pos + accept_len + 1`. Unlike qwen's TP rollback (which must
    /// re-scan the sharded GatedDeltaNet mixer), Gemma has NO recurrent state
    /// — every rank's KV bytes are already correct at the rewound frontier
    /// (`KvCache::truncate` is counter-only), so this is IDENTICAL work on
    /// every rank and needs no all-reduce. Delegates straight to
    /// `Gemma4Model::verify_rollback` (`self.inner`), which already implements
    /// exactly this contract for the single-node verify path.
    fn gemma_tp_verify_rollback(&mut self, start_pos: usize, t: usize, accept_len: usize) -> PyResult<()> {
        if t == 0 {
            return Ok(());
        }
        self.inner.verify_rollback(start_pos, t, accept_len);
        Ok(())
    }


    /// GEMMA GATE-2a — single-node GPU bit-exactness of the batched TP verify.
    /// The Gemma analog of qwen's `debug_tp_qwen35_verify_vs_serial`: at TP-N
    /// (all ranks, same `all_reduce` the driver uses — native vCCL FFI or the
    /// Python callback), prove that the batched `forward_tp_gemma_verify` emits
    /// per-position `[vocab]` logits BIT-IDENTICAL to running T serial
    /// `forward_tp_gemma` calls at consecutive positions from `start_pos`. This
    /// is the mandatory gate the qwen arm SKIPPED (it shipped straight to an A/B
    /// that failed with GPU-only T>1 bugs — flash layout / f16-vs-f32 / quant
    /// dispatch that are INVISIBLE at T=1). `tokens` is the T=DEPTH+1 verify
    /// batch (`[bonus, draft_0..draft_{K-1}]`; the token VALUES are irrelevant
    /// to the gate — both paths embed the IDENTICAL sequence, only that the two
    /// paths AGREE matters).
    ///
    /// STATE snapshot/restore is FAR simpler than qwen's: Gemma has NO
    /// GatedDeltaNet, so the ONLY state either forward advances is each layer's
    /// `KvCache::seq_len` write-counter (KV bytes are overwrite-in-place; see
    /// `KvCache::truncate`). kv-shared layers never advance their own counter
    /// (both forwards leave them untouched), so snapshotting EVERY cache's
    /// `seq_len` and truncating back is robust regardless of sharing. There is
    /// no `spec_verify_gdn_inputs`/`spec_verify_span` to clear (the gemma verify
    /// sets no pending recurrent/span state — cf. `forward_tp_gemma_verify_impl`).
    /// The verify uses the TARGET's own forward (not the assistant/drafter — the
    /// drafter is a separate path); this gate is target-verify-vs-serial only.
    ///
    /// Returns per-position `(cos, maxdiff, argmax_serial, argmax_batched)` over
    /// the FULL replicated `[vocab]` logits (both forwards go through the same
    /// sharded lm-head → comparable), so the driver gates on
    /// `cos==1.0 && argmax_serial==argmax_batched` at EVERY position. Leaves
    /// resident state exactly as it was at entry (final KV-counter restore).
    fn debug_gemma_verify_vs_serial(&mut self, py: Python<'_>, tokens: Vec<u32>,
                                    start_pos: usize, all_reduce: PyObject)
        -> PyResult<Vec<(f64, f64, i64, i64)>>
    {
        let t = tokens.len();
        if t == 0 {
            return Err(PyRuntimeError::new_err("debug_gemma_verify_vs_serial: empty tokens"));
        }
        // Snapshot every layer's KV write-counter (the ONLY state either forward
        // advances — no GatedDeltaNet, KV bytes overwrite-in-place).
        let snap: Vec<usize> = self.inner.kv_caches.iter().map(|c| c.seq_len).collect();
        // Serial reference: single-token TP forward per position (advances the
        // sharded KV state), full replicated [vocab] logits — the exact path the
        // PP_SPEC=0 baseline decodes with.
        let mut serial: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (i, &tok) in tokens.iter().enumerate() {
            serial.push(self.forward_tp_gemma(py, tok, start_pos + i, all_reduce.clone_ref(py))?);
        }
        // Restore to the pre-serial KV frontier, then the batched verify over all
        // T tokens from the IDENTICAL start state.
        for (l, c) in self.inner.kv_caches.iter_mut().enumerate() {
            c.truncate(snap[l]);
        }
        let batched = self.forward_tp_gemma_verify_impl(py, tokens, start_pos, all_reduce.clone_ref(py))?;
        // Restore once more so the harness is state-neutral for the caller
        // (harness, not a real verify — nothing rolls it back).
        for (l, c) in self.inner.kv_caches.iter_mut().enumerate() {
            c.truncate(snap[l]);
        }
        if batched.len() % t != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "debug_gemma_verify_vs_serial: verify result {} not divisible by T={t}", batched.len())));
        }
        let width = batched.len() / t;
        let argmax = |v: &[f32]| -> i64 {
            let (mut bi, mut bv) = (0i64, f32::NEG_INFINITY);
            for (i, &x) in v.iter().enumerate() { if x > bv { bv = x; bi = i as i64; } }
            bi
        };
        let mut out = Vec::with_capacity(t);
        for i in 0..t {
            let s = &serial[i];
            let b = &batched[i * width..(i + 1) * width];
            let n = s.len().min(b.len());
            let (mut dot, mut ns, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f64);
            for j in 0..n {
                dot += s[j] as f64 * b[j] as f64;
                ns += s[j] as f64 * s[j] as f64;
                nb += b[j] as f64 * b[j] as f64;
                maxd = maxd.max((s[j] as f64 - b[j] as f64).abs());
            }
            let cos = dot / (ns.sqrt() * nb.sqrt() + 1e-12);
            out.push((cos, maxd, argmax(s), argmax(b)));
        }
        Ok(out)
    }


    /// INC-5b piece 3 — fetch the target's pre-`model.norm` hidden stashed at
    /// absolute position `pos` (`stash_gemma_hidden`, `gemma_forward.rs`).
    /// This is the EAGLE drafter's step-0 `recurrent_hidden` seed
    /// (`GEMMA31B_SPEC_PLAN.md` §1.1): the backbone `[h]` hidden that
    /// PRODUCED the token at `pos+1` — i.e. the driver calls this with the
    /// new bonus token's OWN position (`R-1` after `R += accept_len+1`) to
    /// seed the next draft cycle. Replicated (identical on every TP rank —
    /// no gather needed). Errors if nothing was stashed at `pos` (fell off
    /// the ring or no forward/verify has reached it yet).
    fn gemma_hidden_for_pos(&self, pos: usize) -> PyResult<Vec<f32>> {
        self.gemma_hidden_at(pos).ok_or_else(|| PyRuntimeError::new_err(format!(
            "gemma_hidden_for_pos: no hidden stashed at pos={pos} (ring cap or never forwarded)")))
    }


    /// INC-5b piece 3 — the TARGET's `embed_tokens(token_id) * embed_scale`
    /// (backbone `[h]`, REPLICATED/identical on every TP rank) — the
    /// `prev_token_embed` half of the EAGLE drafter's `inputs_embeds`
    /// (`GEMMA31B_SPEC_PLAN.md` §1.1). Reuses `gemma_embed_and_ple` (which for
    /// the 31B, `hidden_size_per_layer_input=0` / no PLE, returns exactly this
    /// with an empty PLE tail) rather than duplicating the f16-host embed
    /// lookup + scale.
    fn gemma_embed_token(&mut self, token_id: u32) -> PyResult<Vec<f32>> {
        Ok(self.gemma_embed_and_ple(token_id).0)
    }


    /// INC-5b piece 3 — this rank's LOCAL K/V for one target layer, up to the
    /// current seq_len (`[seq_len, local_kv_heads, head_dim]` flat, matching
    /// `cpu_sdpa`'s expected layout). Used to assemble the EAGLE drafter's
    /// borrowed-KV all-gather (`GEMMA31B_SPEC_PLAN.md` §1.5): the driver reads
    /// this on EVERY rank for the two borrowed target layers (58 last-sliding,
    /// 59 last-full for the 31B) and concatenates the per-rank kv-head slices
    /// via `vccl.Communicator.all_gather` (a python-level primitive — no new
    /// FFI needed) to reassemble the FULL kv-head set the Q-only drafter's
    /// cross-attention requires. Returns `(k_flat, v_flat, seq_len)`.
    ///
    /// ABSOLUTE-POSITION contract, so it REFUSES a wrapped ring. The returned
    /// layout is `[seq_len, ...]` in ascending absolute order, which a
    /// `capacity < seq_len` sliding ring simply no longer holds — it retains
    /// only the last `capacity` positions, in slot order. `k_up_to_now()` would
    /// slice `seq_len * stride` out of a `capacity * stride` buffer, so the
    /// wrapped case is a raw index panic with no explanation; this names the
    /// cause instead. (`windowed_view` is the ring-correct reader, but it
    /// returns `window` rows, not `seq_len` — a different contract than the
    /// drafter's all-gather expects, so widening it is INC-5b work, not a
    /// silent substitution here.)
    fn gemma_kv_layer(&self, layer_idx: usize) -> PyResult<(Vec<f32>, Vec<f32>, usize)> {
        let cache = self.inner.kv_caches.get(layer_idx).ok_or_else(|| PyRuntimeError::new_err(
            format!("gemma_kv_layer: layer_idx {layer_idx} out of range ({} caches)", self.inner.kv_caches.len())))?;
        if cache.has_wrapped() {
            return Err(PyRuntimeError::new_err(format!(
                "gemma_kv_layer: layer {layer_idx} KV ring has wrapped (seq_len {} > capacity {}); \
                 the absolute-position [seq_len, kv_heads, head_dim] layout this returns no longer \
                 exists. Set VLLM_VULKAN_KV_RING_DISABLE=1 to keep full-size caches.",
                cache.seq_len, cache.capacity)));
        }
        Ok((cache.k_up_to_now().to_vec(), cache.v_up_to_now().to_vec(), cache.seq_len))
    }


    /// Rust-native body of `forward_tp_gemma_verify` (INC-5b piece 2). See
    /// that method's doc comment for the full design rationale.
    fn forward_tp_gemma_verify_impl(
        &mut self,
        py: Python<'_>,
        tokens: Vec<u32>,
        start_pos: usize,
        all_reduce: PyObject,
    ) -> PyResult<Vec<f32>> {
        let cfg = self.inner.config.clone();
        let n = self.tp_size.max(1);
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let ple_dim = cfg.hidden_size_per_layer_input;
        let vocab = cfg.vocab_size;
        let num_q = cfg.num_attention_heads;
        if num_q % n != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "forward_tp_gemma_verify: num_q={num_q} not divisible by tp_size={n}")));
        }
        let t = tokens.len();
        if t == 0 {
            return Ok(Vec::new());
        }

        // Enlarge the registered reduce scratch to [T*h] (single-token TP only
        // registers [h]); no-op / safe fallback when registration is off.
        self.ensure_reduce_scratch(t * h);
        let native = self.native_comm_enabled();
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (scratch_addr, scratch_len) = if self.reduce_scratch_handle != 0 {
            (self.reduce_scratch.as_ptr() as usize, self.reduce_scratch.len())
        } else { (0usize, 0usize) };
        let reduce = move |py: Python<'_>, mut partial: Vec<f32>| -> PyResult<Vec<f32>> {
            if native {
                if scratch_addr != 0 && partial.len() <= scratch_len {
                    unsafe {
                        vccl_ffi::all_reduce_via_scratch(
                            py, comm, scratch_addr, scratch_len, &mut partial)
                    }.map_err(PyRuntimeError::new_err)?;
                } else {
                    vccl_ffi::all_reduce_f32_sum_inplace(py, comm, &mut partial)
                        .map_err(PyRuntimeError::new_err)?;
                }
                Ok(partial)
            } else {
                let out = all_reduce.call1(py, (partial,))?;
                out.extract::<Vec<f32>>(py)
            }
        };

        // Embed all T tokens + their per-position PLE inputs up front
        // (replicated on every rank — pure functions of the token id).
        let mut hiddens: Vec<Vec<f32>> = Vec::with_capacity(t);
        let mut ples: Vec<Vec<f32>> = Vec::with_capacity(t);
        for &tok in &tokens {
            let (hid, ple) = self.gemma_embed_and_ple(tok);
            hiddens.push(hid);
            ples.push(ple);
        }

        // Layer-major sweep (see doc comment): for each layer, advance every
        // position in position order so each position's self-attention sees
        // exactly the KV appended by the earlier positions at this layer.
        for layer_idx in 0..cfg.num_hidden_layers {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
            let ffn_inter = cfg.layer_intermediate_size(layer_idx);
            if ffn_inter % n != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "forward_tp_gemma_verify: inter={ffn_inter} not divisible by tp_size={n}")));
            }

            // ── ATTENTION: T sequential sharded mixers → [T*h] partial ──────
            let inln_w = self.gemma_w(&ln("input_layernorm.weight"));
            let residuals: Vec<Vec<f32>> = hiddens.clone();
            let mut attn_partial = vec![0.0f32; t * h];
            for ti in 0..t {
                let x = model::cpu_rms_norm(&hiddens[ti], &inln_w, eps);
                let p = self.gemma_attn_tp(&cfg, layer_idx, &x, start_pos + ti, n);
                attn_partial[ti * h..(ti + 1) * h].copy_from_slice(&p);
            }
            let attn_out = reduce(py, attn_partial)?;

            // post_attn_norm + residual (Gemma2 norm-then-add), per position.
            let pa_w = self.gemma_w(&ln("post_attention_layernorm.weight"));
            let mut hidden2s: Vec<Vec<f32>> = Vec::with_capacity(t);
            for ti in 0..t {
                let pa_normed = model::cpu_rms_norm(&attn_out[ti * h..(ti + 1) * h], &pa_w, eps);
                let h2: Vec<f32> = residuals[ti].iter().zip(pa_normed.iter())
                    .map(|(&r, &a)| r + a).collect();
                hidden2s.push(h2);
            }

            // ── MLP: T sequential sharded mixers → [T*h] partial ────────────
            let pf_w = self.gemma_w(&ln("pre_feedforward_layernorm.weight"));
            let mut ff_partial = vec![0.0f32; t * h];
            for ti in 0..t {
                let ff_in = model::cpu_rms_norm(&hidden2s[ti], &pf_w, eps);
                let p = self.gemma_mlp_tp(&cfg, layer_idx, &ff_in, n);
                ff_partial[ti * h..(ti + 1) * h].copy_from_slice(&p);
            }
            let ff_out = reduce(py, ff_partial)?;

            // post_ffn_norm + residual + PLE add + layer_scalar, per position
            // (all replicated — identical per rank, no reduce needed).
            let postff_w = self.gemma_w(&ln("post_feedforward_layernorm.weight"));
            for ti in 0..t {
                let ff_normed = model::cpu_rms_norm(&ff_out[ti * h..(ti + 1) * h], &postff_w, eps);
                let mut hidden3: Vec<f32> = hidden2s[ti].iter().zip(ff_normed.iter())
                    .map(|(&r, &f)| r + f).collect();
                let layer_ple = &ples[ti][layer_idx * ple_dim..(layer_idx + 1) * ple_dim];
                self.gemma_ple_add_tp(&cfg, layer_idx, &mut hidden3, layer_ple);
                hiddens[ti] = hidden3;
            }
        }

        // Stash EACH position's pre-final-norm hidden (INC-5b piece 3 drafter
        // seed), keyed by absolute position `start_pos+ti` — row `ti` is the
        // hidden that PRODUCED `logits[ti]`'s argmax candidate, so whichever
        // position ends up the accepted frontier's new bonus token, the
        // driver fetches the EXACT producing hidden (not necessarily the
        // last row — a partial accept's new bonus is an EARLIER position).
        for ti in 0..t {
            self.stash_gemma_hidden(start_pos + ti, &hiddens[ti]);
        }

        // Final norm + LM head + softcap, per position. The tied LM head
        // (`model.embed_tokens.weight`) is kept ONLY host-side as f16 in
        // `q35_f16_host` by the g31b loader, REPLICATED on every rank (not in the
        // TP shard set) — `gemma_tp_lmhead_logits` widens ONLY this rank's vocab/n
        // slice (VLLM_VULKAN_TP_SHARD_LMHEAD, default ON; ~1.4GB peak at TP-4 vs
        // the ~5.6GB full-table widen that OOM-killed the node) and all-gathers
        // per position; tp_size==1 / flag-off falls back to the byte-identical
        // full widen. One all-gather per position (the verify path is the batched
        // spec arm, not the hot single-token decode).
        let norm_w = self.gemma_w("model.norm.weight");
        let cap = cfg.final_logit_softcapping;
        let mut logits = vec![0.0f32; t * vocab];
        for ti in 0..t {
            let normed = model::cpu_rms_norm(&hiddens[ti], &norm_w, eps);
            let row = self.gemma_tp_lmhead_logits(py, &normed, h, vocab, n, cap)?;
            logits[ti * vocab..(ti + 1) * vocab].copy_from_slice(&row);
        }
        Ok(logits)
    }


    /// Pipeline-parallel Gemma4 stage. Like `forward_pp` but carries Gemma's
    /// extra cross-stage state in the message: `[hidden(H) ‖ ple_inputs(total_ple)
    /// ‖ target_kv]`. The first stage embeds + computes ple_inputs (it owns the
    /// PLE table); every stage forwards ple_inputs unchanged. KV-shared layers
    /// (>= first_kv_shared) read the KV of ≤2 earlier "target" layers, which may
    /// live upstream — so each owning stage writes that layer's per-token K/V
    /// into `target_kv`, and downstream stages append it to a replica cache
    /// before running their shared layers. Last stage returns logits.
    fn forward_pp_gemma(&mut self, token_id: u32, msg_in: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        if self.qwen.is_some() {
            return Err(PyRuntimeError::new_err("forward_pp_gemma is for Gemma models"));
        }
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let ple_dim = cfg.hidden_size_per_layer_input;
        let total_ple = cfg.num_hidden_layers * ple_dim;

        // KV-share target layers: see `gemma_pp_kv_targets` doc comment.
        // Empty for a model with no cross-stage KV sharing (g12b); non-empty
        // (≤2 entries, sized per-layer) for a model that does (e2b).
        let targets = gemma_pp_kv_targets(&cfg);
        let tkv_lens = gemma_pp_tkv_lens(&cfg, &targets);
        let tkv_total: usize = tkv_lens.iter().map(|&l| 2 * l).sum();

        // Decode inbound message (or compute fresh on the first stage).
        let (mut hidden, ple_inputs, mut tkv) = if self.pp_first {
            let (hidden, ple) = self.gemma_embed_and_ple(token_id);
            (hidden, ple, vec![0f32; tkv_total])
        } else {
            let want = h + total_ple + tkv_total;
            if msg_in.len() != want {
                return Err(PyRuntimeError::new_err(format!(
                    "forward_pp_gemma msg len {} != {want}", msg_in.len())));
            }
            (msg_in[0..h].to_vec(),
             msg_in[h..h + total_ple].to_vec(),
             msg_in[h + total_ple..].to_vec())
        };

        // Append carried target KV to replica caches for targets NOT owned here.
        let mut off = 0usize;
        for (i, &t) in targets.iter().enumerate() {
            let l = tkv_lens[i];
            if t < self.pp_start || t >= self.pp_end {
                self.inner.kv_caches[t].append(&tkv[off..off + l], &tkv[off + l..off + 2 * l]);
            }
            off += 2 * l;
        }

        // Debug per-layer hidden trace (`VLLM_VULKAN_PP_TRACE=<prefix>`):
        // writes `<prefix>_L<idx>_p<pos>.f32` after every layer, for offline
        // CPU-vs-GPU layer bisection. Off unless the env var is set.
        let trace_prefix = std::env::var("VLLM_VULKAN_PP_TRACE").ok();

        // Run this stage's layers; capture any target layer's fresh KV.
        for layer_idx in self.pp_start..self.pp_end {
            let layer_ple = &ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim];
            hidden = self.forward_layer_gpu_matmuls(layer_idx, &hidden, pos, layer_ple);
            if let Some(pfx) = &trace_prefix {
                let _ = std::fs::write(
                    format!("{pfx}_L{layer_idx}_p{pos}.f32"),
                    f32_slice_to_bytes(&hidden));
            }
            if let Some(i) = targets.iter().position(|&t| t == layer_idx) {
                let l = tkv_lens[i];
                let stride = cfg.layer_num_kv_heads(layer_idx) * cfg.layer_head_dim(layer_idx);
                let cache = &self.inner.kv_caches[layer_idx];
                // RING SLOT, not the absolute position. A sliding layer's cache
                // is a `capacity`-slot ring written at `pos % capacity`
                // (`KvCache::append`), and `capacity < max_seq_len`, so the
                // absolute `seq_len - 1` indexes off the end of `cache.k` the
                // moment the ring wraps. See `crate::kv_last_slot`.
                let p = crate::kv_last_slot(cache);
                let o: usize = tkv_lens[..i].iter().map(|&x| 2 * x).sum();
                tkv[o..o + l].copy_from_slice(&cache.k[p * stride..p * stride + l]);
                tkv[o + l..o + 2 * l].copy_from_slice(&cache.v[p * stride..p * stride + l]);
            }
        }

        if self.pp_last {
            Ok(self.gemma_final(&hidden))
        } else {
            let mut out = Vec::with_capacity(h + total_ple + tkv_total);
            out.extend_from_slice(&hidden);
            out.extend_from_slice(&ple_inputs);
            out.extend_from_slice(&tkv);
            Ok(out)
        }
    }


    /// Last-stage twin of `forward_pp_gemma` that argmaxes in Rust and returns
    /// just `(argmax_token, max_logit)` — the full ~262k-vocab logit Vec NEVER
    /// crosses the pyo3 boundary (the same `Vec<f32>→PyList` vocab marshal that
    /// cost ~15 ms/tok on the qwen3.6 PP-5 last stage; see
    /// `forward_pp_qwen35_argmax`). Gemma has no fused `pp_step_gemma`, so its PP
    /// driver otherwise pays the marshal on every decode token. `forward_pp_gemma`
    /// already returns the full logits on `pp_last`, so this just calls it
    /// Rust-internally and argmaxes with the same strict-`>` first-max tie-break
    /// as the driver's python argmax → byte-identical token. ONLY valid on the
    /// last PP stage (mid stages emit a composite hidden+ple+kv message, not a
    /// logit vector — argmaxing that is meaningless), hence the guard.
    fn forward_pp_gemma_argmax(&mut self, token_id: u32, msg_in: Vec<f32>, pos: usize) -> PyResult<(u32, f32)> {
        if !self.pp_last {
            return Err(PyRuntimeError::new_err(
                "forward_pp_gemma_argmax is only valid on the last PP stage"));
        }
        let out = self.forward_pp_gemma(token_id, msg_in, pos)?;
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in out.iter().enumerate() {
            if v > bv { bv = v; bi = i; }
        }
        Ok((bi as u32, bv))
    }


    /// FUSED native-vCCL PP DECODE step for Gemma (mirrors `pp_step_nemotron`/
    /// `pp_step_kimi`/`pp_step_laguna`): recv the previous stage's composite
    /// `[hidden+ple+target_kv]` message (if not first) natively into a
    /// `comm_register`'d scratch → run the SAME `forward_pp_gemma` compute body
    /// (so the cross-stage KV-share append AND the `gemma_hidden_ring` stash are
    /// preserved BIT-FOR-BIT — only the transport around the compute changes) →
    /// native-send the composite message onward OR Rust-argmax the logits on the
    /// last stage. The composite message and the ~262k-vocab logits NEVER cross
    /// the pyo3 boundary — only the token id in and `(argmax token, logit)` out
    /// on the last stage. Kills the per-hop `Vec<f32>→PyList`+`comm.send/recv`
    /// marshal `forward_pp_gemma` otherwise pays (Gemma-12B is PP-2 = one hop,
    /// so ~2 ms/tok — small; parity/correctness dominates).
    ///
    /// `recv_from < 0` ⇒ first stage (embeds `token_id`; the recv is skipped and
    /// `forward_pp_gemma` is fed an empty message, which it ignores there);
    /// `send_to < 0` ⇒ last stage (returns `Some((tok, logit))`, argmaxed with
    /// the same strict-`>` first-max tie-break as `forward_pp_gemma_argmax` and
    /// the python driver → byte-identical token). The composite message is
    /// `pp_gemma_msg_len()` f32 on EVERY stage (see its doc — same length in and
    /// out on mid stages). Requires `set_collective_comm` +
    /// `VLLM_VULKAN_NATIVE_COMM!=0`; the driver should gate the call on
    /// `VLLM_VULKAN_GEMMA_NATIVE_HOP`.
    fn pp_step_gemma(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        pos: usize,
        recv_from: i32,
        send_to: i32,
    ) -> PyResult<Option<(u32, f32)>> {
        if self.qwen.is_some() {
            return Err(PyRuntimeError::new_err("pp_step_gemma is for Gemma models"));
        }
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_gemma: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let msg_len = self.pp_gemma_msg_len();
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);

        // Pin the persistent [msg_len] PP-hop scratch (recv + send sides) with the
        // RDMA transport ONCE — vCCL's send/recv then skip the per-call
        // `ibv_reg_mr`/dereg temp-MR (the "buffer not registered with the comm"
        // WARN). Mirrors `pp_step_nemotron`; gated by `VLLM_VULKAN_REG_REDUCE` and
        // libvccl exposing `vcclCommRegister`; on failure we fall back to the
        // fresh-Vec `recv_f32`/`send_f32` path (still native — no PyList — just a
        // per-call regMr). The composite message is identical on every stage, so
        // both recv (in) and send (out) scratch are `msg_len`; the LAST stage
        // produces logits (never sent) and so allocates no send scratch.
        let want_reg = self.flags.reg_reduce
            && !comm.is_null()
            && vccl_ffi::registration_available();
        if want_reg {
            if do_recv && self.gemma_pp_hop.recv_handle == 0 {
                self.gemma_pp_hop.recv_scratch = vec![0.0f32; msg_len];
                let addr = self.gemma_pp_hop.recv_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, msg_len * std::mem::size_of::<f32>()) {
                    Ok(hd) => self.gemma_pp_hop.recv_handle = hd,
                    Err(e) => {
                        log::warn!("pp_step_gemma: register recv scratch failed: {e}; per-call regMr");
                        self.gemma_pp_hop.recv_scratch.clear();
                    }
                }
            }
            if !is_last && self.gemma_pp_hop.send_handle == 0 {
                self.gemma_pp_hop.send_scratch = vec![0.0f32; msg_len];
                let addr = self.gemma_pp_hop.send_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, msg_len * std::mem::size_of::<f32>()) {
                    Ok(hd) => self.gemma_pp_hop.send_handle = hd,
                    Err(e) => {
                        log::warn!("pp_step_gemma: register send scratch failed: {e}; per-call regMr");
                        self.gemma_pp_hop.send_scratch.clear();
                    }
                }
            }
        }

        // 1) recv the previous stage's composite message INTO the registered
        //    scratch (fast pre-pinned MR), or empty on the first stage (the compute
        //    embeds token_id and ignores the message there).
        let msg_in: Vec<f32> = if do_recv {
            if self.gemma_pp_hop.recv_handle != 0 {
                vccl_ffi::recv_f32_into(py, comm, &mut self.gemma_pp_hop.recv_scratch, recv_from)
                    .map_err(PyRuntimeError::new_err)?;
                self.gemma_pp_hop.recv_scratch.clone()
            } else {
                vccl_ffi::recv_f32(py, comm, msg_len, recv_from).map_err(PyRuntimeError::new_err)?
            }
        } else {
            Vec::new()
        };

        // 2) run the EXACT `forward_pp_gemma` compute body — same KV-share append,
        //    same per-layer forward, same `gemma_hidden_ring` stash, same final
        //    norm/logits. Only the transport around it changed (PyList → native
        //    scratch), so the output is byte-identical to the marshalled path.
        let out = self.forward_pp_gemma(token_id, msg_in, pos)?;

        // 3) mid/first stage: native-send the composite [msg_len] onward FROM the
        //    registered scratch (fall back to a fresh-Vec send if unregistered or
        //    the width differs). Last stage: Rust argmax the logits.
        if !is_last {
            if self.gemma_pp_hop.send_handle != 0 && out.len() == self.gemma_pp_hop.send_scratch.len() {
                self.gemma_pp_hop.send_scratch.copy_from_slice(&out);
                vccl_ffi::send_f32(py, comm, &self.gemma_pp_hop.send_scratch, send_to)
                    .map_err(PyRuntimeError::new_err)?;
            } else {
                vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            }
            Ok(None)
        } else {
            // Strict-`>` first-max tie-break, identical to `forward_pp_gemma_argmax`.
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in out.iter().enumerate() {
                if v > bv { bv = v; bi = i; }
            }
            Ok(Some((bi as u32, bv)))
        }
    }


    /// DISTRIBUTED-SERVE twin of `pp_step_gemma` (mirrors `pp_step_qwen35_logits`
    /// / `pp_step_laguna_logits`): instead of argmaxing on the last stage, the
    /// last stage rings the FULL `[vocab]` logits back to rank0 (raw f32 over
    /// vCCL — NO `Vec<f32>→PyList` marshal) so vLLM's Sampler on rank0 sees the
    /// whole distribution. This is the `pp_step_gemma_logits` seam the general
    /// `scripts/serve_dist.py` launcher resolves by name
    /// (`pp_step_<model_type>_logits`) for `--model-type gemma`.
    ///
    /// The launcher calls it pos-free — `(token_id, recv_from, send_to,
    /// last_rank)` — so the decode `pos` is derived from the resident KV
    /// `seq_len` (the max over this stage's owned/replica caches, identical to
    /// `Qwen35Model::current_decode_pos`; `KvCache::seq_len` is the ABSOLUTE
    /// token position, unchanged by windowing), which prefill fills to the
    /// prompt length and each decode step advances by one — exactly the `pos`
    /// the Python PP driver passed to `pp_step_gemma` explicitly.
    ///
    /// Bit-exact with `pp_step_gemma`'s last-stage logits: it runs the SAME
    /// `forward_pp_gemma` compute body (same composite `[hidden‖ple‖target_kv]`
    /// message decode, same per-layer forward, same KV-share append, same final
    /// norm/logits) — only the transport of the RESULT differs (full vocab rung
    /// back vs argmax returned). Gemma is dense (no dual per-layer state), so the
    /// inter-stage message is the composite `pp_gemma_msg_len()` on every hop
    /// (NOT qwen3.6's bare `[H]`); g12b (no PLE, no KV-share) degenerates that to
    /// `[H]`.
    ///
    ///  - FIRST  (`recv_from<0`, `send_to>=0`): embed `token_id` → `[msg_len]` →
    ///    send onward; then recv the `[vocab]` ring-back from `last_rank`; return
    ///    it.
    ///  - MID    (`recv_from>=0`, `send_to>=0`): recv `[msg_len]` → stage forward
    ///    → send `[msg_len]` onward; return `None`.
    ///  - LAST   (`recv_from>=0`, `send_to<0`): recv `[msg_len]` → decode →
    ///    `[vocab]`; send `[vocab]` to rank0; return `None`.
    ///  - STANDALONE N=1 (`recv_from<0` && `send_to<0`): embed → `[vocab]`;
    ///    return `Some([vocab])` with no wire.
    ///
    /// Uses the same plain `send_f32`/`recv_f32` transport as
    /// `pp_step_qwen35_logits` (the `comm_register`'d hop of `pp_step_gemma` is a
    /// later perf lever for this seam). Requires `set_collective_comm` +
    /// `VLLM_VULKAN_NATIVE_COMM!=0`.
    fn pp_step_gemma_logits(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
    ) -> PyResult<Option<Vec<f32>>> {
        if self.qwen.is_some() {
            return Err(PyRuntimeError::new_err("pp_step_gemma_logits is for Gemma models"));
        }
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_gemma_logits: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let vocab = self.inner.config.vocab_size;
        let msg_len = self.pp_gemma_msg_len();
        // Decode pos = ABSOLUTE token position = max resident KV seq_len across
        // this stage's caches (owned appending layers + any target replicas),
        // exactly `Qwen35Model::current_decode_pos`'s contract. Prefill leaves it
        // at the prompt length; each decode step advances one. Read BEFORE the
        // forward (the forward appends one K/V per layer, bumping seq_len).
        let pos = self.inner.kv_caches.iter().map(|c| c.seq_len).max().unwrap_or(0);
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;

        // 1) recv the previous stage's [msg_len] composite (GIL dropped inside
        //    recv_f32), or empty on the first stage (it embeds token_id).
        let msg_in: Vec<f32> = if do_recv {
            vccl_ffi::recv_f32(py, comm, msg_len, recv_from).map_err(PyRuntimeError::new_err)?
        } else {
            Vec::new()
        };

        // 2) resident stage forward. [msg_len] composite on mid stages, [vocab]
        //    on the last. BIT-IDENTICAL to `pp_step_gemma`/`forward_pp_gemma`.
        let out = self.forward_pp_gemma(token_id, msg_in, pos)?;

        // 3) route the result.
        if is_first && is_last {
            // STANDALONE N=1: rank0 is both first and last; `out` is already [vocab].
            return Ok(Some(out));
        }
        if !is_last {
            // FIRST / MID: forward `[msg_len]` onward, then (rank0 only) recv the ring-back.
            vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            if is_first {
                // rank0: ring the [vocab] back from the last stage through the
                // registered `pp_vocab_ring` (no per-step temp-MR).
                let logits = self.pp_recv_vocab(py, vocab, last_rank)?;
                Ok(Some(logits))
            } else {
                Ok(None)
            }
        } else {
            // LAST stage: ring the full [vocab] back to rank0 (peer 0) through the
            // registered `pp_vocab_ring`. No argmax.
            self.pp_send_vocab(py, &out, 0)?;
            Ok(None)
        }
    }


    /// FAST-sampler twin of `pp_step_gemma_logits` (mirrors `pp_step_qwen35_topk`
    /// / `pp_step_laguna_topk`): the LAST stage top-`k`-selects IN RUST and rings
    /// back only the `[2*k]` pack `[k logits][k indices-as-f32]` instead of the
    /// full 262k-vocab logit vector. Kills the gemma served path's dominant
    /// `[vocab]`→wire→Python marshal (the ~39 ms/tok SLOW-ring cost). Greedy
    /// (`top[0]`) == the exact global argmax → byte-identical greedy token.
    /// Resolved BY NAME (`pp_step_gemma_topk`) by serve_head's `DistHead`.
    fn pp_step_gemma_topk(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
        k: usize,
    ) -> PyResult<Option<Vec<(u32, f32)>>> {
        if self.qwen.is_some() {
            return Err(PyRuntimeError::new_err("pp_step_gemma_topk is for Gemma models"));
        }
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_gemma_topk: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        if k == 0 {
            return Err(PyRuntimeError::new_err("pp_step_gemma_topk: k must be >= 1"));
        }
        let msg_len = self.pp_gemma_msg_len();
        let pos = self.inner.kv_caches.iter().map(|c| c.seq_len).max().unwrap_or(0);
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;

        // 1) recv the previous stage's [msg_len] composite, or empty on first.
        let msg_in: Vec<f32> = if do_recv {
            vccl_ffi::recv_f32(py, comm, msg_len, recv_from).map_err(PyRuntimeError::new_err)?
        } else {
            Vec::new()
        };

        // 2) resident stage forward. [msg_len] mid, [vocab] last.
        let out = self.forward_pp_gemma(token_id, msg_in, pos)?;

        // 3) route the result.
        if is_first && is_last {
            return Ok(Some(topk_select(&out, k)));
        }
        if !is_last {
            vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            if is_first {
                let packed = self.pp_recv_vocab(py, 2 * k, last_rank)?;
                Ok(Some(unpack_topk(&packed, k)))
            } else {
                Ok(None)
            }
        } else {
            let top = topk_select(&out, k);
            let packed = pack_topk(&top, k);
            self.pp_send_vocab(py, &packed, 0)?;
            Ok(None)
        }
    }


    /// DISTRIBUTED-SERVE cache-populating prefill for Gemma4 (the companion to
    /// the `pp_step_gemma_logits` decode seam that `scripts/serve_head.py` /
    /// `serve_dist.py` resolves as `forward_pp_gemma_prefill`). Streams the whole
    /// prompt through this PP stage, populating the resident KV (owned layers +
    /// cross-stage target replicas) and advancing every attention `seq_len` to
    /// the prompt length, so the subsequent `pp_step_gemma_logits` decode picks
    /// up at `pos == seq` (the max KV seq_len; the launcher does NOT pass `pos`).
    ///
    /// Like qwen3.6 (and unlike Laguna's separate batched prefill kernel), Gemma
    /// prefill here is a teacher-forced loop over the SAME single-token
    /// `forward_pp_gemma` the decode seam runs, so prefill and decode populate
    /// identical resident state by construction (prefill ≡ decode; the windowed
    /// KV ring, value-less globals and KV-share append are all exercised
    /// per-token exactly as in decode — no separate batched kernel, no fold flag
    /// to honor). Each position's `forward_pp_gemma` appends one K/V per owned
    /// layer (seq_len += 1) and, for cross-stage targets not owned here, one
    /// replica K/V.
    ///
    ///  - FIRST stage (`pp_first`): `tokens` = full prompt `[seq]`; `hidden_in`
    ///    ignored. Embeds each token at its position, returns `[seq*msg_len]` (all
    ///    positions' composite messages) to ship onward. If ALSO last (NR==1):
    ///    returns the LAST position's `[vocab]` logits.
    ///  - MID stage: `hidden_in` = `[seq*msg_len]` from the previous stage;
    ///    `tokens` ignored. Returns `[seq*msg_len]`.
    ///  - LAST stage (`pp_last`): `hidden_in` = `[seq*msg_len]`; returns the LAST
    ///    position's `[vocab]` logits (rank0's first sampled token).
    ///
    /// `msg_len == pp_gemma_msg_len()` (composite `[hidden‖ple‖target_kv]`;
    /// degenerates to `[H]` on g12b — no PLE, no KV-share), mirroring
    /// `forward_pp_qwen35_prefill`'s `[seq*H]` accumulation with the composite
    /// width in place of `H`.
    fn forward_pp_gemma_prefill(
        &mut self,
        tokens: Vec<u32>,
        hidden_in: Vec<f32>,
        seq: usize,
    ) -> PyResult<Vec<f32>> {
        if self.qwen.is_some() {
            return Err(PyRuntimeError::new_err("forward_pp_gemma_prefill is for Gemma models"));
        }
        let msg_len = self.pp_gemma_msg_len();
        // pp_first / pp_last live on VulkanModel (the same fields `forward_pp_gemma`
        // reads), NOT on a per-arch model struct — Gemma IS `self.inner`.
        let (first, last) = (self.pp_first, self.pp_last);
        // Validate BOTH stage inputs against `seq` before the loop below indexes
        // them. The first-stage `tokens[pos]` was previously unchecked: a short
        // `tokens` panics (a pyo3 panic, not a Python exception), and a long one
        // silently prefills only the first `seq` ids while the caller believes
        // the whole prompt went in. See `pp_gemma_prefill_check`.
        pp_gemma_prefill_check(first, tokens.len(), hidden_in.len(), seq, msg_len)
            .map_err(PyRuntimeError::new_err)?;
        let mut out: Vec<f32> = if last { Vec::new() } else { Vec::with_capacity(seq * msg_len) };
        for pos in 0..seq {
            let step = if first {
                self.forward_pp_gemma(tokens[pos], Vec::new(), pos)?
            } else {
                let slice = hidden_in[pos * msg_len..(pos + 1) * msg_len].to_vec();
                self.forward_pp_gemma(0, slice, pos)?
            };
            if last {
                out = step; // keep only the last position's [vocab]
            } else {
                out.extend_from_slice(&step); // accumulate [seq*msg_len]
            }
        }
        Ok(out)
    }


    fn pp_gemma_msg_len(&self) -> usize {
        let cfg = &self.inner.config;
        let h = cfg.hidden_size;
        let total_ple = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input;
        let targets = gemma_pp_kv_targets(cfg);
        let tkv_total: usize = gemma_pp_tkv_lens(cfg, &targets).iter().map(|&l| 2 * l).sum();
        h + total_ple + tkv_total
    }


    /// P3: Batched Gemma4 prefill. Process all T prompt tokens in ONE forward,
    /// using the validated tiled GEMM (gpu_gemm) for projections/FFN and flash
    /// attention (gpu_flash_attn) for causal/windowed attention. Returns the
    /// LAST token's logits (with softcap), identical to running the per-token
    /// decode path (forward_gpu / forward_layer_gpu_matmuls) sequentially.
    ///
    /// Norms / RoPE / PLE are done on CPU per-token for a first correct version
    /// (mirroring forward_layer_gpu_matmuls' math exactly); the wins are the
    /// batched GEMM + flash attention. KV is reset before prefill and all T
    /// tokens are appended per (non-shared) layer; KV-shared layers read the
    /// target cache (already populated by an earlier layer this forward).
    fn forward_prefill_gemma(&mut self, tokens: Vec<u32>) -> PyResult<Vec<f32>> {
        if self.qwen.is_some() {
            return Err(PyRuntimeError::new_err(
                "forward_prefill_gemma needs a Gemma4 model (got Qwen3)"));
        }
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let num_q = cfg.num_attention_heads;
        // NOTE: KV head count is PER-LAYER (`cfg.layer_num_kv_heads`) — shadowed
        // inside the layer loop; g12b global layers are MQA(1), sliding GQA(8).
        let ple_dim = cfg.hidden_size_per_layer_input;
        let t = tokens.len();
        if t == 0 { return Err(PyRuntimeError::new_err("empty prompt")); }
        if t > self.max_seq_len {
            return Err(PyRuntimeError::new_err(format!(
                "prompt length {t} exceeds max_seq_len {} — construct VulkanModel with a larger max_seq_len",
                self.max_seq_len)));
        }
        // Per-layer KV ring (Phase-0 host half): sliding layers now allocate a
        // `sliding_window`-sized RING host cache. This batched prefill appends
        // ALL T tokens up front (so the RESIDENT/decode continuation ring holds
        // the correct last-`window` positions) but sources the ATTENTION K/V
        // for every NON-shared layer from the freshly-computed local `k`/`v`
        // buffers — which hold all T post-norm/post-rope positions in ascending
        // absolute order, byte-identical to what a full (uniform) cache would
        // return via `k_up_to_now()`. Reading the local buffers instead of the
        // (possibly wrapped) ring keeps the fast batched flash / gpu_sdpa
        // dispatch AND is bit-exact vs the pre-ring / KV_RING_DISABLE=1 path
        // (same floats → same shader output), while remaining correct across a
        // ring wrap (t > sliding_window). See the attention block below.
        //
        // The ONE case that cannot be served from local buffers is a KV-SHARED
        // *sliding* layer: it reads a DIFFERENT (earlier) layer's cache, whose
        // ring has already been overwritten down to its last `window` positions
        // by that layer's append-all. Serving it correctly needs the target's
        // full transient K/V retained across the forward (deferred). Gemma-12B
        // / 31B have `num_kv_shared_layers == 0` so this never fires there; only
        // E2B (20 shared layers) hits it. Refuse rather than silently corrupt.
        if !self.flags.kv_ring_disable && cfg.num_kv_shared_layers > 0
            && cfg.sliding_window < t {
            return Err(PyRuntimeError::new_err(format!(
                "forward_prefill_gemma: prompt length {t} exceeds sliding_window {} on a model \
                 with {} KV-shared layers and the windowed host KV ring enabled; the shared-sliding \
                 read targets an already-wrapped ring (deferred). Set VLLM_VULKAN_KV_RING_DISABLE=1 \
                 for full-cache batched prefill, or drive prefill token-by-token via the \
                 ring-correct resident decode path.",
                cfg.sliding_window, cfg.num_kv_shared_layers)));
        }
        self.reset_kv_cache();

        // ── Embed + PLE for all T tokens. gemma_embed_and_ple returns, per
        // token: (hidden[h] (×embed_scale), ple_inputs[num_layers*ple_dim]).
        // Stack into hidden[T,h] and ple_all[T, num_layers*ple_dim].
        let total_ple = cfg.num_hidden_layers * ple_dim;
        let mut hidden = vec![0f32; t * h];
        let mut ple_all = vec![0f32; t * total_ple];
        for (ti, &tok) in tokens.iter().enumerate() {
            let (hv, ple) = self.gemma_embed_and_ple(tok);
            hidden[ti * h..(ti + 1) * h].copy_from_slice(&hv);
            ple_all[ti * total_ple..(ti + 1) * total_ple].copy_from_slice(&ple);
        }

        // Flash is preferred; falls back to per-token gpu_sdpa windowed slice
        // (P2a-style) only if explicitly requested. We report which was used.
        let use_sdpa = self.flags.prefill_sdpa;

        // Layer range: full model runs [0, num_layers) and finishes with the LM
        // head. A layer-limited single-node load (`layer_end < num_layers`, e.g.
        // the on-node ring A/B where the full 12B weights + 2GB tied embed leave
        // no GTT headroom for T=1200 activations) runs [0, pp_end) and returns
        // the last-position HIDDEN state instead of logits — a self-consistency
        // signature that is byte-identical between ring-ON and ring-OFF iff the
        // windowed prefill is correct across the wrap. Prefill must start at
        // layer 0 (embed); a mid-stack PP start has no upstream hidden here.
        if self.pp_start != 0 {
            return Err(PyRuntimeError::new_err(
                "forward_prefill_gemma requires layer_start == 0 (prefill from the embedding)"));
        }
        let layer_end = self.pp_end.min(cfg.num_hidden_layers);

        for layer_idx in 0..layer_end {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
            let is_full = cfg.is_full_attention(layer_idx);
            let head_dim = cfg.layer_head_dim(layer_idx);
            // Per-layer KV head count: g12b global (full-attention) layers are
            // value-less MQA(1) @ head_dim 512; sliding layers GQA(8) @ 256.
            // The function-scope `num_kv` (sliding count) would mis-size every
            // global layer's K/V matvec. Mirrors the decode path's per-layer
            // `cfg.layer_num_kv_heads(layer_idx)`.
            let num_kv = cfg.layer_num_kv_heads(layer_idx);
            // Value-less global attention: no `v_proj` tensor on disk — V is
            // derived from the RAW (pre-k_norm) k_proj output via weightless
            // rms-norm. Mirrors cfg.layer_uses_k_eq_v / forward_layer_gpu_matmuls.
            let uses_k_eq_v = cfg.layer_uses_k_eq_v(layer_idx);
            let q_dim = num_q * head_dim;
            let kv_dim = num_kv * head_dim;
            let is_kv_shared = cfg.is_kv_shared(layer_idx);
            let ffn_inter = cfg.layer_intermediate_size(layer_idx);

            // Pre-extract norm weights (avoids borrow conflicts during GPU calls).
            let inln_w   = self.inner.weights.f32_slice(&ln("input_layernorm.weight")).to_vec();
            let q_norm_w = self.inner.weights.f32_slice(&ln("self_attn.q_norm.weight")).to_vec();
            let k_norm_w = if !is_kv_shared {
                Some(self.inner.weights.f32_slice(&ln("self_attn.k_norm.weight")).to_vec())
            } else { None };
            let pa_w     = self.inner.weights.f32_slice(&ln("post_attention_layernorm.weight")).to_vec();
            let pf_w     = self.inner.weights.f32_slice(&ln("pre_feedforward_layernorm.weight")).to_vec();
            let postff_w = self.inner.weights.f32_slice(&ln("post_feedforward_layernorm.weight")).to_vec();
            // PLE norm weight only exists on has_ple() checkpoints (E2B); g12b /
            // g31b carry no per-layer-embedding tensors (see the PLE block below,
            // gated on cfg.has_ple() exactly like the decode path).
            let ple_norm_w = if cfg.has_ple() {
                self.inner.weights.f32_slice(&ln("post_per_layer_input_norm.weight")).to_vec()
            } else { Vec::new() };
            let layer_scalar = self.inner.weights.f32_slice(&ln("layer_scalar"))[0];

            // ── ATTENTION ──────────────────────────────────────────────────
            let residual = hidden.clone();
            // input_layernorm (row-batched over T).
            let x = model::cpu_rms_norm(&hidden, &inln_w, eps);
            // Batched projections via tiled GEMM (weight read once for all T).
            let mut q = self.gemma_prefill_matmul(&ln("self_attn.q_proj.weight"), &x, t, h, q_dim);
            let mut k = self.gemma_prefill_matmul(&ln("self_attn.k_proj.weight"), &x, t, h, kv_dim);
            // Value-less global layers have no v_proj: V is the RAW (pre-k_norm)
            // k_proj output, cloned here BEFORE the norm loop mutates `k` in
            // place. The weightless v-norm below (in the `!is_kv_shared` block)
            // then applies, and V is NOT roped — bit-identical to the decode
            // path's `v_final = k_final.clone()` before k-norm/rope.
            let mut v = if uses_k_eq_v {
                k.clone()
            } else {
                self.gemma_prefill_matmul(&ln("self_attn.v_proj.weight"), &x, t, h, kv_dim)
            };

            // q-norm (weighted, per head, per token). k/v-norm + RoPE per token.
            let (theta, rotary_dim) = if is_full {
                (1_000_000.0f32, head_dim / 4)
            } else {
                (10_000.0f32, head_dim)
            };
            for ti in 0..t {
                let qrow = &mut q[ti * q_dim..(ti + 1) * q_dim];
                for hi in 0..num_q {
                    let s = &mut qrow[hi * head_dim..(hi + 1) * head_dim];
                    let n = model::cpu_rms_norm(s, &q_norm_w, eps);
                    s.copy_from_slice(&n);
                }
                if let Some(knw) = k_norm_w.as_ref() {
                    let krow = &mut k[ti * kv_dim..(ti + 1) * kv_dim];
                    for hi in 0..num_kv {
                        let s = &mut krow[hi * head_dim..(hi + 1) * head_dim];
                        let n = model::cpu_rms_norm(s, knw, eps);
                        s.copy_from_slice(&n);
                    }
                    let vrow = &mut v[ti * kv_dim..(ti + 1) * kv_dim];
                    for hi in 0..num_kv {
                        let s = &mut vrow[hi * head_dim..(hi + 1) * head_dim];
                        let n = model::cpu_rms_norm_no_weight(s, head_dim, eps);
                        s.copy_from_slice(&n);
                    }
                }
                // RoPE for position ti (partial rotary on global layers).
                let mut qr = q[ti * q_dim..(ti + 1) * q_dim].to_vec();
                let mut kr = k[ti * kv_dim..(ti + 1) * kv_dim].to_vec();
                model::cpu_rope_with_basis(&mut qr, &mut kr, ti, num_q, num_kv, head_dim, rotary_dim, head_dim, theta);
                q[ti * q_dim..(ti + 1) * q_dim].copy_from_slice(&qr);
                k[ti * kv_dim..(ti + 1) * kv_dim].copy_from_slice(&kr);
            }

            // KV cache: append all T tokens to this layer (or read shared target).
            let target_cache_idx = if is_kv_shared {
                self.inner.kv_shared_target(layer_idx)
            } else {
                for ti in 0..t {
                    let kk = &k[ti * kv_dim..(ti + 1) * kv_dim];
                    let vv = &v[ti * kv_dim..(ti + 1) * kv_dim];
                    self.inner.kv_caches[layer_idx].append(kk, vv);
                }
                layer_idx
            };
            let window = if is_full { 0usize } else { cfg.sliding_window };

            // Attention output, laid out [T (pos), num_q, head_dim] (cpu_sdpa /
            // o_proj convention). Flash returns exactly [pos][head][head_dim].
            let attn_out: Vec<f32> = if use_sdpa {
                // P2a-style fallback: per-token windowed gpu_sdpa over the cache
                // slice visible to that position.
                let mut out = vec![0f32; t * q_dim];
                let stride = num_kv * head_dim;
                for ti in 0..t {
                    let qrow = &q[ti * q_dim..(ti + 1) * q_dim];
                    let slen = ti + 1; // positions 0..=ti are visible
                    let kv_start = if window > 0 { slen.saturating_sub(window) } else { 0 };
                    let vlen = slen - kv_start;
                    // Non-shared layers read the local `k`/`v` buffers (all T post-
                    // rope positions, ascending absolute order) — byte-identical to
                    // `cache.k_up_to_now()[kv_start..slen]` when the ring has not
                    // wrapped, and CORRECT across a wrap (the ring only physically
                    // retains the last `window` positions). Shared layers keep the
                    // target-cache read (guarded above against a wrapped ring).
                    let (ck, cv) = if is_kv_shared {
                        let cache = &self.inner.kv_caches[target_cache_idx];
                        (cache.k_up_to_now()[kv_start * stride..slen * stride].to_vec(),
                         cache.v_up_to_now()[kv_start * stride..slen * stride].to_vec())
                    } else {
                        (k[kv_start * stride..slen * stride].to_vec(),
                         v[kv_start * stride..slen * stride].to_vec())
                    };
                    let ao = self.gpu_sdpa(qrow, &ck, &cv, num_q, num_kv, head_dim, vlen, 1.0);
                    out[ti * q_dim..(ti + 1) * q_dim].copy_from_slice(&ao);
                }
                out
            } else {
                // Flash: Q must be [head][pos][head_dim]; q above is [pos][head][hd].
                let kv_len = t; // we appended exactly the T prefill tokens
                let mut qflash = vec![0f32; num_q * t * head_dim];
                for ti in 0..t {
                    for hi in 0..num_q {
                        let src = &q[ti * q_dim + hi * head_dim..ti * q_dim + (hi + 1) * head_dim];
                        let dst = (hi * t + ti) * head_dim;
                        qflash[dst..dst + head_dim].copy_from_slice(src);
                    }
                }
                // K/V [kv_pos][num_kv][head_dim]. Non-shared layers read the local
                // `k`/`v` buffers (all T post-rope positions) — byte-identical to
                // `cache.k_up_to_now()[..kv_len*stride]` when the ring has not
                // wrapped, and CORRECT across a wrap (the shrunk ring only physically
                // holds the last `window` positions, but the local buffers hold all
                // T, and the windowed causal mask below discards the out-of-window
                // rows exactly as a full uniform cache would). Shared layers keep the
                // target-cache read (guarded above against a wrapped ring).
                let (kbuf, vbuf) = if is_kv_shared {
                    let cache = &self.inner.kv_caches[target_cache_idx];
                    (cache.k_up_to_now()[..kv_len * num_kv * head_dim].to_vec(),
                     cache.v_up_to_now()[..kv_len * num_kv * head_dim].to_vec())
                } else {
                    (k[..kv_len * num_kv * head_dim].to_vec(),
                     v[..kv_len * num_kv * head_dim].to_vec())
                };
                // Mask [T × kv_len]: query i attends key j iff j<=i (causal) AND
                // (global: always; sliding: j > i-window). off=0 since kv_len==T.
                let neg = f32::NEG_INFINITY;
                let mut mask = vec![0f32; t * kv_len];
                for i in 0..t {
                    let lo = if window > 0 { i.saturating_sub(window - 1) } else { 0 };
                    for j in 0..kv_len {
                        if j > i || j < lo { mask[i * kv_len + j] = neg; }
                    }
                }
                // gpu_flash_attn returns [pos][head][head_dim] == [T, num_q, hd].
                self.gpu_flash_attn(&qflash, &kbuf, &vbuf, num_q, num_kv, head_dim,
                                    t, kv_len, 1.0, Some(&mask))
            };

            // o_proj (batched GEMM): [T,q_dim] @ Wo[h,q_dim]^T -> [T,h].
            let o_proj = self.gemma_prefill_matmul(&ln("self_attn.o_proj.weight"), &attn_out, t, q_dim, h);
            // Sandwich: post_attention_layernorm on o_proj BEFORE residual add.
            let pa_normed = model::cpu_rms_norm(&o_proj, &pa_w, eps);
            let hidden2: Vec<f32> = residual.iter().zip(pa_normed.iter())
                .map(|(&r, &a)| r + a).collect();
            let residual2 = hidden2.clone();

            // ── FFN ────────────────────────────────────────────────────────
            let ff_in = model::cpu_rms_norm(&hidden2, &pf_w, eps);
            let gate = self.gemma_prefill_matmul(&ln("mlp.gate_proj.weight"), &ff_in, t, h, ffn_inter);
            let up = self.gemma_prefill_matmul(&ln("mlp.up_proj.weight"), &ff_in, t, h, ffn_inter);
            let gate_act = model::cpu_gelu(&gate);
            let mid: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
            let ff_out = self.gemma_prefill_matmul(&ln("mlp.down_proj.weight"), &mid, t, ffn_inter, h);
            // Sandwich: post_feedforward_layernorm on ff_out BEFORE residual add.
            let ff_normed = model::cpu_rms_norm(&ff_out, &postff_w, eps);
            let mut hidden3: Vec<f32> = residual2.iter().zip(ff_normed.iter())
                .map(|(&r, &f)| r + f).collect();

            // ── PLE per token + layer_scalar ───────────────────────────────
            // The per-layer-embedding contribution (per_layer_input_gate /
            // per_layer_projection / post_per_layer_input_norm) only exists on
            // has_ple() checkpoints (E2B). g12b / g31b carry no PLE tensors, so
            // the block is gated exactly like the decode path
            // (`gemma_layer_tail_full` / `gemma_resident_layer`); the layer_scalar
            // multiply still runs unconditionally.
            if cfg.has_ple() {
                let gate_ple = self.gemma_prefill_matmul(&ln("per_layer_input_gate.weight"), &hidden3, t, h, ple_dim);
                let gate_ple_act = model::cpu_gelu(&gate_ple);
                // per-token layer PLE input is ple_all[ti][layer_idx][ple_dim].
                let mut gated = vec![0f32; t * ple_dim];
                for ti in 0..t {
                    let lp = &ple_all[ti * total_ple + layer_idx * ple_dim
                                      ..ti * total_ple + (layer_idx + 1) * ple_dim];
                    for d in 0..ple_dim {
                        gated[ti * ple_dim + d] = gate_ple_act[ti * ple_dim + d] * lp[d];
                    }
                }
                let contrib = self.gemma_prefill_matmul(&ln("per_layer_projection.weight"), &gated, t, ple_dim, h);
                let contrib_normed = model::cpu_rms_norm(&contrib, &ple_norm_w, eps);
                for i in 0..t * h { hidden3[i] += contrib_normed[i]; }
            }
            hidden3.iter_mut().for_each(|val| *val *= layer_scalar);

            hidden = hidden3;
        }

        // Layer-limited single-node load (ring A/B): no LM head — return the
        // last-position hidden [h] as the self-consistency signature.
        let last = &hidden[(t - 1) * h..t * h];
        if layer_end < cfg.num_hidden_layers {
            return Ok(last.to_vec());
        }

        // Full model: final RMS norm + LM head + softcap on the LAST token only.
        // gemma_final's GPU tied-embed lm_head reads the persistent activation
        // buffers (act_ptr_mut(ACT_QKV_IN)); the decode path lazily inits them on
        // its first layer, but this batched prefill never runs that path, so init
        // here (no-op if already ready). If init fails, gemma_final falls back to
        // the host lm_head matmul (logit_p null branch).
        let _ = self.init_act_bufs();
        Ok(self.gemma_final(last))
    }


    /// DEBUG: validate the Gemma4 prefill flash path (head_dim 256/512, GQA, a
    /// per-query causal or causal+windowed mask) against a per-query cpu_sdpa
    /// reference, using synthetic random tensors (NO model load required).
    ///
    /// Args: head_dim (256 sliding or 512 global), window (0 = full causal,
    /// >0 = sliding-window size W so query i attends keys [max(0,i-W+1), i]),
    /// t (#query positions = prompt len), kv_len (#keys), num_q, num_kv, scale,
    /// seed. Builds Q[t,num_q,hd]->[head][pos][hd], K/V[kv,num_kv,hd], the mask
    /// [t × kv_len], runs gpu_flash_attn, and a per-query cpu_sdpa reference.
    /// Returns (cosine, max_abs_diff) over the [t, num_q, hd] output.
    #[allow(clippy::too_many_arguments)]
    fn debug_gemma_flash(&mut self, head_dim: usize, window: usize, t: usize,
                         kv_len: usize, num_q: usize, num_kv: usize, scale: f32,
                         seed: u64) -> PyResult<(f64, f64)> {
        // Deterministic xorshift RNG in [-1, 1).
        let mut s = seed | 1;
        let mut rng = move || {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
        };
        // Q: [head][pos][head_dim]  (gpu_flash_attn / cpu_sdpa convention)
        let q: Vec<f32> = (0..num_q * t * head_dim).map(|_| rng()).collect();
        // K, V: [kv][num_kv][head_dim]
        let k: Vec<f32> = (0..kv_len * num_kv * head_dim).map(|_| rng()).collect();
        let v: Vec<f32> = (0..kv_len * num_kv * head_dim).map(|_| rng()).collect();

        // Mask [t × kv_len]: -inf where key j is NOT visible to query i.
        // Align query i to key position (kv_len - t) + i so the last query sees
        // the last key (decode-style prefill alignment). For the pure square
        // T==kv_len case this reduces to the standard lower-triangular mask.
        let neg = f32::NEG_INFINITY;
        let off = kv_len.saturating_sub(t);
        let mut mask = vec![0f32; t * kv_len];
        for i in 0..t {
            let qpos = off + i;                       // absolute position of query i
            let lo = if window > 0 { qpos.saturating_sub(window - 1) } else { 0 };
            for j in 0..kv_len {
                if j > qpos || j < lo { mask[i * kv_len + j] = neg; }
            }
        }

        let gpu = self.gpu_flash_attn(&q, &k, &v, num_q, num_kv, head_dim,
                                      t, kv_len, scale, Some(&mask));

        // Reference: per (head h, query i) cpu_sdpa over the visible key range.
        let mut cpu = vec![0f32; num_q * t * head_dim];
        for h in 0..num_q {
            let kvh = h / (num_q / num_kv);
            for i in 0..t {
                let qpos = off + i;
                let lo = if window > 0 { qpos.saturating_sub(window - 1) } else { 0 };
                let hi = qpos.min(kv_len - 1);        // last visible key
                let qslice = &q[(h * t + i) * head_dim..(h * t + i + 1) * head_dim];
                let mut kpack = Vec::with_capacity((hi - lo + 1) * head_dim);
                let mut vpack = Vec::with_capacity((hi - lo + 1) * head_dim);
                for kv in lo..=hi {
                    let base = (kv * num_kv + kvh) * head_dim;
                    kpack.extend_from_slice(&k[base..base + head_dim]);
                    vpack.extend_from_slice(&v[base..base + head_dim]);
                }
                let o = model::cpu_sdpa(qslice, &kpack, &vpack, 1, 1, head_dim,
                                        hi - lo + 1, scale, None);
                let dst = (i * num_q + h) * head_dim;  // [pos i][head h]
                cpu[dst..dst + head_dim].copy_from_slice(&o);
            }
        }

        let mut dot = 0f64; let mut ng = 0f64; let mut nc = 0f64; let mut maxd = 0f64;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            dot += (*a as f64) * (*b as f64); ng += (*a as f64).powi(2); nc += (*b as f64).powi(2);
            maxd = maxd.max((*a as f64 - *b as f64).abs());
        }
        Ok((dot / (ng.sqrt() * nc.sqrt() + 1e-12), maxd))
    }


    /// DEBUG (gemma-31B): confirms the global-layer KV-dim OOB fix. Returns
    /// `(gpu_k_proj_out_features, num_key_value_heads*layer_head_dim,
    ///   layer_num_kv_heads*layer_head_dim)`.
    ///
    /// On a GLOBAL layer (idx%6==5, e.g. 5) it reads `(2048, 8192, 2048)`:
    ///   • FIRST  = actual loaded GPU k_proj rows (4 kv * 512 = 2048),
    ///   • MIDDLE = the stale-constant kv_dim the pre-fix dispatch used
    ///              (`num_key_value_heads`=16 * 512 = 8192),
    ///   • THIRD  = the fixed per-layer kv_dim (`layer_num_kv_heads`=4 * 512).
    /// MIDDLE > FIRST is the smoking gun (the 4× OOB read past the 2048-row
    /// buffer → robustBufferAccess(0) phantom heads); the fix makes the
    /// dispatch use THIRD == FIRST. On a sliding layer all three are 4096.
    fn debug_gemma_kv_dims(&self, layer_idx: usize) -> PyResult<(usize, usize, usize)> {
        let cfg = &self.inner.config;
        let hd = cfg.layer_head_dim(layer_idx);
        let stale = cfg.num_key_value_heads * hd;
        let correct = cfg.layer_num_kv_heads(layer_idx) * hd;
        let name = format!("model.layers.{layer_idx}.self_attn.k_proj.weight");
        let gpu_out = if let Some(w) = self.gpu_weights.get(&name) {
            gpu_weight_out_features(w, cfg.hidden_size)
        } else if let Some(t) = self.inner.weights.tensors.get(&name) {
            // Host-f32 fallback (NVFP4/FP8 GPU path declined): row count = len/hidden.
            t.data.len() / cfg.hidden_size
        } else {
            0
        };
        Ok((gpu_out, stale, correct))
    }


}


impl VulkanModel {

    /// Gemma TP LM-head with optional vocab-sharding
    /// (`VLLM_VULKAN_TP_SHARD_LMHEAD`, default ON). Given the final-normed hidden
    /// `[h]`, returns the FULL softcapped `[vocab]` logits, identical on every
    /// rank — the shared tail for `forward_tp_gemma` and (per-position)
    /// `forward_tp_gemma_verify_impl`. Not a pymethod (`&[f32]` arg isn't
    /// PyO3-representable) — lives in the plain impl like `gemma_attn_tp`.
    ///
    /// The tied embed/lm_head table (`model.embed_tokens.weight`) is REPLICATED
    /// on every TP rank (host-side f16 in `q35_f16_host`, needed whole for the
    /// INPUT embedding lookup), so it is not in the projection shard set. The
    /// pre-sharding tail widened that WHOLE table to f32
    /// (`gemma_lm_head_host_f32`, ~262144×5376×4 ≈ 5.6GB) on every rank — the
    /// transient that OOM-killed (rc=-9) the ~14GB TP-4 node atop the resident
    /// weights.
    ///
    /// Replicated path (tp_size==1, flag off, or native all-gather unavailable):
    /// full-table widen + `cpu_matmul` — byte-identical to the pre-sharding tail.
    ///
    /// Sharded path (§2.2, mirrors the qwen `forward_tp_qwen35` tail): each rank
    /// widens ONLY its contiguous vocab-row slice `[lo,lo+per)` of the tied f16
    /// table to f32 (peak ≈ per·h·4 B = 65536·5376·4 ≈ 1.4GB at TP-4, 4× under
    /// the 5.6GB full widen), `cpu_matmul`s that slice against `normed` → `[per]`
    /// logits, softcaps its own slice (tanh/`cap` is elementwise, so pre- vs
    /// post-gather softcap are identical), then all-gathers the per-rank slices
    /// (padded to the uniform `max_per`) and reassembles them in vocab order.
    /// Because each output logit is the SAME per-row f32 dot product of `normed`
    /// against the SAME f32-widened weight row, the sharded logits are
    /// bit-identical to the replicated tail (stronger than argmax-equivalent).
    fn gemma_tp_lmhead_logits(
        &mut self,
        py: Python<'_>,
        normed: &[f32],
        h: usize,
        vocab: usize,
        n: usize,
        cap: f32,
    ) -> PyResult<Vec<f32>> {
        let softcap = |logits: &mut [f32]| {
            logits.iter_mut().for_each(|l| *l = (*l / cap).tanh() * cap);
        };
        let want_shard = n > 1
            && self.flags.tp_shard_lmhead
            && self.native_comm_enabled()
            && vccl_ffi::allgather_available();
        // Fail-fast on the multi-rank NO-native-comm footgun: the vocab-sharded
        // lm_head needs the native all-gather to reassemble, and there is no
        // callback all-gather — so `VLLM_VULKAN_NATIVE_COMM=0` silently reverts to
        // the `!want_shard` whole-vocab f32 widen below (~5.6GB/rank on gemma-31B,
        // ~4GB on 12B). Every rank widens simultaneously → rank0 OOM-kill (rc=-9)
        // then survivors hang at the next reduce. Turn that confusing OOM/hang into
        // an actionable error instead of a 5.6GB alloc on a 14GB node.
        if n > 1 && !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "gemma TP lm_head (n>1) requires native comm: VLLM_VULKAN_NATIVE_COMM=0 \
                 has no all-gather callback, so it falls back to a whole-vocab f32 widen \
                 (~4GB on 12B / ~5.6GB on 31B per rank) that OOM-kills the ~14GB nodes. \
                 Set VLLM_VULKAN_NATIVE_COMM=1.",
            ));
        }
        if !want_shard {
            // Replicated / TP-1 / no-allgather fallback: full-table f32 widen.
            let lm_w = self.gemma_lm_head_host_f32().into_owned();
            let mut logits = model::cpu_matmul(normed, &lm_w, 1, h, vocab);
            softcap(&mut logits);
            return Ok(logits);
        }
        // Sharded: this rank produces ONLY its contiguous [lo,lo+per) vocab rows.
        let (lo, per) = tp::tp_vocab_shard_range(vocab, self.tp_rank, n);
        // GPU LM-head shard (VLLM_VULKAN_GEMMA_GPU_LMHEAD, default on): the loader
        // uploaded this rank's [lo,lo+per) embed slice into
        // gpu_weights["model.embed_tokens.weight"] as F16 (tp_size>1 branch), but
        // ONLY when the global quant is F16 — so its presence is a sufficient
        // signal that `gemma_matvec` (matvec_variant(true, per)) will dispatch the
        // matching f16 shader against it. This moves the ~1000ms/shard CPU matmul
        // on-device (~10ms). Absent (flag off / non-f16 quant / upload failed) →
        // the CPU f16-shard widen. NOTE: `gemma_matvec`'s OWN CPU fallback would
        // read the WHOLE f32 table (rows [0,per), not [lo,per)) — wrong for rank>0
        // — so route to it only when the GPU shard buffer is actually present.
        let gpu_shard = self.engine.is_some()
            && self.gpu_weights.contains_key("model.embed_tokens.weight");
        // `embed_packed` (mlx4-packed lm_head) only exists on the qwen3_5 model.
        #[cfg(feature = "qwen35")]
        let packed_lm = self.qwen35.as_ref().and_then(|q| q.embed_packed.as_ref());
        #[cfg(not(feature = "qwen35"))]
        let packed_lm: Option<&crate::model::PackedEmbed> = None;
        let mut local = if gpu_shard {
            self.gemma_matvec("model.embed_tokens.weight", normed, h, per, true)
        } else if let Some(pe) = packed_lm {
            // Packed embed = tied lm_head: decode this rank's [lo,per) rows on the
            // fly (no whole f16 table), then CPU matmul the shard.
            let mut w: Vec<f32> = Vec::with_capacity(per * h);
            for row in lo..lo + per { w.extend_from_slice(&pe.row_f32(row)); }
            model::cpu_matmul(normed, &w, 1, h, per)
        } else if let Some(f16v) = self.q35_f16_host.get("model.embed_tokens.weight") {
            let sl = &f16v[lo * h..(lo + per) * h];
            let w: Vec<f32> = sl.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            model::cpu_matmul(normed, &w, 1, h, per)
        } else {
            // f32 store (test constructions that never populate the f16 table).
            let w = self.inner.weights.f32_slice("model.embed_tokens.weight");
            let sl = w[lo * h..(lo + per) * h].to_vec();
            model::cpu_matmul(normed, &sl, 1, h, per)
        };
        softcap(&mut local);
        // All-gather padded to the uniform max_per (vcclAllGather needs an equal
        // sendcount per rank); reassemble in vocab order. rem==0 (262144/4) needs
        // no padding, but pad-and-slice guards the not-divisible-by-n case too.
        let (base, rem) = (vocab / n, vocab % n);
        let max_per = base + usize::from(rem > 0);
        let mut padded = local;
        padded.resize(max_per, 0.0);
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let recv = vccl_ffi::all_gather_f32(py, comm, &padded, n)
            .map_err(PyRuntimeError::new_err)?;
        let mut logits = vec![0.0f32; vocab];
        for r in 0..n {
            let (rlo, rper) = tp::tp_vocab_shard_range(vocab, r, n);
            logits[rlo..rlo + rper].copy_from_slice(&recv[r * max_per..r * max_per + rper]);
        }
        Ok(logits)
    }


}

/// Argument validation for `forward_pp_gemma_prefill`, split out so it is
/// unit-testable without a loaded Gemma model or a GPU.
///
/// The stage kind decides WHICH input carries the prompt, and each is indexed
/// `seq` times by the caller's loop:
///   - FIRST stage: `tokens[pos]` for `pos in 0..seq`  → needs `tokens_len == seq`
///   - MID/LAST   : `hidden_in[pos*msg_len..(pos+1)*msg_len]` → `hidden_len == seq*msg_len`
///
/// EXACT equality both ways, not `>=`. A SHORT input panics on the index; a LONG
/// one is worse than a panic — it prefills a silent prefix of the prompt and the
/// KV/`seq_len` state then disagrees with what the caller thinks it sent.
pub(crate) fn pp_gemma_prefill_check(
    first: bool,
    tokens_len: usize,
    hidden_len: usize,
    seq: usize,
    msg_len: usize,
) -> Result<(), String> {
    if seq == 0 {
        return Err("forward_pp_gemma_prefill: empty prompt".to_string());
    }
    if first {
        if tokens_len != seq {
            return Err(format!(
                "forward_pp_gemma_prefill: tokens.len()={tokens_len} != seq={seq}"));
        }
    } else if hidden_len != seq * msg_len {
        return Err(format!(
            "forward_pp_gemma_prefill: hidden_in.len()={} != seq*msg_len={}",
            hidden_len, seq * msg_len));
    }
    Ok(())
}

#[cfg(test)]
mod pp_prefill_arg_tests {
    use super::pp_gemma_prefill_check;

    #[test]
    fn empty_prompt_is_refused_first_and_mid() {
        assert!(pp_gemma_prefill_check(true, 0, 0, 0, 16).is_err());
        assert!(pp_gemma_prefill_check(false, 0, 0, 0, 16).is_err());
    }

    /// Exact-length inputs still pass, per stage kind. A first stage ignores
    /// `hidden_in` entirely (the launcher passes an empty vec), and a mid stage
    /// ignores `tokens` — neither may become a new requirement.
    #[test]
    fn exact_lengths_pass_and_the_unused_input_is_ignored() {
        assert!(pp_gemma_prefill_check(true, 4, 0, 4, 16).is_ok());   // first: hidden_in empty
        assert!(pp_gemma_prefill_check(false, 0, 64, 4, 16).is_ok()); // mid: tokens empty
    }

    /// THE REGRESSION GUARD (short side). `tokens.len() < seq` used to reach
    /// `tokens[pos]` and panic out of pyo3.
    #[test]
    fn first_stage_short_tokens_is_refused() {
        let e = pp_gemma_prefill_check(true, 3, 0, 4, 16).unwrap_err();
        assert!(e.contains("tokens.len()=3 != seq=4"), "{e}");
    }

    /// THE REGRESSION GUARD (long side). `tokens.len() > seq` never panicked —
    /// it silently prefilled the first `seq` ids and dropped the rest, leaving
    /// the resident KV describing a DIFFERENT prompt than the caller sent. A
    /// guard that only covers the short case leaves this alive.
    #[test]
    fn first_stage_long_tokens_is_refused() {
        let e = pp_gemma_prefill_check(true, 9, 0, 4, 16).unwrap_err();
        assert!(e.contains("tokens.len()=9 != seq=4"), "{e}");
    }

    /// The pre-existing mid/last-stage check keeps its exact message.
    #[test]
    fn mid_stage_hidden_mismatch_keeps_its_message() {
        let e = pp_gemma_prefill_check(false, 0, 63, 4, 16).unwrap_err();
        assert_eq!(e, "forward_pp_gemma_prefill: hidden_in.len()=63 != seq*msg_len=64");
    }

    /// Property: whenever a FIRST stage is accepted, every index the loop takes
    /// (`tokens[0..seq]`) is in bounds AND no token is left unconsumed. This is
    /// the invariant the loop relies on; it fails in both directions without the
    /// exact-equality check.
    #[test]
    fn ok_implies_first_stage_consumes_exactly_the_prompt() {
        for seq in 1..8usize {
            for tokens_len in 0..12usize {
                if pp_gemma_prefill_check(true, tokens_len, 0, seq, 16).is_ok() {
                    assert!(seq <= tokens_len, "seq {seq} > tokens_len {tokens_len}: would panic");
                    assert_eq!(tokens_len, seq, "tokens_len {tokens_len} != seq {seq}: silent drop");
                }
            }
        }
    }
}
