// SPDX-License-Identifier: Apache-2.0
//! Per-model pyo3 seam for `qwen35` — moved verbatim out of the monolithic
//! `VulkanModel` `#[pymethods]` block in `lib.rs` (Phase A upstream refactor).
//! The Phase A move itself was behavior-preserving (method bodies byte-for-byte
//! identical), but this file is NOT move-only any more: later work ADDED methods
//! here — `forward_tp_qwen35_verify_impl`, `qwen35_tp_verify_rollback_impl`,
//! `debug_tp_qwen35_verify_vs_serial`, and the MTP pre-norm capture in
//! `qwen35_tp_forward_normed` among them. Review those as new code.
//! Kept as separate `#[pymethods] impl VulkanModel` block(s) via pyo3's
//! `multiple-pymethods` feature so a per-model upstream PR can carve this file.
#![allow(clippy::all)]

use crate::*;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;


#[pymethods]
impl VulkanModel {

    /// Fused PP step: native vCCL recv (if not first) → resident qwen3.6 forward
    /// → native vCCL send (if not last) OR Rust argmax (last stage). The H-float
    /// hidden and the vocab-float logits NEVER cross the pyo3 boundary — only the
    /// token id in and (argmax token, logit) out on the last stage. Kills the two
    /// dominant per-token glue costs the attribution run measured: the last
    /// stage's Vec<f32>→PyList vocab marshal + pure-python argmax (~43 ms/tok at
    /// vocab=248320) and the per-hop PyList↔ctypes hidden conversions.
    ///
    /// Design: blocking send/recv INSIDE the step, NOT overlapped — single-stream
    /// autoregressive decode has a strict data dependency (token N+1 needs token
    /// N's argmax; stage s+1 needs stage s's hidden), so there is no cross-token
    /// or cross-stage pipelining to exploit on one stream; overlap would add
    /// complexity for zero win. Requires `set_collective_comm` +
    /// `VLLM_VULKAN_NATIVE_COMM!=0`. `recv_from < 0` ⇒ first stage (embeds
    /// `token_id`); `send_to < 0` ⇒ last stage (returns `Some((tok, logit))`).
    fn pp_step_qwen35(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        pos: usize,
        recv_from: i32,
        send_to: i32,
    ) -> PyResult<Option<(u32, f32)>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_qwen35: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let h = self.qwen35.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("pp_step_qwen35 needs a qwen3_5 model"))?
            .config.hidden_size;
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);

        // 1) receive the previous stage's hidden (GIL dropped inside recv_f32),
        //    or empty on the first stage (it embeds token_id).
        let hidden_in: Vec<f32> = if do_recv {
            vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
        } else {
            Vec::new()
        };

        // 2) run the resident stage forward (holds the GIL — pure compute, same
        //    as forward_pp_qwen35 today). H floats on mid stages, vocab on last.
        let out = self.forward_pp_qwen35_impl(token_id, hidden_in, pos)?;

        // 3) send onward (native, no PyList) OR argmax in Rust on the last stage.
        if !is_last {
            vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            Ok(None)
        } else {
            // Same first-max tie-break as the driver's python argmax (strict >).
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in out.iter().enumerate() {
                if v > bv { bv = v; bi = i; }
            }
            Ok(Some((bi as u32, bv)))
        }
    }


    /// DISTRIBUTED-SERVE twin of `pp_step_qwen35` (mirrors `pp_step_laguna_logits`):
    /// instead of argmaxing on the last stage, the last stage rings the FULL
    /// `[vocab]` logits back to rank0 (raw f32 over vCCL — NO `Vec<f32>→PyList`
    /// marshal) so vLLM's Sampler on rank0 sees the whole distribution. This is
    /// the `pp_step_qwen35_logits` seam the general `scripts/serve_dist.py`
    /// launcher resolves by name (`pp_step_<model_type>_logits`) for
    /// `--model-type qwen35`.
    ///
    /// The launcher calls it pos-free — `(token_id, recv_from, send_to, last_rank)`
    /// — so the decode `pos` is derived from the resident KV `seq_len`
    /// (`Qwen35Model::current_decode_pos`), which prefill fills to the prompt
    /// length and each decode step advances by one (exactly the value the Python
    /// PP driver used to pass explicitly). Bit-exact with `pp_step_qwen35`'s
    /// last-stage logits (same resident forward; only the transport of the result
    /// differs — full vocab rung back vs argmax returned).
    ///
    ///  - FIRST  (`recv_from<0`, `send_to>=0`): embed `token_id` → `[H]` → send
    ///    onward; then recv the `[vocab]` ring-back from `last_rank`; return it.
    ///  - MID    (`recv_from>=0`, `send_to>=0`): recv `[H]` → stage forward →
    ///    send `[H]` onward; return `None`.
    ///  - LAST   (`recv_from>=0`, `send_to<0`): recv `[H]` → decode → `[vocab]`;
    ///    send `[vocab]` to rank0; return `None`.
    ///  - STANDALONE N=1 (`recv_from<0` && `send_to<0`): embed → `[vocab]`;
    ///    return `Some([vocab])` with no wire.
    ///
    /// Uses the same plain `send_f32`/`recv_f32` transport as `pp_step_qwen35`
    /// (qwen3.6 has no pre-pinned PP scratch); a `comm_register`'d vocab ring
    /// (as in `pp_step_laguna_logits`) is a later perf lever. Requires
    /// `set_collective_comm` + `VLLM_VULKAN_NATIVE_COMM!=0`.
    fn pp_step_qwen35_logits(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
    ) -> PyResult<Option<Vec<f32>>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_qwen35_logits: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let (h, vocab, pos) = {
            let m = self.qwen35.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("pp_step_qwen35_logits needs a qwen3_5 model"))?;
            (m.config.hidden_size, m.config.vocab_size, m.current_decode_pos())
        };
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;

        // 1) recv the previous stage's [H] hidden (GIL dropped inside recv_f32),
        //    or empty on the first stage (it embeds token_id).
        let hidden_in: Vec<f32> = if do_recv {
            vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
        } else {
            Vec::new()
        };

        // 2) resident stage forward. [H] on mid stages, [vocab] on the last.
        let out = self.forward_pp_qwen35_impl(token_id, hidden_in, pos)?;

        // 3) route the result.
        if is_first && is_last {
            // STANDALONE N=1: rank0 is both first and last; `out` is already [vocab].
            return Ok(Some(out));
        }
        if !is_last {
            // FIRST / MID: forward `[H]` onward, then (rank0 only) recv the ring-back.
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


    /// FAST-sampler twin of `pp_step_qwen35_logits` (mirrors `pp_step_laguna_topk`):
    /// instead of ringing the FULL `[vocab]` logits back to rank0 for a Python
    /// full-vocab sample, the LAST stage does the top-`k` selection IN RUST
    /// (`topk_select`, strict-`>` earliest-index-wins) and rings back only the
    /// `[2*k]` pack `[k logits][k indices-as-f32]`. rank0's Python `_sample_topk`
    /// then samples over those K candidates. This kills the qwen35 served path's
    /// dominant `[vocab]`→wire→Python cost (152k/248k floats every token) — the
    /// same lever laguna already has and qwen35/gemma lacked (leaving them on the
    /// ~36 ms/tok SLOW ring). Greedy (`top[0]`) == the exact global argmax, so the
    /// greedy served token is BYTE-IDENTICAL to the full-vocab path.
    ///
    /// Reuses the shared `pp_vocab_ring` (repinned to `2*k`) for the ring-back —
    /// only one of `_logits`/`_topk` drives a given decode loop, so there is no
    /// size contention. Resolved BY NAME (`pp_step_qwen35_topk`) by serve_head's
    /// `DistHead` (`has_topk`). Requires `set_collective_comm` +
    /// `VLLM_VULKAN_NATIVE_COMM!=0`.
    ///
    ///  - FIRST (`recv_from<0`, `send_to>=0`): embed → `[H]` → send onward; recv
    ///    the `[2*k]` ring-back from `last_rank`; return `Some(Vec<(tok,logit)>)`.
    ///  - MID (`recv_from>=0`, `send_to>=0`): recv `[H]` → forward → send `[H]`;
    ///    return `None`.
    ///  - LAST (`recv_from>=0`, `send_to<0`): recv `[H]` → decode → `[vocab]` →
    ///    top-k → send `[2*k]` to rank0; return `None`.
    ///  - STANDALONE N=1 (`recv_from<0` && `send_to<0`): embed → `[vocab]` →
    ///    top-k; return `Some(Vec<(tok,logit)>)` with no wire.
    fn pp_step_qwen35_topk(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
        k: usize,
    ) -> PyResult<Option<Vec<(u32, f32)>>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_qwen35_topk: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        if k == 0 {
            return Err(PyRuntimeError::new_err("pp_step_qwen35_topk: k must be >= 1"));
        }
        let (h, pos) = {
            let m = self.qwen35.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("pp_step_qwen35_topk needs a qwen3_5 model"))?;
            (m.config.hidden_size, m.current_decode_pos())
        };
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;

        // 1) recv the previous stage's [H] hidden, or empty on the first stage.
        let hidden_in: Vec<f32> = if do_recv {
            vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
        } else {
            Vec::new()
        };

        // 2) resident stage forward. [H] on mid stages, [vocab] on the last.
        let out = self.forward_pp_qwen35_impl(token_id, hidden_in, pos)?;

        // 3) route the result.
        if is_first && is_last {
            // STANDALONE N=1: rank0 is both first and last; top-k `out` locally.
            return Ok(Some(topk_select(&out, k)));
        }
        if !is_last {
            // FIRST / MID: forward `[H]` onward, then (rank0 only) recv the ring-back.
            vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            if is_first {
                let packed = self.pp_recv_vocab(py, 2 * k, last_rank)?;
                Ok(Some(unpack_topk(&packed, k)))
            } else {
                Ok(None)
            }
        } else {
            // LAST stage: top-k over [vocab], pack [k logits][k indices], ring [2*k].
            let top = topk_select(&out, k);
            let packed = pack_topk(&top, k);
            self.pp_send_vocab(py, &packed, 0)?;
            Ok(None)
        }
    }


    /// Megatron tensor-parallel forward for one Qwen3.6 (qwen3_5 hybrid) token
    /// (decode). Each rank runs the WHOLE layer stack on its 1/N head shard of
    /// every projection (col-shard q/k/v/gate/up + GatedDeltaNet in_proj_*;
    /// row-shard o/down/out_proj). The two per-layer partial sums — after the
    /// attention sub-block's o_proj/out_proj, and after the MLP's down_proj — are
    /// completed by a vCCL all-reduce(SUM) via the Python `all_reduce` callback.
    /// embed / model.norm / lm_head are replicated, so every rank ends with the
    /// identical full logit vector.
    ///
    /// The novel part vs the dense `forward_tp` is the HYBRID layer: full_attention
    /// layers shard the 24 q / 4 KV heads (GQA ratio 6 preserved per rank);
    /// linear_attention (GatedDeltaNet) layers shard the 48 value / 16 key heads —
    /// the conv1d (per-channel) and the delta-rule recurrence (per v-head [K,V]
    /// state) are head/channel-independent, so each rank owns 12 v-heads / 4
    /// k-heads of conv + recurrence state. Mirrors forward_qwen35_gpu's per-op
    /// math; only WHICH heads each rank computes + the 2 all-reduces differ.
    ///
    /// Correctness gate: TP logits == single-node forward_qwen35_gpu (argmax-exact,
    /// cos=1.0 modulo f32 reduction order across the all-reduce).
    fn forward_tp_qwen35(&mut self, py: Python<'_>, token_id: u32, pos: usize,
                         all_reduce: PyObject) -> PyResult<Vec<f32>> {
        let (cfg, normed) = self.qwen35_tp_forward_normed(py, token_id, pos, all_reduce)?;
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
        let (local, lo) = self.tp_lmhead_local(&lm_name, &normed, h, vocab);
        match self.lm_shard {
            // Unsharded (replicated) tail: every rank already has the full
            // vector — identical to pre-sharding behavior, no gather needed.
            None => Ok(local),
            // Sharded tail (mode b, §2.3): all-gather this rank's slice,
            // padded to max_per (uniform sendcount), and reassemble in vocab
            // order. rem=0 (e.g. 27B/4) needs no padding at all.
            Some(LmShard { v, .. }) => {
                let n = self.tp_size.max(1);
                let (base, rem) = (v / n, v % n);
                let max_per = base + usize::from(rem > 0);
                let mut padded = local;
                padded.resize(max_per, 0.0);
                let comm = self.collective_comm as *mut std::os::raw::c_void;
                let recv = vccl_ffi::all_gather_f32(py, comm, &padded, n)
                    .map_err(PyRuntimeError::new_err)?;
                let mut logits = vec![0.0f32; v];
                for r in 0..n {
                    let (rlo, rper) = tp::tp_vocab_shard_range(v, r, n);
                    logits[rlo..rlo + rper].copy_from_slice(&recv[r * max_per..r * max_per + rper]);
                }
                let _ = lo;
                Ok(logits)
            }
        }
    }


    /// TP greedy decode: like `forward_argmax` but for the vocab-sharded lm_head
    /// (§2.3a) — each rank computes ITS local argmax, all-gathers the tiny
    /// `[max_val, local_idx]` pair (2 f32 lanes/rank, idx bit-cast), and merges
    /// in Rust (`tp::tp_argmax_merge`) with the SAME strict-`>`-from-index-0
    /// tie-break as `forward_argmax`/`pp_step_qwen35` — so TP argmax is
    /// guaranteed to equal single-node argmax on bit-identical logits (see
    /// `tp_lmhead_local`'s R1 caveat: matvec kernel geometry is picked by this
    /// rank's local row count, not the full vocab, so near-tie ulp drift vs a
    /// single-node run is possible and is the on-node empirical gate).
    /// Falls back to a plain full-vector argmax (no gather) when unsharded.
    fn forward_tp_argmax_qwen35(&mut self, py: Python<'_>, token_id: u32, pos: usize,
                                 all_reduce: PyObject) -> PyResult<(u32, f32)> {
        let (cfg, normed) = self.qwen35_tp_forward_normed(py, token_id, pos, all_reduce)?;
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
        let (local, lo) = self.tp_lmhead_local(&lm_name, &normed, h, vocab);
        match self.lm_shard {
            None => {
                let (val, idx) = tp::strict_argmax(&local);
                Ok((idx, val))
            }
            Some(_) => {
                let n = self.tp_size.max(1);
                let (val, idx) = tp::strict_argmax(&local);
                // Pack [val, idx_as_f32_bits] — the allgather is a byte mover;
                // the F32 datatype tag is only a width tag (idx is bit-cast).
                let send = [val, f32::from_bits(idx)];
                let comm = self.collective_comm as *mut std::os::raw::c_void;
                let recv = vccl_ffi::all_gather_f32(py, comm, &send, n)
                    .map_err(PyRuntimeError::new_err)?;
                let locals: Vec<(f32, u32)> = (0..n)
                    .map(|r| (recv[r * 2], recv[r * 2 + 1].to_bits()))
                    .collect();
                let _ = lo;
                Ok(tp::tp_argmax_merge(&locals, vocab, n))
            }
        }
    }


    /// TP sampling: like `forward_topk` but for the vocab-sharded lm_head
    /// (§2.3c) — each rank computes its local top-`k` (identical algorithm to
    /// `forward_topk`, global-indexed), all-gathers `n*2k` f32 lanes, and
    /// merges via `tp::tp_topk_merge` (sort by value desc / global_idx asc,
    /// take k — reproduces `forward_topk`'s earliest-index-wins tie rule).
    /// Falls back to a plain full-vector top-k (no gather) when unsharded.
    fn forward_tp_topk_qwen35(&mut self, py: Python<'_>, token_id: u32, pos: usize,
                               k: usize, all_reduce: PyObject) -> PyResult<Vec<(u32, f32)>> {
        let (cfg, normed) = self.qwen35_tp_forward_normed(py, token_id, pos, all_reduce)?;
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
        let (local, lo) = self.tp_lmhead_local(&lm_name, &normed, h, vocab);
        let k = k.min(vocab);
        match self.lm_shard {
            None => Ok(tp::strict_topk_local(&local, 0, k)),
            Some(_) => {
                let n = self.tp_size.max(1);
                let local_topk = tp::strict_topk_local(&local, lo, k);
                // Pad to exactly k pairs with (-inf, idx=u32::MAX) so every
                // rank sends the SAME uniform sendcount (vcclAllGather
                // requirement); padding entries always lose the value-desc
                // sort and are truncated away by tp_topk_merge.
                let mut send = Vec::with_capacity(2 * k);
                for i in 0..k {
                    if let Some(&(idx, val)) = local_topk.get(i) {
                        send.push(val);
                        send.push(f32::from_bits(idx));
                    } else {
                        send.push(f32::NEG_INFINITY);
                        send.push(f32::from_bits(u32::MAX));
                    }
                }
                let comm = self.collective_comm as *mut std::os::raw::c_void;
                let recv = vccl_ffi::all_gather_f32(py, comm, &send, n)
                    .map_err(PyRuntimeError::new_err)?;
                let mut candidates = Vec::with_capacity(n * k);
                for i in 0..(n * k) {
                    let val = recv[i * 2];
                    let idx = recv[i * 2 + 1].to_bits();
                    if idx != u32::MAX { candidates.push((idx, val)); }
                }
                Ok(tp::tp_topk_merge(candidates, k))
            }
        }
    }


    /// Length of the inter-stage message for `forward_pp_gemma`
    /// (`hidden + ple_inputs + target_kv`). Ranks size their vCCL recv buffers
    /// with this; it is identical on every stage.
    /// Pipeline-parallel forward for one qwen3.6 (qwen3_5) token (cross-stage).
    /// First stage: pass the token id (`hidden_in` ignored — pass `[]`).
    /// Mid/last stages: pass the previous stage's hidden state in `hidden_in`.
    /// Returns the hidden state to send to the next stage, or the full logit
    /// vector on the last stage. The dual per-layer state (DeltaNet/KV) is
    /// resident per stage and advanced in place (NOT carried in the message).
    fn forward_pp_qwen35(&mut self, token_id: u32, hidden_in: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        self.forward_pp_qwen35_impl(token_id, hidden_in, pos)
    }


    /// DISTRIBUTED-SERVE cache-populating prefill for qwen3.6 (the companion to
    /// the `pp_step_qwen35_logits` decode seam that `scripts/serve_head.py`
    /// resolves as `forward_pp_qwen35_prefill`). Streams the whole prompt through
    /// this PP stage, populating the resident KV + advancing the attention
    /// `seq_len` to the prompt length, so the subsequent `pp_step_qwen35_logits`
    /// decode picks up at `current_decode_pos() == seq` (the value the generic
    /// launcher does NOT pass explicitly).
    ///
    /// Unlike Laguna — whose prefill uses a SEPARATE batched `[seq]` kernel and so
    /// must branch on the `laguna_1cb` fold to share the decode's cache backing —
    /// qwen3.6 has no batched prefill kernel: prefill here is a teacher-forced loop
    /// over the SAME single-token `forward_pp_qwen35_impl` the decode seam runs, so
    /// prefill and decode populate identical resident state by construction (no
    /// fold flag to honor). Each position's `forward_pp_qwen35_impl` appends one
    /// K/V (seq_len += 1) and advances the DeltaNet recurrence in place.
    ///
    ///  - FIRST stage (`pp_first`): `tokens` = full prompt `[seq]`; `hidden_in`
    ///    ignored. Embeds each token at its position, returns `[seq*H]` (all
    ///    positions' hidden) to ship onward. If ALSO last (NR==1): returns the LAST
    ///    position's `[vocab]` logits.
    ///  - MID stage: `hidden_in` = `[seq*H]` from the previous stage; `tokens`
    ///    ignored. Returns `[seq*H]`.
    ///  - LAST stage (`pp_last`): `hidden_in` = `[seq*H]`; returns the LAST
    ///    position's `[vocab]` logits (rank0's first sampled token).
    fn forward_pp_qwen35_prefill(
        &mut self,
        tokens: Vec<u32>,
        hidden_in: Vec<f32>,
        seq: usize,
    ) -> PyResult<Vec<f32>> {
        let h = {
            let m = self.qwen35.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("forward_pp_qwen35_prefill needs a qwen3_5 model")
            })?;
            m.config.hidden_size
        };
        // pp_first / pp_last live on VulkanModel (the same fields
        // `forward_pp_qwen35_impl` reads), NOT on Qwen35Model.
        let (first, last) = (self.pp_first, self.pp_last);
        if seq == 0 {
            return Err(PyRuntimeError::new_err("forward_pp_qwen35_prefill: empty prompt"));
        }
        if !first && hidden_in.len() != seq * h {
            return Err(PyRuntimeError::new_err(format!(
                "forward_pp_qwen35_prefill: hidden_in.len()={} != seq*H={}",
                hidden_in.len(), seq * h)));
        }
        if first && tokens.len() < seq {
            // The first stage indexes `tokens[pos]` for `pos in 0..seq`; only
            // `hidden_in` was length-checked above.
            return Err(PyRuntimeError::new_err(format!(
                "forward_pp_qwen35_prefill: tokens.len()={} < seq={seq}", tokens.len())));
        }
        let mut out: Vec<f32> = if last { Vec::new() } else { Vec::with_capacity(seq * h) };
        for pos in 0..seq {
            let step = if first {
                self.forward_pp_qwen35_impl(tokens[pos], Vec::new(), pos)?
            } else {
                let slice = hidden_in[pos * h..(pos + 1) * h].to_vec();
                self.forward_pp_qwen35_impl(0, slice, pos)?
            };
            if last {
                out = step; // keep only the last position's [vocab]
            } else {
                out.extend_from_slice(&step); // accumulate [seq*H]
            }
        }
        Ok(out)
    }


    /// Batched prefill for a qwen3.6 (qwen3_5) model: embeds `tokens` and runs
    /// every decoder layer batched over T instead of T sequential per-token
    /// forwards (see `plan-batched-prefill.md`). `start_pos` is the KV/
    /// DeltaNet position this batch continues from — NOT reset here; call
    /// `reset_kv_cache()` first for a fresh sequence. Returns the LAST
    /// token's logits. See `forward_qwen35_prefill_impl` (qwen35_forward.rs)
    /// for the CPU-fallback vs GPU-resident path split.
    fn forward_qwen35_prefill(&mut self, tokens: Vec<u32>, start_pos: usize) -> PyResult<Vec<f32>> {
        self.forward_qwen35_prefill_impl(tokens, start_pos)
    }


    /// HOST-SIDE STREAMING ORACLE helper: embed `tokens` -> `[T*hidden]` via
    /// this stage's f16 embed table. Only valid on a window that OWNS
    /// `embed_tokens` (the first window, `pp_first`). Used by the layer-windowed
    /// CPU golden driver (`scripts/qwen38_cpu_golden_stream.py`) so the whole
    /// 27B never has to be resident at once (the ~102.5GB f32 wall). CPU-only.
    fn qwen35_embed_prompt(&mut self, tokens: Vec<u32>) -> PyResult<Vec<f32>> {
        let h = self.qwen35.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("qwen35_embed_prompt needs a qwen3_5 model"))?
            .config.hidden_size;
        let mut hv = vec![0.0f32; tokens.len() * h];
        for (ti, &tok) in tokens.iter().enumerate() {
            hv[ti * h..(ti + 1) * h].copy_from_slice(&self.q35_embed_row(tok as usize, h));
        }
        Ok(hv)
    }


    /// HOST-SIDE STREAMING ORACLE helper: run this instance's RESIDENT layer
    /// window `[pp_start, pp_end)` over `t` tokens from `hidden_in` (`[t*hidden]`
    /// row-major), returning the window's `[t*hidden]` output — or, on the last
    /// window (`pp_last`), the LAST-position logits (`[vocab]`, applies
    /// `model.norm` + lm_head). This is EXACTLY the CPU-fallback prefill path
    /// (`forward_qwen35_prefill_impl`, engine None) factored so a Python driver
    /// can chain K windowed instances and stream all 64 layers with only one
    /// window (~a few GB) resident at a time. Reuses the validated
    /// `forward_pp_range_batched` kernels bit-for-bit (Milestone A/B, cos=1.0 vs
    /// the qwen3.6-mlx oracle). CPU-only (engine must be None).
    fn forward_qwen35_window(&mut self, hidden_in: Vec<f32>, start_pos: usize, t: usize) -> PyResult<Vec<f32>> {
        if self.engine.is_some() {
            return Err(PyRuntimeError::new_err(
                "forward_qwen35_window is the CPU-only streaming oracle path (engine must be None)"));
        }
        let (pp_start, pp_end, pp_last) = (self.pp_start, self.pp_end, self.pp_last);
        let (h, eps, vocab) = {
            let cfg = &self.qwen35.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("forward_qwen35_window needs a qwen3_5 model"))?
                .config;
            (cfg.hidden_size, cfg.rms_norm_eps, cfg.vocab_size)
        };
        if t == 0 {
            // `0 != 0` is false, so the length check below PASSES for t==0 and
            // the last-window `(t - 1) * h` slice underflows `usize`.
            return Err(PyRuntimeError::new_err("forward_qwen35_window: t must be >= 1"));
        }
        if hidden_in.len() != t * h {
            return Err(PyRuntimeError::new_err(format!(
                "forward_qwen35_window: hidden_in len {} != t*hidden {}", hidden_in.len(), t * h)));
        }
        let qm = self.qwen35.as_mut().unwrap();
        let hidden = qm.forward_pp_range_batched(&hidden_in, start_pos, t, pp_start, pp_end);
        if !pp_last {
            return Ok(hidden);
        }
        // Last window: final norm + lm_head on the last position (mirrors the
        // CPU-fallback last-stage in forward_qwen35_prefill_impl).
        let norm_w = qm.weights.f32_slice("model.norm.weight").to_vec();
        let lm_name = qm.lm_head_name.clone();
        let last = &hidden[(t - 1) * h..t * h];
        let normed = model::cpu_rms_norm(last, &norm_w, eps);
        let lm_w = self.q35_f16_host.get(&lm_name)
            .ok_or_else(|| PyRuntimeError::new_err(
                "qwen3_5 lm_head f16 host missing (streaming oracle last window)"))?;
        let lm_f32: Vec<f32> = lm_w.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        Ok(model::cpu_matmul(&normed, &lm_f32, 1, h, vocab))
    }


    /// P3: last-stage forward that argmaxes the logits in Rust and returns
    /// `(token, logit)` — the vocab-wide logit Vec (152k/248k floats) never
    /// crosses the pyo3 boundary, the same win `pp_step_qwen35` gets. Runs the
    /// resident forward (which stashes the pre-`model.norm` hidden for the MTP
    /// head), so the spec driver drives pass A / pass B with this and then
    /// drafts off the stash. Caller must strip any control word from
    /// `hidden_in` first (this expects a plain [hidden] payload). ONLY
    /// meaningful on the last stage (a mid stage would argmax over a hidden
    /// vector — the driver never calls it there).
    fn forward_pp_qwen35_argmax(&mut self, token_id: u32, hidden_in: Vec<f32>, pos: usize) -> PyResult<(u32, f32)> {
        let out = self.forward_pp_qwen35_impl(token_id, hidden_in, pos)?;
        // P4 α fix: `forward_pp_qwen35_impl` just stashed this pass's producing
        // pre-`model.norm` hidden in `q35_last_prenorm`. Record it BY POSITION so a
        // chain refill off a middle verify pass (`mtp_draft_chain`/`mtp_draft_after`
        // with `head_pos = cposA+k`) drafts `d_1` from the hidden that PRODUCED the
        // new real, not merely the freshest (deepest-pass) one. Only while a head is
        // loaded (the stash is skipped otherwise). Bounded ring; most-recent wins.
        if self.mtp_head.is_some() {
            if let Some(pn) = self.q35_last_prenorm.clone() {
                self.q35_prenorm_ring.push_back((pos, pn));
                while self.q35_prenorm_ring.len() > Self::PRENORM_RING_CAP {
                    self.q35_prenorm_ring.pop_front();
                }
            }
        }
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, &v) in out.iter().enumerate() {
            if v > bv { bv = v; bi = i; }
        }
        Ok((bi as u32, bv))
    }


    /// STEP-7 bit-exactness harness (MTP re-gate GPU-verify localization): run
    /// the batched T-token VERIFY on the GPU and compare EACH of the T positions'
    /// stage output to the SERIAL single-token path (`forward_pp_qwen35_impl` —
    /// the working PP_SPEC=0 decode) at the same positions. Both start from the
    /// SAME resident state (snapshotted at `start_pos` and restored between), so
    /// row `i` of the batched verify MUST equal the serial forward of `tokens[i]`
    /// at `start_pos+i` if the batched dispatch is correct. Returns per-position
    /// `(cos, max_abs_diff, argmax_serial, argmax_batched)` over the stage output
    /// (hidden `[h]` on a non-last stage, logits `[vocab]` on the last).
    ///
    /// Localization protocol (run on ONE node, layer-limited load, e.g.
    /// start=0 end=8 so 35B fits a 14GB node):
    ///   * cos≈1 & argmax match every position  → batched verify is correct;
    ///     the garbage is in the DRIVER (position/state threading) or the head.
    ///   * cos≪1 → the batched forward diverges. Re-run with
    ///     `VLLM_VULKAN_SPEC_NO_COLS=1` (forces the qwen35_gemm fallback): if cos
    ///     recovers, the bug is the `mul_mat_vec_{f16,q8_0}_cols` dispatch; if not,
    ///     it is the batched GDN scan / full-attn / MoE (`_prefill_gpu`) mixers.
    /// Leaves resident state as it was at entry (final restore).
    fn debug_qwen35_verify_vs_serial(&mut self, tokens: Vec<u32>, start_pos: usize)
        -> PyResult<Vec<(f64, f64, i64, i64)>>
    {
        let t = tokens.len();
        if t == 0 {
            return Err(PyRuntimeError::new_err("debug_qwen35_verify_vs_serial: empty tokens"));
        }
        // Snapshot resident state at start_pos into a scratch ring slot.
        self.spec_snapshot_impl(0).map_err(PyRuntimeError::new_err)?;
        // Serial reference: single-token forward per position (advances state).
        let mut serial: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (i, &tok) in tokens.iter().enumerate() {
            serial.push(self.forward_pp_qwen35_impl(tok, Vec::new(), start_pos + i)?);
        }
        // Restore, then the batched verify over all T tokens.
        self.spec_restore_impl(0).map_err(PyRuntimeError::new_err)?;
        let batched = self.forward_qwen35_verify_impl(tokens, start_pos)?;
        // Restore once more so the harness is state-neutral for the caller.
        self.spec_restore_impl(0).map_err(PyRuntimeError::new_err)?;
        if batched.len() % t != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "verify result {} not divisible by T={t}", batched.len())));
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


    /// Design-A batched VERIFY (MTP re-gate, spec §2). First/single stage: embed
    /// `tokens` (`[s_R, d_1..d_D]`, T=D+1) and run the verify stack, returning
    /// this stage's output — the FULL `[T*vocab]` logits on the last stage
    /// (row `ti` = candidate `out(ti)`), or the `[T*h]` hidden on a non-last
    /// stage (the batched-PP-hop payload). Advances resident state through all
    /// T tokens and captures the GDN inputs for `qwen35_verify_rollback`.
    /// Qwen3.6 (`qwen3_5`) only. Prefer `forward_qwen35_verify_argmax` on the
    /// last stage (skips marshalling `T*vocab` floats to Python).
    fn forward_qwen35_verify(&mut self, tokens: Vec<u32>, start_pos: usize) -> PyResult<Vec<f32>> {
        self.forward_qwen35_verify_impl(tokens, start_pos)
    }


    /// Design-A batched VERIFY, last-stage argmax form (spec §2). Runs the verify
    /// and argmaxes EACH of the T positions in Rust, returning `[out(0)..out(D)]`
    /// = `[a_out, b_1_out .. b_D_out]` — the exact inputs `resolve_chain` needs,
    /// without marshalling the `T*vocab` logit block to Python (the same win
    /// `forward_pp_qwen35_argmax` gets per token). ONLY meaningful on the last
    /// stage (a non-last stage would argmax over a hidden vector).
    fn forward_qwen35_verify_argmax(&mut self, tokens: Vec<u32>, start_pos: usize) -> PyResult<Vec<u32>> {
        let logits = self.forward_qwen35_verify_impl(tokens, start_pos)?;
        let vocab = self.qwen35.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forward_qwen35_verify_argmax needs a qwen3_5 model"))?
            .config.vocab_size;
        if vocab == 0 || logits.len() % vocab != 0 {
            return Err(PyRuntimeError::new_err(
                "forward_qwen35_verify_argmax: not a last-stage [T*vocab] result (call only on the last stage)"));
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


    /// Design-A partial-accept ROLLBACK (spec §3, option-B). After a batched
    /// verify + `resolve_chain` giving `accept_len=k`, roll resident recurrent
    /// state back to exactly the `k+1` committed tokens (full-attn KV counter
    /// truncate + GDN re-scan of the committed prefix; full accept is a no-op).
    /// `slot` must hold a `spec_snapshot` taken pre-verify at position `R`. See
    /// `qwen35_verify_rollback_impl`.
    fn qwen35_verify_rollback(&mut self, slot: usize, accept_len: usize) -> PyResult<()> {
        self.qwen35_verify_rollback_impl(slot, accept_len).map_err(PyRuntimeError::new_err)
    }


    /// P2 — Design-A batched TENSOR-PARALLEL verify (dense-27B TP-4 spec arm).
    /// Embeds `tokens` (`[s_R, d_1..d_D]`, T=D+1), runs every layer on this rank's
    /// 1/N shard, and all-reduces the two per-layer `[T*h]` partials ONCE each
    /// (comm amortized over all T). Returns the FULL replicated `[T*vocab]` logits
    /// — every rank identical, so the driver resolves the chain LOCALLY (no PP
    /// ring / bcast). `all_reduce` is the vCCL SUM callback (used only when native
    /// comm is disabled). Advances resident state through T tokens + captures the
    /// GDN inputs for `qwen35_tp_verify_rollback`. Prefer the `_argmax` form on the
    /// hot path (skips marshalling `T*vocab` floats).
    fn forward_tp_qwen35_verify(&mut self, py: Python<'_>, tokens: Vec<u32>, start_pos: usize,
                                all_reduce: PyObject) -> PyResult<Vec<f32>> {
        self.forward_tp_qwen35_verify_impl(py, tokens, start_pos, all_reduce)
    }


    /// P2 — TP batched verify, argmax form: runs `forward_tp_qwen35_verify` and
    /// argmaxes EACH of the T replicated-logit rows in Rust, returning
    /// `[out(0)..out(D)]` = `[a_out, b_1_out .. b_D_out]` (the exact inputs the
    /// LOCAL `resolve_chain` needs) without marshalling `T*vocab` floats to Python.
    fn forward_tp_qwen35_verify_argmax(&mut self, py: Python<'_>, tokens: Vec<u32>, start_pos: usize,
                                       all_reduce: PyObject) -> PyResult<Vec<u32>> {
        let logits = self.forward_tp_qwen35_verify_impl(py, tokens, start_pos, all_reduce)?;
        let vocab = self.qwen35.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forward_tp_qwen35_verify_argmax needs a qwen3_5 model"))?
            .config.vocab_size;
        if vocab == 0 || logits.len() % vocab != 0 {
            return Err(PyRuntimeError::new_err("forward_tp_qwen35_verify_argmax: bad [T*vocab] result"));
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


    /// P2 — TP partial-accept ROLLBACK (option-B). Rolls each rank's 1/N recurrent
    /// shard back to the `k+1` committed tokens after a TP verify + LOCAL
    /// `resolve_chain`. TP-specific (GDN re-scan via the sharded mixer); `slot`
    /// holds a `spec_snapshot` taken pre-verify at position `R`.
    fn qwen35_tp_verify_rollback(&mut self, slot: usize, accept_len: usize) -> PyResult<()> {
        self.qwen35_tp_verify_rollback_impl(slot, accept_len).map_err(PyRuntimeError::new_err)
    }


    /// P2 MANDATORY GATE — single-node (runs at TP-N) GPU bit-exactness harness
    /// for the batched TP verify: the TENSOR-PARALLEL analog of
    /// `debug_qwen35_verify_vs_serial`. Runs the PROVEN serial single-token
    /// `forward_tp_qwen35` per position over `tokens` (advancing the sharded
    /// KV/GDN state) as the reference, then the batched `forward_tp_qwen35_verify`
    /// over the SAME T tokens from the SAME start state (snapshot/restore brackets),
    /// and returns per-position `(cos, maxd, argmax_serial, argmax_batched)` over
    /// the FULL replicated `[vocab]` logits. Gate on cluster BEFORE any A/B:
    ///   * cos≈1 AND argmax match at EVERY position ⇒ the batched verify FORWARD
    ///     is bit-exact; any on-cluster identity failure is then in the DRIVER /
    ///     `qwen35_tp_verify_rollback` state-threading (position/KV/GDN rewind),
    ///     NOT the verify math (the step-7 lesson: "cos≈1 ⇒ the garbage is in the
    ///     driver").
    ///   * cos≪1 at a position ⇒ the batched forward itself diverges there —
    ///     re-run with the same `VLLM_VULKAN_SPEC_*` bisection knobs.
    /// NOTE (why the three PP `forward_qwen35_verify_core` fixes do NOT port here):
    /// the TP verify runs the per-token CPU-recurrence mixers `qwen35_delta_net_tp`
    /// / `qwen35_gated_attention_tp` (f32 `cpu_sdpa`, host-authoritative GDN state
    /// — `dn_gpu` is EMPTY in TP) sequentially `0..T`, the SAME mixers the serial
    /// reference uses. So there is no flash head-major transpose (no flash), no GDN
    /// conv-window staleness (no batched prepass; no GPU-authoritative DN advancing
    /// a stale host copy), and no f16-flash-vs-f32 residual (already `cpu_sdpa`) to
    /// port. This harness PROVES that by construction on real hardware.
    /// Uses the SAME `all_reduce` the driver uses (native FFI or callback). Leaves
    /// resident state as it was at entry (final restore + verify-span clear).
    fn debug_tp_qwen35_verify_vs_serial(&mut self, py: Python<'_>, tokens: Vec<u32>,
                                        start_pos: usize, all_reduce: PyObject)
        -> PyResult<Vec<(f64, f64, i64, i64)>>
    {
        let t = tokens.len();
        if t == 0 {
            return Err(PyRuntimeError::new_err("debug_tp_qwen35_verify_vs_serial: empty tokens"));
        }
        // Snapshot the resident (sharded) state at start_pos into scratch slot 0.
        self.spec_snapshot_impl(0).map_err(PyRuntimeError::new_err)?;
        // Serial reference: single-token TP forward per position (advances the
        // sharded KV/GDN state), full replicated [vocab] logits — the exact path
        // the PP_SPEC=0 baseline decodes with.
        let mut serial: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (i, &tok) in tokens.iter().enumerate() {
            serial.push(self.forward_tp_qwen35(py, tok, start_pos + i, all_reduce.clone_ref(py))?);
        }
        // Restore to the pre-serial state, then the batched verify over all T
        // tokens from the IDENTICAL start state.
        self.spec_restore_impl(0).map_err(PyRuntimeError::new_err)?;
        let batched = self.forward_tp_qwen35_verify_impl(py, tokens, start_pos, all_reduce.clone_ref(py))?;
        // Restore once more so the harness is state-neutral for the caller, and
        // drop the verify's pending span/captured GDN inputs (harness, not a real
        // verify — nothing should roll it back).
        self.spec_restore_impl(0).map_err(PyRuntimeError::new_err)?;
        self.spec_verify_gdn_inputs.clear();
        self.spec_verify_span = None;
        if batched.len() % t != 0 {
            return Err(PyRuntimeError::new_err(format!(
                "debug_tp_qwen35_verify_vs_serial: verify result {} not divisible by T={t}", batched.len())));
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


    /// Design-A batched PP hop (spec §4): one pipeline traversal of a T=D+1
    /// verify batch. First stage embeds `tokens` and runs the verify stack;
    /// mid/last stages recv the previous stage's `[T*h]` hidden. Non-last stages
    /// send `[T*h]` onward and return `None`; the last stage argmaxes all T
    /// positions in Rust and returns `[a_out, b_1_out .. b_D_out]`. Buffer-size
    /// sibling of `pp_step_qwen35` (which moves `[h]`); `T` is inferred from
    /// `tokens.len()` on the first stage and the recv count on later stages.
    fn pp_step_qwen35_verify(
        &mut self,
        py: Python<'_>,
        tokens: Vec<u32>,
        start_pos: usize,
        t: usize,
        recv_from: i32,
        send_to: i32,
    ) -> PyResult<Option<Vec<u32>>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_qwen35_verify: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let h = self.qwen35.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("pp_step_qwen35_verify needs a qwen3_5 model"))?
            .config.hidden_size;
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);

        // 1) First stage embeds `tokens`; mid/last stages recv the previous
        //    stage's [T*h] hidden (GIL dropped inside recv_f32).
        let out = if do_recv {
            let hidden = vccl_ffi::recv_f32(py, comm, t * h, recv_from).map_err(PyRuntimeError::new_err)?;
            self.forward_qwen35_verify_core(hidden, start_pos, t)?
        } else {
            self.forward_qwen35_verify_impl(tokens, start_pos)?
        };

        // 2) Non-last: send the [T*h] stage hidden onward. Last: argmax all T.
        if !is_last {
            vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            Ok(None)
        } else {
            let vocab = self.qwen35.as_ref().unwrap().config.vocab_size;
            if vocab == 0 || out.len() % vocab != 0 {
                return Err(PyRuntimeError::new_err(
                    "pp_step_qwen35_verify: not a last-stage [T*vocab] result"));
            }
            let tt = out.len() / vocab;
            let mut outs = Vec::with_capacity(tt);
            for ti in 0..tt {
                let row = &out[ti * vocab..(ti + 1) * vocab];
                let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
                for (i, &v) in row.iter().enumerate() {
                    if v > bv { bv = v; bi = i; }
                }
                outs.push(bi as u32);
            }
            Ok(Some(outs))
        }
    }


    /// DEBUG (Qwen3 only): run a full token sequence through the pure-CPU
    /// reference path and return `(per_layer_hidden_for_last_token, logits)`.
    ///
    /// Always uses the CPU reference (never the f16 GPU path), so it is a
    /// deterministic oracle for numerical parity testing against HuggingFace.
    #[cfg(feature = "gemma")]
    fn debug_qwen_sequence(&mut self, token_ids: Vec<u32>) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        let q = self.qwen.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("debug_qwen_sequence requires a Qwen3 model")
        })?;
        if token_ids.is_empty() {
            return Err(PyRuntimeError::new_err("token_ids must not be empty"));
        }
        // The CPU reference path reads projection weights from host RAM, which
        // are absent in lean-host mode (VLLM_VULKAN_LEAN_HOST).
        if !q.weights.tensors.contains_key("model.layers.0.self_attn.q_proj.weight") {
            return Err(PyRuntimeError::new_err(
                "CPU reference unavailable: projection weights not retained in host \
                 RAM (lean-host mode). Unset VLLM_VULKAN_LEAN_HOST to enable.",
            ));
        }
        for cache in q.kv_caches.iter_mut() {
            cache.seq_len = 0;
        }
        let last = token_ids.len() - 1;
        for (i, &tok) in token_ids.iter().enumerate() {
            if i < last {
                let _ = q.forward(tok, i);
            } else {
                return Ok(q.forward_capture(tok, i));
            }
        }
        unreachable!()
    }


    /// DEBUG (Qwen3 GPU path): run a sequence through the GPU forward and return
    /// (per-layer hidden for the last token, final logits).  Compare against the
    /// transformers reference to localise where the GPU path diverges.
    #[cfg(feature = "gemma")]
    fn debug_qwen_gpu_sequence(&mut self, token_ids: Vec<u32>) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        if self.qwen.is_none() {
            return Err(PyRuntimeError::new_err("requires a Qwen3 model"));
        }
        if self.engine.is_none() {
            return Err(PyRuntimeError::new_err("requires the GPU path"));
        }
        if token_ids.is_empty() {
            return Err(PyRuntimeError::new_err("token_ids must not be empty"));
        }
        for cache in self.qwen.as_mut().unwrap().kv_caches.iter_mut() {
            cache.seq_len = 0;
        }
        let last = token_ids.len() - 1;
        let mut logits = Vec::new();
        let mut layers: Vec<Vec<f32>> = Vec::new();
        for (i, &tok) in token_ids.iter().enumerate() {
            if i < last {
                let _ = self.forward_qwen_gpu(tok, i);
            } else {
                logits = self.forward_qwen_gpu_cap(tok, i, Some(&mut layers));
            }
        }
        Ok((layers, logits))
    }


}


impl VulkanModel {

    /// Shared per-layer TP loop for Qwen3.6 (qwen3_5), factored out of
    /// `forward_tp_qwen35` so the three lm_head consumption modes (full
    /// logits, argmax, top-k — §2.3 of the vocab-sharding plan) don't
    /// triplicate the attention/MLP/all-reduce loop. Runs every resident
    /// layer, then the replicated final norm, and returns `(cfg, normed)` —
    /// callers do their own (sharded or replicated) lm_head tail from there.
    fn qwen35_tp_forward_normed(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        pos: usize,
        all_reduce: PyObject,
    ) -> PyResult<(qwen35::Qwen35Config, Vec<f32>)> {
        let cfg = self
            .qwen35
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forward_tp_qwen35 needs a qwen3_5 model"))?
            .config
            .clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let n = self.tp_size.max(1);

        let native = self.native_comm_enabled();
        let bf16_reduce = self.flags.tp_bf16_reduce;
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (scratch_addr, scratch_len) = if self.reduce_scratch_handle != 0 {
            (self.reduce_scratch.as_ptr() as usize, self.reduce_scratch.len())
        } else { (0usize, 0usize) };

        let reduce = |py: Python<'_>, mut partial: Vec<f32>| -> PyResult<Vec<f32>> {
            if native {
                let t = std::time::Instant::now();
                let ph = if bf16_reduce {
                    if scratch_addr != 0 && partial.len() <= scratch_len * 2 {
                        unsafe {
                            vccl_ffi::all_reduce_bf16_via_scratch(
                                py, comm, scratch_addr, scratch_len, &mut partial)
                        }.map_err(PyRuntimeError::new_err)?
                    } else {
                        vccl_ffi::all_reduce_bf16_sum_inplace(py, comm, &mut partial)
                            .map_err(PyRuntimeError::new_err)?
                    }
                } else if scratch_addr != 0 && partial.len() <= scratch_len {
                    unsafe {
                        vccl_ffi::all_reduce_via_scratch(
                            py, comm, scratch_addr, scratch_len, &mut partial)
                    }.map_err(PyRuntimeError::new_err)?
                } else {
                    vccl_ffi::all_reduce_f32_sum_inplace(py, comm, &mut partial)
                        .map_err(PyRuntimeError::new_err)?
                };
                prof_add("native_allreduce", t);
                prof_add_ns("allr_copy_in", ph.copy_in_ns);
                prof_add_ns("allr_wire", ph.wire_ns);
                prof_add_ns("allr_copy_out", ph.copy_out_ns);
                Ok(partial)
            } else {
                let out = all_reduce.call1(py, (partial,))?;
                out.extract::<Vec<f32>>(py)
            }
        };

        let mut hidden: Vec<f32> = self.q35_embed_row(token_id as usize, h);

        // Q35_TP_FUSED (default ON as of 2026-07-25; =0 for host oracle): collapse
        // each rank's per-reduce-segment host-orchestrated compute into ONE fused CB
        // (GDN + dense MLP). The all-reduce boundaries are UNCHANGED. Full-attn keeps
        // the host path.
        let tp_fused = q35_tp_fused_enabled() && self.engine.is_some();
        for layer_idx in self.pp_start..self.pp_end {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

            let residual = hidden.clone();
            let in_ln = self.qwen35_w(&ln("input_layernorm.weight"));
            let x = model::cpu_rms_norm(&hidden, &in_ln, eps);
            let attn_partial = match cfg.layer_types[layer_idx] {
                qwen35::LayerType::FullAttention =>
                    self.qwen35_gated_attention_tp(&cfg, layer_idx, &x, pos, n),
                qwen35::LayerType::LinearAttention => {
                    let fused = if tp_fused {
                        self.qwen35_delta_net_tp_fused(&cfg, layer_idx, &x, n)
                    } else { None };
                    match fused {
                        Some(v) => v,
                        None => self.qwen35_delta_net_tp(&cfg, layer_idx, &x, n),
                    }
                }
            };
            let attn_out = reduce(py, attn_partial)?;
            let h1: Vec<f32> = residual.iter().zip(&attn_out).map(|(&r, &a)| r + a).collect();

            let residual2 = h1.clone();
            let post_ln = self.qwen35_w(&ln("post_attention_layernorm.weight"));
            let ff_in = model::cpu_rms_norm(&h1, &post_ln, eps);
            let mlp_fused = if tp_fused {
                self.qwen35_dense_mlp_tp_fused(&cfg, layer_idx, &ff_in, n)
            } else { None };
            let mlp_partial = match mlp_fused {
                Some(v) => v,
                None => self.qwen35_dense_mlp_tp(&cfg, layer_idx, &ff_in, n),
            };
            let mlp_out = reduce(py, mlp_partial)?;
            hidden = residual2.iter().zip(&mlp_out).map(|(&r, &m)| r + m).collect();
        }

        let norm_w = self.qwen35_w("model.norm.weight");
        let normed = model::cpu_rms_norm(&hidden, &norm_w, eps);
        // MTP pre-norm capture (TP path). On the PP path only the LAST stage
        // computes the pre-`model.norm` residual and stashes it (see
        // `forward_pp_qwen35_impl` → `q35_last_prenorm`, recorded BY POSITION by
        // `forward_pp_qwen35_argmax`). In TP there is no last stage — every rank
        // runs all layers and, after the per-layer all-reduces, holds the full
        // REPLICATED `hidden` (the exact pre-norm residual the MTP head consumes).
        // Stash it into the SAME position-keyed `q35_prenorm_ring`/`q35_last_prenorm`
        // the PP path fills, so the spec driver's bootstrap
        // `mtp_draft_chain(real, head_base, DEPTH)` — which runs BEFORE the first
        // batched verify (the verify's own `stash_verify_prenorm` covers cycles
        // 2+) — finds the producing hidden via `prenorm_for_pos(head_base)`.
        // Gated on a loaded head → a no-op for plain (non-spec) TP decode, mirroring
        // the PP `mtp_head.is_some()` gate. `pos` is the position FED this pass, so
        // it keys identically to the PP `(pos, pn)` push.
        if self.mtp_head.is_some() {
            self.q35_last_prenorm = Some(hidden.clone());
            self.q35_prenorm_ring.push_back((pos, hidden));
            while self.q35_prenorm_ring.len() > Self::PRENORM_RING_CAP {
                self.q35_prenorm_ring.pop_front();
            }
        }
        Ok((cfg, normed))
    }


    /// P2 — Design-A batched TENSOR-PARALLEL verify for the dense-27B TP-4
    /// spec-decode arm. The TP analog of `forward_qwen35_verify_core`: every rank
    /// runs ALL layers on its 1/N projection shard over the T verify tokens
    /// (`[s_R, d_1..d_D]`), and the two per-layer partials (after attn o_proj /
    /// GDN out_proj, and after MLP down_proj) are completed by ONE all-reduce over
    /// the WHOLE `[T*h]` batch — paying the per-layer comm tax ONCE for all T
    /// tokens (vs T single-token `forward_tp_qwen35` calls' T reduces). THIS is the
    /// amortization the dense-TP arm exists for (comm-bound: reduce COUNT stays
    /// 2/layer, only the payload grows `[h]→[T*h]`).
    ///
    /// The per-rank mixers are the PROVEN single-token TP mixers
    /// (`qwen35_gated_attention_tp` / `qwen35_delta_net_tp` / `qwen35_dense_mlp_tp`)
    /// run sequentially over `t=0..T` IN ORDER, so the resident KV / GDN recurrent
    /// state advances token-by-token exactly as T single-token `forward_tp_qwen35`
    /// calls would — sidestepping the batched-prepass GDN conv-staleness bug class
    /// (see [[qwen36-mtp-draft-head]] step 7). Only the REDUCE is batched (SUM is
    /// linear ⇒ one `[T*h]` reduce is bit-identical to T `[h]` reduces).
    ///
    /// Contract mirrors `forward_qwen35_verify_core`: sets `spec_verify_span`,
    /// captures each GDN layer's batched `[T*h]` input into `spec_verify_gdn_inputs`
    /// (for `qwen35_tp_verify_rollback`), stashes the T pre-norm residuals, and
    /// returns the FULL replicated `[T*vocab]` logits (every rank identical ⇒ the
    /// driver does LOCAL per-rank `resolve_chain`/argmax, no PP ring). Engine-less
    /// (Mac) FALLS BACK to the serial-bit-exact batched CPU reference (identity
    /// gate); the sharded reduce path is GPU/cluster-only and REQUIRES the
    /// single-node `debug_qwen35_verify_vs_serial` bit-exactness gate before any
    /// multi-node A/B (the mandatory step-7 lesson).
    fn forward_tp_qwen35_verify_impl(
        &mut self,
        py: Python<'_>,
        tokens: Vec<u32>,
        start_pos: usize,
        all_reduce: PyObject,
    ) -> PyResult<Vec<f32>> {
        let cfg = self.qwen35.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("forward_tp_qwen35_verify needs a qwen3_5 model"))?
            .config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let vocab = cfg.vocab_size;
        let n = self.tp_size.max(1);
        let t = tokens.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        // Validate BEFORE `spec_verify_span` is set: `tokens` comes straight from
        // Python, and a panic below would abort through the pyo3 boundary having
        // already left a pending verify span behind.
        let embed_w = self.q35_f16_host.get("model.embed_tokens.weight")
            .ok_or_else(|| PyRuntimeError::new_err(
                "forward_tp_qwen35_verify: qwen3_5 embed_tokens f16 host missing"))?;
        if let Some(&tok) = tokens.iter().find(|&&tok| (tok as usize + 1) * h > embed_w.len()) {
            return Err(PyRuntimeError::new_err(format!(
                "forward_tp_qwen35_verify: token id {tok} is outside the embedding table \
                 ({} rows of {h}, vocab {vocab})", embed_w.len() / h.max(1))));
        }

        // The CPU-fallback branch below reads the f16-host lm_head, but only
        // AFTER `forward_pp_range_batched_capture` has advanced the resident KV
        // and GDN state. A missing table there returns an `Err` with a pending
        // `spec_verify_span` and a model that has already moved — the same
        // ordering defect the embed check above was hoisted to fix. Resolve the
        // name and check its presence while nothing has changed yet.
        if self.engine.is_none() {
            let lm_name = self.qwen35.as_ref()
                .map(|m| m.lm_head_name.clone())
                .unwrap_or_default();
            if !self.q35_f16_host.contains_key(&lm_name) {
                return Err(PyRuntimeError::new_err(format!(
                    "forward_tp_qwen35_verify: qwen3_5 lm_head f16 host missing \
                     (TP verify CPU-fallback, '{lm_name}')")));
            }
        }

        self.spec_verify_gdn_inputs.clear();
        self.spec_verify_span = Some((start_pos, t));

        // Embed all T tokens (replicated on every rank).
        let mut hidden: Vec<f32> = {
            let w = self.q35_f16_host.get("model.embed_tokens.weight")
                .expect("checked above");
            let mut hv = vec![0.0f32; t * h];
            for (ti, &tok) in tokens.iter().enumerate() {
                let row = &w[tok as usize * h..(tok as usize + 1) * h];
                for j in 0..h { hv[ti * h + j] = half::f16::from_bits(row[j]).to_f32(); }
            }
            hv
        };

        // Engine-less (Mac): identity-gate via the serial-bit-exact batched CPU
        // reference (the reduce path below is GPU/cluster-only).
        if self.engine.is_none() {
            let mut cap: Vec<(usize, Vec<f32>)> = Vec::new();
            let qm = self.qwen35.as_mut().unwrap();
            let out = qm.forward_pp_range_batched_capture(
                &hidden, start_pos, t, 0, cfg.num_hidden_layers, Some(&mut cap));
            self.spec_verify_gdn_inputs = cap;
            let qm = self.qwen35.as_mut().unwrap();
            let norm_w = qm.weights.f32_slice("model.norm.weight").to_vec();
            let lm_name = qm.lm_head_name.clone();
            let lm_w = self.q35_f16_host.get(&lm_name)
                .ok_or_else(|| PyRuntimeError::new_err(
                    "forward_tp_qwen35_verify: qwen3_5 lm_head f16 host missing \
                     (TP verify CPU-fallback)"))?;
            let lm_f32: Vec<f32> = lm_w.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
            self.stash_verify_prenorm(&out, start_pos, t, h);
            let mut logits = vec![0.0f32; t * vocab];
            for ti in 0..t {
                let normed = model::cpu_rms_norm(&out[ti * h..(ti + 1) * h], &norm_w, eps);
                let l = model::cpu_matmul(&normed, &lm_f32, 1, h, vocab);
                logits[ti * vocab..(ti + 1) * vocab].copy_from_slice(&l);
            }
            return Ok(logits);
        }

        // GPU-resident TP path: size the pre-registered RDMA reduce scratch to the
        // MAX payload this verify reduces — the WHOLE `[T*h]` batch (T=DEPTH+1),
        // not the single-token `[h]` that `set_collective_comm` registers. This is
        // the ONE registration that matters for the batched verify: every one of
        // the 2/layer × num_layers reduces below is `[T*h]`, so if the scratch were
        // only `[h]` (or unregistered) each reduce would overflow it and fall back
        // to the non-scratch all-reduce that pays a per-call `ibv_reg_mr`.
        // `ensure_reduce_scratch` GROWS MONOTONICALLY and early-returns once
        // `reduce_scratch.len() >= t*h`, so this registers exactly ONCE (on the
        // first verify, or the first verify at a larger T) and is then reused by
        // every reduce this cycle AND across all later cycles — the single-token
        // draft reduces (`[h] <= [T*h]`) reuse it too. The `_scratch`/`_regmr`
        // profiling split in the reduce closure below reports whether this is in
        // fact being hit on each node.
        self.ensure_reduce_scratch(t * h);
        let native = self.native_comm_enabled();
        let bf16_reduce = self.flags.tp_bf16_reduce;
        let comm_ptr = self.collective_comm as *mut std::os::raw::c_void;
        let (scratch_addr, scratch_len) = if self.reduce_scratch_handle != 0 {
            (self.reduce_scratch.as_ptr() as usize, self.reduce_scratch.len())
        } else { (0usize, 0usize) };
        // `move` so it doesn't borrow `self` (the layer loop mutably borrows self
        // via the TP mixers); captured fields are all Copy / the PyObject callback.
        //
        // PROFILING (parity with the single-token draft reduce in
        // `qwen35_tp_forward_normed`, which emits `native_allreduce` +
        // `allr_copy_in/wire/copy_out`): the batched verify reduce previously
        // DISCARDED the returned `Phases` and emitted NOTHING, so an A/B PROFILE
        // run could not attribute the 2/layer × num_layers verify reduces at all —
        // the observability gap that let the `[T*h]` comm cost be mis-blamed on a
        // per-call `regMr`. We now time each reduce and bucket it by which path it
        // took: `native_allreduce_verify_scratch` = it HIT the pre-registered
        // `[T*h]` scratch (`ensure_reduce_scratch(t*h)` above, the fast path);
        // `native_allreduce_verify_regmr` = it FELL BACK to the non-scratch
        // all-reduce that pays a per-call `ibv_reg_mr` (scratch absent or still too
        // small). A nonzero `_regmr` count in the profile is the smoking gun that
        // the scratch sizing/registration is not taking effect on that node.
        let reduce = move |py: Python<'_>, mut partial: Vec<f32>| -> PyResult<Vec<f32>> {
            if native {
                let t0 = std::time::Instant::now();
                let (ph, hit_scratch) = if bf16_reduce {
                    if scratch_addr != 0 && partial.len() <= scratch_len * 2 {
                        (unsafe { vccl_ffi::all_reduce_bf16_via_scratch(py, comm_ptr, scratch_addr, scratch_len, &mut partial) }
                            .map_err(PyRuntimeError::new_err)?, true)
                    } else {
                        (vccl_ffi::all_reduce_bf16_sum_inplace(py, comm_ptr, &mut partial)
                            .map_err(PyRuntimeError::new_err)?, false)
                    }
                } else if scratch_addr != 0 && partial.len() <= scratch_len {
                    (unsafe { vccl_ffi::all_reduce_via_scratch(py, comm_ptr, scratch_addr, scratch_len, &mut partial) }
                        .map_err(PyRuntimeError::new_err)?, true)
                } else {
                    (vccl_ffi::all_reduce_f32_sum_inplace(py, comm_ptr, &mut partial)
                        .map_err(PyRuntimeError::new_err)?, false)
                };
                if hit_scratch {
                    prof_add("native_allreduce_verify_scratch", t0);
                } else {
                    prof_add("native_allreduce_verify_regmr", t0);
                }
                prof_add_ns("allr_verify_copy_in", ph.copy_in_ns);
                prof_add_ns("allr_verify_wire", ph.wire_ns);
                prof_add_ns("allr_verify_copy_out", ph.copy_out_ns);
                Ok(partial)
            } else {
                let out = all_reduce.call1(py, (partial,))?;
                out.extract::<Vec<f32>>(py)
            }
        };

        for layer_idx in 0..cfg.num_hidden_layers {
            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
            let in_ln = self.qwen35_w(&ln("input_layernorm.weight"));
            let post_ln = self.qwen35_w(&ln("post_attention_layernorm.weight"));
            let residual = hidden.clone();

            // ── Attention: sharded mixers → [T*h] partial ────────────────────
            // perf/quant-batched-cols: for T>=2 the projections stream ONCE
            // across the T verify tokens through the single-stream cols kernel
            // (`qwen35_*_tp_batched`, mirroring the PP STEP-6 swap) instead of
            // the T serial per-token matvecs that re-streamed the weight T times
            // (the [[gemm-batching-misses-quantized-hotpath]] NO-GO). T==1 keeps
            // the byte-identical per-token path (no cols reuse to gain).
            let attn_partial = if t >= 2 {
                let x_all: Vec<f32> = (0..t)
                    .flat_map(|ti| model::cpu_rms_norm(&hidden[ti * h..(ti + 1) * h], &in_ln, eps))
                    .collect();
                match cfg.layer_types[layer_idx] {
                    qwen35::LayerType::FullAttention =>
                        self.qwen35_gated_attention_tp_batched(&cfg, layer_idx, &x_all, start_pos, t, n),
                    qwen35::LayerType::LinearAttention => {
                        // §3 capture: GDN inputs = the concatenated normed x rows.
                        self.spec_verify_gdn_inputs.push((layer_idx, x_all.clone()));
                        self.qwen35_delta_net_tp_batched(&cfg, layer_idx, &x_all, t, n)
                    }
                }
            } else {
                let is_lin = matches!(cfg.layer_types[layer_idx], qwen35::LayerType::LinearAttention);
                let mut attn_partial = vec![0.0f32; t * h];
                let mut gdn_x = if is_lin { Some(vec![0.0f32; t * h]) } else { None };
                for ti in 0..t {
                    let xt = model::cpu_rms_norm(&hidden[ti * h..(ti + 1) * h], &in_ln, eps);
                    let p = match cfg.layer_types[layer_idx] {
                        qwen35::LayerType::FullAttention =>
                            self.qwen35_gated_attention_tp(&cfg, layer_idx, &xt, start_pos + ti, n),
                        qwen35::LayerType::LinearAttention => {
                            if let Some(g) = gdn_x.as_mut() { g[ti * h..(ti + 1) * h].copy_from_slice(&xt); }
                            self.qwen35_delta_net_tp(&cfg, layer_idx, &xt, n)
                        }
                    };
                    attn_partial[ti * h..(ti + 1) * h].copy_from_slice(&p);
                }
                if let Some(g) = gdn_x { self.spec_verify_gdn_inputs.push((layer_idx, g)); }
                attn_partial
            };
            let attn_out = reduce(py, attn_partial)?;
            let h1: Vec<f32> = residual.iter().zip(&attn_out).map(|(&r, &a)| r + a).collect();

            // ── MLP: sharded dense-SwiGLU mixers → [T*h] partial ──────────────
            let residual2 = h1.clone();
            let mlp_partial = if t >= 2 {
                let ff_all: Vec<f32> = (0..t)
                    .flat_map(|ti| model::cpu_rms_norm(&h1[ti * h..(ti + 1) * h], &post_ln, eps))
                    .collect();
                self.qwen35_dense_mlp_tp_batched(&cfg, layer_idx, &ff_all, t, n)
            } else {
                let mut mlp_partial = vec![0.0f32; t * h];
                for ti in 0..t {
                    let ff = model::cpu_rms_norm(&h1[ti * h..(ti + 1) * h], &post_ln, eps);
                    let p = self.qwen35_dense_mlp_tp(&cfg, layer_idx, &ff, n);
                    mlp_partial[ti * h..(ti + 1) * h].copy_from_slice(&p);
                }
                mlp_partial
            };
            let mlp_out = reduce(py, mlp_partial)?;
            hidden = residual2.iter().zip(&mlp_out).map(|(&r, &m)| r + m).collect();
        }

        // Pre-norm stash (MTP refill) + final norm + per-position lm_head.
        self.stash_verify_prenorm(&hidden, start_pos, t, h);
        let norm_w = self.qwen35_w("model.norm.weight");
        let lm_name = self.qwen35.as_ref().unwrap().lm_head_name.clone();
        let comm2 = self.collective_comm as *mut std::os::raw::c_void;
        let mut logits = vec![0.0f32; t * vocab];
        for ti in 0..t {
            let normed = model::cpu_rms_norm(&hidden[ti * h..(ti + 1) * h], &norm_w, eps);
            let (local, _lo) = self.tp_lmhead_local(&lm_name, &normed, h, vocab);
            let row = match self.lm_shard {
                None => local,
                Some(LmShard { v, .. }) => {
                    let (base, rem) = (v / n, v % n);
                    let max_per = base + usize::from(rem > 0);
                    let mut padded = local;
                    padded.resize(max_per, 0.0);
                    let recv = vccl_ffi::all_gather_f32(py, comm2, &padded, n)
                        .map_err(PyRuntimeError::new_err)?;
                    let mut lg = vec![0.0f32; v];
                    for r in 0..n {
                        let (rlo, rper) = tp::tp_vocab_shard_range(v, r, n);
                        lg[rlo..rlo + rper].copy_from_slice(&recv[r * max_per..r * max_per + rper]);
                    }
                    lg
                }
            };
            logits[ti * vocab..(ti + 1) * vocab].copy_from_slice(&row);
        }
        Ok(logits)
    }


    /// P2 — TENSOR-PARALLEL analog of `qwen35_verify_rollback_impl` (option-B).
    /// After a `forward_tp_qwen35_verify` + LOCAL `resolve_chain` giving
    /// `accept_len=k`, roll each rank's 1/N recurrent shard back to exactly the
    /// `k+1` committed tokens. IDENTICAL to the PP rollback EXCEPT the GDN re-scan
    /// uses the SHARDED `qwen35_delta_net_tp` mixer (the same mixer the TP verify
    /// used) — the PP path's `qwen35_linear_prefill_gpu` is the FULL (unsharded)
    /// mixer and would corrupt the per-rank GDN state. Full accept (`k+1==T`) is a
    /// no-op; full-attn KV re-expose is the shared counter-only advance. No
    /// all-reduce (state re-scan is per-rank local).
    pub(crate) fn qwen35_tp_verify_rollback_impl(&mut self, slot: usize, accept_len: usize) -> Result<(), String> {
        let (r, t_span) = self.spec_verify_span
            .ok_or("qwen35_tp_verify_rollback: no verify pending (call forward_tp_qwen35_verify first)")?;
        let commit_len = accept_len + 1;
        if commit_len > t_span {
            return Err(format!("qwen35_tp_verify_rollback: accept_len {accept_len} exceeds verify T {t_span}"));
        }
        if commit_len == t_span {
            self.spec_verify_gdn_inputs.clear();
            self.spec_verify_span = None;
            return Ok(());
        }
        // 1) Restore GDN (device+host) + rewind full-attn KV counter to R.
        self.spec_restore_impl(slot)?;
        let cfg = self.qwen35.as_ref().ok_or("tp_verify_rollback: no qwen3_5 model")?.config.clone();
        let h = cfg.hidden_size;
        let n = self.tp_size.max(1);
        // 2) Re-expose the committed full-attn K/V (bytes still valid; counter-only).
        if let Some(m) = self.qwen35.as_mut() { m.set_full_attn_seq_len(r + commit_len); }
        // 3) GDN-only re-scan of the committed prefix through the SHARDED mixer —
        //    bit-identical to the verify's first commit_len tokens (same mixer).
        let inputs = std::mem::take(&mut self.spec_verify_gdn_inputs);
        for (layer_idx, x) in &inputs {
            for ti in 0..commit_len {
                let xt = &x[ti * h..(ti + 1) * h];
                let _ = self.qwen35_delta_net_tp(&cfg, *layer_idx, xt, n);
            }
        }
        self.spec_verify_span = None;
        Ok(())
    }


}

/// Input validation at the qwen35 pyo3 seam.
///
/// Reuses the `qwen35_prefill_tests` fixture in `lib.rs` (a real `VulkanModel`
/// with `qwen35: Some(..)`, `engine: None`) rather than rebuilding one, so these
/// gates run against the same harness the prefill parity gates do.
#[cfg(all(test, feature = "qwen35"))]
mod pyseam_qwen35_input_tests {
    use super::*;

    /// PyO3-boundary input validation. Every method below is reachable from
    /// Python with arbitrary arguments, and a Rust panic there ABORTS through
    /// the boundary instead of raising — so each of these must be an `Err`.
    ///
    /// Each case is a real hole a sibling method on the same file already
    /// closed: `forward_pp_qwen35_prefill` rejects `seq == 0` but not a short
    /// `tokens`; `forward_qwen35_window` length-checks `hidden_in` with a test
    /// that is vacuously true at `t == 0`; the TP verify's embed lookup panics
    /// where `forward_qwen35_window`'s lm_head lookup returns a named error.
    /// Every full-attention layer's KV frontier. The probe for "this refusal
    /// did not advance the model", which `spec_verify_span` alone cannot show.
    fn kv_frontiers(vm: &VulkanModel) -> Vec<usize> {
        vm.qwen35.as_ref().map(|m| m.layer_state.iter().filter_map(|s| match s {
            crate::qwen35::LayerState::Full(kv) => Some(kv.seq_len),
            crate::qwen35::LayerState::Linear(_) => None,
        }).collect()).unwrap_or_default()
    }

    #[test]
    fn qwen35_pymethods_reject_bad_input_instead_of_panicking() {
        // Formatting a `PyErr` needs the interpreter, so initialise it up front
        // rather than only for the `with_gil` block below.
        pyo3::prepare_freethreaded_python();

        // `t == 0` passes `hidden_in.len() != t * h` (0 != 0 is false) and then
        // underflows the last-window `(t - 1) * h` slice.
        {
            let mut vm = crate::qwen35_prefill_tests::tiny_qwen35_vulkan_model();
            let e = vm.forward_qwen35_window(Vec::new(), 0, 0)
                .expect_err("t == 0 must be refused");
            assert!(format!("{e}").contains("t must be >= 1"), "got: {e}");
            // ...and a well-formed t == 1 call still works.
            vm.forward_qwen35_window(vec![0.0f32; crate::qwen35_prefill_tests::H], 0, 1)
                .expect("t == 1 must still be served");
        }

        // The first stage indexes `tokens[pos]` for `pos in 0..seq`; only
        // `hidden_in` was length-checked.
        {
            let mut vm = crate::qwen35_prefill_tests::tiny_qwen35_vulkan_model();
            let e = vm.forward_pp_qwen35_prefill(vec![1, 2], Vec::new(), 4)
                .expect_err("tokens shorter than seq must be refused");
            assert!(format!("{e}").contains("tokens.len()=2 < seq=4"), "got: {e}");
            // ...and tokens.len() == seq is still served.
            vm.forward_pp_qwen35_prefill(vec![1, 2, 3, 4], Vec::new(), 4)
                .expect("a full-length prompt must still be served");
        }

        // The TP verify embeds Python-supplied ids out of `q35_f16_host`.
        pyo3::Python::with_gil(|py| {
            // (a) an out-of-range token id sliced the f16 embed table directly.
            let mut vm = crate::qwen35_prefill_tests::tiny_qwen35_vulkan_model();
            let e = vm.forward_tp_qwen35_verify_impl(py, vec![0, crate::qwen35_prefill_tests::VOCAB as u32], 0, py.None())
                .expect_err("a token id at vocab_size must be refused");
            assert!(format!("{e}").contains("outside the embedding table"), "got: {e}");
            // ...and the refusal must not leave a pending verify span behind,
            // which a panic at the embed slice would have done.
            assert!(vm.spec_verify_span.is_none(),
                    "a refused verify must not leave `spec_verify_span` set");

            // (b) the f16 host table absent (the lean-host load shape).
            let mut vm = crate::qwen35_prefill_tests::tiny_qwen35_vulkan_model();
            vm.q35_f16_host.remove("model.embed_tokens.weight");
            let e = vm.forward_tp_qwen35_verify_impl(py, vec![0, 1], 0, py.None())
                .expect_err("a missing embed table must be refused");
            assert!(format!("{e}").contains("embed_tokens f16 host missing"), "got: {e}");

            // (c) the lm_head table absent, on the same CPU-fallback path.
            //
            // Refusing is only half of it. The lm_head read sits AFTER
            // `forward_pp_range_batched_capture`, so a refusal raised where the
            // read happens leaves a pending `spec_verify_span` on a model whose
            // resident KV / GDN state has already advanced by T tokens — the
            // caller's rollback then has no span to roll back to. Assert BOTH
            // the span and the KV frontiers, so the check cannot slide back
            // down past the state change.
            let mut vm = crate::qwen35_prefill_tests::tiny_qwen35_vulkan_model();
            vm.q35_f16_host.remove("lm_head.weight");
            let before = kv_frontiers(&vm);
            let e = vm.forward_tp_qwen35_verify_impl(py, vec![0, 1], 0, py.None())
                .expect_err("a missing lm_head table must be refused");
            assert!(format!("{e}").contains("lm_head f16 host missing"), "got: {e}");
            assert!(vm.spec_verify_span.is_none(),
                    "a refused verify must not leave `spec_verify_span` set");
            assert_eq!(kv_frontiers(&vm), before,
                       "a refused verify must not advance the model's KV frontiers");

            // ...and the valid call still returns [T*vocab] logits.
            let mut vm = crate::qwen35_prefill_tests::tiny_qwen35_vulkan_model();
            let ok = vm.forward_tp_qwen35_verify_impl(py, vec![0, 1], 0, py.None())
                .expect("a well-formed TP verify must still be served");
            assert_eq!(ok.len(), 2 * crate::qwen35_prefill_tests::VOCAB);
        });
    }

    /// The SIBLING of the TP-verify ordering defect, in the single-node /
    /// PP verify core.
    ///
    /// `forward_qwen35_verify_core` set `spec_verify_span`, ran
    /// `forward_pp_range_batched_capture`, and only THEN read the f16-host
    /// lm_head — with an `expect`, so a missing table aborted the process
    /// through the pyo3 boundary; converted to an `Err` it would still have
    /// returned with a pending verify span on a model whose KV had already
    /// advanced by T. Both properties are asserted here so the check cannot
    /// slide back down past the state change.
    #[test]
    fn qwen35_verify_core_refuses_a_missing_lm_head_before_it_advances_state() {
        pyo3::prepare_freethreaded_python();
        let h = crate::qwen35_prefill_tests::H;
        let mut vm = crate::qwen35_prefill_tests::tiny_qwen35_vulkan_model();
        vm.q35_f16_host.remove("lm_head.weight");
        let before = kv_frontiers(&vm);
        let e = vm.forward_qwen35_verify_core(vec![0.0f32; 2 * h], 0, 2)
            .expect_err("a missing lm_head table must be refused");
        assert!(format!("{e}").contains("lm_head f16 host missing"), "got: {e}");
        assert!(vm.spec_verify_span.is_none(),
                "a refused verify must not leave `spec_verify_span` set");
        assert_eq!(kv_frontiers(&vm), before,
                   "a refused verify must not advance the model's KV frontiers");

        // ...and with the table present the same call is still served.
        let mut vm = crate::qwen35_prefill_tests::tiny_qwen35_vulkan_model();
        let ok = vm.forward_qwen35_verify_core(vec![0.0f32; 2 * h], 0, 2)
            .expect("a well-formed verify must still be served");
        assert_eq!(ok.len(), 2 * crate::qwen35_prefill_tests::VOCAB);
    }

}
