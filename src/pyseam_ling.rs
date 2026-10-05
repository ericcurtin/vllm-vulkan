// SPDX-License-Identifier: Apache-2.0
//! Per-model pyo3 seam for `ling` — moved verbatim out of the monolithic
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

    // ── Ling / BailingMoeV3 serve seams ─────────────────────────────────────
    // The `serve_dist` launcher resolves the per-arch PP seam BY NAME
    // (`forward_pp_<arch>_prefill` / `pp_step_<arch>_logits`) for
    // `--model-type bailing`. Implemented: `forward_pp_bailing_prefill`,
    // `forward_pp_bailing`, `reset_bailing_decode_state` and
    // `pp_step_bailing_logits`. Only `pp_step_bailing` (the hidden-only PP step
    // without the logits ring-back) is still a stub that returns an error. The
    // seams follow the kimi ones, except that the prefill resets the decode
    // state itself and `pp_step_bailing_logits` takes no `pos` (Ling tracks its
    // decode position internally).
    fn forward_pp_bailing_prefill(
        &mut self,
        tokens: Vec<u32>,
        hidden_in: Vec<f32>,
        seq: usize,
    ) -> PyResult<Vec<f32>> {
        // SERVE-path prefill: run the prompt through the STATEFUL resident decode
        // path (`forward_pp_stage` per position), leaving this window's resident
        // decode state (KDA per-key-channel recurrence + conv sliding window + MLA KV
        // cache — host OR GPU) POPULATED so the subsequent `pp_step_bailing_logits`
        // decode continues from the prompt context (not from a fresh zero state).
        //
        // Bit-identical to the stateless `forward_window` scan on the same positions
        // (the decode==prefill contract, gate `ling_decode_eq_prefill_real_weights`),
        // and — crucially — it works for the GPU-resident window: `load_gpu_resident`
        // CONSUMES `layers`/`embed`/`final_norm`/`lm_head` into the GPU stage, so the
        // old CPU `forward_window`/`embed`/`lm_head` path was empty (returned
        // "missing embed") under the recommended `VLLM_VULKAN_LING_GPU_RESIDENT=1`
        // config. `forward_pp_stage` dispatches to that GPU stage.
        let (h, first, last) = {
            let m = self.ling.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("forward_pp_bailing_prefill needs a bailing_hybrid model")
            })?;
            (m.cfg.hidden_size,
             m.layer_start == 0,
             m.layer_end >= m.cfg.num_hidden_layers)
        };
        if seq == 0 {
            return Err(PyRuntimeError::new_err("forward_pp_bailing_prefill: empty prompt"));
        }
        if first {
            if tokens.len() < seq {
                return Err(PyRuntimeError::new_err(format!(
                    "forward_pp_bailing_prefill: tokens.len()={} < seq={}",
                    tokens.len(), seq)));
            }
            // Check every consumed token BEFORE the reset below, so a bad prompt
            // raises without clobbering the current decode state.
            let vocab = self.ling.as_ref().unwrap().cfg.vocab_size;
            if let Some((i, &t)) = tokens[..seq].iter().enumerate().find(|(_, &t)| t as usize >= vocab) {
                return Err(PyRuntimeError::new_err(format!(
                    "forward_pp_bailing_prefill: token[{i}]={t} >= vocab_size {vocab}")));
            }
        } else if hidden_in.len() != seq * h {
            return Err(PyRuntimeError::new_err(format!(
                "forward_pp_bailing_prefill: hidden_in.len()={} != seq*H={}",
                hidden_in.len(), seq * h)));
        }
        // Fresh decode session for this request (idempotent with the OP_RESET
        // request boundary — safe to reset here so a missed reset never leaks state).
        self.ling.as_mut().unwrap().reset_decode_state().map_err(PyRuntimeError::new_err)?;
        // Advance state through every prompt position. First stage embeds each token;
        // mid/last stages consume the previous stage's per-position hidden. Mid/first
        // stages accumulate the [seq*H] hidden to ship onward; the last stage
        // computes the [vocab] lm_head only at the LAST position (state still
        // advances through all seq).
        let mut out: Vec<f32> = if last { Vec::new() } else { Vec::with_capacity(seq * h) };
        for t in 0..seq {
            let m = self.ling.as_mut().unwrap();
            let want_logits = t + 1 == seq;
            let o = if first {
                m.forward_pp_stage_opt(tokens[t], &[], want_logits)
            } else {
                m.forward_pp_stage_opt(0, &hidden_in[t * h..(t + 1) * h], want_logits)
            }.map_err(PyRuntimeError::new_err)?;
            if last {
                if t + 1 == seq {
                    out = o;
                }
            } else {
                out.extend_from_slice(&o);
            }
        }
        Ok(out)
    }


    /// STATEFUL RESIDENT-DECODE seam: one PP-stage single-token decode step,
    /// advancing this window's resident decode state (KDA recurrence + conv sliding
    /// window + MLA KV cache) IN PLACE. First stage embeds `token_id` (ignores
    /// `hidden_in`); mid stages consume + return the `[H]` hidden; the last stage
    /// returns `[vocab]` logits (final_norm + untied lm_head on the decoded token).
    /// This is the non-comm seam the per-process fleet harness chains across stages
    /// (file/NAS hidden handoff), the twin of `forward_pp_bailing_prefill`. Chaining
    /// it over a token stream is BIT-IDENTICAL to the prefill window forward — the
    /// decode==prefill contract (Rust gate `ling_decode_eq_prefill_real_weights`).
    fn forward_pp_bailing(&mut self, token_id: u32, hidden_in: Vec<f32>, pos: usize)
        -> PyResult<Vec<f32>> {
        let m = self.ling.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("forward_pp_bailing needs a bailing_hybrid model")
        })?;
        m.forward_pp_stage(token_id, &hidden_in, pos).map_err(PyRuntimeError::new_err)
    }


    /// Reset (re-zero) the Ling resident decode state — start a fresh decode session
    /// (new sequence). Mirrors `reset_decode_state` on the other resident arches.
    fn reset_bailing_decode_state(&mut self) -> PyResult<()> {
        let m = self.ling.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("reset_bailing_decode_state needs a bailing_hybrid model")
        })?;
        m.reset_decode_state().map_err(PyRuntimeError::new_err)
    }


    #[allow(clippy::too_many_arguments)]
    fn pp_step_bailing(
        &mut self,
        _token_id: u32,
        _pos: usize,
        _recv_from: i32,
        _send_to: i32,
    ) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "pp_step_bailing: not implemented; use pp_step_bailing_logits (the PP decode step with the logits ring-back)",
        ))
    }


    /// DISTRIBUTED-SERVE native-vCCL PP decode step for Ling (mirrors
    /// `pp_step_kimi_logits`): recv the previous stage's `[H]` hidden → resident
    /// stateful decode of this window (state advances IN PLACE) → send the `[H]`
    /// hidden onward, OR (last stage) ring the FULL `[vocab]` logits back to rank0 so
    /// vLLM's Sampler sees the whole distribution. The launcher calls it pos-free —
    /// Ling's `forward_pp_stage` tracks its decode position internally (KDA recurrence
    /// + conv window + MLA KV cache advance in place; `pos` is ignored). Bit-exact
    /// with the prefill window forward on the same token stream. Uses the plain
    /// (per-call-registered) hidden hop + the shared registered `pp_vocab_ring` for
    /// the ring-back; requires `set_collective_comm` + `VLLM_VULKAN_NATIVE_COMM!=0`.
    fn pp_step_bailing_logits(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
    ) -> PyResult<Option<Vec<f32>>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_bailing_logits: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let (h, vocab) = {
            let m = self.ling.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("pp_step_bailing_logits needs a bailing_hybrid model"))?;
            (m.cfg.hidden_size, m.cfg.vocab_size)
        };
        // Before any recv: a miswired ring must error, not block or send garbage.
        self.ling_check_role(recv_from, send_to)?;
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;

        // 1) recv the previous stage's [H] hidden (empty on the first stage, which
        //    embeds token_id).
        let hidden_in: Vec<f32> = if do_recv {
            vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
        } else {
            Vec::new()
        };

        // 2) resident stateful decode of this window (advances state in place). [H]
        //    on mid stages, [vocab] on the last.
        let out = self.ling.as_mut().unwrap().forward_pp_stage(token_id, &hidden_in, 0)
            .map_err(PyRuntimeError::new_err)?;

        // 3) route the result.
        if is_first && is_last {
            return Ok(Some(out)); // STANDALONE N=1: `out` is already [vocab].
        }
        if !is_last {
            vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            if is_first {
                let logits = self.pp_recv_vocab(py, vocab, last_rank)?;
                Ok(Some(logits))
            } else {
                Ok(None)
            }
        } else {
            self.pp_send_vocab(py, &out, 0)?;
            Ok(None)
        }
    }


}


impl VulkanModel {
    /// The `pp_step_bailing_logits` ring role (`recv_from < 0` = first stage,
    /// `send_to < 0` = last stage) must match the layer window this model loaded.
    /// A mismatch would recv on the first stage (blocking forever) or send a
    /// `[vocab]` output as a hidden vector from a mis-wired tail.
    fn ling_check_role(&self, recv_from: i32, send_to: i32) -> PyResult<()> {
        let m = match self.ling.as_ref() {
            Some(m) => m,
            None => return Ok(()), // the callers report the missing model themselves
        };
        let (first, last) = (recv_from < 0, send_to < 0);
        let (pp_first, pp_last) = (m.layer_start == 0, m.layer_end >= m.cfg.num_hidden_layers);
        if first != pp_first || last != pp_last {
            return Err(PyRuntimeError::new_err(format!(
                "pp_step_bailing: ring role (recv_from={recv_from} -> first={first}, send_to={send_to} \
                 -> last={last}) does not match this stage (first={pp_first}, last={pp_last})")));
        }
        Ok(())
    }
}
