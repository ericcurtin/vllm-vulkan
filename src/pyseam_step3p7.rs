// SPDX-License-Identifier: Apache-2.0
//! Per-model pyo3 seam for `step3p7` — moved verbatim out of the monolithic
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

    /// Step-3.7 PP-window prefill seam (fit-to-validate harness). On the first stage
    /// `tokens` is the prompt (embedded here); on later stages `hidden_in` is the
    /// previous stage's `[seq*H]` hidden. Returns `[seq*H]` hidden for a middle stage,
    /// or `[vocab]` last-position logits for the last stage. Mirrors the Ling
    /// `forward_pp_bailing_prefill` seam; the heavy lifting is in
    /// `Step3p7Model::pp_prefill` so all math stays in the bit-exact step3p7 module.
    fn forward_pp_step3p7_prefill(
        &self,
        tokens: Vec<u32>,
        hidden_in: Vec<f32>,
        seq: usize,
    ) -> PyResult<Vec<f32>> {
        let m = self.step3p7.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("forward_pp_step3p7_prefill needs a step3p7 model")
        })?;
        m.pp_prefill(&tokens, hidden_in, seq).map_err(PyRuntimeError::new_err)
    }


    /// SERVE-path stateful prefill: reset the decode session, then advance state through
    /// every prompt position (first stage embeds each token; mid/last consume the prev
    /// stage's per-position hidden). Mid/first accumulate `[seq*H]`; last keeps only the
    /// last position's `[vocab]`. Mirrors `forward_pp_bailing_prefill`.
    fn forward_pp_step3p7_decode_prefill(
        &mut self,
        tokens: Vec<u32>,
        hidden_in: Vec<f32>,
        seq: usize,
    ) -> PyResult<Vec<f32>> {
        let (h, first, last) = {
            let m = self.step3p7.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("forward_pp_step3p7_decode_prefill needs a step3p7 model")
            })?;
            (m.config.hidden_size, m.pp_first, m.pp_last)
        };
        if seq == 0 {
            return Err(PyRuntimeError::new_err("forward_pp_step3p7_decode_prefill: empty prompt"));
        }
        if first && tokens.len() < seq {
            return Err(PyRuntimeError::new_err(format!(
                "forward_pp_step3p7_decode_prefill: tokens.len()={} < seq={seq}", tokens.len())));
        }
        if !first && hidden_in.len() != seq * h {
            return Err(PyRuntimeError::new_err(format!(
                "forward_pp_step3p7_decode_prefill: hidden_in.len()={} != seq*H={}", hidden_in.len(), seq * h)));
        }
        self.step3p7.as_mut().unwrap().reset_decode_state();
        let mut out: Vec<f32> = if last { Vec::new() } else { Vec::with_capacity(seq * h) };
        for t in 0..seq {
            let m = self.step3p7.as_mut().unwrap();
            let o = if first {
                m.decode_step(tokens[t], &[]).map_err(PyRuntimeError::new_err)?
            } else {
                m.decode_step(0, &hidden_in[t * h..(t + 1) * h]).map_err(PyRuntimeError::new_err)?
            };
            if last {
                if t + 1 == seq { out = o; }
            } else {
                out.extend_from_slice(&o);
            }
        }
        Ok(out)
    }


    /// Single-token non-comm decode step (first stage embeds `token_id`, ignores
    /// `hidden_in`; mid returns `[H]`; last returns `[vocab]`). The per-process fleet
    /// harness chains this across stages via file/NAS hidden handoff. `_pos` is tracked
    /// internally (the KV cache length).
    fn forward_pp_step3p7(&mut self, token_id: u32, hidden_in: Vec<f32>, _pos: usize)
        -> PyResult<Vec<f32>> {
        let m = self.step3p7.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("forward_pp_step3p7 needs a step3p7 model")
        })?;
        m.decode_step(token_id, &hidden_in).map_err(PyRuntimeError::new_err)
    }


    /// Reset the step3p7 decode session (new sequence).
    fn reset_step3p7_decode_state(&mut self) -> PyResult<()> {
        let m = self.step3p7.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("reset_step3p7_decode_state needs a step3p7 model")
        })?;
        m.reset_decode_state();
        Ok(())
    }


    /// DISTRIBUTED-SERVE native-vCCL PP decode step (mirrors `pp_step_bailing_logits`):
    /// recv the previous stage's `[H]` hidden → resident stateful decode of this window
    /// (state + KV advance IN PLACE) → send `[H]` onward, OR (last stage) ring the FULL
    /// `[vocab]` back to rank0. TP-2 all-reduce is folded inside the resident decode.
    /// Requires `set_collective_comm` + `VLLM_VULKAN_NATIVE_COMM!=0`.
    fn pp_step_step3p7_logits(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
    ) -> PyResult<Option<Vec<f32>>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_step3p7_logits: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let (h, vocab) = {
            let m = self.step3p7.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("pp_step_step3p7_logits needs a step3p7 model"))?;
            (m.config.hidden_size, m.config.vocab_size)
        };
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;

        let hidden_in: Vec<f32> = if do_recv {
            vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
        } else {
            Vec::new()
        };
        let out = self.step3p7.as_mut().unwrap()
            .decode_step(token_id, &hidden_in).map_err(PyRuntimeError::new_err)?;

        if is_first && is_last {
            return Ok(Some(out));
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
