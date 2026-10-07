// SPDX-License-Identifier: Apache-2.0
//! Per-model pyo3 seam for `kimi` (Kimi-Linear PP serve entry points). Kept as a
//! separate `#[pymethods] impl VulkanModel` block via pyo3's `multiple-pymethods`
//! feature so the per-model code stays out of the monolithic block in `lib.rs`.
#![allow(clippy::all)]

use crate::*;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;


#[pymethods]
impl VulkanModel {

    /// Kimi-Linear PP stage step (Python-driven hop; mirrors `forward_pp_nemotron`).
    /// First stage embeds `token_id` (pass `hidden_in=[]`); mid/tail stages consume
    /// the previous stage's `[H]` hidden. Returns the `[H]` hidden to ship onward,
    /// or the `[vocab]` logits on the last stage. The resident KDA recurrence + conv
    /// window + MLA KV cache advance IN PLACE inside the owning stage.
    fn forward_pp_kimi(&mut self, token_id: u32, hidden_in: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        let m = self.kimi.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("forward_pp_kimi needs a kimi_linear model"))?;
        m.forward_pp_stage(token_id, &hidden_in, pos).map_err(PyRuntimeError::new_err)
    }


    /// DISTRIBUTED-SERVE cache-populating prefill for Kimi-Linear (the companion
    /// to `pp_step_kimi_logits`, resolved by `serve_head.py` as
    /// `forward_pp_kimi_prefill`). Streams the whole prompt through this PP stage,
    /// advancing the resident KDA recurrence + short-conv window + MLA KV cache IN
    /// PLACE, so the subsequent `pp_step_kimi_logits` decode resumes from the
    /// prompt's end.
    ///
    /// Kimi is POSITION-INTERNAL: `KimiModel::forward_pp_stage` ignores `pos` and
    /// advances its own resident recurrence/window, so there is no attention
    /// `seq_len` to fill (unlike nemotron/qwen3.6) — the loop order is what carries
    /// position. Like the other hybrids, prefill reuses the SAME single-token
    /// `forward_pp_stage` the decode seam runs, so prefill≡decode cache backing is
    /// automatic (no batched kernel, no fold flag). `pos` is passed as the loop
    /// index for form only.
    ///
    ///  - FIRST stage (`layer_start == 0`): `tokens` = full prompt `[seq]`. Returns
    ///    `[seq*H]`. If ALSO last (NR==1): the LAST position's `[vocab]`.
    ///  - MID stage: `hidden_in` = `[seq*H]`; returns `[seq*H]`.
    ///  - LAST stage (`layer_end == num_hidden_layers`): `hidden_in` = `[seq*H]`;
    ///    returns the LAST position's `[vocab]` logits.
    fn forward_pp_kimi_prefill(
        &mut self,
        tokens: Vec<u32>,
        hidden_in: Vec<f32>,
        seq: usize,
    ) -> PyResult<Vec<f32>> {
        let (h, first, last) = {
            let m = self.kimi.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("forward_pp_kimi_prefill needs a kimi_linear model")
            })?;
            (m.cfg.hidden_size, m.layer_start == 0, m.layer_end == m.cfg.num_hidden_layers)
        };
        if seq == 0 {
            return Err(PyRuntimeError::new_err("forward_pp_kimi_prefill: empty prompt"));
        }
        if first {
            // Check every consumed token up front: `tokens` comes straight from
            // Python, and a short or out-of-vocab prompt must raise, not panic,
            // and must not advance the decode state part-way.
            if tokens.len() < seq {
                return Err(PyRuntimeError::new_err(format!(
                    "forward_pp_kimi_prefill: tokens.len()={} < seq={}", tokens.len(), seq)));
            }
            let vocab = self.kimi.as_ref().unwrap().cfg.vocab_size;
            if let Some((i, &t)) = tokens[..seq].iter().enumerate().find(|(_, &t)| t as usize >= vocab) {
                return Err(PyRuntimeError::new_err(format!(
                    "forward_pp_kimi_prefill: token[{i}]={t} >= vocab_size {vocab}")));
            }
        } else if hidden_in.len() != seq * h {
            return Err(PyRuntimeError::new_err(format!(
                "forward_pp_kimi_prefill: hidden_in.len()={} != seq*H={}",
                hidden_in.len(), seq * h)));
        }
        let m = self.kimi.as_mut().unwrap();
        let mut out: Vec<f32> = if last { Vec::new() } else { Vec::with_capacity(seq * h) };
        for pos in 0..seq {
            // The last stage computes the [vocab] lm_head only at the last position.
            let want_logits = pos + 1 == seq;
            let step = if first {
                m.forward_pp_stage_opt(tokens[pos], &[], want_logits)
            } else {
                m.forward_pp_stage_opt(0, &hidden_in[pos * h..(pos + 1) * h], want_logits)
            }.map_err(PyRuntimeError::new_err)?;
            if last {
                if want_logits { out = step; } // keep only the last position's [vocab]
            } else {
                out.extend_from_slice(&step); // accumulate [seq*H]
            }
        }
        Ok(out)
    }


    /// Fused native-vCCL PP step for Kimi-Linear (mirrors `pp_step_nemotron`): recv
    /// the previous stage's hidden (if not first) → resident stage forward → send
    /// onward (native, no PyList) OR Rust argmax on the last stage. `recv_from < 0`
    /// ⇒ first stage (embeds `token_id`); `send_to < 0` ⇒ last stage (returns
    /// `Some((tok, logit))`). Requires `set_collective_comm` + `VLLM_VULKAN_NATIVE_COMM!=0`.
    fn pp_step_kimi(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        pos: usize,
        recv_from: i32,
        send_to: i32,
    ) -> PyResult<Option<(u32, f32)>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_kimi: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let h = self.kimi.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("pp_step_kimi needs a kimi_linear model"))?
            .cfg.hidden_size;
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);

        // Pin the persistent [H] PP-hop scratches ONCE so vCCL's send/recv skip the
        // per-call `ibv_reg_mr`/dereg temp MR (the ~700 ms/tok Kimi PP-3 comm floor).
        // Gated by `VLLM_VULKAN_REG_REDUCE` + libvccl exposing `vcclCommRegister`;
        // otherwise the hop helpers fall back to a fresh Vec (correct, slower).
        let want_reg = self.flags.reg_reduce && vccl_ffi::registration_available();
        let km = self.kimi.as_mut().unwrap();
        pin_pp_hops(comm, want_reg, &mut km.pp_hop_recv, &mut km.pp_hop_send, h, do_recv, !is_last,
                    "pp_step_kimi");
        // 1) recv the previous stage's hidden (empty on the first stage: it embeds token_id).
        let hidden_in = if do_recv { pp_hop_recv(py, comm, &mut km.pp_hop_recv, h, recv_from)? } else { Vec::new() };

        let out = self.kimi.as_mut().unwrap().forward_pp_stage(token_id, &hidden_in, pos)
            .map_err(PyRuntimeError::new_err)?;

        if !is_last {
            // 3) send the [H] hidden onward.
            pp_hop_send(py, comm, &mut self.kimi.as_mut().unwrap().pp_hop_send, &out, send_to)?;
            Ok(None)
        } else {
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in out.iter().enumerate() {
                if v > bv { bv = v; bi = i; }
            }
            Ok(Some((bi as u32, bv)))
        }
    }


    /// DISTRIBUTED-SERVE twin of `pp_step_kimi` (mirrors `pp_step_laguna_logits`):
    /// the last stage rings the FULL `[vocab]` logits back to rank0 (raw f32 over
    /// vCCL, NO `Vec<f32>→PyList` marshal) instead of argmaxing, so vLLM's Sampler
    /// on rank0 sees the whole distribution. This is the `pp_step_kimi_logits`
    /// seam `scripts/serve_dist.py` resolves for `--model-type kimi`.
    ///
    /// The launcher calls it pos-free — `(token_id, recv_from, send_to, last_rank)`.
    /// Kimi's `forward_pp_stage` tracks its decode position internally (the KDA
    /// recurrence + conv window + MLA KV cache advance in place and it IGNORES the
    /// `pos` argument), so `0` is passed. Reuses the pre-pinned `[H]` hidden
    /// scratch (`pp_hop_recv`/`pp_hop_send`, `pin_pp_hop`) exactly as `pp_step_kimi`;
    /// the `[vocab]` ring-back goes through the registered `pp_vocab_ring`
    /// (`pp_send_vocab`/`pp_recv_vocab`). Bit-exact with `pp_step_kimi`'s
    /// last-stage logits. Requires
    /// `set_collective_comm` + `VLLM_VULKAN_NATIVE_COMM!=0`.
    fn pp_step_kimi_logits(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
    ) -> PyResult<Option<Vec<f32>>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_kimi_logits: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let (h, vocab) = {
            let m = self.kimi.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("pp_step_kimi_logits needs a kimi_linear model"))?;
            (m.cfg.hidden_size, m.cfg.vocab_size)
        };
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;

        // Pin the [H] PP-hop scratches once and recv the previous stage's hidden,
        // exactly as `pp_step_kimi`.
        let want_reg = self.flags.reg_reduce && vccl_ffi::registration_available();
        let km = self.kimi.as_mut().unwrap();
        pin_pp_hops(comm, want_reg, &mut km.pp_hop_recv, &mut km.pp_hop_send, h, do_recv, !is_last,
                    "pp_step_kimi_logits");
        let hidden_in = if do_recv { pp_hop_recv(py, comm, &mut km.pp_hop_recv, h, recv_from)? } else { Vec::new() };

        // 2) resident stage forward (Kimi ignores pos → internal tracking). [H]
        //    on mid stages, [vocab] on the last.
        let out = self.kimi.as_mut().unwrap().forward_pp_stage(token_id, &hidden_in, 0)
            .map_err(PyRuntimeError::new_err)?;

        // 3) route the result.
        if is_first && is_last {
            return Ok(Some(out)); // STANDALONE N=1: `out` is already [vocab].
        }
        if !is_last {
            // FIRST / MID: forward `[H]` onward, then (rank0 only) recv the ring-back.
            pp_hop_send(py, comm, &mut self.kimi.as_mut().unwrap().pp_hop_send, &out, send_to)?;
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


}
