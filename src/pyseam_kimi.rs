// SPDX-License-Identifier: Apache-2.0
//! Per-model pyo3 seam for `kimi` — moved verbatim out of the monolithic
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

    /// Kimi-Linear PP stage step (Python-driven hop; mirrors `forward_pp_nemotron`).
    /// First stage embeds `token_id` (pass `hidden_in=[]`); mid/tail stages consume
    /// the previous stage's `[H]` hidden. Returns the `[H]` hidden to ship onward,
    /// or the `[vocab]` logits on the last stage. The resident KDA recurrence + conv
    /// window + MLA KV cache advance IN PLACE inside the owning stage.
    fn forward_pp_kimi(&mut self, token_id: u32, hidden_in: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        let m = self.kimi.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("forward_pp_kimi needs a kimi_linear model"))?;
        Ok(m.forward_pp_stage(token_id, &hidden_in, pos))
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
        if !first && hidden_in.len() != seq * h {
            return Err(PyRuntimeError::new_err(format!(
                "forward_pp_kimi_prefill: hidden_in.len()={} != seq*H={}",
                hidden_in.len(), seq * h)));
        }
        let m = self.kimi.as_mut().unwrap();
        let mut out: Vec<f32> = if last { Vec::new() } else { Vec::with_capacity(seq * h) };
        for pos in 0..seq {
            let step = if first {
                m.forward_pp_stage(tokens[pos], &[], pos)
            } else {
                m.forward_pp_stage(0, &hidden_in[pos * h..(pos + 1) * h], pos)
            };
            if last {
                out = step; // keep only the last position's [vocab]
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

        // Pin the persistent [H] PP-hop scratch with the RDMA transport ONCE
        // (recv side + send side), so vCCL's send/recv skip the per-call
        // `ibv_reg_mr`/dereg temp-MR — the "buffer not registered with the comm"
        // warning and the ~700 ms/tok Kimi PP-3 comm floor. Registration is
        // gated by `VLLM_VULKAN_REG_REDUCE` (same lever as the TP reduce scratch)
        // and libvccl exposing `vcclCommRegister`; on failure we fall back to the
        // fresh-Vec recv_f32/send_f32 path (correct, just per-call regMr).
        let want_reg = self.flags.reg_reduce
            && !comm.is_null()
            && vccl_ffi::registration_available();
        if want_reg {
            let km = self.kimi.as_mut().unwrap();
            if do_recv && km.pp_recv_handle == 0 {
                km.pp_recv_scratch = vec![0.0f32; h];
                let addr = km.pp_recv_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => km.pp_recv_handle = hd,
                    Err(e) => {
                        log::warn!("pp_step_kimi: register recv scratch failed: {e}; per-call regMr");
                        km.pp_recv_scratch.clear();
                    }
                }
            }
            if !is_last && km.pp_send_handle == 0 {
                km.pp_send_scratch = vec![0.0f32; h];
                let addr = km.pp_send_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => km.pp_send_handle = hd,
                    Err(e) => {
                        log::warn!("pp_step_kimi: register send scratch failed: {e}; per-call regMr");
                        km.pp_send_scratch.clear();
                    }
                }
            }
        }

        // 1) recv the previous stage's hidden INTO the registered scratch (fast
        //    pre-pinned MR), or empty on the first stage (it embeds token_id).
        let hidden_in: Vec<f32> = if do_recv {
            let km = self.kimi.as_mut().unwrap();
            if km.pp_recv_handle != 0 {
                vccl_ffi::recv_f32_into(py, comm, &mut km.pp_recv_scratch, recv_from)
                    .map_err(PyRuntimeError::new_err)?;
                km.pp_recv_scratch.clone()
            } else {
                vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
            }
        } else {
            Vec::new()
        };

        let out = self.kimi.as_mut().unwrap().forward_pp_stage(token_id, &hidden_in, pos);

        if !is_last {
            // 3) send onward FROM the registered scratch (mid/first stage out is
            //    [H]); fall back to a fresh-Vec send if registration is off.
            let km = self.kimi.as_mut().unwrap();
            if km.pp_send_handle != 0 && out.len() == km.pp_send_scratch.len() {
                km.pp_send_scratch.copy_from_slice(&out);
                vccl_ffi::send_f32(py, comm, &km.pp_send_scratch, send_to)
                    .map_err(PyRuntimeError::new_err)?;
            } else {
                vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            }
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
    /// scratch (`pp_recv/send_scratch`, `comm_register`'d) exactly as
    /// `pp_step_kimi`; the `[vocab]` ring-back uses plain `send_f32`/`recv_f32` (a
    /// registered vocab scratch, as in `pp_step_laguna_logits`, is a later perf
    /// lever). Bit-exact with `pp_step_kimi`'s last-stage logits. Requires
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

        // Pin the persistent [H] PP-hop scratch ONCE (recv + send), mirroring
        // `pp_step_kimi`; on failure fall back to the fresh-Vec recv/send path.
        let want_reg = self.flags.reg_reduce
            && !comm.is_null()
            && vccl_ffi::registration_available();
        if want_reg {
            let km = self.kimi.as_mut().unwrap();
            if do_recv && km.pp_recv_handle == 0 {
                km.pp_recv_scratch = vec![0.0f32; h];
                let addr = km.pp_recv_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => km.pp_recv_handle = hd,
                    Err(e) => { log::warn!("pp_step_kimi_logits: register recv scratch failed: {e}; per-call regMr"); km.pp_recv_scratch.clear(); }
                }
            }
            if !is_last && km.pp_send_handle == 0 {
                km.pp_send_scratch = vec![0.0f32; h];
                let addr = km.pp_send_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => km.pp_send_handle = hd,
                    Err(e) => { log::warn!("pp_step_kimi_logits: register send scratch failed: {e}; per-call regMr"); km.pp_send_scratch.clear(); }
                }
            }
        }

        // 1) recv the previous stage's [H] hidden INTO the registered scratch, or
        //    empty on the first stage (it embeds token_id).
        let hidden_in: Vec<f32> = if do_recv {
            let km = self.kimi.as_mut().unwrap();
            if km.pp_recv_handle != 0 {
                vccl_ffi::recv_f32_into(py, comm, &mut km.pp_recv_scratch, recv_from)
                    .map_err(PyRuntimeError::new_err)?;
                km.pp_recv_scratch.clone()
            } else {
                vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
            }
        } else {
            Vec::new()
        };

        // 2) resident stage forward (Kimi ignores pos → internal tracking). [H]
        //    on mid stages, [vocab] on the last.
        let out = self.kimi.as_mut().unwrap().forward_pp_stage(token_id, &hidden_in, 0);

        // 3) route the result.
        if is_first && is_last {
            return Ok(Some(out)); // STANDALONE N=1: `out` is already [vocab].
        }
        if !is_last {
            // FIRST / MID: forward `[H]` onward FROM the registered scratch, then
            // (rank0 only) recv the ring-back.
            let km = self.kimi.as_mut().unwrap();
            if km.pp_send_handle != 0 && out.len() == km.pp_send_scratch.len() {
                km.pp_send_scratch.copy_from_slice(&out);
                vccl_ffi::send_f32(py, comm, &km.pp_send_scratch, send_to)
                    .map_err(PyRuntimeError::new_err)?;
            } else {
                vccl_ffi::send_f32(py, comm, &out, send_to).map_err(PyRuntimeError::new_err)?;
            }
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
