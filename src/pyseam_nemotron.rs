// SPDX-License-Identifier: Apache-2.0
//! Per-model pyo3 seam for `nemotron` — moved verbatim out of the monolithic
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

    /// Pipeline-parallel forward for one Nemotron-H-Puzzle decode token
    /// (cross-stage building block). First stage: pass the token id (`hidden_in`
    /// ignored — pass `[]`). Mid/last stages: pass the previous stage's hidden
    /// state. Returns the `hidden_size` hidden to send onward, or the full
    /// `[vocab_size]` logits on the last stage. The per-layer Mamba
    /// (conv_state + ssm_state) and NoPE-attention KV state is resident per
    /// stage and advances in place across decode steps (NOT carried in the
    /// message) — see `NemotronModel::forward_pp_stage`.
    fn forward_pp_nemotron(&mut self, token_id: u32, hidden_in: Vec<f32>, pos: usize) -> PyResult<Vec<f32>> {
        let m = self.nemotron.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("forward_pp_nemotron needs a nemotron_h_puzzle model"))?;
        Ok(m.forward_pp_stage(token_id, &hidden_in, pos))
    }


    /// DISTRIBUTED-SERVE cache-populating prefill for Nemotron-H-Puzzle (the
    /// companion to `pp_step_nemotron_logits`, resolved by `serve_head.py` as
    /// `forward_pp_nemotron_prefill`). Streams the whole prompt through this PP
    /// stage, advancing the resident Mamba (conv_state + ssm_state) and
    /// NoPE-attention KV IN PLACE and filling the attention `seq_len` to the
    /// prompt length, so `pp_step_nemotron_logits` resumes at
    /// `current_decode_pos() == seq` (the launcher does NOT pass `pos`).
    ///
    /// Like qwen3.6 (and unlike Laguna's separate batched prefill kernel),
    /// Nemotron has no batched prefill: prefill is a teacher-forced loop over the
    /// SAME single-token `NemotronModel::forward_pp_stage` the decode seam runs, so
    /// prefill and decode populate identical resident state by construction (no
    /// fold flag to honor). Each position advances every resident layer's
    /// recurrent/KV state by one.
    ///
    ///  - FIRST stage (`pp_start == 0`): `tokens` = full prompt `[seq]`. Returns
    ///    `[seq*H]` (all positions' hidden). If ALSO last (NR==1): the LAST
    ///    position's `[vocab]`.
    ///  - MID stage: `hidden_in` = `[seq*H]`; returns `[seq*H]`.
    ///  - LAST stage (`pp_end >= num_hidden_layers`): `hidden_in` = `[seq*H]`;
    ///    returns the LAST position's `[vocab]` logits.
    fn forward_pp_nemotron_prefill(
        &mut self,
        tokens: Vec<u32>,
        hidden_in: Vec<f32>,
        seq: usize,
    ) -> PyResult<Vec<f32>> {
        let (h, first, last) = {
            let m = self.nemotron.as_ref().ok_or_else(|| {
                PyRuntimeError::new_err("forward_pp_nemotron_prefill needs a nemotron_h_puzzle model")
            })?;
            (m.config.hidden_size, m.pp_start == 0, m.pp_end >= m.config.num_hidden_layers)
        };
        if seq == 0 {
            return Err(PyRuntimeError::new_err("forward_pp_nemotron_prefill: empty prompt"));
        }
        if !first && hidden_in.len() != seq * h {
            return Err(PyRuntimeError::new_err(format!(
                "forward_pp_nemotron_prefill: hidden_in.len()={} != seq*H={}",
                hidden_in.len(), seq * h)));
        }
        let m = self.nemotron.as_mut().unwrap();
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


    /// Fused native-vCCL PP step for Nemotron (mirrors `pp_step_qwen35`): recv
    /// the previous stage's hidden (if not first) → resident stage forward →
    /// send onward (native, no PyList) OR Rust argmax on the last stage. Only
    /// the token id in and `(argmax_token, logit)` out cross the pyo3 boundary
    /// on the last stage. `recv_from < 0` ⇒ first stage (embeds `token_id`);
    /// `send_to < 0` ⇒ last stage (returns `Some((tok, logit))`). Requires
    /// `set_collective_comm` + `VLLM_VULKAN_NATIVE_COMM!=0`.
    fn pp_step_nemotron(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        pos: usize,
        recv_from: i32,
        send_to: i32,
    ) -> PyResult<Option<(u32, f32)>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_nemotron: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let h = self.nemotron.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("pp_step_nemotron needs a nemotron_h_puzzle model"))?
            .config.hidden_size;
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);

        // Pin the persistent [H] PP-hop scratch (recv + send sides) with the RDMA
        // transport ONCE, so vCCL's send/recv skip the per-call `ibv_reg_mr`/dereg
        // temp-MR — the "buffer (H*4 B) not registered with the comm" warning.
        // Mirrors `pp_step_kimi`; gated by `VLLM_VULKAN_REG_REDUCE` and libvccl
        // exposing `vcclCommRegister`; on failure we fall back to the fresh-Vec
        // recv_f32/send_f32 path (correct, just per-call regMr).
        let want_reg = self.flags.reg_reduce
            && !comm.is_null()
            && vccl_ffi::registration_available();
        if want_reg {
            let nem = self.nemotron.as_mut().unwrap();
            if do_recv && nem.pp_recv_handle == 0 {
                nem.pp_recv_scratch = vec![0.0f32; h];
                let addr = nem.pp_recv_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => nem.pp_recv_handle = hd,
                    Err(e) => {
                        log::warn!("pp_step_nemotron: register recv scratch failed: {e}; per-call regMr");
                        nem.pp_recv_scratch.clear();
                    }
                }
            }
            if !is_last && nem.pp_send_handle == 0 {
                nem.pp_send_scratch = vec![0.0f32; h];
                let addr = nem.pp_send_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => nem.pp_send_handle = hd,
                    Err(e) => {
                        log::warn!("pp_step_nemotron: register send scratch failed: {e}; per-call regMr");
                        nem.pp_send_scratch.clear();
                    }
                }
            }
        }

        // 1) recv the previous stage's hidden INTO the registered scratch (fast
        //    pre-pinned MR), or empty on the first stage (it embeds token_id).
        let hidden_in: Vec<f32> = if do_recv {
            let nem = self.nemotron.as_mut().unwrap();
            if nem.pp_recv_handle != 0 {
                vccl_ffi::recv_f32_into(py, comm, &mut nem.pp_recv_scratch, recv_from)
                    .map_err(PyRuntimeError::new_err)?;
                nem.pp_recv_scratch.clone()
            } else {
                vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
            }
        } else {
            Vec::new()
        };

        let out = self.nemotron.as_mut().unwrap().forward_pp_stage(token_id, &hidden_in, pos);

        if !is_last {
            // send onward FROM the registered scratch; fall back to a fresh-Vec
            // send if registration is off or the width doesn't match.
            let nem = self.nemotron.as_mut().unwrap();
            if nem.pp_send_handle != 0 && out.len() == nem.pp_send_scratch.len() {
                nem.pp_send_scratch.copy_from_slice(&out);
                vccl_ffi::send_f32(py, comm, &nem.pp_send_scratch, send_to)
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


    /// DISTRIBUTED-SERVE twin of `pp_step_nemotron` (mirrors `pp_step_laguna_logits`):
    /// the last stage rings the FULL `[vocab]` logits back to rank0 (raw f32 over
    /// vCCL, NO `Vec<f32>→PyList` marshal) instead of argmaxing, so vLLM's Sampler
    /// on rank0 sees the whole distribution. This is the `pp_step_nemotron_logits`
    /// seam `scripts/serve_dist.py` resolves for `--model-type nemotron`.
    ///
    /// The launcher calls it pos-free — `(token_id, recv_from, send_to, last_rank)`
    /// — so the decode `pos` is derived from the resident NoPE-attention KV
    /// `seq_len` (`NemotronModel::current_decode_pos`), which prefill fills to the
    /// prompt length and each decode step advances by one. Reuses the pre-pinned
    /// `[H]` hidden scratch (`pp_recv/send_scratch`, `comm_register`'d) exactly
    /// as `pp_step_nemotron`; the `[vocab]` ring-back uses plain `send_f32`/
    /// `recv_f32` (a registered vocab scratch, as in `pp_step_laguna_logits`, is a
    /// later perf lever). Bit-exact with `pp_step_nemotron`'s last-stage logits.
    /// Requires `set_collective_comm` + `VLLM_VULKAN_NATIVE_COMM!=0`.
    fn pp_step_nemotron_logits(
        &mut self,
        py: Python<'_>,
        token_id: u32,
        recv_from: i32,
        send_to: i32,
        last_rank: i32,
    ) -> PyResult<Option<Vec<f32>>> {
        if !self.native_comm_enabled() {
            return Err(PyRuntimeError::new_err(
                "pp_step_nemotron_logits: native comm not enabled (set_collective_comm + VLLM_VULKAN_NATIVE_COMM!=0)"));
        }
        let (h, vocab, pos) = {
            let m = self.nemotron.as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("pp_step_nemotron_logits needs a nemotron_h_puzzle model"))?;
            (m.config.hidden_size, m.config.vocab_size, m.current_decode_pos())
        };
        let comm = self.collective_comm as *mut std::os::raw::c_void;
        let (do_recv, is_last) = pp_step_role(recv_from, send_to);
        let is_first = recv_from < 0;

        // Pin the persistent [H] PP-hop scratch ONCE (recv + send), mirroring
        // `pp_step_nemotron`; on failure fall back to the fresh-Vec recv/send path.
        let want_reg = self.flags.reg_reduce
            && !comm.is_null()
            && vccl_ffi::registration_available();
        if want_reg {
            let nem = self.nemotron.as_mut().unwrap();
            if do_recv && nem.pp_recv_handle == 0 {
                nem.pp_recv_scratch = vec![0.0f32; h];
                let addr = nem.pp_recv_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => nem.pp_recv_handle = hd,
                    Err(e) => { log::warn!("pp_step_nemotron_logits: register recv scratch failed: {e}; per-call regMr"); nem.pp_recv_scratch.clear(); }
                }
            }
            if !is_last && nem.pp_send_handle == 0 {
                nem.pp_send_scratch = vec![0.0f32; h];
                let addr = nem.pp_send_scratch.as_ptr() as usize;
                match vccl_ffi::comm_register(comm, addr, h * std::mem::size_of::<f32>()) {
                    Ok(hd) => nem.pp_send_handle = hd,
                    Err(e) => { log::warn!("pp_step_nemotron_logits: register send scratch failed: {e}; per-call regMr"); nem.pp_send_scratch.clear(); }
                }
            }
        }

        // 1) recv the previous stage's [H] hidden INTO the registered scratch, or
        //    empty on the first stage (it embeds token_id).
        let hidden_in: Vec<f32> = if do_recv {
            let nem = self.nemotron.as_mut().unwrap();
            if nem.pp_recv_handle != 0 {
                vccl_ffi::recv_f32_into(py, comm, &mut nem.pp_recv_scratch, recv_from)
                    .map_err(PyRuntimeError::new_err)?;
                nem.pp_recv_scratch.clone()
            } else {
                vccl_ffi::recv_f32(py, comm, h, recv_from).map_err(PyRuntimeError::new_err)?
            }
        } else {
            Vec::new()
        };

        // 2) resident stage forward. [H] on mid stages, [vocab] on the last.
        let out = self.nemotron.as_mut().unwrap().forward_pp_stage(token_id, &hidden_in, pos);

        // 3) route the result.
        if is_first && is_last {
            return Ok(Some(out)); // STANDALONE N=1: `out` is already [vocab].
        }
        if !is_last {
            // FIRST / MID: forward `[H]` onward FROM the registered scratch, then
            // (rank0 only) recv the ring-back.
            let nem = self.nemotron.as_mut().unwrap();
            if nem.pp_send_handle != 0 && out.len() == nem.pp_send_scratch.len() {
                nem.pp_send_scratch.copy_from_slice(&out);
                vccl_ffi::send_f32(py, comm, &nem.pp_send_scratch, send_to)
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


    /// Phase-1 MTP acceptance sim: draft `depth` tokens (greedy/argmax
    /// throughout) off the LAST stage's most recent decode step. `token_next`
    /// is the base model's own just-committed greedy token from THAT SAME
    /// step (the caller already has it — every PP driver argmaxes the last
    /// stage's logits to pick the next fed token, native or not — so it is
    /// passed explicitly rather than re-derived here). Resets the head's own
    /// KV first (Phase 1 measures PER-CYCLE acceptance; see `nemotron_mtp`
    /// module doc on why continuous multi-cycle refill is out of scope).
    /// Requires `VLLM_VULKAN_NEMOTRON_MTP=1` at load AND this rank to be the
    /// last PP stage (where the head + `lm_head` were loaded).
    fn nem_mtp_draft(&mut self, token_next: u32, depth: usize) -> PyResult<Vec<u32>> {
        let hidden_pre = {
            let m = self.nemotron.as_ref().ok_or_else(|| PyRuntimeError::new_err(
                "nem_mtp_draft needs a nemotron_h_puzzle model"))?;
            if m.last_pre_norm_hidden.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "nem_mtp_draft: no cached hidden yet (call forward_pp_nemotron/pp_step_nemotron first)"));
            }
            m.last_pre_norm_hidden.clone()
        };
        self.nem_mtp_draft_impl(&hidden_pre, token_next, depth)
    }


    /// Phase-1 DECOUPLED mode: like `nem_mtp_draft`, but `hidden` is supplied
    /// explicitly (a `h_pre` record replayed from a `VLLM_VULKAN_NEMOTRON_MTP_TRACE`
    /// dump) instead of read from the last live `forward_pp_stage` call. Lets
    /// a single-node, head-only process (no base-model layers loaded at all —
    /// construct with `layer_start=layer_end=num_hidden_layers` so only
    /// `lm_head`/`backbone.norm_f` load, see `nemotron_mtp` module doc) replay
    /// an entire trace offline: no PP topology, no ~7GB co-residency OOM risk
    /// on the base model's last stage. Requires `VLLM_VULKAN_NEMOTRON_MTP=1`
    /// at construction (loads the head + `lm_head`, same as the inline path).
    fn nem_mtp_draft_from_hidden(&mut self, hidden: Vec<f32>, token_next: u32, depth: usize) -> PyResult<Vec<u32>> {
        if self.nemotron.is_none() {
            return Err(PyRuntimeError::new_err(
                "nem_mtp_draft_from_hidden needs a nemotron_h_puzzle model (load with \
                 layer_start=layer_end=num_hidden_layers for the offline head-only mode)"));
        }
        self.nem_mtp_draft_impl(&hidden, token_next, depth)
    }


}


impl VulkanModel {

    /// Shared body for `nem_mtp_draft`/`nem_mtp_draft_from_hidden`: draft
    /// `depth` tokens off `hidden_pre` (greedy/argmax throughout). Gathers a
    /// RAW pointer to the shared lm_head weight (2.15GB, `[vocab,h]` f32 —
    /// NEVER cloned) BEFORE the mutable borrow of `nemotron_mtp_head` below,
    /// the same disjoint-borrow dance `nemotron.rs`'s `NemMvKind` gathering
    /// already uses in this codebase. Sound because `self.nemotron` is not
    /// mutated or dropped for the lifetime of this call. Not a `#[pymethods]`
    /// item (takes `&[f32]`, not a PyO3-representable borrow) — kept in this
    /// plain `impl` block deliberately.
    fn nem_mtp_draft_impl(&mut self, hidden_pre: &[f32], token_next: u32, depth: usize) -> PyResult<Vec<u32>> {
        let (lm_ptr, lm_len) = {
            let m = self.nemotron.as_ref().ok_or_else(|| PyRuntimeError::new_err(
                "nem_mtp_draft needs a nemotron_h_puzzle model"))?;
            let w = m.weights.f32_slice(&m.lm_head_name);
            (w.as_ptr(), w.len())
        };
        let lm_head_weight: &[f32] = unsafe { std::slice::from_raw_parts(lm_ptr, lm_len) };
        let head = self.nemotron_mtp_head.as_mut().ok_or_else(|| PyRuntimeError::new_err(
            "nem_mtp_draft: no MTP head loaded (set VLLM_VULKAN_NEMOTRON_MTP=1 on the LAST PP stage)"))?;
        head.reset();
        let first_embed = head.embed(token_next);
        Ok(head.head_chain_cpu(&first_embed, hidden_pre, depth, lm_head_weight))
    }


}
