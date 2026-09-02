// SPDX-License-Identifier: Apache-2.0
//! THE decoder-layer body, recorded exactly once.
//!
//! Before this module the same layer body existed as seven near-copies
//! (`forward_layer_gpu_matmuls`, `gemma_resident_layer`, `gemma_resident_layer_1cb`,
//! `gemma_attn_tp_1cb`, `gemma_mlp_tp_1cb`, `gpu_layer_1cb`, `gpu_layer_2cb`, plus
//! the three `*_tp_full` LEVER-1 pieces). Every blocker found in review so far was
//! the same shape: a fix landed in some copies and was missed in others — a constant
//! `num_kv` against a per-layer `head_dim`; an unconditional `v_proj` index that
//! panicked on value-less global layers; an unconditional `ple: Some(..)`; and
//! `matvec_variant` keyed on the GLOBAL quant flag rather than each weight's own
//! recorded format, which had to be fixed THREE separate times in three copies.
//!
//! ## What is shared and what is not
//!
//! Shared (this module): the DISPATCH SEQUENCE of a layer — which kernels run, in
//! what order, with which push constants, workgroup counts and barriers.
//!
//! NOT shared (each entry point keeps its own): the SUBMISSION STRATEGY. The
//! 1-CB / 2-CB split, the TP sub-block split around the all-reduce, resident-KV
//! append vs host KV append, the PLE tail, and the profiling buckets all exist for
//! measured reasons (fence tax, submit collapse, comm) and are deliberately left
//! at the call sites.
//!
//! ## How it stays honest
//!
//! The core never sees a `compute::Buffer`. It emits `Step`s naming ABSTRACT slots
//! (`Slot::Q`) and abstract weights (`ProjW::V`, `NormW::KNorm`); a `LayerRecorder`
//! resolves those against one concrete arena. `GpuRecorder` resolves to the caller's
//! `UR_*` or `GR_*` buffers and records into a command buffer; `PlanRecorder`
//! (tests) resolves to strings, which is how the dispatch sequence can be gated in
//! CI on a machine with no Vulkan device.

use crate::compute;
use crate::flags::QuantFormat;
use crate::MvKind;
use crate::{ew_mul_pc, ew_unary_pc, rmsnorm_pc, rope_neox_pc};

// ─── Abstract activation slots ───────────────────────────────────────────────

/// One activation buffer of a decoder layer, named by ROLE rather than by arena.
/// `GpuRecorder` maps these onto the caller's own slot set (`UR_*` for the unified
/// engine, `GR_*` for the gemma resident/TP engine — e.g. both `UR_ACT` and
/// `GR_GELU` are `Slot::Act`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Slot {
    Ha = 0,
    Hb = 1,
    X = 2,
    Q = 3,
    K = 4,
    V = 5,
    Attn = 6,
    O = 7,
    On = 8,
    Ffin = 9,
    Gate = 10,
    Up = 11,
    Act = 12,
    Mid = 13,
    Down = 14,
    Downn = 15,
    Pos = 16,
    Ff = 17,
    Idx = 18,
    Dummy = 19,
}
pub(crate) const SLOT_COUNT: usize = 20;

/// One RMS-norm WEIGHT of a decoder layer. `FfnIn` is the norm applied to the
/// post-attention residual before the FFN — qwen calls that tensor
/// `post_attention_layernorm` (PRE-norm) and gemma calls it
/// `pre_feedforward_layernorm` (SANDWICH); the ROLE is the same, so it is one
/// entry here and the caller supplies whichever tensor its arch names.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NormW {
    InputLn = 0,
    QNorm = 1,
    KNorm = 2,
    FfnIn = 3,
    PostAttn = 4,
    PostFfn = 5,
}
pub(crate) const NORM_COUNT: usize = 6;

/// One projection WEIGHT of a decoder layer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ProjW {
    Q = 0,
    K = 1,
    V = 2,
    O = 3,
    Gate = 4,
    Up = 5,
    Down = 6,
}
pub(crate) const PROJ_COUNT: usize = 7;

impl ProjW {
    /// Tensor suffix under `model.layers.{i}.`, in `ProjW` order.
    pub(crate) const SUFFIXES: [&'static str; PROJ_COUNT] = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];
    pub(crate) fn suffix(self) -> &'static str { Self::SUFFIXES[self as usize] }
}

// ─── The step vocabulary ─────────────────────────────────────────────────────

/// One recorded operation of a decoder layer.
///
/// Deliberately COARSE: each variant carries the layer-level intent (`Rms` over
/// `n` columns in `groups` head-sized groups) and the recorder owns the push
/// constants and workgroup arithmetic. That is the point — the `(elem + 511)/512`
/// for the activation vs `(elem + 255)/256` for the multiply, and the no-weight
/// rms-norm's dummy binding-1, each exist in exactly one place now.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Step {
    /// Matvec through `record_gemma_mv` — dispatch chosen from the WEIGHT'S OWN
    /// recorded format/aux, never the global quant snapshot.
    Mv { w: ProjW, inp: Slot, out: Slot, k: usize, n: usize },
    /// RMS norm over `groups` contiguous `n`-column groups. `w: None` is the
    /// NO-WEIGHT form (`rms_norm_f32`, binding 1 an ignored dummy).
    Rms { src: Slot, w: Option<NormW>, dst: Slot, n: usize, eps: f32, groups: u32 },
    /// NeoX RoPE, in place.
    Rope { buf: Slot, heads: usize, head_dim: usize, rotary_dim: usize, freq_dim: usize, theta: f32 },
    /// FFN activation (`silu_f32` / `gelu_f32`).
    Unary { shader: &'static str, src: Slot, dst: Slot, elems: usize },
    /// Elementwise multiply.
    Mul { a: Slot, b: Slot, dst: Slot, elems: usize },
    /// Residual add (3 inputs + the harmless binding-3 dummy).
    Add { a: Slot, b: Slot, dst: Slot, elems: usize },
    /// Full pipeline barrier.
    Barrier,
}

/// Consumes the layer body's steps. Implemented by `GpuRecorder` (records into a
/// command buffer) and by the test-only `PlanRecorder` (collects strings).
pub(crate) trait LayerRecorder {
    fn step(&mut self, s: Step) -> Result<(), String>;
}

// ─── Parameters ──────────────────────────────────────────────────────────────

/// Everything the ATTENTION FRONT (input norm → q/k/v → q/k/v-norm → RoPE) needs.
///
/// `num_q` / `q_dim` are THIS RANK's counts: a TP caller passes `r_num_q` /
/// `r_q_dim` and K/V stay replicated at the full `num_kv` / `kv_dim`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrontParams {
    /// Record `input_layernorm(HA) → X` as the first dispatch. False for
    /// `gemma_attn_tp_1cb`, whose contract is that the caller already normed `x`
    /// on the CPU and wrote it into the X slot.
    pub(crate) input_norm: bool,
    pub(crate) hidden: usize,
    pub(crate) head_dim: usize,
    pub(crate) num_q: usize,
    pub(crate) num_kv: usize,
    pub(crate) q_dim: usize,
    pub(crate) kv_dim: usize,
    pub(crate) eps: f32,
    /// Weighted k-norm present (gemma: `!is_kv_shared`; qwen: always).
    pub(crate) k_norm: bool,
    /// Weightless v-norm present (gemma: `!is_kv_shared`; qwen: never).
    pub(crate) v_norm: bool,
    /// Value-less global attention: no `v_proj` tensor on disk at all; V is the
    /// weightless norm of the RAW (pre-k_norm) K.
    pub(crate) uses_k_eq_v: bool,
    pub(crate) rotary_dim: usize,
    pub(crate) freq_dim: usize,
    pub(crate) theta: f32,
}

/// Everything the layer TAIL (o_proj → residual → FFN → residual) needs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TailParams {
    pub(crate) hidden: usize,
    /// o_proj's INPUT width — this rank's `num_q * head_dim`.
    pub(crate) o_in_dim: usize,
    /// FFN width — this rank's `intermediate_size`.
    pub(crate) inter: usize,
    pub(crate) eps: f32,
    /// Gemma SANDWICH norms (post_attn / post_ffn applied to the sublayer OUTPUT
    /// before the residual add). False = qwen PRE-norm.
    pub(crate) sandwich: bool,
    /// `silu_f32` (qwen) or `gelu_f32` (gemma).
    pub(crate) act_shader: &'static str,
}

// ─── The body ────────────────────────────────────────────────────────────────

/// ATTENTION FRONT: `[input_layernorm →] q/k/v → q/k/v-norm → RoPE`.
///
/// Emits no leading and no trailing barrier — the caller composes.
///
/// THE VALUE-LESS CASE. When `uses_k_eq_v` there is no `v_proj` tensor at all, so
/// no V matvec is emitted; instead V is derived as the weightless norm of the RAW
/// K, recorded BEFORE the weighted k-norm overwrites K in place, with a barrier
/// between them (the CPU reference's `v_raw = k_raw.clone()` taken before k_norm —
/// `model.rs::forward_layer`). Getting this wrong is silent: V is simply the wrong
/// tensor and the logits drift.
pub(crate) fn record_front<R: LayerRecorder>(r: &mut R, p: &FrontParams) -> Result<(), String> {
    if p.input_norm {
        r.step(Step::Rms { src: Slot::Ha, w: Some(NormW::InputLn), dst: Slot::X,
                           n: p.hidden, eps: p.eps, groups: 1 })?;
        r.step(Step::Barrier)?;
    }
    r.step(Step::Mv { w: ProjW::Q, inp: Slot::X, out: Slot::Q, k: p.hidden, n: p.q_dim })?;
    r.step(Step::Mv { w: ProjW::K, inp: Slot::X, out: Slot::K, k: p.hidden, n: p.kv_dim })?;
    if !p.uses_k_eq_v {
        r.step(Step::Mv { w: ProjW::V, inp: Slot::X, out: Slot::V, k: p.hidden, n: p.kv_dim })?;
    }
    r.step(Step::Barrier)?;
    let g_q = p.num_q as u32;
    let g_kv = p.num_kv as u32;
    r.step(Step::Rms { src: Slot::Q, w: Some(NormW::QNorm), dst: Slot::Q,
                       n: p.head_dim, eps: p.eps, groups: g_q })?;
    if p.uses_k_eq_v {
        r.step(Step::Rms { src: Slot::K, w: None, dst: Slot::V,
                           n: p.head_dim, eps: p.eps, groups: g_kv })?;
        r.step(Step::Barrier)?;
        if p.k_norm {
            r.step(Step::Rms { src: Slot::K, w: Some(NormW::KNorm), dst: Slot::K,
                               n: p.head_dim, eps: p.eps, groups: g_kv })?;
        }
    } else {
        if p.k_norm {
            r.step(Step::Rms { src: Slot::K, w: Some(NormW::KNorm), dst: Slot::K,
                               n: p.head_dim, eps: p.eps, groups: g_kv })?;
        }
        if p.v_norm {
            r.step(Step::Rms { src: Slot::V, w: None, dst: Slot::V,
                               n: p.head_dim, eps: p.eps, groups: g_kv })?;
        }
    }
    r.step(Step::Barrier)?;
    // V is NEVER RoPE'd (value-less or not).
    r.step(Step::Rope { buf: Slot::Q, heads: p.num_q, head_dim: p.head_dim,
                        rotary_dim: p.rotary_dim, freq_dim: p.freq_dim, theta: p.theta })?;
    r.step(Step::Rope { buf: Slot::K, heads: p.num_kv, head_dim: p.head_dim,
                        rotary_dim: p.rotary_dim, freq_dim: p.freq_dim, theta: p.theta })?;
    Ok(())
}

/// TAIL piece 1 — `o_proj: ATTN → O`.
pub(crate) fn record_o_proj<R: LayerRecorder>(r: &mut R, p: &TailParams) -> Result<(), String> {
    r.step(Step::Mv { w: ProjW::O, inp: Slot::Attn, out: Slot::O, k: p.o_in_dim, n: p.hidden })
}

/// TAIL piece 2 — post-attention residual. SANDWICH: `post_attn_norm(O) → ON`,
/// then `HB = HA + ON`. PRE-norm: `HB = HA + O`.
pub(crate) fn record_attn_residual<R: LayerRecorder>(r: &mut R, p: &TailParams) -> Result<(), String> {
    let src = if p.sandwich {
        r.step(Step::Rms { src: Slot::O, w: Some(NormW::PostAttn), dst: Slot::On,
                           n: p.hidden, eps: p.eps, groups: 1 })?;
        r.step(Step::Barrier)?;
        Slot::On
    } else {
        Slot::O
    };
    r.step(Step::Add { a: Slot::Ha, b: src, dst: Slot::Hb, elems: p.hidden })
}

/// TAIL piece 3a — the FFN-input norm `HB → FFIN` (qwen `post_attention_layernorm`,
/// gemma `pre_feedforward_layernorm`).
pub(crate) fn record_ffn_in_norm<R: LayerRecorder>(r: &mut R, p: &TailParams) -> Result<(), String> {
    r.step(Step::Rms { src: Slot::Hb, w: Some(NormW::FfnIn), dst: Slot::Ffin,
                       n: p.hidden, eps: p.eps, groups: 1 })
}

/// TAIL piece 3b — `gate/up → act(gate) → act*up → down`.
pub(crate) fn record_ffn_body<R: LayerRecorder>(r: &mut R, p: &TailParams) -> Result<(), String> {
    r.step(Step::Mv { w: ProjW::Gate, inp: Slot::Ffin, out: Slot::Gate, k: p.hidden, n: p.inter })?;
    r.step(Step::Mv { w: ProjW::Up, inp: Slot::Ffin, out: Slot::Up, k: p.hidden, n: p.inter })?;
    r.step(Step::Barrier)?;
    r.step(Step::Unary { shader: p.act_shader, src: Slot::Gate, dst: Slot::Act, elems: p.inter })?;
    r.step(Step::Barrier)?;
    r.step(Step::Mul { a: Slot::Act, b: Slot::Up, dst: Slot::Mid, elems: p.inter })?;
    r.step(Step::Barrier)?;
    r.step(Step::Mv { w: ProjW::Down, inp: Slot::Mid, out: Slot::Down, k: p.inter, n: p.hidden })
}

/// TAIL piece 4 — post-FFN residual. SANDWICH: `post_ffn_norm(DOWN) → DOWNN`,
/// then `HA = HB + DOWNN`. PRE-norm: `HA = HB + DOWN`.
pub(crate) fn record_ffn_residual<R: LayerRecorder>(r: &mut R, p: &TailParams) -> Result<(), String> {
    let src = if p.sandwich {
        r.step(Step::Rms { src: Slot::Down, w: Some(NormW::PostFfn), dst: Slot::Downn,
                           n: p.hidden, eps: p.eps, groups: 1 })?;
        r.step(Step::Barrier)?;
        Slot::Downn
    } else {
        Slot::Down
    };
    r.step(Step::Add { a: Slot::Hb, b: src, dst: Slot::Ha, elems: p.hidden })
}

/// The WHOLE tail, pieces 1–4 with the barriers between them: `o_proj → residual
/// → ffn_in_norm → FFN → residual2`. Hidden leaves in HA.
pub(crate) fn record_tail<R: LayerRecorder>(r: &mut R, p: &TailParams) -> Result<(), String> {
    record_o_proj(r, p)?;
    r.step(Step::Barrier)?;
    record_attn_residual(r, p)?;
    r.step(Step::Barrier)?;
    record_ffn_in_norm(r, p)?;
    r.step(Step::Barrier)?;
    record_ffn_body(r, p)?;
    r.step(Step::Barrier)?;
    record_ffn_residual(r, p)
}

// ─── The GPU recorder ────────────────────────────────────────────────────────

/// A projection weight's dispatch descriptor (`gemma_res_mv_kind`'s output plus
/// the buffer pointer). `MvKind` is `Copy` raw pointers gathered BEFORE
/// `engine.as_mut()` so the `gpu_weights` borrow ends first.
#[derive(Clone, Copy)]
pub(crate) struct MvW {
    pub(crate) ptr: *const compute::Buffer,
    pub(crate) format: QuantFormat,
    pub(crate) kind: MvKind,
}

/// The concrete arena one recording resolves against: activation slots, norm
/// weights, projection weights. Entries the layer does not use stay null/None —
/// e.g. `projs[ProjW::V]` is `None` exactly on a value-less layer, and the
/// two-command-buffer callers fill only the half each CB touches.
pub(crate) struct LayerPtrs {
    pub(crate) bufs: [*const compute::Buffer; SLOT_COUNT],
    pub(crate) norms: [*const compute::Buffer; NORM_COUNT],
    pub(crate) projs: [Option<MvW>; PROJ_COUNT],
}

impl Default for LayerPtrs {
    fn default() -> Self {
        LayerPtrs {
            bufs: [std::ptr::null(); SLOT_COUNT],
            norms: [std::ptr::null(); NORM_COUNT],
            projs: [None; PROJ_COUNT],
        }
    }
}

impl LayerPtrs {
    pub(crate) fn set_norm(&mut self, n: NormW, p: *const compute::Buffer) { self.norms[n as usize] = p; }
}

/// Records the layer body into one command buffer.
///
/// Every `Step` maps to EXACTLY the dispatch the seven hand-written copies made:
/// same shader name, same binding order, same push constants, same workgroup
/// arithmetic. This function is the whole audit surface for that claim.
pub(crate) struct GpuRecorder<'e, 'p> {
    pub(crate) eng: &'e mut compute::ComputeEngine,
    pub(crate) cb: ash::vk::CommandBuffer,
    pub(crate) p: &'p LayerPtrs,
}

impl GpuRecorder<'_, '_> {
    fn buf(&self, s: Slot) -> *const compute::Buffer {
        let p = self.p.bufs[s as usize];
        assert!(!p.is_null(), "layer_core: activation slot {s:?} was not bound for this recording");
        p
    }
    fn norm(&self, n: NormW) -> *const compute::Buffer {
        let p = self.p.norms[n as usize];
        assert!(!p.is_null(), "layer_core: norm weight {n:?} was not bound for this recording");
        p
    }
    fn proj(&self, w: ProjW) -> MvW {
        self.p.projs[w as usize]
            .unwrap_or_else(|| panic!("layer_core: projection {w:?} was not bound for this recording"))
    }
}

impl LayerRecorder for GpuRecorder<'_, '_> {
    fn step(&mut self, s: Step) -> Result<(), String> {
        match s {
            Step::Mv { w, inp, out, k, n } => {
                let m = self.proj(w);
                let (i, o) = (self.buf(inp), self.buf(out));
                unsafe {
                    crate::gemma_forward::record_gemma_mv(
                        self.eng, self.cb, m.ptr, m.format, m.kind, i, o, k, n);
                }
                Ok(())
            }
            Step::Rms { src, w, dst, n, eps, groups } => {
                let pc = rmsnorm_pc(n, eps);
                let (s_p, d_p) = (self.buf(src), self.buf(dst));
                match w {
                    Some(nw) => {
                        let w_p = self.norm(nw);
                        unsafe {
                            self.eng.record_to(self.cb, "rms_norm_f32_mul",
                                &[&*s_p, &*w_p, &*d_p], &pc, (groups, 1, 1))
                        }
                    }
                    None => {
                        // NO-WEIGHT form: binding 1 is an ignored dummy for the
                        // do_multiply=false path (the FF slot, as in every copy).
                        let f_p = self.buf(Slot::Ff);
                        unsafe {
                            self.eng.record_to(self.cb, "rms_norm_f32",
                                &[&*s_p, &*f_p, &*d_p], &pc, (groups, 1, 1))
                        }
                    }
                }
            }
            Step::Rope { buf, heads, head_dim, rotary_dim, freq_dim, theta } => {
                let pc = rope_neox_pc(heads, head_dim, rotary_dim, freq_dim, theta);
                let wgy = (((head_dim / 2) as u32) + 255) / 256;
                let (b, pos, ff, idx) =
                    (self.buf(buf), self.buf(Slot::Pos), self.buf(Slot::Ff), self.buf(Slot::Idx));
                unsafe {
                    self.eng.record_to(self.cb, "rope_neox_f32_f32",
                        &[&*b, &*pos, &*ff, &*b, &*idx], &pc, (heads as u32, wgy, 1))
                }
            }
            Step::Unary { shader, src, dst, elems } => {
                let pc = ew_unary_pc(elems as u32);
                let (s_p, d_p) = (self.buf(src), self.buf(dst));
                // gelu.comp / silu.comp are local_size_x = 512.
                unsafe {
                    self.eng.record_to(self.cb, shader, &[&*s_p, &*d_p], &pc,
                        ((elems as u32 + 511) / 512, 1, 1))
                }
            }
            Step::Mul { a, b, dst, elems } => {
                let pc = ew_mul_pc(elems as u32);
                let (a_p, b_p, d_p) = (self.buf(a), self.buf(b), self.buf(dst));
                // mul.comp is local_size_x = 256.
                unsafe {
                    self.eng.record_to(self.cb, "mul_f32_f32_f32", &[&*a_p, &*b_p, &*d_p], &pc,
                        ((elems as u32 + 255) / 256, 1, 1))
                }
            }
            Step::Add { a, b, dst, elems } => {
                let pc = ew_mul_pc(elems as u32);
                let (a_p, b_p, d_p, dummy) =
                    (self.buf(a), self.buf(b), self.buf(dst), self.buf(Slot::Dummy));
                unsafe {
                    self.eng.record_to(self.cb, "add_f32_f32_f32",
                        &[&*a_p, &*b_p, &*d_p, &*dummy], &pc, ((elems as u32 + 255) / 256, 1, 1))
                }
            }
            Step::Barrier => { self.eng.record_barrier_to(self.cb); Ok(()) }
        }
    }
}

// ─── The plan recorder (tests) ───────────────────────────────────────────────

/// Collects the layer body as human-readable steps instead of recording it, so
/// the dispatch sequence can be asserted in CI with no Vulkan device present.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct PlanRecorder {
    pub(crate) steps: Vec<String>,
}

#[cfg(test)]
impl LayerRecorder for PlanRecorder {
    fn step(&mut self, s: Step) -> Result<(), String> {
        self.steps.push(match s {
            Step::Mv { w, inp, out, k, n } => format!("mv {w:?} {inp:?}->{out:?} {k}x{n}"),
            Step::Rms { src, w, dst, n, eps: _, groups } => match w {
                Some(nw) => format!("rms_mul {src:?}*{nw:?}->{dst:?} n={n} g={groups}"),
                None => format!("rms {src:?}->{dst:?} n={n} g={groups}"),
            },
            Step::Rope { buf, heads, head_dim, rotary_dim, freq_dim, theta } =>
                format!("rope {buf:?} heads={heads} hd={head_dim} rd={rotary_dim} fd={freq_dim} theta={theta}"),
            Step::Unary { shader, src, dst, elems } => format!("{shader} {src:?}->{dst:?} n={elems}"),
            Step::Mul { a, b, dst, elems } => format!("mul {a:?}*{b:?}->{dst:?} n={elems}"),
            Step::Add { a, b, dst, elems } => format!("add {a:?}+{b:?}->{dst:?} n={elems}"),
            Step::Barrier => "barrier".to_string(),
        });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn plan_front(p: &FrontParams) -> Vec<String> {
    let mut r = PlanRecorder::default();
    record_front(&mut r, p).unwrap();
    r.steps
}

#[cfg(test)]
pub(crate) fn plan_tail(p: &TailParams) -> Vec<String> {
    let mut r = PlanRecorder::default();
    record_tail(&mut r, p).unwrap();
    r.steps
}

// ─── The one place a layer weight becomes a dispatch descriptor ──────────────

impl crate::VulkanModel {
    /// Resolve one projection of `layer_idx` to its dispatch descriptor.
    ///
    /// THE INVARIANT: the shader and its BINDING COUNT come from the weight's OWN
    /// recorded `format`/`aux` (`gemma_res_mv_kind`), never from the process-wide
    /// `VLLM_VULKAN_QUANT` snapshot that `matvec_variant` reads. `gemma4_unified`
    /// uploads attention as Mlx4 and the MLP as Q8_0 UNCONDITIONALLY regardless of
    /// that flag, and the packed formats bind extra buffers the plain matvec does
    /// not (Mlx4 needs scales AND biases), so a global-keyed selector reads 4-bit
    /// nibbles as f16 and yields NaN logits. That defect was found and fixed three
    /// separate times in three copies of the layer body; this is now the only
    /// place it can be fixed, or missed.
    pub(crate) fn layer_proj_mv(&self, layer_idx: usize, w: ProjW) -> MvW {
        let name = format!("model.layers.{layer_idx}.{}", w.suffix());
        let gw = &self.gpu_weights[&name];
        let (format, kind) = crate::gemma_forward::gemma_res_mv_kind(gw);
        MvW { ptr: &gw.buffer as *const compute::Buffer, format, kind }
    }

    /// Resolve the projections in `want` for `layer_idx`.
    ///
    /// `ProjW::V` is SKIPPED when `uses_k_eq_v`, even if the caller asks for it:
    /// value-less global gemma layers carry no `v_proj` tensor on disk at all, and
    /// `self.gpu_weights[..]` is an INDEX, not a lookup — asking for it panics.
    /// That panic was one of the five review blockers, and it was a blocker
    /// precisely because one copy of the body gated the lookup and another did
    /// not. Gating it HERE means a caller cannot get it wrong.
    pub(crate) fn layer_proj_weights(
        &self, layer_idx: usize, uses_k_eq_v: bool, want: &[ProjW],
    ) -> [Option<MvW>; PROJ_COUNT] {
        let mut out = [None; PROJ_COUNT];
        for &w in want {
            if w == ProjW::V && uses_k_eq_v { continue; }
            out[w as usize] = Some(self.layer_proj_mv(layer_idx, w));
        }
        out
    }
}

/// The three projections an attention FRONT records (V dropped when value-less).
pub(crate) const FRONT_PROJS: [ProjW; 3] = [ProjW::Q, ProjW::K, ProjW::V];
/// The four projections a layer TAIL records.
pub(crate) const TAIL_PROJS: [ProjW; 4] = [ProjW::O, ProjW::Gate, ProjW::Up, ProjW::Down];

// ─── Equivalence gates ───────────────────────────────────────────────────────

/// Golden dispatch-plan gates for the shared layer body.
///
/// The expected plans below were transcribed BY HAND from the seven hand-written
/// bodies as they stood at `c4788eb`, before they were replaced by
/// `layer_core::record_front` / `record_tail`. They are the equivalence evidence
/// for that replacement: each `EXPECTED_*` is what `gemma_resident_layer`,
/// `gpu_layer_2cb` &c. recorded, step for step, barrier for barrier.
///
/// They do NOT replace the on-cluster argmax-exact gate — no Vulkan device is
/// involved here. They catch the ONE failure mode this refactor could introduce
/// (a dispatch, an ordering or a barrier that moved) on every `cargo test`.
#[cfg(test)]
mod layer_body_equivalence {
    use super::*;
    use crate::model::{Gemma4Config, Qwen3Config, SynthGemmaSpec};
    use crate::gemma_forward::{gemma_front_params, gemma_tail_params};
    use crate::unified_layer::LayerSpec;

    /// The gate geometry of `tiny_synthetic_gemma`: 4 layers, `attention_period`
    /// 3 (so layer 2 is the global one), `attention_k_eq_v` — i.e. it carries BOTH
    /// a sliding layer and a value-less global layer, which is exactly the split
    /// the blockers lived on.
    fn tiny_cfg() -> Gemma4Config {
        let spec = SynthGemmaSpec::tiny();
        assert!(spec.attention_k_eq_v, "the value-less case must be reachable in this fixture");
        crate::model::tiny_synthetic_gemma(64).config.clone()
    }

    fn tiny_qwen_cfg() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
        }
    }

    fn assert_plan(got: Vec<String>, want: &[&str], what: &str) {
        let want: Vec<String> = want.iter().map(|s| s.to_string()).collect();
        assert_eq!(got, want,
            "{what}: the recorded dispatch plan drifted from the pre-consolidation body");
    }

    // ── FRONT ───────────────────────────────────────────────────────────────

    /// `gemma_resident_layer` / `gemma_resident_layer_1cb` / `gpu_layer_2cb`
    /// CB1 on a SLIDING (non-value-less, non-KV-shared) layer.
    #[test]
    fn front_gemma_sliding_matches_prerefactor() {
        let cfg = tiny_cfg();
        assert!(!cfg.is_full_attention(0) && !cfg.layer_uses_k_eq_v(0));
        let p = gemma_front_params(&cfg, 0, true, cfg.num_attention_heads);
        assert_plan(plan_front(&p), &[
            "rms_mul Ha*InputLn->X n=128 g=1",
            "barrier",
            "mv Q X->Q 128x128",
            "mv K X->K 128x64",
            "mv V X->V 128x64",
            "barrier",
            "rms_mul Q*QNorm->Q n=32 g=4",
            "rms_mul K*KNorm->K n=32 g=2",
            "rms V->V n=32 g=2",
            "barrier",
            "rope Q heads=4 hd=32 rd=32 fd=32 theta=10000",
            "rope K heads=2 hd=32 rd=32 fd=32 theta=10000",
        ], "gemma sliding front");
    }

    /// The VALUE-LESS global layer. Three things must hold, and each was a
    /// blocker when one copy of the body got it wrong:
    ///   - no `v_proj` matvec at all (there is no such tensor on disk),
    ///   - V derived from the RAW K, recorded BEFORE the weighted k-norm
    ///     overwrites K, with a barrier between them,
    ///   - the per-layer geometry (global head dim 64, PARTIAL rotary 64/4 = 16
    ///     with the frequency basis still the full 64, theta 1e6).
    #[test]
    fn front_gemma_valueless_global_matches_prerefactor() {
        let cfg = tiny_cfg();
        assert!(cfg.layer_uses_k_eq_v(2), "layer 2 must be the value-less global layer");
        let p = gemma_front_params(&cfg, 2, true, cfg.num_attention_heads);
        assert_plan(plan_front(&p), &[
            "rms_mul Ha*InputLn->X n=128 g=1",
            "barrier",
            "mv Q X->Q 128x256",
            "mv K X->K 128x128",
            "barrier",
            "rms_mul Q*QNorm->Q n=64 g=4",
            "rms K->V n=64 g=2",
            "barrier",
            "rms_mul K*KNorm->K n=64 g=2",
            "barrier",
            "rope Q heads=4 hd=64 rd=16 fd=64 theta=1000000",
            "rope K heads=2 hd=64 rd=16 fd=64 theta=1000000",
        ], "gemma value-less global front");
    }

    /// The value-less V must come from the RAW K. If the weighted k-norm were
    /// recorded first, V would be the NORMED K — silently wrong logits, no error.
    #[test]
    fn valueless_v_is_derived_before_k_norm() {
        let cfg = tiny_cfg();
        let plan = plan_front(&gemma_front_params(&cfg, 2, true, cfg.num_attention_heads));
        let v_derive = plan.iter().position(|s| s == "rms K->V n=64 g=2").expect("V derive missing");
        let k_norm = plan.iter().position(|s| s == "rms_mul K*KNorm->K n=64 g=2").expect("k-norm missing");
        assert!(v_derive < k_norm, "V must be derived from the RAW K, before k-norm: {plan:?}");
        assert_eq!(plan[v_derive + 1], "barrier",
            "the V derive and the in-place k-norm read/write the same buffer — they need a barrier");
        assert!(!plan.iter().any(|s| s.starts_with("mv V ")),
            "a value-less layer has no v_proj tensor; it must not be dispatched: {plan:?}");
    }

    /// The per-layer KV-head count must come from `layer_num_kv_heads`, not the
    /// model-level `num_key_value_heads` (the sliding count). With MQA(1) globals
    /// the global layer's K/V are 8x narrower than the sliding layers'; reading the
    /// model level mis-sizes every one of them.
    #[test]
    fn global_layer_uses_per_layer_kv_head_count() {
        let mut cfg = tiny_cfg();
        cfg.num_key_value_heads = 2;
        cfg.num_global_key_value_heads = 1;   // MQA globals, as on g12b
        let sliding = plan_front(&gemma_front_params(&cfg, 0, true, cfg.num_attention_heads));
        let global = plan_front(&gemma_front_params(&cfg, 2, true, cfg.num_attention_heads));
        assert!(sliding.contains(&"mv K X->K 128x64".to_string()), "{sliding:?}");   // 2 * 32
        assert!(global.contains(&"mv K X->K 128x64".to_string()), "{global:?}");     // 1 * 64
        assert!(global.contains(&"rms_mul K*KNorm->K n=64 g=1".to_string()),
            "the global layer must norm ONE 64-wide KV head, not two: {global:?}");
    }

    /// KV-shared layers (E2B) carry no k_norm tensor and their K/V are discarded
    /// in favour of the target layer's cache, so neither k-norm nor v-norm is
    /// recorded — matching `gemma_resident_layer`'s `if let Some(knp)` gate.
    #[test]
    fn front_gemma_kv_shared_skips_k_and_v_norm() {
        let mut cfg = tiny_cfg();
        cfg.num_kv_shared_layers = 1;         // layer 3 becomes KV-shared
        assert!(cfg.is_kv_shared(3) && !cfg.is_kv_shared(0));
        let p = gemma_front_params(&cfg, 3, true, cfg.num_attention_heads);
        assert_plan(plan_front(&p), &[
            "rms_mul Ha*InputLn->X n=128 g=1",
            "barrier",
            "mv Q X->Q 128x128",
            "mv K X->K 128x64",
            "mv V X->V 128x64",
            "barrier",
            "rms_mul Q*QNorm->Q n=32 g=4",
            "barrier",
            "rope Q heads=4 hd=32 rd=32 fd=32 theta=10000",
            "rope K heads=2 hd=32 rd=32 fd=32 theta=10000",
        ], "gemma KV-shared front");
    }

    /// Qwen is the degenerate case: full rotary, one theta, weighted k-norm but
    /// NO v-norm. Matches `gpu_layer_2cb`'s CB1 for a `LayerSpec::qwen`.
    #[test]
    fn front_qwen_matches_prerefactor() {
        let cfg = tiny_qwen_cfg();
        let p = LayerSpec::qwen(&cfg, 0).front_params(true);
        assert_plan(plan_front(&p), &[
            "rms_mul Ha*InputLn->X n=64 g=1",
            "barrier",
            "mv Q X->Q 64x64",
            "mv K X->K 64x32",
            "mv V X->V 64x32",
            "barrier",
            "rms_mul Q*QNorm->Q n=16 g=4",
            "rms_mul K*KNorm->K n=16 g=2",
            "barrier",
            "rope Q heads=4 hd=16 rd=16 fd=16 theta=1000000",
            "rope K heads=2 hd=16 rd=16 fd=16 theta=1000000",
        ], "qwen front");
    }

    /// `gemma_attn_tp_1cb`: the caller normed `x` on the CPU, so `input_layernorm`
    /// must NOT be recorded; q is sharded to this rank's heads while K/V stay
    /// replicated at the full per-layer width.
    #[test]
    fn front_tp_shards_q_only_and_skips_input_norm() {
        let cfg = tiny_cfg();
        let n = 2;                            // TP=2
        let r_num_q = cfg.num_attention_heads / n;
        let p = gemma_front_params(&cfg, 0, false, r_num_q);
        assert_plan(plan_front(&p), &[
            "mv Q X->Q 128x64",               // r_num_q(2) * head_dim(32)
            "mv K X->K 128x64",               // replicated: num_kv(2) * 32
            "mv V X->V 128x64",
            "barrier",
            "rms_mul Q*QNorm->Q n=32 g=2",
            "rms_mul K*KNorm->K n=32 g=2",
            "rms V->V n=32 g=2",
            "barrier",
            "rope Q heads=2 hd=32 rd=32 fd=32 theta=10000",
            "rope K heads=2 hd=32 rd=32 fd=32 theta=10000",
        ], "gemma TP front");
    }

    // ── TAIL ────────────────────────────────────────────────────────────────

    /// `gemma_resident_layer`'s CB2 / `gpu_layer_2cb`'s CB2 on gemma: SANDWICH
    /// norms on each sublayer OUTPUT before the residual add.
    #[test]
    fn tail_gemma_sandwich_matches_prerefactor() {
        let cfg = tiny_cfg();
        let p = gemma_tail_params(&cfg, 128, cfg.layer_intermediate_size(0));
        assert_plan(plan_tail(&p), &[
            "mv O Attn->O 128x128",
            "barrier",
            "rms_mul O*PostAttn->On n=128 g=1",
            "barrier",
            "add Ha+On->Hb n=128",
            "barrier",
            "rms_mul Hb*FfnIn->Ffin n=128 g=1",
            "barrier",
            "mv Gate Ffin->Gate 128x256",
            "mv Up Ffin->Up 128x256",
            "barrier",
            "gelu_f32 Gate->Act n=256",
            "barrier",
            "mul Act*Up->Mid n=256",
            "barrier",
            "mv Down Mid->Down 256x128",
            "barrier",
            "rms_mul Down*PostFfn->Downn n=128 g=1",
            "barrier",
            "add Hb+Downn->Ha n=128",
        ], "gemma tail");
    }

    /// Qwen PRE-norm: no sandwich norms, the residual adds the raw sublayer
    /// output, and the activation is SiLU rather than GELU.
    #[test]
    fn tail_qwen_prenorm_matches_prerefactor() {
        let cfg = tiny_qwen_cfg();
        let p = LayerSpec::qwen(&cfg, 0).tail_params();
        assert_plan(plan_tail(&p), &[
            "mv O Attn->O 64x64",
            "barrier",
            "add Ha+O->Hb n=64",
            "barrier",
            "rms_mul Hb*FfnIn->Ffin n=64 g=1",
            "barrier",
            "mv Gate Ffin->Gate 64x128",
            "mv Up Ffin->Up 64x128",
            "barrier",
            "silu_f32 Gate->Act n=128",
            "barrier",
            "mul Act*Up->Mid n=128",
            "barrier",
            "mv Down Mid->Down 128x64",
            "barrier",
            "add Hb+Down->Ha n=64",
        ], "qwen tail");
    }

    /// The TP LEVER-1 split (`gemma_mlp_tp_full` + `gemma_layer_tail_full`) and
    /// the TP sub-block split (`gemma_mlp_tp_1cb`) must compose back into exactly
    /// the single-node tail. This is what stops the two from drifting apart again.
    #[test]
    fn tp_tail_pieces_compose_into_the_single_node_tail() {
        let cfg = tiny_cfg();
        let p = gemma_tail_params(&cfg, 128, cfg.layer_intermediate_size(0));

        let mut whole = PlanRecorder::default();
        record_tail(&mut whole, &p).unwrap();

        let mut split = PlanRecorder::default();
        record_o_proj(&mut split, &p).unwrap();               // gemma_attn_tp_*'s CB-B
        split.step(Step::Barrier).unwrap();
        record_attn_residual(&mut split, &p).unwrap();        // gemma_mlp_tp_full
        split.step(Step::Barrier).unwrap();
        record_ffn_in_norm(&mut split, &p).unwrap();
        split.step(Step::Barrier).unwrap();
        record_ffn_body(&mut split, &p).unwrap();             // == gemma_mlp_tp_1cb's CB
        split.step(Step::Barrier).unwrap();
        record_ffn_residual(&mut split, &p).unwrap();         // gemma_layer_tail_full

        assert_eq!(whole.steps, split.steps,
            "the TP pieces no longer compose into the single-node tail");
    }

    /// PLE lives OUTSIDE the layer body (a host-glued tail submit), so turning it
    /// on must not perturb a single dispatch of the front or the tail. The
    /// unconditional `ple: Some(..)` blocker was about the DESCRIPTOR, not the
    /// body; this pins that the body stays PLE-blind.
    #[test]
    fn ple_does_not_perturb_the_layer_body() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.hidden_size_per_layer_input, 0, "fixture is the no-PLE (g12b-like) shape");
        let mut with_ple = cfg.clone();
        with_ple.hidden_size_per_layer_input = 32;
        assert!(with_ple.has_ple());
        for li in 0..cfg.num_hidden_layers {
            let a = plan_front(&gemma_front_params(&cfg, li, true, cfg.num_attention_heads));
            let b = plan_front(&gemma_front_params(&with_ple, li, true, cfg.num_attention_heads));
            assert_eq!(a, b, "PLE changed layer {li}'s front");
            let ta = plan_tail(&gemma_tail_params(&cfg, 128, cfg.layer_intermediate_size(li)));
            let tb = plan_tail(&gemma_tail_params(&with_ple, 128, with_ple.layer_intermediate_size(li)));
            assert_eq!(ta, tb, "PLE changed layer {li}'s tail");
        }
    }

    /// Follow-up 3: value-less global layers are now 1-CB eligible.
    ///
    /// The old `is_layer_1cb_eligible` returned `!spec.uses_k_eq_v` because the
    /// hand-written 1-CB front indexed `v_proj` unconditionally. Now that both
    /// entry points record the SAME front, the value-less plan the 1-CB path gets
    /// is exactly the plan the 2-CB path was already executing correctly — which
    /// is what makes the exclusion unnecessary rather than merely inconvenient.
    #[test]
    fn valueless_layer_is_now_one_cb_eligible() {
        let cfg = tiny_cfg();
        let li = 2;
        let spec = LayerSpec::gemma(&cfg, li, Vec::new(), 1.0);
        assert!(spec.uses_k_eq_v, "layer {li} must be the value-less global layer");

        // The plan the 1-CB path will now record for it...
        let one_cb = plan_front(&spec.front_params(true));
        // ...is the plan the 2-CB path was already recording for it.
        let two_cb = plan_front(&gemma_front_params(&cfg, li, true, cfg.num_attention_heads));
        assert_eq!(one_cb, two_cb,
            "the 1-CB front must record what the 2-CB front recorded for a value-less layer");

        // And it never asks for the tensor that is not on disk.
        assert!(!one_cb.iter().any(|s| s.starts_with("mv V ")), "{one_cb:?}");
        assert!(!crate::unified_layer::unified_layer_weight_keys(&spec, li)
            .iter().any(|k| k.contains("v_proj")),
            "the pre-flight must not demand v_proj for a value-less layer");
    }

    /// The 1-CB and 2-CB unified entry points differ ONLY in where attention runs.
    /// They derive their front and tail from the same `LayerSpec`, so their bodies
    /// are the same plan — which is why `is_layer_1cb_eligible` no longer has to
    /// exclude any layer type on capability grounds.
    #[test]
    fn one_cb_and_two_cb_bodies_are_the_same_plan() {
        let cfg = tiny_cfg();
        for li in 0..cfg.num_hidden_layers {
            let spec = LayerSpec::gemma(&cfg, li, Vec::new(), 1.0);
            // Both entry points call `spec.front_params(true)` / `spec.tail_params()`,
            // and that plan is the one the gemma resident path records.
            assert_eq!(
                plan_front(&spec.front_params(true)),
                plan_front(&gemma_front_params(&cfg, li, true, cfg.num_attention_heads)),
                "LayerSpec::gemma and gemma_front_params disagree on layer {li}'s front");
            assert_eq!(
                plan_tail(&spec.tail_params()),
                plan_tail(&gemma_tail_params(
                    &cfg, cfg.num_attention_heads * cfg.layer_head_dim(li),
                    cfg.layer_intermediate_size(li))),
                "LayerSpec::gemma and gemma_tail_params disagree on layer {li}'s tail");
        }
    }
}
