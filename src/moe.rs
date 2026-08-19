// SPDX-License-Identifier: Apache-2.0
//! Qwen3.6 (`qwen3_5`) Mixture-of-Experts block — CPU reference (Milestone C).
//!
//! This is the standalone MoE math for the 35B-A3B model, ported op-by-op from
//! the qwen3.6-mlx reference (`qwen3.6-mlx/src/moe.rs`) and validated against a
//! tiny MLX oracle (`examples/moe_oracle.rs`). It is NOT yet wired into the live
//! forward path — that swap (dense MLP -> MoE) and the GPU expert kernels / EP
//! come later, after pipeline-parallel lands.
//!
//! For each token `x` (hidden vector):
//! ```text
//!   logits  = gate · x                         # [num_experts]
//!   gates   = softmax(logits)                   # over all experts
//!   idx     = top-k experts by gates            # argpartition(gates, N-k)[N-k:]
//!   scores  = gates[idx] / sum(gates[idx])      # renormalised
//!   routed  = Σ_k scores_k · expert_k(x)        # SwiGLU @ moe_intermediate_size
//!   shared  = sigmoid(shared_expert_gate · x) · shared_expert(x)   # SwiGLU
//!   out     = routed + shared
//! ```
//! The router normalisation is **softmax-then-topk-then-renorm** (the gates are
//! softmaxed over ALL experts first, then the selected top-k subset is divided
//! by its own sum). Confirmed against the MLX `MoeBlock::forward`.
#![allow(dead_code)]

use crate::model::{cpu_matmul, cpu_silu, ModelWeights};

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable softmax over a slice (in place semantics returned as Vec).
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

/// One SwiGLU FFN: `down(silu(gate·x) * (up·x))`.
/// `gate_w`/`up_w` are `[inter, hidden]`, `down_w` is `[hidden, inter]` (row-major,
/// each as the MLX `Linear` weight `[out, in]` applied as `x @ W^T`). Public so
/// the oracle validator can reconstruct a single expert's contribution.
pub fn expert_swiglu(x: &[f32], gate_w: &[f32], up_w: &[f32], down_w: &[f32], hidden: usize, inter: usize) -> Vec<f32> {
    let gate = cpu_matmul(x, gate_w, 1, hidden, inter);
    let up = cpu_matmul(x, up_w, 1, hidden, inter);
    let act = cpu_silu(&gate);
    let mid: Vec<f32> = act.iter().zip(&up).map(|(&g, &u)| g * u).collect();
    cpu_matmul(&mid, down_w, 1, inter, hidden)
}

/// Rayon-parallel SwiGLU for the t=1 (decode) case — the shared expert. The two
/// gate/up matvecs run on the SAME thread pool but across the `inter` output rows;
/// the down matvec across the `hidden` rows. Bit-comparable (per-row dot product,
/// within-row order unchanged) to `expert_swiglu`; ~min(cores) faster. STEP-0
/// localized the 35B-A3B per-stage compute to this CPU shared expert (~25ms/8
/// layers single-thread).
pub fn expert_swiglu_par(x: &[f32], gate_w: &[f32], up_w: &[f32], down_w: &[f32], hidden: usize, inter: usize) -> Vec<f32> {
    use crate::model::cpu_matvec_par;
    let gate = cpu_matvec_par(x, gate_w, hidden, inter);
    let up = cpu_matvec_par(x, up_w, hidden, inter);
    let act = cpu_silu(&gate);
    let mid: Vec<f32> = act.iter().zip(&up).map(|(&g, &u)| g * u).collect();
    cpu_matvec_par(&mid, down_w, inter, hidden)
}

/// Rayon-parallel router (the gate matmul 2048->256 is the cost; top-k is cheap).
/// Bit-comparable to `route`.
pub fn route_par(x: &[f32], gate_w: &[f32], hidden: usize, num_experts: usize, top_k: usize) -> Routing {
    use crate::model::cpu_matvec_par;
    let logits = cpu_matvec_par(x, gate_w, hidden, num_experts);
    route_from_logits(&logits, top_k)
}

/// Router tail shared by all logit producers (CPU matmul, rayon matvec, or the
/// WS2 GPU gate matvec): softmax over all experts, top-k by gate, renormalise
/// the selected subset. Identical math to the body `route`/`route_par` always
/// ran — only the logits source varies.
pub fn route_from_logits(logits: &[f32], top_k: usize) -> Routing {
    let num_experts = logits.len();
    let gates = softmax(logits);
    let mut order: Vec<usize> = (0..num_experts).collect();
    order.sort_unstable_by(|&a, &b| {
        gates[b].partial_cmp(&gates[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut indices: Vec<usize> = order[..top_k].to_vec();
    indices.sort_unstable();
    let sel: Vec<f32> = indices.iter().map(|&i| gates[i]).collect();
    let sum: f32 = sel.iter().sum();
    let scores: Vec<f32> = sel.iter().map(|&s| if sum > 0.0 { s / sum } else { 0.0 }).collect();
    Routing { gates, indices, scores }
}

/// Result of the router for one token (for validation + the main path).
#[derive(Debug, Clone)]
pub struct Routing {
    /// softmax over all experts, length `num_experts`.
    pub gates: Vec<f32>,
    /// selected expert indices, length `top_k`.
    pub indices: Vec<usize>,
    /// renormalised scores for the selected experts, length `top_k`.
    pub scores: Vec<f32>,
}

/// Router: softmax(gate·x), select top-k by softmaxed gate, renormalise the
/// selected subset. Mirrors the MLX `argpartition(gates, N-k)[N-k:]` selection;
/// the within-top-k ORDER here is ascending-by-gate (the partition pivot order),
/// but since the routed output is an order-independent weighted SUM, only the
/// SELECTED SET matters for the final output. The returned `indices` are sorted
/// ascending by expert id for a deterministic, comparable layout.
pub fn route(x: &[f32], gate_w: &[f32], hidden: usize, num_experts: usize, top_k: usize) -> Routing {
    let logits = cpu_matmul(x, gate_w, 1, hidden, num_experts);
    route_from_logits(&logits, top_k)
}

/// Per-layer dequantized MoE weights for the CPU reference path.
/// All weights are row-major f32 in the MLX `Linear` `[out, in]` convention;
/// the `switch_*` tensors are stacked `[num_experts, out, in]`.
/// `Default` (all-empty Vecs) lets the MTP head `std::mem::take` its `moe_w`
/// out to hand the CPU-fallback MoE closure ownership while the head is mutably
/// borrowed for its KV — the same take/restore dance the dense projections use.
#[derive(Default)]
pub struct MoeWeights {
    pub gate: Vec<f32>,            // [num_experts, hidden]
    pub switch_gate: Vec<f32>,     // [num_experts, moe_inter, hidden]
    pub switch_up: Vec<f32>,       // [num_experts, moe_inter, hidden]
    pub switch_down: Vec<f32>,     // [num_experts, hidden, moe_inter]
    pub shared_gate: Vec<f32>,     // [shared_inter, hidden]
    pub shared_up: Vec<f32>,       // [shared_inter, hidden]
    pub shared_down: Vec<f32>,     // [hidden, shared_inter]
    pub shared_expert_gate: Vec<f32>, // [1, hidden]
}

/// MoE block dimensions.
#[derive(Debug, Clone, Copy)]
pub struct MoeDims {
    pub hidden: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
}

/// Compute one token's MoE output, returning the routing (for validation) and
/// the final hidden-sized output vector.
pub fn moe_forward_token(x: &[f32], w: &MoeWeights, d: MoeDims) -> (Routing, Vec<f32>) {
    let h = d.hidden;
    let mi = d.moe_inter;

    let routing = route(x, &w.gate, h, d.num_experts, d.top_k);

    // Routed experts: weighted sum of selected SwiGLU expert outputs.
    let mut routed = vec![0.0f32; h];
    let gate_stride = mi * h; // per-expert gate/up tensor size
    let down_stride = h * mi; // per-expert down tensor size
    for (slot, &e) in routing.indices.iter().enumerate() {
        let gate_w = &w.switch_gate[e * gate_stride..(e + 1) * gate_stride];
        let up_w = &w.switch_up[e * gate_stride..(e + 1) * gate_stride];
        let down_w = &w.switch_down[e * down_stride..(e + 1) * down_stride];
        let expert_out = expert_swiglu(x, gate_w, up_w, down_w, h, mi);
        let s = routing.scores[slot];
        for (r, &o) in routed.iter_mut().zip(&expert_out) {
            *r += s * o;
        }
    }

    // Shared expert, gated by sigmoid(shared_expert_gate · x).
    let shared_out = expert_swiglu(x, &w.shared_gate, &w.shared_up, &w.shared_down, h, d.shared_inter);
    let sg_logit = cpu_matmul(x, &w.shared_expert_gate, 1, h, 1)[0];
    let sg = sigmoid(sg_logit);

    let out: Vec<f32> = routed
        .iter()
        .zip(&shared_out)
        .map(|(&r, &s)| r + sg * s)
        .collect();

    (routing, out)
}

/// Same math as `moe_forward_token`, parallelized over the `top_k` routed
/// experts via rayon (plan §P2 remainder — the MTP draft head's D budget: its
/// dense-f32 CPU MoE, unlike the main forward's 4-bit-resident+rayon path, was
/// single-threaded and dominated D; ~256 experts / mi=512 / top_k=8 costs
/// ~30ms on one core, ~4-6ms across a modern core count). Each routed expert's
/// `expert_swiglu` is independent (no shared mutable state) — rayon only
/// parallelizes their computation; the final weighted reduction still walks
/// `routing.indices` in the SAME order as the serial version, so the result is
/// bit-identical (this is a wall-clock optimization, not a numerics change).
pub fn moe_forward_token_rayon(x: &[f32], w: &MoeWeights, d: MoeDims) -> (Routing, Vec<f32>) {
    use rayon::prelude::*;
    let h = d.hidden;
    let mi = d.moe_inter;

    let routing = route(x, &w.gate, h, d.num_experts, d.top_k);

    let gate_stride = mi * h;
    let down_stride = h * mi;
    let expert_outs: Vec<Vec<f32>> = routing
        .indices
        .par_iter()
        .map(|&e| {
            let gate_w = &w.switch_gate[e * gate_stride..(e + 1) * gate_stride];
            let up_w = &w.switch_up[e * gate_stride..(e + 1) * gate_stride];
            let down_w = &w.switch_down[e * down_stride..(e + 1) * down_stride];
            expert_swiglu(x, gate_w, up_w, down_w, h, mi)
        })
        .collect();

    let mut routed = vec![0.0f32; h];
    for (slot, expert_out) in expert_outs.iter().enumerate() {
        let s = routing.scores[slot];
        for (r, &o) in routed.iter_mut().zip(expert_out) {
            *r += s * o;
        }
    }

    let shared_out = expert_swiglu(x, &w.shared_gate, &w.shared_up, &w.shared_down, h, d.shared_inter);
    let sg_logit = cpu_matmul(x, &w.shared_expert_gate, 1, h, 1)[0];
    let sg = sigmoid(sg_logit);

    let out: Vec<f32> = routed
        .iter()
        .zip(&shared_out)
        .map(|(&r, &s)| r + sg * s)
        .collect();

    (routing, out)
}

/// A single MLX-affine-quantized 3D switch tensor `[num_experts, out, in]`,
/// stored RESIDENT in its packed 4-bit form (never dequantized in bulk). Only
/// the few experts a token routes to are dequantized on the fly, so a whole
/// MoE layer's experts cost ~0.4GB (4-bit) instead of ~3.2GB (f32) — the
/// difference between fitting an 8-layer PP stage on a 15GB node and OOMing.
#[derive(Clone)]
pub struct QuantSwitch {
    pub packed: Vec<u32>,   // [E * out * (in/per_word)]
    pub scales: Vec<f32>,   // [E * out * groups]
    pub biases: Vec<f32>,   // [E * out * groups]
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
    pub bits: usize,
}

impl QuantSwitch {
    /// Dequantize ONE expert's full `[out, in]` weight (row-major) to f32.
    pub fn dequant_expert(&self, expert: usize) -> Vec<f32> {
        let per_word = 32 / self.bits;
        let mask = (1u32 << self.bits) - 1;
        let groups = self.in_features / self.group_size;
        let words_per_row = self.in_features / per_word;
        let rows = self.out_features; // per-expert out rows
        let pack_stride = rows * words_per_row; // u32 per expert
        let sb_stride = rows * groups; // scales/biases per expert
        let pb = &self.packed[expert * pack_stride..(expert + 1) * pack_stride];
        let ps = &self.scales[expert * sb_stride..(expert + 1) * sb_stride];
        let pbias = &self.biases[expert * sb_stride..(expert + 1) * sb_stride];
        let mut w = vec![0.0f32; rows * self.in_features];
        for o in 0..rows {
            for i in 0..self.in_features {
                let word = pb[o * words_per_row + i / per_word];
                let q = ((word >> ((i % per_word) * self.bits)) & mask) as f32;
                let g = i / self.group_size;
                w[o * self.in_features + i] = ps[o * groups + g] * q + pbias[o * groups + g];
            }
        }
        w
    }

    /// Drop the host-resident packed/scales/biases Vecs (metadata — out/in
    /// features, group_size, bits — kept intact so shape queries still work).
    /// Callers must only do this once a GPU-resident mirror of the same data
    /// fully exists (see `ensure_moe_gpu_layer`'s `VLLM_VULKAN_MOE_HOST_FREE`
    /// gate) — after this call, `dequant_expert` and any other host-data
    /// reader (e.g. the CPU quant-MoE fallback) will operate on empty Vecs.
    pub fn free_host_data(&mut self) {
        self.packed = Vec::new();
        self.scales = Vec::new();
        self.biases = Vec::new();
    }
}

/// Per-layer 4-bit-resident expert weights (router gate + shared expert stay
/// host f32 in `ModelWeights` — they are small). Indexed by GLOBAL layer index.
#[derive(Default)]
pub struct QuantMoeLayers {
    pub gate: std::collections::HashMap<usize, QuantSwitch>,  // gate_proj experts
    pub up: std::collections::HashMap<usize, QuantSwitch>,
    pub down: std::collections::HashMap<usize, QuantSwitch>,
}

/// MoE forward for one token using 4-bit-RESIDENT experts: only the top-k routed
/// experts are dequantized (on the fly), so the layer's ~32B-param expert set
/// never materializes as f32. Router gate + shared expert come from host f32
/// `ModelWeights`. Math is bit-identical to `moe_forward_token`.
pub fn moe_forward_token_quant(
    x: &[f32],
    weights: &ModelWeights,
    q: &QuantMoeLayers,
    layer_idx: usize,
    d: MoeDims,
) -> (Routing, Vec<f32>) {
    let h = d.hidden;
    let mi = d.moe_inter;
    let p = format!("model.layers.{layer_idx}.mlp");
    let s = |name: &str| weights.f32_slice(&format!("{p}.{name}"));

    let routing = route(x, s("gate.weight"), h, d.num_experts, d.top_k);

    let qg = &q.gate[&layer_idx];
    let qu = &q.up[&layer_idx];
    let qd = &q.down[&layer_idx];

    let mut routed = vec![0.0f32; h];
    for (slot, &e) in routing.indices.iter().enumerate() {
        let gw = qg.dequant_expert(e);
        let uw = qu.dequant_expert(e);
        let dw = qd.dequant_expert(e);
        let expert_out = expert_swiglu(x, &gw, &uw, &dw, h, mi);
        let sc = routing.scores[slot];
        for (r, &o) in routed.iter_mut().zip(&expert_out) {
            *r += sc * o;
        }
    }

    let shared_out = expert_swiglu(
        x,
        s("shared_expert.gate_proj.weight"),
        s("shared_expert.up_proj.weight"),
        s("shared_expert.down_proj.weight"),
        h,
        d.shared_inter,
    );
    let sg_logit = cpu_matmul(x, s("shared_expert_gate.weight"), 1, h, 1)[0];
    let sg = sigmoid(sg_logit);

    let out: Vec<f32> = routed
        .iter()
        .zip(&shared_out)
        .map(|(&r, &sh)| r + sg * sh)
        .collect();

    (routing, out)
}

/// Compute one token's MoE output by BORROWING the per-layer expert tensors
/// straight from a `ModelWeights` map (no per-call copy of the ~1.6GB switch
/// tensors — the difference between usable and unusable in the live forward).
/// The weights must already be DEQUANTIZED f32 (the extended `load_qwen35_weights`
/// MoE path); `switch_mlp.*` are flattened `[E*out, in]`, identical to
/// `MoeWeights`. Math is bit-identical to `moe_forward_token`.
pub fn moe_forward_token_borrowed(
    x: &[f32],
    weights: &ModelWeights,
    layer_idx: usize,
    d: MoeDims,
) -> (Routing, Vec<f32>) {
    let h = d.hidden;
    let mi = d.moe_inter;
    let p = format!("model.layers.{layer_idx}.mlp");
    let s = |name: &str| weights.f32_slice(&format!("{p}.{name}"));

    let gate_w = s("gate.weight");
    let switch_gate = s("switch_mlp.gate_proj.weight");
    let switch_up = s("switch_mlp.up_proj.weight");
    let switch_down = s("switch_mlp.down_proj.weight");

    let routing = route(x, gate_w, h, d.num_experts, d.top_k);

    let mut routed = vec![0.0f32; h];
    let gate_stride = mi * h;
    let down_stride = h * mi;
    for (slot, &e) in routing.indices.iter().enumerate() {
        let gw = &switch_gate[e * gate_stride..(e + 1) * gate_stride];
        let uw = &switch_up[e * gate_stride..(e + 1) * gate_stride];
        let dw = &switch_down[e * down_stride..(e + 1) * down_stride];
        let expert_out = expert_swiglu(x, gw, uw, dw, h, mi);
        let sc = routing.scores[slot];
        for (r, &o) in routed.iter_mut().zip(&expert_out) {
            *r += sc * o;
        }
    }

    let shared_out = expert_swiglu(
        x,
        s("shared_expert.gate_proj.weight"),
        s("shared_expert.up_proj.weight"),
        s("shared_expert.down_proj.weight"),
        h,
        d.shared_inter,
    );
    let sg_logit = cpu_matmul(x, s("shared_expert_gate.weight"), 1, h, 1)[0];
    let sg = sigmoid(sg_logit);

    let out: Vec<f32> = routed
        .iter()
        .zip(&shared_out)
        .map(|(&r, &sh)| r + sg * sh)
        .collect();

    (routing, out)
}

/// Load one layer's dequantized MoE weights from a `ModelWeights` map using the
/// real checkpoint tensor names (`model.layers.{i}.mlp.*`). The map is expected
/// to already hold DEQUANTIZED f32 tensors (e.g. produced by the validated
/// `load_qwen35_weights` quant path, or the oracle's `.dequant` keys).
///
/// `dequant_suffix` lets the oracle (which stores switch tensors under
/// `.dequant`) and the real loader (plain `.weight`) share this code.
pub fn load_moe_weights(
    weights: &ModelWeights,
    layer_idx: usize,
    switch_suffix: &str,
) -> MoeWeights {
    let p = format!("model.layers.{layer_idx}.mlp");
    let get = |name: String| weights.f32_slice(&name).to_vec();
    MoeWeights {
        gate: get(format!("{p}.gate.weight")),
        switch_gate: get(format!("{p}.switch_mlp.gate_proj.{switch_suffix}")),
        switch_up: get(format!("{p}.switch_mlp.up_proj.{switch_suffix}")),
        switch_down: get(format!("{p}.switch_mlp.down_proj.{switch_suffix}")),
        shared_gate: get(format!("{p}.shared_expert.gate_proj.weight")),
        shared_up: get(format!("{p}.shared_expert.up_proj.weight")),
        shared_down: get(format!("{p}.shared_expert.down_proj.weight")),
        shared_expert_gate: get(format!("{p}.shared_expert_gate.weight")),
    }
}
