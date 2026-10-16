// SPDX-License-Identifier: Apache-2.0
//! Single-sequence LFM2/LFM2.5 inference with typed quantized weights.
//! All forward arithmetic runs on Vulkan. Only coefficients are expanded to
//! FP32 at load time; matrix weights remain packed. No FP16/subgroup extension.

use crate::lfm2_quant::{allocate, upload, Checkpoint, Matrix, Target};
use crate::{
    compute::{Buffer, ComputeEngine},
    device::ComputeDevice,
};
use ash::vk;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{collections::HashMap, path::Path};

#[derive(Clone, Debug)]
struct Config {
    h: usize,
    f: usize,
    vocab: usize,
    heads: usize,
    kvheads: usize,
    dim: usize,
    length: usize,
    layers: Vec<bool>,
    eps: f32,
    theta: f32,
    max_position: usize,
}

impl Config {
    /// Validate supported LFM2 dimensions, operators and unscaled RoPE settings.
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        let number = |name: &str| -> Result<usize, String> {
            v[name]
                .as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .filter(|&x| x > 0 && x <= 1_000_000)
                .ok_or_else(|| format!("lfm2: invalid {name}"))
        };
        if v["model_type"] != "lfm2" {
            return Err("lfm2: expected model_type=lfm2".into());
        }
        if v["tie_word_embeddings"] != true
            || v["conv_bias"].as_bool().unwrap_or(false)
            || v["block_auto_adjust_ff_dim"].as_bool().unwrap_or(false)
            || !v["block_use_swiglu"].as_bool().unwrap_or(true)
            || !v["use_pos_enc"].as_bool().unwrap_or(true)
        {
            return Err("lfm2: requires tied embeddings, bias-free conv, explicit FFN size, SwiGLU and RoPE".into());
        }
        let (h, f, vocab, heads, kvheads) = (
            number("hidden_size")?,
            number("intermediate_size")?,
            number("vocab_size")?,
            number("num_attention_heads")?,
            number("num_key_value_heads")?,
        );
        let dim = v["head_dim"]
            .as_u64()
            .map(|x| x as usize)
            .unwrap_or(h / heads);
        let length = number("conv_L_cache")?;
        if h % 64 != 0
            || f % 64 != 0
            || h > 16384
            || f > 65536
            || vocab > 262140
            || heads % kvheads != 0
            || dim == 0
            || dim > 256
            || dim % 2 != 0
            || heads.checked_mul(dim) != Some(h)
            || length > 32
        {
            return Err("lfm2: unsupported dimensions (group64, even head_dim <=256, integral GQA required)".into());
        }
        let layers: Vec<bool> = v["layer_types"]
            .as_array()
            .ok_or("lfm2: missing layer_types")?
            .iter()
            .map(|x| match x.as_str() {
                Some("conv") => Ok(false),
                Some("full_attention") => Ok(true),
                _ => Err("lfm2: unsupported layer type".to_string()),
            })
            .collect::<Result<_, _>>()?;
        if layers.len() != number("num_hidden_layers")? || layers.len() > 512 {
            return Err("lfm2: layer_types length mismatch".into());
        }
        let rope = &v["rope_parameters"];
        if rope["rope_type"].as_str().unwrap_or("default") != "default"
            || v.get("rope_scaling").is_some_and(|x| !x.is_null())
        {
            return Err("lfm2: only default unscaled RoPE is supported".into());
        }
        let theta = rope["rope_theta"]
            .as_f64()
            .or_else(|| v["rope_theta"].as_f64())
            .unwrap_or(10000.0) as f32;
        let eps = v["norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        if !theta.is_finite() || theta <= 0.0 || !eps.is_finite() || eps <= 0.0 {
            return Err("lfm2: invalid RoPE theta or norm epsilon".into());
        }
        Ok(Self {
            h,
            f,
            vocab,
            heads,
            kvheads,
            dim,
            length,
            layers,
            theta,
            eps,
            max_position: number("max_position_embeddings")?,
        })
    }
}

enum Operator {
    Conv {
        input: Matrix,
        out: Matrix,
        weight: Buffer,
        state: Buffer,
    },
    Attention {
        q: Matrix,
        k: Matrix,
        v: Matrix,
        out: Matrix,
        qnorm: Buffer,
        knorm: Buffer,
        kc: Buffer,
        vc: Buffer,
    },
}
struct Layer {
    opnorm: Buffer,
    ffnnorm: Buffer,
    op: Operator,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
}
struct Scratch {
    hidden: Buffer,
    norm: Buffer,
    bcx: Buffer,
    op: Buffer,
    proj: Buffer,
    gate: Buffer,
    up: Buffer,
    mid: Buffer,
    q: Buffer,
    qn: Buffer,
    qr: Buffer,
    k: Buffer,
    kn: Buffer,
    v: Buffer,
    logits: Buffer,
}

/// A standalone, single-sequence GPU forward engine; independent of vLLM's scheduler.
#[pyclass(unsendable)]
pub struct Lfm2Model {
    // Drop buffers before their engine and owning logical device.
    embed: Matrix,
    finalnorm: Buffer,
    layers: Vec<Layer>,
    scratch: Scratch,
    engine: ComputeEngine,
    _device: ComputeDevice,
    config: Config,
    max_seq_len: usize,
    position: usize,
    poisoned: bool,
}

impl Lfm2Model {
    /// Validate checkpoint residency and allocate one GPU sequence with bounded buffers.
    fn load(
        path: &Path,
        device_idx: usize,
        max_seq_len: usize,
        quant: &str,
        group_size: usize,
        max_buffer_bytes: Option<usize>,
    ) -> Result<Self, String> {
        let ck = Checkpoint::open(path)?;
        let c = Config::parse(&ck.config)?;
        let target = Target::parse(quant, group_size)?;
        if max_seq_len == 0 || max_seq_len > c.max_position {
            return Err("lfm2: max_seq_len must be positive and <= max_position_embeddings".into());
        }
        let device = ComputeDevice::create(device_idx)?;
        let all = crate::include_all_shaders();
        let required = [
            "mul_mat_vec_mlx4_f32_f32",
            "lfm2_embed",
            "lfm2_matvec",
            "lfm2_norm",
            "lfm2_elementwise",
            "lfm2_rope",
            "lfm2_attention",
        ];
        // The shared cache builds two legacy F16 variant families eagerly.
        // Capability filtering may remove both; LFM2 never dispatches them.
        let selected: HashMap<&str, &[u8]> = required
            .into_iter()
            .chain(["matmul_f16_f32_fp32", "mul_mat_vec_f16_f32_f32"])
            .map(|name| {
                all.get(name)
                    .map(|b| (name, b.as_slice()))
                    .ok_or_else(|| format!("lfm2: required shader {name} missing"))
            })
            .collect::<Result<_, _>>()?;
        let mut e = ComputeEngine::new(
            device.instance.clone(),
            device.physical_device,
            device.device.clone(),
            device.compute_queue,
            device.compute_queue_family,
            device.caps(),
            &selected,
        )?;
        for name in required.into_iter().chain(["mul_mat_vec_mlx4_f32_f32_r4"]) {
            if !e.has_pipeline(name) {
                return Err(format!(
                    "lfm2: device cannot execute required pipeline {name}"
                ));
            }
        }
        let limit = max_buffer_bytes.unwrap_or(e.storage_buffer_limit() as usize);
        if limit == 0 {
            return Err("lfm2: max_buffer_bytes must be positive".into());
        }
        let matrix =
            |e: &mut ComputeEngine, name: &str, n, k| ck.matrix(e, name, n, k, target, limit);
        let embed = matrix(&mut e, "embed_tokens", c.vocab, c.h)?;
        let finalnorm = ck.upload_floats(&mut e, "embedding_norm.weight", &[c.h])?;
        let mut layers = Vec::new();
        for (i, &attention) in c.layers.iter().enumerate() {
            let root = format!("layers.{i}");
            let opnorm =
                ck.upload_floats(&mut e, &format!("{root}.operator_norm.weight"), &[c.h])?;
            let ffnnorm = ck.upload_floats(&mut e, &format!("{root}.ffn_norm.weight"), &[c.h])?;
            let op = if attention {
                let prefix = format!("{root}.self_attn");
                let kv = c.kvheads * c.dim;
                let bytes = max_seq_len
                    .checked_mul(kv)
                    .and_then(|x| x.checked_mul(4))
                    .ok_or("lfm2: KV cache size overflow")?;
                Operator::Attention {
                    q: matrix(&mut e, &format!("{prefix}.q_proj"), c.h, c.h)?,
                    k: matrix(&mut e, &format!("{prefix}.k_proj"), kv, c.h)?,
                    v: matrix(&mut e, &format!("{prefix}.v_proj"), kv, c.h)?,
                    out: matrix(&mut e, &format!("{prefix}.out_proj"), c.h, c.h)?,
                    qnorm: ck.upload_floats(
                        &mut e,
                        &format!("{prefix}.q_layernorm.weight"),
                        &[c.dim],
                    )?,
                    knorm: ck.upload_floats(
                        &mut e,
                        &format!("{prefix}.k_layernorm.weight"),
                        &[c.dim],
                    )?,
                    kc: allocate(&e, bytes)?,
                    vc: allocate(&e, bytes)?,
                }
            } else {
                let prefix = format!("{root}.conv");
                let name = format!("{prefix}.conv.weight");
                // MLX stores [channels,kernel,1]; HF stores [channels,1,kernel].
                let weight = match ck.floats(&name, &[c.h, c.length, 1]) {
                    Ok(v) => v,
                    Err(_) => ck.floats(&name, &[c.h, 1, c.length])?,
                };
                Operator::Conv {
                    input: matrix(&mut e, &format!("{prefix}.in_proj"), 3 * c.h, c.h)?,
                    out: matrix(&mut e, &format!("{prefix}.out_proj"), c.h, c.h)?,
                    weight: upload(&mut e, bytemuck::cast_slice(&weight))?,
                    state: allocate(&e, 4 * c.h * (c.length - 1).max(1))?,
                }
            };
            layers.push(Layer {
                opnorm,
                ffnnorm,
                op,
                gate: matrix(&mut e, &format!("{root}.feed_forward.w1"), c.f, c.h)?,
                up: matrix(&mut e, &format!("{root}.feed_forward.w3"), c.f, c.h)?,
                down: matrix(&mut e, &format!("{root}.feed_forward.w2"), c.h, c.f)?,
            });
        }
        let f32buf = |n| allocate(&e, 4 * n);
        let scratch = Scratch {
            hidden: f32buf(c.h)?,
            norm: f32buf(c.h)?,
            bcx: f32buf(3 * c.h)?,
            op: f32buf(c.h)?,
            proj: f32buf(c.h)?,
            gate: f32buf(c.f)?,
            up: f32buf(c.f)?,
            mid: f32buf(c.f)?,
            q: f32buf(c.h)?,
            qn: f32buf(c.h)?,
            qr: f32buf(c.h)?,
            k: f32buf(c.kvheads * c.dim)?,
            kn: f32buf(c.kvheads * c.dim)?,
            v: f32buf(c.kvheads * c.dim)?,
            logits: e.alloc_host_coherent_storage((4 * c.vocab) as u64)?,
        };
        Ok(Self {
            embed,
            finalnorm,
            layers,
            scratch,
            engine: e,
            _device: device,
            config: c,
            max_seq_len,
            position: 0,
            poisoned: false,
        })
    }

    /// Reject invalid requests before modifying recurrent state or cache positions.
    fn validate(&self, tokens: &[u32], position: usize) -> Result<(), String> {
        if self.poisoned {
            return Err("lfm2: prior GPU failure; recreate the model before reuse".into());
        }
        if tokens.is_empty() {
            return Err("lfm2: tokens must not be empty".into());
        }
        if position != self.position {
            return Err(format!(
                "lfm2: position {position} must equal cache position {}",
                self.position
            ));
        }
        if tokens.iter().any(|&x| x as usize >= self.config.vocab) {
            return Err("lfm2: token outside vocabulary".into());
        }
        if position
            .checked_add(tokens.len())
            .is_none_or(|end| end > self.max_seq_len)
        {
            return Err("lfm2: sequence exceeds max_seq_len".into());
        }
        Ok(())
    }

    /// Execute a validated token chunk and return final logits, poisoning on GPU failure.
    fn run(&mut self, tokens: &[u32], position: usize) -> Result<Vec<f32>, String> {
        self.validate(tokens, position)?;
        // A runtime failure may leave partial recurrent state or a broken fence.
        // Reject reuse; validation errors above leave live state untouched.
        self.poisoned = true;
        for (i, &token) in tokens.iter().enumerate() {
            self.step(token, i + 1 == tokens.len())?;
            self.position += 1;
        }
        let mut bytes = vec![0u8; 4 * self.config.vocab];
        self.scratch.logits.read(&mut bytes)?;
        self.poisoned = false;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    /// Record and submit one token through convolution, attention and feed-forward layers.
    fn step(&mut self, token: u32, logits: bool) -> Result<(), String> {
        let c = &self.config;
        let s = &self.scratch;
        let e = &mut self.engine;
        e.reset_descriptor_sets();
        let cb = e.begin_batch()?;
        let result = (|| {
            self.embed.embed(e, cb, token, &s.hidden)?;
            for layer in &self.layers {
                norm(e, cb, &s.hidden, &layer.opnorm, &s.norm, c.h, 1, c.eps)?;
                match &layer.op {
                    Operator::Conv {
                        input,
                        out,
                        weight,
                        state,
                    } => {
                        input.record(e, cb, &s.norm, &s.bcx)?;
                        elementwise(
                            e,
                            cb,
                            2,
                            &s.bcx,
                            weight,
                            &s.op,
                            state,
                            c.h,
                            c.length,
                            self.position,
                        )?;
                        out.record(e, cb, &s.op, &s.proj)?;
                    }
                    Operator::Attention {
                        q,
                        k,
                        v,
                        out,
                        qnorm,
                        knorm,
                        kc,
                        vc,
                    } => {
                        q.record(e, cb, &s.norm, &s.q)?;
                        k.record(e, cb, &s.norm, &s.k)?;
                        v.record(e, cb, &s.norm, &s.v)?;
                        norm(e, cb, &s.q, qnorm, &s.qn, c.dim, c.heads, c.eps)?;
                        norm(e, cb, &s.k, knorm, &s.kn, c.dim, c.kvheads, c.eps)?;
                        let mut pc = vec![
                            c.dim as u32,
                            c.heads as u32,
                            c.kvheads as u32,
                            self.position as u32,
                            c.theta.to_bits(),
                        ];
                        e.record_to(
                            cb,
                            "lfm2_rope",
                            &[&s.qn, &s.kn, &s.v, &s.qr, kc, vc],
                            bytemuck::cast_slice(&pc),
                            ((c.heads * c.dim / 2).div_ceil(64) as u32, 1, 1),
                        )?;
                        e.record_barrier_to(cb);
                        pc.pop();
                        e.record_to(
                            cb,
                            "lfm2_attention",
                            &[&s.qr, kc, vc, &s.op],
                            bytemuck::cast_slice(&pc),
                            (c.heads as u32, 1, 1),
                        )?;
                        e.record_barrier_to(cb);
                        out.record(e, cb, &s.op, &s.proj)?;
                    }
                }
                elementwise(e, cb, 0, &s.hidden, &s.proj, &s.hidden, &s.mid, c.h, 0, 0)?;
                norm(e, cb, &s.hidden, &layer.ffnnorm, &s.norm, c.h, 1, c.eps)?;
                layer.gate.record(e, cb, &s.norm, &s.gate)?;
                layer.up.record(e, cb, &s.norm, &s.up)?;
                elementwise(e, cb, 1, &s.gate, &s.up, &s.mid, &s.proj, c.f, 0, 0)?;
                layer.down.record(e, cb, &s.mid, &s.proj)?;
                elementwise(e, cb, 0, &s.hidden, &s.proj, &s.hidden, &s.mid, c.h, 0, 0)?;
            }
            if logits {
                norm(e, cb, &s.hidden, &self.finalnorm, &s.norm, c.h, 1, c.eps)?;
                self.embed.record(e, cb, &s.norm, &s.logits)?;
            }
            Ok(())
        })();
        // Close a partial recording as well; the instance stays poisoned on any
        // error and must be recreated, rather than reusing unknown device state.
        let submit = e.submit_batch(cb);
        result.and(submit)
    }
}

/// Record FP32 RMS normalization and a barrier for subsequent GPU reads.
fn norm(
    e: &mut ComputeEngine,
    cb: vk::CommandBuffer,
    x: &Buffer,
    w: &Buffer,
    y: &Buffer,
    width: usize,
    rows: usize,
    eps: f32,
) -> Result<(), String> {
    e.record_to(
        cb,
        "lfm2_norm",
        &[x, w, y],
        bytemuck::cast_slice(&[width as u32, eps.to_bits()]),
        (rows as u32, 1, 1),
    )?;
    e.record_barrier_to(cb);
    Ok(())
}
/// Record residual, activation or recurrent-convolution work and its barrier.
#[allow(clippy::too_many_arguments)]
fn elementwise(
    e: &mut ComputeEngine,
    cb: vk::CommandBuffer,
    mode: u32,
    x: &Buffer,
    y: &Buffer,
    out: &Buffer,
    state: &Buffer,
    width: usize,
    length: usize,
    position: usize,
) -> Result<(), String> {
    e.record_to(
        cb,
        "lfm2_elementwise",
        &[x, y, out, state],
        bytemuck::cast_slice(&[mode, width as u32, length as u32, position as u32]),
        (width.div_ceil(64) as u32, 1, 1),
    )?;
    e.record_barrier_to(cb);
    Ok(())
}

#[pymethods]
impl Lfm2Model {
    /// Create a single-sequence Vulkan engine with explicit device and residency options.
    #[new]
    #[pyo3(signature=(model_path,device_idx=0,max_seq_len=2048,quant="auto",group_size=64,max_buffer_bytes=None))]
    fn new(
        model_path: &str,
        device_idx: usize,
        max_seq_len: usize,
        quant: &str,
        group_size: usize,
        max_buffer_bytes: Option<usize>,
    ) -> PyResult<Self> {
        Self::load(
            Path::new(model_path),
            device_idx,
            max_seq_len,
            quant,
            group_size,
            max_buffer_bytes,
        )
        .map_err(PyValueError::new_err)
    }
    /// Export canonical packed residency bytes for independent reference validation.
    #[staticmethod]
    #[pyo3(signature=(source,destination,quant,group_size=64))]
    fn convert_checkpoint(
        source: &str,
        destination: &str,
        quant: &str,
        group_size: usize,
    ) -> PyResult<()> {
        let ck = Checkpoint::open(Path::new(source)).map_err(PyValueError::new_err)?;
        let c = Config::parse(&ck.config).map_err(PyValueError::new_err)?;
        let mut specs = vec![("embed_tokens".to_string(), c.vocab, c.h)];
        for (i, &attention) in c.layers.iter().enumerate() {
            let p = format!("layers.{i}");
            for (suffix, n, k) in [
                ("feed_forward.w1", c.f, c.h),
                ("feed_forward.w3", c.f, c.h),
                ("feed_forward.w2", c.h, c.f),
            ] {
                specs.push((format!("{p}.{suffix}"), n, k));
            }
            if attention {
                for (suffix, n) in [
                    ("q_proj", c.h),
                    ("k_proj", c.kvheads * c.dim),
                    ("v_proj", c.kvheads * c.dim),
                    ("out_proj", c.h),
                ] {
                    specs.push((format!("{p}.self_attn.{suffix}"), n, c.h));
                }
            } else {
                specs.push((format!("{p}.conv.in_proj"), 3 * c.h, c.h));
                specs.push((format!("{p}.conv.out_proj"), c.h, c.h));
            }
        }
        drop(ck);
        crate::lfm2_quant::convert_checkpoint(
            Path::new(source),
            Path::new(destination),
            quant,
            group_size,
            &specs,
        )
        .map_err(PyValueError::new_err)
    }
    /// Count logical matrices by the actual resident format, independent of flags.
    fn weight_formats(&self) -> HashMap<String, usize> {
        let mut formats = HashMap::new();
        let mut add = |m: &Matrix| *formats.entry(m.label().to_string()).or_insert(0) += 1;
        add(&self.embed);
        for l in &self.layers {
            add(&l.gate);
            add(&l.up);
            add(&l.down);
            match &l.op {
                Operator::Conv { input, out, .. } => {
                    add(input);
                    add(out);
                }
                Operator::Attention { q, k, v, out, .. } => {
                    add(q);
                    add(k);
                    add(v);
                    add(out);
                }
            }
        }
        formats
    }
    /// Append exactly one token at the current cache position and return full logits.
    #[pyo3(signature=(token_id,position=None))]
    fn forward(&mut self, token_id: u32, position: Option<usize>) -> PyResult<Vec<f32>> {
        self.run(&[token_id], position.unwrap_or(self.position))
            .map_err(PyRuntimeError::new_err)
    }
    /// Append a nonempty prompt chunk. Return logits for its final token.
    fn forward_prefill(&mut self, tokens: Vec<u32>) -> PyResult<Vec<f32>> {
        self.run(&tokens, self.position)
            .map_err(PyRuntimeError::new_err)
    }
    /// Start a new independent sequence. Stale KV is unreachable; conv clears on step zero.
    fn reset_kv_cache(&mut self) -> PyResult<()> {
        if self.poisoned {
            return Err(PyRuntimeError::new_err(
                "lfm2: prior GPU failure; recreate the model before reuse",
            ));
        }
        self.position = 0;
        Ok(())
    }
    /// Return the next token position in the current cached sequence.
    #[getter]
    fn position(&self) -> usize {
        self.position
    }
    /// Return the number of configured convolution and attention blocks.
    fn num_layers(&self) -> usize {
        self.layers.len()
    }
    /// Report that this engine executes its forward operations on Vulkan.
    fn has_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Build a minimal supported configuration for parser regression tests.
    fn config() -> serde_json::Value {
        serde_json::json!({"model_type":"lfm2","hidden_size":64,"intermediate_size":128,"vocab_size":128,
            "num_attention_heads":2,"num_key_value_heads":1,"num_hidden_layers":2,"conv_L_cache":3,
            "layer_types":["conv","full_attention"],"max_position_embeddings":64,"tie_word_embeddings":true,
            "quantization":{"bits":4,"group_size":64,"mode":"affine"}})
    }
    /// Reject configurations outside the implemented architecture and dimension limits.
    #[test]
    fn config_rejects_unsupported_architecture_and_formats() {
        let v = config();
        assert!(Config::parse(&v).is_ok());
        for (key, value) in [
            ("model_type", serde_json::json!("qwen3")),
            ("conv_bias", serde_json::json!(true)),
            ("tie_word_embeddings", serde_json::json!(false)),
            ("hidden_size", serde_json::json!(65)),
            ("num_key_value_heads", serde_json::json!(3)),
            ("layer_types", serde_json::json!(["conv", "mamba"])),
            ("intermediate_size", serde_json::json!(0)),
            ("head_dim", serde_json::json!(512)),
            ("block_auto_adjust_ff_dim", serde_json::json!(true)),
            ("rope_scaling", serde_json::json!({"factor":2})),
        ] {
            let mut bad = v.clone();
            bad[key] = value;
            assert!(Config::parse(&bad).is_err(), "{key}");
        }
    }
    /// Check RoPE defaults, legacy fallback, nested precedence and invalid bases.
    #[test]
    fn rope_theta_defaults_legacy_values_and_nested_precedence() {
        let mut v = config();
        assert_eq!(Config::parse(&v).unwrap().theta, 10000.0);
        v["rope_theta"] = serde_json::json!(7.0);
        assert_eq!(Config::parse(&v).unwrap().theta, 7.0);
        v["rope_parameters"] = serde_json::Value::Null;
        assert_eq!(Config::parse(&v).unwrap().theta, 7.0);
        v["rope_parameters"] = serde_json::json!({"rope_theta":37.0});
        assert_eq!(Config::parse(&v).unwrap().theta, 37.0);
        v["rope_parameters"] = serde_json::Value::Null;
        for theta in [0.0, -1.0, 1e100] {
            v["rope_theta"] = serde_json::json!(theta);
            assert!(Config::parse(&v).is_err());
        }
    }
}
