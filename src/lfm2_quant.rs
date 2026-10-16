// SPDX-License-Identifier: Apache-2.0
//! Typed LFM2 weight storage. The same codec serves embedding and projection.
use crate::compute::{Buffer, ComputeEngine};
use ash::vk;
use memmap2::Mmap;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde_json::Value;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum Format {
    F32,
    F16,
    Bf16,
    Mlx,
    Q8,
    Q4,
    Q4K,
    Fp8,
    Nvfp4,
}
impl Format {
    /// Return the canonical checkpoint format identifier.
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Mlx => "mlx",
            Self::Q8 => "q8_0",
            Self::Q4 => "q4_0",
            Self::Q4K => "q4_k",
            Self::Fp8 => "fp8",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Target {
    format: Option<Format>,
    bits: usize,
    group: usize,
}
impl Target {
    /// Resolve native or explicit residency and validate MLX grouping options.
    pub fn parse(name: &str, group: usize) -> Result<Self, String> {
        let (format,bits)=match name {
            "auto"=>(None,0), "f32"=>(Some(Format::F32),0), "f16"=>(Some(Format::F16),0),
            "bf16"=>(Some(Format::Bf16),0), "q8_0"=>(Some(Format::Q8),0),
            "q4_0"=>(Some(Format::Q4),0), "q4_k"=>(Some(Format::Q4K),0),
            "mlx2"=>(Some(Format::Mlx),2), "mlx4"=>(Some(Format::Mlx),4),
            "mlx6"=>(Some(Format::Mlx),6), "mlx8"=>(Some(Format::Mlx),8),
            _=>return Err(format!("lfm2: unsupported residency quant {name:?}; use auto, f32/f16/bf16, q4_0/q4_k/q8_0, mlx2/4/6/8")),
        };
        if format == Some(Format::Mlx) && ![32, 64, 128].contains(&group) {
            return Err("lfm2: MLX group_size must be 32, 64 or 128".into());
        }
        Ok(Self {
            format,
            bits,
            group,
        })
    }
}

pub(super) struct Checkpoint {
    maps: Vec<Mmap>,
    names: HashMap<String, usize>,
    pub config: Value,
    pub prefix: &'static str,
    pub directory: PathBuf,
}
impl Checkpoint {
    /// Map checkpoint shards and validate unique tensor names against the shard index.
    pub fn open(path: &Path) -> Result<Self, String> {
        let dir = if path.is_dir() {
            path
        } else {
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."))
        };
        let config = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| format!("lfm2 config: {e}"))?,
        )
        .map_err(|e| format!("lfm2 config: {e}"))?;
        let index = dir.join("model.safetensors.index.json");
        let mapping: Option<HashMap<String, String>> = if index.exists() {
            let v: Value =
                serde_json::from_slice(&std::fs::read(&index).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            Some(
                serde_json::from_value(v["weight_map"].clone())
                    .map_err(|e| format!("lfm2 index: {e}"))?,
            )
        } else {
            None
        };
        let files = if let Some(mapping) = &mapping {
            let mut files: Vec<_> = mapping.values().cloned().collect();
            files.sort();
            files.dedup();
            if files.is_empty() {
                return Err("lfm2: empty checkpoint index".into());
            }
            files
                .into_iter()
                .map(|name| {
                    if Path::new(&name).components().count() != 1 || !name.ends_with(".safetensors")
                    {
                        Err(format!("lfm2: invalid shard filename {name:?}"))
                    } else {
                        Ok(dir.join(name))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        } else if path.is_file() {
            vec![path.to_path_buf()]
        } else if dir.join("model.safetensors").exists() {
            vec![dir.join("model.safetensors")]
        } else {
            crate::model::discover_shards(&dir.join("model.safetensors"))
        };
        let mut maps = Vec::new();
        let mut names = HashMap::new();
        for (i, path) in files.iter().enumerate() {
            let file = std::fs::File::open(path)
                .map_err(|e| format!("lfm2 open {}: {e}", path.display()))?;
            // Read-only mappings outlive every TensorView returned below.
            let map = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
            let st = SafeTensors::deserialize(&map)
                .map_err(|e| format!("lfm2 {}: {e}", path.display()))?;
            for name in st.names() {
                if names.insert(name.to_string(), i).is_some() {
                    return Err(format!("lfm2: duplicate tensor {name}"));
                }
                if let Some(mapping) = &mapping {
                    if mapping.get(name).map(String::as_str)
                        != path.file_name().and_then(|p| p.to_str())
                    {
                        return Err(format!(
                            "lfm2: tensor {name} disagrees with checkpoint index"
                        ));
                    }
                }
            }
            maps.push(map);
        }
        if let Some(mapping) = mapping {
            if mapping.keys().any(|k| !names.contains_key(k)) {
                return Err("lfm2: index references missing tensor".into());
            }
        }
        let prefix = if names.contains_key("language_model.model.embed_tokens.weight") {
            "language_model.model."
        } else {
            "model."
        };
        Ok(Self {
            maps,
            names,
            config,
            prefix,
            directory: dir.to_path_buf(),
        })
    }
    /// Borrow an indexed tensor by its complete checkpoint name.
    fn raw(&self, name: &str) -> Result<TensorView<'_>, String> {
        let i = *self
            .names
            .get(name)
            .ok_or_else(|| format!("lfm2: missing tensor {name}"))?;
        SafeTensors::deserialize(&self.maps[i])
            .map_err(|e| e.to_string())?
            .tensor(name)
            .map_err(|e| e.to_string())
    }
    /// Resolve a logical tensor name beneath the detected model prefix.
    fn get(&self, name: &str) -> Result<TensorView<'_>, String> {
        self.raw(&format!("{}{name}", self.prefix))
    }
    /// Test whether a prefixed logical tensor is present in the checkpoint.
    fn has(&self, name: &str) -> bool {
        self.names.contains_key(&format!("{}{name}", self.prefix))
    }
    /// Decode finite floating coefficients after checking the expected tensor shape.
    pub fn floats(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = self.get(name)?;
        if t.shape() != shape {
            return Err(format!(
                "{name}: unexpected shape/dtype {:?}/{:?}, expected {shape:?}",
                t.shape(),
                t.dtype()
            ));
        }
        float_values(&t, name)
    }
    /// Validate floating coefficients and upload their FP32 representation.
    pub fn upload_floats(
        &self,
        e: &mut ComputeEngine,
        name: &str,
        shape: &[usize],
    ) -> Result<Buffer, String> {
        upload(e, bytemuck::cast_slice(&self.floats(name, shape)?))
    }
    /// Read global quantization settings using the supported exporter keys.
    fn quantization(&self) -> &Value {
        self.config
            .get("quantization")
            .or_else(|| self.config.get("quantization_config"))
            .unwrap_or(&Value::Null)
    }
    /// Resolve matrix-specific metadata and indicate whether it overrides global settings.
    fn metadata(&self, name: &str) -> (&Value, bool) {
        let q = self.quantization();
        for key in [
            format!("{}{name}", self.prefix),
            format!("model.{name}"),
            name.to_string(),
        ] {
            if let Some(v) = q
                .get("tensor_formats")
                .and_then(|m| m.get(&key))
                .or_else(|| q.get(&key))
            {
                return (v, true);
            }
        }
        (q, false)
    }
    /// Validate one matrix encoding and coefficients before GPU upload or conversion.
    fn cpu_matrix(&self, name: &str, n: usize, k: usize) -> Result<CpuMatrix, String> {
        let t = self.get(&format!("{name}.weight"))?;
        let (q, per_matrix_metadata) = self.metadata(name);
        let mut c = CpuMatrix {
            format: Format::F32,
            bits: 0,
            group: 1,
            per_row: false,
            n,
            k,
            data: t.data().to_vec(),
            scales: vec![1.0],
            biases: vec![0.0],
        };
        let shape_error = || {
            format!(
                "{name}: unexpected shape/dtype {:?}/{:?} for [{n},{k}]",
                t.shape(),
                t.dtype()
            )
        };
        match t.dtype() {
            Dtype::F32 | Dtype::F16 | Dtype::BF16 => {
                if t.shape() != [n, k] {
                    return Err(shape_error());
                }
                float_values(&t, name)?; // reject non-finite floating weights
                c.format = match t.dtype() {
                    Dtype::F16 => Format::F16,
                    Dtype::BF16 => Format::Bf16,
                    _ => Format::F32,
                };
                // Global exporter metadata may describe quantized layers while
                // leaving embeddings/dense exceptions in their native dtype.
                if per_matrix_metadata && q["format"].as_str().is_some_and(|f| f != c.format.name())
                {
                    return Err(format!(
                        "{name}: floating shape/dtype disagrees with format metadata"
                    ));
                }
                if self.has(&format!("{name}.scales")) || self.has(&format!("{name}.weight_scale"))
                {
                    return Err(format!(
                        "{name}: floating weight shape/dtype is incompatible with quantization auxiliaries"
                    ));
                }
            }
            Dtype::U32 => {
                if per_matrix_metadata && q["format"].as_str().is_some_and(|f| f != "mlx") {
                    return Err(format!(
                        "{name}: packed MLX dtype disagrees with format metadata"
                    ));
                }
                let mode = q.get("mode").or_else(|| self.quantization().get("mode"));
                if t.shape().len() != 2
                    || t.shape()[0] != n
                    || mode.and_then(Value::as_str) != Some("affine")
                {
                    return Err(format!(
                        "{}; packed MLX requires affine metadata",
                        shape_error()
                    ));
                }
                let bits = t.shape()[1]
                    .checked_mul(32)
                    .filter(|v| v % k == 0)
                    .map(|v| v / k)
                    .unwrap_or(0);
                let s = self.get(&format!("{name}.scales"))?;
                let gs = if s.shape().len() == 2
                    && s.shape()[0] == n
                    && s.shape()[1] > 0
                    && k % s.shape()[1] == 0
                {
                    k / s.shape()[1]
                } else {
                    0
                };
                if ![2, 4, 6, 8].contains(&bits)
                    || ![32, 64, 128].contains(&gs)
                    || q["bits"].as_u64() != Some(bits as u64)
                    || q["group_size"].as_u64() != Some(gs as u64)
                {
                    return Err(format!(
                        "{name}: MLX shape/dtype or bits/group_size metadata mismatch"
                    ));
                }
                c.format = Format::Mlx;
                c.bits = bits;
                c.group = gs;
                c.scales = self.floats(&format!("{name}.scales"), &[n, k / gs])?;
                c.biases = self.floats(&format!("{name}.biases"), &[n, k / gs])?;
                let maximum = ((1u32 << bits) - 1) as f32;
                if c.scales
                    .iter()
                    .zip(&c.biases)
                    .any(|(&s, &b)| !(maximum * s + b).is_finite())
                {
                    return Err(format!("{name}: MLX dequantization overflow"));
                }
            }
            Dtype::F8_E4M3 | Dtype::U8 => {
                match q["format"].as_str() {
                    Some("q4_0") => c.format = Format::Q4,
                    Some("q4_k") => c.format = Format::Q4K,
                    Some("q8_0") => c.format = Format::Q8,
                    _ if self.has(&format!("{name}.weight_scale_2")) => c.format = Format::Nvfp4,
                    _ if t.dtype() == Dtype::F8_E4M3 || q["format"] == "fp8" => {
                        c.format = Format::Fp8
                    }
                    _ => return Err(format!("{name}: unsupported byte weight metadata")),
                }
                if per_matrix_metadata && q["format"].as_str().is_some_and(|f| f != c.format.name())
                {
                    return Err(format!(
                        "{name}: byte dtype/auxiliaries disagree with format metadata"
                    ));
                }
                if matches!(c.format, Format::Q4 | Format::Q4K | Format::Q8) {
                    if t.dtype() != Dtype::U8
                        || k % c.block_size() != 0
                        || t.shape() != [n, c.row_bytes()]
                    {
                        return Err(shape_error());
                    }
                    // Encoded half scales are validated by decoding bounded row chunks.
                    for start in (0..n).step_by(64) {
                        if c.decode_rows(start, (n - start).min(64))
                            .iter()
                            .any(|v| !v.is_finite())
                        {
                            return Err(format!("{name}: non-finite GGML block scale"));
                        }
                    }
                } else if c.format == Format::Nvfp4 {
                    if t.dtype() != Dtype::U8 || k % 16 != 0 || t.shape() != [n, k / 2] {
                        return Err(shape_error());
                    }
                    c.group = 16;
                    let scale = self.get(&format!("{name}.weight_scale"))?;
                    if scale.shape() != [n, k / 16] {
                        return Err(format!("{name}: NVFP4 scale shape mismatch"));
                    }
                    let values = if scale.dtype() == Dtype::F8_E4M3 {
                        if scale
                            .data()
                            .iter()
                            .any(|b| b & 0x80 != 0 || b & 0x7f == 0x7f)
                        {
                            return Err(format!("{name}: invalid NVFP4 block scale"));
                        }
                        scale
                            .data()
                            .iter()
                            .map(|&b| crate::model::e4m3_to_f32(b))
                            .collect()
                    } else {
                        float_values(&scale, name)?
                    };
                    let global = float_values(&self.get(&format!("{name}.weight_scale_2"))?, name)?;
                    if global.len() != 1 || global[0] < 0.0 || values.iter().any(|v| *v < 0.0) {
                        return Err(format!("{name}: invalid NVFP4 global/block scale"));
                    }
                    c.scales = values.into_iter().map(|v| v * global[0]).collect();
                } else {
                    if t.shape() != [n, k] {
                        return Err(shape_error());
                    }
                    let scale = self.get(&format!("{name}.weight_scale"))?;
                    c.scales = float_values(&scale, name)?;
                    if ![vec![], vec![1], vec![1, 1], vec![n], vec![n, 1]]
                        .contains(&scale.shape().to_vec())
                    {
                        return Err(format!(
                            "{name}: FP8 requires tensor or per-output-row scale"
                        ));
                    }
                    c.per_row = c.scales.len() == n;
                }
            }
            _ => return Err(shape_error()),
        }
        if c.scales.iter().any(|v| !v.is_finite()) {
            return Err(format!("{name}: non-finite folded scale"));
        }
        // Finite folded coefficients can still overflow when multiplied by codes.
        // Check bounded chunks even when auto residency bypasses transcoding.
        if matches!(c.format, Format::Fp8 | Format::Nvfp4) {
            for start in (0..n).step_by(64) {
                if c.decode_rows(start, (n - start).min(64))
                    .iter()
                    .any(|v| !v.is_finite())
                {
                    return Err(format!(
                        "{name}: non-finite decoded {} weight",
                        c.format.name()
                    ));
                }
            }
        }
        Ok(c)
    }
    /// Load, optionally transcode and upload a matrix within the device buffer limit.
    pub fn matrix(
        &self,
        e: &mut ComputeEngine,
        name: &str,
        n: usize,
        k: usize,
        target: Target,
        limit: usize,
    ) -> Result<Matrix, String> {
        let cpu = self.cpu_matrix(name, n, k)?.transcode(target)?;
        Matrix::upload(e, cpu, limit)
    }
}

/// Decode finite F32, F16 or BF16 tensor values into FP32 coefficients.
fn float_values(t: &TensorView<'_>, name: &str) -> Result<Vec<f32>, String> {
    let v: Vec<f32> = match t.dtype() {
        Dtype::F32 => t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        Dtype::F16 => t
            .data()
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::BF16 => t
            .data()
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
            .collect(),
        _ => return Err(format!("{name}: expected F32/F16/BF16 coefficients")),
    };
    if v.iter().any(|x| !x.is_finite()) {
        return Err(format!("{name}: non-finite floating values"));
    }
    Ok(v)
}

struct CpuMatrix {
    format: Format,
    bits: usize,
    group: usize,
    per_row: bool,
    n: usize,
    k: usize,
    data: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}
impl CpuMatrix {
    /// Return the column alignment required by the packed GGML codec.
    fn block_size(&self) -> usize {
        if self.format == Format::Q4K {
            256
        } else {
            32
        }
    }
    /// Calculate one packed matrix row without expanding its coefficients.
    fn row_bytes(&self) -> usize {
        match self.format {
            Format::F32 => self.k * 4,
            Format::F16 | Format::Bf16 => self.k * 2,
            Format::Mlx => self.k * self.bits / 8,
            Format::Q8 => self.k / 32 * 34,
            Format::Q4 => self.k / 32 * 18,
            Format::Q4K => self.k / 256 * 144,
            Format::Fp8 => self.k,
            Format::Nvfp4 => self.k / 2,
        }
    }
    /// Expand a bounded row range with the codec matching the stored format.
    fn decode_rows(&self, start: usize, rows: usize) -> Vec<f32> {
        let bytes = &self.data[start * self.row_bytes()..(start + rows) * self.row_bytes()];
        match self.format {
            Format::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Format::F16 => bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
                .collect(),
            Format::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
                .collect(),
            Format::Mlx => {
                let packed: Vec<u32> = bytes
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let g = self.k / self.group;
                crate::model::dequantize_mlx_affine_bits(
                    &packed,
                    &self.scales[start * g..(start + rows) * g],
                    &self.biases[start * g..(start + rows) * g],
                    rows,
                    self.k,
                    self.group,
                    self.bits,
                )
            }
            Format::Q4 => crate::model::dequant_q4_0_to_f32(bytes),
            Format::Q8 => crate::model::dequant_q8_0_to_f32(bytes),
            Format::Q4K => crate::model::dequant_q4_k_to_f32(bytes),
            Format::Fp8 => crate::model::dequantize_fp8(
                bytes,
                if self.per_row {
                    &self.scales[start..start + rows]
                } else {
                    &self.scales
                },
                rows,
                self.k,
            ),
            Format::Nvfp4 => (0..rows * self.k)
                .map(|i| {
                    let code = (bytes[i / 2] >> ((i % 2) * 4)) & 15;
                    crate::model::NVFP4_E2M1_LUT[code as usize]
                        * self.scales[start * (self.k / self.group) + i / self.group]
                })
                .collect(),
        }
    }
    /// Convert residency in bounded chunks and reject encoded or decoded overflow.
    fn transcode(self, target: Target) -> Result<Self, String> {
        let Some(format) = target.format else {
            return Ok(self);
        };
        let block = if format == Format::Q4K {
            256
        } else if matches!(format, Format::Q4 | Format::Q8) {
            32
        } else if format == Format::Mlx {
            target.group
        } else {
            1
        };
        if self.k % block != 0 {
            return Err(format!(
                "lfm2: {} requires column alignment {block}",
                format.name()
            ));
        }
        let mut out = Self {
            format,
            bits: target.bits,
            group: if format == Format::Mlx {
                target.group
            } else {
                1
            },
            per_row: false,
            n: self.n,
            k: self.k,
            data: Vec::new(),
            scales: Vec::new(),
            biases: Vec::new(),
        };
        for start in (0..self.n).step_by(128) {
            let rows = (self.n - start).min(128);
            let values = self.decode_rows(start, rows);
            if values.iter().any(|v| !v.is_finite()) {
                return Err("lfm2: non-finite decoded weight during residency conversion".into());
            }
            match format {
                Format::F32 => out.data.extend_from_slice(bytemuck::cast_slice(&values)),
                Format::F16 | Format::Bf16 => {
                    for &v in &values {
                        let bits = if format == Format::F16 {
                            let h = half::f16::from_f32(v);
                            if !h.is_finite() {
                                return Err("lfm2: F16 residency overflow".into());
                            }
                            h.to_bits()
                        } else {
                            let h = half::bf16::from_f32(v);
                            if !h.is_finite() {
                                return Err("lfm2: BF16 residency overflow".into());
                            }
                            h.to_bits()
                        };
                        out.data.extend_from_slice(&bits.to_le_bytes());
                    }
                }
                Format::Q4 => out.data.extend(crate::model::quantize_q4_0(&values)),
                Format::Q8 => out.data.extend(crate::model::quantize_q8_0(&values)),
                Format::Q4K => out.data.extend(crate::model::quantize_q4_k(&values)),
                Format::Mlx => {
                    for row in values.chunks_exact(self.k) {
                        let mut packed = vec![0u32; self.k * target.bits / 32];
                        for (g, group) in row.chunks_exact(target.group).enumerate() {
                            let lo = group.iter().copied().fold(f32::INFINITY, f32::min);
                            let hi = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let maximum = (1u32 << target.bits) - 1;
                            let scale = if hi == lo {
                                0.0
                            } else {
                                (hi - lo) / maximum as f32
                            };
                            if !scale.is_finite() {
                                return Err("lfm2: MLX residency scale overflow".into());
                            }
                            out.scales.push(scale);
                            out.biases.push(lo);
                            for (j, &v) in group.iter().enumerate() {
                                let q = if scale == 0.0 {
                                    0
                                } else {
                                    ((v - lo) / scale).round().clamp(0.0, maximum as f32) as u32
                                };
                                let bit = (g * target.group + j) * target.bits;
                                let wi = bit / 32;
                                let shift = bit % 32;
                                packed[wi] |= q << shift;
                                if shift + target.bits > 32 {
                                    packed[wi + 1] |= q >> (32 - shift);
                                }
                            }
                        }
                        out.data.extend_from_slice(bytemuck::cast_slice(&packed));
                    }
                }
                _ => unreachable!("Target::parse excludes native FP8/NVFP4 conversion"),
            }
        }
        if out.scales.is_empty() {
            out.scales.push(1.0);
            out.biases.push(0.0);
        }
        // Half-precision GGML block scales can overflow even for finite F32
        // inputs. Validate the encoded bytes before any GPU allocation.
        for start in (0..out.n).step_by(128) {
            if out
                .decode_rows(start, (out.n - start).min(128))
                .iter()
                .any(|v| !v.is_finite())
            {
                return Err(format!(
                    "lfm2: {} residency dequantization overflow",
                    format.name()
                ));
            }
        }
        Ok(out)
    }
    /// Report the resident format, including MLX bit width and group size.
    fn label(&self) -> String {
        if self.format == Format::Mlx {
            format!("mlx{}_g{}", self.bits, self.group)
        } else {
            self.format.name().into()
        }
    }
}

/// Reject empty or oversized requests before allocating a GPU storage buffer.
pub(super) fn allocate(e: &ComputeEngine, size: usize) -> Result<Buffer, String> {
    let limit = e.storage_buffer_limit() as usize;
    if size == 0 || size > limit {
        return Err(format!(
            "lfm2: storage buffer size {size} exceeds device limit {limit}"
        ));
    }
    e.alloc_device(size as u64)
}
/// Upload packed data while respecting alignment and storage-buffer limits.
pub(super) fn upload(e: &mut ComputeEngine, bytes: &[u8]) -> Result<Buffer, String> {
    let mut padded = Vec::new();
    let data = if bytes.len() % 4 == 0 {
        bytes
    } else {
        padded.extend_from_slice(bytes);
        padded.resize(bytes.len().div_ceil(4) * 4, 0);
        &padded
    };
    let b = allocate(e, data.len())?;
    e.upload(&b, data)?;
    Ok(b)
}
struct Part {
    w: Buffer,
    s: Buffer,
    b: Buffer,
    start: usize,
    rows: usize,
}
pub(super) struct Matrix {
    parts: Vec<Part>,
    format: Format,
    bits: usize,
    group: usize,
    per_row: bool,
    k: usize,
    label: String,
}
impl Matrix {
    /// Upload packed data while respecting alignment and storage-buffer limits.
    fn upload(e: &mut ComputeEngine, cpu: CpuMatrix, limit: usize) -> Result<Self, String> {
        // Raw byte buffers are padded to uint32 storage. Round the budget down
        // before choosing rows, so padding never exceeds the requested limit.
        let limit = limit.min(e.storage_buffer_limit() as usize) & !3;
        let mut rows = limit / cpu.row_bytes();
        let aux_per_row = if matches!(cpu.format, Format::Mlx | Format::Nvfp4) {
            4 * cpu.k / cpu.group
        } else if cpu.per_row {
            4
        } else {
            0
        };
        if aux_per_row > 0 {
            rows = rows.min(limit / aux_per_row);
        }
        if rows == 0 || limit < 4 {
            return Err(
                "lfm2: max_buffer_bytes cannot hold one matrix row and its auxiliaries".into(),
            );
        }
        let mut parts = Vec::new();
        for start in (0..cpu.n).step_by(rows) {
            let count = (cpu.n - start).min(rows);
            let (scales, biases) = if matches!(cpu.format, Format::Mlx | Format::Nvfp4) {
                let g = cpu.k / cpu.group;
                (
                    &cpu.scales[start * g..(start + count) * g],
                    if cpu.format == Format::Mlx {
                        &cpu.biases[start * g..(start + count) * g]
                    } else {
                        &cpu.biases[..]
                    },
                )
            } else if cpu.per_row {
                (&cpu.scales[start..start + count], &cpu.biases[..])
            } else {
                (&cpu.scales[..], &cpu.biases[..])
            };
            parts.push(Part {
                w: upload(
                    e,
                    &cpu.data[start * cpu.row_bytes()..(start + count) * cpu.row_bytes()],
                )?,
                s: upload(e, bytemuck::cast_slice(scales))?,
                b: upload(e, bytemuck::cast_slice(biases))?,
                start,
                rows: count,
            });
        }
        Ok(Self {
            parts,
            format: cpu.format,
            bits: cpu.bits,
            group: cpu.group,
            per_row: cpu.per_row,
            k: cpu.k,
            label: cpu.label(),
        })
    }
    /// Build the shader push constants for one matrix part and embedding token.
    fn pc(&self, part: &Part, token: usize) -> [u32; 8] {
        [
            self.k as u32,
            part.rows as u32,
            self.format as u32,
            self.bits as u32,
            self.group as u32,
            self.per_row as u32,
            part.start as u32,
            token as u32,
        ]
    }
    /// Report the resident format, including MLX bit width and group size.
    pub fn label(&self) -> &str {
        &self.label
    }
    /// Record packed matrix-vector projection dispatches and the resulting barrier.
    pub fn record(
        &self,
        e: &mut ComputeEngine,
        cb: vk::CommandBuffer,
        input: &Buffer,
        out: &Buffer,
    ) -> Result<(), String> {
        for part in &self.parts {
            if self.parts.len() == 1 && self.format == Format::Mlx && self.bits == 4 {
                e.record_to(
                    cb,
                    "mul_mat_vec_mlx4_f32_f32_r4",
                    &[&part.w, &part.s, &part.b, input, out],
                    bytemuck::cast_slice(&[
                        self.k as u32,
                        part.rows as u32,
                        self.group as u32,
                        0,
                        0,
                    ]),
                    (part.rows.div_ceil(4) as u32, 1, 1),
                )?;
            } else {
                let groups = part.rows.div_ceil(4);
                e.record_to(
                    cb,
                    "lfm2_matvec",
                    &[&part.w, &part.s, &part.b, input, out],
                    bytemuck::cast_slice(&self.pc(part, 0)),
                    (groups.min(256) as u32, groups.div_ceil(256) as u32, 1),
                )?;
            }
        }
        e.record_barrier_to(cb);
        Ok(())
    }
    /// Record the selected packed embedding row and synchronize subsequent reads.
    pub fn embed(
        &self,
        e: &mut ComputeEngine,
        cb: vk::CommandBuffer,
        token: u32,
        out: &Buffer,
    ) -> Result<(), String> {
        let part = self
            .parts
            .iter()
            .find(|p| (token as usize) >= p.start && (token as usize) < p.start + p.rows)
            .ok_or("lfm2: invalid embedding row")?;
        e.record_to(
            cb,
            "lfm2_embed",
            &[&part.w, &part.s, &part.b, out],
            bytemuck::cast_slice(&self.pc(part, token as usize - part.start)),
            (self.k.div_ceil(64) as u32, 1, 1),
        )?;
        e.record_barrier_to(cb);
        Ok(())
    }
}

/// Stage a conversion in an exclusive sibling directory until atomic publication.
struct ConversionOutput {
    path: PathBuf,
    complete: bool,
}

impl ConversionOutput {
    /// Reserve a unique sibling staging directory without claiming the destination.
    fn create(destination: &Path) -> Result<Self, String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);

        match std::fs::symlink_metadata(destination) {
            Ok(_) => return Err("lfm2: conversion destination must not exist".into()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        for _ in 0..1024 {
            let path = parent.join(format!(
                ".lfm2-conversion-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        complete: false,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
        Err("lfm2: cannot create a unique conversion staging directory".into())
    }

    /// Publish all shards together, preserving even an empty concurrent destination.
    fn publish(mut self, destination: &Path) -> Result<(), String> {
        publish_conversion(&self.path, destination)?;
        self.complete = true;
        Ok(())
    }
}

impl Drop for ConversionOutput {
    /// Remove unpublished staging data while leaving the requested destination untouched.
    fn drop(&mut self) {
        if !self.complete {
            if let Err(error) = std::fs::remove_dir_all(&self.path) {
                log::warn!(
                    "lfm2: cannot remove incomplete conversion {}: {error}",
                    self.path.display()
                );
            }
        }
    }
}

/// Rename without replacement; a check followed by plain rename has a race.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn publish_conversion(source: &Path, destination: &Path) -> Result<(), String> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let source = CString::new(source.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
    let destination =
        CString::new(destination.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
    // SAFETY: Both C strings remain valid and NUL-terminated for the syscall.
    // Exclusive rename publishes atomically and cannot replace another writer.
    let result = unsafe {
        #[cfg(target_os = "linux")]
        {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                destination.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        }
        #[cfg(target_os = "macos")]
        {
            libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL)
        }
    };
    if result == 0 {
        Ok(())
    } else {
        Err(format!(
            "lfm2: cannot publish converted checkpoint: {}",
            std::io::Error::last_os_error()
        ))
    }
}

/// Reject export on platforms lacking an implemented exclusive rename.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn publish_conversion(_source: &Path, _destination: &Path) -> Result<(), String> {
    Err("lfm2: atomic checkpoint export requires Linux or macOS".into())
}

/// Export one logical matrix per safetensors shard. The same canonical CPU
/// encoders used for residency conversion produce inspectable packed bytes;
/// the independent oracle decodes those bytes without calling this module.
pub(super) fn convert_checkpoint(
    source: &Path,
    destination: &Path,
    quant: &str,
    group_size: usize,
    specs: &[(String, usize, usize)],
) -> Result<(), String> {
    let ck = Checkpoint::open(source)?;
    let target = Target::parse(quant, group_size)?;
    let output = ConversionOutput::create(destination)?;
    let staging = &output.path;
    let mut weight_map = HashMap::new();
    let mut metadata = serde_json::Map::new();
    for (i, (name, n, k)) in specs.iter().enumerate() {
        let cpu = ck.cpu_matrix(name, *n, *k)?.transcode(target)?;
        let root = format!("{}{name}", ck.prefix);
        let mut tensors = HashMap::new();
        let (dtype, shape) = match cpu.format {
            Format::F32 => (Dtype::F32, vec![*n, *k]),
            Format::F16 => (Dtype::F16, vec![*n, *k]),
            Format::Bf16 => (Dtype::BF16, vec![*n, *k]),
            Format::Mlx => (Dtype::U32, vec![*n, k * cpu.bits / 32]),
            Format::Fp8 => (Dtype::F8_E4M3, vec![*n, *k]),
            _ => (Dtype::U8, vec![*n, cpu.row_bytes()]),
        };
        tensors.insert(format!("{root}.weight"), (dtype, shape, cpu.data.clone()));
        let info = if cpu.format == Format::Mlx {
            tensors.insert(
                format!("{root}.scales"),
                (
                    Dtype::F32,
                    vec![*n, k / cpu.group],
                    bytemuck::cast_slice(&cpu.scales).to_vec(),
                ),
            );
            tensors.insert(
                format!("{root}.biases"),
                (
                    Dtype::F32,
                    vec![*n, k / cpu.group],
                    bytemuck::cast_slice(&cpu.biases).to_vec(),
                ),
            );
            serde_json::json!({"format":"mlx","mode":"affine","bits":cpu.bits,"group_size":cpu.group})
        } else if matches!(cpu.format, Format::Fp8 | Format::Nvfp4) {
            let shape = if cpu.format == Format::Nvfp4 {
                vec![*n, k / cpu.group]
            } else if cpu.per_row {
                vec![*n]
            } else {
                vec![1]
            };
            tensors.insert(
                format!("{root}.weight_scale"),
                (
                    Dtype::F32,
                    shape,
                    bytemuck::cast_slice(&cpu.scales).to_vec(),
                ),
            );
            if cpu.format == Format::Nvfp4 {
                tensors.insert(
                    format!("{root}.weight_scale_2"),
                    (Dtype::F32, vec![1], 1f32.to_le_bytes().to_vec()),
                );
            }
            serde_json::json!({"format":cpu.format.name()})
        } else {
            serde_json::json!({"format":cpu.format.name()})
        };
        metadata.insert(root, info);
        let file = format!("model-{i:05}.safetensors");
        write_tensors(&staging.join(&file), &tensors)?;
        for name in tensors.keys() {
            weight_map.insert(name.clone(), file.clone());
        }
    }
    let roots: Vec<String> = specs
        .iter()
        .map(|(n, _, _)| format!("{}{n}.", ck.prefix))
        .collect();
    let mut small = HashMap::new();
    for name in ck
        .names
        .keys()
        .filter(|n| !roots.iter().any(|root| n.starts_with(root)))
    {
        let t = ck.raw(name)?;
        small.insert(
            name.clone(),
            (t.dtype(), t.shape().to_vec(), t.data().to_vec()),
        );
    }
    if !small.is_empty() {
        let file = "model-small.safetensors";
        write_tensors(&staging.join(file), &small)?;
        for name in small.keys() {
            weight_map.insert(name.clone(), file.to_string());
        }
    }
    let mut config = ck.config.clone();
    config
        .as_object_mut()
        .ok_or("lfm2: config must be an object")?
        .remove("quantization");
    config["quantization_config"] =
        serde_json::json!({"quant_method":"vllm_vulkan","tensor_formats":metadata});
    std::fs::write(
        staging.join("config.json"),
        serde_json::to_vec_pretty(&config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        staging.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"weight_map":weight_map}))
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
    ] {
        let path = ck.directory.join(name);
        if path.exists() {
            std::fs::copy(path, staging.join(name)).map_err(|e| e.to_string())?;
        }
    }
    output.publish(destination)
}

/// Serialize validated tensor views into a single safetensors shard.
fn write_tensors(
    path: &Path,
    tensors: &HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
) -> Result<(), String> {
    let views: Vec<_> = tensors
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            TensorView::new(*dtype, shape.clone(), bytes).map(|v| (name.clone(), v))
        })
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    safetensors::tensor::serialize_to_file(views, &None, path).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Create a private RAII workspace for conversion-publication regression tests.
    fn conversion_workspace() -> ConversionOutput {
        let destination = std::env::temp_dir().join(format!(
            "lfm2-output-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        ConversionOutput::create(&destination).unwrap()
    }

    /// Keep another writer's destination intact when an incomplete conversion drops.
    #[test]
    fn failed_conversion_cleanup_preserves_replaced_destination() {
        let root = conversion_workspace();
        let destination = root.path.join("converted");
        let output = ConversionOutput::create(&destination).unwrap();
        // Model an external writer replacing the destination before failure.
        if destination.exists() {
            std::fs::remove_dir_all(&destination).unwrap();
        }
        std::fs::create_dir(&destination).unwrap();
        let sentinel = destination.join("keep.txt");
        std::fs::write(&sentinel, b"other writer").unwrap();
        drop(output);
        let preserved = std::fs::read(&sentinel);
        assert_eq!(preserved.unwrap(), b"other writer");
    }

    /// Expose all staged output files together only after successful publication.
    #[test]
    fn conversion_is_visible_only_after_complete_publication() {
        let root = conversion_workspace();
        let destination = root.path.join("converted");
        let output = ConversionOutput::create(&destination).unwrap();
        let staging = output.path.clone();
        std::fs::write(staging.join("shard"), b"weights").unwrap();
        std::fs::write(staging.join("config.json"), b"{}").unwrap();
        assert!(!destination.exists());
        output.publish(&destination).unwrap();
        assert!(!staging.exists());
        assert_eq!(
            std::fs::read(destination.join("shard")).unwrap(),
            b"weights"
        );
        assert_eq!(
            std::fs::read(destination.join("config.json")).unwrap(),
            b"{}"
        );
    }

    /// Reject publication over concurrent directories, files and dangling symlinks.
    #[cfg(unix)]
    #[test]
    fn conversion_publication_preserves_concurrent_destinations() {
        let root = conversion_workspace();
        for kind in ["empty_directory", "directory", "file", "symlink"] {
            let destination = root.path.join(kind);
            let output = ConversionOutput::create(&destination).unwrap();
            let staging = output.path.clone();
            std::fs::write(staging.join("shard"), b"our weights").unwrap();
            match kind {
                "empty_directory" => std::fs::create_dir(&destination).unwrap(),
                "directory" => {
                    std::fs::create_dir(&destination).unwrap();
                    std::fs::write(destination.join("keep"), b"other writer").unwrap();
                }
                "file" => std::fs::write(&destination, b"other writer").unwrap(),
                _ => std::os::unix::fs::symlink("missing-target", &destination).unwrap(),
            }
            assert!(output.publish(&destination).is_err(), "{kind}");
            assert!(!staging.exists(), "{kind}: staging was not cleaned");
            match kind {
                "empty_directory" => {
                    assert_eq!(std::fs::read_dir(&destination).unwrap().count(), 0);
                }
                "directory" => assert_eq!(
                    std::fs::read(destination.join("keep")).unwrap(),
                    b"other writer"
                ),
                "file" => assert_eq!(std::fs::read(&destination).unwrap(), b"other writer"),
                _ => assert_eq!(
                    std::fs::read_link(&destination).unwrap(),
                    Path::new("missing-target")
                ),
            }
        }
    }

    /// Allow one staged conversion to publish while preserving it from a second writer.
    #[test]
    fn concurrent_conversions_have_one_publication_winner() {
        let root = conversion_workspace();
        let destination = root.path.join("converted");
        let first = ConversionOutput::create(&destination).unwrap();
        let second = ConversionOutput::create(&destination).unwrap();
        assert_ne!(first.path, second.path);
        std::fs::write(first.path.join("shard"), b"first").unwrap();
        std::fs::write(second.path.join("shard"), b"second").unwrap();
        first.publish(&destination).unwrap();
        let staging = second.path.clone();
        assert!(second.publish(&destination).is_err());
        assert!(!staging.exists());
        assert_eq!(std::fs::read(destination.join("shard")).unwrap(), b"first");
    }
    /// Validate residency names and reject incompatible packed column alignments.
    #[test]
    fn residency_is_explicit_and_alignment_checked() {
        for name in [
            "auto", "f32", "f16", "bf16", "q4_0", "q4_k", "q8_0", "mlx2", "mlx4", "mlx6", "mlx8",
        ] {
            assert!(Target::parse(name, 64).is_ok());
        }
        for name in ["Q4_K", "q8", "fp8", "nvfp4", "made_up"] {
            assert!(Target::parse(name, 64).is_err());
        }
        assert!(Target::parse("mlx4", 48).is_err());
        let c = CpuMatrix {
            format: Format::F32,
            bits: 0,
            group: 1,
            per_row: false,
            n: 1,
            k: 64,
            data: vec![0; 256],
            scales: vec![1.],
            biases: vec![0.],
        };
        assert!(c
            .transcode(Target::parse("q4_k", 64).unwrap())
            .err()
            .unwrap()
            .contains("alignment"));
    }
    /// Check finite codec round trips including cross-word MLX packed values.
    #[test]
    fn mlx_cross_word_conversion_and_block_codecs_are_finite() {
        let values: Vec<f32> = (0..1024)
            .map(|i| ((i * 31 % 101) as f32 - 50.) / 57.)
            .collect();
        for name in [
            "f32", "f16", "bf16", "q4_0", "q4_k", "q8_0", "mlx2", "mlx4", "mlx6", "mlx8",
        ] {
            let c = CpuMatrix {
                format: Format::F32,
                bits: 0,
                group: 1,
                per_row: false,
                n: 4,
                k: 256,
                data: bytemuck::cast_slice(&values).to_vec(),
                scales: vec![1.],
                biases: vec![0.],
            };
            let q = c.transcode(Target::parse(name, 64).unwrap()).unwrap();
            let decoded = q.decode_rows(0, 4);
            assert_eq!(decoded.len(), 1024);
            assert!(decoded.iter().all(|v| v.is_finite()));
            let error: f32 = decoded
                .iter()
                .zip(&values)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            assert!(error < 30., "{name}: {error}");
        }
    }
    /// Reject conversions whose encoded scales overflow despite finite input weights.
    #[test]
    fn encoded_scale_overflow_is_rejected() {
        for name in ["f16", "bf16", "q4_0", "q4_k", "q8_0", "mlx2"] {
            let values: Vec<f32> = (0..256)
                .map(|i| if i % 2 == 0 { f32::MAX } else { -f32::MAX })
                .collect();
            let c = CpuMatrix {
                format: Format::F32,
                bits: 0,
                group: 1,
                per_row: false,
                n: 1,
                k: 256,
                data: bytemuck::cast_slice(&values).to_vec(),
                scales: vec![1.],
                biases: vec![0.],
            };
            assert!(
                c.transcode(Target::parse(name, 64).unwrap()).is_err(),
                "{name}"
            );
        }
    }
}
