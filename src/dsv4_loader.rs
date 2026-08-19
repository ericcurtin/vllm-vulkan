//! DeepSeek-V4-Flash — real sharded MLX-affine checkpoint reader ([`Dsv4Src`]).
//!
//! Streams individual (or single-expert-sliced) tensors out of the 18-shard
//! `.safetensors` checkpoint and dequantizes on demand, mirroring
//! `golden.py::RealSource` / `dsv4_common.py`. Quantized linears/experts use the
//! validated [`crate::model::dequantize_mlx_affine_bits`] (2/6/8-bit, contiguous
//! LSB-first bitstream). This is the CPU-resident-per-tensor source for the M1
//! single-node full-forward gate; the M2 GPU-resident streaming loader
//! (`Dsv4GpuStage::from_ckpt_streamed`) will reuse the same name map + dequant.
//!
//! Layout (from the real checkpoint headers):
//!   * MLA / DSA linears  — U32 packed, BF16 scales/biases, 6-bit gs128
//!   * embed / lm_head / shared_experts — 8-bit gs64
//!   * routed experts (3D `[E,·,·]`) — 2-bit gs128
//!   * norms / router gate.weight / q_norm — plain BF16 ; sinks / hc / corr_bias — F32
//!   * hash tid2eid — I32

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::dsv4_forward::Dsv4Src;
use crate::model::dequantize_mlx_affine_bits;

struct Loc {
    shard: String,
    dtype: String,
    shape: Vec<usize>,
    start: usize, // absolute byte offset in the shard file
    end: usize,
}

/// A still-packed MLX-affine quantized linear, widened to what the GPU
/// `mul_mat_vec_mlx{2,6,8}` kernels bind (`packed` u32 words verbatim, `scales`/
/// `biases` bf16→f32). `bits` ∈ {2,6,8}; `gs` = group size (128 for 2/6-bit,
/// 64 for 8-bit). See [`Dsv4RealSrc::raw_linear`].
pub struct RawQ {
    pub packed: Vec<u32>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
    pub out: usize,
    pub inn: usize,
    pub bits: usize,
    pub gs: usize,
}

pub struct Dsv4RealSrc {
    index: HashMap<String, Loc>,
    mmaps: HashMap<String, Mmap>,
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}
fn f16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}
fn f32_of(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn u32_of(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

impl Dsv4RealSrc {
    pub fn open(dir: &str) -> Result<Dsv4RealSrc, String> {
        let dir = PathBuf::from(dir);
        let idx_path = dir.join("model.safetensors.index.json");
        let idx_raw = std::fs::read_to_string(&idx_path).map_err(|e| format!("index.json: {e}"))?;
        let idx: serde_json::Value = serde_json::from_str(&idx_raw).map_err(|e| format!("{e}"))?;
        let wm = idx["weight_map"].as_object().ok_or("no weight_map")?;
        let mut shard_files: Vec<String> = wm.values().filter_map(|v| v.as_str().map(String::from)).collect();
        shard_files.sort();
        shard_files.dedup();

        let mut mmaps = HashMap::new();
        let mut index: HashMap<String, Loc> = HashMap::new();
        for shard in &shard_files {
            let path = dir.join(shard);
            let file = std::fs::File::open(&path).map_err(|e| format!("open {shard}: {e}"))?;
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {shard}: {e}"))?;
            // header: u64 LE length, then JSON; data starts at 8 + len.
            let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header: serde_json::Value =
                serde_json::from_slice(&mmap[8..8 + hlen]).map_err(|e| format!("header {shard}: {e}"))?;
            let data_base = 8 + hlen;
            for (name, meta) in header.as_object().ok_or("bad header")? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = meta["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> =
                    meta["shape"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap() as usize).collect()).unwrap_or_default();
                let off = meta["data_offsets"].as_array().ok_or("no offsets")?;
                let start = data_base + off[0].as_u64().unwrap() as usize;
                let end = data_base + off[1].as_u64().unwrap() as usize;
                index.insert(name.clone(), Loc { shard: shard.clone(), dtype, shape, start, end });
            }
            mmaps.insert(shard.clone(), mmap);
        }
        Ok(Dsv4RealSrc { index, mmaps })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// True when `name` is an MLX-affine QUANTIZED linear (has a `.scales` sibling),
    /// vs a plain bf16/f32 tensor. Lets the GPU-resident stage decide whether to
    /// upload packed words (`raw_linear`) or fall back to host dequant.
    pub fn is_quant(&self, name: &str) -> bool {
        self.index.contains_key(&format!("{name}.scales"))
    }

    /// Raw (still-packed) 2D quantized linear `name.{weight,scales,biases}` → the
    /// exact `(packed u32, scales f32, biases f32, bits, gs)` a `mul_mat_vec_mlx{2,6,8}`
    /// kernel consumes GPU-resident (scales/biases widened bf16→f32, as the shaders
    /// read `float[]`). `bits`/`gs` are inferred from the header shapes exactly as
    /// `dequant2d` does, so the resident matvec is bit-faithful to the CPU dequant.
    pub fn raw_linear(&self, name: &str, out_f: usize, in_f: usize) -> RawQ {
        let packed = u32_of(self.bytes(&format!("{name}.weight")));
        let scales = bf16_to_f32(self.bytes(&format!("{name}.scales")));
        let biases = bf16_to_f32(self.bytes(&format!("{name}.biases")));
        let packed_cols = packed.len() / out_f;
        let scale_cols = scales.len() / out_f;
        let bits = packed_cols * 32 / in_f;
        let gs = in_f / scale_cols;
        RawQ { packed, scales, biases, out: out_f, inn: in_f, bits, gs }
    }

    /// Raw single-expert `e` slice of a 3D `[E,out,packed]` switch tensor (2-bit
    /// routed experts), same widened layout as `raw_linear`. Mirrors `expert()`.
    pub fn raw_expert(&self, name: &str, e: usize, out_f: usize, in_f: usize) -> RawQ {
        let wloc = self.index.get(&format!("{name}.weight")).unwrap_or_else(|| panic!("missing expert {name}.weight"));
        let packed_cols = wloc.shape[2];
        let groups = self.index[&format!("{name}.scales")].shape[2];
        let per_e_w = out_f * packed_cols;
        let per_e_s = out_f * groups;
        let wb = self.bytes(&format!("{name}.weight"));
        let packed = u32_of(&wb[e * per_e_w * 4..(e + 1) * per_e_w * 4]);
        let sb = self.bytes(&format!("{name}.scales"));
        let scales = bf16_to_f32(&sb[e * per_e_s * 2..(e + 1) * per_e_s * 2]);
        let bb = self.bytes(&format!("{name}.biases"));
        let biases = bf16_to_f32(&bb[e * per_e_s * 2..(e + 1) * per_e_s * 2]);
        let bits = packed_cols * 32 / in_f;
        let gs = in_f / groups;
        RawQ { packed, scales, biases, out: out_f, inn: in_f, bits, gs }
    }

    fn bytes(&self, name: &str) -> &[u8] {
        let loc = self.index.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        &self.mmaps[&loc.shard][loc.start..loc.end]
    }

    /// Decode a plain (non-quantized) tensor → f32.
    fn plain_f32(&self, name: &str) -> Vec<f32> {
        let loc = self.index.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        let b = &self.mmaps[&loc.shard][loc.start..loc.end];
        match loc.dtype.as_str() {
            "BF16" => bf16_to_f32(b),
            "F16" => f16_to_f32(b),
            "F32" => f32_of(b),
            d => panic!("plain_f32 unsupported dtype {d} for {name}"),
        }
    }

    /// Dequantize a 2D quantized linear `prefix.{weight,scales,biases}` → f32 [out,in].
    fn dequant2d(&self, prefix: &str, out_f: usize, in_f: usize) -> Vec<f32> {
        let packed = u32_of(self.bytes(&format!("{prefix}.weight")));
        let scales = bf16_to_f32(self.bytes(&format!("{prefix}.scales")));
        let biases = bf16_to_f32(self.bytes(&format!("{prefix}.biases")));
        let packed_cols = packed.len() / out_f;
        let scale_cols = scales.len() / out_f;
        let bits = packed_cols * 32 / in_f;
        let gs = in_f / scale_cols;
        debug_assert_eq!((in_f * bits + 31) / 32, packed_cols, "{prefix} bits infer");
        dequantize_mlx_affine_bits(&packed, &scales, &biases, out_f, in_f, gs, bits)
    }
}

impl Dsv4Src for Dsv4RealSrc {
    fn linear(&self, name: &str, out_f: usize, in_f: usize) -> Vec<f32> {
        if self.index.contains_key(&format!("{name}.scales")) {
            return self.dequant2d(name, out_f, in_f);
        }
        // plain (e.g. bf16 router gate); tensor may be `name` or `name.weight`.
        let key = if self.index.contains_key(name) { name.to_string() } else { format!("{name}.weight") };
        let v = self.plain_f32(&key);
        assert_eq!(v.len(), out_f * in_f, "plain linear {name} shape {} vs {}x{}", v.len(), out_f, in_f);
        v
    }

    fn dense(&self, name: &str) -> Vec<f32> {
        let key = if self.index.contains_key(name) { name.to_string() } else { format!("{name}.weight") };
        self.plain_f32(&key)
    }

    fn expert(&self, name: &str, e: usize, out_f: usize, in_f: usize) -> Vec<f32> {
        // 3D packed [E, out, packed_cols]; scales/biases [E, out, groups].
        let wloc = self.index.get(&format!("{name}.weight")).unwrap_or_else(|| panic!("missing expert {name}.weight"));
        let packed_cols = wloc.shape[2];
        let groups = self.index[&format!("{name}.scales")].shape[2];
        let per_e_w = out_f * packed_cols; // u32 elems
        let per_e_s = out_f * groups; // scale elems
        let wb = self.bytes(&format!("{name}.weight"));
        let packed = u32_of(&wb[e * per_e_w * 4..(e + 1) * per_e_w * 4]);
        let sb = self.bytes(&format!("{name}.scales"));
        let scales = bf16_to_f32(&sb[e * per_e_s * 2..(e + 1) * per_e_s * 2]);
        let bb = self.bytes(&format!("{name}.biases"));
        let biases = bf16_to_f32(&bb[e * per_e_s * 2..(e + 1) * per_e_s * 2]);
        let bits = packed_cols * 32 / in_f;
        let gs = in_f / groups;
        dequantize_mlx_affine_bits(&packed, &scales, &biases, out_f, in_f, gs, bits)
    }

    fn dense_i64(&self, name: &str) -> Vec<i64> {
        let loc = self.index.get(name).unwrap_or_else(|| panic!("missing i64 {name}"));
        let b = &self.mmaps[&loc.shard][loc.start..loc.end];
        match loc.dtype.as_str() {
            "I32" => b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
            "I64" => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            d => panic!("dense_i64 unsupported dtype {d} for {name}"),
        }
    }

    /// Dequantize only the needed rows of the 8-bit gs64 embedding table.
    fn embed_rows(&self, ids: &[u32], vocab: usize, h: usize) -> Vec<f32> {
        let _ = vocab;
        let name = "model.embed_tokens";
        let wloc = self.index.get(&format!("{name}.weight")).unwrap_or_else(|| panic!("no embed"));
        let packed_cols = wloc.shape[1];
        let groups = self.index[&format!("{name}.scales")].shape[1];
        let bits = packed_cols * 32 / h;
        let gs = h / groups;
        let wb = self.bytes(&format!("{name}.weight"));
        let sb = self.bytes(&format!("{name}.scales"));
        let bb = self.bytes(&format!("{name}.biases"));
        let mut out = vec![0.0f32; ids.len() * h];
        for (i, &t) in ids.iter().enumerate() {
            let t = t as usize;
            let packed = u32_of(&wb[t * packed_cols * 4..(t + 1) * packed_cols * 4]);
            let scales = bf16_to_f32(&sb[t * groups * 2..(t + 1) * groups * 2]);
            let biases = bf16_to_f32(&bb[t * groups * 2..(t + 1) * groups * 2]);
            let row = dequantize_mlx_affine_bits(&packed, &scales, &biases, 1, h, gs, bits);
            out[i * h..(i + 1) * h].copy_from_slice(&row);
        }
        out
    }
}

/// Convenience: does the checkpoint dir look like a DSV4 checkpoint?
pub fn is_dsv4_dir(dir: &str) -> bool {
    Path::new(dir).join("model.safetensors.index.json").exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const DIR: &str = "/Volumes/Shared_Drive/models/DeepSeek-V4-Flash-0731-2.4bit-mixed";

    fn approx(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label} len");
        let e = a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(e < tol, "{label} max_abs_err {e:.3e}");
    }

    /// Validate the Rust reader vs golden.py::RealSource on a few real tensors
    /// (6/8/2-bit + plain + embed rows). Skips if the checkpoint isn't present.
    #[test]
    fn loader_matches_realsource() {
        if !is_dsv4_dir(DIR) {
            eprintln!("SKIP: checkpoint not present");
            return;
        }
        let probe: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/loader_probe.json")).unwrap(),
        ).unwrap();
        let f = |v: &Value| -> Vec<f32> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
        let src = Dsv4RealSrc::open(DIR).unwrap();

        // 6-bit MLA
        let w = src.linear("model.layers.0.attn.wq_a", 1024, 4096);
        assert!((w.iter().sum::<f32>() - probe["wq_a"]["sum"].as_f64().unwrap() as f32).abs() < 0.05, "wq_a sum");
        approx(&w[0..6], &f(&probe["wq_a"]["r0c"]), 1e-3, "wq_a r0");
        approx(&w[5 * 4096 + 100..5 * 4096 + 106], &f(&probe["wq_a"]["r5c"]), 1e-3, "wq_a r5");
        // 8-bit shared
        let w = src.linear("model.layers.0.ffn.shared_experts.gate_proj", 2048, 4096);
        approx(&w[0..6], &f(&probe["shared_g"]["r0c"]), 1e-3, "shared_g r0");
        // 2-bit routed expert
        let w = src.expert("model.layers.0.ffn.switch_mlp.gate_proj", 5, 2048, 4096);
        approx(&w[0..6], &f(&probe["exp5_g"]["r0c"]), 1e-3, "exp5_g r0");
        let w = src.expert("model.layers.0.ffn.switch_mlp.down_proj", 5, 4096, 2048);
        approx(&w[0..6], &f(&probe["exp5_d"]["r0c"]), 1e-3, "exp5_d r0");
        // embed rows
        let ids: Vec<u32> = probe["embed"]["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let er = src.embed_rows(&ids, 129280, 4096);
        for (k, row) in probe["embed"]["rows"].as_array().unwrap().iter().enumerate() {
            approx(&er[k * 4096..k * 4096 + 6], &f(row), 1e-3, "embed row");
        }
        // plain bf16 + f32
        approx(&src.dense("model.layers.0.attn.q_norm.weight")[0..6], &f(&probe["q_norm"]), 1e-3, "q_norm");
        approx(&src.dense("model.layers.0.attn.attn_sink")[0..6], &f(&probe["sink"]), 1e-6, "sink");
        eprintln!("loader_matches_realsource: OK");
    }

    /// ★ M1 GATE: full 43-layer CPU-resident forward on the REAL 86GB checkpoint
    /// reproduces the golden argmax == 11111 (` Paris`). Streams per-layer dequant.
    /// `#[ignore]` (heavy); run with:
    ///   cargo test --lib dsv4_loader::tests::m1_gate_full_forward_argmax_11111 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn m1_gate_full_forward_argmax_11111() {
        use crate::dsv4::Dsv4Config;
        use crate::dsv4_forward::dsv4_forward;
        if !is_dsv4_dir(DIR) {
            eprintln!("SKIP: checkpoint not present");
            return;
        }
        let cfg_j: Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv4/real_config.json")).unwrap(),
        ).unwrap();
        let cfg = Dsv4Config::from_json(&cfg_j).unwrap();
        let input_ids: Vec<u32> = vec![671, 6102, 294, 8760, 344]; // BOS + "The capital of France is"
        let t0 = std::time::Instant::now();
        let src = Dsv4RealSrc::open(DIR).unwrap();
        let logits = dsv4_forward(&cfg, &input_ids, &src);
        let vocab = cfg.vocab_size;
        let last = &logits[(input_ids.len() - 1) * vocab..];
        let finite = last.iter().all(|x| x.is_finite());
        // argmax + top-10
        let mut idxs: Vec<usize> = (0..vocab).collect();
        idxs.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap_or(std::cmp::Ordering::Equal));
        let argmax = idxs[0];
        eprintln!(
            "M1 GATE: finite={finite} argmax={argmax} (expect 11111)  top10={:?}  wall={:.1}s",
            &idxs[..10],
            t0.elapsed().as_secs_f32()
        );
        assert!(finite, "logits not finite");
        assert_eq!(argmax, 11111, "argmax {argmax} != golden 11111 (' Paris')");
    }
}
