//! Little-endian decoders for the plain (non-quantized) safetensors dtypes,
//! shared by the per-arch loaders (nemotron `decode_plain`, the dsv4 mmap
//! loader). Always compiled, so a single-arch feature build still has them.

/// BF16 bytes -> f32.
pub fn bf16_le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// F16 bytes -> f32.
pub fn f16_le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// F32 bytes -> f32.
pub fn f32_le(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// A plain BF16 / F16 / F32 tensor's bytes -> f32; any other dtype is an error.
pub fn decode_plain(dtype: safetensors::Dtype, b: &[u8]) -> Result<Vec<f32>, String> {
    Ok(match dtype {
        safetensors::Dtype::BF16 => bf16_le_to_f32(b),
        safetensors::Dtype::F16 => f16_le_to_f32(b),
        safetensors::Dtype::F32 => f32_le(b),
        other => return Err(format!("unsupported plain safetensors dtype {other:?}")),
    })
}
