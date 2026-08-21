// SPDX-License-Identifier: Apache-2.0
//! Shared attention-KV prefix serialize/restore primitive — the surface built
//! ONCE for BOTH the NAS-backed prefix cache (`docs/nas-prefix-cache-*.md`) and
//! the session-KV-continuation sibling (`docs/session-kv-continuation-*.md`).
//!
//! ## What this is
//!
//! `qwen35::export_prefix`/`import_prefix` (the `PFX1` blob) is the TEMPLATE but
//! is qwen3_5-RECURRENT-only (its per-layer body is a fixed-size DeltaNet state,
//! and it serializes an ENTIRE PP stage into one opaque blob keyed by topology).
//! This module generalizes the seam to **attention-KV** archs (gemma, Laguna)
//! with two deliberate changes the scope pins:
//!
//! 1. **Canonical `(layer, kv_head)` tiles, topology-agnostic** (scope §4.1). A
//!    tile is addressed by its ABSOLUTE `(layer, kv_head)` — never by which PP
//!    stage / TP rank happens to hold it — so a prefix warmed under any layout
//!    restores under any other (PP is free; cross-TP-degree gated on-cluster).
//!    Each impl serializes only the tiles it OWNS right now; the store is the
//!    union of everyone's tiles (`kvstore::tile_key`).
//! 2. **`TILE1` blob body** — one length-prefixed `(K, V)` pair for one tile,
//!    dtype-tagged (f32 today, f16 the NAS default, int8 a later gated phase).
//!    Every section length-prefixed so a truncated/corrupt tile is a clean
//!    `Err`, never a bad memcpy (same discipline as `qwen35::import_prefix`).
//!
//! ## Bit-exactness contract
//!
//! - Full-attention layers store rows `[0, upto)`.
//! - Sliding-window layers store the `window`-bounded slice `[max(0, upto-window),
//!   upto)` with the absolute base position (`window_base`) in the header. Both
//!   gemma (`KvCache`) and Laguna (`ResidentKvPlane`) already resolve reads
//!   through a `window`-sized ring whose slot is `abs % capacity`; restoring the
//!   last `window` positions at their absolute base and setting `seq_len = upto`
//!   reproduces exactly the ring state a fresh forward would have left, so a
//!   resume-then-tail-prefill is argmax-exact vs a cold full prefill.
//!
//! ★ Laguna caveat (verified in code, `laguna_gpu.rs:184`): `ResidentKvPlane` is
//! host-coherent UMA mapped memory; its `k_up_to_now()` is a direct `&[f32]`
//! view with NO device→host readback. So the Laguna export is a plain memcpy off
//! the mapped planes, byte-for-byte the gemma host-KV shape — NOT the qwen35
//! `dn_gpu_sync` readback dance. `sync_before_export`/`sync_after_import` are
//! therefore no-ops for gemma and Laguna (they exist for a future recurrent-arch
//! unification, P4).
//!
//! ## v1 storage note — K AND V always stored
//!
//! The scope's plan suggested "for k_eq_v global layers write K only (V aliases
//! K)". That is INCORRECT for this codebase's cache representation: even on a
//! `k_eq_v` layer the CACHED V differs from the cached K — V derives from the
//! raw K projection but receives the weightless v-norm and NO RoPE, while K
//! receives k-norm + RoPE (`model.rs::forward_layer` step 2-4). So a tile ALWAYS
//! stores both K and V. The `k_eq_v` flag is retained in the header for
//! provenance only; it never changes what bytes are stored. (Documented
//! deviation.)

/// KV element dtype at rest in a `TILE1` blob / on the NAS.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvDtype {
    /// Bit-exact (host round-trip asserts `f32::to_bits` equality). v1 default
    /// for correctness gates.
    F32,
    /// Half precision — the NAS default (`VLLM_VULKAN_KV_NAS_DTYPE=f16`): 2× the
    /// fetch token-rate + capacity, argmax-exact enough (cluster gate).
    F16,
    /// int8-at-rest — a later gated phase (quality gate required); NOT wired to a
    /// packing here beyond the tag, reserved so the format is versioned.
    Int8,
}

impl KvDtype {
    pub fn tag(self) -> u8 {
        match self {
            KvDtype::F32 => 0,
            KvDtype::F16 => 1,
            KvDtype::Int8 => 2,
        }
    }
    pub fn from_tag(t: u8) -> Result<Self, String> {
        match t {
            0 => Ok(KvDtype::F32),
            1 => Ok(KvDtype::F16),
            2 => Ok(KvDtype::Int8),
            other => Err(format!("TILE1: unknown kv dtype tag {other}")),
        }
    }
    /// Parse `VLLM_VULKAN_KV_NAS_DTYPE` (default f16). int8 is accepted at the
    /// tag level but its packing is a later phase — callers that pass int8 into
    /// `write_section` today get an explicit `Err` from the writer.
    pub fn from_env() -> Self {
        match std::env::var("VLLM_VULKAN_KV_NAS_DTYPE").ok().as_deref() {
            Some("f32") => KvDtype::F32,
            Some("int8") => KvDtype::Int8,
            _ => KvDtype::F16,
        }
    }
}

/// Per-layer KV geometry that changes tile bytes (folded into the content
/// fingerprint so any dim drift is a clean MISS, never stale KV).
#[derive(Clone, Copy, Debug)]
pub struct LayerKvGeom {
    pub kv_heads: usize,
    pub head_dim: usize,
    pub is_full: bool,
    /// Sliding window (0 or irrelevant for full-attn layers).
    pub window: usize,
    pub k_eq_v: bool,
}

/// The config-derived inputs to the topology-agnostic content fingerprint
/// (scope §3.3 / §4.1). Deliberately holds NO pp/tp — topology leaves the key.
pub struct KvContentDims {
    /// Architecture discriminator (gemma=0, laguna=1, …) so two archs with
    /// coincidentally-identical layer geometry never collide.
    pub arch_tag: u8,
    pub num_layers: usize,
    pub layers: Vec<LayerKvGeom>,
    /// An arch-identifying rope / scaling fold (e.g. gemma sliding_window +
    /// periods; Laguna full_rope_theta bits) — folded so a rope-param change
    /// invalidates cached KV even when head geometry is unchanged.
    pub rope_ident: u64,
}

/// One canonical KV tile this rank currently owns: absolute `(layer, kv_head)`
/// plus the geometry needed to serialize it.
#[derive(Clone, Copy, Debug)]
pub struct TileSpec {
    pub layer: usize,
    pub kv_head: usize,
    pub head_dim: usize,
    pub is_full: bool,
    pub window: usize,
    pub k_eq_v: bool,
}

/// Canonical-tile KV prefix export/import. A model implements this for the KV
/// tiles it currently OWNS (a PP stage owns a layer window; a TP rank owns a
/// kv_head sub-range of every layer). Tiles are addressed by ABSOLUTE
/// `(layer, kv_head)` so a snapshot is topology-agnostic (scope §4.1).
pub trait KvPrefixExport {
    /// Config-derived content fingerprint inputs (dims that change tile bytes).
    fn kv_content_dims(&self) -> KvContentDims;
    /// The absolute `(layer, kv_head)` tiles resident on THIS rank right now.
    fn owned_tiles(&self) -> Vec<TileSpec>;
    /// Serialize one owned tile's K and V for the boundary `upto` into a `TILE1`
    /// blob (full-attn stores `[0, upto)`; sliding stores `[max(0, upto-window),
    /// upto)` with the absolute base in the header).
    fn export_tile(&self, layer: usize, kv_head: usize, upto: usize, dtype: KvDtype)
        -> Result<Vec<u8>, String>;
    /// Restore one tile's K/V into this rank's resident KV at its `(layer,
    /// kv_head)`. Returns the number of rows restored. Does NOT set seq_len —
    /// the caller calls `set_seq_len(upto)` once after a full-coverage import.
    fn import_tile(&mut self, layer: usize, kv_head: usize, blob: &[u8]) -> Result<usize, String>;
    /// Set every restored layer's valid position to `n` (seq_len) after import.
    fn set_seq_len(&mut self, n: usize);
    /// Sync GPU-authoritative KV into the host-visible copy BEFORE an export.
    /// No-op for gemma / Laguna (host-coherent mapped memory — no readback).
    /// Reserved for the P4 recurrent-arch unification.
    fn sync_before_export(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// Push a host-restored KV back to GPU-authoritative buffers AFTER an import.
    /// No-op for gemma / Laguna.
    fn sync_after_import(&mut self) -> Result<(), String> {
        Ok(())
    }
}

// ───────────────────────── content fingerprint ──────────────────────────────

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

#[inline]
fn fnv_bytes(acc: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *acc ^= b as u64;
        *acc = acc.wrapping_mul(FNV_PRIME);
    }
}

/// Topology-agnostic content fingerprint (scope §3.3): folds every dim that
/// changes tile bytes PLUS `weights_id`, `tokenizer_hash`, and the `kv_dtype`
/// tag — but NO pp/tp (that leaves the key, scope §4.1). A re-quant
/// (`weights_id` change), tokenizer bump (`tokenizer_hash` change), or dtype
/// switch therefore MISSES cleanly and never serves stale KV.
pub fn content_fingerprint(
    dims: &KvContentDims,
    weights_id: u64,
    tokenizer_hash: u64,
    kv_dtype: KvDtype,
) -> u64 {
    let mut acc = FNV_OFFSET;
    fnv_bytes(&mut acc, &[dims.arch_tag]);
    fnv_bytes(&mut acc, &(dims.num_layers as u64).to_le_bytes());
    fnv_bytes(&mut acc, &dims.rope_ident.to_le_bytes());
    for g in &dims.layers {
        fnv_bytes(&mut acc, &(g.kv_heads as u64).to_le_bytes());
        fnv_bytes(&mut acc, &(g.head_dim as u64).to_le_bytes());
        fnv_bytes(&mut acc, &[g.is_full as u8]);
        fnv_bytes(&mut acc, &(g.window as u64).to_le_bytes());
        fnv_bytes(&mut acc, &[g.k_eq_v as u8]);
    }
    fnv_bytes(&mut acc, &weights_id.to_le_bytes());
    fnv_bytes(&mut acc, &tokenizer_hash.to_le_bytes());
    fnv_bytes(&mut acc, &[kv_dtype.tag()]);
    acc
}

// ────────────────────────── TILE1 blob body ─────────────────────────────────
//
//   magic "TILE1"  [5]u8
//   version        u32   = 1
//   layer          u64
//   kv_head        u64
//   dtype          u8    (0=f32, 1=f16, 2=int8)
//   window_base    u64   (abs pos of first stored row; 0 for full-attn)
//   n_rows         u64
//   head_dim       u64
//   k_eq_v         u8    (provenance only; V is ALWAYS stored — see module doc)
//   K section (length-prefixed: u64 byte-len, then packed bytes)
//   V section (length-prefixed)                      <- always present in v1

const TILE_MAGIC: &[u8; 5] = b"TILE1";

/// Header fields parsed off a `TILE1` blob (before the K/V sections).
pub struct TileHeader {
    pub layer: usize,
    pub kv_head: usize,
    pub dtype: KvDtype,
    pub window_base: usize,
    pub n_rows: usize,
    pub head_dim: usize,
    pub k_eq_v: bool,
    /// Byte offset of the first (K) section within the blob.
    pub body_off: usize,
}

/// Serialize one tile's K + V rows (`k`/`v` are `[n_rows * head_dim]` f32 in
/// ascending-absolute-position row order) into a full `TILE1` blob.
pub fn write_tile(
    layer: usize,
    kv_head: usize,
    dtype: KvDtype,
    window_base: usize,
    head_dim: usize,
    k_eq_v: bool,
    k: &[f32],
    v: &[f32],
) -> Result<Vec<u8>, String> {
    if head_dim == 0 {
        return Err("write_tile: head_dim == 0".to_string());
    }
    if k.len() != v.len() {
        return Err(format!("write_tile: K len {} != V len {}", k.len(), v.len()));
    }
    if k.len() % head_dim != 0 {
        return Err(format!(
            "write_tile: K len {} not a multiple of head_dim {head_dim}",
            k.len()
        ));
    }
    let n_rows = k.len() / head_dim;
    let mut out = Vec::with_capacity(64 + k.len() * 2 * 4);
    out.extend_from_slice(TILE_MAGIC);
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(layer as u64).to_le_bytes());
    out.extend_from_slice(&(kv_head as u64).to_le_bytes());
    out.push(dtype.tag());
    out.extend_from_slice(&(window_base as u64).to_le_bytes());
    out.extend_from_slice(&(n_rows as u64).to_le_bytes());
    out.extend_from_slice(&(head_dim as u64).to_le_bytes());
    out.push(k_eq_v as u8);
    write_section(&mut out, k, dtype)?;
    write_section(&mut out, v, dtype)?;
    Ok(out)
}

/// Parse a `TILE1` header (validates magic/version, leaves `body_off` at the
/// first section). A truncated/corrupt header is a clean `Err`.
pub fn read_tile_header(blob: &[u8]) -> Result<TileHeader, String> {
    let mut pos = 0usize;
    if blob.len() < 5 || &blob[0..5] != TILE_MAGIC {
        return Err("read_tile: bad magic (not a TILE1 blob)".to_string());
    }
    pos += 5;
    let version = read_u32(blob, &mut pos)?;
    if version != 1 {
        return Err(format!("read_tile: unsupported version {version}"));
    }
    let layer = read_u64(blob, &mut pos)? as usize;
    let kv_head = read_u64(blob, &mut pos)? as usize;
    let dtype = KvDtype::from_tag(read_u8(blob, &mut pos)?)?;
    let window_base = read_u64(blob, &mut pos)? as usize;
    let n_rows = read_u64(blob, &mut pos)? as usize;
    let head_dim = read_u64(blob, &mut pos)? as usize;
    let k_eq_v = read_u8(blob, &mut pos)? != 0;
    Ok(TileHeader {
        layer,
        kv_head,
        dtype,
        window_base,
        n_rows,
        head_dim,
        k_eq_v,
        body_off: pos,
    })
}

/// Read the (K, V) row data out of a `TILE1` blob as f32 (converting from the
/// stored dtype). Returns `(k, v)`, each `[n_rows * head_dim]`.
pub fn read_tile_body(blob: &[u8], hdr: &TileHeader) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut pos = hdr.body_off;
    let k = read_section(blob, &mut pos, hdr.dtype)?;
    let v = read_section(blob, &mut pos, hdr.dtype)?;
    let expect = hdr.n_rows * hdr.head_dim;
    if k.len() != expect || v.len() != expect {
        return Err(format!(
            "read_tile: section len (K {}, V {}) != n_rows*head_dim ({expect})",
            k.len(),
            v.len()
        ));
    }
    Ok((k, v))
}

fn write_section(out: &mut Vec<u8>, data: &[f32], dtype: KvDtype) -> Result<(), String> {
    match dtype {
        KvDtype::F32 => {
            out.extend_from_slice(&((data.len() * 4) as u64).to_le_bytes());
            for &x in data {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        KvDtype::F16 => {
            out.extend_from_slice(&((data.len() * 2) as u64).to_le_bytes());
            for &x in data {
                out.extend_from_slice(&f32_to_f16(x).to_le_bytes());
            }
        }
        KvDtype::Int8 => {
            return Err(
                "write_section: int8-at-rest is a later gated phase (not packed in P1)".to_string(),
            );
        }
    }
    Ok(())
}

fn read_section(buf: &[u8], pos: &mut usize, dtype: KvDtype) -> Result<Vec<f32>, String> {
    let len = read_u64(buf, pos)? as usize;
    if *pos + len > buf.len() {
        return Err("read_section: truncated section body".to_string());
    }
    let out = match dtype {
        KvDtype::F32 => {
            if len % 4 != 0 {
                return Err(format!("read_section: f32 len {len} not a multiple of 4"));
            }
            let n = len / 4;
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let o = *pos + i * 4;
                v.push(f32::from_le_bytes(buf[o..o + 4].try_into().unwrap()));
            }
            v
        }
        KvDtype::F16 => {
            if len % 2 != 0 {
                return Err(format!("read_section: f16 len {len} not a multiple of 2"));
            }
            let n = len / 2;
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let o = *pos + i * 2;
                let bits = u16::from_le_bytes(buf[o..o + 2].try_into().unwrap());
                v.push(f16_to_f32(bits));
            }
            v
        }
        KvDtype::Int8 => {
            return Err(
                "read_section: int8-at-rest is a later gated phase (not unpacked in P1)".to_string(),
            );
        }
    };
    *pos += len;
    Ok(out)
}

fn read_u8(buf: &[u8], pos: &mut usize) -> Result<u8, String> {
    if *pos + 1 > buf.len() {
        return Err("TILE1 blob truncated (u8)".to_string());
    }
    let v = buf[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > buf.len() {
        return Err("TILE1 blob truncated (u32)".to_string());
    }
    let v = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > buf.len() {
        return Err("TILE1 blob truncated (u64)".to_string());
    }
    let v = u64::from_le_bytes(buf[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

// ────────────────────── IEEE-754 half precision ─────────────────────────────
//
// KV values are well within normal f16 range; this handles normals, zeros,
// inf/NaN, and subnormals (round-to-nearest-even not required — the on-cluster
// argmax gate is what certifies f16 acceptability, this only needs a stable,
// standard round-trip).

/// f32 → IEEE-754 binary16 bits (round toward nearest; ties→away is acceptable).
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf / NaN.
        let m16 = if mant != 0 { 0x0200 } else { 0 }; // keep NaN non-zero mantissa
        return sign | 0x7c00 | m16;
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow → inf.
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        // Subnormal or underflow to zero.
        if unbiased < -25 {
            return sign;
        }
        // Add implicit leading 1, then shift into subnormal range with rounding.
        let full_mant = mant | 0x0080_0000;
        let shift = (-14 - unbiased) as u32 + 13;
        if shift >= 32 {
            return sign;
        }
        let round_bit = 1u32 << (shift - 1);
        let sub = (full_mant + round_bit) >> shift;
        return sign | (sub as u16);
    }
    // Normal. Round the 23-bit mantissa to 10 bits (nearest); a rounding carry
    // (mant_rounded == 0x400) propagates into the exponent by construction when
    // ADDED to the shifted exponent — and if that overflows the exponent to 31
    // it becomes inf, the correct IEEE result of rounding the max normal up.
    let e16 = ((unbiased + 15) as u32) << 10;
    let mant_rounded = (mant + 0x0000_1000) >> 13; // 0..=0x400
    let combined = e16 + mant_rounded;
    sign | (combined as u16)
}

/// IEEE-754 binary16 bits → f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalize.
            let mut e = -14i32;
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let e32 = ((e + 127) as u32) << 23;
            sign | e32 | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN.
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        let e32 = (exp + 112) << 23; // (exp - 15 + 127) << 23
        sign | e32 | (mant << 13)
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_stable_and_representable() {
        // Exactly-representable f16 values must round-trip bit-exact through
        // f32 and back.
        for &x in &[0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 100.0, -256.0, 0.125] {
            let back = f16_to_f32(f32_to_f16(x));
            assert_eq!(back.to_bits(), x.to_bits(), "f16 exact value {x} drifted");
        }
        // Arbitrary values: round-trip must land within f16 resolution and be
        // IDEMPOTENT (encode(decode(encode(x))) == encode(x)).
        for i in 0..1000 {
            let x = (i as f32 - 500.0) * 0.0137;
            let once = f32_to_f16(x);
            let twice = f32_to_f16(f16_to_f32(once));
            assert_eq!(once, twice, "f16 pack not idempotent for {x}");
            let rel = ((f16_to_f32(once) - x) / x.abs().max(1e-6)).abs();
            assert!(rel < 1e-2, "f16 rel error {rel} too large for {x}");
        }
    }

    #[test]
    fn tile_blob_roundtrip_f32_bit_exact() {
        let head_dim = 8usize;
        let n_rows = 5usize;
        let k: Vec<f32> = (0..n_rows * head_dim).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let v: Vec<f32> = (0..n_rows * head_dim).map(|i| (i as f32) * -0.25 + 1.0).collect();
        let blob = write_tile(3, 1, KvDtype::F32, 0, head_dim, false, &k, &v).unwrap();
        let hdr = read_tile_header(&blob).unwrap();
        assert_eq!(hdr.layer, 3);
        assert_eq!(hdr.kv_head, 1);
        assert_eq!(hdr.n_rows, n_rows);
        assert_eq!(hdr.head_dim, head_dim);
        let (gk, gv) = read_tile_body(&blob, &hdr).unwrap();
        for (a, b) in k.iter().zip(&gk) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in v.iter().zip(&gv) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn tile_blob_truncation_is_clean_err() {
        let k = vec![1.0f32; 8];
        let blob = write_tile(0, 0, KvDtype::F32, 0, 8, false, &k, &k).unwrap();
        assert!(read_tile_header(&blob[..3]).is_err(), "short magic must Err");
        let hdr = read_tile_header(&blob).unwrap();
        // Chop a section byte off the end → clean Err, never a panic.
        assert!(read_tile_body(&blob[..blob.len() - 1], &hdr).is_err());
    }

    #[test]
    fn content_fingerprint_misses_on_drift() {
        let dims = KvContentDims {
            arch_tag: 0,
            num_layers: 2,
            layers: vec![
                LayerKvGeom { kv_heads: 2, head_dim: 32, is_full: false, window: 1024, k_eq_v: false },
                LayerKvGeom { kv_heads: 1, head_dim: 64, is_full: true, window: 0, k_eq_v: true },
            ],
            rope_ident: 0xABCD,
        };
        let base = content_fingerprint(&dims, 111, 222, KvDtype::F16);
        // Same inputs → same fp.
        assert_eq!(base, content_fingerprint(&dims, 111, 222, KvDtype::F16));
        // Weights bump (requant) → miss.
        assert_ne!(base, content_fingerprint(&dims, 999, 222, KvDtype::F16));
        // Tokenizer bump → miss.
        assert_ne!(base, content_fingerprint(&dims, 111, 333, KvDtype::F16));
        // dtype switch → miss.
        assert_ne!(base, content_fingerprint(&dims, 111, 222, KvDtype::F32));
        // arch tag change → miss.
        let mut d2 = KvContentDims {
            arch_tag: 1,
            num_layers: dims.num_layers,
            layers: dims.layers.clone(),
            rope_ident: dims.rope_ident,
        };
        assert_ne!(base, content_fingerprint(&d2, 111, 222, KvDtype::F16));
        // rope param change → miss.
        d2.arch_tag = 0;
        d2.rope_ident = 0x1234;
        assert_ne!(base, content_fingerprint(&d2, 111, 222, KvDtype::F16));
    }
}
