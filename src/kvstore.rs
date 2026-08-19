// SPDX-License-Identifier: Apache-2.0
//! Content-addressed KV-cache backend for the LMCache-NAS plan (Phase 1,
//! `plan-lmcache-nas.md`). Model-agnostic: this module knows nothing about
//! `Qwen35Model` — it stores/loads opaque byte blobs (the `PFX1` prefix blobs
//! `qwen35::export_prefix`/`import_prefix` produce) keyed by a content
//! address. `VulkanModel`'s `kv_cache_store`/`kv_cache_load` pymethods
//! (`lib.rs`) are the only callers that know the blobs are `PFX1`.
//!
//! ## Backend
//!
//! Plain `std::fs` file I/O against a root directory (env
//! `VLLM_VULKAN_KV_STORE_DIR`) — filesystem-agnostic, so a local Mac
//! directory today is byte-for-byte the same code path as the real CIFS
//! mount later (§0 of the plan: "just a different path").
//!
//! ## Key scheme (plan §3.4/§3.5)
//!
//! ```text
//! key = FNV( fingerprint || layout.fold() || chunk_chain_hash )
//! ```
//! - `fingerprint`: the model config/layout fingerprint
//!   (`qwen35::prefix_fingerprint` — model dims, layer types, PP range).
//! - `layout`: `LayoutTag` folds in PP range AND tp_rank/tp_size (the plan's
//!   §3.4/§3.5 requirement that a PP-5 chunk can never be loaded under TP-4,
//!   even though `prefix_fingerprint` alone already covers the PP range —
//!   this module adds the TP dimension explicitly so the combined key is
//!   the full parallelism-layout gate).
//! - `chunk_chain_hash`: FNV chain over `token_id[0..k)`, `k` a multiple of
//!   `CHUNK` — chaining (not an independent hash per chunk) is what gives
//!   longest-prefix matching: two requests sharing a token prefix produce
//!   the same chain value up to their point of divergence.
//!
//! File name = lowercase hex of the 64-bit key (16 hex chars).
//!
//! ## v1 storage granularity (PINNED — plan §3.2, task instructions)
//!
//! One blob per chunk boundary `k` (multiple of `CHUNK`), holding the FULL
//! prefix `[0, k)` (i.e. exactly what `export_prefix(k)` produces). This
//! duplicates overlapping KV across boundaries — the per-chunk-concatenation
//! storage optimization the plan's §3.2 mentions is a documented v2 follow-up,
//! NOT built here. Longest-prefix-match = the largest cached boundary <= the
//! request's chunk-aligned prefix length.
//!
//! ## Two tiers (plan §3.1 / §8)
//!
//! - Host-RAM LRU (hot): loaded blobs, capped by `VLLM_VULKAN_KV_LRU_MB`.
//!   **Write-through + drop-on-evict**: `store()` persists to the backend
//!   immediately, so LRU eviction is a pure RAM free — no write-back, no
//!   eviction write-storm (plan §7.3/§8).
//! - Backend (cold): the content-addressed directory.
//!
//! ## In-RAM manifest (plan §8 item 5)
//!
//! A `HashSet<u64>` of keys known to exist on the backend, built once by a
//! directory scan in `open()` and updated on every `store()`. Lookup/longest-
//! prefix-match never stats a candidate file — it only ever consults this
//! set — which is the whole point: CIFS metadata RTTs are the thing being
//! avoided.
//!
//! ## Format
//!
//! Stored bytes are exactly the `PFX1` blob (f32) as produced today. The
//! blob already carries its own version field (see `qwen35.rs`), so
//! **f16-for-NAS (plan §8's default-format traffic lever) drops in later**
//! as a new blob version — DEFERRED here; it needs the cluster argmax-exact
//! quality gate (plan Phase 3), not just a Mac round-trip.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

/// Chunk granularity — plan §3.2, pinned at 256 tokens.
pub(crate) const CHUNK: usize = 256;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

#[inline]
fn fnv_mix(acc: u64, v: u64) -> u64 {
    let mut acc = acc ^ v;
    acc = acc.wrapping_mul(FNV_PRIME);
    acc
}

/// The parallelism-layout dimension of the key (plan §3.4/§3.5): a chunk
/// cached under one PP/TP layout must never be loaded under another, even if
/// the model config fingerprint matches (a PP-5 stage-2 blob is NOT
/// interchangeable with a PP-3 stage-1 blob of otherwise-identical dims —
/// `prefix_fingerprint` already gates PP range; this adds TP explicitly so
/// this module's key is self-contained without relying on callers to also
/// pass the fingerprint's internal PP bytes correctly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayoutTag {
    pub pp_start: usize,
    pub pp_end: usize,
    pub tp_rank: usize,
    pub tp_size: usize,
}

impl LayoutTag {
    fn fold(&self) -> u64 {
        let mut acc = FNV_OFFSET;
        acc = fnv_mix(acc, self.pp_start as u64);
        acc = fnv_mix(acc, self.pp_end as u64);
        acc = fnv_mix(acc, self.tp_rank as u64);
        acc = fnv_mix(acc, self.tp_size as u64);
        acc
    }
}

/// FNV chain-hash of `token_ids[0..boundary)`, chained per `CHUNK`-sized
/// sub-range (`h_i = H(h_{i-1} || chunk_tokens_i)`, plan §3.4). Returns the
/// `(boundary, chain_hash)` pair at every chunk-aligned boundary from `CHUNK`
/// up to `floor(token_ids.len() / CHUNK) * CHUNK`, in ascending order. A
/// prefix shorter than one full chunk yields an empty vec (v1 stores only
/// whole-chunk boundaries — see the module doc's pinned granularity).
pub(crate) fn chunk_boundary_hashes(token_ids: &[u32]) -> Vec<(usize, u64)> {
    let aligned = (token_ids.len() / CHUNK) * CHUNK;
    let mut out = Vec::with_capacity(aligned / CHUNK);
    let mut acc = FNV_OFFSET;
    let mut i = 0usize;
    while i < aligned {
        let end = i + CHUNK;
        for &tok in &token_ids[i..end] {
            acc = fnv_mix(acc, tok as u64);
        }
        out.push((end, acc));
        i = end;
    }
    out
}

/// Combine the config/layout fingerprint, the parallelism layout, and a
/// chunk-chain hash into the final content-address key (plan §3.4).
pub(crate) fn kv_key(fingerprint: u64, layout: LayoutTag, chain_hash: u64) -> u64 {
    let mut acc = FNV_OFFSET;
    acc = fnv_mix(acc, fingerprint);
    acc = fnv_mix(acc, layout.fold());
    acc = fnv_mix(acc, chain_hash);
    acc
}

/// Canonical-tile base key (NAS prefix-cache scope §4.1, §2.1): the content
/// address of a whole prefix boundary WITHOUT any topology in it —
/// `FNV(content_fp ‖ chain_hash)`. `content_fp` is the topology-agnostic
/// `kv_prefix::content_fingerprint` (weights-id ‖ tokenizer-hash ‖ kv-dtype ‖
/// config-dims), NOT the old pp/tp-folding `LayoutTag` scheme. A prefix warmed
/// under any PP/TP layout shares the same `base_key`; the per-rank slice is a
/// `tile_key` off this base.
pub(crate) fn base_key(content_fp: u64, chain_hash: u64) -> u64 {
    let mut acc = FNV_OFFSET;
    acc = fnv_mix(acc, content_fp);
    acc = fnv_mix(acc, chain_hash);
    acc
}

/// Canonical `(layer, kv_head)` tile address off a `base_key`. Topology appears
/// NOWHERE — a PP stage writes its layer window's tiles, a TP rank writes its
/// head shard's tiles, and the store is the natural union (scope §2.2). A reader
/// under a DIFFERENT topology gathers exactly the `(layer, kv_head)` it needs.
pub(crate) fn tile_key(base_key: u64, layer: usize, kv_head: usize) -> u64 {
    let mut acc = FNV_OFFSET;
    acc = fnv_mix(acc, base_key);
    acc = fnv_mix(acc, layer as u64);
    acc = fnv_mix(acc, kv_head as u64);
    acc
}

fn key_to_filename(key: u64) -> String {
    format!("{key:016x}")
}

/// Content-addressed file backend + host-RAM LRU hot tier + in-RAM manifest.
/// See the module doc for the full design; this struct is intentionally
/// model-agnostic (operates on `Vec<u8>` blobs keyed by `u64`).
pub(crate) struct KvStore {
    dir: PathBuf,
    /// Keys known to exist on the backend (plan §8 item 5) — populated by a
    /// one-time scan in `open()`, updated on every `store()`. Lookup never
    /// stats a candidate path.
    manifest: HashSet<u64>,
    /// Host-RAM LRU hot tier: recency order (front = least-recently-used).
    lru_order: VecDeque<u64>,
    lru_data: HashMap<u64, Vec<u8>>,
    lru_bytes: usize,
    lru_cap_bytes: usize,
    /// Test-only instrumentation: counts actual backend writes (NOT calls to
    /// `store()`) so the idempotency test can assert a same-content second
    /// `store()` performs no I/O.
    #[cfg(test)]
    backend_writes: usize,
}

impl KvStore {
    /// Open (creating if needed) a content-addressed store rooted at `dir`,
    /// with a host-RAM LRU capped at `lru_mb` megabytes. Scans the backend
    /// once to build the in-RAM manifest (plan §8 item 5) — every existing
    /// `[0-9a-f]{16}`-named file is assumed to be a valid blob written by
    /// this store (content-addressing means a name collision implies
    /// identical content, so no validation read is needed).
    pub(crate) fn open(dir: impl Into<PathBuf>, lru_mb: usize) -> std::io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        let mut manifest = HashSet::new();
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            if let Some(name) = entry.file_name().to_str() {
                if name.len() == 16 {
                    if let Ok(k) = u64::from_str_radix(name, 16) {
                        manifest.insert(k);
                    }
                }
            }
        }
        Ok(Self {
            dir,
            manifest,
            lru_order: VecDeque::new(),
            lru_data: HashMap::new(),
            lru_bytes: 0,
            lru_cap_bytes: lru_mb.saturating_mul(1024 * 1024),
            #[cfg(test)]
            backend_writes: 0,
        })
    }

    fn path_for(&self, key: u64) -> PathBuf {
        self.dir.join(key_to_filename(key))
    }

    /// `true` iff the backend (per the in-RAM manifest, never a stat) is
    /// known to already hold `key`.
    pub(crate) fn contains(&self, key: u64) -> bool {
        self.manifest.contains(&key)
    }

    /// Persist `blob` under `key`. Write-through + idempotent: if the
    /// manifest already lists `key`, no backend write happens at all (plan
    /// §8 item 5 — content-addressing makes a re-store of identical content
    /// a no-op, not just a redundant-but-harmless write). Always refreshes
    /// the LRU hot-tier copy so a store is immediately hot.
    ///
    /// This intentional skip is logged at `debug` (not silent) — a caller
    /// re-running an IDENTICAL request (same token chain -> same chunk-chain
    /// key) against a store dir that already has that key from a prior
    /// process/run will see `kv_cache_store` return `Ok(())` with no new
    /// backend write, which is correct-by-design, not a swallowed failure —
    /// see the module doc's "Two tiers" / manifest section and the
    /// `content_addressing_is_idempotent`/`manifest_prevents_restat_on_repeat_lookup`
    /// tests below.
    pub(crate) fn store(&mut self, key: u64, blob: &[u8]) -> std::io::Result<()> {
        if !self.manifest.contains(&key) {
            fs::write(self.path_for(key), blob)?;
            self.manifest.insert(key);
            #[cfg(test)]
            {
                self.backend_writes += 1;
            }
        } else {
            log::debug!(
                "KvStore::store: key {key:016x} already in the manifest — idempotent \
                 no-op skip, not a write failure (content-addressed re-store of identical data)"
            );
        }
        self.lru_put(key, blob.to_vec());
        Ok(())
    }

    /// Load `key`'s blob: LRU hit first, else a backend read (which also
    /// populates the LRU). `None` if `key` isn't in the manifest (a clean
    /// cache miss, not an I/O error).
    pub(crate) fn load(&mut self, key: u64) -> Option<Vec<u8>> {
        if let Some(blob) = self.lru_get(key) {
            return Some(blob);
        }
        if !self.manifest.contains(&key) {
            return None;
        }
        let data = fs::read(self.path_for(key)).ok()?;
        self.lru_put(key, data.clone());
        Some(data)
    }

    /// Longest-prefix match (plan §3.5): walk `token_ids`'s chunk-chain
    /// boundaries from longest to shortest and return the first one whose
    /// key is in the manifest. `None` on a total miss (including when
    /// `token_ids` is shorter than one `CHUNK`, in which case there ARE no
    /// chunk-aligned boundaries to check — see the module doc's pinned v1
    /// granularity).
    pub(crate) fn lookup_longest_prefix(
        &self,
        token_ids: &[u32],
        fingerprint: u64,
        layout: LayoutTag,
    ) -> Option<(usize, u64)> {
        for (boundary, chain) in chunk_boundary_hashes(token_ids).into_iter().rev() {
            let key = kv_key(fingerprint, layout, chain);
            if self.manifest.contains(&key) {
                return Some((boundary, key));
            }
        }
        None
    }

    /// Canonical-tile longest-prefix lookup (NAS prefix-cache scope §2.4): walk
    /// the token stream's chunk-chain boundaries longest→shortest and return the
    /// first boundary at which EVERY tile in `needed` (this rank's
    /// `(layer, kv_head)` coverage set) is present in the manifest, together with
    /// that boundary's `base_key`. `None` on a total miss. v1 requires FULL
    /// coverage of the rank's needed set at a boundary (per-layer partial restore
    /// is a P3 refinement) — a single missing tile at a boundary rejects it and
    /// the search falls back to the next-shorter boundary.
    pub(crate) fn lookup_longest_prefix_tiles(
        &self,
        token_ids: &[u32],
        content_fp: u64,
        needed: &[(usize, usize)],
    ) -> Option<(usize, u64)> {
        for (boundary, chain) in chunk_boundary_hashes(token_ids).into_iter().rev() {
            let bk = base_key(content_fp, chain);
            let covered = needed
                .iter()
                .all(|&(layer, kv_head)| self.manifest.contains(&tile_key(bk, layer, kv_head)));
            if covered && !needed.is_empty() {
                return Some((boundary, bk));
            }
        }
        None
    }

    // ─── LRU internals (write-through + drop-on-evict, plan §7.3/§8) ───────

    fn lru_touch(&mut self, key: u64) {
        if let Some(pos) = self.lru_order.iter().position(|&k| k == key) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_back(key);
    }

    fn lru_get(&mut self, key: u64) -> Option<Vec<u8>> {
        if self.lru_data.contains_key(&key) {
            self.lru_touch(key);
            self.lru_data.get(&key).cloned()
        } else {
            None
        }
    }

    fn lru_put(&mut self, key: u64, blob: Vec<u8>) {
        let new_len = blob.len();
        if let Some(old) = self.lru_data.insert(key, blob) {
            self.lru_bytes = self.lru_bytes - old.len() + new_len;
        } else {
            self.lru_bytes += new_len;
        }
        self.lru_touch(key);
        // Drop-on-evict: the byte is already durable on the backend (written
        // by `store()`, or it wouldn't be loadable at all), so eviction is a
        // pure RAM free — no write-back. Never evict the sole remaining
        // entry even if it alone exceeds the cap (a cap smaller than one
        // blob must still be able to serve that one blob).
        while self.lru_bytes > self.lru_cap_bytes && self.lru_order.len() > 1 {
            if let Some(victim) = self.lru_order.pop_front() {
                if let Some(v) = self.lru_data.remove(&victim) {
                    self.lru_bytes -= v.len();
                }
            }
        }
    }

    /// Test-only: is `key` currently RAM-resident in the LRU?
    #[cfg(test)]
    fn lru_contains(&self, key: u64) -> bool {
        self.lru_data.contains_key(&key)
    }

    /// Test-only: does the backend file for `key` exist on disk?
    #[cfg(test)]
    fn backend_has(&self, key: u64) -> bool {
        Path::new(&self.path_for(key)).is_file()
    }
}

#[cfg(all(test, feature = "qwen35"))]
mod tests {
    use super::*;
    use crate::qwen35::kv_prefix_tests::{build_hybrid, populate};
    use crate::qwen35::LayerType;

    fn temp_store_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vllm_vulkan_kvstore_test_{tag}_{}_{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    fn layout() -> LayoutTag {
        LayoutTag { pp_start: 0, pp_end: 2, tp_rank: 0, tp_size: 1 }
    }

    #[test]
    fn content_addressing_is_idempotent() {
        let dir = temp_store_dir("idempotent");
        let mut store = KvStore::open(&dir, 64).unwrap();
        let key = kv_key(0xABCD, layout(), 0x1234);
        let blob = vec![1u8, 2, 3, 4, 5];

        store.store(key, &blob).unwrap();
        assert_eq!(store.backend_writes, 1, "first store must write the backend");
        assert!(store.backend_has(key));

        // Same content, same key -> the manifest already knows it, so the
        // second store must NOT touch the backend again (plan §8 item 5).
        store.store(key, &blob).unwrap();
        assert_eq!(store.backend_writes, 1, "re-store of identical content must be a no-op write");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lru_eviction_drops_ram_not_backend() {
        let dir = temp_store_dir("lru_evict");
        // Cap small enough that a second ~4-byte-ish blob forces eviction of
        // the first (LRU tracks raw blob bytes; use tiny cap via lru_mb=0
        // plus manual byte-cap override through repeated small stores).
        let mut store = KvStore::open(&dir, 0).unwrap();
        store.lru_cap_bytes = 8; // override the MB-rounded cap for a tight test

        let k1 = kv_key(1, layout(), 100);
        let k2 = kv_key(1, layout(), 200);
        let blob1 = vec![0xAAu8; 8];
        let blob2 = vec![0xBBu8; 8];

        store.store(k1, &blob1).unwrap();
        assert!(store.lru_contains(k1));
        store.store(k2, &blob2).unwrap();

        // k1 should have been evicted from RAM (LRU, and k2 pushed it out
        // since combined bytes exceed the 8-byte cap) but its file must
        // still be present on the backend — no write-back needed since it
        // was already durable (write-through, plan §7.3).
        assert!(!store.lru_contains(k1), "k1 must be evicted from the RAM LRU");
        assert!(store.backend_has(k1), "k1's backend file must survive RAM eviction (drop-on-evict, no write-back)");
        assert!(store.contains(k1), "manifest must still know k1 after RAM eviction");

        // And it must still be loadable (backend fallback).
        let reloaded = store.load(k1).expect("k1 must still be loadable from the backend");
        assert_eq!(reloaded, blob1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_prevents_restat_on_repeat_lookup() {
        let dir = temp_store_dir("manifest");
        let mut store = KvStore::open(&dir, 64).unwrap();
        let key = kv_key(7, layout(), 42);
        store.store(key, &[9u8, 9, 9]).unwrap();
        assert_eq!(store.backend_writes, 1);

        // Many repeated stores of the SAME key/content must never re-write —
        // the manifest (not a stat) gates every one of them.
        for _ in 0..10 {
            store.store(key, &[9u8, 9, 9]).unwrap();
        }
        assert_eq!(store.backend_writes, 1, "manifest must gate every repeat store, not just the first");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn longest_prefix_match_hit_partial_and_miss() {
        let dir = temp_store_dir("longest_prefix");
        let mut store = KvStore::open(&dir, 64).unwrap();
        let fp = 0x5EED_5EEDu64;
        let lay = layout();

        // A synthetic token-id stream long enough for 3 chunk boundaries
        // (256, 512, 768), plus a partial 4th chunk (no boundary before 1024).
        let tokens: Vec<u32> = (0..(3 * CHUNK + 200)).map(|i| (i % 997) as u32).collect();
        let boundaries = chunk_boundary_hashes(&tokens);
        assert_eq!(boundaries.len(), 3, "3 full chunks in a stream of 3*CHUNK+100 tokens");

        // Store boundaries 256 and 768 (skip 512) — a genuine partial-hit
        // scenario, not just "store everything".
        for &(boundary, chain) in &[boundaries[0], boundaries[2]] {
            let key = kv_key(fp, lay, chain);
            store.store(key, format!("blob@{boundary}").as_bytes()).unwrap();
        }

        // Exact request length == stored longest boundary (768): must hit 768.
        let req_768 = &tokens[..768];
        let (b, _k) = store.lookup_longest_prefix(req_768, fp, lay).expect("768 must hit");
        assert_eq!(b, 768);

        // A longer request (900 tokens, chunk-aligned boundary 768) must
        // still resolve to the longest cached boundary <= its aligned length,
        // i.e. 768 (the un-stored 512 is skipped correctly, and 1024 was
        // never reached/stored).
        let req_900 = &tokens[..900];
        let (b, _k) = store.lookup_longest_prefix(req_900, fp, lay).expect("900 must longest-prefix-hit at 768");
        assert_eq!(b, 768);

        // A request whose token-chain diverges before 256 (first token
        // different) must miss entirely even though a same-length request
        // hits above — proves the match is on the CHAIN, not just length.
        let mut diverged = tokens[..768].to_vec();
        diverged[0] = diverged[0].wrapping_add(1);
        assert!(
            store.lookup_longest_prefix(&diverged, fp, lay).is_none(),
            "a diverged token chain must miss even at a previously-cached length"
        );

        // A too-short request (< 1 chunk) has no chunk-aligned boundary at
        // all -> miss (v1 granularity, see module doc).
        let short = &tokens[..100];
        assert!(store.lookup_longest_prefix(short, fp, lay).is_none());

        // A different layout (different tp_size) must miss even for the
        // exact same token chain and fingerprint (plan §3.4/§3.5: a PP-5
        // chunk can't be loaded under TP-4).
        let other_layout = LayoutTag { pp_start: 0, pp_end: 2, tp_rank: 0, tp_size: 4 };
        assert!(store.lookup_longest_prefix(req_768, fp, other_layout).is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// File round-trip bit-exact (plan gate + task instructions): store a
    /// REAL synthetic `Qwen35Model::export_prefix` blob, load it back through
    /// the backend (not the LRU — drop the process-level cache by opening a
    /// FRESH `KvStore` handle on the same dir first), import into a fresh
    /// model, and assert `f32::to_bits` equality against the source model's
    /// live state.
    #[test]
    fn file_roundtrip_bitexact_real_prefix_blob() {
        let dir = temp_store_dir("roundtrip");
        let layer_types = vec![LayerType::LinearAttention, LayerType::FullAttention];

        let mut src = build_hybrid(layer_types.clone());
        populate(&mut src, 5, 123);
        let boundary = 5;
        let blob = src.export_prefix(boundary).expect("export_prefix must succeed");

        // Expected bits, captured directly off the live source model.
        let expected_k: Vec<Vec<f32>> = src
            .layer_state
            .iter()
            .filter_map(|s| match s {
                crate::qwen35::LayerState::Full(c) => Some(c.k_upto(boundary).to_vec()),
                _ => None,
            })
            .collect();

        let mut store = KvStore::open(&dir, 64).unwrap();
        let fp = crate::qwen35::prefix_fingerprint(&src.config, src.pp_start, src.pp_end);
        let lay = LayoutTag { pp_start: src.pp_start, pp_end: src.pp_end, tp_rank: 0, tp_size: 1 };
        let key = kv_key(fp, lay, 0xF00D);
        store.store(key, &blob).unwrap();
        drop(store);

        // Fresh handle -> forces a backend read (the freshest process would
        // have an empty LRU anyway, but this makes "the backend, not the RAM
        // cache, is bit-exact" explicit).
        let mut store2 = KvStore::open(&dir, 64).unwrap();
        let loaded = store2.load(key).expect("blob must round-trip through the backend");
        assert_eq!(loaded, blob, "backend round-trip must be byte-identical");

        let mut dst = build_hybrid(layer_types);
        let n = dst.import_prefix(&loaded).expect("import_prefix must succeed on the round-tripped blob");
        assert_eq!(n, boundary);

        let got_k: Vec<Vec<f32>> = dst
            .layer_state
            .iter()
            .filter_map(|s| match s {
                crate::qwen35::LayerState::Full(c) => Some(c.k_upto(boundary).to_vec()),
                _ => None,
            })
            .collect();
        assert_eq!(expected_k.len(), got_k.len());
        for (e, g) in expected_k.iter().zip(got_k.iter()) {
            assert_eq!(e.len(), g.len());
            for (x, y) in e.iter().zip(g.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "K element must be f32::to_bits-identical after NAS round-trip");
            }
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    // ─── Canonical (layer, kv_head)-tile store (NAS prefix-cache scope §4.1) ──

    /// Tiles written as if by a PP-5 fleet (5 disjoint layer-window writers) are
    /// read back as the UNION by a single-node reader that owns every layer —
    /// full coverage, no topology in the key. This is the whole point of the
    /// canonical tiling: a prefix warmed under one layout restores under any
    /// other (PP is bit-exact-portable, scope §4.1).
    #[test]
    fn tile_union_is_topology_agnostic() {
        let dir = temp_store_dir("tile_union");
        let mut store = KvStore::open(&dir, 64).unwrap();
        let content_fp = 0xC0FFEEu64;

        // 20 layers, 2 kv_heads each. A stream with 2 chunk boundaries.
        let num_layers = 20usize;
        let kv_heads = 2usize;
        let tokens: Vec<u32> = (0..(2 * CHUNK + 10)).map(|i| (i % 503) as u32).collect();
        let boundaries = chunk_boundary_hashes(&tokens);
        assert_eq!(boundaries.len(), 2);
        let (target_boundary, chain) = boundaries[1]; // longest = 512
        let bk = base_key(content_fp, chain);

        // Five PP writers, each owning a disjoint contiguous layer window
        // [s*4, s*4+4). Each writes ONLY its own tiles — no coordinator.
        for stage in 0..5usize {
            for layer in (stage * 4)..(stage * 4 + 4) {
                for h in 0..kv_heads {
                    let tk = tile_key(bk, layer, h);
                    store.store(tk, format!("tile@{layer}:{h}").as_bytes()).unwrap();
                }
            }
        }

        // Single-node reader owns ALL layers × heads — the full needed set.
        let needed: Vec<(usize, usize)> = (0..num_layers)
            .flat_map(|l| (0..kv_heads).map(move |h| (l, h)))
            .collect();
        let (b, got_bk) = store
            .lookup_longest_prefix_tiles(&tokens, content_fp, &needed)
            .expect("single-node reader must fully cover the PP-5-written union");
        assert_eq!(b, target_boundary);
        assert_eq!(got_bk, bk);

        // A different content_fp (e.g. a requant) must miss entirely.
        assert!(store
            .lookup_longest_prefix_tiles(&tokens, content_fp ^ 0xFF, &needed)
            .is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// TP-1 warms ALL heads of every layer; a TP-4 reader gathers only its head
    /// shard (head 0 of a 8-head layer, say) per layer → still fully covered,
    /// because the key carries no tp degree — only the `(layer, kv_head)`
    /// address (scope §4.1). Same-degree reuse is always safe; the cross-degree
    /// numeric gate is an on-cluster concern above this byte store.
    #[test]
    fn cross_tp_tile_addressing() {
        let dir = temp_store_dir("cross_tp");
        let mut store = KvStore::open(&dir, 64).unwrap();
        let content_fp = 0xABCDEFu64;
        let num_layers = 4usize;
        let kv_heads = 8usize;
        let tokens: Vec<u32> = (0..(CHUNK + 5)).map(|i| (i % 101) as u32).collect();
        let (_boundary, chain) = chunk_boundary_hashes(&tokens)[0];
        let bk = base_key(content_fp, chain);

        // TP-1 writer: all heads of all layers.
        for l in 0..num_layers {
            for h in 0..kv_heads {
                store.store(tile_key(bk, l, h), b"x").unwrap();
            }
        }

        // TP-4 reader rank 2 owns heads [4,6) of every layer.
        let needed: Vec<(usize, usize)> = (0..num_layers)
            .flat_map(|l| (4..6).map(move |h| (l, h)))
            .collect();
        assert!(store
            .lookup_longest_prefix_tiles(&tokens, content_fp, &needed)
            .is_some());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Dropping a single layer's tiles makes a reader that NEEDS that layer
    /// report the boundary uncovered (→ that layer would be re-prefilled). v1
    /// requires full coverage of the needed set at a boundary.
    #[test]
    fn partial_coverage_forces_layer_reprefill() {
        let dir = temp_store_dir("partial_cov");
        let mut store = KvStore::open(&dir, 64).unwrap();
        let content_fp = 0x1234u64;
        let kv_heads = 2usize;
        let tokens: Vec<u32> = (0..(CHUNK + 3)).map(|i| (i % 61) as u32).collect();
        let (_b, chain) = chunk_boundary_hashes(&tokens)[0];
        let bk = base_key(content_fp, chain);

        // Write layers 0,1,2 but SKIP layer 3.
        for l in 0..3usize {
            for h in 0..kv_heads {
                store.store(tile_key(bk, l, h), b"x").unwrap();
            }
        }
        // Reader needs all 4 layers → miss (layer 3 uncovered).
        let needed_all: Vec<(usize, usize)> = (0..4usize)
            .flat_map(|l| (0..kv_heads).map(move |h| (l, h)))
            .collect();
        assert!(store
            .lookup_longest_prefix_tiles(&tokens, content_fp, &needed_all)
            .is_none());
        // Reader that only needs layers 0..3 → full cover.
        let needed_012: Vec<(usize, usize)> = (0..3usize)
            .flat_map(|l| (0..kv_heads).map(move |h| (l, h)))
            .collect();
        assert!(store
            .lookup_longest_prefix_tiles(&tokens, content_fp, &needed_012)
            .is_some());

        std::fs::remove_dir_all(&dir).ok();
    }
}
