// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! P4.3 KVMem-native prefix sharing: immutable canonical source + checkpoint
//! catalog (host model; the GPU page PIN and DeltaNet checkpoint wiring live
//! in the serve engine).
//!
//! ninfer alignment (hipfire-kvmem-p43-ninfer-alignment.md): cross-session
//! reuse restores from an IMMUTABLE canonical archive with provenance
//! validation, never from a live store. ninfer splits mean-K (branch-private
//! `fork_mean_prefix`) from packed KV (shared `records.fork()`); hipfire's
//! per-slot [`RawStore`] is unified, and core `fork_prefix` gives exactly
//! those semantics on it — packed records become sealed shared sources,
//! raw rows + mean-K statistics stay branch-private, the branch is truncated
//! to the frontier. The canonical archive is itself a `fork_prefix` of the
//! source slot at freeze time; every reuse forks the canonical again, so the
//! shared packed payload is charged once and outlives any single session.
//!
//! Provenance (ninfer `restore_memory_prefix` anti-silent-corruption core):
//! a checkpoint carries its FULL prefix token proof (not just a fingerprint)
//! plus the store geometry identity; a reuse must match both before any
//! mutation. The partial-tail mean sum travels as `mean_tail` (core
//! `mean_checkpoint`), restored after the fork per the core contract.
//!
//! Everything here is host-only and additive: with the catalog empty (the
//! default, `HIPFIRE_KVMEM_PREFIX_PIN` off) no path allocates or mutates
//! anything, keeping the Legacy serve byte-neutral.

use hipfire_kvmem::{RawStore, RawStoreConfig};
use hipfire_runtime::checkpoint_pool::prefix_fingerprint;

/// Page-aligned reuse granularity: GPU prefix sharing is page-granular
/// (`share_published_pages`), so a freeze boundary must be a whole page.
pub const PREFIX_PAGE_TOKENS: u32 = 128;

/// Geometry identity of a [`RawStoreConfig`] (provenance double-safety: all
/// slots of one controller share the config, but a checkpoint must refuse a
/// store whose layout drifted — ninfer `descriptor.layout` equality).
pub fn layout_fingerprint(cfg: &RawStoreConfig) -> u64 {
    // FNV-1a over the layout fields; collision risk is negligible for a
    // provenance check that ALSO requires full token equality, and the
    // catalog never crosses processes.
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    mix(u64::from(cfg.n_layer));
    mix(u64::from(cfg.n_embd_k));
    mix(u64::from(cfg.n_embd_v));
    mix(u64::from(cfg.block_tokens));
    mix(cfg.k_row_bytes);
    mix(cfg.k_gpu_row_bytes);
    mix(cfg.v_gpu_row_bytes);
    h
}

/// 期4 (KVMem/nInfer DFlash2 integration): the DFlash2 draft's conditioning
/// state at a prefix checkpoint — the backend-owned "native full sequence
/// state serialization" that rides with the target checkpoint. Captured only
/// when the draft's live seeded frontier equals the checkpoint frontier (the
/// cached ring cannot reconstruct an earlier frontier); restored with the
/// target prefix so the next draft cycle conditions on the same rows the
/// target verify sees. Native F32 layout in ring-slot order (`row % window`):
/// `[target_hidden w×feature_width][k_l w×kv_dim][v_l w×kv_dim]...`.
#[derive(Debug, Clone, Default)]
pub struct DraftImage {
    /// Draft geometry identity (layers, hidden, feature width, kv dim, window,
    /// block) — a mismatch refuses the restore instead of seeding garbage.
    pub geometry: u64,
    /// Logical frontier (committed rows) at capture.
    pub frontier: u32,
    /// Retained-suffix floor (logical tokens) at capture
    /// (nInfer `dflash_context_floor`).
    pub floor: u32,
    /// Ring window `w` (row modulus) at capture.
    pub window: usize,
    pub feature_width: usize,
    pub kv_dim: usize,
    pub n_layers: usize,
    /// Captured thlog watermarks (rows whose projection was cached).
    pub proj_cached_rows: usize,
    pub full_cached_rows: usize,
    /// Raw blobs in the order above, each the full ring (`window × dim × 4`).
    pub blobs: Vec<Vec<u8>>,
}

impl DraftImage {
    /// Host bytes charged to the prefix catalog (blob payload).
    pub fn nbytes(&self) -> u64 {
        self.blobs.iter().map(|b| b.len() as u64).sum()
    }
}

/// One frozen reusable prefix: the immutable canonical archive plus the
/// provenance a reuse must satisfy. Created by [`crate::kvmem_ctl::KvMemCtl::freeze_prefix`];
/// the engine attaches the pinned GPU pages (`gpu_pages`) after sealing them.
pub struct PrefixCheckpoint {
    /// Catalog-unique id (monotonic per controller).
    pub id: u64,
    /// Reuse boundary in tokens — page-aligned, `<= source fed` at freeze.
    pub frontier: u32,
    /// FULL token proof of the shared prefix (ninfer `std::equal` equivalent;
    /// the fingerprint alone is a fast reject, never the admission proof).
    pub prefix_tokens: Vec<u32>,
    /// SHA-256-derived fast-reject fingerprint of `prefix_tokens`.
    pub fingerprint: u64,
    /// Partial-tail mean-sum checkpoint (`None` when block-aligned). The
    /// fork contract: restore it into the branch after truncation.
    pub mean_tail: Option<Vec<f32>>,
    /// Source slot's layerwise incarnation at freeze (provenance stamp; a
    /// reused branch always bumps its OWN incarnation, never inherits this).
    pub source_incarnation: u64,
    /// Store geometry identity at freeze ([`layout_fingerprint`]).
    pub layout_id: u64,
    /// Physical GPU pages pinned for this prefix (cache refs owned by the
    /// catalog entry). Attached by the engine at freeze; released by the
    /// engine when the entry is evicted/removed. Empty until attached.
    pub gpu_pages: Vec<u32>,
    /// 期2.3: span of the user question that created this checkpoint, in the
    /// freeze request's original coordinates. Import compares it with the new
    /// request's proven span (KVMem/nInfer `layerwise_query_at_source`):
    /// equal and inside the frontier ⇒ the source's KV/score already covers
    /// the question ⇒ KeepSelected. `None` = frozen without a proven span.
    pub original_query: Option<(u64, u64)>,
    /// Immutable canonical archive: `source.fork_prefix(frontier)`. Packed
    /// records are sealed shared sources; mean-K is private to this archive
    /// and re-forked private into every branch. Never fed after freeze.
    canonical: RawStore,
    /// 期4: DFlash2 draft conditioning image captured with this checkpoint
    /// (`None` for AR/MTP sessions, non-windowed drafts, or a frontier that
    /// did not equal the draft's live seeded frontier — those reseed).
    pub draft_image: Option<DraftImage>,
    /// LRU clock stamp (catalog-internal).
    last_used: u64,
}

impl PrefixCheckpoint {
    /// Build a frozen checkpoint around an immutable canonical archive
    /// (`source.fork_prefix(frontier)`). The fingerprint is derived from the
    /// full token proof; `gpu_pages` starts empty (the engine attaches the
    /// pinned pages after sealing them).
    pub fn new(
        id: u64,
        frontier: u32,
        prefix_tokens: Vec<u32>,
        mean_tail: Option<Vec<f32>>,
        source_incarnation: u64,
        layout_id: u64,
        canonical: RawStore,
    ) -> Self {
        let fingerprint = prefix_fingerprint(&prefix_tokens);
        Self {
            id,
            frontier,
            prefix_tokens,
            fingerprint,
            mean_tail,
            source_incarnation,
            layout_id,
            gpu_pages: Vec::new(),
            original_query: None,
            canonical,
            draft_image: None,
            last_used: 0,
        }
    }

    /// Host bytes charged for retention (canonical packed archive + the
    /// optional draft image).
    pub fn host_bytes(&self) -> u64 {
        self.canonical.payload_bytes()
            + self.draft_image.as_ref().map_or(0, |img| img.nbytes())
    }

    /// The canonical archive (read-only use: fork / inspect).
    pub fn canonical(&self) -> &RawStore {
        &self.canonical
    }

    /// Validate a reuse request against this checkpoint's provenance:
    /// the prompt must cover the frontier and match the FULL token proof,
    /// and the destination store geometry must equal the freeze geometry.
    /// Runs BEFORE any mutation (ninfer fail-closed-before-mutation).
    pub fn validate_reuse(&self, prompt: &[u32], dst_layout: u64) -> Result<(), String> {
        if self.layout_id != dst_layout {
            return Err(format!(
                "kvmem prefix {}: store geometry drifted since freeze",
                self.id
            ));
        }
        let frontier = self.frontier as usize;
        if prompt.len() < frontier {
            return Err(format!(
                "kvmem prefix {}: prompt {} tokens shorter than frontier {frontier}",
                self.id,
                prompt.len()
            ));
        }
        if prefix_fingerprint(&prompt[..frontier]) != self.fingerprint {
            return Err(format!(
                "kvmem prefix {}: prompt fingerprint mismatch at frontier {frontier}",
                self.id
            ));
        }
        if prompt[..frontier] != self.prefix_tokens[..] {
            return Err(format!(
                "kvmem prefix {}: prompt token proof mismatch at frontier {frontier}",
                self.id
            ));
        }
        Ok(())
    }
}

/// Engine-wide catalog of frozen prefixes (NOT per-slot: a checkpoint outlives
/// its source session). Bounded count with LRU eviction; eviction RETURNS the
/// displaced entries so the engine can release their GPU cache refs — the
/// catalog itself is host-only and never touches the page pool.
pub struct PrefixCatalog {
    entries: Vec<PrefixCheckpoint>,
    next_id: u64,
    clock: u64,
    max_entries: usize,
    /// Displaced entries awaiting handoff (see `insert` / `insert_displacing`).
    /// Production always drains immediately via `insert_displacing`.
    pending_displaced: Vec<PrefixCheckpoint>,
}

impl PrefixCatalog {
    /// A catalog bounded to `max_entries` frozen prefixes (0 refuses every
    /// insert — the disabled configuration never constructs one in prod).
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            clock: 0,
            max_entries,
            pending_displaced: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// GPU pages pinned resident across every cataloged checkpoint (one count
    /// per shared page-reference; a page pinned by two checkpoints counts
    /// twice, matching the cache-ref bookkeeping the engine releases).
    pub fn pinned_pages(&self) -> u64 {
        self.entries
            .iter()
            .map(|e| e.gpu_pages.len() as u64)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Mint the next checkpoint id (the caller builds the entry, then
    /// [`Self::insert`]s it). Ids are monotonic and never reused.
    pub fn mint_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Insert a frozen checkpoint, evicting LRU entries to stay bounded.
    /// Returns the displaced entries (caller releases their GPU refs).
    /// Refuses a duplicate frontier+proof already present (the existing
    /// entry is touched instead and returned as `Ok(None)` — no second
    /// archive for the same prefix).
    pub fn insert(&mut self, mut entry: PrefixCheckpoint) -> Result<Option<u64>, String> {
        if self.max_entries == 0 {
            return Err("kvmem prefix catalog: disabled (max_entries 0)".to_string());
        }
        if let Some(existing) = self
            .entries
            .iter()
            .find(|e| e.frontier == entry.frontier && e.fingerprint == entry.fingerprint)
        {
            let id = existing.id;
            self.touch(id);
            return Ok(None);
        }
        let mut displaced = Vec::new();
        while self.entries.len() >= self.max_entries {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .expect("nonempty catalog has an LRU entry");
            displaced.push(self.entries.remove(victim));
        }
        self.clock += 1;
        entry.last_used = self.clock;
        let id = entry.id;
        self.entries.push(entry);
        // The caller owns the displaced GPU refs; production drains them via
        // `insert_displacing`. Test-facing `insert` stashes them for
        // `take_displaced`.
        self.pending_displaced.extend(displaced);
        Ok(Some(id))
    }

    /// Engine-facing insert: returns `(inserted id or None when deduped,
    /// displaced entries whose GPU cache refs the engine must release)`.
    pub fn insert_displacing(
        &mut self,
        entry: PrefixCheckpoint,
    ) -> Result<(Option<u64>, Vec<PrefixCheckpoint>), String> {
        let id = self.insert(entry)?;
        Ok((id, std::mem::take(&mut self.pending_displaced)))
    }

    /// Drain any displaced entries left by [`Self::insert`] (test-facing).
    pub fn take_displaced(&mut self) -> Vec<PrefixCheckpoint> {
        std::mem::take(&mut self.pending_displaced)
    }

    /// Look up the LONGEST frozen prefix fully proven by `prompt`
    /// (fingerprint fast reject + full token equality). Touches the winner
    /// (LRU). Returns the checkpoint id.
    pub fn lookup(&mut self, prompt: &[u32], dst_layout: u64) -> Option<u64> {
        let mut best: Option<(u64, u32)> = None;
        for e in &self.entries {
            if e.layout_id != dst_layout || e.frontier as usize > prompt.len() {
                continue;
            }
            let f = e.frontier as usize;
            if prefix_fingerprint(&prompt[..f]) != e.fingerprint {
                continue;
            }
            if prompt[..f] != e.prefix_tokens[..] {
                continue;
            }
            if best.map(|(_, bf)| e.frontier > bf).unwrap_or(true) {
                best = Some((e.id, e.frontier));
            }
        }
        if let Some((id, _)) = best {
            self.touch(id);
            Some(id)
        } else {
            None
        }
    }

    /// Look up the LONGEST frozen prefix fully proven by `prompt` whose
    /// frontier does not exceed `max_frontier`. Import narrowing uses this to
    /// fall back to an earlier coverage checkpoint when the longest one cannot
    /// start inside the layerwise window. Touches the winner (LRU).
    pub fn lookup_at_most(&mut self, prompt: &[u32], dst_layout: u64, max_frontier: u32) -> Option<u64> {
        let mut best: Option<(u64, u32)> = None;
        for e in &self.entries {
            if e.layout_id != dst_layout || e.frontier > max_frontier || e.frontier as usize > prompt.len() {
                continue;
            }
            let f = e.frontier as usize;
            if prefix_fingerprint(&prompt[..f]) != e.fingerprint {
                continue;
            }
            if prompt[..f] != e.prefix_tokens[..] {
                continue;
            }
            if best.map(|(_, bf)| e.frontier > bf).unwrap_or(true) {
                best = Some((e.id, e.frontier));
            }
        }
        if let Some((id, _)) = best {
            self.touch(id);
            Some(id)
        } else {
            None
        }
    }

    pub fn get(&self, id: u64) -> Option<&PrefixCheckpoint> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// P3.4: prune the catalog with the core coverage-retention policy
    /// (count/byte caps + interval/min_gap). Returns the removed entries so the
    /// engine can release their GPU cache refs. A policy error falls back to
    /// keeping every entry (never silently drops a reusable prefix).
    pub fn prune_policy(
        &mut self,
        count: usize,
        bytes: u64,
        interval: u64,
        min_gap: u64,
    ) -> Vec<PrefixCheckpoint> {
        use hipfire_kvmem::{retain_checkpoint_coverage, CoverageInfo, CoveragePurpose};
        let infos: Vec<CoverageInfo> = self
            .entries
            .iter()
            .map(|e| CoverageInfo {
                frontier: e.frontier as u64,
                bytes: e.host_bytes(),
                used: e.last_used,
                identity: e.id,
                purpose: CoveragePurpose::Periodic as u32,
                protections: 0,
            })
            .collect();
        let keep = match retain_checkpoint_coverage(
            &infos,
            count as u64,
            bytes,
            interval,
            min_gap,
        ) {
            Ok(k) => k,
            Err(_) => return Vec::new(),
        };
        let mut removed = Vec::new();
        let mut kept = Vec::with_capacity(self.entries.len());
        for (index, entry) in std::mem::take(&mut self.entries).into_iter().enumerate() {
            if keep.contains(&index) {
                kept.push(entry);
            } else {
                removed.push(entry);
            }
        }
        self.entries = kept;
        removed
    }

    /// Exact-frontier dedup probe: an entry with the same frontier AND
    /// fingerprint already exists (touch it). The engine uses this BEFORE
    /// sealing GPU pages so a deduped freeze never doubles cache refs.
    pub fn find_exact(&mut self, frontier: u32, fingerprint: u64) -> Option<u64> {
        let id = self
            .entries
            .iter()
            .find(|e| e.frontier == frontier && e.fingerprint == fingerprint)
            .map(|e| e.id)?;
        self.touch(id);
        Some(id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut PrefixCheckpoint> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    /// Remove an entry, returning it (caller releases its GPU refs).
    pub fn remove(&mut self, id: u64) -> Option<PrefixCheckpoint> {
        let pos = self.entries.iter().position(|e| e.id == id)?;
        Some(self.entries.remove(pos))
    }

    /// Pop the least-recently-used entry (admission pressure reclaim),
    /// returning it so the caller can release its GPU cache refs. Pages
    /// still held by live block tables survive the ref release.
    pub fn pop_lru(&mut self) -> Option<PrefixCheckpoint> {
        let victim = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(i, _)| i)?;
        Some(self.entries.remove(victim))
    }

    fn touch(&mut self, id: u64) {
        self.clock += 1;
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.last_used = self.clock;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_kvmem::Storage;

    const BLOCK: u32 = 128;
    const LAYERS: u32 = 2;
    const K_WIDTH: usize = 4;
    const ROW_BYTES: usize = 36;

    fn raw_config() -> RawStoreConfig {
        RawStoreConfig {
            n_layer: LAYERS,
            n_embd_k: K_WIDTH as u32,
            n_embd_v: K_WIDTH as u32,
            block_tokens: BLOCK,
            k_row_bytes: 0,
            k_gpu_row_bytes: ROW_BYTES as u64,
            v_gpu_row_bytes: ROW_BYTES as u64,
        }
    }

    fn storage() -> Storage {
        Storage::new_ram(1 << 30).expect("storage")
    }

    /// Fill `n` full blocks of mean-K + packed K/V (deterministic content).
    fn fill(store: &RawStore, blocks: u32) {
        for b in 0..blocks {
            let pos0 = b * BLOCK;
            for il in 0..LAYERS {
                let mut mean = Vec::with_capacity(BLOCK as usize * K_WIDTH);
                for t in 0..BLOCK as usize {
                    for d in 0..K_WIDTH {
                        mean.push((b as usize * 1000 + t * K_WIDTH + d + il as usize * 7) as f32);
                    }
                }
                store.write_layer_mean_k(pos0, BLOCK, il, &mean).expect("mean k");
                let mut rows = vec![0u8; BLOCK as usize * ROW_BYTES];
                for (i, byte) in rows.iter_mut().enumerate() {
                    *byte = ((b as usize + il as usize + i) % 251) as u8;
                }
                store.write_layer_k_gpu(pos0, BLOCK, il, &rows).expect("packed k");
                store.write_layer_v_gpu(pos0, BLOCK, il, &rows).expect("packed v");
            }
        }
    }

    fn checkpoint(storage: &Storage, id: u64, frontier: u32, tokens: &[u32]) -> PrefixCheckpoint {
        let source = RawStore::new(raw_config(), Some(storage)).expect("source");
        fill(&source, frontier / BLOCK);
        let canonical = source.fork_prefix(frontier).expect("freeze fork");
        PrefixCheckpoint {
            id,
            frontier,
            prefix_tokens: tokens.to_vec(),
            fingerprint: prefix_fingerprint(tokens),
            mean_tail: None,
            source_incarnation: 1,
            layout_id: layout_fingerprint(&raw_config()),
            gpu_pages: Vec::new(),
            original_query: None,
            canonical,
            draft_image: None,
            last_used: 0,
        }
    }

    fn tokens(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect()
    }

    #[test]
    fn layout_fingerprint_detects_drift() {
        let base = layout_fingerprint(&raw_config());
        let mut drifted = raw_config();
        drifted.n_layer += 1;
        assert_ne!(base, layout_fingerprint(&drifted));
        let mut drifted = raw_config();
        drifted.k_gpu_row_bytes += 1;
        assert_ne!(base, layout_fingerprint(&drifted));
        assert_eq!(base, layout_fingerprint(&raw_config()));
    }

    #[test]
    fn validate_reuse_requires_full_token_proof() {
        let st = storage();
        let toks = tokens(256);
        let ckpt = checkpoint(&st, 1, 256, &toks);
        let layout = layout_fingerprint(&raw_config());
        // Exact prefix (longer prompt) passes.
        let mut prompt = toks.clone();
        prompt.extend_from_slice(&[9, 9, 9]);
        assert!(ckpt.validate_reuse(&prompt, layout).is_ok());
        // A single flipped token inside the frontier fails BOTH the
        // fingerprint and the full proof.
        let mut bad = toks.clone();
        bad[100] ^= 1;
        assert!(ckpt.validate_reuse(&bad, layout).is_err());
        // Short prompt fails.
        assert!(ckpt.validate_reuse(&toks[..128], layout).is_err());
        // Geometry drift fails even with a perfect token proof.
        assert!(ckpt.validate_reuse(&toks, layout ^ 1).is_err());
    }

    #[test]
    fn fork_shares_packed_and_keeps_mean_private() {
        let st = storage();
        let toks = tokens(256);
        let ckpt = checkpoint(&st, 1, 256, &toks);
        // Branch = canonical.fork_prefix(frontier): the reuse operation.
        let branch = ckpt.canonical().fork_prefix(256).expect("reuse fork");
        // Packed payload round-trips byte-exact through the shared sources.
        for il in 0..LAYERS {
            let mut src = vec![0u8; BLOCK as usize * ROW_BYTES];
            let mut dst = vec![0u8; BLOCK as usize * ROW_BYTES];
            assert!(ckpt.canonical().copy_k_gpu(1, il, &mut src, BLOCK));
            assert!(branch.copy_k_gpu(1, il, &mut dst, BLOCK));
            assert_eq!(src, dst, "shared packed K block 1 layer {il}");
        }
        // Mean-K is private: writing the branch must not touch the canonical.
        let mut before = vec![0f32; K_WIDTH];
        ckpt.canonical().mean_k(0, 0, &mut before);
        let new_mean = vec![42f32; BLOCK as usize * K_WIDTH];
        branch.write_layer_mean_k(0, BLOCK, 0, &new_mean).expect("branch mean write");
        let mut after = vec![0f32; K_WIDTH];
        ckpt.canonical().mean_k(0, 0, &mut after);
        assert_eq!(before, after, "canonical mean-K must stay immutable");
        let mut branch_mean = vec![0f32; K_WIDTH];
        branch.mean_k(0, 0, &mut branch_mean);
        assert!(branch_mean.iter().all(|&v| v == 42.0));
    }

    #[test]
    fn catalog_lookup_prefers_longest_proven_prefix() {
        let st = storage();
        let toks = tokens(512);
        let mut cat = PrefixCatalog::new(4);
        let short = checkpoint(&st, cat.mint_id(), 128, &toks[..128]);
        let long = checkpoint(&st, cat.mint_id(), 384, &toks[..384]);
        cat.insert(short).expect("insert short");
        cat.insert(long).expect("insert long");
        let layout = layout_fingerprint(&raw_config());
        // Prompt covering both picks the longest.
        assert_eq!(cat.lookup(&toks, layout), Some(2));
        // Prompt covering only the short one picks it.
        assert_eq!(cat.lookup(&toks[..256], layout), Some(1));
        // A mismatching prompt picks nothing.
        let mut bad = toks.clone();
        bad[10] ^= 1;
        assert_eq!(cat.lookup(&bad, layout), None);
        // Wrong geometry picks nothing.
        assert_eq!(cat.lookup(&toks, layout ^ 1), None);
    }

    #[test]
    fn catalog_lookup_at_most_caps_the_frontier() {
        let st = storage();
        let toks = tokens(512);
        let mut cat = PrefixCatalog::new(4);
        let short = checkpoint(&st, cat.mint_id(), 128, &toks[..128]);
        let long = checkpoint(&st, cat.mint_id(), 384, &toks[..384]);
        cat.insert(short).expect("insert short");
        cat.insert(long).expect("insert long");
        let layout = layout_fingerprint(&raw_config());
        assert_eq!(cat.lookup_at_most(&toks, layout, 384), Some(2));
        assert_eq!(cat.lookup_at_most(&toks, layout, 383), Some(1));
        assert_eq!(cat.lookup_at_most(&toks, layout, 128), Some(1));
        assert_eq!(cat.lookup_at_most(&toks, layout, 127), None);
        // The full token proof still gates the capped lookup.
        let mut bad = toks.clone();
        bad[10] ^= 1;
        assert_eq!(cat.lookup_at_most(&bad, layout, 512), None);
        assert_eq!(cat.lookup_at_most(&toks, layout ^ 1, 512), None);
    }

    #[test]
    fn catalog_prune_policy_keeps_the_policy_selected_entries() {
        let st = storage();
        let toks = tokens(512);
        let mut cat = PrefixCatalog::new(4);
        for frontier in [128u32, 256, 384] {
            let ckpt = checkpoint(&st, cat.mint_id(), frontier, &toks[..frontier as usize]);
            cat.insert(ckpt).expect("insert");
        }
        assert_eq!(cat.len(), 3);
        // Count=1 keeps the newest frontier under the core cost heuristic.
        let removed = cat.prune_policy(1, 1 << 30, 4096, 0);
        assert_eq!(removed.len(), 2);
        assert_eq!(cat.len(), 1);
        let remaining = cat.lookup(&toks[..384], layout_fingerprint(&raw_config()));
        assert!(remaining.is_some(), "the newest (longest) prefix survives");
        // The same selection is idempotent.
        assert!(cat.prune_policy(1, 1 << 30, 4096, 0).is_empty());
    }

    #[test]
    fn catalog_lru_eviction_returns_displaced() {
        let st = storage();
        let toks = tokens(512);
        let mut cat = PrefixCatalog::new(2);
        let layout = layout_fingerprint(&raw_config());
        for (i, frontier) in [128u32, 256, 384].iter().enumerate() {
            let ckpt = checkpoint(&st, cat.mint_id(), *frontier, &toks[..*frontier as usize]);
            let (id, displaced) = cat.insert_displacing(ckpt).expect("insert");
            assert_eq!(id, Some(i as u64 + 1));
            if i < 2 {
                assert!(displaced.is_empty());
            } else {
                // The third insert evicts the LRU entry (id 1 — untouched
                // since insert; lookup below would have touched it).
                assert_eq!(displaced.len(), 1);
                assert_eq!(displaced[0].id, 1);
            }
        }
        assert_eq!(cat.len(), 2);
        assert!(cat.get(1).is_none());
        assert!(cat.get(2).is_some());
        assert!(cat.get(3).is_some());
        // Touch id 2 via lookup, then insert a fourth: id 3 is now the LRU.
        assert_eq!(cat.lookup(&toks[..256], layout), Some(2));
        let ckpt = checkpoint(&st, cat.mint_id(), 512, &toks);
        let (_, displaced) = cat.insert_displacing(ckpt).expect("insert 4");
        assert_eq!(displaced.len(), 1);
        assert_eq!(displaced[0].id, 3);
    }

    #[test]
    fn catalog_dedupes_same_frontier_and_proof() {
        let st = storage();
        let toks = tokens(256);
        let mut cat = PrefixCatalog::new(4);
        let a = checkpoint(&st, cat.mint_id(), 256, &toks);
        let b = checkpoint(&st, cat.mint_id(), 256, &toks);
        assert_eq!(cat.insert(a).expect("first"), Some(1));
        assert_eq!(cat.insert(b).expect("dedup"), None);
        assert_eq!(cat.len(), 1);
    }

    #[test]
    fn catalog_remove_hands_entry_back() {
        let st = storage();
        let toks = tokens(128);
        let mut cat = PrefixCatalog::new(4);
        let ckpt = checkpoint(&st, cat.mint_id(), 128, &toks);
        cat.insert(ckpt).expect("insert");
        let removed = cat.remove(1).expect("remove");
        assert_eq!(removed.frontier, 128);
        assert!(cat.is_empty());
        assert!(cat.remove(1).is_none());
    }
}