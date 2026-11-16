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

use hipfire_kvmem::{
    plan_content_group_publication, ContentSlot, PublicationReason, RawStore, RawStoreConfig,
};
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

/// 期3 G3 P1.1: adapter metadata of one persistent prefix pack. The pack
/// header (geometry + per-block tokens) lives in the shim; this JSON blob is
/// opaque to it and carries the reuse proof and identity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContentPackMeta {
    pub version: u32,
    pub layout_id: u64,
    pub frontier: u32,
    pub tokens: Vec<u32>,
    pub mean_tail: Option<Vec<f32>>,
    pub group: u64,
    pub original_query: Option<(u64, u64)>,
    pub n_record_layers: u32,
    /// G7: media identity of the image turn inside this prefix (`None` for
    /// text-only checkpoints) and its image token spans (absolute prompt
    /// coordinates). Older packs decode as text-only (serde defaults).
    #[serde(default)]
    pub media_digest: Option<u64>,
    #[serde(default)]
    pub media_spans: Vec<(u32, u32)>,
}

impl ContentPackMeta {
    pub const VERSION: u32 = 1;
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("content pack meta: {e}"))
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("content pack meta: {e}"))
    }
}

/// One pack file discovered in the content directory (metadata only; the
/// payload is loaded on demand).
#[derive(Debug, Clone)]
pub struct ColdPack {
    pub path: std::path::PathBuf,
    pub meta: ContentPackMeta,
    pub fingerprint: u64,
    pub bytes: u64,
    /// Monotonic LRU stamp assigned at scan (file mtime order).
    pub stamp: u64,
}

/// Cold content index: persistent pack files in `HIPFIRE_KVMEM_CACHE_DIR`.
/// A hit loads the payload into a fresh store and publishes a normal catalog
/// entry (the engine then reuses it through the usual proof path).
#[derive(Debug, Default)]
pub struct ColdPackIndex {
    packs: Vec<ColdPack>,
    clock: u64,
}

impl ColdPackIndex {
    pub fn len(&self) -> usize {
        self.packs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.packs.is_empty()
    }
    pub fn total_bytes(&self) -> u64 {
        self.packs.iter().map(|p| p.bytes).sum()
    }
    pub fn packs(&self) -> &[ColdPack] {
        &self.packs
    }

    /// Scan `dir` for `*.hkpack` files, keeping only packs whose geometry and
    /// layout match this engine and whose metadata is well-formed. Files are
    /// ordered by mtime (oldest first) for byte-cap pruning.
    pub fn scan(
        dir: &std::path::Path,
        layout_id: u64,
        n_record_layers: u32,
        block_tokens: u32,
        max_meta_bytes: u64,
    ) -> Self {
        let mut index = Self::default();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return index;
        };
        let mut raw: Vec<(std::time::SystemTime, ColdPack)> = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("hkpack") {
                continue;
            }
            let Ok(info) = RawStore::prefix_pack_info(&path) else {
                continue;
            };
            if info.meta_bytes == 0 || info.meta_bytes > max_meta_bytes {
                continue;
            }
            let Ok(meta_bytes) = RawStore::prefix_pack_meta(&path) else {
                continue;
            };
            let Ok(meta) = ContentPackMeta::decode(&meta_bytes) else {
                continue;
            };
            if meta.version != ContentPackMeta::VERSION
                || meta.layout_id != layout_id
                || meta.n_record_layers != n_record_layers
                || meta.frontier == 0
                || meta.frontier % block_tokens != 0
                || meta.tokens.len() != meta.frontier as usize
            {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            raw.push((
                modified,
                ColdPack {
                    fingerprint: prefix_fingerprint(&meta.tokens),
                    path,
                    meta,
                    bytes: info.file_bytes,
                    stamp: 0,
                },
            ));
        }
        raw.sort_by_key(|(m, _)| *m);
        index.clock = raw.len() as u64;
        index.packs = raw
            .into_iter()
            .enumerate()
            .map(|(i, (_, mut p))| {
                p.stamp = i as u64;
                p
            })
            .collect();
        index
    }

    /// Longest pack whose FULL token proof matches `prompt` (same rule as the
    /// hot catalog; a fingerprint alone is never the admission proof).
    pub fn lookup(&self, prompt: &[u32], layout_id: u64) -> Option<usize> {
        self.lookup_media(prompt, layout_id, None)
    }

    /// Media-aware cold-pack lookup (G7): same full token proof, plus the
    /// pack's media digest must match when it covers a media span.
    pub fn lookup_media(
        &self,
        prompt: &[u32],
        layout_id: u64,
        media: Option<u64>,
    ) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, p) in self.packs.iter().enumerate() {
            if p.meta.layout_id != layout_id || p.meta.frontier as usize > prompt.len() {
                continue;
            }
            if !meta_media_eligible(
                p.meta.frontier,
                &p.meta.media_spans,
                p.meta.media_digest,
                media,
            ) {
                continue;
            }
            let f = p.meta.frontier as usize;
            if prefix_fingerprint(&prompt[..f]) != p.fingerprint {
                continue;
            }
            if prompt[..f] != p.meta.tokens[..] {
                continue;
            }
            if best
                .map(|b| p.meta.frontier > self.packs[b].meta.frontier)
                .unwrap_or(true)
            {
                best = Some(i);
            }
        }
        best
    }

    /// G7 Stage E probe: the longest token-proven pack the media identity
    /// gate rejected (`None` when the token-best pack is eligible or nothing
    /// proves). Read-only — no LRU touch, no counters; trace evidence only.
    pub fn probe_media_reject(
        &self,
        prompt: &[u32],
        layout_id: u64,
        media: Option<u64>,
    ) -> Option<(u32, Option<u64>)> {
        let mut best: Option<&ColdPack> = None;
        for p in &self.packs {
            if p.meta.layout_id != layout_id || p.meta.frontier as usize > prompt.len() {
                continue;
            }
            if meta_media_eligible(
                p.meta.frontier,
                &p.meta.media_spans,
                p.meta.media_digest,
                media,
            ) {
                continue;
            }
            let f = p.meta.frontier as usize;
            if prefix_fingerprint(&prompt[..f]) != p.fingerprint {
                continue;
            }
            if prompt[..f] != p.meta.tokens[..] {
                continue;
            }
            if best
                .map(|b| p.meta.frontier > b.meta.frontier)
                .unwrap_or(true)
            {
                best = Some(p);
            }
        }
        best.map(|p| (p.meta.frontier, p.meta.media_digest))
    }

    /// Delete the oldest packs until the index fits `max_bytes`. Returns the
    /// removed files so callers can log/metrics them.
    pub fn prune_to(&mut self, max_bytes: u64) -> Vec<std::path::PathBuf> {
        let mut removed = Vec::new();
        while self.total_bytes() > max_bytes && !self.packs.is_empty() {
            let victim = self
                .packs
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| p.stamp)
                .map(|(i, _)| i)
                .expect("nonempty");
            let p = self.packs.remove(victim);
            let _ = std::fs::remove_file(&p.path);
            removed.push(p.path);
        }
        removed
    }

    /// Register one freshly written pack (LRU stamp = newest).
    pub fn insert(&mut self, mut pack: ColdPack) {
        self.clock += 1;
        pack.stamp = self.clock;
        self.packs.push(pack);
    }

    pub fn contains_path(&self, path: &std::path::Path) -> bool {
        self.packs.iter().any(|p| p.path == path)
    }

    pub fn remove(&mut self, index: usize) -> Option<ColdPack> {
        if index < self.packs.len() {
            Some(self.packs.remove(index))
        } else {
            None
        }
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
    /// 期3 G3: internal content group (conversation-lineage hash). A capacity
    /// FAIRNESS hint only — it never authorizes reuse or overwrites a proof.
    /// 0 = unknown origin (independent group, as in the core policy).
    pub group: u64,
    /// G7: media identity of the image turn inside this prefix (`None` for
    /// text-only checkpoints). A token proof cannot distinguish two images, so
    /// a checkpoint covering a media span is eligible only when the incoming
    /// request's digest equals this one.
    pub media_digest: Option<u64>,
    /// G7: image token spans (absolute prompt coordinates) inside the prefix.
    pub media_spans: Vec<(u32, u32)>,
    /// 期3 G3 P1.1: persistent pack file backing this checkpoint (set at
    /// freeze when the live frontier matches, and on a cold load). The DN
    /// sidecar sits beside it (`<pack>.dn`).
    pub pack_path: Option<std::path::PathBuf>,
    /// Immutable canonical archive: `source.fork_prefix(frontier)`. Packed
    /// records are sealed shared sources; mean-K is private to this archive
    /// and re-forked private into every branch. Never fed after freeze.
    canonical: RawStore,
    /// 期3 G4: the same canonical bytes as a core `KvSnapshot` (native-proven
    /// sealed block origins + exact spans). The fast reuse path stays
    /// `fork_prefix`; the snapshot is the catalog-visible immutable object the
    /// G2/G3 directories share. `None` when the store had no packed archive
    /// (host-only fixtures) — production q8/non-q8 archives always carry one.
    pub kv_snapshot: Option<hipfire_kvmem::KvSnapshot>,
    /// 期4: DFlash2 draft conditioning image captured with this checkpoint
    /// (`None` for AR/MTP sessions, non-windowed drafts, or a frontier that
    /// did not equal the draft's live seeded frontier — those reseed).
    pub draft_image: Option<DraftImage>,
    /// LRU clock stamp (catalog-internal).
    last_used: u64,
}

/// G7 media eligibility: a prefix without a fully-inside media span is
/// reusable by any request; a prefix covering an image span requires an
/// identical, present media digest (token proofs cannot distinguish images).
pub fn meta_media_eligible(
    frontier: u32,
    spans: &[(u32, u32)],
    digest: Option<u64>,
    media: Option<u64>,
) -> bool {
    let covers = spans.iter().any(|&(_, end)| end <= frontier);
    !covers || (digest.is_some() && digest == media)
}

impl PrefixCheckpoint {
    /// G7: see [`meta_media_eligible`].
    pub fn media_eligible(&self, media: Option<u64>) -> bool {
        meta_media_eligible(self.frontier, &self.media_spans, self.media_digest, media)
    }
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
            group: 0,
            media_digest: None,
            media_spans: Vec::new(),
            pack_path: None,
            canonical,
            kv_snapshot: None,
            draft_image: None,
            last_used: 0,
        }
    }

    /// Host bytes charged for retention (canonical packed archive + full
    /// snapshot metadata + the optional draft image). Gross view: snapshot
    /// origins shared with another catalog entry are charged again.
    pub fn host_bytes(&self) -> u64 {
        self.canonical.payload_bytes()
            + self.kv_snapshot.as_ref().map_or(0, |snap| snap.metadata_bytes() as u64)
            + self.draft_image.as_ref().map_or(0, |img| img.nbytes())
    }

    /// The part of [`Self::host_bytes`] this entry owns exclusively: canonical
    /// payload, draft image, and the snapshot's private manifest. The remaining
    /// charge is per-origin (`SealedKvBlock`) and is attributed once by
    /// [`PrefixCatalog::host_accounting`] because successive freezes of one
    /// session inherit the same origins.
    pub fn own_host_bytes(&self) -> u64 {
        self.canonical.payload_bytes()
            + self.kv_snapshot.as_ref().map_or(0, |snap| snap.manifest_bytes() as u64)
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

/// Host-byte views of the catalog (see [`PrefixCatalog::host_accounting`]).
pub struct HostAccounting {
    /// Sum of every entry's gross [`PrefixCheckpoint::host_bytes`].
    pub gross: u64,
    /// Origin-shared view: each `SealedKvBlock` charged once across entries.
    pub unique: u64,
    /// Per-entry unique charge, aligned with the catalog's entry order.
    pub per_entry_unique: Vec<u64>,
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
    /// archive for the same prefix). The catalog is token-keyed on purpose:
    /// two images render the same token stream, and the media digest (not the
    /// catalog slot) decides whether an entry may be served.
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

    /// 期3 G3: group-aware publication through the core content-directory
    /// policy (`plan_content_group_publication`): internal content groups are a
    /// capacity fairness hint; the whole least-recently-used group retires
    /// before entry trimming, and a redundant version is trimmed before
    /// another group's sole version. Reuse-proof semantics are unchanged (the
    /// caller still validates tokens/layout). Returns the inserted id (None
    /// when deduped), the retired entries (the engine releases their GPU
    /// refs), and the dedup flag.
    pub fn insert_group_aware(
        &mut self,
        mut entry: PrefixCheckpoint,
        max_groups: usize,
    ) -> Result<(Option<u64>, Vec<PrefixCheckpoint>, bool), String> {
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
            return Ok((None, Vec::new(), true));
        }
        let slots: Vec<ContentSlot> = self
            .entries
            .iter()
            .map(|e| ContentSlot {
                occupied: 1,
                pinned: 0,
                last_used: e.last_used,
                identity: 0,
                group: e.group,
            })
            .collect();
        let (slot, reason, retired) =
            plan_content_group_publication(&slots, self.max_entries, max_groups, entry.group)
                .map_err(|e| format!("kvmem content directory: {e}"))?;
        if slot.is_none() {
            return Err(format!(
                "kvmem content directory refused publication ({reason:?})"
            ));
        }
        let mut displaced = Vec::with_capacity(retired.len());
        for &index in retired.iter().rev() {
            if index < self.entries.len() {
                displaced.push(self.entries.remove(index));
            }
        }
        if reason == PublicationReason::Deduplicated {
            // The plan matched an identity this catalog does not mint (the
            // dedup probe above is the authoritative one); never append twice.
            return Ok((None, displaced, true));
        }
        self.clock += 1;
        entry.last_used = self.clock;
        let id = entry.id;
        self.entries.push(entry);
        Ok((Some(id), displaced, false))
    }

    /// Distinct non-zero content groups currently retained.
    pub fn group_count(&self) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        for e in &self.entries {
            if e.group != 0 {
                seen.insert(e.group);
            }
        }
        seen.len()
    }

    /// Drain any displaced entries left by [`Self::insert`] (test-facing).
    pub fn take_displaced(&mut self) -> Vec<PrefixCheckpoint> {
        std::mem::take(&mut self.pending_displaced)
    }

    /// Look up the LONGEST frozen prefix fully proven by `prompt`
    /// (fingerprint fast reject + full token equality). Touches the winner
    /// (LRU). Returns the checkpoint id. Text-only lookup (`media` None):
    /// checkpoints covering an image span are never eligible.
    pub fn lookup(&mut self, prompt: &[u32], dst_layout: u64) -> Option<u64> {
        self.lookup_media(prompt, dst_layout, None)
    }

    /// Media-aware lookup (G7): identical proof, additionally requiring the
    /// checkpoint's media digest to match when the prefix covers a media span.
    pub fn lookup_media(
        &mut self,
        prompt: &[u32],
        dst_layout: u64,
        media: Option<u64>,
    ) -> Option<u64> {
        let mut best: Option<(u64, u32)> = None;
        for e in &self.entries {
            if e.layout_id != dst_layout || e.frontier as usize > prompt.len() {
                continue;
            }
            if !e.media_eligible(media) {
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
        self.lookup_at_most_media(prompt, dst_layout, max_frontier, None)
    }

    /// Media-aware [`Self::lookup_at_most`] (G7).
    pub fn lookup_at_most_media(
        &mut self,
        prompt: &[u32],
        dst_layout: u64,
        max_frontier: u32,
        media: Option<u64>,
    ) -> Option<u64> {
        let mut best: Option<(u64, u32)> = None;
        for e in &self.entries {
            if e.layout_id != dst_layout || e.frontier > max_frontier || e.frontier as usize > prompt.len() {
                continue;
            }
            if !e.media_eligible(media) {
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

    /// G7 Stage E probe: the longest token-proven checkpoint the media
    /// identity gate rejected. Read-only — no LRU touch, no counters; trace
    /// evidence for the cross-image miss gate.
    pub fn probe_media_reject(
        &self,
        prompt: &[u32],
        dst_layout: u64,
        media: Option<u64>,
    ) -> Option<(u32, Option<u64>)> {
        let mut best: Option<&PrefixCheckpoint> = None;
        for e in &self.entries {
            if e.layout_id != dst_layout || e.frontier as usize > prompt.len() {
                continue;
            }
            if e.media_eligible(media) {
                continue;
            }
            let f = e.frontier as usize;
            if prefix_fingerprint(&prompt[..f]) != e.fingerprint {
                continue;
            }
            if prompt[..f] != e.prefix_tokens[..] {
                continue;
            }
            if best.map(|b| e.frontier > b.frontier).unwrap_or(true) {
                best = Some(e);
            }
        }
        best.map(|e| (e.frontier, e.media_digest))
    }

    /// Catalog-wide host-byte accounting. `gross` sums every entry's
    /// [`PrefixCheckpoint::host_bytes`]; `unique` charges a `SealedKvBlock`
    /// origin once across entries that share it (a session's successive
    /// freezes inherit origins, patch 39). Shared origins are attributed to
    /// the first entry referencing them in catalog order, so the per-entry
    /// vector stays deterministic for the retention byte cap.
    pub fn host_accounting(&self) -> HostAccounting {
        let origin_charge = hipfire_kvmem::KvSnapshot::origin_charge_bytes();
        let mut seen = std::collections::HashSet::new();
        let mut gross = 0u64;
        let mut unique = 0u64;
        let mut per_entry_unique = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            gross = gross.saturating_add(entry.host_bytes());
            let mut cost = entry.own_host_bytes();
            if let Some(snap) = entry.kv_snapshot.as_ref() {
                for block in 0..snap.block_count() {
                    if let Some(origin) = snap.origin_id(block) {
                        if seen.insert(origin) {
                            cost = cost.saturating_add(origin_charge);
                        }
                    }
                }
            }
            unique = unique.saturating_add(cost);
            per_entry_unique.push(cost);
        }
        HostAccounting {
            gross,
            unique,
            per_entry_unique,
        }
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
        let accounting = self.host_accounting();
        let infos: Vec<CoverageInfo> = self
            .entries
            .iter()
            .zip(accounting.per_entry_unique.iter())
            .map(|(e, unique)| CoverageInfo {
                frontier: e.frontier as u64,
                bytes: *unique,
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
    /// Exact (frontier, token proof) match. Media identity is checked by the
    /// caller's [`PrefixCheckpoint::media_eligible`] gate, never here.
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

    /// Residency demotion (G3 follow-up): drain the LRU entry's pinned pages
    /// and KEEP the entry in the catalog, so a later reuse restores pages from
    /// the sealed archive (Layerwise). Entries already demoted (no pages) are
    /// skipped. Returns `(id, pages)` or `None` when nothing is pinned.
    pub fn demote_lru(&mut self) -> Option<(u64, Vec<u32>)> {
        let victim = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.gpu_pages.is_empty())
            .min_by_key(|(_, e)| e.last_used)
            .map(|(i, _)| i)?;
        let entry = &mut self.entries[victim];
        Some((entry.id, std::mem::take(&mut entry.gpu_pages)))
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
            group: 0,
            media_digest: None,
            media_spans: Vec::new(),
            pack_path: None,
            canonical,
            kv_snapshot: None,
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

    /// G7: a checkpoint covering an image span is eligible only for the same
    /// media digest; text-only checkpoints stay eligible for everyone.
    #[test]
    fn catalog_media_gate_requires_the_same_digest() {
        let st = storage();
        let toks = tokens(256);
        let mut cat = PrefixCatalog::new(4);
        let layout = layout_fingerprint(&raw_config());
        let mut imaged = checkpoint(&st, cat.mint_id(), 128, &toks[..128]);
        imaged.media_digest = Some(0x11);
        imaged.media_spans = vec![(0, 64)];
        let plain = checkpoint(&st, cat.mint_id(), 256, &toks);
        cat.insert(imaged).expect("insert imaged");
        cat.insert(plain).expect("insert plain");
        assert_eq!(
            cat.lookup_media(&toks[..128], layout, None),
            None,
            "text-only requests must not reuse an image checkpoint"
        );
        assert_eq!(cat.lookup_media(&toks[..128], layout, Some(0x11)), Some(1));
        assert_eq!(cat.lookup_media(&toks[..128], layout, Some(0x22)), None);
        assert_eq!(
            cat.lookup(&toks, layout),
            Some(2),
            "text-only checkpoints remain eligible"
        );
        // The capped variant applies the same gate.
        assert_eq!(cat.lookup_at_most_media(&toks, layout, 128, None), None);
        assert_eq!(
            cat.lookup_at_most_media(&toks, layout, 128, Some(0x11)),
            Some(1)
        );
    }

    /// 期3 G3 follow-up: catalog accounting charges a `SealedKvBlock` origin
    /// once across entries that inherited it, and the retention byte cap sees
    /// that unique total (gross per-entry bytes would prune a sharer).
    #[test]
    fn catalog_accounting_dedups_shared_snapshot_origins() {
        let st = storage();
        let toks = tokens(256);
        let mut cat = PrefixCatalog::new(4);
        let source = RawStore::new(raw_config(), Some(&st)).expect("source");
        fill(&source, 2);
        let mut first = checkpoint(&st, cat.mint_id(), 128, &toks[..128]);
        first.canonical = source.fork_prefix(128).expect("canonical 128");
        let snap1 = first
            .canonical()
            .capture_snapshot(1, None)
            .expect("snapshot 128");
        let shared_origin = snap1.origin_id(0).expect("origin 0");
        assert_eq!(snap1.block_count(), 4, "1 block x 2 layers x (K,V)");
        first.kv_snapshot = Some(snap1);
        let mut second = checkpoint(&st, cat.mint_id(), 256, &toks);
        second.canonical = source.fork_prefix(256).expect("canonical 256");
        let snap2 = second
            .canonical()
            .capture_snapshot(2, first.kv_snapshot.as_ref())
            .expect("snapshot 256");
        assert_eq!(snap2.block_count(), 8);
        assert_eq!(
            snap2.origin_id(0),
            Some(shared_origin),
            "block 0 origins inherited from the first snapshot"
        );
        assert_eq!(snap2.origin_id(2), first.kv_snapshot.as_ref().unwrap().origin_id(2));
        assert_ne!(snap2.origin_id(4), Some(shared_origin), "block 1 is a new origin");
        second.kv_snapshot = Some(snap2);
        let (first_id, second_id) = (first.id, second.id);
        cat.insert(first).expect("insert first");
        cat.insert(second).expect("insert second");

        let origin_charge = hipfire_kvmem::KvSnapshot::origin_charge_bytes();
        let first_entry = cat.get(first_id).expect("first entry");
        let second_entry = cat.get(second_id).expect("second entry");
        let gross = first_entry.host_bytes() + second_entry.host_bytes();
        let accounting = cat.host_accounting();
        assert_eq!(accounting.gross, gross);
        // 4 shared origins out of 12 are charged once.
        assert_eq!(accounting.gross - accounting.unique, 4 * origin_charge);
        assert_eq!(accounting.per_entry_unique.len(), 2);
        assert_eq!(
            accounting.per_entry_unique[0],
            first_entry.own_host_bytes() + 4 * origin_charge
        );
        assert_eq!(
            accounting.per_entry_unique[1],
            second_entry.own_host_bytes() + 4 * origin_charge
        );
        assert_eq!(
            accounting.unique,
            accounting.per_entry_unique.iter().sum::<u64>()
        );

        // A byte cap between the unique and gross totals retains both entries:
        // the shared origins must not be double-charged (gross would prune).
        assert!(gross > accounting.unique + 1, "test cap must discriminate");
        let removed = cat.prune_policy(4, accounting.unique + 1, u64::MAX / 2, 0);
        assert!(
            removed.is_empty(),
            "unique accounting keeps origin-sharing checkpoints under the byte cap"
        );
        assert_eq!(cat.len(), 2);
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

    /// G3 follow-up: admission-pressure demotion drops the LRU entry's pinned
    /// pages but keeps the catalog entry; already-demoted entries are skipped.
    #[test]
    fn catalog_demote_keeps_entry_and_drains_lru_pages() {
        let st = storage();
        let toks = tokens(256);
        let mut cat = PrefixCatalog::new(4);
        let mut first = checkpoint(&st, cat.mint_id(), 128, &toks[..128]);
        first.gpu_pages = vec![1, 2];
        let mut second = checkpoint(&st, cat.mint_id(), 256, &toks);
        second.gpu_pages = vec![3];
        cat.insert(first).expect("insert first");
        cat.insert(second).expect("insert second");
        // Touch the longer prefix so the shorter one is the LRU victim.
        assert!(cat.lookup(&toks, layout_fingerprint(&raw_config())).is_some());
        let (id, pages) = cat.demote_lru().expect("demote first");
        assert_eq!((id, pages), (1, vec![1, 2]));
        let kept = cat.get(1).expect("demoted entry stays cataloged");
        assert!(kept.gpu_pages.is_empty());
        assert_eq!(cat.len(), 2);
        let (id, pages) = cat.demote_lru().expect("demote second");
        assert_eq!((id, pages), (2, vec![3]));
        assert!(cat.demote_lru().is_none(), "nothing left pinned");
        assert_eq!(cat.len(), 2, "demotion never removes entries");
    }

    /// 期3 G3: group-aware publication retires the whole least-recently-used
    /// group before entry trimming; reuse proofs are unchanged.
    #[test]
    fn content_groups_retire_the_lru_whole_group() {
        let st = storage();
        let mut cat = PrefixCatalog::new(4);
        let mut publish = |cat: &mut PrefixCatalog, frontier: u32, seed: u32, group: u64| {
            let mut ck = checkpoint(&st, cat.mint_id(), frontier, &tokens(frontier as usize + seed as usize));
            ck.group = group;
            cat.insert_group_aware(ck, 2).expect("group publish")
        };
        // Group A owns two versions, group B is the newer sole version.
        let (a1, d1, dup1) = publish(&mut cat, 128, 0, 11);
        assert!(a1.is_some() && d1.is_empty() && !dup1);
        let (a2, d2, _) = publish(&mut cat, 256, 1, 11);
        assert!(a2.is_some() && d2.is_empty());
        let (b1, d3, _) = publish(&mut cat, 384, 2, 22);
        assert!(b1.is_some() && d3.is_empty());
        assert_eq!(cat.group_count(), 2);
        // Publishing a third group retires the LRU whole group (A: used 2 < B: 3).
        let (c1, retired, _) = publish(&mut cat, 512, 3, 33);
        assert!(c1.is_some());
        assert_eq!(retired.len(), 2, "both A versions retire with their group");
        assert!(retired.iter().all(|e| e.group == 11));
        assert_eq!(cat.group_count(), 2, "B and C remain");
        assert_eq!(cat.len(), 2, "B + C");
        // Dedup by exact frontier+proof still wins before the group plan.
        let mut dup = checkpoint(&st, cat.mint_id(), 512, &tokens(512 + 3));
        dup.group = 33;
        let (id, none, deduped) = cat.insert_group_aware(dup, 2).expect("dedup");
        assert!(id.is_none() && none.is_empty() && deduped);
    }

    /// G7 Stage C: the media digest scopes whether an entry may be served.
    /// Two images share the `image_pad` token stream, so the token proof alone
    /// never authorizes a media-covered prefix.
    #[test]
    fn media_digest_gates_serving() {
        let st = storage();
        let toks = tokens(256);
        let layout = layout_fingerprint(&raw_config());
        let mut cat = PrefixCatalog::new(4);
        let mut a = checkpoint(&st, cat.mint_id(), 128, &toks[..128]);
        a.media_digest = Some(0xA);
        a.media_spans = vec![(64, 128)];
        let (ida, _) = cat.insert_displacing(a).expect("insert a");
        let ida = ida.expect("stored");
        assert_eq!(cat.lookup_media(&toks[..128], layout, Some(0xA)), Some(ida));
        assert_eq!(cat.lookup_media(&toks[..128], layout, Some(0xB)), None);
        assert_eq!(cat.lookup_media(&toks[..128], layout, None), None);
        // A checkpoint frozen BEFORE the image carries no covered span and is
        // reusable by any request (it holds no image rows).
        let mut pre = checkpoint(&st, cat.mint_id(), 128, &toks[..128]);
        pre.media_digest = Some(0xA);
        pre.media_spans = vec![(200, 260)];
        assert!(pre.media_eligible(Some(0xB)));
        assert!(pre.media_eligible(None));
        // A text checkpoint is never eligible for a media-covered span.
        let mut t = checkpoint(&st, cat.mint_id(), 128, &toks[..128]);
        t.media_digest = None;
        t.media_spans = vec![(64, 128)];
        assert!(!t.media_eligible(Some(0xA)));
        assert!(!t.media_eligible(None));
    }

    /// G7 Stage E: the trace probe names the token-proven candidate that the
    /// media identity gate refused, so a cross-image miss is attributable in
    /// live logs (and is provably NOT a token-proof miss).
    #[test]
    fn media_probe_reports_identity_rejected_candidates() {
        let st = storage();
        let toks = tokens(256);
        let layout = layout_fingerprint(&raw_config());
        let mut cat = PrefixCatalog::new(4);
        let mut a = checkpoint(&st, cat.mint_id(), 128, &toks[..128]);
        a.media_digest = Some(0xA);
        a.media_spans = vec![(64, 128)];
        let (ida, _) = cat.insert_displacing(a).expect("insert a");
        assert!(ida.is_some());
        // Different image: the token proof matches, the media gate refuses.
        assert_eq!(
            cat.probe_media_reject(&toks[..128], layout, Some(0xB)),
            Some((128, Some(0xA)))
        );
        // Text continuation (no media identity) is refused the same way.
        assert_eq!(
            cat.probe_media_reject(&toks[..128], layout, None),
            Some((128, Some(0xA)))
        );
        // Same image (or an eligible text checkpoint) is not a rejection.
        assert_eq!(cat.probe_media_reject(&toks[..128], layout, Some(0xA)), None);
        // A broken token proof is a token miss, not a media rejection.
        let mut bad = toks[..128].to_vec();
        bad[7] ^= 1;
        assert_eq!(cat.probe_media_reject(&bad, layout, Some(0xB)), None);
    }

    /// G7 Stage E: the cold-index probe mirrors the hot one (packs can carry
    /// the media identity too, so a restart miss is attributable as well).
    #[test]
    fn media_probe_reports_cold_identity_rejects() {
        let toks = tokens(128);
        let layout = 0x5151_5151_5151_5151;
        let meta = ContentPackMeta {
            version: ContentPackMeta::VERSION,
            layout_id: layout,
            frontier: 128,
            tokens: toks.clone(),
            mean_tail: None,
            group: 0,
            original_query: None,
            n_record_layers: LAYERS,
            media_digest: Some(0xA),
            media_spans: vec![(64, 128)],
        };
        let mut idx = ColdPackIndex::default();
        idx.clock = 1;
        idx.packs.push(ColdPack {
            path: std::path::PathBuf::from("probe.hkpack"),
            fingerprint: prefix_fingerprint(&toks),
            meta,
            bytes: 1,
            stamp: 1,
        });
        assert_eq!(
            idx.probe_media_reject(&toks, layout, Some(0xB)),
            Some((128, Some(0xA)))
        );
        assert_eq!(idx.probe_media_reject(&toks, layout, Some(0xA)), None);
        assert_eq!(idx.probe_media_reject(&toks, layout ^ 1, Some(0xB)), None);
    }

    /// 期3 G3 P1.1: the cold index scans pack files, proves by full token
    /// equality (never fingerprint alone), and prunes by byte cap.
    #[test]
    fn cold_pack_index_scans_proves_and_prunes() {
        let st = storage();
        let src = RawStore::new(raw_config(), Some(&st)).expect("src");
        fill(&src, 4);
        let dir = std::env::temp_dir().join(format!("kvmem-packidx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let layout = layout_fingerprint(&raw_config());
        let mut write = |frontier: u32, seed: u32, name: &str| {
            let canonical = src.fork_prefix(frontier).expect("fork");
            let toks: Vec<u32> = (0..frontier)
                .map(|i| i.wrapping_mul(31).wrapping_add(seed))
                .collect();
            let meta = ContentPackMeta {
                version: ContentPackMeta::VERSION,
                layout_id: layout,
                frontier,
                tokens: toks.clone(),
                mean_tail: None,
                group: 7,
                original_query: Some((0, 10)),
                n_record_layers: LAYERS,
                media_digest: None,
                media_spans: Vec::new(),
            };
            let path = dir.join(name);
            canonical
                .write_prefix_pack(frontier / BLOCK, &path, &meta.encode().expect("meta"))
                .expect("write pack");
            (path, toks)
        };
        let (p1, t1) = write(128, 0, "a.hkpack");
        let (p2, t2) = write(256, 1, "b.hkpack");
        let mut idx = ColdPackIndex::scan(&dir, layout, LAYERS, BLOCK, 1 << 20);
        assert_eq!(idx.len(), 2, "two well-formed packs");
        assert_eq!(idx.lookup(&t1, layout).map(|i| idx.packs()[i].path.clone()), Some(p1.clone()));
        assert_eq!(idx.lookup(&t2, layout).map(|i| idx.packs()[i].path.clone()), Some(p2.clone()));
        let mut foreign = t1.clone();
        foreign[0] ^= 0xffff;
        assert!(idx.lookup(&foreign, layout).is_none(), "fingerprint alone is no proof");
        assert!(idx.lookup(&t1, layout ^ 1).is_none(), "layout must match");
        let removed = idx.prune_to(0);
        assert_eq!(removed.len(), 2);
        assert!(!p1.exists() && !p2.exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}

