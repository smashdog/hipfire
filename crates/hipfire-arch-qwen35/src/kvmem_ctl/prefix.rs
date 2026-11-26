// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;

impl KvMemCtl {
    // ── P4.3 KVMem-native prefix sharing (host model) ─────────────────────
    //
    // ninfer-aligned: freeze an IMMUTABLE canonical archive at a page-aligned
    // frontier, then reuse it across sessions via provenance-validated forks.
    // The GPU page PIN (seal + cache refs + share_published_pages) and the
    // DeltaNet checkpoint restore are engine-side (serve admit); these methods
    // own the host store half. Every path is additive: with PREFIX_PIN off the
    // engine never calls them and the catalog stays empty (byte-neutral).

    /// Freeze slot `slot`'s prefix `[0, frontier)` into an immutable canonical
    /// archive and catalog it. `frontier` must be page-aligned and `<= fed`;
    /// `prefix_tokens` is the FULL token proof (must equal the fed prefix).
    /// Returns `(checkpoint id, displaced entries, deduped)` — the engine must
    /// release the displaced entries' GPU cache refs, and on `deduped` it must
    /// NOT take new refs (the existing entry already pins the same pages).
    /// Fail-closed before any mutation of the slot (the fork seals the source's
    /// packed records but never changes their bytes; the source keeps appending
    /// to fresh blocks).
    pub fn freeze_prefix(
        &mut self,
        slot: usize,
        frontier: u32,
        prefix_tokens: &[u32],
    ) -> Result<(u64, Vec<PrefixCheckpoint>, bool), String> {
        self.freeze_prefix_with_media(slot, frontier, prefix_tokens, None, Vec::new())
    }

    /// G7: freeze with the image-turn media identity (digest + image token
    /// spans) recorded in the checkpoint and the persistent pack meta. The
    /// engine passes the session's sticky digest on image turns; text-only
    /// freezes pass `(None, vec![])`.
    pub fn freeze_prefix_with_media(
        &mut self,
        slot: usize,
        frontier: u32,
        prefix_tokens: &[u32],
        media_digest: Option<u64>,
        media_spans: Vec<(u32, u32)>,
    ) -> Result<(u64, Vec<PrefixCheckpoint>, bool), String> {
        let st = self
            .slots
            .get(slot)
            .ok_or_else(|| format!("kvmem: freeze_prefix slot {slot} out of range"))?;
        if frontier == 0 || frontier % PREFIX_PAGE_TOKENS != 0 {
            return Err(format!(
                "kvmem freeze_prefix: frontier {frontier} must be a positive multiple of \
                 {PREFIX_PAGE_TOKENS} (GPU sharing is page-granular)"
            ));
        }
        if frontier > st.fed {
            return Err(format!(
                "kvmem freeze_prefix: frontier {frontier} exceeds fed {}",
                st.fed
            ));
        }
        if prefix_tokens.len() != frontier as usize {
            return Err(format!(
                "kvmem freeze_prefix: token proof len {} != frontier {frontier}",
                prefix_tokens.len()
            ));
        }
        let layout_id = layout_fingerprint(&self.raw_cfg);
        let fingerprint = hipfire_runtime::checkpoint_pool::prefix_fingerprint(prefix_tokens);
        // 期2.3: the question that owns this checkpoint (import identity).
        let original_query = st.request_query;
        // Dedup BEFORE forking/sealing: an identical frontier+proof already
        // pins the same pages; a second archive would double the cache refs.
        // The catalog stays token-keyed; the media digest gates serving and
        // scopes the pack path and the recurrent-state key.
        if let Some(existing) = self.prefix_catalog.find_exact(frontier, fingerprint) {
            return Ok((existing, Vec::new(), true));
        }
        // Canonical archive: packed records become sealed shared sources;
        // mean-K stays private to the archive; the branch is truncated to the
        // frontier (core fork_prefix contract). The source store keeps its
        // own mutable statistics for blocks beyond the frontier.
        let mut canonical = st
            .raw
            .fork_prefix(frontier)
            .map_err(|e| format!("kvmem freeze_prefix fork: {e}"))?;
        let blocks = frontier / KV_BLOCK_TOKENS as u32;
        let mean_tail = st
            .raw
            .mean_checkpoint(frontier)
            .map_err(|e| format!("kvmem freeze_prefix mean checkpoint: {e}"))?;
        // 期3 G4: the canonical side is also the core `KvSnapshot` over the
        // sealed packed records of the canonical's account (block-major,
        // layer-major, K then V). The snapshot is the catalog-visible immutable
        // object the G2/G3 directories share. Host-only fixtures without a
        // packed archive carry no snapshot (missing records are not an error).
        // 期3 G3 follow-up (nInfer origin inheritance): reuse the SealedKvBlock
        // objects of this slot's previous checkpoint when the record storage
        // identity matches (same session, shared sealed records).
        let previous_snapshot = st
            .last_freeze_id
            .and_then(|pid| self.prefix_catalog.get(pid))
            .and_then(|c| c.kv_snapshot.as_ref());
        let kv_snapshot = if canonical.has_packed(blocks) {
            let snapshot = canonical
                .capture_snapshot(blocks, previous_snapshot)
                .map_err(|e| format!("kvmem freeze_prefix snapshot: {e}"))?;
            if !snapshot.covers_packed(blocks, self.n_record_layers as u32) {
                return Err(format!(
                    "kvmem freeze_prefix snapshot: incomplete {blocks}×{} packed grid",
                    self.n_record_layers
                ));
            }
            Some(snapshot)
        } else {
            None
        };
        let id = self.prefix_catalog.mint_id();
        let mut entry = PrefixCheckpoint::new(
            id,
            frontier,
            prefix_tokens.to_vec(),
            mean_tail,
            st.lw_incarnation,
            layout_id,
            canonical,
        );
        entry.original_query = original_query;
        entry.group = st.convo_group;
        entry.media_digest = media_digest;
        entry.media_spans = media_spans.clone();
        if self.content_groups > 0 && self.diagnostics.trace {
            eprintln!(
                "[kvmem-content] freeze slot={slot} frontier={frontier} group={} entries={}",
                st.convo_group,
                self.prefix_catalog.len()
            );
        }
        entry.kv_snapshot = kv_snapshot;
        // 期3 G3 P1.1: persist the canonical prefix as a pack file. Failure
        // never blocks the freeze (persistence is an optional tier); the byte
        // cap prunes the oldest packs.
        if let Some(dir) = self.cold_dir.clone() {
            let meta = ContentPackMeta {
                version: ContentPackMeta::VERSION,
                layout_id,
                frontier,
                tokens: entry.prefix_tokens.clone(),
                mean_tail: entry.mean_tail.clone(),
                group: entry.group,
                original_query: entry.original_query,
                n_record_layers: self.n_record_layers as u32,
                media_digest: entry.media_digest,
                media_spans: entry.media_spans.clone(),
            };
            // G7: the file name carries the media identity. Two images with
            // the same token stream share the token fingerprint; without the
            // digest suffix their packs would overwrite each other.
            let media_tag = match media_digest {
                None => String::new(),
                Some(d) => format!("-m{d:016x}"),
            };
            let path = dir.join(format!(
                "p{layout_id:016x}-{frontier:08}-{fingerprint:016x}{media_tag}.hkpack"
            ));
            // Idempotent publish: a previous freeze of this exact token proof
            // already left a complete pack + recurrent sidecar pair. Reusing
            // it avoids rewriting ~pack + DN bytes and guarantees the pair
            // stays coherent (the sidecar is only skipped with its pack).
            let side = sidecar_path(&path);
            let complete_pair = self.cold_packs.contains_path(&path)
                && std::fs::metadata(&path)
                    .map(|m| m.len() > 0)
                    .unwrap_or(false)
                && std::fs::metadata(&side)
                    .map(|m| m.len() >= SIDECAR_MIN_BYTES)
                    .unwrap_or(false);
            let written = if complete_pair {
                if self.diagnostics.trace {
                    eprintln!("[kvmem-pack] reuse complete pack+sidecar frontier={frontier}");
                }
                Ok(())
            } else {
                // A rewrite must never pair a fresh pack with a stale sidecar.
                let _ = std::fs::remove_file(&side);
                self.cold_packs.set_sidecar_bytes(&path, 0);
                meta.encode().and_then(|bytes| {
                    entry
                        .canonical()
                        .write_prefix_pack(blocks, &path, &bytes)
                        .map_err(|e| e.to_string())
                })
            };
            match written {
                Ok(()) => {
                    entry.pack_path = Some(path.clone());
                    if !self.cold_packs.contains_path(&path) {
                        let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        self.cold_packs.insert(ColdPack {
                            path: path.clone(),
                            meta,
                            fingerprint,
                            bytes,
                            sidecar_bytes: 0,
                            draft_bytes: 0,
                            stamp: 0,
                        });
                    }
                    let removed = self.cold_packs.prune_to(self.cold_bytes);
                    if !removed.is_empty() && self.diagnostics.trace {
                        eprintln!(
                            "[kvmem-pack] pruned {} pack(s), {} bytes retained",
                            removed.len(),
                            self.cold_packs.total_bytes()
                        );
                    }
                }
                Err(e) => eprintln!("[kvmem-pack] write skipped: {e}"),
            }
        }
        // 期3 G3: group-aware publication retires the whole least-recently-used
        // group before entry trimming; the plain LRU path stays the default.
        let (inserted, displaced) = if self.content_groups > 0 && st.convo_group != 0 {
            let (id, displaced, _deduped) = self
                .prefix_catalog
                .insert_group_aware(entry, self.content_groups)
                .map_err(|e| format!("kvmem freeze_prefix catalog: {e}"))?;
            self.content_retirements_total += displaced.len() as u64;
            (id, displaced)
        } else {
            self.prefix_catalog
                .insert_displacing(entry)
                .map_err(|e| format!("kvmem freeze_prefix catalog: {e}"))?
        };
        if inserted.is_some() {
            self.prefix_freezes_total += 1;
        }
        let frozen_id = inserted.unwrap_or(id);
        if let Some(st) = self.slots.get_mut(slot) {
            st.last_freeze_id = Some(frozen_id);
        }

        let mut displaced = displaced;
        if let Some((count, bytes, interval, min_gap)) = self.checkpoint_retention {
            let pruned = self
                .prefix_catalog
                .prune_policy(count, bytes, interval, min_gap);
            self.prefix_evictions_total += pruned.len() as u64;
            displaced.extend(pruned);
        }
        self.prefix_evictions_total += 0;
        Ok((inserted.unwrap_or(id), displaced, false))
    }

    /// Look up the longest cataloged prefix fully proven by `prompt`.
    /// Returns the checkpoint id (engine then validates + forks).
    pub fn prefix_lookup(&mut self, prompt: &[u32]) -> Option<u64> {
        self.prefix_lookup_media(prompt, None)
    }

    /// G7: media-aware prefix lookup. A checkpoint covering an image span is
    /// eligible only when the incoming request's media digest matches it.
    pub fn prefix_lookup_media(&mut self, prompt: &[u32], media: Option<u64>) -> Option<u64> {
        let layout = layout_fingerprint(&self.raw_cfg);
        if let Some(hit) = self.prefix_catalog.lookup_media(prompt, layout, media) {
            self.prefix_catalog_hits_total += 1;
            self.prefix_hits_total += 1;
            return Some(hit);
        }
        // 期3 G3 P1.1: a cold pack hit loads the payload and publishes a
        // normal catalog entry; the caller then reuses it through the usual
        // proof path (restart reuse still needs the DN checkpoint, patch 35).
        if let Some(index) = self.cold_packs.lookup_media(prompt, layout, media) {
            let pack = self.cold_packs.packs()[index].clone();
            match self.load_cold_pack(&pack) {
                Ok(id) => {
                    // G7 Stage C: the hot catalog is token-keyed. A pack load
                    // can dedup into a live entry that belongs to a DIFFERENT
                    // image (identical `image_pad` token stream). Serving it
                    // would hand over the other image's KV — stay a miss.
                    let identity_ok = self
                        .prefix_catalog
                        .get(id)
                        .map(|e| e.media_digest == pack.meta.media_digest)
                        .unwrap_or(false);
                    if !identity_ok {
                        self.cold_pack_misses_total += 1;
                        eprintln!(
                            "[kvmem-pack] cold load identity mismatch \
                             (media digest) — miss"
                        );
                        return None;
                    }
                    self.cold_pack_hits_total += 1;
                    self.prefix_hits_total += 1;
                    if self.diagnostics.trace {
                        eprintln!(
                            "[kvmem-pack] cold hit frontier={} path={}",
                            pack.meta.frontier,
                            pack.path.display()
                        );
                    }
                    return Some(id);
                }
                Err(e) => {
                    self.cold_pack_misses_total += 1;
                    eprintln!("[kvmem-pack] cold load failed: {e}");
                }
            }
        }
        // G7 Stage E probe: distinguish a media-identity refusal from a token
        // proof miss. The candidate below proves by FULL token equality; only
        // the image identity gate refused it. Trace-only, read-only.
        if self.diagnostics.trace {
            let layout = layout_fingerprint(&self.raw_cfg);
            let rejected = self
                .prefix_catalog
                .probe_media_reject(prompt, layout, media)
                .or_else(|| self.cold_packs.probe_media_reject(prompt, layout, media));
            if let Some((frontier, digest)) = rejected {
                let fmt = |d: Option<u64>| {
                    d.map(|x| format!("{x:016x}"))
                        .unwrap_or_else(|| "none".to_string())
                };
                eprintln!(
                    "[kvmem-media] lookup miss: token candidate frontier={frontier} \
                     digest={} request digest={} -> media identity gate",
                    fmt(digest),
                    fmt(media)
                );
            }
        }
        self.prefix_misses_total += 1;
        None
    }

    /// Drain catalog entries displaced by a cold pack load so the engine can
    /// release their GPU cache refs.
    pub fn take_cold_displaced(&mut self) -> Vec<PrefixCheckpoint> {
        std::mem::take(&mut self.cold_displaced)
    }

    /// Load one pack file into a fresh store (content account when enabled),
    /// prove the metadata is unchanged, capture the KvSnapshot and publish the
    /// entry into the hot catalog.
    fn load_cold_pack(&mut self, pack: &ColdPack) -> Result<u64, String> {
        let layout = layout_fingerprint(&self.raw_cfg);
        if pack.meta.layout_id != layout
            || pack.meta.n_record_layers as usize != self.n_record_layers
            || pack.meta.frontier == 0
            || pack.meta.frontier % KV_BLOCK_TOKENS as u32 != 0
        {
            return Err("pack geometry/layout mismatch".to_string());
        }
        let store = RawStore::new(self.raw_cfg, Some(&self.storage))
            .map_err(|e| format!("pack store: {e}"))?;
        let meta_bytes = store
            .read_prefix_pack(&pack.path)
            .map_err(|e| format!("pack read: {e}"))?;
        let meta = ContentPackMeta::decode(&meta_bytes).map_err(|e| format!("pack meta: {e}"))?;
        if meta.tokens != pack.meta.tokens || meta.frontier != pack.meta.frontier {
            return Err("pack metadata changed since scan".to_string());
        }
        let blocks = meta.frontier / KV_BLOCK_TOKENS as u32;
        let kv_snapshot = if store.has_packed(blocks) {
            let snap = store
                .capture_snapshot(blocks, None)
                .map_err(|e| format!("pack snapshot: {e}"))?;
            if !snap.covers_packed(blocks, self.n_record_layers as u32) {
                return Err("pack snapshot grid incomplete".to_string());
            }
            Some(snap)
        } else {
            None
        };
        let id = self.prefix_catalog.mint_id();
        let mut entry = PrefixCheckpoint::new(
            id,
            meta.frontier,
            meta.tokens.clone(),
            meta.mean_tail.clone(),
            0,
            layout,
            store,
        );
        entry.original_query = meta.original_query;
        entry.group = meta.group;
        entry.media_digest = meta.media_digest;
        entry.media_spans = meta.media_spans.clone();
        entry.kv_snapshot = kv_snapshot;
        entry.pack_path = Some(pack.path.clone());
        let (inserted, displaced) = if self.content_groups > 0 && entry.group != 0 {
            let (id, displaced, _deduped) = self
                .prefix_catalog
                .insert_group_aware(entry, self.content_groups)
                .map_err(|e| format!("pack catalog: {e}"))?;
            self.content_retirements_total += displaced.len() as u64;
            (id, displaced)
        } else {
            self.prefix_catalog
                .insert_displacing(entry)
                .map_err(|e| format!("pack catalog: {e}"))?
        };
        self.cold_displaced.extend(displaced);
        Ok(inserted.unwrap_or(id))
    }

    /// Look up the longest cataloged prefix fully proven by `prompt` whose
    /// frontier is at most `max_frontier` (import narrowing fallback).
    pub fn prefix_lookup_capped(&mut self, prompt: &[u32], max_frontier: u32) -> Option<u64> {
        self.prefix_lookup_capped_media(prompt, max_frontier, None)
    }

    /// G7: media-aware capped lookup for import narrowing.
    pub fn prefix_lookup_capped_media(
        &mut self,
        prompt: &[u32],
        max_frontier: u32,
        media: Option<u64>,
    ) -> Option<u64> {
        let layout = layout_fingerprint(&self.raw_cfg);
        let hit = self
            .prefix_catalog
            .lookup_at_most_media(prompt, layout, max_frontier, media);
        if hit.is_some() {
            self.prefix_catalog_hits_total += 1;
            self.prefix_hits_total += 1;
        } else {
            // G7 Stage E probe: a capped lookup narrowed to `max_frontier` can
            // miss on the media identity alone; name the refused candidate.
            if self.diagnostics.trace {
                if let Some((frontier, digest)) = self
                    .prefix_catalog
                    .probe_media_reject(prompt, layout, media)
                {
                    let fmt = |d: Option<u64>| {
                        d.map(|x| format!("{x:016x}"))
                            .unwrap_or_else(|| "none".to_string())
                    };
                    eprintln!(
                        "[kvmem-media] capped lookup miss: token candidate frontier={frontier} \
                         digest={} request digest={} cap={max_frontier} -> media identity gate",
                        fmt(digest),
                        fmt(media)
                    );
                }
            }
            self.prefix_misses_total += 1;
        }
        hit
    }

    /// Read a cataloged checkpoint (engine attaches GPU pages / reads the
    /// frontier + mean_tail for the DeltaNet restore pairing).
    pub fn prefix_checkpoint(&self, id: u64) -> Option<&PrefixCheckpoint> {
        self.prefix_catalog.get(id)
    }

    /// 期4: DFlash2 draft image instrumentation (captures / restore hits /
    /// misses). Pure bookkeeping — never read back into selection/residency.
    pub fn note_draft_image_capture(&mut self) {
        self.draft_images_total += 1;
    }

    pub fn note_draft_image_hit(&mut self) {
        self.draft_image_hits_total += 1;
    }

    pub fn note_draft_image_miss(&mut self) {
        self.draft_image_misses_total += 1;
    }

    /// G3 follow-up: a reuse conversion proceeded without pinned pages and
    /// restored them from the sealed archive (demoted or restart entry).
    pub fn note_prefix_restore_hit(&mut self) {
        self.prefix_restore_hits_total += 1;
    }

    /// Mutable access (engine attaches `gpu_pages` after sealing + cache refs).
    pub fn prefix_checkpoint_mut(&mut self, id: u64) -> Option<&mut PrefixCheckpoint> {
        self.prefix_catalog.get_mut(id)
    }

    /// Remove a checkpoint, returning it so the engine can release its GPU
    /// cache refs (session close / explicit release).
    pub fn release_prefix(&mut self, id: u64) -> Option<PrefixCheckpoint> {
        self.prefix_catalog.remove(id)
    }

    /// Pop the LRU checkpoint (admission pressure reclaim); the engine
    /// releases its GPU cache refs. `None` when the catalog is empty.
    pub fn pop_lru_prefix(&mut self) -> Option<PrefixCheckpoint> {
        let popped = self.prefix_catalog.pop_lru();
        if popped.is_some() {
            self.prefix_evictions_total += 1;
        }
        popped
    }

    /// G3 follow-up: admission-pressure residency demotion. Drops the LRU
    /// checkpoint's pinned pages but KEEPS its catalog entry (nInfer: the
    /// directory and GPU residency are separate), so a later reuse restores
    /// the pages from the sealed archive under Layerwise. Returns the entry id
    /// and the released page indices, or `None` when no entry pins any page.
    /// Legacy keeps [`Self::pop_lru_prefix`] (its reuse needs resident pages).
    pub fn demote_lru_prefix(&mut self) -> Option<(u64, Vec<u32>)> {
        let demoted = self.prefix_catalog.demote_lru();
        if demoted.is_some() {
            self.prefix_demotions_total += 1;
        }
        demoted
    }

    /// Record the DFlash draft image written beside a pack. The whole cache
    /// directory (packs + DN + draft sidecars) is re-pruned under the same cap.
    /// When the pack disappeared meanwhile, drop the orphan draft file.
    pub fn note_draft_sidecar(&mut self, path: &std::path::Path, bytes: u64) {
        if !self.cold_packs.set_draft_bytes(path, bytes) {
            let draft = crate::kvmem_prefix::draft_sidecar_path(path);
            let _ = std::fs::remove_file(draft);
            return;
        }
        self.draft_sidecars_total += 1;
        self.draft_sidecar_bytes = self.draft_sidecar_bytes.saturating_add(bytes);
        let removed = self.cold_packs.prune_to(self.cold_bytes);
        if !removed.is_empty() && self.diagnostics.trace {
            eprintln!(
                "[kvmem-pack] pruned {} pack(s), {} disk bytes retained",
                removed.len(),
                self.cold_packs.disk_bytes()
            );
        }
    }

    /// P2: whether KVMem owns persistent DeltaNet checkpoints in host memory.
    pub fn dn_ring_enabled(&self) -> bool {
        self.dn_ring.enabled()
    }

    /// S2: whether an idle session's DeltaNet state may be demoted to a host
    /// image (`HIPFIRE_KVMEM_DN_IDLE`, default off).
    pub fn dn_idle_enabled(&self) -> bool {
        self.dn_idle
    }

    /// P2: store one serialized `DeltaNetSnapshot`. The engine exports the
    /// live state at a page boundary; the ring evicts oldest checkpoints to
    /// stay within `HIPFIRE_KVMEM_DN_HOST_MIB`.
    pub fn note_dn_checkpoint(&mut self, frontier: u64, fp: u64, bytes: Vec<u8>) {
        let len = bytes.len();
        if self.dn_ring.insert(frontier, fp, bytes) {
            if self.diagnostics.trace {
                eprintln!(
                    "[kvmem-dn] host checkpoint frontier={frontier} fp={fp:#x} bytes={len} \
                     ring_bytes={}",
                    self.dn_ring.total_bytes()
                );
            }
        } else if self.diagnostics.trace {
            eprintln!(
                "[kvmem-dn] host checkpoint refused frontier={frontier} bytes={len} \
                 cap={}",
                self.dn_ring.total_bytes()
            );
        }
    }

    /// P2: exact host checkpoint lookup (frontier + token/media fingerprint).
    pub fn dn_checkpoint(&self, frontier: u64, fp: u64) -> Option<&[u8]> {
        self.dn_ring.get(frontier, fp)
    }

    pub fn dn_ring_len(&self) -> u32 {
        self.dn_ring.len() as u32
    }

    pub fn dn_ring_bytes(&self) -> u64 {
        self.dn_ring.total_bytes()
    }

    /// S2: a live slot's recurrent bytes moved into a parked native image.
    /// The legacy catalog already owns those bytes; this ledger tracks the
    /// state-image share so `/metrics` reports it without double-counting
    /// host storage.
    pub fn note_state_demotion(&mut self, bytes: u64) {
        self.state_demotions_total = self.state_demotions_total.saturating_add(1);
        self.state_host_bytes = self.state_host_bytes.saturating_add(bytes);
    }

    /// S2: a parked image was promoted back into a live allocation.
    pub fn note_state_promotion(&mut self, bytes: u64) {
        self.state_promotions_total = self.state_promotions_total.saturating_add(1);
        self.state_host_bytes = self.state_host_bytes.saturating_sub(bytes);
    }

    /// S2: a parked image was successfully installed into a live state.
    pub fn note_state_restore(&mut self) {
        self.state_restores_total = self.state_restores_total.saturating_add(1);
    }

    /// Live observability snapshot (`/metrics` via `EngineStats.kvmem`).
    /// Cheap integer reads from the host stores: no device work, no
    /// allocation, and no path reads it back into selection/residency, so it
    /// cannot change KV bytes.
    pub fn stats(&self) -> KvMemStats {
        let mut payload_bytes = 0u64;
        let mut blocks = 0u64;
        let mut tokens = 0u64;
        for st in &self.slots {
            payload_bytes += st.raw.payload_bytes();
            blocks += u64::from(st.kv.block_count());
            tokens += u64::from(st.kv.total_tokens());
        }
        let page_bytes = (self.k_page_bytes + self.v_page_bytes) as u64;
        let prefix_host = self.prefix_catalog.host_accounting();
        KvMemStats {
            enabled: true,
            layerwise: self.strategy.is_layerwise(),
            trunk_managed: true,
            mtp_head_managed: self.head_plane_enabled() || self.head_live,
            mtp_head_history_managed: self.head_composite_enabled(),
            // S2: the working state is still engine scratch while a slot is
            // active, but parking demotes it into a native history image and a
            // cold admission reallocates/restores it. Together with the host
            // ring, that makes the recurrent state a KVMem-owned tier.
            dn_state_managed: true,
            dflash_managed: false,
            // The engine fills live and parked bytes in `note_stats`.
            dn_state_bytes: 0,
            dn_state_slots: 0,
            dn_checkpoints: self.dn_ring.len() as u32,
            dn_checkpoint_bytes: self.dn_ring.total_bytes(),
            // S0/S2: the engine fills the parked-image count in `note_stats`;
            // the lifecycle counters below are controller-owned.
            state_images: 0,
            state_host_bytes: self.state_host_bytes,
            state_demotions: self.state_demotions_total,
            state_promotions: self.state_promotions_total,
            state_restores: self.state_restores_total,
            media_pins: true,
            host_charged_bytes: self.storage.charged_bytes(),
            disk_bytes: self.storage.disk_bytes(),
            payload_bytes,
            blocks,
            tokens,
            spill_pages: self.spill_pages_total,
            restore_pages: self.restore_pages_total,
            spill_bytes: self.spill_pages_total.saturating_mul(page_bytes),
            restore_bytes: self.restore_pages_total.saturating_mul(page_bytes),
            prefix_entries: self.prefix_catalog.len() as u64,
            prefix_pinned_pages: self.prefix_catalog.pinned_pages(),
            prefix_host_bytes: prefix_host.unique,
            prefix_host_bytes_gross: prefix_host.gross,
            prefix_freezes: self.prefix_freezes_total,
            prefix_hits: self.prefix_hits_total,
            prefix_misses: self.prefix_misses_total,
            prefix_evictions: self.prefix_evictions_total,
            prefix_demotions: self.prefix_demotions_total,
            prefix_catalog_hits: self.prefix_catalog_hits_total,
            prefix_restore_hits: self.prefix_restore_hits_total,
            content_groups: self.prefix_catalog.group_count() as u64,
            content_retirements: self.content_retirements_total,
            // Single tier: packs are files, so there is no second account.
            content_cache_bytes: 0,
            content_cache_disk_bytes: 0,
            content_packs: self.cold_packs.len() as u64,
            content_pack_bytes: self.cold_packs.total_bytes(),
            cold_pack_hits: self.cold_pack_hits_total,
            cold_pack_misses: self.cold_pack_misses_total,
            draft_images: self.draft_images_total,
            draft_image_hits: self.draft_image_hits_total,
            draft_image_misses: self.draft_image_misses_total,
            draft_sidecars: self.draft_sidecars_total,
            draft_sidecar_bytes: self.draft_sidecar_bytes,
        }
    }

    /// 期2.3: the query span stored with a cataloged checkpoint (import
    /// identity). `None` for checkpoints frozen before a proven span existed.
    pub fn prefix_original_query(&self, id: u64) -> Option<(u64, u64)> {
        self.prefix_catalog.get(id).and_then(|c| c.original_query)
    }

    /// 期3 G3 P1.1: the pack file backing a cataloged checkpoint (the DN
    /// sidecar lives beside it as `<pack>.dn`). Returns `None` once the pack
    /// was pruned from the disk index, so callers never create an orphan
    /// sidecar for a vanished pack.
    pub fn prefix_pack_path(&self, id: u64) -> Option<std::path::PathBuf> {
        let path = self
            .prefix_catalog
            .get(id)
            .and_then(|c| c.pack_path.clone())?;
        self.cold_packs.contains_path(&path).then_some(path)
    }

    /// A complete reusable pack + recurrent sidecar pair already exists for
    /// `id` (post-freeze). The engine skips the DN export when true.
    pub fn prefix_sidecar_ready(&self, id: u64) -> bool {
        let Some(pack) = self.prefix_pack_path(id) else {
            return false;
        };
        std::fs::metadata(sidecar_path(&pack))
            .map(|m| m.len() >= SIDECAR_MIN_BYTES)
            .unwrap_or(false)
    }

    /// Record the recurrent sidecar written beside a pack. The whole cache
    /// directory (packs + sidecars) is re-pruned under `HIPFIRE_KVMEM_CACHE_BYTES`.
    /// When the pack disappeared meanwhile, drop the orphan sidecar instead.
    pub fn note_pack_sidecar(&mut self, path: &std::path::Path, bytes: u64) {
        if !self.cold_packs.set_sidecar_bytes(path, bytes) {
            let _ = std::fs::remove_file(sidecar_path(path));
            return;
        }
        let removed = self.cold_packs.prune_to(self.cold_bytes);
        if !removed.is_empty() && self.diagnostics.trace {
            eprintln!(
                "[kvmem-pack] pruned {} pack(s), {} disk bytes retained",
                removed.len(),
                self.cold_packs.disk_bytes()
            );
        }
    }
}
