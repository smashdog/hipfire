// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::transfer::{copy_legacy_page_to_device, copy_legacy_page_to_host};
use super::*;

/// Query-sum/score diagnostics are per-query and high-volume; keep them opt-in
/// (`HIPFIRE_KVMEM_QUERY_DIAG=1`) so normal serve logs stay selection-focused.
fn query_diag_enabled() -> bool {
    std::env::var_os("HIPFIRE_KVMEM_QUERY_DIAG")
        .is_some_and(|value| matches!(value.to_string_lossy().trim(), "1" | "on" | "true" | "yes"))
}

impl KvMemCtl {
    pub fn legacy_config(&self) -> Option<&hipfire_config::kvmem::ResolvedConfig> {
        self.legacy.as_ref()
    }

    pub fn request_query(&self, slot: usize) -> Option<(u64, u64)> {
        self.slots.get(slot).and_then(|state| state.request_query)
    }

    pub fn legacy_query_for_request(
        &self,
        slot: usize,
        source: Option<(u64, u64)>,
        prompt_end: usize,
    ) -> Result<Option<crate::legacy_query::LegacyQuery>, String> {
        let Some(config) = &self.legacy else {
            return Ok(None);
        };
        let frontier = self
            .slots
            .get(slot)
            .ok_or("kvmem: Query slot out of range")?
            .fed;
        let source = source
            .map(|(begin, end)| {
                end.checked_sub(begin)
                    .map(|count| (begin, count))
                    .ok_or_else(|| "kvmem: reversed frontend Query range".to_string())
            })
            .transpose()?;
        let query = crate::legacy_query::LegacyQuery::new(
            source,
            frontier as usize,
            prompt_end,
            config.query_max_tokens,
            config.query_replay_max_tokens,
            KV_BLOCK_TOKENS,
            self.n_fa_layers,
            self.q_width,
        )?;
        // A Query that would restore an empty store has no accepted history to
        // score or rewind to: the public API rejects a zero-page snapshot, and
        // there is nothing behind the query to recall. Leave the request on the
        // ordinary first-pass/prefill path instead of half-starting a probe.
        if query
            .as_ref()
            .is_some_and(|query| query.restore_begin() == 0)
        {
            return Ok(None);
        }
        Ok(query)
    }

    /// Score the immutable pre-Query history using every proven user Q row.
    /// The temporary selector previews LRFU but never commits a logical round;
    /// only the accepted end of first pass/replay may publish that history.
    pub fn select_legacy_query(
        &self,
        source: &LegacyStoreSnapshot,
        query: &crate::legacy_query::LegacyQuery,
    ) -> Result<(Vec<u32>, Vec<f64>), String> {
        let config = self
            .legacy
            .as_ref()
            .ok_or("kvmem: missing Legacy Query policy")?;
        if source.frontier as usize != query.restore_begin()
            || source.layout_id != layout_fingerprint(&self.raw_cfg)
        {
            return Err("kvmem: Query source does not match the captured restore frontier".into());
        }
        let blocks = (source.frontier as usize).div_ceil(KV_BLOCK_TOKENS) as u32;
        if blocks == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let aggregation = match config.score_agg {
            hipfire_config::kvmem::ScoreAgg::Mean => GqaAggregation::Mean,
            hipfire_config::kvmem::ScoreAgg::Max => GqaAggregation::Max,
        };
        let scores = match config.query_scorer {
            hipfire_config::kvmem::QueryScorer::MeanQCosine => {
                let sums = query.query_sums()?;
                let rows: u64 = query.query_row_counts().iter().map(|&n| u64::from(n)).sum();
                let norm = sums
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(*v))
                    .sum::<f64>()
                    .sqrt();
                if query_diag_enabled() {
                    eprintln!(
                        "[kvmem-legacy] QuerySums rows={rows} widths={} norm={norm:.4}",
                        sums.len()
                    );
                }
                mean_key_retrieval_scores(
                    &source.raw,
                    blocks,
                    &sums,
                    query.query_row_counts(),
                    self.n_fa_layers as u32,
                    self.head_dim as u32,
                    self.n_heads as u32,
                    self.n_kv_heads as u32,
                    aggregation,
                )
            }
            hipfire_config::kvmem::QueryScorer::PageSoftmax => page_softmax_scores(
                &source.raw,
                blocks,
                source.frontier as u64,
                KV_BLOCK_TOKENS as u32,
                self.budget_blocks as u32,
                self.kv_cfg.sink_blocks,
                self.kv_cfg.recent_blocks,
                query.query_rows(),
                query.query_row_counts(),
                self.n_fa_layers as u32,
                query.query_len() as u32,
                self.n_heads as u32,
                self.n_kv_heads as u32,
                self.head_dim as u32,
                0.0,
                aggregation,
            ),
        }
        .map_err(|e| format!("kvmem proven Query scoring: {e}"))?;
        if query_diag_enabled() {
            let mut ranked: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
            let top: Vec<String> = ranked
                .iter()
                .take(6)
                .map(|(id, s)| format!("{id}={s:.4}"))
                .collect();
            let low: Vec<String> = ranked
                .iter()
                .rev()
                .take(4)
                .map(|(id, s)| format!("{id}={s:.4}"))
                .collect();
            let counts: Vec<String> = [28u32, 32, 37]
                .into_iter()
                .filter(|&id| id < blocks)
                .map(|id| format!("{id}={}", source.raw.n_tokens(id)))
                .collect();
            eprintln!(
                "[kvmem-legacy] QueryScores top=[{}] low=[{}] raw_tokens=[{}]",
                top.join(","),
                low.join(","),
                counts.join(",")
            );
        }
        let required = hipfire_kvmem::query_prefix_required(
            source.frontier as u64,
            KV_BLOCK_TOKENS as u32,
            self.kv_cfg.sink_blocks,
            self.kv_cfg.recent_blocks,
            &[],
        )
        .map_err(|e| format!("kvmem Query mandatory: {e}"))?;
        if required.len() > self.budget_blocks {
            return Err("kvmem: Query protected prefix and partial tail exceed B".into());
        }
        let selector =
            KvStore::new(&self.kv_cfg).map_err(|e| format!("kvmem Query selector: {e}"))?;
        selector
            .register_append(source.frontier)
            .map_err(|e| e.to_string())?;
        selector
            .restore_retrieval(&source.retrieval)
            .map_err(|e| e.to_string())?;
        selector
            .begin_retrieval_round()
            .map_err(|e| e.to_string())?;
        // Keep a boundary-spanning answer visible: page scoring alone can cut
        // between the page that holds the evidence and the page that holds its
        // tail. Bleed each page's score to its immediate neighbours at 0.9 for
        // the selector only, so the scoring page keeps priority and a boosted
        // neighbour cannot tie-displace it; `scores` is still the original
        // signal for QueryCommit feedback.
        let mut dilated = scores.clone();
        for i in 1..scores.len().saturating_sub(1) {
            dilated[i] = scores[i].max(0.9 * scores[i - 1].max(scores[i + 1]));
        }
        selector
            .set_retrieval_scores(&dilated)
            .map_err(|e| e.to_string())?;
        let selected = selector
            .pick_topk_blocks(&required)
            .map_err(|e| e.to_string())?;
        if selected.len() > self.budget_blocks || selected.iter().any(|&id| id >= blocks) {
            return Err("kvmem: Query selector returned an invalid history budget".into());
        }
        eprintln!(
            "[kvmem-legacy] Query span={}..{} rows={} restore_begin={} history_pages={blocks} selected={selected:?} scorer={:?} agg={:?}",
            query.query().begin, query.query().end, query.query_len(), source.frontier,
            config.query_scorer, config.score_agg,
        );
        Ok((selected, scores))
    }

    pub fn commit_legacy_query(&mut self, slot: usize, scores: &[f64]) -> Result<(), String> {
        if self.legacy.is_none() {
            return Err("kvmem: Query commit without Legacy policy".into());
        }
        let st = self
            .slots
            .get_mut(slot)
            .ok_or("kvmem: Query commit slot out of range")?;
        if st.first_decode_done {
            return Err("kvmem: Query statistical round already published for this request".into());
        }
        st.kv.begin_retrieval_round().map_err(|e| e.to_string())?;
        st.kv
            .set_retrieval_scores(scores)
            .map_err(|e| e.to_string())?;
        let reinforced = st.kv.commit_retrieval_round().map_err(|e| e.to_string())?;
        st.first_decode_done = true;
        eprintln!(
            "[kvmem-legacy] QueryCommit slot={slot} frontier={} lrfu_committed={reinforced}",
            st.fed
        );
        Ok(())
    }

    /// Ordinary prefill/decode pressure is NOT a Query. Only accepted resident
    /// pages are candidates, exactly as NInfer's non-Query prepare_window arm.
    /// No old Q tap, full-history retrieval, or LRFU round is used here.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_legacy_step(
        &mut self,
        gpu: &mut Gpu,
        pool: &mut SlotPool,
        active: &[bool],
        resident: &[bool],
        incoming_rows: &[usize],
        decoding: &[bool],
        replaying: &[bool],
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
    ) -> Result<(), String> {
        if self.legacy.is_none() || self.strategy.is_layerwise() {
            return Err("kvmem: public Legacy pressure requires its admitted contract".into());
        }
        let window_pages = self.budget_blocks + self.gen_reserve as usize / KV_BLOCK_TOKENS;
        for slot in 0..self.slots.len() {
            let live_owner = active.get(slot).copied().unwrap_or(false)
                || resident.get(slot).copied().unwrap_or(false);
            if !live_owner && self.slots[slot].fed != 0 {
                self.reset_slot(slot)?;
            }
            if !active.get(slot).copied().unwrap_or(false) {
                continue;
            }
            let incoming = incoming_rows.get(slot).copied().unwrap_or(0);
            if incoming == 0 {
                continue;
            }
            if incoming as u64 > self.gen_reserve {
                return Err(format!(
                    "kvmem: append {incoming} exceeds Legacy R={}",
                    self.gen_reserve
                ));
            }
            let bt = pool
                .block_table(SlotId(slot))
                .ok_or("kvmem: missing Legacy page table")?;
            let frontier = bt.live_tokens();
            if self.slots[slot].fed as usize != frontier {
                return Err(format!(
                    "kvmem: pressure frontier {frontier} differs from accepted statistics {}",
                    self.slots[slot].fed
                ));
            }
            let end = frontier
                .checked_add(incoming)
                .ok_or("kvmem: append position overflow")?;
            let growing = end.div_ceil(KV_BLOCK_TOKENS) - frontier.div_ceil(KV_BLOCK_TOKENS);
            let candidates: Vec<u32> = (0..bt.num_pages())
                .filter(|&lp| bt.physical(lp).is_some())
                .map(|lp| lp as u32)
                .collect();
            if candidates.len().saturating_add(growing) <= window_pages {
                continue;
            }
            if replaying.get(slot).copied().unwrap_or(false) {
                return Err("kvmem: admitted Query replay exceeded its physical window; Pressure during replay is forbidden".into());
            }
            let required = hipfire_kvmem::query_prefix_required(
                frontier as u64,
                KV_BLOCK_TOKENS as u32,
                self.kv_cfg.sink_blocks,
                self.kv_cfg.recent_blocks,
                &[],
            )
            .map_err(|e| format!("kvmem pressure mandatory: {e}"))?;
            // Hipfire has no accepted attention-mass producer on this Legacy
            // route. The core's empty-feedback frame has zero mass and uses
            // its own deterministic recency tie-break; never recycle Query
            // similarity or rejected verify Q as fake attention feedback.
            let scores = vec![0.0; candidates.len()];
            let keep = hipfire_kvmem::pressure_selection(
                &candidates,
                &scores,
                &required,
                self.budget_blocks as u32,
            )
            .map_err(|e| format!("kvmem resident-only pressure: {e}"))?;
            let mut spill = Vec::new();
            for &lp in &candidates {
                if keep.contains(&lp) {
                    continue;
                }
                let phys = bt
                    .physical(lp as usize)
                    .ok_or("kvmem: pressure candidate vanished")?;
                if !pool.page_pool().is_some_and(|p| p.page_evictable(phys)) {
                    return Err(
                        "kvmem: Legacy pressure cannot evict an externally pinned page".into(),
                    );
                }
                if (lp as usize + 1) * KV_BLOCK_TOKENS > frontier {
                    return Err("kvmem: Legacy pressure attempted to drop a non-full tail".into());
                }
                spill.push((lp as usize, phys));
            }
            if keep.len().saturating_add(growing) > window_pages {
                return Err(
                    "kvmem: protected Legacy working set leaves insufficient append reserve".into(),
                );
            }
            let mut elapsed = std::time::Duration::ZERO;
            let mut ignored_count = 0;
            self.flush_spill_batch(
                gpu,
                pool,
                slot,
                &spill,
                k_arenas,
                v_arenas,
                self.diagnostics.time,
                &mut elapsed,
                &mut ignored_count,
            )?;
            self.spill_pages_total += spill.len() as u64;
            let bytes = spill.len() * self.n_fa_layers * (self.k_page_bytes + self.v_page_bytes);
            eprintln!(
                "[kvmem-legacy] {} slot={slot} frontier={frontier} incoming={incoming} resident_before={} kept={} window_pages={window_pages} spill_pages={} spill_bytes={bytes} restore_bytes=0",
                if decoding.get(slot).copied().unwrap_or(false) { "DecodePressure" } else { "Pressure" },
                candidates.len(), keep.len(), spill.len(),
            );
        }
        Ok(())
    }

    /// Seal a complete host recovery point without changing the live view.
    /// Legacy is write-back, so resident pages must be captured here as well
    /// as the cold pages already held by RawStore. A failure leaves the GPU
    /// view authoritative; callers may decline this optional checkpoint.
    pub fn capture_legacy_snapshot(
        &mut self,
        gpu: &Gpu,
        pool: &SlotPool,
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        slot: usize,
    ) -> Result<LegacyStoreSnapshot, String> {
        if self.strategy.is_layerwise() {
            return Err("kvmem: Legacy snapshot requested from Layerwise executor".into());
        }
        if k_arenas.len() != self.n_fa_layers || v_arenas.len() != self.n_fa_layers {
            return Err("kvmem: Legacy snapshot arena geometry mismatch".into());
        }
        let st = self
            .slots
            .get(slot)
            .ok_or("kvmem: snapshot slot out of range")?;
        let frontier = st.fed;
        let bt = pool
            .block_table(SlotId(slot))
            .ok_or("kvmem: snapshot missing page table")?;
        if bt.live_tokens() != frontier as usize {
            return Err(format!(
                "kvmem: snapshot KV frontier {} differs from statistics {frontier}",
                bt.live_tokens()
            ));
        }
        let blocks = (frontier as usize).div_ceil(KV_BLOCK_TOKENS);
        let mut resident_pages = Vec::new();
        for lp in 0..blocks {
            let rows = (frontier as usize - lp * KV_BLOCK_TOKENS).min(KV_BLOCK_TOKENS);
            if let Some(phys) = bt.physical(lp) {
                copy_legacy_page_to_host(
                    gpu,
                    &st.raw,
                    &mut self.page_k,
                    &mut self.page_v,
                    k_arenas,
                    v_arenas,
                    self.k_page_bytes,
                    self.v_page_bytes,
                    phys,
                    lp,
                    rows,
                    self.diagnostics.verify,
                )?;
                resident_pages.push(lp as u32);
            } else {
                for layer in 0..self.n_fa_layers {
                    if !st.raw.has_k_gpu(lp as u32, layer as u32, rows as u32)
                        || !st.raw.has_v_gpu(lp as u32, layer as u32, rows as u32)
                    {
                        return Err(format!(
                            "kvmem: cold snapshot page {lp} layer {layer} has incomplete native payload"
                        ));
                    }
                }
            }
        }
        let mean_tail = st
            .raw
            .mean_checkpoint(frontier)
            .map_err(|e| format!("kvmem snapshot mean tail: {e}"))?;
        let raw = st
            .raw
            .fork_prefix(frontier)
            .map_err(|e| format!("kvmem snapshot sealed fork: {e}"))?;
        if let Some(tail) = &mean_tail {
            raw.restore_mean_checkpoint(frontier, tail)
                .map_err(|e| format!("kvmem snapshot mean fork: {e}"))?;
        }
        let sealed = raw
            .capture_snapshot(blocks as u32, None)
            .map_err(|e| format!("kvmem snapshot manifest: {e}"))?;
        if !sealed.ready() || !sealed.covers_packed(blocks as u32, self.n_fa_layers as u32) {
            return Err("kvmem: Legacy snapshot is not a complete all-layer image".into());
        }
        let retrieval = st
            .kv
            .retrieval_checkpoint()
            .map_err(|e| format!("kvmem snapshot retrieval: {e}"))?;
        Ok(LegacyStoreSnapshot {
            raw,
            frontier,
            resident_pages,
            mean_tail,
            retrieval,
            sealed,
            layout_id: layout_fingerprint(&self.raw_cfg),
            request_query: st.request_query,
            last_query: st.last_query,
            group: st.convo_group,
        })
    }

    /// Restore a private branch at the snapshot's exact frontier. The caller
    /// has already admitted the selection and must restore the matching native
    /// recurrent image before forwarding. After pool.reset any error is a
    /// destructive-restore failure, never a first-pass fallback.
    pub fn restore_legacy_snapshot(
        &mut self,
        gpu: &Gpu,
        pool: &mut SlotPool,
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        slot: usize,
        source: &LegacyStoreSnapshot,
        selected: &[u32],
    ) -> Result<u64, String> {
        if self.strategy.is_layerwise()
            || slot >= self.slots.len()
            || source.layout_id != layout_fingerprint(&self.raw_cfg)
            || k_arenas.len() != self.n_fa_layers
            || v_arenas.len() != self.n_fa_layers
        {
            return Err("kvmem: Legacy restore identity/geometry mismatch".into());
        }
        let blocks = (source.frontier as usize).div_ceil(KV_BLOCK_TOKENS);
        let mut keep = selected.to_vec();
        keep.sort_unstable();
        keep.dedup();
        if keep.len() != selected.len() || keep.iter().any(|&p| p as usize >= blocks) {
            return Err("kvmem: Legacy restore contains duplicate or out-of-range pages".into());
        }
        if source.frontier as usize % KV_BLOCK_TOKENS != 0 && !keep.contains(&((blocks - 1) as u32))
        {
            return Err("kvmem: Legacy restore must retain the non-full tail page".into());
        }
        if !source.sealed.ready()
            || !source
                .sealed
                .covers_packed(blocks as u32, self.n_fa_layers as u32)
        {
            return Err("kvmem: Legacy restore source is incomplete".into());
        }
        let releasable = pool.block_table(SlotId(slot)).map_or(0, |bt| {
            (0..bt.num_pages())
                .filter(|&lp| {
                    bt.physical(lp).is_some_and(|phys| {
                        pool.page_pool().is_some_and(|p| p.page_evictable(phys))
                    })
                })
                .count()
        });
        if keep.len() > pool.free_pages().saturating_add(releasable) {
            return Err("kvmem: Legacy restore has insufficient physical page capacity".into());
        }
        // Allocate/fork every host owner before invalidating the first-pass view.
        let raw = source
            .raw
            .fork_prefix(source.frontier)
            .map_err(|e| format!("kvmem restore branch: {e}"))?;
        if let Some(tail) = &source.mean_tail {
            raw.restore_mean_checkpoint(source.frontier, tail)
                .map_err(|e| format!("kvmem restore mean tail: {e}"))?;
        }
        let kv = KvStore::new(&self.kv_cfg).map_err(|e| format!("kvmem restore selector: {e}"))?;
        kv.register_append(source.frontier)
            .map_err(|e| format!("kvmem restore selector frontier: {e}"))?;
        kv.restore_retrieval(&source.retrieval)
            .map_err(|e| format!("kvmem restore retrieval: {e}"))?;
        pool.reset(SlotId(slot));
        pool.install_spilled_prefix(SlotId(slot), source.frontier as usize)?;
        let mut restored_bytes = 0u64;
        for &lp in &keep {
            let rows =
                (source.frontier as usize - lp as usize * KV_BLOCK_TOKENS).min(KV_BLOCK_TOKENS);
            let phys = pool
                .page_pool_mut()
                .ok_or("kvmem restore requires paged pool")?
                .alloc_page_unbound()
                .map_err(|e| format!("kvmem restore alloc: {e}"))?;
            pool.bind_restored_page(SlotId(slot), lp as usize, phys)?;
            copy_legacy_page_to_device(
                gpu,
                &raw,
                &mut self.page_k,
                &mut self.page_v,
                k_arenas,
                v_arenas,
                self.k_page_bytes,
                self.v_page_bytes,
                phys,
                lp as usize,
                rows,
                self.diagnostics.verify,
            )?;
            restored_bytes += (rows * self.n_fa_layers * (self.k_page_bytes + self.v_page_bytes)
                / KV_BLOCK_TOKENS) as u64;
        }
        let incarnation = self.slots[slot]
            .lw_incarnation
            .checked_add(1)
            .ok_or("kvmem: restore incarnation overflow")?;
        let mut state =
            self.fresh_slot_state(raw, kv, source.frontier, source.frontier, incarnation);
        state.request_query = source.request_query;
        state.last_query = source.last_query;
        state.convo_group = source.group;
        self.slots[slot] = state;
        self.restore_pages_total += keep.len() as u64;
        eprintln!(
            "[kvmem-legacy] restore slot={slot} frontier={} pages={keep:?} bytes={restored_bytes}",
            source.frontier
        );
        Ok(restored_bytes)
    }

    /// Copy the still-resident pages of `[from, frontier)` from the arenas into
    /// the slot's host archive before a canonical prefix freeze. Legacy only
    /// writes a page to the host store when Pressure spills it, so a freeze
    /// taken while the prefix is resident would see an incomplete packed grid
    /// and refuse to emit a persistent pack. The copy is additive (the GPU
    /// page stays live); a later spill overwrites the same host block with the
    /// same bytes. Layerwise already write-throughs every fed row.
    pub fn archive_resident_prefix(
        &mut self,
        gpu: &Gpu,
        pool: &SlotPool,
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        slot: usize,
        frontier: u32,
        from: u32,
    ) -> Result<(), String> {
        if self.strategy.is_layerwise() {
            return Ok(());
        }
        let fed = self
            .slots
            .get(slot)
            .ok_or_else(|| format!("kvmem: archive prefix slot {slot} out of range"))?
            .fed;
        if frontier == 0
            || frontier % KV_BLOCK_TOKENS as u32 != 0
            || frontier > fed
            || from > frontier
            || from % KV_BLOCK_TOKENS as u32 != 0
        {
            return Err(format!(
                "kvmem: archive prefix range [{from},{frontier}) invalid for fed {fed}"
            ));
        }
        let bt = pool
            .block_table(SlotId(slot))
            .ok_or_else(|| "kvmem: archive prefix slot has no block table".to_string())?;
        if bt.live_tokens() < frontier as usize {
            return Err(format!(
                "kvmem: archive prefix block table holds {} tokens, need {frontier}",
                bt.live_tokens()
            ));
        }
        let first = (from / KV_BLOCK_TOKENS as u32) as usize;
        let end_page = (frontier / KV_BLOCK_TOKENS as u32) as usize;
        let mut pages = Vec::with_capacity(end_page.saturating_sub(first));
        for lp in first..end_page {
            let Some(phys) = bt.physical(lp) else {
                // Already spilled: the host archive owns the packed rows.
                continue;
            };
            let rows = (frontier as usize - lp * KV_BLOCK_TOKENS).min(KV_BLOCK_TOKENS);
            pages.push((lp, phys, rows));
        }
        if !pages.is_empty() {
            let t = std::time::Instant::now();
            self.archive_legacy_pages_to_host(gpu, slot, &pages, k_arenas, v_arenas)?;
            eprintln!(
                "[kvmem-legacy] archived resident prefix slot={slot} from={from} \
                 frontier={frontier} pages={} archive_ms={:.1}",
                pages.len(),
                t.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    /// Materialize a page-less imported prefix into a Legacy-sized resident
    /// working set. A restart cold pack has no pinned GPU pages; without a
    /// restore the suffix prefill would attend over an entirely spilled block
    /// table and lose the imported context, and the engine has no Legacy cold
    /// restore arm. Restore the mandatory sink/recent pages plus the
    /// recency-ranked remainder up to the resident budget. The unselected
    /// pages stay spilled (Legacy attention masks them) and remain available
    /// to Query/flush restore; unlike the Layerwise path these pages are NOT
    /// pinned, because Legacy Pressure must still be able to evict them.
    ///
    /// Returns the number of restored pages. The caller leaves the checkpoint
    /// page list empty so a later reuse re-materializes from the sealed pack.
    pub fn stage_legacy_page_less_import(
        &mut self,
        gpu: &Gpu,
        pool: &mut SlotPool,
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        slot: usize,
        frontier: u32,
    ) -> Result<usize, String> {
        if self.strategy.is_layerwise() {
            return Err("kvmem: Legacy cold import requested from Layerwise executor".into());
        }
        if self.legacy.is_none() {
            return Err("kvmem: Legacy cold import without the admitted Legacy contract".into());
        }
        if slot >= self.slots.len() {
            return Err("kvmem: Legacy cold import slot out of range".into());
        }
        if frontier == 0 || frontier % KV_BLOCK_TOKENS as u32 != 0 {
            return Err(format!(
                "kvmem: Legacy cold import frontier {frontier} is not page aligned"
            ));
        }
        let blocks = (frontier / KV_BLOCK_TOKENS as u32) as usize;
        let candidates: Vec<u32> = (0..blocks as u32).collect();
        let required = hipfire_kvmem::query_prefix_required(
            frontier as u64,
            KV_BLOCK_TOKENS as u32,
            self.kv_cfg.sink_blocks,
            self.kv_cfg.recent_blocks,
            &[],
        )
        .map_err(|e| format!("kvmem Legacy cold import mandatory: {e}"))?;
        let selected = if blocks <= self.budget_blocks as usize {
            // The whole prefix fits the resident window. Restoring it all
            // gives the suffix prefill the same view a just-finished first
            // pass would have.
            candidates
        } else {
            // Legacy keeps its resident window by recency (pressure scores are
            // zero on this route). Restore the same page set the first
            // Pressure step would have kept.
            let scores = vec![0.0f64; blocks];
            hipfire_kvmem::pressure_selection(
                &candidates,
                &scores,
                &required,
                self.budget_blocks as u32,
            )
            .map_err(|e| format!("kvmem Legacy cold import selection: {e}"))?
        };
        if selected.is_empty() {
            return Ok(0);
        }

        // Start from an all-spilled logical table, then bind only the selected
        // working set. The suffix prefill appends pages beyond `frontier`
        // without allocating the masked middle.
        pool.reset(SlotId(slot));
        pool.install_spilled_prefix(SlotId(slot), frontier as usize)
            .map_err(|e| format!("kvmem Legacy cold import table: {e}"))?;
        let mut batch: Vec<(usize, usize, u32)> = Vec::with_capacity(selected.len());
        for &lp in &selected {
            let phys = pool
                .page_pool_mut()
                .ok_or_else(|| "kvmem: Legacy cold import requires a paged pool".to_string())?
                .alloc_page_unbound()
                .map_err(|e| format!("kvmem Legacy cold import alloc: {e}"))?;
            pool.bind_restored_page(SlotId(slot), lp as usize, phys)
                .map_err(|e| format!("kvmem Legacy cold import bind: {e}"))?;
            batch.push((slot, lp as usize, phys));
        }
        let mut t_restore = std::time::Duration::ZERO;
        let mut n_restored = 0usize;
        if let Err(e) = self.flush_restore_batch(
            gpu,
            &batch,
            k_arenas,
            v_arenas,
            self.diagnostics.time,
            &mut t_restore,
            &mut n_restored,
        ) {
            // The caller resets the host slot; release the partial GPU view
            // too so a retry/reuse cannot observe half-restored pages.
            pool.reset(SlotId(slot));
            return Err(e);
        }
        self.restore_pages_total += batch.len() as u64;
        eprintln!(
            "[kvmem-legacy] cold prefix import slot={slot} frontier={frontier} \
             blocks={blocks} resident={} budget={} spilled={} restore_ms={:.1}",
            batch.len(),
            self.budget_blocks,
            blocks.saturating_sub(batch.len()),
            t_restore.as_secs_f64() * 1e3,
        );
        Ok(batch.len())
    }
}
