// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;

impl KvMemCtl {
    /// Arm the one-shot Layerwise selection budget from the import admission
    /// (ninfer `selection_budget`). The next Layerwise carrier selection takes
    /// it; later selections return to the configured residential budget.
    pub fn set_layerwise_import_blocks(&self, slot: usize, blocks: u32) -> Result<(), String> {
        let st = self
            .slots
            .get(slot)
            .ok_or_else(|| format!("kvmem: import budget slot {slot} out of range"))?;
        st.lw_import_blocks.set(Some(blocks.max(1)));
        Ok(())
    }

    /// Consume the one-shot import selection budget (first Layerwise selection
    /// after a fenced import). `None` for every ordinary selection.
    pub fn take_layerwise_import_blocks(&self, slot: usize) -> Option<u32> {
        self.slots
            .get(slot)
            .and_then(|st| st.lw_import_blocks.take())
    }

    /// P2.2a fenced-import admission (ninfer `prepare_import_start`): build the
    /// pure window/import policy from the restored exact prefix. Pure host
    /// arithmetic; it touches no catalog, device or selection state.
    pub fn import_window(
        &self,
        prompt_end: u64,
        output_tokens: u64,
        frontier: u32,
        query: Option<(u64, u64)>,
        previous_query: Option<(u64, u64)>,
    ) -> Result<ImportDecision, String> {
        // ninfer `state.visible_tokens` is the native extent at request start;
        // hipfire's Layerwise driver materializes a compact view of at most the
        // resident budget B, while the logical history may be longer.
        let visible = (frontier as u64).min((self.budget_blocks as u64) * KV_BLOCK_TOKENS as u64);
        layerwise_import_decision(
            prompt_end,
            output_tokens,
            self.window_capacity_tokens(),
            self.gen_reserve,
            frontier,
            visible,
            query,
            previous_query,
            true,
        )
    }

    /// Drop the slot's host state. Called when a fresh session takes the slot
    /// and lazily when a closed slot is swept at step start.
    pub fn reset_slot(&mut self, slot: usize) -> Result<(), String> {
        if slot >= self.slots.len() {
            return Err(format!("kvmem: reset_slot slot {slot} out of range"));
        }
        let raw = RawStore::new(self.raw_cfg, Some(&self.storage))
            .map_err(|e| format!("kvmem raw store: {e}"))?;
        let kv = KvStore::new(&self.kv_cfg).map_err(|e| format!("kvmem store: {e}"))?;
        // A fresh store starts at the configured select_budget; re-apply the
        // watermark keep target so a reset slot keeps the same hysteresis band.
        if let Some((_, keep)) = self.watermark {
            kv.set_runtime_select_budget((keep * KV_BLOCK_TOKENS) as u32)
                .map_err(|e| format!("kvmem watermark select budget: {e}"))?;
        }
        // A fresh session bumps the layerwise incarnation so re-bootstrapped
        // canonical stamps never alias the previous session's (design §3.9g).
        let lw_incarnation = self.slots[slot].lw_incarnation + 1;
        self.slots[slot] = self.fresh_slot_state(raw, kv, 0, 0, lw_incarnation);
        Ok(())
    }

    /// Build a fresh session's slot state around the given stores. Shared by
    /// [`Self::reset_slot`] (cold session, fed 0) and [`Self::fork_prefix_into`]
    /// (P4.3 reused prefix, fed = frontier) so a reused prefix never inherits
    /// stale selection / pressure / head state — the field initialization is
    /// identical by construction.
    pub(super) fn fresh_slot_state(
        &self,
        raw: RawStore,
        kv: KvStore,
        fed: u32,
        turn_start: u32,
        lw_incarnation: u64,
    ) -> SlotState {
        SlotState {
            raw,
            kv,
            last_select_full: 0,
            fed,
            turn_start,
            first_decode_done: false,
            q_ring: vec![
                0f32;
                if self.query_span > 1 {
                    self.n_fa_layers * self.query_span * self.q_width
                } else {
                    0
                }
            ],
            q_ring_len: 0,
            q_ring_cursor: 0,
            lw_incarnation,
            lw_seq: 0,
            // Decision A (§3.10): a fresh session drops the pressure owner and
            // its keep-set. The next admitted query lazily reconstructs the
            // owner under the bumped incarnation, so accepted mass / selected
            // pages never alias the previous session. Legacy: both stay empty.
            lw_pressure: None,
            lw_output_pressure: None,
            lw_keep: Vec::new(),
            lw_query_due: false,
            request_query: None,
            last_query: None,
            convo_group: 0,
            last_freeze_id: None,
            lw_import_blocks: Cell::new(None),
            lw_pending_output: None,
            lw_pending_feed: None,
            // P4.1② G3: a fresh session drops the head resident map and its
            // per-Query binding, mirroring the trunk pressure owners. The next
            // admitted MTP verify lazily reconstructs the map under the bumped
            // incarnation. Legacy / AR-only: both stay None (byte-neutral).
            lw_head: None,
            lw_aux_selection: None,
        }
    }

    /// P4.3 fed coordination: truncate slot `slot`'s host store to a
    /// BLOCK-ALIGNED `frontier`. hipfire feeds mean-K as running sums only
    /// (`add_mean_f32` keeps no raw rows), so a mid-block partial tail cannot
    /// be reconstructed — the sum is irreversible. Block-aligned truncation
    /// drops whole blocks and keeps every retained mean exact; a mid-block
    /// reuse point refuses here and falls back to the cold rebuild (the
    /// freeze granularity is page-aligned anyway, so restorable checkpoints
    /// only ever exist at block-aligned frontiers).
    /// The caller MUST have already restored the DeltaNet recurrent state to
    /// the same frontier (the host store cannot rewind recurrent state — a
    /// truncated KV paired with a stale DN state silently corrupts the
    /// suffix). Legacy-only: Layerwise canonical views refuse truncation
    /// until H4 wires the archive provenance.
    pub fn truncate_slot(&mut self, slot: usize, frontier: u32) -> Result<(), String> {
        if self.strategy.is_layerwise() {
            return Err(
                "kvmem truncate_slot: Layerwise canonical views refuse truncation (H4)".to_string(),
            );
        }
        if frontier % KV_BLOCK_TOKENS as u32 != 0 {
            return Err(format!(
                "kvmem truncate_slot: frontier {frontier} is not block-aligned \
                 (sum-only mean-K cannot reconstruct a partial tail)"
            ));
        }
        let st = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| format!("kvmem: truncate_slot slot {slot} out of range"))?;
        if frontier > st.fed {
            return Err(format!(
                "kvmem truncate_slot: frontier {frontier} exceeds fed {}",
                st.fed
            ));
        }
        if frontier == st.fed {
            return Ok(());
        }
        st.raw
            .truncate_to(frontier)
            .map_err(|e| format!("kvmem truncate_slot raw: {e}"))?;
        st.kv
            .truncate_to(frontier)
            .map_err(|e| format!("kvmem truncate_slot kv: {e}"))?;
        st.fed = frontier;
        // The truncated store is a new selection baseline: force a fresh
        // decision and drop the previous request's Q span.
        st.last_select_full = 0;
        st.first_decode_done = false;
        st.q_ring_len = 0;
        st.q_ring_cursor = 0;
        Ok(())
    }

    /// Reuse checkpoint `id` into `dst_slot`: validate the FULL provenance
    /// (geometry + fingerprint + token proof) BEFORE any mutation, then fork
    /// the canonical archive into a fresh branch (shared sealed packed KV,
    /// branch-private mean-K), restore the partial-tail mean sum, and rebuild
    /// the slot state at `fed = frontier` (incarnation bumped, all selection /
    /// pressure / head state cleared — identical to a cold reset except the
    /// reused prefix). Returns the frontier the suffix prefill starts from.
    pub fn fork_prefix_into(
        &mut self,
        dst_slot: usize,
        id: u64,
        prompt: &[u32],
    ) -> Result<u32, String> {
        if dst_slot >= self.slots.len() {
            return Err(format!(
                "kvmem: fork_prefix_into slot {dst_slot} out of range"
            ));
        }
        let ckpt = self
            .prefix_catalog
            .get(id)
            .ok_or_else(|| format!("kvmem fork_prefix_into: checkpoint {id} not found"))?;
        ckpt.validate_reuse(prompt, layout_fingerprint(&self.raw_cfg))
            .map_err(|e| format!("kvmem fork_prefix_into: {e}"))?;
        let frontier = ckpt.frontier;
        // Single tier: every canonical lives in the session account, so reuse
        // is always the zero-copy sealed-record fork.
        let branch = ckpt
            .canonical()
            .fork_prefix(frontier)
            .map_err(|e| format!("kvmem fork_prefix_into fork: {e}"))?;
        if let Some(tail) = ckpt.mean_tail.as_ref() {
            branch
                .restore_mean_checkpoint(frontier, tail)
                .map_err(|e| format!("kvmem fork_prefix_into mean restore: {e}"))?;
        }
        let kv = KvStore::new(&self.kv_cfg).map_err(|e| format!("kvmem store: {e}"))?;
        kv.register_append(frontier)
            .map_err(|e| format!("kvmem fork_prefix_into append: {e}"))?;
        if let Some((_, keep)) = self.watermark {
            kv.set_runtime_select_budget((keep * KV_BLOCK_TOKENS) as u32)
                .map_err(|e| format!("kvmem watermark select budget: {e}"))?;
        }
        // A reused prefix is a NEW session: bump the incarnation so its
        // canonical stamps never alias the source's (design §3.9g).
        let lw_incarnation = self.slots[dst_slot].lw_incarnation + 1;
        self.slots[dst_slot] =
            self.fresh_slot_state(branch, kv, frontier, frontier, lw_incarnation);
        Ok(frontier)
    }

    /// Record that a request begins on `slot` (cold prefill or a
    /// continuation that reuses the existing prefix). Returns the store
    /// length the request starts from — a continuation's reuse point must
    /// equal it, because `feed_rows` only extends contiguously.
    pub fn begin_request(&mut self, slot: usize) -> Result<u32, String> {
        let st = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| format!("kvmem: begin_request slot {slot} out of range"))?;
        st.turn_start = st.fed;
        st.first_decode_done = false;
        // A cancelled verify may leave captured rows that were never committed.
        st.lw_pending_feed = None;
        // A new turn's question must not inherit the previous turn's Q span.
        st.q_ring_len = 0;
        st.q_ring_cursor = 0;
        // [P4.1② fix] arm the per-request scorer Query: the first Layerwise
        // decode/verify step of this continuation must re-score the resident pool
        // with the current question (ninfer per-request Query), not refine the
        // frozen previous-turn keep-set by Pressure (which reuses a stale
        // compact-tail binding and corrupts cross-turn recall).
        // 期2.3 (KVMem/nInfer `layerwise_query_at_source`): skip that Query
        // when the new question is already effective at the source — a proven
        // span identical to the previous turn's and wholly inside the accepted
        // prefix (`finalize_layerwise_query_source`: keep_selected). An
        // unproven span (`None`) keeps the historical per-request Query.
        st.lw_query_due = match st.request_query {
            None => true,
            Some(q) => {
                let same = st.last_query == Some(q);
                let inside = q.0 < q.1 && q.1 <= st.fed as u64;
                !(same && inside)
            }
        };
        Ok(st.fed)
    }

    /// 期2.3: install the current request's frontend-proven question span.
    /// Called at admit for every request (cold, reuse, continuation) BEFORE
    /// `begin_request` / the reuse import decision; the previous request's
    /// span moves to `last_query` (continuation identity).
    pub fn set_request_query(
        &mut self,
        slot: usize,
        query: Option<(u64, u64)>,
    ) -> Result<(), String> {
        let st = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| format!("kvmem: set_request_query slot {slot} out of range"))?;
        st.last_query = st.request_query;
        st.request_query = query;
        Ok(())
    }

    /// 期3 G3: install the request's internal content group (conversation
    /// lineage hash). Group metadata is a capacity fairness hint only — the
    /// reuse path still validates tokens/layout/native proofs.
    pub fn set_request_group(&mut self, slot: usize, group: u64) -> Result<(), String> {
        let st = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| format!("kvmem: set_request_group slot {slot} out of range"))?;
        st.convo_group = group;
        Ok(())
    }

    /// 期2.3: arm or clear the one-shot per-request scorer Query. The reuse
    /// import uses it: a NEW question forces the Query at the first decode
    /// step; KeepSelected (query already effective at the source) leaves it
    /// clear.
    pub fn set_layerwise_query_due(&mut self, slot: usize, due: bool) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_query_due = due;
        }
    }

    /// Host-store length of `slot` (tokens whose statistics are recorded).
    pub fn fed_tokens(&self, slot: usize) -> Option<u32> {
        self.slots.get(slot).map(|s| s.fed)
    }

    /// Slot `slot`'s layerwise stamp counters `(incarnation, seq)` (design
    /// §3.9g). The orchestration reads these to synthesize a fresh
    /// `MemoryStamp` per re-bootstrapped query; `None` when out of range.
    pub fn layerwise_counters(&self, slot: usize) -> Option<(u64, u64)> {
        self.slots.get(slot).map(|s| (s.lw_incarnation, s.lw_seq))
    }

    /// Advance slot `slot`'s layerwise transaction counter after an admitted
    /// query (design §3.9g). No-op when out of range. Legacy never calls it.
    pub fn advance_layerwise_seq(&mut self, slot: usize) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_seq += 1;
        }
    }

    /// [P4.1② fix] Consume slot `slot`'s per-request scorer-Query gate. Returns
    /// true exactly once per continuation request (armed by `begin_request`),
    /// then clears it so subsequent decode/verify steps of the same request
    /// refine by Pressure. The orchestration calls this in its mutable window
    /// before borrowing the controller for the carrier and passes the result as
    /// `force_query`. No-op (false) when out of range / under Legacy.
    pub fn take_layerwise_query_due(&mut self, slot: usize) -> bool {
        match self.slots.get_mut(slot) {
            Some(st) => std::mem::replace(&mut st.lw_query_due, false),
            None => false,
        }
    }

    /// Test-only: seed slot `slot`'s consumed frontier (`fed`) without the GPU
    /// statistics-feed machinery (`feed_rows` needs live model taps). The P4.1②
    /// G1 `engine_carrier_rows` GPU fixture populates the slot's statistics store
    /// directly through [`Self::layerwise_stores`] and uses this to set the host
    /// length the bootstrap reads. Never compiled into production (zero impact
    /// on the Legacy byte-neutral spine).
    #[cfg(test)]
    pub(crate) fn test_set_fed(&mut self, slot: usize, fed: u32) -> Result<(), String> {
        let st = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| format!("kvmem: test_set_fed slot {slot} out of range"))?;
        st.kv
            .register_append(fed)
            .map_err(|e| format!("kvmem: test_set_fed append: {e}"))?;
        st.fed = fed;
        Ok(())
    }
}
