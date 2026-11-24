// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::transfer::restore_page;
use super::*;

impl KvMemCtl {
    /// Lazily construct slot `slot`'s session-lifetime prefill pressure owner
    /// (decision A, design §3.10). Idempotent within a session: once built it is
    /// reused for every admitted query until [`Self::reset_slot`] drops it (the
    /// incarnation bump on reset guarantees a fresh session reconstructs under a
    /// distinct identity). `session_id` / `incarnation` are the ENGINE's 0-based
    /// counters; they are normalized to the core's 1-based logical identity with
    /// the SAME `+1` bijection `bootstrap_layerwise` applies, so the owner's
    /// session matches the canonical `MemoryStamp.session` the picker validates.
    /// `fa_layers` are the dense FA ordinals `0..n_fa_layers`.
    ///
    /// Options pin `block_tokens = 128` (the catalog layout block size, which the
    /// core requires to match exactly) and derive the select ceiling from the
    /// per-slot resident budget; interval/stride/half_life carry the ninfer
    /// defaults. Layerwise-only: under Legacy this returns `Err` (fail-closed —
    /// the owner is never created, keeping the default path byte-neutral).
    /// Admission is finite and fail-closed BEFORE any acquisition: the owner's
    /// complete mutable-history ceiling must fit the metadata pressure partition
    /// (③-E4, [`resolve_pressure_admission`]), mirroring ninfer's allocation-phase
    /// contract.
    pub fn ensure_layerwise_pressure(
        &mut self,
        slot: usize,
        session_id: u64,
        incarnation: u64,
        fa_layers: &[u32],
    ) -> Result<(), String> {
        resolve_pressure_admission(
            self.strategy,
            self.n_fa_layers,
            self.budget_blocks,
            &self.layerwise_quotas,
        )?;
        let st = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| format!("kvmem: ensure_layerwise_pressure slot {slot} out of range"))?;
        if st.lw_pressure.is_some() {
            return Ok(());
        }
        let session = SessionIdentity {
            id: session_id + 1,
            incarnation: incarnation + 1,
        };
        let options = PrefillOptions {
            block_tokens: KV_BLOCK_TOKENS as u32,
            select_tokens: (self.budget_blocks * KV_BLOCK_TOKENS) as u64,
            reserve_tokens: KV_BLOCK_TOKENS as u64,
            interval: 1024,
            stride: 32,
            half_life: 8192.0,
        };
        let pressure = PrefillPressure::new(session, fa_layers, &options)
            .map_err(|e| format!("kvmem: layerwise pressure create: {e}"))?;
        st.lw_pressure = Some(pressure);
        Ok(())
    }

    /// Immutable borrow of slot `slot`'s pressure owner (`None` under Legacy,
    /// before the first admitted query, or out of range). The owner stays owned
    /// by this controller; the driver borrows it for `select`/`snapshot`.
    pub fn layerwise_pressure(&self, slot: usize) -> Option<&PrefillPressure> {
        self.slots.get(slot).and_then(|st| st.lw_pressure.as_ref())
    }

    /// Mutable borrow of slot `slot`'s pressure owner for `accept`/`restore`
    /// (`None` under Legacy, before construction, or out of range). The caller
    /// must have called [`Self::ensure_layerwise_pressure`] first this session.
    pub fn layerwise_pressure_mut(&mut self, slot: usize) -> Option<&mut PrefillPressure> {
        self.slots
            .get_mut(slot)
            .and_then(|st| st.lw_pressure.as_mut())
    }

    /// Slot `slot`'s persistent PER-LAYER resident keep-set (decision A, design
    /// §3.10a): `keep[ordinal]` holds the canonical block ids selected for FA
    /// ordinal `ordinal` by the previous step (pressure.select on a Pressure
    /// step, the ② scorer on a Query step). Drives each layer's canonical
    /// `resident` list. Empty before the first selection / under Legacy; `None`
    /// when out of range. Borrowed for the bootstrap synthesis (an absent ordinal
    /// entry reads as `[]`).
    pub fn layerwise_keep_set(&self, slot: usize) -> Option<&[Vec<u64>]> {
        self.slots.get(slot).map(|st| st.lw_keep.as_slice())
    }

    /// Replace slot `slot`'s per-layer keep-set with the latest selection result
    /// (decision A). `keep[ordinal]` is the canonical sealed-page ids restored
    /// for FA ordinal `ordinal` this step; the bootstrap filters any id beyond
    /// the current sealed history defensively. No-op when out of range. Legacy
    /// never calls it (the keep-set stays empty).
    pub fn set_layerwise_keep_set(&mut self, slot: usize, keep: Vec<Vec<u64>>) {
        // D3-A S3 attribution trace (opt-in): the per-layer keep-set is the
        // refinement input that decides which pages each FA layer attends.
        if self.diagnostics.trace_keep {
            let mut line = String::new();
            for (i, ids) in keep.iter().enumerate() {
                line.push_str(&format!(" L{i}:"));
                for (j, id) in ids.iter().enumerate() {
                    if j > 0 {
                        line.push(',');
                    }
                    line.push_str(&id.to_string());
                }
            }
            eprintln!(
                "[kvmem-keep] slot {slot} ordinals={} per_layer={}{}",
                keep.len(),
                keep.first().map(|v| v.len()).unwrap_or(0),
                line
            );
        }
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_keep = keep;
        }
    }

    /// Move slot `slot`'s pressure owner OUT of the controller (design §3.10a
    /// borrow resolution). The scoped serve orchestration takes the owner before
    /// building the carrier (which borrows the slot's `&RawStore`/`&KvStore`), so
    /// `select`/`accept` can use `&mut PrefillPressure` independently of those
    /// borrows; [`Self::restore_layerwise_pressure`] puts it back after the
    /// forward. `None` under Legacy, before construction, or out of range. The
    /// slot keeps its keep-set / counters; only the pressure owner is detached.
    pub fn take_layerwise_pressure(&mut self, slot: usize) -> Option<PrefillPressure> {
        self.slots
            .get_mut(slot)
            .and_then(|st| st.lw_pressure.take())
    }

    /// Return a previously [`Self::take_layerwise_pressure`]-detached owner to
    /// slot `slot` (design §3.10a). No-op when out of range. The orchestration
    /// calls this exactly once after the forward, before restoring the `KvMemCtl`
    /// into the rig, so the session-lifetime owner persists to the next step.
    pub fn restore_layerwise_pressure(&mut self, slot: usize, pressure: PrefillPressure) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_pressure = Some(pressure);
        }
    }

    /// Construct slot `slot`'s persistent OUTPUT-scope pressure owner (P4.1②,
    /// design §A.6) if absent — the MTP-verify sibling of
    /// [`Self::ensure_layerwise_pressure`]. A verify span accepts a PREFIX of its
    /// rows (partial acceptance), which the all-or-nothing prefill owner cannot
    /// express (core layerwise_pressure.cpp:79). NonThinking semantic scope (every
    /// verify row is body output), interval=1 (sample every body row), ninfer
    /// decay/commit defaults — mirroring the ① partial-accept fixture. Admission
    /// is the same finite fail-closed ceiling as the prefill owner (③-E4). The
    /// caller gates on Layerwise + MTP, so the owner is never built on the AR-only
    /// / Legacy path (byte-neutral). Idempotent: an existing owner is retained.
    pub fn ensure_layerwise_output_pressure(
        &mut self,
        slot: usize,
        session_id: u64,
        incarnation: u64,
        fa_layers: &[u32],
    ) -> Result<(), String> {
        resolve_pressure_admission(
            self.strategy,
            self.n_fa_layers,
            self.budget_blocks,
            &self.layerwise_quotas,
        )?;
        let st = self.slots.get_mut(slot).ok_or_else(|| {
            format!("kvmem: ensure_layerwise_output_pressure slot {slot} out of range")
        })?;
        if st.lw_output_pressure.is_some() {
            return Ok(());
        }
        let session = SessionIdentity {
            id: session_id + 1,
            incarnation: incarnation + 1,
        };
        let prefill_opts = PrefillOptions {
            block_tokens: KV_BLOCK_TOKENS as u32,
            select_tokens: (self.budget_blocks * KV_BLOCK_TOKENS) as u64,
            reserve_tokens: KV_BLOCK_TOKENS as u64,
            interval: 1024,
            stride: 32,
            half_life: 8192.0,
        };
        let output_opts = OutputOptions {
            scope: OutputScope::NonThinking as u32,
            interval: 1,
            decay: 0.5,
            commit_max_bytes: 16_777_216,
        };
        let pressure =
            OutputPressure::new_external(session, fa_layers, &prefill_opts, &output_opts)
                .map_err(|e| format!("kvmem: layerwise output pressure create: {e}"))?;
        st.lw_output_pressure = Some(pressure);
        Ok(())
    }

    /// Immutable borrow of slot `slot`'s output-pressure owner (`None` under
    /// Legacy / AR-only, before the first admitted verify, or out of range). The
    /// driver borrows it for `select`/`snapshot`; the orchestration owns it for
    /// `begin_scope`/`prepare_scope`/`accept`.
    pub fn layerwise_output_pressure(&self, slot: usize) -> Option<&OutputPressure> {
        self.slots
            .get(slot)
            .and_then(|st| st.lw_output_pressure.as_ref())
    }

    /// Mutable borrow of slot `slot`'s output-pressure owner for
    /// `begin_scope`/`prepare_scope`/`accept`/`restore` (`None` under Legacy /
    /// AR-only, before construction, or out of range). The caller must have called
    /// [`Self::ensure_layerwise_output_pressure`] first this session.
    pub fn layerwise_output_pressure_mut(&mut self, slot: usize) -> Option<&mut OutputPressure> {
        self.slots
            .get_mut(slot)
            .and_then(|st| st.lw_output_pressure.as_mut())
    }

    /// Move slot `slot`'s output-pressure owner OUT of the controller (P4.1②
    /// borrow resolution, mirrors [`Self::take_layerwise_pressure`]). The scoped
    /// serve orchestration takes the owner before building the verify carrier
    /// (which borrows the slot's `&RawStore`/`&KvStore`), so
    /// `begin_scope`/`prepare_scope`/`accept` can use `&mut OutputPressure`
    /// independently of those borrows; [`Self::restore_layerwise_output_pressure`]
    /// puts it back after the forward. `None` under Legacy / AR-only, before
    /// construction, or out of range. The slot keeps its keep-set / counters /
    /// prefill owner; only the output owner is detached.
    pub fn take_layerwise_output_pressure(&mut self, slot: usize) -> Option<OutputPressure> {
        self.slots
            .get_mut(slot)
            .and_then(|st| st.lw_output_pressure.take())
    }

    /// Return a previously [`Self::take_layerwise_output_pressure`]-detached
    /// owner to slot `slot` (P4.1②). No-op when out of range. The orchestration
    /// calls this exactly once after the verify forward, before restoring the
    /// `KvMemCtl` into the rig, so the session-lifetime owner persists to the next
    /// verify step.
    pub fn restore_layerwise_output_pressure(&mut self, slot: usize, pressure: OutputPressure) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_output_pressure = Some(pressure);
        }
    }

    /// Store slot `slot`'s pending output-scope acceptance snapshot (P4.1②,
    /// design §A.6). The orchestration calls this after the verify forward extracts
    /// the carrier's captured mass ([`crate::layerwise_driver::LayerwiseQuery::extract_output_acceptance`]),
    /// BEFORE freeing the carrier and restoring the controller, so the partial
    /// acceptance survives to the post-sampling publish in `mtp_verify_accept_step`.
    /// No-op when out of range. Legacy / AR-only never call it (byte-neutral).
    pub fn set_layerwise_pending_output(&mut self, slot: usize, snap: OutputAcceptanceSnapshot) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_pending_output = Some(snap);
        }
    }

    /// Move slot `slot`'s pending output-scope acceptance snapshot OUT of the
    /// controller (P4.1②). `mtp_verify_accept_step` calls this after `feed_rows`
    /// (the host store advanced by the same `result.advance`) and publishes it via
    /// [`crate::layerwise_driver::publish_output_acceptance`] at the confirmed
    /// prefix. `None` under Legacy / AR-only / when no verify is in flight / out of
    /// range, so the publish is a no-op on every non-Layerwise-MTP path.
    pub fn take_layerwise_pending_output(
        &mut self,
        slot: usize,
    ) -> Option<OutputAcceptanceSnapshot> {
        self.slots
            .get_mut(slot)
            .and_then(|st| st.lw_pending_output.take())
    }

    /// P4.1② G3: whether the composite head aux plane is active (Layerwise AND
    /// native MTP). The orchestration gates every head-plane path on this; under
    /// Legacy / AR-only it is false and the head keeps the historical
    /// full-resident KvCache path (byte-neutral).
    pub fn head_plane_enabled(&self) -> bool {
        self.head_plane
    }

    /// P4.1③ H5: the resolved head-history policy (FirstLayer auto-default under
    /// Layerwise+MTP; UpstreamGlobal opt-in). The orchestration branches the head
    /// aux binding on this at the prefill-seed and the decode/verify carrier
    /// restore-hooks. Unspecified under Legacy / AR-only (never read there).
    ///
    /// A1 starvation guard (single source of truth): below
    /// [`MTP_HISTORY_UG_MIN_BLOCKS`] the global Mean-K reduction dilutes the
    /// draft-critical signal at tiny head budgets (recite sweep 8-block
    /// regression 0.381 vs FL 0.782), so an opted-in UpstreamGlobal is reported
    /// as FirstLayer here. EVERY decision point (prefill-seed aux binding plus
    /// the AR-decode and MTP-verify carrier restore-hooks in serve_engine) reads
    /// this accessor, so the fallback is uniform and the UG core algorithm is
    /// untouched when engaged (≥ floor). ninfer-faithful operational guard, not
    /// a core-algorithm change.
    pub fn layerwise_mtp_history(&self) -> LayerwiseMtpHistory {
        if self.layerwise_mtp_history == LayerwiseMtpHistory::UpstreamGlobal
            && self.budget_blocks < MTP_HISTORY_UG_MIN_BLOCKS
        {
            LayerwiseMtpHistory::FirstLayer
        } else {
            self.layerwise_mtp_history
        }
    }

    /// P4.1② G3: the head aux plane's ordinal in the composite host record
    /// (`= n_fa_layers`, the appended LAST ordinal). The head aux-plane
    /// write-through gathers the head's `k_gpu[0]`/`v_gpu[0]` into this ordinal;
    /// the trunk GPU paged arena (`0..n_fa_layers`) never addresses it. Only
    /// meaningful when [`Self::head_plane_enabled`].
    pub fn aux_ordinal(&self) -> u32 {
        self.aux_ordinal
    }

    /// P4.1② G3: lazily construct slot `slot`'s session-lifetime head resident
    /// map (design §A.8), the head-plane sibling of
    /// [`Self::ensure_layerwise_pressure`]. Idempotent within a session: once
    /// built it is reused for every admitted MTP verify until
    /// [`Self::reset_slot`] drops it. Layerwise + MTP only: under Legacy / AR-only
    /// this returns `Err` (fail-closed — the map is never created, keeping the
    /// default path byte-neutral). The head does NOT mass-score, so there is no
    /// admission ceiling here (unlike the trunk pressure owners); the map starts
    /// empty and the first FirstLayer binding (G5) populates it.
    pub fn ensure_head_plane(&mut self, slot: usize) -> Result<(), String> {
        if !self.head_plane {
            return Err("kvmem: head plane requires layerwise recall and native MTP".to_string());
        }
        let st = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| format!("kvmem: ensure_head_plane slot {slot} out of range"))?;
        if st.lw_head.is_none() {
            st.lw_head = Some(HeadPlaneState::default());
        }
        Ok(())
    }

    /// Immutable borrow of slot `slot`'s head resident map (`None` under Legacy /
    /// AR-only, before the first admitted verify, or out of range). The driver
    /// reads `auxiliary_pages` / `auxiliary_removed_tokens` to build the aux
    /// `LayerWorkingSetPlan` and the RoPE shift.
    pub fn head_plane(&self, slot: usize) -> Option<&HeadPlaneState> {
        self.slots.get(slot).and_then(|st| st.lw_head.as_ref())
    }

    /// Mutable borrow of slot `slot`'s head resident map for the post-restore
    /// update (`auxiliary_pages` / `auxiliary_view_version` /
    /// `auxiliary_removed_tokens`, ninfer NL:10609-10612). The caller must have
    /// called [`Self::ensure_head_plane`] first this session.
    pub fn head_plane_mut(&mut self, slot: usize) -> Option<&mut HeadPlaneState> {
        self.slots.get_mut(slot).and_then(|st| st.lw_head.as_mut())
    }

    /// Move slot `slot`'s head resident map OUT of the controller (P4.1② G3
    /// borrow resolution, mirrors [`Self::take_layerwise_pressure`]). The scoped
    /// serve orchestration takes the map before building the verify carrier
    /// (which borrows the slot's `&RawStore`/`&KvStore`), so the aux plan /
    /// boundary rebuild can use `&mut HeadPlaneState` independently of those
    /// borrows; [`Self::restore_head_plane`] puts it back after the forward.
    /// `None` under Legacy / AR-only, before construction, or out of range.
    pub fn take_head_plane(&mut self, slot: usize) -> Option<HeadPlaneState> {
        self.slots.get_mut(slot).and_then(|st| st.lw_head.take())
    }

    /// Return a previously [`Self::take_head_plane`]-detached map to slot `slot`
    /// (P4.1② G3). No-op when out of range. The orchestration calls this exactly
    /// once after the verify forward, before restoring the `KvMemCtl` into the
    /// rig, so the session-lifetime map persists to the next verify step.
    pub fn restore_head_plane(&mut self, slot: usize, head: HeadPlaneState) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_head = Some(head);
        }
    }

    /// P4.1② G3: validate slot `slot`'s head resident map against the accepted
    /// `frontier` (ninfer `KvmemLayerViews::validate`, NL:12267-12270). Returns
    /// the reconstructed aux extent, or `Err` if the map is absent / does not
    /// reconstruct the frontier. The boundary rebuild (G4) calls this before
    /// attending the head stripe, so a corrupted head history fails closed.
    pub fn validate_head_plane(&self, slot: usize, frontier: u64) -> Result<u64, String> {
        let head = self
            .head_plane(slot)
            .ok_or_else(|| format!("kvmem: validate_head_plane slot {slot} has no head map"))?;
        head.validate(frontier)
    }

    /// P4.1② G3: store slot `slot`'s per-Query FirstLayer auxiliary-selection
    /// binding (① primitive). The restore-hook (G5) creates a fresh
    /// [`AuxiliarySelection`] per verify span and binds the head history from the
    /// target's first FA layer's restored ids; the boundary rebuild consumes it
    /// via [`Self::take_aux_selection`]. No-op when out of range. Legacy / AR-only
    /// never call it (byte-neutral).
    pub fn set_aux_selection(&mut self, slot: usize, sel: AuxiliarySelection) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_aux_selection = Some(sel);
        }
    }

    /// Move slot `slot`'s per-Query auxiliary-selection binding OUT of the
    /// controller (P4.1② G3, mirrors [`Self::take_layerwise_pending_output`]).
    /// `None` under Legacy / AR-only / when no verify is in flight / out of range,
    /// so the boundary rebuild is a no-op on every non-Layerwise-MTP path.
    pub fn take_aux_selection(&mut self, slot: usize) -> Option<AuxiliarySelection> {
        self.slots
            .get_mut(slot)
            .and_then(|st| st.lw_aux_selection.take())
    }

    /// Clone slot `slot`'s current auxiliary-selection binding WITHOUT
    /// consuming it. The binding describes the trunk working set selected at
    /// the last pressure step and stays valid until the next selection; every
    /// MTP draft over that window must restore the same head view, so the
    /// draft path peeks instead of taking the one-shot binding (K12).
    /// `None` under Legacy / AR-only / before any selection / out of range.
    pub fn peek_aux_selection(&self, slot: usize) -> Option<AuxiliarySelection> {
        self.slots
            .get(slot)
            .and_then(|st| st.lw_aux_selection.clone())
    }

    /// P4.1② G3b: host-side seal of the head auxiliary plane. Seals this feed's
    /// gathered packed Q8 K/V rows into the slot's COMPOSITE host record at the
    /// aux ordinal (`aux_ordinal`, the appended `n_fa`-th layer), mirroring the
    /// trunk `write_through_archive` seal but for the single head plane. Split
    /// from the GPU gather so the seal + readback round-trip is host-testable.
    ///
    /// Fail-closed unless the head plane is active (Layerwise + MTP): Legacy /
    /// AR-only never co-archive the head, keeping the default byte-neutral.
    /// `k_rows`/`v_rows` are the contiguous packed rows for `[pos0, pos0+n_rows)`
    /// (`n_rows * per_pos_bytes` / `n_rows * per_pos_v_bytes` bytes).
    pub(super) fn seal_head_plane(
        &mut self,
        slot: usize,
        pos0: usize,
        n_rows: usize,
        k_rows: &[u8],
        v_rows: &[u8],
    ) -> Result<(), String> {
        if !self.head_plane {
            return Err(
                "kvmem head write-through: head plane inactive (requires Layerwise + MTP)"
                    .to_string(),
            );
        }
        if slot >= self.slots.len() {
            return Err(format!(
                "kvmem head write-through: slot {slot} out of range"
            ));
        }
        if n_rows == 0 {
            return Ok(());
        }
        let aux = self.aux_ordinal;
        let raw = &mut self.slots[slot].raw;
        raw.write_layer_k_gpu(pos0 as u32, n_rows as u32, aux, k_rows)
            .map_err(|e| format!("kvmem head write-through K seal (aux {aux}): {e}"))?;
        raw.write_layer_v_gpu(pos0 as u32, n_rows as u32, aux, v_rows)
            .map_err(|e| format!("kvmem head write-through V seal (aux {aux}): {e}"))?;
        Ok(())
    }

    /// P4.1② G3b: head auxiliary-plane write-through (GPU gather + seal). Gather
    /// this feed's packed Q8 K/V rows out of the head's single-layer GPU cache
    /// and seal them into the composite host record at the aux ordinal, so EVERY
    /// accepted head token's payload survives for drop-evict + stripe
    /// restore-before-attend (design §A.8 Q1). Mirrors the trunk
    /// `write_through_archive`, but the head cache is position-major CONTIGUOUS
    /// (a plain `KvCache`, no block table): the fed range is ONE contiguous
    /// gather per plane at `src_pos0 * row_bytes`.
    ///
    /// `src_pos0` is the fed range's start position WITHIN the head tensor. The
    /// full-resident head `KvCache` (G3b) is addressed at the global frontier
    /// (`src_pos0 == pos0`); the bounded head pool (G3c, window + 64 scratch)
    /// slides, so its caller passes the pool-local offset. The head tensor's
    /// per-position byte stride MUST equal the trunk geometry (`per_pos_bytes` /
    /// `per_pos_v_bytes`) — the head shares the trunk KV geometry.
    ///
    /// Only called when `head_plane_enabled()`; fails closed otherwise (before
    /// any GPU work), keeping Legacy / AR-only byte-neutral.
    pub fn head_write_through_archive(
        &mut self,
        gpu: &Gpu,
        head_k: &GpuTensor,
        head_v: &GpuTensor,
        slot: usize,
        pos0: usize,
        n_rows: usize,
        src_pos0: usize,
    ) -> Result<(), String> {
        if !self.head_plane {
            return Err(
                "kvmem head write-through: head plane inactive (requires Layerwise + MTP)"
                    .to_string(),
            );
        }
        if n_rows == 0 {
            return Ok(());
        }
        let krow = self.k_page_bytes / KV_BLOCK_TOKENS;
        let vrow = self.v_page_bytes / KV_BLOCK_TOKENS;
        // The feed must fit the tap row capacity (the trunk staging bound); the
        // head gather allocates its own contiguous buffers below.
        if n_rows * krow > self.wt_k.len() || n_rows * vrow > self.wt_v.len() {
            return Err(format!(
                "kvmem head write-through: feed of {n_rows} rows exceeds the staging \
                 capacity ({} K / {} V bytes)",
                self.wt_k.len(),
                self.wt_v.len()
            ));
        }
        // The head cache is contiguous position-major: one D2H per plane into a
        // freshly sized buffer (the head is a single layer, so unlike the trunk
        // it needs no shared zero-alloc staging arena).
        let mut k_rows = vec![0u8; n_rows * krow];
        let mut v_rows = vec![0u8; n_rows * vrow];
        gpu.hip
            .memcpy_dtoh_at(&mut k_rows, &head_k.buf, src_pos0 * krow)
            .map_err(|e| format!("kvmem head write-through K gather: {e:?}"))?;
        gpu.hip
            .memcpy_dtoh_at(&mut v_rows, &head_v.buf, src_pos0 * vrow)
            .map_err(|e| format!("kvmem head write-through V gather: {e:?}"))?;
        self.seal_head_plane(slot, pos0, n_rows, &k_rows, &v_rows)
    }

    /// P4.1② G3c: the MTP head's BOUNDED token capacity (design §A.8 Q1, the
    /// hipfire analog of ninfer `layerwise_mtp_auxiliary_capacity`): the trunk
    /// layerwise working-set window (`window_tokens`) PLUS ONE native draft-scratch
    /// block (`KV_BLOCK_TOKENS`). Under Layerwise the head is bounded to the trunk
    /// window — accepted head history beyond it is co-archived to the composite
    /// record's aux plane and restored-before-attend (G4), never full-resident.
    /// Fail-closed off the Layerwise+MTP path and on a non-page-aligned / zero
    /// window (the ① [`mtp_auxiliary_capacity`] guard), keeping Legacy byte-neutral.
    pub fn head_pool_capacity(&self, window_tokens: usize) -> Result<usize, String> {
        if !self.head_plane {
            return Err(
                "kvmem head pool: head plane inactive (requires Layerwise + MTP)".to_string(),
            );
        }
        mtp_auxiliary_capacity(window_tokens, KV_BLOCK_TOKENS)
    }

    /// P4.1② f2 (E): the bounded head KV token capacity for LIVE allocation, or
    /// `None` off the Layerwise+MTP path. The window is the trunk layerwise
    /// working-set window (`budget_blocks * KV_BLOCK_TOKENS`, the same source the
    /// G4 head carrier uses); the capacity adds one native draft-scratch block
    /// ([`Self::head_pool_capacity`]). Legacy / AR-only (`head_plane` false) return
    /// `None` ⇒ the caller keeps the full-resident head cache (byte-neutral). A
    /// non-page-aligned / zero window fails closed to `None` rather than bounding
    /// to a bad geometry.
    pub fn head_bounded_capacity(&self) -> Option<usize> {
        if !self.head_plane {
            return None;
        }
        let window = self.budget_blocks().checked_mul(KV_BLOCK_TOKENS)?;
        self.head_pool_capacity(window).ok()
    }

    /// P4.1② f2 (E): the head prefill COMPACT-write shift at `frontier` — the
    /// hipfire analog of ninfer `removed_tokens` / `cache_position_shift`
    /// (NL:8442): `max(0, frontier - window)`. The bounded head staging cache
    /// writes prompt K/V at `global - shift` (COMPACT) while RoPE stays GLOBAL
    /// (`rope_shift = shift`). Returns `0` off the Layerwise+MTP path (`head_plane`
    /// false) ⇒ compact == global, byte-neutral. The resident head cache is only
    /// per-chunk archive staging under Layerwise (the G4 carrier is the attention
    /// source), so each chunk re-stages independently — no slide.
    pub fn head_prefill_shift(&self, frontier: usize) -> usize {
        if !self.head_plane {
            return 0;
        }
        let window = self.budget_blocks().saturating_mul(KV_BLOCK_TOKENS);
        frontier.saturating_sub(window)
    }

    /// P4.1② G3c: build the MTP head's bounded composite-slot pool geometry (①
    /// [`mtp_head_pool`]): ONE fa layer (the head aux plane), `window_tokens` of
    /// accepted history + one draft-scratch block, `window_tokens / KV_BLOCK_TOKENS
    /// + 1` slots. The head owns this pool INDEPENDENTLY of the trunk (ninfer's
    /// independent resident map + `LayerWorkingSetPlan`, quota = window). The K/V
    /// strides are the shared trunk/head Q8 geometry (`per_pos_bytes`). Pure host
    /// geometry — the caller (serve) allocates the bounded head KV from it,
    /// replacing the historical full-resident head `KvCache`. Fail-closed off the
    /// Layerwise+MTP path; Legacy / AR-only allocate nothing (byte-neutral).
    pub fn head_pool(&self, window_tokens: usize) -> Result<LayerwisePagePool, String> {
        if !self.head_plane {
            return Err(
                "kvmem head pool: head plane inactive (requires Layerwise + MTP)".to_string(),
            );
        }
        mtp_head_pool(
            window_tokens,
            KV_BLOCK_TOKENS,
            self.k_page_bytes / KV_BLOCK_TOKENS,
            self.v_page_bytes / KV_BLOCK_TOKENS,
        )
    }

    /// P4.1② G3c: the MTP head's INDEPENDENT `LayerWorkingSetPlan` (design §A.8
    /// Q1), the hipfire analog of ninfer's `aux_catalog.plan_layer` (NL:10578-10584).
    /// The head does NOT mass-score: it follows the target's first FA layer's
    /// restored ids (FirstLayer, bound by G5) and plans its OWN aux-plane residency
    /// against an independent quota (`window_tokens`). Pure host data — no GPU, no
    /// state machine — so the plan is computable before the verify carrier borrows
    /// the slot's stores.
    ///
    /// Builds a SINGLE-layer (aux ordinal `n_fa`) [`MemoryCatalog`] over the
    /// COMPOSITE Main+aux descriptor ([`LayerwiseMtpQ8Layout`], `Q8_MAIN_MTP_FORMAT`),
    /// exactly mirroring ninfer's `LayerwiseMemoryCatalog(archive.descriptor,
    /// {{model, aux}})`: the canonical frame is the sealed full pages of the accepted
    /// head history, the resident list is the head's own map
    /// ([`HeadPlaneState::auxiliary_pages`], widened `u32`→`u64` at this boundary,
    /// NL:10575), and the stamp's `view_version` is the head's independent
    /// `auxiliary_view_version` (NL:10576). The selection budget is ninfer's
    /// `{window, 0, window, retained}` — `host_payload_bytes` is the full composite
    /// record envelope (`full_pages * record_bytes`), the exact value the core
    /// `plan_working_set` charges (`host_records * layout.record_bytes`).
    ///
    /// `selected` are the FirstLayer-bound target page ids (a subset of the sealed
    /// history); `fed` is the accepted head frontier (the canonical frame is its
    /// full-page boundary `fed / 128 * 128`, the partial tail is the suffix §3.9g);
    /// `session_id` is the engine session (mapped 1-based like the trunk bootstrap).
    /// `head` is the DETACHED resident map (G3 borrow resolution: the orchestration
    /// `take_head_plane`s it before building the carrier that borrows the slot's
    /// `&RawStore`, so this `&self` plan coexists with this borrow); the caller
    /// gates existence via `take_head_plane` → `Some`. Fail-closed off the
    /// Layerwise+MTP path, on a non-page-aligned / zero window, or with no sealed
    /// full page — Legacy / AR-only never plan the head (byte-neutral). Returns the
    /// core [`LayerPlan`] the G4 stripe restore executes; the post-restore map
    /// update (NL:10609-10612) is G4's concern.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_head_plane(
        &self,
        slot: usize,
        head: &HeadPlaneState,
        selected: &[u64],
        fed: u64,
        window_tokens: usize,
        session_id: u64,
        max_position: u32,
        model: &str,
    ) -> Result<LayerPlan, String> {
        if !self.head_plane {
            return Err(
                "kvmem head plan: head plane inactive (requires Layerwise + MTP)".to_string(),
            );
        }
        // `head` is the DETACHED resident map (G3 borrow resolution: the serve
        // orchestration `take_head_plane`s it before building the carrier, which
        // borrows the slot's `&RawStore`). The plan reads it independently of that
        // borrow; the caller guarantees it exists (`take_head_plane` → `Some`).
        let (incarnation, seq) = self
            .layerwise_counters(slot)
            .ok_or_else(|| format!("kvmem head plan: slot {slot} out of range"))?;
        let bt = KV_BLOCK_TOKENS as u64;
        if window_tokens == 0 || window_tokens % KV_BLOCK_TOKENS != 0 {
            return Err(format!(
                "kvmem head plan: window {window_tokens} must be a positive multiple of {KV_BLOCK_TOKENS}"
            ));
        }
        let full_pages = fed / bt;
        if full_pages == 0 {
            return Err("kvmem head plan: no sealed full page yet".to_string());
        }
        let frontier = full_pages * bt;
        // Composite Main+aux descriptor; the head plans ONLY the aux ordinal.
        let mtp = LayerwiseMtpQ8Layout::resolve(
            self.kv_mode,
            self.n_fa_layers,
            self.n_kv_heads,
            self.head_dim,
        )
        .map_err(|e| format!("kvmem head plan: mtp layout: {e}"))?;
        let layout = mtp
            .payload_layout()
            .map_err(|e| format!("kvmem head plan: manifest: {e}"))?;
        let record_bytes = layout.record_bytes();
        let descriptor = MemoryDescriptor::new(
            &layerwise_backend(model, self.kv_mode),
            &layout,
            &layerwise_capabilities(max_position),
        )
        .map_err(|e| format!("kvmem head plan: descriptor: {e}"))?;
        // 1-based logical identity (mirrors bootstrap_layerwise; the core requires
        // every identity field != 0 and block.session == stamp.session).
        let session = SessionIdentity {
            id: session_id + 1,
            incarnation: incarnation + 1,
        };
        let revision = seq + 1;
        let blocks: Vec<BlockCopies> = (0..full_pages)
            .map(|page| BlockCopies {
                block: BlockDescriptor {
                    session,
                    id: page,
                    original_start: page * bt,
                    valid_tokens: KV_BLOCK_TOKENS as u32,
                    content_version: 1,
                    statistics_version: 1,
                },
                host_version: 1,
            })
            .collect();
        // The head's independent resident map (defensive filter to the sealed
        // history, mirroring the trunk keep-set filter).
        let resident: Vec<u64> = head
            .auxiliary_pages
            .iter()
            .copied()
            .map(u64::from)
            .filter(|&id| id < full_pages)
            .collect();
        let stamp = MemoryStamp {
            valid: 1,
            session,
            history_revision: revision,
            // The head's OWN view version (ninfer NL:10576), not the trunk's.
            view_version: head.auxiliary_view_version,
            frontier,
            execution_history: revision,
        };
        let aux = LayerMemorySnapshot {
            model_layer: self.aux_ordinal,
            memory: MemorySnapshot {
                stamp,
                blocks,
                resident,
            },
        };
        let catalog = MemoryCatalog::new(&descriptor, &[aux])
            .map_err(|e| format!("kvmem head plan: catalog: {e}"))?;
        let budget = MemoryBudget {
            selected_tokens: window_tokens as u64,
            reserve_tokens: 0,
            device_tokens: window_tokens as u64,
            host_payload_bytes: full_pages * record_bytes,
        };
        let selection = LayerSelection {
            model_layer: self.aux_ordinal,
            expected: stamp,
            selected: selected.to_vec(),
            mandatory: Vec::new(),
            budget,
        };
        catalog
            .plan_layer(&selection)
            .map_err(|e| format!("kvmem head plan: plan_layer aux {}: {e}", self.aux_ordinal))
    }

    /// P4.1② G4: publish the head's post-restore resident map (ninfer
    /// NL:10609-10612). After the aux stripe restore returns its [`ExecutionView`],
    /// the head's INDEPENDENT map becomes the restored compact view:
    /// `auxiliary_pages` = the view's canonical block ids (chronological),
    /// `auxiliary_view_version` = the view stamp's view version, and
    /// `auxiliary_removed_tokens` = `frontier - view.visible_tokens` — the RoPE
    /// shift for the head's restored rows (NL:5848/10328). The head ALGORITHM is
    /// unchanged (full-history attention); only its KV residency map advances.
    ///
    /// Validates the ninfer aux invariant (`visible + removed == frontier`,
    /// NL:10614 `maps.validate`) BEFORE publishing, so a corrupted or
    /// non-chronological restore can never be attended. `head` is the DETACHED
    /// resident map (G3 borrow resolution): the publish writes it in place
    /// (`&self`, no `&mut ctl`), so it coexists with the verify carrier's
    /// `&RawStore` borrow; the caller `restore_head_plane`s it back after the
    /// forward. Fail-closed off the Layerwise+MTP path, when the visible extent
    /// exceeds the frontier, or on a block id that does not fit `u32` — Legacy /
    /// AR-only never publish a head map (byte-neutral).
    pub fn accept_head_restore(
        &self,
        slot: usize,
        head: &mut HeadPlaneState,
        view: &ExecutionView,
        frontier: u64,
    ) -> Result<(), String> {
        if !self.head_plane {
            return Err(
                "kvmem head restore: head plane inactive (requires Layerwise + MTP)".to_string(),
            );
        }
        let removed = auxiliary_removed_tokens(frontier, view.visible_tokens)
            .map_err(|e| format!("kvmem head restore: {e}"))?;
        let mut pages = Vec::with_capacity(view.blocks.len());
        for b in &view.blocks {
            pages.push(u32::try_from(b.block.id).map_err(|_| {
                format!("kvmem head restore: block id {} overflow u32", b.block.id)
            })?);
        }
        // ninfer restores into a TEMPORARY map and commits only after validate
        // (NL:10614/10621): build the candidate, validate the aux invariant, and
        // publish atomically so a failed restore never corrupts the live map.
        // `head` is the DETACHED resident map (G3 borrow resolution); the caller
        // `restore_head_plane`s it back after the forward.
        let candidate = HeadPlaneState {
            auxiliary_pages: pages,
            auxiliary_view_version: view.stamp.view_version,
            auxiliary_removed_tokens: removed,
        };
        candidate
            .validate(frontier)
            .map_err(|e| format!("kvmem head restore: slot {slot}: {e}"))?;
        *head = candidate;
        Ok(())
    }

    /// P4.1② G4 (f2-D): publish the head's accepted canonical append after the
    /// consume pass co-archives the accepted verify rows — the hipfire analog of
    /// ninfer `accept` calling `maps.accepted_auxiliary_append(begin, end)`
    /// (NL:10387). Unlike [`Self::accept_head_restore`] (which publishes a
    /// DETACHED map during a restore, `&self`), the consume runs with no carrier
    /// borrowing the slot, so this advances the LIVE map in place (`&mut self`).
    /// `begin`/`end` are the GLOBAL accepted frontier span (the same `pos` /
    /// `pos + advance` fed to [`Self::feed_rows`]); the archive is indexed by
    /// absolute position. Fail-closed off the Layerwise+MTP path, on a missing
    /// head map, or on a map that does not reconstruct `end` — Legacy / AR-only
    /// never publish a head append (byte-neutral).
    pub fn accept_head_append(&mut self, slot: usize, begin: u64, end: u64) -> Result<(), String> {
        if !self.head_plane {
            return Err(
                "kvmem head append: head plane inactive (requires Layerwise + MTP)".to_string(),
            );
        }
        let head = self
            .head_plane_mut(slot)
            .ok_or_else(|| format!("kvmem head append: slot {slot} has no head map"))?;
        head.append_accepted(begin, end)
            .map_err(|e| format!("kvmem head append: slot {slot}: {e}"))
    }

    /// Feed the tapped K rows [row_start, row_start + n_rows) of `slot`'s
    /// step batch to the per-layer key statistics. `pos0` is the absolute
    /// position of the first row and must be the store's current length.
    pub fn feed_rows(
        &mut self,
        gpu: &Gpu,
        taps: &KvMemTaps,
        slot: usize,
        pos0: usize,
        n_rows: usize,
        row_start: usize,
        pool: &SlotPool,
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
    ) -> Result<(), String> {
        if n_rows == 0 {
            return Ok(());
        }
        if slot >= self.slots.len() {
            return Err(format!("kvmem: feed_rows slot {slot} out of range"));
        }
        let n_fa = self.n_fa_layers;
        let trace = self.diagnostics.trace;
        let st = &mut self.slots[slot];
        if st.fed != pos0 as u32 {
            return Err(format!(
                "kvmem slot {slot}: statistics feed at position {pos0} but the \
                 store holds {} tokens (feed must be contiguous)",
                st.fed
            ));
        }
        let timed = self.diagnostics.time;
        let t_all = std::time::Instant::now();
        let mut t_read = std::time::Duration::ZERO;
        let mut t_write = std::time::Duration::ZERO;
        // K3: gather every FullAttention layer's fed K rows into the pinned
        // slab with one stream-ordered D2H per layer, then a single wait. The
        // per-layer host writes below stay in ordinal order, so the archived
        // statistics are byte-identical to the former 16 blocking reads.
        let row_bytes = n_rows * self.k_width * std::mem::size_of::<f32>();
        let need = n_fa * row_bytes;
        self.ensure_xfer_slab(gpu, need)?;
        let stream = gpu.steady_copy_stream();
        let t = std::time::Instant::now();
        {
            let slab = self
                .xfer_slab
                .as_mut()
                .expect("xfer slab after ensure")
                .as_bytes_mut();
            taps.read_k_rows_all_layers_into(gpu, row_start, n_rows, &mut slab[..need], stream)
                .map_err(|e| format!("kvmem tap read (batch {n_fa} layers): {e:?}"))?;
        }
        if timed {
            t_read += t.elapsed();
        }
        let st = &mut self.slots[slot];
        for il in 0..n_fa {
            let t = std::time::Instant::now();
            let bytes = &self
                .xfer_slab
                .as_mut()
                .expect("xfer slab after ensure")
                .as_bytes_mut()[il * row_bytes..(il + 1) * row_bytes];
            let rows = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n_rows * self.k_width)
            };
            st.raw
                .write_layer_mean_k(pos0 as u32, n_rows as u32, il as u32, rows)
                .map_err(|e| format!("kvmem write stats (layer {il}): {e}"))?;
            if timed {
                t_write += t.elapsed();
            }
        }
        st.kv
            .register_append(n_rows as u32)
            .map_err(|e| format!("kvmem register append: {e}"))?;
        st.fed = (pos0 + n_rows) as u32;
        // Query-span ring: append this step's tapped last Q row per layer so the
        // next selection round can score blocks against a span of recent query
        // rows. No-op at query_span == 1 (the mean-K path reads the tap directly
        // at selection time), keeping the default byte-neutral.
        if self.query_span > 1 {
            let span = self.query_span;
            let qw = self.q_width;
            let cursor = st.q_ring_cursor;
            for il in 0..n_fa {
                taps.read_q_slot(gpu, il, slot, &mut self.q_row)
                    .map_err(|e| format!("kvmem q tap read (layer {il}): {e:?}"))?;
                let dst = il * span * qw + cursor * qw;
                st.q_ring[dst..dst + qw].copy_from_slice(&self.q_row);
            }
            st.q_ring_cursor = (cursor + 1) % span;
            st.q_ring_len = (st.q_ring_len + 1).min(span);
        }
        // Layerwise write-through (design §3.7): seal this feed's packed Q8 K/V
        // into the host archive so every fed token's payload survives for
        // drop-evict + COW restore (D4). Legacy skips this entirely — the arena
        // params are unused on the default path, keeping it byte-neutral.
        if self.strategy.is_layerwise() {
            self.write_through_archive(gpu, pool, k_arenas, v_arenas, slot, pos0, n_rows)?;
        }
        if trace {
            eprintln!(
                "[kvmem] slot {slot} feed rows {pos0}..{} ({n_rows})",
                pos0 + n_rows
            );
        }
        if timed {
            eprintln!(
                "[kvmem-time] slot {slot} feed {n_rows} rows @{pos0}: \
                 total {:.1} ms (tap read {:.1} ms, host stats write {:.1} ms)",
                t_all.elapsed().as_secs_f64() * 1e3,
                t_read.as_secs_f64() * 1e3,
                t_write.as_secs_f64() * 1e3,
            );
        }
        Ok(())
    }

    /// Layerwise write-through (design §3.7). Gather this feed's packed Q8 K/V
    /// rows out of the GPU arenas into the contiguous staging buffers and seal
    /// them into the slot's host archive (`write_layer_k_gpu`/`v_gpu`), so EVERY
    /// fed token's payload survives — not just the spilled blocks the legacy
    /// write-back path stores. The sealed archive is the source of truth for
    /// drop-evict + COW restore (D4).
    ///
    /// The fed range is contiguous at the frontier, so its logical pages are
    /// resident (never spilled): `physical()` returning `None` is a hard error,
    /// not a silent skip. The core `write_layer_k_gpu` is token-granular (it
    /// splits the range into blocks itself), so one contiguous gather + one
    /// seal call per layer suffices even when the feed spans several pages or
    /// ends on a partial tail. Only called when `strategy == Layerwise`.
    ///
    /// D2 gathers synchronously (matching the legacy `spill_page` D2H path) for
    /// correctness; K4 batches all layers through the pinned slab with one
    /// stream-ordered wait per feed instead of 32 blocking round-trips.
    pub(super) fn write_through_archive(
        &mut self,
        gpu: &Gpu,
        pool: &SlotPool,
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        slot: usize,
        pos0: usize,
        n_rows: usize,
    ) -> Result<(), String> {
        debug_assert_eq!(k_arenas.len(), self.n_fa_layers);
        debug_assert_eq!(v_arenas.len(), self.n_fa_layers);
        let bt = pool
            .block_table(SlotId(slot))
            .ok_or_else(|| format!("kvmem write-through: slot {slot} has no block table"))?;
        let kpb = self.k_page_bytes;
        let vpb = self.v_page_bytes;
        let krow = kpb / KV_BLOCK_TOKENS;
        let vrow = vpb / KV_BLOCK_TOKENS;
        let first_lp = pos0 / KV_BLOCK_TOKENS;
        let last_lp = (pos0 + n_rows - 1) / KV_BLOCK_TOKENS;
        // Resolve every fed page's physical index up front so a non-resident
        // page fails closed before any D2H or archive mutation.
        let mut phys_pages = Vec::with_capacity(last_lp - first_lp + 1);
        for lp in first_lp..=last_lp {
            let phys = bt.physical(lp).ok_or_else(|| {
                format!(
                    "kvmem write-through: slot {slot} logical page {lp} is not resident \
                     (spilled or out of range) at feed position {pos0}"
                )
            })?;
            phys_pages.push(phys);
        }
        let layer_k = n_rows * krow;
        let layer_v = n_rows * vrow;
        let need_k = self.n_fa_layers * layer_k;
        let need_v = self.n_fa_layers * layer_v;
        self.ensure_xfer_slab(gpu, need_k + need_v)?;
        let stream = gpu.steady_copy_stream();
        let gather = (|| -> Result<(), String> {
            let slab = self
                .xfer_slab
                .as_mut()
                .expect("xfer slab after ensure")
                .as_bytes_mut();
            for il in 0..self.n_fa_layers {
                for (i, lp) in (first_lp..=last_lp).enumerate() {
                    let phys = phys_pages[i] as usize;
                    let row_lo = pos0.max(lp * KV_BLOCK_TOKENS);
                    let row_hi = (pos0 + n_rows).min((lp + 1) * KV_BLOCK_TOKENS);
                    let cnt = row_hi - row_lo;
                    let intra = row_lo - lp * KV_BLOCK_TOKENS;
                    let dst = il * layer_k + (row_lo - pos0) * krow;
                    let src = phys * kpb + intra * krow;
                    gpu.hip
                        .memcpy_dtoh_async_at(
                            &mut slab[dst..dst + cnt * krow],
                            &k_arenas[il].buf,
                            src,
                            stream,
                        )
                        .map_err(|e| format!("kvmem write-through K gather (layer {il}): {e:?}"))?;
                }
                for (i, lp) in (first_lp..=last_lp).enumerate() {
                    let phys = phys_pages[i] as usize;
                    let row_lo = pos0.max(lp * KV_BLOCK_TOKENS);
                    let row_hi = (pos0 + n_rows).min((lp + 1) * KV_BLOCK_TOKENS);
                    let cnt = row_hi - row_lo;
                    let intra = row_lo - lp * KV_BLOCK_TOKENS;
                    let dst = need_k + il * layer_v + (row_lo - pos0) * vrow;
                    let src = phys * vpb + intra * vrow;
                    gpu.hip
                        .memcpy_dtoh_async_at(
                            &mut slab[dst..dst + cnt * vrow],
                            &v_arenas[il].buf,
                            src,
                            stream,
                        )
                        .map_err(|e| format!("kvmem write-through V gather (layer {il}): {e:?}"))?;
                }
            }
            Ok(())
        })();
        if let Err(e) = gather {
            // Drain any copies already issued before the error; the slab must
            // not be refilled or dropped with an in-flight D2H.
            let _ = match stream {
                Some(s) => gpu.hip.stream_synchronize(s),
                None => gpu.hip.stream_synchronize_default(),
            };
            return Err(e);
        }
        match stream {
            Some(s) => gpu.hip.stream_synchronize(s),
            None => gpu.hip.stream_synchronize_default(),
        }
        .map_err(|e| format!("kvmem write-through sync: {e:?}"))?;
        let slab = self
            .xfer_slab
            .as_mut()
            .expect("xfer slab after ensure")
            .as_bytes_mut();
        for il in 0..self.n_fa_layers {
            let k0 = il * layer_k;
            self.slots[slot]
                .raw
                .write_layer_k_gpu(
                    pos0 as u32,
                    n_rows as u32,
                    il as u32,
                    &slab[k0..k0 + layer_k],
                )
                .map_err(|e| format!("kvmem write-through K seal (layer {il}): {e}"))?;
            let v0 = need_k + il * layer_v;
            self.slots[slot]
                .raw
                .write_layer_v_gpu(
                    pos0 as u32,
                    n_rows as u32,
                    il as u32,
                    &slab[v0..v0 + layer_v],
                )
                .map_err(|e| format!("kvmem write-through V seal (layer {il}): {e}"))?;
        }
        Ok(())
    }

    /// Refresh the retrieval query for `slot` and score its resident+spilled
    /// blocks. At `query_span == 1` each layer contributes its tapped last Q row
    /// (post-RMSNorm, pre-RoPE, one token) — the frame
    /// `mean_key_retrieval_scores` expects. At `query_span > 1` the per-slot host
    /// ring supplies a span of recent Q rows scored by `page_softmax_scores`.
    pub(super) fn refresh_scores(
        &mut self,
        gpu: &Gpu,
        taps: &KvMemTaps,
        slot: usize,
        pages: usize,
    ) -> Result<Vec<f64>, String> {
        // Query span > 1: score blocks by page-softmax over the ring's recent Q
        // rows instead of the single-row mean-K cosine. The span == 1 path below
        // is untouched, so the default stays byte-identical.
        if self.query_span > 1 {
            let span = self.query_span;
            let qw = self.q_width;
            let n_fa = self.n_fa_layers;
            let st = &self.slots[slot];
            let len = st.q_ring_len;
            if len == 0 {
                return Err(format!(
                    "kvmem slot {slot}: query span empty at selection (no fed Q rows)"
                ));
            }
            let cursor = st.q_ring_cursor;
            // Extract the ring's valid rows in chronological order, layer-major
            // ([layer × row × q_width]) — the PageQueryView data layout.
            for il in 0..n_fa {
                for t in 0..len {
                    let src = (cursor + span - len + t) % span;
                    let s0 = il * span * qw + src * qw;
                    let d0 = il * len * qw + t * qw;
                    self.q_span[d0..d0 + qw].copy_from_slice(&st.q_ring[s0..s0 + qw]);
                }
                self.q_layer_rows[il] = len as u32;
            }
            let accepted = st.fed as u64;
            let sink = self.kv_cfg.sink_blocks;
            let recent = self.kv_cfg.recent_blocks;
            let budget = self.budget_blocks as u32;
            let scores = page_softmax_scores(
                &st.raw,
                pages as u32,
                accepted,
                KV_BLOCK_TOKENS as u32,
                budget,
                sink,
                recent,
                &self.q_span[..self.n_record_layers * len * qw],
                &self.q_layer_rows,
                self.n_record_layers as u32,
                len as u32,
                self.n_heads as u32,
                self.n_kv_heads as u32,
                self.head_dim as u32,
                0.0,
                GqaAggregation::Mean,
            )
            .map_err(|e| format!("kvmem page-softmax scores: {e}"))?;
            if self.diagnostics.trace {
                let sum: f64 = scores.iter().sum();
                let top = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                eprintln!(
                    "[kvmem] slot {slot} query-span: rows={len} pages={pages} \
                     score_sum={sum:.4} top={top:.4}"
                );
            }
            return Ok(scores);
        }
        for il in 0..self.n_fa_layers {
            taps.read_q_slot(gpu, il, slot, &mut self.q_row)
                .map_err(|e| format!("kvmem q tap read (layer {il}): {e:?}"))?;
            self.sums[il * self.q_width..(il + 1) * self.q_width].copy_from_slice(&self.q_row);
        }
        mean_key_retrieval_scores(
            &self.slots[slot].raw,
            pages as u32,
            &self.sums,
            &self.counts,
            self.n_record_layers as u32,
            self.head_dim as u32,
            self.n_heads as u32,
            self.n_kv_heads as u32,
            GqaAggregation::Mean,
        )
        .map_err(|e| format!("kvmem retrieval scores: {e}"))
    }

    /// Keep each over-budget slot's working set resident: spill unselected
    /// full pages, restore selected spilled pages. Must run before the step's
    /// page provisioning so a restore sees every spill's freed page.
    ///
    /// `prefill_final[s]` marks the prefill chunk that completes the prompt.
    /// That chunk uses the *retrieval* selection (scored against the Q row of
    /// the chunk just before it) instead of the sink+newest pressure policy,
    /// so a page holding an answer buried in the prompt is back in VRAM for
    /// the forward whose logits produce the first sampled token.
    ///
    /// `resident[s]` marks slots whose host state belongs to a session kept
    /// resident for continuation reuse even though no request is in flight.
    /// Their state (statistics, spilled payloads, selection watermark) is the
    /// reused prefix and must survive idle steps; only a slot with neither an
    /// in-flight request nor a resident session is dropped.
    /// G3 cold-reuse fix: materialize a page-less imported prefix (restart
    /// pack load or a demoted/restart entry) into fresh pool pages BEFORE the
    /// suffix prefill. Without this the block table holds no prefix pages and
    /// the suffix prefill's attention reads unbound/stale rows — the model
    /// loses the whole imported context. Returns the physical pages (the
    /// caller pins them as cache refs so the entry becomes hot).
    pub fn stage_page_less_import(
        &mut self,
        gpu: &Gpu,
        pool: &mut SlotPool,
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        slot: usize,
        frontier: u32,
    ) -> Result<Vec<u32>, String> {
        if frontier == 0 || frontier % KV_BLOCK_TOKENS as u32 != 0 {
            return Err(format!(
                "kvmem stage: frontier {frontier} is not page aligned"
            ));
        }
        let pages = (frontier / KV_BLOCK_TOKENS as u32) as usize;
        pool.set_seq_len(SlotId(slot), frontier as usize)
            .map_err(|e| format!("kvmem stage provision: {e}"))?;
        let phys_pages: Vec<u32> = {
            let bt = pool
                .block_table(SlotId(slot))
                .ok_or_else(|| "kvmem stage: slot has no block table".to_string())?;
            if bt.num_pages() < pages {
                return Err(format!(
                    "kvmem stage: block table holds {} pages, need {pages}",
                    bt.num_pages()
                ));
            }
            (0..pages)
                .map(|lp| {
                    bt.physical(lp)
                        .ok_or_else(|| format!("kvmem stage: logical page {lp} unbound"))
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        // K5: use the same chunked pinned-slab restore as spill/restore
        // selection instead of one blocking D2H+H2D round-trip per page. The
        // batch path validates the host payload, scatters with stream-ordered
        // async H2D and drains the whole chunk before the suffix prefill.
        let batch: Vec<(usize, usize, u32)> = phys_pages
            .iter()
            .enumerate()
            .map(|(lp, &phys)| (slot, lp, phys))
            .collect();
        let mut t_restore = std::time::Duration::ZERO;
        let mut n_restored = 0usize;
        self.flush_restore_batch(
            gpu,
            &batch,
            k_arenas,
            v_arenas,
            false,
            &mut t_restore,
            &mut n_restored,
        )?;
        Ok(phys_pages)
    }
}
