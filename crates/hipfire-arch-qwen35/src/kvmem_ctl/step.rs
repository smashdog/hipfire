// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;

impl KvMemCtl {
    pub fn prepare_step(
        &mut self,
        gpu: &mut Gpu,
        pool: &mut SlotPool,
        taps: &KvMemTaps,
        active: &[bool],
        resident: &[bool],
        prefill: &[bool],
        prefill_final: &[bool],
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
    ) -> Result<(), String> {
        debug_assert!(pool.is_paged(), "kvmem requires the paged slot pool");
        debug_assert_eq!(k_arenas.len(), self.n_fa_layers);
        debug_assert_eq!(v_arenas.len(), self.n_fa_layers);
        let n_slots = self.slots.len();
        let timed = self.diagnostics.time;
        let t_all = std::time::Instant::now();
        let mut t_select = std::time::Duration::ZERO;
        let mut t_spill = std::time::Duration::ZERO;
        let mut t_restore = std::time::Duration::ZERO;
        let mut n_spilled = 0usize;
        let mut n_restored = 0usize;

        // Return host state for slots whose session closed since last step.
        // An idle slot that still holds a resident session keeps its state:
        // it is the next turn's continuation prefix.
        for s in 0..n_slots {
            let live = active.get(s).copied().unwrap_or(false)
                || resident.get(s).copied().unwrap_or(false);
            if !live && (self.slots[s].fed != 0 || self.slots[s].last_select_full != 0) {
                self.reset_slot(s)?;
            }
        }

        // ── Pass 1: select, then spill every unselected resident full page ──
        let mut plans: Vec<Option<StepPlan>> = (0..n_slots).map(|_| None).collect();
        for s in 0..n_slots {
            if !active.get(s).copied().unwrap_or(false) {
                continue;
            }
            let prefill_step = prefill.get(s).copied().unwrap_or(false);
            let (pages, live) = match pool.block_table(SlotId(s)) {
                Some(bt) => (bt.num_pages(), bt.live_tokens()),
                None => continue,
            };
            // Watermark hysteresis: when GPU_HIGH/LOW_WATERMARK is set, spill is
            // triggered at the high-watermark page count (>= the keep target the
            // core selection budget was narrowed to), instead of the single
            // budget threshold. Unset => the historical budget_blocks trigger.
            let spill_trigger = self.watermark.map(|(t, _)| t).unwrap_or(self.budget_blocks);
            if pages <= spill_trigger {
                continue;
            }
            let full_pages = live / KV_BLOCK_TOKENS;
            // The chunk that completes the prompt gets the retrieval policy
            // too — the tokens before it are the question, and this chunk's
            // forward produces the logits the first answer token comes from.
            // A slot with no fed rows has no tapped query yet (single-chunk
            // prompt); it keeps the pressure policy.
            let final_prefill = prefill_step
                && prefill_final.get(s).copied().unwrap_or(false)
                && self.slots[s].fed > self.slots[s].turn_start;
            // The first decode step of a request takes a selection decision
            // even when the block interval has not elapsed: the prompt-
            // completing chunk just captured this turn's question in the Q
            // tap, and scoring the resident set against it is what restores
            // the page holding the answer. `fed > turn_start` rejects a
            // stale tap from the previous request (empty continuation
            // suffix: no rows of this turn exist to score with).
            let first_decode = !prefill_step
                && !self.slots[s].first_decode_done
                && self.slots[s].fed > self.slots[s].turn_start;
            if !final_prefill
                && !first_decode
                && full_pages < self.slots[s].last_select_full.saturating_add(self.interval)
            {
                continue;
            }
            let t_pick = std::time::Instant::now();
            let picked = if prefill_step && !final_prefill {
                self.slots[s]
                    .kv
                    .pick_prefill_pressure_blocks(&[])
                    .map_err(|e| format!("kvmem prefill selection: {e}"))?
            } else {
                // SELECT_METHOD gates which signal feeds the shared pick_topk
                // ranking (the core's executor-level policy). Retrieval (0, the
                // default) scores the resident set against the tapped mean-K
                // query. Recency (2) feeds no learned signal: zeroing the scores
                // makes pick_topk keep sink + recent and fill the rest by recency
                // (the core's deterministic id tie-break), and skips the Q-tap
                // read + mean-K scoring entirely. H2O (1) is rejected at
                // construction until the attention-heat tap lands.
                //
                // LRFU round lifecycle: begin before previewing scores, commit
                // after the pick so the selector ranks the decayed candidate and
                // the round's reinforcement is then sealed. Both are no-ops when
                // lrfu is off, so the default Retrieval path stays byte-neutral.
                self.slots[s]
                    .kv
                    .begin_retrieval_round()
                    .map_err(|e| format!("kvmem begin round: {e}"))?;
                if self.select_method == 2 {
                    let zeros = vec![0.0f64; pages];
                    self.slots[s]
                        .kv
                        .set_retrieval_scores(&zeros)
                        .map_err(|e| format!("kvmem set scores: {e}"))?;
                } else {
                    let scores = self.refresh_scores(gpu, taps, s, pages)?;
                    self.slots[s]
                        .kv
                        .set_retrieval_scores(&scores)
                        .map_err(|e| format!("kvmem set scores: {e}"))?;
                }
                let picked = self.slots[s]
                    .kv
                    .pick_topk_blocks(&[])
                    .map_err(|e| format!("kvmem decode selection: {e}"))?;
                self.slots[s]
                    .kv
                    .commit_retrieval_round()
                    .map_err(|e| format!("kvmem commit round: {e}"))?;
                picked
            };
            if timed {
                t_select += t_pick.elapsed();
            }
            let mut keep = vec![false; pages];
            for &id in &picked {
                if let Some(k) = keep.get_mut(id as usize) {
                    *k = true;
                }
            }
            // Collect this slot's page actions before mutating the pool.
            let (spill, restore) = {
                let bt = pool
                    .block_table(SlotId(s))
                    .expect("kvmem: block table vanished inside one step");
                debug_assert_eq!(bt.num_pages(), pages);
                let mut spill = Vec::new();
                let mut restore = Vec::new();
                for lp in 0..full_pages {
                    if keep[lp] {
                        if bt.is_spilled(lp) {
                            restore.push(lp);
                        }
                    } else if !bt.is_spilled(lp) {
                        match bt.physical(lp) {
                            Some(phys) => {
                                // P4.3 prefix PIN: a page the pool refuses to
                                // evict (shared-prefix cache refs, or any other
                                // non-exclusive lease) is PIN resident — force
                                // keep instead of spilling. `stage_out_page`
                                // would fail closed on it anyway; keeping it
                                // here turns that hard error into the intended
                                // residency. With no pinned pages every page
                                // is exclusively table-owned ⇒ byte-neutral.
                                let evictable = pool
                                    .page_pool()
                                    .map(|pp| pp.page_evictable(phys))
                                    .unwrap_or(true);
                                if evictable {
                                    spill.push((lp, phys));
                                } else {
                                    keep[lp] = true;
                                }
                            }
                            None => {
                                return Err(format!(
                                    "kvmem slot {s}: logical page {lp} is neither \
                                     resident nor spilled"
                                ))
                            }
                        }
                    }
                }
                (spill, restore)
            };
            self.flush_spill_batch(
                gpu,
                pool,
                s,
                &spill,
                k_arenas,
                v_arenas,
                timed,
                &mut t_spill,
                &mut n_spilled,
            )?;
            if self.diagnostics.trace {
                eprintln!(
                    "[kvmem] slot {s} ({}): {pages} pages, full {full_pages}, \
                     budget {}, kept {}, spilled {}, restore {}",
                    if prefill_step {
                        if final_prefill {
                            "prefill-final"
                        } else {
                            "prefill"
                        }
                    } else {
                        "decode"
                    },
                    self.budget_blocks,
                    keep.iter().filter(|k| **k).count(),
                    spill.len(),
                    restore.len(),
                );
                if self.diagnostics.trace_pages {
                    eprintln!("[kvmem] slot {s} picked {picked:?} restore {restore:?}");
                }
            }
            self.slots[s].last_select_full = full_pages;
            if !prefill_step {
                self.slots[s].first_decode_done = true;
            }
            plans[s] = Some(StepPlan { keep, restore });
            // P4.1② f2 (G5 prefill seed): seed the FirstLayer head aux selection
            // from this step's trunk working set. The first MTP draft of a request
            // runs BEFORE any decode/verify carrier restore-hook — the draft phase
            // precedes `prepare_step` in the serve loop and the prefill is
            // full-resident with no carrier — so the G5 carrier binding
            // (serve_engine decode/verify arms) has not fired yet and the draft
            // would fall back to full-resident, overflowing the bounded head cache
            // once `fed > window`. This is the hipfire analog of ninfer's
            // restore-hook `bind_auxiliary_selection` (NL:9985-9990) firing on the
            // trunk's first restored layer; the draft `take_aux_selection`s it to
            // engage the bounded head restore. The selected pages are full pages
            // strictly below the frontier, so the visible extent is invariant
            // between this step's `fed` and the later draft's `fed`. Additive +
            // gated: Legacy / AR-only (`head_plane_enabled()` false) never bind
            // (byte-neutral); zero core C++ change.
            if self.head_plane_enabled() {
                let frontier = u64::from(self.slots[s].fed);
                // The head catalog (`plan_head_plane`) spans only SEALED FULL
                // pages (`id < frontier/128`); the partial suffix page
                // (`id == frontier/128`) is the draft-scratch boundary handled
                // separately by `HeadSuffix`, NOT a canonical aux block. `picked`
                // includes that suffix, so filter it out here — mirroring
                // `restored_keep_set()` (suffix excluded), the source the G5
                // carrier binding uses. `auxiliary_visible_tokens` clips the
                // suffix to the frontier so `bind` would accept it, but
                // `plan_layer` then rejects `id == full_pages` as an "unknown
                // selected block" (out of catalog). Filtering to full pages also
                // keeps the visible extent invariant (each contributes exactly
                // one block) between this step's `fed` and the later draft's.
                let full_pages = frontier / KV_BLOCK_TOKENS as u64;
                let mut sel: Vec<u64> = picked
                    .iter()
                    .map(|&id| u64::from(id))
                    .filter(|&id| id < full_pages)
                    .collect();
                sel.sort_unstable();
                sel.dedup();
                // P4.1③ H4: branch the head-history policy. FirstLayer (the
                // auto-default) binds the trunk working set's full pages (② G5).
                // The A1 starvation floor is applied centrally in
                // `layerwise_mtp_history()` (below-floor UpstreamGlobal reports
                // FirstLayer), so this site and the serve_engine carrier hooks
                // fall back uniformly; the UG algorithm is unchanged when engaged.
                // UpstreamGlobal restores ALL canonical pages when the whole
                // history fits the budget (ninfer `begin==history_tokens` fast
                // path, NL:10058), else runs the global Mean-K probe over the
                // captured prefill Q tap (H2b — no private forward, no rollback).
                // The probe needs a fresh Q tap (`final_prefill`/`first_decode`
                // ran `refresh_scores`); an intermediate prefill step with a stale
                // tap keeps the picked seed, overwritten by the prompt-completing
                // step's probe before the first draft. Legacy / AR-only never
                // reach this block (byte-neutral); zero core C++ change.
                let aux = if self.layerwise_mtp_history() == LayerwiseMtpHistory::UpstreamGlobal {
                    if full_pages <= self.budget_blocks as u64 {
                        let all: Vec<u64> = (0..full_pages).collect();
                        crate::layerwise_mtp::bind_first_layer_auxiliary(
                            &all,
                            frontier,
                            KV_BLOCK_TOKENS as u64,
                        )
                        .map_err(|e| format!("kvmem head aux restore-all (slot {s}): {e}"))?
                    } else if final_prefill || first_decode {
                        // The probe admits against a FRESH page-aligned sealed
                        // Mean-K TopK index (the same `layerwise_policy` the trunk
                        // carrier uses), NOT the slot's shared Legacy `kv` store
                        // (which tracks all fed tokens and carries retrieval-round
                        // state — the core rejects it as a probe index). The
                        // identity frontier is the page-aligned `full_pages*128`
                        // (== policy.total_tokens); the bind clips to the native
                        // `frontier` (fed) exactly like the FirstLayer seed.
                        let probe_frontier = full_pages * KV_BLOCK_TOKENS as u64;
                        let probe_policy = self
                            .layerwise_policy(self.slots[s].fed)
                            .map_err(|e| format!("kvmem head aux probe policy (slot {s}): {e}"))?;
                        let id = crate::layerwise_probe::probe_identity(
                            probe_frontier,
                            s as u64,
                            self.slots[s].lw_incarnation,
                            self.slots[s].lw_seq,
                        );
                        let q_rows = &self.sums[..self.n_fa_layers * self.q_width];
                        crate::layerwise_probe::probe_head_history_qrows(
                            gpu,
                            q_rows,
                            &id,
                            &probe_policy,
                            &self.slots[s].raw,
                            self.n_heads as u32,
                            self.n_kv_heads as u32,
                            self.head_dim as u32,
                            0.0, // → 1/sqrt(head_dim), the trunk scorer's default scale
                            KV_BLOCK_TOKENS as u32,
                            frontier,
                        )
                        .map_err(|e| format!("kvmem head aux probe (slot {s}): {e}"))?
                    } else {
                        crate::layerwise_mtp::bind_first_layer_auxiliary(
                            &sel,
                            frontier,
                            KV_BLOCK_TOKENS as u64,
                        )
                        .map_err(|e| format!("kvmem head aux seed (slot {s}): {e}"))?
                    }
                } else {
                    crate::layerwise_mtp::bind_first_layer_auxiliary(
                        &sel,
                        frontier,
                        KV_BLOCK_TOKENS as u64,
                    )
                    .map_err(|e| format!("kvmem head aux seed (slot {s}): {e}"))?
                };
                self.set_aux_selection(s, aux);
            }
        }

        // ── Pass 2: restore the selected spilled pages ──────────────────────
        // Alloc + bind every selected page first (same order as before, so the
        // free-list interaction is unchanged), then move all the bytes in one
        // batched flush. No forward runs between bind and restore.
        let mut restore_batch: Vec<(usize, usize, u32)> = Vec::new();
        for s in 0..n_slots {
            let Some(plan) = plans[s].take() else {
                continue;
            };
            for lp in plan.restore {
                debug_assert!(plan.keep[lp]);
                let phys = pool
                    .page_pool_mut()
                    .ok_or_else(|| "kvmem restore: pool is not paged".to_string())?
                    .alloc_page_unbound()
                    .map_err(|e| format!("kvmem restore alloc: {e}"))?;
                pool.bind_restored_page(SlotId(s), lp, phys)
                    .map_err(|e| format!("kvmem restore bind: {e}"))?;
                restore_batch.push((s, lp, phys));
            }
            if self.diagnostics.verify {
                if let Some(bt) = pool.block_table(SlotId(s)) {
                    let mut seen: std::collections::HashMap<u32, usize> =
                        std::collections::HashMap::new();
                    let mut dups = Vec::new();
                    for lp in 0..bt.num_pages() {
                        if let Some(phys) = bt.physical(lp) {
                            if let Some(prev) = seen.insert(phys, lp) {
                                dups.push((prev, lp, phys));
                            }
                        }
                    }
                    let mut map = Vec::with_capacity(bt.num_pages());
                    for lp in 0..bt.num_pages() {
                        map.push(bt.raw_entry(lp).unwrap_or(u32::MAX));
                    }
                    eprintln!(
                        "[kvmem-verify] slot {s} resident {} pages, dups {dups:?} map {map:?}",
                        seen.len()
                    );
                }
            }
        }
        self.flush_restore_batch(
            gpu,
            &restore_batch,
            k_arenas,
            v_arenas,
            timed,
            &mut t_restore,
            &mut n_restored,
        )?;
        // Observability counters (pure bookkeeping; no path reads them back).
        self.spill_pages_total += n_spilled as u64;
        self.restore_pages_total += n_restored as u64;
        if timed && (n_spilled > 0 || n_restored > 0 || t_select > std::time::Duration::ZERO) {
            eprintln!(
                "[kvmem-time] prepare: total {:.1} ms (select {:.1} ms, spill {} pages \
                 {:.1} ms, restore {} pages {:.1} ms)",
                t_all.elapsed().as_secs_f64() * 1e3,
                t_select.as_secs_f64() * 1e3,
                n_spilled,
                t_spill.as_secs_f64() * 1e3,
                n_restored,
                t_restore.as_secs_f64() * 1e3,
            );
        }
        Ok(())
    }
}
