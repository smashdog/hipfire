// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Stable serve-engine seam for optional KVMem orchestration. The generic
// engine calls these hooks unconditionally; feature builds delegate to the
// KVMem implementation, while stock builds compile to explicit no-ops.

use super::{convo_group, EngineConfig, InFlight, ModelRig, Rig, SpecDraftRows};
use hipfire_runtime::scheduler::{PendingWork, Scheduler};
use hipfire_runtime::serve::{EngineStats, SubmitRequest};
use hipfire_runtime::session_table::SessionId;
use hipfire_runtime::slot_batch::SlotBatch;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::page_pool::PAGE_TOKENS;
use rdna_compute::slot_pool::{SlotId, SlotPool};
use rdna_compute::Gpu;
use std::sync::{Arc, Mutex};

use crate::qwen35::{PrefillBatchScratch, Qwen35Config};

#[cfg(feature = "kvmem")]
use super::kvmem_engine;

pub(super) fn kvmem_requested() -> Result<bool, String> {
    #[cfg(feature = "kvmem")]
    {
        Ok(crate::kvmem_ctl::requested())
    }
    #[cfg(not(feature = "kvmem"))]
    {
        if let Some(config) =
            hipfire_config::active_process_config().and_then(|process| process.kvmem.as_ref())
        {
            return if config.enabled {
                Err("KVMem requires a worker built with the kvmem feature".into())
            } else {
                Ok(false)
            };
        }
        match hipfire_config::developer_var("HIPFIRE_KVMEM")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .as_deref()
        {
            None | Some("0") | Some("off") | Some("false") => Ok(false),
            Some(other) => Err(format!(
                "HIPFIRE_KVMEM={other} needs a binary built with the `kvmem` \
                 feature (cargo build --features kvmem)"
            )),
        }
    }
}

pub(super) fn kvmem_enabled(rig: &Rig) -> bool {
    #[cfg(feature = "kvmem")]
    {
        rig.model.kvmem.is_some()
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = rig;
        false
    }
}

pub(super) fn kvmem_legacy_enabled(rig: &Rig) -> bool {
    #[cfg(feature = "kvmem")]
    {
        rig.model
            .kvmem
            .as_ref()
            .is_some_and(|ctl| !ctl.strategy().is_layerwise())
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = rig;
        false
    }
}

pub(super) fn kvmem_dflash_validation_error(
    kvmem_on: bool,
    dflash_draft: bool,
    mtp_k: usize,
    n_slots: usize,
) -> Option<String> {
    #[cfg(feature = "kvmem")]
    {
        kvmem_engine::kvmem_dflash_validation_error(kvmem_on, dflash_draft, mtp_k, n_slots)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (kvmem_on, dflash_draft, mtp_k, n_slots);
        None
    }
}

pub(super) fn legacy_continuation_matches(accepted: &[u32], rendered: &[u32]) -> bool {
    #[cfg(feature = "kvmem")]
    {
        kvmem_engine::legacy_continuation_matches(accepted, rendered)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        !accepted.is_empty() && rendered.starts_with(accepted)
    }
}

pub(super) fn trace_continuation_divergence(
    tokenizer: &Tokenizer,
    label: &str,
    accepted: &[u32],
    rendered: &[u32],
) {
    #[cfg(feature = "kvmem")]
    {
        kvmem_engine::trace_continuation_divergence(tokenizer, label, accepted, rendered);
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (tokenizer, label, accepted, rendered);
    }
}

pub(super) struct KvMemBuildState {
    pub prefix_pin: bool,
    pub prefix_coverage: bool,
    pub prefix_coverage_interval: usize,
    pub checkpoint_policy: Option<(u64, u64)>,
    #[cfg(feature = "kvmem")]
    pub plan: Option<crate::kvmem_ctl::KvMemPlan>,
    #[cfg(feature = "kvmem")]
    pub options: Option<crate::kvmem_ctl::KvMemOptions>,
}

impl KvMemBuildState {
    pub(super) fn pages_total(&self) -> Option<usize> {
        #[cfg(feature = "kvmem")]
        {
            self.plan.as_ref().map(|plan| plan.pages_total)
        }
        #[cfg(not(feature = "kvmem"))]
        {
            None
        }
    }
}

pub(super) fn resolve_build(
    cfg: &EngineConfig,
    kvmem_on: bool,
    prefill_chunk: usize,
    spec_rows: usize,
) -> Result<KvMemBuildState, String> {
    #[cfg(feature = "kvmem")]
    {
        let boot = kvmem_engine::resolve_prefix_boot(cfg, kvmem_on, prefill_chunk, spec_rows)?;
        Ok(KvMemBuildState {
            prefix_pin: boot.prefix_pin,
            prefix_coverage: boot.prefix_coverage,
            prefix_coverage_interval: boot.prefix_coverage_interval,
            checkpoint_policy: boot.checkpoint_policy,
            plan: boot.plan,
            options: boot.options,
        })
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (cfg, kvmem_on, prefill_chunk, spec_rows);
        Ok(KvMemBuildState {
            prefix_pin: false,
            prefix_coverage: false,
            prefix_coverage_interval: 1024,
            checkpoint_policy: None,
        })
    }
}

pub(super) fn prepare_step(
    rig: &mut Rig,
    slots: &[Option<InFlight>],
    work: &[PendingWork],
    batch: &SlotBatch,
    n: usize,
) -> Result<(), String> {
    #[cfg(feature = "kvmem")]
    {
        kvmem_engine::prepare_kvmem_step(rig, slots, work, batch, n)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (rig, slots, work, batch, n);
        Ok(())
    }
}

pub(super) fn feed_step(
    rig: &mut Rig,
    batch: &SlotBatch,
    slots: &[Option<InFlight>],
    work: &[PendingWork],
    spec_drafts: &[Option<SpecDraftRows>],
    n: usize,
) -> Result<(), String> {
    #[cfg(feature = "kvmem")]
    {
        kvmem_engine::kvmem_feed_step(rig, batch, slots, work, spec_drafts, n)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (rig, batch, slots, work, spec_drafts, n);
        Ok(())
    }
}

pub(super) fn reset_slot(rig: &mut Rig, slot: SlotId, req: &SubmitRequest) -> Result<(), String> {
    #[cfg(feature = "kvmem")]
    {
        kvmem_engine::reset_kvmem_slot(rig, slot, req).map(|_| ())
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (rig, slot, req);
        Ok(())
    }
}

pub(super) fn prefix_reuse(
    rig: &mut Rig,
    slot: SlotId,
    id: SessionId,
    req: &SubmitRequest,
) -> Option<usize> {
    #[cfg(feature = "kvmem")]
    {
        kvmem_engine::kvmem_prefix_reuse(rig, slot, id, req)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (rig, slot, id, req);
        None
    }
}

pub(super) fn filter_continuation(rig: &Rig, matched: Option<SessionId>) -> Option<SessionId> {
    #[cfg(feature = "kvmem")]
    {
        matched.filter(|existing| {
            if !super::kvmem_enabled(rig) {
                return true;
            }
            let Some(slot) = rig.sessions.get(*existing).and_then(|s| s.slot) else {
                return false;
            };
            let held = rig
                .pool
                .descriptors()
                .get(slot.0)
                .map(|d| d.seq_len.max(0) as u32);
            match (rig.model.kvmem.as_ref(), held) {
                (Some(ctl), Some(held)) => ctl.fed_tokens(slot.0) == Some(held),
                _ => false,
            }
        })
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = rig;
        matched
    }
}

pub(super) fn ensure_host_headroom(rig: &Rig, additional_tokens: usize) -> Result<(), String> {
    #[cfg(feature = "kvmem")]
    {
        if let Some(ctl) = rig.model.kvmem.as_ref() {
            ctl.ensure_host_headroom(additional_tokens)?;
        }
    }
    let _ = (rig, additional_tokens);
    Ok(())
}

/// Stable classifier for the host-tier capacity fault. The admission check
/// should make this unreachable; it remains the last line of defence so a
/// charge admitted by a concurrent slot cannot tear down every request.
pub(super) fn is_host_capacity_error(reason: &str) -> bool {
    reason.contains("kvmem host capacity:") || reason.contains("native KV RAM capacity exhausted")
}

/// Drop every slot's host state after a capacity fault and let the serve loop
/// continue. The caller has already run `fail_all_active`, so no session or
/// work item references these slots; `reset_slot` releases the raw stores and
/// bumps the Layerwise incarnation, so the next occupant cannot alias the
/// partially-written session.
pub(super) fn recover_host_capacity(rig: &mut Rig) -> bool {
    #[cfg(feature = "kvmem")]
    {
        let Some(ctl) = rig.model.kvmem.as_mut() else {
            return false;
        };
        for slot in 0..ctl.slot_count() {
            if let Err(e) = ctl.reset_slot(slot) {
                eprintln!("[kvmem] host-capacity reset slot={slot} failed: {e}");
                return false;
            }
        }
        return true;
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = rig;
        false
    }
}

/// Continuation sibling: the retained prefix is already charged to the host
/// tier, so only the new suffix plus the generation budget is additional.
pub(super) fn ensure_continuation_host_headroom(
    rig: &Rig,
    slot: SlotId,
    extended_tokens: usize,
    max_tokens: usize,
) -> Result<(), String> {
    #[cfg(feature = "kvmem")]
    {
        if let Some(ctl) = rig.model.kvmem.as_ref() {
            let already = ctl.fed_tokens(slot.0).unwrap_or(0) as usize;
            let additional = extended_tokens
                .saturating_sub(already)
                .saturating_add(max_tokens.max(1));
            ctl.ensure_host_headroom(additional)?;
        }
    }
    let _ = (rig, slot, extended_tokens, max_tokens);
    Ok(())
}

pub(super) fn begin_continuation_request(
    rig: &mut Rig,
    slot: SlotId,
    existing: SessionId,
    req: &SubmitRequest,
    reused: usize,
    extended: &[u32],
) -> Result<(), String> {
    #[cfg(feature = "kvmem")]
    {
        if !super::kvmem_enabled(rig) {
            return Ok(());
        }
        // A continuation resuming below the stored frontier truncates only
        // when a DeltaNet checkpoint at exactly that frontier pairs with it.
        if rig.kvmem_prefix_pin {
            let fed_now = rig
                .model
                .kvmem
                .as_ref()
                .and_then(|ctl| ctl.fed_tokens(slot.0))
                .unwrap_or(0);
            if (reused as u32) < fed_now {
                let truncated = (|| -> Result<(), String> {
                    let domain = rig
                        .cache_domain
                        .as_ref()
                        .ok_or_else(|| "kvmem truncate: no cache domain".to_string())?;
                    let ckpt_pool = rig
                        .model
                        .checkpoint_pool
                        .as_mut()
                        .ok_or_else(|| "kvmem truncate: no checkpoint pool".to_string())?;
                    let media = rig.sessions.get(existing).and_then(|s| s.media_digest);
                    let fp = crate::checkpoint::checkpoint_fingerprint(&extended[..reused], media);
                    let restored = ckpt_pool
                        .peek(domain, reused as u64, fp)
                        .map(|snapshot| {
                            snapshot
                                .restore_to(&mut rig.model.dn_states[slot.0], &mut rig.gpu)
                                .is_ok()
                        })
                        .unwrap_or(false);
                    if !restored {
                        return Err("no DeltaNet checkpoint at the truncation frontier".to_string());
                    }
                    let ctl = rig
                        .model
                        .kvmem
                        .as_mut()
                        .ok_or_else(|| "kvmem truncate: no controller".to_string())?;
                    ctl.truncate_slot(slot.0, reused as u32)
                })();
                if let Err(e) = truncated {
                    let _ = rig.model.dn_states[slot.0].reset(&mut rig.gpu);
                    if rig.gpu.slot_trace() {
                        eprintln!("[kvmem-prefix] continuation truncate refused: {e}");
                    }
                }
            }
        }
        let _ = rig.model.kvmem.as_mut().map(|ctl| {
            ctl.set_request_query(slot.0, req.query)
                .and_then(|_| ctl.set_request_group(slot.0, convo_group(&req.convo)))
        });
        let fed = rig
            .model
            .kvmem
            .as_mut()
            .map(|ctl| ctl.begin_request(slot.0))
            .transpose();
        match fed {
            Ok(Some(n)) if n as usize == reused => Ok(()),
            Ok(Some(n)) => Err(format!(
                "kvmem store holds {n} tokens but the continuation resumes at {reused}"
            )),
            Ok(None) => Err("kvmem controller missing".to_string()),
            Err(e) => Err(e),
        }
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (rig, slot, existing, req, reused, extended);
        Ok(())
    }
}

pub(super) fn page_demand_allowed(rig: &Rig) -> bool {
    #[cfg(feature = "kvmem")]
    {
        !super::kvmem_enabled(rig)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = rig;
        true
    }
}

pub(super) fn evict_for_admit(
    rig: &mut Rig,
    victim: SessionId,
    stats: &Arc<Mutex<EngineStats>>,
) -> bool {
    #[cfg(feature = "kvmem")]
    {
        if super::kvmem_enabled(rig) {
            if let Err(e) = kvmem_engine::park_legacy_history(rig, victim) {
                eprintln!(
                    "[kvmem-legacy] HistoryPark session={} failed: {e}",
                    victim.0
                );
            }
            rig.swap.forget(victim.0);
            rig.sessions.close(&mut rig.pool, &mut rig.adm, victim);
            true
        } else {
            super::serve_engine_admit::evict(rig, victim, stats)
        }
    }
    #[cfg(not(feature = "kvmem"))]
    {
        super::serve_engine_admit::evict(rig, victim, stats)
    }
}

pub(super) fn head_plane_enabled(model: &ModelRig) -> bool {
    #[cfg(feature = "kvmem")]
    {
        model
            .kvmem
            .as_ref()
            .map(|ctl| ctl.head_plane_enabled())
            .unwrap_or(false)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = model;
        false
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn accept_mtp_verify(
    model: &mut ModelRig,
    gpu: &mut Gpu,
    pool: &mut SlotPool,
    slot: SlotId,
    pos: usize,
    advance: usize,
    hidden_row_offset: usize,
    draft: &crate::mtp_spec::MtpDraftOutput,
    state: &mut crate::mtp_spec::MtpSpecState,
    head_consume: bool,
) -> Result<(), String> {
    #[cfg(feature = "kvmem")]
    {
        let mut kvmem = model.kvmem.take();
        if let Some(ctl) = kvmem.as_mut() {
            let taps = model
                .pbs
                .kvmem_taps
                .as_ref()
                .expect("kvmem controller without statistics taps");
            ctl.feed_rows(
                gpu,
                taps,
                slot.0,
                pos,
                advance,
                hidden_row_offset,
                pool,
                &model.k_arenas,
                &model.v_arenas,
            )
            .map_err(|e| format!("kvmem statistics feed: {e}"))?;
            if ctl.strategy().is_layerwise() {
                if let Some(snap) = ctl.take_layerwise_pending_output(slot.0) {
                    let mut owner =
                        ctl.take_layerwise_output_pressure(slot.0).ok_or_else(|| {
                            "kvmem: layerwise verify pending without output owner".to_string()
                        })?;
                    let pub_res = crate::layerwise_driver::publish_output_acceptance(
                        &mut owner,
                        snap,
                        advance as u32,
                    );
                    ctl.restore_layerwise_output_pressure(slot.0, owner);
                    pub_res.map_err(|e| format!("kvmem: layerwise output publish: {e}"))?;
                }
            }
            if head_consume && advance > 0 {
                let head = model
                    .mtp_head
                    .as_ref()
                    .ok_or_else(|| "mtp head consume: MTP head not loaded".to_string())?;
                let rope_shift = ctl.head_prefill_shift(pos + advance);
                let tokens = crate::mtp_spec::mtp_takeover_fill_tokens(
                    draft.last_committed,
                    &draft.candidates,
                    advance,
                );
                crate::mtp_spec::mtp_takeover_fill_head_kv(
                    gpu,
                    head,
                    state,
                    &model.weights,
                    pos,
                    &tokens,
                    rope_shift,
                )
                .map_err(|e| format!("mtp head consume fill: {e}"))?;
                ctl.head_write_through_archive(
                    gpu,
                    &state.mtp_kv.inner.k_gpu[0],
                    &state.mtp_kv.inner.v_gpu[0],
                    slot.0,
                    pos,
                    advance,
                    pos - rope_shift,
                )
                .map_err(|e| format!("mtp head consume archive: {e}"))?;
                ctl.accept_head_append(slot.0, pos as u64, (pos + advance) as u64)
                    .map_err(|e| format!("mtp head consume append: {e}"))?;
            }
        }
        model.kvmem = kvmem;
        Ok(())
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (
            model,
            gpu,
            pool,
            slot,
            pos,
            advance,
            hidden_row_offset,
            draft,
            state,
            head_consume,
        );
        Ok(())
    }
}

pub(super) fn accept_dflash_verify(
    model: &mut ModelRig,
    gpu: &mut Gpu,
    pool: &mut SlotPool,
    slot: SlotId,
    pos: usize,
    committed: usize,
    hidden_row_offset: usize,
) -> Result<(), String> {
    #[cfg(feature = "kvmem")]
    {
        if let Some(ctl) = model.kvmem.as_mut() {
            let taps = model
                .pbs
                .kvmem_taps
                .as_ref()
                .expect("kvmem controller without statistics taps");
            ctl.feed_rows(
                gpu,
                taps,
                slot.0,
                pos,
                committed,
                hidden_row_offset,
                pool,
                &model.k_arenas,
                &model.v_arenas,
            )
            .map_err(|e| format!("kvmem statistics feed: {e}"))?;
            if ctl.strategy().is_layerwise() {
                if let Some(snap) = ctl.take_layerwise_pending_output(slot.0) {
                    let mut owner =
                        ctl.take_layerwise_output_pressure(slot.0).ok_or_else(|| {
                            "kvmem: layerwise verify pending without output owner".to_string()
                        })?;
                    let pub_res = crate::layerwise_driver::publish_output_acceptance(
                        &mut owner,
                        snap,
                        committed as u32,
                    );
                    ctl.restore_layerwise_output_pressure(slot.0, owner);
                    pub_res.map_err(|e| format!("kvmem: layerwise output publish: {e}"))?;
                }
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (model, gpu, pool, slot, pos, committed, hidden_row_offset);
        Ok(())
    }
}

pub(super) fn mtp_head_capacity(model: &ModelRig) -> Option<usize> {
    #[cfg(feature = "kvmem")]
    {
        model
            .kvmem
            .as_ref()
            .and_then(|ctl| ctl.head_bounded_capacity())
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = model;
        None
    }
}

pub(super) fn mtp_head_prefill_shift(model: &ModelRig, end: usize) -> usize {
    #[cfg(feature = "kvmem")]
    {
        model
            .kvmem
            .as_ref()
            .map(|ctl| ctl.head_prefill_shift(end))
            .unwrap_or(0)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (model, end);
        0
    }
}

pub(super) fn dflash_context_floor(model: &ModelRig, slot: SlotId, pos: usize) -> usize {
    #[cfg(feature = "kvmem")]
    {
        model
            .kvmem
            .as_ref()
            .map(|k| k.draft_conditioning_floor(slot.0, pos))
            .unwrap_or(0)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (model, slot, pos);
        0
    }
}

pub(super) fn layerwise_reuse(rig: &Rig) -> bool {
    #[cfg(feature = "kvmem")]
    {
        rig.kvmem_prefix_pin
            && rig
                .model
                .kvmem
                .as_ref()
                .is_some_and(|ctl| ctl.strategy().is_layerwise())
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = rig;
        false
    }
}

pub(super) fn prepare_schedule(
    rig: &mut Rig,
    slots: &[Option<InFlight>],
    work: &[PendingWork],
) -> Result<(Vec<usize>, bool), String> {
    #[cfg(feature = "kvmem")]
    {
        let caps = kvmem_engine::prepare_legacy_query_checkpoints(rig, slots, work)?;
        let legacy_contract = rig
            .model
            .kvmem
            .as_ref()
            .is_some_and(|ctl| ctl.legacy_config().is_some());
        Ok((caps, legacy_contract))
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (rig, slots);
        Ok((Vec::new(), false))
    }
}

pub(super) fn schedule_batch(
    sched: &mut Scheduler,
    work: &mut [PendingWork],
    remaining_rows: usize,
    prefill_min_tokens: usize,
    eligible: &[bool],
    hold_tail: &[bool],
    legacy_query_caps: &[usize],
) -> SlotBatch {
    #[cfg(feature = "kvmem")]
    {
        sched.next_batch_eligible_capped(
            work,
            remaining_rows,
            prefill_min_tokens,
            eligible,
            hold_tail,
            legacy_query_caps,
        )
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (hold_tail, legacy_query_caps);
        sched.next_batch_eligible(work, remaining_rows, prefill_min_tokens, eligible)
    }
}

pub(super) fn hold_tail_mask(rig: &Rig, work: &[PendingWork]) -> Vec<bool> {
    #[cfg(feature = "kvmem")]
    {
        match rig.model.kvmem.as_ref() {
            Some(ctl) => {
                let budget = ctl.budget_blocks();
                work.iter()
                    .map(|w| {
                        let left = w.remaining_prompt.len();
                        !w.decoding
                            && left > 1
                            && (w.next_pos + left).div_ceil(PAGE_TOKENS) > budget
                    })
                    .collect()
            }
            None => Vec::new(),
        }
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (rig, work);
        Vec::new()
    }
}

pub(super) fn enable_taps(
    pbs: &mut PrefillBatchScratch,
    gpu: &mut Gpu,
    config: &Qwen35Config,
    cfg: &EngineConfig,
    max_batch: usize,
) -> Result<(), String> {
    #[cfg(feature = "kvmem")]
    {
        kvmem_engine::enable_kvmem_taps(pbs, gpu, config, cfg, max_batch)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (pbs, gpu, config, cfg, max_batch);
        Ok(())
    }
}

pub(super) fn note_stats(rig: &Rig, stats: &mut EngineStats) {
    #[cfg(feature = "kvmem")]
    {
        if let Some(kvmem) = rig.model.kvmem.as_ref() {
            stats.note_kvmem(kvmem.stats());
        }
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (rig, stats);
    }
}

pub(super) fn coverage_capture(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &[PendingWork],
) {
    #[cfg(feature = "kvmem")]
    {
        kvmem_engine::kvmem_coverage_capture(rig, slots, work);
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = (rig, slots, work);
    }
}

pub(super) fn forward_time_enabled(rig: &Rig) -> bool {
    #[cfg(feature = "kvmem")]
    {
        rig.model
            .kvmem
            .as_ref()
            .is_some_and(|ctl| ctl.diagnostics().time)
    }
    #[cfg(not(feature = "kvmem"))]
    {
        let _ = rig;
        false
    }
}
