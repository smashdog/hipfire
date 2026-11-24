// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// KVMem-specific serve-engine helpers. This module is the first slice of
// P1-A: pure KVMem orchestration lives here, while generic slot admission,
// scheduling and batch execution stay in `serve_engine.rs`.

use super::EngineConfig;
use crate::qwen35::LayerType;
use hipfire_runtime::scheduler::{PendingWork, SpecKind};
use hipfire_runtime::session_table::SessionId;
use hipfire_runtime::slot_batch::SlotBatch;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::page_pool::PAGE_TOKENS;
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{Gpu, HipResult};

use super::{
    convo_group, kvmem_legacy_enabled, InFlight, ModelRig, Rig, SlotDecodeGraph, SpecDraftRows,
    SubmitRequest,
};

fn kvmem_trace(rig: &Rig) -> bool {
    rig.model
        .kvmem
        .as_ref()
        .is_some_and(|ctl| ctl.diagnostics().trace)
}

/// G7: the session's media identity + image spans over `tokens` (the frozen
/// prefix). Text-only sessions and pre-image prefixes yield `(None, vec![])`.
#[cfg(feature = "kvmem")]
pub(super) fn kvmem_session_media(rig: &Rig, session: SessionId) -> (Option<u64>, Vec<(u32, u32)>) {
    let Some(sess) = rig.sessions.get(session) else {
        return (None, Vec::new());
    };
    if !sess.has_image || sess.media_n_visual == 0 {
        return (sess.media_digest.filter(|_| sess.has_image), Vec::new());
    }
    let Some(pad) = rig.tokenizer.special_token_id("<|image_pad|>") else {
        return (sess.media_digest, Vec::new());
    };
    // Spans over the FULL session token stream: a mid-image boundary must be
    // recognized as a split even though the prefix truncates the run.
    let spans: Vec<(u32, u32)> =
        crate::kvmem_media::spans_from_pad_runs(&sess.tokens, pad, sess.media_n_visual)
            .into_iter()
            .map(|s| (s.begin as u32, s.end as u32))
            .collect();
    (sess.media_digest, spans)
}

/// P2.2b2: freeze the canonical host prefix, release displaced GPU refs and PIN
/// this prefix's pages. Shared by the prefill-boundary freeze (after its
/// DeltaNet capture) and the terminal endpoint freeze (which reuses a coverage
/// checkpoint instead of capturing at a frontier the live recurrent state no
/// longer holds).
#[cfg(feature = "kvmem")]
pub(super) fn kvmem_catalog_and_pin(
    rig: &mut Rig,
    s: usize,
    boundary: u32,
    tokens: &[u32],
    page_indices: &[u32],
    media_digest: Option<u64>,
    media_spans: Vec<(u32, u32)>,
) -> Result<(), String> {
    let ctl = rig
        .model
        .kvmem
        .as_mut()
        .ok_or_else(|| "kvmem freeze: no controller".to_string())?;
    let (id, displaced, deduped) =
        ctl.freeze_prefix_with_media(s, boundary, tokens, media_digest, media_spans)?;
    for d in displaced {
        if let Some(pp) = rig.pool.page_pool_mut() {
            for &phys in &d.gpu_pages {
                let _ = pp.release_cache_ref(phys);
            }
        }
    }
    if !deduped {
        let pp = rig
            .pool
            .page_pool_mut()
            .ok_or_else(|| "kvmem freeze: not paged".to_string())?;
        let mut taken: Vec<u32> = Vec::new();
        let mut ref_err = None;
        for &phys in page_indices {
            match pp.add_cache_ref(phys) {
                Ok(()) => taken.push(phys),
                Err(e) => {
                    ref_err = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = ref_err {
            for phys in taken {
                let _ = pp.release_cache_ref(phys);
            }
            ctl.release_prefix(id);
            return Err(format!("kvmem freeze cache ref: {e}"));
        }
        if let Some(ckpt) = ctl.prefix_checkpoint_mut(id) {
            ckpt.gpu_pages = page_indices.to_vec();
        }
    }
    // 期4: co-archive the DFlash2 draft conditioning image (KVMem/nInfer
    // "native full sequence-state serialization"). Only when the draft's live
    // seeded frontier equals this checkpoint's frontier — the cached ring
    // cannot reconstruct an earlier frontier, so those checkpoints carry no
    // image and reseed on reuse. Capture failure never drops the KV
    // checkpoint (the target prefix stays reusable; only the draft reseeds).
    // Layerwise only: Legacy keeps the historical AR-on-reuse path and stays
    // byte/time-neutral (no image copies).
    if !deduped && ctl.strategy().is_layerwise() {
        let boundary_usize = boundary as usize;
        let capture = match (
            rig.model.dflash.as_ref(),
            rig.model.dflash_states[s].as_ref(),
        ) {
            (Some(shared), Some(st)) if st.seeded_through == boundary_usize => {
                let floor = ctl.draft_conditioning_floor(s, boundary_usize);
                Some(crate::dflash_slot::dflash_capture_image(
                    &rig.gpu,
                    shared,
                    st,
                    boundary_usize,
                    floor,
                ))
            }
            _ => None,
        };
        match capture {
            Some(Ok(img)) => {
                if let Some(ckpt) = ctl.prefix_checkpoint_mut(id) {
                    ckpt.draft_image = Some(img);
                }
                ctl.note_draft_image_capture();
            }
            Some(Err(e)) => {
                eprintln!("[kvmem-draft-image] capture skipped at {boundary}: {e}");
            }
            None => {}
        }
    }
    // 期3 G3 P1.1 dn-persist: co-locate the recurrent state with the pack.
    // Live-frontier freezes export the live DN state; endpoint/coverage
    // freezes label an earlier frontier whose state lives in the checkpoint
    // pool (the coverage capture), so export THAT snapshot's bytes. Without a
    // source the pack carries no sidecar (restart reuse falls back to cold).
    if !deduped {
        if let Some(pack) = ctl.prefix_pack_path(id) {
            if ctl.prefix_sidecar_ready(id) {
                if ctl.diagnostics().trace {
                    eprintln!("[kvmem-pack] DN sidecar reused at frontier {boundary}");
                }
            } else {
                let dn_path = crate::kvmem_prefix::sidecar_path(&pack);
                let live = ctl.fed_tokens(s) == Some(boundary);
                let exported: Result<Vec<u8>, String> = if live {
                    crate::speculative::DeltaNetSnapshot::export_host_from_state(
                        &rig.gpu,
                        &rig.model.dn_states[s],
                    )
                } else {
                    let domain = rig
                        .cache_domain
                        .as_ref()
                        .ok_or_else(|| "no cache domain".to_string())?;
                    let fp = crate::checkpoint::checkpoint_fingerprint(
                        &tokens[..boundary as usize],
                        media_digest,
                    );
                    match rig
                        .model
                        .checkpoint_pool
                        .as_ref()
                        .and_then(|pool| pool.peek(domain, boundary as u64, fp))
                    {
                        Some(snapshot) => snapshot.export_host(&rig.gpu),
                        None => Err("no recurrent checkpoint at the frozen frontier".to_string()),
                    }
                };
                match exported {
                    Ok(bytes) => {
                        let tmp = std::path::PathBuf::from(format!("{}.tmp", dn_path.display()));
                        match std::fs::write(&tmp, &bytes)
                            .and_then(|_| std::fs::rename(&tmp, &dn_path))
                        {
                            Ok(()) => {
                                ctl.note_pack_sidecar(&pack, bytes.len() as u64);
                                if ctl.diagnostics().trace {
                                    eprintln!(
                                        "[kvmem-pack] DN sidecar {} bytes at frontier {boundary} (live={live})",
                                        bytes.len()
                                    );
                                }
                            }
                            Err(e) => eprintln!("[kvmem-pack] DN sidecar skipped: {e}"),
                        }
                    }
                    Err(e) => eprintln!("[kvmem-pack] DN sidecar skipped: {e}"),
                }
            }
        }
    }
    Ok(())
}

/// P2.2b2 decode-side coverage capture (ninfer `CheckpointCapturePlan`):
/// capture one recurrent checkpoint at each page-aligned frontier that is also
/// a multiple of the configured interval. The live state cannot rewind, so the
/// terminal endpoint freeze can only label a frontier whose DN state exists;
/// coverage captures provide exactly those anchors. Soft: a pool refusal or a
/// HIP error skips the boundary, never the serve loop.
#[cfg(feature = "kvmem")]
pub(super) fn kvmem_coverage_capture(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &[PendingWork],
) {
    if !rig.kvmem_prefix_pin || !rig.kvmem_prefix_coverage {
        return;
    }
    let Some(domain) = rig.cache_domain.clone() else {
        return;
    };
    let block = crate::kvmem_ctl::KV_BLOCK_TOKENS as u64;
    let interval = rig.kvmem_prefix_coverage_interval as u64;
    for s in 0..slots.len() {
        if !work[s].decoding {
            continue;
        }
        let Some(frame) = slots[s].as_ref() else {
            continue;
        };
        let session = frame.session;
        let Some(sess) = rig.sessions.get(session) else {
            continue;
        };
        if sess.has_image {
            continue;
        }
        let Some(fed) = rig.model.kvmem.as_ref().and_then(|ctl| ctl.fed_tokens(s)) else {
            continue;
        };
        let fed = fed as u64;
        if fed == 0 || fed % block != 0 || fed % interval != 0 {
            continue;
        }
        let boundary = fed as u32;
        if sess.tokens.len() < boundary as usize {
            continue;
        }
        let tokens: Vec<u32> = sess.tokens[..boundary as usize].to_vec();
        let fp = crate::checkpoint::checkpoint_fingerprint(&tokens, None);
        let model = &mut rig.model;
        let Some(pool) = model.checkpoint_pool.as_mut() else {
            continue;
        };
        if pool.peek(&domain, fed, fp).is_some() {
            continue;
        }
        let state = &model.dn_states[s];
        match crate::checkpoint::capture_checkpoint(
            &mut rig.gpu,
            pool,
            &domain,
            fed,
            &tokens,
            None, // coverage is text-only (vision sessions skip it)
            state,
        ) {
            Ok(id) if id == crate::checkpoint::CheckpointId::NONE => {
                // Pool ceiling refused capture; the boundary stays unreusable.
            }
            Ok(_) => {
                if let Some(f) = slots[s].as_mut() {
                    f.last_coverage_boundary = fed as usize;
                }
                if rig.gpu.slot_trace() {
                    eprintln!("[kvmem-prefix] slot {s} coverage checkpoint at {fed}");
                }
            }
            Err(e) => eprintln!("[kvmem-prefix] coverage capture skipped: {e:?}"),
        }
    }
}

/// P2.2b2 terminal endpoint freeze: freeze the page-aligned accepted frontier
/// (including generated rows) using an existing coverage checkpoint as the
/// recurrent-state proof. Pages must be resident (PIN is page-granular), so a
/// conversation that overflowed the resident budget simply skips the endpoint.
#[cfg(feature = "kvmem")]
pub(super) fn kvmem_freeze_endpoint(rig: &mut Rig, slots: &mut [Option<InFlight>], s: usize) {
    let trace = kvmem_trace(rig);
    if !rig.kvmem_prefix_pin {
        return;
    }
    let Some(frame) = slots[s].as_ref() else {
        return;
    };
    let session = frame.session;
    let last_frozen = frame.last_frozen_boundary;
    let Some(sess) = rig.sessions.get(session) else {
        return;
    };
    let Some(fed) = rig.model.kvmem.as_ref().and_then(|ctl| ctl.fed_tokens(s)) else {
        return;
    };
    let block = crate::kvmem_ctl::KV_BLOCK_TOKENS as u32;
    let aligned = (fed / block) * block;
    let anchor = frame.last_coverage_boundary as u32;
    let mut boundary = aligned;
    if boundary == 0 || (boundary as usize) <= last_frozen {
        boundary = anchor;
    }
    if boundary == 0
        || boundary > fed
        || (boundary as usize) <= last_frozen
        || sess.tokens.len() < boundary as usize
    {
        return;
    }
    let n_pages = (boundary / block) as usize;
    let Some(bt) = rig.pool.block_table(SlotId(s)) else {
        return;
    };
    if bt.num_pages() < n_pages {
        return;
    }
    let page_indices: Vec<u32> = bt.page_indices()[..n_pages].to_vec();
    if page_indices
        .iter()
        .any(|&p| p == rdna_compute::page_pool::SPILLED_PAGE)
    {
        return;
    }
    let tokens: Vec<u32> = sess.tokens[..boundary as usize].to_vec();
    // G7: the endpoint frontier must not split an image span; carry the
    // session's media identity into the checkpoint.
    let (media_digest, all_spans) = kvmem_session_media(rig, session);
    if let Some(&(b, e)) = all_spans
        .iter()
        .find(|&&(b, e)| b < boundary && boundary < e)
    {
        // G7 Stage E probe: the endpoint replay anchor refuses this frontier.
        if trace {
            eprintln!(
                "[kvmem-media] endpoint freeze skip boundary={boundary} \
                 splits span=[{b},{e})"
            );
        }
        return;
    }
    let media_spans: Vec<(u32, u32)> = all_spans
        .iter()
        .copied()
        .filter(|&(_, e)| e <= boundary)
        .collect();
    // G7 Stage E probe: endpoint frontier carries a media-aligned identity.
    if trace {
        let fmt = |d: Option<u64>| {
            d.map(|x| format!("{x:016x}"))
                .unwrap_or_else(|| "none".to_string())
        };
        eprintln!(
            "[kvmem-media] endpoint freeze boundary={boundary} digest={} spans={} inside={}",
            fmt(media_digest),
            all_spans.len(),
            media_spans.len()
        );
    }
    let Some(domain) = rig.cache_domain.as_ref() else {
        return;
    };
    let Some(pool) = rig.model.checkpoint_pool.as_ref() else {
        return;
    };
    if pool
        .peek(
            domain,
            boundary as u64,
            crate::checkpoint::checkpoint_fingerprint(&tokens, media_digest),
        )
        .is_none()
    {
        return;
    }
    match kvmem_catalog_and_pin(
        rig,
        s,
        boundary,
        &tokens,
        &page_indices,
        media_digest,
        media_spans,
    ) {
        Ok(()) => {
            if let Some(f) = slots[s].as_mut() {
                f.last_frozen_boundary = boundary as usize;
            }
            if rig.gpu.slot_trace() {
                eprintln!("[kvmem-prefix] slot {s} endpoint freeze at {boundary}");
            }
        }
        Err(e) => eprintln!("[kvmem-prefix] endpoint freeze skipped: {e}"),
    }
}

#[cfg(feature = "kvmem")]
enum LegacyDraftImage {
    None,
    Mtp(crate::legacy_draft::LegacyMtpImage),
    Dflash(crate::legacy_draft::LegacyDflashImage),
}

#[cfg(feature = "kvmem")]
pub(super) struct LegacyNativeImage {
    pub(super) store: crate::kvmem_ctl::LegacyStoreSnapshot,
    dn: Vec<u8>,
    draft: LegacyDraftImage,
}

#[cfg(feature = "kvmem")]
impl LegacyNativeImage {
    fn auxiliary_bytes(&self) -> Result<usize, String> {
        let draft = match &self.draft {
            LegacyDraftImage::None => 0,
            LegacyDraftImage::Mtp(image) => image.host_bytes(),
            LegacyDraftImage::Dflash(image) => image.host_bytes(),
        };
        self.dn
            .len()
            .checked_add(draft)
            .ok_or_else(|| "kvmem: native image size overflow".into())
    }

    pub(super) fn supports(&self, kind: SpecKind) -> bool {
        matches!(
            (kind, &self.draft),
            (SpecKind::None, _)
                | (SpecKind::Mtp, LegacyDraftImage::Mtp(_))
                | (SpecKind::Dflash, LegacyDraftImage::Dflash(_))
        ) || self.store.frontier == 0
    }
}

#[cfg(feature = "kvmem")]
pub(super) struct LegacyQueryRun {
    pub(super) query: crate::legacy_query::LegacyQuery,
    pub(super) source: Option<LegacyNativeImage>,
    scores: Option<Vec<f64>>,
    kind: SpecKind,
}

#[cfg(feature = "kvmem")]
fn capture_legacy_native_image(
    rig: &mut Rig,
    slot: usize,
    kind: SpecKind,
) -> Result<LegacyNativeImage, String> {
    rig.gpu
        .hip
        .device_synchronize()
        .map_err(|e| format!("kvmem snapshot drain: {e}"))?;
    let ctl = rig
        .model
        .kvmem
        .as_mut()
        .ok_or("kvmem: snapshot without controller")?;
    let frontier = ctl
        .fed_tokens(slot)
        .ok_or("kvmem: snapshot slot out of range")? as usize;
    let dn = crate::speculative::DeltaNetSnapshot::export_host_from_state(
        &rig.gpu,
        &rig.model.dn_states[slot],
    )?;
    let draft = if frontier == 0 {
        LegacyDraftImage::None
    } else {
        match kind {
            SpecKind::None => LegacyDraftImage::None,
            SpecKind::Mtp => LegacyDraftImage::Mtp(crate::legacy_draft::LegacyMtpImage::capture(
                &rig.gpu,
                rig.model.mtp_states[slot]
                    .as_ref()
                    .ok_or("kvmem: missing MTP history at snapshot")?,
                frontier,
            )?),
            SpecKind::Dflash => {
                // A checkpoint/image is taken at a prefill boundary, never in
                // the middle of a speculative verify round. Any retained draft
                // rows are the previous round's request-local rollback scratch
                // and must not be frozen into (or block) the published image.
                if let Some(st) = rig.model.dflash_states[slot].as_mut() {
                    crate::dflash_slot::dflash_abandon_inflight(&mut rig.gpu, st);
                }
                LegacyDraftImage::Dflash(crate::legacy_draft::LegacyDflashImage::capture(
                    &rig.gpu,
                    rig.model
                        .dflash
                        .as_ref()
                        .ok_or("kvmem: missing DFlash model")?,
                    rig.model.dflash_states[slot]
                        .as_ref()
                        .ok_or("kvmem: missing DFlash history at snapshot")?,
                    frontier,
                    0,
                )?)
            }
        }
    };
    let store = ctl.capture_legacy_snapshot(
        &rig.gpu,
        &rig.pool,
        &rig.model.k_arenas,
        &rig.model.v_arenas,
        slot,
    )?;
    eprintln!("[kvmem-legacy] capture slot={slot} frontier={frontier} payload_bytes={} dn_bytes={} spec={kind:?}", store.payload_bytes(), dn.len());
    Ok(LegacyNativeImage { store, dn, draft })
}

/// Park a finished session's native state in the inactive-history catalog so
/// a later request whose complete rendered prompt extends it can restore
/// instead of re-prefilling (`Sessions` = non-active histories). Called while
/// the slot is still intact, right before its session is closed: after the
/// close the parked image is the only owner of this frontier, so the catalog
/// charge is the real host cost (KV payload + recurrent bytes), not a
/// duplicate of a resident store.
#[cfg(feature = "kvmem")]
pub(super) fn park_legacy_history(
    rig: &mut Rig,
    session: SessionId,
) -> Result<Option<u64>, String> {
    if !kvmem_legacy_enabled(rig) || rig.legacy_history.is_none() {
        return Ok(None);
    }
    let Some(sess) = rig.sessions.get(session) else {
        return Ok(None);
    };
    if sess.tokens.is_empty() {
        return Ok(None);
    }
    let Some(slot) = sess.slot else {
        return Ok(None);
    };
    let tokens = sess.tokens.clone();
    // A VL history is parked with the session's sticky media digest. The
    // catalog then refuses it for any request whose current image digest
    // differs, so two images with identical `image_pad` token ids can never
    // share a restored prefix.
    let media = sess.media_digest.filter(|_| sess.has_image);
    // Catalog group is the conversation identity: several parked versions of
    // the same chat (different turn counts) must contend as versions of one
    // group. The first user turn is the stable root of a rendered conversation;
    // an empty convo has no identity, so fall back to a nonzero per-session
    // group. Group 0 is reserved for "unknown".
    let mut group = convo_group(&sess.convo[..sess.convo.len().min(1)]);
    if group == 0 {
        group = session
            .0
            .checked_add(1)
            .ok_or("kvmem: history group overflow")?;
    }
    // Park with the spec provenance the slot actually ran: a draft-less image
    // can only be reused by a draft-less request, so dropping a live MTP/DFlash
    // image would make the parked history unusable under the default config.
    // The idle slot's work entry was cleared at Done, but the draft states
    // persist, so the capture either proves its native frontier or reports a
    // miss (then the target-only image is parked).
    let run_kind = rig.legacy_spec_kind[slot.0];
    if run_kind == SpecKind::Dflash {
        // The idle slot may still hold the last cycle's unconsumed draft
        // (request-local scratch). Park only the published conditioning state.
        if let Some(st) = rig.model.dflash_states[slot.0].as_mut() {
            crate::dflash_slot::dflash_abandon_inflight(&mut rig.gpu, st);
        }
    }
    let image = match capture_legacy_native_image(rig, slot.0, run_kind) {
        Ok(image) => image,
        Err(e) => {
            eprintln!(
                "[kvmem-legacy] HistoryPark slot={} draft capture ({run_kind:?}) failed: {e}",
                slot.0
            );
            match capture_legacy_native_image(rig, slot.0, SpecKind::None) {
                Ok(image) => image,
                Err(e) => {
                    eprintln!("[kvmem-legacy] HistoryPark slot={} skipped: {e}", slot.0);
                    return Ok(None);
                }
            }
        }
    };
    if image.store.frontier == 0 {
        return Ok(None);
    }
    let charge = usize::try_from(image.store.payload_bytes())
        .ok()
        .and_then(|bytes| bytes.checked_add(image.auxiliary_bytes().ok()?))
        .ok_or("kvmem: history image size overflow")?;
    let source = image.store.frontier as usize;
    let payload = image.store.payload_bytes();
    let catalog = rig
        .legacy_history
        .as_mut()
        .ok_or("kvmem: catalog vanished")?;
    let publication = match catalog.publish(
        session.0,
        group,
        tokens,
        source,
        media,
        charge,
        std::rc::Rc::new(image),
    ) {
        Ok(publication) => publication,
        Err(e) => {
            eprintln!(
                "[kvmem-legacy] HistoryPark session={} skipped: {e}",
                session.0
            );
            return Ok(None);
        }
    };
    for retired in &publication.retired {
        eprintln!(
            "[kvmem-legacy] HistoryRetire id={} source_session={}",
            retired.id, retired.source_session
        );
    }
    eprintln!(
        "[kvmem-legacy] HistoryPark id={} session={} frontier={} charge_bytes={charge} kv_bytes={payload} catalog_entries={}",
        publication.id,
        session.0,
        source,
        catalog.len()
    );
    Ok(Some(publication.id))
}

/// Restore target and draft as one execution frontier. The source stays owned
/// by its caller; any error invalidates this destination before more inference.
#[cfg(feature = "kvmem")]
pub(super) fn restore_legacy_native_image(
    rig: &mut Rig,
    slot: usize,
    session: u64,
    source: &LegacyNativeImage,
    selected: &[u32],
    kind: SpecKind,
) -> Result<u64, String> {
    if !source.supports(kind) {
        return Err(format!(
            "kvmem: cached source has no complete {kind:?} history"
        ));
    }
    rig.gpu
        .hip
        .device_synchronize()
        .map_err(|e| format!("kvmem restore drain: {e}"))?;
    // Decode the recurrent image into a private temporary first. Invalid image
    // shape/length cannot partially overwrite the live recurrent state.
    let mut snapshot =
        crate::speculative::DeltaNetSnapshot::new_for(&mut rig.gpu, &rig.model.dn_states[slot])
            .map_err(|e| format!("kvmem restore DN scratch: {e}"))?;
    let result = (|| {
        snapshot.import_host(&rig.gpu, &source.dn)?;
        let bytes = rig
            .model
            .kvmem
            .as_mut()
            .ok_or("kvmem: restore without controller")?
            .restore_legacy_snapshot(
                &rig.gpu,
                &mut rig.pool,
                &rig.model.k_arenas,
                &rig.model.v_arenas,
                slot,
                &source.store,
                selected,
            )?;
        snapshot
            .restore_to(&mut rig.model.dn_states[slot], &mut rig.gpu)
            .map_err(|e| format!("kvmem restore DN: {e}"))?;
        if source.store.frontier == 0 {
            if let Some(state) = rig.model.mtp_states[slot].as_mut() {
                state
                    .reset(&mut rig.gpu)
                    .map_err(|e| format!("kvmem zero-frontier MTP reset: {e}"))?;
            }
            if let Some(state) = rig.model.dflash_states[slot].take() {
                state.free_gpu(&mut rig.gpu);
            }
        } else {
            match (&source.draft, kind) {
                (LegacyDraftImage::Mtp(image), SpecKind::Mtp) => image.restore(
                    &mut rig.gpu,
                    rig.model.mtp_states[slot]
                        .as_mut()
                        .ok_or("kvmem: missing destination MTP state")?,
                    source.store.frontier as usize,
                )?,
                (LegacyDraftImage::Dflash(image), SpecKind::Dflash) => {
                    // Cold admission frees the previous occupant's draft state;
                    // allocate the empty destination the same way the first
                    // prefill chunk would, then let the image reinstate it.
                    if rig.model.dflash_states[slot].is_none() {
                        let st = crate::dflash_slot::new_dflash_slot_state(
                            &mut rig.gpu,
                            rig.model
                                .dflash
                                .as_ref()
                                .ok_or("kvmem: missing destination DFlash model")?,
                            &rig.model.config,
                            &rig.model.dn_states[slot],
                        )
                        .map_err(|e| format!("kvmem restore DFlash state alloc: {e}"))?;
                        rig.model.dflash_states[slot] = Some(st);
                    }
                    image.restore(
                        &mut rig.gpu,
                        rig.model
                            .dflash
                            .as_ref()
                            .ok_or("kvmem: missing destination DFlash model")?,
                        rig.model.dflash_states[slot]
                            .as_mut()
                            .ok_or("kvmem: missing destination DFlash state")?,
                        session,
                        source.store.frontier as usize,
                    )?
                }
                (_, SpecKind::None) => {
                    // A non-zero frontier can be restored without a native
                    // draft image (the catalog parks store+DN only), but the
                    // destination's draft state then carries another
                    // request's history. NInfer restores a draft only from a
                    // matching serialized image; anything else is reset, not
                    // reused.
                    if let Some(state) = rig.model.mtp_states[slot].as_mut() {
                        state
                            .reset(&mut rig.gpu)
                            .map_err(|e| format!("kvmem draftless-restore MTP reset: {e}"))?;
                    }
                    if let Some(state) = rig.model.dflash_states[slot].take() {
                        state.free_gpu(&mut rig.gpu);
                    }
                }
                _ => return Err("kvmem: native draft image kind mismatch".into()),
            }
        }
        if hipfire_config::developer_bool("HIPFIRE_KVMEM_VERIFY", false) {
            let actual = crate::speculative::DeltaNetSnapshot::export_host_from_state(
                &rig.gpu,
                &rig.model.dn_states[slot],
            )?;
            if actual != source.dn {
                return Err("kvmem: recurrent/conv native image readback mismatch".into());
            }
            eprintln!("[kvmem-verify] restore slot={slot} frontier={} dn_bytes={} recurrent_conv_mismatch=0", source.store.frontier, source.dn.len());
        }
        Ok(bytes)
    })();
    snapshot.free_gpu(&mut rig.gpu);
    result
}

#[cfg(feature = "kvmem")]
pub(super) fn prepare_legacy_query_checkpoints(
    rig: &mut Rig,
    slots: &[Option<InFlight>],
    work: &[PendingWork],
) -> Result<Vec<usize>, String> {
    let mut caps = vec![usize::MAX; work.len()];
    if !rig
        .model
        .kvmem
        .as_ref()
        .is_some_and(|c| c.legacy_config().is_some())
    {
        return Ok(caps);
    }
    for slot in 0..work.len() {
        if slots[slot].is_none() || work[slot].decoding {
            continue;
        }
        if rig.legacy_queries[slot].is_none() && !rig.legacy_query_decided[slot] {
            let ctl = rig.model.kvmem.as_ref().unwrap();
            let end = work[slot]
                .next_pos
                .checked_add(work[slot].remaining_prompt.len())
                .ok_or("kvmem: prompt end overflow")?;
            match ctl.legacy_query_for_request(slot, ctl.request_query(slot), end)? {
                Some(query) => {
                    rig.legacy_queries[slot] = Some(LegacyQueryRun {
                        query,
                        source: None,
                        scores: None,
                        kind: work[slot].spec,
                    });
                }
                None => {
                    // No recall for this request (no recoverable history, the
                    // query sits at the accepted frontier, or a checkpoint is
                    // impossible). Latch the decision: retrying after the
                    // frontier advanced would build a Query that crosses
                    // already-accepted rows and abort the request.
                    rig.legacy_query_decided[slot] = true;
                }
            }
        }
        let due = rig.legacy_queries[slot]
            .as_ref()
            .is_some_and(|run| run.query.checkpoint_due(work[slot].next_pos));
        if due {
            let (kind, restore_begin, prompt_end) = {
                let run = rig.legacy_queries[slot].as_ref().unwrap();
                (
                    run.kind,
                    run.query.restore_begin() as u64,
                    run.query.prompt_end() as u64,
                )
            };
            // A Query only earns its checkpoint when the probe can actually
            // need cold pages. If the whole restore frontier is resident and
            // the prompt (plus speculative workspace) fits the window, the
            // post-probe policy is already known to be `window_resident`:
            // skip the checkpoint (which copies every resident page D2H into
            // the host fork and exports the recurrent DN) and keep the intact
            // first pass. This is the expensive all-resident cold-reuse case
            // (the K7 full-budget arm).
            let window_tokens = rig
                .model
                .kvmem
                .as_ref()
                .expect("Legacy query gate checked controller presence")
                .window_capacity_tokens();
            let workspace = if kind.active() {
                rig.model.spec_rows as u64
            } else {
                0
            };
            let resident_pages = rig.pool.block_table(SlotId(slot)).map_or(0, |bt| {
                (0..bt.num_pages())
                    .filter(|&lp| bt.physical(lp).is_some())
                    .count()
            });
            let resident_tokens = resident_pages as u64 * crate::kvmem_ctl::KV_BLOCK_TOKENS as u64;
            let window_resident = resident_tokens >= restore_begin
                && prompt_end.saturating_add(workspace) <= window_tokens;
            if window_resident {
                let run = rig.legacy_queries[slot].as_mut().unwrap();
                run.query.bypass("window_resident_precapture")?;
                eprintln!(
                    "[kvmem-legacy] QueryBypass slot={slot} reason=window_resident \
                     resident_pages={resident_pages} frontier={restore_begin} \
                     prompt_end={prompt_end} first_pass_preserved=true"
                );
                if let Some(run) = rig.legacy_queries[slot].as_ref() {
                    caps[slot] = run.query.cap_prefill(
                        work[slot].next_pos,
                        work[slot].remaining_prompt.len().min(rig.prefill_chunk),
                    );
                }
                continue;
            }
            let capture = capture_legacy_native_image(rig, slot, kind);
            let run = rig.legacy_queries[slot].as_mut().unwrap();
            match capture {
                Ok(image) if image.store.frontier as usize == run.query.restore_begin() => {
                    run.source = Some(image);
                    run.query.checkpoint_captured()?;
                }
                Ok(image) => {
                    // A frontier mismatch means the native owner is not at the
                    // recovery point this Query needs. Recall is optional; the
                    // intact first pass stays authoritative.
                    let reason = format!(
                        "checkpoint_frontier {} != recovery point {}",
                        image.store.frontier,
                        run.query.restore_begin()
                    );
                    run.query.bypass(&reason)?;
                    eprintln!("[kvmem-legacy] QueryBypass slot={slot} reason={reason} first_pass_preserved=true");
                }
                Err(error) => {
                    // Nothing restored or published: the live first-pass view
                    // remains authoritative, matching NInfer's optional-source
                    // materialization failure path.
                    let reason = format!("checkpoint_unavailable: {error}");
                    run.query.bypass(&reason)?;
                    eprintln!("[kvmem-legacy] QueryBypass slot={slot} reason={reason} first_pass_preserved=true");
                }
            }
        }
        if let Some(run) = rig.legacy_queries[slot].as_ref() {
            caps[slot] = run.query.cap_prefill(
                work[slot].next_pos,
                work[slot].remaining_prompt.len().min(rig.prefill_chunk),
            );
        }
    }
    Ok(caps)
}

/// Outcome of the post-probe Query policy evaluation.
#[cfg(feature = "kvmem")]
enum LegacyQueryDecision {
    /// Keep the intact first pass. Publish the score round only when one was
    /// actually scored (a refused selection produced none).
    Skip {
        reason: String,
        scores: Option<Vec<f64>>,
    },
    /// Admitted: the destructive rewind is allowed and required.
    Replay {
        selected: Vec<u32>,
        scores: Vec<f64>,
        plan: hipfire_kvmem::LegacyReplayPlan,
    },
}

/// Decide what a completed Query probe does, mirroring NInfer: score the
/// pre-Query history, then admit a full causal replay only when the window
/// actually needed recall (cold pages or prompt+workspace beyond the resident
/// window) and the core plan admits the span. Refusals never disturb the
/// first pass.
#[cfg(feature = "kvmem")]
fn plan_legacy_query_replay(
    rig: &Rig,
    run: &mut LegacyQueryRun,
) -> Result<LegacyQueryDecision, String> {
    let Some(control) = rig.model.kvmem.as_ref() else {
        return Ok(LegacyQueryDecision::Skip {
            reason: "controller_missing".into(),
            scores: None,
        });
    };
    let Some(source) = run.source.as_ref() else {
        return Ok(LegacyQueryDecision::Skip {
            reason: "checkpoint_missing".into(),
            scores: None,
        });
    };
    let range = run.query.query();
    if range.begin == 0 {
        // NInfer requires history in front of the Query; nothing to recall.
        return Ok(LegacyQueryDecision::Skip {
            reason: "query_at_frontier".into(),
            scores: None,
        });
    }
    let (selected, scores) = match control.select_legacy_query(&source.store, &run.query) {
        Ok(result) => result,
        Err(e) => {
            return Ok(LegacyQueryDecision::Skip {
                reason: format!("selection_refused: {e}"),
                scores: None,
            })
        }
    };
    let prompt_end = run.query.prompt_end() as u64;
    let restore_begin = run.query.restore_begin() as u64;
    let window_tokens = control.window_capacity_tokens();
    let device_pages = control.budget_blocks() as u64 + control.gen_reserve().div_ceil(128u64);
    let history_pages = restore_begin
        .div_ceil(128u64)
        .min(control.budget_blocks() as u64);
    let workspace = if run.kind.active() {
        rig.model.spec_rows as u64
    } else {
        0
    };
    // `resident_pages` counts accepted pages with a live physical binding; the
    // token estimate over-counts a partial tail, so "spilled" is conservative.
    let resident_tokens = source.store.resident_pages.len() as u64 * 128;
    let needs_recall =
        resident_tokens < restore_begin || prompt_end.saturating_add(workspace) > window_tokens;
    if !needs_recall {
        // The whole context is resident: a rewind would recreate the same view.
        return Ok(LegacyQueryDecision::Skip {
            reason: "window_resident".into(),
            scores: Some(scores),
        });
    }
    let plan = match run.query.replay_plan(
        history_pages,
        device_pages,
        workspace.min(u32::MAX as u64) as u32,
    ) {
        Ok(plan) => plan,
        Err(e) => {
            return Ok(LegacyQueryDecision::Skip {
                reason: format!("plan_error: {e}"),
                scores: Some(scores),
            })
        }
    };
    if !plan.admitted {
        return Ok(LegacyQueryDecision::Skip {
            reason: format!("replay_refused: {}", plan.reason),
            scores: Some(scores),
        });
    }
    Ok(LegacyQueryDecision::Replay {
        selected,
        scores,
        plan,
    })
}

/// One Query decision per iteration, after the step's accepted rows are fed.
/// Probe completion is measured at the accepted native frontier. A refused or
/// pointless replay keeps the intact first pass and still publishes the score
/// round exactly once; a replayed query publishes it once after the replay
/// reaches the prompt end again (the restore rolled LRFU back to pre-Query).
#[cfg(feature = "kvmem")]
pub(super) fn advance_legacy_query_replay(
    rig: &mut Rig,
    slots: &[Option<InFlight>],
    work: &mut [PendingWork],
    rewound: &mut [bool],
) -> Result<(), String> {
    use crate::legacy_query::LegacyQueryPhase;
    for s in 0..work.len() {
        let phase = match rig.legacy_queries[s].as_ref() {
            None => continue,
            Some(run) => run.query.phase(),
        };
        match phase {
            LegacyQueryPhase::AwaitingCheckpoint => continue,
            LegacyQueryPhase::Complete | LegacyQueryPhase::Bypassed => {
                rig.legacy_queries[s] = None;
                rig.legacy_query_decided[s] = true;
                continue;
            }
            LegacyQueryPhase::Replaying => {
                let Some(run) = rig.legacy_queries[s].as_ref() else {
                    continue;
                };
                if work[s].next_pos != run.query.prompt_end() {
                    continue;
                }
                let prompt_end = run.query.prompt_end();
                let mut run = rig.legacy_queries[s].take().unwrap();
                run.query.finish_replay(prompt_end)?;
                let scores = match run.scores {
                    Some(scores) => scores,
                    None => return Err("kvmem: Query replay without recorded scores".into()),
                };
                let ctl = rig
                    .model
                    .kvmem
                    .as_mut()
                    .ok_or("kvmem: Query replay without controller")?;
                ctl.commit_legacy_query(s, &scores)?;
                eprintln!("[kvmem-legacy] QueryReplayDone slot={s} prompt_end={prompt_end}");
                continue;
            }
            LegacyQueryPhase::Probing => {}
        }
        if slots[s].is_none() {
            rig.legacy_queries[s] = None;
            rig.legacy_query_decided[s] = true;
            continue;
        }
        {
            let Some(run) = rig.legacy_queries[s].as_mut() else {
                continue;
            };
            if !run.query.probe_complete(work[s].next_pos) {
                continue;
            }
        }
        let mut run = match rig.legacy_queries[s].take() {
            Some(run) => run,
            None => continue,
        };
        let decision = plan_legacy_query_replay(rig, &mut run)?;
        match decision {
            LegacyQueryDecision::Skip { reason, scores } => {
                if let Some(scores) = scores {
                    let ctl = rig
                        .model
                        .kvmem
                        .as_mut()
                        .ok_or("kvmem: Query commit without controller")?;
                    ctl.commit_legacy_query(s, &scores)?;
                }
                eprintln!(
                    "[kvmem-legacy] QuerySkip slot={s} reason={reason} first_pass_preserved=true"
                );
                rig.legacy_query_decided[s] = true;
            }
            LegacyQueryDecision::Replay {
                selected,
                scores,
                plan,
            } => {
                let prompt_end = run.query.prompt_end();
                if let Err(e) = run.query.start_replay(&plan) {
                    // Nothing destructive ran: the first pass stays authoritative.
                    // Latch the bypass (still Probing here) so the run is retired
                    // on the next iteration instead of silently vanishing with an
                    // unfinished phase.
                    let reason = format!("start_replay: {e}");
                    run.query.bypass(&reason)?;
                    rig.legacy_queries[s] = Some(run);
                    eprintln!("[kvmem-legacy] QuerySkip slot={s} reason={reason} first_pass_preserved=true");
                    continue;
                }
                run.scores = Some(scores);
                let session = slots[s]
                    .as_ref()
                    .map(|f| f.session)
                    .ok_or("kvmem: Query replay without an active flight")?;
                let source = run
                    .source
                    .as_ref()
                    .ok_or("kvmem: Query replay without checkpoint")?;
                let restore_bytes = match restore_legacy_native_image(
                    rig, s, session.0, source, &selected, run.kind,
                ) {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        rig.legacy_queries[s] = None;
                        rig.legacy_query_decided[s] = true;
                        return Err(format!("kvmem: Query replay restore failed: {e}"));
                    }
                };
                let tokens = rig
                    .sessions
                    .get(session)
                    .map(|sess| sess.tokens.clone())
                    .unwrap_or_default();
                if tokens.len() < prompt_end {
                    rig.legacy_queries[s] = None;
                    rig.legacy_query_decided[s] = true;
                    return Err(format!(
                        "kvmem: Query replay suffix needs {prompt_end} tokens, session holds {}",
                        tokens.len()
                    ));
                }
                let begin = plan.begin as usize;
                work[s].next_pos = begin;
                work[s].decoding = false;
                work[s].remaining_prompt = tokens[begin..prompt_end].to_vec();
                rewound[s] = true;
                rig.legacy_queries[s] = Some(run);
                eprintln!(
                    "[kvmem-legacy] QueryReplay slot={s} from={begin} prompt_end={prompt_end} selected={} restore_bytes={restore_bytes}",
                    selected.len()
                );
            }
        }
    }
    Ok(())
}

pub(super) fn capture_legacy_query_rows(
    rig: &mut Rig,
    batch: &SlotBatch,
    n: usize,
) -> Result<Vec<(usize, usize, usize, usize)>, String> {
    // ── KVMem Legacy Query row capture ─────────────────────────────
    // The tap kernel copies exactly the batch rows this step forwards
    // for the real user Query span; every other row is untouched. The
    // mask is re-set on EVERY step (empty when nothing probes) so a
    // later step can never read a stale row set. Verify rows are never
    // captured: they exist only while a slot is decoding, and a
    // Probing Query owns the prompt prefill.
    #[cfg(feature = "kvmem")]
    let mut legacy_query_reads: Vec<(usize, usize, usize, usize)> = Vec::new();
    #[cfg(feature = "kvmem")]
    if let Some(taps) = rig.model.pbs.kvmem_taps.as_mut() {
        let mut rows: Vec<u32> = Vec::new();
        let mut row = 0usize;
        for s in 0..n {
            let m = batch.m_per_slot.get(s).copied().unwrap_or(0);
            if m == 0 {
                continue;
            }
            let pos0 = batch.positions[row] as usize;
            row += m;
            let Some(run) = rig.legacy_queries[s].as_ref() else {
                continue;
            };
            if run.source.is_none() {
                continue;
            }
            let Some((off, count)) = run.query.capture_span(pos0, m) else {
                continue;
            };
            let compact = rows.len();
            let base = row - m + off;
            rows.extend((base..base + count).map(|r| r as u32));
            legacy_query_reads.push((s, compact, count, pos0 + off));
        }
        taps.set_query_rows(&mut rig.gpu, &rows)
            .map_err(|e| format!("kvmem: Query row mask: {e}"))?;
    }
    Ok(legacy_query_reads)
}

pub(super) fn prepare_kvmem_step(
    rig: &mut Rig,
    slots: &[Option<InFlight>],
    work: &[PendingWork],
    batch: &SlotBatch,
    n: usize,
) -> Result<(), String> {
    // ── KVMem residency pass (HIPFIRE_KVMEM) ────────────────────────
    // Runs BEFORE provisioning: a step's new pages come from the
    // pool's free list, so the pages this step sheds must be back in
    // it first. Restored pages are rebound in the slot's block table
    // before the forward resolves its writes through it; every byte
    // move is arena<->host memcpy, so a spill/restore round trips
    // bit-exactly. Selection only touches pages strictly below the
    // write frontier, so the step's own writes can never land on a
    // sentinel entry.
    #[cfg(feature = "kvmem")]
    if rig.model.kvmem.is_some() {
        let active: Vec<bool> = (0..n).map(|s| slots[s].is_some()).collect();
        // Idle slots that still hold a resident session keep their
        // KVMem host state: it backs the next turn's continuation
        // prefix (the session was kept at terminal instead of being
        // closed, see the terminal handler).
        let resident: Vec<bool> = (0..n)
            .map(|s| {
                rig.sessions
                    .iter()
                    .any(|(_, sess)| sess.slot.map(|sl| sl.0) == Some(s))
            })
            .collect();
        let prefill: Vec<bool> = (0..n).map(|s| !work[s].decoding).collect();
        // The batch drained `remaining_prompt`, so an empty remainder
        // means this chunk completes the prompt: its forward yields
        // the logits the first answer token is sampled from, and the
        // residency pass must have the answer's pages back by then.
        let prefill_final: Vec<bool> = (0..n)
            .map(|s| !work[s].decoding && work[s].remaining_prompt.is_empty())
            .collect();
        let legacy_pressure = rig
            .model
            .kvmem
            .as_ref()
            .is_some_and(|ctl| ctl.legacy_config().is_some());
        let step = if legacy_pressure {
            // Legacy Pressure candidates are accepted resident pages
            // only (never spilled pages, never full-history retrieval),
            // and no Pressure is legal inside an admitted Query replay.
            let incoming: Vec<usize> = (0..n)
                .map(|s| batch.m_per_slot.get(s).copied().unwrap_or(0))
                .collect();
            let decoding: Vec<bool> = (0..n).map(|s| work[s].decoding).collect();
            let replaying: Vec<bool> = (0..n)
                .map(|s| {
                    rig.legacy_queries
                        .get(s)
                        .and_then(|run| run.as_ref())
                        .is_some_and(|run| {
                            run.query.phase() == crate::legacy_query::LegacyQueryPhase::Replaying
                        })
                })
                .collect();
            let ctl = rig.model.kvmem.as_mut().unwrap();
            ctl.prepare_legacy_step(
                &mut rig.gpu,
                &mut rig.pool,
                &active,
                &resident,
                &incoming,
                &decoding,
                &replaying,
                &rig.model.k_arenas,
                &rig.model.v_arenas,
            )
        } else {
            let ctl = rig.model.kvmem.as_mut().unwrap();
            let taps = rig
                .model
                .pbs
                .kvmem_taps
                .as_ref()
                .expect("kvmem controller without statistics taps");
            ctl.prepare_step(
                &mut rig.gpu,
                &mut rig.pool,
                taps,
                &active,
                &resident,
                &prefill,
                &prefill_final,
                &rig.model.k_arenas,
                &rig.model.v_arenas,
            )
        };
        step?;
    }
    Ok(())
}

pub(super) fn freeze_prefill_boundaries(
    mut rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    batch: &SlotBatch,
    freeze_snapshot: &[(bool, usize)],
    n: usize,
    fwd_ok: bool,
) {
    let trace = kvmem_trace(&rig);
    // ── P4.3 KVMem canonical-prefix freeze (HIPFIRE_KVMEM_PREFIX_PIN) ──
    // END of the iteration: the trunk feed AND the MTP head aux
    // co-archive (spec prefill section) have both completed, so the
    // composite archive covers the chunk on BOTH planes (freezing
    // earlier snapshots an aux plane one chunk behind the trunk and
    // the branch's composite export fails closed). Freeze
    // the newly completed page-aligned prefix into the canonical
    // catalog: DeltaNet checkpoint + immutable host fork + sealed GPU
    // page PIN (cache refs). The chunk must END page-aligned (the live
    // recurrent state sits at next_pos and cannot be relabelled to an
    // earlier boundary, spec §4.5). G7: vision sessions freeze too; a
    // boundary that would split an image span is skipped below, and the
    // checkpoint carries the image identity. Generated rows are never
    // frozen — the shared prefix is prompt content; generation stays
    // session-private. Opportunistic:
    // a failed freeze skips the boundary (the request continues cold-
    // reusable at earlier anchors), it never poisons the serve loop.
    #[cfg(feature = "kvmem")]
    if rig.kvmem_prefix_pin && fwd_ok {
        for s in 0..n {
            if batch.m_per_slot[s] == 0 || !freeze_snapshot[s].0 {
                continue;
            }
            let boundary = freeze_snapshot[s].1;
            if boundary == 0 || boundary % PAGE_TOKENS != 0 {
                continue;
            }
            let last_frozen = slots[s]
                .as_ref()
                .map(|f| f.last_frozen_boundary)
                .unwrap_or(0);
            if boundary <= last_frozen {
                continue;
            }
            let session = match slots[s].as_ref().map(|f| f.session) {
                Some(sid) => sid,
                None => continue,
            };
            let tokens: Vec<u32> = match rig.sessions.get(session) {
                Some(sess) if sess.tokens.len() >= boundary => sess.tokens[..boundary].to_vec(),
                _ => continue,
            };
            // G7: never freeze a boundary that splits an image span;
            // carry the session's media identity into the checkpoint.
            let (media_digest, all_spans) = kvmem_session_media(&rig, session);
            if all_spans
                .iter()
                .any(|&(b, e)| (b as usize) < boundary && boundary < e as usize)
            {
                if rig.gpu.slot_trace() {
                    eprintln!(
                        "[kvmem-prefix] slot {s} boundary {boundary} not frozen: \
                     splits a media span"
                    );
                }
                // G7 Stage E probe: the media replay anchor refuses
                // this freeze boundary (name the span it would split).
                if trace {
                    if let Some(&(b, e)) = all_spans
                        .iter()
                        .find(|&&(b, e)| (b as usize) < boundary && boundary < e as usize)
                    {
                        eprintln!(
                            "[kvmem-media] freeze skip boundary={boundary} \
                         splits span=[{b},{e})"
                        );
                    }
                }
                continue;
            }
            let media_spans: Vec<(u32, u32)> = all_spans
                .iter()
                .copied()
                .filter(|&(_, e)| (e as usize) <= boundary)
                .collect();
            // G7 Stage E probe: the frozen boundary is media-aligned
            // (never inside a span) and carries the identity.
            if trace {
                let fmt = |d: Option<u64>| {
                    d.map(|x| format!("{x:016x}"))
                        .unwrap_or_else(|| "none".to_string())
                };
                eprintln!(
                    "[kvmem-media] freeze boundary={boundary} digest={} \
                 spans={} inside={}",
                    fmt(media_digest),
                    all_spans.len(),
                    media_spans.len()
                );
            }
            // PIN requires residency: every prefix page must still be
            // bound (a spilled page has no physical handle to pin).
            let n_pages = boundary / PAGE_TOKENS;
            let page_indices: Vec<u32> = match rig.pool.block_table(SlotId(s)) {
                Some(bt) if bt.num_pages() >= n_pages => bt.page_indices()[..n_pages].to_vec(),
                _ => continue,
            };
            if page_indices
                .iter()
                .any(|&p| p == rdna_compute::page_pool::SPILLED_PAGE)
            {
                if rig.gpu.slot_trace() {
                    eprintln!(
                        "[kvmem-prefix] slot {s} boundary {boundary} not frozen: \
                     prefix pages spilled"
                    );
                }
                continue;
            }
            // P3.2 spread policy: capture only at coverage-due
            // frontiers (periodic stride from the request's reuse base).
            #[cfg(feature = "kvmem")]
            if rig.kvmem_checkpoint_policy.is_some() {
                let due = slots[s]
                    .as_ref()
                    .and_then(|f| f.checkpoint_plan.as_ref())
                    .and_then(|p| p.due(boundary as u64).ok().flatten());
                if due.is_none() {
                    continue;
                }
            }
            let freeze_result = (|| -> Result<(), String> {
                // 1. DeltaNet checkpoint FIRST (fail-closed pairing:
                // a cataloged prefix without a restorable recurrent
                // state is unusable for the hybrid model). The live
                // state sits exactly at `boundary` (chunk end).
                let domain = rig
                    .cache_domain
                    .as_ref()
                    .ok_or_else(|| "kvmem freeze: no cache domain".to_string())?;
                let ckpt_pool = rig
                    .model
                    .checkpoint_pool
                    .as_mut()
                    .ok_or_else(|| "kvmem freeze: no checkpoint pool".to_string())?;
                let cid = crate::checkpoint::capture_checkpoint(
                    &mut rig.gpu,
                    ckpt_pool,
                    domain,
                    boundary as u64,
                    &tokens,
                    media_digest,
                    &rig.model.dn_states[s],
                )
                .map_err(|e| format!("kvmem freeze DN capture: {e:?}"))?;
                if cid == crate::checkpoint::CheckpointId::NONE {
                    return Err("kvmem freeze: checkpoint pool ceiling refused capture".to_string());
                }
                // Legacy writes a page to the host archive only when
                // Pressure spills it. Copy the still-resident prefix pages
                // first so the canonical fork owns a complete packed grid
                // and can emit a persistent restart pack.
                {
                    let model = &mut rig.model;
                    if let Some(ctl) = model.kvmem.as_mut() {
                        ctl.archive_resident_prefix(
                            &rig.gpu,
                            &rig.pool,
                            &model.k_arenas,
                            &model.v_arenas,
                            s,
                            boundary as u32,
                            last_frozen as u32,
                        )?;
                    }
                }
                // 2-4. Host freeze + displaced release + page PIN.
                kvmem_catalog_and_pin(
                    &mut rig,
                    s,
                    boundary as u32,
                    &tokens,
                    &page_indices,
                    media_digest,
                    media_spans.clone(),
                )
            })();
            match freeze_result {
                Ok(()) => {
                    if let Some(f) = slots[s].as_mut() {
                        f.last_frozen_boundary = boundary;
                        #[cfg(feature = "kvmem")]
                        if let Some(plan) = f.checkpoint_plan.as_mut() {
                            plan.saved(boundary as u64);
                        }
                    }
                    if rig.gpu.slot_trace() {
                        eprintln!(
                            "[kvmem-prefix] slot {s} froze canonical prefix at \
                         boundary {boundary}"
                        );
                    }
                }
                Err(e) => {
                    eprintln!("[kvmem-prefix] freeze skipped: {e}");
                }
            }
        }
    }
}

pub(super) struct PrefixBoot {
    pub prefix_pin: bool,
    pub prefix_coverage: bool,
    pub prefix_coverage_interval: usize,
    pub checkpoint_policy: Option<(u64, u64)>,
    pub plan: Option<crate::kvmem_ctl::KvMemPlan>,
    /// One start-time KVMem option snapshot. The controller consumes this
    /// instead of reading the process environment.
    pub options: Option<crate::kvmem_ctl::KvMemOptions>,
}

pub(super) fn resolve_prefix_boot(
    cfg: &EngineConfig,
    kvmem_on: bool,
    prefill_chunk: usize,
    spec_rows: usize,
) -> Result<PrefixBoot, String> {
    let options = if kvmem_on {
        Some(crate::kvmem_ctl::KvMemOptions::resolve(
            cfg.kvmem.as_ref(),
            |name| hipfire_config::developer_var(name).ok(),
        )?)
    } else {
        None
    };
    if kvmem_on
        && cfg.kvmem.is_none()
        && options
            .as_ref()
            .is_some_and(|options| options.strategy == crate::kvmem_ctl::RecallStrategy::Legacy)
        && hipfire_config::developer_var("HIPFIRE_KVMEM")
            .ok()
            .is_some_and(|v| matches!(v.trim(), "1" | "on" | "true"))
    {
        return Err(
            "kvmem: HIPFIRE_KVMEM enabled Legacy without a ResolvedConfig; \
             pass EngineConfig.kvmem from the shared config resolver"
                .into(),
        );
    }
    let prefix_pin = kvmem_on
        && hipfire_config::developer_var("HIPFIRE_KVMEM_PREFIX_PIN")
            .ok()
            .is_some_and(|v| v == "1" || v == "on" || v == "true");
    let prefix_coverage = prefix_pin
        && hipfire_config::developer_var("HIPFIRE_KVMEM_PREFIX_COVERAGE")
            .ok()
            .is_some_and(|v| v == "1" || v == "on" || v == "true");
    let prefix_coverage_interval =
        hipfire_config::developer_var("HIPFIRE_KVMEM_PREFIX_COVERAGE_INTERVAL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024)
            .clamp(crate::kvmem_ctl::KV_BLOCK_TOKENS, 65536)
            / crate::kvmem_ctl::KV_BLOCK_TOKENS
            * crate::kvmem_ctl::KV_BLOCK_TOKENS;
    let checkpoint_policy = if prefix_pin {
        match hipfire_config::developer_var("HIPFIRE_KVMEM_CHECKPOINT_POLICY")
            .ok()
            .as_deref()
        {
            None | Some("legacy") | Some("0") => None,
            Some("spread") => {
                let interval = hipfire_config::developer_var("HIPFIRE_KVMEM_CHECKPOINT_INTERVAL")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(4096)
                    .clamp(crate::kvmem_ctl::KV_BLOCK_TOKENS as u64, 1 << 20)
                    / crate::kvmem_ctl::KV_BLOCK_TOKENS as u64
                    * crate::kvmem_ctl::KV_BLOCK_TOKENS as u64;
                let min_gap = hipfire_config::developer_var("HIPFIRE_KVMEM_CHECKPOINT_MIN_GAP")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0)
                    .min(interval);
                Some((interval, min_gap))
            }
            Some(other) => {
                return Err(format!(
                    "HIPFIRE_KVMEM_CHECKPOINT_POLICY={other} invalid (legacy|spread)"
                ))
            }
        }
    } else {
        None
    };
    if checkpoint_policy.is_some() {
        eprintln!(
            "  [hipfire] KVMem checkpoint policy: spread (interval {} tokens, min_gap {})",
            checkpoint_policy.unwrap().0,
            checkpoint_policy.unwrap().1
        );
    }
    if prefix_pin {
        eprintln!(
            "  [hipfire] KVMem prefix PIN enabled: canonical-prefix freeze + \
             shared sealed pages (radix stays off)"
        );
    }
    if prefix_coverage {
        eprintln!(
            "  [hipfire] KVMem prefix coverage capture enabled (interval {} tokens)",
            prefix_coverage_interval
        );
    }
    let plan = if kvmem_on {
        Some(if let Some(config) = &cfg.kvmem {
            let explicit_pages = options.as_ref().and_then(|options| options.plan_pages);
            let budget_tokens = crate::kvmem_ctl::effective_legacy_budget(
                config.budget_tokens as usize,
                crate::kvmem_ctl::implicit_legacy_budget(config),
                cfg.cap_tokens,
                config.gen_reserve_tokens as usize,
            );
            if budget_tokens != config.budget_tokens as usize {
                eprintln!(
                    "  [hipfire] KVMem Legacy history budget default {} clamped to {} \
                     (context {} - reserve {})",
                    config.budget_tokens, budget_tokens, cfg.cap_tokens, config.gen_reserve_tokens
                );
            }
            crate::kvmem_ctl::resolve_legacy_plan(
                cfg.n_slots,
                cfg.cap_tokens,
                prefill_chunk.max(spec_rows),
                budget_tokens,
                config.gen_reserve_tokens as usize,
                explicit_pages,
            )?
        } else {
            crate::kvmem_ctl::resolve_plan(
                cfg.n_slots,
                cfg.cap_tokens,
                prefill_chunk.max(spec_rows),
                options
                    .as_ref()
                    .ok_or("kvmem: missing start-time option snapshot")?,
            )?
        })
    } else {
        None
    };
    Ok(PrefixBoot {
        prefix_pin,
        prefix_coverage,
        prefix_coverage_interval,
        checkpoint_policy,
        plan,
        options,
    })
}

/// Layerwise AR decode / spec-verify step (P1-A). The caller handles
/// admission failure so the scheduler can poison every active slot; every
/// controller/output-owner restore stays inside this helper.
#[allow(clippy::too_many_arguments)]
pub(super) fn layerwise_forward(
    rig: &mut Rig,
    batch: &SlotBatch,
    slots: &[Option<InFlight>],
    work: &[PendingWork],
    graph: &mut SlotDecodeGraph,
    lm_head_skip: &[bool],
    any_verify: bool,
    any_dflash_rows: bool,
) -> Result<HipResult<()>, String> {
    Ok(
        match layerwise_admissible(&rig, batch, &slots, &work, any_verify, any_dflash_rows) {
            Some((slot, session_id, media_pages)) => {
                let mut kvmem = rig.model.kvmem.take();
                let n_fa = rig
                    .model
                    .config
                    .layer_types
                    .iter()
                    .filter(|t| **t == LayerType::FullAttention)
                    .count();
                let fa_layers: Vec<u32> = (0..n_fa as u32).collect();
                // Ensure + DETACH the persistent OUTPUT-scope owner (A'-1:
                // AR decode rides the same OutputPressure infrastructure as
                // MTP verify, design §3.10a borrow resolution): the carrier
                // borrows `&ctl` (keys/policy), so the owner is taken out into
                // an independent local to allow `&mut` accept after the
                // forward, then restored on EVERY exit path. Layerwise-only
                // (ensure fails closed under Legacy, which never reaches
                // this arm). The slot keeps its keep-set / counters.
                let mut output_owner = None;
                // [P4.1② fix] consume the per-request Query gate in the
                // mutable window (before the immutable ctl borrow below).
                let mut force_query = false;
                if let Some(k) = kvmem.as_mut() {
                    let incarnation = k.layerwise_counters(slot).map(|(inc, _)| inc).unwrap_or(0);
                    if let Err(e) = k.ensure_layerwise_output_pressure(
                        slot,
                        session_id,
                        incarnation,
                        &fa_layers,
                    ) {
                        rig.model.kvmem = kvmem;
                        return Err(e);
                    }
                    output_owner = k.take_layerwise_output_pressure(slot);
                    force_query = k.take_layerwise_query_due(slot);
                }
                let ctl = kvmem
                    .as_ref()
                    .expect("admissible layerwise step has a controller");
                // G5: the accepted frontier this restore opens at (the
                // single AR decode row's position). The head aux history is
                // bound from the first FA layer's restored view here, so
                // the first MTP draft (next step) has a selection to take.
                let begin = ctl.fed_tokens(slot).unwrap_or(0) as u64;
                // Fresh page-aligned selection policy over the sealed
                // full pages only (the core picker rejects a partial
                // tail). The slot's shared `kv` store keeps tracking all
                // fed tokens for Legacy and is never borrowed/mutated by
                // Layerwise; this policy is scoped to the query and
                // dropped after the carrier is freed.
                let lw_policy = match ctl.layerwise_policy(ctl.fed_tokens(slot).unwrap_or(0)) {
                    Ok(p) => p,
                    Err(e) => {
                        if let Some(p) = output_owner {
                            if let Some(k) = kvmem.as_mut() {
                                k.restore_layerwise_output_pressure(slot, p);
                            }
                        }
                        rig.model.kvmem = kvmem;
                        return Err(e);
                    }
                };
                // A'-1: AR decode rides engine_carrier_rows (OutputPressure,
                // native mass tracking) with width=1 / query=None / prefill=None
                // ⇒ always scorer Query selection (mirrors MTP verify). The
                // page-aligned PrefillPressure mid-page freeze that lost needles
                // is gone; the OutputPressure window advances accepted_end
                // natively (fed→fed+1) but does not drive selection (prefill=None).
                let built = crate::layerwise_driver::engine_carrier_rows(
                    &mut rig.gpu,
                    ctl,
                    &lw_policy,
                    slot,
                    session_id,
                    n_fa,
                    rig.model.config.n_heads,
                    rig.model.config.n_kv_heads,
                    rig.model.config.head_dim,
                    rig.cap_tokens as u32,
                    "qwen35",
                    1,
                    None,
                    output_owner
                        .as_mut()
                        .expect("admissible layerwise AR step has an output owner"),
                    None,
                    force_query,
                    &media_pages,
                );
                match built {
                    Ok(Some(mut carrier)) => {
                        let fwd = rig.model.forward_step(
                            &mut rig.gpu,
                            batch,
                            &mut rig.pool,
                            graph,
                            lm_head_skip,
                            any_verify,
                            any_dflash_rows,
                            Some(&mut carrier),
                        );
                        // Keep-set contribution (design §3.10a, A'-1): read the
                        // previous resident pool while the carrier still borrows
                        // ctl, then REPLACE with this step's restored pages. AR
                        // decode is always scorer Query (OutputPressure, prefill=None),
                        // so the keep-set is replaced every step — the scorer's
                        // restored view re-admits cold-recalled needle pages, and a
                        // growing union would diverge per-layer extents (bug #8).
                        let head_plane = ctl.head_plane_enabled();
                        // P4.1③ H2a/H4: UpstreamGlobal head-history probe
                        // over the carrier's CAPTURED activation Q tapes,
                        // run BEFORE `free` ends the carrier borrow (the
                        // tapes live in the per-step carrier). Pure host
                        // selection — no KV mutation, no rollback. FirstLayer
                        // (the auto-default) skips this and binds from the
                        // first restored ordinal after free (② G5). Legacy /
                        // AR-only never reach it (byte-neutral).
                        let mtp_history = ctl.layerwise_mtp_history();
                        let probe_aux: Option<
                            Result<crate::layerwise_mtp::AuxiliarySelection, String>,
                        > = if head_plane
                            && mtp_history
                                == crate::layerwise_mtp::LayerwiseMtpHistory::UpstreamGlobal
                        {
                            Some(match carrier.activation() {
                                Some(act) => match ctl.layerwise_stores(slot) {
                                    Some((keys, _)) => crate::layerwise_probe::probe_head_history(
                                        &mut rig.gpu,
                                        act,
                                        &lw_policy,
                                        keys,
                                        rig.model.config.n_heads as u32,
                                        rig.model.config.n_kv_heads as u32,
                                        rig.model.config.head_dim as u32,
                                        0.0,
                                        crate::layerwise_layout::LAYERWISE_BLOCK_TOKENS as u32,
                                        begin,
                                    ),
                                    None => {
                                        Err("layerwise probe: slot has no keys store".to_string())
                                    }
                                },
                                None => {
                                    Err("layerwise probe: UpstreamGlobal carrier has no activation"
                                        .to_string())
                                }
                            })
                        } else {
                            None
                        };
                        let prev_keep: Vec<Vec<u64>> = ctl
                            .layerwise_keep_set(slot)
                            .map(|s| s.to_vec())
                            .unwrap_or_default();
                        let restored_res = carrier.restored_keep_set();
                        // G5: the first FA ordinal's restored canonical ids
                        // for the FirstLayer head aux binding, captured before
                        // `free` ends the carrier borrow. `None` on a restore
                        // error (surfaced through `keep_res` below). Cloned out
                        // so `restored_res` still moves into `keep_res` (owned Err).
                        let aux_first: Option<Vec<u64>> =
                            restored_res.as_ref().ok().and_then(|r| r.first().cloned());
                        let keep_res = restored_res.map(|restored| {
                            // A'-1: AR decode is always scorer Query (prefill=None
                            // ⇒ no Pressure refinement), so REPLACE the keep-set
                            // every step (fresh current-Q re-selection, consistent
                            // per-layer extent) exactly like the MTP verify regime
                            // (bug #8). UNION with a non-empty prior diverges the
                            // per-layer resident counts and trips the catalog
                            // geometry guard; the scorer's restored view already
                            // re-admits cold-recalled needle pages each step, so a
                            // growing union is redundant. Always-Query also satisfies
                            // the continuation fix (current question re-scored).
                            crate::layerwise_driver::combine_keep_set(
                                &prev_keep, &restored, true, n_fa,
                            )
                        });
                        // Accept the captured native mass into the detached
                        // OUTPUT-scope owner at the confirmed AR prefix (accepted=1:
                        // the single decode row is always accepted). Advances the
                        // window's accepted_end to fed+1 for the next step's begin.
                        // The &mut owner is independent of the carrier's ctl borrow,
                        // so this runs before free. AR publishes immediately (no
                        // MTP-style deferred pending_output: the prefix is known).
                        let accept_res = match output_owner.as_mut() {
                            Some(p) => carrier
                                .accept_output_pressure(p, session_id + 1, begin, 1)
                                .map(|_| false),
                            None => Ok(false),
                        };
                        // Release the carrier exactly once (its GPU
                        // buffers have no Drop). `free` consumes it,
                        // ending its borrow of `kvmem`/`lw_policy` before
                        // the mutable counter bump / keep-set / restore.
                        let free_res = carrier.free(&mut rig.gpu);
                        // G5: FirstLayer head aux binding (pure-host, after the
                        // carrier borrow ends). `head_plane` ⇒ Layerwise + MTP ⇒
                        // FirstLayer under ②. `Ok(None)` when the head plane is
                        // inactive (AR-only) or the restore errored (handled below).
                        let aux_res = if head_plane {
                            if mtp_history
                                == crate::layerwise_mtp::LayerwiseMtpHistory::UpstreamGlobal
                            {
                                probe_aux.transpose()
                            } else {
                                aux_first
                                    .as_deref()
                                    .map(|first| {
                                        crate::layerwise_mtp::bind_first_layer_auxiliary(
                                            first,
                                            begin,
                                            crate::layerwise_layout::LAYERWISE_BLOCK_TOKENS as u64,
                                        )
                                    })
                                    .transpose()
                            }
                        } else {
                            Ok(None)
                        };
                        // Return the detached owner on EVERY path so the
                        // session-lifetime pressure state persists.
                        if let Some(p) = output_owner {
                            if let Some(k) = kvmem.as_mut() {
                                k.restore_layerwise_output_pressure(slot, p);
                            }
                        }
                        if let Some(e) = accept_res.err().or_else(|| free_res.err()) {
                            rig.model.kvmem = kvmem;
                            return Err(e);
                        }
                        let new_keep = match keep_res {
                            Ok(k) => k,
                            Err(e) => {
                                rig.model.kvmem = kvmem;
                                return Err(e);
                            }
                        };
                        let aux_sel = match aux_res {
                            Ok(s) => s,
                            Err(e) => {
                                rig.model.kvmem = kvmem;
                                return Err(e);
                            }
                        };
                        if let Some(k) = kvmem.as_mut() {
                            k.advance_layerwise_seq(slot);
                            k.set_layerwise_keep_set(slot, new_keep);
                            if let Some(s) = aux_sel {
                                k.set_aux_selection(slot, s);
                            }
                        }
                        rig.model.kvmem = kvmem;
                        fwd
                    }
                    // Before the fixed-block floor (see
                    // `layerwise_floor_tokens`): restore the controller +
                    // owner and run Legacy; no transaction.
                    Ok(None) => {
                        if let Some(p) = output_owner {
                            if let Some(k) = kvmem.as_mut() {
                                k.restore_layerwise_output_pressure(slot, p);
                            }
                        }
                        rig.model.kvmem = kvmem;
                        rig.model.forward_step(
                            &mut rig.gpu,
                            batch,
                            &mut rig.pool,
                            graph,
                            lm_head_skip,
                            any_verify,
                            any_dflash_rows,
                            None,
                        )
                    }
                    Err(e) => {
                        if let Some(p) = output_owner {
                            if let Some(k) = kvmem.as_mut() {
                                k.restore_layerwise_output_pressure(slot, p);
                            }
                        }
                        rig.model.kvmem = kvmem;
                        return Err(e);
                    }
                }
            }
            None => {
                // P4.1② (design §A.6 / §B D1.4②): AR admission rejects
                // `any_verify`, so a Layerwise MTP verify span lands here.
                // Admit it before falling back to Legacy: multi-row trunk
                // accepted-only write-through + drop-evict + restore over
                // the verify batch (Path B), through the OUTPUT-scope owner
                // (partial acceptance). The accepted prefix is published
                // post-sampling in `mtp_verify_accept_step`, not here.
                if let Some((slot, session_id, width)) = layerwise_mtp_verify_admissible(
                    &rig,
                    batch,
                    &slots,
                    &work,
                    any_verify,
                    any_dflash_rows,
                ) {
                    let mut kvmem = rig.model.kvmem.take();
                    let n_fa = rig
                        .model
                        .config
                        .layer_types
                        .iter()
                        .filter(|t| **t == LayerType::FullAttention)
                        .count();
                    let fa_layers: Vec<u32> = (0..n_fa as u32).collect();
                    // Ensure + DETACH the persistent OUTPUT-scope owner
                    // (the MTP-verify sibling of the AR prefill owner). A
                    // verify span accepts a PREFIX of its rows, which the
                    // all-or-nothing prefill owner cannot express. Taken
                    // out so `begin_scope`/`prepare_scope`/`accept` use
                    // `&mut OutputPressure` independently of the carrier's
                    // `&ctl` borrow; restored on EVERY exit path.
                    let mut output_owner = None;
                    // [P4.1② fix] consume the per-request Query gate in the
                    // mutable window (before the immutable ctl borrow).
                    let mut force_query = false;
                    if let Some(k) = kvmem.as_mut() {
                        let incarnation =
                            k.layerwise_counters(slot).map(|(inc, _)| inc).unwrap_or(0);
                        if let Err(e) = k.ensure_layerwise_output_pressure(
                            slot,
                            session_id,
                            incarnation,
                            &fa_layers,
                        ) {
                            rig.model.kvmem = kvmem;
                            return Err(e);
                        }
                        output_owner = k.take_layerwise_output_pressure(slot);
                        force_query = k.take_layerwise_query_due(slot);
                    }
                    let ctl = kvmem
                        .as_ref()
                        .expect("admissible layerwise verify has a controller");
                    // Native consumed-row frontier the output window opens
                    // at (== the verify span's first position); the accepted
                    // prefix advances it by `result.advance` post-sampling.
                    let begin = ctl.fed_tokens(slot).unwrap_or(0) as u64;
                    let request = session_id + 1;
                    let lw_policy = match ctl.layerwise_policy(ctl.fed_tokens(slot).unwrap_or(0)) {
                        Ok(p) => p,
                        Err(e) => {
                            if let Some(p) = output_owner {
                                if let Some(k) = kvmem.as_mut() {
                                    k.restore_layerwise_output_pressure(slot, p);
                                }
                            }
                            rig.model.kvmem = kvmem;
                            return Err(e);
                        }
                    };
                    let built = crate::layerwise_driver::engine_carrier_rows(
                        &mut rig.gpu,
                        ctl,
                        &lw_policy,
                        slot,
                        session_id,
                        n_fa,
                        rig.model.config.n_heads,
                        rig.model.config.n_kv_heads,
                        rig.model.config.head_dim,
                        rig.cap_tokens as u32,
                        "qwen35",
                        width,
                        None,
                        output_owner
                            .as_mut()
                            .expect("admissible layerwise verify has an output owner"),
                        None,
                        force_query,
                        &[],
                    );
                    match built {
                        Ok(Some(mut carrier)) => {
                            let fwd = rig.model.forward_step(
                                &mut rig.gpu,
                                batch,
                                &mut rig.pool,
                                graph,
                                lm_head_skip,
                                any_verify,
                                any_dflash_rows,
                                Some(&mut carrier),
                            );
                            let head_plane = ctl.head_plane_enabled();
                            // P4.1③ H2a/H4: UpstreamGlobal head-history probe
                            // over the carrier's CAPTURED activation Q tapes,
                            // run BEFORE `free` (the tapes live in the per-step
                            // carrier). Pure host selection — no KV mutation, no
                            // rollback. FirstLayer binds from the first restored
                            // ordinal after free (② G5). Legacy / AR-only never
                            // reach it (byte-neutral).
                            let mtp_history = ctl.layerwise_mtp_history();
                            let probe_aux: Option<
                                Result<crate::layerwise_mtp::AuxiliarySelection, String>,
                            > = if head_plane
                                && mtp_history
                                    == crate::layerwise_mtp::LayerwiseMtpHistory::UpstreamGlobal
                            {
                                Some(match carrier.activation() {
                                    Some(act) => {
                                        match ctl.layerwise_stores(slot) {
                                            Some((keys, _)) => {
                                                crate::layerwise_probe::probe_head_history(
                                                    &mut rig.gpu,
                                                    act,
                                                    &lw_policy,
                                                    keys,
                                                    rig.model.config.n_heads as u32,
                                                    rig.model.config.n_kv_heads as u32,
                                                    rig.model.config.head_dim as u32,
                                                    0.0,
                                                    crate::layerwise_layout::LAYERWISE_BLOCK_TOKENS
                                                        as u32,
                                                    begin,
                                                )
                                            }
                                            None => Err("layerwise probe: slot has no keys store"
                                                .to_string()),
                                        }
                                    }
                                    None => Err(
                                        "layerwise probe: UpstreamGlobal carrier has no activation"
                                            .to_string(),
                                    ),
                                })
                            } else {
                                None
                            };
                            let restored_res = carrier.restored_keep_set();
                            // G5: the first FA ordinal's restored canonical ids
                            // for the FirstLayer head aux binding, captured before
                            // `free` ends the carrier borrow. `None` on a restore
                            // error (surfaced through `keep_res` below). Cloned out
                            // so `restored_res` still moves into `keep_res` (owned Err).
                            let aux_first: Option<Vec<u64>> =
                                restored_res.as_ref().ok().and_then(|r| r.first().cloned());
                            // Keep-set (design §3.10a, ninfer REPLACE): the
                            // MTP composite verify regime forces REPLACE
                            // (pressure = true, empty prev). The trunk+aux
                            // planes select asymmetric page ids, so a growing
                            // union diverges the per-layer resident counts and
                            // trips the catalog's cross-layer extent invariant
                            // (bug #8). The restored view already admits the
                            // scorer's full re-selection (cold recall included).
                            let keep_res = restored_res.map(|restored| {
                                crate::layerwise_driver::combine_keep_set(
                                    &[],
                                    &restored,
                                    true,
                                    n_fa,
                                )
                            });
                            // Extract the OWNED acceptance snapshot before
                            // freeing the carrier: the accepted prefix
                            // (`result.advance`) is known only post-sampling,
                            // so the publish runs in `mtp_verify_accept_step`
                            // (after `feed_rows`), outside this scoped take.
                            let snap_res = carrier.extract_output_acceptance(request, begin);
                            let free_res = carrier.free(&mut rig.gpu);
                            // G5: FirstLayer head aux binding (pure-host, after the
                            // carrier borrow ends). `head_plane` ⇒ Layerwise + MTP ⇒
                            // FirstLayer under ②. The head consume (post-accept) and
                            // the next span's draft take this binding. `Ok(None)` when
                            // the head plane is inactive or the restore errored.
                            let aux_res = if head_plane {
                                if mtp_history
                                    == crate::layerwise_mtp::LayerwiseMtpHistory::UpstreamGlobal
                                {
                                    probe_aux.transpose()
                                } else {
                                    aux_first
                                        .as_deref()
                                        .map(|first| {
                                            crate::layerwise_mtp::bind_first_layer_auxiliary(
                                                first,
                                                begin,
                                                crate::layerwise_layout::LAYERWISE_BLOCK_TOKENS
                                                    as u64,
                                            )
                                        })
                                        .transpose()
                                }
                            } else {
                                Ok(None)
                            };
                            if let Some(p) = output_owner {
                                if let Some(k) = kvmem.as_mut() {
                                    k.restore_layerwise_output_pressure(slot, p);
                                }
                            }
                            if let Some(e) = free_res.err() {
                                rig.model.kvmem = kvmem;
                                return Err(e);
                            }
                            let new_keep = match keep_res {
                                Ok(k) => k,
                                Err(e) => {
                                    rig.model.kvmem = kvmem;
                                    return Err(e);
                                }
                            };
                            let snap = match snap_res {
                                Ok(s) => s,
                                Err(e) => {
                                    rig.model.kvmem = kvmem;
                                    return Err(e);
                                }
                            };
                            let aux_sel = match aux_res {
                                Ok(s) => s,
                                Err(e) => {
                                    rig.model.kvmem = kvmem;
                                    return Err(e);
                                }
                            };
                            if let Some(k) = kvmem.as_mut() {
                                k.advance_layerwise_seq(slot);
                                k.set_layerwise_keep_set(slot, new_keep);
                                if let Some(s) = snap {
                                    k.set_layerwise_pending_output(slot, s);
                                }
                                if let Some(s) = aux_sel {
                                    k.set_aux_selection(slot, s);
                                }
                            }
                            rig.model.kvmem = kvmem;
                            fwd
                        }
                        // Before the fixed-block floor (see
                        // `layerwise_floor_tokens`): restore the owner +
                        // controller and run Legacy; the verify still
                        // commits via `feed_rows`, no output publish.
                        Ok(None) => {
                            if let Some(p) = output_owner {
                                if let Some(k) = kvmem.as_mut() {
                                    k.restore_layerwise_output_pressure(slot, p);
                                }
                            }
                            rig.model.kvmem = kvmem;
                            rig.model.forward_step(
                                &mut rig.gpu,
                                batch,
                                &mut rig.pool,
                                graph,
                                lm_head_skip,
                                any_verify,
                                any_dflash_rows,
                                None,
                            )
                        }
                        Err(e) => {
                            if let Some(p) = output_owner {
                                if let Some(k) = kvmem.as_mut() {
                                    k.restore_layerwise_output_pressure(slot, p);
                                }
                            }
                            rig.model.kvmem = kvmem;
                            return Err(e);
                        }
                    }
                } else {
                    rig.model.forward_step(
                        &mut rig.gpu,
                        batch,
                        &mut rig.pool,
                        graph,
                        lm_head_skip,
                        any_verify,
                        any_dflash_rows,
                        None,
                    )
                }
            }
        },
    )
}

/// KVMem-native canonical-prefix reuse (P4.3). Returns the reused token
/// frontier when the lookup converts, or `None` after unwinding to cold.
pub(super) fn kvmem_prefix_reuse(
    rig: &mut Rig,
    slot: SlotId,
    session_id: SessionId,
    req: &SubmitRequest,
) -> Option<usize> {
    let trace = kvmem_trace(rig);
    if req.force_cold {
        if trace {
            eprintln!("[kvmem-prefix] force_cold: canonical prefix reuse skipped");
        }
        return None;
    }
    #[cfg(feature = "kvmem")]
    let req_media: Option<u64> = req
        .visual_data
        .as_ref()
        .map(|vd| vd.media_digest())
        .or_else(|| rig.sessions.get(session_id).and_then(|s| s.media_digest));
    // G7 Stage C: the reuse frontier goes through the Stage A media anchor.
    // The restored prefix must never split a vision span of the incoming
    // prompt; a frozen checkpoint already satisfies this by construction, so a
    // failure here is an invariant violation and the reuse closes to cold.
    #[cfg(feature = "kvmem")]
    let req_media_spans: Vec<crate::kvmem_media::MediaSpan> = {
        let n_visual = req
            .visual_data
            .as_ref()
            .map(|vd| vd.n_visual_tokens)
            .or_else(|| rig.sessions.get(session_id).map(|s| s.media_n_visual))
            .unwrap_or(0);
        match (req_media, rig.tokenizer.special_token_id("<|image_pad|>")) {
            (Some(_), Some(pad)) if n_visual > 0 => {
                crate::kvmem_media::spans_from_pad_runs(&req.prompt_tokens, pad, n_visual)
            }
            _ => Vec::new(),
        }
    };
    #[cfg(feature = "kvmem")]
    let anchor_reuse = |frontier: u32| -> Result<u32, String> {
        if req_media_spans.is_empty() {
            return Ok(frontier);
        }
        let aligned = crate::kvmem_media::media_anchored_frontier(
            0,
            frontier as u64,
            req.prompt_tokens.len() as u64,
            &req_media_spans,
            crate::kvmem_ctl::KV_BLOCK_TOKENS as u64,
        )
        .map_err(|e| format!("kvmem reuse: media anchor: {e}"))?;
        if aligned != frontier as u64 {
            return Err(format!(
                "kvmem reuse: media anchor moves frontier {frontier} -> {aligned}"
            ));
        }
        // G7 Stage E probe: the live media replay anchor accepted this
        // frontier (block-aligned and not splitting any image span).
        if trace {
            let fmt = |d: Option<u64>| {
                d.map(|x| format!("{x:016x}"))
                    .unwrap_or_else(|| "none".to_string())
            };
            eprintln!(
                "[kvmem-media] reuse anchor accept digest={} spans={} candidate={frontier} \
             aligned={aligned}",
                fmt(req_media),
                req_media_spans.len()
            );
        }
        Ok(frontier)
    };
    #[cfg(feature = "kvmem")]
    if rig.kvmem_prefix_pin {
        // 期3 G3 P1.1: a cold pack hit publishes a catalog entry and may
        // displace others; release their GPU cache refs outside the borrow.
        let mut cold_displaced = Vec::new();
        let hit = rig.model.kvmem.as_mut().and_then(|ctl| {
            let hit = ctl.prefix_lookup_media(&req.prompt_tokens, req_media);
            cold_displaced = ctl.take_cold_displaced();
            hit
        });
        for d in cold_displaced {
            if let Some(pp) = rig.pool.page_pool_mut() {
                for &phys in &d.gpu_pages {
                    let _ = pp.release_cache_ref(phys);
                }
            }
        }
        if let Some(ckpt_id) = hit {
            let reuse = (|| -> Result<usize, String> {
                let ctl = rig
                    .model
                    .kvmem
                    .as_mut()
                    .ok_or_else(|| "kvmem reuse: no controller".to_string())?;
                // 1. Host fork: validates the FULL provenance (geometry +
                // fingerprint + token proof) before any mutation.
                // Stage C: the candidate frontier must pass the media anchor
                // BEFORE the fork mutates the slot; a violation closes the
                // reuse to a cold prefill (the slot is still the earlier
                // reset state here).
                let candidate = ctl
                    .prefix_checkpoint(ckpt_id)
                    .map(|c| c.frontier)
                    .ok_or_else(|| format!("kvmem reuse: checkpoint {ckpt_id} missing"))?;
                anchor_reuse(candidate)?;
                let mut frontier = ctl.fork_prefix_into(slot.0, ckpt_id, &req.prompt_tokens)?;
                let mut reuse_id = ckpt_id;
                // G3 cold-reuse fix: a page-less import (restart pack load or an
                // admission-demoted entry) has no resident prefix pages for a
                // first-step Query selection to score. A Pressure import with a
                // NEW question would reselect over empty residency and drop the
                // whole prefix; keep the source's selected views instead and
                // suppress the first-step Query.
                let page_less = ctl
                    .prefix_checkpoint(reuse_id)
                    .map(|c| c.gpu_pages.is_empty())
                    .unwrap_or(false);
                // 1b. P2.2a/P2.2b fenced-import admission (ninfer
                // `prepare_import_start`): the restored exact prefix is measured
                // against the layerwise window capacity B+R. A pressure import
                // arms a one-shot selection budget for the first Layerwise step.
                // When the FULL checkpoint would have to narrow, fall back to the
                // longest coverage checkpoint at or below the narrowed extent
                // (its DeltaNet state exists at its own page-aligned frontier);
                // no such checkpoint closes the reuse to cold.
                if ctl.strategy().is_layerwise() {
                    // 期2.3: the import decision carries the frontend query
                    // identity — KeepSelected (already effective at the
                    // source) suppresses the per-request Query; a NEW proven
                    // question arms it. Unproven spans keep the historical
                    // behaviour (no forced Query on the cold-reuse path).
                    let mut keep_selected = false;
                    let admission = (|| -> Result<Option<u32>, String> {
                        let prev = ctl.prefix_original_query(reuse_id);
                        let query = if page_less { prev } else { req.query };
                        let decision = ctl
                            .import_window(
                                req.prompt_tokens.len() as u64,
                                req.max_tokens as u64,
                                frontier,
                                query,
                                prev,
                            )
                            .map_err(|e| format!("kvmem reuse: import admission: {e}"))?;
                        keep_selected = decision.keep_selected;
                        let start = decision.start;
                        if !start.pressure() {
                            return Ok(None);
                        }
                        if start.restore_tokens < frontier as u64 {
                            let cap = start.restore_tokens as u32;
                            let shorter =
                                ctl.prefix_lookup_capped_media(&req.prompt_tokens, cap, req_media);
                            let Some(sid) = shorter.filter(|&sid| sid != reuse_id) else {
                                return Err(format!(
                                    "kvmem reuse: import pressure narrows {} -> {} tokens \
                                 and no coverage checkpoint fits",
                                    frontier, start.restore_tokens
                                ));
                            };
                            let candidate = ctl
                                .prefix_checkpoint(sid)
                                .map(|c| c.frontier)
                                .ok_or_else(|| format!("kvmem reuse: checkpoint {sid} missing"))?;
                            anchor_reuse(candidate)?;
                            frontier = ctl.fork_prefix_into(slot.0, sid, &req.prompt_tokens)?;
                            reuse_id = sid;
                            let prev = ctl.prefix_original_query(reuse_id);
                            let query = if page_less { prev } else { req.query };
                            let decision = ctl
                                .import_window(
                                    req.prompt_tokens.len() as u64,
                                    req.max_tokens as u64,
                                    frontier,
                                    query,
                                    prev,
                                )
                                .map_err(|e| format!("kvmem reuse: import retry: {e}"))?;
                            keep_selected = decision.keep_selected;
                            let retry = decision.start;
                            if retry.pressure() {
                                if retry.restore_tokens < frontier as u64 {
                                    return Err(format!(
                                    "kvmem reuse: coverage checkpoint {} narrows {} -> {} tokens",
                                    reuse_id, frontier, retry.restore_tokens
                                ));
                                }
                                let blocks = (retry.selection_budget
                                    / crate::kvmem_ctl::KV_BLOCK_TOKENS as u64)
                                    as u32;
                                return Ok(Some(blocks));
                            }
                            return Ok(None);
                        }
                        let blocks = (start.selection_budget
                            / crate::kvmem_ctl::KV_BLOCK_TOKENS as u64)
                            as u32;
                        Ok(Some(blocks))
                    })();
                    match admission {
                        Ok(Some(blocks)) => ctl
                            .set_layerwise_import_blocks(slot.0, blocks)
                            .map_err(|e| format!("kvmem reuse: import budget: {e}"))?,
                        Ok(None) => {}
                        Err(e) => {
                            // The fork already reset this slot to the checkpoint
                            // prefix; a refused admission must hand it back cold.
                            let _ = ctl.reset_slot(slot.0);
                            let _ = rig.model.dn_states[slot.0].reset(&mut rig.gpu);
                            return Err(e);
                        }
                    }
                    // 期2.3: arm the first-step Query only for a NEW proven
                    // question. KeepSelected leaves it clear so the source's
                    // selected views stand; an unproven span keeps the
                    // historical (no forced Query on the cold-reuse path).
                    let due = req.query.is_some() && !keep_selected && !page_less;
                    if req.query.is_some() {
                        ctl.set_layerwise_query_due(slot.0, due);
                    }
                    if trace {
                        eprintln!(
                        "[kvmem-query] reuse slot {} query={:?} page_less={} keep_selected={} -> due={}",
                        slot.0,
                        req.query,
                        page_less,
                        keep_selected,
                        due
                    );
                    }
                }
                let gpu_pages = ctl
                    .prefix_checkpoint(reuse_id)
                    .map(|c| c.gpu_pages.clone())
                    .unwrap_or_default();
                // 2. DeltaNet restore (fail-closed pairing, spec §5.4 S4):
                // shared KV must never pair with a zeroed/partial recurrent
                // state. On any failure below, reset BOTH halves.
                let domain = rig
                    .cache_domain
                    .as_ref()
                    .ok_or_else(|| "kvmem reuse: no cache domain".to_string())?;
                let ckpt_pool = rig
                    .model
                    .checkpoint_pool
                    .as_mut()
                    .ok_or_else(|| "kvmem reuse: no checkpoint pool".to_string())?;
                // G7 Stage C: the recurrent state is keyed by the checkpoint's
                // media identity. The catalog gate proved it equals the
                // request's digest when the prefix covers an image span; a
                // pre-image/plain prefix carries no media and keeps the
                // historical key.
                let ckpt_media = ctl.prefix_checkpoint(reuse_id).and_then(|c| c.media_digest);
                let fp = crate::checkpoint::checkpoint_fingerprint(
                    &req.prompt_tokens[..frontier as usize],
                    ckpt_media,
                );
                // 期3 G3 P1.1 dn-persist: rebuild the recurrent checkpoint from
                // the pack sidecar when this process has no pool entry (restart).
                if ckpt_pool.peek(domain, frontier as u64, fp).is_none() {
                    if let Some(pack) = ctl.prefix_pack_path(reuse_id) {
                        let dn_path = std::path::PathBuf::from(format!("{}.dn", pack.display()));
                        match std::fs::read(&dn_path) {
                            Ok(bytes) => {
                                match crate::speculative::DeltaNetSnapshot::new_for(
                                    &mut rig.gpu,
                                    &rig.model.dn_states[slot.0],
                                ) {
                                    Ok(mut snap) => match snap.import_host(&rig.gpu, &bytes) {
                                        Ok(()) => {
                                            let (cid, displaced) = ckpt_pool.insert(
                                                domain.clone(),
                                                frontier as u64,
                                                fp,
                                                snap,
                                            );
                                            for d in displaced {
                                                d.free_gpu(&mut rig.gpu);
                                            }
                                            if cid != crate::checkpoint::CheckpointId::NONE
                                                && hipfire_config::developer_var(
                                                    "HIPFIRE_KVMEM_TRACE",
                                                )
                                                .ok()
                                                .is_some_and(|v| {
                                                    v == "1" || v == "on" || v == "true"
                                                })
                                            {
                                                eprintln!(
                                                "[kvmem-pack] DN restored from sidecar frontier={frontier}"
                                            );
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("[kvmem-pack] DN sidecar rejected: {e}")
                                        }
                                    },
                                    Err(e) => {
                                        eprintln!("[kvmem-pack] DN sidecar alloc failed: {e:?}")
                                    }
                                }
                            }
                            Err(e) => eprintln!("[kvmem-pack] DN sidecar missing: {e}"),
                        }
                    }
                }
                let restored = ckpt_pool
                    .peek(domain, frontier as u64, fp)
                    .map(|snapshot| {
                        snapshot
                            .restore_to(&mut rig.model.dn_states[slot.0], &mut rig.gpu)
                            .is_ok()
                    })
                    .unwrap_or(false);
                if !restored {
                    let _ = ctl.reset_slot(slot.0);
                    let _ = rig.model.dn_states[slot.0].reset(&mut rig.gpu);
                    return Err("kvmem reuse: DeltaNet checkpoint restore failed".to_string());
                }
                // 3. Share the pinned pages into the (empty) block table.
                // 期3 G3 P1.1: a restart-loaded pack carries no pinned GPU
                // pages. Layerwise restores pages from the sealed archive on
                // demand (carrier restore), so the share is skipped; Legacy
                // has no cold restore path and fails closed to a cold prefill.
                if gpu_pages.is_empty() {
                    if ctl.strategy().is_layerwise() {
                        // G3 cold-reuse fix: a page-less import has no bound
                        // prefix pages, so the suffix prefill would attend over
                        // unbound rows and lose the imported context.
                        // Materialize the prefix from the archived store into
                        // pool pages first, then pin them so later reuses are
                        // hot.
                        let staged = ctl.stage_page_less_import(
                            &rig.gpu,
                            &mut rig.pool,
                            &rig.model.k_arenas,
                            &rig.model.v_arenas,
                            slot.0,
                            frontier,
                        );
                        let pages = match staged {
                            Ok(pages) => pages,
                            Err(e) => {
                                let _ = ctl.reset_slot(slot.0);
                                let _ = rig.model.dn_states[slot.0].reset(&mut rig.gpu);
                                return Err(format!(
                                    "kvmem reuse: cold prefix staging failed: {e}"
                                ));
                            }
                        };
                        let mut taken: Vec<u32> = Vec::new();
                        let mut ref_err = None;
                        if let Some(pp) = rig.pool.page_pool_mut() {
                            for &phys in &pages {
                                match pp.add_cache_ref(phys) {
                                    Ok(()) => taken.push(phys),
                                    Err(e) => {
                                        ref_err = Some(e);
                                        break;
                                    }
                                }
                            }
                        } else {
                            ref_err = Some("not paged".to_string());
                        }
                        if let Some(e) = ref_err {
                            if let Some(pp) = rig.pool.page_pool_mut() {
                                for &phys in &taken {
                                    let _ = pp.release_cache_ref(phys);
                                }
                            }
                            let _ = ctl.reset_slot(slot.0);
                            let _ = rig.model.dn_states[slot.0].reset(&mut rig.gpu);
                            return Err(format!("kvmem reuse: cold prefix pin failed: {e}"));
                        }
                        if let Some(ckpt) = ctl.prefix_checkpoint_mut(reuse_id) {
                            ckpt.gpu_pages = pages;
                        }
                        ctl.note_prefix_restore_hit();
                    } else {
                        // Legacy has no pinned-page cold restore arm. Restore
                        // a recency/sink working set into an all-spilled block
                        // table so the suffix prefill can attend the imported
                        // prefix. Do not pin these pages: Legacy Pressure must
                        // remain free to evict them, and the checkpoint keeps
                        // no page list so a later reuse re-materializes from
                        // the sealed pack.
                        match ctl.stage_legacy_page_less_import(
                            &rig.gpu,
                            &mut rig.pool,
                            &rig.model.k_arenas,
                            &rig.model.v_arenas,
                            slot.0,
                            frontier,
                        ) {
                            Ok(pages) => {
                                ctl.note_prefix_restore_hit();
                                // The cold fork above rebuilt the slot state and
                                // dropped the request query. Re-arm it here, clipped
                                // to the imported frontier: rows before the frontier
                                // are already in the imported raw, so only the new
                                // suffix rows can be captured and scored. Without
                                // this, Legacy never starts a Query and the spilled
                                // middle pages stay unmasked forever (K7 recall gap).
                                if let Some((begin, end)) = req.query {
                                    let begin = begin.max(u64::from(frontier));
                                    let trimmed = (begin < end).then_some((begin, end));
                                    if let Err(e) = ctl.set_request_query(slot.0, trimmed) {
                                        let _ = ctl.reset_slot(slot.0);
                                        let _ = rig.model.dn_states[slot.0].reset(&mut rig.gpu);
                                        return Err(format!(
                                            "kvmem reuse: Legacy cold Query re-arm failed: {e}"
                                        ));
                                    }
                                    if trace {
                                        eprintln!(
                                            "[kvmem-legacy] cold pack query slot={} \
                                             span={trimmed:?} frontier={frontier}",
                                            slot.0
                                        );
                                    }
                                }
                                if trace {
                                    eprintln!(
                                        "[kvmem-legacy] cold pack reuse slot={} \
                                         frontier={frontier} resident_pages={pages}",
                                        slot.0
                                    );
                                }
                            }
                            Err(e) => {
                                let _ = ctl.reset_slot(slot.0);
                                let _ = rig.model.dn_states[slot.0].reset(&mut rig.gpu);
                                return Err(format!(
                                    "kvmem reuse: Legacy cold prefix staging failed: {e}"
                                ));
                            }
                        }
                    }
                } else if rig.pool.share_published_pages(slot, &gpu_pages).is_err() {
                    let _ = ctl.reset_slot(slot.0);
                    let _ = rig.model.dn_states[slot.0].reset(&mut rig.gpu);
                    return Err("kvmem reuse: GPU page share failed".to_string());
                }
                // 4. 期4: restore the DFlash2 draft conditioning image that
                //    rides with this checkpoint (nInfer native full-state
                //    serialization; logical ring + captured watermarks). A
                //    missing/unrestorable image reseeds the draft from scratch
                //    — acceptance degrades, output stays verify-exact.
                if let Some(shared) = rig.model.dflash.as_ref() {
                    let restored = {
                        match ctl
                            .prefix_checkpoint(reuse_id)
                            .and_then(|c| c.draft_image.as_ref())
                        {
                            Some(img) => {
                                if rig.model.dflash_states[slot.0].is_none() {
                                    match crate::dflash_slot::new_dflash_slot_state(
                                        &mut rig.gpu,
                                        shared,
                                        &rig.model.config,
                                        &rig.model.dn_states[slot.0],
                                    ) {
                                        Ok(st) => rig.model.dflash_states[slot.0] = Some(st),
                                        Err(e) => {
                                            eprintln!("[kvmem-draft-image] state alloc failed: {e}")
                                        }
                                    }
                                }
                                match rig.model.dflash_states[slot.0].as_mut() {
                                    Some(st) => match crate::dflash_slot::dflash_restore_image(
                                        &mut rig.gpu,
                                        shared,
                                        st,
                                        img,
                                        session_id.0,
                                        frontier as usize,
                                    ) {
                                        Ok(()) => true,
                                        Err(e) => {
                                            eprintln!("[kvmem-draft-image] restore skipped: {e}");
                                            false
                                        }
                                    },
                                    None => false,
                                }
                            }
                            None => false,
                        }
                    };
                    if restored {
                        ctl.note_draft_image_hit();
                    } else {
                        ctl.note_draft_image_miss();
                        if let Some(st) = rig.model.dflash_states[slot.0].as_mut() {
                            crate::dflash_slot::reset_dflash_slot(st, session_id.0);
                        }
                    }
                }
                Ok(frontier as usize)
            })();
            match reuse {
                Ok(frontier) => {
                    if rig.gpu.slot_trace() {
                        eprintln!(
                            "[slot-trace] KVMem prefix HIT: reused {}/{} tokens",
                            frontier,
                            req.prompt_tokens.len()
                        );
                    }
                    return Some(frontier);
                }
                Err(e) => {
                    eprintln!("[kvmem-prefix] reuse did not convert: {e} — cold prefill");
                }
            }
        }
    }
    None
}

/// Feed every KV row this step wrote into the host store, including the
/// AR-decode head aux co-archive. Returns an error for the caller to poison.
pub(super) fn kvmem_feed_step(
    rig: &mut Rig,
    batch: &SlotBatch,
    slots: &[Option<InFlight>],
    work: &[PendingWork],
    spec_drafts: &[Option<SpecDraftRows>],
    n: usize,
) -> Result<(), String> {
    // ── KVMem statistics feed (HIPFIRE_KVMEM) ───────────────────────
    // Every KV row this step wrote (prefill chunk or plain decode row)
    // is summarized into the slot's host store now, from the taps the
    // forward captured. Verify steps are fed in the accept path
    // instead — only the committed prefix of their rows survives, and
    // its size is not known until after the sampler reads the hidden
    // rows. Fed positions are contiguous by construction: a slot's
    // batch rows always start at the store's current length.
    if rig.model.kvmem.is_none() {
        return Ok(());
    }
    {
        let mut row = 0usize;
        let mut feed_err: Option<String> = None;
        // P4.1③ root-cause fix: take the controller OUT so the AR-decode
        // head aux co-archive below can borrow head/weights/hidden
        // disjointly alongside `&mut ctl` (mirrors the MTP verify accept
        // path's `self.kvmem.take()`).
        let mut kvmem = rig.model.kvmem.take();
        for s in 0..n {
            let m = batch.m_per_slot[s];
            if m == 0 {
                continue;
            }
            let row_start = row;
            let pos0 = batch.positions[row] as usize;
            row += m;
            if spec_drafts[s].is_some() {
                continue;
            }
            let fed: Result<(), String> = (|| {
                let ctl = kvmem.as_mut().unwrap();
                let taps = rig
                    .model
                    .pbs
                    .kvmem_taps
                    .as_ref()
                    .expect("kvmem controller without statistics taps");
                ctl.feed_rows(
                    &rig.gpu,
                    taps,
                    s,
                    pos0,
                    m,
                    row_start,
                    &rig.pool,
                    &rig.model.k_arenas,
                    &rig.model.v_arenas,
                )?;
                // P4.1③ root-cause fix (AR-decode head aux co-archive):
                // the head aux plane was co-archived ONLY in MTP prefill
                // (mtp_prefill) + MTP verify accept; plain AR-decode rows
                // fed here advanced the trunk Main plane but NOT the aux
                // ordinal, so the head aux frontier lagged the trunk. The
                // next composite (Main+aux) suffix-block export then threw
                // "packed KV layer ... partial frontier" (masked downstream
                // as "layer 0 mass not captured"). Co-archive the head K/V
                // for these accepted AR rows exactly like the verify accept:
                // norm the trunk PRE-final-norm residual (pbs.x_batch) into
                // the head's post-norm staging, recompute the head K/V, and
                // seal at the GLOBAL pos. Prefill rows (decoding=false) are
                // already co-archived by mtp_prefill ⇒ skipped (no double).
                if work[s].decoding && ctl.head_plane_enabled() {
                    if let (Some(head), Some(state)) = (
                        rig.model.mtp_head.as_ref(),
                        rig.model.mtp_states[s].as_mut(),
                    ) {
                        let tokens: Vec<u32> = slots[s]
                            .as_ref()
                            .map(|f| f.session)
                            .and_then(|sid| rig.sessions.get(sid))
                            .map(|sess| sess.tokens[pos0..pos0 + m].to_vec())
                            .unwrap_or_default();
                        if tokens.len() == m {
                            let dim = rig.model.config.dim;
                            let rope_shift = ctl.head_prefill_shift(pos0 + m);
                            let raw = rig.model.pbs.x_batch.sub_offset(row_start * dim, m * dim);
                            let staged = state.takeover_fill_hidden.sub_offset(0, m * dim);
                            rig.gpu
                                .rmsnorm_batched(
                                    &raw,
                                    &rig.model.weights.output_norm,
                                    &staged,
                                    m,
                                    dim,
                                    rig.model.config.norm_eps,
                                )
                                .map_err(|e| format!("kvmem AR head hidden norm: {e:?}"))?;
                            crate::mtp_spec::mtp_takeover_fill_head_kv(
                                &mut rig.gpu,
                                head,
                                state,
                                &rig.model.weights,
                                pos0,
                                &tokens,
                                rope_shift,
                            )
                            .map_err(|e| format!("kvmem AR head fill: {e}"))?;
                            ctl.head_write_through_archive(
                                &rig.gpu,
                                &state.mtp_kv.inner.k_gpu[0],
                                &state.mtp_kv.inner.v_gpu[0],
                                s,
                                pos0,
                                m,
                                pos0 - rope_shift,
                            )?;
                            ctl.accept_head_append(s, pos0 as u64, (pos0 + m) as u64)?;
                        }
                    }
                }
                Ok(())
            })();
            if let Err(e) = fed {
                feed_err = Some(e);
                break;
            }
        }
        rig.model.kvmem = kvmem;
        if let Some(e) = feed_err {
            return Err(e);
        }
    }
    Ok(())
}

/// P4.3 canonical-prefix reclaim under page pressure (admission path).
pub(super) fn kvmem_reclaim_prefixes(rig: &mut Rig) {
    // ── P4.3 canonical-prefix reclaim under page pressure ────────────────
    // Pinned canonical prefixes hold pool pages beyond the per-slot working
    // sets. When the free list cannot back one slot's resident budget plus
    // headroom, Layerwise DEMOTES the LRU catalog entries: pinned pages drop,
    // the catalog entry survives, and a later reuse restores pages from the
    // sealed archive (nInfer keeps the directory separate from residency).
    // Legacy reuse requires resident pages, so it keeps the historical pop
    // (entry removed). Pages still held by live block tables survive the ref
    // release. Bounded: an entry demotes once and the catalog is finite.
    if !rig.kvmem_prefix_pin {
        return;
    }
    {
        let headroom = rig
            .model
            .kvmem
            .as_ref()
            .map(|ctl| ctl.budget_blocks() + 2)
            .unwrap_or(2);
        let layerwise = rig
            .model
            .kvmem
            .as_ref()
            .map(|ctl| ctl.strategy().is_layerwise())
            .unwrap_or(false);
        loop {
            let free = rig.pool.page_pool().map(|pp| pp.free_pages()).unwrap_or(0);
            if free >= headroom {
                break;
            }
            if layerwise {
                let Some((id, pages)) = rig
                    .model
                    .kvmem
                    .as_mut()
                    .and_then(|ctl| ctl.demote_lru_prefix())
                else {
                    break;
                };
                if let Some(pp) = rig.pool.page_pool_mut() {
                    for &phys in &pages {
                        let _ = pp.release_cache_ref(phys);
                    }
                }
                if rig.gpu.slot_trace() {
                    eprintln!(
                        "[kvmem-prefix] demoted canonical prefix {id} ({} pages) under \
                     admission pressure; catalog entry retained",
                        pages.len()
                    );
                }
            } else {
                let victim = rig
                    .model
                    .kvmem
                    .as_mut()
                    .and_then(|ctl| ctl.pop_lru_prefix());
                let Some(victim) = victim else { break };
                if let Some(pp) = rig.pool.page_pool_mut() {
                    for &phys in &victim.gpu_pages {
                        let _ = pp.release_cache_ref(phys);
                    }
                }
                if rig.gpu.slot_trace() {
                    eprintln!(
                        "[kvmem-prefix] reclaimed canonical prefix {} ({} pages) under \
                     admission pressure",
                        victim.id,
                        victim.gpu_pages.len()
                    );
                }
            }
        }
    }
}

/// Decide whether this step is an admissible single-slot Layerwise decode and,
/// if so, return `(slot, session_id, media_pages)` (design §3.9g). Layerwise
/// recalls the compacted selected working set per layer; it admits ONLY a
/// single text-only OR uniform-M-RoPE q8 paged autoregressive decode row with
/// no speculative-verify / DFlash rows. Every other step — prefill, multi-slot,
/// spec-verify, non-q8, non-paged, or Legacy strategy — returns `None` and runs
/// the byte-neutral Legacy path. The host-store length gate (`fed >= 128`) lives
/// in `engine_carrier_rows`.
///
/// G7 Stage D (VL): an image-conversation decode row is admissible when its
/// M-RoPE phases are the uniform past-prompt form `[p+delta; 3]` (in-prompt
/// phases never reach a decode row) and the image's canonical pages fit the
/// resident budget. `media_pages` are those pinned pages — MANDATORY in every
/// per-layer selection, so the vision KV can never be dropped by the keep-set
/// refinement. VL + native spec stays Legacy (the MTP head-plane draft selection
/// is text-only), and a media working set that cannot fit the budget keeps the
/// byte-neutral Legacy path instead of failing closed.
#[cfg(feature = "kvmem")]
pub(super) fn layerwise_admissible(
    rig: &Rig,
    batch: &SlotBatch,
    slots: &[Option<InFlight>],
    work: &[PendingWork],
    any_verify: bool,
    any_dflash_rows: bool,
) -> Option<(usize, u64, Vec<u32>)> {
    let ctl = rig.model.kvmem.as_ref()?;
    if !ctl.strategy().is_layerwise() {
        return None;
    }
    if any_verify || any_dflash_rows || !rig.pool.is_paged() || !rig.model.kv_tier.is_q8() {
        return None;
    }
    // The paged GQA decode tile must engage (the throat's `paged_decode` gate);
    // an operator who disabled it falls back to Legacy rather than failing closed.
    if !hipfire_config::developer_bool("HIPFIRE_KVMEM_GQA_DECODE", true) {
        return None;
    }
    // Exactly one decode row total ⇒ a single-slot autoregressive decode step.
    if batch.total_rows() != 1 {
        return None;
    }
    let slot = batch.m_per_slot.iter().position(|&m| m == 1)?;
    if !work[slot].decoding {
        return None;
    }
    // No vision-tower embedding rows this step (decode rows never carry ext_emb;
    // the image pads live in the prefill). M-RoPE phases are admitted only in
    // the past-prompt uniform form: a decode row rotates all three axes at the
    // same shifted position, which the attention seam reproduces bit-for-bit.
    if !batch.ext_emb.iter().all(|&e| e < 0) {
        return None;
    }
    if batch.pos3.iter().any(|p| p[0] != p[1] || p[1] != p[2]) {
        return None;
    }
    let frame = slots[slot].as_ref()?;
    let sess = rig.sessions.get(frame.session)?;
    let mut media_pages: Vec<u32> = Vec::new();
    if sess.has_image {
        // The MTP head plane restores an aux selection without media semantics;
        // keep VL+MTP on the Legacy path (out of Stage D scope). Escha-trellis
        // dense decode lowers to a program executor with no M-RoPE arm.
        if work[slot].spec.active() || crate::qwen35::forward::has_escha_layers(&rig.model.weights)
        {
            return None;
        }
        let fed = ctl.fed_tokens(slot).unwrap_or(0) as u64;
        let block = crate::layerwise_layout::LAYERWISE_BLOCK_TOKENS as u64;
        let full_pages = fed / block;
        let (_, ranges) = kvmem_session_media(rig, frame.session);
        let spans: Vec<crate::kvmem_media::MediaSpan> = ranges
            .iter()
            .map(|&(begin, end)| crate::kvmem_media::MediaSpan {
                begin: begin as u64,
                end: end as u64,
            })
            .collect();
        let pages = crate::kvmem_media::pages_from_spans(&spans, block, full_pages);
        // The core Query selection charges the sink pages and every mandatory
        // pin inside the same budget and fails closed when they do not fit. An
        // image that cannot be represented keeps the Legacy path (the strategy
        // is opt-in; a normal request must not poison).
        let budget = ctl.budget_blocks() as u64;
        if full_pages > budget && pages.len() as u64 + ctl.sink_blocks() as u64 > budget {
            if rig.gpu.slot_trace() {
                eprintln!(
                    "[kvmem-layerwise] slot {slot} VL decode stays Legacy: media pages {} \
                     + sink {} exceed the resident budget {budget}",
                    pages.len(),
                    ctl.sink_blocks()
                );
            }
            return None;
        }
        media_pages = pages.into_iter().map(|p| p as u32).collect();
        if rig.gpu.slot_trace() {
            eprintln!(
                "[kvmem-layerwise] slot {slot} VL decode admitted: media pages {} {:?} \
                 (fed {fed}, budget {budget})",
                media_pages.len(),
                media_pages
            );
        }
    }
    Some((slot, frame.session.0, media_pages))
}

/// Layerwise spec-verify admission (P4.1② design §A.6 / §B D1.4②; 期4 DFlash2).
/// The multi-row sibling of [`layerwise_admissible`]: admits a single-slot MTP or
/// DFlash2 verify span under Layerwise so the trunk's accepted-only write-through
/// + drop-evict + restore-before-attend runs over the whole `width`-row verifier
/// batch (Path B `run_layers_slots`, already width-aware). Returns
/// `(slot, session_id, width)` where `width` is the slot's verify-row count
/// (`m_per_slot`, == the drafted span from `inject_spec_verify_tokens`).
/// Everything else threads `None` (Legacy byte-neutral): non-Layerwise,
/// non-paged / non-q8, GQA-decode disabled, non-spec rows, multi-slot batches,
/// prefill, image / M-RoPE steps, and DFlash2 shapes whose draft context is not
/// the all-sliding windowed ring (split/legacy drafts keep the Legacy capped
/// path — the KVMem/nInfer DFlash2 integration admits the windowed chain only).
/// The accepted PREFIX is published post-sampling through the OUTPUT-scope owner
/// (`accept_output_pressure`), not the all-or-nothing prefill owner the single-row
/// AR path uses. Automatic: no dedicated on/off env (design §B D1.5) — gated solely
/// on the Layerwise strategy + native spec being active this step. DFlash2 never
/// carries the MTP head plane (load validation keeps `mtp_k == 0`), so the
/// composite head co-archive below is MTP-only by construction.
#[cfg(feature = "kvmem")]
pub(super) fn layerwise_mtp_verify_admissible(
    rig: &Rig,
    batch: &SlotBatch,
    slots: &[Option<InFlight>],
    work: &[PendingWork],
    any_verify: bool,
    any_dflash_rows: bool,
) -> Option<(usize, u64, u64)> {
    let ctl = rig.model.kvmem.as_ref()?;
    if !ctl.strategy().is_layerwise() {
        return None;
    }
    // MTP and DFlash2 verify spans share this Path B (期4): both are multi-row
    // accepted-prefix mechanics. DFlash2 is admitted only in the all-sliding
    // windowed draft shape — the KVMem/nInfer integration semantics (logical
    // ring + compact target positions + suffix bound); split/legacy drafts stay
    // Legacy capped.
    if !any_verify || !rig.pool.is_paged() || !rig.model.kv_tier.is_q8() {
        return None;
    }
    // Same paged GQA decode tile gate as AR: an operator who disabled it falls
    // back to Legacy rather than failing closed.
    if !hipfire_config::developer_bool("HIPFIRE_KVMEM_GQA_DECODE", true) {
        return None;
    }
    // A verify span is multi-row; exactly one slot contributes rows (single-slot
    // focus — multi-slot batching is P4.2). The verify slot's `m_per_slot` is the
    // drafted span width.
    if batch.total_rows() <= 1 {
        return None;
    }
    let slot = batch.m_per_slot.iter().position(|&m| m > 1)?;
    if batch
        .m_per_slot
        .iter()
        .enumerate()
        .any(|(s, &m)| s != slot && m != 0)
    {
        return None;
    }
    let width = batch.m_per_slot[slot] as u64;
    if !work[slot].decoding {
        return None;
    }
    if work[slot].spec != SpecKind::Mtp && work[slot].spec != SpecKind::Dflash {
        return None;
    }
    // The batch-wide DFlash flag and the admitted slot's kind must agree.
    if work[slot].spec == SpecKind::Dflash && !any_dflash_rows {
        return None;
    }
    if work[slot].spec == SpecKind::Mtp && any_dflash_rows {
        return None;
    }
    if work[slot].spec == SpecKind::Dflash {
        let draft = rig.model.dflash.as_ref()?;
        if draft.window.is_none() || !draft.draft_config.all_layers_sliding {
            return None;
        }
    }
    // 期4: never clip the speculative extent. If the full verify width cannot
    // fit the Layerwise window capacity this step stays Legacy (the nInfer
    // "prepare the complete verification width before binding the target
    // window" rule; a narrow round is not a legal fallback).
    if width > ctl.window_capacity_tokens() {
        return None;
    }
    // Text-only this step (same throat as AR): no vision-tower embedding rows, no
    // M-RoPE phases, no image conversation.
    if !batch.ext_emb.iter().all(|&e| e < 0) || !batch.pos3.is_empty() {
        return None;
    }
    let frame = slots[slot].as_ref()?;
    let sess = rig.sessions.get(frame.session)?;
    if sess.has_image {
        return None;
    }
    Some((slot, frame.session.0, width))
}

/// Conversation hashes locate candidates; only exact accepted tokens prove
/// reuse. Include the final accepted-but-not-yet-materialized token too: a
/// changed assistant/tool suffix must not inherit an earlier recurrent state.
pub(super) fn legacy_continuation_matches(accepted: &[u32], rendered: &[u32]) -> bool {
    !accepted.is_empty() && rendered.starts_with(accepted)
}

/// KVMem-observability: locate and report where the freshly rendered prompt
/// stops matching the session's accepted token stream. Split token-level
/// divergence from text-level divergence: identical decoded text means the
/// stored generation re-encoded to a non-canonical (sampled) segmentation.
/// Only called under `HIPFIRE_SLOT_TRACE=1`.
pub(super) fn trace_continuation_divergence(
    tokenizer: &Tokenizer,
    label: &str,
    accepted: &[u32],
    rendered: &[u32],
) {
    let first_tok = accepted
        .iter()
        .zip(rendered.iter())
        .position(|(a, b)| a != b)
        .or_else(|| (accepted.len() != rendered.len()).then(|| accepted.len().min(rendered.len())));
    let accepted_text = tokenizer.decode(accepted);
    let rendered_text = tokenizer.decode(rendered);
    let canonical = tokenizer.encode(&accepted_text);
    eprintln!(
        "[kvmem-prefix] {label}: accepted={} rendered={} first_token_div={:?} accepted_roundtrips={}",
        accepted.len(),
        rendered.len(),
        first_tok,
        canonical.as_slice() == accepted,
    );
    if let Some(i) = first_tok {
        let lo = i.saturating_sub(3);
        eprintln!(
            "[kvmem-prefix]   tokens: accepted[{lo}..]={:?} rendered[{lo}..]={:?}",
            accepted.get(lo..(i + 6).min(accepted.len())),
            rendered.get(lo..(i + 6).min(rendered.len())),
        );
    }
    let a_chars: Vec<char> = accepted_text.chars().collect();
    let r_chars: Vec<char> = rendered_text.chars().collect();
    let first_char = a_chars
        .iter()
        .zip(r_chars.iter())
        .position(|(a, b)| a != b)
        .or_else(|| (a_chars.len() != r_chars.len()).then(|| a_chars.len().min(r_chars.len())));
    let window = |chars: &[char], at: usize| {
        let lo = at.saturating_sub(70);
        let hi = (at + 90).min(chars.len());
        chars[lo..hi].iter().collect::<String>()
    };
    match first_char {
        None => eprintln!("[kvmem-prefix]   decoded text identical"),
        Some(at) => eprintln!(
            "[kvmem-prefix]   text diverges at char {at} (len {} vs {}):\n    accepted: {:?}\n    rendered: {:?}",
            a_chars.len(),
            r_chars.len(),
            window(&a_chars, at),
            window(&r_chars, at),
        ),
    }
}

/// 期4: KVMem × DFlash2 load-time validation matrix (KVMem/nInfer integration
/// pattern). `None` = the combination is legal. Unsupported combinations are
/// refused at load, never silently degraded:
/// * the native MTP head and a DFlash2 draft are mutually exclusive (the main
///   model does not load its MTP head for DFlash2);
/// * the KVMem DFlash lane is single-sequence.
pub(super) fn kvmem_dflash_validation_error(
    kvmem_on: bool,
    dflash_draft: bool,
    mtp_k: usize,
    n_slots: usize,
) -> Option<String> {
    if !kvmem_on || !dflash_draft {
        return None;
    }
    if mtp_k > 0 {
        return Some(format!(
            "KVMem + DFlash2: MTP head (mtp_k={mtp_k}) and a DFlash2 draft are mutually \
             exclusive; serve with --spec dflash (or set speculation.mtp off)"
        ));
    }
    if n_slots != 1 {
        return Some(format!(
            "KVMem + DFlash2: single-slot serve only (SLOTS=1), got n_slots={n_slots}"
        ));
    }
    None
}

pub(super) fn read_legacy_query_rows(
    rig: &mut Rig,
    legacy_query_reads: &[(usize, usize, usize, usize)],
) {
    if !legacy_query_reads.is_empty() {
        let mut failure: Option<String> = None;
        let taps = rig.model.pbs.kvmem_taps.as_ref();
        if let Some(taps) = taps {
            let total_rows: usize = legacy_query_reads.iter().map(|e| e.2).sum();
            let q_width = taps.q_width;
            let mut values: Vec<f32> = Vec::new();
            'layers: for layer in 0..taps.n_fa_layers {
                if let Err(e) = taps.read_query_rows(&rig.gpu, layer, 0, total_rows, &mut values) {
                    failure = Some(format!("Query readback layer {layer}: {e}"));
                    break 'layers;
                }
                if values.len() != total_rows * q_width {
                    failure = Some(format!(
                        "Query readback layer {layer} returned {} values, expected {}",
                        values.len(),
                        total_rows * q_width
                    ));
                    break 'layers;
                }
                for &(s, compact, count, abs_begin) in legacy_query_reads {
                    let Some(run) = rig.legacy_queries[s].as_mut() else {
                        continue;
                    };
                    let slice = &values[compact * q_width..(compact + count) * q_width];
                    if let Err(e) = run.query.record_layer_rows(layer, abs_begin, slice) {
                        failure = Some(format!("Query row record layer {layer}: {e}"));
                        break 'layers;
                    }
                }
            }
        } else {
            failure = Some("Query capture is not enabled".into());
        }
        if let Some(reason) = failure {
            let mut seen: Vec<usize> = Vec::new();
            for &(s, _, _, _) in legacy_query_reads {
                if seen.contains(&s) {
                    continue;
                }
                seen.push(s);
                let Some(run) = rig.legacy_queries[s].as_mut() else {
                    continue;
                };
                if run.query.phase() == crate::legacy_query::LegacyQueryPhase::Probing {
                    let _ = run.query.bypass(&reason);
                    eprintln!("[kvmem-legacy] QueryBypass slot={s} reason={reason} first_pass_preserved=true");
                }
            }
        }
    }
}

pub(super) fn reset_kvmem_slot(
    rig: &mut Rig,
    slot: SlotId,
    req: &SubmitRequest,
) -> Result<bool, String> {
    if rig.model.kvmem.is_none() {
        return Ok(false);
    }
    let ctl = rig.model.kvmem.as_mut().expect("controller checked above");
    ctl.reset_slot(slot.0)
        .map_err(|e| format!("kvmem reset failed: {e}"))?;
    ctl.set_request_query(slot.0, req.query)
        .and_then(|_| ctl.set_request_group(slot.0, convo_group(&req.convo)))
        .map_err(|e| format!("kvmem request metadata: {e}"))?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn enable_kvmem_taps(
    pbs: &mut crate::qwen35::PrefillBatchScratch,
    gpu: &mut Gpu,
    config: &crate::qwen35::Qwen35Config,
    cfg: &EngineConfig,
    max_batch: usize,
) -> Result<(), String> {
    pbs.enable_kvmem_taps(gpu, config, cfg.n_slots)
        .map_err(|e| format!("kvmem taps: {e}"))?;
    if let Some(legacy) = cfg.kvmem.as_ref().filter(|c| c.enabled) {
        let query_rows = (legacy.query_max_tokens as usize)
            .checked_mul(cfg.n_slots)
            .ok_or("kvmem: Query row capacity overflow")?
            .min(max_batch);
        let taps = pbs
            .kvmem_taps
            .as_mut()
            .ok_or("kvmem: missing native Query taps")?;
        let needed = taps
            .query_capture_capacity_bytes(query_rows)
            .map_err(|e| format!("kvmem Query workspace: {e}"))?;
        let (free, _) = gpu
            .device_mem_info()
            .map_err(|e| format!("kvmem Query workspace: {e}"))?;
        if needed > free {
            return Err(format!(
                "kvmem: Query workspace requires {needed} bytes, only {free} free; no fallback"
            ));
        }
        taps.enable_query_capture(gpu, query_rows)
            .map_err(|e| format!("kvmem Query capture allocation: {e}"))?;
        eprintln!(
            "[kvmem-config] query_capture_rows={query_rows} query_workspace_bytes={}",
            taps.query_capacity_bytes()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_kvmem_controller(
    cfg: &EngineConfig,
    config: &crate::qwen35::Qwen35Config,
    n_slots: usize,
    n_fa_layers: usize,
    kv_mode: hipfire_runtime::kv_mode::KvMode,
    per_pos_bytes: usize,
    per_pos_v_bytes: usize,
    max_batch: usize,
    mtp_head: bool,
    spec_rows: usize,
    prefill_chunk: usize,
    plan: Option<crate::kvmem_ctl::KvMemPlan>,
    mut options: Option<crate::kvmem_ctl::KvMemOptions>,
) -> Result<Option<crate::kvmem_ctl::KvMemCtl>, String> {
    let Some(plan) = plan else {
        return Ok(None);
    };
    // Restate an implicit default budget at the clamped value the plan was
    // admitted with, so the controller and operator-facing record agree.
    let mut legacy_config = cfg.kvmem.as_ref().map(|config| {
        let budget_tokens = crate::kvmem_ctl::effective_legacy_budget(
            config.budget_tokens as usize,
            crate::kvmem_ctl::implicit_legacy_budget(config),
            cfg.cap_tokens,
            config.gen_reserve_tokens as usize,
        );
        let mut config = config.clone();
        if budget_tokens != config.budget_tokens as usize {
            config.budget_tokens = budget_tokens as u32;
            if let Some(report) = config.report.get_mut("budget_tokens") {
                report.effective = serde_json::json!(budget_tokens);
            }
            config.notes.push(format!(
                "budget_tokens clamped to {budget_tokens} at admission \
                 (context {} - reserve {})",
                cfg.cap_tokens, config.gen_reserve_tokens
            ));
        }
        config
    });
    // Scale the host tier with the geometry the operator already admitted.
    // The default 16 GiB was sized for the historical 8K working set; at 262k
    // a single slot retains 8 GiB and two slots exceed the default. Explicit
    // HOST_BYTES values stay strict: a too-small limit is a startup error, not
    // a mid-serve capacity fault that poisons every active request.
    let strategy = if legacy_config.is_some() {
        crate::kvmem_ctl::RecallStrategy::Legacy
    } else {
        options
            .as_ref()
            .map(|options| options.strategy)
            .unwrap_or(crate::kvmem_ctl::RecallStrategy::Legacy)
    };
    let n_record_layers = n_fa_layers
        + if strategy.is_layerwise() && mtp_head {
            1
        } else {
            0
        };
    let required_host = crate::kvmem_ctl::required_host_bytes(
        n_slots,
        cfg.cap_tokens,
        n_record_layers,
        per_pos_bytes,
        per_pos_v_bytes,
    );
    let (current_host, implicit_host) = match legacy_config.as_ref() {
        Some(config) => (
            config.host_bytes,
            config
                .report
                .get("host_bytes")
                .is_none_or(|report| report.source == hipfire_config::kvmem::Source::Default),
        ),
        None => (
            options
                .as_ref()
                .and_then(|options| options.host_bytes)
                .unwrap_or(crate::kvmem_ctl::DEFAULT_HOST_BYTES),
            options
                .as_ref()
                .is_none_or(|options| options.host_bytes.is_none()),
        ),
    };
    let spill_configured = match legacy_config.as_ref() {
        Some(config) => config.disk_path.is_some(),
        None => options
            .as_ref()
            .and_then(|options| options.spill_dir.as_ref())
            .is_some_and(|dir| !dir.is_empty()),
    };
    let resolved_host = if spill_configured {
        // A disk tier turns an over-RAM charge into a spill instead of a hard
        // failure, so an explicit RAM cap below the full-context requirement
        // stays valid here. Layerwise pays a synchronous spill per feed; the
        // warning is the operator's cue to raise the RAM cap.
        if current_host < required_host {
            eprintln!(
                "[kvmem-config] host_bytes={current_host} is below the full-context \
                 requirement {required_host}; the spill tier absorbs the excess \
                 (Layerwise pays synchronous spill per feed)"
            );
        }
        current_host
    } else {
        crate::kvmem_ctl::resolve_host_bytes(current_host, required_host, implicit_host)?
    };
    if resolved_host != current_host {
        eprintln!(
            "[kvmem-config] host_bytes auto-raised {current_host} -> {resolved_host} \
             (full-context requirement {required_host}, slots={n_slots}, \
             cap_tokens={}, record_layers={n_record_layers})",
            cfg.cap_tokens
        );
        if let Some(config) = legacy_config.as_mut() {
            config.host_bytes = resolved_host;
            config.notes.push(format!(
                "host_bytes auto-raised from {current_host} to {resolved_host} \
                 (full-context requirement for {n_slots} slot(s) at {} tokens)",
                cfg.cap_tokens
            ));
            if let Some(report) = config.report.get_mut("host_bytes") {
                report.effective = serde_json::json!(resolved_host);
            }
        } else if let Some(options) = options.as_mut() {
            options.host_bytes = Some(resolved_host);
        }
    }
    if let Some(config) = legacy_config.as_ref() {
        eprintln!(
            "[kvmem-config] contract={} budget_tokens={} gen_reserve_tokens={} \
             sink_tokens={} recent_tokens={} sessions={} host_bytes={} \
             disk_bytes={} disk_path={:?} query_max_tokens={} \
             query_replay_max_tokens={} query_scorer={:?} score_agg={:?} \
             lrfu={} lrfu_lambda={} lrfu_top_fraction={} slopes=native \
             pages_total={} budget_blocks={} grow_blocks={} interval={} \
             slots={} spec_rows={} prefill_chunk={} cap_tokens={}",
            hipfire_config::kvmem::CONTRACT_VERSION,
            config.budget_tokens,
            config.gen_reserve_tokens,
            config.sink_tokens,
            config.recent_tokens,
            config.sessions,
            config.host_bytes,
            config.disk_bytes,
            config.disk_path.as_deref(),
            config.query_max_tokens,
            config.query_replay_max_tokens,
            config.query_scorer,
            config.score_agg,
            config.lrfu,
            config.lrfu_lambda,
            config.lrfu_top_fraction,
            plan.pages_total,
            plan.budget_blocks,
            plan.grow_blocks,
            plan.interval,
            n_slots,
            spec_rows,
            prefill_chunk,
            cfg.cap_tokens,
        );
    } else {
        eprintln!(
            "[kvmem-config] host_bytes={resolved_host} required={required_host} \
             slots={n_slots} cap_tokens={} record_layers={n_record_layers} \
             strategy={strategy:?}",
            cfg.cap_tokens
        );
    }
    Ok(Some(crate::kvmem_ctl::KvMemCtl::new_with_config(
        n_slots,
        &plan,
        &crate::kvmem_ctl::KvMemGeometry {
            n_fa_layers,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            kv_mode,
            per_pos_bytes,
            per_pos_v_bytes,
            max_rows: max_batch,
            mtp_head,
        },
        legacy_config.as_ref(),
        options.ok_or("kvmem: missing start-time option snapshot")?,
    )?))
}

impl ModelRig {
    /// P4.1② f2 (C): the draft phase's head-layerwise restore-before-attend.
    ///
    /// Under Layerwise+MTP with a bound FirstLayer aux selection (G5), restore
    /// the head's bounded aux arena from the B+D-populated composite archive,
    /// then run `draft` through the [`HeadLayerwise`] handle. Mirroring the trunk
    /// verify carrier, the handle is created, attended and freed *inside* the
    /// `Ok(Some)` arm so its borrow of `ctl` never escapes to the `&mut ctl`
    /// cleanup siblings. On any gate miss (Legacy / AR-only / cold start /
    /// unbound aux / absent head map / no sealed page) `draft` runs full-resident
    /// (`None`, byte-neutral) and anything consumed is put back. A hard restore
    /// error restores the head map + aux binding and propagates (fail-closed).
    /// `policy` is a block local: the handle borrows it but is freed before it
    /// drops, so nothing referencing it escapes.
    #[cfg(all(feature = "deltanet", feature = "kvmem"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mtp_draft_head_restore<'a, F>(
        gpu: &mut Gpu,
        ctl: &'a mut crate::kvmem_ctl::KvMemCtl,
        slot: usize,
        session_id: u64,
        cap_tokens: usize,
        config: &crate::qwen35::Qwen35Config,
        draft: F,
    ) -> Result<crate::mtp_spec::MtpDraftOutput, String>
    where
        F: for<'b> FnOnce(
            &mut Gpu,
            Option<&mut crate::layerwise_driver::HeadLayerwise<'b>>,
        ) -> Result<crate::mtp_spec::MtpDraftOutput, String>,
    {
        use crate::layerwise_layout::LAYERWISE_BLOCK_TOKENS as BT;
        let fed = match ctl.fed_tokens(slot) {
            Some(f) => f,
            None => return draft(gpu, None),
        };
        // No sealed full page yet ⇒ nothing to recall; stay full-resident.
        if fed < BT {
            return draft(gpu, None);
        }
        let aux = match ctl.peek_aux_selection(slot) {
            Some(a) => a,
            None => return draft(gpu, None),
        };
        let mut head = match ctl.take_head_plane(slot) {
            Some(h) => h,
            None => {
                ctl.set_aux_selection(slot, aux);
                return draft(gpu, None);
            }
        };
        // Committed: every failure below restores the head map + aux binding.
        let policy = match ctl.layerwise_policy(fed) {
            Ok(p) => p,
            Err(e) => {
                ctl.restore_head_plane(slot, head);
                ctl.set_aux_selection(slot, aux);
                return Err(e);
            }
        };
        let window = ctl.budget_blocks() * BT as usize;
        let n_fa = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count();
        // The carrier borrows `ctl` + `policy`; confine it to the `Ok(Some)` arm
        // and run+free the draft handle there so the borrow never escapes to the
        // `None`/`Err` cleanup siblings (mirrors the trunk verify carrier match).
        match crate::layerwise_driver::engine_carrier_head(
            gpu,
            ctl,
            &policy,
            slot,
            session_id,
            n_fa,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            cap_tokens as u32,
            "qwen35",
            /* width */ 1,
            window,
        ) {
            Ok(Some(carrier)) => {
                let selected: Vec<u64> = aux
                    .selected()
                    .map(|s| s.iter().map(|&x| x as u64).collect())
                    .unwrap_or_default();
                let suffix = crate::layerwise_driver::HeadSuffix {
                    page: fed / BT,
                    partial_tokens: fed % BT,
                    rows: 1,
                };
                let rope_pos = match gpu.hip.malloc(4) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = carrier.free(gpu);
                        ctl.restore_head_plane(slot, head);
                        ctl.set_aux_selection(slot, aux);
                        return Err(format!("mtp head rope_pos alloc: {e}"));
                    }
                };
                match crate::layerwise_driver::HeadLayerwise::restore(
                    gpu,
                    ctl,
                    &mut head,
                    carrier,
                    slot,
                    &selected,
                    fed as u64,
                    window,
                    session_id,
                    cap_tokens as u32,
                    "qwen35",
                    &suffix,
                    rope_pos,
                ) {
                    Ok(mut hlw) => {
                        // Attend the draft through the handle, then free it
                        // *here* so its `ctl` borrow ends before the `&mut ctl`
                        // head-map restore below (verify-pattern confinement).
                        let res = draft(gpu, Some(&mut hlw));
                        let free_res = hlw.free(gpu);
                        ctl.restore_head_plane(slot, head);
                        if let Err(e) = free_res {
                            return Err(format!("mtp head carrier free: {e}"));
                        }
                        res
                    }
                    Err(e) => {
                        ctl.restore_head_plane(slot, head);
                        ctl.set_aux_selection(slot, aux);
                        Err(e)
                    }
                }
            }
            Ok(None) => {
                ctl.restore_head_plane(slot, head);
                ctl.set_aux_selection(slot, aux);
                draft(gpu, None)
            }
            Err(e) => {
                ctl.restore_head_plane(slot, head);
                ctl.set_aux_selection(slot, aux);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod kvmem_dflash_validation_tests {
    use super::kvmem_dflash_validation_error;

    /// 期4 rejection matrix (KVMem/nInfer integration pattern): unsupported
    /// combinations are refused at load; everything outside the KVMem+DFlash
    /// intersection is untouched.
    #[test]
    fn accepts_legal_combinations() {
        // No KVMem: DFlash behaviour is unchanged (Legacy/Layerwise untouched).
        assert!(kvmem_dflash_validation_error(false, true, 4, 2).is_none());
        // No DFlash: MTP / AR loads unchanged.
        assert!(kvmem_dflash_validation_error(true, false, 4, 2).is_none());
        // KVMem + DFlash2 single slot, no MTP head.
        assert!(kvmem_dflash_validation_error(true, true, 0, 1).is_none());
        assert!(kvmem_dflash_validation_error(false, false, 0, 1).is_none());
    }

    #[test]
    fn rejects_mtp_head_with_dflash() {
        let e = kvmem_dflash_validation_error(true, true, 4, 1).expect("must reject");
        assert!(e.contains("mutually exclusive"), "{e}");
    }

    #[test]
    fn rejects_multi_slot_kvmem_dflash() {
        let e = kvmem_dflash_validation_error(true, true, 0, 2).expect("must reject");
        assert!(e.contains("single-slot"), "{e}");
    }
}
