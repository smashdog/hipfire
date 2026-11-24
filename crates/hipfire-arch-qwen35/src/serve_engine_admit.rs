// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Admission, continuation, eviction and parked-session lifecycle.

use super::*;

/// Admit one request, evicting an idle session if that is what it takes.
pub(super) fn admit(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    stats: &Arc<Mutex<EngineStats>>,
    req: SubmitRequest,
) {
    let busy: Vec<SessionId> = slots.iter().flatten().map(|f| f.session).collect();

    if let Some(vd) = req.visual_data.as_ref() {
        let reject = |reason: String| {
            let _ = send_event(
                &req.reply,
                Event::Rejected {
                    class: RejectClass::Validation,
                    reason,
                },
            );
            lock_stats(stats).note_rejected();
        };
        if rig.model.vision_weights.is_none() {
            reject("VL request but model has no vision encoder".to_string());
            return;
        }
        // Batched VL runs the paged block-table path like any slot; only the
        // legacy sequential fallback needs a contiguous slab view.
        if rig.pool.is_paged() && rig.vl_sequential {
            reject("VL sequential path requires contiguous KV slots".to_string());
            return;
        }
        if rig.tokenizer.special_token_id("<|image_pad|>").is_none() {
            reject("VL tokenizer missing <|image_pad|>".to_string());
            return;
        }
        if vd.n_visual_tokens == 0 {
            reject("VL request has no visual tokens".to_string());
            return;
        }
        if vd.mrope_positions.len() != req.prompt_tokens.len() {
            reject(format!(
                "VL mrope_positions length {} != prompt_tokens {}",
                vd.mrope_positions.len(),
                req.prompt_tokens.len()
            ));
            return;
        }
    }

    // Bound parked-session growth before matching: close swapped/cold
    // sessions idle past the TTL, then enforce the non-resident cap. A
    // resident session is never a sweep victim.
    sweep_sessions(rig);

    // Multi-turn continuation. The client resends the whole conversation, so a
    // follow-up turn is matched by its USER turns and then built by APPENDING
    // `continuation` to the session's exact stored tokens. Appending rather
    // than re-rendering is what makes it a strict extension: the generated
    // turn began after an OpenThink opener that history rendering does not
    // replay, and re-encoding the decoded reply is not guaranteed to round
    // trip. Reuse also keeps the DeltaNet state, which is the point.
    // Set when a matched session's snapshot was lost and restore marked it
    // Cold: the cold path below re-prefills into that same session instead of
    // minting a new id (which would strand the old entry).
    let mut cold_reuse: Option<SessionId> = None;
    if req.visual_data.is_none() && !req.continuation.tokens().is_empty() {
        if rig.gpu.slot_trace() {
            let shape = match &req.continuation {
                Continuation::ToolResults { session, .. } => {
                    format!("tool-results of session {session}")
                }
                _ => "new user turn".to_string(),
            };
            eprintln!(
                "[slot-trace] continuation attempt ({shape}): convo={:?} suffix={} tokens",
                req.convo,
                req.continuation.tokens().len(),
            );
            for (id, sess) in rig.sessions.iter() {
                eprintln!(
                    "[slot-trace]   session {} convo={:?} slot={:?} residency={:?} tokens={}",
                    id,
                    sess.convo,
                    sess.slot,
                    sess.residency,
                    sess.tokens.len()
                );
            }
        }
        // The suffix shape decides which match is legal — never both. A user
        // turn searches for the session one turn behind; tool results confirm
        // the session that emitted the calls being answered. Either way the
        // suffix is appended to the session's exact stored tokens, so the KV
        // and DeltaNet state extend strictly.
        let matched = match &req.continuation {
            Continuation::Cold => None,
            Continuation::UserTurn(_) => rig.sessions.find_continuation(&req.convo, &busy),
            Continuation::ToolResults { session, .. } => {
                rig.sessions
                    .confirm_reentry(SessionId(*session), &req.convo, &busy)
            }
        };
        let matched = if kvmem_legacy_enabled(rig) {
            match matched {
                Some(existing)
                    if !rig.sessions.get(existing).is_some_and(|session| {
                        legacy_continuation_matches(&session.tokens, &req.prompt_tokens)
                            && media_continuation_matches(session, req.visual_data.as_ref())
                    }) =>
                {
                    // The client's history does not extend the accepted
                    // stream (most commonly a replayed assistant turn that
                    // omits the reasoning it was streamed with, so the
                    // template renders an empty think block). The claim is
                    // consumed before submit, so a rejection here would
                    // strand the conversation — no retry could re-claim.
                    // Degrade to the cold path below instead: it prefills
                    // the complete rendered prompt (tool results included)
                    // and re-anchors the session on that canonical render.
                    if rig.gpu.slot_trace() {
                        if let Some(session) = rig.sessions.get(existing) {
                            trace_continuation_divergence(
                                &rig.tokenizer,
                                "tool continuation mismatch",
                                &session.tokens,
                                &req.prompt_tokens,
                            );
                        }
                        eprintln!(
                            "[kvmem-prefix] session {} full rendered prefix mismatch; cold recompute",
                            existing.0
                        );
                    }
                    None
                }
                candidate => candidate,
            }
        } else {
            matched
        };
        // KVMem: a finished session is kept resident precisely so its next
        // turn extends it in place — the trunk KV (resident pages plus
        // spilled payloads and row statistics in the host store), the
        // DeltaNet state and the MTP head KV all remain valid, so the shared
        // prefix is never re-prefilled. A session that holds no slot has no
        // host tier that could survive (a swap snapshot cannot represent
        // spilled pages), so it is not resumable; the cold path re-prefills.
        // The filter also refuses a session whose host store does not cover
        // the slot's materialized prefix exactly — feeding rows onto any
        // other prefix would break the store's contiguity contract.
        let matched = kvmem_hooks::filter_continuation(rig, matched);
        if let Some(existing) = matched {
            let base = rig
                .sessions
                .get(existing)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            // A matched session may be SWAPPED. Bring it back before use --
            // this is the half of SP6 that makes eviction worth doing, and
            // without it evicted snapshots accumulate and their sessions are
            // never seen again.
            let mut slot = rig.sessions.get(existing).and_then(|s| s.slot);
            // Resident = this session never left the slot, so the slot's
            // trunk KV/DN state AND the MTP head's private KV (filled through
            // this session's tokens) are still exactly right. Only a
            // swap-restore loses the head KV — the swap snapshot does not
            // carry it — and must reset before the suffix re-fills.
            let mtp_head_kv_valid = slot.is_some();
            if slot.is_none() {
                let free = rig.pool.acquire().or_else(|| {
                    rig.sessions
                        .lru_idle_victim(&busy)
                        .filter(|v| *v != existing)
                        .and_then(|v| {
                            if evict(rig, v, stats) {
                                lock_stats(stats).note_eviction();
                            }
                            rig.pool.acquire()
                        })
                });
                if let Some(target) = free {
                    if restore(rig, existing, target, stats) {
                        lock_stats(stats).note_restore();
                        slot = Some(target);
                    } else {
                        // restore marked it Cold; its tokens survive, so the
                        // cold path below re-prefills the same session rather
                        // than minting a new id and stranding this entry.
                        cold_reuse = Some(existing);
                        rig.pool.release(target);
                    }
                }
            }
            if let Some(slot) = slot {
                let extended = if kvmem_legacy_enabled(rig) {
                    // This is the same full rendered input that proved the
                    // prefix above and supplied req.query. Reconstructing a
                    // suffix by hand would invalidate both token provenance
                    // and the absolute Query coordinates.
                    req.prompt_tokens.clone()
                } else {
                    let mut extended = base;
                    extended.extend_from_slice(req.continuation.tokens());
                    extended
                };
                // Prefill-side context-cap guard (mirrors decode hit_ctx_cap).
                // A prompt/extension already at the KV cap overflows set_seq_len
                // during prefill before any decode step can stop it.
                if extended.len() >= rig.cap_tokens {
                    // A session at the KV cap can never grow: report the
                    // condition as a REJECTION, not a Done — a successful
                    // empty completion here made every subsequent turn on
                    // the session a silent no-op (it would re-enter this
                    // guard forever while the resident session pinned its
                    // slot). The session is closed so the slot is released;
                    // a follow-up on the same conversation re-attempts cold
                    // and receives the same actionable rejection.
                    let reason = format!(
                        "context window full: the conversation holds {} of \
                         {} tokens and cannot extend — start a new \
                         conversation",
                        extended.len(),
                        rig.cap_tokens
                    );
                    let _ = send_event(
                        &req.reply,
                        Event::Rejected {
                            class: RejectClass::Validation,
                            reason,
                        },
                    );
                    rig.swap.forget(existing.0);
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, existing);
                    return;
                }
                // vLLM-style admission belt, mirroring the daemon's cold-path
                // belt: the extension plus the requested generation budget
                // must fit the context. Without this, a request admitted in
                // the (cap - max_tokens, cap) window decodes until the ctx-cap
                // guard truncates it — a silent "length" finish the client
                // never asked for. The SESSION is kept (it still has room to
                // grow): only this request is rejected, so a retry with a
                // smaller max_tokens can still continue the conversation.
                if extended.len() + req.max_tokens.max(1) > rig.cap_tokens {
                    let reason = format!(
                        "context window budget: the conversation would hold {} \
                         tokens and the request asks for {} more, beyond the {} \
                         token context — reduce max_tokens or start a new \
                         conversation",
                        extended.len(),
                        req.max_tokens.max(1),
                        rig.cap_tokens
                    );
                    let _ = send_event(
                        &req.reply,
                        Event::Rejected {
                            class: RejectClass::Validation,
                            reason,
                        },
                    );
                    return;
                }
                // Page-demand reclaim (spec §5.4 S4 row 3), mirroring the
                // cold-admit path: the suffix plus generation budget must be
                // backable by the pool BEFORE begin_turn mutates the table —
                // otherwise the forward dies mid-prefill on "need N more
                // pages". Evict radix leaves first, then LRU idle sessions;
                // the matched session itself is never a victim.
                //
                // Under KVMem the question does not apply: the suffix keeps
                // spilling into the host store as it grows, so demand is
                // bounded by the resident budget + step headroom, not by the
                // full context (and this check would reject every long
                // continuation against a pool that only holds the budget).
                if rig.pool.is_paged() && kvmem_hooks::page_demand_allowed(rig) {
                    // `held` = tokens already materialized in this slot's KV.
                    // For a RESIDENT session that is the descriptor's seq_len
                    // (the real frontier), NOT sess.next_pos — next_pos is
                    // only stamped at begin_turn/mark_resident and still holds
                    // the PREVIOUS turn's reuse point, which over-counts the
                    // suffix and evicts/rejects requests the pool could back.
                    let held = rig
                        .pool
                        .descriptors()
                        .get(slot.0)
                        .map(|d| d.seq_len.max(0) as usize)
                        .unwrap_or_else(|| {
                            rig.sessions.get(existing).map(|s| s.next_pos).unwrap_or(0)
                        });
                    let needed_pages = extended
                        .len()
                        .saturating_sub(held)
                        .saturating_add(req.max_tokens.max(1))
                        .div_ceil(PAGE_TOKENS)
                        .saturating_add(1);
                    if let (Some(idx), Some(domain)) =
                        (rig.prefix_index.as_mut(), rig.cache_domain.as_ref())
                    {
                        let free = rig.pool.page_pool().map(|pp| pp.free_pages()).unwrap_or(0);
                        if free < needed_pages {
                            let page_bytes = rig
                                .pool
                                .page_pool()
                                .map(|pp| pp.k_page_bytes() + pp.v_page_bytes())
                                .unwrap_or(0);
                            let pp = rig.pool.page_pool_mut().expect("paged pool");
                            idx.evict_unpinned_leaves(
                                pp,
                                needed_pages.saturating_sub(free) * page_bytes,
                            );
                        }
                        let _ = domain;
                    }
                    let mut exclude: Vec<SessionId> = busy.clone();
                    exclude.push(existing);
                    loop {
                        let free = rig.pool.page_pool().map(|pp| pp.free_pages()).unwrap_or(0);
                        if free >= needed_pages {
                            break;
                        }
                        let Some(victim) = rig.sessions.lru_idle_victim(&exclude) else {
                            break;
                        };
                        if !evict(rig, victim, stats) {
                            break;
                        }
                        lock_stats(stats).note_eviction();
                    }
                    let free = rig.pool.page_pool().map(|pp| pp.free_pages()).unwrap_or(0);
                    if free < needed_pages {
                        // Keep the session resident — a smaller retry can
                        // still continue it; only this request is refused.
                        let _ = send_event(
                            &req.reply,
                            Event::Rejected {
                                class: RejectClass::Overload,
                                reason: format!(
                                    "page demand exceeds pool: need \
                                     {needed_pages} pages, {free} free \
                                     after reclaim"
                                ),
                            },
                        );
                        lock_stats(stats).note_rejected();
                        return;
                    }
                }
                // Compile the JSON Schema BEFORE any session mutation (spec
                // §7 G1/G2; P4 exit "strict requests never silently
                // unconstrained"). A compile failure rejects the request and
                // leaves the matched session untouched — the daemon
                // pre-validates, so this only fires on drift between that
                // gate and the engine's compiler.
                let grammar_constraint = match req.json_schema.as_ref() {
                    Some(schema) => match cached_compile_schema(rig, schema) {
                        Ok(compiled) => Some(GrammarConstraint {
                            matcher: grammar::json_schema::SchemaMatcher::from_compiled(&compiled),
                            mask_buf: Vec::new(),
                            in_think: req.started_in_think,
                            think_open_id: rig.tokenizer.special_token_id("<think>"),
                            think_close_id: rig.tokenizer.special_token_id("</think>"),
                            think_tail: Vec::new(),
                            think_budget: req.think_budget,
                            think_tokens_used: 0,
                            mask_cache: std::collections::HashMap::new(),
                            mask_cache_order: std::collections::VecDeque::new(),
                        }),
                        Err(e) => {
                            let _ = send_event(
                                &req.reply,
                                Event::Rejected {
                                    class: RejectClass::Validation,
                                    reason: format!("json_schema compile failed on admit: {e}"),
                                },
                            );
                            lock_stats(stats).note_rejected();
                            return;
                        }
                    },
                    None => None,
                };
                // Resize the session's admission grant to the new turn's
                // actual need BEFORE any mutation (spec §5.1 + §5.4 row 1:
                // a failed grant is a pre-execution rejection, not a
                // mid-turn shortfall). The old full-cap grant is shrunk or
                // grown here; a growth that does not fit the budget fails
                // this request cleanly and leaves the session untouched.
                let turn_grant = extended
                    .len()
                    .saturating_add(req.max_tokens.max(1))
                    .min(rig.cap_tokens);
                if let Err(e) = rig
                    .sessions
                    .resize_grant(&mut rig.adm, existing, turn_grant)
                {
                    // A real budget shortfall is a capacity signal (429
                    // Overload — retryable), not an internal fault (500).
                    // UnknownSession/PoolFull are engine inconsistencies and
                    // stay Internal.
                    let class = match e {
                        hipfire_runtime::admission::AdmitError::WouldExceedBudget { .. } => {
                            RejectClass::Overload
                        }
                        _ => RejectClass::Internal,
                    };
                    let _ = send_event(
                        &req.reply,
                        Event::Rejected {
                            class,
                            reason: format!(
                                "admission grant for the extended turn does not fit: {e}"
                            ),
                        },
                    );
                    lock_stats(stats).note_rejected();
                    return;
                }
                match rig.sessions.begin_turn(&mut rig.pool, existing, &extended) {
                    Ok(plan) => {
                        // KVMem: the slot's host store must already cover
                        // exactly the reuse point — the forwarded frontier
                        // the block table represents. `begin_request` stamps
                        // the turn boundary (the prompt-completing chunk only
                        // scores retrieval when its Q tap came from THIS
                        // request) and reports the stored length; feeding
                        // rows onto any other prefix would break the store's
                        // contiguity contract. Fail closed — close the
                        // session so the slot and its host tier rebuild on
                        // the next cold attempt — rather than corrupt it.
                        if let Err(reason) = kvmem_hooks::begin_continuation_request(
                            rig,
                            slot,
                            existing,
                            &req,
                            plan.reused,
                            &extended,
                        ) {
                            let _ = send_event(
                                &req.reply,
                                Event::Rejected {
                                    class: RejectClass::Internal,
                                    reason,
                                },
                            );
                            rig.swap.forget(existing.0);
                            rig.sessions.close(&mut rig.pool, &mut rig.adm, existing);
                            lock_stats(stats).note_rejected();
                            return;
                        }
                        if let Some(sess) = rig.sessions.get_mut(existing) {
                            sess.tokens = extended.clone();
                            sess.convo = req.convo.clone();
                        }
                        if send_event(
                            &req.reply,
                            Event::Accepted {
                                session: existing.0,
                                reused: plan.reused,
                                prefill: extended.len() - plan.reused,
                            },
                        )
                        .is_err()
                        {
                            // Mutated session must not remain reusable — forget swap,
                            // close SessionTable/pool/admission so no poisoned KV/DN
                            // state survives. Preserve typed Continuation semantics
                            // and max-terminal-token accounting (caller retains base
                            // for retransmission).
                            rig.swap.forget(existing.0);
                            rig.sessions.close(&mut rig.pool, &mut rig.adm, existing);
                            return;
                        }
                        work[slot.0].remaining_prompt = extended[plan.reused..].to_vec();
                        work[slot.0].next_pos = plan.reused;
                        work[slot.0].decoding = false;
                        // A text follow-up on a conversation that carries an
                        // image turn keeps that turn's M-RoPE phase offset: KV
                        // rows are addressed by token index, but the image
                        // turn's keys hold compressed-grid phases, so this
                        // turn's rows must phase-shift by the same delta.
                        // (Image turns themselves are forced Cold and rebuild
                        // the offset from their own table.)
                        work[slot.0].pos3_delta = rig
                            .sessions
                            .get(existing)
                            .map(|s| s.rope_delta)
                            .unwrap_or(0);
                        // MTP: activate if enabled. Vision continuations stay on
                        // the sequential M-RoPE path (no MTP), matching cold
                        // admits; sampling and penalties ride the MTP path via
                        // `spec_req_sampling` (see the cold-admit note). The head
                        // KV reset is only needed
                        // when the head KV was lost (swap-restore) or holds
                        // another session's rows: a session resident on this slot
                        // since its last turn keeps a head KV that matches its
                        // committed prefix, so the suffix's head-fill extends it
                        // in place.
                        // Head-KV validity gates drafting: after a swap-restore
                        // the head's private KV is reset to zeros, and the
                        // suffix head-fill only covers NEW rows — drafting over
                        // a zeroed prefix span collapses acceptance to ~1
                        // (pure drafting overhead), so the turn retires to AR
                        // instead. A session resident on this slot keeps a head
                        // KV matching its committed prefix and drafts fine.
                        let mtp_ok = rig.model.mtp_head.is_some()
                            && rig.model.mtp_k > 0
                            && req.visual_data.is_none()
                            && req.json_schema.is_none()
                            && mtp_head_kv_valid
                            && rig.model.mtp_states[slot.0].is_some();
                        // DFlash2 admit: same request-shape gates as MTP (text
                        // only, no grammar). Sampling and token penalties are
                        // honored by the verify accept (point-mass residual +
                        // exact-AR prepass), so neither bars admission. The
                        // draft's private state
                        // survives on this slot across turns, so a continuation
                        // whose reuse point equals the ring's watermark
                        // (`seeded_through`) drafts over a fully-seeded
                        // target_hidden — including radix-reused prefixes,
                        // which MTP cannot do. An edit that truncates below the
                        // watermark leaves stale ring rows: drop the state and
                        // run AR (a partial ring would poison the draft's
                        // context with the pre-edit suffix's hiddens).
                        let dflash_ring_valid = rig.model.dflash_states[slot.0]
                            .as_ref()
                            .is_some_and(|st| st.seeded_through == plan.reused);
                        // Legacy-mode ctx fit (sequential `spec_ctx_request_fits`
                        // parity): the Legacy scatter writes target_hidden at the
                        // ABSOLUTE position into a `ctx_capacity`-row buffer, so a
                        // request that can grow past `ctx_capacity` would drive an
                        // out-of-bounds scatter (a release `assert!` panic in
                        // `sub_offset` — the serve-hang class). Windowed mode has
                        // no such bound (its rings wrap), so the gate is Legacy-
                        // only. `+ block_size` keeps the last verify window's
                        // `advance` rows inside the buffer.
                        let dflash_ctx_fits = rig.model.dflash.as_ref().map_or(false, |d| {
                            d.window.is_some()
                                || req
                                    .prompt_tokens
                                    .len()
                                    .saturating_add(req.max_tokens.max(1))
                                    .saturating_add(d.block_size)
                                    <= d.ctx_capacity
                        });
                        let dflash_ok = rig.model.dflash.is_some()
                            && req.visual_data.is_none()
                            && req.json_schema.is_none()
                            && dflash_ctx_fits
                            && (plan.reused == 0 || dflash_ring_valid);
                        if !dflash_ring_valid {
                            if let Some(st) = rig.model.dflash_states[slot.0].take() {
                                st.free_gpu(&mut rig.gpu);
                            }
                        }
                        work[slot.0].spec = if mtp_ok {
                            SpecKind::Mtp
                        } else if dflash_ok {
                            SpecKind::Dflash
                        } else {
                            SpecKind::None
                        };
                        #[cfg(feature = "kvmem")]
                        {
                            rig.legacy_spec_kind[slot.0] = work[slot.0].spec;
                        }
                        rig.spec_counters[slot.0] = SpecCounters {
                            requested: work[slot.0].spec.active(),
                            ..Default::default()
                        };
                        rig.model.spec_req_sampling[slot.0] = if work[slot.0].spec.active() {
                            Some(MtpReqSampling::new(&req, &extended))
                        } else {
                            None
                        };
                        work[slot.0].spec_retire = hipfire_runtime::spec_retire::SpecRetire::new(
                            rig.model.spec_retire_break_even,
                        );
                        if !work[slot.0].spec.active() {
                            if let Some(state) = rig.model.mtp_states[slot.0].as_mut() {
                                let _ = state.reset(&mut rig.gpu);
                            }
                        }
                        install_sample_params(rig, slot, &req);
                        // Publication continuity (spec §4.6 C6): the session's
                        // previously published boundary seeds the new InFlight so
                        // this turn's publications start BEYOND pages the radix
                        // already owns. Re-publishing from 0 would take a second
                        // cache ref on already-owned pages — orphaned on release
                        // (nothing but the index releases exactly one ref per
                        // node page), stranding pages Sealed/CacheOnly forever:
                        // a monotonic pool drain across turns. The boundary is
                        // clamped to the turn's reuse point: a client edit that
                        // truncates below it invalidates the published claim.
                        let session_pub = rig
                            .sessions
                            .get(existing)
                            .map(|s| s.published_boundary.min(plan.reused))
                            .unwrap_or(0);
                        if let Some(sess) = rig.sessions.get_mut(existing) {
                            sess.published_boundary = session_pub;
                        }
                        let repeat_window_req = effective_repeat_window(
                            req.repeat_window,
                            req.repeat_penalty,
                            req.presence_penalty,
                            req.frequency_penalty,
                        );
                        slots[slot.0] = Some(InFlight {
                            session: existing,
                            reply: req.reply,
                            produced: 0,
                            max_tokens: req.max_tokens.max(1),
                            repeat_window_req,
                            last_published_boundary: session_pub,
                            // P4.3: freezing resumes at the turn's reuse point
                            // (the prefix below it was fed by earlier turns).
                            last_frozen_boundary: plan.reused,
                            last_coverage_boundary: 0,
                            #[cfg(feature = "kvmem")]
                            checkpoint_plan: rig.kvmem_checkpoint_policy.and_then(|(i, _)| {
                                hipfire_kvmem::CheckpointCapturePlan::new(
                                    plan.reused as u64,
                                    i,
                                    0,
                                    0,
                                )
                                .ok()
                            }),
                            grammar: grammar_constraint,
                            // Continuation reuse is session-local (convo hash),
                            // not a radix lookup — no pin to hold.
                            pin_ticket: PinTicket::none(),
                        });
                        rig.sessions.touch(existing);
                        if rig.gpu.slot_trace() {
                            eprintln!(
                            "[slot-trace] continuation HIT -- session {} reused {} of {} tokens",
                            existing.0,
                            plan.reused,
                            extended.len()
                        );
                        }
                        // FairQueue: admit the continued request with its
                        // remaining prefill suffix as the cache-locality weight.
                        let uncached = work[slot.0].remaining_prompt.len() as u64;
                        let _ = rig.fair_queue.admit(existing.0, "default", uncached);
                        let mut st = lock_stats(stats);
                        st.note_admitted();
                        st.note_prefix_hit();
                        return;
                    }
                    Err(e) => {
                        // The session was restored onto a slot (or was
                        // already resident) but its turn cannot begin —
                        // falling through to the cold-admit path would let
                        // lru_idle_victim close the just-restored session
                        // (it holds a slot but is not in `busy`) and
                        // double-occupy the slot. Reject and close instead.
                        let _ = send_event(
                            &req.reply,
                            Event::Rejected {
                                class: RejectClass::Internal,
                                reason: format!("continuation begin_turn failed: {e}"),
                            },
                        );
                        rig.swap.forget(existing.0);
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, existing);
                        lock_stats(stats).note_rejected();
                        return;
                    }
                }
            }
        }
        if rig.gpu.slot_trace() {
            eprintln!("[slot-trace] continuation MISS -- falling back to cold prefill");
        }
    }

    // Prefill-side context-cap guard for cold admits. Rejected, not Done: a
    // successful empty completion would read as the model choosing silence;
    // the prompt physically cannot fit the slot's KV.
    if req.prompt_tokens.len() >= rig.cap_tokens {
        let reason = format!(
            "prompt of {} tokens exceeds the slot context cap of {} tokens",
            req.prompt_tokens.len(),
            rig.cap_tokens
        );
        let _ = send_event(
            &req.reply,
            Event::Rejected {
                class: RejectClass::Validation,
                reason,
            },
        );
        lock_stats(stats).note_rejected();
        return;
    }

    // Try to open; when admission refuses, park the LRU idle resident session
    // to swap and retry — bounded by how many idle residents there are.
    //
    // Admission is scoped to GPU residency (#776): parking a session returns
    // its slot AND its grant while its snapshot stays restorable (restore
    // re-takes the grant first). So one remedy covers both shortages —
    // PoolFull (a slot is missing) and WouldExceedBudget (grant bytes are
    // missing). Swapped and Cold sessions hold neither, so they are never
    // victims here. Only IDLE sessions are touched — never an active one.
    //
    // The grant is the REQUEST's maximum remaining growth (prompt +
    // max_tokens, clamped to the context cap), not the whole cap (spec
    // §5.1). A full-cap grant charged ~1.7 GiB of KV credit per session on
    // a big-cap deployment: after trunk weights, the SECOND concurrent
    // request's grant did not fit and it parked behind the first until its
    // session closed — slots serialized for long generations even though
    // both had free pages and free rows.
    let grant_tokens = req
        .prompt_tokens
        .len()
        .saturating_add(req.max_tokens.max(1))
        .min(rig.cap_tokens);
    let mut opened = admit_cold(rig, &mut cold_reuse, grant_tokens);
    // KVMem: never *evict* a resident session to make room. Its pages are the
    // residency decision KVMem already made, and a swap snapshot cannot
    // represent spilled logical pages — a victim would be lost work at best
    // and a corrupt snapshot at worst. A resident session is also never
    // continuable under KVMem (every turn cold-prefills), so an idle one is
    // simply closed: its slot and grant come back, and the slot's KVMem host
    // tier is rebuilt when the new session begins. Without this a full slot
    // set could only park (nothing else frees the slot), which is the
    // serve-hang the terminal close above prevents.
    while opened.is_err() {
        // Re-derive the victim each pass: the previous park consumed it.
        let Some(victim) = rig.sessions.lru_idle_victim(&busy) else {
            break;
        };
        if !kvmem_hooks::evict_for_admit(rig, victim, stats) {
            let _ = send_event(
                &req.reply,
                Event::Rejected {
                    class: RejectClass::Internal,
                    reason: "eviction failed".to_string(),
                },
            );
            lock_stats(stats).note_rejected();
            return;
        }
        lock_stats(stats).note_eviction();
        opened = admit_cold(rig, &mut cold_reuse, grant_tokens);
    }
    let id = match opened {
        Ok(id) => id,
        Err(e) => {
            // Every idle resident is already parked, so what grant bytes
            // remain belong to generating sessions. With none generating the
            // request can never fit — reject it as Overload at once rather
            // than burn the queue timeout. Otherwise it waits below: a
            // finishing generation turns idle and the retry can park it.
            if matches!(e, AdmitError::WouldExceedBudget { .. }) && busy.is_empty() {
                let _ = send_event(
                    &req.reply,
                    Event::Rejected {
                        class: RejectClass::Overload,
                        reason: format!("admission budget exhausted: {e}"),
                    },
                );
                lock_stats(stats).note_rejected();
                return;
            }
            // R-A4 replacement (spec §5.3 S3): when every resident session
            // is generating and no idle victim can make room, enqueue the
            // request in the bounded WaitQueue instead of rejecting
            // immediately. The request is parked and retried when a slot
            // frees; a per-waiter tick deadline drives timeout expiry.
            if should_wait_instead_of_reject(true, rig.wait_queue.max_count() > 0) {
                let waiter_id = rig.next_waiter_id;
                rig.next_waiter_id += 1;
                // queue_bytes is the canonical pending-input bytes from the
                // HTTP admission queue (spec §5.3). Fall back to 1 so a
                // request that bypassed the byte-bounded queue still has a
                // non-zero byte charge (WaitQueue enforces a non-zero byte
                // cap but a zero-byte waiter would never trip it).
                let bytes = req.queue_bytes.max(1);
                // Preserve the retrying request's ORIGINAL admission age (see
                // the drain loop's `pending_repark_age` handoff); a fresh
                // submission is stamped now.
                let enqueued_at = rig
                    .pending_repark_age
                    .take()
                    .unwrap_or_else(std::time::Instant::now);
                match rig.wait_queue.try_enqueue(waiter_id, bytes, rig.tick) {
                    Ok(()) => {
                        rig.parked_requests
                            .insert(waiter_id, ParkedRequest { req, enqueued_at });
                        return;
                    }
                    Err(we) => {
                        // Queue full or byte cap exceeded: surface as a
                        // typed overload rejection (HTTP 429 at the front
                        // end). The request is not parked.
                        let _ = send_event(
                            &req.reply,
                            Event::Rejected {
                                class: RejectClass::Overload,
                                reason: format!("serve queue full: {we}"),
                            },
                        );
                        lock_stats(stats).note_rejected();
                        return;
                    }
                }
            }
            // WaitQueue disabled (should not happen — Rig::build refuses
            // zero caps): fall back to the original immediate reject.
            let _ = send_event(
                &req.reply,
                Event::Rejected {
                    class: RejectClass::Internal,
                    reason: format!("{e:?}"),
                },
            );
            lock_stats(stats).note_rejected();
            return;
        }
    };

    let slot = match rig.sessions.get(id).and_then(|s| s.slot) {
        Some(s) => s,
        None => {
            let _ = send_event(
                &req.reply,
                Event::Rejected {
                    class: RejectClass::Internal,
                    reason: "admitted session holds no slot".to_string(),
                },
            );
            lock_stats(stats).note_rejected();
            return;
        }
    };

    // Reset the slot's recurrent state before reusing it. `seq_len = 0` clears
    // the KV, but DeltaNet state lives OUTSIDE the KV arena, so without this a
    // new conversation inherits the previous occupant's recurrent state --
    // which shows up as degenerate or echoed output on every request after the
    // first. Same trap as the swap unit: KV alone is not the whole state.
    if let Err(e) = rig.model.dn_states[slot.0].reset(&mut rig.gpu) {
        let _ = send_event(
            &req.reply,
            Event::Rejected {
                class: RejectClass::Internal,
                reason: format!("state reset failed: {e}"),
            },
        );
        rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
        lock_stats(stats).note_rejected();
        return;
    }

    // Reset MTP head KV for the new conversation (stale positions from the
    // previous occupant would poison the first draft step).
    if let Some(state) = rig.model.mtp_states[slot.0].as_mut() {
        let _ = state.reset(&mut rig.gpu);
    }

    // Drop the previous occupant's DFlash2 draft state: its target_hidden
    // ring and draft KV are conversation-scoped, and the next request
    // re-allocates lazily on its first prefill chunk.
    if let Some(st) = rig.model.dflash_states[slot.0].take() {
        st.free_gpu(&mut rig.gpu);
    }

    // Reset the KVMem host state and request metadata for the new conversation.
    if let Err(e) = kvmem_hooks::reset_slot(rig, slot, &req) {
        let _ = send_event(
            &req.reply,
            Event::Rejected {
                class: RejectClass::Internal,
                reason: e,
            },
        );
        rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
        lock_stats(stats).note_rejected();
        return;
    }

    // ── P4.3 KVMem-native prefix reuse (HIPFIRE_KVMEM_PREFIX_PIN) ─────────
    // Cross-session reuse from the canonical catalog (the radix stays OFF
    // under KVMem). Order: provenance-validated host fork → DeltaNet restore
    // → GPU page share. Any failure unwinds to the cold state (host reset +
    // DN reset; a refused share is non-mutating by contract).
    //
    // G7: vision requests participate too, gated by the media identity. A
    // checkpoint covering an image span is eligible only when the incoming
    // request's digest (this turn's image, or the session's sticky digest for
    // a text continuation) equals it — token proofs cannot distinguish two
    // images (identical `image_pad` ids).
    let mut prefix_reused: usize = 0;
    let mut prefix_hit = false;
    if let Some(frontier) = kvmem_hooks::prefix_reuse(&mut *rig, slot, id, &req) {
        prefix_reused = frontier;
        prefix_hit = true;
    }

    // ── Prefix cache lookup (spec §4.5–4.6) ──────────────────────────────
    // Cross-session prefix reuse: consult the radix index for reusable KV
    // pages and the checkpoint pool for recurrent state. This is SEPARATE
    // from the session-local continuation path (convo hashes) — the radix
    // index is keyed by exact token sequences across sessions. Vision
    // requests skip the radix entirely (spec X2: Vision+prefix reuse OFF).
    // The lookup's pin on the matched radix path. Held until the request
    // terminates (released by release_pin_ticket on every exit path) or
    // released immediately below when the hit does not convert.
    let mut pin_ticket = PinTicket::none();
    if should_lookup_prefix(rig.prefix_cache, req.visual_data.is_some()) {
        rig.lookup_count += 1;
        let domain = rig.cache_domain.as_ref().unwrap();
        let (lookup_result, hit_handles, ticket) = {
            let pp = rig
                .pool
                .page_pool()
                .expect("prefix_cache requires paged pool");
            rig.prefix_index.as_mut().unwrap().lookup_with_pages(
                domain,
                &req.prompt_tokens,
                pp,
                None,
            )
        };
        pin_ticket = ticket;
        if let PrefixLookupResult::Hit(lookup) = &lookup_result {
            let ckpt_pool = rig.model.checkpoint_pool.as_mut().unwrap();
            // Truthful drafter decision (spec §4.5, made before execution):
            // there is no drafter checkpoint in the pool, so a resumed
            // request that will run MTP brings the head up via the existing
            // reseed path (head KV refills during the suffix prefill);
            // everything else takes AR. MTP honors sampling and token
            // penalties (residual acceptance + exact-AR prepass), so neither
            // bars the prediction — keep it in sync with the admit `mtp_ok`.
            let mtp_will_be_active = rig.model.mtp_head.is_some()
                && rig.model.mtp_k > 0
                && req.visual_data.is_none()
                && req.json_schema.is_none();
            let drafter = if mtp_will_be_active {
                DrafterDecision::Reseed
            } else {
                DrafterDecision::Ar
            };
            match plan_resume(ckpt_pool, domain, &req.prompt_tokens, lookup, drafter) {
                Ok(plan) => {
                    let boundary = plan.boundary as usize;
                    if reused_tokens_from_plan(boundary, req.prompt_tokens.len()) > 0 {
                        let n_pages = boundary / PAGE_TOKENS;
                        let phys_pages: Vec<u32> = hit_handles
                            .iter()
                            .take(n_pages)
                            .map(|h| h.handle.phys)
                            .collect();
                        if n_pages > 0 && phys_pages.len() == n_pages {
                            // Restore the recurrent state FIRST and
                            // fail-closed (spec §5.4 S4): the slot's
                            // DeltaNetState was just reset, so a failed
                            // restore must not leave shared KV paired with
                            // zeroed recurrent state — that silently
                            // corrupts the suffix. On failure we fall back
                            // to the cold path below (begin_turn resets the
                            // table; the DN state is still the reset one).
                            let restore_ok = ckpt_pool
                                .peek(
                                    domain,
                                    boundary as u64,
                                    crate::checkpoint::prefix_fingerprint(
                                        &req.prompt_tokens[..boundary],
                                    ),
                                )
                                .map(|snapshot| {
                                    snapshot
                                        .restore_to(&mut rig.model.dn_states[slot.0], &mut rig.gpu)
                                        .is_ok()
                                })
                                .unwrap_or(false);
                            let shared_ok = restore_ok
                                && rig.pool.share_published_pages(slot, &phys_pages).is_ok();
                            if shared_ok {
                                prefix_reused = boundary;
                                prefix_hit = true;
                                if rig.gpu.slot_trace() {
                                    eprintln!(
                                        "[slot-trace] prefix cache HIT: \
                                         reused {}/{} tokens",
                                        boundary,
                                        req.prompt_tokens.len()
                                    );
                                }
                            } else {
                                // Fail-closed on BOTH halves of the pairing
                                // (spec §5.4 S4): a failed restore may have
                                // advanced the DN state part-way to
                                // S_boundary, and a failed share leaves it at
                                // S_boundary. Either way a cold prefill from
                                // position 0 on top of that state corrupts
                                // the suffix silently — reset to the initial
                                // state before falling through.
                                let _ = rig.model.dn_states[slot.0].reset(&mut rig.gpu);
                                eprintln!(
                                    "[prefix-cache] hit at boundary {boundary} did not \
                                     convert (restore_ok={restore_ok}) — cold prefill"
                                );
                            }
                        }
                    }
                    // The plan pinned this boundary against eviction; the
                    // restore attempt is over (or was skipped) either way,
                    // so release it on every Ok(plan) exit path.
                    if plan.boundary > 0 {
                        ckpt_pool.unpin(
                            domain,
                            plan.boundary,
                            crate::checkpoint::prefix_fingerprint(
                                &req.prompt_tokens[..plan.boundary as usize],
                            ),
                        );
                    }
                }
                Err(_) => {} // No checkpoint at any boundary: cold prefill
            }
        }
        // Release the pin when the hit did not convert — a miss means we
        // won't use those pages, so the path must be evictable again.
        if !prefix_hit {
            release_pin_ticket(rig, &mut pin_ticket);
        }
    }

    // ── Cache reclaim under page pressure (spec §4.4 C4) ────────────────
    // Running work has priority over cache-only residency: before this
    // request's prefill starts allocating, make sure the pool can back the
    // suffix plus the generation budget. If free pages are short, evict
    // oldest unpinned radix leaves (their cache-only pages return to the
    // free list) rather than letting a mid-prefill provision fail.
    if rig.prefix_cache {
        let suffix_tokens = req.prompt_tokens.len().saturating_sub(prefix_reused);
        let needed_pages = suffix_tokens
            .saturating_add(req.max_tokens.max(1))
            .div_ceil(PAGE_TOKENS)
            .saturating_add(1);
        let page_bytes = rig
            .pool
            .page_pool()
            .map(|pp| pp.k_page_bytes() + pp.v_page_bytes())
            .unwrap_or(0);
        let free = rig.pool.page_pool().map(|pp| pp.free_pages()).unwrap_or(0);
        if page_bytes > 0 && free < needed_pages {
            let evict_bytes = needed_pages.saturating_sub(free) * page_bytes;
            let evicted = if let (Some(idx), Some(_domain)) =
                (rig.prefix_index.as_mut(), rig.cache_domain.as_ref())
            {
                let pp = rig.pool.page_pool_mut().expect("paged pool");
                idx.evict_unpinned_leaves(pp, evict_bytes)
            } else {
                Vec::new()
            };
            if !evicted.is_empty() && rig.gpu.slot_trace() {
                eprintln!(
                    "[slot-trace] reclaimed {} cache-only pages under admission pressure",
                    evicted.len()
                );
            }
        }
    }

    #[cfg(feature = "kvmem")]
    kvmem_engine::kvmem_reclaim_prefixes(rig);

    // ── Idle-session reclaim under page pressure (spec §5.4 S4 row 3) ──
    // Radix-leaf eviction only frees cache-only pages. Pages held by idle
    // resident sessions' table refs are neither free nor radix-evictable,
    // so a shortfall here means the forward WILL fail mid-prefill with
    // "need N more pages". Evict LRU idle sessions (snapshot to swap, or
    // cold on snapshot failure) until the demand is backed; if no victim
    // remains, reject at admit instead of letting the forward die.
    //
    // KVMem skips this entirely: the prompt's length is NOT a page demand —
    // everything past the per-slot residency budget is the host store's
    // job, and the pool is sized for every slot's resident working set. The
    // only growth a slot can need is its selection headroom, which the plan
    // already reserved.
    if rig.pool.is_paged() && kvmem_hooks::page_demand_allowed(rig) {
        let suffix_tokens = req.prompt_tokens.len().saturating_sub(prefix_reused);
        let needed_pages = suffix_tokens
            .saturating_add(req.max_tokens.max(1))
            .div_ceil(PAGE_TOKENS)
            .saturating_add(1);
        // The just-opened session holds a slot but is not in `busy` —
        // exclude it explicitly so it is never its own eviction victim.
        let mut exclude: Vec<SessionId> = busy.clone();
        exclude.push(id);
        loop {
            let free = rig.pool.page_pool().map(|pp| pp.free_pages()).unwrap_or(0);
            if free >= needed_pages {
                break;
            }
            let Some(victim) = rig.sessions.lru_idle_victim(&exclude) else {
                break;
            };
            if !evict(rig, victim, stats) {
                break;
            }
            lock_stats(stats).note_eviction();
            if rig.gpu.slot_trace() {
                eprintln!(
                    "[slot-trace] evicted idle session {victim:?} for page demand \
                     ({free} free < {needed_pages} needed)"
                );
            }
        }
        let free = rig.pool.page_pool().map(|pp| pp.free_pages()).unwrap_or(0);
        if free < needed_pages {
            let _ = send_event(
                &req.reply,
                Event::Rejected {
                    class: RejectClass::Overload,
                    reason: format!(
                        "page demand exceeds pool: need {needed_pages} pages, \
                         {free} free after reclaim"
                    ),
                },
            );
            release_pin_ticket(rig, &mut pin_ticket);
            rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
            lock_stats(stats).note_rejected();
            return;
        }
    }

    // Session-local prefix reuse (convo hash continuation) OR prefix cache
    // hit. When the prefix cache hit, begin_turn is skipped because it
    // would reset seq_len to 0, undoing the shared pages.
    let reused = if prefix_hit {
        if let Some(sess) = rig.sessions.get_mut(id) {
            sess.next_pos = prefix_reused;
        }
        prefix_reused
    } else {
        // Prefix reuse: a fresh session reuses nothing, a continued one
        // reuses its common prefix. Either way the suffix is what gets
        // prefilled.
        let plan = match rig
            .sessions
            .begin_turn(&mut rig.pool, id, &req.prompt_tokens)
        {
            Ok(p) => p,
            Err(e) => {
                let _ = send_event(
                    &req.reply,
                    Event::Rejected {
                        class: RejectClass::Internal,
                        reason: e,
                    },
                );
                rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
                lock_stats(stats).note_rejected();
                return;
            }
        };
        plan.reused
    };
    if let Some(sess) = rig.sessions.get_mut(id) {
        sess.tokens = req.prompt_tokens.clone();
        sess.convo = req.convo.clone();
        // An image turn defines the conversation's M-RoPE phase offset for
        // every later text continuation; a text-only conversation stays 0
        // (fresh sessions are, but this path also serves continuation-miss
        // re-prefills, which must not inherit anything stale).
        sess.rope_delta = req
            .visual_data
            .as_ref()
            .map(|vd| vd.rope_delta)
            .unwrap_or(0);
        // Sticky image marker: `rope_delta` is 0 for a thin (single-row/col)
        // image, so it cannot stand in for "this conversation carries an
        // image turn". `has_image` is the publish/lookup gate for the prefix
        // cache (spec §6 X2).
        sess.has_image = sess.has_image || req.visual_data.is_some();
        // G7: sticky media identity of the newest image turn. A text
        // continuation inherits it so a checkpoint covering the image span
        // stays eligible only for the same image.
        if let Some(vd) = req.visual_data.as_ref() {
            sess.media_digest = Some(vd.media_digest());
            sess.media_n_visual = vd.n_visual_tokens;
        }
    }

    if send_event(
        &req.reply,
        Event::Accepted {
            session: id.0,
            reused: reused,
            prefill: req.prompt_tokens.len() - reused,
        },
    )
    .is_err()
    {
        // A prefix hit still holds its PinTicket — release it before
        // closing so the path does not pin the radix forever.
        release_pin_ticket(rig, &mut pin_ticket);
        rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
        return;
    }

    work[slot.0].decoding = false;
    // Built before `req.visual_data` is moved out below; SlotSampleParams is
    // Copy so this is a plain host-side struct build.
    let penalized = request_penalized(&req);
    let sample_params = SlotSampleParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        repeat_window: effective_repeat_window(
            req.repeat_window,
            req.repeat_penalty,
            req.presence_penalty,
            req.frequency_penalty,
        ) as i32,
        repeat_penalty: req.repeat_penalty,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        min_p: req.min_p,
    };
    // Compile the JSON Schema (if present) into a per-request SchemaMatcher
    // cursor (spec §7 G1/G2). The schema was already validated in
    // validate_generate_caps; recompile here to build the per-request
    // mutable cursor. CompiledSchema is immutable and Send+Sync; the
    // SchemaMatcher is request-local mutable state (spec §7.2 G2).
    let grammar_constraint = req.json_schema.as_ref().and_then(|schema| {
        match cached_compile_schema(rig, schema) {
            Ok(compiled) => Some(GrammarConstraint {
                matcher: grammar::json_schema::SchemaMatcher::from_compiled(&compiled),
                mask_buf: Vec::new(),
                in_think: req.started_in_think,
                think_open_id: rig.tokenizer.special_token_id("<think>"),
                think_tail: Vec::new(),
                think_close_id: rig.tokenizer.special_token_id("</think>"),
                think_budget: req.think_budget,
                think_tokens_used: 0,
                mask_cache: std::collections::HashMap::new(),
                mask_cache_order: std::collections::VecDeque::new(),
            }),
            Err(e) => {
                // Should not happen (validated at submit), but fail closed.
                let _ = send_event(
                    &req.reply,
                    Event::Rejected {
                        class: RejectClass::Validation,
                        reason: format!("json_schema compile failed on admit: {e}"),
                    },
                );
                release_pin_ticket(rig, &mut pin_ticket);
                rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
                lock_stats(stats).note_rejected();
                return None;
            }
        }
    });
    // If compilation failed and we already sent Rejected, bail out.
    if req.json_schema.is_some() && grammar_constraint.is_none() {
        return;
    }
    let grammar_constrained = grammar_constraint.is_some();
    if let Some(vd) = req.visual_data {
        // E14: Legacy KVMem does not support vision + speculative decoding.
        // The image branch below would force `SpecKind::None`, silently
        // disabling the draft the user asked for. Refuse explicitly instead
        // of faking an AR pass.
        #[cfg(feature = "kvmem")]
        if kvmem_legacy_enabled(rig) && (rig.model.mtp_head.is_some() || rig.model.dflash.is_some())
        {
            let _ = send_event(
                &req.reply,
                Event::Rejected {
                    class: RejectClass::Validation,
                    reason: "kvmem: vision + speculative decoding is UNSUPPORTED; \
                             run image requests with --spec off"
                        .to_string(),
                },
            );
            release_pin_ticket(rig, &mut pin_ticket);
            rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
            lock_stats(stats).note_rejected();
            return;
        }
        // G7: a converted prefix hit prefills only the suffix. The M-RoPE
        // table stays absolute (`base = 0`, indexed by the absolute `pos`),
        // and visual_idx skips the image pads that already live in the
        // imported prefix (freeze never splits an image span, so the prefix
        // holds the whole image or none of it).
        let pad = rig
            .tokenizer
            .special_token_id("<|image_pad|>")
            .unwrap_or(u32::MAX);
        let prefix_pads = req.prompt_tokens[..reused]
            .iter()
            .filter(|&&t| t == pad)
            .count()
            .min(vd.n_visual_tokens);
        work[slot.0].remaining_prompt = req.prompt_tokens[reused..].to_vec();
        work[slot.0].next_pos = reused;
        work[slot.0].pos3_delta = 0; // phases come from the VL table below
                                     // Drop any previous request's vision state: the new request's
                                     // embeddings are not produced until its first batched chunk, and the
                                     // scheduler holds the slot's rows until then. Free the old matrix +
                                     // job explicitly — dropping them unfreed leaks VRAM (the buffers are
                                     // neither pooled nor hipFree'd by Drop).
        rig.model.clear_slot_vl_state(&mut rig.gpu, slot.0);
        work[slot.0].vl_prefill = Some(VlPrefill {
            patches: vd.patches,
            grid_h: vd.grid_h,
            grid_w: vd.grid_w,
            n_visual_tokens: vd.n_visual_tokens,
            embeddings: Vec::new(),
            dim: rig.model.config.dim,
            visual_idx: prefix_pads,
            image_pad_id: pad,
            mrope_positions: vd.mrope_positions,
            rope_delta: vd.rope_delta,
            base: 0,
        });
        work[slot.0].spec = SpecKind::None;
    } else {
        work[slot.0].remaining_prompt = req.prompt_tokens[reused..].to_vec();
        work[slot.0].next_pos = reused;
        work[slot.0].vl_prefill = None;
        work[slot.0].pos3_delta = 0; // fresh session: no image-turn offset
                                     // Activate MTP for text-only requests when the head is loaded.
                                     // Sampling and token penalties ride the MTP path: the
                                     // request's `MtpSamplingConfig` (parked in
                                     // `spec_req_sampling`) makes verify use residual acceptance
                                     // against the penalized target nucleus, reproducing the AR
                                     // distribution. Grammar-constrained requests stay on AR:
                                     // constrained MTP requires joint verification (applying
                                     // grammar masks before acceptance, not only after a bad token
                                     // has advanced state), which does not yet exist (spec §6 X2:
                                     // "MTP + grammar → AR selected before execution until exact
                                     // joint acceptance is implemented and validated").
                                     // Radix-reuse retires drafting: the head's private KV is zero for
                                     // the reused span (shared pages carry only trunk KV), and drafting
                                     // over a zeroed prefix collapses acceptance to ~1 — pure overhead
                                     // on exactly the turns reuse was meant to accelerate. AR produces
                                     // identical output without the overhead.
                                     // P4.3: under KVMem PREFIX_PIN + Layerwise the reused prefix's head
                                     // history lives in the shared composite aux archive and the draft
                                     // carrier restores it before attend (P4.1② G4) — the zeroed-head-KV
                                     // concern above does not apply, so reuse keeps MTP (and the suffix
                                     // prefill co-archives the aux plane the composite export requires).
                                     // Legacy reuse stays AR (no aux restore path). The radix is off under
                                     // KVMem, so `reused > 0` here is always a P4.3 canonical reuse.
        let kvmem_lw_reuse = kvmem_hooks::layerwise_reuse(rig);
        let mtp_ok = rig.model.mtp_head.is_some()
            && rig.model.mtp_k > 0
            && !grammar_constrained
            && (reused == 0 || kvmem_lw_reuse);
        // DFlash2 admit on the new-session arm: same request-shape gates as
        // MTP (text only, no grammar), including the reused==0 rule — a radix
        // hit's reused span was never forwarded this turn, so its
        // extract-layer hiddens are not captured and the ring would carry
        // uninitialized rows for 0..reused (the append_committed gap guard).
        // Continuations are the exception: their ring survives on the slot
        // and the continuation arm checks its watermark. Sampling and token
        // penalties are honored by the verify accept (point-mass residual +
        // exact-AR prepass), so neither bars admission.
        // Legacy-mode ctx fit (sequential `spec_ctx_request_fits` parity):
        // the Legacy scatter writes target_hidden at the ABSOLUTE position
        // into a `ctx_capacity`-row buffer, so a request that can grow past
        // `ctx_capacity` would drive an out-of-bounds scatter (a release
        // `assert!` panic in `sub_offset` — the serve-hang class). Windowed
        // mode has no such bound (its rings wrap), so the gate is Legacy-only.
        let dflash_ctx_fits = rig.model.dflash.as_ref().map_or(false, |d| {
            d.window.is_some()
                || req
                    .prompt_tokens
                    .len()
                    .saturating_add(req.max_tokens.max(1))
                    .saturating_add(d.block_size)
                    <= d.ctx_capacity
        });
        // 期4: a Layerwise canonical reuse may keep DFlash2 when the
        // checkpoint carried the draft conditioning image and it was restored
        // at exactly the reused frontier (nInfer native full-state
        // serialization). Without an image the ring has no rows for
        // 0..reused, so the request runs AR (the historical rule). Legacy
        // reuse keeps AR unconditionally (byte/time-neutral fail-closed path).
        let dflash_reused_ready = kvmem_lw_reuse
            && reused > 0
            && rig.model.dflash_states[slot.0]
                .as_ref()
                .is_some_and(|st| st.seeded_through == reused);
        let dflash_ok = rig.model.dflash.is_some()
            && !grammar_constrained
            && dflash_ctx_fits
            && (reused == 0 || dflash_reused_ready);
        work[slot.0].spec = if mtp_ok {
            SpecKind::Mtp
        } else if dflash_ok {
            SpecKind::Dflash
        } else {
            SpecKind::None
        };
        #[cfg(feature = "kvmem")]
        {
            rig.legacy_spec_kind[slot.0] = work[slot.0].spec;
        }
        rig.spec_counters[slot.0] = SpecCounters {
            requested: work[slot.0].spec.active(),
            ..Default::default()
        };
        rig.model.spec_req_sampling[slot.0] = if work[slot.0].spec.active() {
            Some(MtpReqSampling::new(&req, &req.prompt_tokens))
        } else {
            None
        };
        work[slot.0].spec_retire =
            hipfire_runtime::spec_retire::SpecRetire::new(rig.model.spec_retire_break_even);
        if rig.gpu.slot_trace() {
            eprintln!(
                "[slot-trace] admit(cold) slot {} spec={:?} temp={} top_k={} top_p={} \
                 penalized={} reused={} prompt={}",
                slot.0,
                work[slot.0].spec,
                req.temperature,
                req.top_k,
                req.top_p,
                penalized,
                reused,
                req.prompt_tokens.len()
            );
        }
        // ── Legacy inactive-history restore (Sessions = non-active) ─────────
        // A parked native image is content-addressed by its complete accepted
        // token prefix, and the request's rendered tokens extend that prefix.
        // Restoring reinstates target KV, the recurrent state and the parked
        // working set, so only the suffix is prefilled — the full accepted
        // prefix (assistant/tool turns included) proves the reuse, never the
        // conversation hash. Only exact draft provenance is reused; a
        // draft-less image retires speculation for this request instead of
        // inheriting another request's draft rows.
        #[cfg(feature = "kvmem")]
        if kvmem_legacy_enabled(rig) && reused == 0 && req.prompt_tokens.len() > 1 {
            let prompt = &req.prompt_tokens;
            // A restored frontier must leave at least the last prompt token to
            // prefill: that step produces the logits the first output token is
            // sampled from. A parked image whose frontier already covers the
            // whole rendered prompt cannot seed a decode (there is no KV op to
            // re-open the tail), so it is not a candidate here.
            let max_frontier = prompt.len() - 1;
            let media = req
                .visual_data
                .as_ref()
                .map(|vd| vd.media_digest())
                .or_else(|| rig.sessions.get(id).and_then(|s| s.media_digest));
            let candidate = rig.legacy_history.as_ref().and_then(|catalog| {
                catalog
                    .lookup(prompt, media, max_frontier, true)
                    .ok()
                    .and_then(|ids| ids.into_iter().next())
                    .and_then(|entry_id| {
                        catalog.get(entry_id).map(|entry| {
                            (
                                entry_id,
                                entry.source_session,
                                std::rc::Rc::clone(&entry.value),
                            )
                        })
                    })
            });
            if let Some((entry_id, source_session, image)) = candidate {
                // A parked image carries no native draft state (parking captures
                // the target store only). Restoring it under MTP/DFlash would
                // silently degrade the admitted execution model to AR, which the
                // acceptance contract forbids. Treat that as a cache miss and
                // cold-prefill with the requested spec instead — nothing has been
                // reset at this point.
                if !image.supports(work[slot.0].spec) {
                    eprintln!(
                        "[kvmem-legacy] HistorySkip slot={} entry={entry_id} reason=spec_mismatch requested={:?} frontier={}",
                        slot.0, work[slot.0].spec, image.store.frontier,
                    );
                } else {
                    let kind = work[slot.0].spec;
                    let selected = image.store.resident_pages.clone();
                    match restore_legacy_native_image(rig, slot.0, id.0, &image, &selected, kind) {
                        Ok(bytes) => {
                            let frontier = image.store.frontier as usize;
                            work[slot.0].spec = kind;
                            rig.spec_counters[slot.0] = SpecCounters {
                                requested: kind.active(),
                                ..Default::default()
                            };
                            if !kind.active() {
                                rig.model.spec_req_sampling[slot.0] = None;
                            }
                            work[slot.0].next_pos = frontier;
                            work[slot.0].remaining_prompt = prompt[frontier..].to_vec();
                            if let Some(sess) = rig.sessions.get_mut(id) {
                                sess.next_pos = frontier;
                            }
                            if let Some(catalog) = rig.legacy_history.as_mut() {
                                let _ = catalog.touch(entry_id);
                            }
                            eprintln!(
                            "[kvmem-legacy] HistoryRestore slot={} session={} entry={entry_id} source_session={source_session} frontier={frontier} suffix={} restore_bytes={bytes} spec={kind:?}",
                            slot.0,
                            id.0,
                            prompt.len() - frontier,
                        );
                        }
                        Err(e) => {
                            // The store was reset before the failure: this slot
                            // must not serve on a torn frontier. Fail the request
                            // (never fall through into a cold prefill the store
                            // cannot back).
                            let _ = send_event(
                                &req.reply,
                                Event::Rejected {
                                    class: RejectClass::Internal,
                                    reason: format!("kvmem: history restore failed: {e}"),
                                },
                            );
                            rig.sessions.close(&mut rig.pool, &mut rig.adm, id);
                            lock_stats(stats).note_rejected();
                            return;
                        }
                    }
                }
            }
        }
    }
    rig.sample_params[slot.0] = sample_params;
    // A prefix-cache hit's shared pages are already radix-owned up to
    // `reused`; seed both the session and the InFlight so this request's
    // publications start beyond them (spec §4.6 C6, see the continuation
    // admit for the re-publication leak this prevents).
    if prefix_hit {
        if let Some(sess) = rig.sessions.get_mut(id) {
            sess.published_boundary = reused;
        }
    }
    let repeat_window_req = effective_repeat_window(
        req.repeat_window,
        req.repeat_penalty,
        req.presence_penalty,
        req.frequency_penalty,
    );
    slots[slot.0] = Some(InFlight {
        session: id,
        reply: req.reply,
        produced: 0,
        max_tokens: req.max_tokens.max(1),
        repeat_window_req,
        last_published_boundary: reused,
        last_frozen_boundary: reused,
        last_coverage_boundary: 0,
        #[cfg(feature = "kvmem")]
        checkpoint_plan: rig.kvmem_checkpoint_policy.and_then(|(i, _)| {
            hipfire_kvmem::CheckpointCapturePlan::new(reused as u64, i, 0, 0).ok()
        }),
        grammar: grammar_constraint,
        pin_ticket,
    });
    if rig.gpu.slot_trace() {
        eprintln!(
            "[slot-trace] cold ADMIT session {} slot={} prompt={} tokens",
            id.0,
            slot.0,
            req.prompt_tokens.len()
        );
    }
    // FairQueue: admit the cold request with its prefill suffix as the
    // cache-locality weight (0 for a fully-reused prefix-cache hit).
    let uncached = work[slot.0].remaining_prompt.len() as u64;
    let _ = rig.fair_queue.admit(id.0, "default", uncached);
    if prefix_hit {
        lock_stats(stats).note_reused_tokens(prefix_reused);
    }
    lock_stats(stats).note_admitted();
}

/// Read the session sweep policy from the process config. The engine already
/// reads developer config directly; these two keys are process-scoped policy,
/// like the swap budgets.
pub(super) fn session_policy_from_config() -> (Option<std::time::Duration>, usize) {
    let process = hipfire_config::active_or_local_process_config();
    let value = |key: &str| {
        process.values.get(key).and_then(|value| match value {
            hipfire_config::ConfigValue::Integer(i) => u64::try_from(*i).ok(),
            _ => None,
        })
    };
    let ttl = value("serve.multi_slot_session_ttl_secs").unwrap_or(3600);
    let max = value("serve.multi_slot_max_sessions").unwrap_or(0);
    (
        if ttl == 0 {
            None
        } else {
            Some(std::time::Duration::from_secs(ttl))
        },
        usize::try_from(max).unwrap_or(usize::MAX),
    )
}

/// Close parked sessions that aged out and enforce the hard cap. The table
/// owns the age/limit policy inputs; the engine drops their swap entries.
///
/// The TTL is wall-clock and the cap is generous (16 parked conversations per
/// slot, at least 64): a parked session is only a cache entry, so closing one
/// costs a cold re-prefill, never correctness.
pub(super) fn sweep_sessions(rig: &mut Rig) {
    const CAP_MIN: usize = 64;
    const PER_SLOT: usize = 16;
    let ttl = rig.session_ttl.unwrap_or(std::time::Duration::MAX);
    let cap = if rig.session_max == 0 {
        rig.n_slots.saturating_mul(PER_SLOT).max(CAP_MIN)
    } else {
        rig.session_max
    };
    let closed = rig.sessions.sweep_expired(
        &mut rig.pool,
        &mut rig.adm,
        std::time::Instant::now(),
        ttl,
        cap,
    );
    if closed.is_empty() {
        return;
    }
    if let Ok(mut ids) = rig.closed_sessions.lock() {
        ids.extend(closed.iter().map(|id| id.0));
    }
    if rig.gpu.slot_trace() {
        eprintln!(
            "[slot-trace] swept {} parked session(s) (ttl {ttl:?}, cap {cap})",
            closed.len()
        );
    }
    for id in closed {
        rig.swap.forget(id.0);
        // The session is gone; drop any fairness grant it still holds. (The
        // normal Close command does the same.)
        let _ = rig.fair_queue.remove(id.0);
    }
}

/// Admit the cold path's session. When a matched session's snapshot was lost,
/// restore marked that session Cold; reopen the same id (it still owns the
/// conversation identity) instead of minting a new one, which would strand the
/// old entry in the table forever.
fn admit_cold(
    rig: &mut Rig,
    reuse: &mut Option<SessionId>,
    grant_tokens: usize,
) -> Result<SessionId, AdmitError> {
    if let Some(id) = *reuse {
        match rig
            .sessions
            .reopen_cold(&mut rig.pool, &mut rig.adm, id, grant_tokens)
        {
            Ok(id) => return Ok(id),
            Err(AdmitError::UnknownSession(_)) => *reuse = None,
            Err(e) => return Err(e),
        }
    }
    rig.sessions.open(&mut rig.pool, &mut rig.adm, grant_tokens)
}

/// Capture an idle session's state, park it, and free its slot.
///
/// Returns false if anything went wrong, in which case the session is marked
/// `Cold`: its tokens survive, so it re-prefills next time. Slow, never wrong.
pub(super) fn evict(rig: &mut Rig, victim: SessionId, stats: &Arc<Mutex<EngineStats>>) -> bool {
    let Some(sess) = rig.sessions.get(victim) else {
        return false;
    };
    let Some(slot) = sess.slot else { return false };
    let tokens = sess.tokens.clone();

    let dn_refs = dn_buffers(&rig.model.dn_states[slot.0]);
    let snap = capture_slot(
        &mut rig.gpu,
        &rig.pool,
        slot,
        &rig.model.k_arenas,
        &rig.model.v_arenas,
        &dn_refs,
        &tokens,
        rig.stamp,
    );
    drop(dn_refs);

    let evicted = match snap {
        Ok(snap) => match rig.swap.park(victim.0, snap) {
            Ok(()) => {
                rig.sessions
                    .mark_swapped(&mut rig.pool, &mut rig.adm, victim);
                true
            }
            Err(e) => {
                warn_once(
                    &SWAP_PARK_WARNED,
                    &format!(
                        "swap park failed for session {}: {e}; the session is Cold and will re-prefill",
                        victim.0
                    ),
                );
                rig.sessions.mark_cold(&mut rig.pool, &mut rig.adm, victim);
                true
            }
        },
        Err(e) => {
            warn_once(
                &SWAP_PARK_WARNED,
                &format!(
                    "swap snapshot capture failed for session {}: {e}; the session is Cold and will re-prefill",
                    victim.0
                ),
            );
            rig.sessions.mark_cold(&mut rig.pool, &mut rig.adm, victim);
            true
        }
    };
    lock_stats(stats).note_swap(&rig.swap);
    evicted
}

/// Restore a previously swapped session into `slot`. Any failure marks it
/// `Cold` so the caller re-prefills from tokens.
fn restore(rig: &mut Rig, id: SessionId, slot: SlotId, stats: &Arc<Mutex<EngineStats>>) -> bool {
    // A swapped session gave its VRAM admission back on eviction; take it
    // again before its state re-enters the GPU. If the budget cannot cover it
    // the snapshot is dropped and the session goes Cold (re-prefill path).
    if let Err(e) = rig.sessions.readmit(&mut rig.adm, id) {
        warn_once(
            &SWAP_RESTORE_WARNED,
            &format!(
                "swap restore failed for session {} ({e}); the session is Cold and will re-prefill",
                id.0
            ),
        );
        rig.swap.forget(id.0);
        rig.sessions.mark_cold(&mut rig.pool, &mut rig.adm, id);
        lock_stats(stats).note_swap(&rig.swap);
        return false;
    }
    match rig.swap.unpark(id.0) {
        Ok(snap) => {
            let dn_refs = dn_buffers(&rig.model.dn_states[slot.0]);
            let r = restore_slot(
                &mut rig.gpu,
                &mut rig.pool,
                slot,
                &rig.model.k_arenas,
                &rig.model.v_arenas,
                &dn_refs,
                &snap,
                rig.stamp,
            );
            drop(dn_refs);
            // MTP head KV is per-generation, not per-session. Reset it so
            // the restored session starts a fresh MTP prefill.
            if let Some(state) = rig.model.mtp_states[slot.0].as_mut() {
                let _ = state.reset(&mut rig.gpu);
            }
            // DFlash2 draft state is likewise per-generation: the restored
            // session re-seeds target_hidden from its suffix forward.
            if let Some(st) = rig.model.dflash_states[slot.0].take() {
                st.free_gpu(&mut rig.gpu);
            }
            match r {
                Ok(()) => {
                    rig.sessions.mark_resident(id, slot, snap.seq_len);
                    lock_stats(stats).note_swap(&rig.swap);
                    true
                }
                Err(e) => {
                    warn_once(
                        &SWAP_RESTORE_WARNED,
                        &format!(
                            "swap restore failed for session {} ({e}); the session is Cold and will re-prefill",
                            id.0
                        ),
                    );
                    rig.sessions.mark_cold(&mut rig.pool, &mut rig.adm, id);
                    lock_stats(stats).note_swap(&rig.swap);
                    false
                }
            }
        }
        Err(e) => {
            warn_once(
                &SWAP_RESTORE_WARNED,
                &format!(
                    "swap restore failed for session {} ({e}); the session is Cold and will re-prefill",
                    id.0
                ),
            );
            rig.sessions.mark_cold(&mut rig.pool, &mut rig.adm, id);
            lock_stats(stats).note_swap(&rig.swap);
            false
        }
    }
}
