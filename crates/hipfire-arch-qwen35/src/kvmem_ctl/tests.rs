// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
mod head_plane;
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;
use rdna_compute::DType;

#[test]
fn legacy_window_plan_keeps_budget_separate_from_reserve() {
    let p = resolve_legacy_plan(1, 32768, 256, 8192, 1024, None).unwrap();
    assert_eq!(p.budget_blocks, 64);
    assert_eq!(p.pages_total, 72);
    assert_eq!(p.grow_blocks, 2);
    let two = resolve_legacy_plan(2, 32768, 512, 8192, 2048, None).unwrap();
    assert_eq!(two.pages_total, 160);
    assert_eq!(two.budget_blocks, 64);
}

#[test]
fn legacy_window_plan_refuses_silent_budget_clamping() {
    let error = resolve_legacy_plan(1, 65536, 256, 36864, 1024, Some(160)).unwrap_err();
    assert!(error.contains("not reduced automatically"), "{error}");
    assert!(resolve_legacy_plan(1, 32768, 256, 8192, 1024, Some(71)).is_err());
    assert!(resolve_legacy_plan(1, 32768, 256, 8192, 1024, Some(72)).is_ok());
}

#[test]
fn legacy_window_plan_checks_context_step_and_alignment() {
    assert!(resolve_legacy_plan(1, 9216, 1024, 8192, 1024, None).is_ok());
    assert!(resolve_legacy_plan(1, 9215, 1024, 8192, 1024, None).is_err());
    assert!(resolve_legacy_plan(1, 32768, 1025, 8192, 1024, None).is_err());
    assert!(resolve_legacy_plan(1, 32768, 128, 255, 1024, None).is_err());
    assert!(resolve_legacy_plan(1, 32768, 128, 8193, 1024, None).is_err());
    assert!(resolve_legacy_plan(1, 32768, 128, 8192, 127, None).is_err());
    assert!(resolve_legacy_plan(0, 32768, 256, 8192, 1024, None).is_err());
}

#[test]
fn implicit_legacy_budget_clamps_to_context_and_explicit_stays_loud() {
    // Stock 32k deployment with no explicit budget: the default 32768
    // would fail B + R <= C; the implicit clamp makes it admissible.
    let implicit = effective_legacy_budget(32768, true, 32768, 1024);
    assert_eq!(implicit, 31744);
    assert!(resolve_legacy_plan(1, 32768, 256, implicit, 1024, None).is_ok());
    // An explicit budget is never reduced: the operator asked for it.
    assert_eq!(effective_legacy_budget(32768, false, 32768, 1024), 32768);
    assert!(resolve_legacy_plan(1, 32768, 256, 32768, 1024, None).is_err());
    // Clamping floors to whole blocks and leaves sub-minimum contexts loud.
    assert_eq!(effective_legacy_budget(32768, true, 9216, 1025), 8064);
    assert_eq!(effective_legacy_budget(2048, true, 32768, 1024), 2048);
    assert_eq!(effective_legacy_budget(32768, true, 512, 1024), 0);
}

/// Small Q8 geometry: 2 fa layers, 2 KV heads, head_dim 64 →
/// `per_pos_bytes = 2 * (64/32) * 34 = 136`.
fn geom() -> KvMemGeometry {
    KvMemGeometry {
        n_fa_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 64,
        kv_mode: KvMode::Q8,
        per_pos_bytes: 136,
        per_pos_v_bytes: 136,
        max_rows: 512,
        mtp_head: false,
    }
}
/// P4.1② G3: the same tiny geometry with the native MTP head enabled, so the
/// composite host record (`n_fa_layers + 1` ordinals) and the head-plane
/// lifecycle can be exercised host-only.
fn geom_mtp() -> KvMemGeometry {
    KvMemGeometry {
        mtp_head: true,
        ..geom()
    }
}
fn plan() -> KvMemPlan {
    KvMemPlan {
        pages_total: 16,
        budget_blocks: 4,
        grow_blocks: 2,
        interval: 1,
    }
}

fn test_options() -> KvMemOptions {
    let mut options = KvMemOptions::default();
    options.host_bytes = Some(16 << 20);
    options
}

fn cache_test_options(dir: &std::path::Path, bytes: u64) -> KvMemOptions {
    let mut options = test_options();
    options.content_cache_dir = Some(dir.to_string_lossy().into_owned());
    options.content_cache_bytes = Some(bytes);
    options
}

/// [P4.1② fix] host-only: the per-request scorer-Query gate is armed by
/// `begin_request` (continuation admit), consumed exactly once by
/// `take_layerwise_query_due`, re-armed by the next request, and cleared by
/// `reset_slot`. Legacy never reads it (the carrier returns None early), so
/// arming it is byte-neutral there.
#[test]
fn per_request_query_gate_lifecycle() {
    let mut ctl = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("layerwise ctl");
    // Cold slot: no request begun ⇒ gate idle.
    assert!(!ctl.take_layerwise_query_due(0));
    // A continuation admit arms the gate; the first decode step consumes it.
    ctl.begin_request(0).expect("begin_request");
    assert!(
        ctl.take_layerwise_query_due(0),
        "first take after begin_request"
    );
    assert!(
        !ctl.take_layerwise_query_due(0),
        "gate consumed exactly once"
    );
    // The next continuation re-arms it (per-request Query semantics).
    ctl.begin_request(0).expect("begin_request 2");
    assert!(ctl.take_layerwise_query_due(0));
    // A fresh session drops the gate.
    ctl.begin_request(0).expect("begin_request 3");
    ctl.reset_slot(0).expect("reset_slot");
    assert!(
        !ctl.take_layerwise_query_due(0),
        "reset_slot clears the gate"
    );
    // Out-of-range slot is a no-op (false), never panics.
    assert!(!ctl.take_layerwise_query_due(99));
}

/// Host-only: the strategy parser/resolver mirror the core ordinals and
/// reject unknown spellings loudly, and the write-through staging is
/// P4.1③ H5: the head-history policy resolution contract (pure, env-snapshot
/// independent). FirstLayer auto-default under Layerwise+MTP; UpstreamGlobal
/// opt-in; invalid spelling rejected; a resolved policy off the Layerwise+MTP
/// path rejected (no silent fallback). Legacy / AR-only carry Unspecified.
#[test]
fn layerwise_mtp_history_resolution() {
    use crate::layerwise_mtp::LayerwiseMtpHistory::{FirstLayer, Unspecified, UpstreamGlobal};
    // Absent → FirstLayer auto-default under Layerwise+MTP.
    assert_eq!(
        resolve_layerwise_mtp_history(None, RecallStrategy::Layerwise, true).unwrap(),
        FirstLayer
    );
    // Explicit opt-in + trimming.
    assert_eq!(
        resolve_layerwise_mtp_history(
            Some("upstream-global".into()),
            RecallStrategy::Layerwise,
            true
        )
        .unwrap(),
        UpstreamGlobal
    );
    assert_eq!(
        resolve_layerwise_mtp_history(
            Some("  first-layer  ".into()),
            RecallStrategy::Layerwise,
            true
        )
        .unwrap(),
        FirstLayer
    );
    // Invalid spelling rejected loudly.
    assert!(
        resolve_layerwise_mtp_history(Some("bogus".into()), RecallStrategy::Layerwise, true)
            .is_err()
    );
    // A resolved policy requires Layerwise + MTP (no silent fallback).
    assert!(resolve_layerwise_mtp_history(
        Some("upstream-global".into()),
        RecallStrategy::Legacy,
        true
    )
    .is_err());
    assert!(resolve_layerwise_mtp_history(
        Some("first-layer".into()),
        RecallStrategy::Layerwise,
        false
    )
    .is_err());
    // Legacy / AR-only carry Unspecified unread (byte-neutral).
    assert_eq!(
        resolve_layerwise_mtp_history(None, RecallStrategy::Legacy, false).unwrap(),
        Unspecified
    );
    assert_eq!(
        resolve_layerwise_mtp_history(None, RecallStrategy::Legacy, true).unwrap(),
        Unspecified
    );
}

/// A1 starvation floor (single source of truth): the `layerwise_mtp_history()`
/// accessor reports FirstLayer for an opted-in UpstreamGlobal whenever
/// `budget_blocks < MTP_HISTORY_UG_MIN_BLOCKS`, so EVERY decision point
/// (prefill-seed aux binding + the serve_engine AR-decode / MTP-verify carrier
/// restore-hooks) falls back uniformly. At/above the floor UG is reported
/// unchanged; FirstLayer / Unspecified are never altered. Host-only.
#[test]
fn layerwise_mtp_history_starvation_floor() {
    use crate::layerwise_mtp::LayerwiseMtpHistory::{FirstLayer, Unspecified, UpstreamGlobal};
    let mut ctl = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("layerwise mtp ctl");
    // plan() budget_blocks=4 (< floor). Opted-in UG reports FirstLayer.
    ctl.layerwise_mtp_history = UpstreamGlobal;
    assert!(ctl.budget_blocks < MTP_HISTORY_UG_MIN_BLOCKS);
    assert_eq!(
        ctl.layerwise_mtp_history(),
        FirstLayer,
        "below-floor UG falls back"
    );
    // Exactly at the floor UG engages (>= guard).
    ctl.budget_blocks = MTP_HISTORY_UG_MIN_BLOCKS;
    assert_eq!(
        ctl.layerwise_mtp_history(),
        UpstreamGlobal,
        "at-floor UG engages"
    );
    // Above the floor UG engages.
    ctl.budget_blocks = MTP_HISTORY_UG_MIN_BLOCKS + 8;
    assert_eq!(
        ctl.layerwise_mtp_history(),
        UpstreamGlobal,
        "above-floor UG engages"
    );
    // One below the floor falls back again.
    ctl.budget_blocks = MTP_HISTORY_UG_MIN_BLOCKS - 1;
    assert_eq!(
        ctl.layerwise_mtp_history(),
        FirstLayer,
        "one-below-floor falls back"
    );
    // FirstLayer / Unspecified are never altered by the floor.
    ctl.layerwise_mtp_history = FirstLayer;
    assert_eq!(ctl.layerwise_mtp_history(), FirstLayer);
    ctl.layerwise_mtp_history = Unspecified;
    assert_eq!(ctl.layerwise_mtp_history(), Unspecified);
}

/// allocated ONLY under Layerwise (Legacy stays byte-neutral — no extra
/// host buffer, no arena touch). The strategy is INJECTED, not `set_var`d:
/// the process config is a start-time snapshot, so a runtime env write after
/// the first config read would be an order-dependent coin flip.
#[test]
fn recall_strategy_parse_and_staging_gate() {
    assert_eq!(
        parse_recall_strategy("legacy"),
        Some(RecallStrategy::Legacy)
    );
    assert_eq!(
        parse_recall_strategy("Layerwise"),
        Some(RecallStrategy::Layerwise)
    );
    assert_eq!(parse_recall_strategy("0"), Some(RecallStrategy::Legacy));
    assert_eq!(parse_recall_strategy("1"), Some(RecallStrategy::Layerwise));
    assert_eq!(
        parse_recall_strategy("  LEGACY  "),
        Some(RecallStrategy::Legacy)
    );
    assert_eq!(parse_recall_strategy("bogus"), None);
    assert!(!RecallStrategy::Legacy.is_layerwise());
    assert!(RecallStrategy::Layerwise.is_layerwise());

    // Pure resolution contract (the exact body `new` runs on the env value).
    assert_eq!(
        resolve_recall_strategy(None).unwrap(),
        RecallStrategy::Legacy
    );
    assert_eq!(
        resolve_recall_strategy(Some("legacy".into())).unwrap(),
        RecallStrategy::Legacy
    );
    assert_eq!(
        resolve_recall_strategy(Some("layerwise".into())).unwrap(),
        RecallStrategy::Layerwise
    );
    assert!(resolve_recall_strategy(Some("bogus".into())).is_err());

    // Small host ceiling for the tiny test geometry. The store commits
    // lazily, so the default ceiling is also safe; this just keeps the
    // reservation modest when this test is the first config reader.

    // Legacy: no write-through staging.
    let legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert_eq!(legacy.strategy(), RecallStrategy::Legacy);
    assert!(legacy.wt_k.is_empty() && legacy.wt_v.is_empty());
    // Layerwise: staging sized to max_rows * per_pos_bytes.
    let lw = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("layerwise ctl");
    assert_eq!(lw.strategy(), RecallStrategy::Layerwise);
    assert_eq!(lw.wt_k.len(), 512 * 136);
    assert_eq!(lw.wt_v.len(), 512 * 136);

    // The env wrapper resolves the Legacy default when the knob is absent
    // (the ambient test env never sets RECALL_STRATEGY, so the snapshot
    // cannot carry it regardless of read order).
    let default = KvMemCtl::new_default(1, &plan(), &geom()).expect("default ctl");
    assert_eq!(default.strategy(), RecallStrategy::Legacy);
    assert!(default.wt_k.is_empty() && default.wt_v.is_empty());
}

/// Host-only: the Layerwise fail-closed policy guards (design §4, R2d-D1).
/// The knobs are read through the process-start config snapshot
/// (`hipfire_config::developer_var` is a `OnceLock`), so a runtime
/// `set_var` is invisible — the guard decision is extracted pure
/// (`resolve_layerwise_policy`, same injection rationale as
/// `resolve_recall_strategy`) and the full table is tested here: under
/// Layerwise each global working-set knob the per-layer subsystem replaces
/// (select_method≠retrieval, lrfu, watermark, query_span>1) is an explicit
/// Err naming both the knob and the strategy — no silent override/fallback;
/// Legacy admits every combination (the guard is Layerwise-specific, so the
/// default serve stays byte-neutral). The all-default Layerwise controller
/// still constructs end-to-end, proving the wired guard call admits the
/// default path.
#[test]
fn layerwise_policy_knobs_fail_closed() {
    // (select_method, lrfu, watermark_enabled, query_span, expected):
    // None = admitted, Some(fragment) = Err naming that knob. Each
    // non-default knob alone fails closed; the combined row proves the
    // first violation wins.
    let cases: &[(u32, bool, bool, usize, Option<&str>)] = &[
        (0, false, false, 1, None),
        (2, false, false, 1, Some("SELECT_METHOD")),
        (0, true, false, 1, Some("LRFU")),
        (0, false, true, 1, Some("WATERMARK")),
        (0, false, false, 2, Some("QUERY_SPAN")),
        (2, true, true, 8, Some("SELECT_METHOD")),
    ];
    for &(sm, lrfu, wm, qs, expected) in cases {
        let lw = resolve_layerwise_policy(RecallStrategy::Layerwise, sm, lrfu, wm, qs);
        match expected {
            None => lw.unwrap_or_else(|e| {
                panic!("default Layerwise must admit (sm={sm} lrfu={lrfu} wm={wm} qs={qs}): {e}")
            }),
            Some(frag) => {
                let msg = lw.err().unwrap_or_else(|| {
                    panic!("Layerwise must fail closed for sm={sm} lrfu={lrfu} wm={wm} qs={qs}")
                });
                assert!(
                    msg.contains("layerwise"),
                    "guard message names the strategy for {frag}: {msg}"
                );
                assert!(
                    msg.contains(frag),
                    "guard message names the knob {frag}: {msg}"
                );
            }
        }
        // Legacy admits every combination — byte-neutral guard gating.
        resolve_layerwise_policy(RecallStrategy::Legacy, sm, lrfu, wm, qs).unwrap_or_else(|e| {
            panic!("Legacy must admit sm={sm} lrfu={lrfu} wm={wm} qs={qs}: {e}")
        });
    }

    // End-to-end: the wired guard call admits the all-default Layerwise
    // construction (HOST_BYTES is best-effort through the snapshot; the
    // store commits lazily, so the default ceiling is also safe).
    let ctl = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("default layerwise ctl");
    assert_eq!(ctl.strategy(), RecallStrategy::Layerwise);
}

/// The Layerwise driver borrows the slot's statistics store + policy through
/// `layerwise_stores`: distinct objects per slot, `None` out of range, the
/// policy borrow is live (`&self` budget set), and `reset_slot` builds a
/// fresh empty store (the inline `SlotState` address is stable, so state —
/// not pointer identity — is the probe) — the lifetime contract that forces
/// the driver to be rebuilt per query (§3.9b).
#[test]
fn layerwise_stores_accessor_borrows_slot_state() {
    let mut ctl =
        KvMemCtl::new_with_strategy(2, &plan(), &geom(), RecallStrategy::Layerwise).expect("ctl");
    let (raw0, kv0) = ctl.layerwise_stores(0).expect("slot 0");
    let (raw1, kv1) = ctl.layerwise_stores(1).expect("slot 1");
    assert!(
        !std::ptr::eq(raw0, raw1),
        "per-slot raw stores are distinct"
    );
    assert!(!std::ptr::eq(kv0, kv1), "per-slot policies are distinct");
    assert!(ctl.layerwise_stores(2).is_none(), "out of range → None");
    // The policy borrow is live: a &self budget set through it succeeds.
    // The core admits a runtime budget only inside [sink+recent allocation,
    // configured select_budget]; the configured maximum is always in that
    // window, so it is the safe probe (the fixture's sink+recent floor is
    // strictly below it).
    kv0.set_runtime_select_budget(ctl.kv_cfg.select_budget)
        .expect("budget through borrow");
    // reset_slot builds a FRESH C++ store. SlotState lives inline in the
    // Vec, so its address is stable across reset — probe STATE, not pointer
    // identity. Allocate blocks on the old borrow; the post-reset borrow
    // must report a smaller (fresh, empty) allocation. This is the lifetime
    // contract that forces the driver to be rebuilt per query (§3.9b): a
    // borrow taken before reset references a store that no longer exists.
    raw0.ensure_blocks(4).expect("allocate on old store");
    let old_bytes = raw0.allocated_bytes();
    ctl.reset_slot(0).expect("reset");
    let (raw0_new, _) = ctl.layerwise_stores(0).expect("slot 0 after reset");
    assert!(
        raw0_new.allocated_bytes() < old_bytes,
        "reset yields a fresh empty store"
    );
}

/// Host-only: the decision-A (§3.10) persistent pressure owner + keep-set
/// lifecycle. Legacy never creates the owner (fail-closed `ensure`, `None`
/// borrow, empty keep-set — byte-neutral). Layerwise constructs lazily and
/// idempotently with `block_tokens=128` pinned to the catalog and the select
/// ceiling derived from the resident budget; the keep-set round-trips; and
/// `reset_slot` drops BOTH the owner and the keep-set (a fresh session never
/// aliases the previous session's accepted mass / selected pages). Finally,
/// the bootstrap canonical `resident` is driven by the keep-set (filtered to
/// the sealed history), replacing the pure re-bootstrap `[]`.
#[test]
fn layerwise_pressure_owner_and_keep_set_lifecycle() {
    let fa_layers = [0u32, 1];

    // Legacy: the owner is never created; ensure fails closed.
    let mut legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert!(legacy.layerwise_pressure(0).is_none(), "Legacy: no owner");
    assert_eq!(
        legacy.layerwise_keep_set(0),
        Some(&[][..]),
        "Legacy: empty keep-set"
    );
    assert!(
        legacy
            .ensure_layerwise_pressure(0, 0, 0, &fa_layers)
            .is_err(),
        "Legacy: ensure must fail closed (byte-neutral)"
    );
    assert!(
        legacy.layerwise_pressure(0).is_none(),
        "Legacy: still no owner after ensure"
    );

    // Layerwise: lazy, idempotent construction.
    let mut lw = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("layerwise ctl");
    assert!(
        lw.layerwise_pressure(0).is_none(),
        "before ensure: no owner"
    );
    assert!(
        lw.layerwise_pressure_mut(0).is_none(),
        "before ensure: no mut owner"
    );
    lw.ensure_layerwise_pressure(0, 0, 0, &fa_layers)
        .expect("construct owner");
    let opts = lw
        .layerwise_pressure(0)
        .expect("owner")
        .options()
        .expect("options");
    assert_eq!(
        opts.block_tokens, KV_BLOCK_TOKENS as u32,
        "block_tokens pinned to catalog"
    );
    assert_eq!(
        opts.select_tokens,
        (plan().budget_blocks * KV_BLOCK_TOKENS) as u64,
        "select ceiling = resident budget"
    );
    // Idempotent within a session: a second ensure is a no-op, owner persists.
    lw.ensure_layerwise_pressure(0, 0, 0, &fa_layers)
        .expect("idempotent ensure");
    assert!(
        lw.layerwise_pressure(0).is_some(),
        "owner persists across ensure"
    );
    assert!(
        lw.layerwise_pressure_mut(0).is_some(),
        "mut borrow available"
    );

    // Per-layer keep-set round-trip (selection-state persistence).
    assert_eq!(
        lw.layerwise_keep_set(0),
        Some(&[][..]),
        "initial keep-set empty"
    );
    lw.set_layerwise_keep_set(0, vec![vec![0, 2, 3], vec![1, 2]]);
    assert_eq!(
        lw.layerwise_keep_set(0),
        Some(&[vec![0u64, 2, 3], vec![1u64, 2]][..]),
        "per-layer keep-set round-trip"
    );

    // take/restore detaches the owner for the scoped forward borrow, then
    // returns it (the slot keeps its keep-set / counters throughout).
    let detached = lw.take_layerwise_pressure(0).expect("take owner");
    assert!(lw.layerwise_pressure(0).is_none(), "owner detached");
    assert_eq!(
        lw.layerwise_keep_set(0),
        Some(&[vec![0u64, 2, 3], vec![1u64, 2]][..]),
        "keep-set survives take"
    );
    lw.restore_layerwise_pressure(0, detached);
    assert!(lw.layerwise_pressure(0).is_some(), "owner restored");

    // Out-of-range accessors are None / Err (no panic).
    assert!(lw.layerwise_pressure(9).is_none());
    assert!(lw.layerwise_keep_set(9).is_none());
    assert!(lw.take_layerwise_pressure(9).is_none());
    assert!(lw.ensure_layerwise_pressure(9, 0, 0, &fa_layers).is_err());

    // reset_slot drops the owner AND the keep-set; incarnation bumps.
    let (inc_before, _) = lw.layerwise_counters(0).expect("counters");
    lw.reset_slot(0).expect("reset");
    assert!(lw.layerwise_pressure(0).is_none(), "reset drops the owner");
    assert_eq!(
        lw.layerwise_keep_set(0),
        Some(&[][..]),
        "reset clears the keep-set"
    );
    let (inc_after, _) = lw.layerwise_counters(0).expect("counters after reset");
    assert_eq!(inc_after, inc_before + 1, "reset bumps incarnation");
    // A fresh session reconstructs under the new incarnation.
    lw.ensure_layerwise_pressure(0, 0, inc_after, &fa_layers)
        .expect("reconstruct");
    assert!(lw.layerwise_pressure(0).is_some(), "reconstructed owner");

    // The bootstrap canonical resident is driven PER-LAYER by the keep-set,
    // filtered to the sealed history (fed=512 ⇒ 4 full pages; id 99 is out of
    // history). Ordinal 0 keeps [0,2], ordinal 1 keeps [1] (independent).
    let boot = crate::layerwise_driver::bootstrap_layerwise(
        512,
        0,
        inc_after,
        0,
        &fa_layers,
        4,
        2,
        64,
        4,
        136,
        &[vec![0, 2, 99], vec![1]],
    )
    .expect("bootstrap with keep-set");
    assert_eq!(
        boot.canonical[0].memory.resident,
        vec![0u64, 2],
        "ordinal 0 filtered keep-set"
    );
    assert_eq!(
        boot.canonical[1].memory.resident,
        vec![1u64],
        "ordinal 1 independent keep-set"
    );
    let boot_empty = crate::layerwise_driver::bootstrap_layerwise(
        512,
        0,
        inc_after,
        0,
        &fa_layers,
        4,
        2,
        64,
        4,
        136,
        &[],
    )
    .expect("bootstrap empty keep-set");
    for snap in &boot_empty.canonical {
        assert!(
            snap.memory.resident.is_empty(),
            "empty keep-set ⇒ re-bootstrap []"
        );
    }
}

/// Host-only: the P4.1② MTP-verify output-pressure owner lifecycle — the
/// partial-acceptance sibling of the prefill owner above. Legacy never
/// creates it (fail-closed `ensure`, `None` borrow — byte-neutral, the AR
/// path is untouched). Layerwise constructs lazily and idempotently in its
/// OWN request space (session id/incarnation offset +1 from the prefill
/// owner), NonThinking scope, under the same finite fail-closed admission
/// ceiling; `take`/`restore` detach it for the scoped verify borrow without
/// disturbing the prefill owner / keep-set; and `reset_slot` drops it (a
/// fresh session never aliases the previous session's accepted output mass).
/// Exercises all five owner methods so none is dead code.
#[test]
fn layerwise_output_pressure_owner_lifecycle() {
    let fa_layers = [0u32, 1];

    // Legacy: the output owner is never created; ensure fails closed.
    let mut legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert!(
        legacy.layerwise_output_pressure(0).is_none(),
        "Legacy: no output owner"
    );
    assert!(
        legacy
            .ensure_layerwise_output_pressure(0, 0, 0, &fa_layers)
            .is_err(),
        "Legacy: ensure must fail closed (byte-neutral)"
    );
    assert!(
        legacy.layerwise_output_pressure(0).is_none(),
        "Legacy: still no output owner after ensure"
    );

    // Layerwise: lazy, idempotent construction.
    let mut lw = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("layerwise ctl");
    assert!(
        lw.layerwise_output_pressure(0).is_none(),
        "before ensure: no output owner"
    );
    assert!(
        lw.layerwise_output_pressure_mut(0).is_none(),
        "before ensure: no mut output owner"
    );
    lw.ensure_layerwise_output_pressure(0, 7, 3, &fa_layers)
        .expect("construct output owner");
    // The output window lives in its own request space: session id/incarnation +1.
    {
        let owner = lw.layerwise_output_pressure(0).expect("output owner");
        let snap = owner.snapshot().expect("snapshot");
        let session = snap.session().expect("session");
        assert_eq!(session.id, 8, "output session id = session_id + 1");
        assert_eq!(
            session.incarnation, 4,
            "output incarnation = incarnation + 1"
        );
        assert_eq!(
            snap.layer_count(),
            fa_layers.len(),
            "one output state per FA layer"
        );
        assert_eq!(
            snap.read().expect("read").accepted_end,
            0,
            "fresh owner: no accepted output"
        );
    }
    // Idempotent within a session: a second ensure is a no-op, owner persists.
    lw.ensure_layerwise_output_pressure(0, 7, 3, &fa_layers)
        .expect("idempotent ensure");
    assert!(
        lw.layerwise_output_pressure(0).is_some(),
        "output owner persists across ensure"
    );
    assert!(
        lw.layerwise_output_pressure_mut(0).is_some(),
        "mut borrow available"
    );

    // The output owner is independent of the prefill owner: constructing it
    // does not create the prefill owner, and vice versa.
    assert!(
        lw.layerwise_pressure(0).is_none(),
        "output ensure does not build the prefill owner"
    );
    lw.ensure_layerwise_pressure(0, 7, 3, &fa_layers)
        .expect("construct prefill owner");
    assert!(
        lw.layerwise_output_pressure(0).is_some(),
        "prefill ensure leaves the output owner"
    );

    // take/restore detaches the output owner for the scoped verify borrow,
    // then returns it (the prefill owner / keep-set survive throughout).
    lw.set_layerwise_keep_set(0, vec![vec![0, 2], vec![1]]);
    let detached = lw
        .take_layerwise_output_pressure(0)
        .expect("take output owner");
    assert!(
        lw.layerwise_output_pressure(0).is_none(),
        "output owner detached"
    );
    assert!(
        lw.layerwise_pressure(0).is_some(),
        "prefill owner survives output take"
    );
    assert_eq!(
        lw.layerwise_keep_set(0),
        Some(&[vec![0u64, 2], vec![1u64]][..]),
        "keep-set survives output take"
    );
    lw.restore_layerwise_output_pressure(0, detached);
    assert!(
        lw.layerwise_output_pressure(0).is_some(),
        "output owner restored"
    );

    // Out-of-range accessors are None / Err (no panic).
    assert!(lw.layerwise_output_pressure(9).is_none());
    assert!(lw.layerwise_output_pressure_mut(9).is_none());
    assert!(lw.take_layerwise_output_pressure(9).is_none());
    assert!(lw
        .ensure_layerwise_output_pressure(9, 0, 0, &fa_layers)
        .is_err());
    // restore out of range is a silent no-op (never panics, never aliases).
    let stray = lw
        .take_layerwise_output_pressure(0)
        .expect("take for stray restore");
    lw.restore_layerwise_output_pressure(9, stray);
    assert!(
        lw.layerwise_output_pressure(0).is_none(),
        "stray restore did not alias slot 0"
    );
    lw.ensure_layerwise_output_pressure(0, 7, 3, &fa_layers)
        .expect("rebuild after stray");

    // reset_slot drops the output owner AND bumps incarnation; the prefill
    // owner is dropped too (a fresh session never aliases either).
    let (inc_before, _) = lw.layerwise_counters(0).expect("counters");
    lw.reset_slot(0).expect("reset");
    assert!(
        lw.layerwise_output_pressure(0).is_none(),
        "reset drops the output owner"
    );
    assert!(
        lw.layerwise_pressure(0).is_none(),
        "reset drops the prefill owner"
    );
    let (inc_after, _) = lw.layerwise_counters(0).expect("counters after reset");
    assert_eq!(inc_after, inc_before + 1, "reset bumps incarnation");
    // A fresh session reconstructs under the new incarnation.
    lw.ensure_layerwise_output_pressure(0, 7, inc_after, &fa_layers)
        .expect("reconstruct");
    assert!(
        lw.layerwise_output_pressure(0).is_some(),
        "reconstructed output owner"
    );
}

/// Host-only regression for P4.1② bug #9: a continuation turn's unsampled
/// Legacy prefill advances the native consumed frontier without the persistent
/// output window observing it. hipfire keeps ONE window per session
/// (request = session_id + 1, `begin_scope` idempotent, never finished), so after
/// the previous turn's verify `accept` the window is left `prepared_ = true`
/// with `accepted_end` at the old frontier; the next turn's `prepare_scope` at
/// the new frontier was rejected ("native grammar verifier differs from accepted
/// frontier"). `sync_prefill` repositions the accepted feedback to the validated
/// native prefix via the core restore primitive (ninfer accept_prefill semantics,
/// NL:8482/17094), preserving the accumulated pending mass + sample count and
/// clearing prepared_. Idempotent / no-op when accepted_end >= position.
#[test]
fn layerwise_output_sync_prefill_continuation() {
    use hipfire_kvmem::{
        MemoryStamp, OutputAcceptance, OutputMass, OutputOptions, OutputPressure, OutputScope,
        OutputScopeRow, OutputScopeState, PrefillOptions, SessionIdentity,
    };
    let session = SessionIdentity {
        id: 8,
        incarnation: 4,
    };
    let fa = [0u32, 1];
    let prefill_opts = PrefillOptions {
        block_tokens: 128,
        select_tokens: 128,
        reserve_tokens: 128,
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
    let mut p = OutputPressure::new_external(session, &fa, &prefill_opts, &output_opts)
        .expect("output owner");
    let stamp = |frontier: u64| MemoryStamp {
        valid: 1,
        session,
        history_revision: 1,
        view_version: 1,
        frontier,
        execution_history: 1,
    };
    let request = 8u64;
    let rows = vec![OutputScopeRow::new(false, false, true); 4];

    // Turn 1: open the window at frontier 384, prepare 4 body rows, accept all
    // four (each row contributes unit mass to logical page 0).
    let f0 = 384u64;
    p.begin_scope(&stamp(f0), request, OutputScopeState::new(false, false))
        .expect("begin_scope turn1");
    let mask = p
        .prepare_scope(request, f0, &rows)
        .expect("prepare_scope turn1");
    assert_eq!(mask, 0b1111, "all four verify rows sampled");
    let acc = OutputAcceptance {
        history: stamp(f0 + 4),
        request,
        accepted: 4,
        rows: vec![0, 1, 2, 3],
        layers: fa
            .iter()
            .map(|&l| OutputMass {
                model_layer: l,
                history: vec![0],
                masses: vec![1.0, 1.0, 1.0, 1.0],
            })
            .collect(),
    };
    p.accept(&acc).expect("accept turn1");
    let s = p.snapshot().expect("snap").read().expect("read");
    assert_eq!(s.accepted_end, f0 + 4, "accept advances accepted_end");
    assert_eq!(
        s.layers[0].pending,
        vec![(0u64, 4.0)],
        "pending mass seeded"
    );
    assert_eq!(s.layers[0].samples, 4, "four sampled rows");

    // Continuation turn 2: Legacy prefill advances the native frontier to 1024
    // WITHOUT the window observing it; begin_scope is a same-request no-op.
    let f1 = 1024u64;
    p.begin_scope(&stamp(f1), request, OutputScopeState::new(false, false))
        .expect("begin_scope no-op");
    // #9 reproduction: prepare_scope at the new frontier is rejected because
    // accepted_end still lags at f0 + 4 (and prepared_ is left true by accept).
    let err = p
        .prepare_scope(request, f1, &rows)
        .expect_err("prefill gap must reject prepare_scope");
    assert!(
        err.0.contains("accepted frontier"),
        "unexpected error: {err}"
    );

    // The fix: sync the accepted feedback to the validated native prefix.
    p.sync_prefill(f1).expect("sync_prefill");
    let s = p.snapshot().expect("snap").read().expect("read");
    assert_eq!(
        s.accepted_end, f1,
        "sync advances accepted_end to the native frontier"
    );
    assert_eq!(
        s.layers[0].pending,
        vec![(0u64, 4.0)],
        "sync preserves pending mass"
    );
    assert_eq!(s.layers[0].samples, 4, "sync preserves sample count");
    assert!(s.active, "window stays active");
    let mask = p
        .prepare_scope(request, f1, &rows)
        .expect("prepare_scope turn2 after sync");
    assert_eq!(mask, 0b1111, "verifier matches the synced frontier");

    // Idempotent: syncing again to the same frontier is a no-op.
    p.sync_prefill(f1).expect("sync idempotent");
    assert_eq!(
        p.snapshot()
            .expect("snap")
            .read()
            .expect("read")
            .accepted_end,
        f1
    );
    // No-op guard: a position at/below accepted_end never rewinds the frontier.
    p.sync_prefill(f0).expect("sync below frontier is a no-op");
    assert_eq!(
        p.snapshot()
            .expect("snap")
            .read()
            .expect("read")
            .accepted_end,
        f1
    );
}

/// Host-only: the pending output-acceptance snapshot's per-slot lifecycle
/// (P4.1②, design §A.6). The orchestration stores the verify carrier's
/// extracted snapshot before freeing it; `mtp_verify_accept_step` takes it
/// after `feed_rows` to publish at the confirmed prefix. Verifies set/take
/// (consume-once), out-of-range safety, independence from the output owner /
/// keep-set, and that `reset_slot` drops it (a fresh session never publishes a
/// stale verify). Legacy never stores one in production (byte-neutral).
#[test]
fn layerwise_pending_output_lifecycle() {
    let mk_snap = |begin: u64| OutputAcceptanceSnapshot {
        base_stamp: hipfire_kvmem::MemoryStamp {
            valid: 1,
            session: hipfire_kvmem::SessionIdentity {
                id: 8,
                incarnation: 4,
            },
            history_revision: 1,
            view_version: 1,
            frontier: 384,
            execution_history: 1,
        },
        request: 8,
        begin,
        eval_rows: 4,
        mass_rows: vec![0, 1, 2, 3],
        layers: Vec::new(),
    };

    let mut lw = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("layerwise ctl");
    // Nothing pending before the first verify.
    assert!(
        lw.take_layerwise_pending_output(0).is_none(),
        "fresh slot: nothing pending"
    );
    // Store + take consumes once; fields survive the round-trip.
    lw.set_layerwise_pending_output(0, mk_snap(384));
    let got = lw
        .take_layerwise_pending_output(0)
        .expect("pending snapshot");
    assert_eq!(got.request, 8);
    assert_eq!(got.begin, 384);
    assert_eq!(got.eval_rows, 4);
    assert_eq!(got.mass_rows, vec![0, 1, 2, 3]);
    assert_eq!(got.base_stamp.frontier, 384);
    assert!(
        lw.take_layerwise_pending_output(0).is_none(),
        "take consumes the snapshot"
    );
    // Out-of-range: set is a silent no-op, take is None (never panics/aliases).
    lw.set_layerwise_pending_output(9, mk_snap(512));
    assert!(
        lw.take_layerwise_pending_output(9).is_none(),
        "out-of-range take"
    );
    assert!(
        lw.take_layerwise_pending_output(0).is_none(),
        "stray set did not alias slot 0"
    );
    // Pending is independent of the output owner / keep-set.
    lw.set_layerwise_pending_output(0, mk_snap(640));
    lw.set_layerwise_keep_set(0, vec![vec![0, 2], vec![1]]);
    assert!(
        lw.layerwise_output_pressure(0).is_none(),
        "pending does not build the owner"
    );
    assert!(
        lw.take_layerwise_pending_output(0).is_some(),
        "pending survives keep-set write"
    );
    // reset_slot drops the pending snapshot (fresh session never publishes stale).
    lw.set_layerwise_pending_output(0, mk_snap(768));
    lw.reset_slot(0).expect("reset");
    assert!(
        lw.take_layerwise_pending_output(0).is_none(),
        "reset drops the pending snapshot"
    );
}

/// Host-only: the SESSION-lifetime pressure owner's finite, fail-closed
/// admission (design §3.11, ③-E4; ninfer essence = "complete mutable history
/// ceiling"). The owner's retained history is bounded by
/// `n_fa_layers × ((budget_blocks + 1) × 24 + overhead)` and must fit the
/// metadata pressure partition (the last transient 1/8, carved from E1's
/// source share) BEFORE `PrefillPressure::new` acquires anything. Pure
/// decision table (the quotas are read through the process-start `OnceLock`
/// snapshot, so a runtime `set_var` is invisible — same injection rationale
/// as `resolve_layerwise_policy`): the default geometry admits, a too-small
/// partition fails closed naming the METADATA knob, the exact boundary
/// admits while one byte over rejects, Legacy never admits the owner, and a
/// saturating-huge geometry stays finite (Err, never a wrap). The wired
/// success path is proven end-to-end by the lifecycle test above.
#[test]
fn layerwise_pressure_admission_fail_closed() {
    let d = LayerwiseQuotas::default();
    // Ceiling formula: 2 layers × ((2+1)×24 + 64) = 2 × 136 = 272B.
    assert_eq!(layerwise_pressure_ceiling(2, 2), 272);
    assert_eq!(layerwise_pressure_ceiling(1, 0), 88);
    // Partition = metadata / 8 (the last transient eighth).
    assert_eq!(layerwise_pressure_cap(d.metadata), 128 << 20);

    // Default quotas admit the tiny geometry with nothing acquired.
    resolve_pressure_admission(RecallStrategy::Layerwise, 2, 2, &d)
        .unwrap_or_else(|e| panic!("default Layerwise pressure must admit: {e}"));

    // Legacy never admits the owner (byte-neutral fail-closed).
    let legacy = resolve_pressure_admission(RecallStrategy::Legacy, 2, 2, &d)
        .err()
        .unwrap_or_else(|| panic!("Legacy must not admit a pressure owner"));
    assert!(
        legacy.contains("Layerwise"),
        "Legacy guard names the strategy: {legacy}"
    );

    // A partition too small for the ceiling fails closed naming METADATA.
    for metadata in [0u64, 8, 703] {
        let q = LayerwiseQuotas {
            metadata,
            ..d.clone()
        };
        let msg = resolve_pressure_admission(RecallStrategy::Layerwise, 1, 0, &q)
            .err()
            .unwrap_or_else(|| panic!("metadata={metadata} must fail closed (ceiling 88B)"));
        assert!(
            msg.contains("layerwise"),
            "admission names the strategy: {msg}"
        );
        assert!(
            msg.contains("HIPFIRE_KVMEM_LAYERWISE_METADATA_MIB"),
            "admission names the knob: {msg}"
        );
    }

    // Exact boundary: ceiling 88B == partition 88B admits; one byte short rejects.
    let exact = LayerwiseQuotas {
        metadata: 704,
        ..d.clone()
    }; // 704/8 == 88
    resolve_pressure_admission(RecallStrategy::Layerwise, 1, 0, &exact)
        .unwrap_or_else(|e| panic!("ceiling == partition must admit: {e}"));

    // Saturating-huge geometry stays finite and fails closed (no wrap).
    assert_eq!(layerwise_pressure_ceiling(usize::MAX, usize::MAX), u64::MAX);
    assert!(
        resolve_pressure_admission(RecallStrategy::Layerwise, usize::MAX, usize::MAX, &d).is_err(),
        "saturating-huge ceiling must fail closed against a finite partition"
    );
}

/// Host-only: the five activation quotas resolve from MiB env integers with
/// the ninfer seeds as defaults, reject malformed/overflowing values loudly
/// (fail-closed, mirroring `resolve_recall_strategy`), and trim whitespace.
/// Pure resolution contract — no process-config snapshot, no GPU.
#[test]
fn layerwise_quotas_resolve_defaults_overrides_fail_closed() {
    let d = LayerwiseQuotas::default();
    // Absent env → the ninfer engine seeds (64/32/8/1024/96 MiB).
    assert_eq!(
        resolve_layerwise_quotas(None, None, None, None, None, None).unwrap(),
        d
    );
    assert_eq!(d.activation_ram, 64 << 20);
    assert_eq!(d.activation_disk, 32 << 20);
    assert_eq!(d.staging, 8 << 20);
    assert_eq!(d.metadata, 1024 << 20);
    assert_eq!(d.activation_metadata, 96 << 20);
    assert_eq!(d.disk_directory, None);

    // Explicit MiB overrides (whitespace trimmed) + the spill-dir passthrough.
    let dir = PathBuf::from("/tmp/hipfire-quota-test");
    let q = resolve_layerwise_quotas(
        Some(" 128 ".into()),
        Some("64".into()),
        Some("16".into()),
        Some("2048".into()),
        Some("192".into()),
        Some(dir.clone()),
    )
    .unwrap();
    assert_eq!(q.activation_ram, 128 << 20);
    assert_eq!(q.activation_disk, 64 << 20);
    assert_eq!(q.staging, 16 << 20);
    assert_eq!(q.metadata, 2048 << 20);
    assert_eq!(q.activation_metadata, 192 << 20);
    assert_eq!(q.disk_directory, Some(dir));

    // Zero is a valid explicit quota (admission rejects it later, not here).
    assert_eq!(
        resolve_layerwise_quotas(Some("0".into()), None, None, None, None, None)
            .unwrap()
            .activation_ram,
        0
    );

    // Malformed / negative / fractional / empty values fail closed (no seed).
    assert!(resolve_layerwise_quotas(Some("abc".into()), None, None, None, None, None).is_err());
    assert!(resolve_layerwise_quotas(Some("-1".into()), None, None, None, None, None).is_err());
    assert!(resolve_layerwise_quotas(Some("1.5".into()), None, None, None, None, None).is_err());
    assert!(resolve_layerwise_quotas(None, Some("".into()), None, None, None, None).is_err());
    // u64::MAX MiB overflows the byte conversion.
    assert!(
        resolve_layerwise_quotas(Some(u64::MAX.to_string()), None, None, None, None, None).is_err()
    );
    // Each quota rejects independently.
    assert!(resolve_layerwise_quotas(None, None, Some("x".into()), None, None, None).is_err());
    assert!(resolve_layerwise_quotas(None, None, None, Some("x".into()), None, None).is_err());
    assert!(resolve_layerwise_quotas(None, None, None, None, Some("x".into()), None).is_err());
}

/// Host-only: the quota accessor is gated by strategy — Legacy carries the
/// unread ninfer seeds (byte-neutral: a malformed quota env can never perturb
/// the default path), while Layerwise resolves them. The ambient test env
/// sets no LAYERWISE_*_MIB, so both resolve to the seeds here; the parsing
/// contract itself is covered by the pure-resolver test above.
#[test]
fn layerwise_quotas_gated_by_strategy() {
    let legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert_eq!(legacy.strategy(), RecallStrategy::Legacy);
    assert_eq!(legacy.layerwise_quotas(), &LayerwiseQuotas::default());

    let lw = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("layerwise ctl");
    assert_eq!(lw.strategy(), RecallStrategy::Layerwise);
    assert_eq!(lw.layerwise_quotas(), &LayerwiseQuotas::default());
}

/// GPU qualification of the write-through gather + seal (design §3.7).
/// Fills the arenas with an independent byte pattern, drives two contiguous
/// feeds (the second starts mid-block), and asserts the sealed archive reads
/// back byte-exact against the pattern oracle resolved through the slot's
/// REAL logical→physical map (a dummy slot offsets the physical pages so
/// `phys != lp`, catching any gather that assumes identity mapping). Covers
/// multi-page spans, a partial tail, and fail-closed on a non-resident page.
///
/// `#[ignore]`d: needs a GPU + HIP toolchain.
///   cargo test -p hipfire-arch-qwen35 --features kvmem --release \
///     --lib kvmem_ctl::tests::write_through_archive_readback_byte_exact -- --ignored
#[test]
#[ignore]
fn write_through_archive_readback_byte_exact() {
    let n_fa = 2usize;
    let row = 136usize;
    // Strategy injected (the config snapshot ignores runtime set_var).
    let mut ctl =
        KvMemCtl::new_with_strategy(2, &plan(), &geom(), RecallStrategy::Layerwise).expect("ctl");

    let mut gpu = Gpu::init().expect("gpu init");
    let mut pool = SlotPool::new_paged_with_strides(2, 1024, row, row, 16).expect("pool");
    let dummy = pool.acquire().expect("dummy slot");
    pool.set_seq_len(dummy, 256).expect("dummy seq"); // grabs the first physical pages
    let slot = pool.acquire().expect("test slot");
    pool.set_seq_len(slot, 512).expect("test seq"); // 4 logical pages, phys != lp

    let kpb = KV_BLOCK_TOKENS * row;
    let arena_bytes = pool.k_arena_bytes();
    // Independent oracle: a deterministic byte pattern keyed on the PHYSICAL
    // page, intra-page position, byte index, layer, and K/V plane.
    let pat = |phys: usize, intra: usize, b: usize, il: usize, kind: usize| -> u8 {
        ((phys * 131 + intra * 17 + b * 7 + il * 251 + kind * 97) & 0xff) as u8
    };
    let mut k_arenas = Vec::new();
    let mut v_arenas = Vec::new();
    for il in 0..n_fa {
        let mut hk = vec![0u8; arena_bytes];
        let mut hv = vec![0u8; arena_bytes];
        for phys in 0..16 {
            for intra in 0..KV_BLOCK_TOKENS {
                for b in 0..row {
                    let off = phys * kpb + intra * row + b;
                    hk[off] = pat(phys, intra, b, il, 0);
                    hv[off] = pat(phys, intra, b, il, 1);
                }
            }
        }
        let tk = gpu
            .alloc_tensor(&[arena_bytes], DType::Raw)
            .expect("k arena");
        gpu.hip.memcpy_htod(&tk.buf, &hk).expect("k h2d");
        let tv = gpu
            .alloc_tensor(&[arena_bytes], DType::Raw)
            .expect("v arena");
        gpu.hip.memcpy_htod(&tv.buf, &hv).expect("v h2d");
        k_arenas.push(tk);
        v_arenas.push(tv);
    }
    gpu.hip.device_synchronize().expect("sync");

    // Two contiguous feeds: 0..300, then 300..400 (mid-block start on page 2).
    ctl.write_through_archive(&gpu, &pool, &k_arenas, &v_arenas, slot.0, 0, 300)
        .expect("feed 0..300");
    ctl.write_through_archive(&gpu, &pool, &k_arenas, &v_arenas, slot.0, 300, 100)
        .expect("feed 300..400");

    // Readback byte-exact against the oracle, per logical page / layer.
    let bt = pool.block_table(slot).expect("block table");
    let total_fed = 400usize;
    let mut kbuf = vec![0u8; kpb];
    let mut vbuf = vec![0u8; kpb];
    for lp in 0..bt.num_pages() {
        let phys = bt.physical(lp).expect("fed page resident") as usize;
        let fed = total_fed
            .saturating_sub(lp * KV_BLOCK_TOKENS)
            .min(KV_BLOCK_TOKENS);
        if fed == 0 {
            continue;
        }
        for il in 0..n_fa {
            let raw = &ctl.slots[slot.0].raw;
            assert!(
                raw.has_k_gpu(lp as u32, il as u32, fed as u32),
                "K present lp={lp}"
            );
            assert!(
                raw.has_v_gpu(lp as u32, il as u32, fed as u32),
                "V present lp={lp}"
            );
            assert!(raw.copy_k_gpu(lp as u32, il as u32, &mut kbuf[..fed * row], fed as u32));
            assert!(raw.copy_v_gpu(lp as u32, il as u32, &mut vbuf[..fed * row], fed as u32));
            for intra in 0..fed {
                for b in 0..row {
                    assert_eq!(
                        kbuf[intra * row + b],
                        pat(phys, intra, b, il, 0),
                        "K mismatch lp={lp} phys={phys} il={il} intra={intra} b={b}"
                    );
                    assert_eq!(
                        vbuf[intra * row + b],
                        pat(phys, intra, b, il, 1),
                        "V mismatch lp={lp} phys={phys} il={il} intra={intra} b={b}"
                    );
                }
            }
        }
    }
    // Fail-closed: a feed beyond the provisioned pages has no resident
    // physical page, so the gather must error before any archive mutation.
    assert!(
        ctl.write_through_archive(&gpu, &pool, &k_arenas, &v_arenas, slot.0, 512, 128)
            .is_err(),
        "non-resident page must fail closed"
    );

    for t in k_arenas {
        gpu.free_tensor(t).expect("free k arena");
    }
    for t in v_arenas {
        gpu.free_tensor(t).expect("free v arena");
    }
}

// ── P4.3 KVMem-native prefix sharing (host model) ─────────────────────

/// Fill slot `slot` with `blocks` full blocks of deterministic mean-K +
/// packed K/V and advance `fed` (host-only; no GPU).
fn prefix_fill(ctl: &mut KvMemCtl, slot: usize, blocks: u32, seed: u32) {
    let bt = KV_BLOCK_TOKENS as u32;
    let k_width = ctl.raw_cfg.n_embd_k as usize;
    let row_bytes = ctl.raw_cfg.k_gpu_row_bytes as usize;
    for b in 0..blocks {
        let pos0 = b * bt;
        for il in 0..ctl.raw_cfg.n_layer {
            let mut mean = Vec::with_capacity(bt as usize * k_width);
            for t in 0..bt as usize {
                for d in 0..k_width {
                    mean.push(
                        (seed as usize * 100003
                            + b as usize * 1009
                            + t * k_width
                            + d
                            + il as usize * 7) as f32,
                    );
                }
            }
            ctl.slots[slot]
                .raw
                .write_layer_mean_k(pos0, bt, il, &mean)
                .expect("mean k");
            let mut rows = vec![0u8; bt as usize * row_bytes];
            for (i, byte) in rows.iter_mut().enumerate() {
                *byte = ((seed as usize + b as usize + il as usize + i) % 251) as u8;
            }
            ctl.slots[slot]
                .raw
                .write_layer_k_gpu(pos0, bt, il, &rows)
                .expect("packed k");
            ctl.slots[slot]
                .raw
                .write_layer_v_gpu(pos0, bt, il, &rows)
                .expect("packed v");
        }
    }
    ctl.test_set_fed(slot, blocks * bt).expect("set fed");
}

fn prefix_tokens(n: usize, seed: u32) -> Vec<u32> {
    (0..n as u32)
        .map(|i| i.wrapping_mul(2654435761).wrapping_add(seed))
        .collect()
}

/// Host-only: freeze → cross-slot fork round-trips the prefix byte-exact
/// (packed shared, mean-K private), the branch starts at fed = frontier
/// with a bumped incarnation, and the source slot is untouched.
#[test]
fn prefix_freeze_fork_roundtrip() {
    let mut ctl =
        KvMemCtl::new_with_strategy(2, &plan(), &geom(), RecallStrategy::Legacy).expect("ctl");
    prefix_fill(&mut ctl, 0, 2, 1);
    let toks = prefix_tokens(256, 1);
    let (id, displaced, deduped) = ctl.freeze_prefix(0, 256, &toks).expect("freeze");
    assert!(displaced.is_empty());
    assert!(!deduped);
    assert_eq!(ctl.prefix_checkpoint(id).expect("ckpt").frontier, 256);

    // Reuse into slot 1 with a longer prompt (suffix to prefill).
    let mut prompt = toks.clone();
    prompt.extend_from_slice(&[7, 8, 9]);
    let frontier = ctl.fork_prefix_into(1, id, &prompt).expect("fork");
    assert_eq!(frontier, 256);
    assert_eq!(ctl.fed_tokens(1), Some(256));
    assert_eq!(ctl.slots[1].turn_start, 256);
    assert_eq!(ctl.slots[1].lw_incarnation, ctl.slots[0].lw_incarnation + 1);

    // Packed payload byte-exact through the shared sources.
    let bt = KV_BLOCK_TOKENS as u32;
    let row_bytes = ctl.raw_cfg.k_gpu_row_bytes as usize;
    for il in 0..ctl.raw_cfg.n_layer {
        let mut src = vec![0u8; bt as usize * row_bytes];
        let mut dst = vec![0u8; bt as usize * row_bytes];
        assert!(ctl.slots[0].raw.copy_k_gpu(1, il, &mut src, bt));
        assert!(ctl.slots[1].raw.copy_k_gpu(1, il, &mut dst, bt));
        assert_eq!(src, dst, "shared packed K block 1 layer {il}");
        assert!(ctl.slots[0].raw.copy_v_gpu(1, il, &mut src, bt));
        assert!(ctl.slots[1].raw.copy_v_gpu(1, il, &mut dst, bt));
        assert_eq!(src, dst, "shared packed V block 1 layer {il}");
    }
    // Mean-K matches at fork, then diverges privately.
    let k_width = ctl.raw_cfg.n_embd_k as usize;
    let mut m0 = vec![0f32; k_width];
    let mut m1 = vec![0f32; k_width];
    ctl.slots[0].raw.mean_k(0, 0, &mut m0);
    ctl.slots[1].raw.mean_k(0, 0, &mut m1);
    assert_eq!(m0, m1, "mean-K equal at fork");
    let overwrite = vec![42f32; bt as usize * k_width];
    ctl.slots[1]
        .raw
        .write_layer_mean_k(0, bt, 0, &overwrite)
        .expect("branch mean");
    ctl.slots[1].raw.mean_k(0, 0, &mut m1);
    assert!(m1.iter().all(|&v| v == 42.0));
    ctl.slots[0].raw.mean_k(0, 0, &mut m0);
    assert!(
        m0.iter().any(|&v| v != 42.0),
        "source mean-K must stay private"
    );
    // Source fed untouched.
    assert_eq!(ctl.fed_tokens(0), Some(256));
}

/// Host-only: freeze validates frontier alignment / fed bound / proof
/// length BEFORE any mutation; fork validates the FULL provenance.
#[test]
fn prefix_freeze_fork_fail_closed() {
    let mut ctl =
        KvMemCtl::new_with_strategy(2, &plan(), &geom(), RecallStrategy::Legacy).expect("ctl");
    prefix_fill(&mut ctl, 0, 2, 2);
    let toks = prefix_tokens(256, 2);
    assert!(
        ctl.freeze_prefix(0, 0, &toks[..0]).is_err(),
        "zero frontier"
    );
    assert!(
        ctl.freeze_prefix(0, 100, &toks[..100]).is_err(),
        "unaligned"
    );
    assert!(ctl.freeze_prefix(0, 384, &toks).is_err(), "frontier > fed");
    assert!(
        ctl.freeze_prefix(0, 128, &toks).is_err(),
        "proof len != frontier"
    );
    assert!(
        ctl.freeze_prefix(99, 128, &toks[..128]).is_err(),
        "bad slot"
    );
    // Nothing cataloged by the failed attempts.
    assert!(ctl.prefix_lookup(&toks).is_none());

    let (id, _, _) = ctl.freeze_prefix(0, 256, &toks).expect("freeze");
    // Bad provenance: flipped token / short prompt / unknown id / bad slot.
    let mut bad = toks.clone();
    bad[100] ^= 1;
    assert!(ctl.fork_prefix_into(1, id, &bad).is_err(), "token proof");
    assert!(
        ctl.fork_prefix_into(1, id, &toks[..128]).is_err(),
        "short prompt"
    );
    assert!(
        ctl.fork_prefix_into(1, id + 100, &toks).is_err(),
        "unknown ckpt"
    );
    assert!(ctl.fork_prefix_into(99, id, &toks).is_err(), "bad slot");
    // The refused forks left slot 1 cold.
    assert_eq!(ctl.fed_tokens(1), Some(0));
}

/// Host-only: an identical frontier+proof dedups (same id, no second
/// archive, deduped flag set so the engine never doubles GPU cache refs).
#[test]
fn prefix_freeze_dedup() {
    let mut ctl =
        KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy).expect("ctl");
    prefix_fill(&mut ctl, 0, 1, 3);
    let toks = prefix_tokens(128, 3);
    let (id1, _, dup1) = ctl.freeze_prefix(0, 128, &toks).expect("freeze 1");
    let (id2, displaced, dup2) = ctl.freeze_prefix(0, 128, &toks).expect("freeze 2");
    assert!(!dup1);
    assert!(dup2);
    assert_eq!(id1, id2);
    assert!(displaced.is_empty());
}

/// Host-only: the bounded catalog evicts LRU entries and hands them back
/// for GPU cache-ref release; lookup picks the longest proven prefix.
#[test]
fn prefix_catalog_eviction_and_lookup() {
    let mut ctl =
        KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy).expect("ctl");
    // Five DISTINCT 128-token prefixes (different proofs) into a bound-4
    // catalog: the fifth freeze displaces the first.
    let mut ids = Vec::new();
    for seed in 0..5u32 {
        ctl.reset_slot(0).expect("reset");
        prefix_fill(&mut ctl, 0, 1, 10 + seed);
        let toks = prefix_tokens(128, 10 + seed);
        let (id, displaced, dup) = ctl.freeze_prefix(0, 128, &toks).expect("freeze");
        assert!(!dup);
        if seed < 4 {
            assert!(displaced.is_empty(), "no eviction before the bound");
        } else {
            assert_eq!(displaced.len(), 1);
            assert_eq!(displaced[0].id, ids[0], "LRU victim is the first entry");
        }
        ids.push(id);
    }
    assert!(ctl.prefix_checkpoint(ids[0]).is_none(), "evicted");
    assert!(ctl.prefix_checkpoint(ids[4]).is_some());
    // Lookup proves the newest prefix; a foreign prompt misses.
    let toks = prefix_tokens(128, 14);
    assert_eq!(ctl.prefix_lookup(&toks), Some(ids[4]));
    assert_eq!(ctl.prefix_lookup(&prefix_tokens(128, 99)), None);
    // Explicit release hands the entry back.
    let released = ctl.release_prefix(ids[4]).expect("release");
    assert_eq!(released.frontier, 128);
    assert!(ctl.prefix_lookup(&toks).is_none());
}

/// Host-only: truncate_slot rewinds the host store to a BLOCK-ALIGNED
/// frontier (whole blocks dropped, retained means exact); a mid-block
/// frontier refuses (sum-only mean-K cannot reconstruct a partial tail).
/// Layerwise refuses (H4).
#[test]
fn prefix_truncate_slot_block_aligned() {
    let mut ctl =
        KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy).expect("ctl");
    prefix_fill(&mut ctl, 0, 2, 5);
    let bt = KV_BLOCK_TOKENS as u32;
    // Mid-block frontier refuses (partial tail irreconstructible).
    assert!(ctl.truncate_slot(0, bt + 64).is_err(), "mid-block refuses");
    assert_eq!(ctl.fed_tokens(0), Some(2 * bt), "refusal leaves fed intact");
    // Block-aligned truncation drops block 1, keeps block 0's mean exact.
    ctl.truncate_slot(0, bt).expect("truncate");
    assert_eq!(ctl.fed_tokens(0), Some(bt));
    let k_width = ctl.raw_cfg.n_embd_k as usize;
    assert_eq!(ctl.slots[0].raw.mean_k_tokens(0, 0), bt);
    let mut mean = vec![0f32; k_width];
    ctl.slots[0].raw.mean_k(0, 0, &mut mean);
    // prefix_fill seed 5, block 0, layer 0: value = 5*100003 + t*k_width + d.
    let expect: f32 = (0..bt as usize)
        .map(|t| (5 * 100003 + t * k_width) as f32)
        .sum::<f32>()
        / bt as f32;
    assert!(
        (mean[0] - expect).abs() < 1e-3,
        "retained mean: {} vs {expect}",
        mean[0]
    );
    // The block repository rewound too.
    assert_eq!(ctl.slots[0].kv.total_tokens(), bt);
    // Idempotent at the same frontier; over-frontier refuses.
    ctl.truncate_slot(0, bt).expect("noop truncate");
    assert!(ctl.truncate_slot(0, bt + 1).is_err(), "frontier > fed");
    // Layerwise refuses truncation (canonical views, H4).
    let mut lw = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("lw ctl");
    prefix_fill(&mut lw, 0, 1, 6);
    assert!(lw.truncate_slot(0, bt).is_err(), "Layerwise refuses");
}
#[test]
fn import_admission_maps_ninfer_window_semantics() {
    // Non-pressure: the exact prefix starts inside the window.
    let ok = layerwise_import_admission(1024, 64, 4096 + 1024, 1024, 1024, 1024).expect("import");
    assert!(!ok.pressure());
    assert_eq!(ok.budget_tokens, 4096);
    assert_eq!(ok.restore_tokens, 1024);
    assert_eq!(
        ok.selection_budget, 0,
        "exact import keeps ordinary selection"
    );

    // Prefill-probe overflow: a small generation reserve puts the prefill
    // boundary below the compact visible extent.
    let probe = layerwise_import_admission(4096, 64, 1024 + 128, 128, 2048, 1024).expect("import");
    assert!(probe.prefill_probe_overflow, "{probe:?}");
    assert!(probe.pressure());
    assert_eq!(probe.budget_tokens, 1024);
    assert_eq!(probe.restore_tokens, 1024, "narrows the imported extent");

    // Terminal reserve overflow also narrows the exact prefix.
    let term = layerwise_import_admission(2048, 512, 1024 + 128, 128, 2048, 1024).expect("import");
    assert!(term.terminal_reserve_overflow, "{term:?}");
    assert_eq!(term.budget_tokens, 512);
    assert_eq!(term.restore_tokens, 512);
    assert!(term.restore_tokens < 2048, "this is the narrowing case");
}

#[test]
fn gen_reserve_is_positive_and_adds_to_the_window() {
    // The constructor rejects zero; capacity is B + R by construction.
    let capacity = 32 * KV_BLOCK_TOKENS as u64 + DEFAULT_GEN_RESERVE;
    assert_eq!(capacity, 4096 + 1024);
    assert!(DEFAULT_GEN_RESERVE > 0);
}

/// 期2.3 (KVMem/nInfer `layerwise_query_at_source`): a proven span equal
/// to the source's recorded span and wholly inside the exact prefix is
/// already effective; everything else stays a new Query.
#[test]
fn query_identity_matches_ninfer() {
    let q = (1000u64, 1100u64);
    assert_eq!(
        super::layerwise_query_identity(Some(q), Some(q), 2048, true),
        None
    );
    assert_eq!(
        super::layerwise_query_identity(Some(q), Some(q), 1024, true),
        Some(q)
    );
    assert_eq!(
        super::layerwise_query_identity(Some(q), Some((0, 10)), 2048, true),
        Some(q)
    );
    assert_eq!(
        super::layerwise_query_identity(Some(q), Some(q), 2048, false),
        Some(q)
    );
    assert_eq!(
        super::layerwise_query_identity(None, Some(q), 2048, true),
        None
    );
}

/// 期2.3: KeepSelected only with a proven, already-effective query that
/// fits; a new query keeps Pressure and reserves the Query envelope.
#[test]
fn import_decision_keep_selected_vs_new_query() {
    let q = (1000u64, 1100u64);
    let keep = super::layerwise_import_decision(
        2048,
        64,
        4096 + 1024,
        1024,
        2048,
        2048,
        Some(q),
        Some(q),
        true,
    )
    .expect("keep decision");
    assert!(keep.keep_selected, "same query inside the prefix must keep");
    assert_eq!(keep.effective_query, None);
    assert!(
        !keep.start.pressure(),
        "keep-selected is not a pressure import"
    );

    let fresh = super::layerwise_import_decision(
        2048,
        64,
        4096 + 1024,
        1024,
        2048,
        2048,
        Some((1200, 1300)),
        Some(q),
        true,
    )
    .expect("new-query decision");
    assert!(!fresh.keep_selected);
    assert_eq!(fresh.effective_query, Some((1200, 1300)));

    let unproven = super::layerwise_import_decision(
        2048,
        64,
        4096 + 1024,
        1024,
        2048,
        2048,
        None,
        Some(q),
        true,
    )
    .expect("unproven decision");
    assert!(
        !unproven.keep_selected,
        "an unproven span never claims keep"
    );
    assert_eq!(unproven.effective_query, None);

    let too_long = super::layerwise_import_decision(
        9000,
        64,
        4096 + 1024,
        1024,
        2048,
        2048,
        Some(q),
        Some(q),
        true,
    )
    .expect("fit decision");
    assert!(!too_long.keep_selected, "an oversized suffix cannot keep");
}

/// 期2.3: the continuation Query gate skips a repeated in-prefix question
/// and arms for a new one; unproven spans keep the historical Query.
#[test]
fn begin_request_query_gate() {
    let mut ctl =
        KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise).expect("ctl");
    prefix_fill(&mut ctl, 0, 1, 0);
    let q = (0u64, 100u64);
    ctl.set_request_query(0, Some(q)).expect("set 1");
    ctl.set_request_query(0, Some(q)).expect("set 2");
    ctl.begin_request(0).expect("begin");
    assert!(
        !ctl.take_layerwise_query_due(0),
        "same in-prefix question keeps"
    );
    ctl.set_request_query(0, Some((120, 220))).expect("set 3");
    ctl.begin_request(0).expect("begin 2");
    assert!(
        ctl.take_layerwise_query_due(0),
        "a new question arms the Query"
    );
    ctl.set_request_query(0, None).expect("set 4");
    ctl.begin_request(0).expect("begin 3");
    assert!(
        ctl.take_layerwise_query_due(0),
        "unproven keeps the historical Query"
    );
}

/// 期3 G3: the content-groups env switches the catalog to the core
/// group-aware publication plan; a third group retires the LRU whole group.
#[test]
fn content_groups_mode_retires_whole_groups() {
    let mut options = test_options();
    options.content_groups = Some(2);
    let mut ctl = KvMemCtl::new_with_options(1, &plan(), &geom(), options, None).expect("ctl");
    for (seed, group) in [(1u32, 11u64), (2, 22), (3, 33)] {
        ctl.reset_slot(0).expect("reset");
        ctl.set_request_group(0, group).expect("group");
        prefix_fill(&mut ctl, 0, 1, seed * 10);
        let toks = prefix_tokens(128, seed * 10);
        ctl.freeze_prefix(0, 128, &toks).expect("freeze");
    }
    let st = ctl.stats();
    assert!(
        st.content_groups >= 1 && st.content_groups <= 2,
        "groups {}",
        st.content_groups
    );
    assert!(
        st.content_retirements >= 1,
        "retirements {}",
        st.content_retirements
    );
}

/// 期3 G3 single-tier: a freeze with a cache dir writes ONE persistent pack
/// (no second account copy), the canonical stays in the session account,
/// and reuse forks it directly (Mean-K preserved).
#[test]
fn single_tier_pack_freeze_and_reuse() {
    let dir = std::env::temp_dir().join(format!("kvmem-cache-test-{}", std::process::id()));
    let options = cache_test_options(&dir, 16_777_216);
    let mut ctl = KvMemCtl::new_with_options(2, &plan(), &geom(), options, None).expect("ctl");
    ctl.set_request_group(0, 42).expect("group");
    prefix_fill(&mut ctl, 0, 1, 5);
    let toks = prefix_tokens(128, 5);
    let (id, displaced, dup) = ctl.freeze_prefix(0, 128, &toks).expect("freeze");
    assert!(!dup && displaced.is_empty());
    let cache_stats = ctl.stats();
    assert!(
        cache_stats.content_packs >= 1 && cache_stats.content_pack_bytes > 0,
        "persistent pack tier written (packs {} bytes {})",
        cache_stats.content_packs,
        cache_stats.content_pack_bytes
    );
    assert_eq!(
        cache_stats.content_cache_bytes + cache_stats.content_cache_disk_bytes,
        0,
        "no second content account"
    );
    let frontier = ctl.fork_prefix_into(1, id, &toks).expect("fork");
    assert_eq!(frontier, 128);
    let width = ctl.raw_cfg.n_embd_k as usize;
    let mut m0 = vec![0f32; width];
    let mut m1 = vec![0f32; width];
    ctl.slots[0].raw.mean_k(0, 0, &mut m0);
    ctl.slots[1].raw.mean_k(0, 0, &mut m1);
    assert_eq!(m0, m1, "reused Mean-K copied exactly");
}

/// 期3 G3 P1.1: a frozen prefix persists as a pack; a fresh controller
/// (restart) scans it, a cold lookup loads it into the catalog, and reuse
/// forks from the loaded canonical.
#[test]
fn content_pack_survives_restart_and_cold_lookup_loads() {
    let dir = std::env::temp_dir().join(format!("kvmem-restart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let options = cache_test_options(&dir, 268_435_456);
    let toks = prefix_tokens(128, 6);
    {
        let mut ctl =
            KvMemCtl::new_with_options(1, &plan(), &geom(), options.clone(), None).expect("ctl");
        ctl.set_request_group(0, 9).expect("group");
        prefix_fill(&mut ctl, 0, 1, 6);
        ctl.freeze_prefix(0, 128, &toks).expect("freeze");
        let st = ctl.stats();
        assert!(
            st.content_packs >= 1,
            "pack written (packs {})",
            st.content_packs
        );
        assert!(st.content_pack_bytes > 0);
    }
    // Restart: a fresh controller scans the pack directory.
    let mut ctl2 = KvMemCtl::new_with_options(2, &plan(), &geom(), options, None).expect("ctl2");
    assert!(ctl2.stats().content_packs >= 1, "pack scanned at startup");
    assert!(ctl2.prefix_catalog.is_empty(), "hot catalog starts empty");
    let hit = ctl2.prefix_lookup(&toks).expect("cold pack hit");
    assert!(
        ctl2.prefix_checkpoint(hit).is_some(),
        "catalog entry loaded"
    );
    assert!(ctl2.stats().cold_pack_hits >= 1);
    let frontier = ctl2.fork_prefix_into(1, hit, &toks).expect("reuse fork");
    assert_eq!(frontier, 128);
    let width = ctl2.raw_cfg.n_embd_k as usize;
    let mut m1 = vec![0f32; width];
    ctl2.slots[1].raw.mean_k(0, 0, &mut m1);
    assert!(
        m1.iter().any(|v| *v != 0.0),
        "reused Mean-K loaded from the pack"
    );
    // A foreign prompt is neither a hot nor a cold proof.
    ctl2.reset_slot(0).expect("reset");
    assert!(ctl2.prefix_lookup(&prefix_tokens(128, 77)).is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

/// 期3 G3 optional micro-opt: a complete pack + recurrent sidecar pair is
/// idempotent across restarts — a re-freeze of the same token proof reuses
/// the bytes instead of rewriting the artifact. A missing sidecar forces a
/// coherent pack rewrite (the engine then exports a fresh DN image).
#[test]
fn content_pack_idempotent_publish_reuses_complete_pair() {
    let dir = std::env::temp_dir().join(format!("kvmem-idempotent-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let options = cache_test_options(&dir, 16_777_216);
    let toks = prefix_tokens(128, 8);
    let (pack, side) = {
        let mut ctl =
            KvMemCtl::new_with_options(1, &plan(), &geom(), options.clone(), None).expect("ctl");
        prefix_fill(&mut ctl, 0, 1, 8);
        let (id, _, _) = ctl.freeze_prefix(0, 128, &toks).expect("freeze");
        let pack = ctl.prefix_pack_path(id).expect("pack path");
        let side = sidecar_path(&pack);
        std::fs::write(&side, vec![7u8; 32]).expect("fake sidecar");
        ctl.note_pack_sidecar(&pack, 32);
        assert!(ctl.prefix_sidecar_ready(id), "pair is complete");
        (pack, side)
    };
    let before = std::fs::metadata(&pack).expect("pack metadata");
    std::thread::sleep(std::time::Duration::from_millis(20));
    // Restart: the scan sees pack + sidecar; the same freeze reuses both.
    let mut ctl2 =
        KvMemCtl::new_with_options(1, &plan(), &geom(), options.clone(), None).expect("ctl2");
    prefix_fill(&mut ctl2, 0, 1, 8);
    let (id2, _, _) = ctl2.freeze_prefix(0, 128, &toks).expect("freeze 2");
    assert!(ctl2.prefix_sidecar_ready(id2), "reused pair stays ready");
    let after = std::fs::metadata(&pack).expect("pack metadata 2");
    assert_eq!(before.len(), after.len());
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(before.ino(), after.ino(), "complete pair is not rewritten");
    }
    assert_eq!(std::fs::read(&side).expect("sidecar read"), vec![7u8; 32]);
    // Dropping the sidecar breaks the pair: the next freeze rewrites the
    // pack and readiness stays false until the engine exports a new image.
    std::fs::remove_file(&side).expect("drop sidecar");
    let mut ctl3 = KvMemCtl::new_with_options(1, &plan(), &geom(), options, None).expect("ctl3");
    prefix_fill(&mut ctl3, 0, 1, 8);
    let (id3, _, _) = ctl3.freeze_prefix(0, 128, &toks).expect("freeze 3");
    assert!(!ctl3.prefix_sidecar_ready(id3), "no sidecar, no ready pair");
    assert!(pack.exists(), "pack rewritten for the next DN export");
    let _ = std::fs::remove_dir_all(&dir);
}

/// G7: media identity rides the pack round trip and gates both hot and
/// cold lookup (text-only requests never reuse an image prefix; a
/// different image digest never does).
#[test]
fn vl_media_pack_roundtrip_and_gated_reuse() {
    let dir = std::env::temp_dir().join(format!("kvmem-vl-media-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let options = cache_test_options(&dir, 268_435_456);
    let toks = prefix_tokens(128, 11);
    {
        let mut ctl =
            KvMemCtl::new_with_options(1, &plan(), &geom(), options.clone(), None).expect("ctl");
        prefix_fill(&mut ctl, 0, 1, 11);
        let (id, _, _) = ctl
            .freeze_prefix_with_media(0, 128, &toks, Some(0xabc), vec![(4, 36)])
            .expect("freeze with media");
        let ckpt = ctl.prefix_checkpoint(id).expect("ckpt");
        assert_eq!(ckpt.media_digest, Some(0xabc));
        assert_eq!(ckpt.media_spans, vec![(4, 36)]);
    }
    let mut ctl2 = KvMemCtl::new_with_options(1, &plan(), &geom(), options, None).expect("ctl2");
    assert_eq!(ctl2.prefix_lookup_media(&toks, None), None);
    assert_eq!(ctl2.prefix_lookup_media(&toks, Some(0xdef)), None);
    let hit = ctl2
        .prefix_lookup_media(&toks, Some(0xabc))
        .expect("same media cold hit");
    let ckpt = ctl2.prefix_checkpoint(hit).expect("entry");
    assert_eq!(ckpt.media_digest, Some(0xabc));
    assert_eq!(ckpt.media_spans, vec![(4, 36)]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 期3 G3 follow-up: captures inherit the previous checkpoint's sealed
/// block origins when the record storage identity matches.
#[test]
fn snapshots_inherit_origins_across_freezes() {
    let mut ctl =
        KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy).expect("ctl");
    prefix_fill(&mut ctl, 0, 2, 8);
    let t1 = prefix_tokens(128, 8);
    let (id1, _, _) = ctl.freeze_prefix(0, 128, &t1).expect("freeze 128");
    let o1_0 = {
        let snap = ctl
            .prefix_checkpoint(id1)
            .and_then(|c| c.kv_snapshot.as_ref())
            .expect("snapshot 1");
        snap.origin_id(0).expect("origin 0")
    };
    let t2 = prefix_tokens(256, 8);
    let (id2, _, _) = ctl.freeze_prefix(0, 256, &t2).expect("freeze 256");
    let snap2 = ctl
        .prefix_checkpoint(id2)
        .and_then(|c| c.kv_snapshot.as_ref())
        .expect("snapshot 2");
    assert_eq!(
        snap2.origin_id(0),
        Some(o1_0),
        "block 0 origin inherited from the previous checkpoint"
    );
    assert_ne!(snap2.origin_id(1), Some(o1_0), "block 1 is a new origin");
    assert!(snap2.origin_id(2).is_some());
}
