// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;
/// P4.1② G3 host-only: the composite head aux record is trunk-byte-neutral.
/// Legacy carries `mtp_head` UNREAD (trunk-only record, no head plane);
/// Layerwise AR-only likewise; Layerwise + MTP appends exactly one aux ordinal
/// (`n_fa_layers + 1`). Because the core lays each layer out as an independent
/// per-block allocation (raw_kv_store.cpp:154/404), the SAME trunk K/V reads
/// back identically with or without the aux ordinal, and writing the aux plane
/// never perturbs a trunk ordinal.
#[test]
fn head_plane_composite_record_trunk_byte_neutral() {
    let n_fa = geom().n_fa_layers;
    let row = geom().per_pos_bytes;
    let bt = KV_BLOCK_TOKENS;

    // Legacy + MTP: the head plane is a Layerwise-only feature, so Legacy
    // keeps the trunk-only record even with mtp_head set (byte-neutral).
    let legacy_mtp = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Legacy)
        .expect("legacy mtp ctl");
    assert!(
        !legacy_mtp.head_plane_enabled(),
        "Legacy: head plane inactive"
    );
    assert_eq!(
        legacy_mtp.raw_cfg.n_layer, n_fa as u32,
        "Legacy: trunk-only record"
    );

    // Layerwise AR-only: no head plane, trunk-only record.
    let lw_ar = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("layerwise ar ctl");
    assert!(!lw_ar.head_plane_enabled(), "AR-only: head plane inactive");
    assert_eq!(
        lw_ar.raw_cfg.n_layer, n_fa as u32,
        "AR-only: trunk-only record"
    );

    // Layerwise + MTP: composite record, aux ordinal = n_fa.
    let lw_mtp = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("layerwise mtp ctl");
    assert!(
        lw_mtp.head_plane_enabled(),
        "Layerwise+MTP: head plane active"
    );
    assert_eq!(
        lw_mtp.raw_cfg.n_layer,
        (n_fa + 1) as u32,
        "composite record"
    );
    assert_eq!(lw_mtp.aux_ordinal(), n_fa as u32, "aux ordinal = n_fa");

    let pattern = |il: usize, is_v: usize| -> Vec<u8> {
        (0..bt * row)
            .map(|i| ((il * 7 + is_v * 101 + i * 13) & 0xff) as u8)
            .collect()
    };
    // Write the identical trunk K/V to ordinals 0..n_fa-1 and read K back.
    let write_trunk = |ctl: &KvMemCtl| -> Vec<Vec<u8>> {
        let (raw, _kv) = ctl.layerwise_stores(0).expect("slot stores");
        raw.ensure_blocks(2).expect("ensure blocks");
        for il in 0..n_fa {
            raw.write_layer_k_gpu(0, bt as u32, il as u32, &pattern(il, 0))
                .expect("write K");
            raw.write_layer_v_gpu(0, bt as u32, il as u32, &pattern(il, 1))
                .expect("write V");
        }
        (0..n_fa)
            .map(|il| {
                let mut buf = vec![0u8; bt * row];
                assert!(raw.copy_k_gpu(0, il as u32, &mut buf, bt as u32), "read K");
                buf
            })
            .collect()
    };
    assert_eq!(
        write_trunk(&lw_ar),
        write_trunk(&lw_mtp),
        "trunk K bytes identical with/without the aux ordinal"
    );

    // The aux ordinal is a SEPARATE allocation: empty before any head write,
    // and writing it leaves every trunk ordinal byte-identical.
    let (raw, _kv) = lw_mtp.layerwise_stores(0).expect("slot stores");
    let aux = lw_mtp.aux_ordinal();
    assert!(
        !raw.has_k_gpu(0, aux, bt as u32),
        "aux plane empty before head write"
    );
    raw.write_layer_k_gpu(0, bt as u32, aux, &pattern(99, 0))
        .expect("write aux K");
    assert!(
        raw.has_k_gpu(0, aux, bt as u32),
        "aux plane present after write"
    );
    for il in 0..n_fa {
        let mut trunk = vec![0u8; bt * row];
        assert!(raw.copy_k_gpu(0, il as u32, &mut trunk, bt as u32));
        assert_eq!(
            trunk,
            pattern(il, 0),
            "trunk ordinal {il} unchanged by aux write"
        );
    }
}

/// P4.1② G3 regression (live serve HTTP 500): the composite head-aux record has
/// `n_layer = n_fa_layers + 1`, so the trunk retrieval query MUST pass the full
/// record layer count (`n_record_layers`) with the aux ordinal's token count
/// zeroed. Passing the trunk-only `n_fa_layers` trips the core mean-key geometry
/// check (`query.layers == n_layer`) and throws — the exact Err that surfaced as
/// HTTP 500 at the first decode selection under Layerwise+MTP. Host-only: an
/// empty store exercises the geometry gate without reading any mean-K payload.
#[test]
fn head_plane_retrieval_query_geometry() {
    let n_fa = geom().n_fa_layers;

    let lw = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("layerwise mtp ctl");
    assert!(lw.head_plane_enabled());
    // Query buffers span the FULL composite record; the aux ordinal is skipped.
    assert_eq!(lw.n_record_layers, n_fa + 1, "composite record layer count");
    assert_eq!(
        lw.sums.len(),
        (n_fa + 1) * lw.q_width,
        "sums sized for the record"
    );
    assert_eq!(lw.counts.len(), n_fa + 1, "counts sized for the record");
    assert_eq!(
        lw.counts[n_fa], 0,
        "aux ordinal zeroed ⇒ trunk-only scoring"
    );
    for il in 0..n_fa {
        assert_eq!(lw.counts[il], 1, "trunk ordinal {il} active");
    }

    let (raw, _kv) = lw.layerwise_stores(0).expect("slot stores");
    // The query refresh_scores issues: full record layer count. The core
    // geometry check passes (empty store ⇒ no mean-K read, scores all zero).
    let scores = mean_key_retrieval_scores(
        raw,
        2,
        &lw.sums,
        &lw.counts,
        lw.n_record_layers as u32,
        lw.head_dim as u32,
        lw.n_heads as u32,
        lw.n_kv_heads as u32,
        GqaAggregation::Mean,
    )
    .expect("composite-record retrieval query satisfies the core geometry check");
    assert_eq!(scores.len(), 2);

    // The pre-fix trunk-only query (n_fa_layers against n_layer = n_fa+1) is
    // exactly what threw the live 500 — it must still be rejected.
    let bad = mean_key_retrieval_scores(
        raw,
        2,
        &lw.sums[..n_fa * lw.q_width],
        &lw.counts[..n_fa],
        n_fa as u32,
        lw.head_dim as u32,
        lw.n_heads as u32,
        lw.n_kv_heads as u32,
        GqaAggregation::Mean,
    );
    assert!(
        bad.is_err(),
        "trunk-only layer count must trip the geometry check"
    );
    assert!(format!("{}", bad.unwrap_err()).contains("incompatible query/key geometry"));

    // Legacy / AR-only: n_record_layers == n_fa ⇒ the default query is unchanged.
    let ar = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("ar ctl");
    assert_eq!(ar.n_record_layers, n_fa);
    assert_eq!(ar.counts.len(), n_fa);
    assert_eq!(ar.sums.len(), n_fa * ar.q_width);
    let legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert_eq!(
        legacy.n_record_layers, n_fa,
        "Legacy: trunk-only record even with MTP"
    );
}

/// P4.1② G3 host-only: the head-plane lifecycle mirrors the trunk pressure
/// owners. Legacy / AR-only fail closed (no map — byte-neutral); Layerwise +
/// MTP constructs lazily and idempotently; the map take/restore round-trips
/// around the scoped carrier borrow; the per-Query aux binding is bind-once
/// and transient; `reset_slot` drops BOTH so a fresh session never aliases
/// the previous head history.
#[test]
fn head_plane_lifecycle() {
    // Legacy: fail-closed, no map.
    let mut legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert!(legacy.head_plane(0).is_none(), "Legacy: no head map");
    assert!(
        legacy.ensure_head_plane(0).is_err(),
        "Legacy: ensure fails closed"
    );
    assert!(legacy.head_plane(0).is_none(), "Legacy: still no map");

    // Layerwise AR-only: head plane inactive, ensure fails closed.
    let mut ar = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("ar ctl");
    assert!(
        ar.ensure_head_plane(0).is_err(),
        "AR-only: ensure fails closed"
    );

    // Layerwise + MTP: lazy, idempotent construction.
    let mut lw = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("mtp ctl");
    assert!(lw.head_plane(0).is_none(), "before ensure: no map");
    lw.ensure_head_plane(0).expect("construct head map");
    assert!(lw.head_plane(0).is_some(), "after ensure: map present");
    lw.ensure_head_plane(0).expect("idempotent");
    assert_eq!(lw.validate_head_plane(0, 0).expect("empty map"), 0);

    // take/restore round-trip (the scoped carrier borrow resolution).
    let mut head = lw.take_head_plane(0).expect("take map");
    assert!(lw.head_plane(0).is_none(), "taken: slot empty");
    head.auxiliary_pages = vec![0, 1, 2];
    head.auxiliary_view_version = 7;
    lw.restore_head_plane(0, head);
    let restored = lw.head_plane(0).expect("restored map");
    assert_eq!(restored.auxiliary_pages, vec![0, 1, 2]);
    assert_eq!(restored.auxiliary_view_version, 7);
    // 300 tokens over 128-blocks: 128 + 128 + 44 = 300, removed 0.
    assert_eq!(lw.validate_head_plane(0, 300).expect("validate"), 300);

    // aux_selection per-Query transient: bind-once, set/take round-trip.
    assert!(lw.take_aux_selection(0).is_none(), "no binding yet");
    let mut sel = AuxiliarySelection::new(300, KV_BLOCK_TOKENS as u64, true);
    sel.bind(&[0, 1, 2], 300).expect("bind");
    lw.set_aux_selection(0, sel);
    let taken = lw.take_aux_selection(0).expect("take binding");
    assert!(taken.is_bound());
    assert_eq!(taken.selected(), Some(&[0u32, 1, 2][..]));
    assert!(lw.take_aux_selection(0).is_none(), "taken: binding empty");

    // reset_slot drops BOTH the map and the binding (fresh session).
    lw.reset_slot(0).expect("reset");
    assert!(lw.head_plane(0).is_none(), "reset: map dropped");
    assert!(lw.take_aux_selection(0).is_none(), "reset: binding dropped");
    lw.ensure_head_plane(0).expect("re-construct");
    assert_eq!(
        lw.head_plane(0).expect("map").auxiliary_pages,
        Vec::<u32>::new()
    );
}

/// P4.1② G3 host-only: the head resident-map invariant (ninfer
/// `KvmemLayerViews::validate`, NL:12267-12270) reconstructs the accepted
/// frontier from `visible_tokens(pages) + removed_tokens`, and fails closed
/// on a wrong extent or a non-chronological map. The fully-evicted head
/// (empty map, removed == frontier) is admitted.
#[test]
fn head_plane_validate_fail_closed() {
    let mut head = HeadPlaneState {
        auxiliary_pages: vec![0, 1],
        auxiliary_view_version: 0,
        auxiliary_removed_tokens: 44,
    };
    // visible([0,1], 300, 128) = 256; 256 + 44 = 300.
    assert_eq!(head.validate(300).expect("reconstructs"), 300);
    // Wrong removed count → extent != frontier.
    head.auxiliary_removed_tokens = 0;
    assert!(head.validate(300).is_err(), "256 != 300");
    // Non-chronological map → fail closed before the extent check.
    head.auxiliary_pages = vec![1, 0];
    head.auxiliary_removed_tokens = 44;
    assert!(head.validate(300).is_err(), "non-chronological");
    // Fully-evicted head: empty map, removed == frontier.
    let evicted = HeadPlaneState {
        auxiliary_pages: vec![],
        auxiliary_view_version: 0,
        auxiliary_removed_tokens: 300,
    };
    assert_eq!(evicted.validate(300).expect("fully evicted"), 300);
}

/// P4.1② G3b host-only: the head auxiliary-plane write-through seal lands
/// the gathered packed Q8 K/V at the AUX ordinal of the composite record and
/// reads back byte-identical, leaves every trunk ordinal untouched, addresses
/// successive blocks by position, and fails closed off the Layerwise+MTP
/// path. The GPU gather (`head_write_through_archive`) is a contiguous D2H
/// proven by the production smoke; this exercises the host seal + readback
/// round-trip it feeds.
#[test]
fn head_plane_write_through_seal_readback() {
    let n_fa = geom().n_fa_layers;
    let row = geom().per_pos_bytes;
    let vrow = geom().per_pos_v_bytes;
    let bt = KV_BLOCK_TOKENS;

    let mut lw = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("mtp ctl");
    let aux = lw.aux_ordinal();
    assert_eq!(aux, n_fa as u32);

    let k_pat: Vec<u8> = (0..bt * row).map(|i| ((i * 13 + 7) & 0xff) as u8).collect();
    let v_pat: Vec<u8> = (0..bt * vrow)
        .map(|i| ((i * 29 + 101) & 0xff) as u8)
        .collect();
    let trunk_pat = |il: usize, is_v: usize| -> Vec<u8> {
        (0..bt * row)
            .map(|i| ((il * 7 + is_v * 101 + i * 13) & 0xff) as u8)
            .collect()
    };

    // Seed the trunk ordinals, then seal the head aux plane at pos0 = 0.
    {
        let (raw, _kv) = lw.layerwise_stores(0).expect("slot stores");
        raw.ensure_blocks(2).expect("ensure blocks");
        for il in 0..n_fa {
            raw.write_layer_k_gpu(0, bt as u32, il as u32, &trunk_pat(il, 0))
                .expect("trunk K");
            raw.write_layer_v_gpu(0, bt as u32, il as u32, &trunk_pat(il, 1))
                .expect("trunk V");
        }
        assert!(!raw.has_k_gpu(0, aux, bt as u32), "aux empty before seal");
    }
    lw.seal_head_plane(0, 0, bt, &k_pat, &v_pat)
        .expect("seal head plane");

    // Read back the aux plane byte-identical; trunk ordinals unchanged.
    {
        let (raw, _kv) = lw.layerwise_stores(0).expect("slot stores");
        assert!(raw.has_k_gpu(0, aux, bt as u32), "aux present after seal");
        let mut kb = vec![0u8; bt * row];
        let mut vb = vec![0u8; bt * vrow];
        assert!(raw.copy_k_gpu(0, aux, &mut kb, bt as u32), "read aux K");
        assert!(raw.copy_v_gpu(0, aux, &mut vb, bt as u32), "read aux V");
        assert_eq!(kb, k_pat, "aux K byte-identical");
        assert_eq!(vb, v_pat, "aux V byte-identical");
        for il in 0..n_fa {
            let mut tk = vec![0u8; bt * row];
            let mut tv = vec![0u8; bt * row];
            assert!(raw.copy_k_gpu(0, il as u32, &mut tk, bt as u32));
            assert!(raw.copy_v_gpu(0, il as u32, &mut tv, bt as u32));
            assert_eq!(tk, trunk_pat(il, 0), "trunk K {il} unchanged by head seal");
            assert_eq!(tv, trunk_pat(il, 1), "trunk V {il} unchanged by head seal");
        }
    }

    // A second feed at the next block seals at pos0 = bt without disturbing
    // the first block (position addressing).
    let k_pat2: Vec<u8> = (0..bt * row).map(|i| ((i * 17 + 3) & 0xff) as u8).collect();
    let v_pat2: Vec<u8> = (0..bt * vrow)
        .map(|i| ((i * 31 + 5) & 0xff) as u8)
        .collect();
    lw.seal_head_plane(0, bt, bt, &k_pat2, &v_pat2)
        .expect("seal second block");
    {
        let (raw, _kv) = lw.layerwise_stores(0).expect("slot stores");
        let mut kb = vec![0u8; bt * row];
        assert!(
            raw.copy_k_gpu(1, aux, &mut kb, bt as u32),
            "read aux K block 1"
        );
        assert_eq!(kb, k_pat2, "second block byte-identical");
        let mut kb0 = vec![0u8; bt * row];
        assert!(raw.copy_k_gpu(0, aux, &mut kb0, bt as u32));
        assert_eq!(kb0, k_pat, "first block intact");
    }

    // n_rows == 0 is a no-op; out-of-range slot fails closed.
    lw.seal_head_plane(0, 0, 0, &[], &[])
        .expect("empty feed no-op");
    assert!(
        lw.seal_head_plane(9, 0, bt, &k_pat, &v_pat).is_err(),
        "slot out of range"
    );

    // Fail-closed off the Layerwise+MTP path (Legacy / AR-only never co-archive).
    let mut legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert!(
        legacy.seal_head_plane(0, 0, bt, &k_pat, &v_pat).is_err(),
        "Legacy fail-closed"
    );
    let mut ar = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("ar ctl");
    assert!(
        ar.seal_head_plane(0, 0, bt, &k_pat, &v_pat).is_err(),
        "AR-only fail-closed"
    );
}

/// P4.1② G3c host-only: the bounded head-pool geometry is the trunk window +
/// ONE draft-scratch block (① `mtp_auxiliary_capacity` / `mtp_head_pool`),
/// owned independently of the trunk; fail-closed off Layerwise+MTP and on a
/// bad window. The head pool is a 1-layer composite whose slot bytes equal the
/// composite record's aux-plane bytes (① F2 invariant).
#[test]
fn head_pool_bounded_geometry() {
    let bt = KV_BLOCK_TOKENS;
    let row = geom().per_pos_bytes;
    let vrow = geom().per_pos_v_bytes;
    let window = 4 * bt; // page-aligned trunk working-set window

    let lw = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("mtp ctl");
    // capacity = window + one scratch block.
    assert_eq!(
        lw.head_pool_capacity(window).expect("capacity"),
        window + bt
    );
    // pool: window/block + 1 slots, ONE head layer.
    let pool = lw.head_pool(window).expect("head pool");
    assert_eq!(
        pool.n_slots(),
        window / bt + 1,
        "window pages + one scratch page"
    );
    assert_eq!(pool.n_layers(), 1, "head is a single aux layer");
    // The head carrier (engine_carrier_head) sizes the bounded head arena +
    // block-table staging from head_pool_capacity (TOKENS) converted to pages.
    // Lock that conversion against the pool's slot count: a token-vs-page
    // regression would size the head staging to (window+bt) PAGES, giving
    // ctx_len = (window+bt)*bt, and the paged head attention would walk unmapped
    // block-table tiles → GPU hang (the head forward never returns).
    assert_eq!(
        lw.head_pool_capacity(window).expect("capacity") / bt,
        pool.n_slots(),
        "head_pool_capacity tokens / block == head_pool slots (carrier conversion)"
    );
    // slot bytes == the composite record's aux-plane bytes (block × (K+V)).
    assert_eq!(pool.layout().slot_bytes(), bt * (row + vrow));
    assert_eq!(
        pool.layout().arena_bytes(),
        pool.n_slots() * bt * (row + vrow)
    );

    // Fail-closed: non-page-aligned window, zero window.
    assert!(
        lw.head_pool_capacity(window + 1).is_err(),
        "non-page-aligned"
    );
    assert!(lw.head_pool_capacity(0).is_err(), "zero window");
    assert!(lw.head_pool(0).is_err(), "zero window pool");

    // Legacy / AR-only: no head pool (byte-neutral).
    let legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert!(
        legacy.head_pool_capacity(window).is_err(),
        "Legacy fail-closed"
    );
    assert!(legacy.head_pool(window).is_err(), "Legacy pool fail-closed");
    let ar = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("ar ctl");
    assert!(
        ar.head_pool_capacity(window).is_err(),
        "AR-only fail-closed"
    );
}

/// Host-only: the head's INDEPENDENT aux-plane `LayerWorkingSetPlan` (P4.1②
/// G3c, design §A.8 Q1, ninfer NL:10578-10584). Verifies the single-layer aux
/// catalog over the composite descriptor plans the head's own resident map
/// against the window quota: the aux ordinal, the restored/visible block
/// counts, the head's independent view-version bump, the device-token envelope,
/// the resident-map defensive filter, the quota gate, and every fail-closed
/// path (Legacy / AR-only / no map / bad window / no sealed page / cold id).
#[test]
fn head_plane_independent_plan() {
    let bt = KV_BLOCK_TOKENS as u64;
    let window = (4 * bt) as usize; // 512 tokens, page-aligned
    let fed = 3 * bt; // 3 sealed full pages, frontier 384
    let aux = geom_mtp().n_fa_layers as u32; // == 2

    let mut lw = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("mtp ctl");
    // No head map yet ⇒ the caller's detach fails closed (the plan reads a
    // DETACHED map; `take_head_plane` is the existence gate, G3 borrow res).
    assert!(lw.take_head_plane(0).is_none(), "no map");
    lw.ensure_head_plane(0).expect("head map");
    let mut head = lw.take_head_plane(0).expect("detach map");
    // Resident page 0; id 99 is outside the sealed history and must be
    // filtered (not rejected as an invalid resident block).
    head.auxiliary_pages = vec![0, 99];
    head.auxiliary_view_version = 7;
    // FirstLayer-bound target ids [0, 1]: page 1 is a cold restore (resident
    // holds only 0), page 0 stays. The head plans its OWN aux ordinal.
    let head_plan = lw
        .plan_head_plane(0, &head, &[0, 1], fed, window, 0, 4096, "test")
        .expect("head plan");
    assert_eq!(head_plan.model_layer(), aux, "plans the aux ordinal");
    assert_eq!(head_plan.next_block_count(), 2, "two selected pages");
    assert_eq!(
        head_plan.next_visible_tokens(),
        2 * bt,
        "selected page tokens"
    );
    assert_eq!(
        head_plan.restore_count(),
        1,
        "page 1 restored, page 0 resident"
    );
    // device envelope = page_ceil(reserve 0) + 2 full pages = 256 tokens.
    assert_eq!(head_plan.device_tokens_required(), 2 * bt, "device tokens");
    // The head's independent view version bumps (7 → 8), not the trunk's.
    assert_eq!(
        head_plan.next_stamp().expect("stamp").view_version,
        8,
        "aux view bump"
    );

    // Quota gate: a one-page window cannot hold two selected pages.
    assert!(
        lw.plan_head_plane(0, &head, &[0, 1], fed, bt as usize, 0, 4096, "test")
            .is_err(),
        "window quota exceeded"
    );
    // Cold-history id (outside the sealed frame) is rejected by the core.
    assert!(
        lw.plan_head_plane(0, &head, &[5], fed, window, 0, 4096, "test")
            .is_err(),
        "cold id"
    );
    // Bad window / no sealed page.
    assert!(
        lw.plan_head_plane(0, &head, &[0], fed, window + 1, 0, 4096, "test")
            .is_err(),
        "unaligned"
    );
    assert!(
        lw.plan_head_plane(0, &head, &[0], fed, 0, 0, 4096, "test")
            .is_err(),
        "zero window"
    );
    assert!(
        lw.plan_head_plane(0, &head, &[0], bt - 1, window, 0, 4096, "test")
            .is_err(),
        "no page"
    );
    // Return the detached map (session-lifetime persistence).
    lw.restore_head_plane(0, head);

    // Legacy / AR-only: no head plan (byte-neutral); a default detached map
    // is inert, the flag gate fails closed first.
    let legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert!(
        legacy
            .plan_head_plane(
                0,
                &HeadPlaneState::default(),
                &[0],
                fed,
                window,
                0,
                4096,
                "test"
            )
            .is_err(),
        "Legacy"
    );
    let ar = KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
        .expect("ar ctl");
    assert!(
        ar.plan_head_plane(
            0,
            &HeadPlaneState::default(),
            &[0],
            fed,
            window,
            0,
            4096,
            "test"
        )
        .is_err(),
        "AR-only"
    );
}

/// Host-only: the head's post-restore resident-map publish (P4.1② G4, ninfer
/// NL:10609-10612). The restored compact view becomes the head's independent
/// map (pages / view version / removed tokens), the aux invariant validates,
/// the publish is transactional (a failed accept never corrupts the live map),
/// and every fail-closed path holds (Legacy / no map / visible>frontier /
/// non-chronological).
#[test]
fn head_plane_accept_restore() {
    use hipfire_kvmem::ViewBlock;
    let bt = KV_BLOCK_TOKENS as u64;
    let frontier = 3 * bt; // 384
    let session = SessionIdentity {
        id: 1,
        incarnation: 1,
    };
    let mk_view = |ids: &[u64], visible: u64, view_version: u64| ExecutionView {
        stamp: MemoryStamp {
            valid: 1,
            session,
            history_revision: 1,
            view_version,
            frontier,
            execution_history: 1,
        },
        blocks: ids
            .iter()
            .enumerate()
            .map(|(i, &id)| ViewBlock {
                block: BlockDescriptor {
                    session,
                    id,
                    original_start: id * bt,
                    valid_tokens: KV_BLOCK_TOKENS as u32,
                    content_version: 1,
                    statistics_version: 1,
                },
                compact_start: i as u64 * bt,
            })
            .collect(),
        visible_tokens: visible,
    };

    let mut lw = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("mtp ctl");
    let view = mk_view(&[0, 1], 2 * bt, 8);
    // No map yet ⇒ the caller's detach fails closed (no mutation).
    assert!(lw.take_head_plane(0).is_none(), "no map");
    lw.ensure_head_plane(0).expect("head map");
    let mut head = lw.take_head_plane(0).expect("detach map");
    // Publish: pages [0,1], view version 8, removed = 384 - 256 = 128.
    lw.accept_head_restore(0, &mut head, &view, frontier)
        .expect("accept");
    assert_eq!(head.auxiliary_pages, vec![0, 1], "restored pages");
    assert_eq!(head.auxiliary_view_version, 8, "view version");
    assert_eq!(
        head.auxiliary_removed_tokens,
        (frontier - 2 * bt) as u32,
        "RoPE shift"
    );
    // Return the map so the on-ctl validate path sees the published extent.
    lw.restore_head_plane(0, head);
    assert_eq!(
        lw.validate_head_plane(0, frontier).expect("validate"),
        frontier
    );

    // Detach again for the fail-closed accepts (transactional on the map).
    let mut head = lw.take_head_plane(0).expect("detach map");
    // visible > frontier ⇒ fail-closed (removed would underflow).
    let over = mk_view(&[0, 1], frontier + bt, 9);
    assert!(
        lw.accept_head_restore(0, &mut head, &over, frontier)
            .is_err(),
        "visible>frontier"
    );
    // Non-chronological restored map ⇒ validate fails closed.
    let bad = mk_view(&[1, 0], 2 * bt, 9);
    assert!(
        lw.accept_head_restore(0, &mut head, &bad, frontier)
            .is_err(),
        "non-chronological"
    );
    // Transactional: the failed accepts left the published map untouched.
    assert_eq!(head.auxiliary_view_version, 8, "atomic publish");
    lw.restore_head_plane(0, head);

    // Legacy: no head plane ⇒ fail-closed (byte-neutral); a default detached
    // map is inert, the flag gate fails closed first.
    let legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert!(
        legacy
            .accept_head_restore(0, &mut HeadPlaneState::default(), &view, frontier)
            .is_err(),
        "Legacy"
    );
}

#[test]
fn head_plane_accept_append() {
    let bt = KV_BLOCK_TOKENS as u64;
    let mut lw = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("mtp ctl");
    lw.ensure_head_plane(0).expect("head map");

    // Cold prefill append from an empty map: a page-aligned begin needs no
    // partial anchor; pages [0,1,2] reconstruct 384 with removed == 0 and the
    // view version stays 0 (an append is not a restore).
    lw.accept_head_append(0, 0, 3 * bt).expect("prefill append");
    {
        let head = lw.head_plane(0).expect("map");
        assert_eq!(head.auxiliary_pages, vec![0, 1, 2], "prefill pages");
        assert_eq!(head.auxiliary_removed_tokens, 0, "full-resident removed");
        assert_eq!(
            head.auxiliary_view_version, 0,
            "append does not bump view version"
        );
    }
    assert_eq!(lw.validate_head_plane(0, 3 * bt).expect("validate"), 3 * bt);

    // §3.9g sub-page append: the 16-token partial tail (384..400) rides the
    // suffix stripe, NOT a canonical block, so no new page joins and the
    // canonical frame stays page-rounded 384; removed stays 0.
    lw.accept_head_append(0, 3 * bt, 3 * bt + 16)
        .expect("sub-page append");
    assert_eq!(lw.head_plane(0).unwrap().auxiliary_pages, vec![0, 1, 2]);
    assert_eq!(lw.head_plane(0).unwrap().auxiliary_removed_tokens, 0);
    assert_eq!(lw.validate_head_plane(0, 3 * bt).expect("validate"), 3 * bt);

    // Filling the suffix page to the next boundary seals it canonical: page 3
    // (the old suffix page) becomes a full page and joins; frame advances 512.
    lw.accept_head_append(0, 3 * bt + 16, 4 * bt)
        .expect("seal suffix append");
    assert_eq!(lw.head_plane(0).unwrap().auxiliary_pages, vec![0, 1, 2, 3]);
    assert_eq!(lw.head_plane(0).unwrap().auxiliary_removed_tokens, 0);
    assert_eq!(lw.validate_head_plane(0, 4 * bt).expect("validate"), 4 * bt);

    // Fail-closed: end < begin is rejected and leaves the map empty
    // (transactional). The §3.9g suffix stripe is the compact-shift anchor, so
    // a sub-page begin needs no resident partial page (ninfer's NL:12343
    // anchor guard does not apply to the suffix representation).
    let mut lw2 = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("mtp ctl 2");
    lw2.ensure_head_plane(0).expect("head map");
    assert!(lw2.accept_head_append(0, 200, 100).is_err(), "end < begin");
    assert!(
        lw2.head_plane(0).unwrap().auxiliary_pages.is_empty(),
        "atomic"
    );
    // A sub-page begin on a fresh map seals the first full page: frame 0->128.
    lw2.accept_head_append(0, 100, 200)
        .expect("sub-page begin append");
    assert_eq!(lw2.head_plane(0).unwrap().auxiliary_pages, vec![0]);

    // Subset resident map (a post-restore compacted view with a gap): the
    // append recomputes removed = end - visible, preserving the aux invariant
    // and leaving the view version untouched.
    let mut lw3 = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Layerwise)
        .expect("mtp ctl 3");
    lw3.ensure_head_plane(0).expect("head map");
    {
        let head = lw3.head_plane_mut(0).expect("map");
        head.auxiliary_pages = vec![0, 1];
        head.auxiliary_removed_tokens = bt as u32; // frontier 384, visible 256
        head.auxiliary_view_version = 8;
    }
    lw3.accept_head_append(0, 3 * bt, 4 * bt)
        .expect("subset append");
    {
        let head = lw3.head_plane(0).expect("map");
        assert_eq!(head.auxiliary_pages, vec![0, 1, 3], "subset + new page");
        assert_eq!(
            head.auxiliary_removed_tokens, bt as u32,
            "removed recomputed"
        );
        assert_eq!(head.auxiliary_view_version, 8, "view version unchanged");
    }
    assert_eq!(
        lw3.validate_head_plane(0, 4 * bt).expect("validate"),
        4 * bt
    );

    // Legacy: no head plane ⇒ fail-closed (byte-neutral).
    let mut legacy = KvMemCtl::new_with_strategy(1, &plan(), &geom_mtp(), RecallStrategy::Legacy)
        .expect("legacy ctl");
    assert!(legacy.accept_head_append(0, 0, bt).is_err(), "Legacy");
}
