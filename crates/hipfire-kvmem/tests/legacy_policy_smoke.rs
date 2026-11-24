// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The hipfire authors
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only legacy policy/FFI regressions. No model or GPU is required.

use hipfire_kvmem::{
    bounded_query, pressure_selection, query_prefix_required, query_replay_plan, KvStore,
    KvStoreConfig, QueryRange, RetrievalCheckpoint,
};

fn store(lrfu: bool) -> KvStore {
    let kv = KvStore::new(&KvStoreConfig {
        block_tokens: 4,
        select_budget: 12,
        sink_blocks: 1,
        recent_blocks: 1,
        lrfu: u32::from(lrfu),
        lrfu_lambda: 0.5,
        lrfu_top_fraction: 1.0,
        ..KvStoreConfig::default()
    })
    .expect("store");
    kv.register_append(24).expect("six blocks");
    kv
}

fn assert_checkpoint(snapshot: &RetrievalCheckpoint, round: u64, committed: bool, scores: &[f64]) {
    assert_eq!(snapshot.round().unwrap(), round);
    assert_eq!(snapshot.committed().unwrap(), committed);
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&snapshot.scores().unwrap()), bits(scores));
}

#[test]
fn committed_history_restores_exactly_and_repeatedly_without_consuming_the_source() {
    let kv = store(true);
    assert_checkpoint(&kv.retrieval_checkpoint().unwrap(), 0, false, &[]);
    let first_scores = [0.0, 0.25, 1.0, 0.5, -0.25, 0.0];
    kv.begin_retrieval_round().unwrap();
    kv.set_retrieval_scores(&first_scores).unwrap();
    assert_checkpoint(&kv.retrieval_checkpoint().unwrap(), 1, false, &[]);
    assert_eq!(kv.pick_topk_blocks(&[]).unwrap(), [0, 2, 5]);
    assert!(kv.commit_retrieval_round().unwrap());
    let first = kv.retrieval_checkpoint().unwrap();
    let cloned = first.try_clone().unwrap();
    assert_checkpoint(&first, 1, true, &first_scores);

    for _ in 0..2 {
        kv.begin_retrieval_round().unwrap();
        let open = kv.retrieval_checkpoint().unwrap();
        assert_checkpoint(&open, 2, false, &first_scores);
        kv.set_retrieval_scores(&[0.0, 0.5, 0.0, 2.0, -0.5, 0.0])
            .unwrap();
        let second_scores = [0.0, 0.625, 0.5, 2.25, -0.625, 0.0];
        for (block, expected) in second_scores.iter().enumerate() {
            assert_eq!(kv.selection_score(block as u32).unwrap(), *expected);
        }
        assert_eq!(kv.pick_topk_blocks(&[]).unwrap(), [0, 3, 5]);
        assert!(kv.commit_retrieval_round().unwrap());
        let second = kv.retrieval_checkpoint().unwrap();
        assert_checkpoint(&second, 2, true, &second_scores);

        // Restore also discards an uncommitted probe instead of retaining it.
        kv.begin_retrieval_round().unwrap();
        kv.set_retrieval_scores(&[99.0; 6]).unwrap();
        kv.restore_retrieval(&first).unwrap();
        assert_eq!(kv.pick_topk_blocks(&[]).unwrap(), [0, 2, 5]);
        assert!(
            !kv.commit_retrieval_round().unwrap(),
            "no duplicate reinforcement"
        );
        for (block, expected) in first_scores.iter().enumerate() {
            assert_eq!(kv.selection_score(block as u32).unwrap(), *expected);
        }
        assert_checkpoint(&kv.retrieval_checkpoint().unwrap(), 1, true, &first_scores);
        assert_checkpoint(&first, 1, true, &first_scores);
        assert_checkpoint(&cloned, 1, true, &first_scores);
        assert_checkpoint(&second, 2, true, &second_scores);
    }
    drop(first);
    drop(kv);
    let other = store(true);
    other.restore_retrieval(&cloned).unwrap();
    assert_checkpoint(&cloned, 1, true, &first_scores);
    assert_eq!(other.selection_score(2).unwrap(), 1.0);
}

#[test]
fn open_round_snapshot_restores_committed_scores_but_not_the_preview() {
    let kv = store(true);
    kv.begin_retrieval_round().unwrap();
    kv.set_retrieval_scores(&[2.0; 6]).unwrap();
    assert!(kv.commit_retrieval_round().unwrap());
    kv.begin_retrieval_round().unwrap();
    kv.set_retrieval_scores(&[4.0; 6]).unwrap();
    let open = kv.retrieval_checkpoint().unwrap();
    assert_eq!(kv.selection_score(0).unwrap(), 5.0);
    assert_checkpoint(&open, 2, false, &[2.0; 6]);
    kv.restore_retrieval(&open).unwrap();
    assert_eq!(kv.selection_score(0).unwrap(), 2.0);
    assert!(
        !kv.commit_retrieval_round().unwrap(),
        "restored preview is absent"
    );
    assert_checkpoint(&open, 2, false, &[2.0; 6]);
}

#[test]
fn score_bounds_do_not_truncate_a_longer_logical_checkpoint() {
    let kv = store(true);
    kv.begin_retrieval_round().unwrap();
    kv.set_retrieval_scores(&[1.0; 6]).unwrap();
    kv.commit_retrieval_round().unwrap();
    let snapshot = kv.retrieval_checkpoint().unwrap();
    kv.truncate_to(8).unwrap();
    kv.restore_retrieval(&snapshot).unwrap();
    assert_eq!(kv.selection_score(1).unwrap(), 1.0);
    assert!(kv.selection_score(2).is_err());
    assert!(kv.selection_score(u32::MAX).is_err());
    assert_checkpoint(&snapshot, 1, true, &[1.0; 6]);
    kv.register_append(16).unwrap();
    assert_eq!(kv.selection_score(5).unwrap(), 1.0);

    let plain = store(false);
    plain.set_retrieval_scores(&[3.0; 6]).unwrap();
    assert_eq!(plain.selection_score(5).unwrap(), 3.0);
    assert!(plain.selection_score(6).is_err());
    assert_checkpoint(&plain.retrieval_checkpoint().unwrap(), 0, false, &[]);
}

#[test]
fn pressure_never_recalls_high_scoring_cold_history_and_protects_mandatory() {
    let kv = store(false);
    kv.set_retrieval_scores(&[0.4, 0.2, 1e9, 0.3, 1e8, 0.1])
        .unwrap();
    let resident = [0, 1, 3, 5]; // 2 and 4 are cold, however high their score.
    let scores: Vec<_> = resident
        .iter()
        .map(|&id| kv.selection_score(id).unwrap())
        .collect();
    assert_eq!(
        pressure_selection(&resident, &scores, &[5], 3).unwrap(),
        [0, 3, 5]
    );
    assert!(pressure_selection(&resident, &scores, &[2], 3).is_err());
    assert!(
        pressure_selection(&resident, &scores, &[6], 3).is_err(),
        "future mandatory"
    );
    assert_eq!(
        pressure_selection(&resident, &scores, &[5, 5], 1).unwrap(),
        [5]
    );
}

#[test]
fn pressure_rejects_inconsistent_frames_and_preserves_core_tie_order() {
    assert!(pressure_selection(&[], &[], &[], 0).is_err());
    assert!(pressure_selection(&[1], &[1.0], &[], 0).is_err());
    assert!(pressure_selection(&[1, 2], &[1.0, 2.0], &[1, 2], 1).is_err());
    assert!(pressure_selection(&[1, 1], &[1.0, 2.0], &[], 2).is_err());
    assert!(pressure_selection(&[1, 2], &[1.0], &[], 2).is_err());
    assert!(pressure_selection(&[1], &[1.0, 2.0], &[], 2).is_err());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
        assert!(pressure_selection(&[1], &[bad], &[], 1).is_err());
    }
    assert_eq!(
        pressure_selection(&[], &[], &[], 1).unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        pressure_selection(&[1, 8, 3, 6], &[1.0; 4], &[], 2).unwrap(),
        [6, 8]
    );
    assert_eq!(
        pressure_selection(&[6, 3, 8, 1], &[1.0; 4], &[], 2).unwrap(),
        [6, 8]
    );
    assert_eq!(
        pressure_selection(&[8, 1], &[1.0, 2.0], &[], 9).unwrap(),
        [1, 8]
    );
}

#[test]
fn bounded_query_preserves_zero_and_clamps_without_overflow() {
    for count in [511, 512, 513] {
        let query = bounded_query(100, count, 100 + count, 512).unwrap();
        assert_eq!(query.end, 100 + count);
        assert_eq!(query.end - query.begin, count.min(512));
    }
    assert_eq!(
        bounded_query(100, 20, 120, 0).unwrap(),
        QueryRange {
            begin: 120,
            end: 120
        }
    );
    assert_eq!(
        bounded_query(200, u64::MAX, 120, 512).unwrap(),
        QueryRange {
            begin: 120,
            end: 120
        }
    );
    assert_eq!(
        bounded_query(u64::MAX - 2, u64::MAX, u64::MAX, 1).unwrap(),
        QueryRange {
            begin: u64::MAX - 1,
            end: u64::MAX
        }
    );
}

#[test]
fn replay_uses_full_span_and_reports_core_refusal_reasons() {
    let query = QueryRange {
        begin: 100,
        end: 120,
    };
    for span in [2047, 2048, 2049] {
        let plan = query_replay_plan(query, 64, 64 + span, 2048, 64, 1, 100, 0).unwrap();
        assert_eq!((plan.begin, plan.end), (64, 64 + span));
        assert_eq!(plan.admitted, span <= 2048);
        assert_eq!(
            plan.reason,
            if span <= 2048 {
                "bounded_replay"
            } else {
                "replay_token_limit"
            }
        );
    }
    let zero = query_replay_plan(query, 100, 140, 0, 64, 2, 100, 0).unwrap();
    assert!(!zero.admitted);
    assert_eq!(zero.reason, "replay_token_limit");
    let empty = query_replay_plan(
        QueryRange {
            begin: 120,
            end: 120,
        },
        100,
        140,
        0,
        0,
        2,
        0,
        0,
    )
    .unwrap();
    assert!(!empty.admitted);
    assert_eq!(empty.reason, "no_query");
    for (restore, end, block) in [(101, 140, 64), (100, 119, 64), (100, 99, 64), (100, 140, 0)] {
        let plan = query_replay_plan(query, restore, end, 2048, block, 2, 100, 0).unwrap();
        assert!(!plan.admitted);
        assert_eq!(plan.reason, "invalid_replay_boundary");
    }
    let overflow = query_replay_plan(
        QueryRange {
            begin: u64::MAX - 1,
            end: u64::MAX,
        },
        u64::MAX - 1,
        u64::MAX,
        2048,
        64,
        0,
        u64::MAX,
        1,
    )
    .unwrap();
    assert!(!overflow.admitted);
    assert_eq!(overflow.reason, "invalid_replay_boundary");
}

#[test]
fn replay_capacity_counts_partial_overlap_and_workspace_once() {
    let query = QueryRange {
        begin: 100,
        end: 120,
    };
    assert!(
        query_replay_plan(query, 100, 140, 2048, 64, 2, 3, 4)
            .unwrap()
            .admitted
    );
    for (history, device, workspace) in [(2, 2, 4), (2, 3, 53), (4, 3, 0)] {
        let plan =
            query_replay_plan(query, 100, 140, 2048, 64, history, device, workspace).unwrap();
        assert!(!plan.admitted);
        assert_eq!(plan.reason, "replay_device_capacity");
    }
    assert!(
        query_replay_plan(query, 100, 140, 2048, 64, 2, 3, 52)
            .unwrap()
            .admitted
    );
}

#[test]
fn query_prefix_deduplicates_media_preserves_partial_tail_and_rejects_overflow() {
    assert_eq!(
        query_prefix_required(321, 64, 1, 1, &[2, 20, 2]).unwrap(),
        [0, 2, 5]
    );
    assert_eq!(query_prefix_required(321, 64, 1, 0, &[]).unwrap(), [0, 5]);
    assert_eq!(
        query_prefix_required(320, 64, 0, 0, &[]).unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        query_prefix_required(128, 64, 2, 2, &[0, 1, 9]).unwrap(),
        [0, 1]
    );
    assert!(query_prefix_required(0, 64, 9, 9, &[0]).unwrap().is_empty());
    assert!(query_prefix_required(0, 0, 0, 0, &[]).is_err());
    assert!(query_prefix_required(u64::MAX, 1, 0, 0, &[]).is_err());
    assert!(query_prefix_required(u64::from(u32::MAX) + 1, 1, 0, 0, &[]).is_err());
    assert!(query_prefix_required(u64::from(u32::MAX), 1, 0, 0, &[])
        .unwrap()
        .is_empty());
}
