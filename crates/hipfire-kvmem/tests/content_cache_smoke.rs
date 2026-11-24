// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The hipfire authors
// hipfire — see LICENSE or NOTICE in the project root.

//! Host-only smoke test for the content-cache policy wrappers: cache-owner
//! resolution, candidate ranking, snapshot coverage, directory publication /
//! eviction and group-aware retention. Pure metadata; no payload or directory
//! mutation happens here.

use hipfire_kvmem::{
    content_snapshot_covers, plan_content_group_publication, rank_content_candidates,
    resolve_cache_owner, select_content_eviction, select_content_publication, ContentCandidate,
    ContentSlot, PublicationReason,
};

fn cand(slot: u64, frontier: u64, cold: u64, used: u64, cont: bool) -> ContentCandidate {
    ContentCandidate {
        slot,
        frontier,
        cold_bytes: cold,
        last_used: used,
        continuation: cont as i32,
    }
}

fn mk(occupied: bool, pinned: bool, used: u64, identity: u64, group: u64) -> ContentSlot {
    ContentSlot {
        occupied: occupied as i32,
        pinned: pinned as i32,
        last_used: used,
        identity,
        group,
    }
}

#[test]
fn ranking_filters_orders_and_caps() {
    let c = vec![
        cand(0, 100, 10, 5, false),
        cand(1, 200, 20, 9, false),
        cand(2, 0, 10, 9, true), // zero frontier is not a candidate
        cand(3, 150, 10, 7, true),
    ];
    let all = rank_content_candidates(&c, true, 16).expect("rank");
    let ids: Vec<u64> = all.iter().map(|v| v.slot).collect();
    assert_eq!(ids, vec![1, 3, 0]);
    // Without a new user query only proven continuations survive.
    let cont = rank_content_candidates(&c, false, 16).expect("rank");
    assert_eq!(cont.iter().map(|v| v.slot).collect::<Vec<_>>(), vec![3]);
    // The cap applies after ordering.
    let capped = rank_content_candidates(&c, true, 2).expect("rank");
    assert_eq!(
        capped.iter().map(|v| v.slot).collect::<Vec<_>>(),
        vec![1, 3]
    );
}

#[test]
fn snapshot_coverage_requires_every_named_origin() {
    assert!(content_snapshot_covers(&[1, 2, 3], &[1, 3]));
    assert!(!content_snapshot_covers(&[1, 2], &[1, 3]));
    assert!(!content_snapshot_covers(&[0, 1], &[1]), "unknown origin");
    assert!(
        !content_snapshot_covers(&[1], &[]),
        "empty coverage is no proof"
    );
}

#[test]
fn publication_inserts_dedupes_and_replaces() {
    let (slot, reason) = select_content_publication(&[], 4, None).expect("publication");
    assert_eq!((slot, reason), (Some(0), PublicationReason::Inserted));
    let slots = vec![mk(true, false, 1, 7, 0), mk(true, false, 2, 8, 0)];
    let (slot, reason) = select_content_publication(&slots, 4, Some(7)).expect("publication");
    assert_eq!((slot, reason), (Some(0), PublicationReason::Deduplicated));
    let (slot, reason) = select_content_publication(&slots, 2, None).expect("publication");
    assert_eq!(
        (slot, reason),
        (Some(0), PublicationReason::Replaced),
        "LRU unpinned victim"
    );
    let pinned = vec![mk(true, true, 1, 7, 0), mk(true, true, 2, 8, 0)];
    let (slot, reason) = select_content_publication(&pinned, 2, None).expect("publication");
    assert_eq!((slot, reason), (None, PublicationReason::AllPinned));
    let (slot, reason) = select_content_publication(&slots, 0, None).expect("publication");
    assert_eq!((slot, reason), (None, PublicationReason::Disabled));
}

#[test]
fn eviction_prefers_redundant_group_versions() {
    // Group 1 has two retained versions; group 2 one. The stale redundant
    // version goes first even though group 2 is older.
    let slots = vec![
        mk(true, false, 3, 10, 1),
        mk(true, false, 9, 11, 1),
        mk(true, false, 1, 12, 2),
    ];
    assert_eq!(select_content_eviction(&slots).expect("eviction"), Some(0));
    let pinned = vec![mk(true, true, 1, 10, 1), mk(true, true, 2, 11, 1)];
    assert_eq!(select_content_eviction(&pinned).expect("eviction"), None);
}

#[test]
fn group_publication_retires_whole_groups() {
    let slots = vec![
        mk(true, false, 1, 10, 1),
        mk(true, false, 5, 11, 1),
        mk(true, false, 9, 12, 2),
    ];
    let (slot, reason, retired) =
        plan_content_group_publication(&slots, 8, 2, 4).expect("group publication");
    assert_eq!(retired, vec![0, 1], "the oldest whole group retires");
    assert_eq!(slot, Some(0), "the first released slot is reused");
    assert_eq!(reason, PublicationReason::Replaced);
    // A pinned sole group cannot be retired for the new group.
    let pinned = vec![mk(true, true, 1, 10, 1), mk(true, true, 5, 11, 1)];
    let (slot, reason, retired) =
        plan_content_group_publication(&pinned, 8, 1, 2).expect("group publication");
    assert_eq!(
        (slot, reason, retired),
        (None, PublicationReason::AllPinned, vec![])
    );
    assert_eq!(
        plan_content_group_publication(&slots, 8, 2, 0)
            .expect("group publication")
            .1,
        PublicationReason::Incomplete,
        "an unknown incoming group never publishes"
    );
}

#[test]
fn cache_owner_resolution_matches_the_core_precedence() {
    assert_eq!(
        resolve_cache_owner(Some("conv"), Some("p"), None, None).expect("owner"),
        Some("conv".into())
    );
    assert_eq!(
        resolve_cache_owner(None, Some("p"), Some("p"), None).expect("owner"),
        Some("p".into())
    );
    assert!(resolve_cache_owner(None, Some("p"), Some("other"), None).is_err());
    assert_eq!(
        resolve_cache_owner(None, Some(""), None, Some("legacy")).expect("owner"),
        Some("legacy".into())
    );
    assert_eq!(
        resolve_cache_owner(None, None, None, None).expect("owner"),
        None
    );
    assert!(resolve_cache_owner(Some(""), None, None, None).is_err());
    let long = "x".repeat(300);
    assert_eq!(
        resolve_cache_owner(None, None, None, Some(&long)).expect("owner"),
        None
    );
}
