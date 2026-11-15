// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The hipfire authors
// hipfire — see LICENSE or NOTICE in the project root.

//! Host-only smoke test for the core checkpoint policy wrappers: the causal
//! capture plan (semantic cuts + periodic timing) and the coverage retention
//! selector (count/bytes + interval spacing, protected roles). Pure metadata:
//! no payload, catalog mutation or device work.

use hipfire_kvmem::{
    retain_checkpoint_coverage, CheckpointCapturePlan, CoverageInfo, CoveragePurpose,
};

fn info(frontier: u64, bytes: u64, identity: u64, purpose: CoveragePurpose) -> CoverageInfo {
    CoverageInfo {
        frontier,
        bytes,
        used: identity,
        identity,
        purpose: purpose as u32,
        protections: 0,
    }
}

#[test]
fn capture_plan_cuts_at_semantic_boundaries_and_schedules_periodic() {
    let mut plan = CheckpointCapturePlan::new(0, 256, 1024, 2048).expect("plan");
    assert_eq!(plan.due(256).expect("due"), Some(CoveragePurpose::Periodic));
    assert_eq!(plan.due(300).expect("due"), Some(CoveragePurpose::Periodic));
    plan.saved(256);
    // The periodic cursor advanced: 300 is now below the next 512 slot.
    assert_eq!(plan.due(300).expect("due"), None);
    assert_eq!(plan.due(512).expect("due"), Some(CoveragePurpose::Periodic));
    // The root cut clamps a chunk that would run past it.
    assert_eq!(plan.cut(512, 2000), 1024);
    assert_eq!(plan.due(1024).expect("due"), Some(CoveragePurpose::Root));
    plan.saved(1024);
    // The tail cut is next, even with the periodic cursor behind it.
    assert_eq!(plan.cut(1024, 3000), 2048);
    assert_eq!(plan.due(2048).expect("due"), Some(CoveragePurpose::Tail));
    plan.saved(2048);
    assert_eq!(plan.cut(2048, 3000), 3000);
    assert_eq!(plan.due(2048).expect("due"), None);
}

#[test]
fn capture_plan_constructor_marks_passed_semantic_points_done() {
    let plan = CheckpointCapturePlan::new(4096, 256, 0, 0).expect("plan");
    // Both semantic points are at/below the base, so only periodic fires.
    assert_eq!(plan.cut(4096, 8000), 8000);
    assert_eq!(plan.due(4096).expect("due"), None, "the periodic cursor starts one stride ahead");
    assert_eq!(plan.due(4352).expect("due"), Some(CoveragePurpose::Periodic));
}

#[test]
fn retention_applies_count_and_keeps_the_newest() {
    let infos: Vec<CoverageInfo> = (1..=6)
        .map(|i| info(i * 100, 10, i as u64, CoveragePurpose::Periodic))
        .collect();
    let keep = retain_checkpoint_coverage(&infos, 1, 10_000, 100, 0).expect("retain");
    assert_eq!(keep, vec![5], "one slot keeps the newest frontier");
}

#[test]
fn retention_spaces_periodic_points_by_min_gap() {
    let infos = vec![
        info(10, 10, 1, CoveragePurpose::Periodic),
        info(20, 10, 2, CoveragePurpose::Periodic),
        info(30, 10, 3, CoveragePurpose::Periodic),
        info(40, 10, 4, CoveragePurpose::Periodic),
    ];
    let keep = retain_checkpoint_coverage(&infos, 10, 10_000, 100, 15).expect("retain");
    assert_eq!(keep, vec![0, 2], "10 and 30 survive a 15-token gap");
}

#[test]
fn retention_skips_absent_identities_and_enforces_the_byte_budget() {
    let infos = vec![
        info(10, 10, 0, CoveragePurpose::Periodic),
        info(20, 10, 2, CoveragePurpose::Periodic),
    ];
    assert_eq!(
        retain_checkpoint_coverage(&infos, 10, 10_000, 100, 0).expect("retain"),
        vec![1],
        "absent slots are never retained"
    );
    let infos = vec![
        info(10, 60, 1, CoveragePurpose::Periodic),
        info(20, 60, 2, CoveragePurpose::Periodic),
    ];
    assert_eq!(
        retain_checkpoint_coverage(&infos, 10, 100, 100, 0).expect("retain"),
        vec![1],
        "120 bytes cannot fit a 100-byte budget; the newest survives"
    );
}

#[test]
fn retention_protects_the_input_and_newest_endpoint_roles() {
    let infos = vec![
        info(10, 10, 1, CoveragePurpose::Input),
        info(20, 10, 2, CoveragePurpose::Periodic),
        info(30, 10, 3, CoveragePurpose::Periodic),
        info(40, 10, 4, CoveragePurpose::Endpoint),
    ];
    // Two slots: the input role and the newest endpoint bypass ordinary
    // cost-based eviction.
    let keep = retain_checkpoint_coverage(&infos, 2, 10_000, 100, 0).expect("retain");
    assert!(keep.contains(&0) && keep.contains(&3), "{keep:?}");
}
