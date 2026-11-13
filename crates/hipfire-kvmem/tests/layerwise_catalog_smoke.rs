// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only smoke test for the layerwise catalog + planning foundation: no GPU,
//! no model. Mirrors kvmem's LayerwiseMemoryCatalog / LayerWorkingSetPlan over the
//! hipfire Q8 {Key,Value} layout adapter: build a two-layer canonical catalog,
//! plan a per-layer resident swap (restore/evict, drop-evict with a sealed host
//! archive so no spill), read the reduced plan + per-layer transfer regions, and
//! exercise the bulk plan and the fail-closed admission rejections.

use hipfire_kvmem::{
    BackendIdentity, BlockCopies, BlockDescriptor, LayerMemorySnapshot, LayerSelection,
    MemoryBudget, MemoryCapabilities, MemoryDescriptor, MemorySnapshot, MemoryStamp, PayloadLayout,
    RawStoreConfig, SessionIdentity,
};

const BLOCK: u32 = 4;
const ROW_BYTES: usize = 36;
const LAYERS: u32 = 2;
const RECORD_BYTES: u64 = (ROW_BYTES * BLOCK as usize * LAYERS as usize * 2) as u64; // 576
const PLANE_BYTES: u64 = (ROW_BYTES * BLOCK as usize) as u64; // 144
const LAYER_BYTES: u64 = PLANE_BYTES * 2; // 288 (K+V for one layer)

fn raw_config() -> RawStoreConfig {
    RawStoreConfig {
        n_layer: LAYERS,
        n_embd_k: 8,
        n_embd_v: 8,
        block_tokens: BLOCK,
        k_row_bytes: 0,
        k_gpu_row_bytes: ROW_BYTES as u64,
        v_gpu_row_bytes: ROW_BYTES as u64,
    }
}

fn backend() -> BackendIdentity {
    BackendIdentity {
        backend: "hipfire".into(),
        revision: "kvmem-v046".into(),
        model: "two-layer-fixture".into(),
        state_schema: "state-v1".into(),
    }
}

fn session() -> SessionIdentity {
    SessionIdentity { id: 42, incarnation: 3 }
}

fn descriptor() -> MemoryDescriptor {
    let layout =
        PayloadLayout::from_raw(&raw_config(), "hipfire-q8-packed-v1", &[0, 1]).expect("layout");
    let cap = MemoryCapabilities {
        page_tokens: BLOCK,
        max_position: 1 << 20,
        native_payload: 1,
        sparse_view: 1,
        original_rope_positions: 1,
        pre_rope_statistics: 1,
        state_checkpoint: 1,
        asynchronous_transfer: 0,
    };
    MemoryDescriptor::new(&backend(), &layout, &cap).expect("descriptor")
}

/// Accepted history stamp; only `view_version` differs between target layers.
fn stamp(view_version: u64) -> MemoryStamp {
    MemoryStamp {
        valid: 1,
        session: session(),
        history_revision: 5,
        view_version,
        frontier: 8,
        execution_history: 9,
    }
}

fn block(id: u64, start: u64, content_version: u64) -> BlockDescriptor {
    BlockDescriptor {
        session: session(),
        id,
        original_start: start,
        valid_tokens: BLOCK,
        content_version,
        statistics_version: 2,
    }
}

/// Two full blocks (100@0, 200@4), both with completed current host copies.
fn snapshot(view_version: u64, resident: Vec<u64>) -> MemorySnapshot {
    MemorySnapshot {
        stamp: stamp(view_version),
        blocks: vec![
            BlockCopies { block: block(100, 0, 1), host_version: 1 },
            BlockCopies { block: block(200, 4, 2), host_version: 2 },
        ],
        resident,
    }
}

fn layers() -> Vec<LayerMemorySnapshot> {
    vec![
        LayerMemorySnapshot { model_layer: 0, memory: snapshot(6, vec![100]) },
        LayerMemorySnapshot { model_layer: 1, memory: snapshot(11, vec![200]) },
    ]
}

fn budget() -> MemoryBudget {
    MemoryBudget {
        selected_tokens: BLOCK as u64,
        reserve_tokens: 0,
        device_tokens: BLOCK as u64,
        host_payload_bytes: RECORD_BYTES * 4,
    }
}

/// Swap layer 0's resident set from [100] to [200].
fn sel0() -> LayerSelection {
    LayerSelection {
        model_layer: 0,
        expected: stamp(6),
        selected: vec![200],
        mandatory: vec![],
        budget: budget(),
    }
}

/// Swap layer 1's resident set from [200] to [100].
fn sel1() -> LayerSelection {
    LayerSelection {
        model_layer: 1,
        expected: stamp(11),
        selected: vec![100],
        mandatory: vec![],
        budget: budget(),
    }
}

#[test]
fn catalog_snapshots_and_per_layer_plan() {
    let desc = descriptor();
    let catalog = hipfire_kvmem::MemoryCatalog::new(&desc, &layers()).expect("catalog");

    // Catalog stored both target layers in model order with their own stamps.
    assert_eq!(catalog.layer_count(), 2);
    assert_eq!(catalog.model_layer(0), 0);
    assert_eq!(catalog.model_layer(1), 1);
    assert_eq!(catalog.snapshot_stamp(0).expect("stamp 0"), stamp(6));
    assert_eq!(catalog.snapshot_stamp(1).expect("stamp 1"), stamp(11));
    assert_eq!(catalog.snapshot_block_count(0), 2);
    let b0 = catalog.snapshot_block(0, 0).expect("block 0");
    assert_eq!(b0.block.id, 100);
    assert_eq!(b0.host_version, 1);
    assert_eq!(catalog.snapshot_block(0, 1).expect("block 1").block.id, 200);
    assert_eq!(catalog.snapshot_resident_count(0), 1);
    assert_eq!(catalog.snapshot_resident(0, 0), 100);
    assert_eq!(catalog.snapshot_resident(1, 0), 200);

    // Plan layer 0's swap: restore 200, drop-evict 100 (sealed host copy → no spill).
    let plan = catalog.plan_layer(&sel0()).expect("plan layer 0");
    assert_eq!(plan.model_layer(), 0);
    assert_eq!(plan.restore_count(), 1);
    assert_eq!(plan.restore_block(0).expect("restore").id, 200);
    assert_eq!(plan.evict_count(), 1);
    assert_eq!(plan.evict_block(0).expect("evict").id, 100);
    assert_eq!(plan.spill_count(), 0);
    assert_eq!(plan.mandatory_count(), 0);
    assert_eq!(plan.next_visible_tokens(), BLOCK as u64);
    assert_eq!(plan.next_block_count(), 1);
    let nb = plan.next_block(0).expect("next block");
    assert_eq!(nb.block.id, 200);
    assert_eq!(nb.compact_start, 0);
    let ns = plan.next_stamp().expect("next stamp");
    assert_eq!(ns.view_version, 7); // 6 + 1
    assert_eq!(ns.frontier, 8);
    assert_eq!(ns.history_revision, 5);
    assert_eq!(plan.restore_bytes(), LAYER_BYTES);
    assert_eq!(plan.spill_bytes(), 0);
    assert_eq!(plan.device_tokens_required(), BLOCK as u64);
    assert_eq!(plan.host_payload_bytes_required(), RECORD_BYTES * 2);

    // The per-layer transfer regions are a separate KvRegions handle.
    let regions = plan.payload().expect("payload regions");
    assert_eq!(regions.region_count(), 2);
    assert_eq!(regions.plane_bytes(), LAYER_BYTES);
    assert_eq!(regions.packed_bytes(), LAYER_BYTES); // alignment 1, no padding
    assert_eq!(regions.record_bytes(), RECORD_BYTES);

    // Layer 1 swaps the other way.
    let plan1 = catalog.plan_layer(&sel1()).expect("plan layer 1");
    assert_eq!(plan1.model_layer(), 1);
    assert_eq!(plan1.restore_block(0).expect("restore").id, 100);
    assert_eq!(plan1.evict_block(0).expect("evict").id, 200);
    assert_eq!(plan1.next_stamp().expect("stamp").view_version, 12); // 11 + 1
    assert_eq!(plan1.restore_bytes(), LAYER_BYTES);
}

#[test]
fn bulk_plan_shared_archive_accounting() {
    let desc = descriptor();
    let catalog = hipfire_kvmem::MemoryCatalog::new(&desc, &layers()).expect("catalog");
    let plan = catalog.plan(&[sel0(), sel1()]).expect("bulk plan");
    assert_eq!(plan.layer_count(), 2);
    let history = plan.history().expect("history");
    assert_eq!(history.history_revision, 5);
    assert_eq!(history.frontier, 8);
    assert_eq!(history.view_version, 0); // bulk history identity carries no view version
    // Union of canonical full-record IDs, NOT the sum of per-layer requirements.
    assert_eq!(plan.host_payload_bytes_required(), RECORD_BYTES * 2);
    let l0 = plan.layer(0).expect("layer 0");
    let l1 = plan.layer(1).expect("layer 1");
    assert_eq!(l0.model_layer(), 0);
    assert_eq!(l1.model_layer(), 1);
    assert_eq!(l0.restore_block(0).expect("restore").id, 200);
    assert_eq!(l1.restore_block(0).expect("restore").id, 100);
    assert!(plan.layer(2).is_err(), "out-of-range layer index");
}

#[test]
fn fail_closed_admission_and_stale_selection() {
    let desc = descriptor();
    let catalog = hipfire_kvmem::MemoryCatalog::new(&desc, &layers()).expect("catalog");

    // A stale expected view version is rejected.
    let mut stale = sel0();
    stale.expected.view_version = 999;
    assert!(catalog.plan_layer(&stale).is_err(), "stale expected stamp");

    // An unknown target layer is rejected.
    let mut unknown = sel0();
    unknown.model_layer = 5;
    assert!(catalog.plan_layer(&unknown).is_err(), "unknown layer");

    // A selection over budget is rejected (device_tokens too small for the page).
    let mut tight = sel0();
    tight.budget.device_tokens = 0;
    assert!(catalog.plan_layer(&tight).is_err(), "device budget exceeded");

    // Inconsistent resident extent between layers fails catalog admission.
    let mut bad_extent = layers();
    bad_extent[1].memory.resident = vec![100, 200]; // 8 tokens vs layer 0's 4
    assert!(
        hipfire_kvmem::MemoryCatalog::new(&desc, &bad_extent).is_err(),
        "inconsistent resident extent"
    );

    // A divergent accepted history fails catalog admission.
    let mut bad_history = layers();
    bad_history[1].memory.stamp.history_revision = 6;
    assert!(
        hipfire_kvmem::MemoryCatalog::new(&desc, &bad_history).is_err(),
        "divergent history"
    );

    // A cold block without a completed current host copy fails admission.
    let mut no_host = layers();
    no_host[0].memory.blocks[1].host_version = 0; // block 200 cold in layer 0
    assert!(
        hipfire_kvmem::MemoryCatalog::new(&desc, &no_host).is_err(),
        "cold history without host copy"
    );
}