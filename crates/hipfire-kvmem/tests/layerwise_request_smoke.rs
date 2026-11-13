// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only smoke test for the layerwise Query request state machine: no GPU,
//! no model. Mirrors kvmem's LayerwiseQueryRequest scalar reference over the C
//! ABI — admit a frozen two-layer transaction, project the current-forward Q
//! (host Mean-K path), freeze the per-layer selection, report the restored view,
//! attend, complete and accept the canonical suffix. The statistics oracle uses
//! head_dim 1 (n_embd_k 1); the packed catalog layout is decoupled (model layers
//! 2/7 vs statistics ordinals 0/1) so a swapped model_layer/query.layer is caught.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::rc::Rc;

use hipfire_kvmem::{
    BackendIdentity, BlockCopies, BlockDescriptor, ExecutionView, KvMemError, KvStore, KvStoreConfig,
    LayerMemorySnapshot, LayerQueryGeometry, LayerRequestSpec, LayerwiseIdentity, LayerwisePhase,
    LayerwiseRequest, MemoryBudget, MemoryCapabilities, MemoryDescriptor, MemorySnapshot,
    MemoryStamp, NativeCurrentLayerQuery, NativeQueryScalar, PayloadLayout, QueryRange,
    QueryScoreFrame, RawStore, RawStoreConfig, SessionIdentity,
};

const BLOCK: u32 = 4;
const EVAL_ROWS: u32 = 4; // evaluation span [8, 12); query span [9, 11)
const ROW_BYTES: u64 = 4; // one head_dim-1 F32 token row
const PLANE_BYTES: u64 = ROW_BYTES * BLOCK as u64; // 16
const LAYER_BYTES: u64 = PLANE_BYTES * 2; // 32 (K+V for one layer)

fn session() -> SessionIdentity {
    SessionIdentity { id: 42, incarnation: 3 }
}

/// Statistics-only keys store: head_dim 1, no packed rows.
fn stats_config() -> RawStoreConfig {
    RawStoreConfig {
        n_layer: 2,
        n_embd_k: 1,
        n_embd_v: 0,
        block_tokens: BLOCK,
        k_row_bytes: 0,
        k_gpu_row_bytes: 0,
        v_gpu_row_bytes: 0,
    }
}

/// Packed payload layout source: model layers 2 and 7 (n_layer 8 admits them).
fn payload_config() -> RawStoreConfig {
    RawStoreConfig {
        n_layer: 8,
        n_embd_k: 1,
        n_embd_v: 1,
        block_tokens: BLOCK,
        k_row_bytes: 0,
        k_gpu_row_bytes: ROW_BYTES,
        v_gpu_row_bytes: ROW_BYTES,
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

fn descriptor() -> MemoryDescriptor {
    let layout =
        PayloadLayout::from_raw(&payload_config(), "scalar-request-reference", &[2, 7])
            .expect("layout");
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

fn identity() -> LayerwiseIdentity {
    LayerwiseIdentity {
        history: stamp(6),
        request: 77,
        evaluation: QueryRange { begin: 8, end: 12 },
        query: QueryRange { begin: 9, end: 11 },
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

fn initial() -> Vec<LayerMemorySnapshot> {
    vec![
        LayerMemorySnapshot { model_layer: 2, memory: snapshot(6, vec![100]) },
        LayerMemorySnapshot { model_layer: 7, memory: snapshot(11, vec![200]) },
    ]
}

fn budget() -> MemoryBudget {
    MemoryBudget {
        selected_tokens: BLOCK as u64,
        reserve_tokens: BLOCK as u64,
        device_tokens: 2 * BLOCK as u64,
        host_payload_bytes: 192,
    }
}

fn geometry(layer: u32) -> LayerQueryGeometry {
    LayerQueryGeometry { layer, query_heads: 1, kv_heads: 1, head_dim: 1, attention_scale: 1.0 }
}

fn specs() -> Vec<LayerRequestSpec> {
    vec![
        LayerRequestSpec { model_layer: 2, query: geometry(0), budget: budget(), mandatory: vec![] },
        LayerRequestSpec { model_layer: 7, query: geometry(1), budget: budget(), mandatory: vec![] },
    ]
}

fn policy() -> KvStore {
    let cfg = KvStoreConfig {
        block_tokens: BLOCK,
        select_budget: BLOCK,
        sink_blocks: 0,
        recent_blocks: 0,
        ..KvStoreConfig::default()
    };
    let kv = KvStore::new(&cfg).expect("policy");
    kv.register_append(2 * BLOCK).expect("append");
    kv
}

/// block0 mean_k = -1, block1 mean_k = +1 for both statistics ordinals.
fn keys() -> RawStore {
    let raw = RawStore::new(stats_config(), None).expect("keys");
    for il in 0..2u32 {
        raw.write_layer_mean_k(0, BLOCK, il, &[-1.0; BLOCK as usize]).expect("mean k block0");
        raw.write_layer_mean_k(BLOCK, BLOCK, il, &[1.0; BLOCK as usize]).expect("mean k block1");
    }
    raw
}

/// One target layer's full forward via the HOST Mean-K path: project (tiled),
/// then freeze/restore/attend/complete. Returns the published view for acceptance.
fn execute_layer(
    request: &LayerwiseRequest,
    id: &LayerwiseIdentity,
    model_layer: u32,
    ordinal: u32,
    q: &[f32],
    chunk: u32,
    expect_ordinal: u32,
    expect_block: u64,
) -> ExecutionView {
    request.begin_layer(id, model_layer).expect("begin_layer");
    assert_eq!(request.phase(), LayerwisePhase::Projecting);

    // Report every evaluation row exactly once; the scorer keeps only the query
    // intersection [9, 11), so the opposed header/trailer logits are a trap.
    let mut first = 0u32;
    while first < EVAL_ROWS {
        let count = chunk.min(EVAL_ROWS - first);
        request
            .projected(
                id,
                8 + first as u64,
                count,
                &q[first as usize..(first + count) as usize],
                geometry(ordinal),
            )
            .expect("projected");
        first += count;
        if first != EVAL_ROWS {
            assert!(
                request.prepare_restore(id).is_err(),
                "prepare_restore before full Query projection must be Pending"
            );
        }
    }

    restore_attend_complete(request, id, model_layer, ordinal, chunk, expect_ordinal, expect_block)
}

/// Freeze this layer's selection, restore the packed compact view, attend
/// (tiled) and complete. Shared by the host and native projection paths: the
/// selection/plan contract is identical regardless of how Q was scored.
fn restore_attend_complete(
    request: &LayerwiseRequest,
    id: &LayerwiseIdentity,
    model_layer: u32,
    ordinal: u32,
    chunk: u32,
    expect_ordinal: u32,
    expect_block: u64,
) -> ExecutionView {
    let plan = request.prepare_restore(id).expect("prepare_restore");
    assert_eq!(request.phase(), LayerwisePhase::Restoring);
    assert_eq!(plan.model_layer(), model_layer);
    assert_eq!(plan.next_block_count(), 1);
    assert_eq!(plan.next_block(0).expect("next block").block.id, expect_block);
    assert_eq!(plan.next_block(0).expect("next block").compact_start, 0);
    assert_eq!(plan.restore_count(), 1);
    assert_eq!(plan.restore_block(0).expect("restore").id, expect_block);
    assert_eq!(plan.spill_count(), 0); // sealed host archive → drop-evict, no spill
    assert_eq!(plan.restore_bytes(), LAYER_BYTES);
    assert_eq!(plan.next_visible_tokens(), BLOCK as u64);

    // Frozen selection: exactly one block, no fixed bands, 4 historical tokens.
    assert_eq!(request.selection_layer(), ordinal);
    assert_eq!(request.selection_selected_count(), 1);
    assert_eq!(request.selection_selected(0), expect_ordinal);
    assert_eq!(request.selection_fixed_count(), 0);
    assert_eq!(request.selection_historical_tokens(), BLOCK as u64);
    assert_eq!(request.selection_charged_tokens(), BLOCK as u64);
    assert_eq!(request.selection_score_count(), 2);
    let s0 = request.selection_score(0).expect("score 0");
    let s1 = request.selection_score(1).expect("score 1");
    assert!(s0 >= 0.0 && s1 >= 0.0, "page mass must be non-negative: {s0} {s1}");
    if expect_ordinal == 1 {
        assert!(s1 > s0, "aligned block1 should win: {s0} {s1}");
    } else {
        assert!(s0 > s1, "aligned block0 should win: {s0} {s1}");
    }
    assert!(request.selection_score(2).is_err(), "out-of-range score index");

    // The driver restores the packed bytes then reports the view it published;
    // for the host ABI smoke that view is exactly the plan's compact next view.
    let view = ExecutionView {
        stamp: plan.next_stamp().expect("next stamp"),
        blocks: (0..plan.next_block_count())
            .map(|i| plan.next_block(i).expect("next block"))
            .collect(),
        visible_tokens: plan.next_visible_tokens(),
    };
    request.restored(id, &view).expect("restored");
    assert_eq!(request.phase(), LayerwisePhase::Attending);

    let mut first = 0u32;
    while first < EVAL_ROWS {
        let count = chunk.min(EVAL_ROWS - first);
        request.attended(id, 8 + first as u64, count as u64).expect("attended");
        first += count;
    }
    request.complete_layer(id).expect("complete_layer");
    view
}

/// Accepted canonical snapshots: frozen prefix unchanged + resident suffix 300@8.
fn accepted_snapshots(views: &[ExecutionView], selected: &[u64]) -> Vec<LayerMemorySnapshot> {
    [2u32, 7u32]
        .iter()
        .enumerate()
        .map(|(i, &model_layer)| {
            let mut st = views[i].stamp;
            st.history_revision += 1;
            st.view_version += 1;
            st.execution_history += 1;
            st.frontier = 12;
            let mut blocks = vec![
                BlockCopies { block: block(100, 0, 1), host_version: 1 },
                BlockCopies { block: block(200, 4, 2), host_version: 2 },
            ];
            blocks.push(BlockCopies {
                block: BlockDescriptor {
                    session: session(),
                    id: 300,
                    original_start: 8,
                    valid_tokens: BLOCK,
                    content_version: 3,
                    statistics_version: 1,
                },
                host_version: 0, // resident suffix: no sealed host copy required
            });
            LayerMemorySnapshot {
                model_layer,
                memory: MemorySnapshot { stamp: st, blocks, resident: vec![selected[i], 300] },
            }
        })
        .collect()
}

#[test]
fn scalar_reference_selection_restore_and_acceptance() {
    // The two layers' true Query rows oppose each other; headers/trailers carry
    // strong opposed logits that would reverse the choice if accidentally scored.
    let layer0_q = [-1000.0f32, 1.0, 1.0, -1000.0];
    let layer1_q = [1000.0f32, -2.0, -2.0, 1000.0];

    for chunk in [1u32, 2, 3, 4] {
        let desc = descriptor();
        let catalog = hipfire_kvmem::MemoryCatalog::new(&desc, &initial()).expect("catalog");
        let pol = policy();
        let key = keys();
        let id = identity();
        let revision = Rc::new(Cell::new(id.history));
        let rev = revision.clone();
        let request = LayerwiseRequest::new(
            &id,
            &catalog,
            &pol,
            &key,
            &specs(),
            BLOCK as u64,
            move || rev.get(),
        )
        .expect("request");
        assert_eq!(request.phase(), LayerwisePhase::BetweenLayers);

        let v0 = execute_layer(&request, &id, 2, 0, &layer0_q, chunk, 1, 200);
        assert_eq!(request.phase(), LayerwisePhase::BetweenLayers);
        let v1 = execute_layer(&request, &id, 7, 1, &layer1_q, chunk, 0, 100);
        assert_eq!(request.phase(), LayerwisePhase::AwaitingAcceptance);

        let work = request.work().expect("work");
        assert_eq!(work.projected_rows, 2 * EVAL_ROWS as u64);
        assert_eq!(work.attended_rows, 2 * EVAL_ROWS as u64);
        assert_eq!(work.completed_layers, 2);

        // Acceptance errors must not expose a partially committed result.
        let accepted = accepted_snapshots(&[v0, v1], &[200, 100]);
        let mut bad = accepted.clone();
        bad.pop();
        assert!(request.accepted(&id, &bad).is_err(), "incomplete acceptance");
        assert_eq!(request.phase(), LayerwisePhase::AwaitingAcceptance);
        assert_eq!(request.accepted_layer_count(), 0);

        // The index lease may end now: final acceptance does not re-read source.
        revision.set({
            let mut s = id.history;
            s.history_revision += 1;
            s
        });
        request.accepted(&id, &accepted).expect("accepted");
        assert_eq!(request.phase(), LayerwisePhase::Accepted);

        assert_eq!(request.accepted_layer_count(), 2);
        assert_eq!(request.accepted_model_layer(0), 2);
        assert_eq!(request.accepted_model_layer(1), 7);
        assert_eq!(request.accepted_resident_count(0), 2);
        assert_eq!(request.accepted_resident(0, 0), 200);
        assert_eq!(request.accepted_resident(0, 1), 300);
        assert_eq!(request.accepted_resident(1, 0), 100);
        assert_eq!(request.accepted_resident(1, 1), 300);
        for i in 0..2 {
            let st = request.accepted_stamp(i).expect("accepted stamp");
            assert_eq!(st.frontier, 12);
            assert_eq!(st.history_revision, 6);
            assert_eq!(request.accepted_block_count(i), 3);
            let suffix = request.accepted_block(i, 2).expect("suffix block");
            assert_eq!(suffix.block.id, 300);
            assert_eq!(suffix.block.original_start, 8);
            assert_eq!(suffix.block.valid_tokens, BLOCK);
            assert_eq!(suffix.host_version, 0);
        }

        // A closed request rejects further layers and re-acceptance.
        assert!(request.begin_layer(&id, 2).is_err(), "accepted request is closed");
        assert!(request.accepted(&id, &accepted).is_err(), "double acceptance");
    }
}

#[test]
fn fail_closed_revision_identity_and_phase() {
    let desc = descriptor();
    let catalog = hipfire_kvmem::MemoryCatalog::new(&desc, &initial()).expect("catalog");
    let pol = policy();
    let key = keys();
    let id = identity();

    // A source revision that advances after admission aborts at the next boundary.
    {
        let revision = Rc::new(Cell::new(id.history)); // valid at admission
        let rev = revision.clone();
        let request = LayerwiseRequest::new(
            &id, &catalog, &pol, &key, &specs(), BLOCK as u64, move || rev.get(),
        )
        .expect("request");
        revision.set({
            let mut s = id.history;
            s.history_revision += 1; // index advanced past the frozen stamp
            s
        });
        assert!(request.begin_layer(&id, 2).is_err(), "stale source revision");
        assert_eq!(request.phase(), LayerwisePhase::Aborted);
    }

    // A foreign request id is Stale; a future model layer is InvalidPlan.
    {
        let rev = id.history;
        let request = LayerwiseRequest::new(
            &id, &catalog, &pol, &key, &specs(), BLOCK as u64, move || rev,
        )
        .expect("request");
        let mut foreign = id;
        foreign.request += 1;
        assert!(request.begin_layer(&foreign, 2).is_err(), "foreign identity");
        assert!(request.begin_layer(&id, 7).is_err(), "future model layer out of order");
        assert_eq!(request.phase(), LayerwisePhase::BetweenLayers);

        // Premature prepare_restore (partial evaluation coverage) is Pending.
        request.begin_layer(&id, 2).expect("begin_layer");
        request
            .projected(&id, 8, 1, &[-1000.0], geometry(0))
            .expect("partial projected");
        assert!(request.prepare_restore(&id).is_err(), "incomplete projection");
        assert_eq!(request.phase(), LayerwisePhase::Projecting);
        request.abort();
        assert_eq!(request.phase(), LayerwisePhase::Aborted);
    }
}

#[test]
fn admission_rejects_budget_and_geometry() {
    let desc = descriptor();
    let catalog = hipfire_kvmem::MemoryCatalog::new(&desc, &initial()).expect("catalog");
    let pol = policy();
    let key = keys();
    let id = identity();
    let rev = id.history;

    // selected_tokens must equal the policy's select budget.
    let mut bad = specs();
    bad[0].budget.selected_tokens = 2 * BLOCK as u64;
    assert!(
        LayerwiseRequest::new(&id, &catalog, &pol, &key, &bad, BLOCK as u64, move || rev).is_err(),
        "selected_tokens != policy budget"
    );

    // Statistics ordinals must be unique and within the keys store.
    let mut dup = specs();
    dup[1].query.layer = 0;
    assert!(
        LayerwiseRequest::new(&id, &catalog, &pol, &key, &dup, BLOCK as u64, move || rev).is_err(),
        "duplicate statistics layer"
    );

    // native_history_tokens must be block-aligned and not exceed the frontier.
    assert!(
        LayerwiseRequest::new(&id, &catalog, &pol, &key, &specs(), 3, move || rev).is_err(),
        "unaligned native history extent"
    );
}

// ── native (device-ready) Query projection ──
//
// The same transaction driven through projected_native: a Rust scorer closure
// stands in for hipfire's GPU qualified Mean-K kernel. The core hands it ONLY the
// query-intersection rows it sliced out of each evaluation tile plus the ONE
// frozen per-block frame; the returned page masses accumulate across tiles before
// a single pick. These tests prove the C-ABI scorer plumbing — tile slicing, frame
// delivery, score extent, fail-closed abort vs recoverable validation — with no
// GPU and no model.

/// Build a full-evaluation native Q tile (F32, head_dim 1, token-major). Strides
/// and `bytes` are BYTE quantities; the token axis is outermost here but the core
/// qualifies any non-overlapping permutation.
fn native_query(q: &[f32], ordinal: u32) -> NativeCurrentLayerQuery {
    NativeCurrentLayerQuery {
        data: q.as_ptr() as *const c_void,
        bytes: q.len() * std::mem::size_of::<f32>(),
        layer: ordinal,
        rows: q.len() as u32,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        attention_scale: 1.0,
        scalar: NativeQueryScalar::Float32 as u32,
        row_stride: 4,
        head_stride: 4,
        dimension_stride: 4,
    }
}

/// Host reference stand-in for the GPU qualified Mean-K scorer: page mass = sum
/// over the sliced Query rows of exp(q · mean_k · scale). It reads each operand
/// through the native byte strides, logs every value it actually scores (so a
/// test can prove the ±1000 header/trailer traps were never touched), asserts the
/// frozen frame shape, and forces masked blocks to exactly 0.
fn reference_scorer(
    mean_k: Vec<f32>,
    log: Rc<RefCell<Vec<f32>>>,
) -> impl FnMut(&NativeCurrentLayerQuery, &QueryScoreFrame<'_>) -> Result<Vec<f64>, KvMemError> {
    move |tile, frame| {
        // The frozen denominator: 2 canonical blocks, both scorable, 4 tokens each.
        assert_eq!(frame.starts, &[0u64, 4], "frozen frame starts");
        assert_eq!(frame.counts, &[4u32, 4], "frozen frame counts");
        assert_eq!(frame.score_mask, &[1u8, 1], "frozen frame score mask");
        assert_eq!(frame.accepted_through, 8, "frozen frame frontier");
        // Read each sliced Query row exactly once (token-major, head_dim 1) and
        // log it, so the operand log proves which rows entered scoring.
        let mut operands = Vec::with_capacity(tile.rows as usize);
        for r in 0..tile.rows as usize {
            let q = unsafe {
                let p = (tile.data as *const u8).add(r * tile.row_stride) as *const f32;
                *p
            };
            log.borrow_mut().push(q);
            operands.push(q as f64);
        }
        let mut scores = Vec::with_capacity(frame.starts.len());
        for i in 0..frame.starts.len() {
            if frame.score_mask[i] == 0 {
                scores.push(0.0);
                continue;
            }
            let mut mass = 0.0f64;
            for &q in &operands {
                mass += (q * mean_k[i] as f64 * tile.attention_scale as f64).exp();
            }
            scores.push(mass);
        }
        Ok(scores)
    }
}

/// One target layer's full forward via the NATIVE projection path. `log` collects
/// every Q operand the scorer actually reads, in tile order.
fn execute_layer_native(
    request: &LayerwiseRequest,
    id: &LayerwiseIdentity,
    model_layer: u32,
    ordinal: u32,
    q: &[f32],
    chunk: u32,
    expect_ordinal: u32,
    expect_block: u64,
    log: Rc<RefCell<Vec<f32>>>,
) -> ExecutionView {
    request.begin_layer(id, model_layer).expect("begin_layer");
    assert_eq!(request.phase(), LayerwisePhase::Projecting);

    let mean_k = vec![-1.0f32, 1.0]; // block0 = -1, block1 = +1 (both ordinals)
    let mut first = 0u32;
    while first < EVAL_ROWS {
        let count = chunk.min(EVAL_ROWS - first);
        let tile = native_query(&q[first as usize..(first + count) as usize], ordinal);
        request
            .projected_native(
                id,
                8 + first as u64,
                &tile,
                reference_scorer(mean_k.clone(), log.clone()),
            )
            .expect("projected_native");
        first += count;
        if first != EVAL_ROWS {
            assert!(
                request.prepare_restore(id).is_err(),
                "prepare_restore before full native Query projection must be Pending"
            );
        }
    }

    restore_attend_complete(request, id, model_layer, ordinal, chunk, expect_ordinal, expect_block)
}

#[test]
fn native_reference_selection_restore_and_acceptance() {
    // Identical opposed Query rows to the host reference; the native scorer must
    // see only the intersection [9, 11) and pick the same blocks for every tiling.
    let layer0_q = [-1000.0f32, 1.0, 1.0, -1000.0];
    let layer1_q = [1000.0f32, -2.0, -2.0, 1000.0];

    for chunk in [1u32, 2, 3, 4] {
        let desc = descriptor();
        let catalog = hipfire_kvmem::MemoryCatalog::new(&desc, &initial()).expect("catalog");
        let pol = policy();
        let key = keys();
        let id = identity();
        let revision = Rc::new(Cell::new(id.history));
        let rev = revision.clone();
        let request = LayerwiseRequest::new(
            &id,
            &catalog,
            &pol,
            &key,
            &specs(),
            BLOCK as u64,
            move || rev.get(),
        )
        .expect("request");
        assert_eq!(request.phase(), LayerwisePhase::BetweenLayers);

        let log0 = Rc::new(RefCell::new(Vec::new()));
        let v0 = execute_layer_native(&request, &id, 2, 0, &layer0_q, chunk, 1, 200, log0.clone());
        // The scorer read ONLY the two aligned query rows, never the ±1000 traps,
        // and the accumulated mass picked block1 (ordinal 1 → block 200).
        assert_eq!(*log0.borrow(), vec![1.0f32, 1.0], "chunk {chunk}: layer0 sliced operands");
        assert_eq!(request.phase(), LayerwisePhase::BetweenLayers);

        let log1 = Rc::new(RefCell::new(Vec::new()));
        let v1 = execute_layer_native(&request, &id, 7, 1, &layer1_q, chunk, 0, 100, log1.clone());
        assert_eq!(*log1.borrow(), vec![-2.0f32, -2.0], "chunk {chunk}: layer1 sliced operands");
        assert_eq!(request.phase(), LayerwisePhase::AwaitingAcceptance);

        let work = request.work().expect("work");
        assert_eq!(work.projected_rows, 2 * EVAL_ROWS as u64);
        assert_eq!(work.attended_rows, 2 * EVAL_ROWS as u64);
        assert_eq!(work.completed_layers, 2);

        // The index lease may end now: final acceptance does not re-read source.
        revision.set({
            let mut s = id.history;
            s.history_revision += 1;
            s
        });
        let accepted = accepted_snapshots(&[v0, v1], &[200, 100]);
        request.accepted(&id, &accepted).expect("accepted");
        assert_eq!(request.phase(), LayerwisePhase::Accepted);
        assert_eq!(request.accepted_layer_count(), 2);
        assert_eq!(request.accepted_resident(0, 0), 200);
        assert_eq!(request.accepted_resident(0, 1), 300);
        assert_eq!(request.accepted_resident(1, 0), 100);
        assert_eq!(request.accepted_resident(1, 1), 300);
        assert!(request.begin_layer(&id, 2).is_err(), "accepted request is closed");
    }
}

#[test]
fn native_scorer_fail_closed() {
    let desc = descriptor();
    let catalog = hipfire_kvmem::MemoryCatalog::new(&desc, &initial()).expect("catalog");
    let pol = policy();
    let key = keys();
    let id = identity();
    let rev = id.history;
    let q = [-1000.0f32, 1.0, 1.0, -1000.0];

    // Every scorer-side failure happens INSIDE the projection try block, so the
    // request must terminate Aborted (never unwind across the C ABI, never pick).
    let cases: Vec<(&str, Box<dyn FnMut(&NativeCurrentLayerQuery, &QueryScoreFrame<'_>) -> Result<Vec<f64>, KvMemError>>)> = vec![
        ("reported failure", Box::new(|_, _| Err(KvMemError("scorer failed".into())))),
        ("wrong score extent", Box::new(|_, f| Ok(vec![1.0; f.starts.len() + 1]))),
        ("negative page mass", Box::new(|_, f| { let mut s = vec![1.0; f.starts.len()]; s[0] = -1.0; Ok(s) })),
        ("nonfinite page mass", Box::new(|_, f| { let mut s = vec![1.0; f.starts.len()]; s[0] = f64::INFINITY; Ok(s) })),
        ("panicking scorer", Box::new(|_, _| panic!("native scorer panic"))),
    ];
    for (name, mut scorer) in cases {
        let request = LayerwiseRequest::new(
            &id, &catalog, &pol, &key, &specs(), BLOCK as u64, move || rev,
        )
        .expect("request");
        request.begin_layer(&id, 2).expect("begin_layer");
        let tile = native_query(&q, 0);
        assert!(
            request.projected_native(&id, 8, &tile, &mut scorer).is_err(),
            "{name} must propagate as an error"
        );
        assert_eq!(request.phase(), LayerwisePhase::Aborted, "{name} must abort fail-closed");
        // A closed request rejects further projection.
        assert!(request.projected_native(&id, 8, &tile, &mut scorer).is_err(), "{name}: closed");
    }
}

#[test]
fn native_geometry_and_stride_validation_is_recoverable() {
    let desc = descriptor();
    let catalog = hipfire_kvmem::MemoryCatalog::new(&desc, &initial()).expect("catalog");
    let pol = policy();
    let key = keys();
    let id = identity();
    let rev = id.history;
    let request = LayerwiseRequest::new(
        &id, &catalog, &pol, &key, &specs(), BLOCK as u64, move || rev,
    )
    .expect("request");
    request.begin_layer(&id, 2).expect("begin_layer");

    let q = [-1000.0f32, 1.0, 1.0, -1000.0];
    let ok_scorer = |_: &NativeCurrentLayerQuery, f: &QueryScoreFrame<'_>| Ok(vec![1.0; f.starts.len()]);

    // Geometry/stride validation precedes any progress mutation and lives OUTSIDE
    // the projection try block: it rejects the tile but leaves the layer open.
    let mut wrong_layer = native_query(&q, 0);
    wrong_layer.layer = 1; // admitted statistics ordinal for model2 is 0
    assert!(
        request.projected_native(&id, 8, &wrong_layer, ok_scorer).is_err(),
        "statistics layer mismatch"
    );
    assert_eq!(request.phase(), LayerwisePhase::Projecting, "validation must not abort");

    let mut bad_stride = native_query(&q, 0);
    bad_stride.row_stride = 0; // overlapping/zero stride is not a native permutation
    assert!(
        request.projected_native(&id, 8, &bad_stride, ok_scorer).is_err(),
        "invalid native stride"
    );
    assert_eq!(request.phase(), LayerwisePhase::Projecting, "validation must not abort");

    // The rejected tiles consumed no coverage: a correct full tile still admits,
    // scores the query intersection and freezes the selection.
    let log = Rc::new(RefCell::new(Vec::new()));
    let tile = native_query(&q, 0);
    request
        .projected_native(&id, 8, &tile, reference_scorer(vec![-1.0, 1.0], log.clone()))
        .expect("corrected native tile");
    assert_eq!(*log.borrow(), vec![1.0f32, 1.0], "only the query intersection is scored");
    let plan = request.prepare_restore(&id).expect("prepare_restore after recovery");
    assert_eq!(plan.next_block(0).expect("next block").block.id, 200);
    assert_eq!(request.phase(), LayerwisePhase::Restoring);
}