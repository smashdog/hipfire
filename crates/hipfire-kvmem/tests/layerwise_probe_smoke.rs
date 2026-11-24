// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only smoke test for the layerwise MTP private-probe selection
//! (UpstreamGlobal head history): no GPU, no model. Mirrors kvmem's
//! LayerwiseProbeSelection scalar reference over the C ABI — admit a frozen
//! two-target probe against a finite metadata account, append the private TARGET
//! forward's pre-RoPE Q token-major (a host Mean-K stand-in scorer), then finish
//! to the global-reduction head-history pick. Proves the C-ABI scorer plumbing,
//! the uniform L-layer global averaging (NOT a per-layer pick), the token-major
//! coverage contract, tiling invariance and fail-closed abort — with no GPU and
//! no model. The statistics oracle uses head_dim 1 (n_embd_k 1); the two target
//! layers share one frozen page denominator.

use std::cell::Cell;
use std::ffi::c_void;
use std::rc::Rc;

use hipfire_kvmem::{
    KvMemError, KvStore, KvStoreConfig, LayerQueryGeometry, LayerwiseIdentity, MemoryStamp,
    NativeCurrentLayerQuery, NativeQueryScalar, ProbePhase, ProbeSelection, QueryRange,
    QueryScoreFrame, RawStore, RawStoreConfig, SessionIdentity, Storage,
};

const BLOCK: u32 = 4;
// evaluation span [8, 12); query span [9, 11); frontier 8 == policy.total_tokens.
const QUERY_ROWS: u32 = 2;

fn session() -> SessionIdentity {
    SessionIdentity {
        id: 42,
        incarnation: 3,
    }
}

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

fn stamp() -> MemoryStamp {
    MemoryStamp {
        valid: 1,
        session: session(),
        history_revision: 5,
        view_version: 6,
        frontier: 8,
        execution_history: 9,
    }
}

fn identity() -> LayerwiseIdentity {
    LayerwiseIdentity {
        history: stamp(),
        request: 77,
        evaluation: QueryRange { begin: 8, end: 12 },
        query: QueryRange { begin: 9, end: 11 },
    }
}

fn geometry(layer: u32) -> LayerQueryGeometry {
    LayerQueryGeometry {
        layer,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        attention_scale: 1.0,
    }
}

fn targets() -> Vec<LayerQueryGeometry> {
    vec![geometry(0), geometry(1)]
}

/// Sealed plain Mean-K TopK index: 2 canonical blocks of 4 tokens, budget 1 block.
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
        raw.write_layer_mean_k(0, BLOCK, il, &[-1.0; BLOCK as usize])
            .expect("mean k block0");
        raw.write_layer_mean_k(BLOCK, BLOCK, il, &[1.0; BLOCK as usize])
            .expect("mean k block1");
    }
    raw
}

/// A finite metadata account large enough for the probe accumulator bound.
fn metadata() -> Storage {
    Storage::new_ram(1 << 20).expect("metadata account")
}

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
/// over the sliced Query rows of exp(q · mean_k · scale). Asserts the frozen frame
/// shape and forces masked blocks to exactly 0.
fn reference_scorer(
    mean_k: Vec<f32>,
) -> impl FnMut(&NativeCurrentLayerQuery, &QueryScoreFrame<'_>) -> Result<Vec<f64>, KvMemError> {
    move |tile, frame| {
        assert_eq!(frame.starts, &[0u64, 4], "frozen frame starts");
        assert_eq!(frame.counts, &[4u32, 4], "frozen frame counts");
        assert_eq!(frame.score_mask, &[1u8, 1], "frozen frame score mask");
        assert_eq!(frame.accepted_through, 8, "frozen frame frontier");
        let mut operands = Vec::with_capacity(tile.rows as usize);
        for r in 0..tile.rows as usize {
            let q = unsafe {
                let p = (tile.data as *const u8).add(r * tile.row_stride) as *const f32;
                *p
            };
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

/// Append one Query tile token-major: every target layer in geometry order. Both
/// layers carry the same aligned Q (favor block1) so the global L-layer average
/// still picks block1 — proving the reduction averages layers, not picks per-layer.
fn append_tile(probe: &ProbeSelection, id: &LayerwiseIdentity, begin: u64, q: &[f32]) {
    for ordinal in 0..2u32 {
        let tile = native_query(q, ordinal);
        probe
            .append_native(id, begin, &tile, reference_scorer(vec![-1.0, 1.0]))
            .expect("append_native");
    }
}

fn admit(revision: MemoryStamp) -> (ProbeSelection, Storage, KvStore, RawStore) {
    let meta = metadata();
    let pol = policy();
    let key = keys();
    let id = identity();
    let bound = ProbeSelection::metadata_bound(&pol, 2).expect("metadata bound");
    assert!(bound > 0, "probe metadata bound must be positive");
    let probe = ProbeSelection::new(&id, &pol, &key, &targets(), move || revision, &meta, &[])
        .expect("probe create");
    (probe, meta, pol, key)
}

#[test]
fn global_reduction_selection_and_tiling_invariance() {
    // Both target layers favor block1 (mean_k +1). The global average picks block1
    // for every tiling of the query span [9, 11), and the accumulated mass is
    // tiling-invariant (the core averages L exactly once per token row).
    let mut reference: Option<Vec<f64>> = None;
    for chunk in [1u32, 2] {
        let (probe, _meta, _pol, _key) = admit(stamp());
        assert_eq!(probe.phase(), ProbePhase::Collecting);
        assert_eq!(probe.scored_rows(), 0);

        let q = [1.0f32; QUERY_ROWS as usize];
        let mut first = 0u32;
        while first < QUERY_ROWS {
            let count = chunk.min(QUERY_ROWS - first);
            append_tile(
                &probe,
                &identity(),
                9 + first as u64,
                &q[first as usize..(first + count) as usize],
            );
            first += count;
            if first != QUERY_ROWS {
                // Token-major coverage incomplete: finish must be Pending.
                assert!(
                    probe.finish(&identity()).is_err(),
                    "finish before full query coverage"
                );
                assert_eq!(probe.phase(), ProbePhase::Collecting);
            }
        }
        assert_eq!(probe.scored_rows(), QUERY_ROWS as u64);
        assert_eq!(
            probe.next_layer(),
            0,
            "layer cursor wraps after each complete tile"
        );

        probe.finish(&identity()).expect("finish");
        assert_eq!(probe.phase(), ProbePhase::Complete);

        // Budget 1 block → exactly the aligned block1 (ordinal 1).
        assert_eq!(probe.selected_count(), 1);
        assert_eq!(probe.selected_ids(), vec![1u32]);
        assert_eq!(probe.fixed_count(), 0);
        assert_eq!(probe.historical_tokens(), BLOCK as u64);
        assert_eq!(probe.charged_tokens(), BLOCK as u64);
        assert_eq!(probe.score_count(), 2);
        let s0 = probe.score(0).expect("score 0");
        let s1 = probe.score(1).expect("score 1");
        assert!(s0 >= 0.0 && s1 >= 0.0, "page mass non-negative: {s0} {s1}");
        assert!(
            s1 > s0,
            "aligned block1 must win the global reduction: {s0} {s1}"
        );
        assert!(probe.score(2).is_err(), "out-of-range score index");

        // Global L-layer average: 2 layers × 2 query rows × exp(±1), averaged /2.
        let expect1 = 2.0 * (1.0f64).exp();
        let expect0 = 2.0 * (-1.0f64).exp();
        assert!(
            (s1 - expect1).abs() < 1e-9,
            "block1 global mass {s1} != {expect1}"
        );
        assert!(
            (s0 - expect0).abs() < 1e-9,
            "block0 global mass {s0} != {expect0}"
        );

        match &reference {
            None => reference = Some(vec![s0, s1]),
            Some(r) => assert!(
                (r[0] - s0).abs() < 1e-9 && (r[1] - s1).abs() < 1e-9,
                "tiling invariant"
            ),
        }
    }
}

#[test]
fn metadata_bound_requires_finite_account_and_layers() {
    let pol = policy();
    let bound = ProbeSelection::metadata_bound(&pol, 2).expect("bound");
    assert!(bound > 0);
    // Zero target layers is rejected fail-closed.
    assert!(
        ProbeSelection::metadata_bound(&pol, 0).is_err(),
        "zero target layers"
    );
    // The bound must fit a finite account; an undersized account rejects create.
    let tiny = Storage::new_ram(1).expect("tiny account");
    let key = keys();
    let id = identity();
    let rev = stamp();
    assert!(
        ProbeSelection::new(&id, &pol, &key, &targets(), move || rev, &tiny, &[]).is_err(),
        "metadata account too small for the probe bound"
    );
}

#[test]
fn fail_closed_revision_identity_and_coverage() {
    let pol = policy();
    let key = keys();
    let id = identity();

    // A source revision that advances after admission aborts at the next boundary.
    {
        let meta = metadata();
        let revision = Rc::new(Cell::new(stamp()));
        let rev = revision.clone();
        let probe = ProbeSelection::new(&id, &pol, &key, &targets(), move || rev.get(), &meta, &[])
            .expect("probe");
        revision.set({
            let mut s = stamp();
            s.history_revision += 1;
            s
        });
        let tile = native_query(&[1.0], 0);
        assert!(
            probe
                .append_native(&id, 9, &tile, reference_scorer(vec![-1.0, 1.0]))
                .is_err(),
            "stale source revision"
        );
        assert_eq!(probe.phase(), ProbePhase::Aborted);
    }

    // A foreign request id is Stale; an out-of-order layer is InvalidPlan.
    {
        let (probe, _meta, _pol, _key) = admit(stamp());
        let mut foreign = id;
        foreign.request += 1;
        let tile = native_query(&[1.0], 0);
        assert!(
            probe
                .append_native(&foreign, 9, &tile, reference_scorer(vec![-1.0, 1.0]))
                .is_err(),
            "foreign identity"
        );
        // Layer 1 before layer 0 breaks token-major target coverage.
        let wrong = native_query(&[1.0], 1);
        assert!(
            probe
                .append_native(&id, 9, &wrong, reference_scorer(vec![-1.0, 1.0]))
                .is_err(),
            "out-of-order target layer"
        );
        assert_eq!(
            probe.phase(),
            ProbePhase::Collecting,
            "validation must not abort"
        );
        probe.abort();
        assert_eq!(probe.phase(), ProbePhase::Aborted);
    }

    // A wrong geometry (head_dim mismatch) is rejected at create.
    {
        let meta = metadata();
        let rev = stamp();
        let mut bad = targets();
        bad[1].head_dim = 2; // kv_heads*head_dim != n_embd_k
        assert!(
            ProbeSelection::new(&id, &pol, &key, &bad, move || rev, &meta, &[]).is_err(),
            "geometry differs from upstream LTHD"
        );
    }
}

#[test]
fn native_scorer_fail_closed() {
    let (probe, _meta, _pol, _key) = admit(stamp());
    let id = identity();
    let tile = native_query(&[1.0, 1.0], 0);
    // A failing scorer propagates as HK_ERR (never unwinds across the C ABI). The
    // core accumulator does NOT auto-abort — the private-probe DRIVER owns the
    // abort/rollback envelope (mirrors ninfer native_layerwise_probe's try/catch
    // calling selection.abort()), so the phase stays Collecting until the driver
    // aborts. This is the contract H2's run_private_probe must honor.
    let mut failing = |_: &NativeCurrentLayerQuery, _: &QueryScoreFrame<'_>| {
        Err(KvMemError("scorer failed".into()))
    };
    assert!(
        probe.append_native(&id, 9, &tile, &mut failing).is_err(),
        "scorer failure"
    );
    assert_eq!(
        probe.phase(),
        ProbePhase::Collecting,
        "core leaves abort to the driver"
    );
    // The driver aborts fail-closed; a closed probe rejects further projection.
    probe.abort();
    assert_eq!(probe.phase(), ProbePhase::Aborted);
    assert!(
        probe
            .append_native(&id, 9, &tile, reference_scorer(vec![-1.0, 1.0]))
            .is_err(),
        "closed"
    );
}
