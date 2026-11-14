// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only smoke test for the engine-owned projected-Q history tape
//! (`kvmem::QueryProjectionTape`) and its finite quota accounts: no GPU, no
//! model. This is the layerwise activation-quota substrate (ninfer native Query
//! model). It proves the C-ABI plumbing for:
//!   * finite metadata admission — `metadata_bound` reserved against an
//!     independent account BEFORE any payload, fail-closed when it cannot fit;
//!   * RAM tier — full-layer reservation up front, exact contiguous coverage,
//!     byte-exact multi-plane roundtrip, cross-chunk reads, chunk-source bounds;
//!   * disk tier — a layer larger than RAM, written directly to a SpillFile span;
//!   * source survival — an exported sealed chunk outlives the tape AND its
//!     account (RAM allocation / disk extent both pin their own lifetime);
//!   * reservation accounting — charge on reserve, release unused on drop.
//!
//! The byte oracle is plane/row/byte-distinct so a swapped plane id, a wrong
//! per-plane offset, or a truncated read is caught rather than silently passing.

use hipfire_kvmem::{
    LayerwiseIdentity, MemoryStamp, QueryPlaneBytes, QueryProjectionLayout, QueryProjectionPlane,
    QueryProjectionTape, QueryRange, SessionIdentity, SpillFile, Storage,
};

// One fixed model layer, one exact evaluation span [8, 12) = 4 token rows.
const MODEL_LAYER: u32 = 2;
const EVAL_BEGIN: u64 = 8;
const EVAL_END: u64 = 12;
const EVAL_ROWS: u64 = EVAL_END - EVAL_BEGIN; // 4
const Q_PLANE: u32 = 10;
const Q_STRIDE: u64 = 16; // 4 F32 per token row
const R_PLANE: u32 = 20;
const R_STRIDE: u64 = 8; // 2 F32 per token row
const ROW_BYTES: u64 = Q_STRIDE + R_STRIDE; // 24
const TOTAL_BYTES: u64 = ROW_BYTES * EVAL_ROWS; // 96

fn session() -> SessionIdentity {
    SessionIdentity { id: 42, incarnation: 3 }
}

/// Accepted history stamp; frontier == evaluation.begin (identity validity).
fn stamp() -> MemoryStamp {
    MemoryStamp {
        valid: 1,
        session: session(),
        history_revision: 5,
        view_version: 6,
        frontier: EVAL_BEGIN,
        execution_history: 9,
    }
}

fn identity() -> LayerwiseIdentity {
    LayerwiseIdentity {
        history: stamp(),
        request: 77,
        evaluation: QueryRange { begin: EVAL_BEGIN, end: EVAL_END },
        query: QueryRange { begin: 9, end: 11 },
    }
}

/// Two opaque projected planes (Q + residual), in chunk storage order.
fn layout() -> QueryProjectionLayout {
    QueryProjectionLayout {
        format: "hipfire-qwen35-projected-q".into(),
        version: 1,
        planes: vec![
            QueryProjectionPlane { id: Q_PLANE, row_stride: Q_STRIDE },
            QueryProjectionPlane { id: R_PLANE, row_stride: R_STRIDE },
        ],
    }
}

/// Deterministic plane/row/byte-distinct oracle: `count` rows of one plane
/// starting at absolute token `begin`. Used both to feed append and to verify
/// readback, so any plane/offset/coverage error shows up as a byte mismatch.
fn plane_bytes(plane: u32, stride: u64, begin: u64, count: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity((stride * count) as usize);
    for r in 0..count {
        let t = begin + r;
        for b in 0..stride {
            v.push(
                (plane as u8)
                    .wrapping_mul(31)
                    .wrapping_add((t as u8).wrapping_mul(7))
                    .wrapping_add(b as u8),
            );
        }
    }
    v
}

/// Borrowed append payload for `[begin, begin+count)`: one buffer per plane.
fn payload(begin: u64, count: u64) -> (Vec<u8>, Vec<u8>) {
    (plane_bytes(Q_PLANE, Q_STRIDE, begin, count), plane_bytes(R_PLANE, R_STRIDE, begin, count))
}

fn append(
    tape: &mut QueryProjectionTape,
    id: &LayerwiseIdentity,
    begin: u64,
    count: u64,
) -> hipfire_kvmem::TapeChunk {
    let (q, r) = payload(begin, count);
    tape.append(
        id,
        MODEL_LAYER,
        begin,
        count,
        &[QueryPlaneBytes { id: Q_PLANE, data: &q }, QueryPlaneBytes { id: R_PLANE, data: &r }],
    )
    .expect("append")
}

#[test]
fn ram_tape_roundtrip_metadata_and_coverage() {
    let lay = layout();
    let id = identity();
    let bound = QueryProjectionTape::metadata_bound(&lay, 4).expect("metadata bound");
    assert!(bound > 0, "finite metadata bound must be positive");

    let activation = Storage::new_ram(TOTAL_BYTES).expect("activation account");
    let metadata = Storage::new_ram(bound).expect("metadata account");
    let mut tape =
        QueryProjectionTape::new_ram(&activation, &metadata, 4, &id, MODEL_LAYER, &lay)
            .expect("ram tape");

    // Admission state: full layer reserved up front, metadata reserved before
    // any payload, nothing covered yet.
    assert_eq!(tape.total_bytes(), TOTAL_BYTES);
    assert_eq!(tape.metadata_bytes(), bound);
    assert_eq!(tape.max_chunks(), 4);
    assert_eq!(tape.model_layer(), MODEL_LAYER);
    assert_eq!(tape.covered_end(), EVAL_BEGIN);
    assert_eq!(tape.chunk_count(), 0);
    assert!(!tape.complete());
    assert!(!tape.aborted());
    assert_eq!(activation.charged_bytes(), TOTAL_BYTES);
    assert_eq!(metadata.charged_bytes(), bound);
    assert_eq!(tape.reserved_bytes(), TOTAL_BYTES);

    // Finish before any coverage is rejected.
    assert!(tape.finish(&id, MODEL_LAYER).is_err(), "finish without coverage");

    // Chunk A [8, 10): a sealed chunk is readable before full completion.
    let ca = append(&mut tape, &id, 8, 2);
    assert_eq!(tape.covered_end(), 10);
    assert_eq!(tape.chunk_count(), 1);
    assert_eq!(tape.reserved_bytes(), TOTAL_BYTES - 2 * ROW_BYTES);
    assert_eq!(ca.begin(), 8);
    assert_eq!(ca.end(), 10);
    assert_eq!(ca.model_layer(), MODEL_LAYER);
    assert_eq!(ca.size() as u64, 2 * ROW_BYTES);
    let mut got = vec![0u8; (2 * Q_STRIDE) as usize];
    tape.read_rows(&id, MODEL_LAYER, Q_PLANE, 8, 2, &mut got).expect("read sealed A");
    assert_eq!(got, plane_bytes(Q_PLANE, Q_STRIDE, 8, 2));

    // A repeated/out-of-order begin is rejected (coverage must be contiguous).
    let (q, r) = payload(8, 1);
    assert!(
        tape.append(
            &id,
            MODEL_LAYER,
            8,
            1,
            &[QueryPlaneBytes { id: Q_PLANE, data: &q }, QueryPlaneBytes { id: R_PLANE, data: &r }],
        )
        .is_err(),
        "repeated begin"
    );

    // Chunk B [10, 12) completes coverage; the reservation is fully consumed.
    append(&mut tape, &id, 10, 2);
    assert_eq!(tape.covered_end(), EVAL_END);
    assert_eq!(tape.chunk_count(), 2);
    assert_eq!(tape.reserved_bytes(), 0);
    tape.finish(&id, MODEL_LAYER).expect("finish");
    assert!(tape.complete());
    assert_eq!(tape.reserved_bytes(), 0);
    assert_eq!(activation.charged_bytes(), TOTAL_BYTES); // sealed records stay charged

    // Full-layer byte-exact roundtrip, both planes.
    let mut q = vec![0u8; (EVAL_ROWS * Q_STRIDE) as usize];
    tape.read_rows(&id, MODEL_LAYER, Q_PLANE, EVAL_BEGIN, EVAL_ROWS, &mut q).expect("read all Q");
    assert_eq!(q, plane_bytes(Q_PLANE, Q_STRIDE, EVAL_BEGIN, EVAL_ROWS));
    let mut r = vec![0u8; (EVAL_ROWS * R_STRIDE) as usize];
    tape.read_rows(&id, MODEL_LAYER, R_PLANE, EVAL_BEGIN, EVAL_ROWS, &mut r).expect("read all R");
    assert_eq!(r, plane_bytes(R_PLANE, R_STRIDE, EVAL_BEGIN, EVAL_ROWS));

    // Cross-chunk read [9, 11) spans A and B.
    let mut mid = vec![0u8; (2 * Q_STRIDE) as usize];
    tape.read_rows(&id, MODEL_LAYER, Q_PLANE, 9, 2, &mut mid).expect("cross-chunk read");
    assert_eq!(mid, plane_bytes(Q_PLANE, Q_STRIDE, 9, 2));

    // A chunk source reads only its own rows.
    let c0 = tape.chunk(&id, MODEL_LAYER, 0).expect("chunk 0");
    let mut c0q = vec![0u8; (2 * Q_STRIDE) as usize];
    c0.read_rows(&id, MODEL_LAYER, Q_PLANE, 8, 2, &mut c0q).expect("chunk 0 read");
    assert_eq!(c0q, plane_bytes(Q_PLANE, Q_STRIDE, 8, 2));
    assert!(
        c0.read_rows(&id, MODEL_LAYER, Q_PLANE, 10, 1, &mut vec![0u8; Q_STRIDE as usize]).is_err(),
        "chunk 0 cannot read row 10"
    );

    // Fail-closed reads/appends after completion.
    assert!(tape.append(&id, MODEL_LAYER, 12, 0, &[]).is_err(), "append after complete");
    assert!(
        tape.read_rows(&id, MODEL_LAYER, 99, 8, 1, &mut vec![0u8; Q_STRIDE as usize]).is_err(),
        "unknown plane"
    );
    assert!(
        tape.read_rows(&id, MODEL_LAYER, Q_PLANE, 8, 5, &mut vec![0u8; (5 * Q_STRIDE) as usize])
            .is_err(),
        "read beyond covered rows"
    );
    assert!(
        tape.read_rows(&id, MODEL_LAYER, Q_PLANE, 8, 1, &mut vec![0u8; 15]).is_err(),
        "destination bytes != count*stride"
    );
    let mut foreign = id;
    foreign.request += 1;
    assert!(
        tape.read_rows(&foreign, MODEL_LAYER, Q_PLANE, 8, 1, &mut vec![0u8; Q_STRIDE as usize])
            .is_err(),
        "foreign request identity"
    );
    assert!(
        tape.read_rows(&id, MODEL_LAYER + 1, Q_PLANE, 8, 1, &mut vec![0u8; Q_STRIDE as usize])
            .is_err(),
        "foreign model layer"
    );
}

#[test]
fn metadata_and_chunk_limit_fail_closed() {
    let lay = layout();
    let id = identity();
    let bound = QueryProjectionTape::metadata_bound(&lay, 4).expect("bound");

    // The metadata account must fit the complete conservative bound; one byte
    // short fails closed BEFORE any payload is admitted.
    let activation = Storage::new_ram(TOTAL_BYTES).expect("activation");
    let tiny = Storage::new_ram(bound - 1).expect("tiny metadata");
    assert!(
        QueryProjectionTape::new_ram(&activation, &tiny, 4, &id, MODEL_LAYER, &lay).is_err(),
        "metadata bound must be reserved before payload"
    );
    assert_eq!(activation.charged_bytes(), 0, "failed admission charges no payload");

    // The activation account must fit the full layer reservation.
    let small_act = Storage::new_ram(TOTAL_BYTES - 1).expect("small activation");
    let meta = Storage::new_ram(bound).expect("metadata");
    assert!(
        QueryProjectionTape::new_ram(&small_act, &meta, 4, &id, MODEL_LAYER, &lay).is_err(),
        "activation must fit total layer bytes"
    );

    // max_chunks is enforced independently of row coverage: with a 4-row span
    // and max_chunks 2, two 1-row chunks exhaust the directory while coverage
    // is still incomplete, so a third chunk is rejected.
    let act2 = Storage::new_ram(TOTAL_BYTES).expect("activation 2");
    let meta2 = Storage::new_ram(bound).expect("metadata 2");
    let mut tape =
        QueryProjectionTape::new_ram(&act2, &meta2, 2, &id, MODEL_LAYER, &lay).expect("tape");
    append(&mut tape, &id, 8, 1);
    append(&mut tape, &id, 9, 1);
    assert_eq!(tape.chunk_count(), 2);
    assert_eq!(tape.covered_end(), 10);
    let (q, r) = payload(10, 1);
    assert!(
        tape.append(
            &id,
            MODEL_LAYER,
            10,
            1,
            &[QueryPlaneBytes { id: Q_PLANE, data: &q }, QueryPlaneBytes { id: R_PLANE, data: &r }],
        )
        .is_err(),
        "chunk count exceeds admitted metadata limit"
    );
    assert!(tape.finish(&id, MODEL_LAYER).is_err(), "finish before full coverage");

    // metadata_bound rejects degenerate limits/layouts.
    assert!(QueryProjectionTape::metadata_bound(&lay, 0).is_err(), "zero max_chunks");
    let mut empty_format = lay.clone();
    empty_format.format = String::new();
    assert!(QueryProjectionTape::metadata_bound(&empty_format, 4).is_err(), "empty format");
    let mut zero_version = lay.clone();
    zero_version.version = 0;
    assert!(QueryProjectionTape::metadata_bound(&zero_version, 4).is_err(), "zero version");
    let mut no_planes = lay.clone();
    no_planes.planes.clear();
    assert!(QueryProjectionTape::metadata_bound(&no_planes, 4).is_err(), "empty planes");

    // The constructor rejects duplicate plane ids, zero stride, and an invalid
    // identity (evaluation.begin must equal the history frontier).
    let act3 = Storage::new_ram(TOTAL_BYTES).expect("activation 3");
    let meta3 = Storage::new_ram(bound).expect("metadata 3");
    let mut dup = lay.clone();
    dup.planes[1].id = dup.planes[0].id;
    assert!(
        QueryProjectionTape::new_ram(&act3, &meta3, 4, &id, MODEL_LAYER, &dup).is_err(),
        "duplicate plane id"
    );
    let mut zero_stride = lay.clone();
    zero_stride.planes[0].row_stride = 0;
    assert!(
        QueryProjectionTape::new_ram(&act3, &meta3, 4, &id, MODEL_LAYER, &zero_stride).is_err(),
        "zero plane stride"
    );
    let mut bad_id = id;
    bad_id.history.frontier = EVAL_BEGIN + 1;
    assert!(
        QueryProjectionTape::new_ram(&act3, &meta3, 4, &bad_id, MODEL_LAYER, &lay).is_err(),
        "evaluation.begin != frontier"
    );
}

#[test]
fn disk_tape_tiering_and_source_survival() {
    let lay = layout();
    let id = identity();
    let bound = QueryProjectionTape::metadata_bound(&lay, 4).expect("bound");
    let dir = std::env::temp_dir().join(format!("hipfire-tape-disk-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");

    // The disk tier permits a layer larger than any RAM account: the whole span
    // is reserved against the SpillFile and written directly, no tape-owned RAM.
    let spill = SpillFile::new(&dir, TOTAL_BYTES).expect("spill account");
    assert_eq!(spill.capacity_bytes(), TOTAL_BYTES);
    let meta = Storage::new_ram(bound).expect("metadata");
    let mut tape =
        QueryProjectionTape::new_disk(&spill, &meta, 4, &id, MODEL_LAYER, &lay).expect("disk tape");
    assert_eq!(tape.total_bytes(), TOTAL_BYTES);
    assert_eq!(spill.charged_bytes(), TOTAL_BYTES);
    assert_eq!(tape.reserved_bytes(), TOTAL_BYTES);

    append(&mut tape, &id, 8, 2);
    append(&mut tape, &id, 10, 2);
    assert_eq!(tape.reserved_bytes(), 0);
    assert_eq!(spill.charged_bytes(), TOTAL_BYTES);
    tape.finish(&id, MODEL_LAYER).expect("finish");
    assert!(tape.complete());

    // Byte-exact readback from disk, both planes and a cross-chunk range.
    let mut q = vec![0u8; (EVAL_ROWS * Q_STRIDE) as usize];
    tape.read_rows(&id, MODEL_LAYER, Q_PLANE, EVAL_BEGIN, EVAL_ROWS, &mut q).expect("disk read Q");
    assert_eq!(q, plane_bytes(Q_PLANE, Q_STRIDE, EVAL_BEGIN, EVAL_ROWS));
    let mut mid = vec![0u8; (2 * R_STRIDE) as usize];
    tape.read_rows(&id, MODEL_LAYER, R_PLANE, 9, 2, &mut mid).expect("disk cross-chunk R");
    assert_eq!(mid, plane_bytes(R_PLANE, R_STRIDE, 9, 2));

    // Export a sealed chunk, then drop the tape AND both accounts. The disk
    // extent pins its own file range + quota, so the source stays readable.
    let survivor = tape.chunk(&id, MODEL_LAYER, 1).expect("chunk 1");
    drop(tape);
    drop(spill);
    drop(meta);
    let mut s = vec![0u8; (2 * Q_STRIDE) as usize];
    survivor.read_rows(&id, MODEL_LAYER, Q_PLANE, 10, 2, &mut s).expect("surviving disk source");
    assert_eq!(s, plane_bytes(Q_PLANE, Q_STRIDE, 10, 2));
    let mut s20 = vec![0u8; (2 * R_STRIDE) as usize];
    survivor.read_rows(&id, MODEL_LAYER, R_PLANE, 10, 2, &mut s20).expect("surviving disk R");
    assert_eq!(s20, plane_bytes(R_PLANE, R_STRIDE, 10, 2));
    drop(survivor);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn ram_source_survives_abort_and_account_drop() {
    let lay = layout();
    let id = identity();
    let bound = QueryProjectionTape::metadata_bound(&lay, 4).expect("bound");
    let activation = Storage::new_ram(TOTAL_BYTES).expect("activation");
    let metadata = Storage::new_ram(bound).expect("metadata");
    let mut tape =
        QueryProjectionTape::new_ram(&activation, &metadata, 4, &id, MODEL_LAYER, &lay)
            .expect("ram tape");
    append(&mut tape, &id, 8, 2);
    let survivor = tape.chunk(&id, MODEL_LAYER, 0).expect("chunk 0");

    // Abort releases the unused reservation and tape-owned references, but an
    // exported source keeps its own allocation alive.
    tape.abort();
    assert!(tape.aborted());
    assert!(!tape.complete());
    assert!(tape.finish(&id, MODEL_LAYER).is_err(), "aborted tape cannot finish");
    assert!(tape.read_rows(&id, MODEL_LAYER, Q_PLANE, 8, 1, &mut vec![0u8; Q_STRIDE as usize]).is_err());
    drop(tape);
    drop(activation);
    drop(metadata);

    let mut s = vec![0u8; (2 * Q_STRIDE) as usize];
    survivor.read_rows(&id, MODEL_LAYER, Q_PLANE, 8, 2, &mut s).expect("surviving RAM source");
    assert_eq!(s, plane_bytes(Q_PLANE, Q_STRIDE, 8, 2));
}

#[test]
fn reservation_allocate_and_release() {
    let storage = Storage::new_ram(100).expect("account");
    assert_eq!(storage.charged_bytes(), 0);
    let res = storage.reserve(60).expect("reserve");
    assert_eq!(res.remaining(), 60);
    assert_eq!(storage.charged_bytes(), 60); // reservations count like payload

    let data: Vec<u8> = (0..40u8).collect();
    let rec = res.allocate(&data).expect("allocate");
    assert_eq!(res.remaining(), 20);
    assert_eq!(rec.size(), 40);
    assert_eq!(storage.charged_bytes(), 60); // 40 record + 20 unused reservation
    rec.seal().expect("seal");
    let mut got = vec![0u8; 40];
    rec.read(0, &mut got).expect("read record");
    assert_eq!(got, data);

    // An over-allocation fails and retains the reservation unchanged.
    assert!(res.allocate(&vec![0u8; 21]).is_err(), "allocation exceeds reservation");
    assert_eq!(res.remaining(), 20);

    drop(rec);
    assert_eq!(storage.charged_bytes(), 20); // record released, unused reserved
    drop(res);
    assert_eq!(storage.charged_bytes(), 0); // unused capacity returned
}