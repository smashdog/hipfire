// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only smoke test for the layerwise native payload foundation: no GPU, no
//! model. Exercises the exact building blocks the packed-page driver relies on:
//! the Q8 PayloadLayout adapter, MemoryDescriptor validation, per-layer
//! NativeKvLayerRegions compact-extent reads from a sealed host record, the
//! NativeKvArchive write-through store and the raw-KV export/import bridge.

use hipfire_kvmem::{
    BackendIdentity, BlockDescriptor, HostRecord, KvRegions, MemoryCapabilities, MemoryDescriptor,
    NativeArchive, NativePayload, PayloadLayout, PayloadPlane, PayloadPlaneKind, RawStore,
    RawStoreConfig, SessionIdentity, Storage,
};

const BLOCK: u32 = 4;
const ROW_BYTES: usize = 36; // opaque packed Q8 row; contents only round-trip
const LAYERS: u32 = 2;
const RECORD_BYTES: usize = ROW_BYTES * BLOCK as usize * LAYERS as usize * 2; // K+V per layer

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

fn block0() -> BlockDescriptor {
    BlockDescriptor {
        session: session(),
        id: 0,
        original_start: 0,
        valid_tokens: BLOCK,
        content_version: 1,
        statistics_version: 1,
    }
}

/// Deterministic packed row for (layer, kind, token): kind 0 = K, 1 = V.
fn pack_row(layer: u32, kind: u32, token: u32) -> [u8; ROW_BYTES] {
    let mut row = [0u8; ROW_BYTES];
    let seed = (layer * 100 + kind * 10 + token) as u8;
    for (i, b) in row.iter_mut().enumerate() {
        *b = seed.wrapping_add(i as u8).wrapping_mul(7);
    }
    row
}

/// Write packed K/V for block 0, every layer, into the raw store.
fn feed_block0(raw: &RawStore) {
    for il in 0..LAYERS {
        let mut k = Vec::new();
        let mut v = Vec::new();
        for t in 0..BLOCK {
            k.extend_from_slice(&pack_row(il, 0, t));
            v.extend_from_slice(&pack_row(il, 1, t));
        }
        raw.write_layer_k_gpu(0, BLOCK, il, &k).expect("feed k");
        raw.write_layer_v_gpu(0, BLOCK, il, &v).expect("feed v");
    }
}

#[test]
fn layout_adapter_and_descriptor() {
    let layout = PayloadLayout::from_raw(&raw_config(), "hipfire-q8-packed-v1", &[0, 1])
        .expect("layout from raw");
    assert_eq!(layout.block_tokens(), BLOCK);
    assert_eq!(layout.record_bytes(), RECORD_BYTES as u64);
    assert_eq!(layout.plane_count(), LAYERS * 2);

    // Planes are layer-major {Key,Value}, contiguous, stride = packed row bytes.
    let mut expect_offset = 0u64;
    for il in 0..LAYERS {
        for kind in [PayloadPlaneKind::Key, PayloadPlaneKind::Value] {
            let p = layout.plane(il * 2 + kind as u32).expect("plane");
            assert_eq!(p.layer, il);
            assert_eq!(p.kind, kind as u32);
            assert_eq!(p.offset, expect_offset);
            assert_eq!(p.bytes, (ROW_BYTES * BLOCK as usize) as u64);
            assert_eq!(p.token_stride, ROW_BYTES as u64);
            assert_eq!(p.alignment, 1);
            expect_offset += p.bytes;
        }
    }
    assert_eq!(expect_offset, RECORD_BYTES as u64);

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
    let _desc = MemoryDescriptor::new(&backend(), &layout, &cap).expect("descriptor");

    // An empty attention-layer list is rejected, not silently zero-sized.
    assert!(PayloadLayout::from_raw(&raw_config(), "fmt", &[]).is_err());
}

#[test]
fn export_regions_read_and_import_roundtrip() {
    let storage = Storage::new_ram(1 << 30).expect("storage");
    let raw = RawStore::new(raw_config(), Some(&storage)).expect("raw store");
    feed_block0(&raw);

    let layout = PayloadLayout::from_raw(&raw_config(), "hipfire-q8-packed-v1", &[0, 1])
        .expect("layout");

    // Export gathers all layers' packed K/V into one canonical record.
    let payload = NativePayload::export(&raw, &backend(), &layout, &block0()).expect("export");
    assert_eq!(payload.size(), RECORD_BYTES);
    assert_eq!(payload.block().expect("block"), block0());

    // The exported bytes are exactly the layer-major {K,V} packed rows.
    let bytes = payload.bytes();
    let mut expect = Vec::new();
    for il in 0..LAYERS {
        for t in 0..BLOCK {
            expect.extend_from_slice(&pack_row(il, 0, t));
        }
        for t in 0..BLOCK {
            expect.extend_from_slice(&pack_row(il, 1, t));
        }
    }
    assert_eq!(bytes, expect.as_slice());

    // Seal the canonical record as a host source, then read layer 0's compact
    // extent (its K plane then V plane) out of the full multi-layer record.
    let record = HostRecord::allocate(&storage, bytes).expect("allocate record");
    record.seal().expect("seal");
    assert_eq!(record.size(), RECORD_BYTES);
    let source = record.source().expect("source");
    assert_eq!(source.size(), RECORD_BYTES);

    let regions = KvRegions::new(&layout, 0, None).expect("regions layer 0");
    let plane_bytes = (ROW_BYTES * BLOCK as usize) as u64;
    assert_eq!(regions.region_count(), 2);
    assert_eq!(regions.plane_bytes(), 2 * plane_bytes);
    assert_eq!(regions.packed_bytes(), 2 * plane_bytes); // alignment 1, no padding
    assert_eq!(regions.record_bytes(), RECORD_BYTES as u64);
    let (k_plane, k_off) = regions.region(0).expect("region 0");
    let (v_plane, v_off) = regions.region(1).expect("region 1");
    assert_eq!(k_plane.kind, PayloadPlaneKind::Key as u32);
    assert_eq!(v_plane.kind, PayloadPlaneKind::Value as u32);
    assert_eq!(k_off, 0);
    assert_eq!(v_off, plane_bytes);

    let mut staging = vec![0u8; regions.packed_bytes() as usize];
    // Bounded reads that deliberately cross a plane boundary, like the core test.
    let split = 13usize;
    regions.read(&source, 0, &mut staging[..split], 7).expect("read part 1");
    regions.read(&source, split as u64, &mut staging[split..], 7).expect("read part 2");
    let mut expect_layer0 = Vec::new();
    for t in 0..BLOCK {
        expect_layer0.extend_from_slice(&pack_row(0, 0, t));
    }
    for t in 0..BLOCK {
        expect_layer0.extend_from_slice(&pack_row(0, 1, t));
    }
    assert_eq!(staging, expect_layer0);

    // Layer 1 reads its own stripe from the same canonical record.
    let regions1 = KvRegions::new(&layout, 1, None).expect("regions layer 1");
    let mut staging1 = vec![0u8; regions1.packed_bytes() as usize];
    regions1.read(&source, 0, &mut staging1, 64 << 20).expect("read layer 1");
    let mut expect_layer1 = Vec::new();
    for t in 0..BLOCK {
        expect_layer1.extend_from_slice(&pack_row(1, 0, t));
    }
    for t in 0..BLOCK {
        expect_layer1.extend_from_slice(&pack_row(1, 1, t));
    }
    assert_eq!(staging1, expect_layer1);

    // Import the canonical record into a FRESH raw store and verify packed
    // readback per layer/plane (statistics are untouched by import).
    let raw2 = RawStore::new(raw_config(), Some(&storage)).expect("raw store 2");
    payload.import(&raw2, &backend(), &layout).expect("import");
    for il in 0..LAYERS {
        assert!(raw2.has_k_gpu(0, il, BLOCK));
        assert!(raw2.has_v_gpu(0, il, BLOCK));
        let mut k = vec![0u8; BLOCK as usize * ROW_BYTES];
        let mut v = vec![0u8; BLOCK as usize * ROW_BYTES];
        assert!(raw2.copy_k_gpu(0, il, &mut k, BLOCK));
        assert!(raw2.copy_v_gpu(0, il, &mut v, BLOCK));
        for t in 0..BLOCK as usize {
            assert_eq!(&k[t * ROW_BYTES..(t + 1) * ROW_BYTES], &pack_row(il, 0, t as u32));
            assert_eq!(&v[t * ROW_BYTES..(t + 1) * ROW_BYTES], &pack_row(il, 1, t as u32));
        }
    }
}

#[test]
fn archive_write_through_and_stale_rejection() {
    let storage = Storage::new_ram(1 << 30).expect("storage");
    let raw = RawStore::new(raw_config(), Some(&storage)).expect("raw store");
    feed_block0(&raw);
    let layout = PayloadLayout::from_raw(&raw_config(), "hipfire-q8-packed-v1", &[0, 1])
        .expect("layout");
    let payload = NativePayload::export(&raw, &backend(), &layout, &block0()).expect("export");

    let archive =
        NativeArchive::new(&backend(), session(), &layout, (RECORD_BYTES * 4) as u64)
            .expect("archive");
    assert_eq!(archive.bytes(), 0);
    archive.put(&payload).expect("put");
    assert_eq!(archive.bytes(), RECORD_BYTES as u64);

    // get returns a new caller-owned handle with identical bytes/identity.
    let fetched = archive.get(&block0()).expect("get");
    assert_eq!(fetched.bytes(), payload.bytes());
    assert_eq!(fetched.block().expect("block"), block0());

    // Re-putting the identical version is idempotent (no double charge).
    archive.put(&payload).expect("idempotent put");
    assert_eq!(archive.bytes(), RECORD_BYTES as u64);

    // Upgrade to a higher content_version (replaces in place, no double charge).
    // content_version 0 is invalid at construction, so genuine staleness must be
    // exercised against a stored version above the minimum valid version 1.
    let mut upgraded = block0();
    upgraded.content_version = 2;
    let newer = NativePayload::create(&backend(), &layout, &upgraded, payload.bytes())
        .expect("upgraded payload");
    archive.put(&newer).expect("upgrade put");
    assert_eq!(archive.bytes(), RECORD_BYTES as u64);

    // A stale (lower content_version) write is rejected.
    let stale = NativePayload::create(&backend(), &layout, &block0(), payload.bytes())
        .expect("stale payload");
    assert!(archive.put(&stale).is_err(), "stale write must be rejected");

    // A foreign backend identity is rejected.
    let mut foreign = backend();
    foreign.model = "other-model".into();
    let foreign_payload =
        NativePayload::create(&foreign, &layout, &upgraded, payload.bytes()).expect("foreign");
    assert!(archive.put(&foreign_payload).is_err(), "foreign write must be rejected");

    // A mismatched get identity is rejected (wrong frontier, same version).
    let mut wrong = block0();
    wrong.content_version = 2;
    wrong.valid_tokens = BLOCK - 1;
    assert!(archive.get(&wrong).is_err());

    archive.erase(0);
    assert_eq!(archive.bytes(), 0);
    assert!(archive.get(&upgraded).is_err(), "erased record is missing");
}

#[test]
fn generic_layout_create_matches_adapter() {
    // Hand-build the same two-layer Q8 layout via the explicit plane builder.
    let plane_bytes = (ROW_BYTES * BLOCK as usize) as u64;
    let mut planes = Vec::new();
    let mut offset = 0u64;
    for il in 0..LAYERS {
        for kind in [PayloadPlaneKind::Key, PayloadPlaneKind::Value] {
            planes.push(PayloadPlane {
                layer: il,
                kind: kind as u32,
                offset,
                bytes: plane_bytes,
                alignment: 1,
                token_stride: ROW_BYTES as u64,
                rank: 0,
                head_begin: 0,
                head_count: 0,
            });
            offset += plane_bytes;
        }
    }
    let layout = PayloadLayout::create(
        "hipfire-q8-packed-v1",
        1,
        BLOCK,
        offset,
        1,
        &planes,
    )
    .expect("generic layout");
    assert_eq!(layout.record_bytes(), RECORD_BYTES as u64);
    assert_eq!(layout.plane_count(), LAYERS * 2);
    assert_eq!(layout.plane(0).expect("plane 0"), planes[0]);
    assert!(layout.plane(LAYERS * 2).is_err(), "out-of-range plane index");
}