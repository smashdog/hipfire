// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! P4.3 H1 host-only smoke: KVMem-native prefix fork / packed exchange C ABI.
//! No GPU, no model. Exercises fork_prefix (sealed shared packed sources),
//! fork_mean_prefix (mean-K only), and export_packed/import_packed round-trip.

use hipfire_kvmem::{
    prefix_anchor, prefix_restore_limit, PackedSelection, PrefixRange, RawStore, RawStoreConfig,
    Storage,
};

const BLOCK: u32 = 4;
const K_WIDTH: usize = 8;
const ROW_BYTES: usize = 36;
const LAYERS: u32 = 2;

fn raw_config() -> RawStoreConfig {
    RawStoreConfig {
        n_layer: LAYERS,
        n_embd_k: K_WIDTH as u32,
        n_embd_v: K_WIDTH as u32,
        block_tokens: BLOCK,
        k_row_bytes: 0,
        k_gpu_row_bytes: ROW_BYTES as u64,
        v_gpu_row_bytes: ROW_BYTES as u64,
    }
}

fn pack_row(seed: u8) -> Vec<u8> {
    (0..ROW_BYTES).map(|i| seed.wrapping_add(i as u8).wrapping_mul(3)).collect()
}

/// Fill `n_blocks` packed blocks (all layers) + mean-K for block 0.
fn fill(raw: &RawStore, n_blocks: u32) {
    for b in 0..n_blocks {
        for il in 0..LAYERS {
            let mut rows = Vec::new();
            for t in 0..BLOCK {
                rows.extend_from_slice(&pack_row((b * 30 + il * 10 + t) as u8));
            }
            raw.write_layer_k_gpu(b * BLOCK, BLOCK, il, &rows).expect("k");
            raw.write_layer_v_gpu(b * BLOCK, BLOCK, il, &rows).expect("v");
        }
    }
    let k: Vec<f32> = (0..BLOCK as usize * K_WIDTH).map(|i| i as f32).collect();
    raw.write_layer_mean_k(0, BLOCK, 0, &k).expect("mean k");
}

#[test]
fn fork_prefix_shares_sealed_packed_sources() {
    let storage = Storage::new_ram(1 << 30).expect("storage");
    let raw = RawStore::new(raw_config(), Some(&storage)).expect("raw");
    fill(&raw, 3);

    // Fork the first 2 blocks (token_pos = 2*BLOCK). The branch shares the
    // sealed packed sources; raw rows/stats are private copies.
    let branch = raw.fork_prefix(2 * BLOCK).expect("fork_prefix");
    for b in 0..2 {
        for il in 0..LAYERS {
            assert!(branch.has_k_gpu(b, il, BLOCK), "branch k block {b} layer {il}");
            assert!(branch.has_v_gpu(b, il, BLOCK), "branch v block {b} layer {il}");
            let mut src = vec![0u8; BLOCK as usize * ROW_BYTES];
            let mut dst = vec![0u8; BLOCK as usize * ROW_BYTES];
            assert!(raw.copy_k_gpu(b, il, &mut src, BLOCK));
            assert!(branch.copy_k_gpu(b, il, &mut dst, BLOCK));
            assert_eq!(src, dst, "shared packed K bytes block {b} layer {il}");
        }
    }
    // The branch is truncated to the fork prefix: block 2 is gone.
    assert!(!branch.has_k_gpu(2, 0, BLOCK), "branch must not retain block 2");
    // The source keeps all 3 blocks (fork does not mutate the source).
    assert!(raw.has_k_gpu(2, 0, BLOCK), "source retains block 2");
}

#[test]
fn fork_mean_prefix_copies_index_only() {
    let storage = Storage::new_ram(1 << 30).expect("storage");
    let raw = RawStore::new(raw_config(), Some(&storage)).expect("raw");
    // mean-K-only store (NO packed payload): fork_mean_prefix refuses to discard
    // raw KV, so the source must carry only the mean-K index.
    let k: Vec<f32> = (0..BLOCK as usize * K_WIDTH).map(|i| i as f32).collect();
    raw.write_layer_mean_k(0, BLOCK, 0, &k).expect("mean k");

    let branch = raw.fork_mean_prefix(BLOCK).expect("fork_mean_prefix");
    // Mean-K index copied for block 0.
    assert_eq!(branch.mean_k_tokens(0, 0), BLOCK, "mean-K index forked");
    let mut src = vec![f32::NAN; K_WIDTH];
    let mut dst = vec![f32::NAN; K_WIDTH];
    raw.mean_k(0, 0, &mut src);
    branch.mean_k(0, 0, &mut dst);
    assert_eq!(src, dst, "mean-K bytes match");
    // No packed KV copied (payload lives in a separately shared archive).
    assert!(!branch.has_k_gpu(0, 0, BLOCK), "mean fork must not copy packed K");
}

#[test]
fn export_import_packed_roundtrip() {
    let storage = Storage::new_ram(1 << 30).expect("storage");
    let raw = RawStore::new(raw_config(), Some(&storage)).expect("raw");
    fill(&raw, 2);

    // Export block 0 (all layers, full BLOCK tokens).
    let selections: Vec<PackedSelection> = (0..LAYERS)
        .map(|il| PackedSelection { block: 0, layer: il, valid_tokens: BLOCK })
        .collect();
    let snap = raw.export_packed(&selections, usize::MAX).expect("export_packed");
    assert_eq!(snap.entry_count(), LAYERS as usize, "one entry per layer plane pair");

    // Import into a fresh store: packed bindings replaced, readback matches.
    let dst = RawStore::new(raw_config(), Some(&storage)).expect("dst");
    dst.import_packed(snap, usize::MAX).expect("import_packed");
    for il in 0..LAYERS {
        assert!(dst.has_k_gpu(0, il, BLOCK), "imported k layer {il}");
        let mut src = vec![0u8; BLOCK as usize * ROW_BYTES];
        let mut out = vec![0u8; BLOCK as usize * ROW_BYTES];
        assert!(raw.copy_k_gpu(0, il, &mut src, BLOCK));
        assert!(dst.copy_k_gpu(0, il, &mut out, BLOCK));
        assert_eq!(src, out, "imported packed K bytes layer {il}");
    }
}

#[test]
fn fork_prefix_rejects_overlong_prefix() {
    let storage = Storage::new_ram(1 << 30).expect("storage");
    let raw = RawStore::new(raw_config(), Some(&storage)).expect("raw");
    fill(&raw, 1);
    // Forking beyond the retained prefix fails closed (core logic_error).
    assert!(raw.fork_prefix(5 * BLOCK).is_err(), "overlong fork must fail");
}

#[test]
fn prefix_anchor_clamps_and_floors_groups() {
    // Plain clamp: lcp inside [base, query] passes through.
    assert_eq!(prefix_anchor(0, 100, 200, &[]).expect("anchor"), 100);
    assert_eq!(prefix_anchor(64, 10, 200, &[]).expect("anchor"), 64, "clamp to base");
    assert_eq!(prefix_anchor(0, 500, 200, &[]).expect("anchor"), 200, "clamp to query");
    // Indivisible group floors a mid-group anchor to the group boundary.
    let groups = [PrefixRange { begin: 80, end: 120 }];
    assert_eq!(prefix_anchor(0, 100, 200, &groups).expect("anchor"), 80);
    // Repeated flooring across overlapping groups.
    let groups = [
        PrefixRange { begin: 80, end: 120 },
        PrefixRange { begin: 60, end: 90 },
    ];
    assert_eq!(prefix_anchor(0, 100, 200, &groups).expect("anchor"), 60);
    // An anchor exactly at a group boundary is not split.
    let groups = [PrefixRange { begin: 80, end: 120 }];
    assert_eq!(prefix_anchor(0, 80, 200, &groups).expect("anchor"), 80);
    assert_eq!(prefix_anchor(0, 120, 200, &groups).expect("anchor"), 120);
    // query < base fails closed.
    assert!(prefix_anchor(300, 100, 200, &[]).is_err(), "query precedes base");
    // A group that would be split below base fails closed.
    let groups = [PrefixRange { begin: 10, end: 90 }];
    assert!(prefix_anchor(50, 80, 200, &groups).is_err(), "group split below base");
    // Invalid group (begin > end) fails closed.
    let groups = [PrefixRange { begin: 90, end: 10 }];
    assert!(prefix_anchor(0, 80, 200, &groups).is_err(), "invalid group");
}

#[test]
fn prefix_restore_limit_arithmetic() {
    assert_eq!(prefix_restore_limit(100, 200, 300, false), 100);
    assert_eq!(prefix_restore_limit(300, 200, 300, false), 200);
    // needs_logits reserves the last evaluated position.
    assert_eq!(prefix_restore_limit(300, 300, 300, true), 299);
    assert_eq!(prefix_restore_limit(100, 300, 300, true), 100);
    // eval_end 0 stays 0 (no subtraction underflow).
    assert_eq!(prefix_restore_limit(0, 0, 0, true), 0);
}
#[test]
fn capture_snapshot_covers_the_sealed_packed_grid() {
    let storage = Storage::new_ram(1 << 30).expect("storage");
    let raw = RawStore::new(raw_config(), Some(&storage)).expect("raw");
    fill(&raw, 3);
    assert!(raw.has_packed(2));
    // The freeze contract seals the records via fork_prefix; capture that
    // canonical prefix (block-major, layer-major, K then V).
    let canonical = raw.fork_prefix(2 * BLOCK).expect("fork");
    let snap = canonical.capture_snapshot(2, None).expect("capture");
    assert!(snap.ready());
    assert_eq!(snap.block_count(), (2 * LAYERS * 2) as usize);
    assert_eq!(snap.span_count(), snap.block_count());
    assert!(snap.covers_packed(2, LAYERS));
    assert!(!snap.covers_packed(2, LAYERS + 1));
    let per_record = BLOCK as u64 * ROW_BYTES as u64;
    assert_eq!(snap.bytes(), (2 * LAYERS) as u64 * 2 * per_record);
    assert!(snap.metadata_bytes() > 0);
    assert_eq!(snap.missing_bytes(), 0);
    // Origin inheritance: re-capturing the same sealed records reuses the
    // proven origins (same grid and bytes, no payload copy).
    let snap2 = canonical.capture_snapshot(2, Some(&snap)).expect("capture 2");
    assert!(snap2.covers_packed(2, LAYERS));
    assert_eq!(snap2.bytes(), snap.bytes());
}

#[test]
fn snapshot_capture_refuses_a_store_without_packed_records() {
    let storage = Storage::new_ram(1 << 20).expect("storage");
    let raw = RawStore::new(raw_config(), Some(&storage)).expect("raw");
    let k: Vec<f32> = (0..BLOCK as usize * K_WIDTH).map(|i| i as f32).collect();
    raw.write_layer_mean_k(0, BLOCK, 0, &k).expect("mean k");
    assert!(!raw.has_packed(1));
    assert!(raw.capture_snapshot(1, None).is_err());
}

#[test]
fn prefix_pack_roundtrips_packed_bytes_mean_and_metadata() {
    let dir = std::env::temp_dir().join(format!("kvmem-pack-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("p.hkpack");
    let a = Storage::new_ram(1 << 30).expect("a");
    let src = RawStore::new(raw_config(), Some(&a)).expect("src");
    fill(&src, 3);
    let canonical = src.fork_prefix(2 * BLOCK).expect("fork");
    let meta = b"proof:frontier=256,layout=7,group=42".to_vec();
    canonical.write_prefix_pack(2, &path, &meta).expect("write pack");
    let info = RawStore::prefix_pack_info(&path).expect("info");
    assert_eq!(info.blocks, 2);
    assert_eq!(info.n_layer, LAYERS);
    assert_eq!(info.block_tokens, BLOCK);
    assert_eq!(info.meta_bytes, meta.len() as u64);
    assert!(info.file_bytes > 0);

    let b = Storage::new_ram(1 << 30).expect("b");
    let dst = RawStore::new(raw_config(), Some(&b)).expect("dst");
    let loaded = dst.read_prefix_pack(&path).expect("read pack");
    assert_eq!(loaded, meta);
    assert!(!dst.has_k_gpu(2, 0, BLOCK), "only two blocks persisted");
    for blk in 0..2 {
        for il in 0..LAYERS {
            let mut x = vec![0u8; BLOCK as usize * ROW_BYTES];
            let mut y = vec![0u8; BLOCK as usize * ROW_BYTES];
            assert!(canonical.copy_k_gpu(blk, il, &mut x, BLOCK));
            assert!(dst.copy_k_gpu(blk, il, &mut y, BLOCK));
            assert_eq!(x, y, "K block {blk} layer {il}");
            assert!(canonical.copy_v_gpu(blk, il, &mut x, BLOCK));
            assert!(dst.copy_v_gpu(blk, il, &mut y, BLOCK));
            assert_eq!(x, y, "V block {blk} layer {il}");
            let mut m0 = vec![0f32; K_WIDTH];
            let mut m1 = vec![0f32; K_WIDTH];
            canonical.mean_k(blk, il, &mut m0);
            dst.mean_k(blk, il, &mut m1);
            assert_eq!(m0, m1, "Mean-K block {blk} layer {il}");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
