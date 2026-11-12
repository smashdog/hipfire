// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only smoke test for the KVMem C ABI adapter: no GPU, no model.
//!
//! Exercises the exact contracts the serve controller relies on: mean-K
//! statistics, packed K/V spill + readback, block selection budgets, the
//! retrieval scorer and truncation checkpoints.

use hipfire_kvmem::{
    mean_key_retrieval_scores, GqaAggregation, KvStore, KvStoreConfig, RawStore, RawStoreConfig,
    Storage,
};

const BLOCK: u32 = 4;
const K_WIDTH: usize = 8;
const ROW_BYTES: usize = 36; // opaque packed row; contents only round-trip
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

fn kv_config(budget_tokens: u32) -> KvStoreConfig {
    KvStoreConfig {
        block_tokens: BLOCK,
        select_budget: budget_tokens,
        sink_blocks: 1,
        recent_blocks: 1,
        ..KvStoreConfig::default()
    }
}

fn pack_row(seed: u8) -> [u8; ROW_BYTES] {
    let mut row = [0u8; ROW_BYTES];
    for (i, b) in row.iter_mut().enumerate() {
        *b = seed.wrapping_add(i as u8).wrapping_mul(3);
    }
    row
}

#[test]
fn statistics_packed_roundtrip_and_selection() {
    let storage = Storage::new_ram(1 << 30).expect("storage");
    assert_eq!(storage.charged_bytes(), 0);

    let raw = RawStore::new(raw_config(), Some(&storage)).expect("raw store");

    // ── Mean-K statistics: 4 tokens of layer 0 ──
    let mut k = Vec::with_capacity(BLOCK as usize * K_WIDTH);
    for t in 0..BLOCK as usize {
        for d in 0..K_WIDTH {
            k.push((t * K_WIDTH + d) as f32);
        }
    }
    raw.write_layer_mean_k(0, BLOCK, 0, &k).expect("mean k");
    assert_eq!(raw.mean_k_tokens(0, 0), BLOCK);
    let mut mean = vec![f32::NAN; K_WIDTH];
    raw.mean_k(0, 0, &mut mean);
    for (d, m) in mean.iter().enumerate() {
        let expect = (0..BLOCK as usize).map(|t| (t * K_WIDTH + d) as f32).sum::<f32>()
            / BLOCK as f32;
        assert!((m - expect).abs() < 1e-6, "dim {d}: {m} vs {expect}");
    }

    // ── Packed Q8 spill + readback, all layers of block 1 ──
    for il in 0..LAYERS {
        let mut rows = Vec::new();
        for t in 0..BLOCK {
            rows.extend_from_slice(&pack_row((il as u8 + 1) * 10 + t as u8));
        }
        raw.write_layer_k_gpu(BLOCK, BLOCK, il, &rows).expect("spill k");
        raw.write_layer_v_gpu(BLOCK, BLOCK, il, &rows).expect("spill v");
    }
    for il in 0..LAYERS {
        assert!(raw.has_k_gpu(1, il, BLOCK));
        assert!(raw.has_v_gpu(1, il, BLOCK));
        let mut out = vec![0u8; BLOCK as usize * ROW_BYTES];
        assert!(raw.copy_k_gpu(1, il, &mut out, BLOCK));
        for t in 0..BLOCK as usize {
            assert_eq!(&out[t * ROW_BYTES..(t + 1) * ROW_BYTES],
                       &pack_row((il as u8 + 1) * 10 + t as u8));
        }
    }

    // ── Block repository: 12 tokens = 3 blocks ──
    let kv = KvStore::new(&kv_config(2 * BLOCK)).expect("kv store");
    kv.register_append(12).expect("append");
    assert_eq!(kv.block_count(), 3);
    assert_eq!(kv.total_tokens(), 12);

    // Sink (block 0) + newest block under a 2-block budget.
    let picked = kv.pick_prefill_pressure_blocks(&[]).expect("pressure pick");
    assert_eq!(picked, vec![0, 2], "pressure selection keeps sink + newest");

    // ── Retrieval scorer: query aligned with block 1's mean key ──
    // Put layer-0/1 statistics on every block so has_block() is true.
    for block in 0..3u32 {
        for il in 0..LAYERS {
            let pos = block * BLOCK;
            let rows: Vec<f32> = (0..BLOCK as usize * K_WIDTH)
                .map(|i| if block == 1 { 1.0 } else { 0.0 } + i as f32 * 0.0)
                .collect();
            raw.write_layer_mean_k(pos, BLOCK, il, &rows).expect("mean k");
        }
    }
    let sums = vec![1.0f32; LAYERS as usize * K_WIDTH];
    let counts = vec![BLOCK; LAYERS as usize];
    let scores = mean_key_retrieval_scores(
        &raw,
        3,
        &sums,
        &counts,
        LAYERS,
        K_WIDTH as u32,
        1,
        1,
        GqaAggregation::Mean,
    )
    .expect("scores");
    assert_eq!(scores.len(), 3);
    assert!(scores[1] > scores[2], "aligned block should win: {scores:?}");

    // Bad geometry is an error, not a silent zero.
    assert!(mean_key_retrieval_scores(
        &raw,
        3,
        &sums,
        &counts,
        LAYERS,
        K_WIDTH as u32 + 1,
        1,
        1,
        GqaAggregation::Mean,
    )
    .is_err());

    // ── Truncation: drop 2 partial tokens of the trailing block ──
    assert_eq!(raw.n_tokens(1), BLOCK);
    raw.truncate_to(10).expect("truncate raw");
    assert_eq!(raw.n_tokens(1), BLOCK); // block 1 untouched (full)
    kv.truncate_to(10).expect("truncate kv");
    assert_eq!(kv.total_tokens(), 10);
    assert_eq!(kv.block_count(), 3);

    // Checkpoint of a partial tail: block 2 keeps only 2 of its 4 written
    // tokens, so truncate_to cleared its statistics...
    let stride = 1 + K_WIDTH;
    let mut ckpt = raw
        .mean_checkpoint(10)
        .expect("checkpoint query")
        .expect("block-aligned position still has a state vector");
    assert_eq!(ckpt.len(), LAYERS as usize * stride);
    for il in 0..LAYERS as usize {
        assert_eq!(ckpt[il * stride], 0.0, "tail statistics must be cleared");
    }

    // ...and restoring a saved checkpoint reinstates the partial tail.
    // Checkpoint layout is count + SUM per layer, so 1.0 over 2 tokens = 0.5.
    for il in 0..LAYERS as usize {
        ckpt[il * stride] = 2.0;
        for d in 0..K_WIDTH {
            ckpt[il * stride + 1 + d] = 1.0;
        }
    }
    raw.restore_mean_checkpoint(10, &ckpt).expect("restore");
    assert_eq!(raw.mean_k_tokens(2, 0), 2);
    let mut restored = vec![f32::NAN; K_WIDTH];
    raw.mean_k(2, 0, &mut restored);
    assert!(restored.iter().all(|v| (*v - 0.5).abs() < 1e-6));
}
