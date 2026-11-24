// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-build contract for Query-span page-softmax scoring.

use hipfire_kvmem::{page_softmax_scores, GqaAggregation, RawStore, RawStoreConfig, Storage};

#[test]
fn page_softmax_near_tie_keeps_the_aligned_ranking() {
    const BLOCK_TOKENS: u32 = 4;
    const WIDTH: u32 = 8;
    let storage = Storage::new_ram(1 << 20).expect("storage");
    let raw = RawStore::new(
        RawStoreConfig {
            n_layer: 1,
            n_embd_k: WIDTH,
            n_embd_v: WIDTH,
            block_tokens: BLOCK_TOKENS,
            k_row_bytes: 0,
            k_gpu_row_bytes: 0,
            v_gpu_row_bytes: 0,
        },
        Some(&storage),
    )
    .expect("raw store");

    let full = vec![1.0f32; (BLOCK_TOKENS * WIDTH) as usize];
    let near = vec![0.999f32; (BLOCK_TOKENS * WIDTH) as usize];
    raw.write_layer_mean_k(0, BLOCK_TOKENS, 0, &full)
        .expect("first block");
    raw.write_layer_mean_k(BLOCK_TOKENS, BLOCK_TOKENS, 0, &near)
        .expect("near-tie block");

    let query = vec![1.0f32; WIDTH as usize];
    let rows = [1u32];
    let scores = page_softmax_scores(
        &raw,
        2,
        u64::from(2 * BLOCK_TOKENS),
        BLOCK_TOKENS,
        2,
        0,
        0,
        &query,
        &rows,
        1,
        1,
        1,
        1,
        WIDTH,
        1.0 / (WIDTH as f32).sqrt(),
        GqaAggregation::Mean,
    )
    .expect("page-softmax scores");

    assert_eq!(scores.len(), 2);
    assert!(
        scores[0] > scores[1],
        "near-tie ranking flipped: {scores:?}"
    );
    assert!((scores[0] + scores[1] - 1.0).abs() < 1e-12);
}
