// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compile-time guards for the two configuration structs copied into the C ABI.
//!
//! The C++ side asserts the same numbers in
//! `backends/hipfire/hipfire_kvmem_shim.h`. Keep all three in sync when the
//! shim's `HK_ABI_VERSION` changes.

use std::mem::{align_of, offset_of, size_of};

use super::{KvStoreConfig, RawStoreConfig};

const _: () = {
    assert!(size_of::<RawStoreConfig>() == 40);
    assert!(align_of::<RawStoreConfig>() == 8);
    assert!(offset_of!(RawStoreConfig, n_layer) == 0);
    assert!(offset_of!(RawStoreConfig, n_embd_k) == 4);
    assert!(offset_of!(RawStoreConfig, n_embd_v) == 8);
    assert!(offset_of!(RawStoreConfig, block_tokens) == 12);
    assert!(offset_of!(RawStoreConfig, k_row_bytes) == 16);
    assert!(offset_of!(RawStoreConfig, k_gpu_row_bytes) == 24);
    assert!(offset_of!(RawStoreConfig, v_gpu_row_bytes) == 32);
};

const _: () = {
    assert!(size_of::<KvStoreConfig>() == 112);
    assert!(align_of::<KvStoreConfig>() == 8);
    assert!(offset_of!(KvStoreConfig, block_tokens) == 0);
    assert!(offset_of!(KvStoreConfig, select_budget) == 4);
    assert!(offset_of!(KvStoreConfig, prefill_budget) == 8);
    assert!(offset_of!(KvStoreConfig, gen_budget) == 12);
    assert!(offset_of!(KvStoreConfig, sink_blocks) == 16);
    assert!(offset_of!(KvStoreConfig, recent_blocks) == 20);
    assert!(offset_of!(KvStoreConfig, select_method) == 24);
    assert!(offset_of!(KvStoreConfig, retrieval_method) == 28);
    assert!(offset_of!(KvStoreConfig, update_mode) == 32);
    assert!(offset_of!(KvStoreConfig, lrfu) == 36);
    assert!(offset_of!(KvStoreConfig, lrfu_lambda) == 40);
    assert!(offset_of!(KvStoreConfig, lrfu_top_fraction) == 48);
    assert!(offset_of!(KvStoreConfig, estimated_gpu_block_capacity) == 56);
    assert!(offset_of!(KvStoreConfig, gpu_resident_block_bytes) == 64);
    assert!(offset_of!(KvStoreConfig, estimated_block_bytes) == 72);
    assert!(offset_of!(KvStoreConfig, cpu_tier_bytes) == 80);
    assert!(offset_of!(KvStoreConfig, gpu_memory_ratio) == 88);
    assert!(offset_of!(KvStoreConfig, gpu_high_watermark) == 96);
    assert!(offset_of!(KvStoreConfig, gpu_low_watermark) == 104);
};
