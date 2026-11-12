// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `kvmem_tap_f32` device test on the Qwen3.8-27B FullAttention shape
//! (n_heads=24, n_kv_heads=4, head_dim=256). Checks the three contracts the
//! KVMem statistics path relies on:
//!
//! 1. every K row reaches `k_tap` byte-identically, in batch order;
//! 2. only each slot's LAST row reaches `q_tap[slot]`, and no other q_tap
//!    row is written (pre-filled with a sentinel);
//! 3. the Q/K inputs are not modified (the tap must leave the model
//!    arithmetic bit-identical to an untapped forward).
//!
//! `#[ignore]`d: needs a wave32 GPU and a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --test kvmem_tap -- --ignored

use rdna_compute::{DType, Gpu};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const QW: usize = NH * HD;
const KW: usize = NKV * HD;
const SLOTS: usize = 3;
const MAX_ROWS: usize = 8;
/// Per-slot row counts, packed like `SlotBatch::m_per_slot`: slot 0 owns
/// rows 0-1, slot 1 rows 2-4, slot 2 rows 5-6.
const M_PER_SLOT: [usize; SLOTS] = [2, 3, 2];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFF) as f32 / 32768.0 - 1.0
    }
}

#[test]
#[ignore = "needs a GPU and a HIP toolchain"]
fn kvmem_tap_copies_k_rows_and_slot_tail_q() {
    let n_rows: usize = M_PER_SLOT.iter().sum();
    let mut row_slot = Vec::with_capacity(n_rows);
    for (slot, &m) in M_PER_SLOT.iter().enumerate() {
        row_slot.extend(std::iter::repeat(slot as i32).take(m));
    }

    let mut rng = Rng(0x5eed_c0de_1234_5678);
    let mut q = vec![0f32; n_rows * QW];
    let mut k = vec![0f32; n_rows * KW];
    for v in q.iter_mut() {
        *v = rng.unit();
    }
    for v in k.iter_mut() {
        *v = rng.unit();
    }

    let mut gpu = Gpu::init().expect("gpu init");
    let q_dev = gpu.alloc_tensor(&[n_rows * QW], DType::F32).expect("q");
    let k_dev = gpu.alloc_tensor(&[n_rows * KW], DType::F32).expect("k");
    let q_tap = gpu.alloc_tensor(&[SLOTS * QW], DType::F32).expect("q_tap");
    let k_tap = gpu
        .alloc_tensor(&[MAX_ROWS * KW], DType::F32)
        .expect("k_tap");
    let row_slot_dev = gpu.alloc_tensor(&[MAX_ROWS], DType::F32).expect("row_slot");

    let to_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
    gpu.hip
        .memcpy_htod(&q_dev.buf, &to_bytes(&q))
        .expect("q h2d");
    gpu.hip
        .memcpy_htod(&k_dev.buf, &to_bytes(&k))
        .expect("k h2d");
    // Sentinel: -7.0 in every q_tap slot. Any row the kernel does not own
    // must stay at the sentinel.
    let sentinel = vec![-7.0f32; SLOTS * QW];
    gpu.hip
        .memcpy_htod(&q_tap.buf, &to_bytes(&sentinel))
        .expect("q_tap h2d");
    let rs_bytes: Vec<u8> = row_slot.iter().flat_map(|x| x.to_ne_bytes()).collect();
    gpu.hip
        .memcpy_htod(&row_slot_dev.buf, &rs_bytes)
        .expect("row_slot h2d");

    gpu.kvmem_tap_f32(
        &q_dev,
        &k_dev,
        &q_tap,
        &k_tap,
        &row_slot_dev,
        n_rows,
        NH,
        NKV,
        HD,
        SLOTS,
    )
    .expect("tap launch");
    gpu.hip.device_synchronize().expect("sync");

    let read = |buf: &rdna_compute::GpuTensor, elems: usize| -> Vec<f32> {
        let mut bytes = vec![0u8; elems * 4];
        gpu.hip.memcpy_dtoh(&mut bytes, &buf.buf).expect("d2h");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    // 1. K rows, batch order, byte-identical.
    let k_out = read(&k_tap, MAX_ROWS * KW);
    for r in 0..n_rows {
        let got = &k_out[r * KW..(r + 1) * KW];
        let want = &k[r * KW..(r + 1) * KW];
        assert_eq!(got, want, "k_tap row {r} differs from the input row");
    }
    // Rows past n_rows keep the stale contents; the kernel must not write
    // them. (Not asserted: the buffer is uninitialized by contract.)

    // 2. Q: only each slot's last row, and only at that slot's index.
    let q_out = read(&q_tap, SLOTS * QW);
    let last_row = |slot: usize| -> usize { M_PER_SLOT[..=slot].iter().sum::<usize>() - 1 };
    for slot in 0..SLOTS {
        let src = last_row(slot);
        let got = &q_out[slot * QW..(slot + 1) * QW];
        let want = &q[src * QW..(src + 1) * QW];
        assert_eq!(got, want, "q_tap slot {slot} does not hold row {src}");
    }

    // 3. Inputs unmodified.
    let q_back = read(&q_dev, n_rows * QW);
    let k_back = read(&k_dev, n_rows * KW);
    assert_eq!(q_back, q, "tap modified Q");
    assert_eq!(k_back, k, "tap modified K");
}
