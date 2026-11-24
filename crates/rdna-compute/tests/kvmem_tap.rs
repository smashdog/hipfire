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

use rdna_compute::fp8_e4m3::{bf16_rne, bf16_to_f32};
use rdna_compute::kvmem_tap::{query_capture_capacity_bytes, validate_query_source_rows};
use rdna_compute::{DType, Gpu, GpuTensor};

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

#[test]
fn query_source_rows_are_explicit_unique_and_bounded() {
    // Packed slot spans [0,2), [2,5), [5,7); deliberately not slot tails.
    let rows = [5, 0, 3];
    validate_query_source_rows(&rows, 7, 3).unwrap();
    assert_eq!(
        rows,
        [5, 0, 3],
        "validation must not reorder the caller's rows"
    );
    validate_query_source_rows(&[], 7, 0).unwrap();
    for bad in [&[0, 0][..], &[7][..], &[u32::MAX][..], &[0, 1, 2, 3][..]] {
        assert_eq!(validate_query_source_rows(bad, 7, 3).unwrap_err().code, 1);
    }
}

#[test]
fn query_capture_admission_is_bounded_and_overflow_checked() {
    assert_eq!(
        query_capture_capacity_bytes(2, 7, 259, 3).unwrap(),
        2 * 3 * 259 * 2 + 256
    );
    assert_eq!(query_capture_capacity_bytes(1, 1, 1, 1).unwrap(), 512);
    // Logical max_rows does not inflate the allocation beyond capture capacity.
    assert_eq!(
        query_capture_capacity_bytes(2, 7, 259, 3).unwrap(),
        query_capture_capacity_bytes(2, 1024, 259, 3).unwrap(),
    );
    for args in [
        (0, 7, 259, 3),
        (2, 7, 0, 3),
        (2, 7, 259, 0),
        (2, 7, 259, 8),
        (usize::MAX, 7, 259, 3),
        (2, usize::MAX, 259, 3),
        (2, 7, usize::MAX, 3),
    ] {
        assert_eq!(
            query_capture_capacity_bytes(args.0, args.1, args.2, args.3)
                .unwrap_err()
                .code,
            1
        );
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn read_query_test_bytes(gpu: &Gpu, tensor: &GpuTensor) -> Vec<u8> {
    let mut bytes = vec![0u8; tensor.byte_size()];
    gpu.dtoh_stream_ordered_batch(&mut [(&tensor.buf, bytes.as_mut_slice())])
        .expect("stream-ordered test readback");
    bytes
}

fn read_query_test_bf16(gpu: &Gpu, tensor: &GpuTensor) -> Vec<u16> {
    read_query_test_bytes(gpu, tensor)
        .chunks_exact(2)
        .map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]]))
        .collect()
}

/// Independent reference: existing CPU BF16 RNE, not the GPU gather itself.
/// Non-selected rows are NaNs; crossing slot spans and nonmonotonic order must
/// not accidentally sample a slot tail or any poison. Width exercises a partial
/// second workgroup, and destination guards catch wrong layer/capacity strides.
#[test]
#[ignore = "needs a GPU and a HIP toolchain"]
fn query_capture_explicit_rows_cross_slots_and_poison() {
    const ROWS: usize = 7;
    const WIDTH: usize = 259;
    const CAPACITY: usize = 4;
    const LAYERS: usize = 2;
    let source_rows = [5u32, 0, 3];
    let packed_slots = [0, 0, 1, 1, 1, 2, 2];
    assert_eq!(source_rows.map(|r| packed_slots[r as usize]), [2, 0, 1]);
    validate_query_source_rows(&source_rows, ROWS, CAPACITY).unwrap();
    let mut gpu = Gpu::init().expect("gpu init");
    gpu.active_stream = Some(gpu.hip.stream_create_non_blocking().expect("stream"));
    let q = gpu.alloc_tensor(&[ROWS * WIDTH], DType::F32).unwrap();
    let table = gpu.alloc_tensor(&[CAPACITY * 4], DType::Raw).unwrap();
    let output = gpu
        .alloc_tensor(&[LAYERS * CAPACITY * WIDTH], DType::BF16)
        .unwrap();
    let sentinel = bf16_rne(-7.0);
    let mut expected = vec![sentinel; LAYERS * CAPACITY * WIDTH];
    let sentinel_bytes: Vec<u8> = expected.iter().flat_map(|v| v.to_ne_bytes()).collect();
    gpu.htod_stream_ordered(&output.buf, 0, &sentinel_bytes)
        .unwrap();
    let table_bytes: Vec<u8> = source_rows.iter().flat_map(|r| r.to_ne_bytes()).collect();
    gpu.htod_stream_ordered(&table.buf, 0, &table_bytes)
        .unwrap();
    let output_ptr = output.buf.as_ptr();
    for layer in 0..LAYERS {
        let mut input = vec![f32::NAN; ROWS * WIDTH];
        for (dest, &source) in source_rows.iter().enumerate() {
            for col in 0..WIDTH {
                let value = (layer * 10000 + source as usize * WIDTH + col) as f32 / 317.0;
                input[source as usize * WIDTH + col] = value;
                expected[(layer * CAPACITY + dest) * WIDTH + col] = bf16_rne(value);
            }
        }
        let input_bytes = f32_bytes(&input);
        gpu.htod_stream_ordered(&q.buf, 0, &input_bytes).unwrap();
        let view = output.sub_offset(layer * CAPACITY * WIDTH, CAPACITY * WIDTH);
        gpu.kvmem_capture_query_bf16(&q, &table, &view, ROWS, source_rows.len(), WIDTH)
            .unwrap();
        assert_eq!(
            read_query_test_bytes(&gpu, &q),
            input_bytes,
            "gather changed Q (including poison)"
        );
        assert_eq!(
            read_query_test_bf16(&gpu, &output),
            expected,
            "wrong layer/order or a poisoned row leaked"
        );
    }
    // Reuse the same device allocations for a shorter selection. The stale
    // table tail still contains [0,3], but neither it nor the destination tail
    // belongs to this step. No clear/accumulate kernel may touch those rows.
    let input = vec![2.25f32; ROWS * WIDTH];
    gpu.htod_stream_ordered(&q.buf, 0, &f32_bytes(&input))
        .unwrap();
    gpu.htod_stream_ordered(&table.buf, 0, &2u32.to_ne_bytes())
        .unwrap();
    gpu.kvmem_capture_query_bf16(&q, &table, &output, ROWS, 1, WIDTH)
        .unwrap();
    expected[..WIDTH].fill(bf16_rne(2.25));
    assert_eq!(read_query_test_bf16(&gpu, &output), expected);
    gpu.kvmem_capture_query_bf16(&q, &table, &output, ROWS, 0, WIDTH)
        .unwrap();
    assert_eq!(
        read_query_test_bf16(&gpu, &output),
        expected,
        "empty selection wrote data"
    );
    assert_eq!(output.buf.as_ptr(), output_ptr);
    gpu.free_tensor(q).unwrap();
    gpu.free_tensor(table).unwrap();
    gpu.free_tensor(output).unwrap();
    let stream = gpu.active_stream.take().unwrap();
    gpu.hip.stream_destroy(stream).unwrap();
}

/// Fixed decimal expectations distinguish RNE from truncation and ties-away.
/// The primitive also preserves non-finites so the owner can explicitly reject
/// them; equality on NaNs is deliberately NOT used as a passing oracle.
#[test]
#[ignore = "needs a GPU and a HIP toolchain"]
fn query_capture_bf16_conversion_oracle_and_validation() {
    let finite_cases = [
        (0.0f32, 0.0f32),
        (-0.0, -0.0),
        (1.0, 1.0),
        (1.00390625, 1.0),
        (1.01171875, 1.015625),
        (-1.00390625, -1.0),
        (-1.01171875, -1.015625),
        (1.004, 1.0078125),
        (-1.004, -1.0078125),
        (f32::MIN_POSITIVE, f32::MIN_POSITIVE),
    ];
    let mut input: Vec<f32> = finite_cases.iter().map(|&(v, _)| v).collect();
    input.extend([f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
    let width = input.len();
    let mut gpu = Gpu::init().expect("gpu init");
    let q = gpu.upload_f32(&input, &[width]).unwrap();
    let table = gpu.alloc_tensor(&[4], DType::Raw).unwrap();
    gpu.htod_stream_ordered(&table.buf, 0, &0u32.to_ne_bytes())
        .unwrap();
    let output = gpu.alloc_tensor(&[width], DType::BF16).unwrap();
    gpu.kvmem_capture_query_bf16(&q, &table, &output, 1, 1, width)
        .unwrap();
    let bits = read_query_test_bf16(&gpu, &output);
    for (i, &(source, expected)) in finite_cases.iter().enumerate() {
        assert_eq!(bits[i], bf16_rne(source), "CPU BF16 primitive at {i}");
        assert_eq!(
            bf16_to_f32(bits[i]).to_bits(),
            expected.to_bits(),
            "fixed RNE oracle at {i}"
        );
    }
    assert_eq!(bf16_to_f32(bits[width - 3]), f32::INFINITY);
    assert_eq!(bf16_to_f32(bits[width - 2]), f32::NEG_INFINITY);
    assert!(bf16_to_f32(bits[width - 1]).is_nan());
    assert_eq!(read_query_test_bytes(&gpu, &q), f32_bytes(&input));
    // Invalid dimensions, dtypes, short buffers, and aliases are typed errors,
    // before any kernel can write the model's input storage.
    assert_eq!(
        gpu.kvmem_capture_query_bf16(&q, &table, &output, 1, 2, width)
            .unwrap_err()
            .code,
        1
    );
    assert_eq!(
        gpu.kvmem_capture_query_bf16(&q, &table, &output, usize::MAX, 1, width)
            .unwrap_err()
            .code,
        1
    );
    assert_eq!(
        gpu.kvmem_capture_query_bf16(&output, &table, &output, 1, 1, width)
            .unwrap_err()
            .code,
        1
    );
    let short = output.sub_offset(0, width - 1);
    assert_eq!(
        gpu.kvmem_capture_query_bf16(&q, &table, &short, 1, 1, width)
            .unwrap_err()
            .code,
        1
    );
    let mut alias = q.sub_offset(0, width);
    alias.dtype = DType::BF16;
    alias.shape = vec![width];
    assert_eq!(
        gpu.kvmem_capture_query_bf16(&q, &table, &alias, 1, 1, width)
            .unwrap_err()
            .code,
        1
    );
    drop(alias);
    drop(short);
    gpu.free_tensor(q).unwrap();
    gpu.free_tensor(table).unwrap();
    gpu.free_tensor(output).unwrap();
}
