// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `kvmem_page_score_current_layer` device qualification on a real RDNA GPU.
//!
//! Port of the KVMem core `layerwise_scoring_cuda_test.cpp` oracle: a represented
//! input FP64 reference for the per-(row,head) page softmax with Mean GQA
//! aggregation. Checks the contracts the layerwise driver relies on:
//!
//! 1. per-page mass matches the FP64 oracle within `2e-6 + 3e-5*expected`
//!    (the warp-tree float reduction vs the host double reference);
//! 2. raw mass is conserved (`sum == rows`), so no Query row/head is dropped;
//! 3. F32 / F16 / BF16 native scalars and contiguous / token-minor byte strides
//!    all recover the same logical Q (the pack kernel's stride permutation);
//! 4. multi-batch rows (rows > 64) and the real Qwen3.8-27B FA shape agree;
//! 5. an empty score domain yields zero mass but still validates Q finiteness;
//! 6. a nonfinite / FP16-overflowing operand is rejected fail-closed.
//!
//! `#[ignore]`d: needs a wave32 GPU and a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --test kvmem_page_score -- --ignored

use rdna_compute::kvmem_page_score::{PageQueryScalar, PageScoreQuery};
use rdna_compute::{DType, Gpu};

const SCALE: f32 = 0.0625;

fn scalar_bytes(s: PageQueryScalar) -> usize {
    match s {
        PageQueryScalar::F32 => 4,
        PageQueryScalar::F16 | PageQueryScalar::BF16 => 2,
    }
}

/// Encode a logical f32 Q value into its native scalar bytes, and return the
/// FP16-represented value the pack kernel will produce (the oracle operand).
fn represent(value: f32, s: PageQueryScalar, dest: &mut [u8]) -> f32 {
    let decoded = match s {
        PageQueryScalar::F32 => {
            dest.copy_from_slice(&value.to_ne_bytes());
            value
        }
        PageQueryScalar::F16 => {
            let h = half::f16::from_f32(value);
            dest.copy_from_slice(&h.to_bits().to_ne_bytes());
            h.to_f32()
        }
        PageQueryScalar::BF16 => {
            let b = half::bf16::from_f32(value);
            dest.copy_from_slice(&b.to_bits().to_ne_bytes());
            b.to_f32()
        }
    };
    // Explicit FP16 operand boundary, independent of the native storage scalar.
    half::f16::from_f32(decoded).to_f32()
}

struct Case {
    worst: f64,
    mass_err: f64,
}

fn qualify(s: PageQueryScalar, token_minor: bool, rows: usize, blocks: usize,
           heads: usize, kvheads: usize, dim: usize) -> Case {
    let kw = kvheads * dim;
    let qw = heads * dim;
    let sb = scalar_bytes(s);

    // Canonical Mean-K (raw f32; the launcher rounds to FP16 on upload).
    let mut means = vec![0f32; blocks * kw];
    for b in 0..blocks {
        for d in 0..kw {
            means[b * kw + d] = (((b * 13 + d * 7) % 101) as i32 - 50) as f32 / 16.0;
        }
    }
    let rep_means: Vec<f32> = means.iter().map(|&m| half::f16::from_f32(m).to_f32()).collect();

    // Logical Q rows and their native strided encoding.
    let row_stride = if token_minor { sb } else { (qw + 8) * sb };
    let dimension_stride = if token_minor { rows * sb } else { sb };
    let head_stride = if token_minor { (rows * dim + 8) * sb } else { dim * sb };
    let bytes = (rows - 1) * row_stride + (heads - 1) * head_stride
        + (dim - 1) * dimension_stride + sb;
    let mut native = vec![0xffu8; bytes];
    let mut rep_query = vec![0f32; rows * qw];
    for r in 0..rows {
        for h in 0..heads {
            for d in 0..dim {
                let index = (r * heads + h) * dim + d;
                let value = (((index * 17) % 127) as i32 - 63) as f32 / 31.0;
                let off = r * row_stride + h * head_stride + d * dimension_stride;
                rep_query[index] = represent(value, s, &mut native[off..off + sb]);
            }
        }
    }

    // FP64 oracle: per-(row,head) softmax over all blocks, mean over heads.
    let mut expected = vec![0f64; blocks];
    for r in 0..rows {
        for h in 0..heads {
            let kvhead = h / (heads / kvheads);
            let mut logits = vec![0f64; blocks];
            let mut maximum = f64::NEG_INFINITY;
            for b in 0..blocks {
                let mut dot = 0f64;
                for d in 0..dim {
                    dot += rep_query[(r * heads + h) * dim + d] as f64
                        * rep_means[(b * kvheads + kvhead) * dim + d] as f64;
                }
                logits[b] = dot * SCALE as f64;
                maximum = maximum.max(logits[b]);
            }
            let mut total = 0f64;
            for b in 0..blocks {
                logits[b] = (logits[b] - maximum).exp();
                total += logits[b];
            }
            for b in 0..blocks {
                expected[b] += logits[b] / total / heads as f64;
            }
        }
    }

    let mut gpu = Gpu::init().expect("gpu init");
    let q_dev = gpu.alloc_tensor(&[bytes], DType::Raw).expect("q dev");
    gpu.hip.memcpy_htod(&q_dev.buf, &native).expect("q h2d");
    let view = PageScoreQuery {
        data: q_dev.buf.as_ptr(),
        bytes,
        rows: rows as u32,
        query_heads: heads as u32,
        kv_heads: kvheads as u32,
        head_dim: dim as u32,
        attention_scale: SCALE,
        scalar: s,
        row_stride,
        head_stride,
        dimension_stride,
    };
    let mut actual = vec![0f64; blocks];
    gpu.kvmem_page_score_current_layer(&view, &means, blocks, &mut actual)
        .expect("score launch");

    let mut worst = 0f64;
    for b in 0..blocks {
        assert!(actual[b].is_finite(), "nonfinite mass at page {b}");
        let err = (actual[b] - expected[b]).abs();
        worst = worst.max(err);
        assert!(
            err <= 2e-6 + expected[b] * 3e-5,
            "page {b} mass {} differs from oracle {} (err {err})",
            actual[b], expected[b]
        );
    }
    let mass: f64 = actual.iter().sum();
    let mass_err = (mass - rows as f64).abs();
    assert!(mass_err < rows as f64 * 2e-5, "lost raw mass: {mass} vs {rows}");

    // Tiling: row-sliced calls must sum to the same per-page denominator.
    let mut tiled = vec![0f64; blocks];
    let tile = 7usize;
    let mut first = 0usize;
    while first < rows {
        let n = tile.min(rows - first);
        let sub = PageScoreQuery {
            data: (q_dev.buf.as_ptr() as usize + first * row_stride) as *const std::ffi::c_void,
            bytes: bytes - first * row_stride,
            rows: n as u32,
            ..view
        };
        let mut part = vec![0f64; blocks];
        gpu.kvmem_page_score_current_layer(&sub, &means, blocks, &mut part)
            .expect("tiled score launch");
        for b in 0..blocks {
            tiled[b] += part[b];
        }
        first += n;
    }
    for b in 0..blocks {
        assert!(
            (tiled[b] - expected[b]).abs() <= 2e-6 + expected[b] * 3e-5,
            "tiled page {b} changed the formula"
        );
    }

    // Empty score domain: zero mass, but Q finiteness is still enforced.
    let mut empty: Vec<f64> = Vec::new();
    gpu.kvmem_page_score_current_layer(&view, &[], 0, &mut empty)
        .expect("empty domain score");
    assert!(empty.is_empty());

    // A nonfinite first operand must be rejected, even with an empty domain.
    let bad_bits: &[u8] = match s {
        PageQueryScalar::F32 => &0x7fc0_0000u32.to_ne_bytes(),
        PageQueryScalar::F16 => &0x7e00u16.to_ne_bytes(),
        PageQueryScalar::BF16 => &0x7fc0u16.to_ne_bytes(),
    };
    gpu.hip.memcpy_htod(&q_dev.buf, bad_bits).expect("bad h2d");
    for empty_domain in [false, true] {
        let n = if empty_domain { 0 } else { blocks };
        let m: &[f32] = if empty_domain { &[] } else { &means };
        let mut o = vec![0f64; n];
        let res = gpu.kvmem_page_score_current_layer(&view, m, n, &mut o);
        assert!(res.is_err(), "nonfinite operand accepted (empty={empty_domain})");
    }
    // Restore and confirm the failure did not poison a retry.
    gpu.hip.memcpy_htod(&q_dev.buf, &native).expect("restore h2d");
    let mut retry = vec![0f64; blocks];
    gpu.kvmem_page_score_current_layer(&view, &means, blocks, &mut retry)
        .expect("retry score");
    assert_eq!(retry, actual, "operand failure poisoned the retry");

    gpu.free_tensor(q_dev).expect("free q");
    Case { worst, mass_err }
}

#[test]
#[ignore = "needs a GPU and a HIP toolchain"]
fn page_score_matches_fp64_oracle() {
    for s in [PageQueryScalar::F32, PageQueryScalar::F16, PageQueryScalar::BF16] {
        for token_minor in [false, true] {
            let c = qualify(s, token_minor, 129, 7, 8, 2, 64);
            eprintln!(
                "scalar={s:?} token_minor={token_minor} worst_abs={} mass_err={}",
                c.worst, c.mass_err
            );
        }
    }
    // Real Qwen3.8-27B FullAttention shape (n_heads=24, n_kv_heads=4, head_dim=256)
    // at the N-gate page budget; rows>64 exercises the multi-batch loop.
    let c = qualify(PageQueryScalar::F32, false, 129, 45, 24, 4, 256);
    eprintln!("qwen35-shape worst_abs={} mass_err={}", c.worst, c.mass_err);
    // A large block count exercises the shared-mem normalize reduction tail.
    let c = qualify(PageQueryScalar::BF16, false, 2, 8193, 8, 2, 64);
    eprintln!("wide-blocks worst_abs={} mass_err={}", c.worst, c.mass_err);
}