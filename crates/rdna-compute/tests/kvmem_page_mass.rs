// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `kvmem_page_mass` device qualification on a real RDNA GPU.
//!
//! FP64 oracle for the ninfer-faithful pressure attention-mass recompute: sample
//! native Q against a represented Q8_0 K cache, normalize each page's token
//! logsumexp across the COMPLETE visible cache, average over query heads. Checks:
//!
//! 1. per-(sample,page) mass matches the FP64 oracle within `2e-6 + 3e-5*expected`;
//! 2. mass is conserved per sample (`sum_p mass == 1`);
//! 3. F32 / F16 / BF16 native Q scalars all recover the same logical Q;
//! 4. the real Qwen3.8-27B FA shape (24 Q / 4 KV / 256) at 128 tokens/page agrees;
//! 5. a permuted block table and partial (mid-page) causal ends are honored;
//! 6. an unmapped VISIBLE physical page is rejected fail-closed (NaN mass).
//!
//! `#[ignore]`d: needs a wave32 GPU and a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --test kvmem_page_mass -- --ignored

use rdna_compute::kvmem_page_mass::{PageMassQuery, PageMassScalar, PAGE_TOKENS};
use rdna_compute::{DType, Gpu};

const SCALE: f32 = 0.0625; // 1/sqrt(256)
const DIM: usize = 256;
const BLK: usize = 34;
const SEG: usize = DIM / 32 * BLK; // 272

fn scalar_bytes(s: PageMassScalar) -> usize {
    match s {
        PageMassScalar::F32 => 4,
        PageMassScalar::F16 | PageMassScalar::BF16 => 2,
    }
}

/// Encode a logical f32 Q value into native scalar bytes; return the represented
/// value the kernel will read (F32 exact, F16/BF16 rounded).
fn represent_q(value: f32, s: PageMassScalar, dest: &mut [u8]) -> f32 {
    match s {
        PageMassScalar::F32 => {
            dest.copy_from_slice(&value.to_ne_bytes());
            value
        }
        PageMassScalar::F16 => {
            let h = half::f16::from_f32(value);
            dest.copy_from_slice(&h.to_bits().to_ne_bytes());
            h.to_f32()
        }
        PageMassScalar::BF16 => {
            let b = half::bf16::from_f32(value);
            dest.copy_from_slice(&b.to_bits().to_ne_bytes());
            b.to_f32()
        }
    }
}

/// Build a page-major Q8_0 K cache and its FP64 represented values. Layout per
/// token row: kv_heads * SEG bytes; block b = f16 scale at 34*b + 32 int8 codes.
fn build_k(physical_pages: usize, kvheads: usize) -> (Vec<u8>, Vec<f64>) {
    let row_stride = kvheads * SEG;
    let mut bytes = vec![0u8; physical_pages * PAGE_TOKENS as usize * row_stride];
    let mut rep = vec![0f64; physical_pages * PAGE_TOKENS as usize * kvheads * DIM];
    for pp in 0..physical_pages {
        for t in 0..PAGE_TOKENS as usize {
            for h in 0..kvheads {
                for b in 0..DIM / 32 {
                    // Logical K block, then q8_0 quantize.
                    let mut vals = [0f32; 32];
                    let mut amax = 0f32;
                    for i in 0..32 {
                        let d = b * 32 + i;
                        let idx = ((pp * 137 + t * 31 + h * 17 + d * 7) % 251) as i32 - 125;
                        vals[i] = idx as f32 / 40.0;
                        amax = amax.max(vals[i].abs());
                    }
                    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                    let scale_h = half::f16::from_f32(scale);
                    let scale_rep = scale_h.to_f32();
                    let base = (pp * PAGE_TOKENS as usize + t) * row_stride + h * SEG + b * BLK;
                    bytes[base..base + 2].copy_from_slice(&scale_h.to_bits().to_ne_bytes());
                    for i in 0..32 {
                        let code = if scale_rep > 0.0 {
                            (vals[i] / scale_rep).round().clamp(-127.0, 127.0) as i8
                        } else {
                            0
                        };
                        bytes[base + 2 + i] = code as u8;
                        let d = b * 32 + i;
                        let ri = ((pp * PAGE_TOKENS as usize + t) * kvheads + h) * DIM + d;
                        rep[ri] = code as f64 * scale_rep as f64;
                    }
                }
            }
        }
    }
    (bytes, rep)
}

struct Case {
    worst: f64,
    conservation: f64,
}

#[allow(clippy::too_many_arguments)]
fn qualify(
    s: PageMassScalar,
    samples: usize,
    pages: usize,
    heads: usize,
    kvheads: usize,
    physical_pages: usize,
    permute: bool,
    partial: bool,
) -> Case {
    let sb = scalar_bytes(s);
    let pt = PAGE_TOKENS as usize;
    let qrows = samples + 3; // sampled rows are a subset of a larger Q tensor

    // Native Q [qrows, heads, DIM], contiguous.
    let mut native = vec![0u8; qrows * heads * DIM * sb];
    let mut rep_q = vec![0f32; qrows * heads * DIM];
    for r in 0..qrows {
        for h in 0..heads {
            for d in 0..DIM {
                let index = (r * heads + h) * DIM + d;
                let value = (((index * 17) % 127) as i32 - 63) as f32 / 31.0;
                let off = index * sb;
                rep_q[index] = represent_q(value, s, &mut native[off..off + sb]);
            }
        }
    }

    let (k_bytes, rep_k) = build_k(physical_pages, kvheads);

    // Block table: logical page -> physical page (optionally permuted).
    let mut table = vec![0i32; pages];
    for p in 0..pages {
        table[p] = if permute {
            ((p * 7 + 3) % physical_pages) as i32
        } else {
            p as i32
        };
    }
    // Sampled rows + compact causal ends.
    let mut rows = vec![0i32; samples];
    let mut positions = vec![0i32; samples];
    for s in 0..samples {
        rows[s] = (s % qrows) as i32;
        positions[s] = if partial {
            // Mid-page ends spread across the visible domain.
            ((s * 197 + 40) % (pages * pt)) as i32
        } else {
            (pages * pt - 1) as i32
        };
    }

    // FP64 oracle.
    let mut expected = vec![0f64; samples * pages];
    for s in 0..samples {
        let visible = positions[s] as usize + 1;
        let qrow = rows[s] as usize;
        for h in 0..heads {
            let kvhead = h / (heads / kvheads);
            let mut lse = vec![f64::NEG_INFINITY; pages];
            for p in 0..pages {
                if p * pt >= visible {
                    continue;
                }
                let phys = table[p] as usize;
                let mut maximum = f64::NEG_INFINITY;
                let mut logits = Vec::new();
                for off in 0..pt {
                    if p * pt + off >= visible {
                        break;
                    }
                    let mut dot = 0f64;
                    for d in 0..DIM {
                        let qi = (qrow * heads + h) * DIM + d;
                        let ki = ((phys * pt + off) * kvheads + kvhead) * DIM + d;
                        dot += rep_q[qi] as f64 * rep_k[ki];
                    }
                    let logit = dot * SCALE as f64;
                    maximum = maximum.max(logit);
                    logits.push(logit);
                }
                let total: f64 = logits.iter().map(|l| (l - maximum).exp()).sum();
                lse[p] = maximum + total.ln();
            }
            let gmax = lse.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let gsum: f64 = lse.iter().map(|l| (l - gmax).exp()).sum();
            for p in 0..pages {
                expected[s * pages + p] += (lse[p] - gmax).exp() / gsum / heads as f64;
            }
        }
    }

    let mut gpu = Gpu::init().expect("gpu init");
    let q_dev = gpu.alloc_tensor(&[native.len()], DType::Raw).expect("q dev");
    gpu.hip.memcpy_htod(&q_dev.buf, &native).expect("q h2d");
    let k_dev = gpu.alloc_tensor(&[k_bytes.len()], DType::Raw).expect("k dev");
    gpu.hip.memcpy_htod(&k_dev.buf, &k_bytes).expect("k h2d");
    let view = PageMassQuery {
        data: q_dev.buf.as_ptr(),
        rows: qrows as u32,
        query_heads: heads as u32,
        kv_heads: kvheads as u32,
        head_dim: DIM as u32,
        attention_scale: SCALE,
        scalar: s,
    };
    let mut actual = vec![0f32; samples * pages];
    gpu.kvmem_page_mass(&view, &rows, &positions, &table, k_dev.buf.as_ptr(), physical_pages, &mut actual)
        .expect("mass launch");

    let mut worst = 0f64;
    let mut conservation = 0f64;
    for s in 0..samples {
        let mut sum = 0f64;
        for p in 0..pages {
            let a = actual[s * pages + p] as f64;
            let e = expected[s * pages + p];
            assert!(a.is_finite(), "nonfinite mass at sample {s} page {p}");
            let err = (a - e).abs();
            worst = worst.max(err);
            assert!(
                err <= 2e-6 + e * 3e-5,
                "sample {s} page {p} mass {a} differs from oracle {e} (err {err})"
            );
            sum += a;
        }
        conservation = conservation.max((sum - 1.0).abs());
        assert!((sum - 1.0).abs() < 2e-4, "sample {s} lost mass: {sum} vs 1");
    }

    // An unmapped VISIBLE physical page must be rejected fail-closed.
    let mut bad_table = table.clone();
    bad_table[0] = -1;
    let mut o = vec![0f32; samples * pages];
    let res = gpu.kvmem_page_mass(&view, &rows, &positions, &bad_table, k_dev.buf.as_ptr(), physical_pages, &mut o);
    assert!(res.is_err(), "unmapped visible page accepted");
    // Retry with the good table must not be poisoned.
    let mut retry = vec![0f32; samples * pages];
    gpu.kvmem_page_mass(&view, &rows, &positions, &table, k_dev.buf.as_ptr(), physical_pages, &mut retry)
        .expect("retry mass");
    assert_eq!(retry, actual, "NaN failure poisoned the retry");

    gpu.free_tensor(q_dev).expect("free q");
    gpu.free_tensor(k_dev).expect("free k");
    Case { worst, conservation }
}

#[test]
#[ignore = "needs a GPU and a HIP toolchain"]
fn page_mass_matches_fp64_oracle() {
    for s in [PageMassScalar::F32, PageMassScalar::F16, PageMassScalar::BF16] {
        let c = qualify(s, 5, 7, 8, 2, 7, false, false);
        eprintln!("scalar={s:?} worst_abs={} conservation={}", c.worst, c.conservation);
    }
    // Real Qwen3.8-27B FA shape, full visible domain, identity table.
    let c = qualify(PageMassScalar::F32, 12, 45, 24, 4, 45, false, false);
    eprintln!("qwen35-shape worst_abs={} conservation={}", c.worst, c.conservation);
    // Permuted block table + partial mid-page causal ends + extra physical pages.
    let c = qualify(PageMassScalar::BF16, 16, 33, 24, 4, 40, true, true);
    eprintln!("permuted-partial worst_abs={} conservation={}", c.worst, c.conservation);
    // Sample cap boundary (64) and a wide page count (normalize tail).
    let c = qualify(PageMassScalar::F16, 64, 129, 24, 4, 129, true, true);
    eprintln!("cap-wide worst_abs={} conservation={}", c.worst, c.conservation);
}