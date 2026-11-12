//! Deterministic CPU-vs-GPU parity for the MQ3G256V2 (qt=49) int8 small-batch
//! verify GEMV (`gemm_mq3g256v2_q8x`) — the gfx1100 q8x route ported from MQ4.
//!
//! ## Why this harness exists
//!
//! MQ3G256V2 stores 3-bit codes (0..7) in a 104-byte group: an 8-byte header
//! with two independent fp16 grids `[s0 z0]` (weights 0..127) and `[s1 z1]`
//! (weights 128..255), then a 96-byte payload where weight `w` lives at bit
//! `3*w` (LSB-first). The q8x kernel reads 12 payload bytes per lane (lane `l`
//! owns weights `32*l .. 32*l+31`), unpacks 32 codes, and dots them against
//! int8-quantized activations with `v_dot4_i32_iu8`. The new 3-bit unpack
//! (`q8x3_unpack`) is the key risk: a wrong bit shift or half-select silently
//! corrupts every projection.
//!
//! A fixture with equal halves cannot discriminate a half-select bug. This
//! harness builds DISJOINT halves — half0 in `[-1,1]`, half1 in `[96,160]` —
//! so a kernel that decodes half1 with half0's header reconstructs `~0`
//! instead of `~128` and fails by >100% relative error. The same construction
//! defeats a swapped half and a mis-shifted 3-bit unpack.
//!
//! The route quantizes activations to int8 (q8_1-class, ~4e-3 rel-L2), so it
//! is NOT byte-exact; tolerances mirror `mq4v2_residual_parity`: relative RMS
//! below 5% against the exact-dequant fp32 reference, with the single-header
//! bug required to stay at least 10x worse.
//!
//! Coverage: single-weight and `_multi` (two weights sharing x) launches,
//! both residual (`Y += W·X`) and overwrite (`Y = W·X`) semantics, across
//! batches 2..5 and K∈{256,5120,8192} so both the v4 (persistent, all-K LDS)
//! and v3 (LDS-staged chunks) variants are exercised. The overwrite case
//! uploads a NON-ZERO Y init so an accidental accumulate is caught.
//!
//! Run: `cargo run --release -p hipfire-runtime --example mq3v2_q8x_parity`

use half::f16;
use rdna_compute::Gpu;

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 104;
const LEVELS: f32 = 7.0; // 3-bit codes 0..7

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn build_disjoint_halves(m: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for c in 0..k {
            let gi = c % GROUP;
            let idx = r * k + c;
            if gi < HALF {
                w[idx] = prng(idx, seed ^ 0xA5A5_0001) * 2.0 - 1.0;
            } else {
                w[idx] = 96.0 + prng(idx, seed ^ 0x5A5A_0002) * 64.0;
            }
        }
    }
    w
}

/// Pack f32 weights into the canonical MQ3G256V2 wire format: per-128 fp16
/// `[step lo]` header pairs, then 3-bit codes at bit `3*w` of the 96-byte
/// payload (LSB-first across bytes).
fn pack_mq3g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut q = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let s = &w[src + off..src + off + HALF];
                let lo = s.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / LEVELS } else { 0.0 };
                let sb = if hi == lo {
                    0u16
                } else {
                    f16::from_f32(step).to_bits()
                };
                let zb = f16::from_f32(lo).to_bits();
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&sb.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&zb.to_le_bytes());
                let st = f16::from_bits(sb).to_f32();
                let z = f16::from_bits(zb).to_f32();
                if st == 0.0 {
                    continue;
                }
                let inv = 1.0 / st;
                for i in 0..HALF {
                    q[off + i] = ((s[i] - z) * inv + 0.5).floor().clamp(0.0, LEVELS) as u8;
                }
            }
            // 3-bit LSB-first: weight w occupies bits [3w, 3w+3).
            for w in 0..GROUP {
                let code = q[w] as u32;
                let bit = 3 * w;
                let byte = bit / 8;
                let shift = bit % 8;
                let val = code << shift;
                blob[dst + 8 + byte] |= (val & 0xFF) as u8;
                if shift > 5 {
                    // code spans into the next byte (shift 6 or 7)
                    blob[dst + 8 + byte + 1] |= (val >> 8) as u8;
                }
            }
        }
    }
    blob
}

/// Exact dequant: weight w at bit 3w, header half selected by w<128.
fn decode_w(blob: &[u8], m: usize, k: usize) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..gpr {
            let src = (r * gpr + g) * GROUP_BYTES;
            let h_a = u32::from_le_bytes([blob[src], blob[src + 1], blob[src + 2], blob[src + 3]]);
            let h_b =
                u32::from_le_bytes([blob[src + 4], blob[src + 5], blob[src + 6], blob[src + 7]]);
            let sc0 = f16::from_bits((h_a & 0xFFFF) as u16).to_f32();
            let zp0 = f16::from_bits((h_a >> 16) as u16).to_f32();
            let sc1 = f16::from_bits((h_b & 0xFFFF) as u16).to_f32();
            let zp1 = f16::from_bits((h_b >> 16) as u16).to_f32();
            for wi in 0..GROUP {
                let (sc, zp) = if wi < HALF { (sc0, zp0) } else { (sc1, zp1) };
                let bit = 3 * wi;
                let byte = bit / 8;
                let shift = bit % 8;
                let lo = blob[src + 8 + byte] as u32;
                let hi = if shift > 5 {
                    blob[src + 8 + byte + 1] as u32
                } else {
                    0
                };
                let code = ((lo | (hi << 8)) >> shift) & 7;
                let wk = g * GROUP + wi;
                w[r * k + wk] = sc * (code as f32) + zp;
            }
        }
    }
    w
}

/// Bug baseline: decode every weight with half0's header (single-header read).
fn decode_w_single_header(blob: &[u8], m: usize, k: usize) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..gpr {
            let src = (r * gpr + g) * GROUP_BYTES;
            let h_a = u32::from_le_bytes([blob[src], blob[src + 1], blob[src + 2], blob[src + 3]]);
            let sc0 = f16::from_bits((h_a & 0xFFFF) as u16).to_f32();
            let zp0 = f16::from_bits((h_a >> 16) as u16).to_f32();
            for wi in 0..GROUP {
                let bit = 3 * wi;
                let byte = bit / 8;
                let shift = bit % 8;
                let lo = blob[src + 8 + byte] as u32;
                let hi = if shift > 5 {
                    blob[src + 8 + byte + 1] as u32
                } else {
                    0
                };
                let code = ((lo | (hi << 8)) >> shift) & 7;
                let wk = g * GROUP + wi;
                w[r * k + wk] = sc0 * (code as f32) + zp0;
            }
        }
    }
    w
}

fn ref_gemm(w: &[f32], x: &[f32], y_init: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    // y = y_init + W·X, W [m,k] row-major, X [n,k] row-major, Y [n,m] row-major
    let mut out = vec![0.0f64; n * m];
    for nn in 0..n {
        for mm in 0..m {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += w[mm * k + kk] as f64 * x[nn * k + kk] as f64;
            }
            out[nn * m + mm] = y_init[nn * m + mm] as f64 + acc;
        }
    }
    out
}

fn rel_err(got: &[f32], want: &[f64]) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut wi = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1.0);
        let e = ((g as f64 - w).abs()) / denom;
        if e > worst {
            worst = e;
            wi = i;
        }
    }
    (worst, wi)
}

fn rel_rms(got: &[f32], want: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        num += (g as f64 - w).powi(2);
        den += w * w;
    }
    (num / den.max(1e-30)).sqrt()
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("mq3v2_q8x_parity: no GPU ({e}) — skipping");
            return;
        }
    };
    eprintln!(
        "mq3v2_q8x_parity: arch={} is_gfx1100={} q8x_flag={}",
        gpu.arch,
        gpu.arch_caps.is_gfx1100(),
        gpu.flags.gfx1100_q8x_verify
    );

    // (M, K, batch): K=256/5120 hit v4 (batch<=4), K=8192 hits v3 (batch>2);
    // batch=5 always v3. M varies to exercise row tiling.
    let shapes: &[(usize, usize, usize)] = &[
        (16, 256, 2),
        (32, 256, 4),
        (32, 256, 5),
        (32, 5120, 3),
        (32, 5120, 4),
        (32, 5120, 5),
        (64, 8192, 2),
        (32, 8192, 3),
        (64, 8192, 4),
        (32, 8192, 5),
    ];

    let mut failures = Vec::new();

    for &(m, k, n) in shapes {
        assert!(k % GROUP == 0);
        assert!((2..=5).contains(&n));
        let w_f32 = build_disjoint_halves(m, k, 0);
        let blob = pack_mq3g256v2(&w_f32, m, k);
        assert_eq!(blob.len(), m * (k / GROUP) * GROUP_BYTES);

        let w_correct = decode_w(&blob, m, k);
        let w_bug = decode_w_single_header(&blob, m, k);
        let x: Vec<f32> = (0..n * k)
            .map(|i| prng(i, 0xC0FF_EE00) * 2.0 - 1.0)
            .collect();
        let y_init: Vec<f32> = (0..n * m).map(|i| prng(i, 0xDEAD_BEEF) * 0.5).collect();

        let want = ref_gemm(&w_correct, &x, &y_init, m, k, n);
        let want_bug = ref_gemm(&w_bug, &x, &y_init, m, k, n);
        let bug_rel = {
            let b: Vec<f32> = want_bug.iter().map(|&v| v as f32).collect();
            rel_rms(&b, &want)
        };
        let y_zero = vec![0.0f32; n * m];
        let want0 = ref_gemm(&w_correct, &x, &y_zero, m, k, n);
        eprintln!("shape M={m} K={k} batch={n}: bug baseline rel {bug_rel:.3e}");
        assert!(
            bug_rel > 0.5,
            "fixture not discriminating at M={m} K={k} n={n}: bug_rel {bug_rel:.3e}"
        );

        let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
        let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();

        // ── single weight, residual (Y += W·X) ──
        {
            let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
            gpu.gemm_mq3g256v2_q8x(&[(&d_a, &d_y, m)], &d_x, k, n, true)
                .unwrap_or_else(|e| panic!("single-res M={m} K={k} n={n}: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let got = gpu.download_f32(&d_y).unwrap();
            let (worst, wi) = rel_err(&got, &want);
            let rms = rel_rms(&got, &want);
            let status = if rms < 0.05 { "ok" } else { "FAIL" };
            eprintln!("  single-res  M={m} K={k} n={n}: rel-rms {rms:.3e}, worst {worst:.3e} at {wi} {status}");
            if rms >= 0.05 {
                failures.push(format!("single-res M={m} K={k} n={n} rms {rms:.3e}"));
            }
            assert!(rms < bug_rel * 0.1, "half-select/unpack bug single-res M={m} K={k} n={n}: rms {rms:.3e}");
        }

        // ── single weight, overwrite (Y = W·X); non-zero init catches accumulate ──
        {
            let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
            gpu.gemm_mq3g256v2_q8x(&[(&d_a, &d_y, m)], &d_x, k, n, false)
                .unwrap_or_else(|e| panic!("single-ow M={m} K={k} n={n}: {e}"));
            gpu.hip.device_synchronize().unwrap();
            let got = gpu.download_f32(&d_y).unwrap();
            let rms = rel_rms(&got, &want0);
            let status = if rms < 0.05 { "ok" } else { "FAIL" };
            eprintln!("  single-ow   M={m} K={k} n={n}: rel-rms {rms:.3e} {status}");
            if rms >= 0.05 {
                failures.push(format!("single-ow M={m} K={k} n={n} rms {rms:.3e}"));
            }
        }

        // ── multi (two weights sharing x), residual + overwrite ──
        {
            let w2 = build_disjoint_halves(m, k, 0x1234_5678);
            let blob2 = pack_mq3g256v2(&w2, m, k);
            let w2_correct = decode_w(&blob2, m, k);
            let d_a2 = gpu.upload_raw(&blob2, &[blob2.len()]).unwrap();
            let want2 = ref_gemm(&w2_correct, &x, &y_init, m, k, n);
            let want2_0 = ref_gemm(&w2_correct, &x, &y_zero, m, k, n);

            for residual in [true, false] {
                let d_y = gpu.upload_f32(&y_init, &[n * m]).unwrap();
                let d_y2 = gpu.upload_f32(&y_init, &[n * m]).unwrap();
                gpu.gemm_mq3g256v2_q8x(
                    &[(&d_a, &d_y, m), (&d_a2, &d_y2, m)],
                    &d_x,
                    k,
                    n,
                    residual,
                )
                .unwrap_or_else(|e| panic!("multi res={residual} M={m} K={k} n={n}: {e}"));
                gpu.hip.device_synchronize().unwrap();
                let got = gpu.download_f32(&d_y).unwrap();
                let got2 = gpu.download_f32(&d_y2).unwrap();
                let (w1, w2ref) = if residual { (&want, &want2) } else { (&want0, &want2_0) };
                let rms1 = rel_rms(&got, w1);
                let rms2 = rel_rms(&got2, w2ref);
                let tag = if residual { "multi-res" } else { "multi-ow " };
                let status = if rms1 < 0.05 && rms2 < 0.05 { "ok" } else { "FAIL" };
                eprintln!("  {tag}   M={m} K={k} n={n}: rel-rms w1 {rms1:.3e} w2 {rms2:.3e} {status}");
                if rms1 >= 0.05 || rms2 >= 0.05 {
                    failures.push(format!("{tag} M={m} K={k} n={n} rms {rms1:.3e}/{rms2:.3e}"));
                }
            }
        }
    }

    if failures.is_empty() {
        eprintln!("\nmq3v2_q8x_parity: PASS — 3-bit unpack + half-select correct, single/multi × res/overwrite, v3+v4, batch 2..5");
    } else {
        eprintln!("\nmq3v2_q8x_parity: FAIL — {:?}", failures);
        std::process::exit(1);
    }
}