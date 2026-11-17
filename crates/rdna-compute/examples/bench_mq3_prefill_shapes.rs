// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! MQ3G256V2 prefill GEMM throughput at production shapes (gfx1100).
//!
//! Phase 0 evidence for the weight-prefill lane:
//!   base (batch-tile 1) vs generic mqv2 batch-tile weight-reuse arms
//!   (gate BT6/BT12, residual BT4/6/8, qkvza/qkv BT4/BT12).
//!
//! Shapes are the Qwen3.8-27B prefill launches observed in
//! `logs/phase0-trace-mq3xt-b12k` (per launch, N = 1024-row chunk):
//!   gate_up  : gate 17408 + up 17408 rows, K=5120
//!   residual : down 5120 x K=17408; o_proj 5120 x K=6144
//!   qkvza    : (qkv 14336, z 2048, beta 48, alpha 48), K=5120   [split: total
//!              M=16480 from trace; per-matrix split is a close hypothesis]
//!   qkv      : (q 12288, k 1024, v 1024), K=5120                [total M=14336]
//!
//! Each arm prints µs, TFLOPS and effective weight-read GiB/s, plus a raw-bit
//! comparison against the base output (full-shape parity signal; the tiny-shape
//! raw-bit screen is `test_mqv2_bt_gfx11` with HIPFIRE_MQV2_BT_SCREEN_BITS3=1).
//!
//! Exit 0 on success; 1 if any parity arm differs or a launch fails.

use rdna_compute::kv_slots::half_from_f32;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const GROUP: usize = 256;
const HALF: usize = 128;
const BITS: u8 = 3;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let mut exp = ((bits >> 10) & 0x1f) as u32;
    let mut mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            while mant & 0x0400 == 0 {
                mant <<= 1;
                exp -= 1;
            }
            sign | (exp << 23) | ((mant & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn build_weights(m: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32 ^ seed;
            for i in 0..HALF {
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

/// Synthetic MQ3G256V2 packer (104 B groups: dual fp16 header + 96 B 3-bit).
/// Identical packing to `test_mqv2_bt_gfx11::pack_mqv2(bits=3, …)`.
fn pack_mq3(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    assert_eq!(w.len(), m * k);
    let gb = 8 + 32 * BITS as usize; // 104
    let mq = 7.0f32;
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * gb];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * gb;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / mq } else { 0.0 };
                let s_bits = if hi == lo { 0u16 } else { half_from_f32(step) };
                let z_bits = half_from_f32(lo);
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16_to_f32(s_bits);
                let z_rt = f16_to_f32(z_bits);
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, mq);
                    codes[off + i] = q as u8;
                }
            }
            for chunk in 0..32 {
                let ci = chunk * 8;
                let qq = [
                    codes[ci] & 7,
                    codes[ci + 1] & 7,
                    codes[ci + 2] & 7,
                    codes[ci + 3] & 7,
                    codes[ci + 4] & 7,
                    codes[ci + 5] & 7,
                    codes[ci + 6] & 7,
                    codes[ci + 7] & 7,
                ];
                let b0 = qq[0] | (qq[1] << 3) | ((qq[2] & 3) << 6);
                let b1 = ((qq[2] >> 2) & 1) | (qq[3] << 1) | (qq[4] << 4) | ((qq[5] & 1) << 7);
                let b2 = ((qq[5] >> 1) & 3) | (qq[6] << 2) | (qq[7] << 5);
                let bo = dst + 8 + chunk * 3;
                blob[bo] = b0;
                blob[bo + 1] = b1;
                blob[bo + 2] = b2;
            }
        }
    }
    blob
}

fn htod_f32(gpu: &Gpu, t: &GpuTensor, host: &[f32]) {
    gpu.hip
        .memcpy_htod(&t.buf, unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
        })
        .expect("htod f32");
}

fn upload_blob(gpu: &Gpu, blob: &[u8]) -> GpuTensor {
    gpu.upload_raw(blob, &[blob.len()]).expect("upload_raw")
}

struct Shape {
    label: &'static str,
    /// Weight rows per matrix (sums to the total M of the production launch).
    ms: Vec<usize>,
    k: usize,
    n: usize,
}

struct ResultRow {
    arm: String,
    us: f64,
    tflops: f64,
    wgibs: f64,
    bit_mism: usize,
    max_abs: f32,
}

fn time_launch(
    gpu: &mut Gpu,
    warmup: usize,
    runs: usize,
    mut f: impl FnMut(&mut Gpu) -> Result<(), String>,
) -> Result<f64, String> {
    for _ in 0..warmup {
        f(gpu)?;
    }
    gpu.hip.device_synchronize().map_err(|e| format!("{e:?}"))?;
    let t0 = Instant::now();
    for _ in 0..runs {
        f(gpu)?;
    }
    gpu.hip.device_synchronize().map_err(|e| format!("{e:?}"))?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / runs as f64)
}

fn compare(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut mism = 0usize;
    let mut max_abs = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            mism += 1;
            max_abs = max_abs.max((x - y).abs());
        }
    }
    (mism, max_abs)
}

fn print_row(r: &ResultRow) {
    println!(
        "    {:<10} {:9.1} µs  {:6.1} TFLOPS  {:7.1} GiB/s-w  bitmism={} maxabs={:.3e}",
        r.arm, r.us, r.tflops, r.wgibs, r.bit_mism, r.max_abs
    );
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    let arch = gpu.arch.clone();
    if arch != "gfx1100" {
        eprintln!("SKIP: arch {arch} is not exact gfx1100 (Phase 0 target)");
        return;
    }
    eprintln!("arch {arch} — MQ3G256V2 prefill shapes: base vs BT arms");

    let shapes = vec![
        Shape { label: "gate_up_n1024", ms: vec![17408, 17408], k: 5120, n: 1024 },
        Shape { label: "gate_up_n96", ms: vec![17408, 17408], k: 5120, n: 96 },
        Shape { label: "gate_up_n80", ms: vec![17408, 17408], k: 5120, n: 80 },
        Shape { label: "gate_up_n64", ms: vec![17408, 17408], k: 5120, n: 64 },
        Shape { label: "resid_down_n1024", ms: vec![5120], k: 17408, n: 1024 },
        Shape { label: "resid_down_n80", ms: vec![5120], k: 17408, n: 80 },
        Shape { label: "resid_down_n64", ms: vec![5120], k: 17408, n: 64 },
        Shape { label: "resid_o_n1024", ms: vec![5120], k: 6144, n: 1024 },
        Shape { label: "qkvza_n1024", ms: vec![14336, 2048, 48, 48], k: 5120, n: 1024 },
        Shape { label: "qkv_n1024", ms: vec![12288, 1024, 1024], k: 5120, n: 1024 },
        // Boundary-row tails: region routing + partial tiles vs base.
        Shape { label: "qkvza_tails", ms: vec![40, 28, 36, 16], k: 512, n: 128 },
        Shape { label: "qkv_tails", ms: vec![40, 32, 48], k: 512, n: 128 },
    ];

    let mut failures = 0usize;
    for sh in &shapes {
        println!(
            "\n== {}  M={}x?  K={}  N={} ==",
            sh.label,
            sh.ms.iter().sum::<usize>(),
            sh.k,
            sh.n
        );
        let gb = 8 + 32 * BITS as usize;
        let blobs: Vec<Vec<u8>> = sh
            .ms
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let w = build_weights(m, sh.k, 0x1000_u32 * (i as u32 + 1) + sh.k as u32);
                pack_mq3(&w, m, sh.k)
            })
            .collect();
        let w_bytes: f64 = sh
            .ms
            .iter()
            .map(|&m| (m * (sh.k / GROUP) * gb) as f64)
            .sum();
        let tflops = |m: usize, us: f64| 2.0 * m as f64 * sh.n as f64 * sh.k as f64 / (us * 1e6);
        let wgibs = |us: f64| w_bytes / (us * 1e-6) / 1_073_741_824.0;

        // Host X and per-matrix device tensors.
        let x_host: Vec<f32> = (0..sh.n * sh.k)
            .map(|i| prng(i, 0xC0FF_EE00 ^ (sh.n as u32)))
            .collect();
        let d_x = gpu.alloc_tensor(&[sh.n * sh.k], DType::F32).expect("x");
        htod_f32(&gpu, &d_x, &x_host);
        let d_w: Vec<GpuTensor> = blobs.iter().map(|b| upload_blob(&gpu, b)).collect();
        let total_m: usize = sh.ms.iter().sum();
        let d_y: Vec<GpuTensor> = sh
            .ms
            .iter()
            .map(|&m| {
                let t = gpu.alloc_tensor(&[sh.n * m], DType::F32).expect("y");
                htod_f32(&gpu, &t, &vec![0.0f32; sh.n * m]);
                t
            })
            .collect();
        gpu.hip.device_synchronize().unwrap();

        // Base arm reference output.
        let ok = match sh.ms.len() {
            2 => gpu.gemm_gate_up_mq3g256v2_wmma_gfx11(
                &d_w[0], &d_w[1], &d_x, &d_y[0], &d_y[1], sh.ms[0], sh.ms[1], sh.k, sh.n,
            ),
            1 => gpu.gemm_mq3g256v2_residual_wmma_gfx11(
                &d_w[0], &d_x, &d_y[0], sh.ms[0], sh.k, sh.n,
            ),
            4 => gpu.gemm_qkvza_mq3g256v2_wmma_gfx11(
                &d_w[0], &d_w[1], &d_w[2], &d_w[3], &d_x, &d_y[0], &d_y[1], &d_y[2],
                &d_y[3], sh.ms[0], sh.ms[1], sh.ms[2], sh.ms[3], sh.k, sh.n,
            ),
            3 => gpu.gemm_qkv_mq3g256v2_wmma_gfx11(
                &d_w[0], &d_w[1], &d_w[2], &d_x, &d_y[0], &d_y[1], &d_y[2], sh.ms[0],
                sh.ms[1], sh.ms[2], sh.k, sh.n,
            ),
            _ => unreachable!(),
        };
        if let Err(e) = ok {
            eprintln!("  base launch FAIL: {e:?}");
            failures += 1;
            continue;
        }
        gpu.hip.device_synchronize().unwrap();
        let y_base: Vec<Vec<f32>> = d_y
            .iter()
            .map(|t| gpu.download_f32(t).expect("dl base"))
            .collect();

        let mut rows: Vec<ResultRow> = Vec::new();
        let us = time_launch(&mut gpu, 2, 10, |g| match sh.ms.len() {
            2 => g
                .gemm_gate_up_mq3g256v2_wmma_gfx11(
                    &d_w[0], &d_w[1], &d_x, &d_y[0], &d_y[1], sh.ms[0], sh.ms[1], sh.k, sh.n,
                )
                .map_err(|e| format!("{e:?}")),
            1 => g
                .gemm_mq3g256v2_residual_wmma_gfx11(
                    &d_w[0], &d_x, &d_y[0], sh.ms[0], sh.k, sh.n,
                )
                .map_err(|e| format!("{e:?}")),
            4 => g
                .gemm_qkvza_mq3g256v2_wmma_gfx11(
                    &d_w[0], &d_w[1], &d_w[2], &d_w[3], &d_x, &d_y[0], &d_y[1], &d_y[2],
                    &d_y[3], sh.ms[0], sh.ms[1], sh.ms[2], sh.ms[3], sh.k, sh.n,
                )
                .map_err(|e| format!("{e:?}")),
            3 => g
                .gemm_qkv_mq3g256v2_wmma_gfx11(
                    &d_w[0], &d_w[1], &d_w[2], &d_x, &d_y[0], &d_y[1], &d_y[2], sh.ms[0],
                    sh.ms[1], sh.ms[2], sh.k, sh.n,
                )
                .map_err(|e| format!("{e:?}")),
            _ => unreachable!(),
        })
        .expect("base timing");
        rows.push(ResultRow {
            arm: "base".into(),
            us,
            tflops: tflops(total_m, us),
            wgibs: wgibs(us),
            bit_mism: 0,
            max_abs: 0.0,
        });

        // BT candidate arms (mq3 gfx11 generic module, production-weight-reuse kernels).
        let bt_arms: Vec<usize> = match sh.ms.len() {
            2 => vec![6, 12],
            1 => vec![4, 6, 8],
            _ => vec![4, 12],
        };
        // MW-LDS arms (weights staged in LDS across waves; all four families).
        let mw_arms: Vec<usize> = match sh.ms.len() {
            1 | 2 | 3 | 4 => vec![4, 8],
            _ => vec![],
        };
        let mut arms: Vec<(String, Box<dyn Fn(&mut Gpu) -> Result<(), String> + '_>)> = Vec::new();
        let dw: &Vec<GpuTensor> = &d_w;
        let dx: &GpuTensor = &d_x;
        let dy: &Vec<GpuTensor> = &d_y;
        let sh_ref: &Shape = sh;
        for &bt in &bt_arms {
            arms.push((
                format!("bt{bt}"),
                Box::new(move |g: &mut Gpu| -> Result<(), String> {
                    match sh_ref.ms.len() {
                        2 => g.gemm_gate_up_mqv2_wmma_gfx11_bt(
                            BITS, bt, &dw[0], &dw[1], dx, &dy[0], &dy[1], sh_ref.ms[0],
                            sh_ref.ms[1], sh_ref.k, sh_ref.n,
                        ),
                        1 => g.gemm_mqv2_residual_wmma_gfx11_bt(
                            BITS, bt, &dw[0], dx, &dy[0], sh_ref.ms[0], sh_ref.k, sh_ref.n,
                        ),
                        4 => g.gemm_qkvza_mqv2_wmma_gfx11_bt(
                            BITS, bt, &dw[0], &dw[1], &dw[2], &dw[3], dx, &dy[0],
                            &dy[1], &dy[2], &dy[3], sh_ref.ms[0], sh_ref.ms[1], sh_ref.ms[2], sh_ref.ms[3],
                            sh_ref.k, sh_ref.n,
                        ),
                        3 => g.gemm_qkv_mqv2_wmma_gfx11_bt(
                            BITS, bt, &dw[0], &dw[1], &dw[2], dx, &dy[0], &dy[1],
                            &dy[2], sh_ref.ms[0], sh_ref.ms[1], sh_ref.ms[2], sh_ref.k, sh_ref.n,
                        ),
                        _ => unreachable!(),
                    }
                    .map_err(|e| format!("{e:?}"))
                }),
            ));
        }
        for &nw in &mw_arms {
            arms.push((
                format!("mw{nw}"),
                Box::new(move |g: &mut Gpu| -> Result<(), String> {
                    match sh_ref.ms.len() {
                        2 => g.gemm_gate_up_mqv2_wmma_gfx11_mw_lds(
                            BITS, nw, &dw[0], &dw[1], dx, &dy[0], &dy[1], sh_ref.ms[0],
                            sh_ref.ms[1], sh_ref.k, sh_ref.n,
                        ),
                        1 => g.gemm_mqv2_residual_wmma_gfx11_mw_lds(
                            BITS, nw, &dw[0], dx, &dy[0], sh_ref.ms[0], sh_ref.k, sh_ref.n,
                        ),
                        4 => g.gemm_qkvza_mqv2_wmma_gfx11_mw_lds(
                            BITS, nw, &dw[0], &dw[1], &dw[2], &dw[3], dx, &dy[0], &dy[1],
                            &dy[2], &dy[3], sh_ref.ms[0], sh_ref.ms[1], sh_ref.ms[2],
                            sh_ref.ms[3], sh_ref.k, sh_ref.n,
                        ),
                        3 => g.gemm_qkv_mqv2_wmma_gfx11_mw_lds(
                            BITS, nw, &dw[0], &dw[1], &dw[2], dx, &dy[0], &dy[1], &dy[2],
                            sh_ref.ms[0], sh_ref.ms[1], sh_ref.ms[2], sh_ref.k, sh_ref.n,
                        ),
                        _ => unreachable!(),
                    }
                    .map_err(|e| format!("{e:?}"))
                }),
            ));
        }
        for (label, run) in arms.iter() {
            // Reset Y before the arm so a failed silent launch is visible.
            for (i, t) in d_y.iter().enumerate() {
                htod_f32(&gpu, t, &vec![0.0f32; sh.n * sh.ms[i]]);
            }
            if let Err(e) = run(&mut gpu) {
                eprintln!("    {label} launch FAIL: {e}");
                failures += 1;
                continue;
            }
            gpu.hip.device_synchronize().unwrap();
            let y_bt: Vec<Vec<f32>> = d_y
                .iter()
                .map(|t| gpu.download_f32(t).expect("dl arm"))
                .collect();
            let (mut mism, mut max_abs) = (0usize, 0.0f32);
            for (a, b) in y_base.iter().zip(y_bt.iter()) {
                let (m, d) = compare(a, b);
                mism += m;
                max_abs = max_abs.max(d);
            }
            let us = time_launch(&mut gpu, 2, 10, |g| run(g)).expect("arm timing");
            rows.push(ResultRow {
                arm: label.clone(),
                us,
                tflops: tflops(total_m, us),
                wgibs: wgibs(us),
                bit_mism: mism,
                max_abs,
            });
        }

        let base_us = rows[0].us;
        for r in &rows {
            print_row(r);
            if r.arm != "base" && r.bit_mism != 0 {
                failures += 1;
            }
        }
        for r in rows.iter().skip(1) {
            println!("    {} speedup vs base: {:.2}x", r.arm, base_us / r.us);
        }
    }

    if failures > 0 {
        eprintln!("\nFAIL: {failures} arm(s) failed or differed from base");
        std::process::exit(1);
    }
    eprintln!("\nPASS: all arms launched; base/branch outputs bit-identical");
}
