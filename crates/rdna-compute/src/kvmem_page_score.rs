// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KVMem layerwise current-layer page-softmax scorer (Rust launch seam).
//!
//! Drives the four HIP kernels in `kernels/src/kvmem_page_score.hip` — a faithful
//! port of the qualified KVMem core `PageSoftmaxCuda::score_current_layer` device
//! orchestration (Mean GQA aggregation). The core library is left untouched
//! (KVMEM_ROCM stays OFF); this is hipfire's own GPU scorer, called directly from
//! Rust at an eager Query-selection boundary.
//!
//! Crate boundary: `rdna-compute` has no `hipfire-kvmem` dependency, so this
//! launcher never reads a `RawKvStore`. The caller (the layerwise driver in
//! `hipfire-arch-qwen35`) gathers the canonical Mean-K rows for the frozen score
//! frame's enabled blocks via the shim (`hk_raw_store_mean_k`) and hands them
//! here as a host `&[f32]` slice; the device Q tensor stays on its producing
//! stream/device and is passed by pointer + byte strides. Only the per-page
//! attention masses cross back to the host.
//!
//! Operand boundary matches the core exactly: Mean-K is rounded to FP16 on the
//! host before upload (`f32_to_f16_rne_bits`, round-to-nearest-even, identical to
//! `__float2half_rn(recall_fp16(x))` for in-range finite operands), and Q is
//! rounded to FP16 by the pack kernel, so every dot product is exact in float32.
//! The warp-tree reduction differs from the host SIMD reduction only in float
//! association order (parity is within `2e-6 + 3e-5*expected`, not bit-exact).

use crate::dispatch::{DType, Gpu};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Must match `kernels/src/kvmem_page_score.hip`.
pub const KVMEM_PAGE_SCORE_SRC: &str =
    include_str!("../../../kernels/src/kvmem_page_score.hip");

const MODULE: &str = "kvmem_page_score";
const PACK: &str = "kvmem_pack_current_query";
const LOGITS: &str = "kvmem_page_logits";
const NORMALIZE: &str = "kvmem_normalize_pages";
const ACCUMULATE: &str = "kvmem_accumulate_pages";

/// Core logits budget: one batch row of logits is <= 64 MiB.
const LOGITS_LIMIT: usize = 64 << 20;
const MAX_BATCH: u32 = 64;
const PACK_BLOCK: u32 = 256;
const PAGE_BLOCK: u32 = 128;

/// Native Q scalar type; ordinals mirror `kvmem::NativeQueryScalar` and
/// `hk_native_query_scalar` (F32=0, F16=1, BF16=2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum PageQueryScalar {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

/// Borrowed normalized, pre-RoPE Q from exactly one target attention layer,
/// resident on the device. Strides and extent are BYTES; the token axis need not
/// be outermost. Mirrors `kvmem::NativeCurrentLayerQuery` minus the layer ordinal
/// (the caller resolves the statistics layer before gathering Mean-K).
#[derive(Clone, Copy, Debug)]
pub struct PageScoreQuery {
    pub data: *const c_void,
    pub bytes: usize,
    pub rows: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Zero selects `1/sqrt(head_dim)`.
    pub attention_scale: f32,
    pub scalar: PageQueryScalar,
    pub row_stride: usize,
    pub head_stride: usize,
    pub dimension_stride: usize,
}

/// Host f32->f16 round-to-nearest-even (matches `half::f16::from_f32` /
/// `__float2half_rn` for finite values). Replicated here because `half` is a
/// dev-only dependency; the production launcher must not depend on it.
fn f32_to_f16_rne_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00 | ((mant >> 13) as u16 & 0x1ff)
        };
    }
    let exp16 = exp - 127 + 15;
    if exp16 >= 31 {
        return sign | 0x7c00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let mant_full = mant | 0x0080_0000;
        let shift = (14 - exp16) as u32;
        let half_mant = mant_full >> shift;
        let rem = mant_full & ((1u32 << shift) - 1);
        let half = half_mant as u16;
        let round_up =
            rem > (1u32 << (shift - 1)) || (rem == (1u32 << (shift - 1)) && (half & 1) != 0);
        return sign | (half + u16::from(round_up));
    }
    let half_mant = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let mut half = ((exp16 as u16) << 10) | half_mant;
    let round_up = rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) != 0);
    if round_up {
        half = half.wrapping_add(1);
    }
    sign | half
}

impl Gpu {
    /// Score one target layer's current native Q against `blocks` enabled
    /// canonical pages, writing per-page attention mass into `out` (length
    /// `blocks`). `means` holds `blocks * kv_heads * head_dim` host F32 Mean-K
    /// rows in enabled-page order (page-major, then kv-head, then channel).
    ///
    /// Mirrors `PageSoftmaxCuda::score_current_layer`: the Q operand is validated
    /// for finiteness even when the score domain is empty (`blocks == 0`), in
    /// which case every output mass is zero. Runs at an eager boundary; the
    /// caller owns request/index identity and the frozen score frame.
    pub fn kvmem_page_score_current_layer(
        &mut self,
        q: &PageScoreQuery,
        means: &[f32],
        blocks: usize,
        out: &mut [f64],
    ) -> HipResult<()> {
        if q.data.is_null()
            || q.rows == 0
            || q.query_heads == 0
            || q.kv_heads == 0
            || q.head_dim == 0
            || q.query_heads % q.kv_heads != 0
            || !q.attention_scale.is_finite()
            || q.attention_scale < 0.0
        {
            return Err(HipError::new(
                0,
                "kvmem_page_score: invalid current-layer Query geometry",
            ));
        }
        if out.len() != blocks {
            return Err(HipError::new(
                0,
                &format!(
                    "kvmem_page_score: out length {} != blocks {blocks}",
                    out.len()
                ),
            ));
        }
        let kw = q.kv_heads as usize * q.head_dim as usize;
        let qw = q.query_heads as usize * q.head_dim as usize;
        if means.len() != blocks * kw {
            return Err(HipError::new(
                0,
                &format!(
                    "kvmem_page_score: means length {} != blocks*kw {}",
                    means.len(),
                    blocks * kw
                ),
            ));
        }
        let row_logits_bytes = blocks * q.query_heads as usize * 4;
        if blocks > 0 && row_logits_bytes > LOGITS_LIMIT {
            return Err(HipError::new(
                0,
                "kvmem_page_score: needs a larger per-row logits budget",
            ));
        }
        for v in out.iter_mut() {
            *v = 0.0;
        }

        self.bind_thread()?;
        self.ensure_kernel(MODULE, KVMEM_PAGE_SCORE_SRC, PACK)?;
        self.ensure_kernel(MODULE, KVMEM_PAGE_SCORE_SRC, LOGITS)?;
        self.ensure_kernel(MODULE, KVMEM_PAGE_SCORE_SRC, NORMALIZE)?;
        self.ensure_kernel(MODULE, KVMEM_PAGE_SCORE_SRC, ACCUMULATE)?;

        let batch = if blocks > 0 {
            (MAX_BATCH as usize).min(LOGITS_LIMIT / row_logits_bytes).max(1) as u32
        } else {
            MAX_BATCH
        };

        // Workspace (Raw == byte-sized). queries/norms are always needed (Q is
        // validated even for an empty domain); means/logits/output only when
        // there is something to score.
        let queries = self.alloc_tensor(&[batch as usize * qw * 2], DType::Raw)?;
        let norms = self.alloc_tensor(
            &[batch as usize * q.query_heads as usize * 2 * 4],
            DType::Raw,
        )?;
        let means_dev = if blocks > 0 {
            Some(self.alloc_tensor(&[blocks * kw * 2], DType::Raw)?)
        } else {
            None
        };
        let logits = if blocks > 0 {
            Some(self.alloc_tensor(
                &[batch as usize * row_logits_bytes],
                DType::Raw,
            )?)
        } else {
            None
        };
        let output = if blocks > 0 {
            Some(self.zeros(&[blocks * 4], DType::Raw)?)
        } else {
            None
        };

        if let Some(md) = &means_dev {
            let mut half_bytes = Vec::with_capacity(blocks * kw * 2);
            for &m in means {
                half_bytes.extend_from_slice(&f32_to_f16_rne_bits(m).to_ne_bytes());
            }
            self.hip.memcpy_htod(&md.buf, &half_bytes)?;
        }

        let scale = if q.attention_scale > 0.0 {
            q.attention_scale
        } else {
            1.0 / (q.head_dim as f32).sqrt()
        };
        let weight = 1.0 / q.query_heads as f32;
        let scalar = q.scalar as i32;
        let finite_required = i32::from(blocks > 0);
        let qp = queries.buf.as_ptr();
        let np = norms.buf.as_ptr();
        let mp = means_dev.as_ref().map(|t| t.buf.as_ptr());
        let lp = logits.as_ref().map(|t| t.buf.as_ptr());
        let op = output.as_ref().map(|t| t.buf.as_ptr());

        let mut operand_error = false;
        let mut first = 0u32;
        while first < q.rows {
            let rows = batch.min(q.rows - first);
            // Norms are not live until after packing: borrow their first word as
            // the operand-error flag, read it before the normalizer overwrites it.
            self.hip.memcpy_htod(&norms.buf, &[0u8; 4])?;
            let input = (q.data as usize + first as usize * q.row_stride) as *const c_void;
            self.page_score_pack(
                input,
                qp,
                np,
                rows,
                q.query_heads,
                q.head_dim,
                q.row_stride,
                q.head_stride,
                q.dimension_stride,
                scalar,
                finite_required,
            )?;
            self.hip.device_synchronize()?;
            let mut invalid = [0u8; 4];
            self.hip.memcpy_dtoh(&mut invalid, &norms.buf)?;
            if u32::from_ne_bytes(invalid) != 0 {
                operand_error = true;
                break;
            }
            if blocks > 0 {
                let distributions = rows as i32 * q.query_heads as i32;
                if let Some(limit) = hip_bridge::grid_yz_limit_for_arch(self.arch.as_str()) {
                    if distributions as u32 > limit {
                        return Err(HipError::new(
                            0,
                            &format!(
                                "kvmem_page_score: distributions {distributions} exceeds grid-Y limit {limit}"
                            ),
                        ));
                    }
                }
                let mp = mp.unwrap();
                let lp = lp.unwrap();
                let op = op.unwrap();
                self.page_score_logits(
                    qp,
                    mp,
                    lp,
                    blocks as i32,
                    q.query_heads as i32,
                    q.kv_heads as i32,
                    q.head_dim as i32,
                    scale,
                    distributions,
                )?;
                self.page_score_normalize(lp, np, blocks as i32, distributions)?;
                self.page_score_accumulate(
                    lp,
                    np,
                    op,
                    blocks as i32,
                    distributions,
                    weight,
                )?;
            }
            first += rows;
        }

        if !operand_error {
            if let Some(o) = &output {
                self.hip.device_synchronize()?;
                let mut scores = vec![0u8; blocks * 4];
                self.hip.memcpy_dtoh(&mut scores, &o.buf)?;
                for (j, chunk) in scores.chunks_exact(4).enumerate() {
                    out[j] =
                        f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
                }
            }
        }

        self.free_tensor(queries)?;
        self.free_tensor(norms)?;
        if let Some(t) = means_dev {
            self.free_tensor(t)?;
        }
        if let Some(t) = logits {
            self.free_tensor(t)?;
        }
        if let Some(t) = output {
            self.free_tensor(t)?;
        }

        if operand_error {
            return Err(HipError::new(
                0,
                "kvmem_page_score: native Query contains a nonfinite or overflowing FP16 operand",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn page_score_pack(
        &mut self,
        input: *const c_void,
        output: *mut c_void,
        invalid: *mut c_void,
        rows: u32,
        heads: u32,
        dim: u32,
        row_stride: usize,
        head_stride: usize,
        dimension_stride: usize,
        scalar: i32,
        finite_required: i32,
    ) -> HipResult<()> {
        let count = rows as usize * heads as usize * dim as usize;
        let grid_x = (count as u32).div_ceil(PACK_BLOCK);
        let mut params: Vec<*mut c_void> = vec![
            &input as *const _ as *mut c_void,
            &output as *const _ as *mut c_void,
            &invalid as *const _ as *mut c_void,
            &rows as *const _ as *mut c_void,
            &heads as *const _ as *mut c_void,
            &dim as *const _ as *mut c_void,
            &row_stride as *const _ as *mut c_void,
            &head_stride as *const _ as *mut c_void,
            &dimension_stride as *const _ as *mut c_void,
            &scalar as *const _ as *mut c_void,
            &finite_required as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(PACK, [grid_x, 1, 1], [PACK_BLOCK, 1, 1], 0, &mut params, || {
            let mut b = KernargBlob::new();
            b.push_ptr(input);
            b.push_ptr(output);
            b.push_ptr(invalid);
            b.push_u32(rows);
            b.push_u32(heads);
            b.push_u32(dim);
            b.push_u64(row_stride as u64);
            b.push_u64(head_stride as u64);
            b.push_u64(dimension_stride as u64);
            b.push_i32(scalar);
            b.push_i32(finite_required);
            b
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn page_score_logits(
        &mut self,
        queries: *mut c_void,
        means: *mut c_void,
        logits: *mut c_void,
        blocks: i32,
        qheads: i32,
        kvheads: i32,
        dim: i32,
        scale: f32,
        distributions: i32,
    ) -> HipResult<()> {
        let grid_x = (blocks as u32).div_ceil(4);
        let mut params: Vec<*mut c_void> = vec![
            &queries as *const _ as *mut c_void,
            &means as *const _ as *mut c_void,
            &logits as *const _ as *mut c_void,
            &blocks as *const _ as *mut c_void,
            &qheads as *const _ as *mut c_void,
            &kvheads as *const _ as *mut c_void,
            &dim as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            LOGITS,
            [grid_x, distributions as u32, 1],
            [PAGE_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(queries);
                b.push_ptr(means);
                b.push_ptr(logits);
                b.push_i32(blocks);
                b.push_i32(qheads);
                b.push_i32(kvheads);
                b.push_i32(dim);
                b.push_f32(scale);
                b
            },
        )
    }

    fn page_score_normalize(
        &mut self,
        logits: *mut c_void,
        norms: *mut c_void,
        blocks: i32,
        distributions: i32,
    ) -> HipResult<()> {
        let mut params: Vec<*mut c_void> = vec![
            &logits as *const _ as *mut c_void,
            &norms as *const _ as *mut c_void,
            &blocks as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            NORMALIZE,
            [distributions as u32, 1, 1],
            [PAGE_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(logits);
                b.push_ptr(norms);
                b.push_i32(blocks);
                b
            },
        )
    }

    fn page_score_accumulate(
        &mut self,
        logits: *mut c_void,
        norms: *mut c_void,
        output: *mut c_void,
        blocks: i32,
        distributions: i32,
        weight: f32,
    ) -> HipResult<()> {
        let grid_x = (blocks as u32).div_ceil(PAGE_BLOCK);
        let mut params: Vec<*mut c_void> = vec![
            &logits as *const _ as *mut c_void,
            &norms as *const _ as *mut c_void,
            &output as *const _ as *mut c_void,
            &blocks as *const _ as *mut c_void,
            &distributions as *const _ as *mut c_void,
            &weight as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            ACCUMULATE,
            [grid_x, 1, 1],
            [PAGE_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(logits);
                b.push_ptr(norms);
                b.push_ptr(output);
                b.push_i32(blocks);
                b.push_i32(distributions);
                b.push_f32(weight);
                b
            },
        )
    }
}