// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KVMem layerwise pressure attention-mass tap (Rust launch seam).
//!
//! Drives the three HIP kernels in `kernels/src/kvmem_page_mass.hip` — a faithful
//! port of ninfer's `attention_page_mass` recompute kernel. It samples a target
//! layer's native Q against the represented persistent Q8_0 K cache and returns
//! the per-page attention mass (normalized across the complete visible cache,
//! averaged over query heads). This is a SEPARATE K-only recompute: the legacy
//! attention kernels are untouched, so the default path stays byte-neutral.
//!
//! Crate boundary: `rdna-compute` has no `hipfire-kvmem` dependency. The caller
//! (the layerwise driver in `hipfire-arch-qwen35`) owns the device Q tensor and
//! the Q8_0 K cache, selects the sampled rows / compact causal ends / block-table
//! row, and hands them here. Only the per-page masses cross back to the host.
//!
//! Operand boundary: Q is read in its native scalar (F32/F16/BF16) and converted
//! to float; the dot is exact in float32. The warp-tree reduction differs from a
//! host double reference only in float association order (parity within
//! `2e-6 + 3e-5*expected`, not bit-exact). An unmapped/out-of-range VISIBLE
//! physical page produces NaN, which this launcher rejects fail-closed.

use crate::dispatch::{DType, Gpu};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Must match `kernels/src/kvmem_page_mass.hip`.
pub const KVMEM_PAGE_MASS_SRC: &str = include_str!("../../../kernels/src/kvmem_page_mass.hip");

const MODULE: &str = "kvmem_page_mass";
const LSE: &str = "kvmem_page_mass_lse";
const NORMALIZE: &str = "kvmem_page_mass_normalize";
const AVERAGE: &str = "kvmem_page_mass_average";

/// hipfire layerwise page size (tokens). Must match `LAYERWISE_BLOCK_TOKENS` in
/// hipfire-arch-qwen35/src/layerwise_layout.rs and the core catalog block_tokens.
pub const PAGE_TOKENS: u32 = 128;
/// ninfer sample cap: at most 64 sampled Q rows per call.
const MAX_SAMPLES: usize = 64;
const BLOCK: u32 = 128;
const DIM: u32 = 256;

/// Native Q scalar type; ordinals mirror `kvmem::NativeQueryScalar` (F32=0/F16=1/BF16=2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum PageMassScalar {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

/// Borrowed native Q from exactly one target attention layer, resident on the
/// device, contiguous `[rows, query_heads, head_dim]` (head_dim innermost). Q must
/// be in the SAME rotational space as the stored K (post-RoPE) for the mass to
/// reflect real attention; the caller resolves that before sampling.
#[derive(Clone, Copy, Debug)]
pub struct PageMassQuery {
    pub data: *const c_void,
    pub rows: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Zero selects `1/sqrt(head_dim)`.
    pub attention_scale: f32,
    pub scalar: PageMassScalar,
}

impl Gpu {
    /// Sample `rows.len()` Q rows against `pages` logical pages of the Q8_0 K
    /// cache, writing per-page attention mass into `out` (length
    /// `samples * pages`, sample-major then page). `keys` is the page-major Q8_0
    /// K cache device base (`physical_pages` pages, each `PAGE_TOKENS` tokens,
    /// token-major `kv_heads * (head_dim/32)*34` bytes/token). `table` maps each
    /// logical page to its physical page (-1 = unmapped). `positions[s]` is the
    /// inclusive compact causal end of sample `s` (visible = positions[s]+1).
    ///
    /// Mirrors ninfer `attention_page_mass`: each page's token logsumexp is
    /// normalized across the COMPLETE visible cache, then heads are averaged.
    /// No V read, no allocation/host-copy/state-mutation beyond the disjoint
    /// workspaces. Runs at a post-attention boundary; the caller owns identity.
    #[allow(clippy::too_many_arguments)]
    pub fn kvmem_page_mass(
        &mut self,
        q: &PageMassQuery,
        rows: &[i32],
        positions: &[i32],
        table: &[i32],
        keys: *const c_void,
        physical_pages: usize,
        out: &mut [f32],
    ) -> HipResult<()> {
        let samples = rows.len();
        let pages = table.len();
        if q.data.is_null()
            || keys.is_null()
            || q.rows == 0
            || q.query_heads == 0
            || q.kv_heads == 0
            || q.head_dim != DIM
            || q.query_heads % q.kv_heads != 0
            || !q.attention_scale.is_finite()
            || q.attention_scale < 0.0
        {
            return Err(HipError::new(0, "kvmem_page_mass: invalid Query/K geometry"));
        }
        if samples == 0 || samples > MAX_SAMPLES || positions.len() != samples {
            return Err(HipError::new(
                0,
                "kvmem_page_mass: samples must be in [1, 64] and match positions",
            ));
        }
        if pages == 0 || physical_pages == 0 {
            return Err(HipError::new(0, "kvmem_page_mass: empty page domain"));
        }
        if out.len() != samples * pages {
            return Err(HipError::new(
                0,
                &format!(
                    "kvmem_page_mass: out length {} != samples*pages {}",
                    out.len(),
                    samples * pages
                ),
            ));
        }
        let span = pages as i64 * PAGE_TOKENS as i64;
        for s in 0..samples {
            if rows[s] < 0 || rows[s] as u32 >= q.rows {
                return Err(HipError::new(0, "kvmem_page_mass: sampled row out of range"));
            }
            if positions[s] < 0 || positions[s] as i64 >= span {
                return Err(HipError::new(0, "kvmem_page_mass: compact causal end out of range"));
            }
        }
        let distributions = samples as u32 * q.query_heads;
        if let Some(limit) = hip_bridge::grid_yz_limit_for_arch(self.arch.as_str()) {
            if distributions > limit || pages as u32 > limit {
                return Err(HipError::new(
                    0,
                    "kvmem_page_mass: sample/page grid exceeds device limit",
                ));
            }
        }
        for v in out.iter_mut() {
            *v = 0.0;
        }

        self.bind_thread()?;
        self.ensure_kernel(MODULE, KVMEM_PAGE_MASS_SRC, LSE)?;
        self.ensure_kernel(MODULE, KVMEM_PAGE_MASS_SRC, NORMALIZE)?;
        self.ensure_kernel(MODULE, KVMEM_PAGE_MASS_SRC, AVERAGE)?;

        let scratch = self.alloc_tensor(&[pages * distributions as usize * 4], DType::Raw)?;
        let norms = self.alloc_tensor(&[2 * distributions as usize * 4], DType::Raw)?;
        let output = self.zeros(&[pages * samples * 4], DType::Raw)?;
        let rows_dev = self.alloc_tensor(&[samples * 4], DType::Raw)?;
        let pos_dev = self.alloc_tensor(&[samples * 4], DType::Raw)?;
        let table_dev = self.alloc_tensor(&[pages * 4], DType::Raw)?;

        let result = (|| -> HipResult<()> {
            self.hip.memcpy_htod(&rows_dev.buf, bytemuck_i32(rows))?;
            self.hip.memcpy_htod(&pos_dev.buf, bytemuck_i32(positions))?;
            self.hip.memcpy_htod(&table_dev.buf, bytemuck_i32(table))?;

            let scale = if q.attention_scale > 0.0 {
                q.attention_scale
            } else {
                1.0 / (q.head_dim as f32).sqrt()
            };
            let qp = q.data;
            let rp = rows_dev.buf.as_ptr();
            let pp = pos_dev.buf.as_ptr();
            let tp = table_dev.buf.as_ptr();
            let kp = keys;
            let sp = scratch.buf.as_ptr();
            let np = norms.buf.as_ptr();
            let op = output.buf.as_ptr();

            self.mass_lse(
                qp, rp, pp, tp, kp, q.query_heads as i32, q.kv_heads as i32, pages as i32,
                physical_pages as i32, PAGE_TOKENS as i32, q.head_dim as i32, q.scalar as i32,
                scale, sp, distributions,
            )?;
            self.mass_normalize(sp, pages as i32, np, distributions)?;
            self.mass_average(sp, np, pages as i32, q.query_heads as i32, op, samples as u32)?;
            self.hip.device_synchronize()?;

            let mut bytes = vec![0u8; pages * samples * 4];
            self.hip.memcpy_dtoh(&mut bytes, &output.buf)?;
            for (j, chunk) in bytes.chunks_exact(4).enumerate() {
                let v = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !v.is_finite() {
                    return Err(HipError::new(
                        0,
                        "kvmem_page_mass: unmapped/out-of-range visible physical page (NaN mass)",
                    ));
                }
                out[j] = v;
            }
            Ok(())
        })();

        self.free_tensor(scratch)?;
        self.free_tensor(norms)?;
        self.free_tensor(output)?;
        self.free_tensor(rows_dev)?;
        self.free_tensor(pos_dev)?;
        self.free_tensor(table_dev)?;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn mass_lse(
        &mut self,
        q: *const c_void,
        rows: *mut c_void,
        positions: *mut c_void,
        table: *mut c_void,
        keys: *const c_void,
        qheads: i32,
        kvheads: i32,
        pages: i32,
        physical_pages: i32,
        page_tokens: i32,
        dim: i32,
        scalar: i32,
        scale: f32,
        scratch: *mut c_void,
        distributions: u32,
    ) -> HipResult<()> {
        let mut params: Vec<*mut c_void> = vec![
            &q as *const _ as *mut c_void,
            &rows as *const _ as *mut c_void,
            &positions as *const _ as *mut c_void,
            &table as *const _ as *mut c_void,
            &keys as *const _ as *mut c_void,
            &qheads as *const _ as *mut c_void,
            &kvheads as *const _ as *mut c_void,
            &pages as *const _ as *mut c_void,
            &physical_pages as *const _ as *mut c_void,
            &page_tokens as *const _ as *mut c_void,
            &dim as *const _ as *mut c_void,
            &scalar as *const _ as *mut c_void,
            &scale as *const _ as *mut c_void,
            &scratch as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            LSE,
            [pages as u32, distributions, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(q);
                b.push_ptr(rows);
                b.push_ptr(positions);
                b.push_ptr(table);
                b.push_ptr(keys);
                b.push_i32(qheads);
                b.push_i32(kvheads);
                b.push_i32(pages);
                b.push_i32(physical_pages);
                b.push_i32(page_tokens);
                b.push_i32(dim);
                b.push_i32(scalar);
                b.push_f32(scale);
                b.push_ptr(scratch);
                b
            },
        )
    }

    fn mass_normalize(
        &mut self,
        lse: *mut c_void,
        pages: i32,
        norms: *mut c_void,
        distributions: u32,
    ) -> HipResult<()> {
        let mut params: Vec<*mut c_void> = vec![
            &lse as *const _ as *mut c_void,
            &pages as *const _ as *mut c_void,
            &norms as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            NORMALIZE,
            [distributions, 1, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(lse);
                b.push_i32(pages);
                b.push_ptr(norms);
                b
            },
        )
    }

    fn mass_average(
        &mut self,
        lse: *mut c_void,
        norms: *mut c_void,
        pages: i32,
        heads: i32,
        output: *mut c_void,
        samples: u32,
    ) -> HipResult<()> {
        let grid_x = (pages as u32).div_ceil(BLOCK);
        let mut params: Vec<*mut c_void> = vec![
            &lse as *const _ as *mut c_void,
            &norms as *const _ as *mut c_void,
            &pages as *const _ as *mut c_void,
            &heads as *const _ as *mut c_void,
            &output as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            AVERAGE,
            [grid_x, samples, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(lse);
                b.push_ptr(norms);
                b.push_i32(pages);
                b.push_i32(heads);
                b.push_ptr(output);
                b
            },
        )
    }
}

/// Reinterpret an `&[i32]` as little-endian bytes for upload without a bytemuck
/// dependency (the production launcher mirrors the scorer's no-`half` rule).
fn bytemuck_i32(values: &[i32]) -> &[u8] {
    // SAFETY: i32 is plain-old-data; the byte slice aliases the same memory for
    // the lifetime of the borrow, and HIP memcpy reads it synchronously.
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}