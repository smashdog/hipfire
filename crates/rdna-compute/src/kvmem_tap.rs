// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KVMem statistics tap launcher (one launch per FullAttention layer).
//!
//! KVMem's retrieval statistics are stored in the post-RMSNorm / pre-RoPE
//! frame (that is the frame historical K rows were summarized in), but the
//! FA layer body applies RoPE in place to `fa_q_batch` / `fa_k_batch`. This
//! kernel copies those rows out before the rotation, and never touches its
//! inputs — the model arithmetic stays bit-identical to an untapped forward.
//!
//! See `kernels/src/kvmem_tap.hip` for the layout contract and
//! `crates/hipfire-arch-qwen35/src/kvmem_taps.rs` for the owner of the two
//! tap buffers.

use crate::Gpu;
use crate::GpuTensor;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use std::ffi::c_void;

/// Must match `kernels/src/kvmem_tap.hip`.
pub const KVMEM_TAP_SYMBOL: &str = "kvmem_tap_f32";

const TAP_BLOCK: u32 = 256;

impl Gpu {
    /// Copy this step's Q/K rows into the KVMem tap buffers.
    ///
    /// `q` is `[n_rows, n_heads*head_dim]`, `k` is
    /// `[n_rows, n_kv_heads*head_dim]`, both post-norm / pre-RoPE and F32.
    /// K rows land at `k_tap[row]` for every row; Q rows land at
    /// `q_tap[slot]` for each slot's LAST row only (`row_slot` is the
    /// per-row slot id, packed per slot). Neither input is modified.
    #[allow(clippy::too_many_arguments)]
    pub fn kvmem_tap_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        q_tap: &GpuTensor,
        k_tap: &GpuTensor,
        row_slot: &GpuTensor,
        n_rows: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_slots: usize,
    ) -> HipResult<()> {
        if n_rows == 0 || n_slots == 0 {
            return Ok(());
        }
        if let Some(limit) = hip_bridge::grid_yz_limit_for_arch(self.arch.as_str()) {
            if n_rows as u32 > limit {
                return Err(HipError::new(
                    0,
                    &format!(
                        "kvmem_tap_f32: n_rows ({n_rows}) exceeds the device grid-Y limit ({limit})"
                    ),
                ));
            }
        }
        let q_width = n_heads * head_dim;
        let k_width = n_kv_heads * head_dim;
        if q.numel() < n_rows * q_width
            || k.numel() < n_rows * k_width
            || q_tap.numel() < n_slots * q_width
            || k_tap.numel() < n_rows * k_width
            || row_slot.numel() < n_rows
        {
            return Err(HipError::new(
                0,
                &format!(
                    "kvmem_tap_f32: buffer undersized (n_rows={n_rows}, n_slots={n_slots}, \
                     q_width={q_width}, k_width={k_width}; got q={}, k={}, q_tap={}, \
                     k_tap={}, row_slot={})",
                    q.numel(),
                    k.numel(),
                    q_tap.numel(),
                    k_tap.numel(),
                    row_slot.numel()
                ),
            ));
        }
        self.bind_thread()?;
        self.ensure_kernel(
            KVMEM_TAP_SYMBOL,
            crate::kernels::KVMEM_TAP_SRC,
            KVMEM_TAP_SYMBOL,
        )?;
        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut qtp = q_tap.buf.as_ptr();
        let mut ktp = k_tap.buf.as_ptr();
        let mut rsp = row_slot.buf.as_ptr();
        let mut nr = n_rows as i32;
        let mut qw = q_width as i32;
        let mut kw = k_width as i32;
        let mut ns = n_slots as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut qtp as *mut _ as *mut c_void,
            &mut ktp as *mut _ as *mut c_void,
            &mut rsp as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut qw as *mut _ as *mut c_void,
            &mut kw as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
        ];
        let grid_x = ((q_width.max(k_width)) as u32).div_ceil(TAP_BLOCK);
        self.launch_maybe_blob(
            KVMEM_TAP_SYMBOL,
            [grid_x, n_rows as u32, 2],
            [TAP_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(kp);
                b.push_ptr(qtp);
                b.push_ptr(ktp);
                b.push_ptr(rsp);
                b.push_i32(nr);
                b.push_i32(qw);
                b.push_i32(kw);
                b.push_i32(ns);
                b
            },
        )
    }
}
