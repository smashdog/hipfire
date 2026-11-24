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

use crate::DType;
use crate::Gpu;
use crate::GpuTensor;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use std::ffi::c_void;

/// Must match `kernels/src/kvmem_tap.hip`.
pub const KVMEM_TAP_SYMBOL: &str = "kvmem_tap_f32";

/// Explicit-row F32 -> BF16 Query gather; independent of the legacy slot-tail tap.
pub const KVMEM_QUERY_CAPTURE_SYMBOL: &str = "kvmem_capture_query_bf16";

const TAP_BLOCK: u32 = 256;

/// Query API validation errors use hipErrorInvalidValue, never a panic or code 0.
fn query_error(message: &str) -> HipError {
    HipError::new(1, &format!("KVMem Query capture: {message}"))
}

/// Exact device allocation request for the optional capture, including the
/// 256-byte minimum of each `alloc_tensor_exact` allocation. No slot or logical
/// context multiplier: BF16 [layers, capacity, q_width] + raw u32 [capacity].
pub fn query_capture_capacity_bytes(
    layers: usize,
    max_rows: usize,
    q_width: usize,
    capacity: usize,
) -> HipResult<usize> {
    if layers == 0
        || q_width == 0
        || q_width > i32::MAX as usize
        || max_rows > i32::MAX as usize
        || capacity == 0
        || capacity > max_rows
    {
        return Err(query_error("invalid layers, width, max_rows or capacity"));
    }
    let values = layers
        .checked_mul(capacity)
        .and_then(|n| n.checked_mul(q_width))
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| query_error("BF16 allocation size overflow"))?;
    let indices = capacity
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| query_error("source-row allocation size overflow"))?;
    values
        .max(256)
        .checked_add(indices.max(256))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| query_error("capture allocation exceeds addressable size"))
}

/// Validate caller-selected batch rows without sorting, clipping, or allocating.
/// Ordering is meaningful; duplicates must not count a Query token twice.
pub fn validate_query_source_rows(
    source_rows: &[u32],
    max_rows: usize,
    capacity: usize,
) -> HipResult<()> {
    if source_rows.len() > capacity {
        return Err(query_error("source-row count exceeds capture capacity"));
    }
    for (i, &row) in source_rows.iter().enumerate() {
        if row as usize >= max_rows {
            return Err(query_error(&format!(
                "source row {row} is outside {max_rows} rows"
            )));
        }
        if source_rows[..i].contains(&row) {
            return Err(query_error(&format!("duplicate source row {row}")));
        }
    }
    Ok(())
}

fn query_buffer_extent(tensor: &GpuTensor, dtype: DType, bytes: usize) -> HipResult<()> {
    let logical_bytes = tensor
        .shape
        .iter()
        .try_fold(dtype.size(), |n, &dim| n.checked_mul(dim));
    if tensor.dtype != dtype
        || logical_bytes.is_none_or(|n| n < bytes)
        || tensor.buf.size() < bytes
        || tensor.buf.as_ptr().is_null()
    {
        return Err(query_error("invalid dtype or undersized/null buffer"));
    }
    Ok(())
}

fn query_buffers_overlap(a: &GpuTensor, a_bytes: usize, b: &GpuTensor, b_bytes: usize) -> bool {
    let a_start = a.buf.as_ptr() as usize;
    let b_start = b.buf.as_ptr() as usize;
    match (a_start.checked_add(a_bytes), b_start.checked_add(b_bytes)) {
        (Some(a_end), Some(b_end)) => a_start < b_end && b_start < a_end,
        _ => true,
    }
}

impl Gpu {
    /// Gather explicitly selected Q rows in table order, converting only the
    /// stored copy with HIP's `hip_bfloat16` RNE primitive. Inputs are read-only.
    /// `source_rows` is Raw storage containing native u32 batch indices, each
    /// `< n_rows`; the owner must validate their values before uploading them.
    /// The destination is BF16 [capacity, q_width], with only the first
    /// `n_query_rows` rows written. An empty selection launches nothing.
    pub fn kvmem_capture_query_bf16(
        &mut self,
        q: &GpuTensor,
        source_rows: &GpuTensor,
        query_rows: &GpuTensor,
        n_rows: usize,
        n_query_rows: usize,
        q_width: usize,
    ) -> HipResult<()> {
        if n_query_rows == 0 {
            return Ok(());
        }
        if n_rows == 0
            || n_rows > i32::MAX as usize
            || n_query_rows > n_rows
            || q_width == 0
            || q_width > i32::MAX as usize
        {
            return Err(query_error("invalid gather dimensions"));
        }
        if hip_bridge::grid_yz_limit_for_arch(self.arch.as_str())
            .is_some_and(|limit| n_query_rows > limit as usize)
        {
            return Err(query_error("selected rows exceed device grid-Y limit"));
        }
        let q_bytes = n_rows.checked_mul(q_width).and_then(|n| n.checked_mul(4));
        let out_bytes = n_query_rows
            .checked_mul(q_width)
            .and_then(|n| n.checked_mul(2));
        let row_bytes = n_query_rows.checked_mul(std::mem::size_of::<u32>());
        let (Some(q_bytes), Some(out_bytes), Some(row_bytes)) = (q_bytes, out_bytes, row_bytes)
        else {
            return Err(query_error("gather extent overflow"));
        };
        query_buffer_extent(q, DType::F32, q_bytes)?;
        query_buffer_extent(source_rows, DType::Raw, row_bytes)?;
        query_buffer_extent(query_rows, DType::BF16, out_bytes)?;
        if query_buffers_overlap(q, q_bytes, query_rows, out_bytes)
            || query_buffers_overlap(source_rows, row_bytes, query_rows, out_bytes)
            || query_buffers_overlap(q, q_bytes, source_rows, row_bytes)
        {
            return Err(query_error("gather buffers must not alias"));
        }
        self.bind_thread()?;
        self.ensure_kernel(
            KVMEM_QUERY_CAPTURE_SYMBOL,
            crate::kernels::KVMEM_TAP_SRC,
            KVMEM_QUERY_CAPTURE_SYMBOL,
        )?;
        let mut qp = q.buf.as_ptr();
        let mut rp = source_rows.buf.as_ptr();
        let mut op = query_rows.buf.as_ptr();
        let mut nr = n_rows as i32;
        let mut nq = n_query_rows as i32;
        let mut qw = q_width as i32;
        let mut params: [*mut c_void; 6] = [
            &mut qp as *mut _ as *mut c_void,
            &mut rp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut qw as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KVMEM_QUERY_CAPTURE_SYMBOL,
            [(q_width as u32).div_ceil(TAP_BLOCK), n_query_rows as u32, 1],
            [TAP_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(qp);
                b.push_ptr(rp);
                b.push_ptr(op);
                b.push_i32(nr);
                b.push_i32(nq);
                b.push_i32(qw);
                b
            },
        )
    }

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
