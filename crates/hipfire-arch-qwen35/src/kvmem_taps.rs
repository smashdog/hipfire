// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KVMem statistics tap buffers for the multi-slot forward.
//!
//! KVMem's retrieval statistics live in the post-RMSNorm / pre-RoPE frame
//! (the frame every historical K row was summarized in). The FullAttention
//! layer body applies RoPE in place to `pbs.fa_q_batch` / `pbs.fa_k_batch`,
//! so the statistics must be copied out between k_norm and RoPE. This module
//! owns those copies:
//!
//! - `k_tap`: every K row of the step, per FullAttention layer, in batch
//!   order. Read back to the host after the forward and fed to the per-slot
//!   key statistics (`write_layer_mean_k`).
//! - `q_tap`: each slot's LAST Q row, per FullAttention layer. Read back on
//!   selection rounds and fed to the retrieval query view.
//!
//! The tap kernel only copies; it never writes Q/K. An enabled tap therefore
//! leaves the model arithmetic bit-identical to an untapped forward.
//!
//! Owned by [`crate::qwen35::batch::PrefillBatchScratch`] as
//! `kvmem_taps: Option<KvMemTaps>`; `None` (the default) means no launch at
//! all. The fields are plain device buffers with no `Drop` — release through
//! `PrefillBatchScratch::free_gpu`, which frees the taps alongside the rest.

use hip_bridge::HipResult;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

/// Per-layer tap buffers. Sized once from the engine's `max_batch` and slot
/// capacity; the per-step row count only bounds what the tap kernel touches.
pub struct KvMemTaps {
    /// FullAttention layer count (`config.layer_types` filter).
    pub n_fa_layers: usize,
    /// Slot capacity of the owning pool (`descriptors().len()`).
    pub n_slots: usize,
    /// Row capacity of `k_tap` (the owning `PrefillBatchScratch::max_batch`).
    pub max_rows: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// `n_heads * head_dim` — one Q row.
    pub q_width: usize,
    /// `n_kv_heads * head_dim` — one K row (the KVMem statistics row width).
    pub k_width: usize,
    /// `[n_fa_layers × n_slots × q_width]` F32.
    pub q_tap: GpuTensor,
    /// `[n_fa_layers × max_rows × k_width]` F32.
    pub k_tap: GpuTensor,
}

impl KvMemTaps {
    /// Allocate the tap buffers. `max_rows` must be `>=` the owning scratch's
    /// `max_batch` row count the engine will forward in one step.
    pub fn new(
        gpu: &mut Gpu,
        n_fa_layers: usize,
        n_slots: usize,
        max_rows: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> HipResult<Self> {
        assert!(
            n_fa_layers > 0 && n_slots > 0 && max_rows > 0 && head_dim > 0,
            "KvMemTaps::new: zero-sized tap request \
             (n_fa_layers={n_fa_layers}, n_slots={n_slots}, max_rows={max_rows}, \
             head_dim={head_dim})"
        );
        let q_width = n_heads * head_dim;
        let k_width = n_kv_heads * head_dim;
        let q_tap = gpu.alloc_tensor(&[n_fa_layers * n_slots * q_width], DType::F32)?;
        let k_tap = gpu.alloc_tensor(&[n_fa_layers * max_rows * k_width], DType::F32)?;
        Ok(Self {
            n_fa_layers,
            n_slots,
            max_rows,
            n_heads,
            n_kv_heads,
            head_dim,
            q_width,
            k_width,
            q_tap,
            k_tap,
        })
    }

    /// Copy this step's Q/K rows of FullAttention layer `layer` into the tap
    /// buffers. Must be called after q_norm/k_norm and BEFORE RoPE — RoPE
    /// rotates both in place.
    ///
    /// `row_slot` is the per-row slot-id device array already uploaded for
    /// this step (`SlotDescStaging::row_slot_dev`); rows are packed per slot
    /// so each slot's last row is identified in-kernel.
    pub fn tap_layer(
        &self,
        gpu: &mut Gpu,
        layer: usize,
        q: &GpuTensor,
        k: &GpuTensor,
        row_slot: &GpuTensor,
        n_rows: usize,
    ) -> HipResult<()> {
        assert!(
            layer < self.n_fa_layers,
            "KvMemTaps::tap_layer: FullAttention layer index {layer} out of \
             range (model has {})",
            self.n_fa_layers
        );
        assert!(
            n_rows <= self.max_rows,
            "KvMemTaps::tap_layer: {n_rows} rows exceed the tap capacity ({})",
            self.max_rows
        );
        let q_layer = self.q_tap.sub_offset(
            layer * self.n_slots * self.q_width,
            self.n_slots * self.q_width,
        );
        let k_layer = self
            .k_tap
            .sub_offset(layer * self.max_rows * self.k_width, n_rows * self.k_width);
        // Diagnostic: HIPFIRE_KVMEM_TAP_COPY=0 skips the copy launch entirely
        // (the tap buffers keep whatever they held). Timing investigations
        // only — never a serving configuration.
        if std::env::var("HIPFIRE_KVMEM_TAP_COPY").map(|v| v == "0").unwrap_or(false) {
            return Ok(());
        }
        let t = std::time::Instant::now();
        let r = gpu.kvmem_tap_f32(
            q,
            k,
            &q_layer,
            &k_layer,
            row_slot,
            n_rows,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.n_slots,
        );
        if std::env::var("HIPFIRE_KVMEM_TIME").map(|v| v == "1").unwrap_or(false) {
            eprintln!(
                "[kvmem-time] tap layer {layer} {n_rows} rows launch {:.2} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        r
    }

    /// Read back the first `n_rows` K rows of FullAttention layer `layer`
    /// (D2H, synchronous). The rows are in batch order, `k_width` floats
    /// each — exactly what `write_layer_mean_k` consumes.
    pub fn read_k_rows(
        &self,
        gpu: &Gpu,
        layer: usize,
        n_rows: usize,
        out: &mut Vec<f32>,
    ) -> HipResult<()> {
        assert!(
            layer < self.n_fa_layers && n_rows <= self.max_rows,
            "KvMemTaps::read_k_rows: layer {layer}/{}, rows {n_rows}/{} out of range",
            self.n_fa_layers,
            self.max_rows
        );
        read_f32_at(
            gpu,
            &self.k_tap,
            layer * self.max_rows * self.k_width,
            n_rows * self.k_width,
            out,
        )
    }

    /// Read back `n_rows` K rows of FullAttention layer `layer` starting at
    /// batch row `row_start` (D2H, synchronous). Spec verify rows belong to a
    /// slot at a non-zero batch offset, so the commit path reads with one.
    pub fn read_k_rows_at(
        &self,
        gpu: &Gpu,
        layer: usize,
        row_start: usize,
        n_rows: usize,
        out: &mut Vec<f32>,
    ) -> HipResult<()> {
        assert!(
            layer < self.n_fa_layers
                && row_start.saturating_add(n_rows) <= self.max_rows,
            "KvMemTaps::read_k_rows_at: layer {layer}/{}, rows \
             {row_start}..{}/{n_rows}..{} out of range",
            self.n_fa_layers,
            row_start.saturating_add(n_rows),
            self.max_rows
        );
        read_f32_at(
            gpu,
            &self.k_tap,
            layer * self.max_rows * self.k_width + row_start * self.k_width,
            n_rows * self.k_width,
            out,
        )
    }

    /// Read back one slot's tapped Q row of FullAttention layer `layer`
    /// (D2H, synchronous), `q_width` floats.
    pub fn read_q_slot(
        &self,
        gpu: &Gpu,
        layer: usize,
        slot: usize,
        out: &mut Vec<f32>,
    ) -> HipResult<()> {
        assert!(
            layer < self.n_fa_layers && slot < self.n_slots,
            "KvMemTaps::read_q_slot: layer {layer}/{}, slot {slot}/{} out of range",
            self.n_fa_layers,
            self.n_slots
        );
        read_f32_at(
            gpu,
            &self.q_tap,
            (layer * self.n_slots + slot) * self.q_width,
            self.q_width,
            out,
        )
    }
}

/// Synchronous F32 D2H of `len` elements starting at `elem_offset`.
fn read_f32_at(
    gpu: &Gpu,
    buf: &GpuTensor,
    elem_offset: usize,
    len: usize,
    out: &mut Vec<f32>,
) -> HipResult<()> {
    out.clear();
    out.resize(len, 0.0);
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr() as *mut u8,
            len * std::mem::size_of::<f32>(),
        )
    };
    gpu.hip
        .memcpy_dtoh_at(bytes, &buf.buf, elem_offset * std::mem::size_of::<f32>())
}
