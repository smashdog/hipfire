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
//!   selection rounds and fed to the retrieval query view (legacy behavior).
//! - optional `query_capture`: explicitly selected batch Q rows, in caller
//!   order, BF16 [layer, capacity, q_width]. No Query detection or accumulation
//!   happens here; the scheduler owns Query spans and accepted-row publication.
//!
//! The tap kernel only copies; it never writes Q/K. An enabled tap therefore
//! leaves the model arithmetic bit-identical to an untapped forward.
//!
//! Owned by [`crate::qwen35::batch::PrefillBatchScratch`] as
//! `kvmem_taps: Option<KvMemTaps>`; `None` (the default) means no launch at
//! all. The fields are plain device buffers with no `Drop` — release through
//! `PrefillBatchScratch::free_gpu`, which frees the taps alongside the rest.

use hip_bridge::HipError;
use hip_bridge::HipResult;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Device buffers are released by `PrefillBatchScratch::free_gpu`, like the
/// old taps. Metadata/storage are allocated only when capture is enabled.
pub(crate) struct QueryCapture {
    pub(crate) values: GpuTensor,
    pub(crate) source_rows: GpuTensor,
    capacity: usize,
    row_count: usize,
    required_rows: usize,
    // Keep tap_layer(&self): existing forward closures borrow the taps shared.
    captured_rows: Vec<AtomicUsize>,
    readback: Mutex<Vec<u16>>,
}

impl QueryCapture {
    fn invalidate(&mut self) {
        self.row_count = 0;
        self.required_rows = 0;
        for rows in &self.captured_rows {
            rows.store(0, Ordering::Relaxed);
        }
    }
}

fn query_error(message: &str) -> HipError {
    HipError::new(1, &format!("KvMemTaps Query capture: {message}"))
}

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
    /// None by default: old Q/K taps and Layerwise behavior are unchanged.
    pub(crate) query_capture: Option<QueryCapture>,
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
            query_capture: None,
        })
    }

    /// Pre-allocation admission cost, including the exact allocator's 256-byte
    /// minimum for each of the two buffers. Independent of slots and logical C.
    pub fn query_capture_capacity_bytes(&self, max_query_rows: usize) -> HipResult<usize> {
        rdna_compute::kvmem_tap::query_capture_capacity_bytes(
            self.n_fa_layers,
            self.max_rows,
            self.q_width,
            max_query_rows,
        )
    }

    /// Actual device bytes owned by Query capture only (zero when disabled).
    /// `free_tensor` returns these allocations to the pool at scratch teardown;
    /// physical release follows the pool's existing drain lifecycle.
    pub fn query_capacity_bytes(&self) -> usize {
        self.query_capture.as_ref().map_or(0, |capture| {
            capture.values.buf.size() + capture.source_rows.buf.size()
        })
    }

    /// Allocate once at load. The same capacity is idempotent; resizing is an
    /// error, never an implicit per-forward reallocation. Both allocations are
    /// exact-sized so admission is not defeated by a larger pooled allocation.
    pub fn enable_query_capture(&mut self, gpu: &mut Gpu, max_query_rows: usize) -> HipResult<()> {
        self.query_capture_capacity_bytes(max_query_rows)?;
        if let Some(capture) = &self.query_capture {
            return if capture.capacity == max_query_rows {
                Ok(())
            } else {
                Err(query_error("capture already enabled with a different capacity"))
            };
        }
        if gpu.graphs.capture_mode || gpu.replay.is_recording() {
            return Err(query_error("enable capture before graph capture/forward"));
        }
        let layer_elements = max_query_rows * self.q_width;
        let mut readback = Vec::new();
        readback.try_reserve_exact(layer_elements).map_err(|e| {
            HipError::new(2, &format!("Query capture host readback allocation: {e}"))
        })?;
        readback.resize(layer_elements, 0u16);
        let mut captured_rows = Vec::new();
        captured_rows.try_reserve_exact(self.n_fa_layers).map_err(|e| {
            HipError::new(2, &format!("Query capture layer metadata allocation: {e}"))
        })?;
        captured_rows.resize_with(self.n_fa_layers, || AtomicUsize::new(0));
        let values = gpu.alloc_tensor_exact(
            &[self.n_fa_layers * layer_elements],
            DType::BF16,
        )?;
        let source_rows = match gpu.alloc_tensor_exact(
            &[max_query_rows * std::mem::size_of::<u32>()],
            DType::Raw,
        ) {
            Ok(rows) => rows,
            Err(e) => {
                // The first allocation has no queued users yet.
                let _ = gpu.free_tensor(values);
                return Err(e);
            }
        };
        self.query_capture = Some(QueryCapture {
            values,
            source_rows,
            capacity: max_query_rows,
            row_count: 0,
            required_rows: 0,
            captured_rows,
            readback: Mutex::new(readback),
        });
        Ok(())
    }

    /// Configure *this step* before forward, on its active stream. Indices are
    /// packed batch rows, not slot-relative offsets. Call for every forward,
    /// including `&[]` on non-Query steps; no last-N or slot-tail inference.
    ///
    /// Even invalid input/upload failure clears the old mask and all readable
    /// layer ranges. H2D completes before return, so the caller's slice need not
    /// outlive this call. Nonempty Query capture requires eager execution: a
    /// retained graph's fixed host row count must never replay a stale mask.
    pub fn set_query_rows(&mut self, gpu: &mut Gpu, source_rows: &[u32]) -> HipResult<()> {
        let Some(capture) = self.query_capture.as_mut() else {
            return if source_rows.is_empty() {
                Ok(())
            } else {
                Err(query_error("capture is not enabled"))
            };
        };
        capture.invalidate();
        rdna_compute::kvmem_tap::validate_query_source_rows(
            source_rows,
            self.max_rows,
            capture.capacity,
        )?;
        if source_rows.is_empty() {
            return Ok(());
        }
        if gpu.graphs.capture_mode || gpu.replay.is_recording() {
            return Err(query_error("set_query_rows must run before graph capture/forward"));
        }
        // SAFETY: u32 has no padding; bytes borrow the input only until the
        // stream-ordered blocking upload returns. Graph capture was refused.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                source_rows.as_ptr().cast::<u8>(),
                std::mem::size_of_val(source_rows),
            )
        };
        gpu.htod_stream_ordered(&capture.source_rows.buf, 0, bytes)?;
        capture.required_rows = source_rows.iter().copied().max().unwrap_or(0) as usize + 1;
        capture.row_count = source_rows.len();
        Ok(())
    }

    /// Read compact captured rows, NOT batch row offsets, of one FA layer.
    /// Only successfully enqueued rows from the latest `set_query_rows` are
    /// readable; D2H waits on the same active stream as forward. BF16 expands
    /// with Hipfire's existing primitive, matching NInfer's represented-value
    /// readback. Non-finite Query values are errors, never silently sanitized.
    pub fn read_query_rows(
        &self,
        gpu: &Gpu,
        layer: usize,
        row_start: usize,
        n_rows: usize,
        out: &mut Vec<f32>,
    ) -> HipResult<()> {
        out.clear();
        let capture = self.query_capture.as_ref()
            .ok_or_else(|| query_error("capture is not enabled"))?;
        let valid = capture.captured_rows.get(layer)
            .ok_or_else(|| query_error("FullAttention layer is out of range"))?
            .load(Ordering::Relaxed);
        if row_start.checked_add(n_rows).is_none_or(|end| end > valid) {
            return Err(query_error("read exceeds this step's captured row range"));
        }
        if n_rows == 0 {
            return Ok(());
        }
        if gpu.graphs.capture_mode || gpu.replay.is_recording() {
            return Err(query_error("readback is not allowed during graph capture"));
        }
        let elements = n_rows * self.q_width;
        let offset = (layer * capture.capacity + row_start) * self.q_width;
        let mut host = capture.readback.lock()
            .map_err(|_| query_error("host readback lock is poisoned"))?;
        gpu.bind_thread()?;
        // SAFETY: initialized u16 storage sized once to capacity * q_width;
        // every bit pattern is valid, and the stream is drained before reuse.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), elements * 2)
        };
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                gpu.hip.memcpy_dtoh_async_at(bytes, &capture.values.buf, offset * 2, Some(stream))?;
                gpu.hip.stream_synchronize(stream)?;
            }
            None => gpu.hip.memcpy_dtoh_at(bytes, &capture.values.buf, offset * 2)?,
        }
        gpu.take_device_wait_error()?;
        out.try_reserve(elements).map_err(|e| {
            HipError::new(2, &format!("Query capture FP32 output allocation: {e}"))
        })?;
        for &bits in &host[..elements] {
            let value = rdna_compute::fp8_e4m3::bf16_to_f32(bits);
            if !value.is_finite() {
                out.clear();
                return Err(query_error("non-finite captured Query value"));
            }
            out.push(value);
        }
        Ok(())
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
        if layer >= self.n_fa_layers || n_rows > self.max_rows {
            return Err(query_error("tap layer or batch row count is out of range"));
        }
        if let Some(capture) = &self.query_capture {
            capture.captured_rows[layer].store(0, Ordering::Relaxed);
            if capture.required_rows > n_rows {
                return Err(query_error("selected source row is outside the actual forward batch"));
            }
            if capture.row_count > 0 && (gpu.graphs.capture_mode || gpu.replay.is_recording()) {
                return Err(query_error("nonempty Query capture requires an eager forward"));
            }
        }
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
        if hipfire_config::developer_var("HIPFIRE_KVMEM_TAP_COPY").as_deref() == Ok("0") {
            if self.query_capture.as_ref().is_some_and(|capture| capture.row_count > 0) {
                return Err(query_error("Query capture cannot run with tap copies disabled"));
            }
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
        if hipfire_config::developer_bool("HIPFIRE_KVMEM_TIME", false) {
            eprintln!(
                "[kvmem-time] tap layer {layer} {n_rows} rows launch {:.2} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        r?;
        if let Some(capture) = &self.query_capture {
            if capture.row_count > 0 {
                let layer_elements = capture.capacity * self.q_width;
                let query_layer = capture.values.sub_offset(layer * layer_elements, layer_elements);
                gpu.kvmem_capture_query_bf16(
                    q,
                    &capture.source_rows,
                    &query_layer,
                    n_rows,
                    capture.row_count,
                    self.q_width,
                )?;
                capture.captured_rows[layer].store(capture.row_count, Ordering::Relaxed);
            }
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The owner, not the primitive, protects current-step valid ranges. This
    /// also checks that the optional capture never replaces the old Q-slot tap.
    #[test]
    #[ignore = "needs a GPU and a HIP toolchain"]
    fn query_capture_owner_step_lifecycle() {
        let mut gpu = Gpu::init().expect("gpu init");
        gpu.active_stream = Some(gpu.hip.stream_create_non_blocking().unwrap());
        let mut taps = KvMemTaps::new(&mut gpu, 2, 3, 7, 1, 1, 3).unwrap();
        assert!(taps.query_capture.is_none());
        assert_eq!(taps.query_capacity_bytes(), 0);
        taps.set_query_rows(&mut gpu, &[]).unwrap();
        assert_eq!(taps.set_query_rows(&mut gpu, &[0]).unwrap_err().code, 1);
        taps.enable_query_capture(&mut gpu, 3).unwrap();
        assert_eq!(taps.query_capacity_bytes(), taps.query_capture_capacity_bytes(3).unwrap());
        let pointers = {
            let capture = taps.query_capture.as_ref().unwrap();
            (capture.values.buf.as_ptr(), capture.source_rows.buf.as_ptr())
        };
        taps.enable_query_capture(&mut gpu, 3).unwrap();
        assert_eq!(taps.enable_query_capture(&mut gpu, 2).unwrap_err().code, 1);
        let input: Vec<f32> = (0..21).map(|n| n as f32 + 0.25).collect();
        let q = gpu.upload_f32(&input, &[21]).unwrap();
        let k = gpu.upload_f32(&input, &[21]).unwrap();
        let slots = gpu.alloc_tensor(&[7], DType::F32).unwrap();
        let slot_bytes: Vec<u8> = [0i32, 0, 1, 1, 1, 2, 2].iter()
            .flat_map(|n| n.to_ne_bytes()).collect();
        gpu.htod_stream_ordered(&slots.buf, 0, &slot_bytes).unwrap();
        let mut out = Vec::new();
        taps.set_query_rows(&mut gpu, &[5, 0, 3]).unwrap();
        assert_eq!(taps.read_query_rows(&gpu, 0, 0, 1, &mut out).unwrap_err().code, 1);
        taps.tap_layer(&mut gpu, 0, &q, &k, &slots, 7).unwrap();
        taps.read_query_rows(&gpu, 0, 1, 2, &mut out).unwrap();
        assert_eq!(out, [&input[0..3], &input[9..12]].concat());
        assert_eq!(taps.read_query_rows(&gpu, 1, 0, 1, &mut out).unwrap_err().code, 1);
        assert_eq!(taps.read_query_rows(&gpu, 0, 3, 1, &mut out).unwrap_err().code, 1);
        assert_eq!(taps.read_query_rows(&gpu, 0, usize::MAX, 1, &mut out).unwrap_err().code, 1);
        taps.tap_layer(&mut gpu, 1, &q, &k, &slots, 7).unwrap();
        taps.read_query_rows(&gpu, 1, 0, 1, &mut out).unwrap();
        assert_eq!(out, input[15..18]);
        // read_query_rows waited on the nonblocking stream before oldtap D2H.
        taps.read_q_slot(&gpu, 0, 2, &mut out).unwrap();
        assert_eq!(out, input[18..21], "legacy slot-tail tap must remain unchanged");
        for invalid in [&[1, 1][..], &[7][..], &[0, 1, 2, 3][..]] {
            assert_eq!(taps.set_query_rows(&mut gpu, invalid).unwrap_err().code, 1);
            assert_eq!(taps.read_query_rows(&gpu, 0, 0, 1, &mut out).unwrap_err().code, 1);
            assert_eq!(taps.query_capture.as_ref().unwrap().row_count, 0);
        }
        // Legal against max_rows, but not against this smaller actual forward.
        taps.set_query_rows(&mut gpu, &[6]).unwrap();
        assert_eq!(taps.tap_layer(&mut gpu, 0, &q, &k, &slots, 6).unwrap_err().code, 1);
        assert_eq!(taps.read_query_rows(&gpu, 0, 0, 1, &mut out).unwrap_err().code, 1);
        taps.set_query_rows(&mut gpu, &[2]).unwrap();
        taps.tap_layer(&mut gpu, 0, &q, &k, &slots, 7).unwrap();
        taps.read_query_rows(&gpu, 0, 0, 1, &mut out).unwrap();
        assert_eq!(out, input[6..9]);
        assert_eq!(taps.read_query_rows(&gpu, 0, 1, 1, &mut out).unwrap_err().code, 1);
        taps.set_query_rows(&mut gpu, &[]).unwrap();
        taps.tap_layer(&mut gpu, 0, &q, &k, &slots, 7).unwrap();
        assert_eq!(taps.read_query_rows(&gpu, 0, 0, 1, &mut out).unwrap_err().code, 1);
        taps.read_query_rows(&gpu, 0, 0, 0, &mut out).unwrap();
        assert!(out.is_empty());
        // A poisoned selected row is reported, not turned into a zero vector.
        let poison: Vec<u8> = [f32::NAN; 21].iter().flat_map(|n| n.to_ne_bytes()).collect();
        gpu.htod_stream_ordered(&q.buf, 0, &poison).unwrap();
        taps.set_query_rows(&mut gpu, &[0]).unwrap();
        taps.tap_layer(&mut gpu, 0, &q, &k, &slots, 7).unwrap();
        assert_eq!(taps.read_query_rows(&gpu, 0, 0, 1, &mut out).unwrap_err().code, 1);
        assert!(out.is_empty());
        let capture = taps.query_capture.unwrap();
        assert_eq!((capture.values.buf.as_ptr(), capture.source_rows.buf.as_ptr()), pointers);
        gpu.free_tensor(capture.values).unwrap();
        gpu.free_tensor(capture.source_rows).unwrap();
        gpu.free_tensor(taps.q_tap).unwrap();
        gpu.free_tensor(taps.k_tap).unwrap();
        gpu.free_tensor(q).unwrap();
        gpu.free_tensor(k).unwrap();
        gpu.free_tensor(slots).unwrap();
        let stream = gpu.active_stream.take().unwrap();
        gpu.hip.stream_destroy(stream).unwrap();
    }
}
