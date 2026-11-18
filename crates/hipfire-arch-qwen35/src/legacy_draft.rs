// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt and hipfire contributors

//! Raw host images of Legacy (absolute-position, full-resident) draft history.
//!
//! These are NOT a scheduler, a prefix proof, or an in-flight request image.
//! The caller must bind the image to the same loaded target/draft weights,
//! execution configuration, token history and target KV/DeltaNet frontier.
//! Geometry alone does not identify a model. In particular, a bounded Layerwise
//! head staging buffer cannot be admitted just because its prefix fits: the
//! caller must prove Legacy addressing, with no external head position shift.
//!
//! Capture/restore must run at a committed, idle boundary, with no pending MTP
//! proposal/verify output in the engine and no concurrent owner of the buffers.
//! MtpSpecState has no in-flight marker; that gate belongs to its caller. The
//! DFlash wrapper additionally checks its native `draft` marker. Missing source
//! state is an error at the call site, never an empty image or a switch to AR.
//!
//! MtpSpecState field audit:
//! * History: `prev_hidden` + its position; head K/V (including inline quant
//!   scales), rotation tables, and the validated single-layer cache metadata.
//! * Request-local: `sampling`, `rng`, `gpu_rng_state`, `penalty`,
//!   `penalty_timing`, and the request's `p_min` policy. These are NOT migrated.
//!   Install the destination request's sampling/seed/history before execution;
//!   resuming an interrupted request needs a separate sampler/penalty image.
//! * Scratch: all verify/proposal/logit/argmax/token/gather/repeat buffers,
//!   `mtp_scratch` (including t_mtp_out), takeover fill buffers, `trunk_pbs`,
//!   `trunk_snap`, and `trunk_gdn_tape`. Their contents are per-window outputs,
//!   not the accepted target state (the caller snapshots live DeltaNet instead).
//! * Infrastructure: stream/event handles, max_n/verify_capacity, lazy scratch
//!   allocations and graph diagnostics stay with the destination. Recorded MTP
//!   graphs and their warm-position identity are invalidated, never migrated.
//!
//! Restore validates the entire image and destination before any device write.
//! An upload failure is returned, not converted to a fallback success; the caller
//! must invalidate the combined target/draft restore on ANY error. Images remain
//! reusable. Copies are native bytes, never dequantized or numerically converted.

use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu, GpuTensor};
use saddle_core::kv::{KvCache, KvMode, SlotKvTierPlan, VMode};

use crate::mtp_head::MtpKvMode;
use crate::mtp_spec::MtpSpecState;

fn product(label: &str, factors: &[usize]) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |n, &f| {
        n.checked_mul(f)
            .ok_or_else(|| format!("native draft image: {label} overflow"))
    })
}

fn check_len(label: &str, actual: usize, expected: usize) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "native draft image: {label} length {actual} != {expected}"
        ));
    }
    Ok(())
}

/// Check tensor metadata without invoking numel()'s unchecked product. Packed
/// head caches deliberately use F32 *containers*; their payload is not F32.
fn check_f32_container(t: &GpuTensor, elems: usize, label: &str) -> Result<(), String> {
    let bytes = product(label, &[elems, 4])?;
    if t.dtype != DType::F32 || t.shape.as_slice() != [elems] {
        return Err(format!(
            "native draft image: {label} tensor shape/dtype mismatch"
        ));
    }
    if !t.buf.is_hip_allocation() || t.buf.as_ptr().is_null() || t.buf.size() < bytes {
        return Err(format!(
            "native draft image: {label} is not a resident owning buffer"
        ));
    }
    Ok(())
}

fn quiesce(gpu: &Gpu) -> Result<(), String> {
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return Err("native draft image: recording is in progress".into());
    }
    gpu.bind_thread()
        .map_err(|e| format!("native draft image bind: {e}"))?;
    // A default-stream memcpy alone does not drain a nonblocking side stream.
    gpu.hip
        .device_synchronize()
        .map_err(|e| format!("native draft image drain: {e}"))
}

fn grab(gpu: &Gpu, buf: &DeviceBuffer, len: usize) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0; len];
    if len != 0 {
        gpu.hip
            .memcpy_dtoh(&mut bytes, buf)
            .map_err(|e| format!("native draft image D2H: {e}"))?;
    }
    Ok(bytes)
}

fn put(gpu: &Gpu, buf: &DeviceBuffer, bytes: &[u8]) -> Result<(), String> {
    if !bytes.is_empty() {
        gpu.hip
            .memcpy_htod(buf, bytes)
            .map_err(|e| format!("native draft image H2D: {e}"))?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MtpLayout {
    mode: MtpKvMode,
    max_seq: usize,
    n_head_kv: usize,
    head_dim: usize,
    hidden_dim: usize,
    k_stride: usize,
    v_stride: usize,
    rotation_elems: usize,
}

impl MtpLayout {
    fn new(
        mode: MtpKvMode,
        max_seq: usize,
        n_head_kv: usize,
        head_dim: usize,
        hidden_dim: usize,
    ) -> Result<Self, String> {
        if max_seq == 0 || hidden_dim == 0 {
            return Err("MTP image: zero capacity/hidden dimension".into());
        }
        // Guard even the unchecked intermediate products in the shared helper.
        product("head geometry", &[n_head_kv, head_dim, 34])?;
        let plan = SlotKvTierPlan::resolve(Self::kv_mode(mode), n_head_kv, head_dim)
            .map_err(|e| format!("MTP image layout: {e}"))?;
        let layout = Self {
            mode,
            max_seq,
            n_head_kv,
            head_dim,
            hidden_dim,
            k_stride: plan.k_bytes_per_pos,
            v_stride: plan.v_bytes_per_pos,
            rotation_elems: plan.givens_len.or(plan.fwht_len).unwrap_or(0),
        };
        layout.cache_elems(layout.k_stride)?;
        layout.cache_elems(layout.v_stride)?;
        product("hidden", &[hidden_dim, 4])?;
        Ok(layout)
    }

    fn kv_mode(mode: MtpKvMode) -> KvMode {
        match mode {
            MtpKvMode::Q8 => KvMode::Q8,
            MtpKvMode::Asym3 => KvMode::Asym3,
            MtpKvMode::Fwht4 => KvMode::Fwht4,
        }
    }

    fn cache_elems(&self, stride: usize) -> Result<usize, String> {
        product("head allocation", &[self.max_seq, stride])?
            .checked_add(3)
            .map(|n| n / 4)
            .ok_or_else(|| "MTP image: head allocation rounding overflow".into())
    }

    fn payload_lengths(
        &self,
        frontier: usize,
        hidden_pos: Option<usize>,
    ) -> Result<[usize; 5], String> {
        if frontier > self.max_seq
            || frontier > i32::MAX as usize
            || hidden_pos != frontier.checked_sub(1)
        {
            return Err(format!(
                "MTP image: frontier {frontier}, capacity {}, prev_hidden_pos {hidden_pos:?} disagree",
                self.max_seq
            ));
        }
        let rotation = product("rotation", &[self.rotation_elems, 4])?;
        Ok([
            product("K prefix", &[frontier, self.k_stride])?,
            product("V prefix", &[frontier, self.v_stride])?,
            if frontier == 0 {
                0
            } else {
                product("hidden", &[self.hidden_dim, 4])?
            },
            rotation,
            rotation,
        ])
    }

    fn for_state(state: &MtpSpecState) -> Result<Self, String> {
        let head = &state.mtp_kv;
        let kv = &head.inner;
        let hidden_dim = match state.prev_hidden.shape.as_slice() {
            [n] => *n,
            _ => return Err("MTP image: prev_hidden must be a vector".into()),
        };
        let layout = Self::new(
            head.kv_mode,
            head.max_seq,
            head.n_head_kv,
            head.head_dim,
            hidden_dim,
        )?;
        let flags = (
            kv.quant_q8,
            kv.quant_asym4,
            kv.quant_asym3,
            kv.quant_asym2,
            kv.quant_fwht,
        );
        // Exactly the metadata produced by Qwen35MtpHeadKvCache's three native
        // constructors. In particular these are NOT INT8 + separate scales,
        // adaptive/boundary tiers, VMM aliases, or compacted/shifted head rows.
        if flags != KvCache::vmm_mode_flags(Self::kv_mode(head.kv_mode))
            || !kv.quantized
            || kv.quant_int8
            || kv.quant_hfq4
            || kv.quant_bf16
            || kv.quant_fp8
            || kv.v_mode != VMode::Q8
            || kv.boundary_layers != 0
            || !kv.layer_is_boundary.is_empty()
            || kv.compact_offset != 0
            || kv.max_seq != head.max_seq
            || kv.physical_cap != head.max_seq
            || kv.n_kv_heads != head.n_head_kv
            || kv.head_dim != head.head_dim
            || kv.kv_dim != product("KV dimension", &[head.n_head_kv, head.head_dim])?
            || kv.k_gpu.len() != 1
            || kv.v_gpu.len() != 1
            || !kv.k_scales.is_empty()
            || !kv.v_scales.is_empty()
        {
            return Err(
                "MTP image: unsupported or inconsistent single-layer native KV metadata".into(),
            );
        }
        check_f32_container(&kv.k_gpu[0], layout.cache_elems(layout.k_stride)?, "MTP K")?;
        check_f32_container(&kv.v_gpu[0], layout.cache_elems(layout.v_stride)?, "MTP V")?;
        check_f32_container(&state.prev_hidden, hidden_dim, "prev_hidden")?;
        match (&kv.givens_cos, &kv.givens_sin, layout.rotation_elems) {
            (None, None, 0) => {}
            (Some(a), Some(b), n) if n != 0 => {
                check_f32_container(a, n, "rotation table 1")?;
                check_f32_container(b, n, "rotation table 2")?;
            }
            _ => return Err("MTP image: rotation table presence mismatch".into()),
        }
        if state.max_n == 0 || state.verify_capacity < state.max_n {
            return Err("MTP image: invalid destination proposal/verify capacity".into());
        }
        check_f32_container(&state.mtp_positions, state.max_n, "MTP positions")?;
        if state.mtp_scratch.pos_buf.size() < 4
            || !state.mtp_scratch.pos_buf.is_hip_allocation()
            || state.mtp_scratch.pos_buf.as_ptr().is_null()
        {
            return Err("MTP image: invalid head position buffer".into());
        }
        Ok(layout)
    }
}

/// An immutable native MTP accepted-boundary image. The layout encodes all
/// admitted layer metadata: one non-boundary layer, zero compact offset, Q8 V,
/// no separate scale planes, and the mode's exact quantization flags. Restore
/// requires that metadata unchanged, rather than silently rewriting the format.
#[derive(Clone, Debug)]
pub struct LegacyMtpImage {
    layout: MtpLayout,
    frontier: usize,
    prev_hidden_pos: Option<usize>,
    // K, V, prev_hidden, rotation table 1, rotation table 2 (native bytes).
    blobs: [Vec<u8>; 5],
}

impl LegacyMtpImage {
    pub fn capture(gpu: &Gpu, state: &MtpSpecState, frontier: usize) -> Result<Self, String> {
        let layout = MtpLayout::for_state(state)?;
        let lengths = layout.payload_lengths(frontier, state.prev_hidden_pos)?;
        checked_host_bytes(std::mem::size_of::<Self>(), lengths)?;
        quiesce(gpu)?;
        let kv = &state.mtp_kv.inner;
        let mut blobs = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        blobs[0] = grab(gpu, &kv.k_gpu[0].buf, lengths[0])?;
        blobs[1] = grab(gpu, &kv.v_gpu[0].buf, lengths[1])?;
        blobs[2] = grab(gpu, &state.prev_hidden.buf, lengths[2])?;
        if let (Some(a), Some(b)) = (&kv.givens_cos, &kv.givens_sin) {
            blobs[3] = grab(gpu, &a.buf, lengths[3])?;
            blobs[4] = grab(gpu, &b.buf, lengths[4])?;
        }
        Ok(Self {
            layout,
            frontier,
            prev_hidden_pos: state.prev_hidden_pos,
            blobs,
        })
    }

    pub fn restore(
        &self,
        gpu: &mut Gpu,
        state: &mut MtpSpecState,
        frontier: usize,
    ) -> Result<(), String> {
        let layout = MtpLayout::for_state(state)?;
        if self.layout != layout || self.frontier != frontier {
            return Err("MTP image: destination format/geometry/frontier mismatch".into());
        }
        let lengths = self
            .layout
            .payload_lengths(frontier, self.prev_hidden_pos)?;
        validate_lengths(&self.blobs, &lengths)?;
        // Allocate small scratch clears before mutation. At frontier zero the
        // saved hidden is absent: do not replay arbitrary uninitialized bytes.
        let position_zeros = vec![0; product("positions", &[state.max_n, 4])?];
        let hidden_zeros = if frontier == 0 {
            vec![0; product("hidden", &[layout.hidden_dim, 4])?]
        } else {
            Vec::new()
        };
        state
            .invalidate_native_image_graphs(gpu)
            .map_err(|e| format!("MTP image graph invalidation: {e}"))?;
        // No whole-capacity reset/copy is necessary: head attention is bounded
        // by the caller's restored frontier and overwrites each new row before
        // reading it. Rejected/old tail rows are not part of this image.
        let kv = &state.mtp_kv.inner;
        put(gpu, &kv.k_gpu[0].buf, &self.blobs[0])?;
        put(gpu, &kv.v_gpu[0].buf, &self.blobs[1])?;
        if let (Some(a), Some(b)) = (&kv.givens_cos, &kv.givens_sin) {
            put(gpu, &a.buf, &self.blobs[3])?;
            put(gpu, &b.buf, &self.blobs[4])?;
        }
        put(
            gpu,
            &state.prev_hidden.buf,
            if frontier == 0 {
                &hidden_zeros
            } else {
                &self.blobs[2]
            },
        )?;
        put(gpu, &state.mtp_positions.buf, &position_zeros)?;
        put(gpu, &state.mtp_scratch.pos_buf, &0i32.to_ne_bytes())?;
        quiesce(gpu)?;
        // Only publish after every write completes. On a partial upload the
        // position remains None, and the caller must invalidate the owner.
        state.prev_hidden_pos = self.prev_hidden_pos;
        Ok(())
    }

    /// Owned host footprint (fixed metadata plus raw payload; allocator overhead
    /// is excluded). Capacity tails of GPU buffers are never charged/copied.
    pub fn host_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.blobs.iter().map(Vec::len).sum::<usize>()
    }
}

fn validate_lengths(blobs: &[Vec<u8>], expected: &[usize]) -> Result<(), String> {
    check_len("blob count", blobs.len(), expected.len())?;
    for (i, (blob, &len)) in blobs.iter().zip(expected).enumerate() {
        check_len(&format!("blob {i}"), blob.len(), len)?;
    }
    Ok(())
}

fn checked_host_bytes(
    base: usize,
    lengths: impl IntoIterator<Item = usize>,
) -> Result<usize, String> {
    lengths.into_iter().try_fold(base, |sum, len| {
        sum.checked_add(len)
            .ok_or_else(|| "native draft image: host byte count overflow".into())
    })
}

/// DFlash deliberately reuses the existing native ring algorithm. The wrapper
/// adds admission and *complete preflight* (the underlying restore otherwise
/// checks a later blob only after writing earlier ones).
#[cfg(feature = "kvmem")]
mod dflash_image {
    use super::*;
    use crate::dflash_slot::{self, DflashShared, DflashSlotState};
    use crate::kvmem_prefix::DraftImage;
    use hipfire_runtime::dflash::DraftCtxMode;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Layout {
        geometry: u64,
        window: usize,
        feature_width: usize,
        kv_dim: usize,
        n_layers: usize,
        max_ctx: usize,
        lengths: Vec<usize>,
    }

    impl Layout {
        fn for_state(
            shared: &DflashShared,
            state: &DflashSlotState,
            frontier: usize,
        ) -> Result<Self, String> {
            let d = &shared.draft_config;
            let s = &state.scratch;
            let w = shared
                .window
                .filter(|&w| w != 0)
                .ok_or_else(|| "DFlash image: windowed draft required".to_string())?;
            if !d.all_layers_sliding
                || d.n_layers == 0
                || d.hidden == 0
                || d.target_streams_per_layer != 1
                || d.target_layer_ids.is_empty()
                || d.n_kv_heads == 0
                || d.head_dim == 0
                || s.ctx_mode != (DraftCtxMode::Windowed { w, w_full: w })
                || shared.ctx_capacity != w
                || s.max_ctx_len != shared.physical_ctx
                || frontier > s.max_ctx_len
                || frontier > i32::MAX as usize
                || shared.block_size == 0
                || s.max_block_size != shared.block_size
                || s.feature_stage_rows.is_some()
                || s.online.is_some()
                || s.lane_ws.is_some()
                || s.k_full_cached.is_some()
                || s.v_full_cached.is_some()
                || s.k_cat_full.is_some()
                || s.v_cat_full.is_some()
            {
                return Err(
                    "DFlash image: unsupported windowed/all-sliding native geometry or state"
                        .into(),
                );
            }
            if state.draft.is_some() {
                return Err(
                    "DFlash image: in-flight draft must be consumed before snapshot/restore".into(),
                );
            }
            check_len("DFlash K layers", s.k_ctx_cached.len(), d.n_layers)?;
            check_len("DFlash V layers", s.v_ctx_cached.len(), d.n_layers)?;
            let feature_width = product(
                "DFlash features",
                &[
                    d.target_layer_ids.len(),
                    d.target_streams_per_layer,
                    d.hidden,
                ],
            )?;
            let kv_dim = product("DFlash KV dimension", &[d.n_kv_heads, d.head_dim])?;
            let hidden_elems = product("DFlash hidden ring", &[w, feature_width])?;
            let kv_elems = product("DFlash KV ring", &[w, kv_dim])?;
            check_f32_container(&s.target_hidden, hidden_elems, "DFlash target_hidden")?;
            // Native helpers copy the entire owning allocation, which the pool
            // may overallocate. Derive geometry from config/shape, not buf.size;
            // record actual extents so every destination plane can be checked
            // before the helper performs its first write.
            let count = d
                .n_layers
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| "DFlash image: layer count overflow".to_string())?;
            let mut lengths = Vec::with_capacity(count);
            lengths.push(s.target_hidden.buf.size());
            for (k, v) in s.k_ctx_cached.iter().zip(&s.v_ctx_cached) {
                check_f32_container(k, kv_elems, "DFlash K ring")?;
                check_f32_container(v, kv_elems, "DFlash V ring")?;
                lengths.extend([k.buf.size(), v.buf.size()]);
            }
            checked_host_bytes(0, lengths.iter().copied())?;
            Ok(Self {
                geometry: dflash_slot::dflash_geometry_id(shared),
                window: w,
                feature_width,
                kv_dim,
                n_layers: d.n_layers,
                max_ctx: s.max_ctx_len,
                lengths,
            })
        }

        fn validate_image(&self, img: &DraftImage, frontier: usize) -> Result<(), String> {
            if img.geometry != self.geometry
                || img.frontier as usize != frontier
                || img.window != self.window
                || img.n_layers != self.n_layers
                || img.feature_width != self.feature_width
                || img.kv_dim != self.kv_dim
                || frontier > self.max_ctx
                || frontier > i32::MAX as usize
                || img.floor as usize > frontier
                || img.proj_cached_rows > frontier
                || img.full_cached_rows > frontier
            {
                return Err(
                    "DFlash image: geometry/frontier/floor/cache watermark mismatch".into(),
                );
            }
            validate_lengths(&img.blobs, &self.lengths)
        }
    }

    /// Native conditioning state only: target-hidden and per-layer K/V rings,
    /// contiguous absolute positions/watermarks, seeded frontier and suffix floor.
    /// Sampling, penalty, RNG, verify scratch and rollback snapshots are request
    /// local, as in the MTP contract above. The destination session is explicit.
    #[derive(Clone, Debug)]
    pub struct LegacyDflashImage {
        layout: Layout,
        image: DraftImage,
    }

    impl LegacyDflashImage {
        pub fn capture(
            gpu: &Gpu,
            shared: &DflashShared,
            state: &DflashSlotState,
            frontier: usize,
            floor: usize,
        ) -> Result<Self, String> {
            let layout = Layout::for_state(shared, state, frontier)?;
            let log = &state.scratch.thlog;
            // `state.session` is a bookkeeping echo only (the DFlash inference
            // path never consults it): a freshly seeded state has it `None`
            // while still holding a complete native frontier. The proof that
            // this image belongs to a request is the caller's token/model/DN
            // frontier check, not this label.
            if state.seeded_through != frontier
                || log.uploaded_rows() != frontier
                || log.abs_positions().len() != frontier
                || log
                    .abs_positions()
                    .iter()
                    .enumerate()
                    .any(|(p, &a)| a != p as i32)
                || log.proj_cached_rows() > frontier
                || log.full_cached_rows() > frontier
                || floor > frontier
                || state.scratch.ctx_floor != floor
            {
                return Err(format!(
                    "DFlash image: inconsistent native state (session={:?} frontier={frontier} \
                     seeded_through={} uploaded_rows={} abs_positions={} proj_cached={} \
                     full_cached={} floor={floor} ctx_floor={})",
                    state.session,
                    state.seeded_through,
                    log.uploaded_rows(),
                    log.abs_positions().len(),
                    log.proj_cached_rows(),
                    log.full_cached_rows(),
                    state.scratch.ctx_floor,
                ));
            }
            quiesce(gpu)?;
            let image = dflash_slot::dflash_capture_image(gpu, shared, state, frontier, floor)?;
            layout.validate_image(&image, frontier)?;
            let result = Self { layout, image };
            result.checked_host_bytes()?;
            Ok(result)
        }

        pub fn restore(
            &self,
            gpu: &mut Gpu,
            shared: &DflashShared,
            state: &mut DflashSlotState,
            session: u64,
            frontier: usize,
        ) -> Result<(), String> {
            let layout = Layout::for_state(shared, state, frontier)?;
            if self.layout != layout {
                return Err("DFlash image: destination geometry/capacity mismatch".into());
            }
            self.layout.validate_image(&self.image, frontier)?;
            quiesce(gpu)?;
            // DFlash FFN/segment recordings contain only position-independent
            // block transforms (dflash.rs DraftGraphState). Context projection,
            // RoPE, ring concat and attention run OUTSIDE them. Buffer addresses
            // are unchanged here and caller binds the same weights: those are
            // not stale-frontier recordings. Native restore resets all thlog
            // position/cache watermarks; target_hidden_proj is scratch for each
            // newly projected range, not missing accepted history.
            state.session = None;
            state.seeded_through = 0;
            dflash_slot::dflash_restore_image(gpu, shared, state, &self.image, session, frontier)?;
            // The helper publishes the binding after its copies. Withdraw it
            // until all streams finish, so a synchronization error also leaves
            // the owner unbound rather than appearing successfully restored.
            state.session = None;
            state.seeded_through = 0;
            quiesce(gpu)?;
            // The old helper does not install ctx_floor.
            state.scratch.ctx_floor = self.image.floor as usize;
            state.session = Some(session);
            state.seeded_through = frontier;
            Ok(())
        }

        fn checked_host_bytes(&self) -> Result<usize, String> {
            let meta = product(
                "DFlash image metadata",
                &[self.layout.lengths.len(), std::mem::size_of::<usize>()],
            )?;
            let blob_meta = product(
                "DFlash blob metadata",
                &[self.image.blobs.len(), std::mem::size_of::<Vec<u8>>()],
            )?;
            checked_host_bytes(
                std::mem::size_of::<Self>(),
                [meta, blob_meta]
                    .into_iter()
                    .chain(self.image.blobs.iter().map(Vec::len)),
            )
        }

        pub fn host_bytes(&self) -> usize {
            // Private fields, checked once at construction; restore never mutates
            // this image or consumes any of its payload.
            self.checked_host_bytes()
                .expect("validated DFlash host byte count")
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn fixture() -> (Layout, DraftImage) {
            // Logical rings are smaller than these pooled owning allocations.
            let layout = Layout {
                geometry: 7,
                window: 4,
                feature_width: 8,
                kv_dim: 4,
                n_layers: 2,
                max_ctx: 64,
                lengths: vec![256; 5],
            };
            let image = DraftImage {
                geometry: layout.geometry,
                frontier: 6,
                floor: 2,
                window: layout.window,
                feature_width: layout.feature_width,
                kv_dim: layout.kv_dim,
                n_layers: layout.n_layers,
                proj_cached_rows: 6,
                full_cached_rows: 0,
                blobs: layout.lengths.iter().map(|&len| vec![0; len]).collect(),
            };
            (layout, image)
        }

        #[test]
        fn native_allocation_extents_are_checked_including_last_layer() {
            let (layout, image) = fixture();
            assert!(layout.validate_image(&image, 6).is_ok());
            assert!(layout.validate_image(&image, 6).is_ok()); // reusable
            let mut truncated = image.clone();
            truncated.blobs.last_mut().unwrap().pop();
            assert!(layout.validate_image(&truncated, 6).is_err());
            let mut different_destination = layout.clone();
            *different_destination.lengths.last_mut().unwrap() += 4;
            assert!(different_destination.validate_image(&image, 6).is_err());
        }

        #[test]
        fn frontier_floor_and_watermarks_cannot_be_silently_clamped() {
            let (layout, image) = fixture();
            assert!(layout.validate_image(&image, 5).is_err());
            let mut bad = image.clone();
            bad.floor = 7;
            assert!(layout.validate_image(&bad, 6).is_err());
            let mut bad = image.clone();
            bad.proj_cached_rows = 7;
            assert!(layout.validate_image(&bad, 6).is_err());
            let mut bad = image;
            bad.full_cached_rows = 7;
            assert!(layout.validate_image(&bad, 6).is_err());
        }

        #[test]
        fn geometry_and_missing_blob_are_rejected() {
            let (layout, image) = fixture();
            let mut bad = image.clone();
            bad.geometry += 1;
            assert!(layout.validate_image(&bad, 6).is_err());
            let mut bad = image.clone();
            bad.feature_width += 1;
            assert!(layout.validate_image(&bad, 6).is_err());
            let mut bad = image;
            bad.blobs.pop();
            assert!(layout.validate_image(&bad, 6).is_err());
        }
    }
}

#[cfg(feature = "kvmem")]
pub use dflash_image::LegacyDflashImage;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_head_layouts_have_independent_strides_and_inline_scales() {
        for (mode, hd, k, v, rotation) in [
            (MtpKvMode::Q8, 256, 1088, 1088, 0),
            (MtpKvMode::Asym3, 256, 400, 1088, 128),
            (MtpKvMode::Fwht4, 256, 528, 1088, 128),
            (MtpKvMode::Fwht4, 128, 272, 544, 128),
        ] {
            let layout = MtpLayout::new(mode, 4096, 4, hd, 4096).unwrap();
            assert_eq!(
                (layout.k_stride, layout.v_stride, layout.rotation_elems),
                (k, v, rotation)
            );
            assert_eq!(
                layout.payload_lengths(3, Some(2)).unwrap(),
                [3 * k, 3 * v, 16384, rotation * 4, rotation * 4]
            );
        }
    }

    #[test]
    fn packed_prefix_excludes_f32_container_rounding() {
        let layout = MtpLayout::new(MtpKvMode::Q8, 3, 1, 32, 64).unwrap();
        assert_eq!(layout.cache_elems(layout.k_stride).unwrap(), 26); // 104B allocation
        assert_eq!(
            layout.payload_lengths(3, Some(2)).unwrap(),
            [102, 102, 256, 0, 0]
        );
    }

    #[test]
    fn frontier_and_empty_history_are_fail_closed() {
        let layout = MtpLayout::new(MtpKvMode::Q8, 8, 1, 256, 64).unwrap();
        assert_eq!(layout.payload_lengths(0, None).unwrap(), [0; 5]);
        assert!(layout.payload_lengths(0, Some(0)).is_err());
        assert!(layout.payload_lengths(1, None).is_err());
        assert!(layout.payload_lengths(8, Some(6)).is_err());
        assert!(layout.payload_lengths(9, Some(8)).is_err());
        assert!(layout.payload_lengths(8, Some(7)).is_ok());
    }

    #[test]
    fn malformed_geometry_and_overflows_are_errors() {
        assert!(MtpLayout::new(MtpKvMode::Asym3, 8, 1, 128, 64).is_err());
        assert!(MtpLayout::new(MtpKvMode::Q8, 8, 0, 256, 64).is_err());
        assert!(MtpLayout::new(MtpKvMode::Q8, 8, 1, 33, 64).is_err());
        assert!(MtpLayout::new(MtpKvMode::Fwht4, 0, 1, 256, 64).is_err());
        assert!(MtpLayout::new(MtpKvMode::Q8, usize::MAX, 4, 256, 64).is_err());
        assert!(MtpLayout::new(MtpKvMode::Q8, 8, usize::MAX, 256, 64).is_err());
        assert!(checked_host_bytes(1, [usize::MAX]).is_err());
    }

    #[test]
    fn all_blob_lengths_are_preflighted_including_last_plane() {
        let blobs = [vec![0; 4], vec![0; 8], vec![0; 16]];
        assert!(validate_lengths(&blobs, &[4, 8, 16]).is_ok());
        assert!(validate_lengths(&blobs, &[4, 8, 15]).is_err());
        assert!(validate_lengths(&blobs, &[4, 8]).is_err());
        assert!(validate_lengths(&blobs, &[4, 8, 16, 0]).is_err());
    }
}
