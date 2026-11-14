// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Layerwise packed-page composite-slot pool + arena geometry (Phase 3.2 ①地基).
//!
//! This is the host substrate for the `RECALL_STRATEGY=Layerwise` subsystem. It
//! is a CLEAN SIBLING to the legacy [`crate::page_pool::PagePool`] /
//! [`crate::slot_pool::SlotPool`]: nothing here is wired into the live serve
//! allocation in sub-patch ①, so the default Legacy path stays byte-neutral.
//! Sub-patch ② drives this pool from the per-layer recall driver.
//!
//! # Model (ninfer-faithful packed pages, design §3.1)
//!
//! * One physical **composite slot** holds `block_tokens` positions × EVERY fa
//!   layer × {Key, Value}. Its byte layout is layer-major {K, V} per layer with
//!   `token_stride == k_gpu_row_bytes` (Q8 scales inline) — byte-for-byte the
//!   core `raw_kv_payload_layout` record, so a sealed-archive layer region maps
//!   straight onto a slot's layer plane with no repacking.
//! * Each layer selects its OWN canonical pages into the shared compact extent:
//!   slot `s`'s layer-`i` stripe comes from canonical page `pages_i[s]`, and
//!   different layers may map different canonical pages into the same slot
//!   (mixed-stripe destinations). A canonical page is resident iff ANY layer
//!   selects it.
//! * Per-layer **compact block table** = an ordered vector of physical slots
//!   (compact position → slot); the canonical page IDs are read from the slot
//!   stripes. This replaces the legacy one-shared-sentinel-table-per-slot.
//! * **Shared compact extent**: every layer's `visible_tokens` is equal (the
//!   core validates this because each layer gets an equal budget).
//! * **Drop-evict** (design §3.7): eviction is a bare drop — the sealed host
//!   archive is the source of truth, so there is NO write-back. A physical slot
//!   returns to the free list only when ALL of its layer stripes have dropped.
//!
//! Everything here is pure host logic over plain integers (no `hipfire-kvmem`
//! dependency): the geometry and the pool are CPU-unit-testable. The two thin
//! GPU seams ([`PackedArenaLayout::alloc_tensor`] / [`PackedArenaLayout::stripe_views`])
//! only apply the tested geometry through the existing `Gpu` primitives and are
//! exercised by the sub-patch ② integration.

use crate::{DType, Gpu, GpuTensor};
use hip_bridge::HipResult;

/// Tokens per composite page. Must match the engine page size
/// ([`crate::page_pool::PAGE_TOKENS`]) and `KV_BLOCK_TOKENS` in the controller;
/// the pool takes `block_tokens` explicitly so tests can use a small fixture.
pub const LAYERWISE_PAGE_TOKENS: usize = 128;

// =========================================================================
// Packed arena geometry
// =========================================================================

/// Byte geometry of the shared packed arena: `n_slots` composite slots, each
/// `block_tokens` positions × `n_layers` fa layers × {K, V}. Layer-major {K, V}
/// per layer, matching the core `raw_kv_payload_layout` record layout exactly.
///
/// All sizes are pre-computed with checked arithmetic at construction, so the
/// accessors and offset helpers cannot overflow: every offset is bounded by
/// [`Self::arena_bytes`], which already fits `usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedArenaLayout {
    n_slots: usize,
    n_layers: usize,
    block_tokens: usize,
    k_stride: usize,
    v_stride: usize,
    layer_stripe_bytes: usize,
    slot_bytes: usize,
    arena_bytes: usize,
}

impl PackedArenaLayout {
    /// Resolve the geometry. Every argument must be non-zero; `n_slots` must fit
    /// the `u32` physical-slot index space; every intermediate product is
    /// checked. Returns `Err` (never panics) on a bad argument or overflow.
    pub fn new(
        n_slots: usize,
        n_layers: usize,
        block_tokens: usize,
        k_stride: usize,
        v_stride: usize,
    ) -> Result<Self, String> {
        if n_slots == 0 {
            return Err("PackedArenaLayout: n_slots must be positive".to_string());
        }
        if n_layers == 0 {
            return Err("PackedArenaLayout: n_layers must be positive".to_string());
        }
        if block_tokens == 0 {
            return Err("PackedArenaLayout: block_tokens must be positive".to_string());
        }
        if k_stride == 0 {
            return Err("PackedArenaLayout: k_stride must be positive".to_string());
        }
        if v_stride == 0 {
            return Err("PackedArenaLayout: v_stride must be positive".to_string());
        }
        if n_slots > u32::MAX as usize {
            return Err(format!(
                "PackedArenaLayout: {n_slots} slots exceed the u32 slot-index space"
            ));
        }
        const OVERFLOW: &str = "PackedArenaLayout: arena size overflows usize";
        let kv_stride = k_stride.checked_add(v_stride).ok_or_else(|| OVERFLOW.to_string())?;
        let layer_stripe_bytes = block_tokens
            .checked_mul(kv_stride)
            .ok_or_else(|| OVERFLOW.to_string())?;
        let slot_bytes = n_layers
            .checked_mul(layer_stripe_bytes)
            .ok_or_else(|| OVERFLOW.to_string())?;
        let arena_bytes = n_slots
            .checked_mul(slot_bytes)
            .ok_or_else(|| OVERFLOW.to_string())?;
        Ok(Self {
            n_slots,
            n_layers,
            block_tokens,
            k_stride,
            v_stride,
            layer_stripe_bytes,
            slot_bytes,
            arena_bytes,
        })
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }
    /// Per-position K row bytes (Q8: `n_kv_heads * (head_dim / 32) * 34`).
    pub fn k_stride(&self) -> usize {
        self.k_stride
    }
    /// Per-position V row bytes (Q8: equal to the K stride).
    pub fn v_stride(&self) -> usize {
        self.v_stride
    }
    /// Bytes of one layer's {K, V} stripe inside a slot
    /// (`block_tokens * (k_stride + v_stride)`).
    pub fn layer_stripe_bytes(&self) -> usize {
        self.layer_stripe_bytes
    }
    /// Bytes of one composite slot (`n_layers * layer_stripe_bytes`). Equal to
    /// the core record bytes for the same geometry.
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    /// Total packed arena bytes (`n_slots * slot_bytes`).
    pub fn arena_bytes(&self) -> usize {
        self.arena_bytes
    }
    /// Bytes of one layer's K plane inside a slot (`block_tokens * k_stride`).
    pub fn k_plane_bytes(&self) -> usize {
        self.block_tokens * self.k_stride
    }
    /// Bytes of one layer's V plane inside a slot (`block_tokens * v_stride`).
    pub fn v_plane_bytes(&self) -> usize {
        self.block_tokens * self.v_stride
    }

    /// Byte offset of composite slot `slot` within the arena.
    pub fn slot_base(&self, slot: usize) -> Result<usize, String> {
        if slot >= self.n_slots {
            return Err(format!(
                "PackedArenaLayout: slot {slot} out of range ({} slots)",
                self.n_slots
            ));
        }
        // Bounded by arena_bytes (a valid usize), so this cannot overflow.
        Ok(slot * self.slot_bytes)
    }

    /// Byte offset of layer `layer`'s K plane inside slot `slot`.
    pub fn layer_k_offset(&self, slot: usize, layer: usize) -> Result<usize, String> {
        if layer >= self.n_layers {
            return Err(format!(
                "PackedArenaLayout: layer {layer} out of range ({} layers)",
                self.n_layers
            ));
        }
        Ok(self.slot_base(slot)? + layer * self.layer_stripe_bytes)
    }

    /// Byte offset of layer `layer`'s V plane inside slot `slot` (its K plane
    /// followed by `block_tokens * k_stride` bytes).
    pub fn layer_v_offset(&self, slot: usize, layer: usize) -> Result<usize, String> {
        Ok(self.layer_k_offset(slot, layer)? + self.k_plane_bytes())
    }

    /// Allocate the zero-filled packed arena as a raw-byte device tensor. Thin
    /// seam over [`Gpu::zeros`]; the geometry is host-tested above. Sub-patch ②
    /// owns the lifetime (mutually exclusive with the legacy per-layer arenas).
    pub fn alloc_tensor(&self, gpu: &mut Gpu) -> HipResult<GpuTensor> {
        gpu.zeros(&[self.arena_bytes], DType::Raw)
    }

    /// Non-owning {K, V} byte views of layer `layer`'s stripe inside slot `slot`.
    /// The returned tensors are VIEWS into `arena` — never free them. Thin seam
    /// over [`GpuTensor::sub_offset`] at the host-tested plane offsets.
    pub fn stripe_views(
        &self,
        arena: &GpuTensor,
        slot: usize,
        layer: usize,
    ) -> Result<(GpuTensor, GpuTensor), String> {
        if arena.dtype != DType::Raw {
            return Err("PackedArenaLayout: arena tensor must be DType::Raw".to_string());
        }
        if arena.buf.size() < self.arena_bytes {
            return Err(format!(
                "PackedArenaLayout: arena tensor too small ({} < {} bytes)",
                arena.buf.size(),
                self.arena_bytes
            ));
        }
        let k_off = self.layer_k_offset(slot, layer)?;
        let v_off = self.layer_v_offset(slot, layer)?;
        // Raw dtype: element size 1, so offsets/lengths are byte counts.
        Ok((
            arena.sub_offset(k_off, self.k_plane_bytes()),
            arena.sub_offset(v_off, self.v_plane_bytes()),
        ))
    }
}

// =========================================================================
// Composite-slot pool
// =========================================================================

/// A generation-checked handle to a physical composite slot. A handle from a
/// prior allocation cycle (the slot was freed and re-allocated) mismatches and
/// is rejected before any stripe is touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotHandle {
    pub phys: u32,
    pub generation: u32,
}

/// One fa layer's compact resident view: the ordered physical slots backing its
/// selected canonical pages, plus the shared `visible_tokens`. The canonical
/// page IDs are NOT stored here — they are read from the slot stripes, so there
/// is a single source of truth for residency.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayerView {
    /// Compact position → physical slot (device block-table order).
    slots: Vec<u32>,
    /// Σ valid_tokens of the selected pages. Equal across layers (core rule).
    visible_tokens: usize,
}

/// Per-slot metadata: generation, free flag, and the canonical page resident in
/// each layer's stripe (`None` = no stripe). The stripe refcount is the number
/// of `Some` entries — the drop-evict lease.
#[derive(Debug, Clone)]
struct SlotMeta {
    generation: u32,
    free: bool,
    stripes: Vec<Option<u32>>,
}

impl SlotMeta {
    fn new(n_layers: usize) -> Self {
        Self {
            generation: 0,
            free: true,
            stripes: vec![None; n_layers],
        }
    }
    fn refs(&self) -> u32 {
        self.stripes.iter().filter(|s| s.is_some()).count() as u32
    }
}

/// Composite-slot free-list pool for the layerwise packed arena (design §3.5).
///
/// Tracks physical composite slots and per-layer stripe residency. A slot is
/// reserved by [`Self::alloc_slot`] and returns to the free list only when
/// [`Self::free_slot`] is called with every layer stripe dropped — the bare
/// drop-evict invariant. Per-layer compact views ([`Self::set_view`]) record the
/// device block-table order over the resident slots.
#[derive(Debug)]
pub struct LayerwisePagePool {
    layout: PackedArenaLayout,
    slots: Vec<SlotMeta>,
    /// Free slot indices (exactly the slots with `free == true`). Popped from
    /// the end, like the legacy pool.
    free: Vec<u32>,
    views: Vec<LayerView>,
}

impl LayerwisePagePool {
    /// Create a pool over `layout`: every slot free at generation 0, every layer
    /// view empty.
    pub fn new(layout: PackedArenaLayout) -> Result<Self, String> {
        let n_slots = layout.n_slots();
        let n_layers = layout.n_layers();
        Ok(Self {
            layout,
            slots: (0..n_slots).map(|_| SlotMeta::new(n_layers)).collect(),
            free: (0..n_slots as u32).rev().collect(),
            views: vec![LayerView::default(); n_layers],
        })
    }

    pub fn layout(&self) -> &PackedArenaLayout {
        &self.layout
    }
    pub fn n_slots(&self) -> usize {
        self.layout.n_slots()
    }
    pub fn n_layers(&self) -> usize {
        self.layout.n_layers()
    }
    /// Number of free (unreserved) physical slots.
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    fn validate_handle(&self, h: SlotHandle) -> Result<usize, String> {
        let phys = h.phys as usize;
        let meta = self
            .slots
            .get(phys)
            .ok_or_else(|| format!("LayerwisePagePool: slot {} out of range", h.phys))?;
        if meta.free {
            return Err(format!(
                "LayerwisePagePool: slot {} is free (stale handle)",
                h.phys
            ));
        }
        if meta.generation != h.generation {
            return Err(format!(
                "LayerwisePagePool: slot {} generation mismatch (handle {}, current {})",
                h.phys, h.generation, meta.generation
            ));
        }
        Ok(phys)
    }

    fn validate_layer(&self, layer: usize) -> Result<(), String> {
        if layer >= self.n_layers() {
            return Err(format!(
                "LayerwisePagePool: layer {layer} out of range ({} layers)",
                self.n_layers()
            ));
        }
        Ok(())
    }

    /// Reserve a free composite slot, bumping its generation. The slot is owned
    /// by the caller (out of the free list) even before any stripe is bound.
    /// `Err` when the arena is exhausted.
    pub fn alloc_slot(&mut self) -> Result<SlotHandle, String> {
        let phys = self
            .free
            .pop()
            .ok_or_else(|| "LayerwisePagePool: no free composite slot".to_string())?;
        let meta = &mut self.slots[phys as usize];
        debug_assert!(meta.free && meta.refs() == 0);
        meta.free = false;
        meta.generation = meta.generation.wrapping_add(1);
        Ok(SlotHandle {
            phys,
            generation: meta.generation,
        })
    }

    /// Bind canonical page `page` into slot `h`'s layer-`layer` stripe. The slot
    /// must be reserved and that layer's stripe must be empty (drop first).
    pub fn bind_stripe(&mut self, h: SlotHandle, layer: usize, page: u32) -> Result<(), String> {
        let phys = self.validate_handle(h)?;
        self.validate_layer(layer)?;
        let meta = &mut self.slots[phys];
        if meta.stripes[layer].is_some() {
            return Err(format!(
                "LayerwisePagePool: slot {} layer {layer} already holds page {} (drop first)",
                h.phys,
                meta.stripes[layer].unwrap()
            ));
        }
        meta.stripes[layer] = Some(page);
        Ok(())
    }

    /// Drop slot `h`'s layer-`layer` stripe. Returns `true` when EVERY layer
    /// stripe has now dropped (the slot is freeable via [`Self::free_slot`]).
    /// Does not itself return the slot to the free list — the driver decides
    /// whether to free or reuse it. `Err` on a stale handle or an empty stripe.
    pub fn drop_stripe(&mut self, h: SlotHandle, layer: usize) -> Result<bool, String> {
        let phys = self.validate_handle(h)?;
        self.validate_layer(layer)?;
        let meta = &mut self.slots[phys];
        if meta.stripes[layer].is_none() {
            return Err(format!(
                "LayerwisePagePool: slot {} layer {layer} has no stripe to drop",
                h.phys
            ));
        }
        meta.stripes[layer] = None;
        Ok(meta.refs() == 0)
    }

    /// Return a reserved slot to the free list. Requires every layer stripe to
    /// have dropped (`refs == 0`) — the drop-evict invariant: a slot is never
    /// reclaimed while any layer still references it.
    pub fn free_slot(&mut self, h: SlotHandle) -> Result<(), String> {
        let phys = self.validate_handle(h)?;
        let meta = &self.slots[phys];
        if meta.refs() != 0 {
            return Err(format!(
                "LayerwisePagePool: slot {} still holds {} layer stripe(s); drop all before free",
                h.phys,
                meta.refs()
            ));
        }
        let meta = &mut self.slots[phys];
        meta.free = true;
        self.free.push(h.phys);
        Ok(())
    }

    /// The canonical page resident in slot `phys`'s layer-`layer` stripe, or
    /// `None` if the slot is free / out of range / has no stripe there.
    pub fn stripe_page(&self, phys: u32, layer: usize) -> Option<u32> {
        self.slots.get(phys as usize)?.stripes.get(layer).copied().flatten()
    }

    /// Number of layer stripes currently bound in slot `phys` (0 if free).
    pub fn slot_refs(&self, phys: u32) -> u32 {
        self.slots.get(phys as usize).map(SlotMeta::refs).unwrap_or(0)
    }

    /// True when slot `phys` is reserved (allocated, not in the free list).
    pub fn is_allocated(&self, phys: u32) -> bool {
        self.slots.get(phys as usize).map(|m| !m.free).unwrap_or(false)
    }

    /// The current generation-checked handle for reserved slot `phys`, or `None`
    /// when it is free / out of range. Lets a caller that located a slot by index
    /// (e.g. via [`Self::find_page_slot`]) drop/free it without re-tracking handles.
    pub fn handle(&self, phys: u32) -> Option<SlotHandle> {
        let meta = self.slots.get(phys as usize)?;
        if meta.free {
            return None;
        }
        Some(SlotHandle { phys, generation: meta.generation })
    }

    /// The reserved slot whose layer-`layer` stripe holds canonical page `page`,
    /// or `None` when no slot holds it there. Scans the (small) slot table; the
    /// per-layer stripe is the single source of truth for residency.
    pub fn find_page_slot(&self, layer: usize, page: u32) -> Option<u32> {
        if layer >= self.n_layers() {
            return None;
        }
        (0..self.slots.len() as u32).find(|&phys| self.stripe_page(phys, layer) == Some(page))
    }

    /// True when ANY layer's stripe in ANY slot references canonical page `page`
    /// (the page is resident in the composite arena).
    pub fn is_page_resident(&self, page: u32) -> bool {
        self.slots
            .iter()
            .any(|m| m.stripes.iter().any(|s| *s == Some(page)))
    }

    /// Set layer `layer`'s compact block table: `slots` is the ordered physical
    /// slot per compact position, `visible_tokens` the shared extent. Every slot
    /// must be reserved, distinct, and hold a stripe for this layer. An empty
    /// `slots` clears the view.
    pub fn set_view(
        &mut self,
        layer: usize,
        slots: &[u32],
        visible_tokens: usize,
    ) -> Result<(), String> {
        self.validate_layer(layer)?;
        let mut seen = std::collections::HashSet::with_capacity(slots.len());
        for &phys in slots {
            if !seen.insert(phys) {
                return Err(format!(
                    "LayerwisePagePool: layer {layer} view repeats slot {phys}"
                ));
            }
            let meta = self
                .slots
                .get(phys as usize)
                .ok_or_else(|| format!("LayerwisePagePool: view slot {phys} out of range"))?;
            if meta.free {
                return Err(format!(
                    "LayerwisePagePool: layer {layer} view references free slot {phys}"
                ));
            }
            if meta.stripes[layer].is_none() {
                return Err(format!(
                    "LayerwisePagePool: layer {layer} view slot {phys} has no layer-{layer} stripe"
                ));
            }
        }
        self.views[layer] = LayerView {
            slots: slots.to_vec(),
            visible_tokens,
        };
        Ok(())
    }

    /// Layer `layer`'s compact block table (physical slots in compact order).
    pub fn compact_slots(&self, layer: usize) -> Result<&[u32], String> {
        self.validate_layer(layer)?;
        Ok(&self.views[layer].slots)
    }

    /// Layer `layer`'s selected canonical page IDs in compact order, read from
    /// the resident slot stripes (the device block-table payload).
    pub fn canonical_pages(&self, layer: usize) -> Result<Vec<u32>, String> {
        self.validate_layer(layer)?;
        self.views[layer]
            .slots
            .iter()
            .map(|&phys| {
                self.stripe_page(phys, layer).ok_or_else(|| {
                    format!(
                        "LayerwisePagePool: layer {layer} view slot {phys} lost its stripe"
                    )
                })
            })
            .collect()
    }

    /// Layer `layer`'s shared compact extent (`Σ valid_tokens`).
    pub fn visible_tokens(&self, layer: usize) -> Result<usize, String> {
        self.validate_layer(layer)?;
        Ok(self.views[layer].visible_tokens)
    }

    /// Validate the shared compact extent: every layer's `visible_tokens` is
    /// equal (the core enforces this because each layer gets an equal budget).
    pub fn validate_shared_extent(&self) -> Result<(), String> {
        let first = self.views[0].visible_tokens;
        for (layer, view) in self.views.iter().enumerate() {
            if view.visible_tokens != first {
                return Err(format!(
                    "LayerwisePagePool: layer {layer} visible_tokens {} != layer 0 {first} \
                     (shared compact extent must be equal)",
                    view.visible_tokens
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_pool::PAGE_TOKENS;

    /// Small fixture geometry: 4 slots × 2 layers × 4 tokens, K stride 8 / V 6.
    fn layout() -> PackedArenaLayout {
        PackedArenaLayout::new(4, 2, 4, 8, 6).unwrap()
    }

    #[test]
    fn page_tokens_matches_legacy() {
        assert_eq!(LAYERWISE_PAGE_TOKENS, PAGE_TOKENS);
    }

    #[test]
    fn arena_geometry_matches_core_record_layout() {
        let l = layout();
        // layer stripe = 4 * (8 + 6) = 56; slot = 2 * 56 = 112; arena = 4 * 112.
        assert_eq!(l.layer_stripe_bytes(), 56);
        assert_eq!(l.slot_bytes(), 112);
        assert_eq!(l.arena_bytes(), 448);
        assert_eq!(l.k_plane_bytes(), 32);
        assert_eq!(l.v_plane_bytes(), 24);

        // Independently accumulate the core layer-major {K,V} record manifest.
        let mut record = 0usize;
        for layer in 0..l.n_layers() {
            let k_off = record;
            record += l.block_tokens() * l.k_stride();
            let v_off = record;
            record += l.block_tokens() * l.v_stride();
            // Slot 2's layer planes sit at slot_base(2) + the record offsets.
            assert_eq!(l.layer_k_offset(2, layer).unwrap(), 2 * 112 + k_off);
            assert_eq!(l.layer_v_offset(2, layer).unwrap(), 2 * 112 + v_off);
        }
        assert_eq!(record, l.slot_bytes());
        assert_eq!(l.slot_base(0).unwrap(), 0);
        assert_eq!(l.slot_base(3).unwrap(), 336);
    }

    #[test]
    fn q8_stride_geometry() {
        // Q8: n_kv_heads=4, head_dim=256 → 4 * (256/32) * 34 = 1088 bytes/pos.
        let stride = 4 * (256 / 32) * 34;
        let l = PackedArenaLayout::new(8, 3, LAYERWISE_PAGE_TOKENS, stride, stride).unwrap();
        assert_eq!(l.k_stride(), 1088);
        assert_eq!(l.layer_stripe_bytes(), 128 * 2 * 1088);
        assert_eq!(l.slot_bytes(), 3 * 128 * 2 * 1088);
        assert_eq!(l.arena_bytes(), 8 * l.slot_bytes());
    }

    #[test]
    fn rejects_zero_and_overflow() {
        assert!(PackedArenaLayout::new(0, 2, 4, 8, 6).is_err());
        assert!(PackedArenaLayout::new(4, 0, 4, 8, 6).is_err());
        assert!(PackedArenaLayout::new(4, 2, 0, 8, 6).is_err());
        assert!(PackedArenaLayout::new(4, 2, 4, 0, 6).is_err());
        assert!(PackedArenaLayout::new(4, 2, 4, 8, 0).is_err());
        assert!(PackedArenaLayout::new(u32::MAX as usize + 1, 2, 4, 8, 6).is_err());
        // Overflowing product.
        assert!(PackedArenaLayout::new(usize::MAX, 2, 4, 8, 6).is_err());
        assert!(PackedArenaLayout::new(4, usize::MAX, 4, 8, 6).is_err());
    }

    #[test]
    fn offset_bounds_are_checked() {
        let l = layout();
        assert!(l.slot_base(4).is_err());
        assert!(l.layer_k_offset(0, 2).is_err());
        assert!(l.layer_v_offset(4, 0).is_err());
        assert!(l.layer_k_offset(3, 1).is_ok());
    }

    #[test]
    fn alloc_bind_drop_free_lifecycle() {
        let mut pool = LayerwisePagePool::new(layout()).unwrap();
        assert_eq!(pool.free_count(), 4);
        let h = pool.alloc_slot().unwrap();
        assert_eq!(pool.free_count(), 3);
        assert!(pool.is_allocated(h.phys));
        assert_eq!(pool.slot_refs(h.phys), 0);

        // Mixed canonical pages in the same composite slot.
        pool.bind_stripe(h, 0, 100).unwrap();
        pool.bind_stripe(h, 1, 200).unwrap();
        assert_eq!(pool.slot_refs(h.phys), 2);
        assert_eq!(pool.stripe_page(h.phys, 0), Some(100));
        assert_eq!(pool.stripe_page(h.phys, 1), Some(200));

        // Dropping one layer does NOT free the slot.
        assert!(!pool.drop_stripe(h, 0).unwrap());
        assert_eq!(pool.slot_refs(h.phys), 1);
        assert!(pool.free_slot(h).is_err(), "slot still holds a layer stripe");
        // Dropping the last layer reports freeable, then free succeeds.
        assert!(pool.drop_stripe(h, 1).unwrap());
        assert_eq!(pool.slot_refs(h.phys), 0);
        pool.free_slot(h).unwrap();
        assert!(!pool.is_allocated(h.phys));
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn generation_rejects_stale_handle() {
        let mut pool = LayerwisePagePool::new(layout()).unwrap();
        let h1 = pool.alloc_slot().unwrap();
        pool.free_slot(h1).unwrap();
        let h2 = pool.alloc_slot().unwrap();
        assert_eq!(h1.phys, h2.phys);
        assert_ne!(h1.generation, h2.generation);
        assert!(pool.bind_stripe(h1, 0, 7).is_err(), "stale generation");
        assert!(pool.drop_stripe(h1, 0).is_err());
        assert!(pool.free_slot(h1).is_err());
        pool.bind_stripe(h2, 0, 7).unwrap();
    }

    #[test]
    fn mixed_stripe_destinations_and_residency() {
        let mut pool = LayerwisePagePool::new(layout()).unwrap();
        let s0 = pool.alloc_slot().unwrap();
        let s1 = pool.alloc_slot().unwrap();
        // Layer 0 selects [100, 200]; layer 1 selects [200, 100] — same slots,
        // different canonical pages per layer stripe.
        pool.bind_stripe(s0, 0, 100).unwrap();
        pool.bind_stripe(s0, 1, 200).unwrap();
        pool.bind_stripe(s1, 0, 200).unwrap();
        pool.bind_stripe(s1, 1, 100).unwrap();
        pool.set_view(0, &[s0.phys, s1.phys], 8).unwrap();
        pool.set_view(1, &[s0.phys, s1.phys], 8).unwrap();
        assert_eq!(pool.canonical_pages(0).unwrap(), vec![100, 200]);
        assert_eq!(pool.canonical_pages(1).unwrap(), vec![200, 100]);
        assert_eq!(pool.compact_slots(0).unwrap(), &[s0.phys, s1.phys]);
        assert!(pool.is_page_resident(100));
        assert!(pool.is_page_resident(200));
        assert!(!pool.is_page_resident(999));
        pool.validate_shared_extent().unwrap();
    }

    #[test]
    fn shared_extent_must_be_equal() {
        let mut pool = LayerwisePagePool::new(layout()).unwrap();
        let s0 = pool.alloc_slot().unwrap();
        pool.bind_stripe(s0, 0, 1).unwrap();
        pool.bind_stripe(s0, 1, 2).unwrap();
        pool.set_view(0, &[s0.phys], 8).unwrap();
        pool.set_view(1, &[s0.phys], 4).unwrap();
        assert!(pool.validate_shared_extent().is_err());
        pool.set_view(1, &[s0.phys], 8).unwrap();
        pool.validate_shared_extent().unwrap();
        assert_eq!(pool.visible_tokens(0).unwrap(), 8);
    }

    #[test]
    fn set_view_validates_residency() {
        let mut pool = LayerwisePagePool::new(layout()).unwrap();
        let s0 = pool.alloc_slot().unwrap();
        pool.bind_stripe(s0, 0, 1).unwrap();
        // Layer 1 has no stripe in s0.
        assert!(pool.set_view(1, &[s0.phys], 4).is_err());
        // A free slot cannot back a view.
        let s1 = pool.alloc_slot().unwrap();
        pool.free_slot(s1).unwrap();
        assert!(pool.set_view(0, &[s1.phys], 4).is_err());
        // Repeated slot in one view is rejected.
        assert!(pool.set_view(0, &[s0.phys, s0.phys], 8).is_err());
        // Out-of-range layer.
        assert!(pool.set_view(2, &[s0.phys], 4).is_err());
        // Valid single-layer view; empty view clears.
        pool.set_view(0, &[s0.phys], 4).unwrap();
        pool.set_view(0, &[], 0).unwrap();
        assert!(pool.compact_slots(0).unwrap().is_empty());
    }

    #[test]
    fn alloc_exhaustion_and_double_bind() {
        let mut pool = LayerwisePagePool::new(layout()).unwrap();
        let mut handles = Vec::new();
        for _ in 0..4 {
            handles.push(pool.alloc_slot().unwrap());
        }
        assert_eq!(pool.free_count(), 0);
        assert!(pool.alloc_slot().is_err());
        let h = handles[0];
        pool.bind_stripe(h, 0, 1).unwrap();
        assert!(pool.bind_stripe(h, 0, 2).is_err(), "double bind without drop");
        assert!(pool.drop_stripe(h, 1).is_err(), "no stripe to drop");
        assert!(pool.bind_stripe(h, 2, 1).is_err(), "layer out of range");
    }

    #[test]
    fn free_slot_rejects_stale_and_double_free() {
        let mut pool = LayerwisePagePool::new(layout()).unwrap();
        let h = pool.alloc_slot().unwrap();
        pool.free_slot(h).unwrap();
        assert!(pool.free_slot(h).is_err(), "double free / stale handle");
    }
}