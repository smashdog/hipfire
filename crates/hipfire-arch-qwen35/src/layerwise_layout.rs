// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Q8 packed-page layout adapter for the layerwise subsystem (Phase 3.2 ①地基).
//!
//! The hipfire analog of ninfer's `native_window_descriptor`: it walks hipfire's
//! Q8 KV geometry and fills the core GENERIC `kvmem::PayloadLayout` manifest
//! (layer-major {Key, Value} per fa layer, `token_stride == k_gpu_row_bytes`,
//! Q8 scales inline). The core `raw_kv_payload_layout` bridge accepts Key/Value
//! token-row planes directly, so no separate scale plane / `NativeKvArchive`
//! generic path is needed (design §3.4).
//!
//! Clean sibling to the legacy [`crate::kvmem_ctl`]: this adapter is NOT wired
//! into the live serve allocation in sub-patch ①, so Legacy stays byte-neutral.
//! It reuses the engine's OWN stride authority ([`SlotKvTierPlan::resolve`]) so
//! the manifest can never drift from the arena the kernels actually walk, and
//! bridges to the crate-agnostic [`PackedArenaLayout`] so a composite slot's
//! byte layout is provably identical to the core record (see the tests).

use hipfire_kvmem::{PayloadLayout, RawStoreConfig};
use rdna_compute::layerwise_pool::PackedArenaLayout;
use saddle_core::kv::{KvMode, SlotKvTierPlan};

/// Backend-owned manifest identity for the hipfire Q8 packed-page record.
pub const Q8_PACKED_FORMAT: &str = "hipfire-q8-packed-v1";

/// Backend-owned manifest identity for the hipfire Q8 Main+aux MTP composite
/// record (P4.1①). The hipfire analog of ninfer's `kvmem-main-mtp-pages-v4`: ONE
/// composite host record co-archives the trunk (Main plane, `n_fa` layers) AND
/// the MTP head (auxiliary plane, ONE layer) so accepted-only write-through seals
/// both planes in a single record. The head shares the trunk's Q8 KV geometry.
pub const Q8_MAIN_MTP_FORMAT: &str = "hipfire-q8-main-mtp-v1";

/// Tokens per composite page — the engine page size (`KV_BLOCK_TOKENS` /
/// `PAGE_TOKENS`). Kept as a literal here so the adapter has no legacy import.
pub const LAYERWISE_BLOCK_TOKENS: u32 = 128;

/// Resolved Q8 packed-page geometry: the core [`RawStoreConfig`] bridge inputs,
/// the per-position K/V strides, and the fa-layer manifest list.
///
/// `fa_layers` carries the layer IDs the core `RawStore` is indexed by. hipfire's
/// controller stores packed rows by fa ORDINAL (`0..n_fa_layers`), so the caller
/// passes ordinals here and `n_layer_total == n_fa_layers`; the model-layer ↔
/// ordinal mapping is the sub-patch ② catalog's concern.
#[derive(Debug, Clone)]
pub struct LayerwiseQ8Layout {
    raw_cfg: RawStoreConfig,
    k_stride: usize,
    v_stride: usize,
    fa_layers: Vec<u32>,
}

impl LayerwiseQ8Layout {
    /// Resolve from model geometry. `n_layer_total` is the core store's layer
    /// count (`cfg.n_layer`); `fa_layers` the attention-layer IDs to manifest
    /// (non-empty, unique, each `< n_layer_total`). K/V strides come from the
    /// engine's Q8 tier plan (`n_kv_heads * (head_dim / 32) * 34`, scales inline).
    /// Fails closed on bad geometry — never silently defaults.
    pub fn resolve(
        n_layer_total: usize,
        n_kv_heads: usize,
        head_dim: usize,
        fa_layers: &[u32],
    ) -> Result<Self, String> {
        if n_layer_total == 0 || n_layer_total > u32::MAX as usize {
            return Err(format!(
                "layerwise Q8 layout: n_layer_total {n_layer_total} out of range"
            ));
        }
        if fa_layers.is_empty() {
            return Err("layerwise Q8 layout: empty attention-layer list".to_string());
        }
        let mut seen = std::collections::HashSet::with_capacity(fa_layers.len());
        for &layer in fa_layers {
            if layer as usize >= n_layer_total {
                return Err(format!(
                    "layerwise Q8 layout: fa layer {layer} >= n_layer_total {n_layer_total}"
                ));
            }
            if !seen.insert(layer) {
                return Err(format!("layerwise Q8 layout: duplicate fa layer {layer}"));
            }
        }
        // The engine's own Q8 stride authority (validates head_dim % 32, etc.).
        let plan = SlotKvTierPlan::resolve(KvMode::Q8, n_kv_heads, head_dim)
            .map_err(|e| format!("layerwise Q8 layout: {e}"))?;
        let k_stride = plan.k_bytes_per_pos;
        let v_stride = plan.v_bytes_per_pos;
        let k_width = n_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "layerwise Q8 layout: kv_dim overflowed".to_string())?;
        let raw_cfg = RawStoreConfig {
            n_layer: n_layer_total as u32,
            n_embd_k: k_width as u32,
            n_embd_v: k_width as u32,
            block_tokens: LAYERWISE_BLOCK_TOKENS,
            // Opaque packed rows: the packed stride lives in *_gpu_row_bytes.
            k_row_bytes: 0,
            k_gpu_row_bytes: k_stride as u64,
            v_gpu_row_bytes: v_stride as u64,
        };
        Ok(Self {
            raw_cfg,
            k_stride,
            v_stride,
            fa_layers: fa_layers.to_vec(),
        })
    }

    /// The core store bridge config (matches the legacy controller's convention).
    pub fn raw_config(&self) -> &RawStoreConfig {
        &self.raw_cfg
    }
    pub fn k_stride(&self) -> usize {
        self.k_stride
    }
    pub fn v_stride(&self) -> usize {
        self.v_stride
    }
    pub fn fa_layers(&self) -> &[u32] {
        &self.fa_layers
    }

    /// Build the core generic `PayloadLayout` manifest (layer-major {Key, Value}
    /// per fa layer). One record == one canonical page == one composite slot.
    pub fn payload_layout(&self) -> Result<PayloadLayout, String> {
        self.payload_layout_with_format(Q8_PACKED_FORMAT)
    }

    /// Build the core generic `PayloadLayout` manifest under an explicit backend
    /// format tag (P4.1①: the Main+aux MTP composite uses [`Q8_MAIN_MTP_FORMAT`]).
    /// The plane geometry is identical regardless of tag — the tag is the core's
    /// opaque provenance/admission key, so a composite record's Main and aux
    /// planes map straight onto slot stripes with no repacking.
    pub fn payload_layout_with_format(&self, format: &str) -> Result<PayloadLayout, String> {
        PayloadLayout::from_raw(&self.raw_cfg, format, &self.fa_layers)
            .map_err(|e| format!("layerwise Q8 layout: {e}"))
    }

    /// The crate-agnostic packed arena geometry for `n_slots` composite slots.
    /// Its `slot_bytes()` equals [`Self::payload_layout`]'s `record_bytes()`, and
    /// its per-layer plane offsets equal the manifest plane offsets — so a sealed
    /// archive layer region maps straight onto a slot stripe with no repacking.
    pub fn packed_arena_layout(&self, n_slots: usize) -> Result<PackedArenaLayout, String> {
        PackedArenaLayout::new(
            n_slots,
            self.fa_layers.len(),
            LAYERWISE_BLOCK_TOKENS as usize,
            self.k_stride,
            self.v_stride,
        )
    }
}

/// Main+aux composite Q8 layout (P4.1①): the trunk Main plane plus the MTP head's
/// single auxiliary plane, manifested as ONE core record over `n_fa_layers + 1`
/// layers (the head is the LAST ordinal). Clean sibling to [`LayerwiseQ8Layout`]
/// — NOT wired into live serve in sub-patch ①; it reuses the trunk adapter's
/// stride authority ([`SlotKvTierPlan::resolve`]) so the composite byte layout is
/// provably identical to the core record (slot_bytes == record_bytes, plane
/// offsets aligned). The head ALGORITHM is unchanged (full-history attention);
/// only its KV RESIDENCY is re-architected (bounded pool + aux-plane co-archive,
/// sub-patch ②). The Main plane is byte-identical to the standalone trunk record
/// for the same ordinals, so a trunk-only restore is unaffected by the aux plane.
#[derive(Debug, Clone)]
pub struct LayerwiseMtpQ8Layout {
    /// Trunk Main plane (`n_fa` layers, ordinals `0..n_fa`).
    main: LayerwiseQ8Layout,
    /// Composite Main+aux (`n_fa + 1` layers; head ordinal == `n_fa`).
    composite: LayerwiseQ8Layout,
    /// The head's auxiliary-plane layer ordinal (== `n_fa_layers`).
    head_layer: u32,
    n_fa_layers: usize,
}

impl LayerwiseMtpQ8Layout {
    /// Resolve from model geometry. `n_fa_layers` is the trunk FA-layer count
    /// (>= 1); the head adds ONE auxiliary plane at ordinal `n_fa_layers`. K/V
    /// strides come from the engine's Q8 tier plan (shared by trunk and head).
    /// Fails closed on bad geometry — never silently defaults.
    pub fn resolve(
        n_fa_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, String> {
        if n_fa_layers == 0 {
            return Err("layerwise MTP Q8 layout: n_fa_layers == 0".to_string());
        }
        let main_ordinals: Vec<u32> = (0..n_fa_layers as u32).collect();
        let main = LayerwiseQ8Layout::resolve(n_fa_layers, n_kv_heads, head_dim, &main_ordinals)
            .map_err(|e| format!("layerwise MTP Q8 layout (main): {e}"))?;
        let head_layer = n_fa_layers as u32;
        let mut composite_ordinals = main_ordinals;
        composite_ordinals.push(head_layer);
        let composite = LayerwiseQ8Layout::resolve(
            n_fa_layers + 1,
            n_kv_heads,
            head_dim,
            &composite_ordinals,
        )
        .map_err(|e| format!("layerwise MTP Q8 layout (composite): {e}"))?;
        Ok(Self { main, composite, head_layer, n_fa_layers })
    }

    /// The trunk Main-plane adapter (ordinals `0..n_fa`, standalone Q8 format).
    pub fn main(&self) -> &LayerwiseQ8Layout {
        &self.main
    }
    /// The head's auxiliary-plane layer ordinal (== `n_fa_layers`).
    pub fn head_layer(&self) -> u32 {
        self.head_layer
    }
    /// The trunk FA-layer count (Main plane layers).
    pub fn n_fa_layers(&self) -> usize {
        self.n_fa_layers
    }
    /// The composite core store layer count (`n_fa_layers + 1`, Main + aux).
    pub fn n_layer_total(&self) -> usize {
        self.n_fa_layers + 1
    }
    /// Per-position K row bytes (shared trunk/head Q8 geometry).
    pub fn k_stride(&self) -> usize {
        self.composite.k_stride()
    }
    /// Per-position V row bytes (shared trunk/head Q8 geometry).
    pub fn v_stride(&self) -> usize {
        self.composite.v_stride()
    }

    /// The composite Main+aux core manifest under [`Q8_MAIN_MTP_FORMAT`]:
    /// layer-major {Key, Value} for every trunk FA layer THEN the head's aux
    /// plane. One record == one canonical page == one composite slot.
    pub fn payload_layout(&self) -> Result<PayloadLayout, String> {
        self.composite.payload_layout_with_format(Q8_MAIN_MTP_FORMAT)
    }

    /// The crate-agnostic packed arena geometry for `n_slots` composite slots
    /// (`n_fa + 1` layers). Its `slot_bytes()` equals [`Self::payload_layout`]'s
    /// `record_bytes()`; the head's aux plane is the LAST layer stripe per slot.
    pub fn packed_arena_layout(&self, n_slots: usize) -> Result<PackedArenaLayout, String> {
        self.composite.packed_arena_layout(n_slots)
    }

    /// Byte offset of the head's auxiliary K plane inside composite slot `slot`.
    pub fn aux_k_offset(&self, arena: &PackedArenaLayout, slot: usize) -> Result<usize, String> {
        arena.layer_k_offset(slot, self.n_fa_layers)
    }
    /// Byte offset of the head's auxiliary V plane inside composite slot `slot`.
    pub fn aux_v_offset(&self, arena: &PackedArenaLayout, slot: usize) -> Result<usize, String> {
        arena.layer_v_offset(slot, self.n_fa_layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_kvmem::PayloadPlaneKind;

    /// Q8 geometry: 4 KV heads × head_dim 256 → 4 * 8 * 34 = 1088 bytes/pos.
    fn resolved() -> LayerwiseQ8Layout {
        LayerwiseQ8Layout::resolve(2, 4, 256, &[0, 1]).unwrap()
    }

    #[test]
    fn resolve_q8_strides_and_raw_config() {
        let l = resolved();
        assert_eq!(l.k_stride(), 1088);
        assert_eq!(l.v_stride(), 1088);
        let cfg = l.raw_config();
        assert_eq!(cfg.n_layer, 2);
        assert_eq!(cfg.n_embd_k, 4 * 256);
        assert_eq!(cfg.n_embd_v, 4 * 256);
        assert_eq!(cfg.block_tokens, 128);
        assert_eq!(cfg.k_row_bytes, 0);
        assert_eq!(cfg.k_gpu_row_bytes, 1088);
        assert_eq!(cfg.v_gpu_row_bytes, 1088);
        assert_eq!(l.fa_layers(), &[0, 1]);
    }

    #[test]
    fn rejects_bad_geometry() {
        // head_dim not divisible by 32 (engine Q8 gate).
        assert!(LayerwiseQ8Layout::resolve(2, 4, 100, &[0, 1]).is_err());
        // zero KV heads.
        assert!(LayerwiseQ8Layout::resolve(2, 0, 256, &[0, 1]).is_err());
        // empty / out-of-range / duplicate fa-layer list.
        assert!(LayerwiseQ8Layout::resolve(2, 4, 256, &[]).is_err());
        assert!(LayerwiseQ8Layout::resolve(2, 4, 256, &[0, 2]).is_err());
        assert!(LayerwiseQ8Layout::resolve(2, 4, 256, &[1, 1]).is_err());
        // zero layer count.
        assert!(LayerwiseQ8Layout::resolve(0, 4, 256, &[0]).is_err());
    }

    #[test]
    fn payload_layout_manifest_is_layer_major_key_value() {
        let l = resolved();
        let manifest = l.payload_layout().unwrap();
        assert_eq!(manifest.block_tokens(), 128);
        assert_eq!(manifest.plane_count(), 4); // 2 layers × {K, V}
        let plane_bytes = (1088 * 128) as u64;
        assert_eq!(manifest.record_bytes(), 4 * plane_bytes);
        let mut offset = 0u64;
        for layer in 0..2u32 {
            for kind in [PayloadPlaneKind::Key, PayloadPlaneKind::Value] {
                let p = manifest.plane(layer * 2 + kind as u32).unwrap();
                assert_eq!(p.layer, layer);
                assert_eq!(p.kind, kind as u32);
                assert_eq!(p.offset, offset);
                assert_eq!(p.bytes, plane_bytes);
                assert_eq!(p.token_stride, 1088);
                assert_eq!(p.alignment, 1);
                offset += plane_bytes;
            }
        }
        assert_eq!(offset, manifest.record_bytes());
    }

    #[test]
    fn packed_arena_matches_core_record() {
        let l = resolved();
        let manifest = l.payload_layout().unwrap();
        let arena = l.packed_arena_layout(8).unwrap();
        // One composite slot == one core record, byte for byte.
        assert_eq!(arena.slot_bytes() as u64, manifest.record_bytes());
        assert_eq!(arena.arena_bytes(), 8 * arena.slot_bytes());
        assert_eq!(arena.n_layers(), 2);
        assert_eq!(arena.block_tokens(), 128);
        // Slot 0's per-layer plane offsets == the manifest plane offsets.
        for layer in 0..2usize {
            let k = manifest.plane((layer * 2) as u32).unwrap();
            let v = manifest.plane((layer * 2 + 1) as u32).unwrap();
            assert_eq!(arena.layer_k_offset(0, layer).unwrap() as u64, k.offset);
            assert_eq!(arena.layer_v_offset(0, layer).unwrap() as u64, v.offset);
            assert_eq!(arena.k_plane_bytes() as u64, k.bytes);
            assert_eq!(arena.v_plane_bytes() as u64, v.bytes);
        }
    }

    // ── P4.1① Main+aux MTP composite ──────────────────────────────────────
    /// MTP composite fixture: 2 trunk FA layers + 1 head aux layer, Q8 geometry.
    fn mtp_resolved() -> LayerwiseMtpQ8Layout {
        LayerwiseMtpQ8Layout::resolve(2, 4, 256).unwrap()
    }

    #[test]
    fn mtp_resolve_geometry_and_head_ordinal() {
        let m = mtp_resolved();
        assert_eq!(m.n_fa_layers(), 2);
        assert_eq!(m.n_layer_total(), 3);
        assert_eq!(m.head_layer(), 2);
        assert_eq!(m.k_stride(), 1088);
        assert_eq!(m.v_stride(), 1088);
        // Main plane is the standalone trunk record (byte-identical ordinals).
        assert_eq!(m.main().fa_layers(), &[0, 1]);
        assert_eq!(m.main().k_stride(), 1088);
    }

    #[test]
    fn mtp_composite_manifest_is_main_then_aux() {
        let m = mtp_resolved();
        let manifest = m.payload_layout().unwrap();
        assert_eq!(manifest.block_tokens(), 128);
        assert_eq!(manifest.plane_count(), 6); // (2 trunk + 1 head) × {K, V}
        let plane_bytes = (1088 * 128) as u64;
        assert_eq!(manifest.record_bytes(), 6 * plane_bytes);
        // Layer-major {K,V}: trunk layers 0,1 then head aux layer 2.
        let mut offset = 0u64;
        for layer in 0..3u32 {
            for kind in [PayloadPlaneKind::Key, PayloadPlaneKind::Value] {
                let p = manifest.plane(layer * 2 + kind as u32).unwrap();
                assert_eq!(p.layer, layer);
                assert_eq!(p.kind, kind as u32);
                assert_eq!(p.offset, offset);
                assert_eq!(p.bytes, plane_bytes);
                assert_eq!(p.token_stride, 1088);
                offset += plane_bytes;
            }
        }
        assert_eq!(offset, manifest.record_bytes());
        // The head aux plane is the LAST layer (ordinal 2 == n_fa).
        let aux_k = manifest.plane(m.head_layer() * 2).unwrap();
        assert_eq!(aux_k.layer, 2);
        assert_eq!(aux_k.offset, 4 * plane_bytes);
    }

    #[test]
    fn mtp_composite_arena_matches_core_record() {
        let m = mtp_resolved();
        let manifest = m.payload_layout().unwrap();
        let arena = m.packed_arena_layout(8).unwrap();
        assert_eq!(arena.slot_bytes() as u64, manifest.record_bytes());
        assert_eq!(arena.n_layers(), 3);
        assert_eq!(arena.arena_bytes(), 8 * arena.slot_bytes());
        // Head aux plane offsets == the manifest's last-layer plane offsets.
        for slot in 0..8usize {
            let aux_k = manifest.plane(m.head_layer() * 2).unwrap();
            let aux_v = manifest.plane(m.head_layer() * 2 + 1).unwrap();
            let base = (slot * arena.slot_bytes()) as u64;
            assert_eq!(m.aux_k_offset(&arena, slot).unwrap() as u64, base + aux_k.offset);
            assert_eq!(m.aux_v_offset(&arena, slot).unwrap() as u64, base + aux_v.offset);
        }
    }

    #[test]
    fn mtp_main_plane_byte_identical_to_trunk_record() {
        let m = mtp_resolved();
        let composite = m.payload_layout().unwrap();
        let trunk = m.main().payload_layout().unwrap();
        // The first n_fa layers' planes are identical in both manifests.
        for layer in 0..m.n_fa_layers() as u32 {
            for kind in [PayloadPlaneKind::Key, PayloadPlaneKind::Value] {
                let c = composite.plane(layer * 2 + kind as u32).unwrap();
                let t = trunk.plane(layer * 2 + kind as u32).unwrap();
                assert_eq!(c.offset, t.offset);
                assert_eq!(c.bytes, t.bytes);
                assert_eq!(c.token_stride, t.token_stride);
            }
        }
        // Composite record == trunk record + one aux layer {K,V}.
        let plane_bytes = (1088 * 128) as u64;
        assert_eq!(composite.record_bytes(), trunk.record_bytes() + 2 * plane_bytes);
    }

    #[test]
    fn mtp_rejects_bad_geometry() {
        assert!(LayerwiseMtpQ8Layout::resolve(0, 4, 256).is_err()); // no trunk
        assert!(LayerwiseMtpQ8Layout::resolve(2, 4, 100).is_err()); // head_dim % 32
        assert!(LayerwiseMtpQ8Layout::resolve(2, 0, 256).is_err()); // zero KV heads
    }
}