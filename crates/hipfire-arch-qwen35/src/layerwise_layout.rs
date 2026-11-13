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
        PayloadLayout::from_raw(&self.raw_cfg, Q8_PACKED_FORMAT, &self.fa_layers)
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
}