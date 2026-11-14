// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-layer layerwise recall driver (Phase 3.2 ②-D3).
//!
//! Clean sibling to the legacy copy-only `AttentionTap` path: this module owns
//! the per-layer composite-slot recall subsystem driven by the core
//! `kvmem::LayerwiseQueryRequest` state machine (wrapped by
//! [`hipfire_kvmem::LayerwiseRequest`]). It is gated on
//! `RecallStrategy::Layerwise`; the default Legacy serve never enters it, so it
//! stays byte-neutral.
//!
//! Boundary (design §3.5/§3.9): the core library is strategy-agnostic and does
//! NOT execute tensors, copy GPU bytes, fence streams or publish state. This
//! driver does those things directly (no `MemoryBackend`/`MemoryTransfer` impl —
//! the layerwise path never consumes them): it gathers canonical Mean-K, scores
//! the borrowed native Q tile on hipfire's OWN HIP launcher (Rust→HIP, no shim),
//! reads the sealed archive's layer regions into the pinned slab, H2D-scatters
//! them into the packed composite arena stripe, fences, and reports the actual
//! post-fence `ExecutionView` back to the state machine.
//!
//! D3 builds the scorer bridge + packed arena + selection/plan orchestration;
//! D4 adds the COW stripe-write guards, drop-evict and the three paged forward
//! path integrations. Single serve thread; the FFI handles are `!Send`/`!Sync`.

use crate::kvmem_ctl::{KvMemCtl, LayerwiseQuotas};
use crate::layerwise_layout::{LayerwiseQ8Layout, LAYERWISE_BLOCK_TOKENS, Q8_PACKED_FORMAT};
use hip_bridge::{DeviceBuffer, PinnedTransferBuffer};
use hipfire_kvmem::{
    BackendIdentity, BlockCopies, BlockDescriptor, ExecutionView, HostRecord, HostSource, KvMemError,
    KvRegions, KvStore, LayerMemorySnapshot, LayerPlan, LayerQueryGeometry, LayerRequestSpec,
    LayerwiseIdentity, LayerwisePhase, LayerwiseRequest, LayerwiseWork, MemoryBudget,
    MemoryCapabilities, MemoryCatalog, MemoryDescriptor, MemorySnapshot, MemoryStamp,
    NativeCurrentLayerQuery, NativePayload, PayloadLayout, PrefillAcceptance, PrefillMass,
    PrefillPressure, QueryPlaneBytes, QueryProjectionLayout,
    QueryProjectionPlane, QueryProjectionTape, QueryRange, QueryScoreFrame, RawStore, Reservation,
    SessionIdentity, SpillFile, Storage,
};
use rdna_compute::kvmem_page_mass::PageMassQuery;
use rdna_compute::kvmem_page_score::{PageQueryScalar, PageScoreQuery};
use rdna_compute::kv_slots::KvSlotDesc;
use rdna_compute::layerwise_pool::{LayerwisePagePool, PackedArenaLayout};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashMap;

/// Map a core `NativeQueryScalar` ordinal (`native_layer_query.hpp`) to the
/// launcher's `PageQueryScalar`. The ordinals are identical (F32=0, F16=1,
/// BF16=2); an unknown ordinal fails closed rather than guessing a width.
fn page_scalar(scalar: u32) -> Result<PageQueryScalar, KvMemError> {
    match scalar {
        0 => Ok(PageQueryScalar::F32),
        1 => Ok(PageQueryScalar::F16),
        2 => Ok(PageQueryScalar::BF16),
        other => Err(KvMemError(format!(
            "layerwise scorer: unknown native query scalar ordinal {other}"
        ))),
    }
}

/// GPU scorer bridge: the core `NativeScorer` → hipfire's own HIP page-softmax
/// launcher ([`Gpu::kvmem_page_score_current_layer`], D1).
///
/// Faithful to the host reference `layerwise_mean_key_scores`
/// (`kvmem/src/host/layerwise_scoring.cpp`) / `graph_recall.cpp:333-396`:
///
/// * The frame is dense-canonical, so entry `i` is canonical block
///   `starts[i] / block_tokens`. Only `score_mask != 0` blocks enter the softmax
///   domain; `transport_mask` NEVER affects scoring (`layerwise_scoring.hpp:23`).
/// * Mean-K is gathered for the enabled blocks in canonical order into a
///   page-major `[enabled, kv_heads, head_dim]` F32 buffer. The launcher rounds
///   Mean-K to FP16 host-side (matching `recall_fp16`), so raw F32 rows are
///   passed — no pre-rounding here.
/// * The borrowed native Q tile stays on its producing device/stream; it is
///   passed by pointer + BYTE strides (token axis need not be outermost).
/// * Exactly `frame.starts.len()` scores are returned: enabled blocks carry their
///   per-page attention mass, masked blocks are 0. The core `accumulate`
///   (`layerwise_selection.cpp:156-165`) enforces finite, non-negative, masked=0.
///
/// `il` is the RawKvStore statistics-layer ordinal (the caller resolves the
/// model-layer ↔ ordinal mapping). Called synchronously by
/// [`hipfire_kvmem::LayerwiseRequest::projected_native`]'s trampoline, which drops
/// the borrowed `keys` pointer — so the driver captures its own `&RawStore`.
fn score_current_layer_native(
    gpu: &mut Gpu,
    keys: &RawStore,
    il: u32,
    block_tokens: u32,
    q: &NativeCurrentLayerQuery,
    frame: &QueryScoreFrame<'_>,
) -> Result<Vec<f64>, KvMemError> {
    let n = frame.starts.len();
    if frame.counts.len() != n || frame.score_mask.len() != n {
        return Err(KvMemError(format!(
            "layerwise scorer: frame extent mismatch (starts {n}, counts {}, mask {})",
            frame.counts.len(),
            frame.score_mask.len()
        )));
    }
    if block_tokens == 0 {
        return Err(KvMemError("layerwise scorer: zero block_tokens".to_string()));
    }
    let kw = q.kv_heads as usize * q.head_dim as usize;
    // Enabled blocks in canonical order — the launcher's softmax domain.
    let enabled: Vec<usize> = (0..n).filter(|&i| frame.score_mask[i] != 0).collect();
    let mut means = vec![0f32; enabled.len() * kw];
    let mut row = vec![0f32; kw];
    for (j, &i) in enabled.iter().enumerate() {
        let block = (frame.starts[i] / block_tokens as u64) as u32;
        keys.mean_k(block, il, &mut row);
        means[j * kw..(j + 1) * kw].copy_from_slice(&row);
    }
    let pq = PageScoreQuery {
        data: q.data,
        bytes: q.bytes,
        rows: q.rows,
        query_heads: q.query_heads,
        kv_heads: q.kv_heads,
        head_dim: q.head_dim,
        attention_scale: q.attention_scale,
        scalar: page_scalar(q.scalar)?,
        row_stride: q.row_stride,
        head_stride: q.head_stride,
        dimension_stride: q.dimension_stride,
    };
    let mut masses = vec![0f64; enabled.len()];
    gpu.kvmem_page_score_current_layer(&pq, &means, enabled.len(), &mut masses)
        .map_err(|e| KvMemError(format!("layerwise scorer: page-score launch: {e:?}")))?;
    // Scatter back to the full canonical extent; masked blocks stay 0.
    let mut out = vec![0f64; n];
    for (j, &i) in enabled.iter().enumerate() {
        out[i] = masses[j];
    }
    Ok(out)
}

/// Owner of the layerwise composite packed arena: ONE device allocation backing
/// `n_slots` composite slots, plus the free-list pool tracking per-layer stripe
/// residency over it (design §3.5). Mutually exclusive with the legacy per-layer
/// `k_arenas`/`v_arenas`; allocated only on the Layerwise path, so the default
/// Legacy serve never touches it and stays byte-neutral.
///
/// Thin by construction: the geometry lives in [`PackedArenaLayout`] and
/// residency in [`LayerwisePagePool`] (both host-tested in ①); the device bytes
/// are a single zero-filled `DType::Raw` tensor. This wrapper only ties their
/// lifetimes together and forwards the stripe-view seam. `GpuTensor` has no
/// `Drop`, so the arena is released exactly once via [`Self::free`]; the stripe
/// views are non-owning aliases and are never freed.
pub struct LayerwiseArena {
    pool: LayerwisePagePool,
    arena: GpuTensor,
    /// Compact position → physical composite slot, SHARED across every ordinal
    /// (design §3.1 mixed-stripe destinations). Position `i` is the same slot
    /// for all layers; layer `ordinal`'s stripe in that slot holds the page it
    /// selected for position `i`. Grows lazily as ordinals bind their compact
    /// next-view — the first ordinal to reach a position allocates the slot,
    /// later ordinals reuse it (one slot per position, NOT per (ordinal, page)).
    position_slots: Vec<u32>,
}

impl LayerwiseArena {
    /// Resolve the Q8 packed geometry for `n_slots` composite slots, allocate the
    /// zero-filled arena, and build the pool over it. Fails closed on bad
    /// geometry or an allocation error — never partially constructs (the pool is
    /// built only after the device tensor exists).
    pub fn new(gpu: &mut Gpu, q8: &LayerwiseQ8Layout, n_slots: usize) -> Result<Self, String> {
        let layout = q8.packed_arena_layout(n_slots)?;
        let arena = layout.alloc_tensor(gpu).map_err(|e| {
            format!(
                "layerwise arena: alloc {} bytes: {e:?}",
                layout.arena_bytes()
            )
        })?;
        let pool = LayerwisePagePool::new(layout)?;
        Ok(Self { pool, arena, position_slots: Vec::new() })
    }

    /// The packed geometry (slot/layer/plane offsets, byte sizes).
    pub fn layout(&self) -> &PackedArenaLayout {
        self.pool.layout()
    }

    /// Mutable access to the composite-slot pool (alloc/bind/drop/free, views).
    pub fn pool(&mut self) -> &mut LayerwisePagePool {
        &mut self.pool
    }

    /// Read-only access to the composite-slot pool (residency queries).
    pub fn pool_ref(&self) -> &LayerwisePagePool {
        &self.pool
    }

    /// Bind compact `position`'s composite slot for `ordinal` to canonical
    /// `page` (design §3.1 mixed-stripe destinations). The slot is SHARED across
    /// ordinals by position: the first ordinal to reach a position allocates it
    /// (`alloc_slot`), every later ordinal reuses the SAME slot and binds its own
    /// layer stripe to the page it selected for that position. This is what keeps
    /// the arena at `selected_pages + suffix` slots regardless of the FA-layer
    /// count — one slot per compact position, NOT one per (ordinal, page).
    ///
    /// Returns `(phys, fresh)`: `fresh` is true iff this call newly bound the
    /// ordinal's stripe (so it holds no valid bytes yet and MUST be restored,
    /// the device-truth invariant); false iff the stripe was already bound to
    /// this page (idempotent reuse — keeps the plan's cold/hot decision). A
    /// stripe already bound to a DIFFERENT page fails closed (`bind_stripe`
    /// rejects double-bind): the scoped per-query driver restores each ordinal
    /// exactly once into a fresh arena, so that never arises in production.
    pub fn bind_position_stripe(
        &mut self,
        position: usize,
        ordinal: usize,
        page: u32,
    ) -> Result<(u32, bool), String> {
        let phys = match self.position_slots.get(position) {
            Some(&p) => p,
            None => {
                let h = self.pool.alloc_slot().map_err(|e| {
                    format!("layerwise arena: alloc position {position}: {e}")
                })?;
                self.position_slots.push(h.phys);
                h.phys
            }
        };
        if self.pool.stripe_page(phys, ordinal) == Some(page) {
            return Ok((phys, false));
        }
        let h = self
            .pool
            .handle(phys)
            .ok_or_else(|| format!("layerwise arena: position {position} slot {phys} handle"))?;
        self.pool.bind_stripe(h, ordinal, page).map_err(|e| {
            format!("layerwise arena: bind position {position} layer {ordinal} page {page}: {e}")
        })?;
        Ok((phys, true))
    }

    /// The backing device arena (one `DType::Raw` tensor, `arena_bytes` long).
    pub fn arena(&self) -> &GpuTensor {
        &self.arena
    }

    /// Non-owning {K, V} byte views of slot `slot`'s layer-`layer` stripe, at the
    /// layout's plane offsets. The returned tensors alias the arena — never free
    /// them; release the arena itself via [`Self::free`].
    pub fn stripe_views(&self, slot: usize, layer: usize) -> Result<(GpuTensor, GpuTensor), String> {
        self.pool.layout().stripe_views(&self.arena, slot, layer)
    }

    /// The shifted V base view for the composite-arena paged addressing scheme
    /// (design §3.8a, GPU byte-exact verified in D4a). The UNCHANGED paged GQA
    /// decode kernel reads K at `k_cache + tile_off` and V at `v_cache + tile_off`
    /// with the SAME `tile_off` but SEPARATE bases. The composite layout
    /// interleaves {K,V} planes per global stripe `g = slot*n_layers + layer`
    /// (K at byte `2g*k_plane_bytes`, V at `(2g+1)*k_plane_bytes`), so pairing
    /// `k_cache = `[`Self::arena`] (base) with `v_cache = base + k_plane_bytes`
    /// makes `block_table[lp] = 2*g` address stripe `g`'s K and V planes exactly
    /// (see [`encode_layer_block_table`]).
    ///
    /// This requires equal K/V strides (Q8_0): the single shared `tile_off` only
    /// lands on both planes when `k_plane_bytes == v_plane_bytes`. A mismatched
    /// stride fails closed rather than mis-addressing V (non-q8 is deferred to
    /// P4.4). Non-owning view — never free it; release the arena via [`Self::free`].
    pub fn v_cache_view(&self) -> Result<GpuTensor, String> {
        let layout = self.pool.layout();
        if layout.k_stride() != layout.v_stride() {
            return Err(format!(
                "layerwise arena: shifted-base paged addressing requires equal K/V strides \
                 (Q8_0); got k_stride {} != v_stride {}",
                layout.k_stride(),
                layout.v_stride()
            ));
        }
        let k_plane = layout.k_plane_bytes();
        let arena_bytes = layout.arena_bytes();
        Ok(self.arena.sub_offset(k_plane, arena_bytes - k_plane))
    }

    /// Release the device arena, consuming the wrapper. The pool drops with it;
    /// stripe views are non-owning so nothing else needs freeing. Call exactly
    /// once (the arena has no `Drop`).
    pub fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        gpu.free_tensor(self.arena)
            .map_err(|e| format!("layerwise arena: free: {e:?}"))
    }
}

/// Encode one layer's paged block table for the composite-arena addressing
/// scheme (design §3.8a). `compact_slots` are the physical composite slots of
/// the layer's compact next-view (the selected history pages, in compact order);
/// `suffix_slot` is the physical slot holding the freshly-written suffix page
/// (the current prefill/decode frontier), appended as the last logical page when
/// present. `layer` is the arena layer ordinal and `n_layers` the total fa-layer
/// count; the kernel-facing physical index is `2*(slot*n_layers + layer)` — the
/// global stripe `g = slot*n_layers + layer`, whose K plane sits at `2g` and V at
/// `2g+1` (the shifted [`LayerwiseArena::v_cache_view`] base supplies the +1).
///
/// Returns a `max_pages`-long i32 table: mapped logical pages carry their stripe
/// index, the rest are `-1`. The kernel's `phys < 0` branch emits empty partials
/// (the reduce skips `p[1] <= 0`), so an unmapped/dropped logical page is masked
/// for free. Fails closed on bad geometry, an out-of-range layer, a table too
/// small for the mapped pages, or a stripe index that would overflow i32 (a
/// wrapped-negative index would be silently treated as a dropped page).
fn encode_layer_block_table(
    compact_slots: &[u32],
    suffix_slot: Option<u32>,
    layer: usize,
    n_layers: usize,
    max_pages: usize,
) -> Result<Vec<i32>, String> {
    if n_layers == 0 {
        return Err("layerwise block table: n_layers == 0".to_string());
    }
    if layer >= n_layers {
        return Err(format!(
            "layerwise block table: layer {layer} >= n_layers {n_layers}"
        ));
    }
    let mapped = compact_slots.len() + usize::from(suffix_slot.is_some());
    if mapped > max_pages {
        return Err(format!(
            "layerwise block table: {mapped} mapped pages > max_pages {max_pages}"
        ));
    }
    let stripe = |slot: u32| -> Result<i32, String> {
        // g = slot*n_layers + layer; kernel index = 2*g. Compute in u64 and
        // bound-check against i32::MAX so a huge product fails closed instead of
        // wrapping negative (which the kernel would read as a dropped page).
        let g = (slot as u64)
            .checked_mul(n_layers as u64)
            .and_then(|v| v.checked_add(layer as u64))
            .ok_or_else(|| "layerwise block table: stripe index overflow".to_string())?;
        let idx = g
            .checked_mul(2)
            .ok_or_else(|| "layerwise block table: stripe index overflow".to_string())?;
        if idx > i32::MAX as u64 {
            return Err(format!(
                "layerwise block table: stripe index {idx} exceeds i32::MAX"
            ));
        }
        Ok(idx as i32)
    };
    let mut bt = vec![-1i32; max_pages];
    for (lp, &slot) in compact_slots.iter().enumerate() {
        bt[lp] = stripe(slot)?;
    }
    if let Some(slot) = suffix_slot {
        bt[compact_slots.len()] = stripe(slot)?;
    }
    Ok(bt)
}

/// Per-layer paged descriptor staging for the Layerwise composite arena
/// (design §3.9a, ②-D4b). Clean additive sibling to the legacy
/// [`crate::forward_slots::SlotDescStaging`]: the legacy path keeps ONE
/// [`KvSlotDesc`] per sequence shared across all fa layers (same block table,
/// per-layer arenas), whereas Layerwise selects different pages per layer, so it
/// needs a PER-LAYER block table + descriptor.
///
/// Owns `n_layers` stable-address device buffers — one `max_pages`-entry i32
/// block table and one [`KvSlotDesc`] per layer. Each descriptor's `block_table`
/// pointer is fixed for the staging's lifetime (mirrors the legacy invariant),
/// so a layer's table is re-uploaded IN PLACE every step without re-pointing the
/// descriptor. The paged decode path never graph-captures
/// (`forward_slots.rs`), so these host uploads are capture-safe; the stable
/// addresses also suit replay. There is no dirty-tracking — the caller rebuilds
/// the active layers each step (batch composition legitimately changes).
///
/// Single-sequence layerwise flow: one descriptor per layer, `row_slot` maps
/// every active row to slot 0. The K/V base tensors are NOT owned here — they
/// are the shared composite arena base ([`LayerwiseArena::arena`]) and its
/// shifted V view ([`LayerwiseArena::v_cache_view`]), supplied by the caller at
/// attention time and shared across layers. `GpuTensor` has no `Drop`; release
/// exactly once via [`Self::free_gpu`].
pub struct LayerwiseDescStaging {
    descs: Vec<GpuTensor>,
    block_tables: Vec<GpuTensor>,
    /// Per-layer compacted causal bound: one i32 `[seq_len - 1]`. The ported
    /// paged kernels treat `positions[]` (NOT `desc.seq_len`) as authoritative
    /// for the causal bound, and Layerwise attends the COMPACTED selected
    /// working set — so this is the compacted visible extent, distinct from the
    /// real decode position used for RoPE (D4a parity proof: `positions=[SEQ-1]`
    /// with `SEQ` the compacted extent). Stable address, re-uploaded in place.
    positions: Vec<GpuTensor>,
    /// Single-sequence row→slot map: one i32 `[0]`, stable address. The paged
    /// writer/attention take `row_slot` to resolve each active row's descriptor;
    /// the layerwise flow has one sequence, so every row maps to slot 0 (mirrors
    /// the legacy `SlotDescStaging::row_slot_dev`). Uploaded once at construction.
    row_slot: GpuTensor,
    n_layers: usize,
    max_pages: usize,
}

impl LayerwiseDescStaging {
    /// Allocate `n_layers` stable-address {block table, descriptor} pairs. The
    /// descriptor is sized from the [`KvSlotDesc`] ABI (`size_of`), not a stale
    /// field count. Fails closed on bad geometry or an allocation error — never
    /// partially constructs (everything allocated so far is freed on failure).
    pub fn new(gpu: &mut Gpu, n_layers: usize, max_pages: usize) -> Result<Self, String> {
        if n_layers == 0 {
            return Err("layerwise desc staging: n_layers == 0".to_string());
        }
        if max_pages == 0 {
            return Err("layerwise desc staging: max_pages == 0".to_string());
        }
        let desc_bytes = std::mem::size_of::<KvSlotDesc>();
        let mut descs: Vec<GpuTensor> = Vec::with_capacity(n_layers);
        let mut block_tables: Vec<GpuTensor> = Vec::with_capacity(n_layers);
        let mut positions: Vec<GpuTensor> = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let table = match gpu.alloc_tensor(&[max_pages * 4], DType::Raw) {
                Ok(t) => t,
                Err(e) => {
                    Self::free_partial(gpu, &mut descs, &mut block_tables, &mut positions);
                    return Err(format!("layerwise desc staging: block table alloc: {e:?}"));
                }
            };
            block_tables.push(table);
            let desc = match gpu.alloc_tensor(&[desc_bytes], DType::Raw) {
                Ok(t) => t,
                Err(e) => {
                    Self::free_partial(gpu, &mut descs, &mut block_tables, &mut positions);
                    return Err(format!("layerwise desc staging: desc alloc: {e:?}"));
                }
            };
            descs.push(desc);
            let pos = match gpu.alloc_tensor(&[4], DType::Raw) {
                Ok(t) => t,
                Err(e) => {
                    Self::free_partial(gpu, &mut descs, &mut block_tables, &mut positions);
                    return Err(format!("layerwise desc staging: positions alloc: {e:?}"));
                }
            };
            positions.push(pos);
        }
        // Single-sequence row→slot map ([0]), allocated after the per-layer
        // buffers so a loop failure frees only the Vecs (free_partial). A failure
        // here frees everything allocated so far — never partially constructs.
        let row_slot = match gpu.alloc_tensor(&[4], DType::Raw) {
            Ok(t) => t,
            Err(e) => {
                Self::free_partial(gpu, &mut descs, &mut block_tables, &mut positions);
                return Err(format!("layerwise desc staging: row_slot alloc: {e:?}"));
            }
        };
        let zero = 0i32;
        if let Err(e) = gpu.hip.memcpy_htod(&row_slot.buf, unsafe {
            std::slice::from_raw_parts(&zero as *const i32 as *const u8, 4)
        }) {
            let _ = gpu.free_tensor(row_slot);
            Self::free_partial(gpu, &mut descs, &mut block_tables, &mut positions);
            return Err(format!("layerwise desc staging: row_slot H2D: {e:?}"));
        }
        Ok(Self {
            descs,
            block_tables,
            positions,
            row_slot,
            n_layers,
            max_pages,
        })
    }

    /// Re-encode and upload layer `layer`'s block table + descriptor in place.
    /// `compact_slots`/`suffix_slot` are the layer's compact next-view physical
    /// slots and optional suffix slot (see [`encode_layer_block_table`]);
    /// `seq_len` is the layer's visible token extent. The descriptor points at
    /// the layer's STABLE table address, so re-uploading never re-points it.
    /// Fails closed on an out-of-range layer, a negative `seq_len`, or a
    /// `seq_len` needing more tiles than there are mapped pages (the kernel
    /// would otherwise read an unmapped `-1` table entry as a real page).
    pub fn update_layer(
        &self,
        gpu: &mut Gpu,
        layer: usize,
        compact_slots: &[u32],
        suffix_slot: Option<u32>,
        seq_len: i32,
    ) -> Result<(), String> {
        if layer >= self.n_layers {
            return Err(format!(
                "layerwise desc staging: layer {layer} >= n_layers {}",
                self.n_layers
            ));
        }
        if seq_len < 0 {
            return Err(format!("layerwise desc staging: negative seq_len {seq_len}"));
        }
        let bt = encode_layer_block_table(
            compact_slots,
            suffix_slot,
            layer,
            self.n_layers,
            self.max_pages,
        )?;
        let mapped = compact_slots.len() + usize::from(suffix_slot.is_some());
        let tiles = (seq_len as usize).div_ceil(LAYERWISE_BLOCK_TOKENS as usize);
        if tiles > mapped {
            return Err(format!(
                "layerwise desc staging: seq_len {seq_len} needs {tiles} tiles > {mapped} \
                 mapped pages (layer {layer})"
            ));
        }
        // Upload the table in place (stable address; the descriptor already
        // points at it). bt.len() == max_pages, matching the allocation.
        gpu.hip
            .memcpy_htod(&self.block_tables[layer].buf, unsafe {
                std::slice::from_raw_parts(bt.as_ptr() as *const u8, bt.len() * 4)
            })
            .map_err(|e| format!("layerwise desc staging: table H2D layer {layer}: {e:?}"))?;
        let desc = KvSlotDesc {
            block_table: self.block_tables[layer].buf.as_ptr() as u64,
            legacy_k_base: 0,
            legacy_v_base: 0,
            seq_len,
            page_tokens: LAYERWISE_BLOCK_TOKENS as i32,
        };
        gpu.hip
            .memcpy_htod(&self.descs[layer].buf, unsafe {
                std::slice::from_raw_parts(
                    &desc as *const KvSlotDesc as *const u8,
                    std::mem::size_of::<KvSlotDesc>(),
                )
            })
            .map_err(|e| format!("layerwise desc staging: desc H2D layer {layer}: {e:?}"))?;
        // Compacted causal bound for the ported kernels (positions[] is
        // authoritative). seq_len == 0 → -1 (empty working set → empty partials).
        let pos = seq_len - 1;
        gpu.hip
            .memcpy_htod(&self.positions[layer].buf, unsafe {
                std::slice::from_raw_parts(&pos as *const i32 as *const u8, 4)
            })
            .map_err(|e| format!("layerwise desc staging: positions H2D layer {layer}: {e:?}"))?;
        Ok(())
    }

    /// Layer `layer`'s descriptor tensor (one [`KvSlotDesc`], stable address).
    pub fn descs(&self, layer: usize) -> Result<&GpuTensor, String> {
        self.descs.get(layer).ok_or_else(|| {
            format!(
                "layerwise desc staging: desc layer {layer} >= n_layers {}",
                self.n_layers
            )
        })
    }

    /// Layer `layer`'s block-table tensor (`max_pages` i32, stable address).
    pub fn block_table(&self, layer: usize) -> Result<&GpuTensor, String> {
        self.block_tables.get(layer).ok_or_else(|| {
            format!(
                "layerwise desc staging: table layer {layer} >= n_layers {}",
                self.n_layers
            )
        })
    }

    /// Layer `layer`'s compacted causal-bound tensor (one i32 `[seq_len - 1]`,
    /// stable address). Hand to the paged kernel as `positions` — it is the
    /// COMPACTED visible extent, not the real decode position (which drives RoPE).
    pub fn positions(&self, layer: usize) -> Result<&GpuTensor, String> {
        self.positions.get(layer).ok_or_else(|| {
            format!(
                "layerwise desc staging: positions layer {layer} >= n_layers {}",
                self.n_layers
            )
        })
    }

    /// The number of fa layers staged.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
    /// The per-layer block-table capacity, in pages.
    pub fn max_pages(&self) -> usize {
        self.max_pages
    }
    /// The single-sequence row→slot map (one i32 `[0]`, stable address). Hand to
    /// the paged writer/attention as `row_slot`.
    pub fn row_slot(&self) -> &GpuTensor {
        &self.row_slot
    }

    fn free_partial(
        gpu: &mut Gpu,
        descs: &mut Vec<GpuTensor>,
        block_tables: &mut Vec<GpuTensor>,
        positions: &mut Vec<GpuTensor>,
    ) {
        for t in descs.drain(..) {
            let _ = gpu.free_tensor(t);
        }
        for t in block_tables.drain(..) {
            let _ = gpu.free_tensor(t);
        }
        for t in positions.drain(..) {
            let _ = gpu.free_tensor(t);
        }
    }

    /// Release every device buffer, consuming the staging. Call exactly once
    /// (the tensors have no `Drop`). The shared K/V base views are non-owning and
    /// are NOT freed here — the arena owner releases them.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in self.descs {
            let _ = gpu.free_tensor(t);
        }
        for t in self.block_tables {
            let _ = gpu.free_tensor(t);
        }
        for t in self.positions {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.row_slot);
    }
}

/// Per-query Layerwise forward bundle, threaded through the forward spine as
/// `Option<&mut LayerwiseQuery>` exactly like `SpecVerifyCapture` (Legacy `None`
/// is byte-neutral; the whole carrier is `kvmem`-gated). Owns the driver (which
/// borrows the slot's statistics store + policy and owns the composite arena)
/// plus the per-layer descriptor staging. Constructed at the serve-loop forward
/// site from the taken-out `KvMemCtl` (D4b-4) or by a test fixture; the GPU
/// buffers have no `Drop`, so release exactly once via [`Self::free`].
///
/// All Layerwise-specific forward logic is encapsulated here and in the driver
/// — the legacy spine only threads the opaque handle and branches on it, so the
/// subsystem stays cleanly separated from the legacy copy-only path.
pub struct LayerwiseQuery<'a> {
    pub driver: LayerwiseDriver<'a>,
    pub staging: LayerwiseDescStaging,
    /// Physical composite-arena slot holding the freshly-written current-token
    /// page (the suffix). Per-query constant across layers; set by the engine
    /// orchestration (D4b-4) or a test fixture before the forward. `None` means
    /// no suffix page is mapped — only valid when the attend step does NOT write
    /// a current token (the paged decode arm always writes, so the wired Path C
    /// sets this). [`Self::restore_layer`] reads it to extend the block table.
    pub suffix_slot: Option<u32>,
    /// Logical page holding the current token (the suffix page) = `fed / 128`
    /// (design §3.9g). Per-query constant; set by the engine orchestration. The
    /// suffix stripe is restored from this page's already-fed tokens before the
    /// live current-token write. 0 when unset (page-boundary fixtures).
    pub suffix_page: u32,
    /// Tokens already fed into the suffix page before this step (`fed % 128`).
    /// The compacted causal bound extends by `partial_tokens + rows` (not just
    /// `rows`): the suffix stripe holds the partial page's fed tokens PLUS the
    /// freshly written evaluation rows. 0 at a page boundary (the D4b-3 path
    /// fixtures), where the suffix page starts empty.
    pub partial_tokens: u32,
    /// Ninfer-faithful activation-quota subsystem (③-E1): the five live quota
    /// accounts + one projected-Q tape per FA layer. `None` for the certified ②
    /// fixtures (which build the carrier without quotas); the engine orchestration
    /// attaches it under Layerwise via [`Self::attach_activation`]. Purely additive
    /// — capturing Q into host tapes never touches the arena/scorer/view, so the
    /// forward residual is byte-identical with or without it.
    pub activation: Option<LayerwiseActivation>,
    /// Pressure-mode transaction state (design §3.10a, decision A). `Some` ⇒ this
    /// step restores the persistent pressure owner's resident-only selection
    /// directly (no scorer, no query state machine) and captures attention mass
    /// for `PrefillPressure::accept`. `None` ⇒ the certified ② scorer Query path.
    /// The pressure OWNER lives in the engine orchestration (taken out of
    /// `KvMemCtl` for the step); the carrier holds only the selection inputs, the
    /// sample schedule and the per-layer captured masses. Purely additive — the
    /// Legacy spine never constructs it.
    pub pressure: Option<PressureStep>,
    /// Mode-independent attention-mass capture state (design §3.10a, E3). Present
    /// whenever a persistent pressure owner backs the step — Query OR Pressure.
    /// Mass is captured on the page-first (new-page) Query steps where the logical
    /// frontier advances (`PrefillPressure::rows` samples); mid-page Pressure steps
    /// carry an empty schedule (no-op) and only refine the keep-set. Splitting this
    /// out of [`PressureStep`] is what lets cold-start Query steps accumulate the
    /// pressure history that later enables Pressure refinement (otherwise the path
    /// deadlocks in Query forever). `None` for the certified ② fixtures.
    pub mass: Option<MassCapture>,
}

/// One ordinal's restored compact page geometry for the pressure mass packet
/// (design §3.10a). Filled by [`LayerwiseQuery::restore_layer`] in Pressure mode.
struct PressurePages {
    /// Physical composite slots of the selected history pages, compact order
    /// (the K-gather source stripes; the suffix slot is appended separately).
    compact_slots: Vec<u32>,
    /// Original logical block starts per compact page (`view.blocks[i].original_start`),
    /// with the suffix page's logical start (`suffix_page * block_tokens`) last.
    history: Vec<u64>,
    /// Compact segment starts per page (`i * block_tokens`, suffix at index
    /// `selected_pages`); the core zips these against `history` for exposure.
    starts: Vec<u64>,
}

/// Pressure-mode RESTORE selection (design §3.10a, decision A). Present only on
/// a Pressure step: the carrier restores these resident-only pages directly (no
/// scorer, no query state machine). The mass-capture state is separate
/// ([`MassCapture`]) and runs in BOTH modes so cold-start Query steps accumulate
/// the pressure history that later enables Pressure refinement.
pub struct PressureStep {
    /// Per-ordinal pressure-selected canonical page ids (the restore input).
    selected: Vec<Vec<u64>>,
    /// Per-ordinal bootstrap spec budget (the `plan_layer` envelope).
    budgets: Vec<MemoryBudget>,
}

impl PressureStep {
    /// Build a Pressure-mode restore selection. `selected`/`budgets` are
    /// per-ordinal (one entry per FA layer, equal length, nonempty).
    pub fn new(selected: Vec<Vec<u64>>, budgets: Vec<MemoryBudget>) -> Result<Self, String> {
        if selected.is_empty() || budgets.len() != selected.len() {
            return Err(format!(
                "layerwise pressure step: {} selections != {} budgets",
                selected.len(),
                budgets.len()
            ));
        }
        Ok(Self { selected, budgets })
    }
}

/// Mode-independent attention-mass capture state (design §3.10a, E3). Present
/// whenever a persistent pressure owner backs the step (Query OR Pressure). The
/// engine precomputes the sample schedule (`mass_rows` via `PrefillPressure::rows`);
/// `pages`/`masses` are filled during the forward and published by
/// [`LayerwiseQuery::accept_pressure`]. Empty `mass_rows` ⇒ the core `rows()` gate
/// skipped this step (no capture, no accept).
pub struct MassCapture {
    /// The packet's logical frontier (`full_pages * block_tokens`); the accepted
    /// history stamp frontier the pressure owner advances to.
    frontier: u64,
    /// The compact local begin of the sampled decode row
    /// (`selected_pages * block_tokens + partial_tokens`).
    local_begin: u64,
    /// The single decode evaluation row count (admission guarantees 1).
    eval_rows: u64,
    /// The admitted sample schedule; empty ⇒ no capture/accept this step.
    mass_rows: Vec<u32>,
    /// Per-ordinal restored page geometry (filled during restore_layer).
    pages: Vec<Option<PressurePages>>,
    /// Per-ordinal captured masses (`rows × pages`, f64; filled by capture_mass_layer).
    masses: Vec<Option<Vec<f64>>>,
}

impl MassCapture {
    /// Build the mass-capture state for one step boundary. `n_layers` is the FA
    /// ordinal count; `mass_rows` is the `PrefillPressure::rows` schedule.
    pub fn new(
        n_layers: usize,
        frontier: u64,
        local_begin: u64,
        eval_rows: u64,
        mass_rows: Vec<u32>,
    ) -> Self {
        Self {
            frontier,
            local_begin,
            eval_rows,
            mass_rows,
            pages: (0..n_layers).map(|_| None).collect(),
            masses: (0..n_layers).map(|_| None).collect(),
        }
    }
}

/// Owned GPU handles for one layer's paged attention binding, extracted from the
/// carrier BEFORE it is borrowed by the tap closure (design §3.9d). The paged
/// `PagedAttentionKv` references these LOCAL handles — not the carrier — so the
/// tap can `borrow_mut()` the carrier to re-upload the descriptor/positions
/// CONTENTS in place (their addresses are stable) while the attention op holds
/// these views. The tensors are non-owning aliases of the staging buffers (Raw
/// `sub_offset(0, numel)`); never free them — the staging owns the memory.
pub struct LayerwiseLayerViews {
    pub descs: GpuTensor,
    pub positions: GpuTensor,
    pub row_slot: GpuTensor,
    /// The paged tile-iteration bound: `max_pages × block_tokens`, the static
    /// page-rounded UPPER bound on the compacted extent (known before the tap
    /// fires). D4a proved the kernel masks unmapped `-1` table entries to empty
    /// partials and treats `positions[]` as the authoritative causal bound, so
    /// this upper bound is byte-exact equivalent to the exact compacted extent
    /// (at most one extra empty tile). The tap updates `descs`/`positions`
    /// contents at the stable addresses these views alias.
    pub ctx_len: usize,
}

/// The projected-Q plane id in the hipfire layerwise activation layout. The
/// tape stores one opaque F32 Q plane per FA layer (post-RMSNorm/pre-RoPE); the
/// id is an engine-owned tag the E3 output pressure reads back by plane.
const LAYERWISE_Q_PLANE: u32 = 1;
/// Engine-owned tape format tag (opaque to the core; consistent across a query).
const LAYERWISE_Q_FORMAT: &str = "hipfire-qwen35-projected-q";
/// Conservative engine-side descriptor bounds for the finite metadata-envelope
/// admission (design §3.11). These mirror ninfer's `sizeof`-based control bound:
/// they need not match vendor STL internals, only be geometry-derived, finite
/// and conservative so shrinking a quota genuinely fails admission closed.
const SOURCE_VERSION_DESC_BYTES: u64 = 64;
const FROZEN_SNAPSHOT_BASE_BYTES: u64 = 128;
const FROZEN_BLOCK_DESC_BYTES: u64 = 64;
const FEEDBACK_DESC_BYTES: u64 = 256;

/// Ninfer-faithful layerwise activation-quota subsystem (design §3.11, ③-E1).
///
/// Owns the five LIVE quota accounts and one projected-Q history tape per FA
/// layer. The tapes persist the current query's post-RMSNorm/pre-RoPE Q (one
/// decode row per layer) into quota-backed host storage so the E3 output
/// pressure can recompute per-page attention mass from it (`read_rows`). This is
/// the substrate that makes the five `HIPFIRE_KVMEM_LAYERWISE_*_MIB` quotas live
/// rather than dead knobs:
///
/// - `activation_ram` → [`Storage`] RAM backing the per-layer Q tapes (less the
///   pinned staging slice), exactly ninfer's `ram_tapes = host_bytes - pin_bytes`.
/// - `activation_disk` → [`SpillFile`] disk backing when the tape peak exceeds
///   the RAM account (ninfer's `use_disk` tier).
/// - `activation_metadata` → [`Storage`] + a finite control [`Reservation`]; each
///   tape additionally reserves its `metadata_bound` here before any payload.
/// - `staging` → the pinned [`PinnedTransferBuffer`] the Q D2H readback lands in
///   before each `append` (ninfer's `PinnedHostBuffer(pin_bytes)`).
/// - `metadata` → the retained source-version / frozen-index / feedback envelope,
///   partitioned ninfer-style 5/8 – 1/8 – 1/8 – 1/8 (source / frozen / feedback /
///   pressure) with finite-reservation admission. The last eighth is the
///   SESSION-lifetime pressure owner's partition (③-E4), carved from the source
///   share so the two lifetimes jointly fit one quota with no double-spend.
///
/// Purely ADDITIVE: it never touches the composite arena, the GPU scorer, or the
/// resident view, so the Legacy path and the certified ② forward stay
/// byte-neutral. Created only under Layerwise (the carrier's `attach_activation`);
/// every account is a per-query resource released when the carrier is freed.
pub struct LayerwiseActivation {
    /// activation_ram: RAM backing for the Q tapes (`None` on the disk tier).
    ram: Option<Storage>,
    /// activation_disk: disk backing when the tape peak exceeds the RAM account.
    disk: Option<SpillFile>,
    /// activation_metadata: finite tape descriptor/chunk-table account.
    metadata: Storage,
    /// control_bytes reserved from `metadata` before any tape payload (ninfer
    /// `metadata_controls`); held for the query lifetime.
    controls: Reservation,
    /// metadata envelope partitions (5th quota): source-version / frozen-index /
    /// feedback finite reservations, ninfer-style 5/8 – 1/8 – 1/8 (the remaining
    /// 1/8 is the SESSION-lifetime pressure partition, ③-E4).
    source_versions: Reservation,
    frozen_index: Reservation,
    feedback: Reservation,
    /// The metadata envelope account backing the three partitions above; retained
    /// for the query so the reservations stay charged against a live account.
    envelope: Storage,
    /// staging: pinned Q D2H readback buffer (`pin_bytes`).
    staging: PinnedTransferBuffer,
    /// One projected-Q tape per FA ordinal (model_layer == ordinal).
    tapes: Vec<QueryProjectionTape>,
    layout: QueryProjectionLayout,
    identity: LayerwiseIdentity,
    q_row_bytes: usize,
    use_disk: bool,
}

/// The admitted activation plan: every quota bound and the RAM/disk tier
/// decision computed by [`LayerwiseActivation::admit`] BEFORE any resource is
/// acquired. Pure data (no accounts, no GPU), so admission is host-testable and
/// fail-closed — a quota too small for the model geometry yields `Err` with
/// nothing allocated. [`LayerwiseActivation::new`] allocates strictly against it.
struct ActivationPlan {
    layout: QueryProjectionLayout,
    max_chunks: u64,
    control_bytes: u64,
    pin_bytes: u64,
    ram_tapes: u64,
    use_disk: bool,
    source_bound: u64,
    frozen_bound: u64,
    feedback_bound: u64,
}

impl LayerwiseActivation {
    /// Pure admission (ninfer-faithful): compute every quota bound and the
    /// RAM/disk tier decision from the model geometry, validating ALL five
    /// quotas before the first allocation. No `Gpu`, no accounts — a too-small
    /// quota returns `Err` with nothing acquired, so the fail-closed contract is
    /// host-testable. [`Self::new`] allocates strictly against this plan.
    fn admit(
        identity: &LayerwiseIdentity,
        n_layers: usize,
        full_pages: u64,
        q_row_bytes: usize,
        quotas: &LayerwiseQuotas,
    ) -> Result<ActivationPlan, String> {
        if n_layers == 0 || q_row_bytes == 0 {
            return Err("layerwise activation: zero-sized geometry".to_string());
        }
        let eval_rows = identity
            .evaluation
            .end
            .checked_sub(identity.evaluation.begin)
            .filter(|r| *r > 0)
            .ok_or_else(|| "layerwise activation: empty evaluation span".to_string())?;
        let layout = QueryProjectionLayout {
            format: LAYERWISE_Q_FORMAT.to_string(),
            version: 1,
            planes: vec![QueryProjectionPlane {
                id: LAYERWISE_Q_PLANE,
                row_stride: q_row_bytes as u64,
            }],
        };
        // One decode tile covers the whole evaluation span in a single chunk.
        let max_chunks: u64 = 1;
        let q_row_bytes_u64 = q_row_bytes as u64;
        let per_tape_bytes = eval_rows
            .checked_mul(q_row_bytes_u64)
            .ok_or_else(|| "layerwise activation: tape row bytes overflow".to_string())?;
        let tape_peak = per_tape_bytes
            .checked_mul(n_layers as u64)
            .ok_or_else(|| "layerwise activation: tape peak overflow".to_string())?;
        let per_tape_meta = QueryProjectionTape::metadata_bound(&layout, max_chunks)
            .map_err(|e| format!("layerwise activation: metadata bound: {e}"))?;

        // ── activation_metadata: finite control reservation + per-tape bounds ──
        let control_bytes = (std::mem::size_of::<Self>() as u64)
            .saturating_add((n_layers as u64).saturating_mul(
                (std::mem::size_of::<QueryProjectionTape>() as u64).saturating_add(64),
            ))
            .saturating_add((layout.format.len() as u64).saturating_mul(2))
            .saturating_add(256);
        let metadata_peak = control_bytes
            .checked_add(per_tape_meta.saturating_mul(n_layers as u64))
            .ok_or_else(|| "layerwise activation: metadata peak overflow".to_string())?;
        if quotas.activation_metadata == 0 || metadata_peak > quotas.activation_metadata {
            return Err(format!(
                "layerwise activation: complete activation metadata {metadata_peak}B exceeds \
                 activation_metadata quota {}B; raise HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_METADATA_MIB",
                quotas.activation_metadata
            ));
        }

        // ── metadata envelope (5th quota): explicit 5/8 – 1/8 – 1/8 – 1/8 partition ──
        // source / frozen / feedback / pressure. The pressure partition (the
        // last transient eighth) is reserved by the SESSION-lifetime pressure
        // owner's finite admission (③-E4, `resolve_pressure_admission`), not by
        // this per-query activation; carving it out of the source share here
        // keeps the envelope finitely reserved with no double-spend across the
        // two lifetimes (per-query source/frozen/feedback + per-session pressure
        // jointly ≤ the metadata quota).
        let transient = quotas.metadata / 8;
        if transient == 0 {
            return Err(
                "layerwise activation: metadata quota is too small (HIPFIRE_KVMEM_LAYERWISE_METADATA_MIB)"
                    .to_string(),
            );
        }
        let source_cap = quotas.metadata - 3 * transient;
        let source_bound = full_pages.saturating_mul(SOURCE_VERSION_DESC_BYTES);
        let frozen_bound = (n_layers as u64).saturating_mul(
            FROZEN_SNAPSHOT_BASE_BYTES.saturating_add(full_pages.saturating_mul(FROZEN_BLOCK_DESC_BYTES)),
        );
        let feedback_bound = (n_layers as u64).saturating_mul(FEEDBACK_DESC_BYTES);
        if source_bound > source_cap || frozen_bound > transient || feedback_bound > transient {
            return Err(format!(
                "layerwise activation: retained metadata envelope (source {source_bound}B / \
                 frozen {frozen_bound}B / feedback {feedback_bound}B) exceeds the partitioned \
                 metadata quota {}B; raise HIPFIRE_KVMEM_LAYERWISE_METADATA_MIB",
                quotas.metadata
            ));
        }

        // ── staging: pinned Q readback (ninfer pin_bytes = chunk * row_bytes) ──
        let pin_bytes = per_tape_bytes;
        if pin_bytes > quotas.staging || pin_bytes > quotas.activation_ram {
            return Err(format!(
                "layerwise activation: Q staging {pin_bytes}B exceeds staging quota {}B or \
                 activation_ram {}B; raise HIPFIRE_KVMEM_LAYERWISE_STAGING_MIB / ACTIVATION_RAM_MIB",
                quotas.staging, quotas.activation_ram
            ));
        }

        // ── activation tier: RAM unless the tape peak exceeds ram_tapes ──
        let ram_tapes = quotas.activation_ram - pin_bytes;
        let use_disk = if tape_peak <= ram_tapes {
            false
        } else {
            if tape_peak > quotas.activation_disk {
                return Err(format!(
                    "layerwise activation: tape peak {tape_peak}B exceeds activation_disk quota \
                     {}B; raise HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_DISK_MIB",
                    quotas.activation_disk
                ));
            }
            if quotas.disk_directory.is_none() {
                return Err(
                    "layerwise activation: tape peak exceeds RAM but no disk directory \
                     (set HIPFIRE_KVMEM_SPILL_DIR)"
                        .to_string(),
                );
            }
            true
        };

        Ok(ActivationPlan {
            layout,
            max_chunks,
            control_bytes,
            pin_bytes,
            ram_tapes,
            use_disk,
            source_bound,
            frozen_bound,
            feedback_bound,
        })
    }

    /// Construct the activation subsystem for one query against the admitted
    /// plan. `n_layers` is the FA ordinal count, `full_pages` the sealed
    /// canonical page count (the source-version envelope geometry), `q_row_bytes`
    /// the F32 Q plane row width (`n_heads * head_dim * 4`). ALL quota arithmetic
    /// and tier admission precede the first allocation (ninfer-faithful
    /// fail-closed, [`Self::admit`]): a quota too small for the model geometry
    /// returns `Err` rather than partially constructing.
    pub fn new(
        gpu: &mut Gpu,
        identity: LayerwiseIdentity,
        n_layers: usize,
        full_pages: u64,
        q_row_bytes: usize,
        quotas: &LayerwiseQuotas,
    ) -> Result<Self, String> {
        let plan = Self::admit(&identity, n_layers, full_pages, q_row_bytes, quotas)?;

        // ── activation_metadata account + finite control reservation ──
        let metadata = Storage::new_ram(quotas.activation_metadata)
            .map_err(|e| format!("layerwise activation: metadata account: {e}"))?;
        let controls = metadata
            .reserve(plan.control_bytes)
            .map_err(|e| format!("layerwise activation: control reservation: {e}"))?;

        // ── metadata envelope (5th quota): three finite partitions ──
        let envelope = Storage::new_ram(quotas.metadata)
            .map_err(|e| format!("layerwise activation: metadata envelope: {e}"))?;
        let source_versions = envelope
            .reserve(plan.source_bound)
            .map_err(|e| format!("layerwise activation: source-version reservation: {e}"))?;
        let frozen_index = envelope
            .reserve(plan.frozen_bound)
            .map_err(|e| format!("layerwise activation: frozen-index reservation: {e}"))?;
        let feedback = envelope
            .reserve(plan.feedback_bound)
            .map_err(|e| format!("layerwise activation: feedback reservation: {e}"))?;

        // ── staging: pinned Q readback ──
        let staging = gpu
            .hip
            .transfer_buffer(plan.pin_bytes as usize)
            .map_err(|e| format!("layerwise activation: pinned staging {}B: {e:?}", plan.pin_bytes))?;

        // ── activation tier account (RAM, or disk when the peak exceeds it) ──
        let (ram, disk) = if plan.use_disk {
            let dir = quotas
                .disk_directory
                .as_ref()
                .expect("admitted disk tier has a directory");
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("layerwise activation: disk dir {}: {e}", dir.display()))?;
            let disk = SpillFile::new(dir, quotas.activation_disk)
                .map_err(|e| format!("layerwise activation: disk account: {e}"))?;
            (None, Some(disk))
        } else {
            let ram = Storage::new_ram(plan.ram_tapes)
                .map_err(|e| format!("layerwise activation: RAM account {}B: {e}", plan.ram_tapes))?;
            (Some(ram), None)
        };

        // ── one projected-Q tape per FA ordinal (admits against the accounts) ──
        let mut tapes = Vec::with_capacity(n_layers);
        for ordinal in 0..n_layers {
            let model_layer = ordinal as u32;
            let tape = if plan.use_disk {
                QueryProjectionTape::new_disk(
                    disk.as_ref().expect("disk tier has a spill account"),
                    &metadata,
                    plan.max_chunks,
                    &identity,
                    model_layer,
                    &plan.layout,
                )
            } else {
                QueryProjectionTape::new_ram(
                    ram.as_ref().expect("RAM tier has a storage account"),
                    &metadata,
                    plan.max_chunks,
                    &identity,
                    model_layer,
                    &plan.layout,
                )
            }
            .map_err(|e| format!("layerwise activation: tape {ordinal}: {e}"))?;
            tapes.push(tape);
        }

        Ok(Self {
            ram,
            disk,
            metadata,
            controls,
            source_versions,
            frozen_index,
            feedback,
            envelope,
            staging,
            tapes,
            layout: plan.layout,
            identity,
            q_row_bytes,
            use_disk: plan.use_disk,
        })
    }

    /// Persist ordinal `ordinal`'s projected Q into its tape. `q` is the same
    /// post-RMSNorm/pre-RoPE native Q tile the scorer reads (a live GPU pointer);
    /// it is D2H-read back into the pinned staging slice (blocking, race-free)
    /// and appended as the layer's single evaluation chunk, then finished to
    /// assert exact coverage. Additive: the arena/scorer/view are untouched.
    pub fn capture_layer(
        &mut self,
        gpu: &mut Gpu,
        ordinal: usize,
        q: &NativeCurrentLayerQuery,
    ) -> Result<(), String> {
        let Self {
            tapes,
            staging,
            identity,
            q_row_bytes,
            ..
        } = self;
        let tape = tapes
            .get_mut(ordinal)
            .ok_or_else(|| format!("layerwise activation: ordinal {ordinal} out of range"))?;
        let model_layer = tape.model_layer();
        let begin = identity.evaluation.begin;
        let count = identity.evaluation.end - begin;
        let bytes = (count as usize)
            .checked_mul(*q_row_bytes)
            .ok_or_else(|| "layerwise activation: capture bytes overflow".to_string())?;
        if bytes > staging.size() || bytes > q.bytes {
            return Err(format!(
                "layerwise activation: Q readback {bytes}B exceeds staging {}B or query {}B",
                staging.size(),
                q.bytes
            ));
        }
        // Non-owning view of the live Q GPU tile (the scorer reads the same ptr).
        // SAFETY: `q.data` is valid GPU memory for `q.bytes` for this call (the
        // native query contract); the view is borrowed and never freed.
        let q_buf = unsafe { DeviceBuffer::from_raw(q.data as *mut std::ffi::c_void, q.bytes) };
        let host = staging.as_bytes_mut();
        gpu.hip
            .memcpy_dtoh(&mut host[..bytes], &q_buf)
            .map_err(|e| format!("layerwise activation: Q D2H {ordinal}: {e:?}"))?;
        let plane = QueryPlaneBytes {
            id: LAYERWISE_Q_PLANE,
            data: &host[..bytes],
        };
        let chunk = tape
            .append(identity, model_layer, begin, count, &[plane])
            .map_err(|e| format!("layerwise activation: append {ordinal}: {e}"))?;
        // The tape retains coverage; the exported chunk handle is not needed here
        // (E3 reads back through the tape). Drop it to release the short lease.
        drop(chunk);
        tape.finish(identity, model_layer)
            .map_err(|e| format!("layerwise activation: finish {ordinal}: {e}"))?;
        Ok(())
    }

    /// Read back ordinal `ordinal`'s persisted Q plane (E3 output pressure). The
    /// destination must hold `count * q_row_bytes` bytes; `begin`/`count` are
    /// token rows within the evaluation span. Fails closed on a foreign range.
    pub fn read_layer(
        &self,
        ordinal: usize,
        begin: u64,
        count: u64,
        dest: &mut [u8],
    ) -> Result<(), String> {
        let tape = self
            .tapes
            .get(ordinal)
            .ok_or_else(|| format!("layerwise activation: ordinal {ordinal} out of range"))?;
        tape.read_rows(
            &self.identity,
            tape.model_layer(),
            LAYERWISE_Q_PLANE,
            begin,
            count,
            dest,
        )
        .map_err(|e| format!("layerwise activation: read {ordinal}: {e}"))
    }

    /// The Q plane row width in bytes (`n_heads * head_dim * 4`).
    pub fn q_row_bytes(&self) -> usize {
        self.q_row_bytes
    }
    /// Number of per-layer tapes (FA ordinal count).
    pub fn n_layers(&self) -> usize {
        self.tapes.len()
    }
    /// Whether the tape peak forced the disk tier.
    pub fn use_disk(&self) -> bool {
        self.use_disk
    }
    /// The query identity the tapes were admitted under.
    pub fn identity(&self) -> LayerwiseIdentity {
        self.identity
    }
    /// The engine-owned projected-Q layout.
    pub fn layout(&self) -> &QueryProjectionLayout {
        &self.layout
    }

    // ── account introspection (the five quotas are live, charged accounts) ──
    // These read the RAII owner fields so the finite charges are observable for
    // admission tests and E3 diagnostics; the accounts are held for the query so
    // the tape reservations stay backed.

    /// activation_ram charge (0 on the disk tier).
    pub fn ram_charged_bytes(&self) -> u64 {
        self.ram.as_ref().map(|s| s.charged_bytes()).unwrap_or(0)
    }
    /// activation_disk charge (0 on the RAM tier).
    pub fn disk_charged_bytes(&self) -> u64 {
        self.disk.as_ref().map(|s| s.charged_bytes()).unwrap_or(0)
    }
    /// activation_metadata charge (control reservation + per-tape metadata bounds).
    pub fn metadata_charged_bytes(&self) -> u64 {
        self.metadata.charged_bytes()
    }
    /// metadata envelope charge (source-version + frozen-index + feedback).
    pub fn envelope_charged_bytes(&self) -> u64 {
        self.envelope.charged_bytes()
    }
    /// Unused control reservation (activation_metadata).
    pub fn control_remaining(&self) -> u64 {
        self.controls.remaining()
    }
    /// Unused source-version partition (metadata envelope).
    pub fn source_versions_remaining(&self) -> u64 {
        self.source_versions.remaining()
    }
    /// Unused frozen-index partition (metadata envelope).
    pub fn frozen_index_remaining(&self) -> u64 {
        self.frozen_index.remaining()
    }
    /// Unused feedback partition (metadata envelope).
    pub fn feedback_remaining(&self) -> u64 {
        self.feedback.remaining()
    }
}

impl<'a> LayerwiseQuery<'a> {
    pub fn new(driver: LayerwiseDriver<'a>, staging: LayerwiseDescStaging) -> Self {
        Self {
            driver,
            staging,
            suffix_slot: None,
            suffix_page: 0,
            partial_tokens: 0,
            activation: None,
            pressure: None,
            mass: None,
        }
    }

    /// Attach the activation-quota subsystem (③-E1) after [`Self::build`]. Derives
    /// the tape geometry from the admitted query (FA ordinal count, sealed
    /// canonical page count, evaluation identity) and the caller-supplied F32 Q
    /// row width (`n_heads * head_dim * 4`). The engine orchestration calls this
    /// under Layerwise; the certified ② fixtures skip it (carrier stays `None`).
    /// Fails closed if a quota cannot admit the model geometry.
    pub fn attach_activation(
        &mut self,
        gpu: &mut Gpu,
        quotas: &LayerwiseQuotas,
        q_row_bytes: usize,
    ) -> Result<(), String> {
        let identity = self
            .driver
            .query_identity()
            .ok_or_else(|| "layerwise attach_activation: no active query".to_string())?;
        let n_layers = self.driver.canonical().len();
        let full_pages = self
            .driver
            .canonical()
            .first()
            .map(|s| s.memory.blocks.len() as u64)
            .unwrap_or(0);
        self.activation = Some(LayerwiseActivation::new(
            gpu, identity, n_layers, full_pages, q_row_bytes, quotas,
        )?);
        Ok(())
    }

    /// The attached activation subsystem (E3 output pressure reads its tapes).
    pub fn activation(&self) -> Option<&LayerwiseActivation> {
        self.activation.as_ref()
    }
    pub fn activation_mut(&mut self) -> Option<&mut LayerwiseActivation> {
        self.activation.as_mut()
    }

    /// Restore ordinal `ordinal`'s selected pages into the composite arena and
    /// publish its per-layer block table + compacted causal bound. `q` is the
    /// post-norm/pre-RoPE Q-tile descriptor (the scorer's native query). The
    /// suffix page comes from [`Self::suffix_slot`]; the causal bound is the
    /// compacted visible extent INCLUDING the suffix: `visible_tokens` (Σ
    /// selected historical page tokens) plus the evaluation span `rows` when a
    /// suffix page is mapped. `positions=[extent-1]` then places the freshly
    /// written current token at the suffix page's offset 0 (logical index
    /// `visible_tokens`). Returns that compacted extent. Fails closed without an
    /// active query (the evaluation span is read from its identity).
    pub fn restore_layer(
        &mut self,
        gpu: &mut Gpu,
        ordinal: usize,
        q: &NativeCurrentLayerQuery,
    ) -> Result<u64, String> {
        // Pressure mode (design §3.10a): restore the persistent owner's
        // resident-only selection directly (no scorer, no query state machine);
        // Query mode: the certified ② scorer projection. Extract the pressure
        // inputs first so the driver borrow never overlaps the carrier borrow.
        let pressure_input = self
            .pressure
            .as_ref()
            .map(|p| (p.selected[ordinal].clone(), p.budgets[ordinal]));
        let view = if let Some((selected, budget)) = pressure_input {
            self.driver.restore_layer_selected(gpu, ordinal, &selected, budget)?
        } else {
            self.driver.project_restore(gpu, ordinal, q)?
        };
        // ③-E1: persist this layer's projected Q into its quota-backed tape (the
        // E3 output-pressure recompute base). Additive — a separate host copy of
        // the same Q tile the scorer just read; the arena/view are untouched, so
        // the compacted extent and residual below are byte-identical. Pressure
        // mode attaches no activation (it reads the live post-RoPE Q for mass).
        if let Some(act) = self.activation.as_mut() {
            act.capture_layer(gpu, ordinal, q)?;
        }
        let compact = self
            .driver
            .arena()
            .pool_ref()
            .compact_slots(ordinal)?
            .to_vec();
        // Pressure mode runs no query state machine, so the evaluation row count
        // comes from the mass-capture identity; Query mode reads it from the
        // active query (the certified ② path, byte-identical with/without mass).
        let rows = if self.pressure.is_some() {
            self.mass
                .as_ref()
                .ok_or("layerwise pressure restore without mass capture")?
                .eval_rows
        } else {
            self.driver
                .evaluation_span()
                .ok_or("layerwise query: restore_layer without an active query")?
                .1
        };
        // Record the restored compact page geometry for the pressure mass packet
        // (history = original logical starts, suffix last; starts = compact
        // segment offsets). The K-gather sources are the compact physical slots.
        // Runs in BOTH modes whenever a pressure owner backs the step (Query
        // new-page steps capture mass; Pressure mid-page steps carry an empty
        // schedule but still record geometry for the keep-set).
        if let Some(m) = self.mass.as_mut() {
            let bt = self.driver.block_tokens() as u64;
            let mut history: Vec<u64> =
                view.blocks.iter().map(|b| b.block.original_start).collect();
            history.push(self.suffix_page as u64 * bt);
            let starts: Vec<u64> = (0..history.len() as u64).map(|i| i * bt).collect();
            m.pages[ordinal] = Some(PressurePages { compact_slots: compact.clone(), history, starts });
        }
        // The paged decode arm always writes a current token, so the wired path
        // sets suffix_slot; `None` (no suffix page) attends history only. The
        // suffix stripe holds the partial page's already-fed tokens (restored
        // here for mid-page decode) PLUS the freshly written evaluation rows, so
        // the causal bound extends by `partial_tokens + rows` (design §3.9g). At
        // a page boundary partial_tokens == 0 and the restore is a no-op.
        if let Some(slot) = self.suffix_slot {
            self.driver.restore_suffix_page(
                gpu,
                ordinal,
                slot,
                self.suffix_page,
                self.partial_tokens,
            )?;
        }
        let suffix_extent = if self.suffix_slot.is_some() {
            self.partial_tokens as u64 + rows
        } else {
            0
        };
        let extent = view.visible_tokens + suffix_extent;
        let seq_len = i32::try_from(extent).map_err(|_| {
            format!("layerwise query: compacted extent {extent} exceeds i32")
        })?;
        self.staging
            .update_layer(gpu, ordinal, &compact, self.suffix_slot, seq_len)?;
        Ok(extent)
    }

    /// Report ordinal `ordinal`'s attended evaluation span to the state machine
    /// and close its attending phase. Reads the span from the active query's
    /// identity (the same `rows` [`Self::restore_layer`] added to the compacted
    /// causal bound), so the engine need not re-derive it. Call AFTER the paged
    /// attention has consumed the restored working set. Pressure mode runs no
    /// query state machine, so this is a no-op there (the mass hook captures the
    /// attended boundary separately).
    pub fn attend_complete_layer(&mut self, ordinal: usize) -> Result<(), String> {
        if self.pressure.is_some() {
            return Ok(());
        }
        let (begin, rows) = self
            .driver
            .evaluation_span()
            .ok_or("layerwise query: attend_complete without an active query")?;
        self.driver.attend_complete(ordinal, begin, rows)
    }

    /// Whether this carrier runs the Pressure-mode restore (design §3.10a).
    pub fn is_pressure(&self) -> bool {
        self.pressure.is_some()
    }

    /// Capture ordinal `ordinal`'s per-page attention mass (design §3.10a, E3).
    /// Call at the post-attention boundary (post-RoPE `q`, restored arena K).
    /// No-op when no pressure owner backs the step or the `rows()` schedule is
    /// empty (most decode steps). Gathers the selected K stripes + suffix and
    /// runs the E2 mass kernel, storing `rows × pages` f64 masses for
    /// [`Self::accept_pressure`].
    pub fn capture_mass_layer(
        &mut self,
        gpu: &mut Gpu,
        ordinal: usize,
        q: &PageMassQuery,
    ) -> Result<(), String> {
        let (mass_rows, local_begin, compact_slots, suffix_slot) = {
            let m = match self.mass.as_ref() {
                Some(m) => m,
                None => return Ok(()),
            };
            if m.mass_rows.is_empty() {
                return Ok(());
            }
            let pages = m.pages[ordinal]
                .as_ref()
                .ok_or_else(|| format!("layerwise mass: layer {ordinal} not restored"))?;
            (m.mass_rows.clone(), m.local_begin, pages.compact_slots.clone(), self.suffix_slot)
        };
        let rows: Vec<i32> = mass_rows.iter().map(|&r| r as i32).collect();
        let positions: Vec<i32> = mass_rows
            .iter()
            .map(|&r| (local_begin + r as u64) as i32)
            .collect();
        let masses = self.driver.page_mass(
            gpu,
            ordinal,
            q,
            &compact_slots,
            suffix_slot,
            &rows,
            &positions,
        )?;
        let masses64: Vec<f64> = masses.iter().map(|&m| m as f64).collect();
        if let Some(m) = self.mass.as_mut() {
            m.masses[ordinal] = Some(masses64);
        }
        Ok(())
    }

    /// Accept the whole-model captured mass into the persistent pressure owner
    /// (design §3.10a). Builds one [`PrefillAcceptance`] from every FA layer's
    /// captured packet and publishes it at the complete native boundary. Returns
    /// `Ok(false)` (no-op) when no pressure owner backs the step or the `rows()`
    /// schedule is empty (mid-page Pressure steps); `Ok(true)` when the owner
    /// accepted; `Err` on a validation throw. Call AFTER the forward (every
    /// layer's mass captured), BEFORE the carrier is freed.
    pub fn accept_pressure(&mut self, pressure: &mut PrefillPressure) -> Result<bool, String> {
        let m = match self.mass.as_ref() {
            Some(m) => m,
            None => return Ok(false),
        };
        if m.mass_rows.is_empty() {
            return Ok(false);
        }
        let fa_layers = self.driver.fa_layers();
        let mut layers = Vec::with_capacity(fa_layers.len());
        for ordinal in 0..fa_layers.len() {
            let pages = m.pages[ordinal]
                .as_ref()
                .ok_or_else(|| format!("layerwise pressure accept: layer {ordinal} not restored"))?;
            let masses = m.masses[ordinal]
                .as_ref()
                .ok_or_else(|| format!("layerwise pressure accept: layer {ordinal} mass not captured"))?;
            layers.push(PrefillMass {
                model_layer: fa_layers[ordinal],
                starts: pages.starts.clone(),
                history: pages.history.clone(),
                masses: masses.clone(),
            });
        }
        // The accepted history stamp: the frozen canonical identity with frontier
        // at this step's complete native boundary (== bootstrap full_pages*block).
        let mut stamp = self.driver.canonical()[0].memory.stamp;
        stamp.frontier = m.frontier;
        let acceptance = PrefillAcceptance {
            history: stamp,
            count: m.eval_rows,
            accepted: m.eval_rows,
            local_begin: m.local_begin,
            rows: m.mass_rows.clone(),
            layers,
        };
        pressure
            .accept(&acceptance)
            .map_err(|e| format!("layerwise pressure accept: {e}"))
    }

    /// The per-ordinal restored canonical page ids (the keep-set contribution of
    /// this step). Pressure mode ⇒ the pressure selection (`choice.selected`);
    /// Query mode ⇒ the scorer's restored view blocks. The engine orchestration
    /// unions these with the previous keep-set on a Query step (pool growth) or
    /// replaces the keep-set on a Pressure step (truncate to budget). The suffix
    /// page is excluded (it is not a canonical block).
    pub fn restored_keep_set(&self) -> Result<Vec<Vec<u64>>, String> {
        let n = self.driver.fa_layers().len();
        let bt = self.driver.block_tokens() as u64;
        let mut out = Vec::with_capacity(n);
        if self.pressure.is_some() {
            let m = self
                .mass
                .as_ref()
                .ok_or("layerwise keep-set: pressure step without mass capture")?;
            for ordinal in 0..n {
                let pages = m.pages[ordinal]
                    .as_ref()
                    .ok_or_else(|| format!("layerwise keep-set: layer {ordinal} not restored"))?;
                // history last entry is the suffix; the rest are original starts.
                let ids: Vec<u64> = pages.history[..pages.history.len() - 1]
                    .iter()
                    .map(|&s| s / bt)
                    .collect();
                out.push(ids);
            }
        } else {
            for ordinal in 0..n {
                let view = self
                    .driver
                    .view(ordinal)
                    .ok_or_else(|| format!("layerwise keep-set: layer {ordinal} not restored"))?;
                out.push(view.blocks.iter().map(|b| b.block.id).collect());
            }
        }
        Ok(out)
    }

    /// Extract ordinal `ordinal`'s owned paged-binding handles (see
    /// [`LayerwiseLayerViews`]). Call BEFORE moving the carrier into the tap's
    /// `RefCell`: the returned views alias the staging's stable-address buffers,
    /// so the attention op can hold them while the tap re-uploads their contents.
    pub fn layer_views(&self, ordinal: usize) -> Result<LayerwiseLayerViews, String> {
        let descs = self.staging.descs(ordinal)?;
        let positions = self.staging.positions(ordinal)?;
        let row_slot = self.staging.row_slot();
        Ok(LayerwiseLayerViews {
            descs: descs.sub_offset(0, descs.numel()),
            positions: positions.sub_offset(0, positions.numel()),
            row_slot: row_slot.sub_offset(0, row_slot.numel()),
            ctx_len: self.staging.max_pages() * LAYERWISE_BLOCK_TOKENS as usize,
        })
    }

    /// Build a per-query carrier from the engine's bootstrap (design §3.9g):
    /// construct the scoped driver over the borrowed slot stores, admit the
    /// query, allocate the suffix slot, and record the suffix page geometry.
    /// The caller owns the returned carrier and MUST [`Self::free`] it exactly
    /// once after the forward (the GPU buffers have no `Drop`). `n_slots` is the
    /// composite-arena slot count (selected budget + suffix headroom); the
    /// per-layer block table is sized to the same bound.
    pub fn build(
        gpu: &mut Gpu,
        keys: &'a RawStore,
        policy: &'a KvStore,
        q8: &LayerwiseQ8Layout,
        backend: BackendIdentity,
        capabilities: MemoryCapabilities,
        bootstrap: LayerwiseBootstrap,
        n_slots: usize,
        archive_bytes: u64,
    ) -> Result<Self, String> {
        Self::build_with_pressure(
            gpu, keys, policy, q8, backend, capabilities, bootstrap, n_slots, archive_bytes, None, None,
        )
    }

    /// [`Self::build`] with an explicit Pressure-mode transaction (design §3.10a).
    /// `pressure == None` admits the ② scorer Query (begin_query); `Some(step)`
    /// skips the query state machine entirely — the carrier restores the
    /// pressure-selected pages directly via [`LayerwiseDriver::restore_layer_selected`]
    /// and captures mass for the persistent owner. `mass` is the mode-independent
    /// attention-mass capture state (present in BOTH modes when a pressure owner
    /// backs the step; `None` for the certified ② fixtures). The activation-quota
    /// subsystem is attached separately by the engine ONLY in Query mode (it
    /// admits under the query identity); Pressure mode reads the live post-RoPE Q
    /// for mass.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_pressure(
        gpu: &mut Gpu,
        keys: &'a RawStore,
        policy: &'a KvStore,
        q8: &LayerwiseQ8Layout,
        backend: BackendIdentity,
        capabilities: MemoryCapabilities,
        bootstrap: LayerwiseBootstrap,
        n_slots: usize,
        archive_bytes: u64,
        pressure: Option<PressureStep>,
        mass: Option<MassCapture>,
    ) -> Result<Self, String> {
        let n_layers = bootstrap.canonical.len();
        let mut driver = LayerwiseDriver::new(
            gpu,
            keys,
            policy,
            q8,
            backend,
            capabilities,
            bootstrap.canonical,
            n_slots,
            archive_bytes,
        )?;
        if pressure.is_none() {
            driver.begin_query(
                bootstrap.identity,
                bootstrap.specs,
                bootstrap.native_history_tokens,
            )?;
        }
        // The per-layer block table maps the compacted history (selected_pages)
        // plus the one suffix page; `n_slots` may carry extra arena headroom, so
        // derive the tight table bound from the selected extent rather than the
        // arena size (the kernel masks any `-1` padding either way — D4a).
        let selected_pages =
            (bootstrap.native_history_tokens / LAYERWISE_BLOCK_TOKENS as u64) as usize;
        let max_pages = selected_pages + 1;
        let staging = LayerwiseDescStaging::new(gpu, n_layers, max_pages)?;
        let suffix_phys = driver
            .arena_mut()
            .pool()
            .alloc_slot()
            .map_err(|e| format!("layerwise build: suffix slot: {e}"))?
            .phys;
        let mut carrier = Self::new(driver, staging);
        carrier.suffix_slot = Some(suffix_phys);
        carrier.suffix_page = bootstrap.suffix_page;
        carrier.partial_tokens = bootstrap.partial_tokens;
        carrier.pressure = pressure;
        carrier.mass = mass;
        Ok(carrier)
    }

    /// Release the composite arena + per-layer staging buffers, consuming self.
    /// The borrowed slot stores are unaffected (the `KvMemCtl` still owns them).
    pub fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        self.staging.free_gpu(gpu);
        self.driver.free(gpu)
    }
}

/// Engine-side layerwise query bootstrap (design §3.9g): the canonical history,
/// identity, per-layer specs and suffix geometry synthesized from the session +
/// host-store length. Pure host data (no GPU); the scoped driver re-exports the
/// canonical pages' packed K/V from the borrowed `RawStore` on demand, so NO
/// snapshot persistence is needed — the host store is the source of truth.
pub struct LayerwiseBootstrap {
    pub canonical: Vec<LayerMemorySnapshot>,
    pub identity: LayerwiseIdentity,
    pub specs: Vec<LayerRequestSpec>,
    /// Logical page holding the current token (`fed / 128`).
    pub suffix_page: u32,
    /// Tokens already fed into the suffix page (`fed % 128`).
    pub partial_tokens: u32,
    /// Selected working-set extent the core picker reproduces.
    pub native_history_tokens: u64,
}

/// Synthesize a fresh layerwise query bootstrap for one decode step (design
/// §3.9g). `fed` is the host-store length BEFORE this step's token (the current
/// token position `p`). `full_pages = fed / 128` sealed pages form the canonical
/// history; the current partial page is the suffix. Fails closed when there is
/// no sealed history page (`fed < 128`): the caller falls back to Legacy. `seq` /
/// `incarnation` come from the slot's layerwise counters; `record_bytes` is the
/// sealed canonical record size (all fa layers × one page K+V) for the host-payload
/// budget.
///
/// `resident` is the slot's persistent PER-LAYER keep-set (decision A, §3.10a):
/// `resident[ordinal]` holds the canonical block ids selected for FA ordinal
/// `ordinal` by the previous step (pressure.select on a Pressure step, the ②
/// scorer on a Query step). It drives each layer's `MemorySnapshot.resident`
/// (replacing the pure re-bootstrap `[]`); ids beyond the current sealed history
/// are filtered defensively, and an absent ordinal entry reads as `[]`. An empty
/// slice reproduces the ② re-bootstrap behaviour (every selected page freshly
/// restored).
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_layerwise(
    fed: u32,
    session_id: u64,
    incarnation: u64,
    seq: u64,
    fa_layers: &[u32],
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    budget_blocks: u32,
    record_bytes: u64,
    resident: &[Vec<u64>],
) -> Result<LayerwiseBootstrap, String> {
    let bt = LAYERWISE_BLOCK_TOKENS;
    if fa_layers.is_empty() {
        return Err("layerwise bootstrap: empty fa_layers".to_string());
    }
    let full_pages = fed / bt;
    if full_pages == 0 {
        return Err(format!(
            "layerwise bootstrap: no sealed history page (fed {fed} < {bt}); use Legacy"
        ));
    }
    let partial_tokens = fed % bt;
    let suffix_page = full_pages;
    // The kvmem core requires a NON-ZERO logical history identity
    // (memory_contract.cpp:157, layerwise_selection.cpp:85): session.id,
    // session.incarnation, history_revision, execution_history and the request
    // id must all be != 0. The engine's identities are 0-based (SessionTable
    // mints SessionId(0) first; a fresh slot's incarnation/seq start at 0), so
    // map them to the core's 1-based logical identities here at the synthesis
    // boundary. The +1 is bijective (distinct engine sessions / reset counts /
    // transaction counts stay distinct) and monotonic; the engine counters
    // themselves are untouched (Legacy byte-neutral).
    let session = SessionIdentity { id: session_id + 1, incarnation: incarnation + 1 };
    let revision = seq + 1;
    let selected_pages = full_pages.min(budget_blocks.max(1));
    let selected_tokens = selected_pages as u64 * bt as u64;
    let frontier = full_pages as u64 * bt as u64;
    let base_stamp = MemoryStamp {
        valid: 1,
        session,
        history_revision: revision,
        view_version: revision,
        frontier,
        execution_history: revision,
    };
    // Sealed full pages only (id == dense ordinal == original_start / 128, the
    // scorer frame's page index). host_version == content_version (write-through
    // sealed). The partial page is NOT canonical — it is the suffix (§3.9g).
    let blocks: Vec<BlockCopies> = (0..full_pages)
        .map(|page| BlockCopies {
            block: BlockDescriptor {
                session,
                id: page as u64,
                original_start: page as u64 * bt as u64,
                valid_tokens: bt,
                content_version: 1,
                statistics_version: 1,
            },
            host_version: 1,
        })
        .collect();
    let mut canonical = Vec::with_capacity(fa_layers.len());
    let mut specs = Vec::with_capacity(fa_layers.len());
    for (ordinal, &model_layer) in fa_layers.iter().enumerate() {
        // Decision A (§3.10a): the persistent PER-LAYER keep-set drives each
        // layer's canonical resident list. Keep only ids within the current
        // sealed history (the keep-set is a subset of a previous step's history,
        // which only grows within a session, so this filter is defensive). An
        // absent ordinal entry reads as `[]` (cold start / re-bootstrap).
        let resident_ids: Vec<u64> = resident
            .get(ordinal)
            .map(|ids| ids.iter().copied().filter(|&id| id < full_pages as u64).collect())
            .unwrap_or_default();
        let mut stamp = base_stamp;
        // Per-layer view_version diverges (the catalog's shared-history
        // admission allows this, mirroring the core fixture).
        stamp.view_version = revision + ordinal as u64;
        canonical.push(LayerMemorySnapshot {
            model_layer,
            memory: MemorySnapshot { stamp, blocks: blocks.clone(), resident: resident_ids.clone() },
        });
        specs.push(LayerRequestSpec {
            model_layer,
            query: LayerQueryGeometry {
                layer: ordinal as u32,
                query_heads: n_heads,
                kv_heads: n_kv_heads,
                head_dim,
                attention_scale: 0.0,
            },
            budget: MemoryBudget {
                selected_tokens,
                reserve_tokens: bt as u64,
                device_tokens: (selected_pages as u64 + 1) * bt as u64,
                host_payload_bytes: full_pages as u64 * record_bytes,
            },
            mandatory: Vec::new(),
        });
    }
    // The core ties the evaluation span to the committed logical frontier:
    // `evaluation.begin == history.frontier == policy.total_tokens()`
    // (layerwise_selection.cpp:88,90), and the layerwise selection index holds
    // ONLY sealed full pages (`prepare_selection` requires total_tokens % block
    // == 0 and every block full). So the frontier is the full-page boundary
    // `full_pages * 128` — NOT `fed` — and the single decode row is evaluated at
    // logical position `frontier`. The partial tail (`fed % 128` already-fed
    // tokens) is NOT part of the logical canonical frame: it lives only in the
    // statistics store and is restored as the physical suffix stripe (§3.9g),
    // extending the compacted causal bound by `partial_tokens + rows`.
    let identity = LayerwiseIdentity {
        history: base_stamp,
        request: session_id + 1,
        evaluation: QueryRange { begin: frontier, end: frontier + 1 },
        query: QueryRange { begin: frontier, end: frontier + 1 },
    };
    Ok(LayerwiseBootstrap {
        canonical,
        identity,
        specs,
        suffix_page,
        partial_tokens,
        native_history_tokens: selected_tokens,
    })
}

/// Production backend identity for the hipfire Q8 packed layerwise archive
/// (design §3.9g). Tags the sealed records' provenance; the core treats it as an
/// opaque admission key (must match across a transaction's export/restore).
pub fn layerwise_backend(model: &str) -> BackendIdentity {
    BackendIdentity {
        backend: Q8_PACKED_FORMAT.to_string(),
        revision: env!("CARGO_PKG_VERSION").to_string(),
        model: model.to_string(),
        state_schema: "state-v1".to_string(),
    }
}

/// Production memory capabilities for the hipfire Q8 layerwise path (design
/// §3.9g): native packed payload, sparse compacted views, original RoPE
/// positions, pre-RoPE statistics, state checkpoint; synchronous transfer (the
/// driver fences its own steady-copy stream).
pub fn layerwise_capabilities(max_position: u32) -> MemoryCapabilities {
    MemoryCapabilities {
        page_tokens: LAYERWISE_BLOCK_TOKENS,
        max_position: u64::from(max_position),
        native_payload: 1,
        sparse_view: 1,
        original_rope_positions: 1,
        pre_rope_statistics: 1,
        state_checkpoint: 1,
        asynchronous_transfer: 0,
    }
}

/// Engine-side per-step carrier builder (design §3.9g orchestration). Given the
/// taken-out controller and the admitted slot's identity + model geometry,
/// synthesize the bootstrap and build the scoped carrier borrowing the slot's
/// statistics store and the caller-supplied selection policy. Returns `None`
/// (Legacy fallback, byte-neutral) when the strategy is not Layerwise, the slot
/// is out of range, or no sealed history page exists yet (`fed < 128` — the
/// prompt-completing prefill still runs Legacy). The caller owns the returned
/// carrier and MUST [`LayerwiseQuery::free`] it exactly once after the forward;
/// the `ctl` and `policy` borrows must outlive it (scoped take/restore).
///
/// `policy` MUST be a fresh, page-aligned selection index over ONLY the sealed
/// full pages (`KvMemCtl::layerwise_policy`): the core `prepare_selection`
/// rejects a partial tail (`total_tokens % block == 0`, every block full). The
/// slot's shared `kv` store tracks ALL fed tokens for the Legacy path and is
/// deliberately NOT borrowed here, so Layerwise never mutates it (byte-neutral).
///
/// `fa_layers` are the DENSE fa ordinals `0..n_fa_layers`: the `KvMemCtl` store
/// indexes statistics/arena layers by ordinal (`raw_cfg.n_layer == n_fa_layers`,
/// fed via `write_layer_*_gpu(.., il, ..)`), so the layerwise `model_layer` is the
/// ordinal — not the sparse model layer id. `max_position` is the engine KV
/// capacity (an upper bound on any decode position); `model` tags the backend
/// identity (opaque admission key, consistent across the transaction).
///
/// `pressure` is the slot's detached persistent pressure owner (design §3.10a,
/// decision A), borrowed for the trigger / `select` / `rows` schedule (all
/// `&self`). `None` ⇒ the certified ② scorer Query path with no mass capture
/// (the Legacy spine and the ② fixtures). `Some` ⇒ the engine decides Query
/// (scorer cold-start / new-page admission) vs Pressure (resident-only mass
/// refinement) per the Q2/Q3 trigger, and attaches the mode-independent mass
/// capture. The orchestration owns the owner for the post-forward `accept`
/// (scoped take/restore — the carrier borrows `ctl`, so the owner is detached).
#[allow(clippy::too_many_arguments)]
pub fn engine_carrier<'a>(
    gpu: &mut Gpu,
    ctl: &'a KvMemCtl,
    policy: &'a KvStore,
    slot: usize,
    session_id: u64,
    n_fa_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_position: u32,
    model: &str,
    pressure: Option<&PrefillPressure>,
) -> Result<Option<LayerwiseQuery<'a>>, String> {
    if !ctl.strategy().is_layerwise() {
        return Ok(None);
    }
    let (fed, incarnation, seq, stores) = match (
        ctl.fed_tokens(slot),
        ctl.layerwise_counters(slot),
        ctl.layerwise_stores(slot),
    ) {
        (Some(fed), Some((incarnation, seq)), Some(stores)) => (fed, incarnation, seq, stores),
        _ => return Ok(None),
    };
    // No sealed full page yet ⇒ nothing to recall; the step stays Legacy.
    if fed < LAYERWISE_BLOCK_TOKENS {
        return Ok(None);
    }
    // Borrow only the statistics store from the slot; the selection policy is
    // the caller-supplied fresh page-aligned index (the slot's shared `kv`
    // store tracks the partial tail for Legacy and must not be used here).
    let (keys, _slot_policy) = stores;
    let fa_layers: Vec<u32> = (0..n_fa_layers as u32).collect();
    let q8 = LayerwiseQ8Layout::resolve(n_fa_layers, n_kv_heads, head_dim, &fa_layers)
        .map_err(|e| format!("layerwise engine: q8 layout: {e}"))?;
    let record_bytes = q8
        .payload_layout()
        .map_err(|e| format!("layerwise engine: manifest: {e}"))?
        .record_bytes();
    let bootstrap = bootstrap_layerwise(
        fed,
        session_id,
        incarnation,
        seq,
        &fa_layers,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        ctl.budget_blocks() as u32,
        record_bytes,
        // Decision A (§3.10a): the slot's persistent PER-LAYER keep-set drives
        // each layer's canonical resident list (empty before the first selection
        // ⇒ ② re-bootstrap).
        ctl.layerwise_keep_set(slot).unwrap_or(&[]),
    )?;
    let selected_pages =
        (bootstrap.native_history_tokens / LAYERWISE_BLOCK_TOKENS as u64) as usize;
    // Arena: one composite slot per compact history position (selected_pages,
    // SHARED across all FA layers — design §3.1 mixed-stripe destinations) plus
    // the suffix page, plus one slot of headroom. The position-shared model needs
    // `selected_pages + 1` regardless of the layer count; the extra slot keeps a
    // scratch bind from exhausting the pool.
    let n_slots = selected_pages + 2;
    // Archive (host RAM restore source): `seal_record_source` caches ONE sealed
    // canonical record per unique page, keyed by page id, lazily re-exported on
    // first restore. Different FA layers select DIFFERENT pages into the shared
    // compact positions, so the union of pages sealed across all ordinals can
    // reach the FULL sealed history (full_pages), NOT the per-layer selected
    // budget — sizing to `selected_pages` exhausts the archive mid-restore once
    // the union exceeds the budget (real 16-layer serve: "alloc block 45: native
    // KV RAM capacity exhausted"). Size to the full canonical history + suffix
    // (+1 headroom); the GPU arena above stays compact regardless.
    let full_pages = (fed / LAYERWISE_BLOCK_TOKENS) as u64;
    let archive_bytes = (full_pages + 2) * record_bytes;
    // ── Pressure trigger (design §3.10a, Q2/Q3) ──────────────────────────────
    // A new canonical page is sealed exactly when `fed` lands on a page boundary
    // (`fed % 128 == 0`): the scorer Query admits it (pressure has no cold recall).
    // Mid-page steps refine the resident keep-set by accumulated attention mass
    // ONLY when the history overflows the budget AND every FA layer has a positive
    // accepted pressure state. `pressure.select` Pending (any layer without a
    // positive-mass selection) falls back to Query. The owner is borrowed (`&self`
    // select/rows/snapshot); the orchestration owns it for the post-forward accept.
    let budget_blocks = ctl.budget_blocks() as u32;
    let new_page = fed % LAYERWISE_BLOCK_TOKENS == 0;
    let keep_empty = ctl
        .layerwise_keep_set(slot)
        .map(|k| k.iter().all(|ids| ids.is_empty()))
        .unwrap_or(true);
    let pressure_positive = match pressure {
        Some(p) => p
            .snapshot()
            .ok()
            .and_then(|s| s.read().ok())
            .map(|snap| !snap.layers.is_empty() && snap.layers.iter().all(|l| l.samples > 0))
            .unwrap_or(false),
        None => false,
    };
    let pressure_step: Option<PressureStep> = match pressure {
        Some(p) if !new_page && !keep_empty && pressure_positive && full_pages > budget_blocks as u64 => {
            let layout = q8
                .payload_layout()
                .map_err(|e| format!("layerwise engine: manifest: {e}"))?;
            let descriptor = MemoryDescriptor::new(
                &layerwise_backend(model),
                &layout,
                &layerwise_capabilities(max_position),
            )
            .map_err(|e| format!("layerwise engine: descriptor: {e}"))?;
            let catalog = MemoryCatalog::new(&descriptor, &bootstrap.canonical)
                .map_err(|e| format!("layerwise engine: catalog: {e}"))?;
            let budget_tokens = budget_blocks as u64 * LAYERWISE_BLOCK_TOKENS as u64;
            let choices = p
                .select(&catalog, &[], budget_tokens, 0)
                .map_err(|e| format!("layerwise engine: pressure select: {e}"))?;
            let all = choices
                .read_all()
                .map_err(|e| format!("layerwise engine: pressure choices: {e}"))?;
            if all.len() != n_fa_layers || all.iter().any(|c| c.selected.is_empty()) {
                None // Pending ⇒ scorer Query fallback
            } else {
                let mut selected = Vec::with_capacity(n_fa_layers);
                for ordinal in 0..n_fa_layers {
                    let choice = all
                        .iter()
                        .find(|c| c.model_layer == fa_layers[ordinal])
                        .ok_or_else(|| {
                            format!("layerwise engine: pressure choice missing layer {ordinal}")
                        })?;
                    selected.push(choice.selected.clone());
                }
                let budgets: Vec<MemoryBudget> = bootstrap.specs.iter().map(|s| s.budget).collect();
                Some(PressureStep::new(selected, budgets)?)
            }
        }
        _ => None,
    };
    // Mode-independent mass-capture state (design §3.10a): present whenever a
    // pressure owner backs the step. The sample schedule is gated by the core
    // `rows()` (frontier advance + interval/near); empty ⇒ no capture/accept this
    // step. `local_begin` is the compact causal tail of the single decode row
    // (selected_pages*128 + partial_tokens), identical in both modes because the
    // picker reproduces `native_history_tokens` (Σ selected page tokens).
    let mass = match pressure {
        Some(p) => {
            let frontier = bootstrap.identity.history.frontier;
            let local_begin = bootstrap.native_history_tokens + bootstrap.partial_tokens as u64;
            let mass_rows = p
                .rows(frontier - 1, 1, local_begin, false)
                .map_err(|e| format!("layerwise engine: pressure rows: {e}"))?;
            Some(MassCapture::new(n_fa_layers, frontier, local_begin, 1, mass_rows))
        }
        None => None,
    };
    let mut carrier = LayerwiseQuery::build_with_pressure(
        gpu,
        keys,
        policy,
        &q8,
        layerwise_backend(model),
        layerwise_capabilities(max_position),
        bootstrap,
        n_slots,
        archive_bytes,
        pressure_step,
        mass,
    )?;
    // ③-E1: attach the activation-quota subsystem (five live accounts + per-FA-layer
    // projected-Q tapes) ONLY in Query mode — it admits under the query identity.
    // Pressure mode runs no query state machine and reads the live post-RoPE Q for
    // mass instead. On a quota admission failure the just-built carrier owns GPU
    // buffers with no Drop, so free it before propagating the error.
    if !carrier.is_pressure() {
        if let Err(e) = carrier.attach_activation(gpu, ctl.layerwise_quotas(), n_heads * head_dim * 4) {
            let _ = carrier.free(gpu);
            return Err(e);
        }
    }
    Ok(Some(carrier))
}

/// Combine the previous per-layer keep-set with this step's restored canonical
/// page ids (design §3.10a keep-set lifecycle, ninfer window-faithful). A Query
/// step UNIONS (`prev ∪ restored`, sorted/deduped) so the resident pool grows to
/// admit scorer selections and newly sealed pages; a Pressure step TRUNCATES to
/// the refined `restored` selection (the mass/exposure reorder prunes the pool
/// back to budget). `n_layers` is the FA ordinal count; an absent ordinal entry
/// reads as `[]`. Pure host logic — the arena is still freshly restored every
/// step (device-truth); the keep-set persists only the SELECTION state.
pub fn combine_keep_set(
    prev: &[Vec<u64>],
    restored: &[Vec<u64>],
    pressure: bool,
    n_layers: usize,
) -> Vec<Vec<u64>> {
    (0..n_layers)
        .map(|ordinal| {
            let mut ids = if pressure {
                restored.get(ordinal).cloned().unwrap_or_default()
            } else {
                let mut u = prev.get(ordinal).cloned().unwrap_or_default();
                u.extend(restored.get(ordinal).into_iter().flatten().copied());
                u
            };
            ids.sort_unstable();
            ids.dedup();
            ids
        })
        .collect()
}

/// Lazily seal one canonical block's packed record from the statistics store
/// (`keys`) into the archive `storage`, caching the immutable [`HostRecord`], and
/// return a deferred read source. Idempotent: a cached block is not re-exported.
/// This is the write-through archive's sealed restore source (design §3.5/§3.8);
/// the core `export_raw_kv_payload` bridge gathers the RawStore's packed K/V into
/// the canonical record, so the statistics store and the sealed archive stay
/// distinct (mirrors the core `layerwise_request_test.cpp` Fixture).
fn seal_record_source(
    records: &mut HashMap<u64, HostRecord>,
    storage: &Storage,
    keys: &RawStore,
    backend: &BackendIdentity,
    layout: &PayloadLayout,
    block: &BlockDescriptor,
    block_tokens: u32,
) -> Result<HostSource, String> {
    if !records.contains_key(&block.id) {
        // The core raw-KV bridge addresses the statistics store by DENSE ORDINAL
        // (original_start / block_tokens) and requires block.id == that ordinal,
        // whereas the catalog carries a driver-assigned native id (e.g. 200).
        // Re-tag the export descriptor to the canonical ordinal; the cache stays
        // keyed by the native id the plan/selection use.
        let mut export_block = *block;
        export_block.id = block.original_start / block_tokens as u64;
        let payload = NativePayload::export(keys, backend, layout, &export_block)
            .map_err(|e| format!("layerwise seal: export block {}: {e}", block.id))?;
        let rec = HostRecord::allocate(storage, payload.bytes())
            .map_err(|e| format!("layerwise seal: alloc block {}: {e}", block.id))?;
        rec.seal()
            .map_err(|e| format!("layerwise seal: seal block {}: {e}", block.id))?;
        records.insert(block.id, rec);
    }
    records
        .get(&block.id)
        .expect("cached record")
        .source()
        .map_err(|e| format!("layerwise seal: source block {}: {e}", block.id))
}

/// Execute one layer's reduced plan against the composite arena: drop evicted
/// stripes, bind the compact next-view's pages to physical slots (reusing
/// resident ones), restore every cold page from the sealed archive
/// (KvRegions::read → pinned slab → H2D stripe), fence ONCE, and return the
/// post-fence [`ExecutionView`] rebuilt EXACTLY from the plan's next view.
///
/// Device-truth reconciliation: the plan's restore set is derived from the
/// canonical snapshot's resident list (strategy truth), but a freshly bound
/// stripe has no valid bytes regardless of that claim — the per-query scoped
/// driver starts from an empty arena while the snapshot describes the last
/// accepted working set. Every freshly allocated stripe is therefore restored
/// even when the plan marks its page hot; only an EXISTING binding keeps the
/// plan's decision.
///
/// Direct drive (design §3.5): no `MemoryBackend`/`MemoryTransfer`. `ordinal` is
/// both the statistics layer and the arena layer index. D3 does the whole-plane
/// H2D; D4 wraps the write in the COW stripe guards. The canonical page tag is
/// the dense ordinal `original_start / block_tokens` (the scorer frame's index).
#[allow(clippy::too_many_arguments)]
fn restore_layer_pages(
    gpu: &mut Gpu,
    arena: &mut LayerwiseArena,
    storage: &Storage,
    keys: &RawStore,
    backend: &BackendIdentity,
    layout: &PayloadLayout,
    records: &mut HashMap<u64, HostRecord>,
    slab: &mut PinnedTransferBuffer,
    plan: &LayerPlan,
    ordinal: usize,
    block_tokens: u32,
) -> Result<ExecutionView, String> {
    let regions = plan
        .payload()
        .map_err(|e| format!("layerwise restore: payload regions: {e}"))?;
    let k_plane = arena.layout().k_plane_bytes();
    let v_plane = arena.layout().v_plane_bytes();
    let extent = regions.packed_bytes() as usize;
    if extent != k_plane + v_plane {
        return Err(format!(
            "layerwise restore: layer extent {extent} != K{k_plane}+V{v_plane} planes"
        ));
    }
    if slab.size() < extent {
        return Err(format!(
            "layerwise restore: pinned slab {} < layer extent {extent}",
            slab.size()
        ));
    }
    let page_of = |b: &BlockDescriptor| (b.original_start / block_tokens as u64) as u32;

    // 1. Evict: drop the plan's evicted pages' stripes for this layer; free a
    //    slot once EVERY layer stripe has dropped (bare drop-evict invariant).
    for i in 0..plan.evict_count() {
        let b = plan
            .evict_block(i)
            .map_err(|e| format!("layerwise restore: evict {i}: {e}"))?;
        let page = page_of(&b);
        if let Some(phys) = arena.pool_ref().find_page_slot(ordinal, page) {
            let h = arena.pool_ref().handle(phys).expect("reserved slot handle");
            let all_dropped = arena
                .pool()
                .drop_stripe(h, ordinal)
                .map_err(|e| format!("layerwise restore: drop slot {phys}: {e}"))?;
            if all_dropped {
                arena
                    .pool()
                    .free_slot(h)
                    .map_err(|e| format!("layerwise restore: free slot {phys}: {e}"))?;
            }
        }
    }

    // Restore set (block IDs the plan marks cold for this layer).
    let mut restore_ids = std::collections::HashSet::new();
    for r in 0..plan.restore_count() {
        let b = plan
            .restore_block(r)
            .map_err(|e| format!("layerwise restore: restore {r}: {e}"))?;
        restore_ids.insert(b.id);
    }

    // 2. Bind the compact next view by POSITION: compact position i maps to the
    //    SAME physical composite slot across every ordinal (design §3.1
    //    mixed-stripe destinations); each layer binds its own stripe to the page
    //    it selected for that position. The slot is allocated once (first ordinal
    //    to reach i) and reused thereafter — one slot per position, NOT per
    //    (ordinal, page), so the arena stays at selected_pages + suffix slots
    //    regardless of the FA-layer count.
    let next_count = plan.next_block_count();
    let mut compact: Vec<u32> = Vec::with_capacity(next_count);
    let mut restore_targets: Vec<(u32, BlockDescriptor)> = Vec::new();
    for i in 0..next_count {
        let vb = plan
            .next_block(i)
            .map_err(|e| format!("layerwise restore: next {i}: {e}"))?;
        let page = page_of(&vb.block);
        let (phys, fresh) = arena.bind_position_stripe(i, ordinal, page)?;
        compact.push(phys);
        // The plan's cold set is STRATEGY truth (derived from the canonical
        // snapshot's resident list); the pool is DEVICE truth. A freshly bound
        // stripe holds no valid bytes no matter what the snapshot claims is
        // resident — the per-query scoped driver starts from an empty arena
        // while the snapshot describes the last accepted working set — so it
        // must ALWAYS be restored. An existing binding keeps the plan's
        // decision (a bound stripe was restored or suffix-written already).
        if fresh || restore_ids.contains(&vb.block.id) {
            restore_targets.push((phys, vb.block));
        }
    }

    // 3. Restore cold pages: read each page's packed extent (K plane then V
    //    plane) into its OWN disjoint slab slice and H2D-scatter both planes
    //    into the slot stripe, stream-ordered. The slices never overlap, so the
    //    next page's host read cannot clobber a previous page's in-flight DMA
    //    (a single reusable slab raced the copy engine once the records were
    //    cached and the loop ran tight); ONE fence (step 5) drains every copy
    //    before the view publishes.
    let needed = restore_targets.len().checked_mul(extent).ok_or_else(|| {
        format!(
            "layerwise restore: staged extent {} * {extent} overflow",
            restore_targets.len()
        )
    })?;
    if slab.size() < needed {
        return Err(format!(
            "layerwise restore: pinned slab {} < {needed} for {} staged cold pages",
            slab.size(),
            restore_targets.len()
        ));
    }
    let stream = gpu.steady_copy_stream();
    for (i, (phys, block)) in restore_targets.iter().enumerate() {
        let source = seal_record_source(records, storage, keys, backend, layout, block, block_tokens)?;
        let k_off = arena
            .layout()
            .layer_k_offset(*phys as usize, ordinal)
            .map_err(|e| format!("layerwise restore: k offset: {e}"))?;
        let v_off = arena
            .layout()
            .layer_v_offset(*phys as usize, ordinal)
            .map_err(|e| format!("layerwise restore: v offset: {e}"))?;
        let base = i * extent;
        let host = slab.as_bytes_mut();
        regions
            .read(&source, 0, &mut host[base..base + extent], extent)
            .map_err(|e| format!("layerwise restore: read block {}: {e}", block.id))?;
        gpu.hip
            .memcpy_htod_async_at_on(&arena.arena().buf, k_off, &host[base..base + k_plane], stream)
            .map_err(|e| format!("layerwise restore: H2D K block {}: {e:?}", block.id))?;
        gpu.hip
            .memcpy_htod_async_at_on(
                &arena.arena().buf,
                v_off,
                &host[base + k_plane..base + extent],
                stream,
            )
            .map_err(|e| format!("layerwise restore: H2D V block {}: {e:?}", block.id))?;
    }

    // 4. Publish the compact block table for this layer.
    let visible = plan.next_visible_tokens();
    arena
        .pool()
        .set_view(ordinal, &compact, visible as usize)
        .map_err(|e| format!("layerwise restore: set_view layer {ordinal}: {e}"))?;

    // 5. ONE fence: every restored stripe is resident before the view publishes.
    match stream {
        Some(s) => gpu.hip.stream_synchronize(s),
        None => gpu.hip.stream_synchronize_default(),
    }
    .map_err(|e| format!("layerwise restore: fence: {e:?}"))?;

    // 6. Rebuild the ExecutionView EXACTLY from the plan's next view (restored()
    //    requires same_view equality: stamp, visible_tokens, each block +
    //    compact_start verbatim).
    let stamp = plan
        .next_stamp()
        .map_err(|e| format!("layerwise restore: next stamp: {e}"))?;
    let mut blocks = Vec::with_capacity(next_count);
    for i in 0..next_count {
        blocks.push(
            plan.next_block(i)
                .map_err(|e| format!("layerwise restore: next {i}: {e}"))?,
        );
    }
    Ok(ExecutionView { stamp, blocks, visible_tokens: visible })
}

/// One admitted layerwise Query transaction: the core request state machine plus
/// the per-ordinal reduced plans and post-fence views the driver produced. The
/// request OWNS a catalog copy but borrows the driver's `keys`/`policy`, so it
/// must be dropped (via [`LayerwiseDriver::end_query`]) before those borrows end.
struct ActiveQuery {
    request: LayerwiseRequest,
    identity: LayerwiseIdentity,
    plans: Vec<Option<LayerPlan>>,
    views: Vec<Option<ExecutionView>>,
}

/// Per-layer layerwise recall transaction owner (design §3.5/§3.9, ②-D3).
///
/// Clean sibling to the legacy copy-only path: it wires the core
/// [`LayerwiseRequest`] state machine to hipfire's OWN device resources — the
/// composite packed [`LayerwiseArena`], the sealed write-through archive
/// ([`Storage`] + cached [`HostRecord`]s), the reusable pinned slab, and the
/// GPU scorer bridge ([`score_current_layer_native`], Rust→HIP, no shim). It
/// drives `begin_layer → projected_native → prepare_restore → restore →
/// restored → attended → complete_layer` per ordinal, then publishes the
/// accepted per-layer snapshots and advances the canonical history.
///
/// Direct drive: NO `MemoryBackend`/`MemoryTransfer` (the layerwise path never
/// consumes them). Single serve thread; the FFI handles are `!Send`/`!Sync`.
/// Gated on `RecallStrategy::Layerwise`, so the default Legacy serve never
/// constructs it and stays byte-neutral. D4 adds the COW stripe-write guards and
/// the three paged forward-path integrations on top of this orchestration.
pub struct LayerwiseDriver<'a> {
    keys: &'a RawStore,
    policy: &'a KvStore,
    arena: LayerwiseArena,
    archive: Storage,
    backend: BackendIdentity,
    layout: PayloadLayout,
    descriptor: MemoryDescriptor,
    /// ordinal → model layer (the statistics/arena layer index is the ordinal).
    fa_layers: Vec<u32>,
    block_tokens: u32,
    records: HashMap<u64, HostRecord>,
    slab: PinnedTransferBuffer,
    canonical: Vec<LayerMemorySnapshot>,
    stamp: MemoryStamp,
    active: Option<ActiveQuery>,
}

impl<'a> LayerwiseDriver<'a> {
    /// Wire the subsystem over the engine's statistics store (`keys`) and policy
    /// (`policy`), allocating the composite arena (`n_slots`), the sealed archive
    /// (`archive_bytes`) and a pinned staging slab sized to `n_slots` layer
    /// {K,V} extents — one disjoint slice per in-flight cold page, so the
    /// restore loop's stream-ordered H2D copies never share a host buffer (a
    /// single reusable slab would be overwritten by the next page's read before
    /// the previous async DMA drains). `canonical` is the frozen per-layer
    /// history (ordinal order); the frozen transaction history stamp is taken
    /// from its first layer. Fails closed — never partially constructs.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &mut Gpu,
        keys: &'a RawStore,
        policy: &'a KvStore,
        q8: &LayerwiseQ8Layout,
        backend: BackendIdentity,
        capabilities: MemoryCapabilities,
        canonical: Vec<LayerMemorySnapshot>,
        n_slots: usize,
        archive_bytes: u64,
    ) -> Result<Self, String> {
        if canonical.is_empty() {
            return Err("layerwise driver: empty canonical history".to_string());
        }
        let layout = q8
            .payload_layout()
            .map_err(|e| format!("layerwise driver: manifest: {e}"))?;
        let descriptor = MemoryDescriptor::new(&backend, &layout, &capabilities)
            .map_err(|e| format!("layerwise driver: descriptor: {e}"))?;
        let block_tokens = q8.raw_config().block_tokens;
        let fa_layers: Vec<u32> = canonical.iter().map(|s| s.model_layer).collect();
        let stamp = canonical[0].memory.stamp;
        let arena = LayerwiseArena::new(gpu, q8, n_slots)?;
        let extent =
            (arena.layout().k_plane_bytes() + arena.layout().v_plane_bytes()) as usize;
        // One disjoint staging slice per composite slot: the restore loop binds
        // at most `n_slots` cold pages and issues their H2D copies stream-ordered
        // before a single fence, so each in-flight copy needs its own host bytes.
        let slab_bytes = extent
            .checked_mul(n_slots)
            .ok_or_else(|| format!("layerwise driver: slab extent {extent} * {n_slots} overflow"))?;
        let slab = gpu
            .hip
            .transfer_buffer(slab_bytes)
            .map_err(|e| format!("layerwise driver: pinned slab {slab_bytes}B: {e:?}"))?;
        let archive = Storage::new_ram(archive_bytes)
            .map_err(|e| format!("layerwise driver: archive {archive_bytes}B: {e}"))?;
        Ok(Self {
            keys,
            policy,
            arena,
            archive,
            backend,
            layout,
            descriptor,
            fa_layers,
            block_tokens,
            records: HashMap::new(),
            slab,
            canonical,
            stamp,
            active: None,
        })
    }

    /// Admit a frozen layerwise Query transaction. `specs` has one entry per
    /// canonical layer (matching model layers); `native_history_tokens` is the
    /// selected working-set extent the core picker must reproduce (Σ selected
    /// page tokens). Builds the catalog from the frozen canonical history, sets
    /// the policy's runtime selected budget, and creates the request with a
    /// revision closure returning the frozen history (single-threaded: the
    /// source cannot mutate mid-transaction). Rejects if a query is already
    /// active — call [`Self::end_query`] first.
    pub fn begin_query(
        &mut self,
        identity: LayerwiseIdentity,
        specs: Vec<LayerRequestSpec>,
        native_history_tokens: u64,
    ) -> Result<(), String> {
        if self.active.is_some() {
            return Err("layerwise driver: a query is already active (end_query first)".to_string());
        }
        if specs.len() != self.canonical.len() {
            return Err(format!(
                "layerwise driver: {} specs != {} canonical layers",
                specs.len(),
                self.canonical.len()
            ));
        }
        for (i, s) in specs.iter().enumerate() {
            if s.model_layer != self.fa_layers[i] {
                return Err(format!(
                    "layerwise driver: spec {i} model_layer {} != canonical {}",
                    s.model_layer, self.fa_layers[i]
                ));
            }
        }
        let catalog = MemoryCatalog::new(&self.descriptor, &self.canonical)
            .map_err(|e| format!("layerwise driver: catalog: {e}"))?;
        let select_budget =
            specs.iter().map(|s| s.budget.selected_tokens).max().unwrap_or(0) as u32;
        self.policy
            .set_runtime_select_budget(select_budget)
            .map_err(|e| format!("layerwise driver: select budget {select_budget}: {e}"))?;
        let frozen = identity.history;
        let request = LayerwiseRequest::new(
            &identity,
            &catalog,
            self.policy,
            self.keys,
            &specs,
            native_history_tokens,
            move || frozen,
        )
        .map_err(|e| format!("layerwise driver: request: {e}"))?;
        let n = self.canonical.len();
        self.active = Some(ActiveQuery {
            request,
            identity,
            plans: (0..n).map(|_| None).collect(),
            views: (0..n).map(|_| None).collect(),
        });
        Ok(())
    }

    /// Run ordinal `ordinal`'s projection + restore: open the layer, score the
    /// borrowed native Q tile on hipfire's HIP launcher (the core slices it to
    /// the query-span intersection), freeze the selection into a [`LayerPlan`],
    /// execute the plan against the arena (evict/bind/restore/fence), and report
    /// the actual post-fence [`ExecutionView`] back to the state machine. Returns
    /// the published view; the plan and view are retained for acceptance.
    pub fn project_restore(
        &mut self,
        gpu: &mut Gpu,
        ordinal: usize,
        q: &NativeCurrentLayerQuery,
    ) -> Result<ExecutionView, String> {
        if ordinal >= self.fa_layers.len() {
            return Err(format!("layerwise driver: ordinal {ordinal} out of range"));
        }
        let model_layer = self.fa_layers[ordinal];
        let il = ordinal as u32;
        let bt = self.block_tokens;
        let keys = self.keys;
        // Split borrow: `active` holds self.active; the restore free function
        // takes the disjoint arena/archive/records/slab/identity fields directly.
        let active = self.active.as_mut().ok_or("layerwise driver: no active query")?;
        let id = active.identity;
        active
            .request
            .begin_layer(&id, model_layer)
            .map_err(|e| format!("layerwise driver: begin_layer {model_layer}: {e}"))?;
        let begin = id.evaluation.begin;
        active
            .request
            .projected_native(&id, begin, q, |query, frame| {
                score_current_layer_native(gpu, keys, il, bt, query, frame)
            })
            .map_err(|e| format!("layerwise driver: projected_native {model_layer}: {e}"))?;
        let plan = active
            .request
            .prepare_restore(&id)
            .map_err(|e| format!("layerwise driver: prepare_restore {model_layer}: {e}"))?;
        let view = restore_layer_pages(
            gpu,
            &mut self.arena,
            &self.archive,
            self.keys,
            &self.backend,
            &self.layout,
            &mut self.records,
            &mut self.slab,
            &plan,
            ordinal,
            self.block_tokens,
        )?;
        active
            .request
            .restored(&id, &view)
            .map_err(|e| format!("layerwise driver: restored {model_layer}: {e}"))?;
        active.plans[ordinal] = Some(plan);
        active.views[ordinal] = Some(view.clone());
        Ok(view)
    }

    /// Build a fresh catalog over the frozen canonical history (design §3.10a).
    /// The per-layer `resident` lists are the persistent keep-set baked in by
    /// [`bootstrap_layerwise`]; the pressure owner's `select` reads them as the
    /// resident-only candidate pool. Pure host data — no GPU, no state machine.
    pub fn pressure_catalog(&self) -> Result<MemoryCatalog, String> {
        MemoryCatalog::new(&self.descriptor, &self.canonical)
            .map_err(|e| format!("layerwise pressure: catalog: {e}"))
    }

    /// Pressure-driven restore (design §3.10a, decision A): restore ordinal
    /// `ordinal`'s pressure-`selected` pages WITHOUT the scorer or the query
    /// state machine. Builds the selection plan directly from
    /// `catalog.plan_layer` (the core `plan_working_set` accepts any
    /// catalog-subset selection; `expected` must equal the layer's frozen
    /// stamp) and executes it through the SAME certified [`restore_layer_pages`]
    /// the ② scorer path uses (evict/bind/restore/fence). Returns the post-fence
    /// [`ExecutionView`]. The caller (the carrier's Pressure mode) owns the
    /// transaction lifecycle — this method never touches `self.active`, so it
    /// runs with no admitted query. `budget` is the layer's bootstrap spec
    /// budget (selected/reserve/device/host envelope).
    pub fn restore_layer_selected(
        &mut self,
        gpu: &mut Gpu,
        ordinal: usize,
        selected: &[u64],
        budget: MemoryBudget,
    ) -> Result<ExecutionView, String> {
        if ordinal >= self.fa_layers.len() {
            return Err(format!("layerwise pressure: ordinal {ordinal} out of range"));
        }
        let model_layer = self.fa_layers[ordinal];
        let expected = self.canonical[ordinal].memory.stamp;
        let catalog = self.pressure_catalog()?;
        let selection = hipfire_kvmem::LayerSelection {
            model_layer,
            expected,
            selected: selected.to_vec(),
            mandatory: Vec::new(),
            budget,
        };
        let plan = catalog
            .plan_layer(&selection)
            .map_err(|e| format!("layerwise pressure: plan_layer {model_layer}: {e}"))?;
        restore_layer_pages(
            gpu,
            &mut self.arena,
            &self.archive,
            self.keys,
            &self.backend,
            &self.layout,
            &mut self.records,
            &mut self.slab,
            &plan,
            ordinal,
            self.block_tokens,
        )
    }

    /// Recompute ordinal `ordinal`'s per-page attention mass for the sampled Q
    /// rows (design §3.10a, E2/E3): D2D-gather the selected history K stripes
    /// (`compact_slots`, compact order) PLUS the suffix stripe into ONE
    /// contiguous page-major Q8_0 buffer — the composite arena interleaves K/V
    /// planes per `(slot, layer)` stripe, but [`Gpu::kvmem_page_mass`] reads a
    /// uniform `PAGE_TOKENS × kv_heads × 272` page stride (`== k_plane_bytes`).
    /// The identity table `[0..pages]` then maps each logical page to its
    /// gathered physical page. `positions[s]` is the inclusive compact causal
    /// end of sample `s` in the gathered frame; `rows[s]` indexes the Q tile.
    /// Returns `samples × pages` f32 masses (sample-major), each row summing to
    /// 1 over the visible cache. The gather buffer is freed before returning;
    /// the arena and the attention kernels are untouched (additive recompute).
    #[allow(clippy::too_many_arguments)]
    pub fn page_mass(
        &mut self,
        gpu: &mut Gpu,
        ordinal: usize,
        q: &PageMassQuery,
        compact_slots: &[u32],
        suffix_slot: Option<u32>,
        rows: &[i32],
        positions: &[i32],
    ) -> Result<Vec<f32>, String> {
        if ordinal >= self.fa_layers.len() {
            return Err(format!("layerwise mass: ordinal {ordinal} out of range"));
        }
        let layout = self.arena.layout();
        let k_plane = layout.k_plane_bytes();
        let pages = compact_slots.len() + usize::from(suffix_slot.is_some());
        if pages == 0 {
            return Err("layerwise mass: empty page domain".to_string());
        }
        let gather_bytes = pages
            .checked_mul(k_plane)
            .ok_or_else(|| format!("layerwise mass: gather {pages} * {k_plane} overflow"))?;
        let gather = gpu
            .alloc_tensor(&[gather_bytes], DType::Raw)
            .map_err(|e| format!("layerwise mass: gather alloc {gather_bytes}B: {e:?}"))?;
        let result = (|| -> Result<Vec<f32>, String> {
            // Gather each selected history page's K plane, then the suffix K
            // plane, into consecutive page-major slots of the scratch buffer.
            for (i, &slot) in compact_slots.iter().enumerate() {
                let src = layout
                    .layer_k_offset(slot as usize, ordinal)
                    .map_err(|e| format!("layerwise mass: k offset slot {slot}: {e}"))?;
                gpu.hip
                    .memcpy_dtod_at(&gather.buf, i * k_plane, &self.arena.arena().buf, src, k_plane)
                    .map_err(|e| format!("layerwise mass: gather K page {i}: {e:?}"))?;
            }
            if let Some(slot) = suffix_slot {
                let src = layout
                    .layer_k_offset(slot as usize, ordinal)
                    .map_err(|e| format!("layerwise mass: suffix k offset slot {slot}: {e}"))?;
                let dst = compact_slots.len() * k_plane;
                gpu.hip
                    .memcpy_dtod_at(&gather.buf, dst, &self.arena.arena().buf, src, k_plane)
                    .map_err(|e| format!("layerwise mass: gather suffix K: {e:?}"))?;
            }
            let table: Vec<i32> = (0..pages as i32).collect();
            let mut out = vec![0f32; rows.len() * pages];
            gpu.kvmem_page_mass(
                q,
                rows,
                positions,
                &table,
                gather.buf.as_ptr() as *const std::ffi::c_void,
                pages,
                &mut out,
            )
            .map_err(|e| format!("layerwise mass: kernel: {e:?}"))?;
            Ok(out)
        })();
        let _ = gpu.free_tensor(gather);
        result
    }

    /// Restore the suffix page's already-fed tokens into `suffix_slot`'s ordinal
    /// stripe (design §3.9g, mid-page decode). The current partial page
    /// (`suffix_page`) holds `partial_tokens` fed tokens in the statistics store;
    /// they are sealed/exported and H2D-scattered into the suffix stripe so the
    /// live current-token write (at offset `partial_tokens`) extends a valid
    /// prefix rather than landing in an empty stripe. Bytes beyond
    /// `partial_tokens` are masked by the compacted causal bound. A page boundary
    /// (`partial_tokens == 0`) is a no-op — the suffix page starts empty. Reuses
    /// the certified `seal_record_source` + `KvRegions` restore path.
    pub fn restore_suffix_page(
        &mut self,
        gpu: &mut Gpu,
        ordinal: usize,
        suffix_slot: u32,
        suffix_page: u32,
        partial_tokens: u32,
    ) -> Result<(), String> {
        if partial_tokens == 0 {
            return Ok(());
        }
        if ordinal >= self.fa_layers.len() {
            return Err(format!("layerwise suffix: ordinal {ordinal} out of range"));
        }
        let bt = self.block_tokens;
        let session = self.canonical[0].memory.stamp.session;
        let block = BlockDescriptor {
            session,
            id: suffix_page as u64,
            original_start: suffix_page as u64 * bt as u64,
            valid_tokens: partial_tokens,
            content_version: 1,
            statistics_version: 1,
        };
        let source = seal_record_source(
            &mut self.records,
            &self.archive,
            self.keys,
            &self.backend,
            &self.layout,
            &block,
            bt,
        )?;
        let regions = KvRegions::new(&self.layout, ordinal as u32, None)
            .map_err(|e| format!("layerwise suffix: regions: {e}"))?;
        let k_plane = self.arena.layout().k_plane_bytes() as usize;
        let v_plane = self.arena.layout().v_plane_bytes() as usize;
        let extent = k_plane + v_plane;
        if self.slab.size() < extent {
            return Err(format!(
                "layerwise suffix: pinned slab {} < layer extent {extent}",
                self.slab.size()
            ));
        }
        {
            let host = self.slab.as_bytes_mut();
            regions
                .read(&source, 0, &mut host[..extent], extent)
                .map_err(|e| format!("layerwise suffix: read page {suffix_page}: {e}"))?;
        }
        let k_off = self
            .arena
            .layout()
            .layer_k_offset(suffix_slot as usize, ordinal)
            .map_err(|e| format!("layerwise suffix: k offset: {e}"))?;
        let v_off = self
            .arena
            .layout()
            .layer_v_offset(suffix_slot as usize, ordinal)
            .map_err(|e| format!("layerwise suffix: v offset: {e}"))?;
        let host = self.slab.as_bytes_mut();
        gpu.hip
            .memcpy_htod_offset(&self.arena.arena().buf, k_off, &host[..k_plane])
            .map_err(|e| format!("layerwise suffix: H2D K page {suffix_page}: {e:?}"))?;
        gpu.hip
            .memcpy_htod_offset(&self.arena.arena().buf, v_off, &host[k_plane..extent])
            .map_err(|e| format!("layerwise suffix: H2D V page {suffix_page}: {e:?}"))?;
        Ok(())
    }

    /// Report `rows` attended native rows starting at token `begin` for ordinal
    /// `ordinal` (after its restore) and close the layer's attending phase. The
    /// engine runs the actual attention between [`Self::project_restore`] and
    /// this call; D3 reports the full evaluation span at once.
    pub fn attend_complete(&mut self, ordinal: usize, begin: u64, rows: u64) -> Result<(), String> {
        let active = self.active.as_mut().ok_or("layerwise driver: no active query")?;
        let id = active.identity;
        active
            .request
            .attended(&id, begin, rows)
            .map_err(|e| format!("layerwise driver: attended {ordinal}: {e}"))?;
        active
            .request
            .complete_layer(&id)
            .map_err(|e| format!("layerwise driver: complete_layer {ordinal}: {e}"))?;
        Ok(())
    }

    /// The active transaction's evaluation span `(begin, rows)` — the native
    /// token rows this query attends (the suffix extent). `None` between queries.
    /// The per-layer restore adds `rows` to the compacted causal bound so the
    /// freshly-written suffix page is visible, and [`Self::attend_complete`]
    /// reports the same span back to the state machine. Single-row decode → 1.
    pub fn evaluation_span(&self) -> Option<(u64, u64)> {
        let active = self.active.as_ref()?;
        let ev = active.identity.evaluation;
        Some((ev.begin, ev.end - ev.begin))
    }

    /// The active transaction's full identity, or `None` between queries. The
    /// activation subsystem (③-E1) admits its projected-Q tapes under this same
    /// identity so the E3 pressure readback proves index provenance.
    pub fn query_identity(&self) -> Option<LayerwiseIdentity> {
        self.active.as_ref().map(|a| a.identity)
    }

    /// Publish the accepted per-layer snapshots and advance the canonical history.
    /// Mirrors the core `layerwise_request_test.cpp` `accepted()` recipe EXACTLY:
    /// each layer's stamp is its restored view's stamp with history_revision,
    /// view_version and execution_history incremented and frontier set to the
    /// evaluation end; the frozen prefix blocks plus the evaluated native
    /// `suffix` (host_version 0) form the new catalog; resident is the restored
    /// next-view ids plus the suffix id. Per-layer view_version diverges (the
    /// catalog's shared-history admission allows this). Leaves the request in
    /// the Accepted phase for [`Self::accepted_snapshots`] readback; call
    /// [`Self::end_query`] to release it.
    pub fn accept(&mut self, suffix: BlockCopies) -> Result<Vec<LayerMemorySnapshot>, String> {
        let active = self.active.as_mut().ok_or("layerwise driver: no active query")?;
        let id = active.identity;
        let end = id.evaluation.end;
        let n = self.canonical.len();
        let mut snapshots = Vec::with_capacity(n);
        for ordinal in 0..n {
            let view = active.views[ordinal]
                .as_ref()
                .ok_or_else(|| format!("layerwise driver: layer {ordinal} not restored"))?;
            let mut stamp = view.stamp;
            stamp.history_revision += 1;
            stamp.view_version += 1;
            stamp.execution_history += 1;
            stamp.frontier = end;
            let mut blocks = self.canonical[ordinal].memory.blocks.clone();
            blocks.push(suffix);
            let mut resident: Vec<u64> = view.blocks.iter().map(|b| b.block.id).collect();
            resident.push(suffix.block.id);
            snapshots.push(LayerMemorySnapshot {
                model_layer: self.fa_layers[ordinal],
                memory: MemorySnapshot { stamp, blocks, resident },
            });
        }
        active
            .request
            .accepted(&id, &snapshots)
            .map_err(|e| format!("layerwise driver: accepted: {e}"))?;
        self.canonical = snapshots.clone();
        self.stamp = snapshots[0].memory.stamp;
        Ok(snapshots)
    }

    /// Read back the core-stored accepted snapshots (Accepted phase only) via the
    /// per-index FFI accessors, reconstructing the owned per-layer snapshots.
    pub fn accepted_snapshots(&self) -> Result<Vec<LayerMemorySnapshot>, String> {
        let active = self.active.as_ref().ok_or("layerwise driver: no active query")?;
        if active.request.phase() != LayerwisePhase::Accepted {
            return Err("layerwise driver: transaction is not accepted".to_string());
        }
        let n = active.request.accepted_layer_count();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let stamp = active
                .request
                .accepted_stamp(i)
                .map_err(|e| format!("layerwise driver: accepted_stamp {i}: {e}"))?;
            let bc = active.request.accepted_block_count(i);
            let mut blocks = Vec::with_capacity(bc);
            for b in 0..bc {
                blocks.push(
                    active
                        .request
                        .accepted_block(i, b)
                        .map_err(|e| format!("layerwise driver: accepted_block {i},{b}: {e}"))?,
                );
            }
            let rc = active.request.accepted_resident_count(i);
            let mut resident = Vec::with_capacity(rc);
            for r in 0..rc {
                resident.push(active.request.accepted_resident(i, r));
            }
            out.push(LayerMemorySnapshot {
                model_layer: active.request.accepted_model_layer(i),
                memory: MemorySnapshot { stamp, blocks, resident },
            });
        }
        Ok(out)
    }

    /// Release the completed transaction (drops the core request and its revision
    /// closure). Idempotent; required before the next [`Self::begin_query`].
    pub fn end_query(&mut self) {
        self.active = None;
    }

    /// The frozen canonical history (advanced by [`Self::accept`]).
    pub fn canonical(&self) -> &[LayerMemorySnapshot] {
        &self.canonical
    }
    /// The current canonical history stamp (first layer's accepted stamp).
    pub fn stamp(&self) -> MemoryStamp {
        self.stamp
    }
    /// ordinal → model layer (the dense FA ordinals `0..n_fa_layers`).
    pub fn fa_layers(&self) -> &[u32] {
        &self.fa_layers
    }
    /// The canonical page size in tokens (pinned to the catalog layout).
    pub fn block_tokens(&self) -> u32 {
        self.block_tokens
    }
    /// The composite arena (residency queries / stripe views).
    pub fn arena(&self) -> &LayerwiseArena {
        &self.arena
    }
    pub fn arena_mut(&mut self) -> &mut LayerwiseArena {
        &mut self.arena
    }
    /// The active transaction's phase, or `None` when between queries.
    pub fn phase(&self) -> Option<LayerwisePhase> {
        self.active.as_ref().map(|a| a.request.phase())
    }
    /// The active transaction's cumulative projection/attendance counters.
    pub fn work(&self) -> Result<LayerwiseWork, String> {
        let active = self.active.as_ref().ok_or("layerwise driver: no active query")?;
        active
            .request
            .work()
            .map_err(|e| format!("layerwise driver: work: {e}"))
    }
    /// The reduced plan retained for ordinal `ordinal` (after project_restore).
    pub fn plan(&self, ordinal: usize) -> Option<&LayerPlan> {
        self.active.as_ref()?.plans.get(ordinal)?.as_ref()
    }
    /// The post-fence restored view retained for ordinal `ordinal` (after
    /// project_restore). `None` between queries or before the layer restores.
    pub fn view(&self, ordinal: usize) -> Option<&ExecutionView> {
        self.active.as_ref()?.views.get(ordinal)?.as_ref()
    }
    /// The number of canonical blocks sealed into the write-through archive.
    pub fn sealed_records(&self) -> usize {
        self.records.len()
    }

    /// Release the device arena, consuming the driver. The archive, records,
    /// slab, descriptor and any active request drop with it (their C++ handles
    /// are freed by their own `Drop`); `keys`/`policy` are borrowed and unaffected.
    pub fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        self.arena.free(gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layerwise_layout::Q8_PACKED_FORMAT;
    use hipfire_kvmem::{
        BackendIdentity, BlockCopies, BlockDescriptor, KvStore, KvStoreConfig, LayerMemorySnapshot,
        LayerQueryGeometry, LayerSelection, MemoryBudget, MemoryCapabilities, MemoryCatalog,
        MemoryDescriptor, MemorySnapshot, MemoryStamp, QueryRange, RawStoreConfig, SessionIdentity,
        Storage,
    };
    use rdna_compute::DType;

    /// Q8-ish geometry: 2 KV heads × head_dim 64 → kw = 128; 4 query heads →
    /// GQA group 2, qw = 256.
    const KV_HEADS: u32 = 2;
    const Q_HEADS: u32 = 4;
    const HEAD_DIM: u32 = 64;
    const KW: usize = (KV_HEADS * HEAD_DIM) as usize;
    const QW: usize = (Q_HEADS * HEAD_DIM) as usize;
    const BT: u32 = 128;

    fn raw_cfg(n_layer: u32) -> RawStoreConfig {
        RawStoreConfig {
            n_layer,
            n_embd_k: KW as u32,
            n_embd_v: KW as u32,
            block_tokens: BT,
            k_row_bytes: 0,
            k_gpu_row_bytes: 136,
            v_gpu_row_bytes: 136,
        }
    }

    /// Feed each block's Mean-K as a constant per-block vector so the gathered
    /// rows are deterministic and block-distinct.
    fn feed_mean_k(keys: &RawStore, n_blocks: u32, il: u32) {
        let mut row = vec![0f32; KW];
        for b in 0..n_blocks {
            for d in 0..KW {
                row[d] = (((b as usize * 7 + d * 3) % 17) as f32) * 0.1 - 0.8;
            }
            // 128 identical rows → block mean == the row vector.
            let tile: Vec<f32> = row.iter().copied().cycle().take(BT as usize * KW).collect();
            keys.write_layer_mean_k(b * BT, BT, il, &tile).expect("mean-k feed");
        }
    }

    /// Build a contiguous `[rows, query_heads, head_dim]` F32 device Q tile and
    /// its native descriptor (BYTE strides, token axis outermost here).
    fn device_q(gpu: &mut Gpu, rows: u32) -> (rdna_compute::GpuTensor, NativeCurrentLayerQuery) {
        let qw = QW;
        let mut host = vec![0f32; rows as usize * qw];
        for (i, v) in host.iter_mut().enumerate() {
            *v = (((i * 13) % 29) as f32) * 0.05 - 0.7;
        }
        let t = gpu.alloc_tensor(&[host.len() * 4], DType::Raw).expect("q alloc");
        gpu.hip.memcpy_htod(&t.buf, unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
        })
        .expect("q h2d");
        let q = NativeCurrentLayerQuery {
            data: t.buf.as_ptr() as *const std::ffi::c_void,
            bytes: host.len() * 4,
            layer: 0,
            rows,
            query_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            attention_scale: 0.0, // → 1/sqrt(head_dim)
            scalar: 0,            // F32
            row_stride: qw * 4,
            head_stride: HEAD_DIM as usize * 4,
            dimension_stride: 4,
        };
        (t, q)
    }

    fn frame(n: usize, mask: &[u8]) -> (Vec<u64>, Vec<u32>, Vec<u8>, Vec<u8>) {
        let starts: Vec<u64> = (0..n).map(|i| (i as u64) * BT as u64).collect();
        let counts: Vec<u32> = vec![BT; n];
        (starts, counts, mask.to_vec(), vec![0u8; n])
    }

    /// Keep-set lifecycle (design §3.10a): a Query step UNIONS the resident pool
    /// (sorted/deduped, admitting scorer selections + newly sealed pages); a
    /// Pressure step TRUNCATES to the refined selection. Pure host logic — no GPU.
    #[test]
    fn combine_keep_set_query_union_pressure_truncate() {
        let prev = vec![vec![0u64, 2, 3], vec![1, 2]];
        let restored = vec![vec![3u64, 4, 0], vec![2, 5]];
        // Query: union per ordinal, sorted + deduped.
        let q = combine_keep_set(&prev, &restored, false, 2);
        assert_eq!(q, vec![vec![0, 2, 3, 4], vec![1, 2, 5]]);
        // Pressure: truncate to the restored selection (sorted + deduped).
        let p = combine_keep_set(&prev, &restored, true, 2);
        assert_eq!(p, vec![vec![0, 3, 4], vec![2, 5]]);
        // Absent ordinal entries read as []; n_layers governs the output length.
        let short = combine_keep_set(&[], &restored, false, 3);
        assert_eq!(short, vec![vec![0, 3, 4], vec![2, 5], Vec::<u64>::new()]);
        // Empty restored on a Query step keeps the previous pool.
        let keep = combine_keep_set(&prev, &[vec![], vec![]], false, 2);
        assert_eq!(keep, vec![vec![0, 2, 3], vec![1, 2]]);
        // Empty restored on a Pressure step truncates to empty.
        let empty = combine_keep_set(&prev, &[vec![], vec![]], true, 2);
        assert_eq!(empty, vec![Vec::<u64>::new(), Vec::<u64>::new()]);
    }

    /// The bridge gathers Mean-K in enabled order, maps the native Q descriptor,
    /// scores on-device, and scatters back with masked blocks = 0. Verified
    /// against a DIRECT launcher call on the independently-gathered enabled
    /// subset (the launcher's numerics are D1-qualified; this isolates the
    /// bridge's gather/scatter/mapping), plus finiteness and an all-masked domain.
    #[test]
    #[ignore]
    fn scorer_bridge_matches_launcher_and_masks() {
        let mut gpu = Gpu::init().expect("gpu");
        let storage = Storage::new_ram(16 << 20).expect("storage");
        let keys = RawStore::new(raw_cfg(1), Some(&storage)).expect("raw store");
        let n = 5usize;
        keys.ensure_blocks(n as u32).expect("ensure blocks");
        feed_mean_k(&keys, n as u32, 0);

        let rows = 3u32;
        let (qt, q) = device_q(&mut gpu, rows);

        // Mask block 1 and block 3 → enabled = [0, 2, 4].
        let mask = vec![1u8, 0, 1, 0, 1];
        let (starts, counts, score_mask, transport_mask) = frame(n, &mask);
        let fr = QueryScoreFrame {
            starts: &starts,
            counts: &counts,
            score_mask: &score_mask,
            transport_mask: &transport_mask,
            accepted_through: n as u64 * BT as u64,
        };

        let out = score_current_layer_native(&mut gpu, &keys, 0, BT, &q, &fr).expect("bridge");
        assert_eq!(out.len(), n);
        for &v in &out {
            assert!(v.is_finite() && v >= 0.0, "score finite non-negative: {v}");
        }
        assert_eq!(out[1], 0.0, "masked block 1 must be 0");
        assert_eq!(out[3], 0.0, "masked block 3 must be 0");

        // Independent direct-launcher reference on the enabled subset.
        let enabled: Vec<usize> = (0..n).filter(|&i| mask[i] != 0).collect();
        let mut means = vec![0f32; enabled.len() * KW];
        let mut row = vec![0f32; KW];
        for (j, &i) in enabled.iter().enumerate() {
            keys.mean_k(i as u32, 0, &mut row);
            means[j * KW..(j + 1) * KW].copy_from_slice(&row);
        }
        let pq = PageScoreQuery {
            data: q.data,
            bytes: q.bytes,
            rows: q.rows,
            query_heads: q.query_heads,
            kv_heads: q.kv_heads,
            head_dim: q.head_dim,
            attention_scale: q.attention_scale,
            scalar: PageQueryScalar::F32,
            row_stride: q.row_stride,
            head_stride: q.head_stride,
            dimension_stride: q.dimension_stride,
        };
        let mut masses = vec![0f64; enabled.len()];
        gpu.kvmem_page_score_current_layer(&pq, &means, enabled.len(), &mut masses)
            .expect("direct launcher");
        for (j, &i) in enabled.iter().enumerate() {
            assert_eq!(out[i], masses[j], "enabled block {i} mass mismatch");
        }
        // Mass conservation over the scored domain (Σ pages == rows), the D1
        // contract: no Query row is dropped.
        let sum: f64 = masses.iter().sum();
        assert!((sum - rows as f64).abs() < 1e-6, "mass sum {sum} != rows {rows}");

        // All-masked domain: every score 0, Q still validated by the launcher.
        let zero_mask = vec![0u8; n];
        let (s2, c2, m2, t2) = frame(n, &zero_mask);
        let fr2 = QueryScoreFrame {
            starts: &s2,
            counts: &c2,
            score_mask: &m2,
            transport_mask: &t2,
            accepted_through: n as u64 * BT as u64,
        };
        let out2 = score_current_layer_native(&mut gpu, &keys, 0, BT, &q, &fr2).expect("empty domain");
        assert!(out2.iter().all(|&v| v == 0.0), "all-masked → all zero");

        gpu.free_tensor(qt).expect("free q");
    }

    /// The arena owner allocates ONE zero-filled Raw tensor of `arena_bytes`, the
    /// pool tracks composite-slot residency over it, and `stripe_views` lands on
    /// the layout's plane offsets (base + slot/layer offset, plane-sized). Drives
    /// the full alloc → bind → view → drop-evict → free lifecycle on device,
    /// including the generation guard on slot reuse.
    #[test]
    #[ignore]
    fn arena_owner_alloc_bind_view_free() {
        let mut gpu = Gpu::init().expect("gpu");
        // 2 fa layers, 4 KV heads × head_dim 256 → Q8 stride 1088 (the layout
        // adapter's host fixture). 3 composite slots ≈ 1.6 MiB arena.
        let q8 = LayerwiseQ8Layout::resolve(2, 4, 256, &[0, 1]).expect("q8 layout");
        let n_slots = 3usize;
        let mut arena = LayerwiseArena::new(&mut gpu, &q8, n_slots).expect("arena");
        let layout = *arena.layout();
        assert_eq!(layout.n_slots(), n_slots);
        assert_eq!(layout.n_layers(), 2);
        assert_eq!(layout.k_stride(), 1088);
        assert_eq!(arena.arena().buf.size(), layout.arena_bytes());
        assert_eq!(arena.arena().dtype, DType::Raw);
        assert_eq!(arena.pool_ref().free_count(), n_slots);

        // Reserve a slot; bind distinct canonical pages into both layer stripes.
        let h = arena.pool().alloc_slot().expect("alloc slot");
        assert_eq!(arena.pool_ref().free_count(), n_slots - 1);
        assert!(arena.pool_ref().is_allocated(h.phys));
        arena.pool().bind_stripe(h, 0, 10).expect("bind l0");
        arena.pool().bind_stripe(h, 1, 20).expect("bind l1");
        assert_eq!(arena.pool_ref().slot_refs(h.phys), 2);
        assert_eq!(arena.pool_ref().stripe_page(h.phys, 0), Some(10));
        assert_eq!(arena.pool_ref().stripe_page(h.phys, 1), Some(20));
        assert!(arena.pool_ref().is_page_resident(10));
        assert!(arena.pool_ref().is_page_resident(20));

        // Stripe views alias the arena at exactly the layout plane offsets.
        let base = arena.arena().buf.as_ptr() as usize;
        for layer in 0..2usize {
            let (k, v) = arena.stripe_views(h.phys as usize, layer).expect("views");
            let k_off = layout.layer_k_offset(h.phys as usize, layer).unwrap();
            let v_off = layout.layer_v_offset(h.phys as usize, layer).unwrap();
            assert_eq!(k.buf.as_ptr() as usize, base + k_off, "l{layer} K offset");
            assert_eq!(v.buf.as_ptr() as usize, base + v_off, "l{layer} V offset");
            assert_eq!(k.buf.size(), layout.k_plane_bytes());
            assert_eq!(v.buf.size(), layout.v_plane_bytes());
            assert_eq!(k.dtype, DType::Raw);
        }

        // A second slot's views carry the slot_base stride (not slot 0's).
        let hb = arena.pool().alloc_slot().expect("alloc slot b");
        assert_ne!(hb.phys, h.phys);
        arena.pool().bind_stripe(hb, 0, 30).expect("bind b l0");
        let (kb, _vb) = arena.stripe_views(hb.phys as usize, 0).expect("views b");
        assert_eq!(
            kb.buf.as_ptr() as usize,
            base + layout.layer_k_offset(hb.phys as usize, 0).unwrap()
        );

        // Compact views over the resident slots; shared extent = 1 page each.
        arena.pool().set_view(0, &[h.phys, hb.phys], 256).expect("view l0");
        arena.pool().set_view(1, &[h.phys], 128).expect("view l1");
        // Mismatched extents are rejected (core shared-extent rule).
        assert!(arena.pool_ref().validate_shared_extent().is_err());
        arena.pool().set_view(1, &[h.phys], 256).expect("fix l1 extent");
        arena.pool_ref().validate_shared_extent().expect("shared extent");
        assert_eq!(arena.pool_ref().compact_slots(0).unwrap(), &[h.phys, hb.phys]);
        assert_eq!(arena.pool_ref().canonical_pages(0).unwrap(), vec![10, 30]);
        assert_eq!(arena.pool_ref().canonical_pages(1).unwrap(), vec![20]);
        assert_eq!(arena.pool_ref().visible_tokens(0).unwrap(), 256);

        // Drop-evict: a slot frees only after EVERY layer stripe drops.
        assert!(!arena.pool().drop_stripe(h, 0).expect("drop l0"));
        assert!(arena.pool().free_slot(h).is_err(), "refs!=0 must reject free");
        assert!(arena.pool().drop_stripe(h, 1).expect("drop l1"));
        arena.pool().free_slot(h).expect("free slot");
        assert_eq!(arena.pool_ref().free_count(), n_slots - 1); // hb still held

        // Reuse bumps the generation; the stale handle is rejected.
        let h2 = arena.pool().alloc_slot().expect("realloc");
        assert_eq!(h2.phys, h.phys);
        assert_ne!(h2.generation, h.generation);
        assert!(arena.pool().bind_stripe(h, 0, 99).is_err(), "stale handle");
        arena.pool().bind_stripe(h2, 0, 99).expect("fresh handle");

        // Tear down: drop the remaining stripes, free both slots, free the arena.
        arena.pool().drop_stripe(h2, 0).expect("drop h2");
        arena.pool().free_slot(h2).expect("free h2");
        arena.pool().drop_stripe(hb, 0).expect("drop hb");
        arena.pool().free_slot(hb).expect("free hb");
        assert_eq!(arena.pool_ref().free_count(), n_slots);
        arena.free(&mut gpu).expect("free arena");
    }

    // ── catalog / identity / spec construction fixture ──
    // Mirrors the core `layerwise_request_test.cpp` Fixture (two target layers,
    // two canonical pages, native IDs deliberately ≠ dense ordinals) but with
    // hipfire Q8 geometry: 1 KV head × head_dim 32 → stride 34, page 128.
    const FX_LAYERS: u32 = 2; // fa layers 0,1 (ordinal == model layer here)
    const FX_KV_HEADS: usize = 1;
    const FX_HEAD_DIM: usize = 32; // Q8 stride = 1 * (32/32) * 34 = 34
    const FX_BT: u32 = 128;
    const FX_STRIDE: u64 = 34;
    const FX_PLANE: u64 = FX_BT as u64 * FX_STRIDE; // 4352
    const FX_LAYER_BYTES: u64 = 2 * FX_PLANE; // 8704 (one layer K+V)
    const FX_RECORD: u64 = FX_LAYERS as u64 * FX_LAYER_BYTES; // 17408
    const FX_FRONTIER: u64 = 2 * FX_BT as u64; // 256

    fn fx_session() -> SessionIdentity {
        SessionIdentity { id: 42, incarnation: 3 }
    }
    fn fx_stamp() -> MemoryStamp {
        MemoryStamp {
            valid: 1,
            session: fx_session(),
            history_revision: 5,
            view_version: 6,
            frontier: FX_FRONTIER,
            execution_history: 9,
        }
    }
    fn fx_block(id: u64, start: u64, cv: u64) -> BlockDescriptor {
        BlockDescriptor {
            session: fx_session(),
            id,
            original_start: start,
            valid_tokens: FX_BT,
            content_version: cv,
            statistics_version: cv,
        }
    }
    fn fx_layout() -> LayerwiseQ8Layout {
        LayerwiseQ8Layout::resolve(FX_LAYERS as usize, FX_KV_HEADS, FX_HEAD_DIM, &[0, 1]).unwrap()
    }
    fn fx_backend() -> BackendIdentity {
        BackendIdentity {
            backend: Q8_PACKED_FORMAT.to_string(),
            revision: "fx-rev".to_string(),
            model: "fx-model".to_string(),
            state_schema: "state-v1".to_string(),
        }
    }
    fn fx_capabilities() -> MemoryCapabilities {
        MemoryCapabilities {
            page_tokens: FX_BT,
            max_position: 4096,
            native_payload: 1,
            sparse_view: 1,
            original_rope_positions: 1,
            pre_rope_statistics: 1,
            state_checkpoint: 1,
            asynchronous_transfer: 0,
        }
    }
    fn fx_budget() -> MemoryBudget {
        MemoryBudget {
            selected_tokens: FX_BT as u64,
            reserve_tokens: FX_BT as u64,
            device_tokens: 2 * FX_BT as u64,
            // host_payload_bytes covers EVERY sealed canonical record (both
            // pages have host_version != 0), not just the restored one
            // (memory_contract.cpp:173,220: host_records × record_bytes).
            host_payload_bytes: 2 * FX_RECORD,
        }
    }
    /// Two canonical pages (100@0, 200@128); layer 0 keeps 100 resident, layer 1
    /// keeps 200 resident (equal 128-token extent). host_version==content_version
    /// (sealed-host admission). Per-layer view_version diverges like the fixture.
    fn fx_snapshots() -> Vec<LayerMemorySnapshot> {
        let blocks = vec![
            BlockCopies { block: fx_block(100, 0, 1), host_version: 1 },
            BlockCopies { block: fx_block(200, FX_BT as u64, 2), host_version: 2 },
        ];
        (0..FX_LAYERS)
            .map(|l| {
                let mut stamp = fx_stamp();
                stamp.view_version += l as u64;
                LayerMemorySnapshot {
                    model_layer: l,
                    memory: MemorySnapshot {
                        stamp,
                        blocks: blocks.clone(),
                        resident: vec![if l == 0 { 100 } else { 200 }],
                    },
                }
            })
            .collect()
    }

    /// Construction recipe validated host-side (no GPU/scorer/restore): the Q8
    /// manifest, descriptor and catalog admit the fixture history, and the
    /// offline bulk `plan_layer` reduces a cold-block selection to the expected
    /// compact restore/next geometry. This isolates the budget/identity math the
    /// state-machine driver relies on.
    #[test]
    fn catalog_offline_plan_matches_core_geometry() {
        let q8 = fx_layout();
        let layout = q8.payload_layout().expect("manifest");
        assert_eq!(layout.record_bytes(), FX_RECORD);
        assert_eq!(layout.block_tokens(), FX_BT);
        let descriptor =
            MemoryDescriptor::new(&fx_backend(), &layout, &fx_capabilities()).expect("descriptor");
        let catalog = MemoryCatalog::new(&descriptor, &fx_snapshots()).expect("catalog");
        assert_eq!(catalog.layer_count(), FX_LAYERS as usize);
        assert_eq!(catalog.model_layer(0), 0);
        assert_eq!(catalog.model_layer(1), 1);
        assert_eq!(catalog.snapshot_block_count(0), 2);
        assert_eq!(catalog.snapshot_resident(0, 0), 100);
        assert_eq!(catalog.snapshot_resident(1, 0), 200);

        // Offline plan for layer 0 selecting the COLD page 200 (resident=[100]).
        let sel = LayerSelection {
            model_layer: 0,
            expected: fx_stamp(),
            selected: vec![200],
            mandatory: vec![],
            budget: fx_budget(),
        };
        let plan = catalog.plan_layer(&sel).expect("plan_layer");
        assert_eq!(plan.model_layer(), 0);
        assert_eq!(plan.restore_count(), 1, "cold page 200 must restore");
        assert_eq!(plan.restore_block(0).unwrap().id, 200);
        assert_eq!(plan.spill_count(), 0);
        // Selecting only the cold page evicts the resident page 100; its host
        // copy is current (host_version==content_version) so it drops WITHOUT
        // spill — the bare drop-evict invariant.
        assert_eq!(plan.evict_count(), 1);
        assert_eq!(plan.evict_block(0).unwrap().id, 100);
        assert_eq!(plan.next_block_count(), 1);
        let nb = plan.next_block(0).unwrap();
        assert_eq!(nb.block.id, 200);
        assert_eq!(nb.compact_start, 0);
        assert_eq!(plan.next_visible_tokens(), FX_BT as u64);
        let next_stamp = plan.next_stamp().expect("next stamp");
        assert_eq!(next_stamp.frontier, FX_FRONTIER);
        assert_eq!(next_stamp.view_version, fx_stamp().view_version + 1);
        // Per-layer transfer regions: one layer's {K, V} planes.
        let regions = plan.payload().expect("payload regions");
        assert_eq!(regions.region_count(), 2);
        println!(
            "[fx] restore_bytes={} spill_bytes={} device_tokens={} host_bytes={} \
             regions.packed={} plane={} record={}",
            plan.restore_bytes(),
            plan.spill_bytes(),
            plan.device_tokens_required(),
            plan.host_payload_bytes_required(),
            regions.packed_bytes(),
            regions.plane_bytes(),
            regions.record_bytes()
        );
        assert_eq!(plan.restore_bytes(), FX_LAYER_BYTES, "one layer K+V");
    }

    /// Deterministic packed Q8 row bytes for block ordinal `b`, layer `il`, K/V.
    fn fx_pack(b: u32, il: u32, is_v: bool) -> Vec<u8> {
        let stride = FX_STRIDE as usize;
        let mut v = vec![0u8; FX_BT as usize * stride];
        for t in 0..FX_BT as usize {
            for j in 0..stride {
                let base = if is_v {
                    b * 11 + il * 17 + (t as u32) * 5 + 100
                } else {
                    b * 7 + il * 13 + (t as u32) * 3
                };
                v[t * stride + j] = ((base + j as u32) % 251) as u8;
            }
        }
        v
    }

    /// The restore data path (design §3.5): seal the write-through archive, read
    /// a cold layer's packed extent via KvRegions into the pinned slab, H2D
    /// into the composite arena stripe, fence, and publish the compact view +
    /// [`ExecutionView`]. Verified byte-exact against the fed Q8 pattern (D2H
    /// readback), plus the pool compact table and the exact `plan.next` view.
    #[test]
    #[ignore]
    fn restore_layer_archive_to_arena_byte_exact() {
        let mut gpu = Gpu::init().expect("gpu");
        let q8 = fx_layout();
        let layout = q8.payload_layout().expect("manifest");
        let backend = fx_backend();
        let descriptor =
            MemoryDescriptor::new(&backend, &layout, &fx_capabilities()).expect("descriptor");
        let catalog = MemoryCatalog::new(&descriptor, &fx_snapshots()).expect("catalog");

        // Statistics store: feed packed K/V for both canonical pages, both layers.
        let keys_storage = Storage::new_ram(16 << 20).expect("keys storage");
        let keys = RawStore::new(q8.raw_config().clone(), Some(&keys_storage)).expect("keys");
        keys.ensure_blocks(2).expect("ensure blocks");
        for b in 0..2u32 {
            for il in 0..FX_LAYERS {
                keys.write_layer_k_gpu(b * FX_BT, FX_BT, il, &fx_pack(b, il, false))
                    .expect("feed K");
                keys.write_layer_v_gpu(b * FX_BT, FX_BT, il, &fx_pack(b, il, true))
                    .expect("feed V");
            }
        }

        // Sealed archive + reusable pinned slab + composite arena.
        let archive = Storage::new_ram(16 << 20).expect("archive storage");
        let mut records: HashMap<u64, HostRecord> = HashMap::new();
        let mut slab = gpu.hip.transfer_buffer(FX_LAYER_BYTES as usize).expect("slab");
        let mut arena = LayerwiseArena::new(&mut gpu, &q8, 3).expect("arena");

        // Offline plan: layer 0 selects the cold page 200 (catalog resident=[100]).
        let sel = LayerSelection {
            model_layer: 0,
            expected: fx_stamp(),
            selected: vec![200],
            mandatory: vec![],
            budget: fx_budget(),
        };
        let plan = catalog.plan_layer(&sel).expect("plan");
        let view = restore_layer_pages(
            &mut gpu, &mut arena, &archive, &keys, &backend, &layout, &mut records, &mut slab,
            &plan, 0, FX_BT,
        )
        .expect("restore");

        // Compact view binds exactly the selected page (dense ordinal 1).
        assert_eq!(arena.pool_ref().compact_slots(0).unwrap().len(), 1);
        assert_eq!(arena.pool_ref().canonical_pages(0).unwrap(), vec![1]);
        assert_eq!(arena.pool_ref().visible_tokens(0).unwrap(), FX_BT as usize);
        let phys = arena.pool_ref().compact_slots(0).unwrap()[0];
        assert_eq!(arena.pool_ref().stripe_page(phys, 0), Some(1));

        // D2H readback: the stripe holds block 200's layer-0 K then V, byte-exact.
        let k_off = arena.layout().layer_k_offset(phys as usize, 0).unwrap();
        let v_off = arena.layout().layer_v_offset(phys as usize, 0).unwrap();
        let mut k_read = vec![0u8; FX_PLANE as usize];
        let mut v_read = vec![0u8; FX_PLANE as usize];
        gpu.hip
            .memcpy_dtoh_at(&mut k_read, &arena.arena().buf, k_off)
            .expect("d2h K");
        gpu.hip
            .memcpy_dtoh_at(&mut v_read, &arena.arena().buf, v_off)
            .expect("d2h V");
        assert_eq!(k_read, fx_pack(1, 0, false), "restored K bytes");
        assert_eq!(v_read, fx_pack(1, 0, true), "restored V bytes");

        // ExecutionView == plan.next exactly (restored() same_view contract).
        assert_eq!(view.visible_tokens, plan.next_visible_tokens());
        assert_eq!(view.stamp, plan.next_stamp().unwrap());
        assert_eq!(view.blocks.len(), plan.next_block_count());
        for i in 0..view.blocks.len() {
            assert_eq!(view.blocks[i], plan.next_block(i).unwrap());
        }
        // The cold record was sealed once and cached.
        assert_eq!(records.len(), 1);
        assert!(records.contains_key(&200));

        arena.free(&mut gpu).expect("free arena");
    }

    /// Full layerwise transaction on device (mirrors the core
    /// `layerwise_request_test.cpp` `execute()` + `current_forward_and_acceptance()`):
    /// two fa layers, each selects its COLD page via the GPU scorer, restores it
    /// byte-exact into the composite arena, attends, and accepts. Verifies the
    /// per-layer selection {{200},{100}}, the cumulative work counters, the
    /// accepted resident sets {200,300}/{100,300}, frontier == evaluation.end,
    /// the three-block catalog, the advanced stamps, the core readback equality,
    /// and the canonical-history advance. Mean-K is fed so layer 0 favors page
    /// 200 and layer 1 favors page 100; the shared all-+1 Q tile then selects the
    /// cold page in each layer through the query-span intersection.
    #[test]
    #[ignore]
    fn driver_end_to_end_layerwise_transaction() {
        let mut gpu = Gpu::init().expect("gpu");
        let q8 = fx_layout();
        let kw = FX_KV_HEADS * FX_HEAD_DIM; // 32 (query_heads == kv_heads == 1)

        // Policy: one-page selected budget, no sink/recent (pure score selection).
        let policy = KvStore::new(&KvStoreConfig {
            block_tokens: FX_BT,
            select_budget: FX_BT,
            sink_blocks: 0,
            recent_blocks: 0,
            ..Default::default()
        })
        .expect("policy");
        policy.register_append(FX_FRONTIER as u32).expect("append");

        // Statistics + packed store: both canonical pages, both layers.
        let keys_storage = Storage::new_ram(16 << 20).expect("keys storage");
        let keys = RawStore::new(q8.raw_config().clone(), Some(&keys_storage)).expect("keys");
        keys.ensure_blocks(2).expect("ensure blocks");
        for b in 0..2u32 {
            for il in 0..FX_LAYERS {
                keys.write_layer_k_gpu(b * FX_BT, FX_BT, il, &fx_pack(b, il, false))
                    .expect("feed K");
                keys.write_layer_v_gpu(b * FX_BT, FX_BT, il, &fx_pack(b, il, true))
                    .expect("feed V");
            }
        }
        // Mean-K: layer 0 favors ordinal 1 (page 200), layer 1 favors ordinal 0
        // (page 100). sign = +1 iff (b==1) == (il==0).
        for il in 0..FX_LAYERS {
            for b in 0..2u32 {
                let sign = if (b == 1) == (il == 0) { 1.0f32 } else { -1.0f32 };
                let tile: Vec<f32> = vec![sign; FX_BT as usize * kw];
                keys.write_layer_mean_k(b * FX_BT, FX_BT, il, &tile).expect("mean-k");
            }
        }

        // Shared all-+1 native Q tile: 4 evaluation rows, query_heads=1.
        let rows = 4u32;
        let host = vec![1.0f32; rows as usize * kw];
        let qt = gpu.alloc_tensor(&[host.len() * 4], DType::Raw).expect("q alloc");
        gpu.hip
            .memcpy_htod(&qt.buf, unsafe {
                std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
            })
            .expect("q h2d");
        let q_base = NativeCurrentLayerQuery {
            data: qt.buf.as_ptr() as *const std::ffi::c_void,
            bytes: host.len() * 4,
            layer: 0,
            rows,
            query_heads: 1,
            kv_heads: FX_KV_HEADS as u32,
            head_dim: FX_HEAD_DIM as u32,
            attention_scale: 0.0,
            scalar: 0,
            row_stride: kw * 4,
            head_stride: FX_HEAD_DIM * 4,
            dimension_stride: 4,
        };

        let mut driver = LayerwiseDriver::new(
            &mut gpu,
            &keys,
            &policy,
            &q8,
            fx_backend(),
            fx_capabilities(),
            fx_snapshots(),
            3,
            16 << 20,
        )
        .expect("driver");

        let identity = LayerwiseIdentity {
            history: fx_stamp(),
            request: 77,
            evaluation: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER + rows as u64 },
            query: QueryRange { begin: FX_FRONTIER + 1, end: FX_FRONTIER + 3 },
        };
        let geom = |ordinal: u32| LayerQueryGeometry {
            layer: ordinal,
            query_heads: 1,
            kv_heads: FX_KV_HEADS as u32,
            head_dim: FX_HEAD_DIM as u32,
            attention_scale: 0.0,
        };
        let specs = vec![
            LayerRequestSpec { model_layer: 0, query: geom(0), budget: fx_budget(), mandatory: vec![] },
            LayerRequestSpec { model_layer: 1, query: geom(1), budget: fx_budget(), mandatory: vec![] },
        ];
        // native_history_tokens == the selected working-set extent (one page).
        driver.begin_query(identity, specs, FX_BT as u64).expect("begin_query");
        assert_eq!(driver.phase(), Some(LayerwisePhase::BetweenLayers));

        let mut selected = Vec::new();
        for ordinal in 0..2usize {
            let mut q = q_base;
            q.layer = ordinal as u32;
            let view = driver.project_restore(&mut gpu, ordinal, &q).expect("project_restore");
            let want = if ordinal == 0 { 200u64 } else { 100u64 };
            assert_eq!(view.blocks.len(), 1, "one selected page");
            assert_eq!(view.blocks[0].block.id, want, "layer {ordinal} selects the cold page");
            assert_eq!(view.visible_tokens, FX_BT as u64);
            selected.push(view.blocks[0].block.id);
            // The arena stripe binds exactly the selected canonical page.
            let page = if ordinal == 0 { 1u32 } else { 0u32 };
            assert_eq!(driver.arena().pool_ref().canonical_pages(ordinal).unwrap(), vec![page]);
            driver
                .attend_complete(ordinal, identity.evaluation.begin, rows as u64)
                .expect("attend_complete");
        }
        assert_eq!(selected, vec![200u64, 100u64]);
        assert_eq!(driver.phase(), Some(LayerwisePhase::AwaitingAcceptance));
        let work = driver.work().expect("work");
        assert_eq!(work.projected_rows, 8, "2 layers × 4 evaluation rows");
        assert_eq!(work.attended_rows, 8);
        assert_eq!(work.completed_layers, 2);
        assert_eq!(driver.sealed_records(), 2, "both cold pages sealed once");

        // Accept: evaluated native suffix (id 300) at the evaluation frontier.
        let suffix = BlockCopies {
            block: BlockDescriptor {
                session: fx_session(),
                id: 300,
                original_start: identity.evaluation.begin,
                valid_tokens: rows,
                content_version: 3,
                statistics_version: 1,
            },
            host_version: 0,
        };
        let snapshots = driver.accept(suffix).expect("accept");
        assert_eq!(driver.phase(), Some(LayerwisePhase::Accepted));
        let readback = driver.accepted_snapshots().expect("readback");
        assert_eq!(readback, snapshots, "core readback == published snapshots");
        for ordinal in 0..2usize {
            let s = &snapshots[ordinal];
            assert_eq!(s.model_layer, ordinal as u32);
            assert_eq!(s.memory.stamp.frontier, identity.evaluation.end);
            assert_eq!(s.memory.stamp.history_revision, fx_stamp().history_revision + 1);
            assert_eq!(s.memory.stamp.execution_history, fx_stamp().execution_history + 1);
            // plan.next adds one view_version, accept adds another; per-layer base diverges.
            assert_eq!(
                s.memory.stamp.view_version,
                fx_stamp().view_version + ordinal as u64 + 2
            );
            assert_eq!(s.memory.blocks.len(), 3, "frozen prefix + evaluated suffix");
            assert_eq!(s.memory.blocks[2].block.id, 300);
            assert_eq!(s.memory.blocks[2].host_version, 0);
            let want_resident = if ordinal == 0 { vec![200u64, 300] } else { vec![100u64, 300] };
            assert_eq!(s.memory.resident, want_resident);
        }
        // Canonical history advanced to the accepted snapshots.
        assert_eq!(driver.canonical(), snapshots.as_slice());
        assert_eq!(driver.stamp(), snapshots[0].memory.stamp);

        driver.end_query();
        assert_eq!(driver.phase(), None);
        gpu.free_tensor(qt).expect("free q");
        driver.free(&mut gpu).expect("free driver");
    }

    /// Host-only: activation admission is fail-closed BEFORE any allocation
    /// (ninfer-faithful, design §3.11/§5.1 E1). A too-small quota for the model
    /// geometry yields `Err` with nothing acquired; the RAM/disk tier decision
    /// flips on the tape peak. No GPU — `admit` is pure arithmetic plus the
    /// shim's `metadata_bound`, so the whole admission contract is host-testable.
    #[test]
    fn layerwise_activation_admission_fail_closed() {
        let id = LayerwiseIdentity {
            history: fx_stamp(),
            request: 91,
            evaluation: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER + 1 },
            query: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER + 1 },
        };
        let n_layers = 2usize;
        let full_pages = 2u64;
        let q_row_bytes = QW * 4; // 1024
        let base = LayerwiseQuotas::default();

        // The seeds admit the tiny geometry on the RAM tier; the plan carries the
        // exact geometry-derived bounds the constructor will reserve.
        let plan = LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &base)
            .expect("admit default");
        assert!(!plan.use_disk);
        assert_eq!(plan.pin_bytes, q_row_bytes as u64); // one decode row
        assert_eq!(plan.ram_tapes, base.activation_ram - plan.pin_bytes);
        assert!(plan.control_bytes > 0);
        assert_eq!(plan.source_bound, full_pages * SOURCE_VERSION_DESC_BYTES);
        assert_eq!(
            plan.frozen_bound,
            n_layers as u64 * (FROZEN_SNAPSHOT_BASE_BYTES + full_pages * FROZEN_BLOCK_DESC_BYTES)
        );
        assert_eq!(plan.feedback_bound, n_layers as u64 * FEEDBACK_DESC_BYTES);

        // Zero-sized geometry / empty evaluation span fail closed.
        assert!(LayerwiseActivation::admit(&id, 0, full_pages, q_row_bytes, &base).is_err());
        assert!(LayerwiseActivation::admit(&id, n_layers, full_pages, 0, &base).is_err());
        let empty = LayerwiseIdentity {
            evaluation: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER },
            ..id
        };
        assert!(LayerwiseActivation::admit(&empty, n_layers, full_pages, q_row_bytes, &base).is_err());

        // activation_metadata too small (control + per-tape bounds cannot fit).
        let mut q = base.clone();
        q.activation_metadata = 1;
        assert!(LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &q).is_err());
        q.activation_metadata = 0;
        assert!(LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &q).is_err());

        // metadata envelope too small: transient partition == 0, then bounds exceed.
        let mut q = base.clone();
        q.metadata = 7; // 7/8 == 0 → transient zero
        assert!(LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &q).is_err());
        let mut q = base.clone();
        q.metadata = 8; // transient 1; frozen_bound 512 > 1
        assert!(LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &q).is_err());

        // staging too small for the pinned Q readback.
        let mut q = base.clone();
        q.staging = (q_row_bytes as u64) - 1;
        assert!(LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &q).is_err());

        // activation_ram too small for even the staging slice.
        let mut q = base.clone();
        q.activation_ram = (q_row_bytes as u64) - 1;
        assert!(LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &q).is_err());

        // Disk tier: tape_peak = 2*1024 = 2048; pin = 1024; activation_ram = 2048
        // → ram_tapes = 1024 < 2048 → disk. Without a directory it fails closed.
        let mut q = base.clone();
        q.activation_ram = 2048;
        q.staging = 1024;
        q.activation_disk = 4096;
        q.disk_directory = None;
        assert!(LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &q).is_err());
        // With a directory the disk tier is admitted.
        q.disk_directory = Some(std::path::PathBuf::from("/tmp/hipfire-activation-admit"));
        let plan = LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &q)
            .expect("admit disk tier");
        assert!(plan.use_disk);
        // activation_disk too small for the tape peak → fail closed.
        q.activation_disk = 2047;
        assert!(LayerwiseActivation::admit(&id, n_layers, full_pages, q_row_bytes, &q).is_err());
    }

    /// GPU: the activation subsystem captures each FA layer's projected Q via a
    /// pinned D2H readback into its quota-backed tape and reads it back byte-exact
    /// (design §5.1 E1, RAM tier). Also proves the readback is NON-DESTRUCTIVE —
    /// the scorer's GPU Q tile is byte-unchanged after capture, the additive
    /// guarantee — and that all five quotas are live, charged accounts.
    #[test]
    #[ignore]
    fn layerwise_activation_q_readback_byte_exact() {
        let mut gpu = Gpu::init().expect("gpu");
        let id = LayerwiseIdentity {
            history: fx_stamp(),
            request: 92,
            evaluation: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER + 1 },
            query: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER + 1 },
        };
        let n_layers = 2usize;
        let full_pages = 2u64;
        let q_row_bytes = QW * 4; // 1024 (4 query heads × 64 dim × 4)
        let quotas = LayerwiseQuotas::default();
        let mut act =
            LayerwiseActivation::new(&mut gpu, id, n_layers, full_pages, q_row_bytes, &quotas)
                .expect("activation");
        assert!(!act.use_disk());
        assert_eq!(act.n_layers(), n_layers);
        assert_eq!(act.q_row_bytes(), q_row_bytes);

        // Per-layer distinct F32 Q oracle (one decode row), uploaded to the GPU.
        let mut tiles = Vec::new();
        for ordinal in 0..n_layers {
            let host: Vec<f32> = (0..QW)
                .map(|i| (((i * 13 + ordinal * 101) % 29) as f32) * 0.05 - 0.7)
                .collect();
            let bytes = unsafe {
                std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
            }
            .to_vec();
            let t = gpu.alloc_tensor(&[host.len() * 4], DType::Raw).expect("q alloc");
            gpu.hip.memcpy_htod(&t.buf, &bytes).expect("q h2d");
            let q = NativeCurrentLayerQuery {
                data: t.buf.as_ptr() as *const std::ffi::c_void,
                bytes: host.len() * 4,
                layer: ordinal as u32,
                rows: 1,
                query_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                attention_scale: 0.0,
                scalar: 0,
                row_stride: QW * 4,
                head_stride: HEAD_DIM as usize * 4,
                dimension_stride: 4,
            };
            act.capture_layer(&mut gpu, ordinal, &q).expect("capture");

            // Byte-exact tape readback (absolute token begin == evaluation.begin).
            let mut got = vec![0u8; q_row_bytes];
            act.read_layer(ordinal, FX_FRONTIER, 1, &mut got).expect("read");
            assert_eq!(got, bytes, "layer {ordinal} Q tape byte-exact");

            // Non-destructive: the GPU Q tile is unchanged after the D2H capture.
            let mut after = vec![0u8; q_row_bytes];
            gpu.hip.memcpy_dtoh_at(&mut after, &t.buf, 0).expect("q d2h");
            assert_eq!(after, bytes, "layer {ordinal} GPU Q tile untouched (additive)");
            tiles.push(t);
        }

        // All five quotas are live, charged accounts (RAM tier).
        let tape_peak = (n_layers * q_row_bytes) as u64;
        assert_eq!(act.ram_charged_bytes(), tape_peak, "activation_ram charged the tape peak");
        assert_eq!(act.disk_charged_bytes(), 0, "RAM tier: no disk charge");
        // activation_metadata == the unused control reservation + each tape's bound.
        let lay = act.layout().clone();
        let per_tape_meta = QueryProjectionTape::metadata_bound(&lay, 1).expect("bound");
        assert_eq!(
            act.metadata_charged_bytes(),
            act.control_remaining() + per_tape_meta * n_layers as u64
        );
        assert!(act.metadata_charged_bytes() <= quotas.activation_metadata);
        assert!(act.control_remaining() > 0);
        // The envelope partitions are reserved exactly (unused → remaining == bound).
        let source_bound = full_pages * SOURCE_VERSION_DESC_BYTES;
        let frozen_bound =
            n_layers as u64 * (FROZEN_SNAPSHOT_BASE_BYTES + full_pages * FROZEN_BLOCK_DESC_BYTES);
        let feedback_bound = n_layers as u64 * FEEDBACK_DESC_BYTES;
        assert_eq!(act.source_versions_remaining(), source_bound);
        assert_eq!(act.frozen_index_remaining(), frozen_bound);
        assert_eq!(act.feedback_remaining(), feedback_bound);
        assert_eq!(
            act.envelope_charged_bytes(),
            source_bound + frozen_bound + feedback_bound
        );
        assert!(act.envelope_charged_bytes() <= quotas.metadata);

        // A foreign ordinal / range fails closed.
        assert!(act.read_layer(n_layers, FX_FRONTIER, 1, &mut vec![0u8; q_row_bytes]).is_err());
        assert!(act.read_layer(0, FX_FRONTIER + 1, 1, &mut vec![0u8; q_row_bytes]).is_err());

        for t in tiles {
            gpu.free_tensor(t).expect("free q");
        }
        drop(act);
    }

    /// GPU: when the tape peak exceeds the RAM account the activation admits the
    /// DISK tier ([`SpillFile`]) and the Q readback is still byte-exact from disk
    /// (design §5.1 E1). Forces the tier with a small activation_ram + a temp
    /// spill directory; asserts the disk account is charged and RAM is not.
    #[test]
    #[ignore]
    fn layerwise_activation_disk_tier_readback_byte_exact() {
        let mut gpu = Gpu::init().expect("gpu");
        let id = LayerwiseIdentity {
            history: fx_stamp(),
            request: 93,
            evaluation: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER + 1 },
            query: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER + 1 },
        };
        let n_layers = 2usize;
        let full_pages = 2u64;
        let q_row_bytes = QW * 4; // 1024
        let dir =
            std::env::temp_dir().join(format!("hipfire-activation-disk-{}", std::process::id()));
        // tape_peak = 2048; pin = 1024; activation_ram = 2048 → ram_tapes 1024 < 2048.
        let quotas = LayerwiseQuotas {
            activation_ram: 2048,
            activation_disk: 8192,
            staging: 1024,
            disk_directory: Some(dir.clone()),
            ..LayerwiseQuotas::default()
        };
        let mut act =
            LayerwiseActivation::new(&mut gpu, id, n_layers, full_pages, q_row_bytes, &quotas)
                .expect("disk activation");
        assert!(act.use_disk(), "tape peak forces the disk tier");
        assert_eq!(act.ram_charged_bytes(), 0, "disk tier: no RAM tape charge");

        let mut tiles = Vec::new();
        for ordinal in 0..n_layers {
            let host: Vec<f32> = (0..QW)
                .map(|i| (((i * 7 + ordinal * 53) % 23) as f32) * 0.1 - 0.5)
                .collect();
            let bytes = unsafe {
                std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
            }
            .to_vec();
            let t = gpu.alloc_tensor(&[host.len() * 4], DType::Raw).expect("q alloc");
            gpu.hip.memcpy_htod(&t.buf, &bytes).expect("q h2d");
            let q = NativeCurrentLayerQuery {
                data: t.buf.as_ptr() as *const std::ffi::c_void,
                bytes: host.len() * 4,
                layer: ordinal as u32,
                rows: 1,
                query_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                attention_scale: 0.0,
                scalar: 0,
                row_stride: QW * 4,
                head_stride: HEAD_DIM as usize * 4,
                dimension_stride: 4,
            };
            act.capture_layer(&mut gpu, ordinal, &q).expect("capture disk");
            let mut got = vec![0u8; q_row_bytes];
            act.read_layer(ordinal, FX_FRONTIER, 1, &mut got).expect("read disk");
            assert_eq!(got, bytes, "layer {ordinal} disk Q tape byte-exact");
            tiles.push(t);
        }
        let tape_peak = (n_layers * q_row_bytes) as u64;
        assert_eq!(act.disk_charged_bytes(), tape_peak, "activation_disk charged the tape peak");
        assert!(act.metadata_charged_bytes() > 0);
        assert!(act.envelope_charged_bytes() > 0);
        for t in tiles {
            gpu.free_tensor(t).expect("free q");
        }
        drop(act);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// GPU carrier-level additive proof (design §5.1 E1): attaching the activation
    /// subsystem to a real [`LayerwiseQuery`] carrier and restoring each layer
    /// captures the projected Q into the tapes WITHOUT perturbing the certified ②
    /// restore — the selected resident page and compacted extent are identical to
    /// the no-activation driver transaction (`driver_end_to_end_layerwise_transaction`),
    /// the tape holds the Q byte-exact, and the scorer's GPU Q tile is byte-unchanged
    /// after capture. The activation is a pure host side-copy on the Layerwise path.
    #[test]
    #[ignore]
    fn layerwise_carrier_activation_additive() {
        let mut gpu = Gpu::init().expect("gpu");
        let q8 = fx_layout();
        let kw = FX_KV_HEADS * FX_HEAD_DIM; // 32 (query_heads == kv_heads == 1)
        let q_row_bytes = kw * 4; // 128
        let rows = 1u32; // single decode row

        // Policy + packed store: both canonical pages, both layers (the fx fixture).
        let policy = KvStore::new(&KvStoreConfig {
            block_tokens: FX_BT,
            select_budget: FX_BT,
            sink_blocks: 0,
            recent_blocks: 0,
            ..Default::default()
        })
        .expect("policy");
        policy.register_append(FX_FRONTIER as u32).expect("append");
        let keys_storage = Storage::new_ram(16 << 20).expect("keys storage");
        let keys = RawStore::new(q8.raw_config().clone(), Some(&keys_storage)).expect("keys");
        keys.ensure_blocks(2).expect("ensure blocks");
        for b in 0..2u32 {
            for il in 0..FX_LAYERS {
                keys.write_layer_k_gpu(b * FX_BT, FX_BT, il, &fx_pack(b, il, false)).expect("K");
                keys.write_layer_v_gpu(b * FX_BT, FX_BT, il, &fx_pack(b, il, true)).expect("V");
            }
        }
        // Mean-K sign trick: layer 0 favors page 200, layer 1 page 100 (as certified).
        for il in 0..FX_LAYERS {
            for b in 0..2u32 {
                let sign = if (b == 1) == (il == 0) { 1.0f32 } else { -1.0f32 };
                let tile = vec![sign; FX_BT as usize * kw];
                keys.write_layer_mean_k(b * FX_BT, FX_BT, il, &tile).expect("mean-k");
            }
        }

        // Shared all-+1 native Q tile (one row) — the certified selection input.
        let host = vec![1.0f32; rows as usize * kw];
        let q_bytes = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
        }
        .to_vec();
        let qt = gpu.alloc_tensor(&[host.len() * 4], DType::Raw).expect("q alloc");
        gpu.hip.memcpy_htod(&qt.buf, &q_bytes).expect("q h2d");
        let q_base = NativeCurrentLayerQuery {
            data: qt.buf.as_ptr() as *const std::ffi::c_void,
            bytes: host.len() * 4,
            layer: 0,
            rows,
            query_heads: 1,
            kv_heads: FX_KV_HEADS as u32,
            head_dim: FX_HEAD_DIM as u32,
            attention_scale: 0.0,
            scalar: 0,
            row_stride: kw * 4,
            head_stride: FX_HEAD_DIM * 4,
            dimension_stride: 4,
        };

        let driver = LayerwiseDriver::new(
            &mut gpu,
            &keys,
            &policy,
            &q8,
            fx_backend(),
            fx_capabilities(),
            fx_snapshots(),
            3,
            16 << 20,
        )
        .expect("driver");
        let staging = LayerwiseDescStaging::new(&mut gpu, FX_LAYERS as usize, 3).expect("staging");
        let mut carrier = LayerwiseQuery::new(driver, staging);
        // History-only restore (no suffix write) isolates the capture seam.
        carrier.suffix_slot = None;

        let identity = LayerwiseIdentity {
            history: fx_stamp(),
            request: 94,
            evaluation: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER + rows as u64 },
            query: QueryRange { begin: FX_FRONTIER, end: FX_FRONTIER + rows as u64 },
        };
        let geom = |ordinal: u32| LayerQueryGeometry {
            layer: ordinal,
            query_heads: 1,
            kv_heads: FX_KV_HEADS as u32,
            head_dim: FX_HEAD_DIM as u32,
            attention_scale: 0.0,
        };
        let specs = vec![
            LayerRequestSpec { model_layer: 0, query: geom(0), budget: fx_budget(), mandatory: vec![] },
            LayerRequestSpec { model_layer: 1, query: geom(1), budget: fx_budget(), mandatory: vec![] },
        ];
        carrier.driver.begin_query(identity, specs, FX_BT as u64).expect("begin_query");

        // Attach the activation subsystem (the engine_carrier wiring under test).
        carrier
            .attach_activation(&mut gpu, &LayerwiseQuotas::default(), q_row_bytes)
            .expect("attach_activation");
        assert!(carrier.activation().is_some());
        assert_eq!(carrier.activation().unwrap().n_layers(), FX_LAYERS as usize);

        for ordinal in 0..FX_LAYERS as usize {
            let mut q = q_base;
            q.layer = ordinal as u32;
            let extent = carrier.restore_layer(&mut gpu, ordinal, &q).expect("restore_layer");
            // Identical to the certified no-activation transaction: one selected
            // page (FX_BT tokens), layer 0 → canonical page 1, layer 1 → page 0.
            assert_eq!(extent, FX_BT as u64, "compacted extent unperturbed by capture");
            let page = if ordinal == 0 { 1u32 } else { 0u32 };
            assert_eq!(
                carrier.driver.arena().pool_ref().canonical_pages(ordinal).unwrap(),
                vec![page],
                "selection unperturbed by capture"
            );
            // The tape captured this layer's Q byte-exact.
            let mut got = vec![0u8; q_row_bytes];
            carrier
                .activation()
                .unwrap()
                .read_layer(ordinal, FX_FRONTIER, rows as u64, &mut got)
                .expect("read tape");
            assert_eq!(got, q_bytes, "layer {ordinal} carrier tape byte-exact");
            // Non-destructive: the scorer's GPU Q tile is unchanged after restore.
            let mut after = vec![0u8; q_row_bytes];
            gpu.hip.memcpy_dtoh_at(&mut after, &qt.buf, 0).expect("q d2h");
            assert_eq!(after, q_bytes, "layer {ordinal} GPU Q tile untouched (additive)");
            carrier.attend_complete_layer(ordinal).expect("attend_complete");
        }
        assert_eq!(carrier.driver.phase(), Some(LayerwisePhase::AwaitingAcceptance));

        carrier.driver.end_query();
        gpu.free_tensor(qt).expect("free q");
        carrier.free(&mut gpu).expect("free carrier");
    }

    /// D4a addressing proof (design §3.8/§3.9): the composite [`LayerwiseArena`]
    /// is addressable by the UNCHANGED paged GQA decode tile
    /// (`attention_flash_q8_0_tile_gqa_paged_gfx1100`). The kernel computes one
    /// `tile_off = phys*page_tokens*(n_kv_heads*GQA_SEG)` and reads K at
    /// `k_cache+tile_off`, V at `v_cache+tile_off` — separate bases, same offset.
    /// Encoding `block_table[lp] = 2*(slot*n_layers+layer)`, `k_cache = arena
    /// base`, `v_cache = arena.sub_offset(k_plane_bytes)` resolves K to the
    /// stripe's K plane and V to its V plane (Q8_0 equal K/V strides), byte-for-
    /// byte the core record layout with no repacking. Proven by byte-exact
    /// parity against a legacy contiguous K/V arena holding the identical bytes,
    /// plus the `phys < 0` spill mask (drop-evict composes for free). Production-
    /// admitted shape: gfx1100, head_dim 256, n_heads == n_kv_heads*6, tile 128.
    #[test]
    #[ignore]
    fn composite_arena_paged_attention_addressing_parity() {
        use rdna_compute::kv_slots::KvSlotDesc;

        let mut gpu = Gpu::init().expect("gpu");
        const KVH: usize = 1;
        const NH: usize = 6; // n_heads == n_kv_heads*6 (kernel admission gate)
        const HD: usize = 256;
        const SEG: usize = (HD / 32) * 34; // 272 bytes per (pos, kv head)
        const BT: usize = 128;
        const PLANE: usize = BT * SEG; // 34816
        const NL: usize = 2; // fa layers
        const NP: usize = 3; // logical pages
        const NSLOTS: usize = 4;
        const LAYER: usize = 1; // exercise a non-zero layer offset
        const SEQ: i32 = (NP * BT) as i32; // 384
        const MAXTILES: usize = (SEQ as usize + BT - 1) / BT; // 3

        let q8 = LayerwiseQ8Layout::resolve(NL, KVH, HD, &[0, 1]).expect("q8 layout");
        assert_eq!(q8.k_stride(), SEG);
        assert_eq!(q8.v_stride(), SEG);

        // Deterministic valid-Q8 page bytes: per 34-byte block a finite f16 scale
        // (K=1.0, V=0.5 — distinct so a K/V plane swap is caught) + 32 varying
        // int8 codes keyed by (page, layer, pos, block, byte).
        fn pack(page: u32, layer: u32, is_v: bool) -> Vec<u8> {
            let scale: u16 = if is_v { 0x3800 } else { 0x3C00 };
            let mut v = vec![0u8; BT * SEG];
            for t in 0..BT {
                for j in 0..(SEG / 34) {
                    let off = t * SEG + j * 34;
                    v[off] = (scale & 0xFF) as u8;
                    v[off + 1] = (scale >> 8) as u8;
                    for c in 0..32 {
                        let base = page * 7 + layer * 13 + (t as u32) * 3
                            + (j as u32) * 5 + if is_v { 100 } else { 0 };
                        v[off + 2 + c] = ((base + c as u32) % 251) as u8;
                    }
                }
            }
            v
        }
        fn upload_i32(gpu: &mut Gpu, vals: &[i32]) -> GpuTensor {
            let t = gpu.alloc_tensor(&[vals.len() * 4], DType::Raw).expect("i32 alloc");
            gpu.hip
                .memcpy_htod(&t.buf, unsafe {
                    std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4)
                })
                .expect("i32 h2d");
            t
        }
        fn read_f32(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
            let mut v = vec![0f32; n];
            let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
            gpu.hip.memcpy_dtoh_at(bytes, &t.buf, 0).expect("f32 d2h");
            v
        }

        // Non-trivial logical→physical slot map (slot 1 unused catches stride bugs).
        let map = [2usize, 0, 3];

        // ── composite arena: fill the target layer's stripes ──
        let arena = LayerwiseArena::new(&mut gpu, &q8, NSLOTS).expect("arena");
        let layout = *arena.layout();
        assert_eq!(layout.k_plane_bytes(), PLANE);
        for lp in 0..NP {
            let slot = map[lp];
            let koff = layout.layer_k_offset(slot, LAYER).unwrap();
            let voff = layout.layer_v_offset(slot, LAYER).unwrap();
            gpu.hip
                .memcpy_htod_offset(&arena.arena().buf, koff, &pack(lp as u32, LAYER as u32, false))
                .expect("comp K");
            gpu.hip
                .memcpy_htod_offset(&arena.arena().buf, voff, &pack(lp as u32, LAYER as u32, true))
                .expect("comp V");
        }
        let arena_bytes = layout.arena_bytes();
        let v_view = arena.arena().sub_offset(PLANE, arena_bytes - PLANE);

        // ── legacy contiguous reference arenas (same bytes, compact order) ──
        let ref_k = gpu.alloc_tensor(&[NP * PLANE], DType::Raw).expect("ref K");
        let ref_v = gpu.alloc_tensor(&[NP * PLANE], DType::Raw).expect("ref V");
        for lp in 0..NP {
            gpu.hip
                .memcpy_htod_offset(&ref_k.buf, lp * PLANE, &pack(lp as u32, LAYER as u32, false))
                .expect("ref K");
            gpu.hip
                .memcpy_htod_offset(&ref_v.buf, lp * PLANE, &pack(lp as u32, LAYER as u32, true))
                .expect("ref V");
        }

        // ── shared launch tensors ──
        let mut qh = vec![0f32; NH * HD];
        for (i, v) in qh.iter_mut().enumerate() {
            *v = (((i * 11) % 31) as f32) * 0.04 - 0.6;
        }
        let q = gpu.alloc_tensor(&[qh.len() * 4], DType::Raw).expect("q");
        gpu.hip
            .memcpy_htod(&q.buf, unsafe {
                std::slice::from_raw_parts(qh.as_ptr() as *const u8, qh.len() * 4)
            })
            .expect("q h2d");
        let positions = upload_i32(&mut gpu, &[SEQ - 1]);
        let row_slot = upload_i32(&mut gpu, &[0]);
        let partials = gpu.alloc_tensor(&[NH * MAXTILES * (2 + HD) * 4], DType::Raw).expect("partials");
        let out = gpu.alloc_tensor(&[NH * HD * 4], DType::Raw).expect("out");

        // Composite encodes 2*(slot*NL+LAYER); reference is the compact index.
        let comp_bt: Vec<i32> = (0..NP).map(|lp| (2 * (map[lp] * NL + LAYER)) as i32).collect();
        let ref_bt: Vec<i32> = (0..NP).map(|lp| lp as i32).collect();

        // Run the unchanged paged decode kernel on a (k_cache, v_cache, table) triple.
        let run = |gpu: &mut Gpu, kc: &GpuTensor, vc: &GpuTensor, bt: &[i32]| -> (Vec<f32>, Vec<f32>) {
            let bt_t = upload_i32(gpu, bt);
            let desc = KvSlotDesc {
                block_table: bt_t.buf.as_ptr() as u64,
                legacy_k_base: 0,
                legacy_v_base: 0,
                seq_len: SEQ,
                page_tokens: BT as i32,
            };
            let descs = gpu.alloc_tensor(&[32], DType::Raw).expect("descs");
            gpu.hip
                .memcpy_htod(&descs.buf, unsafe {
                    std::slice::from_raw_parts(&desc as *const KvSlotDesc as *const u8, 32)
                })
                .expect("desc h2d");
            let engaged = gpu
                .attention_flash_q8_0_gqa_decode_paged(
                    &q, kc, vc, &out, &positions, NH, KVH, HD, SEQ as usize, &partials, &descs,
                    &row_slot,
                )
                .expect("paged decode");
            assert!(engaged, "paged GQA tile must engage for the admitted shape");
            let p = read_f32(gpu, &partials, NH * MAXTILES * (2 + HD));
            let o = read_f32(gpu, &out, NH * HD);
            gpu.free_tensor(bt_t).expect("free bt");
            gpu.free_tensor(descs).expect("free descs");
            (p, o)
        };

        let (cp, co) = run(&mut gpu, arena.arena(), &v_view, &comp_bt);
        let (rp, ro) = run(&mut gpu, &ref_k, &ref_v, &ref_bt);

        // Resident tiles compute real mass (Σ exp ≥ 1); byte-exact parity proves
        // the composite addressing reads exactly the legacy bytes.
        for h in 0..NH {
            for t in 0..MAXTILES {
                assert!(cp[(h * MAXTILES + t) * (2 + HD) + 1] >= 1.0, "resident tile mass");
            }
        }
        assert!(
            cp.iter().map(|f| f.to_bits()).eq(rp.iter().map(|f| f.to_bits())),
            "composite vs legacy partials must be byte-exact"
        );
        assert!(
            co.iter().map(|f| f.to_bits()).eq(ro.iter().map(|f| f.to_bits())),
            "composite vs legacy out must be byte-exact"
        );

        // ── spill mask: drop logical page 1 in BOTH layouts → empty partials ──
        let mut comp_spill = comp_bt.clone();
        comp_spill[1] = -1;
        let mut ref_spill = ref_bt.clone();
        ref_spill[1] = -1;
        let (csp, cso) = run(&mut gpu, arena.arena(), &v_view, &comp_spill);
        let (rsp, rso) = run(&mut gpu, &ref_k, &ref_v, &ref_spill);
        for h in 0..NH {
            let idx = (h * MAXTILES + 1) * (2 + HD);
            assert_eq!(csp[idx + 1], 0.0, "spilled tile empty sum");
            assert_eq!(csp[idx], -1e30f32, "spilled tile sentinel max");
        }
        assert!(
            csp.iter().map(|f| f.to_bits()).eq(rsp.iter().map(|f| f.to_bits())),
            "spill partials must be byte-exact"
        );
        assert!(
            cso.iter().map(|f| f.to_bits()).eq(rso.iter().map(|f| f.to_bits())),
            "spill out must be byte-exact"
        );

        gpu.free_tensor(q).expect("free q");
        gpu.free_tensor(positions).expect("free pos");
        gpu.free_tensor(row_slot).expect("free row_slot");
        gpu.free_tensor(partials).expect("free partials");
        gpu.free_tensor(out).expect("free out");
        gpu.free_tensor(ref_k).expect("free ref K");
        gpu.free_tensor(ref_v).expect("free ref V");
        arena.free(&mut gpu).expect("free arena");
    }

    /// Pure-host encoding: the stripe formula `2*(slot*n_layers+layer)`, suffix
    /// append, `-1` padding, and every fail-closed guard. Mirrors the D4a
    /// GPU-verified encoding (compact [2,0,3], layer 1 of 2 → [10,2,14]).
    #[test]
    fn encode_layer_block_table_maps_stripes_and_masks() {
        // compact [2,0,3] + suffix 1, layer 1 of 2, max_pages 4 → all mapped.
        let bt = encode_layer_block_table(&[2, 0, 3], Some(1), 1, 2, 4).expect("encode");
        assert_eq!(bt, vec![10, 2, 14, 6]);
        // No suffix → trailing logical pages padded -1.
        let bt = encode_layer_block_table(&[2, 0], None, 1, 2, 4).expect("encode");
        assert_eq!(bt, vec![10, 2, -1, -1]);
        // Layer 0 of 2: stripe = 2*(slot*2) = 4*slot.
        let bt = encode_layer_block_table(&[3, 1], None, 0, 2, 3).expect("encode");
        assert_eq!(bt, vec![12, 4, -1]);
        // Empty compact + no suffix → all -1 (fully masked layer).
        let bt = encode_layer_block_table(&[], None, 0, 2, 2).expect("encode");
        assert_eq!(bt, vec![-1, -1]);
        // Single fa layer: stripe = 2*slot (n_layers 1).
        let bt = encode_layer_block_table(&[5], Some(0), 0, 1, 2).expect("encode");
        assert_eq!(bt, vec![10, 0]);

        // ── fail-closed guards ──
        assert!(encode_layer_block_table(&[0], None, 0, 0, 4).is_err(), "n_layers 0");
        assert!(encode_layer_block_table(&[0], None, 2, 2, 4).is_err(), "layer >= n_layers");
        assert!(
            encode_layer_block_table(&[0, 1, 2], Some(3), 0, 2, 3).is_err(),
            "mapped > max_pages"
        );
        // u32::MAX slot × n_layers 2 × 2 overflows i32 → must fail, not wrap
        // negative (a wrapped index would be read as a dropped page).
        assert!(
            encode_layer_block_table(&[u32::MAX], None, 0, 2, 4).is_err(),
            "stripe index overflow"
        );
    }

    /// The staging uploads each layer's encoded table + a descriptor pointing at
    /// the STABLE table address; re-updating a layer keeps the pointer fixed.
    /// Also verifies the shifted V base view geometry and the fail-closed guards.
    /// (D4a already proved the kernel reads these bytes byte-exact; this isolates
    /// the staging primitive's upload/addressing contract.)
    #[test]
    #[ignore]
    fn layerwise_desc_staging_upload_readback() {
        let mut gpu = Gpu::init().expect("gpu");
        const KVH: usize = 1;
        const HD: usize = 256;
        const NL: usize = 2;
        const NSLOTS: usize = 4;
        const MAXP: usize = 4;
        let q8 = LayerwiseQ8Layout::resolve(NL, KVH, HD, &[0, 1]).expect("q8 layout");
        let arena = LayerwiseArena::new(&mut gpu, &q8, NSLOTS).expect("arena");
        let staging = LayerwiseDescStaging::new(&mut gpu, NL, MAXP).expect("staging");
        assert_eq!(staging.n_layers(), NL);
        assert_eq!(staging.max_pages(), MAXP);

        fn read_i32(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> Vec<i32> {
            let mut v = vec![0i32; n];
            let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
            gpu.hip.memcpy_dtoh_at(bytes, &t.buf, 0).expect("i32 d2h");
            v
        }
        fn read_desc(gpu: &mut Gpu, t: &GpuTensor) -> KvSlotDesc {
            let mut d = KvSlotDesc {
                block_table: 0,
                legacy_k_base: 0,
                legacy_v_base: 0,
                seq_len: 0,
                page_tokens: 0,
            };
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    &mut d as *mut KvSlotDesc as *mut u8,
                    std::mem::size_of::<KvSlotDesc>(),
                )
            };
            gpu.hip.memcpy_dtoh_at(bytes, &t.buf, 0).expect("desc d2h");
            d
        }

        // Per-layer compact views + suffix; seq_len = mapped pages × 128.
        let compact = [vec![2u32, 0, 3], vec![1u32, 3]];
        let suffix = [Some(1u32), None];
        let seq = [4 * 128i32, 2 * 128i32];
        for layer in 0..NL {
            staging
                .update_layer(&mut gpu, layer, &compact[layer], suffix[layer], seq[layer])
                .expect("update");
            let want =
                encode_layer_block_table(&compact[layer], suffix[layer], layer, NL, MAXP)
                    .expect("encode");
            assert_eq!(
                read_i32(&mut gpu, staging.block_table(layer).expect("bt"), MAXP),
                want,
                "table layer {layer}"
            );
            let d = read_desc(&mut gpu, staging.descs(layer).expect("desc"));
            assert_eq!(
                d.block_table,
                staging.block_table(layer).expect("bt").buf.as_ptr() as u64,
                "stable ptr layer {layer}"
            );
            assert_eq!(d.seq_len, seq[layer]);
            assert_eq!(d.page_tokens, 128);
            assert_eq!(d.legacy_k_base, 0);
            assert_eq!(d.legacy_v_base, 0);
            // Compacted causal bound = [seq_len - 1] (positions[] is the
            // kernel's authoritative bound, distinct from the RoPE position).
            assert_eq!(
                read_i32(&mut gpu, staging.positions(layer).expect("pos"), 1),
                vec![seq[layer] - 1],
                "positions layer {layer}"
            );
        }

        // Re-update layer 0 with a different view: pointer fixed, content changes.
        let ptr0 = staging.block_table(0).expect("bt").buf.as_ptr() as u64;
        staging.update_layer(&mut gpu, 0, &[0], Some(2), 2 * 128).expect("re-update");
        assert_eq!(
            staging.block_table(0).expect("bt").buf.as_ptr() as u64,
            ptr0,
            "stable address across re-upload"
        );
        let d0 = read_desc(&mut gpu, staging.descs(0).expect("desc"));
        assert_eq!(d0.block_table, ptr0);
        assert_eq!(d0.seq_len, 2 * 128);
        assert_eq!(
            read_i32(&mut gpu, staging.block_table(0).expect("bt"), MAXP),
            encode_layer_block_table(&[0], Some(2), 0, NL, MAXP).expect("encode")
        );
        assert_eq!(
            read_i32(&mut gpu, staging.positions(0).expect("pos"), 1),
            vec![2 * 128 - 1],
            "positions re-uploaded in place"
        );

        // Shifted V base view: base + k_plane_bytes, length arena_bytes - k_plane.
        let k_plane = arena.layout().k_plane_bytes();
        let arena_bytes = arena.layout().arena_bytes();
        let v_view = arena.v_cache_view().expect("v view");
        assert_eq!(
            v_view.buf.as_ptr() as u64,
            arena.arena().buf.as_ptr() as u64 + k_plane as u64
        );
        assert_eq!(v_view.buf.size(), arena_bytes - k_plane);

        // ── fail-closed guards (all reject BEFORE any upload) ──
        assert!(staging.update_layer(&mut gpu, NL, &[0], None, 128).is_err(), "layer range");
        assert!(staging.update_layer(&mut gpu, 0, &[0], None, -1).is_err(), "negative seq_len");
        // seq_len needs 3 tiles but only 1 mapped page.
        assert!(staging.update_layer(&mut gpu, 0, &[0], None, 3 * 128).is_err(), "tiles > mapped");
        assert!(staging.descs(NL).is_err(), "desc accessor range");
        assert!(staging.block_table(NL).is_err(), "table accessor range");
        assert!(staging.positions(NL).is_err(), "positions accessor range");
        assert!(LayerwiseDescStaging::new(&mut gpu, 0, MAXP).is_err(), "n_layers 0");
        assert!(LayerwiseDescStaging::new(&mut gpu, NL, 0).is_err(), "max_pages 0");

        staging.free_gpu(&mut gpu);
        arena.free(&mut gpu).expect("free arena");
    }

    /// Shared GPU fixture for the Layerwise path-parity proofs (D4b-3). Path C
    /// (the attend seam) and Path A (the whole non-escha paged forward) both
    /// compare against the UNCHANGED legacy paged GQA decode over the identical
    /// compacted working set, so they share the geometry, the deterministic
    /// valid-Q8 page packer, the H2D/D2H helpers, the micro dense config, the
    /// Q8 `KvCache` literal and the reference block-table descriptor.
    /// Production-admitted shape: gfx1100, head_dim 256, n_heads ==
    /// n_kv_heads*6, attention tile 128 (same as the D4a addressing proof).
    mod path_fixture {
        use super::*;
        use crate::qwen35::config::{LayerType, Qwen35Config};
        use crate::qwen35::weights::FullAttnLayerWeights;
        use hipfire_runtime::llama::{KvCache, VMode, WeightTensor};

        pub const KVH: usize = 1;
        pub const NH: usize = 6; // n_heads == n_kv_heads*6 (kernel admission gate)
        pub const HD: usize = 256;
        pub const SEG: usize = (HD / 32) * 34; // 272 bytes per (pos, kv head)
        pub const BT: usize = 128;
        pub const PLANE: usize = BT * SEG; // 34816 (one page plane)
        pub const NL: usize = 2; // fa layers
        pub const NSLOTS: usize = 4;
        pub const MAXP: usize = 3; // logical pages per layer (history + suffix + pad)
        pub const FRONTIER: u64 = 256; // frozen history extent (2 pages)
        pub const EXTENT: i32 = (BT + 1) as i32; // 129: one history page + 1 suffix token
        pub const CTX: usize = MAXP * BT; // 384 (static page-rounded upper bound)
        pub const MAXTILES: usize = (CTX + BT - 1) / BT; // 3
        pub const QDIM: usize = NH * HD; // 1536
        pub const KVDIM: usize = KVH * HD; // 256
        pub const DIM: usize = 256; // hidden == FFN intermediate (HFQ4 k%256==0)

        /// Deterministic valid-Q8 page bytes (mirrors the D4a proof): per 34-byte
        /// block a finite f16 scale (K=1.0, V=0.5) + 32 varying int8 codes keyed
        /// by (page, layer, pos, block, byte).
        pub fn pack(page: u32, layer: u32, is_v: bool) -> Vec<u8> {
            let scale: u16 = if is_v { 0x3800 } else { 0x3C00 };
            let mut v = vec![0u8; BT * SEG];
            for t in 0..BT {
                for j in 0..(SEG / 34) {
                    let off = t * SEG + j * 34;
                    v[off] = (scale & 0xFF) as u8;
                    v[off + 1] = (scale >> 8) as u8;
                    for c in 0..32 {
                        let base = page * 7 + layer * 13 + (t as u32) * 3
                            + (j as u32) * 5 + if is_v { 100 } else { 0 };
                        v[off + 2 + c] = ((base + c as u32) % 251) as u8;
                    }
                }
            }
            v
        }

        pub fn upload_i32(gpu: &mut Gpu, vals: &[i32]) -> GpuTensor {
            let t = gpu.alloc_tensor(&[vals.len() * 4], DType::Raw).expect("i32 alloc");
            gpu.hip
                .memcpy_htod(&t.buf, unsafe {
                    std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4)
                })
                .expect("i32 h2d");
            t
        }
        pub fn upload_f32(gpu: &mut Gpu, vals: &[f32]) -> GpuTensor {
            let t = gpu.alloc_tensor(&[vals.len()], DType::F32).expect("f32 alloc");
            put_f32(gpu, &t, vals);
            t
        }
        pub fn put_f32(gpu: &mut Gpu, t: &GpuTensor, vals: &[f32]) {
            gpu.hip
                .memcpy_htod(&t.buf, unsafe {
                    std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * 4)
                })
                .expect("f32 h2d");
        }
        pub fn read_f32(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
            let mut v = vec![0f32; n];
            let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
            gpu.hip.memcpy_dtoh_at(bytes, &t.buf, 0).expect("f32 d2h");
            v
        }
        pub fn read_i32(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> Vec<i32> {
            let mut v = vec![0i32; n];
            let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
            gpu.hip.memcpy_dtoh_at(bytes, &t.buf, 0).expect("i32 d2h");
            v
        }
        pub fn read_bytes(gpu: &mut Gpu, t: &GpuTensor, n: usize) -> Vec<u8> {
            let mut v = vec![0u8; n];
            gpu.hip.memcpy_dtoh_at(&mut v, &t.buf, 0).expect("bytes d2h");
            v
        }
        pub fn read_desc(gpu: &mut Gpu, t: &GpuTensor) -> KvSlotDesc {
            let mut d = KvSlotDesc {
                block_table: 0,
                legacy_k_base: 0,
                legacy_v_base: 0,
                seq_len: 0,
                page_tokens: 0,
            };
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    &mut d as *mut KvSlotDesc as *mut u8,
                    std::mem::size_of::<KvSlotDesc>(),
                )
            };
            gpu.hip.memcpy_dtoh_at(bytes, &t.buf, 0).expect("desc d2h");
            d
        }

        /// The micro dense model both paths run: `NL` FullAttention layers,
        /// head_dim 256 / 6 query heads / 1 KV head, FFN intermediate == DIM.
        pub fn micro_config() -> Qwen35Config {
            Qwen35Config {
                dim: DIM,
                n_layers: NL,
                vocab_size: 8,
                norm_eps: 1e-6,
                eos_token: 0,
                n_heads: NH,
                n_kv_heads: KVH,
                head_dim: HD,
                rope_theta: 1e7,
                partial_rotary_factor: 0.25, // n_rot = 64
                is_vl_text: false,
                mrope_interleaved: false,
                mrope_section: [0, 0, 0],
                linear_num_key_heads: 1,
                linear_num_value_heads: 1,
                linear_key_head_dim: 32,
                linear_value_head_dim: 32,
                conv_kernel_dim: 4,
                hidden_dim: DIM,
                num_experts: 0,
                num_experts_per_tok: 0,
                moe_intermediate_size: 0,
                shared_expert_intermediate_size: 0,
                has_shared_expert: false,
                norm_topk_prob: false,
                layer_types: vec![LayerType::FullAttention; NL],
                paged_experts: false,
                vram_budget_bytes: u64::MAX,
                reap_keep: None,
            }
        }

        /// The Q8 paged `KvCache` literal: `k_gpu`/`v_gpu` are model-layer-order
        /// arena views (composite bases for Layerwise, contiguous reference
        /// arenas for the legacy run). No scales, no compaction, no boundary.
        pub fn q8_kv_cache(k_gpu: Vec<GpuTensor>, v_gpu: Vec<GpuTensor>) -> KvCache {
            KvCache {
                k_gpu,
                v_gpu,
                k_scales: Vec::new(),
                v_scales: Vec::new(),
                kv_dim: KVDIM,
                max_seq: NSLOTS * BT,
                physical_cap: NSLOTS * BT,
                n_kv_heads: KVH,
                head_dim: HD,
                quantized: true,
                quant_q8: true,
                quant_int8: false,
                quant_hfq4: false,
                quant_asym4: false,
                quant_asym3: false,
                quant_asym2: false,
                boundary_layers: 0,
                givens_cos: None,
                givens_sin: None,
                quant_fwht: false,
                quant_bf16: false,
                quant_fp8: false,
                v_mode: VMode::Q8,
                layer_is_boundary: Vec::new(),
                compact_offset: 0,
            }
        }

        /// One-row reference descriptor over `block_table`: compacted extent
        /// (one history page + the suffix token), 128-token pages.
        pub fn upload_ref_descs(gpu: &mut Gpu, block_table: &GpuTensor) -> GpuTensor {
            let d = KvSlotDesc {
                block_table: block_table.buf.as_ptr() as u64,
                legacy_k_base: 0,
                legacy_v_base: 0,
                seq_len: EXTENT,
                page_tokens: BT as i32,
            };
            let t = gpu.alloc_tensor(&[32], DType::Raw).expect("ref descs");
            gpu.hip
                .memcpy_htod(&t.buf, unsafe {
                    std::slice::from_raw_parts(&d as *const KvSlotDesc as *const u8, 32)
                })
                .expect("ref desc h2d");
            t
        }

        /// Minimal F32 [1,1] weight: the attend seam never reads projection
        /// content (proj/FFN bypassed) and n_heads=6 + non-MQ4 wo short-circuit
        /// the fused epilogue, so wo is only a dispatch placeholder.
        pub fn dummy_weight(gpu: &mut Gpu) -> WeightTensor {
            WeightTensor {
                buf: gpu.zeros(&[1], DType::F32).expect("dummy weight"),
                gpu_dtype: DType::F32,
                m: 1,
                k: 1,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            }
        }
        /// Path C's layer: real norms, placeholder projections (attend only).
        pub fn dummy_fa_layer(gpu: &mut Gpu) -> FullAttnLayerWeights {
            FullAttnLayerWeights {
                attn_norm: upload_f32(gpu, &vec![1.0; DIM]),
                wq: dummy_weight(gpu),
                wk: dummy_weight(gpu),
                wv: dummy_weight(gpu),
                wo: dummy_weight(gpu),
                q_norm: upload_f32(gpu, &vec![1.0; HD]),
                k_norm: upload_f32(gpu, &vec![1.0; HD]),
                ffn_norm: upload_f32(gpu, &vec![1.0; DIM]),
                w_gate: dummy_weight(gpu),
                w_up: dummy_weight(gpu),
                w_down: dummy_weight(gpu),
                biases: None,
                escha: None,
            }
        }

        /// Deterministic HFQ4-G256 weight bytes: per 256-element group an F32
        /// scale + F32 zero + 128 packed 4-bit codes (136 B/group). Same packer
        /// the rdna-compute GEMV benches use, so the values are well-scaled.
        pub fn hfq4g256_bytes(m: usize, k: usize, seed: u8) -> Vec<u8> {
            assert_eq!(k % 256, 0, "HFQ4-G256 needs k % 256 == 0");
            let gpr = k / 256;
            let bpr = gpr * 136;
            let mut out = vec![0u8; m * bpr];
            let mix = |x: u64| {
                let h = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((h ^ (h >> 33)).wrapping_mul(0xff51afd7ed558ccd)) ^ (h >> 28)
            };
            let s0 = seed as u64;
            for row in 0..m {
                for g in 0..gpr {
                    let off = row * bpr + g * 136;
                    let r1 = mix(s0 ^ ((row as u64) << 16) ^ (g as u64));
                    let r2 = mix(s0 ^ ((row as u64) * 7 + g as u64));
                    let scale = 0.01 + (((r1 as u32) % 4001) as f32) * 1e-5;
                    let zero = (((r2 as u32) % 1500) as f32) * 1e-4 - 0.075;
                    out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
                    out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
                    for byte_i in 0..128 {
                        let r =
                            mix(s0 ^ ((row as u64) << 24) ^ ((g as u64) << 12) ^ (byte_i as u64));
                        out[off + 8 + byte_i] = (r & 0xff) as u8;
                    }
                }
            }
            out
        }
        pub fn dense_weight(gpu: &mut Gpu, m: usize, k: usize, seed: u8) -> WeightTensor {
            let bytes = hfq4g256_bytes(m, k, seed);
            let buf = gpu.alloc_tensor(&[bytes.len()], DType::Raw).expect("hfq alloc");
            gpu.hip.memcpy_htod(&buf.buf, &bytes).expect("hfq h2d");
            WeightTensor {
                buf,
                gpu_dtype: DType::HFQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            }
        }
        /// Path A's layer: real HFQ4-G256 projections + FFN, so the certified
        /// `execute_gated_attention` (project + attend + output) and
        /// `dense_ffn` executors actually run. `wq` emits the interleaved
        /// [q, gate] pair (2*QDIM rows); `wo` reads QDIM back to DIM.
        pub fn dense_fa_layer(gpu: &mut Gpu, seed: u8) -> FullAttnLayerWeights {
            FullAttnLayerWeights {
                attn_norm: upload_f32(gpu, &vec![1.0; DIM]),
                wq: dense_weight(gpu, 2 * QDIM, DIM, seed),
                wk: dense_weight(gpu, KVDIM, DIM, seed.wrapping_add(1)),
                wv: dense_weight(gpu, KVDIM, DIM, seed.wrapping_add(2)),
                wo: dense_weight(gpu, DIM, QDIM, seed.wrapping_add(3)),
                q_norm: upload_f32(gpu, &vec![1.0; HD]),
                k_norm: upload_f32(gpu, &vec![1.0; HD]),
                ffn_norm: upload_f32(gpu, &vec![1.0; DIM]),
                w_gate: dense_weight(gpu, DIM, DIM, seed.wrapping_add(4)),
                w_up: dense_weight(gpu, DIM, DIM, seed.wrapping_add(5)),
                w_down: dense_weight(gpu, DIM, DIM, seed.wrapping_add(6)),
                biases: None,
                escha: None,
            }
        }
    }

    /// D4b-3 Path C end-to-end GPU proof (design §3.9d). A test-constructed
    /// [`LayerwiseQuery`] carrier drives the REAL `run_attend_layerwise` seam
    /// (`qwen35/forward.rs`) through `superop::run_layer_program` for a 2-FA-layer
    /// micro dense model, and the restored compacted working set's attention
    /// output is byte-exact vs the UNCHANGED legacy paged GQA decode kernel over a
    /// contiguous arena holding the identical bytes. Exercises the whole Path C
    /// chain: post-norm Q → GPU score → select → COW restore → per-layer
    /// block-table / descriptor / positions staging → paged KV write + attend →
    /// sigmoid gate. The projection / FFN super-ops are unchanged legacy and are
    /// bypassed (scratch pre-filled, one-op ATTEND_FULL program); the attend op is
    /// the D4b-3 seam. Production-admitted shape: gfx1100, head_dim 256,
    /// n_heads == n_kv_heads*6, tile 128 (same as the D4a addressing proof).
    ///
    /// Non-circular reference: the historical page is reconstructed independently
    /// from `pack(selected, layer)` (catches a wrong restore), while the suffix
    /// page is the D2H of the composite stripe the attend's legacy paged writer
    /// produced (the write is shared unchanged code, not the layerwise seam). The
    /// composite historical stripe is ALSO asserted == `pack(selected, layer)`
    /// directly. Deterministic selection: an all-+1 Q (post-rmsnorm ≈ +1) against
    /// the mean-K sign trick (layer 0 favors canonical page 1, layer 1 page 0).
    #[test]
    #[ignore]
    fn layerwise_path_c_attend_matches_legacy_paged() {
        use crate::qwen35::forward::{Qwen35Bindings, Qwen35Scratch};
        use crate::qwen35::weights::{DeltaNetState, LayerWeights, StateQuant};
        use hipfire_dispatch::context::DispatchCtx;
        use hipfire_dispatch::pipeline::superop::{
            self, OpBinding, OpFlavor, SuperOp, SuperOpKind, WeightSlot,
        };
        // Geometry, page packer, H2D/D2H helpers, micro config, Q8 KvCache and
        // the dummy-projection layer builder (shared with the Path A proof).
        use path_fixture::*;

        let mut gpu = Gpu::init().expect("gpu");
        if gpu.arch != "gfx1100" {
            // The paged GQA decode tile and the ported scorer admit gfx1100 only.
            return;
        }

        // ── statistics + packed store: both canonical pages, both layers ──
        let q8 = LayerwiseQ8Layout::resolve(NL, KVH, HD, &[0, 1]).expect("q8 layout");
        assert_eq!(q8.k_stride(), SEG);
        assert_eq!(q8.v_stride(), SEG);
        let record_bytes = q8.payload_layout().expect("manifest").record_bytes();
        let budget = || MemoryBudget {
            selected_tokens: BT as u64,
            reserve_tokens: BT as u64,
            device_tokens: 2 * BT as u64,
            host_payload_bytes: 2 * record_bytes, // both sealed canonical pages
        };
        let policy = KvStore::new(&KvStoreConfig {
            block_tokens: BT as u32,
            select_budget: BT as u32,
            sink_blocks: 0,
            recent_blocks: 0,
            ..Default::default()
        })
        .expect("policy");
        policy.register_append(FRONTIER as u32).expect("append");
        let keys_storage = Storage::new_ram(16 << 20).expect("keys storage");
        let keys = RawStore::new(q8.raw_config().clone(), Some(&keys_storage)).expect("keys");
        keys.ensure_blocks(2).expect("ensure blocks");
        for b in 0..2u32 {
            for il in 0..NL {
                keys.write_layer_k_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, false))
                    .expect("feed K");
                keys.write_layer_v_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, true))
                    .expect("feed V");
            }
        }
        // Mean-K sign trick: layer 0 favors canonical page 1, layer 1 page 0.
        for il in 0..NL {
            for b in 0..2u32 {
                let sign = if (b == 1) == (il == 0) { 1.0f32 } else { -1.0f32 };
                let tile = vec![sign; BT * KVDIM];
                keys.write_layer_mean_k(b * BT as u32, BT as u32, il as u32, &tile).expect("mean-k");
            }
        }

        // ── driver + carrier; reserve the suffix slot, extract OWNED arena views ──
        let mut driver = LayerwiseDriver::new(
            &mut gpu,
            &keys,
            &policy,
            &q8,
            fx_backend(),
            fx_capabilities(),
            fx_snapshots(),
            NSLOTS,
            16 << 20,
        )
        .expect("driver");
        let suffix_phys = driver.arena_mut().pool().alloc_slot().expect("suffix slot").phys;
        // Non-owning aliases of the composite arena base / shifted V base. Both
        // layers share the same base (the per-layer block table differentiates
        // the stripe). Extracted BEFORE the driver moves into the carrier, so
        // kv_cache does not borrow the carrier. NEVER freed (arena owner frees).
        let arena_numel = driver.arena().arena().numel();
        let k0 = driver.arena().arena().sub_offset(0, arena_numel);
        let k1 = driver.arena().arena().sub_offset(0, arena_numel);
        let v0 = driver.arena().v_cache_view().expect("v view 0");
        let v1 = driver.arena().v_cache_view().expect("v view 1");
        let staging = LayerwiseDescStaging::new(&mut gpu, NL, MAXP).expect("staging");
        let mut carrier = LayerwiseQuery::new(driver, staging);
        carrier.suffix_slot = Some(suffix_phys);

        let identity = LayerwiseIdentity {
            history: fx_stamp(),
            request: 77,
            evaluation: QueryRange { begin: FRONTIER, end: FRONTIER + 1 },
            query: QueryRange { begin: FRONTIER, end: FRONTIER + 1 },
        };
        let geom = |ordinal: u32| LayerQueryGeometry {
            layer: ordinal,
            query_heads: NH as u32,
            kv_heads: KVH as u32,
            head_dim: HD as u32,
            attention_scale: 0.0,
        };
        let specs = vec![
            LayerRequestSpec { model_layer: 0, query: geom(0), budget: budget(), mandatory: vec![] },
            LayerRequestSpec { model_layer: 1, query: geom(1), budget: budget(), mandatory: vec![] },
        ];
        carrier.driver.begin_query(identity, specs, BT as u64).expect("begin_query");
        assert_eq!(carrier.driver.phase(), Some(LayerwisePhase::BetweenLayers));

        // ── micro dense model: 2 FullAttention layers, dummy projections ──
        let config = micro_config();
        let s = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, CTX).expect("scratch");
        let dn_state =
            DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::Q8).expect("dn state");
        let ctx = DispatchCtx::new(&gpu);
        let kv_cache = q8_kv_cache(vec![k0, k1], vec![v0, v1]);
        let layers = vec![
            LayerWeights::FullAttn(dummy_fa_layer(&mut gpu)),
            LayerWeights::FullAttn(dummy_fa_layer(&mut gpu)),
        ];
        // One-op ATTEND_FULL program (WeightSlot(0) == q35_op::ATTEND_FULL).
        let program = vec![SuperOp {
            kind: SuperOpKind::Attend,
            binding: OpBinding {
                key: None,
                weights: vec![WeightSlot(0)],
                scratch: Vec::new(),
                flavor: OpFlavor::None,
            },
        }];

        // ── legacy contiguous reference arena + shared launch tensors ──
        let ref_k = gpu.alloc_tensor(&[2 * PLANE], DType::Raw).expect("ref K");
        let ref_v = gpu.alloc_tensor(&[2 * PLANE], DType::Raw).expect("ref V");
        // F32 [QDIM]: sigmoid_mul_f32 sizes from out.numel().
        let ref_out = gpu.alloc_tensor(&[QDIM], DType::F32).expect("ref out");
        let ref_partials =
            gpu.alloc_tensor(&[NH * MAXTILES * (2 + HD) * 4], DType::Raw).expect("ref partials");
        let ref_positions = upload_i32(&mut gpu, &[EXTENT - 1]); // [128] compacted bound
        let ref_row_slot = upload_i32(&mut gpu, &[0]);
        let ref_bt = upload_i32(&mut gpu, &[0, 1, -1]); // history, suffix, masked pad
        let ref_descs = upload_ref_descs(&mut gpu, &ref_bt);

        // ── per-layer Path C forward + byte-exact legacy parity ──
        for l in 0..NL {
            // Pre-fill the decode row the attend seam consumes. Q part = +1 →
            // post-rmsnorm ≈ +1 → deterministic mean-K selection. Gate part is
            // layer-specific so a gate mixup is caught by the sigmoid parity.
            let mut qg = vec![0f32; QDIM * 2];
            for h in 0..NH {
                for d in 0..HD {
                    qg[h * 2 * HD + d] = 1.0;
                    qg[h * 2 * HD + HD + d] = (((h * HD + d + l * 7) % 17) as f32) * 0.25 - 2.0;
                }
            }
            put_f32(&mut gpu, &s.fa_q_full, &qg);
            let kh: Vec<f32> = (0..KVDIM).map(|d| (((d + l * 5) % 11) as f32) * 0.1 - 0.5).collect();
            put_f32(&mut gpu, &s.fa_k, &kh);
            let vh: Vec<f32> = (0..KVDIM).map(|d| (((d * 3 + l * 7) % 13) as f32) * 0.1 - 0.6).collect();
            put_f32(&mut gpu, &s.fa_v, &vh);
            // Real decode position (RoPE); the compacted address comes from the
            // staged positions, not pos_buf (compact_offset == 0).
            gpu.hip
                .memcpy_htod(&s.pos_buf, &(FRONTIER as i32).to_ne_bytes())
                .expect("pos h2d");

            // Drive the REAL run_attend → run_attend_layerwise seam. The binding
            // borrows the carrier mutably; the block scope drops it before the
            // carrier is inspected below.
            {
                let mut bind = Qwen35Bindings {
                    layer: &layers[l],
                    s: &s,
                    config: &config,
                    kv_cache: &kv_cache,
                    dn_state: &dn_state,
                    pos: FRONTIER as usize,
                    layer_idx: l,
                    delta_layer_idx: 0,
                    k_dim: 32,
                    v_dim: 32,
                    n_v_heads: 1,
                    hd: 32,
                    precomputed_attn_x_rot: false,
                    fa_output_prerotated: false,
                    defer_routed_combine: false,
                    paged: None,
                    kvmem_taps: None,
                    row_slot: None,
                    fa_layer_idx: l,
                    layerwise: Some(&mut carrier),
                };
                superop::run_layer_program(&mut gpu, &ctx, &program, &mut bind)
                    .expect("layerwise attend");
            }

            // Selection + COW restore: the layer picked the expected cold page.
            let sel = if l == 0 { 1u32 } else { 0u32 }; // canonical page index
            assert_eq!(
                carrier.driver.arena().pool_ref().canonical_pages(l).unwrap(),
                vec![sel],
                "layer {l} selection"
            );
            let hist_phys = carrier.driver.arena().pool_ref().compact_slots(l).unwrap()[0];
            assert_ne!(hist_phys, suffix_phys, "history vs suffix slot");
            // Independent restore check: the composite historical stripe holds
            // exactly the selected canonical page's packed bytes.
            let (hk, hv) =
                carrier.driver.arena().stripe_views(hist_phys as usize, l).expect("hist stripe");
            assert_eq!(
                read_bytes(&mut gpu, &hk, PLANE),
                pack(sel, l as u32, false),
                "layer {l} restored K bytes"
            );
            assert_eq!(
                read_bytes(&mut gpu, &hv, PLANE),
                pack(sel, l as u32, true),
                "layer {l} restored V bytes"
            );

            // Suffix stripe the attend's legacy paged writer produced → ref page 1.
            let (sk, sv) =
                carrier.driver.arena().stripe_views(suffix_phys as usize, l).expect("suffix stripe");
            let suffix_k = read_bytes(&mut gpu, &sk, PLANE);
            let suffix_v = read_bytes(&mut gpu, &sv, PLANE);
            gpu.hip.memcpy_htod_offset(&ref_k.buf, 0, &pack(sel, l as u32, false)).expect("ref K0");
            gpu.hip.memcpy_htod_offset(&ref_k.buf, PLANE, &suffix_k).expect("ref K1");
            gpu.hip.memcpy_htod_offset(&ref_v.buf, 0, &pack(sel, l as u32, true)).expect("ref V0");
            gpu.hip.memcpy_htod_offset(&ref_v.buf, PLANE, &suffix_v).expect("ref V1");

            // Legacy paged decode over the identical compacted working set. q/gate
            // are the attend's own post-RoPE fa_q / deinterleaved fa_gate (read
            // only here); the kernel writes ref_out, then the same sigmoid gate.
            let engaged = gpu
                .attention_flash_q8_0_gqa_decode_paged(
                    &s.fa_q, &ref_k, &ref_v, &ref_out, &ref_positions, NH, KVH, HD, CTX,
                    &ref_partials, &ref_descs, &ref_row_slot,
                )
                .expect("legacy paged decode");
            assert!(engaged, "legacy paged GQA tile must engage");
            gpu.sigmoid_mul_f32(&ref_out, &s.fa_gate).expect("ref sigmoid gate");
            let lw_out = read_f32(&mut gpu, &s.fa_attn_out, QDIM);
            let ro = read_f32(&mut gpu, &ref_out, QDIM);
            assert!(
                ro.iter().map(|f| f.to_bits()).eq(lw_out.iter().map(|f| f.to_bits())),
                "layer {l} Path C attention must be byte-exact vs legacy paged"
            );

            // Staging contract: compacted causal bound = extent-1, desc.seq_len =
            // extent (visible_tokens 128 + suffix 1), page_tokens 128.
            assert_eq!(
                read_i32(&mut gpu, carrier.staging.positions(l).unwrap(), 1),
                vec![EXTENT - 1],
                "layer {l} positions"
            );
            let d = read_desc(&mut gpu, carrier.staging.descs(l).unwrap());
            assert_eq!(d.seq_len, EXTENT, "layer {l} desc seq_len");
            assert_eq!(d.page_tokens, BT as i32, "layer {l} desc page_tokens");
        }

        // Both layers restored + attended → the transaction awaits acceptance.
        assert_eq!(carrier.driver.phase(), Some(LayerwisePhase::AwaitingAcceptance));
        let work = carrier.driver.work().expect("work");
        assert_eq!(work.projected_rows, 2, "2 layers × 1 evaluation row");
        assert_eq!(work.attended_rows, 2);
        assert_eq!(work.completed_layers, 2);

        // ── teardown (kv_cache views are non-owning aliases — never freed) ──
        carrier.free(&mut gpu).expect("free carrier");
        s.free_gpu(&mut gpu).expect("free scratch");
        dn_state.free_gpu(&mut gpu);
        for layer in layers {
            layer.free_gpu(&mut gpu);
        }
        gpu.free_tensor(ref_k).expect("free ref K");
        gpu.free_tensor(ref_v).expect("free ref V");
        gpu.free_tensor(ref_out).expect("free ref out");
        gpu.free_tensor(ref_partials).expect("free ref partials");
        gpu.free_tensor(ref_positions).expect("free ref pos");
        gpu.free_tensor(ref_row_slot).expect("free ref row_slot");
        gpu.free_tensor(ref_bt).expect("free ref bt");
        gpu.free_tensor(ref_descs).expect("free ref descs");
    }

    /// D4b-3 Path A end-to-end GPU proof (design §3.9b/§3.9c). The WHOLE
    /// non-escha paged decode forward (`forward_scratch_layers_paged` →
    /// `forward_scratch_layers_paged_layerwise`) runs a 2-FA-layer micro dense
    /// model with REAL HFQ4-G256 projections + FFN on the Layerwise carrier, and
    /// the final residual stream is byte-exact vs the UNCHANGED legacy Path A
    /// forward over contiguous reference arenas holding the identical compacted
    /// working set (each layer's selected history page + the suffix page the
    /// paged writer produced).
    ///
    /// Both runs pass the SAME statistics tap, so both take the unfused Q/K prep
    /// route: the comparison isolates the Layerwise seam (per-layer binding
    /// rebuild + score/restore + composite addressing), not the fused-vs-unfused
    /// prep equivalence (certified separately). Projection and FFN stay on the
    /// certified `execute_gated_attention` / `dense_ffn` executors — the seam
    /// changes KV addressing only.
    ///
    /// Non-circular reference: each layer's historical page is reconstructed
    /// independently from `pack(selected, layer)`, while the suffix page is the
    /// D2H of the composite stripe the legacy paged writer produced (shared
    /// unchanged code). The composite historical stripe is ALSO asserted ==
    /// `pack(selected, layer)` directly, and after the legacy run the reference
    /// suffix page is asserted == the composite one (the writer addressed both
    /// identically). Selection is READ from the pool — the real HFQ4 Q direction
    /// decides it — so the proof assumes only that the restore matches the
    /// choice, never a particular page.
    #[test]
    #[ignore]
    fn layerwise_path_a_forward_matches_legacy_paged() {
        use crate::kvmem_taps::KvMemTaps;
        use crate::qwen35::forward::{forward_scratch_layers_paged, Qwen35Scratch};
        use crate::qwen35::weights::{DeltaNetState, LayerWeights, Qwen35Weights, StateQuant};
        use hipfire_dispatch::pipeline::hybrid::PagedAttentionKv;
        use hipfire_runtime::llama::EmbeddingFormat;
        use path_fixture::*;

        let mut gpu = Gpu::init().expect("gpu");
        if gpu.arch != "gfx1100" {
            // The paged GQA decode tile and the ported scorer admit gfx1100 only.
            return;
        }

        // ── statistics + packed store: both canonical pages, both layers ──
        let q8 = LayerwiseQ8Layout::resolve(NL, KVH, HD, &[0, 1]).expect("q8 layout");
        let record_bytes = q8.payload_layout().expect("manifest").record_bytes();
        let budget = || MemoryBudget {
            selected_tokens: BT as u64,
            reserve_tokens: BT as u64,
            device_tokens: 2 * BT as u64,
            host_payload_bytes: 2 * record_bytes,
        };
        let policy = KvStore::new(&KvStoreConfig {
            block_tokens: BT as u32,
            select_budget: BT as u32,
            sink_blocks: 0,
            recent_blocks: 0,
            ..Default::default()
        })
        .expect("policy");
        policy.register_append(FRONTIER as u32).expect("append");
        let keys_storage = Storage::new_ram(16 << 20).expect("keys storage");
        let keys = RawStore::new(q8.raw_config().clone(), Some(&keys_storage)).expect("keys");
        keys.ensure_blocks(2).expect("ensure blocks");
        for b in 0..2u32 {
            for il in 0..NL {
                keys.write_layer_k_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, false))
                    .expect("feed K");
                keys.write_layer_v_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, true))
                    .expect("feed V");
            }
        }
        // Mean-K sign trick: layer 0 favors canonical page 1, layer 1 page 0
        // (which page actually wins depends on the projected Q's sign).
        for il in 0..NL {
            for b in 0..2u32 {
                let sign = if (b == 1) == (il == 0) { 1.0f32 } else { -1.0f32 };
                let tile = vec![sign; BT * KVDIM];
                keys.write_layer_mean_k(b * BT as u32, BT as u32, il as u32, &tile).expect("mean-k");
            }
        }

        // ── driver + carrier; reserve the suffix slot, extract OWNED arena views ──
        let mut driver = LayerwiseDriver::new(
            &mut gpu,
            &keys,
            &policy,
            &q8,
            fx_backend(),
            fx_capabilities(),
            fx_snapshots(),
            NSLOTS,
            16 << 20,
        )
        .expect("driver");
        let suffix_phys = driver.arena_mut().pool().alloc_slot().expect("suffix slot").phys;
        let arena_numel = driver.arena().arena().numel();
        let mut k_views = Vec::with_capacity(NL);
        let mut v_views = Vec::with_capacity(NL);
        for _ in 0..NL {
            k_views.push(driver.arena().arena().sub_offset(0, arena_numel));
            v_views.push(driver.arena().v_cache_view().expect("v view"));
        }
        let staging = LayerwiseDescStaging::new(&mut gpu, NL, MAXP).expect("staging");
        let mut carrier = LayerwiseQuery::new(driver, staging);
        carrier.suffix_slot = Some(suffix_phys);

        let identity = LayerwiseIdentity {
            history: fx_stamp(),
            request: 78,
            evaluation: QueryRange { begin: FRONTIER, end: FRONTIER + 1 },
            query: QueryRange { begin: FRONTIER, end: FRONTIER + 1 },
        };
        let geom = |ordinal: u32| LayerQueryGeometry {
            layer: ordinal,
            query_heads: NH as u32,
            kv_heads: KVH as u32,
            head_dim: HD as u32,
            attention_scale: 0.0,
        };
        let specs = vec![
            LayerRequestSpec { model_layer: 0, query: geom(0), budget: budget(), mandatory: vec![] },
            LayerRequestSpec { model_layer: 1, query: geom(1), budget: budget(), mandatory: vec![] },
        ];
        carrier.driver.begin_query(identity, specs, BT as u64).expect("begin_query");
        assert_eq!(carrier.driver.phase(), Some(LayerwisePhase::BetweenLayers));

        // ── micro dense model: REAL HFQ4-G256 projections + FFN ──
        let config = micro_config();
        let s = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, CTX).expect("scratch");
        let dn_state =
            DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::Q8).expect("dn state");
        let weights = Qwen35Weights {
            token_embd: gpu.zeros(&[config.vocab_size * DIM], DType::F32).expect("embd"),
            embd_format: EmbeddingFormat::F32,
            output_norm: upload_f32(&mut gpu, &vec![1.0; DIM]),
            output: dummy_weight(&mut gpu),
            layers: vec![
                LayerWeights::FullAttn(dense_fa_layer(&mut gpu, 11)),
                LayerWeights::FullAttn(dense_fa_layer(&mut gpu, 29)),
            ],
            mq3v2_symmetric: false,
            moe_has_mq6: false,
            pager: None,
            lm_head_aliases_embd: false,
            ep_shard: None,
        };
        // The statistics tap both runs share (a pure copy into the tap buffers):
        // its presence keeps the unfused Q/K prep on BOTH sides.
        let taps = KvMemTaps::new(&mut gpu, NL, NSLOTS, 1, NH, KVH, HD).expect("taps");

        // ── legacy contiguous reference: one arena per layer + shared table ──
        let mut ref_k: Vec<GpuTensor> = Vec::with_capacity(NL);
        let mut ref_v: Vec<GpuTensor> = Vec::with_capacity(NL);
        for _ in 0..NL {
            ref_k.push(gpu.alloc_tensor(&[2 * PLANE], DType::Raw).expect("ref K"));
            ref_v.push(gpu.alloc_tensor(&[2 * PLANE], DType::Raw).expect("ref V"));
        }
        let ref_positions = upload_i32(&mut gpu, &[EXTENT - 1]); // [128] compacted bound
        let ref_row_slot = upload_i32(&mut gpu, &[0]);
        let ref_bt = upload_i32(&mut gpu, &[0, 1, -1]); // history, suffix, masked pad
        let ref_descs = upload_ref_descs(&mut gpu, &ref_bt);
        let ref_paged = PagedAttentionKv {
            positions: &ref_positions,
            descs: &ref_descs,
            row_slot: &ref_row_slot,
            ctx_len: CTX,
        };

        // The decode row's residual input, identical for both runs.
        let x0: Vec<f32> = (0..DIM).map(|i| (((i * 7) % 13) as f32) * 0.1 - 0.6).collect();
        let seed_row = |gpu: &mut Gpu| {
            put_f32(gpu, &s.x, &x0);
            gpu.hip
                .memcpy_htod(&s.pos_buf, &(FRONTIER as i32).to_ne_bytes())
                .expect("pos h2d");
        };

        // ── Layerwise Path A forward (the real dispatch, not a stub) ──
        seed_row(&mut gpu);
        forward_scratch_layers_paged(
            &mut gpu,
            &weights,
            &config,
            FRONTIER as usize,
            &q8_kv_cache(k_views, v_views),
            &dn_state,
            &s,
            ref_paged, // FA layers override it with their own restored view
            Some(&taps),
            &ref_row_slot,
            Some(&mut carrier),
            NL,
        )
        .expect("layerwise Path A forward");
        let lw_x = read_f32(&mut gpu, &s.x, DIM);
        assert!(lw_x.iter().all(|v| v.is_finite()), "layerwise residual must be finite");

        // ── per-layer selection, restore and staging contract ──
        let mut sel_pages = Vec::with_capacity(NL);
        let mut suffix_bytes = Vec::with_capacity(NL);
        for l in 0..NL {
            let pages = carrier.driver.arena().pool_ref().canonical_pages(l).unwrap();
            assert_eq!(pages.len(), 1, "layer {l}: budget selects exactly one page");
            let sel = pages[0];
            sel_pages.push(sel);
            let hist_phys = carrier.driver.arena().pool_ref().compact_slots(l).unwrap()[0];
            assert_ne!(hist_phys, suffix_phys, "layer {l}: history vs suffix slot");
            let (hk, hv) =
                carrier.driver.arena().stripe_views(hist_phys as usize, l).expect("hist stripe");
            assert_eq!(
                read_bytes(&mut gpu, &hk, PLANE),
                pack(sel, l as u32, false),
                "layer {l} restored K bytes"
            );
            assert_eq!(
                read_bytes(&mut gpu, &hv, PLANE),
                pack(sel, l as u32, true),
                "layer {l} restored V bytes"
            );
            assert_eq!(
                read_i32(&mut gpu, carrier.staging.positions(l).unwrap(), 1),
                vec![EXTENT - 1],
                "layer {l} positions"
            );
            let d = read_desc(&mut gpu, carrier.staging.descs(l).unwrap());
            assert_eq!(d.seq_len, EXTENT, "layer {l} desc seq_len");
            assert_eq!(d.page_tokens, BT as i32, "layer {l} desc page_tokens");

            // Reference arena for this layer: independently reconstructed
            // history page + the suffix page the paged writer produced.
            let (sk, sv) =
                carrier.driver.arena().stripe_views(suffix_phys as usize, l).expect("suffix stripe");
            let suffix_k = read_bytes(&mut gpu, &sk, PLANE);
            let suffix_v = read_bytes(&mut gpu, &sv, PLANE);
            gpu.hip.memcpy_htod_offset(&ref_k[l].buf, 0, &pack(sel, l as u32, false)).expect("ref K0");
            gpu.hip.memcpy_htod_offset(&ref_k[l].buf, PLANE, &suffix_k).expect("ref K1");
            gpu.hip.memcpy_htod_offset(&ref_v[l].buf, 0, &pack(sel, l as u32, true)).expect("ref V0");
            gpu.hip.memcpy_htod_offset(&ref_v[l].buf, PLANE, &suffix_v).expect("ref V1");
            suffix_bytes.push((suffix_k, suffix_v));
        }

        // Both layers restored + attended → the transaction awaits acceptance.
        assert_eq!(carrier.driver.phase(), Some(LayerwisePhase::AwaitingAcceptance));
        let work = carrier.driver.work().expect("work");
        assert_eq!(work.projected_rows as usize, NL, "2 layers × 1 evaluation row");
        assert_eq!(work.attended_rows as usize, NL);
        assert_eq!(work.completed_layers as usize, NL);

        // ── legacy Path A forward over the identical compacted working set ──
        // Non-owning aliases so the reference arenas stay inspectable after the
        // run (the composite views above are aliases for the same reason).
        let ref_k_views: Vec<GpuTensor> =
            ref_k.iter().map(|t| t.sub_offset(0, t.numel())).collect();
        let ref_v_views: Vec<GpuTensor> =
            ref_v.iter().map(|t| t.sub_offset(0, t.numel())).collect();
        seed_row(&mut gpu);
        forward_scratch_layers_paged(
            &mut gpu,
            &weights,
            &config,
            FRONTIER as usize,
            &q8_kv_cache(ref_k_views, ref_v_views),
            &dn_state,
            &s,
            ref_paged,
            Some(&taps),
            &ref_row_slot,
            None,
            NL,
        )
        .expect("legacy Path A forward");
        let ref_x = read_f32(&mut gpu, &s.x, DIM);
        assert!(
            ref_x.iter().map(|f| f.to_bits()).eq(lw_x.iter().map(|f| f.to_bits())),
            "Path A layerwise residual must be byte-exact vs legacy paged\n\
             selected={sel_pages:?}\nlayerwise={lw_x:?}\nlegacy={ref_x:?}"
        );
        // The legacy paged writer landed the same suffix bytes in the reference
        // arena: both runs really attended the identical compacted working set.
        for l in 0..NL {
            let (sk, sv) = &suffix_bytes[l];
            assert_eq!(
                read_bytes(&mut gpu, &ref_k[l], 2 * PLANE)[PLANE..],
                sk[..],
                "layer {l} reference suffix K"
            );
            assert_eq!(
                read_bytes(&mut gpu, &ref_v[l], 2 * PLANE)[PLANE..],
                sv[..],
                "layer {l} reference suffix V"
            );
        }

        // ── teardown (composite arena views are non-owning — never freed) ──
        carrier.free(&mut gpu).expect("free carrier");
        weights.free_gpu(&mut gpu);
        s.free_gpu(&mut gpu).expect("free scratch");
        dn_state.free_gpu(&mut gpu);
        gpu.free_tensor(taps.q_tap).expect("free q tap");
        gpu.free_tensor(taps.k_tap).expect("free k tap");
        for t in ref_k {
            gpu.free_tensor(t).expect("free ref K");
        }
        for t in ref_v {
            gpu.free_tensor(t).expect("free ref V");
        }
        gpu.free_tensor(ref_positions).expect("free ref pos");
        gpu.free_tensor(ref_row_slot).expect("free ref row_slot");
        gpu.free_tensor(ref_bt).expect("free ref bt");
        gpu.free_tensor(ref_descs).expect("free ref descs");
    }

    /// D4b-3 Path B end-to-end GPU proof (design §3.9b/§3.9f). The Path B
    /// attention seam (`forward_slots::layerwise_fa_slots`) is driven through the
    /// REAL `run_layers_slots` slots layer loop — the batched `kv_write_slots` +
    /// `tier_attend_slots(paged_decode = Some(ctx_len))` ladder, NOT Path A/C's
    /// `kv_attention_paged` — for a 2-FA-layer micro dense model, and the residual
    /// is byte-exact vs the UNCHANGED legacy slots forward (`None` carrier) over
    /// the identical compacted working set. Exercises the whole Path B chain:
    /// post-norm Q → stats tap (legacy slot-id row_slot) → GPU score → select →
    /// COW restore → per-layer block-table/descriptor/positions staging →
    /// composite-arena KV write + paged GQA-shared decode → attend_complete.
    ///
    /// Why a DENSE micro model proves the MoE path: the dense and MoE FullAttention
    /// slots bodies (`run_fullattn_layer_slots` / `run_fullattn_moe_layer_slots`)
    /// share the IDENTICAL factored attention seam — both dispatch to the same
    /// `layerwise_fa_slots` / `legacy_fa_attend_slots` helpers (the 4b-7 span was
    /// deduplicated precisely so one seam serves both). The MoE FFN
    /// (`prefill_moe_ffn_body_batched`) is a separate certified subsystem,
    /// UNCHANGED on both arms and orthogonal to KV addressing; synthesizing a
    /// runnable sealed-MoE fixture (MQ4G256 experts + top-K==8 + the full routing
    /// program) would test that subsystem, not this seam. The throat's MoE
    /// admission + fall-through fix is host dispatch logic covered by the D4b-4
    /// orchestration and the B/N gates. This mirrors Path A, which likewise tested
    /// its path entry (`forward_scratch_layers_paged`) directly, not the throat.
    ///
    /// Non-circular reference (same argument as Path A): each layer's historical
    /// page is reconstructed independently from `pack(selected, layer)` (catches a
    /// wrong restore), while the suffix page is the D2H of the composite stripe
    /// the legacy paged writer produced (shared unchanged code). The composite
    /// historical stripe is ALSO asserted == `pack(selected, layer)` directly, and
    /// after the legacy run the reference suffix page is asserted == the composite
    /// one (the writer addressed both identically). Selection is READ from the
    /// pool — the real HFQ4 Q direction decides it — so the proof assumes only
    /// that the restore matches the choice, never a particular page. Both runs
    /// pass the SAME statistics tap and the SAME slots projection/FFN ladder, so
    /// the comparison isolates the Layerwise KV addressing + restore interleaving.
    #[test]
    #[ignore]
    fn layerwise_path_b_forward_matches_legacy_slots() {
        use crate::forward_slots::{run_layers_slots, LayerKvAddr, SlotDescStaging, SlotKvTier};
        use crate::qwen35::weights::{DeltaNetState, LayerWeights, Qwen35Weights};
        use crate::qwen35::{PrefillBatchScratch, Qwen35Scratch};
        use hipfire_runtime::llama::EmbeddingFormat;
        use hipfire_runtime::slot_batch::SlotBatch;
        use path_fixture::*;
        use rdna_compute::kv_slots::KvSlotDesc;
        use rdna_compute::slot_pool::SlotId;

        let mut gpu = Gpu::init().expect("gpu");
        if gpu.arch != "gfx1100" {
            // The paged GQA decode tile and the ported scorer admit gfx1100 only.
            return;
        }

        // ── statistics + packed store: both canonical pages, both layers ──
        let q8 = LayerwiseQ8Layout::resolve(NL, KVH, HD, &[0, 1]).expect("q8 layout");
        let record_bytes = q8.payload_layout().expect("manifest").record_bytes();
        let budget = || MemoryBudget {
            selected_tokens: BT as u64,
            reserve_tokens: BT as u64,
            device_tokens: 2 * BT as u64,
            host_payload_bytes: 2 * record_bytes,
        };
        let policy = KvStore::new(&KvStoreConfig {
            block_tokens: BT as u32,
            select_budget: BT as u32,
            sink_blocks: 0,
            recent_blocks: 0,
            ..Default::default()
        })
        .expect("policy");
        policy.register_append(FRONTIER as u32).expect("append");
        let keys_storage = Storage::new_ram(16 << 20).expect("keys storage");
        let keys = RawStore::new(q8.raw_config().clone(), Some(&keys_storage)).expect("keys");
        keys.ensure_blocks(2).expect("ensure blocks");
        for b in 0..2u32 {
            for il in 0..NL {
                keys.write_layer_k_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, false))
                    .expect("feed K");
                keys.write_layer_v_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, true))
                    .expect("feed V");
            }
        }
        // Mean-K sign trick: layer 0 favors canonical page 1, layer 1 page 0
        // (which page actually wins depends on the projected Q's sign).
        for il in 0..NL {
            for b in 0..2u32 {
                let sign = if (b == 1) == (il == 0) { 1.0f32 } else { -1.0f32 };
                let tile = vec![sign; BT * KVDIM];
                keys.write_layer_mean_k(b * BT as u32, BT as u32, il as u32, &tile).expect("mean-k");
            }
        }

        // ── driver + carrier; reserve the suffix slot ──
        let mut driver = LayerwiseDriver::new(
            &mut gpu,
            &keys,
            &policy,
            &q8,
            fx_backend(),
            fx_capabilities(),
            fx_snapshots(),
            NSLOTS,
            16 << 20,
        )
        .expect("driver");
        let suffix_phys = driver.arena_mut().pool().alloc_slot().expect("suffix slot").phys;
        let staging = LayerwiseDescStaging::new(&mut gpu, NL, MAXP).expect("staging");
        let mut carrier = LayerwiseQuery::new(driver, staging);
        carrier.suffix_slot = Some(suffix_phys);

        let identity = LayerwiseIdentity {
            history: fx_stamp(),
            request: 79,
            evaluation: QueryRange { begin: FRONTIER, end: FRONTIER + 1 },
            query: QueryRange { begin: FRONTIER, end: FRONTIER + 1 },
        };
        let geom = |ordinal: u32| LayerQueryGeometry {
            layer: ordinal,
            query_heads: NH as u32,
            kv_heads: KVH as u32,
            head_dim: HD as u32,
            attention_scale: 0.0,
        };
        let specs = vec![
            LayerRequestSpec { model_layer: 0, query: geom(0), budget: budget(), mandatory: vec![] },
            LayerRequestSpec { model_layer: 1, query: geom(1), budget: budget(), mandatory: vec![] },
        ];
        carrier.driver.begin_query(identity, specs, BT as u64).expect("begin_query");
        assert_eq!(carrier.driver.phase(), Some(LayerwisePhase::BetweenLayers));

        // ── micro dense model: REAL HFQ4-G256 projections + FFN ──
        let config = micro_config();
        let s = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, CTX).expect("scratch");
        let weights = Qwen35Weights {
            token_embd: gpu.zeros(&[config.vocab_size * DIM], DType::F32).expect("embd"),
            embd_format: EmbeddingFormat::F32,
            output_norm: upload_f32(&mut gpu, &vec![1.0; DIM]),
            output: dummy_weight(&mut gpu),
            layers: vec![
                LayerWeights::FullAttn(dense_fa_layer(&mut gpu, 11)),
                LayerWeights::FullAttn(dense_fa_layer(&mut gpu, 29)),
            ],
            mq3v2_symmetric: false,
            moe_has_mq6: false,
            pager: None,
            lm_head_aliases_embd: false,
            ep_shard: None,
        };

        // ── slots-path scratch + batch + tier + legacy descriptor staging ──
        // The slots ladder reads the residual from `pbs.x_batch` and the RoPE /
        // causal position from `pbs.positions` (NOT `s.x` / `s.pos_buf`, which
        // belong to the reference decode route Path A exercises). `seed_row`
        // below sets that position to the compact bound `EXTENT-1`.
        let mut pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, 1, false).expect("pbs");
        // Both runs share the statistics tap (pure copy): the Layerwise seam reads
        // the legacy slot-id row_slot from the Arena addr for it, exactly as the
        // legacy seam does.
        pbs.enable_kvmem_taps(&mut gpu, &config, NSLOTS).expect("taps");
        let kv = SlotKvTier::q8();
        let batch = SlotBatch::build(&[(SlotId(0), &[1u32], (EXTENT - 1) as usize)]);
        let dn_states: &[DeltaNetState] = &[]; // no DeltaNet layers in the micro model
        let n = batch.total_rows();
        assert_eq!(n, 1);

        // Legacy reference arenas (one per FA layer): history page at offset 0,
        // suffix page at offset PLANE. Filled AFTER the Layerwise run from the
        // selected page + the composite suffix stripe; the Layerwise arm never
        // reads them (it overrides addressing with the composite arena).
        let mut ref_k: Vec<GpuTensor> = Vec::with_capacity(NL);
        let mut ref_v: Vec<GpuTensor> = Vec::with_capacity(NL);
        for _ in 0..NL {
            ref_k.push(gpu.alloc_tensor(&[2 * PLANE], DType::Raw).expect("ref K"));
            ref_v.push(gpu.alloc_tensor(&[2 * PLANE], DType::Raw).expect("ref V"));
        }
        // Legacy SlotDescStaging: one slot, one row, compacted 2-page working set
        // (history physical page 0, suffix physical page 1, masked pad). The
        // descriptor's block_table pointer is stable; contents uploaded once.
        let legacy_staging = SlotDescStaging::new(&mut gpu, 1, 1, MAXP).expect("legacy staging");
        gpu.hip
            .memcpy_htod(&legacy_staging.row_slot_dev.buf, &0i32.to_ne_bytes())
            .expect("legacy row_slot h2d");
        let bt_host = [0i32, 1, -1];
        gpu.hip
            .memcpy_htod(&legacy_staging.block_table_devs[0].buf, unsafe {
                std::slice::from_raw_parts(bt_host.as_ptr() as *const u8, MAXP * 4)
            })
            .expect("legacy block table h2d");
        let legacy_desc = KvSlotDesc {
            block_table: legacy_staging.block_table_devs[0].buf.as_ptr() as u64,
            legacy_k_base: 0,
            legacy_v_base: 0,
            seq_len: EXTENT,
            page_tokens: BT as i32,
        };
        gpu.hip
            .memcpy_htod(&legacy_staging.descs_dev.buf, unsafe {
                std::slice::from_raw_parts(
                    &legacy_desc as *const KvSlotDesc as *const u8,
                    std::mem::size_of::<KvSlotDesc>(),
                )
            })
            .expect("legacy desc h2d");

        // The legacy kv_addr closure: Arena addressing through the reference
        // arenas + shared staging, `paged_decode = Some(CTX)` so BOTH arms take
        // the identical `attention_flash_q8_0_gqa_decode_paged` route (the
        // Layerwise arm passes the same; single_slot/tiles are ignored there).
        // Captures only shared references ⇒ Copy ⇒ reused across both runs.
        let kv_addr = |fa: usize| LayerKvAddr::Arena {
            k_cache: &ref_k[fa],
            v_cache: &ref_v[fa],
            desc_staging: &legacy_staging,
            single_slot: None,
            n_tiles: None,
            paged_verify: None,
            paged_decode: Some(CTX),
        };

        // The decode row's residual input, identical for both runs.
        //
        // Position: the slots path drives RoPE AND the legacy KV write/attend
        // causal bound from the SAME `pbs.positions` buffer (`fa_rope_slots` +
        // `layer_kv_write_attend`'s Arena arm). Unlike Path A's reference-decode
        // route — which separates `s.pos_buf` (RoPE @ frontier) from
        // `ref_positions` (compact paged bound) — the legacy slots arm has no
        // second position source. So for the legacy reference to attend the SAME
        // compacted working set (extent EXTENT) as the Layerwise carrier,
        // `pbs.positions` must carry the COMPACT bound `EXTENT-1`, not the real
        // decode frontier. Both runs then RoPE at EXTENT-1 and attend [0, EXTENT);
        // the Layerwise arm's carrier staging independently carries that same
        // compact bound, and its suffix write lands at suffix-page offset 0 in
        // BOTH arms (legacy logical page 1 → physical page 1 = ref offset PLANE).
        //
        // The real engine's Layerwise slots arm decouples RoPE@frontier from
        // attend@compact; that decoupling is proven at the kernel level by Path
        // A/C. Here the coincident bound isolates the slots seam's composite
        // addressing + per-layer restore interleaving — Path B's mandate — while
        // still exercising real subset selection (1 of 2 canonical pages).
        let x0: Vec<f32> = (0..DIM).map(|i| (((i * 7) % 13) as f32) * 0.1 - 0.6).collect();
        let seed_row = |gpu: &mut Gpu, pbs: &PrefillBatchScratch| {
            put_f32(gpu, &pbs.x_batch, &x0);
            gpu.hip
                .memcpy_htod(&pbs.positions.buf, &((EXTENT - 1) as i32).to_ne_bytes())
                .expect("pos h2d");
        };

        // ── Layerwise Path B forward (the real slots dispatch, not a stub) ──
        seed_row(&mut gpu, &pbs);
        run_layers_slots(
            &mut gpu,
            &weights,
            &config,
            &batch,
            dn_states,
            kv_addr,
            &kv,
            &pbs,
            &s,
            /*q8_wmma_arch=*/ false,
            n,
            /*physical_cap=*/ CTX,
            /*max_ctx_len=*/ CTX,
            /*use_mrope=*/ false,
            /*layer_end=*/ NL,
            /*spec_capture=*/ None,
            Some(&mut carrier),
        )
        .expect("layerwise Path B forward");
        let lw_x = read_f32(&mut gpu, &pbs.x_batch, DIM);
        assert!(lw_x.iter().all(|v| v.is_finite()), "layerwise residual must be finite");

        // ── per-layer selection, restore and staging contract ──
        let mut sel_pages = Vec::with_capacity(NL);
        let mut suffix_bytes = Vec::with_capacity(NL);
        for l in 0..NL {
            let pages = carrier.driver.arena().pool_ref().canonical_pages(l).unwrap();
            assert_eq!(pages.len(), 1, "layer {l}: budget selects exactly one page");
            let sel = pages[0];
            sel_pages.push(sel);
            let hist_phys = carrier.driver.arena().pool_ref().compact_slots(l).unwrap()[0];
            assert_ne!(hist_phys, suffix_phys, "layer {l}: history vs suffix slot");
            let (hk, hv) =
                carrier.driver.arena().stripe_views(hist_phys as usize, l).expect("hist stripe");
            assert_eq!(
                read_bytes(&mut gpu, &hk, PLANE),
                pack(sel, l as u32, false),
                "layer {l} restored K bytes"
            );
            assert_eq!(
                read_bytes(&mut gpu, &hv, PLANE),
                pack(sel, l as u32, true),
                "layer {l} restored V bytes"
            );
            assert_eq!(
                read_i32(&mut gpu, carrier.staging.positions(l).unwrap(), 1),
                vec![EXTENT - 1],
                "layer {l} positions"
            );
            let d = read_desc(&mut gpu, carrier.staging.descs(l).unwrap());
            assert_eq!(d.seq_len, EXTENT, "layer {l} desc seq_len");
            assert_eq!(d.page_tokens, BT as i32, "layer {l} desc page_tokens");

            // Reference arena for this layer: independently reconstructed
            // history page + the suffix page the paged writer produced.
            let (sk, sv) =
                carrier.driver.arena().stripe_views(suffix_phys as usize, l).expect("suffix stripe");
            let suffix_k = read_bytes(&mut gpu, &sk, PLANE);
            let suffix_v = read_bytes(&mut gpu, &sv, PLANE);
            gpu.hip.memcpy_htod_offset(&ref_k[l].buf, 0, &pack(sel, l as u32, false)).expect("ref K0");
            gpu.hip.memcpy_htod_offset(&ref_k[l].buf, PLANE, &suffix_k).expect("ref K1");
            gpu.hip.memcpy_htod_offset(&ref_v[l].buf, 0, &pack(sel, l as u32, true)).expect("ref V0");
            gpu.hip.memcpy_htod_offset(&ref_v[l].buf, PLANE, &suffix_v).expect("ref V1");
            suffix_bytes.push((suffix_k, suffix_v));
        }

        // Both layers restored + attended → the transaction awaits acceptance.
        assert_eq!(carrier.driver.phase(), Some(LayerwisePhase::AwaitingAcceptance));
        let work = carrier.driver.work().expect("work");
        assert_eq!(work.projected_rows as usize, NL, "2 layers × 1 evaluation row");
        assert_eq!(work.attended_rows as usize, NL);
        assert_eq!(work.completed_layers as usize, NL);

        // ── legacy Path B forward over the identical compacted working set ──
        seed_row(&mut gpu, &pbs);
        run_layers_slots(
            &mut gpu,
            &weights,
            &config,
            &batch,
            dn_states,
            kv_addr,
            &kv,
            &pbs,
            &s,
            /*q8_wmma_arch=*/ false,
            n,
            /*physical_cap=*/ CTX,
            /*max_ctx_len=*/ CTX,
            /*use_mrope=*/ false,
            /*layer_end=*/ NL,
            /*spec_capture=*/ None,
            None,
        )
        .expect("legacy Path B forward");
        let ref_x = read_f32(&mut gpu, &pbs.x_batch, DIM);
        assert!(
            ref_x.iter().map(|f| f.to_bits()).eq(lw_x.iter().map(|f| f.to_bits())),
            "Path B layerwise residual must be byte-exact vs legacy slots\n\
             selected={sel_pages:?}\nlayerwise={lw_x:?}\nlegacy={ref_x:?}"
        );
        // The legacy paged writer landed the same suffix bytes in the reference
        // arena: both runs really attended the identical compacted working set.
        for l in 0..NL {
            let (sk, sv) = &suffix_bytes[l];
            assert_eq!(
                read_bytes(&mut gpu, &ref_k[l], 2 * PLANE)[PLANE..],
                sk[..],
                "layer {l} reference suffix K"
            );
            assert_eq!(
                read_bytes(&mut gpu, &ref_v[l], 2 * PLANE)[PLANE..],
                sv[..],
                "layer {l} reference suffix V"
            );
        }

        // ── teardown (composite arena views are non-owning — never freed) ──
        carrier.free(&mut gpu).expect("free carrier");
        weights.free_gpu(&mut gpu);
        s.free_gpu(&mut gpu).expect("free scratch");
        pbs.free_gpu(&mut gpu).expect("free pbs");
        legacy_staging.free_gpu(&mut gpu);
        kv.free_gpu(&mut gpu);
        for t in ref_k {
            gpu.free_tensor(t).expect("free ref K");
        }
        for t in ref_v {
            gpu.free_tensor(t).expect("free ref V");
        }
    }

    /// D4b-4 engine-synthesis end-to-end GPU proof (design §3.9g). Unlike Path B
    /// (a hand-built `fx_snapshots` carrier at a page boundary), this builds the
    /// carrier through the PRODUCTION synthesis the serve orchestration uses —
    /// [`bootstrap_layerwise`] + [`LayerwiseQuery::build`] over a directly-fed
    /// `RawStore` — and drives the REAL `run_layers_slots` slots ladder for a
    /// 2-FA-layer micro dense model (real HFQ4-G256 projections + FFN). It covers
    /// BOTH suffix geometries the engine admits:
    ///
    ///   * `partial == 0` (fed 256, a page boundary): the suffix page starts
    ///     empty, `restore_suffix_page` is a no-op, extent = 128 + 0 + 1 = 129 —
    ///     reproducing Path B's geometry but via the production bootstrap, so the
    ///     canonical/identity/specs synthesis is proven against the same seam.
    ///   * `partial == 40` (fed 296, MID-PAGE): the suffix page (logical page 2)
    ///     holds 40 already-fed tokens that `restore_suffix_page` re-exports from
    ///     the host store into the suffix stripe BEFORE the live current-token
    ///     write at offset 40; extent = 128 + 40 + 1 = 169. This is the new
    ///     correctness surface D4b-4 added (partial restore + `suffix_extent =
    ///     partial_tokens + rows`), which the D4b-3 page-boundary paths never hit.
    ///
    /// Each case is byte-exact vs the UNCHANGED legacy paged GQA decode over the
    /// identical compacted working set (1 of 2 canonical pages selected by the
    /// mean-K sign trick + the suffix stripe). Non-circular reference: the
    /// historical page is reconstructed independently from `pack(selected,
    /// layer)`; the mid-page suffix's fed prefix is ALSO asserted == the fed
    /// `pack(2, layer)` bytes directly (catches a wrong/missing partial restore);
    /// the suffix stripe the shared paged writer produced is D2H'd into the
    /// reference arena so both arms attend identical bytes. The coincident
    /// RoPE/attend bound (`pbs.positions = extent-1`) isolates the slots seam's
    /// composite addressing + per-layer restore interleaving, exactly as Path B.
    #[test]
    #[ignore]
    fn layerwise_path_d_engine_bootstrap_midpage_matches_legacy_slots() {
        use crate::forward_slots::{run_layers_slots, LayerKvAddr, SlotDescStaging, SlotKvTier};
        use crate::qwen35::weights::{DeltaNetState, LayerWeights, Qwen35Weights};
        use crate::qwen35::{PrefillBatchScratch, Qwen35Scratch};
        use hipfire_runtime::llama::EmbeddingFormat;
        use hipfire_runtime::slot_batch::SlotBatch;
        use path_fixture::*;
        use rdna_compute::kv_slots::KvSlotDesc;
        use rdna_compute::slot_pool::SlotId;

        let mut gpu = Gpu::init().expect("gpu");
        if gpu.arch != "gfx1100" {
            // The paged GQA decode tile and the ported scorer admit gfx1100 only.
            return;
        }

        // Production geometry: 2 FA layers, DENSE ordinals [0, 1] (the engine's
        // `fa_layers`, matching KvMemCtl's `raw_cfg.n_layer == n_fa_layers`).
        let q8 = LayerwiseQ8Layout::resolve(NL, KVH, HD, &[0, 1]).expect("q8 layout");
        let record_bytes = q8.payload_layout().expect("manifest").record_bytes();

        // Micro dense model shared by both cases (real HFQ4 projections + FFN).
        let config = micro_config();
        let s = Qwen35Scratch::new_with_kv_max(&mut gpu, &config, 64, CTX).expect("scratch");
        let weights = Qwen35Weights {
            token_embd: gpu.zeros(&[config.vocab_size * DIM], DType::F32).expect("embd"),
            embd_format: EmbeddingFormat::F32,
            output_norm: upload_f32(&mut gpu, &vec![1.0; DIM]),
            output: dummy_weight(&mut gpu),
            layers: vec![
                LayerWeights::FullAttn(dense_fa_layer(&mut gpu, 11)),
                LayerWeights::FullAttn(dense_fa_layer(&mut gpu, 29)),
            ],
            mq3v2_symmetric: false,
            moe_has_mq6: false,
            pager: None,
            lm_head_aliases_embd: false,
            ep_shard: None,
        };
        // The decode row's residual input, identical for every run.
        let x0: Vec<f32> = (0..DIM).map(|i| (((i * 7) % 13) as f32) * 0.1 - 0.6).collect();
        let seed_row = |gpu: &mut Gpu, pbs: &PrefillBatchScratch, extent: i32| {
            put_f32(gpu, &pbs.x_batch, &x0);
            gpu.hip
                .memcpy_htod(&pbs.positions.buf, &(extent - 1).to_ne_bytes())
                .expect("pos h2d");
        };

        // Page boundary (fed 256) AND mid-page suffix (fed 296).
        for partial in [0u32, 40u32] {
            let full_pages = 2u32;
            let fed = full_pages * BT as u32 + partial;
            let budget_blocks = 1u32; // select exactly 1 of the 2 canonical pages
            let selected_pages = full_pages.min(budget_blocks.max(1)) as usize;
            let extent = (selected_pages * BT + partial as usize + 1) as i32;

            // ── host store: 2 sealed canonical pages + the partial suffix page ──
            // The selection policy indexes ONLY the sealed full pages, mirroring
            // the production fresh page-aligned policy (`KvMemCtl::layerwise_policy`):
            // the core picker rejects a partial tail. The partial suffix page is
            // fed to the statistics store (`keys`) below, NOT to this policy.
            let policy = KvStore::new(&KvStoreConfig {
                block_tokens: BT as u32,
                select_budget: BT as u32,
                sink_blocks: 0,
                recent_blocks: 0,
                ..Default::default()
            })
            .expect("policy");
            policy.register_append(full_pages * BT as u32).expect("append");
            let keys_storage = Storage::new_ram(16 << 20).expect("keys storage");
            let keys = RawStore::new(q8.raw_config().clone(), Some(&keys_storage)).expect("keys");
            // The suffix page (logical page `full_pages`) must exist for the
            // mid-page re-export; at a page boundary it is allocated but unused.
            keys.ensure_blocks(full_pages + 1).expect("ensure blocks");
            for b in 0..full_pages {
                for il in 0..NL {
                    keys.write_layer_k_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, false))
                        .expect("feed K");
                    keys.write_layer_v_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, true))
                        .expect("feed V");
                }
            }
            if partial > 0 {
                for il in 0..NL {
                    let kb = pack(full_pages, il as u32, false);
                    let vb = pack(full_pages, il as u32, true);
                    keys.write_layer_k_gpu(
                        full_pages * BT as u32, partial, il as u32,
                        &kb[..partial as usize * SEG],
                    )
                    .expect("feed suffix K");
                    keys.write_layer_v_gpu(
                        full_pages * BT as u32, partial, il as u32,
                        &vb[..partial as usize * SEG],
                    )
                    .expect("feed suffix V");
                }
            }
            // Mean-K sign trick: layer 0 favors canonical page 1, layer 1 page 0.
            for il in 0..NL {
                for b in 0..full_pages {
                    let sign = if (b == 1) == (il == 0) { 1.0f32 } else { -1.0f32 };
                    let tile = vec![sign; BT * KVDIM];
                    keys.write_layer_mean_k(b * BT as u32, BT as u32, il as u32, &tile).expect("mean-k");
                }
            }

            // ── PRODUCTION synthesis: bootstrap_layerwise + LayerwiseQuery::build ──
            // 0-based engine identities (SessionTable mints SessionId(0) first; a
            // fresh slot's incarnation/seq start at 0) — the hardest case for the
            // core's non-zero logical-identity requirement, normalized by bootstrap.
            let bootstrap = bootstrap_layerwise(
                fed,
                /*session_id=*/ 0,
                /*incarnation=*/ 0,
                /*seq=*/ 0,
                &[0, 1],
                NH as u32,
                KVH as u32,
                HD as u32,
                budget_blocks,
                record_bytes,
                /*resident=*/ &[],
            )
            .expect("bootstrap");
            assert_eq!(bootstrap.suffix_page, full_pages, "suffix page = fed/128");
            assert_eq!(bootstrap.partial_tokens, partial, "partial = fed%128");
            assert_eq!(bootstrap.canonical.len(), NL, "one snapshot per fa ordinal");
            // The synthesized identity must satisfy the core's non-zero logical
            // identity + frontier-aligned evaluation invariants (memory_contract.cpp:157,
            // layerwise_selection.cpp:85-90): 0-based engine counters map to 1-based
            // core identities, and evaluation.begin == frontier == full_pages*128.
            let frontier = full_pages as u64 * BT as u64;
            assert_eq!(bootstrap.identity.history.session.id, 1, "session id normalized");
            assert_eq!(bootstrap.identity.history.session.incarnation, 1, "incarnation normalized");
            assert_eq!(bootstrap.identity.history.history_revision, 1, "revision normalized");
            assert_eq!(bootstrap.identity.history.execution_history, 1, "execution normalized");
            assert_eq!(bootstrap.identity.request, 1, "request id normalized");
            assert_eq!(bootstrap.identity.history.frontier, frontier, "frontier = full pages");
            assert_eq!(bootstrap.identity.evaluation.begin, frontier, "evaluation begins at frontier");
            assert_eq!(bootstrap.identity.evaluation.end, frontier + 1, "single decode row");
            let n_slots = selected_pages + 2;
            let archive_bytes = (selected_pages as u64 + 2) * record_bytes;
            let mut carrier = LayerwiseQuery::build(
                &mut gpu,
                &keys,
                &policy,
                &q8,
                layerwise_backend("qwen35"),
                layerwise_capabilities(CTX as u32),
                bootstrap,
                n_slots,
                archive_bytes,
            )
            .expect("build carrier");
            let suffix_phys = carrier.suffix_slot.expect("build reserves the suffix slot");
            assert_eq!(carrier.suffix_page, full_pages);
            assert_eq!(carrier.partial_tokens, partial);
            assert_eq!(carrier.driver.phase(), Some(LayerwisePhase::BetweenLayers));

            // ── slots scratch + batch + legacy descriptor staging ──
            let mut pbs = PrefillBatchScratch::new_opt(&mut gpu, &config, 1, false).expect("pbs");
            pbs.enable_kvmem_taps(&mut gpu, &config, NSLOTS).expect("taps");
            let kv = SlotKvTier::q8();
            let batch = SlotBatch::build(&[(SlotId(0), &[1u32], (extent - 1) as usize)]);
            let dn_states: &[DeltaNetState] = &[];
            let n = batch.total_rows();
            assert_eq!(n, 1);

            let mut ref_k: Vec<GpuTensor> = Vec::with_capacity(NL);
            let mut ref_v: Vec<GpuTensor> = Vec::with_capacity(NL);
            for _ in 0..NL {
                ref_k.push(gpu.alloc_tensor(&[2 * PLANE], DType::Raw).expect("ref K"));
                ref_v.push(gpu.alloc_tensor(&[2 * PLANE], DType::Raw).expect("ref V"));
            }
            let legacy_staging = SlotDescStaging::new(&mut gpu, 1, 1, MAXP).expect("legacy staging");
            gpu.hip
                .memcpy_htod(&legacy_staging.row_slot_dev.buf, &0i32.to_ne_bytes())
                .expect("legacy row_slot h2d");
            let bt_host = [0i32, 1, -1];
            gpu.hip
                .memcpy_htod(&legacy_staging.block_table_devs[0].buf, unsafe {
                    std::slice::from_raw_parts(bt_host.as_ptr() as *const u8, MAXP * 4)
                })
                .expect("legacy block table h2d");
            let legacy_desc = KvSlotDesc {
                block_table: legacy_staging.block_table_devs[0].buf.as_ptr() as u64,
                legacy_k_base: 0,
                legacy_v_base: 0,
                seq_len: extent,
                page_tokens: BT as i32,
            };
            gpu.hip
                .memcpy_htod(&legacy_staging.descs_dev.buf, unsafe {
                    std::slice::from_raw_parts(
                        &legacy_desc as *const KvSlotDesc as *const u8,
                        std::mem::size_of::<KvSlotDesc>(),
                    )
                })
                .expect("legacy desc h2d");
            let kv_addr = |fa: usize| LayerKvAddr::Arena {
                k_cache: &ref_k[fa],
                v_cache: &ref_v[fa],
                desc_staging: &legacy_staging,
                single_slot: None,
                n_tiles: None,
                paged_verify: None,
                paged_decode: Some(CTX),
            };

            // ── Layerwise forward (production carrier, real slots dispatch) ──
            seed_row(&mut gpu, &pbs, extent);
            run_layers_slots(
                &mut gpu, &weights, &config, &batch, dn_states, kv_addr, &kv, &pbs, &s,
                /*q8_wmma_arch=*/ false, n, /*physical_cap=*/ CTX, /*max_ctx_len=*/ CTX,
                /*use_mrope=*/ false, /*layer_end=*/ NL, /*spec_capture=*/ None,
                Some(&mut carrier),
            )
            .expect("layerwise Path D forward");
            let lw_x = read_f32(&mut gpu, &pbs.x_batch, DIM);
            assert!(lw_x.iter().all(|v| v.is_finite()), "layerwise residual must be finite");

            // ── per-layer selection / restore / staging contract ──
            let mut sel_pages = Vec::with_capacity(NL);
            let mut suffix_bytes = Vec::with_capacity(NL);
            for l in 0..NL {
                let pages = carrier.driver.arena().pool_ref().canonical_pages(l).unwrap();
                assert_eq!(pages.len(), selected_pages, "layer {l}: budget selects one page");
                let sel = pages[0];
                sel_pages.push(sel);
                let hist_phys = carrier.driver.arena().pool_ref().compact_slots(l).unwrap()[0];
                assert_ne!(hist_phys, suffix_phys, "layer {l}: history vs suffix slot");
                let (hk, hv) =
                    carrier.driver.arena().stripe_views(hist_phys as usize, l).expect("hist stripe");
                assert_eq!(read_bytes(&mut gpu, &hk, PLANE), pack(sel, l as u32, false), "layer {l} K");
                assert_eq!(read_bytes(&mut gpu, &hv, PLANE), pack(sel, l as u32, true), "layer {l} V");
                assert_eq!(
                    read_i32(&mut gpu, carrier.staging.positions(l).unwrap(), 1),
                    vec![extent - 1],
                    "layer {l} positions = compacted causal bound"
                );
                let d = read_desc(&mut gpu, carrier.staging.descs(l).unwrap());
                assert_eq!(d.seq_len, extent, "layer {l} desc seq_len");
                assert_eq!(d.page_tokens, BT as i32, "layer {l} desc page_tokens");

                let (sk, sv) =
                    carrier.driver.arena().stripe_views(suffix_phys as usize, l).expect("suffix stripe");
                let suffix_k = read_bytes(&mut gpu, &sk, PLANE);
                let suffix_v = read_bytes(&mut gpu, &sv, PLANE);
                // Mid-page: the suffix stripe's fed prefix must be the partial
                // page's already-fed bytes, restored from the host store BEFORE
                // the live write at offset `partial` (which lands just past it).
                if partial > 0 {
                    let fed_bytes = partial as usize * SEG;
                    assert_eq!(
                        suffix_k[..fed_bytes],
                        pack(full_pages, l as u32, false)[..fed_bytes],
                        "layer {l} mid-page suffix K fed prefix restored"
                    );
                    assert_eq!(
                        suffix_v[..fed_bytes],
                        pack(full_pages, l as u32, true)[..fed_bytes],
                        "layer {l} mid-page suffix V fed prefix restored"
                    );
                }
                // Reference arena: independent history page + the suffix stripe
                // the shared paged writer produced (D2H → both arms identical).
                gpu.hip.memcpy_htod_offset(&ref_k[l].buf, 0, &pack(sel, l as u32, false)).expect("ref K0");
                gpu.hip.memcpy_htod_offset(&ref_k[l].buf, PLANE, &suffix_k).expect("ref K1");
                gpu.hip.memcpy_htod_offset(&ref_v[l].buf, 0, &pack(sel, l as u32, true)).expect("ref V0");
                gpu.hip.memcpy_htod_offset(&ref_v[l].buf, PLANE, &suffix_v).expect("ref V1");
                suffix_bytes.push((suffix_k, suffix_v));
            }
            assert_eq!(carrier.driver.phase(), Some(LayerwisePhase::AwaitingAcceptance));
            let work = carrier.driver.work().expect("work");
            assert_eq!(work.projected_rows as usize, NL, "2 layers × 1 evaluation row");
            assert_eq!(work.attended_rows as usize, NL);
            assert_eq!(work.completed_layers as usize, NL);

            // ── legacy forward over the identical compacted working set ──
            seed_row(&mut gpu, &pbs, extent);
            run_layers_slots(
                &mut gpu, &weights, &config, &batch, dn_states, kv_addr, &kv, &pbs, &s,
                /*q8_wmma_arch=*/ false, n, /*physical_cap=*/ CTX, /*max_ctx_len=*/ CTX,
                /*use_mrope=*/ false, /*layer_end=*/ NL, /*spec_capture=*/ None,
                None,
            )
            .expect("legacy Path D forward");
            let ref_x = read_f32(&mut gpu, &pbs.x_batch, DIM);
            assert!(
                ref_x.iter().map(|f| f.to_bits()).eq(lw_x.iter().map(|f| f.to_bits())),
                "Path D (partial={partial}, fed={fed}, extent={extent}) layerwise residual \
                 must be byte-exact vs legacy slots\nselected={sel_pages:?}\n\
                 layerwise={lw_x:?}\nlegacy={ref_x:?}"
            );
            for l in 0..NL {
                let (sk, sv) = &suffix_bytes[l];
                assert_eq!(read_bytes(&mut gpu, &ref_k[l], 2 * PLANE)[PLANE..], sk[..], "layer {l} ref suffix K");
                assert_eq!(read_bytes(&mut gpu, &ref_v[l], 2 * PLANE)[PLANE..], sv[..], "layer {l} ref suffix V");
            }

            // ── per-case teardown (composite arena views are non-owning) ──
            carrier.free(&mut gpu).expect("free carrier");
            pbs.free_gpu(&mut gpu).expect("free pbs");
            legacy_staging.free_gpu(&mut gpu);
            kv.free_gpu(&mut gpu);
            for t in ref_k {
                gpu.free_tensor(t).expect("free ref K");
            }
            for t in ref_v {
                gpu.free_tensor(t).expect("free ref V");
            }
        }

        weights.free_gpu(&mut gpu);
        s.free_gpu(&mut gpu).expect("free scratch");
    }

    /// Carrier-level pressure transaction (design §3.10a, decision A, ③-E3.5):
    /// the FULL ninfer-faithful mass → accept → select → Pressure-restore chain
    /// over a directly-fed `RawStore`, driven through the production
    /// [`bootstrap_layerwise`] + [`LayerwiseQuery::build_with_pressure`] seam (no
    /// `run_layers_slots`; the carrier is exercised directly so the mass hook and
    /// the resident-only selection are isolated from the projection ladder).
    ///
    /// Geometry: `path_fixture` (head_dim 256 / 6 Q heads / 1 KV head — the mass
    /// kernel hard-gates head_dim==256), 3 sealed canonical pages, a ONE-page
    /// resident budget (so the history overflows the budget and Pressure
    /// refinement is admitted). Two phases mirror the engine's Q2/Q3 trigger:
    ///
    ///   * Phase 1 — a NEW-PAGE Query step (fed 384, a page boundary): the ②
    ///     scorer selects one page per layer (mean-K sign trick: layer 0 → page
    ///     2, layer 1 → page 1), the carrier restores it byte-exact, captures
    ///     attention mass against a controlled post-RoPE Q tile (the current
    ///     token is written into the suffix stripe first, exactly as the paged
    ///     writer does), and `accept_pressure` publishes the whole-model packet
    ///     into a fresh persistent owner. The Query keep-set UNION is verified.
    ///   * Phase 2 — a MID-PAGE Pressure step (fed 424, partial 40): the owner's
    ///     `select` refines a RESIDENT-ONLY keep-set (candidates come exclusively
    ///     from the accepted resident map — NO cold recall), the carrier restores
    ///     the pressure-selected pages byte-exact into a FRESH arena (the
    ///     device-truth invariant: a freshly bound stripe is always restored even
    ///     when the plan marks it resident), the mid-page suffix fed prefix is
    ///     restored byte-exact, the empty `rows()` schedule makes mass/accept a
    ///     no-op, and the Pressure keep-set TRUNCATE is verified.
    ///
    /// Byte-exact references are the independent `pack(page, layer, K/V)` pattern
    /// (non-circular). The mass leg is proven by `accept` returning `Ok(true)`
    /// (the core enforces per-row |Σmass−1|≤.005) plus a positive-mass snapshot;
    /// the E2 kernel test already qualifies the mass math against an FP64 oracle.
    #[test]
    #[ignore]
    fn carrier_pressure_mass_accept_select_byte_exact() {
        use hipfire_kvmem::{PrefillOptions, PressureSource};
        use path_fixture::*;
        use rdna_compute::kvmem_page_mass::PageMassScalar;

        let mut gpu = Gpu::init().expect("gpu");
        if gpu.arch != "gfx1100" {
            // The ported scorer + mass kernel admit gfx1100 only.
            return;
        }

        let q8 = LayerwiseQ8Layout::resolve(NL, KVH, HD, &[0, 1]).expect("q8 layout");
        let record_bytes = q8.payload_layout().expect("manifest").record_bytes();
        let full_pages = 3u32;
        let budget_blocks = 1u32; // one-page resident budget ⇒ history overflows
        let frontier = full_pages as u64 * BT as u64; // 384

        // ── host store: 3 sealed canonical pages + a 40-token partial suffix ──
        let policy = KvStore::new(&KvStoreConfig {
            block_tokens: BT as u32,
            select_budget: BT as u32,
            sink_blocks: 0,
            recent_blocks: 0,
            ..Default::default()
        })
        .expect("policy");
        policy.register_append(full_pages * BT as u32).expect("append");
        let keys_storage = Storage::new_ram(16 << 20).expect("keys storage");
        let keys = RawStore::new(q8.raw_config().clone(), Some(&keys_storage)).expect("keys");
        keys.ensure_blocks(full_pages + 1).expect("ensure blocks");
        for b in 0..full_pages {
            for il in 0..NL {
                keys.write_layer_k_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, false))
                    .expect("feed K");
                keys.write_layer_v_gpu(b * BT as u32, BT as u32, il as u32, &pack(b, il as u32, true))
                    .expect("feed V");
            }
        }
        // The partial suffix page (logical page 3): 40 already-fed tokens for the
        // mid-page Pressure step's `restore_suffix_page`.
        let partial = 40u32;
        for il in 0..NL {
            let kb = pack(full_pages, il as u32, false);
            let vb = pack(full_pages, il as u32, true);
            keys.write_layer_k_gpu(full_pages * BT as u32, partial, il as u32, &kb[..partial as usize * SEG])
                .expect("feed suffix K");
            keys.write_layer_v_gpu(full_pages * BT as u32, partial, il as u32, &vb[..partial as usize * SEG])
                .expect("feed suffix V");
        }
        // Mean-K sign trick: layer 0 favors page 2, layer 1 favors page 1.
        for il in 0..NL {
            for b in 0..full_pages {
                let sign = if b == (2 - il as u32) { 1.0f32 } else { -1.0f32 };
                let tile = vec![sign; BT * KVDIM];
                keys.write_layer_mean_k(b * BT as u32, BT as u32, il as u32, &tile).expect("mean-k");
            }
        }

        // The persistent session-lifetime pressure owner (mirrors
        // `KvMemCtl::ensure_layerwise_pressure`: 1-based session, ninfer defaults).
        let mut pressure = PrefillPressure::new(
            SessionIdentity { id: 1, incarnation: 1 },
            &[0, 1],
            &PrefillOptions {
                block_tokens: BT as u32,
                select_tokens: budget_blocks as u64 * BT as u64,
                reserve_tokens: BT as u64,
                interval: 1024,
                stride: 32,
                half_life: 8192.0,
            },
        )
        .expect("pressure owner");

        // Shared all-+1 native Q tile (scorer pre-RoPE + mass post-RoPE; the test
        // feeds arbitrary packed K, so the same tile serves both — the E2 kernel
        // test qualifies the mass math, here we prove the integration).
        let q_dev = upload_f32(&mut gpu, &vec![1.0f32; QDIM]);
        let native_q = NativeCurrentLayerQuery {
            data: q_dev.buf.as_ptr() as *const std::ffi::c_void,
            bytes: QDIM * 4,
            layer: 0,
            rows: 1,
            query_heads: NH as u32,
            kv_heads: KVH as u32,
            head_dim: HD as u32,
            attention_scale: 0.0,
            scalar: 0, // F32
            row_stride: QDIM * 4,
            head_stride: HD * 4,
            dimension_stride: 4,
        };
        let mass_q = PageMassQuery {
            data: q_dev.buf.as_ptr() as *const std::ffi::c_void,
            rows: 1,
            query_heads: NH as u32,
            kv_heads: KVH as u32,
            head_dim: HD as u32,
            attention_scale: 0.0,
            scalar: PageMassScalar::F32,
        };
        let n_slots = 1 + 2; // selected_pages(1) + suffix + headroom
        let archive_bytes = (full_pages as u64 + 2) * record_bytes;

        // ── Phase 1: NEW-PAGE Query step (fed 384) with mass capture ──────────
        let bootstrap1 = bootstrap_layerwise(
            frontier as u32, 0, 0, 0, &[0, 1], NH as u32, KVH as u32, HD as u32,
            budget_blocks, record_bytes, &[],
        )
        .expect("bootstrap1");
        assert_eq!(bootstrap1.identity.history.frontier, frontier);
        assert_eq!(bootstrap1.native_history_tokens, BT as u64, "one selected page");
        assert_eq!(bootstrap1.partial_tokens, 0, "page boundary");
        let local_begin1 = bootstrap1.native_history_tokens; // 128
        let mass_rows1 = pressure.rows(frontier - 1, 1, local_begin1, false).expect("rows1");
        assert_eq!(mass_rows1, vec![0u32], "first sample admits the single decode row");
        let mass1 = MassCapture::new(NL, frontier, local_begin1, 1, mass_rows1);
        let mut carrier1 = LayerwiseQuery::build_with_pressure(
            &mut gpu, &keys, &policy, &q8, layerwise_backend("qwen35"),
            layerwise_capabilities(CTX as u32), bootstrap1, n_slots, archive_bytes,
            None, Some(mass1),
        )
        .expect("build query carrier");
        assert!(!carrier1.is_pressure(), "phase 1 is a scorer Query step");
        let suffix_phys1 = carrier1.suffix_slot.expect("suffix slot");

        let mut keep1: Vec<Vec<u64>> = Vec::with_capacity(NL);
        for ordinal in 0..NL {
            // The scorer validates q.layer == the admitted spec geometry layer
            // (the dense ordinal), so tag the shared Q tile per layer.
            let layer_q = NativeCurrentLayerQuery { layer: ordinal as u32, ..native_q };
            carrier1.restore_layer(&mut gpu, ordinal, &layer_q).expect("query restore");
            let pages = carrier1.driver.arena().pool_ref().canonical_pages(ordinal).unwrap();
            assert_eq!(pages.len(), 1, "layer {ordinal}: budget selects one page");
            let sel = pages[0];
            // The mean-K trick: layer 0 → page 2, layer 1 → page 1.
            assert_eq!(sel, 2 - ordinal as u32, "layer {ordinal} scorer selection");
            let hist_phys = carrier1.driver.arena().pool_ref().compact_slots(ordinal).unwrap()[0];
            let (hk, hv) = carrier1.driver.arena().stripe_views(hist_phys as usize, ordinal).expect("hist stripe");
            assert_eq!(read_bytes(&mut gpu, &hk, PLANE), pack(sel, ordinal as u32, false), "layer {ordinal} query K byte-exact");
            assert_eq!(read_bytes(&mut gpu, &hv, PLANE), pack(sel, ordinal as u32, true), "layer {ordinal} query V byte-exact");
            // Write the current token into the suffix stripe (as the paged writer
            // does) so the mass gather sees a deterministic suffix K/V row.
            let k_off = carrier1.driver.arena().layout().layer_k_offset(suffix_phys1 as usize, ordinal).unwrap();
            let v_off = carrier1.driver.arena().layout().layer_v_offset(suffix_phys1 as usize, ordinal).unwrap();
            gpu.hip.memcpy_htod_offset(&carrier1.driver.arena().arena().buf, k_off, &pack(full_pages, ordinal as u32, false)[..SEG]).expect("suffix K write");
            gpu.hip.memcpy_htod_offset(&carrier1.driver.arena().arena().buf, v_off, &pack(full_pages, ordinal as u32, true)[..SEG]).expect("suffix V write");
            carrier1.capture_mass_layer(&mut gpu, ordinal, &mass_q).expect("capture mass");
            // Close this layer's attending phase so the state machine admits the
            // next ordinal's begin_layer (the production forward calls this after
            // the paged attention; mass capture precedes it, post-attention).
            carrier1.attend_complete_layer(ordinal).expect("attend complete");
            keep1.push(vec![sel as u64]);
        }
        // Whole-model acceptance: the core enforces per-row mass conservation
        // (|Σmass−1|≤.005), so Ok(true) proves it. The selected canonical page
        // must carry positive mass (it drives the resident-only selection); the
        // suffix page may legitimately underflow to 0 mass with this arbitrary
        // packed-K / all-+1-Q fixture and is never a selection candidate.
        assert!(carrier1.accept_pressure(&mut pressure).expect("accept"), "phase 1 must accept");
        let snap1 = pressure.snapshot().expect("snapshot1").read().expect("read snap1");
        assert_eq!(snap1.layers.len(), NL);
        for l in &snap1.layers {
            assert_eq!(l.samples, 1, "one accepted sample");
            assert_eq!(l.end, frontier, "accepted frontier");
            assert!(!l.values.is_empty(), "accepted mass values");
            assert!(l.values.iter().all(|v| v.mass.is_finite() && v.mass >= 0.0), "non-negative finite mass");
            let sum: f64 = l.values.iter().map(|v| v.mass).sum();
            assert!((sum - 1.0).abs() <= 0.005, "mass conserved across pages: {sum}");
            let sel_start = keep1[l.model_layer as usize][0] * BT as u64;
            let sel_val = l.values.iter().find(|v| v.original_start == sel_start).expect("selected page accepted");
            assert!(sel_val.mass > 0.0, "selected page carries positive mass");
        }
        // Query keep-set == the scorer's restored selection; UNION with empty prev.
        let restored1 = carrier1.restored_keep_set().expect("keep1");
        assert_eq!(restored1, keep1, "query keep-set == scorer selection");
        let combined1 = combine_keep_set(&[], &restored1, false, NL);
        assert_eq!(combined1, keep1, "query union with empty prev");
        carrier1.free(&mut gpu).expect("free carrier1");

        // ── Phase 2: MID-PAGE Pressure step (fed 424, partial 40) ─────────────
        let fed2 = frontier as u32 + partial; // 424
        let bootstrap2 = bootstrap_layerwise(
            fed2, 0, 0, /*seq=*/ 1, &[0, 1], NH as u32, KVH as u32, HD as u32,
            budget_blocks, record_bytes, &combined1,
        )
        .expect("bootstrap2");
        assert_eq!(bootstrap2.partial_tokens, partial);
        assert_eq!(bootstrap2.identity.history.frontier, frontier, "frontier unchanged mid-page");
        // The resident list now carries the keep-set (decision A).
        assert_eq!(bootstrap2.canonical[0].memory.resident, vec![2u64]);
        assert_eq!(bootstrap2.canonical[1].memory.resident, vec![1u64]);
        let local_begin2 = bootstrap2.native_history_tokens + partial as u64; // 168
        // The frontier did not advance ⇒ rows() admits no sample (no capture/accept).
        let mass_rows2 = pressure.rows(frontier - 1, 1, local_begin2, false).expect("rows2");
        assert!(mass_rows2.is_empty(), "mid-page Pressure step samples nothing");

        // Resident-only selection over a catalog whose resident == the keep-set.
        let descriptor2 = MemoryDescriptor::new(
            &layerwise_backend("qwen35"), &q8.payload_layout().expect("manifest2"),
            &layerwise_capabilities(CTX as u32),
        )
        .expect("descriptor2");
        let catalog2 = MemoryCatalog::new(&descriptor2, &bootstrap2.canonical).expect("catalog2");
        let choices = pressure
            .select(&catalog2, &[], budget_blocks as u64 * BT as u64, 0)
            .expect("select")
            .read_all()
            .expect("read choices");
        assert_eq!(choices.len(), NL);
        let mut selected2: Vec<Vec<u64>> = Vec::with_capacity(NL);
        for ordinal in 0..NL {
            let c = &choices[ordinal];
            assert_eq!(c.model_layer, ordinal as u32);
            assert_eq!(c.source, PressureSource::Prefill);
            assert!(c.samples > 0, "positive accepted state");
            assert!(!c.selected.is_empty(), "layer {ordinal}: not Pending");
            // NO cold recall: every selected id is resident (a subset of the
            // keep-set). With one resident page and a one-page budget the choice
            // is exactly that page — a cold page (0) must never appear.
            let resident = &bootstrap2.canonical[ordinal].memory.resident;
            assert!(c.selected.iter().all(|id| resident.contains(id)), "layer {ordinal}: resident-only, no cold recall");
            assert_eq!(c.selected, resident.clone(), "layer {ordinal}: selects the resident page");
            selected2.push(c.selected.clone());
        }

        let mass2 = MassCapture::new(NL, frontier, local_begin2, 1, mass_rows2);
        let budgets2: Vec<MemoryBudget> = bootstrap2.specs.iter().map(|s| s.budget).collect();
        let step2 = PressureStep::new(selected2.clone(), budgets2).expect("pressure step");
        let mut carrier2 = LayerwiseQuery::build_with_pressure(
            &mut gpu, &keys, &policy, &q8, layerwise_backend("qwen35"),
            layerwise_capabilities(CTX as u32), bootstrap2, n_slots, archive_bytes,
            Some(step2), Some(mass2),
        )
        .expect("build pressure carrier");
        assert!(carrier2.is_pressure(), "phase 2 is a Pressure step");
        let suffix_phys2 = carrier2.suffix_slot.expect("suffix slot2");

        for ordinal in 0..NL {
            carrier2.restore_layer(&mut gpu, ordinal, &native_q).expect("pressure restore");
            let pages = carrier2.driver.arena().pool_ref().canonical_pages(ordinal).unwrap();
            assert_eq!(pages.len(), 1);
            let sel = selected2[ordinal][0] as u32;
            assert_eq!(pages[0], sel, "layer {ordinal}: pressure-selected page bound");
            // Byte-exact: the resident page is restored into the FRESH arena even
            // though the plan marks it resident (device-truth invariant).
            let hist_phys = carrier2.driver.arena().pool_ref().compact_slots(ordinal).unwrap()[0];
            let (hk, hv) = carrier2.driver.arena().stripe_views(hist_phys as usize, ordinal).expect("hist stripe2");
            assert_eq!(read_bytes(&mut gpu, &hk, PLANE), pack(sel, ordinal as u32, false), "layer {ordinal} pressure K byte-exact");
            assert_eq!(read_bytes(&mut gpu, &hv, PLANE), pack(sel, ordinal as u32, true), "layer {ordinal} pressure V byte-exact");
            // The mid-page suffix fed prefix is restored byte-exact.
            let (sk, sv) = carrier2.driver.arena().stripe_views(suffix_phys2 as usize, ordinal).expect("suffix stripe2");
            let fed_bytes = partial as usize * SEG;
            assert_eq!(read_bytes(&mut gpu, &sk, PLANE)[..fed_bytes], pack(full_pages, ordinal as u32, false)[..fed_bytes], "layer {ordinal} suffix K fed prefix");
            assert_eq!(read_bytes(&mut gpu, &sv, PLANE)[..fed_bytes], pack(full_pages, ordinal as u32, true)[..fed_bytes], "layer {ordinal} suffix V fed prefix");
            // Empty schedule ⇒ mass capture is a no-op (no kernel launch).
            carrier2.capture_mass_layer(&mut gpu, ordinal, &mass_q).expect("pressure mass no-op");
        }
        // No frontier advance ⇒ accept is a no-op; the owner state is unchanged.
        assert!(!carrier2.accept_pressure(&mut pressure).expect("pressure accept no-op"), "mid-page must not accept");
        let snap2 = pressure.snapshot().expect("snapshot2").read().expect("read snap2");
        assert_eq!(snap2, snap1, "Pressure step leaves the accepted state unchanged");
        // Pressure keep-set == the refined selection; TRUNCATE replaces the pool.
        let restored2 = carrier2.restored_keep_set().expect("keep2");
        assert_eq!(restored2, selected2, "pressure keep-set == selection");
        let combined2 = combine_keep_set(&combined1, &restored2, true, NL);
        assert_eq!(combined2, selected2, "pressure truncate to budget");
        carrier2.free(&mut gpu).expect("free carrier2");

        gpu.free_tensor(q_dev).expect("free q");
    }
}