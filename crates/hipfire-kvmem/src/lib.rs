// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KVMem host-offload adapter for hipfire: a thin safe wrapper over the C ABI
//! in `kvmem/backends/hipfire/hipfire_kvmem_shim.h`.
//!
//! The wrapped stores are the portable KVMem host library: block statistics
//! (`RawKvStore`), block bookkeeping + selection (`KvMemStore`) and the
//! query-conditioned Mean-K retrieval scorer. Nothing here touches the GPU —
//! the engine copies packed Q8 KV rows to the host and hands them to
//! [`RawStore::write_layer_k_gpu`]/[`RawStore::write_layer_v_gpu`], and memcpy's
//! [`RawStore::copy_k_gpu`]/[`RawStore::copy_v_gpu`] rows back on restore.
//!
//! Threading: every handle owns C++ state that is not `Sync` (the C ABI is
//! serialized by the engine's serve loop), so these types are deliberately
//! neither `Send` nor `Sync`.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::path::Path;
use std::ptr::{self, NonNull};

/// GQA score aggregation for [`mean_key_retrieval_scores`] (Mean = 0, Max = 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GqaAggregation {
    Mean,
    Max,
}

impl GqaAggregation {
    fn as_u32(self) -> u32 {
        match self {
            GqaAggregation::Mean => 0,
            GqaAggregation::Max => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KvMemError(pub String);

impl fmt::Display for KvMemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kvmem: {}", self.0)
    }
}

impl std::error::Error for KvMemError {}

type Result<T> = std::result::Result<T, KvMemError>;

/// `hk_raw_store_config` — mirrors `kvmem::RawKvStoreConfig` (minus the
/// storage handle, which is a separate argument).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RawStoreConfig {
    pub n_layer: u32,
    pub n_embd_k: u32,
    pub n_embd_v: u32,
    pub block_tokens: u32,
    pub k_row_bytes: u64,
    pub k_gpu_row_bytes: u64,
    pub v_gpu_row_bytes: u64,
}

/// `hk_kv_store_config` — mirrors the subset of `kvmem::KvMemStoreConfig`
/// hipfire drives. Enum-valued fields carry the C++ enum ordinals.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct KvStoreConfig {
    pub block_tokens: u32,
    pub select_budget: u32,
    pub prefill_budget: u32,
    pub gen_budget: u32,
    pub sink_blocks: u32,
    pub recent_blocks: u32,
    pub select_method: u32,
    pub retrieval_method: u32,
    pub update_mode: u32,
    pub lrfu: u32,
    pub lrfu_lambda: f64,
    pub lrfu_top_fraction: f64,
    pub estimated_gpu_block_capacity: u32,
    pub gpu_resident_block_bytes: u64,
    pub estimated_block_bytes: u64,
    pub cpu_tier_bytes: u64,
    pub gpu_memory_ratio: f64,
    pub gpu_high_watermark: f64,
    pub gpu_low_watermark: f64,
}

impl Default for KvStoreConfig {
    fn default() -> Self {
        // kvmem::KvMemStoreConfig defaults; select_budget is the caller's.
        Self {
            block_tokens: 128,
            select_budget: 131_072,
            prefill_budget: 0,
            gen_budget: 32_768,
            sink_blocks: 1,
            recent_blocks: 0,
            select_method: 0,    // Retrieval
            retrieval_method: 0, // MeanK
            update_mode: 0,      // Interval
            lrfu: 0,
            lrfu_lambda: 0.5,
            lrfu_top_fraction: 0.9,
            estimated_gpu_block_capacity: 0,
            gpu_resident_block_bytes: 0,
            estimated_block_bytes: 0,
            cpu_tier_bytes: 0,
            gpu_memory_ratio: 0.50,
            gpu_high_watermark: 0.95,
            gpu_low_watermark: 0.85,
        }
    }
}

#[repr(C)]
struct HkStorage {
    _private: [u8; 0],
}

#[repr(C)]
struct HkRawStore {
    _private: [u8; 0],
}

#[repr(C)]
struct HkKvStore {
    _private: [u8; 0],
}

// ── layerwise native payload foundation: C struct mirrors + opaque handles ──

/// `hk_session_identity` — mirrors `kvmem::SessionIdentity`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionIdentity {
    pub id: u64,
    pub incarnation: u64,
}

/// `hk_block_descriptor` — mirrors `kvmem::BlockDescriptor`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockDescriptor {
    pub session: SessionIdentity,
    pub id: u64,
    pub original_start: u64,
    pub valid_tokens: u32,
    pub content_version: u64,
    pub statistics_version: u64,
}

/// `hk_payload_plane_kind` ordinals (append-only, mirrors `kvmem::PayloadPlaneKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PayloadPlaneKind {
    Key = 0,
    Value = 1,
    KeyScale = 2,
    ValueScale = 3,
    PackedKeyValue = 4,
}

/// `hk_payload_plane` — mirrors `kvmem::PayloadPlane`. `kind` carries a
/// [`PayloadPlaneKind`] ordinal; `token_stride == 0` means an opaque page.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadPlane {
    pub layer: u32,
    pub kind: u32,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub token_stride: u64,
    pub rank: u32,
    pub head_begin: u32,
    pub head_count: u32,
}

/// `hk_memory_capabilities` — mirrors `kvmem::MemoryCapabilities` (bools as 0/1).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryCapabilities {
    pub page_tokens: u32,
    pub max_position: u64,
    pub native_payload: u32,
    pub sparse_view: u32,
    pub original_rope_positions: u32,
    pub pre_rope_statistics: u32,
    pub state_checkpoint: u32,
    pub asynchronous_transfer: u32,
}

/// Owned backend/model/state identity. The C ABI borrows NUL-terminated strings;
/// [`BackendIdentity::ffi`] keeps them alive for the duration of a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendIdentity {
    pub backend: String,
    pub revision: String,
    pub model: String,
    pub state_schema: String,
}

#[repr(C)]
struct HkBackendIdentity {
    backend: *const c_char,
    revision: *const c_char,
    model: *const c_char,
    state_schema: *const c_char,
}

impl BackendIdentity {
    /// Build the borrowed C view. The returned `CString`s must outlive the view.
    fn ffi(&self) -> Result<(CString, CString, CString, CString, HkBackendIdentity)> {
        let cv = |s: &str, what: &str| {
            CString::new(s).map_err(|e| KvMemError(format!("{what}: {e}")))
        };
        let b = cv(&self.backend, "backend")?;
        let r = cv(&self.revision, "revision")?;
        let m = cv(&self.model, "model")?;
        let s = cv(&self.state_schema, "state_schema")?;
        let view = HkBackendIdentity {
            backend: b.as_ptr(),
            revision: r.as_ptr(),
            model: m.as_ptr(),
            state_schema: s.as_ptr(),
        };
        Ok((b, r, m, s, view))
    }
}

#[repr(C)]
struct HkPayloadLayout {
    _private: [u8; 0],
}
#[repr(C)]
struct HkMemoryDescriptor {
    _private: [u8; 0],
}
#[repr(C)]
struct HkKvRegions {
    _private: [u8; 0],
}
#[repr(C)]
struct HkHostRecord {
    _private: [u8; 0],
}
#[repr(C)]
struct HkHostSource {
    _private: [u8; 0],
}
#[repr(C)]
struct HkNativePayload {
    _private: [u8; 0],
}
#[repr(C)]
struct HkNativeArchive {
    _private: [u8; 0],
}

// ── layerwise catalog + planning: C struct mirrors + opaque handles ──

/// `hk_memory_stamp` — mirrors `kvmem::MemoryStamp` (`valid` as 0/1).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStamp {
    pub valid: u32,
    pub session: SessionIdentity,
    pub history_revision: u64,
    pub view_version: u64,
    pub frontier: u64,
    pub execution_history: u64,
}

/// `hk_block_copies` — a logical block plus its host-copy version (0: none).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockCopies {
    pub block: BlockDescriptor,
    pub host_version: u64,
}

/// `hk_memory_budget` — mirrors `kvmem::MemoryBudget`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    pub selected_tokens: u64,
    pub reserve_tokens: u64,
    pub device_tokens: u64,
    pub host_payload_bytes: u64,
}

/// `hk_view_block` — a published view block (attention compact address only).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewBlock {
    pub block: BlockDescriptor,
    pub compact_start: u64,
}

/// Owned `kvmem::MemorySnapshot`: complete logical catalog plus resident set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySnapshot {
    pub stamp: MemoryStamp,
    pub blocks: Vec<BlockCopies>,
    pub resident: Vec<u64>,
}

/// Owned `kvmem::LayerMemorySnapshot`: one target layer's canonical snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerMemorySnapshot {
    pub model_layer: u32,
    pub memory: MemorySnapshot,
}

/// Owned `kvmem::LayerWorkingSetSelection`: one layer's selection input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSelection {
    pub model_layer: u32,
    pub expected: MemoryStamp,
    pub selected: Vec<u64>,
    pub mandatory: Vec<u64>,
    pub budget: MemoryBudget,
}

impl LayerSelection {
    /// Borrowed C view; the returned struct points into `self`'s vectors.
    fn ffi(&self) -> HkLayerSelection {
        HkLayerSelection {
            model_layer: self.model_layer,
            expected: self.expected,
            selected: self.selected.as_ptr(),
            selected_count: self.selected.len(),
            mandatory: self.mandatory.as_ptr(),
            mandatory_count: self.mandatory.len(),
            budget: self.budget,
        }
    }
}

#[repr(C)]
struct HkMemorySnapshot {
    stamp: MemoryStamp,
    blocks: *const BlockCopies,
    block_count: usize,
    resident: *const u64,
    resident_count: usize,
}
#[repr(C)]
struct HkLayerMemorySnapshot {
    model_layer: u32,
    memory: HkMemorySnapshot,
}
#[repr(C)]
struct HkLayerSelection {
    model_layer: u32,
    expected: MemoryStamp,
    selected: *const u64,
    selected_count: usize,
    mandatory: *const u64,
    mandatory_count: usize,
    budget: MemoryBudget,
}

#[repr(C)]
struct HkMemoryCatalog {
    _private: [u8; 0],
}
#[repr(C)]
struct HkLayerPlan {
    _private: [u8; 0],
}
#[repr(C)]
struct HkLayerwisePlan {
    _private: [u8; 0],
}

// ── layerwise Query request state machine: C struct mirrors + opaque handle ──

/// `hk_query_range` — a half-open `[begin, end)` token range.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryRange {
    pub begin: u64,
    pub end: u64,
}

/// `hk_layerwise_identity` — the frozen transaction identity proven at every
/// native boundary: history stamp, request id, evaluation span and query span.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerwiseIdentity {
    pub history: MemoryStamp,
    pub request: u64,
    pub evaluation: QueryRange,
    pub query: QueryRange,
}

/// `hk_layer_query_geometry` — one target layer's attention geometry. The
/// `layer` ordinal indexes `RawKvStore` statistics; `attention_scale == 0`
/// selects `1/sqrt(head_dim)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerQueryGeometry {
    pub layer: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub attention_scale: f32,
}

/// `hk_layerwise_work` — cumulative projection/attendance counters.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerwiseWork {
    pub projected_rows: u64,
    pub attended_rows: u64,
    pub completed_layers: u32,
}

/// Mirrors `kvmem::LayerwiseRequestPhase` ordinals (`hk_layerwise_phase`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerwisePhase {
    BetweenLayers = 0,
    Projecting = 1,
    Restoring = 2,
    Attending = 3,
    AwaitingAcceptance = 4,
    Accepted = 5,
    Aborted = 6,
}

impl LayerwisePhase {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::BetweenLayers,
            1 => Self::Projecting,
            2 => Self::Restoring,
            3 => Self::Attending,
            4 => Self::AwaitingAcceptance,
            5 => Self::Accepted,
            _ => Self::Aborted,
        }
    }
}

/// Owned `kvmem::LayerwiseRequestLayer`: one target layer's admission spec.
/// `mandatory` holds canonical chronological policy ordinals (u32).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerRequestSpec {
    pub model_layer: u32,
    pub query: LayerQueryGeometry,
    pub budget: MemoryBudget,
    pub mandatory: Vec<u32>,
}

/// Owned `kvmem::ExecutionView`: the post-fence native view reported to
/// `restored()`. `blocks` are published compact-address view blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionView {
    pub stamp: MemoryStamp,
    pub blocks: Vec<ViewBlock>,
    pub visible_tokens: u64,
}

impl ExecutionView {
    /// Borrowed C view; the returned struct points into `self`'s vectors.
    fn ffi(&self) -> HkExecutionView {
        HkExecutionView {
            stamp: self.stamp,
            blocks: self.blocks.as_ptr(),
            block_count: self.blocks.len(),
            visible_tokens: self.visible_tokens,
        }
    }
}

#[repr(C)]
struct HkLayerRequestSpec {
    model_layer: u32,
    query: LayerQueryGeometry,
    budget: MemoryBudget,
    mandatory: *const u32,
    mandatory_count: usize,
}
#[repr(C)]
struct HkCurrentLayerQuery {
    data: *const f32,
    layer: u32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    attention_scale: f32,
}
#[repr(C)]
struct HkExecutionView {
    stamp: MemoryStamp,
    blocks: *const ViewBlock,
    block_count: usize,
    visible_tokens: u64,
}

#[repr(C)]
struct HkLayerwiseRequest {
    _private: [u8; 0],
}

// ── native (device-ready) Query projection: C struct mirrors ──

/// Mirrors `kvmem::NativeQueryScalar` (`hk_native_query_scalar`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum NativeQueryScalar {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
}

/// `hk_native_current_layer_query` — a borrowed native Q tensor (device or host)
/// reported to [`LayerwiseRequest::projected_native`]. Strides and `bytes` are
/// BYTE quantities and the token axis need not be outermost, so a backend can
/// expose its real tensor without a CPU copy or layout conversion. `scalar` is a
/// [`NativeQueryScalar`] ordinal. The data pointer must stay valid and aligned
/// for the scorer call; the request proves index identity, not the pointer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NativeCurrentLayerQuery {
    pub data: *const c_void,
    pub bytes: usize,
    pub layer: u32,
    pub rows: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub attention_scale: f32,
    pub scalar: u32,
    pub row_stride: usize,
    pub head_stride: usize,
    pub dimension_stride: usize,
}

/// Borrowed `kvmem::QueryScoreFrame`: the frozen per-block score denominator
/// handed to a native scorer. Every array has one entry per canonical block;
/// `score_mask` marks scorable blocks (a masked block MUST receive score 0) and
/// `transport_mask` marks blocks needing transfer.
#[derive(Debug, Clone, Copy)]
pub struct QueryScoreFrame<'a> {
    pub starts: &'a [u64],
    pub counts: &'a [u32],
    pub score_mask: &'a [u8],
    pub transport_mask: &'a [u8],
    pub accepted_through: u64,
}

#[repr(C)]
struct HkQueryScoreFrame {
    starts: *const u64,
    start_count: usize,
    counts: *const u32,
    count_count: usize,
    score_mask: *const u8,
    score_mask_count: usize,
    transport_mask: *const u8,
    transport_mask_count: usize,
    accepted_through: u64,
}

mod ffi {
    use super::*;

    extern "C" {
        pub fn hk_last_error() -> *const c_char;
        pub fn hk_clear_error();
        pub fn hk_version() -> *const c_char;

        pub fn hk_storage_create(
            capacity_bytes: u64,
            spill_dir: *const c_char,
            spill_capacity_bytes: u64,
        ) -> *mut HkStorage;
        pub fn hk_storage_destroy(s: *mut HkStorage);
        pub fn hk_storage_charged_bytes(s: *const HkStorage) -> u64;
        pub fn hk_storage_disk_bytes(s: *const HkStorage) -> u64;
        pub fn hk_storage_trim(s: *mut HkStorage, resident_target: u64);

        pub fn hk_raw_store_create(
            cfg: *const RawStoreConfig,
            storage: *mut HkStorage,
        ) -> *mut HkRawStore;
        pub fn hk_raw_store_destroy(st: *mut HkRawStore);
        pub fn hk_raw_store_ensure_blocks(st: *mut HkRawStore, block_count: u32);
        pub fn hk_raw_store_write_layer_mean_k(
            st: *mut HkRawStore,
            pos0: u32,
            n: u32,
            il: u32,
            k: *const f32,
        );
        pub fn hk_raw_store_write_layer_k_gpu(
            st: *mut HkRawStore,
            pos0: u32,
            n: u32,
            il: u32,
            k: *const u8,
        );
        pub fn hk_raw_store_write_layer_v_gpu(
            st: *mut HkRawStore,
            pos0: u32,
            n: u32,
            il: u32,
            v: *const u8,
        );
        pub fn hk_raw_store_copy_k_gpu(
            st: *const HkRawStore,
            block: u32,
            il: u32,
            out: *mut u8,
            n: u32,
        ) -> c_int;
        pub fn hk_raw_store_copy_v_gpu(
            st: *const HkRawStore,
            block: u32,
            il: u32,
            out: *mut u8,
            n: u32,
        ) -> c_int;
        pub fn hk_raw_store_mean_k(st: *const HkRawStore, block: u32, il: u32, out: *mut f32);
        pub fn hk_raw_store_mean_k_tokens(st: *const HkRawStore, block: u32, il: u32) -> u32;
        pub fn hk_raw_store_n_tokens(st: *const HkRawStore, block: u32) -> u32;
        pub fn hk_raw_store_has_k_gpu(
            st: *const HkRawStore,
            block: u32,
            il: u32,
            n: u32,
        ) -> c_int;
        pub fn hk_raw_store_has_v_gpu(
            st: *const HkRawStore,
            block: u32,
            il: u32,
            n: u32,
        ) -> c_int;
        pub fn hk_raw_store_truncate_to(st: *mut HkRawStore, token_pos: u32);
        pub fn hk_raw_store_clear(st: *mut HkRawStore);
        pub fn hk_raw_store_allocated_bytes(st: *const HkRawStore) -> u64;
        pub fn hk_raw_store_payload_bytes(st: *const HkRawStore) -> u64;
        pub fn hk_raw_store_mean_checkpoint(
            st: *const HkRawStore,
            token_pos: u32,
            out: *mut f32,
            capacity: usize,
        ) -> usize;
        pub fn hk_raw_store_restore_mean_checkpoint(
            st: *mut HkRawStore,
            token_pos: u32,
            state: *const f32,
            len: usize,
        );

        pub fn hk_kv_store_create(cfg: *const KvStoreConfig) -> *mut HkKvStore;
        pub fn hk_kv_store_destroy(ks: *mut HkKvStore);
        pub fn hk_kv_store_register_append(ks: *mut HkKvStore, n_new_tokens: u32);
        pub fn hk_kv_store_truncate_to(ks: *mut HkKvStore, token_pos: u32);
        pub fn hk_kv_store_block_count(ks: *const HkKvStore) -> u32;
        pub fn hk_kv_store_total_tokens(ks: *const HkKvStore) -> u32;
        pub fn hk_kv_store_select_budget_tokens(ks: *const HkKvStore) -> u32;
        pub fn hk_kv_store_set_runtime_select_budget(ks: *mut HkKvStore, tokens: u32);
        pub fn hk_kv_store_pick_topk_blocks(
            ks: *const HkKvStore,
            mandatory: *const u32,
            n_mandatory: u32,
            out: *mut u32,
            capacity: u32,
        ) -> u32;
        pub fn hk_kv_store_pick_prefill_pressure_blocks(
            ks: *const HkKvStore,
            mandatory: *const u32,
            n_mandatory: u32,
            out: *mut u32,
            capacity: u32,
        ) -> u32;
        pub fn hk_kv_store_set_retrieval_scores(ks: *mut HkKvStore, scores: *const f64, n: u32);
        pub fn hk_kv_store_begin_retrieval_round(ks: *mut HkKvStore);
        pub fn hk_kv_store_commit_retrieval_round(ks: *mut HkKvStore) -> c_int;
        pub fn hk_kv_store_prefill_needs_offload(
            ks: *const HkKvStore,
            resident_tokens: u32,
            incoming_tokens: u32,
            pool_tokens: u32,
        ) -> c_int;
        pub fn hk_kv_store_gpu_high_watermark_tokens(
            ks: *const HkKvStore,
            pool_tokens: u32,
        ) -> u32;
        pub fn hk_kv_store_gpu_low_watermark_tokens(
            ks: *const HkKvStore,
            pool_tokens: u32,
        ) -> u32;
        pub fn hk_kv_store_allocated_bytes(ks: *const HkKvStore) -> u64;

        pub fn hk_mean_key_retrieval_scores(
            keys: *const HkRawStore,
            blocks: u32,
            sums: *const f32,
            token_counts: *const u32,
            layers: u32,
            head_dim: u32,
            query_heads: u32,
            kv_heads: u32,
            aggregation: u32,
            out: *mut f64,
        ) -> c_int;

        pub fn hk_page_softmax_scores(
            keys: *const HkRawStore,
            blocks: u32,
            accepted_through: u64,
            block_tokens: u32,
            budget_blocks: u32,
            sink_blocks: u32,
            recent_blocks: u32,
            q_data: *const f32,
            layer_rows: *const u32,
            layers: u32,
            rows: u32,
            query_heads: u32,
            kv_heads: u32,
            head_dim: u32,
            attention_scale: f32,
            aggregation: u32,
            out: *mut f64,
        ) -> c_int;

        // ── layerwise native payload foundation ──
        pub fn hk_payload_layout_from_raw(
            cfg: *const RawStoreConfig,
            format: *const c_char,
            layers: *const u32,
            n_layers: u32,
        ) -> *mut HkPayloadLayout;
        pub fn hk_payload_layout_create(
            format: *const c_char,
            version: u32,
            block_tokens: u32,
            record_bytes: u64,
            alignment: u64,
            planes: *const PayloadPlane,
            n_planes: u32,
        ) -> *mut HkPayloadLayout;
        pub fn hk_payload_layout_destroy(l: *mut HkPayloadLayout);
        pub fn hk_payload_layout_record_bytes(l: *const HkPayloadLayout) -> u64;
        pub fn hk_payload_layout_block_tokens(l: *const HkPayloadLayout) -> u32;
        pub fn hk_payload_layout_plane_count(l: *const HkPayloadLayout) -> u32;
        pub fn hk_payload_layout_plane(
            l: *const HkPayloadLayout,
            i: u32,
            out: *mut PayloadPlane,
        ) -> c_int;

        pub fn hk_memory_descriptor_create(
            identity: *const HkBackendIdentity,
            layout: *const HkPayloadLayout,
            capabilities: *const MemoryCapabilities,
        ) -> *mut HkMemoryDescriptor;
        pub fn hk_memory_descriptor_destroy(d: *mut HkMemoryDescriptor);

        pub fn hk_host_record_allocate(
            s: *mut HkStorage,
            bytes: *const u8,
            len: usize,
        ) -> *mut HkHostRecord;
        pub fn hk_host_record_destroy(r: *mut HkHostRecord);
        pub fn hk_host_record_seal(r: *mut HkHostRecord);
        pub fn hk_host_record_size(r: *const HkHostRecord) -> usize;
        pub fn hk_host_record_read(
            r: *const HkHostRecord,
            offset: usize,
            dest: *mut u8,
            bytes: usize,
        ) -> c_int;
        pub fn hk_host_record_source(r: *const HkHostRecord) -> *mut HkHostSource;
        pub fn hk_host_source_destroy(s: *mut HkHostSource);
        pub fn hk_host_source_size(s: *const HkHostSource) -> usize;

        pub fn hk_kv_regions_create(
            layout: *const HkPayloadLayout,
            layer: u32,
            has_rank: c_int,
            rank: u32,
        ) -> *mut HkKvRegions;
        pub fn hk_kv_regions_destroy(r: *mut HkKvRegions);
        pub fn hk_kv_regions_packed_bytes(r: *const HkKvRegions) -> u64;
        pub fn hk_kv_regions_plane_bytes(r: *const HkKvRegions) -> u64;
        pub fn hk_kv_regions_record_bytes(r: *const HkKvRegions) -> u64;
        pub fn hk_kv_regions_region_count(r: *const HkKvRegions) -> u32;
        pub fn hk_kv_regions_region(
            r: *const HkKvRegions,
            i: u32,
            out_plane: *mut PayloadPlane,
            out_packed_offset: *mut u64,
        ) -> c_int;
        pub fn hk_kv_regions_read(
            r: *const HkKvRegions,
            source: *const HkHostSource,
            packed_offset: u64,
            dest: *mut u8,
            bytes: usize,
            max_read_bytes: usize,
        ) -> c_int;

        pub fn hk_native_payload_export(
            raw: *const HkRawStore,
            backend: *const HkBackendIdentity,
            layout: *const HkPayloadLayout,
            block: *const BlockDescriptor,
        ) -> *mut HkNativePayload;
        pub fn hk_native_payload_create(
            backend: *const HkBackendIdentity,
            layout: *const HkPayloadLayout,
            block: *const BlockDescriptor,
            bytes: *const u8,
            len: usize,
        ) -> *mut HkNativePayload;
        pub fn hk_native_payload_destroy(p: *mut HkNativePayload);
        pub fn hk_native_payload_bytes(p: *const HkNativePayload) -> *const u8;
        pub fn hk_native_payload_size(p: *const HkNativePayload) -> usize;
        pub fn hk_native_payload_block(p: *const HkNativePayload, out: *mut BlockDescriptor)
            -> c_int;
        pub fn hk_native_payload_import(
            raw: *mut HkRawStore,
            backend: *const HkBackendIdentity,
            layout: *const HkPayloadLayout,
            p: *const HkNativePayload,
        ) -> c_int;

        pub fn hk_native_archive_create(
            backend: *const HkBackendIdentity,
            session: SessionIdentity,
            layout: *const HkPayloadLayout,
            capacity_bytes: u64,
        ) -> *mut HkNativeArchive;
        pub fn hk_native_archive_destroy(a: *mut HkNativeArchive);
        pub fn hk_native_archive_put(a: *mut HkNativeArchive, p: *const HkNativePayload) -> c_int;
        pub fn hk_native_archive_get(
            a: *const HkNativeArchive,
            expected: *const BlockDescriptor,
        ) -> *mut HkNativePayload;
        pub fn hk_native_archive_erase(a: *mut HkNativeArchive, block_id: u64);
        pub fn hk_native_archive_bytes(a: *const HkNativeArchive) -> u64;

        pub fn hk_memory_catalog_create(
            canonical: *const HkMemoryDescriptor,
            layers: *const HkLayerMemorySnapshot,
            n_layers: usize,
        ) -> *mut HkMemoryCatalog;
        pub fn hk_memory_catalog_destroy(c: *mut HkMemoryCatalog);
        pub fn hk_memory_catalog_layer_count(c: *const HkMemoryCatalog) -> usize;
        pub fn hk_memory_catalog_model_layer(c: *const HkMemoryCatalog, i: usize) -> u32;
        pub fn hk_memory_catalog_snapshot_stamp(
            c: *const HkMemoryCatalog,
            model_layer: u32,
            out: *mut MemoryStamp,
        ) -> c_int;
        pub fn hk_memory_catalog_snapshot_block_count(
            c: *const HkMemoryCatalog,
            model_layer: u32,
        ) -> usize;
        pub fn hk_memory_catalog_snapshot_block(
            c: *const HkMemoryCatalog,
            model_layer: u32,
            i: usize,
            out: *mut BlockCopies,
        ) -> c_int;
        pub fn hk_memory_catalog_snapshot_resident_count(
            c: *const HkMemoryCatalog,
            model_layer: u32,
        ) -> usize;
        pub fn hk_memory_catalog_snapshot_resident(
            c: *const HkMemoryCatalog,
            model_layer: u32,
            i: usize,
        ) -> u64;
        pub fn hk_memory_catalog_plan_layer(
            c: *const HkMemoryCatalog,
            sel: *const HkLayerSelection,
            out: *mut *mut HkLayerPlan,
        ) -> c_int;
        pub fn hk_memory_catalog_plan(
            c: *const HkMemoryCatalog,
            sels: *const HkLayerSelection,
            n_sels: usize,
            out: *mut *mut HkLayerwisePlan,
        ) -> c_int;

        pub fn hk_layer_plan_destroy(p: *mut HkLayerPlan);
        pub fn hk_layer_plan_model_layer(p: *const HkLayerPlan) -> u32;
        pub fn hk_layer_plan_restore_bytes(p: *const HkLayerPlan) -> u64;
        pub fn hk_layer_plan_spill_bytes(p: *const HkLayerPlan) -> u64;
        pub fn hk_layer_plan_device_tokens_required(p: *const HkLayerPlan) -> u64;
        pub fn hk_layer_plan_host_payload_bytes_required(p: *const HkLayerPlan) -> u64;
        pub fn hk_layer_plan_next_stamp(p: *const HkLayerPlan, out: *mut MemoryStamp) -> c_int;
        pub fn hk_layer_plan_next_visible_tokens(p: *const HkLayerPlan) -> u64;
        pub fn hk_layer_plan_next_block_count(p: *const HkLayerPlan) -> usize;
        pub fn hk_layer_plan_next_block(
            p: *const HkLayerPlan,
            i: usize,
            out: *mut ViewBlock,
        ) -> c_int;
        pub fn hk_layer_plan_restore_count(p: *const HkLayerPlan) -> usize;
        pub fn hk_layer_plan_restore_block(
            p: *const HkLayerPlan,
            i: usize,
            out: *mut BlockDescriptor,
        ) -> c_int;
        pub fn hk_layer_plan_spill_count(p: *const HkLayerPlan) -> usize;
        pub fn hk_layer_plan_spill_block(
            p: *const HkLayerPlan,
            i: usize,
            out: *mut BlockDescriptor,
        ) -> c_int;
        pub fn hk_layer_plan_evict_count(p: *const HkLayerPlan) -> usize;
        pub fn hk_layer_plan_evict_block(
            p: *const HkLayerPlan,
            i: usize,
            out: *mut BlockDescriptor,
        ) -> c_int;
        pub fn hk_layer_plan_mandatory_count(p: *const HkLayerPlan) -> usize;
        pub fn hk_layer_plan_mandatory(p: *const HkLayerPlan, i: usize) -> u64;
        pub fn hk_layer_plan_payload(p: *const HkLayerPlan, out: *mut *mut HkKvRegions) -> c_int;

        pub fn hk_layerwise_plan_destroy(p: *mut HkLayerwisePlan);
        pub fn hk_layerwise_plan_layer_count(p: *const HkLayerwisePlan) -> usize;
        pub fn hk_layerwise_plan_history(p: *const HkLayerwisePlan, out: *mut MemoryStamp)
            -> c_int;
        pub fn hk_layerwise_plan_host_payload_bytes_required(p: *const HkLayerwisePlan) -> u64;
        pub fn hk_layerwise_plan_layer(
            p: *const HkLayerwisePlan,
            i: usize,
            out: *mut *mut HkLayerPlan,
        ) -> c_int;

        // ── layerwise Query request state machine ──
        pub fn hk_layerwise_request_create(
            identity: *const LayerwiseIdentity,
            catalog: *const HkMemoryCatalog,
            policy: *const HkKvStore,
            keys: *const HkRawStore,
            specs: *const HkLayerRequestSpec,
            n_specs: usize,
            native_history_tokens: u64,
            revision: Option<unsafe extern "C" fn(*mut c_void) -> MemoryStamp>,
            revision_user: *mut c_void,
        ) -> *mut HkLayerwiseRequest;
        pub fn hk_layerwise_request_destroy(r: *mut HkLayerwiseRequest);
        pub fn hk_layerwise_request_abort(r: *mut HkLayerwiseRequest);
        pub fn hk_layerwise_request_phase(r: *const HkLayerwiseRequest) -> u32;
        pub fn hk_layerwise_request_work(
            r: *const HkLayerwiseRequest,
            out: *mut LayerwiseWork,
        ) -> c_int;
        pub fn hk_layerwise_request_begin_layer(
            r: *mut HkLayerwiseRequest,
            id: *const LayerwiseIdentity,
            model_layer: u32,
        ) -> c_int;
        pub fn hk_layerwise_request_projected(
            r: *mut HkLayerwiseRequest,
            id: *const LayerwiseIdentity,
            begin: u64,
            q: *const HkCurrentLayerQuery,
        ) -> c_int;
        pub fn hk_layerwise_request_prepare_restore(
            r: *mut HkLayerwiseRequest,
            id: *const LayerwiseIdentity,
            out: *mut *mut HkLayerPlan,
        ) -> c_int;
        pub fn hk_layerwise_request_restored(
            r: *mut HkLayerwiseRequest,
            id: *const LayerwiseIdentity,
            view: *const HkExecutionView,
        ) -> c_int;
        pub fn hk_layerwise_request_attended(
            r: *mut HkLayerwiseRequest,
            id: *const LayerwiseIdentity,
            begin: u64,
            count: u64,
        ) -> c_int;
        pub fn hk_layerwise_request_complete_layer(
            r: *mut HkLayerwiseRequest,
            id: *const LayerwiseIdentity,
        ) -> c_int;
        pub fn hk_layerwise_request_accepted(
            r: *mut HkLayerwiseRequest,
            id: *const LayerwiseIdentity,
            snapshots: *const HkLayerMemorySnapshot,
            n: usize,
        ) -> c_int;
        pub fn hk_layerwise_request_selection_layer(r: *const HkLayerwiseRequest) -> u32;
        pub fn hk_layerwise_request_selection_score_count(r: *const HkLayerwiseRequest) -> usize;
        pub fn hk_layerwise_request_selection_score(
            r: *const HkLayerwiseRequest,
            i: usize,
            out: *mut f64,
        ) -> c_int;
        pub fn hk_layerwise_request_selection_fixed_count(r: *const HkLayerwiseRequest) -> usize;
        pub fn hk_layerwise_request_selection_fixed(r: *const HkLayerwiseRequest, i: usize) -> u32;
        pub fn hk_layerwise_request_selection_selected_count(r: *const HkLayerwiseRequest) -> usize;
        pub fn hk_layerwise_request_selection_selected(
            r: *const HkLayerwiseRequest,
            i: usize,
        ) -> u32;
        pub fn hk_layerwise_request_selection_historical_tokens(
            r: *const HkLayerwiseRequest,
        ) -> u64;
        pub fn hk_layerwise_request_selection_charged_tokens(r: *const HkLayerwiseRequest) -> u64;
        pub fn hk_layerwise_request_accepted_layer_count(r: *const HkLayerwiseRequest) -> usize;
        pub fn hk_layerwise_request_accepted_model_layer(
            r: *const HkLayerwiseRequest,
            i: usize,
        ) -> u32;
        pub fn hk_layerwise_request_accepted_stamp(
            r: *const HkLayerwiseRequest,
            i: usize,
            out: *mut MemoryStamp,
        ) -> c_int;
        pub fn hk_layerwise_request_accepted_block_count(
            r: *const HkLayerwiseRequest,
            i: usize,
        ) -> usize;
        pub fn hk_layerwise_request_accepted_block(
            r: *const HkLayerwiseRequest,
            i: usize,
            b: usize,
            out: *mut BlockCopies,
        ) -> c_int;
        pub fn hk_layerwise_request_accepted_resident_count(
            r: *const HkLayerwiseRequest,
            i: usize,
        ) -> usize;
        pub fn hk_layerwise_request_accepted_resident(
            r: *const HkLayerwiseRequest,
            i: usize,
            b: usize,
        ) -> u64;
        pub fn hk_layerwise_request_projected_native(
            r: *mut HkLayerwiseRequest,
            id: *const LayerwiseIdentity,
            begin: u64,
            q: *const NativeCurrentLayerQuery,
            scorer: Option<
                unsafe extern "C" fn(
                    *mut c_void,
                    *const HkRawStore,
                    *const NativeCurrentLayerQuery,
                    *const HkQueryScoreFrame,
                    *mut f64,
                    usize,
                ) -> c_int,
            >,
            scorer_user: *mut c_void,
        ) -> c_int;
    }
}

/// Copy the thread-local error string (None when the last call succeeded).
fn last_error() -> Option<String> {
    let p = unsafe { ffi::hk_last_error() };
    if p.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }
}

fn err_from_last() -> KvMemError {
    KvMemError(last_error().unwrap_or_else(|| "unknown error".to_owned()))
}

fn clear_error() {
    unsafe { ffi::hk_clear_error() };
}

/// Run `f` with a clean diagnostic slot and map a published error to `Err`.
fn checked(f: impl FnOnce()) -> Result<()> {
    clear_error();
    f();
    match last_error() {
        None => Ok(()),
        Some(e) => Err(KvMemError(e)),
    }
}

/// kvmem library/ABI version stamped by the shim.
pub fn version() -> String {
    let p = unsafe { ffi::hk_version() };
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// Host payload storage backing packed Q8 KV rows: `capacity_bytes` of RAM and,
/// when `spill_dir` is `Some`, a `spill_capacity_bytes` disk tier.
pub struct Storage {
    ptr: NonNull<HkStorage>,
}

impl Storage {
    pub fn new_ram(capacity_bytes: u64) -> Result<Self> {
        Self::new(capacity_bytes, None, 0)
    }

    pub fn new_tiered(
        capacity_bytes: u64,
        spill_dir: &Path,
        spill_capacity_bytes: u64,
    ) -> Result<Self> {
        Self::new(capacity_bytes, Some(spill_dir), spill_capacity_bytes)
    }

    fn new(capacity_bytes: u64, spill_dir: Option<&Path>, spill_capacity: u64) -> Result<Self> {
        let dir = match spill_dir {
            Some(d) => {
                let s = d.to_string_lossy().into_owned();
                Some(CString::new(s).map_err(|e| KvMemError(format!("spill dir: {e}")))?)
            }
            None => None,
        };
        clear_error();
        let ptr = unsafe {
            ffi::hk_storage_create(
                capacity_bytes,
                dir.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
                spill_capacity,
            )
        };
        NonNull::new(ptr)
            .map(|ptr| Self { ptr })
            .ok_or_else(err_from_last)
    }

    pub fn charged_bytes(&self) -> u64 {
        unsafe { ffi::hk_storage_charged_bytes(self.ptr.as_ptr()) }
    }

    pub fn disk_bytes(&self) -> u64 {
        unsafe { ffi::hk_storage_disk_bytes(self.ptr.as_ptr()) }
    }

    /// Idle reclamation of sealed cold payloads down to `resident_target` bytes.
    pub fn trim(&mut self, resident_target: u64) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_storage_trim(p, resident_target) })
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        unsafe { ffi::hk_storage_destroy(self.ptr.as_ptr()) };
    }
}

/// Block statistics + packed Q8 KV rows for one engine sequence.
///
/// The C++ object serializes internally; hipfire calls it from the serve
/// thread only, so `&self` mutating methods are sound for that usage.
pub struct RawStore {
    ptr: NonNull<HkRawStore>,
}

impl RawStore {
    pub fn new(cfg: RawStoreConfig, storage: Option<&Storage>) -> Result<Self> {
        clear_error();
        let ptr = unsafe {
            ffi::hk_raw_store_create(
                &cfg,
                storage.map_or(ptr::null_mut(), |s| s.ptr.as_ptr()),
            )
        };
        NonNull::new(ptr)
            .map(|ptr| Self { ptr })
            .ok_or_else(err_from_last)
    }

    pub fn ensure_blocks(&self, block_count: u32) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_raw_store_ensure_blocks(p, block_count) })
    }

    /// Feed `n` token-major F32 K rows (post-RMSNorm, pre-RoPE), `n_embd_k`
    /// floats per token.
    pub fn write_layer_mean_k(&self, pos0: u32, n: u32, il: u32, k: &[f32]) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_raw_store_write_layer_mean_k(p, pos0, n, il, k.as_ptr()) })
    }

    /// Feed `n` packed Q8 K rows exactly as laid out in the engine arena.
    pub fn write_layer_k_gpu(&self, pos0: u32, n: u32, il: u32, k: &[u8]) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_raw_store_write_layer_k_gpu(p, pos0, n, il, k.as_ptr()) })
    }

    /// Feed `n` packed Q8 V rows exactly as laid out in the engine arena.
    pub fn write_layer_v_gpu(&self, pos0: u32, n: u32, il: u32, v: &[u8]) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_raw_store_write_layer_v_gpu(p, pos0, n, il, v.as_ptr()) })
    }

    pub fn copy_k_gpu(&self, block: u32, il: u32, out: &mut [u8], n: u32) -> bool {
        unsafe { ffi::hk_raw_store_copy_k_gpu(self.ptr.as_ptr(), block, il, out.as_mut_ptr(), n) != 0 }
    }

    pub fn copy_v_gpu(&self, block: u32, il: u32, out: &mut [u8], n: u32) -> bool {
        unsafe { ffi::hk_raw_store_copy_v_gpu(self.ptr.as_ptr(), block, il, out.as_mut_ptr(), n) != 0 }
    }

    pub fn mean_k(&self, block: u32, il: u32, out: &mut [f32]) {
        unsafe { ffi::hk_raw_store_mean_k(self.ptr.as_ptr(), block, il, out.as_mut_ptr()) };
    }

    pub fn mean_k_tokens(&self, block: u32, il: u32) -> u32 {
        unsafe { ffi::hk_raw_store_mean_k_tokens(self.ptr.as_ptr(), block, il) }
    }

    pub fn n_tokens(&self, block: u32) -> u32 {
        unsafe { ffi::hk_raw_store_n_tokens(self.ptr.as_ptr(), block) }
    }

    pub fn has_k_gpu(&self, block: u32, il: u32, n: u32) -> bool {
        unsafe { ffi::hk_raw_store_has_k_gpu(self.ptr.as_ptr(), block, il, n) != 0 }
    }

    pub fn has_v_gpu(&self, block: u32, il: u32, n: u32) -> bool {
        unsafe { ffi::hk_raw_store_has_v_gpu(self.ptr.as_ptr(), block, il, n) != 0 }
    }

    /// Drop everything at/after `token_pos`, including a partial trailing block.
    pub fn truncate_to(&self, token_pos: u32) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_raw_store_truncate_to(p, token_pos) })
    }

    pub fn clear(&self) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_raw_store_clear(p) })
    }

    pub fn allocated_bytes(&self) -> u64 {
        unsafe { ffi::hk_raw_store_allocated_bytes(self.ptr.as_ptr()) }
    }

    pub fn payload_bytes(&self) -> u64 {
        unsafe { ffi::hk_raw_store_payload_bytes(self.ptr.as_ptr()) }
    }

    /// Partial-tail mean-sum checkpoint (needed before speculative draft rows
    /// that may be rejected). `None` when the position is block-aligned.
    pub fn mean_checkpoint(&self, token_pos: u32) -> Result<Option<Vec<f32>>> {
        let p = self.ptr.as_ptr();
        let len = unsafe { ffi::hk_raw_store_mean_checkpoint(p, token_pos, ptr::null_mut(), 0) };
        if len == 0 {
            return match last_error() {
                None => Ok(None),
                Some(e) => Err(KvMemError(e)),
            };
        }
        let mut buf = vec![0.0f32; len];
        clear_error();
        let written =
            unsafe { ffi::hk_raw_store_mean_checkpoint(p, token_pos, buf.as_mut_ptr(), len) };
        if written == len {
            Ok(Some(buf))
        } else {
            Err(err_from_last())
        }
    }

    pub fn restore_mean_checkpoint(&self, token_pos: u32, state: &[f32]) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe {
            ffi::hk_raw_store_restore_mean_checkpoint(p, token_pos, state.as_ptr(), state.len())
        })
    }
}

impl Drop for RawStore {
    fn drop(&mut self) {
        unsafe { ffi::hk_raw_store_destroy(self.ptr.as_ptr()) };
    }
}

/// Block repository + working-set selection for one engine sequence.
pub struct KvStore {
    ptr: NonNull<HkKvStore>,
}

impl KvStore {
    pub fn new(cfg: &KvStoreConfig) -> Result<Self> {
        clear_error();
        let ptr = unsafe { ffi::hk_kv_store_create(cfg) };
        NonNull::new(ptr)
            .map(|ptr| Self { ptr })
            .ok_or_else(err_from_last)
    }

    pub fn register_append(&self, n_new_tokens: u32) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_kv_store_register_append(p, n_new_tokens) })
    }

    pub fn truncate_to(&self, token_pos: u32) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_kv_store_truncate_to(p, token_pos) })
    }

    pub fn block_count(&self) -> u32 {
        unsafe { ffi::hk_kv_store_block_count(self.ptr.as_ptr()) }
    }

    pub fn total_tokens(&self) -> u32 {
        unsafe { ffi::hk_kv_store_total_tokens(self.ptr.as_ptr()) }
    }

    pub fn select_budget_tokens(&self) -> u32 {
        unsafe { ffi::hk_kv_store_select_budget_tokens(self.ptr.as_ptr()) }
    }

    pub fn set_runtime_select_budget(&self, tokens: u32) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_kv_store_set_runtime_select_budget(p, tokens) })
    }

    pub fn pick_topk_blocks(&self, mandatory: &[u32]) -> Result<Vec<u32>> {
        let p = self.ptr.as_ptr();
        pick(|out, cap| unsafe {
            ffi::hk_kv_store_pick_topk_blocks(p, mandatory.as_ptr(), mandatory.len() as u32, out, cap)
        })
    }

    pub fn pick_prefill_pressure_blocks(&self, mandatory: &[u32]) -> Result<Vec<u32>> {
        let p = self.ptr.as_ptr();
        pick(|out, cap| unsafe {
            ffi::hk_kv_store_pick_prefill_pressure_blocks(
                p,
                mandatory.as_ptr(),
                mandatory.len() as u32,
                out,
                cap,
            )
        })
    }

    pub fn set_retrieval_scores(&self, scores: &[f64]) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe {
            ffi::hk_kv_store_set_retrieval_scores(p, scores.as_ptr(), scores.len() as u32)
        })
    }

    /// Open an LRFU retrieval round. No-op (and error-free) when lrfu is off, so
    /// the default Retrieval path is unaffected. Must precede set_retrieval_scores
    /// when lrfu is on; pair with commit_retrieval_round after pick_topk_blocks.
    pub fn begin_retrieval_round(&self) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_kv_store_begin_retrieval_round(p) })
    }

    /// Close the round and reward the picked blocks. Returns Ok(false) when lrfu
    /// is off (nothing committed); Ok(true) when a round was committed.
    pub fn commit_retrieval_round(&self) -> Result<bool> {
        let p = self.ptr.as_ptr();
        let mut committed = false;
        checked(|| unsafe {
            committed = ffi::hk_kv_store_commit_retrieval_round(p) != 0;
        })?;
        Ok(committed)
    }

    pub fn prefill_needs_offload(
        &self,
        resident_tokens: u32,
        incoming_tokens: u32,
        pool_tokens: u32,
    ) -> bool {
        unsafe {
            ffi::hk_kv_store_prefill_needs_offload(
                self.ptr.as_ptr(),
                resident_tokens,
                incoming_tokens,
                pool_tokens,
            ) != 0
        }
    }

    pub fn gpu_high_watermark_tokens(&self, pool_tokens: u32) -> u32 {
        unsafe { ffi::hk_kv_store_gpu_high_watermark_tokens(self.ptr.as_ptr(), pool_tokens) }
    }

    pub fn gpu_low_watermark_tokens(&self, pool_tokens: u32) -> u32 {
        unsafe { ffi::hk_kv_store_gpu_low_watermark_tokens(self.ptr.as_ptr(), pool_tokens) }
    }

    pub fn allocated_bytes(&self) -> u64 {
        unsafe { ffi::hk_kv_store_allocated_bytes(self.ptr.as_ptr()) }
    }
}

impl Drop for KvStore {
    fn drop(&mut self) {
        unsafe { ffi::hk_kv_store_destroy(self.ptr.as_ptr()) };
    }
}

/// Two-call selection helper: `pick` writes the required count on out=NULL.
fn pick(mut f: impl FnMut(*mut u32, u32) -> u32) -> Result<Vec<u32>> {
    clear_error();
    let needed = f(ptr::null_mut(), 0);
    if needed == u32::MAX {
        return Err(err_from_last());
    }
    if needed == 0 {
        return match last_error() {
            None => Ok(Vec::new()),
            Some(e) => Err(KvMemError(e)),
        };
    }
    let mut ids = vec![0u32; needed as usize];
    clear_error();
    let written = f(ids.as_mut_ptr(), needed);
    if written != needed {
        return Err(KvMemError(format!(
            "selection count changed between calls ({needed} -> {written})"
        )));
    }
    if let Some(e) = last_error() {
        return Err(KvMemError(e));
    }
    Ok(ids)
}

/// Score every historical block against a post-RMSNorm/pre-RoPE query sum.
///
/// `sums` is `layers * query_heads * head_dim` FP32 elements (layer-major);
/// `token_counts` has one entry per layer and must match the store's
/// `n_layer`. Returns one score per block.
pub fn mean_key_retrieval_scores(
    keys: &RawStore,
    blocks: u32,
    sums: &[f32],
    token_counts: &[u32],
    layers: u32,
    head_dim: u32,
    query_heads: u32,
    kv_heads: u32,
    aggregation: GqaAggregation,
) -> Result<Vec<f64>> {
    let need = layers as usize * query_heads as usize * head_dim as usize;
    if sums.len() < need {
        return Err(KvMemError(format!(
            "query sums too short: {} < {need}",
            sums.len()
        )));
    }
    if token_counts.len() < layers as usize {
        return Err(KvMemError(format!(
            "token_counts too short: {} < {layers}",
            token_counts.len()
        )));
    }
    let mut out = vec![0.0f64; blocks as usize];
    clear_error();
    let rc = unsafe {
        ffi::hk_mean_key_retrieval_scores(
            keys.ptr.as_ptr(),
            blocks,
            sums.as_ptr(),
            token_counts.as_ptr(),
            layers,
            head_dim,
            query_heads,
            kv_heads,
            aggregation.as_u32(),
            out.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(err_from_last());
    }
    Ok(out)
}

/// Query-span page-softmax retrieval scores (one per block).
///
/// `q_data` is `layers * rows * query_heads * head_dim` FP32 post-RMSNorm /
/// pre-RoPE Q rows in layer/token/head/channel order; `layer_rows[l]` is the
/// number of present rows for layer `l` (`<= rows`). The accepted index is
/// frozen from `accepted_through` / `block_tokens` / `budget_blocks` /
/// `sink_blocks` / `recent_blocks`; `blocks` must equal the resulting frame
/// block count. Each block's softmax mass is averaged over the query rows
/// (FP16-rounded operands; `attention_scale == 0` selects `1/sqrt(head_dim)`).
/// Selection math only: it mutates no payload, frontier or residency.
#[allow(clippy::too_many_arguments)]
pub fn page_softmax_scores(
    keys: &RawStore,
    blocks: u32,
    accepted_through: u64,
    block_tokens: u32,
    budget_blocks: u32,
    sink_blocks: u32,
    recent_blocks: u32,
    q_data: &[f32],
    layer_rows: &[u32],
    layers: u32,
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    attention_scale: f32,
    aggregation: GqaAggregation,
) -> Result<Vec<f64>> {
    let need = layers as usize * rows as usize * query_heads as usize * head_dim as usize;
    if q_data.len() < need {
        return Err(KvMemError(format!(
            "query rows too short: {} < {need}",
            q_data.len()
        )));
    }
    if layer_rows.len() < layers as usize {
        return Err(KvMemError(format!(
            "layer_rows too short: {} < {layers}",
            layer_rows.len()
        )));
    }
    let mut out = vec![0.0f64; blocks as usize];
    clear_error();
    let rc = unsafe {
        ffi::hk_page_softmax_scores(
            keys.ptr.as_ptr(),
            blocks,
            accepted_through,
            block_tokens,
            budget_blocks,
            sink_blocks,
            recent_blocks,
            q_data.as_ptr(),
            layer_rows.as_ptr(),
            layers,
            rows,
            query_heads,
            kv_heads,
            head_dim,
            attention_scale,
            aggregation.as_u32(),
            out.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(err_from_last());
    }
    Ok(out)
}

// ── layerwise native payload foundation: safe wrappers ──
//
// These wrap the portable KVMem layerwise building blocks the hipfire packed-page
// driver orchestrates. They are additive: the legacy residency path never calls
// them, and none mutates a RawStore's statistics or the engine's resident view.
// Like the other handles, they own non-Sync C++ state and are neither Send nor Sync.

/// Backend-owned packed payload layout (`kvmem::PayloadLayout`).
pub struct PayloadLayout {
    ptr: NonNull<HkPayloadLayout>,
}

impl PayloadLayout {
    /// Build the hipfire Q8 adapter layout from a packed [`RawStoreConfig`]: one
    /// {Key,Value} token-row plane per attention layer, scales inline. `layers`
    /// are the FullAttention statistics ordinals.
    pub fn from_raw(cfg: &RawStoreConfig, format: &str, layers: &[u32]) -> Result<Self> {
        let fmt = CString::new(format).map_err(|e| KvMemError(format!("format: {e}")))?;
        clear_error();
        let ptr = unsafe {
            ffi::hk_payload_layout_from_raw(cfg, fmt.as_ptr(), layers.as_ptr(), layers.len() as u32)
        };
        NonNull::new(ptr).map(|ptr| Self { ptr }).ok_or_else(err_from_last)
    }

    /// Build from an explicit plane list (validated by the core layout rules).
    pub fn create(
        format: &str,
        version: u32,
        block_tokens: u32,
        record_bytes: u64,
        alignment: u64,
        planes: &[PayloadPlane],
    ) -> Result<Self> {
        let fmt = CString::new(format).map_err(|e| KvMemError(format!("format: {e}")))?;
        clear_error();
        let ptr = unsafe {
            ffi::hk_payload_layout_create(
                fmt.as_ptr(),
                version,
                block_tokens,
                record_bytes,
                alignment,
                planes.as_ptr(),
                planes.len() as u32,
            )
        };
        NonNull::new(ptr).map(|ptr| Self { ptr }).ok_or_else(err_from_last)
    }

    pub fn record_bytes(&self) -> u64 {
        unsafe { ffi::hk_payload_layout_record_bytes(self.ptr.as_ptr()) }
    }

    pub fn block_tokens(&self) -> u32 {
        unsafe { ffi::hk_payload_layout_block_tokens(self.ptr.as_ptr()) }
    }

    pub fn plane_count(&self) -> u32 {
        unsafe { ffi::hk_payload_layout_plane_count(self.ptr.as_ptr()) }
    }

    pub fn plane(&self, i: u32) -> Result<PayloadPlane> {
        let mut out = PayloadPlane {
            layer: 0,
            kind: 0,
            offset: 0,
            bytes: 0,
            alignment: 0,
            token_stride: 0,
            rank: 0,
            head_begin: 0,
            head_count: 0,
        };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_payload_layout_plane(p, i, &mut out); })?;
        Ok(out)
    }
}

impl Drop for PayloadLayout {
    fn drop(&mut self) {
        unsafe { ffi::hk_payload_layout_destroy(self.ptr.as_ptr()) };
    }
}

/// Backend identity + layout + capabilities (`kvmem::MemoryDescriptor`).
pub struct MemoryDescriptor {
    ptr: NonNull<HkMemoryDescriptor>,
}

impl MemoryDescriptor {
    pub fn new(
        identity: &BackendIdentity,
        layout: &PayloadLayout,
        capabilities: &MemoryCapabilities,
    ) -> Result<Self> {
        let (_b, _r, _m, _s, id) = identity.ffi()?;
        clear_error();
        let ptr = unsafe {
            ffi::hk_memory_descriptor_create(&id, layout.ptr.as_ptr(), capabilities)
        };
        NonNull::new(ptr).map(|ptr| Self { ptr }).ok_or_else(err_from_last)
    }
}

impl Drop for MemoryDescriptor {
    fn drop(&mut self) {
        unsafe { ffi::hk_memory_descriptor_destroy(self.ptr.as_ptr()) };
    }
}

/// A sealed host record: the restore source for [`KvRegions::read`].
pub struct HostRecord {
    ptr: NonNull<HkHostRecord>,
}

impl HostRecord {
    /// Copy `bytes` into the storage account as a new record.
    pub fn allocate(storage: &Storage, bytes: &[u8]) -> Result<Self> {
        clear_error();
        let ptr = unsafe {
            ffi::hk_host_record_allocate(storage.ptr.as_ptr(), bytes.as_ptr(), bytes.len())
        };
        NonNull::new(ptr).map(|ptr| Self { ptr }).ok_or_else(err_from_last)
    }

    /// Make the record immutable (and eligible for tiering).
    pub fn seal(&self) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_host_record_seal(p) })
    }

    pub fn size(&self) -> usize {
        unsafe { ffi::hk_host_record_size(self.ptr.as_ptr()) }
    }

    pub fn read(&self, offset: usize, dest: &mut [u8]) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_host_record_read(p, offset, dest.as_mut_ptr(), dest.len()); })
    }

    /// Borrow a deferred read source (does not pin the RAM tier).
    pub fn source(&self) -> Result<HostSource> {
        clear_error();
        let ptr = unsafe { ffi::hk_host_record_source(self.ptr.as_ptr()) };
        NonNull::new(ptr).map(|ptr| HostSource { ptr }).ok_or_else(err_from_last)
    }
}

impl Drop for HostRecord {
    fn drop(&mut self) {
        unsafe { ffi::hk_host_record_destroy(self.ptr.as_ptr()) };
    }
}

/// A deferred read source borrowed from a [`HostRecord`].
pub struct HostSource {
    ptr: NonNull<HkHostSource>,
}

impl HostSource {
    pub fn size(&self) -> usize {
        unsafe { ffi::hk_host_source_size(self.ptr.as_ptr()) }
    }
}

impl Drop for HostSource {
    fn drop(&mut self) {
        unsafe { ffi::hk_host_source_destroy(self.ptr.as_ptr()) };
    }
}

/// One model layer's planes packed into a compact extent
/// (`kvmem::NativeKvLayerRegions`).
pub struct KvRegions {
    ptr: NonNull<HkKvRegions>,
}

impl KvRegions {
    /// `rank == None` selects every rank of the layer; otherwise only that rank.
    pub fn new(layout: &PayloadLayout, layer: u32, rank: Option<u32>) -> Result<Self> {
        let (has_rank, rank) = match rank {
            Some(r) => (1, r),
            None => (0, 0),
        };
        clear_error();
        let ptr =
            unsafe { ffi::hk_kv_regions_create(layout.ptr.as_ptr(), layer, has_rank, rank) };
        NonNull::new(ptr).map(|ptr| Self { ptr }).ok_or_else(err_from_last)
    }

    pub fn packed_bytes(&self) -> u64 {
        unsafe { ffi::hk_kv_regions_packed_bytes(self.ptr.as_ptr()) }
    }

    pub fn plane_bytes(&self) -> u64 {
        unsafe { ffi::hk_kv_regions_plane_bytes(self.ptr.as_ptr()) }
    }

    pub fn record_bytes(&self) -> u64 {
        unsafe { ffi::hk_kv_regions_record_bytes(self.ptr.as_ptr()) }
    }

    pub fn region_count(&self) -> u32 {
        unsafe { ffi::hk_kv_regions_region_count(self.ptr.as_ptr()) }
    }

    pub fn region(&self, i: u32) -> Result<(PayloadPlane, u64)> {
        let mut plane = PayloadPlane {
            layer: 0,
            kind: 0,
            offset: 0,
            bytes: 0,
            alignment: 0,
            token_stride: 0,
            rank: 0,
            head_begin: 0,
            head_count: 0,
        };
        let mut packed_offset = 0u64;
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_kv_regions_region(p, i, &mut plane, &mut packed_offset); })?;
        Ok((plane, packed_offset))
    }

    /// Copy a bounded slice of the packed layer out of a sealed host source into
    /// caller-owned staging (zero-fills interplane padding). `max_read_bytes`
    /// bounds each underlying I/O; cold sources stay cold.
    pub fn read(
        &self,
        source: &HostSource,
        packed_offset: u64,
        dest: &mut [u8],
        max_read_bytes: usize,
    ) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe {
            ffi::hk_kv_regions_read(
                p,
                source.ptr.as_ptr(),
                packed_offset,
                dest.as_mut_ptr(),
                dest.len(),
                max_read_bytes,
            );
        })
    }
}

impl Drop for KvRegions {
    fn drop(&mut self) {
        unsafe { ffi::hk_kv_regions_destroy(self.ptr.as_ptr()) };
    }
}

/// One canonical block's opaque packed record (`kvmem::NativeKvPayload`).
pub struct NativePayload {
    ptr: NonNull<HkNativePayload>,
}

impl NativePayload {
    /// Gather a [`RawStore`]'s packed K/V for one block into a canonical record.
    pub fn export(
        raw: &RawStore,
        backend: &BackendIdentity,
        layout: &PayloadLayout,
        block: &BlockDescriptor,
    ) -> Result<Self> {
        let (_b, _r, _m, _s, id) = backend.ffi()?;
        clear_error();
        let ptr = unsafe {
            ffi::hk_native_payload_export(raw.ptr.as_ptr(), &id, layout.ptr.as_ptr(), block)
        };
        NonNull::new(ptr).map(|ptr| Self { ptr }).ok_or_else(err_from_last)
    }

    /// Wrap caller-supplied packed bytes (length must equal the layout record).
    pub fn create(
        backend: &BackendIdentity,
        layout: &PayloadLayout,
        block: &BlockDescriptor,
        bytes: &[u8],
    ) -> Result<Self> {
        let (_b, _r, _m, _s, id) = backend.ffi()?;
        clear_error();
        let ptr = unsafe {
            ffi::hk_native_payload_create(&id, layout.ptr.as_ptr(), block, bytes.as_ptr(), bytes.len())
        };
        NonNull::new(ptr).map(|ptr| Self { ptr }).ok_or_else(err_from_last)
    }

    /// Borrow the packed bytes (valid until this handle is dropped/mutated).
    pub fn bytes(&self) -> &[u8] {
        unsafe {
            let p = ffi::hk_native_payload_bytes(self.ptr.as_ptr());
            let n = ffi::hk_native_payload_size(self.ptr.as_ptr());
            if p.is_null() || n == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(p, n)
            }
        }
    }

    pub fn size(&self) -> usize {
        unsafe { ffi::hk_native_payload_size(self.ptr.as_ptr()) }
    }

    pub fn block(&self) -> Result<BlockDescriptor> {
        let mut out = BlockDescriptor {
            session: SessionIdentity { id: 0, incarnation: 0 },
            id: 0,
            original_start: 0,
            valid_tokens: 0,
            content_version: 0,
            statistics_version: 0,
        };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_native_payload_block(p, &mut out); })?;
        Ok(out)
    }

    /// Write the packed K/V back into a [`RawStore`] (statistics untouched).
    pub fn import(
        &self,
        raw: &RawStore,
        backend: &BackendIdentity,
        layout: &PayloadLayout,
    ) -> Result<()> {
        let (_b, _r, _m, _s, id) = backend.ffi()?;
        let p = self.ptr.as_ptr();
        checked(|| unsafe {
            ffi::hk_native_payload_import(raw.ptr.as_ptr(), &id, layout.ptr.as_ptr(), p);
        })
    }
}

impl Drop for NativePayload {
    fn drop(&mut self) {
        unsafe { ffi::hk_native_payload_destroy(self.ptr.as_ptr()) };
    }
}

/// Bounded write-through store of canonical packed records
/// (`kvmem::NativeKvArchive`).
pub struct NativeArchive {
    ptr: NonNull<HkNativeArchive>,
}

impl NativeArchive {
    pub fn new(
        backend: &BackendIdentity,
        session: SessionIdentity,
        layout: &PayloadLayout,
        capacity_bytes: u64,
    ) -> Result<Self> {
        let (_b, _r, _m, _s, id) = backend.ffi()?;
        clear_error();
        let ptr = unsafe {
            ffi::hk_native_archive_create(&id, session, layout.ptr.as_ptr(), capacity_bytes)
        };
        NonNull::new(ptr).map(|ptr| Self { ptr }).ok_or_else(err_from_last)
    }

    /// Copy a payload into the archive (the caller keeps its handle). Stale or
    /// foreign writes are rejected.
    pub fn put(&self, payload: &NativePayload) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_native_archive_put(p, payload.ptr.as_ptr()); })
    }

    /// Fetch a NEW caller-owned handle for the stored record matching `expected`.
    pub fn get(&self, expected: &BlockDescriptor) -> Result<NativePayload> {
        clear_error();
        let ptr = unsafe { ffi::hk_native_archive_get(self.ptr.as_ptr(), expected) };
        NonNull::new(ptr).map(|ptr| NativePayload { ptr }).ok_or_else(err_from_last)
    }

    pub fn erase(&self, block_id: u64) {
        unsafe { ffi::hk_native_archive_erase(self.ptr.as_ptr(), block_id) };
    }

    pub fn bytes(&self) -> u64 {
        unsafe { ffi::hk_native_archive_bytes(self.ptr.as_ptr()) }
    }
}

impl Drop for NativeArchive {
    fn drop(&mut self) {
        unsafe { ffi::hk_native_archive_destroy(self.ptr.as_ptr()) };
    }
}

// ── layerwise catalog + planning: safe wrappers ──

/// Immutable pure-planning catalog over one canonical accepted history
/// (`kvmem::LayerwiseMemoryCatalog`). Validates that every target layer shares
/// one history/catalog and a consistent resident extent; plans per layer.
pub struct MemoryCatalog {
    ptr: NonNull<HkMemoryCatalog>,
}

impl MemoryCatalog {
    pub fn new(canonical: &MemoryDescriptor, layers: &[LayerMemorySnapshot]) -> Result<Self> {
        let hk: Vec<HkLayerMemorySnapshot> = layers
            .iter()
            .map(|l| HkLayerMemorySnapshot {
                model_layer: l.model_layer,
                memory: HkMemorySnapshot {
                    stamp: l.memory.stamp,
                    blocks: l.memory.blocks.as_ptr(),
                    block_count: l.memory.blocks.len(),
                    resident: l.memory.resident.as_ptr(),
                    resident_count: l.memory.resident.len(),
                },
            })
            .collect();
        clear_error();
        let ptr = unsafe {
            ffi::hk_memory_catalog_create(canonical.ptr.as_ptr(), hk.as_ptr(), hk.len())
        };
        NonNull::new(ptr).map(|ptr| Self { ptr }).ok_or_else(err_from_last)
    }

    pub fn layer_count(&self) -> usize {
        unsafe { ffi::hk_memory_catalog_layer_count(self.ptr.as_ptr()) }
    }
    pub fn model_layer(&self, i: usize) -> u32 {
        unsafe { ffi::hk_memory_catalog_model_layer(self.ptr.as_ptr(), i) }
    }
    pub fn snapshot_stamp(&self, model_layer: u32) -> Result<MemoryStamp> {
        let mut out: MemoryStamp = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_memory_catalog_snapshot_stamp(p, model_layer, &mut out); })?;
        Ok(out)
    }
    pub fn snapshot_block_count(&self, model_layer: u32) -> usize {
        unsafe { ffi::hk_memory_catalog_snapshot_block_count(self.ptr.as_ptr(), model_layer) }
    }
    pub fn snapshot_block(&self, model_layer: u32, i: usize) -> Result<BlockCopies> {
        let mut out: BlockCopies = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_memory_catalog_snapshot_block(p, model_layer, i, &mut out); })?;
        Ok(out)
    }
    pub fn snapshot_resident_count(&self, model_layer: u32) -> usize {
        unsafe { ffi::hk_memory_catalog_snapshot_resident_count(self.ptr.as_ptr(), model_layer) }
    }
    pub fn snapshot_resident(&self, model_layer: u32, i: usize) -> u64 {
        unsafe { ffi::hk_memory_catalog_snapshot_resident(self.ptr.as_ptr(), model_layer, i) }
    }

    /// Pure per-layer plan after that layer's Q exists.
    pub fn plan_layer(&self, sel: &LayerSelection) -> Result<LayerPlan> {
        let hk = sel.ffi();
        let mut out: *mut HkLayerPlan = ptr::null_mut();
        let p = self.ptr.as_ptr();
        clear_error();
        let rc = unsafe { ffi::hk_memory_catalog_plan_layer(p, &hk, &mut out) };
        if rc != 0 {
            return Err(err_from_last());
        }
        NonNull::new(out).map(|ptr| LayerPlan { ptr }).ok_or_else(err_from_last)
    }

    /// Bulk validation of already-known selections (one entry per target layer).
    pub fn plan(&self, sels: &[LayerSelection]) -> Result<LayerwisePlan> {
        let hk: Vec<HkLayerSelection> = sels.iter().map(|s| s.ffi()).collect();
        let mut out: *mut HkLayerwisePlan = ptr::null_mut();
        let p = self.ptr.as_ptr();
        clear_error();
        let rc = unsafe { ffi::hk_memory_catalog_plan(p, hk.as_ptr(), hk.len(), &mut out) };
        if rc != 0 {
            return Err(err_from_last());
        }
        NonNull::new(out).map(|ptr| LayerwisePlan { ptr }).ok_or_else(err_from_last)
    }
}

impl Drop for MemoryCatalog {
    fn drop(&mut self) {
        unsafe { ffi::hk_memory_catalog_destroy(self.ptr.as_ptr()) };
    }
}

/// One layer's reduced working-set plan (`kvmem::LayerWorkingSetPlan`). The
/// per-layer transfer regions are obtained separately via [`LayerPlan::payload`].
pub struct LayerPlan {
    ptr: NonNull<HkLayerPlan>,
}

impl LayerPlan {
    pub fn model_layer(&self) -> u32 {
        unsafe { ffi::hk_layer_plan_model_layer(self.ptr.as_ptr()) }
    }
    pub fn restore_bytes(&self) -> u64 {
        unsafe { ffi::hk_layer_plan_restore_bytes(self.ptr.as_ptr()) }
    }
    pub fn spill_bytes(&self) -> u64 {
        unsafe { ffi::hk_layer_plan_spill_bytes(self.ptr.as_ptr()) }
    }
    pub fn device_tokens_required(&self) -> u64 {
        unsafe { ffi::hk_layer_plan_device_tokens_required(self.ptr.as_ptr()) }
    }
    pub fn host_payload_bytes_required(&self) -> u64 {
        unsafe { ffi::hk_layer_plan_host_payload_bytes_required(self.ptr.as_ptr()) }
    }
    pub fn next_stamp(&self) -> Result<MemoryStamp> {
        let mut out: MemoryStamp = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layer_plan_next_stamp(p, &mut out); })?;
        Ok(out)
    }
    pub fn next_visible_tokens(&self) -> u64 {
        unsafe { ffi::hk_layer_plan_next_visible_tokens(self.ptr.as_ptr()) }
    }
    pub fn next_block_count(&self) -> usize {
        unsafe { ffi::hk_layer_plan_next_block_count(self.ptr.as_ptr()) }
    }
    pub fn next_block(&self, i: usize) -> Result<ViewBlock> {
        let mut out: ViewBlock = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layer_plan_next_block(p, i, &mut out); })?;
        Ok(out)
    }
    pub fn restore_count(&self) -> usize {
        unsafe { ffi::hk_layer_plan_restore_count(self.ptr.as_ptr()) }
    }
    pub fn restore_block(&self, i: usize) -> Result<BlockDescriptor> {
        let mut out: BlockDescriptor = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layer_plan_restore_block(p, i, &mut out); })?;
        Ok(out)
    }
    pub fn spill_count(&self) -> usize {
        unsafe { ffi::hk_layer_plan_spill_count(self.ptr.as_ptr()) }
    }
    pub fn spill_block(&self, i: usize) -> Result<BlockDescriptor> {
        let mut out: BlockDescriptor = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layer_plan_spill_block(p, i, &mut out); })?;
        Ok(out)
    }
    pub fn evict_count(&self) -> usize {
        unsafe { ffi::hk_layer_plan_evict_count(self.ptr.as_ptr()) }
    }
    pub fn evict_block(&self, i: usize) -> Result<BlockDescriptor> {
        let mut out: BlockDescriptor = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layer_plan_evict_block(p, i, &mut out); })?;
        Ok(out)
    }
    pub fn mandatory_count(&self) -> usize {
        unsafe { ffi::hk_layer_plan_mandatory_count(self.ptr.as_ptr()) }
    }
    pub fn mandatory(&self, i: usize) -> u64 {
        unsafe { ffi::hk_layer_plan_mandatory(self.ptr.as_ptr(), i) }
    }
    /// A NEW caller-owned [`KvRegions`] handle for this layer's transfer planes.
    pub fn payload(&self) -> Result<KvRegions> {
        let mut out: *mut HkKvRegions = ptr::null_mut();
        let p = self.ptr.as_ptr();
        clear_error();
        let rc = unsafe { ffi::hk_layer_plan_payload(p, &mut out) };
        if rc != 0 {
            return Err(err_from_last());
        }
        NonNull::new(out).map(|ptr| KvRegions { ptr }).ok_or_else(err_from_last)
    }
}

impl Drop for LayerPlan {
    fn drop(&mut self) {
        unsafe { ffi::hk_layer_plan_destroy(self.ptr.as_ptr()) };
    }
}

/// Bulk layerwise plan (`kvmem::LayerwiseWorkingSetPlan`): one reduced plan per
/// target layer plus the shared canonical archive accounting.
pub struct LayerwisePlan {
    ptr: NonNull<HkLayerwisePlan>,
}

impl LayerwisePlan {
    pub fn layer_count(&self) -> usize {
        unsafe { ffi::hk_layerwise_plan_layer_count(self.ptr.as_ptr()) }
    }
    pub fn history(&self) -> Result<MemoryStamp> {
        let mut out: MemoryStamp = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_plan_history(p, &mut out); })?;
        Ok(out)
    }
    pub fn host_payload_bytes_required(&self) -> u64 {
        unsafe { ffi::hk_layerwise_plan_host_payload_bytes_required(self.ptr.as_ptr()) }
    }
    /// A NEW caller-owned [`LayerPlan`] handle for target layer `i`.
    pub fn layer(&self, i: usize) -> Result<LayerPlan> {
        let mut out: *mut HkLayerPlan = ptr::null_mut();
        let p = self.ptr.as_ptr();
        clear_error();
        let rc = unsafe { ffi::hk_layerwise_plan_layer(p, i, &mut out) };
        if rc != 0 {
            return Err(err_from_last());
        }
        NonNull::new(out).map(|ptr| LayerPlan { ptr }).ok_or_else(err_from_last)
    }
}

impl Drop for LayerwisePlan {
    fn drop(&mut self) {
        unsafe { ffi::hk_layerwise_plan_destroy(self.ptr.as_ptr()) };
    }
}

// ── layerwise Query request state machine: safe wrapper ──

/// Owner state behind the C source-revision callback. Boxed so its heap address
/// is stable across moves of the request; freed only after the C++ request.
struct RevisionState {
    f: Box<dyn FnMut() -> MemoryStamp>,
}

/// C trampoline: prove the owner's current frozen index stamp. A panicking owner
/// cannot prove the stamp, so return an invalid (zeroed) stamp and let the core
/// revision check fail closed (Stale) rather than unwind across the C ABI.
unsafe extern "C" fn revision_trampoline(user: *mut c_void) -> MemoryStamp {
    let state = &mut *(user as *mut RevisionState);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (state.f)())) {
        Ok(stamp) => stamp,
        Err(_) => std::mem::zeroed(),
    }
}

/// Borrow a C array as a slice; a zero count yields an empty slice even when the
/// pointer is null (which `slice::from_raw_parts` would otherwise reject).
unsafe fn slice_or_empty<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

/// C trampoline for the owner's native scorer. The closure captures its own
/// statistics/device handle, so the borrowed `keys` pointer is not used here. A
/// panicking or failing scorer (or a wrong score extent) returns HK_ERR and the
/// request aborts fail-closed rather than unwinding across the C ABI.
unsafe extern "C" fn scorer_trampoline<F>(
    user: *mut c_void,
    _keys: *const HkRawStore,
    tile: *const NativeCurrentLayerQuery,
    frame: *const HkQueryScoreFrame,
    out: *mut f64,
    cap: usize,
) -> c_int
where
    F: FnMut(&NativeCurrentLayerQuery, &QueryScoreFrame<'_>) -> Result<Vec<f64>>,
{
    let f = &mut *(user as *mut F);
    let tile = &*tile;
    let cf = &*frame;
    let view = QueryScoreFrame {
        starts: slice_or_empty(cf.starts, cf.start_count),
        counts: slice_or_empty(cf.counts, cf.count_count),
        score_mask: slice_or_empty(cf.score_mask, cf.score_mask_count),
        transport_mask: slice_or_empty(cf.transport_mask, cf.transport_mask_count),
        accepted_through: cf.accepted_through,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(tile, &view)));
    match result {
        Ok(Ok(scores)) if scores.len() == cap => {
            std::ptr::copy_nonoverlapping(scores.as_ptr(), out, cap);
            0 // HK_OK
        }
        _ => -1, // HK_ERR
    }
}

/// An engine-safe-boundary transaction mirroring `kvmem::LayerwiseQueryRequest`.
///
/// It connects real projection reports to the qualified Mean-K scorer/picker and
/// the native layer plan. It does NOT execute tensors, copy GPU bytes, fence
/// streams or publish state — those stay with the hipfire driver and must precede
/// each report. The `catalog`/`policy`/`keys` handles are borrowed and must
/// outlive the request; `revision` proves the frozen index stamp at each native
/// boundary. Like the other wrappers this type is deliberately neither `Send`
/// nor `Sync` (the serve loop serializes the C ABI).
pub struct LayerwiseRequest {
    ptr: NonNull<HkLayerwiseRequest>,
    // Ownership anchor for the C SourceRevision trampoline's user pointer: the
    // C++ request reads it through `revision_trampoline`, never via this field,
    // and it is freed only after the C++ request is destroyed (see Drop).
    #[allow(dead_code)]
    revision: Box<RevisionState>,
}

impl LayerwiseRequest {
    /// Admit a frozen layerwise Query transaction. `specs` has one entry per
    /// target layer (matching the catalog); `native_history_tokens` is the
    /// already-resident native suffix the driver will append after acceptance.
    pub fn new(
        identity: &LayerwiseIdentity,
        catalog: &MemoryCatalog,
        policy: &KvStore,
        keys: &RawStore,
        specs: &[LayerRequestSpec],
        native_history_tokens: u64,
        revision: impl FnMut() -> MemoryStamp + 'static,
    ) -> Result<Self> {
        let hk: Vec<HkLayerRequestSpec> = specs
            .iter()
            .map(|s| HkLayerRequestSpec {
                model_layer: s.model_layer,
                query: s.query,
                budget: s.budget,
                mandatory: s.mandatory.as_ptr(),
                mandatory_count: s.mandatory.len(),
            })
            .collect();
        let mut state = Box::new(RevisionState { f: Box::new(revision) });
        let user = &mut *state as *mut RevisionState as *mut c_void;
        clear_error();
        let ptr = unsafe {
            ffi::hk_layerwise_request_create(
                identity,
                catalog.ptr.as_ptr(),
                policy.ptr.as_ptr(),
                keys.ptr.as_ptr(),
                hk.as_ptr(),
                hk.len(),
                native_history_tokens,
                Some(revision_trampoline),
                user,
            )
        };
        NonNull::new(ptr)
            .map(|ptr| Self { ptr, revision: state })
            .ok_or_else(err_from_last)
    }

    /// Abort the transaction (idempotent); moves the phase to `Aborted`.
    pub fn abort(&self) {
        unsafe { ffi::hk_layerwise_request_abort(self.ptr.as_ptr()) };
    }
    pub fn phase(&self) -> LayerwisePhase {
        LayerwisePhase::from_u32(unsafe { ffi::hk_layerwise_request_phase(self.ptr.as_ptr()) })
    }
    pub fn work(&self) -> Result<LayerwiseWork> {
        let mut out: LayerwiseWork = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_work(p, &mut out); })?;
        Ok(out)
    }

    /// Open target layer `model_layer` for projection.
    pub fn begin_layer(&self, id: &LayerwiseIdentity, model_layer: u32) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_begin_layer(p, id, model_layer); })
    }

    /// Report `rows` contiguous `[rows, query_heads, head_dim]` F32 current-forward
    /// rows (post-normalization/pre-RoPE) for this layer, starting at token `begin`.
    pub fn projected(
        &self,
        id: &LayerwiseIdentity,
        begin: u64,
        rows: u32,
        data: &[f32],
        geometry: LayerQueryGeometry,
    ) -> Result<()> {
        let q = HkCurrentLayerQuery {
            data: data.as_ptr(),
            layer: geometry.layer,
            rows,
            query_heads: geometry.query_heads,
            kv_heads: geometry.kv_heads,
            head_dim: geometry.head_dim,
            attention_scale: geometry.attention_scale,
        };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_projected(p, id, begin, &q); })
    }

    /// Report a borrowed native Q tensor (device or host) for this layer and let
    /// the owner's `scorer` compute per-block page mass against the frozen frame.
    /// Only the tensor's intersection with the explicit user Query enters
    /// scoring. The scorer must complete its native stream before returning and
    /// write exactly `frame.starts.len()` non-negative scores (masked blocks 0).
    /// The closure may borrow the statistics/device handles it needs; it is
    /// called synchronously, so no `'static` bound is required.
    pub fn projected_native<F>(
        &self,
        id: &LayerwiseIdentity,
        begin: u64,
        query: &NativeCurrentLayerQuery,
        mut scorer: F,
    ) -> Result<()>
    where
        F: FnMut(&NativeCurrentLayerQuery, &QueryScoreFrame<'_>) -> Result<Vec<f64>>,
    {
        let p = self.ptr.as_ptr();
        let user = &mut scorer as *mut F as *mut c_void;
        checked(|| unsafe {
            ffi::hk_layerwise_request_projected_native(
                p,
                id,
                begin,
                query,
                Some(scorer_trampoline::<F>),
                user,
            );
        })
    }

    /// Freeze this layer's selection and obtain a NEW caller-owned [`LayerPlan`]
    /// describing the compact restore/spill/evict working set.
    pub fn prepare_restore(&self, id: &LayerwiseIdentity) -> Result<LayerPlan> {
        let mut out: *mut HkLayerPlan = ptr::null_mut();
        let p = self.ptr.as_ptr();
        clear_error();
        let rc = unsafe { ffi::hk_layerwise_request_prepare_restore(p, id, &mut out) };
        if rc != 0 {
            return Err(err_from_last());
        }
        NonNull::new(out).map(|ptr| LayerPlan { ptr }).ok_or_else(err_from_last)
    }

    /// Report the post-fence native view the driver actually restored.
    pub fn restored(&self, id: &LayerwiseIdentity, view: &ExecutionView) -> Result<()> {
        let v = view.ffi();
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_restored(p, id, &v); })
    }

    /// Report `count` attended rows starting at token `begin` for this layer.
    pub fn attended(&self, id: &LayerwiseIdentity, begin: u64, count: u64) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_attended(p, id, begin, count); })
    }

    /// Close this layer's attending phase (moves to `AwaitingAcceptance`).
    pub fn complete_layer(&self, id: &LayerwiseIdentity) -> Result<()> {
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_complete_layer(p, id); })
    }

    /// Publish the accepted per-layer snapshots (one per target layer), advancing
    /// the canonical history. Moves the phase to `Accepted`.
    pub fn accepted(&self, id: &LayerwiseIdentity, snapshots: &[LayerMemorySnapshot]) -> Result<()> {
        let hk: Vec<HkLayerMemorySnapshot> = snapshots
            .iter()
            .map(|l| HkLayerMemorySnapshot {
                model_layer: l.model_layer,
                memory: HkMemorySnapshot {
                    stamp: l.memory.stamp,
                    blocks: l.memory.blocks.as_ptr(),
                    block_count: l.memory.blocks.len(),
                    resident: l.memory.resident.as_ptr(),
                    resident_count: l.memory.resident.len(),
                },
            })
            .collect();
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_accepted(p, id, hk.as_ptr(), hk.len()); })
    }

    // ── frozen selection (after prepare_restore, through attending) ──
    pub fn selection_layer(&self) -> u32 {
        unsafe { ffi::hk_layerwise_request_selection_layer(self.ptr.as_ptr()) }
    }
    pub fn selection_score_count(&self) -> usize {
        unsafe { ffi::hk_layerwise_request_selection_score_count(self.ptr.as_ptr()) }
    }
    pub fn selection_score(&self, i: usize) -> Result<f64> {
        let mut out = 0f64;
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_selection_score(p, i, &mut out); })?;
        Ok(out)
    }
    pub fn selection_fixed_count(&self) -> usize {
        unsafe { ffi::hk_layerwise_request_selection_fixed_count(self.ptr.as_ptr()) }
    }
    pub fn selection_fixed(&self, i: usize) -> u32 {
        unsafe { ffi::hk_layerwise_request_selection_fixed(self.ptr.as_ptr(), i) }
    }
    pub fn selection_selected_count(&self) -> usize {
        unsafe { ffi::hk_layerwise_request_selection_selected_count(self.ptr.as_ptr()) }
    }
    pub fn selection_selected(&self, i: usize) -> u32 {
        unsafe { ffi::hk_layerwise_request_selection_selected(self.ptr.as_ptr(), i) }
    }
    pub fn selection_historical_tokens(&self) -> u64 {
        unsafe { ffi::hk_layerwise_request_selection_historical_tokens(self.ptr.as_ptr()) }
    }
    pub fn selection_charged_tokens(&self) -> u64 {
        unsafe { ffi::hk_layerwise_request_selection_charged_tokens(self.ptr.as_ptr()) }
    }

    // ── accepted per-layer snapshots (Accepted phase only) ──
    pub fn accepted_layer_count(&self) -> usize {
        unsafe { ffi::hk_layerwise_request_accepted_layer_count(self.ptr.as_ptr()) }
    }
    pub fn accepted_model_layer(&self, i: usize) -> u32 {
        unsafe { ffi::hk_layerwise_request_accepted_model_layer(self.ptr.as_ptr(), i) }
    }
    pub fn accepted_stamp(&self, i: usize) -> Result<MemoryStamp> {
        let mut out: MemoryStamp = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_accepted_stamp(p, i, &mut out); })?;
        Ok(out)
    }
    pub fn accepted_block_count(&self, i: usize) -> usize {
        unsafe { ffi::hk_layerwise_request_accepted_block_count(self.ptr.as_ptr(), i) }
    }
    pub fn accepted_block(&self, i: usize, b: usize) -> Result<BlockCopies> {
        let mut out: BlockCopies = unsafe { std::mem::zeroed() };
        let p = self.ptr.as_ptr();
        checked(|| unsafe { ffi::hk_layerwise_request_accepted_block(p, i, b, &mut out); })?;
        Ok(out)
    }
    pub fn accepted_resident_count(&self, i: usize) -> usize {
        unsafe { ffi::hk_layerwise_request_accepted_resident_count(self.ptr.as_ptr(), i) }
    }
    pub fn accepted_resident(&self, i: usize, b: usize) -> u64 {
        unsafe { ffi::hk_layerwise_request_accepted_resident(self.ptr.as_ptr(), i, b) }
    }
}

impl Drop for LayerwiseRequest {
    fn drop(&mut self) {
        // Destroy the C++ request first so its SourceRevision lambda can no
        // longer call the trampoline; `self.revision` is freed afterwards.
        unsafe { ffi::hk_layerwise_request_destroy(self.ptr.as_ptr()) };
    }
}
