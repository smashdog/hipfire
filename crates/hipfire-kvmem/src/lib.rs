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

use std::ffi::{c_char, c_int, CStr, CString};
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
