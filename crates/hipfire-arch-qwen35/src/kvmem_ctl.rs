// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KVMem host-offload controller for the multi-slot serve engine.
//!
//! The engine keeps every slot's KV in paged GPU arenas. KVMem owns only the
//! *residency* decision for the FullAttention layers: a full 128-token page is
//! either resident through the slot's block table or replaced by the
//! [`SPILLED_PAGE`] sentinel, its packed Q8 bytes living in the KVMem host
//! store. The masked attention kernels skip sentinel pages, so a spilled page
//! is invisible to the model until a later selection restores it.
//!
//! Residency policy ([`KvMemCtl::prepare_step`], once per serve step, before
//! any page is provisioned):
//!
//! * prefill: keep the sink and the newest blocks
//!   (`pick_prefill_pressure_blocks`), except the chunk that completes the
//!   prompt — its forward produces the first sampled token, so it scores the
//!   history against the question the way decode does,
//! * decode/verify: score every block against the slot's last Q row
//!   (`mean_key_retrieval_scores`) and keep sink + recent + the best scores
//!   (`pick_topk_blocks`),
//! * pass 1 spills every unselected resident full page, pass 2 restores every
//!   selected spilled page. Spilling first maximises free pages for restore.
//!
//! Only *full* pages are spilled; the partial tail page is always resident
//! (new writes land there), and `ensure_capacity` never re-extends a spilled
//! logical page — the sentinel keeps the logical page count intact.
//!
//! Statistics ([`KvMemCtl::feed_rows`]) are the post-RMSNorm/pre-RoPE K rows
//! the tap buffers captured during the forward, fed only for rows the
//! scheduler kept. A rejected draft row never enters the host store, so a
//! later write to that position accumulates onto exactly the bytes the GPU
//! kept. Fed rows are always contiguous from the store's current length:
//! prefill chunks and decode rows extend the frontier by exactly their row
//! count, and a verify step commits exactly `committed.len()` rows from the
//! verify start.
//!
//! Everything here is host-side; the only GPU traffic is the page-sized D2H /
//! H2D byte copies on spill and restore. Those copies move the arena's bytes
//! verbatim, so a spill/restore round is bit-exact.

use std::cell::Cell;
use std::path::PathBuf;

use hipfire_kvmem::{
    mean_key_retrieval_scores, page_softmax_scores, BlockCopies, BlockDescriptor, ExecutionView,
    GqaAggregation, ImportStart, KvStore, KvStoreConfig, LayerMemorySnapshot, LayerPlan,
    LayerSelection, MemoryBudget, MemoryCatalog, MemoryDescriptor, MemorySnapshot, MemoryStamp,
    OutputOptions, OutputPressure, OutputScope, PrefillOptions, PrefillPressure, QueryRange,
    RawStore, RawStoreConfig, SessionIdentity, Storage, WindowEnvelope, WindowPlan, WindowPolicy,
    WindowState,
};
use hipfire_runtime::serve::KvMemStats;
use rdna_compute::layerwise_pool::{mtp_auxiliary_capacity, mtp_head_pool, LayerwisePagePool};
use rdna_compute::page_pool::SPILLED_PAGE;
use rdna_compute::slot_pool::{SlotId, SlotPool};
use rdna_compute::{Gpu, GpuTensor};
use saddle_core::kv::KvMode;

use crate::kvmem_prefix::{
    layout_fingerprint, sidecar_path, ColdPack, ColdPackIndex, ContentPackMeta, PrefixCatalog,
    PrefixCheckpoint, PREFIX_PAGE_TOKENS, SIDECAR_MIN_BYTES,
};
use crate::kvmem_taps::KvMemTaps;
use crate::layerwise_driver::{
    layerwise_backend, layerwise_capabilities, OutputAcceptanceSnapshot,
};
use crate::layerwise_layout::LayerwiseMtpQ8Layout;
use crate::layerwise_mtp::{
    auxiliary_removed_tokens, auxiliary_visible_tokens, AuxiliarySelection, LayerwiseMtpHistory,
};

mod import;
mod layerwise;
mod legacy;
mod options;
mod plan;
mod prefix;
mod slot;
mod step;
mod transfer;

pub use import::*;
pub(crate) use options::{KvMemDiagnostics, KvMemOptions};

/// KVMem blocks are 128 tokens — the engine's page size (`PAGE_TOKENS`).
pub const KV_BLOCK_TOKENS: usize = 128;

/// Physical pages per slot when the operator has not sized the pool: the
/// historical 8K-token working set. A larger `multi_slot_ctx` rides on KVMem
/// (only this much stays resident), which is the point of the integration.
const DEFAULT_POOL_BLOCKS_PER_SLOT: usize = 64;

/// Blocks written between selection rounds. Larger = fewer selection passes
/// (cheaper) but more transient residency; the pool headroom is sized for it.
const DEFAULT_SELECT_INTERVAL: usize = 4;

/// Resident-block floor when the step growth alone would squeeze the budget.
/// A huge prefill chunk must fit alongside a working set worth scoring; the
/// pool grows past `DEFAULT_POOL_BLOCKS_PER_SLOT` instead of starving it.
const DEFAULT_RESIDENT_FLOOR: usize = 32;

/// P4.1③ A1 starvation guard: UpstreamGlobal head-history selection only engages
/// when the head budget is at least this many blocks; below it the policy falls
/// back to FirstLayer (the trunk keep-set). Grounded by the recite benefit sweep
/// (2026-11-17): at 8 blocks (BUDGET=1024) the global Mean-K reduction dilutes the
/// draft-critical signal and UG acceptance regresses (0.381 vs FL 0.782, early
/// EOS); at 16 blocks (BUDGET=2048) UG wins (+16.7%, 5/5). The UG core algorithm
/// is unchanged when engaged (faithful to ninfer's literal Mean-K/TopK picker);
/// this is an operational envelope guard, not a selection-semantics change.
const MTP_HISTORY_UG_MIN_BLOCKS: usize = 16;

/// Host payload capacity for the packed Q8 pages (RAM tier).
const DEFAULT_HOST_BYTES: u64 = 16 << 30;

/// Disk-tier capacity when `HIPFIRE_KVMEM_SPILL_DIR` is set.
const DEFAULT_SPILL_BYTES: u64 = 64 << 30;
/// 期3 G3-SSD: default content-cache disk allowance (independent account).
const DEFAULT_CONTENT_CACHE_BYTES: u64 = 2 << 30;

/// Generation reserve R added on top of the resident budget B to form the
/// layerwise window capacity (ninfer `device_tokens = selected + reserve`,
/// CLI default_reserve). Import admission measures the prompt tail + output
/// against B + R.
const DEFAULT_GEN_RESERVE: u64 = 1024;

/// Upper bound on the reusable pinned transfer slab used by the batched
/// spill/restore path (`HIPFIRE_KVMEM_BATCH_XFER`, default on). Pages are
/// gathered into the slab in chunks of at most this many bytes, each chunk
/// drained with ONE stream wait, so pinned host RAM stays bounded regardless
/// of how many pages a step moves. A single page larger than the cap still
/// fits (the slab grows to one page). Mirrors llama's 32 MiB stage-in slab.
const XFER_SLAB_CAP: usize = 64 << 20;

fn fnv64(buf: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in buf {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `HIPFIRE_KVMEM_SELECT_METHOD`: `retrieval` (default) | `h2o` | `recency`, or
/// the core `KvMemMethod` ordinals 0/1/2. Retrieval feeds the mean-K query
/// scores (the historical behaviour); Recency feeds no learned signal so
/// `pick_topk_blocks` keeps sink + recent and fills the rest by recency. H2O
/// needs the per-block attention-heat tap (the `hipfire-h2o-heat` patch) and is
/// rejected loudly until that lands, rather than silently degenerating.
fn parse_select_method(v: &str) -> Option<u32> {
    match v.trim().to_ascii_lowercase().as_str() {
        "retrieval" | "0" => Some(0),
        "h2o" | "1" => Some(1),
        "recency" | "2" => Some(2),
        _ => None,
    }
}

/// KVMem recall execution model (`HIPFIRE_KVMEM_RECALL_STRATEGY`). Mirrors the
/// core `kvmem::RecallStrategy` ordinals (`layerwise_options.hpp:11`):
/// `Legacy = 0` is the historical whole-working-set residency + `pick_topk`
/// selection (the byte-neutral default); `Layerwise = 1` enables the per-layer
/// composite-slot recall subsystem driven by the core `LayerwiseQueryRequest`
/// state machine. The strategy is resolved once at controller construction and
/// every Layerwise-only path is gated on it, so the default Legacy serve stays
/// bit-identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RecallStrategy {
    Legacy = 0,
    Layerwise = 1,
}

impl RecallStrategy {
    /// True when the per-layer recall subsystem is enabled.
    pub fn is_layerwise(self) -> bool {
        matches!(self, RecallStrategy::Layerwise)
    }
}

/// `HIPFIRE_KVMEM_RECALL_STRATEGY`: `legacy` (default) | `layerwise`, or the
/// core ordinals 0/1. An unknown value is rejected loudly at construction
/// rather than silently falling back to Legacy.
fn parse_recall_strategy(v: &str) -> Option<RecallStrategy> {
    match v.trim().to_ascii_lowercase().as_str() {
        "legacy" | "0" => Some(RecallStrategy::Legacy),
        "layerwise" | "1" => Some(RecallStrategy::Layerwise),
        _ => None,
    }
}

/// Resolve the recall strategy from the raw `HIPFIRE_KVMEM_RECALL_STRATEGY`
/// value (already trimmed by the engine snapshot). Absent → Legacy (the byte-neutral
/// default); an unrecognised spelling is rejected loudly rather than silently
/// degenerating. Pure so the resolution contract is testable without touching
/// the process-start config snapshot.
fn resolve_recall_strategy(raw: Option<String>) -> Result<RecallStrategy, String> {
    match raw {
        Some(v) => parse_recall_strategy(&v).ok_or_else(|| {
            format!("kvmem: HIPFIRE_KVMEM_RECALL_STRATEGY={v} invalid (legacy|layerwise)")
        }),
        None => Ok(RecallStrategy::Legacy),
    }
}

/// Layerwise fail-closed policy guard (design §4, R2d-D1). The per-layer
/// composite-slot subsystem drives residency by admission + per-layer
/// current-Q GPU scoring + pressure mass/exposure, NOT the global working-set
/// policy knobs. ninfer forces lrfu=false and retrieval selection; the
/// watermark hysteresis and the multi-row query-span ring belong to the Legacy
/// global path. Under Layerwise any non-default value is an explicit error
/// naming the strategy (no silent override/fallback), mirroring the core
/// selection guards (layerwise_selection.cpp:25) and the engine watermark
/// validation style. Legacy admits every combination — the caller has already
/// range-checked the values, so this guard is Layerwise-specific and the
/// default serve stays byte-neutral. Pure so the decision table is testable
/// without the process-start config snapshot (same rationale as
/// [`resolve_recall_strategy`]: `developer_var` is a `OnceLock`, a runtime
/// `set_var` is invisible).
fn resolve_layerwise_policy(
    strategy: RecallStrategy,
    select_method: u32,
    lrfu: bool,
    watermark_enabled: bool,
    query_span: usize,
) -> Result<(), String> {
    if !strategy.is_layerwise() {
        return Ok(());
    }
    if select_method != 0 {
        return Err(format!(
            "kvmem: SELECT_METHOD={select_method} incompatible with \
             RECALL_STRATEGY=layerwise (per-layer admission drives retrieval \
             selection); use retrieval or unset HIPFIRE_KVMEM_SELECT_METHOD"
        ));
    }
    if lrfu {
        return Err("kvmem: LRFU=1 incompatible with RECALL_STRATEGY=layerwise \
             (ninfer forces lrfu=false; per-layer mass/exposure replaces the \
             decay history); unset HIPFIRE_KVMEM_LRFU"
            .to_string());
    }
    if watermark_enabled {
        return Err("kvmem: GPU_HIGH/LOW_WATERMARK incompatible with \
             RECALL_STRATEGY=layerwise (admission-driven pressure replaces the \
             global watermark hysteresis); unset HIPFIRE_KVMEM_GPU_HIGH_WATERMARK \
             and HIPFIRE_KVMEM_GPU_LOW_WATERMARK"
            .to_string());
    }
    if query_span > 1 {
        return Err(format!(
            "kvmem: QUERY_SPAN={query_span} incompatible with \
             RECALL_STRATEGY=layerwise (per-layer current-Q GPU scoring replaces \
             the multi-row page-softmax ring); use 1 or unset HIPFIRE_KVMEM_QUERY_SPAN"
        ));
    }
    Ok(())
}

/// Resolve `HIPFIRE_KVMEM_LAYERWISE_MTP_HISTORY` (P4.1③ H5): `first-layer` |
/// `upstream-global`. Absent → the ninfer FirstLayer auto-default under
/// Layerwise+MTP (else Unspecified, carried unread). An unrecognised spelling is
/// rejected loudly; a resolved policy off the Layerwise+MTP path is rejected by
/// the core mirror `validate` (no silent fallback). Pure so the resolution
/// contract is testable without the process-start config snapshot (same rationale
/// as [`resolve_recall_strategy`]). Legacy / AR-only never read it ⇒ byte-neutral.
fn resolve_layerwise_mtp_history(
    raw: Option<String>,
    strategy: RecallStrategy,
    mtp_enabled: bool,
) -> Result<LayerwiseMtpHistory, String> {
    let parsed = match raw {
        Some(v) => LayerwiseMtpHistory::parse(v.trim()).map_err(|_| {
            format!(
                "kvmem: HIPFIRE_KVMEM_LAYERWISE_MTP_HISTORY={v} invalid \
                 (first-layer|upstream-global)"
            )
        })?,
        None => LayerwiseMtpHistory::Unspecified,
    };
    parsed.resolve_default(strategy, mtp_enabled)
}

/// Ninfer-faithful layerwise activation quotas (design §3.11, ③-E1). Five
/// explicit host-resource envelopes backing the projected-Q history tapes and
/// their transfer/metadata staging. Resolved once at controller construction
/// from `HIPFIRE_KVMEM_LAYERWISE_*_MIB` (MiB units), defaulting to the ninfer
/// engine seeds (64/32/8/1024/96 MiB). Only the Layerwise path creates accounts
/// against them — under Legacy the defaults are carried unread, so the legacy
/// serve stays byte-neutral. Orthogonal to the GPU selection budget
/// (`budget_tokens`): these bound HOST activation memory, not device residency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerwiseQuotas {
    /// `activation_ram`: RAM backing for the per-layer Q tapes (less staging).
    pub activation_ram: u64,
    /// `activation_disk`: disk backing when the tape peak exceeds the RAM account.
    pub activation_disk: u64,
    /// `staging`: pinned Q D2H readback transfer budget.
    pub staging: u64,
    /// `metadata`: retained source-version / frozen-index / feedback envelope.
    pub metadata: u64,
    /// `activation_metadata`: finite tape descriptor/chunk-table account.
    pub activation_metadata: u64,
    /// Disk-tier directory (reuses `HIPFIRE_KVMEM_SPILL_DIR`); required only when
    /// the tape peak forces the disk tier.
    pub disk_directory: Option<PathBuf>,
}

impl Default for LayerwiseQuotas {
    /// The ninfer engine seeds (`native_query_runner` activation limits).
    fn default() -> Self {
        Self {
            activation_ram: 64 << 20,
            activation_disk: 32 << 20,
            staging: 8 << 20,
            metadata: 1024 << 20,
            activation_metadata: 96 << 20,
            disk_directory: None,
        }
    }
}

/// Resolve the five activation quotas from raw env values (MiB integers). Absent
/// → the ninfer seed; an unparseable or overflowing value is rejected loudly
/// rather than silently degenerating (mirrors [`resolve_recall_strategy`]). Pure
/// so the resolution contract is testable without the process-start config
/// snapshot. `disk_directory` is the already-resolved spill dir (if any).
fn resolve_layerwise_quotas(
    ram: Option<String>,
    disk: Option<String>,
    staging: Option<String>,
    metadata: Option<String>,
    activation_metadata: Option<String>,
    disk_directory: Option<PathBuf>,
) -> Result<LayerwiseQuotas, String> {
    let d = LayerwiseQuotas::default();
    fn mib(raw: Option<String>, name: &str, def: u64) -> Result<u64, String> {
        match raw {
            Some(v) => {
                let m: u64 = v
                    .trim()
                    .parse()
                    .map_err(|_| format!("kvmem: {name}={v} invalid (MiB integer)"))?;
                m.checked_mul(1 << 20)
                    .ok_or_else(|| format!("kvmem: {name}={v} MiB overflow"))
            }
            None => Ok(def),
        }
    }
    Ok(LayerwiseQuotas {
        activation_ram: mib(
            ram,
            "HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_RAM_MIB",
            d.activation_ram,
        )?,
        activation_disk: mib(
            disk,
            "HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_DISK_MIB",
            d.activation_disk,
        )?,
        staging: mib(staging, "HIPFIRE_KVMEM_LAYERWISE_STAGING_MIB", d.staging)?,
        metadata: mib(metadata, "HIPFIRE_KVMEM_LAYERWISE_METADATA_MIB", d.metadata)?,
        activation_metadata: mib(
            activation_metadata,
            "HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_METADATA_MIB",
            d.activation_metadata,
        )?,
        disk_directory,
    })
}

/// Byte size of one retained [`LayerwisePrefillValue`] (`original_start: u64` +
/// `mass: f64` + `exposure: f64` = 24). The pressure owner keeps at most one such
/// value per resident logical block start per FA layer.
const PRESSURE_VALUE_BYTES: u64 = 24;
/// Per-layer fixed overhead of a `LayerwisePrefillState` (`model_layer: u32` +
/// `end/samples/chunks: u64` + the `values` Vec header). A conservative constant
/// so the ceiling bounds the whole mutable history, not just the value payload.
const PRESSURE_STATE_OVERHEAD: u64 = 64;

/// Finite mutable-history ceiling of one SESSION-lifetime pressure owner
/// (design §3.11, ③-E4; ninfer essence = "complete mutable history ceiling").
/// The core `LayerwisePrefillPressure::accept` rebuilds each layer's retained
/// `values` from ONLY the current packet's `history` keys, and the driver feeds
/// it the restored compact working set (≤ `budget_blocks`) plus the suffix page
/// — so a layer retains at most `budget_blocks + 1` values. Across `n_fa_layers`
/// independent target layers the complete mutable history is bounded by
/// `n_fa_layers × ((budget_blocks + 1) × 24 + overhead)`. Saturating arithmetic
/// keeps the bound finite for any geometry; a too-small partition then fails
/// closed rather than wrapping. Pure (no accounts, no GPU) so admission is
/// host-testable before `PrefillPressure::new` acquires anything.
fn layerwise_pressure_ceiling(n_fa_layers: usize, budget_blocks: usize) -> u64 {
    let per_layer = (budget_blocks as u64)
        .saturating_add(1)
        .saturating_mul(PRESSURE_VALUE_BYTES)
        .saturating_add(PRESSURE_STATE_OVERHEAD);
    (n_fa_layers as u64).saturating_mul(per_layer)
}

/// The pressure partition of the metadata envelope: the last transient eighth,
/// carved out of E1's source share (3/4 → 5/8) so the per-query activation
/// (source/frozen/feedback) and the per-session pressure jointly fit one metadata
/// quota with no double-spend across the two lifetimes. Mirrors
/// `LayerwiseActivation::admit`'s `transient = metadata / 8`.
fn layerwise_pressure_cap(metadata: u64) -> u64 {
    metadata / 8
}

/// Fail-closed finite admission for the SESSION-lifetime pressure owner
/// (design §3.11, ③-E4). The owner's complete mutable-history ceiling must fit
/// the metadata pressure partition BEFORE `PrefillPressure::new` is called — a
/// quota too small for the model geometry yields `Err` with nothing acquired,
/// exactly ninfer's allocation-phase admission contract. Layerwise-only: under
/// Legacy the owner is never created (byte-neutral). Pure so the decision is
/// testable without the process-start config snapshot (same `OnceLock` rationale
/// as [`resolve_recall_strategy`]).
fn resolve_pressure_admission(
    strategy: RecallStrategy,
    n_fa_layers: usize,
    budget_blocks: usize,
    quotas: &LayerwiseQuotas,
) -> Result<(), String> {
    if !strategy.is_layerwise() {
        return Err("kvmem: pressure owner requires RECALL_STRATEGY=Layerwise".to_string());
    }
    let ceiling = layerwise_pressure_ceiling(n_fa_layers, budget_blocks);
    let cap = layerwise_pressure_cap(quotas.metadata);
    if cap == 0 || ceiling > cap {
        return Err(format!(
            "kvmem: layerwise pressure history ceiling {ceiling}B exceeds the metadata \
             pressure partition {cap}B (1/8 of {}B); raise \
             HIPFIRE_KVMEM_LAYERWISE_METADATA_MIB",
            quotas.metadata
        ));
    }
    Ok(())
}

/// True when the resolved process config or its start-time environment fallback
/// enables KVMem. The serve layer separately refuses a non-`kvmem` build.
pub(crate) fn requested() -> bool {
    if let Some(config) =
        hipfire_config::active_process_config().and_then(|process| process.kvmem.as_ref())
    {
        return config.enabled;
    }
    hipfire_config::developer_var("HIPFIRE_KVMEM")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .is_some_and(|value| matches!(value.as_str(), "1" | "on" | "true"))
}

/// Resolved pool geometry for a KVMem-enabled engine.
#[derive(Debug, Clone, Copy)]
pub struct KvMemPlan {
    /// Physical pages the paged pool is built with (all slots together).
    pub pages_total: usize,
    /// Per-slot budget of resident pages (full pages + the partial tail).
    pub budget_blocks: usize,
    /// Pages one serve step can add to a slot's frontier.
    pub grow_blocks: usize,
    /// Selection cadence in full blocks.
    pub interval: usize,
}

/// Resolve the pool size and per-slot budget from the operator env.
///
/// Defaults preserve upstream behaviour for contexts that fit the historical
/// 8K-token working set: the pool is built exactly as `legacy_equivalent`, the
/// budget equals the slot cap, and no page is ever spilled. A `multi_slot_ctx`
/// beyond that keeps the 8K working set and lets KVMem hold the rest.
pub fn resolve_plan(
    n_slots: usize,
    cap_tokens: usize,
    max_step_rows: usize,
    options: &KvMemOptions,
) -> Result<KvMemPlan, String> {
    if n_slots == 0 || cap_tokens == 0 {
        return Err("kvmem: engine has no slots or no token capacity".to_string());
    }
    let cap_blocks = cap_tokens.div_ceil(KV_BLOCK_TOKENS).max(1);
    let grow_blocks = max_step_rows.div_ceil(KV_BLOCK_TOKENS).max(1);
    let interval = options
        .plan_select_interval
        .unwrap_or(DEFAULT_SELECT_INTERVAL)
        .clamp(1, 256);
    // Residency transiently overshoots the budget by one step's growth plus
    // the blocks written before the next selection can fire. One prefill
    // chunk is written in a single forward, so the whole chunk must fit on
    // top of the budget — `max_step_rows` is therefore the largest
    // per-slot row count, not the whole batch's.
    let headroom = grow_blocks.saturating_add(interval).saturating_add(1); // the partial tail page
    let explicit_pages = options.plan_pages.map(|p| p.max(n_slots));
    let pages_per_slot = match explicit_pages {
        Some(total) => (total / n_slots).max(1),
        None => cap_blocks
            .min(DEFAULT_POOL_BLOCKS_PER_SLOT)
            .max(headroom.saturating_add(DEFAULT_RESIDENT_FLOOR)),
    };
    let pages_total = explicit_pages.unwrap_or(pages_per_slot * n_slots);
    // A slot whose cap fits the pool never spills (upstream parity). Past
    // that, the budget is what stays resident and the headroom is the room
    // the frontier needs between selection rounds.
    let budget_max = if cap_blocks <= pages_per_slot {
        cap_blocks
    } else {
        pages_per_slot
            .saturating_sub(headroom)
            .max(DEFAULT_RESIDENT_FLOOR.min(cap_blocks))
            .min(pages_per_slot.saturating_sub(1))
    };
    let budget_blocks = match options.plan_budget_tokens {
        Some(tokens) => (tokens / KV_BLOCK_TOKENS).clamp(1, budget_max),
        None => budget_max,
    };
    if budget_blocks < 2 {
        return Err(format!(
            "kvmem: per-slot residency budget is {budget_blocks} block(s); \
             at least 2 are required (sink + recent). Raise the pool \
             (HIPFIRE_SLOTS_PAGED_PAGES) or HIPFIRE_KVMEM_BUDGET_TOKENS"
        ));
    }
    Ok(KvMemPlan {
        pages_total,
        budget_blocks,
        grow_blocks,
        interval,
    })
}

/// Effective Legacy history budget for the admission site. The config layer
/// defaults `budget_tokens` to 32768 and `gen_reserve_tokens` to 1024 without
/// knowing the logical context, so a default budget meeting C=32768 would
/// fail `B + R <= C` on every stock 32k deployment. An *implicit* budget (no
/// explicit env/CLI/snapshot input, `Source::Default`) is therefore clamped to
/// `C - R` floored to whole blocks; an explicitly requested budget is never
/// reduced — `resolve_legacy_plan` still rejects it, mirroring the
/// "not reduced automatically" contract for explicit physical pools.
pub fn effective_legacy_budget(
    budget_tokens: usize,
    implicit: bool,
    cap_tokens: usize,
    reserve_tokens: usize,
) -> usize {
    if !implicit {
        return budget_tokens;
    }
    let ceiling = cap_tokens
        .saturating_sub(reserve_tokens)
        .saturating_div(KV_BLOCK_TOKENS)
        .saturating_mul(KV_BLOCK_TOKENS);
    budget_tokens.min(ceiling)
}

/// Whether `budget_tokens` is the built-in default (no env/CLI/snapshot
/// input). Only an implicit budget is ever clamped to the logical context;
/// one lookup shared by the admission site and the controller restatement so
/// the two cannot drift.
pub fn implicit_legacy_budget(config: &hipfire_config::kvmem::ResolvedConfig) -> bool {
    config
        .report
        .get("budget_tokens")
        .is_none_or(|report| report.source == hipfire_config::kvmem::Source::Default)
}

/// Admit the public Legacy contract after token normalization. Unlike the
/// historical env-only pool planner, the physical pool never silently reduces
/// the requested history budget. NInfer's device window is B + R: a pressure
/// selection returns to B and every subsequent append can reuse R.
#[allow(clippy::too_many_arguments)]
pub fn resolve_legacy_plan(
    n_slots: usize,
    cap_tokens: usize,
    max_step_rows: usize,
    budget_tokens: usize,
    reserve_tokens: usize,
    explicit_pages: Option<usize>,
) -> Result<KvMemPlan, String> {
    if n_slots == 0 || cap_tokens == 0 || max_step_rows == 0 {
        return Err("kvmem: Legacy requires positive slots, context and step capacity".into());
    }
    if budget_tokens < 2 * KV_BLOCK_TOKENS
        || reserve_tokens < KV_BLOCK_TOKENS
        || budget_tokens % KV_BLOCK_TOKENS != 0
        || reserve_tokens % KV_BLOCK_TOKENS != 0
    {
        return Err(format!(
            "kvmem: normalized Legacy B/R must be multiples of {KV_BLOCK_TOKENS}, \
             with B >= {} and R >= {KV_BLOCK_TOKENS}",
            2 * KV_BLOCK_TOKENS
        ));
    }
    let window = budget_tokens
        .checked_add(reserve_tokens)
        .ok_or("kvmem: Legacy B + R overflow")?;
    if window > cap_tokens {
        return Err(format!(
            "kvmem: Legacy B + R ({budget_tokens} + {reserve_tokens} = {window}) \
             exceeds logical context C={cap_tokens}"
        ));
    }
    if max_step_rows > reserve_tokens {
        return Err(format!(
            "kvmem: native append/verify width {max_step_rows} exceeds R={reserve_tokens}; \
             normalize prefill or increase --kvmem-gen-reserve"
        ));
    }
    let required_pages = (window / KV_BLOCK_TOKENS)
        .checked_mul(n_slots)
        .ok_or("kvmem: Legacy physical pool page count overflow")?;
    let pages_total = explicit_pages.unwrap_or(required_pages);
    if pages_total < required_pages {
        return Err(format!(
            "kvmem: explicit physical pool has {pages_total} pages, but Legacy \
             B={budget_tokens}, R={reserve_tokens}, slots={n_slots} require at least \
             {required_pages} pages; the history budget is not reduced automatically"
        ));
    }
    Ok(KvMemPlan {
        pages_total,
        budget_blocks: budget_tokens / KV_BLOCK_TOKENS,
        grow_blocks: max_step_rows.div_ceil(KV_BLOCK_TOKENS),
        // Public Legacy selects at actual window pressure, not every N blocks.
        // Retain the field for the unchanged Layerwise/env-only planner.
        interval: 1,
    })
}

/// Model geometry the controller has to address.
#[derive(Debug, Clone, Copy)]
pub struct KvMemGeometry {
    pub n_fa_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// P4.4: the KV tier this controller's geometry resolves to. The layerwise
    /// layout adapter registers a per-tier format identity from it (format
    /// isolation); Legacy carries it for the same admission/identity purpose.
    pub kv_mode: KvMode,
    /// K arena bytes per position (packed Q8 row of one KV head set).
    pub per_pos_bytes: usize,
    /// V arena bytes per position.
    pub per_pos_v_bytes: usize,
    /// Tap buffer row capacity (`PrefillBatchScratch::max_batch`).
    pub max_rows: usize,
    /// P4.1② G3: this model runs the independent native MTP head (the engine's
    /// `mtp_head.is_some() && mtp_k > 0` enablement). Under Layerwise this turns
    /// the host co-archive into the COMPOSITE `kvmem-main-mtp-pages-v4` record
    /// (`n_fa_layers + 1` ordinals, the head aux plane at ordinal `n_fa_layers`)
    /// and activates the per-slot head-plane state. Under Legacy it is carried
    /// unread (the head keeps the historical full-resident KvCache path), so the
    /// default serve stays byte-neutral. The trunk GPU paged arena is UNCHANGED
    /// (`n_fa_layers`); only the host record gains the appended aux layer.
    pub mtp_head: bool,
}

struct SlotState {
    raw: RawStore,
    kv: KvStore,
    /// Full block count at the last selection round.
    last_select_full: usize,
    /// Host mirror of the store's token count. Every feed must start here.
    fed: u32,
    /// Store length when the current request began (cold prefill or
    /// continuation). The prompt-completing chunk only takes the retrieval
    /// policy when the tapped query was produced by THIS request
    /// (`fed > turn_start`): a continuation whose whole suffix is one chunk
    /// still holds the previous turn's final tap, which is not a query for
    /// the new question.
    turn_start: u32,
    /// True once a decode-side selection decision has been taken for the
    /// current request. The FIRST decode step of a request always runs a
    /// selection (when the trunk exceeds the budget), so a new turn's
    /// question — whose Q rows the prompt-completing chunk just captured —
    /// re-scores the resident set immediately instead of inheriting the
    /// previous turn's pages until the block interval elapses. Without it a
    /// follow-up question about a fact in a spilled page answers from
    /// whatever the last turn happened to keep.
    first_decode_done: bool,
    /// Query-span host ring (only allocated when `query_span > 1`): layer-major
    /// `[n_fa_layers × query_span × q_width]` F32 post-RMSNorm/pre-RoPE Q rows.
    /// Each fed step appends the slot's tapped last Q row per layer; the ring
    /// keeps the most recent `query_span` rows for page-softmax scoring. Empty
    /// when `query_span == 1` (the mean-K path reads the tap directly).
    q_ring: Vec<f32>,
    /// Valid rows in `q_ring` (`<= query_span`); shared across layers.
    q_ring_len: usize,
    /// Next write slot in `q_ring` (`0..query_span`); shared across layers.
    q_ring_cursor: usize,
    /// Layerwise session incarnation (design §3.9g). Bumped every time the slot
    /// is reset to a fresh session, so a re-bootstrapped canonical history never
    /// aliases a previous session's `MemoryStamp`. Legacy ignores it.
    lw_incarnation: u64,
    /// Layerwise transaction counter (design §3.9g). Bumped once per admitted
    /// layerwise query; drives the synthesized `MemoryStamp` revisions
    /// (history_revision / view_version / execution_history). The host store
    /// (RawStore) remains the byte-persistence base; under decision A (§3.10)
    /// the selection state additionally persists in `lw_pressure` / `lw_keep`.
    lw_seq: u64,
    /// Session-lifetime prefill attention-mass pressure owner (decision A,
    /// design §3.10). Constructed lazily on the first admitted Layerwise query
    /// of a session ([`KvMemCtl::ensure_layerwise_pressure`]); `None` under
    /// Legacy (never created — byte-neutral) and after [`KvMemCtl::reset_slot`].
    /// Accumulates accepted per-page mass and refines the resident keep-set via
    /// `select` (no cold recall). Replaces the pure re-bootstrap "persist NO
    /// snapshots" model: the production path now calls `accept`.
    lw_pressure: Option<PrefillPressure>,
    /// Session-lifetime OUTPUT-scope attention-mass pressure owner (P4.1②,
    /// design §A.6). The MTP-verify sibling of `lw_pressure`: a verify span
    /// accepts a PREFIX of its rows (partial acceptance), which the all-or-nothing
    /// `PrefillPressure` cannot express (core layerwise_pressure.cpp:79).
    /// Constructed lazily on the first admitted Layerwise MTP verify of a session
    /// ([`KvMemCtl::ensure_layerwise_output_pressure`]); `None` under Legacy /
    /// AR-only sessions (never created — byte-neutral) and after
    /// [`KvMemCtl::reset_slot`]. NonThinking semantic scope (every verify row is
    /// body output), mirroring the ① partial-accept fixture. Coexists with
    /// `lw_pressure`: the output `select` seeds from the prefill snapshot.
    lw_output_pressure: Option<OutputPressure>,
    /// Persistent PER-LAYER resident keep-set (decision A, design §3.10a):
    /// `lw_keep[ordinal]` is the canonical block ids chosen for FA ordinal
    /// `ordinal` by the previous step's `pressure.select` (Pressure step) or the
    /// ② scorer (Query step). Drives each layer's canonical `resident` list
    /// (replacing the re-bootstrap `[]`); this is SELECTION-state persistence
    /// (session-lifetime, cleared on reset_slot), distinct from the host store's
    /// byte persistence. Per-layer because `pressure.select` returns independent
    /// per-layer choices (the composite arena's mixed-stripe destinations let
    /// each FA layer keep its own mass-ranked pages). Empty before the first
    /// selection / under Legacy; an absent ordinal entry reads as `[]`.
    lw_keep: Vec<Vec<u64>>,
    /// P2.2a: one-shot Layerwise selection budget (blocks) from the import
    /// admission (ninfer `selection_budget`). `Cell` so the driver's shared
    /// `&KvMemCtl` can take it; every fresh-session reset clears it.
    lw_import_blocks: Cell<Option<u32>>,
    /// [P4.1② fix] per-request scorer-Query gate (ninfer per-request Query
    /// semantics). Set by `begin_request` when a continuation reuses the prefix;
    /// consumed (take) by the orchestration at the request's first Layerwise
    /// decode/verify step to force a scorer Query so the CURRENT question
    /// re-scores the resident pool (Pressure alone freezes the previous turn's
    /// selection and reuses its stale compact-tail binding, corrupting recall).
    /// Cleared on reset_slot; false under Legacy (never read there).
    lw_query_due: bool,
    /// 期2.3: frontend-proven span of the CURRENT request's user question
    /// (original coordinates). `None` = unproven (historical behaviour).
    request_query: Option<(u64, u64)>,
    /// Span of the PREVIOUS accepted request's question on this slot —
    /// continuation identity for `layerwise_query_at_source`.
    last_query: Option<(u64, u64)>,
    /// 期3 G3: this request's internal content group (conversation-lineage
    /// hash). Fairness hint only; a frozen checkpoint carries it as metadata.
    convo_group: u64,
    /// 期3 G3 follow-up: the slot's last frozen checkpoint — the origin-inherit
    /// source for the next capture (same sealed records ⇒ shared blocks).
    last_freeze_id: Option<u64>,
    /// Pending output-scope acceptance snapshot for the in-flight MTP verify
    /// (P4.1②, design §A.6). The orchestration extracts it from the verify carrier
    /// after the forward (mass captured) and stores it here BEFORE freeing the
    /// carrier and restoring the controller; `mtp_verify_accept_step` takes it
    /// after `feed_rows` and publishes it at the scheduler-confirmed prefix
    /// (`result.advance`), which is known only post-sampling. `None` under Legacy /
    /// AR-only / between verify steps (byte-neutral); cleared on `reset_slot`. The
    /// snapshot is owned, so it survives the scoped controller take that the
    /// carrier's `&KvMemCtl` borrow requires.
    lw_pending_output: Option<OutputAcceptanceSnapshot>,
    /// P4.1② G3: the MTP head's independent aux-plane resident map (design §A.8).
    /// Constructed lazily on the first admitted Layerwise MTP verify
    /// ([`KvMemCtl::ensure_head_plane`]); `None` under Legacy / AR-only (never
    /// created — byte-neutral) and after [`KvMemCtl::reset_slot`]. The head
    /// follows the target's first-layer selection (FirstLayer); this map + the
    /// aux `LayerWorkingSetPlan` + removed-token accounting ARE its independent
    /// "pressure" (no mass owner). Taken/restored around the scoped verify
    /// carrier exactly like the trunk pressure owners.
    lw_head: Option<HeadPlaneState>,
    /// P4.1② G3: the per-Query FirstLayer auxiliary-selection binding (① primitive
    /// [`AuxiliarySelection`]). Bind-once per verify span: the restore-hook binds
    /// the head history from the target's FIRST FA layer's restored ids (G5,
    /// NL:9985-9990), the boundary rebuild consumes it, then the next span
    /// replaces it. `None` under Legacy / AR-only / between verifies (byte-neutral);
    /// cleared on `reset_slot`. Transient like `lw_pending_output`, NOT
    /// session-lifetime like `lw_head`.
    lw_aux_selection: Option<AuxiliarySelection>,
}

/// P4.1② G3: the MTP head's INDEPENDENT aux-plane resident map (design §A.8),
/// the per-slot mirror of ninfer `KvmemLayerViews`' auxiliary fields
/// (ninfer-layerwise-recall.patch NL:12215-12220). The head does NOT mass-score:
/// under FirstLayer it follows the target's first-FA-layer selection, so its
/// "pressure" is this resident map + an independent `LayerWorkingSetPlan`
/// (computed via `MemoryCatalog::plan_layer` for the aux ordinal, quota =
/// `kvmem_window_tokens`) + removed-token accounting — NOT a
/// `PrefillPressure`/`OutputPressure` owner. Session-lifetime (evolves across
/// verifies, dropped on `reset_slot`); `None` under Legacy / AR-only so the
/// default path allocates nothing (byte-neutral).
///
/// Field types mirror ninfer EXACTLY: `auxiliary_pages` is `u32` canonical block
/// ids (NL:12217, widened to `u64` only at the `plan_layer` resident boundary,
/// NL:10575), `auxiliary_view_version` is `u64` (NL:12218), and
/// `auxiliary_removed_tokens` is `u32` (NL:12219). The head pool's block geometry
/// is `KV_BLOCK_TOKENS` (128), vs ninfer's 64.
#[derive(Debug, Clone, Default)]
pub struct HeadPlaneState {
    /// The head's resident canonical block ids, chronological (ninfer
    /// `auxiliary_pages`). Drives the aux `plan_layer` resident list and the
    /// head stripe restore-before-attend. Empty before the first selection.
    auxiliary_pages: Vec<u32>,
    /// Head view version, bumped per admitted head selection (ninfer
    /// `auxiliary_view_version`); drives the synthesized aux `MemoryStamp`
    /// ([`KvMemCtl::plan_head_plane`] reads it at the aux stamp boundary, NL:10576).
    auxiliary_view_version: u64,
    /// Tokens dropped from the head history by drop-evict since the last
    /// boundary rebuild (ninfer `auxiliary_removed_tokens`); the RoPE shift for
    /// the head's restored rows (NL:5848/10328).
    auxiliary_removed_tokens: u32,
}

impl HeadPlaneState {
    /// P4.1② G4 (f2): the head's restored compact page ids (chronological) — the
    /// aux resident list the head `plan_layer` consumes and the write-through
    /// source. Read by the draft phase to build the next selection.
    pub fn auxiliary_pages(&self) -> &[u32] {
        &self.auxiliary_pages
    }
    /// P4.1② G4 (f2): the RoPE shift for the head's restored rows
    /// (`frontier - visible_tokens`, ninfer NL:5848/10328). The draft phase adds
    /// this to each row's global RoPE position.
    pub fn removed_tokens(&self) -> u32 {
        self.auxiliary_removed_tokens
    }
    /// ninfer `KvmemLayerViews::validate` aux invariant (NL:12267-12270):
    /// `visible_tokens(auxiliary_pages, frontier) + auxiliary_removed_tokens ==
    /// frontier`. The visible-token reduction is the ① `auxiliary_visible_tokens`
    /// mirror (block geometry `KV_BLOCK_TOKENS`). Fail-closed: a non-chronological
    /// map or an extent that does not reconstruct the frontier is an error, so a
    /// corrupted head history can never be attended.
    fn validate(&self, frontier: u64) -> Result<u64, String> {
        let visible =
            auxiliary_visible_tokens(&self.auxiliary_pages, frontier, KV_BLOCK_TOKENS as u64)?;
        let extent = visible + self.auxiliary_removed_tokens as u64;
        if extent != frontier {
            return Err(format!(
                "native MTP head view does not reconstruct the accepted frontier \
                 (visible {visible} + removed {} != {frontier})",
                self.auxiliary_removed_tokens
            ));
        }
        Ok(extent)
    }

    /// P4.1② G4 (f2-D): publish the accepted canonical append to the head's
    /// resident map — the hipfire analog of ninfer `accepted_auxiliary_append`
    /// (NL:12309-12314) + its `append` helper (NL:12341-12347, block geometry
    /// `KV_BLOCK_TOKENS` instead of ninfer's 64). After the consume pass seals the
    /// accepted head rows `[begin, end)` into the composite aux plane, the new
    /// canonical pages join the resident selection map and the RoPE shift is
    /// recomputed as `end - visible_tokens(next, end)`. The view version is
    /// UNCHANGED (only a stripe restore bumps it, NL:10611); this is a selection
    /// state advance, not a restore. The map persists across drafts as SELECTION
    /// state only — the head arena is freshly restored every step (device-truth),
    /// exactly like the trunk keep-set ([`combine_keep_set`]).
    ///
    /// P4.1② G6 (§3.9g): the canonical logical frame is the PAGE-ROUNDED accepted
    /// frontier (`frontier / bt * bt`), matching [`Self::accept_head_restore`] and
    /// the trunk keep-set. The partial tail (`frontier % bt` tokens) rides the
    /// PHYSICAL SUFFIX STRIPE (`crate::layerwise_driver::HeadSuffix`) — it is NOT a
    /// canonical aux block,
    /// so ninfer's partial-page anchor guard (NL:12343-12344) does not apply: the
    /// suffix stripe is always resident and is the compact-shift anchor. This is
    /// the faithful adaptation of ninfer `accepted_auxiliary_append` to hipfire's
    /// paged suffix representation — the partial page's DATA is still sealed +
    /// attended via the suffix; only its bookkeeping differs from ninfer's
    /// contiguous canonical map. The RoPE shift is recomputed against the
    /// page-rounded `end` frame so it stays consistent with the next restore.
    ///
    /// Fail-closed on `end < begin`, a non-chronological map, or an extent that
    /// does not reconstruct the page-rounded `end` frame — so a corrupted head
    /// history can never be attended. Pure host logic; Legacy / AR-only never call
    /// it.
    fn append_accepted(&mut self, begin: u64, end: u64) -> Result<(), String> {
        let bt = KV_BLOCK_TOKENS as u64;
        if end < begin {
            return Err(format!(
                "native MTP head append: invalid range [{begin}, {end})"
            ));
        }
        let begin_frame = begin / bt * bt;
        let end_frame = end / bt * bt;
        // Validate the existing map is a chronological accepted view up to the
        // page-rounded `begin` (ninfer `append`'s `visible_tokens(pages, begin)`
        // guard, adapted to the canonical frame).
        auxiliary_visible_tokens(&self.auxiliary_pages, begin_frame, bt)?;
        // Append the newly-sealed FULL pages `[begin/bt, end/bt)`. The old suffix
        // page (`begin/bt`) becomes canonical once the append fills it; the new
        // partial suffix page (`end/bt` when `end % bt != 0`) stays the suffix
        // stripe and is never pushed (it is not a canonical block).
        let mut id = begin / bt;
        while id < end / bt {
            let page = u32::try_from(id)
                .map_err(|_| format!("native MTP head append: page {id} overflow u32"))?;
            if self.auxiliary_pages.last() != Some(&page) {
                self.auxiliary_pages.push(page);
            }
            id += 1;
        }
        let visible = auxiliary_visible_tokens(&self.auxiliary_pages, end_frame, bt)?;
        let removed = end_frame.checked_sub(visible).ok_or_else(|| {
            format!("native MTP head append: visible {visible} exceeds frontier {end_frame}")
        })?;
        self.auxiliary_removed_tokens = u32::try_from(removed)
            .map_err(|_| format!("native MTP head append: removed {removed} overflow u32"))?;
        self.validate(end_frame)?;
        Ok(())
    }
}

/// Per-slot residency plan for one step.
struct StepPlan {
    /// Logical pages selected to stay resident.
    keep: Vec<bool>,
    /// Spilled logical pages selected for restore.
    restore: Vec<usize>,
}

/// Immutable host-owned Legacy recovery point. It owns no GPU page references;
/// restoring forks a private raw/statistics branch and never consumes the source.
/// The engine pairs this object with the exact native recurrent/conv image and
/// accepted token proof at `frontier` before publishing it as reusable history.
pub struct LegacyStoreSnapshot {
    raw: RawStore,
    pub frontier: u32,
    pub resident_pages: Vec<u32>,
    mean_tail: Option<Vec<f32>>,
    retrieval: hipfire_kvmem::RetrievalCheckpoint,
    /// Retains the core's immutable sealed origins and proves every layer/page
    /// was captured, including the valid portion of a non-full tail page.
    sealed: hipfire_kvmem::KvSnapshot,
    layout_id: u64,
    request_query: Option<(u64, u64)>,
    last_query: Option<(u64, u64)>,
    group: u64,
}

impl LegacyStoreSnapshot {
    pub fn payload_bytes(&self) -> u64 {
        self.sealed.bytes()
    }
}

pub struct KvMemCtl {
    /// Declared before `storage` so the per-slot stores (which hold a
    /// shared_ptr into it) are destroyed first.
    slots: Vec<SlotState>,
    #[allow(dead_code)]
    storage: Storage,
    raw_cfg: RawStoreConfig,
    kv_cfg: KvStoreConfig,
    /// Admitted public Legacy policy. None is the unchanged Layerwise/internal
    /// route; public consumers never re-read the corresponding environment keys.
    legacy: Option<hipfire_config::kvmem::ResolvedConfig>,
    budget_blocks: usize,
    /// P2.2a generation reserve R (ninfer `reserve_tokens`): layerwise window
    /// capacity is `budget_blocks*128 + gen_reserve`. `HIPFIRE_KVMEM_GEN_RESERVE`.
    gen_reserve: u64,
    interval: usize,
    diagnostics: KvMemDiagnostics,
    n_fa_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// P4.4: KV tier (per-tier layerwise format identity + admission).
    kv_mode: KvMode,
    q_width: usize,
    k_page_bytes: usize,
    v_page_bytes: usize,
    /// Packed page staging (one page per arena).
    page_k: Vec<u8>,
    page_v: Vec<u8>,
    /// Layerwise write-through staging (design §3.7). Only sized when
    /// `strategy == Layerwise`: contiguous packed Q8 K/V rows for one feed call
    /// (`max_rows * per_pos_bytes`), gathered from the GPU arenas before the
    /// sealed-archive write. Empty under Legacy (no allocation, byte-neutral).
    wt_k: Vec<u8>,
    wt_v: Vec<u8>,
    /// Tap readback staging (`read_k_rows_at`).
    staging: Vec<f32>,
    /// Retrieval query staging: layer-major raw Q rows.
    sums: Vec<f32>,
    q_row: Vec<f32>,
    counts: Vec<u32>,
    /// Reusable pinned slab for the batched spill/restore path. Grown on demand
    /// (never shrunk); each chunk is drained by a stream wait before the slab is
    /// refilled, so no async copy is ever pending across a reuse or a drop.
    xfer_slab: Option<hip_bridge::PinnedTransferBuffer>,
    /// Resolved `HIPFIRE_KVMEM_BATCH_XFER` (default on). When false, or when no
    /// steady-copy stream is active, spill/restore use the per-page path.
    batch_xfer: bool,
    /// Resolved `HIPFIRE_KVMEM_RECALL_STRATEGY` (default Legacy). Gates every
    /// Layerwise-only path (write-through archive, per-layer driver, packed
    /// residency); Legacy keeps the historical serve bit-identical.
    strategy: RecallStrategy,
    /// Resolved layerwise activation quotas (design §3.11, ③-E1). Only the
    /// Layerwise carrier creates accounts against these; under Legacy the
    /// ninfer seeds are carried unread (byte-neutral).
    layerwise_quotas: LayerwiseQuotas,
    /// Resolved `HIPFIRE_KVMEM_SELECT_METHOD` ordinal (0 Retrieval, 2 Recency).
    /// H2O (1) is rejected at construction until the attention-heat tap lands.
    select_method: u32,
    /// `HIPFIRE_KVMEM_GPU_HIGH/LOW_WATERMARK` hysteresis band, in pages, when
    /// either env is set: `(spill_trigger, keep_target)`. The core watermark
    /// ratios are applied to the per-slot resident budget pool, so
    /// `keep_target <= spill_trigger <= budget_blocks` — the band can only make
    /// residency more conservative than the default, never overshoot the pool.
    /// `None` (default) keeps the historical single-threshold budget behaviour.
    watermark: Option<(usize, usize)>,
    /// Resolved `HIPFIRE_KVMEM_QUERY_SPAN` (default 1, range 1..=512). At 1 the
    /// retrieval query is the single tapped last Q row scored by
    /// `mean_key_retrieval_scores` (byte-identical to the historical path). At
    /// >1 the per-slot host ring accumulates recent Q rows and selection scores
    /// blocks by `page_softmax_scores` over that span.
    query_span: usize,
    /// Page-softmax query staging (only sized when `query_span > 1`):
    /// layer-major `[n_fa_layers × query_span × q_width]` chronological Q rows.
    q_span: Vec<f32>,
    /// Per-layer present row count for the page-softmax view (`query_span > 1`).
    q_layer_rows: Vec<u32>,
    /// P4.1② G3: true iff Layerwise AND the model runs the native MTP head
    /// (`geom.mtp_head`). Gates the composite host record (`n_fa_layers + 1`
    /// ordinals) and every head-plane path. False under Legacy / AR-only, so the
    /// default serve keeps the historical `n_fa_layers` record byte-neutral.
    head_plane: bool,
    /// P4.1③ H5: resolved `HIPFIRE_KVMEM_LAYERWISE_MTP_HISTORY` head-history
    /// policy. FirstLayer (the ninfer auto-default) binds the head aux from the
    /// trunk's first-layer restored view; UpstreamGlobal runs the global Mean-K
    /// private-probe selection (`layerwise_probe`). Only meaningful when
    /// `head_plane`; under Legacy / AR-only it is Unspecified and never read, so
    /// the default serve stays byte-neutral.
    layerwise_mtp_history: LayerwiseMtpHistory,
    /// P4.1② G3: the head aux plane's ordinal in the composite host record
    /// (`= n_fa_layers`, the appended LAST ordinal; ninfer
    /// `model = layer_types.size()`, NL:10577). Only meaningful when `head_plane`;
    /// the trunk GPU paged arena stays `n_fa_layers` and never addresses it.
    aux_ordinal: u32,
    /// P4.1② G3: the composite host record's layer count (`n_fa_layers + 1` when
    /// `head_plane`, else `n_fa_layers`). The trunk retrieval query must pass THIS
    /// (not `n_fa_layers`) to satisfy the core's `query.layers == n_layer` geometry
    /// check; the appended aux ordinal is skipped by zeroing its token count /
    /// layer rows, so trunk selection still scores only `0..n_fa_layers`. Equals
    /// `n_fa_layers` under Legacy / AR-only ⇒ the default query is byte-neutral.
    n_record_layers: usize,
    /// P4.3: engine-wide frozen-prefix catalog (KVMem-native sharing). Empty
    /// unless the serve engine freezes a prefix (`HIPFIRE_KVMEM_PREFIX_PIN`
    /// on); an empty catalog allocates nothing and no default path reads it,
    /// so the Legacy serve stays byte-neutral.
    prefix_catalog: PrefixCatalog,
    /// P3.4 retention policy `(count, bytes, interval, min_gap)` when the spread
    /// checkpoint policy is active; `None` keeps the historical LRU-only catalog.
    checkpoint_retention: Option<(usize, u64, u64, u64)>,
    /// 期3 G3: `HIPFIRE_KVMEM_CONTENT_GROUPS` (0 = plain LRU catalog; N > 0
    /// enables group-aware publication with `max_groups = N` and the physical
    /// version limit `C = max(2N, 1)`).
    content_groups: usize,
    /// Entries retired by the group-aware publication plan.
    content_retirements_total: u64,
    /// 期3 G3 P1.1: persistent pack directory (same cache dir) and byte cap.
    cold_dir: Option<PathBuf>,
    cold_bytes: u64,
    /// Pack files discovered at startup / registered at freeze (metadata only).
    cold_packs: ColdPackIndex,
    /// Catalog entries displaced by a cold pack load (the engine releases
    /// their GPU cache refs after `prefix_lookup`).
    cold_displaced: Vec<PrefixCheckpoint>,
    cold_pack_hits_total: u64,
    cold_pack_misses_total: u64,
    /// Observability counters (`KvMemCtl::stats`). Pure bookkeeping: no path
    /// reads them back into selection/residency, so they cannot change KV
    /// bytes; `/metrics` exports them via `EngineStats.kvmem`.
    spill_pages_total: u64,
    restore_pages_total: u64,
    prefix_freezes_total: u64,
    prefix_hits_total: u64,
    prefix_misses_total: u64,
    prefix_evictions_total: u64,
    /// G3 follow-up: hot catalog hits (never read a pack) and reuse
    /// conversions that restored a demoted/restart entry from the archive.
    prefix_catalog_hits_total: u64,
    prefix_restore_hits_total: u64,
    /// G3 follow-up: admission-pressure residency demotions (entry retained,
    /// pinned pages dropped; Layerwise restores them from the archive).
    prefix_demotions_total: u64,
    /// 期4: DFlash2 draft conditioning images captured with prefix
    /// checkpoints, restored on reuse, and missing/unrestorable (reseed).
    draft_images_total: u64,
    draft_image_hits_total: u64,
    draft_image_misses_total: u64,
}

/// `SPILLED_PAGE` is re-exported for engine-side diagnostics; keep the import
/// honest for builds where the constant is only used in docs.
#[allow(dead_code)]
const _: u32 = SPILLED_PAGE;

#[cfg(test)]
mod tests;
