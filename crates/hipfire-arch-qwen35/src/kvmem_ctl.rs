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

use std::path::PathBuf;

use hipfire_kvmem::{
    mean_key_retrieval_scores, page_softmax_scores, GqaAggregation, KvStore, KvStoreConfig,
    PrefillOptions, PrefillPressure, RawStore, RawStoreConfig, SessionIdentity, Storage,
};
use rdna_compute::page_pool::SPILLED_PAGE;
use rdna_compute::slot_pool::{SlotId, SlotPool};
use rdna_compute::{Gpu, GpuTensor};

use crate::kvmem_taps::KvMemTaps;

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

/// Host payload capacity for the packed Q8 pages (RAM tier).
const DEFAULT_HOST_BYTES: u64 = 16 << 30;

/// Disk-tier capacity when `HIPFIRE_KVMEM_SPILL_DIR` is set.
const DEFAULT_SPILL_BYTES: u64 = 64 << 30;

/// Upper bound on the reusable pinned transfer slab used by the batched
/// spill/restore path (`HIPFIRE_KVMEM_BATCH_XFER`, default on). Pages are
/// gathered into the slab in chunks of at most this many bytes, each chunk
/// drained with ONE stream wait, so pinned host RAM stays bounded regardless
/// of how many pages a step moves. A single page larger than the cap still
/// fits (the slab grows to one page). Mirrors llama's 32 MiB stage-in slab.
const XFER_SLAB_CAP: usize = 64 << 20;

fn env_raw(name: &str) -> Option<String> {
    hipfire_config::developer_var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn env_flag(name: &str) -> bool {
    matches!(
        env_raw(name).as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

fn env_usize(name: &str) -> Option<usize> {
    env_raw(name).and_then(|v| v.parse::<usize>().ok())
}

/// Diagnostic (`HIPFIRE_KVMEM_VERIFY=1`): checksum spilled/restored page
/// bytes and read back each restored page, so a silent page-copy mismatch is
/// visible in the log. Debug only; no effect on the copy itself.
fn verify_flag() -> bool {
    env_flag("HIPFIRE_KVMEM_VERIFY")
}

fn fnv64(buf: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in buf {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `HIPFIRE_KVMEM_TIME=1`: print per-call breakdowns of the controller's
/// host-side work (feed / selection / spills / restores) so the KVMem cost
/// can be separated from the forward's. Off = no clocks taken.
fn time_flag() -> bool {
    env_flag("HIPFIRE_KVMEM_TIME")
}

/// `HIPFIRE_KVMEM_BATCH_XFER` (default on): gather spilled/restored pages into
/// a reusable pinned slab and move each chunk with stream-ordered async copies
/// drained by ONE wait, instead of a blocking null-stream memcpy per page per
/// layer. Byte-neutral (same payloads, same arena offsets); set to `0` to fall
/// back to the per-page path for A/B timing or rollback. The batched path also
/// requires an active steady-copy stream and is skipped during graph capture.
fn batch_xfer_flag() -> bool {
    !matches!(
        env_raw("HIPFIRE_KVMEM_BATCH_XFER").as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

fn env_u64(name: &str) -> Option<u64> {
    env_raw(name).and_then(|v| v.parse::<u64>().ok())
}

fn env_f64(name: &str) -> Option<f64> {
    env_raw(name).and_then(|v| v.parse::<f64>().ok())
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
/// value (already trimmed by [`env_raw`]). Absent → Legacy (the byte-neutral
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
        return Err(
            "kvmem: LRFU=1 incompatible with RECALL_STRATEGY=layerwise \
             (ninfer forces lrfu=false; per-layer mass/exposure replaces the \
             decay history); unset HIPFIRE_KVMEM_LRFU"
                .to_string(),
        );
    }
    if watermark_enabled {
        return Err(
            "kvmem: GPU_HIGH/LOW_WATERMARK incompatible with \
             RECALL_STRATEGY=layerwise (admission-driven pressure replaces the \
             global watermark hysteresis); unset HIPFIRE_KVMEM_GPU_HIGH_WATERMARK \
             and HIPFIRE_KVMEM_GPU_LOW_WATERMARK"
                .to_string(),
        );
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
        activation_ram: mib(ram, "HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_RAM_MIB", d.activation_ram)?,
        activation_disk: mib(disk, "HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_DISK_MIB", d.activation_disk)?,
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

/// True when `HIPFIRE_KVMEM` names an enabling value. Callers must also check
/// that the crate was built with the `kvmem` feature — the engine refuses the
/// knob loudly when it was not.
pub fn requested() -> bool {
    env_flag("HIPFIRE_KVMEM")
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
) -> Result<KvMemPlan, String> {
    if n_slots == 0 || cap_tokens == 0 {
        return Err("kvmem: engine has no slots or no token capacity".to_string());
    }
    let cap_blocks = cap_tokens.div_ceil(KV_BLOCK_TOKENS).max(1);
    let grow_blocks = max_step_rows.div_ceil(KV_BLOCK_TOKENS).max(1);
    let interval = env_usize("HIPFIRE_KVMEM_SELECT_INTERVAL")
        .unwrap_or(DEFAULT_SELECT_INTERVAL)
        .clamp(1, 256);
    // Residency transiently overshoots the budget by one step's growth plus
    // the blocks written before the next selection can fire. One prefill
    // chunk is written in a single forward, so the whole chunk must fit on
    // top of the budget — `max_step_rows` is therefore the largest
    // per-slot row count, not the whole batch's.
    let headroom = grow_blocks
        .saturating_add(interval)
        .saturating_add(1); // the partial tail page
    let explicit_pages = env_usize("HIPFIRE_SLOTS_PAGED_PAGES").map(|p| p.max(n_slots));
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
    let budget_blocks = match env_usize("HIPFIRE_KVMEM_BUDGET_TOKENS") {
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

/// Model geometry the controller has to address.
#[derive(Debug, Clone, Copy)]
pub struct KvMemGeometry {
    pub n_fa_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// K arena bytes per position (packed Q8 row of one KV head set).
    pub per_pos_bytes: usize,
    /// V arena bytes per position.
    pub per_pos_v_bytes: usize,
    /// Tap buffer row capacity (`PrefillBatchScratch::max_batch`).
    pub max_rows: usize,
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
}

/// Per-slot residency plan for one step.
struct StepPlan {
    /// Logical pages selected to stay resident.
    keep: Vec<bool>,
    /// Spilled logical pages selected for restore.
    restore: Vec<usize>,
}

pub struct KvMemCtl {
    /// Declared before `storage` so the per-slot stores (which hold a
    /// shared_ptr into it) are destroyed first.
    slots: Vec<SlotState>,
    #[allow(dead_code)]
    storage: Storage,
    raw_cfg: RawStoreConfig,
    kv_cfg: KvStoreConfig,
    budget_blocks: usize,
    interval: usize,
    trace: bool,
    n_fa_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
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
}

impl KvMemCtl {
    pub fn new(
        n_slots: usize,
        plan: &KvMemPlan,
        geom: &KvMemGeometry,
    ) -> Result<Self, String> {
        let strategy = resolve_recall_strategy(env_raw("HIPFIRE_KVMEM_RECALL_STRATEGY"))?;
        Self::new_with_strategy(n_slots, plan, geom, strategy)
    }

    /// Construct with the recall strategy injected instead of read from the
    /// process config. The config is a START-TIME SNAPSHOT
    /// (`hipfire_config::active_or_local_process_config` is a `OnceLock`), so a
    /// test that `set_var`s `HIPFIRE_KVMEM_RECALL_STRATEGY` after the first
    /// config read is an order-dependent coin flip — inject here instead
    /// (mirrors `HfqFile::open_with_reap_plan`). [`Self::new`] resolves the env
    /// once and delegates, so the production path is unchanged.
    fn new_with_strategy(
        n_slots: usize,
        plan: &KvMemPlan,
        geom: &KvMemGeometry,
        strategy: RecallStrategy,
    ) -> Result<Self, String> {
        if n_slots == 0 || geom.n_fa_layers == 0 || geom.head_dim == 0 {
            return Err("kvmem: zero-sized controller request".to_string());
        }
        let k_width = geom.n_kv_heads * geom.head_dim;
        let q_width = geom.n_heads * geom.head_dim;
        let raw_cfg = RawStoreConfig {
            n_layer: geom.n_fa_layers as u32,
            n_embd_k: k_width as u32,
            n_embd_v: k_width as u32,
            block_tokens: KV_BLOCK_TOKENS as u32,
            k_row_bytes: 0,
            k_gpu_row_bytes: geom.per_pos_bytes as u64,
            v_gpu_row_bytes: geom.per_pos_v_bytes as u64,
        };
        let sink = env_usize("HIPFIRE_KVMEM_SINK_BLOCKS").unwrap_or(1);
        let recent = env_usize("HIPFIRE_KVMEM_RECENT_BLOCKS").unwrap_or(1);
        if sink + recent > plan.budget_blocks {
            return Err(format!(
                "kvmem: sink ({sink}) + recent ({recent}) blocks exceed the \
                 residency budget ({}); lower HIPFIRE_KVMEM_SINK_BLOCKS / \
                 HIPFIRE_KVMEM_RECENT_BLOCKS or raise the budget",
                plan.budget_blocks
            ));
        }
        let budget_tokens = (plan.budget_blocks * KV_BLOCK_TOKENS) as u32;

        // ── opt-in policy knobs (defaults reproduce the historical behaviour) ──
        // LRFU replaces the raw retrieval score with a decaying heavy-hitter
        // history in the core's `selection_score`; lambda/top tune that decay.
        let lrfu = env_flag("HIPFIRE_KVMEM_LRFU");
        let lrfu_lambda = env_f64("HIPFIRE_KVMEM_LRFU_LAMBDA").unwrap_or(0.5);
        let lrfu_top = env_f64("HIPFIRE_KVMEM_LRFU_TOP").unwrap_or(0.9);
        let select_method = match env_raw("HIPFIRE_KVMEM_SELECT_METHOD") {
            Some(v) => parse_select_method(&v).ok_or_else(|| {
                format!("kvmem: HIPFIRE_KVMEM_SELECT_METHOD={v} invalid (retrieval|h2o|recency)")
            })?,
            None => 0,
        };
        if select_method == 1 {
            return Err(
                "kvmem: SELECT_METHOD=h2o needs the per-block attention-heat tap \
                 (hipfire-h2o-heat patch); use retrieval or recency for now"
                    .to_string(),
            );
        }
        // The core KvMemStore constructor enforces this; fail here with a clear
        // message instead of an opaque C++ exception at store creation.
        if lrfu && select_method != 0 {
            return Err("kvmem: LRFU requires SELECT_METHOD=retrieval (core constraint)".to_string());
        }
        if lrfu {
            if !(0.0..=1.0).contains(&lrfu_lambda) {
                return Err(format!("kvmem: LRFU_LAMBDA={lrfu_lambda} out of range [0,1]"));
            }
            if !(lrfu_top > 0.0 && lrfu_top <= 1.0) {
                return Err(format!("kvmem: LRFU_TOP={lrfu_top} out of range (0,1]"));
            }
        }
        // Watermark ratios are stored on the config; the page thresholds are
        // derived from the CORE watermark functions after the stores exist.
        let wm_high_env = env_f64("HIPFIRE_KVMEM_GPU_HIGH_WATERMARK");
        let wm_low_env = env_f64("HIPFIRE_KVMEM_GPU_LOW_WATERMARK");
        let watermark_enabled = wm_high_env.is_some() || wm_low_env.is_some();
        let wm_high = wm_high_env.unwrap_or(0.95);
        let wm_low = wm_low_env.unwrap_or(0.85);
        if watermark_enabled
            && (!(wm_high > 0.0 && wm_high <= 1.0)
                || !(wm_low > 0.0 && wm_low <= 1.0)
                || wm_low > wm_high)
        {
            return Err(format!(
                "kvmem: GPU_HIGH/LOW_WATERMARK must satisfy 0 < low <= high <= 1 \
                 (got high={wm_high}, low={wm_low})"
            ));
        }

        // Query span: 1 keeps the single-row mean-K retrieval (byte-identical);
        // >1 accumulates a per-slot host Q ring and scores by page-softmax.
        let query_span = env_usize("HIPFIRE_KVMEM_QUERY_SPAN").unwrap_or(1);
        if !(1..=512).contains(&query_span) {
            return Err(format!(
                "kvmem: HIPFIRE_KVMEM_QUERY_SPAN={query_span} out of range [1,512]"
            ));
        }

        // ── Layerwise fail-closed policy guards (design §4, R2d-D1) ──
        // Pure decision table (see resolve_layerwise_policy): under Layerwise
        // each global working-set knob the per-layer subsystem replaces is an
        // explicit Err; Legacy is admitted unconditionally (range checks above
        // already ran), so the default serve stays byte-neutral.
        resolve_layerwise_policy(strategy, select_method, lrfu, watermark_enabled, query_span)?;

        // Recall execution model. Legacy (the default) keeps the historical
        // whole-working-set path byte-identical. Layerwise enables the per-layer
        // composite-slot subsystem: D2 adds the write-through sealed host
        // archive; the driver + packed GPU residency land in D3/D4. The strategy
        // is injected by `new` (env) or `new_with_strategy` (tests) so every
        // downstream Layerwise-only path shares one switch.

        let kv_cfg = KvStoreConfig {
            block_tokens: KV_BLOCK_TOKENS as u32,
            select_budget: budget_tokens,
            prefill_budget: 0,
            gen_budget: budget_tokens,
            sink_blocks: sink as u32,
            recent_blocks: recent as u32,
            select_method,
            lrfu: lrfu as u32,
            lrfu_lambda,
            lrfu_top_fraction: lrfu_top,
            gpu_high_watermark: wm_high,
            gpu_low_watermark: wm_low,
            ..KvStoreConfig::default()
        };

        let host_bytes = env_u64("HIPFIRE_KVMEM_HOST_BYTES")
            .unwrap_or(DEFAULT_HOST_BYTES)
            .max(1);
        let storage = match env_raw("HIPFIRE_KVMEM_SPILL_DIR") {
            Some(dir) => {
                let dir = PathBuf::from(dir);
                std::fs::create_dir_all(&dir)
                    .map_err(|e| format!("kvmem spill dir {}: {e}", dir.display()))?;
                let spill_bytes = env_u64("HIPFIRE_KVMEM_SPILL_BYTES")
                    .unwrap_or(DEFAULT_SPILL_BYTES)
                    .max(1);
                Storage::new_tiered(host_bytes, &dir, spill_bytes)
            }
            None => Storage::new_ram(host_bytes),
        }
        .map_err(|e| format!("kvmem storage: {e}"))?;

        let mut slots = Vec::with_capacity(n_slots);
        let ring_cap = if query_span > 1 {
            geom.n_fa_layers * query_span * q_width
        } else {
            0
        };
        for _ in 0..n_slots {
            slots.push(SlotState {
                raw: RawStore::new(raw_cfg, Some(&storage))
                    .map_err(|e| format!("kvmem raw store: {e}"))?,
                kv: KvStore::new(&kv_cfg).map_err(|e| format!("kvmem store: {e}"))?,
                last_select_full: 0,
                fed: 0,
                turn_start: 0,
                first_decode_done: false,
                q_ring: vec![0f32; ring_cap],
                q_ring_len: 0,
                q_ring_cursor: 0,
                lw_incarnation: 0,
                lw_seq: 0,
                lw_pressure: None,
                lw_keep: Vec::new(),
            });
        }

        // Derive the watermark hysteresis band from the CORE watermark functions
        // applied to the per-slot resident budget pool (budget_blocks pages):
        // keep_target <= spill_trigger <= budget_blocks, so the band can only
        // make residency more conservative than the default, never overshoot the
        // physical pool. The keep target narrows the core selection budget via
        // set_runtime_select_budget (re-applied on reset_slot).
        let watermark = if watermark_enabled {
            let high_tok = slots[0].kv.gpu_high_watermark_tokens(budget_tokens);
            let low_tok = slots[0].kv.gpu_low_watermark_tokens(budget_tokens);
            let trigger = (high_tok as usize / KV_BLOCK_TOKENS).clamp(1, plan.budget_blocks);
            let keep = (low_tok as usize / KV_BLOCK_TOKENS).clamp(1, trigger);
            if keep < sink + recent {
                return Err(format!(
                    "kvmem: watermark keep target ({keep} blocks) is below sink+recent \
                     ({}); raise GPU_LOW_WATERMARK or lower SINK/RECENT_BLOCKS",
                    sink + recent
                ));
            }
            for st in &slots {
                st.kv
                    .set_runtime_select_budget((keep * KV_BLOCK_TOKENS) as u32)
                    .map_err(|e| format!("kvmem watermark select budget: {e}"))?;
            }
            Some((trigger, keep))
        } else {
            None
        };

        if (time_flag() || env_flag("HIPFIRE_KVMEM_TRACE"))
            && (lrfu || select_method != 0 || watermark_enabled || query_span > 1 || strategy.is_layerwise())
        {
            eprintln!(
                "[kvmem] policy: strategy={strategy:?} select_method={select_method} lrfu={lrfu} \
                 lambda={lrfu_lambda} top={lrfu_top} watermark={watermark:?} \
                 query_span={query_span} budget_blocks={}",
                plan.budget_blocks
            );
        }

        // Resolve the layerwise activation quotas only under Layerwise; Legacy
        // carries the unread ninfer seeds so a malformed quota env can never
        // change the byte-neutral default path (design §3.11).
        let layerwise_quotas = if strategy.is_layerwise() {
            resolve_layerwise_quotas(
                env_raw("HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_RAM_MIB"),
                env_raw("HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_DISK_MIB"),
                env_raw("HIPFIRE_KVMEM_LAYERWISE_STAGING_MIB"),
                env_raw("HIPFIRE_KVMEM_LAYERWISE_METADATA_MIB"),
                env_raw("HIPFIRE_KVMEM_LAYERWISE_ACTIVATION_METADATA_MIB"),
                env_raw("HIPFIRE_KVMEM_SPILL_DIR").map(PathBuf::from),
            )?
        } else {
            LayerwiseQuotas::default()
        };

        Ok(Self {
            slots,
            storage,
            raw_cfg,
            kv_cfg,
            budget_blocks: plan.budget_blocks,
            interval: plan.interval,
            trace: env_flag("HIPFIRE_KVMEM_TRACE"),
            n_fa_layers: geom.n_fa_layers,
            n_heads: geom.n_heads,
            n_kv_heads: geom.n_kv_heads,
            head_dim: geom.head_dim,
            q_width,
            k_page_bytes: KV_BLOCK_TOKENS * geom.per_pos_bytes,
            v_page_bytes: KV_BLOCK_TOKENS * geom.per_pos_v_bytes,
            page_k: vec![0u8; KV_BLOCK_TOKENS * geom.per_pos_bytes],
            page_v: vec![0u8; KV_BLOCK_TOKENS * geom.per_pos_v_bytes],
            wt_k: vec![0u8; if strategy.is_layerwise() { geom.max_rows * geom.per_pos_bytes } else { 0 }],
            wt_v: vec![0u8; if strategy.is_layerwise() { geom.max_rows * geom.per_pos_v_bytes } else { 0 }],
            staging: vec![0f32; geom.max_rows * k_width],
            sums: vec![0f32; geom.n_fa_layers * q_width],
            q_row: vec![0f32; q_width],
            counts: vec![1u32; geom.n_fa_layers],
            xfer_slab: None,
            batch_xfer: batch_xfer_flag(),
            strategy,
            layerwise_quotas,
            select_method,
            watermark,
            query_span,
            q_span: vec![0f32; ring_cap],
            q_layer_rows: vec![0u32; if query_span > 1 { geom.n_fa_layers } else { 0 }],
        })
    }

    /// Resolved recall execution model (`HIPFIRE_KVMEM_RECALL_STRATEGY`).
    /// The engine gates the Layerwise subsystem (write-through archive,
    /// per-layer driver, packed residency) on this; Legacy is the default.
    pub fn strategy(&self) -> RecallStrategy {
        self.strategy
    }

    /// Resolved layerwise activation quotas (design §3.11, ③-E1). The Layerwise
    /// carrier creates its projected-Q tape accounts against these; Legacy never
    /// reads them. Borrowed for the carrier's lifetime (the accounts are built
    /// per query from the copied limits, so no borrow outlives the controller).
    pub fn layerwise_quotas(&self) -> &LayerwiseQuotas {
        &self.layerwise_quotas
    }

    /// Borrow slot `slot`'s statistics store (`RawStore`) and policy (`KvStore`)
    /// for the Layerwise driver (design §3.9a/§3.9b). `LayerwiseDriver<'a>`
    /// borrows these for the query's lifetime; the caller must NOT [`Self::reset_slot`]
    /// while the borrow is live — reset replaces the `SlotState` wholesale (a new
    /// `RawStore`/`KvStore`), invalidating it, so the driver is rebuilt per query.
    /// `None` when the slot is out of range. The stores stay owned by this
    /// controller: the driver only reads statistics and sets the runtime select
    /// budget through them (both `&self` operations on the C++ handles).
    pub fn layerwise_stores(&self, slot: usize) -> Option<(&RawStore, &KvStore)> {
        self.slots.get(slot).map(|st| (&st.raw, &st.kv))
    }

    /// Synthesize a fresh, page-aligned selection policy for one layerwise
    /// query (design §3.9g refinement). The core layerwise picker
    /// (`prepare_selection`) admits ONLY a canonical index of sealed full pages
    /// (`total_tokens % block == 0`, every block full), but the shared per-slot
    /// `kv` store tracks ALL fed tokens — including the partial tail — for the
    /// Legacy path. So the layerwise carrier owns this fresh policy indexing the
    /// `fed / 128` sealed canonical pages; the partial tail lives only in the
    /// statistics store (`raw`) and is restored as the suffix stripe. Scoped to
    /// the query and never shared, so the Legacy `kv` store is untouched
    /// (byte-neutral). The driver's `begin_query` sets the runtime select budget,
    /// so the config's initial budget is only a placeholder here.
    pub fn layerwise_policy(&self, fed: u32) -> Result<KvStore, String> {
        let full_pages = fed / KV_BLOCK_TOKENS as u32;
        let policy =
            KvStore::new(&self.kv_cfg).map_err(|e| format!("kvmem layerwise policy: {e}"))?;
        policy
            .register_append(full_pages * KV_BLOCK_TOKENS as u32)
            .map_err(|e| format!("kvmem layerwise policy append: {e}"))?;
        Ok(policy)
    }

    /// Per-slot resident page budget (full pages + partial tail).
    pub fn budget_blocks(&self) -> usize {
        self.budget_blocks
    }

    /// Drop the slot's host state. Called when a fresh session takes the slot
    /// and lazily when a closed slot is swept at step start.
    pub fn reset_slot(&mut self, slot: usize) -> Result<(), String> {
        if slot >= self.slots.len() {
            return Err(format!("kvmem: reset_slot slot {slot} out of range"));
        }
        let raw = RawStore::new(self.raw_cfg, Some(&self.storage))
            .map_err(|e| format!("kvmem raw store: {e}"))?;
        let kv = KvStore::new(&self.kv_cfg).map_err(|e| format!("kvmem store: {e}"))?;
        // A fresh store starts at the configured select_budget; re-apply the
        // watermark keep target so a reset slot keeps the same hysteresis band.
        if let Some((_, keep)) = self.watermark {
            kv.set_runtime_select_budget((keep * KV_BLOCK_TOKENS) as u32)
                .map_err(|e| format!("kvmem watermark select budget: {e}"))?;
        }
        // A fresh session bumps the layerwise incarnation so re-bootstrapped
        // canonical stamps never alias the previous session's (design §3.9g).
        let lw_incarnation = self.slots[slot].lw_incarnation + 1;
        self.slots[slot] = SlotState {
            raw,
            kv,
            last_select_full: 0,
            fed: 0,
            turn_start: 0,
            first_decode_done: false,
            q_ring: vec![
                0f32;
                if self.query_span > 1 {
                    self.n_fa_layers * self.query_span * self.q_width
                } else {
                    0
                }
            ],
            q_ring_len: 0,
            q_ring_cursor: 0,
            lw_incarnation,
            lw_seq: 0,
            // Decision A (§3.10): a fresh session drops the pressure owner and
            // its keep-set. The next admitted query lazily reconstructs the
            // owner under the bumped incarnation, so accepted mass / selected
            // pages never alias the previous session. Legacy: both stay empty.
            lw_pressure: None,
            lw_keep: Vec::new(),
        };
        Ok(())
    }

    /// Record that a request begins on `slot` (cold prefill or a
    /// continuation that reuses the existing prefix). Returns the store
    /// length the request starts from — a continuation's reuse point must
    /// equal it, because `feed_rows` only extends contiguously.
    pub fn begin_request(&mut self, slot: usize) -> Result<u32, String> {
        let st = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| format!("kvmem: begin_request slot {slot} out of range"))?;
        st.turn_start = st.fed;
        st.first_decode_done = false;
        // A new turn's question must not inherit the previous turn's Q span.
        st.q_ring_len = 0;
        st.q_ring_cursor = 0;
        Ok(st.fed)
    }

    /// Host-store length of `slot` (tokens whose statistics are recorded).
    pub fn fed_tokens(&self, slot: usize) -> Option<u32> {
        self.slots.get(slot).map(|s| s.fed)
    }

    /// Slot `slot`'s layerwise stamp counters `(incarnation, seq)` (design
    /// §3.9g). The orchestration reads these to synthesize a fresh
    /// `MemoryStamp` per re-bootstrapped query; `None` when out of range.
    pub fn layerwise_counters(&self, slot: usize) -> Option<(u64, u64)> {
        self.slots.get(slot).map(|s| (s.lw_incarnation, s.lw_seq))
    }

    /// Advance slot `slot`'s layerwise transaction counter after an admitted
    /// query (design §3.9g). No-op when out of range. Legacy never calls it.
    pub fn advance_layerwise_seq(&mut self, slot: usize) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_seq += 1;
        }
    }

    /// Lazily construct slot `slot`'s session-lifetime prefill pressure owner
    /// (decision A, design §3.10). Idempotent within a session: once built it is
    /// reused for every admitted query until [`Self::reset_slot`] drops it (the
    /// incarnation bump on reset guarantees a fresh session reconstructs under a
    /// distinct identity). `session_id` / `incarnation` are the ENGINE's 0-based
    /// counters; they are normalized to the core's 1-based logical identity with
    /// the SAME `+1` bijection `bootstrap_layerwise` applies, so the owner's
    /// session matches the canonical `MemoryStamp.session` the picker validates.
    /// `fa_layers` are the dense FA ordinals `0..n_fa_layers`.
    ///
    /// Options pin `block_tokens = 128` (the catalog layout block size, which the
    /// core requires to match exactly) and derive the select ceiling from the
    /// per-slot resident budget; interval/stride/half_life carry the ninfer
    /// defaults. Layerwise-only: under Legacy this returns `Err` (fail-closed —
    /// the owner is never created, keeping the default path byte-neutral).
    /// Admission is finite and fail-closed BEFORE any acquisition: the owner's
    /// complete mutable-history ceiling must fit the metadata pressure partition
    /// (③-E4, [`resolve_pressure_admission`]), mirroring ninfer's allocation-phase
    /// contract.
    pub fn ensure_layerwise_pressure(
        &mut self,
        slot: usize,
        session_id: u64,
        incarnation: u64,
        fa_layers: &[u32],
    ) -> Result<(), String> {
        resolve_pressure_admission(
            self.strategy,
            self.n_fa_layers,
            self.budget_blocks,
            &self.layerwise_quotas,
        )?;
        let st = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| format!("kvmem: ensure_layerwise_pressure slot {slot} out of range"))?;
        if st.lw_pressure.is_some() {
            return Ok(());
        }
        let session = SessionIdentity { id: session_id + 1, incarnation: incarnation + 1 };
        let options = PrefillOptions {
            block_tokens: KV_BLOCK_TOKENS as u32,
            select_tokens: (self.budget_blocks * KV_BLOCK_TOKENS) as u64,
            reserve_tokens: KV_BLOCK_TOKENS as u64,
            interval: 1024,
            stride: 32,
            half_life: 8192.0,
        };
        let pressure = PrefillPressure::new(session, fa_layers, &options)
            .map_err(|e| format!("kvmem: layerwise pressure create: {e}"))?;
        st.lw_pressure = Some(pressure);
        Ok(())
    }

    /// Immutable borrow of slot `slot`'s pressure owner (`None` under Legacy,
    /// before the first admitted query, or out of range). The owner stays owned
    /// by this controller; the driver borrows it for `select`/`snapshot`.
    pub fn layerwise_pressure(&self, slot: usize) -> Option<&PrefillPressure> {
        self.slots.get(slot).and_then(|st| st.lw_pressure.as_ref())
    }

    /// Mutable borrow of slot `slot`'s pressure owner for `accept`/`restore`
    /// (`None` under Legacy, before construction, or out of range). The caller
    /// must have called [`Self::ensure_layerwise_pressure`] first this session.
    pub fn layerwise_pressure_mut(&mut self, slot: usize) -> Option<&mut PrefillPressure> {
        self.slots.get_mut(slot).and_then(|st| st.lw_pressure.as_mut())
    }

    /// Slot `slot`'s persistent PER-LAYER resident keep-set (decision A, design
    /// §3.10a): `keep[ordinal]` holds the canonical block ids selected for FA
    /// ordinal `ordinal` by the previous step (pressure.select on a Pressure
    /// step, the ② scorer on a Query step). Drives each layer's canonical
    /// `resident` list. Empty before the first selection / under Legacy; `None`
    /// when out of range. Borrowed for the bootstrap synthesis (an absent ordinal
    /// entry reads as `[]`).
    pub fn layerwise_keep_set(&self, slot: usize) -> Option<&[Vec<u64>]> {
        self.slots.get(slot).map(|st| st.lw_keep.as_slice())
    }

    /// Replace slot `slot`'s per-layer keep-set with the latest selection result
    /// (decision A). `keep[ordinal]` is the canonical sealed-page ids restored
    /// for FA ordinal `ordinal` this step; the bootstrap filters any id beyond
    /// the current sealed history defensively. No-op when out of range. Legacy
    /// never calls it (the keep-set stays empty).
    pub fn set_layerwise_keep_set(&mut self, slot: usize, keep: Vec<Vec<u64>>) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_keep = keep;
        }
    }

    /// Move slot `slot`'s pressure owner OUT of the controller (design §3.10a
    /// borrow resolution). The scoped serve orchestration takes the owner before
    /// building the carrier (which borrows the slot's `&RawStore`/`&KvStore`), so
    /// `select`/`accept` can use `&mut PrefillPressure` independently of those
    /// borrows; [`Self::restore_layerwise_pressure`] puts it back after the
    /// forward. `None` under Legacy, before construction, or out of range. The
    /// slot keeps its keep-set / counters; only the pressure owner is detached.
    pub fn take_layerwise_pressure(&mut self, slot: usize) -> Option<PrefillPressure> {
        self.slots.get_mut(slot).and_then(|st| st.lw_pressure.take())
    }

    /// Return a previously [`Self::take_layerwise_pressure`]-detached owner to
    /// slot `slot` (design §3.10a). No-op when out of range. The orchestration
    /// calls this exactly once after the forward, before restoring the `KvMemCtl`
    /// into the rig, so the session-lifetime owner persists to the next step.
    pub fn restore_layerwise_pressure(&mut self, slot: usize, pressure: PrefillPressure) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.lw_pressure = Some(pressure);
        }
    }

    /// Feed the tapped K rows [row_start, row_start + n_rows) of `slot`'s
    /// step batch to the per-layer key statistics. `pos0` is the absolute
    /// position of the first row and must be the store's current length.
    pub fn feed_rows(
        &mut self,
        gpu: &Gpu,
        taps: &KvMemTaps,
        slot: usize,
        pos0: usize,
        n_rows: usize,
        row_start: usize,
        pool: &SlotPool,
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
    ) -> Result<(), String> {
        if n_rows == 0 {
            return Ok(());
        }
        if slot >= self.slots.len() {
            return Err(format!("kvmem: feed_rows slot {slot} out of range"));
        }
        let n_fa = self.n_fa_layers;
        let trace = self.trace;
        let st = &mut self.slots[slot];
        if st.fed != pos0 as u32 {
            return Err(format!(
                "kvmem slot {slot}: statistics feed at position {pos0} but the \
                 store holds {} tokens (feed must be contiguous)",
                st.fed
            ));
        }
        let timed = time_flag();
        let t_all = std::time::Instant::now();
        let mut t_read = std::time::Duration::ZERO;
        let mut t_write = std::time::Duration::ZERO;
        for il in 0..n_fa {
            let t = std::time::Instant::now();
            taps.read_k_rows_at(gpu, il, row_start, n_rows, &mut self.staging)
                .map_err(|e| format!("kvmem tap read (layer {il}): {e:?}"))?;
            if timed {
                t_read += t.elapsed();
            }
            let t = std::time::Instant::now();
            st.raw
                .write_layer_mean_k(pos0 as u32, n_rows as u32, il as u32, &self.staging)
                .map_err(|e| format!("kvmem write stats (layer {il}): {e}"))?;
            if timed {
                t_write += t.elapsed();
            }
        }
        st.kv
            .register_append(n_rows as u32)
            .map_err(|e| format!("kvmem register append: {e}"))?;
        st.fed = (pos0 + n_rows) as u32;
        // Query-span ring: append this step's tapped last Q row per layer so the
        // next selection round can score blocks against a span of recent query
        // rows. No-op at query_span == 1 (the mean-K path reads the tap directly
        // at selection time), keeping the default byte-neutral.
        if self.query_span > 1 {
            let span = self.query_span;
            let qw = self.q_width;
            let cursor = st.q_ring_cursor;
            for il in 0..n_fa {
                taps.read_q_slot(gpu, il, slot, &mut self.q_row)
                    .map_err(|e| format!("kvmem q tap read (layer {il}): {e:?}"))?;
                let dst = il * span * qw + cursor * qw;
                st.q_ring[dst..dst + qw].copy_from_slice(&self.q_row);
            }
            st.q_ring_cursor = (cursor + 1) % span;
            st.q_ring_len = (st.q_ring_len + 1).min(span);
        }
        // Layerwise write-through (design §3.7): seal this feed's packed Q8 K/V
        // into the host archive so every fed token's payload survives for
        // drop-evict + COW restore (D4). Legacy skips this entirely — the arena
        // params are unused on the default path, keeping it byte-neutral.
        if self.strategy.is_layerwise() {
            self.write_through_archive(gpu, pool, k_arenas, v_arenas, slot, pos0, n_rows)?;
        }
        if trace {
            eprintln!(
                "[kvmem] slot {slot} feed rows {pos0}..{} ({n_rows})",
                pos0 + n_rows
            );
        }
        if timed {
            eprintln!(
                "[kvmem-time] slot {slot} feed {n_rows} rows @{pos0}: \
                 total {:.1} ms (tap read {:.1} ms, host stats write {:.1} ms)",
                t_all.elapsed().as_secs_f64() * 1e3,
                t_read.as_secs_f64() * 1e3,
                t_write.as_secs_f64() * 1e3,
            );
        }
        Ok(())
    }

    /// Layerwise write-through (design §3.7). Gather this feed's packed Q8 K/V
    /// rows out of the GPU arenas into the contiguous staging buffers and seal
    /// them into the slot's host archive (`write_layer_k_gpu`/`v_gpu`), so EVERY
    /// fed token's payload survives — not just the spilled blocks the legacy
    /// write-back path stores. The sealed archive is the source of truth for
    /// drop-evict + COW restore (D4).
    ///
    /// The fed range is contiguous at the frontier, so its logical pages are
    /// resident (never spilled): `physical()` returning `None` is a hard error,
    /// not a silent skip. The core `write_layer_k_gpu` is token-granular (it
    /// splits the range into blocks itself), so one contiguous gather + one
    /// seal call per layer suffices even when the feed spans several pages or
    /// ends on a partial tail. Only called when `strategy == Layerwise`.
    ///
    /// D2 gathers synchronously (matching the legacy `spill_page` D2H path) for
    /// correctness; async pipelining over the Phase 1 pinned slab to overlap
    /// compute is a later refinement within sub-patch ②.
    fn write_through_archive(
        &mut self,
        gpu: &Gpu,
        pool: &SlotPool,
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        slot: usize,
        pos0: usize,
        n_rows: usize,
    ) -> Result<(), String> {
        debug_assert_eq!(k_arenas.len(), self.n_fa_layers);
        debug_assert_eq!(v_arenas.len(), self.n_fa_layers);
        let bt = pool
            .block_table(SlotId(slot))
            .ok_or_else(|| format!("kvmem write-through: slot {slot} has no block table"))?;
        let kpb = self.k_page_bytes;
        let vpb = self.v_page_bytes;
        let krow = kpb / KV_BLOCK_TOKENS;
        let vrow = vpb / KV_BLOCK_TOKENS;
        if n_rows * krow > self.wt_k.len() || n_rows * vrow > self.wt_v.len() {
            return Err(format!(
                "kvmem write-through: feed of {n_rows} rows exceeds the staging capacity \
                 ({} K / {} V bytes)",
                self.wt_k.len(),
                self.wt_v.len()
            ));
        }
        let first_lp = pos0 / KV_BLOCK_TOKENS;
        let last_lp = (pos0 + n_rows - 1) / KV_BLOCK_TOKENS;
        // Resolve every fed page's physical index up front so a non-resident
        // page fails closed before any D2H or archive mutation.
        let mut phys_pages = Vec::with_capacity(last_lp - first_lp + 1);
        for lp in first_lp..=last_lp {
            let phys = bt.physical(lp).ok_or_else(|| {
                format!(
                    "kvmem write-through: slot {slot} logical page {lp} is not resident \
                     (spilled or out of range) at feed position {pos0}"
                )
            })?;
            phys_pages.push(phys);
        }
        for il in 0..self.n_fa_layers {
            // Gather the fed K rows page-by-page into the contiguous buffer.
            for (i, lp) in (first_lp..=last_lp).enumerate() {
                let phys = phys_pages[i] as usize;
                let row_lo = pos0.max(lp * KV_BLOCK_TOKENS);
                let row_hi = (pos0 + n_rows).min((lp + 1) * KV_BLOCK_TOKENS);
                let cnt = row_hi - row_lo;
                let intra = row_lo - lp * KV_BLOCK_TOKENS;
                let dst = (row_lo - pos0) * krow;
                let src = phys * kpb + intra * krow;
                gpu.hip
                    .memcpy_dtoh_at(&mut self.wt_k[dst..dst + cnt * krow], &k_arenas[il].buf, src)
                    .map_err(|e| format!("kvmem write-through K gather (layer {il}): {e:?}"))?;
            }
            self.slots[slot]
                .raw
                .write_layer_k_gpu(pos0 as u32, n_rows as u32, il as u32, &self.wt_k[..n_rows * krow])
                .map_err(|e| format!("kvmem write-through K seal (layer {il}): {e}"))?;
            // Gather the fed V rows.
            for (i, lp) in (first_lp..=last_lp).enumerate() {
                let phys = phys_pages[i] as usize;
                let row_lo = pos0.max(lp * KV_BLOCK_TOKENS);
                let row_hi = (pos0 + n_rows).min((lp + 1) * KV_BLOCK_TOKENS);
                let cnt = row_hi - row_lo;
                let intra = row_lo - lp * KV_BLOCK_TOKENS;
                let dst = (row_lo - pos0) * vrow;
                let src = phys * vpb + intra * vrow;
                gpu.hip
                    .memcpy_dtoh_at(&mut self.wt_v[dst..dst + cnt * vrow], &v_arenas[il].buf, src)
                    .map_err(|e| format!("kvmem write-through V gather (layer {il}): {e:?}"))?;
            }
            self.slots[slot]
                .raw
                .write_layer_v_gpu(pos0 as u32, n_rows as u32, il as u32, &self.wt_v[..n_rows * vrow])
                .map_err(|e| format!("kvmem write-through V seal (layer {il}): {e}"))?;
        }
        Ok(())
    }

    /// Refresh the retrieval query for `slot` and score its resident+spilled
    /// blocks. At `query_span == 1` each layer contributes its tapped last Q row
    /// (post-RMSNorm, pre-RoPE, one token) — the frame
    /// `mean_key_retrieval_scores` expects. At `query_span > 1` the per-slot host
    /// ring supplies a span of recent Q rows scored by `page_softmax_scores`.
    fn refresh_scores(
        &mut self,
        gpu: &Gpu,
        taps: &KvMemTaps,
        slot: usize,
        pages: usize,
    ) -> Result<Vec<f64>, String> {
        // Query span > 1: score blocks by page-softmax over the ring's recent Q
        // rows instead of the single-row mean-K cosine. The span == 1 path below
        // is untouched, so the default stays byte-identical.
        if self.query_span > 1 {
            let span = self.query_span;
            let qw = self.q_width;
            let n_fa = self.n_fa_layers;
            let st = &self.slots[slot];
            let len = st.q_ring_len;
            if len == 0 {
                return Err(format!(
                    "kvmem slot {slot}: query span empty at selection (no fed Q rows)"
                ));
            }
            let cursor = st.q_ring_cursor;
            // Extract the ring's valid rows in chronological order, layer-major
            // ([layer × row × q_width]) — the PageQueryView data layout.
            for il in 0..n_fa {
                for t in 0..len {
                    let src = (cursor + span - len + t) % span;
                    let s0 = il * span * qw + src * qw;
                    let d0 = il * len * qw + t * qw;
                    self.q_span[d0..d0 + qw].copy_from_slice(&st.q_ring[s0..s0 + qw]);
                }
                self.q_layer_rows[il] = len as u32;
            }
            let accepted = st.fed as u64;
            let sink = self.kv_cfg.sink_blocks;
            let recent = self.kv_cfg.recent_blocks;
            let budget = self.budget_blocks as u32;
            let scores = page_softmax_scores(
                &st.raw,
                pages as u32,
                accepted,
                KV_BLOCK_TOKENS as u32,
                budget,
                sink,
                recent,
                &self.q_span[..n_fa * len * qw],
                &self.q_layer_rows,
                n_fa as u32,
                len as u32,
                self.n_heads as u32,
                self.n_kv_heads as u32,
                self.head_dim as u32,
                0.0,
                GqaAggregation::Mean,
            )
            .map_err(|e| format!("kvmem page-softmax scores: {e}"))?;
            if self.trace {
                let sum: f64 = scores.iter().sum();
                let top = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                eprintln!(
                    "[kvmem] slot {slot} query-span: rows={len} pages={pages} \
                     score_sum={sum:.4} top={top:.4}"
                );
            }
            return Ok(scores);
        }
        for il in 0..self.n_fa_layers {
            taps.read_q_slot(gpu, il, slot, &mut self.q_row)
                .map_err(|e| format!("kvmem q tap read (layer {il}): {e:?}"))?;
            self.sums[il * self.q_width..(il + 1) * self.q_width]
                .copy_from_slice(&self.q_row);
        }
        mean_key_retrieval_scores(
            &self.slots[slot].raw,
            pages as u32,
            &self.sums,
            &self.counts,
            self.n_fa_layers as u32,
            self.head_dim as u32,
            self.n_heads as u32,
            self.n_kv_heads as u32,
            GqaAggregation::Mean,
        )
        .map_err(|e| format!("kvmem retrieval scores: {e}"))
    }

    /// Keep each over-budget slot's working set resident: spill unselected
    /// full pages, restore selected spilled pages. Must run before the step's
    /// page provisioning so a restore sees every spill's freed page.
    ///
    /// `prefill_final[s]` marks the prefill chunk that completes the prompt.
    /// That chunk uses the *retrieval* selection (scored against the Q row of
    /// the chunk just before it) instead of the sink+newest pressure policy,
    /// so a page holding an answer buried in the prompt is back in VRAM for
    /// the forward whose logits produce the first sampled token.
    ///
    /// `resident[s]` marks slots whose host state belongs to a session kept
    /// resident for continuation reuse even though no request is in flight.
    /// Their state (statistics, spilled payloads, selection watermark) is the
    /// reused prefix and must survive idle steps; only a slot with neither an
    /// in-flight request nor a resident session is dropped.
    pub fn prepare_step(
        &mut self,
        gpu: &Gpu,
        pool: &mut SlotPool,
        taps: &KvMemTaps,
        active: &[bool],
        resident: &[bool],
        prefill: &[bool],
        prefill_final: &[bool],
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
    ) -> Result<(), String> {
        debug_assert!(pool.is_paged(), "kvmem requires the paged slot pool");
        debug_assert_eq!(k_arenas.len(), self.n_fa_layers);
        debug_assert_eq!(v_arenas.len(), self.n_fa_layers);
        let n_slots = self.slots.len();
        let timed = time_flag();
        let t_all = std::time::Instant::now();
        let mut t_select = std::time::Duration::ZERO;
        let mut t_spill = std::time::Duration::ZERO;
        let mut t_restore = std::time::Duration::ZERO;
        let mut n_spilled = 0usize;
        let mut n_restored = 0usize;

        // Return host state for slots whose session closed since last step.
        // An idle slot that still holds a resident session keeps its state:
        // it is the next turn's continuation prefix.
        for s in 0..n_slots {
            let live = active.get(s).copied().unwrap_or(false)
                || resident.get(s).copied().unwrap_or(false);
            if !live && (self.slots[s].fed != 0 || self.slots[s].last_select_full != 0) {
                self.reset_slot(s)?;
            }
        }

        // ── Pass 1: select, then spill every unselected resident full page ──
        let mut plans: Vec<Option<StepPlan>> = (0..n_slots).map(|_| None).collect();
        for s in 0..n_slots {
            if !active.get(s).copied().unwrap_or(false) {
                continue;
            }
            let prefill_step = prefill.get(s).copied().unwrap_or(false);
            let (pages, live) = match pool.block_table(SlotId(s)) {
                Some(bt) => (bt.num_pages(), bt.live_tokens()),
                None => continue,
            };
            // Watermark hysteresis: when GPU_HIGH/LOW_WATERMARK is set, spill is
            // triggered at the high-watermark page count (>= the keep target the
            // core selection budget was narrowed to), instead of the single
            // budget threshold. Unset => the historical budget_blocks trigger.
            let spill_trigger = self.watermark.map(|(t, _)| t).unwrap_or(self.budget_blocks);
            if pages <= spill_trigger {
                continue;
            }
            let full_pages = live / KV_BLOCK_TOKENS;
            // The chunk that completes the prompt gets the retrieval policy
            // too — the tokens before it are the question, and this chunk's
            // forward produces the logits the first answer token comes from.
            // A slot with no fed rows has no tapped query yet (single-chunk
            // prompt); it keeps the pressure policy.
            let final_prefill = prefill_step
                && prefill_final.get(s).copied().unwrap_or(false)
                && self.slots[s].fed > self.slots[s].turn_start;
            // The first decode step of a request takes a selection decision
            // even when the block interval has not elapsed: the prompt-
            // completing chunk just captured this turn's question in the Q
            // tap, and scoring the resident set against it is what restores
            // the page holding the answer. `fed > turn_start` rejects a
            // stale tap from the previous request (empty continuation
            // suffix: no rows of this turn exist to score with).
            let first_decode = !prefill_step
                && !self.slots[s].first_decode_done
                && self.slots[s].fed > self.slots[s].turn_start;
            if !final_prefill
                && !first_decode
                && full_pages
                    < self.slots[s]
                        .last_select_full
                        .saturating_add(self.interval)
            {
                continue;
            }
            let t_pick = std::time::Instant::now();
            let picked = if prefill_step && !final_prefill {
                self.slots[s]
                    .kv
                    .pick_prefill_pressure_blocks(&[])
                    .map_err(|e| format!("kvmem prefill selection: {e}"))?
            } else {
                // SELECT_METHOD gates which signal feeds the shared pick_topk
                // ranking (the core's executor-level policy). Retrieval (0, the
                // default) scores the resident set against the tapped mean-K
                // query. Recency (2) feeds no learned signal: zeroing the scores
                // makes pick_topk keep sink + recent and fill the rest by recency
                // (the core's deterministic id tie-break), and skips the Q-tap
                // read + mean-K scoring entirely. H2O (1) is rejected at
                // construction until the attention-heat tap lands.
                //
                // LRFU round lifecycle: begin before previewing scores, commit
                // after the pick so the selector ranks the decayed candidate and
                // the round's reinforcement is then sealed. Both are no-ops when
                // lrfu is off, so the default Retrieval path stays byte-neutral.
                self.slots[s]
                    .kv
                    .begin_retrieval_round()
                    .map_err(|e| format!("kvmem begin round: {e}"))?;
                if self.select_method == 2 {
                    let zeros = vec![0.0f64; pages];
                    self.slots[s]
                        .kv
                        .set_retrieval_scores(&zeros)
                        .map_err(|e| format!("kvmem set scores: {e}"))?;
                } else {
                    let scores = self.refresh_scores(gpu, taps, s, pages)?;
                    self.slots[s]
                        .kv
                        .set_retrieval_scores(&scores)
                        .map_err(|e| format!("kvmem set scores: {e}"))?;
                }
                let picked = self.slots[s]
                    .kv
                    .pick_topk_blocks(&[])
                    .map_err(|e| format!("kvmem decode selection: {e}"))?;
                self.slots[s]
                    .kv
                    .commit_retrieval_round()
                    .map_err(|e| format!("kvmem commit round: {e}"))?;
                picked
            };
            if timed {
                t_select += t_pick.elapsed();
            }
            let mut keep = vec![false; pages];
            for &id in &picked {
                if let Some(k) = keep.get_mut(id as usize) {
                    *k = true;
                }
            }
            // Collect this slot's page actions before mutating the pool.
            let (spill, restore) = {
                let bt = pool
                    .block_table(SlotId(s))
                    .expect("kvmem: block table vanished inside one step");
                debug_assert_eq!(bt.num_pages(), pages);
                let mut spill = Vec::new();
                let mut restore = Vec::new();
                for lp in 0..full_pages {
                    if keep[lp] {
                        if bt.is_spilled(lp) {
                            restore.push(lp);
                        }
                    } else if !bt.is_spilled(lp) {
                        match bt.physical(lp) {
                            Some(phys) => spill.push((lp, phys)),
                            None => {
                                return Err(format!(
                                    "kvmem slot {s}: logical page {lp} is neither \
                                     resident nor spilled"
                                ))
                            }
                        }
                    }
                }
                (spill, restore)
            };
            self.flush_spill_batch(
                gpu,
                pool,
                s,
                &spill,
                k_arenas,
                v_arenas,
                timed,
                &mut t_spill,
                &mut n_spilled,
            )?;
            if self.trace {
                eprintln!(
                    "[kvmem] slot {s} ({}): {pages} pages, full {full_pages}, \
                     budget {}, kept {}, spilled {}, restore {}",
                    if prefill_step {
                        if final_prefill {
                            "prefill-final"
                        } else {
                            "prefill"
                        }
                    } else {
                        "decode"
                    },
                    self.budget_blocks,
                    keep.iter().filter(|k| **k).count(),
                    spill.len(),
                    restore.len(),
                );
                if env_flag("HIPFIRE_KVMEM_TRACE_PAGES") {
                    eprintln!(
                        "[kvmem] slot {s} picked {picked:?} restore {restore:?}"
                    );
                }
            }
            self.slots[s].last_select_full = full_pages;
            if !prefill_step {
                self.slots[s].first_decode_done = true;
            }
            plans[s] = Some(StepPlan { keep, restore });
        }

        // ── Pass 2: restore the selected spilled pages ──────────────────────
        // Alloc + bind every selected page first (same order as before, so the
        // free-list interaction is unchanged), then move all the bytes in one
        // batched flush. No forward runs between bind and restore.
        let mut restore_batch: Vec<(usize, usize, u32)> = Vec::new();
        for s in 0..n_slots {
            let Some(plan) = plans[s].take() else {
                continue;
            };
            for lp in plan.restore {
                debug_assert!(plan.keep[lp]);
                let phys = pool
                    .page_pool_mut()
                    .ok_or_else(|| "kvmem restore: pool is not paged".to_string())?
                    .alloc_page_unbound()
                    .map_err(|e| format!("kvmem restore alloc: {e}"))?;
                pool.bind_restored_page(SlotId(s), lp, phys)
                    .map_err(|e| format!("kvmem restore bind: {e}"))?;
                restore_batch.push((s, lp, phys));
            }
            if verify_flag() {
                if let Some(bt) = pool.block_table(SlotId(s)) {
                    let mut seen: std::collections::HashMap<u32, usize> =
                        std::collections::HashMap::new();
                    let mut dups = Vec::new();
                    for lp in 0..bt.num_pages() {
                        if let Some(phys) = bt.physical(lp) {
                            if let Some(prev) = seen.insert(phys, lp) {
                                dups.push((prev, lp, phys));
                            }
                        }
                    }
                    let mut map = Vec::with_capacity(bt.num_pages());
                    for lp in 0..bt.num_pages() {
                        map.push(bt.raw_entry(lp).unwrap_or(u32::MAX));
                    }
                    eprintln!(
                        "[kvmem-verify] slot {s} resident {} pages, dups {dups:?} map {map:?}",
                        seen.len()
                    );
                }
            }
        }
        self.flush_restore_batch(
            gpu,
            &restore_batch,
            k_arenas,
            v_arenas,
            timed,
            &mut t_restore,
            &mut n_restored,
        )?;
        if timed && (n_spilled > 0 || n_restored > 0 || t_select > std::time::Duration::ZERO) {
            eprintln!(
                "[kvmem-time] prepare: total {:.1} ms (select {:.1} ms, spill {} pages \
                 {:.1} ms, restore {} pages {:.1} ms)",
                t_all.elapsed().as_secs_f64() * 1e3,
                t_select.as_secs_f64() * 1e3,
                n_spilled,
                t_spill.as_secs_f64() * 1e3,
                n_restored,
                t_restore.as_secs_f64() * 1e3,
            );
        }
        Ok(())
    }

    /// Grow the reusable pinned slab to at least `need` bytes. The previous
    /// slab (if any) was drained by the prior chunk's stream wait, so dropping
    /// it here frees no in-flight copy. The first allocation takes the whole
    /// chunk cap (`XFER_SLAB_CAP`, or `need` when a single page exceeds it) so a
    /// later, larger chunk never re-page-locks pinned memory mid-prefill —
    /// `hipHostMalloc` is costly and would otherwise land in the TTFT path every
    /// time the batch grows. Every chunk is `<= cap` by construction, so this
    /// allocates at most once per controller.
    fn ensure_xfer_slab(&mut self, gpu: &Gpu, need: usize) -> Result<(), String> {
        if self.xfer_slab.as_ref().is_none_or(|b| b.size() < need) {
            let alloc = need.max(XFER_SLAB_CAP);
            self.xfer_slab = None;
            let buf = gpu
                .hip
                .transfer_buffer(alloc)
                .map_err(|e| format!("kvmem transfer slab alloc ({alloc} B): {e:?}"))?;
            if time_flag() {
                eprintln!("[kvmem-time] xfer slab alloc {alloc} B (need {need})");
            }
            self.xfer_slab = Some(buf);
        }
        Ok(())
    }

    /// Bytes one full page occupies in the slab across every arena (K+V).
    fn page_xfer_bytes(&self) -> usize {
        self.n_fa_layers * (self.k_page_bytes + self.v_page_bytes)
    }

    /// Spill this slot's selected pages (`(lp, phys)` list) to the host store.
    /// Batched path: gather every page's K/V out of the arenas into the pinned
    /// slab with stream-ordered async D2H, ONE wait per chunk, then write the
    /// host payloads and mark each page spilled. All reads drain before any
    /// `stage_out_page` frees a physical page, and the whole batch completes
    /// inside Pass 1 before Pass 2 can reuse a freed page. The copies ride the
    /// steady-copy stream when one is installed, else the legacy null stream —
    /// the same stream prefill compute uses, so ordering holds either way. Falls
    /// back to the per-page blocking path only when batching is off or a graph
    /// is being captured.
    #[allow(clippy::too_many_arguments)]
    fn flush_spill_batch(
        &mut self,
        gpu: &Gpu,
        pool: &mut SlotPool,
        s: usize,
        spills: &[(usize, u32)],
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        timed: bool,
        t_spill: &mut std::time::Duration,
        n_spilled: &mut usize,
    ) -> Result<(), String> {
        if spills.is_empty() {
            return Ok(());
        }
        let stream = gpu.steady_copy_stream();
        let batched = self.batch_xfer && !gpu.graphs.capture_mode;
        if timed {
            eprintln!(
                "[kvmem-time] spill xfer={} pages={} stream={}",
                if batched { "batch" } else { "page" },
                spills.len(),
                stream.is_some()
            );
        }
        if !batched {
            for &(lp, phys) in spills {
                let t = std::time::Instant::now();
                spill_page(
                    gpu,
                    &self.slots[s].raw,
                    &mut self.page_k,
                    &mut self.page_v,
                    k_arenas,
                    v_arenas,
                    self.k_page_bytes,
                    self.v_page_bytes,
                    phys,
                    lp,
                )?;
                pool.stage_out_page(SlotId(s), lp)
                    .map_err(|e| format!("kvmem stage-out: {e}"))?;
                if timed {
                    *t_spill += t.elapsed();
                    *n_spilled += 1;
                }
            }
            return Ok(());
        }
        let per_page = self.page_xfer_bytes();
        let chunk_pages = (XFER_SLAB_CAP / per_page).max(1);
        let kpb = self.k_page_bytes;
        let vpb = self.v_page_bytes;
        let n_layers = self.n_fa_layers;
        for chunk in spills.chunks(chunk_pages) {
            let t = std::time::Instant::now();
            self.ensure_xfer_slab(gpu, chunk.len() * per_page)?;
            let slab = self
                .xfer_slab
                .as_mut()
                .expect("transfer slab")
                .as_bytes_mut();
            // Phase A: async D2H gather every page/layer into the slab.
            let mut off = 0usize;
            for &(_lp, phys) in chunk {
                for il in 0..n_layers {
                    gpu.hip
                        .memcpy_dtoh_async_at(
                            &mut slab[off..off + kpb],
                            &k_arenas[il].buf,
                            phys as usize * kpb,
                            stream,
                        )
                        .map_err(|e| format!("kvmem spill K read (layer {il}): {e:?}"))?;
                    off += kpb;
                    gpu.hip
                        .memcpy_dtoh_async_at(
                            &mut slab[off..off + vpb],
                            &v_arenas[il].buf,
                            phys as usize * vpb,
                            stream,
                        )
                        .map_err(|e| format!("kvmem spill V read (layer {il}): {e:?}"))?;
                    off += vpb;
                }
            }
            // ONE wait drains the whole chunk before the host reads the slab.
            match stream {
                Some(s) => gpu.hip.stream_synchronize(s),
                None => gpu.hip.stream_synchronize_default(),
            }
            .map_err(|e| format!("kvmem spill sync: {e:?}"))?;
            // Phase B: write host payloads, then mark each page spilled.
            let mut off = 0usize;
            for &(lp, phys) in chunk {
                let pos0 = (lp * KV_BLOCK_TOKENS) as u32;
                for il in 0..n_layers {
                    let ksrc = &slab[off..off + kpb];
                    self.slots[s]
                        .raw
                        .write_layer_k_gpu(pos0, KV_BLOCK_TOKENS as u32, il as u32, ksrc)
                        .map_err(|e| format!("kvmem spill K store (layer {il}): {e}"))?;
                    off += kpb;
                    let vsrc = &slab[off..off + vpb];
                    self.slots[s]
                        .raw
                        .write_layer_v_gpu(pos0, KV_BLOCK_TOKENS as u32, il as u32, vsrc)
                        .map_err(|e| format!("kvmem spill V store (layer {il}): {e}"))?;
                    off += vpb;
                    if il == 0 && verify_flag() {
                        eprintln!(
                            "[kvmem-verify] spill lp={lp} phys={phys} \
                             k={:016x} v={:016x}",
                            fnv64(ksrc),
                            fnv64(vsrc)
                        );
                    }
                }
                pool.stage_out_page(SlotId(s), lp)
                    .map_err(|e| format!("kvmem stage-out: {e}"))?;
            }
            if timed {
                *t_spill += t.elapsed();
                *n_spilled += chunk.len();
            }
        }
        Ok(())
    }

    /// Restore the selected spilled pages (`(slot, lp, phys)` list, already
    /// alloc'd and bound by the caller) back into the arenas. Batched path:
    /// gather host payloads into the pinned slab, scatter them with
    /// stream-ordered async H2D, ONE wait per chunk so every page is resident
    /// before `forward_step`. The copies ride the steady-copy stream when one is
    /// installed, else the legacy null stream — the same stream prefill compute
    /// uses, so the restored bytes are visible to the next attention either way.
    /// Falls back to the per-page blocking path only when batching is off or a
    /// graph is captured.
    #[allow(clippy::too_many_arguments)]
    fn flush_restore_batch(
        &mut self,
        gpu: &Gpu,
        batch: &[(usize, usize, u32)],
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        timed: bool,
        t_restore: &mut std::time::Duration,
        n_restored: &mut usize,
    ) -> Result<(), String> {
        if batch.is_empty() {
            return Ok(());
        }
        let stream = gpu.steady_copy_stream();
        let batched = self.batch_xfer && !gpu.graphs.capture_mode;
        if timed {
            eprintln!(
                "[kvmem-time] restore xfer={} pages={} stream={}",
                if batched { "batch" } else { "page" },
                batch.len(),
                stream.is_some()
            );
        }
        if !batched {
            for &(s, lp, phys) in batch {
                let t = std::time::Instant::now();
                restore_page(
                    gpu,
                    &self.slots[s].raw,
                    &mut self.page_k,
                    &mut self.page_v,
                    k_arenas,
                    v_arenas,
                    self.k_page_bytes,
                    self.v_page_bytes,
                    phys,
                    lp,
                )?;
                if timed {
                    *t_restore += t.elapsed();
                    *n_restored += 1;
                }
            }
            return Ok(());
        }
        let per_page = self.page_xfer_bytes();
        let chunk_pages = (XFER_SLAB_CAP / per_page).max(1);
        let kpb = self.k_page_bytes;
        let vpb = self.v_page_bytes;
        let n_layers = self.n_fa_layers;
        let verify = verify_flag();
        for chunk in batch.chunks(chunk_pages) {
            let t = std::time::Instant::now();
            self.ensure_xfer_slab(gpu, chunk.len() * per_page)?;
            let slab = self
                .xfer_slab
                .as_mut()
                .expect("transfer slab")
                .as_bytes_mut();
            // Phase A: gather host payloads into the slab.
            let mut off = 0usize;
            for &(s, lp, _phys) in chunk {
                let block = lp as u32;
                for il in 0..n_layers {
                    if !self.slots[s].raw.copy_k_gpu(
                        block,
                        il as u32,
                        &mut slab[off..off + kpb],
                        KV_BLOCK_TOKENS as u32,
                    ) {
                        return Err(format!(
                            "kvmem restore: block {block} layer {il} has no K payload"
                        ));
                    }
                    off += kpb;
                    if !self.slots[s].raw.copy_v_gpu(
                        block,
                        il as u32,
                        &mut slab[off..off + vpb],
                        KV_BLOCK_TOKENS as u32,
                    ) {
                        return Err(format!(
                            "kvmem restore: block {block} layer {il} has no V payload"
                        ));
                    }
                    off += vpb;
                }
            }
            // Phase B: async H2D scatter the slab into every arena page.
            let mut off = 0usize;
            for &(_s, _lp, phys) in chunk {
                for il in 0..n_layers {
                    gpu.hip
                        .memcpy_htod_async_at_on(
                            &k_arenas[il].buf,
                            phys as usize * kpb,
                            &slab[off..off + kpb],
                            stream,
                        )
                        .map_err(|e| format!("kvmem restore K write (layer {il}): {e:?}"))?;
                    off += kpb;
                    gpu.hip
                        .memcpy_htod_async_at_on(
                            &v_arenas[il].buf,
                            phys as usize * vpb,
                            &slab[off..off + vpb],
                            stream,
                        )
                        .map_err(|e| format!("kvmem restore V write (layer {il}): {e:?}"))?;
                    off += vpb;
                }
            }
            // ONE wait: every restored page is resident before forward_step.
            match stream {
                Some(s) => gpu.hip.stream_synchronize(s),
                None => gpu.hip.stream_synchronize_default(),
            }
            .map_err(|e| format!("kvmem restore sync: {e:?}"))?;
            // VERIFY: read back layer 0 of each page and compare to the slab.
            if verify {
                let mut off = 0usize;
                for &(_s, lp, phys) in chunk {
                    let ksrc = &slab[off..off + kpb];
                    let vsrc = &slab[off + kpb..off + kpb + vpb];
                    let mut rb_k = vec![0u8; kpb];
                    let mut rb_v = vec![0u8; vpb];
                    gpu.hip
                        .memcpy_dtoh_at(&mut rb_k, &k_arenas[0].buf, phys as usize * kpb)
                        .map_err(|e| format!("kvmem verify K readback (layer 0): {e:?}"))?;
                    gpu.hip
                        .memcpy_dtoh_at(&mut rb_v, &v_arenas[0].buf, phys as usize * vpb)
                        .map_err(|e| format!("kvmem verify V readback (layer 0): {e:?}"))?;
                    let mk = ksrc.iter().zip(&rb_k).filter(|(a, b)| a != b).count();
                    let mv = vsrc.iter().zip(&rb_v).filter(|(a, b)| a != b).count();
                    eprintln!(
                        "[kvmem-verify] restore lp={lp} phys={phys} \
                         k={:016x} v={:016x} readback_mismatch k={mk} v={mv}",
                        fnv64(ksrc),
                        fnv64(vsrc)
                    );
                    off += per_page;
                }
            }
            if timed {
                *t_restore += t.elapsed();
                *n_restored += chunk.len();
            }
        }
        Ok(())
    }
}

/// Copy a full page's K/V bytes out of the arenas into the host store, then
/// (the caller) marks the logical page spilled.
#[allow(clippy::too_many_arguments)]
fn spill_page(
    gpu: &Gpu,
    raw: &RawStore,
    page_k: &mut [u8],
    page_v: &mut [u8],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    k_page_bytes: usize,
    v_page_bytes: usize,
    phys: u32,
    lp: usize,
) -> Result<(), String> {
    debug_assert_eq!(page_k.len(), k_page_bytes);
    debug_assert_eq!(page_v.len(), v_page_bytes);
    let pos0 = (lp * KV_BLOCK_TOKENS) as u32;
    for il in 0..k_arenas.len() {
        gpu.hip
            .memcpy_dtoh_at(page_k, &k_arenas[il].buf, phys as usize * k_page_bytes)
            .map_err(|e| format!("kvmem spill K read (layer {il}): {e:?}"))?;
        raw.write_layer_k_gpu(pos0, KV_BLOCK_TOKENS as u32, il as u32, page_k)
            .map_err(|e| format!("kvmem spill K store (layer {il}): {e}"))?;
        gpu.hip
            .memcpy_dtoh_at(page_v, &v_arenas[il].buf, phys as usize * v_page_bytes)
            .map_err(|e| format!("kvmem spill V read (layer {il}): {e:?}"))?;
        raw.write_layer_v_gpu(pos0, KV_BLOCK_TOKENS as u32, il as u32, page_v)
            .map_err(|e| format!("kvmem spill V store (layer {il}): {e}"))?;
        if il == 0 && verify_flag() {
            eprintln!(
                "[kvmem-verify] spill lp={lp} phys={phys} \
                 k={:016x} v={:016x}",
                fnv64(page_k),
                fnv64(page_v)
            );
        }
    }
    Ok(())
}

/// Copy a spilled page's bytes back into physical page `phys` of every arena.
/// The caller has already bound `phys` to the logical page, so the next
/// forward sees the restored bytes.
#[allow(clippy::too_many_arguments)]
fn restore_page(
    gpu: &Gpu,
    raw: &RawStore,
    page_k: &mut [u8],
    page_v: &mut [u8],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    k_page_bytes: usize,
    v_page_bytes: usize,
    phys: u32,
    lp: usize,
) -> Result<(), String> {
    let block = lp as u32;
    for il in 0..k_arenas.len() {
        if !raw.copy_k_gpu(block, il as u32, page_k, KV_BLOCK_TOKENS as u32) {
            return Err(format!(
                "kvmem restore: block {block} layer {il} has no K payload"
            ));
        }
        gpu.hip
            .memcpy_htod_offset(&k_arenas[il].buf, phys as usize * k_page_bytes, page_k)
            .map_err(|e| format!("kvmem restore K write (layer {il}): {e:?}"))?;
        if !raw.copy_v_gpu(block, il as u32, page_v, KV_BLOCK_TOKENS as u32) {
            return Err(format!(
                "kvmem restore: block {block} layer {il} has no V payload"
            ));
        }
        gpu.hip
            .memcpy_htod_offset(&v_arenas[il].buf, phys as usize * v_page_bytes, page_v)
            .map_err(|e| format!("kvmem restore V write (layer {il}): {e:?}"))?;
        if il == 0 && verify_flag() {
            let mut rb_k = vec![0u8; page_k.len()];
            let mut rb_v = vec![0u8; page_v.len()];
            gpu.hip
                .memcpy_dtoh_at(&mut rb_k, &k_arenas[il].buf, phys as usize * k_page_bytes)
                .map_err(|e| format!("kvmem verify K readback (layer {il}): {e:?}"))?;
            gpu.hip
                .memcpy_dtoh_at(&mut rb_v, &v_arenas[il].buf, phys as usize * v_page_bytes)
                .map_err(|e| format!("kvmem verify V readback (layer {il}): {e:?}"))?;
            let mk = page_k.iter().zip(&rb_k).filter(|(a, b)| a != b).count();
            let mv = page_v.iter().zip(&rb_v).filter(|(a, b)| a != b).count();
            eprintln!(
                "[kvmem-verify] restore lp={lp} phys={phys} \
                 k={:016x} v={:016x} readback_mismatch k={mk} v={mv}",
                fnv64(page_k),
                fnv64(page_v)
            );
        }
    }
    Ok(())
}

/// `SPILLED_PAGE` is re-exported for engine-side diagnostics; keep the import
/// honest for builds where the constant is only used in docs.
#[allow(dead_code)]
const _: u32 = SPILLED_PAGE;

#[cfg(test)]
mod tests {
    use super::*;
    use rdna_compute::DType;

    /// Small Q8 geometry: 2 fa layers, 2 KV heads, head_dim 64 →
    /// `per_pos_bytes = 2 * (64/32) * 34 = 136`.
    fn geom() -> KvMemGeometry {
        KvMemGeometry {
            n_fa_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 64,
            per_pos_bytes: 136,
            per_pos_v_bytes: 136,
            max_rows: 512,
        }
    }
    fn plan() -> KvMemPlan {
        KvMemPlan { pages_total: 16, budget_blocks: 4, grow_blocks: 2, interval: 1 }
    }

    /// Host-only: the strategy parser/resolver mirror the core ordinals and
    /// reject unknown spellings loudly, and the write-through staging is
    /// allocated ONLY under Layerwise (Legacy stays byte-neutral — no extra
    /// host buffer, no arena touch). The strategy is INJECTED, not `set_var`d:
    /// the process config is a start-time snapshot, so a runtime env write after
    /// the first config read would be an order-dependent coin flip.
    #[test]
    fn recall_strategy_parse_and_staging_gate() {
        assert_eq!(parse_recall_strategy("legacy"), Some(RecallStrategy::Legacy));
        assert_eq!(parse_recall_strategy("Layerwise"), Some(RecallStrategy::Layerwise));
        assert_eq!(parse_recall_strategy("0"), Some(RecallStrategy::Legacy));
        assert_eq!(parse_recall_strategy("1"), Some(RecallStrategy::Layerwise));
        assert_eq!(parse_recall_strategy("  LEGACY  "), Some(RecallStrategy::Legacy));
        assert_eq!(parse_recall_strategy("bogus"), None);
        assert!(!RecallStrategy::Legacy.is_layerwise());
        assert!(RecallStrategy::Layerwise.is_layerwise());

        // Pure resolution contract (the exact body `new` runs on the env value).
        assert_eq!(resolve_recall_strategy(None).unwrap(), RecallStrategy::Legacy);
        assert_eq!(
            resolve_recall_strategy(Some("legacy".into())).unwrap(),
            RecallStrategy::Legacy
        );
        assert_eq!(
            resolve_recall_strategy(Some("layerwise".into())).unwrap(),
            RecallStrategy::Layerwise
        );
        assert!(resolve_recall_strategy(Some("bogus".into())).is_err());

        // Small host ceiling for the tiny test geometry. The store commits
        // lazily, so the default ceiling is also safe; this just keeps the
        // reservation modest when this test is the first config reader.
        std::env::set_var("HIPFIRE_KVMEM_HOST_BYTES", "16777216");

        // Legacy: no write-through staging.
        let legacy =
            KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy)
                .expect("legacy ctl");
        assert_eq!(legacy.strategy(), RecallStrategy::Legacy);
        assert!(legacy.wt_k.is_empty() && legacy.wt_v.is_empty());
        // Layerwise: staging sized to max_rows * per_pos_bytes.
        let lw =
            KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
                .expect("layerwise ctl");
        assert_eq!(lw.strategy(), RecallStrategy::Layerwise);
        assert_eq!(lw.wt_k.len(), 512 * 136);
        assert_eq!(lw.wt_v.len(), 512 * 136);

        // The env wrapper resolves the Legacy default when the knob is absent
        // (the ambient test env never sets RECALL_STRATEGY, so the snapshot
        // cannot carry it regardless of read order).
        std::env::remove_var("HIPFIRE_KVMEM_RECALL_STRATEGY");
        let default = KvMemCtl::new(1, &plan(), &geom()).expect("default ctl");
        assert_eq!(default.strategy(), RecallStrategy::Legacy);
        assert!(default.wt_k.is_empty() && default.wt_v.is_empty());

        std::env::remove_var("HIPFIRE_KVMEM_HOST_BYTES");
    }

    /// Host-only: the Layerwise fail-closed policy guards (design §4, R2d-D1).
    /// The knobs are read through the process-start config snapshot
    /// (`hipfire_config::developer_var` is a `OnceLock`), so a runtime
    /// `set_var` is invisible — the guard decision is extracted pure
    /// (`resolve_layerwise_policy`, same injection rationale as
    /// `resolve_recall_strategy`) and the full table is tested here: under
    /// Layerwise each global working-set knob the per-layer subsystem replaces
    /// (select_method≠retrieval, lrfu, watermark, query_span>1) is an explicit
    /// Err naming both the knob and the strategy — no silent override/fallback;
    /// Legacy admits every combination (the guard is Layerwise-specific, so the
    /// default serve stays byte-neutral). The all-default Layerwise controller
    /// still constructs end-to-end, proving the wired guard call admits the
    /// default path.
    #[test]
    fn layerwise_policy_knobs_fail_closed() {
        // (select_method, lrfu, watermark_enabled, query_span, expected):
        // None = admitted, Some(fragment) = Err naming that knob. Each
        // non-default knob alone fails closed; the combined row proves the
        // first violation wins.
        let cases: &[(u32, bool, bool, usize, Option<&str>)] = &[
            (0, false, false, 1, None),
            (2, false, false, 1, Some("SELECT_METHOD")),
            (0, true, false, 1, Some("LRFU")),
            (0, false, true, 1, Some("WATERMARK")),
            (0, false, false, 2, Some("QUERY_SPAN")),
            (2, true, true, 8, Some("SELECT_METHOD")),
        ];
        for &(sm, lrfu, wm, qs, expected) in cases {
            let lw = resolve_layerwise_policy(RecallStrategy::Layerwise, sm, lrfu, wm, qs);
            match expected {
                None => lw.unwrap_or_else(|e| {
                    panic!("default Layerwise must admit (sm={sm} lrfu={lrfu} wm={wm} qs={qs}): {e}")
                }),
                Some(frag) => {
                    let msg = lw.err().unwrap_or_else(|| {
                        panic!(
                            "Layerwise must fail closed for sm={sm} lrfu={lrfu} wm={wm} qs={qs}"
                        )
                    });
                    assert!(
                        msg.contains("layerwise"),
                        "guard message names the strategy for {frag}: {msg}"
                    );
                    assert!(
                        msg.contains(frag),
                        "guard message names the knob {frag}: {msg}"
                    );
                }
            }
            // Legacy admits every combination — byte-neutral guard gating.
            resolve_layerwise_policy(RecallStrategy::Legacy, sm, lrfu, wm, qs)
                .unwrap_or_else(|e| {
                    panic!("Legacy must admit sm={sm} lrfu={lrfu} wm={wm} qs={qs}: {e}")
                });
        }

        // End-to-end: the wired guard call admits the all-default Layerwise
        // construction (HOST_BYTES is best-effort through the snapshot; the
        // store commits lazily, so the default ceiling is also safe).
        std::env::set_var("HIPFIRE_KVMEM_HOST_BYTES", "16777216");
        let ctl =
            KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
                .expect("default layerwise ctl");
        assert_eq!(ctl.strategy(), RecallStrategy::Layerwise);
        std::env::remove_var("HIPFIRE_KVMEM_HOST_BYTES");
    }

    /// The Layerwise driver borrows the slot's statistics store + policy through
    /// `layerwise_stores`: distinct objects per slot, `None` out of range, the
    /// policy borrow is live (`&self` budget set), and `reset_slot` builds a
    /// fresh empty store (the inline `SlotState` address is stable, so state —
    /// not pointer identity — is the probe) — the lifetime contract that forces
    /// the driver to be rebuilt per query (§3.9b).
    #[test]
    fn layerwise_stores_accessor_borrows_slot_state() {
        std::env::set_var("HIPFIRE_KVMEM_HOST_BYTES", "16777216");
        let mut ctl =
            KvMemCtl::new_with_strategy(2, &plan(), &geom(), RecallStrategy::Layerwise)
                .expect("ctl");
        let (raw0, kv0) = ctl.layerwise_stores(0).expect("slot 0");
        let (raw1, kv1) = ctl.layerwise_stores(1).expect("slot 1");
        assert!(!std::ptr::eq(raw0, raw1), "per-slot raw stores are distinct");
        assert!(!std::ptr::eq(kv0, kv1), "per-slot policies are distinct");
        assert!(ctl.layerwise_stores(2).is_none(), "out of range → None");
        // The policy borrow is live: a &self budget set through it succeeds.
        // The core admits a runtime budget only inside [sink+recent allocation,
        // configured select_budget]; the configured maximum is always in that
        // window, so it is the safe probe (the fixture's sink+recent floor is
        // strictly below it).
        kv0.set_runtime_select_budget(ctl.kv_cfg.select_budget)
            .expect("budget through borrow");
        // reset_slot builds a FRESH C++ store. SlotState lives inline in the
        // Vec, so its address is stable across reset — probe STATE, not pointer
        // identity. Allocate blocks on the old borrow; the post-reset borrow
        // must report a smaller (fresh, empty) allocation. This is the lifetime
        // contract that forces the driver to be rebuilt per query (§3.9b): a
        // borrow taken before reset references a store that no longer exists.
        raw0.ensure_blocks(4).expect("allocate on old store");
        let old_bytes = raw0.allocated_bytes();
        ctl.reset_slot(0).expect("reset");
        let (raw0_new, _) = ctl.layerwise_stores(0).expect("slot 0 after reset");
        assert!(
            raw0_new.allocated_bytes() < old_bytes,
            "reset yields a fresh empty store"
        );
        std::env::remove_var("HIPFIRE_KVMEM_HOST_BYTES");
    }

    /// Host-only: the decision-A (§3.10) persistent pressure owner + keep-set
    /// lifecycle. Legacy never creates the owner (fail-closed `ensure`, `None`
    /// borrow, empty keep-set — byte-neutral). Layerwise constructs lazily and
    /// idempotently with `block_tokens=128` pinned to the catalog and the select
    /// ceiling derived from the resident budget; the keep-set round-trips; and
    /// `reset_slot` drops BOTH the owner and the keep-set (a fresh session never
    /// aliases the previous session's accepted mass / selected pages). Finally,
    /// the bootstrap canonical `resident` is driven by the keep-set (filtered to
    /// the sealed history), replacing the pure re-bootstrap `[]`.
    #[test]
    fn layerwise_pressure_owner_and_keep_set_lifecycle() {
        std::env::set_var("HIPFIRE_KVMEM_HOST_BYTES", "16777216");
        let fa_layers = [0u32, 1];

        // Legacy: the owner is never created; ensure fails closed.
        let mut legacy =
            KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy)
                .expect("legacy ctl");
        assert!(legacy.layerwise_pressure(0).is_none(), "Legacy: no owner");
        assert_eq!(legacy.layerwise_keep_set(0), Some(&[][..]), "Legacy: empty keep-set");
        assert!(
            legacy.ensure_layerwise_pressure(0, 0, 0, &fa_layers).is_err(),
            "Legacy: ensure must fail closed (byte-neutral)"
        );
        assert!(legacy.layerwise_pressure(0).is_none(), "Legacy: still no owner after ensure");

        // Layerwise: lazy, idempotent construction.
        let mut lw =
            KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
                .expect("layerwise ctl");
        assert!(lw.layerwise_pressure(0).is_none(), "before ensure: no owner");
        assert!(lw.layerwise_pressure_mut(0).is_none(), "before ensure: no mut owner");
        lw.ensure_layerwise_pressure(0, 0, 0, &fa_layers).expect("construct owner");
        let opts = lw.layerwise_pressure(0).expect("owner").options().expect("options");
        assert_eq!(opts.block_tokens, KV_BLOCK_TOKENS as u32, "block_tokens pinned to catalog");
        assert_eq!(
            opts.select_tokens,
            (plan().budget_blocks * KV_BLOCK_TOKENS) as u64,
            "select ceiling = resident budget"
        );
        // Idempotent within a session: a second ensure is a no-op, owner persists.
        lw.ensure_layerwise_pressure(0, 0, 0, &fa_layers).expect("idempotent ensure");
        assert!(lw.layerwise_pressure(0).is_some(), "owner persists across ensure");
        assert!(lw.layerwise_pressure_mut(0).is_some(), "mut borrow available");

        // Per-layer keep-set round-trip (selection-state persistence).
        assert_eq!(lw.layerwise_keep_set(0), Some(&[][..]), "initial keep-set empty");
        lw.set_layerwise_keep_set(0, vec![vec![0, 2, 3], vec![1, 2]]);
        assert_eq!(
            lw.layerwise_keep_set(0),
            Some(&[vec![0u64, 2, 3], vec![1u64, 2]][..]),
            "per-layer keep-set round-trip"
        );

        // take/restore detaches the owner for the scoped forward borrow, then
        // returns it (the slot keeps its keep-set / counters throughout).
        let detached = lw.take_layerwise_pressure(0).expect("take owner");
        assert!(lw.layerwise_pressure(0).is_none(), "owner detached");
        assert_eq!(
            lw.layerwise_keep_set(0),
            Some(&[vec![0u64, 2, 3], vec![1u64, 2]][..]),
            "keep-set survives take"
        );
        lw.restore_layerwise_pressure(0, detached);
        assert!(lw.layerwise_pressure(0).is_some(), "owner restored");

        // Out-of-range accessors are None / Err (no panic).
        assert!(lw.layerwise_pressure(9).is_none());
        assert!(lw.layerwise_keep_set(9).is_none());
        assert!(lw.take_layerwise_pressure(9).is_none());
        assert!(lw.ensure_layerwise_pressure(9, 0, 0, &fa_layers).is_err());

        // reset_slot drops the owner AND the keep-set; incarnation bumps.
        let (inc_before, _) = lw.layerwise_counters(0).expect("counters");
        lw.reset_slot(0).expect("reset");
        assert!(lw.layerwise_pressure(0).is_none(), "reset drops the owner");
        assert_eq!(lw.layerwise_keep_set(0), Some(&[][..]), "reset clears the keep-set");
        let (inc_after, _) = lw.layerwise_counters(0).expect("counters after reset");
        assert_eq!(inc_after, inc_before + 1, "reset bumps incarnation");
        // A fresh session reconstructs under the new incarnation.
        lw.ensure_layerwise_pressure(0, 0, inc_after, &fa_layers).expect("reconstruct");
        assert!(lw.layerwise_pressure(0).is_some(), "reconstructed owner");

        // The bootstrap canonical resident is driven PER-LAYER by the keep-set,
        // filtered to the sealed history (fed=512 ⇒ 4 full pages; id 99 is out of
        // history). Ordinal 0 keeps [0,2], ordinal 1 keeps [1] (independent).
        let boot = crate::layerwise_driver::bootstrap_layerwise(
            512, 0, inc_after, 0, &fa_layers, 4, 2, 64, 4, 136,
            &[vec![0, 2, 99], vec![1]],
        )
        .expect("bootstrap with keep-set");
        assert_eq!(boot.canonical[0].memory.resident, vec![0u64, 2], "ordinal 0 filtered keep-set");
        assert_eq!(boot.canonical[1].memory.resident, vec![1u64], "ordinal 1 independent keep-set");
        let boot_empty = crate::layerwise_driver::bootstrap_layerwise(
            512, 0, inc_after, 0, &fa_layers, 4, 2, 64, 4, 136, &[],
        )
        .expect("bootstrap empty keep-set");
        for snap in &boot_empty.canonical {
            assert!(snap.memory.resident.is_empty(), "empty keep-set ⇒ re-bootstrap []");
        }

        std::env::remove_var("HIPFIRE_KVMEM_HOST_BYTES");
    }

    /// Host-only: the SESSION-lifetime pressure owner's finite, fail-closed
    /// admission (design §3.11, ③-E4; ninfer essence = "complete mutable history
    /// ceiling"). The owner's retained history is bounded by
    /// `n_fa_layers × ((budget_blocks + 1) × 24 + overhead)` and must fit the
    /// metadata pressure partition (the last transient 1/8, carved from E1's
    /// source share) BEFORE `PrefillPressure::new` acquires anything. Pure
    /// decision table (the quotas are read through the process-start `OnceLock`
    /// snapshot, so a runtime `set_var` is invisible — same injection rationale
    /// as `resolve_layerwise_policy`): the default geometry admits, a too-small
    /// partition fails closed naming the METADATA knob, the exact boundary
    /// admits while one byte over rejects, Legacy never admits the owner, and a
    /// saturating-huge geometry stays finite (Err, never a wrap). The wired
    /// success path is proven end-to-end by the lifecycle test above.
    #[test]
    fn layerwise_pressure_admission_fail_closed() {
        let d = LayerwiseQuotas::default();
        // Ceiling formula: 2 layers × ((2+1)×24 + 64) = 2 × 136 = 272B.
        assert_eq!(layerwise_pressure_ceiling(2, 2), 272);
        assert_eq!(layerwise_pressure_ceiling(1, 0), 88);
        // Partition = metadata / 8 (the last transient eighth).
        assert_eq!(layerwise_pressure_cap(d.metadata), 128 << 20);

        // Default quotas admit the tiny geometry with nothing acquired.
        resolve_pressure_admission(RecallStrategy::Layerwise, 2, 2, &d)
            .unwrap_or_else(|e| panic!("default Layerwise pressure must admit: {e}"));

        // Legacy never admits the owner (byte-neutral fail-closed).
        let legacy = resolve_pressure_admission(RecallStrategy::Legacy, 2, 2, &d)
            .err()
            .unwrap_or_else(|| panic!("Legacy must not admit a pressure owner"));
        assert!(legacy.contains("Layerwise"), "Legacy guard names the strategy: {legacy}");

        // A partition too small for the ceiling fails closed naming METADATA.
        for metadata in [0u64, 8, 703] {
            let q = LayerwiseQuotas { metadata, ..d.clone() };
            let msg = resolve_pressure_admission(RecallStrategy::Layerwise, 1, 0, &q)
                .err()
                .unwrap_or_else(|| panic!("metadata={metadata} must fail closed (ceiling 88B)"));
            assert!(msg.contains("layerwise"), "admission names the strategy: {msg}");
            assert!(
                msg.contains("HIPFIRE_KVMEM_LAYERWISE_METADATA_MIB"),
                "admission names the knob: {msg}"
            );
        }

        // Exact boundary: ceiling 88B == partition 88B admits; one byte short rejects.
        let exact = LayerwiseQuotas { metadata: 704, ..d.clone() }; // 704/8 == 88
        resolve_pressure_admission(RecallStrategy::Layerwise, 1, 0, &exact)
            .unwrap_or_else(|e| panic!("ceiling == partition must admit: {e}"));

        // Saturating-huge geometry stays finite and fails closed (no wrap).
        assert_eq!(layerwise_pressure_ceiling(usize::MAX, usize::MAX), u64::MAX);
        assert!(
            resolve_pressure_admission(RecallStrategy::Layerwise, usize::MAX, usize::MAX, &d)
                .is_err(),
            "saturating-huge ceiling must fail closed against a finite partition"
        );
    }

    /// Host-only: the five activation quotas resolve from MiB env integers with
    /// the ninfer seeds as defaults, reject malformed/overflowing values loudly
    /// (fail-closed, mirroring `resolve_recall_strategy`), and trim whitespace.
    /// Pure resolution contract — no process-config snapshot, no GPU.
    #[test]
    fn layerwise_quotas_resolve_defaults_overrides_fail_closed() {
        let d = LayerwiseQuotas::default();
        // Absent env → the ninfer engine seeds (64/32/8/1024/96 MiB).
        assert_eq!(
            resolve_layerwise_quotas(None, None, None, None, None, None).unwrap(),
            d
        );
        assert_eq!(d.activation_ram, 64 << 20);
        assert_eq!(d.activation_disk, 32 << 20);
        assert_eq!(d.staging, 8 << 20);
        assert_eq!(d.metadata, 1024 << 20);
        assert_eq!(d.activation_metadata, 96 << 20);
        assert_eq!(d.disk_directory, None);

        // Explicit MiB overrides (whitespace trimmed) + the spill-dir passthrough.
        let dir = PathBuf::from("/tmp/hipfire-quota-test");
        let q = resolve_layerwise_quotas(
            Some(" 128 ".into()),
            Some("64".into()),
            Some("16".into()),
            Some("2048".into()),
            Some("192".into()),
            Some(dir.clone()),
        )
        .unwrap();
        assert_eq!(q.activation_ram, 128 << 20);
        assert_eq!(q.activation_disk, 64 << 20);
        assert_eq!(q.staging, 16 << 20);
        assert_eq!(q.metadata, 2048 << 20);
        assert_eq!(q.activation_metadata, 192 << 20);
        assert_eq!(q.disk_directory, Some(dir));

        // Zero is a valid explicit quota (admission rejects it later, not here).
        assert_eq!(
            resolve_layerwise_quotas(Some("0".into()), None, None, None, None, None)
                .unwrap()
                .activation_ram,
            0
        );

        // Malformed / negative / fractional / empty values fail closed (no seed).
        assert!(resolve_layerwise_quotas(Some("abc".into()), None, None, None, None, None).is_err());
        assert!(resolve_layerwise_quotas(Some("-1".into()), None, None, None, None, None).is_err());
        assert!(resolve_layerwise_quotas(Some("1.5".into()), None, None, None, None, None).is_err());
        assert!(resolve_layerwise_quotas(None, Some("".into()), None, None, None, None).is_err());
        // u64::MAX MiB overflows the byte conversion.
        assert!(resolve_layerwise_quotas(
            Some(u64::MAX.to_string()), None, None, None, None, None
        ).is_err());
        // Each quota rejects independently.
        assert!(resolve_layerwise_quotas(None, None, Some("x".into()), None, None, None).is_err());
        assert!(resolve_layerwise_quotas(None, None, None, Some("x".into()), None, None).is_err());
        assert!(resolve_layerwise_quotas(None, None, None, None, Some("x".into()), None).is_err());
    }

    /// Host-only: the quota accessor is gated by strategy — Legacy carries the
    /// unread ninfer seeds (byte-neutral: a malformed quota env can never perturb
    /// the default path), while Layerwise resolves them. The ambient test env
    /// sets no LAYERWISE_*_MIB, so both resolve to the seeds here; the parsing
    /// contract itself is covered by the pure-resolver test above.
    #[test]
    fn layerwise_quotas_gated_by_strategy() {
        std::env::set_var("HIPFIRE_KVMEM_HOST_BYTES", "16777216");
        let legacy =
            KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Legacy)
                .expect("legacy ctl");
        assert_eq!(legacy.strategy(), RecallStrategy::Legacy);
        assert_eq!(legacy.layerwise_quotas(), &LayerwiseQuotas::default());

        let lw =
            KvMemCtl::new_with_strategy(1, &plan(), &geom(), RecallStrategy::Layerwise)
                .expect("layerwise ctl");
        assert_eq!(lw.strategy(), RecallStrategy::Layerwise);
        assert_eq!(lw.layerwise_quotas(), &LayerwiseQuotas::default());
        std::env::remove_var("HIPFIRE_KVMEM_HOST_BYTES");
    }

    /// GPU qualification of the write-through gather + seal (design §3.7).
    /// Fills the arenas with an independent byte pattern, drives two contiguous
    /// feeds (the second starts mid-block), and asserts the sealed archive reads
    /// back byte-exact against the pattern oracle resolved through the slot's
    /// REAL logical→physical map (a dummy slot offsets the physical pages so
    /// `phys != lp`, catching any gather that assumes identity mapping). Covers
    /// multi-page spans, a partial tail, and fail-closed on a non-resident page.
    ///
    /// `#[ignore]`d: needs a GPU + HIP toolchain.
    ///   cargo test -p hipfire-arch-qwen35 --features kvmem --release \
    ///     --lib kvmem_ctl::tests::write_through_archive_readback_byte_exact -- --ignored
    #[test]
    #[ignore]
    fn write_through_archive_readback_byte_exact() {
        std::env::set_var("HIPFIRE_KVMEM_HOST_BYTES", "16777216");
        let n_fa = 2usize;
        let row = 136usize;
        // Strategy injected (the config snapshot ignores runtime set_var).
        let mut ctl =
            KvMemCtl::new_with_strategy(2, &plan(), &geom(), RecallStrategy::Layerwise)
                .expect("ctl");

        let mut gpu = Gpu::init().expect("gpu init");
        let mut pool = SlotPool::new_paged_with_strides(2, 1024, row, row, 16).expect("pool");
        let dummy = pool.acquire().expect("dummy slot");
        pool.set_seq_len(dummy, 256).expect("dummy seq"); // grabs the first physical pages
        let slot = pool.acquire().expect("test slot");
        pool.set_seq_len(slot, 512).expect("test seq"); // 4 logical pages, phys != lp

        let kpb = KV_BLOCK_TOKENS * row;
        let arena_bytes = pool.k_arena_bytes();
        // Independent oracle: a deterministic byte pattern keyed on the PHYSICAL
        // page, intra-page position, byte index, layer, and K/V plane.
        let pat = |phys: usize, intra: usize, b: usize, il: usize, kind: usize| -> u8 {
            ((phys * 131 + intra * 17 + b * 7 + il * 251 + kind * 97) & 0xff) as u8
        };
        let mut k_arenas = Vec::new();
        let mut v_arenas = Vec::new();
        for il in 0..n_fa {
            let mut hk = vec![0u8; arena_bytes];
            let mut hv = vec![0u8; arena_bytes];
            for phys in 0..16 {
                for intra in 0..KV_BLOCK_TOKENS {
                    for b in 0..row {
                        let off = phys * kpb + intra * row + b;
                        hk[off] = pat(phys, intra, b, il, 0);
                        hv[off] = pat(phys, intra, b, il, 1);
                    }
                }
            }
            let tk = gpu.alloc_tensor(&[arena_bytes], DType::Raw).expect("k arena");
            gpu.hip.memcpy_htod(&tk.buf, &hk).expect("k h2d");
            let tv = gpu.alloc_tensor(&[arena_bytes], DType::Raw).expect("v arena");
            gpu.hip.memcpy_htod(&tv.buf, &hv).expect("v h2d");
            k_arenas.push(tk);
            v_arenas.push(tv);
        }
        gpu.hip.device_synchronize().expect("sync");

        // Two contiguous feeds: 0..300, then 300..400 (mid-block start on page 2).
        ctl.write_through_archive(&gpu, &pool, &k_arenas, &v_arenas, slot.0, 0, 300)
            .expect("feed 0..300");
        ctl.write_through_archive(&gpu, &pool, &k_arenas, &v_arenas, slot.0, 300, 100)
            .expect("feed 300..400");

        // Readback byte-exact against the oracle, per logical page / layer.
        let bt = pool.block_table(slot).expect("block table");
        let total_fed = 400usize;
        let mut kbuf = vec![0u8; kpb];
        let mut vbuf = vec![0u8; kpb];
        for lp in 0..bt.num_pages() {
            let phys = bt.physical(lp).expect("fed page resident") as usize;
            let fed = total_fed.saturating_sub(lp * KV_BLOCK_TOKENS).min(KV_BLOCK_TOKENS);
            if fed == 0 {
                continue;
            }
            for il in 0..n_fa {
                let raw = &ctl.slots[slot.0].raw;
                assert!(raw.has_k_gpu(lp as u32, il as u32, fed as u32), "K present lp={lp}");
                assert!(raw.has_v_gpu(lp as u32, il as u32, fed as u32), "V present lp={lp}");
                assert!(raw.copy_k_gpu(lp as u32, il as u32, &mut kbuf[..fed * row], fed as u32));
                assert!(raw.copy_v_gpu(lp as u32, il as u32, &mut vbuf[..fed * row], fed as u32));
                for intra in 0..fed {
                    for b in 0..row {
                        assert_eq!(
                            kbuf[intra * row + b], pat(phys, intra, b, il, 0),
                            "K mismatch lp={lp} phys={phys} il={il} intra={intra} b={b}"
                        );
                        assert_eq!(
                            vbuf[intra * row + b], pat(phys, intra, b, il, 1),
                            "V mismatch lp={lp} phys={phys} il={il} intra={intra} b={b}"
                        );
                    }
                }
            }
        }
        // Fail-closed: a feed beyond the provisioned pages has no resident
        // physical page, so the gather must error before any archive mutation.
        assert!(
            ctl.write_through_archive(&gpu, &pool, &k_arenas, &v_arenas, slot.0, 512, 128).is_err(),
            "non-resident page must fail closed"
        );

        for t in k_arenas {
            gpu.free_tensor(t).expect("free k arena");
        }
        for t in v_arenas {
            gpu.free_tensor(t).expect("free v arena");
        }
        std::env::remove_var("HIPFIRE_KVMEM_HOST_BYTES");
    }
}
