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
    mean_key_retrieval_scores, GqaAggregation, KvStore, KvStoreConfig, RawStore, RawStoreConfig,
    Storage,
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
}

impl KvMemCtl {
    pub fn new(
        n_slots: usize,
        plan: &KvMemPlan,
        geom: &KvMemGeometry,
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
        let kv_cfg = KvStoreConfig {
            block_tokens: KV_BLOCK_TOKENS as u32,
            select_budget: budget_tokens,
            prefill_budget: 0,
            gen_budget: budget_tokens,
            sink_blocks: sink as u32,
            recent_blocks: recent as u32,
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
        for _ in 0..n_slots {
            slots.push(SlotState {
                raw: RawStore::new(raw_cfg, Some(&storage))
                    .map_err(|e| format!("kvmem raw store: {e}"))?,
                kv: KvStore::new(&kv_cfg).map_err(|e| format!("kvmem store: {e}"))?,
                last_select_full: 0,
                fed: 0,
                turn_start: 0,
                first_decode_done: false,
            });
        }

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
            staging: vec![0f32; geom.max_rows * k_width],
            sums: vec![0f32; geom.n_fa_layers * q_width],
            q_row: vec![0f32; q_width],
            counts: vec![1u32; geom.n_fa_layers],
            xfer_slab: None,
            batch_xfer: batch_xfer_flag(),
        })
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
        self.slots[slot] = SlotState {
            raw,
            kv,
            last_select_full: 0,
            fed: 0,
            turn_start: 0,
            first_decode_done: false,
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
        Ok(st.fed)
    }

    /// Host-store length of `slot` (tokens whose statistics are recorded).
    pub fn fed_tokens(&self, slot: usize) -> Option<u32> {
        self.slots.get(slot).map(|s| s.fed)
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

    /// Refresh the retrieval query for `slot` from the tap and score its
    /// resident+spilled blocks. Each layer contributes its tapped last Q row
    /// (post-RMSNorm, pre-RoPE, one token) — the frame
    /// `mean_key_retrieval_scores` expects.
    fn refresh_scores(
        &mut self,
        gpu: &Gpu,
        taps: &KvMemTaps,
        slot: usize,
        pages: usize,
    ) -> Result<Vec<f64>, String> {
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
            if pages <= self.budget_blocks {
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
                let scores = self.refresh_scores(gpu, taps, s, pages)?;
                self.slots[s]
                    .kv
                    .set_retrieval_scores(&scores)
                    .map_err(|e| format!("kvmem set scores: {e}"))?;
                self.slots[s]
                    .kv
                    .pick_topk_blocks(&[])
                    .map_err(|e| format!("kvmem decode selection: {e}"))?
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
