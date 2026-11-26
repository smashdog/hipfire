#![allow(dead_code)]

//! Legacy MTP-head paged window (P1b follow-up).
//!
//! The Legacy head used to be one contiguous cache plus a whole-buffer
//! `slide_to_shift` whenever the window advanced. ninfer instead keeps the MTP
//! KV in a paged cache and remaps a block table (`mtp_attention_window`), so a
//! window move only changes the table, never the bytes.
//!
//! This module is the host-side page-table model for the same trick. It does
//! not allocate or touch device memory: the existing bounded
//! [`Qwen35MtpHeadKvCache`] keeps the physical page slots, and the caller
//! uploads [`Self::block_table`] + [`Self::visible_tokens`] to the head's
//! `KvSlotDesc`. Keeping the geometry host-only makes the eviction/sink rules
//! unit-testable without a GPU.
//!
//! [`Qwen35MtpHeadKvCache`]: crate::mtp_head::Qwen35MtpHeadKvCache

use std::collections::VecDeque;

use hip_bridge::DeviceBuffer;
use rdna_compute::layerwise_pool::SlotHandle;
use rdna_compute::{Gpu, GpuTensor};
use saddle_core::kv::KvMode;

use crate::layerwise_driver::{LayerwiseArena, LayerwiseDescStaging};
use crate::layerwise_layout::LayerwiseQ8Layout;
use crate::mtp_head::HeadPagedKv;

/// Logical page size of the head KV. Must match `KV_BLOCK_TOKENS` and the
/// layerwise page geometry; the head shares the trunk's Q8 row layout.
pub const HEAD_PAGE_TOKENS: u32 = 128;

/// Widest accepted span the head plane stages at once: a full prompt prefill
/// chunk (`HIPFIRE_PREFILL_CHUNK` default 1024, capped at 1024) or an MTP
/// verify burst (`mtp_k + 1`, ceiling 64). Larger override chunks fall back to
/// the contiguous path for that chunk.
const HEAD_STAGE_ROWS: usize = 1024;

/// The head's bounded window in LOGICAL pages.
///
/// Page 0 is the attention sink and is always mapped. Pages `1..` are a FIFO
/// of the most recent pages. The compact frame is
/// `[sink page][oldest retained window page]...[newest page]`, exactly the
/// frame the paged head kernel addresses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyHeadWindow {
    window_pages: u32,
    /// Retained window pages, oldest first. Never contains the sink page.
    pages: VecDeque<u32>,
    /// Number of tokens already written through the head (global frontier).
    frontier: u32,
}

impl LegacyHeadWindow {
    /// `window_pages` is the number of recent pages the bounded cache can hold
    /// after the sink page. Zero is invalid: the draft span would have nowhere
    /// to append.
    pub fn new(window_pages: u32) -> Result<Self, String> {
        if window_pages == 0 {
            return Err("legacy head window: window_pages == 0".to_string());
        }
        Ok(Self {
            window_pages,
            pages: VecDeque::new(),
            frontier: 0,
        })
    }

    pub fn window_pages(&self) -> u32 {
        self.window_pages
    }

    pub fn frontier(&self) -> u32 {
        self.frontier
    }

    /// Number of retained window pages (excluding the sink).
    pub fn retained_pages(&self) -> u32 {
        self.pages.len() as u32
    }

    /// Sink page is always logical page 0.
    pub fn sink_page(&self) -> u32 {
        0
    }

    /// The first retained window page, or `None` before the first page.
    pub fn first_window_page(&self) -> Option<u32> {
        self.pages.front().copied()
    }

    /// The newest retained page, or `None` before the first page.
    pub fn last_page(&self) -> Option<u32> {
        self.pages.back().copied()
    }

    /// Rebuild the window for `frontier` tokens. This is the only mutator:
    /// pages are derived from the frontier, never from an independent cursor,
    /// so restore and live append always agree.
    ///
    /// The sink page is `0`; the retained window is the last
    /// `window_pages` pages before `frontier.div_ceil(HEAD_PAGE_TOKENS)`.
    pub fn set_frontier(&mut self, frontier: u32) {
        self.frontier = frontier;
        self.pages.clear();
        // The newest page is the page of the last written token, not the page
        // the next token will start. At `frontier == 128` page 1 has no rows
        // yet, so the window is only the sink.
        let last = frontier.saturating_sub(1) / HEAD_PAGE_TOKENS;
        if last == 0 {
            return;
        }
        // Page 0 is the sink. Retained window pages start at 1 and run through
        // `last` inclusive (a partial frontier's page is the newest page).
        let keep = self.window_pages;
        let first = (last + 1).saturating_sub(keep).max(1);
        for page in first..=last {
            self.pages.push_back(page);
        }
    }

    /// Advance the frontier by `tokens` (accepted/committed head rows).
    pub fn advance(&mut self, tokens: u32) {
        self.set_frontier(self.frontier.saturating_add(tokens));
    }

    /// Compact logical pages: sink first, then retained window pages oldest to
    /// newest. This is the order the block table must carry.
    pub fn compact_pages(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.pages.len() + 1);
        out.push(self.sink_page());
        out.extend(self.pages.iter().copied());
        out
    }

    /// Physical slot for a logical page in the bounded head cache. The sink is
    /// slot 0; window pages use `1 + (page - 1) % (window_pages + 1)`. Retained
    /// pages are collision-free, and the extra modulo slot is the draft's
    /// spare page (one page beyond the retained frontier). A page keeps its
    /// physical slot when the window slides, so no retained byte moves.
    pub fn physical_slot(&self, page: u32) -> u32 {
        if page == self.sink_page() {
            0
        } else {
            1 + (page - 1) % (self.window_pages + 1)
        }
    }

    /// Physical slots in compact order (the block table payload).
    pub fn compact_slots(&self) -> Vec<u32> {
        self.compact_pages()
            .into_iter()
            .map(|page| self.physical_slot(page))
            .collect()
    }

    /// Compact visible token count BEFORE the next draft write: one full sink
    /// page plus the retained window from its first page through `frontier`.
    /// The next draft row writes at [`Self::write_position`].
    pub fn visible_tokens(&self) -> u32 {
        if self.frontier == 0 {
            return 0;
        }
        match self.first_window_page() {
            // The sink page is still filling (or exactly full at `frontier ==
            // HEAD_PAGE_TOKENS`): the compact frame is the global prefix.
            None | Some(1) => self.frontier,
            // The window has slid: the sink page plus the retained tail.
            Some(first) => HEAD_PAGE_TOKENS
                .saturating_add(self.frontier - first.saturating_mul(HEAD_PAGE_TOKENS)),
        }
    }

    /// Compact write position for the next head row.
    pub fn write_position(&self) -> u32 {
        self.visible_tokens()
    }

    /// Causal bound the paged kernel scans for a `width`-row draft/verify
    /// span. `width == 1` is the single draft row; the span starts at the
    /// current compact frontier.
    pub fn seq_len_for_width(&self, width: u32) -> u32 {
        self.visible_tokens().saturating_add(width)
    }

    /// Descriptor extent and physical block table for one draft row. The draft
    /// page is derived from the GLOBAL frontier, not the compact `seq_len`:
    /// after the window slides those coordinate systems differ. At a compact
    /// page boundary the next global page is not retained yet and must use the
    /// arena's spare physical slot.
    fn draft_layout(&self, row: usize) -> Result<(u32, Vec<u32>), String> {
        let row = u32::try_from(row).map_err(|_| "legacy head plane: draft row overflow")?;
        let seq_len = self
            .visible_tokens()
            .checked_add(row)
            .and_then(|v| v.checked_add(1))
            .ok_or("legacy head plane: draft position overflow")?;
        let global_pos = self
            .frontier
            .checked_add(row)
            .ok_or("legacy head plane: global draft position overflow")?;
        let draft_page = global_pos / HEAD_PAGE_TOKENS;
        let mut compact_slots = self.compact_slots();
        if !self.contains(draft_page) {
            compact_slots.push(self.physical_slot(draft_page));
        }
        let tiles = seq_len.div_ceil(HEAD_PAGE_TOKENS) as usize;
        if tiles > compact_slots.len() {
            return Err(format!(
                "legacy head plane: draft seq_len {seq_len} needs {tiles} tiles > {} mapped pages",
                compact_slots.len()
            ));
        }
        Ok((seq_len, compact_slots))
    }

    /// Whether the retained window contains `page`.
    pub fn contains(&self, page: u32) -> bool {
        page == self.sink_page() || self.pages.iter().any(|&p| p == page)
    }

    /// Compact-frame offset of `page`, or `None` when the page is not mapped.
    /// The sink is compact 0; the retained window follows it oldest first.
    pub fn compact_index(&self, page: u32) -> Option<u32> {
        if page == self.sink_page() {
            return Some(0);
        }
        let first = self.first_window_page()?;
        if !self.contains(page) {
            return None;
        }
        Some(HEAD_PAGE_TOKENS + (page - first) * HEAD_PAGE_TOKENS)
    }

    /// Largest token count the bounded cache can address in the compact frame:
    /// the sink, every retained window page, and the draft's spare page.
    pub fn capacity_tokens(&self) -> u32 {
        (self.window_pages + 2) * HEAD_PAGE_TOKENS
    }
}

/// The Legacy head's bounded paged arena.
///
/// Unlike [`LegacyHeadWindow`] this owns device bytes: one `window_pages + 1`
/// slot arena (sink + window), the one-layer descriptor staging the paged head
/// kernel consumes, and a 1-element GLOBAL RoPE position buffer. It is a draft
/// cache: accepted rows are written through to the composite aux plane, and the
/// next draft rebuilds this arena's selected pages from that archive. That is
/// the Legacy counterpart of the Layerwise head carrier — no trunk pages, no
/// per-layer policy.
pub struct LegacyHeadPlane {
    window: LegacyHeadWindow,
    arena: LayerwiseArena,
    staging: LayerwiseDescStaging,
    slots: Vec<SlotHandle>,
    k_view: GpuTensor,
    v_view: GpuTensor,
    rope_pos: DeviceBuffer,
    visible: u32,
    /// Frontier whose accepted rows this arena already holds. `restore` may
    /// skip the archive copy when the caller asks for exactly this frontier
    /// (the normal verify -> next draft path).
    synced_frontier: Option<u32>,
    /// Last compact page reported by `restore`, so E2E logs prove the paged
    /// path (not the contiguous fallback) served a frontier.
    reported_page: u32,
    /// Last page reported by a successful `mark_filled`.
    reported_fill_page: u32,
    /// `HIPFIRE_KVMEM_TRACE` diagnostics (the normal path is silent).
    trace: bool,
}

impl LegacyHeadPlane {
    /// Allocate a plane for `window_pages` recent pages plus one sink page.
    /// `n_kv_heads`/`head_dim` are the MTP head's attention geometry; `kv_mode`
    /// is the trunk tier identity (the head shares its Q8 row layout).
    pub fn new(
        gpu: &mut Gpu,
        kv_mode: KvMode,
        n_kv_heads: usize,
        head_dim: usize,
        window_pages: u32,
    ) -> Result<Self, String> {
        let window = LegacyHeadWindow::new(window_pages)?;
        let layout = LayerwiseQ8Layout::resolve(kv_mode, 1, n_kv_heads, head_dim, &[0])
            .map_err(|e| format!("legacy head plane layout: {e}"))?;
        // Sink + retained window + one draft spare page.
        let capacity = (window_pages + 2) as usize;
        let mut arena = LayerwiseArena::new(gpu, &layout, capacity)
            .map_err(|e| format!("legacy head plane arena: {e}"))?;
        let staging = match LayerwiseDescStaging::new_with_width(gpu, 1, capacity, HEAD_STAGE_ROWS)
        {
            Ok(s) => s,
            Err(e) => {
                let _ = arena.free(gpu);
                return Err(format!("legacy head plane staging: {e}"));
            }
        };
        // Allocate every physical slot up front. The pool is only mutated here
        // and in `restore`; positions never move, so a page keeps its slot when
        // the window slides (the whole point of the block table).
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            match arena.pool().alloc_slot() {
                Ok(handle) => slots.push(handle),
                Err(e) => {
                    staging.free_gpu(gpu);
                    let _ = arena.free(gpu);
                    return Err(format!("legacy head plane slot: {e}"));
                }
            }
        }
        let (k_view, v_view) = match arena.stripe_views(slots[0].phys as usize, 0) {
            Ok((k, _)) => {
                let v = match arena.v_cache_view() {
                    Ok(v) => v,
                    Err(e) => {
                        staging.free_gpu(gpu);
                        let _ = arena.free(gpu);
                        return Err(format!("legacy head plane V view: {e}"));
                    }
                };
                (k, v)
            }
            Err(e) => {
                staging.free_gpu(gpu);
                let _ = arena.free(gpu);
                return Err(format!("legacy head plane K view: {e}"));
            }
        };
        let rope_pos = match gpu.hip.malloc(4) {
            Ok(b) => b,
            Err(e) => {
                staging.free_gpu(gpu);
                let _ = arena.free(gpu);
                return Err(format!("legacy head plane rope buffer: {e:?}"));
            }
        };
        Ok(Self {
            window,
            arena,
            staging,
            slots,
            k_view,
            v_view,
            rope_pos,
            visible: 0,
            synced_frontier: None,
            reported_page: u32::MAX,
            reported_fill_page: u32::MAX,
            trace: std::env::var("HIPFIRE_KVMEM_TRACE")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(false),
        })
    }

    pub fn window(&self) -> &LegacyHeadWindow {
        &self.window
    }

    /// Compact-position → physical arena slot. Each logical page keeps the
    /// slot [`LegacyHeadWindow::physical_slot`] assigns it, so a window slide
    /// only changes the table order; no retained page's bytes move.
    fn compact_phys_slots(&self) -> Vec<u32> {
        self.window
            .compact_pages()
            .into_iter()
            .map(|page| self.slots[self.window.physical_slot(page) as usize].phys)
            .collect()
    }

    /// Rebuild the arena's sink + recent window from the composite aux plane.
    /// The caller has already restored the trunk and written the accepted head
    /// rows through to the aux plane; this is a read-only archive view.
    pub fn restore(
        &mut self,
        gpu: &mut Gpu,
        ctl: &super::KvMemCtl,
        slot: usize,
        frontier: u32,
    ) -> Result<(), String> {
        // The verify/prefill fill writes accepted rows straight into this
        // arena (`stage_rows` + `mark_filled`), so the next draft usually
        // asks for the exact frontier already resident. Copying the append
        // pages back from the archive would undo that work.
        if self.synced_frontier == Some(frontier) && self.window.frontier() == frontier {
            if self.trace {
                eprintln!(
                    "[kvmem-head] paged restore hit slot={slot} frontier={frontier} visible={}",
                    self.visible
                );
            }
            return Ok(());
        }
        self.window.set_frontier(frontier);
        self.visible = self.window.visible_tokens();
        self.synced_frontier = None;
        let pages = self.window.compact_pages();
        if pages.len() > self.slots.len() {
            return Err(format!(
                "legacy head plane: {} compact pages > {} slots",
                pages.len(),
                self.slots.len()
            ));
        }
        let (k_row, v_row) = ctl.head_row_bytes();
        let page_tokens = HEAD_PAGE_TOKENS as usize;
        let mut k_buf = vec![0u8; page_tokens * k_row];
        let mut v_buf = vec![0u8; page_tokens * v_row];
        for &page in &pages {
            // Pages keep their physical slot across window slides, so a
            // retained page never moves bytes when the window advances.
            let idx = self.window.physical_slot(page) as usize;
            let phys = self.slots[idx].phys;
            let resident = self.arena.pool_ref().stripe_page(phys, 0) == Some(page);
            // The sink and the last two window pages are append frontiers:
            // verify/prefill wrote newer accepted rows through to the archive
            // since the previous draft, so their resident copy is stale. Every
            // older page is immutable and can stay mapped.
            let append_frontier = page == 0
                || self.window.last_page() == Some(page)
                || self.window.last_page().is_some_and(|last| page + 1 == last);
            if resident && !append_frontier {
                continue;
            }
            let rows = if page == 0 {
                page_tokens.min(frontier as usize)
            } else {
                (frontier.saturating_sub(page * HEAD_PAGE_TOKENS)).min(HEAD_PAGE_TOKENS) as usize
            };
            if rows == 0 {
                continue;
            }
            ctl.copy_legacy_head_page(slot, page, rows as u32, &mut k_buf, &mut v_buf)
                .map_err(|e| format!("legacy head plane copy page {page}: {e}"))?;
            let handle = self.slots[idx];
            let pool = self.arena.pool();
            if pool.stripe_page(phys, 0).is_some() {
                pool.drop_stripe(handle, 0)
                    .map_err(|e| format!("legacy head plane drop slot {idx}: {e}"))?;
            }
            pool.bind_stripe(handle, 0, page)
                .map_err(|e| format!("legacy head plane bind page {page}: {e}"))?;
            let (k, v) = self
                .arena
                .stripe_views(phys as usize, 0)
                .map_err(|e| format!("legacy head plane views {idx}: {e}"))?;
            gpu.hip
                .memcpy_htod_offset(&k.buf, 0, &k_buf[..rows * k_row])
                .map_err(|e| format!("legacy head plane K page {page}: {e:?}"))?;
            gpu.hip
                .memcpy_htod_offset(&v.buf, 0, &v_buf[..rows * v_row])
                .map_err(|e| format!("legacy head plane V page {page}: {e:?}"))?;
        }
        // Bind exactly the compact pages in the table. Sink first, then the
        // retained window oldest-to-newest.
        let compact_slots = self.compact_phys_slots();
        self.staging
            .update_layer(gpu, 0, &compact_slots, None, None, self.visible as i32)
            .map_err(|e| format!("legacy head plane table: {e}"))?;
        if self.visible > 0 {
            let bounds = [self.visible as i32 - 1];
            self.staging
                .upload_positions(gpu, 0, &bounds)
                .map_err(|e| format!("legacy head plane positions: {e}"))?;
        }
        self.synced_frontier = Some(frontier);
        let page = self.visible / HEAD_PAGE_TOKENS;
        if page != self.reported_page {
            self.reported_page = page;
            eprintln!(
                "[kvmem-head] paged restore slot={slot} frontier={} visible={} pages={} compact={:?}",
                self.window.frontier(),
                self.visible,
                self.window.retained_pages(),
                self.window.compact_pages(),
            );
        }
        Ok(())
    }

    /// Stage an accepted `rows`-token span that begins exactly at the current
    /// frontier and ends at `frontier_after`. The window is remapped for the
    /// post-fill frontier (so a span that crosses a page boundary lands in the
    /// newly mapped page), every destination page must stay resident, and the
    /// compact write slots are returned with the global-to-compact RoPE shift.
    /// The caller must call [`Self::mark_filled`] only after the forward
    /// succeeds; a failed forward leaves the arena un-synced so the next draft
    /// restores from the archive.
    pub fn stage_rows(
        &mut self,
        gpu: &mut Gpu,
        start_pos: u32,
        rows: u32,
        frontier_after: u32,
    ) -> Result<HeadPagedKv<'_>, String> {
        if rows == 0 {
            return Err("legacy head plane: stage_rows with zero rows".to_string());
        }
        if start_pos.checked_add(rows) != Some(frontier_after) {
            return Err(format!(
                "legacy head plane: stage_rows span {start_pos}+{rows} != {frontier_after}"
            ));
        }
        let old_frontier = self.window.frontier();
        let old_visible = self.visible;
        if start_pos != old_frontier {
            return Err(format!(
                "legacy head plane: stage_rows starts at {start_pos}, frontier is {old_frontier}"
            ));
        }
        self.window.set_frontier(frontier_after);
        let rollback = |this: &mut Self| {
            this.window.set_frontier(old_frontier);
            this.visible = old_visible;
        };
        let visible = self.window.visible_tokens();
        let page_tokens = HEAD_PAGE_TOKENS;
        let start_page = start_pos / page_tokens;
        let end_page = (frontier_after - 1) / page_tokens;
        let Some(page_base) = self.window.compact_index(start_page) else {
            rollback(self);
            return Err(format!(
                "legacy head plane: stage start page {start_page} left the window"
            ));
        };
        if !self.window.contains(end_page) {
            rollback(self);
            return Err(format!(
                "legacy head plane: stage end page {end_page} left the window"
            ));
        }
        let compact_start = page_base + start_pos % page_tokens;
        let Some(end) = compact_start.checked_add(rows) else {
            rollback(self);
            return Err("legacy head plane: compact stage span overflow".to_string());
        };
        if end != visible || end > self.window.capacity_tokens() {
            rollback(self);
            return Err(format!(
                "legacy head plane: compact span {}..{end} does not end at visible {visible}",
                compact_start
            ));
        }
        let compact_slots = self.compact_phys_slots();
        if let Err(e) = self
            .staging
            .update_layer(gpu, 0, &compact_slots, None, None, end as i32)
        {
            rollback(self);
            return Err(format!("legacy head plane stage rows: {e}"));
        }
        let positions: Vec<i32> = (compact_start..end).map(|p| p as i32).collect();
        if let Err(e) = self.staging.upload_positions(gpu, 0, &positions) {
            rollback(self);
            return Err(format!("legacy head plane stage rows positions: {e}"));
        }
        self.visible = visible;
        self.synced_frontier = None;
        Ok(self.paged_kv())
    }

    /// Record that a [`Self::stage_rows`] span reached the archive and the
    /// arena now holds `frontier` exactly.
    pub fn mark_filled(&mut self, frontier: u32) {
        if self.window.frontier() == frontier {
            self.synced_frontier = Some(frontier);
            let page = frontier / HEAD_PAGE_TOKENS;
            if self.trace && page != self.reported_fill_page {
                self.reported_fill_page = page;
                eprintln!(
                    "[kvmem-head] paged fill slot-synced frontier={frontier} visible={} pages={}",
                    self.visible,
                    self.window.retained_pages()
                );
            }
        }
    }

    /// Advance the descriptor's COMPACT write position for one draft row and
    /// return the kernel-facing view. `row` is the 0-based draft row within the
    /// chain; the global RoPE position is written separately by the caller.
    pub fn stage_row(&mut self, gpu: &mut Gpu, row: usize) -> Result<HeadPagedKv<'_>, String> {
        let (seq_len, compact_slots) = self.window.draft_layout(row)?;
        let compact_slots = compact_slots
            .into_iter()
            .map(|phys| {
                self.slots
                    .get(phys as usize)
                    .map(|slot| slot.phys)
                    .ok_or_else(|| format!("legacy head plane: draft slot {phys} missing"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.staging
            .update_layer(gpu, 0, &compact_slots, None, None, seq_len as i32)
            .map_err(|e| format!("legacy head plane stage row {row}: {e}"))?;
        let bounds = [seq_len as i32 - 1];
        self.staging
            .upload_positions(gpu, 0, &bounds)
            .map_err(|e| format!("legacy head plane stage row {row} positions: {e}"))?;
        Ok(self.paged_kv())
    }

    /// Stage one draft row and its GLOBAL RoPE position; returns the
    /// kernel-facing paged view. The row's KV write lands at compact
    /// `visible + row` inside the sink/window arena.
    pub fn stage_row_with_rope(
        &mut self,
        gpu: &mut Gpu,
        row: usize,
        pos: usize,
    ) -> Result<HeadPagedKv<'_>, String> {
        let rope = i32::try_from(pos)
            .map_err(|_| format!("legacy head plane: rope pos {pos} overflow i32"))?;
        gpu.hip
            .memcpy_htod(&self.rope_pos, &rope.to_ne_bytes())
            .map_err(|e| format!("legacy head plane rope pos {pos}: {e:?}"))?;
        self.stage_row(gpu, row)
    }

    /// Gather packed Q8 K/V rows `[pos0, pos0 + n_rows)` out of the arena into
    /// caller buffers. This is the paged-only replacement for the contiguous
    /// cache's single D2H gather: split by 128-token page, then copy the
    /// in-page row range from that page's stable physical slot.
    pub fn gather_rows(
        &self,
        gpu: &Gpu,
        ctl: &super::KvMemCtl,
        pos0: usize,
        n_rows: usize,
        k_rows: &mut [u8],
        v_rows: &mut [u8],
    ) -> Result<(), String> {
        if n_rows == 0 {
            return Ok(());
        }
        let (krow, vrow) = ctl.head_row_bytes();
        if k_rows.len() < n_rows * krow || v_rows.len() < n_rows * vrow {
            return Err(format!(
                "legacy head plane gather: {} rows need {} K / {} V bytes, got {} / {}",
                n_rows,
                n_rows * krow,
                n_rows * vrow,
                k_rows.len(),
                v_rows.len()
            ));
        }
        let mut row = 0usize;
        while row < n_rows {
            let pos = pos0 + row;
            let page = (pos / HEAD_PAGE_TOKENS as usize) as u32;
            let in_page = pos % HEAD_PAGE_TOKENS as usize;
            let take = n_rows
                .saturating_sub(row)
                .min(HEAD_PAGE_TOKENS as usize - in_page);
            let Some(_compact) = self.window.compact_index(page) else {
                return Err(format!(
                    "legacy head plane gather: page {page} is outside the retained window"
                ));
            };
            let phys = self.window.physical_slot(page) as usize;
            let handle = self
                .slots
                .get(phys)
                .ok_or_else(|| format!("legacy head plane gather: slot {phys} missing"))?;
            let (k_view, v_view) = self
                .arena
                .stripe_views(handle.phys as usize, 0)
                .map_err(|e| format!("legacy head plane gather views: {e}"))?;
            let src_row = in_page;
            let dst = row * krow;
            gpu.hip
                .memcpy_dtoh_at(
                    &mut k_rows[dst..dst + take * krow],
                    &k_view.buf,
                    src_row * krow,
                )
                .map_err(|e| format!("legacy head plane gather K: {e}"))?;
            let dst = row * vrow;
            gpu.hip
                .memcpy_dtoh_at(
                    &mut v_rows[dst..dst + take * vrow],
                    &v_view.buf,
                    src_row * vrow,
                )
                .map_err(|e| format!("legacy head plane gather V: {e}"))?;
            row += take;
        }
        Ok(())
    }

    pub fn paged_kv(&self) -> HeadPagedKv<'_> {
        HeadPagedKv {
            k_view: &self.k_view,
            v_view: &self.v_view,
            positions: self
                .staging
                .positions(0)
                .expect("legacy head plane positions"),
            descs: self.staging.descs(0).expect("legacy head plane descs"),
            row_slot: self.staging.row_slot(),
            ctx_len: self.window.capacity_tokens() as usize,
            rope_pos_buf: &self.rope_pos,
        }
    }

    pub fn rope_pos(&self) -> &DeviceBuffer {
        &self.rope_pos
    }

    pub fn visible_tokens(&self) -> u32 {
        self.visible
    }

    pub fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        let Self {
            arena,
            staging,
            rope_pos,
            ..
        } = self;
        let _ = gpu.hip.free(rope_pos);
        staging.free_gpu(gpu);
        arena.free(gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_keeps_sink_and_drops_oldest() {
        let mut w = LegacyHeadWindow::new(3).unwrap();
        w.set_frontier(5 * HEAD_PAGE_TOKENS + 7);
        assert_eq!(w.sink_page(), 0);
        assert_eq!(w.compact_pages(), vec![0, 3, 4, 5]);
        assert_eq!(w.retained_pages(), 3);
        assert_eq!(w.visible_tokens(), 3 * HEAD_PAGE_TOKENS + 7);
        assert_eq!(w.write_position(), 3 * HEAD_PAGE_TOKENS + 7);

        w.advance(HEAD_PAGE_TOKENS);
        assert_eq!(w.compact_pages(), vec![0, 4, 5, 6]);
        assert!(!w.contains(3));
    }

    #[test]
    fn physical_slots_stay_put_and_do_not_collide() {
        let mut w = LegacyHeadWindow::new(4).unwrap();
        w.set_frontier(9 * HEAD_PAGE_TOKENS);
        let before = w.compact_slots();
        assert_eq!(before, vec![0, 5, 1, 2, 3]);
        w.advance(HEAD_PAGE_TOKENS);
        let after = w.compact_slots();
        assert_eq!(after, vec![0, 1, 2, 3, 4]);
        // The pages that stayed in both windows kept the same slot.
        for page in [6, 7, 8, 9] {
            assert_eq!(w.physical_slot(page), 1 + (page - 1) % 5);
        }
        // The draft's next page uses the spare slot, not a retained one.
        assert_eq!(w.physical_slot(10), 5);
        assert!(!w.contains(10));
        // All compact slots are distinct (no aliasing in the arena).
        let mut slots = after.clone();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), after.len());
        assert_eq!(w.capacity_tokens(), (4 + 2) * HEAD_PAGE_TOKENS);
    }

    #[test]
    fn draft_boundary_uses_the_next_global_page_spare() {
        let mut w = LegacyHeadWindow::new(32).unwrap();
        w.set_frontier(36 * HEAD_PAGE_TOKENS);
        assert_eq!(w.visible_tokens(), 33 * HEAD_PAGE_TOKENS);
        // The compact coordinate still names retained page 33, but the next
        // global draft row belongs to page 36 and needs the spare slot.
        assert!(w.contains(33));
        assert!(!w.contains(36));
        let (seq_len, slots) = w.draft_layout(0).unwrap();
        assert_eq!(seq_len, 33 * HEAD_PAGE_TOKENS + 1);
        assert_eq!(slots.len(), 34);
        assert_eq!(seq_len.div_ceil(HEAD_PAGE_TOKENS) as usize, slots.len());
        assert_eq!(*slots.last().unwrap(), w.physical_slot(36));
        let mut unique = slots;
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 34);

        // Accepting that row slides the frame and maps the same global page at
        // the new compact write position.
        w.advance(1);
        assert!(w.contains(36));
        assert_eq!(w.compact_index(36), Some(w.visible_tokens() - 1));
    }

    #[test]
    fn before_the_first_page_only_the_sink_is_visible() {
        let mut w = LegacyHeadWindow::new(2).unwrap();
        assert_eq!(w.visible_tokens(), 0);
        assert_eq!(w.compact_pages(), vec![0]);
        w.set_frontier(1);
        assert_eq!(w.compact_pages(), vec![0]);
        assert_eq!(w.visible_tokens(), 1);
        assert_eq!(w.write_position(), 1);
    }

    #[test]
    fn zero_window_is_refused() {
        assert!(LegacyHeadWindow::new(0).is_err());
    }
}
