// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only Legacy Query transaction. Native checkpoint images, accepted
//! prefill provenance, history selection and GPU restore belong to the caller.
//!
//! `source` is a frontend-proven `(begin, token_count)` in the fully rendered
//! prompt, never a tool continuation or a previous step's Q. Core query_policy
//! owns suffix bounding and replay admission. Rows are represented post-norm,
//! pre-RoPE BF16 values converted to FP32 by the taps, not re-normalized here.

use hipfire_kvmem::{bounded_query, query_replay_plan, LegacyReplayPlan, QueryRange};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegacyQueryPhase {
    AwaitingCheckpoint,
    Probing,
    Replaying,
    Complete,
    Bypassed,
}

/// One request's bounded Query and single-use replay transaction.
///
/// Fields are read-only through getters so callers cannot bypass transitions.
/// Layer ordinals must match the statistics store's FullAttention layers. The
/// storage is `[layer][query token][Q channel]`, not prompt- or context-sized.
#[derive(Debug)]
pub struct LegacyQuery {
    query: QueryRange,
    restore_begin: usize,
    prompt_end: usize,
    phase: LegacyQueryPhase,
    initial_frontier: usize,
    block_tokens: u32,
    replay_limit: u32,
    n_layers: usize,
    n_score_layers: usize,
    q_width: usize,
    query_len: usize,
    layer_stride: usize,
    rows: Vec<f32>,
    row_counts: Vec<u32>,
    probe_finished: bool,
    issued_plan: Option<LegacyReplayPlan>,
    bypass_reason: Option<String>,
}

impl LegacyQuery {
    /// Construct only for new user Q. Positive limits are explicit (normally
    /// Q=512, L=2048); zero is rejected, never treated as unlimited.
    ///
    /// Validate the proven span rather than silently clamping a malformed
    /// frontend range. A bounded Query crossing an already accepted frontier
    /// needs a checkpoint from the past, which this transaction does not own.
    pub fn new(
        source: Option<(u64, u64)>,
        frontier: usize,
        prompt_end: usize,
        query_limit: u32,
        replay_limit: u32,
        block_tokens: usize,
        n_layers: usize,
        q_width: usize,
    ) -> Result<Option<Self>, String> {
        Self::new_with_score_layers(
            source,
            frontier,
            prompt_end,
            query_limit,
            replay_limit,
            block_tokens,
            n_layers,
            n_layers,
            q_width,
        )
    }

    /// Composite KVMem records can append an MTP-head ordinal that has no Q
    /// probe. Allocate its zeroed scoring row/count so core geometry matches,
    /// while completeness still requires only the trunk's active layers.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_score_layers(
        source: Option<(u64, u64)>,
        frontier: usize,
        prompt_end: usize,
        query_limit: u32,
        replay_limit: u32,
        block_tokens: usize,
        n_layers: usize,
        n_score_layers: usize,
        q_width: usize,
    ) -> Result<Option<Self>, String> {
        if query_limit == 0 || replay_limit == 0 {
            return Err("legacy query: query and replay limits must be positive".into());
        }
        let block_tokens = u32::try_from(block_tokens)
            .ok()
            .filter(|&block| block != 0)
            .ok_or("legacy query: block_tokens must be a positive u32")?;
        if n_layers == 0 || q_width == 0 {
            return Err("legacy query: n_layers and q_width must be positive".into());
        }
        if n_score_layers < n_layers {
            return Err("legacy query: scoring layers are smaller than active layers".into());
        }
        if frontier > prompt_end {
            return Err(format!(
                "legacy query: frontier {frontier} exceeds prompt_end {prompt_end}"
            ));
        }
        let total =
            u64::try_from(prompt_end).map_err(|_| "legacy query: prompt_end exceeds u64")?;
        let accepted = u64::try_from(frontier).map_err(|_| "legacy query: frontier exceeds u64")?;
        let Some((begin, count)) = source else {
            return Ok(None);
        };
        let end = begin
            .checked_add(count)
            .ok_or("legacy query: source begin + token_count overflow")?;
        if begin > total || end > total {
            return Err(format!(
                "legacy query: source [{begin},{end}) exceeds prompt_end {total}"
            ));
        }
        let query = bounded_query(begin, count, total, query_limit)
            .map_err(|e| format!("legacy query: bounded_query: {e}"))?;
        if query.begin >= query.end || query.end <= accepted {
            return Ok(None);
        }
        if query.begin < accepted {
            return Err(format!(
                "legacy query: Query [{},{}) crosses accepted frontier {frontier}; \
                 missing pre-Query checkpoint for already accepted rows",
                query.begin, query.end
            ));
        }
        let query_begin =
            usize::try_from(query.begin).map_err(|_| "legacy query: Query begin exceeds usize")?;
        let query_len = usize::try_from(query.end - query.begin)
            .map_err(|_| "legacy query: Query length exceeds usize")?;
        let block = block_tokens as usize;
        let aligned = (query_begin / block)
            .checked_mul(block)
            .ok_or("legacy query: checkpoint alignment overflow")?;
        let restore_begin = frontier.max(aligned);
        if restore_begin > query_begin {
            return Err("legacy query: restore boundary is after Query begin".into());
        }
        let layer_stride = query_len
            .checked_mul(q_width)
            .ok_or("legacy query: Query length * q_width overflow")?;
        let elements = n_score_layers
            .checked_mul(layer_stride)
            .ok_or("legacy query: scoring layers * Query layer stride overflow")?;
        // Validate both layouts before allocating either buffer. No allocation
        // depends on C, pre-Query history length, or the configured Q maximum
        // when the actual user Query is shorter.
        check_layout::<f32>(elements, "Query rows")?;
        check_layout::<u32>(n_score_layers, "Query row counts")?;
        Ok(Some(Self {
            query,
            restore_begin,
            prompt_end,
            phase: LegacyQueryPhase::AwaitingCheckpoint,
            initial_frontier: frontier,
            block_tokens,
            replay_limit,
            n_layers,
            n_score_layers,
            q_width,
            query_len,
            layer_stride,
            rows: zeroed(elements, "Query rows")?,
            row_counts: zeroed(n_score_layers, "Query row counts")?,
            probe_finished: false,
            issued_plan: None,
            bypass_reason: None,
        }))
    }

    pub fn query(&self) -> QueryRange {
        self.query
    }

    pub fn restore_begin(&self) -> usize {
        self.restore_begin
    }

    pub fn prompt_end(&self) -> usize {
        self.prompt_end
    }

    pub fn phase(&self) -> LegacyQueryPhase {
        self.phase
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn q_width(&self) -> usize {
        self.q_width
    }

    pub fn query_len(&self) -> usize {
        self.query_len
    }

    /// Per-request replay span limit (`query_replay_max_tokens`). Zero is
    /// impossible here: construction rejects it.
    pub fn replay_limit(&self) -> u32 {
        self.replay_limit
    }

    pub fn bypass_reason(&self) -> Option<&str> {
        self.bypass_reason.as_deref()
    }

    pub fn checkpoint_due(&self, position: usize) -> bool {
        self.phase == LegacyQueryPhase::AwaitingCheckpoint && position == self.restore_begin
    }

    /// Cut only a pre-Query step that would cross the restore boundary.
    /// Check `checkpoint_due` BEFORE dispatch; at the boundary this returns
    /// `want`, not a zero-length step waiting for an implicit state change.
    /// It never turns a positive `want` into zero, even at invalid positions.
    /// Zero `want` means no work. The caller still validates native positions
    /// and caps normal prefill to the remaining prompt (including after bypass).
    pub fn cap_prefill(&self, position: usize, want: usize) -> usize {
        if self.phase == LegacyQueryPhase::AwaitingCheckpoint
            && position >= self.initial_frontier
            && position < self.restore_begin
        {
            want.min(self.restore_begin - position)
        } else {
            want
        }
    }

    /// Call only after the complete native image at `restore_begin` was
    /// captured successfully, before evaluating any Query row.
    pub fn checkpoint_captured(&mut self) -> Result<(), String> {
        self.require_phase(LegacyQueryPhase::AwaitingCheckpoint, "capture checkpoint")?;
        self.phase = LegacyQueryPhase::Probing;
        Ok(())
    }

    /// Abandon recall while the first-pass native state is still intact.
    /// Once replay is armed, restore may have destroyed that state: failure
    /// must be reported by the native owner, never relabeled as safe fallback.
    pub fn bypass(&mut self, reason: &str) -> Result<(), String> {
        if !matches!(
            self.phase,
            LegacyQueryPhase::AwaitingCheckpoint | LegacyQueryPhase::Probing
        ) {
            return Err(format!(
                "legacy query: cannot bypass {:?}; first-pass fallback is no longer available",
                self.phase
            ));
        }
        if reason.trim().is_empty() {
            return Err("legacy query: bypass requires a diagnostic reason".into());
        }
        self.bypass_reason = Some(reason.to_owned());
        self.rows = Vec::new();
        self.row_counts = Vec::new();
        self.issued_plan = None;
        self.probe_finished = false;
        self.phase = LegacyQueryPhase::Bypassed;
        Ok(())
    }

    /// Return `(batch_offset, row_count)` for an accepted prefill batch's
    /// intersection with the real bounded user Query. Malformed/overflowing
    /// batches, batches outside this probe and all replay batches are ignored.
    pub fn capture_span(&self, begin: usize, rows: usize) -> Option<(usize, usize)> {
        if self.phase != LegacyQueryPhase::Probing || rows == 0 || begin < self.restore_begin {
            return None;
        }
        let end = begin.checked_add(rows)?;
        if end > self.prompt_end {
            return None;
        }
        let first = begin.max(self.query.begin as usize);
        let last = end.min(self.query.end as usize);
        (first < last).then(|| (first - begin, last - first))
    }

    /// Append consecutive accepted prefill Q rows for one layer. `begin` is
    /// absolute, not a batch offset. Only pass the intersection selected by
    /// `capture_span`, in token order. Draft/verify rows are NOT valid input;
    /// their acceptance provenance cannot be inferred from a float slice.
    /// Every validation precedes mutation, including finite-value checks.
    pub fn record_layer_rows(
        &mut self,
        layer: usize,
        begin: usize,
        values: &[f32],
    ) -> Result<(), String> {
        self.require_phase(LegacyQueryPhase::Probing, "record Query rows")?;
        if layer >= self.n_layers {
            return Err(format!(
                "legacy query: layer {layer} outside {} layers",
                self.n_layers
            ));
        }
        if values.is_empty() || values.len() % self.q_width != 0 {
            return Err(format!(
                "legacy query: layer {layer} values length {} is not positive whole Q rows of width {}",
                values.len(), self.q_width
            ));
        }
        let expected = self.query.begin as usize + self.row_counts[layer] as usize;
        if begin != expected {
            return Err(format!(
                "legacy query: layer {layer} expected token {expected}, got {begin} (duplicate or noncontiguous rows)"
            ));
        }
        let count = values.len() / self.q_width;
        let end = begin
            .checked_add(count)
            .ok_or("legacy query: captured token end overflow")?;
        if end > self.query.end as usize {
            return Err(format!(
                "legacy query: layer {layer} capture [{begin},{end}) exceeds Query end {}",
                self.query.end
            ));
        }
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "legacy query: nonfinite Q in layer {layer}, token {}, channel {}",
                begin + index / self.q_width,
                index % self.q_width
            ));
        }
        // Products are bounded by the checked allocation layout in new().
        let offset = layer * self.layer_stride + self.row_counts[layer] as usize * self.q_width;
        self.rows[offset..offset + values.len()].copy_from_slice(values);
        self.row_counts[layer] = (end - self.query.begin as usize) as u32;
        Ok(())
    }

    /// Observe the accepted native frontier. A successful observation latches
    /// the full first-pass proof for `replay_plan`/`start_replay`; recording the
    /// final Q row alone is not enough (template suffixes must also finish).
    /// False leaves state untouched. The native owner must not rewind the
    /// first pass between this proof and `start_replay`.
    pub fn probe_complete(&mut self, frontier: usize) -> bool {
        if self.phase != LegacyQueryPhase::Probing
            || frontier != self.prompt_end
            || !self.rows_complete()
        {
            return false;
        }
        self.probe_finished = true;
        true
    }

    /// Layer-major bounded storage. Consult counts before reading an incomplete
    /// probe; unaccepted slots are zero-initialized, not valid Query samples.
    pub fn query_rows(&self) -> &[f32] {
        &self.rows
    }

    pub fn query_row_counts(&self) -> &[u32] {
        &self.row_counts
    }

    /// `[layer][Q channel]` FP32 sums for core MeanKeyQueryView. Each channel
    /// accumulates represented BF16->FP32 rows in increasing token order, just
    /// like NInfer accumulate_query_sum. Never reduce chunk sums or divide by
    /// chunk count: core uses the actual token counts and owns averaging/GQA.
    /// Rows remain frozen and readable through replay and successful completion.
    pub fn query_sums(&self) -> Result<Vec<f32>, String> {
        if !self.rows_complete() {
            return Err("legacy query: cannot sum incomplete or bypassed Query rows".into());
        }
        let mut sums: Vec<f32> = zeroed(self.n_score_layers * self.q_width, "Query sums")?;
        for (layer, values) in self.rows.chunks_exact(self.layer_stride).enumerate() {
            let sum = &mut sums[layer * self.q_width..(layer + 1) * self.q_width];
            for row in values.chunks_exact(self.q_width) {
                for (channel, (total, value)) in sum.iter_mut().zip(row).enumerate() {
                    *total += *value;
                    if !total.is_finite() {
                        return Err(format!(
                            "legacy query: nonfinite FP32 Query sum in layer {layer}, channel {channel}"
                        ));
                    }
                }
            }
        }
        Ok(sums)
    }

    /// Issue a core plan for the ENTIRE `[restore_begin, prompt_end)` replay.
    /// `history_pages` is the selected pre-Query history, including its partial
    /// page, and `workspace` is the native draft workspace in tokens.
    /// Refusal is `Ok(plan)` with the exact core reason and no phase change.
    /// Only the latest issued plan can arm replay; the caller must keep its
    /// selected history/capacity/workspace unchanged until native restore.
    pub fn replay_plan(
        &mut self,
        history_pages: u64,
        device_pages: u64,
        workspace: u32,
    ) -> Result<LegacyReplayPlan, String> {
        self.require_complete_probe("plan replay")?;
        let plan = query_replay_plan(
            self.query,
            self.restore_begin as u64,
            self.prompt_end as u64,
            self.replay_limit,
            self.block_tokens,
            history_pages,
            device_pages,
            workspace,
        )
        .map_err(|e| format!("legacy query: query_replay_plan: {e}"))?;
        self.issued_plan = Some(plan.clone());
        Ok(plan)
    }

    /// Arm exactly one admitted replay BEFORE any destructive native restore.
    /// A refused, edited, foreign or superseded plan leaves the probe intact.
    pub fn start_replay(&mut self, plan: &LegacyReplayPlan) -> Result<(), String> {
        self.require_complete_probe("start replay")?;
        if !plan.admitted {
            return Err(format!(
                "legacy query: replay not admitted: {}",
                plan.reason
            ));
        }
        if plan.begin != self.restore_begin as u64 || plan.end != self.prompt_end as u64 {
            return Err(format!(
                "legacy query: replay plan [{},{}) does not match [{},{})",
                plan.begin, plan.end, self.restore_begin, self.prompt_end
            ));
        }
        if self.issued_plan.as_ref() != Some(plan) {
            return Err(
                "legacy query: replay plan was not the latest plan issued by this probe".into(),
            );
        }
        self.phase = LegacyQueryPhase::Replaying;
        Ok(())
    }

    /// Publish completion only once, after the entire causal replay has been
    /// accepted and native state/logits are usable at exactly `prompt_end`.
    pub fn finish_replay(&mut self, frontier: usize) -> Result<(), String> {
        self.require_phase(LegacyQueryPhase::Replaying, "finish replay")?;
        if frontier != self.prompt_end {
            return Err(format!(
                "legacy query: replay frontier {frontier} does not match prompt_end {}",
                self.prompt_end
            ));
        }
        self.phase = LegacyQueryPhase::Complete;
        Ok(())
    }

    fn rows_complete(&self) -> bool {
        self.row_counts.len() == self.n_score_layers
            && self.row_counts[..self.n_layers]
                .iter()
                .all(|&n| n as usize == self.query_len)
    }

    fn require_phase(&self, phase: LegacyQueryPhase, action: &str) -> Result<(), String> {
        if self.phase == phase {
            Ok(())
        } else {
            Err(format!(
                "legacy query: cannot {action} in {:?}; expected {phase:?}",
                self.phase
            ))
        }
    }

    fn require_complete_probe(&self, action: &str) -> Result<(), String> {
        self.require_phase(LegacyQueryPhase::Probing, action)?;
        if !self.probe_finished || !self.rows_complete() {
            return Err(format!(
                "legacy query: cannot {action} before complete first-pass proof"
            ));
        }
        Ok(())
    }
}

fn check_layout<T>(len: usize, label: &str) -> Result<(), String> {
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| format!("legacy query: {label} byte size overflow"))?;
    if bytes > isize::MAX as usize {
        return Err(format!(
            "legacy query: {label} exceeds maximum allocation size"
        ));
    }
    Ok(())
}

fn zeroed<T: Default + Clone>(len: usize, label: &str) -> Result<Vec<T>, String> {
    check_layout::<T>(len, label)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| format!("legacy query: cannot allocate {label}: {e}"))?;
    values.resize(len, T::default());
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> LegacyQuery {
        LegacyQuery::new(Some((10, 5)), 0, 20, 512, 2048, 8, 2, 2)
            .unwrap()
            .unwrap()
    }

    fn probing() -> LegacyQuery {
        let mut query = small();
        query.checkpoint_captured().unwrap();
        query
    }

    fn ready() -> LegacyQuery {
        let mut query = probing();
        query.record_layer_rows(0, 10, &[1.0; 10]).unwrap();
        query.record_layer_rows(1, 10, &[2.0; 10]).unwrap();
        assert!(query.probe_complete(20));
        query
    }

    #[test]
    fn composite_score_layer_is_zero_padded_without_blocking_probe() {
        let mut query =
            LegacyQuery::new_with_score_layers(Some((10, 5)), 0, 20, 512, 2048, 8, 2, 3, 2)
                .unwrap()
                .unwrap();
        query.checkpoint_captured().unwrap();
        query.record_layer_rows(0, 10, &[1.0; 10]).unwrap();
        query.record_layer_rows(1, 10, &[2.0; 10]).unwrap();
        assert!(query.probe_complete(20));
        assert_eq!(query.query_row_counts(), &[5, 5, 0]);
        assert_eq!(query.query_rows().len(), 3 * 5 * 2);
        let sums = query.query_sums().unwrap();
        assert_eq!(sums.len(), 3 * 2);
        assert_eq!(&sums[4..], &[0.0, 0.0]);
    }

    #[test]
    fn query_511_512_513_uses_core_bounded_user_suffix() {
        for count in [511_u64, 512, 513] {
            let end = 1000 + count;
            let query = LegacyQuery::new(
                Some((1000, count)),
                0,
                end as usize + 7,
                512,
                2048,
                64,
                2,
                3,
            )
            .unwrap()
            .unwrap();
            let len = count.min(512);
            assert_eq!(
                query.query(),
                QueryRange {
                    begin: end - len,
                    end
                }
            );
            assert_eq!(query.query_len(), len as usize);
            assert_eq!(query.query_rows().len(), 2 * len as usize * 3);
            assert_eq!(query.query_row_counts(), &[0, 0]);
        }
    }

    #[test]
    fn none_empty_and_already_accepted_user_span_do_not_create_query() {
        for source in [None, Some((20, 0)), Some((10, 10))] {
            assert!(LegacyQuery::new(source, 20, 30, 512, 2048, 8, 2, 2)
                .unwrap()
                .is_none());
        }
        // No previous Q or taps can enter the tool-continuation constructor.
        assert!(LegacyQuery::new(None, 0, 1000, 512, 2048, 64, 1, 1)
            .unwrap()
            .is_none());
    }

    #[test]
    fn checkpoint_page_alignment_and_partial_native_frontier() {
        let query = LegacyQuery::new(Some((150, 10)), 0, 170, 512, 2048, 64, 1, 1)
            .unwrap()
            .unwrap();
        assert_eq!(query.restore_begin(), 128);
        assert!(!query.checkpoint_due(127));
        assert!(query.checkpoint_due(128));
        assert!(!query.checkpoint_due(129));
        assert_eq!(query.cap_prefill(0, 256), 128);
        assert_eq!(query.cap_prefill(100, 64), 28);
        assert_eq!(query.cap_prefill(100, 1), 1);

        let mut partial = LegacyQuery::new(Some((150, 10)), 140, 170, 512, 2048, 64, 1, 1)
            .unwrap()
            .unwrap();
        assert_eq!(partial.restore_begin(), 140);
        assert!(partial.checkpoint_due(140));
        // The owner must capture at 140; capping itself must not spin at 0.
        for position in [0, 139, 140, 150, 170, usize::MAX] {
            assert_eq!(partial.cap_prefill(position, 16), 16);
            assert_eq!(partial.cap_prefill(position, 0), 0);
        }
        partial.checkpoint_captured().unwrap();
        assert!(!partial.checkpoint_due(140));
        assert_eq!(partial.cap_prefill(140, 16), 16);
    }

    #[test]
    fn partial_query_across_accepted_frontier_fails_without_past_checkpoint() {
        let error = LegacyQuery::new(Some((100, 100)), 150, 220, 512, 2048, 64, 1, 1).unwrap_err();
        assert!(error.contains("missing pre-Query checkpoint"));
        // Truncating to a suffix entirely ahead of the frontier needs no past Q.
        let query = LegacyQuery::new(Some((100, 100)), 150, 220, 10, 2048, 64, 1, 1)
            .unwrap()
            .unwrap();
        assert_eq!(query.query().begin, 190);
        assert_eq!(query.restore_begin(), 150);
    }

    #[test]
    fn invalid_options_spans_and_layout_overflow_fail_before_allocation() {
        for (q, l, block, layers, width) in [
            (0, 2048, 8, 1, 1),
            (512, 0, 8, 1, 1),
            (512, 2048, 0, 1, 1),
            (512, 2048, 8, 0, 1),
            (512, 2048, 8, 1, 0),
            (512, 2048, 8, 1, usize::MAX),
            (512, 2048, 8, usize::MAX, 2),
        ] {
            assert!(LegacyQuery::new(Some((0, 2)), 0, 2, q, l, block, layers, width).is_err());
        }
        for source in [Some((21, 0)), Some((19, 2)), Some((u64::MAX, 1))] {
            assert!(LegacyQuery::new(source, 0, 20, 512, 2048, 8, 1, 1).is_err());
        }
        assert!(LegacyQuery::new(None, 21, 20, 512, 2048, 8, 1, 1).is_err());
        assert!(LegacyQuery::new(Some((0, 1)), 0, 1, 512, 2048, 8, 1, usize::MAX / 2).is_err());
        assert!(LegacyQuery::new(
            Some((0, 1)),
            0,
            1,
            512,
            2048,
            8,
            1,
            isize::MAX as usize / 4 + 1
        )
        .is_err());
    }

    #[test]
    fn large_prompt_and_limit_allocate_only_actual_query_rows() {
        let query = LegacyQuery::new(Some((1_000_000, 1)), 0, 1_000_010, u32::MAX, 2048, 64, 2, 3)
            .unwrap()
            .unwrap();
        assert_eq!(query.query_rows().len(), 6);
        assert_eq!(query.query_len(), 1);
    }

    #[test]
    fn query_limit_one_captures_actual_user_not_last_step_or_template() {
        let mut query = LegacyQuery::new(Some((10, 5)), 0, 30, 1, 2048, 8, 1, 1)
            .unwrap()
            .unwrap();
        assert_eq!(query.query(), QueryRange { begin: 14, end: 15 });
        assert_eq!(query.capture_span(8, 16), None);
        query.checkpoint_captured().unwrap();
        assert_eq!(query.capture_span(8, 16), Some((6, 1)));
        assert_eq!(query.capture_span(24, 6), None);
        query.record_layer_rows(0, 14, &[1.0078125]).unwrap();
        assert!(!query.probe_complete(15));
        assert!(query.replay_plan(0, 8, 0).is_err());
        assert!(query.probe_complete(30));
        assert_eq!(query.query_sums().unwrap(), &[1.0078125]);
    }

    #[test]
    fn multiple_chunks_and_layers_require_full_real_token_coverage() {
        let mut query = probing();
        assert_eq!(query.capture_span(8, 4), Some((2, 2)));
        query
            .record_layer_rows(0, 10, &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        query
            .record_layer_rows(1, 10, &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        assert_eq!(query.query_row_counts(), &[2, 2]);
        assert!(!query.probe_complete(20));
        assert!(query.query_sums().is_err());
        assert_eq!(query.capture_span(12, 8), Some((0, 3)));
        query
            .record_layer_rows(1, 12, &[50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
            .unwrap();
        assert_eq!(query.query_row_counts(), &[2, 5]);
        assert!(!query.probe_complete(20));
        query
            .record_layer_rows(0, 12, &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            .unwrap();
        assert!(!query.probe_complete(19));
        assert!(!query.probe_complete(21));
        assert!(query.probe_complete(20));
        assert_eq!(query.query_row_counts(), &[5, 5]);
        assert_eq!(
            query.query_rows(),
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                60.0, 70.0, 80.0, 90.0, 100.0,
            ]
        );
        assert_eq!(query.query_sums().unwrap(), &[25.0, 30.0, 250.0, 300.0]);
    }

    #[test]
    fn malformed_capture_does_not_mutate_rows_counts_or_phase() {
        let mut query = probing();
        query.record_layer_rows(0, 10, &[1.0, 2.0]).unwrap();
        let rows = query.query_rows().to_vec();
        let counts = query.query_row_counts().to_vec();
        for (layer, begin, values) in [
            (0, 10, vec![1.0, 2.0]), // duplicate
            (0, 12, vec![1.0, 2.0]), // missing row
            (0, 9, vec![1.0, 2.0]),  // before Query
            (2, 10, vec![1.0, 2.0]), // invalid layer
            (0, 11, vec![1.0]),      // partial Q row
            (0, 11, vec![]),
            (0, 11, vec![0.0; 10]), // beyond Query
            (0, 11, vec![f32::NAN, 0.0]),
            (0, 11, vec![0.0, f32::INFINITY]),
            (0, 11, vec![0.0, f32::NEG_INFINITY]),
            (0, usize::MAX, vec![1.0, 2.0]),
        ] {
            assert!(query.record_layer_rows(layer, begin, &values).is_err());
            assert_eq!(query.query_rows(), rows.as_slice());
            assert_eq!(query.query_row_counts(), counts.as_slice());
            assert_eq!(query.phase(), LegacyQueryPhase::Probing);
        }
        for (begin, rows) in [(8, 0), (0, 12), (8, 13), (usize::MAX, 2), (15, 5)] {
            assert_eq!(query.capture_span(begin, rows), None);
        }
    }

    #[test]
    fn mean_sums_are_chronological_fp32_not_chunk_sums_or_means() {
        // All inputs are exactly BF16-representable. FP32 left-to-right sum of
        // the first channel is 1, not the FP64 result 2; chunk grouping must not
        // change that chronological rounding boundary.
        let values = [16_777_216.0, 0.5, 1.0, 0.25, -16_777_216.0, -0.5, 1.0, 0.75];
        let mut expected = None;
        for split in [1, 2, 3, 4] {
            let mut query = LegacyQuery::new(Some((0, 4)), 0, 4, 512, 2048, 8, 1, 2)
                .unwrap()
                .unwrap();
            query.checkpoint_captured().unwrap();
            query.record_layer_rows(0, 0, &values[..split * 2]).unwrap();
            if split < 4 {
                query
                    .record_layer_rows(0, split, &values[split * 2..])
                    .unwrap();
            }
            assert!(query.probe_complete(4));
            let sums = query.query_sums().unwrap();
            assert_eq!(sums, &[1.0, 1.0]);
            let bits: Vec<_> = sums.iter().map(|v| v.to_bits()).collect();
            if let Some(ref expected) = expected {
                assert_eq!(&bits, expected);
            } else {
                expected = Some(bits);
            }
        }
    }

    #[test]
    fn nonfinite_fp32_reduction_is_an_error_without_mutation() {
        let mut query = LegacyQuery::new(Some((0, 2)), 0, 2, 512, 2048, 8, 1, 1)
            .unwrap()
            .unwrap();
        query.checkpoint_captured().unwrap();
        let largest_bf16 = f32::from_bits(0x7f7f_0000);
        query.record_layer_rows(0, 0, &[largest_bf16; 2]).unwrap();
        assert!(query.query_sums().unwrap_err().contains("nonfinite FP32"));
        assert_eq!(query.phase(), LegacyQueryPhase::Probing);
        assert_eq!(query.query_row_counts(), &[2]);
    }

    #[test]
    fn replay_2047_2048_2049_counts_full_causal_span_not_query_text() {
        for length in [2047, 2048, 2049] {
            let end = 128 + length;
            let mut query = LegacyQuery::new(Some((137, 1)), 0, end, 512, 2048, 128, 1, 1)
                .unwrap()
                .unwrap();
            assert_eq!(query.restore_begin(), 128);
            query.checkpoint_captured().unwrap();
            query.record_layer_rows(0, 137, &[1.0]).unwrap();
            assert!(query.probe_complete(end));
            let plan = query.replay_plan(1, 64, 0).unwrap();
            assert_eq!((plan.begin, plan.end), (128, end as u64));
            assert_eq!(plan.admitted, length <= 2048);
            assert_eq!(query.phase(), LegacyQueryPhase::Probing);
            if length > 2048 {
                assert_eq!(plan.reason, "replay_token_limit");
                assert!(query
                    .start_replay(&plan)
                    .unwrap_err()
                    .contains("replay_token_limit"));
                assert_eq!(query.query_sums().unwrap(), &[1.0]);
                query.bypass(&plan.reason).unwrap();
                assert_eq!(query.bypass_reason(), Some("replay_token_limit"));
            } else {
                query.start_replay(&plan).unwrap();
                query.finish_replay(end).unwrap();
            }
        }
    }

    #[test]
    fn capacity_admission_counts_partial_page_once_and_draft_workspace() {
        let mut query = LegacyQuery::new(Some((75, 1)), 70, 129, 512, 2048, 64, 1, 1)
            .unwrap()
            .unwrap();
        assert_eq!(query.restore_begin(), 70);
        query.checkpoint_captured().unwrap();
        query.record_layer_rows(0, 75, &[1.0]).unwrap();
        assert!(query.probe_complete(129));
        let admitted = query.replay_plan(2, 3, 0).unwrap();
        assert!(admitted.admitted);
        let denied = query.replay_plan(2, 3, 64).unwrap();
        assert!(!denied.admitted);
        assert_eq!(denied.reason, "replay_device_capacity");
        assert!(query.start_replay(&admitted).is_err()); // superseded plan
        assert!(query.start_replay(&denied).is_err());
        assert_eq!(query.phase(), LegacyQueryPhase::Probing);
        let denied_history = query.replay_plan(4, 3, 0).unwrap();
        assert_eq!(denied_history.reason, "replay_device_capacity");
        query.bypass(&denied_history.reason).unwrap();
        assert_eq!(query.bypass_reason(), Some("replay_device_capacity"));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn core_workspace_overflow_keeps_exact_refusal_reason() {
        let end = usize::MAX;
        let begin = end - 1;
        let mut query = LegacyQuery::new(Some((begin as u64, 1)), begin, end, 512, 2048, 64, 1, 1)
            .unwrap()
            .unwrap();
        query.checkpoint_captured().unwrap();
        query.record_layer_rows(0, begin, &[1.0]).unwrap();
        assert!(query.probe_complete(end));
        let plan = query.replay_plan(0, u64::MAX, 1).unwrap();
        assert!(!plan.admitted);
        assert_eq!(plan.reason, "invalid_replay_boundary");
        assert_eq!(query.phase(), LegacyQueryPhase::Probing);
        assert!(
            LegacyQuery::new(Some((0, 1)), 0, 1, 512, 2048, u32::MAX as usize + 1, 1, 1).is_err()
        );
    }

    #[test]
    fn replay_requires_observed_complete_probe_and_matching_issued_admission() {
        let mut query = probing();
        let fake = LegacyReplayPlan {
            begin: 8,
            end: 20,
            admitted: true,
            reason: "bounded_replay".into(),
        };
        assert!(query.start_replay(&fake).is_err());
        assert!(query.replay_plan(0, 8, 0).is_err());
        query.record_layer_rows(0, 10, &[1.0; 10]).unwrap();
        query.record_layer_rows(1, 10, &[1.0; 10]).unwrap();
        assert!(query.start_replay(&fake).is_err()); // template suffix not proven
        assert!(query.probe_complete(20));
        assert!(query.start_replay(&fake).is_err()); // no issued plan
        let plan = query.replay_plan(0, 8, 0).unwrap();
        for changed in [
            LegacyReplayPlan {
                begin: 9,
                ..plan.clone()
            },
            LegacyReplayPlan {
                end: 19,
                ..plan.clone()
            },
            LegacyReplayPlan {
                admitted: false,
                ..plan.clone()
            },
            LegacyReplayPlan {
                reason: "edited".into(),
                ..plan.clone()
            },
        ] {
            assert!(query.start_replay(&changed).is_err());
            assert_eq!(query.phase(), LegacyQueryPhase::Probing);
        }
        query.start_replay(&plan).unwrap();
    }

    #[test]
    fn replay_freezes_query_and_completes_exactly_once() {
        let mut query = ready();
        let rows = query.query_rows().to_vec();
        let counts = query.query_row_counts().to_vec();
        let sums = query.query_sums().unwrap();
        let plan = query.replay_plan(0, 8, 0).unwrap();
        query.start_replay(&plan).unwrap();
        assert_eq!(query.capture_span(8, 12), None);
        assert!(query.record_layer_rows(0, 10, &[99.0; 10]).is_err());
        assert!(!query.probe_complete(20));
        assert!(query.checkpoint_captured().is_err());
        assert!(query.start_replay(&plan).is_err());
        assert!(query.replay_plan(0, 8, 0).is_err());
        assert!(query.bypass("restore failed").is_err());
        for frontier in [0, 15, 19, 21] {
            assert!(query.finish_replay(frontier).is_err());
            assert_eq!(query.phase(), LegacyQueryPhase::Replaying);
        }
        assert_eq!(query.query_rows(), rows.as_slice());
        assert_eq!(query.query_row_counts(), counts.as_slice());
        assert_eq!(query.query_sums().unwrap(), sums);
        query.finish_replay(20).unwrap();
        assert_eq!(query.phase(), LegacyQueryPhase::Complete);
        assert!(query.finish_replay(20).is_err());
        assert!(query.start_replay(&plan).is_err());
        assert!(query.checkpoint_captured().is_err());
        assert!(query.bypass("already published").is_err());
        assert!(query.record_layer_rows(0, 10, &[1.0, 2.0]).is_err());
        assert_eq!(query.capture_span(8, 12), None);
        assert!(!query.probe_complete(20));
        assert_eq!(query.query_sums().unwrap(), sums);
        assert_eq!(query.bypass_reason(), None);
    }

    #[test]
    fn bypass_only_before_destructive_restore_releases_q_and_keeps_reason() {
        for mut query in [small(), probing(), ready()] {
            assert!(query.bypass(" ").is_err());
            assert!(!query.query_rows().is_empty());
            query.bypass("checkpoint unavailable").unwrap();
            assert_eq!(query.phase(), LegacyQueryPhase::Bypassed);
            assert_eq!(query.bypass_reason(), Some("checkpoint unavailable"));
            assert!(query.query_rows().is_empty());
            assert!(query.query_row_counts().is_empty());
            assert_eq!(query.rows.capacity(), 0);
            assert_eq!(query.row_counts.capacity(), 0);
            assert!(query.query_sums().is_err());
            assert!(!query.checkpoint_due(8));
            assert_eq!(query.cap_prefill(0, 20), 20);
            assert_eq!(query.capture_span(8, 12), None);
            assert!(!query.probe_complete(20));
            assert!(query.checkpoint_captured().is_err());
            assert!(query.finish_replay(20).is_err());
            assert!(query.record_layer_rows(0, 10, &[1.0, 2.0]).is_err());
            assert!(query.replay_plan(0, 8, 0).is_err());
            assert!(query.bypass("overwrite diagnostic").is_err());
            assert_eq!(query.bypass_reason(), Some("checkpoint unavailable"));
        }
    }

    #[test]
    fn invalid_early_transitions_do_not_skip_checkpoint_or_probe() {
        let mut query = small();
        assert!(query.record_layer_rows(0, 10, &[1.0, 2.0]).is_err());
        assert!(query.finish_replay(20).is_err());
        assert!(!query.probe_complete(20));
        assert!(query.replay_plan(0, 8, 0).is_err());
        assert_eq!(query.phase(), LegacyQueryPhase::AwaitingCheckpoint);
        query.checkpoint_captured().unwrap();
        assert!(query.checkpoint_captured().is_err());
        assert!(query.finish_replay(20).is_err());
        assert_eq!(query.phase(), LegacyQueryPhase::Probing);
    }
}
