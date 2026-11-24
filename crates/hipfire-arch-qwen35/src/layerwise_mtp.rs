// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Layerwise MTP history policy + auxiliary-selection binding primitives
//! (Phase 4.1 ① mtp-foundation).
//!
//! Pure-host, additive foundation for the independent MTP head's draft-owned
//! K/V history selection. Mirrors the core `kvmem/layerwise_mtp.hpp` policy
//! enum/parse/validate and ninfer's `bind_auxiliary_selection` guard
//! (ninfer-layerwise-recall.patch NL:10189-10204) + FirstLayer source rule
//! (NL:9985-9990). NOT wired into live serve in ①: the bound IDs are produced
//! and validated here; ② feeds them into the head's `LayerWorkingSetPlan`
//! (`plan_layer`) and ③ adds the UpstreamGlobal private probe. The default
//! Legacy serve never constructs any of this, so it stays byte-neutral.
//!
//! Two user-approved history alternatives (core layerwise_mtp.hpp:10-19): both
//! leave the target's fresh-Q selection unchanged and restore the head's OWN
//! canonical K/V for the selected original token IDs.
//!   * FirstLayer (ninfer default) — NO private probe; the head history is bound
//!     from the FIRST target FA layer's restored view (cheap reuse of the real
//!     forward's first-layer restore).
//!   * UpstreamGlobal — a SEPARATE private target probe pass before the real
//!     Query (core `LayerwiseProbeSelection` global Mean-K reduction). The probe
//!     subsystem lands in ③; ① only carries the policy + the fail-closed guard
//!     that rejects supplied target IDs.

use crate::kvmem_ctl::RecallStrategy;

/// MTP head history policy. Ordinals mirror the core `kvmem::LayerwiseMtpHistory`
/// (layerwise_mtp.hpp:15-19) exactly so a future shim entry can pass the raw u8
/// without translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerwiseMtpHistory {
    Unspecified = 0,
    UpstreamGlobal = 1,
    FirstLayer = 2,
}

impl LayerwiseMtpHistory {
    /// Core `layerwise_mtp_history_name` (layerwise_mtp.hpp:21-28).
    pub fn name(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::UpstreamGlobal => "upstream-global",
            Self::FirstLayer => "first-layer",
        }
    }

    /// Core `parse_layerwise_mtp_history` (layerwise_mtp.hpp:30-35). The CLI
    /// wrapper trims before calling; the literal spellings are exact.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "upstream-global" => Ok(Self::UpstreamGlobal),
            "first-layer" => Ok(Self::FirstLayer),
            _ => Err("layerwise MTP history must be upstream-global or first-layer".to_string()),
        }
    }

    /// Core `validate_layerwise_mtp_history` (layerwise_mtp.hpp:37-43): a
    /// resolved policy requires Layerwise recall AND native MTP; Layerwise+MTP
    /// with an unresolved policy is an error (the backend must declare a default).
    pub fn validate(self, strategy: RecallStrategy, mtp_enabled: bool) -> Result<(), String> {
        if self != Self::Unspecified && (!strategy.is_layerwise() || !mtp_enabled) {
            return Err(
                "--kvmem-layerwise-mtp-history requires layerwise recall and native MTP"
                    .to_string(),
            );
        }
        if strategy.is_layerwise() && mtp_enabled && self == Self::Unspecified {
            return Err("layerwise MTP policy was not resolved; select --kvmem-layerwise-mtp-history upstream-global|first-layer or use a backend-declared default".to_string());
        }
        Ok(())
    }

    /// ninfer FirstLayer auto-default (NL:201/13468/14019): Layerwise+MTP with an
    /// omitted policy resolves to FirstLayer, then validates. Any other state is
    /// returned unchanged for `validate` to accept/reject. Pure so the resolution
    /// contract is testable without the process-start config snapshot.
    pub fn resolve_default(
        self,
        strategy: RecallStrategy,
        mtp_enabled: bool,
    ) -> Result<Self, String> {
        let resolved = if strategy.is_layerwise() && mtp_enabled && self == Self::Unspecified {
            Self::FirstLayer
        } else {
            self
        };
        resolved.validate(strategy, mtp_enabled)?;
        Ok(resolved)
    }

    /// UpstreamGlobal needs its own private target probe (③); FirstLayer reuses
    /// the real forward's first-layer restore and never probes.
    pub fn requires_private_probe(self) -> bool {
        matches!(self, Self::UpstreamGlobal)
    }
}

/// Pure-host mirror of ninfer `KvmemLayerViews::visible_tokens(pages, frontier)`
/// (NL:12239-12249), with the block size parameterized (hipfire Q8 page = 128
/// tokens vs ninfer's 64). `pages` are canonical block IDs in chronological
/// order; each contributes `min(block_tokens, frontier - id*block_tokens)`
/// visible tokens, so the final block is clipped to the accepted frontier. Fails
/// closed on a non-chronological map (duplicate/decreasing ID) or a block that
/// starts at/beyond the frontier — exactly ninfer's "not a chronological accepted
/// view" guard. This subsumes ninfer's separate is_sorted/adjacent_find checks in
/// `bind_auxiliary_selection` (NL:10193-10194).
pub fn auxiliary_visible_tokens(
    pages: &[u32],
    frontier: u64,
    block_tokens: u64,
) -> Result<u64, String> {
    if block_tokens == 0 {
        return Err("auxiliary visible tokens: block_tokens must be positive".to_string());
    }
    let mut total = 0u64;
    let mut previous = 0u32;
    let mut first = true;
    for &id in pages {
        let begin = id as u64 * block_tokens;
        if (!first && id <= previous) || begin >= frontier {
            return Err("native layer map is not a chronological accepted view".to_string());
        }
        total += (frontier - begin).min(block_tokens);
        previous = id;
        first = false;
    }
    Ok(total)
}

/// Pure-host auxiliary(head)-selection bind primitive. Mirrors the VALIDATION
/// half of ninfer `bind_auxiliary_selection` (NL:10189-10204): the selected
/// canonical IDs must cover exactly the one target historical extent, be
/// chronological/unique, bind once, and require an actual head pool. The live
/// `LayerWorkingSetPlan` construction from the head catalog (the second half of
/// ninfer's bind) is ②'s wiring; ① produces and validates the bound IDs only.
#[derive(Debug, Clone)]
pub struct AuxiliarySelection {
    history_tokens: u64,
    block_tokens: u64,
    has_head_pool: bool,
    bound: Option<Vec<u32>>,
}

impl AuxiliarySelection {
    pub fn new(history_tokens: u64, block_tokens: u64, has_head_pool: bool) -> Self {
        Self {
            history_tokens,
            block_tokens,
            has_head_pool,
            bound: None,
        }
    }

    /// Bind the head history to `selected` canonical block IDs covering
    /// `[0, frontier)`. Fail-closed mirrors of ninfer's guards:
    ///   * no head pool → "native auxiliary selection has no actual auxiliary
    ///     pool" (NL:9963-9964)
    ///   * already bound → bind-once (ninfer `auxiliary_plan` non-null, NL:10191)
    ///   * extent ≠ history_tokens / non-chronological / duplicate → "native MTP
    ///     history binding differs from the one selected target extent"
    ///     (NL:10192-10195)
    pub fn bind(&mut self, selected: &[u32], frontier: u64) -> Result<(), String> {
        if !self.has_head_pool {
            return Err("native auxiliary selection has no actual auxiliary pool".to_string());
        }
        if self.bound.is_some() {
            return Err(
                "native MTP history binding differs from the one selected target extent"
                    .to_string(),
            );
        }
        let visible = auxiliary_visible_tokens(selected, frontier, self.block_tokens)?;
        if visible != self.history_tokens {
            return Err(
                "native MTP history binding differs from the one selected target extent"
                    .to_string(),
            );
        }
        self.bound = Some(selected.to_vec());
        Ok(())
    }

    pub fn is_bound(&self) -> bool {
        self.bound.is_some()
    }

    pub fn selected(&self) -> Option<&[u32]> {
        self.bound.as_deref()
    }

    pub fn history_tokens(&self) -> u64 {
        self.history_tokens
    }
}

/// One target FA layer's restored view, as the aux-binding source. `model_layer`
/// is the core model-layer ordinal; `restored_ids` are the canonical block IDs
/// the real forward restored for that layer (chronological).
#[derive(Debug, Clone)]
pub struct SourceLayerView {
    pub model_layer: u32,
    pub restored_ids: Vec<u32>,
}

/// Policy admission for a supplied auxiliary selection (ninfer NL:9900-9902,
/// 9925-9927). FirstLayer binds from the first target layer's restore and cannot
/// accept a preselected history; UpstreamGlobal requires its own private probe
/// and cannot accept supplied target IDs. An empty preselection is always
/// admitted (the normal path: the source is derived, not supplied).
pub fn admit_auxiliary_source(
    policy: LayerwiseMtpHistory,
    preselected: &[u32],
) -> Result<(), String> {
    if preselected.is_empty() {
        return Ok(());
    }
    match policy {
        LayerwiseMtpHistory::FirstLayer => {
            Err("first-layer MTP cannot accept a preselected auxiliary history".to_string())
        }
        LayerwiseMtpHistory::UpstreamGlobal => Err(
            "upstream-global MTP requires its own private probe, not supplied target IDs"
                .to_string(),
        ),
        LayerwiseMtpHistory::Unspecified => Err(
            "layerwise MTP policy was not resolved; cannot admit an auxiliary selection"
                .to_string(),
        ),
    }
}

/// FirstLayer source rule (ninfer NL:9985-9990): the head history is bound from
/// the FIRST target FA layer's restored block IDs. Returns that first layer's
/// IDs; fails closed if no source layer was restored. The caller binds them via
/// `AuxiliarySelection::bind` only when `model == source_layers.front()` and the
/// policy is FirstLayer (the live restore-hook condition, wired in ②).
pub fn first_layer_auxiliary_ids(source_layers: &[SourceLayerView]) -> Result<&[u32], String> {
    source_layers
        .first()
        .map(|s| s.restored_ids.as_slice())
        .ok_or_else(|| {
            "native Query restored no target layer for first-layer MTP binding".to_string()
        })
}

/// P4.1② G5: the LIVE FirstLayer auxiliary-selection bind — the hipfire mirror of
/// ninfer's restore-hook condition (NL:9985-9990: `model == source_layers.front()`
/// && `backend_kv_pages` && policy FirstLayer → `bind_auxiliary_selection(ids)`,
/// NL:10189-10204). Called from the trunk restore hook with the FIRST target FA
/// ordinal's restored canonical page ids (`restored_keep_set()[0]`, chronological,
/// suffix page excluded) and the accepted `frontier` (the query's
/// `evaluation.begin` == `ctl.fed_tokens(slot)`).
///
/// Under FirstLayer the head history IS the trunk's first-layer restored view, so
/// the target extent `history_tokens = auxiliary_visible_tokens(restored_first,
/// frontier, block_tokens)` (ninfer `auxiliary_tokens = history_tokens`, NL:9911);
/// [`AuxiliarySelection::bind`] then re-validates the ids are chronological/unique
/// and cover exactly that extent, failing closed on a corrupt restore.
/// `has_head_pool = true`: the caller gates on [`KvMemCtl::head_plane_enabled`]
/// (Layerwise + native MTP), which under ② is exactly the FirstLayer policy — ③
/// adds the UpstreamGlobal private-probe branch. Returns the bound selection for
/// [`KvMemCtl::set_aux_selection`]; Legacy / AR-only never reach this (the whole
/// Layerwise orchestration is skipped, byte-neutral).
pub fn bind_first_layer_auxiliary(
    restored_first: &[u64],
    frontier: u64,
    block_tokens: u64,
) -> Result<AuxiliarySelection, String> {
    let selected: Vec<u32> = restored_first
        .iter()
        .map(|&id| {
            u32::try_from(id)
                .map_err(|_| format!("first-layer aux binding: page id {id} overflow u32"))
        })
        .collect::<Result<Vec<u32>, String>>()?;
    let history_tokens = auxiliary_visible_tokens(&selected, frontier, block_tokens)?;
    let mut sel = AuxiliarySelection::new(history_tokens, block_tokens, true);
    sel.bind(&selected, frontier)?;
    Ok(sel)
}

// ── head restored-position + RoPE-shift contract (P4.1② G4) ───────────────
//
// Pure-host mirror of ninfer's native MTP head restore (NL:10285-10296) and
// COW boundary rebuild (NL:5849-5886). The bounded head pool restores a COMPACT
// resident view of the accepted history, so each verify-span row carries TWO
// distinct coordinates:
//   * a COMPACT pool-local KV-cache write position (where the new K/V lands in
//     the bounded head arena), shifted down by the dropped history; and
//   * a GLOBAL RoPE position (where the head attends), kept aligned with the
//     trunk's original frame by an accumulated `rope_delta`.
// ② produces and validates these coordinates here; the GPU boundary-rebuild +
// head-forward wiring consumes them. Legacy never constructs any of this.

/// ninfer `auxiliary_removed_tokens = evaluation.begin - auxiliary_tokens`
/// (NL:10285): the count of accepted head-history tokens dropped from the bounded
/// resident view by the restore. `evaluation_begin` is the target's accepted
/// frontier (the verify span's begin); `auxiliary_tokens` is the head view's
/// restored visible tokens. Fail-closed if the visible view exceeds the frontier
/// (underflow) or the difference overflows `u32` (ninfer casts to `uint32_t`).
pub fn auxiliary_removed_tokens(
    evaluation_begin: u64,
    auxiliary_tokens: u64,
) -> Result<u32, String> {
    let removed = evaluation_begin.checked_sub(auxiliary_tokens).ok_or_else(|| {
        format!(
            "auxiliary removed tokens: visible {auxiliary_tokens} exceeds frontier {evaluation_begin}"
        )
    })?;
    u32::try_from(removed).map_err(|_| format!("auxiliary removed tokens: {removed} overflow u32"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── policy enum: core mirror ──────────────────────────────────────────
    #[test]
    fn policy_ordinals_and_names_mirror_core() {
        assert_eq!(LayerwiseMtpHistory::Unspecified as u8, 0);
        assert_eq!(LayerwiseMtpHistory::UpstreamGlobal as u8, 1);
        assert_eq!(LayerwiseMtpHistory::FirstLayer as u8, 2);
        assert_eq!(LayerwiseMtpHistory::Unspecified.name(), "unspecified");
        assert_eq!(
            LayerwiseMtpHistory::UpstreamGlobal.name(),
            "upstream-global"
        );
        assert_eq!(LayerwiseMtpHistory::FirstLayer.name(), "first-layer");
    }

    #[test]
    fn policy_parse_exact_spellings() {
        assert_eq!(
            LayerwiseMtpHistory::parse("first-layer").unwrap(),
            LayerwiseMtpHistory::FirstLayer
        );
        assert_eq!(
            LayerwiseMtpHistory::parse("upstream-global").unwrap(),
            LayerwiseMtpHistory::UpstreamGlobal
        );
        // Core parse is exact: no case-fold, no alias, no empty.
        assert!(LayerwiseMtpHistory::parse("FirstLayer").is_err());
        assert!(LayerwiseMtpHistory::parse("first_layer").is_err());
        assert!(LayerwiseMtpHistory::parse("").is_err());
        assert!(LayerwiseMtpHistory::parse("legacy").is_err());
    }

    #[test]
    fn policy_validate_requires_layerwise_and_mtp() {
        use RecallStrategy::{Layerwise, Legacy};
        // A resolved policy needs BOTH layerwise recall and native MTP.
        assert!(LayerwiseMtpHistory::FirstLayer
            .validate(Legacy, true)
            .is_err());
        assert!(LayerwiseMtpHistory::FirstLayer
            .validate(Layerwise, false)
            .is_err());
        assert!(LayerwiseMtpHistory::FirstLayer
            .validate(Layerwise, true)
            .is_ok());
        assert!(LayerwiseMtpHistory::UpstreamGlobal
            .validate(Layerwise, true)
            .is_ok());
        // Unspecified is fine off the layerwise+MTP path...
        assert!(LayerwiseMtpHistory::Unspecified
            .validate(Legacy, false)
            .is_ok());
        assert!(LayerwiseMtpHistory::Unspecified
            .validate(Legacy, true)
            .is_ok());
        // ...but unresolved ON layerwise+MTP is an error (must declare a default).
        assert!(LayerwiseMtpHistory::Unspecified
            .validate(Layerwise, true)
            .is_err());
    }

    #[test]
    fn policy_resolve_default_first_layer() {
        use RecallStrategy::{Layerwise, Legacy};
        // ninfer auto-default: layerwise+MTP+omitted → FirstLayer.
        assert_eq!(
            LayerwiseMtpHistory::Unspecified
                .resolve_default(Layerwise, true)
                .unwrap(),
            LayerwiseMtpHistory::FirstLayer
        );
        // An explicit policy is preserved.
        assert_eq!(
            LayerwiseMtpHistory::UpstreamGlobal
                .resolve_default(Layerwise, true)
                .unwrap(),
            LayerwiseMtpHistory::UpstreamGlobal
        );
        assert_eq!(
            LayerwiseMtpHistory::FirstLayer
                .resolve_default(Layerwise, true)
                .unwrap(),
            LayerwiseMtpHistory::FirstLayer
        );
        // Off the layerwise+MTP path an omitted policy stays omitted (Legacy neutral).
        assert_eq!(
            LayerwiseMtpHistory::Unspecified
                .resolve_default(Legacy, false)
                .unwrap(),
            LayerwiseMtpHistory::Unspecified
        );
        assert_eq!(
            LayerwiseMtpHistory::Unspecified
                .resolve_default(Legacy, true)
                .unwrap(),
            LayerwiseMtpHistory::Unspecified
        );
        // A resolved policy off-path is still rejected by validate.
        assert!(LayerwiseMtpHistory::FirstLayer
            .resolve_default(Legacy, true)
            .is_err());
    }

    #[test]
    fn policy_requires_private_probe() {
        assert!(LayerwiseMtpHistory::UpstreamGlobal.requires_private_probe());
        assert!(!LayerwiseMtpHistory::FirstLayer.requires_private_probe());
        assert!(!LayerwiseMtpHistory::Unspecified.requires_private_probe());
    }

    // ── extent helper: ninfer visible_tokens mirror ───────────────────────
    #[test]
    fn visible_tokens_clips_last_block() {
        // 300 tokens over 128-token blocks: 128 + 128 + 44.
        assert_eq!(auxiliary_visible_tokens(&[0, 1, 2], 300, 128).unwrap(), 300);
        // Exact full blocks.
        assert_eq!(auxiliary_visible_tokens(&[0, 1], 256, 128).unwrap(), 256);
        // ninfer's native 64-token block geometry also works.
        assert_eq!(auxiliary_visible_tokens(&[0, 1, 2], 150, 64).unwrap(), 150);
    }

    #[test]
    fn visible_tokens_empty_is_zero() {
        assert_eq!(auxiliary_visible_tokens(&[], 300, 128).unwrap(), 0);
    }

    #[test]
    fn visible_tokens_rejects_non_chronological() {
        assert!(auxiliary_visible_tokens(&[1, 0], 300, 128).is_err()); // decreasing
        assert!(auxiliary_visible_tokens(&[0, 0], 300, 128).is_err()); // duplicate
        assert!(auxiliary_visible_tokens(&[2], 256, 128).is_err()); // begin >= frontier
        assert!(auxiliary_visible_tokens(&[0, 1, 2], 300, 0).is_err()); // zero block
    }

    // ── bind primitive: ninfer bind_auxiliary_selection guard ─────────────
    #[test]
    fn bind_covers_exact_extent() {
        let mut sel = AuxiliarySelection::new(300, 128, true);
        assert!(!sel.is_bound());
        assert_eq!(sel.history_tokens(), 300);
        sel.bind(&[0, 1, 2], 300).unwrap();
        assert!(sel.is_bound());
        assert_eq!(sel.selected(), Some(&[0u32, 1, 2][..]));
    }

    #[test]
    fn bind_requires_head_pool() {
        let mut sel = AuxiliarySelection::new(300, 128, false);
        let err = sel.bind(&[0, 1, 2], 300).unwrap_err();
        assert!(err.contains("no actual auxiliary pool"), "{err}");
        assert!(!sel.is_bound());
    }

    #[test]
    fn bind_is_once() {
        let mut sel = AuxiliarySelection::new(300, 128, true);
        sel.bind(&[0, 1, 2], 300).unwrap();
        assert!(sel.bind(&[0, 1, 2], 300).is_err(), "bind-once");
    }

    #[test]
    fn bind_rejects_wrong_extent() {
        let mut sel = AuxiliarySelection::new(300, 128, true);
        // [0,1] covers 256 ≠ 300.
        let err = sel.bind(&[0, 1], 300).unwrap_err();
        assert!(
            err.contains("differs from the one selected target extent"),
            "{err}"
        );
        assert!(!sel.is_bound());
    }

    #[test]
    fn bind_rejects_non_chronological() {
        let mut sel = AuxiliarySelection::new(300, 128, true);
        assert!(sel.bind(&[1, 0, 2], 300).is_err());
        assert!(sel.bind(&[0, 0, 2], 300).is_err());
    }

    // ── policy admission + FirstLayer source rule ─────────────────────────
    #[test]
    fn admit_empty_preselection_always_ok() {
        assert!(admit_auxiliary_source(LayerwiseMtpHistory::FirstLayer, &[]).is_ok());
        assert!(admit_auxiliary_source(LayerwiseMtpHistory::UpstreamGlobal, &[]).is_ok());
    }

    #[test]
    fn admit_rejects_preselected_per_policy() {
        let err = admit_auxiliary_source(LayerwiseMtpHistory::FirstLayer, &[0, 1]).unwrap_err();
        assert!(
            err.contains("first-layer MTP cannot accept a preselected"),
            "{err}"
        );
        let err = admit_auxiliary_source(LayerwiseMtpHistory::UpstreamGlobal, &[0, 1]).unwrap_err();
        assert!(err.contains("requires its own private probe"), "{err}");
        assert!(admit_auxiliary_source(LayerwiseMtpHistory::Unspecified, &[0]).is_err());
    }

    #[test]
    fn first_layer_ids_come_from_front() {
        let sources = vec![
            SourceLayerView {
                model_layer: 4,
                restored_ids: vec![0, 1, 2],
            },
            SourceLayerView {
                model_layer: 8,
                restored_ids: vec![3, 4],
            },
        ];
        assert_eq!(first_layer_auxiliary_ids(&sources).unwrap(), &[0u32, 1, 2]);
        assert!(first_layer_auxiliary_ids(&[]).is_err());
    }

    #[test]
    fn first_layer_bind_end_to_end() {
        // Pure-host fixture of the ② restore-hook flow: resolve policy, admit the
        // (empty) supplied source, derive IDs from the first restored FA layer,
        // bind them against the target extent. No GPU, no live serve.
        let policy = LayerwiseMtpHistory::Unspecified
            .resolve_default(RecallStrategy::Layerwise, true)
            .unwrap();
        assert_eq!(policy, LayerwiseMtpHistory::FirstLayer);
        assert!(!policy.requires_private_probe());
        admit_auxiliary_source(policy, &[]).unwrap();
        let sources = vec![
            SourceLayerView {
                model_layer: 4,
                restored_ids: vec![0, 1, 2],
            },
            SourceLayerView {
                model_layer: 8,
                restored_ids: vec![0, 1, 2],
            },
        ];
        let ids = first_layer_auxiliary_ids(&sources).unwrap();
        let mut sel = AuxiliarySelection::new(300, 128, true);
        sel.bind(ids, 300).unwrap();
        assert_eq!(sel.selected(), Some(&[0u32, 1, 2][..]));
    }

    #[test]
    fn bind_first_layer_auxiliary_live_hook() {
        // G5 restore-hook mirror: the first FA ordinal's restored canonical ids
        // (u64) bind the head history at the accepted frontier. The extent is
        // DERIVED from the restored view (FirstLayer mirrors the trunk's first
        // layer), then `bind` re-validates chronological/unique + extent.
        // [0,1,2] @ frontier 300 ⇒ 128 + 128 + 44 = 300 visible.
        let sel = bind_first_layer_auxiliary(&[0, 1, 2], 300, 128).unwrap();
        assert!(sel.is_bound());
        assert_eq!(sel.selected(), Some(&[0u32, 1, 2][..]));
        assert_eq!(sel.history_tokens(), 300);
        // Partial tail: [0,1] @ frontier 200 ⇒ 128 + 72 = 200.
        let sel = bind_first_layer_auxiliary(&[0, 1], 200, 128).unwrap();
        assert_eq!(sel.history_tokens(), 200);
        assert_eq!(sel.selected(), Some(&[0u32, 1][..]));
        // Cold start: no restored history ⇒ empty bound selection, extent 0.
        let sel = bind_first_layer_auxiliary(&[], 0, 128).unwrap();
        assert_eq!(sel.history_tokens(), 0);
        assert_eq!(sel.selected(), Some(&[][..]));
    }

    #[test]
    fn bind_first_layer_auxiliary_fails_closed() {
        // Non-chronological / duplicate restored map ⇒ corrupt restore.
        assert!(bind_first_layer_auxiliary(&[1, 0], 300, 128).is_err());
        assert!(bind_first_layer_auxiliary(&[0, 0], 300, 128).is_err());
        // A page starting at/beyond the frontier is not an accepted view.
        assert!(bind_first_layer_auxiliary(&[0, 5], 300, 128).is_err());
        // Canonical id wider than u32 fails closed (ninfer casts to uint32_t).
        assert!(bind_first_layer_auxiliary(&[u64::from(u32::MAX) + 1], 300, 128).is_err());
    }

    // ── head restored-position + RoPE-shift contract (ninfer mirror) ──────
    #[test]
    fn removed_tokens_frontier_minus_visible() {
        // ninfer NL:10285: evaluation.begin - auxiliary_tokens.
        assert_eq!(auxiliary_removed_tokens(384, 256).unwrap(), 128);
        assert_eq!(auxiliary_removed_tokens(384, 384).unwrap(), 0);
        assert_eq!(auxiliary_removed_tokens(0, 0).unwrap(), 0);
        // visible > frontier ⇒ underflow fail-closed.
        assert!(auxiliary_removed_tokens(256, 384).is_err());
        // difference overflow u32 fail-closed.
        assert!(auxiliary_removed_tokens(u32::MAX as u64 + 2, 1).is_err());
    }
}
