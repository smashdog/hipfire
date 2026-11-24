// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;

/// 期2.3: the fenced-import decision — the core `ImportStart` plus the
/// frontend-query identity outcome (KVMem/nInfer
/// `finalize_layerwise_query_source`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportDecision {
    pub start: ImportStart,
    /// The query the first decode step must evaluate at the source
    /// (`None` = already effective → the source keeps its selected views).
    pub effective_query: Option<(u64, u64)>,
    /// KeepSelected: the checkpoint's KV/score already covers this request's
    /// question and the budgets fit, so no per-request Query is needed.
    pub keep_selected: bool,
}

/// Pure query identity at a checkpoint source (nInfer
/// `layerwise_query_at_source`): a proven span equal to the source's recorded
/// span and wholly inside the exact native prefix is already effective.
pub fn layerwise_query_identity(
    query: Option<(u64, u64)>,
    previous_query: Option<(u64, u64)>,
    frontier: u64,
    exact_native_prefix: bool,
) -> Option<(u64, u64)> {
    match query {
        Some(q) if exact_native_prefix && previous_query == Some(q) && q.1 <= frontier => None,
        other => other,
    }
}

/// Pure engine→core mapping for the fenced-import admission (期2.2a/2.3): the
/// exact restored prefix feeds one `WindowPlan` over the Layerwise capacity
/// `B + R`, with the frontend query identity deciding Pressure vs
/// KeepSelected (`finalize_layerwise_query_source`).
pub fn layerwise_import_decision(
    prompt_end: u64,
    output_tokens: u64,
    capacity_tokens: u64,
    gen_reserve: u64,
    frontier: u32,
    visible_tokens: u64,
    query: Option<(u64, u64)>,
    previous_query: Option<(u64, u64)>,
    exact_native_prefix: bool,
) -> Result<ImportDecision, String> {
    let effective =
        layerwise_query_identity(query, previous_query, frontier as u64, exact_native_prefix);
    let block = KV_BLOCK_TOKENS as u64;
    let limit = capacity_tokens.saturating_sub(block);
    let remaining = prompt_end.saturating_sub(frontier as u64);
    let fits = visible_tokens <= limit
        && remaining <= limit.saturating_sub(visible_tokens)
        && output_tokens
            <= limit
                .saturating_sub(visible_tokens)
                .saturating_sub(remaining);
    // A proven span is required to claim KeepSelected: an unproven request
    // (`None`) must not suppress the historical per-request Query.
    let keep_selected =
        exact_native_prefix && frontier > 0 && query.is_some() && effective.is_none() && fits;
    let envelope = WindowEnvelope {
        prompt_end,
        block_tokens: KV_BLOCK_TOKENS as u32,
        generation_tokens: gen_reserve,
        output_tokens,
        rolling_active_window: false,
        prefill_window: if keep_selected {
            WindowPolicy::KeepSelected
        } else {
            WindowPolicy::Pressure
        },
        query: effective.map(|(begin, end)| QueryRange { begin, end }),
        replay: effective.map(|(begin, _)| QueryRange {
            begin,
            end: prompt_end,
        }),
        reserve_query_output: effective.is_some(),
        probe_tokens: 0,
        pressure_probe_tokens: 0,
        has_query_mandatory: false,
    };
    let plan = WindowPlan::new(&envelope, capacity_tokens)
        .map_err(|e| format!("kvmem import window: {e}"))?;
    let start = plan
        .import_plan(WindowState {
            history: frontier as u64,
            visible_tokens,
        })
        .map_err(|e| format!("kvmem import plan: {e}"))?;
    Ok(ImportDecision {
        start,
        effective_query: effective,
        keep_selected,
    })
}

/// Pure engine→core mapping for the fenced-import admission: the exact restored
/// prefix (`frontier`, page-aligned) feeds one `WindowPlan` over the layerwise
/// capacity `B + R`, and the returned `ImportStart` is the core's decision.
pub fn layerwise_import_admission(
    prompt_end: u64,
    output_tokens: u64,
    capacity_tokens: u64,
    gen_reserve: u64,
    frontier: u32,
    visible_tokens: u64,
) -> Result<ImportStart, String> {
    let envelope = WindowEnvelope {
        prompt_end,
        block_tokens: KV_BLOCK_TOKENS as u32,
        generation_tokens: gen_reserve,
        output_tokens,
        rolling_active_window: false,
        prefill_window: WindowPolicy::Pressure,
        query: None,
        replay: None,
        reserve_query_output: false,
        probe_tokens: 0,
        pressure_probe_tokens: 0,
        has_query_mandatory: false,
    };
    let plan = WindowPlan::new(&envelope, capacity_tokens)
        .map_err(|e| format!("kvmem import window: {e}"))?;
    plan.import_plan(WindowState {
        history: frontier as u64,
        visible_tokens,
    })
    .map_err(|e| format!("kvmem import plan: {e}"))
}
