// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 The hipfire authors
// hipfire — see LICENSE or NOTICE in the project root.

//! Host-only smoke test for the layerwise window/import policy wrappers: the
//! literal 72a81cc action/admit/selection policy (`kvmem::LayerwiseWindowPlan`)
//! and the fenced-import admission tests (`kvmem::plan_layerwise_import`).
//! Pure host logic: no GPU, no model, no catalog mutation.

use hipfire_kvmem::{
    ImportStart, QueryRange, WindowAction, WindowEnvelope, WindowPlan, WindowPolicy, WindowState,
};

const BLOCK: u32 = 128;

fn envelope(prompt_end: u64, generation: u64, output: u64, capacity: u64) -> WindowEnvelope {
    WindowEnvelope {
        prompt_end,
        block_tokens: BLOCK,
        generation_tokens: generation,
        output_tokens: output,
        rolling_active_window: false,
        prefill_window: WindowPolicy::Pressure,
        query: None,
        replay: None,
        reserve_query_output: false,
        probe_tokens: 0,
        pressure_probe_tokens: 0,
        has_query_mandatory: false,
    }
}

#[test]
fn prefill_action_admit_and_checkpoints() {
    let capacity = 2048;
    let plan = WindowPlan::new(&envelope(1024, 256, 64, capacity), capacity).expect("plan");
    let state = WindowState {
        history: 0,
        visible_tokens: 0,
    };
    assert_eq!(plan.action(state).expect("action"), WindowAction::None);

    // Admit tiles whole blocks up to the prompt end.
    let first = plan.admit(state, 300).expect("admit");
    assert_eq!(first, 256, "300 rows floor to two blocks");
    let mut history = first;
    let mut visible = first;
    while history < 1024 {
        let step = plan
            .admit(
                WindowState {
                    history,
                    visible_tokens: visible,
                },
                1024 - history,
            )
            .expect("admit");
        assert!(step > 0 && step % BLOCK as u64 == 0, "step {step}");
        history += step;
        visible += step;
    }
    assert_eq!(history, 1024);
    assert_eq!(plan.selection_budget(0).expect("budget"), capacity - 256);
    assert_eq!(
        plan.prefill_checkpoint().expect("checkpoint"),
        Some(768),
        "branch-safe prefill checkpoint"
    );
    assert_eq!(plan.generation_checkpoint().expect("gen"), 896);
    assert_eq!(plan.prefill_boundary(0).expect("boundary"), 1792);
    assert!(plan.query_evaluation().expect("evaluation").is_none());
}

#[test]
fn query_boundary_is_atomic_and_action_dispatches() {
    let capacity = 3072; // room for the 768-row query plus its reserve
    let mut e = envelope(2048, 1920, 64, capacity);
    e.query = Some(QueryRange {
        begin: 400,
        end: 600,
    });
    e.replay = Some(QueryRange {
        begin: 256,
        end: 2048,
    });
    e.reserve_query_output = true;
    let plan = WindowPlan::new(&e, capacity).expect("plan");
    assert_eq!(
        plan.query_evaluation().expect("evaluation"),
        Some(QueryRange {
            begin: 256,
            end: 2048
        }),
        "evaluation floors the replay begin and takes the replay end"
    );
    // Tiling stops exactly at the aligned evaluation origin.
    let state = WindowState {
        history: 0,
        visible_tokens: 0,
    };
    let mut history = 0;
    while history < 256 {
        let step = plan
            .admit(
                WindowState {
                    history,
                    visible_tokens: history,
                },
                512,
            )
            .expect("admit");
        assert_eq!(step, 256 - history, "tile clipped to the query origin");
        history += step;
    }
    let at = WindowState {
        history: 256,
        visible_tokens: 256,
    };
    assert_eq!(plan.action(at).expect("action"), WindowAction::Query);
    let rows = plan.admit(at, 4096).expect("admit");
    assert_eq!(rows, 1792, "atomic query admits its complete evaluation");
    // A short desired tile cannot start the atomic query.
    let none = plan.admit(at, 100).expect("admit");
    assert_eq!(none, 0);
}

#[test]
fn keep_selected_exhausts_at_the_block_boundary() {
    let capacity = 2048;
    let mut e = envelope(1024, 256, 64, capacity);
    e.prefill_window = WindowPolicy::KeepSelected;
    let plan = WindowPlan::new(&e, capacity).expect("plan");
    // Boundary = capacity - 1 * block for KeepSelected.
    assert_eq!(plan.prefill_boundary(0).expect("boundary"), 1920);
    // visible_tokens never exceeds history, so the boundary is reached on the
    // decode side of the request (prompt_end == 1024 here).
    let decode = WindowState {
        history: 1920,
        visible_tokens: 1920,
    };
    assert_eq!(
        plan.action(decode).expect("action"),
        WindowAction::Exhausted
    );
    let under = WindowState {
        history: 1792,
        visible_tokens: 1792,
    };
    assert_eq!(plan.action(under).expect("action"), WindowAction::None);
}

#[test]
fn semantic_chunk_steps_before_pressure() {
    let capacity = 2048;
    let mut e = envelope(1024, 256, 64, capacity);
    e.prefill_window = WindowPolicy::SemanticChunk;
    let plan = WindowPlan::new(&e, capacity).expect("plan");
    assert_eq!(
        plan.action(WindowState {
            history: 0,
            visible_tokens: 0
        })
        .expect("action"),
        WindowAction::Chunk
    );
    // SemanticChunk has no branch-safe prefill checkpoint.
    assert_eq!(plan.prefill_checkpoint().expect("checkpoint"), None);
}

#[test]
fn import_plan_imports_an_exact_prefix() {
    let capacity = 2048;
    let plan = WindowPlan::new(&envelope(1024, 1024, 64, capacity), capacity).expect("plan");
    let start = plan
        .import_plan(WindowState {
            history: 512,
            visible_tokens: 512,
        })
        .expect("import");
    assert_eq!(
        start,
        ImportStart {
            terminal_reserve_overflow: false,
            prefill_probe_overflow: false,
            query_replay_overflow: false,
            budget_tokens: 1024,
            restore_tokens: 512,
            // Only a Pressure import narrows the selection budget; an exact
            // import keeps the ordinary selection path (core leaves it 0).
            selection_budget: 0,
        }
    );
    assert!(!start.pressure());
}

#[test]
fn import_plan_reports_terminal_reserve_overflow_with_exact_budgets() {
    let capacity = 1152;
    let plan = WindowPlan::new(&envelope(1024, 256, 64, capacity), capacity).expect("plan");
    let start = plan
        .import_plan(WindowState {
            history: 1024,
            visible_tokens: 1024,
        })
        .expect("import");
    assert!(start.terminal_reserve_overflow, "{start:?}");
    assert!(start.pressure());
    // available = 1152 - 64 - 0 - 128 = 960 -> 896 aligned (startup bound)
    assert_eq!(start.budget_tokens, 896);
    assert_eq!(start.restore_tokens, 896);
    assert_eq!(start.selection_budget, 896);
}

#[test]
fn import_plan_reports_prefill_probe_overflow() {
    let capacity = 1152;
    let plan = WindowPlan::new(&envelope(2048, 256, 64, capacity), capacity).expect("plan");
    let start = plan
        .import_plan(WindowState {
            history: 1024,
            visible_tokens: 1024,
        })
        .expect("import");
    assert!(start.prefill_probe_overflow, "{start:?}");
    assert!(!start.terminal_reserve_overflow);
    assert_eq!(start.budget_tokens, capacity - 256);
    assert_eq!(start.restore_tokens, start.budget_tokens);
}

#[test]
fn invalid_envelopes_fail_closed() {
    let capacity = 2048;
    // Query outside the prompt is rejected at construction.
    let mut e = envelope(1024, 512, 64, capacity);
    e.query = Some(QueryRange {
        begin: 900,
        end: 1200,
    });
    assert!(WindowPlan::new(&e, capacity).is_err());
    // An import whose block remainders differ must fail closed.
    let plan = WindowPlan::new(&envelope(1024, 1024, 64, capacity), capacity).expect("plan");
    assert!(plan
        .import_plan(WindowState {
            history: 512,
            visible_tokens: 500
        })
        .is_err());
}
