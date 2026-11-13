// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-arch-qwen35: Qwen3.5 architecture (dense + MoE / A3B / A10B / A17B).
//!
//! This crate implements the [`hipfire_runtime::arch::Architecture`] trait
//! for Qwen3.5. It owns the model forward pass, weight loading, KV-state
//! layout, and the speculative-decoding glue that today is qwen35-specific
//! (`speculative.rs`; `pflash.rs` evacuated to `hipfire-pflash` per lean-up
//! map B3 — retained legacy research, not mainline).
//!
//! Future work (per docs/plans/engine-modularization.prd Phase 2):
//!   - `speculative.rs` will become arch-generic and move back into
//!     `hipfire-runtime`. It lives here today because the existing impl is
//!     deeply coupled to `qwen35::*` symbols (config, weights, scratch,
//!     forward functions). `pflash.rs` was evacuated to `hipfire-pflash`
//!     per lean-up map B3 (§5.1) — retained legacy research, historical
//!     reproduction only. PR 8 freezes the dep direction `arch-qwen35 →
//!     runtime`, but accepts that today's spec is not generic enough to
//!     live above the arch boundary.
//!
//! The `arch` module exposes the trait impl for use by the runtime's
//! daemon and other consumers via `hipfire_arch_qwen35::Qwen35`.

/// Producer capability: the Qwen3.5/3.6 native MTP drafter (`mtp_speculator`,
/// `mtp_head`) has no measured autoregressive floor — it never prices a draft
/// window against the AR step or retires itself when drafting loses. So
/// `speculation.mtp = auto` keeps plain AR (the head is not loaded); only an
/// explicit `on` / `--spec mtp` arms it.
pub const MTP_MEASURED_AR_FLOOR: bool = false;

// Qwen3.5 is a hybrid DeltaNet + FullAttention architecture; all the
// runtime infrastructure it touches is `deltanet`-gated. When the parent
// build doesn't enable the feature, the crate is a no-op stub. This keeps
// `cargo build --no-default-features` working and matches the gating that
// was on `engine::qwen35` pre-Phase-2.
#[cfg(feature = "deltanet")]
pub mod arch;
#[cfg(feature = "deltanet")]
pub mod arch_model;
#[cfg(feature = "deltanet")]
pub mod carrier;
/// Qwen3.5 hybrid-state checkpoint pool for serving prefix-cache resume
/// (spec §4.5 C5). Byte-bounded LRU of immutable `DeltaNetSnapshot`
/// bundles keyed by `(CacheDomain, boundary_p)`, with `plan_resume`
/// last-token semantics and GPU-backed capture/restore paths.
#[cfg(feature = "deltanet")]
pub mod checkpoint;
/// Minimal DFlash batched-lane seam (lane state wrapper, lane draft, shared
/// verify head) used by the Gate 0 replay probe. Deltanet-gated like
/// `dflash_spec`.
#[cfg(feature = "deltanet")]
pub mod dflash_cb;
/// Design A slice A4: the planned (railgun) DFlash draft emitter over the
/// frozen plan sink, its shape family / patch table, and the oracle binder.
pub mod dflash_draft_emit;
/// Railgun design A slice A6: the certified-window owner, admission
/// (`HIPFIRE_DFLASH_RAILGUN_WINDOW`), and the §6 window transaction.
pub mod dflash_railgun;
/// Railgun design A slice A8: the GPU-facing `WindowBuilder` (plan, author,
/// lower, stage, read back) the owner installs for `HIPFIRE_DFLASH_RAILGUN_WINDOW=1`.
pub mod dflash_railgun_builder;
/// DFlash2 speculative decode for the multi-slot serve engine: shared draft
/// state, per-slot draft context, and the draft/verify-accept steps that
/// plug into `serve_engine`'s batched forward. Deltanet-gated like
/// `dflash_spec` (the draft forward consumes `ModelSlot`-family types).
#[cfg(feature = "deltanet")]
pub mod dflash_slot;
/// Qwen3.5 DFlash / DDTree speculative-decode state (`DflashState`,
/// `load_dflash_state`) and the `DflashSpeculator` impl of the arch-generic
/// `hipfire_runtime::spec::Speculator`. Deltanet-gated — it owns `ModelSlot`-
/// based draft verify.
#[cfg(feature = "deltanet")]
pub mod dflash_spec;
/// Railgun design A slice A5: the DFlash verify-window emitter
/// (`plan_dflash_verify`) and its patch-table proof. Not deltanet-gated,
/// matching `speculative`, which it consumes.
pub mod dflash_verify_emit;
/// Retained-PM4 route state for the fixed B=16 DFlash2 target-verify forward
/// (`DflashVerifyPm4`). Owns the phase machine, admission binding, and
/// route-proof counters; `speculative` owns the GPU half.
/// Not deltanet-gated, matching `speculative`, which consumes it.
pub mod dflash_verify_pm4;
/// Railgun design A slice A8: the pure half of the window builder (A4 draft +
/// A5 verify emissions -> authored `WindowPlanParts` with rebased patches,
/// copy/fill nodes and exact-artifact facts).
pub mod dflash_window_build;
/// Compressed MTP draft vocabulary sliced from the trunk lm_head (exact gfx1100 default).
pub mod draft_vocab;
/// SP3 Task 2 — `forward_batch_slots`, the N-slot forward pass. A PARALLEL
/// entry point to `qwen35::forward_prefill_batch_with_pbs_opts` (never a
/// modification of it — see the module doc for why), routing attention
/// and KV-write through SP1's slot-aware `_slots` kernels and DeltaNet
/// through SP2's per-slot `DeltaNetState`. Q8_0-only; depends on `qwen35`
/// for weight/config/scratch types, hence deltanet-gated like it.
#[cfg(feature = "deltanet")]
pub mod forward_slots;
/// GGUF -> hipfire tensor-name map and hyper-parameters for the dense
/// Qwen3.5 family (the GGUF-native carrier's per-arch block).
#[cfg(feature = "deltanet")]
pub mod gguf_map;
/// GGUF-native weight source for the dense Qwen3.5 family: ggml block bytes
/// as `DType::Ggml*` weights with the exporter's data edits undone.
#[cfg(feature = "deltanet")]
pub mod gguf_source;
/// KVMem host-offload statistics taps: post-norm/pre-RoPE Q/K copies taken
/// between k_norm and RoPE in each FullAttention layer. Owned by
/// `PrefillBatchScratch::kvmem_taps`; `None` (the default) launches nothing.
#[cfg(feature = "deltanet")]
pub mod kvmem_taps;
/// KVMem host-offload controller: per-slot paged-page residency
/// (spill/restore via `SPILLED_PAGE` sentinels), block-statistics feeding
/// and retrieval selection for the multi-slot serve engine. Enabled with
/// `HIPFIRE_KVMEM=1`; requires the paged Q8 pool.
#[cfg(all(feature = "deltanet", feature = "kvmem"))]
pub mod kvmem_ctl;
/// Layerwise Q8 packed-page layout adapter (Phase 3.2 ①地基): bridges hipfire Q8
/// geometry to the core generic `PayloadLayout` manifest + the crate-agnostic
/// packed arena. Clean sibling to `kvmem_ctl`; not wired into live serve in ①.
#[cfg(feature = "kvmem")]
pub mod layerwise_layout;
#[cfg(feature = "deltanet")]
pub(crate) mod layer_driver;
#[cfg(feature = "deltanet")]
pub mod mtp_compose;
#[cfg(feature = "deltanet")]
pub mod mtp_dense_tp;
#[cfg(feature = "deltanet")]
pub mod mtp_head;
#[cfg(feature = "deltanet")]
pub mod mtp_probe;
#[cfg(feature = "deltanet")]
pub mod mtp_spec;
/// Qwen3.5 `MtpDrafter` impl (the arch half of the unified MTP spec-decode
/// core). Deltanet-gated — it touches `ModelSlot` + `MtpSpecState`.
#[cfg(feature = "deltanet")]
pub mod mtp_speculator;
#[cfg(feature = "deltanet")]
pub(crate) mod paro_moe;
#[cfg(feature = "deltanet")]
pub mod qwen35;
#[cfg(feature = "deltanet")]
#[cfg(feature = "deltanet")]
pub mod serve_engine;
/// Shared-weight host v1: typed export/import of a dense Qwen3.5 weight set
/// as read-only POSIX-FD VMM allocations. Linux only.
#[cfg(all(feature = "deltanet", target_os = "linux"))]
pub mod shared_weights;
/// Qwen3.5 impls of the arch-generic `hipfire_runtime::spec` seam
/// (`impl SpecTarget for ModelSlot`). Deltanet-gated — it touches `ModelSlot`.
#[cfg(feature = "deltanet")]
mod spec_impl;
pub mod speculative;

/// Grammar-guided decoding for tool-call format — re-exported from `saddle_core::grammar::json`.
///
/// Unified in `saddle-core` (lean-up map B1). This re-export preserves the
/// `hipfire_arch_qwen35::grammar` path for existing consumers while the
/// implementation lives in `saddle_core::grammar::json`.
pub use saddle_core::grammar::json as grammar;

/// qwen35 grammar `Config` resolver that restores the `HIPFIRE_QWEN35_*` env
/// overrides lost in the B1 unification. Reads the two tunables via
/// `hipfire_config::developer_var` with the same parse/bounds as the
/// pre-merge `grammar.rs:128-153` and falls back to `Config::default()`.
/// Exposed here so the daemon example (outside this crate) can use the same
/// single source of truth.
pub mod grammar_config;
pub use grammar_config::{resolve_grammar_config, resolve_qwen35_grammar_config};

/// Per-token spec-decode emission (`SpecEmit`). Pure CPU; named here because it
/// drives the qwen35 `grammar` matcher. Built via [`spec_emit::Qwen35Emit::from_ctx`].
pub mod spec_emit;

#[cfg(feature = "deltanet")]
pub use arch::Qwen35;

#[cfg(feature = "deltanet")]
pub use carrier::{free_qwen35_bundle, load_bundle as load_qwen35_bundle, Qwen35Bundle};
#[cfg(feature = "deltanet")]
pub use mtp_compose::{spec_step_dflash_mtp_tree, MtpComposeTreeResult, MtpComposeTreeState};
#[cfg(feature = "deltanet")]
pub use mtp_speculator::{build_qwen35_mtp_speculator, Qwen35MtpDrafter};

/// G4b (Escha-W2 port, Task 9): expose the arch-6 MoE router selection step
/// so `examples/escha_router_contract.rs` can call the *actual* production
/// router — not a reimplementation of it — on an arbitrary `.hfq` layer and
/// compare its top-K expert SET against EschaLabs' shipped fixture.
///
/// This calls exactly the same primitives the production decode/prefill path
/// calls (`hipfire_runtime::llama::weight_gemv` for the router GEMV, then
/// whichever top-K kernel `hipfire_dispatch::pipeline::run_moe_decode` would
/// pick for this GPU's arch — the fused `moe_router_softmax_topk_k8_wave64_exact`
/// on gfx1100/gfx1151, or the reference two-launch `softmax_f32` +
/// `moe_topk_renorm_k8` everywhere else). No selection math is duplicated
/// here; the kernel-choice arch check itself is
/// `hipfire_dispatch::pipeline::exact_wave64_router_predicate` — the exact
/// function `run_moe_decode` calls, not a copy of its logic — so this
/// helper cannot silently drift from production's actual gate.
///
/// This test model's `.hfq` (`gate.weight` quant_type=1/F16) does not carry
/// escha routed-expert dtypes, so it never exercises the escha-only f16
/// router-logits round-trip (review Fix 1); this helper intentionally omits
/// that step to stay a pure probe of the pre-existing selection kernels.
#[cfg(feature = "deltanet")]
pub fn escha_router_topk_for_test(
    hfq_path: &str,
    layer: usize,
    x: &[f32],
    n_tokens: usize,
    hidden: usize,
    top_k: usize,
) -> Result<Vec<u32>, String> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_runtime::hfq::{load_weight_tensor_pread, HfqFile};
    use hipfire_runtime::llama::weight_gemv;
    use rdna_compute::{DType, Gpu};

    if x.len() != n_tokens * hidden {
        return Err(format!(
            "escha_router_topk_for_test: x.len()={} != n_tokens*hidden={}",
            x.len(),
            n_tokens * hidden
        ));
    }

    let hfq = HfqFile::open(std::path::Path::new(hfq_path)).map_err(|e| e.to_string())?;
    let config = qwen35::config_from_hfq(&hfq)?;
    let n_exp = config.num_experts;
    let norm_topk = config.norm_topk_prob;

    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    let ctx = DispatchCtx::new(&gpu);
    // The *actual* predicate `run_moe_decode` (hipfire-dispatch/src/pipeline/mod.rs)
    // uses to pick the fused exact-wave64 router kernel over the reference
    // two-launch path — extracted to `exact_wave64_router_predicate` so this
    // helper cannot silently drift from production's real gate (review Fix 2:
    // the previous `is_gfx1151() || is_gfx1100()` copy here happened to agree
    // with production for this model/GPU, but wasn't actually the same check —
    // production also requires `n_exp == 256` and honors the
    // `HIPFIRE_GFX1100_ROUTER_W64` override on gfx1100).
    // HIPFIRE_MOE_ROUTER_SHARED_FUSE-gated shared-expert fusion is a pure perf
    // variant of the same math and is left out here — it doesn't change which
    // experts get selected.
    let gfx1100_router_mode = hipfire_config::developer_var("HIPFIRE_GFX1100_ROUTER_W64").ok();
    // Inline equivalent of the production gate in
    // `hipfire-dispatch/src/pipeline/moe_program.rs` (the sealed MoE program
    // computes `exact_wave64_router` locally rather than exposing a helper):
    // n_exp == 256 and (gfx1100 without the "0"/"approx" override, or gfx1151).
    let use_exact_wave64 = n_exp == 256
        && ((ctx.arch.is_gfx1100()
            && !matches!(gfx1100_router_mode.as_deref(), Some("0" | "approx")))
            || ctx.arch.is_gfx1151());

    fn exact_name(s: &str) -> Vec<String> {
        vec![s.to_string()]
    }
    let weight_name = format!("model.language_model.layers.{layer}.mlp.gate.weight");
    let router = load_weight_tensor_pread(
        &hfq,
        &mut gpu,
        &weight_name,
        n_exp,
        hidden,
        exact_name,
        hipfire_runtime::weight_backend::MemoryTarget::Device,
    )
    .map_err(|e| e.to_string())?;

    let logits = gpu
        .alloc_tensor(&[n_exp], DType::F32)
        .map_err(|e| e.to_string())?;
    // topk_idx carries raw i32 selections in an f32-tagged buffer — the same
    // "i32-in-F32 alias" convention `moe_ffn_decode_impl` uses for its scratch
    // (see qwen35/forward.rs `capture_expert_stats`'s `ti[krank].to_bits()`).
    let topk_idx = gpu
        .alloc_tensor(&[top_k], DType::F32)
        .map_err(|e| e.to_string())?;
    let topk_w = gpu
        .alloc_tensor(&[top_k], DType::F32)
        .map_err(|e| e.to_string())?;

    let mut out = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        let row = &x[t * hidden..(t + 1) * hidden];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len() * 4) };
        let x_gpu = gpu
            .upload_raw(bytes, &[hidden])
            .map_err(|e| e.to_string())?;

        weight_gemv(&mut gpu, &router, &x_gpu, &logits).map_err(|e| e.to_string())?;

        if use_exact_wave64 {
            gpu.moe_router_softmax_topk_k8_wave64_exact(
                &logits, &topk_idx, &topk_w, n_exp, norm_topk,
            )
            .map_err(|e| e.to_string())?;
        } else {
            gpu.softmax_f32(&logits).map_err(|e| e.to_string())?;
            gpu.moe_topk_renorm_k8(&logits, &topk_idx, &topk_w, n_exp, norm_topk)
                .map_err(|e| e.to_string())?;
        }

        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
        let idx_f32 = gpu.download_f32(&topk_idx).map_err(|e| e.to_string())?;
        for v in idx_f32.iter().take(top_k) {
            out.push((v.to_bits() as i32) as u32);
        }

        let _ = gpu.free_tensor(x_gpu);
    }

    let _ = gpu.free_tensor(logits);
    let _ = gpu.free_tensor(topk_idx);
    let _ = gpu.free_tensor(topk_w);
    router.free_all(&mut gpu);

    Ok(out)
}
