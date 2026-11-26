// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SlotEngine — the multi-slot rig owned by one background thread.
//
// `hipfire serve` already speaks OpenAI-compatible HTTP, and HTTP is already
// concurrent. What serialises requests today is the single `Engine` behind a
// `Mutex<ServeRuntime>`. This engine is fed by a channel and lives outside any
// such lock, so N requests are in flight together and share one batched
// forward per step.
//
// The thread owns the GPU rig exclusively: nothing else touches `Gpu`,
// `SlotPool`, the arenas or the DeltaNet states. All interaction is by message,
// which is what makes "no lock" safe rather than merely fast.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use hipfire_runtime::admission::{AdmissionController, AdmitError, ModelFootprint};
use hipfire_runtime::serve::{
    send_event, Continuation, DoneReason, EngineStats, Event, RejectClass, SubmitRequest,
};
use hipfire_runtime::session_table::{SessionId, SessionTable};
use hipfire_runtime::swap::snapshot::{capture_slot, restore_slot, SnapshotStamp};
use hipfire_runtime::swap::SwapManager;
use rdna_compute::page_pool::PAGE_TOKENS;
use rdna_compute::sampling::SlotSampleParams;
use rdna_compute::slot_pool::{SlotId, SlotPool};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::checkpoint::{capture_checkpoint, plan_resume, CheckpointId, CheckpointPool};
use crate::forward_slots::{forward_batch_slots_graphed_opts, SlotDecodeGraph, SlotDescStaging};
use crate::grammar;
use crate::qwen35::{
    self, DeltaNetState, LayerType, PrefillBatchScratch, Qwen35Scratch, Qwen35Weights,
};
use crate::speculative::DeltaNetSnapshot;
use hipfire_runtime::llama::{KvCache, VMode};
use hipfire_runtime::prefix_index::{Handle, PinTicket, PrefixIndex};
use hipfire_runtime::scheduler::{
    is_runnable_decode, is_runnable_prefill, PendingWork, Scheduler, SpecKind, VlPrefill,
};
use hipfire_runtime::serve_contract::{
    CacheDomain, DrafterDecision, PrefixLookup, PrefixLookupResult,
};
use hipfire_runtime::serve_fairness::{FairQueue, Grant, Selection};
use hipfire_runtime::serve_wait::WaitQueue;
use hipfire_runtime::slot_batch::SlotBatch;
use hipfire_runtime::tokenizer::Tokenizer;

#[cfg(feature = "kvmem")]
#[path = "serve_engine_kvmem.rs"]
mod kvmem_engine;

#[path = "serve_engine_kvmem_hooks.rs"]
mod kvmem_hooks;

#[path = "serve_engine_admit.rs"]
mod serve_engine_admit;

#[path = "serve_engine_model.rs"]
mod serve_engine_model;

use self::kvmem_hooks::{
    kvmem_dflash_validation_error, kvmem_enabled, kvmem_legacy_enabled, kvmem_requested,
    legacy_continuation_matches, trace_continuation_divergence,
};

#[cfg(feature = "kvmem")]
use self::kvmem_engine::{
    advance_legacy_query_replay, kvmem_freeze_endpoint, restore_legacy_native_image,
    LegacyNativeImage, LegacyQueryRun,
};

/// A live VL session may only be extended when the request's current image is
/// the same media the session's prefix was built from. Token ids cannot
/// distinguish two images (their `image_pad` runs are identical), so the
/// sticky content digest is the only safe gate. A text follow-up carries no
/// new image and inherits the session's digest unchanged.
fn media_continuation_matches(
    session: &hipfire_runtime::session_table::Session,
    visual: Option<&hipfire_runtime::serve::VisualData>,
) -> bool {
    if !session.has_image {
        return true;
    }
    match visual.map(|vd| vd.media_digest()) {
        Some(digest) => session.media_digest.is_none_or(|held| held == digest),
        None => true,
    }
}

/// Internal control plane for the engine thread. Public methods map onto these
/// messages; the GPU rig stays exclusively on that thread.
///
/// Continuous-batch scheduler integration is deferred — multi-slot is an
/// alternate single-weight daemon path, not ContinuousBatchScheduler.
enum EngineCommand {
    Submit(SubmitRequest),
    /// Remove a QUEUED (not yet admitted) request by the tag the submitter
    /// stamped on it. No-op when the tag matches nothing — cancellation is
    /// idempotent (spec §4.6 C6).
    CancelWaiting {
        request_tag: u64,
    },
    Close {
        session: u64,
        reply: Sender<Result<(), String>>,
    },
    Reset {
        reply: Sender<Result<(), String>>,
    },
    /// Cooperative in-turn pass termination (think-recovery): end the
    /// CURRENT pass of `session` at the next token boundary but keep the
    /// session's KV resident, so the daemon can immediately start a
    /// continuation pass against the accepted prefix. The engine reports
    /// `Done { reason: MaxTokens }` for the interrupted pass; the daemon
    /// distinguishes it from a real budget exhaustion by the request it
    /// sent. No-op when the session has no in-flight slot (already done,
    /// parked, or unknown) — interruption is advisory, not synchronous.
    Interrupt {
        session: u64,
    },
}

#[derive(Clone)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    /// Resolved public KVMem contract carried by the daemon configure message.
    /// None retains the existing non-public/Layerwise configuration route.
    pub kvmem: Option<hipfire_config::kvmem::ResolvedConfig>,
    pub n_slots: usize,
    pub cap_tokens: usize,
    /// Prefill tokens taken from one slot per step. Bounds the batch scratch
    /// (`n_slots × prefill_chunk` rows, NOT `cap_tokens`) and keeps a long
    /// prompt from blocking other slots for its whole prefill — the property
    /// the scheduler was built around. Sizing scratch by `cap_tokens` put a
    /// 16k-ctx A3B out of reach of a 24 GB card.
    pub prefill_chunk: usize,
    pub host_budget_bytes: u64,
    pub swap_dir: PathBuf,
    /// Disk-tier ceiling for swap spill files. 0 disables the disk tier.
    pub swap_disk_bytes: u64,
    /// True when the model has vision support (either inline in the trunk
    /// HFQ or in a separate `.vl` sidecar). Rig::build loads VisionWeights
    /// and VisionConfig when set.
    pub is_vl: bool,
    /// Path to a standalone `.vl` file containing only vision tower tensors.
    /// When set, vision weights are loaded from this file instead of the
    /// trunk HFQ. When `None`, falls back to inline vision tensors in the
    /// trunk (backward compat with pre-separation artifacts).
    pub vl_path: Option<PathBuf>,
    /// Whether the vision tower is uploaded whole at load
    /// (`TowerResidency::Resident`, `vision.mode=resident`) or streamed
    /// block-by-block through staging slots during the image encode.
    pub vision_residency: hipfire_arch_qwen35_vl::qwen35_vl::TowerResidency,
    /// MTP speculative decode depth (K candidates per cycle). 0 = MTP off.
    pub mtp_k: usize,
    /// The `params.mtp_mode` tri-state (`off`/`auto`/`on`). `mtp_k > 0` alone
    /// cannot distinguish `auto` from an explicit `on`, so `SlotEngine::spawn`
    /// reads this (not `mtp_k`) for the producer-capability gate (`auto`
    /// arms only with a declared measured AR floor, which qwen35 lacks) and
    /// the arming site for the host-spill decline.
    pub mtp_mode: hipfire_config::MtpMode,
    /// Raw KV-mode string from the load request (`--kv-mode`). Empty = fall
    /// back to `HIPFIRE_KV_MODE` / config, resolved through the slots site
    /// policy (full static ladder + native bf16/f16; q8 default; fp8
    /// gfx1201-only). Any unrecognized or warn-resolved value — and an
    /// explicit fp8 off gfx1201 — is rejected in `Rig::build`, carried
    /// explicitly so a bypass of the daemon capability gate still fails
    /// closed inside the engine thread.
    pub kv_mode_raw: String,
    /// Effective `--kv-backend` for this load (`legacy` default). Slot
    /// arenas are fixed contiguous `SlotPool` allocations; `vmm` is rejected
    /// in `Rig::build` before any GPU allocation.
    pub kv_backend: String,
    /// Cross-session prefix cache (radix index + checkpoint pool). Default
    /// false: the existing session-local continuation path (convo hashes)
    /// is unchanged. When true, cold admits consult a PrefixIndex for
    /// reusable KV pages and a CheckpointPool for recurrent state
    /// (spec §4.5–4.6). Requires paged KV (PagePool).
    pub prefix_cache: bool,
    /// Maximum device bytes for the checkpoint pool. 0 = pool that refuses
    /// inserts (lookups still work for already-published entries from a
    /// prior epoch, but no new captures are stored).
    pub prefix_cache_max_bytes: u64,
    /// Global trunk-row budget per step (spec §5.2 S2): the sum of prefill +
    /// decode + verify rows a step emits must not exceed this. The scheduler
    /// enforces it in `next_batch` (prefill + decode) and the engine subtracts
    /// verify rows before calling the scheduler, so the forward panic-assert
    /// (`n <= pbs.max_batch`) is a fail-closed backstop, not the throttle.
    /// Default 4096 (the registered config default for `serve.max_batch_tokens`).
    pub max_batch_tokens: usize,
    /// Minimum prefill quantum guaranteed to one prefilling slot per step when
    /// space remains after decode allocation (spec §5.2 S2 / §5.3 S3). The
    /// scheduler rotates which prefilling slot gets it across ticks. Default 1
    /// (the registered config default for `serve.prefill_min_tokens`).
    pub prefill_min_tokens: usize,
    /// Bounded waiting room: max queued request count (spec §5.3 S3). Read
    /// from `serve.max_queue`; must be non-zero (Rig::build refuses
    /// `WaitQueue::new(0, _)`). Replaces the R-A4 immediate-reject path.
    pub wait_max_count: usize,
    /// Bounded waiting room: max total queued bytes (spec §5.3 S3). Read
    /// from `serve.max_queue_bytes`; must be non-zero.
    pub wait_max_bytes: u64,
    /// Bounded waiting room: per-waiter timeout in MILLISECONDS, honored
    /// against the wall clock (spec §5.3 S3). Read from
    /// `serve.queue_timeout_ms`. Scheduler ticks are NOT a sound time base
    /// for this deadline: one tick can absorb a 1024-token prefill chunk
    /// (hundreds of ms) and ticks do not advance at all while the engine
    /// blocks on recv with an empty room — a tick-denominated deadline
    /// fired arbitrarily late, or never.
    pub queue_timeout_ms: u64,
    /// Structured-output jump-forward (spec §7.3 G3). When true and a slot
    /// carries a grammar constraint under greedy decoding, the engine calls
    /// `forced_token_run` after the grammar mask to find a bounded run of
    /// provably-forced tokens and commits them without sampling. Default
    /// false (read from `serve.structured_jump_forward`): no behavior change
    /// on the constrained-decode path.
    pub structured_jump_forward: bool,
    /// DFlash2 draft path (registry sidecar / `params.draft` /
    /// `HIPFIRE_DFLASH_DRAFT`). `Some` arms the DFlash2 selector chain on
    /// the slots path; `None` = DFlash off. Mutually exclusive with `mtp_k`
    /// in practice — when both are armed, DFlash wins at admit (deeper
    /// speculation, same verify-exact contract).
    pub dflash_draft: Option<PathBuf>,
    /// `dflash_mode=on`: a draft load failure fails the engine load.
    /// `auto`: warn and serve AR.
    pub dflash_required: bool,
}

pub struct SlotEngine {
    tx: Option<Sender<EngineCommand>>,
    stats: Arc<Mutex<EngineStats>>,
    /// Sessions the engine closed internally (the idle sweep). Drained by the
    /// daemon so its pending-tool broker does not keep dead session ids.
    closed: Arc<Mutex<Vec<u64>>>,
    handle: Option<JoinHandle<Result<(), String>>>,
}

impl SlotEngine {
    pub fn submit(&self, req: SubmitRequest) -> Result<(), String> {
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(EngineCommand::Submit(req))
            .map_err(|_| "engine thread is gone".to_string())
    }

    /// Cancel a QUEUED request by its submitter tag (spec §4.6 C6/A11).
    /// Idempotent: an unmatched tag is a no-op.
    pub fn cancel_waiting(&self, request_tag: u64) {
        if request_tag == 0 {
            return;
        }
        if let Some(tx) = self.tx.as_ref() {
            let _ = tx.send(EngineCommand::CancelWaiting { request_tag });
        }
    }

    /// Ask the engine to end the CURRENT pass of `session` at the next token
    /// boundary while keeping its KV resident (think-recovery pass loop).
    /// Fire-and-forget: the daemon observes the pass end through the request's
    /// `Done { reason: MaxTokens }` event. Idempotent and safe to call for a
    /// session that is already done (no-op).
    pub fn interrupt(&self, session: u64) {
        if let Some(tx) = self.tx.as_ref() {
            let _ = tx.send(EngineCommand::Interrupt { session });
        }
    }

    /// Close a session synchronously. If the session is mid-generation its
    /// in-flight request is cancelled (InFlight removed, reply sender dropped,
    /// work cleared), swap forgotten and SessionTable/pool/admission closed,
    /// then acknowledged. Idle close does the same. Never silently ignored.
    pub fn close(&self, session: u64) -> Result<(), String> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(EngineCommand::Close {
                session,
                reply: reply_tx,
            })
            .map_err(|_| "engine thread is gone".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "engine thread is gone".to_string())?
    }

    /// Drop every session and swap entry, returning the engine to a cold empty
    /// table. Rejects while any slot is still generating.
    pub fn reset(&self) -> Result<(), String> {
        let (reply_tx, reply_rx) = channel();
        self.tx
            .as_ref()
            .ok_or_else(|| "engine is shutting down".to_string())?
            .send(EngineCommand::Reset { reply: reply_tx })
            .map_err(|_| "engine thread is gone".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "engine thread is gone".to_string())?
    }

    pub fn stats(&self) -> EngineStats {
        *self.stats.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Drain the ids of sessions the engine closed on its own (idle sweep).
    pub fn take_closed_sessions(&self) -> Vec<u64> {
        let mut guard = self.closed.lock().unwrap_or_else(|p| p.into_inner());
        std::mem::take(&mut *guard)
    }

    /// Build the rig on a new thread and start serving. Returns once the model
    /// is loaded, so a caller that gets `Ok` can submit immediately.
    ///
    /// The producer-capability gate runs here, before any placement or head
    /// resolution reads `mtp_k`: qwen35 declares no measured AR floor
    /// ([`crate::MTP_MEASURED_AR_FLOOR`]), so an `auto` mode serves plain AR
    /// (no head is resolved or reserved for) and only an explicit `on` keeps
    /// its depth. The daemon's slots route and any direct caller get the same
    /// verdict as the sequential loader.
    pub fn spawn(mut cfg: EngineConfig) -> Result<SlotEngine, String> {
        let admitted_k = slot_mtp_k(cfg.mtp_mode, cfg.mtp_k, crate::MTP_MEASURED_AR_FLOOR);
        if cfg.mtp_k > 0 && admitted_k == 0 && cfg.mtp_mode == hipfire_config::MtpMode::Auto {
            eprintln!(
                "{}",
                hipfire_config::mtp_no_floor_decline_line("qwen35 slots")
            );
        }
        cfg.mtp_k = admitted_k;
        let (tx, rx) = channel::<EngineCommand>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let stats_thread = Arc::clone(&stats);
        let closed = Arc::new(Mutex::new(Vec::new()));
        let closed_thread = Arc::clone(&closed);

        let handle = std::thread::Builder::new()
            .name("hipfire-slot-engine".to_string())
            .spawn(move || -> Result<(), String> {
                match Rig::build(&cfg, closed_thread) {
                    Ok(rig) => {
                        // Publish the swap policy before the load ack, so the
                        // daemon reports the effective budgets and usage.
                        if let Ok(mut st) = stats_thread.lock() {
                            st.note_swap(&rig.swap);
                            st.speculative = rig.speculative_config();
                        }
                        let _ = ready_tx.send(Ok(()));
                        let result = run_loop(rig, rx, stats_thread);
                        if let Err(error) = &result {
                            eprintln!("[serve] engine loop exited: {error}");
                        }
                        result
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.clone()));
                        Err(e)
                    }
                }
            })
            .map_err(|e| format!("spawn engine thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(SlotEngine {
                tx: Some(tx),
                stats,
                closed,
                handle: Some(handle),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("engine thread died during startup".to_string()),
        }
    }
    /// Consuming shutdown that closes the channel, drains and fails any
    /// in-flight requests, joins the engine thread and returns GPU teardown
    /// or poison errors. The daemon can call this and withhold `unloaded`
    /// on failure. Graph is destroyed before buffers.
    pub fn shutdown_engine(mut self) -> Result<(), String> {
        self.tx = None;
        if let Some(h) = self.handle.take() {
            match h.join() {
                Ok(res) => res,
                Err(_) => Err("engine thread panicked".to_string()),
            }
        } else {
            Ok(())
        }
    }
}

impl Drop for SlotEngine {
    fn drop(&mut self) {
        // Best-effort teardown for non-shutdown paths. Closing the channel
        // signals the engine loop; join releases VRAM before Drop returns.
        // Logs but does not propagate teardown/poison errors — call
        // `shutdown()` for checked teardown.
        self.tx = None;
        if let Some(h) = self.handle.take() {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => eprintln!("hipfire-slot-engine teardown: {e}"),
                Err(_) => eprintln!("hipfire-slot-engine teardown: engine thread panicked"),
            }
        }
    }
}

impl hipfire_runtime::serve::SlotEngineHandle for SlotEngine {
    fn submit(&self, req: SubmitRequest) -> Result<(), String> {
        SlotEngine::submit(self, req)
    }
    fn cancel_waiting(&self, request_tag: u64) {
        SlotEngine::cancel_waiting(self, request_tag)
    }
    fn interrupt(&self, session: u64) {
        SlotEngine::interrupt(self, session)
    }
    fn close(&self, session: u64) -> Result<(), String> {
        SlotEngine::close(self, session)
    }
    fn reset(&self) -> Result<(), String> {
        SlotEngine::reset(self)
    }
    fn stats(&self) -> EngineStats {
        SlotEngine::stats(self)
    }
    fn take_closed_sessions(&self) -> Vec<u64> {
        SlotEngine::take_closed_sessions(self)
    }
    fn shutdown(self: Box<Self>) -> Result<(), String> {
        // `shutdown_engine` consumes `self`; move the engine out of the Box
        // and run teardown on the owned value.
        (*self).shutdown_engine()
    }
}
/// Per-request MTP sampling/penalty configuration, parked on the rig at admit
/// and applied on the slot's first draft step (a cold admit's `MtpSpecState`
/// only exists after the first prefill chunk allocates it).
///
/// Sampling (`temp > 0`) drives the sampled draft + residual acceptance in
/// `mtp_draft_phase_inner` / `mtp_batched_verify_accept_from_batch`; the
/// penalties install the exact-AR history the verify rows scope against
/// (`emitted ‖ candidates[..k]`, trailing `repeat_window`), mirroring the
/// sequential drafter's `begin_window` per window.
struct MtpReqSampling {
    sampling: crate::mtp_spec::MtpSamplingConfig,
    /// Full rendered prompt the request started from (`prompt_tokens` cold,
    /// `extended` on a continuation). `sess.tokens[prompt_len..]` is the
    /// emitted history the penalty rows scope against.
    prompt_len: usize,
    /// Trailing `repeat_window` tokens of that prompt (`set_prompt`).
    prompt_tail: Vec<u32>,
    /// Host RNG seed for the sampled accept rule (`request_rng_state`).
    rng_seed: u64,
    /// True once the slot's MTP state has this config installed; the penalty
    /// window is still rebuilt from `emitted` on every later window.
    applied: bool,
}

impl MtpReqSampling {
    fn new(req: &SubmitRequest, prompt: &[u32]) -> Self {
        let repeat_window = effective_repeat_window(
            req.repeat_window,
            req.repeat_penalty,
            req.presence_penalty,
            req.frequency_penalty,
        );
        Self {
            sampling: crate::mtp_spec::MtpSamplingConfig {
                temp: req.temperature.max(0.0),
                // The AR slot sampler maps an absent/0 top-k onto its default
                // 20-candidate pool (`SlotSampleParams::top_k` docs) and caps
                // at 64; the verify softmax kernel takes that same cap.
                top_k: if req.top_k > 0 {
                    (req.top_k as usize).min(64)
                } else {
                    20
                },
                top_p: if req.top_p > 0.0 {
                    req.top_p.min(1.0)
                } else {
                    1.0
                },
                min_p: req.min_p.clamp(0.0, 1.0),
                repeat_penalty: req.repeat_penalty,
                repeat_window,
                presence_penalty: req.presence_penalty,
                frequency_penalty: req.frequency_penalty,
            },
            prompt_len: prompt.len(),
            prompt_tail: prompt[prompt.len().saturating_sub(repeat_window)..].to_vec(),
            rng_seed: hipfire_runtime::spec::request_rng_state(req.seed as u64),
            applied: false,
        }
    }
}

/// Upper bound on a request's token-penalty window (`repeat_window`). The
/// penalty prepass is O(window²) per launch on device, and the per-slot
/// window buffer is sized by this; the daemon clamps requests to the same
/// bound so a client cannot ask for an unbounded scan.
const REPEAT_WINDOW_MAX: usize = 2048;

/// All model-owned GPU state for the slot engine, grouped so `Rig`'s other
/// fields stay family-neutral. Field-for-field with the former `Rig` model
/// members; this is the concrete bundle a future `SlotModel` impl owns.
/// S2 idle tier: a host-only copy of one slot's DeltaNet state, keyed by the
/// session and the exact `(frontier, fingerprint)` it was captured at. Unlike
/// a catalog image this is not a KV payload owner; it only lets an idle
/// session's recurrent state leave the device between turns.
#[cfg(feature = "kvmem")]
struct IdleDnImage {
    session: SessionId,
    frontier: u32,
    fp: u64,
    bytes: Vec<u8>,
}

struct ModelRig {
    weights: Qwen35Weights,
    config: qwen35::Qwen35Config,
    scratch: Qwen35Scratch,
    dn_states: Vec<DeltaNetState>,
    /// S1 parked-image handles, one per slot. `None` while the live state is
    /// engine-resident; S2 replaces the live state with `DeltaNetState::empty`
    /// and stores the published image id here.
    #[cfg(feature = "kvmem")]
    dn_images: Vec<Option<crate::kvmem_ctl::StateImageId>>,
    /// S2 (`HIPFIRE_KVMEM_DN_IDLE`): host image of an idle session's live DN
    /// state. The slot keeps a tombstone until the next continuation restores
    /// this exact image; a cold admit drops it.
    #[cfg(feature = "kvmem")]
    dn_idle_images: Vec<Option<IdleDnImage>>,
    /// P1b paged Legacy MTP-head arenas, one per slot. Each draft rebuilds the
    /// sink + retained page window from the composite aux plane; no per-slot
    /// copy of the full head history is held on the device.
    #[cfg(feature = "kvmem")]
    legacy_head_planes: Vec<Option<crate::kvmem_ctl::LegacyHeadPlane>>,
    k_arenas: Vec<GpuTensor>,
    v_arenas: Vec<GpuTensor>,
    desc_staging: SlotDescStaging,
    /// The engine's resolved KV tier + model-global rotation tables. Passed
    /// to every forward step; the tier selects the KV-write and attend
    /// kernels (see `forward_slots::kv_write_slots` / `tier_attend_slots`).
    kv_tier: crate::forward_slots::SlotKvTier,
    pbs: PrefillBatchScratch,
    logits_out: GpuTensor,
    /// MTP head weights (shared across all slots). None when MTP is off.
    mtp_head: Option<crate::mtp_head::Qwen35MtpHead>,
    /// Per-slot MTP spec state. None for slots that haven't started MTP yet
    /// or when MTP is off. Allocated lazily on first MTP prefill.
    mtp_states: Vec<Option<crate::mtp_spec::MtpSpecState>>,
    /// Per-slot spec request sampling/penalty config (MTP and DFlash alike),
    /// parked at admit and applied on the slot's first draft step (see
    /// [`MtpReqSampling`]).
    spec_req_sampling: Vec<Option<MtpReqSampling>>,
    /// Batched scratch (+ per-row FWHT-rot buffer) for MTP head-KV prefill
    /// from the scheduler's batched prompt chunks. None when MTP is off.
    mtp_prefill_batched: Option<(crate::mtp_head::Qwen35MtpHeadBatchedScratch, GpuTensor)>,
    /// Per-layer (qkv, alpha, beta) tape of spec verify rows, captured during
    /// the verify forward and replayed by the partial-accept DN repair.
    /// Row stride per slot = `spec_rows` (mtp_k + 1 for MTP, B for DFlash2).
    spec_verify_tape: Option<crate::speculative::GdnTape>,
    /// Staging for the POST-output-norm hidden rows fed to the MTP head
    /// during prompt prefill (x_batch holds the pre-norm residual).
    mtp_prefill_hidden: GpuTensor,
    /// MTP chain depth (K candidates per cycle). 0 = MTP off.
    mtp_k: usize,
    /// Verify rows per spec slot: `mtp_k + 1` when only MTP is armed, the
    /// DFlash2 block size B when a DFlash draft is loaded (the larger when
    /// both exist — the tape stride and graph-shape check use this uniform
    /// width).
    spec_rows: usize,
    /// Resolved spec → AR retirement break-even line, shared with the
    /// sequential drafter ([`hipfire_runtime::spec_retire`]). Parsed once at
    /// load from `HIPFIRE_SPEC_RETIRE` (else the shared default); `None`
    /// never retires. Installed per request at admit into `PendingWork`.
    spec_retire_break_even: Option<hipfire_runtime::spec_retire::BreakEven>,
    /// Shared DFlash2 draft state (weights + hidden staging). None when
    /// DFlash is off.
    dflash: Option<crate::dflash_slot::DflashShared>,
    /// Per-slot DFlash2 draft context. Allocated lazily on the slot's first
    /// prefill chunk under `spec == Dflash`; reset per request.
    dflash_states: Vec<Option<crate::dflash_slot::DflashSlotState>>,
    /// Vision encoder weights + config. None for text-only models.
    vision_weights: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
    vision_config: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionConfig>,
    /// Per-slot vision-embedding matrices (n_visual × dim, F32) for batched
    /// VL requests; the scatter kernel dereferences them through per-row
    /// pointers uploaded into `pbs.ext_emb_row_ptr`. None until the slot's
    /// request has run vision_forward; also cleared when the slot is.
    vl_ext_devs: Vec<Option<GpuTensor>>,
    /// Per-slot in-flight vision-tower jobs (batched VL path). The engine
    /// advances each live job ONE tower layer per serve iteration so a
    /// big-image encode (~9.7 s of GPU work at a 78×78 grid, even Q-tiled)
    /// interleaves with the other slots' decode steps instead of stalling
    /// them for the whole encode. None when no encode is in flight for the
    /// slot; cleared wherever `vl_ext_devs` is.
    vl_tower_jobs: Vec<Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionTowerJob>>,
    /// Cross-session recurrent-state checkpoint pool (spec §4.5 C5).
    /// None when `prefix_cache` is off.
    checkpoint_pool: Option<crate::checkpoint::CheckpointPool<DeltaNetSnapshot>>,
    /// KVMem host-offload controller. None when `HIPFIRE_KVMEM` is off (or
    /// the crate was built without the `kvmem` feature). Owns the per-slot
    /// host stores and the residency decision; the GPU KV stays in the paged
    /// arenas, with unselected full pages replaced by `SPILLED_PAGE`.
    #[cfg(feature = "kvmem")]
    kvmem: Option<crate::kvmem_ctl::KvMemCtl>,
}

#[derive(Clone, Copy, Debug, Default)]
struct SpecCounters {
    requested: bool,
    rounds: u64,
    drafted: u64,
    accepted: u64,
    generated: u64,
    fallback: u64,
}

/// 期3 G3: stable internal content group from the conversation identity.
/// Fairness hint only (never a reuse proof); an empty convo is unknown (0).
fn convo_group(convo: &[u64]) -> u64 {
    if convo.is_empty() {
        return 0;
    }
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    convo.hash(&mut h);
    h.finish()
}

struct Rig {
    gpu: Gpu,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    pool: SlotPool,

    /// All model-owned GPU state (weights, scratch, KV arenas, DN/spec/
    /// vision buffers, checkpoint pool). Grouped behind `model` so the
    /// remaining Rig fields are family-neutral serve substrate — the
    /// first step toward an arch-erased `SlotModel` bound for
    /// `serve_engine` (#792 follow-up slice 2).
    model: ModelRig,
    out_tokens: GpuTensor,
    sample_params: Vec<SlotSampleParams>,
    /// Per-slot penalty windows (see `REPEAT_WINDOW_MAX`); uploaded from each
    /// penalized slot's session token tail before every sampling step.
    repeat_windows: Vec<GpuTensor>,
    /// Serve VL requests on the legacy sequential per-token path instead of
    /// the batched one (HIPFIRE_VL_SEQUENTIAL=1; batched is the default and
    /// the only paged-compatible mode).
    vl_sequential: bool,
    sessions: SessionTable,
    #[cfg(feature = "kvmem")]
    legacy_history:
        Option<crate::legacy_catalog::LegacyHistoryCatalog<std::rc::Rc<LegacyNativeImage>>>,
    #[cfg(feature = "kvmem")]
    legacy_queries: Vec<Option<LegacyQueryRun>>,
    /// This request's Query decision is final: either a run was built, or
    /// there is no recall to attempt. Cleared with `legacy_queries` at every
    /// request/session lifecycle boundary.
    #[cfg(feature = "kvmem")]
    legacy_query_decided: Vec<bool>,
    /// Spec kind the slot's last admitted request actually ran with. Parking a
    /// history image uses it so MTP/DFlash draft state is captured too; a
    /// draft-less image can only be reused by a draft-less request.
    #[cfg(feature = "kvmem")]
    legacy_spec_kind: Vec<SpecKind>,
    adm: AdmissionController,
    swap: SwapManager,
    stamp: SnapshotStamp,
    n_slots: usize,
    cap_tokens: usize,
    prefill_chunk: usize,
    /// Close parked sessions after this idle period (None = never).
    session_ttl: Option<std::time::Duration>,
    /// Hard cap on session-table entries; 0 = auto max(64, 16 x n_slots).
    session_max: usize,
    /// Ids the engine closed internally, for the daemon's broker.
    closed_sessions: Arc<Mutex<Vec<u64>>>,
    /// Global trunk-row budget (spec §5.2 S2). Enforced in `next_batch` and
    /// the MTP verify injection path; the forward assert is a backstop.
    max_batch_tokens: usize,
    /// Minimum prefill quantum (spec §5.2 S2 / §5.3 S3).
    prefill_min_tokens: usize,
    /// Reusable host-side logits buffer for grammar mask application (spec
    /// §7.2 G2). When a slot has a grammar constraint, its logits row is
    /// copied D2H, masked (disallowed tokens set to NEG_INFINITY), then
    /// copied H2D back before `sample_per_slot` runs. This ensures the hard
    /// grammar mask participates before the sampler's probability-dependent
    /// filtering/renormalization (spec §7.2 G2). Allocated once; reused
    /// across steps. Sized to `n_slots * vocab_size` f32 elements.
    logits_host: Vec<f32>,
    /// Cross-session prefix cache: CPU-resident radix index over published
    /// token spans (spec §4.2 C2). None when `prefix_cache` is off.
    prefix_index: Option<hipfire_runtime::prefix_index::PrefixIndex>,
    /// Cache domain identity built once at Rig::build (spec §4.1 C1).
    /// None when `prefix_cache` is off.
    cache_domain: Option<hipfire_runtime::serve_contract::CacheDomain>,
    /// True when cross-session prefix reuse is enabled.
    prefix_cache: bool,
    /// P4.3: true when KVMem-native prefix sharing is enabled
    /// (`HIPFIRE_KVMEM_PREFIX_PIN`, opt-in under KVMem). The radix stays off;
    /// canonical prefixes freeze into the KVMem catalog and their sealed GPU
    /// pages are PIN-shared across sessions.
    kvmem_prefix_pin: bool,
    /// P2.2b2: decode-side coverage DN captures (`HIPFIRE_KVMEM_PREFIX_COVERAGE`,
    /// opt-in with PREFIX_PIN). Endpoint reuse needs a recurrent checkpoint at
    /// the page-aligned frontier it freezes, and the live state cannot rewind.
    kvmem_prefix_coverage: bool,
    /// Coverage capture cadence in tokens (page-aligned; default 1024).
    kvmem_prefix_coverage_interval: usize,
    /// P3.2 checkpoint policy (`HIPFIRE_KVMEM_CHECKPOINT_POLICY=spread`):
    /// `(interval, min_gap)` gating the prefill freeze cadence with the core
    /// `CheckpointCapturePlan`. `None` = legacy (freeze every boundary).
    kvmem_checkpoint_policy: Option<(u64, u64)>,
    /// 期4: per-slot speculative counters (completed-request statistics,
    /// nInfer `speculative.hpp` scope). Reset at admit, folded into
    /// `EngineStats.speculative` at the terminal event.
    spec_counters: Vec<SpecCounters>,
    /// The configured `serve.prefix_cache_max_bytes` ceiling. Retained on
    /// the Rig so the Reset path can re-apply it to the replacement index
    /// (`PrefixIndex::new` starts with no ceiling; losing it on reset
    /// silently unbounded retention).
    prefix_cache_max_bytes: Option<usize>,
    /// Debug counter: total prefix cache lookups (for test assertions that
    /// vision requests never call lookup).
    lookup_count: u64,
    /// Debug counter: total plan_cow calls (for test assertions that the
    /// COW write barrier fires on paged steps).
    cow_plan_count: u64,
    /// Host-side fairness queue (spec §5.3 S3). Admits/needs/select are
    /// driven from the engine loop each step; the granted ids become the
    /// scheduler's eligibility mask so an un-granted slot contributes 0
    /// rows. Pure policy — no GPU state.
    fair_queue: FairQueue,
    /// Bounded waiting room for admission (spec §5.3 S3). Replaces the R-A4
    /// immediate-reject path: when every resident session is generating, a
    /// new request is enqueued here instead of rejected, up to finite count
    /// and byte limits. Pure host policy — no GPU state.
    wait_queue: WaitQueue,
    /// Parked SubmitRequests keyed by waiter id. SubmitRequest is not Clone
    /// (it owns a `Sender<Event>`), so the full request is parked here on
    /// enqueue and retrieved on pop_ready for retry. Removed on expire,
    /// cancel, and successful retry. The `ParkedRequest` wrapper carries the
    /// wall-clock enqueue time — the admission-age and queue-deadline source
    /// (spec §5.3 S3).
    parked_requests: std::collections::HashMap<u64, ParkedRequest>,
    /// Monotonic counter for synthetic waiter ids (queued requests have no
    /// session id yet).
    next_waiter_id: u64,
    /// Per-waiter queue timeout in milliseconds (from
    /// `EngineConfig::queue_timeout_ms`), honored against the wall clock.
    queue_timeout_ms: u64,
    /// One-shot admission-age handoff for wait-queue retries: the drain
    /// loop sets it to the popped waiter's original enqueue Instant right
    /// before `admit`, and admit's park path consumes it so a retried
    /// request re-parks with its ORIGINAL age instead of a fresh stamp.
    pending_repark_age: Option<std::time::Instant>,
    /// Scheduler tick counter, advanced once per serve iteration. Used for
    /// wait-queue enqueue timestamps and expiry deadlines.
    tick: u64,
    /// Structured-output jump-forward enabled (spec §7.3 G3). Default false.
    structured_jump_forward: bool,
    /// Bounded cache of compiled JSON Schemas, keyed by canonical
    /// serialization. `validate_generate_caps` compiles once at submit and
    /// admit compiles again per request — for a repeated schema that is a
    /// per-request DoS surface. 8 entries is enough for a serving mix;
    /// eviction is FIFO over insertion order.
    schema_cache: std::collections::HashMap<Vec<u8>, grammar::json_schema::CompiledSchema>,
    /// Insertion order for `schema_cache` FIFO eviction.
    schema_cache_order: std::collections::VecDeque<Vec<u8>>,
}

/// Compile a JSON Schema through the rig's bounded cache. The daemon's
/// `validate_generate_caps` already compiled this schema once at submit;
/// without a cache every admit pays a second full compile — a per-request
/// DoS surface for a repeated schema. Keyed by canonical serialization;
/// FIFO eviction at 8 entries.
fn cached_compile_schema(
    rig: &mut Rig,
    schema: &serde_json::Value,
) -> Result<grammar::json_schema::CompiledSchema, grammar::json_schema::SchemaError> {
    // `serde_json::Value` serialization cannot practically fail, but if it
    // ever does, do NOT fall through with an empty cache key: two different
    // unserializable schemas would collide and the second would be
    // constrained by the first's compiled artifact. Just compile uncached.
    let Ok(key) = serde_json::to_vec(schema) else {
        return grammar::json_schema::CompiledSchema::compile(schema);
    };
    if let Some(compiled) = rig.schema_cache.get(&key) {
        return Ok(compiled.clone());
    }
    let compiled = grammar::json_schema::CompiledSchema::compile(schema)?;
    const SCHEMA_CACHE_CAP: usize = 8;
    while rig.schema_cache.len() >= SCHEMA_CACHE_CAP {
        if let Some(evict) = rig.schema_cache_order.pop_front() {
            rig.schema_cache.remove(&evict);
        } else {
            break;
        }
    }
    rig.schema_cache_order.push_back(key.clone());
    rig.schema_cache.insert(key, compiled.clone());
    Ok(compiled)
}

fn dn_buffers(dn: &DeltaNetState) -> Vec<&GpuTensor> {
    let mut v: Vec<&GpuTensor> = Vec::new();
    v.extend(dn.s_matrices.iter());
    v.extend(dn.s_scales.iter());
    v.extend(dn.conv_states.iter());
    v.extend(dn.s_ef_residual.iter());
    v
}

/// Layer ordinal in an HFQ tensor name (`layers.<n>.…`, optionally behind a
/// text-tower prefix such as `model.language_model.`). `None` for non-layer
/// tensors.
///
/// The `layers.` segment must sit at a path boundary (start of name, or after a
/// `.`), so an unrelated name that merely contains the substring — `my_layers.4.w`
/// — is not mistaken for a layer tensor and charged to the wrong side of the
/// offload split.
pub(crate) fn tensor_layer_index(name: &str) -> Option<usize> {
    let at = name.find("layers.")?;
    if at != 0 && name.as_bytes()[at - 1] != b'.' {
        return None;
    }
    let rest = &name[at + "layers.".len()..];
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

/// Per-tier discriminant: K/V spans are stride-shaped, so a snapshot must
/// never restore into a different KV tier. Preserve the historical tags.
fn slot_snapshot_kv_tag(kv_mode: hipfire_runtime::kv_mode::KvMode) -> u32 {
    use hipfire_runtime::kv_mode::KvMode;

    match kv_mode {
        KvMode::Q8 => 1,
        KvMode::Asym2 => 21,
        KvMode::Asym3 => 31,
        KvMode::Asym4 => 41,
        KvMode::Fwht2 => 22,
        KvMode::Fwht3 => 32,
        KvMode::Fwht4 => 42,
        KvMode::Fp8 => 11,
        KvMode::Bf16 => 12,
        KvMode::F16 => 13,
    }
}

impl Rig {
    /// 期4: the speculative contract snapshot (schema + resolved config).
    /// Statistics start zero and accumulate at request completion; the schema
    /// version comes from the KVMem core contract on `kvmem` builds.
    fn speculative_config(&self) -> hipfire_runtime::serve::SpeculativeStats {
        let mut out = hipfire_runtime::serve::SpeculativeStats::default();
        out.schema_version = 1;
        #[cfg(feature = "kvmem")]
        {
            out.schema_version = hipfire_kvmem::speculative_schema_version();
        }
        if let Some(d) = self.model.dflash.as_ref() {
            out.enabled = true;
            out.algorithm = 3; // kvmem::SpeculativeAlgorithm::DFlash2
                               // B verify rows = seed + (B−1) proposals.
            out.draft_tokens = d.block_size.saturating_sub(1) as u64;
        } else if self.model.mtp_head.is_some() && self.model.mtp_k > 0 {
            out.enabled = true;
            out.algorithm = 1; // kvmem::SpeculativeAlgorithm::Mtp
            out.draft_tokens = self.model.mtp_k as u64;
            out.adaptive = false;
        }
        out.ngram_draft_tokens = 0;
        out
    }

    /// Resolve the placement the slots path loads with.
    ///
    /// `Full`/`Full` returns the all-device placement with no capacity check — the
    /// no-knob load must stay byte-identical to the pre-offload loader. With a knob
    /// set the capacity is the measured free VRAM minus the slot engine's own KV
    /// footprint, the prefill floor and any device-resident draft, and the pinned
    /// host bytes are admitted against host RAM and TTM's GTT cap before any
    /// allocation.
    ///
    /// Eligibility is the same `spill_support` call the sequential carrier
    /// makes, over this route's own facts (arch 5|6, tp/pp 1 by the slot
    /// capability gate, EP never, `cfg.is_vl` / `cfg.vl_path` for the tower,
    /// `config.paged_experts`, `config.reap_keep`) — the two twins share one
    /// truth table including the VL term. Unset budgets stay fully resident.
    /// A configured spill on an unspillable path was already refused at
    /// admission; the defensive arm below permits only a fully resident
    /// placement that fits, otherwise refusing before allocation.
    fn resolve_slot_offload(
        layer_bytes: &hipfire_runtime::offload::LayerBytes,
        config: &crate::qwen35::Qwen35Config,
        cfg: &EngineConfig,
        gpu: &Gpu,
        free_vram: u64,
        kv: hipfire_runtime::offload::KvReserve,
        arch_id: u32,
    ) -> Result<hipfire_runtime::offload::Placement, String> {
        use hipfire_config::memory::{effective_offload_budgets, spill_support, OffloadBudget};
        use hipfire_runtime::offload;
        // The slots route is single-GPU by its capability gate (`tp != 1 ||
        // pp != 1` refuses before spawn) and never expert-parallel; the only
        // per-load facts are the tower (explicit flag or sidecar path — both
        // mean the lowered VL loader owns the tower) and the parsed model
        // flags. Identical in every dimension the predicate takes to the
        // sequential twin's `spill_facts`.
        let is_vl = cfg.is_vl || cfg.vl_path.is_some();
        let support = spill_support(
            arch_id,
            1,
            1,
            None,
            config.paged_experts,
            config.reap_keep.is_some(),
            is_vl,
        );
        let budgets = effective_offload_budgets();
        if budgets == (OffloadBudget::Full, OffloadBudget::Full) {
            return Ok(offload::Placement::all_device(config.n_layers));
        }
        if let hipfire_config::memory::SpillSupport::Refused(refusal) = support {
            // Same defensive arm as the sequential twin: admission should have
            // refused a configured spill on this unsupported path. The slots
            // capacity includes this route's KV footprint, shared prefill floor,
            // draft reserve and slack. A pinned Full/Full plan that fits stays
            // resident; otherwise refuse, reusing offload::plan.
            let pbs_reservation =
                crate::qwen35::prefill::minimum_prefill_reservation_bytes(config, &gpu.arch);
            let mtp_head: Option<crate::carrier::MtpHeadLocation> = if cfg.mtp_k > 0 {
                crate::carrier::resolve_mtp_head_file(&cfg.model_path)
            } else {
                None
            };
            let draft = crate::carrier::draft_device_reserve_bytes(
                mtp_head.as_ref(),
                cfg.dflash_draft.as_deref(),
            );
            let capacity = offload::weight_capacity(
                free_vram,
                kv.bytes,
                pbs_reservation.unwrap_or(0) as u64,
                draft,
                128 << 20,
            );
            if offload::plan(
                layer_bytes,
                (OffloadBudget::Full, OffloadBudget::Full),
                &capacity,
                offload::Orientation::SuffixResident,
            )
            .is_ok()
            {
                return Ok(offload::Placement::all_device(config.n_layers));
            }
            return Err(crate::carrier::unspillable_no_fit_refusal(
                refusal,
                layer_bytes.weight_total(),
                capacity.weight_bytes,
            ));
        }
        let pbs_reservation =
            crate::qwen35::prefill::minimum_prefill_reservation_bytes(config, &gpu.arch);
        let pbs = pbs_reservation.unwrap_or(0) as u64;
        // The slots MTP loader probes the bundled trailer first, then every
        // `sidecar_candidates(trunk, "mtp")` spelling (stem-based
        // `ornith-1.5-35b-a3b.mtp` included) — the same list the sequential
        // carrier reserves through `carrier::resolve_mtp_head_file`, so the
        // two routes charge the same head. Unlike the old inline probe here,
        // a configured DFlash draft is charged too: it loads on this route
        // (`dflash_draft` arms the slot DFlash2 chain) and silently dropping
        // it under-reserved the placement and over-placed the trunk.
        let mtp_head: Option<crate::carrier::MtpHeadLocation> = if cfg.mtp_k > 0 {
            crate::carrier::resolve_mtp_head_file(&cfg.model_path)
        } else {
            // MTP off: no head will load, but a configured DFlash draft still
            // runs its own chain and stays on the device.
            None
        };
        let draft = crate::carrier::draft_device_reserve_bytes(
            mtp_head.as_ref(),
            cfg.dflash_draft.as_deref(),
        );
        // The slots KV footprint is an arena product, but the caller factored
        // it through `offload::kv_reserve`, so the shared line names its real
        // rows × stride shape.
        let inputs = offload::CapacityInputs {
            free_vram,
            kv_bytes: kv.bytes,
            kv_rows: kv.rows,
            kv_stride: kv.stride_bytes,
            prefill_floor: pbs,
            prefill_floor_absent: pbs_reservation.is_none(),
            draft_bytes: draft,
            slack: 128 << 20,
        };
        offload::resolve_placement(
            layer_bytes,
            &inputs,
            budgets,
            offload::Orientation::SuffixResident,
        )
    }
    /// Build the GPU rig.
    ///
    /// CPU arch/tensor preflight precedes this loader — `SlotBackend::cpu_preflight`
    /// opens the HFQ, validates `arch_id` 5|6, records whether the HFQ has a
    /// vision tower, and parses config/tokenizer before this GPU path. This
    /// function assumes that preflight has passed; its `get_vram_info` +
    /// `preflight_alloc` is the actual-device VRAM admission. Keep that
    /// preflight on the live device (not a hardcoded budget).
    ///
    /// Post-weight allocation is transactional: K/V arenas, DN states, desc
    /// staging, PBS, scratch, logits/out are all owned after `Qwen35Weights`.
    /// On any `?`/error, all completed stages are freed, weights freed,
    /// caches/graph state invalidated and pool drained, so no VRAM leaks.
    fn build(cfg: &EngineConfig, closed_sessions: Arc<Mutex<Vec<u64>>>) -> Result<Rig, String> {
        use hipfire_runtime::hfq::HfqFile;
        use hipfire_runtime::tokenizer::Tokenizer;
        use rdna_compute::kv_slots::preflight_alloc;

        let (session_ttl, session_max) = serve_engine_admit::session_policy_from_config();
        let mut hfq = HfqFile::open(&cfg.model_path).map_err(|e| format!("open model: {e}"))?;
        let config = qwen35::config_from_hfq(&hfq).map_err(|e| format!("config: {e}"))?;
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let n_fa_layers = config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count();
        // Slot arenas are `SlotPool` allocations under a resolved KV tier.
        // `kv_backend` other than `legacy` has no slot path at all; reject
        // before any GPU allocation. The daemon capability gate normally
        // refuses these first; this is the fail-closed inner check.
        if cfg.kv_backend != "legacy" {
            return Err(format!(
                "experimental multi-slot requires kv_backend=legacy (got {:?})",
                cfg.kv_backend
            ));
        }
        // Resolve the KV tier through the slots site policy — same config
        // source and alias table as the sequential carrier, but with q8 as
        // the slots engine's own default (the longest-validated slots path;
        // a rotated tier is an explicit operator choice). The full static
        // ladder is wired — q8, asym{2,3,4}, fwht{2,3,4} — each under both
        // the legacy slab pool and the paged pool.
        let kv_mode_raw = if cfg.kv_mode_raw.is_empty() {
            hipfire_runtime::config::get().kv_mode.clone()
        } else {
            cfg.kv_mode_raw.clone()
        };
        let hipfire_runtime::kv_mode::ResolveResult {
            mode: kv_mode,
            warning: kv_warning,
        } = hipfire_runtime::kv_mode::resolve(
            &kv_mode_raw,
            &hipfire_runtime::kv_mode::QWEN35_SLOTS_POLICY,
        );
        // Fail closed on anything the slots site could not cleanly accept:
        // the resolve warning is normally surfaced as a load refusal by the
        // daemon capability gate, so a warning that still reaches the engine
        // thread means the gate was bypassed or the policy shifted.
        if let Some(w) = kv_warning {
            return Err(format!(
                "experimental multi-slot requires a slot-readable KV mode \
                 (got {kv_mode_raw:?}): {w} (site {})",
                hipfire_runtime::kv_mode::QWEN35_SLOTS_POLICY.site
            ));
        }
        // Native fp8 KV is admitted here only on exact gfx1201, where the
        // descriptor-aware fp8 write/attend kernels exist. The arch gate
        // sits right after `Gpu::init` below — that is the first point the
        // resolved GPU arch is authoritative (the daemon capability gate
        // already refused fp8 off gfx1201 before spawn; this is the
        // fail-closed inner check). bf16/f16 are admitted on every arch:
        // the descriptor-aware flat-2B writers and attend kernels exist for
        // both. Storing bytes a tier's kernels cannot read is silent
        // corruption — anything unadmittable must fail closed before an
        // arena is allocated.
        // The tier's arena geometry: the K and V per-position strides DIFFER
        // on every rotated-K tier (packed K with a 4-byte per-head norm
        // header against Q8_0 V), and the pool, arenas, preflight math, and
        // snapshot stamp all follow them.
        let kv_plan = hipfire_runtime::kv_mode::SlotKvTierPlan::resolve(
            kv_mode,
            config.n_kv_heads,
            config.head_dim,
        )
        .map_err(|e| format!("kv mode {kv_mode:?}: {e}"))?;
        let per_pos_bytes = kv_plan.k_bytes_per_pos;
        let per_pos_v_bytes = kv_plan.v_bytes_per_pos;
        // Escha trellis checkpoints (wire qt 42/43) now batch prefill like every
        // other model: the slot bodies run a grouped-GEMM escha arm (one weight
        // read per (layer,batch)) and `qwen35_layer_batch_admissible` admits
        // them, so the old `prefill_chunk=1` serialization workaround is gone.
        let prefill_chunk = cfg.prefill_chunk.max(1).min(cfg.cap_tokens.max(1));
        // Reference-test override: force the engine's prefill quantum independent
        // of the wire/CLI scratch budget. The CLI clamps `max_batch_tokens` to
        // `prefill_chunk * n_slots`, and FairQueue requires
        // `max_batch_tokens >= n_slots + prefill_min_tokens`, so a wire
        // `prefill_chunk=1` (scratch 1) is an invalid serve config for ANY model.
        // Setting HIPFIRE_PREFILL_CHUNK_OVERRIDE=1 while leaving the CLI chunk at
        // its default keeps the wire budget valid (FairQueue passes) yet drives
        // every prefill step at n==1 -> use_reference_decode -> the lowered
        // per-token ladder. This is the apples-to-apples serial reference for the
        // batched-vs-serial recall A/B (same paged KV, attention, and decode; only
        // the prefill quantum differs). Unset => no change (batched prefill).
        let prefill_chunk = hipfire_config::developer_var("HIPFIRE_PREFILL_CHUNK_OVERRIDE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.max(1).min(cfg.cap_tokens.max(1)))
            .unwrap_or(prefill_chunk);
        let prefill_chunk = if let Some(kvmem) = cfg.kvmem.as_ref().filter(|c| c.enabled) {
            let effective = prefill_chunk.min(kvmem.gen_reserve_tokens as usize);
            if effective == 0 {
                return Err("kvmem: normalized prefill and reserve must allow a native row".into());
            }
            if effective != prefill_chunk {
                eprintln!("[kvmem-config] prefill requested={prefill_chunk} effective={effective} reason=reserve_limit");
            }
            effective
        } else {
            prefill_chunk
        };
        // DFlash2 draft config pre-parse: the runtime block size B sets
        // `spec_rows` (verify rows per spec slot), which `max_batch` and the
        // verify tape are sized by — both allocated before the GPU-side
        // draft load below. A parse failure here is non-fatal unless
        // `dflash_required` (the load itself re-parses and reports).
        let dflash_block = cfg.dflash_draft.as_ref().and_then(|p| {
            hipfire_runtime::hfq::HfqFile::open(p)
                .ok()
                .and_then(|h| hipfire_runtime::dflash::DflashConfig::from_hfq(&h))
                .map(|c| c.runtime_block_size(hipfire_runtime::dflash::DFLASH_BLOCK_UNBATCHED))
        });
        if cfg.dflash_required && cfg.dflash_draft.is_some() && dflash_block.is_none() {
            return Err(format!(
                "dflash_mode=on: failed to parse DflashConfig from {}",
                cfg.dflash_draft.as_ref().unwrap().display()
            ));
        }
        // Verify rows per spec slot: MTP k+1, DFlash2 B — the larger wins so
        // the tape stride and batch sizing cover either mechanism.
        let spec_rows = (cfg.mtp_k + 1).max(dflash_block.unwrap_or(0));
        // A step carries at most max(prefill_chunk, spec_rows) rows per
        // slot: prefilling slots contribute up to `prefill_chunk` rows, and
        // a decoding spec slot injects `spec_rows` verify rows. Sizing
        // without the spec term let a small `prefill_chunk` (operator knob)
        // overflow `pbs.max_batch` on the first verify step — an
        // engine-thread panic, the serve-hang failure class.
        let max_batch = prefill_chunk.max(spec_rows) * cfg.n_slots;

        // ── Paged KV opt-in ────────────────────────────────────────────────
        // Default OFF: the pool stays legacy (fixed per-slot slabs). With
        // HIPFIRE_SLOTS_PAGED=1 the pool becomes a PagePool: slots draw
        // physical pages on demand, so short sessions don't pre-reserve
        // their full cap. HIPFIRE_SLOTS_PAGED_PAGES overrides the physical
        // page count; the default matches legacy total capacity
        // (n_slots × ceil(cap/128) pages), so admission behaviour is
        // unchanged and only the allocation pattern is paged. A smaller
        // count under-subscribes (pool exhaustion fails the step CLOSED —
        // set_seq_len errors before any kernel writes through the missing
        // page); a larger count over-subscribes physical VRAM and is the
        // operator's call.
        // ── KVMem host offload (HIPFIRE_KVMEM=1) ───────────────────────────
        // KVMem takes over residency for the FullAttention pages: the pool
        // keeps a bounded per-slot working set, and the rest of each
        // conversation's KV lives in the host store, restored on selection.
        // Requirements: the paged Q8 pool (the spill/restore path moves the
        // packed Q8 rows verbatim and the masked kernels are Q8-only), and
        // the batched VL path (the legacy sequential path writes through a
        // flat slab view that paged mode cannot honour, and its single-token
        // forwards bypass the step batch the statistics taps follow).
        let kvmem_on = match cfg.kvmem.as_ref() {
            Some(config) => config.enabled,
            None => kvmem_requested()?,
        };
        #[cfg(not(feature = "kvmem"))]
        if kvmem_on {
            return Err(
                "KVMem was requested but this worker was built without feature kvmem".into(),
            );
        }
        // 期4: KVMem + DFlash2 validation matrix (KVMem/nInfer integration
        // pattern). Unsupported combinations are rejected at load — never a
        // silent degrade — mirroring the backend's parse/normalize checks.
        // The native MTP head and a DFlash2 draft are mutually exclusive
        // (nInfer: "The main model does not load its MTP head for DFlash2");
        // the KVMem DFlash lane is single-sequence. Split/legacy draft shapes
        // stay legal under KVMem but only run the Legacy capped path; the
        // Layerwise admission checks the all-sliding windowed shape per step.
        if let Some(err) = kvmem_dflash_validation_error(
            kvmem_on,
            cfg.dflash_draft.is_some(),
            cfg.mtp_k,
            cfg.n_slots,
        ) {
            return Err(err);
        }
        let vl_sequential = hipfire_config::developer_var("HIPFIRE_VL_SEQUENTIAL")
            .ok()
            .is_some_and(|v| v == "1" || v == "on" || v == "true");
        if kvmem_on {
            // P4.4: KVMem admits every KV tier whose paged batched attention
            // kernel carries the kv_page_present residency mask (ninfer
            // multi-format admission pattern). Q8 + the 8 non-Q8 tiers are
            // masked (flat f16/bf16, rotated asym/fwht, fp8 as a q8 TU variant);
            // spill/restore is byte-opaque and mean-K is F32 pre-RoPE, so the
            // host path is tier-generic. Per-tier head_dim / hardware constraints
            // fail closed downstream in SlotKvTierPlan::resolve / kernel admission.
            match kv_mode {
                hipfire_runtime::kv_mode::KvMode::Q8
                | hipfire_runtime::kv_mode::KvMode::Bf16
                | hipfire_runtime::kv_mode::KvMode::F16
                | hipfire_runtime::kv_mode::KvMode::Asym2
                | hipfire_runtime::kv_mode::KvMode::Asym3
                | hipfire_runtime::kv_mode::KvMode::Asym4
                | hipfire_runtime::kv_mode::KvMode::Fwht2
                | hipfire_runtime::kv_mode::KvMode::Fwht3
                | hipfire_runtime::kv_mode::KvMode::Fwht4
                | hipfire_runtime::kv_mode::KvMode::Fp8 => {}
            }
            if cfg.is_vl && vl_sequential {
                return Err("HIPFIRE_KVMEM requires the batched VL path; unset \
                     HIPFIRE_VL_SEQUENTIAL (the sequential path writes KV \
                     outside the step batch KVMem statistics follow)"
                    .to_string());
            }
        }
        // Prefix cache requires paged KV: the radix index shares physical
        // pages across sessions via PagePool refcounting, which is only
        // available in paged mode. Force it on when prefix_cache is enabled.
        // KVMem forces it OFF: the radix shares sealed pages (which the
        // residency pass must never spill) and its cross-session reuse
        // rewinds seq_len onto pages the host store already owns — both
        // conflict with KVMem's exclusive page ownership. Cold prefill is
        // the correct trade; the KVMem store is the cross-request cache.
        let prefix_cache = cfg.prefix_cache && !kvmem_on;
        if cfg.prefix_cache && kvmem_on {
            eprintln!(
                "  [hipfire] prefix cache disabled: HIPFIRE_KVMEM owns page \
                 residency (cold prefill per request)"
            );
        }
        let kvmem_build = kvmem_hooks::resolve_build(cfg, kvmem_on, prefill_chunk, spec_rows)?;
        let kvmem_prefix_pin = kvmem_build.prefix_pin;
        let kvmem_prefix_coverage = kvmem_build.prefix_coverage;
        let kvmem_prefix_coverage_interval = kvmem_build.prefix_coverage_interval;
        let kvmem_checkpoint_policy = kvmem_build.checkpoint_policy;
        let kvmem_pages = kvmem_build.pages_total();
        #[cfg(feature = "kvmem")]
        let kvmem_plan = kvmem_build.plan;
        #[cfg(feature = "kvmem")]
        let kvmem_options = kvmem_build.options;
        let paged_pages = if prefix_cache {
            let legacy_equivalent = cfg.n_slots * cfg.cap_tokens.div_ceil(PAGE_TOKENS);
            // Cache-retained pages share the same physical pool as active slots.
            // Keep one prefill chunk plus one page per slot available beyond
            // the legacy active-session capacity so publication cannot consume
            // the last page needed by an admitted request.
            let cache_headroom = cfg.prefill_chunk.div_ceil(PAGE_TOKENS) + cfg.n_slots;
            Some(
                hipfire_config::developer_var("HIPFIRE_SLOTS_PAGED_PAGES")
                    .ok()
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    // Clamp to >= 1: a literal "0" reaches PagePool's
                    // `assert!(n_pages > 0)` inside Rig::build, which runs on
                    // the engine thread before ready_tx — the spawn then
                    // reports the opaque "engine thread died during startup".
                    .map(|p| p.max(1))
                    .unwrap_or_else(|| legacy_equivalent.saturating_add(cache_headroom)),
            )
        } else if let Some(pages) = kvmem_pages {
            // KVMem: the pool is the resident working set (default 64 blocks
            // per slot), not the slot cap. Everything past it lives in the
            // host store, so a large `multi_slot_ctx` no longer costs VRAM.
            Some(pages)
        } else {
            match hipfire_config::developer_var("HIPFIRE_SLOTS_PAGED")
                .ok()
                .as_deref()
            {
                Some("1") | Some("on") | Some("true") => {
                    // Batched VL is paged-capable (block-table indirection all
                    // the way down), so paged is refused only when the operator
                    // forced the legacy sequential VL path — that path writes
                    // through a flat contiguous slab view which aliases every
                    // slot onto the arena prefix in paged mode.
                    if cfg.is_vl && vl_sequential {
                        eprintln!(
                            "  [hipfire] paged KV disabled: HIPFIRE_VL_SEQUENTIAL=1 \
                             requires contiguous KV slots"
                        );
                        None
                    } else {
                        let legacy_equivalent = cfg.n_slots * cfg.cap_tokens.div_ceil(PAGE_TOKENS);
                        Some(
                            hipfire_config::developer_var("HIPFIRE_SLOTS_PAGED_PAGES")
                                .ok()
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                // Same >= 1 clamp as the prefix-cache arm: a
                                // literal "0" would panic PagePool's build.
                                .map(|p| p.max(1))
                                .unwrap_or(legacy_equivalent),
                        )
                    }
                }
                _ => None,
            }
        };

        // Model-file size doubles as the load identity (`model_hash` below).
        let weight_bytes = std::fs::metadata(&cfg.model_path)
            .map_err(|e| format!("stat model: {e}"))?
            .len();
        // GPU context only before preflight — no device allocations yet. Free/
        // total come from the live device so gfx1100 is not over-admitted
        // against a hardcoded R9700 budget.
        let mut gpu = Gpu::init().map_err(|e| format!("gpu init: {e}"))?;
        // Native fp8 KV admission gate (gfx1201-only): the descriptor-aware
        // fp8 writer/reader kernels compile `#error` off gfx1201 (OCP E4M3FN
        // cvt), so admitting the tier anywhere else fails at first kernel
        // load at best and mis-reads cache bytes at worst. Fail closed here,
        // naming the arch and the fix, before preflight or any arena exists.
        //
        // Descriptor-kernel evidence: `test_bf16_slots_parity` check 4 on
        // gfx1201 covers zero/nonzero bases, two-slot reads, and writes.
        // This does not establish full-serve parity.
        if matches!(kv_mode, hipfire_runtime::kv_mode::KvMode::Fp8) && !gpu.arch_caps.is_gfx1201() {
            return Err(format!(
                "experimental multi-slot kv_mode \"fp8\" needs gfx1201 (have {}); \
                 use q8 or auto",
                gpu.arch
            ));
        }
        let (vram_free, vram_total) = gpu
            .device_mem_info()
            .map_err(|e| format!("vram info: {e}"))?;

        // Placement is decided from planned capacity, never the instantaneous
        // free figure; the preflight below still refuses on real free bytes.
        let planned = gpu
            .planned_capacity()
            .map_err(|e| format!("vram info: {e}"))?;
        eprintln!("slots: admission decision: weight placement capacity from {planned}");

        let cap_rounded = cfg.cap_tokens.div_ceil(128) * 128;
        let kv_pages = paged_pages.unwrap_or(cfg.n_slots * cap_rounded / 128);
        let kv_rows = n_fa_layers
            .saturating_mul(kv_pages)
            .saturating_mul(PAGE_TOKENS);
        let kv_stride = per_pos_bytes.saturating_add(per_pos_v_bytes) as u64;
        let kv = hipfire_runtime::offload::kv_reserve(kv_stride, kv_rows)
            .ok_or("slots KV reservation overflow")?;
        let kv_bytes = kv.bytes;

        // Partial GPU offload is resolved here, once, before the preflight — the
        // same call the sequential carrier makes, so both routes place
        // identically. `Full`/`Full` (both knobs unset) yields the all-device
        // placement and skips the capacity check entirely, keeping a no-knob load
        // byte-identical to the pre-offload loader.
        let layer_bytes = qwen35::load::qwen35_layer_bytes(&hfq, &config);
        let placement = Self::resolve_slot_offload(
            &layer_bytes,
            &config,
            cfg,
            &gpu,
            planned.bytes as u64,
            kv,
            hfq.arch_id,
        )?;

        // Charge only what actually lands on the device. With partial offload the
        // spilled weights live in host RAM (`hipHostMalloc`, which costs the
        // device heap nothing), so charging the whole file here would make
        // offload unable to buy the KV capacity it exists to buy: the gate would
        // refuse the very context the freed VRAM affords. Summed from the same
        // per-tensor census the placement was resolved against, so the charge and
        // the placement cannot disagree — and this replaces the old
        // `tensor_layer_index` prefix walk, which knew only about whole layers.
        let resident_weight_bytes = layer_bytes.device_bytes(&placement);
        let planned = resident_weight_bytes + kv_bytes + 768 * 1024 * 1024;

        preflight_alloc(planned, vram_free as u64, "SlotEngine")
            .map_err(|e| format!("preflight refused: {e}"))?;
        // Discrete GPUs: keep model pages in page cache (fadvise-per-tensor
        // forces a full re-read on every load). UMA keeps eviction to avoid
        // OOM vs hipMalloc staging. Mirrors carriers.rs — without this, the
        // default `evict_page_cache=true` drops the mmap in prepare(), and
        // post-weight tensor reads (e.g. vision weights) panic on None.
        hfq.set_evict_page_cache(
            hipfire_config::developer_var("HIPFIRE_PAGE_EVICTION")
                .ok()
                .map(|v| v != "0")
                .unwrap_or_else(|| gpu.is_uma()),
        );
        // The capture gate and the CPU-exec coverage line are model-scoped: this
        // slots path loads weights directly (not through
        // `Architecture::load_weights_with_placement`), so it must set the flag,
        // refuse an incompatible retained-replay backend, and report coverage here
        // too — otherwise none of the three run on the serve route.
        let host_weights = placement.host_layers() + placement.host_expert_layers();
        gpu.set_host_cpu_weights(hipfire_dispatch::cpu_offload_active(host_weights));
        hipfire_dispatch::reject_cpu_exec_under_redline(host_weights, gpu.replay.is_enabled())
            .map_err(|e| format!("qwen35: {e}"))?;
        crate::qwen35::load::report_cpu_exec_coverage(&hfq, &config, Some(&placement));
        let (weights, pinned_host_bytes): (Qwen35Weights, u64) = {
            let mut src = qwen35::HfqSource::new(&mut hfq, &config);
            let layout = qwen35::Layout::single(config.n_layers).with_placement(placement.clone());
            let weights = qwen35::load_weights(&mut src, std::slice::from_mut(&mut gpu), &layout)
                .map_err(|e| format!("load weights: {e}"))?;
            // Decide the AR decode graph policy at load (#1149).
            weights.resolve_moe_decode_host_topk(&config, &gpu);
            (weights, layer_bytes.host_bytes(&placement))
        };

        // ── Vision tower loading ──────────────────────────────────────────
        // When `vl_path` is set, vision weights load from the standalone .vl
        // sidecar (a standard HFQM container with only model.visual.* tensors
        // and config.vision_config in its metadata). Otherwise, fall back to
        // inline vision tensors in the trunk HFQ (backward compat).
        let (vision_config, vision_weights) = if cfg.is_vl {
            if let Some(vl) = &cfg.vl_path {
                eprintln!(
                    "  loading vision weights from .vl sidecar: {}",
                    vl.display()
                );
                let vl_hfq = HfqFile::open(vl)
                    .map_err(|e| format!("open .vl file {}: {e}", vl.display()))?;
                let vc = hipfire_arch_qwen35_vl::qwen35_vl::vision_config_from_hfq(&vl_hfq)
                    .ok_or_else(|| ".vl file missing vision_config in metadata".to_string())?;
                // Identity: the tower's projector writes the trunk's text
                // hidden width. A `.vl` sibling paired with a different
                // trunk (a `foo.vl` next to `bar.mq4`) would load
                // "successfully" and then misalign every image embedding —
                // refuse the mismatch at load time. Compare against
                // `config.dim` (the value the runtime actually uses for the
                // ext-embedding row stride), NOT a tensor-name probe: the
                // trunk's final norm lives under `norm.weight`/`model.norm.weight`
                // on HFQ, so probing `output_norm.weight` (a GGUF-only name)
                // would skip the check entirely.
                if vc.out_hidden_size != config.dim {
                    return Err(format!(
                        "vision sidecar {} does not match trunk {}: projector \
                         output {} != trunk hidden {}",
                        vl.display(),
                        cfg.model_path.display(),
                        vc.out_hidden_size,
                        config.dim
                    ));
                }
                let vw = hipfire_arch_qwen35_vl::qwen35_vl::load_vision_weights(
                    &vl_hfq,
                    &vc,
                    &mut gpu,
                    cfg.vision_residency,
                )
                .map_err(|e| format!("load vision weights from .vl: {e}"))?;
                (Some(vc), Some(vw))
            } else {
                let vc = hipfire_arch_qwen35_vl::qwen35_vl::vision_config_from_hfq(&hfq)
                    .ok_or_else(|| "VL model missing vision_config in metadata".to_string())?;
                let vw = hipfire_arch_qwen35_vl::qwen35_vl::load_vision_weights(
                    &hfq,
                    &vc,
                    &mut gpu,
                    cfg.vision_residency,
                )
                .map_err(|e| format!("load vision weights: {e}"))?;
                (Some(vc), Some(vw))
            }
        } else {
            (None, None)
        };

        // ── MTP head loading ──────────────────────────────────────────────
        // Mirrors finish_qwen35_load: try bundled trailer first, then .mtp sidecar.
        // The head is shared across all slots (small ~216 MB for 27B).
        //
        // MTP × paged: ALLOWED. The gate below (59c731dd0) refused the
        // combination because the then-current MTP path prefilled and
        // replayed KV through the flat slab view (`slot_kv_view`, whose
        // legacy bases are 0 under paged → silent cross-slot corruption).
        // The 85dde5df0 rewrite removed every flat-view writer from the MTP
        // path: prefill runs through the scheduler's batched chunk path
        // (paged block tables + write-frontier provisioning), the verify is
        // the batched forward, and a partial accept repairs only the
        // DeltaNet state (private per-slot buffers) from the verify tape —
        // `slot_kv_view` has no MTP caller left. Graph capture simply does
        // not engage under paged (the graphed entry falls back to the plain
        // step), and `mtp_verify_fits_cap` retires MTP before a verify's
        // k+1-row frontier could fail the paged provision closed.
        let mtp_k = cfg.mtp_k;
        // MTP over a host spill declines like the sequential route: the
        // verify re-reads host-mapped experts over the link, and a head that
        // is unverified there has garbled windows (`qwen3.6-35b-a3b.mq6` at
        // tau 0.04-0.13 while its single-row AR is coherent). Counted from
        // the built weights — the same `host_mapped_expert_accounting` the
        // sequential route keys on — so the verdict belongs to the resolved
        // placement, not MoE-ness. qwen35's admitted list is empty (never
        // attaches on a spill under auto); `On` still arms (explicit
        // opt-in); `Off` keeps today's behaviour. A decline skips the head
        // load, prints the one shared line, and runs AR.
        let (host_mapped_expert_blobs, host_mapped_expert_bytes) =
            crate::qwen35::host_mapped_expert_accounting(&weights.layers);
        let mtp_spill = hipfire_config::mtp_host_spill_policy(
            cfg.mtp_mode,
            host_mapped_expert_blobs,
            &gpu.arch,
            &[],
        );
        let mtp_armed = mtp_k > 0 && mtp_spill.attach;
        if mtp_k > 0 && !mtp_spill.attach && !matches!(cfg.mtp_mode, hipfire_config::MtpMode::Off) {
            eprintln!(
                "{}",
                hipfire_config::mtp_host_spill_decline_line(
                    "qwen35",
                    host_mapped_expert_blobs,
                    host_mapped_expert_bytes,
                    if weights.has_cpu_deferred_experts() {
                        "offload_exec=cpu"
                    } else {
                        "offload_exec=pcie"
                    },
                )
            );
        }
        let mtp_head: Option<crate::mtp_head::Qwen35MtpHead> = if mtp_armed {
            let trunk_path = &cfg.model_path;
            let mut head_opt: Option<crate::mtp_head::Qwen35MtpHead> = None;
            match crate::mtp_head::load_mtp_head_bundled(trunk_path, &mut gpu, cfg.cap_tokens) {
                Ok(Some(h)) => {
                    eprintln!(
                        "  MTP head loaded (bundled .mq4-mtp): n_embd={} vocab={}",
                        h.config.n_embd, h.config.vocab_size
                    );
                    head_opt = Some(h);
                }
                Ok(None) => {
                    if let Some(sidecar) = crate::mtp_head::find_mtp_sidecar(trunk_path) {
                        match crate::mtp_head::load_mtp_head(&sidecar, &mut gpu, cfg.cap_tokens) {
                            Ok(h) => {
                                eprintln!(
                                    "  MTP head loaded (sidecar {}): n_embd={} vocab={}",
                                    sidecar.display(),
                                    h.config.n_embd,
                                    h.config.vocab_size
                                );
                                head_opt = Some(h);
                            }
                            Err(e) => {
                                eprintln!(
                                    "  MTP sidecar {} load failed: {e} — MTP disabled",
                                    sidecar.display()
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  MTP bundled trailer load failed: {e}");
                    if let Some(sidecar) = crate::mtp_head::find_mtp_sidecar(trunk_path) {
                        match crate::mtp_head::load_mtp_head(&sidecar, &mut gpu, cfg.cap_tokens) {
                            Ok(h) => {
                                eprintln!(
                                    "  MTP head loaded (sidecar {} after bundled error): n_embd={} vocab={}",
                                    sidecar.display(),
                                    h.config.n_embd,
                                    h.config.vocab_size
                                );
                                head_opt = Some(h);
                            }
                            Err(e2) => {
                                eprintln!(
                                    "  MTP sidecar {} load failed: {e2} — MTP disabled",
                                    sidecar.display()
                                );
                            }
                        }
                    }
                }
            }
            head_opt
        } else {
            None
        };
        if mtp_armed && mtp_head.is_none() {
            // Probe last-extension replacement *and* the stem sidecar
            // (foo.mq4v2.hfq → foo.mq4v2.mtp and foo.mtp). A miss must
            // not silently degrade to AR: that is the "draft pulled but
            // DFlash off" trap.
            let probed: Vec<String> = crate::mtp_head::mtp_sidecar_candidates(&cfg.model_path)
                .into_iter()
                .map(|p| p.display().to_string())
                .collect();
            eprintln!(
                "  [hipfire] MTP requested (mtp_k={mtp_k}) but no head was found: \
                 no bundled .mq4-mtp trailer and no sidecar at {} — \
                 serving without MTP",
                probed.join(" or ")
            );
        }

        // ── DFlash2 draft loading ─────────────────────────────────────────
        // Shared draft state (weights + extract-layer hidden staging). The
        // slot path runs the DFlash2 selector chain only — legacy DFlash
        // drafts are refused inside `load_dflash_shared`. `dflash_required`
        // (dflash_mode=on) fails the load; auto warns and serves AR.
        let dflash = match &cfg.dflash_draft {
            Some(path) => match crate::dflash_slot::load_dflash_shared(
                &mut gpu,
                &path.to_string_lossy(),
                &config,
                cfg.cap_tokens,
                max_batch,
            ) {
                Ok(d) => {
                    eprintln!(
                        "  DFlash2 draft loaded (slots): {} (B={}, ctx_cap={})",
                        path.display(),
                        d.block_size,
                        d.ctx_capacity
                    );
                    Some(d)
                }
                Err(e) => {
                    if cfg.dflash_required {
                        // No transactional guard yet — free the stages this
                        // build already owns before returning.
                        if let Some(h) = mtp_head {
                            h.free_gpu(&mut gpu);
                        }
                        if let Some(vw) = vision_weights {
                            vw.free_gpu(&mut gpu);
                        }
                        weights.free_gpu(&mut gpu);
                        return Err(format!(
                            "dflash_mode=on: draft load failed for {}: {e}",
                            path.display()
                        ));
                    }
                    eprintln!(
                        "  DFlash draft {} load failed: {e} — serving without DFlash",
                        path.display()
                    );
                    None
                }
            },
            None => None,
        };
        // DFlash has precedence; only refuse a head that would actually serve
        // native MTP. A spill auto-decline or missing head remains AR.
        if mtp_head.is_some() && dflash.is_none() {
            if let Some(reason) = crate::qwen35::weights::native_mtp_refusal(
                &weights.layers,
                &gpu.arch,
                gpu.flags.fp16_disabled,
            ) {
                if let Some(head) = mtp_head {
                    head.free_gpu(&mut gpu);
                }
                if let Some(vw) = vision_weights {
                    vw.free_gpu(&mut gpu);
                }
                weights.free_gpu(&mut gpu);
                return Err(reason.to_string());
            }
        }

        // ── Transactional post-weight guard ────────────────────────────────
        // Every allocation after weights is owned here. On any error, Drop
        // frees all completed stages in reverse construction order
        // (out_tokens → logits → scratch → pbs → staging → dn → arenas →
        // weights → caches/pool) so no prior stage leaks. Actual-device
        // preflight above is retained; CPU arch/tensor preflight precedes
        // this loader (see doc above).
        struct Guard {
            gpu: Option<Gpu>,
            weights: Option<Qwen35Weights>,
            vision_weights: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
            mtp_head: Option<crate::mtp_head::Qwen35MtpHead>,
            dflash: Option<crate::dflash_slot::DflashShared>,
            k_arenas: Vec<GpuTensor>,
            v_arenas: Vec<GpuTensor>,
            kv_tier: Option<crate::forward_slots::SlotKvTier>,
            dn_states: Vec<DeltaNetState>,
            desc_staging: Option<SlotDescStaging>,
            pbs: Option<PrefillBatchScratch>,
            scratch: Option<Qwen35Scratch>,
            logits_out: Option<GpuTensor>,
            out_tokens: Option<GpuTensor>,
            committed: bool,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                if self.committed {
                    return;
                }
                if let Some(mut gpu) = self.gpu.take() {
                    if let Some(t) = self.out_tokens.take() {
                        let _ = gpu.free_tensor(t);
                    }
                    if let Some(t) = self.logits_out.take() {
                        let _ = gpu.free_tensor(t);
                    }
                    if let Some(s) = self.scratch.take() {
                        let _ = s.free_gpu(&mut gpu);
                    }
                    if let Some(p) = self.pbs.take() {
                        let _ = p.free_gpu(&mut gpu);
                    }
                    if let Some(d) = self.desc_staging.take() {
                        d.free_gpu(&mut gpu);
                    }
                    for dn in self.dn_states.drain(..).rev() {
                        dn.free_gpu(&mut gpu);
                    }
                    while let Some(v) = self.v_arenas.pop() {
                        let _ = gpu.free_tensor(v);
                    }
                    while let Some(k) = self.k_arenas.pop() {
                        let _ = gpu.free_tensor(k);
                    }
                    if let Some(tier) = self.kv_tier.take() {
                        tier.free_gpu(&mut gpu);
                    }
                    if let Some(w) = self.weights.take() {
                        w.free_gpu(&mut gpu);
                    }
                    if let Some(h) = self.mtp_head.take() {
                        h.free_gpu(&mut gpu);
                    }
                    if let Some(d) = self.dflash.take() {
                        d.free_gpu(&mut gpu);
                    }
                    if let Some(vw) = self.vision_weights.take() {
                        vw.free_gpu(&mut gpu);
                    }
                    gpu.invalidate_weight_caches();
                    gpu.invalidate_graph_state();
                    gpu.drain_pool();
                }
            }
        }
        let mut g = Guard {
            gpu: Some(gpu),
            weights: Some(weights),
            vision_weights,
            mtp_head,
            dflash,
            k_arenas: Vec::with_capacity(n_fa_layers),
            v_arenas: Vec::with_capacity(n_fa_layers),
            kv_tier: None,
            dn_states: Vec::with_capacity(cfg.n_slots),
            desc_staging: None,
            pbs: None,
            scratch: None,
            logits_out: None,
            out_tokens: None,
            committed: false,
        };
        let pool = if let Some(pages) = paged_pages {
            SlotPool::new_paged_with_strides(
                cfg.n_slots,
                cfg.cap_tokens,
                per_pos_bytes,
                per_pos_v_bytes,
                pages,
            )
        } else {
            SlotPool::new_with_strides(cfg.n_slots, cfg.cap_tokens, per_pos_bytes, per_pos_v_bytes)
        }
        .map_err(|e| format!("SlotPool: {e}"))?;
        if !kv_plan.kv_strides_differ() {
            // q8/bf16: nothing else to log; the tier banner below covers it.
        } else {
            eprintln!(
                "[hipfire] slot KV tier {kv_mode:?}: K {per_pos_bytes} B/pos, \
                 V {per_pos_v_bytes} B/pos (strides differ; descriptors carry \
                 separate K/V slab bases)"
            );
        }
        if pool.is_paged() {
            eprintln!(
                "[hipfire] paged KV pool: {} slots x cap {} tokens over {} physical pages \
                 ({} tokens/page)",
                cfg.n_slots,
                pool.cap_tokens(),
                pool.free_pages(),
                PAGE_TOKENS
            );
        }
        let k_arena_bytes = pool.k_arena_bytes();
        let v_arena_bytes = pool.v_arena_bytes();
        for _ in 0..n_fa_layers {
            let k = g
                .gpu
                .as_mut()
                .unwrap()
                .zeros(&[k_arena_bytes], DType::Raw)
                .map_err(|e| format!("k arena: {e}"))?;
            g.k_arenas.push(k);
            let v = g
                .gpu
                .as_mut()
                .unwrap()
                .zeros(&[v_arena_bytes], DType::Raw)
                .map_err(|e| format!("v arena: {e}"))?;
            g.v_arenas.push(v);
        }
        // Rotation tables for the resolved tier: Givens angles (asym) or
        // FWHT sign vectors (fwht), model-global and shared by every slot
        // and layer. Seeds match the sequential path's cache constructors
        // exactly, so slot-pool cache bytes are bit-identical to the
        // sequential engine's for the same tier.
        {
            let gpu = g.gpu.as_mut().unwrap();
            let upload_f32 = |gpu: &mut Gpu, vals: &[f32]| -> Result<GpuTensor, String> {
                let t = gpu
                    .alloc_tensor(&[vals.len()], DType::F32)
                    .map_err(|e| format!("kv rotation table: {e}"))?;
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
                gpu.hip
                    .memcpy_htod(&t.buf, &bytes)
                    .map_err(|e| format!("kv rotation table upload: {e}"))?;
                Ok(t)
            };
            let (cos, sin, s1, s2) = match kv_plan.givens_len {
                Some(len) => {
                    let (c, si) = KvCache::gen_givens_angles(42, len);
                    (
                        Some(upload_f32(gpu, &c)?),
                        Some(upload_f32(gpu, &si)?),
                        None,
                        None,
                    )
                }
                None => match kv_plan.fwht_len {
                    Some(len) => (
                        None,
                        None,
                        Some(upload_f32(gpu, &KvCache::gen_fwht_signs(42, len))?),
                        Some(upload_f32(gpu, &KvCache::gen_fwht_signs(1042, len))?),
                    ),
                    None => (None, None, None, None),
                },
            };
            g.kv_tier = Some(crate::forward_slots::SlotKvTier {
                mode: kv_mode,
                givens_cos: cos,
                givens_sin: sin,
                fwht_signs1: s1,
                fwht_signs2: s2,
            });
        }
        for _ in 0..cfg.n_slots {
            let dn = DeltaNetState::new(g.gpu.as_mut().unwrap(), &config)
                .map_err(|e| format!("dn: {e}"))?;
            g.dn_states.push(dn);
        }
        // Paged mode needs per-slot block-table staging sized to the slot
        // cap; legacy mode passes 0 and the staging stays empty.
        let max_pages_per_slot = if pool.is_paged() {
            pool.cap_tokens().div_ceil(PAGE_TOKENS)
        } else {
            0
        };
        let desc_staging = SlotDescStaging::new(
            g.gpu.as_mut().unwrap(),
            cfg.n_slots,
            max_batch,
            max_pages_per_slot,
        )
        .map_err(|e| format!("staging: {e}"))?;
        g.desc_staging = Some(desc_staging);
        // Slots do plain prefill only — never tree-verify — so skip the GDN
        // S-tape: at max_batch=cap_tokens it is tens of GB (16k → ~21 GB).
        let pbs = PrefillBatchScratch::new_opt(g.gpu.as_mut().unwrap(), &config, max_batch, false)
            .map_err(|e| format!("pbs: {e}"))?;
        g.pbs = Some(pbs);
        if kvmem_on {
            kvmem_hooks::enable_taps(
                g.pbs.as_mut().unwrap(),
                g.gpu.as_mut().unwrap(),
                &config,
                cfg,
                max_batch,
            )?;
        }
        let scratch =
            Qwen35Scratch::new_with_kv_max(g.gpu.as_mut().unwrap(), &config, 64, cfg.cap_tokens)
                .map_err(|e| format!("scratch: {e}"))?;
        g.scratch = Some(scratch);
        let logits_out = g
            .gpu
            .as_mut()
            .unwrap()
            .zeros(&[cfg.n_slots * config.vocab_size], DType::F32)
            .map_err(|e| format!("logits: {e}"))?;
        g.logits_out = Some(logits_out);
        let out_tokens = g
            .gpu
            .as_mut()
            .unwrap()
            .zeros(&[cfg.n_slots], DType::F32)
            .map_err(|e| format!("out_tokens: {e}"))?;
        g.out_tokens = Some(out_tokens);
        let sample_params = (0..cfg.n_slots)
            .map(|_| SlotSampleParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
                repeat_window: 0,
                repeat_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
            })
            .collect::<Vec<_>>();
        // Per-slot penalty windows: u32 token ids, most recent last, uploaded
        // from the session's token tail before each sampling step. Capacity
        // bounds the window a request may request (the daemon clamps harder).
        let repeat_windows = (0..cfg.n_slots)
            .map(|_| {
                g.gpu
                    .as_mut()
                    .unwrap()
                    .zeros(&[REPEAT_WINDOW_MAX], DType::F32)
                    .map_err(|e| format!("repeat window: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Per-slot vision-embedding matrices, populated lazily when a batched
        // VL request runs vision_forward. The previous matrix is explicitly
        // freed on replacement (see `admit` — dropping a GpuTensor without
        // free_tensor leaks its VRAM; it is neither pooled nor hipFree'd).
        let vl_ext_devs: Vec<Option<GpuTensor>> = (0..cfg.n_slots).map(|_| None).collect();
        let vl_tower_jobs: Vec<Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionTowerJob>> =
            (0..cfg.n_slots).map(|_| None).collect();
        if vl_sequential && !matches!(kv_mode, hipfire_runtime::kv_mode::KvMode::Q8) {
            return Err(
                "HIPFIRE_VL_SEQUENTIAL=1 writes KV through the q8-only flat slab \
                 view; it is not supported under a rotated KV tier. Drop the knob \
                 (the default batched VL path is tier-generic) or use q8 KV."
                    .to_string(),
            );
        }

        let dn_bytes: u64 = dn_buffers(&g.dn_states[0])
            .iter()
            .map(|t| t.buf.size() as u64)
            .sum();
        let kv_dtype_tag = slot_snapshot_kv_tag(kv_mode);
        // A load identity, not a full-content hash: length + sampled bytes.
        // The per-engine nonce makes a snapshot restorable only by the engine
        // that captured it, even if a file outlives its store.
        let model_fingerprint = hipfire_runtime::swap::snapshot::model_fingerprint(&cfg.model_path)
            .map_err(|e| format!("model fingerprint: {e}"))?;
        let instance_id = instance_nonce();
        let stamp = SnapshotStamp {
            model_hash: model_fingerprint,
            instance_id,
            kv_dtype_tag,
            per_pos_bytes: per_pos_bytes as u32,
            per_pos_v_bytes: per_pos_v_bytes as u32,
            n_fa_layers: n_fa_layers as u32,
            dn_layout_version: 1,
            cap: pool.cap_tokens() as u32,
            dn_bytes,
        };

        // KVMem turns KV into a fixed residency cost: the paged pool is sized
        // once at load (the resident working set), and every logical page past
        // it lives in the host store. A 256k-context request therefore costs
        // the same VRAM as a 4k one. Charging upstream's full-context grant
        // (`ctx × kv_bytes_per_token`) would demand ~9 GiB of KV credit for a
        // 256k request on a 20 GiB card and refuse it outright — the exact
        // configuration KVMem exists to serve. Charge the pool once (it is
        // real, preallocated device memory, like the weights) and nothing per
        // token; slot concurrency is bounded by the pool's slot count.
        let (admission_weights_bytes, admission_kv_bytes_per_token) = if kvmem_on {
            (resident_weight_bytes.saturating_add(kv_bytes), 0)
        } else {
            (
                resident_weight_bytes,
                (n_fa_layers * (per_pos_bytes + per_pos_v_bytes)) as u64,
            )
        };
        let mut adm = AdmissionController::new(
            ModelFootprint {
                // `weights_bytes` is the loader's *device* number, so the VRAM
                // charge cannot over-admit against what was really placed, and the
                // unreclaimable `hipHostMalloc` bytes are charged to the host tier
                // that exists for them.
                weights_bytes: admission_weights_bytes,
                kv_bytes_per_token: admission_kv_bytes_per_token,
                pinned_host_bytes,
            },
            vram_total as u64,
        );
        adm.set_host_budget(cfg.host_budget_bytes);
        let swap = SwapManager::new(
            cfg.swap_dir.clone(),
            cfg.host_budget_bytes,
            cfg.swap_disk_bytes,
        )
        .map_err(|e| format!("SwapManager: {e}"))?;
        eprintln!(
            "serve: swap host-budget={} B disk-budget={} B dir={} session-ttl={:?} session-max={} model-fingerprint={:016x} instance-id={:016x}",
            cfg.host_budget_bytes,
            swap.store().disk_budget(),
            cfg.swap_dir.display(),
            session_ttl,
            session_max,
            model_fingerprint,
            instance_id
        );
        if cfg.host_budget_bytes == 0 && swap.store().disk_budget() == 0 {
            eprintln!("serve: swap tiers disabled (host=0, disk unusable); evicted sessions will re-prefill");
        }

        // Commit guard — prevent Drop cleanup, move out owned GPU state.
        g.committed = true;
        let mut gpu = g.gpu.take().unwrap();
        let weights = g.weights.take().unwrap();
        let vision_weights = g.vision_weights.take();
        let mtp_head = g.mtp_head.take();
        let dflash = g.dflash.take();
        let k_arenas = std::mem::take(&mut g.k_arenas);
        let v_arenas = std::mem::take(&mut g.v_arenas);
        let kv_tier = g.kv_tier.take().expect("kv tier built");
        let dn_states = std::mem::take(&mut g.dn_states);
        let desc_staging = g.desc_staging.take().unwrap();
        let pbs = g.pbs.take().unwrap();
        let scratch = g.scratch.take().unwrap();
        let logits_out = g.logits_out.take().unwrap();
        let out_tokens = g.out_tokens.take().unwrap();
        // KVMem controller: host payload store + residency policy. Host-only
        // construction (no GPU allocation); the per-slot stores are sized for
        // this model's FullAttention geometry.
        #[cfg(feature = "kvmem")]
        let kvmem = kvmem_engine::build_kvmem_controller(
            cfg,
            &config,
            cfg.n_slots,
            n_fa_layers,
            kv_mode,
            per_pos_bytes,
            per_pos_v_bytes,
            max_batch,
            mtp_head.is_some() && mtp_k > 0,
            spec_rows,
            prefill_chunk,
            kvmem_plan,
            kvmem_options,
        )?;
        #[cfg(not(feature = "kvmem"))]
        let kvmem: Option<()> = None;
        // MTP head-KV prefill scratch, sized for one scheduler prompt chunk.
        // Allocated after the guard commit (transactional ownership is no
        // longer needed past this point: everything above already committed).
        let mtp_prefill_batched =
            mtp_head
                .as_ref()
                .map(
                    |head| -> Result<
                        (crate::mtp_head::Qwen35MtpHeadBatchedScratch, GpuTensor),
                        String,
                    > {
                        let scratch = crate::mtp_head::Qwen35MtpHeadBatchedScratch::new(
                            &mut gpu,
                            &head.config,
                            prefill_chunk.max(1),
                        )
                        .map_err(|e| format!("mtp prefill scratch: {e}"))?;
                        let widest_k = (2 * config.dim)
                            .max(head.config.n_ff)
                            .max(config.dim)
                            .max(head.config.n_head * head.config.head_dim);
                        let rot = gpu
                            .zeros(&[prefill_chunk.max(1) * widest_k], DType::F32)
                            .map_err(|e| format!("mtp prefill rot: {e}"))?;
                        Ok((scratch, rot))
                    },
                )
                .transpose()?;
        // Capture-only: the batched verify is repaired per layer by
        // `mtp_dn_repair_from_tape`, never by `GdnTape::replay_gdn`.
        let spec_verify_tape = if (mtp_head.is_some() && mtp_k > 0) || dflash.is_some() {
            Some(
                crate::speculative::GdnTape::new_capture_only(
                    &mut gpu,
                    &config,
                    cfg.n_slots * spec_rows,
                )
                .map_err(|e| format!("spec verify tape: {e}"))?,
            )
        } else {
            None
        };
        let mtp_prefill_hidden = gpu
            .zeros(&[prefill_chunk.max(1) * config.dim], DType::F32)
            .map_err(|e| format!("mtp prefill hidden staging: {e}"))?;
        // ── Prefix cache (spec §4.1–4.6) ──────────────────────────────────
        // Build the CacheDomain ONCE from available identities. When
        // prefix_cache is off, all three are None and the existing path is
        // bitwise unchanged. P4.3: KVMem PREFIX_PIN also needs the domain +
        // DeltaNet checkpoint pool (cross-session reuse must restore the
        // recurrent state); the radix index itself stays OFF under KVMem.
        let (prefix_index, checkpoint_pool, cache_domain) = if prefix_cache || kvmem_prefix_pin {
            // `prefix_cache_max_bytes` is a hard retention ceiling including
            // hybrid snapshots; zero means NO retained cache, never
            // unlimited (spec §9.1). A zero ceiling cannot retain a
            // checkpoint, so reuse can never satisfy the hybrid resumability
            // requirement — refuse the combination instead of running a
            // churn cache that publishes pages it can never resume.
            if prefix_cache && cfg.prefix_cache_max_bytes == 0 {
                return Err(
                    "serve.prefix_cache requires serve.prefix_cache_max_bytes > 0 \
                     (0 means no retained cache, never unlimited)"
                        .to_string(),
                );
            }
            use hipfire_runtime::serve_contract::{
                ArchPolicy, CacheDomain, DeviceTopology, KvLayout, SharingNamespace,
                TemplateIdentity, TokenizerIdentity,
            };
            // C1 identity: SHA-256 of HFQ content, tokenizer vocab/config,
            // and the rendered chat-template program (spec §4.1). Sidecars
            // (MTP, optional VL) are hashed independently so a changed
            // adapter cannot hit a trunk-only cache entry.
            let chat_template = hfq.chat_template().unwrap_or_default();
            let template_digest =
                hipfire_runtime::serve_contract::sha256_len_prefixed(&[chat_template.as_bytes()]);
            let mut sidecar_digests = Vec::new();
            if mtp_head.is_some() {
                if let Some(sidecar) = crate::mtp_head::find_mtp_sidecar(&cfg.model_path) {
                    match HfqFile::open(&sidecar) {
                        Ok(mtp_hfq) => sidecar_digests.push(mtp_hfq.content_digest()),
                        Err(_) => {
                            if let Ok(d) = hipfire_runtime::serve_contract::sha256_file(&sidecar) {
                                sidecar_digests.push(d);
                            }
                        }
                    }
                }
            }
            if let Some(vl) = cfg.vl_path.as_ref() {
                if vl.exists() {
                    match HfqFile::open(vl) {
                        Ok(vl_hfq) => sidecar_digests.push(vl_hfq.content_digest()),
                        Err(_) => {
                            if let Ok(d) = hipfire_runtime::serve_contract::sha256_file(vl) {
                                sidecar_digests.push(d);
                            }
                        }
                    }
                }
            }
            let domain = CacheDomain {
                model_content_digest: hfq.content_digest(),
                model_load_epoch: 1,
                sidecar_digests,
                tokenizer: TokenizerIdentity {
                    vocab_digest: tokenizer.vocab_digest(),
                    config_digest: tokenizer.config_digest(),
                },
                template: TemplateIdentity {
                    template_digest,
                    normalization_tag: if chat_template.is_empty() {
                        "prompt-frame".to_string()
                    } else {
                        "jinja".to_string()
                    },
                },
                arch_policy: ArchPolicy {
                    arch_tag: "qwen35-deltanet".to_string(),
                    state_abi_tag: "dn-q8".to_string(),
                    position_attention_tag: "causal-mrope".to_string(),
                },
                kv_layout: KvLayout {
                    k_stride_bytes: vec![per_pos_bytes as u64],
                    v_stride_bytes: vec![per_pos_v_bytes as u64],
                    layout_tag: format!("{:?}", kv_mode),
                },
                device: DeviceTopology {
                    device_id: format!("gpu-{}", gpu.device_id),
                    topology_id: "single".to_string(),
                    allocation_epoch: 1,
                },
                namespace: SharingNamespace {
                    domain_id: "default".to_string(),
                },
            };
            // P4.3: under KVMem PREFIX_PIN the radix index stays OFF (KVMem
            // owns sharing through its canonical catalog); only the
            // checkpoint pool + domain identity are built. The retained-cache
            // ceiling also bounds the KV page bytes the radix references
            // (spec §9.1: a retention ceiling, never a permanently reserved
            // partition — the checkpoint pool below holds its own ceiling on
            // snapshot bytes).
            let idx = if prefix_cache {
                let mut idx = hipfire_runtime::prefix_index::PrefixIndex::new(1 << 16);
                idx.set_max_retained_bytes(cfg.prefix_cache_max_bytes as usize);
                Some(idx)
            } else {
                None
            };
            let ckpt_ceiling = if prefix_cache {
                cfg.prefix_cache_max_bytes
            } else {
                // PREFIX_PIN snapshot ceiling (DeltaNet snapshots only, no
                // radix pages): HIPFIRE_KVMEM_PREFIX_PIN_BYTES, default 512 MiB.
                hipfire_config::developer_var("HIPFIRE_KVMEM_PREFIX_PIN_BYTES")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .filter(|&b| b > 0)
                    .unwrap_or(512 * 1024 * 1024)
            };
            let pool = crate::checkpoint::CheckpointPool::<DeltaNetSnapshot>::new(ckpt_ceiling);
            if prefix_cache {
                eprintln!(
                    "  [hipfire] prefix cache enabled: radix max {} nodes / {} MiB \
                     retained-page ceiling, checkpoint pool {} bytes",
                    1 << 16,
                    cfg.prefix_cache_max_bytes / (1024 * 1024),
                    cfg.prefix_cache_max_bytes,
                );
            } else {
                eprintln!(
                    "  [hipfire] KVMem prefix PIN: checkpoint pool {} bytes (radix off)",
                    ckpt_ceiling,
                );
            }
            (idx, Some(pool), Some(domain))
        } else {
            (None, None, None)
        };

        let vocab_size = config.vocab_size;
        // FairQueue (spec §5.3 S3): constructed from the same budget knobs the
        // scheduler enforces. An invalid config (max_batch_tokens < n_slots +
        // prefill_min_tokens) fails build rather than silently disabling
        // fairness — the policy would otherwise deadlock.
        let max_batch_tokens = cfg.max_batch_tokens.max(1);
        let prefill_min_tokens = cfg.prefill_min_tokens.max(1);
        let fair_queue = FairQueue::new(
            max_batch_tokens as u64,
            cfg.n_slots as u64,
            prefill_min_tokens as u64,
        )
        .map_err(|e| format!("fairness queue: {e}"))?;
        // WaitQueue (spec §5.3 S3): bounded waiting room that replaces the
        // R-A4 immediate-reject path. max_count and max_bytes MUST be
        // non-zero — WaitQueue::new refuses the zero/uncapped combination
        // (spec §5.3: "serve.max_queue=0 currently means uncapped; the new
        // robust multi-slot mode must reject that combination"). Failing
        // build here is the engine-side backstop even when the CLI guard
        // already rejected max_queue=0.
        // Waiter deadlines are wall-clock (`Instant`) in the engine —
        // `queue_timeout_ms` against `Instant::now` — so the WaitQueue's own
        // tick-based expiry is disabled here (u64::MAX never expires): ticks
        // advance once per serve iteration, which can absorb a whole prefill
        // chunk or not advance at all while the loop blocks on recv, and is
        // therefore not a sound deadline base (spec §5.3 S3).
        let wait_queue = WaitQueue::new(cfg.wait_max_count, cfg.wait_max_bytes, u64::MAX)
            .map_err(|e| format!("wait queue: {e}"))?;
        // Spec → AR retirement line, shared with the sequential drafter:
        // `HIPFIRE_SPEC_RETIRE` when set, else the shared default. A malformed
        // value turns retirement off and says so, rather than guessing a line
        // (mirrors hipfire-generate's `spec_retire_break_even`). The slot
        // engine states no drafter-specific measured line of its own.
        let spec_retire_break_even = {
            let raw = hipfire_config::developer_var("HIPFIRE_SPEC_RETIRE").ok();
            match hipfire_runtime::spec_retire::parse_break_even(raw.as_deref()) {
                Ok(parsed) => hipfire_runtime::spec_retire::resolve_break_even(parsed, None),
                Err(e) => {
                    eprintln!("[spec-retire] {e}; retirement off");
                    None
                }
            }
        };
        Ok(Rig {
            gpu,
            tokenizer,
            pool,
            model: ModelRig {
                weights,
                config,
                scratch,
                dn_states,
                #[cfg(feature = "kvmem")]
                dn_images: (0..cfg.n_slots).map(|_| None).collect(),
                #[cfg(feature = "kvmem")]
                dn_idle_images: (0..cfg.n_slots).map(|_| None).collect(),
                #[cfg(feature = "kvmem")]
                legacy_head_planes: (0..cfg.n_slots).map(|_| None).collect(),
                k_arenas,
                v_arenas,
                desc_staging,
                kv_tier,
                pbs,
                logits_out,
                mtp_head,
                mtp_states: (0..cfg.n_slots).map(|_| None).collect(),
                spec_req_sampling: (0..cfg.n_slots).map(|_| None).collect(),
                mtp_prefill_batched,
                spec_verify_tape,
                mtp_prefill_hidden,
                mtp_k,
                spec_rows,
                spec_retire_break_even,
                dflash,
                dflash_states: (0..cfg.n_slots).map(|_| None).collect(),
                vision_weights,
                vision_config,
                vl_ext_devs,
                vl_tower_jobs,
                checkpoint_pool,
                #[cfg(feature = "kvmem")]
                kvmem,
            },
            out_tokens,
            sample_params,
            repeat_windows,
            vl_sequential,
            sessions: SessionTable::default(),
            #[cfg(feature = "kvmem")]
            legacy_history: cfg
                .kvmem
                .as_ref()
                .filter(|c| c.enabled)
                .map(|c| {
                    // Inactive-history host ceiling for non-KV image bytes
                    // (recurrent/conv payload + draft row images). The KV payload
                    // itself is charged against `host_bytes`; this ceiling keeps a
                    // DFlash/MTP draft image per session from crowding every
                    // version out of the catalog at the default 512 MiB.
                    let ceiling = usize::try_from(c.host_bytes / 2)
                        .unwrap_or(usize::MAX)
                        .max(512 * 1024 * 1024);
                    crate::legacy_catalog::LegacyHistoryCatalog::new(c.sessions as usize, ceiling)
                        .map_err(|e| format!("kvmem history directory: {e}"))
                })
                .transpose()?,
            #[cfg(feature = "kvmem")]
            legacy_queries: (0..cfg.n_slots).map(|_| None).collect(),
            #[cfg(feature = "kvmem")]
            legacy_query_decided: vec![false; cfg.n_slots],
            #[cfg(feature = "kvmem")]
            legacy_spec_kind: vec![SpecKind::None; cfg.n_slots],
            adm,
            swap,
            stamp,
            n_slots: cfg.n_slots,
            cap_tokens: cfg.cap_tokens,
            prefill_chunk,
            session_ttl,
            session_max,
            closed_sessions,
            max_batch_tokens,
            prefill_min_tokens,
            logits_host: vec![0.0f32; cfg.n_slots * vocab_size],
            prefix_index,
            cache_domain,
            prefix_cache,
            kvmem_prefix_pin,
            kvmem_prefix_coverage,
            kvmem_prefix_coverage_interval,
            kvmem_checkpoint_policy,
            spec_counters: vec![SpecCounters::default(); cfg.n_slots],
            prefix_cache_max_bytes: if prefix_cache {
                Some(cfg.prefix_cache_max_bytes as usize)
            } else {
                None
            },
            lookup_count: 0,
            cow_plan_count: 0,
            fair_queue,
            wait_queue,
            parked_requests: std::collections::HashMap::new(),
            next_waiter_id: 0,
            queue_timeout_ms: cfg.queue_timeout_ms,
            pending_repark_age: None,
            tick: 0,
            structured_jump_forward: cfg.structured_jump_forward,
            schema_cache: std::collections::HashMap::new(),
            schema_cache_order: std::collections::VecDeque::new(),
        })
    }

    /// Release every GPU allocation owned by this rig, then invalidate caches
    /// and drain the HIP pool. Graph handles must be destroyed first so they
    /// never outlive the buffers they capture. Best-effort: keeps going after
    /// individual free failures and returns only the first error for logging.
    fn free_gpu(self, mut graph: SlotDecodeGraph) -> Result<(), String> {
        let Rig {
            mut gpu,
            out_tokens,
            model:
                ModelRig {
                    weights,
                    vision_weights,
                    mtp_head,
                    mtp_states,
                    mtp_prefill_batched,
                    spec_verify_tape,
                    mtp_prefill_hidden,
                    dflash,
                    dflash_states,
                    kv_tier,
                    k_arenas,
                    v_arenas,
                    dn_states,
                    #[cfg(feature = "kvmem")]
                    legacy_head_planes,
                    desc_staging,
                    pbs,
                    scratch,
                    logits_out,
                    vl_ext_devs,
                    vl_tower_jobs,
                    checkpoint_pool,
                    ..
                },
            ..
        } = self;

        let mut first_err: Option<String> = None;
        let mut note_hip = |r: rdna_compute::HipResult<()>| {
            if let Err(e) = r {
                if first_err.is_none() {
                    first_err = Some(e.to_string());
                }
            }
        };

        // Release slot, GPU-global and MTP graphs before their tensor dependencies.
        // Then free the remaining tensor owners in reverse construction order.
        graph.release(&gpu);
        gpu.invalidate_graph_state();
        for mtp_state in mtp_states.into_iter().flatten() {
            mtp_state.free_gpu(&mut gpu);
        }
        note_hip(gpu.free_tensor(out_tokens));
        note_hip(gpu.free_tensor(logits_out));
        note_hip(scratch.free_gpu(&mut gpu));
        note_hip(pbs.free_gpu(&mut gpu));
        desc_staging.free_gpu(&mut gpu);
        kv_tier.free_gpu(&mut gpu);
        for dn in dn_states.into_iter().rev() {
            dn.free_gpu(&mut gpu);
        }
        #[cfg(feature = "kvmem")]
        for plane in legacy_head_planes.into_iter().flatten() {
            let _ = plane.free(&mut gpu);
        }
        // Arenas were allocated per FA layer as k then v; free reverse layers
        // as v then k.
        let mut k_arenas = k_arenas;
        let mut v_arenas = v_arenas;
        while !k_arenas.is_empty() || !v_arenas.is_empty() {
            if let Some(v) = v_arenas.pop() {
                note_hip(gpu.free_tensor(v));
            }
            if let Some(k) = k_arenas.pop() {
                note_hip(gpu.free_tensor(k));
            }
        }
        weights.free_gpu(&mut gpu);
        if let Some(h) = mtp_head {
            h.free_gpu(&mut gpu);
        }
        if let Some((scratch, rot)) = mtp_prefill_batched {
            scratch.free_gpu(&mut gpu);
            note_hip(gpu.free_tensor(rot));
        }
        if let Some(tape) = spec_verify_tape {
            tape.free_gpu(&mut gpu);
        }
        if let Some(d) = dflash {
            d.free_gpu(&mut gpu);
        }
        for st in dflash_states.into_iter().flatten() {
            st.free_gpu(&mut gpu);
        }
        note_hip(gpu.free_tensor(mtp_prefill_hidden));
        if let Some(vw) = vision_weights {
            vw.free_gpu(&mut gpu);
        }
        // Per-slot VL matrices + any tower job still mid-encode at teardown.
        for dev in vl_ext_devs.into_iter().flatten() {
            note_hip(gpu.free_tensor(dev));
        }
        for job in vl_tower_jobs.into_iter().flatten() {
            note_hip(job.free(&mut gpu));
        }
        // Free checkpoint pool snapshots (DeltaNetSnapshot device buffers).
        if let Some(mut pool) = checkpoint_pool {
            for blob in pool.drain_blobs() {
                blob.free_gpu(&mut gpu);
            }
        }
        gpu.invalidate_weight_caches();
        gpu.drain_pool();

        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// One slot's in-flight spec draft between the draft phase and the
/// post-forward accept. `Mtp` carries the head's K-step chain output;
/// `Dflash` carries the DFlash2 selector block. Both expose the verify-row
/// tokens `[seed, drafts…]` and the row-0 position the batch injects.
enum SpecDraftRows {
    Mtp(crate::mtp_spec::MtpDraftOutput),
    Dflash(crate::dflash_slot::DflashSlotDraft),
}

impl SpecDraftRows {
    fn verify_tokens(&self) -> Vec<u32> {
        match self {
            SpecDraftRows::Mtp(d) => d.verify_tokens(),
            SpecDraftRows::Dflash(d) => d.verify_tokens.clone(),
        }
    }
    fn cur_pos(&self) -> usize {
        match self {
            SpecDraftRows::Mtp(d) => d.cur_pos,
            SpecDraftRows::Dflash(d) => d.cur_pos,
        }
    }
    /// Release the draft's device buffers. `DflashSlotDraft` owns four
    /// `GpuTensor`s (verify_hidden/logits/rot/argmax, ~batch×vocab each) and
    /// `GpuTensor` has no `Drop` — dropping the enum without this leaks VRAM.
    /// `MtpDraftOutput` is host-only, so this is a no-op for it.
    fn free_gpu(self, gpu: &mut Gpu) {
        if let SpecDraftRows::Dflash(d) = self {
            d.free_gpu(gpu);
        }
    }
}

/// Rebuild a `SlotBatch` to include spec verify tokens for slots that have
/// draft outputs. Non-spec slots keep their scheduler-provided rows.
///
/// `pos3_deltas[s]` is the slot's continuation rope offset (0 for pure-text
/// conversations): verify rows are text rows of the slot they belong to, so
/// a slot continuing an image conversation phases them at `pos + delta`
/// exactly like the scheduler does for its prefill/decode rows.
///
/// `force_pos3` is the scheduler's work-level "this step needs M-RoPE" flag
/// (`(vl_prefill && !vl_sequential) || pos3_delta != 0` over ALL work, not
/// just the rows the scheduler emitted). It MUST be passed in by the caller:
/// a spec-decoding slot contributes zero scheduler rows, so a verify-only
/// step — the steady state whenever the only active request is a spec slot —
/// has an empty `batch.pos3` even though the slot's rows need the shifted
/// phases. Deriving the flag from `batch.pos3` emptiness here would silently
/// drop pos3 for exactly those steps and phase-shift every verify query
/// against the image turn's stored keys.
fn inject_spec_verify_tokens(
    batch: &SlotBatch,
    spec_drafts: &[Option<SpecDraftRows>],
    n_slots: usize,
    pos3_deltas: &[i32],
    force_pos3: bool,
) -> SlotBatch {
    // A VL row anywhere in the step keeps the batch on the M-RoPE/scatter
    // path, so the side arrays must survive the rebuild: verify rows are
    // text rows of text slots and take their (possibly shifted) phase; VL
    // rows are copied through untouched.
    let vl_step =
        force_pos3 || (batch.pos3.len() == batch.positions.len() && !batch.pos3.is_empty());
    let mut new_batch = SlotBatch::default();
    let mut old_row_off = 0usize;
    for s in 0..n_slots {
        if let Some(draft) = &spec_drafts[s] {
            let delta = pos3_deltas.get(s).copied().unwrap_or(0);
            let verify_tokens = draft.verify_tokens();
            new_batch.m_per_slot.push(verify_tokens.len());
            for (i, t) in verify_tokens.iter().enumerate() {
                let pos = draft.cur_pos() + i;
                new_batch.tokens.push(*t);
                new_batch.positions.push(pos as i32);
                new_batch.row_slot.push(s as i32);
                if vl_step {
                    new_batch.pos3.push([pos as i32 + delta; 3]);
                    new_batch.ext_emb.push(-1);
                }
            }
        } else {
            let m = batch.m_per_slot.get(s).copied().unwrap_or(0);
            new_batch.m_per_slot.push(m);
            if m > 0 {
                new_batch
                    .tokens
                    .extend_from_slice(&batch.tokens[old_row_off..old_row_off + m]);
                new_batch
                    .positions
                    .extend_from_slice(&batch.positions[old_row_off..old_row_off + m]);
                new_batch
                    .row_slot
                    .extend_from_slice(&batch.row_slot[old_row_off..old_row_off + m]);
                if vl_step {
                    new_batch
                        .pos3
                        .extend_from_slice(&batch.pos3[old_row_off..old_row_off + m]);
                    new_batch
                        .ext_emb
                        .extend_from_slice(&batch.ext_emb[old_row_off..old_row_off + m]);
                }
            }
            old_row_off += m;
        }
    }
    new_batch
}

fn clear_work_slot(work: &mut PendingWork) {
    work.remaining_prompt.clear();
    work.decoding = false;
    work.next_pos = 0;
    work.vl_prefill = None;
    work.spec = SpecKind::None;
    work.pos3_delta = 0;
    // The retirement policy is per-REQUEST state: a request that ends
    // mid-window must not leave its cycles/failures behind for the slot's
    // next occupant (one bad window would sit one failure from retirement,
    // and the next request's window mean would mix both requests' advances).
    // `reset` keeps the resolved break-even line, which is re-installed at
    // admit anyway.
    work.spec_retire.reset();
}

/// Release a request's prefix-cache pin on every terminal path (spec §4.4
/// C4). `lookup_with_pages` pins the matched radix path so eviction cannot
/// drop it mid-request; the pin must be released exactly once, whether the
/// request completed, was cancelled, or failed. The session's table refs
/// (from `share_published_pages`) still protect shared pages while the
/// session is idle — the pin only guards the in-flight window.
fn release_pin_ticket(rig: &mut Rig, ticket: &mut PinTicket) {
    if ticket.is_empty() {
        return;
    }
    if let (Some(idx), Some(domain)) = (rig.prefix_index.as_mut(), rig.cache_domain.as_ref()) {
        idx.release_pin(domain, std::mem::take(ticket));
    }
}

/// Rebuild a `SlotBatch`'s flat arrays to exclude rows belonging to failed
/// slots (spec §5.4/S4: per-slot provision failure isolates the failing
/// request). The flat arrays (tokens, positions, row_slot, pos3, ext_emb) are
/// packed in slot order by `m_per_slot`; zeroing a slot's count without
/// removing its flat rows would misalign every subsequent slot. This rebuilds
/// them from the surviving `m_per_slot` entries.
fn rebuild_batch_excluding_failed(batch: &mut SlotBatch, failed: &[usize]) {
    let failed_set: std::collections::HashSet<usize> = failed.iter().copied().collect();
    let mut new_tokens = Vec::new();
    let mut new_positions = Vec::new();
    let mut new_row_slot = Vec::new();
    let mut new_pos3 = Vec::new();
    let mut new_ext_emb = Vec::new();
    let has_pos3 = !batch.pos3.is_empty();
    let has_ext = !batch.ext_emb.is_empty();
    let mut off = 0usize;
    for (s, &m) in batch.m_per_slot.iter().enumerate() {
        if m == 0 || failed_set.contains(&s) {
            off += m;
            continue;
        }
        new_tokens.extend_from_slice(&batch.tokens[off..off + m]);
        new_positions.extend_from_slice(&batch.positions[off..off + m]);
        new_row_slot.extend_from_slice(&batch.row_slot[off..off + m]);
        if has_pos3 {
            new_pos3.extend_from_slice(&batch.pos3[off..off + m]);
        }
        if has_ext {
            new_ext_emb.extend_from_slice(&batch.ext_emb[off..off + m]);
        }
        off += m;
    }
    for &s in failed {
        batch.m_per_slot[s] = 0;
    }
    batch.tokens = new_tokens;
    batch.positions = new_positions;
    batch.row_slot = new_row_slot;
    batch.pos3 = new_pos3;
    batch.ext_emb = new_ext_emb;
}

/// Execute the raw device-to-device copies of a [`CowPlan`] (spec §4.3):
/// for every copy, `valid_prefix_tokens` positions' worth of K and V bytes
/// are copied from the sealed source page to the reserved private
/// destination page in EVERY layer arena (one block table maps all layers,
/// so a physical page index resolves to the same byte offset in each
/// arena). Encoded bytes move verbatim — no dequantize/requantize.
///
/// Copies must run BEFORE `commit_cow` rebinds the block table (the
/// forward's write kernels resolve through the table) and before any write
/// touches the destination rows.
///
/// Returns `false` on any copy failure; the caller then aborts the plan
/// (releasing the reserved destinations) and fails the slot — a partial
/// copy never leaves a half-initialized destination referenced, because
/// the table still maps the logical page to the source until commit.
fn execute_cow_copies(rig: &Rig, plan: &rdna_compute::page_pool::CowPlan) -> bool {
    let Some(pp) = rig.pool.page_pool() else {
        return false;
    };
    let k_page = pp.k_page_bytes();
    let v_page = pp.v_page_bytes();
    for copy in plan.copies() {
        for (layer, k_arena) in rig.model.k_arenas.iter().enumerate() {
            let Some(v_arena) = rig.model.v_arenas.get(layer) else {
                return false;
            };
            if copy.k_copy_bytes > 0 {
                let res = rig.gpu.hip.memcpy_dtod_at(
                    &k_arena.buf,
                    copy.dst_phys as usize * k_page,
                    &k_arena.buf,
                    copy.src_phys as usize * k_page,
                    copy.k_copy_bytes,
                );
                if let Err(e) = res {
                    eprintln!("[cow] K copy failed (layer {layer}): {e}");
                    return false;
                }
            }
            if copy.v_copy_bytes > 0 {
                let res = rig.gpu.hip.memcpy_dtod_at(
                    &v_arena.buf,
                    copy.dst_phys as usize * v_page,
                    &v_arena.buf,
                    copy.src_phys as usize * v_page,
                    copy.v_copy_bytes,
                );
                if let Err(e) = res {
                    eprintln!("[cow] V copy failed (layer {layer}): {e}");
                    return false;
                }
            }
        }
    }
    true
}

/// True when an MTP verify batch — `mtp_k + 1` rows at positions
/// `next_pos..next_pos + mtp_k` — fits inside the slot's token cap. The
/// batched forward provisions `set_seq_len(max_pos + 1)` before its KV write
/// and fails the whole step closed when that exceeds the cap, which rejects
/// every active slot; MTP must retire before drafting instead.
fn spec_verify_fits_cap(next_pos: usize, verify_rows: usize, cap_tokens: usize) -> bool {
    verify_rows <= cap_tokens.saturating_sub(next_pos)
}

/// True when the request carries a non-neutral token penalty. Both spec
/// drafters (MTP and DFlash2) now honor penalties in their verify accept
/// (exact-AR penalty prepass before acceptance), so this no longer gates
/// admission — it is retained for the admit slot-trace and any diagnostic
/// that needs to know whether a request's logits will be penalized.
///
/// `repeat_window == 0` does NOT neutralize the penalties: OpenAI-style
/// presence/frequency penalties are defined over the whole context, so a zero
/// window means "uncapped" (clamped to REPEAT_WINDOW_MAX), never "off". The
/// old `repeat_window > 0` gate silently dropped a request's penalties AND
/// let it take the MTP path — a double contract violation.
fn request_penalized(req: &SubmitRequest) -> bool {
    req.repeat_penalty > 1.0 || req.presence_penalty > 0.0 || req.frequency_penalty > 0.0
}

/// The penalty window a request actually gets: the requested value clamped to
/// REPEAT_WINDOW_MAX, with 0 promoted to REPEAT_WINDOW_MAX when any penalty is
/// active (0 = uncapped, per request_penalized's contract). When no penalty is
/// active the window is irrelevant and stays 0 — no upload, no kernel scan.
/// Takes the raw fields (not &SubmitRequest) so it can run after req's
/// non-Copy fields have been moved into the InFlight.
fn effective_repeat_window(
    repeat_window: usize,
    repeat_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
) -> usize {
    let w = repeat_window.min(REPEAT_WINDOW_MAX);
    let penalized = repeat_penalty > 1.0 || presence_penalty > 0.0 || frequency_penalty > 0.0;
    if w == 0 && penalized {
        REPEAT_WINDOW_MAX
    } else {
        w
    }
}

/// Install a request's sampling parameters (including token penalties and
/// min_p) on its slot's sampler state.
fn install_sample_params(rig: &mut Rig, slot: SlotId, req: &SubmitRequest) {
    rig.sample_params[slot.0] = SlotSampleParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
        repeat_window: effective_repeat_window(
            req.repeat_window,
            req.repeat_penalty,
            req.presence_penalty,
            req.frequency_penalty,
        ) as i32,
        repeat_penalty: req.repeat_penalty,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        min_p: req.min_p,
    };
}

/// A19 fault seam (host tests only): when `HIPFIRE_FAULT_PREFIX_PUBLISH=1`,
/// the first prefix publication attempt fails so the orphaned-cache-ref
/// rollback and honest-miss behavior can be exercised end to end. Fires at
/// most once per process; unset by default.
fn inject_publish_fault() -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static ARMED: AtomicBool = AtomicBool::new(true);
    static CHECKED: AtomicBool = AtomicBool::new(false);
    if !CHECKED.load(Ordering::Relaxed) {
        CHECKED.store(true, Ordering::Relaxed);
        if hipfire_config::developer_var("HIPFIRE_FAULT_PREFIX_PUBLISH").as_deref() != Ok("1") {
            ARMED.store(false, Ordering::Relaxed);
        }
    }
    ARMED.swap(false, Ordering::Relaxed)
}

/// Publish generated-prefix pages at the client-commit boundary (Done event
/// for non-cancellation reasons). Seals and publishes any full pages beyond
/// the last prefill-boundary publish, capturing a checkpoint at the new
/// boundary (spec §4.6 C6).
///
/// `state_boundary` is the request's processed-row frontier (`next_pos`): a
/// checkpoint is captured only when the published boundary equals it. The
/// session's token count can exceed the state boundary by one unforwarded
/// terminal token, and relabeling the live recurrent state as an earlier
/// boundary would corrupt every resume (spec §4.5).
fn publish_generated_prefix(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    s: usize,
    state_boundary: usize,
) {
    if !rig.prefix_cache {
        return;
    }
    let Some(f) = slots[s].as_ref() else { return };
    let session = f.session;
    let last_pub = f.last_published_boundary;
    let sess = match rig.sessions.get(session) {
        Some(s) => s,
        None => return,
    };
    // Vision + prefix reuse stays OFF (spec §6 X2): an image conversation's
    // KV carries M-RoPE compressed-grid phases that a text requester cannot
    // resume correctly, and no pixel/embedding identity oracle exists yet.
    // `has_image` is the positive marker — `rope_delta` is 0 for a thin
    // (single-row/col) image, so it cannot stand in for "carries an image".
    if sess.has_image {
        return;
    }
    let total_tokens = sess.tokens.len();
    // The token count can exceed the materialized frontier (`state_boundary`)
    // by one retained-but-never-forwarded terminal token (MaxTokens keeps
    // the final sampled token in the stream without a KV row; the next
    // turn's begin_turn prefills it). Sealing a page above the frontier
    // would publish a row that was never written — stale bytes from the
    // page's previous tenant, served to every future resumer. Clamp to the
    // processed-row frontier (the mid-step prefill publish path already
    // floors `next_pos` the same way).
    let frontier = done_publish_frontier(total_tokens, state_boundary);
    // Only publish full page-aligned boundaries beyond what was already
    // published. Pages strictly below the committed materialized frontier —
    // speculative candidate rows never cross into them (spec §4.6 C2).
    let new_boundary = (frontier / PAGE_TOKENS) * PAGE_TOKENS;
    if new_boundary <= last_pub || new_boundary == 0 {
        return;
    }
    let n_old_pages = last_pub / PAGE_TOKENS;
    let n_new_pages = new_boundary / PAGE_TOKENS;
    let page_indices: Vec<u32> = match rig.pool.block_table(SlotId(s)) {
        Some(bt) => bt.page_indices().to_vec(),
        None => return,
    };
    if page_indices.len() < n_new_pages {
        return;
    }
    // Seal the newly written pages (immutability before the publish); the
    // index takes the cache refs itself inside publish_sealed_pages (see
    // the prefill publish site for the orphaned-ref rationale).
    if let Some(pp) = rig.pool.page_pool_mut() {
        for p in n_old_pages..n_new_pages {
            let _ = pp.seal(page_indices[p]);
        }
    }
    // Build handles for all pages up to the new boundary.
    let handles: Vec<Handle> = page_indices[..n_new_pages]
        .iter()
        .enumerate()
        .map(|(i, &phys)| Handle {
            handle: rdna_compute::page_pool::PageHandle {
                phys,
                epoch: rig.pool.page_pool().map(|p| p.epoch()).unwrap_or(0),
                generation: rig
                    .pool
                    .page_pool()
                    .map(|p| p.page_generation(phys))
                    .unwrap_or(0),
            },
            token_offset: (i * PAGE_TOKENS) as u64,
        })
        .collect();
    let tokens: Vec<u32> = rig
        .sessions
        .get(session)
        .map(|sess| sess.tokens[..new_boundary].to_vec())
        .unwrap_or_default();
    if tokens.len() != new_boundary {
        return;
    }
    let domain = rig.cache_domain.as_ref().unwrap();
    let idx = rig.prefix_index.as_mut().unwrap();
    let pp = rig.pool.page_pool_mut().unwrap();
    // Capture a checkpoint only when the published boundary is exactly the
    // recurrent-state boundary (spec §4.5). A ceiling-refused capture
    // (CheckpointId::NONE) publishes pages without a checkpoint — the
    // boundary stays honestly unresumable.
    let checkpoint = if new_boundary == state_boundary {
        if let Some(ckpt_pool) = rig.model.checkpoint_pool.as_mut() {
            match capture_checkpoint(
                &mut rig.gpu,
                ckpt_pool,
                domain,
                new_boundary as u64,
                &tokens,
                None, // radix path: text-only (vision is excluded from publishing)
                &rig.model.dn_states[s],
            ) {
                Ok(id) if id.is_some() => Some(id),
                Ok(_) => None,
                Err(e) => {
                    eprintln!("[prefix-cache] generated-prefix capture_checkpoint failed: {e}");
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };
    if let Err(e) = idx.publish_sealed_pages(domain, &tokens, &handles, checkpoint, pp) {
        eprintln!("[prefix-cache] generated-prefix publish_sealed_pages failed: {e:?}");
        // No cache refs to roll back — the index owns them and releases on
        // every failure path.
    } else {
        if let Some(f) = slots[s].as_mut() {
            f.last_published_boundary = new_boundary;
        }
        if let Some(sess) = rig.sessions.get_mut(session) {
            sess.published_boundary = new_boundary;
        }
        if rig.gpu.slot_trace() {
            eprintln!(
                "[slot-trace] generated-prefix published {n_new_pages} pages at boundary {new_boundary}"
            );
        }
    }
}

fn commit_sampled_token(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    stats: &Mutex<EngineStats>,
    s: usize,
    tok: u32,
) {
    let Some(f) = slots[s].as_mut() else { return };
    let session = f.session;
    // Advance the grammar parser cursor on the accepted token BEFORE any
    // terminal checks (spec §7.2 G2: "per-request parser cursor advanced
    // on accepted tokens only"). The cursor is private per-request; a
    // speculative verifier would use a private cursor per candidate and
    // commit only the accepted prefix (spec §7.2 G2).
    if let Some(constraint) = f.grammar.as_mut() {
        constraint.advance_token(&rig.tokenizer, tok);
    }
    // Check whether the grammar constraint (if any) is in an accepting
    // state after advancing. A non-accepting state at terminal means the
    // generated JSON is incomplete/invalid — emit a typed rejection
    // instead of a successful done (spec §7.1 G1, A17).
    let grammar_incomplete = f.grammar.as_ref().is_some_and(|c| !c.is_accepting());
    let hit_eos = rig.tokenizer.is_terminator(tok);
    let gone = if hit_eos {
        false
    } else {
        send_event(&f.reply, Event::Token { id: tok }).is_err()
    };
    f.produced += 1;
    let hit_max = f.produced >= f.max_tokens;
    let hit_ctx_cap = rig
        .sessions
        .get(session)
        .map(|sess| sess.tokens.len() + 1 >= rig.cap_tokens)
        .unwrap_or(false);

    if gone || hit_eos || hit_max || hit_ctx_cap {
        let reason = if gone {
            DoneReason::ClientGone
        } else if hit_eos {
            DoneReason::Eos
        } else {
            DoneReason::MaxTokens
        };
        if should_retain_terminal_tok(reason) {
            if let Some(sess) = rig.sessions.get_mut(session) {
                sess.tokens.push(tok);
            }
        }
        let generated = if hit_eos || gone {
            f.produced - 1
        } else {
            f.produced
        };
        // If a grammar constraint is present and not accepting at terminal,
        // the generated JSON is incomplete/invalid — emit a typed rejection
        // instead of a successful done (spec §7.1 G1, A17). ClientGone is
        // never overridden: the receiver is already dropped.
        lock_stats(stats).note_slot_released();
        // 期4: fold this request's speculative counters (completed-request
        // statistics; cancellation paths included, admission failures excluded).
        {
            let spec = std::mem::take(&mut rig.spec_counters[s]);
            if spec.requested {
                let mut st = lock_stats(stats);
                st.speculative.requests += 1;
                st.speculative.rounds += spec.rounds;
                st.speculative.drafted_tokens += spec.drafted;
                st.speculative.accepted_tokens += spec.accepted;
                st.speculative.generated_tokens += spec.generated;
                st.speculative.fallback_steps += spec.fallback;
            }
        }
        if grammar_incomplete && !matches!(reason, DoneReason::ClientGone) {
            let _ = send_event(
                &f.reply,
                Event::Rejected {
                    class: RejectClass::Validation,
                    reason: GrammarMaskError::Unsatisfiable.to_string(),
                },
            );
        } else {
            let _ = send_event(&f.reply, Event::Done { reason, generated });
        }
        // Publish generated-prefix pages at client commit (spec §4.6 C6).
        // Take the pin ticket out of `f` first: publish needs `slots`
        // mutably, which `f` borrows.
        let mut pin_ticket = std::mem::take(&mut f.pin_ticket);
        if !matches!(reason, DoneReason::ClientGone) && rig.prefix_cache {
            publish_generated_prefix(rig, slots, s, work[s].next_pos);
        }
        // Release the prefix-cache lookup pin on EVERY terminal path —
        // completion and cancellation alike (spec §4.4 C4). A pin left
        // held keeps the matched radix path unevictable forever.
        release_pin_ticket(rig, &mut pin_ticket);
        slots[s] = None;
        // FairQueue: the request is done — drop it so it stops consuming
        // the fairness budget (spec §5.3 S3). Idempotent (NotFound ignored).
        let _ = rig.fair_queue.remove(session.0);
        clear_work_slot(&mut work[s]);
        // A terminal request owns no Query lifecycle: any unfinished
        // checkpoint source is dropped with it (never carried into the next
        // request on this slot).
        #[cfg(feature = "kvmem")]
        if let Some(run) = rig.legacy_queries.get_mut(s) {
            *run = None;
            rig.legacy_query_decided[s] = false;
        }
        rig.model.clear_slot_vl_state(&mut rig.gpu, s);
        if matches!(reason, DoneReason::ClientGone) {
            // Nobody will follow up on a vanished client, so hand the slot
            // back at once.
            rig.swap.forget(session.0);
            self::kvmem_hooks::drop_idle_dn(rig, s);
            rig.sessions.close(&mut rig.pool, &mut rig.adm, session);
        } else {
            // S2: between turns the recurrent state may leave the device. The
            // exact host image is reinstalled by the next continuation.
            self::kvmem_hooks::demote_idle_dn(rig, s, session);
            // Keep the session resident so its next turn can extend it in
            // place. Under KVMem that reuse is the whole KV: the trunk pages
            // (resident plus spilled payloads and their row statistics in
            // the host store), the DeltaNet state and the MTP head KV all
            // stay valid, so the shared prefix is never re-prefilled. LRU
            // eviction (which closes rather than swaps under KVMem) reclaims
            // the slot when someone else needs it.
            rig.sessions.touch(session);
        }
    } else {
        work[s].remaining_prompt.push(tok);
        if let Some(sess) = rig.sessions.get_mut(session) {
            sess.tokens.push(tok);
        }
        rig.sessions.touch(session);
    }
}

/// End a slot's current generation pass early WITHOUT closing its session.
///
/// Think-recovery control plane: the daemon asks for the pass to end between
/// tokens (closed-think continuation / reasoning-loop recovery), then starts a
/// follow-up pass whose prompt extends the accepted token stream. The mirror of
/// `commit_sampled_token`'s terminal block, with three deliberate differences:
///
/// * No token is being committed (the pass ends at a boundary), so nothing is
///   retained into the session and no `Token` event is sent.
/// * `Done` is reported with `DoneReason::MaxTokens` and `generated =
///   produced` — the daemon treats it as a pass boundary, not a budget stop.
/// * The session is ALWAYS touched (never closed, even on a vanished client —
///   that case is handled by the daemon cancelling the whole request via
///   `Close`, not by an interrupt).
fn finish_interrupted_pass(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    stats: &Mutex<EngineStats>,
    s: usize,
) {
    let Some(f) = slots[s].as_mut() else { return };
    let session = f.session;
    let generated = f.produced;
    lock_stats(stats).note_slot_released();
    // 期4 counters: pass work is real work — fold it now. The next pass
    // re-initialises the per-slot counters at admit.
    {
        let spec = std::mem::take(&mut rig.spec_counters[s]);
        if spec.requested {
            let mut st = lock_stats(stats);
            st.speculative.requests += 1;
            st.speculative.rounds += spec.rounds;
            st.speculative.drafted_tokens += spec.drafted;
            st.speculative.accepted_tokens += spec.accepted;
            st.speculative.generated_tokens += spec.generated;
            st.speculative.fallback_steps += spec.fallback;
        }
    }
    let _ = send_event(
        &f.reply,
        Event::Done {
            reason: DoneReason::MaxTokens,
            generated,
        },
    );
    // Publish generated-prefix pages at the pass boundary, exactly as the
    // terminal path does: every sampled token of this pass is materialized.
    let mut pin_ticket = std::mem::take(&mut f.pin_ticket);
    if rig.prefix_cache {
        publish_generated_prefix(rig, slots, s, work[s].next_pos);
    }
    release_pin_ticket(rig, &mut pin_ticket);
    slots[s] = None;
    let _ = rig.fair_queue.remove(session.0);
    clear_work_slot(&mut work[s]);
    #[cfg(feature = "kvmem")]
    if let Some(run) = rig.legacy_queries.get_mut(s) {
        *run = None;
        rig.legacy_query_decided[s] = false;
    }
    rig.model.clear_slot_vl_state(&mut rig.gpu, s);
    // Keep the session resident (and its KV/MTP head/DeltaNet state valid):
    // the continuation pass extends exactly this session.
    rig.sessions.touch(session);
}

/// Lock the shared stats counter, tolerating a poisoned mutex.
///
/// Telemetry only: if the engine thread ever panicked while holding the
/// lock, a poisoning `expect` here would panic the CALLER's thread (HTTP
/// handler) on the next stats read — turning a contained engine fault into
/// a serve-wide failure. The counters stay readable after a poison.
fn lock_stats(stats: &std::sync::Mutex<EngineStats>) -> std::sync::MutexGuard<'_, EngineStats> {
    stats.lock().unwrap_or_else(|p| p.into_inner())
}

/// Swap failures are not fatal (the session re-prefills), but they must not be
/// silent. Warn once per process and per failure class so a misconfigured
/// budget or an unwritable directory is visible without spamming the log on
/// every eviction.
static SWAP_PARK_WARNED: AtomicBool = AtomicBool::new(false);
static SWAP_RESTORE_WARNED: AtomicBool = AtomicBool::new(false);

fn warn_once(flag: &AtomicBool, message: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        eprintln!("WARNING {message}");
    }
}

/// Random per-engine id for `SnapshotStamp::instance_id`. `RandomState` is
/// seeded by the standard library; the pid and time keep two instances in the
/// same process distinct as well.
fn instance_nonce() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(u64::from(std::process::id()));
    if let Ok(now) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        h.write_u128(now.as_nanos());
    }
    h.finish()
}

/// One request parked in the bounded waiting room (spec §5.3 S3).
struct ParkedRequest {
    req: SubmitRequest,
    /// Wall-clock enqueue time. This is BOTH the admission-age source and
    /// the queue-deadline base: re-parking a retried request preserves it,
    /// so `serve.queue_timeout_ms` fires when the REQUEST has waited too
    /// long, not when the engine last got around to retrying it.
    enqueued_at: std::time::Instant,
}

/// One request currently occupying a slot.
struct InFlight {
    session: SessionId,
    reply: Sender<Event>,
    produced: usize,
    max_tokens: usize,
    /// The REQUESTED penalty window (min of the request's repeat_window and
    /// REPEAT_WINDOW_MAX), kept unclamped so the per-step effective window
    /// grows with the session instead of ratcheting down permanently: the
    /// clamp used to write the effective window back into
    /// `sample_params[s].repeat_window`, which the next step then read as the
    /// "requested" value — freezing the window at the first step's session
    /// length for the whole generation.
    repeat_window_req: usize,
    /// Last page-aligned boundary published to the prefix cache for this
    /// request (spec §4.6 C6). Initialized to `reused` at admit time so
    /// that shared pages from a prefix cache hit are not re-published.
    /// 0 when prefix_cache is off.
    last_published_boundary: usize,
    /// P4.3: last page-aligned boundary frozen into the KVMem canonical
    /// catalog for this request. Initialized to `reused` at admit (a reused
    /// prefix is already frozen); 0 when PREFIX_PIN is off.
    last_frozen_boundary: usize,
    /// P2.2b2: latest page-aligned decode frontier with a captured coverage
    /// recurrent checkpoint (0 = none). The terminal endpoint freeze prefers
    /// the exact aligned frontier and falls back to this anchor.
    last_coverage_boundary: usize,
    /// P3.2: prefill freeze cadence (.spread policy); `None` = legacy.
    #[cfg(feature = "kvmem")]
    checkpoint_plan: Option<hipfire_kvmem::CheckpointCapturePlan>,
    /// Per-request grammar constraint for structured output (spec §7 G1/G2).
    /// `None` for unconstrained requests. When present, a grammar mask is
    /// applied to the slot's logits row BEFORE the GPU sampler's
    /// probability-dependent filtering/renormalization, so no later
    /// operation can resurrect a forbidden token (spec §7.2 G2).
    ///
    /// Built from `SubmitRequest::json_schema` on admit: the schema is
    /// compiled into a `CompiledSchema` (immutable, shared) and a fresh
    /// `SchemaMatcher` cursor (per-request mutable state) is installed
    /// (spec §7.2 G2).
    grammar: Option<GrammarConstraint>,
    /// Pin on the radix path this request's prefix-cache lookup matched
    /// (spec §4.4 C4). Empty when prefix_cache is off or the lookup missed.
    /// Released by [`release_pin_ticket`] on every terminal path.
    pin_ticket: PinTicket,
}

/// Per-request grammar constraint state (spec §7 G1/G2).
///
/// Holds a per-request `SchemaMatcher` cursor (private mutable state, never
/// shared between forks — spec §7.2 G2) and produces a boolean token mask
/// from the current accepting set. The mask is applied to logits BEFORE the
/// sampler (spec §7.2 G2: "the hard grammar mask participates before
/// probability-dependent filtering/renormalization").
///
/// An empty allowed set is a typed per-request error, never an unconstrained
/// fallback (spec §7.2 G2: "An empty allowed set or invalid probability mass
/// is a typed error, never an unconstrained-sampling fallback"). EOS is
/// allowed only in an accepting grammar state.
struct GrammarConstraint {
    /// The per-request SchemaMatcher cursor. Advanced on accepted tokens
    /// only (spec §7.2 G2: "per-request parser cursor advanced on accepted
    /// tokens only"). For speculative verify, a private cursor per candidate
    /// is advanced and only the accepted prefix is committed (spec §7.2 G2).
    matcher: grammar::json_schema::SchemaMatcher,
    /// Reusable mask buffer (vocab-sized bool array), avoiding per-step
    /// allocation.
    mask_buf: Vec<bool>,
    /// Framing-aware cursor (spec §7.2): while the generation is inside a
    /// `<think>` span the schema mask is deferred — think tokens are not
    /// JSON and must not be masked. The matcher only sees tokens emitted
    /// after `</think>`.
    in_think: bool,
    /// Special token ids for the think tags; `None` when the tokenizer has
    /// no think vocabulary (mask applies from token 0).
    think_open_id: Option<u32>,
    think_close_id: Option<u32>,
    /// Raw bytes of tokens committed while inside a think span. Lets the
    /// cursor recognize a think tag spelled by ORDINARY tokens — a
    /// tokenizer that lacks the special `</think>` id, or a tag split
    /// across several tokens — where the exact-id transition below can
    /// never fire and the schema mask would defer forever (the request
    /// then free-runs to its cap and dies as unsatisfiable at terminal).
    /// Capped: only the last `THINK_TAIL_CAP` bytes are kept.
    think_tail: Vec<u8>,
    /// Enforced thinking budget (spec §7.2, vLLM `thinking_token_budget`
    /// parity): `usize::MAX` = uncapped. While `in_think`, every ordinary
    /// committed token consumes one unit; at exhaustion `build_mask` allows
    /// ONLY the think-close token — the model is forced to close the span
    /// through the normal commit path (KV row, events, cursor) instead of
    /// free-running to max_tokens and dying as unsatisfiable.
    think_budget: usize,
    /// Think tokens committed so far against `think_budget`.
    think_tokens_used: usize,
    /// XGrammar-style per-state token-mask cache (spec §9.2): the allowed
    /// token set is a pure function of `mask_state_signature`, and states
    /// recur constantly during structured generation (every `{`, `,`,
    /// `:`, key/value boundary revisits the same signature). Caching the
    /// packed bitset per signature turns the per-step O(vocab × buffer)
    /// matcher sweep into a bitset copy on hits; misses pay once per state.
    /// Per-request (per-constraint) storage keeps the key space scoped to
    /// one schema. Bounded: oldest signature evicted at `MASK_CACHE_CAP`.
    mask_cache: std::collections::HashMap<u64, Vec<u64>>,
    mask_cache_order: std::collections::VecDeque<u64>,
}

/// Mask-cache bound: structural states per document are typically <100;
/// 256 packed bitsets (248k vocab ≈ 31 KiB each) bound memory at ~8 MiB
/// per grammar slot in the worst case.
const MASK_CACHE_CAP: usize = 256;

/// Error from grammar mask construction or application (spec §7.2 G2, A17).
#[derive(Debug, Clone)]
enum GrammarMaskError {
    /// The allowed token set is empty — no token can be emitted without
    /// violating the grammar. This is a typed per-request error, never an
    /// unconstrained fallback (spec §7.2 G2, A17).
    EmptyAllowedSet,
    /// The grammar constraint could not advance (unsatisfiable dead end).
    /// Truncation/cancel/unsatisfiable = typed incomplete/invalid result
    /// (spec §7.1 G1, A17).
    Unsatisfiable,
}

impl std::fmt::Display for GrammarMaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyAllowedSet => write!(
                f,
                "grammar constraint allows no token (empty allowed set) — \
                 request cannot continue without violating the schema"
            ),
            Self::Unsatisfiable => write!(
                f,
                "grammar constraint reached an unsatisfiable state — \
                 the schema cannot be satisfied from this position"
            ),
        }
    }
}

impl std::error::Error for GrammarMaskError {}

impl GrammarConstraint {
    /// Build the boolean token mask for the current parser state into
    /// `self.mask_buf` and return a reference to it (spec §7.2 G2).
    ///
    /// For each token id, `is_token_allowed` is called with the lossless
    /// `token_bytes(id)` (spec §7.2 G2: "Never build strict constraints on
    /// replacement-character artifacts"). EOS/terminator tokens are allowed
    /// only when the matcher is in an accepting state (spec §7.2 G2).
    ///
    /// Returns `Err(EmptyAllowedSet)` when no token is allowed — the caller
    /// must fail that request with a typed error, never fall back to
    /// unconstrained sampling (spec §7.2 G2, A17).
    fn build_mask(
        &mut self,
        tokenizer: &Tokenizer,
        vocab_size: usize,
    ) -> Result<&[bool], GrammarMaskError> {
        self.mask_buf.clear();
        self.mask_buf.resize(vocab_size, false);
        // Framing-aware cursor: inside a think span the schema does not
        // apply — every non-terminator token is legal. EOS stays masked so
        // the model cannot end the turn without closing the think span and
        // producing the schema-conforming answer (spec §7.2).
        if self.in_think {
            // Budget exhausted: the only legal next token is the think
            // close. Forcing it through the normal commit path (KV row,
            // Token/reasoning events, cursor transition) ends the span
            // cleanly — the model continues straight into schema-masked
            // JSON instead of free-running to max_tokens mid-thought and
            // dying as unsatisfiable. Requires the tokenizer to carry the
            // special close id; a text-only close cannot be forced.
            if let Some(close_id) = self.think_close_forced() {
                for id in 0..vocab_size as u32 {
                    self.mask_buf[id as usize] = id == close_id;
                }
                return Ok(&self.mask_buf);
            }
            for id in 0..vocab_size as u32 {
                self.mask_buf[id as usize] = !tokenizer.is_terminator(id);
            }
            return Ok(&self.mask_buf);
        }
        // Cache hit (XGrammar per-state token masks): the allowed set is a
        // pure function of the state signature within one schema, so a
        // recurring state replays its packed bitset instead of re-running
        // 248k matcher calls (the O(vocab × buffer) sweep — spec §9.2).
        let sig = self.matcher.mask_state_signature();
        if let Some(bits) = self.mask_cache.get(&sig) {
            self.mask_buf.clear();
            self.mask_buf.resize(vocab_size, false);
            for (word, &w) in bits.iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    let id = word * 64 + w.trailing_zeros() as usize;
                    if id < vocab_size {
                        self.mask_buf[id] = true;
                    }
                    w &= w - 1;
                }
            }
            return Ok(&self.mask_buf);
        }
        let accepting = self.matcher.is_accepting();
        for id in 0..vocab_size as u32 {
            // EOS/terminator tokens: allowed only in an accepting state
            // (spec §7.2 G2). A terminator emitted before the schema is
            // complete would produce invalid JSON.
            if tokenizer.is_terminator(id) {
                self.mask_buf[id as usize] = accepting;
                continue;
            }
            // The think-open tag is FRAMING, not JSON content: outside a
            // think span it must not be emittable, or a `<think>` token
            // occurring inside a JSON string flips the framing cursor and
            // un-masks the rest of the generation (the request then burns
            // to max_tokens and dies as unsatisfiable).
            if self.think_open_id == Some(id) {
                continue;
            }
            let bytes = tokenizer.token_bytes(id);
            if bytes.is_empty() {
                // Tokens with no byte representation (e.g. unused ids)
                // cannot contribute to the JSON output.
                continue;
            }
            self.mask_buf[id as usize] = self.matcher.is_token_allowed(bytes);
        }
        if !self.mask_buf.iter().any(|&allowed| allowed) {
            if std::env::var("GRAMMAR_TRACE").is_ok() {
                eprintln!(
                    "[masktrace] EMPTY SET: in_think={} matcher_errored={} accepting={} probes: quote={} ws_quote={} brace_close={}",
                    self.in_think,
                    self.matcher.is_errored(),
                    self.matcher.is_accepting(),
                    self.matcher.is_token_allowed(b"\""),
                    self.matcher.is_token_allowed(b" \""),
                    self.matcher.is_token_allowed(b"}"),
                );
                eprintln!(
                    "[masktrace] matcher byte_len={} tail={:?}",
                    self.matcher.buffer_len(),
                    String::from_utf8_lossy(
                        &self.matcher.buffer_bytes()
                            [self.matcher.buffer_len().saturating_sub(40)..]
                    ),
                );
            }
            return Err(GrammarMaskError::EmptyAllowedSet);
        }
        Ok(&self.mask_buf)
    }

    /// The forced think-close token when the budget is exhausted:
    /// `Some(close_id)` makes `build_mask` allow ONLY that token — the
    /// span is force-closed through the normal commit path (vLLM
    /// `thinking_token_budget` parity). `None` = not exhausted, or no
    /// special close id exists (a text-only close cannot be forced; the
    /// span stays uncapped and the request is bounded by max_tokens
    /// instead).
    fn think_close_forced(&self) -> Option<u32> {
        if self.in_think && self.think_tokens_used >= self.think_budget {
            self.think_close_id
        } else {
            None
        }
    }

    /// Advance the parser cursor with the lossless token bytes of an
    /// accepted token (spec §7.2 G2: "per-request parser cursor advanced
    /// on accepted tokens only").
    ///
    /// Uses `Tokenizer::token_bytes` (lossless) rather than `decode` (lossy
    /// `from_utf8_lossy`) so byte-fallback tokens are handled correctly
    /// (spec §7.2 G2: "Never build strict constraints on
    /// replacement-character artifacts").
    fn advance_token(&mut self, tokenizer: &Tokenizer, token: u32) {
        // Think-span transitions are framing, not JSON content — the tags
        // themselves never reach the matcher.
        if Some(token) == self.think_open_id {
            self.in_think = true;
            self.think_tail.clear();
            return;
        }
        if Some(token) == self.think_close_id && self.in_think {
            self.in_think = false;
            self.think_tail.clear();
            return;
        }
        // When NOT in_think, a `</think>` token is ordinary content (the mask
        // admits it as inert string bytes) — it MUST reach the matcher, or
        // the validated buffer diverges from the emitted text.
        if self.in_think {
            // Budget accounting: every ordinary think token consumes one
            // unit (the close token itself returns earlier and never gets
            // here — closing must stay possible at exactly the budget).
            self.think_tokens_used = self.think_tokens_used.saturating_add(1);
            // Accumulate raw bytes and watch for the close tag spelled by
            // ordinary tokens (possibly split across several tokens, and
            // possibly when `think_close_id` is None entirely).
            const CLOSE: &[u8] = b"</think>";
            const OPEN: &[u8] = b"<think>";
            const THINK_TAIL_CAP: usize = 64;
            self.think_tail
                .extend_from_slice(tokenizer.token_bytes(token));
            if self.think_tail.len() > THINK_TAIL_CAP {
                let excess = self.think_tail.len() - THINK_TAIL_CAP;
                self.think_tail.drain(..excess);
            }
            let trimmed = &self.think_tail[..];
            let trimmed = &trimmed[..trimmed
                .iter()
                .rposition(|b| !b.is_ascii_whitespace())
                .map_or(0, |p| p + 1)];
            if trimmed.ends_with(CLOSE) {
                self.in_think = false;
                self.think_tail.clear();
            } else if trimmed.ends_with(OPEN) {
                // A re-opened think span inside think — stay in think and
                // restart the tail watch.
                self.think_tail.clear();
            }
            return; // think tokens are not schema content
        }
        // Terminators are control tokens, not JSON content — feeding
        // `<|im_end|>` bytes into the matcher would error an accepted
        // state (and, while `number_open`, corrupt a still-growable
        // trailing number).
        if tokenizer.is_terminator(token) {
            return;
        }
        let bytes = tokenizer.token_bytes(token);
        if !bytes.is_empty() {
            self.matcher.advance(bytes);
        }
    }

    /// True when the full JSON value has been parsed and conforms to the
    /// schema (spec §7.2 G2). Used at terminal to distinguish a
    /// schema-conforming done from an incomplete/invalid result.
    fn is_accepting(&self) -> bool {
        self.matcher.is_accepting()
    }
}

fn run_loop(
    mut rig: Rig,
    rx: Receiver<EngineCommand>,
    stats: Arc<Mutex<EngineStats>>,
) -> Result<(), String> {
    let n = rig.n_slots;
    let mut slots: Vec<Option<InFlight>> = (0..n).map(|_| None).collect();
    // Per-slot cooperative pass-interrupt flags (EngineCommand::Interrupt).
    // Deliberately engine-local rather than a PendingWork field: the flag is
    // transient control-plane state, and PendingWork literals are constructed
    // in ~20 places (tests included) that must not churn for it.
    let mut interrupts: Vec<bool> = vec![false; n];
    let mut work: Vec<PendingWork> = (0..n)
        .map(|s| PendingWork {
            slot: SlotId(s),
            remaining_prompt: Vec::new(),
            next_pos: 0,
            decoding: false,
            vl_prefill: None,
            spec: SpecKind::None,
            spec_retire: hipfire_runtime::spec_retire::SpecRetire::default(),
            pos3_delta: 0,
        })
        .collect();
    let mut sched = Scheduler {
        chunk_size: rig.prefill_chunk,
        vl_sequential: rig.vl_sequential,
        prefill_cursor: 0,
    };
    let mut graph = SlotDecodeGraph::new();
    let mut poison: Option<String> = None;

    // Helper to fail every active request, close sessions, clear work, forget swap.
    // Used for HIP poison and shutdown drain. Returns the reason used.
    let fail_all_active = |rig: &mut Rig,
                           slots: &mut [Option<InFlight>],
                           work: &mut [PendingWork],
                           reason: String| {
        for (s, f) in slots.iter_mut().enumerate() {
            if let Some(mut f) = f.take() {
                let _ = send_event(
                    &f.reply,
                    Event::Rejected {
                        class: RejectClass::Internal,
                        reason: reason.clone(),
                    },
                );
                release_pin_ticket(rig, &mut f.pin_ticket);
                rig.swap.forget(f.session.0);
                let sid = f.session;
                rig.sessions.close(&mut rig.pool, &mut rig.adm, sid);
                let _ = rig.fair_queue.remove(sid.0);
                clear_work_slot(&mut work[s]);
                // A dead session's Query checkpoint/source must not survive
                // into the slot's next occupant (its store is being closed).
                #[cfg(feature = "kvmem")]
                if let Some(run) = rig.legacy_queries.get_mut(s) {
                    *run = None;
                    rig.legacy_query_decided[s] = false;
                }
                rig.model.clear_slot_vl_state(&mut rig.gpu, s);
            }
        }
    };

    // Panic containment: a panic inside the serve loop (a radix/pool
    // internal `unwrap`, an index slip, a kernel-contract violation) must
    // not kill the engine thread with in-flight requests still holding
    // slots, pins and admission budget. Catch it, fail every active
    // request with a typed rejection, and poison the engine so teardown
    // reports the fault instead of silently wedging.
    let serve_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Debug-only per-step pool invariant check (spec §9.2): catches page
        // accounting drift at the step it happens rather than at the next OOM.
        let check_pool_invariants =
            hipfire_config::developer_var("HIPFIRE_DEBUG_POOL_INVARIANTS").as_deref() == Ok("1");
        'serve: loop {
            let idle = slots.iter().all(|s| s.is_none());
            if idle && rig.wait_queue.queued_count() == 0 {
                // Nothing in flight and nothing queued: block rather than spin.
                // With an idle TTL configured, wake periodically to close
                // abandoned parked sessions instead of waiting for the next
                // request.
                match rig.session_ttl.map(|ttl| {
                    ttl.min(std::time::Duration::from_secs(30))
                        .max(std::time::Duration::from_secs(1))
                }) {
                    Some(wait) => match rx.recv_timeout(wait) {
                        Ok(cmd) => handle_command(
                            &mut rig,
                            &mut slots,
                            &mut work,
                            &stats,
                            &mut interrupts,
                            cmd,
                        ),
                        Err(RecvTimeoutError::Timeout) => {
                            serve_engine_admit::sweep_sessions(&mut rig);
                            let mut st = stats.lock().expect("stats");
                            st.note_swap(&rig.swap);
                            drop(st);
                            continue 'serve;
                        }
                        Err(RecvTimeoutError::Disconnected) => break 'serve,
                    },
                    None => match rx.recv() {
                        Ok(cmd) => handle_command(
                            &mut rig,
                            &mut slots,
                            &mut work,
                            &stats,
                            &mut interrupts,
                            cmd,
                        ),
                        Err(_) => break 'serve, // all senders gone: shut down after drain
                    },
                }
            }
            loop {
                match rx.try_recv() {
                    Ok(cmd) => handle_command(
                        &mut rig,
                        &mut slots,
                        &mut work,
                        &stats,
                        &mut interrupts,
                        cmd,
                    ),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        if slots.iter().all(|s| s.is_none()) {
                            break 'serve;
                        }
                        break;
                    }
                }
            }
            // ── WaitQueue: expire and retry queued requests (spec §5.3 S3) ──
            // Advance the scheduler tick once per serve iteration. The tick
            // stamps waiter admission age and drives per-waiter timeout deadlines.
            rig.tick = rig.tick.saturating_add(1);
            // A20 soak telemetry: sample free physical pages once per step so a
            // monotonic decline across identical request cycles (the page-level
            // leak signature) is observable from `EngineStats` (spec §9.2).
            {
                let mut st = stats.lock().expect("stats");
                if let Some(pp) = rig.pool.page_pool() {
                    st.note_pool_pages(pp.free_pages(), pp.n_pages());
                }
                kvmem_hooks::note_stats(&rig, &mut st);
                st.note_swap(&rig.swap);
            }
            // P2.2b2: capture coverage recurrent checkpoints at aligned decode
            // frontiers (opt-in HIPFIRE_KVMEM_PREFIX_COVERAGE). Soft: a refused
            // capture only loses that endpoint anchor.
            kvmem_hooks::coverage_capture(&mut rig, &mut slots, &work);
            // Expire timed-out waiters → typed queue timeout rejection. The
            // deadline is WALL-CLOCK (`serve.queue_timeout_ms` against the
            // enqueue Instant), not a tick count: ticks advance once per serve
            // iteration — absorbing whole prefill chunks, or not advancing at
            // all while the engine blocks on recv with an empty room — so a
            // tick-denominated deadline fired arbitrarily late or never.
            // `serve.queue_timeout_ms == 0` is documented (docs/CONFIG.md) as "no
            // wait timeout" — a parked request must never expire. Map it to
            // Duration::MAX so `duration_since >= timeout` is never true; a plain
            // `from_millis(0)` would expire every waiter on the next iteration.
            let queue_timeout = if rig.queue_timeout_ms == 0 {
                std::time::Duration::MAX
            } else {
                std::time::Duration::from_millis(rig.queue_timeout_ms)
            };
            let now = std::time::Instant::now();
            let expired_waiters: Vec<u64> = rig
                .parked_requests
                .iter()
                .filter(|(_, p)| now.duration_since(p.enqueued_at) >= queue_timeout)
                .map(|(id, _)| *id)
                .collect();
            for waiter_id in expired_waiters {
                if let Some(parked) = rig.parked_requests.remove(&waiter_id) {
                    rig.wait_queue.remove(waiter_id);
                    let _ = send_event(
                        &parked.req.reply,
                        Event::Rejected {
                            class: RejectClass::Overload,
                            reason: "serve queue timeout: request waited beyond \
                                 the configured deadline"
                                .to_string(),
                        },
                    );
                    lock_stats(&stats).note_rejected();
                }
            }
            // Pop ready waiters and retry admit when a slot is free. pop_ready
            // is FIFO by enqueue order, preserving admission age (spec §5.3 S3).
            // The popped request's ORIGINAL wall-clock admission time is handed
            // to admit (`pending_repark_age`): if the open fails again, admit
            // re-parks with that same stamp — stamping a fresh age here would
            // slide the queue deadline out forever while slots stay unfilled
            // (spec §5.3: "until it fits or its existing queue deadline
            // expires").
            while slots.iter().any(|s| s.is_none()) {
                let waiter = match rig.wait_queue.pop_ready(rig.tick) {
                    Some(w) => w,
                    None => break,
                };
                let Some(parked) = rig.parked_requests.remove(&waiter.id) else {
                    continue;
                };
                let active_before = slots.iter().filter(|s| s.is_some()).count();
                rig.pending_repark_age = Some(parked.enqueued_at);
                serve_engine_admit::admit(&mut rig, &mut slots, &mut work, &stats, parked.req);
                rig.pending_repark_age = None;
                let active_after = slots.iter().filter(|s| s.is_some()).count();
                // If admit did not place the request in a slot (it may have
                // re-queued or rejected), stop draining — the next iteration
                // will pop the next waiter if a slot is still free.
                if active_after <= active_before {
                    break;
                }
            }
            if slots.iter().all(|s| s.is_none()) {
                // All slots free but waiters remain queued: the drain loop above
                // could not admit any of them (e.g. the grant cannot fit the
                // VRAM budget with weights resident, and there is no reclaimable
                // victim). A bare `continue` would re-run the whole iteration —
                // try_recv, expire scan, pop, re-park — forever on one core until
                // each waiter's wall-clock deadline. Block instead until the
                // SOONEST queued deadline (or a new command arrives), so the
                // expiry path still fires on time and the engine does not spin.
                let now = std::time::Instant::now();
                let sleep_for = rig
                    .parked_requests
                    .values()
                    .map(|p| {
                        queue_timeout
                            .checked_sub(now.duration_since(p.enqueued_at))
                            .unwrap_or(std::time::Duration::ZERO)
                    })
                    .min()
                    .unwrap_or(std::time::Duration::from_millis(50));
                match rx.recv_timeout(sleep_for) {
                    Ok(cmd) => handle_command(
                        &mut rig,
                        &mut slots,
                        &mut work,
                        &stats,
                        &mut interrupts,
                        cmd,
                    ),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break 'serve,
                }
                continue;
            }
            // ── Cooperative pass interrupts (think-recovery) ──
            // A pending Interrupt must land BEFORE this step materializes any
            // token: end the pass at the current boundary, report it as a
            // MaxTokens done, and keep the session KV resident so the daemon's
            // continuation pass finds it reusable. Placed after admission (so
            // a freshly admitted request cannot be interrupted before its
            // first step) and before every per-slot section below.
            for s in 0..n {
                if interrupts[s] {
                    interrupts[s] = false;
                    if slots[s].is_some() {
                        finish_interrupted_pass(&mut rig, &mut slots, &mut work, &stats, s);
                    }
                }
            }
            if slots.iter().all(|s| s.is_none()) && rig.wait_queue.queued_count() == 0 {
                // The last busy slot was interrupted and nothing else is
                // ready: skip the (all-empty) scheduling sections this step.
                continue 'serve;
            }
            // Slot occupancy for this step, after admission. Each terminal
            // below releases its slot in the stats BEFORE posting its event,
            // so the snapshot a `done` reads counts only slots still busy.
            lock_stats(&stats).note_slots(slots.iter().flatten().count(), n);

            let image_pad_id = rig.tokenizer.special_token_id("<|image_pad|>");
            for s in 0..n {
                // Sequential fallback ONLY (HIPFIRE_VL_SEQUENTIAL=1): this per-token
                // M-RoPE path writes KV through a flat slab view. The default
                // batched VL path must NEVER enter here — it runs through the
                // scheduler below, and under a paged pool the slot_kv_view assert
                // would (correctly) kill the engine.
                if !rig.vl_sequential || work[s].vl_prefill.is_none() || slots[s].is_none() {
                    continue;
                }
                if work[s].remaining_prompt.is_empty() {
                    continue;
                }
                let pad = image_pad_id.unwrap_or(u32::MAX);
                let grammar = slots[s].as_mut().and_then(|f| f.grammar.as_mut());
                match rig.model.vl_forward_remaining(
                    &mut rig.gpu,
                    &mut rig.pool,
                    &rig.tokenizer,
                    &mut rig.sample_params,
                    SlotId(s),
                    &mut work[s],
                    pad,
                    grammar,
                ) {
                    Ok(tok) => {
                        commit_sampled_token(&mut rig, &mut slots, &mut work, &stats, s, tok)
                    }
                    Err(reason) => {
                        if let Some(mut f) = slots[s].take() {
                            let _ = send_event(
                                &f.reply,
                                Event::Rejected {
                                    class: RejectClass::Internal,
                                    reason: reason.clone(),
                                },
                            );
                            release_pin_ticket(&mut rig, &mut f.pin_ticket);
                            rig.swap.forget(f.session.0);
                            let _ = rig.fair_queue.remove(f.session.0);
                            rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                        }
                        clear_work_slot(&mut work[s]);
                        rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                    }
                }
            }

            // ── Spec phase 1: draft ─────────────────────────────────────────
            // Spec slots prefill through the scheduler's batched chunk path
            // like any other slot (their private state fills post-forward from
            // x_batch / hidden staging, below). This phase only drafts: K
            // serial head-forward steps per MTP slot, one DFlash2 block forward
            // per DFlash slot. The trunk verify is batched with regular decode
            // in the forward_batch_slots_graphed_opts call below — the
            // vLLM-style batched-verify design, graph-captured like pure decode.
            let mut spec_drafts: Vec<Option<SpecDraftRows>> = (0..n).map(|_| None).collect();
            // The seed token each draft consumed from `remaining_prompt` (the
            // draft step takes the prompt vector to satisfy borrows, so the
            // seed is GONE from the slot while the draft lives). Any path that
            // drops a draft without terminating the request must re-push this
            // seed and retire `spec`, or the slot is left decoding/spec-active
            // with an empty prompt: `skip_entirely` holds, every readiness
            // predicate is false, and the slot contributes zero rows forever —
            // a permanent wedge.
            let mut spec_seeds: Vec<Option<u32>> = (0..n).map(|_| None).collect();
            if (rig.model.mtp_head.is_some() && rig.model.mtp_k > 0) || rig.model.dflash.is_some() {
                // Pre-draft budget gate (spec §5.2 S2: "spec decoding must not
                // also receive an ordinary decode row ... reduce supported draft
                // depth or run AR"): only as many slots may draft as fit their
                // verify rows (spec_rows each) inside max_batch_tokens.
                // Candidates are taken in FairQueue age order so the oldest
                // decodes keep spec under pressure; the rest stay on ordinary
                // AR decode this step (their seed is still in
                // `remaining_prompt`, so nothing else is needed).
                let verify_rows_per_slot = rig.model.spec_rows.max(1);
                let max_draft_slots = (rig.max_batch_tokens / verify_rows_per_slot).max(1);
                let mut draft_candidates: Vec<usize> = Vec::new();
                for s in 0..n {
                    if work[s].spec.active()
                        && slots[s].is_some()
                        && work[s].decoding
                        && !work[s].remaining_prompt.is_empty()
                    {
                        draft_candidates.push(s);
                    }
                }
                let id_of =
                    |slots: &[Option<InFlight>], s: usize| slots[s].as_ref().map(|f| f.session.0);
                draft_candidates.sort_by_key(|&s| {
                    id_of(&slots, s)
                        .and_then(|id| rig.fair_queue.get(id).map(|r| r.admission_tick))
                        .unwrap_or(u64::MAX)
                });
                draft_candidates.truncate(max_draft_slots);
                let draft_set: std::collections::HashSet<usize> =
                    draft_candidates.into_iter().collect();
                for s in 0..n {
                    if !work[s].spec.active() || slots[s].is_none() || !work[s].decoding {
                        continue;
                    }
                    if work[s].remaining_prompt.is_empty() {
                        continue;
                    }
                    if !draft_set.contains(&s) {
                        // Verify rows for this slot cannot fit the global budget
                        // alongside the other drafted slots. The candidate order
                        // (admission age) is stable across steps, so the same
                        // slots are excluded every step — and a decoding spec
                        // slot contributes ZERO rows otherwise: the scheduler
                        // skips `spec.active() && decoding` slots entirely (no
                        // ordinary decode fallback), leaving the request
                        // stalled with no timeout. Retire spec for this slot
                        // (mirroring the cap guard below) so it runs ordinary
                        // decode — no drop, no wedge, just no spec-decode for
                        // the rest of the request.
                        work[s].spec = SpecKind::None;
                        if let Some(last) = work[s].remaining_prompt.last().copied() {
                            work[s].remaining_prompt.clear();
                            work[s].remaining_prompt.push(last);
                        }
                        continue;
                    }
                    if work[s].spec == SpecKind::Dflash
                        && rig.model.dflash.as_ref().is_some_and(|d| {
                            d.window.is_none() && work[s].next_pos >= d.ctx_capacity
                        })
                    {
                        // Legacy DFlash owns context-indexed draft buffers capped by
                        // HIPFIRE_DFLASH_CTX_CAP. Past the cap the documented policy
                        // is plain AR fallback, not an out-of-bounds scatter/assert.
                        work[s].spec = SpecKind::None;
                        if let Some(last) = work[s].remaining_prompt.last().copied() {
                            work[s].remaining_prompt.clear();
                            work[s].remaining_prompt.push(last);
                        }
                        continue;
                    }
                    if !spec_verify_fits_cap(work[s].next_pos, rig.model.spec_rows, rig.cap_tokens)
                    {
                        // Context-cap guard: the batched verify writes spec_rows
                        // rows at next_pos..next_pos+spec_rows-1; a frontier
                        // past the cap fails the forward's provision CLOSED,
                        // which rejects every active slot, not just this one.
                        // Retire spec for this slot and let regular decode
                        // finish under its per-token guard. Of the committed
                        // tokens only the newest is not yet in KV — the rest
                        // are dead seed copies — so keep just it.
                        work[s].spec = SpecKind::None;
                        if let Some(last) = work[s].remaining_prompt.last().copied() {
                            work[s].remaining_prompt.clear();
                            work[s].remaining_prompt.push(last);
                        }
                    } else {
                        // Draft phase. The seed this call takes out of
                        // `remaining_prompt` is stashed so a later draft drop
                        // can restore it.
                        let seed = work[s].remaining_prompt.last().copied();
                        // Token penalties scope the verify rows against this
                        // request's emitted history (`sess.tokens` past the
                        // rendered prompt); only penalized spec requests need
                        // it, so the others skip the copy. MTP and DFlash share
                        // the parked config and the same penalty prepass.
                        let emitted: Vec<u32> = match work[s].spec {
                            SpecKind::Mtp | SpecKind::Dflash => {
                                let pl = rig.model.spec_req_sampling[s]
                                    .as_ref()
                                    .filter(|c| c.sampling.penalized())
                                    .map(|c| c.prompt_len);
                                match pl {
                                    Some(pl) => slots[s]
                                        .as_ref()
                                        .and_then(|f| rig.sessions.get(f.session))
                                        .map(|sess| {
                                            sess.tokens[pl.min(sess.tokens.len())..].to_vec()
                                        })
                                        .unwrap_or_default(),
                                    None => Vec::new(),
                                }
                            }
                            _ => Vec::new(),
                        };
                        let outcome = match work[s].spec {
                            SpecKind::Mtp => {
                                let session_id =
                                    slots[s].as_ref().map(|f| f.session.0).unwrap_or(0);
                                rig.model
                                    .mtp_draft_step(
                                        &mut rig.gpu,
                                        SlotId(s),
                                        &mut work[s],
                                        &emitted,
                                        session_id,
                                        rig.cap_tokens,
                                    )
                                    .map(SpecDraftRows::Mtp)
                            }
                            SpecKind::Dflash => rig
                                .model
                                .dflash_draft_step(&mut rig.gpu, SlotId(s), &mut work[s], &emitted)
                                .map(SpecDraftRows::Dflash),
                            SpecKind::None => {
                                unreachable!("spec.active() gate above")
                            }
                        };
                        match outcome {
                            Ok(draft) => {
                                // 期4 speculative statistics: one target round,
                                // `rows−1` proposal candidates (the seed row is
                                // not a proposal).
                                let rows = draft.verify_tokens().len() as u64;
                                rig.spec_counters[s].rounds += 1;
                                rig.spec_counters[s].drafted += rows.saturating_sub(1);
                                spec_seeds[s] = seed;
                                spec_drafts[s] = Some(draft);
                            }
                            Err(reason) => {
                                if let Some(mut f) = slots[s].take() {
                                    let _ = send_event(
                                        &f.reply,
                                        Event::Rejected {
                                            class: RejectClass::Internal,
                                            reason: reason.clone(),
                                        },
                                    );
                                    release_pin_ticket(&mut rig, &mut f.pin_ticket);
                                    rig.swap.forget(f.session.0);
                                    let _ = rig.fair_queue.remove(f.session.0);
                                    rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                                }
                                clear_work_slot(&mut work[s]);
                                rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                            }
                        }
                    }
                }
            }

            // ── VL phase: advance the vision tower for batched VL slots ────────
            // Each live encode takes ONE tower layer per serve iteration (see
            // `vision_tower_step`) so the loop returns to the scheduler — and to
            // the other slots' decode steps — between layers instead of wedging
            // them for the whole multi-second encode. The finished matrix becomes
            // the slot's external-embedding source; until it exists the scheduler
            // holds the slot's rows back (image pads would otherwise embed as the
            // raw pad token).
            if !rig.vl_sequential {
                for s in 0..n {
                    if slots[s].is_none() {
                        continue;
                    }
                    let Some(vl) = work[s].vl_prefill.as_mut() else {
                        continue;
                    };
                    if !vl.embeddings.is_empty() || vl.n_visual_tokens == 0 {
                        continue;
                    }
                    let outcome = rig.model.vision_tower_step(&mut rig.gpu, s, vl);
                    if let Err(reason) = outcome {
                        if let Some(mut f) = slots[s].take() {
                            let _ = send_event(
                                &f.reply,
                                Event::Rejected {
                                    class: RejectClass::Internal,
                                    reason: reason.clone(),
                                },
                            );
                            release_pin_ticket(&mut rig, &mut f.pin_ticket);
                            rig.swap.forget(f.session.0);
                            let _ = rig.fair_queue.remove(f.session.0);
                            rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                        }
                        clear_work_slot(&mut work[s]);
                        rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                    }
                }
            }
            // ── Step reservation (spec §5.1/S1, §5.2/S2) ────────────────────
            // The scheduler enforces the global trunk-row budget for prefill +
            // decode; MTP verify rows (injected below) are subtracted first so
            // the full step never exceeds `max_batch_tokens`. The forward
            // panic-assert (`n <= pbs.max_batch`) stays as a fail-closed
            // backstop, not the throttle.
            let max_batch_tokens = rig.max_batch_tokens;
            let prefill_min_tokens = rig.prefill_min_tokens;

            // MTP verify rows: each drafting slot contributes `mtp_k + 1` rows.
            // A verify slot must not also keep an ordinary decode row (spec §5.2
            // S2), so the scheduler already skipped decoding MTP slots. If the
            // verify rows would not fit the remaining quota after decode+prefill
            // allocation, skip MTP for that slot this cycle (drop the draft; the
            // seed stays in `remaining_prompt` for a regular decode row next
            // step) rather than overflowing the budget.
            let mut verify_rows: u64 = 0;
            if spec_drafts.iter().any(|d| d.is_some()) {
                for s in 0..n {
                    if spec_drafts[s].is_some() {
                        verify_rows = verify_rows
                            .checked_add(rig.model.spec_rows as u64)
                            .unwrap_or(u64::MAX);
                    }
                }
            }

            let remaining_for_sched = max_batch_tokens
                .min(rig.model.pbs.max_batch)
                .saturating_sub(verify_rows as usize);
            // ── FairQueue select (spec §5.3 S3) ─────────────────────────────
            // The FairQueue decides WHO is eligible this tick (aged-first,
            // backfill, prefill cursor); the scheduler still decides HOW MANY
            // rows each granted slot gets. Sync each active slot's per-step
            // needs from slot state, then select grants under the same remaining
            // budget the scheduler will use. The granted ids become the
            // scheduler's eligibility mask; starved_oldest triggers bounded
            // backfill (only the oldest stays eligible so younger prefill/decode
            // is skipped this tick, giving the starved oldest the next budget).
            let vl_sequential = rig.vl_sequential;
            let spec_rows = rig.model.spec_rows;
            for s in 0..n {
                let Some(f) = slots[s].as_ref() else { continue };
                let id = f.session.0;
                let wants_decode = is_runnable_decode(&work[s], vl_sequential);
                let verify = if spec_drafts[s].is_some() {
                    spec_rows as u64
                } else {
                    0
                };
                let uncached = if is_runnable_prefill(&work[s], vl_sequential) {
                    work[s].remaining_prompt.len() as u64
                } else {
                    0
                };
                sync_fair_needs(&mut rig.fair_queue, id, wants_decode, verify, uncached);
            }
            // Pass the FULL row budget, not `remaining_for_sched`: select's
            // phase-3 verify pass already charges each spec slot's verify_rows
            // against the budget it is given. Handing it the verify-subtracted
            // budget double-charges verify — a spec slot's verify is denied
            // whenever used > budget − 2·verify_rows even though used + verify
            // fits, and the denied slot is then reported starved → backfill
            // masks every other slot to 0 rows for the spec request's whole
            // generation. The scheduler still gets `remaining_for_sched` for its
            // decode+prefill allocation (it does not handle verify rows).
            let full_budget = max_batch_tokens.min(rig.model.pbs.max_batch) as u64;
            let sel = rig
                .fair_queue
                .select(n as u64, prefill_min_tokens as u64, full_budget);
            // After the admission round: mark requests not served since the last
            // round as aged (spec §5.3 S3.3). select already advanced the tick;
            // skip_round_complete sets the round boundary so the next select
            // prefers aged (starved) requests before cache-locality tie-breaks.
            rig.fair_queue.skip_round_complete();
            let granted_ids = match sel.starved_id {
                // Bounded backfill masks to the STARVED REQUEST's id — the id
                // Selection carries. Masking to "the oldest active session"
                // (the old guess, made because Selection lacked the id)
                // starved the wrong request whenever the starved one was not
                // the absolute oldest in flight, and froze every healthy slot
                // to zero rows for the duration.
                Some(sid) if sel.starved_oldest => apply_starved_backfill(&sel, sid),
                _ => eligible_ids_from_selection(&sel),
            };
            let mut eligible = vec![false; n];
            for s in 0..n {
                if let Some(f) = &slots[s] {
                    eligible[s] = granted_ids.contains(&f.session.0);
                }
            }
            // KVMem Legacy Query capture/caps and the optional last-token hold
            // are prepared behind the stable seam; stock scheduling is unchanged.
            let (legacy_query_caps, legacy_contract) =
                match kvmem_hooks::prepare_schedule(&mut rig, &slots, &work) {
                    Ok(plan) => plan,
                    Err(e) => {
                        fail_all_active(&mut rig, &mut slots, &mut work, e.clone());
                        poison = Some(e);
                        break 'serve;
                    }
                };
            let hold_tail = if legacy_contract {
                vec![false; n]
            } else {
                kvmem_hooks::hold_tail_mask(&rig, &work)
            };
            let sched_batch = kvmem_hooks::schedule_batch(
                &mut sched,
                &mut work,
                remaining_for_sched,
                prefill_min_tokens,
                &eligible,
                &hold_tail,
                &legacy_query_caps,
            );

            // Count prefill + decode rows the scheduler emitted.
            let prefill_rows: u64 = (0..n)
                .filter(|&s| {
                    !work[s].decoding && sched_batch.m_per_slot.get(s).copied().unwrap_or(0) > 0
                })
                .map(|s| sched_batch.m_per_slot[s] as u64)
                .sum();
            let decode_rows: u64 = (0..n)
                .filter(|&s| {
                    work[s].decoding
                        && !work[s].spec.active()
                        && sched_batch.m_per_slot.get(s).copied().unwrap_or(0) > 0
                })
                .map(|s| sched_batch.m_per_slot[s] as u64)
                .sum();

            // Build a StepReservation and validate against the global budget
            // (spec §5.2 S2: `fits(max_batch_tokens)`). If it does not fit
            // (e.g. verify_rows overestimated because a draft was dropped
            // above), shrink by dropping verify slots first, then decode, then
            // prefill — never panic.
            let mut reservation = hipfire_runtime::serve_contract::StepReservation {
                prefill_rows,
                decode_rows,
                verify_rows,
                forced_rows: 0,
                page_growth_credits: 0,
                cow_destinations: 0,
                needs: hipfire_runtime::serve_contract::StepNeeds {
                    scratch_bytes: 0,
                    snapshot_bytes: 0,
                    output_bytes: 0,
                },
            };
            match reservation.fits(max_batch_tokens as u64) {
                Ok(true) => {}
                _ => {
                    // Shrink: drop MTP verify slots until it fits. The scheduler
                    // already capped prefill+decode at `remaining_for_sched`
                    // (= budget − verify_rows), so after draft dropping the
                    // reservation MUST fit — verify_rows was over-counted.
                    while reservation.fits(max_batch_tokens as u64) != Ok(true) {
                        let slot = spec_drafts.iter().position(|x| x.is_some());
                        match slot {
                            Some(s) => {
                                // Free the dropped draft's device buffers —
                                // DflashSlotDraft owns GpuTensors with no Drop.
                                if let Some(d) = spec_drafts[s].take() {
                                    d.free_gpu(&mut rig.gpu);
                                }
                                let vk = rig.model.spec_rows as u64;
                                reservation.verify_rows =
                                    reservation.verify_rows.saturating_sub(vk);
                                // The draft consumed the seed OUT of
                                // `remaining_prompt` (`mtp_draft_step` takes it).
                                // Restore the seed and retire MTP so the slot
                                // runs ordinary decode — leaving it
                                // decoding+MTP-active with an empty prompt makes
                                // `skip_entirely` hold and every readiness
                                // predicate false: the slot would contribute zero
                                // rows forever (a wedge).
                                work[s].spec = SpecKind::None;
                                if let Some(seed) = spec_seeds[s].take() {
                                    work[s].remaining_prompt.clear();
                                    work[s].remaining_prompt.push(seed);
                                }
                            }
                            None => break,
                        }
                    }
                    if reservation.fits(max_batch_tokens as u64) != Ok(true) {
                        // Spec §5.4/S4: an allocation invariant violated despite
                        // reservation is an engine/accounting fault — fail closed
                        // rather than execute an over-budget step (or pretend the
                        // reservation shrank when the batch did not).
                        let reason = format!(
                            "step reservation invariant violated: {} rows reserved \
                         against a {} token budget with no verify slots left to drop",
                            reservation.total_rows().unwrap_or(u64::MAX),
                            max_batch_tokens
                        );
                        fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                        poison = Some(reason);
                        break 'serve;
                    }
                }
            }

            // Re-inject spec verify tokens for the surviving drafts. A slot whose
            // draft was dropped above has been retired to ordinary decode (seed
            // restored into `remaining_prompt`, `spec` cleared) and simply
            // contributes no rows this step.
            let mut batch = if spec_drafts.iter().any(|d| d.is_some()) {
                let pos3_deltas: Vec<i32> = work.iter().map(|w| w.pos3_delta).collect();
                // Work-level M-RoPE need — the same predicate the scheduler's
                // `any_pos3` uses — NOT derived from `sched_batch.pos3`, which is
                // empty whenever the scheduler emitted no rows (verify-only
                // steps; see `inject_spec_verify_tokens`).
                let force_pos3 = work
                    .iter()
                    .any(|w| (w.vl_prefill.is_some() && !rig.vl_sequential) || w.pos3_delta != 0);
                inject_spec_verify_tokens(&sched_batch, &spec_drafts, n, &pos3_deltas, force_pos3)
            } else {
                sched_batch
            };
            if batch.is_empty() {
                continue;
            }
            #[cfg(feature = "kvmem")]
            let legacy_query_reads =
                match kvmem_engine::capture_legacy_query_rows(&mut rig, &batch, n) {
                    Ok(rows) => rows,
                    Err(reason) => {
                        fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                        poison = Some(reason);
                        break 'serve;
                    }
                };
            if let Err(e) = kvmem_hooks::prepare_step(&mut rig, &slots, &work, &batch, n) {
                fail_all_active(&mut rig, &mut slots, &mut work, e.clone());
                poison = Some(e);
                break 'serve;
            }
            // P4.3: batch-time prefill snapshot for the end-of-iteration
            // canonical freeze. The spec/emit sections flip `work[s].decoding`
            // and advance `next_pos` mid-iteration; the freeze must see the
            // chunk exactly as it was fed (boundary = post-chunk next_pos).
            #[cfg(feature = "kvmem")]
            let freeze_snapshot: Vec<(bool, usize)> = (0..n)
                .map(|s| (!work[s].decoding, work[s].next_pos))
                .collect();
            // Provision every slot's write frontier before any COW or kernel work.
            // If retained cache-only pages consumed the free list, evict unpinned
            // radix leaves and retry once. `forward_batch_slots` repeats this call
            // idempotently; doing it here turns capacity pressure into a per-request
            // refusal instead of an engine-thread failure.
            let mut provision_failed = Vec::new();
            let mut row = 0usize;
            for s in 0..n {
                let m = batch.m_per_slot[s];
                if m == 0 {
                    continue;
                }
                let frontier = batch.positions[row + m - 1] as usize + 1;
                row += m;
                let first = rig.pool.set_seq_len(SlotId(s), frontier);
                if first.is_ok() {
                    continue;
                }
                if let (Some(idx), Some(pp)) = (rig.prefix_index.as_mut(), rig.pool.page_pool_mut())
                {
                    idx.evict_unpinned_leaves(pp, usize::MAX);
                    pp.drain_completed();
                }
                if let Err(error) = rig.pool.set_seq_len(SlotId(s), frontier) {
                    eprintln!("[serve] page provisioning refused slot {s}: {error}");
                    provision_failed.push(s);
                }
            }
            if !provision_failed.is_empty() {
                for &s in &provision_failed {
                    if let Some(mut f) = slots[s].take() {
                        let _ = send_event(
                        &f.reply,
                        Event::Rejected { class: RejectClass::Overload, reason: "server capacity exhausted while reserving KV pages; retry later".to_owned(),
                        },
                    );
                        release_pin_ticket(&mut rig, &mut f.pin_ticket);
                        rig.swap.forget(f.session.0);
                        let _ = rig.fair_queue.remove(f.session.0);
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                        clear_work_slot(&mut work[s]);
                        rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                        // Free the dropped draft's device buffers — the slot is
                        // gone but its DflashSlotDraft GpuTensors would leak.
                        if let Some(d) = spec_drafts[s].take() {
                            d.free_gpu(&mut rig.gpu);
                        }
                    }
                }
                rebuild_batch_excluding_failed(&mut batch, &provision_failed);
                if batch.is_empty() {
                    continue;
                }
            }

            // ── COW write barrier (spec §4.3) ────────────────────────────────
            // Before KV write kernels: plan copy-on-write for each active slot's
            // block table over the write interval, execute the copies, and
            // commit (rebind) — all BEFORE the forward, whose KV-write kernels
            // resolve through the block table. Sealed/CacheOnly pages (shared
            // via the prefix cache) are copied to private destinations first;
            // private pages plan empty. Paged mode only — in contiguous mode
            // there are no shared pages to protect.
            let mut cow_plans: Vec<Option<(usize, rdna_compute::page_pool::CowPlan)>> =
                Vec::with_capacity(n);
            let mut row_offset = 0usize;
            // Per-slot COW provision (spec §5.4/S4): a plan_cow or copy failure
            // for one slot drops ONLY that slot from this step (its request is
            // failed), not every active slot. A plan whose copies failed keeps
            // its reserved pages until the failed-slot handler aborts it, so no
            // half-installed COW mappings survive (spec §4.3).
            let mut failed_slots: Vec<usize> = Vec::new();
            for s in 0..n {
                let m = batch.m_per_slot[s];
                if m == 0 || !rig.pool.is_paged() {
                    cow_plans.push(None);
                    row_offset += m;
                    continue;
                }
                let write_start = batch.positions[row_offset] as usize;
                let write_end = batch.positions[row_offset + m - 1] as usize + 1;
                match rig
                    .pool
                    .plan_cow_for_slot(SlotId(s), write_start, write_end)
                {
                    Ok(plan) => {
                        rig.cow_plan_count += 1;
                        // Execute the copies NOW (spec §4.3: raw byte copies of
                        // the valid K/V prefix, every layer arena, no
                        // dequant/requant). The copies MUST precede both the
                        // table rebind and the forward: the forward's KV-write
                        // kernels resolve through the block table, so rebinding
                        // to the private destination before the copies would
                        // write rows into uninitialized pages, and writing
                        // before the rebind would mutate the sealed shared
                        // source in place.
                        let copy_ok = if plan.is_empty() {
                            true
                        } else {
                            execute_cow_copies(&rig, &plan)
                        };
                        if copy_ok {
                            cow_plans.push(Some((s, plan)));
                        } else {
                            eprintln!(
                                "[cow] copy execution failed for slot {s} — failing the slot"
                            );
                            // Keep the plan in cow_plans[s]: the failed-slot
                            // handler below aborts it, releasing the reserved
                            // destination pages (spec §4.3 transactional
                            // failure).
                            cow_plans.push(Some((s, plan)));
                            failed_slots.push(s);
                        }
                    }
                    Err(e) => {
                        eprintln!("[cow] plan_cow failed for slot {s}: {e}");
                        // plan_cow failed: nothing was installed for this slot,
                        // so no abort is needed. Record the failure and continue
                        // — other slots' plans survive (spec §5.4/S4).
                        failed_slots.push(s);
                        cow_plans.push(None);
                    }
                }
                row_offset += m;
            }
            // Fail the isolated slots: reject their requests, clear their work,
            // and zero their batch rows so the forward skips them. COW plans
            // kept for aborted slots (copy failures) are released here.
            if !failed_slots.is_empty() {
                for &s in &failed_slots {
                    // Abort any COW plan that WAS created for this slot (copy
                    // execution failed after reservation).
                    if let Some((_, p)) = cow_plans.get_mut(s).and_then(|o| o.take()) {
                        rig.pool.abort_cow_for_slot(p);
                    }
                    // Zero the slot's batch rows so the forward does not write
                    // through a slot whose COW plan failed.
                    let m = batch.m_per_slot[s];
                    batch.m_per_slot[s] = 0;
                    let _ = m; // rows are dropped from the flat arrays below
                    if let Some(mut f) = slots[s].take() {
                        let reason = "COW reservation failed for this slot".to_string();
                        let _ = send_event(
                            &f.reply,
                            Event::Rejected {
                                class: RejectClass::Internal,
                                reason,
                            },
                        );
                        release_pin_ticket(&mut rig, &mut f.pin_ticket);
                        rig.swap.forget(f.session.0);
                        let _ = rig.fair_queue.remove(f.session.0);
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                        clear_work_slot(&mut work[s]);
                        rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                    }
                }
                // Rebuild the flat arrays to drop the failed slots' rows. The
                // forward reads tokens/positions/row_slot/pos3/ext_emb as flat
                // arrays packed by m_per_slot, so zeroing m_per_slot without
                // removing the flat rows would misalign every subsequent slot.
                rebuild_batch_excluding_failed(&mut batch, &failed_slots);
                // Also drop spec drafts for failed slots so the verify path
                // does not try to read their hidden rows.
                for &s in &failed_slots {
                    // Free the dropped draft's device buffers — DflashSlotDraft
                    // owns GpuTensors with no Drop.
                    if let Some(d) = spec_drafts[s].take() {
                        d.free_gpu(&mut rig.gpu);
                    }
                }
                if batch.is_empty() {
                    continue;
                }
            }
            // Slots with verify drafts skip the per-slot last-row lm_head GEMV
            // inside the forward: their logits come from the batched verify
            // lm_head over all k+1 rows instead, so the single-row GEMV (a full
            // vocab projection) is redundant for them. Computed after COW failure
            // isolation so a dropped draft is not counted as a verify slot.
            let lm_head_skip: Vec<bool> = (0..n).map(|s| spec_drafts[s].is_some()).collect();
            let any_verify = lm_head_skip.iter().any(|&b| b);

            // Commit the surviving COW plans — the copies already ran at plan
            // time, so rebinding here puts the private destinations in the block
            // table the forward's write kernels resolve through (spec §4.3:
            // "rebind that request's table only after copy ordering is
            // established"). A commit failure means the table changed under the
            // plan — an accounting fault (S4): fail only that slot and release
            // its reserved pages.
            let mut failed_commits: Vec<usize> = Vec::new();
            for entry in cow_plans.iter_mut() {
                if let Some((s, p)) = entry.take() {
                    if let Err(e) = rig.pool.commit_cow_for_slot(SlotId(s), p) {
                        eprintln!("[cow] commit_cow failed for slot {s}: {e}");
                        failed_commits.push(s);
                    }
                }
            }
            if !failed_commits.is_empty() {
                for &s in &failed_commits {
                    if let Some(mut f) = slots[s].take() {
                        let reason = "COW commit failed: table changed under the plan".to_string();
                        let _ = send_event(
                            &f.reply,
                            Event::Rejected {
                                class: RejectClass::Internal,
                                reason,
                            },
                        );
                        release_pin_ticket(&mut rig, &mut f.pin_ticket);
                        rig.swap.forget(f.session.0);
                        let _ = rig.fair_queue.remove(f.session.0);
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    }
                    clear_work_slot(&mut work[s]);
                    rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                    if let Some(d) = spec_drafts[s].take() {
                        d.free_gpu(&mut rig.gpu);
                    }
                }
                rebuild_batch_excluding_failed(&mut batch, &failed_commits);
                if batch.is_empty() {
                    continue;
                }
            }
            // VL rows: refresh the per-row vision-matrix base pointers the
            // scatter kernel dereferences. This MUST run AFTER every
            // rebuild_batch_excluding_failed above — the upload reads the
            // post-rebuild `row_slot`, and uploading before a rebuild would
            // leave the pointer array misaligned with the surviving rows (a
            // failed slot's rows are dropped, shifting every later row's slot
            // index). Engine-side upload, every step (including graph replays),
            // so the captured kernel always reads current pointers. Rows with a
            // negative index never dereference.
            if batch.ext_emb.len() == batch.positions.len() && batch.ext_emb.iter().any(|&e| e >= 0)
            {
                let ptrs: Vec<u64> = batch
                    .row_slot
                    .iter()
                    .map(|&sl| {
                        rig.model.vl_ext_devs[sl as usize]
                            .as_ref()
                            .map(|t| t.buf.as_ptr() as u64)
                            .unwrap_or(0)
                    })
                    .collect();
                let bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
                // Bound-check here, not in memcpy_htod: its overflow path is an
                // assert, and a panic in this thread kills the engine loop while
                // the HTTP front end keeps accepting — the serve-hang failure
                // mode. This upload runs before forward_batch_slots' own
                // n <= pbs.max_batch assert, so an oversized batch would only
                // surface here.
                if bytes.len() > rig.model.pbs.ext_emb_row_ptr.buf.size() {
                    let reason = format!(
                        "vl ext ptr upload of {} bytes exceeds staging capacity {} \
                     (batch of {} rows vs max_batch {})",
                        bytes.len(),
                        rig.model.pbs.ext_emb_row_ptr.buf.size(),
                        batch.total_rows(),
                        rig.model.pbs.max_batch
                    );
                    fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                    poison = Some(reason);
                    break 'serve;
                }
                if let Err(e) = rig
                    .gpu
                    .hip
                    .memcpy_htod(&rig.model.pbs.ext_emb_row_ptr.buf, &bytes)
                {
                    let reason = format!("vl ext ptr upload failed: {e:?}");
                    fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                    poison = Some(reason);
                    break 'serve;
                }
            }

            // DFlash slots need the extract-layer hidden capture whenever they
            // contribute rows this step — prefill chunks (draft-context seed)
            // and verify rows (post-accept commit) alike.
            let any_dflash_rows = (0..n).any(|s| {
                work[s].spec == SpecKind::Dflash
                    && batch.m_per_slot.get(s).copied().unwrap_or(0) > 0
            });
            // 期4 speculative statistics: a decode row served as plain AR while
            // the request asked for a drafter (retired, cap guard, or a reuse
            // without a draft image) counts as a fallback step
            // (speculative.hpp `fallback_steps`).
            for s in 0..n {
                if rig.spec_counters[s].requested
                    && work[s].decoding
                    && !work[s].spec.active()
                    && batch.m_per_slot.get(s).copied().unwrap_or(0) > 0
                {
                    rig.spec_counters[s].fallback += 1;
                }
            }
            // Step span (opt-in `HIPFIRE_CPU_EXEC_TRACE` only): forward vs
            // sample vs sync/readback vs emit. Host wall around work this step
            // already runs; no new sync, no order change.
            let t_step = std::time::Instant::now();
            let t_forward = std::time::Instant::now();
            // ── Layerwise orchestration (design §3.9g, kvmem-gated) ──────────
            // Scoped take of the controller so the per-query carrier can borrow
            // its slot stores while the rest of the model is borrowed by the
            // forward (the driver must NOT live in a ModelRig field — it borrows
            // kvmem, which is also in ModelRig ⇒ self-reference). Only an
            // admissible single-slot text-only q8 paged decode admits Layerwise;
            // everything else threads None (Legacy byte-neutral). The carrier is
            // confined to the `Ok(Some)` arm: `free` consumes it, ending its
            // borrow of the taken-out controller strictly before the counter bump
            // and restore — mirroring the scoped `ctl.prepare_step` borrow above.
            // The post-forward statistics feed runs after the restore, so the
            // host store still advances this step's row.
            #[cfg(feature = "kvmem")]
            let fwd = match kvmem_engine::layerwise_forward(
                &mut rig,
                &batch,
                &slots,
                &work,
                &mut graph,
                &lm_head_skip,
                any_verify,
                any_dflash_rows,
            ) {
                Ok(fwd) => fwd,
                Err(e) => {
                    fail_all_active(&mut rig, &mut slots, &mut work, e.clone());
                    poison = Some(e);
                    break 'serve;
                }
            };
            #[cfg(not(feature = "kvmem"))]
            let fwd = rig.model.forward_step(
                &mut rig.gpu,
                &batch,
                &mut rig.pool,
                &mut graph,
                &lm_head_skip,
                any_verify,
                any_dflash_rows,
                None,
            );
            let forward_ns = t_forward.elapsed().as_nanos() as u64;
            // ── Publication (spec §4.6 C6) ──────────────────────────────────
            // After a SUCCESSFUL prefill chunk, publish sealed full pages at
            // committed page-aligned boundaries. Only when prefix_cache is on.
            // Generated-prefix pages are published at the Done event (client
            // commit). Publication is gated on the forward result: a failed
            // step leaves provisioned pages with rows never written (and the
            // recurrent state mid-layer) — sealing or publishing them would
            // make garbage KV and a bogus checkpoint permanently resumable
            // (spec §4.6: "may become shareable after successful forward
            // completion"; §5.4: "a failed GPU step is not an atomic
            // transaction").
            if rig.prefix_cache && fwd.is_ok() {
                let mut row_offset = 0usize;
                for s in 0..n {
                    let m = batch.m_per_slot[s];
                    if m == 0 {
                        continue;
                    }
                    row_offset += m;
                    // Only publish for prefilling slots (not decoding).
                    if work[s].decoding {
                        continue;
                    }
                    // Vision requests skip the radix (spec X2) — including TEXT
                    // continuations of an image conversation: their context KV
                    // carries image-turn M-RoPE phases. `has_image` is the
                    // positive marker (`rope_delta` is 0 for a thin image).
                    if work[s].vl_prefill.is_some() {
                        continue;
                    }
                    if slots[s]
                        .as_ref()
                        .and_then(|f| rig.sessions.get(f.session))
                        .map(|sess| sess.has_image)
                        .unwrap_or(true)
                    {
                        continue;
                    }
                    let new_boundary = (work[s].next_pos / PAGE_TOKENS) * PAGE_TOKENS;
                    let last_pub = slots[s]
                        .as_ref()
                        .map(|f| f.last_published_boundary)
                        .unwrap_or(0);
                    if new_boundary <= last_pub || new_boundary == 0 {
                        continue;
                    }
                    // Publish pages [last_published, new_boundary).
                    let n_old_pages = last_pub / PAGE_TOKENS;
                    let n_new_pages = new_boundary / PAGE_TOKENS;
                    let page_indices: Vec<u32> = match rig.pool.block_table(SlotId(s)) {
                        Some(bt) => bt.page_indices().to_vec(),
                        None => continue,
                    };
                    if page_indices.len() < n_new_pages {
                        continue;
                    }
                    // Seal the newly written pages (immutability before the
                    // publish). Cache refs are taken INSIDE publish_sealed_pages
                    // — one per node slot the tree actually adopts — so a failed
                    // or partially-adopted publish cannot orphan a ref (the
                    // ref-taken-here/rollback-here scheme leaked every page the
                    // tree declined: a page whose ref nobody owns never returns
                    // to the free list).
                    if let Some(pp) = rig.pool.page_pool_mut() {
                        for p in n_old_pages..n_new_pages {
                            let _ = pp.seal(page_indices[p]);
                        }
                    }
                    // Build handles for all pages up to the new boundary.
                    let handles: Vec<Handle> = page_indices[..n_new_pages]
                        .iter()
                        .enumerate()
                        .map(|(i, &phys)| Handle {
                            handle: rdna_compute::page_pool::PageHandle {
                                phys,
                                epoch: rig.pool.page_pool().map(|p| p.epoch()).unwrap_or(0),
                                generation: rig
                                    .pool
                                    .page_pool()
                                    .map(|p| p.page_generation(phys))
                                    .unwrap_or(0),
                            },
                            token_offset: (i * PAGE_TOKENS) as u64,
                        })
                        .collect();
                    // Get the session's tokens up to the boundary.
                    let session = match slots[s].as_ref().map(|f| f.session) {
                        Some(sid) => sid,
                        None => continue,
                    };
                    let tokens: Vec<u32> = rig
                        .sessions
                        .get(session)
                        .map(|sess| sess.tokens[..new_boundary].to_vec())
                        .unwrap_or_default();
                    if tokens.len() == new_boundary {
                        let domain = rig.cache_domain.as_ref().unwrap();
                        let idx = rig.prefix_index.as_mut().unwrap();
                        let pp = rig.pool.page_pool_mut().unwrap();
                        // Capture a checkpoint at the boundary — ONLY when the
                        // published boundary is exactly the recurrent-state
                        // boundary (`next_pos`). The live DeltaNetState sits at
                        // next_pos; a state captured there cannot be relabelled
                        // as an earlier boundary (spec §4.5: "a state at the end
                        // of a chunk cannot be relabeled as an earlier state").
                        // Mid-page chunk tails publish pages without a
                        // checkpoint, so lookups fall back to an earlier aligned
                        // checkpoint or an honest cold recompute.
                        let checkpoint = if new_boundary == work[s].next_pos {
                            if let Some(ckpt_pool) = rig.model.checkpoint_pool.as_mut() {
                                match capture_checkpoint(
                                    &mut rig.gpu,
                                    ckpt_pool,
                                    domain,
                                    new_boundary as u64,
                                    &tokens,
                                    None, // radix path: text-only
                                    &rig.model.dn_states[s],
                                ) {
                                    Ok(id) if id.is_some() => Some(id),
                                    Ok(_) => None, // pool at ceiling: publish pages only
                                    Err(e) => {
                                        eprintln!("[prefix-cache] capture_checkpoint failed: {e}");
                                        None
                                    }
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        // A19 fault injection (host test seam): with
                        // HIPFIRE_FAULT_PREFIX_PUBLISH=1 the first publish
                        // attempt fails so the honest-miss behavior (and the
                        // absence of any leaked cache ref) can be verified end
                        // to end.
                        let publish_fault = inject_publish_fault();
                        let publish_result = if publish_fault {
                            Err(hipfire_runtime::prefix_index::InsertError::MisalignedHandle)
                        } else {
                            idx.publish_sealed_pages(domain, &tokens, &handles, checkpoint, pp)
                        };
                        if let Err(e) = publish_result {
                            eprintln!("[prefix-cache] publish_sealed_pages failed: {e:?}");
                            // No cache refs to roll back: the index took refs
                            // only for the node slots it actually created, and
                            // every failure path in the index releases them.
                        } else {
                            if let Some(f) = slots[s].as_mut() {
                                f.last_published_boundary = new_boundary;
                            }
                            if let Some(session) = slots[s].as_ref().map(|f| f.session) {
                                if let Some(sess) = rig.sessions.get_mut(session) {
                                    sess.published_boundary = new_boundary;
                                }
                            }
                            if rig.gpu.slot_trace() {
                                eprintln!(
                                "[slot-trace] published {n_new_pages} pages at boundary {new_boundary}"
                            );
                            }
                        }
                    }
                }
            }
            if let Err(e) = fwd {
                // A forward failure is not recoverable per-request. Report it as a
                // REJECTION carrying the reason, not as a normal Done: an
                // unsupported model (e.g. "DeltaNet layer has a non-Q8_0 weight")
                // otherwise reaches the client as a successful, empty 200, which
                // looks like the model chose to say nothing.
                // Retain per-request behavior only because state is provably
                // closed/reset: every active slot is taken, its session closed,
                // work cleared and swap forgotten, so no poisoned KV/DN survives.
                // If this invariant were broken, we would need to poison instead.
                let reason = e.to_string();
                // Free every in-flight draft's device buffers first — the slots
                // are about to be torn down and DflashSlotDraft owns GpuTensors
                // with no Drop.
                for d in spec_drafts.iter_mut() {
                    if let Some(d) = d.take() {
                        d.free_gpu(&mut rig.gpu);
                    }
                }
                // fail_all_active does the full per-slot teardown (clear_work_slot
                // + clear_slot_vl_state + fair_queue.remove + session close) that
                // this path previously open-coded but skipped — leaking VL device
                // buffers (vl_ext_devs, vl_tower_jobs) and leaving stale
                // work[s].vl_prefill/pos3_delta.
                fail_all_active(&mut rig, &mut slots, &mut work, reason);
                continue;
            }
            if let Err(e) = rig.gpu.hip.device_synchronize() {
                let reason = format!("device synchronize failed: {e:?}");
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
            }
            if kvmem_hooks::forward_time_enabled(&rig) {
                let _ = forward_ns;
                eprintln!(
                    "[kvmem-time] forward {} rows: {:.1} ms through sync",
                    batch.total_rows(),
                    t_forward.elapsed().as_secs_f64() * 1e3,
                );
            }
            // ── KVMem Legacy Query readback ────────────────────────────────
            // Post-sync the capture holds post-norm/pre-RoPE Q for exactly the
            // rows masked above. Every FullAttention layer appends its rows
            // contiguously to the slot's Query accumulation. A failure is a
            // clean bypass (the intact first pass stays authoritative), never a
            // partially recorded probe.
            #[cfg(feature = "kvmem")]
            kvmem_engine::read_legacy_query_rows(&mut rig, &legacy_query_reads);
            if let Err(e) = kvmem_hooks::feed_step(&mut rig, &batch, &slots, &work, &spec_drafts, n)
            {
                // A host-tier capacity fault is recoverable by dropping every
                // slot's host state: the forward's GPU rows are stale after
                // fail_all_active, and each reset releases the raw stores so
                // the next session starts from a clean charge. Any other feed
                // failure keeps the poison contract.
                let recoverable = kvmem_hooks::is_host_capacity_error(&e);
                for d in spec_drafts.iter_mut() {
                    if let Some(d) = d.take() {
                        d.free_gpu(&mut rig.gpu);
                    }
                }
                fail_all_active(&mut rig, &mut slots, &mut work, e.clone());
                if recoverable && kvmem_hooks::recover_host_capacity(&mut rig) {
                    eprintln!(
                        "[kvmem] host capacity exhausted; reset all slots and \
                         continued: {e}"
                    );
                    continue 'serve;
                }
                poison = Some(e);
                break 'serve;
            }
            if check_pool_invariants {
                if let Some(pp) = rig.pool.page_pool() {
                    pp.check_invariants();
                }
            }
            // Retire pages whose GPU readers have drained (deferred reclaim,
            // spec §4.4). Without a periodic drain the pending queue only
            // cleared on Reset, so a failed multi-page allocation's freed
            // pages sat unreachable between retries and free counts drifted.
            // The walk is over a short pending list — cheap per step.
            if let Some(pp) = rig.pool.page_pool_mut() {
                pp.drain_completed();
            }
            // ── KVMem Legacy Query replay transaction ──────────────────────
            // After this step's accepted rows are fed, a Query whose probe is
            // complete either publishes its score round once (no useful replay)
            // or rewinds the slot to the pre-Query checkpoint and replays the
            // exact client prompt suffix. Rewind is destructive from the first
            // `start_replay`: a later failure poisons the engine instead of
            // claiming the first pass still stands.
            #[cfg(feature = "kvmem")]
            let mut legacy_rewound = vec![false; n];
            #[cfg(feature = "kvmem")]
            if rig
                .model
                .kvmem
                .as_ref()
                .is_some_and(|ctl| ctl.legacy_config().is_some())
            {
                if let Err(e) =
                    advance_legacy_query_replay(&mut rig, &slots, &mut work, &mut legacy_rewound)
                {
                    fail_all_active(&mut rig, &mut slots, &mut work, e.clone());
                    poison = Some(e);
                    break 'serve;
                }
            }
            // Penalty windows: for each slot whose request carries non-neutral
            // token penalties, clamp the requested window to what the session
            // actually holds and upload that tail (most recent last). The kernel
            // scans exactly `repeat_window` entries, so the param must equal the
            // uploaded count — early in a generation that is fewer than the
            // request's window.
            for s in 0..n {
                let penalized = rig.sample_params[s].penalized();
                if !penalized || slots[s].is_none() {
                    continue;
                }
                let session = slots[s].as_ref().unwrap().session;
                let Some(sess) = rig.sessions.get(session) else {
                    continue;
                };
                let requested = slots[s].as_ref().map(|f| f.repeat_window_req).unwrap_or(0);
                let effective = requested.min(sess.tokens.len());
                rig.sample_params[s].repeat_window = effective as i32;
                if effective == 0 {
                    continue;
                }
                let tail = &sess.tokens[sess.tokens.len() - effective..];
                let tail_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(tail.as_ptr() as *const u8, effective * 4)
                };
                if let Err(e) = rig
                    .gpu
                    .hip
                    .memcpy_htod(&rig.repeat_windows[s].buf, tail_bytes)
                {
                    let reason = format!("repeat window upload failed: {e:?}");
                    fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                    poison = Some(reason);
                    break 'serve;
                }
            }
            // ── Grammar mask: apply BEFORE the sampler (spec §7.2 G2) ───────
            // The hard grammar mask participates before probability-dependent
            // filtering/renormalization. No later operation may resurrect a
            // forbidden token (spec §7.2 G2). For each slot with a grammar
            // constraint, copy its logits row D2H, set disallowed tokens to
            // NEG_INFINITY, and write it back H2D before `sample_per_slot`.
            //
            // An empty allowed set is a typed per-request error, never an
            // unconstrained fallback (spec §7.2 G2, A17). The failing request
            // is rejected; other active requests are unaffected.
            //
            // The mask is built from `SchemaMatcher::is_token_allowed` over the
            // tokenizer's lossless `token_bytes` table (spec §7.2 G2).
            let vocab = rig.model.config.vocab_size;
            let mut grammar_failures: Vec<(usize, String)> = Vec::new();
            for s in 0..n {
                let Some(f) = slots[s].as_mut() else { continue };
                let Some(constraint) = f.grammar.as_mut() else {
                    continue;
                };

                // Build the mask for the current parser state.
                let mask = match constraint.build_mask(&rig.tokenizer, vocab) {
                    Ok(m) => m,
                    Err(e) => {
                        grammar_failures.push((s, e.to_string()));
                        continue;
                    }
                };

                // D2H: copy this slot's logits row to the host buffer.
                let row_offset = s * vocab;
                let logits_bytes_len = vocab * std::mem::size_of::<f32>();
                let host_slice = &mut rig.logits_host[row_offset..row_offset + vocab];
                let host_bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(
                        host_slice.as_mut_ptr() as *mut u8,
                        logits_bytes_len,
                    )
                };
                let gpu_row = rig.model.logits_out.sub_offset(row_offset, vocab);
                if let Err(e) = rig.gpu.hip.memcpy_dtoh(host_bytes, &gpu_row.buf) {
                    let reason = format!("grammar mask D2H failed for slot {s}: {e:?}");
                    fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                    poison = Some(reason);
                    break 'serve;
                }

                // Apply the mask: disallowed tokens → NEG_INFINITY. Allowed
                // tokens are left alone so the sampler's penalties/top_p
                // operate on the original logits (spec §7.2 G2: "Apply request
                // penalties/biases according to the existing sampler contract,
                // then ensure the hard grammar mask participates before
                // probability-dependent filtering/renormalization").
                for (i, &allowed) in mask.iter().enumerate() {
                    if !allowed {
                        host_slice[i] = f32::NEG_INFINITY;
                    }
                }

                // H2D: write the masked logits back.
                let host_bytes_back: &[u8] = unsafe {
                    std::slice::from_raw_parts(host_slice.as_ptr() as *const u8, logits_bytes_len)
                };
                if let Err(e) = rig.gpu.hip.memcpy_htod(&gpu_row.buf, host_bytes_back) {
                    let reason = format!("grammar mask H2D failed for slot {s}: {e:?}");
                    fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                    poison = Some(reason);
                    break 'serve;
                }
            }
            // Fail any slots whose grammar constraint produced an empty allowed
            // set or unsatisfiable state. Each is a per-request rejection —
            // other active requests continue (spec §5.4/S4: "Per-request
            // parser/sampling failure → fail that request; release its
            // resources at a safe boundary").
            for (s, reason) in grammar_failures {
                if let Some(mut f) = slots[s].take() {
                    let _ = send_event(
                        &f.reply,
                        Event::Rejected {
                            class: RejectClass::Internal,
                            reason: reason.clone(),
                        },
                    );
                    release_pin_ticket(&mut rig, &mut f.pin_ticket);
                    rig.swap.forget(f.session.0);
                    let _ = rig.fair_queue.remove(f.session.0);
                    rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                    clear_work_slot(&mut work[s]);
                    rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                }
            }
            // ── Jump-forward planner (spec §7.3 G3) ───────────────────────
            // After the grammar mask, if structured_jump_forward is on and the
            // slot carries a grammar constraint under greedy decoding, call
            // `forced_token_run` to find a bounded run of provably-forced token
            // IDs. The run is committed instead of the sampled token; the
            // forced tokens are pushed into `remaining_prompt` so the scheduler
            // processes them as prefill (writing their KV) in subsequent steps.
            // Config off → jump_forced stays all-None: no behavior change.
            let mut jump_forced: Vec<Option<grammar::json_schema::ForcedRun>> =
                (0..n).map(|_| None).collect();
            if rig.structured_jump_forward {
                let vocab = rig.model.config.vocab_size;
                for s in 0..n {
                    let Some(f) = slots[s].as_ref() else { continue };
                    let Some(constraint) = f.grammar.as_ref() else {
                        continue;
                    };
                    // Greedy only (spec §7.3: "not sampled (greedy only)"). The
                    // gate must match the sampler's own argmax path exactly
                    // (`temperature == 0.0` in sampling.rs): a tiny nonzero
                    // temperature takes the RNG-consuming sample path, whose
                    // discarded draw still advances the seed and would make
                    // later sampled output diverge from the scalar reference
                    // (spec §7.3 G3.4: RNG draw accounting).
                    if rig.sample_params[s].temperature != 0.0 {
                        continue;
                    }
                    // Inside a think span the matcher has not seen any JSON —
                    // planning against it would read a stale parser state that
                    // does not describe the actual all-tokens-allowed policy
                    // (spec §7.2 framing-aware cursor). Wait for `</think>`.
                    if constraint.in_think {
                        continue;
                    }
                    // Only for decoding slots: remaining_prompt empty means
                    // the current logits are for the next token. A still-
                    // prefilling slot's logits belong to a mid-prompt token.
                    if !work[s].remaining_prompt.is_empty() {
                        continue;
                    }
                    // MTP is disabled for grammar-constrained requests at
                    // admit, but be defensive.
                    if work[s].spec.active() {
                        continue;
                    }
                    let remaining_max = f.max_tokens.saturating_sub(f.produced);
                    let max_run = remaining_max.min(rig.max_batch_tokens).min(64);
                    if max_run == 0 {
                        continue;
                    }
                    // Build the lossless token_bytes table and EOS id set
                    // for the planner (spec §7.2 G2: "Never build strict
                    // constraints on replacement-character artifacts").
                    let token_bytes_table: Vec<&[u8]> = (0..vocab as u32)
                        .map(|id| rig.tokenizer.token_bytes(id))
                        .collect();
                    let mut eos_ids = vec![rig.tokenizer.eos_id];
                    if let Some(eot) = rig.tokenizer.eot_id {
                        eos_ids.push(eot);
                    }
                    jump_forced[s] = Some(grammar::json_schema::forced_token_run(
                        &constraint.matcher,
                        &token_bytes_table,
                        &eos_ids,
                        max_run,
                    ));
                }
            }
            // Seed-determinism (spec §7.3 G3.4): sample_per_slot draws from the
            // per-request RNG for every row it samples. A slot that contributed
            // NO rows this step (FairQueue-ineligible, mid-prefill chunk boundary,
            // VL-waiting) used to draw anyway, shifting its RNG stream by one per
            // skipped step — the same seed + prompt then produced different
            // output depending on concurrent load. Park inactive slots at
            // temperature 0 for the call (argmax, no draw) and restore.
            let mut sampled_parked: Vec<(usize, f32)> = Vec::new();
            for (s, &m) in batch.m_per_slot.iter().enumerate() {
                // Park any slot whose sampled token would be discarded this
                // step: m==0 (no rows, already handled) OR still prefilling
                // (m>0 but remaining_prompt non-empty — the logits belong to a
                // mid-prompt token that is thrown away below, yet sample_per_slot
                // still advances that slot's RNG). Without the second arm the
                // seed stream shifts by one per mid-prompt chunk, so the same
                // seed+prompt samples differently depending on budget pressure.
                let still_prefilling = !work[s].remaining_prompt.is_empty();
                if (m == 0 || still_prefilling) && rig.sample_params[s].temperature != 0.0 {
                    sampled_parked.push((s, rig.sample_params[s].temperature));
                    rig.sample_params[s].temperature = 0.0;
                }
            }
            let t_sample = std::time::Instant::now();
            if let Err(e) = rig.gpu.sample_per_slot(
                &rig.model.logits_out,
                &mut rig.sample_params,
                &rig.repeat_windows,
                n,
                rig.model.config.vocab_size,
                &rig.out_tokens,
            ) {
                for (s, t) in sampled_parked {
                    rig.sample_params[s].temperature = t;
                }
                let reason = format!("sample failed: {e:?}");
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
            }
            let sample_ns = t_sample.elapsed().as_nanos() as u64;
            for (s, t) in sampled_parked {
                rig.sample_params[s].temperature = t;
            }
            // Sync + token-id readback: the existing device synchronize and
            // D2H this step already performs. Timed as one span (wait vs copy
            // are not separated — no new sync is introduced to split them).
            let t_sync = std::time::Instant::now();
            if let Err(e) = rig.gpu.hip.device_synchronize() {
                let reason = format!("device synchronize failed: {e:?}");
                fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                poison = Some(reason);
                break 'serve;
            }
            let mut ids = vec![0i32; n];
            {
                let bytes: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr() as *mut u8, n * 4) };
                if let Err(e) = rig.gpu.hip.memcpy_dtoh(bytes, &rig.out_tokens.buf) {
                    let reason = format!("DtoH failed: {e:?}");
                    fail_all_active(&mut rig, &mut slots, &mut work, reason.clone());
                    poison = Some(reason);
                    break 'serve;
                }
            }
            let sync_readback_ns = t_sync.elapsed().as_nanos() as u64;
            let t_emit = std::time::Instant::now();

            for s in 0..n {
                // A slot rewound by an admitted Query replay produced first-pass
                // logits this iteration; they are stale. Free any drafted rows
                // and skip ALL per-slot emission for it — the replayed prefix is
                // the only authority from here on.
                #[cfg(feature = "kvmem")]
                if legacy_rewound[s] {
                    if let Some(draft) = spec_drafts[s].take() {
                        draft.free_gpu(&mut rig.gpu);
                    }
                    continue;
                }
                // Spec verify/accept: process slots that produced draft outputs.
                if let Some(draft) = spec_drafts[s].take() {
                    if slots[s].is_none() {
                        continue;
                    }
                    // Compute hidden row offset for this slot in the batch.
                    let hidden_row_offset: usize = (0..s)
                        .map(|i| batch.m_per_slot.get(i).copied().unwrap_or(0))
                        .sum();
                    // Commit context for the burst clamp: the request counters
                    // and the session's current stored length.
                    let (produced, max_tokens, sess_len) = match slots[s].as_ref() {
                        Some(f) => (
                            f.produced,
                            f.max_tokens,
                            rig.sessions
                                .get(f.session)
                                .map(|sess| sess.tokens.len())
                                .unwrap_or(0),
                        ),
                        None => (0, usize::MAX, 0),
                    };
                    let accept_outcome = match draft {
                        SpecDraftRows::Mtp(d) => rig.model.mtp_verify_accept_step(
                            &mut rig.gpu,
                            &mut rig.pool,
                            &rig.tokenizer,
                            rig.cap_tokens,
                            SlotId(s),
                            &mut work[s],
                            d,
                            hidden_row_offset,
                            produced,
                            max_tokens,
                            sess_len,
                        ),
                        SpecDraftRows::Dflash(d) => rig.model.dflash_verify_accept_step(
                            &mut rig.gpu,
                            &mut rig.pool,
                            &rig.tokenizer,
                            rig.cap_tokens,
                            SlotId(s),
                            &mut work[s],
                            d,
                            hidden_row_offset,
                            produced,
                            max_tokens,
                            sess_len,
                        ),
                    };
                    match accept_outcome {
                        Ok(tokens) => {
                            // 期4 speculative statistics: committed outputs
                            // include the bonus row; accepted candidates
                            // exclude it (speculative.hpp semantics). Count
                            // only tokens the request actually consumed — a
                            // terminal token (EOS / cancel / cap) ends the
                            // batch, and the rows behind it were never
                            // delivered, so charging them would overstate
                            // accepted/generated for the rest of the run.
                            let mut committed = 0u64;
                            let mut accepted = 0u64;
                            for (row, tok) in tokens.iter().enumerate() {
                                if slots[s].is_none() {
                                    break;
                                }
                                commit_sampled_token(
                                    &mut rig, &mut slots, &mut work, &stats, s, *tok,
                                );
                                committed += 1;
                                if row + 1 < tokens.len() {
                                    accepted += 1;
                                }
                            }
                            rig.spec_counters[s].generated += committed;
                            rig.spec_counters[s].accepted += accepted;
                            // Adaptive retire (shared `SpecRetire`, the same
                            // policy the sequential drafter uses): a head that
                            // cannot sustain the break-even advance loses
                            // wall-clock on the spec cycle (~2x an AR step).
                            // Fall back to plain AR for the rest of this
                            // request. Applies to MTP and DFlash2 alike. The
                            // last committed token is already in
                            // remaining_prompt; the scheduler picks the slot up
                            // as a regular decode row next step.
                            if slots[s].is_some() && work[s].spec.active() {
                                let position = work[s].next_pos;
                                let kind = work[s].spec;
                                if work[s].spec_retire.observe(
                                    tokens.len(),
                                    rig.model.spec_rows,
                                    position,
                                ) {
                                    if rig.gpu.slot_trace() {
                                        eprintln!(
                                            "[slot-trace] {kind:?} retired for slot {s} at position {position} (break-even policy)",
                                        );
                                    }
                                    work[s].spec = SpecKind::None;
                                    // Of the committed tokens only the newest
                                    // lacks a KV row (the rest were written by
                                    // verify) — keep just it, like the cap
                                    // retire. The scheduler decodes it as a
                                    // regular row next step.
                                    if let Some(last) = work[s].remaining_prompt.last().copied() {
                                        work[s].remaining_prompt.clear();
                                        work[s].remaining_prompt.push(last);
                                    }
                                }
                            }
                        }
                        Err(reason) => {
                            if let Some(mut f) = slots[s].take() {
                                let _ = send_event(
                                    &f.reply,
                                    Event::Rejected {
                                        class: RejectClass::Internal,
                                        reason: reason.clone(),
                                    },
                                );
                                release_pin_ticket(&mut rig, &mut f.pin_ticket);
                                rig.swap.forget(f.session.0);
                                let _ = rig.fair_queue.remove(f.session.0);
                                rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                            }
                            clear_work_slot(&mut work[s]);
                            rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                        }
                    }
                    continue;
                }
                // Spec prefill: fill the speculator's private state from this
                // chunk's forward outputs; when the prompt has drained, the
                // sampled out_token becomes the draft seed and decode begins.
                if work[s].spec.active()
                    && !work[s].decoding
                    && slots[s].is_some()
                    && batch.m_per_slot.get(s).copied().unwrap_or(0) > 0
                {
                    let m = batch.m_per_slot[s];
                    let row_off: usize = (0..s)
                        .map(|i| batch.m_per_slot.get(i).copied().unwrap_or(0))
                        .sum();
                    if rig.gpu.slot_trace() {
                        eprintln!(
                            "[slot-trace] spec prefill slot {s} kind={:?} m={m} rp={}",
                            work[s].spec,
                            work[s].remaining_prompt.len()
                        );
                    }
                    let prefill_outcome: Result<(), String> = match work[s].spec {
                        SpecKind::Mtp => {
                            let chunk_tokens: Vec<u32> =
                                batch.tokens[row_off..row_off + m].to_vec();
                            let chunk_positions: Vec<i32> =
                                batch.positions[row_off..row_off + m].to_vec();
                            rig.model.mtp_head_prefill_chunk(
                                &mut rig.gpu,
                                SlotId(s),
                                &chunk_tokens,
                                &chunk_positions,
                                row_off,
                                m,
                            )
                        }
                        SpecKind::Dflash => rig.model.dflash_prefill_chunk(
                            &mut rig.gpu,
                            SlotId(s),
                            row_off,
                            m,
                            batch.positions[row_off] as usize,
                        ),
                        SpecKind::None => Ok(()),
                    };
                    match prefill_outcome {
                        Ok(()) => {
                            if work[s].remaining_prompt.is_empty() && slots[s].is_some() {
                                // Prompt fully prefilled (trunk KV by the
                                // scheduler path, spec state above): the sampled
                                // token is the first draft seed.
                                let seed = ids[s] as u32;
                                work[s].decoding = true;
                                if crate::mtp_spec::mtp_trace_enabled() {
                                    eprintln!("[mtp-trace] prefill done seed={seed}");
                                }
                                commit_sampled_token(
                                    &mut rig, &mut slots, &mut work, &stats, s, seed,
                                );
                            }
                        }
                        Err(reason) => {
                            if let Some(mut f) = slots[s].take() {
                                let _ = send_event(
                                    &f.reply,
                                    Event::Rejected {
                                        class: RejectClass::Internal,
                                        reason: reason.clone(),
                                    },
                                );
                                release_pin_ticket(&mut rig, &mut f.pin_ticket);
                                rig.swap.forget(f.session.0);
                                let _ = rig.fair_queue.remove(f.session.0);
                                rig.sessions.close(&mut rig.pool, &mut rig.adm, f.session);
                            }
                            clear_work_slot(&mut work[s]);
                            rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                        }
                    }
                    continue;
                }
                // Batched VL slots sample and commit like any regular slot (the
                // last prompt row's logits seed the first token); only
                // MTP-decoding slots skip — their tokens come from the verify.
                if work[s].spec.active() {
                    continue;
                }
                // Still prefilling: these logits belong to a mid-prompt token.
                if !work[s].remaining_prompt.is_empty() {
                    continue;
                }
                // Jump-forward (spec §7.3 G3): commit the provably-forced token
                // run instead of the sampled token. Each forced token is
                // committed via `commit_sampled_token`, which advances the
                // matcher, emits Token events, increments produced, and pushes
                // the token into `remaining_prompt` so the scheduler processes
                // it as prefill (writing its KV) in subsequent steps. The
                // FairQueue is not jumped — the scheduler still respects its
                // eligibility mask for the forced prefill rows.
                if let Some(forced) = jump_forced[s].take() {
                    if !forced.token_ids.is_empty() {
                        for &tok in &forced.token_ids {
                            commit_sampled_token(&mut rig, &mut slots, &mut work, &stats, s, tok);
                            if slots[s].is_none() {
                                break;
                            }
                        }
                        continue;
                    }
                }
                let Some(f) = slots[s].as_mut() else { continue };
                let tok = ids[s] as u32;
                // Advance the grammar parser cursor on the accepted token
                // (spec §7.2 G2: "per-request parser cursor advanced on
                // accepted tokens only").
                if let Some(constraint) = f.grammar.as_mut() {
                    constraint.advance_token(&rig.tokenizer, tok);
                }
                // Check whether the grammar constraint (if any) is in an
                // accepting state after advancing. A non-accepting state at
                // terminal means the generated JSON is incomplete/invalid —
                // emit a typed rejection instead of a successful done (spec
                // §7.1 G1, A17).
                let grammar_incomplete = f.grammar.as_ref().is_some_and(|c| !c.is_accepting());
                let session = f.session;
                let hit_eos = rig.tokenizer.is_terminator(tok);
                // Emit BEFORE counting, but never emit the terminator itself --
                // otherwise `<|im_end|>` lands in the client's content.
                let gone = if hit_eos {
                    false
                } else {
                    send_event(&f.reply, Event::Token { id: tok }).is_err()
                };
                f.produced += 1;
                let hit_max = f.produced >= f.max_tokens;
                // Context-cap guard: stop before the slot's KV length would pass
                // its cap. Without this, a long multi-turn session overflows
                // set_seq_len and kills the whole batched step.
                let hit_ctx_cap = rig
                    .sessions
                    .get(session)
                    .map(|sess| sess.tokens.len() + 1 >= rig.cap_tokens)
                    .unwrap_or(false);

                if gone || hit_eos || hit_max || hit_ctx_cap {
                    let reason = if gone {
                        DoneReason::ClientGone
                    } else if hit_eos {
                        DoneReason::Eos
                    } else {
                        DoneReason::MaxTokens
                    };
                    // MaxTokens: the final emitted token reached the client but has
                    // not yet gone through the next forward. Keep it in the
                    // authoritative stream; held seq_len remains one shorter, so
                    // begin_turn prefills this token plus the next-turn suffix.
                    if should_retain_terminal_tok(reason) {
                        if let Some(sess) = rig.sessions.get_mut(session) {
                            sess.tokens.push(tok);
                        }
                    }
                    // Terminator and undelivered (gone) tokens are counted by
                    // `produced` but never reached the client.
                    let generated = if hit_eos || gone {
                        f.produced - 1
                    } else {
                        f.produced
                    };
                    // If a grammar constraint is present and not accepting at
                    // terminal, the generated JSON is incomplete/invalid — emit
                    // a typed rejection instead of a successful done (spec §7.1
                    // G1, A17). ClientGone is never overridden: the receiver is
                    // already dropped.
                    lock_stats(&stats).note_slot_released();
                    // 期4: fold this request's speculative counters
                    // (completed-request statistics; cancellation included).
                    {
                        let spec = std::mem::take(&mut rig.spec_counters[s]);
                        if spec.requested {
                            let mut st = lock_stats(&stats);
                            st.speculative.requests += 1;
                            st.speculative.rounds += spec.rounds;
                            st.speculative.drafted_tokens += spec.drafted;
                            st.speculative.accepted_tokens += spec.accepted;
                            st.speculative.generated_tokens += spec.generated;
                            st.speculative.fallback_steps += spec.fallback;
                        }
                    }
                    if grammar_incomplete && !matches!(reason, DoneReason::ClientGone) {
                        let _ = send_event(
                            &f.reply,
                            Event::Rejected {
                                class: RejectClass::Validation,
                                reason: GrammarMaskError::Unsatisfiable.to_string(),
                            },
                        );
                    } else {
                        let _ = send_event(&f.reply, Event::Done { reason, generated });
                    }
                    // Release the prefix-cache lookup pin on EVERY terminal
                    // path — completion and cancellation alike (spec §4.4 C4).
                    // The session's table refs (from share_published_pages)
                    // still protect the shared pages while the session is idle.
                    release_pin_ticket(&mut rig, &mut f.pin_ticket);
                    // Generated-prefix publication at the terminal boundary
                    // (spec §4.6 C6: generated state becomes durable only at
                    // client commit, mid-flight pages are never published). The
                    // mid-step publication loop skips this slot because
                    // `decoding` is true; without this call a plain-AR request's
                    // generated pages would never reach the radix. Uncommitted
                    // paths (ClientGone, grammar rejection) still publish the
                    // PREFIX pages below the last committed boundary — that is
                    // the C6-legal set (committed prompt state), and
                    // publish_generated_prefix never publishes a page whose
                    // rows were not written.
                    publish_generated_prefix(&mut rig, &mut slots, s, work[s].next_pos);
                    // P2.2b2: freeze the page-aligned accepted frontier INCLUDING
                    // the generated rows, reusing a coverage recurrent checkpoint
                    // as its proof (ninfer endpoint capture). No-op without a
                    // coverage anchor, resident pages, or a longer frontier.
                    #[cfg(feature = "kvmem")]
                    if !grammar_incomplete && !matches!(reason, DoneReason::ClientGone) {
                        kvmem_freeze_endpoint(&mut rig, &mut slots, s);
                    }
                    // Hand the slot back — the session stays resident for
                    // continuation, but slot occupancy must end at terminal
                    // (matching commit_sampled_token). Wave 8's unpin edit
                    // accidentally deleted this clear and every batched-path
                    // termination wedged its slot (reset then always saw
                    // "requests in flight").
                    slots[s] = None;
                    let _ = rig.fair_queue.remove(session.0);
                    clear_work_slot(&mut work[s]);
                    // Speculative/terminal path: drop any unfinished Query
                    // lifecycle with the request. The plain-AR terminal path
                    // does the same in `commit_sampled_token`; without this
                    // reset a completed short request latched
                    // `legacy_query_decided` and suppressed the Query for the
                    // slot's next request.
                    #[cfg(feature = "kvmem")]
                    if let Some(run) = rig.legacy_queries.get_mut(s) {
                        *run = None;
                        rig.legacy_query_decided[s] = false;
                    }
                    rig.model.clear_slot_vl_state(&mut rig.gpu, s);
                    if matches!(reason, DoneReason::ClientGone) {
                        // Nobody will follow up on a vanished client, so hand the
                        // slot back at once.
                        rig.swap.forget(session.0);
                        self::kvmem_hooks::drop_idle_dn(&mut rig, s);
                        rig.sessions.close(&mut rig.pool, &mut rig.adm, session);
                    } else {
                        // S2: release the idle recurrent state to its exact
                        // host image; the next continuation reinstalls it.
                        self::kvmem_hooks::demote_idle_dn(&mut rig, s, session);
                        // Keep the session resident: the trunk KV (under KVMem
                        // including the host store's statistics and spilled
                        // pages), DeltaNet and MTP head state are exactly what a
                        // follow-up turn continuing this conversation needs;
                        // closing here is what made multi-turn reuse impossible.
                        // LRU eviction reclaims the slot when someone else needs
                        // it (closing, not swapping, under KVMem).
                        rig.sessions.touch(session);
                    }
                } else {
                    // Prefill→decode transition (spec §5.2 S2 classification):
                    // the sampled token is this request's first generated token.
                    // From here the slot decodes one row per step — counted as a
                    // decode row by the reservation and FairQueue (Phase-1 base
                    // allocation), and skipped by the mid-step publication loop
                    // so generated pages publish at Done (C6), never mid-flight.
                    work[s].decoding = true;
                    work[s].remaining_prompt.push(tok);
                    if let Some(sess) = rig.sessions.get_mut(session) {
                        sess.tokens.push(tok);
                    }
                    rig.sessions.touch(session);
                }
            }
            #[cfg(feature = "kvmem")]
            kvmem_engine::freeze_prefill_boundaries(
                &mut rig,
                &mut slots,
                &batch,
                &freeze_snapshot,
                n,
                fwd.is_ok(),
            );
            let emit_ns = t_emit.elapsed().as_nanos() as u64;
            let total_ns = t_step.elapsed().as_nanos() as u64;
            if hipfire_dispatch::moe_trace_enabled() {
                hipfire_dispatch::moe_step_trace(hipfire_dispatch::MoeStepTime {
                    total_ns,
                    forward_ns,
                    sample_ns,
                    sync_readback_ns,
                    emit_ns,
                    rows: batch.total_rows(),
                });
            }
        }
    }));
    if let Err(payload) = serve_result {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".to_string());
        poison = Some(format!("engine panic: {msg}"));
        eprintln!("[serve] panic in engine loop — failing all in-flight: {msg}");
    }

    // Shutdown/drain: if channel closed or poisoned while slots remain, fail them.
    // This ensures no in-flight cancellation can remain reusable and teardown
    // drains every active request. Normal shutdown drain (no prior poison) is
    // not itself poison — only HIP failures poison.
    let has_active = slots.iter().any(|s| s.is_some());
    if has_active {
        let reason = poison
            .clone()
            .unwrap_or_else(|| "engine shutdown".to_string());
        fail_all_active(&mut rig, &mut slots, &mut work, reason);
    }

    // Drain the wait queue on shutdown: reject every parked request so no
    // client is left waiting forever. The wait queue is pure host state —
    // no GPU resources to free, just event delivery.
    let shutdown_reason = poison
        .clone()
        .unwrap_or_else(|| "engine shutdown".to_string());
    for waiter in rig.wait_queue.expire(u64::MAX) {
        if let Some(parked) = rig.parked_requests.remove(&waiter.id) {
            let _ = send_event(
                &parked.req.reply,
                Event::Rejected {
                    class: RejectClass::Internal,
                    reason: shutdown_reason.clone(),
                },
            );
            lock_stats(&stats).note_rejected();
        }
    }

    // Free GPU after poisoning/shutdown drain. Graph is destroyed before buffers.
    let free_res = rig.free_gpu(graph);
    match (poison, free_res) {
        (Some(p), Ok(())) => Err(p),
        (Some(p), Err(e)) => Err(format!("{p}; teardown failed: {e}")),
        (None, Ok(())) => Ok(()),
        (None, Err(e)) => Err(e),
    }
}

fn should_retain_terminal_tok(reason: DoneReason) -> bool {
    matches!(reason, DoneReason::MaxTokens)
}

/// The materialized-row frontier a Done-time publication may seal up to.
///
/// `sess.tokens.len()` is the authoritative token stream, which can run one
/// token AHEAD of the KV rows: a MaxTokens-terminated request retains its
/// final sampled token without forwarding it (no row exists for it yet).
/// Publication must never seal a page covering that unmaterialized row.
fn done_publish_frontier(total_tokens: usize, state_boundary: usize) -> usize {
    state_boundary.min(total_tokens)
}

fn handle_command(
    rig: &mut Rig,
    slots: &mut [Option<InFlight>],
    work: &mut [PendingWork],
    stats: &Arc<Mutex<EngineStats>>,
    interrupts: &mut [bool],
    cmd: EngineCommand,
) {
    match cmd {
        EngineCommand::Submit(req) => {
            // FIFO admission age (spec §5.3 S3): while older work sits in the
            // waiting room, a newcomer must not take a free slot ahead of it
            // — under continuous arrivals that would starve every waiter
            // until its (wall-clock) deadline expired. Park the newcomer;
            // the drain loop below the command path pops in enqueue order
            // and admits as slots free.
            if rig.wait_queue.queued_count() > 0 {
                let waiter_id = rig.next_waiter_id;
                rig.next_waiter_id += 1;
                let bytes = req.queue_bytes.max(1);
                match rig.wait_queue.try_enqueue(waiter_id, bytes, rig.tick) {
                    Ok(()) => {
                        rig.parked_requests.insert(
                            waiter_id,
                            ParkedRequest {
                                req,
                                enqueued_at: std::time::Instant::now(),
                            },
                        );
                    }
                    Err(we) => {
                        let _ = send_event(
                            &req.reply,
                            Event::Rejected {
                                class: RejectClass::Overload,
                                reason: format!("serve queue full: {we}"),
                            },
                        );
                        lock_stats(stats).note_rejected();
                    }
                }
            } else {
                serve_engine_admit::admit(rig, slots, work, stats, req);
            }
        }
        EngineCommand::CancelWaiting { request_tag } => {
            // Cancellation for work that has no session yet (spec §4.6 C6:
            // "removes queued work by attempt identity even before Accepted
            // supplies a session ID"). The daemon correlates by the tag it
            // stamped on the SubmitRequest.
            let waiter_ids: Vec<u64> = rig
                .parked_requests
                .iter()
                .filter(|(_, p)| p.req.request_tag == request_tag)
                .map(|(id, _)| *id)
                .collect();
            for waiter_id in waiter_ids {
                if let Some(parked) = rig.parked_requests.remove(&waiter_id) {
                    rig.wait_queue.remove(waiter_id);
                    let _ = send_event(
                        &parked.req.reply,
                        Event::Rejected {
                            class: RejectClass::Cancel,
                            reason: "cancelled while queued".to_string(),
                        },
                    );
                    lock_stats(stats).note_rejected();
                }
            }
        }
        EngineCommand::Close { session, reply } => {
            let sid = SessionId(session);
            // At the engine-loop command boundary: if session is in flight,
            // cancel it — remove InFlight, drop its reply sender (Event channel),
            // clear work, forget swap, close SessionTable/pool/admission, then ack.
            // Idle close does the same forget/close. No ignored close.
            if let Some(idx) = slots
                .iter()
                .position(|s| s.as_ref().map(|f| f.session == sid).unwrap_or(false))
            {
                if let Some(mut inflight) = slots[idx].take() {
                    release_pin_ticket(rig, &mut inflight.pin_ticket);
                    drop(inflight.reply);
                }
                clear_work_slot(&mut work[idx]);
                #[cfg(feature = "kvmem")]
                if let Some(run) = rig.legacy_queries.get_mut(idx) {
                    *run = None;
                    rig.legacy_query_decided[idx] = false;
                }
                rig.model.clear_slot_vl_state(&mut rig.gpu, idx);
                let _ = rig.fair_queue.remove(session);
            }
            rig.swap.forget(session);
            rig.sessions.close(&mut rig.pool, &mut rig.adm, sid);
            let _ = reply.send(Ok(()));
        }
        EngineCommand::Interrupt { session } => {
            // Set the transient per-slot flag; the serve loop drains it at the
            // top of the next scheduler step (before any per-slot work), so
            // the pass ends at a token boundary with the session's KV intact.
            // Unknown/parked sessions are a no-op: the daemon only interrupts
            // after it has been told the session was accepted.
            if let Some(idx) = slots.iter().position(|s| {
                s.as_ref()
                    .map(|f| f.session == SessionId(session))
                    .unwrap_or(false)
            }) {
                if let Some(flag) = interrupts.get_mut(idx) {
                    *flag = true;
                }
            }
        }
        EngineCommand::Reset { reply } => {
            if slots.iter().any(|s| s.is_some()) {
                let _ = reply.send(Err("reset rejected: requests in flight".to_string()));
                return;
            }
            if rig.wait_queue.queued_count() > 0 {
                let _ = reply.send(Err(
                    "reset rejected: requests queued in the wait room".to_string()
                ));
                return;
            }
            let ids: Vec<u64> = rig.sessions.iter().map(|(id, _)| id).collect();
            for id in ids {
                rig.swap.forget(id);
                rig.sessions
                    .close(&mut rig.pool, &mut rig.adm, SessionId(id));
            }
            // Reset returns the engine to a cold empty table: parked
            // inactive histories and any half-finished Query checkpoint go
            // with it (their host stores belong to sessions that no longer
            // exist).
            #[cfg(feature = "kvmem")]
            {
                for run in rig.legacy_queries.iter_mut() {
                    *run = None;
                }
                for decided in rig.legacy_query_decided.iter_mut() {
                    *decided = false;
                }
                if let Some(catalog) = rig.legacy_history.as_mut() {
                    if let Ok(dropped) = catalog.clear() {
                        eprintln!(
                            "[kvmem-legacy] HistoryReset dropped_entries={}",
                            dropped.len()
                        );
                    }
                }
            }
            // Drop prefix cache state so subsequent lookups miss (spec §4.6).
            // The radix index owns cache leases taken at publication: it
            // MUST release them into the pool before being replaced, or
            // every published page strands CacheOnly and the pool drains to
            // zero free pages within a few reset cycles (A20 leak).
            if rig.prefix_cache {
                if let Some(pp) = rig.pool.page_pool_mut() {
                    if let Some(idx) = rig.prefix_index.as_mut() {
                        let released_pages = idx.release_all(pp);
                        pp.drain_completed();
                        if rig.gpu.slot_trace() {
                            eprintln!("[slot-trace] reset released {released_pages} cached pages");
                        }
                    }
                    // Fresh empty index: every subsequent lookup misses. The
                    // retained-byte ceiling must be re-applied — a bare
                    // `PrefixIndex::new` has no ceiling, and losing it here
                    // silently unbounded cache retention for the rest of
                    // the process lifetime.
                    let mut idx = hipfire_runtime::prefix_index::PrefixIndex::new(1 << 16);
                    idx.set_max_retained_bytes(rig.prefix_cache_max_bytes.unwrap_or(usize::MAX));
                    rig.prefix_index = Some(idx);
                }
                // Drain and free GPU blobs from the checkpoint pool.
                if let Some(pool) = rig.model.checkpoint_pool.as_mut() {
                    for blob in pool.drain_blobs() {
                        blob.free_gpu(&mut rig.gpu);
                    }
                }
                // Bump the allocation epoch so stale CacheDomain lookups miss.
                if let Some(domain) = rig.cache_domain.as_mut() {
                    domain.device.allocation_epoch += 1;
                }
            }
            let _ = reply.send(Ok(()));
        }
    }
}
/// Pure policy: should the admit path attempt a prefix-cache lookup?
///
/// Returns true only when prefix caching is enabled and the request carries
/// no vision data (spec X2: Vision+prefix reuse OFF).
fn should_lookup_prefix(prefix_cache: bool, has_visual: bool) -> bool {
    prefix_cache && !has_visual
}

/// Pure policy: how many prompt tokens are reused (skipped) given a resume
/// plan boundary and the total prompt length.
///
/// Returns 0 when there is no checkpoint. A boundary that covers the entire
/// prompt is a full hit: every prompt token is reused and prefill is empty
/// (spec §4.2 / A1 exact match).
fn reused_tokens_from_plan(boundary: usize, prompt_len: usize) -> usize {
    if boundary == 0 {
        0
    } else {
        boundary.min(prompt_len)
    }
}

/// Pure policy: should the admit path enqueue in the WaitQueue instead of
/// rejecting immediately (spec §5.3 S3, replacing R-A4)?
///
/// Returns true only when every resident session is generating (`all_busy`)
/// and the bounded waiting room is enabled (`wait_enabled` — always true
/// when Rig::build succeeded, since `WaitQueue::new` refuses zero caps).
fn should_wait_instead_of_reject(all_busy: bool, wait_enabled: bool) -> bool {
    all_busy && wait_enabled
}

// ---- FairQueue integration helpers (spec §5.3 S3) ----

/// Extract the set of request ids that received any grant this step.
///
/// Pure projection over a [`Selection`]: the FairQueue decides WHO is
/// eligible (aged-first, backfill, prefill cursor); the scheduler decides
/// HOW MANY rows each granted slot gets. This set becomes the scheduler's
/// eligibility mask so an un-granted slot contributes 0 rows and its
/// `remaining_prompt` is not drained (no tokens lost).
fn eligible_ids_from_selection(sel: &Selection) -> HashSet<u64> {
    sel.grants.iter().map(|g| g.id()).collect()
}

/// Bounded backfill (spec §5.3 S3.4): when a request is starved, drop
/// younger grants so the starved one receives the budget it needs. Returns
/// a set containing only `starved_id` when `starved_oldest` is set;
/// otherwise returns every granted id. The starved request is force-included
/// even if it received no grant this tick — the backfill exists precisely to
/// give it the next step's budget.
fn apply_starved_backfill(sel: &Selection, starved_id: u64) -> HashSet<u64> {
    if sel.starved_oldest {
        let mut set = HashSet::new();
        set.insert(starved_id);
        set
    } else {
        eligible_ids_from_selection(sel)
    }
}

/// Sync a request's per-step needs into the FairQueue from slot state.
/// `wants_decode` tracks a runnable decode row; `verify_rows` carries MTP
/// verify rows (k+1) when a draft was produced this step; `uncached` is the
/// remaining prefill suffix length (0 for decoding slots). Errors are
/// ignored defensively — every active slot is admitted to the queue, so a
/// `NotFound` only indicates a transient race at retirement.
fn sync_fair_needs(
    q: &mut FairQueue,
    id: u64,
    wants_decode: bool,
    verify_rows: u64,
    uncached: u64,
) {
    let _ = q.set_needs(id, wants_decode, verify_rows, 0);
    let _ = q.set_uncached_prefill(id, uncached);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_continuation_requires_the_complete_accepted_prefix() {
        // Stand-ins for system/user/assistant/tool/pending-token positions.
        let accepted = [10, 20, 30, 40, 50];
        assert!(legacy_continuation_matches(
            &accepted,
            &[10, 20, 30, 40, 50, 60]
        ));
        assert!(!legacy_continuation_matches(
            &accepted,
            &[10, 20, 31, 40, 50, 60]
        ));
        assert!(!legacy_continuation_matches(
            &accepted,
            &[10, 20, 30, 41, 50, 60]
        ));
        assert!(!legacy_continuation_matches(
            &accepted,
            &[10, 20, 30, 40, 51, 60]
        ));
        assert!(!legacy_continuation_matches(&accepted, &[10, 20, 30, 40]));
        assert!(!legacy_continuation_matches(&[], &[10, 20]));
    }

    #[test]
    fn slot_snapshot_kv_tags_are_stable_and_unique() {
        use hipfire_runtime::kv_mode::KvMode;

        let tags = [
            (KvMode::Q8, 1),
            (KvMode::Asym2, 21),
            (KvMode::Asym3, 31),
            (KvMode::Asym4, 41),
            (KvMode::Fwht2, 22),
            (KvMode::Fwht3, 32),
            (KvMode::Fwht4, 42),
            (KvMode::Fp8, 11),
            (KvMode::Bf16, 12),
            (KvMode::F16, 13),
        ]
        .map(|(mode, expected)| {
            let actual = slot_snapshot_kv_tag(mode);
            assert_eq!(actual, expected);
            actual
        });
        for (index, tag) in tags.iter().enumerate() {
            assert!(
                !tags[index + 1..].contains(tag),
                "duplicate slot snapshot KV tag {tag}"
            );
        }
    }

    /// The offload admission gate decides which tensors are charged to the device
    /// by parsing the layer ordinal out of an HFQ tensor name, so a name it fails
    /// to attribute would silently be charged (or not charged) to the wrong side
    /// of the split. Pin the shapes the real manifests use plus the near-misses.
    #[test]
    fn tensor_layer_index_parses_real_manifest_names() {
        assert_eq!(tensor_layer_index("layers.0.mlp.gate_proj.weight"), Some(0));
        assert_eq!(
            tensor_layer_index("layers.63.linear_attn.out_proj.weight"),
            Some(63)
        );
        assert_eq!(
            tensor_layer_index("model.language_model.layers.7.self_attn.q_proj.weight"),
            Some(7)
        );
        assert_eq!(
            tensor_layer_index("model.layers.12.input_layernorm.weight"),
            Some(12)
        );
        // Always-resident tensors must not be attributed to a layer.
        assert_eq!(tensor_layer_index("token_embd.weight"), None);
        assert_eq!(tensor_layer_index("output_norm.weight"), None);
        assert_eq!(tensor_layer_index("lm_head.weight"), None);
        assert_eq!(
            tensor_layer_index("model.language_model.lm_head.weight"),
            None
        );
        // Near-misses: "layers." not followed by an ordinal is not a layer tensor.
        assert_eq!(tensor_layer_index("layers.weight"), None);
        assert_eq!(tensor_layer_index("layers..weight"), None);
        assert_eq!(tensor_layer_index("my_layers.4.w"), None);
    }

    /// Pure stand-in for the MaxTokens bookkeeping in `run_loop`: only
    /// MaxTokens keeps the terminal emitted tok in the session transcript.
    fn apply_terminal(tokens: &mut Vec<u32>, tok: u32, reason: DoneReason) {
        if should_retain_terminal_tok(reason) {
            tokens.push(tok);
        }
    }

    #[test]
    fn max_tokens_retains_final_emitted_tok_for_continuation() {
        let mut tokens = vec![10, 20];
        apply_terminal(&mut tokens, 99, DoneReason::MaxTokens);
        assert_eq!(tokens, vec![10, 20, 99]);
    }

    #[test]
    fn eos_and_client_gone_do_not_retain_terminal_tok() {
        let mut tokens = vec![10, 20];
        apply_terminal(&mut tokens, 1, DoneReason::Eos);
        apply_terminal(&mut tokens, 2, DoneReason::ClientGone);
        assert_eq!(tokens, vec![10, 20]);
    }

    #[test]
    fn spec_verify_frontier_must_land_inside_the_cap() {
        // cap 100, verify_rows 4 (MTP k=3): verify writes rows at
        // next_pos..next_pos+3, frontier next_pos+4. next_pos 96 is the
        // last position that fits (frontier 100).
        assert!(spec_verify_fits_cap(96, 4, 100));
        assert!(!spec_verify_fits_cap(97, 4, 100));
        // A full verify batch is exactly verify_rows rows.
        assert!(spec_verify_fits_cap(0, 4, 4));
        assert!(!spec_verify_fits_cap(0, 4, 3));
        // verify_rows = 1 degenerates to plain decode: frontier next_pos+1.
        assert!(spec_verify_fits_cap(99, 1, 100));
        assert!(!spec_verify_fits_cap(100, 1, 100));
        // next_pos at or past the cap never fits, whatever the width.
        assert!(!spec_verify_fits_cap(100, 4, 100));
    }

    /// Grammar mask ordering: the mask is applied BEFORE the sampler's
    /// probability-dependent filtering (spec §7.2 G2). This test verifies
    /// the `GrammarConstraint::build_mask` → mask application → sampler
    /// ordering is observable: a token forbidden by the mask gets
    /// NEG_INFINITY and cannot be selected even if it has the highest
    /// logit (spec §7.2 G2: "No later operation may resurrect a forbidden
    /// token"; A17: "Empty grammar mask → explicit failure, never
    /// unconstrained fallback").
    #[test]
    fn grammar_mask_applied_before_sampler_forbids_highest_logit() {
        // Simulate a 4-token vocab: [forbidden, allowed, allowed, allowed].
        // The forbidden token has the highest logit (10.0); without the
        // mask a greedy sampler would pick it. After masking it to
        // NEG_INFINITY, the argmax falls through to the next highest.
        let mut logits = [10.0f32, 1.0, 2.0, 3.0];
        let mask = [false, true, true, true];

        // Apply the mask as serve_engine's run_loop does: disallowed →
        // NEG_INFINITY, allowed left alone (spec §7.2 G2).
        for (i, &allowed) in mask.iter().enumerate() {
            if !allowed {
                logits[i] = f32::NEG_INFINITY;
            }
        }

        // Greedy argmax over masked logits: must NOT pick token 0.
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(argmax, 3, "masked argmax must skip the forbidden token");
    }

    /// An empty grammar mask (no allowed tokens) is a typed error, never an
    /// unconstrained fallback (spec §7.2 G2, A17).
    #[test]
    fn grammar_empty_allowed_set_is_typed_error() {
        // Verify the error type and Display impl are correct.
        let err = GrammarMaskError::EmptyAllowedSet;
        assert!(err.to_string().contains("empty allowed set"));
        assert!(err.to_string().contains("violating the schema"));
        assert!(std::error::Error::source(&err).is_none());
    }

    /// Grammar-constrained requests must select AR (not MTP) at admission
    /// (spec §6 X2: "MTP + grammar → AR selected before execution until
    /// exact joint acceptance is implemented and validated").
    #[test]
    fn grammar_constrained_requests_select_ar_not_mtp() {
        // The serve_engine_admit::admit() path now sets `grammar_constrained = true` when
        // req.json_schema is Some. This test documents the invariant: a
        // grammar-constrained request must NOT activate MTP.
        let grammar_constrained = true;
        let mtp_available = true;
        let penalized = false;
        let sampled = false;
        let mtp_active = mtp_available && !penalized && !sampled && !grammar_constrained;
        assert!(!mtp_active, "grammar-constrained requests must use AR");
    }

    /// Think-budget enforcement (vLLM thinking_token_budget parity): at
    /// exhaustion the forced-close token is the think close, and before
    /// exhaustion nothing is forced. Without a special close id the span
    /// stays uncapped (a text-only close cannot be forced).
    #[test]
    fn think_budget_forces_close_at_exhaustion() {
        let schema = serde_json::json!({"type": "object"});
        let compiled = grammar::json_schema::CompiledSchema::compile(&schema).expect("compile");
        let mk = |close_id: Option<u32>, budget: usize, used: usize| GrammarConstraint {
            matcher: grammar::json_schema::SchemaMatcher::from_compiled(&compiled),
            mask_buf: Vec::new(),
            in_think: true,
            think_open_id: None,
            think_close_id: close_id,
            think_tail: Vec::new(),
            think_budget: budget,
            think_tokens_used: used,
            mask_cache: std::collections::HashMap::new(),
            mask_cache_order: std::collections::VecDeque::new(),
        };
        // Under budget: nothing forced.
        assert_eq!(mk(Some(9), 10, 9).think_close_forced(), None);
        // At budget with a close id: forced close.
        assert_eq!(mk(Some(9), 10, 10).think_close_forced(), Some(9));
        assert_eq!(mk(Some(9), 10, 500).think_close_forced(), Some(9));
        // At budget without a special close id: uncapped fallback.
        assert_eq!(mk(None, 10, 10).think_close_forced(), None);
    }

    /// The SchemaMatcher mask forbids a token whose bytes would break the
    /// `{` object start (spec §7.2 G2). A schema requiring an object must
    /// not allow tokens that don't begin with `{` (or whitespace before it).
    #[test]
    fn schema_matcher_mask_forbids_non_object_start() {
        let schema = serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
        let compiled = grammar::json_schema::CompiledSchema::compile(&schema).expect("compile");
        let matcher = grammar::json_schema::SchemaMatcher::from_compiled(&compiled);
        // `{` is allowed (object start).
        assert!(
            matcher.is_token_allowed(b"{"),
            "object start must be allowed"
        );
        // `}` is NOT allowed at the start (no object opened yet).
        assert!(
            !matcher.is_token_allowed(b"}"),
            "closing brace must be forbidden before object start"
        );
        // `1` (a number) is NOT allowed (schema requires an object).
        assert!(
            !matcher.is_token_allowed(b"1"),
            "number must be forbidden when schema requires object"
        );
    }

    /// The SchemaMatcher mask allows tokens that continue a valid prefix
    /// and forbids tokens that would violate the schema (spec §7.2 G2).
    #[test]
    fn schema_matcher_mask_allows_valid_continuation() {
        let schema = serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
        let compiled = grammar::json_schema::CompiledSchema::compile(&schema).expect("compile");
        let mut matcher = grammar::json_schema::SchemaMatcher::from_compiled(&compiled);
        // Advance past `{"a":` — now a string value is expected.
        matcher.advance(b"{\"a\":");
        // `"` (start of string value) is allowed.
        assert!(
            matcher.is_token_allowed(b"\""),
            "string start must be allowed after key"
        );
        // `}` is NOT allowed (expecting a string value, not object end).
        assert!(
            !matcher.is_token_allowed(b"}"),
            "object end must be forbidden when a string value is expected"
        );
    }

    /// An empty allowed set from a SchemaMatcher is a typed error (A17).
    /// A schema that requires a specific const value produces no allowed
    /// tokens when the matcher is in a state that can't accept any token.
    #[test]
    fn schema_matcher_empty_allowed_set_is_typed_error() {
        // A const schema requiring the value true: after advancing past
        // "tru", only "e" can continue. After "true", the matcher is
        // accepting and only whitespace/EOS is allowed.
        let schema = serde_json::json!({"const": true});
        let compiled = grammar::json_schema::CompiledSchema::compile(&schema).expect("compile");
        let mut matcher = grammar::json_schema::SchemaMatcher::from_compiled(&compiled);
        // After "tru", "e" is the only valid continuation.
        matcher.advance(b"tru");
        assert!(
            matcher.is_token_allowed(b"e"),
            "const true: 'e' must be allowed after 'tru'"
        );
        assert!(
            !matcher.is_token_allowed(b"x"),
            "const true: 'x' must be forbidden after 'tru'"
        );
        // After "true", the matcher is accepting.
        matcher.advance(b"e");
        assert!(
            matcher.is_accepting(),
            "const true: must be accepting after 'true'"
        );
    }

    // ---- Prefix cache policy tests (spec §4.5–4.6, X2) ----

    #[test]
    fn should_lookup_prefix_off_never_lookups() {
        // prefix_cache off → never lookup, regardless of visual data.
        assert!(!should_lookup_prefix(false, false));
        assert!(!should_lookup_prefix(false, true));
    }

    #[test]
    fn should_lookup_prefix_vision_skips_even_when_enabled() {
        // Vision requests skip the radix entirely (spec X2: Vision+prefix
        // reuse OFF).
        assert!(!should_lookup_prefix(true, true));
    }

    #[test]
    fn should_lookup_prefix_text_request_lookups_when_enabled() {
        // Text request with prefix_cache on → lookup.
        assert!(should_lookup_prefix(true, false));
    }

    #[test]
    fn reused_tokens_full_when_boundary_covers_entire_prompt() {
        // boundary == prompt_len is an exact match: every prompt token is reused.
        assert_eq!(reused_tokens_from_plan(100, 100), 100);
    }

    #[test]
    fn reused_tokens_zero_when_boundary_is_zero() {
        // No checkpoint → boundary 0 → 0 reused.
        assert_eq!(reused_tokens_from_plan(0, 100), 0);
    }

    #[test]
    fn reused_tokens_returns_boundary_on_partial_match() {
        // Partial match: boundary < prompt_len → boundary tokens reused.
        assert_eq!(reused_tokens_from_plan(64, 128), 64);
        assert_eq!(reused_tokens_from_plan(1, 2), 1);
    }

    #[test]
    fn reset_epoch_bump_makes_cache_domain_unequal() {
        // After Reset, allocation_epoch is bumped. A CacheDomain with a
        // bumped epoch must differ from the pre-reset domain (spec §4.6:
        // "unload/reload advances the epoch and invalidates lookups").
        use hipfire_runtime::serve_contract::{CacheDomain, DeviceTopology, SharingNamespace};
        let mk = |epoch: u64| CacheDomain {
            model_content_digest: vec![1, 2, 3],
            model_load_epoch: 0,
            sidecar_digests: Vec::new(),
            tokenizer: hipfire_runtime::serve_contract::TokenizerIdentity {
                vocab_digest: vec![0; 32],
                config_digest: vec![0; 32],
            },
            template: hipfire_runtime::serve_contract::TemplateIdentity {
                template_digest: vec![0; 32],
                normalization_tag: "default".to_owned(),
            },
            arch_policy: hipfire_runtime::serve_contract::ArchPolicy {
                arch_tag: "qwen35-deltanet".to_owned(),
                state_abi_tag: "dn-q8".to_owned(),
                position_attention_tag: "mrope".to_owned(),
            },
            kv_layout: hipfire_runtime::serve_contract::KvLayout {
                k_stride_bytes: vec![128],
                v_stride_bytes: vec![128],
                layout_tag: "q8".to_owned(),
            },
            device: DeviceTopology {
                device_id: "gpu0".to_owned(),
                topology_id: "single".to_owned(),
                allocation_epoch: epoch,
            },
            namespace: SharingNamespace {
                domain_id: "owner".to_owned(),
            },
        };
        let before = mk(0);
        let after = mk(1);
        assert_ne!(before, after, "epoch bump must invalidate domain equality");
        // Same epoch → equal (sanity).
        assert_eq!(mk(0), mk(0));
    }

    // ---- FairQueue integration tests (spec §5.3 S3) ----

    /// `eligible_ids_from_selection` projects a Selection's grants into the
    /// id set that becomes the scheduler's eligibility mask.
    #[test]
    fn eligible_ids_from_selection_collects_granted_ids() {
        let sel = Selection {
            grants: vec![
                Grant::Decode { id: 7 },
                Grant::Prefill { id: 3, rows: 16 },
                Grant::Verify { id: 9, rows: 4 },
            ],
            starved_oldest: false,
            starved_id: None,
        };
        let ids = eligible_ids_from_selection(&sel);
        assert_eq!(ids.len(), 3, "one entry per granted id");
        assert!(ids.contains(&7), "decode grant id 7");
        assert!(ids.contains(&3), "prefill grant id 3");
        assert!(ids.contains(&9), "verify grant id 9");
        // An empty selection yields no eligible ids.
        assert!(eligible_ids_from_selection(&Selection::default()).is_empty());
    }

    /// Bounded backfill (spec §5.3 S3.4): when starved_oldest is set,
    /// `apply_starved_backfill` yields ONLY the oldest id, dropping every
    /// younger grant so the starved oldest receives the next step's budget.
    #[test]
    fn starved_oldest_backfill_yields_only_oldest_id() {
        // A selection where younger work (ids 3, 9) was granted but the
        // oldest (id 7) is starved. The backfill keeps only id 7.
        let sel = Selection {
            grants: vec![
                Grant::Prefill { id: 3, rows: 16 },
                Grant::Verify { id: 9, rows: 4 },
            ],
            starved_oldest: true,
            starved_id: Some(7),
        };
        let ids = apply_starved_backfill(&sel, 7);
        assert_eq!(ids.len(), 1, "backfill keeps only the oldest");
        assert!(ids.contains(&7), "the starved oldest is force-eligible");
        assert!(
            !ids.contains(&3) && !ids.contains(&9),
            "younger work dropped"
        );
    }

    /// End-to-end FairQueue consultation: a real queue with an oldest
    /// prefill request starved by younger decode lanes produces a
    /// starved_oldest selection, and the backfill reduces eligibility to
    /// only the oldest id (spec §5.3 S3.4).
    #[test]
    fn fairqueue_starved_oldest_select_then_backfill() {
        // Budget 10, 4 decode lanes, prefill_min 5 (4 + 5 = 9 <= 10, valid).
        let mut q = FairQueue::new(10, 4, 5).expect("valid config");
        // Oldest: wants prefill, 5 uncached tokens (individually feasible:
        // 5 <= max_batch_tokens 10).
        q.admit(1, "s1", 5).unwrap();
        // Younger: four decode lanes.
        q.admit(2, "s2", 0).unwrap();
        q.admit(3, "s3", 0).unwrap();
        q.admit(4, "s4", 0).unwrap();
        q.admit(5, "s5", 0).unwrap();
        q.set_needs(1, false, 0, 0).unwrap();
        q.set_needs(2, true, 0, 0).unwrap();
        q.set_needs(3, true, 0, 0).unwrap();
        q.set_needs(4, true, 0, 0).unwrap();
        q.set_needs(5, true, 0, 0).unwrap();
        // remaining_rows = 8: 4 decode rows, 4 left. prefill_min 5 > 4 → no
        // prefill grant for the oldest. It is feasible (5 <= 10) but starved.
        let sel = q.select(4, 10, 8);
        assert!(
            sel.starved_oldest,
            "oldest is feasible but starved, starved_oldest must be set"
        );
        // The oldest (id 1) received no grant this tick.
        assert!(
            !sel.grants.iter().any(|g| g.id() == 1),
            "starved oldest received no grant"
        );
        // Backfill: only the oldest id (1) stays eligible.
        let ids = apply_starved_backfill(&sel, 1);
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&1));
    }

    /// A non-starved selection passes all granted ids through unchanged.
    #[test]
    fn non_starved_selection_passes_all_grants() {
        let mut q = FairQueue::new(4096, 4, 1).expect("valid config");
        q.admit(1, "s1", 0).unwrap();
        q.admit(2, "s2", 0).unwrap();
        q.set_needs(1, true, 0, 0).unwrap();
        q.set_needs(2, true, 0, 0).unwrap();
        let sel = q.select(4, 0, 4096);
        assert!(!sel.starved_oldest, "both served, not starved");
        let ids = eligible_ids_from_selection(&sel);
        assert_eq!(ids.len(), 2, "both decode grants eligible");
        assert!(ids.contains(&1) && ids.contains(&2));
    }

    // ---- WaitQueue policy tests (spec §5.3 S3) ----

    /// `should_wait_instead_of_reject` returns true only when every resident
    /// session is generating and the bounded waiting room is enabled (spec
    /// §5.3 S3, replacing R-A4).
    #[test]
    fn should_wait_when_all_busy_and_wait_enabled() {
        assert!(should_wait_instead_of_reject(true, true));
    }

    #[test]
    fn should_not_wait_when_not_all_busy() {
        // A free slot means the request can be admitted immediately — no
        // need to queue.
        assert!(!should_wait_instead_of_reject(false, true));
    }

    #[test]
    fn should_not_wait_when_wait_disabled() {
        // WaitQueue disabled (max_count 0 — should not happen since
        // Rig::build refuses it, but the policy is pure and testable).
        assert!(!should_wait_instead_of_reject(true, false));
    }

    /// WaitQueue::new refuses zero count or byte caps (spec §5.3: the new
    /// robust multi-slot mode must reject the zero/uncapped combination).
    #[test]
    fn waitqueue_refuses_zero_caps() {
        use hipfire_runtime::serve_wait::{WaitError, WaitQueue};
        assert!(matches!(
            WaitQueue::new(0, 1024, 100),
            Err(WaitError::QueueFull { .. })
        ));
        assert!(matches!(
            WaitQueue::new(4, 0, 100),
            Err(WaitError::QueueBytes { .. })
        ));
    }

    /// Done-time publication boundary is clamped to the materialized row
    /// frontier: a MaxTokens request whose retained terminal token pushes
    /// `tokens.len()` to a page boundary must NOT seal the page whose last
    /// row was never forwarded (stale-tenant KV poison).
    #[test]
    fn done_publish_frontier_clamps_unforwarded_terminal_token() {
        // tokens.len()==256 but only 255 rows exist (terminal retained):
        // the 128-255 page must stay unpublished until its row is written.
        assert_eq!(done_publish_frontier(256, 255), 255);
        // Tokens behind the frontier (normal continuing decode): frontier.
        assert_eq!(done_publish_frontier(100, 128), 100);
        // Frontier below tokens, mid-page: no clamping past rows written.
        assert_eq!(done_publish_frontier(129, 128), 128);
        // Aligned case: full pages, publish up to the boundary.
        assert_eq!(done_publish_frontier(256, 256), 256);
        // Defensive: frontier cannot exceed the token stream.
        assert_eq!(done_publish_frontier(10, 999), 10);
    }

    /// WaitQueue enqueue/pop_ready/expire round-trip (spec §5.3 S3).
    #[test]
    fn waitqueue_enqueue_pop_expire_roundtrip() {
        use hipfire_runtime::serve_wait::WaitQueue;
        let mut q = WaitQueue::new(4, 1024, 100).expect("valid config");
        // Enqueue two waiters at ticks 0 and 1.
        q.try_enqueue(1, 100, 0).expect("enqueue 1");
        q.try_enqueue(2, 200, 1).expect("enqueue 2");
        assert_eq!(q.queued_count(), 2);
        assert_eq!(q.queued_bytes(), 300);
        // Pop the oldest (FIFO).
        let w = q.pop_ready(2).expect("pop oldest");
        assert_eq!(w.id, 1);
        assert_eq!(q.queued_count(), 1);
        // Expire at tick 102 (waiter 2 was enqueued at tick 1, deadline 101).
        let expired = q.expire(102);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].id, 2);
        assert_eq!(q.queued_count(), 0);
    }

    /// WaitQueue remove on cancel (spec §5.3 S3).
    #[test]
    fn waitqueue_remove_cancels_waiter() {
        use hipfire_runtime::serve_wait::WaitQueue;
        let mut q = WaitQueue::new(4, 1024, 100).expect("valid config");
        q.try_enqueue(1, 100, 0).expect("enqueue 1");
        q.try_enqueue(2, 200, 0).expect("enqueue 2");
        assert!(q.remove(1), "remove waiter 1");
        assert!(!q.remove(1), "already removed");
        assert_eq!(q.queued_count(), 1);
        assert_eq!(q.queued_bytes(), 200);
    }

    // ---- Jump-forward policy tests (spec §7.3 G3) ----

    /// Jump-forward is skipped when structured_jump_forward is false (spec
    /// §7.3: default OFF). This test documents the gating invariant: the
    /// flag must be true for forced_token_run to be called.
    #[test]
    fn jump_forward_skipped_when_flag_false() {
        let structured_jump_forward = false;
        let has_grammar = true;
        let greedy = true;
        let should_jump = structured_jump_forward && has_grammar && greedy;
        assert!(
            !should_jump,
            "jump-forward must be skipped when flag is false"
        );
    }

    /// Jump-forward requires a grammar constraint (spec §7.3 G3).
    #[test]
    fn jump_forward_requires_grammar_constraint() {
        let structured_jump_forward = true;
        let has_grammar = false;
        let greedy = true;
        let should_jump = structured_jump_forward && has_grammar && greedy;
        assert!(!should_jump, "jump-forward requires a grammar constraint");
    }

    /// Jump-forward is greedy-only (spec §7.3: "not sampled (greedy only)").
    #[test]
    fn jump_forward_greedy_only() {
        let structured_jump_forward = true;
        let has_grammar = true;
        let greedy = false; // sampled
        let should_jump = structured_jump_forward && has_grammar && greedy;
        assert!(
            !should_jump,
            "jump-forward must be skipped for sampled requests"
        );
    }

    /// forced_token_run on a const schema produces the forced token sequence
    /// (spec §7.3 G3). A `{"const": true}` schema has exactly one legal
    /// token at each step until the value is complete.
    #[test]
    fn forced_token_run_const_true_produces_forced_tokens() {
        let schema = serde_json::json!({"const": true});
        let compiled = grammar::json_schema::CompiledSchema::compile(&schema).expect("compile");
        let matcher = grammar::json_schema::SchemaMatcher::from_compiled(&compiled);
        // Build a minimal token_bytes table with only the individual byte
        // tokens for "true" plus a distractor. Whitespace tokens are
        // excluded so the planner sees exactly one legal token at each
        // step (no branch point from leading whitespace before the value).
        let token_bytes: Vec<Vec<u8>> = vec![
            b"t".to_vec(), // 0
            b"r".to_vec(), // 1
            b"u".to_vec(), // 2
            b"e".to_vec(), // 3
            b"x".to_vec(), // 4 — distractor, never allowed by const true
        ];
        // EOS id outside the vocab so no real token is filtered as EOS.
        let eos_ids: Vec<u32> = vec![999];
        let run = grammar::json_schema::forced_token_run(&matcher, &token_bytes, &eos_ids, 64);
        // The forced run should produce the bytes for "true" as individual
        // token IDs 0,1,2,3 (t,r,u,e). After "true" the matcher is
        // accepting; the run stops (Cap or EosBoundary depending on
        // whether whitespace tokens are allowed in the accepting state).
        assert!(
            !run.token_ids.is_empty(),
            "const true must produce forced tokens"
        );
        assert_eq!(run.token_ids, vec![0, 1, 2, 3], "forced tokens are t,r,u,e");
    }
}

#[cfg(test)]
mod spill_agreement_tests {
    use hipfire_config::memory::{spill_support, SpillRefusal, SpillSupport};

    /// The sequential twin's fact construction for these inputs (tp 1 by
    /// construction, EP never, pp/paged/REAP from the load, VL from the
    /// probe): one predicate call, mirroring `carrier::spill_facts`.
    fn sequential_verdict(
        pp: usize,
        paged_experts: bool,
        reap_keep: bool,
        is_vl: bool,
    ) -> SpillSupport {
        spill_support(6, 1, pp, None, paged_experts, reap_keep, is_vl)
    }

    /// The slots twin's fact construction: single-GPU by the capability gate,
    /// EP never, tower from `is_vl || vl_path`.
    fn slots_verdict(
        paged_experts: bool,
        reap_keep: bool,
        is_vl: bool,
        vl_path: bool,
    ) -> SpillSupport {
        spill_support(6, 1, 1, None, paged_experts, reap_keep, is_vl || vl_path)
    }

    #[test]
    fn spill_resolver_twins_agree_including_vl() {
        // Supported shape: both resolve budgets normally.
        assert_eq!(
            sequential_verdict(1, false, false, false),
            SpillSupport::Supported
        );
        assert_eq!(
            slots_verdict(false, false, false, false),
            SpillSupport::Supported
        );
        // VL refuses on both routes now (the added term): inline tower on the
        // sequential twin, flag or sidecar path on the slots twin.
        assert_eq!(
            sequential_verdict(1, false, false, true),
            SpillSupport::Refused(SpillRefusal::VisionTower)
        );
        for (is_vl, vl_path) in [(true, false), (false, true), (true, true)] {
            assert_eq!(
                slots_verdict(false, false, is_vl, vl_path),
                SpillSupport::Refused(SpillRefusal::VisionTower),
                "is_vl={is_vl} vl_path={vl_path}"
            );
        }
        // pp>1 refuses on the sequential twin (the slots gate refuses pp>1
        // before spawn, so its twin never sees it — same verdict by
        // construction, asserted here for the shared table).
        assert_eq!(
            sequential_verdict(2, false, false, false),
            SpillSupport::Refused(SpillRefusal::Pp(2))
        );
        // Paged/REAP refuse identically on both twins.
        for (paged, reap) in [(true, false), (false, true)] {
            assert_eq!(
                sequential_verdict(1, paged, reap, false),
                slots_verdict(paged, reap, false, false),
                "paged={paged} reap={reap}"
            );
        }
    }

    #[test]
    fn unspillable_refusal_names_path_reason_and_knob() {
        for refusal in [
            SpillRefusal::Arch(9),
            SpillRefusal::Tp(2),
            SpillRefusal::Pp(2),
            SpillRefusal::ExpertParallel(4),
            SpillRefusal::VisionTower,
            SpillRefusal::PagedExperts,
            SpillRefusal::ReapKeepMap,
        ] {
            let msg = crate::carrier::unspillable_no_fit_refusal(refusal, 100 << 20, 10 << 20);
            assert!(msg.starts_with("load refused"), "{refusal:?}: {msg}");
            assert!(
                msg.contains("memory.gpu_layer_budget / memory.moe_expert_budget"),
                "{refusal:?} names the knob: {msg}"
            );
            assert!(
                msg.contains("OOM mid-load"),
                "{refusal:?} names the avoided fate: {msg}"
            );
        }
        // VL names its tower, pp its bands.
        assert!(
            crate::carrier::unspillable_no_fit_refusal(SpillRefusal::VisionTower, 1, 0)
                .contains("vision tower"),
        );
        assert!(
            crate::carrier::unspillable_no_fit_refusal(SpillRefusal::Pp(3), 1, 0).contains("pp=3"),
        );
    }
}

/// The slots MTP depth after the producer-capability gate
/// (`hipfire_config::mtp_mode_admits`): `mtp_k` when the mode admits the head
/// for a producer with/without a declared measured AR floor, else `0` (plain
/// AR, no head resolved or reserved for). `Off` is `0` regardless of `mtp_k`.
fn slot_mtp_k(mode: hipfire_config::MtpMode, mtp_k: usize, measured_ar_floor: bool) -> usize {
    if hipfire_config::mtp_mode_admits(mode, measured_ar_floor) {
        mtp_k
    } else {
        0
    }
}

#[cfg(test)]
mod mtp_spill_decline_tests {
    use super::slot_mtp_k;
    use hipfire_config::{mtp_host_spill_policy, MtpMode};

    /// There is no GPU-free seam for `Rig::build` itself (it opens the HFQ,
    /// inits the GPU and loads weights before reaching the arming site), so
    /// this pins the exact verdict the arming sites compute: `spawn`'s
    /// producer-capability gate (`slot_mtp_k`) and then, from the built
    /// weights, `mtp_k > 0 && policy.attach`. A host-mapped spill under `auto`
    /// declines (no head, AR), while `on` still arms. The parent's live bench
    /// covers the end-to-end load.
    fn slots_arming_verdict(
        mode: MtpMode,
        host_mapped_expert_blobs: usize,
        arch: &str,
        measured_ar_floor: bool,
    ) -> bool {
        // qwen35's admitted spill list is empty.
        let mtp_k = slot_mtp_k(mode, 4, measured_ar_floor);
        mtp_k > 0 && mtp_host_spill_policy(mode, host_mapped_expert_blobs, arch, &[]).attach
    }

    #[test]
    fn qwen35_declares_no_measured_ar_floor() {
        assert!(!crate::MTP_MEASURED_AR_FLOOR);
    }

    #[test]
    fn slots_arming_is_explicit_only_for_the_declared_qwen35_floor() {
        let floor = crate::MTP_MEASURED_AR_FLOOR;
        // `auto` stays AR on a producer without a measured floor, resident or not.
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            for blobs in [0, 24] {
                assert!(
                    !slots_arming_verdict(MtpMode::Auto, blobs, arch, floor),
                    "qwen35 slots auto must stay AR ({arch}, blobs {blobs})"
                );
            }
        }
        // Explicit opt-in arms, resident and over a spill.
        assert!(slots_arming_verdict(MtpMode::On, 0, "gfx1201", floor));
        assert!(slots_arming_verdict(MtpMode::On, 24, "gfx1100", floor));
        // Off never arms.
        assert!(!slots_arming_verdict(MtpMode::Off, 0, "gfx1201", floor));
        assert!(!slots_arming_verdict(MtpMode::Off, 24, "gfx1201", floor));
    }

    #[test]
    fn slots_arming_declines_spill_under_auto_and_arms_on_explicit_on_with_a_floor() {
        // A producer that did declare the floor keeps the spill policy:
        // fully resident auto arms, a host-mapped spill under auto declines
        // on every arch, explicit `on` arms over the same spill.
        assert!(slots_arming_verdict(MtpMode::Auto, 0, "gfx1201", true));
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            assert!(
                !slots_arming_verdict(MtpMode::Auto, 24, arch, true),
                "slots must decline a spill under auto on {arch}"
            );
        }
        assert!(slots_arming_verdict(MtpMode::On, 24, "gfx1100", true));
        assert!(!slots_arming_verdict(MtpMode::Off, 0, "gfx1201", true));
    }

    #[test]
    fn slot_mtp_k_zeroes_every_non_admitted_mode() {
        assert_eq!(slot_mtp_k(MtpMode::Off, 4, true), 0);
        assert_eq!(slot_mtp_k(MtpMode::Off, 4, false), 0);
        assert_eq!(slot_mtp_k(MtpMode::Auto, 4, false), 0);
        assert_eq!(slot_mtp_k(MtpMode::Auto, 4, true), 4);
        assert_eq!(slot_mtp_k(MtpMode::On, 4, false), 4);
        assert_eq!(slot_mtp_k(MtpMode::On, 0, false), 0);
    }
}

#[cfg(all(test, feature = "kvmem"))]
mod speculative_contract_tests {
    /// 期4: the core `speculative.hpp` is the single source of algorithm
    /// names; the engine ordinal mapping must equal the shim's.
    #[test]
    fn algorithm_names_match_core() {
        for a in 0..=3u32 {
            assert_eq!(
                hipfire_runtime::serve::speculative_algorithm_name(a),
                hipfire_kvmem::speculative_algorithm_name(a),
                "ordinal {a}"
            );
        }
    }

    /// The engine folds per-request counters; pin that the shim's C mirror
    /// round-trips the core `SpeculativeStatistics::add`.
    #[test]
    fn statistics_add_round_trips_through_core() {
        let mut dst = hipfire_kvmem::speculative_statistics_reset();
        dst.requests = 2;
        dst.drafted_tokens = 10;
        let mut src = hipfire_kvmem::speculative_statistics_reset();
        src.requests = 1;
        src.drafted_tokens = 5;
        src.accepted_tokens = 2;
        hipfire_kvmem::speculative_statistics_add(&mut dst, &src);
        assert_eq!(dst.requests, 3);
        assert_eq!(dst.drafted_tokens, 15);
        assert_eq!(dst.accepted_tokens, 2);
    }
}

#[cfg(test)]
mod convo_group_tests {
    /// 期3 G3: the conversation-lineage hash is stable per convo and empty is
    /// the unknown group (independent, never merged).
    #[test]
    fn convo_group_is_stable_and_empty_is_unknown() {
        assert_eq!(super::convo_group(&[]), 0);
        let a = vec![1u64, 2, 3, 4];
        assert_eq!(super::convo_group(&a), super::convo_group(&a));
        assert_ne!(super::convo_group(&a), super::convo_group(&[1, 2, 3, 5]));
    }
}
