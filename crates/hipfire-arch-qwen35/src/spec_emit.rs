// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Qwen3.5 per-token spec-decode emission (`SpecEmit`).
//!
//! Relocated out of the daemon example: the emitter names qwen35-local
//! `grammar::{Matcher, ToolSchema}`, so it is a model-family definition and
//! belongs in this crate. The daemon obtains it arch-erased via
//! `Carrier::make_spec_emitter` → `Qwen35Emit::from_ctx`, which builds the
//! grammar `ToolSchema` list from the request's raw tool JSON internally (that
//! JSON→schema extraction also used to live in the daemon).
//!
//! Client-visible text and structured tool calls are owned by the runtime
//! [`ToolOutputRouter`]: EosFilter-emitted UTF-8 is routed incrementally;
//! protocol markers never enter `ClientEvent::Token`; complete calls are held
//! on [`FinishSummary`] until the daemon wrapper classifies a tool-safe terminal.

use crate::grammar;
use hipfire_runtime::emit_text::{
    currently_in_think, ThinkBudget, ThinkBudgetStep, ThinkOutputRouter, ThinkRouteEvent,
    ToolOutputRouter, ToolRouteEvent, QWEN_TEMPLATE_THINK_CLOSE,
};
use hipfire_runtime::eos_filter::{EosFilter, FilterAction};
use hipfire_runtime::prompt_frame::AssistantPrefix;
use hipfire_runtime::spec::{
    ClientEvent, EmitOutcome, FinishSummary, SpecEmit, SpecEmitCtx, StopReason,
};
use hipfire_runtime::stop_sequence::{MinTokens, StopMatcher};
use hipfire_runtime::tokenizer::Tokenizer;

pub struct Qwen35Emit<'a> {
    tokenizer: &'a Tokenizer,
    filter: EosFilter,
    /// Raw bytes of `streamed_tokens`, grown one token at a time; its length
    /// is the offset past which bytes have not yet been fed to the filter.
    stream_bytes: Vec<u8>,
    /// Every committed token in order, for byte decoding + the cache-store the
    /// daemon does after the loop (exposed via [`Self::streamed_tokens`]).
    streamed_tokens: Vec<u32>,
    /// Incremental tool-protocol authority. Fed only EosFilter-emitted UTF-8.
    router: ToolOutputRouter,
    /// Incremental reasoning/content authority, upstream of tool routing.
    think_router: ThinkOutputRouter,
    /// Classifier-authorized visible prose accumulated this turn (cache fp).
    visible_acc: String,
    /// Latched when the router fails closed mid-stream or at finish.
    router_malformed: bool,
    /// Filter saw a decoded stop marker (`stop_at`) this turn.
    filter_stopped: bool,
    grammar_active: bool,
    grammar_matcher: grammar::Matcher,
    /// Set when a committed token violated the grammar; the daemon reads it via
    /// [`Self::grammar_violated`] to force the post-turn KV/recurrent reset.
    grammar_violated: bool,
    eos_token: u32,
    im_end_token: Option<u32>,
    /// User `stop` sequences, matched on answer text before it is emitted.
    stop: StopMatcher,
    /// vLLM-compatible `min_tokens`: EOS, decoded end-of-turn markers and user
    /// stop sequences stay inactive until this many tokens are committed.
    min_tokens: MinTokens,
    /// Whether the most recently pushed token was sampled with stops active
    /// (`min_tokens` already reached). A token sampled before that is an
    /// ordinary token even when it is the EOS id.
    stops_active: bool,
    /// Explicit `max_think_tokens` accounting (shared with the AR producer).
    think_budget: ThinkBudget,
    open_think_prefix: bool,
    /// Text spliced when the think budget runs out.
    think_close: String,
    /// Think-budget force-close continuation drained by `take_forced` so the
    /// target advances over `</think>` and continues into visible output.
    forced: Vec<u32>,
    /// `generated` counter at the point of the most recent `observe` — only used
    /// for the attractor-detect log message (byte-for-byte stderr parity).
    generated_hint: usize,
}

fn think_continuation_text() -> String {
    hipfire_config::developer_var("HIPFIRE_THINK_CONTINUATION")
        .unwrap_or_else(|_| "</think>\n\n".to_string())
}

impl<'a> Qwen35Emit<'a> {
    /// Build the qwen35 emitter from the model-independent [`SpecEmitCtx`],
    /// extracting the grammar `ToolSchema` list from the request's raw tool JSON
    /// (`ctx.tools`). Returns the arch-erased `Box<dyn SpecEmit>` the daemon
    /// drives. The JSON→schema extraction mirrors the daemon's old
    /// `tool_schemas_dflash` builder. A spent think budget splices
    /// `HIPFIRE_THINK_CONTINUATION` (default `</think>\n\n`).
    pub fn from_ctx(ctx: SpecEmitCtx<'a>) -> Box<dyn SpecEmit + 'a> {
        Self::with_think_close(ctx, think_continuation_text(), false)
    }

    /// [`Self::from_ctx`] for serving: malformed tool spans are released as
    /// ordinary assistant text (PolyStrata-style fallback) instead of ending
    /// the turn with `malformed_protocol`, so a degenerate call degrades the
    /// answer instead of failing the request.
    pub fn from_ctx_recover(ctx: SpecEmitCtx<'a>) -> Box<dyn SpecEmit + 'a> {
        Self::with_think_close(ctx, think_continuation_text(), true)
    }

    /// [`Self::from_ctx`] for Qwen4, whose AR producer splices the template's
    /// own close ([`QWEN_TEMPLATE_THINK_CLOSE`]) when the think budget runs
    /// out; MTP must splice the same tokens to produce the same text.
    pub fn from_ctx_template_think_close(ctx: SpecEmitCtx<'a>) -> Box<dyn SpecEmit + 'a> {
        Self::with_think_close(ctx, QWEN_TEMPLATE_THINK_CLOSE.to_string(), false)
    }

    fn with_think_close(
        ctx: SpecEmitCtx<'a>,
        think_close: String,
        tool_recover: bool,
    ) -> Box<dyn SpecEmit + 'a> {
        let tool_protocol_enabled = ctx.tools.is_some();
        let tool_schemas: Vec<grammar::ToolSchema> = ctx
            .tools
            .filter(|_| ctx.enable_grammar)
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        let func = t.get("function").unwrap_or(t);
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())?
                            .to_string();
                        // Required-field list from JSON schema's
                        // `parameters.required`. Empty if the tool
                        // declares no required args.
                        let required: Vec<String> = func
                            .get("parameters")
                            .and_then(|p| p.get("required"))
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();
                        Some(grammar::ToolSchema { name, required })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let grammar_active = !tool_schemas.is_empty();
        let open_think_prefix = matches!(ctx.assistant_prefix, AssistantPrefix::OpenThink);
        Box::new(Self {
            tokenizer: ctx.tokenizer,
            filter: EosFilter::new(hipfire_runtime::eos_filter::qwen35_eos_filter_config()),
            stream_bytes: Vec::new(),
            streamed_tokens: Vec::new(),
            router: if tool_protocol_enabled {
                ToolOutputRouter::new().with_recover(tool_recover)
            } else {
                ToolOutputRouter::disabled()
            },
            think_router: ThinkOutputRouter::new(open_think_prefix),
            visible_acc: String::new(),
            router_malformed: false,
            filter_stopped: false,
            grammar_active,
            grammar_matcher: grammar::Matcher::with_config(
                tool_schemas,
                crate::grammar_config::resolve_qwen35_grammar_config(),
            ),
            grammar_violated: false,
            eos_token: ctx.eos,
            im_end_token: ctx.im_end,
            stop: StopMatcher::new(&ctx.stop),
            min_tokens: MinTokens(ctx.min_tokens),
            stops_active: ctx.min_tokens == 0,
            think_budget: ThinkBudget::new(ctx.max_think),
            open_think_prefix,
            think_close,
            forced: Vec::new(),
            generated_hint: 0,
        })
    }

    /// Route one content chunk. Visible prose becomes
    /// `ClientEvent::Token`; complete tool calls stay buffered in the router
    /// until [`SpecEmit::finish`]. Malformed protocol latches fail-closed.
    fn route_content_text(&mut self, text: &str, events: &mut Vec<ClientEvent>) {
        if text.is_empty() || self.router_malformed {
            return;
        }
        match self.router.push(text) {
            Ok(evs) => {
                for ev in evs {
                    match ev {
                        ToolRouteEvent::VisibleText(vt) => {
                            self.visible_acc.push_str(vt.as_str());
                            events.push(ClientEvent::Token(vt.into_string()));
                        }
                        ToolRouteEvent::ToolCall(_) => {
                            // Buffered in router; released only on tool-safe finish.
                        }
                    }
                }
            }
            Err(err) => {
                if hipfire_config::developer_var("HIPFIRE_SLOT_TRACE").is_ok() {
                    eprintln!(
                        "[tool-router] malformed at gen={}: {err} | body tail={:?}",
                        self.generated_hint,
                        self.router.debug_buffer(),
                    );
                }
                self.router_malformed = true;
            }
        }
    }

    /// Route classified reasoning/content events. Content passes the user
    /// `stop` matcher, then the tool protocol router; reasoning bypasses both
    /// and is never cached as visible prose. Once a stop sequence matches,
    /// nothing after it is routed.
    fn route_think_events(
        &mut self,
        channel_events: Vec<ThinkRouteEvent>,
        events: &mut Vec<ClientEvent>,
    ) {
        for channel_event in channel_events {
            if self.router_malformed || self.stop.matched() {
                break;
            }
            match channel_event {
                ThinkRouteEvent::Reasoning(text) => {
                    events.push(ClientEvent::Reasoning(text));
                }
                ThinkRouteEvent::Content(text) => {
                    let text = self.stop.push(&text);
                    self.route_content_text(&text, events);
                }
            }
        }
    }

    /// Route one EosFilter-emitted UTF-8 chunk through reasoning classification.
    fn route_filter_text(&mut self, text: &str, events: &mut Vec<ClientEvent>) {
        if text.is_empty() {
            return;
        }
        let mut channel_events = Vec::new();
        self.think_router.push_into(text, &mut channel_events);
        self.route_think_events(channel_events, events);
    }

    /// Apply one filter observe step into the semantic router.
    fn apply_filter_action(&mut self, action: FilterAction, events: &mut Vec<ClientEvent>) {
        match action {
            FilterAction::Emit(text_bytes) => {
                let text = std::str::from_utf8(&text_bytes).unwrap_or("");
                self.route_filter_text(text, events);
            }
            FilterAction::EmitAndStop(text_bytes) => {
                let text = std::str::from_utf8(&text_bytes).unwrap_or("");
                self.route_filter_text(text, events);
                self.filter_stopped = true;
            }
            FilterAction::Stop => {
                self.filter_stopped = true;
            }
            FilterAction::Hold => {}
        }
    }

    /// Push a committed token, run it through the byte filter + router, and
    /// return the `Committed` + (optional) visible `Token` events — the shared
    /// tail of `begin`'s first-token emit and `observe`'s per-token emit.
    fn push_and_filter(&mut self, token: u32) -> Vec<ClientEvent> {
        let mut events = Vec::new();
        // `min_tokens`: the token about to be committed is the
        // `streamed_tokens.len() + 1`-th; EOS and stops apply only once
        // `min_tokens` tokens precede it.
        self.stops_active = self.min_tokens.stops_active(self.streamed_tokens.len());
        self.filter.set_stops_suspended(!self.stops_active);
        self.stop.set_suspended(!self.stops_active);
        self.streamed_tokens.push(token);
        events.push(ClientEvent::Committed {
            id: token,
            idx: self.streamed_tokens.len() - 1,
        });
        let fed = self.stream_bytes.len();
        self.tokenizer
            .decode_token_bytes_into(token, &mut self.stream_bytes);
        let action = self.filter.observe(&self.stream_bytes[fed..]);
        self.apply_filter_action(action, &mut events);
        events
    }

    /// Drain pending EOT-prefix prose, finalize partial think markers, then
    /// release answer text held back as a possible stop prefix.
    fn drain_pending_into_router(&mut self, events: &mut Vec<ClientEvent>) {
        let pending = self.filter.flush_pending();
        if !pending.is_empty() {
            let text = std::str::from_utf8(&pending).unwrap_or("");
            self.route_filter_text(text, events);
        }

        let mut channel_events = Vec::new();
        self.think_router.finish_into(&mut channel_events);
        self.route_think_events(channel_events, events);
        let held = self.stop.finish();
        self.route_content_text(&held, events);
    }

    /// Whether a streamed token id is a decoded EOT / terminator.
    fn token_is_eot(&self, token: u32) -> bool {
        token == self.eos_token
            || self.im_end_token == Some(token)
            || self.tokenizer.is_terminator(token)
    }

    /// True when this turn ended on a decoded EOT (token id or filter stop_at).
    pub fn decoded_eot(&self) -> bool {
        self.filter_stopped
            || (self.stops_active
                && self
                    .streamed_tokens
                    .last()
                    .copied()
                    .map(|t| self.token_is_eot(t))
                    .unwrap_or(false))
    }

    /// Count the just-pushed token against the think budget. Queues the close
    /// on the first exhaustion; a second one (span re-opened) is a stop.
    fn think_budget_stop(&mut self) -> Option<StopReason> {
        if !self.think_budget.enabled() {
            return None;
        }
        let raw_str = std::str::from_utf8(&self.stream_bytes).unwrap_or("");
        let in_think = currently_in_think(raw_str, self.open_think_prefix);
        match self.think_budget.observe(in_think) {
            ThinkBudgetStep::Continue => None,
            ThinkBudgetStep::ForceClose => {
                self.forced = self.tokenizer.encode(&self.think_close);
                self.think_budget.closing(self.forced.len());
                eprintln!(
                    "[think-cap] route=spec think budget reached; closing with tokens {:?}",
                    self.forced
                );
                None
            }
            // A pathological re-open after the injected close hard-stops.
            ThinkBudgetStep::Exhausted => Some(StopReason::ThinkCap),
        }
    }

    /// Producer-authorized visible text for asst-turn cache fingerprinting.
    pub fn visible_text(&self) -> &str {
        &self.visible_acc
    }
}

impl<'a> SpecEmit for Qwen35Emit<'a> {
    fn begin(&mut self, first_token: u32) -> EmitOutcome {
        // First-token emit (committed + filtered token), then seed the grammar
        // matcher with the first token's text. Mirrors generate_dflash 4452-4489.
        let events = self.push_and_filter(first_token);
        if self.grammar_active {
            let text = self.tokenizer.decode(&[first_token]);
            self.grammar_matcher.advance(&text);
        }
        // First-token EOS guard: if the prefill's first token is itself a
        // terminator we must NOT enter the spec loop (the inline `while
        // !first_token_is_eos`). Surface as a stop so the caller skips the loop.
        // Same precedence as `observe`: EOT token id, then filter stop_at, then
        // user stop-sequence — so generate_spec never enters spec.step.
        if (self.stops_active && self.token_is_eot(first_token)) || self.filter_stopped {
            return EmitOutcome {
                events,
                stop: Some(StopReason::Eos),
            };
        }
        if self.stop.matched() {
            return EmitOutcome {
                events,
                stop: Some(StopReason::StopSequence),
            };
        }
        // The first token counts against the think budget like every later one.
        let stop = self.think_budget_stop();
        EmitOutcome { events, stop }
    }

    fn observe(&mut self, token: u32) -> EmitOutcome {
        // Grammar pre-check (POST-acceptance, before emit). A rejected token is
        // NOT emitted; treat as a grammar violation → stop. Mirrors 4565-4584.
        //
        // This is the **tool-call grammar** guard (the `<tool_call>` format), NOT
        // the JSON Schema `response_format` constraint (spec §7 G1/G2). The
        // `response_format` constraint is applied as a pre-sampling mask in
        // the slot engine's `run_loop` (serve_engine.rs) BEFORE the GPU
        // sampler, so no forbidden token can be sampled — this guard never
        // sees a `response_format` violation (spec §7.2 G2: "the hard
        // grammar mask participates before probability-dependent
        // filtering/renormalization").
        //
        // For the slots path: `tools: None` is passed (slots.rs), so
        // `grammar_active` is false and this guard is inactive. When tools
        // are eventually supported on the slots path, the tool-call grammar
        // mask should also be applied pre-sampling in serve_engine.rs
        // rather than post-acceptance here (spec §7.2 G2: "Constrained slot
        // requests use AR until the verifier applies constraints before
        // acceptance, not only after a bad token has advanced state").
        if self.grammar_active {
            let text = self.tokenizer.decode(&[token]);
            if !self.grammar_matcher.is_token_allowed(&text) {
                eprintln!(
                    "[grammar-dflash] rejected token id={} text={:?} (matcher.state={:?}) — forcing EOS | {}",
                    token, text, self.grammar_matcher.state(), self.grammar_matcher.debug_close_reject(),
                );
                self.grammar_violated = true;
                return EmitOutcome {
                    events: Vec::new(),
                    stop: Some(StopReason::GrammarViolation),
                };
            }
            let was_detected = self.grammar_matcher.attractor_detected();
            self.grammar_matcher.advance(&text);
            if !was_detected && self.grammar_matcher.attractor_detected() {
                eprintln!(
                    "[grammar-dflash-ngram] attractor detected in tool_call args at gen={} — forcing close",
                    self.generated_hint,
                );
            }
        }

        // Emit the token (committed + filtered + routed).
        let mut events = self.push_and_filter(token);

        // EOS check (token id). Mirrors 4608-4612. Inactive below `min_tokens`.
        if self.stops_active && self.token_is_eot(token) {
            return EmitOutcome {
                events,
                stop: Some(StopReason::Eos),
            };
        }
        // Filter-decoded stop marker (byte form of im_end / endoftext).
        if self.filter_stopped {
            return EmitOutcome {
                events,
                stop: Some(StopReason::Eos),
            };
        }

        // User stop sequence matched in the answer text this token completed.
        if self.stop.matched() {
            return EmitOutcome {
                events,
                stop: Some(StopReason::StopSequence),
            };
        }

        // max_think_tokens enforcement: a spent budget queues the close
        // (`take_forced`); a re-opened span spending it again stops.
        let stop = self.think_budget_stop();
        EmitOutcome { events, stop }
    }

    fn finish(mut self: Box<Self>) -> FinishSummary {
        // Drain filter pending, then finalize the incremental router. Structured
        // ToolCalls are held on FinishSummary for the wrapper — generate_spec
        // must NOT render them before length/malformed/open-think is known.
        let mut events = Vec::new();
        self.drain_pending_into_router(&mut events);
        let open_think = self.think_router.in_think();
        let decoded_eot = self.decoded_eot();

        if open_think && !self.stop.matched() {
            // The turn ended inside `<think>`: release the drained reasoning and
            // any held answer prose, never a tool call. The wrapper ends it as
            // a reasoning-only `stop` (or `length`) with no cache store.
            if let Ok(evs) = self.router.finish() {
                for ev in evs {
                    if let ToolRouteEvent::VisibleText(vt) = ev {
                        self.visible_acc.push_str(vt.as_str());
                        events.push(ClientEvent::Token(vt.into_string()));
                    }
                }
            }
            return FinishSummary {
                events,
                finish_reason: "open_think",
                tool_calls: 0,
                visible_text: std::mem::take(&mut self.visible_acc),
                decoded_eot,
                open_think: true,
            };
        }

        if self.router_malformed || self.router.is_malformed() {
            if hipfire_config::developer_var("HIPFIRE_SLOT_TRACE").is_ok() {
                eprintln!(
                    "[tool-router] finish malformed: {:?} | body tail={:?}",
                    self.router.malformed_detail(),
                    self.router.debug_buffer(),
                );
            }
            let _ = self.router.finish();
            return FinishSummary {
                events,
                finish_reason: "malformed_protocol",
                tool_calls: 0,
                visible_text: std::mem::take(&mut self.visible_acc),
                decoded_eot,
                open_think: false,
            };
        }

        let buffered_before = self.router.tool_calls().to_vec();
        match self.router.finish() {
            // A stop sequence that cut a tool call short leaves no call and
            // no malformed verdict: the call text was never part of the answer.
            Err(_) if self.stop.matched() => FinishSummary {
                events,
                finish_reason: "stop",
                tool_calls: 0,
                visible_text: std::mem::take(&mut self.visible_acc),
                decoded_eot,
                open_think: false,
            },
            // The output ended inside a tool call (unclosed span, or a body the
            // final flush rejects). Like an open think span, this is its own
            // reason: the wrapper lets a pure length exit win (AR's rule in
            // `qwen_ar_finish_route`) and fails closed only when the model
            // ended the turn there. A mid-stream latch stays `malformed_protocol`.
            Err(err) => {
                if hipfire_config::developer_var("HIPFIRE_SLOT_TRACE").is_ok() {
                    eprintln!(
                        "[tool-router] finish truncated: {err} | stop_matched={}",
                        self.stop.matched(),
                    );
                }
                FinishSummary {
                    events,
                    finish_reason: "truncated_tool_call",
                    tool_calls: 0,
                    visible_text: std::mem::take(&mut self.visible_acc),
                    decoded_eot,
                    open_think: false,
                }
            }
            Ok(evs) => {
                let mut finished_calls = buffered_before;
                for ev in evs {
                    match ev {
                        ToolRouteEvent::VisibleText(vt) => {
                            self.visible_acc.push_str(vt.as_str());
                            events.push(ClientEvent::Token(vt.into_string()));
                        }
                        ToolRouteEvent::ToolCall(tc) => {
                            finished_calls.push(tc);
                        }
                    }
                }
                let visible_text = std::mem::take(&mut self.visible_acc);
                if !finished_calls.is_empty() {
                    let n = finished_calls.len();
                    events.push(ClientEvent::ToolCalls(finished_calls));
                    FinishSummary {
                        events,
                        finish_reason: "tool_calls",
                        tool_calls: n,
                        visible_text,
                        decoded_eot,
                        open_think: false,
                    }
                } else {
                    FinishSummary {
                        events,
                        finish_reason: "stop",
                        tool_calls: 0,
                        visible_text,
                        decoded_eot,
                        open_think: false,
                    }
                }
            }
        }
    }

    /// The full committed-token stream (incl. the first token), for the daemon's
    /// post-loop asst-turn cache store.
    fn streamed_tokens(&self) -> &[u32] {
        &self.streamed_tokens
    }

    /// Whether a committed token tripped the grammar matcher (daemon forces a
    /// full KV/recurrent reset for the next turn when true).
    fn grammar_violated(&self) -> bool {
        self.grammar_violated
    }

    fn decoded_eot(&self) -> bool {
        // Prefer the inherent helper (token id + filter stop_at).
        Qwen35Emit::decoded_eot(self)
    }

    /// Hint the emitter of the daemon's current `generated` count so the
    /// attractor-detect log message reports the same number it did inline.
    fn set_generated_hint(&mut self, generated: usize) {
        self.generated_hint = generated;
    }

    fn take_forced(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.forced)
    }

    fn window_cap(&self) -> Option<usize> {
        if !self.think_budget.enabled() {
            return None;
        }
        let raw_str = std::str::from_utf8(&self.stream_bytes).unwrap_or("");
        let in_think = currently_in_think(raw_str, self.open_think_prefix);
        self.think_budget.window_cap(in_think)
    }

    /// Whether the stream currently sits inside an unclosed think span. The
    /// think router's flag is authoritative: a held-back PARTIAL close marker
    /// keeps the span open (the close has not been consumed), which matches
    /// PolyStrata's `parser.state == "reasoning"` with a non-empty buffer.
    fn open_think_now(&self) -> bool {
        self.think_router.in_think()
    }

    /// Clean token boundary: neither the byte filter nor the think router is
    /// holding back bytes (a partial EOT prefix or a partial think marker).
    /// The recovery decision (#728) may only be taken at such a boundary.
    fn at_clean_boundary(&self) -> bool {
        !self.filter.has_pending() && !self.think_router.has_pending()
    }

    /// The family's think-close text, for the daemon's #1053 injection. This
    /// is the same string `think_budget_stop` splices, so the injected close
    /// is byte-identical to what a budget force-close would have produced.
    fn think_close_text(&self) -> Option<&str> {
        Some(&self.think_close)
    }

    /// Feed a daemon-injected token (the think-close splice, either the
    /// #1053 repair or the multi-slot budget close) through the emission
    /// pipeline as pure STATE REPAIR: bytes are decoded and routed through
    /// the think router / content router exactly as if generated, but no
    /// `Committed` event is posted, no grammar cursor moves, and neither the
    /// EOT filter nor the stop matcher observes the bytes — an injected
    /// close must never arm a terminator or stop sequence. Each injected
    /// token also discharges one queued budget-close token: the splice
    /// delivers them without an `observe` call, and a stale `closing` count
    /// would swallow the continuation's first tokens.
    fn observe_injected(&mut self, token: u32) -> Vec<ClientEvent> {
        let mut events = Vec::new();
        self.think_budget.consume_closing(1);
        let fed = self.stream_bytes.len();
        self.tokenizer
            .decode_token_bytes_into(token, &mut self.stream_bytes);
        let text = std::str::from_utf8(&self.stream_bytes[fed..])
            .unwrap_or("")
            .to_string();
        self.route_filter_text(&text, &mut events);
        events
    }
}

// ── Deterministic non-GPU producer tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::prompt_frame::ToolCall;
    use hipfire_runtime::spec::SpecEmit;
    use hipfire_runtime::tokenizer::Tokenizer;

    fn json_escape(s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    /// Mirror of production GPT-2 byte_to_unicode (prompt_frame test helper).
    fn byte_to_gpt2_char_test(b: u8) -> char {
        let mut bs: Vec<u32> = Vec::new();
        bs.extend((b'!' as u32)..=(b'~' as u32));
        bs.extend((0xA1u32)..=(0xACu32));
        bs.extend((0xAEu32)..=(0xFFu32));
        let mut cs: Vec<u32> = bs.clone();
        let mut n: u32 = 0;
        for byte in 0u32..=255u32 {
            if !bs.contains(&byte) {
                bs.push(byte);
                cs.push(256 + n);
                n += 1;
            }
        }
        for (bb, cc) in bs.into_iter().zip(cs.into_iter()) {
            if bb == b as u32 {
                return char::from_u32(cc).unwrap();
            }
        }
        char::from_u32(b as u32).unwrap()
    }

    /// Minimal BPE tokenizer covering ASCII + ChatML / think specials.
    fn test_tokenizer() -> Tokenizer {
        let mut entries: Vec<String> = Vec::new();
        entries.push(r#""<|im_start|>": 0"#.to_string());
        entries.push(r#""<|im_end|>": 1"#.to_string());
        entries.push(r#""<think>": 2"#.to_string());
        entries.push(r#""</think>": 3"#.to_string());
        entries.push(r#""system": 4"#.to_string());
        entries.push(r#""user": 5"#.to_string());
        entries.push(r#""assistant": 6"#.to_string());
        entries.push(r#""\n": 7"#.to_string());
        entries.push(r#""Ġ": 8"#.to_string());
        entries.push(r#""<|endoftext|>": 9"#.to_string());
        for b in 0u32..=255u32 {
            let ch = byte_to_gpt2_char_test(b as u8);
            let escaped = json_escape(&ch.to_string());
            entries.push(format!(r#""{}": {}"#, escaped, 100 + b));
        }
        let vocab_block = entries.join(", ");
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", "vocab": {{ {vocab} }}, "merges": []}},
                "added_tokens": [
                    {{"id": 0, "content": "<|im_start|>", "special": true}},
                    {{"id": 1, "content": "<|im_end|>", "special": true}},
                    {{"id": 2, "content": "<think>", "special": true}},
                    {{"id": 3, "content": "</think>", "special": true}},
                    {{"id": 9, "content": "<|endoftext|>", "special": true}}
                ]
            }}"#,
            vocab = vocab_block,
        );
        Tokenizer::from_hf_json(&json).expect("test tokenizer")
    }

    fn make_emit<'a>(tok: &'a Tokenizer) -> Box<dyn SpecEmit + 'a> {
        Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: tok,
            eos: 9,
            im_end: Some(1),
            tools: Some(&[]),
            enable_grammar: true,
            stop: Vec::new(),
            min_tokens: 0,
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        })
    }

    fn make_emit_recover<'a>(tok: &'a Tokenizer) -> Box<dyn SpecEmit + 'a> {
        Qwen35Emit::from_ctx_recover(SpecEmitCtx {
            tokenizer: tok,
            eos: 9,
            im_end: Some(1),
            tools: Some(&[]),
            enable_grammar: true,
            stop: Vec::new(),
            min_tokens: 0,
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        })
    }

    /// Drive the production emitter with whole-string encodes (one observe per
    /// encoded token). Returns (stream events, finish, streamed_tokens).
    fn drive_text(text: &str) -> (Vec<ClientEvent>, FinishSummary, Vec<u32>) {
        drive_text_with(text, make_emit)
    }

    fn drive_text_recover(text: &str) -> (Vec<ClientEvent>, FinishSummary, Vec<u32>) {
        drive_text_with(text, make_emit_recover)
    }

    fn drive_text_with(
        text: &str,
        build: for<'a> fn(&'a Tokenizer) -> Box<dyn SpecEmit + 'a>,
    ) -> (Vec<ClientEvent>, FinishSummary, Vec<u32>) {
        let tok = test_tokenizer();
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "encode produced no tokens for {text:?}");
        let mut emit = build(&tok);
        let mut stream = Vec::new();
        let mut first = true;
        for id in &ids {
            let outcome = if first {
                first = false;
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if outcome.stop.is_some() {
                break;
            }
        }
        let streamed = emit.streamed_tokens().to_vec();
        let finish = emit.finish();
        (stream, finish, streamed)
    }

    fn tokens_text(events: &[ClientEvent]) -> String {
        let mut s = String::new();
        for ev in events {
            if let ClientEvent::Token(t) = ev {
                s.push_str(t);
            }
        }
        s
    }

    fn reasoning_text(events: &[ClientEvent]) -> String {
        let mut s = String::new();
        for ev in events {
            if let ClientEvent::Reasoning(t) = ev {
                s.push_str(t);
            }
        }
        s
    }

    fn held_calls(finish: &FinishSummary) -> Vec<ToolCall> {
        finish
            .events
            .iter()
            .find_map(|ev| match ev {
                ClientEvent::ToolCalls(c) => Some(c.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    fn hermes_call(name: &str, args: &str) -> String {
        format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
    }

    #[test]
    fn prose_only_is_visible_stop() {
        let (stream, finish, raw) = drive_text("Hello world");
        assert!(!raw.is_empty());
        assert_eq!(tokens_text(&stream), "Hello world");
        assert_eq!(finish.finish_reason, "stop");
        assert_eq!(finish.tool_calls, 0);
        assert!(held_calls(&finish).is_empty());
        assert!(!tokens_text(&stream).contains("<tool_call>"));
    }

    #[test]
    fn valid_hermes_call_held_on_finish() {
        let body = format!("Sure.{}", hermes_call("get_weather", r#"{"city":"SF"}"#));
        let (stream, finish, raw) = drive_text(&body);
        assert!(!raw.is_empty());
        assert_eq!(tokens_text(&stream), "Sure.");
        assert!(!tokens_text(&stream).contains("<tool_call>"));
        assert!(!tokens_text(&stream).contains("get_weather"));
        assert_eq!(finish.finish_reason, "tool_calls");
        assert_eq!(finish.tool_calls, 1);
        let calls = held_calls(&finish);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn xml_tool_call_held_when_tools_present_and_grammar_off() {
        // Qwen3.5/3.8 XML-native: grammar stays off, but tools must still
        // enable ToolOutputRouter or `<tool_call>` leaks as assistant text.
        let tok = test_tokenizer();
        let tools = [serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: Some(&tools),
            enable_grammar: false,
            stop: Vec::new(),
            min_tokens: 0,
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let text = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        let mut stream = Vec::new();
        let mut first = true;
        for id in &ids {
            let outcome = if first {
                first = false;
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if outcome.stop.is_some() {
                break;
            }
        }
        let finish = emit.finish();
        assert!(!tokens_text(&stream).contains("<tool_call>"));
        assert_eq!(finish.finish_reason, "tool_calls");
        assert_eq!(finish.tool_calls, 1);
        let calls = held_calls(&finish);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
    }

    #[test]
    fn multiple_calls_and_surrounding_prose() {
        let body = format!(
            "A{}B{}",
            hermes_call("alpha", r#"{}"#),
            hermes_call("beta", r#"{"x":1}"#)
        );
        let (stream, finish, _) = drive_text(&body);
        assert_eq!(tokens_text(&stream), "AB");
        assert_eq!(finish.finish_reason, "tool_calls");
        assert_eq!(finish.tool_calls, 2);
        let names: Vec<_> = held_calls(&finish).into_iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn marker_split_across_tokens_never_leaks() {
        let body = hermes_call("t", r#"{}"#);
        let (stream, finish, _) = drive_text(&body);
        let visible = tokens_text(&stream);
        assert!(!visible.contains("<tool_call>"));
        assert!(!visible.contains("</tool_call>"));
        assert!(!visible.contains("\"name\""));
        assert_eq!(finish.finish_reason, "tool_calls");
        assert_eq!(finish.tool_calls, 1);
    }

    #[test]
    fn unclosed_span_is_truncated_tool_call_with_no_calls() {
        let (stream, finish, _) = drive_text("pre<tool_call>{\"name\":\"x\"");
        assert!(!tokens_text(&stream).contains("<tool_call>"));
        assert_eq!(finish.finish_reason, "truncated_tool_call");
        assert_eq!(finish.tool_calls, 0);
        assert!(held_calls(&finish).is_empty());
    }

    #[test]
    fn malformed_empty_body_is_fail_closed() {
        let (stream, finish, _) = drive_text("x<tool_call></tool_call>");
        assert!(!tokens_text(&stream).contains("<tool_call>"));
        assert_eq!(finish.finish_reason, "malformed_protocol");
        assert_eq!(finish.tool_calls, 0);
    }

    #[test]
    fn recover_mode_releases_malformed_spans_as_answer_text() {
        // Strict mode fails these turns closed (`malformed_protocol` /
        // `truncated_tool_call`); the serving constructor must instead end
        // them normally with the degenerate span as assistant content.
        let stacked = "<tool_call>{\"name\":\"a\",\"arguments\":{}}<tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
        let (stream, finish, _) = drive_text_recover(stacked);
        assert_eq!(finish.finish_reason, "stop");
        assert_eq!(finish.tool_calls, 0);
        assert!(held_calls(&finish).is_empty());
        let mut visible = tokens_text(&stream);
        visible.push_str(&tokens_text(&finish.events));
        assert_eq!(visible, stacked);

        let unclosed = "pre<tool_call>{\"name\":\"x\"";
        let (stream2, finish2, _) = drive_text_recover(unclosed);
        assert_eq!(finish2.finish_reason, "stop");
        assert_eq!(finish2.tool_calls, 0);
        let mut visible2 = tokens_text(&stream2);
        visible2.push_str(&tokens_text(&finish2.events));
        assert_eq!(visible2, unclosed);
    }

    #[test]
    fn think_bytes_suppressed_from_token_channel() {
        let (stream, finish, raw) = drive_text("<think>secret</think>answer");
        assert!(!raw.is_empty());
        let visible = tokens_text(&stream);
        assert!(!visible.contains("<think>"));
        assert!(!visible.contains("secret"));
        assert!(visible.contains("answer"));
        assert_eq!(reasoning_text(&stream), "secret");
        assert_eq!(finish.finish_reason, "stop");
    }

    #[test]
    fn im_end_marker_suppressed_and_decoded_eot() {
        let tok = test_tokenizer();
        let mut ids = tok.encode("hi");
        ids.push(1); // <|im_end|>
        let mut emit = make_emit(&tok);
        let mut stream = Vec::new();
        let mut first = true;
        let mut stopped = false;
        for id in &ids {
            let outcome = if first {
                first = false;
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if outcome.stop == Some(StopReason::Eos) {
                stopped = true;
                break;
            }
        }
        assert!(stopped, "im_end must stop via Eos");
        let visible = tokens_text(&stream);
        assert!(!visible.contains("<|im_end|>"));
        assert!(visible.contains("hi"));
        let finish = emit.finish();
        assert_eq!(finish.finish_reason, "stop");
        assert_eq!(finish.tool_calls, 0);
    }

    fn min_tokens_emit<'a>(
        tok: &'a Tokenizer,
        min_tokens: usize,
        stop: Vec<String>,
    ) -> Box<dyn SpecEmit + 'a> {
        Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            enable_grammar: false,
            stop,
            min_tokens,
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        })
    }

    /// Feed `ids` (first through `begin`); returns (events, index of the token
    /// that stopped, its reason).
    fn drive_ids(
        emit: &mut Box<dyn SpecEmit + '_>,
        ids: &[u32],
    ) -> (Vec<ClientEvent>, Option<(usize, StopReason)>) {
        let mut stream = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            let outcome = if i == 0 {
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            stream.extend(outcome.events);
            if let Some(reason) = outcome.stop {
                return (stream, Some((i, reason)));
            }
        }
        (stream, None)
    }

    #[test]
    fn min_tokens_holds_eos_inactive_until_reached() {
        let tok = test_tokenizer();
        let a = tok.encode("a")[0];
        let b = tok.encode("b")[0];
        // <|im_end|> (1) and <|endoftext|> (9) as 2nd and 3rd token, then a
        // real stop as the 5th: min_tokens = 4 keeps the first two ordinary.
        let ids = [a, 1, 9, b, 1];
        let mut emit = min_tokens_emit(&tok, 4, Vec::new());
        let (stream, stop) = drive_ids(&mut emit, &ids);
        assert_eq!(stop, Some((4, StopReason::Eos)), "stops on the 5th token");
        let committed = stream
            .iter()
            .filter(|e| matches!(e, ClientEvent::Committed { .. }))
            .count();
        assert_eq!(committed, 5, "every token below the limit is committed");
        let visible = tokens_text(&stream);
        assert_eq!(visible, "ab", "suppressed EOS markers are never shown");
        assert!(emit.decoded_eot());
    }

    #[test]
    fn min_tokens_boundary_token_is_the_first_that_may_stop() {
        let tok = test_tokenizer();
        let a = tok.encode("a")[0];
        // min_tokens = 2: tokens 1 and 2 are ordinary, the 3rd may stop.
        let mut emit = min_tokens_emit(&tok, 2, Vec::new());
        let (_, stop) = drive_ids(&mut emit, &[a, a, 1]);
        assert_eq!(stop, Some((2, StopReason::Eos)));
        // An EOS as the 2nd token does not stop.
        let mut emit = min_tokens_emit(&tok, 2, Vec::new());
        let (_, stop) = drive_ids(&mut emit, &[a, 1]);
        assert_eq!(stop, None);
        // min_tokens = 1: the first token cannot stop, the second can.
        let mut emit = min_tokens_emit(&tok, 1, Vec::new());
        let (_, stop) = drive_ids(&mut emit, &[1, 1]);
        assert_eq!(stop, Some((1, StopReason::Eos)));
    }

    #[test]
    fn min_tokens_unreached_ends_as_length_not_decoded_eot() {
        let tok = test_tokenizer();
        let a = tok.encode("a")[0];
        let mut emit = min_tokens_emit(&tok, 8, Vec::new());
        // The cap arrives (the loop stops feeding) after a suppressed EOS.
        let (_, stop) = drive_ids(&mut emit, &[a, 1]);
        assert_eq!(stop, None);
        assert!(
            !emit.decoded_eot(),
            "an EOS below min_tokens is not a decoded end of turn"
        );
    }

    #[test]
    fn min_tokens_zero_stops_on_the_first_token_like_before() {
        let tok = test_tokenizer();
        let mut emit = min_tokens_emit(&tok, 0, Vec::new());
        let (_, stop) = drive_ids(&mut emit, &[1]);
        assert_eq!(stop, Some((0, StopReason::Eos)));
    }

    #[test]
    fn min_tokens_holds_stop_sequences_inactive_until_reached() {
        let tok = test_tokenizer();
        let z = tok.encode("Z")[0];
        // `Z` is the stop: the first two are ordinary text, the 3rd stops.
        let mut emit = min_tokens_emit(&tok, 2, vec!["Z".to_string()]);
        let (stream, stop) = drive_ids(&mut emit, &[z, z, z]);
        assert_eq!(stop, Some((2, StopReason::StopSequence)));
        assert_eq!(tokens_text(&stream), "ZZ");
        // Reached before the stop completes: matching resumes mid-stream.
        let mut emit = min_tokens_emit(&tok, 1, vec!["ZZ".to_string()]);
        let (stream, stop) = drive_ids(&mut emit, &[z, z, z]);
        assert_eq!(stop, Some((2, StopReason::StopSequence)));
        assert_eq!(tokens_text(&stream), "Z", "text after resume is trimmed");
    }

    #[test]
    fn raw_committed_once_per_token_including_markers() {
        let body = format!("x{}", hermes_call("t", r#"{}"#));
        let (stream, _, raw) = drive_text(&body);
        let committed: Vec<_> = stream
            .iter()
            .filter_map(|ev| match ev {
                ClientEvent::Committed { id, idx } => Some((*id, *idx)),
                _ => None,
            })
            .collect();
        assert_eq!(committed.len(), raw.len());
        for (i, (id, idx)) in committed.iter().enumerate() {
            assert_eq!(*idx, i);
            assert_eq!(*id, raw[i]);
        }
    }

    #[test]
    fn finish_does_not_use_whole_output_extraction_authority() {
        let open = "<tool_call>{\"name\": \"recovered\", \"arguments\": {}}";
        let (_stream, finish, _) = drive_text(open);
        assert_eq!(finish.finish_reason, "truncated_tool_call");
        assert_eq!(finish.tool_calls, 0);
        assert!(held_calls(&finish).is_empty());
    }

    #[test]
    fn begin_first_token_stop_sequence_terminates_before_observe() {
        // A stop string completed by the first generated token must surface
        // StopSequence from begin so generate_spec never enters spec.step.
        let tok = test_tokenizer();
        let ids = tok.encode("STOP");
        assert!(!ids.is_empty());
        let first = ids[0];
        let first_text = tok.decode(&[first]);
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            enable_grammar: false,
            stop: vec![first_text.clone()],
            min_tokens: 0,
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let outcome = emit.begin(first);
        assert_eq!(
            outcome.stop,
            Some(StopReason::StopSequence),
            "first token completing stop {first_text:?} must stop in begin"
        );
        assert!(
            outcome
                .events
                .iter()
                .any(|e| matches!(e, ClientEvent::Committed { id, .. } if *id == first)),
            "stop still commits the raw first token"
        );
        // Later-token path unchanged: multi-token stop still matches on observe.
        let multi = tok.encode("abSTOP");
        assert!(multi.len() > 1, "need multi-token path for observe parity");
        let mut emit2 = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            enable_grammar: false,
            stop: vec!["STOP".to_string()],
            min_tokens: 0,
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let mut saw_stop = false;
        let mut first = true;
        let mut n = 0usize;
        for id in &multi {
            let outcome = if first {
                first = false;
                emit2.begin(*id)
            } else {
                emit2.observe(*id)
            };
            n += 1;
            if outcome.stop == Some(StopReason::StopSequence) {
                saw_stop = true;
                break;
            }
            assert!(
                outcome.stop.is_none(),
                "unexpected early stop: {:?}",
                outcome.stop
            );
        }
        assert!(saw_stop, "multi-token stop must still fire via observe");
        assert!(n > 1, "multi-token stop must not only hit begin");
    }

    #[test]
    fn stop_sequence_text_is_never_a_token_event() {
        let tok = test_tokenizer();
        let ids = tok.encode("ab STOP cd");
        let mut emit = Qwen35Emit::from_ctx(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            enable_grammar: false,
            stop: vec!["STOP".to_string()],
            min_tokens: 0,
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::Plain,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        });
        let mut tokens = String::new();
        let mut saw_stop = false;
        for (i, id) in ids.iter().enumerate() {
            let outcome = if i == 0 {
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            for e in &outcome.events {
                if let ClientEvent::Token(t) = e {
                    tokens.push_str(t);
                }
            }
            if outcome.stop == Some(StopReason::StopSequence) {
                saw_stop = true;
                break;
            }
        }
        assert!(saw_stop);
        let fin = emit.finish();
        for e in &fin.events {
            if let ClientEvent::Token(t) = e {
                tokens.push_str(t);
            }
        }
        assert_eq!(tokens, "ab ");
        assert_eq!(fin.finish_reason, "stop");
        assert_eq!(fin.visible_text, "ab ");
    }

    #[test]
    fn filter_config_matches_ar_semantic_v2() {
        let cfg = hipfire_runtime::eos_filter::qwen35_eos_filter_config();
        assert!(!cfg.strip_think);
        assert!(!cfg.started_in_think);
        assert!(cfg.stop_at.iter().any(|s| s == b"<|im_end|>"));
        assert!(cfg.stop_at.iter().any(|s| s == b"<|endoftext|>"));
    }

    /// A Qwen4-framed emitter (template think close) with a think budget.
    fn budget_emit<'a>(
        tok: &'a Tokenizer,
        max_think: usize,
        assistant_prefix: AssistantPrefix,
    ) -> Box<dyn SpecEmit + 'a> {
        Qwen35Emit::from_ctx_template_think_close(SpecEmitCtx {
            tokenizer: tok,
            eos: 9,
            im_end: Some(1),
            tools: None,
            enable_grammar: false,
            stop: Vec::new(),
            min_tokens: 0,
            max_think,
            max_tokens: 256,
            assistant_prefix,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::NonThink,
            decoded_vocab: None,
        })
    }

    /// Thinking off reaches the emitter as `max_think_tokens == 1` behind a
    /// closed-think prefix. `window_cap` is the per-window emit clamp of
    /// native Qwen4 MTP, so the answer must never be capped: a cap of 1 turns
    /// every window into one token (τ 0).
    #[test]
    fn thinking_off_never_caps_mtp_windows() {
        let tok = test_tokenizer();
        let mut emit = budget_emit(&tok, 1, AssistantPrefix::ClosedThink);
        assert_eq!(emit.window_cap(), None);
        let ids = tok.encode("Hello world, a plain answer with no thinking.");
        for (i, id) in ids.iter().enumerate() {
            let outcome = if i == 0 {
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            assert!(outcome.stop.is_none(), "token {i}");
            assert_eq!(emit.window_cap(), None, "token {i}");
        }
        assert!(emit.take_forced().is_empty());
    }

    /// An explicit budget still caps windows inside the span, so the
    /// force-close lands on a window's last token, and over the queued close
    /// tokens; the answer after the close is uncapped.
    #[test]
    fn think_budget_caps_windows_in_span_and_over_close() {
        const MAX: usize = 4;
        let tok = test_tokenizer();
        let mut emit = budget_emit(&tok, MAX, AssistantPrefix::OpenThink);
        assert_eq!(emit.window_cap(), Some(MAX));
        let ids = tok.encode("abcd");
        assert_eq!(ids.len(), MAX);
        for (i, id) in ids.iter().enumerate() {
            let outcome = if i == 0 {
                emit.begin(*id)
            } else {
                emit.observe(*id)
            };
            assert!(outcome.stop.is_none(), "token {i}");
            if i + 1 < MAX {
                assert_eq!(emit.window_cap(), Some(MAX - 1 - i), "token {i}");
            }
        }
        // The MAX-th thinking token forced the close.
        let forced = emit.take_forced();
        assert!(!forced.is_empty());
        assert_eq!(emit.window_cap(), Some(forced.len() + 1));
        for (j, id) in forced.iter().enumerate() {
            assert!(emit.observe(*id).stop.is_none(), "close token {j}");
            let left = forced.len() - 1 - j;
            let want = (left > 0).then_some(left + 1);
            assert_eq!(emit.window_cap(), want, "close token {j}");
        }
        for id in tok.encode("The answer.") {
            assert!(emit.observe(id).stop.is_none());
            assert_eq!(emit.window_cap(), None);
        }
    }

    /// #1053 state repair: the daemon feeds "\n" + think_close through the same router the generated
    /// stream uses. The pre-close newline is reasoning, the close marker is consumed, the trailing
    /// newlines are stripped from the answer, and no `Committed` event (nor grammar/terminator
    /// state) is touched by the injected ids.
    #[test]
    fn injected_think_close_repairs_the_span_without_client_trace() {
        let tok = test_tokenizer();
        let mut emit = Qwen35Emit::from_ctx_recover(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: Some(&[]),
            enable_grammar: true,
            stop: Vec::new(),
            min_tokens: 0,
            max_think: 0,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::OpenThink,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::Low,
            decoded_vocab: None,
        });
        // The prompt opened the span: reasoning is live and the boundary is clean.
        assert!(emit.open_think_now());
        assert_eq!(emit.think_close_text(), Some("</think>\n\n"));
        let mut first = true;
        let mut stream = Vec::new();
        for id in tok.encode("plan") {
            let out = if first {
                first = false;
                emit.begin(id)
            } else {
                emit.observe(id)
            };
            stream.extend(out.events);
        }
        assert!(!reasoning_text(&stream).is_empty());
        assert!(emit.open_think_now(), "still inside the span");
        assert!(emit.at_clean_boundary());

        // The injection, byte for byte what `slots.rs` encodes.
        let mut injected = Vec::new();
        for id in tok.encode("\n</think>\n\n") {
            injected.extend(emit.observe_injected(id));
        }
        assert!(
            reasoning_text(&injected).contains('\n'),
            "pre-close newline is reasoning"
        );
        assert!(
            !emit.open_think_now(),
            "the close token flipped the span state"
        );
        assert!(
            injected
                .iter()
                .all(|ev| !matches!(ev, ClientEvent::Committed { .. })),
            "injected tokens never post Committed"
        );
        assert!(
            tokens_text(&injected).trim().is_empty(),
            "the post-close newlines are stripped from the answer"
        );

        // The continuation pass resumes as visible content.
        let mut visible = String::new();
        for id in tok.encode("Hello") {
            visible.push_str(&tokens_text(&emit.observe(id).events));
        }
        assert_eq!(visible, "Hello");
    }

    /// `max_think_tokens` on the emitter (the multi-slot budget close): the
    /// trip token queues the family close text, the daemon-routed splice
    /// discharges the queued count, and a span re-opened afterwards spends
    /// the budget once more before it is a hard `ThinkCap` stop.
    #[test]
    fn think_budget_queues_the_family_close_and_stops_on_reopen() {
        let tok = test_tokenizer();
        let mut emit = Qwen35Emit::from_ctx_recover(SpecEmitCtx {
            tokenizer: &tok,
            eos: 9,
            im_end: Some(1),
            tools: Some(&[]),
            enable_grammar: true,
            stop: Vec::new(),
            min_tokens: 0,
            max_think: 2,
            max_tokens: 256,
            assistant_prefix: AssistantPrefix::OpenThink,
            think_mode: hipfire_runtime::prompt_frame::ThinkMode::Low,
            decoded_vocab: None,
        });
        // Two reasoning tokens spend the budget; the second one trips.
        let mut forced = Vec::new();
        for id in tok.encode("ab") {
            let out = emit.observe(id);
            assert!(out.stop.is_none(), "inside the span: no hard stop yet");
            forced.extend(emit.take_forced());
        }
        assert_eq!(forced, tok.encode("</think>\n\n"));
        assert!(
            emit.take_forced().is_empty(),
            "forced payload is taken once"
        );
        // The daemon routes those tokens as an in-turn splice.
        for id in &forced {
            emit.observe_injected(*id);
        }
        assert!(!emit.open_think_now());
        assert!(
            emit.take_forced().is_empty(),
            "the splice never re-queues a close"
        );
        // The continuation streams as visible content...
        let mut visible = String::new();
        for id in tok.encode("OK") {
            visible.push_str(&tokens_text(&emit.observe(id).events));
        }
        assert_eq!(visible, "OK");
        // ...and a re-opened span spends the same budget once more; a second
        // exhaustion is the hard stop instead of another splice.
        let reopen = tok.encode("<think>");
        assert_eq!(reopen.len(), 1);
        assert!(emit.observe(reopen[0]).stop.is_none());
        let out = emit.observe(tok.encode("x")[0]);
        assert_eq!(out.stop, Some(StopReason::ThinkCap));
        assert!(
            emit.take_forced().is_empty(),
            "a second exhaustion never splices again"
        );
    }
}
