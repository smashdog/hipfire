//! PolyStrata's think-recovery semantics as pure logic, ported for the hipfire multi-slot serve path.
//!
//! Three mechanisms, each a knob in `hipfire-think-recovery-design.md` §4:
//! - #1053 close-retry: a reply that ended on a natural stop with its thinking still open gets one injected
//!   `\n</think>\n\n` and continues (`injection_text`, `close_retry_skip_reason`).
//! - #728 loop recovery: reasoning that repeats whole passages is detected every 512 output tokens at a clean
//!   boundary (`reasoning_repeat_coverage`, `LoopCheck`, `loop_skip_reason`) and either ends the reply as "length"
//!   (mode stop) or resumes it with the low-effort sentence spliced into the first system message
//!   (`focused_recovery_prompt`).
//! - #606 repeat guard: one token 256 times in a row ends the reply (`RepeatRunGuard`).
//!
//! Reference: /mnt/ai/llm/PolyStrata/serve/server.py — `REASONING_CLOSE`/`closed_with` (:167-175),
//! `reasoning_repeat_coverage` (:188-212), `focused_recovery_prompt` (:215-229), the #606 run guard (:4975-4979),
//! the #728 detect gate (:4999-5011) and the recovery action (:5072-5091). No engine, tokenizer or I/O here; S2
//! wires these decisions into `slots.rs` and the runtime's `SpecEmit` state.

// Loop detection primitives live in `hipfire-runtime` so the sequential AR decode
// loop (`hipfire-generate`) shares them without depending on this crate; re-exported
// here under the names the multi-slot gates and tests already use.
pub use hipfire_runtime::reasoning_loop::{
    reasoning_repeat_coverage, LoopCheck, LoopMode, HIGH_EFFORT_INSTRUCTION, LOOP_CHECK_EVERY,
    LOOP_COVERAGE, LOOP_HISTORY_WORDS, LOOP_TAIL_CHARS, LOOP_TAIL_PERIOD, LOOP_WIDTH,
    LOOP_WINDOW_WORDS, LOW_EFFORT_INSTRUCTION,
};
/// One token repeated this many times in a row ends the reply as "length" (`REPEAT_STOP_TOKENS`).
pub const REPEAT_STOP_TOKENS: usize = 256;
/// From this `max_tokens` on, a thinking reply without an explicit budget is closed early enough to leave the
/// answer its room (`ANSWER_ROOM_MIN_CAP`, #1053's answer room).
pub const ANSWER_ROOM_MIN_CAP: usize = 1024;
/// The answer never gets less than this many tokens of the cap (`ANSWER_RESERVE_MIN`).
pub const ANSWER_RESERVE_MIN: usize = 512;
/// Margin the reference keeps between prompt + answer and the context (`CTX_SLACK`).
pub const CTX_SLACK: usize = 8;

/// The wrap-up sentence a budget-capped thinking span ends with (`REASONING_WRAP_UP` minus its close).
pub const WRAP_UP_SENTENCE: &str =
    "\n\nI have thought about this long enough; time to give my answer.";

/// The chat template's turn-end token, which closes the first (system) message in a rendered prompt.
pub const IM_END: &str = "<|im_end|>";
/// The chat template's turn start, which a system message's rendering begins with.
pub const SYSTEM_PREFIX: &str = "<|im_start|>system\n";

/// `closed_with`: the reference injects `REASONING_CLOSE = "\n</think>\n\n"` and, for a template whose reasoning
/// ends some other way, the same line but ending in that model's close tag (Gemma `<channel|>`). hipfire's emitter
/// already carries the model's close text (`HIPFIRE_THINK_CONTINUATION`, default `</think>\n\n`), so prefixing one
/// newline reproduces both: `"\n" + "</think>\n\n"` is the reference string verbatim, `"\n" + "<channel|>"` its
/// Gemma form.
pub fn injection_text(think_close: &str) -> String {
    let mut s = String::with_capacity(1 + think_close.len());
    s.push('\n');
    s.push_str(think_close);
    s
}

/// `closed_with(REASONING_WRAP_UP, think_end)`: the reference's wrap-up is `REASONING_WRAP_UP =
/// "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"` with the model's own close
/// tag substituted. `injection_text` already carries that substitution, so sentence + it is exact for Qwen and
/// the Gemma form for `<channel|>`.
pub fn wrap_up_text(think_close: &str) -> String {
    format!("{WRAP_UP_SENTENCE}{}", injection_text(think_close))
}

/// Why the answer room moved the thinking budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnswerRoomWhy {
    /// No budget was asked for and `max_tokens >= ANSWER_ROOM_MIN_CAP`.
    Derived,
    /// No budget was asked for but the cap is whatever the context has left.
    ByContext,
    /// An explicit budget would have spent the answer's room (#984).
    Clamped,
}

/// The answer room (`answer_room`): `max_new` tokens count the thinking, so a long-thinking reply could otherwise
/// be cut off inside the span with no answer. Mirrors the reference exactly:
///
/// * reserve = `min(max(ANSWER_RESERVE_MIN, max_new / 4), max_new / 2)`;
/// * with no explicit budget, derive one at `max_new - reserve` when `max_new >= ANSWER_ROOM_MIN_CAP` (or the cap
///   is all the context has left, `max_new >= cap - CTX_SLACK - prompt_len`);
/// * with an explicit budget above `max_new - reserve`, clamp it down to that (#984).
///
/// Returns `None` when the request is left as it was (explicit budget that already leaves room, or a cap below
/// the minimum that is not context-bound).
pub fn answer_room_budget(
    max_new: usize,
    cap: usize,
    prompt_len: usize,
    explicit: Option<usize>,
) -> Option<(usize, AnswerRoomWhy)> {
    if max_new == 0 {
        return None;
    }
    let reserve = ANSWER_RESERVE_MIN.max(max_new / 4).min(max_new / 2);
    // The engine's `think_budget` uses 1 as its never-enter-the-span sentinel, so a derived budget below 2 would
    // change meaning; such degenerate caps are left alone (they cannot fit a wrap-up either).
    let usable = |budget: usize| (budget >= 2).then_some(budget);
    match explicit {
        None => {
            let by_context = cap > 0 && max_new >= cap.saturating_sub(CTX_SLACK + prompt_len);
            if max_new >= ANSWER_ROOM_MIN_CAP {
                usable((max_new - reserve).max(1)).map(|b| (b, AnswerRoomWhy::Derived))
            } else if by_context {
                usable((max_new - reserve).max(1)).map(|b| (b, AnswerRoomWhy::ByContext))
            } else {
                None
            }
        }
        Some(budget) => {
            let room = max_new.saturating_sub(reserve).max(1);
            (budget > room).then(|| usable(room)).flatten().map(|b| (b, AnswerRoomWhy::Clamped))
        }
    }
}

/// #606: the current run of one repeated token. `observe` returns true when this token makes the run reach the
/// limit (the reference counts after appending the token and breaks at `run_len >= limit`).
#[derive(Clone, Debug)]
pub struct RepeatRunGuard {
    limit: usize,
    last: Option<u32>,
    run: usize,
}

impl RepeatRunGuard {
    pub fn new(limit: usize) -> Self {
        Self { limit, last: None, run: 0 }
    }

    pub fn observe(&mut self, token: u32) -> bool {
        self.run = match self.last {
            Some(t) if t == token => self.run + 1,
            _ => 1,
        };
        self.last = Some(token);
        self.limit > 0 && self.run >= self.limit
    }

    pub fn run(&self) -> usize {
        self.run
    }

    pub fn limit(&self) -> usize {
        self.limit
    }
}

/// #1053's trigger, as skip reasons in trace order. `None` means inject. The reference's other guards map as:
/// `not wrap` and `stops.hit is None` hold structurally here (no budget wrap; the multi-slot stop list is always
/// empty); `parser.buf != parser.think_end` (a held `</think>`) cannot happen because the emitter consumes a
/// complete close tag immediately; `not opens` is #537's forced-call opening, which has no hipfire equivalent.
#[derive(Clone, Debug)]
pub struct CloseRetryInputs {
    pub enabled: bool,
    pub thinking: bool,
    /// DoneReason was a natural stop token (Eos), not max-tokens/client-gone/rejected.
    pub natural_stop: bool,
    /// Any content reached the client, sticky (reasoning alone does not count).
    pub answered: bool,
    pub cancelled: bool,
    /// The emitter finished with its thinking still open.
    pub open_think: bool,
    pub already_injected: bool,
    /// `max_tokens - n`: output budget left before this pass ends.
    pub budget_left: usize,
    /// Tokens the injection (plus following text) will add on top of the current output.
    pub extra_len: usize,
}

pub fn close_retry_skip_reason(i: &CloseRetryInputs) -> Option<&'static str> {
    if i.cancelled {
        return Some("cancelled");
    }
    if !i.enabled {
        return Some("disabled");
    }
    if !i.thinking {
        return Some("no thinking");
    }
    if !i.natural_stop {
        return Some("not a natural stop");
    }
    if i.already_injected {
        return Some("already retried");
    }
    if i.answered {
        return Some("already answered");
    }
    if !i.open_think {
        return Some("thinking closed");
    }
    // The reference needs `max_new - n - len(extra) >= 1` and gives up silently otherwise.
    if i.budget_left <= i.extra_len {
        return Some("no headroom");
    }
    None
}

/// #728's decision once a look ran at a clean boundary with its coverage. `None` means act (stop or recover);
/// otherwise the reason the look is a no-op and generation continues.
#[derive(Clone, Debug)]
pub struct LoopDecisionInputs {
    pub mode: LoopMode,
    pub coverage: f64,
    pub already_recovered: bool,
    pub has_images: bool,
    /// Output tokens so far (n).
    pub n: usize,
    /// The request's max_tokens for this pass.
    pub max_new: usize,
}

pub fn loop_skip_reason(i: &LoopDecisionInputs) -> Option<&'static str> {
    if i.mode == LoopMode::Off {
        return Some("off");
    }
    if i.coverage < LOOP_COVERAGE {
        return Some("below threshold");
    }
    if i.mode == LoopMode::Stop {
        return None;
    }
    if i.already_recovered {
        return Some("already recovered");
    }
    if i.has_images {
        return Some("vision");
    }
    if i.n >= i.max_new {
        return Some("budget exhausted");
    }
    None
}

/// The id splice of `focused_recovery_prompt` (server.py:215-229): the prompt's ids with the xhigh effort sentence
/// of the first system message replaced by the low one, then every token generated so far. A splice of token ids —
/// the prompt is never decoded and encoded again, so a literal `</think>` or vision marker in it stays what it was.
/// `None` when the sentence is not there as whole tokens in the first system message.
///
/// `im_end` is the turn-end id; `head_is_system` is the caller's `decode(head).starts_with("<|im_start|>system\n")`
/// (the reference decodes once; a pure core takes the answer as a bool so it needs no tokenizer).
pub fn splice_effort_ids(
    ids: &[u32],
    im_end: u32,
    head_is_system: bool,
    old: &[u32],
    new: &[u32],
    generated: &[u32],
) -> Option<Vec<u32>> {
    if old.is_empty() {
        return None;
    }
    let end_pos = ids.iter().position(|&t| t == im_end)?;
    if !head_is_system {
        return None;
    }
    let head = &ids[..end_pos];
    let at = head.windows(old.len()).position(|w| w == old)?;
    let mut out = Vec::with_capacity(ids.len() + new.len() + generated.len());
    out.extend_from_slice(&ids[..at]);
    out.extend_from_slice(new);
    out.extend_from_slice(&ids[at + old.len()..]);
    out.extend_from_slice(generated);
    Some(out)
}

/// `focused_recovery_prompt` with the runtime tokenizer: the effort sentences are encoded without special-token
/// treatment (hipfire's `encode` only rewrites strings that spell a known special token and prepends BOS only for
/// models that ask for one; Qwen-family carriers do not).
pub fn focused_recovery_prompt(
    tok: &hipfire_runtime::tokenizer::Tokenizer,
    ids: &[u32],
    generated: &[u32],
) -> Option<Vec<u32>> {
    let im_end = tok.special_token_id(IM_END)?;
    let end_pos = ids.iter().position(|&t| t == im_end)?;
    let head_is_system = tok.decode(&ids[..end_pos]).starts_with(SYSTEM_PREFIX);
    let old = tok.encode(HIGH_EFFORT_INSTRUCTION);
    let new = tok.encode(LOW_EFFORT_INSTRUCTION);
    splice_effort_ids(ids, im_end, head_is_system, &old, &new, generated)
}

/// What the per-token gates decided for one sampled token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAction {
    /// Hand the token to the emitter.
    Deliver,
    /// #606: a token run reached its limit — end the pass; the token is not rendered.
    RepeatStop,
    /// #728 (stop): the reasoning reads as a loop — end the pass as "length".
    LoopStop,
    /// #728 (recover): end the pass and restart it with the low-effort sentence.
    Recover,
}

/// Per-token gate result plus the trace line the reference would log.
#[derive(Debug, Clone, PartialEq)]
pub struct GateOutcome {
    pub action: GateAction,
    pub trace: Option<String>,
    /// Loop-check skip reason (`loop_skip_reason`), for `/metrics`. `None`
    /// unless this token ran the periodic check and declined.
    pub skip: Option<&'static str>,
}

/// Everything the #728 gate needs to know about the CURRENT pass and request.
pub struct GateCtx {
    pub loop_mode: LoopMode,
    /// The emitter's think state as of the PREVIOUS token (the reference looks at its parser before
    /// feeding the new token).
    pub open_think: bool,
    /// No partial bytes held in the EOT filter or think router.
    pub clean_boundary: bool,
    pub has_images: bool,
    /// The request's `max_tokens`.
    pub max_new: usize,
}

/// The request-level bookkeeping the gate order reads and writes: sampled-token count (`n`, injections
/// included), the #606 run, the #728 schedule, the rendered reasoning text and the sticky `answered` flag.
/// Mirrors the loop in the reference (`server.py` :4975-5011) — the state machine lives here so a synthetic
/// event stream can drive it in a host test, while `slots.rs` only renders what it returns.
pub struct StreamGates {
    n: usize,
    repeat: RepeatRunGuard,
    loop_next: LoopCheck,
    recovered: bool,
    reasoning: String,
    answered: bool,
}

impl StreamGates {
    pub fn new(repeat_limit: usize) -> Self {
        Self {
            n: 0,
            repeat: RepeatRunGuard::new(repeat_limit),
            loop_next: LoopCheck::new(),
            recovered: false,
            reasoning: String::new(),
            answered: false,
        }
    }

    /// `n`: every sampled token of the request, rendered or not.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Sticky: some content or tool call reached the client (reasoning does not count).
    pub fn answered(&self) -> bool {
        self.answered
    }

    pub fn recovered(&self) -> bool {
        self.recovered
    }

    pub fn mark_recovered(&mut self) {
        self.recovered = true;
    }

    /// #1053: injected tokens count toward `n` like sampled ones.
    pub fn note_injected(&mut self, count: usize) {
        self.n += count;
    }

    /// A natural stop sampled a terminator: it joins the sampled stream at +1 (`raw_ids`).
    pub fn note_eos(&mut self) {
        self.n += 1;
    }

    /// Reasoning text as streamed to the client — the loop coverage reads only this channel.
    pub fn push_reasoning(&mut self, text: &str) {
        self.reasoning.push_str(text);
    }

    pub fn mark_answered(&mut self) {
        self.answered = true;
    }

    /// A token arriving after a gate already ended the pass: it still counts toward `n` and the repeat
    /// run (the reference keeps appending before it notices its break), but no gate looks at it again.
    pub fn note_dropped(&mut self, tok: u32) {
        self.n += 1;
        let _ = self.repeat.observe(tok);
    }

    /// #606 then #728, in the reference's order. `recover_splice` is only consulted when a recover
    /// decision is reached; it reports whether the effort sentence spliced into `gen_ids` still
    /// yields a prompt (the second splice reuses the tokens sampled past the decision point).
    pub fn note_token(
        &mut self,
        tok: u32,
        ctx: &GateCtx,
        recover_splice: &mut dyn FnMut() -> bool,
    ) -> GateOutcome {
        self.n += 1;
        if self.repeat.observe(tok) {
            return GateOutcome {
                action: GateAction::RepeatStop,
                trace: Some(format!(
                    "repeat-stop: run={} token={tok} at n={}; ending as length",
                    self.repeat.run(),
                    self.n
                )),
                skip: None,
            };
        }
        if ctx.loop_mode != LoopMode::Off
            && !self.recovered
            && self.loop_next.due(self.n)
            && ctx.open_think
            && ctx.clean_boundary
        {
            self.loop_next.advance(self.n);
            let coverage = reasoning_repeat_coverage(&self.reasoning);
            let decision = loop_skip_reason(&LoopDecisionInputs {
                mode: ctx.loop_mode,
                coverage,
                already_recovered: self.recovered,
                has_images: ctx.has_images,
                n: self.n,
                max_new: ctx.max_new,
            });
            match decision {
                None => {
                    if ctx.loop_mode == LoopMode::Stop {
                        return GateOutcome {
                            action: GateAction::LoopStop,
                            trace: Some(format!(
                                "loop-stop: coverage {coverage:.3} at n={}; ending as length",
                                self.n
                            )),
                            skip: None,
                        };
                    }
                    if recover_splice() {
                        return GateOutcome {
                            action: GateAction::Recover,
                            trace: Some(format!(
                                "loop-recover: coverage {coverage:.3} at n={}; restarting with the \
                                 low-effort sentence",
                                self.n
                            )),
                            skip: None,
                        };
                    }
                    // No effort sentence to splice: the reference keeps generating.
                }
                Some(reason) if reason != "below threshold" => {
                    return GateOutcome {
                        action: GateAction::Deliver,
                        trace: Some(format!(
                            "loop-check skip at n={}: {reason} (coverage {coverage:.3})",
                            self.n
                        )),
                        skip: Some(reason),
                    };
                }
                Some(_) => {}
            }
        }
        GateOutcome {
            action: GateAction::Deliver,
            trace: None,
            skip: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injection_matches_reference_strings() {
        assert_eq!(injection_text("</think>\n\n"), "\n</think>\n\n");
        assert_eq!(injection_text("<channel|>"), "\n<channel|>");
    }

    #[test]
    fn wrap_up_matches_reference_string() {
        assert_eq!(
            wrap_up_text("</think>\n\n"),
            "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"
        );
        assert_eq!(
            wrap_up_text("<channel|>"),
            "\n\nI have thought about this long enough; time to give my answer.\n<channel|>"
        );
    }

    #[test]
    fn answer_room_derives_reserve_from_the_cap() {
        // 8192-cap agent harness: a quarter kept for the answer.
        assert_eq!(answer_room_budget(8192, 32768, 100, None), Some((6144, AnswerRoomWhy::Derived)));
        // Floor and half-cap bounds.
        assert_eq!(answer_room_budget(1024, 32768, 100, None), Some((512, AnswerRoomWhy::Derived)));
        assert_eq!(answer_room_budget(32768, 131072, 100, None), Some((24576, AnswerRoomWhy::Derived)));
        // Below the minimum cap with context to spare: untouched.
        assert_eq!(answer_room_budget(1000, 32768, 100, None), None);
        // Below the minimum but the cap is all the context has left (#16).
        assert_eq!(
            answer_room_budget(900, 1000, 92, None),
            Some((450, AnswerRoomWhy::ByContext))
        );
        assert_eq!(answer_room_budget(0, 1000, 92, None), None);
    }

    #[test]
    fn answer_room_clamps_an_explicit_budget_that_eats_the_answer() {
        assert_eq!(answer_room_budget(8192, 32768, 100, Some(8192)), Some((6144, AnswerRoomWhy::Clamped)));
        assert_eq!(answer_room_budget(1024, 32768, 100, Some(512)), None);
        assert_eq!(answer_room_budget(8192, 32768, 100, Some(6144)), None);
        // A degenerate cap leaves the sentinel-adjacent budgets alone.
        assert_eq!(answer_room_budget(2, 32768, 0, Some(2)), None);
    }

    #[test]
    fn repeat_run_guard_counts_to_limit() {
        let mut g = RepeatRunGuard::new(REPEAT_STOP_TOKENS);
        for _ in 0..255 {
            assert!(!g.observe(7));
        }
        assert_eq!(g.run(), 255);
        assert!(g.observe(7));
        assert_eq!(g.run(), 256);
        assert!(!g.observe(8));
        assert_eq!(g.run(), 1);
    }

    #[test]
    fn repeat_run_guard_limit_zero_is_off() {
        let mut g = RepeatRunGuard::new(0);
        for _ in 0..1000 {
            assert!(!g.observe(1));
        }
    }

    fn close_inputs() -> CloseRetryInputs {
        CloseRetryInputs {
            enabled: true,
            thinking: true,
            natural_stop: true,
            answered: false,
            cancelled: false,
            open_think: true,
            already_injected: false,
            budget_left: 10,
            extra_len: 3,
        }
    }

    #[test]
    fn close_retry_triggers_only_when_every_guard_passes() {
        assert_eq!(close_retry_skip_reason(&close_inputs()), None);
        // One headroom token is enough.
        let mut i = close_inputs();
        i.budget_left = i.extra_len + 1;
        assert_eq!(close_retry_skip_reason(&i), None);
        let mut i = close_inputs();
        i.budget_left = i.extra_len;
        assert_eq!(close_retry_skip_reason(&i), Some("no headroom"));
    }

    #[test]
    fn close_retry_reports_first_failing_guard() {
        let cases: [(fn(&mut CloseRetryInputs), &'static str); 8] = [
            (|i| i.cancelled = true, "cancelled"),
            (|i| i.enabled = false, "disabled"),
            (|i| i.thinking = false, "no thinking"),
            (|i| i.natural_stop = false, "not a natural stop"),
            (|i| i.already_injected = true, "already retried"),
            (|i| i.answered = true, "already answered"),
            (|i| i.open_think = false, "thinking closed"),
            (|i| i.budget_left = 0, "no headroom"),
        ];
        for (set, want) in cases {
            let mut i = close_inputs();
            set(&mut i);
            assert_eq!(close_retry_skip_reason(&i), Some(want));
        }
    }

    fn loop_inputs() -> LoopDecisionInputs {
        LoopDecisionInputs {
            mode: LoopMode::Recover,
            coverage: 0.5,
            already_recovered: false,
            has_images: false,
            n: 1000,
            max_new: 2000,
        }
    }

    #[test]
    fn loop_decision_matrix() {
        assert_eq!(loop_skip_reason(&loop_inputs()), None);
        let mut i = loop_inputs();
        i.mode = LoopMode::Off;
        assert_eq!(loop_skip_reason(&i), Some("off"));
        let mut i = loop_inputs();
        i.coverage = 0.249;
        assert_eq!(loop_skip_reason(&i), Some("below threshold"));
        let mut i = loop_inputs();
        i.mode = LoopMode::Stop;
        i.already_recovered = true;
        i.has_images = true;
        assert_eq!(loop_skip_reason(&i), None);
        let mut i = loop_inputs();
        i.already_recovered = true;
        assert_eq!(loop_skip_reason(&i), Some("already recovered"));
        let mut i = loop_inputs();
        i.has_images = true;
        assert_eq!(loop_skip_reason(&i), Some("vision"));
        let mut i = loop_inputs();
        i.n = i.max_new;
        assert_eq!(loop_skip_reason(&i), Some("budget exhausted"));
    }

    #[test]
    fn splice_replaces_whole_sentence_tokens() {
        let ids = [1u32, 2, 10, 11, 12, 99, 5, 6];
        let out = splice_effort_ids(&ids, 99, true, &[10, 11, 12], &[20, 21], &[30]).unwrap();
        assert_eq!(out, vec![1, 2, 20, 21, 99, 5, 6, 30]);
    }

    #[test]
    fn splice_declines_without_system_head_or_whole_tokens() {
        let ids = [1u32, 2, 10, 11, 12, 99];
        assert!(splice_effort_ids(&ids, 98, true, &[10, 11], &[20], &[]).is_none());
        assert!(splice_effort_ids(&ids, 99, false, &[10, 11], &[20], &[]).is_none());
        assert!(splice_effort_ids(&ids, 99, true, &[11, 10], &[20], &[]).is_none());
        assert!(splice_effort_ids(&ids, 99, true, &[], &[20], &[]).is_none());
    }

    #[test]
    fn effort_sentences_match_the_reference_text() {
        assert!(HIGH_EFFORT_INSTRUCTION.starts_with("Reasoning effort is set to xhigh. Please think carefully"));
        assert!(HIGH_EFFORT_INSTRUCTION.ends_with("clarity in the final answer."));
        assert!(LOW_EFFORT_INSTRUCTION.starts_with("Reasoning effort is set to low. Keep your thinking brief"));
        assert!(LOW_EFFORT_INSTRUCTION.ends_with("without unnecessary elaboration."));
    }

    // ── StreamGates: the reference's per-token gate order on a synthetic stream ──

    fn gate_ctx(mode: LoopMode) -> GateCtx {
        GateCtx {
            loop_mode: mode,
            open_think: true,
            clean_boundary: true,
            has_images: false,
            max_new: 100_000,
        }
    }

    /// Exact 2048-char tail with period 8: the reference's tail check reads it as a full loop (1.0).
    fn periodic_reasoning() -> String {
        "abcdefgh".repeat(LOOP_TAIL_CHARS / 8)
    }

    fn feed(gates: &mut StreamGates, ctx: &GateCtx, upto_n: usize, mut splice: impl FnMut() -> bool) -> GateOutcome {
        let mut last = GateOutcome {
            action: GateAction::Deliver,
            trace: None,
            skip: None,
        };
        while gates.n() < upto_n {
            last = gates.note_token((gates.n() % 7) as u32, ctx, &mut splice);
        }
        last
    }

    #[test]
    fn gates_repeat_run_stops_at_256_and_counts_the_token() {
        let ctx = gate_ctx(LoopMode::Off);
        let mut gates = StreamGates::new(REPEAT_STOP_TOKENS);
        for _ in 0..255 {
            let out = gates.note_token(42, &ctx, &mut || true);
            assert_eq!(out.action, GateAction::Deliver);
        }
        let out = gates.note_token(42, &ctx, &mut || true);
        assert_eq!(out.action, GateAction::RepeatStop);
        assert_eq!(gates.n(), 256);
        assert!(out.trace.unwrap().contains("run=256"));
    }

    #[test]
    fn gates_loop_stop_fires_at_512_on_repeated_reasoning() {
        let ctx = gate_ctx(LoopMode::Stop);
        let mut gates = StreamGates::new(REPEAT_STOP_TOKENS);
        gates.push_reasoning(&periodic_reasoning());
        let first = feed(&mut gates, &ctx, 511, || true);
        assert_eq!(first.action, GateAction::Deliver);
        let out = gates.note_token(0, &ctx, &mut || true);
        assert_eq!(out.action, GateAction::LoopStop);
        assert_eq!(gates.n(), 512);
        assert!(out.trace.unwrap().contains("coverage 1.000"));
    }

    #[test]
    fn gates_loop_check_waits_for_a_clean_think_boundary_but_keeps_the_schedule() {
        let mut ctx = gate_ctx(LoopMode::Stop);
        let mut gates = StreamGates::new(REPEAT_STOP_TOKENS);
        gates.push_reasoning(&periodic_reasoning());
        // Due at 512 but not clean: keep generating; the look stays pending.
        ctx.clean_boundary = false;
        let out = feed(&mut gates, &ctx, 512, || true);
        assert_eq!(out.action, GateAction::Deliver);
        // The first clean token afterwards takes the look immediately.
        ctx.clean_boundary = true;
        let out = gates.note_token(0, &ctx, &mut || true);
        assert_eq!(out.action, GateAction::LoopStop);
        assert_eq!(gates.n(), 513);
    }

    #[test]
    fn gates_loop_check_requires_open_think_and_off_mode_never_looks() {
        let mut ctx = gate_ctx(LoopMode::Stop);
        let mut gates = StreamGates::new(REPEAT_STOP_TOKENS);
        gates.push_reasoning(&periodic_reasoning());
        ctx.open_think = false;
        let out = feed(&mut gates, &ctx, 512, || true);
        assert_eq!(out.action, GateAction::Deliver);
        // Mode off: no look even with everything open.
        let ctx_off = gate_ctx(LoopMode::Off);
        let mut gates = StreamGates::new(REPEAT_STOP_TOKENS);
        gates.push_reasoning(&periodic_reasoning());
        let out = feed(&mut gates, &ctx_off, 512, || true);
        assert_eq!(out.action, GateAction::Deliver);
    }

    #[test]
    fn gates_recover_needs_a_splice_and_stops_after_one() {
        let ctx = gate_ctx(LoopMode::Recover);
        let mut gates = StreamGates::new(REPEAT_STOP_TOKENS);
        gates.push_reasoning(&periodic_reasoning());
        // No effort sentence to splice: keep generating to the next check.
        let out = feed(&mut gates, &ctx, 512, || false);
        assert_eq!(out.action, GateAction::Deliver);
        let mid = feed(&mut gates, &ctx, 1024, || false);
        assert_eq!(mid.action, GateAction::Deliver);
        // Splice available at the next check: recover once.
        let mut gates = StreamGates::new(REPEAT_STOP_TOKENS);
        gates.push_reasoning(&periodic_reasoning());
        let out = feed(&mut gates, &ctx, 511, || true);
        assert_eq!(out.action, GateAction::Deliver);
        let out = gates.note_token(0, &ctx, &mut || true);
        assert_eq!(out.action, GateAction::Recover);
        gates.mark_recovered();
        assert!(gates.recovered());
        // A recovered request never looks again.
        let out = feed(&mut gates, &ctx, 1200, || true);
        assert_eq!(out.action, GateAction::Deliver);
    }

    #[test]
    fn gates_dropped_tokens_count_but_do_not_re_decide() {
        let mut gates = StreamGates::new(REPEAT_STOP_TOKENS);
        gates.note_dropped(7);
        gates.note_dropped(7);
        assert_eq!(gates.n(), 2);
        // The run guard still sees them (the reference appends before its break).
        for _ in 0..254 {
            gates.note_dropped(7);
        }
        assert_eq!(gates.n(), 256);
    }

    #[test]
    fn gates_injected_and_eos_tokens_count_toward_n() {
        let mut gates = StreamGates::new(REPEAT_STOP_TOKENS);
        gates.note_token(1, &gate_ctx(LoopMode::Off), &mut || true);
        gates.note_injected(3);
        gates.note_eos();
        assert_eq!(gates.n(), 5);
    }
}
