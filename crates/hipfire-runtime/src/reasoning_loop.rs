//! PolyStrata's reasoning-loop detector (#728) as pure logic.
//!
//! Shared by every served path that can recover from a thinking loop:
//! - the multi-slot emitter (`hipfire-daemon`'s `think_recovery::StreamGates`),
//! - the sequential AR decode loop (`hipfire-generate::ar::generate`).
//!
//! Reference: /mnt/ai/llm/PolyStrata/serve/server.py — `reasoning_repeat_coverage`
//! (:188-212), the #728 detect gate (:4999-5011) and the recovery action (:5072-5091).
//! No engine, tokenizer or I/O here: callers own the cadence trigger and the
//! clean-boundary predicate (`n >= next_at`, no held marker prefix, no partial
//! UTF-8).

use std::collections::{HashMap, HashSet};

/// Output tokens between two looks at the reasoning (`LOOP_CHECK_EVERY`).
pub const LOOP_CHECK_EVERY: usize = 512;
/// Share of the recent window inside 12-word passages seen three times that reads as a loop (`LOOP_COVERAGE`).
pub const LOOP_COVERAGE: f64 = 0.25;
/// How far back the passages are counted (`LOOP_HISTORY_WORDS`).
pub const LOOP_HISTORY_WORDS: usize = 30_000;
/// Characters at the end of the reasoning also checked for a short period (`LOOP_TAIL_CHARS`, #1753).
pub const LOOP_TAIL_CHARS: usize = 2048;
/// A period at most this long in that tail reads as a fully repeated window (`LOOP_TAIL_PERIOD`).
pub const LOOP_TAIL_PERIOD: usize = 64;
/// Passage width in words for the coverage count.
pub const LOOP_WIDTH: usize = 12;
/// Recent window in words for the coverage count.
pub const LOOP_WINDOW_WORDS: usize = 2000;

/// The template's own xhigh sentence, which the recovery pass swaps for the low one (`EFFORT_TEXT["xhigh"]`).
pub const HIGH_EFFORT_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the \
task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and \
clarity in the final answer.";
/// The low sentence spliced in by the recovery pass (`EFFORT_TEXT["low"]`).
pub const LOW_EFFORT_INSTRUCTION: &str =
    "Reasoning effort is set to low. Keep your thinking brief and focused, \
moving directly to the conclusion without unnecessary elaboration.";

/// The reasoning-loop knob's three states. The reference has off/True("recover")/"stop"; served defaults to
/// recover, library and test callers default to off (design §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopMode {
    Off,
    Stop,
    Recover,
}

impl LoopMode {
    /// Parses a configured value. Unknown strings return `None` so the caller applies its own default.
    pub fn from_config_value(v: &str) -> Option<LoopMode> {
        match v.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" | "no" => Some(LoopMode::Off),
            "stop" => Some(LoopMode::Stop),
            "recover" | "1" | "true" | "on" | "yes" => Some(LoopMode::Recover),
            _ => None,
        }
    }
}

/// Python's `\w` for a `str` pattern: letters, digits and underscore (plus whatever `is_alphanumeric` covers in
/// the numeric categories Nd/Nl/No). Combining marks are not word characters in either implementation.
fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// `re.findall(r"\w+|[^\w\s]", lower)`: runs of word characters, and every other non-space character alone.
fn split_words(lower: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut run_start: Option<usize> = None;
    for (idx, ch) in lower.char_indices() {
        if is_word_char(ch) {
            run_start.get_or_insert(idx);
            continue;
        }
        if let Some(s) = run_start.take() {
            out.push(&lower[s..idx]);
        }
        if !ch.is_whitespace() {
            out.push(&lower[idx..idx + ch.len_utf8()]);
        }
    }
    if let Some(s) = run_start {
        out.push(&lower[s..]);
    }
    out
}

/// `reasoning_repeat_coverage` (server.py:188-212), verbatim semantics:
/// 1. `tail = text[-2048:].lower()`; exactly 2048 chars and some period `p` in `1..=64` with `tail[p:] == tail[:-p]`
///    reads as 1.0 (a short cycle, e.g. one long number repeating a block).
/// 2. Otherwise words of the lowercased text, last 30,000 of them; fewer than 2,000 words reads as 0.0.
/// 3. Count every 12-word passage; a passage seen at least three times covers the word positions it occupies when
///    its start lies in the last 2,000 words. Coverage = covered positions / (len(words) - start).
pub fn reasoning_repeat_coverage(text: &str) -> f64 {
    let orig: Vec<char> = text.chars().collect();
    if orig.len() >= LOOP_TAIL_CHARS {
        let tail: String = orig[orig.len() - LOOP_TAIL_CHARS..].iter().collect();
        let tail: Vec<char> = tail.to_lowercase().chars().collect();
        if tail.len() == LOOP_TAIL_CHARS
            && (1..=LOOP_TAIL_PERIOD).any(|p| tail[p..] == tail[..LOOP_TAIL_CHARS - p])
        {
            return 1.0;
        }
    }
    let lower = text.to_lowercase();
    let words = split_words(&lower);
    let words = if words.len() > LOOP_HISTORY_WORDS {
        &words[words.len() - LOOP_HISTORY_WORDS..]
    } else {
        &words[..]
    };
    if words.len() < LOOP_WINDOW_WORDS {
        return 0.0;
    }
    // Intern the words to ids so a 12-gram key is a plain array (the reference keys tuples; this is the same
    // equality, without per-gram allocation).
    let mut ids: HashMap<&str, u32> = HashMap::new();
    let mut seq: Vec<u32> = Vec::with_capacity(words.len());
    for w in words {
        let next = ids.len() as u32;
        seq.push(*ids.entry(*w).or_insert(next));
    }
    let mut counts: HashMap<[u32; LOOP_WIDTH], u32> = HashMap::new();
    for i in 0..=seq.len() - LOOP_WIDTH {
        let mut gram = [0u32; LOOP_WIDTH];
        gram.copy_from_slice(&seq[i..i + LOOP_WIDTH]);
        *counts.entry(gram).or_insert(0) += 1;
    }
    let start = seq.len() - LOOP_WINDOW_WORDS;
    let mut covered: HashSet<usize> = HashSet::new();
    for i in start..=seq.len() - LOOP_WIDTH {
        let mut gram = [0u32; LOOP_WIDTH];
        gram.copy_from_slice(&seq[i..i + LOOP_WIDTH]);
        if counts[&gram] >= 3 {
            covered.extend(i..i + LOOP_WIDTH);
        }
    }
    covered.len() as f64 / (seq.len() - start) as f64
}

/// #728's check cadence: a look is due when the request has produced `n >= next_at` output tokens (counted across
/// passes, injections included). The reference re-arms to `n + 512` each time a look runs at a clean boundary.
#[derive(Clone, Debug)]
pub struct LoopCheck {
    next_at: usize,
}

impl LoopCheck {
    pub fn new() -> Self {
        Self {
            next_at: LOOP_CHECK_EVERY,
        }
    }

    pub fn due(&self, n: usize) -> bool {
        n >= self.next_at
    }

    pub fn advance(&mut self, n: usize) {
        self.next_at = n + LOOP_CHECK_EVERY;
    }

    pub fn next_at(&self) -> usize {
        self.next_at
    }
}

impl Default for LoopCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words_text(n: usize, prefix: &str) -> String {
        (0..n)
            .map(|i| format!("{prefix}{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn loop_mode_parsing() {
        assert_eq!(LoopMode::from_config_value("off"), Some(LoopMode::Off));
        assert_eq!(LoopMode::from_config_value("STOP"), Some(LoopMode::Stop));
        assert_eq!(
            LoopMode::from_config_value("recover"),
            Some(LoopMode::Recover)
        );
        assert_eq!(LoopMode::from_config_value("true"), Some(LoopMode::Recover));
        assert_eq!(LoopMode::from_config_value(""), None);
        assert_eq!(LoopMode::from_config_value("sometimes"), None);
    }

    #[test]
    fn coverage_needs_two_thousand_words() {
        assert_eq!(reasoning_repeat_coverage(""), 0.0);
        assert_eq!(reasoning_repeat_coverage(&words_text(1999, "w")), 0.0);
    }

    #[test]
    fn coverage_periodic_tail_reads_full() {
        let mut text = words_text(100, "w");
        text.push(' ');
        // 205 full copies of a 10-char cycle: the last 2048 chars are periodic with p=10.
        text.push_str(&"0123456789".repeat(205));
        assert_eq!(reasoning_repeat_coverage(&text), 1.0);
    }

    #[test]
    fn coverage_repeated_passage_crosses_threshold() {
        // A 200-word passage four times, then 1200 unique words: 2000 words, ~800 covered.
        let passage = words_text(200, "p");
        let mut text = Vec::new();
        for _ in 0..4 {
            text.push(passage.clone());
        }
        text.push(words_text(1200, "u"));
        let cov = reasoning_repeat_coverage(&text.join(" "));
        assert!(cov >= LOOP_COVERAGE, "coverage {cov}");
        assert!(cov < 0.5, "coverage {cov}");
    }

    #[test]
    fn coverage_unique_words_read_zero() {
        assert_eq!(reasoning_repeat_coverage(&words_text(2500, "u")), 0.0);
    }

    #[test]
    fn words_split_like_python() {
        assert_eq!(
            split_words("ab, cd!_x9\n"),
            vec!["ab", ",", "cd", "!", "_x9"]
        );
    }

    #[test]
    fn loop_check_cadence() {
        let mut c = LoopCheck::new();
        assert!(!c.due(LOOP_CHECK_EVERY - 1));
        assert!(c.due(LOOP_CHECK_EVERY));
        c.advance(LOOP_CHECK_EVERY);
        assert_eq!(c.next_at(), LOOP_CHECK_EVERY * 2);
        assert!(!c.due(LOOP_CHECK_EVERY + 1));
    }
}
