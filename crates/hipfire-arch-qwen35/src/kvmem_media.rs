//! G7 multimodal × KVMem — Stage A foundation: vision spans are indivisible
//! prefix groups.
//!
//! Mirrors nInfer `native_layerwise_media.h` (`validate_layerwise_media`,
//! `layerwise_media_replay_begin`) and composes the core
//! `kvmem::prefix_anchor` primitive (base/query clamping + split rejection).
//! Pure host logic; no engine call site yet (additive, byte-neutral).

use hipfire_kvmem::{prefix_anchor, PrefixRange};

/// One indivisible media (vision) span in prompt-token coordinates.
/// `end` is exclusive; `begin < end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MediaSpan {
    pub begin: u64,
    pub end: u64,
}

/// Build spans from `(slot_start, len)` image-pad runs, dropping empty runs.
pub fn spans_from_runs(runs: &[(usize, usize)]) -> Vec<MediaSpan> {
    let mut spans: Vec<MediaSpan> = runs
        .iter()
        .filter(|(_, len)| *len > 0)
        .map(|&(start, len)| MediaSpan {
            begin: start as u64,
            end: (start + len) as u64,
        })
        .collect();
    spans.sort_by_key(|s| s.begin);
    spans
}

/// True when `boundary` would cut a media span (strictly inside it).
pub fn media_split(spans: &[MediaSpan], boundary: u64) -> bool {
    spans.iter().any(|s| s.begin < boundary && boundary < s.end)
}

/// True when any media span is fully inside the prefix (`end <= frontier`).
pub fn media_inside(spans: &[MediaSpan], frontier: u64) -> bool {
    spans.iter().any(|s| s.end <= frontier)
}

/// `image_pad` runs of `tokens` (absolute prompt coordinates), capped at
/// `n_visual` rows — the engine-side mirror of `vl_prefill::image_spans`
/// without its embedding-row bookkeeping.
pub fn spans_from_pad_runs(tokens: &[u32], pad: u32, n_visual: usize) -> Vec<MediaSpan> {
    let mut spans: Vec<MediaSpan> = Vec::new();
    let mut used = 0usize;
    let mut i = 0usize;
    while i < tokens.len() && used < n_visual {
        if tokens[i] != pad {
            i += 1;
            continue;
        }
        let start = i;
        while i < tokens.len() && tokens[i] == pad && used + (i - start) < n_visual {
            i += 1;
        }
        let len = i - start;
        spans.push(MediaSpan {
            begin: start as u64,
            end: (start + len) as u64,
        });
        used += len;
    }
    spans
}

/// Canonical page ids (`block`-sized) any of `spans` touches, restricted to
/// the sealed history `[0, frontier_pages)`. Sorted and deduped. A media span
/// is indivisible (nInfer media group): every page it intersects must be
/// pinned in the per-layer keep-set, and a page past the sealed frontier is
/// the suffix stripe (always attended), not a canonical pin.
pub fn pages_from_spans(spans: &[MediaSpan], block: u64, frontier_pages: u64) -> Vec<u64> {
    let mut pages: Vec<u64> = Vec::new();
    if block == 0 {
        return pages;
    }
    for span in spans {
        if span.begin >= span.end {
            continue;
        }
        let first = span.begin / block;
        let last = (span.end - 1) / block;
        for id in first..=last {
            if id < frontier_pages {
                pages.push(id);
            }
        }
    }
    pages.sort_unstable();
    pages.dedup();
    pages
}

/// nInfer `validate_layerwise_media` for one flat, ordered span list.
pub fn validate_spans(prompt_len: u64, spans: &[MediaSpan], block: u64) -> Result<(), String> {
    if block == 0 {
        return Err("media spans: block must be positive".to_string());
    }
    let mut end = 0u64;
    for span in spans {
        if span.begin >= span.end {
            return Err(format!(
                "media span [{}, {}) is empty or inverted",
                span.begin, span.end
            ));
        }
        if span.begin < end {
            return Err(format!(
                "media span [{}, {}) overlaps or precedes its predecessor (ordered, non-overlapping required)",
                span.begin, span.end
            ));
        }
        if span.end > prompt_len {
            return Err(format!(
                "media span [{}, {}) exceeds the prompt length {prompt_len}",
                span.begin, span.end
            ));
        }
        end = span.end;
    }
    Ok(())
}

/// nInfer `layerwise_media_replay_begin`: floor `limit` to `block`, then step
/// back to `floor(span.begin, block)` for any media span the candidate would
/// split (`first < begin < end`), iterating to a fixed point. The result is
/// block-aligned and never lands inside a media span.
pub fn media_replay_begin(
    limit: u64,
    prompt_len: u64,
    spans: &[MediaSpan],
    block: u64,
) -> Result<u64, String> {
    if limit > prompt_len {
        return Err(format!(
            "media replay: limit {limit} exceeds the prompt length {prompt_len}"
        ));
    }
    validate_spans(prompt_len, spans, block)?;
    let mut begin = limit / block * block;
    loop {
        let mut moved = false;
        for span in spans {
            if span.begin < begin && begin < span.end {
                begin = span.begin / block * block;
                moved = true;
            }
        }
        if !moved {
            return Ok(begin);
        }
    }
}

/// Media-anchored reuse frontier: clamp/validate with the core
/// `prefix_anchor` (a group split below `base` is an error), then apply the
/// nInfer block-alignment fixed point. The result is block-aligned, lies in
/// `[base, prompt_len]`, and never splits a media span.
pub fn media_anchored_frontier(
    base: u64,
    frontier: u64,
    prompt_len: u64,
    spans: &[MediaSpan],
    block: u64,
) -> Result<u64, String> {
    validate_spans(prompt_len, spans, block)?;
    let ranges: Vec<PrefixRange> = spans
        .iter()
        .map(|s| PrefixRange {
            begin: s.begin,
            end: s.end,
        })
        .collect();
    // Core anchor: clamps into [base, prompt_len] and refuses a floor below
    // `base` (the restored prefix would split an input group).
    let anchored = prefix_anchor(base, frontier, prompt_len, &ranges)
        .map_err(|e| format!("media anchor: {e}"))?;
    let aligned = media_replay_begin(anchored, prompt_len, spans, block)?;
    if aligned < base {
        return Err(format!(
            "media anchor: frontier {frontier} cannot be media-aligned above base {base} \
             (expected {aligned})"
        ));
    }
    Ok(aligned)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: u64 = 128;

    #[test]
    fn split_and_inside_predicates() {
        let spans = [MediaSpan { begin: 200, end: 300 }];
        assert!(!media_split(&spans, 200));
        assert!(media_split(&spans, 256));
        assert!(!media_split(&spans, 300));
        assert!(!media_inside(&spans, 299));
        assert!(media_inside(&spans, 300));
        assert!(!media_inside(&[], 1000));
    }

    #[test]
    fn pad_runs_respect_n_visual() {
        let toks = [1u32, 9, 9, 9, 2, 9, 9];
        let spans = spans_from_pad_runs(&toks, 9, 5);
        // 3 + 2 rows used; both runs kept.
        assert_eq!(
            spans,
            vec![
                MediaSpan { begin: 1, end: 4 },
                MediaSpan { begin: 5, end: 7 },
            ]
        );
        // Cap inside the second run.
        let spans = spans_from_pad_runs(&toks, 9, 4);
        assert_eq!(
            spans,
            vec![
                MediaSpan { begin: 1, end: 4 },
                MediaSpan { begin: 5, end: 6 },
            ]
        );
        assert!(spans_from_pad_runs(&toks, 9, 0).is_empty());
    }

    #[test]
    fn spans_from_runs_drops_empty_and_sorts() {
        let spans = spans_from_runs(&[(400, 0), (300, 5), (100, 7)]);
        assert_eq!(
            spans,
            vec![
                MediaSpan { begin: 100, end: 107 },
                MediaSpan { begin: 300, end: 305 },
            ]
        );
    }

    #[test]
    fn replay_floors_without_media() {
        assert_eq!(media_replay_begin(280, 1000, &[], BLOCK).unwrap(), 256);
        assert_eq!(media_replay_begin(384, 1000, &[], BLOCK).unwrap(), 384);
    }

    #[test]
    fn replay_steps_back_out_of_a_span() {
        // Candidate 256 (floored from 280) falls inside (200, 300) -> floor(200)=128.
        let spans = [MediaSpan { begin: 200, end: 300 }];
        assert_eq!(media_replay_begin(280, 1000, &spans, BLOCK).unwrap(), 128);
    }

    #[test]
    fn replay_reaches_a_fixed_point_across_nested_spans() {
        // 384 inside A(270,400) -> floor(270)=256; 256 inside B(200,280) -> floor(200)=128.
        let spans = [
            MediaSpan { begin: 200, end: 280 },
            MediaSpan { begin: 270, end: 400 },
        ];
        // Non-overlapping spans whose page floors cascade: 384 -> B start
        // page 256, which falls inside A -> A start page 128.
        let spans = [
            MediaSpan { begin: 150, end: 260 },
            MediaSpan { begin: 270, end: 400 },
        ];
        assert_eq!(media_replay_begin(390, 1000, &spans, BLOCK).unwrap(), 128);
    }

    #[test]
    fn replay_keeps_a_span_start_boundary() {
        // The candidate sits exactly on a span start: not split, unchanged.
        let spans = [MediaSpan { begin: 256, end: 400 }];
        assert_eq!(media_replay_begin(256, 1000, &spans, BLOCK).unwrap(), 256);
        // One block later steps back to the span start page.
        assert_eq!(media_replay_begin(384, 1000, &spans, BLOCK).unwrap(), 256);
    }

    #[test]
    fn replay_rejects_bad_inputs() {
        let spans = [MediaSpan { begin: 100, end: 50 }];
        assert!(media_replay_begin(128, 1000, &spans, BLOCK).is_err());
        let spans = [MediaSpan { begin: 100, end: 200 }, MediaSpan { begin: 150, end: 300 }];
        assert!(media_replay_begin(128, 1000, &spans, BLOCK).is_err());
        let spans = [MediaSpan { begin: 900, end: 1100 }];
        assert!(media_replay_begin(128, 1000, &spans, BLOCK).is_err());
        assert!(media_replay_begin(1001, 1000, &[], BLOCK).is_err());
        assert!(media_replay_begin(128, 1000, &[], 0).is_err());
    }

    #[test]
    fn anchored_frontier_aligns_and_respects_base() {
        let spans = [MediaSpan { begin: 300, end: 400 }];
        // Frontier 384 splits the span: anchor steps to page 256 (>= base 0).
        assert_eq!(media_anchored_frontier(0, 384, 1000, &spans, BLOCK).unwrap(), 256);
        // No media: plain block alignment.
        assert_eq!(media_anchored_frontier(0, 300, 1000, &[], BLOCK).unwrap(), 256);
        // A span start on a page boundary lands exactly on it.
        let spans = [MediaSpan { begin: 256, end: 400 }];
        assert_eq!(media_anchored_frontier(0, 384, 1000, &spans, BLOCK).unwrap(), 256);
        // A frontier inside the span floors below `base`: refused split.
        let spans = [MediaSpan { begin: 100, end: 300 }];
        assert!(media_anchored_frontier(128, 280, 1000, &spans, BLOCK).is_err());
        assert_eq!(media_anchored_frontier(0, 280, 1000, &spans, BLOCK).unwrap(), 0);
        // Frontier beyond the prompt is clamped by the core anchor.
        assert_eq!(media_anchored_frontier(0, 5000, 1000, &[], BLOCK).unwrap(), 896);
    }

    #[test]
    fn anchored_frontier_is_deterministic() {
        let spans = [MediaSpan { begin: 200, end: 300 }, MediaSpan { begin: 700, end: 800 }];
        let a = media_anchored_frontier(0, 768, 1000, &spans, BLOCK).unwrap();
        let b = media_anchored_frontier(0, 768, 1000, &spans, BLOCK).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, 640);
    }

    #[test]
    fn pages_cover_every_touched_page() {
        // One span spanning a page boundary: both pages.
        let spans = [MediaSpan { begin: 100, end: 200 }];
        assert_eq!(pages_from_spans(&spans, BLOCK, 10), vec![0, 1]);
        // Page-aligned span: exactly its pages, both ends inclusive.
        let spans = [MediaSpan { begin: 128, end: 384 }];
        assert_eq!(pages_from_spans(&spans, BLOCK, 10), vec![1, 2]);
        // End on a boundary does not touch the next page.
        let spans = [MediaSpan { begin: 100, end: 128 }];
        assert_eq!(pages_from_spans(&spans, BLOCK, 10), vec![0]);
        // Two spans on the same page dedup.
        let spans = [
            MediaSpan { begin: 10, end: 20 },
            MediaSpan { begin: 30, end: 40 },
        ];
        assert_eq!(pages_from_spans(&spans, BLOCK, 10), vec![0]);
    }

    #[test]
    fn pages_are_clamped_to_the_sealed_frontier() {
        // A span reaching past the sealed history keeps only sealed pages; the
        // rest is the suffix stripe (always attended, never a canonical pin).
        let spans = [MediaSpan { begin: 200, end: 600 }];
        assert_eq!(pages_from_spans(&spans, BLOCK, 3), vec![1, 2]);
        assert_eq!(pages_from_spans(&spans, BLOCK, 0), Vec::<u64>::new());
        // Degenerate spans are ignored; block 0 yields nothing.
        let spans = [MediaSpan { begin: 5, end: 5 }];
        assert!(pages_from_spans(&spans, BLOCK, 3).is_empty());
        assert!(pages_from_spans(&[MediaSpan { begin: 0, end: 5 }], 0, 3).is_empty());
    }
}
