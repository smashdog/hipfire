//! Qwen3.5 DeltaNet checkpoint plumbing (spec §4.5 C5).
//!
//! The generic, family-free checkpoint pool lives in
//! [`hipfire_runtime::checkpoint_pool`]. This module holds only the
//! Qwen3.5-specific pieces: `CheckpointBlob` for `DeltaNetSnapshot`, the
//! GPU capture/restore over a live `DeltaNetState`, and re-exports so
//! existing `crate::checkpoint::…` paths keep working.

use crate::qwen35::DeltaNetState;
use crate::speculative::DeltaNetSnapshot;
use hip_bridge::{HipError, HipResult};
pub use hipfire_runtime::checkpoint_pool::{
    plan_resume, prefix_fingerprint, CheckpointBlob, CheckpointId, CheckpointPool,
};
use hipfire_runtime::serve_contract::CacheDomain;
use rdna_compute::Gpu;

/// G7: recurrent-state checkpoint key.
///
/// A media (vision) prefix cannot be identified by its token stream alone:
/// every image renders the same `image_pad` ids, so two different images in
/// the same text produce an identical `prefix_fingerprint`. Their recurrent
/// states differ (different visual embeddings feed the hybrid layers), so the
/// pool key must carry the media digest. `media_digest == None` is the
/// historical text-only key (byte-neutral for every non-VL path).
pub fn checkpoint_fingerprint(tokens: &[u32], media_digest: Option<u64>) -> u64 {
    let fp = prefix_fingerprint(tokens);
    match media_digest {
        None => fp,
        Some(digest) => {
            // splitmix64-style avalanche over (fp, digest).
            let mut x = fp ^ digest.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        }
    }
}

impl CheckpointBlob for DeltaNetSnapshot {
    fn bytes_len(&self) -> u64 {
        DeltaNetSnapshot::bytes_len(self)
    }
}

/// Capture a checkpoint from the live `DeltaNetState` at boundary `p`
/// and insert it into `pool` (spec §4.5 C5).
///
/// `p` must be page-aligned (`p % 128 == 0` or `p == 0`). Allocates a
/// fresh `DeltaNetSnapshot` via [`DeltaNetSnapshot::new_for`], copies the
/// live state into it via [`DeltaNetSnapshot::save_from`], and inserts it
/// into the pool. The pool evicts oldest unpinned entries as needed.
///
/// Returns the minted [`CheckpointId`], [`CheckpointId::NONE`] when the
/// pool ceiling cannot cover the capture even after evicting every
/// unpinned entry, or an error if snapshot allocation or the
/// device-to-device copy fails.
///
/// **P2-wire will call this** from the serve engine's commit/prefill
/// path at page-aligned completed boundaries.
pub fn capture_checkpoint(
    gpu: &mut Gpu,
    pool: &mut CheckpointPool<DeltaNetSnapshot>,
    domain: &CacheDomain,
    p: u64,
    boundary_tokens: &[u32],
    media_digest: Option<u64>,
    state: &DeltaNetState,
) -> HipResult<CheckpointId> {
    if !CheckpointPool::<DeltaNetSnapshot>::is_aligned(p) {
        return Err(HipError::new(
            0,
            "capture_checkpoint: boundary not page-aligned",
        ));
    }

    // Pre-check the pool ceiling BEFORE allocating the snapshot and paying
    // for the device-to-device copy: a capture the pool cannot afford is a
    // soft refusal, indistinguishable from a hard failure only after the
    // work is already spent.
    let bytes = DeltaNetSnapshot::bytes_for(state);
    let fp = checkpoint_fingerprint(boundary_tokens, media_digest);
    if !pool.can_afford(&(domain.clone(), p, fp), bytes) {
        return Ok(CheckpointId::NONE);
    }

    let mut snap = DeltaNetSnapshot::new_for(gpu, state)?;
    snap.save_from(state, gpu)?;

    let (id, displaced) = pool.insert(domain.clone(), p, fp, snap);
    // Displaced blobs (LRU evictions / same-key replacement / a ceiling
    // refusal of this very capture) own device memory with no freeing
    // `Drop` — free them here or they leak for the process lifetime.
    for blob in displaced {
        blob.free_gpu(gpu);
    }
    Ok(id)
}

/// Restore an immutable cached checkpoint into a caller-owned **private**
/// `DeltaNetSnapshot` via device-to-device copy (spec §4.5 C5).
///
/// The pool's snapshot is never mutated; `dst` receives a private copy.
/// `dst` must have been pre-allocated with matching shapes (e.g. via
/// [`DeltaNetSnapshot::new_for`] against the same model state).
///
/// Returns an error if the checkpoint is not found or the copy fails.
///
/// **P2-wire will call this** when executing a [`ResumePlan`] to obtain
/// private recurrent state for a running request.
pub fn restore_private(
    gpu: &mut Gpu,
    pool: &mut CheckpointPool<DeltaNetSnapshot>,
    domain: &CacheDomain,
    p: u64,
    fp: u64,
    dst: &mut DeltaNetSnapshot,
) -> HipResult<()> {
    let src = pool
        .peek(domain, p, fp)
        .ok_or_else(|| HipError::new(0, "restore_private: checkpoint not found"))?;
    src.copy_to(dst, gpu)
}

// ───────────────────────────────────────────────────────────────────────────
// Tests (host-only — no GPU/HIP required)
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_checkpoint_key_is_the_historical_fingerprint() {
        let toks: Vec<u32> = (0..256).map(|i| i * 7 + 3).collect();
        assert_eq!(
            checkpoint_fingerprint(&toks, None),
            prefix_fingerprint(&toks)
        );
    }

    #[test]
    fn media_checkpoint_key_is_content_scoped() {
        // One token stream, two images: identical proof, distinct states.
        let toks: Vec<u32> = (0..300).map(|i| i % 5).collect();
        let a = checkpoint_fingerprint(&toks, Some(0xA11CE));
        let b = checkpoint_fingerprint(&toks, Some(0xB0B));
        let a2 = checkpoint_fingerprint(&toks, Some(0xA11CE));
        assert_ne!(a, b);
        assert_eq!(a, a2);
        assert_ne!(a, prefix_fingerprint(&toks));
        assert_ne!(b, prefix_fingerprint(&toks));
    }

    #[test]
    fn media_key_depends_on_both_halves() {
        let t1: Vec<u32> = (0..256).collect();
        let t2: Vec<u32> = (0..256).map(|i| i ^ 1).collect();
        let d = Some(42);
        assert_ne!(
            checkpoint_fingerprint(&t1, d),
            checkpoint_fingerprint(&t2, d)
        );
    }
}
