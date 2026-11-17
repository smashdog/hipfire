// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! P4.1③ UpstreamGlobal MTP head-history private-probe selection.
//!
//! The UpstreamGlobal history policy selects the MTP head's draft-owned K/V
//! history by a GLOBAL Mean-K reduction over the target's pre-RoPE Q (core
//! `kvmem::LayerwiseProbeSelection`, wrapped by [`hipfire_kvmem::ProbeSelection`]),
//! rather than FirstLayer's reuse of the trunk's first-layer restored view. Both
//! leave the target's fresh-Q trunk selection unchanged and restore the head's
//! OWN canonical K/V for the selected original token IDs; only the head-history
//! SELECTION differs (design §A.9, Q10=dual).
//!
//! ## hipfire adaptation (documented architectural deviation, Q4=b)
//!
//! ninfer runs a separate private target-layer-major prefill forward per Query
//! (`native_layerwise_probe`, NL:4166-4398) to produce the probe Q. hipfire's ②
//! trunk carrier ALREADY captures the identical post-RMSNorm/pre-RoPE Q into its
//! per-FA-ordinal activation tapes during `restore_layer` (the same `native_q`
//! the trunk scorer reads). [`probe_head_history`] reuses those captured Q tapes
//! — byte-identical to what a private forward would produce — so the global
//! reduction picks the same head history without a second forward per draft.
//! The probe is a pure host selection over already-captured Q: it mutates NO KV
//! and needs NO rollback (unlike ninfer's private forward). The first draft of a
//! request (before any trunk carrier has captured Q) is handled by the
//! probe-window / restore-all fast path in the engine orchestration (H2b/H3).
//!
//! Additive and gated: only the UpstreamGlobal policy reaches this module; the
//! default Legacy serve and the FirstLayer policy never construct a probe, so
//! both stay byte-neutral. ZERO core C++ change.

use crate::layerwise_driver::{score_current_layer_native, LayerwiseActivation};
use crate::layerwise_mtp::{auxiliary_visible_tokens, AuxiliarySelection};
use hipfire_kvmem::{
    KvStore, LayerQueryGeometry, LayerwiseIdentity, MemoryStamp, NativeCurrentLayerQuery,
    ProbeSelection, QueryRange, RawStore, SessionIdentity, Storage,
};
use rdna_compute::{DType, Gpu};

/// Synthesize the page-aligned probe identity for a prefill-seed / carrier
/// head-history selection at accepted frontier `frontier` (mirrors
/// `bootstrap_layerwise_rows`' 1-based logical identity mapping: the core
/// requires non-zero session/revision/request). The evaluation span is the single
/// frontier token `[frontier, frontier+1)`; the probe Q source (activation tape
/// or prefill Q tap) supplies that token's per-layer pre-RoPE Q. `history.frontier
/// == policy.total_tokens()` must hold (the caller passes the accepted `fed`).
pub fn probe_identity(
    frontier: u64,
    session_id: u64,
    incarnation: u64,
    seq: u64,
) -> LayerwiseIdentity {
    let session = SessionIdentity { id: session_id + 1, incarnation: incarnation + 1 };
    let revision = seq + 1;
    LayerwiseIdentity {
        history: MemoryStamp {
            valid: 1,
            session,
            history_revision: revision,
            view_version: revision,
            frontier,
            execution_history: revision,
        },
        request: session_id + 1,
        evaluation: QueryRange { begin: frontier, end: frontier + 1 },
        query: QueryRange { begin: frontier, end: frontier + 1 },
    }
}

/// Upper bound on the probe Q tile rows uploaded to the device per append. The
/// head verify width is small (≤64); chunking bounds the transient device Q
/// buffer regardless of the evaluation span. Token-major coverage is preserved
/// (the core accumulates across tiles before the single global pick).
const PROBE_Q_CHUNK: u64 = 64;

/// Shared UpstreamGlobal probe orchestration over an arbitrary per-layer Q
/// source. `read_q(il, begin, rows)` returns the packed F32 Q tile bytes for FA
/// ordinal `il` over the token span `[begin, begin+rows)` (row width
/// `q_row_bytes`). The core accumulator averages the L layers (one global Mean-K
/// reduction) before the single head-history pick; this driver owns the abort
/// envelope (the `ProbeSelection` is dropped on every error path, retiring its
/// metadata lease), mirroring ninfer `native_layerwise_probe`'s try/catch →
/// `abort()`. Fails closed (returning `Err`, never publishing a partial
/// selection) on any core/scorer/geometry error.
#[allow(clippy::too_many_arguments)]
fn run_probe_selection<F>(
    gpu: &mut Gpu,
    id: &LayerwiseIdentity,
    policy: &KvStore,
    keys: &RawStore,
    n_layers: usize,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    attention_scale: f32,
    block_tokens: u32,
    frontier: u64,
    q_row_bytes: usize,
    mut read_q: F,
) -> Result<AuxiliarySelection, String>
where
    F: FnMut(usize, u64, u64) -> Result<Vec<u8>, String>,
{
    if n_layers == 0 {
        return Err("layerwise probe: no target layers".to_string());
    }
    if block_tokens == 0 {
        return Err("layerwise probe: zero block_tokens".to_string());
    }
    if q_row_bytes == 0 {
        return Err("layerwise probe: zero Q row width".to_string());
    }
    let qb = id.query.begin;
    let qe = id.query.end;
    if qe <= qb {
        return Err("layerwise probe: empty query span".to_string());
    }

    // One target geometry per FA ordinal (token-major scoring order). All layers
    // share the frozen page denominator and the same represented head geometry.
    let targets: Vec<LayerQueryGeometry> = (0..n_layers)
        .map(|il| LayerQueryGeometry {
            layer: il as u32,
            query_heads: n_heads,
            kv_heads: n_kv_heads,
            head_dim,
            attention_scale,
        })
        .collect();

    // Admit the probe accumulator against an independent FINITE metadata account
    // sized to the conservative core bound (ninfer `probe_metadata`, NL:10125).
    let bound = ProbeSelection::metadata_bound(policy, n_layers as u32)
        .map_err(|e| format!("layerwise probe: metadata bound: {e}"))?;
    let metadata =
        Storage::new_ram(bound).map_err(|e| format!("layerwise probe: metadata account: {e}"))?;
    let rev = id.history;
    let probe = ProbeSelection::new(id, policy, keys, &targets, move || rev, &metadata, &[])
        .map_err(|e| format!("layerwise probe: admit: {e}"))?;

    // Token-major: for each Query tile [begin, begin+rows), append every target
    // layer in geometry order. The core slices each tile to the query intersection
    // and averages the L layers (one global reduction) before the single pick.
    let mut begin = qb;
    while begin < qe {
        let rows = (qe - begin).min(PROBE_Q_CHUNK);
        for il in 0..n_layers {
            let host = read_q(il, begin, rows)
                .map_err(|e| format!("layerwise probe: read Q layer {il}: {e}"))?;
            let tile = gpu
                .alloc_tensor(&[host.len()], DType::Raw)
                .map_err(|e| format!("layerwise probe: Q tile alloc: {e:?}"))?;
            let upload = gpu.hip.memcpy_htod(&tile.buf, &host);
            if let Err(e) = upload {
                let _ = gpu.free_tensor(tile);
                return Err(format!("layerwise probe: Q tile H2D layer {il}: {e:?}"));
            }
            let q = NativeCurrentLayerQuery {
                data: tile.buf.as_ptr() as *const std::ffi::c_void,
                bytes: host.len(),
                layer: il as u32,
                rows: rows as u32,
                query_heads: n_heads,
                kv_heads: n_kv_heads,
                head_dim,
                attention_scale,
                scalar: 0, // NativeQueryScalar::Float32 (the Q source stores F32)
                row_stride: q_row_bytes,
                head_stride: (head_dim as usize) * 4,
                dimension_stride: 4,
            };
            let il32 = il as u32;
            let appended = probe.append_native(id, begin, &q, |query, frame| {
                score_current_layer_native(gpu, keys, il32, block_tokens, query, frame)
            });
            // The scorer read the tile synchronously; release it on every path.
            let freed = gpu.free_tensor(tile);
            appended.map_err(|e| format!("layerwise probe: append layer {il}: {e}"))?;
            freed.map_err(|e| format!("layerwise probe: Q tile free: {e:?}"))?;
        }
        begin += rows;
    }

    probe
        .finish(id)
        .map_err(|e| format!("layerwise probe: finish: {e}"))?;
    let selected = probe.selected_ids();

    // Bind the picked head history against the accepted extent (mirrors
    // FirstLayer): the visible extent is derived from the selected canonical
    // pages clipped to `frontier`, then re-validated chronological/unique/exact.
    let history_tokens = auxiliary_visible_tokens(&selected, frontier, block_tokens as u64)
        .map_err(|e| format!("layerwise probe: head extent: {e}"))?;
    let mut sel = AuxiliarySelection::new(history_tokens, block_tokens as u64, true);
    sel.bind(&selected, frontier)
        .map_err(|e| format!("layerwise probe: bind head history: {e}"))?;
    Ok(sel)
}

/// Run the UpstreamGlobal private-probe head-history selection over the trunk
/// carrier's captured activation Q tapes, returning the bound auxiliary(head)
/// selection for [`crate::kvmem_ctl::KvMemCtl::set_aux_selection`].
///
/// `activation` is the trunk carrier's attached [`LayerwiseActivation`] (its
/// per-FA-ordinal Q tapes hold the target's post-norm/pre-RoPE Q over the
/// evaluation span; its identity is the page-aligned Query identity whose
/// `history.frontier == policy.total_tokens()`). `policy`/`keys` are the SAME
/// page-aligned sealed Mean-K TopK index the trunk Query admitted. `frontier` is
/// the accepted native frontier (`fed`) the head history extent is clipped to
/// (mirrors FirstLayer's `bind_first_layer_auxiliary`). The geometry
/// (`n_heads`/`n_kv_heads`/`head_dim`/`attention_scale`) must match the keys
/// store (`kv_heads * head_dim == n_embd_k`) and the real forward's Q.
///
/// Fails closed (returning `Err`, never publishing a partial selection) on any
/// core/scorer/geometry error; the caller aborts the step. The core accumulator
/// does NOT auto-abort on a scorer throw — this driver owns the abort envelope
/// (the `ProbeSelection` is dropped on every error path, retiring its metadata
/// lease), mirroring ninfer `native_layerwise_probe`'s try/catch → `abort()`.
pub fn probe_head_history(
    gpu: &mut Gpu,
    activation: &LayerwiseActivation,
    policy: &KvStore,
    keys: &RawStore,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    attention_scale: f32,
    block_tokens: u32,
    frontier: u64,
) -> Result<AuxiliarySelection, String> {
    let id = activation.identity();
    let n_layers = activation.n_layers();
    let q_row_bytes = activation.q_row_bytes();
    run_probe_selection(
        gpu, &id, policy, keys, n_layers, n_heads, n_kv_heads, head_dim, attention_scale,
        block_tokens, frontier, q_row_bytes,
        |il, begin, rows| {
            let mut host = vec![0u8; (rows as usize) * q_row_bytes];
            activation.read_layer(il, begin, rows, &mut host)?;
            Ok(host)
        },
    )
}

/// Run the UpstreamGlobal head-history selection over already-captured host Q
/// rows (the hipfire prefill-seed Q tap, H2b). `q_rows` is layer-major
/// `[il][span_token][q_width]` packed F32 (post-RMSNorm/pre-RoPE — the same
/// representation as the activation tape Q), with `span_rows = id.query.end -
/// id.query.begin` tokens per layer and `q_width = n_heads * head_dim`. `id` is
/// the page-aligned probe identity whose `history.frontier == policy.total_tokens()`
/// and `evaluation.begin == frontier`; the caller synthesizes it from the slot's
/// layerwise counters + accepted frontier (mirroring `bootstrap_layerwise_rows`).
///
/// This is the prefill-seed analog of [`probe_head_history`]: hipfire's prefill
/// is full-resident with no trunk carrier, but the Legacy Q tap already captures
/// the per-layer pre-RoPE Q the trunk recall scores against. Reusing it makes the
/// probe a pure host selection — NO private forward, NO KV mutation, NO rollback
/// (documented Q4=b adaptation; ninfer instead re-runs a private prefill forward
/// over the Query evaluation span, `run_private_probe` NL:10112). Only the
/// UpstreamGlobal policy reaches this; Legacy / FirstLayer never construct a
/// probe (byte-neutral). ZERO core C++ change.
#[allow(clippy::too_many_arguments)]
pub fn probe_head_history_qrows(
    gpu: &mut Gpu,
    q_rows: &[f32],
    id: &LayerwiseIdentity,
    policy: &KvStore,
    keys: &RawStore,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    attention_scale: f32,
    block_tokens: u32,
    frontier: u64,
) -> Result<AuxiliarySelection, String> {
    let q_width = (n_heads as usize) * (head_dim as usize);
    let q_row_bytes = q_width * 4;
    let qb = id.query.begin;
    let qe = id.query.end;
    if qe <= qb {
        return Err("layerwise probe: empty query span".to_string());
    }
    if q_width == 0 {
        return Err("layerwise probe: zero Q geometry".to_string());
    }
    let span_rows = (qe - qb) as usize;
    let per_layer = span_rows * q_width;
    if q_rows.len() % per_layer != 0 {
        return Err(format!(
            "layerwise probe: Q rows {} not a multiple of per-layer span {per_layer}",
            q_rows.len()
        ));
    }
    let n_layers = q_rows.len() / per_layer;
    let eval_begin = id.evaluation.begin;
    run_probe_selection(
        gpu, id, policy, keys, n_layers, n_heads, n_kv_heads, head_dim, attention_scale,
        block_tokens, frontier, q_row_bytes,
        |il, begin, rows| {
            // Token-major tile for layer il: tokens [begin, begin+rows) relative
            // to the evaluation span start. q_rows is layer-major [il][token][qw].
            let t0 = (begin - eval_begin) as usize;
            let mut host = Vec::with_capacity((rows as usize) * q_row_bytes);
            for t in t0..t0 + (rows as usize) {
                let s0 = (il * span_rows + t) * q_width;
                let row = &q_rows[s0..s0 + q_width];
                let bytes =
                    unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, q_width * 4) };
                host.extend_from_slice(bytes);
            }
            Ok(host)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvmem_ctl::LayerwiseQuotas;
    use hipfire_kvmem::{
        KvStoreConfig, LayerwiseIdentity, MemoryStamp, QueryRange, RawStoreConfig, SessionIdentity,
    };

    // Q8-ish geometry mirroring the ② activation fixture: 2 KV heads × 64 dim →
    // kw 128; 4 query heads → qw 256; 128-token pages; 2-page accepted frontier.
    const KV_HEADS: u32 = 2;
    const Q_HEADS: u32 = 4;
    const HEAD_DIM: u32 = 64;
    const KW: usize = (KV_HEADS * HEAD_DIM) as usize;
    const QW: usize = (Q_HEADS * HEAD_DIM) as usize;
    const BT: u32 = 128;
    const FRONTIER: u64 = 2 * BT as u64; // 256

    fn stamp() -> MemoryStamp {
        MemoryStamp {
            valid: 1,
            session: SessionIdentity { id: 42, incarnation: 3 },
            history_revision: 5,
            view_version: 6,
            frontier: FRONTIER,
            execution_history: 9,
        }
    }

    fn identity() -> LayerwiseIdentity {
        LayerwiseIdentity {
            history: stamp(),
            request: 92,
            evaluation: QueryRange { begin: FRONTIER, end: FRONTIER + 1 },
            query: QueryRange { begin: FRONTIER, end: FRONTIER + 1 },
        }
    }

    /// GPU end-to-end: capture each FA layer's pre-RoPE Q into the activation
    /// tapes (the ② trunk carrier's capture path), then run the UpstreamGlobal
    /// probe over those tapes. Both layers' Q favor block1 (mean_k +1 vs -1), so
    /// the global L-layer reduction picks canonical block1 and binds a 128-token
    /// head history. Proves the tape→device→scorer→global-pick→bind wiring with
    /// no model and no live serve (additive; Legacy/FirstLayer never reach it).
    #[test]
    #[ignore]
    fn probe_head_history_global_reduction_gpu() {
        let mut gpu = Gpu::init().expect("gpu");
        let id = identity();
        let n_layers = 2usize;
        let q_row_bytes = QW * 4;
        let quotas = LayerwiseQuotas::default();
        let mut act =
            LayerwiseActivation::new(&mut gpu, id, n_layers, 2, q_row_bytes, &quotas)
                .expect("activation");

        // Capture a positive-Q oracle (favors mean_k +1) for both target layers.
        let mut tiles = Vec::new();
        for ordinal in 0..n_layers {
            let host = vec![0.1f32; QW];
            let bytes = unsafe {
                std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4)
            }
            .to_vec();
            let t = gpu.alloc_tensor(&[bytes.len()], DType::Raw).expect("q alloc");
            gpu.hip.memcpy_htod(&t.buf, &bytes).expect("q h2d");
            let q = NativeCurrentLayerQuery {
                data: t.buf.as_ptr() as *const std::ffi::c_void,
                bytes: bytes.len(),
                layer: ordinal as u32,
                rows: 1,
                query_heads: Q_HEADS,
                kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                attention_scale: 1.0,
                scalar: 0,
                row_stride: QW * 4,
                head_stride: HEAD_DIM as usize * 4,
                dimension_stride: 4,
            };
            act.capture_layer(&mut gpu, ordinal, &q).expect("capture");
            tiles.push(t);
        }

        // Sealed plain Mean-K TopK index: 2 canonical 128-token pages, budget 1.
        let cfg = KvStoreConfig {
            block_tokens: BT,
            select_budget: BT,
            sink_blocks: 0,
            recent_blocks: 0,
            ..KvStoreConfig::default()
        };
        let policy = KvStore::new(&cfg).expect("policy");
        policy.register_append(2 * BT).expect("append");
        let keys = RawStore::new(
            RawStoreConfig {
                n_layer: n_layers as u32,
                n_embd_k: KW as u32,
                n_embd_v: KW as u32,
                block_tokens: BT,
                k_row_bytes: 0,
                k_gpu_row_bytes: 136,
                v_gpu_row_bytes: 136,
            },
            None,
        )
        .expect("keys");
        for il in 0..n_layers as u32 {
            keys.write_layer_mean_k(0, BT, il, &[-1.0; BT as usize * KW]).expect("mean k block0");
            keys.write_layer_mean_k(BT, BT, il, &[1.0; BT as usize * KW]).expect("mean k block1");
        }

        let sel = probe_head_history(
            &mut gpu, &act, &policy, &keys, Q_HEADS, KV_HEADS, HEAD_DIM, 1.0, BT, FRONTIER,
        )
        .expect("probe head history");
        assert_eq!(sel.selected(), Some(&[1u32][..]), "global reduction picks block1");
        assert_eq!(sel.history_tokens(), BT as u64, "one 128-token head history page");
        assert!(sel.is_bound());

        for t in tiles {
            let _ = gpu.free_tensor(t);
        }
    }

    /// GPU end-to-end for the H2b prefill-seed host-Q-tap variant: the same
    /// 2-page Mean-K index (block0 -1 / block1 +1) and a positive-Q oracle, but
    /// the Q comes from packed host rows (the Legacy tap representation) instead
    /// of activation tapes. The global L-layer reduction must pick block1 and
    /// bind a 128-token head history — proving the tap→device→scorer→global-pick
    /// →bind wiring with no model, no carrier and no live serve.
    #[test]
    #[ignore]
    fn probe_head_history_qrows_global_reduction_gpu() {
        let mut gpu = Gpu::init().expect("gpu");
        let id = identity();
        let n_layers = 2usize;

        // Sealed plain Mean-K TopK index: 2 canonical 128-token pages, budget 1.
        let cfg = KvStoreConfig {
            block_tokens: BT,
            select_budget: BT,
            sink_blocks: 0,
            recent_blocks: 0,
            ..KvStoreConfig::default()
        };
        let policy = KvStore::new(&cfg).expect("policy");
        policy.register_append(2 * BT).expect("append");
        let keys = RawStore::new(
            RawStoreConfig {
                n_layer: n_layers as u32,
                n_embd_k: KW as u32,
                n_embd_v: KW as u32,
                block_tokens: BT,
                k_row_bytes: 0,
                k_gpu_row_bytes: 136,
                v_gpu_row_bytes: 136,
            },
            None,
        )
        .expect("keys");
        for il in 0..n_layers as u32 {
            keys.write_layer_mean_k(0, BT, il, &[-1.0; BT as usize * KW]).expect("mean k block0");
            keys.write_layer_mean_k(BT, BT, il, &[1.0; BT as usize * KW]).expect("mean k block1");
        }

        // Layer-major host Q rows: [il][span_token=1][QW], positive (favors +1).
        let q_rows = vec![0.1f32; n_layers * QW];
        let sel = probe_head_history_qrows(
            &mut gpu, &q_rows, &id, &policy, &keys, Q_HEADS, KV_HEADS, HEAD_DIM, 1.0, BT, FRONTIER,
        )
        .expect("probe head history qrows");
        assert_eq!(sel.selected(), Some(&[1u32][..]), "global reduction picks block1");
        assert_eq!(sel.history_tokens(), BT as u64, "one 128-token head history page");
        assert!(sel.is_bound());
    }
}