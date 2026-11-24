// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Single-row paged decode helpers: the reference-shaped legacy path and the
//! Layerwise carrier path. These are the slot-engine KVMem-specific entry
//! points into [`forward_scratch_layers_paged`]; keeping them out of the
//! upstream `forward_slots.rs` spine leaves only the dispatch choice there.

use super::batch::PrefillBatchScratch;
use super::config::{LayerType, Qwen35Config};
use super::forward::Qwen35Scratch;
use super::weights::{DeltaNetState, Qwen35Weights};
use super::TapBinding;
use crate::forward_slots::SlotDescStaging;
use hip_bridge::{HipError, HipResult};
use hipfire_runtime::llama::{KvCache, VMode};
use hipfire_runtime::slot_batch::SlotBatch;
use rdna_compute::slot_pool::SlotPool;
use rdna_compute::{Gpu, GpuTensor};

/// Single-row paged decode through the reference's own kernel ladder.
///
/// [`forward_batch_slots`] normally runs the slots GEMM ladder, which at
/// n == 1 is both numerically different from the reference sequential
/// decode (WMMA GEMM vs GEMV, batched norms vs the fused decode norms) and
/// measurably slower per step on gfx1100. This fork hands the row to
/// [`crate::qwen35::forward::forward_scratch_layers_paged`]: the reference's
/// `DecodeBinding` layer loop with each full-attention layer bound to the
/// slot's block table. The hidden state moves `pbs.x_batch` row 0 → `s.x` →
/// back; the per-slot lm_head still runs from `pbs.x_batch` in
/// [`final_logits_per_slot`], unchanged.
///
/// The Q8 arenas are viewed in model-layer order (null tensors on DeltaNet
/// layers), the same shape `ModelRig::slot_kv_view` gives the contiguous
/// engine — the paged kernels only use those tensors as arena bases.
#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_decode_reference_layers(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pool: &SlotPool,
    slot_idx: usize,
    dn_states: &[DeltaNetState],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    desc_staging: &SlotDescStaging,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    ctx_len: usize,
    layer_end: usize,
) -> HipResult<()> {
    #[cfg(feature = "kvmem")]
    let tap = TapBinding::from_optional(pbs.kvmem_taps.as_ref(), &desc_staging.row_slot_dev);
    #[cfg(not(feature = "kvmem"))]
    let tap: Option<TapBinding<'_>> = None;
    let dim = config.dim;
    let row_bytes = dim * 4;
    // Hidden state of the decode row in; the reference writes the same
    // logical position into pos_buf before its layer loop (the RoPE angle
    // and the paged tile/reduce causal bound both read pos_buf[0]).
    gpu.memcpy_dtod_at_auto(&s.x.buf, 0, &pbs.x_batch.buf, 0, row_bytes)?;
    let pos = batch.positions[0] as usize;
    let pos_i32 = pos as i32;
    gpu.hip.memcpy_htod(&s.pos_buf, &pos_i32.to_ne_bytes())?;

    // Borrowed Q8 view of the paged arenas, model-layer order.
    let phys_cap = pool.cap_tokens();
    let mut k_gpu = Vec::with_capacity(config.n_layers);
    let mut v_gpu = Vec::with_capacity(config.n_layers);
    let mut fa = 0usize;
    for t in &config.layer_types {
        if *t == LayerType::FullAttention {
            k_gpu.push(k_arenas[fa].sub_offset(0, k_arenas[fa].numel()));
            v_gpu.push(v_arenas[fa].sub_offset(0, v_arenas[fa].numel()));
            fa += 1;
        } else {
            k_gpu.push(GpuTensor::null_for_test());
            v_gpu.push(GpuTensor::null_for_test());
        }
    }
    let kv_view = KvCache {
        k_gpu,
        v_gpu,
        k_scales: Vec::new(),
        v_scales: Vec::new(),
        kv_dim: config.n_kv_heads * config.head_dim,
        max_seq: phys_cap,
        physical_cap: phys_cap,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        quantized: true,
        quant_q8: true,
        quant_int8: false,
        quant_hfq4: false,
        quant_asym4: false,
        quant_asym3: false,
        quant_asym2: false,
        boundary_layers: 0,
        givens_cos: None,
        givens_sin: None,
        quant_fwht: false,
        quant_bf16: false,
        quant_fp8: false,
        v_mode: VMode::Q8,
        layer_is_boundary: Vec::new(),
        compact_offset: 0,
    };
    crate::qwen35::forward::forward_scratch_layers_paged(
        gpu,
        weights,
        config,
        pos,
        &kv_view,
        &dn_states[slot_idx],
        s,
        hipfire_dispatch::pipeline::hybrid::PagedAttentionKv {
            positions: &pbs.positions,
            descs: &desc_staging.descs_dev,
            row_slot: &desc_staging.row_slot_dev,
            ctx_len,
        },
        tap,
        /* layerwise */ None,
        /* use_mrope */ false,
        layer_end,
    )?;
    // Back into the [n_slots × dim] batch buffer the head reads.
    gpu.memcpy_dtod_at_auto(&pbs.x_batch.buf, 0, &s.x.buf, 0, row_bytes)?;
    Ok(())
}

/// Layerwise single-row q8 paged decode (D4b-3). Mirrors
/// [`paged_decode_reference_layers`] but the KV ladder is the carrier's
/// composite arena instead of the legacy per-layer slot arenas: every
/// FullAttention layer shares the arena base (K) and the shifted V base, and
/// the per-layer block table (`staging.descs(fa)`, installed inside
/// `run_attend_layerwise`) selects that layer's restored stripe + suffix. The
/// `paged` placeholder's descs/positions/ctx_len are overridden per layer by
/// the seam, so only the composite `kv_cache` views matter here. Escha-trellis
/// dense layers lower to Path C (`forward_scratch_layers_paged_lowered`);
/// non-escha dense layers reach the Path A guard until that path is wired
/// (fail-closed, never a silent legacy fallback). The carrier is threaded as
/// `Some`; the engine passes `None` until D4b-4 orchestration.
#[cfg(all(feature = "deltanet", feature = "kvmem"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_decode_layerwise(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    batch: &SlotBatch,
    pool: &SlotPool,
    slot_idx: usize,
    dn_states: &[DeltaNetState],
    desc_staging: &SlotDescStaging,
    pbs: &PrefillBatchScratch,
    s: &Qwen35Scratch,
    lq: &mut crate::layerwise_driver::LayerwiseQuery<'_>,
    layer_end: usize,
) -> HipResult<()> {
    let dim = config.dim;
    let row_bytes = dim * 4;
    gpu.memcpy_dtod_at_auto(&s.x.buf, 0, &pbs.x_batch.buf, 0, row_bytes)?;
    let pos = batch.positions[0] as usize;
    gpu.hip
        .memcpy_htod(&s.pos_buf, &(pos as i32).to_ne_bytes())?;
    // G7 Stage D: a VL decode row (image conversation, past the prompt) carries
    // uniform M-RoPE phases `[p+delta; 3]` from the batch. Upload them to the
    // single-row scratch so the attention prep can run the 3D RoPE kernel; the
    // KV write below still uses the carrier's COMPACT positions, and the phases
    // are the absolute ones the prefill stored keys under.
    let use_mrope = batch.pos3.len() == batch.positions.len() && !batch.pos3.is_empty();
    if use_mrope {
        let bytes: Vec<u8> = batch.pos3[0].iter().flat_map(|v| v.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&s.pos_buf3, &bytes)?;
    }

    // Composite arena views: non-owning aliases (the carrier's arena owns the
    // allocation). K base = arena base; V base = arena shifted by one K plane.
    // Extracted before the mutable carrier borrow below; never freed here.
    let arena_numel = lq.driver.arena().arena().numel();
    let mut k_gpu = Vec::with_capacity(config.n_layers);
    let mut v_gpu = Vec::with_capacity(config.n_layers);
    for t in &config.layer_types {
        if *t == LayerType::FullAttention {
            k_gpu.push(lq.driver.arena().arena().sub_offset(0, arena_numel));
            v_gpu.push(
                lq.driver
                    .arena()
                    .v_cache_view()
                    .map_err(|e| HipError::new(0, &e))?,
            );
        } else {
            k_gpu.push(GpuTensor::null_for_test());
            v_gpu.push(GpuTensor::null_for_test());
        }
    }
    let phys_cap = pool.cap_tokens();
    let kv_view = KvCache {
        k_gpu,
        v_gpu,
        k_scales: Vec::new(),
        v_scales: Vec::new(),
        kv_dim: config.n_kv_heads * config.head_dim,
        max_seq: phys_cap,
        physical_cap: phys_cap,
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        quantized: true,
        quant_q8: true,
        quant_int8: false,
        quant_hfq4: false,
        quant_asym4: false,
        quant_asym3: false,
        quant_asym2: false,
        boundary_layers: 0,
        givens_cos: None,
        givens_sin: None,
        quant_fwht: false,
        quant_bf16: false,
        quant_fp8: false,
        v_mode: VMode::Q8,
        layer_is_boundary: Vec::new(),
        compact_offset: 0,
    };
    crate::qwen35::forward::forward_scratch_layers_paged(
        gpu,
        weights,
        config,
        pos,
        &kv_view,
        &dn_states[slot_idx],
        s,
        hipfire_dispatch::pipeline::hybrid::PagedAttentionKv {
            positions: &pbs.positions,
            descs: &desc_staging.descs_dev,
            row_slot: &desc_staging.row_slot_dev,
            ctx_len: 0, // overridden per layer by run_attend_layerwise
        },
        TapBinding::from_optional(pbs.kvmem_taps.as_ref(), &desc_staging.row_slot_dev),
        Some(lq),
        use_mrope,
        layer_end,
    )?;
    // Back into the [n_slots × dim] batch buffer the head reads.
    gpu.memcpy_dtod_at_auto(&pbs.x_batch.buf, 0, &s.x.buf, 0, row_bytes)?;
    Ok(())
}
