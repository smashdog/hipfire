// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! KVMem Layerwise full-attention path: the non-escha Path A decode loop, the
//! per-layer attend seam, and the per-layer restore/capture hooks. Split out of
//! the upstream `forward.rs` spine; the generic forward only dispatches here
//! when a Layerwise carrier is present.

use super::config::Qwen35Config;
use super::forward::{attend_full_opcode, op_code, Qwen35Bindings, Qwen35Scratch};
use super::program::{dense_ffn, hybrid_dims, DecodeBinding};
use super::weights::{DeltaNetState, LayerWeights, Qwen35Weights};
use super::TapBinding;
use hip_bridge::{HipError, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_dispatch::pipeline::gated_mlp::MlpOutput;
use hipfire_dispatch::pipeline::hybrid;
use hipfire_dispatch::pipeline::hybrid::{AttentionTap, MixerOutput};
use hipfire_dispatch::pipeline::superop::OpBinding;
use hipfire_dispatch::pipeline::Step;
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::llama;
use rdna_compute::Gpu;

/// Layerwise paged decode for non-escha dense models (D4b-3 Path A). The
/// legacy [`forward_scratch_layers_paged`] body with ONE structural change:
/// every full-attention layer rebuilds its binding through
/// [`layerwise_fa_attend`], which swaps the row's block-table address for this
/// layer's restored composite-arena view and fires the per-layer score+restore
/// at the post-norm/pre-RoPE tap. A one-shot `DecodeBinding` cannot carry a
/// per-layer paged override, hence the rebuild (design §3.9b "转 Path C 形").
///
/// Everything else stays on the certified legacy executor: the mixer step runs
/// `execute_gated_attention` (project + attend + output in one op, so the QKV
/// projection and the output-gate/rotation handshake are untouched) and the
/// FFN runs `dense_ffn`'s `Step::GatedMlp` — NOT the lowered program's split
/// PROJ_GATE_UP/RESID_DOWN_SWIGLU ops, whose numerics differ. DeltaNet layers
/// keep the caller's `paged` (their recurrent state never reads a block table).
///
/// `kv_cache` is the caller's composite-arena view (shared K/V arena bases,
/// per-layer addressing via the block table); `paged` is only read by the
/// non-FA layers, the FA layers take their address from the carrier's staging.
/// Dense-FFN only, no final norm / lm_head — same contract as the legacy
/// sibling.
#[cfg(all(feature = "deltanet", feature = "kvmem"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_scratch_layers_paged_layerwise(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    config: &Qwen35Config,
    pos: usize,
    kv_cache: &llama::KvCache,
    dn_state: &DeltaNetState,
    s: &Qwen35Scratch,
    paged: hybrid::PagedAttentionKv<'_>,
    tap: Option<TapBinding<'_>>,
    lq: &mut crate::layerwise_driver::LayerwiseQuery<'_>,
    // G7 Stage D: `true` when the decode row's 3D M-RoPE phases were uploaded to
    // `s.pos_buf3` by the caller; the FA seam must rotate with them instead of
    // the 1D `pos_buf`. `false` reproduces the text path byte-for-byte.
    use_mrope: bool,
    layer_end: usize,
) -> HipResult<()> {
    let ctx = DispatchCtx::new(gpu);
    let dims = hybrid_dims(config);
    let mut delta_layer_idx = 0usize;
    let mut fa_layer_idx = 0usize;
    for layer_idx in 0..layer_end {
        let layer = &weights.layers[layer_idx];
        if matches!(
            layer,
            LayerWeights::DeltaNetMoe(_) | LayerWeights::FullAttnMoe(_)
        ) {
            return Err(HipError::new(
                0,
                "layerwise paged decode admits dense-FFN models only \
                 (the slots engine keeps its own MoE kernels)",
            ));
        }
        if matches!(layer, LayerWeights::FullAttn(_)) {
            let fa = fa_layer_idx;
            // The all-in-one legacy attention op: `execute_gated_attention`
            // projects, attends (firing our tap between Q/K norm and RoPE) and
            // runs the output projection with the epilogue's own prerotated
            // flag, so the returned flag is consumed internally here.
            layerwise_fa_attend(
                gpu,
                &ctx,
                s,
                config,
                kv_cache,
                dn_state,
                pos,
                fa,
                tap,
                lq,
                use_mrope.then_some(config.mrope_section),
                |gpu, ctx, binding, tap| {
                    let mixer = binding.mixer(
                        layer,
                        layer_idx,
                        delta_layer_idx,
                        false,
                        Some(tap),
                        MixerOutput::Residual,
                    );
                    execute_steps(gpu, ctx, &[mixer])?;
                    Ok(false)
                },
            )
            .map_err(|e| HipError::new(0, &e.to_string()))?;
            fa_layer_idx += 1;
        } else {
            let binding = DecodeBinding {
                dims,
                s,
                kv_cache,
                dn_state,
                position: pos,
                mrope_section: None,
                paged: Some(paged),
            };
            let mixer = binding.mixer(
                layer,
                layer_idx,
                delta_layer_idx,
                false,
                None,
                MixerOutput::Residual,
            );
            execute_steps(gpu, &ctx, &[mixer]).map_err(|e| HipError::new(0, &e.to_string()))?;
            delta_layer_idx += 1;
        }
        match dense_ffn(&dims, s, layer, MlpOutput::Residual) {
            Some(ffn) => execute_steps(gpu, &ctx, &[Step::GatedMlp(ffn)])
                .map_err(|e| HipError::new(0, &e.to_string()))?,
            None => {
                return Err(HipError::new(
                    0,
                    "layerwise paged decode admits dense-FFN models only",
                ))
            }
        }
    }
    Ok(())
}

/// Layerwise per-layer attend (design §3.9d, ②-D4b-3). Clean sibling to the
/// legacy copy-only tap: at the post-norm/pre-RoPE seam it feeds the current
/// token's K statistics (when taps are on) and runs the carrier's
/// `restore_layer` (GPU score → select → COW restore → per-layer block-table /
/// descriptor staging), then lets the UNCHANGED paged attention core consume the
/// restored compacted working set. The carrier handle persists across layers in
/// `self.layerwise`; only FullAttention layers reach here (the `run_attend`
/// dispatcher gates on the layer kind).
///
/// Borrow mechanics: the attention op must hold the per-layer descriptor /
/// positions / row_slot handles while the tap mutates the carrier. So the owned
/// shallow [`LayerwiseLayerViews`] are extracted FIRST (they alias the staging's
/// stable-address buffers), the op references those LOCAL views, and the carrier
/// moves into a `RefCell` the tap closure `borrow_mut()`s. `update_layer` takes
/// `&self` and the restore's H2D only changes GPU CONTENTS (never the handles),
/// so the kernel reads the freshly uploaded table/descriptor at launch.
#[cfg(all(feature = "deltanet", feature = "kvmem"))]
impl<'a, 'b> Qwen35Bindings<'a, 'b> {
    pub(crate) fn run_attend_layerwise(
        &mut self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        op: &OpBinding,
    ) -> Result<(), DispatchError> {
        // Copy the context out of `self` so the executor closure borrows only
        // locals, never `self` (the carrier is taken out below and restored after).
        let (s, config) = (self.s, self.config);
        let fa = self.fa_layer_idx;
        let tap = self.tap;
        let (layer, layer_idx, delta_layer_idx) =
            (self.layer, self.layer_idx, self.delta_layer_idx);
        let (kv_cache, dn_state, pos) = (self.kv_cache, self.dn_state, self.pos);
        let prerotated_input = self.precomputed_attn_x_rot;
        let lq = self
            .layerwise
            .take()
            .expect("run_attend_layerwise called with a layerwise handle");
        // Path C executor: the lowered program runs PROJ_QKV / RESID_WO as
        // separate ops, so this fires ONLY the attend op (proj already filled
        // fa_q_full; the returned prerotated flag feeds the later RESID_WO op).
        let prerotated = layerwise_fa_attend(
            gpu,
            ctx,
            s,
            config,
            kv_cache,
            dn_state,
            pos,
            fa,
            tap,
            &mut *lq,
            None, // the lowered (escha) executor has no M-RoPE arm
            |gpu, ctx, binding, tap| {
                let step = binding.mixer(
                    layer,
                    layer_idx,
                    delta_layer_idx,
                    prerotated_input,
                    Some(tap),
                    MixerOutput::Residual,
                );
                match (op_code(op), step) {
                    (opcode, Step::GatedAttention(aop)) if opcode == attend_full_opcode() => {
                        aop.attend(gpu, ctx)
                    }
                    (code, _) => Err(DispatchError::Hip(format!(
                        "layerwise ATTEND opcode {code} does not apply to a FullAttention layer"
                    ))),
                }
            },
        )?;
        self.layerwise = Some(lq);
        self.fa_output_prerotated = prerotated;
        Ok(())
    }
}

/// Shared Layerwise full-attention seam (D4b-3, design §3.9d). Fires the
/// per-layer score+restore at the post-norm/pre-RoPE tap, builds the attention
/// binding with the PER-LAYER paged override (the caller's composite `kv_cache`
/// + this layer's staging descs/positions/row_slot/ctx_len), runs the caller's
/// attention executor, then reports the attended evaluation span. The carrier
/// rides a `RefCell` so the `Fn` tap closure can borrow it mutably while the op
/// holds only the owned `views` (no conflict); the closure is scoped inside the
/// block so it drops before `into_inner` reclaims the carrier. Used by BOTH the
/// lowered Path C executor (`run_attend_layerwise`, attend-only op) and the
/// non-escha Path A executor (all-in-one `execute_gated_attention`), so the
/// restore interleaving lives in exactly one place. Returns the executor's
/// `prerotated` flag (Path C feeds it to the separate RESID_WO op; Path A
/// consumes it internally and ignores the return).
#[cfg(all(feature = "deltanet", feature = "kvmem"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn layerwise_fa_attend(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    s: &Qwen35Scratch,
    config: &Qwen35Config,
    kv_cache: &llama::KvCache,
    dn_state: &DeltaNetState,
    pos: usize,
    fa: usize,
    tap: Option<TapBinding<'_>>,
    lq: &mut crate::layerwise_driver::LayerwiseQuery<'_>,
    // G7 Stage D: `Some(section)` rotates with the 3D phases in `s.pos_buf3`
    // (VL decode); `None` keeps the 1D `pos_buf` path byte-identical.
    mrope_section: Option<[usize; 3]>,
    attend: impl FnOnce(
        &mut Gpu,
        &DispatchCtx,
        &DecodeBinding<'_>,
        AttentionTap<'_>,
    ) -> Result<bool, DispatchError>,
) -> Result<bool, DispatchError> {
    use hipfire_kvmem::NativeCurrentLayerQuery;
    use std::cell::RefCell;

    // Owned per-layer binding handles, extracted before the carrier borrow.
    let views = lq.layer_views(fa).map_err(DispatchError::Hip)?;
    let cell = RefCell::new(lq);

    // The Q tile the scorer reads: post-norm/pre-RoPE `s.fa_q`, laid out
    // `[n_heads, head_dim]` F32 by the unfused prep (one decode row). BYTE
    // strides, token axis outermost; attention_scale 0 → 1/sqrt(head_dim).
    let qw = config.n_heads * config.head_dim;
    let native_q = NativeCurrentLayerQuery {
        data: s.fa_q.buf.as_ptr() as *const std::ffi::c_void,
        bytes: qw * 4,
        layer: fa as u32,
        rows: 1,
        query_heads: config.n_heads as u32,
        kv_heads: config.n_kv_heads as u32,
        head_dim: config.head_dim as u32,
        attention_scale: 0.0,
        scalar: 0, // F32
        row_stride: qw * 4,
        head_stride: config.head_dim * 4,
        dimension_stride: 4,
    };

    let prerotated = {
        // Stats feed (current K → tap buffer) then score+restore. Both borrow
        // the carrier through the RefCell; the op holds only the owned `views`.
        let tap_fn = |gpu: &mut Gpu| -> Result<(), DispatchError> {
            if let Some(tap) = tap {
                tap.tap_layer(gpu, fa, &s.fa_q, &s.fa_k, 1)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
            cell.borrow_mut()
                .restore_layer(gpu, fa, &native_q)
                .map_err(DispatchError::Hip)?;
            Ok(())
        };
        let binding = DecodeBinding {
            dims: hybrid_dims(config),
            s,
            kv_cache,
            dn_state,
            position: pos,
            mrope_section,
            paged: Some(hybrid::PagedAttentionKv {
                positions: &views.positions,
                descs: &views.descs,
                row_slot: &views.row_slot,
                ctx_len: views.ctx_len,
            }),
        };
        attend(gpu, ctx, &binding, &tap_fn as AttentionTap<'_>)?
    };

    // The tap closure and binding are dropped; reclaim the carrier and report
    // the attended evaluation span. On the error paths above the carrier is
    // dropped with the cell (its GPU buffers are freed by the owner's `free`).
    let lq = cell.into_inner();
    // ③-E3 mass hook (design §3.10a): sample this layer's per-page attention
    // mass at the post-attention boundary. `s.fa_q` is now POST-RoPE — the
    // attend op rotated it in place (rope_partial_interleaved_f32, the non-fused
    // arm taken whenever the restore tap is present) — the SAME rotational space
    // as the restored arena K, and the current token's K is written. No-op unless
    // a pressure owner backs the step AND the core `rows()` schedule admitted it
    // (the page-first Query steps where the frontier advances). Purely additive:
    // the mass kernel reads disjoint workspaces and never touches the attention
    // output / arena / view, so the forward residual is byte-identical with or
    // without it. This is the Dense Path A/C sibling of the MoE Path B hook in
    // `forward_slots::layerwise_fa_slots`; the real Qwen3.5 dense model decodes
    // through THIS seam, so the production pressure accept needs it here too.
    {
        use rdna_compute::kvmem_page_mass::{PageMassQuery, PageMassScalar};
        let mass_q = PageMassQuery {
            data: s.fa_q.buf.as_ptr() as *const std::ffi::c_void,
            rows: 1,
            query_heads: config.n_heads as u32,
            kv_heads: config.n_kv_heads as u32,
            head_dim: config.head_dim as u32,
            attention_scale: 0.0,
            scalar: PageMassScalar::F32,
        };
        lq.capture_mass_layer(gpu, fa, &mass_q)
            .map_err(DispatchError::Hip)?;
    }
    lq.attend_complete_layer(fa).map_err(DispatchError::Hip)?;
    Ok(prerotated)
}
