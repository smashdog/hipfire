//! Escha-W2 trellis transform tables — borrowed view shared with the model.
//!
//! The dense escha decode path (`hipfire_arch_qwen35::qwen35::escha`) runs the
//! H128 rotations and trellis GEMV through `rdna_compute::escha_*` directly,
//! bypassing the fused gemv family (which cannot read a trellis code). This
//! module therefore only carries the borrowed table view the model builds plus
//! the H128 launch-budget constants the parity gates assert against.
//!
//! The routed-MoE executors (`escha_routed_decode`, `escha_routed_decode_indexed`,
//! `escha_routed_prefill_indexed`) that previously lived here were removed when
//! the sealed-MoE rewrite landed: they depended on the pre-seal `MoeParams` /
//! `MoeResolution` layout. Escha MoE (escha-35b) is out of scope for the dense
//! 27B export this backend targets.

use rdna_compute::GpuTensor;

/// Borrowed view of one layer's Escha-W2 transform tables plus the decode
/// scratch the phase structure needs. Built by the model
/// (`hipfire_arch_qwen35::qwen35::escha::EschaMoeTables::refs`); this crate
/// never owns or allocates any of it.
pub struct EschaRoutedRefs<'a> {
    /// `[n_exp, hidden]` f32 — folded `rin` for the gate_up projection.
    pub gate_up_rin: &'a GpuTensor,
    /// `[n_exp, 2*mi]` f32 — folded `rout` for gate_up. Carries the per-expert
    /// prune mask (zeros); see the zero contract in `escha_h128.hip`.
    pub gate_up_rout: &'a GpuTensor,
    /// `[n_exp, mi]` f32 — folded `rin` for the down projection.
    pub down_rin: &'a GpuTensor,
    /// `[n_exp, hidden]` f32 — folded `rout` for down.
    pub down_rout: &'a GpuTensor,
    /// `[k]` i32 — this token's selected expert ids (device).
    pub ids: &'a GpuTensor,
    /// `[k]` f32 — this token's combine weights, ALREADY f16-rounded.
    pub weights: &'a GpuTensor,
    /// `[k, hidden]` f32 scratch.
    pub xh_gu: &'a GpuTensor,
    /// `[k, 2*mi]` f32 scratch.
    pub mid_gu: &'a GpuTensor,
    /// `[k, 2*mi]` f32 scratch.
    pub y_gu: &'a GpuTensor,
    /// `[k, mi]` f32 scratch.
    pub h: &'a GpuTensor,
    /// `[k, mi]` f32 scratch.
    pub xh_dn: &'a GpuTensor,
    /// `[k, hidden]` f32 scratch.
    pub mid_dn: &'a GpuTensor,
    /// `[k, hidden]` f32 scratch — the per-slot expert outputs the combine
    /// reduces.
    pub y_dn: &'a GpuTensor,
}

/// Number of H128-family launches a routed escha layer issues per call. Pinned
/// as a constant so the launch budget in the parity gates is a checked claim
/// rather than a comment.
pub const ESCHA_H128_LAUNCHES_PER_LAYER: usize = 4;

/// H128 launches for a whole decode step. Independent of `k` — batching across
/// experts is the entire point.
pub fn escha_launches_per_token(n_layers: usize) -> usize {
    n_layers * ESCHA_H128_LAUNCHES_PER_LAYER
}