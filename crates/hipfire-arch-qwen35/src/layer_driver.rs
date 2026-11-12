// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The single qwen35 per-layer weight schema, generic over a runtime
//! `WeightBackend`. `load_weights` (HFQ), `load_weights_paroquant` (PaRo), and
//! `load_layer_into` (multi-GPU HFQ) all funnel through `load_layer`.

use crate::qwen35::escha::EschaProj;
use crate::qwen35::weights::{
    free_moe_ffn_with, free_weight_with, DeltaNetBiases, DeltaNetEscha, FullAttnBiases, FullAttnEscha,
};
use crate::qwen35::{
    DeltaNetLayerWeights, DeltaNetMoeLayerWeights, FullAttnLayerWeights, FullAttnMoeLayerWeights,
    LayerType, LayerWeights, MoeFfnWeights, Qwen35Config,
};
use hip_bridge::HipResult;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::weight_backend::WeightBackend;
use rdna_compute::GpuTensor;

/// All owners allocated while one layer is being assembled. Every field stays
/// optional until publication so an error can drain only the owners that
/// actually exist.
#[derive(Default)]
struct PendingLayer {
    attn_norm: Option<GpuTensor>,
    wqkv: Option<WeightTensor>,
    wz: Option<WeightTensor>,
    w_alpha: Option<WeightTensor>,
    w_beta: Option<WeightTensor>,
    a_log: Option<GpuTensor>,
    dt_bias: Option<GpuTensor>,
    conv_weight: Option<GpuTensor>,
    norm_weight: Option<GpuTensor>,
    wo: Option<WeightTensor>,
    wq: Option<WeightTensor>,
    wk: Option<WeightTensor>,
    wv: Option<WeightTensor>,
    q_norm: Option<GpuTensor>,
    k_norm: Option<GpuTensor>,
    ffn_norm: Option<GpuTensor>,
    w_gate: Option<WeightTensor>,
    w_up: Option<WeightTensor>,
    w_down: Option<WeightTensor>,
    ffn: Option<MoeFfnWeights>,
    // Escha dense sidecars. A layer is DeltaNet OR FullAttn, so the projection
    // slots are shared by name where the projection is common (o/gate/up/down);
    // qkv/z are DeltaNet-only and q/k/v FullAttn-only. Staged individually so a
    // mid-build failure drains exactly the owners that exist, like the weights.
    e_qkv: Option<EschaProj>,
    e_z: Option<EschaProj>,
    e_q: Option<EschaProj>,
    e_k: Option<EschaProj>,
    e_v: Option<EschaProj>,
    e_o: Option<EschaProj>,
    e_gate: Option<EschaProj>,
    e_up: Option<EschaProj>,
    e_down: Option<EschaProj>,
    e_ids: Option<GpuTensor>,
    e_iota: Option<GpuTensor>,
    // Dense output biases (escha exports only); same shared-name scheme.
    b_qkv: Option<GpuTensor>,
    b_z: Option<GpuTensor>,
    b_q: Option<GpuTensor>,
    b_k: Option<GpuTensor>,
    b_v: Option<GpuTensor>,
    b_o: Option<GpuTensor>,
    b_gate: Option<GpuTensor>,
    b_up: Option<GpuTensor>,
    b_down: Option<GpuTensor>,
}

impl PendingLayer {
    fn cleanup<B: WeightBackend>(&mut self, b: &mut B) {
        if let Some(ffn) = self.ffn.take() {
            let mut free = |tensor: GpuTensor| b.free_tensor(tensor);
            free_moe_ffn_with(ffn, &mut free);
        }
        for weight in [
            self.wqkv.take(),
            self.wz.take(),
            self.w_alpha.take(),
            self.w_beta.take(),
            self.wo.take(),
            self.wq.take(),
            self.wk.take(),
            self.wv.take(),
            self.w_gate.take(),
            self.w_up.take(),
            self.w_down.take(),
        ]
        .into_iter()
        .flatten()
        {
            let mut free = |tensor: GpuTensor| b.free_tensor(tensor);
            free_weight_with(weight, &mut free);
        }
        for tensor in [
            self.attn_norm.take(),
            self.a_log.take(),
            self.dt_bias.take(),
            self.conv_weight.take(),
            self.norm_weight.take(),
            self.q_norm.take(),
            self.k_norm.take(),
            self.ffn_norm.take(),
        ]
        .into_iter()
        .flatten()
        {
            b.free_tensor(tensor);
        }
        for proj in [
            self.e_qkv.take(),
            self.e_z.take(),
            self.e_q.take(),
            self.e_k.take(),
            self.e_v.take(),
            self.e_o.take(),
            self.e_gate.take(),
            self.e_up.take(),
            self.e_down.take(),
        ]
        .into_iter()
        .flatten()
        {
            b.free_tensor(proj.rin);
            b.free_tensor(proj.rout);
            b.free_tensor(proj.ptr0);
        }
        for tensor in [
            self.e_ids.take(),
            self.e_iota.take(),
            self.b_qkv.take(),
            self.b_z.take(),
            self.b_q.take(),
            self.b_k.take(),
            self.b_v.take(),
            self.b_o.take(),
            self.b_gate.take(),
            self.b_up.take(),
            self.b_down.take(),
        ]
        .into_iter()
        .flatten()
        {
            b.free_tensor(tensor);
        }
    }
}

/// Load one layer's weights. `load_moe` builds the MoE FFN block for MoE layers
/// (format-specific: HFQ `load_moe_ffn` vs PaRo `paro_load_moe_ffn`), supplied by
/// the caller so MoE layout stays arch-owned.
/// Largest token batch an escha layer's `ids` table serves. Decode reads a
/// 1-element prefix; batched prefill reads `n`. 4096 covers every prefill
/// chunk hipfire issues and costs 16 KB per layer.
const ESCHA_MAX_SLOTS: usize = 4096;

/// Build an `EschaProj` from the backend's sidecars, or `None` when the
/// weight is not a trellis code.
fn eproj<B: WeightBackend>(
    b: &mut B,
    rel: &str,
    w: &hipfire_runtime::llama::WeightTensor,
) -> hip_bridge::HipResult<Option<crate::qwen35::escha::EschaProj>> {
    Ok(b.escha_sidecars(rel, w)?
        .map(|(rin, rout, ptr0)| crate::qwen35::escha::EschaProj { rin, rout, ptr0 }))
}

/// Every coded projection in a layer is escha or none is — the export does not
/// mix. So probe one and require the rest, exactly as the biases do: a
/// half-escha layer is a corrupt checkpoint, and falling back per projection
/// would silently run some through the fused MQ paths on trellis bytes.
fn need_eproj<B: WeightBackend>(
    b: &mut B,
    rel: &str,
    w: &hipfire_runtime::llama::WeightTensor,
) -> hip_bridge::HipResult<crate::qwen35::escha::EschaProj> {
    eproj(b, rel, w)?.ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!("{rel}: layer has escha projections but this one is not coded"),
        )
    })
}

/// A bias that MUST be there.
///
/// Escha ships all of a layer's biases together or none, so once the first
/// probe finds one the rest are mandatory. A half-biased layer is a corrupt
/// checkpoint, and silently substituting zeros would degrade output without
/// failing — the same trap as the dropped MTP head.
fn need_bias<B: WeightBackend>(
    b: &mut B,
    rel: &str,
    n: usize,
) -> hip_bridge::HipResult<rdna_compute::GpuTensor> {
    b.bias_opt(rel, n)?.ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!(
                "{rel}: layer has some escha biases but not this one — a partially biased \
                 layer is a corrupt checkpoint, not a model to run with zeros"
            ),
        )
    })
}

pub(crate) fn load_layer<B: WeightBackend>(
    b: &mut B,
    config: &Qwen35Config,
    layer_idx: usize,
    mut load_moe: impl FnMut(&mut B, &Qwen35Config, usize) -> HipResult<MoeFfnWeights>,
) -> HipResult<LayerWeights> {
    b.set_layer(layer_idx);
    let is_moe = config.num_experts > 0;
    let qkv_dim = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let d_inner = config.linear_num_value_heads * config.linear_value_head_dim;
    let q_out_dim = config.n_heads * config.head_dim * 2;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let o_in = config.n_heads * config.head_dim;
    let mut pending = PendingLayer::default();

    macro_rules! stage {
        ($slot:ident, $load:expr) => {
            match $load {
                Ok(owner) => pending.$slot = Some(owner),
                Err(err) => {
                    pending.cleanup(b);
                    return Err(err);
                }
            }
        };
    }
    macro_rules! take {
        ($slot:ident) => {
            pending.$slot.take().expect(concat!(
                "load_layer: missing staged owner ",
                stringify!($slot)
            ))
        };
    }
    // Stage an escha projection set or a bias set. Probe the first member; if
    // it is absent the whole set is absent (a non-escha layer). If present,
    // every other member is mandatory — the export never mixes coded and plain
    // projections, nor partial biases — so each is staged with `stage!` and a
    // failure drains the owners staged so far. The probe's own owner is parked
    // in `pending` before the requires run so it is reclaimed on failure too.
    macro_rules! stage_sidecars {
        ($probe_slot:ident, $probe:expr, $($slot:ident => $load:expr),+ $(,)?) => {
            match $probe {
                Ok(None) => {}
                Ok(Some(owner)) => {
                    pending.$probe_slot = Some(owner);
                    $( stage!($slot, $load); )+
                }
                Err(err) => {
                    pending.cleanup(b);
                    return Err(err);
                }
            }
        };
    }

    let layer = match (config.layer_types[layer_idx], is_moe) {
        (LayerType::LinearAttention, false) => {
            stage!(attn_norm, b.norm("input_layernorm.weight", &[config.dim]));
            stage!(wqkv, b.proj("linear_attn.in_proj_qkv", qkv_dim, config.dim));
            stage!(wz, b.proj("linear_attn.in_proj_z", d_inner, config.dim));
            stage!(
                w_alpha,
                b.proj(
                    "linear_attn.in_proj_a",
                    config.linear_num_value_heads,
                    config.dim,
                )
            );
            stage!(
                w_beta,
                b.proj(
                    "linear_attn.in_proj_b",
                    config.linear_num_value_heads,
                    config.dim,
                )
            );
            stage!(
                a_log,
                b.raw_f32("linear_attn.A_log", config.linear_num_value_heads)
            );
            stage!(
                dt_bias,
                b.raw_f32("linear_attn.dt_bias", config.linear_num_value_heads)
            );
            stage!(
                conv_weight,
                b.raw_f32(
                    "linear_attn.conv1d.weight",
                    qkv_dim * config.conv_kernel_dim,
                )
            );
            stage!(
                norm_weight,
                b.raw_f32("linear_attn.norm.weight", config.linear_value_head_dim)
            );
            stage!(wo, b.proj("linear_attn.out_proj", config.dim, d_inner));
            stage!(
                ffn_norm,
                b.norm("post_attention_layernorm.weight", &[config.dim])
            );
            stage!(
                w_gate,
                b.proj("mlp.gate_proj", config.hidden_dim, config.dim)
            );
            stage!(w_up, b.proj("mlp.up_proj", config.hidden_dim, config.dim));
            stage!(
                w_down,
                b.proj("mlp.down_proj", config.dim, config.hidden_dim)
            );
            // Escha dense sidecars are keyed off each weight's dtype and device
            // pointer, so they are built from the staged tensors. in_proj_a/b
            // are on escha's ignore list and stay plain (no sidecar, no bias).
            stage_sidecars!(
                e_qkv,
                eproj(b, "linear_attn.in_proj_qkv", pending.wqkv.as_ref().unwrap()),
                e_z => need_eproj(b, "linear_attn.in_proj_z", pending.wz.as_ref().unwrap()),
                e_o => need_eproj(b, "linear_attn.out_proj", pending.wo.as_ref().unwrap()),
                e_gate => need_eproj(b, "mlp.gate_proj", pending.w_gate.as_ref().unwrap()),
                e_up => need_eproj(b, "mlp.up_proj", pending.w_up.as_ref().unwrap()),
                e_down => need_eproj(b, "mlp.down_proj", pending.w_down.as_ref().unwrap()),
                e_ids => b.zeros_i32(ESCHA_MAX_SLOTS),
                e_iota => b.iota_i32(ESCHA_MAX_SLOTS),
            );
            stage_sidecars!(
                b_qkv,
                b.bias_opt("linear_attn.in_proj_qkv.bias", qkv_dim),
                b_z => need_bias(b, "linear_attn.in_proj_z.bias", d_inner),
                b_o => need_bias(b, "linear_attn.out_proj.bias", config.dim),
                b_gate => need_bias(b, "mlp.gate_proj.bias", config.hidden_dim),
                b_up => need_bias(b, "mlp.up_proj.bias", config.hidden_dim),
                b_down => need_bias(b, "mlp.down_proj.bias", config.dim),
            );
            LayerWeights::DeltaNet(DeltaNetLayerWeights {
                attn_norm: take!(attn_norm),
                wqkv: take!(wqkv),
                wz: take!(wz),
                w_alpha: take!(w_alpha),
                w_beta: take!(w_beta),
                a_log: take!(a_log),
                dt_bias: take!(dt_bias),
                conv_weight: take!(conv_weight),
                norm_weight: take!(norm_weight),
                wo: take!(wo),
                ffn_norm: take!(ffn_norm),
                w_gate: take!(w_gate),
                w_up: take!(w_up),
                w_down: take!(w_down),
                biases: pending.b_qkv.take().map(|qkv| DeltaNetBiases {
                    qkv,
                    z: take!(b_z),
                    o: take!(b_o),
                    gate: take!(b_gate),
                    up: take!(b_up),
                    down: take!(b_down),
                }),
                escha: pending.e_qkv.take().map(|qkv| DeltaNetEscha {
                    qkv,
                    z: take!(e_z),
                    o: take!(e_o),
                    gate: take!(e_gate),
                    up: take!(e_up),
                    down: take!(e_down),
                    ids: take!(e_ids),
                    iota: take!(e_iota),
                }),
            })
        }
        (LayerType::FullAttention, false) => {
            stage!(attn_norm, b.norm("input_layernorm.weight", &[config.dim]));
            stage!(wq, b.proj("self_attn.q_proj", q_out_dim, config.dim));
            stage!(wk, b.proj("self_attn.k_proj", kv_dim, config.dim));
            stage!(wv, b.proj("self_attn.v_proj", kv_dim, config.dim));
            stage!(wo, b.proj("self_attn.o_proj", config.dim, o_in));
            stage!(
                q_norm,
                b.norm("self_attn.q_norm.weight", &[config.head_dim])
            );
            stage!(
                k_norm,
                b.norm("self_attn.k_norm.weight", &[config.head_dim])
            );
            stage!(
                ffn_norm,
                b.norm("post_attention_layernorm.weight", &[config.dim])
            );
            stage!(
                w_gate,
                b.proj("mlp.gate_proj", config.hidden_dim, config.dim)
            );
            stage!(w_up, b.proj("mlp.up_proj", config.hidden_dim, config.dim));
            stage!(
                w_down,
                b.proj("mlp.down_proj", config.dim, config.hidden_dim)
            );
            stage_sidecars!(
                e_q,
                eproj(b, "self_attn.q_proj", pending.wq.as_ref().unwrap()),
                e_k => need_eproj(b, "self_attn.k_proj", pending.wk.as_ref().unwrap()),
                e_v => need_eproj(b, "self_attn.v_proj", pending.wv.as_ref().unwrap()),
                e_o => need_eproj(b, "self_attn.o_proj", pending.wo.as_ref().unwrap()),
                e_gate => need_eproj(b, "mlp.gate_proj", pending.w_gate.as_ref().unwrap()),
                e_up => need_eproj(b, "mlp.up_proj", pending.w_up.as_ref().unwrap()),
                e_down => need_eproj(b, "mlp.down_proj", pending.w_down.as_ref().unwrap()),
                e_ids => b.zeros_i32(ESCHA_MAX_SLOTS),
                e_iota => b.iota_i32(ESCHA_MAX_SLOTS),
            );
            stage_sidecars!(
                b_q,
                b.bias_opt("self_attn.q_proj.bias", q_out_dim),
                b_k => need_bias(b, "self_attn.k_proj.bias", kv_dim),
                b_v => need_bias(b, "self_attn.v_proj.bias", kv_dim),
                b_o => need_bias(b, "self_attn.o_proj.bias", config.dim),
                b_gate => need_bias(b, "mlp.gate_proj.bias", config.hidden_dim),
                b_up => need_bias(b, "mlp.up_proj.bias", config.hidden_dim),
                b_down => need_bias(b, "mlp.down_proj.bias", config.dim),
            );
            LayerWeights::FullAttn(FullAttnLayerWeights {
                attn_norm: take!(attn_norm),
                wq: take!(wq),
                wk: take!(wk),
                wv: take!(wv),
                wo: take!(wo),
                q_norm: take!(q_norm),
                k_norm: take!(k_norm),
                ffn_norm: take!(ffn_norm),
                w_gate: take!(w_gate),
                w_up: take!(w_up),
                w_down: take!(w_down),
                biases: pending.b_q.take().map(|q| FullAttnBiases {
                    q,
                    k: take!(b_k),
                    v: take!(b_v),
                    o: take!(b_o),
                    gate: take!(b_gate),
                    up: take!(b_up),
                    down: take!(b_down),
                }),
                escha: pending.e_q.take().map(|q| FullAttnEscha {
                    q,
                    k: take!(e_k),
                    v: take!(e_v),
                    o: take!(e_o),
                    gate: take!(e_gate),
                    up: take!(e_up),
                    down: take!(e_down),
                    ids: take!(e_ids),
                    iota: take!(e_iota),
                }),
            })
        }
        (LayerType::LinearAttention, true) => {
            stage!(attn_norm, b.norm("input_layernorm.weight", &[config.dim]));
            stage!(wqkv, b.proj("linear_attn.in_proj_qkv", qkv_dim, config.dim));
            stage!(wz, b.proj("linear_attn.in_proj_z", d_inner, config.dim));
            stage!(
                w_alpha,
                b.proj(
                    "linear_attn.in_proj_a",
                    config.linear_num_value_heads,
                    config.dim,
                )
            );
            stage!(
                w_beta,
                b.proj(
                    "linear_attn.in_proj_b",
                    config.linear_num_value_heads,
                    config.dim,
                )
            );
            stage!(
                a_log,
                b.raw_f32("linear_attn.A_log", config.linear_num_value_heads)
            );
            stage!(
                dt_bias,
                b.raw_f32("linear_attn.dt_bias", config.linear_num_value_heads)
            );
            stage!(
                conv_weight,
                b.raw_f32(
                    "linear_attn.conv1d.weight",
                    qkv_dim * config.conv_kernel_dim,
                )
            );
            stage!(
                norm_weight,
                b.raw_f32("linear_attn.norm.weight", config.linear_value_head_dim)
            );
            stage!(wo, b.proj("linear_attn.out_proj", config.dim, d_inner));
            stage!(
                ffn_norm,
                b.norm("post_attention_layernorm.weight", &[config.dim])
            );
            stage!(ffn, load_moe(b, config, layer_idx));
            LayerWeights::DeltaNetMoe(DeltaNetMoeLayerWeights {
                attn_norm: take!(attn_norm),
                wqkv: take!(wqkv),
                wz: take!(wz),
                w_alpha: take!(w_alpha),
                w_beta: take!(w_beta),
                a_log: take!(a_log),
                dt_bias: take!(dt_bias),
                conv_weight: take!(conv_weight),
                norm_weight: take!(norm_weight),
                wo: take!(wo),
                ffn_norm: take!(ffn_norm),
                ffn: take!(ffn),
            })
        }
        (LayerType::FullAttention, true) => {
            stage!(attn_norm, b.norm("input_layernorm.weight", &[config.dim]));
            stage!(wq, b.proj("self_attn.q_proj", q_out_dim, config.dim));
            stage!(wk, b.proj("self_attn.k_proj", kv_dim, config.dim));
            stage!(wv, b.proj("self_attn.v_proj", kv_dim, config.dim));
            stage!(wo, b.proj("self_attn.o_proj", config.dim, o_in));
            stage!(
                q_norm,
                b.norm("self_attn.q_norm.weight", &[config.head_dim])
            );
            stage!(
                k_norm,
                b.norm("self_attn.k_norm.weight", &[config.head_dim])
            );
            stage!(
                ffn_norm,
                b.norm("post_attention_layernorm.weight", &[config.dim])
            );
            stage!(ffn, load_moe(b, config, layer_idx));
            LayerWeights::FullAttnMoe(FullAttnMoeLayerWeights {
                attn_norm: take!(attn_norm),
                wq: take!(wq),
                wk: take!(wk),
                wv: take!(wv),
                wo: take!(wo),
                q_norm: take!(q_norm),
                k_norm: take!(k_norm),
                ffn_norm: take!(ffn_norm),
                ffn: take!(ffn),
            })
        }
    };
    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hip_bridge::{HipError, HipResult};
    use rdna_compute::{DType, Gpu};

    struct FaultBackend {
        gpu: Gpu,
        calls: usize,
        fail_at: Option<usize>,
        freed: usize,
        live: usize,
    }

    impl FaultBackend {
        fn new(gpu: Gpu) -> Self {
            Self {
                gpu,
                calls: 0,
                fail_at: None,
                freed: 0,
                live: 0,
            }
        }

        fn next(&mut self) -> HipResult<()> {
            self.calls += 1;
            if self.fail_at == Some(self.calls) {
                Err(HipError::new(
                    0,
                    &format!("fault at layer operation {}", self.calls),
                ))
            } else {
                Ok(())
            }
        }

        fn alloc_tensor(&mut self) -> HipResult<GpuTensor> {
            let tensor = self.gpu.alloc_tensor(&[1], DType::F32)?;
            self.live += 1;
            Ok(tensor)
        }

        fn alloc_weight(&mut self, m: usize, k: usize) -> HipResult<WeightTensor> {
            Ok(WeightTensor {
                buf: self.alloc_tensor()?,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }

        fn alloc_moe(&mut self) -> HipResult<MoeFfnWeights> {
            let mut buffers = Vec::with_capacity(7);
            for _ in 0..7 {
                match self.alloc_tensor() {
                    Ok(tensor) => buffers.push(tensor),
                    Err(err) => {
                        for tensor in buffers.drain(..) {
                            self.free_tensor(tensor);
                        }
                        return Err(err);
                    }
                }
            }
            fn weight_from(buffers: &mut Vec<GpuTensor>) -> WeightTensor {
                WeightTensor {
                    buf: buffers.pop().expect("MoE test buffer"),
                    gpu_dtype: DType::F32,
                    m: 1,
                    k: 1,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                }
            }
            let (expert_execution_plan, expert_table, expert_binding) =
                crate::qwen35::weights::test_expert_binding().expect("test owner expert metadata");

            Ok(MoeFfnWeights {
                router: weight_from(&mut buffers),
                experts: Vec::new(),
                packed_expert_owners: None,
                shared_expert: crate::qwen35::SharedExpertWeights {
                    gate: weight_from(&mut buffers),
                    up: weight_from(&mut buffers),
                    down: weight_from(&mut buffers),
                },
                shared_expert_gate: weight_from(&mut buffers),
                expert_gate_up_ptrs: buffers.pop().expect("MoE test pointer buffer"),
                expert_down_ptrs: buffers.pop().expect("MoE test pointer buffer"),
                expert_down_awq_ptrs: None,
                cpu_deferred_experts: false,
                expert_dtype_tags: None,
                layer_idx: 0,
                expert_shape: None,
                paro_shared: None,
                global_expert_dtypes: None,
                ep_dummy_buffers: Vec::new(),
                ep_dummy_experts: Vec::new(),
                retired_expert_weights: Vec::new(),
                mixed_expert_gate_up_tiers: None,
                mixed_expert_down_tiers: None,
                expert_execution_plan,
                expert_table,
                expert_binding,
                expert_iu4: None,
                escha: None,
            })
        }

        fn assert_drained(&self) {
            assert_eq!(self.live, 0, "all GPU owners must be reclaimed");
        }
    }

    impl WeightBackend for FaultBackend {
        fn set_layer(&mut self, _layer: usize) {}

        fn proj(&mut self, _rel: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
            self.next()?;
            self.alloc_weight(m, k)
        }

        fn norm(&mut self, _rel: &str, _shape: &[usize]) -> HipResult<GpuTensor> {
            self.next()?;
            self.alloc_tensor()
        }

        fn raw_f32(&mut self, _rel: &str, _n: usize) -> HipResult<GpuTensor> {
            self.next()?;
            self.alloc_tensor()
        }

        fn bias(&mut self, _rel: &str, _n: usize) -> HipResult<GpuTensor> {
            Err(HipError::new(0, "test backend does not load biases"))
        }

        fn bias_opt(&mut self, _rel: &str, _n: usize) -> HipResult<Option<GpuTensor>> {
            Ok(None)
        }

        fn escha_sidecars(
            &mut self,
            _rel: &str,
            _w: &WeightTensor,
        ) -> HipResult<Option<(GpuTensor, GpuTensor, GpuTensor)>> {
            Ok(None)
        }

        fn zeros_i32(&mut self, _n: usize) -> HipResult<GpuTensor> {
            Err(HipError::new(0, "test backend does not build escha ids"))
        }

        fn iota_i32(&mut self, _n: usize) -> HipResult<GpuTensor> {
            Err(HipError::new(0, "test backend does not build escha iota"))
        }

        fn free_tensor(&mut self, tensor: GpuTensor) {
            self.gpu
                .free_tensor(tensor)
                .expect("test owner free must succeed");
            self.live = self.live.checked_sub(1).expect("owner freed twice");
            self.freed += 1;
        }
    }

    fn test_config(moe: bool) -> Qwen35Config {
        Qwen35Config {
            dim: 1,
            n_layers: 1,
            vocab_size: 1,
            norm_eps: 1e-5,
            eos_token: 0,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            rope_theta: 1.0,
            partial_rotary_factor: 1.0,
            is_vl_text: false,
            mrope_interleaved: false,
            mrope_section: [0; 3],
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 1,
            conv_kernel_dim: 1,
            hidden_dim: 1,
            num_experts: if moe { 1 } else { 0 },
            num_experts_per_tok: if moe { 1 } else { 0 },
            moe_intermediate_size: 1,
            shared_expert_intermediate_size: 1,
            has_shared_expert: moe,
            norm_topk_prob: false,
            layer_types: vec![LayerType::FullAttention],
            paged_experts: false,
            vram_budget_bytes: u64::MAX,
            reap_keep: None,
        }
    }

    fn no_moe(
        _backend: &mut FaultBackend,
        _config: &Qwen35Config,
        _layer: usize,
    ) -> HipResult<MoeFfnWeights> {
        Err(HipError::new(0, "dense test must not load MoE"))
    }

    fn free_weight<B: WeightBackend>(backend: &mut B, weight: WeightTensor) {
        let mut free = |tensor: GpuTensor| backend.free_tensor(tensor);
        free_weight_with(weight, &mut free);
    }

    fn free_test_layer<B: WeightBackend>(backend: &mut B, layer: LayerWeights) {
        match layer {
            LayerWeights::FullAttn(layer) => {
                let FullAttnLayerWeights {
                    attn_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    ffn_norm,
                    w_gate,
                    w_up,
                    w_down,
                    // The fault backend never produces escha sidecars or biases
                    // (both probes return None), so there is nothing to free.
                    ..
                } = layer;
                for tensor in [attn_norm, q_norm, k_norm, ffn_norm] {
                    backend.free_tensor(tensor);
                }
                for weight in [wq, wk, wv, wo, w_gate, w_up, w_down] {
                    free_weight(backend, weight);
                }
            }
            LayerWeights::FullAttnMoe(layer) => {
                let FullAttnMoeLayerWeights {
                    attn_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    ffn_norm,
                    ffn,
                } = layer;
                for tensor in [attn_norm, q_norm, k_norm, ffn_norm] {
                    backend.free_tensor(tensor);
                }
                for weight in [wq, wk, wv, wo] {
                    free_weight(backend, weight);
                }
                let mut free = |tensor: GpuTensor| backend.free_tensor(tensor);
                free_moe_ffn_with(ffn, &mut free);
            }
            _ => panic!("test helper only handles full-attention variants"),
        }
    }

    #[test]
    #[ignore = "requires a real HIP GPU"]
    fn dense_layer_failure_reclaims_owners_and_retry_succeeds() {
        let Some(gpu) = Gpu::init().ok() else {
            eprintln!("skip: no GPU");
            return;
        };
        let config = test_config(false);
        let mut backend = FaultBackend::new(gpu);
        backend.fail_at = Some(11);

        let failed = load_layer(&mut backend, &config, 0, no_moe);
        assert!(failed.is_err());
        assert_eq!(backend.calls, 11);
        assert_eq!(backend.freed, 10);
        backend.assert_drained();

        backend.calls = 0;
        backend.fail_at = None;
        let layer = load_layer(&mut backend, &config, 0, no_moe).expect("retry");
        assert_eq!(backend.live, 11);
        free_test_layer(&mut backend, layer);
        assert_eq!(backend.freed, 21);
        backend.assert_drained();
    }

    #[test]
    #[ignore = "requires a real HIP GPU"]
    fn moe_boundary_failure_reclaims_attention_owners_and_retry_succeeds() {
        let Some(gpu) = Gpu::init().ok() else {
            eprintln!("skip: no GPU");
            return;
        };
        let config = test_config(true);
        let mut backend = FaultBackend::new(gpu);

        let failed = load_layer(
            &mut backend,
            &config,
            0,
            |_backend: &mut FaultBackend,
             _config: &Qwen35Config,
             _layer: usize|
             -> HipResult<MoeFfnWeights> {
                Err(HipError::new(0, "injected late MoE failure"))
            },
        );
        assert!(failed.is_err());
        assert_eq!(backend.calls, 8);
        assert_eq!(backend.freed, 8);
        backend.assert_drained();

        backend.calls = 0;
        let layer = load_layer(&mut backend, &config, 0, |backend, _, _| {
            backend.alloc_moe()
        })
        .expect("MoE retry");
        assert_eq!(backend.live, 15);
        free_test_layer(&mut backend, layer);
        assert_eq!(backend.freed, 23);
        backend.assert_drained();
    }
}
