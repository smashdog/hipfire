// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// Model-owned GPU state operations: forward and speculative methods.

use super::*;

#[cfg(all(feature = "deltanet", feature = "kvmem"))]
use crate::kvmem_ctl::LegacyHeadPlane;
#[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
type LegacyHeadPlane = core::marker::PhantomData<()>;

#[cfg(all(feature = "deltanet", feature = "kvmem"))]
const KV_BLOCK_TOKENS: usize = crate::kvmem_ctl::KV_BLOCK_TOKENS;
#[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
const KV_BLOCK_TOKENS: usize = 128;

impl ModelRig {
    /// One batched forward step across the N slots (spec §5.2). Wraps
    /// `forward_slots::forward_batch_slots_graphed_opts` with every model
    /// field drawn from `self`; the capture is built internally from
    /// `any_verify`/`any_dflash_rows` so the engine loop never names model
    /// buffers. `gpu`/`pool`/`graph`/`batch`/`lm_head_skip` are the
    /// family-neutral engine pieces passed in by the caller. `layerwise` is the
    /// scoped per-query Layerwise carrier (design §3.9g): the caller builds it
    /// from a taken-out `KvMemCtl` for an admissible single-slot decode step and
    /// threads it here; Legacy / inadmissible steps pass `None` (byte-neutral).
    pub(super) fn forward_step(
        &mut self,
        gpu: &mut Gpu,
        batch: &SlotBatch,
        pool: &mut SlotPool,
        graph: &mut SlotDecodeGraph,
        lm_head_skip: &[bool],
        any_verify: bool,
        any_dflash_rows: bool,
        layerwise: crate::LayerwiseCarrierFwd<'_, '_>,
    ) -> rdna_compute::HipResult<()> {
        let mut capture = (any_verify || any_dflash_rows).then(|| {
            let tape = self
                .spec_verify_tape
                .as_mut()
                .expect("spec drafts require the verify tape");
            crate::forward_slots::SpecVerifyCapture {
                tape,
                verify_slots: lm_head_skip,
                stride: self.spec_rows,
                hidden: if any_dflash_rows {
                    self.dflash
                        .as_ref()
                        .map(|d| d.hidden_capture(self.config.dim))
                } else {
                    None
                },
            }
        });
        crate::forward_slots::forward_batch_slots_graphed_opts(
            gpu,
            &self.weights,
            &self.config,
            batch,
            pool,
            &mut self.dn_states,
            &self.k_arenas,
            &self.v_arenas,
            &mut self.desc_staging,
            &self.kv_tier,
            &self.pbs,
            &self.scratch,
            &self.logits_out,
            graph,
            self.spec_rows,
            lm_head_skip,
            capture.as_mut(),
            layerwise,
        )
    }
}

/// 期4: per-slot speculative counters, folded into
/// [`hipfire_runtime::serve::EngineStats::speculative`] when the request
/// terminates (nInfer completed-request statistics: cancellations count,
/// admission failures do not). `drafted` excludes the seed row and `accepted`
/// excludes the bonus row, following the `speculative.hpp` definitions.
impl ModelRig {
    /// Borrowed Q8 KvCache viewing this slot's contiguous K/V slabs.
    /// Sequential `forward_scratch_*` writes here; the batched path reads
    /// the same bytes via `legacy_base`. Views are `DeviceBuffer::Borrowed`
    /// — dropping the KvCache does not free the arenas.
    pub(super) fn slot_kv_view(&self, pool: &SlotPool, slot: SlotId) -> KvCache {
        // Paged slots have no contiguous slab: every legacy base is 0, so this
        // view would alias the arena prefix for ALL slots. Sequential writers
        // (VL, MTP prefill/replay) must be gated off under paged KV at
        // activation — if one reaches here, fail loudly rather than corrupt.
        assert!(
            !pool.is_paged(),
            "slot_kv_view: flat slab view requested under a paged pool — a \
             sequential KV writer would alias every slot onto the arena prefix"
        );
        assert!(
            self.kv_tier.is_q8(),
            "slot_kv_view: this view hardwires the q8 layout, but the engine \
             runs the {:?} KV tier — a sequential write here would corrupt the \
             rotated-K arena",
            self.kv_tier.mode
        );
        let cap = pool.cap_tokens();
        let per_pos_bytes = self.config.n_kv_heads * (self.config.head_dim / 32) * 34;
        let base = pool.descriptors()[slot.0].legacy_k_base as usize;
        let span = cap * per_pos_bytes;
        let mut k_gpu = Vec::with_capacity(self.config.n_layers);
        let mut v_gpu = Vec::with_capacity(self.config.n_layers);
        let mut fa = 0usize;
        for t in &self.config.layer_types {
            if *t == LayerType::FullAttention {
                k_gpu.push(self.k_arenas[fa].sub_offset(base, span));
                v_gpu.push(self.v_arenas[fa].sub_offset(base, span));
                fa += 1;
            } else {
                k_gpu.push(GpuTensor::null_for_test());
                v_gpu.push(GpuTensor::null_for_test());
            }
        }
        KvCache {
            k_gpu,
            v_gpu,
            k_scales: vec![],
            v_scales: vec![],
            kv_dim: self.config.n_kv_heads * self.config.head_dim,
            max_seq: cap,
            physical_cap: cap,
            n_kv_heads: self.config.n_kv_heads,
            head_dim: self.config.head_dim,
            quantized: true,
            quant_q8: true,
            quant_int8: false,
            quant_hfq4: false,
            quant_asym4: false,
            quant_asym3: false,
            quant_asym2: false,
            quant_fwht: false,
            quant_bf16: false,
            quant_fp8: false,
            boundary_layers: 0,
            givens_cos: None,
            givens_sin: None,
            layer_is_boundary: vec![],
            compact_offset: 0,
            v_mode: VMode::Q8,
        }
    }
}

/// Sequential VL step: encode the image if needed, prefill/decode every
/// remaining token with M-RoPE, then sample the next token from scratch.logits.
///
/// VL slots stay on this path for the whole request because batched 1D RoPE
/// disagrees with M-RoPE on image tokens and on decode (`t=h=w = pos+delta`).
impl ModelRig {
    /// Sequential VL step: encode the image if needed, prefill/decode every
    /// remaining token with M-RoPE, then sample the next token from
    /// `scratch.logits`. VL slots stay on this path for the whole request
    /// because batched 1D RoPE disagrees with M-RoPE on image tokens and on
    /// decode (`t=h=w = pos+delta`).
    pub(super) fn vl_forward_remaining(
        &mut self,
        mut gpu: &mut Gpu,
        pool: &mut SlotPool,
        tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
        sample_params: &mut [SlotSampleParams],
        slot: SlotId,
        work: &mut PendingWork,
        image_pad_id: u32,
        mut grammar: Option<&mut GrammarConstraint>,
    ) -> Result<u32, String> {
        let mut vl = work
            .vl_prefill
            .take()
            .ok_or_else(|| "vl_forward_remaining: slot has no VL state".to_string())?;
        let first_step = vl.embeddings.is_empty();
        if first_step {
            let weights = self
                .vision_weights
                .as_ref()
                .ok_or_else(|| "VL request but model has no vision encoder".to_string())?;
            let config = self
                .vision_config
                .as_ref()
                .ok_or_else(|| "VL request but model has no vision config".to_string())?;
            let emb = hipfire_arch_qwen35_vl::qwen35_vl::vision_forward(
                &mut gpu,
                weights,
                config,
                &vl.patches,
                vl.grid_h,
                vl.grid_w,
            )
            .map_err(|e| format!("vision_forward: {e}"))?;
            vl.embeddings = emb;
            vl.dim = self.config.dim;
            vl.patches.clear();
            hipfire_runtime::llama::reset_cpu_sampler_rng(sample_params[slot.0].seed);
        }

        let prompt_tokens = std::mem::take(&mut work.remaining_prompt);
        let mut pos = work.next_pos;
        let mut kv_cache = self.slot_kv_view(pool, slot);
        let mrope_ctx = qwen35::MropeCtx::new(
            &self.config,
            vl.base,
            vl.mrope_positions.clone(),
            vl.rope_delta,
        );
        let mut visual_idx = vl.visual_idx;
        for &token in &prompt_tokens {
            if token == image_pad_id && visual_idx < vl.n_visual_tokens {
                let dim = vl.dim;
                let start = visual_idx * dim;
                let emb = vl
                    .embeddings
                    .get(start..start + dim)
                    .ok_or_else(|| "VL embedding slice out of range".to_string())?;
                qwen35::forward_scratch_embed_mrope(
                    &mut gpu,
                    &self.weights,
                    &self.config,
                    emb,
                    pos,
                    &mut kv_cache,
                    &mut self.dn_states[slot.0],
                    &self.scratch,
                    Some(&mrope_ctx),
                )
                .map_err(|e| format!("VL prefill embed: {e}"))?;
                visual_idx += 1;
            } else {
                qwen35::forward_scratch_mrope(
                    &mut gpu,
                    &self.weights,
                    &self.config,
                    token,
                    pos,
                    &mut kv_cache,
                    &mut self.dn_states[slot.0],
                    &self.scratch,
                    Some(&mrope_ctx),
                )
                .map_err(|e| format!("VL forward: {e}"))?;
            }
            pos += 1;
        }
        vl.visual_idx = visual_idx;
        work.next_pos = pos;
        pool.set_seq_len(slot, pos)
            .map_err(|e| format!("VL set_seq_len: {e}"))?;
        work.vl_prefill = Some(vl);

        let mut logits = gpu
            .download_f32(&self.scratch.logits)
            .map_err(|e| format!("VL download logits: {e}"))?;
        // Hard grammar mask BEFORE the host sampler (spec §7.2 G2 — the same
        // contract the batched path enforces on logits_out). This sequential
        // path used to sample unconstrained: a schema-violating first token
        // could leave the matcher accepting and sail through as a success.
        if let Some(constraint) = grammar.as_deref_mut() {
            let mask = constraint
                .build_mask(&tokenizer, self.config.vocab_size)
                .map_err(|e| format!("VL grammar mask: {e}"))?;
            for (i, &allowed) in mask.iter().enumerate() {
                if !allowed && i < logits.len() {
                    logits[i] = f32::NEG_INFINITY;
                }
            }
        }
        let sp = &sample_params[slot.0];
        let cfg = hipfire_runtime::sampler::SamplerConfig {
            temperature: sp.temperature,
            top_p: sp.top_p,
            // Apply the REQUESTED penalties/min_p: this sequential path used to
            // hardcode neutral values, silently downgrading a penalized VL
            // request's semantics (spec §6 X2: "An optimization bypass is
            // allowed, a silent semantic downgrade is not").
            repeat_penalty: sp.repeat_penalty,
            repeat_window: sp.repeat_window.max(0) as usize,
            presence_penalty: sp.presence_penalty,
            frequency_penalty: sp.frequency_penalty,
            blocked_tokens: Vec::new(),
            top_k: if sp.top_k > 0 {
                Some(sp.top_k as u32)
            } else {
                None
            },
            min_p: (sp.min_p > 0.0).then_some(sp.min_p),
        };
        Ok(hipfire_runtime::sampler::sample_cpu(&mut logits, &[], &cfg))
    }
}

/// Advance slot `s`'s batched-VL vision-tower encode by ONE layer, starting
/// the [`VisionTowerJob`] on the first call and finishing it (merger epilogue
/// + ext-embedding upload) on the last.
///
/// Why one layer at a time: this engine thread is the GPU's exclusive owner,
/// so a monolithic `vision_forward` wedges every OTHER slot's decode step for
/// the whole tower pass — measured ~9.7 s at a 78×78 grid (6084 patches) even
/// on the Q-tiled attention kernel, i.e. the whole serve freezes for the
/// duration of one big image request. One tower layer (~0.4 s at that grid)
/// is the largest GPU chunk a VL slot adds between other slots' steps, and
/// the scheduler already holds the VL slot's rows back until its embeddings
/// exist (`vl_slots_waiting_on_vision_forward_are_skipped`).
///
/// `job` is the caller-owned per-slot job slot. Contract on return:
/// `Ok(false)` → `job` is `Some` (encode continues next iteration);
/// `Ok(true)` → encode finished, embeddings spliced, `job` is `None`;
/// `Err` → the job was freed, `job` is `None`.
impl ModelRig {
    /// Advance slot `s`'s vision-tower job by one tower layer, creating it on
    /// the first call from `vl.patches`. `Ok(true)` when the encode finished
    /// and the slot's ext-embedding matrix is uploaded into
    /// `vl_ext_devs[s]`; `Ok(false)` = still mid-encode (returns to the
    /// scheduler so other slots keep decoding). `Err` → the job was freed.
    pub(super) fn vision_tower_step(
        &mut self,
        gpu: &mut Gpu,
        s: usize,
        vl: &mut VlPrefill,
    ) -> Result<bool, String> {
        let vision_weights = &self.vision_weights;
        let vision_config = &self.vision_config;
        let vl_ext_devs = &mut self.vl_ext_devs;
        let config = &self.config;
        let job = &mut self.vl_tower_jobs[s];
        if job.is_none() {
            let weights = vision_weights
                .as_ref()
                .ok_or("VL request but model has no vision encoder")?;
            let vconfig = vision_config
                .as_ref()
                .ok_or("VL request but model has no vision config")?;
            let started = hipfire_arch_qwen35_vl::qwen35_vl::VisionTowerJob::new(
                gpu,
                weights,
                vconfig,
                &vl.patches,
                vl.grid_h,
                vl.grid_w,
            )
            .map_err(|e| format!("vision_forward: {e}"))?;
            *job = Some(started);
            // Patches now live on-device for the rest of the encode; free the
            // host copy (~48 MB at the 2 MP budget).
            vl.patches = Vec::new();
        }
        let weights = vision_weights
            .as_ref()
            .ok_or("VL request but model has no vision encoder")?;
        let vconfig = vision_config
            .as_ref()
            .ok_or("VL request but model has no vision config")?;
        let done = match job
            .as_mut()
            .expect("vision tower job present")
            .step_layer(gpu, weights, vconfig)
        {
            Ok(done) => done,
            Err(e) => {
                let reason = format!("vision_forward: {e}");
                if let Some(j) = job.take() {
                    let _ = j.free(gpu);
                }
                return Err(reason);
            }
        };
        if !done {
            return Ok(false);
        }
        let j = job.take().expect("vision tower job present");
        // `finish` consumes the job and frees its own device buffers on every
        // error path (the tower's working set is neither pooled nor released by
        // Drop, so a failed merger epilogue must not leak it).
        let emb = j
            .finish(gpu, weights, vconfig)
            .map_err(|e| format!("vision_forward: {e}"))?;
        let dev = gpu
            .zeros(&[emb.len()], DType::F32)
            .map_err(|e| format!("vl ext alloc: {e}"))?;
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(emb.as_ptr() as *const u8, emb.len() * 4) };
        if let Err(e) = gpu.hip.memcpy_htod(&dev.buf, bytes) {
            let _ = gpu.free_tensor(dev);
            return Err(format!("vl ext upload: {e}"));
        }
        *vl_ext_devs.get_mut(s).expect("vl_ext_devs slot") = Some(dev);
        vl.embeddings = emb;
        vl.dim = config.dim;
        Ok(true)
    }
}

/// Free slot `s`'s VL GPU state (ext-embedding matrix + any tower job still
/// mid-encode). For every site that clears a slot's work: dropping either
/// without freeing leaks VRAM — the buffers are neither pooled nor hipFree'd
/// by Drop.
impl ModelRig {
    /// Free slot `s`'s VL GPU state (ext-embedding matrix + any tower job
    /// still mid-encode). For every site that clears a slot's work: dropping
    /// either without freeing leaks VRAM — the buffers are neither pooled
    /// nor hipFree'd by Drop.
    pub(super) fn clear_slot_vl_state(&mut self, gpu: &mut Gpu, s: usize) {
        if let Some(dev) = self.vl_ext_devs[s].take() {
            let _ = gpu.free_tensor(dev);
        }
        if let Some(job) = self.vl_tower_jobs[s].take() {
            let _ = job.free(gpu);
        }
    }
}

/// MTP head-KV fill for one scheduler prefill chunk, post-forward.
///
/// The trunk side of an MTP slot's prompt flows through the scheduler's
/// batched chunk path like any other prefill (1D RoPE, shared forward with
/// other slots). This runs right AFTER that forward: the chunk's hidden
/// rows sit in `pbs.x_batch`, and the head consumes exactly
/// `(embed(token_i), trunk_hidden(token_i))` per row, so one
/// `mtp_head_forward_block_batched(kv_only = true)` call fills the head's
/// private KV for the chunk. `state.prev_hidden` is left pointing at the
/// chunk's last row — the correct draft seed hidden once the prompt drains.
///
/// `row_off` is the slot's first row in the step batch; `m` its row count.
impl ModelRig {
    pub(super) fn mtp_head_prefill_chunk(
        &mut self,
        mut gpu: &mut Gpu,
        slot: SlotId,
        chunk_tokens: &[u32],
        chunk_positions: &[i32],
        row_off: usize,
        m: usize,
    ) -> Result<(), String> {
        let head = self
            .mtp_head
            .as_ref()
            .ok_or_else(|| "mtp_head_prefill_chunk: MTP head not loaded".to_string())?;
        #[cfg(all(feature = "deltanet", feature = "kvmem"))]
        let head_live = self.kvmem.as_ref().is_some_and(|c| c.head_live_enabled());
        #[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
        let head_live = false;
        let head_cap = if self.mtp_states[slot.0].is_none() && !head_live {
            kvmem_hooks::mtp_head_capacity(self)
        } else {
            None
        };
        let rope_shift = kvmem_hooks::mtp_head_prefill_shift(self, chunk_positions[0] as usize + m);
        let Some((scratch, rot)) = self.mtp_prefill_batched.as_mut() else {
            return Err("mtp_head_prefill_chunk: batched head scratch missing".to_string());
        };
        if self.mtp_states[slot.0].is_none() {
            // P4.1② f2 (E): bound the head KV under Layerwise+MTP; Legacy / AR-only
            // get None ⇒ full-resident (byte-neutral). P1b plane-only keeps just
            // a 128-token placeholder and writes every row into the paged arena.
            let state = if head_live {
                crate::mtp_spec::MtpSpecState::new_for_components_paged_only(
                    &mut gpu,
                    &self.config,
                    &self.dn_states[slot.0],
                    head,
                    self.mtp_k,
                    crate::mtp_head::MtpKvMode::Q8,
                )
                .map_err(|e| format!("mtp state alloc: {e}"))?
            } else {
                crate::mtp_spec::MtpSpecState::new_for_components(
                    &mut gpu,
                    &self.config,
                    &self.dn_states[slot.0],
                    head,
                    self.mtp_k,
                    crate::mtp_head::MtpKvMode::Q8,
                    head_cap,
                )
                .map_err(|e| format!("mtp state alloc: {e}"))?
            };
            self.mtp_states[slot.0] = Some(state);
        }
        if let Some(cvs) = head.weights.compressed_vocab_size {
            let state = self.mtp_states[slot.0].as_mut().unwrap();
            state
                .ensure_compressed_lm_logits(&mut gpu, cvs)
                .map_err(|e| format!("ensure compressed logits: {e}"))?;
            // The draft phase reads `mtp_scratch.logits_compressed` (head-side
            // buffer) — distinct from `mtp_lm_logits_compressed` above. Without
            // this allocation the first draft step on a compressed head panics
            // the engine thread (the sequential speculator was the only caller
            // of `ensure_compressed_logits`).
            state
                .mtp_scratch
                .ensure_compressed_logits(&mut gpu, cvs)
                .map_err(|e| format!("ensure head compressed logits: {e}"))?;
        }

        let dim = self.config.dim;
        let mut state = self.mtp_states[slot.0]
            .take()
            .expect("just ensured it exists");

        // The head consumes POST-output-norm trunk hidden (the convention it was
        // exported with and the one the sequential path feeds it); pbs.x_batch
        // holds the PRE-norm residual, so norm the chunk's rows into staging.
        let hidden_rows = self.pbs.x_batch.sub_offset(row_off * dim, m * dim);
        let staged = self.mtp_prefill_hidden.sub_offset(0, m * dim);
        let norm_outcome = gpu
            .rmsnorm_batched(
                &hidden_rows,
                &self.weights.output_norm,
                &staged,
                m,
                dim,
                self.config.norm_eps,
            )
            .map_err(|e| format!("mtp prefill hidden norm: {e}"));

        // P4.1② f2 (E): bounded head staging under Layerwise+MTP. The resident
        // head cache is a per-chunk archive staging buffer (window + one scratch
        // block), NOT the attention source (the G4 carrier is). Write prompt K/V
        // at COMPACT pool-local positions (global - removed) and rotate at GLOBAL
        // (rope_shift = removed), mirroring ninfer cache_position_shift (NL:8442).
        // Legacy / AR-only: removed = 0 ⇒ compact == global, rope_shift = 0
        // (byte-neutral). Each chunk re-stages independently — no slide.
        let compact_positions: Vec<i32> = if rope_shift == 0 {
            chunk_positions.to_vec()
        } else {
            // Fail closed if the chunk straddles the window front (compact < 0):
            // the scheduler's layerwise prefill credit bounds m <= window, so this
            // guards a mis-sized chunk, never a normal path.
            if (chunk_positions[0] as usize) < rope_shift {
                self.mtp_states[slot.0] = Some(state);
                return Err(format!(
                    "mtp head prefill: chunk at {} exceeds head window (removed {rope_shift})",
                    chunk_positions[0]
                ));
            }
            chunk_positions
                .iter()
                .map(|&p| p - rope_shift as i32)
                .collect()
        };

        let head_cap = state.mtp_kv.max_seq;
        if let Some(&last) = compact_positions.last() {
            if last < 0 || last as usize >= head_cap {
                self.mtp_states[slot.0] = Some(state);
                return Err(format!(
                    "mtp head prefill: compact position {last} outside head cache {head_cap}"
                ));
            }
        }
        // P1b: mirror the chunk's K/V into the paged head arena as well. The
        // arena is restored at the chunk's global start (cold 0 or a prefix
        // checkpoint), then `stage_rows` maps this span and the batched forward
        // writes it alongside the contiguous archive staging. The first draft
        // after the prompt then restores with no archive copy.
        #[cfg(all(feature = "deltanet", feature = "kvmem"))]
        let mut legacy_plane: Option<crate::kvmem_ctl::LegacyHeadPlane> =
            if self.kvmem.as_ref().is_some_and(|c| c.head_live_enabled()) {
                let (kv_mode, window_pages) = {
                    let ctl = self.kvmem.as_ref().unwrap();
                    // P1b: pass only the retained trunk window. The plane itself
                    // adds the sink and draft spare page. A 1024-row prefill
                    // chunk spans at most eight pages, so its start page stays
                    // retained while `stage_rows` remaps to the chunk end.
                    let window = ctl.budget_blocks();
                    (
                        ctl.kv_mode(),
                        u32::try_from(window)
                            .map_err(|_| format!("mtp head plane: window {window} overflow"))?,
                    )
                };
                if window_pages == 0 {
                    self.mtp_states[slot.0] = Some(state);
                    return Err("mtp head plane: head window is empty".to_string());
                }
                let mut plane = match self.legacy_head_planes[slot.0].take() {
                    Some(p) => p,
                    None => crate::kvmem_ctl::LegacyHeadPlane::new(
                        &mut gpu,
                        kv_mode,
                        head.config.n_head_kv,
                        head.config.head_dim,
                        window_pages,
                    )?,
                };
                let start = u32::try_from(chunk_positions[0]).map_err(|_| {
                    format!("mtp head plane: start {} overflow", chunk_positions[0])
                })?;
                let restored = {
                    let ctl = self.kvmem.as_ref().unwrap();
                    plane.restore(&mut gpu, ctl, slot.0, start)
                };
                match restored {
                    Ok(()) => Some(plane),
                    Err(e) => {
                        let _ = plane.free(&mut gpu);
                        return Err(format!("mtp head prefill paged restore: {e}"));
                    }
                }
            } else {
                None
            };
        #[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
        let mut legacy_plane: Option<LegacyHeadPlane> = None;
        // Disjoint field borrows: `scratch`/`rot` from mtp_prefill_batched,
        // `gpu`/`weights`/head separately. No closure — it would capture all of
        // `rig` and collide with the scratch borrow.
        let mtp_outcome = norm_outcome.and_then(|()| {
            // P1b: block-quantized Legacy shift — move the live window before
            // this chunk stages its compact rows.
            if rope_shift > 0 && rope_shift % KV_BLOCK_TOKENS == 0 {
                state
                    .mtp_kv
                    .slide_to_shift(&mut gpu, rope_shift)
                    .map_err(|e| format!("mtp head slide: {e}"))?;
            }
            #[cfg(all(feature = "deltanet", feature = "kvmem"))]
            {
                let start = chunk_positions[0] as u32;
                let end = start.saturating_add(m as u32);
                let mut paged_view = match legacy_plane.as_mut() {
                    Some(plane) => match plane.stage_rows(&mut gpu, start, m as u32, end) {
                        Ok(view) => Some(view),
                        Err(e) => {
                            return Err(format!(
                                "mtp head prefill paged stage slot={slot:?} pos={start} rows={m}: {e}"
                            ));
                        }
                    },
                    None => None,
                };
                let res = crate::mtp_head::mtp_head_forward_block_batched(
                    &mut gpu,
                    head,
                    scratch,
                    &mut state.mtp_kv,
                    chunk_tokens,
                    &staged,
                    &compact_positions,
                    m,
                    &self.weights,
                    Some(rot),
                    /* kv_only */ true,
                    /* rope_shift */ rope_shift as i32,
                    paged_view.as_ref(),
                )
                .map_err(|e| format!("mtp head prefill: {e}"));
                drop(paged_view);
                if res.is_ok() {
                    if let Some(plane) = legacy_plane.as_mut() {
                        plane.mark_filled(end);
                    }
                }
                res
            }
            #[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
            {
                crate::mtp_head::mtp_head_forward_block_batched(
                    &mut gpu,
                    head,
                    scratch,
                    &mut state.mtp_kv,
                    chunk_tokens,
                    &staged,
                    &compact_positions,
                    m,
                    &self.weights,
                    Some(rot),
                    /* kv_only */ true,
                    /* rope_shift */ rope_shift as i32,
                    None,
                )
                .map_err(|e| format!("mtp head prefill: {e}"))
            }
        });
        // Seed hidden for the first draft cycle: the (normed) trunk hidden of
        // the chunk's last token — same convention as the verify-path capture.
        let hidden_outcome = mtp_outcome.and_then(|()| {
            gpu.hip
                .memcpy_dtod_at(
                    &state.prev_hidden.buf,
                    0,
                    &self.mtp_prefill_hidden.buf,
                    (m - 1) * dim * 4,
                    dim * 4,
                )
                .map_err(|e| format!("mtp prev_hidden capture: {e:?}"))
        });
        if hidden_outcome.is_ok() {
            state.prev_hidden_pos = Some(chunk_positions[m - 1] as usize);
        }
        // P4.1② f2 (B+E): the head archive bootstrap. Under Layerwise+MTP,
        // co-archive this chunk's head K/V into the composite aux plane (ordinal
        // `n_fa`) so every prompt token's head payload survives for drop-evict +
        // stripe restore-before-attend (G4). The trunk Main plane is archived
        // separately by `feed_rows` (which write-through archives whenever the
        // strategy is Layerwise, regardless of the prefill arm). The bounded head
        // staging cache is position-major at the shared trunk Q8 stride; E writes
        // it at COMPACT positions, so the gather offset is the pool-local
        // `pos0 - rope_shift` while the canonical archive seal stays at the GLOBAL
        // `pos0`. `ensure_head_plane` is idempotent (session-lifetime map). Legacy /
        // AR-only: `head_plane_enabled()` is false ⇒ no allocation, no seal, and
        // rope_shift == 0 ⇒ gather offset == pos0 (byte-neutral).
        #[cfg_attr(not(feature = "kvmem"), allow(unused_mut))]
        let mut outcome = hidden_outcome;
        #[cfg(feature = "kvmem")]
        if outcome.is_ok() {
            let head_plane = kvmem_hooks::head_composite_enabled(self);
            if head_plane {
                let pos0 = chunk_positions[0] as usize;
                let mut kvmem = self.kvmem.take();
                let res = match kvmem.as_mut() {
                    Some(ctl) => ctl.ensure_head_plane(slot.0).and_then(|()| {
                        match legacy_plane.as_ref() {
                            Some(plane) => {
                                ctl.head_write_through_archive_paged(gpu, plane, slot.0, pos0, m)?
                            }
                            None => ctl.head_write_through_archive(
                                gpu,
                                &state.mtp_kv.inner.k_gpu[0],
                                &state.mtp_kv.inner.v_gpu[0],
                                slot.0,
                                pos0,
                                m,
                                // E: gather from the COMPACT staging offset; the seal
                                // above stays at the GLOBAL canonical pos0.
                                pos0 - rope_shift,
                            )?,
                        }
                        // f2-D: grow the head resident selection map with this
                        // chunk's accepted canonical pages (ninfer prefill
                        // `initialize`/append, NL:12274/12309). Full-resident
                        // transitional: every prompt page stays resident, so the
                        // map reconstructs the frontier with removed == 0; the
                        // consume pass (mtp_verify_accept_step) appends onto it.
                        ctl.accept_head_append(slot.0, pos0 as u64, (pos0 + m) as u64)
                    }),
                    None => Err("mtp head prefill: controller vanished".to_string()),
                };
                self.kvmem = kvmem;
                if let Err(e) = res {
                    outcome = Err(e);
                }
            }
        }
        #[cfg(all(feature = "deltanet", feature = "kvmem"))]
        {
            self.legacy_head_planes[slot.0] = legacy_plane;
        }
        self.mtp_states[slot.0] = Some(state);
        outcome
    }
}

/// MTP draft phase: run K serial head-forward steps, save DN snapshot.
///
/// Called when `work.decoding` (subsequent MTP steps). Takes the seed from
/// `work.remaining_prompt.last()`. Saves DN snapshot to `state.trunk_snap`
/// for rollback during verify. Returns the draft output for the batched
/// verify phase.
impl ModelRig {
    /// MTP draft phase: run K serial head-forward steps, save DN snapshot.
    ///
    /// Called when `work.decoding` (subsequent MTP steps). Takes the seed
    /// from `work.remaining_prompt.last()`. Saves DN snapshot to
    /// `state.trunk_snap` for rollback during verify. Returns the draft
    /// output for the batched verify phase.
    pub(super) fn mtp_draft_step(
        &mut self,
        mut gpu: &mut Gpu,
        slot: SlotId,
        work: &mut PendingWork,
        emitted: &[u32],
        session_id: u64,
        cap_tokens: usize,
    ) -> Result<crate::mtp_spec::MtpDraftOutput, String> {
        // The head-layerwise restore (C) is the only consumer of these; silence
        // them in the no-kvmem default build where the draft stays full-resident.
        #[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
        let _ = (session_id, cap_tokens);
        let head = self
            .mtp_head
            .as_ref()
            .ok_or_else(|| "mtp_draft_step: MTP head not loaded".to_string())?;

        if self.mtp_states[slot.0].is_none() {
            // P4.1② f2 (E): bound the head KV under Layerwise+MTP; Legacy / AR-only
            // get None ⇒ full-resident (byte-neutral). P1b plane-only keeps just
            // a 128-token placeholder and writes every row into the paged arena.
            #[cfg(all(feature = "deltanet", feature = "kvmem"))]
            let head_live = self.kvmem.as_ref().is_some_and(|c| c.head_live_enabled());
            #[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
            let head_live = false;
            let head_cap = if head_live {
                None
            } else {
                kvmem_hooks::mtp_head_capacity(self)
            };
            let state = if head_live {
                crate::mtp_spec::MtpSpecState::new_for_components_paged_only(
                    &mut gpu,
                    &self.config,
                    &self.dn_states[slot.0],
                    head,
                    self.mtp_k,
                    crate::mtp_head::MtpKvMode::Q8,
                )
                .map_err(|e| format!("mtp state alloc: {e}"))?
            } else {
                crate::mtp_spec::MtpSpecState::new_for_components(
                    &mut gpu,
                    &self.config,
                    &self.dn_states[slot.0],
                    head,
                    self.mtp_k,
                    crate::mtp_head::MtpKvMode::Q8,
                    head_cap,
                )
                .map_err(|e| format!("mtp state alloc: {e}"))?
            };
            self.mtp_states[slot.0] = Some(state);
        }
        let mut state = self.mtp_states[slot.0]
            .take()
            .expect("just ensured it exists");

        if let Some(cvs) = head.weights.compressed_vocab_size {
            if let Err(e) = state.ensure_compressed_lm_logits(&mut gpu, cvs) {
                self.mtp_states[slot.0] = Some(state);
                return Err(format!("ensure compressed logits: {e}"));
            }
            // Head-side compressed buffer for the draft phase (see the alloc
            // comment in `mtp_head_prefill_chunk`).
            if let Err(e) = state.mtp_scratch.ensure_compressed_logits(&mut gpu, cvs) {
                self.mtp_states[slot.0] = Some(state);
                return Err(format!("ensure head compressed logits: {e}"));
            }
        }

        // Install this request's sampling/penalty config (parked by admit;
        // a cold admit's state only exists from now on). `set_sampling`
        // reseeds the draft RNG, so it runs exactly once per request — the
        // per-window penalty history is rebuilt from `emitted` below instead.
        let mut req_sampling = self.spec_req_sampling[slot.0].take();
        if let Some(cfg) = req_sampling.as_mut() {
            if !cfg.applied {
                state.set_sampling(cfg.sampling, cfg.rng_seed);
                if cfg.sampling.penalized() {
                    state.penalty = hipfire_runtime::spec_sampling::PenaltyHistory::new(
                        cfg.sampling.repeat_window,
                    );
                    // Generation-only history: the prompt tail seeds row 0,
                    // emitted tokens are added per window, exactly like AR's
                    // `sess.tokens` tail window.
                    state.penalty.set_prompt(&cfg.prompt_tail);
                }
                cfg.applied = true;
            }
            if cfg.sampling.penalized() {
                // Row k of this window scopes `suffix_W(emitted ‖ c[..k])`.
                state.penalty.begin_window(emitted);
            }
        }

        let prompt_tokens = std::mem::take(&mut work.remaining_prompt);
        let pos = work.next_pos;

        // P4.1② f2 (C): restore the head's bounded aux arena before the draft
        // attends (ninfer NL:10274-10339). The controller is taken out so the
        // head map can be driven mutably while the draft closure holds shared
        // borrows of `self.weights`/`config`/`mtp_head`. The restore helper runs
        // the draft *inside* the carrier arm (verify-pattern borrow confinement);
        // `None` (Legacy / AR-only / cold start / unbound aux) keeps the
        // full-resident draft byte-neutral.
        #[cfg(all(feature = "deltanet", feature = "kvmem"))]
        type HeadLw<'a> = crate::layerwise_driver::HeadLayerwise<'a>;
        #[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
        type HeadLw<'a> = core::marker::PhantomData<&'a ()>;

        // P1b: Legacy's bounded live head is addressed by COMPACT KV rows while
        // RoPE stays on the global position. The shift is fixed for this draft
        // span; a span that would start before the window fails loudly instead
        // of aliasing another page.
        #[cfg(all(feature = "deltanet", feature = "kvmem"))]
        let head_shift = self
            .kvmem
            .as_ref()
            .filter(|ctl| ctl.head_composite_enabled())
            .map(|ctl| ctl.head_prefill_shift(pos + self.mtp_k))
            .unwrap_or(0);
        #[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
        let head_shift = 0usize;

        // P1b: rebuild the paged Legacy head's sink + recent window from the
        // composite aux plane before the draft attends. The plane is taken out
        // of the rig so the closure can borrow it disjointly from `state`.
        #[cfg(all(feature = "deltanet", feature = "kvmem"))]
        let mut legacy_plane: Option<crate::kvmem_ctl::LegacyHeadPlane> =
            if self.kvmem.as_ref().is_some_and(|c| c.head_live_enabled()) {
                let (kv_mode, window_pages) = {
                    let ctl = self.kvmem.as_ref().unwrap();
                    // The plane adds the sink and draft spare to this retained
                    // trunk window; see the prefill plane above.
                    let window = ctl.budget_blocks();
                    (
                        ctl.kv_mode(),
                        u32::try_from(window)
                            .map_err(|_| format!("mtp head plane: window {window} overflow"))?,
                    )
                };
                if window_pages == 0 {
                    return Err("mtp head plane: head window is empty".to_string());
                }
                let mut plane = match self.legacy_head_planes[slot.0].take() {
                    Some(p) => p,
                    None => crate::kvmem_ctl::LegacyHeadPlane::new(
                        &mut gpu,
                        kv_mode,
                        head.config.n_head_kv,
                        head.config.head_dim,
                        window_pages,
                    )?,
                };
                let restored = {
                    let ctl = self.kvmem.as_ref().unwrap();
                    plane.restore(
                        &mut gpu,
                        ctl,
                        slot.0,
                        u32::try_from(pos).unwrap_or(u32::MAX),
                    )
                };
                match restored {
                    Ok(()) => Some(plane),
                    Err(e) => {
                        let _ = plane.free(&mut gpu);
                        if state.mtp_kv.paged_only() {
                            self.mtp_states[slot.0] = Some(state);
                            return Err(format!("mtp head paged-only restore: {e}"));
                        }
                        // No sealed aux page yet (cold start / short prompt):
                        // keep the historical contiguous path authoritative
                        // until the archive has something to page in.
                        eprintln!(
                            "[kvmem-head] paged restore skipped slot={slot:?} pos={pos}: {e}"
                        );
                        None
                    }
                }
            } else {
                None
            };
        #[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
        let mut legacy_plane: Option<LegacyHeadPlane> = None;

        // The draft body, factored so the restore helper can run it through the
        // head handle (kvmem) or full-resident (fallback / default build). `gpu`
        // and the handle are parameters; the rest is captured disjointly.
        let mut run_draft = |gpu: &mut Gpu,
                             hlw: Option<&mut HeadLw<'_>>,
                             _plane: Option<&mut LegacyHeadPlane>|
         -> Result<crate::mtp_spec::MtpDraftOutput, String> {
            if prompt_tokens.is_empty() {
                return Err("mtp_draft_step: no seed token".to_string());
            }
            let seed = *prompt_tokens.last().unwrap();

            // Save DN snapshot for rollback during verify phase.
            state
                .trunk_snap
                .save_from(&mut self.dn_states[slot.0], gpu)
                .map_err(|e| format!("mtp dn snapshot: {e}"))?;

            // P1b: slide the bounded Legacy window once, before the draft
            // (and before any proposal graph capture — a host copy inside a
            // captured region would not replay).
            if hlw.is_none() && head_shift > 0 {
                state
                    .mtp_kv
                    .slide_to_shift(gpu, head_shift)
                    .map_err(|e| format!("mtp head draft slide: {e}"))?;
            }
            let draft_outcome = crate::mtp_spec::mtp_draft_phase_inner(
                gpu,
                &self.weights,
                &self.config,
                head,
                &mut state,
                pos,
                seed,
                self.mtp_k,
                /* skip_proposal_graph */ true,
                hlw,
                legacy_plane.as_mut(),
                head_shift,
            )
            .map_err(|e| format!("mtp draft: {e}"));
            if crate::mtp_spec::mtp_trace_enabled() {
                eprintln!("[mtp-trace] draft done pos={pos} seed={seed}");
            }
            draft_outcome
        };

        #[cfg(all(feature = "deltanet", feature = "kvmem"))]
        let mut kvmem_taken = self.kvmem.take();
        #[cfg(all(feature = "deltanet", feature = "kvmem"))]
        let outcome: Result<crate::mtp_spec::MtpDraftOutput, String> = 'c: {
            let ctl = match kvmem_taken.as_mut() {
                Some(c) => c,
                None => break 'c run_draft(&mut gpu, None, None),
            };
            if !ctl.head_plane_enabled() {
                break 'c run_draft(&mut gpu, None, None);
            }
            Self::mtp_draft_head_restore(
                &mut gpu,
                ctl,
                slot.0,
                session_id,
                cap_tokens,
                &self.config,
                run_draft,
            )
        };
        #[cfg(not(all(feature = "deltanet", feature = "kvmem")))]
        let outcome: Result<crate::mtp_spec::MtpDraftOutput, String> =
            run_draft(&mut gpu, None, None);

        #[cfg(all(feature = "deltanet", feature = "kvmem"))]
        {
            self.kvmem = kvmem_taken;
            self.legacy_head_planes[slot.0] = legacy_plane;
        }
        self.mtp_states[slot.0] = Some(state);
        self.spec_req_sampling[slot.0] = req_sampling;
        outcome
    }
}

/// MTP verify/accept phase: after the batched forward, run the trunk lm_head
/// over the slot's verify rows (normed views of `pbs.x_batch`), greedy-accept,
/// and repair the DeltaNet state on partial accepts.
///
/// `hidden_row_offset` is the flat row index where this slot's verify rows
/// begin in the step batch (computed from `batch.m_per_slot` by the caller).
/// `produced`/`max_tokens` are the in-flight request counters, used to clamp
/// the accepted burst to what the commit path can actually keep (see
/// `mtp_batched_verify_accept_from_batch`). Returns committed tokens
/// (excludes seed, includes bonus).
#[allow(clippy::too_many_arguments)]
impl ModelRig {
    /// MTP verify/accept phase: after the batched forward, run the trunk
    /// lm_head over the slot's verify rows (normed views of `pbs.x_batch`),
    /// greedy-accept, and repair the DeltaNet state on partial accepts.
    /// `hidden_row_offset` is the flat row index where this slot's verify
    /// rows begin in the step batch (computed from `batch.m_per_slot` by
    /// the caller). `produced`/`max_tokens` are the in-flight request
    /// counters, used to clamp the accept tail to the request's budget.
    pub(super) fn mtp_verify_accept_step(
        &mut self,
        mut gpu: &mut Gpu,
        pool: &mut SlotPool,
        tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
        cap_tokens: usize,
        slot: SlotId,
        work: &mut PendingWork,
        draft: crate::mtp_spec::MtpDraftOutput,
        hidden_row_offset: usize,
        produced: usize,
        max_tokens: usize,
        sess_len: usize,
    ) -> Result<Vec<u32>, String> {
        let mut state = self.mtp_states[slot.0]
            .take()
            .expect("mtp state must exist from draft phase");

        let pos = work.next_pos;

        // How many tokens of this burst the commit path can keep before a
        // terminator fires: the request's remaining max_tokens budget and its
        // remaining context room (`commit_sampled_token` stops decode when
        // `sess.tokens.len() + 1 >= cap`, so room is `cap - len`).
        let commit_budget = max_tokens
            .saturating_sub(produced)
            .min(cap_tokens.saturating_sub(sess_len))
            .max(1);

        let outcome: Result<Vec<u32>, String> = (|| {
            // P4.1② f2-D: under Layerwise+MTP the consume pass (below) recomputes
            // the accepted head K/V from the trunk's ACTUAL verify hidden (ninfer
            // `final_hidden`, NL:10259-10339). The head pairs token `p` with
            // hidden `p-1`, so fill row 0 must hold the pre-window hidden
            // h_{pos-1}. Stage it NOW: `mtp_batched_verify_accept_from_batch`
            // overwrites `state.prev_hidden` with the last accepted verify row
            // (h_{pos+advance-1}) before returning. Legacy / AR-only skip this
            // (head_plane_enabled() == false ⇒ byte-neutral).
            let head_consume = kvmem_hooks::head_composite_enabled(self);
            if head_consume {
                let dim = self.config.dim;
                gpu.hip
                    .memcpy_dtod_at(
                        &state.takeover_fill_hidden.buf,
                        0,
                        &state.prev_hidden.buf,
                        0,
                        dim * 4,
                    )
                    .map_err(|e| format!("mtp head consume stage: {e:?}"))?;
            }
            let result = crate::mtp_spec::mtp_batched_verify_accept_from_batch(
                &mut gpu,
                &self.weights,
                &self.config,
                &mut self.dn_states[slot.0],
                &self.pbs,
                &mut state,
                &draft,
                hidden_row_offset,
                self.spec_verify_tape
                    .as_ref()
                    .expect("MTP drafts require the verify tape"),
                self.spec_rows,
                slot.0,
                tokenizer.eos_id,
                tokenizer.eot_id,
                commit_budget,
            )
            .map_err(|e| format!("mtp verify: {e}"))?;

            // Update position and seq_len. Stale KV rows past the new frontier
            // (rejected drafts) are overwritten before any later forward reads
            // them: every subsequent write is position-ordered.
            let new_pos = pos + result.advance;
            work.next_pos = new_pos;
            pool.set_seq_len(slot, new_pos)
                .map_err(|e| format!("mtp set_seq_len: {e}"))?;
            // Native recovery identity: after the DN repair above, `prev_hidden`
            // holds the trunk hidden of the last committed verify row, whose
            // absolute position is `new_pos - 1`. A backend recovery image must
            // not infer this from shapes.
            state.prev_hidden_pos = Some(
                new_pos
                    .checked_sub(1)
                    .ok_or_else(|| "mtp verify: empty accepted advance".to_string())?,
            );
            // KVMem statistics, Layerwise output acceptance and optional head
            // aux-plane consume are isolated behind the stable optional seam.
            kvmem_hooks::accept_mtp_verify(
                self,
                gpu,
                pool,
                slot,
                pos,
                result.advance,
                hidden_row_offset,
                &draft,
                &mut state,
                head_consume,
            )?;

            Ok(result.committed)
        })();
        self.mtp_states[slot.0] = Some(state);
        outcome
    }
}

/// DFlash2 draft phase for one slot: run the draft forward over the slot's
/// committed context and save the DN snapshot for the verify rollback.
///
/// Called when `work.decoding` (subsequent spec steps). Takes the seed from
/// `work.remaining_prompt.last()` — same protocol as `mtp_draft_step`.
impl ModelRig {
    pub(super) fn dflash_draft_step(
        &mut self,
        mut gpu: &mut Gpu,
        slot: SlotId,
        work: &mut PendingWork,
        emitted: &[u32],
    ) -> Result<crate::dflash_slot::DflashSlotDraft, String> {
        let shared = self
            .dflash
            .as_ref()
            .ok_or_else(|| "dflash_draft_step: DFlash not loaded".to_string())?;
        let mut st = self.dflash_states[slot.0]
            .take()
            .ok_or_else(|| "dflash_draft_step: slot state missing (admit bug)".to_string())?;

        // Install this request's sampling/penalty config (parked by admit),
        // exactly like the MTP draft step: reseed the accept RNG once per
        // request, then rebuild the penalty history from `emitted` every
        // window. DFlash's draft proposals are deterministic, so only the
        // verify accept consumes the sampling/penalty state.
        let mut req_sampling = self.spec_req_sampling[slot.0].take();
        if let Some(cfg) = req_sampling.as_mut() {
            if !cfg.applied {
                st.sampling = cfg.sampling;
                st.rng = crate::mtp_spec::MtpRng::new(cfg.rng_seed);
                if cfg.sampling.penalized() {
                    st.penalty = hipfire_runtime::spec_sampling::PenaltyHistory::new(
                        cfg.sampling.repeat_window,
                    );
                    st.penalty.set_prompt(&cfg.prompt_tail);
                }
                cfg.applied = true;
            }
            if cfg.sampling.penalized() {
                // Row k of this window scopes `suffix_W(emitted ‖ c[..k])`.
                st.penalty.begin_window(emitted);
            }
        }

        let prompt_tokens = std::mem::take(&mut work.remaining_prompt);
        let pos = work.next_pos;
        let ctx_floor = kvmem_hooks::dflash_context_floor(self, slot, pos);
        let outcome: Result<crate::dflash_slot::DflashSlotDraft, String> = (|| {
            let seed = *prompt_tokens
                .last()
                .ok_or_else(|| "dflash_draft_step: no seed token".to_string())?;
            // Save DN snapshot for rollback during verify phase.
            st.trunk_snap
                .save_from(&mut self.dn_states[slot.0], &mut gpu)
                .map_err(|e| format!("dflash dn snapshot: {e}"))?;
            // 期4 (KVMem/nInfer DFlash2 integration): the draft's local
            // attention lower bound is the retained continuous recent suffix
            // of the target's Layerwise selection (nInfer `dflash_context_floor`
            // = main_frontier − recent_blocks·page_tokens). Logical positions:
            // the draft ring, proposal positions and cyclic slot addressing
            // stay logical; only the target verify uses compact positions.
            // Zero under Legacy / no reselection ⇒ the span is unchanged
            // (byte-neutral).
            crate::dflash_slot::dflash_slot_draft_step(
                &mut gpu,
                shared,
                &self.weights,
                &self.config,
                &mut st,
                seed,
                pos,
                ctx_floor,
            )
        })();

        self.dflash_states[slot.0] = Some(st);
        self.spec_req_sampling[slot.0] = req_sampling;
        outcome
    }
}

/// DFlash2 verify/accept phase: norm the slot's verify rows out of
/// `pbs.x_batch`, run the trunk lm_head, greedy-accept, repair DN state on
/// partial accept, and commit the kept rows' extract-layer hiddens into the
/// slot's `target_hidden` ring. Mirrors `mtp_verify_accept_step`.
#[allow(clippy::too_many_arguments)]
impl ModelRig {
    /// DFlash2 verify/accept phase: norm the slot's verify rows out of
    /// `pbs.x_batch`, run the trunk lm_head, greedy-accept, repair DN state
    /// on partial accept, and commit the kept rows' extract-layer hiddens
    /// into the slot's `target_hidden` ring. Mirrors `mtp_verify_accept_step`.
    pub(super) fn dflash_verify_accept_step(
        &mut self,
        mut gpu: &mut Gpu,
        pool: &mut SlotPool,
        tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
        cap_tokens: usize,
        slot: SlotId,
        work: &mut PendingWork,
        mut draft: crate::dflash_slot::DflashSlotDraft,
        hidden_row_offset: usize,
        produced: usize,
        max_tokens: usize,
        sess_len: usize,
    ) -> Result<Vec<u32>, String> {
        let mut st = self.dflash_states[slot.0]
            .take()
            .expect("dflash state must exist from draft phase");
        let pos = work.next_pos;
        let commit_budget = max_tokens
            .saturating_sub(produced)
            .min(cap_tokens.saturating_sub(sess_len))
            .max(1);

        let outcome: Result<Vec<u32>, String> = (|| {
            let committed = crate::dflash_slot::dflash_slot_verify_accept(
                &mut gpu,
                self.dflash
                    .as_ref()
                    .expect("dflash draft requires shared state"),
                &self.weights,
                &self.config,
                &mut self.dn_states[slot.0],
                &self.pbs,
                &mut st,
                &mut draft,
                pos,
                hidden_row_offset,
                self.spec_verify_tape
                    .as_ref()
                    .expect("DFlash drafts require the verify tape"),
                self.spec_rows,
                slot,
                tokenizer.eos_id,
                tokenizer.eot_id,
                commit_budget,
            )
            .map_err(|e| format!("dflash verify: {e}"))?;

            let new_pos = pos + committed.len();
            work.next_pos = new_pos;
            pool.set_seq_len(slot, new_pos)
                .map_err(|e| format!("dflash set_seq_len: {e}"))?;
            // KVMem statistics and Layerwise output acceptance are isolated
            // behind the stable optional seam.
            kvmem_hooks::accept_dflash_verify(
                self,
                gpu,
                pool,
                slot,
                pos,
                committed.len(),
                hidden_row_offset,
            )?;
            Ok(committed)
        })();

        // Keep the draft's device buffers for the next cycle's reuse.
        st.draft = Some(draft);
        self.dflash_states[slot.0] = Some(st);
        outcome
    }
}

/// DFlash2 prefill chunk: lazily allocate the slot's draft state, scatter
/// this chunk's extract-layer hidden rows (captured into the shared
/// staging during the forward) into the slot's `target_hidden` ring, and
/// advance the thlog watermark. `chunk_pos` is the chunk's first absolute
/// position (`batch.positions[row_off]`).
impl ModelRig {
    pub(super) fn dflash_prefill_chunk(
        &mut self,
        mut gpu: &mut Gpu,
        slot: SlotId,
        row_off: usize,
        m: usize,
        chunk_pos: usize,
    ) -> Result<(), String> {
        if self.dflash_states[slot.0].is_none() {
            let st = crate::dflash_slot::new_dflash_slot_state(
                &mut gpu,
                self.dflash
                    .as_ref()
                    .ok_or_else(|| "dflash prefill: shared state missing".to_string())?,
                &self.config,
                &self.dn_states[slot.0],
            )
            .map_err(|e| format!("dflash state alloc: {e}"))?;
            self.dflash_states[slot.0] = Some(st);
        }
        let st = self.dflash_states[slot.0].as_mut().unwrap();
        let shared = self.dflash.as_ref().unwrap();
        crate::dflash_slot::scatter_staging_rows_to_interleaved(
            &mut gpu, shared, st, row_off, chunk_pos, m,
        )
        .map_err(|e| format!("dflash prefill scatter: {e}"))?;
        st.scratch.thlog.append_committed(chunk_pos, m, 0);
        st.seeded_through = chunk_pos + m;
        Ok(())
    }
}
