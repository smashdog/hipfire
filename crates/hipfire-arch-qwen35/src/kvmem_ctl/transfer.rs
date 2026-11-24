// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use super::*;

impl KvMemCtl {
    /// Grow the reusable pinned slab to at least `need` bytes. The previous
    /// slab (if any) was drained by the prior chunk's stream wait, so dropping
    /// it here frees no in-flight copy. The first allocation takes the whole
    /// chunk cap (`XFER_SLAB_CAP`, or `need` when a single page exceeds it) so a
    /// later, larger chunk never re-page-locks pinned memory mid-prefill —
    /// `hipHostMalloc` is costly and would otherwise land in the TTFT path every
    /// time the batch grows. Every chunk is `<= cap` by construction, so this
    /// allocates at most once per controller.
    fn ensure_xfer_slab(&mut self, gpu: &Gpu, need: usize) -> Result<(), String> {
        if self.xfer_slab.as_ref().is_none_or(|b| b.size() < need) {
            let alloc = need.max(XFER_SLAB_CAP);
            self.xfer_slab = None;
            let buf = gpu
                .hip
                .transfer_buffer(alloc)
                .map_err(|e| format!("kvmem transfer slab alloc ({alloc} B): {e:?}"))?;
            if self.diagnostics.time {
                eprintln!("[kvmem-time] xfer slab alloc {alloc} B (need {need})");
            }
            self.xfer_slab = Some(buf);
        }
        Ok(())
    }

    /// Bytes one full page occupies in the slab across every arena (K+V).
    fn page_xfer_bytes(&self) -> usize {
        self.n_fa_layers * (self.k_page_bytes + self.v_page_bytes)
    }

    /// Spill this slot's selected pages (`(lp, phys)` list) to the host store.
    /// Batched path: gather every page's K/V out of the arenas into the pinned
    /// slab with stream-ordered async D2H, ONE wait per chunk, then write the
    /// host payloads and mark each page spilled. All reads drain before any
    /// `stage_out_page` frees a physical page, and the whole batch completes
    /// inside Pass 1 before Pass 2 can reuse a freed page. The copies ride the
    /// steady-copy stream when one is installed, else the legacy null stream —
    /// the same stream prefill compute uses, so ordering holds either way. Falls
    /// back to the per-page blocking path only when batching is off or a graph
    /// is being captured.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn flush_spill_batch(
        &mut self,
        gpu: &Gpu,
        pool: &mut SlotPool,
        s: usize,
        spills: &[(usize, u32)],
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        timed: bool,
        t_spill: &mut std::time::Duration,
        n_spilled: &mut usize,
    ) -> Result<(), String> {
        if spills.is_empty() {
            return Ok(());
        }
        let stream = gpu.steady_copy_stream();
        let batched = self.batch_xfer && !gpu.graphs.capture_mode;
        if timed {
            eprintln!(
                "[kvmem-time] spill xfer={} pages={} stream={}",
                if batched { "batch" } else { "page" },
                spills.len(),
                stream.is_some()
            );
        }
        if !batched {
            for &(lp, phys) in spills {
                let t = std::time::Instant::now();
                spill_page(
                    gpu,
                    &self.slots[s].raw,
                    &mut self.page_k,
                    &mut self.page_v,
                    k_arenas,
                    v_arenas,
                    self.k_page_bytes,
                    self.v_page_bytes,
                    phys,
                    lp,
                    self.diagnostics.verify,
                )?;
                pool.stage_out_page(SlotId(s), lp)
                    .map_err(|e| format!("kvmem stage-out: {e}"))?;
                if timed {
                    *t_spill += t.elapsed();
                    *n_spilled += 1;
                }
            }
            return Ok(());
        }
        let per_page = self.page_xfer_bytes();
        let chunk_pages = (XFER_SLAB_CAP / per_page).max(1);
        let kpb = self.k_page_bytes;
        let vpb = self.v_page_bytes;
        let n_layers = self.n_fa_layers;
        for chunk in spills.chunks(chunk_pages) {
            let t = std::time::Instant::now();
            self.ensure_xfer_slab(gpu, chunk.len() * per_page)?;
            let slab = self
                .xfer_slab
                .as_mut()
                .expect("transfer slab")
                .as_bytes_mut();
            // Phase A: async D2H gather every page/layer into the slab.
            let mut off = 0usize;
            for &(_lp, phys) in chunk {
                for il in 0..n_layers {
                    gpu.hip
                        .memcpy_dtoh_async_at(
                            &mut slab[off..off + kpb],
                            &k_arenas[il].buf,
                            phys as usize * kpb,
                            stream,
                        )
                        .map_err(|e| format!("kvmem spill K read (layer {il}): {e:?}"))?;
                    off += kpb;
                    gpu.hip
                        .memcpy_dtoh_async_at(
                            &mut slab[off..off + vpb],
                            &v_arenas[il].buf,
                            phys as usize * vpb,
                            stream,
                        )
                        .map_err(|e| format!("kvmem spill V read (layer {il}): {e:?}"))?;
                    off += vpb;
                }
            }
            // ONE wait drains the whole chunk before the host reads the slab.
            match stream {
                Some(s) => gpu.hip.stream_synchronize(s),
                None => gpu.hip.stream_synchronize_default(),
            }
            .map_err(|e| format!("kvmem spill sync: {e:?}"))?;
            // Phase B: write host payloads, then mark each page spilled.
            let mut off = 0usize;
            for &(lp, phys) in chunk {
                let pos0 = (lp * KV_BLOCK_TOKENS) as u32;
                for il in 0..n_layers {
                    let ksrc = &slab[off..off + kpb];
                    self.slots[s]
                        .raw
                        .write_layer_k_gpu(pos0, KV_BLOCK_TOKENS as u32, il as u32, ksrc)
                        .map_err(|e| format!("kvmem spill K store (layer {il}): {e}"))?;
                    off += kpb;
                    let vsrc = &slab[off..off + vpb];
                    self.slots[s]
                        .raw
                        .write_layer_v_gpu(pos0, KV_BLOCK_TOKENS as u32, il as u32, vsrc)
                        .map_err(|e| format!("kvmem spill V store (layer {il}): {e}"))?;
                    off += vpb;
                    if il == 0 && self.diagnostics.verify {
                        eprintln!(
                            "[kvmem-verify] spill lp={lp} phys={phys} \
                             k={:016x} v={:016x}",
                            fnv64(ksrc),
                            fnv64(vsrc)
                        );
                    }
                }
                pool.stage_out_page(SlotId(s), lp)
                    .map_err(|e| format!("kvmem stage-out: {e}"))?;
            }
            if timed {
                *t_spill += t.elapsed();
                *n_spilled += chunk.len();
            }
        }
        Ok(())
    }

    /// Restore the selected spilled pages (`(slot, lp, phys)` list, already
    /// alloc'd and bound by the caller) back into the arenas. Batched path:
    /// gather host payloads into the pinned slab, scatter them with
    /// stream-ordered async H2D, ONE wait per chunk so every page is resident
    /// before `forward_step`. The copies ride the steady-copy stream when one is
    /// installed, else the legacy null stream — the same stream prefill compute
    /// uses, so the restored bytes are visible to the next attention either way.
    /// Falls back to the per-page blocking path only when batching is off or a
    /// graph is captured.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn flush_restore_batch(
        &mut self,
        gpu: &Gpu,
        batch: &[(usize, usize, u32)],
        k_arenas: &[GpuTensor],
        v_arenas: &[GpuTensor],
        timed: bool,
        t_restore: &mut std::time::Duration,
        n_restored: &mut usize,
    ) -> Result<(), String> {
        if batch.is_empty() {
            return Ok(());
        }
        let stream = gpu.steady_copy_stream();
        let batched = self.batch_xfer && !gpu.graphs.capture_mode;
        if timed {
            eprintln!(
                "[kvmem-time] restore xfer={} pages={} stream={}",
                if batched { "batch" } else { "page" },
                batch.len(),
                stream.is_some()
            );
        }
        if !batched {
            for &(s, lp, phys) in batch {
                let t = std::time::Instant::now();
                restore_page(
                    gpu,
                    &self.slots[s].raw,
                    &mut self.page_k,
                    &mut self.page_v,
                    k_arenas,
                    v_arenas,
                    self.k_page_bytes,
                    self.v_page_bytes,
                    phys,
                    lp,
                    self.diagnostics.verify,
                )?;
                if timed {
                    *t_restore += t.elapsed();
                    *n_restored += 1;
                }
            }
            return Ok(());
        }
        let per_page = self.page_xfer_bytes();
        let chunk_pages = (XFER_SLAB_CAP / per_page).max(1);
        let kpb = self.k_page_bytes;
        let vpb = self.v_page_bytes;
        let n_layers = self.n_fa_layers;
        let verify = self.diagnostics.verify;
        for chunk in batch.chunks(chunk_pages) {
            let t = std::time::Instant::now();
            self.ensure_xfer_slab(gpu, chunk.len() * per_page)?;
            let slab = self
                .xfer_slab
                .as_mut()
                .expect("transfer slab")
                .as_bytes_mut();
            // Phase A: gather host payloads into the slab.
            let mut off = 0usize;
            for &(s, lp, _phys) in chunk {
                let block = lp as u32;
                for il in 0..n_layers {
                    if !self.slots[s].raw.copy_k_gpu(
                        block,
                        il as u32,
                        &mut slab[off..off + kpb],
                        KV_BLOCK_TOKENS as u32,
                    ) {
                        return Err(format!(
                            "kvmem restore: block {block} layer {il} has no K payload"
                        ));
                    }
                    off += kpb;
                    if !self.slots[s].raw.copy_v_gpu(
                        block,
                        il as u32,
                        &mut slab[off..off + vpb],
                        KV_BLOCK_TOKENS as u32,
                    ) {
                        return Err(format!(
                            "kvmem restore: block {block} layer {il} has no V payload"
                        ));
                    }
                    off += vpb;
                }
            }
            // Phase B: async H2D scatter the slab into every arena page.
            let mut off = 0usize;
            for &(_s, _lp, phys) in chunk {
                for il in 0..n_layers {
                    gpu.hip
                        .memcpy_htod_async_at_on(
                            &k_arenas[il].buf,
                            phys as usize * kpb,
                            &slab[off..off + kpb],
                            stream,
                        )
                        .map_err(|e| format!("kvmem restore K write (layer {il}): {e:?}"))?;
                    off += kpb;
                    gpu.hip
                        .memcpy_htod_async_at_on(
                            &v_arenas[il].buf,
                            phys as usize * vpb,
                            &slab[off..off + vpb],
                            stream,
                        )
                        .map_err(|e| format!("kvmem restore V write (layer {il}): {e:?}"))?;
                    off += vpb;
                }
            }
            // ONE wait: every restored page is resident before forward_step.
            match stream {
                Some(s) => gpu.hip.stream_synchronize(s),
                None => gpu.hip.stream_synchronize_default(),
            }
            .map_err(|e| format!("kvmem restore sync: {e:?}"))?;
            // VERIFY: read back layer 0 of each page and compare to the slab.
            if verify {
                let mut off = 0usize;
                for &(_s, lp, phys) in chunk {
                    let ksrc = &slab[off..off + kpb];
                    let vsrc = &slab[off + kpb..off + kpb + vpb];
                    let mut rb_k = vec![0u8; kpb];
                    let mut rb_v = vec![0u8; vpb];
                    gpu.hip
                        .memcpy_dtoh_at(&mut rb_k, &k_arenas[0].buf, phys as usize * kpb)
                        .map_err(|e| format!("kvmem verify K readback (layer 0): {e:?}"))?;
                    gpu.hip
                        .memcpy_dtoh_at(&mut rb_v, &v_arenas[0].buf, phys as usize * vpb)
                        .map_err(|e| format!("kvmem verify V readback (layer 0): {e:?}"))?;
                    let mk = ksrc.iter().zip(&rb_k).filter(|(a, b)| a != b).count();
                    let mv = vsrc.iter().zip(&rb_v).filter(|(a, b)| a != b).count();
                    eprintln!(
                        "[kvmem-verify] restore lp={lp} phys={phys} \
                         k={:016x} v={:016x} readback_mismatch k={mk} v={mv}",
                        fnv64(ksrc),
                        fnv64(vsrc)
                    );
                    off += per_page;
                }
            }
            if timed {
                *t_restore += t.elapsed();
                *n_restored += chunk.len();
            }
        }
        Ok(())
    }
}

/// Capture exactly the accepted rows of one Legacy page, including a partial
/// tail. Moving opaque packed bytes preserves K/V scales and format metadata;
/// no K is reconstructed or rotated during a recovery-point transfer.
#[allow(clippy::too_many_arguments)]
pub(super) fn copy_legacy_page_to_host(
    gpu: &Gpu,
    raw: &RawStore,
    page_k: &mut [u8],
    page_v: &mut [u8],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    k_page_bytes: usize,
    v_page_bytes: usize,
    phys: u32,
    lp: usize,
    rows: usize,
    verify: bool,
) -> Result<(), String> {
    if rows == 0
        || rows > KV_BLOCK_TOKENS
        || k_arenas.len() != v_arenas.len()
        || k_page_bytes % KV_BLOCK_TOKENS != 0
        || v_page_bytes % KV_BLOCK_TOKENS != 0
        || page_k.len() < k_page_bytes
        || page_v.len() < v_page_bytes
    {
        return Err("kvmem: invalid Legacy page capture geometry".into());
    }
    let klen = rows * (k_page_bytes / KV_BLOCK_TOKENS);
    let vlen = rows * (v_page_bytes / KV_BLOCK_TOKENS);
    let pos = u32::try_from(
        lp.checked_mul(KV_BLOCK_TOKENS)
            .ok_or("kvmem: Legacy capture position overflow")?,
    )
    .map_err(|_| "kvmem: Legacy capture position exceeds u32")?;
    for layer in 0..k_arenas.len() {
        gpu.hip
            .memcpy_dtoh_at(
                &mut page_k[..klen],
                &k_arenas[layer].buf,
                phys as usize * k_page_bytes,
            )
            .map_err(|e| format!("kvmem capture K layer {layer}: {e:?}"))?;
        gpu.hip
            .memcpy_dtoh_at(
                &mut page_v[..vlen],
                &v_arenas[layer].buf,
                phys as usize * v_page_bytes,
            )
            .map_err(|e| format!("kvmem capture V layer {layer}: {e:?}"))?;
        raw.write_layer_k_gpu(pos, rows as u32, layer as u32, &page_k[..klen])
            .map_err(|e| format!("kvmem capture K archive layer {layer}: {e}"))?;
        raw.write_layer_v_gpu(pos, rows as u32, layer as u32, &page_v[..vlen])
            .map_err(|e| format!("kvmem capture V archive layer {layer}: {e}"))?;
        if verify {
            eprintln!(
                "[kvmem-verify] capture layer={layer} lp={lp} rows={rows} k={:016x} v={:016x}",
                fnv64(&page_k[..klen]),
                fnv64(&page_v[..vlen])
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn copy_legacy_page_to_device(
    gpu: &Gpu,
    raw: &RawStore,
    page_k: &mut [u8],
    page_v: &mut [u8],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    k_page_bytes: usize,
    v_page_bytes: usize,
    phys: u32,
    lp: usize,
    rows: usize,
    verify: bool,
) -> Result<(), String> {
    if rows == 0
        || rows > KV_BLOCK_TOKENS
        || k_arenas.len() != v_arenas.len()
        || k_page_bytes % KV_BLOCK_TOKENS != 0
        || v_page_bytes % KV_BLOCK_TOKENS != 0
        || page_k.len() < k_page_bytes
        || page_v.len() < v_page_bytes
    {
        return Err("kvmem: invalid Legacy page restore geometry".into());
    }
    let klen = rows * (k_page_bytes / KV_BLOCK_TOKENS);
    let vlen = rows * (v_page_bytes / KV_BLOCK_TOKENS);
    for layer in 0..k_arenas.len() {
        if !raw.copy_k_gpu(lp as u32, layer as u32, &mut page_k[..klen], rows as u32)
            || !raw.copy_v_gpu(lp as u32, layer as u32, &mut page_v[..vlen], rows as u32)
        {
            return Err(format!(
                "kvmem: incomplete restore payload layer={layer} lp={lp} rows={rows}"
            ));
        }
        gpu.hip
            .memcpy_htod_offset(
                &k_arenas[layer].buf,
                phys as usize * k_page_bytes,
                &page_k[..klen],
            )
            .map_err(|e| format!("kvmem restore K layer {layer}: {e:?}"))?;
        gpu.hip
            .memcpy_htod_offset(
                &v_arenas[layer].buf,
                phys as usize * v_page_bytes,
                &page_v[..vlen],
            )
            .map_err(|e| format!("kvmem restore V layer {layer}: {e:?}"))?;
        if verify {
            let mut actual_k = vec![0u8; klen];
            let mut actual_v = vec![0u8; vlen];
            gpu.hip
                .memcpy_dtoh_at(
                    &mut actual_k,
                    &k_arenas[layer].buf,
                    phys as usize * k_page_bytes,
                )
                .map_err(|e| format!("kvmem verify K layer {layer}: {e:?}"))?;
            gpu.hip
                .memcpy_dtoh_at(
                    &mut actual_v,
                    &v_arenas[layer].buf,
                    phys as usize * v_page_bytes,
                )
                .map_err(|e| format!("kvmem verify V layer {layer}: {e:?}"))?;
            if actual_k != page_k[..klen] || actual_v != page_v[..vlen] {
                return Err(format!(
                    "kvmem: byte-exact restore failed layer={layer} lp={lp} rows={rows}"
                ));
            }
            eprintln!(
                "[kvmem-verify] restore layer={layer} lp={lp} rows={rows} k={:016x} v={:016x} readback_mismatch k=0 v=0",
                fnv64(&actual_k), fnv64(&actual_v)
            );
        }
    }
    Ok(())
}

/// Copy a full page's K/V bytes out of the arenas into the host store, then
/// (the caller) marks the logical page spilled.
#[allow(clippy::too_many_arguments)]
fn spill_page(
    gpu: &Gpu,
    raw: &RawStore,
    page_k: &mut [u8],
    page_v: &mut [u8],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    k_page_bytes: usize,
    v_page_bytes: usize,
    phys: u32,
    lp: usize,
    verify: bool,
) -> Result<(), String> {
    debug_assert_eq!(page_k.len(), k_page_bytes);
    debug_assert_eq!(page_v.len(), v_page_bytes);
    let pos0 = (lp * KV_BLOCK_TOKENS) as u32;
    for il in 0..k_arenas.len() {
        gpu.hip
            .memcpy_dtoh_at(page_k, &k_arenas[il].buf, phys as usize * k_page_bytes)
            .map_err(|e| format!("kvmem spill K read (layer {il}): {e:?}"))?;
        raw.write_layer_k_gpu(pos0, KV_BLOCK_TOKENS as u32, il as u32, page_k)
            .map_err(|e| format!("kvmem spill K store (layer {il}): {e}"))?;
        gpu.hip
            .memcpy_dtoh_at(page_v, &v_arenas[il].buf, phys as usize * v_page_bytes)
            .map_err(|e| format!("kvmem spill V read (layer {il}): {e:?}"))?;
        raw.write_layer_v_gpu(pos0, KV_BLOCK_TOKENS as u32, il as u32, page_v)
            .map_err(|e| format!("kvmem spill V store (layer {il}): {e}"))?;
        if il == 0 && verify {
            eprintln!(
                "[kvmem-verify] spill lp={lp} phys={phys} \
                 k={:016x} v={:016x}",
                fnv64(page_k),
                fnv64(page_v)
            );
        }
    }
    Ok(())
}

/// Copy a spilled page's bytes back into physical page `phys` of every arena.
/// The caller has already bound `phys` to the logical page, so the next
/// forward sees the restored bytes.
#[allow(clippy::too_many_arguments)]
pub(super) fn restore_page(
    gpu: &Gpu,
    raw: &RawStore,
    page_k: &mut [u8],
    page_v: &mut [u8],
    k_arenas: &[GpuTensor],
    v_arenas: &[GpuTensor],
    k_page_bytes: usize,
    v_page_bytes: usize,
    phys: u32,
    lp: usize,
    verify: bool,
) -> Result<(), String> {
    let block = lp as u32;
    for il in 0..k_arenas.len() {
        if !raw.copy_k_gpu(block, il as u32, page_k, KV_BLOCK_TOKENS as u32) {
            return Err(format!(
                "kvmem restore: block {block} layer {il} has no K payload"
            ));
        }
        gpu.hip
            .memcpy_htod_offset(&k_arenas[il].buf, phys as usize * k_page_bytes, page_k)
            .map_err(|e| format!("kvmem restore K write (layer {il}): {e:?}"))?;
        if !raw.copy_v_gpu(block, il as u32, page_v, KV_BLOCK_TOKENS as u32) {
            return Err(format!(
                "kvmem restore: block {block} layer {il} has no V payload"
            ));
        }
        gpu.hip
            .memcpy_htod_offset(&v_arenas[il].buf, phys as usize * v_page_bytes, page_v)
            .map_err(|e| format!("kvmem restore V write (layer {il}): {e:?}"))?;
        if il == 0 && verify {
            let mut rb_k = vec![0u8; page_k.len()];
            let mut rb_v = vec![0u8; page_v.len()];
            gpu.hip
                .memcpy_dtoh_at(&mut rb_k, &k_arenas[il].buf, phys as usize * k_page_bytes)
                .map_err(|e| format!("kvmem verify K readback (layer {il}): {e:?}"))?;
            gpu.hip
                .memcpy_dtoh_at(&mut rb_v, &v_arenas[il].buf, phys as usize * v_page_bytes)
                .map_err(|e| format!("kvmem verify V readback (layer {il}): {e:?}"))?;
            let mk = page_k.iter().zip(&rb_k).filter(|(a, b)| a != b).count();
            let mv = page_v.iter().zip(&rb_v).filter(|(a, b)| a != b).count();
            eprintln!(
                "[kvmem-verify] restore lp={lp} phys={phys} \
                 k={:016x} v={:016x} readback_mismatch k={mk} v={mv}",
                fnv64(page_k),
                fnv64(page_v)
            );
        }
    }
    Ok(())
}
