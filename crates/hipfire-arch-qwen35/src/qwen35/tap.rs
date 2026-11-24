// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Opaque attention-tap binding for the paged decode paths.
//!
//! `forward.rs` only needs the pair (statistics tap, row-slot map) to build an
//! [`AttentionTap`](hipfire_dispatch::pipeline::hybrid::AttentionTap) closure.
//! Keeping the concrete `KvMemTaps` type behind this small binding lets the
//! upstream forward signatures stay independent of the KVMem module, and the
//! no-`kvmem` build carries a zero-sized placeholder instead of a concrete
//! tap reference.

use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

#[cfg(feature = "kvmem")]
use crate::kvmem_taps::KvMemTaps;

/// The KVMem statistics tap plus the per-row slot map it consumes.
#[cfg(feature = "kvmem")]
#[derive(Clone, Copy)]
pub(crate) struct TapBinding<'a> {
    taps: &'a KvMemTaps,
    row_slot: &'a GpuTensor,
}

#[cfg(feature = "kvmem")]
impl<'a> TapBinding<'a> {
    /// Bind a concrete tap to the row-slot map uploaded for this step.
    pub(crate) fn new(taps: &'a KvMemTaps, row_slot: &'a GpuTensor) -> Self {
        Self { taps, row_slot }
    }

    /// Build an optional binding without exposing `KvMemTaps` to `forward.rs`.
    pub(crate) fn from_optional(
        taps: Option<&'a KvMemTaps>,
        row_slot: &'a GpuTensor,
    ) -> Option<Self> {
        taps.map(|taps| Self::new(taps, row_slot))
    }

    /// Copy this step's Q/K rows for one full-attention layer into the tap
    /// buffers.
    pub(crate) fn tap_layer(
        &self,
        gpu: &mut Gpu,
        layer: usize,
        q: &GpuTensor,
        k: &GpuTensor,
        n_rows: usize,
    ) -> HipResult<()> {
        self.taps.tap_layer(gpu, layer, q, k, self.row_slot, n_rows)
    }
}

/// Zero-sized placeholder used by the default no-`kvmem` build.
#[cfg(not(feature = "kvmem"))]
#[derive(Clone, Copy)]
pub(crate) struct TapBinding<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

#[cfg(not(feature = "kvmem"))]
impl<'a> TapBinding<'a> {
    /// Copy this step's Q/K rows for one full-attention layer into the tap
    /// buffers. No-op when the crate was built without `kvmem`.
    pub(crate) fn tap_layer(
        &self,
        gpu: &mut Gpu,
        layer: usize,
        q: &GpuTensor,
        k: &GpuTensor,
        n_rows: usize,
    ) -> HipResult<()> {
        let _ = (gpu, layer, q, k, n_rows);
        Ok(())
    }
}
