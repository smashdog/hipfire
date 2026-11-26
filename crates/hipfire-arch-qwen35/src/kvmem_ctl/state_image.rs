// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! State-image vocabulary shared by the KVMem takeover stages.
//!
//! A [`StateImage`] is an immutable copy of one piece of engine-owned
//! recurrent/draft state (DeltaNet, DFlash, MTP head). The engine keeps the
//! live object while a slot is active; the image is what the host tier may
//! retain after the live object is parked. This module deliberately contains
//! only the vocabulary and accounting identity — S0 adds no lifecycle and no
//! device work.

#![allow(dead_code)]

/// Which engine-owned state an image contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StateImageKind {
    /// DeltaNet recurrent + convolution state (`DeltaNetState`).
    Dn,
    /// DFlash draft slot state (`DflashSlotState`).
    Dflash,
    /// Legacy MTP head KV (`Qwen35MtpHeadKvCache`).
    MtpHead,
}

impl StateImageKind {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Dn => "dn",
            Self::Dflash => "dflash",
            Self::MtpHead => "mtp_head",
        }
    }
}

/// Who may mutate the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateImageRole {
    /// The image backs a live slot and can still advance.
    Active,
    /// Immutable history anchor; reusable after the frontier is validated.
    Checkpoint,
    /// Only a host copy exists; the device replica may be absent.
    HostOnly,
}

/// Where the bytes currently live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateImageResidency {
    Device,
    Host,
    Both,
}

/// Stable image identity inside one engine instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct StateImageId(pub(crate) u64);

impl StateImageId {
    pub(crate) const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

/// Reuse proof bound to an image.
///
/// The fingerprints are deliberately opaque integers: the layouts and token
/// ledgers that produce them stay in their owning modules. An image is only
/// restorable when every field matches the target request exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateImageStamp {
    /// Model identity (weights + quantization).
    pub(crate) model: u64,
    /// Execution/layout identity (KV tier, RoPE, render configuration).
    pub(crate) layout: u64,
    /// Token-ledger identity at the image frontier.
    pub(crate) tokens: u64,
    /// Sticky media digest; `None` for text-only continuations.
    pub(crate) media: Option<u64>,
    /// Token frontier covered by the image.
    pub(crate) frontier: u32,
}

/// Metadata for one retained state image. The payload stays owned by the
/// state store; this record is what the catalog and metrics can inspect
/// without touching device or host bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateImage {
    pub(crate) id: StateImageId,
    pub(crate) kind: StateImageKind,
    pub(crate) role: StateImageRole,
    pub(crate) residency: StateImageResidency,
    pub(crate) stamp: StateImageStamp,
    /// Host bytes charged for this image (0 when device-only).
    pub(crate) host_bytes: u64,
}

impl StateImage {
    pub(crate) const fn new(
        id: StateImageId,
        kind: StateImageKind,
        role: StateImageRole,
        residency: StateImageResidency,
        stamp: StateImageStamp,
        host_bytes: u64,
    ) -> Self {
        Self {
            id,
            kind,
            role,
            residency,
            stamp,
            host_bytes,
        }
    }

    /// True when the image is safe to hold without a live slot: immutable and
    /// backed by a host copy.
    pub(crate) const fn is_host_checkpoint(self) -> bool {
        matches!(
            self.role,
            StateImageRole::Checkpoint | StateImageRole::HostOnly
        ) && matches!(
            self.residency,
            StateImageResidency::Host | StateImageResidency::Both
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp() -> StateImageStamp {
        StateImageStamp {
            model: 1,
            layout: 2,
            tokens: 3,
            media: None,
            frontier: 128,
        }
    }

    #[test]
    fn labels_are_stable() {
        assert_eq!(StateImageKind::Dn.label(), "dn");
        assert_eq!(StateImageKind::Dflash.label(), "dflash");
        assert_eq!(StateImageKind::MtpHead.label(), "mtp_head");
    }

    #[test]
    fn only_immutable_host_backed_images_are_checkpoints() {
        let checkpoint = StateImage::new(
            StateImageId::new(1),
            StateImageKind::Dn,
            StateImageRole::Checkpoint,
            StateImageResidency::Both,
            stamp(),
            4096,
        );
        assert!(checkpoint.is_host_checkpoint());

        let active = StateImage {
            role: StateImageRole::Active,
            ..checkpoint
        };
        assert!(!active.is_host_checkpoint());

        let device_only = StateImage {
            residency: StateImageResidency::Device,
            ..checkpoint
        };
        assert!(!device_only.is_host_checkpoint());

        let host_only = StateImage {
            role: StateImageRole::HostOnly,
            residency: StateImageResidency::Host,
            ..checkpoint
        };
        assert!(host_only.is_host_checkpoint());
    }
}
