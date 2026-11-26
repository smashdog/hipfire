// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only directory of immutable, complete legacy restore images.
//!
//! `T` is assembled by the caller: sealed host KV references, policy baseline,
//! target state and any required draft/media state must describe the same
//! committed `frontier`. This module neither constructs nor restores native
//! state, owns no device bindings, and never closes a native session.
//!
//! Each catalog belongs to one compatible loaded-model/backend domain. Group
//! and source-session IDs are capacity/diagnostic hints, NEVER reuse proofs.
//! Even identical tokens do not prove identical native state, so publications
//! are independent versions. The only replacement/eviction policy is KVMem's.
//!
//! `byte_limit` and `charge_bytes` account for non-KV snapshots and metadata,
//! not the native KV payload budget H. Shared physical KV payload accounting
//! remains with HostKvStorage. Call `retire_one` at a safe engine boundary when
//! actual H/D pressure requires releasing a history source.

use std::fmt;

use hipfire_kvmem::{
    plan_content_group_publication, rank_content_candidates, select_content_eviction,
    ContentCandidate, ContentSlot, KvMemError, PublicationReason,
};

pub type CatalogResult<T> = Result<T, LegacyCatalogError>;

#[derive(Debug)]
pub enum LegacyCatalogError {
    InvalidConfig(&'static str),
    UnknownGroup,
    IncompleteHistory {
        frontier: usize,
        token_count: usize,
    },
    EntryTooLarge {
        charge_bytes: usize,
        byte_limit: usize,
    },
    PublicationRejected(PublicationReason),
    ByteLimitPinned,
    NotFound(u64),
    Pinned(u64),
    NotPinned(u64),
    Overflow(&'static str),
    Allocation(&'static str),
    InvalidCorePlan(&'static str),
    Core(KvMemError),
}

impl fmt::Display for LegacyCatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "legacy catalog: {message}"),
            Self::UnknownGroup => write!(
                f,
                "legacy catalog: group 0 is unknown; supply a nonzero group"
            ),
            Self::IncompleteHistory {
                frontier,
                token_count,
            } => write!(
                f,
                "legacy catalog: incomplete history (frontier {frontier}, tokens {token_count})"
            ),
            Self::EntryTooLarge {
                charge_bytes,
                byte_limit,
            } => write!(
                f,
                "legacy catalog: non-KV charge {charge_bytes} exceeds byte limit {byte_limit}"
            ),
            Self::PublicationRejected(reason) => {
                write!(f, "legacy catalog: publication rejected ({reason:?})")
            }
            Self::ByteLimitPinned => write!(
                f,
                "legacy catalog: non-KV byte quota cannot be met without retiring pinned entries"
            ),
            Self::NotFound(id) => write!(f, "legacy catalog: entry {id} not found"),
            Self::Pinned(id) => write!(f, "legacy catalog: entry {id} is pinned"),
            Self::NotPinned(id) => write!(f, "legacy catalog: entry {id} is not pinned"),
            Self::Overflow(field) => write!(f, "legacy catalog: {field} overflow"),
            Self::Allocation(field) => write!(f, "legacy catalog: cannot allocate {field}"),
            Self::InvalidCorePlan(message) => {
                write!(f, "legacy catalog: invalid core plan ({message})")
            }
            Self::Core(error) => write!(f, "legacy catalog: {error}"),
        }
    }
}

impl std::error::Error for LegacyCatalogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            _ => None,
        }
    }
}

impl From<KvMemError> for LegacyCatalogError {
    fn from(error: KvMemError) -> Self {
        Self::Core(error)
    }
}

/// A sealed restore source. Only immutable borrows are exposed while retained.
#[derive(Debug)]
pub struct HistoryEntry<T> {
    pub id: u64,
    pub group: u64,
    pub source_session: u64,
    /// Exact token history, including assistant/tool/template tokens. Only the
    /// prefix ending at `frontier` is covered by the native image.
    pub tokens: Vec<u32>,
    pub frontier: usize,
    pub media_digest: Option<u64>,
    /// Caller-calculated non-KV snapshot/metadata charge; never physical KV H.
    pub charge_bytes: usize,
    pub value: T,
    pub last_used: u64,
    pin_count: usize,
}

impl<T> HistoryEntry<T> {
    pub fn is_pinned(&self) -> bool {
        self.pin_count != 0
    }
}

/// Diagnostics only: retirement does not imply closing `source_session`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetiredHistory {
    pub id: u64,
    pub source_session: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct HistoryPublication {
    pub id: u64,
    pub reason: PublicationReason,
    pub retired: Vec<RetiredHistory>,
}

#[derive(Debug)]
pub struct LegacyHistoryCatalog<T> {
    slots: Vec<Option<HistoryEntry<T>>>,
    max_groups: usize,
    max_entries: usize,
    byte_limit: usize,
    charged_bytes: usize,
    last_id: u64,
    clock: u64,
}

fn reserved_vec<T>(capacity: usize, what: &'static str) -> CatalogResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| LegacyCatalogError::Allocation(what))?;
    Ok(values)
}

impl<T> LegacyHistoryCatalog<T> {
    /// Retain at most N capacity groups and 2*N independent versions, following
    /// the core group-publication policy. No capacity-sized allocation is made.
    pub fn new(sessions: usize, byte_limit: usize) -> CatalogResult<Self> {
        if sessions == 0 {
            return Err(LegacyCatalogError::InvalidConfig(
                "sessions must be positive",
            ));
        }
        if byte_limit == 0 {
            return Err(LegacyCatalogError::InvalidConfig(
                "non-KV byte limit must be positive",
            ));
        }
        let max_entries = sessions
            .checked_mul(2)
            .ok_or(LegacyCatalogError::Overflow("2*sessions"))?;
        // The core eviction C ABI returns a signed int slot index.
        if i32::try_from(max_entries).is_err() {
            return Err(LegacyCatalogError::Overflow("core slot index capacity"));
        }
        Ok(Self {
            slots: Vec::new(),
            max_groups: sessions,
            max_entries,
            byte_limit,
            charged_bytes: 0,
            last_id: 0,
            clock: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    pub fn max_groups(&self) -> usize {
        self.max_groups
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn byte_limit(&self) -> usize {
        self.byte_limit
    }

    /// Non-KV charge only; not total host RAM usage or physical KV payload H.
    pub fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }

    pub fn get(&self, id: u64) -> Option<&HistoryEntry<T>> {
        self.slots.iter().flatten().find(|entry| entry.id == id)
    }

    /// Every live entry, for accounting that the catalog itself does not own.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().flatten().map(|entry| &entry.value)
    }

    /// Publish a complete immutable image supplied by the caller. This call
    /// attests that all native components in `value` cover `frontier`; metadata
    /// alone cannot validate native completeness. Zero/out-of-range frontiers
    /// and unknown (zero) groups are rejected, never normalized or merged.
    ///
    /// All quota planning, arithmetic and fallible allocation precede mutation.
    /// On Err the old directory, stamps and IDs are unchanged; the incoming
    /// `tokens`/`value` are consumed even on Err. Old values are dropped only
    /// after the full publication has committed. No native session is closed.
    #[allow(clippy::too_many_arguments)]
    pub fn publish(
        &mut self,
        source_session: u64,
        group: u64,
        tokens: Vec<u32>,
        frontier: usize,
        media_digest: Option<u64>,
        charge_bytes: usize,
        value: T,
    ) -> CatalogResult<HistoryPublication> {
        if group == 0 {
            return Err(LegacyCatalogError::UnknownGroup);
        }
        if frontier == 0 || frontier > tokens.len() {
            return Err(LegacyCatalogError::IncompleteHistory {
                frontier,
                token_count: tokens.len(),
            });
        }
        u64::try_from(frontier).map_err(|_| LegacyCatalogError::Overflow("frontier"))?;
        if charge_bytes > self.byte_limit {
            return Err(LegacyCatalogError::EntryTooLarge {
                charge_bytes,
                byte_limit: self.byte_limit,
            });
        }
        let id = self
            .last_id
            .checked_add(1)
            .ok_or(LegacyCatalogError::Overflow("entry ID"))?;
        let stamp = self
            .clock
            .checked_add(1)
            .ok_or(LegacyCatalogError::Overflow("clock"))?;
        let mut remaining = self.content_slots()?;
        let (slot, reason, core_retired) =
            plan_content_group_publication(&remaining, self.max_entries, self.max_groups, group)?;
        let mut slot = slot.ok_or(LegacyCatalogError::PublicationRejected(reason))?;
        if slot > remaining.len() || slot >= self.max_entries {
            return Err(LegacyCatalogError::InvalidCorePlan(
                "publication slot out of range",
            ));
        }
        let mut retired_slots = reserved_vec(remaining.len(), "retirement plan")?;
        let mut bytes = self.charged_bytes;
        for index in core_retired {
            self.simulate_retirement(index, &mut remaining, &mut bytes, &mut retired_slots)?;
        }
        // Compare by subtraction: even a limit of usize::MAX cannot overflow
        // while deciding which sources to retire. Incoming is not a victim.
        let available = self
            .byte_limit
            .checked_sub(charge_bytes)
            .ok_or(LegacyCatalogError::Overflow("available non-KV bytes"))?;
        while bytes > available {
            let victim =
                select_content_eviction(&remaining)?.ok_or(LegacyCatalogError::ByteLimitPinned)?;
            self.simulate_retirement(victim, &mut remaining, &mut bytes, &mut retired_slots)?;
        }
        // Byte pressure may have made a vacancy after core originally chose
        // append. Reuse that vacancy rather than growing the physical slots.
        if slot == remaining.len() {
            slot = remaining
                .iter()
                .position(|value| value.occupied == 0)
                .unwrap_or(slot);
        }
        if slot < remaining.len() && remaining[slot].occupied != 0 {
            return Err(LegacyCatalogError::InvalidCorePlan(
                "publication would overwrite a live source",
            ));
        }
        let charged_bytes = bytes
            .checked_add(charge_bytes)
            .ok_or(LegacyCatalogError::Overflow("non-KV byte charge"))?;
        let mut retired = reserved_vec(retired_slots.len(), "retired diagnostics")?;
        let mut retired_values = reserved_vec(retired_slots.len(), "retired values")?;
        for &index in &retired_slots {
            let entry = self.slots[index]
                .as_ref()
                .ok_or(LegacyCatalogError::InvalidCorePlan(
                    "retired source missing",
                ))?;
            retired.push(RetiredHistory {
                id: entry.id,
                source_session: entry.source_session,
            });
        }
        if slot == self.slots.len() {
            self.slots
                .len()
                .checked_add(1)
                .ok_or(LegacyCatalogError::Overflow("slot count"))?;
            self.slots
                .try_reserve(1)
                .map_err(|_| LegacyCatalogError::Allocation("directory slot"))?;
        }
        let entry = HistoryEntry {
            id,
            group,
            source_session,
            tokens,
            frontier,
            media_digest,
            charge_bytes,
            value,
            last_used: stamp,
            pin_count: 0,
        };
        let reason = if retired.is_empty() {
            reason
        } else {
            PublicationReason::Replaced
        };

        // Commit: no fallible operations, allocations or native destruction.
        for index in retired_slots {
            retired_values.push(self.slots[index].take());
        }
        if slot == self.slots.len() {
            self.slots.push(Some(entry));
        } else {
            self.slots[slot] = Some(entry);
        }
        self.charged_bytes = charged_bytes;
        self.last_id = id;
        self.clock = stamp;
        drop(retired_values);
        Ok(HistoryPublication {
            id,
            reason,
            retired,
        })
    }

    /// Exact token/media eligibility, then the core ranking, with no touch and
    /// no consumption. `max_frontier` is the caller's legal restore boundary.
    /// Hashes (including Q fingerprints) are not accepted as token proof.
    ///
    /// For a no-new-user-query continuation, require the image to cover the
    /// entire retained token history, not merely an earlier prefix of it. The
    /// caller still executes only append/resident-only pressure for that case.
    /// A new-query lookup may use any proved prefix up to `max_frontier`.
    pub fn lookup(
        &self,
        prompt: &[u32],
        media: Option<u64>,
        max_frontier: usize,
        new_user_query: bool,
    ) -> CatalogResult<Vec<u64>> {
        let mut candidates = reserved_vec(self.len(), "lookup candidates")?;
        for (slot, entry) in self.slots.iter().enumerate() {
            let Some(entry) = entry else { continue };
            if entry.frontier > max_frontier
                || entry.media_digest != media
                || !prompt.starts_with(&entry.tokens[..entry.frontier])
            {
                continue;
            }
            candidates.push(ContentCandidate {
                slot: u64::try_from(slot)
                    .map_err(|_| LegacyCatalogError::Overflow("candidate slot"))?,
                frontier: u64::try_from(entry.frontier)
                    .map_err(|_| LegacyCatalogError::Overflow("candidate frontier"))?,
                // No physical KV/disk-cost estimate is supplied by this API.
                // Equal neutral costs avoid misrepresenting charge_bytes as H.
                cold_bytes: 0,
                last_used: entry.last_used,
                continuation: i32::from(entry.frontier == entry.tokens.len()),
            });
        }
        let ranked = rank_content_candidates(&candidates, new_user_query, candidates.len())?;
        let mut ids = reserved_vec(ranked.len(), "ranked IDs")?;
        for candidate in ranked {
            let slot = usize::try_from(candidate.slot)
                .map_err(|_| LegacyCatalogError::Overflow("ranked slot"))?;
            let entry = self
                .slots
                .get(slot)
                .and_then(Option::as_ref)
                .ok_or(LegacyCatalogError::InvalidCorePlan("ranked source missing"))?;
            ids.push(entry.id);
        }
        Ok(ids)
    }

    /// Call only AFTER a native restore succeeds. A failed restore needs no
    /// catalog operation and leaves the source available for another attempt.
    pub fn touch(&mut self, id: u64) -> CatalogResult<()> {
        let index = self.index(id).ok_or(LegacyCatalogError::NotFound(id))?;
        let stamp = self
            .clock
            .checked_add(1)
            .ok_or(LegacyCatalogError::Overflow("clock"))?;
        if let Some(entry) = &mut self.slots[index] {
            entry.last_used = stamp;
        }
        self.clock = stamp;
        Ok(())
    }

    /// Counted protection across engine operations; every successful pin needs
    /// an unpin on both restore success and failure. Borrowing via `get` is
    /// already protected against concurrent mutable catalog access by Rust.
    pub fn pin(&mut self, id: u64) -> CatalogResult<()> {
        let index = self.index(id).ok_or(LegacyCatalogError::NotFound(id))?;
        if let Some(entry) = &mut self.slots[index] {
            entry.pin_count = entry
                .pin_count
                .checked_add(1)
                .ok_or(LegacyCatalogError::Overflow("pin count"))?;
        }
        Ok(())
    }

    pub fn unpin(&mut self, id: u64) -> CatalogResult<()> {
        let index = self.index(id).ok_or(LegacyCatalogError::NotFound(id))?;
        if let Some(entry) = &mut self.slots[index] {
            entry.pin_count = entry
                .pin_count
                .checked_sub(1)
                .ok_or(LegacyCatalogError::NotPinned(id))?;
        }
        Ok(())
    }

    /// Release one source selected by the same core policy used for metadata
    /// pressure. None means empty or all pinned. Ownership is returned so the
    /// engine can release the host image at its safe boundary and log its IDs.
    pub fn retire_one(&mut self) -> CatalogResult<Option<HistoryEntry<T>>> {
        let slots = self.content_slots()?;
        let Some(index) = select_content_eviction(&slots)? else {
            return Ok(None);
        };
        let id = self
            .slots
            .get(index)
            .and_then(Option::as_ref)
            .ok_or(LegacyCatalogError::InvalidCorePlan(
                "eviction source missing",
            ))?
            .id;
        self.remove(id)
    }

    /// Explicit removal respects pins. Missing IDs are a non-mutating None.
    pub fn remove(&mut self, id: u64) -> CatalogResult<Option<HistoryEntry<T>>> {
        let Some(index) = self.index(id) else {
            return Ok(None);
        };
        let entry = self.slots[index]
            .as_ref()
            .ok_or(LegacyCatalogError::InvalidCorePlan(
                "removal source missing",
            ))?;
        if entry.is_pinned() {
            return Err(LegacyCatalogError::Pinned(id));
        }
        let bytes = self
            .charged_bytes
            .checked_sub(entry.charge_bytes)
            .ok_or(LegacyCatalogError::Overflow("non-KV byte accounting"))?;
        let entry = self.slots[index].take();
        self.charged_bytes = bytes;
        Ok(entry)
    }

    /// All-or-nothing clear, refusing pinned sources. IDs/clock never reset.
    /// Returned images, not native sessions, are released by the caller.
    pub fn clear(&mut self) -> CatalogResult<Vec<HistoryEntry<T>>> {
        if let Some(entry) = self.slots.iter().flatten().find(|entry| entry.is_pinned()) {
            return Err(LegacyCatalogError::Pinned(entry.id));
        }
        let mut entries = reserved_vec(self.len(), "cleared sources")?;
        entries.extend(self.slots.drain(..).flatten());
        self.charged_bytes = 0;
        Ok(entries)
    }

    fn index(&self, id: u64) -> Option<usize> {
        self.slots
            .iter()
            .position(|slot| slot.as_ref().is_some_and(|entry| entry.id == id))
    }

    fn content_slots(&self) -> CatalogResult<Vec<ContentSlot>> {
        let mut slots = reserved_vec(self.slots.len(), "core slot metadata")?;
        for entry in &self.slots {
            slots.push(match entry {
                None => ContentSlot::default(),
                Some(entry) => ContentSlot {
                    occupied: 1,
                    pinned: i32::from(entry.is_pinned()),
                    last_used: entry.last_used,
                    // A unique nonzero catalog label, NOT a native origin proof.
                    // Group publication does not deduplicate by this field.
                    identity: entry.id,
                    group: entry.group,
                },
            });
        }
        Ok(slots)
    }

    fn simulate_retirement(
        &self,
        index: usize,
        remaining: &mut [ContentSlot],
        bytes: &mut usize,
        retired: &mut Vec<usize>,
    ) -> CatalogResult<()> {
        let slot = remaining
            .get_mut(index)
            .ok_or(LegacyCatalogError::InvalidCorePlan(
                "retired slot out of range",
            ))?;
        if slot.occupied == 0 || slot.pinned != 0 {
            return Err(LegacyCatalogError::InvalidCorePlan(
                "retiring an empty or pinned slot",
            ));
        }
        let entry = self.slots[index]
            .as_ref()
            .ok_or(LegacyCatalogError::InvalidCorePlan(
                "retired source missing",
            ))?;
        *bytes = bytes
            .checked_sub(entry.charge_bytes)
            .ok_or(LegacyCatalogError::Overflow("planned non-KV bytes"))?;
        *slot = ContentSlot::default();
        retired.push(index);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn publish(
        catalog: &mut LegacyHistoryCatalog<u8>,
        session: u64,
        group: u64,
        tokens: &[u32],
        charge: usize,
        value: u8,
    ) -> HistoryPublication {
        catalog
            .publish(
                session,
                group,
                tokens.to_vec(),
                tokens.len(),
                None,
                charge,
                value,
            )
            .unwrap()
    }

    fn state(catalog: &LegacyHistoryCatalog<u8>) -> Vec<(u64, u64, u64, usize, u8, usize)> {
        catalog
            .slots
            .iter()
            .flatten()
            .map(|entry| {
                (
                    entry.id,
                    entry.group,
                    entry.last_used,
                    entry.charge_bytes,
                    entry.value,
                    entry.pin_count,
                )
            })
            .collect()
    }

    #[test]
    fn n_plus_one_distinct_groups_retires_the_oldest_for_n_1_2_4() {
        for n in [1, 2, 4] {
            let mut catalog = LegacyHistoryCatalog::new(n, 128).unwrap();
            let mut ids = Vec::new();
            for group in 1..=n {
                let result = publish(
                    &mut catalog,
                    group as u64,
                    group as u64,
                    &[group as u32],
                    1,
                    0,
                );
                assert!(result.retired.is_empty());
                ids.push(result.id);
            }
            let result = publish(&mut catalog, 99, 99, &[99], 1, 9);
            assert_eq!(result.reason, PublicationReason::Replaced);
            assert_eq!(
                result.retired,
                vec![RetiredHistory {
                    id: ids[0],
                    source_session: 1
                }]
            );
            assert_eq!(catalog.len(), n);
            assert_eq!(catalog.max_entries(), 2 * n);
            assert_eq!(catalog.max_groups(), n);
            assert_eq!(catalog.charged_bytes(), n);
            assert!(catalog.get(ids[0]).is_none());
            for &id in &ids[1..] {
                assert!(catalog.get(id).is_some());
            }
        }
    }

    #[test]
    fn same_group_prefix_versions_and_identical_tokens_are_not_deduplicated() {
        let mut catalog = LegacyHistoryCatalog::new(2, 100).unwrap();
        let first = publish(&mut catalog, 7, 10, &[1, 2], 1, 11).id;
        let second = publish(&mut catalog, 7, 10, &[1, 3], 1, 12).id;
        let third = publish(&mut catalog, 7, 10, &[1, 2], 1, 13).id;
        assert_eq!(catalog.len(), 3);
        assert_ne!(first, third);
        assert_eq!(
            catalog.lookup(&[1, 2, 4], None, 3, true).unwrap(),
            vec![third, first]
        );
        assert_eq!(
            catalog.lookup(&[1, 3, 4], None, 3, true).unwrap(),
            vec![second]
        );
        assert_eq!(catalog.get(first).unwrap().value, 11);
        assert_eq!(catalog.get(third).unwrap().value, 13);
    }

    #[test]
    fn slot_and_host_pressure_use_core_redundant_version_priority() {
        let mut catalog = LegacyHistoryCatalog::new(2, 100).unwrap();
        let unique = publish(&mut catalog, 1, 1, &[1], 1, 0).id;
        let redundant = publish(&mut catalog, 2, 2, &[2], 1, 1).id;
        let sibling = publish(&mut catalog, 2, 2, &[2, 3], 1, 2).id;
        publish(&mut catalog, 2, 2, &[2, 3, 4], 1, 3);
        let result = publish(&mut catalog, 2, 2, &[2, 3, 4, 5], 1, 4);
        assert_eq!(
            result.retired,
            vec![RetiredHistory {
                id: redundant,
                source_session: 2
            }]
        );
        assert!(catalog.get(unique).is_some());
        let retired = catalog.retire_one().unwrap().unwrap();
        assert_eq!(retired.id, sibling);
        assert!(catalog.get(unique).is_some());
    }

    #[test]
    fn full_assistant_and_tool_tokens_are_verified_for_every_version() {
        // Identical user/Q tokens do not authorize changed assistant/tool text.
        let mut catalog = LegacyHistoryCatalog::new(2, 100).unwrap();
        let a = publish(&mut catalog, 1, 1, &[10, 11, 20, 21, 30, 31], 1, 1).id;
        let b = publish(&mut catalog, 1, 1, &[10, 11, 20, 22, 30, 32], 1, 2).id;
        assert!(catalog
            .lookup(&[10, 11, 20, 99, 30, 31, 40], None, 7, true)
            .unwrap()
            .is_empty());
        assert!(catalog
            .lookup(&[10, 11, 20, 21, 30, 99, 40], None, 7, false)
            .unwrap()
            .is_empty());
        assert!(catalog.lookup(&[10, 11], None, 7, true).unwrap().is_empty());
        assert_eq!(
            catalog
                .lookup(&[10, 11, 20, 21, 30, 31, 40], None, 7, false)
                .unwrap(),
            vec![a]
        );
        assert_eq!(
            catalog
                .lookup(&[10, 11, 20, 22, 30, 32, 40], None, 7, true)
                .unwrap(),
            vec![b]
        );
    }

    #[test]
    fn media_and_max_frontier_are_hard_eligibility_gates() {
        let mut catalog = LegacyHistoryCatalog::new(2, 100).unwrap();
        let text = publish(&mut catalog, 1, 1, &[1, 2], 1, 0).id;
        let image = catalog
            .publish(2, 2, vec![1, 2], 2, Some(77), 1, 1)
            .unwrap()
            .id;
        assert_eq!(
            catalog.lookup(&[1, 2, 3], None, 2, true).unwrap(),
            vec![text]
        );
        assert_eq!(
            catalog.lookup(&[1, 2, 3], Some(77), 2, true).unwrap(),
            vec![image]
        );
        assert!(catalog
            .lookup(&[1, 2, 3], Some(78), 3, true)
            .unwrap()
            .is_empty());
        assert!(catalog
            .lookup(&[1, 2, 3], Some(77), 1, true)
            .unwrap()
            .is_empty());
        assert!(catalog
            .lookup(&[1, 2, 3], None, 0, true)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn native_frontier_can_precede_transcript_but_is_not_full_continuation() {
        let mut catalog = LegacyHistoryCatalog::new(1, 100).unwrap();
        let id = catalog
            .publish(1, 1, vec![1, 2, 99], 2, None, 1, 0)
            .unwrap()
            .id;
        assert_eq!(catalog.lookup(&[1, 2, 3], None, 2, true).unwrap(), vec![id]);
        assert!(catalog
            .lookup(&[1, 2, 99], None, 3, false)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn lookup_and_get_are_non_consuming_and_only_successful_restore_touches() {
        let mut catalog = LegacyHistoryCatalog::new(2, 100).unwrap();
        let a = publish(&mut catalog, 999, 1, &[1, 2], 1, 1).id;
        let b = publish(&mut catalog, 0, 2, &[1, 2], 1, 2).id;
        let before = state(&catalog);
        for _ in 0..3 {
            assert_eq!(
                catalog.lookup(&[1, 2, 3], None, 2, true).unwrap(),
                vec![b, a]
            );
            // An unsuccessful native restore simply releases its immutable borrow.
            assert_eq!(catalog.get(a).unwrap().value, 1);
        }
        assert_eq!(state(&catalog), before);
        catalog.touch(a).unwrap();
        assert_eq!(
            catalog.lookup(&[1, 2, 3], None, 2, true).unwrap(),
            vec![a, b]
        );
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog.get(a).unwrap().value, 1);
    }

    #[test]
    fn ranking_prefers_longest_frontier_and_does_not_limit_to_sixteen() {
        let mut catalog = LegacyHistoryCatalog::new(10, 100).unwrap();
        let longest = publish(&mut catalog, 1, 1, &[1, 2, 3], 1, 0).id;
        let mut shorter = Vec::new();
        for _ in 0..18 {
            shorter.push(publish(&mut catalog, 1, 1, &[1, 2], 1, 0).id);
        }
        shorter.reverse();
        let ranked = catalog.lookup(&[1, 2, 3, 4], None, 4, true).unwrap();
        assert_eq!(ranked[0], longest);
        assert_eq!(&ranked[1..], shorter.as_slice());
    }

    #[test]
    fn pins_are_counted_and_protect_publication_pressure_remove_and_clear() {
        let mut catalog = LegacyHistoryCatalog::new(1, 100).unwrap();
        let a = publish(&mut catalog, 1, 1, &[1], 1, 0).id;
        let b = publish(&mut catalog, 1, 1, &[1, 2], 1, 1).id;
        catalog.pin(a).unwrap();
        catalog.pin(a).unwrap();
        catalog.pin(b).unwrap();
        let before = state(&catalog);
        assert!(matches!(
            catalog.publish(2, 2, vec![2], 1, None, 1, 2),
            Err(LegacyCatalogError::PublicationRejected(
                PublicationReason::AllPinned
            ))
        ));
        assert!(matches!(
            catalog.publish(1, 1, vec![1, 2, 3], 3, None, 1, 2),
            Err(LegacyCatalogError::PublicationRejected(
                PublicationReason::AllPinned
            ))
        ));
        assert!(catalog.retire_one().unwrap().is_none());
        assert!(matches!(catalog.remove(a), Err(LegacyCatalogError::Pinned(id)) if id == a));
        assert!(matches!(
            catalog.clear(),
            Err(LegacyCatalogError::Pinned(_))
        ));
        assert_eq!(state(&catalog), before);
        catalog.unpin(a).unwrap();
        assert!(catalog.get(a).unwrap().is_pinned());
        catalog.unpin(a).unwrap();
        assert!(!catalog.get(a).unwrap().is_pinned());
        assert!(matches!(catalog.unpin(a), Err(LegacyCatalogError::NotPinned(id)) if id == a));
        assert_eq!(catalog.retire_one().unwrap().unwrap().id, a);
        assert!(catalog.get(b).unwrap().is_pinned());
    }

    #[test]
    fn failed_byte_plan_preserves_even_already_planned_group_retirements() {
        let mut catalog = LegacyHistoryCatalog::new(2, 10).unwrap();
        let a = publish(&mut catalog, 1, 1, &[1], 3, 1).id;
        let b = publish(&mut catalog, 2, 2, &[2], 7, 2).id;
        catalog.pin(b).unwrap();
        let before = state(&catalog);
        let counters = (catalog.clock, catalog.last_id, catalog.charged_bytes());
        // Group quota plans retiring A first. The byte quota then still cannot
        // fit the incoming 4 bytes alongside pinned B: no partial retirement.
        assert!(matches!(
            catalog.publish(3, 3, vec![3], 1, None, 4, 3),
            Err(LegacyCatalogError::ByteLimitPinned)
        ));
        assert_eq!(state(&catalog), before);
        assert_eq!(
            (catalog.clock, catalog.last_id, catalog.charged_bytes()),
            counters
        );
        assert_eq!(catalog.lookup(&[1, 9], None, 2, true).unwrap(), vec![a]);
        assert!(matches!(
            catalog.publish(3, 3, vec![3], 1, None, 11, 3),
            Err(LegacyCatalogError::EntryTooLarge { .. })
        ));
        assert_eq!(state(&catalog), before);
    }

    #[test]
    fn byte_only_simulation_can_retire_multiple_sources_or_fail_atomically() {
        let mut catalog = LegacyHistoryCatalog::new(3, 10).unwrap();
        let pinned = publish(&mut catalog, 1, 1, &[1], 6, 0).id;
        let a = publish(&mut catalog, 2, 2, &[2], 2, 1).id;
        let b = publish(&mut catalog, 2, 2, &[2, 3], 2, 2).id;
        catalog.pin(pinned).unwrap();
        let before = state(&catalog);
        // Both redundant versions would retire, but the final pinned 6 bytes
        // still exceed the remaining 5. Both source versions must survive.
        assert!(matches!(
            catalog.publish(3, 3, vec![3], 1, None, 5, 3),
            Err(LegacyCatalogError::ByteLimitPinned)
        ));
        assert_eq!(state(&catalog), before);
        let result = publish(&mut catalog, 3, 3, &[3], 4, 3);
        assert_eq!(
            result.retired,
            vec![
                RetiredHistory {
                    id: a,
                    source_session: 2
                },
                RetiredHistory {
                    id: b,
                    source_session: 2
                },
            ]
        );
        assert_eq!(catalog.charged_bytes(), 10);
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog.slots.len(), 3);
        assert!(catalog.get(pinned).is_some());
    }

    #[test]
    fn unknown_groups_are_rejected_without_merging_or_spending_ids() {
        let mut catalog = LegacyHistoryCatalog::new(2, 100).unwrap();
        let known = publish(&mut catalog, 1, 1, &[1], 1, 0).id;
        let before = state(&catalog);
        for session in [1, 2, 0] {
            assert!(matches!(
                catalog.publish(session, 0, vec![1], 1, None, 1, 1),
                Err(LegacyCatalogError::UnknownGroup)
            ));
        }
        assert_eq!(state(&catalog), before);
        assert_eq!(catalog.last_id, known);
        let distinct = publish(&mut catalog, 1, 2, &[1], 1, 1).id;
        assert_eq!(
            catalog.lookup(&[1, 2], None, 1, true).unwrap(),
            vec![distinct, known]
        );
    }

    #[test]
    fn clearing_and_empty_slot_reuse_never_recycle_ids() {
        let mut catalog = LegacyHistoryCatalog::new(2, 100).unwrap();
        let a = publish(&mut catalog, 1, 1, &[1], 2, 1).id;
        let b = publish(&mut catalog, 2, 2, &[2], 3, 2).id;
        assert_eq!(catalog.remove(a).unwrap().unwrap().value, 1);
        let c = publish(&mut catalog, 3, 3, &[3], 4, 3).id;
        assert_eq!(catalog.slots.len(), 2);
        assert_eq!(catalog.slots[0].as_ref().unwrap().id, c);
        assert!(c > b);
        assert_eq!(catalog.charged_bytes(), 7);
        assert_eq!(catalog.clear().unwrap().len(), 2);
        assert!(catalog.is_empty());
        assert_eq!(catalog.charged_bytes(), 0);
        let d = publish(&mut catalog, 4, 4, &[4], 0, 4).id;
        assert!(d > c);
        assert!(catalog.get(a).is_none());
        assert!(catalog.remove(a).unwrap().is_none());
        assert!(matches!(
            catalog.touch(a),
            Err(LegacyCatalogError::NotFound(_))
        ));
        assert!(matches!(
            catalog.pin(a),
            Err(LegacyCatalogError::NotFound(_))
        ));
        assert!(matches!(
            catalog.unpin(a),
            Err(LegacyCatalogError::NotFound(_))
        ));
    }

    #[test]
    fn configuration_frontier_and_counter_overflows_fail_closed() {
        assert!(LegacyHistoryCatalog::<u8>::new(0, 1).is_err());
        assert!(LegacyHistoryCatalog::<u8>::new(1, 0).is_err());
        assert!(LegacyHistoryCatalog::<u8>::new(usize::MAX, 1).is_err());
        assert!(LegacyHistoryCatalog::<u8>::new(i32::MAX as usize, 1).is_err());
        let mut catalog = LegacyHistoryCatalog::new(1, 10).unwrap();
        assert!(matches!(
            catalog.publish(1, 1, vec![], 0, None, 1, 0),
            Err(LegacyCatalogError::IncompleteHistory { .. })
        ));
        assert!(matches!(
            catalog.publish(1, 1, vec![1], 2, None, 1, 0),
            Err(LegacyCatalogError::IncompleteHistory { .. })
        ));
        let id = publish(&mut catalog, 1, 1, &[1], 1, 0).id;
        let before = state(&catalog);
        catalog.clock = u64::MAX;
        assert!(matches!(
            catalog.touch(id),
            Err(LegacyCatalogError::Overflow("clock"))
        ));
        assert!(matches!(
            catalog.publish(2, 2, vec![2], 1, None, 1, 1),
            Err(LegacyCatalogError::Overflow("clock"))
        ));
        assert_eq!(state(&catalog), before);
        catalog.clock = 1;
        catalog.last_id = u64::MAX;
        assert!(matches!(
            catalog.publish(2, 2, vec![2], 1, None, 1, 1),
            Err(LegacyCatalogError::Overflow("entry ID"))
        ));
        assert_eq!(state(&catalog), before);
        catalog.slots[0].as_mut().unwrap().pin_count = usize::MAX;
        assert!(matches!(
            catalog.pin(id),
            Err(LegacyCatalogError::Overflow("pin count"))
        ));
        assert_eq!(catalog.get(id).unwrap().pin_count, usize::MAX);
    }

    #[test]
    fn maximum_byte_quota_is_accounted_without_overflow() {
        let mut catalog = LegacyHistoryCatalog::new(2, usize::MAX).unwrap();
        let a = publish(&mut catalog, 1, 1, &[1], usize::MAX, 0).id;
        let result = publish(&mut catalog, 2, 2, &[2], 1, 1);
        assert_eq!(
            result.retired,
            vec![RetiredHistory {
                id: a,
                source_session: 1
            }]
        );
        assert_eq!(catalog.charged_bytes(), 1);
        assert_eq!(catalog.slots.len(), 1);
    }
}
