//! P2: host-resident DeltaNet checkpoint ring.
//!
//! The recurrent state is small per slot (~115 MiB at 48 linear layers) but
//! it cannot be rebuilt from tokens cheaply, and the GPU checkpoint pool is
//! bounded by VRAM. The ring keeps page-aligned `DeltaNetSnapshot` exports in
//! host memory, keyed by `(frontier, fingerprint)`, so a cold prefix reuse can
//! restore the exact state without a live GPU snapshot. The live state stays
//! engine-owned scratch; the persistent checkpoints are KVMem-managed.
//!
//! The ring can also mirror each checkpoint to one immutable file under a
//! cache directory. Files carry a model/layout identity so a ring written by a
//! different geometry is discarded instead of being decoded with the wrong
//! shape. The in-memory queue remains authoritative for LRU order; the disk
//! mirror is a startup seed, not a second cache.

use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const DNR_MAGIC: [u8; 4] = *b"DNR1";
const DNR_VERSION: u32 = 1;
const DNR_HEADER: usize = 40;
const DNR_SUFFIX: &str = "dnr";

struct DnEntry {
    frontier: u64,
    fp: u64,
    bytes: Vec<u8>,
    /// Immutable mirror of `bytes`; `None` when persistence is off, the entry
    /// was recovered from a different process, or the write failed.
    disk: Option<PathBuf>,
}

/// LRU ring of serialized DeltaNet states. `cap_bytes == 0` disables it.
pub struct DnRing {
    entries: VecDeque<DnEntry>,
    bytes: u64,
    cap_bytes: u64,
    disk_dir: Option<PathBuf>,
    disk_cap_bytes: u64,
    disk_bytes: u64,
    identity: u64,
    trace: bool,
}

impl DnRing {
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            cap_bytes,
            disk_dir: None,
            disk_cap_bytes: 0,
            disk_bytes: 0,
            identity: 0,
            trace: false,
        }
    }

    /// Build a ring that also mirrors checkpoints under `dir` and seeds itself
    /// from the files already there. `identity` is the ring's model/layout
    /// fingerprint; files with a different identity are discarded.
    pub fn new_persistent(
        cap_bytes: u64,
        dir: &Path,
        disk_cap_bytes: u64,
        identity: u64,
        trace: bool,
    ) -> Result<Self, String> {
        if cap_bytes == 0 || disk_cap_bytes == 0 {
            return Ok(Self::new(cap_bytes));
        }
        fs::create_dir_all(dir)
            .map_err(|e| format!("create DN ring directory {}: {e}", dir.display()))?;
        let mut ring = Self {
            entries: VecDeque::new(),
            bytes: 0,
            cap_bytes,
            disk_dir: Some(dir.to_path_buf()),
            disk_cap_bytes: disk_cap_bytes.min(cap_bytes),
            disk_bytes: 0,
            identity,
            trace,
        };
        ring.recover()?;
        Ok(ring)
    }

    pub fn enabled(&self) -> bool {
        self.cap_bytes > 0
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn total_bytes(&self) -> u64 {
        self.bytes
    }

    /// Payload bytes currently mirrored on disk (excludes file headers).
    pub fn disk_bytes(&self) -> u64 {
        self.disk_bytes
    }

    pub fn get(&self, frontier: u64, fp: u64) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|e| e.frontier == frontier && e.fp == fp)
            .map(|e| e.bytes.as_slice())
    }

    /// Insert or refresh one checkpoint. Returns `false` when the blob cannot
    /// fit even in an empty ring (the caller keeps its existing source).
    pub fn insert(&mut self, frontier: u64, fp: u64, bytes: Vec<u8>) -> bool {
        if !self.enabled() {
            return false;
        }
        let size = bytes.len() as u64;
        if size > self.cap_bytes {
            return false;
        }
        if let Some(pos) = self
            .entries
            .iter()
            .position(|e| e.frontier == frontier && e.fp == fp)
        {
            self.remove_entry_at(pos);
        }
        while self.bytes.saturating_add(size) > self.cap_bytes {
            if !self.pop_front() {
                break;
            }
        }

        // Keep the disk mirror inside its own cap without evicting entries
        // that are too large to persist at all.
        let mut disk = None;
        if self.disk_dir.is_some() && size <= self.disk_cap_bytes {
            while self.disk_bytes.saturating_add(size) > self.disk_cap_bytes {
                let Some(pos) = self.entries.iter().position(|e| e.disk.is_some()) else {
                    break;
                };
                self.remove_entry_at(pos);
            }
            if let Some(dir) = self.disk_dir.clone() {
                match write_entry(&dir, frontier, fp, self.identity, &bytes) {
                    Ok(path) => {
                        self.disk_bytes = self.disk_bytes.saturating_add(size);
                        disk = Some(path);
                    }
                    Err(e) => {
                        if self.trace {
                            eprintln!("[kvmem-dn] ring disk write frontier={frontier} failed: {e}");
                        }
                    }
                }
            }
        }

        self.bytes = self.bytes.saturating_add(size);
        self.entries.push_back(DnEntry {
            frontier,
            fp,
            bytes,
            disk,
        });
        true
    }

    fn recover(&mut self) -> Result<(), String> {
        let Some(dir) = self.disk_dir.clone() else {
            return Ok(());
        };
        let mut files = Vec::new();
        let read = fs::read_dir(&dir)
            .map_err(|e| format!("read DN ring directory {}: {e}", dir.display()))?;
        for entry in read {
            let entry = entry.map_err(|e| format!("read DN ring entry: {e}"))?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some(DNR_SUFFIX) {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            files.push((modified, path));
        }
        files.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        for (_, path) in files {
            let size = match fs::metadata(&path) {
                Ok(meta) => meta.len(),
                Err(e) => {
                    if self.trace {
                        eprintln!("[kvmem-dn] ring drop {}: {e}", path.display());
                    }
                    let _ = fs::remove_file(&path);
                    continue;
                }
            };
            if size < DNR_HEADER as u64 || size - DNR_HEADER as u64 > self.cap_bytes {
                let _ = fs::remove_file(&path);
                continue;
            }
            let (frontier, fp, bytes) = match read_entry(&path, self.identity) {
                Ok(v) => v,
                Err(e) => {
                    if self.trace {
                        eprintln!("[kvmem-dn] ring drop {}: {e}", path.display());
                    }
                    let _ = fs::remove_file(&path);
                    continue;
                }
            };
            let bytes_len = bytes.len() as u64;
            if bytes_len > self.cap_bytes || bytes_len > self.disk_cap_bytes {
                let _ = fs::remove_file(&path);
                continue;
            }
            if let Some(pos) = self
                .entries
                .iter()
                .position(|e| e.frontier == frontier && e.fp == fp)
            {
                self.remove_entry_at(pos);
            }
            while self.bytes.saturating_add(bytes_len) > self.cap_bytes {
                if !self.pop_front() {
                    break;
                }
            }
            while self.disk_bytes.saturating_add(bytes_len) > self.disk_cap_bytes {
                let Some(pos) = self.entries.iter().position(|e| e.disk.is_some()) else {
                    break;
                };
                self.remove_entry_at(pos);
            }
            self.bytes = self.bytes.saturating_add(bytes_len);
            self.disk_bytes = self.disk_bytes.saturating_add(bytes_len);
            self.entries.push_back(DnEntry {
                frontier,
                fp,
                bytes,
                disk: Some(path),
            });
        }
        if self.trace && !self.entries.is_empty() {
            eprintln!(
                "[kvmem-dn] ring recovered entries={} bytes={} disk={}",
                self.entries.len(),
                self.bytes,
                self.disk_bytes
            );
        }
        Ok(())
    }

    fn pop_front(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        self.remove_entry_at(0);
        true
    }

    fn remove_entry_at(&mut self, pos: usize) {
        let Some(entry) = self.entries.remove(pos) else {
            return;
        };
        let size = entry.bytes.len() as u64;
        self.bytes = self.bytes.saturating_sub(size);
        if let Some(path) = entry.disk {
            self.disk_bytes = self.disk_bytes.saturating_sub(size);
            if let Err(e) = fs::remove_file(&path) {
                if self.trace && e.kind() != std::io::ErrorKind::NotFound {
                    eprintln!("[kvmem-dn] ring unlink {}: {e}", path.display());
                }
            }
        }
    }
}

fn entry_path(dir: &Path, frontier: u64, fp: u64) -> PathBuf {
    dir.join(format!("{frontier:016x}-{fp:016x}.{DNR_SUFFIX}"))
}

fn write_entry(
    dir: &Path,
    frontier: u64,
    fp: u64,
    identity: u64,
    bytes: &[u8],
) -> Result<PathBuf, String> {
    let path = entry_path(dir, frontier, fp);
    let tmp = path.with_extension(format!("{DNR_SUFFIX}.tmp"));
    let result = (|| -> Result<(), String> {
        let mut file =
            fs::File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
        file.write_all(&DNR_MAGIC)
            .and_then(|_| file.write_all(&DNR_VERSION.to_le_bytes()))
            .and_then(|_| file.write_all(&identity.to_le_bytes()))
            .and_then(|_| file.write_all(&frontier.to_le_bytes()))
            .and_then(|_| file.write_all(&fp.to_le_bytes()))
            .and_then(|_| file.write_all(&(bytes.len() as u64).to_le_bytes()))
            .and_then(|_| file.write_all(bytes))
            .and_then(|_| file.sync_all())
            .map_err(|e| format!("write {}: {e}", tmp.display()))?;
        drop(file);
        fs::rename(&tmp, &path).map_err(|e| format!("rename {}: {e}", path.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result.map(|_| path)
}

fn read_entry(path: &Path, identity: u64) -> Result<(u64, u64, Vec<u8>), String> {
    let data = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if data.len() < DNR_HEADER {
        return Err("truncated header".into());
    }
    if data[0..4] != DNR_MAGIC {
        return Err("bad magic".into());
    }
    let version = u32::from_le_bytes(data[4..8].try_into().expect("4 bytes"));
    if version != DNR_VERSION {
        return Err(format!("unsupported version {version}"));
    }
    let stored_identity = u64::from_le_bytes(data[8..16].try_into().expect("8 bytes"));
    if stored_identity != identity {
        return Err(format!(
            "identity {stored_identity:016x} does not match {identity:016x}"
        ));
    }
    let frontier = u64::from_le_bytes(data[16..24].try_into().expect("8 bytes"));
    let fp = u64::from_le_bytes(data[24..32].try_into().expect("8 bytes"));
    let len = u64::from_le_bytes(data[32..40].try_into().expect("8 bytes"));
    if len != (data.len() - DNR_HEADER) as u64 {
        return Err(format!(
            "payload length {len} does not match file {}",
            data.len() - DNR_HEADER
        ));
    }
    Ok((frontier, fp, data[DNR_HEADER..].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let unique = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("hipfire-dn-ring-{}-{unique}", std::process::id()))
    }

    #[test]
    fn evicts_oldest_and_replaces_same_key() {
        let mut ring = DnRing::new(10);
        assert!(ring.insert(128, 1, vec![0; 4]));
        assert!(ring.insert(256, 2, vec![0; 4]));
        assert_eq!(ring.len(), 2);
        assert!(ring.insert(384, 3, vec![0; 4]));
        // 4+4+4 > 10: the oldest (128) is gone.
        assert!(ring.get(128, 1).is_none());
        assert!(ring.get(256, 2).is_some());
        assert!(ring.get(384, 3).is_some());
        // Same key replaces without growing the queue.
        assert!(ring.insert(256, 2, vec![1; 4]));
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.total_bytes(), 8);
    }

    #[test]
    fn disabled_or_oversized_refuses() {
        let mut ring = DnRing::new(0);
        assert!(!ring.insert(128, 1, vec![0; 4]));
        let mut ring = DnRing::new(4);
        assert!(!ring.insert(128, 1, vec![0; 5]));
        assert!(ring.get(128, 1).is_none());
    }

    #[test]
    fn persistent_ring_survives_restart() {
        let dir = temp_dir();
        let mut ring = DnRing::new_persistent(1024, &dir, 1024, 0x1234, false).unwrap();
        assert!(ring.insert(128, 7, vec![1; 16]));
        assert_eq!(ring.disk_bytes(), 16);
        drop(ring);

        let mut ring = DnRing::new_persistent(1024, &dir, 1024, 0x1234, false).unwrap();
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.get(128, 7), Some(&[1u8; 16][..]));
        assert_eq!(ring.disk_bytes(), 16);
        assert!(ring.insert(256, 8, vec![2; 16]));
        drop(ring);

        let ring = DnRing::new_persistent(1024, &dir, 1024, 0x1234, false).unwrap();
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.get(128, 7), Some(&[1u8; 16][..]));
        assert_eq!(ring.get(256, 8), Some(&[2u8; 16][..]));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn persistent_ring_discards_foreign_identity() {
        let dir = temp_dir();
        let mut ring = DnRing::new_persistent(1024, &dir, 1024, 0x1234, false).unwrap();
        assert!(ring.insert(128, 7, vec![1; 16]));
        drop(ring);

        let ring = DnRing::new_persistent(1024, &dir, 1024, 0x5678, false).unwrap();
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.disk_bytes(), 0);
        let files = fs::read_dir(&dir).unwrap().count();
        assert_eq!(files, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn persistent_ring_respects_disk_cap() {
        let dir = temp_dir();
        let mut ring = DnRing::new_persistent(1024, &dir, 32, 0x1234, false).unwrap();
        assert!(ring.insert(128, 1, vec![0; 16]));
        assert!(ring.insert(256, 2, vec![0; 16]));
        assert!(ring.insert(384, 3, vec![0; 16]));
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.disk_bytes(), 32);
        assert!(ring.get(128, 1).is_none());
        assert!(ring.get(256, 2).is_some());
        assert!(ring.get(384, 3).is_some());
        drop(ring);
        let ring = DnRing::new_persistent(1024, &dir, 32, 0x1234, false).unwrap();
        assert_eq!(ring.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }
}
