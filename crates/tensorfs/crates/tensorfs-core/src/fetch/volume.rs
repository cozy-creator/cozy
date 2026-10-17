//! The datacenter volume: one shared directory every machine in a datacenter reads before
//! the Hub and fills from what it fetched (design `datacenter-model-cache.md`).
//!
//! - **Layout.** `objects/aa/bb/<sha256>`, temps in `tmp/`. No index: a digest is its
//!   object's address, and an object's mtime its last use.
//! - **Trust.** None: every byte read goes through the Store's door, which hashes it.
//! - **Writes.** Copied from the verified local object: temp, fsync, rename, so writers of
//!   one object converge and no reader sees part of one.
//! - **Room.** A write that finds the volume short, or that the volume refuses for space (a
//!   network volume's quota, which `statvfs` does not show), removes temps and objects unused
//!   for the TTL, then the least recently used, and is skipped when still short.
//! - **Liveness.** An operation beats after every call into the volume; the fetcher abandons
//!   one silent beyond the transport's stall rule. A hard mount's call cannot be cancelled,
//!   so its thread waits on the kernel.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use crate::platform::durable::Mode;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};


use crate::admission::Disk;
use crate::ids::{Doc, ObjectRef};
use crate::store::Store;

/// Bytes per read or write call into the volume: each call is a beat.
const STEP: usize = 8 << 20;

/// A datacenter volume (or any shared directory) and how long an unused object stays.
#[derive(Debug, Clone)]
pub struct Volume {
    root: PathBuf,
    ttl: Duration,
}

/// How one write-back ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Wrote {
    Written,
    Present,
    /// The volume stayed short after making room.
    Skipped,
    Failed,
}

impl Volume {
    pub fn new(root: impl Into<PathBuf>, ttl: Duration) -> Volume {
        Volume {
            root: root.into(),
            ttl,
        }
    }

    /// The volume at `root` when it is a mount point (its device is not its parent's), else
    /// none: a missing mount leaves a plain directory that must not fill the machine's disk.
    /// Read from cached attributes, so a hung mount cannot hold the caller.
    pub fn mounted(root: impl Into<PathBuf>, ttl: Duration) -> Option<Volume> {
        let root = root.into();
        let device = |path: &Path| crate::platform::durable::device(path).ok();
        (device(&root)? != device(root.parent()?)?).then(|| Volume::new(root, ttl))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn object(&self, sha256: &str) -> PathBuf {
        let path = self.root.join("objects");
        path.join(&sha256[..2]).join(&sha256[2..4]).join(sha256)
    }

    /// `object`'s bytes, or `None` when the volume does not hold it whole.
    pub(crate) fn read(&self, object: &ObjectRef, op: &Op) -> Option<Vec<u8>> {
        if object.length > crate::limits::GRID_BYTES {
            return None;
        }
        let mut file = crate::platform::durable::open_nofollow(&self.object(&object.sha256), OpenOptions::new().read(true)).ok()?;
        op.beat();
        let metadata = file.metadata().ok()?;
        op.beat();
        if !metadata.is_file() || metadata.len() != object.length {
            return None;
        }
        let mut bytes = vec![0; object.length as usize];
        let mut at = 0;
        while at < bytes.len() {
            let end = (at + STEP).min(bytes.len());
            match file.read(&mut bytes[at..end]) {
                Ok(0) | Err(_) => return None,
                Ok(read) => at += read,
            }
            op.beat();
        }
        Some(bytes)
    }

    /// Its last use is now.
    pub(crate) fn touch(&self, sha256: &str, op: &Op) {
        if let Ok(file) = File::open(self.object(sha256)) {
            op.beat();
            let _ = file.set_modified(SystemTime::now());
        }
        op.beat();
    }

    /// Copy `object` from `store` onto the volume, unless an entry of its length is there and
    /// `replace` is not asked.
    pub(crate) fn write(
        &self,
        store: &Store,
        object: &ObjectRef,
        manifest: bool,
        replace: bool,
        op: &Op,
    ) -> Wrote {
        let path = self.object(&object.sha256);
        let present =
            fs::symlink_metadata(&path).is_ok_and(|m| m.is_file() && m.len() == object.length);
        op.beat();
        if present && !replace {
            self.touch(&object.sha256, op);
            return Wrote::Present;
        }
        if !self.room(object.length, op) {
            crate::stats::volume_skipped(object.length);
            return Wrote::Skipped;
        }
        let mut written = self.copy(store, object, manifest, &path, op);
        if written.as_ref().is_err_and(full) && self.make_room(object.length, op) >= object.length {
            written = self.copy(store, object, manifest, &path, op);
        }
        match written {
            Ok(()) => {
                crate::stats::volume_written(object.length);
                Wrote::Written
            }
            Err(error) if full(&error) => {
                crate::stats::volume_skipped(object.length);
                Wrote::Skipped
            }
            Err(_) => Wrote::Failed,
        }
    }

    /// One copy of the verified local object onto the volume: temp, fsync, rename.
    fn copy(
        &self,
        store: &Store,
        object: &ObjectRef,
        manifest: bool,
        path: &Path,
        op: &Op,
    ) -> std::io::Result<()> {
        let source: Option<Box<dyn Read>> = match manifest {
            true => (store.read_manifest(object).ok())
                .map(|m| m.canonical_bytes())
                .filter(|bytes| ObjectRef::of(bytes) == *object)
                .map(|bytes| Box::new(std::io::Cursor::new(bytes)) as Box<dyn Read>),
            false => {
                (store.open_verified(&object.sha256).ok()).map(|f| Box::new(f) as Box<dyn Read>)
            }
        };
        let Some(mut source) = source else {
            return Err(std::io::Error::other("the local object is not readable"));
        };
        let temp = self.root.join("tmp").join(format!(
            "{}.{}.{}",
            object.sha256,
            std::process::id(),
            crate::meta::now_nanos_unique()
        ));
        let written = (|| -> std::io::Result<()> {
            fs::create_dir_all(temp.parent().unwrap())?;
            fs::create_dir_all(path.parent().unwrap())?;
            op.beat();
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o444)
                .open(&temp)?;
            op.beat();
            let mut buffer = vec![0; STEP];
            loop {
                let read = source.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                file.write_all(&buffer[..read])?;
                op.beat();
            }
            file.sync_all()?;
            op.beat();
            fs::rename(&temp, path)?;
            op.beat();
            Ok(())
        })();
        if written.is_err() {
            let _ = fs::remove_file(&temp);
        }
        written
    }

    /// Whether `bytes` more fit beside the volume's reserve, making room when they do not.
    fn room(&self, bytes: u64, op: &Op) -> bool {
        let Ok(disk) = Disk::measure(&self.root) else {
            return false;
        };
        op.beat();
        let need = (disk.reserve() + bytes).saturating_sub(disk.available);
        if need == 0 {
            return true;
        }
        let freed = self.make_room(need, op);
        freed >= need || Disk::measure(&self.root).is_ok_and(|d| d.available >= d.reserve() + bytes)
    }

    /// Remove temps and objects unused for the TTL, then the least recently used objects until
    /// `need` bytes are freed, unless no number of them would do: the bytes freed. Entries
    /// another machine removed first are gone either way.
    fn make_room(&self, need: u64, op: &Op) -> u64 {
        let expired = SystemTime::now()
            .checked_sub(self.ttl)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let mut freed = 0;
        let remove = |path: &Path, length: u64, freed: &mut u64| {
            if fs::remove_file(path).is_ok() {
                *freed += length;
            }
            op.beat();
        };
        let mut kept = vec![];
        let fanout = |dir: &Path| -> Vec<PathBuf> {
            let entries =
                fs::read_dir(dir).map_or(vec![], |d| d.flatten().map(|e| e.path()).collect());
            op.beat();
            entries
        };
        let found = |path: PathBuf, kept: &mut Vec<(SystemTime, u64, PathBuf)>, freed: &mut u64| {
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                return;
            };
            op.beat();
            let used = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            match used < expired {
                true => remove(&path, metadata.len(), freed),
                false => kept.push((used, metadata.len(), path)),
            }
        };
        for temp in fanout(&self.root.join("tmp")) {
            // A temp younger than the TTL may be another machine's write in progress.
            found(temp, &mut vec![], &mut freed);
        }
        for first in fanout(&self.root.join("objects")) {
            for second in fanout(&first) {
                for object in fanout(&second) {
                    found(object, &mut kept, &mut freed);
                }
            }
        }
        // Room that even an emptied volume could not make is not made at all.
        if freed + kept.iter().map(|(_, length, _)| length).sum::<u64>() < need {
            return freed;
        }
        kept.sort();
        for (_, length, path) in kept {
            if freed >= need {
                break;
            }
            remove(&path, length, &mut freed);
        }
        freed
    }
}

/// One volume operation's liveness: when it last moved, and whether the fetcher gave up on it.
pub(crate) struct Op {
    began: Instant,
    /// Nanoseconds from `began` to its last beat.
    beat: AtomicU64,
    abandoned: AtomicBool,
    /// The longest silence any of its fetcher's volume operations has shown between beats.
    worst: std::sync::Arc<AtomicU64>,
}

impl Op {
    pub fn new(worst: std::sync::Arc<AtomicU64>) -> Op {
        Op {
            began: Instant::now(),
            beat: AtomicU64::new(0),
            abandoned: AtomicBool::new(false),
            worst,
        }
    }

    /// It moved: its gap teaches the worst, unless it was already judged stalled.
    fn beat(&self) {
        let now = self.began.elapsed().as_nanos() as u64;
        let gap = now.saturating_sub(self.beat.swap(now, Ordering::AcqRel));
        if !self.abandoned.load(Ordering::Acquire) {
            self.worst.fetch_max(gap, Ordering::AcqRel);
        }
    }

    pub fn silent(&self) -> Duration {
        self.began
            .elapsed()
            .saturating_sub(Duration::from_nanos(self.beat.load(Ordering::Acquire)))
    }

    pub fn abandon(&self) {
        self.abandoned.store(true, Ordering::Release);
    }
}

/// A write refused for space: the disk is full, or the volume's quota is.
fn full(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        std::io::ErrorKind::StorageFull | std::io::ErrorKind::QuotaExceeded
    )
}
