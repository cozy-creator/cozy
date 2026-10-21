//! A directory in the static-origin layout (`refs/`, `blobs/aa/bb/<sha256>`, `tmp/`): an origin
//! of its own, and the datacenter volume tier every machine there reads before its origin and
//! fills from what it fetched (xs-041).
//!
//! - **Trust.** None: every byte read enters the store's admission, which hashes it.
//! - **Writes.** Public objects only, copied from the verified local object: temp, fsync,
//!   rename, so writers of one object converge and no reader sees part of one.
//! - **Room.** A write that finds the volume short, or that it refuses for space (a quota
//!   `statvfs` does not show), removes temps and blobs unused for the TTL, then the least
//!   recently used, and is skipped when still short.
//! - **Liveness.** A read the Fetcher abandoned as stalled (a hung mount cannot be cancelled)
//!   keeps the volume out of every fetch until it returns.

use std::fs::{self, File, OpenOptions};
use std::future::Future;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use platform::durable::Mode;

use super::origin::{parse_repo, split_name, Location, Origin};
use super::ObjectStore;
use crate::err::{Code, Refusal, Result};
use crate::format::{repo_key, ObjectRef, Repo};

/// Bytes per read or write call into the volume: each call is progress.
pub(crate) const STEP: usize = 8 << 20;

#[derive(Clone)]
pub struct VolumeOrigin(Arc<Inner>);

struct Inner {
    root: PathBuf,
    ttl: Duration,
    /// Operations given up on that have not returned.
    abandoned: AtomicUsize,
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

const RUNNING: u8 = 0;
const DONE: u8 = 1;
const ABANDONED: u8 = 2;

impl VolumeOrigin {
    /// The directory at `root`; an unused blob stays at least `ttl`.
    pub fn new(root: impl Into<PathBuf>, ttl: Duration) -> VolumeOrigin {
        VolumeOrigin(Arc::new(Inner {
            root: root.into(),
            ttl,
            abandoned: AtomicUsize::new(0),
        }))
    }

    /// The volume at `root` when it is a mount point (its device is not its parent's): a
    /// missing mount leaves a plain directory that must not fill the machine's disk. Read
    /// from cached attributes, so a hung mount cannot hold the caller.
    pub fn mounted(root: impl Into<PathBuf>, ttl: Duration) -> Option<VolumeOrigin> {
        let root = root.into();
        let device = |path: &Path| platform::durable::device(path).ok();
        (device(&root)? != device(root.parent()?)?).then(|| VolumeOrigin::new(root, ttl))
    }

    pub fn root(&self) -> &Path {
        &self.0.root
    }

    pub(crate) fn blob(&self, object: &ObjectRef) -> PathBuf {
        self.0.root.join(object.key())
    }

    /// No abandoned operation is still out.
    pub(crate) fn answering(&self) -> bool {
        self.0.abandoned.load(Ordering::Acquire) == 0
    }

    /// Its last use is now.
    pub(crate) fn touch(&self, object: &ObjectRef) {
        if let Ok(file) = File::open(self.blob(object)) {
            let _ = file.set_modified(SystemTime::now());
        }
    }

    /// Copy `object` from `store` onto the volume, unless an entry of its length is there and
    /// `replace` is not asked.
    pub(crate) fn write(
        &self,
        store: &dyn ObjectStore,
        object: &ObjectRef,
        replace: bool,
    ) -> Wrote {
        let path = self.blob(object);
        let present =
            fs::symlink_metadata(&path).is_ok_and(|m| m.is_file() && m.len() == object.length);
        if present && !replace {
            self.touch(object);
            return Wrote::Present;
        }
        if !self.room(object.length) {
            return Wrote::Skipped;
        }
        let mut written = self.copy(store, object, &path);
        if written
            .as_ref()
            .is_err_and(platform::durable::quota_or_space)
            && self.make_room(object.length) >= object.length
        {
            written = self.copy(store, object, &path);
        }
        match written {
            Ok(()) => Wrote::Written,
            Err(error) if platform::durable::quota_or_space(&error) => Wrote::Skipped,
            Err(_) => Wrote::Failed,
        }
    }

    fn copy(
        &self,
        store: &dyn ObjectStore,
        object: &ObjectRef,
        path: &Path,
    ) -> std::io::Result<()> {
        static UNIQUE: AtomicU64 = AtomicU64::new(0);
        let mut source = store.open(object)?;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let temp = self.0.root.join("tmp").join(format!(
            "{}.{}.{nanos}.{}",
            object.sha256,
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::Relaxed)
        ));
        let written = (|| -> std::io::Result<()> {
            fs::create_dir_all(temp.parent().unwrap())?;
            fs::create_dir_all(path.parent().unwrap())?;
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o444)
                .open(&temp)?;
            let mut buffer = vec![0; STEP];
            loop {
                let read = source.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                file.write_all(&buffer[..read])?;
            }
            file.sync_all()?;
            fs::rename(&temp, path)
        })();
        if written.is_err() {
            let _ = fs::remove_file(&temp);
        }
        written
    }

    /// Whether `bytes` more fit beside the volume's reserve (1/50, within 1–10 GiB), making
    /// room when they do not.
    fn room(&self, bytes: u64) -> bool {
        let Ok(space) = platform::durable::space(&self.0.root) else {
            return false;
        };
        let reserve = (space.total / 50).clamp(1 << 30, 10 << 30);
        let need = (reserve + bytes).saturating_sub(space.free);
        need == 0 || self.make_room(need) >= need
    }

    /// Remove temps and blobs unused for the TTL, then the least recently used blobs until
    /// `need` bytes are freed, unless no number of them would do: the bytes freed.
    fn make_room(&self, need: u64) -> u64 {
        let expired = SystemTime::now()
            .checked_sub(self.0.ttl)
            .unwrap_or(UNIX_EPOCH);
        let entries = |dir: &Path| -> Vec<PathBuf> {
            fs::read_dir(dir).map_or(vec![], |d| d.flatten().map(|e| e.path()).collect())
        };
        let mut freed = 0;
        let mut kept = vec![];
        let mut visit = |path: PathBuf, keep: bool, kept: &mut Vec<(SystemTime, u64, PathBuf)>| {
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                return;
            };
            let used = metadata.modified().unwrap_or(UNIX_EPOCH);
            if used < expired {
                if fs::remove_file(&path).is_ok() {
                    freed += metadata.len();
                }
            } else if keep {
                kept.push((used, metadata.len(), path));
            }
        };
        for temp in entries(&self.0.root.join("tmp")) {
            // A younger temp may be another machine's write in progress.
            visit(temp, false, &mut kept);
        }
        for first in entries(&self.0.root.join("blobs")) {
            for second in entries(&first) {
                for blob in entries(&second) {
                    visit(blob, true, &mut kept);
                }
            }
        }
        if freed + kept.iter().map(|(_, length, _)| length).sum::<u64>() < need {
            return freed;
        }
        kept.sort();
        for (_, length, path) in kept {
            if freed >= need {
                break;
            }
            if fs::remove_file(path).is_ok() {
                freed += length;
            }
        }
        freed
    }
}

/// `work` on a blocking thread, or `None` once `abandon` resolves first. An abandoned
/// operation keeps `volume` (when given) out of every fetch until it returns.
pub(crate) async fn blocking<T: Send + 'static>(
    volume: Option<&VolumeOrigin>,
    work: impl FnOnce() -> T + Send + 'static,
    abandon: impl Future<Output = ()>,
) -> Option<T> {
    let state = Arc::new(AtomicU8::new(RUNNING));
    let (owner, done) = (volume.cloned(), Arc::clone(&state));
    let handle = tokio::task::spawn_blocking(move || {
        let out = work();
        if done.swap(DONE, Ordering::AcqRel) == ABANDONED {
            if let Some(owner) = owner {
                owner.0.abandoned.fetch_sub(1, Ordering::AcqRel);
            }
        }
        out
    });
    tokio::select! {
        out = handle => out.ok(),
        () = abandon => {
            if let Some(volume) = volume {
                volume.0.abandoned.fetch_add(1, Ordering::AcqRel);
                if state.compare_exchange(RUNNING, ABANDONED, Ordering::AcqRel, Ordering::Acquire).is_err() {
                    volume.0.abandoned.fetch_sub(1, Ordering::AcqRel);
                }
            }
            None
        }
    }
}

#[async_trait]
impl Origin for VolumeOrigin {
    async fn repo_of(&self, name: &str) -> Result<Repo> {
        split_name(name)?;
        let path = self.0.root.join(repo_key(name));
        let read =
            tokio::task::spawn_blocking(move || fs::read(&path).map_err(|error| (path, error)))
                .await
                .map_err(|error| Refusal {
                    code: Code::IO_FAILED,
                    detail: error.to_string(),
                })?;
        let bytes = read.map_err(|(path, error)| Refusal {
            code: match error.kind() {
                std::io::ErrorKind::NotFound => Code::REF_NOT_FOUND,
                _ => Code::IO_FAILED,
            },
            detail: format!("{}: {error}", path.display()),
        })?;
        parse_repo(name, &bytes)
    }

    async fn locate(&self, objects: &[ObjectRef]) -> Result<Vec<Location>> {
        Ok(objects
            .iter()
            .map(|object| Location::File(self.blob(object)))
            .collect())
    }

    fn public(&self) -> bool {
        false
    }
}
