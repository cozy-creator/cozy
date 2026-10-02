//! Immutable blobs at `blobs/xx/yy/<hex>` plus immutable manifests at
//! `manifests/xx/yy/<hex>.json`.
//!
//! Tensor parts, blob segments and documents admit through ONE put path; the store knows
//! no shapes (tensorfs.md §1). Identity is `sha256(stored_bytes)`, so an identity-bearing
//! document admits only as its exact canonical bytes — a pretty twin lands at a different
//! id and refuses `put_expected`.
//!
//! Admission is `writing → verified-temp → installed`: temp in `tmp/`, hash while writing,
//! fsync, 0444, then a no-clobber `link()` into place plus a parent-directory fsync. A
//! crash at any stage leaves a temp, never a partial object under `objects/`. Which of those
//! syncs an admission pays depends on the disk the Store lives on ([`crate::disk`]): none on
//! an ephemeral one, and on a persistent one none inside a pull, whose batch sync
//! covers them ([`crate::unsynced`]).
//!
//! `contains` is a PRESENCE HINT and nothing more. Trusting bytes requires a durable
//! verification record bound to object id, length, file identity (dev, ino), size and
//! change token (mtime). Any mismatch invalidates the record and
//! forces a rehash; a rehash that disagrees removes the corrupt bytes.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::{fchown, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;

use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{hex64, Doc, ObjectRef};
use crate::sha256::{self, Sha256};

/// An addition already held by a cache repository needs no mutation. Authored local
/// aliases and removal operations never gain cache idempotency.
fn cached_repository_satisfies(
    repository: &crate::repository::Repository,
    mutations: &[crate::repository::Mutation],
) -> bool {
    use crate::repository::Mutation;
    !mutations.is_empty()
        && mutations.iter().all(|mutation| {
            if &repository.repo != mutation.repo() || repository.repo.org == "local" {
                return false;
            }
            match mutation {
                Mutation::PutCheckpoint { manifest, .. } => repository
                    .checkpoints
                    .iter()
                    .any(|checkpoint| &checkpoint.manifest == manifest),
                Mutation::UpdateRelease {
                    version,
                    set,
                    remove,
                    ..
                } if remove.is_empty() => repository.releases.iter().any(|release| {
                    &release.version == version
                        && !release.yanked
                        && set
                            .iter()
                            .all(|lane| release.lanes.iter().any(|held| held.same_target(lane)))
                }),
                _ => false,
            }
        })
}

/// Merge cache additions against the repository held by SQLite coordination. Only the
/// requested lane needs an unchanged observation; unrelated lanes/checkpoints commute.
fn merge_cached_repository(
    observed: Option<&[u8]>,
    current: Option<crate::repository::Repository>,
    mutations: &[crate::repository::Mutation],
) -> Result<Option<crate::repository::Repository>> {
    use crate::repository::{Mutation, Repository};
    let previous = observed.map(Repository::parse).transpose()?;
    if previous.is_some() && current.is_none() {
        return refuse(
            Code::REPOSITORY_CONFLICT,
            "repository was removed during pull",
        );
    }
    let mut current = current;
    for mutation in mutations {
        if current
            .as_ref()
            .is_some_and(|repo| cached_repository_satisfies(repo, std::slice::from_ref(mutation)))
        {
            continue;
        }
        let mut mutation = mutation.clone();
        match &mut mutation {
            Mutation::PutCheckpoint { .. } => {}
            Mutation::UpdateRelease {
                version,
                set,
                remove,
                expected_revision,
                ..
            } if remove.is_empty() => {
                let old_release = previous.as_ref().and_then(|repo| {
                    repo.releases
                        .iter()
                        .find(|release| &release.version == version)
                });
                let latest = current.as_ref().and_then(|repo| {
                    repo.releases
                        .iter()
                        .find(|release| &release.version == version)
                });
                if latest.is_some_and(|release| release.yanked)
                    || set.iter().any(|wanted| {
                        let before = old_release.and_then(|release| {
                            release.lanes.iter().find(|lane| lane.lane == wanted.lane)
                        });
                        let actual = latest.and_then(|release| {
                            release.lanes.iter().find(|lane| lane.lane == wanted.lane)
                        });
                        !actual.is_some_and(|lane| lane.same_target(wanted)) && actual != before
                    })
                {
                    return refuse(
                        Code::REPOSITORY_CONFLICT,
                        "requested release lane changed during pull",
                    );
                }
                *expected_revision = latest.map_or(0, |release| release.revision);
            }
            _ => {
                return refuse(
                    Code::REPOSITORY_CONFLICT,
                    "cache commits only add checkpoints and lanes",
                )
            }
        }
        let bytes = current.as_ref().map(Doc::canonical_bytes);
        current = crate::repository::apply(bytes.as_deref(), &mutation)?;
    }
    Ok(current)
}

pub(crate) const BUF: usize = 1 << 20;

#[cfg(target_os = "linux")]
pub(crate) const O_NOFOLLOW: i32 = 0o400_000;
#[cfg(target_os = "macos")]
pub(crate) const O_NOFOLLOW: i32 = 0x0100;

/// ENOSPC on Linux and macOS alike; EDQUOT differs.
const ENOSPC: i32 = 28;
#[cfg(target_os = "linux")]
const EDQUOT: i32 = 122;
#[cfg(target_os = "macos")]
const EDQUOT: i32 = 69;

/// An I/O failure, classified.
///
/// Out of space and denied access need an external change before retrying. The difference is the
/// whole of tfs-064. Before it, a full container disk raised `IO_FAILED` — the same code as
/// a bad sector or a revoked permission — so `pod-supervisor` classified it `Resumable` and
/// the owner re-issued the fetch, which re-planned, re-asked, and failed at the same byte.
/// On rented hardware, forever. Naming the condition is what lets a caller stop.
pub(crate) fn classify_io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    // A full descriptor table is the process's shortage, not the Store's: every open in
    // the Store says so the same way, and a transfer treats it as backpressure.
    if crate::descriptors::exhausted(&e) {
        return Refusal {
            code: Code::FD_HEADROOM,
            detail: format!("{}: no file descriptor ({e})", what.as_ref()),
        };
    }
    if matches!(e.raw_os_error(), Some(ENOSPC) | Some(EDQUOT)) {
        return Refusal {
            code: Code::CAPACITY_EXHAUSTED,
            detail: format!("{}: the filesystem is full ({e})", what.as_ref()),
        };
    }
    Refusal {
        code: if e.kind() == std::io::ErrorKind::PermissionDenied {
            Code::PERMISSION_DENIED
        } else {
            Code::IO_FAILED
        },
        detail: format!("{}: {e}", what.as_ref()),
    }
}

/// SQLite's "database or disk is full" is the disk's verdict, exactly as ENOSPC is. As
/// IO_FAILED it read as a transient catalog fault and bought the object three more times.
pub(crate) fn classify_sql(what: impl AsRef<str>, e: rusqlite::Error) -> Refusal {
    let full = matches!(&e, rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::DiskFull);
    Refusal {
        code: if full {
            Code::CAPACITY_EXHAUSTED
        } else {
            Code::IO_FAILED
        },
        detail: format!("{}: {e}", what.as_ref()),
    }
}

/// The short local spelling used by every call site in this module.
fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    classify_io(what, e)
}

// ---------------------------------------------------- durable verification record

/// One `tensorfs_verified_blobs` row — the local cache fact that these exact bytes were hashed
/// once. Bound to everything that could change underneath it; loss merely forces a rehash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VRecord {
    pub sha256: String,
    pub length: u64,
    pub dev: u64,
    pub ino: u64,
    pub mtime_s: u64,
    pub mtime_ns: u64,
}

impl VRecord {
    fn of(hex: &str, md: &fs::Metadata) -> VRecord {
        VRecord {
            sha256: hex.to_string(),
            length: md.len(),
            dev: md.dev(),
            ino: md.ino(),
            mtime_s: md.mtime().max(0) as u64,
            mtime_ns: md.mtime_nsec().max(0) as u64,
        }
    }
    /// The first disagreeing bound fact, or None when the record still stands.
    fn disagreement(&self, hex: &str, md: &fs::Metadata) -> Option<String> {
        if self.dev != md.dev() {
            return Some(format!("file device {} != {}", self.dev, md.dev()));
        }
        let now = VRecord::of(hex, md);
        if self.sha256 != now.sha256 {
            return Some(format!("object id {} != {}", self.sha256, now.sha256));
        }
        if self.ino != now.ino {
            return Some(format!("file inode {} != {}", self.ino, now.ino));
        }
        if self.length != now.length {
            return Some(format!("size {} != {}", self.length, now.length));
        }
        if self.mtime_s != now.mtime_s || self.mtime_ns != now.mtime_ns {
            return Some(format!(
                "change token {}.{:09} != {}.{:09}",
                self.mtime_s, self.mtime_ns, now.mtime_s, now.mtime_ns
            ));
        }
        None
    }
}

// ---------------------------------------------------------------- fault injection

/// Dev-only kill points. `tfs put --fault <stage>` reaches the stage, announces it, and
/// waits to be killed from outside — a real SIGKILL to a real process, never a simulation.
#[derive(Debug, Clone, Default)]
pub struct Fault {
    pub stage: Option<String>,
    pub ready: Option<PathBuf>,
}

pub const FAULT_STAGES: &[&str] = &[
    "open",    // temp created, no bytes
    "partial", // first chunk written, temp is a partial object
    "written", // all bytes written, not yet durable
    "fsync",   // durable temp, still 0600, not linked
    "chmod",   // 0444 verified temp, one link() from installed
    "linked",  // installed, parent dir not yet fsynced, temp not yet unlinked
    "record",  // verification record written; nothing left to lose
];

impl Fault {
    pub(crate) fn hit(&self, stage: &str) {
        if self.stage.as_deref() != Some(stage) {
            return;
        }
        if let Some(p) = &self.ready {
            let _ = fs::write(p, format!("{stage} {}\n", std::process::id()));
        }
        eprintln!("FAULT-READY {stage} pid={}", std::process::id());
        loop {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

// ---------------------------------------------------------------- the store

/// Where a store reports the bytes it moves. Every chunk hashed on the way IN
/// (`put_stream_held`), every chunk rehashed for a verification (`verify`), and every byte
/// a layer above carries into a candidate on the store's behalf (an inline part, a
/// verified inherit) is announced as it happens, and every object that wins admission is
/// counted — so a supervisor holding the other end can tell "still working" from
/// "stopped" without a clock of its own.
pub trait Progress: Send + Sync {
    fn moved(&self, bytes: u64);
    fn admitted(&self);
    /// `bytes` already announced as moved were thrown away: the stream behind a candidate
    /// failed or proved to be the wrong object. Progress that kept them would claim bytes
    /// the store does not have, and a pull that re-asks would pass 100% before it ends.
    fn discarded(&self, _bytes: u64) {}
}

#[derive(Clone)]
pub struct Store {
    root: PathBuf,
    progress: Option<std::sync::Arc<dyn Progress>>,
    /// The optional mounted immutable object cache THIS Store is bound to, read off
    /// [`REPO_CACHE_BINDING`] when the handle was made. See that constant for why the
    /// binding is a property of the Store and not an argument of every operation.
    cache: Option<crate::repo_cache::RepoObjectCache>,
    /// The optional disk budget THIS Store is held to, read off [`DISK_BUDGET`] when the
    /// handle was made. `None` is the default and means every operation behaves exactly as
    /// it did before tfs-064. See that constant.
    budget: Option<u64>,
    /// This process's verified-blob index for the catalog, shared with every other handle.
    trust: std::sync::Arc<crate::catalog::Trust>,
    /// The filesystem this Store lives on, and so what a crash can take from it.
    disk: crate::disk::Filesystem,
    /// The pull whose batch sync covers this handle's admissions (persistent disk).
    epoch: Option<std::sync::Arc<crate::unsynced::Epoch>>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("root", &self.root)
            .field("observed", &self.progress.is_some())
            .field("repo_cache", &self.cache.as_ref().map(|c| c.root()))
            .field("disk", &self.disk)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Put {
    pub obj: ObjectRef,
    /// This process won the no-clobber admission (vs. found the id already installed).
    pub admitted: bool,
}

/// The store's own scratch names under `tmp/`. [`Store::reap`] collects exactly these two
/// prefixes and nothing else, so a crashed writer of EITHER destination cannot leak a
/// partial multi-gigabyte temp forever, and a `tmp/` entry that is somebody else's stays
/// somebody else's.
pub(crate) const PUT_TEMP: &str = "put-";
pub(crate) const STAGE_TEMP: &str = "stage-";

/// A durable temp whose bytes are PROVEN and whose destination is not yet decided.
///
/// The lock is still held and the mode is still 0600, so nothing can read it and `reap`
/// can tell it has a live writer. See [`Store::stream_verified`].
pub(crate) struct Verified {
    pub(crate) tmp: PathBuf,
    pub(crate) file: File,
    pub(crate) sha256: String,
    pub(crate) length: u64,
}

/// Bytes proven to be `sha256` THROUGH THE DESCRIPTOR THIS HOLDS.
///
/// The point of the type is the fd, not the fields: a `VerifiedFile` cannot be built from a
/// pathname and a promise, so there is no way to spell "verified over there, read over
/// here". Reads start at offset 0 and the descriptor keeps the inode alive for the handle's
/// whole life, so a replacement at the path — same length or not — cannot reach them.
#[derive(Debug)]
pub struct VerifiedFile {
    file: File,
    sha256: String,
    len: u64,
}

impl VerifiedFile {
    /// The id these bytes were proven to have.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The length observed on the verified descriptor (`fstat`, never a second `stat`).
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The pinned descriptor. Whatever a caller does with it, it is still the file that was
    /// verified — that is the whole guarantee, and it travels with the handle.
    pub fn into_file(self) -> File {
        self.file
    }

    /// Positional read through the pinned descriptor — the fd, never the name. Shared-ref
    /// on purpose: `pread` moves no cursor, so concurrent readers need no lock.
    pub fn read_exact_at(&self, buf: &mut [u8], off: u64) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file.read_exact_at(buf, off)
    }

    /// `fstat` on the pinned descriptor. A truncation of THIS inode is visible here; a
    /// replacement at the pathname is not — by design, that swap cannot reach this handle.
    pub fn current_len(&self) -> std::io::Result<u64> {
        self.file.metadata().map(|m| m.len())
    }
}

impl Read for VerifiedFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

impl Seek for VerifiedFile {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Verified { rehashed: bool },
    Invalidated { why: String },
    CorruptRemoved { why: String },
}

#[derive(Debug, Default, Clone)]
pub struct Scrub {
    pub scanned: usize,
    pub sampled: usize,
    pub rehashed: usize,
    pub removed: usize,
    pub bytes_hashed: u64,
}

impl Store {
    /// Open a valid Store, or initialize an absent/empty real directory.
    ///
    /// A symlink, non-directory, partial Store, or directory containing foreign state refuses.
    /// There is no old-layout lookup or fallback root.
    /// Open the Store at `root`, creating it when absent or empty. Concurrent calls on one
    /// new root serialize on the root directory: exactly one creates it, the rest open it.
    pub fn ensure(root: &Path) -> Result<Store> {
        match fs::symlink_metadata(root) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(root)
                    .map_err(|error| io(format!("create Store root {}", root.display()), error))?;
            }
            Err(error) => return Err(io(format!("stat Store root {}", root.display()), error)),
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return refuse(
                    Code::STORE_ERA,
                    format!("{} is not a real Store directory", root.display()),
                )
            }
            Ok(_) => {}
        }
        let _creation = creation_lock(root)?;
        let empty = fs::read_dir(root)
            .map_err(|error| io(format!("read Store root {}", root.display()), error))?
            .next()
            .transpose()
            .map_err(|error| io(format!("read Store root {}", root.display()), error))?
            .is_none();
        if empty {
            Store::init_locked(root)
        } else {
            validate_store_root(root)?;
            let catalog_path = root.join("tensorfs.sqlite");
            match fs::symlink_metadata(&catalog_path) {
                Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {}
                Ok(_) => {
                    return refuse(
                        Code::DURABILITY_UNPROVEN,
                        format!("{} is not a real catalog file", catalog_path.display()),
                    )
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io("stat TensorFS catalog", error)),
            }
            let trust = crate::catalog::Trust::of(root);
            trust.validate()?;
            open_validated(root)
        }
    }

    pub fn init(root: &Path) -> Result<Store> {
        if matches!(fs::symlink_metadata(root), Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink())
        {
            let _creation = creation_lock(root)?;
            return Store::init_locked(root);
        }
        Store::init_locked(root)
    }

    fn init_locked(root: &Path) -> Result<Store> {
        match fs::symlink_metadata(root) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(_) => {
                return refuse(
                    Code::STORE_ERA,
                    format!("{} is not a real Store directory", root.display()),
                )
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(root)
                    .map_err(|error| io(format!("create Store root {}", root.display()), error))?;
            }
            Err(error) => return Err(io(format!("stat Store root {}", root.display()), error)),
        }
        if fs::read_dir(root)
            .map_err(|error| io(format!("read Store root {}", root.display()), error))?
            .next()
            .transpose()
            .map_err(|error| io(format!("read Store root {}", root.display()), error))?
            .is_some()
        {
            return refuse(
                Code::STORE_ERA,
                format!(
                    "{} is not empty; use Store::ensure or Store::open for an existing Store",
                    root.display()
                ),
            );
        }
        for d in ["repos", "manifests", "blobs", "checkouts", "staging", "tmp"] {
            fs::create_dir_all(root.join(d)).map_err(|e| io(format!("mkdir {d}"), e))?;
        }
        crate::catalog::Catalog::initialize(root)?;
        Ok(Store {
            root: root.to_path_buf(),
            progress: None,
            cache: read_repo_cache_binding(root),
            budget: read_disk_budget(root),
            trust: crate::catalog::Trust::of(root),
            disk: crate::disk::filesystem(root),
            epoch: None,
        })
    }

    /// The same store, announcing every byte it moves to `sink`.
    pub fn observed(self, sink: std::sync::Arc<dyn Progress>) -> Store {
        Store {
            progress: Some(sink),
            ..self
        }
    }

    /// Announce `bytes` of progress. The store calls this for every chunk it streams or
    /// rehashes; a layer that admits bytes without streaming them through the store (an
    /// inline part, an inherit proven by record) calls it for the bytes it carried.
    pub fn moved(&self, bytes: u64) {
        if let Some(sink) = &self.progress {
            sink.moved(bytes);
        }
    }

    /// `bytes` already announced as moved were thrown away.
    pub(crate) fn discarded(&self, bytes: u64) {
        if let Some(sink) = &self.progress {
            sink.discarded(bytes);
        }
    }

    /// The three authoritative namespaces identify a TensorFS store. Old mixed-object and
    /// state.json layouts are a prelaunch hardcut and never dual-read.
    pub fn open(root: &Path) -> Result<Store> {
        validate_store_root(root)?;
        open_validated(root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The filesystem this Store lives on. See [`crate::disk`].
    pub fn disk(&self) -> crate::disk::Filesystem {
        self.disk
    }

    pub fn disk_class(&self) -> crate::disk::DiskClass {
        self.disk.class
    }

    /// Whether an admission makes its own bytes and name durable before it reports: not on
    /// an ephemeral disk, and not under a pull's epoch, whose batch sync covers them.
    pub(crate) fn flushes(&self) -> bool {
        match self.disk.class {
            crate::disk::DiskClass::Ephemeral => false,
            crate::disk::DiskClass::Network => true,
            crate::disk::DiskClass::Persistent => self.epoch.is_none(),
        }
    }

    /// The same Store, its admissions covered by `epoch`.
    pub(crate) fn syncing(&self, epoch: Option<std::sync::Arc<crate::unsynced::Epoch>>) -> Store {
        Store {
            epoch,
            ..self.clone()
        }
    }

    /// Make this handle's unsynced admissions durable now, names included. Without an epoch
    /// each admission already made itself durable (or, on an ephemeral disk, needs nothing).
    pub(crate) fn sync_admitted(&self) -> Result<()> {
        match &self.epoch {
            Some(epoch) => epoch.sync_now_named(),
            None => Ok(()),
        }
    }

    /// Sync the last batch and retire the epoch's marker.
    pub(crate) fn finish_admitted(&self) -> Result<()> {
        match &self.epoch {
            Some(epoch) => epoch.finish().map(|_| ()),
            None => Ok(()),
        }
    }

    /// The same Store as if it lived on a disk of `class`: benchmarks of one real disk.
    #[cfg(test)]
    pub(crate) fn with_disk_class(self, class: crate::disk::DiskClass) -> Store {
        Store {
            disk: crate::disk::Filesystem {
                class,
                kind: "injected",
            },
            ..self
        }
    }

    pub(crate) fn trust(&self) -> &crate::catalog::Trust {
        &self.trust
    }

    /// The mounted immutable object cache this Store is bound to, or `None`.
    ///
    /// Every production consumer of the cache reads it from HERE and no operation takes one
    /// as an argument. See [`REPO_CACHE_BINDING`].
    pub fn repo_cache(&self) -> Option<&crate::repo_cache::RepoObjectCache> {
        self.cache.as_ref()
    }

    /// Record (`Some`) or clear (`None`) this Store's cache binding, and return the handle
    /// that carries the result. DECLARATIVE on purpose: a setup that says nothing about the
    /// cache is a setup that says the Store has none, so a pod whose volume went away stops
    /// mirroring instead of quietly keeping a stale binding.
    ///
    /// **This touches the cache path exactly zero times.** The path is checked as a STRING
    /// -- absolute, not the Store and not inside it -- and never stat'd, because a stat on a
    /// hard NFS mount that has stopped answering blocks forever and setup would then hang on
    /// the thing that is only allowed to cost latency. Whether the mount is really there is
    /// answered per object, by the cache itself, as `unavailable`.
    pub fn bind_repo_cache(self, cache: Option<&Path>) -> Result<Store> {
        let path = self.root.join(REPO_CACHE_BINDING);
        let Some(cache) = cache else {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io("clear repo cache binding", error)),
            }
            fsync_dir(&self.root)?;
            return Ok(Store {
                cache: None,
                ..self
            });
        };
        validate_repo_cache_root(cache, &self.root)?;
        let mut bytes = cache.as_os_str().as_encoded_bytes().to_vec();
        bytes.push(b'\n');
        let temp = self.root.join(format!(
            "tmp/repo-cache.{}.{}",
            std::process::id(),
            now_nanos_unique()
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o644)
            .custom_flags(O_NOFOLLOW)
            .open(&temp)
            .map_err(|error| io("open repo cache binding temp", error))?;
        let written = file
            .write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| io("write repo cache binding", error));
        drop(file);
        if let Err(refusal) = written {
            let _ = fs::remove_file(&temp);
            return Err(refusal);
        }
        if let Err(error) = fs::rename(&temp, &path) {
            let _ = fs::remove_file(&temp);
            return Err(io("install repo cache binding", error));
        }
        fsync_dir(&self.root)?;
        Ok(Store {
            cache: Some(crate::repo_cache::RepoObjectCache::new(cache)),
            ..self
        })
    }

    /// The disk budget this Store is held to, or `None`. See [`DISK_BUDGET`].
    pub fn disk_budget(&self) -> Option<u64> {
        self.budget
    }

    /// Record (`Some`) or clear (`None`) this Store's disk budget, and return the handle
    /// that carries the result. DECLARATIVE, exactly like the cache binding: a setup that
    /// says nothing about the budget is a setup that says the Store has none, so a pod
    /// re-provisioned without one stops enforcing instead of quietly keeping a stale number.
    pub fn bind_disk_budget(self, budget: Option<u64>) -> Result<Store> {
        let path = self.root.join(DISK_BUDGET);
        let Some(budget) = budget else {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io("clear disk budget", error)),
            }
            fsync_dir(&self.root)?;
            return Ok(Store {
                budget: None,
                ..self
            });
        };
        if budget == 0 {
            return refuse(
                Code::NUMBER_RANGE,
                "a disk budget of zero admits nothing; clear it instead of setting it to 0",
            );
        }
        let bytes = format!("{budget}\n").into_bytes();
        let temp = self.root.join(format!(
            "tmp/disk-budget.{}.{}",
            std::process::id(),
            now_nanos_unique()
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o644)
            .custom_flags(O_NOFOLLOW)
            .open(&temp)
            .map_err(|error| io("open disk budget temp", error))?;
        let written = file
            .write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| io("write disk budget", error));
        drop(file);
        if let Err(refusal) = written {
            let _ = fs::remove_file(&temp);
            return Err(refusal);
        }
        if let Err(error) = fs::rename(&temp, &path) {
            let _ = fs::remove_file(&temp);
            return Err(io("install disk budget", error));
        }
        fsync_dir(&self.root)?;
        Ok(Store {
            budget: Some(budget),
            ..self
        })
    }

    /// What this Store is holding, in bytes: every installed blob and manifest.
    ///
    /// This is deliberately NOT the filesystem's used space. It counts what TensorFS put
    /// there and nothing else, so the number means the same thing on a shared container
    /// disk as on a dedicated one, and computing it needs no syscall about any other
    /// tenant. Scratch under `tmp/` is excluded: an in-flight temp is charged to the
    /// admission that is already being checked against the budget, and counting it twice
    /// would refuse a pull that fits.
    ///
    /// **Staged carriers ARE counted, and that is a preservation rather than an addition.**
    /// Before tfs-067 a fetched source shard was an ordinary blob and this number included
    /// it; a carrier that left the CAS and left the arithmetic with it would quietly make
    /// the budget mean something narrower than it did, and a store with 200 GB of carriers
    /// on it would report headroom it does not have. They are not reclaimable by `gc` —
    /// they are not in any census — which is exactly why the number that decides whether an
    /// admission fits must see them.
    pub fn occupancy(&self) -> Result<u64> {
        let mut total: u64 = crate::staging::bytes(self)?
            .checked_add(crate::transport::sources::bytes(self)?)
            .ok_or_else(|| Refusal {
                code: Code::ARITH_OVERFLOW,
                detail: "prefix occupancy overflow".into(),
            })?;
        for (namespace, suffix) in [("blobs", ""), ("manifests", ".json")] {
            let base = self.root.join(namespace);
            if !base.is_dir() {
                continue;
            }
            for first in read_dir(&base)? {
                for second in read_dir(&first)? {
                    for object in read_dir(&second)? {
                        let Some(name) = object.file_name().and_then(|value| value.to_str()) else {
                            continue;
                        };
                        let stem = name.strip_suffix(suffix).unwrap_or(name);
                        if suffix.is_empty() == name.ends_with(".json")
                            || hex64("object", stem).is_err()
                        {
                            continue;
                        }
                        let metadata = match fs::symlink_metadata(&object) {
                            Ok(metadata) => metadata,
                            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                            Err(error) => return Err(io("stat store object", error)),
                        };
                        if !metadata.is_file() || metadata.file_type().is_symlink() {
                            continue;
                        }
                        total = total.checked_add(metadata.len()).ok_or_else(|| Refusal {
                            code: Code::ARITH_OVERFLOW,
                            detail: "store occupancy overflow".into(),
                        })?;
                    }
                }
            }
        }
        Ok(total)
    }

    /// Refuse an admission that has not started yet, when the bytes it would ADD would put
    /// this Store past its recorded budget.
    ///
    /// `adding` is what the caller has already established it must move -- for a pull that
    /// is `FetchPlan::wanted_bytes()`, the bytes not already held, so dedup is credited
    /// before anything is refused. A Store with no budget admits everything, which is the
    /// default and the pre-tfs-064 behaviour.
    pub fn admits(&self, adding: u64) -> Result<()> {
        let Some(budget) = self.budget else {
            return Ok(());
        };
        let occupancy = self.occupancy()?;
        let after = occupancy.saturating_add(adding);
        if after <= budget {
            return Ok(());
        }
        refuse(
            Code::CAPACITY_EXHAUSTED,
            format!(
                "this store holds {occupancy} B and this admission would add {adding} B, \
                 which is {} B past its {budget} B disk budget",
                after - budget
            ),
        )
    }

    /// Prepare the Store for read leases created by a separate unprivileged uid.
    ///
    /// TensorFS owns these internal paths and exact modes. Callers request the policy; they do
    /// not create or chmod Store internals themselves.
    pub fn prepare_readers(&self) -> Result<()> {
        for (path, mode) in [
            (self.root.clone(), 0o755),
            (self.root.join("repos"), 0o755),
            (self.root.join("manifests"), 0o755),
            (self.root.join("blobs"), 0o755),
            (self.root.join("checkouts"), 0o755),
            (crate::staging::area(self), 0o755),
            (self.root.join("tmp"), 0o755),
            (self.root.join("tmp/leases"), 0o1733),
            (self.root.join("tmp/writers"), 0o755),
        ] {
            prepare_reader_dir(&path, mode)?;
        }
        let lock_path = self.root.join("tmp/writers/recovery.lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o666)
            .custom_flags(O_NOFOLLOW)
            .open(&lock_path)
            .map_err(|error| io(format!("open {}", lock_path.display()), error))?;
        if !lock
            .metadata()
            .map_err(|error| io("stat recovery lock", error))?
            .is_file()
        {
            return refuse(
                Code::NOT_REGULAR_FILE,
                format!("{} is not a regular file", lock_path.display()),
            );
        }
        lock.set_permissions(fs::Permissions::from_mode(0o666))
            .map_err(|error| io("chmod recovery lock", error))?;
        lock.sync_all()
            .map_err(|error| io("sync recovery lock", error))?;
        fsync_dir(&self.root.join("tmp/writers"))
    }

    /// Public directory ownership follows the Store owner, not whichever trusted writer
    /// first touches a hash prefix. A privileged broker may write beside the Store owner;
    /// its new directories must not strand that owner behind root-owned 0755 prefixes.
    /// Includes repository cache provenance, which the same trusted writers replace.
    /// Existing directories are checked, never silently repaired; payload modes/owners and
    /// operation-private roots are outside this policy.
    pub(crate) fn prepare_owned_directory(&self, path: &Path) -> Result<()> {
        let relative = path.strip_prefix(&self.root).map_err(|_| Refusal {
            code: Code::STORE_ERA,
            detail: "public directory is outside the Store".into(),
        })?;
        let mut parts = relative.components();
        let namespace = parts.next();
        if !(matches!(namespace, Some(std::path::Component::Normal(name))
            if name == "blobs" || name == "manifests" || name == "repos")
            || relative.starts_with("tmp/cache-roots"))
            || parts
                .clone()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return refuse(
                Code::STORE_ERA,
                "directory is outside public Store namespaces",
            );
        }
        let mut current = self.root.clone();
        let mut parent = open_real_directory(&current)?;
        let owner = parent
            .metadata()
            .map_err(|error| io("stat Store owner", error))?;
        for part in relative.components() {
            // The directory fd itself coordinates only mkdir/chown, never payload I/O.
            // This also prevents another writer from seeing a newly created directory
            // before its owner/mode are established. A crashed creator leaves a named,
            // explicitly refused owner mismatch rather than a blind repair permission.
            FileExt::lock_exclusive(&parent)
                .map_err(|error| io("lock Store directory admission", error))?;
            current.push(part.as_os_str());
            let created = match fs::create_dir(&current) {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
                Err(error) => return Err(io("create public Store directory", error)),
            };
            let child = open_real_directory(&current)?;
            let metadata = child
                .metadata()
                .map_err(|error| io("stat Store directory", error))?;
            if created {
                if metadata.uid() != owner.uid() || metadata.gid() != owner.gid() {
                    fchown(&child, Some(owner.uid()), Some(owner.gid()))
                        .map_err(|error| io("assign public directory to Store owner", error))?;
                }
                child
                    .set_permissions(fs::Permissions::from_mode(0o755))
                    .map_err(|error| io("set public Store directory mode", error))?;
                if self.flushes() {
                    child
                        .sync_all()
                        .map_err(|error| io("sync public Store directory", error))?;
                    parent
                        .sync_all()
                        .map_err(|error| io("sync public directory parent", error))?;
                }
            } else if metadata.uid() != owner.uid() || metadata.gid() != owner.gid() {
                return refuse(
                    Code::STORE_ERA,
                    format!(
                        "{} owner differs from Store owner; explicit directory repair required",
                        current.display()
                    ),
                );
            }
            FileExt::unlock(&parent)
                .map_err(|error| io("unlock Store directory admission", error))?;
            parent = child;
        }
        Ok(())
    }

    /// A newly-created repository metadata inode follows the trusted Store owner.
    /// Never called on existing files, immutable payloads, or operation-private state.
    pub(crate) fn assign_new_metadata_owner(&self, file: &File) -> Result<()> {
        let owner = fs::metadata(&self.root).map_err(|error| io("stat Store owner", error))?;
        let current = file
            .metadata()
            .map_err(|error| io("stat metadata owner", error))?;
        if (current.uid(), current.gid()) != (owner.uid(), owner.gid()) {
            fchown(file, Some(owner.uid()), Some(owner.gid()))
                .map_err(|error| io("assign metadata to Store owner", error))?;
        }
        Ok(())
    }

    /// Sorted Manifest ids whose CozyTensors runtime closures are verifiably resident.
    ///
    /// Ordinary snapshots, incomplete models, malformed Manifests, and corrupt/missing reached
    /// objects are conservatively absent from the answer.
    pub fn complete_cozytensors_manifests(&self) -> Result<Vec<String>> {
        let mut manifests = Vec::new();
        for first in read_dir(&self.root.join("manifests"))? {
            if !is_real_dir(&first)? {
                continue;
            }
            for second in read_dir(&first)? {
                if !is_real_dir(&second)? {
                    continue;
                }
                for path in read_dir(&second)? {
                    let metadata = match fs::symlink_metadata(&path) {
                        Ok(metadata) if metadata.is_file() => metadata,
                        Ok(_) => continue,
                        Err(error) => return Err(io("stat Manifest candidate", error)),
                    };
                    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                        continue;
                    };
                    let Some(digest) = name.strip_suffix(".json") else {
                        continue;
                    };
                    if hex64("Manifest candidate", digest).is_err()
                        || self.manifest_path(digest) != path
                    {
                        continue;
                    }
                    let reference = ObjectRef {
                        sha256: digest.to_string(),
                        length: metadata.len(),
                    };
                    let Ok(manifest) = self.read_manifest(&reference) else {
                        continue;
                    };
                    crate::stats::presence_pass();
                    let Ok(walk) = crate::checkpoint::walk_cozytensors(self, &manifest) else {
                        continue;
                    };
                    let complete = walk.distinct().into_iter().all(|object| {
                        self.open_verified(&object.sha256)
                            .is_ok_and(|file| file.len() == object.length)
                    });
                    if complete {
                        manifests.push(reference.id());
                    }
                }
            }
        }
        manifests.sort();
        Ok(manifests)
    }

    /// A malformed id has NO path in this store. `hex[0..2]` panics on a short string, and a
    /// panic is not a refusal — an empty argument from a shell pipeline reached this and
    /// aborted the process instead of producing `OBJECT_ABSENT`. The sentinel never exists
    /// and never embeds the caller's string, so `..` or a leading `/` cannot travel with it;
    /// the caller's own absent-object refusal then names the id it was handed.
    fn shard(&self, base: &str, hex: &str) -> PathBuf {
        if crate::ids::hex64("object id", hex).is_err() {
            return self.root.join(base).join(".malformed-id");
        }
        self.root
            .join(base)
            .join(&hex[0..2])
            .join(&hex[2..4])
            .join(hex)
    }
    pub fn object_path(&self, hex: &str) -> PathBuf {
        self.shard("blobs", hex)
    }
    pub fn blob_path(&self, hex: &str) -> PathBuf {
        self.object_path(hex)
    }
    pub fn manifest_path(&self, hex: &str) -> PathBuf {
        let mut path = self.shard("manifests", hex);
        path.set_extension("json");
        path
    }

    pub fn read_manifest(&self, reference: &ObjectRef) -> Result<crate::manifest::Manifest> {
        let path = self.manifest_path(&reference.sha256);
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW)
            .open(&path)
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    Refusal {
                        code: Code::OBJECT_ABSENT,
                        detail: format!("manifest {} is absent", reference.id()),
                    }
                } else {
                    io(format!("read {}", path.display()), error)
                }
            })?;
        let mut bytes = Vec::new();
        file.take(crate::manifest::Manifest::MAX_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| io("read bounded manifest", error))?;
        if bytes.len() > crate::manifest::Manifest::MAX_BYTES {
            return refuse(Code::SIZE_CAP, "manifest exceeds its document byte cap");
        }
        if crate::sha256::hex_digest(&bytes) != reference.sha256 {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                format!("manifest {} bytes disagree with its key", reference.id()),
            );
        }
        if bytes.len() as u64 != reference.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "manifest {} is {} bytes, expected {}",
                    reference.id(),
                    bytes.len(),
                    reference.length
                ),
            );
        }
        crate::manifest::Manifest::parse(&bytes)
    }

    pub fn put_manifest(&self, manifest: &crate::manifest::Manifest) -> Result<Put> {
        self.put_manifest_held(manifest, None)
    }

    pub fn put_manifest_held(
        &self,
        manifest: &crate::manifest::Manifest,
        operation: Option<&str>,
    ) -> Result<Put> {
        let catalog = crate::catalog::Catalog::open(&self.root)?;
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        let bytes = manifest.canonical_bytes();
        let object = ObjectRef::of(&bytes);
        if let Some(operation) = operation {
            catalog.hold(
                operation,
                &crate::storage::manifest_key(&object.sha256)?,
                object.length,
            )?;
        }
        let path = self.manifest_path(&object.sha256);
        self.prepare_owned_directory(path.parent().unwrap())?;
        if path.exists() {
            self.read_manifest(&object)?;
            return Ok(Put {
                obj: object,
                admitted: false,
            });
        }
        let temporary = path.parent().unwrap().join(format!(
            ".{}.{}.{}",
            path.file_name().unwrap().to_string_lossy(),
            std::process::id(),
            now_nanos_unique()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)
            .map_err(|error| io("create manifest temp", error))?;
        file.write_all(&bytes)
            .map_err(|error| io("write manifest", error))?;
        if self.flushes() {
            file.sync_all()
                .map_err(|error| io("fsync manifest", error))?;
        }
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o444))
            .map_err(|error| io("chmod manifest", error))?;
        let admitted = match fs::hard_link(&temporary, &path) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(error) => return cleanup(&temporary, io("install manifest", error)),
        };
        let _ = fs::remove_file(&temporary);
        if admitted && self.flushes() {
            fsync_dir(path.parent().unwrap())?;
        }
        if let (true, Some(epoch)) = (admitted, &self.epoch) {
            epoch.admitted(path.clone(), object.length)?;
        }
        self.read_manifest(&object)?;
        Ok(Put {
            obj: object,
            admitted,
        })
    }

    pub fn repository_path(&self, repo: &crate::repository::RepositoryName) -> PathBuf {
        self.root
            .join("repos")
            .join(&repo.org)
            .join(format!("{}.json", repo.name))
    }

    /// Local repository CAS. `observed` is exactly the body the caller read, or `None` for
    /// create. The durable rename/delete happens before the SQLite projection commit.
    pub fn apply_repository(
        &self,
        observed: Option<&[u8]>,
        mutation: &crate::repository::Mutation,
        fault: &Fault,
    ) -> Result<Option<crate::repository::Repository>> {
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        self.apply_repository_locked(observed, std::slice::from_ref(mutation), fault, false)
    }

    /// Record one verified pull's checkpoint and optional release lane atomically. The
    /// caller holds the native writer fence from FetchPlan completion through this call.
    /// Retained repository roots still require full snapshots, including ordinary siblings.
    pub(crate) fn apply_cached_repository(
        &self,
        observed: Option<&[u8]>,
        mutations: &[crate::repository::Mutation],
        fault: &Fault,
    ) -> Result<Option<crate::repository::Repository>> {
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        self.apply_repository_locked(observed, mutations, fault, true)
    }

    pub(crate) fn remove_cached_repository(
        &self,
        observed: &[u8],
        repo: &crate::repository::RepositoryName,
        _exclusive: &crate::catalog::RebuildGuard,
    ) -> Result<()> {
        if !crate::cache_roots::matches(self, repo, observed)? {
            return refuse(Code::REPOSITORY_CONFLICT, "cache root ownership changed");
        }
        self.apply_repository_locked(
            Some(observed),
            &[crate::repository::Mutation::DeleteRepository { repo: repo.clone() }],
            &Fault::default(),
            false,
        )?;
        Ok(())
    }

    fn validate_repository_checkpoint(&self, manifest: &ObjectRef) -> Result<()> {
        let parsed = self.read_manifest(manifest)?;
        for (_, entry) in parsed.entries() {
            let object = entry.blob();
            let file = self.open_verified(&object.sha256)?;
            if file.len() != object.length {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    format!("{} length differs from manifest", object.id()),
                );
            }
        }
        if let Some(header_ref) = parsed.header() {
            crate::checkpoint::load_header(self, header_ref)?;
            crate::checkpoint::walk(self, &parsed)?.require_resident(self)?;
        }
        Ok(())
    }

    fn apply_repository_locked(
        &self,
        observed: Option<&[u8]>,
        mutations: &[crate::repository::Mutation],
        fault: &Fault,
        cache: bool,
    ) -> Result<Option<crate::repository::Repository>> {
        let Some(first) = mutations.first() else {
            return refuse(
                Code::REPOSITORY_CONFLICT,
                "repository commit has no mutations",
            );
        };
        let repo = first.repo();
        if mutations.iter().any(|mutation| mutation.repo() != repo) {
            return refuse(
                Code::REPOSITORY_CONFLICT,
                "repository commit spans repositories",
            );
        }
        let catalog = crate::catalog::Catalog::open(&self.root)?;
        let path = self.repository_path(repo);
        let current = match fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(io(format!("read {}", path.display()), error)),
        };
        if !cache && current.as_deref() != observed {
            return refuse(
                Code::REPOSITORY_CONFLICT,
                "repository bytes changed since the caller observed them",
            );
        }
        // Validation may write verification records; finish it before BEGIN IMMEDIATE.
        let mut replacement = observed
            .map(crate::repository::Repository::parse)
            .transpose()?;
        for mutation in mutations {
            match mutation {
                crate::repository::Mutation::PutCheckpoint { manifest, .. }
                | crate::repository::Mutation::ReplaceLocal { manifest, .. } => {
                    self.validate_repository_checkpoint(manifest)?;
                }
                crate::repository::Mutation::UpdateRelease { set, .. } => {
                    for lane in set {
                        self.validate_repository_checkpoint(&lane.manifest)?;
                    }
                }
                crate::repository::Mutation::RemoveCheckpoint { .. }
                | crate::repository::Mutation::YankRelease { .. }
                | crate::repository::Mutation::DeleteRepository { .. } => {}
            }
            // A published checkpoint has additional immutable facts. Reuse its exact
            // manifest without trying to replace those facts with PutCheckpoint defaults.
            if cache
                && matches!(mutation, crate::repository::Mutation::PutCheckpoint { .. })
                && replacement.as_ref().is_some_and(|repository| {
                    cached_repository_satisfies(repository, std::slice::from_ref(mutation))
                })
            {
                continue;
            }
            let bytes = replacement.as_ref().map(Doc::canonical_bytes);
            replacement = crate::repository::apply(bytes.as_deref(), mutation)?;
        }
        if let Some(repository) = &replacement {
            for checkpoint in &repository.checkpoints {
                self.read_manifest(&checkpoint.manifest)?;
            }
        }
        let org = repo.org.clone();
        let name = repo.name.clone();
        let expected = observed;
        let path_for_write = path.clone();
        catalog.commit_repository(&org, &name, || {
            let actual = match fs::read(&path_for_write) {
                Ok(bytes) => Some(bytes),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => return Err(io("read repository under coordination", error)),
            };
            if actual.as_deref() != expected {
                if cache {
                    let latest = actual
                        .as_deref()
                        .map(crate::repository::Repository::parse)
                        .transpose()?;
                    replacement = merge_cached_repository(expected, latest, mutations)?;
                    if replacement.as_ref().map(Doc::canonical_bytes).as_deref()
                        == actual.as_deref()
                    {
                        return crate::storage::Census::open_metadata(&self.root)?
                            .projection_for(&org, &name);
                    }
                } else {
                    return refuse(
                        Code::REPOSITORY_CONFLICT,
                        "repository changed while acquiring SQLite coordination",
                    );
                }
            }
            // Unknown/preexisting ownership cannot be promoted by a later download.
            // Clear first: any crash before the matching replacement receipt preserves it.
            let cache_owned = cache
                && match actual.as_deref() {
                    None => true,
                    Some(bytes) => crate::cache_roots::matches(self, repo, bytes)?,
                };
            crate::cache_roots::clear(self, repo)?;
            fault.hit("repo-before-replace");
            match &replacement {
                Some(repository) => {
                    let bytes = repository.canonical_bytes();
                    if actual.as_deref() != Some(bytes.as_slice()) {
                        let directory = path_for_write.parent().unwrap();
                        self.prepare_owned_directory(directory)?;
                        let temporary = directory.join(format!(
                            ".{}.{}.{}",
                            path_for_write.file_name().unwrap().to_string_lossy(),
                            std::process::id(),
                            now_nanos_unique()
                        ));
                        let mut file = OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .mode(0o600)
                            .open(&temporary)
                            .map_err(|error| io("create repository temp", error))?;
                        if let Err(error) = file.write_all(&bytes) {
                            return cleanup(&temporary, io("write repository", error));
                        }
                        // Repository metadata is private to the trusted Store owner. A
                        // privileged Runtime may publish it before the unprivileged Host
                        // next reads/replaces it; retain 0600 but assign the NEW inode to
                        // the Store owner before it becomes visible at the repository key.
                        if let Err(error) = self.assign_new_metadata_owner(&file) {
                            return cleanup(&temporary, error);
                        }
                        if let Err(error) = file.sync_all() {
                            return cleanup(&temporary, io("fsync repository", error));
                        }
                        drop(file);
                        if let Err(error) = fs::rename(&temporary, &path_for_write) {
                            return cleanup(&temporary, io("rename repository", error));
                        }
                        fsync_dir(directory)?;
                    }
                }
                None => {
                    fs::remove_file(&path_for_write)
                        .map_err(|error| io("delete repository", error))?;
                    fsync_dir(path_for_write.parent().unwrap())?;
                }
            }
            fault.hit("repo-after-replace");
            if cache_owned {
                if let Some(repository) = &replacement {
                    crate::cache_roots::record(self, repo, &repository.canonical_bytes())?;
                }
            }
            crate::storage::Census::open_metadata(&self.root)?.projection_for(&org, &name)
        })?;
        Ok(replacement)
    }
    /// PRESENCE HINT ONLY. True says an object file exists at that id; it says nothing
    /// about its bytes. Resume must consult `record` or rehash.
    pub fn contains(&self, hex: &str) -> bool {
        self.object_path(hex).is_file()
    }

    // ------------------------------------------------------------ admission

    /// THE STREAMING-AND-VERIFY CORE, shared by every destination bytes can land in.
    ///
    /// Open a locked temp under `tmp/`, hash every chunk as it is written, fsync it, and
    /// refuse anything that is not exactly the `ObjectRef` the caller expected. What comes
    /// back is a durable, still-locked, still-0600 temp whose bytes are PROVEN — and
    /// nothing about where it will end up. Destinations differ only in where they link
    /// that temp and what they record about it (`put_stream_held` into `blobs/` with a
    /// verification record; [`crate::staging`] into `staging/` with none).
    ///
    /// **The narrow door must not be the weak one.** A second hashing loop for carriers
    /// would be a second opinion about identity, and the two would disagree exactly once,
    /// silently, on the artifact that mattered. There is one loop and this is it.
    ///
    /// `prefix` names the temp so [`Store::reap`] can recognise it as the store's own
    /// scratch; anything else under `tmp/` belongs to whoever put it there.
    pub(crate) fn stream_verified<R: Read>(
        &self,
        r: &mut R,
        expect: Option<&ObjectRef>,
        fault: &Fault,
        prefix: &str,
    ) -> Result<Verified> {
        // The temp is LOCKED for as long as this process needs it — through the write,
        // the fsync and the link. `reap` takes the same lock to decide: a temp whose lock
        // the kernel still holds has a live writer behind it, whatever its mtime says, and
        // a temp whose lock it hands out is an orphan. The descriptor stays open until the
        // link has landed; a writer that dies at any stage releases the lock with it.
        let (tmp, mut f) = loop {
            let tmp = self.root.join("tmp").join(format!(
                "{prefix}{}-{}",
                std::process::id(),
                now_nanos_unique()
            ));
            let f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&tmp)
                .map_err(|e| io(format!("create temp {}", tmp.display()), e))?;
            match f.try_lock_exclusive() {
                Ok(()) => break (tmp, f),
                // Only a reaper can hold the lock on a name this process just minted: it
                // saw the temp before the lock landed, and it unlinks what it locks. Mint
                // another.
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                Err(e) => return cleanup(&tmp, io("lock temp", e)),
            }
        };
        fault.hit("open");

        let mut h = Sha256::new();
        let mut buf = vec![0u8; BUF];
        let mut len: u64 = 0;
        let mut first = true;
        let discard = |len: u64, refusal: Refusal| {
            if let Some(sink) = &self.progress {
                sink.discarded(len);
            }
            cleanup(&tmp, refusal)
        };
        loop {
            let n = match r.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => return discard(len, io("read source", e)),
            };
            h.update(&buf[..n]);
            if let Err(e) = f.write_all(&buf[..n]) {
                return discard(len, io("write temp", e));
            }
            len += n as u64;
            self.moved(n as u64);
            if first {
                first = false;
                fault.hit("partial");
            }
        }
        fault.hit("written");
        if self.flushes() {
            if let Err(e) = f.sync_all() {
                return discard(len, io("fsync temp", e));
            }
        }
        fault.hit("fsync");

        let hex = sha256::hex(&h.finish());
        if let Some(want) = expect {
            if want.length != len {
                return discard(
                    len,
                    Refusal {
                        code: Code::LENGTH_MISMATCH,
                        detail: format!("expected {} bytes, streamed {len}", want.length),
                    },
                );
            }
            if want.sha256 != hex {
                return discard(
                    len,
                    Refusal {
                        code: Code::OBJECT_ID_MISMATCH,
                        detail: format!("expected sha256:{}, streamed sha256:{hex}", want.sha256),
                    },
                );
            }
        }
        Ok(Verified {
            tmp,
            file: f,
            sha256: hex,
            length: len,
        })
    }

    pub fn put_stream<R: Read>(
        &self,
        r: &mut R,
        expect: Option<&ObjectRef>,
        fault: &Fault,
    ) -> Result<Put> {
        self.put_stream_held(r, expect, fault, None)
    }

    pub fn put_stream_held<R: Read>(
        &self,
        r: &mut R,
        expect: Option<&ObjectRef>,
        fault: &Fault,
        operation: Option<&str>,
    ) -> Result<Put> {
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        let verified = self.stream_verified(r, expect, fault, PUT_TEMP)?;
        self.admit_verified(verified, fault, operation)
    }

    /// Common no-clobber admission after streaming or durable-prefix digest verification.
    ///
    /// An unheld admission (every pull) publishes on the caller's thread, in parallel with
    /// every other, and only its catalog record is serialized: records arriving together
    /// share one transaction and one commit (`Catalog::record_grouped`). Each object is
    /// still published before its record is durable, and reported only once that commit
    /// is. A held admission writes its hold first, so a refused hold publishes nothing.
    pub(crate) fn admit_verified(
        &self,
        verified: Verified,
        fault: &Fault,
        operation: Option<&str>,
    ) -> Result<Put> {
        let catalog = match crate::catalog::Catalog::open(&self.root) {
            Ok(catalog) => catalog,
            Err(refusal) => return cleanup(&verified.tmp, refusal),
        };
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        let object = ObjectRef {
            sha256: verified.sha256.clone(),
            length: verified.length,
        };
        let publish = move || {
            let installed = self.link_verified(&verified, fault)?;
            if let Some(dir) = installed.as_ref().filter(|_| self.flushes()) {
                if let Err(error) = fsync_dir(dir) {
                    let _ = fs::remove_file(&verified.tmp);
                    return Err(error);
                }
            }
            let length = verified.length;
            let record = self.finish_linked(verified, installed.is_some())?;
            if let (Some(epoch), Some(record)) = (&self.epoch, &record) {
                epoch.admitted(self.object_path(&record.sha256), length)?;
            }
            Ok(record)
        };
        let admitted = match operation {
            Some(operation) => catalog.admit_blob(Some(operation), &object, publish)?,
            None => {
                let record = publish()?;
                let admitted = record.is_some();
                catalog.record_grouped(crate::catalog::Recording { record })?;
                admitted
            }
        };
        if admitted {
            if let Some(sink) = &self.progress {
                sink.admitted();
            }
        }
        fault.hit("record");
        Ok(Put {
            obj: object,
            admitted,
        })
    }

    /// Publish one verified temp at its digest, no-clobber. `Some(dir)` when this call
    /// installed it and `dir` must be fsynced before its record; `None` when another writer's
    /// inode already stands there. On refusal the temp is gone.
    pub(crate) fn link_verified(
        &self,
        verified: &Verified,
        fault: &Fault,
    ) -> Result<Option<PathBuf>> {
        let tmp = &verified.tmp;
        if let Err(e) = fs::set_permissions(tmp, fs::Permissions::from_mode(0o444)) {
            return cleanup(tmp, io("chmod 0444", e));
        }
        fault.hit("chmod");
        let dst = self.object_path(&verified.sha256);
        let dir = dst.parent().unwrap().to_path_buf();
        if let Err(error) = self.prepare_owned_directory(&dir) {
            return cleanup(tmp, error);
        }
        // No-clobber admission: link() refuses to replace an installed object.
        let admitted = match fs::hard_link(tmp, &dst) {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(e) => return cleanup(tmp, io(format!("link {}", dst.display()), e)),
        };
        fault.hit("linked");
        Ok(admitted.then_some(dir))
    }

    /// Drop the temp name and descriptor of a linked object; the record for one this call
    /// installed. Losing the no-clobber race does not verify another writer's inode.
    pub(crate) fn finish_linked(
        &self,
        verified: Verified,
        admitted: bool,
    ) -> Result<Option<VRecord>> {
        let _ = fs::remove_file(&verified.tmp);
        drop(verified.file);
        if !admitted {
            return Ok(None);
        }
        Ok(Some(VRecord::of(
            &verified.sha256,
            &stat(&self.object_path(&verified.sha256))?,
        )))
    }

    pub fn put_file(&self, src: &Path, expect: Option<&ObjectRef>, fault: &Fault) -> Result<Put> {
        let mut f = File::open(src).map_err(|e| io(format!("open {}", src.display()), e))?;
        self.put_stream(&mut f, expect, fault)
    }

    /// Copy one exact object between stores through the same verified read and admission
    /// doors used everywhere else. Store layout never escapes this module: the source is
    /// opened once through `open_verified`, and the destination hashes, length-checks and
    /// publishes without clobbering through `put_stream`.
    pub fn transfer_object(&self, destination: &Store, want: &ObjectRef) -> Result<Put> {
        // A retried transfer streams NOTHING: the destination's own predicate
        // (`record_valid` — may these bytes be trusted without hashing them again?, the
        // same question `FetchPlan::of` asks) answers before any byte moves, so a retried
        // 99 GB store-to-store materialization is a stat, not a re-copy.
        if let Ok(rec) = destination.record_valid(&want.sha256) {
            if rec.length == want.length {
                return Ok(Put {
                    obj: want.clone(),
                    admitted: false,
                });
            }
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{} is {} bytes in the destination store, expected {}",
                    want.id(),
                    rec.length,
                    want.length
                ),
            );
        }
        let mut source = self.open_verified(&want.sha256)?;
        if source.len() != want.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{} is {} bytes in the source store, expected {}",
                    want.id(),
                    source.len(),
                    want.length
                ),
            );
        }
        let put = destination.put_stream(&mut source, Some(want), &Fault::default())?;
        if !put.admitted {
            let installed = destination.open_verified(&want.sha256)?;
            if installed.len() != want.length {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    format!(
                        "{} is {} bytes in the destination store, expected {}",
                        want.id(),
                        installed.len(),
                        want.length
                    ),
                );
            }
        }
        Ok(put)
    }

    pub fn transfer_manifest(&self, destination: &Store, want: &ObjectRef) -> Result<Put> {
        let manifest = self.read_manifest(want)?;
        let put = destination.put_manifest(&manifest)?;
        if put.obj != *want {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                "transferred manifest identity changed during admission",
            );
        }
        Ok(put)
    }

    // ------------------------------------------------------------ records

    fn write_record(&self, rec: &VRecord) -> Result<()> {
        crate::catalog::Catalog::open(&self.root)?.write_verification(rec)
    }

    fn record(&self, hex: &str) -> Option<VRecord> {
        self.trust.get(hex)
    }

    fn drop_record(&self, hex: &str) {
        if let Ok(catalog) = crate::catalog::Catalog::open(&self.root) {
            catalog.drop_verification(hex);
        }
    }

    /// The one question the read and resume paths ask: may these bytes be trusted without
    /// hashing them again?
    pub fn record_valid(&self, hex: &str) -> std::result::Result<VRecord, String> {
        let md = match stat(&self.object_path(hex)) {
            Ok(md) => md,
            Err(e) => return Err(e.detail),
        };
        let rec = self
            .record(hex)
            .ok_or_else(|| "no verification record".to_string())?;
        match rec.disagreement(hex, &md) {
            Some(why) => Err(why),
            None => Ok(rec),
        }
    }

    // ------------------------------------------------------------ reads

    /// Open for reading without following a symlink at the final component, and refuse
    /// anything that is not a regular file.
    pub fn open_nofollow(&self, hex: &str) -> Result<File> {
        let p = self.object_path(hex);
        let f = OpenOptions::new()
            .read(true)
            .custom_flags(O_NOFOLLOW)
            .open(&p)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Refusal {
                        code: Code::OBJECT_ABSENT,
                        detail: format!("sha256:{hex} not in {}", self.root.display()),
                    }
                } else {
                    io(format!("open {}", p.display()), e)
                }
            })?;
        let md = f.metadata().map_err(|e| io("stat object", e))?;
        if !md.is_file() {
            return refuse(
                Code::NOT_REGULAR_FILE,
                format!("{} is not a regular file", p.display()),
            );
        }
        Ok(f)
    }

    /// A verified handle: trusted from a valid record, otherwise rehashed. Bytes that
    /// disagree with their id are removed, never returned.
    ///
    /// THE FILE THAT WAS VERIFIED IS THE FILE THAT IS READ. v1 verified a PATHNAME and then
    /// re-opened that pathname to read it — two resolutions of one name, with a window
    /// between them in which a same-length replacement passes every check the record binds
    /// (size, and an mtime the attacker controls). The handle is opened ONCE here; the
    /// fstat, the rehash and every subsequent read go through that one descriptor, which on
    /// Linux pins the inode. A rename over the path after this point does not reach these
    /// bytes at all — the swap is not caught, it is INEFFECTIVE.
    pub fn open_verified(&self, hex: &str) -> Result<VerifiedFile> {
        Ok(self.open_verified_reporting(hex)?.0)
    }

    /// `open_verified` plus the fact a caller may need to report: whether trusting these
    /// bytes cost a rehash (missing or disagreeing record) rather than a record lookup.
    pub fn open_verified_reporting(&self, hex: &str) -> Result<(VerifiedFile, bool)> {
        let (verdict, file) = self.verify_pinned(hex, false)?;
        let rehashed = !matches!(verdict, Verdict::Verified { rehashed: false });
        match (verdict, file) {
            (Verdict::CorruptRemoved { why }, _) => refuse(
                Code::OBJECT_CORRUPT,
                format!("sha256:{hex} was corrupt and removed: {why}"),
            ),
            (_, Some(mut f)) => {
                let len = f.metadata().map_err(|e| io("stat object", e))?.len();
                f.seek(SeekFrom::Start(0)).map_err(|e| io("rewind", e))?;
                Ok((
                    VerifiedFile {
                        file: f,
                        sha256: hex.to_string(),
                        len,
                    },
                    rehashed,
                ))
            }
            (v, None) => refuse(
                Code::OBJECT_CORRUPT,
                format!("sha256:{hex}: {v:?} left no verified descriptor"),
            ),
        }
    }

    /// Bounded range read over verified bytes. Out-of-bounds refuses instead of clamping.
    pub fn read_range(&self, hex: &str, off: u64, len: u64) -> Result<Vec<u8>> {
        let mut vf = self.open_verified(hex)?;
        let size = vf.len;
        let end = off.checked_add(len).ok_or(Refusal {
            code: Code::RANGE_BOUNDS,
            detail: format!("offset {off} + length {len} overflows"),
        })?;
        if end > size {
            return refuse(
                Code::RANGE_BOUNDS,
                format!("range [{off}, {end}) leaves object of {size} bytes"),
            );
        }
        vf.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| io("seek", e))?;
        let mut out = vec![0u8; len as usize];
        vf.file
            .read_exact(&mut out)
            .map_err(|e| io("read range", e))?;
        Ok(out)
    }

    /// Stream a verified object out. The read path does not hash; the pinned verification
    /// decided already, and it decided about THIS descriptor.
    pub fn read_into<W: Write>(&self, hex: &str, w: &mut W) -> Result<u64> {
        let mut vf = self.open_verified(hex)?;
        let mut buf = vec![0u8; BUF];
        let mut n = 0u64;
        loop {
            let k = vf.file.read(&mut buf).map_err(|e| io("read object", e))?;
            if k == 0 {
                return Ok(n);
            }
            w.write_all(&buf[..k]).map_err(|e| io("write out", e))?;
            n += k as u64;
        }
    }

    // ------------------------------------------------------------ verify / corrupt removal

    /// A valid verification record IS the answer; a rehash happens exactly when the record
    /// is missing or any bound fact moved. That rehash is CORRECTNESS, not paranoia, and it
    /// is not optional.
    ///
    /// PARANOID MODE IS DELETED (owner ruling 2026-08-25). The per-read full-rehash option
    /// was 16.6x slower, had no consumer, and gave integrity checking a second home.
    /// `scrub` — the rate-knobbed whole-store sweep — is now the ONLY deliberate
    /// full-rehash door, and it reaches the forced path through `rehash` below rather than
    /// through a flag anyone can pass on a read.
    pub fn verify(&self, hex: &str) -> Result<Verdict> {
        Ok(self.verify_pinned(hex, false)?.0)
    }

    /// The scrub's door: rehash this object whatever its record says. Not reachable from
    /// the read path, the CLI, or the wheel — only from `scrub`.
    fn rehash(&self, hex: &str) -> Result<Verdict> {
        Ok(self.verify_pinned(hex, true)?.0)
    }

    /// Verify through ONE descriptor and hand it back with the verdict.
    ///
    /// Every fact this consults comes from the open fd — `fstat` for the record binding,
    /// the fd itself for the rehash — so the pathname is resolved exactly once. The
    /// returned `File` is the object those facts are about; `None` only after corrupt removal,
    /// where there is nothing to hand out.
    fn verify_pinned(&self, hex: &str, force: bool) -> Result<(Verdict, Option<File>)> {
        let mut f = self.open_nofollow(hex)?;
        let md = f.metadata().map_err(|e| io("fstat object", e))?;
        let disagreement = match self.record(hex) {
            None => Some("no verification record".to_string()),
            Some(rec) => rec.disagreement(hex, &md),
        };
        if disagreement.is_none() && !force {
            return Ok((Verdict::Verified { rehashed: false }, Some(f)));
        }
        let got = self.hash_fd(&mut f)?;
        if got != hex {
            let why = disagreement.unwrap_or_else(|| "scrub rehash".to_string());
            // Remove the NAME: whatever is at the path now must stop satisfying reads. The
            // descriptor is dropped with `f` — corrupt bytes are never handed back.
            self.remove_corrupt_blob(hex)?;
            return Ok((
                Verdict::CorruptRemoved {
                    why: format!("{why}; rehash gave sha256:{got}"),
                },
                None,
            ));
        }
        let md = f.metadata().map_err(|e| io("fstat object", e))?;
        self.write_record(&VRecord::of(hex, &md))?;
        Ok((
            match disagreement {
                None => Verdict::Verified { rehashed: true },
                Some(why) => Verdict::Invalidated { why },
            },
            Some(f),
        ))
    }

    pub fn hash_object(&self, hex: &str) -> Result<String> {
        self.hash_fd(&mut self.open_nofollow(hex)?)
    }

    /// Remove the corrupt authoritative name. A concurrent remover that got there first has
    /// already established the same safe state.
    pub fn remove_corrupt_blob(&self, hex: &str) -> Result<()> {
        let path = self.object_path(hex);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.drop_record(hex);
                return Ok(());
            }
            Err(e) => return Err(io("remove corrupt blob", e)),
        }
        self.drop_record(hex);
        fsync_dir(path.parent().unwrap())?;
        Ok(())
    }

    pub fn remove_corrupt_manifest(&self, hex: &str) -> Result<()> {
        crate::ids::hex64("manifest", hex)?;
        let _writer = crate::catalog::WriterGuard::acquire(&self.root)?;
        let path = self.manifest_path(hex);
        match fs::remove_file(&path) {
            Ok(()) => fsync_dir(path.parent().unwrap())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io("remove corrupt manifest", error)),
        }
        Ok(())
    }

    // ------------------------------------------------------------ scrub / reap

    pub fn objects(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        let base = self.root.join("blobs");
        for a in read_dir(&base)? {
            for b in read_dir(&a)? {
                for o in read_dir(&b)? {
                    if let Some(n) = o.file_name().and_then(|s| s.to_str()) {
                        if hex64("object", n).is_ok() {
                            out.push(n.to_string());
                        }
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// ONE knob, and since the 2026-08-25 owner ruling the ONE deliberate full-rehash door.
    /// `rate` is the fraction of objects rehashed this pass, chosen deterministically from
    /// (pass, id). Everything else gets its record checked, and a record that disagrees is
    /// rehashed anyway.
    pub fn scrub(&self, rate: f64, pass: u64) -> Result<Scrub> {
        let mut s = Scrub::default();
        for hex in self.objects()? {
            s.scanned += 1;
            let sampled = sample(pass, &hex) < rate;
            if sampled {
                s.sampled += 1;
            }
            let size = stat(&self.object_path(&hex)).map(|m| m.len()).unwrap_or(0);
            match if sampled {
                self.rehash(&hex)?
            } else {
                self.verify(&hex)?
            } {
                Verdict::Verified { rehashed } => {
                    if rehashed {
                        s.rehashed += 1;
                        s.bytes_hashed += size;
                    }
                }
                Verdict::Invalidated { .. } => {
                    s.rehashed += 1;
                    s.bytes_hashed += size;
                }
                Verdict::CorruptRemoved { .. } => {
                    s.rehashed += 1;
                    s.bytes_hashed += size;
                    s.removed += 1;
                }
            }
        }
        Ok(s)
    }

    /// Reap the admission temps whose writer is GONE. A temp is scratch by construction:
    /// nothing references it, so removing an orphan can never lose an object — and a temp
    /// with a writer still behind it is one `link()` from being an object, so removing it
    /// would fail that writer at the last step. Liveness is the lock `put_stream_held`
    /// holds on the temp for its whole life: the kernel releases it only when the
    /// descriptor closes, which a dead process does and a slow one does not. Age plays no
    /// part — a writer streaming a large object is older than any clock a reaper could
    /// pick. Only the store's own temps are considered — [`PUT_TEMP`] for an admission and
    /// [`STAGE_TEMP`] for a carrier; anything else under `tmp/` belongs to whoever put it
    /// there.
    ///
    /// A carrier's temp is collected here for exactly the reason an admission's is, and it
    /// is the larger leak: a crashed source fetch would otherwise strand a partial 5 GB
    /// shard under `tmp/` forever, with no name anywhere that would ever mention it.
    pub fn reap(&self) -> Result<(usize, usize)> {
        let (mut gone, mut kept) = (0, 0);
        for p in read_dir(&self.root.join("tmp"))? {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let ours = name.starts_with(PUT_TEMP) || name.starts_with(STAGE_TEMP);
            if !ours || !fs::symlink_metadata(&p).is_ok_and(|m| m.is_file()) {
                continue;
            }
            let f = match File::options().read(true).open(&p) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(io(format!("open temp {}", p.display()), e)),
            };
            if f.try_lock_exclusive().is_err() {
                kept += 1;
                continue;
            }
            match fs::remove_file(&p) {
                Ok(()) => gone += 1,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(io(format!("reap temp {}", p.display()), e)),
            }
        }
        Ok((gone, kept))
    }
}

fn validate_store_root(root: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(root)
        .map_err(|error| io(format!("stat Store root {}", root.display()), error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return refuse(
            Code::STORE_ERA,
            format!("{} is not a real Store directory", root.display()),
        );
    }
    for name in ["repos", "manifests", "blobs"] {
        let path = root.join(name);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(_) => {
                return refuse(
                    Code::STORE_ERA,
                    format!(
                        "{} lacks a real {name}/ namespace: not a TensorFS Store",
                        root.display()
                    ),
                )
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return refuse(
                    Code::STORE_ERA,
                    format!(
                        "{} lacks a real {name}/ namespace: not a TensorFS Store",
                        root.display()
                    ),
                )
            }
            Err(error) => {
                return Err(io(
                    format!("stat Store namespace {}", path.display()),
                    error,
                ))
            }
        }
    }
    Ok(())
}

/// Where a Store records the cache it is bound to: one absolute path, one line, LOCAL.
///
/// **The binding is a property of the Store because the cache is a property of the
/// deployment, not of any request.** The alternative -- a path threaded through every
/// caller -- makes a mount point part of a protocol, of a Python signature, and of a spawn
/// class's environment allowlist, all so that a layer with no other use for the fact can
/// hand it back to the layer that already had it. Recording it once at setup means
/// `prepare_model_source`, `fetch` and a restore all find the same cache without anyone
/// passing one, and the pod's Runtime child never has to learn where the mount is.
///
/// **It lives on local disk beside the Store and never in the catalog.** `tensorfs.sqlite`
/// is a rebuildable projection of repository/manifest/blob files (`Catalog::open` says as
/// much, and refuses a Store whose catalog is absent as "recovery-only until `tfs store
/// rebuild`"). A cache binding is not derivable from any of those files, so a rebuild would
/// drop it silently and mirroring would stop with nothing to see. It is also not on the
/// cache itself, which under #612 holds immutable Manifest/blob paths and nothing else.
///
/// **Reading it can only cost, never refuse.** An absent, unreadable or malformed file is
/// no binding at all -- not an error -- because a Store that will not open is an
/// availability failure and cache state is forbidden to cause one. A path that is not a
/// path refuses at BIND time, where it is a caller's mistake rather than weather.
pub const REPO_CACHE_BINDING: &str = "repo-cache";

/// The longest binding this reads. `PATH_MAX` with room for the newline; a longer file is
/// not a mount point.
const REPO_CACHE_BINDING_MAX: u64 = 4097;

fn validate_repo_cache_root(cache: &Path, store: &Path) -> Result<()> {
    if !cache.is_absolute() {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("repo cache root {} is not absolute", cache.display()),
        );
    }
    if cache.as_os_str().as_encoded_bytes().len() as u64 >= REPO_CACHE_BINDING_MAX {
        return refuse(
            Code::SIZE_CAP,
            "repo cache root is longer than a filesystem path",
        );
    }
    if cache == store || cache.starts_with(store) || store.starts_with(cache) {
        return refuse(
            Code::STORE_ERA,
            format!(
                "repo cache root {} overlaps the Store at {}; the cache is never the Store",
                cache.display(),
                store.display()
            ),
        );
    }
    Ok(())
}

/// Where a Store records the disk budget it is held to: one decimal byte count, one line,
/// LOCAL, beside the cache binding and for the same reasons (see [`REPO_CACHE_BINDING`]).
///
/// **The presence of this record is the whole opt-in.** The owner's requirement was that
/// reclamation must not be default-on -- *"users would be confused if files start
/// disappearing"* -- and his own sentence for turning it on was *"tensorfs should be given
/// some sort of disc budget, and when that's exceeded it starts ejecting"*. So the budget
/// IS the switch: one record rather than a number plus a flag nobody sets. A Store without
/// one behaves byte-for-byte as it did before tfs-064, and `$HOME/.tensorfs` never gets one.
///
/// **The budget is measured against the Store's OWN occupancy, never against the
/// filesystem.** TensorFS does not stat a filesystem for policy (decisions.md 241/941):
/// free space is a caller input, because only the caller can see the image layers, the
/// install root, the media quota and the compile caches sharing the disk. What TensorFS
/// owns is what it is holding, and that it can count without a syscall about anyone else.
pub const DISK_BUDGET: &str = "disk-budget";

/// The longest budget record this reads. A byte count needs 20 digits and a newline; a
/// longer file is not a number.
const DISK_BUDGET_MAX: u64 = 32;

/// Read the budget, or nothing. Like the cache binding, reading can only cost, never
/// refuse: an absent, unreadable or malformed record is no budget at all, because a Store
/// that will not open is an availability failure and a policy record may not cause one. A
/// number that is not a number refuses at BIND time, where it is a caller's mistake.
fn read_disk_budget(root: &Path) -> Option<u64> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(root.join(DISK_BUDGET))
        .ok()?;
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(DISK_BUDGET_MAX).read_to_end(&mut bytes).ok()?;
    std::str::from_utf8(&bytes).ok()?.trim().parse::<u64>().ok()
}

/// Read the binding, or nothing. Weather, never authority -- see [`REPO_CACHE_BINDING`].
fn read_repo_cache_binding(root: &Path) -> Option<crate::repo_cache::RepoObjectCache> {
    let path = root.join(REPO_CACHE_BINDING);
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(&path)
        .ok()?;
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(REPO_CACHE_BINDING_MAX)
        .read_to_end(&mut bytes)
        .ok()?;
    while bytes.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
        bytes.pop();
    }
    if bytes.is_empty() || bytes.contains(&0) {
        return None;
    }
    let cache = PathBuf::from(std::ffi::OsString::from_vec(bytes));
    validate_repo_cache_root(&cache, root).ok()?;
    Some(crate::repo_cache::RepoObjectCache::new(cache))
}

/// Serializes Store creation on one root across threads and processes. A lock on the root
/// directory itself leaves nothing behind in it.
fn creation_lock(root: &Path) -> Result<File> {
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(root)
        .map_err(|error| io(format!("open Store root {}", root.display()), error))?;
    directory
        .lock_exclusive()
        .map_err(|error| io(format!("lock Store root {}", root.display()), error))?;
    Ok(directory)
}

/// No handle is returned while an unclean shutdown's unsynced admissions are unverified.
fn open_validated(root: &Path) -> Result<Store> {
    fs::create_dir_all(root.join("tmp")).map_err(|error| io("mkdir tmp", error))?;
    let store = Store {
        root: root.to_path_buf(),
        progress: None,
        cache: read_repo_cache_binding(root),
        budget: read_disk_budget(root),
        trust: crate::catalog::Trust::of(root),
        disk: crate::disk::filesystem(root),
        epoch: None,
    };
    crate::unsynced::recover(&store)?;
    Ok(store)
}

fn open_real_directory(path: &Path) -> Result<File> {
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)
        .map_err(|error| io(format!("open directory {}", path.display()), error))?;
    if !directory
        .metadata()
        .map_err(|error| io("stat directory", error))?
        .is_dir()
    {
        return refuse(
            Code::STORE_ERA,
            format!("{} is not a real Store directory", path.display()),
        );
    }
    Ok(directory)
}

fn prepare_reader_dir(path: &Path, mode: u32) -> Result<()> {
    match fs::create_dir(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(io(format!("mkdir {}", path.display()), error)),
    }
    let directory = open_real_directory(path)?;
    directory
        .set_permissions(fs::Permissions::from_mode(mode))
        .map_err(|error| io(format!("chmod directory {}", path.display()), error))?;
    directory
        .sync_all()
        .map_err(|error| io(format!("sync directory {}", path.display()), error))
}

fn is_real_dir(path: &Path) -> Result<bool> {
    fs::symlink_metadata(path)
        .map(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
        .map_err(|error| io(format!("stat directory {}", path.display()), error))
}

// ---------------------------------------------------------------- helpers

fn cleanup<T>(tmp: &Path, e: Refusal) -> Result<T> {
    let _ = fs::set_permissions(tmp, fs::Permissions::from_mode(0o600));
    let _ = fs::remove_file(tmp);
    Err(e)
}

impl Store {
    /// Hash an OPEN object from the top. Takes the descriptor, never the name — the caller
    /// has already resolved the path and everything after it is about this inode. A
    /// rehash is bytes moved like any other, and is announced as such.
    fn hash_fd(&self, f: &mut File) -> Result<String> {
        f.seek(SeekFrom::Start(0)).map_err(|e| io("rewind", e))?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; BUF];
        loop {
            let n = f.read(&mut buf).map_err(|e| io("read object", e))?;
            if n == 0 {
                return Ok(sha256::hex(&h.finish()));
            }
            h.update(&buf[..n]);
            self.moved(n as u64);
        }
    }
}

fn stat(p: &Path) -> Result<fs::Metadata> {
    let md = fs::symlink_metadata(p).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Refusal {
                code: Code::OBJECT_ABSENT,
                detail: format!("{} absent", p.display()),
            }
        } else {
            io(format!("stat {}", p.display()), e)
        }
    })?;
    if !md.is_file() {
        return refuse(
            Code::NOT_REGULAR_FILE,
            format!("{} is not a regular file", p.display()),
        );
    }
    Ok(md)
}

fn read_dir(p: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let rd = match fs::read_dir(p) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(io(format!("read_dir {}", p.display()), e)),
    };
    for e in rd {
        out.push(e.map_err(|e| io("dir entry", e))?.path());
    }
    out.sort();
    Ok(out)
}

pub(crate) fn fsync_dir(dir: &Path) -> Result<()> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| io(format!("fsync dir {}", dir.display()), e))
}

/// Make an entry just renamed into `dir` durable. The ancestors up to the store `root` hold
/// only directory entries, which change when a directory is created, so each process syncs
/// them once per `dir` (by inode: a recreated directory syncs again) rather than on every
/// write. A child call's input hold wrote one root and paid four fsyncs; it now pays two.
pub(crate) fn fsync_entry_dir(root: &Path, dir: &Path) -> Result<()> {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SYNCED: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
    fsync_dir(dir)?;
    let meta = fs::metadata(dir).map_err(|e| io(format!("stat {}", dir.display()), e))?;
    let identity = (meta.dev(), meta.ino());
    let synced = SYNCED.get_or_init(Default::default);
    if synced.lock().is_ok_and(|set| set.contains(&identity)) {
        return Ok(());
    }
    for ancestor in dir.ancestors().skip(1) {
        fsync_dir(ancestor)?;
        if ancestor == root {
            break;
        }
    }
    if let Ok(mut set) = synced.lock() {
        set.insert(identity);
    }
    Ok(())
}

pub(crate) fn now_nanos_unique() -> u128 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
        .wrapping_add(N.fetch_add(1, Ordering::Relaxed) as u128)
}

/// Deterministic [0,1) sample position for (pass, object) — the one scrub knob's input.
fn sample(pass: u64, hex: &str) -> f64 {
    let d = sha256::digest(format!("{pass}:{hex}").as_bytes());
    let v = u32::from_be_bytes([d[0], d[1], d[2], d[3]]);
    v as f64 / u32::MAX as f64
}
