//! A pull on a persistent disk admits objects without fsync and makes them durable together.
//!
//! Before its first admission the pull writes a durable MARKER, `tmp/unsynced/<boot>.<pid>.<n>`,
//! holding the time it began and locked for as long as the pull lives; every object it admits
//! is written after that time. Every [`SYNC_EVERY`] bytes and at the end of the walk it fsyncs
//! the objects admitted since the last batch, in parallel — its own objects only, because a
//! `syncfs` also waits for every other writer on a shared filesystem. Once the walk is over and
//! its last batch is synced, nothing written under the marker is unsynced, and it is removed.
//!
//! A marker left by an earlier BOOT means an unclean shutdown may have lost those objects'
//! pages while their names and verification records survived. [`recover`], which runs
//! before any Store handle is returned, rehashes every object whose mtime is not older than
//! the marker and removes the torn ones; the next plan fetches them again. A marker left by
//! a dead process of THIS boot costs no rehash: its pages are still in the page cache. The
//! next pull to finish fsyncs the objects not older than it, and retires it.
//!
//! A lost NAME needs no marker: an object that is absent is fetched again, whatever its
//! record says. So a batch syncs files, not the directories they were linked into.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fs2::FileExt;

use crate::disk::{self, DiskClass};
use crate::err::{Refusal, Result};
use crate::store::{fsync_dir, Store};

const DIR: &str = "tmp/unsynced";
const SUFFIX: &str = ".marker";

/// Bytes a pull admits between batch syncs: the most it leaves unsynced.
pub const SYNC_EVERY: u64 = 8 << 30;

/// Files one batch fsyncs at once. Each waits on the device, not a CPU.
const SYNC_THREADS: usize = 16;

/// Kernel timestamps are coarse and a clock may be stepped; an object's mtime is compared
/// with a marker's time only after this much is subtracted from the marker's.
const SLACK_NANOS: i64 = 60_000_000_000;

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    crate::store::classify_io(what, error)
}

fn now_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

/// What making one pull durable cost, and why it cost that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyncCost {
    pub class: DiskClass,
    pub filesystem: &'static str,
    /// Batch syncs the pull ran: one per [`SYNC_EVERY`] bytes and one at its end.
    pub syncs: u32,
    pub seconds: f64,
}

impl Default for SyncCost {
    fn default() -> Self {
        SyncCost {
            class: DiskClass::Persistent,
            filesystem: "unknown",
            syncs: 0,
            seconds: 0.0,
        }
    }
}

struct Marker {
    path: PathBuf,
    /// Held for the pull's life: a marker whose lock is free has no writer behind it.
    _lock: File,
}

impl Marker {
    fn write(root: &Path, boot: &str, since: i64) -> Result<Marker> {
        let dir = root.join(DIR);
        match fs::create_dir(&dir) {
            Ok(()) => fsync_dir(&root.join("tmp"))?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(io("create the unsynced-admission directory", error)),
        }
        let path = dir.join(format!(
            "{boot}.{}.{}{SUFFIX}",
            std::process::id(),
            crate::store::now_nanos_unique()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o644)
            .open(&path)
            .map_err(|error| io("create an unsynced-admission marker", error))?;
        file.try_lock_exclusive()
            .map_err(|error| io("lock an unsynced-admission marker", error))?;
        file.write_all(format!("since {since}\n").as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|error| io("write an unsynced-admission marker", error))?;
        fsync_dir(&dir)?;
        Ok(Marker { path, _lock: file })
    }
}

/// One marker on disk, as a reader finds it.
struct Found {
    path: PathBuf,
    boot: String,
    /// `None` when the marker is incomplete: it was never made durable, so no admission
    /// ever depended on it.
    since: Option<i64>,
}

fn found(root: &Path) -> Result<Vec<Found>> {
    let entries = match fs::read_dir(root.join(DIR)) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(io("read the unsynced-admission markers", error)),
    };
    let mut out = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|error| io("read an unsynced-admission marker", error))?
            .path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(stem) = name.strip_suffix(SUFFIX) else {
            continue;
        };
        let boot = stem.split('.').next().unwrap_or("").to_string();
        let mut text = String::new();
        let since = File::open(&path)
            .and_then(|file| file.take(64).read_to_string(&mut text))
            .ok()
            .and_then(|_| text.strip_prefix("since "))
            .and_then(|rest| rest.strip_suffix('\n'))
            .and_then(|digits| digits.parse::<i64>().ok());
        out.push(Found { path, boot, since });
    }
    Ok(out)
}

/// Markers whose writer is gone, locked so none can be claimed twice. The lock also
/// proves the writer is gone: a live pull holds its marker's lock.
fn dead(markers: Vec<Found>) -> Vec<(Found, File)> {
    markers
        .into_iter()
        .filter_map(|marker| {
            let file = File::open(&marker.path).ok()?;
            file.try_lock_exclusive().ok()?;
            Some((marker, file))
        })
        .collect()
}

/// fsync every file, [`SYNC_THREADS`] at a time. A file already gone has nothing to lose.
fn sync_files(paths: &[PathBuf]) -> Result<()> {
    let next = AtomicUsize::new(0);
    let failed: Mutex<Option<Refusal>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..SYNC_THREADS.min(paths.len()) {
            scope.spawn(|| {
                while let Some(path) = paths.get(next.fetch_add(1, Ordering::Relaxed)) {
                    match File::open(path).and_then(|file| file.sync_data()) {
                        Ok(()) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => {
                            failed
                                .lock()
                                .unwrap_or_else(|error| error.into_inner())
                                .get_or_insert(io(format!("sync {}", path.display()), error));
                            return;
                        }
                    }
                }
            });
        }
    });
    match failed
        .into_inner()
        .unwrap_or_else(|error| error.into_inner())
    {
        Some(refusal) => Err(refusal),
        None => Ok(()),
    }
}

/// A pull's unsynced admissions on a persistent disk. `None` for every other disk and
/// wherever there is no boot identity to tell a crash from a restart; those Stores keep
/// per-object fsync (persistent) or need none (ephemeral).
pub(crate) struct Epoch {
    root: PathBuf,
    boot: String,
    /// The pull's own markers: the first, and one per backward clock step.
    markers: Mutex<Vec<Marker>>,
    /// The latest time a marker names; an admission whose clock is well behind it writes a
    /// marker for the stepped clock before its record is committed.
    since: AtomicI64,
    /// Objects admitted since the last batch sync.
    pending: Mutex<Vec<PathBuf>>,
    bytes: AtomicU64,
    every: u64,
    next_sync: AtomicU64,
    syncs: AtomicU32,
    sync_nanos: AtomicU64,
}

impl Epoch {
    pub(crate) fn begin(store: &Store) -> Result<Option<Arc<Epoch>>> {
        Epoch::begin_every(store, SYNC_EVERY)
    }

    pub(crate) fn begin_every(store: &Store, every: u64) -> Result<Option<Arc<Epoch>>> {
        if store.disk_class() != DiskClass::Persistent {
            return Ok(None);
        }
        let Some(boot) = disk::boot_id() else {
            return Ok(None);
        };
        let since = now_nanos();
        let marker = Marker::write(store.root(), &boot, since)?;
        Ok(Some(Arc::new(Epoch {
            root: store.root().to_path_buf(),
            boot,
            markers: Mutex::new(vec![marker]),
            since: AtomicI64::new(since),
            pending: Mutex::default(),
            bytes: AtomicU64::new(0),
            every,
            next_sync: AtomicU64::new(every),
            syncs: AtomicU32::new(0),
            sync_nanos: AtomicU64::new(0),
        })))
    }

    /// One object published at `path` under this epoch, before its record is committed.
    pub(crate) fn admitted(&self, path: PathBuf, bytes: u64) -> Result<()> {
        let now = now_nanos();
        if now < self.since.load(Ordering::Acquire) - SLACK_NANOS / 2 {
            let mut markers = self
                .markers
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if now < self.since.load(Ordering::Acquire) - SLACK_NANOS / 2 {
                markers.push(Marker::write(&self.root, &self.boot, now)?);
                self.since.store(now, Ordering::Release);
            }
        }
        self.pending
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(path);
        let total = self.bytes.fetch_add(bytes, Ordering::AcqRel) + bytes;
        let due = self.next_sync.load(Ordering::Acquire);
        if total >= due
            && self
                .next_sync
                .compare_exchange(
                    due,
                    total.saturating_add(self.every),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            self.sync_now()?;
        }
        Ok(())
    }

    /// fsync every object admitted since the last batch. A writer with its own durability
    /// points (a derived checkpoint) calls this there; a pull relies on [`SYNC_EVERY`].
    pub(crate) fn sync_now(&self) -> Result<()> {
        self.sync_batch(false)
    }

    /// The same, plus the directories those objects were linked into: a derived checkpoint
    /// names its objects, so their names must survive with it.
    pub(crate) fn sync_now_named(&self) -> Result<()> {
        self.sync_batch(true)
    }

    fn sync_batch(&self, names: bool) -> Result<()> {
        let batch = std::mem::take(
            &mut *self
                .pending
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        let started = Instant::now();
        sync_files(&batch)?;
        if names {
            let directories: std::collections::BTreeSet<PathBuf> = batch
                .iter()
                .filter_map(|path| path.parent().map(Path::to_path_buf))
                .collect();
            sync_files(&directories.into_iter().collect::<Vec<_>>())?;
        }
        let spent = started.elapsed();
        self.syncs.fetch_add(1, Ordering::Relaxed);
        self.sync_nanos.fetch_add(
            u64::try_from(spent.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        crate::stats::synced(spent);
        Ok(())
    }

    /// The walk is over and nothing is admitting under this epoch: sync the last batch and
    /// retire the markers. Returns the batch syncs this pull ran and their wall time.
    pub(crate) fn finish(&self) -> Result<(u32, f64)> {
        self.sync_now()?;
        let mut markers = self
            .markers
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // Dead markers of this boot are claimed BEFORE the sync, so every page their
        // writers left is older than it.
        let orphans = dead(
            found(&self.root)?
                .into_iter()
                .filter(|marker| marker.boot == self.boot)
                .filter(|marker| !markers.iter().any(|own| own.path == marker.path))
                .collect(),
        );
        if let Some(cutoff) = cutoff(orphans.iter().map(|(marker, _)| marker)) {
            let started = Instant::now();
            let paths: Vec<PathBuf> = suspects(&self.root, cutoff)?
                .into_iter()
                .map(|(_, path, _)| path)
                .collect();
            sync_files(&paths)?;
            crate::stats::synced(started.elapsed());
        }
        for marker in markers.drain(..) {
            let _ = fs::remove_file(&marker.path);
        }
        for (orphan, _lock) in &orphans {
            let _ = fs::remove_file(&orphan.path);
        }
        fsync_dir(&self.root.join(DIR))?;
        Ok((
            self.syncs.load(Ordering::Relaxed),
            self.sync_nanos.load(Ordering::Relaxed) as f64 / 1e9,
        ))
    }
}

/// What a recovery found after an unclean shutdown.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Recovery {
    pub markers: usize,
    pub rehashed: u64,
    pub removed: u64,
    pub bytes: u64,
}

/// Rehash every object an unclean shutdown may have torn, remove the torn ones, and retire
/// the markers that named them. Nothing to do, and no lock taken, without a marker left by
/// an earlier boot.
pub(crate) fn recover(store: &Store) -> Result<Option<Recovery>> {
    let boot = disk::boot_id();
    let earlier = |marker: &Found| boot.as_deref() != Some(marker.boot.as_str());
    if !found(store.root())?.iter().any(earlier) {
        return Ok(None);
    }
    let _writer = crate::catalog::WriterGuard::acquire(store.root())?;
    let dir = File::open(store.root().join(DIR))
        .map_err(|error| io("open the unsynced-admission markers", error))?;
    dir.lock_exclusive()
        .map_err(|error| io("lock the unsynced-admission markers", error))?;
    // Another process may have recovered while this one waited for the lock.
    let claimed = dead(found(store.root())?.into_iter().filter(earlier).collect());
    if claimed.is_empty() {
        return Ok(None);
    }
    let started = Instant::now();
    let mut recovery = Recovery {
        markers: claimed.len(),
        ..Recovery::default()
    };
    if let Some(cutoff) = cutoff(claimed.iter().map(|(marker, _)| marker)) {
        // A torn object's removal is durable (its directory is synced) before its marker goes.
        for (hex, path, manifest) in suspects(store.root(), cutoff)? {
            recovery.rehashed += 1;
            if manifest {
                let bytes = fs::read(&path).map_err(|error| io("read a manifest", error))?;
                recovery.bytes += bytes.len() as u64;
                if crate::sha256::hex_digest(&bytes) != hex {
                    store.remove_corrupt_manifest(&hex)?;
                    recovery.removed += 1;
                }
            } else {
                recovery.bytes += fs::metadata(&path).map_or(0, |metadata| metadata.len());
                if store.hash_object(&hex)? != hex {
                    store.remove_corrupt_blob(&hex)?;
                    recovery.removed += 1;
                }
            }
        }
    }
    for (marker, _lock) in &claimed {
        fs::remove_file(&marker.path)
            .map_err(|error| io("retire an unsynced-admission marker", error))?;
    }
    fsync_dir(&store.root().join(DIR))?;
    crate::stats::recovered(&recovery, started.elapsed());
    Ok(Some(recovery))
}

/// The oldest time any of `markers` names, less the slack. `None` when every one is
/// incomplete: a marker that was never durable covered no admission.
fn cutoff<'a>(markers: impl Iterator<Item = &'a Found>) -> Option<i64> {
    markers
        .filter_map(|marker| marker.since)
        .min()
        .map(|since| since.saturating_sub(SLACK_NANOS))
}

/// `(hex, path, is_manifest)` of every blob and manifest file whose mtime is not older than
/// `cutoff`: everything a marker from that time may cover.
fn suspects(root: &Path, cutoff: i64) -> Result<Vec<(String, PathBuf, bool)>> {
    let read = |dir: &Path| -> Result<Vec<PathBuf>> {
        match fs::read_dir(dir) {
            Ok(entries) => entries
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<_>>()
                .map_err(|error| io("list a Store namespace", error)),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) =>
            {
                Ok(Vec::new())
            }
            Err(error) => Err(io("list a Store namespace", error)),
        }
    };
    let mut out = Vec::new();
    for (namespace, suffix) in [("blobs", ""), ("manifests", ".json")] {
        for first in read(&root.join(namespace))? {
            for second in read(&first)? {
                for path in read(&second)? {
                    let Some(hex) = path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .and_then(|name| name.strip_suffix(suffix))
                        .filter(|hex| crate::ids::hex64("object", hex).is_ok())
                        .map(str::to_string)
                    else {
                        continue;
                    };
                    let metadata = match fs::symlink_metadata(&path) {
                        Ok(metadata) => metadata,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                        Err(error) => return Err(io("stat an object", error)),
                    };
                    let mtime = metadata
                        .mtime()
                        .saturating_mul(1_000_000_000)
                        .saturating_add(metadata.mtime_nsec());
                    if metadata.is_file() && mtime >= cutoff {
                        out.push((hex, path, !suffix.is_empty()));
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod testing {
    use std::os::unix::fs::{FileExt, PermissionsExt};
    use std::path::{Path, PathBuf};

    /// A fresh root beside the test binary: the build directory, a persistent disk.
    pub(crate) fn persistent_root(name: &str) -> PathBuf {
        let exe = std::env::current_exe().unwrap();
        exe.parent()
            .and_then(Path::parent)
            .unwrap()
            .join("durability-tests")
            .join(format!(
                "{name}-{}-{}",
                std::process::id(),
                crate::store::now_nanos_unique()
            ))
    }

    /// What a lost page cache can leave behind: the name, length and mtime survived, the
    /// bytes did not. The verification record still binds, so without recovery these zeros
    /// would be trusted.
    pub(crate) fn tear(path: &Path) {
        let metadata = std::fs::metadata(path).unwrap();
        let modified = metadata.modified().unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.write_all_at(&vec![0u8; metadata.len() as usize], 0)
            .unwrap();
        file.set_modified(modified).unwrap();
        drop(file);
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o444)).unwrap();
    }

    pub(crate) fn markers(root: &Path) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = std::fs::read_dir(root.join(super::DIR))
            .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
            .unwrap_or_default();
        out.retain(|path| path.to_string_lossy().ends_with(super::SUFFIX));
        out.sort();
        out
    }

    /// Stand in for a reboot: the marker now names a boot that has ended.
    pub(crate) fn from_an_earlier_boot(marker: &Path) -> PathBuf {
        let name = marker.file_name().unwrap().to_str().unwrap();
        let (_, rest) = name.split_once('.').unwrap();
        let moved = marker.with_file_name(format!("00000000-0000-0000-0000-000000000000.{rest}"));
        std::fs::rename(marker, &moved).unwrap();
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{from_an_earlier_boot, markers, persistent_root, tear};
    use super::*;
    use crate::ids::ObjectRef;
    use crate::store::Fault;

    fn put(store: &Store, bytes: &[u8]) -> ObjectRef {
        store
            .put_stream(
                &mut &bytes[..],
                Some(&ObjectRef::of(bytes)),
                &Fault::default(),
            )
            .unwrap()
            .obj
    }

    fn persistent_store(name: &str) -> Store {
        let root = persistent_root(name);
        let store = Store::init(&root).unwrap();
        assert_eq!(
            store.disk_class(),
            DiskClass::Persistent,
            "{} is not on a persistent disk",
            root.display()
        );
        store
    }

    #[test]
    fn a_persistent_epoch_syncs_at_each_boundary_and_retires_every_marker() {
        let store = persistent_store("epoch");
        let epoch = Epoch::begin_every(&store, 3 * 1024).unwrap().unwrap();
        let syncing = store.syncing(Some(Arc::clone(&epoch)));
        assert!(store.flushes() && !syncing.flushes());
        let first = markers(store.root());
        assert_eq!(first.len(), 1);
        for index in 0..4u8 {
            put(&syncing, &[index; 1024]);
        }
        assert_eq!(
            epoch.syncs.load(Ordering::Relaxed),
            1,
            "one boundary crossed"
        );
        assert_eq!(
            epoch.pending.lock().unwrap().len(),
            1,
            "the boundary synced the three objects before it"
        );
        assert_eq!(
            markers(store.root()),
            first,
            "the marker stands until the walk is over"
        );
        assert_eq!(epoch.finish().unwrap().0, 2);
        assert!(markers(store.root()).is_empty());
        let _ = fs::remove_dir_all(store.root());
    }

    #[test]
    fn a_dead_marker_of_this_boot_costs_no_rehash_and_the_next_sync_retires_it() {
        let store = persistent_store("same-boot");
        let dead_epoch = Epoch::begin_every(&store, u64::MAX).unwrap().unwrap();
        let object = put(
            &store.syncing(Some(Arc::clone(&dead_epoch))),
            b"admitted then died",
        );
        drop(dead_epoch);
        let live = Epoch::begin_every(&store, u64::MAX).unwrap().unwrap();
        assert_eq!(markers(store.root()).len(), 2);

        // The page cache of this boot still holds every page, so an open rehashes nothing:
        // bytes torn behind a standing record are NOT looked at.
        tear(&store.blob_path(&object.sha256));
        Store::open(store.root()).unwrap();
        assert!(store.blob_path(&object.sha256).exists());
        assert_eq!(markers(store.root()).len(), 2);

        let next = Epoch::begin_every(&store, u64::MAX).unwrap().unwrap();
        next.finish().unwrap();
        let left = markers(store.root());
        assert_eq!(left.len(), 1, "only the live pull's marker stands");
        live.finish().unwrap();
        assert!(markers(store.root()).is_empty());
        let _ = fs::remove_dir_all(store.root());
    }

    #[test]
    fn an_earlier_boots_marker_is_repaired_before_any_handle_is_returned() {
        let store = persistent_store("earlier-boot");
        let kept = put(&store, b"synced before the epoch");
        let epoch = Epoch::begin_every(&store, u64::MAX).unwrap().unwrap();
        let syncing = store.syncing(Some(Arc::clone(&epoch)));
        let intact = put(&syncing, b"its page reached the disk");
        let torn = put(&syncing, b"its page did not");
        drop(syncing);
        drop(epoch);
        let marker = markers(store.root()).pop().unwrap();
        from_an_earlier_boot(&marker);
        tear(&store.blob_path(&torn.sha256));
        assert!(
            store.record_valid(&torn.sha256).is_ok(),
            "the torn object's record still binds"
        );
        fs::write(store.root().join("blobs/stray"), b"not a shard directory").unwrap();

        let reopened = Store::open(store.root()).unwrap();
        assert!(!reopened.blob_path(&torn.sha256).exists());
        assert!(reopened.open_verified(&intact.sha256).is_ok());
        assert!(reopened.open_verified(&kept.sha256).is_ok());
        assert!(markers(store.root()).is_empty());
        let _ = fs::remove_dir_all(store.root());
    }

    #[test]
    fn an_incomplete_marker_covered_nothing_and_is_removed() {
        let store = persistent_store("incomplete");
        let object = put(&store, b"admitted outside any epoch");
        fs::create_dir_all(store.root().join(DIR)).unwrap();
        fs::write(
            store
                .root()
                .join(DIR)
                .join(format!("00000000-0000-0000-0000-000000000000.1.2{SUFFIX}")),
            b"since 17",
        )
        .unwrap();
        tear(&store.blob_path(&object.sha256));
        Store::open(store.root()).unwrap();
        assert!(markers(store.root()).is_empty());
        assert!(
            store.blob_path(&object.sha256).exists(),
            "no admission depended on a marker that was never durable"
        );
        let _ = fs::remove_dir_all(store.root());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn an_ephemeral_store_flushes_nothing_and_opens_no_epoch() {
        let root = Path::new("/dev/shm").join(format!(
            "tensorfs-ephemeral-{}-{}",
            std::process::id(),
            crate::store::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        assert_eq!(store.disk().kind, "tmpfs");
        assert_eq!(store.disk_class(), DiskClass::Ephemeral);
        assert!(!store.flushes());
        assert!(Epoch::begin(&store).unwrap().is_none());
        put(&store, b"no fsync on a disk that dies with the machine");
        let _ = fs::remove_dir_all(root);
    }
}
