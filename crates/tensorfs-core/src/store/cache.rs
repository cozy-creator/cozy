//! Policy for an explicitly enabled library cache. Raw object stores keep their own policy.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::retention::{self, DirectoryGuard};
use super::{fail, invalid, is_digest, now_secs, platform, Pin, Result, Store};
use crate::format::{ObjectRef, Repo};

pub(super) const TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);
pub(super) const SWEEP: Duration = Duration::from_secs(10 * 60);
const RESERVE_MIN: u64 = 1 << 30;
const RESERVE_MAX: u64 = 10 << 30;
const RESERVE_INODES: u64 = 32;

/// The existing filesystem sample seam; pressure tests sample space and delete real files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Disk {
    pub capacity: u64,
    pub available: u64,
    pub inodes: u64,
    pub available_inodes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Need {
    pub bytes: u64,
    pub inodes: u64,
}

impl Disk {
    pub fn measure(root: &Path) -> Result<Self> {
        let sample = platform::durable::space(root).map_err(platform("measure space", root))?;
        Ok(Self {
            capacity: sample.total,
            available: sample.free,
            inodes: sample.inodes,
            available_inodes: sample.free_inodes,
        })
    }

    pub fn reserve(self) -> u64 {
        (self.capacity / 50).clamp(RESERVE_MIN, RESERVE_MAX)
    }

    pub fn need(self, incoming: u64) -> Need {
        Need {
            bytes: self
                .reserve()
                .saturating_add(incoming.max(1))
                .saturating_sub(self.available),
            inodes: if self.inodes == 0 {
                0
            } else {
                (RESERVE_INODES + 1).saturating_sub(self.available_inodes)
            },
        }
    }
}

impl Need {
    pub fn covered(self, bytes: u64, inodes: u64) -> bool {
        bytes >= self.bytes && inodes >= self.inodes
    }

    pub fn empty(self) -> bool {
        self == Self::default()
    }
}

/// Opaque native ownership across a cache fetch batch; not a Python management API.
#[doc(hidden)]
pub struct CacheGuard {
    _retention: Option<Arc<retention::Retention>>,
    _directory: Option<DirectoryGuard>,
}

#[derive(Deserialize, Serialize)]
struct CachedRef {
    fingerprint: String,
}

#[derive(Deserialize, Serialize)]
struct Used {
    last_use: u64,
}

#[derive(Deserialize, Serialize)]
struct FolderUse {
    last_use: u64,
    #[serde(default)]
    owned: bool,
}

pub(super) struct Snapshot {
    pub path: PathBuf,
    pub used: u64,
    record: PathBuf,
    _guard: DirectoryGuard,
}

fn read<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<Option<T>> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| invalid(path, e)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(fail("read cache record", path)(e)),
    }
}

fn fingerprint(repo: &Repo) -> Result<String> {
    Ok(ObjectRef::of(&repo.to_json()?).sha256)
}

impl Store {
    /// Binding-only opt-in. Opening a raw object store never enables automatic eviction.
    #[doc(hidden)]
    pub fn with_cache(mut self) -> Self {
        self.cache_mode = true;
        self
    }

    #[doc(hidden)]
    pub fn cache_hold(&self, objects: &[ObjectRef]) -> Result<CacheGuard> {
        Ok(CacheGuard {
            _retention: if self.cache_mode {
                Some(retention::Retention::new(self, objects.iter().cloned())?)
            } else {
                None
            },
            _directory: None,
        })
    }

    fn cache_state(&self) -> PathBuf {
        self.root.join("tmp").join("cache")
    }

    fn cache_lock(&self) -> Result<platform::lock::DirLock> {
        let state = self.cache_state();
        fs::create_dir_all(&state).map_err(fail("create cache state", &state))?;
        platform::lock::dir(&state).map_err(platform("lock cache maintenance", state))
    }

    fn cache_write(&self, path: &Path, value: &impl Serialize) -> Result<()> {
        let parent = path.parent().expect("a cache record has a parent");
        fs::create_dir_all(parent).map_err(fail("create cache records", parent))?;
        let bytes = serde_json::to_vec(value).map_err(|e| invalid(path, e))?;
        self.commit(path, &bytes, false)
    }

    fn cached_ref_path(&self, name: &str) -> PathBuf {
        let key = ObjectRef::of(name.to_ascii_lowercase().as_bytes()).sha256;
        self.cache_state().join("refs").join(format!("{key}.json"))
    }

    /// Mark exactly the remote metadata the caller installed, never a concurrently changed ref.
    #[doc(hidden)]
    pub fn cache_ref(&self, repo: &Repo) -> Result<()> {
        if !self.cache_mode || repo.name.to_ascii_lowercase().starts_with("local/") {
            return Ok(());
        }
        let _lock = self.lock("refs")?;
        if self.get_ref(&repo.name)?.as_ref() != Some(repo) {
            return Ok(());
        }
        self.cache_write(
            &self.cached_ref_path(&repo.name),
            &CachedRef {
                fingerprint: fingerprint(repo)?,
            },
        )
    }

    pub(super) fn cached_ref(&self, repo: &Repo) -> Result<bool> {
        if !self.cache_mode || repo.name.to_ascii_lowercase().starts_with("local/") {
            return Ok(false);
        }
        let Some(cached) = read::<CachedRef>(&self.cached_ref_path(&repo.name))? else {
            return Ok(false);
        };
        Ok(fingerprint(repo)? == cached.fingerprint)
    }

    /// Called under the ref directory lock by every explicit ref mutation, even raw stores.
    pub(super) fn clear_cached_ref(&self, name: &str) -> Result<bool> {
        let path = self.cached_ref_path(name);
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(fail("clear cached ref classification", path)(e)),
        }
    }

    /// Root ownership must be acquired first. This records cache recency, not a live owner.
    #[doc(hidden)]
    pub fn cache_touch(&self, root: &ObjectRef) -> Result<()> {
        if self.cache_mode {
            self.pin(
                &format!("cache-{}", root.sha256),
                &Pin::new(vec![root.clone()]),
            )?;
        }
        Ok(())
    }

    pub(super) fn cache_entry(&self, path: &Path) -> Option<(PathBuf, PathBuf)> {
        if !self.cache_mode {
            return None;
        }
        for kind in ["snapshots", "views"] {
            let parent = self.root.join(kind);
            let Ok(rest) = path.strip_prefix(&parent) else {
                continue;
            };
            let Some(part) = rest.components().next() else {
                continue;
            };
            let Some(name) = part.as_os_str().to_str() else {
                continue;
            };
            if is_digest(name) {
                let record = self
                    .cache_state()
                    .join("snapshots")
                    .join(format!("{kind}-{name}.json"));
                return Some((parent.join(name), record));
            }
        }
        None
    }

    /// Only internally generated paths receive a recency record; local_dir is never classified.
    #[doc(hidden)]
    pub fn cache_snapshot(&self, path: &Path) -> Result<()> {
        let _acquisition = if self.cache_mode {
            Some(retention::acquire(self)?)
        } else {
            None
        };
        if let Some((entry, record)) = self.cache_entry(path) {
            let _classification = self.cache_lock()?;
            let old = read::<FolderUse>(&record)?;
            if !old.as_ref().is_some_and(|record| record.owned)
                && (old.is_some() || !entry.exists())
            {
                self.cache_write(
                    &record,
                    &FolderUse {
                        last_use: now_secs(),
                        owned: false,
                    },
                )?;
            }
        }
        Ok(())
    }

    /// An explicit local_dir remains user-owned even if it is inside this cache's layout.
    #[doc(hidden)]
    pub fn cache_preserve_snapshot(&self, path: &Path) -> Result<()> {
        let _acquisition = if self.cache_mode {
            Some(retention::acquire(self)?)
        } else {
            None
        };
        if let Some((_, record)) = self.cache_entry(path) {
            let _classification = self.cache_lock()?;
            self.cache_write(
                &record,
                &FolderUse {
                    last_use: now_secs(),
                    owned: true,
                },
            )?;
        }
        Ok(())
    }

    pub(super) fn cache_folder_guard(&self, path: &Path) -> Result<Option<DirectoryGuard>> {
        let Some((entry, record)) = self.cache_entry(path) else {
            return Ok(None);
        };
        if !read::<FolderUse>(&record)?.is_some_and(|record| !record.owned) {
            return Ok(None);
        }
        Ok(Some(retention::directory(self, &entry)?))
    }

    #[doc(hidden)]
    pub fn cache_directory(&self, path: &Path) -> Result<CacheGuard> {
        let _acquisition = if self.cache_mode {
            Some(retention::acquire(self)?)
        } else {
            None
        };
        self.cache_snapshot(path)?;
        let directory = self.cache_folder_guard(path)?;
        Ok(CacheGuard {
            _retention: None,
            _directory: directory,
        })
    }

    pub(super) fn cache_snapshots(&self) -> Result<Vec<Snapshot>> {
        let mut snapshots = Vec::new();
        for kind in ["snapshots", "views"] {
            let dir = self.root.join(kind);
            let entries = match fs::read_dir(&dir) {
                Ok(entries) => entries,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(fail("list cache folders", dir)(e)),
            };
            for entry in entries {
                let entry = entry.map_err(fail("list cache folder", &dir))?;
                let path = entry.path();
                let Some((_, record)) = self.cache_entry(&path) else {
                    continue;
                };
                // Unclassified raw/operator folders are not silently adopted by the cache.
                let Some(used) = read::<FolderUse>(&record)? else {
                    continue;
                };
                if used.owned {
                    continue;
                }
                let metadata = match fs::symlink_metadata(&path) {
                    Ok(meta) if meta.is_dir() && !meta.file_type().is_symlink() => meta,
                    Ok(_) => continue,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                    Err(e) => return Err(fail("stat cache folder", &path)(e)),
                };
                let Some(guard) = retention::try_directory(&path)? else {
                    continue;
                };
                let changed = metadata
                    .modified()
                    .map_err(fail("stat cache folder", &path))?
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                snapshots.push(Snapshot {
                    path,
                    record,
                    used: used.last_use.max(changed),
                    _guard: guard,
                });
            }
        }
        Ok(snapshots)
    }

    #[doc(hidden)]
    pub fn cache_maintain(&self, incoming: u64) -> Result<bool> {
        self.cache_maintain_with(incoming, now_secs(), &|| Disk::measure(&self.root))
    }

    fn cache_maintain_with(
        &self,
        incoming: u64,
        now: u64,
        sample: &dyn Fn() -> Result<Disk>,
    ) -> Result<bool> {
        if !self.cache_mode {
            return Ok(true);
        }
        let disk = sample()?;
        let stamp = self.cache_state().join("sweep.json");
        let due = |last: Option<Used>| {
            last.is_none_or(|last| {
                now.saturating_sub(last.last_use) >= SWEEP.as_secs() || now < last.last_use
            })
        };
        if disk.need(incoming).empty() && !due(read(&stamp)?) {
            return Ok(true);
        }
        let state = self.cache_state();
        fs::create_dir_all(&state).map_err(fail("create cache state", &state))?;
        let _maintenance =
            platform::lock::dir(&state).map_err(platform("lock cache maintenance", &state))?;
        let Some(_collection) = retention::try_collect(self)? else {
            return Ok(sample()?.need(incoming).empty());
        };
        if due(read(&stamp)?) {
            let cutoff = now.saturating_sub(TTL.as_secs());
            for (id, pin) in self.pins()? {
                if id.starts_with("cache-") && pin.last_use <= cutoff {
                    self.unpin_unused(&id, pin.last_use)?;
                }
            }
            for snapshot in self.cache_snapshots()? {
                if snapshot.used <= cutoff {
                    snapshot.remove(self)?;
                }
            }
            self.collect_locked(0, Duration::MAX)?;
            self.cache_write(&stamp, &Used { last_use: now })?;
        }
        if !sample()?.need(incoming).empty() {
            self.cache_pressure(sample()?.need(incoming))?;
        }
        Ok(sample()?.need(incoming).empty())
    }
}

impl Snapshot {
    pub(super) fn files(&self) -> Result<Vec<PathBuf>> {
        fn walk(path: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
            let meta = fs::symlink_metadata(path).map_err(fail("stat cache candidate", path))?;
            if meta.file_type().is_symlink() {
                return Ok(());
            }
            out.push(path.to_owned());
            if meta.is_dir() {
                for child in fs::read_dir(path).map_err(fail("list cache candidate", path))? {
                    walk(
                        &child.map_err(fail("list cache candidate", path))?.path(),
                        out,
                    )?;
                }
            }
            Ok(())
        }
        let mut files = vec![self.record.clone()];
        walk(&self.path, &mut files)?;
        Ok(files)
    }

    pub(super) fn remove(self, store: &Store) -> Result<()> {
        let trash = store.cache_state().join("pruned");
        fs::create_dir_all(&trash).map_err(fail("create cache prune directory", &trash))?;
        let name = self
            .record
            .file_stem()
            .expect("a cache record has a name")
            .to_string_lossy();
        let grave = trash.join(name.as_ref());
        // Maintenance is serialized. Finish a prior interrupted prune before reusing its name.
        if grave.exists() {
            fs::remove_dir_all(&grave).map_err(fail("finish interrupted cache prune", &grave))?;
        }
        match fs::rename(&self.path, &grave) {
            Ok(()) => {
                fs::remove_dir_all(&grave).map_err(fail("remove inactive cache folder", &grave))?
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(fail("retire inactive cache folder", &self.path)(e)),
        }
        store.remove_record(&self.record)?;
        retention::remove_directory_fence(&self.path, &self._guard)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{Body, Component, Dtype, Header, Location, Release, Tensor, Tree};
    use std::fs::File;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Scratch(Store);
    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "tensorfs-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut store = Store::open(&root).unwrap().with_cache();
            store.grace = Duration::ZERO;
            // Suppress the periodic TTL pass: these tests isolate sampled pressure.
            store
                .cache_write(
                    &store.cache_state().join("sweep.json"),
                    &Used {
                        last_use: now_secs(),
                    },
                )
                .unwrap();
            Self(store)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0.root);
        }
    }

    fn payload(store: &Store, byte: u8) -> ObjectRef {
        let object = store.put_bytes(&vec![byte; 32 << 10]).unwrap();
        store.cache_touch(&object).unwrap();
        object
    }

    fn ample() -> Disk {
        Disk {
            capacity: 8 << 30,
            available: 4 << 30,
            inodes: 1000,
            available_inodes: 1000,
        }
    }

    /// Only the pressure sample is injected. Reclamation and post-sweep accounting use real
    /// files/allocated bytes; this does not claim to exhaust a runner filesystem.
    fn sampled(store: &Store, objects: &[ObjectRef], deficit: u64) -> impl Fn() -> Result<Disk> {
        let paths: Vec<_> = objects
            .iter()
            .map(|object| store.path(object).unwrap())
            .collect();
        let allocated = |path: &Path| {
            fs::metadata(path)
                .ok()
                .map_or(0, |metadata| platform::durable::allocated(&metadata))
        };
        let before: u64 = paths.iter().map(|path| allocated(path)).sum();
        move || {
            let current: u64 = paths.iter().map(|path| allocated(path)).sum();
            Ok(Disk {
                available: ample()
                    .reserve()
                    .saturating_sub(deficit)
                    .saturating_add(before.saturating_sub(current)),
                ..ample()
            })
        }
    }

    #[test]
    fn policy_uses_approved_reserve_ttl_and_inode_limits() {
        assert_eq!(TTL.as_secs(), 7 * 86400);
        assert_eq!(SWEEP.as_secs(), 600);
        assert_eq!(ample().reserve(), RESERVE_MIN);
        assert_eq!(
            Disk {
                capacity: u64::MAX,
                ..ample()
            }
            .reserve(),
            RESERVE_MAX
        );
        let disk = Disk {
            available: RESERVE_MIN,
            available_inodes: 32,
            ..ample()
        };
        assert_eq!(
            disk.need(0),
            Need {
                bytes: 1,
                inodes: 1
            }
        );
        assert_eq!(Disk { inodes: 0, ..disk }.need(0).inodes, 0);
    }

    #[test]
    fn pressure_evicts_real_inactive_bytes_and_raw_mode_does_nothing() {
        let scratch = Scratch::new();
        let store = &scratch.0;
        let object = payload(store, 1);
        let sample = sampled(store, std::slice::from_ref(&object), 4096);
        let mut raw = store.clone();
        raw.cache_mode = false;
        assert!(raw.cache_maintain_with(0, now_secs(), &sample).unwrap());
        assert!(store.path(&object).unwrap().exists());
        assert!(store.cache_maintain_with(0, now_secs(), &sample).unwrap());
        assert!(!store.path(&object).unwrap().exists());
        assert!(store
            .get_pin(&format!("cache-{}", object.sha256))
            .unwrap()
            .is_none());
        assert!(
            sample().unwrap().need(0).empty(),
            "real allocated payload bytes disappeared"
        );
    }

    #[test]
    fn active_holds_and_open_fds_refuse_then_release_to_positive_eviction() {
        let scratch = Scratch::new();
        let store = &scratch.0;
        let object = payload(store, 2);
        let sample = sampled(store, std::slice::from_ref(&object), 4096);
        let hold = store.cache_hold(std::slice::from_ref(&object)).unwrap();
        assert!(!store.cache_maintain_with(0, now_secs(), &sample).unwrap());
        assert!(store
            .get_pin(&format!("cache-{}", object.sha256))
            .unwrap()
            .is_some());
        drop(hold);
        let file = File::open(store.path(&object).unwrap()).unwrap();
        file.lock_shared().unwrap();
        assert!(!store.cache_maintain_with(0, now_secs(), &sample).unwrap());
        assert!(store
            .get_pin(&format!("cache-{}", object.sha256))
            .unwrap()
            .is_some());
        drop(file);
        assert!(store.cache_maintain_with(0, now_secs(), &sample).unwrap());
        assert!(!store.path(&object).unwrap().exists());
    }

    #[test]
    fn external_hardlink_is_not_capacity_and_failed_cover_preserves_fresh_roots() {
        let scratch = Scratch::new();
        let store = &scratch.0;
        let linked = payload(store, 3);
        let other = payload(store, 4);
        let copy = store.root.join("manual-copy.bin");
        fs::hard_link(store.path(&linked).unwrap(), &copy).unwrap();
        let sample = sampled(store, &[linked.clone(), other.clone()], 64 << 10);
        assert!(!store.cache_maintain_with(0, now_secs(), &sample).unwrap());
        for object in [&linked, &other] {
            assert!(store
                .get_pin(&format!("cache-{}", object.sha256))
                .unwrap()
                .is_some());
            assert!(store.path(object).unwrap().exists());
        }
        assert_eq!(fs::read(&copy).unwrap(), vec![3; 32 << 10]);
    }

    #[test]
    fn ttl_keeps_remote_resolution_metadata_local_refs_and_manual_snapshots() {
        let scratch = Scratch::new();
        let store = &scratch.0;
        let chunk = payload(store, 5);
        let tensor = Tensor {
            key: "weight".into(),
            dtype: Dtype::U8,
            shape: vec![chunk.length],
            body: Body::Data(Location {
                chunk: 0,
                offset: 0,
                length: chunk.length,
            }),
        };
        let header = Header::new(
            vec![chunk.clone()],
            None,
            vec![Component {
                name: "model".into(),
                tensors: vec![tensor],
                ..Default::default()
            }],
            Default::default(),
        )
        .unwrap();
        let header = store.put_bytes(&header.to_cbor().unwrap()).unwrap();
        let mut tree = Tree::default();
        tree.set_variant("int8", header.clone());
        let tree = store.put_tree(&tree).unwrap();
        let repo = Repo {
            name: "remote/model".into(),
            releases: vec![Release {
                version: "1.0".into(),
                tree: tree.clone(),
                yanked: false,
                revision: 0,
            }],
            ..Default::default()
        };
        store
            .update_ref(&repo.name, |_| Some(repo.clone()))
            .unwrap();
        store.cache_ref(&repo).unwrap();
        store.cache_touch(&tree).unwrap();
        let local = payload(store, 6);
        let local_repo = Repo {
            name: "local/model".into(),
            releases: vec![Release {
                version: "1.0".into(),
                tree: local.clone(),
                yanked: false,
                revision: 0,
            }],
            ..Default::default()
        };
        store
            .update_ref(&local_repo.name, |_| Some(local_repo.clone()))
            .unwrap();
        store.cache_ref(&local_repo).unwrap();
        let inactive = store.root.join("snapshots").join(&tree.sha256);
        drop(store.cache_directory(&inactive).unwrap());
        fs::create_dir_all(&inactive).unwrap();
        fs::write(inactive.join("copy"), b"cache").unwrap();
        let owned = store.root.join("views").join(&tree.sha256);
        drop(store.cache_directory(&owned).unwrap());
        store.cache_preserve_snapshot(&owned).unwrap();
        fs::create_dir_all(&owned).unwrap();
        fs::write(owned.join("copy"), b"user").unwrap();
        let live_path = store.root.join("views").join(&header.sha256);
        let active = store.cache_directory(&live_path).unwrap();
        fs::create_dir_all(&live_path).unwrap();
        assert!(store
            .cache_maintain_with(0, now_secs() + TTL.as_secs() + 1, &|| Ok(ample()))
            .unwrap());
        assert!(store.path(&tree).unwrap().exists());
        assert!(store.path(&header).unwrap().exists());
        assert!(!store.path(&chunk).unwrap().exists());
        assert!(store.path(&local).unwrap().exists());
        assert!(!inactive.exists());
        assert!(owned.join("copy").exists());
        assert!(live_path.exists());
        drop(active);
    }

    #[test]
    fn explicit_ref_update_revokes_cache_classification() {
        let scratch = Scratch::new();
        let store = &scratch.0;
        let object = payload(store, 7);
        let repo = Repo {
            name: "remote/published".into(),
            releases: vec![Release {
                version: "1".into(),
                tree: object.clone(),
                yanked: false,
                revision: 0,
            }],
            ..Default::default()
        };
        store
            .update_ref(&repo.name, |_| Some(repo.clone()))
            .unwrap();
        store.cache_ref(&repo).unwrap();
        assert!(store.cached_ref(&repo).unwrap());
        store.update_ref(&repo.name, |old| old).unwrap();
        assert!(!store.cached_ref(&repo).unwrap());
        assert!(store
            .cache_maintain_with(0, now_secs() + TTL.as_secs() + 1, &|| Ok(ample()))
            .unwrap());
        assert!(store.path(&object).unwrap().exists());
    }
}
