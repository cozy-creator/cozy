//! Native owners retain ordinary tree roots while their kernel-held fence is locked.
//! Final release closes descriptors only; collection reaps unlocked records and directories.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::{fail, invalid, Pin, Result, Store};
use crate::format::{ObjectRef, Tree};

pub(crate) const PREFIX: &str = "hold-";

#[derive(Debug)]
pub(crate) struct Retention {
    store: Store,
    id: String,
    objects: Mutex<BTreeMap<String, ObjectRef>>,
    // Never explicitly unlock: an inherited descriptor in a fork child owns the same fence.
    _fence: File,
}

impl Retention {
    pub(crate) fn new(
        store: &Store,
        objects: impl IntoIterator<Item = ObjectRef>,
    ) -> Result<Arc<Self>> {
        // Internal ownership metadata must not recursively invoke cache admission/maintenance.
        let mut store = store.clone();
        store.cache_mode = false;
        let _acquisition = acquire(&store)?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let parent = store.root().join("tmp/holds");
        fs::create_dir_all(&parent).map_err(fail("create", &parent))?;
        let (id, fence) = loop {
            let tick = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let id = format!(
                "{PREFIX}{}-{tick:x}-{:x}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            );
            let dir = parent.join(&id);
            match fs::create_dir(&dir) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(fail("create", &dir)(e)),
            }
            let path = dir.join("lock");
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(fail("create", &path))?;
            file.lock().map_err(fail("lock", &path))?;
            break (id, file);
        };
        let hold = Arc::new(Self {
            store,
            id,
            objects: Mutex::default(),
            _fence: fence,
        });
        hold.extend(objects)?;
        Ok(hold)
    }

    pub(crate) fn acquire(&self) -> Result<AcquisitionGuard> {
        acquire(&self.store)
    }

    pub(crate) fn extend(&self, objects: impl IntoIterator<Item = ObjectRef>) -> Result<()> {
        let _acquisition = acquire(&self.store)?;
        let mut held = self.objects.lock().unwrap();
        let mut next = held.clone();
        for object in objects {
            if next.get(&object.sha256).is_some_and(|old| old != &object) {
                return Err(invalid(
                    self.store.root(),
                    "a retained digest has conflicting lengths",
                ));
            }
            next.insert(object.sha256.clone(), object);
        }
        if next == *held && self.store.get_pin(&self.id)?.is_some() {
            return Ok(());
        }
        // Non-variant entries retain raw objects without trying to parse chunk bytes as headers.
        let tree = Tree {
            files: next
                .iter()
                .map(|(sha, object)| (format!("held/{sha}"), object.clone()))
                .collect(),
        };
        let root = self.store.put_tree(&tree)?;
        self.store.pin(&self.id, &Pin::new(vec![root]))?;
        *held = next;
        Ok(())
    }
}

fn fence(store: &Store, id: &str) -> PathBuf {
    store.root().join("tmp/holds").join(id).join("lock")
}

/// A reserved hold record is live only while its ordinary kernel fence is held.
pub(crate) fn protected(store: &Store, id: &str) -> Result<bool> {
    if !id.starts_with(PREFIX) {
        return Ok(false);
    }
    let path = fence(store, id);
    let file = match OpenOptions::new().read(true).write(true).open(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(fail("open", &path)(e)),
    };
    match file.try_lock() {
        Ok(()) => Ok(false),
        Err(TryLockError::WouldBlock) => Ok(true),
        Err(TryLockError::Error(e)) => Err(fail("lock", &path)(e)),
    }
}

/// Called by maintenance, never by Drop. A live or fork-inherited fence is never removed.
pub(crate) fn reap(store: &Store) -> Result<usize> {
    let mut removed = 0;
    let mut ids: BTreeSet<String> = store
        .pins()?
        .into_iter()
        .map(|(id, _)| id)
        .filter(|id| id.starts_with(PREFIX))
        .collect();
    let parent = store.root().join("tmp/holds");
    match fs::read_dir(&parent) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry.map_err(fail("list", &parent))?;
                if entry
                    .file_type()
                    .map_err(fail("stat", entry.path()))?
                    .is_dir()
                {
                    if let Some(id) = entry
                        .file_name()
                        .to_str()
                        .filter(|id| id.starts_with(PREFIX))
                    {
                        ids.insert(id.to_owned());
                    }
                }
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(fail("list", &parent)(e)),
    }
    for id in ids {
        let path = fence(store, &id);
        let guard = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(file) => match file.try_lock() {
                Ok(()) => Some(file),
                Err(TryLockError::WouldBlock) => continue,
                Err(TryLockError::Error(e)) => return Err(fail("lock", &path)(e)),
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(fail("open", &path)(e)),
        };
        if store.remove_hold_pin(&id)? {
            removed += 1;
        }
        if let Some(dir) = path.parent() {
            match fs::remove_dir_all(dir) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(fail("remove", dir)(e)),
            }
        }
        drop(guard);
    }
    Ok(removed)
}

#[derive(Debug)]
pub(crate) struct DirectoryGuard(#[allow(dead_code)] File);

#[derive(Debug)]
pub(crate) struct AcquisitionGuard(#[allow(dead_code)] File);

fn collection_file(store: &Store) -> Result<File> {
    let path = store.root().join("tmp/holds/gc.lock");
    fs::create_dir_all(path.parent().expect("a collection fence has a parent"))
        .map_err(fail("create", &path))?;
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(fail("open", &path))
}

/// Only the short metadata-to-retention publication, never a checkpoint's lifetime.
pub(crate) fn acquire(store: &Store) -> Result<AcquisitionGuard> {
    let file = collection_file(store)?;
    file.lock_shared().map_err(fail("lock", store.root()))?;
    Ok(AcquisitionGuard(file))
}

/// Collection skips a concurrent acquisition; it never waits on a caller/provider.
pub(crate) fn try_collect(store: &Store) -> Result<Option<AcquisitionGuard>> {
    let file = collection_file(store)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(AcquisitionGuard(file))),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(e)) => Err(fail("lock", store.root())(e)),
    }
}

fn sidecar(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| invalid(path, "a cache view needs a parent"))?;
    let mut name = OsString::from(".");
    name.push(
        path.file_name()
            .ok_or_else(|| invalid(path, "a cache view needs a name"))?,
    );
    name.push(".tensorfs-lock");
    Ok(parent.join(name))
}

fn directory_file(path: &Path) -> Result<File> {
    let sidecar = sidecar(path)?;
    fs::create_dir_all(sidecar.parent().expect("a sidecar has a parent"))
        .map_err(fail("create", &sidecar))?;
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&sidecar)
        .map_err(fail("open", sidecar))
}

pub(crate) fn directory(store: &Store, path: &Path) -> Result<DirectoryGuard> {
    let _acquisition = acquire(store)?;
    let file = directory_file(path)?;
    file.lock_shared().map_err(fail("lock", path))?;
    Ok(DirectoryGuard(file))
}

pub(crate) fn try_directory(path: &Path) -> Result<Option<DirectoryGuard>> {
    let file = directory_file(path)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(DirectoryGuard(file))),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(e)) => Err(fail("lock", path)(e)),
    }
}

/// Maintenance only, after removing the entry while holding both the exclusive collection
/// gate and this entry's exclusive directory guard. New openers take the shared gate first.
pub(crate) fn remove_directory_fence(path: &Path, _guard: &DirectoryGuard) -> Result<()> {
    let sidecar = sidecar(path)?;
    match fs::remove_file(&sidecar) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(fail("remove", sidecar)(e)),
    }
}
