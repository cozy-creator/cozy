//! One retention regime for every store. What roots an object:
//!
//! - a live hold: an open checkpoint, a writer, an operation in progress (`holds/<id>`, the
//!   digests it holds, under its holder's `flock`; a dead holder's file is reaped);
//! - a header file a Writer wrote or `open(path)` opened, while that file still holds that header
//!   (`files/<sha256 of the path>`);
//! - a `local/` repo ref.
//!
//! Everything else is cache. A use (`uses/<name>`: the digests it keeps, its mtime the last use)
//! keeps them and `snapshots/<name>/` for [`TTL`]; a use a live hold names is in use now. On a low
//! disk the least recently used go first. Nothing is pinned, by hand or forever.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use super::{fail, invalid, is_digest, platform, Result, Store};
use crate::format::{Header, ObjectRef, Repo};

/// How long a use keeps what it names.
pub const TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// How often maintenance expires uses.
const SWEEP: Duration = Duration::from_secs(10 * 60);

/// Digests kept while a process holds them: never unlocked by hand, so a forked child that
/// inherited the descriptor keeps holding them until it exits or execs.
#[derive(Debug)]
pub(crate) struct Hold {
    store: Store,
    file: File,
    held: Mutex<BTreeSet<String>>,
}

impl Hold {
    pub(crate) fn new(store: &Store, digests: impl IntoIterator<Item = String>) -> Result<Arc<Hold>> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let _fence = acquire(store)?;
        loop {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = store.root.join("holds").join(format!("{}-{n}", std::process::id()));
            // Windows locks need read or write access, which append alone does not grant.
            let file = match OpenOptions::new().read(true).append(true).create_new(true).open(&path) {
                Ok(file) => file,
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(fail("create", &path)(e)),
            };
            platform::lock::try_claim(&file).map_err(platform("claim", &path))?;
            let hold = Hold { store: store.clone(), file, held: Mutex::default() };
            hold.extend(digests)?;
            return Ok(Arc::new(hold));
        }
    }

    /// The short fence over reading metadata and holding what it names.
    pub(crate) fn acquire(&self) -> Result<Fence> {
        acquire(&self.store)
    }

    pub(crate) fn extend(&self, digests: impl IntoIterator<Item = String>) -> Result<()> {
        let _fence = acquire(&self.store)?;
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        let mut lines = String::new();
        for digest in digests {
            if !held.contains(&digest) {
                lines.push_str(&digest);
                lines.push('\n');
                held.insert(digest);
            }
        }
        (&self.file)
            .write_all(lines.as_bytes())
            .map_err(fail("write", self.store.root.join("holds")))
    }
}

/// Held shared by whoever reads metadata and then holds or uses what it names; exclusive by a
/// collection, which skips while any is held and never waits.
#[derive(Debug)]
pub(crate) struct Fence(#[allow(dead_code)] File);

fn fence_file(store: &Store) -> Result<File> {
    let path = store.root.join("holds").join("gc.lock");
    let opened = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path);
    opened.map_err(fail("open", &path))
}

pub(crate) fn acquire(store: &Store) -> Result<Fence> {
    let file = fence_file(store)?;
    file.lock_shared().map_err(fail("lock", &store.root))?;
    Ok(Fence(file))
}

pub(crate) fn try_collect(store: &Store) -> Result<Option<Fence>> {
    let file = fence_file(store)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(Fence(file))),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(e)) => Err(fail("lock", &store.root)(e)),
    }
}

/// `files/<id>`: a header file outside the store's own data.
#[derive(Serialize, Deserialize)]
struct FileRoot {
    path: PathBuf,
    header: ObjectRef,
}

/// The header `root`'s file holds, when it still holds it; an unreadable file still holds it.
fn file_header(root: &FileRoot) -> Option<Option<Header>> {
    let kept = |e: io::Error| !matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory);
    match fs::metadata(&root.path) {
        Ok(meta) if !meta.is_file() || meta.len() != root.header.length => None,
        Ok(_) => match fs::read(&root.path) {
            Ok(bytes) if ObjectRef::of(&bytes) == root.header => Some(Header::from_cbor(&bytes).ok()),
            Ok(_) => None,
            Err(e) => kept(e).then_some(None),
        },
        Err(e) => kept(e).then_some(None),
    }
}

/// The lines of `path` (none when absent).
fn lines(path: &Path) -> Result<Option<BTreeSet<String>>> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(Some(text.lines().map(str::to_owned).collect())),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(fail("read", path)(e)),
    }
}

pub(super) fn list(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(fail("list", dir)(e)),
    };
    entries.map(|e| e.map(|e| e.path()).map_err(fail("list", dir))).collect()
}

/// A recorded use: its name and when it was last used, and what it keeps.
pub(super) struct Use {
    pub name: String,
    pub used: SystemTime,
    pub digests: BTreeSet<String>,
}

impl Store {
    pub(crate) fn hold(&self, objects: &[ObjectRef]) -> Result<Arc<Hold>> {
        Hold::new(self, objects.iter().map(|o| o.sha256.clone()))
    }

    /// Records a use of `objects` under `name` (a digest): they and `snapshots/<name>/` stay for
    /// the TTL after the last use. A use only ever adds what it keeps.
    pub(crate) fn used<'a>(&self, name: &str, objects: impl IntoIterator<Item = &'a ObjectRef>) -> Result<()> {
        let path = self.root.join("uses").join(name);
        if !is_digest(name) {
            return Err(invalid(&path, "a use is named by a digest"));
        }
        let _fence = acquire(self)?;
        let _lock = self.lock("uses")?;
        let old = lines(&path)?;
        let mut keep = old.clone().unwrap_or_default();
        keep.extend(objects.into_iter().map(|o| o.sha256.clone()));
        if old.as_ref() == Some(&keep) {
            return touch(&path);
        }
        let text: String = keep.iter().map(|d| format!("{d}\n")).collect();
        self.commit(&path, text.as_bytes(), false)
    }

    /// Roots `header` while the file at `path` (absolute) holds it: one record per path, so saving
    /// another header there replaces it.
    pub(crate) fn root_file(&self, path: &Path, header: &ObjectRef) -> Result<()> {
        let path = platform::durable::canonical(path).map_err(platform("resolve", path))?;
        let id = ObjectRef::of(path.as_os_str().as_encoded_bytes()).sha256;
        let record = self.root.join("files").join(id);
        let root = FileRoot { path, header: header.clone() };
        let bytes = serde_json::to_vec(&root).map_err(|e| invalid(&record, e))?;
        let _fence = acquire(self)?;
        self.commit(&record, &bytes, true)
    }

    /// What the roots keep, each digest once; reaps dead holds and files that let go.
    pub(super) fn roots(&self) -> Result<BTreeSet<String>> {
        let mut roots = BTreeSet::new();
        for path in list(&self.root.join("holds"))? {
            if path.file_name().is_some_and(|n| n == "gc.lock") {
                continue;
            }
            let file = match File::open(&path) {
                Ok(file) => file,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(fail("open", &path)(e)),
            };
            // The claim, not the file, says its holder lives: a Windows lock would bar reading it.
            match platform::lock::try_claim(&file).map_err(platform("claim", &path))? {
                true => drop(fs::remove_file(&path)),
                false => roots.extend(lines(&path)?.unwrap_or_default()),
            }
        }
        for record in list(&self.root.join("files"))? {
            let bytes = fs::read(&record).map_err(fail("read", &record))?;
            let root: FileRoot = serde_json::from_slice(&bytes).map_err(|e| invalid(&record, e))?;
            match file_header(&root) {
                Some(header) => {
                    roots.extend(header.iter().flat_map(Header::objects).map(|o| o.sha256.clone()));
                    roots.insert(root.header.sha256);
                    // A header file inside a snapshot keeps its folder.
                    if let Ok(inside) = root.path.strip_prefix(self.root.join("snapshots")) {
                        roots.extend(inside.iter().next().and_then(|n| n.to_str()).map(str::to_owned));
                    }
                }
                None => drop(self.remove_record(&record)?),
            }
        }
        for repo in self.refs()?.iter().filter(|r| local(r)) {
            for tree in repo.roots() {
                roots.extend(self.closure(tree)?.into_iter().map(|o| o.sha256));
            }
        }
        Ok(roots)
    }

    /// Every recorded use; one a root names is used now.
    pub(super) fn uses(&self, roots: &BTreeSet<String>) -> Result<Vec<Use>> {
        let mut uses = Vec::new();
        for path in list(&self.root.join("uses"))? {
            let Some(name) = path.file_name().and_then(|n| n.to_str()).filter(|n| is_digest(n)) else {
                continue;
            };
            let Some(digests) = lines(&path)? else { continue };
            let used = if roots.contains(name) {
                touch(&path)?;
                SystemTime::now()
            } else {
                fs::metadata(&path).and_then(|m| m.modified()).map_err(fail("stat", &path))?
            };
            uses.push(Use { name: name.to_owned(), used, digests });
        }
        Ok(uses)
    }

    /// The cache's upkeep, run as bytes come in and checkpoints open: once per sweep period,
    /// expire uses past the TTL; then, while the disk is short of `incoming` bytes and a reserve,
    /// drop the least recently used. False when `incoming` still does not fit: skip the write.
    pub fn maintain(&self, incoming: u64) -> Result<bool> {
        let short = || shortfall(&self.root, incoming);
        let stamp = self.root.join("uses").join(".swept");
        let due = fs::metadata(&stamp)
            .and_then(|m| m.modified())
            .map_or(true, |t| t.elapsed().map_or(true, |age| age >= SWEEP));
        if !due && short()? == 0 {
            return Ok(true);
        }
        if let Some(_gc) = try_collect(self)? {
            // A store GC cannot walk (a ref or header of another format) keeps all it holds.
            if let Err(e) = self.upkeep(due, &stamp, &short) {
                tracing::warn!("tensorfs: cache upkeep skipped: {e}");
            }
        }
        Ok(short()? == 0)
    }

    fn upkeep(&self, due: bool, stamp: &Path, short: &dyn Fn() -> Result<u64>) -> Result<()> {
        if due {
            let created = File::create(stamp).and_then(|f| f.set_modified(SystemTime::now()));
            created.map_err(fail("create", stamp))?;
            self.sweep_writers(TTL)?;
            self.collect_locked(u64::MAX, TTL)?;
        }
        loop {
            let need = short()?;
            if need == 0 || self.collect_locked(need, Duration::ZERO)?.uses.is_empty() {
                return Ok(());
            }
        }
    }
}

/// `local/` repos are never evicted.
fn local(repo: &Repo) -> bool {
    repo.name.get(..6).is_some_and(|p| p.eq_ignore_ascii_case("local/"))
}

/// Bytes missing for `incoming` and a reserve of 2% of the disk, within 1 to 10 GiB.
fn shortfall(root: &Path, incoming: u64) -> Result<u64> {
    let space = platform::durable::space(root).map_err(platform("measure space", root))?;
    let reserve = (space.total / 50).clamp(1 << 30, 10 << 30);
    Ok(reserve.saturating_add(incoming.max(1)).saturating_sub(space.free))
}

fn touch(path: &Path) -> Result<()> {
    let file = OpenOptions::new().append(true).open(path).map_err(fail("open", path))?;
    file.set_modified(SystemTime::now()).map_err(fail("touch", path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{Data, Source};

    struct Scratch(Store);
    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!("tensorfs-retention-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&root);
            Self(Store::open(&root).unwrap().with_grace(Duration::ZERO))
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0.root);
        }
    }

    fn age(path: &Path, by: Duration) {
        if path.is_dir() {
            for entry in fs::read_dir(path).unwrap() {
                age(&entry.unwrap().path(), by);
            }
        }
        platform::durable::set_times(path, fs::FileTimes::new().set_modified(SystemTime::now() - by)).unwrap();
    }

    /// Maintenance expires a use past the TTL, and its folder, and reaps abandoned writers; a
    /// live hold keeps its use fresh, and a recent keyed journal still resumes.
    #[test]
    fn maintenance_expires_uses_past_the_ttl_but_not_held_ones() {
        let scratch = Scratch::new();
        let store = &scratch.0;
        let (old, held) = (store.put_bytes(&[1; 4096]).unwrap(), store.put_bytes(&[2; 4096]).unwrap());
        store.used(&old.sha256, [&old]).unwrap();
        store.used(&held.sha256, [&held]).unwrap();
        let folder = store.root.join("snapshots").join(&old.sha256);
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("copy"), b"cache").unwrap();
        let hold = store.hold(std::slice::from_ref(&held)).unwrap();
        let write = |key| {
            let mut w = store.writer(None, Some(key)).unwrap();
            let data = Data::Plain(Source::Read(&mut &[3u8; 64][..]));
            w.get("model").put("weight", &crate::format::Dtype::U8, &[64], data).unwrap();
            w
        };
        drop(write("abandoned"));
        drop(write("recent"));
        let past = TTL + Duration::from_secs(60);
        for path in ["uses", "snapshots", "blobs", "tmp/writers/abandoned"] {
            age(&store.root.join(path), past);
        }
        age(&store.root.join("uses/.swept"), SWEEP);
        assert!(store.maintain(0).unwrap());
        assert!(!store.path(&old).unwrap().exists() && !folder.exists(), "an expired use stayed");
        assert!(!store.root.join("uses").join(&old.sha256).exists());
        assert!(store.reuse(&held).unwrap(), "a held use expired");
        let fresh = fs::metadata(store.root.join("uses").join(&held.sha256)).unwrap().modified().unwrap();
        assert!(fresh.elapsed().unwrap() < SWEEP, "a held use was not refreshed");
        assert!(!store.root.join("tmp/writers/abandoned").exists());
        let recent = store.writer(None, Some("recent")).unwrap();
        assert!(recent.has("model", "weight"), "the recent keyed journal still resumes");
        drop(recent);
        drop(hold);
        assert_eq!(store.collect(u64::MAX, Duration::ZERO).unwrap().uses, std::slice::from_ref(&held.sha256));
        assert!(!store.path(&held).unwrap().exists());
    }

    /// A dead holder's file is reaped by the next collection; a live one's is kept and read.
    #[test]
    fn a_dead_hold_is_reaped_and_a_live_one_roots() {
        let scratch = Scratch::new();
        let store = &scratch.0;
        let object = store.put_bytes(b"held").unwrap();
        let hold = store.hold(std::slice::from_ref(&object)).unwrap();
        hold.extend(["not-an-object".to_owned()]).unwrap();
        store.gc().unwrap();
        assert!(store.reuse(&object).unwrap());
        assert_eq!(list(&store.root.join("holds")).unwrap().len(), 2, "the hold and the fence");
        drop(hold);
        store.gc().unwrap();
        assert!(!store.reuse(&object).unwrap());
        assert_eq!(list(&store.root.join("holds")).unwrap().len(), 1, "only the fence");
    }
}
