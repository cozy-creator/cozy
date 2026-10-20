//! GC: mark what the roots and the uses keep (`retention.rs`), then sweep unmarked objects and
//! snapshot folders older than the grace. A collection may first drop uses: those past a TTL, or
//! least recently used first until enough bytes would be free.
//!
//! A swept object is first renamed into `tmp/gc/`, then deleted only if it is still old: a put
//! or reuse that raced the sweep made it young, and it goes back. A GC killed in between leaves
//! it there, and the next GC puts it back before deciding again.

use std::collections::{BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{self, Read};
use std::time::{Duration, SystemTime};

use super::retention::{self, list, Use};
use super::{fail, fault, invalid, is_digest, same_family, Result, Store};
use crate::format::{self, variant_of, Header, ObjectRef, Tree, HEADER_FORMAT, TREE_FORMAT};

/// Documents bigger than this are never read as trees or headers.
const MAX_DOCUMENT: u64 = 256 << 20;

/// What one collection took.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Collected {
    /// The uses dropped, by name.
    pub uses: Vec<String>,
    pub objects: u64,
    pub bytes: u64,
}

enum Document {
    Tree(Tree),
    Header(Header),
}

impl Store {
    /// What `root` keeps alive, each once: `root`, and for a tree its files and each variant
    /// header's chunks and assets; for a header, its chunks and assets.
    pub fn closure(&self, root: &ObjectRef) -> Result<Vec<ObjectRef>> {
        let mut reach = BTreeSet::from([root.clone()]);
        match self.document(root)? {
            Some(Document::Tree(tree)) => {
                for (path, file) in tree.files {
                    if variant_of(&path).is_some() {
                        reach.extend(self.header_objects(&file)?);
                    }
                    reach.insert(file);
                }
            }
            Some(Document::Header(header)) => reach.extend(header.objects().cloned()),
            None => {}
        }
        Ok(reach.into_iter().collect())
    }

    /// The tree or header `object` is; `None` for any other object or an absent one. One of a
    /// kind this build cannot walk (another major, a pre-cut header) stops GC rather than sweep
    /// what it reaches.
    fn document(&self, object: &ObjectRef) -> Result<Option<Document>> {
        let path = self.path(object)?;
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(fail("open", &path)(error)),
        };
        let length = file.metadata().map_err(fail("stat", &path))?.len();
        let mut first = [0u8];
        if length > MAX_DOCUMENT || file.read_exact(&mut first).is_err() {
            return Ok(None);
        }
        // A tree is a JSON object; a header a CBOR array (five elements: pre-cut).
        if first[0] != b'{' && !(0x85..=0x9f).contains(&first[0]) {
            return Ok(None);
        }
        let mut bytes = first.to_vec();
        file.read_to_end(&mut bytes).map_err(fail("read", &path))?;
        let unwalkable =
            |found: &str| Err(invalid(&path, format!("cannot walk a {found} document")));
        if first[0] == b'{' {
            return match Tree::from_json(&bytes) {
                Ok(tree) => Ok(Some(Document::Tree(tree))),
                Err(format::Error::Format { found, .. }) if same_family(&found, TREE_FORMAT) => {
                    unwalkable(&found)
                }
                Err(_) => Ok(None),
            };
        }
        match Header::from_cbor(&bytes) {
            Ok(header) => Ok(Some(Document::Header(header))),
            Err(format::Error::PreCut) => unwalkable("pre-cut cozytensors"),
            Err(format::Error::Format { found, .. }) if same_family(&found, HEADER_FORMAT) => {
                unwalkable(&found)
            }
            Err(_) => Ok(None),
        }
    }

    /// A header's chunks and assets; none when the header is absent.
    fn header_objects(&self, header: &ObjectRef) -> Result<Vec<ObjectRef>> {
        let path = self.path(header)?;
        match fs::read(&path) {
            Ok(bytes) => Header::from_cbor(&bytes)
                .map(|header| header.objects().cloned().collect())
                .map_err(|error| invalid(&path, error)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(vec![]),
            Err(error) => Err(fail("read", &path)(error)),
        }
    }

    /// Every object under `blobs/`: its length and when it was last written or reused.
    fn objects(&self) -> Result<HashMap<String, (u64, SystemTime)>> {
        let mut objects = HashMap::new();
        for top in list(&self.root.join("blobs"))? {
            for shard in list(&top)? {
                for entry in list(&shard)? {
                    let Some(sha256) = entry.file_name().and_then(|n| n.to_str()) else {
                        continue;
                    };
                    let metadata = match fs::symlink_metadata(&entry) {
                        Ok(metadata) if metadata.is_file() && is_digest(sha256) => metadata,
                        Ok(_) => continue,
                        Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                        Err(error) => return Err(fail("stat", &entry)(error)),
                    };
                    let modified = metadata.modified().map_err(fail("stat", &entry))?;
                    objects.insert(sha256.to_owned(), (metadata.len(), modified));
                }
            }
        }
        Ok(objects)
    }

    /// Sweeps what neither a root nor a use keeps and is older than the grace.
    pub fn gc(&self) -> Result<Collected> {
        self.collect(0, Duration::MAX)
    }

    /// Drops the uses unused for `older_than`, least recently used first, until `need` bytes
    /// would be free (`u64::MAX`: every one), then sweeps. Roots are never dropped. Skips while
    /// another process reads metadata to hold it.
    pub fn collect(&self, need: u64, older_than: Duration) -> Result<Collected> {
        let Some(_gc) = retention::try_collect(self)? else {
            return Ok(Collected::default());
        };
        self.collect_locked(need, older_than)
    }

    /// The caller holds the collection fence for the whole mark and sweep.
    pub(super) fn collect_locked(&self, need: u64, older_than: Duration) -> Result<Collected> {
        let cutoff = SystemTime::now()
            .checked_sub(self.grace)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.restore_trash()?;
        let mut marked = self.roots()?;
        let mut uses = self.uses(&marked)?;
        let objects = self.objects()?;
        let mut collected = Collected {
            uses: self.drop_uses(&mut uses, &marked, &objects, need, older_than, cutoff)?,
            ..Collected::default()
        };
        for kept in uses {
            marked.extend(kept.digests);
            marked.insert(kept.name);
        }
        fault::point("gc.marked");
        let doomed = objects
            .iter()
            .filter(|(sha, (_, modified))| *modified < cutoff && !marked.contains(*sha));
        let doomed: Vec<&String> = doomed.map(|(sha, _)| sha).collect();
        fault::point("gc.listed");
        for sha256 in doomed {
            if let Some(length) = self.discard(sha256, cutoff)? {
                collected.objects += 1;
                collected.bytes += length;
                fault::point("gc.swept");
            }
        }
        for folder in list(&self.root.join("snapshots"))? {
            let name = folder.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            let old = fs::symlink_metadata(&folder)
                .is_ok_and(|m| m.is_dir() && m.modified().is_ok_and(|m| m < cutoff));
            if old && is_digest(name) && !marked.contains(name) {
                match fs::remove_dir_all(&folder) {
                    Err(e) if e.kind() != io::ErrorKind::NotFound => {
                        return Err(fail("remove", &folder)(e))
                    }
                    _ => {}
                }
            }
        }
        self.sweep_tmp(cutoff)?;
        Ok(collected)
    }

    /// Removes the uses `need` asks for from the store and from `uses`: their names.
    fn drop_uses(
        &self,
        uses: &mut Vec<Use>,
        roots: &BTreeSet<String>,
        objects: &HashMap<String, (u64, SystemTime)>,
        need: u64,
        older_than: Duration,
        cutoff: SystemTime,
    ) -> Result<Vec<String>> {
        let Some(since) = SystemTime::now().checked_sub(older_than).filter(|_| need > 0) else {
            return Ok(vec![]);
        };
        uses.sort_by(|a, b| (a.used, &a.name).cmp(&(b.used, &b.name)));
        let mut count: HashMap<&str, u32> = HashMap::new();
        for digest in uses.iter().flat_map(|u| &u.digests) {
            *count.entry(digest).or_default() += 1;
        }
        // What a sweep frees: old objects no root keeps, once no use does.
        let size = |sha: &str| match objects.get(sha) {
            Some((length, modified)) if *modified < cutoff && !roots.contains(sha) => *length,
            _ => 0,
        };
        let mut free: u64 = objects.keys().filter(|sha| !count.contains_key(sha.as_str())).map(|sha| size(sha)).sum();
        let mut dropped = BTreeSet::new();
        // A use a root names frees nothing: it stays.
        for u in uses.iter().filter(|u| u.used <= since && !roots.contains(&u.name)) {
            if free >= need {
                break;
            }
            self.remove_record(&self.root.join("uses").join(&u.name))?;
            for digest in &u.digests {
                let n = count.get_mut(digest.as_str()).expect("a use counts what it keeps");
                *n -= 1;
                if *n == 0 {
                    free = free.saturating_add(size(digest));
                }
            }
            dropped.insert(u.name.clone());
        }
        uses.retain(|u| !dropped.contains(&u.name));
        Ok(dropped.into_iter().collect())
    }

    /// Deletes `sha256` unless it became young since it was listed; its length when deleted.
    fn discard(&self, sha256: &str, cutoff: SystemTime) -> Result<Option<u64>> {
        let blob = self.blob(sha256)?;
        // A reader's shared lock keeps it; holding this one keeps readers out until decided.
        let Some(_held) = exclusive(&blob)? else {
            return Ok(None);
        };
        let trash = self.trash(sha256);
        // Windows refuses to move a file someone holds open; it is collected another time.
        if fs::rename(&blob, &trash).is_err() {
            return Ok(None);
        }
        fault::point("gc.trashed");
        // From here a put or reuse may take it back (`young`): gone from the trash is kept.
        let gone = |error: &io::Error| error.kind() == io::ErrorKind::NotFound;
        let metadata = match fs::metadata(&trash) {
            Ok(metadata) => metadata,
            Err(error) if gone(&error) => return Ok(None),
            Err(error) => return Err(fail("stat", &trash)(error)),
        };
        if metadata.modified().map_err(fail("stat", &trash))? >= cutoff {
            return match fs::rename(&trash, &blob) {
                Err(error) if !gone(&error) => Err(fail("restore", &blob)(error)),
                _ => Ok(None),
            };
        }
        match fs::remove_file(&trash) {
            Ok(()) => Ok(Some(metadata.len())),
            Err(error) if gone(&error) => Ok(None),
            Err(error) => Err(fail("remove", &trash)(error)),
        }
    }

    /// Puts back what a GC killed mid-sweep left in `tmp/gc/`.
    fn restore_trash(&self) -> Result<()> {
        let trash = self.root.join("tmp").join("gc");
        fs::create_dir_all(&trash).map_err(fail("create", &trash))?;
        for entry in list(&trash)? {
            let Some(blob) = entry
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|sha| self.blob(sha).ok())
            else {
                continue;
            };
            let shard = blob.parent().expect("a blob has a shard");
            fs::create_dir_all(shard).map_err(fail("create", shard))?;
            match fs::rename(&entry, &blob) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(fail("restore", &blob)(error)),
            }
        }
        Ok(())
    }

    /// Removes the temps of puts and commits that died before the grace. Directories under
    /// `tmp/` belong to the layers that made them.
    fn sweep_tmp(&self, cutoff: SystemTime) -> Result<()> {
        for entry in list(&self.root.join("tmp"))? {
            let Ok(metadata) = fs::symlink_metadata(&entry) else {
                continue;
            };
            if metadata.is_file() && metadata.modified().is_ok_and(|m| m < cutoff) {
                if let Some(_held) = exclusive(&entry)? {
                    let _ = fs::remove_file(&entry);
                }
            }
        }
        Ok(())
    }
}

fn exclusive(path: &std::path::Path) -> Result<Option<File>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(fail("open collection candidate", path)(error)),
    };
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(error)) => {
            Err(fail("lock collection candidate", path)(error))
        }
    }
}
