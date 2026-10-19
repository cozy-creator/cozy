//! GC: mark from refs and pins through trees and headers to blobs, chunks and assets, then sweep
//! what is unmarked and older than the grace. Raw stores keep caller policy; explicitly enabled
//! library caches add TTL and pressure policy without treating local refs as disposable.
//!
//! A swept object is first renamed into `tmp/gc/`, then deleted only if it is still old: a put
//! or reuse that raced the sweep made it young, and it goes back. A GC killed in between leaves
//! it there, and the next GC puts it back before deciding again.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use platform::durable::{self, FileId};

use super::cache::Need;
use super::retention;

use super::{fail, fault, invalid, is_digest, now_secs, same_family, Result, Store};
use crate::format::{self, variant_of, Header, ObjectRef, Tree, HEADER_FORMAT, TREE_FORMAT};

/// Documents bigger than this are never read as trees or headers.
const MAX_DOCUMENT: u64 = 256 << 20;

/// What one sweep took.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Collected {
    /// Pins dropped to meet `need`.
    pub pins: Vec<String>,
    pub objects: u64,
    pub bytes: u64,
}

/// Distinct bytes on disk. A pin's `marginal` bytes are those no ref and no other pin reaches:
/// what dropping only that pin frees.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub objects: u64,
    pub bytes: u64,
    pub refs: u64,
    pub unreferenced: u64,
    pub pins: BTreeMap<String, PinUsage>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PinUsage {
    pub bytes: u64,
    pub marginal: u64,
    pub last_use: u64,
    pub owner: Option<String>,
}

enum Document {
    Tree(Tree),
    Header(Header),
}

struct Object {
    length: u64,
    modified: SystemTime,
}

struct Pinned {
    id: String,
    last_use: u64,
    owner: Option<String>,
    reach: BTreeSet<String>,
}

/// Every object on disk, what refs reach, what each pin reaches, and how many pins reach each.
struct Census {
    objects: HashMap<String, Object>,
    refs: HashSet<String>,
    pins: Vec<Pinned>,
    count: HashMap<String, u32>,
}

impl Census {
    fn size(&self, sha256: &str) -> u64 {
        self.objects.get(sha256).map_or(0, |object| object.length)
    }
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

    fn census(&self) -> Result<Census> {
        let mut walked: HashMap<ObjectRef, Vec<String>> = HashMap::new();
        let mut reach = |root: &ObjectRef| -> Result<Vec<String>> {
            if let Some(known) = walked.get(root) {
                return Ok(known.clone());
            }
            let found: Vec<String> = self.closure(root)?.into_iter().map(|o| o.sha256).collect();
            walked.insert(root.clone(), found.clone());
            Ok(found)
        };
        let mut refs = HashSet::new();
        for repo in self.refs()? {
            for tree in repo.roots() {
                if self.cached_ref(&repo)? {
                    // Keep offline resolution metadata while allowing cached payload to expire.
                    refs.insert(tree.sha256.clone());
                    if let Some(Document::Tree(tree)) = self.document(tree)? {
                        refs.extend(
                            tree.files
                                .into_iter()
                                .filter_map(|(path, file)| variant_of(&path).map(|_| file.sha256)),
                        );
                    }
                } else {
                    refs.extend(reach(tree)?);
                }
            }
        }
        let mut pins = Vec::new();
        let mut count: HashMap<String, u32> = HashMap::new();
        for (id, pin) in self.pins()? {
            let mut objects = BTreeSet::new();
            for root in &pin.roots {
                objects.extend(reach(root)?);
            }
            for object in &objects {
                *count.entry(object.clone()).or_default() += 1;
            }
            pins.push(Pinned {
                id,
                last_use: pin.last_use,
                owner: pin.owner,
                reach: objects,
            });
        }
        Ok(Census {
            objects: self.objects()?,
            refs,
            pins,
            count,
        })
    }

    /// Every object under `blobs/`, by digest.
    fn objects(&self) -> Result<HashMap<String, Object>> {
        let mut objects = HashMap::new();
        let blobs = self.root.join("blobs");
        for top in list(&blobs)? {
            for shard in list(&top)? {
                for entry in list(&shard)? {
                    let Some(sha256) = entry.file_name().and_then(|n| n.to_str()) else {
                        continue;
                    };
                    if !is_digest(sha256) {
                        continue;
                    }
                    let metadata = match fs::symlink_metadata(&entry) {
                        Ok(metadata) if metadata.is_file() => metadata,
                        Ok(_) => continue,
                        Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                        Err(error) => return Err(fail("stat", &entry)(error)),
                    };
                    let modified = metadata.modified().map_err(fail("stat", &entry))?;
                    objects.insert(
                        sha256.to_owned(),
                        Object {
                            length: metadata.len(),
                            modified,
                        },
                    );
                }
            }
        }
        Ok(objects)
    }

    pub fn usage(&self) -> Result<Usage> {
        let census = self.census()?;
        let mut usage = Usage {
            objects: census.objects.len() as u64,
            bytes: census.objects.values().map(|o| o.length).sum(),
            refs: census.refs.iter().map(|sha| census.size(sha)).sum(),
            ..Usage::default()
        };
        usage.unreferenced = census
            .objects
            .iter()
            .filter(|(sha, _)| !census.refs.contains(*sha) && !census.count.contains_key(*sha))
            .map(|(_, object)| object.length)
            .sum();
        for pin in &census.pins {
            let only_here = |sha: &&String| !census.refs.contains(*sha) && census.count[*sha] == 1;
            usage.pins.insert(
                pin.id.clone(),
                PinUsage {
                    bytes: pin.reach.iter().map(|sha| census.size(sha)).sum(),
                    marginal: pin
                        .reach
                        .iter()
                        .filter(only_here)
                        .map(|sha| census.size(sha))
                        .sum(),
                    last_use: pin.last_use,
                    owner: pin.owner.clone(),
                },
            );
        }
        Ok(usage)
    }

    /// Sweeps every unreferenced object older than the grace.
    pub fn gc(&self) -> Result<Collected> {
        self.collect(0, Duration::MAX)
    }

    /// Drops pins unused for `older_than`, least recently used first, until `need` bytes would
    /// be free, then sweeps. Refs are never dropped.
    pub fn collect(&self, need: u64, older_than: Duration) -> Result<Collected> {
        let Some(_guard) = retention::try_collect(self)? else {
            return Ok(Collected::default());
        };
        self.collect_locked(need, older_than)
    }

    /// The caller owns the exclusive acquisition fence for the complete mark/sweep.
    pub(super) fn collect_locked(&self, need: u64, older_than: Duration) -> Result<Collected> {
        retention::reap(self)?;
        let started = SystemTime::now();
        let cutoff = started
            .checked_sub(self.grace)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        self.restore_trash()?;
        let pins = if need > 0 {
            self.drop_pins(need, older_than, cutoff)?
        } else {
            vec![]
        };
        let census = self.census()?;
        fault::point("gc.marked");
        let doomed: Vec<String> = census
            .objects
            .iter()
            .filter(|(sha, object)| {
                object.modified < cutoff
                    && !census.refs.contains(*sha)
                    && !census.count.contains_key(*sha)
            })
            .map(|(sha, _)| sha.clone())
            .collect();
        fault::point("gc.listed");
        let mut collected = Collected {
            pins,
            ..Collected::default()
        };
        for sha256 in doomed {
            if let Some(length) = self.discard(&sha256, cutoff)? {
                collected.objects += 1;
                collected.bytes += length;
                fault::point("gc.swept");
            }
        }
        self.sweep_tmp(cutoff)?;
        Ok(collected)
    }

    fn drop_pins(
        &self,
        need: u64,
        older_than: Duration,
        cutoff: SystemTime,
    ) -> Result<Vec<String>> {
        let Some(unused_since) = now_secs().checked_sub(older_than.as_secs()) else {
            return Ok(vec![]);
        };
        let mut census = self.census()?;
        let mut free: u64 = census
            .objects
            .iter()
            .filter(|(sha, object)| {
                object.modified < cutoff
                    && !census.refs.contains(*sha)
                    && !census.count.contains_key(*sha)
            })
            .map(|(_, object)| object.length)
            .sum();
        let mut stale: Vec<Pinned> = std::mem::take(&mut census.pins)
            .into_iter()
            .filter(|pin| !pin.id.starts_with("hold-") && pin.last_use <= unused_since)
            .collect();
        stale.sort_by(|a, b| (a.last_use, &a.id).cmp(&(b.last_use, &b.id)));
        let mut dropped = Vec::new();
        for pin in stale {
            if free >= need {
                break;
            }
            if !self.unpin_unused(&pin.id, pin.last_use)? {
                continue;
            }
            for sha in &pin.reach {
                if census.refs.contains(sha) {
                    continue;
                }
                let count = census
                    .count
                    .get_mut(sha)
                    .expect("a pin counts what it reaches");
                *count -= 1;
                if *count == 0 {
                    free += census.size(sha);
                }
            }
            dropped.push(pin.id);
        }
        Ok(dropped)
    }

    /// Choose a covering LRU plan before dropping any fresh cache root. Actual disk space
    /// is sampled again by the caller; namespace bytes are never admission credit.
    pub(super) fn cache_pressure(&self, need: Need) -> Result<()> {
        retention::reap(self)?;
        let census = self.census()?;
        let cutoff = SystemTime::now()
            .checked_sub(self.grace)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let mut pins: Vec<_> = census
            .pins
            .iter()
            .filter(|pin| pin.id.starts_with("cache-"))
            .collect();
        pins.sort_by_key(|pin| (pin.last_use, &pin.id));
        let mut snapshots = self.cache_snapshots()?;
        snapshots.sort_by_key(|snapshot| snapshot.used);
        let mut count = census.count.clone();
        let mut paths = Vec::new();
        let mut selected_pins = Vec::new();
        let mut selected_snapshots = Vec::new();
        let open = open_files();
        let filesystem = FileId::at(&self.root)
            .map_err(super::platform("identify cache filesystem", &self.root))?
            .dev;
        let objects = |count: &HashMap<String, u32>| -> Result<Vec<PathBuf>> {
            census
                .objects
                .iter()
                .filter(|(sha, object)| {
                    object.modified < cutoff
                        && !census.refs.contains(*sha)
                        && count.get(*sha).copied().unwrap_or(0) == 0
                })
                .map(|(sha, _)| self.blob(sha))
                .collect()
        };
        let covered = |paths: &[PathBuf], count: &HashMap<String, u32>| -> Result<bool> {
            let mut all = paths.to_vec();
            all.extend(objects(count)?);
            let (bytes, inodes) = releasable(&all, filesystem, &open)?;
            Ok(need.covered(bytes, inodes))
        };
        // Snapshot copies and root pins share an LRU, and shared chunks accrue only once.
        let mut folders = snapshots.into_iter().peekable();
        let mut pins = pins.into_iter().peekable();
        while !covered(&paths, &count)? {
            let folder_first = match (folders.peek(), pins.peek()) {
                (Some(folder), Some(pin)) => folder.used <= pin.last_use,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => return Ok(()), // No covering plan: preserve fresh cache roots.
            };
            if folder_first {
                let folder = folders.next().expect("peeked folder");
                paths.extend(folder.files()?);
                selected_snapshots.push(folder);
            } else {
                let pin = pins.next().expect("peeked pin");
                for sha in &pin.reach {
                    *count.get_mut(sha).expect("a pin counts what it reaches") -= 1;
                }
                paths.push(self.root.join("pins").join(format!("{}.json", pin.id)));
                selected_pins.push(pin);
            }
        }
        for pin in selected_pins {
            self.unpin_unused(&pin.id, pin.last_use)?;
        }
        for folder in selected_snapshots {
            folder.remove(self)?;
        }
        self.collect_locked(0, Duration::MAX)?;
        Ok(())
    }

    /// Deletes `sha256` unless it became young since it was listed; its size when deleted.
    fn discard(&self, sha256: &str, cutoff: SystemTime) -> Result<Option<u64>> {
        let blob = self.blob(sha256)?;
        let open = open_files();
        let Some(held) = exclusive(&blob)? else {
            return Ok(None);
        };
        let id = FileId::of(&held).map_err(super::platform("identify GC object", &blob))?;
        let external_open = open.contains(&id);
        let trash = self.trash(sha256);
        // Windows refuses to move a file someone holds open; it is collected another time.
        if fs::rename(&blob, &trash).is_err() {
            return Ok(None);
        }
        fault::point("gc.trashed");
        // From here a put or reuse may take it back (`hold`): gone from the trash is kept.
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
        let freed = if !external_open
            && durable::links(&trash).map_err(super::platform("count GC links", &trash))? == 1
        {
            durable::allocated(&metadata)
        } else {
            0
        };
        match fs::remove_file(&trash) {
            Ok(()) => Ok(Some(freed)),
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

fn list(dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(vec![]),
        Err(error) => return Err(fail("list", dir)(error)),
    };
    entries
        .map(|entry| entry.map(|e| e.path()).map_err(fail("list", dir)))
        .collect()
}

/// A process-held descriptor is not free space. Cross-process native object readers also
/// take shared file locks, which the exclusive probe below observes.
fn open_files() -> HashSet<FileId> {
    let Ok(entries) = fs::read_dir("/proc/self/fd") else {
        return HashSet::new();
    };
    entries
        .flatten()
        .filter_map(|entry| FileId::at(&entry.path()).ok())
        .collect()
}

fn exclusive(path: &Path) -> Result<Option<File>> {
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

/// Count only inodes for which this plan removes every hard link on this filesystem.
/// Shared cache aliases are grouped, sparse allocation is respected, and open/locked files
/// contribute no capacity. A conservative estimate can refuse admission, never over-admit.
fn releasable(paths: &[PathBuf], filesystem: u64, open: &HashSet<FileId>) -> Result<(u64, u64)> {
    struct Inode {
        paths: u64,
        links: u64,
        bytes: u64,
        blocked: bool,
    }
    let mut inodes: HashMap<FileId, Inode> = HashMap::new();
    let mut unique = HashSet::new();
    for path in paths {
        if !unique.insert(path) {
            continue;
        }
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(fail("stat collection candidate", path)(error)),
        };
        if metadata.file_type().is_symlink() {
            continue;
        }
        let id =
            FileId::at(path).map_err(super::platform("identify collection candidate", path))?;
        if id.dev != filesystem {
            continue;
        }
        let links = if metadata.is_dir() {
            1
        } else {
            durable::links(path).map_err(super::platform("count collection links", path))?
        };
        let blocked = open.contains(&id) || (!metadata.is_dir() && exclusive(path)?.is_none());
        let inode = inodes.entry(id).or_insert(Inode {
            paths: 0,
            links,
            bytes: durable::allocated(&metadata),
            blocked: false,
        });
        inode.paths += 1;
        inode.blocked |= blocked;
    }
    Ok(inodes
        .values()
        .filter(|inode| !inode.blocked && inode.paths >= inode.links)
        .fold((0, 0), |(bytes, count), inode| {
            (bytes.saturating_add(inode.bytes), count + 1)
        }))
}
