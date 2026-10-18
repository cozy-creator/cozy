//! GC: mark from refs and pins through manifests and headers to chunks and files, then sweep
//! what is unmarked and older than the grace. When to collect, and how much, is the caller's
//! policy (TensorD: TTL, low-disk LRU).
//!
//! A swept object is first renamed into `tmp/gc/`, then deleted only if it is still old: a put
//! or reuse that raced the sweep made it young, and it goes back. A GC killed in between leaves
//! it there, and the next GC puts it back before deciding again.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read};
use std::time::{Duration, SystemTime};

use super::{fail, fault, invalid, is_digest, now_secs, same_family, Result, Store};
use crate::format::{self, Header, Manifest, ObjectRef, MANIFEST_FORMAT};

/// Documents bigger than this are never read as manifests.
const MAX_MANIFEST: u64 = 64 << 20;

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
    /// `root`, and when it is a manifest, its files and its headers' chunks; each once.
    pub fn closure(&self, root: &ObjectRef) -> Result<Vec<ObjectRef>> {
        let mut reach = BTreeSet::from([root.clone()]);
        if let Some(manifest) = self.manifest(root)? {
            for (path, file) in manifest.files {
                if path.ends_with(".cozytensors") {
                    reach.extend(self.chunks(&file)?);
                }
                reach.insert(file);
            }
        }
        Ok(reach.into_iter().collect())
    }

    /// The manifest `object` is; `None` for any other object or an absent one. A manifest of
    /// another major cannot be walked, so GC stops rather than sweep what it reaches.
    fn manifest(&self, object: &ObjectRef) -> Result<Option<Manifest>> {
        let path = self.path(object)?;
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(fail("open", &path)(error)),
        };
        let length = file.metadata().map_err(fail("stat", &path))?.len();
        let mut first = [0u8];
        if length > MAX_MANIFEST || file.read_exact(&mut first).is_err() || first[0] != b'{' {
            return Ok(None);
        }
        let mut bytes = first.to_vec();
        file.read_to_end(&mut bytes).map_err(fail("read", &path))?;
        match Manifest::from_json(&bytes) {
            Ok(manifest) => Ok(Some(manifest)),
            Err(format::Error::Format { found, .. }) if same_family(&found, MANIFEST_FORMAT) => {
                Err(invalid(&path, format!("cannot walk a {found} manifest")))
            }
            Err(_) => Ok(None),
        }
    }

    /// The chunks a header names; none when the header is absent.
    fn chunks(&self, header: &ObjectRef) -> Result<Vec<ObjectRef>> {
        let path = self.path(header)?;
        match fs::read(&path) {
            Ok(bytes) => Header::from_cbor(&bytes)
                .map(|header| header.chunks)
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
            for manifest in repo.roots() {
                refs.extend(reach(manifest)?);
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
            .filter(|pin| pin.last_use <= unused_since)
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

    /// Deletes `sha256` unless it became young since it was listed; its size when deleted.
    fn discard(&self, sha256: &str, cutoff: SystemTime) -> Result<Option<u64>> {
        let blob = self.blob(sha256)?;
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
                let _ = fs::remove_file(&entry);
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
