//! Origin custody: a checkpoint present here only in part, whose absent objects its hub
//! holds. The fetcher attests a checkpoint once its structure is here
//! ([`crate::fetch_v1::Fetcher`]); a derivation that inherits absent objects attests its output
//! with its sources' origin. Nothing here reads an absent object: a reader fetches it through
//! the fetcher, from the [`Origin`] its record names.
//!
//! A record is a cache root held in part: GC keeps what of its checkpoint is present until
//! the record goes, by its TTL or at low disk, where its present objects are evicted first
//! (they can be fetched again) and its structure stays. A record ends with its manifest.
//!
//! `origins/<manifest>` names the origin. `origins/<manifest>.objects` indexes every object
//! the checkpoint reaches, written once at attest, so custody never walks a header again.

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::canon::{Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch_v1::Origin;
use crate::ids::{ObjectRef, Plain};
use crate::storage::{self, HeldKey};
use crate::store::Store;

const FORMAT: &str = "tensorfs.origin/2";
const INDEX: &str = "objects";

#[derive(Debug, Clone, PartialEq, Eq)]
struct Record {
    manifest: ObjectRef,
    origin: Origin,
}

impl Plain for Record {
    const MAX_BYTES: usize = 8192;

    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("Origin", value)?;
        if fields.req_str("format")? != FORMAT {
            return refuse(Code::WRONG_TYPE, "origin custody has an unknown format");
        }
        let hub = fields.req_str("hub")?.to_string();
        let lane = fields.req_str("lane")?.to_string();
        let manifest = ObjectRef::from_value("Origin.manifest", fields.req("manifest")?)?;
        let source = ObjectRef::from_value("Origin.origin_manifest", fields.req("origin_manifest")?)?;
        let refspec = fields.req_str("refspec")?.to_string();
        fields.done()?;
        Ok(Record {
            manifest,
            origin: Origin { hub, refspec, lane, manifest: source },
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("format", Value::str(FORMAT)),
            ("hub", Value::str(self.origin.hub.clone())),
            ("lane", Value::str(self.origin.lane.clone())),
            ("manifest", self.manifest.to_value()),
            ("origin_manifest", self.origin.manifest.to_value()),
            ("refspec", Value::str(self.origin.refspec.clone())),
        ])
    }
}

fn io(what: &str, error: std::io::Error) -> Refusal {
    crate::store::classify_io(format!("origin custody {what}"), error)
}

fn dir(store_root: &Path) -> PathBuf {
    store_root.join("origins")
}

fn index_path(store_root: &Path, manifest: &str) -> PathBuf {
    dir(store_root).join(format!("{manifest}.{INDEX}"))
}

/// Write `bytes` at `path` whole or not at all.
fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap();
    let name = path.file_name().unwrap().to_string_lossy();
    let temporary = parent.join(format!(".{name}.{}", crate::meta::now_nanos_unique()));
    let written = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| io("create", error))?;
        file.write_all(bytes).and_then(|()| file.sync_all()).map_err(|error| io("write", error))?;
        fs::rename(&temporary, path).map_err(|error| io("rename", error))?;
        crate::store::fsync_dir(parent)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written
}

/// Record that `origin` holds every object of `manifest`'s checkpoint this Store lacks, and
/// index the objects it reaches. The manifest, and its header when it has one, must be here.
pub fn attest(store: &Store, manifest: &ObjectRef, origin: &Origin) -> Result<()> {
    let document = store.read_manifest(manifest)?;
    let walk = crate::checkpoint::walk(store, &document)?;
    let index: String = (walk.distinct().into_iter())
        .map(|object| format!("{} {}\n", object.sha256, object.length))
        .collect();
    let dir = dir(store.root());
    fs::create_dir_all(&dir).map_err(|error| io("create", error))?;
    write(&index_path(store.root(), &manifest.sha256), index.as_bytes())?;
    let record = Record { manifest: manifest.clone(), origin: origin.clone() };
    write(&dir.join(&manifest.sha256), &record.canonical_bytes())
}

/// The origin attested for `manifest`, if any.
pub fn of(store_root: &Path, manifest: &ObjectRef) -> Result<Option<Origin>> {
    Ok(read(&dir(store_root).join(&manifest.sha256))?
        .filter(|record| record.manifest == *manifest)
        .map(|record| record.origin))
}

fn read(path: &Path) -> Result<Option<Record>> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io("open", error)),
    };
    let mut bytes = Vec::new();
    file.take(Record::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io("read", error))?;
    // A record this build cannot read roots nothing: what it named is fetched again.
    Ok(Record::parse(&bytes).ok())
}

/// The objects attest indexed for `manifest`.
fn index(store_root: &Path, manifest: &str) -> Result<Vec<ObjectRef>> {
    let lines = match fs::read_to_string(index_path(store_root, manifest)) {
        Ok(lines) => lines,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(error) => return Err(io("read index", error)),
    };
    Ok(lines
        .lines()
        .filter_map(|line| line.split_once(' '))
        .filter_map(|(sha256, length)| Some(ObjectRef { sha256: sha256.to_string(), length: length.parse().ok()? }))
        .collect())
}

/// Every record's file, by its manifest's bare digest.
fn records(store_root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let entries = match fs::read_dir(dir(store_root)) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(error) => return Err(io("list", error)),
    };
    let mut out = vec![];
    for entry in entries {
        let path = entry.map_err(|error| io("list", error))?.path();
        let name = path.file_name().map(|name| name.to_string_lossy().to_string()).unwrap_or_default();
        if !name.starts_with('.') && !name.contains('.') {
            out.push((name, path));
        }
    }
    Ok(out)
}

/// Every attested manifest still here: each a checkpoint its origin keeps whole.
pub(crate) fn manifests(store: &Store) -> Result<Vec<ObjectRef>> {
    let mut out = vec![];
    for (_, path) in records(store.root())? {
        if let Some(record) = read(&path)?.filter(|r| store.manifest_path(&r.manifest.sha256).is_file()) {
            out.push(record.manifest);
        }
    }
    Ok(out)
}

/// Every absent object of an attested manifest still here: what its origin holds for it.
pub fn custodied(store_root: &Path) -> Result<BTreeSet<String>> {
    let mut store = None;
    let mut out = BTreeSet::new();
    for (manifest, _) in records(store_root)? {
        let store = match &store {
            Some(store) => store,
            None => store.insert(Store::open(store_root)?),
        };
        if !store.manifest_path(&manifest).is_file() {
            continue;
        }
        out.extend((index(store_root, &manifest)?.into_iter()).filter(|o| !store.contains(&o.sha256)).map(|o| o.sha256));
    }
    Ok(out)
}

/// GC's roots: every attested manifest here keeps its present closure.
pub(crate) fn held(store: &Store) -> Result<Vec<HeldKey>> {
    let mut out = vec![];
    for manifest in manifests(store)? {
        out.push(HeldKey {
            key: storage::manifest_key(&manifest.sha256)?,
            kind: "manifest".into(),
            length: manifest.length,
            sha256: manifest.sha256,
        });
    }
    Ok(out)
}

/// The structure of an attested checkpoint: its manifest, header and assets, which eviction
/// keeps.
pub(crate) fn structure(store: &Store, manifest: &ObjectRef) -> Result<Vec<ObjectRef>> {
    let document = store.read_manifest(manifest)?;
    let mut out = vec![manifest.clone()];
    if let Some(header) = document.header() {
        out.push(header.clone());
        if let Ok(header) = crate::checkpoint::load_header(store, header) {
            out.extend(header.assets.iter().flat_map(|(_, asset)| asset.segments.clone()));
        }
    }
    Ok(out)
}

/// The objects of an attested checkpoint, as indexed.
pub(crate) fn objects(store: &Store, manifest: &ObjectRef) -> Result<Vec<ObjectRef>> {
    index(store.root(), &manifest.sha256)
}

/// The checkpoint was used now: its TTL starts again.
pub(crate) fn touch(store: &Store, manifest: &ObjectRef) {
    if let Ok(file) = OpenOptions::new().write(true).open(dir(store.root()).join(&manifest.sha256)) {
        let _ = file.set_modified(SystemTime::now());
    }
}

/// When the checkpoint was attested or last used.
pub(crate) fn used(store: &Store, manifest: &ObjectRef) -> Option<SystemTime> {
    fs::metadata(dir(store.root()).join(&manifest.sha256)).and_then(|m| m.modified()).ok()
}

/// The record goes: the checkpoint is whole and recorded, or its TTL ran out.
pub(crate) fn retire(store: &Store, manifest: &ObjectRef) -> Result<()> {
    for path in [dir(store.root()).join(&manifest.sha256), index_path(store.root(), &manifest.sha256)] {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io("remove", error)),
        }
    }
    Ok(())
}

/// Drop the records of manifests GC removed.
pub(crate) fn prune(store: &Store) -> Result<()> {
    for (manifest, path) in records(store.root())? {
        if !store.manifest_path(&manifest).is_file() {
            fs::remove_file(&path).map_err(|error| io("remove", error))?;
            let _ = fs::remove_file(index_path(store.root(), &manifest));
        }
    }
    Ok(())
}
