//! Origin custody: a checkpoint present here only in part, whose absent objects the origin it
//! came from holds. A part ensure (a job's model input made present by its structure, a
//! served slot's components first) attests its manifest; a derivation that inherits absent
//! objects attests its output. Nothing here reads an absent object: a reader fetches it back
//! through the ordinary verified pull, and the store's walks (GC, storage, derivation) accept
//! that it is elsewhere. A record ends with its manifest.

use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::canon::{Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{ObjectRef, Plain};
use crate::store::Store;

const FORMAT: &str = "tensorfs.origin/1";
const EVIDENCE_MAX_BYTES: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Origin {
    manifest: ObjectRef,
    evidence: String,
}

impl Plain for Origin {
    const MAX_BYTES: usize = 4096;

    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("Origin", value)?;
        if fields.req_str("format")? != FORMAT {
            return refuse(Code::WRONG_TYPE, "origin custody has an unknown format");
        }
        let evidence = fields.req_str("evidence")?.to_string();
        let manifest = ObjectRef::from_value("Origin.manifest", fields.req("manifest")?)?;
        fields.done()?;
        Ok(Self { manifest, evidence })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("evidence", Value::str(self.evidence.clone())),
            ("format", Value::str(FORMAT)),
            ("manifest", self.manifest.to_value()),
        ])
    }
}

fn io(what: &str, error: std::io::Error) -> Refusal {
    crate::store::classify_io(format!("origin custody {what}"), error)
}

fn dir(store_root: &Path) -> PathBuf {
    store_root.join("origins")
}

/// Record that `evidence` (opaque to TensorFS: the caller's name for the origin) holds every
/// object of `manifest`'s closure this store lacks. The manifest must be here.
pub fn attest(store: &Store, manifest: &ObjectRef, evidence: &str) -> Result<()> {
    if evidence.is_empty() || evidence.len() > EVIDENCE_MAX_BYTES {
        return refuse(Code::KEY_GRAMMAR, "origin evidence is 1..=1024 bytes");
    }
    store.read_manifest(manifest)?;
    let dir = dir(store.root());
    fs::create_dir_all(&dir).map_err(|error| io("create", error))?;
    let bytes = Origin { manifest: manifest.clone(), evidence: evidence.to_string() }.canonical_bytes();
    let path = dir.join(&manifest.sha256);
    let temporary = dir.join(format!(".{}.{}", manifest.sha256, crate::meta::now_nanos_unique()));
    let written = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| io("create", error))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| io("write", error))?;
        fs::rename(&temporary, &path).map_err(|error| io("rename", error))?;
        crate::store::fsync_dir(&dir)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written
}

/// The evidence attested for `manifest`, if any.
pub fn evidence(store_root: &Path, manifest: &ObjectRef) -> Result<Option<String>> {
    Ok(read(&dir(store_root).join(&manifest.sha256))?
        .filter(|origin| origin.manifest == *manifest)
        .map(|origin| origin.evidence))
}

fn read(path: &Path) -> Result<Option<Origin>> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io("open", error)),
    };
    let mut bytes = Vec::new();
    file.take(Origin::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io("read", error))?;
    Origin::parse(&bytes).map(Some)
}

/// Every absent object of an attested manifest still here: what its origin holds for it.
pub fn custodied(store_root: &Path) -> Result<BTreeSet<String>> {
    let entries = match fs::read_dir(dir(store_root)) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(error) => return Err(io("list", error)),
    };
    let mut store = None;
    let mut out = BTreeSet::new();
    for entry in entries {
        let path = entry.map_err(|error| io("list", error))?.path();
        if path.file_name().is_some_and(|name| name.to_string_lossy().starts_with('.')) {
            continue;
        }
        let Some(origin) = read(&path)? else { continue };
        let store = match &store {
            Some(store) => store,
            None => store.insert(Store::open(store_root)?),
        };
        let manifest = match store.read_manifest(&origin.manifest) {
            Ok(manifest) => manifest,
            Err(error) if error.code == Code::OBJECT_ABSENT => continue,
            Err(error) => return Err(error),
        };
        let walk = match crate::checkpoint::walk(store, &manifest) {
            Ok(walk) => walk,
            Err(error) if error.code == Code::OBJECT_ABSENT => continue,
            Err(error) => return Err(error),
        };
        for object in walk.distinct() {
            if !store.contains(&object.sha256) {
                out.insert(object.sha256.clone());
            }
        }
    }
    Ok(out)
}

/// Drop the records of manifests GC removed.
pub(crate) fn prune(store: &Store) -> Result<()> {
    let entries = match fs::read_dir(dir(store.root())) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(io("list", error)),
    };
    for entry in entries {
        let path = entry.map_err(|error| io("list", error))?.path();
        let name = path.file_name().map(|name| name.to_string_lossy().to_string()).unwrap_or_default();
        if !name.starts_with('.') && !store.manifest_path(&name).is_file() {
            fs::remove_file(&path).map_err(|error| io("remove", error))?;
        }
    }
    Ok(())
}
