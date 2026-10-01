//! Keyed cache roots: ordinary-file trees a caller names by (space, content key), lists by
//! creation time and drops without a tombstone. A root keeps its manifest closure live for
//! GC; when to drop one is the caller's policy.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use fs2::FileExt;

use crate::canon::{Fields, Value};
use crate::catalog::WriterGuard;
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{hex64, Doc, ObjectRef, Plain};
use crate::manifest::Manifest;
use crate::source_artifact::{import_member, import_paths};
use crate::storage::HeldKey;
use crate::store::{Fault, Store, O_NOFOLLOW};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyedRoot {
    pub space: String,
    pub key: String,
    pub manifest: ObjectRef,
    /// Manifest length plus the length of each distinct member object.
    pub bytes: u64,
    pub created_unix_ms: u64,
    /// The newest TensorFS that wrote this root.
    pub tensorfs: Option<String>,
}

impl Plain for KeyedRoot {
    const MAX_BYTES: usize = 4096;

    fn from_value(value: &Value) -> Result<Self> {
        let mut f = Fields::new("KeyedRoot", value)?;
        let space = f.req_str("space")?.to_string();
        let key = f.req_str("key")?.to_string();
        let manifest = ObjectRef::from_value("keyed root manifest", f.req("manifest")?)?;
        let bytes = f.req_uint("bytes")?;
        let created_unix_ms = f.req_uint("created_unix_ms")?;
        let tensorfs = f
            .opt("tensorfs")
            .map(|v| crate::canon::as_str("KeyedRoot", "tensorfs", v).map(str::to_string))
            .transpose()?;
        f.done_written_by(tensorfs.as_deref())?;
        Ok(Self {
            space,
            key,
            manifest,
            bytes,
            created_unix_ms,
            tensorfs,
        })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("space", Value::str(&self.space)),
            ("key", Value::str(&self.key)),
            ("manifest", self.manifest.to_value()),
            ("bytes", Value::uint(self.bytes)),
            ("created_unix_ms", Value::uint(self.created_unix_ms)),
        ];
        if let Some(tensorfs) = &self.tensorfs {
            fields.push(("tensorfs", Value::str(tensorfs.clone())));
        }
        Value::obj(fields)
    }
}

fn io(e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("keyed root: {e}"),
    }
}

fn directory(store: &Store) -> PathBuf {
    store.root().join("roots/keyed")
}

/// `[a-z0-9][a-z0-9._-]{0,63}`: one path component, never `.` or `..`.
fn space_dir(store: &Store, space: &str) -> Result<PathBuf> {
    let b = space.as_bytes();
    let ok = |c: &u8| c.is_ascii_lowercase() || c.is_ascii_digit();
    if b.is_empty() || b.len() > 64 || !ok(&b[0]) || !b.iter().all(|c| ok(c) || b"._-".contains(c))
    {
        return refuse(
            Code::KEY_GRAMMAR,
            format!("keyed root space {space:?} must match [a-z0-9][a-z0-9._-]{{0,63}}"),
        );
    }
    Ok(directory(store).join(space))
}

fn path(store: &Store, space: &str, key: &str) -> Result<PathBuf> {
    hex64("keyed root key", key)?;
    Ok(space_dir(store, space)?.join(format!("{key}.json")))
}

pub fn get(store: &Store, space: &str, key: &str) -> Result<Option<KeyedRoot>> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path(store, space, key)?)
    {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(e)),
    };
    let mut bytes = Vec::new();
    file.take(KeyedRoot::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    let root = KeyedRoot::parse(&bytes)?;
    if root.space != space || root.key != key {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "keyed root path names another key",
        );
    }
    Ok(Some(root))
}

/// Every root in `space`, oldest created first.
pub fn list(store: &Store, space: &str) -> Result<Vec<KeyedRoot>> {
    let entries = match fs::read_dir(space_dir(store, space)?) {
        Ok(entries) => entries,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(io(e)),
    };
    let mut roots = vec![];
    for entry in entries {
        let name = entry.map_err(io)?.file_name();
        let Some(key) = (name.to_str())
            .and_then(|n| n.strip_suffix(".json"))
            .filter(|k| hex64("keyed root key", k).is_ok())
        else {
            continue;
        };
        // A root dropped since the directory was read is simply gone.
        if let Some(root) = get(store, space, key)? {
            roots.push(root);
        }
    }
    roots.sort_by(|a, b| (a.created_unix_ms, &a.key).cmp(&(b.created_unix_ms, &b.key)));
    Ok(roots)
}

/// Import an exact ordinary-file tree and root it at (space, key). The same tree again
/// returns the standing root unchanged; a different tree refuses `TRANSACTION_CONFLICT`.
pub fn put(
    store: &Store,
    space: &str,
    key: &str,
    tree: &Manifest,
    files: &[(String, PathBuf)],
) -> Result<KeyedRoot> {
    let path = path(store, space, key)?;
    let paths = import_paths(tree, files)?;
    let manifest = tree.object_ref();
    let _writer = WriterGuard::acquire(store.root())?;
    let dir = path.parent().expect("root path has a space directory");
    fs::create_dir_all(dir).map_err(io)?;
    let lock = File::open(dir).map_err(io)?;
    lock.lock_exclusive().map_err(io)?;
    if let Some(root) = get(store, space, key)? {
        if root.manifest != manifest {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "keyed root already names another tree",
            );
        }
        return Ok(root);
    }
    let mut members = BTreeMap::new();
    for (member, entry) in tree.entries() {
        let object = entry.blob();
        import_member(store, object, paths[member.as_str()], &Fault::default())?;
        members.insert(&object.sha256, object.length);
    }
    store.put_manifest(tree)?;
    let root = KeyedRoot {
        space: space.into(),
        key: key.into(),
        bytes: manifest.length + members.values().sum::<u64>(),
        manifest,
        created_unix_ms: (crate::meta::now_nanos_unique() / 1_000_000) as u64,
        tensorfs: Some(crate::VERSION.into()),
    };
    let temporary = path.with_extension(format!("tmp-{}", crate::meta::now_nanos_unique()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io)?;
        file.write_all(&root.canonical_bytes())
            .and_then(|()| file.sync_all())
            .map_err(io)?;
        fs::rename(&temporary, &path).map_err(io)?;
        crate::store::fsync_entry_dir(store.root(), dir)
    })();
    let _ = fs::remove_file(temporary);
    result.map(|()| root)
}

/// Remove a root, leaving nothing behind; its bytes are garbage to the next GC.
pub fn remove(store: &Store, space: &str, key: &str) -> Result<bool> {
    let path = path(store, space, key)?;
    let _writer = WriterGuard::acquire(store.root())?;
    match fs::remove_file(&path) {
        Ok(()) => {
            crate::store::fsync_entry_dir(store.root(), path.parent().unwrap()).map(|()| true)
        }
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(io(e)),
    }
}

/// Every live root's manifest; a manifest hold keeps its whole member closure.
pub(crate) fn held_objects(store: &Store) -> Result<Vec<HeldKey>> {
    let spaces = match fs::read_dir(directory(store)) {
        Ok(spaces) => spaces,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(io(e)),
    };
    let mut held = vec![];
    for space in spaces {
        let name = space.map_err(io)?.file_name();
        let Some(space) = name.to_str().filter(|s| space_dir(store, s).is_ok()) else {
            continue;
        };
        for root in list(store, space)? {
            held.push(HeldKey {
                key: crate::storage::manifest_key(&root.manifest.sha256)?,
                kind: "manifest".into(),
                length: root.manifest.length,
                sha256: root.manifest.sha256,
            });
        }
    }
    Ok(held)
}
