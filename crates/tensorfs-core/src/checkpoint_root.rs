//! Independent input custody for an exact, already-materialized repository checkpoint.
//! These filesystem roots retain bytes; they do not claim a computation or issue a receipt.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use fs2::FileExt;

use crate::canon::{Fields, Value};
use crate::catalog::WriterGuard;
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{hex64, prefixed, Doc, ObjectRef, Plain};
use crate::repository::{Repository, RepositoryName};
use crate::storage::HeldKey;
use crate::store::{Store, O_NOFOLLOW};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointRoot {
    pub owner: String,
    pub repository: RepositoryName,
    pub manifest: ObjectRef,
    pub released: bool,
    /// The newest TensorFS that wrote this root.
    pub tensorfs: Option<String>,
}

impl Plain for CheckpointRoot {
    const MAX_BYTES: usize = 4096;

    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("CheckpointRoot", value)?;
        let owner = prefixed("checkpoint root owner", fields.req_str("owner")?)?;
        let repository = repository(fields.req_str("repository")?)?;
        let manifest = ObjectRef::from_value("checkpoint root manifest", fields.req("manifest")?)?;
        let released = match fields.req("released")? {
            Value::Bool(value) => *value,
            _ => return refuse(Code::WRONG_TYPE, "checkpoint root release must be boolean"),
        };
        let tensorfs = fields
            .opt("tensorfs")
            .map(|value| {
                crate::canon::as_str("CheckpointRoot", "tensorfs", value).map(str::to_string)
            })
            .transpose()?;
        fields.done_written_by(tensorfs.as_deref())?;
        if manifest.length == 0 || manifest.length > crate::limits::INT_MAX as u64 {
            return refuse(Code::LENGTH_MISMATCH, "checkpoint root manifest is empty");
        }
        Ok(Self {
            tensorfs,
            owner,
            repository,
            manifest,
            released,
        })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("owner", Value::str(self.owner.clone())),
            (
                "repository",
                Value::str(format!("{}/{}", self.repository.org, self.repository.name)),
            ),
            ("manifest", self.manifest.to_value()),
            ("released", Value::Bool(self.released)),
        ];
        if let Some(tensorfs) = &self.tensorfs {
            fields.push(("tensorfs", Value::str(tensorfs.clone())));
        }
        Value::obj(fields)
    }
}

pub fn repository(value: &str) -> Result<RepositoryName> {
    let (org, name) = value.split_once('/').ok_or_else(|| Refusal {
        code: Code::KEY_GRAMMAR,
        detail: "checkpoint source repository must be org/name".into(),
    })?;
    RepositoryName::new(org, name)
}

fn io(error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("checkpoint root: {error}"),
    }
}

fn directory(store: &Store) -> PathBuf {
    store.root().join("roots/checkpoints")
}

fn path(store: &Store, owner: &str) -> Result<PathBuf> {
    prefixed("checkpoint root owner", owner)?;
    Ok(directory(store).join(format!("{}.json", &owner[7..])))
}

/// Inspect the native root, including its permanent release tombstone.
pub fn read(store: &Store, owner: &str) -> Result<Option<CheckpointRoot>> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path(store, owner)?)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io(error)),
    };
    let mut bytes = Vec::new();
    file.take(CheckpointRoot::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    let root = CheckpointRoot::parse(&bytes)?;
    if root.owner != owner {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "checkpoint root path names another owner",
        );
    }
    Ok(Some(root))
}

fn lock(store: &Store, owner: &str) -> Result<File> {
    let path = path(store, owner)?.with_extension("lock");
    fs::create_dir_all(directory(store)).map_err(io)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(O_NOFOLLOW)
        .open(path)
        .map_err(io)?;
    file.lock_exclusive().map_err(io)?;
    Ok(file)
}

fn write(store: &Store, root: &CheckpointRoot) -> Result<()> {
    let root = &CheckpointRoot {
        tensorfs: Some(crate::VERSION.into()),
        ..root.clone()
    };
    let path = path(store, &root.owner)?;
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
        fs::rename(&temporary, path).map_err(io)?;
        crate::store::fsync_entry_dir(store.root(), &directory(store))
    })();
    let _ = fs::remove_file(temporary);
    result
}

fn identity(owner: &str, source: &str, manifest: ObjectRef) -> Result<CheckpointRoot> {
    hex64("checkpoint root manifest", &manifest.sha256)?;
    if manifest.length == 0 || manifest.length > crate::limits::INT_MAX as u64 {
        return refuse(Code::LENGTH_MISMATCH, "checkpoint root manifest is empty");
    }
    Ok(CheckpointRoot {
        tensorfs: Some(crate::VERSION.into()),
        owner: prefixed("checkpoint root owner", owner)?,
        repository: repository(source)?,
        manifest,
        released: false,
    })
}

fn same_subject(root: &CheckpointRoot, wanted: &CheckpointRoot) -> Result<()> {
    if root.repository != wanted.repository || root.manifest != wanted.manifest {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "checkpoint root already names another subject",
        );
    }
    Ok(())
}

fn verify(store: &Store, manifest: &ObjectRef) -> Result<()> {
    let model = store.read_manifest(manifest)?;
    let closure = crate::checkpoint::walk_cozytensors(store, &model)?;
    closure.require_resident(store)?;
    for object in closure.distinct() {
        let record = store.record_valid(&object.sha256).map_err(|why| Refusal {
            code: Code::OBJECT_CORRUPT,
            detail: format!("checkpoint root closure has no current verification record: {why}"),
        })?;
        if record.length != object.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                "checkpoint root closure record changed length",
            );
        }
    }
    Ok(())
}

/// Retain the runtime closure of an exact native repository checkpoint.
pub fn retain(
    store: &Store,
    owner: &str,
    source: &str,
    manifest: ObjectRef,
) -> Result<CheckpointRoot> {
    let wanted = identity(owner, source, manifest)?;
    let _gc = WriterGuard::acquire(store.root())?;
    let _owner = lock(store, owner)?;
    if let Some(root) = read(store, owner)? {
        same_subject(&root, &wanted)?;
        if root.released {
            return refuse(
                Code::TRANSACTION_CLOSED,
                "checkpoint root was permanently released",
            );
        }
        verify(store, &root.manifest)?;
        return Ok(root);
    }
    verify_source(store, &wanted.repository, &wanted.manifest)?;
    write(store, &wanted)?;
    Ok(wanted)
}

/// Validate all source inputs before the execution owner creates any recipient roots.
pub fn check_source(store: &Store, source: &str, manifest: &ObjectRef) -> Result<()> {
    let repository = repository(source)?;
    let _gc = WriterGuard::acquire(store.root())?;
    verify_source(store, &repository, manifest)
}

fn verify_source(store: &Store, repository: &RepositoryName, manifest: &ObjectRef) -> Result<()> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(store.repository_path(repository))
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Refusal {
                    code: Code::REPOSITORY_ABSENT,
                    detail: "checkpoint source repository is absent".into(),
                }
            } else {
                io(error)
            }
        })?;
    let mut body = Vec::new();
    file.take(Repository::MAX_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .map_err(io)?;
    let actual = Repository::parse(&body)?;
    if &actual.repo != repository
        || !actual
            .checkpoints
            .iter()
            .any(|row| &row.manifest == manifest)
    {
        return refuse(
            Code::ROOT_ABSENT,
            "checkpoint has no exact source repository membership",
        );
    }
    verify(store, manifest)
}

/// Release this exact owner permanently, even before its first acquisition.
pub fn release(
    store: &Store,
    owner: &str,
    source: &str,
    manifest: ObjectRef,
) -> Result<CheckpointRoot> {
    let mut wanted = identity(owner, source, manifest)?;
    let _gc = WriterGuard::acquire(store.root())?;
    let _owner = lock(store, owner)?;
    if let Some(root) = read(store, owner)? {
        same_subject(&root, &wanted)?;
        if root.released {
            return Ok(root);
        }
    }
    wanted.released = true;
    write(store, &wanted)?;
    Ok(wanted)
}

pub(crate) fn held_objects(store: &Store) -> Result<Vec<HeldKey>> {
    let entries = match fs::read_dir(directory(store)) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(io(error)),
    };
    let mut objects = Vec::new();
    for entry in entries {
        let entry = entry.map_err(io)?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(hex) = name.strip_suffix(".json") else {
            continue;
        };
        let root = read(store, &format!("sha256:{hex}"))?.ok_or_else(|| Refusal {
            code: Code::ROOT_ABSENT,
            detail: "checkpoint root disappeared during census".into(),
        })?;
        if !root.released {
            objects.push(HeldKey {
                key: crate::storage::manifest_key(&root.manifest.sha256)?,
                kind: "cozytensors".into(),
                length: root.manifest.length,
                sha256: root.manifest.sha256,
            });
        }
    }
    Ok(objects)
}
