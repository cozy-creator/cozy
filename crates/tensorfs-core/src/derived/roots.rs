//! Private derived roots are filesystem facts, just like repository roots. Transaction
//! metadata journals lifecycle; GC never needs that database to discover retained bytes.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

use crate::canon::{Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{ObjectRef, Plain};
use crate::store::Store;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Root {
    pub transaction: String,
    pub manifest: Option<ObjectRef>,
    pub blobs: Vec<ObjectRef>,
    pub sources: Vec<ObjectRef>,
    pub checkpoint: Option<ObjectRef>,
    pub name: Option<String>,
    pub restore_epoch: Option<u64>,
}

impl Plain for Root {
    const MAX_BYTES: usize = crate::limits::DOC_MAX_BYTES;
    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("PrivateDerivedRoot", value)?;
        let transaction = super::transaction_id(fields.req_str("transaction_id")?)?;
        let manifest = fields
            .opt("manifest")
            .map(|value| ObjectRef::from_value("private root manifest", value))
            .transpose()?;
        let checkpoint = fields
            .opt("checkpoint")
            .map(|value| ObjectRef::from_value("private root checkpoint", value))
            .transpose()?;
        let restore_epoch = fields
            .opt("restore_epoch")
            .map(|value| crate::canon::as_uint("PrivateDerivedRoot", "restore_epoch", value))
            .transpose()?;
        if restore_epoch.is_some_and(|epoch| epoch == 0 || epoch > crate::limits::INT_MAX as u64) {
            return refuse(
                Code::NUMBER_RANGE,
                "restore epoch is outside the canonical range",
            );
        }
        let mut refs = |key| -> Result<Vec<ObjectRef>> {
            fields
                .opt(key)
                .map(|value| {
                    crate::canon::as_arr("PrivateDerivedRoot", key, value)?
                        .iter()
                        .map(|value| ObjectRef::from_value(key, value))
                        .collect()
                })
                .transpose()
                .map(Option::unwrap_or_default)
        };
        let blobs = refs("blobs")?;
        let sources = refs("sources")?;
        let name = fields
            .opt("scratch_root_id")
            .map(|value| {
                let name = crate::canon::as_str("PrivateDerivedRoot", "scratch_root_id", value)?;
                super::name("private scratch root", name)?;
                Ok(name.to_string())
            })
            .transpose()?;
        fields.done()?;
        Ok(Self {
            transaction,
            manifest,
            blobs,
            sources,
            checkpoint,
            name,
            restore_epoch,
        })
    }
    fn to_value(&self) -> Value {
        let mut fields = vec![("transaction_id", Value::str(self.transaction.clone()))];
        if let Some(manifest) = &self.manifest {
            fields.push(("manifest", manifest.to_value()));
        }
        if let Some(checkpoint) = &self.checkpoint {
            fields.push(("checkpoint", checkpoint.to_value()));
        }
        for (name, objects) in [("blobs", &self.blobs), ("sources", &self.sources)] {
            if !objects.is_empty() {
                fields.push((
                    name,
                    Value::arr(objects.iter().map(ObjectRef::to_value).collect()),
                ));
            }
        }
        if let Some(name) = &self.name {
            fields.push(("scratch_root_id", Value::str(name.clone())));
        }
        if let Some(epoch) = self.restore_epoch {
            fields.push(("restore_epoch", Value::uint(epoch)));
        }
        Value::obj(fields)
    }
}

fn io(error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("private derived root: {error}"),
    }
}
fn directory(store: &Store) -> PathBuf {
    store.root().join("roots/derived")
}
fn path(store: &Store, transaction: &str) -> Result<PathBuf> {
    let transaction = super::transaction_id(transaction)?;
    Ok(directory(store).join(format!(
        "{}.json",
        transaction.trim_start_matches("sha256:")
    )))
}

pub(super) fn read(store: &Store, transaction: &str) -> Result<Option<Root>> {
    let path = path(store, transaction)?;
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io(error)),
    };
    let mut bytes = Vec::new();
    file.take(Root::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    let root = Root::parse(&bytes)?;
    if root.transaction != transaction {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "private root path names another transaction",
        );
    }
    Ok(Some(root))
}

pub(super) fn all(store: &Store) -> Result<Vec<Root>> {
    let entries = match fs::read_dir(directory(store)) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(io(error)),
    };
    let mut roots = Vec::new();
    for entry in entries {
        let entry = entry.map_err(io)?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(hex) = name.strip_suffix(".json") else {
            continue;
        };
        let transaction = format!("sha256:{hex}");
        if let Some(root) = read(store, &transaction)? {
            roots.push(root);
        }
    }
    Ok(roots)
}

pub(super) fn write(store: &Store, root: &Root) -> Result<()> {
    let bytes = root.canonical_bytes();
    if bytes.len() > Root::MAX_BYTES {
        return refuse(Code::COUNT_CAP, "private derived root exceeds its byte cap");
    }
    let path = path(store, &root.transaction)?;
    fs::create_dir_all(directory(store)).map_err(io)?;
    let temporary = path.with_extension(format!("tmp-{}", crate::meta::now_nanos_unique()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(io)?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .map_err(io)?;
    fs::rename(&temporary, &path).map_err(io)?;
    for directory in [
        directory(store),
        store.root().join("roots"),
        store.root().to_path_buf(),
    ] {
        File::open(directory)
            .and_then(|file| file.sync_all())
            .map_err(io)?;
    }
    Ok(())
}

pub(super) fn pending(store: &Store, transaction: &str, manifest: &ObjectRef) -> Result<()> {
    let previous = read(store, transaction)?;
    if previous.as_ref().is_some_and(|root| root.name.is_some()) {
        return refuse(
            Code::DISPOSITION_CONFLICT,
            "a named private root cannot become a pending candidate",
        );
    }
    let mut root = previous.unwrap_or_else(|| Root {
        transaction: transaction.into(),
        manifest: None,
        name: None,
        blobs: Vec::new(),
        sources: Vec::new(),
        checkpoint: None,
        restore_epoch: None,
    });
    // The filesystem write precedes SQL commit. Until that succeeds, retry still
    // needs the exact input manifests and accepted work, even if their original
    // owners released them. ADOPT/dispose replaces this conservative retention.
    root.manifest = Some(manifest.clone());
    write(store, &root)
}

pub(super) fn adopt(
    store: &Store,
    transaction: &str,
    manifest: &ObjectRef,
    name: &str,
) -> Result<()> {
    if let Some(root) = read(store, transaction)? {
        if root.manifest.as_ref() != Some(manifest)
            || root.name.as_ref().is_some_and(|old| old != name)
        {
            return refuse(
                Code::DISPOSITION_CONFLICT,
                "private root already binds different output or name",
            );
        }
    }
    if all(store)?
        .iter()
        .any(|root| root.transaction != transaction && root.name.as_deref() == Some(name))
    {
        return refuse(
            Code::DISPOSITION_CONFLICT,
            "private scratch root already names another transaction",
        );
    }
    write(
        store,
        &Root {
            transaction: transaction.into(),
            manifest: Some(manifest.clone()),
            name: Some(name.into()),
            blobs: Vec::new(),
            sources: Vec::new(),
            checkpoint: None,
            restore_epoch: None,
        },
    )
}

pub(super) fn remove(store: &Store, transaction: &str) -> Result<()> {
    match fs::remove_file(path(store, transaction)?) {
        Ok(()) => File::open(directory(store))
            .and_then(|file| file.sync_all())
            .map_err(io),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io(error)),
    }
}
