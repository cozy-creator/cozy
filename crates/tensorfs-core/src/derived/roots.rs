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
    pub manifest: ObjectRef,
    pub name: Option<String>,
}

impl Plain for Root {
    const MAX_BYTES: usize = 16 * 1024;
    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("PrivateDerivedRoot", value)?;
        let transaction = super::transaction_id(fields.req_str("transaction_id")?)?;
        let manifest = ObjectRef::from_value("private root manifest", fields.req("manifest")?)?;
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
            name,
        })
    }
    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("transaction_id", Value::str(self.transaction.clone())),
            ("manifest", self.manifest.to_value()),
        ];
        if let Some(name) = &self.name {
            fields.push(("scratch_root_id", Value::str(name.clone())));
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
    let path = path(store, &root.transaction)?;
    fs::create_dir_all(directory(store)).map_err(io)?;
    let temporary = path.with_extension(format!("tmp-{}", crate::meta::now_nanos_unique()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(io)?;
    file.write_all(&root.canonical_bytes())
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
    if read(store, transaction)?.is_some_and(|root| root.name.is_some()) {
        return refuse(
            Code::DISPOSITION_CONFLICT,
            "a named private root cannot become a pending candidate",
        );
    }
    write(
        store,
        &Root {
            transaction: transaction.into(),
            manifest: manifest.clone(),
            name: None,
        },
    )
}

pub(super) fn adopt(
    store: &Store,
    transaction: &str,
    manifest: &ObjectRef,
    name: &str,
) -> Result<()> {
    if let Some(root) = read(store, transaction)? {
        if root.manifest != *manifest || root.name.as_ref().is_some_and(|old| old != name) {
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
            manifest: manifest.clone(),
            name: Some(name.into()),
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
