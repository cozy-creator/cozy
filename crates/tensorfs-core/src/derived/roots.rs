//! Private derived roots are filesystem facts, just like repository roots. Transaction
//! metadata journals lifecycle; GC never needs that database to discover retained bytes.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::FileExt;
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
    /// The newest TensorFS that wrote this root.
    pub tensorfs: Option<String>,
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
        let tensorfs = fields
            .opt("tensorfs")
            .map(|value| crate::canon::as_str("Root", "tensorfs", value).map(str::to_string))
            .transpose()?;
        fields.done_written_by(tensorfs.as_deref())?;
        Ok(Self {
            tensorfs,
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
        if let Some(tensorfs) = &self.tensorfs {
            fields.push(("tensorfs", Value::str(tensorfs.clone())));
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

/// A root must survive the machine on every disk but an ephemeral one, whose Store dies with
/// its container. A parts journal follows the Store's admissions instead ([`Store::flushes`]):
/// under a writer's epoch its lines become durable with that writer's next checkpoint.
fn durable(store: &Store) -> bool {
    store.disk_class() != crate::disk::DiskClass::Ephemeral
}

/// Create the root directory; true when this call created it (so its parents need a sync).
fn create_directory(store: &Store) -> Result<bool> {
    let created = !directory(store).is_dir();
    fs::create_dir_all(directory(store)).map_err(io)?;
    Ok(created)
}
fn path(store: &Store, transaction: &str) -> Result<PathBuf> {
    let transaction = super::transaction_id(transaction)?;
    Ok(directory(store).join(format!(
        "{}.json",
        transaction.trim_start_matches("sha256:")
    )))
}

/// Accepted-part objects, one appended line per part, beside the private root. Appending
/// costs that part alone. A line the writer died while appending was never accepted.
fn parts_path(store: &Store, transaction: &str) -> Result<PathBuf> {
    Ok(path(store, transaction)?.with_extension("parts"))
}

fn parts_line(objects: &[ObjectRef]) -> Vec<u8> {
    let mut line = crate::canon::write(&Value::arr(
        objects.iter().map(ObjectRef::to_value).collect(),
    ));
    line.push(b'\n');
    line
}

pub(super) fn append_parts(store: &Store, transaction: &str, objects: &[ObjectRef]) -> Result<()> {
    if objects.is_empty() {
        return Ok(());
    }
    let path = parts_path(store, transaction)?;
    let parents = create_directory(store)?;
    let created = !path.exists();
    let mut file = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(&path)
        .map_err(io)?;
    let length = file.metadata().map_err(io)?.len();
    let mut last = [b'\n'];
    if length > 0 {
        file.read_exact_at(&mut last, length - 1).map_err(io)?;
    }
    // Terminate a line torn by a dead writer so this one parses on its own.
    let mut bytes = if last[0] == b'\n' {
        Vec::new()
    } else {
        vec![b'\n']
    };
    bytes.extend(parts_line(objects));
    file.write_all(&bytes).map_err(io)?;
    if store.flushes() {
        file.sync_data().map_err(io)?;
        if created {
            sync_directories(store, parents)?;
        }
    }
    Ok(())
}

/// Replace the journal with exactly these objects (a writer's merged progress).
pub(super) fn rewrite_parts(store: &Store, transaction: &str, objects: &[ObjectRef]) -> Result<()> {
    let path = parts_path(store, transaction)?;
    let parents = create_directory(store)?;
    let temporary = path.with_extension(format!("parts-tmp-{}", crate::meta::now_nanos_unique()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(io)?;
    let bytes = if objects.is_empty() {
        Vec::new()
    } else {
        parts_line(objects)
    };
    file.write_all(&bytes).map_err(io)?;
    if store.flushes() {
        file.sync_all().map_err(io)?;
    }
    fs::rename(&temporary, &path).map_err(io)?;
    if store.flushes() {
        sync_directories(store, parents)?;
    }
    Ok(())
}

/// Every object named by any transaction's parts journal.
pub(super) fn journaled(store: &Store) -> Result<Vec<ObjectRef>> {
    let entries = match fs::read_dir(directory(store)) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(io(error)),
    };
    let mut objects = Vec::new();
    for entry in entries {
        let path = entry.map_err(io)?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("parts") {
            continue;
        }
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(io(error)),
        };
        for line in bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
        {
            let Ok(value) = crate::canon::parse_canonical(line, crate::limits::DOC_MAX_BYTES)
            else {
                continue;
            };
            for value in crate::canon::as_arr("PrivateDerivedParts", "objects", &value)? {
                objects.push(ObjectRef::from_value("private parts journal", value)?);
            }
        }
    }
    Ok(objects)
}

/// A rename needs its own directory synced; the parents only when this write created it.
fn sync_directories(store: &Store, parents: bool) -> Result<()> {
    let own = [directory(store)];
    let all = [
        directory(store),
        store.root().join("roots"),
        store.root().to_path_buf(),
    ];
    for directory in if parents { &all[..] } else { &own[..] } {
        File::open(directory)
            .and_then(|file| file.sync_all())
            .map_err(io)?;
    }
    Ok(())
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
    let root = &Root {
        tensorfs: Some(crate::VERSION.into()),
        ..root.clone()
    };
    let bytes = root.canonical_bytes();
    if bytes.len() > Root::MAX_BYTES {
        return refuse(Code::COUNT_CAP, "private derived root exceeds its byte cap");
    }
    let path = path(store, &root.transaction)?;
    let parents = create_directory(store)?;
    let temporary = path.with_extension(format!("tmp-{}", crate::meta::now_nanos_unique()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(io)?;
    file.write_all(&bytes).map_err(io)?;
    if durable(store) {
        file.sync_all().map_err(io)?;
    }
    fs::rename(&temporary, &path).map_err(io)?;
    if durable(store) {
        sync_directories(store, parents)?;
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
        tensorfs: Some(crate::VERSION.into()),
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
            tensorfs: Some(crate::VERSION.into()),
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
    match fs::remove_file(parts_path(store, transaction)?) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io(error)),
    }
    match fs::remove_file(path(store, transaction)?) {
        Ok(()) if !durable(store) => Ok(()),
        Ok(()) => File::open(directory(store))
            .and_then(|file| file.sync_all())
            .map_err(io),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io(error)),
    }
}
