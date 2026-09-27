//! Remote custody of one model-source operation's converted output.
//!
//! A bounded-disk ingest publishes converted objects while it converts, so its output never
//! has to fit on the pod. Once the publication custodian has verified an object, the caller
//! records that fact here. From then on the operation's journal stops holding that object
//! against GC, a resumed pass does not re-convert it, and finalize or a derivation over the
//! prepared model accept it without local bytes. Nothing READS a custodied object: it is not
//! resident, and a reader still has to fetch it back through the ordinary verified door.
//!
//! The record lives in the ingest session root and is released with it, so an attestation
//! never outlives the operation that made it. The custodian is named by opaque evidence the
//! caller supplies (its publication identity); TensorFS carries it and never interprets it.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use crate::canon::{as_arr, Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::Part;
use crate::ids::{ObjectRef, Plain};
use crate::limits;
use crate::store::Store;

use super::transaction::candidate_dir;

const FILE: &str = "custody.json";
const FORMAT: &str = "tensorfs.source-custody/1";
const EVIDENCE_MAX_BYTES: usize = 1024;

fn io(what: &str, error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("source custody {what}: {error}"),
    }
}

/// Every object this operation's custodian holds, with the evidence that says where.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Custody {
    objects: BTreeMap<String, (u64, String)>,
}

impl Plain for Custody {
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(value: &Value) -> Result<Self> {
        let mut fields = Fields::new("Custody", value)?;
        if fields.req_str("format")? != FORMAT {
            return refuse(Code::WRONG_TYPE, "source custody has an unknown format");
        }
        let mut objects = BTreeMap::new();
        for row in as_arr("Custody", "objects", fields.req("objects")?)? {
            let mut row = Fields::new("Custody.objects", row)?;
            let evidence = row.req_str("evidence")?.to_string();
            let object = ObjectRef::from_value("Custody.object", row.req("object")?)?;
            row.done()?;
            check_evidence(&evidence)?;
            if objects
                .insert(object.sha256, (object.length, evidence))
                .is_some()
            {
                return refuse(Code::DUPLICATE_KEY, "source custody repeats an object");
            }
        }
        fields.done()?;
        Ok(Self { objects })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("format", Value::str(FORMAT)),
            (
                "objects",
                Value::arr(
                    self.objects
                        .iter()
                        .map(|(sha256, (length, evidence))| {
                            Value::obj(vec![
                                ("evidence", Value::str(evidence.clone())),
                                (
                                    "object",
                                    ObjectRef {
                                        sha256: sha256.clone(),
                                        length: *length,
                                    }
                                    .to_value(),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

impl Custody {
    pub fn contains(&self, object: &ObjectRef) -> bool {
        self.objects
            .get(&object.sha256)
            .is_some_and(|(length, _)| *length == object.length)
    }

    /// A part stands on custody alone only when it has segments and the custodian holds
    /// every one of them. An inline part carries its bytes in the journal line itself.
    pub fn holds_part(&self, part: &Part) -> bool {
        !part.segments().is_empty() && part.segments().iter().all(|s| self.contains(s))
    }

    pub fn digests(&self) -> BTreeSet<String> {
        self.objects.keys().cloned().collect()
    }

    pub fn rows(&self) -> Vec<(ObjectRef, String)> {
        self.objects
            .iter()
            .map(|(sha256, (length, evidence))| {
                (
                    ObjectRef {
                        sha256: sha256.clone(),
                        length: *length,
                    },
                    evidence.clone(),
                )
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

fn check_evidence(evidence: &str) -> Result<()> {
    if evidence.is_empty()
        || evidence.len() > EVIDENCE_MAX_BYTES
        || !evidence.bytes().all(|b| (0x21..0x7f).contains(&b))
    {
        return refuse(
            Code::KEY_GRAMMAR,
            "custody evidence is 1..=1024 printable ASCII bytes without spaces",
        );
    }
    Ok(())
}

/// The operation's record, empty when it has none.
pub fn read(store_root: &Path, operation: &str) -> Result<Custody> {
    crate::catalog::source_operation_id(operation)?;
    read_session(store_root, operation)
}

/// Every object the operation's conversion journals name, custodied or not.
pub fn journalled(store_root: &Path, operation: &str) -> Result<Vec<ObjectRef>> {
    crate::catalog::source_operation_id(operation)?;
    super::journal::session_objects(&candidate_dir(store_root, operation))
}

/// A session directory GC already enumerated; its name came from the filesystem.
pub(crate) fn read_session(store_root: &Path, session: &str) -> Result<Custody> {
    let path = candidate_dir(store_root, session).join(FILE);
    let file = match fs::File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Custody::default()),
        Err(error) => return Err(io("open", error)),
    };
    let mut bytes = Vec::new();
    file.take(Custody::MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| io("read", error))?;
    Custody::parse(&bytes)
}

/// Every live session's custody, for the readers that walk more than one operation:
/// GC's census and a derivation over a prepared model.
pub fn live(store_root: &Path) -> Result<BTreeSet<String>> {
    let base = store_root.join("tmp").join("ingest");
    let entries = match fs::read_dir(&base) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(error) => return Err(io("list sessions", error)),
    };
    let mut out = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|error| io("list sessions", error))?;
        if !entry.path().is_dir() {
            continue;
        }
        let session = entry.file_name().to_string_lossy().to_string();
        out.extend(read_session(store_root, &session)?.digests());
    }
    Ok(out)
}

/// Record that the custodian named by `evidence` holds these journalled objects. Only
/// objects this operation's own conversion journal names may be recorded; the record is
/// rewritten whole, atomically, so a crash leaves either the old set or the new one.
pub fn record(store: &Store, operation: &str, objects: &[ObjectRef], evidence: &str) -> Result<()> {
    crate::catalog::source_operation_id(operation)?;
    check_evidence(evidence)?;
    let dir = candidate_dir(store.root(), operation);
    if !dir.is_dir() {
        return refuse(Code::ROOT_ABSENT, "source custody names no live session");
    }
    let journalled: BTreeMap<String, u64> = journalled(store.root(), operation)?
        .into_iter()
        .map(|object| (object.sha256, object.length))
        .collect();
    let mut custody = read(store.root(), operation)?;
    for object in objects {
        if journalled.get(&object.sha256) != Some(&object.length) {
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                format!(
                    "{} is not an object this operation's conversion journal names",
                    object.id()
                ),
            );
        }
        custody
            .objects
            .entry(object.sha256.clone())
            .or_insert_with(|| (object.length, evidence.to_string()));
    }
    let bytes = custody.canonical_bytes();
    if bytes.len() > Custody::MAX_BYTES {
        return refuse(Code::SIZE_CAP, "source custody exceeds its document bound");
    }
    let path = dir.join(FILE);
    let temporary = dir.join(format!(".{FILE}.{}", crate::meta::now_nanos_unique()));
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
