//! The CONVERSION JOURNAL — what a crashed ingest needs and did not have.
//!
//! The derive plane already had this. `derived::TransactionRow::added_parts` is a durable
//! record of every part written into an open transaction, and `meta.rs` states its purpose:
//! SQLite "retains only released-manifest lookup and resumable derived writes". A crashed
//! four-lane resumes. A crashed ingest did not: `transaction::execute` walked the whole plan
//! in one call, held every accepted `Part` in a `Vec` on the stack, and wrote the header at
//! the end. Kill it at 90% and the BYTES survive in the CAS — they are content-addressed and
//! harmless there — but their MEANING does not. A re-run re-reads, re-hashes and re-writes
//! everything already done and only discovers the collision at the final link. It saved
//! storage. It never saved time.
//!
//! This file is the missing op -> `Part` mapping, ported rather than invented:
//!
//! - ONE APPEND-ONLY LINE PER ROLE, canonical JSON, fsynced per op. The record is
//!   `(component, key, role) -> Part` — exactly `AddedPart`, exactly the derive plane's
//!   shape, so the two planes journal the same fact in the same form.
//! - KEYED BY PLAN, not by session. One model-source operation runs one plan PER PROFILE
//!   over the same carriers, so a journal that keyed on the session alone would have two
//!   plans overwriting each other's ops. The file name carries the plan digest; a plan that
//!   changed gets its own file and re-converts, and nothing silently reuses a part that was
//!   computed for different work.
//! - INSIDE THE INGEST SESSION ROOT, which already survives a crash, already resumes
//!   (`transaction::open_root`) and is already the filesystem's own GC hold protocol
//!   (`gc::session_holds`). Journalling a part is therefore the same act as protecting its
//!   objects from the next `tfs gc` pass — which is what `catalog.hold()` looked like it
//!   did and never did (see the note on `catalog::Catalog::hold`).
//!
//! WHAT THE JOURNAL MAY NEVER BECOME is an artifact. The CozyTensors format cannot express
//! *incomplete*: a header over 44 of 48 shards is a structurally valid header for a smaller
//! model. So incompleteness lives HERE and only here — `transaction::finalize` refuses to
//! write a header until every op the plan names has a journalled part.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::canon::{self, Fields, Value};
use crate::dtype::Dtype;
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Body, Part};
use crate::ids::{hex64, ObjectRef, Plain};
use crate::limits;
use crate::store::Store;


fn io(what: impl AsRef<str>, error: impl Into<std::io::Error>) -> Refusal {
    crate::store::classify_io(what, error)
}

/// One journalled op-role: the derive plane's `AddedPart`, on the ingest plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedPart {
    pub component: String,
    pub key: String,
    pub role: String,
    pub part: Part,
}

impl Plain for AddedPart {
    const MAX_BYTES: usize = limits::SPEC_MAX_BYTES;

    fn from_value(value: &Value) -> Result<Self> {
        let mut f = Fields::new("ConversionJournalPart", value)?;
        let component = f.req_str("component")?.to_string();
        let key = f.req_str("key")?.to_string();
        let part = part_from_value(f.req("part")?)?;
        let role = f.req_str("role")?.to_string();
        f.done()?;
        Ok(AddedPart {
            component,
            key,
            role,
            part,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("component", Value::str(self.component.clone())),
            ("key", Value::str(self.key.clone())),
            ("part", part_value(&self.part)),
            ("role", Value::str(self.role.clone())),
        ])
    }
}

/// The same `Part` wire form the derive plane journals (`derived::part_value`), so one
/// reader could serve both planes and the two can never drift into two encodings.
fn part_value(part: &Part) -> Value {
    let mut fields = vec![
        ("dtype", Value::str(part.dtype.name())),
        (
            "shape",
            Value::arr(part.shape.iter().map(|d| Value::uint(*d)).collect()),
        ),
    ];
    match &part.body {
        Body::Inline(value) => fields.push(("inline", Value::str(crate::b64::encode(value)))),
        Body::Segments(segments) => fields.push((
            "segments",
            Value::arr(segments.iter().map(ObjectRef::to_value).collect()),
        )),
    }
    Value::obj(fields)
}

fn part_from_value(value: &Value) -> Result<Part> {
    let mut f = Fields::new("ConversionJournalPart.part", value)?;
    let dtype = Dtype::parse(f.req_str("dtype")?)?;
    let shape = canon::as_arr("ConversionJournalPart.part", "shape", f.req("shape")?)?
        .iter()
        .map(|v| match v {
            Value::Int(i) if *i >= 0 => Ok(*i as u64),
            other => refuse(
                Code::WRONG_TYPE,
                format!("journalled shape dimension is {}", other.kind()),
            ),
        })
        .collect::<Result<Vec<u64>>>()?;
    let inline = f.opt("inline");
    let segments = f.opt("segments");
    f.done()?;
    let body = match (inline, segments) {
        (Some(value), None) => Body::Inline(crate::b64::decode(canon::as_str(
            "ConversionJournalPart.part",
            "inline",
            value,
        )?)?),
        (None, Some(values)) => Body::Segments(
            canon::as_arr("ConversionJournalPart.part", "segments", values)?
                .iter()
                .map(|value| ObjectRef::from_value("ConversionJournalPart.segment", value))
                .collect::<Result<Vec<_>>>()?,
        ),
        _ => {
            return refuse(
                Code::INLINE_EXCLUSIVE,
                "a journalled part carries exactly one of inline or segments",
            )
        }
    };
    let part = Part { dtype, shape, body };
    part.check_bytes("ConversionJournalPart")?;
    Ok(part)
}

fn journal_path(dir: &Path, plan_digest: &str) -> PathBuf {
    dir.join(format!("convert-{plan_digest}.jsonl"))
}

/// The durable op -> `Part` map for ONE plan under ONE session root.
pub struct Journal {
    path: PathBuf,
    parts: BTreeMap<(String, String, String), Part>,
    sink: File,
    /// Roles converted by THIS process, as opposed to replayed off disk.
    pub converted: usize,
    /// Roles this process took from the journal without reading a carrier byte.
    pub resumed: usize,
}

impl Journal {
    /// Replay whatever a previous run left, and open the file for appending. A line that
    /// does not parse ENDS the replay: an append interrupted mid-write leaves a partial last
    /// line, and everything before it is still exactly true.
    pub fn open(dir: &Path, plan_digest: &str) -> Result<Journal> {
        hex64("conversion journal plan", plan_digest)?;
        fs::create_dir_all(dir).map_err(|error| io("mkdir conversion journal", error))?;
        let path = journal_path(dir, plan_digest);
        let mut parts = BTreeMap::new();
        let mut good_bytes = 0u64;
        if let Ok(file) = File::open(&path) {
            for line in BufReader::new(file).split(b'\n') {
                let line = line.map_err(|error| io("read conversion journal", error))?;
                let Ok(entry) = AddedPart::parse(&line) else {
                    break;
                };
                good_bytes += line.len() as u64 + 1;
                parts.insert((entry.component, entry.key, entry.role), entry.part);
            }
        }
        // Truncate the torn tail before appending, so the file stays a sequence of whole
        // records and a second crash cannot bury good records behind a bad one.
        let sink = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| io("open conversion journal", error))?;
        if sink
            .metadata()
            .map_err(|error| io("stat conversion journal", error))?
            .len()
            != good_bytes
        {
            // Through a writing handle: Windows grants an appending one no truncation.
            OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_len(good_bytes))
                .map_err(|error| io("truncate torn conversion journal", error))?;
        }
        Ok(Journal {
            path,
            parts,
            sink,
            converted: 0,
            resumed: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Every role of one op, or nothing. A half-journalled op is re-converted whole: the
    /// roles of one tensor are cheap beside the ambiguity of a partial answer.
    pub fn op(&self, component: &str, key: &str, roles: &[String]) -> Option<Vec<(String, Part)>> {
        let mut found = Vec::with_capacity(roles.len());
        for role in roles {
            let part = self
                .parts
                .get(&(component.to_string(), key.to_string(), role.clone()))?;
            found.push((role.clone(), part.clone()));
        }
        Some(found)
    }

    /// Append one op's roles and make them durable. ONE fsync per op, not per role: an op is
    /// the unit a resume re-does, so it is the unit durability has to be established at.
    pub fn record(&mut self, component: &str, key: &str, roles: &[(String, Part)]) -> Result<()> {
        let mut buffer = Vec::new();
        for (role, part) in roles {
            let entry = AddedPart {
                component: component.to_string(),
                key: key.to_string(),
                role: role.clone(),
                part: part.clone(),
            };
            let line = entry.canonical_bytes();
            if line.len() >= AddedPart::MAX_BYTES {
                return refuse(
                    Code::COUNT_CAP,
                    format!("{component}/{key}#{role}: journal record exceeds its cap"),
                );
            }
            buffer.extend_from_slice(&line);
            buffer.push(b'\n');
        }
        self.sink
            .write_all(&buffer)
            .map_err(|error| io("append conversion journal", error))?;
        self.sink
            .sync_data()
            .map_err(|error| io("sync conversion journal", error))?;
        #[cfg(test)]
        RECORD_FAULT.with(|fault| {
            if let Some(fault) = fault.borrow().as_ref() {
                fault.hit("journal-record");
            }
        });
        for (role, part) in roles {
            self.parts.insert(
                (component.to_string(), key.to_string(), role.clone()),
                part.clone(),
            );
        }
        self.converted += roles.len();
        Ok(())
    }

    /// This journal's exact bytes, for the plane that publishes it somewhere this pod is
    /// not (`crate::durability`). It is read back off the file rather than re-serialized
    /// from `parts`, so what is published is what a `Journal::open` would replay — one
    /// authority for the on-disk form, not two that could drift.
    pub fn snapshot(&self) -> Result<Vec<u8>> {
        fs::read(&self.path).map_err(|error| io("read conversion journal for publication", error))
    }

    /// The latest locally exported cursor is distinct from the owner's acknowledged head.
    /// Persisting it prevents upload ACK lag from forking this journal's immutable chain.
    pub fn local_head(&self) -> Result<Option<ObjectRef>> {
        let bytes = match fs::read(self.path.with_extension("head.json")) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io("read local checkpoint cursor", error)),
        };
        let value = canon::parse_canonical(&bytes, limits::SPEC_MAX_BYTES)?;
        ObjectRef::from_value("local checkpoint cursor", &value).map(Some)
    }

    pub fn save_head(&self, head: &ObjectRef) -> Result<()> {
        if self.local_head()?.as_ref() == Some(head) {
            return Ok(());
        }
        let path = self.path.with_extension("head.json");
        let temporary = path.with_extension(format!("tmp-{}", crate::meta::now_nanos_unique()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| io("create local checkpoint cursor", error))?;
        file.write_all(&canon::write(&head.to_value()))
            .and_then(|()| file.sync_all())
            .map_err(|error| io("sync local checkpoint cursor", error))?;
        fs::rename(&temporary, &path)
            .map_err(|error| io("replace local checkpoint cursor", error))?;
        cozy_platform::durable::sync_dir(path.parent().expect("journal has a parent"))
            .map_err(|error| io("sync local checkpoint cursor directory", error))
    }

    /// Every object this journal names, for the filesystem hold protocol.
    pub fn objects(&self) -> Vec<ObjectRef> {
        let mut out: Vec<ObjectRef> = self
            .parts
            .values()
            .flat_map(|part| part.segments().iter().cloned())
            .collect();
        out.sort_by(|a, b| a.sha256.cmp(&b.sha256));
        out.dedup_by(|a, b| a.sha256 == b.sha256);
        out
    }

}

/// Install a journal a previous pod published, where `Journal::open` will replay it.
///
/// Every line is parsed BEFORE anything is written: a snapshot arrives from a cache, and a
/// cache is not an authority — the bytes were verified against the digest that named them,
/// which says they are the snapshot, not that the snapshot is well formed. Installing it
/// makes no part true either: `convert` re-checks each one against `segments_still_stand`
/// before reusing it, so a journal restored beside objects that did not survive re-converts
/// exactly those ops.
/// Remove one plan's journal and cursor: work that can no longer be continued.
pub fn discard(dir: &Path, plan_digest: &str) -> Result<()> {
    let journal = journal_path(dir, plan_digest);
    for path in [journal.with_extension("head.json"), journal] {
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(io("discard conversion journal", error)),
        }
    }
    Ok(())
}

pub fn install(dir: &Path, plan_digest: &str, bytes: &[u8]) -> Result<()> {
    hex64("conversion journal plan", plan_digest)?;
    for line in bytes.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        AddedPart::parse(line)?;
    }
    if bytes.last().is_some_and(|byte| *byte != b'\n') {
        return refuse(
            Code::TRAILING_BYTES,
            "a published conversion journal ends mid-record",
        );
    }
    fs::create_dir_all(dir).map_err(|error| io("mkdir conversion journal", error))?;
    let path = journal_path(dir, plan_digest);
    let temporary = path.with_extension("jsonl.incoming");
    fs::write(&temporary, bytes)
        .map_err(|error| io("write published conversion journal", error))?;
    fs::rename(&temporary, &path).map_err(|error| io("install published conversion journal", error))
}

/// A journalled part may be reused only if its objects are still admissible. This is
/// `contains` is a presence hint and nothing
/// more, so admission takes a standing verification record or exactly one rehash.
pub fn segments_still_stand(store: &Store, part: &Part) -> bool {
    for segment in part.segments() {
        let record = match store.record_valid(&segment.sha256) {
            Ok(record) => record,
            Err(_) => match store.verify(&segment.sha256) {
                Ok(crate::store::Verdict::Verified { .. }) => {
                    match store.record_valid(&segment.sha256) {
                        Ok(record) => record,
                        Err(_) => return false,
                    }
                }
                _ => return false,
            },
        };
        if record.length != segment.length {
            return false;
        }
    }
    true
}

/// Every object every conversion journal under one session root names.
///
/// `gc::session_holds` reads this beside `IngestSession.candidates`: a session whose
/// conversion is still running has journalled parts and no candidates yet, and the objects
/// under those parts are exactly the in-flight output a GC pass must not reclaim.
pub fn session_objects(dir: &Path) -> Result<Vec<ObjectRef>> {
    let mut out: Vec<ObjectRef> = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return Ok(out),
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("convert-") || !name.ends_with(".jsonl") {
            continue;
        }
        let file = match File::open(entry.path()) {
            Ok(file) => file,
            Err(_) => continue,
        };
        for line in BufReader::new(file).split(b'\n') {
            let line = line.map_err(|error| io("read conversion journal", error))?;
            let Ok(record) = AddedPart::parse(&line) else {
                break;
            };
            out.extend(record.part.segments().iter().cloned());
        }
    }
    out.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    out.dedup_by(|a, b| a.sha256 == b.sha256);
    Ok(out)
}

#[cfg(test)]
thread_local! {
    pub(crate) static RECORD_FAULT: std::cell::RefCell<Option<crate::store::Fault>> = const { std::cell::RefCell::new(None) };
}
