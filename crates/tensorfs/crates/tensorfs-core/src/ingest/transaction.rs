//! The ingest TRANSACTION: everything a border run produces lands under a temporary
//! candidate directory under `tmp/`, and a separate explicit act installs it.
//!
//! The property this file exists to hold: **a crash, a timeout, a rejection or a walked-away
//! session leaves no visible checkpoint.** Candidate objects may well be in the CAS — they
//! are content-addressed and harmless there — but nothing NAMES them except a candidate root.
//! `install` is the coordinator's act, performed against a receipt that binds the exact
//! subject/manifest/header triple, and it is the only repository-release writer.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::canon::{as_arr, Fields, Value};
use crate::checkpoint::{self as ck};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{ascii_name, ObjectRef, Plain};
use crate::limits;
use crate::manifest::Manifest;
use crate::store::Store;


fn io(what: impl AsRef<str>, e: impl Into<std::io::Error>) -> Refusal {
    crate::store::classify_io(what, e)
}

// ---------------------------------------------------------------- the resumable session

/// The candidate namespace. Written first, deleted by reap, promoted only by `install`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestSession {
    pub session: String,
    pub tenant: String,
    pub candidates: Vec<ObjectRef>,
    /// The newest TensorFS that wrote this root.
    pub tensorfs: Option<String>,
}

impl Plain for IngestSession {
    const MAX_BYTES: usize = limits::SPEC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("IngestSession", v)?;
        let mut candidates = Vec::new();
        for c in as_arr("IngestSession", "candidates", f.req("candidates")?)? {
            candidates.push(ObjectRef::from_value("candidate", c)?);
        }
        let session = f.req_str("session")?.to_string();
        let tenant = f.req_str("tenant")?.to_string();
        let tensorfs = f
            .opt("tensorfs")
            .map(|value| {
                crate::canon::as_str("IngestSession", "tensorfs", value).map(str::to_string)
            })
            .transpose()?;
        f.done_written_by(tensorfs.as_deref())?;
        ascii_name("IngestSession.session", &session, limits::MAX_NAME_BYTES)?;
        ascii_name("IngestSession.tenant", &tenant, limits::MAX_NAME_BYTES)?;
        Ok(IngestSession {
            tensorfs,
            session,
            tenant,
            candidates,
        })
    }

    fn to_value(&self) -> Value {
        let mut fields = vec![
            (
                "candidates",
                Value::arr(self.candidates.iter().map(|c| c.to_value()).collect()),
            ),
            ("session", Value::str(self.session.clone())),
            ("tenant", Value::str(self.tenant.clone())),
        ];
        if let Some(tensorfs) = &self.tensorfs {
            fields.push(("tensorfs", Value::str(tensorfs.clone())));
        }
        Value::obj(fields)
    }
}

pub fn candidate_dir(store_root: &Path, session: &str) -> PathBuf {
    store_root.join("tmp").join("ingest").join(session)
}

/// Write the root FIRST: from this instant the session is reapable, whatever happens next.
///
/// The operation row and retained writer marker are created before any final-key write.
/// A same-session retry claims the canonical existing session instead of inventing a second
/// transaction or discarding already verified bytes.
pub fn open_root(
    store: &Store,
    session: &str,
    tenant: &str,
) -> Result<(IngestSession, crate::catalog::WriterGuard)> {
    let store_root = store.root();
    let dir = candidate_dir(store_root, session);
    if dir.is_dir() {
        let root = read_session(store_root, session)?;
        if root.tenant != tenant {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "same ingest session was opened for a different tenant",
            );
        }
        let writer = crate::catalog::Catalog::open(store.root())?
            .resume_operation(session, tenant, "ingest")?;
        return Ok((root, writer));
    }
    fs::create_dir_all(&dir).map_err(|e| io("mkdir ingest candidate", e))?;
    let root = IngestSession {
        tensorfs: Some(crate::VERSION.into()),
        session: session.to_string(),
        tenant: tenant.to_string(),
        candidates: Vec::new(),
    };
    write_session(store_root, &root)?;
    let writer =
        crate::catalog::Catalog::open(store.root())?.begin_operation(session, tenant, "ingest")?;
    Ok((root, writer))
}

/// Resume an install after the worker exited or after the rebuildable catalog was restored.
/// Re-register every final key from the durable candidate Manifest before the repository CAS.
pub fn resume_operation(
    store: &Store,
    session: &str,
    manifest_ref: &ObjectRef,
    manifest: &Manifest,
) -> Result<crate::catalog::WriterGuard> {
    let catalog = crate::catalog::Catalog::open(store.root())?;
    let root = read_session(store.root(), session)?;
    let writer = catalog.resume_operation(session, &root.tenant, "ingest")?;
    catalog.hold(
        session,
        &crate::storage::manifest_key(&manifest_ref.sha256)?,
        manifest_ref.length,
    )?;
    let walk = ck::walk(store, manifest)?;
    for object in walk.distinct() {
        catalog.hold(
            session,
            &crate::storage::blob_key(&object.sha256)?,
            object.length,
        )?;
    }
    Ok(writer)
}

pub fn write_session(store_root: &Path, root: &IngestSession) -> Result<()> {
    let root = &IngestSession {
        tensorfs: Some(crate::VERSION.into()),
        ..root.clone()
    };
    let dir = candidate_dir(store_root, &root.session);
    fs::create_dir_all(&dir).map_err(|e| io("mkdir ingest candidate", e))?;
    // Renamed into place: a concurrent reader sees the whole root or none, never a torn one.
    let staged = dir.join(format!("session.json.{}", crate::meta::now_nanos_unique()));
    fs::write(&staged, root.canonical_bytes()).map_err(|e| io("write ingest session", e))?;
    fs::rename(&staged, dir.join("session.json")).map_err(|e| io("install ingest session", e))
}

pub fn read_session(store_root: &Path, session: &str) -> Result<IngestSession> {
    let p = candidate_dir(store_root, session).join("session.json");
    let read = cozy_platform::durable::open_racing(&p).and_then(|mut file| {
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map(|_| bytes)
    });
    match read {
        Ok(b) => IngestSession::parse(&b),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => refuse(
            Code::ROOT_ABSENT,
            format!("no ingest session {session:?}: reaped, never created, or already installed"),
        ),
        Err(e) => Err(io("read ingest session", e)),
    }
}

pub fn sessions(store: &Store) -> Vec<String> {
    let base = store.root().join("tmp").join("ingest");
    let mut sessions: Vec<String> = fs::read_dir(base)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect();
    sessions.sort();
    sessions
}

/// Make sure the session root exists. The journal lives inside it, and `gc::session_holds`
/// reads it — so this is also the act that puts in-flight conversion output under the
/// filesystem's own hold protocol.
///
/// The model-source path never opened one: `prepare_model_source` went straight to `execute`
/// with an operation id, so its output was named by nothing on disk and the first `tfs gc`
/// pass would have reclaimed all of it (`catalog.hold()` writes rows `gc` never reads).
pub fn ensure_session_root(store_root: &Path, session: &str, tenant: &str) -> Result<()> {
    ascii_name("ingest session", session, limits::MAX_NAME_BYTES)?;
    if session.contains('/') {
        return refuse(
            Code::KEY_GRAMMAR,
            "an ingest session name is one path segment",
        );
    }
    if read_session(store_root, session).is_err() {
        write_session(
            store_root,
            &IngestSession {
                tensorfs: Some(crate::VERSION.into()),
                session: session.to_string(),
                tenant: tenant.to_string(),
                candidates: Vec::new(),
            },
        )?;
    }
    Ok(())
}

/// Name objects in the session root, where the filesystem hold protocol can see them.
/// Additive and idempotent: a resumed run adds what it produced without unnaming what an
/// earlier pass did.
pub fn name_candidates(store_root: &Path, session: &str, refs: &[ObjectRef]) -> Result<()> {
    let mut root = read_session(store_root, session)?;
    for candidate in refs {
        if !root
            .candidates
            .iter()
            .any(|existing| existing.sha256 == candidate.sha256)
        {
            root.candidates.push(candidate.clone());
        }
    }
    root.candidates.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    write_session(store_root, &root)
}

/// Releases a v1 model-source operation and its candidate journal (the v1 derived tests).
#[cfg(test)]
pub(crate) fn release_model_source(store: &Store, operation_id: &str) -> Result<()> {
    crate::catalog::Catalog::open(store.root())?.release_source_preparation(operation_id, || {
        match fs::remove_dir_all(candidate_dir(store.root(), operation_id)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(crate::err::classify_io("remove model-source journal", error)),
        }
    })
}
