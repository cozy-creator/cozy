//! The ingest TRANSACTION: everything a border run produces lands under an expiring
//! quarantine root, and a separate explicit act installs it.
//!
//! The property this file exists to hold: **a crash, a timeout, a rejection or a walked-away
//! session leaves no visible checkpoint.** Candidate objects may well be in the CAS — they
//! are content-addressed and harmless there — but nothing NAMES them except a root that
//! expires. `install` is the coordinator's act, performed against a receipt that binds the
//! exact subject/snapshot/header triple, and it is the only writer of `roots/`.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::canon::{as_arr, as_str, Fields, Value};
use crate::checkpoint::{self as ck, build_snapshot};
use crate::dtype::checked_bytes;
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Closure, Header, Part, Tensor};
use crate::ids::{ascii_name, check_format, Doc, ObjectRef};
use crate::limits;
use crate::manifest::SnapshotManifest;
use crate::spec::EncodingSpec;
use crate::store::Store;

use super::carrier::{self, SourceHeader};
use super::convert::{check_permute_size, Bytes, Plan};

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------- the expiring root

/// The candidate namespace. Written first, deleted by reap, promoted only by `install`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestRoot {
    pub session: String,
    pub tenant: String,
    pub created: u64,
    pub expires_at: u64,
    pub candidates: Vec<ObjectRef>,
}

impl Doc for IngestRoot {
    const FORMAT: &'static str = "tensorfs.ingest_root/1";
    const MAX_BYTES: usize = limits::SPEC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("IngestRoot", v)?;
        check_format(
            "IngestRoot",
            as_str("IngestRoot", "format", f.req("format")?)?,
            Self::FORMAT,
        )?;
        let mut candidates = Vec::new();
        for c in as_arr("IngestRoot", "candidates", f.req("candidates")?)? {
            candidates.push(ObjectRef::from_value("candidate", c)?);
        }
        let created = f.req_uint("created")?;
        let expires_at = f.req_uint("expires_at")?;
        let session = f.req_str("session")?.to_string();
        let tenant = f.req_str("tenant")?.to_string();
        f.done()?;
        ascii_name("IngestRoot.session", &session, limits::MAX_NAME_BYTES)?;
        ascii_name("IngestRoot.tenant", &tenant, limits::MAX_NAME_BYTES)?;
        Ok(IngestRoot {
            session,
            tenant,
            created,
            expires_at,
            candidates,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "candidates",
                Value::arr(self.candidates.iter().map(|c| c.to_value()).collect()),
            ),
            ("created", Value::uint(self.created)),
            ("expires_at", Value::uint(self.expires_at)),
            ("format", Value::str(Self::FORMAT)),
            ("session", Value::str(self.session.clone())),
            ("tenant", Value::str(self.tenant.clone())),
        ])
    }
}

pub fn quarantine_dir(store_root: &Path, session: &str) -> PathBuf {
    store_root.join("ingest").join(session)
}

pub fn installed_dir(store_root: &Path) -> PathBuf {
    store_root.join("roots")
}

/// Write the root FIRST: from this instant the session is reapable, whatever happens next.
pub fn open_root(store_root: &Path, session: &str, tenant: &str, ttl: u64) -> Result<IngestRoot> {
    let dir = quarantine_dir(store_root, session);
    fs::create_dir_all(&dir).map_err(|e| io("mkdir quarantine root", e))?;
    let now = now_secs();
    let root = IngestRoot {
        session: session.to_string(),
        tenant: tenant.to_string(),
        created: now,
        expires_at: now + ttl,
        candidates: Vec::new(),
    };
    write_root(store_root, &root)?;
    Ok(root)
}

pub fn write_root(store_root: &Path, root: &IngestRoot) -> Result<()> {
    let dir = quarantine_dir(store_root, &root.session);
    fs::create_dir_all(&dir).map_err(|e| io("mkdir quarantine root", e))?;
    fs::write(dir.join("root.json"), root.canonical_bytes()).map_err(|e| io("write root", e))
}

pub fn read_root(store_root: &Path, session: &str) -> Result<IngestRoot> {
    let p = quarantine_dir(store_root, session).join("root.json");
    match fs::read(&p) {
        Ok(b) => IngestRoot::parse(&b),
        Err(_) => refuse(
            Code::ROOT_ABSENT,
            format!("no ingest root {session:?}: reaped, never created, or already installed"),
        ),
    }
}

/// Age-bounded reap. An expired root is removed; its candidate objects stay in the CAS as
/// unreferenced content-addressed bytes (object GC is deliberately out of this issue's
/// scope) — what is removed is the only thing that could ever have NAMED them.
pub fn reap(store_root: &Path, now: u64) -> Result<(Vec<String>, Vec<String>)> {
    let base = store_root.join("ingest");
    let (mut gone, mut kept) = (Vec::new(), Vec::new());
    let rd = match fs::read_dir(&base) {
        Ok(r) => r,
        Err(_) => return Ok((gone, kept)),
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let expired = match fs::read(e.path().join("root.json")) {
            // An unparsable or headless root directory is debris by construction: nothing
            // can install from it, so it reaps rather than living forever.
            Ok(b) => IngestRoot::parse(&b).map(|r| r.expires_at <= now).unwrap_or(true),
            Err(_) => true,
        };
        if expired {
            fs::remove_dir_all(e.path()).map_err(|err| io("reap root", err))?;
            gone.push(name);
        } else {
            kept.push(name);
        }
    }
    gone.sort();
    kept.sort();
    Ok((gone, kept))
}

/// The coordinator's act, and the ONLY writer of `roots/`. The receipt must bind the exact
/// triple it is presented with, the snapshot must walk, and the root must not have expired.
pub fn install(
    store: &Store,
    session: &str,
    receipt: &super::IngestVerificationReceipt,
    subject: &super::IngestSubject,
    snap: &SnapshotManifest,
    header: &Header,
) -> Result<String> {
    let root = read_root(store.root(), session)?;
    if root.expires_at <= now_secs() {
        return refuse(
            Code::ROOT_EXPIRED,
            format!(
                "ingest root {session:?} expired at {} — a candidate never installs after \
                 its root does",
                root.expires_at
            ),
        );
    }
    receipt.binds(subject, snap, header)?;
    let walk = ck::walk(store, snap)?;
    let id = snap.checkpoint_id()?;
    let dir = installed_dir(store.root());
    fs::create_dir_all(&dir).map_err(|e| io("mkdir roots", e))?;
    let hex = id.strip_prefix("sha256:").unwrap_or(&id);
    fs::write(
        dir.join(format!("{hex}.json")),
        crate::canon::write(&Value::obj(vec![
            ("checkpoint", Value::str(id.clone())),
            ("header", header.object_ref().to_value()),
            ("installed_at", Value::uint(now_secs())),
            ("objects", Value::uint(walk.distinct().len() as u64)),
            ("receipt", receipt.object_ref().to_value()),
            ("session", Value::str(session.to_string())),
        ])),
    )
    .map_err(|e| io("write installed root", e))?;
    // The candidate is now a checkpoint; the quarantine root has no further job.
    let _ = fs::remove_dir_all(quarantine_dir(store.root(), session));
    Ok(id)
}

/// Installed checkpoints — the VISIBLE set. A crash mid-ingest must leave this unchanged.
pub fn installed(store_root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(installed_dir(store_root)) {
        for e in rd.flatten() {
            if let Some(n) = e.file_name().to_string_lossy().strip_suffix(".json") {
                out.push(n.to_string());
            }
        }
    }
    out.sort();
    out
}

// ---------------------------------------------------------------- executing a plan

#[derive(Debug)]
pub struct Outcome {
    pub header: Header,
    pub closure: Closure,
    pub snapshot: SnapshotManifest,
    pub header_ref: ObjectRef,
    pub snapshot_ref: ObjectRef,
    pub obs: Vec<(String, i64)>,
}

impl Outcome {
    pub fn get(&self, k: &str) -> i64 {
        self.obs
            .iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| *v)
            .unwrap_or(0)
    }
}

/// One source file as the executor sees it: the path plus its already-decoded header.
pub struct SourceFile {
    pub path: PathBuf,
    pub header: SourceHeader,
}

/// Execute a plan through the real CAS. Streaming by construction: an identity-class role
/// is copied through a fixed buffer and never resident, so peak RSS is a function of the
/// buffer and the header, not of the artifact.
pub fn execute(
    store: &Store,
    plan: &Plan,
    files: &[SourceFile],
    specs: &[(String, EncodingSpec)],
    configs: &[(String, crate::ids::ConfigRef)],
) -> Result<Outcome> {
    let t0 = Instant::now();
    let mut comps: Vec<(String, Vec<(String, Tensor)>)> = Vec::new();
    let mut closure = Closure::default();
    let mut enc_refs: Vec<ObjectRef> = Vec::new();

    let (mut written, mut inherited, mut hashed, mut deduped) = (0u64, 0u64, 0u64, 0u64);
    let (mut segments, mut inlines, mut parts) = (0usize, 0usize, 0usize);

    for op in &plan.ops {
        let spec = match specs.iter().find(|(a, _)| *a == op.encoding) {
            Some((_, s)) => s.clone(),
            None => {
                return refuse(
                    Code::UNKNOWN_ENCODING,
                    format!("{}: no spec for target {:?}", op.out_key, op.encoding),
                )
            }
        };
        let sid = spec.object_id();
        if closure.get(&sid).is_none() {
            if let Some(v) = &spec.vectors {
                let _ = v;
            }
            enc_refs.push(spec.object_ref());
            closure.insert(spec.clone());
        }

        let mut tparts: Vec<(String, Part)> = Vec::new();
        for r in &op.roles {
            parts += 1;
            let what = format!("{}/{}#{}", op.component, op.out_key, r.role);
            let part = match &r.bytes {
                Bytes::Stream { file, key } => {
                    let f = &files[*file];
                    let t = match f.header.get(key) {
                        Some(t) => t,
                        None => {
                            return refuse(
                                Code::MISSING_TENSOR,
                                format!("{what}: the plan names carrier key {key:?}, absent now"),
                            )
                        }
                    };
                    let mut region = carrier::Region::open(&f.path, f.header.data_start, t)?;
                    let w = ck::objectize(store, &what, r.dtype, r.shape.clone(), &mut region)?;
                    written += w.bytes;
                    hashed += w.bytes;
                    deduped += w.deduped;
                    segments += w.segments;
                    if w.segments == 0 {
                        inlines += 1;
                    }
                    w.part
                }
                Bytes::Permute { file, key, xform } => {
                    let f = &files[*file];
                    let t = match f.header.get(key) {
                        Some(t) => t,
                        None => {
                            return refuse(
                                Code::MISSING_TENSOR,
                                format!("{what}: the plan names carrier key {key:?}, absent now"),
                            )
                        }
                    };
                    check_permute_size(&what, t.nbytes())?;
                    let mut src = Vec::with_capacity(t.nbytes() as usize);
                    carrier::Region::open(&f.path, f.header.data_start, t)?
                        .read_to_end(&mut src)
                        .map_err(|e| io(format!("{what}: read run"), e))?;
                    let mut out = Vec::with_capacity(xform.out_bytes() as usize);
                    xform.apply(&src, &mut out)?;
                    drop(src);
                    let w = ck::objectize(store, &what, r.dtype, r.shape.clone(), &mut &out[..])?;
                    written += w.bytes;
                    hashed += w.bytes;
                    deduped += w.deduped;
                    segments += w.segments;
                    if w.segments == 0 {
                        inlines += 1;
                    }
                    w.part
                }
                Bytes::Inherit { part } => {
                    // CAS-ASSERTED INHERIT: the objects must already be admitted. Nothing is
                    // read, nothing is hashed, nothing is uploaded — the refs simply carry.
                    for s in part.segments() {
                        if !store.contains(&s.sha256) {
                            return refuse(
                                Code::OBJECT_ABSENT,
                                format!(
                                    "{what}: inherit-by-reference names {} which this store \
                                     has never admitted",
                                    s.id()
                                ),
                            );
                        }
                    }
                    inherited += checked_bytes(&what, &r.shape, r.dtype)?;
                    segments += part.segments().len();
                    (**part).clone()
                }
            };
            tparts.push((r.role.clone(), part));
        }

        let t = Tensor {
            dtype: op.logical_dtype,
            shape: op.logical_shape.clone(),
            encoding: sid,
            parts: tparts,
        };
        match comps.iter_mut().find(|(c, _)| *c == op.component) {
            Some((_, v)) => v.push((op.out_key.clone(), t)),
            None => comps.push((op.component.clone(), vec![(op.out_key.clone(), t)])),
        }
    }

    enc_refs.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    enc_refs.dedup_by(|a, b| a.sha256 == b.sha256);
    let header = Header {
        configs: configs.to_vec(),
        encodings: enc_refs,
        components: comps,
    };
    // Validation BEFORE anything is advertised: the candidate must be a legal checkpoint.
    header.validate(&closure)?;

    for (_, spec) in &closure.specs {
        ck::put_doc(store, spec)?;
    }
    let header_ref = ck::put_doc(store, &header)?;
    let snapshot = build_snapshot(&header, &header_ref, &closure)?;
    let snapshot_ref = ck::put_doc(store, &snapshot)?;

    let obs = vec![
        ("bytes_written".to_string(), written as i64),
        ("bytes_inherited".to_string(), inherited as i64),
        ("bytes_hashed".to_string(), hashed as i64),
        ("bytes_deduped".to_string(), deduped as i64),
        ("tensors".to_string(), plan.ops.len() as i64),
        ("parts".to_string(), parts as i64),
        ("segments".to_string(), segments as i64),
        ("inline_parts".to_string(), inlines as i64),
        ("markers_dropped".to_string(), plan.dropped_markers.len() as i64),
        ("roles_folded".to_string(), plan.folded_roles.len() as i64),
        ("carrier_header_bytes".to_string(), plan.header_bytes as i64),
        ("convert_elapsed_ms".to_string(), t0.elapsed().as_millis() as i64),
    ];

    Ok(Outcome {
        header,
        closure,
        snapshot,
        header_ref,
        snapshot_ref,
        obs,
    })
}
