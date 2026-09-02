//! The ingest TRANSACTION: everything a border run produces lands under a temporary
//! candidate directory under `tmp/`, and a separate explicit act installs it.
//!
//! The property this file exists to hold: **a crash, a timeout, a rejection or a walked-away
//! session leaves no visible checkpoint.** Candidate objects may well be in the CAS — they
//! are content-addressed and harmless there — but nothing NAMES them except a candidate root.
//! `install` is the coordinator's act, performed against a receipt that binds the exact
//! subject/manifest/header triple, and it is the only repository-release writer.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::canon::{as_arr, Fields, Value};
use crate::checkpoint::{self as ck, build_manifest};
use crate::dtype::checked_bytes;
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Closure, Header, Part, Tensor};
use crate::ids::{ascii_name, hex64, Doc, ObjectRef, Plain};
use crate::limits;
use crate::manifest::Manifest;
use crate::repository::{
    CheckpointEvidence, IngestProvenance, Mutation, ReleaseLane, RepositoryName,
    VerificationEvidence,
};
use crate::spec::EncodingSpec;
use crate::store::{Store, Verdict};

use super::carrier::{self, SourceHeader};
use super::convert::{check_permute_size, Bytes, Plan};

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

// ---------------------------------------------------------------- the resumable session

/// The candidate namespace. Written first, deleted by reap, promoted only by `install`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestSession {
    pub session: String,
    pub tenant: String,
    pub candidates: Vec<ObjectRef>,
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
        f.done()?;
        ascii_name("IngestSession.session", &session, limits::MAX_NAME_BYTES)?;
        ascii_name("IngestSession.tenant", &tenant, limits::MAX_NAME_BYTES)?;
        Ok(IngestSession {
            session,
            tenant,
            candidates,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            (
                "candidates",
                Value::arr(self.candidates.iter().map(|c| c.to_value()).collect()),
            ),
            ("session", Value::str(self.session.clone())),
            ("tenant", Value::str(self.tenant.clone())),
        ])
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
    let dir = candidate_dir(store_root, &root.session);
    fs::create_dir_all(&dir).map_err(|e| io("mkdir ingest candidate", e))?;
    fs::write(dir.join("session.json"), root.canonical_bytes())
        .map_err(|e| io("write ingest session", e))
}

pub fn read_session(store_root: &Path, session: &str) -> Result<IngestSession> {
    let p = candidate_dir(store_root, session).join("session.json");
    match fs::read(&p) {
        Ok(b) => IngestSession::parse(&b),
        Err(_) => refuse(
            Code::ROOT_ABSENT,
            format!("no ingest session {session:?}: reaped, never created, or already installed"),
        ),
    }
}

/// Explicitly abandon every session whose writer is gone. Candidate blobs remain unreferenced
/// content-addressed bytes until ordinary GC.
///
/// The operation's retained writer marker arbitrates this against a live worker or install.
pub fn reap(store: &Store) -> Result<(Vec<String>, Vec<String>)> {
    let store_root = store.root();
    let base = store_root.join("tmp").join("ingest");
    let (mut gone, mut kept) = (Vec::new(), Vec::new());
    let rd = match fs::read_dir(&base) {
        Ok(r) => r,
        Err(_) => return Ok((gone, kept)),
    };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let catalog = crate::catalog::Catalog::open(store.root())?;
        match catalog.abandon(&name, true) {
            Ok(()) => {
                fs::remove_dir_all(e.path()).map_err(|err| io("reap session", err))?;
                gone.push(name);
            }
            Err(error) if error.code == Code::STORE_BUSY => kept.push(name),
            Err(error) => return Err(error),
        }
    }
    gone.sort();
    kept.sort();
    Ok((gone, kept))
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

/// The coordinator's act. The receipt must bind the exact triple it is presented with, the
/// manifest must walk, and the candidate root must still exist.
///
/// Durable visibility is the repository release CAS. The operation's exact SQLite holds
/// bridge every final key to that CAS and release only after it commits.
pub struct Install<'a> {
    pub session: &'a str,
    pub receipt: &'a super::IngestVerificationReceipt,
    pub subject: &'a super::IngestSubject,
    pub manifest: &'a Manifest,
    pub header: &'a Header,
    pub stamp: &'a super::stamp::Stamp,
    pub profile: &'a super::IngestProfile,
    pub repo: RepositoryName,
    pub version: &'a str,
    pub lane: &'a str,
    pub source_uri: Option<&'a str>,
    pub declared_license: Option<&'a str>,
    pub observed_repository: Option<&'a str>,
}

pub fn install(store: &Store, request: Install<'_>) -> Result<String> {
    let Install {
        session,
        receipt,
        subject,
        manifest: snap,
        header,
        stamp,
        profile,
        repo,
        version,
        lane,
        source_uri,
        declared_license,
        observed_repository,
    } = request;
    receipt.binds(subject, snap, header, stamp)?;
    let walk = ck::walk(store, snap)?;
    // Containment is not residency: a root never commits over missing objects (tfs-020).
    walk.require_resident(store)?;
    let id = snap.manifest_id();
    read_session(store.root(), session)?;
    let mut sources = subject.sources.clone();
    sources.sort();
    sources.dedup();
    let ingest = IngestProvenance {
        source_dialect: profile.source_dialect.clone(),
        converter_build: profile.converter_build.clone(),
        converter_recipe: profile.converter_recipe.clone(),
        sources,
        source_uri: source_uri.map(str::to_string),
        declared_license: declared_license.map(str::to_string),
    };
    let verification = match &receipt.evidence {
        super::Evidence::ApprovedProducer {
            implementation,
            recipe,
        } => VerificationEvidence::ApprovedProducer {
            implementation: implementation.clone(),
            recipe: recipe.clone(),
        },
        super::Evidence::IndependentNumerical {
            verifier_build,
            report,
        } => VerificationEvidence::IndependentNumerical {
            verifier_build: verifier_build.clone(),
            report: report.clone(),
        },
    };
    let manifest = ObjectRef {
        sha256: id.trim_start_matches("sha256:").to_string(),
        length: snap.canonical_bytes().len() as u64,
    };
    let evidence = CheckpointEvidence {
        digests: ck::digests(store, header)?,
        ingest: Some(ingest),
        verification: Some(verification),
    };
    let checkpoint_manifest = manifest.clone();
    let checkpoint_evidence = evidence.clone();
    let local_mutation = if repo.org == "local" {
        if lane != "local" {
            return refuse(Code::KEY_GRAMMAR, "local ingest requires lane local");
        }
        hex64("local source selection", version)?;
        Some(Mutation::ReplaceLocal {
            repo: repo.clone(),
            manifest: manifest.clone(),
            version: version.to_string(),
            evidence: Box::new(evidence),
        })
    } else {
        None
    };
    let path = store.repository_path(&repo);
    let current = match fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(io("read repository before ingest install", error)),
    };
    if repo.org == "local" {
        let expected = observed_repository.ok_or_else(|| Refusal {
            code: Code::MISSING_FIELD,
            detail: "local ingest install requires --observed <sha256:...|absent>".into(),
        })?;
        match (expected, current.as_deref()) {
            ("absent", None) => {}
            ("absent", Some(_)) => {
                return refuse(
                    Code::REPOSITORY_CONFLICT,
                    "local alias exists but the caller observed absence",
                )
            }
            (digest, Some(bytes)) => {
                let expected = crate::ids::prefixed("observed repository", digest)?;
                let repository = crate::repository::Repository::parse(bytes)?;
                repository.local_checkpoint()?;
                let actual = format!("sha256:{}", repository.document_sha256());
                if actual != expected {
                    return refuse(
                        Code::REPOSITORY_CONFLICT,
                        format!("observed local repository {expected}, current is {actual}"),
                    );
                }
            }
            (_, None) => {
                return refuse(
                    Code::REPOSITORY_CONFLICT,
                    "local alias is absent but the caller observed repository bytes",
                )
            }
        }
    }
    if repo.org == "local" {
        store.apply_repository(
            current.as_deref(),
            local_mutation.as_ref().expect("local mutation"),
            &Default::default(),
        )?;
    } else {
        let checkpointed = store
            .apply_repository(
                current.as_deref(),
                &Mutation::PutCheckpoint {
                    repo: repo.clone(),
                    manifest: checkpoint_manifest,
                    evidence: Some(Box::new(checkpoint_evidence)),
                },
                &Default::default(),
            )?
            .expect("put_checkpoint retains the repository");
        let expected_revision = checkpointed
            .releases
            .iter()
            .find(|release| release.version == version)
            .map_or(0, |release| release.revision);
        let mutation = Mutation::UpdateRelease {
            expected_revision,
            repo: repo.clone(),
            remove: Vec::new(),
            set: vec![ReleaseLane {
                lane: lane.to_string(),
                manifest,
            }],
            version: version.to_string(),
        };
        store.apply_repository(
            Some(&checkpointed.canonical_bytes()),
            &mutation,
            &Default::default(),
        )?;
    }
    crate::catalog::Catalog::open(store.root())?.complete_operation(session)?;
    let _ = fs::remove_dir_all(candidate_dir(store.root(), session));
    Ok(id)
}

/// Installed checkpoints — the VISIBLE set, read from the AUTHORITY rather than from the
/// rendering beside it. A crash mid-ingest must leave this unchanged.
pub fn installed(store: &Store) -> Vec<String> {
    let mut out: Vec<String> =
        match crate::storage::Census::open(store.root()).and_then(|census| census.projection()) {
            Ok(projection) => projection
                .releases
                .into_iter()
                .map(|release| release.manifest_sha256)
                .collect(),
            Err(_) => Vec::new(),
        };
    out.sort();
    out.dedup();
    out
}

// ---------------------------------------------------------------- executing a plan

#[derive(Debug)]
pub struct Outcome {
    pub header: Header,
    pub closure: Closure,
    pub manifest: Manifest,
    pub header_ref: ObjectRef,
    pub manifest_ref: ObjectRef,
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
    operation: &str,
) -> Result<Outcome> {
    plan.require_order()?;
    let t0 = Instant::now();
    let mut comps: Vec<(String, Vec<(String, Tensor)>)> = Vec::new();
    let mut closure = Closure::default();
    let mut encodings: Vec<EncodingSpec> = Vec::new();

    let (mut written, mut inherited, mut hashed, mut deduped) = (0u64, 0u64, 0u64, 0u64);
    let (mut segments, mut inlines, mut parts) = (0usize, 0usize, 0usize);

    // Resolved by DIGEST, never by alias: one alias groups several physical specs, and the
    // plan already selected exactly one PER TENSOR. Computed once, not per op.
    let by_id: Vec<(String, &EncodingSpec)> =
        specs.iter().map(|(_, s)| (s.object_id(), s)).collect();
    // One key index per source file, computed once: `SourceHeader::get` is linear, and a
    // per-op linear probe is O(n²) at MAX_TENSORS.
    let indexes: Vec<std::collections::HashMap<&str, &super::carrier::SourceTensor>> =
        files.iter().map(|f| f.header.index()).collect();

    for op in &plan.ops {
        let spec = match by_id.iter().find(|(id, _)| *id == op.encoding_id) {
            Some((_, s)) => (*s).clone(),
            None => {
                return refuse(
                    Code::UNKNOWN_ENCODING,
                    format!(
                        "{}: the plan selected spec {} ({}), which is not in the resolved set",
                        op.out_key, op.encoding_id, op.encoding
                    ),
                )
            }
        };
        let sid = spec.object_id();
        if closure.get(&sid).is_none() {
            encodings.push(spec.clone());
            closure.insert(spec.clone());
        }

        let mut tparts: Vec<(String, Part)> = Vec::new();
        for r in &op.roles {
            parts += 1;
            let what = format!("{}/{}#{}", op.component, op.out_key, r.role);
            let part = match &r.bytes {
                Bytes::Stream { file, key } => {
                    let f = &files[*file];
                    let t = match indexes[*file].get(key.as_str()).copied() {
                        Some(t) => t,
                        None => {
                            return refuse(
                                Code::MISSING_TENSOR,
                                format!("{what}: the plan names carrier key {key:?}, absent now"),
                            )
                        }
                    };
                    let (rp, abs) = carrier::resolve(&f.path, &f.header, t)?;
                    let mut region = carrier::Region::open(rp, abs, t.nbytes())?;
                    let w = ck::objectize_held(
                        store,
                        &what,
                        r.dtype,
                        r.shape.clone(),
                        &mut region,
                        Some(operation),
                    )?;
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
                    let t = match indexes[*file].get(key.as_str()).copied() {
                        Some(t) => t,
                        None => {
                            return refuse(
                                Code::MISSING_TENSOR,
                                format!("{what}: the plan names carrier key {key:?}, absent now"),
                            )
                        }
                    };
                    check_permute_size(&what, t.nbytes())?;
                    // STREAMED, not buffered: the member's runs are read straight out of the
                    // carrier in destination order, so a permute costs the same constant
                    // memory an identity stream does. Holding the fused tensor would make
                    // peak RSS a function of the artifact (231 MB on the real H3 QKV).
                    let (rp, abs) = carrier::resolve(&f.path, &f.header, t)?;
                    let mut region = carrier::RunRegion::open(rp, abs, xform.runs())?;
                    let w = ck::objectize_held(
                        store,
                        &what,
                        r.dtype,
                        r.shape.clone(),
                        &mut region,
                        Some(operation),
                    )?;
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
                    // VERIFIED INHERIT: the refs carry without re-streaming, but `contains`
                    // is A PRESENCE HINT AND NOTHING MORE (store.rs). Admission into a
                    // committed checkpoint takes a valid verification record or exactly one
                    // rehash — a truncated or swapped object refuses HERE, at execute,
                    // never at first read.
                    for s in part.segments() {
                        let rec = match store.record_valid(&s.sha256) {
                            Ok(rec) => rec,
                            Err(_) => {
                                let v = store.verify(&s.sha256).map_err(|e| Refusal {
                                    code: e.code,
                                    detail: format!("{what}: inherit-by-reference: {}", e.detail),
                                })?;
                                if let Verdict::CorruptRemoved { why } = v {
                                    return refuse(
                                        Code::OBJECT_CORRUPT,
                                        format!(
                                            "{what}: inherit-by-reference names {} whose \
                                             bytes disagree with their id and were removed: {why}",
                                            s.id()
                                        ),
                                    );
                                }
                                store.record_valid(&s.sha256).map_err(|why| Refusal {
                                    code: Code::OBJECT_CORRUPT,
                                    detail: format!(
                                        "{what}: {} rehashed clean but its record does not \
                                         stand: {why}",
                                        s.id()
                                    ),
                                })?
                            }
                        };
                        if rec.length != s.length {
                            return refuse(
                                Code::LENGTH_MISMATCH,
                                format!(
                                    "{what}: inherit-by-reference declares {} at {} B; the \
                                     verified store object is {} B",
                                    s.id(),
                                    s.length,
                                    rec.length
                                ),
                            );
                        }
                    }
                    let bytes = checked_bytes(&what, &r.shape, r.dtype)?;
                    inherited += bytes;
                    store.moved(bytes);
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

    encodings.sort_by_key(EncodingSpec::object_id);
    encodings.dedup_by(|a, b| a.object_id() == b.object_id());
    let header = Header {
        configs: configs.to_vec(),
        assets: Vec::new(),
        encodings,
        components: comps,
    };
    // Validation BEFORE anything is advertised: the candidate must be a legal checkpoint.
    header.validate(&closure)?;

    let header_bytes = header.canonical_bytes()?;
    let header_ref = store
        .put_stream_held(
            &mut header_bytes.as_slice(),
            Some(&ObjectRef::of(&header_bytes)),
            &Default::default(),
            Some(operation),
        )?
        .obj;
    let manifest = build_manifest(&header, &header_ref, &closure)?;
    let manifest_ref = store.put_manifest_held(&manifest, Some(operation))?.obj;

    let obs = vec![
        ("bytes_written".to_string(), written as i64),
        ("bytes_inherited".to_string(), inherited as i64),
        ("bytes_hashed".to_string(), hashed as i64),
        ("bytes_deduped".to_string(), deduped as i64),
        ("tensors".to_string(), plan.ops.len() as i64),
        ("parts".to_string(), parts as i64),
        ("segments".to_string(), segments as i64),
        ("inline_parts".to_string(), inlines as i64),
        (
            "markers_dropped".to_string(),
            plan.dropped_markers.len() as i64,
        ),
        ("roles_folded".to_string(), plan.folded_roles.len() as i64),
        ("carrier_header_bytes".to_string(), plan.header_bytes as i64),
        (
            "convert_elapsed_ms".to_string(),
            t0.elapsed().as_millis() as i64,
        ),
    ];

    Ok(Outcome {
        header,
        closure,
        manifest,
        header_ref,
        manifest_ref,
        obs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tensorfs-ingest-{name}-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ))
    }

    #[test]
    fn same_session_retry_claims_the_existing_root_and_changed_meaning_refuses() {
        let path = temporary("resume");
        let store = Store::init(&path).unwrap();
        let (first, live) = open_root(&store, "session", "tenant").unwrap();
        assert_eq!(
            open_root(&store, "session", "tenant").unwrap_err().code,
            Code::STORE_BUSY
        );
        drop(live);
        let (resumed, resumed_writer) = open_root(&store, "session", "tenant").unwrap();
        assert_eq!(resumed, first);
        drop(resumed_writer);
        assert_eq!(
            open_root(&store, "session", "other").unwrap_err().code,
            Code::TRANSACTION_CONFLICT
        );
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn session_has_no_clock_and_explicit_reap_waits_for_the_writer() {
        let path = temporary("explicit-reap");
        let store = Store::init(&path).unwrap();
        let (root, writer) = open_root(&store, "session", "tenant").unwrap();
        assert_eq!(
            root.canonical_bytes(),
            br#"{"candidates":[],"session":"session","tenant":"tenant"}"#
        );

        let (gone, kept) = reap(&store).unwrap();
        assert!(gone.is_empty());
        assert_eq!(kept, ["session"]);
        assert!(candidate_dir(&path, "session").is_dir());

        drop(writer);
        let (gone, kept) = reap(&store).unwrap();
        assert_eq!(gone, ["session"]);
        assert!(kept.is_empty());
        assert!(!candidate_dir(&path, "session").exists());
        let _ = fs::remove_dir_all(path);
    }
}
