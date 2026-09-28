//! The ingest TRANSACTION: everything a border run produces lands under a temporary
//! candidate directory under `tmp/`, and a separate explicit act installs it.
//!
//! The property this file exists to hold: **a crash, a timeout, a rejection or a walked-away
//! session leaves no visible checkpoint.** Candidate objects may well be in the CAS — they
//! are content-addressed and harmless there — but nothing NAMES them except a candidate root.
//! `install` is the coordinator's act, performed against a receipt that binds the exact
//! subject/manifest/header triple, and it is the only repository-release writer.

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::canon::{as_arr, Fields, Value};
use crate::checkpoint::{self as ck, build_manifest};
use crate::dtype::{checked_bytes, Dtype};
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Closure, Header, Part, Tensor};
use crate::ids::{ascii_name, hex64, Doc, ObjectRef, Plain};
use crate::limits;
use crate::manifest::Manifest;
use crate::repo_cache::{CacheKind, Mirror};
use crate::repository::{Mutation, ReleaseLane, RepositoryName};
use crate::spec::EncodingSpec;
use crate::store::{Store, Verdict};

use super::carrier::{self, SourceHeader};
use super::convert::{check_permute_size, Bytes, Plan, RolePlan};
use super::journal::{self, Journal};
use super::narrow::Narrow;

/// Store one role from its source bytes, narrowing when the plan declares a lane dtype the
/// carrier does not have (`Converter::lane`).
fn store_role<R: Read>(
    store: &Store,
    what: &str,
    role: &RolePlan,
    from: Dtype,
    mut src: R,
    operation: &str,
) -> Result<ck::StoredPart> {
    if from == role.dtype {
        return ck::store_part_held(
            store,
            what,
            role.dtype,
            role.shape.clone(),
            &mut src,
            Some(operation),
        );
    }
    let narrowed = super::narrow::narrows(from, role.dtype)
        .then(|| Narrow::new(src, from))
        .flatten();
    let Some(mut narrow) = narrowed else {
        return refuse(
            Code::DTYPE_MISMATCH,
            format!(
                "{what}: no reviewed narrowing from {} to {}",
                from.name(),
                role.dtype.name()
            ),
        );
    };
    let stored = ck::store_part_held(
        store,
        what,
        role.dtype,
        role.shape.clone(),
        &mut narrow,
        Some(operation),
    );
    match narrow.overflow {
        Some(element) if stored.is_err() => refuse(
            Code::NUMBER_RANGE,
            format!(
                "{what}: element {element} is a finite {} value outside the f16 range",
                from.name()
            ),
        ),
        _ => stored,
    }
}

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
    pub repo: RepositoryName,
    pub version: &'a str,
    pub lane: &'a str,
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
        repo,
        version,
        lane,
        observed_repository,
    } = request;
    receipt.binds(subject, snap, header, stamp)?;
    let walk = ck::walk(store, snap)?;
    // Containment is not residency: a root never commits over missing objects (tfs-020).
    walk.require_resident(store)?;
    let id = snap.manifest_id();
    read_session(store.root(), session)?;
    let manifest = ObjectRef {
        sha256: id.trim_start_matches("sha256:").to_string(),
        length: snap.canonical_bytes().len() as u64,
    };
    let local_mutation = if repo.org == "local" {
        if lane != "local" {
            return refuse(Code::KEY_GRAMMAR, "local ingest requires lane local");
        }
        hex64("local source selection", version)?;
        Some(Mutation::ReplaceLocal {
            repo: repo.clone(),
            manifest: manifest.clone(),
            version: version.to_string(),
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
                    manifest: manifest.clone(),
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
                extra: Default::default(),
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

/// Which carriers this pass may read.
///
/// `Landed` is the per-file pipeline's argument: an op every one of whose roles reads a
/// carrier that has arrived is converted and journalled NOW, and the rest are deferred to a
/// later pass. It is what makes a source carrier transient by construction — converted, then
/// droppable — instead of something the whole 210 GB selection has to be alive for at once.
#[derive(Debug, Clone, Copy)]
pub enum Carriers<'a> {
    /// Every file in `files` is present. The whole-selection run.
    All,
    /// `landed[i]` says whether `files[i]` is readable right now.
    Landed(&'a [bool]),
    /// Physical bodies, after an index has resolved each tensor onto its shard.
    Ready(&'a BTreeSet<PathBuf>),
}

impl Carriers<'_> {
    fn has(&self, file: usize, physical: &Path) -> bool {
        match self {
            Carriers::All => true,
            Carriers::Landed(landed) => landed.get(file).copied().unwrap_or(false),
            Carriers::Ready(paths) => paths.contains(physical),
        }
    }
}

/// What ONE conversion pass did. The byte counters are THIS PASS's work, never the
/// artifact's totals: that is the whole measurement a resume exists to change.
#[derive(Debug, Default, Clone)]
pub struct Progress {
    pub converted_roles: usize,
    pub resumed_roles: usize,
    pub deferred_ops: usize,
    pub written: u64,
    pub inherited: u64,
    pub hashed: u64,
    pub deduped: u64,
    /// Every physical carrier a not-yet-journalled op still needs, including a landed
    /// sibling of an absent carrier. The caller may retire only the complement.
    pub needed: BTreeSet<PathBuf>,
}

impl Progress {
    /// Every op the plan names has a journalled part. Only then may a header exist.
    pub fn complete(&self) -> bool {
        self.deferred_ops == 0
    }
}

/// Shared across every profile in one source-preparation call. A journal op stays atomic;
/// an oversized first op is permitted, and its measured size is reported to the caller.
#[derive(Debug)]
pub struct Budget {
    limit: u64,
    used: u64,
    stopped: bool,
    pub largest_op: u64,
    strict: bool,
    pub required_write_bytes: u64,
}

impl Budget {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            used: 0,
            stopped: false,
            largest_op: 0,
            strict: false,
            required_write_bytes: 0,
        }
    }
    pub fn bounded(limit: u64) -> Self {
        Self {
            strict: true,
            ..Self::new(limit)
        }
    }
    fn take(&mut self, bytes: u64) -> bool {
        if self.stopped {
            return false;
        }
        if self.strict && self.used == 0 && bytes > self.limit {
            self.required_write_bytes = bytes;
            self.stopped = true;
            return false;
        }
        if self.used > 0 && bytes > self.limit.saturating_sub(self.used) {
            self.stopped = true;
            return false;
        }
        self.used = self.used.saturating_add(bytes);
        true
    }
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

fn journal_of(store: &Store, plan: &Plan, files: &[SourceFile], session: &str) -> Result<Journal> {
    ensure_session_root(store.root(), session, "_tensorfs")?;
    let headers: Vec<(PathBuf, &SourceHeader)> = files
        .iter()
        .map(|file| (file.path.clone(), &file.header))
        .collect();
    Journal::open(
        &candidate_dir(store.root(), session),
        &journal::plan_digest(plan, &headers),
    )
}

/// Convert every op whose carriers have landed, journalling each accepted op as it is
/// produced, and REUSING every op a previous run already journalled without reading a byte.
///
/// Streaming by construction, unchanged: an identity-class role is copied through a fixed
/// buffer and never resident, so peak RSS is a function of the buffer and the header, not of
/// the artifact. What is new is that peak *lost work* is now also a function of neither.
/// **`mirror` is the durability half, and it is offered to per OP rather than per run.**
/// The conversion journal above makes an op's MEANING durable on this pod's disk; the mirror
/// makes its BYTES durable off this pod entirely, so a reclaimed rental costs the
/// re-conversion of one checkpoint interval instead of the whole artifact. The offer never
/// blocks and never fails (see [`Mirror`]), so a slow, stalled or absent mount changes what
/// this pass COSTS and never what it produces. `None` is a conversion with no cache, which is
/// every local ingest.
#[allow(clippy::too_many_arguments)]
pub fn convert(
    store: &Store,
    plan: &Plan,
    files: &[SourceFile],
    journal: &mut Journal,
    operation: &str,
    carriers: Carriers<'_>,
    mirror: Option<&Mirror>,
) -> Result<Progress> {
    convert_bounded(
        store,
        plan,
        files,
        journal,
        operation,
        carriers,
        mirror,
        None,
        &super::custody::Custody::default(),
    )
}

#[allow(clippy::too_many_arguments)]
fn convert_bounded(
    store: &Store,
    plan: &Plan,
    files: &[SourceFile],
    journal: &mut Journal,
    operation: &str,
    carriers: Carriers<'_>,
    mirror: Option<&Mirror>,
    mut budget: Option<&mut Budget>,
    custody: &super::custody::Custody,
) -> Result<Progress> {
    plan.require_order()?;
    let mut progress = Progress::default();

    // One key index per source file, computed once: `SourceHeader::get` is linear, and a
    // per-op linear probe is O(n²) at MAX_TENSORS.
    let indexes: Vec<std::collections::HashMap<&str, &super::carrier::SourceTensor>> =
        files.iter().map(|f| f.header.index()).collect();

    for op in &plan.ops {
        let roles: Vec<String> = op.roles.iter().map(|r| r.role.clone()).collect();
        if let Some(done) = journal.op(&op.component, &op.out_key, &roles) {
            // A journalled part carries only if its objects still stand. This is
            // `Bytes::Inherit`'s existing law: `contains` is a presence hint and nothing
            // more, so admission takes a standing verification record or one rehash. A part
            // the operation's publication custodian already holds stands without local bytes.
            if done.iter().all(|(_, part)| {
                custody.holds_part(part) || journal::segments_still_stand(store, part)
            }) {
                progress.resumed_roles += roles.len();
                journal.resumed += roles.len();
                continue;
            }
        }
        let mut dependencies = Vec::new();
        for role in &op.roles {
            let (file, key) = match &role.bytes {
                Bytes::Stream { file, key } | Bytes::Permute { file, key, .. } => (*file, key),
                Bytes::Inherit { .. } => continue,
            };
            let source = &files[file];
            let tensor = indexes[file].get(key.as_str()).ok_or_else(|| Refusal {
                code: Code::MISSING_TENSOR,
                detail: format!(
                    "{}/{}: source key {key:?} is absent",
                    op.component, op.out_key
                ),
            })?;
            let (path, _) = carrier::resolve(&source.path, &source.header, tensor)?;
            dependencies.push((file, path.to_path_buf()));
        }
        let ready = dependencies
            .iter()
            .all(|(file, path)| carriers.has(*file, path));
        let bytes = op.roles.iter().try_fold(0u64, |sum, role| {
            let next = if matches!(role.bytes, Bytes::Inherit { .. }) {
                0
            } else {
                checked_bytes("conversion op", &role.shape, role.dtype)?
            };
            sum.checked_add(next).ok_or_else(|| Refusal {
                code: Code::COUNT_CAP,
                detail: "conversion op byte count overflows".into(),
            })
        })?;
        if !ready
            || budget
                .as_deref_mut()
                .is_some_and(|budget| !budget.take(bytes))
        {
            progress
                .needed
                .extend(dependencies.into_iter().map(|(_, path)| path));
            progress.deferred_ops += 1;
            continue;
        }
        // Retained raw source artifacts remain part of occupancy while new tensor
        // groups are produced. Completed journal groups passed the skip above.
        store.admits(bytes)?;
        let written_before = progress.written;

        let mut tparts: Vec<(String, Part)> = Vec::new();
        for r in &op.roles {
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
                    let region = carrier::Region::open(rp, abs, t.nbytes())?;
                    let w = store_role(store, &what, r, t.dtype, region, operation)?;
                    progress.written += w.bytes;
                    progress.hashed += w.bytes;
                    progress.deduped += w.deduped;
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
                    let w = if xform.streams() {
                        let region = carrier::RunRegion::open(rp, abs, xform.runs())?;
                        store_role(store, &what, r, t.dtype, region, operation)?
                    } else {
                        // Bounded by `check_permute_size`: read once, reorder in memory.
                        let mut source = Vec::with_capacity(t.nbytes() as usize);
                        carrier::Region::open(rp, abs, t.nbytes())?
                            .read_to_end(&mut source)
                            .map_err(|e| io(format!("{what}: read"), e))?;
                        let mut moved = Vec::new();
                        xform.apply(&source, &mut moved)?;
                        store_role(store, &what, r, t.dtype, moved.as_slice(), operation)?
                    };
                    progress.written += w.bytes;
                    progress.hashed += w.bytes;
                    progress.deduped += w.deduped;
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
                    progress.inherited += bytes;
                    store.moved(bytes);
                    (**part).clone()
                }
            };
            tparts.push((r.role.clone(), part));
        }
        // DURABLE BEFORE ADVERTISED: the op's meaning reaches the disk before the next op
        // starts, so a kill here costs one op and never the pass.
        journal.record(&op.component, &op.out_key, &tparts)?;
        progress.converted_roles += tparts.len();
        if let Some(budget) = budget.as_deref_mut() {
            budget.largest_op = budget.largest_op.max(progress.written - written_before);
        }
        // THE MEANING IS ON THIS DISK; NOW THE BYTES LEAVE IT. The order is the guarantee:
        // the journal names the part before anything publishes it, so a crash between the
        // two leaves a journalled op whose objects are only local — which the next run
        // re-converts or re-publishes, never trusts.
        if let Some(mirror) = mirror {
            for (_, part) in &tparts {
                mirror.offer_part(part);
            }
            // One snapshot of the whole op -> part map per checkpoint interval, not per op:
            // it is a file, and hashing it 3,699 times over an H3 tree would spend gigabytes
            // of SHA-256 restating what one snapshot per 4 GiB already says.
            if mirror.progress_due() {
                crate::durability::snapshot_conversion(store, mirror, journal, operation)?;
            }
            // One op is one unit of this pass's work. The chain wakes on this and on nothing
            // else — never on a clock — so a link is taken because bytes became durable.
            mirror.at_unit();
        }
    }
    // The LAST snapshot, whatever the interval says — the same reasoning the last chain link
    // is taken under. A pass that ends part-way into an interval has those ops journalled,
    // and publishing the objects without the map that names them would make a resuming pod
    // re-convert work whose bytes it is already holding.
    if let Some(mirror) = mirror {
        if progress.converted_roles > 0 {
            crate::durability::snapshot_conversion(store, mirror, journal, operation)?;
            mirror.at_unit();
        }
    }
    Ok(progress)
}

/// Build the candidate header and Manifest out of the journal.
///
/// THE FORMAT CANNOT EXPRESS *INCOMPLETE*. A header over 44 of 48 shards is a structurally
/// valid header for a smaller model, and nothing downstream could tell it from a real one.
/// So the refusal lives here: every op the plan names must have a journalled part before any
/// header exists at all.
pub struct Finalize<'a> {
    pub plan: &'a Plan,
    pub journal: &'a Journal,
    pub specs: &'a [(String, EncodingSpec)],
    pub configs: &'a [(String, Vec<u8>)],
    pub operation: &'a str,
    /// What the pass that just ran did. The byte counters ride into the observations, so a
    /// resumed run reports the work IT did and not the artifact's totals.
    pub pass: &'a Progress,
    pub elapsed_ms: i64,
    /// The write-through, so the header and Manifest leave this pod with everything else.
    pub mirror: Option<&'a Mirror>,
}

pub fn finalize(store: &Store, request: Finalize<'_>) -> Result<Outcome> {
    let Finalize {
        plan,
        journal,
        specs,
        configs,
        operation,
        pass,
        elapsed_ms,
        mirror,
    } = request;
    plan.require_order()?;
    let mut comps: Vec<(String, Vec<(String, Tensor)>)> = Vec::new();
    let mut closure = Closure::default();
    let mut encodings: Vec<EncodingSpec> = Vec::new();
    let (mut segments, mut inlines, mut parts) = (0usize, 0usize, 0usize);

    // Resolved by DIGEST, never by alias: one alias groups several physical specs, and the
    // plan already selected exactly one PER TENSOR. Computed once, not per op.
    let by_id: Vec<(String, &EncodingSpec)> =
        specs.iter().map(|(_, s)| (s.object_id(), s)).collect();

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

        let roles: Vec<String> = op.roles.iter().map(|r| r.role.clone()).collect();
        let tparts = match journal.op(&op.component, &op.out_key, &roles) {
            Some(tparts) => tparts,
            None => {
                return refuse(
                    Code::VERIFIER_OUTPUT_INCOMPLETE,
                    format!(
                        "{}/{}: the conversion journal holds {} of {} ops; a CozyTensors \
                         header over part of a plan is a valid header for a DIFFERENT model, \
                         so none is written until every op has landed",
                        op.component,
                        op.out_key,
                        journal.len(),
                        plan.ops.len(),
                    ),
                )
            }
        };
        for (_, part) in &tparts {
            parts += 1;
            match part.segments().len() {
                0 => inlines += 1,
                n => segments += n,
            }
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
    let mut configs = configs.to_vec();
    configs.extend(super::convert::normalization_config(&plan.converter));
    configs.sort();
    let header = Header {
        configs,
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
    // The header and the Manifest go LAST and they go together: they are what makes the rest
    // walkable. With them on the cache a fresh pod restores the whole checkpoint from one
    // digest; without them it has the bytes and no way to name them.
    if let Some(mirror) = mirror {
        mirror.offer(CacheKind::Blob, &header_ref);
        mirror.offer(CacheKind::Manifest, &manifest_ref);
        mirror.at_unit();
    }
    // The two documents nothing else names yet. Written into the session root so the
    // filesystem hold protocol keeps them until the coordinator installs or releases.
    name_candidates(
        store.root(),
        operation,
        &[header_ref.clone(), manifest_ref.clone()],
    )?;

    let obs = vec![
        ("bytes_written".to_string(), pass.written as i64),
        ("bytes_inherited".to_string(), pass.inherited as i64),
        ("bytes_hashed".to_string(), pass.hashed as i64),
        ("bytes_deduped".to_string(), pass.deduped as i64),
        ("tensors".to_string(), plan.ops.len() as i64),
        ("parts".to_string(), parts as i64),
        ("segments".to_string(), segments as i64),
        ("inline_parts".to_string(), inlines as i64),
        ("roles_converted".to_string(), pass.converted_roles as i64),
        ("roles_resumed".to_string(), pass.resumed_roles as i64),
        (
            "markers_dropped".to_string(),
            plan.dropped_markers.len() as i64,
        ),
        ("roles_folded".to_string(), plan.folded_roles.len() as i64),
        ("carrier_header_bytes".to_string(), plan.header_bytes as i64),
        ("convert_elapsed_ms".to_string(), elapsed_ms),
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

/// Execute a plan through the real CAS, journalling every op as it lands.
///
/// The whole-selection entry point: convert what is not already journalled, then finalize.
/// Interrupt it anywhere and the next call re-converts only what the journal does not hold.
#[allow(clippy::too_many_arguments)]
pub fn execute(
    store: &Store,
    plan: &Plan,
    files: &[SourceFile],
    specs: &[(String, EncodingSpec)],
    configs: &[(String, Vec<u8>)],
    operation: &str,
    mirror: Option<&Mirror>,
) -> Result<Outcome> {
    let t0 = Instant::now();
    let mut journal = journal_of(store, plan, files, operation)?;
    let progress = convert(
        store,
        plan,
        files,
        &mut journal,
        operation,
        Carriers::All,
        mirror,
    )?;
    finalize(
        store,
        Finalize {
            plan,
            journal: &journal,
            specs,
            configs,
            operation,
            pass: &progress,
            elapsed_ms: t0.elapsed().as_millis() as i64,
            mirror,
        },
    )
}

/// Convert only what has landed, and finalize when the journal is complete.
///
/// This is the per-file pipeline's entry point: call it every time another carrier arrives.
/// `Ok(None)` means real work was done and durably recorded, and the artifact is still
/// waiting on carriers that have not landed.
#[allow(clippy::too_many_arguments)]
pub fn advance(
    store: &Store,
    plan: &Plan,
    files: &[SourceFile],
    specs: &[(String, EncodingSpec)],
    configs: &[(String, Vec<u8>)],
    operation: &str,
    carriers: Carriers<'_>,
    mirror: Option<&Mirror>,
) -> Result<(Progress, Option<Outcome>)> {
    let mut journal = journal_of(store, plan, files, operation)?;
    advance_journal(
        store,
        plan,
        files,
        specs,
        configs,
        operation,
        carriers,
        mirror,
        &mut journal,
        None,
        &super::custody::Custody::default(),
    )
}

/// The source coordinator supplies a journal keyed by immutable source identities rather
/// than local paths. Conversion and final completeness still have exactly one executor.
#[allow(clippy::too_many_arguments)]
pub fn advance_journal(
    store: &Store,
    plan: &Plan,
    files: &[SourceFile],
    specs: &[(String, EncodingSpec)],
    configs: &[(String, Vec<u8>)],
    operation: &str,
    carriers: Carriers<'_>,
    mirror: Option<&Mirror>,
    journal: &mut Journal,
    budget: Option<&mut Budget>,
    custody: &super::custody::Custody,
) -> Result<(Progress, Option<Outcome>)> {
    let t0 = Instant::now();
    let progress = convert_bounded(
        store, plan, files, journal, operation, carriers, mirror, budget, custody,
    )?;
    if !progress.complete() {
        return Ok((progress, None));
    }
    let outcome = finalize(
        store,
        Finalize {
            plan,
            journal,
            specs,
            configs,
            operation,
            pass: &progress,
            elapsed_ms: t0.elapsed().as_millis() as i64,
            mirror,
        },
    )?;
    Ok((progress, Some(outcome)))
}

/// Convert what has landed within the budget and never finalize. A custodied source
/// operation names no slot's header until every slot's plan is journalled: a finished
/// slot's candidate Manifest would hold its whole closure against GC while the others
/// still need the disk.
#[allow(clippy::too_many_arguments)]
pub fn convert_journal(
    store: &Store,
    plan: &Plan,
    files: &[SourceFile],
    operation: &str,
    carriers: Carriers<'_>,
    journal: &mut Journal,
    budget: Option<&mut Budget>,
    custody: &super::custody::Custody,
) -> Result<Progress> {
    convert_bounded(
        store, plan, files, journal, operation, carriers, None, budget, custody,
    )
}

/// Drop the conversion journal for one plan. The coordinator's act, after its result is
/// installed or released — a crash never reaches here, which is the point.
pub fn forget_journal(
    store: &Store,
    plan: &Plan,
    files: &[SourceFile],
    operation: &str,
) -> Result<()> {
    let headers: Vec<(PathBuf, &SourceHeader)> = files
        .iter()
        .map(|file| (file.path.clone(), &file.header))
        .collect();
    Journal::open(
        &candidate_dir(store.root(), operation),
        &journal::plan_digest(plan, &headers),
    )?
    .remove()
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
            String::from_utf8(root.canonical_bytes()).unwrap(),
            format!(
                r#"{{"candidates":[],"session":"session","tenant":"tenant","tensorfs":"{}"}}"#,
                crate::VERSION
            )
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
