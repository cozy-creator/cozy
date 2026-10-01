//! Local reclamation — the act that removes what nothing on disk names any more.
//!
//! Owner ruling (2026-09-02): "we delete the manifest / cozytensors header; a periodic GC
//! process then finds unreferenced blobs (no local snapshot references them any more) — the
//! candidates — and deletes them. This should only use the file system itself (which has a
//! repo + manifest list), not the database. The database is not the source of truth. The
//! filesystem itself is self-describing."
//!
//! So reachability is computed from `repos/`, `manifests/` and `blobs/` alone, exactly the
//! census `gc plan` walks. `tensorfs.sqlite` is never consulted for liveness; its
//! verified-blob rows are repaired afterwards so no row outlives its object.
//!
//! Holds are the filesystem lock protocol, never a table:
//! - the exclusive recovery lock (`tmp/writers/recovery.lock`) excludes every live writer
//!   and read lease for the whole pass — a store with a live holder refuses `STORE_BUSY`
//!   naming the holder, and GC never runs beside a writer (tfs-020);
//! - an open ingest session root (`tmp/ingest/<session>/session.json`) names its candidate
//!   objects until `tfs ingest reap` abandons it explicitly, so the window between
//!   `ingest run` and `ingest install` — no live process, a dead writer marker — keeps its
//!   bytes. Those are the `kept` objects a report names.
//!
//! Crash safety: an object is unlinked before its catalog row is dropped, so a crash leaves
//! at most a row for an absent object, which the next pass prunes. A re-run converges: what
//! was unlinked is no longer in the census. Nothing under `repos/` is ever touched.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::catalog::{Catalog, WriterGuard};
use crate::err::{Code, Refusal, Result};
use crate::ids::ObjectRef;
use crate::ingest::transaction as tx;
use crate::storage::{self, Census, GcDelete, HeldKey};
use crate::store::Store;

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {error}", what.as_ref()),
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    pub dry_run: bool,
    pub reclaimed_bytes: u64,
    pub reclaimed_blobs: u64,
    pub reclaimed_manifests: u64,
    /// Unreferenced by every repository, kept because an open ingest session names them.
    pub kept_bytes: u64,
    pub kept_objects: u64,
    pub sessions: Vec<String>,
    /// Orphan `put-` temps and dead lease markers removed: scratch nothing live holds.
    pub scratch_reaped: u64,
}

impl Report {
    pub fn line(&self) -> String {
        let verb = if self.dry_run {
            "would reclaim"
        } else {
            "reclaimed"
        };
        format!(
            "{verb} {} B in {} objects, kept {} B in {} objects under retained roots",
            self.reclaimed_bytes,
            self.reclaimed_blobs + self.reclaimed_manifests,
            self.kept_bytes,
            self.kept_objects,
        )
    }

    pub fn json(&self) -> Vec<u8> {
        use crate::canon::Value;
        crate::canon::write(&Value::obj(vec![
            ("dry_run", Value::Bool(self.dry_run)),
            ("kept_bytes", Value::uint(self.kept_bytes)),
            ("kept_objects", Value::uint(self.kept_objects)),
            ("reclaimed_blobs", Value::uint(self.reclaimed_blobs)),
            ("reclaimed_bytes", Value::uint(self.reclaimed_bytes)),
            ("reclaimed_manifests", Value::uint(self.reclaimed_manifests)),
            (
                "sessions",
                Value::arr(
                    self.sessions
                        .iter()
                        .map(|s| Value::str(s.as_str()))
                        .collect(),
                ),
            ),
            ("scratch_reaped", Value::uint(self.scratch_reaped)),
        ]))
    }
}

/// The filesystem's holds: every candidate an open ingest session root names, PLUS every
/// object its conversion journals name. A candidate whose bytes sit in `manifests/` is a
/// manifest hold (its closure stays live through the plan's held-manifest walk); anything
/// else is a blob hold.
///
/// The journal half is what makes an IN-FLIGHT conversion survive a pass. A session whose
/// converter is still running has no candidates at all — the header and Manifest are written
/// last — so before the journal existed the whole in-flight output was unreferenced bytes
/// and the first pass reclaimed all of it. `catalog.hold()` looks like it prevented that and
/// never did: `tensorfs.sqlite` is not consulted for liveness (see the head of this file).
pub fn session_holds(store: &Store) -> Result<(Vec<HeldKey>, Vec<String>)> {
    let mut lines: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut sessions = Vec::new();
    for session in tx::sessions(store) {
        let root = tx::read_session(store.root(), &session)?;
        // An object the operation's publication custodian already holds is not in flight:
        // the journal stops naming it, so the local copy is ordinary unreferenced bytes.
        let custody = crate::ingest::custody::read_session(store.root(), &session)?;
        let mut journalled =
            crate::ingest::journal::session_objects(&tx::candidate_dir(store.root(), &session))?;
        journalled.retain(|object| !custody.contains(object));
        if root.candidates.is_empty() && journalled.is_empty() {
            continue;
        }
        sessions.push(session);
        for candidate in root.candidates.iter().chain(journalled.iter()) {
            let (kind, key) = if store.manifest_path(&candidate.sha256).is_file() {
                ("manifest", storage::manifest_key(&candidate.sha256)?)
            } else {
                ("blob", storage::blob_key(&candidate.sha256)?)
            };
            let line = crate::canon::write(&crate::canon::Value::obj(vec![
                ("key", crate::canon::Value::str(key.clone())),
                ("kind", crate::canon::Value::str(kind)),
                ("length", crate::canon::Value::uint(candidate.length)),
            ]));
            lines.insert(key, line);
        }
    }
    let mut holds = lines
        .values()
        .map(|line| HeldKey::parse_line(line))
        .collect::<Result<Vec<_>>>()?;
    holds.extend(crate::derived::retained_objects(store)?);
    holds.extend(crate::source_artifact::held_objects(store)?);
    holds.extend(crate::checkpoint_root::held_objects(store)?);
    holds.extend(crate::ensure::held_objects(store)?);
    holds.extend(crate::keyed_roots::held_objects(store)?);
    // A full snapshot obligation wins over a runtime-only pin of the same manifest.
    holds.sort_by(|left, right| {
        left.key
            .cmp(&right.key)
            .then_with(|| (left.kind == "cozytensors").cmp(&(right.kind == "cozytensors")))
    });
    if holds
        .windows(2)
        .any(|pair| pair[0].key == pair[1].key && pair[0].length != pair[1].length)
    {
        return Err(Refusal {
            code: crate::err::Code::LENGTH_MISMATCH,
            detail: "retained roots disagree about an object's length".into(),
        });
    }
    holds.dedup_by(|left, right| left.key == right.key);
    Ok((holds, sessions))
}

/// The `wanted` manifests that no root reclamation keeps names. Roots are read cheapest
/// first and the search stops once every wanted manifest is found, so a rooted source costs
/// the roots that name it rather than every checkpoint chain and parts journal the Store
/// has retained, which grow with its history (finding 50). Membership is retention, not
/// caller authorization. A consumer must acquire its live hold before checking roots so
/// GC cannot remove bytes during admission.
pub(crate) fn unretained_manifests(store: &Store, wanted: &[ObjectRef]) -> Result<Vec<ObjectRef>> {
    let manifests = |holds: Vec<HeldKey>| {
        holds
            .into_iter()
            .filter(HeldKey::is_manifest)
            .map(|hold| ObjectRef {
                sha256: hold.sha256,
                length: hold.length,
            })
    };
    let repositories = || -> Result<Vec<ObjectRef>> {
        Ok(storage::read_repositories(store.root())?
            .into_iter()
            .flat_map(|(repository, _)| repository.checkpoints)
            .map(|checkpoint| checkpoint.manifest)
            .collect())
    };
    let named_roots = || -> Result<Vec<ObjectRef>> {
        let mut named = crate::derived::rooted_manifests(store)?;
        named.extend(manifests(crate::source_artifact::held_objects(store)?));
        named.extend(manifests(crate::checkpoint_root::held_objects(store)?));
        named.extend(manifests(crate::keyed_roots::held_objects(store)?));
        Ok(named)
    };
    let everything =
        || -> Result<Vec<ObjectRef>> { Ok(manifests(session_holds(store)?.0).collect()) };
    let passes: [&dyn Fn() -> Result<Vec<ObjectRef>>; 3] =
        [&repositories, &named_roots, &everything];
    let mut missing = wanted.to_vec();
    for pass in passes {
        if missing.is_empty() {
            break;
        }
        let named = pass()?;
        missing.retain(|manifest| !named.contains(manifest));
    }
    Ok(missing)
}

/// One complete pass: exclusivity, orphan temps, census, plan, delete, catalog repair.
pub fn collect(root: &Path, dry_run: bool) -> Result<Report> {
    let store = Store::open(root)?;
    let exclusive = WriterGuard::lock_rebuild(root)?;
    collect_locked(&store, dry_run, &exclusive)
}

pub(crate) fn collect_locked(
    store: &Store,
    dry_run: bool,
    _exclusive: &crate::catalog::RebuildGuard,
) -> Result<Report> {
    let root = store.root();
    let mut report = Report {
        dry_run,
        ..Report::default()
    };
    if !dry_run {
        // Under exclusivity every remaining put- temp and lease marker belongs to a dead
        // process: the kernel released its lock with it.
        report.scratch_reaped = store.reap()?.0 as u64;
        report.scratch_reaped += crate::cache_roots::reap(store, 128)?;
        if let Ok(meta) = crate::meta::Meta::open(store) {
            report.scratch_reaped += meta.reap_holds()?.len() as u64;
        }
    }
    let (holds, sessions) = session_holds(store)?;
    report.sessions = sessions;
    let census = Census::open(root)?;
    let plan = census.gc_plan(&holds)?;
    if !holds.is_empty() {
        let planned: BTreeSet<&str> = plan.iter().map(|row| row.key.as_str()).collect();
        for row in census.gc_plan(&[])? {
            if !planned.contains(row.key.as_str()) {
                report.kept_bytes += row.length;
                report.kept_objects += 1;
            }
        }
    }
    for row in &plan {
        report.reclaimed_bytes += row.length;
        match row.kind {
            "blob" => report.reclaimed_blobs += 1,
            _ => report.reclaimed_manifests += 1,
        }
    }
    if dry_run {
        return Ok(report);
    }
    let catalog = Catalog::open(root).ok();
    let mut parents = BTreeSet::new();
    for row in &plan {
        remove(root, row)?;
        if row.kind == "blob" {
            if let Some(catalog) = &catalog {
                catalog.drop_verification(&row.sha256);
            }
        }
        if let Some(parent) = Path::new(&row.key).parent() {
            parents.insert(root.join(parent));
        }
    }
    for parent in parents.iter().rev() {
        // An emptied fanout directory has no reason to exist; a non-empty one is left.
        let _ = fs::remove_dir(parent);
        if let Some(shard) = parent.parent() {
            let _ = fs::remove_dir(shard);
        }
    }
    if let Some(catalog) = &catalog {
        catalog.prune_verifications(|sha256| store.object_path(sha256).is_file())?;
    }
    Ok(report)
}

fn remove(root: &Path, row: &GcDelete) -> Result<()> {
    let path = root.join(&row.key);
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.len() != row.length => {
            return Err(Refusal {
                code: Code::LENGTH_MISMATCH,
                detail: format!("{} changed length before delete; rerun", row.key),
            })
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(io(format!("stat {}", row.key), error)),
    }
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io(format!("remove {}", row.key), error)),
    }
}

/// One pressure pass on an explicitly managed store: first ordinary garbage, then the
/// smallest useful cache-root drop chosen by the existing dedup-aware planner. The owner
/// keeps its placement generation stable through this call. The native lock covers the
/// entire selection/removal/collection transition, including new readers and downloads.
/// Old roots without transport provenance are deliberately not migrated or inferred.
pub fn collect_cached(root: &Path, protected_manifests: &[String]) -> Result<Report> {
    collect_cached_for(root, protected_manifests, 1)
}

/// The same pass, dropping the cache roots whose marginal bytes cover `need` together.
pub fn collect_cached_for(
    root: &Path,
    protected_manifests: &[String],
    need: u64,
) -> Result<Report> {
    let protected_manifests = protected_manifests
        .iter()
        .map(|digest| {
            crate::ids::hex64(
                "protected manifest",
                digest.strip_prefix("sha256:").unwrap_or(digest),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let store = Store::open(root)?;
    let exclusive = WriterGuard::lock_rebuild(root)?;
    let collected = collect_locked(&store, false, &exclusive)?;
    if collected.reclaimed_bytes >= need.max(1) {
        return Ok(collected);
    }
    let need = need.max(1) - collected.reclaimed_bytes;
    let (holds, _) = session_holds(&store)?;
    let mut protected: BTreeSet<&str> = protected_manifests.iter().map(String::as_str).collect();
    protected.extend(
        holds
            .iter()
            .filter(|hold| hold.kind == "manifest")
            .map(|hold| hold.sha256.as_str()),
    );
    let repositories = storage::read_repositories(root)?;
    let mut keep = Vec::new();
    for (repository, body) in &repositories {
        if repository
            .checkpoints
            .iter()
            .any(|checkpoint| protected.contains(checkpoint.manifest.sha256.as_str()))
            || !crate::cache_roots::matches(&store, &repository.repo, body)?
        {
            keep.push(repository.repo.clone());
        }
    }
    let plan = match crate::reclaim::plan_managed(&store, need, &keep, &holds) {
        Ok(plan) => plan,
        Err(error) if error.code == Code::RECLAIM_INSUFFICIENT => return Ok(collected),
        Err(error) => return Err(error),
    };
    // A bounded pass never removes an arbitrarily large jointly-deduplicated set.
    if plan.victims.len() > 16 {
        return Ok(collected);
    }
    for victim in plan.victims {
        let (repository, body) = repositories
            .iter()
            .find(|(repository, _)| {
                repository.repo.org == victim.org && repository.repo.name == victim.name
            })
            .expect("planner only names census repositories");
        store.remove_cached_repository(body, &repository.repo, &exclusive)?;
    }
    let mut dropped = collect_locked(&store, false, &exclusive)?;
    dropped.reclaimed_bytes += collected.reclaimed_bytes;
    dropped.reclaimed_blobs += collected.reclaimed_blobs;
    dropped.reclaimed_manifests += collected.reclaimed_manifests;
    dropped.scratch_reaped += collected.scratch_reaped;
    Ok(dropped)
}
