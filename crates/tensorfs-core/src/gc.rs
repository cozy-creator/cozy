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
use std::time::Duration;

use crate::catalog::{Catalog, WriterGuard};
use crate::err::{Code, Refusal, Result};
use crate::ids::ObjectRef;
use crate::ingest::transaction as tx;
use crate::repository::{Repository, RepositoryName};
use crate::storage::{self, Census, GcDelete, HeldKey};
use crate::store::Store;

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    crate::store::classify_io(what, error)
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
    holds.extend(crate::object_roots::held_objects(store)?);
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
    // Not a root: a checkpoint here in part whose origin keeps it whole ([`crate::origin`]).
    let origins = || crate::origin::manifests(store);
    let passes: [&dyn Fn() -> Result<Vec<ObjectRef>>; 4] =
        [&repositories, &named_roots, &everything, &origins];
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

/// What the owner process's read leases pin (`Store::ensure_owned`).
fn pinned(store: &Store, owner: Option<&crate::meta::Owner>) -> Result<Vec<HeldKey>> {
    let Some(owner) = owner else {
        return Ok(vec![]);
    };
    owner
        .pinned()
        .into_iter()
        .map(|object| {
            let manifest = store.manifest_path(&object.sha256).is_file();
            Ok(HeldKey {
                key: if manifest {
                    storage::manifest_key(&object.sha256)?
                } else {
                    storage::blob_key(&object.sha256)?
                },
                kind: if manifest { "manifest" } else { "blob" }.into(),
                length: object.length,
                sha256: object.sha256,
            })
        })
        .collect()
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
    let owner = crate::meta::owner(root);
    let _planning = owner.as_ref().map(|owner| owner.gate.write().unwrap());
    let (mut holds, sessions) = session_holds(store)?;
    holds.extend(pinned(store, owner.as_deref())?);
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
    crate::origin::prune(store)?;
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

/// A managed store's roots, as one pass over its cache sees them.
struct Roots {
    repositories: Vec<(Repository, Vec<u8>)>,
    holds: Vec<HeldKey>,
    /// Cache roots a protected or held manifest names: in use.
    used: Vec<RepositoryName>,
    /// `used`, and every root transport did not write. Old roots without transport
    /// provenance are deliberately not migrated or inferred.
    keep: Vec<RepositoryName>,
}

fn roots(store: &Store, protected_manifests: &[String]) -> Result<Roots> {
    let mut protected = protected_manifests
        .iter()
        .map(|digest| {
            crate::ids::hex64(
                "protected manifest",
                digest.strip_prefix("sha256:").unwrap_or(digest),
            )
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let (mut holds, _) = session_holds(store)?;
    holds.extend(pinned(store, crate::meta::owner(store.root()).as_deref())?);
    protected.extend(
        holds
            .iter()
            .filter(|hold| hold.kind == "manifest")
            .map(|hold| hold.sha256.clone()),
    );
    let repositories = storage::read_repositories(store.root())?;
    let (mut used, mut keep) = (Vec::new(), Vec::new());
    for (repository, body) in &repositories {
        let named = repository
            .checkpoints
            .iter()
            .any(|checkpoint| protected.contains(&checkpoint.manifest.sha256));
        let cached = crate::cache_roots::matches(store, &repository.repo, body)?;
        if cached && named {
            used.push(repository.repo.clone());
        }
        if !cached || named {
            keep.push(repository.repo.clone());
        }
    }
    Ok(Roots {
        repositories,
        holds,
        used,
        keep,
    })
}

/// Drop the cache roots `victims` names and collect what only they named, on top of
/// `before`. The native lock covers the removal and the collection, new readers and
/// downloads included.
fn drop_cached(
    store: &Store,
    roots: &Roots,
    victims: &[RepositoryName],
    exclusive: &crate::catalog::RebuildGuard,
    before: Report,
) -> Result<Report> {
    for (repository, body) in &roots.repositories {
        if victims.contains(&repository.repo) {
            store.remove_cached_repository(body, &repository.repo, exclusive)?;
        }
    }
    let mut dropped = collect_locked(store, false, exclusive)?;
    dropped.reclaimed_bytes += before.reclaimed_bytes;
    dropped.reclaimed_blobs += before.reclaimed_blobs;
    dropped.reclaimed_manifests += before.reclaimed_manifests;
    dropped.scratch_reaped += before.scratch_reaped;
    Ok(dropped)
}

/// `collect_cached_for` one byte: the garbage, else the least recently used cache root.
pub fn collect_cached(root: &Path, protected_manifests: &[String]) -> Result<Report> {
    collect_cached_for(root, protected_manifests, 1)
}

/// Free `need` bytes of a managed store: ordinary garbage first, then the fewest
/// least-recently-used cache roots that cover the rest (`reclaim`'s planner). Roots that
/// cannot cover it all stay: the pass then frees the garbage and nothing else.
pub fn collect_cached_for(
    root: &Path,
    protected_manifests: &[String],
    need: u64,
) -> Result<Report> {
    let store = Store::open(root)?;
    let exclusive = WriterGuard::lock_rebuild(root)?;
    let collected = collect_locked(&store, false, &exclusive)?;
    if collected.reclaimed_bytes >= need.max(1) {
        return Ok(collected);
    }
    let need = need.max(1) - collected.reclaimed_bytes;
    let roots = roots(&store, protected_manifests)?;
    let plan = match crate::reclaim::plan_managed(&store, need, &roots.keep, &roots.holds) {
        Ok(plan) => plan,
        Err(error) if error.code == Code::RECLAIM_INSUFFICIENT => return Ok(collected),
        Err(error) => return Err(error),
    };
    // A bounded pass never removes an arbitrarily large jointly-deduplicated set.
    if plan.victims.len() > 16 {
        return Ok(collected);
    }
    let victims = plan
        .victims
        .into_iter()
        .map(|victim| RepositoryName::new(victim.org, victim.name))
        .collect::<Result<Vec<_>>>()?;
    drop_cached(&store, &roots, &victims, &exclusive, collected)
}

/// What `collect_cached_for` could free at most: the garbage and every cache root outside
/// `protected_manifests`. Deletes nothing.
pub fn reclaimable(root: &Path, protected_manifests: &[String]) -> Result<u64> {
    let store = Store::open(root)?;
    let roots = roots(&store, protected_manifests)?;
    crate::reclaim::droppable_bytes(&store, &roots.keep, &roots.holds)
}

/// The cache roots naming `manifests` were used now.
pub fn used(root: &Path, manifests: &[String]) -> Result<()> {
    let store = Store::open(root)?;
    for (repository, _) in storage::read_repositories(root)? {
        if repository.checkpoints.iter().any(|checkpoint| {
            manifests
                .iter()
                .any(|m| m.strip_prefix("sha256:").unwrap_or(m) == checkpoint.manifest.sha256)
        }) {
            crate::cache_roots::touch(&store, &repository.repo)?;
        }
    }
    Ok(())
}

/// A cache's TTL. The cache roots in use (`protected_manifests`, and what the store's
/// holds name) are stamped used now; every other cache root unused for `idle` is dropped
/// and collected, low disk or not. A pass that finds none idle takes no lock and walks no
/// object.
pub fn collect_idle(root: &Path, protected_manifests: &[String], idle: Duration) -> Result<Report> {
    let store = Store::open(root)?;
    let since = crate::cache_roots::since(&store)?;
    let expired = |roots: &Roots| -> Vec<RepositoryName> {
        let unused = |repo: &&RepositoryName| {
            crate::cache_roots::used(&store, repo)
                .and_then(|used| used.max(since).elapsed().ok())
                .is_some_and(|unused| unused > idle)
        };
        let all = roots.repositories.iter().map(|(repository, _)| &repository.repo);
        all.filter(|repo| !roots.keep.contains(*repo))
            .filter(unused)
            .cloned()
            .collect()
    };
    let seen = roots(&store, protected_manifests)?;
    for repo in &seen.used {
        crate::cache_roots::touch(&store, repo)?;
    }
    if expired(&seen).is_empty() {
        return Ok(Report::default());
    }
    let exclusive = WriterGuard::lock_rebuild(root)?;
    let roots = roots(&store, protected_manifests)?;
    drop_cached(&store, &roots, &expired(&roots), &exclusive, Report::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::{self, Meta};
    use crate::read;
    use crate::store::Fault;

    #[test]
    fn the_owners_gc_runs_beside_its_leases_and_keeps_what_they_pin() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-gc-pins-{}-{}",
            std::process::id(),
            meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let meta = Meta::open(&store).unwrap();
        let put = |bytes: &[u8]| {
            store
                .put_stream(&mut &bytes[..], None, &Fault::default())
                .unwrap()
                .obj
        };
        let leased = put(b"bytes a served layout reads");
        let garbage = put(b"bytes nothing references");
        let present = |object: &ObjectRef| store.object_path(&object.sha256).is_file();

        // A lease that is not the owner's excludes GC.
        let (lease, _) = read::acquire(&store, &meta, "m", vec![leased.clone()]).unwrap();
        assert_eq!(collect(&root, false).unwrap_err().code, Code::STORE_BUSY);
        lease.release(&meta).unwrap();

        // The owner's lease is a pin: GC runs, keeps it, and takes what nothing holds.
        let _owner = Store::ensure_owned(&root).unwrap();
        let (lease, _) = read::acquire(&store, &meta, "m", vec![leased.clone()]).unwrap();
        collect(&root, false).unwrap();
        assert!(present(&leased) && !present(&garbage));
        lease.recheck(&meta).unwrap();
        lease.release(&meta).unwrap();
        collect(&root, false).unwrap();
        assert!(!present(&leased));
        fs::remove_dir_all(root).unwrap();
    }
}
