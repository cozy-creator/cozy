//! Local provenance, not reachability: an exact repository body written by Hub transport.
//! Missing/mismatched receipts preserve roots. GC still owns every object decision.
//! A receipt's time is its root's last use.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::time::SystemTime;

use crate::err::{Refusal, Result};
use crate::repository::RepositoryName;
use crate::store::Store;

fn io(error: std::io::Error) -> Refusal {
    crate::store::classify_io("cache root provenance", error)
}

fn path(store: &Store, repo: &RepositoryName) -> PathBuf {
    store
        .root()
        .join("tmp/cache-roots")
        .join(&repo.org)
        .join(&repo.name)
}

pub(crate) fn matches(store: &Store, repo: &RepositoryName, body: &[u8]) -> Result<bool> {
    let path = path(store, repo);
    match fs::symlink_metadata(&path) {
        Ok(meta) if meta.file_type().is_file() && meta.len() == 64 => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(io(error)),
    }
    Ok(fs::read(path).map_err(io)? == crate::sha256::hex_digest(body).as_bytes())
}

/// The root was used now. Nothing to stamp without a receipt.
pub(crate) fn touch(store: &Store, repo: &RepositoryName) -> Result<()> {
    match OpenOptions::new().write(true).open(path(store, repo)) {
        Ok(receipt) => receipt.set_modified(SystemTime::now()).map_err(io),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io(error)),
    }
}

/// When the root was last pulled or `touch`ed; `None` without a receipt.
pub(crate) fn used(store: &Store, repo: &RepositoryName) -> Option<SystemTime> {
    fs::metadata(path(store, repo)).and_then(|m| m.modified()).ok()
}

/// When this store began stamping use, begun now if it has not. A root last used before
/// then counts as used then: an older store's roots are not idle the day they are first
/// swept.
pub(crate) fn since(store: &Store) -> Result<SystemTime> {
    let path = store.root().join("tmp/cache-roots.since");
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
    {
        Ok(file) => store.assign_new_metadata_owner(&file)?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(io(error)),
    }
    fs::metadata(path).and_then(|m| m.modified()).map_err(io)
}

// Both writers are called inside the existing repository CAS and native writer fence.
pub(crate) fn clear(store: &Store, repo: &RepositoryName) -> Result<()> {
    let path = path(store, repo);
    match fs::remove_file(&path) {
        Ok(()) => crate::store::fsync_dir(path.parent().unwrap()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io(error)),
    }
}

pub(crate) fn record(store: &Store, repo: &RepositoryName, body: &[u8]) -> Result<()> {
    let path = path(store, repo);
    let parent = path.parent().unwrap();
    store.prepare_owned_directory(parent)?;
    let temporary = parent.join(format!(
        ".cache-root-{}-{}",
        repo.name,
        crate::meta::now_nanos_unique()
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)
            .map_err(io)?;
        file.write_all(crate::sha256::hex_digest(body).as_bytes())
            .map_err(io)?;
        store.assign_new_metadata_owner(&file)?;
        file.sync_all().map_err(io)?;
        fs::rename(&temporary, &path).map_err(io)?;
        crate::store::fsync_dir(parent)
    })();
    let _ = fs::remove_file(temporary);
    result
}

// Called only under the native exclusive recovery lock. These small provenance facts
// have no lifetime beyond their exact repository; unknown repo bytes are not modified.
pub(crate) fn reap(store: &Store, limit: usize) -> Result<u64> {
    let base = store.root().join("tmp/cache-roots");
    let organizations = match fs::read_dir(&base) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(io(error)),
    };
    let (mut visited, mut removed) = (0, 0);
    for organization in organizations.take(limit) {
        let organization = organization.map_err(io)?;
        if !organization.file_type().map_err(io)?.is_dir() {
            continue;
        }
        let directory = organization.path();
        for entry in fs::read_dir(&directory).map_err(io)? {
            if visited >= limit {
                return Ok(removed);
            }
            visited += 1;
            let entry = entry.map_err(io)?;
            if !entry.file_type().map_err(io)?.is_file() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            // Our interrupted atomic-write temp; no native writer is live at this point.
            let stale = if name.starts_with(".cache-root-") {
                true
            } else if let Ok(repo) =
                RepositoryName::new(organization.file_name().to_string_lossy(), name.as_ref())
            {
                match fs::read(store.repository_path(&repo)) {
                    Ok(body) => !matches(store, &repo, &body)?,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
                    Err(error) => return Err(io(error)),
                }
            } else {
                false
            };
            if stale {
                fs::remove_file(entry.path()).map_err(io)?;
                removed += 1;
            }
        }
        crate::store::fsync_dir(&directory)?;
        if fs::remove_dir(&directory).is_ok() {
            crate::store::fsync_dir(&base)?;
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::WriterGuard;
    use crate::err::Code;
    use crate::ids::{Doc, ObjectRef};
    use crate::manifest::{Draft, Entry};
    use crate::repository::{Mutation, Repository};
    use crate::store::Fault;

    struct Fixture {
        store: Store,
    }
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "tensorfs-cache-root-{}-{}",
                std::process::id(),
                crate::meta::now_nanos_unique()
            ));
            Self {
                store: Store::init(&path).unwrap(),
            }
        }
        fn model(&self, name: &str, data: &[u8], cached: bool) -> (RepositoryName, ObjectRef) {
            let blob = self
                .store
                .put_stream(&mut &data[..], None, &Fault::default())
                .unwrap()
                .obj;
            let manifest = Draft {
                entries: vec![("weights".into(), Entry::File(blob))],
            }
            .seal()
            .unwrap();
            let manifest = self.store.put_manifest(&manifest).unwrap().obj;
            let repo = RepositoryName::new("org", name).unwrap();
            let mutation = Mutation::PutCheckpoint {
                repo: repo.clone(),
                manifest: manifest.clone(),
            };
            if cached {
                self.store
                    .apply_cached_repository(None, &[mutation], &Fault::default())
                    .unwrap();
            } else {
                self.store
                    .apply_repository(None, &mutation, &Fault::default())
                    .unwrap();
            }
            (repo, manifest)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(self.store.root());
        }
    }

    fn age(path: &std::path::Path, days: u64) {
        let then = SystemTime::now() - std::time::Duration::from_secs(days * 86_400);
        let file = OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(then).unwrap();
    }

    #[test]
    fn a_covering_drop_takes_the_least_recently_used_root_and_no_cover_takes_none() {
        let f = Fixture::new();
        let root = f.store.root();
        let (cold, _) = f.model("cold", b"cold", true);
        let (warm, _) = f.model("warm", b"a warm and much larger model", true);
        for stamp in [path(&f.store, &cold), f.store.repository_path(&cold)] {
            age(&stamp, 3);
        }
        let held = |repo: &RepositoryName| f.store.repository_path(repo).exists();
        let total = crate::gc::reclaimable(root, &[]).unwrap();
        assert!(total > 0 && held(&cold) && held(&warm), "reclaimable deletes nothing");
        // Both together do not cover the need: neither goes.
        let none = crate::gc::collect_cached_for(root, &[], total + 1).unwrap();
        assert_eq!(none.reclaimed_bytes, 0);
        assert!(held(&cold) && held(&warm));
        // Either covers one byte: the least recently used goes, not the larger.
        let first = crate::gc::collect_cached_for(root, &[], 1).unwrap();
        assert!(!held(&cold) && held(&warm));
        // Using a root moves it behind the others.
        let (fresh, fresh_manifest) = f.model("fresh", b"pulled last, used first", true);
        age(&path(&f.store, &warm), 2);
        age(&path(&f.store, &fresh), 5);
        age(&f.store.repository_path(&fresh), 5);
        crate::gc::used(root, &[format!("sha256:{}", fresh_manifest.sha256)]).unwrap();
        let second = crate::gc::collect_cached_for(root, &[], 1).unwrap();
        assert!(!held(&warm) && held(&fresh));
        assert_eq!(first.reclaimed_bytes + second.reclaimed_bytes, total);
    }

    #[test]
    fn an_idle_root_expires_while_used_owned_and_untracked_roots_stay() {
        let f = Fixture::new();
        let week = std::time::Duration::from_secs(7 * 86_400);
        let (idle, _) = f.model("idle", b"idle", true);
        let (served, served_manifest) = f.model("served", b"served", true);
        let (owned, _) = f.model("owned", b"owned", false);
        let held = |repo: &RepositoryName| f.store.repository_path(repo).exists();
        let sweep = |protected: &[String]| {
            crate::gc::collect_idle(f.store.root(), protected, week)
                .unwrap()
                .reclaimed_bytes
        };
        let serving = [served_manifest.sha256.clone()];
        for repo in [&idle, &served] {
            age(&path(&f.store, repo), 8);
        }
        // A store that only now begins stamping use counts its roots as used now.
        assert_eq!(sweep(&serving), 0);
        assert!(held(&idle));
        age(&f.store.root().join("tmp/cache-roots.since"), 9);
        assert!(sweep(&serving) > 0);
        assert!(!held(&idle) && held(&served) && held(&owned));
        // The passes stamped the root in use: no longer served, it is not idle yet.
        assert_eq!(sweep(&[]), 0);
        age(&path(&f.store, &served), 8);
        assert!(sweep(&[]) > 0);
        assert!(!held(&served) && held(&owned));
    }

    #[test]
    fn cached_pressure_keeps_owned_shared_and_protected_roots() {
        let f = Fixture::new();
        let (user, user_manifest) = f.model("user", b"shared", false);
        let (cached_shared, _) = f.model("shared", b"shared", true);
        let (active, active_manifest) = f.model("active", b"keep active", true);
        let (victim, victim_manifest) = f.model("victim", b"discard me", true);
        let report = crate::gc::collect_cached(f.store.root(), &[active_manifest.sha256]).unwrap();
        assert!(report.reclaimed_bytes > 0);
        assert!(f.store.repository_path(&user).exists());
        assert!(f.store.repository_path(&cached_shared).exists());
        assert!(f.store.repository_path(&active).exists());
        assert!(!f.store.repository_path(&victim).exists());
        assert!(f.store.read_manifest(&user_manifest).is_ok());
        assert!(f.store.read_manifest(&victim_manifest).is_err());
    }

    #[test]
    fn ordinary_and_mixed_repository_mutations_revoke_cache_ownership() {
        let f = Fixture::new();
        for cached in [true, false] {
            let (repo, manifest) = f.model(if cached { "cached" } else { "user" }, b"keep", cached);
            let body = fs::read(f.store.repository_path(&repo)).unwrap();
            let mutation = Mutation::PutCheckpoint {
                repo: repo.clone(),
                manifest,
            };
            // Even an ordinary no-op explicitly adopts that local root.
            f.store
                .apply_repository(Some(&body), &mutation, &Fault::default())
                .unwrap();
            f.store
                .apply_cached_repository(Some(&body), &[mutation], &Fault::default())
                .unwrap();
            assert!(!matches(&f.store, &repo, &body).unwrap());
        }
        assert_eq!(
            crate::gc::collect_cached(f.store.root(), &[])
                .unwrap()
                .reclaimed_bytes,
            0
        );
    }

    #[test]
    fn missing_mismatched_and_interrupted_receipts_never_authorize_eviction() {
        let f = Fixture::new();
        for name in ["missing", "mismatch", "interrupted"] {
            let (repo, _) = f.model(name, name.as_bytes(), true);
            if name == "missing" {
                clear(&f.store, &repo).unwrap();
            } else {
                fs::write(
                    path(&f.store, &repo),
                    if name == "mismatch" {
                        vec![b'0'; 64]
                    } else {
                        vec![b'0'; 3]
                    },
                )
                .unwrap();
            }
        }
        assert_eq!(
            crate::gc::collect_cached(f.store.root(), &[])
                .unwrap()
                .reclaimed_bytes,
            0
        );
    }

    #[test]
    fn live_writer_refuses_before_root_removal_and_later_reclaims() {
        let f = Fixture::new();
        let (repo, _) = f.model("writer", b"writer held", true);
        let guard = WriterGuard::acquire(f.store.root()).unwrap();
        assert_eq!(
            crate::gc::collect_cached(f.store.root(), &[])
                .unwrap_err()
                .code,
            Code::STORE_BUSY
        );
        assert!(f.store.repository_path(&repo).exists());
        drop(guard);
        assert!(
            crate::gc::collect_cached(f.store.root(), &[])
                .unwrap()
                .reclaimed_bytes
                > 0
        );
    }

    #[test]
    fn native_exclusive_fence_blocks_activation_and_preserves_generation() {
        let f = Fixture::new();
        let (repo, _) = f.model("race", b"race", true);
        let body = fs::read(f.store.repository_path(&repo)).unwrap();
        let exclusive = WriterGuard::lock_rebuild(f.store.root()).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let root = f.store.root().to_path_buf();
        let reader = std::thread::spawn(move || {
            let guard = WriterGuard::acquire(&root).unwrap();
            tx.send(()).unwrap();
            drop(guard);
        });
        assert!(rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());
        f.store
            .remove_cached_repository(&body, &repo, &exclusive)
            .unwrap();
        crate::gc::collect_locked(&f.store, false, &exclusive).unwrap();
        drop(exclusive);
        rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        reader.join().unwrap();
    }

    #[test]
    fn transport_replacement_cannot_mark_a_preexisting_mixed_repository() {
        let f = Fixture::new();
        let (repo, manifest) = f.model("mixed", b"author bytes", false);
        let body = fs::read(f.store.repository_path(&repo)).unwrap();
        let parsed = Repository::parse(&body).unwrap();
        assert_eq!(parsed.checkpoints[0].manifest, manifest);
        f.store
            .apply_cached_repository(
                Some(&body),
                &[Mutation::UpdateRelease {
                    repo: repo.clone(),
                    version: "1".into(),
                    expected_revision: 0,
                    remove: vec![],
                    set: vec![crate::repository::ReleaseLane {
                        extra: Default::default(),
                        lane: "bf16".into(),
                        manifest,
                    }],
                }],
                &Fault::default(),
            )
            .unwrap();
        let body = fs::read(f.store.repository_path(&repo)).unwrap();
        assert!(!matches(&f.store, &repo, &body).unwrap());
        assert_eq!(
            crate::gc::collect_cached(f.store.root(), &[])
                .unwrap()
                .reclaimed_bytes,
            0
        );
    }
    #[test]
    fn retained_ingest_candidate_and_live_native_reader_keep_the_root() {
        let f = Fixture::new();
        let (repo, manifest) = f.model("retained", b"sole retained bytes", true);
        let meta = crate::meta::Meta::open(&f.store).unwrap();
        let lease = meta.acquire_hold("read").unwrap();
        assert_eq!(
            crate::gc::collect_cached(f.store.root(), &[])
                .unwrap_err()
                .code,
            Code::STORE_BUSY
        );
        assert!(f.store.repository_path(&repo).exists());
        lease.release(&meta).unwrap();
        let (mut session, writer) =
            crate::ingest::transaction::open_root(&f.store, "retained-source", "owner").unwrap();
        session.candidates.push(manifest.clone());
        crate::ingest::transaction::write_session(f.store.root(), &session).unwrap();
        drop(writer); // Paused work remains retained even after its process's live lock ends.
        assert_eq!(
            crate::gc::collect_cached(f.store.root(), &[])
                .unwrap()
                .reclaimed_bytes,
            0
        );
        assert!(f.store.repository_path(&repo).exists());
        assert!(f.store.read_manifest(&manifest).is_ok());
    }
    #[test]
    fn orphan_provenance_cleanup_is_bounded_and_never_removes_a_repository() {
        let f = Fixture::new();
        let mut stale = Vec::new();
        for number in 0..5 {
            let (repo, _) = f.model(&format!("old-{number}"), &[number], true);
            fs::remove_file(f.store.repository_path(&repo)).unwrap();
            stale.push(path(&f.store, &repo));
        }
        let guard = WriterGuard::lock_rebuild(f.store.root()).unwrap();
        assert_eq!(reap(&f.store, 2).unwrap(), 2);
        assert_eq!(stale.iter().filter(|path| path.exists()).count(), 3);
        drop(guard);
        let (live, _) = f.model("live", b"live", true);
        let guard = WriterGuard::lock_rebuild(f.store.root()).unwrap();
        assert_eq!(reap(&f.store, 128).unwrap(), 3);
        assert!(f.store.repository_path(&live).exists());
        assert!(path(&f.store, &live).exists());
        drop(guard);
    }
}
