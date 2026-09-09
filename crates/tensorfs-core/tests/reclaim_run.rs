//! Upstream-authorized reclaim and optional cache warming through real Stores and GC.
//! Cache contents never provide custody; shared roots and active readers remain protected.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use tensorfs_core::err::Code;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::manifest::{Draft, Entry};
use tensorfs_core::reclaim::{self, RunOptions};
use tensorfs_core::repo_cache::{CacheKind, CacheRead, RepoObjectCache};
use tensorfs_core::repository::{Mutation, RepositoryName};
use tensorfs_core::storage::Census;
use tensorfs_core::store::{Fault, Store};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-reclaim-run-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn blob(store: &Store, label: &str, length: usize) -> ObjectRef {
    let mut body = vec![0u8; length];
    for (at, byte) in body.iter_mut().enumerate() {
        *byte = (label.as_bytes()[at % label.len()] as usize).wrapping_add(at) as u8;
    }
    let want = ObjectRef::of(&body);
    store
        .put_stream(&mut body.as_slice(), Some(&want), &Fault::default())
        .unwrap();
    want
}

fn model(store: &Store, org: &str, name: &str, objects: &[(&str, ObjectRef)]) {
    let mut entries: Vec<(String, Entry)> = objects
        .iter()
        .map(|(path, object)| ((*path).to_string(), Entry::File(object.clone())))
        .collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let manifest = Draft { entries }.seal().unwrap();
    let put = store.put_manifest(&manifest).unwrap();
    store
        .apply_repository(
            None,
            &Mutation::PutCheckpoint {
                repo: RepositoryName::new(org, name).unwrap(),
                manifest: put.obj,
            },
            &Fault::default(),
        )
        .unwrap();
}

/// The bytes actually on disk under `blobs/` and `manifests/` — the third witness, read
/// from the filesystem rather than from any number the code produced.
fn on_disk(root: &Path) -> u64 {
    fn walk(path: &Path, total: &mut u64) {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match std::fs::symlink_metadata(&path) {
                Ok(meta) if meta.is_dir() => walk(&path, total),
                Ok(meta) if meta.is_file() => *total += meta.len(),
                _ => {}
            }
        }
    }
    let mut total = 0;
    walk(&root.join("blobs"), &mut total);
    walk(&root.join("manifests"), &mut total);
    total
}

struct Fixture {
    root: PathBuf,
    cache: PathBuf,
    store: Store,
    shared: ObjectRef,
    victim_only: ObjectRef,
}

/// `keeper` and `victim` share one object; each has one of its own. The shared object is
/// what every arm here is really about.
fn fixture(name: &str, with_cache: bool) -> Fixture {
    let root = temporary(name);
    let cache = temporary(&format!("{name}-cache"));
    std::fs::create_dir_all(&cache).unwrap();
    let mut store = Store::init(&root)
        .unwrap()
        .bind_disk_budget(Some(1 << 40))
        .unwrap();
    if with_cache {
        store = store.bind_repo_cache(Some(&cache)).unwrap();
    }
    let shared = blob(&store, "shared-base", 120_000);
    let keeper_only = blob(&store, "keeper-own", 30_000);
    let victim_only = blob(&store, "victim-own", 70_000);
    model(
        &store,
        "acme",
        "keeper",
        &[("base.bin", shared.clone()), ("own.bin", keeper_only)],
    );
    model(
        &store,
        "acme",
        "victim",
        &[
            ("base.bin", shared.clone()),
            ("own.bin", victim_only.clone()),
        ],
    );
    Fixture {
        root,
        cache,
        store,
        shared,
        victim_only,
    }
}

fn victim() -> Vec<RepositoryName> {
    vec![RepositoryName::new("acme", "victim").unwrap()]
}

#[test]
fn the_prediction_the_report_and_the_disk_agree_exactly_and_the_shared_object_survives() {
    let f = fixture("end-to-end", true);
    // The arm is worthless over a fixture with no sharing, so the sharing is asserted first.
    let census = Census::open(&f.root).unwrap();
    let index = census.holder_index(&[]).unwrap();
    let victim_position = index
        .position(&RepositoryName::new("acme", "victim").unwrap())
        .unwrap();
    let predicted = index.reclaimable(&BTreeSet::from([victim_position]));
    assert!(
        predicted > 0 && predicted < index.closure_bytes(victim_position),
        "the fixture shares nothing; predicted={predicted} closure={}",
        index.closure_bytes(victim_position)
    );

    let before = on_disk(&f.root);
    let report = reclaim::run(
        &f.store,
        &victim(),
        RunOptions {
            demote: true,
            assume_upstream: true,
        },
    )
    .unwrap();

    // THE SHARED OBJECT SURVIVED, and the retained model still resolves through it. This
    // is asserted FIRST on purpose: a silent corruption discovered later is the worst
    // failure shape here, so the arm should name it rather than name a byte total that
    // happens to differ because of it.
    assert!(
        f.store.contains(&f.shared.sha256),
        "an object the retained model still needs was deleted"
    );
    assert!(f.store.verify(&f.shared.sha256).is_ok());
    assert!(
        Census::open(&f.root)
            .unwrap()
            .gc_plan(&[])
            .unwrap()
            .is_empty(),
        "the retained model's own closure was left collectable"
    );

    // THREE WITNESSES, EXACTLY EQUAL.
    assert_eq!(report.reclaimed_bytes, predicted, "report vs prediction");
    assert_eq!(
        before - on_disk(&f.root),
        predicted,
        "the disk vs the prediction"
    );
    assert_eq!(report.roots_dropped, vec!["acme/victim".to_string()]);

    // Optional warming kept a verified cache copy. Upstream custody, declared above,
    // authorized reclamation; the cache only accelerates this subsequent read.
    assert!(report.demoted_bytes >= f.victim_only.length);
    assert!(!f.store.contains(&f.victim_only.sha256));
    let cache = RepoObjectCache::new(&f.cache);
    assert_eq!(
        cache
            .admit(&f.store, CacheKind::Blob, &f.victim_only)
            .unwrap(),
        CacheRead::Hit,
        "the evicted object did not come back off the cache"
    );
    assert!(f.store.contains(&f.victim_only.sha256));

    let _ = std::fs::remove_dir_all(f.root);
    let _ = std::fs::remove_dir_all(f.cache);
}

#[test]
fn bytes_that_cannot_be_got_back_are_kept_and_the_run_refuses() {
    // decisions.md 842's falsifier — "a pressure sweep evicting a local-only artifact" —
    // armed as a check rather than left as a hope. No cache bound, no --demote, no
    // declaration: the run must refuse and delete nothing.
    let f = fixture("unrecoverable", false);
    let before = on_disk(&f.root);
    let refusal = reclaim::run(&f.store, &victim(), RunOptions::default()).unwrap_err();
    assert_eq!(refusal.code, Code::DURABILITY_UNPROVEN, "{refusal}");
    assert_eq!(on_disk(&f.root), before, "the refusal still deleted bytes");
    assert!(f.store.contains(&f.victim_only.sha256));
    assert!(f
        .store
        .repository_path(&RepositoryName::new("acme", "victim").unwrap())
        .exists());

    // The other side: the caller's declaration is accepted, because only the caller can
    // see the hub. Same class of input as free space (decisions.md 241).
    let report = reclaim::run(
        &f.store,
        &victim(),
        RunOptions {
            demote: false,
            assume_upstream: true,
        },
    )
    .unwrap();
    assert!(report.reclaimed_bytes > 0);
    assert!(!f.store.contains(&f.victim_only.sha256));
    let _ = std::fs::remove_dir_all(f.root);
    let _ = std::fs::remove_dir_all(f.cache);
}

#[test]
fn disposable_cache_never_authorizes_local_reclaim() {
    use std::os::unix::fs::symlink;

    for state in ["corrupt", "symlink", "valid", "removed"] {
        let f = fixture(state, true);
        let cache = RepoObjectCache::new(&f.cache);
        let rows = Census::open(&f.root)
            .unwrap()
            .reclaim_plan(&victim(), &[])
            .unwrap();
        for row in rows {
            let kind = if row.kind == "manifest" {
                CacheKind::Manifest
            } else {
                CacheKind::Blob
            };
            let path = cache.path(kind, &row.sha256).unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let source = f.root.join(&row.key);
            match state {
                "corrupt" => std::fs::write(path, vec![0; row.length as usize]).unwrap(),
                "symlink" => symlink(source, path).unwrap(),
                _ => {
                    std::fs::copy(source, &path).unwrap();
                }
            }
        }
        if state == "removed" {
            // Another cache owner may drop the whole disposable copy at any time.
            std::fs::remove_dir_all(&f.cache).unwrap();
        }
        let before = on_disk(&f.root);
        let outcome = reclaim::run(&f.store, &victim(), RunOptions::default());
        assert!(
            outcome.is_err(),
            "{state} cache authorized deleting the local copy: {outcome:?}"
        );
        assert_eq!(outcome.unwrap_err().code, Code::DURABILITY_UNPROVEN);
        assert_eq!(on_disk(&f.root), before);
        assert!(f.store.repository_path(&victim()[0]).exists());
        assert!(f.store.open_verified(&f.victim_only.sha256).is_ok());
        std::fs::remove_dir_all(f.root).unwrap();
        let _ = std::fs::remove_dir_all(f.cache);
    }
}

#[test]
fn cache_warming_failure_does_not_block_explicit_upstream_reclaim() {
    let f = fixture("cache-unavailable", true);
    std::fs::remove_dir_all(&f.cache).unwrap();
    std::fs::write(&f.cache, b"not a directory").unwrap();
    let before = on_disk(&f.root);
    let refusal = reclaim::run(
        &f.store,
        &victim(),
        RunOptions {
            demote: true,
            assume_upstream: false,
        },
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::DURABILITY_UNPROVEN);
    assert_eq!(on_disk(&f.root), before);
    let report = reclaim::run(
        &f.store,
        &victim(),
        RunOptions {
            demote: true,
            assume_upstream: true,
        },
    )
    .unwrap();
    assert!(report.reclaimed_bytes > 0);
    assert_eq!(report.demoted_objects, 0);
    assert!(!f.store.contains(&f.victim_only.sha256));
    std::fs::remove_dir_all(f.root).unwrap();
    std::fs::remove_file(f.cache).unwrap();
}

#[test]
fn neither_phase_frees_anything_on_its_own() {
    // The tombstone chooses which ROOTS stop being named; the census decides which OBJECTS
    // that frees. Proving each half alone frees nothing is what shows the collection is
    // driven by reachability rather than by the caller's list.
    let f = fixture("phases", false);
    let before = on_disk(&f.root);

    // Collect with no tombstone.
    let collected = tensorfs_core::gc::collect(&f.root, false).unwrap();
    assert_eq!(collected.reclaimed_bytes, 0);
    assert_eq!(on_disk(&f.root), before);

    // Tombstone with no collect.
    let repo = RepositoryName::new("acme", "victim").unwrap();
    let observed = std::fs::read(f.store.repository_path(&repo)).unwrap();
    f.store
        .apply_repository(
            Some(&observed),
            &Mutation::DeleteRepository { repo },
            &Fault::default(),
        )
        .unwrap();
    assert_eq!(
        on_disk(&f.root),
        before,
        "removing the root moved bytes by itself"
    );

    // Only the pair frees, and it frees exactly what the census says.
    let predicted: u64 = Census::open(&f.root)
        .unwrap()
        .gc_plan(&[])
        .unwrap()
        .iter()
        .map(|row| row.length)
        .sum();
    assert!(predicted > 0);
    let collected = tensorfs_core::gc::collect(&f.root, false).unwrap();
    assert_eq!(collected.reclaimed_bytes, predicted);
    assert_eq!(before - on_disk(&f.root), predicted);
    assert!(f.store.contains(&f.shared.sha256), "the shared object went");
    let _ = std::fs::remove_dir_all(f.root);
    let _ = std::fs::remove_dir_all(f.cache);
}

#[test]
fn a_live_reader_holds_the_store_and_releasing_it_lets_the_run_through() {
    // `gc::collect` takes store-wide exclusivity and a read lease holds a shared lock on
    // the same file, so reclamation refuses beside a live reader NAMING the holder. That is
    // decisions.md 1135's exclusivity, unchanged — and it is the sharpest limitation of
    // this issue, so it is pinned in both directions rather than described.
    let f = fixture("busy", false);
    let meta = tensorfs_core::meta::Meta::open(&f.store).unwrap();
    let hold = meta.acquire_hold("read").unwrap();

    let refusal = reclaim::run(
        &f.store,
        &victim(),
        RunOptions {
            demote: false,
            assume_upstream: true,
        },
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::STORE_BUSY, "{refusal}");
    assert!(
        refusal.detail.contains("held by"),
        "the refusal must name the holder: {refusal}"
    );
    // NOTHING WAS DROPPED, so a retry after the reader leaves is a CLEAN retry. The bytes
    // surviving is not enough to prove that: if the tombstone had already run, the model
    // would be un-named, the space still occupied, and the retry would find no root to drop
    // and report success over an empty plan. So the repository document is checked too.
    assert!(f.store.contains(&f.victim_only.sha256));
    assert!(
        f.store
            .repository_path(&RepositoryName::new("acme", "victim").unwrap())
            .exists(),
        "the run tombstoned the root and then refused, leaving a half-evicted store"
    );

    hold.release(&meta).unwrap();
    reclaim::run(
        &f.store,
        &victim(),
        RunOptions {
            demote: false,
            assume_upstream: true,
        },
    )
    .expect("the same run must succeed once the reader is gone");
    assert!(!f.store.contains(&f.victim_only.sha256));
    let _ = std::fs::remove_dir_all(f.root);
    let _ = std::fs::remove_dir_all(f.cache);
}

#[test]
fn a_store_with_no_disk_budget_cannot_be_reclaimed_even_by_asking() {
    let root = temporary("run-not-permitted");
    let store = Store::init(&root).unwrap();
    let object = blob(&store, "obj", 1_000);
    model(&store, "acme", "victim", &[("o.bin", object.clone())]);
    let refusal = reclaim::run(
        &store,
        &victim(),
        RunOptions {
            demote: false,
            assume_upstream: true,
        },
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::RECLAIM_NOT_PERMITTED, "{refusal}");
    assert!(store.contains(&object.sha256));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn one_orphan_blob_does_not_refuse_the_whole_run_and_is_never_pushed_to_the_cache() {
    // FOUND BY REVIEW, NOT BY A FIXTURE. Bytes that are ALREADY unreferenced are what a
    // plain `tfs gc` frees whether or not anybody evicts a model. Asking them to prove a
    // durable upstream made a single orphan blob refuse every reclaim run on the store —
    // a false refusal — and demoting them would push bytes no model names onto a SHARED
    // volume cache that has no garbage collector by design.
    let f = fixture("orphan", true);
    let orphan = blob(&f.store, "nobody-names-me", 45_000);
    assert!(
        !Census::open(&f.root)
            .unwrap()
            .gc_plan(&[])
            .unwrap()
            .is_empty(),
        "the fixture has no orphan; the arm would be vacuous"
    );

    let report = reclaim::run(
        &f.store,
        &victim(),
        RunOptions {
            demote: true,
            assume_upstream: true,
        },
    )
    .expect("an orphan blob must not refuse the run");

    // The orphan is collected, because phase 3 is the ordinary sweep...
    assert!(!f.store.contains(&orphan.sha256));
    // ...but it never reached the shared cache, because nothing was evicting it.
    let cached = RepoObjectCache::new(&f.cache)
        .path(CacheKind::Blob, &orphan.sha256)
        .unwrap();
    assert!(
        !cached.exists(),
        "an object no model names was pushed onto the shared volume cache"
    );
    // The victim's own bytes DID get demoted, so the arm is not passing by demoting nothing.
    let demoted = RepoObjectCache::new(&f.cache)
        .path(CacheKind::Blob, &f.victim_only.sha256)
        .unwrap();
    assert!(demoted.exists(), "the eviction demoted nothing at all");
    assert!(report.reclaimed_bytes > orphan.length);

    let _ = std::fs::remove_dir_all(f.root);
    let _ = std::fs::remove_dir_all(f.cache);
}
