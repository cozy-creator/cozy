use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs_core::ids::ObjectRef;
use tensorfs_core::manifest::Manifest;
use tensorfs_core::repo_cache::{CacheKind, CacheRead, CacheWrite, RepoObjectCache};
use tensorfs_core::store::{Fault, Store};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-repo-cache-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn put_blob(store: &Store, bytes: &[u8]) -> ObjectRef {
    let object = ObjectRef::of(bytes);
    let mut source = bytes;
    store
        .put_stream(&mut source, Some(&object), &Fault::default())
        .unwrap()
        .obj
}

fn write(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

#[test]
fn layout_is_only_canonical_manifest_and_blob_fanout() {
    let root = temporary("layout");
    let cache = RepoObjectCache::new(&root);
    let digest = "0123456789abcdef".repeat(4);
    assert_eq!(
        cache.path(CacheKind::Blob, &digest).unwrap(),
        root.join("blobs/01/23").join(&digest)
    );
    assert_eq!(
        cache.path(CacheKind::Manifest, &digest).unwrap(),
        root.join("manifests/01/23").join(format!("{digest}.json"))
    );
    assert!(!root.exists(), "naming a cache must not initialize a Store");
}

#[test]
fn blobs_and_manifests_round_trip_through_normal_store_admission() {
    let root = temporary("round-trip");
    let source = Store::init(&root.join("source")).unwrap();
    let destination = Store::init(&root.join("destination")).unwrap();
    let cache = RepoObjectCache::new(root.join("cache"));

    let blob = put_blob(&source, b"large-model-segment");
    assert_eq!(
        cache
            .backfill_store(&source, CacheKind::Blob, &blob)
            .unwrap(),
        CacheWrite::Stored
    );
    write(
        &destination.blob_path(&blob.sha256),
        &vec![b'x'; blob.length as usize],
    );
    assert_eq!(
        cache.admit(&destination, CacheKind::Blob, &blob).unwrap(),
        CacheRead::Unavailable,
        "losing no-clobber to corrupt local bytes must not report a hit"
    );
    assert_eq!(
        cache.admit(&destination, CacheKind::Blob, &blob).unwrap(),
        CacheRead::Hit
    );
    let mut admitted = destination.open_verified(&blob.sha256).unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut admitted, &mut bytes).unwrap();
    assert_eq!(bytes, b"large-model-segment");
    assert_eq!(
        cache
            .backfill_store(&source, CacheKind::Blob, &blob)
            .unwrap(),
        CacheWrite::Present
    );

    let manifest = Manifest::from_files(vec![("model.bin".into(), blob.clone())]).unwrap();
    let manifest_ref = source.put_manifest(&manifest).unwrap().obj;
    assert_eq!(
        cache
            .backfill_store(&source, CacheKind::Manifest, &manifest_ref)
            .unwrap(),
        CacheWrite::Stored
    );
    assert_eq!(
        cache
            .admit(&destination, CacheKind::Manifest, &manifest_ref)
            .unwrap(),
        CacheRead::Hit
    );
    assert_eq!(destination.read_manifest(&manifest_ref).unwrap(), manifest);

    let cache_root = cache.root();
    assert!(!cache_root.join("tensorfs.sqlite").exists());
    assert!(!cache_root.join("repos").exists());
    assert!(!cache_root.join("tmp").exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn missing_and_corrupt_cache_objects_fall_through_without_local_admission() {
    let root = temporary("fall-through");
    let destination = Store::init(&root.join("destination")).unwrap();
    let cache = RepoObjectCache::new(root.join("cache"));
    let wanted = ObjectRef::of(b"right");

    assert_eq!(
        cache.admit(&destination, CacheKind::Blob, &wanted).unwrap(),
        CacheRead::Missing
    );
    let cache_path = cache.path(CacheKind::Blob, &wanted.sha256).unwrap();
    write(&cache_path, b"wrong");
    assert_eq!(
        cache.admit(&destination, CacheKind::Blob, &wanted).unwrap(),
        CacheRead::Corrupt
    );
    assert!(!destination.blob_path(&wanted.sha256).exists());
    let source = Store::init(&root.join("source")).unwrap();
    put_blob(&source, b"right");
    assert_eq!(
        cache
            .backfill_store(&source, CacheKind::Blob, &wanted)
            .unwrap(),
        CacheWrite::Unavailable,
        "no-clobber leaves repair of a corrupt final entry explicit"
    );
    assert_eq!(fs::read(&cache_path).unwrap(), b"wrong");
    fs::remove_file(&cache_path).unwrap();
    assert_eq!(
        cache
            .backfill_store(&source, CacheKind::Blob, &wanted)
            .unwrap(),
        CacheWrite::Stored
    );

    let malformed = b"not canonical manifest JSON";
    let manifest_ref = ObjectRef::of(malformed);
    write(
        &cache
            .path(CacheKind::Manifest, &manifest_ref.sha256)
            .unwrap(),
        malformed,
    );
    assert_eq!(
        cache
            .admit(&destination, CacheKind::Manifest, &manifest_ref)
            .unwrap(),
        CacheRead::Corrupt
    );
    assert!(!destination.manifest_path(&manifest_ref.sha256).exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn concurrent_backfills_converge_without_clobbering() {
    let root = temporary("concurrent");
    let source = Arc::new(Store::init(&root.join("source-store")).unwrap());
    let bytes = vec![0x5a; 2 << 20];
    let wanted = put_blob(&source, &bytes);
    let cache = Arc::new(RepoObjectCache::new(root.join("cache")));
    let barrier = Arc::new(Barrier::new(8));
    let mut workers = Vec::new();
    for _ in 0..8 {
        let cache = Arc::clone(&cache);
        let barrier = Arc::clone(&barrier);
        let source = Arc::clone(&source);
        let wanted = wanted.clone();
        workers.push(thread::spawn(move || {
            barrier.wait();
            cache
                .backfill_store(&source, CacheKind::Blob, &wanted)
                .unwrap()
        }));
    }
    let results = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        results
            .iter()
            .filter(|status| **status == CacheWrite::Stored)
            .count(),
        1
    );
    assert!(results
        .iter()
        .all(|status| matches!(status, CacheWrite::Stored | CacheWrite::Present)));
    assert_eq!(
        fs::read(cache.path(CacheKind::Blob, &wanted.sha256).unwrap()).unwrap(),
        bytes
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn incomplete_hidden_temp_is_never_a_hit_and_disabled_cache_is_soft() {
    let root = temporary("partial");
    let cache = RepoObjectCache::new(root.join("cache"));
    let wanted = ObjectRef::of(b"complete");
    let final_path = cache.path(CacheKind::Blob, &wanted.sha256).unwrap();
    write(&final_path.with_file_name(".interrupted.tmp"), b"comp");
    let destination = Store::init(&root.join("destination")).unwrap();
    assert_eq!(
        cache.admit(&destination, CacheKind::Blob, &wanted).unwrap(),
        CacheRead::Missing
    );
    assert!(!destination.blob_path(&wanted.sha256).exists());

    let unavailable_root = root.join("not-a-directory");
    fs::write(&unavailable_root, b"occupied").unwrap();
    let unavailable = RepoObjectCache::new(unavailable_root);
    let source = Store::init(&root.join("source-store")).unwrap();
    put_blob(&source, b"complete");
    assert_eq!(
        unavailable
            .backfill_store(&source, CacheKind::Blob, &wanted)
            .unwrap(),
        CacheWrite::Unavailable
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn cli_emits_one_json_status_and_admits_verified_bytes() {
    let root = temporary("cli");
    fs::create_dir_all(&root).unwrap();
    let cache = root.join("cache");
    let source = root.join("download");
    let store_root = root.join("store");
    let store = Store::init(&store_root).unwrap();
    let bytes = b"from-nfs-cache";
    fs::write(&source, bytes).unwrap();
    let wanted = ObjectRef::of(bytes);
    let tfs = env!("CARGO_BIN_EXE_tfs");

    let store_admit = Command::new(tfs)
        .args([
            "store",
            "admit-file",
            store_root.to_str().unwrap(),
            "blob",
            &wanted.id(),
            &wanted.length.to_string(),
            source.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(store_admit.status.success());
    assert_eq!(store_admit.stdout, b"{\"status\":\"stored\"}\n");
    assert!(store_admit.stderr.is_empty());

    let backfill = Command::new(tfs)
        .args([
            "repo-cache",
            "backfill",
            cache.to_str().unwrap(),
            store_root.to_str().unwrap(),
            "blob",
            &wanted.id(),
            &wanted.length.to_string(),
        ])
        .output()
        .unwrap();
    assert!(backfill.status.success());
    assert_eq!(backfill.stdout, b"{\"status\":\"stored\"}\n");
    assert!(backfill.stderr.is_empty());

    fs::remove_file(store.blob_path(&wanted.sha256)).unwrap();

    let admit = Command::new(tfs)
        .args([
            "repo-cache",
            "admit",
            cache.to_str().unwrap(),
            store_root.to_str().unwrap(),
            "blob",
            &wanted.id(),
            &wanted.length.to_string(),
        ])
        .output()
        .unwrap();
    assert!(admit.status.success());
    assert_eq!(admit.stdout, b"{\"status\":\"hit\"}\n");
    assert!(admit.stderr.is_empty());
    let mut admitted = store.open_verified(&wanted.sha256).unwrap();
    let mut actual = Vec::new();
    std::io::Read::read_to_end(&mut admitted, &mut actual).unwrap();
    assert_eq!(actual, bytes);
    let _ = fs::remove_dir_all(root);
}
