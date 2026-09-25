use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs_core::ids::{Doc, ObjectRef};
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

    // Manifest repairs obey the same complete-object publication rule as blobs.
    let manifest_path = cache
        .path(CacheKind::Manifest, &manifest_ref.sha256)
        .unwrap();
    fs::set_permissions(
        &manifest_path,
        std::os::unix::fs::PermissionsExt::from_mode(0o600),
    )
    .unwrap();
    fs::write(&manifest_path, b"damaged manifest").unwrap();
    assert_eq!(
        cache
            .backfill_store(&source, CacheKind::Manifest, &manifest_ref)
            .unwrap(),
        CacheWrite::Stored
    );
    assert_eq!(
        fs::read(&manifest_path).unwrap(),
        manifest.canonical_bytes()
    );

    let cache_root = cache.root();
    assert!(!cache_root.join("tensorfs.sqlite").exists());
    assert!(!cache_root.join("repos").exists());
    assert!(!cache_root.join("tmp").exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn a_complete_generic_snapshot_rehydrates_a_fresh_store_from_cache() {
    let root = temporary("generic-snapshot");
    let source_root = root.join("source");
    let source = Store::init(&source_root).unwrap();
    let cache = RepoObjectCache::new(root.join("cache"));
    let files: [(&str, &[u8]); 3] = [
        ("dataset/shard-000.jsonl", b"{\"sample\":0}\n"),
        ("dataset/shard-001.jsonl", b"{\"sample\":1}\n"),
        ("metadata/schema.json", b"{\"columns\":[\"sample\"]}"),
    ];
    let objects = files
        .iter()
        .map(|(_, bytes)| put_blob(&source, bytes))
        .collect::<Vec<_>>();
    let manifest = Manifest::from_files(
        files
            .iter()
            .zip(&objects)
            .map(|((path, _), object)| ((*path).to_string(), object.clone()))
            .collect(),
    )
    .unwrap();
    let manifest_ref = source.put_manifest(&manifest).unwrap().obj;

    for object in &objects {
        assert_eq!(
            cache
                .backfill_store(&source, CacheKind::Blob, object)
                .unwrap(),
            CacheWrite::Stored
        );
    }
    assert_eq!(
        cache
            .backfill_store(&source, CacheKind::Manifest, &manifest_ref)
            .unwrap(),
        CacheWrite::Stored
    );
    drop(source);
    fs::remove_dir_all(source_root).unwrap();

    let destination = Store::init(&root.join("fresh-destination")).unwrap();
    assert_eq!(
        cache
            .admit(&destination, CacheKind::Manifest, &manifest_ref)
            .unwrap(),
        CacheRead::Hit
    );
    for object in &objects {
        assert_eq!(
            cache.admit(&destination, CacheKind::Blob, object).unwrap(),
            CacheRead::Hit
        );
    }
    assert_eq!(destination.read_manifest(&manifest_ref).unwrap(), manifest);
    for ((_, expected), object) in files.iter().zip(&objects) {
        let mut admitted = destination.open_verified(&object.sha256).unwrap();
        let mut actual = Vec::new();
        std::io::Read::read_to_end(&mut admitted, &mut actual).unwrap();
        assert_eq!(&actual, expected);
    }
    assert!(!cache.root().join("repos").exists());
    assert!(!cache.root().join("tensorfs.sqlite").exists());
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
        CacheWrite::Present,
        "an exact-length entry at the name is present without being read"
    );
    assert_eq!(
        cache
            .repair_store(&source, CacheKind::Blob, &wanted)
            .unwrap(),
        CacheWrite::Stored,
        "verified local bytes atomically repair a corrupt regular cache entry"
    );
    assert_eq!(fs::read(&cache_path).unwrap(), b"right");
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
fn fifo_symlink_and_directory_entries_are_unavailable() {
    use std::os::unix::fs::symlink;

    let root = temporary("nonregular");
    let cache = RepoObjectCache::new(root.join("cache"));
    let destination = Store::init(&root.join("destination")).unwrap();
    let object = ObjectRef::of(b"verified");
    let path = cache.path(CacheKind::Blob, &object.sha256).unwrap();
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let correct = root.join("correct");
    fs::write(&correct, b"verified").unwrap();
    symlink(&correct, &path).unwrap();
    assert_eq!(
        cache.admit(&destination, CacheKind::Blob, &object).unwrap(),
        CacheRead::Unavailable
    );
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    assert_eq!(
        cache.admit(&destination, CacheKind::Blob, &object).unwrap(),
        CacheRead::Unavailable
    );
    fs::remove_dir(&path).unwrap();
    assert!(Command::new("mkfifo")
        .arg(&path)
        .status()
        .unwrap()
        .success());
    let mut child = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args([
            "repo-cache",
            "admit",
            cache.root().to_str().unwrap(),
            destination.root().to_str().unwrap(),
            "blob",
            &object.id(),
            &object.length.to_string(),
        ])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while child.try_wait().unwrap().is_none() {
        if std::time::Instant::now() >= until {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("cache FIFO waited for a writer before rejecting the nonregular entry");
        }
        thread::sleep(std::time::Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        r#"{"status":"unavailable"}"#
    );
    assert!(!destination.blob_path(&object.sha256).exists());
    fs::remove_file(&path).unwrap();
    fs::copy(&correct, &path).unwrap();
    assert_eq!(
        cache.admit(&destination, CacheKind::Blob, &object).unwrap(),
        CacheRead::Hit
    );
    fs::remove_dir_all(root).unwrap();
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

#[test]
fn the_write_through_publishes_every_offer_and_the_watermark_is_the_offer_prefix() {
    let root = temporary("mirror");
    let store = Store::init(&root.join("store")).unwrap();
    let cache = RepoObjectCache::new(root.join("cache"));
    let objects: Vec<ObjectRef> = (0..64u32)
        .map(|n| {
            put_blob(
                &store,
                &format!("write-through object {n}").repeat(8).into_bytes(),
            )
        })
        .collect();

    let mirror = tensorfs_core::repo_cache::Mirror::start(cache.clone(), store.root(), u64::MAX);
    for object in &objects {
        mirror.offer(CacheKind::Blob, object);
    }
    let (offered, bytes_offered) = mirror.offered();
    assert_eq!(offered, objects.len() as u64);
    mirror.drain();
    let report = mirror.report();

    assert_eq!(report.stored, objects.len() as u64);
    assert_eq!(report.unavailable, 0);
    assert_eq!(report.bytes_stored, bytes_offered);
    // The prefix reaches every offer, and it is a PREFIX: with four writers finishing out of
    // order, a total would have been right for the wrong reason.
    assert_eq!(report.durable.objects, objects.len() as u64);
    assert_eq!(report.durable.bytes, bytes_offered);
    assert!(!report.durable.stalled);

    // Every published object comes back through the ordinary verified door.
    let fresh = Store::init(&root.join("fresh")).unwrap();
    for object in &objects {
        assert_eq!(
            cache.admit(&fresh, CacheKind::Blob, object).unwrap(),
            CacheRead::Hit
        );
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn a_mount_that_refuses_stalls_the_watermark_and_publishes_nothing() {
    let root = temporary("mirror-refuses");
    let store = Store::init(&root.join("store")).unwrap();
    // A regular file where the cache root should be: every directory under it refuses.
    let broken = root.join("not-a-mount");
    fs::write(&broken, b"not a directory").unwrap();
    let object = put_blob(&store, b"an object nobody can publish");

    let mirror = tensorfs_core::repo_cache::Mirror::start(
        RepoObjectCache::new(&broken),
        store.root(),
        u64::MAX,
    );
    mirror.offer(CacheKind::Blob, &object);
    mirror.drain();
    let report = mirror.report();

    assert_eq!(report.offered, 1);
    assert_eq!(report.stored, 0);
    assert_eq!(report.unavailable, 1);
    // The watermark does NOT advance over an object the mount would not take. A journal
    // written past this point would name bytes that are not there.
    assert_eq!(report.durable.objects, 0);
    assert_eq!(report.durable.bytes, 0);
    assert!(report.durable.stalled);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn concurrent_offers_keep_the_offer_list_and_the_sequence_in_step() {
    let root = temporary("mirror-concurrent");
    let store = Store::init(&root.join("store")).unwrap();
    let cache = RepoObjectCache::new(root.join("cache"));
    let objects: Vec<ObjectRef> = (0..128u32)
        .map(|n| {
            put_blob(
                &store,
                &format!("concurrent offer {n}").repeat(4).into_bytes(),
            )
        })
        .collect();

    let mirror = Arc::new(tensorfs_core::repo_cache::Mirror::start(
        cache,
        store.root(),
        u64::MAX,
    ));
    // Several producers, because `Mirror` is a public type and nothing in it says one. The
    // sequence a producer is handed IS the offer list's index, and a journal link built over
    // a range of sequences must name exactly the objects at those positions.
    let start = Arc::new(Barrier::new(4));
    let mut threads = Vec::new();
    for lane in 0..4usize {
        let mirror = Arc::clone(&mirror);
        let start = Arc::clone(&start);
        let objects = objects.clone();
        threads.push(thread::spawn(move || {
            start.wait();
            let mut seen = Vec::new();
            for object in objects.iter().skip(lane).step_by(4) {
                seen.push((mirror.offer(CacheKind::Blob, object), object.clone()));
            }
            seen
        }));
    }
    let mut claimed: Vec<(u64, ObjectRef)> = threads
        .into_iter()
        .flat_map(|thread| thread.join().unwrap())
        .collect();
    claimed.sort_by_key(|(seq, _)| *seq);
    mirror.drain();

    assert_eq!(claimed.len(), objects.len());
    for (position, (seq, object)) in claimed.iter().enumerate() {
        assert_eq!(*seq, position as u64, "sequences are not dense");
        let at = mirror.offers(*seq, seq + 1);
        assert_eq!(
            at,
            vec![(CacheKind::Blob, object.clone())],
            "offer {seq} is not the object its producer was numbered for"
        );
    }
    assert_eq!(mirror.report().durable.objects, objects.len() as u64);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn a_symlinked_fanout_never_redirects_cache_publication() {
    use std::os::unix::fs::symlink;
    let root = temporary("fanout-symlink");
    let source = Store::init(&root.join("source")).unwrap();
    let object = put_blob(&source, b"verified");
    let cache = RepoObjectCache::new(root.join("cache"));
    let outside = root.join("outside");
    fs::create_dir_all(&outside).unwrap();
    fs::create_dir_all(cache.root()).unwrap();
    symlink(&outside, cache.root().join("blobs")).unwrap();
    assert_eq!(
        cache
            .backfill_store(&source, CacheKind::Blob, &object)
            .unwrap(),
        CacheWrite::Unavailable
    );
    assert_eq!(fs::read_dir(outside).unwrap().count(), 0);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn concurrent_repairs_publish_only_verified_complete_bytes() {
    let root = temporary("repairs");
    let source = Arc::new(Store::init(&root.join("source")).unwrap());
    let data = vec![73; 2 << 20];
    let object = put_blob(&source, &data);
    let cache = Arc::new(RepoObjectCache::new(root.join("cache")));
    let path = cache.path(CacheKind::Blob, &object.sha256).unwrap();
    write(&path, &vec![0; data.len()]);
    let barrier = Arc::new(Barrier::new(4));
    let workers = (0..4)
        .map(|_| {
            let (source, cache, object, barrier) = (
                source.clone(),
                cache.clone(),
                object.clone(),
                barrier.clone(),
            );
            thread::spawn(move || {
                barrier.wait();
                cache
                    .repair_store(&source, CacheKind::Blob, &object)
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        assert!(matches!(
            worker.join().unwrap(),
            CacheWrite::Stored | CacheWrite::Present
        ));
    }
    assert_eq!(fs::read(path).unwrap(), data);
    assert_eq!(
        cache
            .repair_store(&source, CacheKind::Blob, &object)
            .unwrap(),
        CacheWrite::Present
    );
    fs::remove_dir_all(root).unwrap();
}
