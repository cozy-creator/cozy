use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs_core::err::Code;
use tensorfs_core::header::{Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::manifest::{Draft, Entry, Manifest};
use tensorfs_core::store::{Fault, Store};
use tensorfs_core::{dtype::Dtype, registry};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-boundary-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn mode(path: &Path) -> u32 {
    fs::symlink_metadata(path).unwrap().permissions().mode() & 0o7777
}

fn model_manifest(store: &Store, body: Body, shape: Vec<u64>, note: &[u8]) -> ObjectRef {
    let spec = registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let header = Header {
        configs: Vec::new(),
        assets: Vec::new(),
        encodings: vec![spec.clone()],
        components: vec![(
            "model".into(),
            vec![(
                "weight".into(),
                Tensor {
                    dtype: Dtype::F32,
                    shape: shape.clone(),
                    encoding: spec.object_id(),
                    parts: vec![(
                        "value".into(),
                        Part {
                            dtype: Dtype::F32,
                            shape,
                            body,
                        },
                    )],
                },
            )],
        )],
    };
    header.validate(&Closure::default()).unwrap();
    let header_bytes = header.canonical_bytes().unwrap();
    let header_ref = store
        .put_stream(
            &mut header_bytes.as_slice(),
            Some(&ObjectRef::of(&header_bytes)),
            &Fault::default(),
        )
        .unwrap()
        .obj;
    let manifest = Draft {
        entries: vec![
            ("README.txt".into(), Entry::File(ObjectRef::of(note))),
            ("model.cozytensors".into(), Entry::CozyTensors(header_ref)),
        ],
    }
    .seal()
    .unwrap();
    store.put_manifest(&manifest).unwrap().obj
}

/// Many writers ensuring one brand-new root at once: one creates it, the others open it,
/// and none is refused as "not empty".
#[test]
fn concurrent_ensures_of_a_new_root_all_succeed() {
    let root = temporary("concurrent-ensure");
    fs::create_dir_all(&root).unwrap();
    let empty = root.join("empty");
    fs::create_dir(&empty).unwrap();
    for target in [root.join("absent"), empty] {
        std::thread::scope(|scope| {
            let ensures: Vec<_> = (0..16).map(|_| scope.spawn(|| Store::ensure(&target))).collect();
            for ensure in ensures {
                ensure.join().unwrap().unwrap();
            }
        });
        Store::open(&target).unwrap();
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn ensure_initializes_only_absent_or_empty_real_roots() {
    let root = temporary("ensure");
    fs::create_dir_all(&root).unwrap();

    let absent = root.join("absent");
    assert_eq!(Store::ensure(&absent).unwrap().root(), absent);
    Store::ensure(&absent).unwrap();
    assert_eq!(Store::init(&absent).unwrap_err().code, Code::STORE_ERA);
    Store::ensure(&absent).unwrap();

    let empty = root.join("empty");
    fs::create_dir(&empty).unwrap();
    Store::ensure(&empty).unwrap();

    let foreign = root.join("foreign");
    fs::create_dir(&foreign).unwrap();
    fs::write(foreign.join("somebody-elses-state"), b"keep").unwrap();
    assert_eq!(Store::ensure(&foreign).unwrap_err().code, Code::STORE_ERA);
    assert_eq!(Store::init(&foreign).unwrap_err().code, Code::STORE_ERA);
    assert!(!foreign.join("repos").exists());

    let partial = root.join("partial");
    fs::create_dir(&partial).unwrap();
    fs::create_dir(partial.join("repos")).unwrap();
    fs::create_dir(partial.join("manifests")).unwrap();
    assert_eq!(Store::ensure(&partial).unwrap_err().code, Code::STORE_ERA);
    assert!(!partial.join("blobs").exists());

    let missing_catalog = root.join("missing-catalog");
    for namespace in ["repos", "manifests", "blobs"] {
        fs::create_dir_all(missing_catalog.join(namespace)).unwrap();
    }
    assert_eq!(
        Store::ensure(&missing_catalog).unwrap_err().code,
        Code::DURABILITY_UNPROVEN
    );
    assert!(!missing_catalog.join("tmp").exists());

    let corrupt_catalog = root.join("corrupt-catalog");
    Store::init(&corrupt_catalog).unwrap();
    fs::remove_dir_all(corrupt_catalog.join("tmp")).unwrap();
    let connection = rusqlite::Connection::open(corrupt_catalog.join("tensorfs.sqlite")).unwrap();
    connection
        .execute_batch("ALTER TABLE tensorfs_verified_blobs DROP COLUMN mtime_ns")
        .unwrap();
    drop(connection);
    assert_eq!(
        Store::ensure(&corrupt_catalog).unwrap_err().code,
        Code::UNKNOWN_FORMAT
    );
    assert!(!corrupt_catalog.join("tmp").exists());

    let real = root.join("real");
    Store::init(&real).unwrap();
    let link = root.join("link");
    symlink(&real, &link).unwrap();
    assert_eq!(Store::ensure(&link).unwrap_err().code, Code::STORE_ERA);
    assert_eq!(Store::init(&link).unwrap_err().code, Code::STORE_ERA);
    assert_eq!(Store::open(&link).unwrap_err().code, Code::STORE_ERA);

    let missing_target = root.join("must-not-be-created");
    let dangling = root.join("dangling");
    symlink(&missing_target, &dangling).unwrap();
    assert_eq!(Store::init(&dangling).unwrap_err().code, Code::STORE_ERA);
    assert!(!missing_target.exists());

    for namespace in ["repos", "manifests", "blobs"] {
        let store_root = root.join(format!("symlink-{namespace}"));
        Store::init(&store_root).unwrap();
        let outside = root.join(format!("outside-{namespace}"));
        fs::create_dir(&outside).unwrap();
        fs::remove_dir(store_root.join(namespace)).unwrap();
        symlink(&outside, store_root.join(namespace)).unwrap();
        assert_eq!(
            Store::ensure(&store_root).unwrap_err().code,
            Code::STORE_ERA
        );
        assert_eq!(Store::open(&store_root).unwrap_err().code, Code::STORE_ERA);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn prepare_readers_owns_exact_modes_and_refuses_internal_symlinks() {
    let root = temporary("readers");
    let store_root = root.join("store");
    fs::create_dir_all(&root).unwrap();
    Store::ensure(&store_root).unwrap();
    Store::open(&store_root).unwrap().prepare_readers().unwrap();
    for path in [
        "",
        "repos",
        "manifests",
        "blobs",
        "checkouts",
        "tmp",
        "tmp/writers",
    ] {
        assert_eq!(mode(&store_root.join(path)), 0o755, "{path}");
    }
    assert_eq!(mode(&store_root.join("tmp/leases")), 0o1733);
    assert_eq!(mode(&store_root.join("tmp/writers/recovery.lock")), 0o666);

    let outside = root.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::set_permissions(&outside, fs::Permissions::from_mode(0o700)).unwrap();
    fs::remove_dir(store_root.join("tmp/leases")).unwrap();
    symlink(&outside, store_root.join("tmp/leases")).unwrap();
    let store = Store::open(&store_root).unwrap();
    assert!(store.prepare_readers().is_err());
    assert_eq!(mode(&outside), 0o700);

    fs::remove_file(store_root.join("tmp/leases")).unwrap();
    fs::create_dir(store_root.join("tmp/leases")).unwrap();
    let outside_file = root.join("outside-lock");
    fs::write(&outside_file, b"not the lock").unwrap();
    fs::set_permissions(&outside_file, fs::Permissions::from_mode(0o600)).unwrap();
    fs::remove_file(store_root.join("tmp/writers/recovery.lock")).unwrap();
    symlink(&outside_file, store_root.join("tmp/writers/recovery.lock")).unwrap();
    assert!(store.prepare_readers().is_err());
    assert_eq!(mode(&outside_file), 0o600);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn complete_cozytensors_census_is_sorted_verified_and_runtime_only() {
    let root = temporary("census");
    let store = Store::ensure(&root).unwrap();
    let complete_a = model_manifest(&store, Body::Inline(vec![0; 4]), vec![1], b"missing-a");
    let complete_b = model_manifest(&store, Body::Inline(vec![1; 4]), vec![1], b"missing-b");
    let missing_part = ObjectRef::of(&vec![2; 260]);
    let _incomplete = model_manifest(
        &store,
        Body::Segments(vec![missing_part]),
        vec![65],
        b"missing-c",
    );
    let ordinary = Manifest::from_files(vec![(
        "dataset.jsonl".into(),
        ObjectRef::of(b"not resident"),
    )])
    .unwrap();
    store.put_manifest(&ordinary).unwrap();
    let corrupt = "11".repeat(32);
    let corrupt_path = store.manifest_path(&corrupt);
    fs::create_dir_all(corrupt_path.parent().unwrap()).unwrap();
    fs::write(corrupt_path, b"not canonical manifest bytes").unwrap();

    let mut expected = vec![complete_a.id(), complete_b.id()];
    expected.sort();
    assert_eq!(store.complete_cozytensors_manifests().unwrap(), expected);

    fs::remove_dir_all(root).unwrap();
}

/// THE BINDING IS A PROPERTY OF THE STORE, and this is the whole of what that claims.
///
/// It survives the process that wrote it (a pod boots, sets up once, and converts for
/// hours under a different handle in a different program); it is LOCAL, never on the cache,
/// because #612 gives the NFS root immutable Manifest and blob paths and nothing else; and
/// no shape of it can turn a Store that would open into a Store that will not, because a
/// cache is only ever allowed to change cost.
#[test]
fn the_repo_cache_binding_survives_reopen_and_can_never_refuse_a_store() {
    let root = temporary("repo-cache-binding");
    let store_root = root.join("store");
    let cache = root.join("repo-cache");
    let store = Store::init(&store_root).unwrap();
    assert!(
        store.repo_cache().is_none(),
        "a new Store is bound to nothing"
    );

    let store = store.bind_repo_cache(Some(&cache)).unwrap();
    assert_eq!(store.repo_cache().map(|c| c.root()), Some(&*cache));
    // A DIFFERENT HANDLE, made the way every later command makes one.
    assert_eq!(
        Store::open(&store_root)
            .unwrap()
            .repo_cache()
            .map(|c| c.root()),
        Some(&*cache)
    );
    assert_eq!(
        Store::ensure(&store_root)
            .unwrap()
            .repo_cache()
            .map(|c| c.root()),
        Some(&*cache)
    );
    // Recording the mount created nothing on it. `bind` performs no I/O on the cache path:
    // a stat on a hard NFS mount that has stopped answering never returns, and setup is not
    // allowed to hang on the one thing that may only cost latency.
    assert!(!cache.exists());

    // Rebinding replaces; the record is DECLARATIVE and never accumulates.
    let elsewhere = root.join("other-cache");
    let store = store.bind_repo_cache(Some(&elsewhere)).unwrap();
    assert_eq!(store.repo_cache().map(|c| c.root()), Some(&*elsewhere));
    let store = store.bind_repo_cache(None).unwrap();
    assert!(store.repo_cache().is_none());
    assert!(Store::open(&store_root).unwrap().repo_cache().is_none());
    // Clearing a binding that is already absent is not an error.
    store.bind_repo_cache(None).unwrap();

    // A cache INSIDE the Store, or a Store inside the cache, is refused at bind time —
    // where it is a caller's mistake and not weather.
    for overlap in [store_root.clone(), store_root.join("blobs"), root.clone()] {
        assert_eq!(
            Store::open(&store_root)
                .unwrap()
                .bind_repo_cache(Some(&overlap))
                .unwrap_err()
                .code,
            Code::STORE_ERA,
            "{} was accepted as a cache root",
            overlap.display()
        );
    }
    assert_eq!(
        Store::open(&store_root)
            .unwrap()
            .bind_repo_cache(Some(Path::new("relative/cache")))
            .unwrap_err()
            .code,
        Code::KEY_GRAMMAR
    );

    // AND THE PART THAT MATTERS MOST: garbage where the binding lives is NO BINDING, never
    // a refusal. A Store that would not open because a cache record was unreadable would be
    // a cache failure that cost availability, which is exactly what #612 forbids.
    let binding = store_root.join("repo-cache");
    for junk in [
        b"not-absolute\n".to_vec(),
        b"\n".to_vec(),
        Vec::new(),
        vec![0u8; 8],
        vec![b'/'; 9000],
    ] {
        fs::write(&binding, &junk).unwrap();
        let reopened = Store::open(&store_root).unwrap();
        assert!(
            reopened.repo_cache().is_none(),
            "{junk:?} was read as a mount point"
        );
        assert!(!reopened.objects().unwrap().is_empty() || reopened.objects().is_ok());
    }
    // A symlink where the binding lives is read through O_NOFOLLOW, so it is not a binding
    // either — and still not a refusal.
    fs::remove_file(&binding).unwrap();
    symlink(&cache, &binding).unwrap();
    assert!(Store::open(&store_root).unwrap().repo_cache().is_none());
    fs::remove_file(&binding).unwrap();

    // The binding is a plain local file beside the namespaces, readable by the pod's other
    // uid, and nothing about it reaches the cache.
    Store::open(&store_root)
        .unwrap()
        .bind_repo_cache(Some(&cache))
        .unwrap();
    assert_eq!(mode(&binding), 0o644);
    assert!(!cache.exists());
    let _ = fs::remove_dir_all(root);
}

// Run this arm as root as well as in the ordinary suite. Separate CLI processes
// exercise the same public Store boundary used by broker writes and Host downloads.
