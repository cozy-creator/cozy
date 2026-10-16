use std::fs;
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs_core::err::Code;
use tensorfs_core::store::Store;

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

// Run this arm as root as well as in the ordinary suite. Separate CLI processes
// exercise the same public Store boundary used by broker writes and Host downloads.
