//! The maintenance command must run before opening a current-format client, and must
//! refuse live pre-cut owners/readers using the same kernel locks as the published SDK.
#![cfg(feature = "store")]

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "tfs-store-migration-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn old(root: &Path) -> (PathBuf, Vec<u8>) {
    for dir in ["repos", "manifests", "blobs", "tmp/writers"] {
        fs::create_dir_all(root.join(dir)).unwrap();
    }
    fs::write(root.join("format"), b"store-format 1\nclaim-protocol 1\n").unwrap();
    let bytes = b"old payload bytes stay at their original content address".to_vec();
    let object = tensorfs::format::ObjectRef::of(&bytes);
    let blob = root
        .join("blobs")
        .join(&object.sha256[..2])
        .join(&object.sha256[2..4])
        .join(&object.sha256);
    fs::create_dir_all(blob.parent().unwrap()).unwrap();
    fs::write(&blob, &bytes).unwrap();
    (blob, bytes)
}

fn migrate(root: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tfs"))
        .arg("--store")
        .arg(root)
        .args(["--json", "migrate-store"])
        .output()
        .unwrap()
}

fn lock(path: &Path) -> fs::File {
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap()
}

#[test]
fn maintenance_upgrades_once_and_preserves_payloads_without_a_client_open() {
    let scratch = Scratch::new();
    let root = scratch.0.join("store");
    let (blob, bytes) = old(&root);
    // The catalog is an opaque legacy projection, not an excuse to delete old state.
    fs::write(root.join("tensorfs.sqlite"), b"preserved legacy catalog").unwrap();
    let ran = migrate(&root);
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&ran.stdout).unwrap();
    assert_eq!(report["migrated"], true);
    assert_eq!(fs::read(&blob).unwrap(), bytes);
    assert_eq!(
        fs::read(root.join("tensorfs.sqlite")).unwrap(),
        b"preserved legacy catalog"
    );
    assert!(
        root.join("manifests").is_file(),
        "an old client must not recreate the retired namespace"
    );
    tensorfs::store::Store::open(&root).unwrap();
    let repeated = migrate(&root);
    assert!(
        repeated.status.success(),
        "{}",
        String::from_utf8_lossy(&repeated.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&repeated.stdout).unwrap();
    assert_eq!(report["migrated"], false);
    assert_eq!(fs::read(&blob).unwrap(), bytes);
}

#[test]
fn maintenance_refuses_live_owner_and_external_reader_before_mutation() {
    let scratch = Scratch::new();
    let root = scratch.0.join("store");
    let (blob, bytes) = old(&root);
    let original = fs::read(root.join("format")).unwrap();
    let owner = lock(&root.join("owner.lock"));
    owner.try_lock().unwrap();
    let first = migrate(&root);
    assert!(!first.status.success());
    assert_eq!(fs::read(root.join("format")).unwrap(), original);
    assert!(root.join("manifests").is_dir());
    drop(owner);
    // Old external reader Holds and WriterGuards share this lock for their whole lifetime.
    let reader = lock(&root.join("tmp/writers/recovery.lock"));
    reader.try_lock_shared().unwrap();
    let second = migrate(&root);
    assert!(!second.status.success());
    assert_eq!(fs::read(root.join("format")).unwrap(), original);
    assert!(root.join("manifests").is_dir());
    assert_eq!(fs::read(&blob).unwrap(), bytes);
    drop(reader);
    let ready = migrate(&root);
    assert!(
        ready.status.success(),
        "{}",
        String::from_utf8_lossy(&ready.stderr)
    );
}

#[test]
fn maintenance_noop_does_not_initialize_an_absent_store() {
    let scratch = Scratch::new();
    let root = scratch.0.join("absent");
    let ran = migrate(&root);
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&ran.stdout).unwrap();
    assert_eq!(report["migrated"], false);
    assert!(
        !root.exists(),
        "maintenance must run before Client::new creates a store"
    );
}
