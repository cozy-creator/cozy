use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use tensorfs_core::catalog::WriterGuard;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::{Fault, Store, Verdict};

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-home-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn run_home(tensorfs_home: Option<&Path>, user_home: &Path, args: &[&str]) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tfs"));
    command
        .args(args)
        .env("HOME", user_home)
        .env("COZY_HOME", user_home.join("must-not-be-read"));
    match tensorfs_home {
        Some(root) => {
            command.env("TENSORFS_HOME", root);
        }
        None => {
            command.env_remove("TENSORFS_HOME");
        }
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

fn migrate(root: &Path, source: &Path, destination: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["home", "migrate", source.to_str().unwrap()])
        .env("TENSORFS_HOME", destination)
        .env("HOME", root.join("poison-home"))
        .env("COZY_HOME", root.join("poison-cozy"))
        .output()
        .unwrap()
}

#[test]
fn cli_root_precedence_never_reads_cozy_home() {
    let root = temporary("precedence");
    let environment = root.join("environment");
    let user = root.join("user");

    assert_eq!(
        run_home(Some(&environment), &user, &["home", "/explicit"]),
        "/explicit"
    );
    assert_eq!(
        run_home(Some(&environment), &user, &["home"]),
        environment.display().to_string()
    );
    assert_eq!(
        run_home(None, &user, &["home"]),
        user.join(".tensorfs").display().to_string()
    );
}

#[test]
fn store_init_without_a_root_uses_only_temp_tensorfs_home() {
    let root = temporary("default-init");
    let poison = temporary("poison");
    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "init"])
        .env("TENSORFS_HOME", &root)
        .env("HOME", &poison)
        .env("COZY_HOME", poison.join("cozy"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    for directory in ["repos", "manifests", "blobs", "tmp"] {
        assert!(root.join(directory).is_dir());
    }
    assert!(!poison.exists());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn migration_is_one_atomic_move_and_keeps_verification_identity() {
    let root = temporary("migration");
    let source = root.join(".cozy/cas");
    let destination = root.join(".tensorfs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    let store = Store::init(&source).unwrap();
    let bytes = b"already-verified-model-object";
    let object = ObjectRef::of(bytes);
    store
        .put_stream(&mut bytes.as_slice(), Some(&object), &Fault::default())
        .unwrap();
    let before = fs::metadata(store.blob_path(&object.sha256)).unwrap();

    let output = migrate(&root, &source, &destination);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!source.exists());
    assert!(destination.is_dir());
    assert!(String::from_utf8(output.stdout)
        .unwrap()
        .contains("device/inode/mtime preserved by same-filesystem rename"));

    let moved = Store::open(&destination).unwrap();
    let after = fs::metadata(moved.blob_path(&object.sha256)).unwrap();
    assert_eq!(before.dev(), after.dev());
    assert_eq!(before.ino(), after.ino());
    assert_eq!(before.mtime(), after.mtime());
    assert_eq!(before.mtime_nsec(), after.mtime_nsec());
    assert_eq!(
        moved.verify(&object.sha256).unwrap(),
        Verdict::Verified { rehashed: false }
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn migration_refuses_any_destination_conflict_without_moving_source() {
    let root = temporary("migration-conflict");
    let source = root.join(".cozy/cas");
    let destination = root.join(".tensorfs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    Store::init(&source).unwrap();
    Store::init(&destination).unwrap();

    let output = migrate(&root, &source, &destination);
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("already exists"));
    assert!(source.is_dir());
    assert!(destination.is_dir());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn migration_refuses_a_live_holder_without_moving_the_store() {
    let root = temporary("migration-live");
    let source = root.join(".cozy/cas");
    let destination = root.join(".tensorfs");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    Store::init(&source).unwrap();
    let _active = WriterGuard::acquire(&source).unwrap();

    let output = migrate(&root, &source, &destination);
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("STORE_BUSY"));
    assert!(source.is_dir());
    assert!(!destination.exists());
    drop(_active);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn migration_refuses_a_source_symlink_without_following_it() {
    let root = temporary("migration-symlink");
    let real = root.join("real-store");
    let source = root.join("legacy-link");
    let destination = root.join(".tensorfs");
    fs::create_dir_all(&root).unwrap();
    Store::init(&real).unwrap();
    std::os::unix::fs::symlink(&real, &source).unwrap();

    let output = migrate(&root, &source, &destination);
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("not a TensorFS Store directory"));
    assert!(source.is_symlink());
    assert!(real.is_dir());
    assert!(!destination.exists());
    fs::remove_dir_all(root).unwrap();
}
