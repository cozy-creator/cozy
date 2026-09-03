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

/// tfs-053. An empty skeleton Store once materialized inside a source checkout because a
/// process ran a store verb with no root from its own working directory. A verb that needs a
/// Store now refuses when nothing was named and nothing is there, and writes nothing at all.
fn refuse_without_a_named_root(name: &str, args: &[&str]) {
    let root = temporary(name);
    let home = root.join("home");
    let cwd = root.join("cwd");
    fs::create_dir_all(&home).unwrap();
    fs::create_dir_all(&cwd).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(args)
        .current_dir(&cwd)
        .env("HOME", &home)
        .env_remove("TENSORFS_HOME")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success(), "tfs {args:?} did not refuse");
    assert!(stderr.contains("STORE_ROOT_ABSENT"), "{stderr}");
    assert_eq!(
        fs::read_dir(&cwd).unwrap().count(),
        0,
        "tfs {args:?} wrote into its working directory"
    );
    assert!(
        !home.join(".tensorfs").exists(),
        "tfs {args:?} conjured a home Store"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn store_verbs_refuse_and_write_nothing_when_no_root_is_named() {
    for (name, args) in [
        ("no-root-init", &["store", "init"][..]),
        ("no-root-ensure", &["store", "ensure"][..]),
        ("no-root-info", &["store", "info"][..]),
        ("no-root-readers", &["store", "prepare-readers"][..]),
        ("no-root-complete", &["store", "complete-cozytensors"][..]),
    ] {
        refuse_without_a_named_root(name, args);
    }
}

#[test]
fn rebuild_refuses_an_absent_census_root_instead_of_creating_one() {
    refuse_without_a_named_root(
        "absent-census",
        &["store", "rebuild", "census", "--rows", "rows.jsonl"],
    );
}

#[test]
fn a_named_root_still_creates_and_store_ensure_stays_idempotent() {
    let root = temporary("named-root");
    let named = root.join("named");
    let environment = root.join("environment");
    let home = root.join("home");
    fs::create_dir_all(&home).unwrap();

    let run = |args: &[&str], tensorfs_home: Option<&Path>| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_tfs"));
        command.args(args).env("HOME", &home);
        match tensorfs_home {
            Some(value) => command.env("TENSORFS_HOME", value),
            None => command.env_remove("TENSORFS_HOME"),
        };
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };

    run(&["store", "init", named.to_str().unwrap()], None);
    assert!(named.join("blobs").is_dir());
    run(&["store", "ensure"], Some(&environment));
    run(&["store", "ensure"], Some(&environment));
    assert!(environment.join("blobs").is_dir());

    // The `$HOME/.tensorfs` fallback is opened once it exists; only creating it needs a name.
    run(
        &["store", "init", home.join(".tensorfs").to_str().unwrap()],
        None,
    );
    run(&["store", "info"], None);
    fs::remove_dir_all(root).unwrap();
}
