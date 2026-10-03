use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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

/// A store verb without a root is a usage error that writes nothing (tfs-053).
#[test]
fn store_verbs_name_their_root_and_write_nothing_without_one() {
    for args in [
        &["store", "init"][..],
        &["store", "ensure"],
        &["store", "info"],
        &["store", "prepare-readers"],
    ] {
        let root = temporary("no-root");
        let home = root.join("home");
        let cwd = root.join("cwd");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&cwd).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
            .args(args)
            .current_dir(&cwd)
            .env("HOME", &home)
            .env("TENSORFS_HOME", root.join("environment"))
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2), "tfs {args:?}");
        assert_eq!(fs::read_dir(&cwd).unwrap().count(), 0, "tfs {args:?} wrote into its working directory");
        assert!(!home.join(".tensorfs").exists() && !root.join("environment").exists());
        fs::remove_dir_all(root).unwrap();
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
    let run = |args: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_tfs")).args(args).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    };
    run(&["store", "init", named.to_str().unwrap()]);
    assert!(named.join("blobs").is_dir());
    run(&["store", "ensure", named.to_str().unwrap()]);
    run(&["store", "ensure", named.to_str().unwrap()]);
    run(&["store", "info", named.to_str().unwrap()]);
    fs::remove_dir_all(root).unwrap();
}
