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
        !home.join(".cache").join("tensorfs").exists(),
        "tfs {args:?} conjured a home Store"
    );
    let _ = fs::remove_dir_all(root);
}

/// `store rebuild` without a root is a usage error that writes nothing (tfs-053).
#[test]
fn store_rebuild_names_its_root_and_writes_nothing_without_one() {
    let root = temporary("no-root");
    let home = root.join("home");
    let cwd = root.join("cwd");
    fs::create_dir_all(&home).unwrap();
    fs::create_dir_all(&cwd).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["store", "rebuild"])
        .current_dir(&cwd)
        .env("HOME", &home)
        .env("TENSORFS_HOME", root.join("environment"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(
        fs::read_dir(&cwd).unwrap().count(),
        0,
        "tfs store rebuild wrote into its working directory"
    );
    assert!(!home.join(".cache").join("tensorfs").exists() && !root.join("environment").exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn rebuild_refuses_an_absent_census_root_instead_of_creating_one() {
    refuse_without_a_named_root(
        "absent-census",
        &["store", "rebuild", "census", "--rows", "rows.jsonl"],
    );
}
