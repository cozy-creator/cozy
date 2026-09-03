use std::fs;
use std::path::{Path, PathBuf};
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
