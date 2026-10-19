//! The missing-tool rule for package authoring, through the real binary on a PATH that lacks the
//! tool. uv is essential and cozy can fetch it: a lock with no uv anywhere fetches the pinned one
//! into `~/.cozy/tools` and completes. git is essential and cozy can't: a project with a git
//! dependency and no git is refused naming git, its install command and `cozy doctor`, before uv
//! runs and fails on it.
#![cfg(unix)]
mod common;

use std::path::Path;

use common::{Home, Out};
use serde_json::Value;

/// `cozy --json package lock` in `dir` with `path` as the whole PATH.
fn lock(home: &Home, dir: &Path, path: &Path) -> Out {
    common::out(
        home.command()
            .current_dir(dir)
            .env("PATH", path)
            .args(["--json", "package", "lock"])
            .output()
            .unwrap(),
    )
}

#[test]
fn a_lock_with_no_uv_fetches_the_pinned_one() {
    let Some(home) = Home::new() else { return };
    let tiny = common::package::project(&home, "tiny");
    let nothing = tempfile::tempdir().unwrap();
    let out = lock(&home, &tiny, nothing.path());
    assert_eq!(out.code, 0, "{out:?}");
    let locked: Value = serde_json::from_str(&out.stdout).unwrap();
    let lock = tiny.join("uv.lock");
    assert_eq!(locked["lock"].as_str(), lock.to_str(), "{locked}");
    assert!(lock.is_file());
    let uv = home
        .root()
        .join("tools/uv")
        .join(platform::tools::UV_VERSION)
        .join("uv");
    assert!(
        uv.is_file(),
        "{} is not where the pinned uv went",
        uv.display()
    );
}

#[test]
fn a_git_dependency_with_no_git_is_refused_naming_its_install() {
    let Some(home) = Home::new() else { return };
    let tiny = common::package::project(&home, "tiny");
    let pyproject = tiny.join("pyproject.toml");
    let text = std::fs::read_to_string(&pyproject).unwrap().replace(
        "\"msgspec>=0.19\"]",
        "\"msgspec>=0.19\", \"tinier @ git+https://github.com/cozy-creator/tinier@0123456789abcdef0123456789abcdef01234567\"]",
    );
    std::fs::write(&pyproject, text).unwrap();
    // A PATH holding only this runner's package manager: git is absent, and its install known.
    let bin = tempfile::tempdir().unwrap();
    let apt = Path::new("/usr/bin/apt-get");
    if apt.exists() {
        std::os::unix::fs::symlink(apt, bin.path().join("apt-get")).unwrap();
    }
    let out = lock(&home, &tiny, bin.path());
    assert_eq!(out.code, 1, "{out:?}");
    let refusal: Value = serde_json::from_str(&out.stdout).unwrap();
    let error = &refusal["error"];
    assert_eq!(
        (error["kind"].as_str(), error["code"].as_str()),
        (Some("unavailable"), Some("git_unavailable")),
        "{refusal}"
    );
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .starts_with("git not found: "),
        "{refusal}"
    );
    let next: Vec<&str> = error["next"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let install = match apt.exists() {
        true => "sudo apt-get install -y git",
        false => "install git with this computer's package manager",
    };
    assert_eq!(next, [install, "cozy doctor --json"], "{refusal}");
    // Refused before uv was needed: nothing was fetched for a command that could not complete.
    assert!(!home.root().join("tools/uv").exists());
}
