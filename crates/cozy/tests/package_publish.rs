//! `cozy package lock|publish|yank` through the real binary against a real Tensorhub: Tensorhub's
//! dev-build `cozy_ref` job starts one (TENSORHUB_URL, TENSORHUB_LOG) with MinIO, and uv is on the
//! runner. Elsewhere it skips. One test, two accounts: the Hub allows a sign-up a minute.
mod common;

use std::path::{Path, PathBuf};

use common::{Home, Out};
use serde_json::Value;

/// A home signed in to the CI Hub as `email`, and its account.
fn publisher(email: &str) -> Option<(Home, String)> {
    let home = Home::with_hub()?;
    let account = home.sign_in(email);
    Some((home, account))
}

/// A fixture project copied into the home, where `cozy` runs on it.
fn project(home: &Home, fixture: &str) -> PathBuf {
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/package")
        .join(fixture);
    let to = home.path().join("src").join(fixture);
    copy(&from, &to);
    to
}

fn copy(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        match entry.file_type().unwrap().is_dir() {
            true => copy(&entry.path(), &to.join(entry.file_name())),
            false => drop(std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap()),
        }
    }
}

fn cozy(home: &Home, dir: &Path, args: &[&str]) -> Out {
    common::out(home.command().current_dir(dir).args(args).output().unwrap())
}

fn json(home: &Home, dir: &Path, args: &[&str]) -> Value {
    let out = cozy(home, dir, &[&["--json"], args].concat());
    assert_eq!(out.code, 0, "{args:?}: {out:?}");
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

fn refused(home: &Home, dir: &Path, args: &[&str]) -> Value {
    let out = cozy(home, dir, &[&["--json"], args].concat());
    assert_eq!(out.code, 1, "{args:?}: {out:?}");
    out.refusal()
}

fn read(path: PathBuf) -> String {
    std::fs::read_to_string(path).unwrap()
}

/// - What must never ship is refused, and so is a project that declares the account index itself.
/// - A release publishes once: the second publish uploads nothing.
/// - A package naming its author's packages through `{ index = "tensorhub" }` locks against that
///   account's index on this Hub without its authored files changing, and publishes.
/// - Another account publishes the same tree: its lock is rebound to that account's index, every
///   version kept.
/// - A yank is idempotent and shows in `info`.
#[test]
fn packages_lock_publish_and_yank_on_a_real_hub() {
    let Some((home, account)) = publisher("publisher@cozy.test") else {
        return;
    };
    let url = std::env::var("TENSORHUB_URL").unwrap();
    let tiny = project(&home, "tiny");
    let user = project(&home, "tiny-user");

    std::fs::write(tiny.join(".env"), "TOKEN=x\n").unwrap();
    assert_eq!(
        refused(&home, &tiny, &["package", "publish"])["code"],
        "package_source_refused"
    );
    std::fs::remove_file(tiny.join(".env")).unwrap();
    let authored = read(user.join("pyproject.toml"));
    let declared = format!(
        "{authored}\n[[tool.uv.index]]\nname = \"tensorhub\"\nurl = \"https://x.invalid/simple/\"\n"
    );
    std::fs::write(user.join("pyproject.toml"), declared).unwrap();
    assert_eq!(
        refused(&home, &user, &["package", "lock"])["code"],
        "account_index_declared"
    );
    std::fs::write(user.join("pyproject.toml"), &authored).unwrap();

    let locked = json(&home, &tiny, &["package", "lock"]);
    assert_eq!(locked["changed"], true, "{locked}");
    assert!(locked["account_index"].is_null(), "{locked}");
    assert_eq!(json(&home, &tiny, &["package", "lock"])["changed"], false);
    let published = json(&home, &tiny, &["package", "publish"]);
    assert_eq!(
        published["package"],
        format!("{account}/tiny"),
        "{published}"
    );
    assert_eq!(published["status"], "published", "{published}");
    assert!(
        published["uploaded_bytes"].as_u64().unwrap() > 0,
        "{published}"
    );
    let again = json(&home, &tiny, &["package", "publish"]);
    assert_eq!(again["status"], "already published", "{again}");
    assert_eq!(again["uploaded_bytes"], 0, "{again}");

    let locked = json(&home, &user, &["package", "lock"]);
    let index = format!("{}/v1/index/{account}/simple/", url.trim_end_matches('/'));
    assert_eq!(locked["account_index"], index.as_str(), "{locked}");
    assert!(read(user.join("uv.lock")).contains(&index));
    assert_eq!(read(user.join("pyproject.toml")), authored);
    let published = json(&home, &user, &["package", "publish"]);
    assert_eq!(
        published["package"],
        format!("{account}/tiny-user"),
        "{published}"
    );
    let info = json(
        &home,
        &user,
        &["package", "info", &format!("{account}/Tiny-User")],
    );
    assert_eq!(info["releases"][0]["release"], "0.1.0", "{info}");

    // The same trees, locked for the first account, published by a second.
    let Some((second, other)) = publisher("second@cozy.test") else {
        return;
    };
    for fixture in ["tiny", "tiny-user"] {
        let dir = project(&second, fixture);
        std::fs::copy(
            home.path().join("src").join(fixture).join("uv.lock"),
            dir.join("uv.lock"),
        )
        .unwrap();
        let published = json(&second, &dir, &["package", "publish"]);
        assert_eq!(
            published["package"],
            format!("{other}/{fixture}"),
            "{published}"
        );
        assert_eq!(published["status"], "published", "{published}");
    }

    let release = format!("{account}/tiny@0.1.0");
    assert_eq!(
        json(&home, &tiny, &["package", "yank", &release])["changed"],
        true
    );
    assert_eq!(
        json(&home, &tiny, &["package", "yank", &release])["changed"],
        false
    );
    let info = json(
        &home,
        &tiny,
        &["package", "info", &format!("{account}/tiny")],
    );
    assert_eq!(info["releases"][0]["yanked"], true, "{info}");
}
