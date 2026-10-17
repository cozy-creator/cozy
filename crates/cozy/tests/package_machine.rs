//! `cozy package install|list|update|remove --on` through the real binary: Calcifer, a real TensorD
//! (COZY_TEST_TENSORD, the cozy-runtime wheel's) added as a machine, and a real Tensorhub (its
//! dev-build `cozy_ref` job) holding a torch-free package this test publishes.
mod common;

use std::path::Path;

use common::{Home, Out, package, tensord};
use serde_json::Value;

fn cozy(home: &Home, dir: &Path, args: &[&str]) -> Out {
    common::out(home.command().current_dir(dir).args(args).output().unwrap())
}

fn json(home: &Home, dir: &Path, args: &[&str]) -> Value {
    let out = cozy(home, dir, &[&["--json"], args].concat());
    assert_eq!(out.code, 0, "{args:?}: {out:?}");
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

/// What a person debugging this test wants: the runs and the machine as Calcifer holds them.
fn state(home: &Home, dir: &Path) -> String {
    let runs = cozy(home, dir, &["--json", "run", "list"]);
    let machine = cozy(home, dir, &["--json", "machine", "show", "t1"]);
    format!("runs: {}\nmachine: {}", runs.stdout, machine.stdout)
}

/// Follows an install with `--json` (NDJSON) and answers its exit code and last event.
fn install(home: &Home, dir: &Path, package: &str) -> (i32, Value) {
    let out = cozy(
        home,
        dir,
        &["--json", "package", "install", package, "--on", "t1"],
    );
    let last = out
        .stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str(line).ok());
    (
        out.code,
        last.unwrap_or_else(|| panic!("{out:?}\n{}", state(home, dir))),
    )
}

/// The releases of `package` machine `t1` holds, from `cozy package list --on t1`, which reads the
/// machine now.
fn held(home: &Home, dir: &Path, package: &str) -> Vec<String> {
    let listed = json(home, dir, &["package", "list", "--on", "t1"]);
    let rows = listed["packages"].as_array().unwrap();
    let held = rows.iter().filter(|r| r["package"] == package);
    held.map(|r| r["release"].as_str().unwrap().to_owned())
        .collect()
}

/// A published package installs on a machine as a run the machine resolves, shows in `list` with
/// its Hub, moves to its Hub's newest release with `update` (the older one leaving), and leaves
/// with `remove`; a package the Hub lacks ends its run with the machine's reason.
#[test]
fn packages_install_update_and_remove_on_a_machine() {
    let Some(home) = Home::with_hub() else { return };
    let account = home.sign_in("installer@cozy.test");
    let _calcifer = home.want_calcifer();
    let _machine = tensord::added(&home, "t1");
    let package = package::publish(&home, "tiny", "0.1.0");
    assert_eq!(package, format!("{account}/tiny"));
    let dir = package::project(&home, "tiny");

    for _ in 0..2 {
        let (code, last) = install(&home, &dir, &package);
        let state = state(&home, &dir);
        assert_eq!(
            (code, last["type"].as_str()),
            (0, Some("run.completed")),
            "{last}\n{state}"
        );
        assert_eq!(held(&home, &dir, &package), ["0.1.0"], "{state}");
    }

    package::publish(&home, "tiny", "0.2.0");
    let updated = json(&home, &dir, &["package", "update", "--on", "t1"]);
    let row = &updated["packages"][0];
    let moved = (row["to"].as_str(), row["status"].as_str());
    assert_eq!(moved, (Some("0.2.0"), Some("updated")), "{updated}");
    assert_eq!(
        held(&home, &dir, &package),
        ["0.2.0"],
        "{updated}\n{}",
        state(&home, &dir)
    );
    let current = json(&home, &dir, &["package", "update", &package, "--on", "t1"]);
    assert_eq!(current["packages"][0]["status"], "current", "{current}");

    let (code, last) = install(&home, &dir, &format!("{account}/nope"));
    assert_eq!(
        (code, last["type"].as_str()),
        (1, Some("run.failed")),
        "{last}"
    );

    let removed = json(&home, &dir, &["package", "remove", &package, "--on", "t1"]);
    assert_eq!(removed["removed"][0]["release"], "0.2.0", "{removed}");
    assert!(
        held(&home, &dir, &package).is_empty(),
        "{removed}\n{}",
        state(&home, &dir)
    );
    let gone = cozy(
        &home,
        &dir,
        &["--json", "package", "remove", &package, "--on", "t1"],
    );
    assert_eq!(gone.code, 1, "{gone:?}");
    assert_eq!(gone.refusal()["code"], "package_not_installed", "{gone:?}");
}
