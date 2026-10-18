//! `cozy package install DIR|script.py --on` through the real binary: local code installed live on
//! a real TensorD (COZY_TEST_TENSORD) added as a machine, its files sent through Calcifer.
// TensorD runs on Unix only for now, and so does its fixture (common::tensord).
#![cfg(unix)]
mod common;

use std::path::Path;

use common::{Home, Out, package, tensord};
use serde_json::Value;

fn cozy(home: &Home, args: &[&str]) -> Out {
    common::out(home.command().args(args).output().unwrap())
}

/// Installs `path` on `t1` with `--json` and answers its last event, which must complete it.
fn install(home: &Home, path: &Path) -> Value {
    let path = path.to_string_lossy();
    let out = cozy(home, &["--json", "package", "install", &path, "--on", "t1"]);
    let last: Option<Value> = out
        .stdout
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str(l).ok());
    let last = last.unwrap_or_else(|| panic!("{out:?}"));
    assert_eq!(
        (out.code, last["type"].as_str()),
        (0, Some("run.completed")),
        "{last}\n{out:?}"
    );
    last
}

/// The local packages machine `t1` holds, from `cozy package list --on t1`.
fn held(home: &Home) -> Vec<(String, String)> {
    let out = cozy(home, &["--json", "package", "list", "--on", "t1"]);
    assert_eq!(out.code, 0, "{out:?}");
    let listed: Value = serde_json::from_str(&out.stdout).unwrap();
    let rows = listed["packages"].as_array().unwrap().iter();
    let local = rows.filter(|r| r["package"].as_str().unwrap().starts_with("local/"));
    local
        .map(|r| {
            (
                r["package"].as_str().unwrap().into(),
                r["release"].as_str().unwrap().into(),
            )
        })
        .collect()
}

/// A project directory installs as `local/<name>` and an unchanged reinstall keeps one install; a
/// PEP 723 script installs as the package it is; `remove` takes both off the machine.
#[test]
fn local_code_installs_live_on_a_machine() {
    let Some(home) = Home::new() else { return };
    let Some(_) = common::needs("COZY_TEST_TENSORD", common::TENSORD) else {
        return;
    };
    let _machine = tensord::added(&home, "t1");

    let project = package::project(&home, "tiny");
    install(&home, &project);
    assert_eq!(held(&home), [("local/tiny".to_owned(), "0.1.0".to_owned())]);
    install(&home, &project);
    assert_eq!(held(&home), [("local/tiny".to_owned(), "0.1.0".to_owned())]);

    let script = home.path().join("hello.py");
    let text = "# /// script\n# requires-python = \">=3.12\"\n# [tool.cozy]\n# accelerator = false\n# ///\n\
                def main() -> int:\n    return 1\n";
    std::fs::write(&script, text).unwrap();
    install(&home, &script);
    let held_now = held(&home);
    let named = held_now
        .iter()
        .find(|(p, _)| p.starts_with("local/cozy-script-"));
    let (script_package, _) = named.unwrap_or_else(|| panic!("{held_now:?}")).clone();

    for package in ["local/tiny", script_package.as_str()] {
        let out = cozy(
            &home,
            &["--json", "package", "remove", package, "--on", "t1"],
        );
        assert_eq!(out.code, 0, "{out:?}");
    }
    assert!(held(&home).is_empty(), "{:?}", held(&home));
}
