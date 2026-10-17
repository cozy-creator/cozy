//! A run that completes on a real machine: the job's Tensorhub (Tensorhub's dev-build `cozy_ref`
//! job: TENSORHUB_URL, TENSORHUB_LOG) holds a just-published torch-free package, and TensorD from
//! the cozy-runtime wheel (COZY_TEST_TENSORD) installs it and runs it. Elsewhere it skips.
#![cfg(unix)]
mod common;

use std::path::{Path, PathBuf};

use common::Home;
use serde_json::Value;

/// L-package's `tiny` fixture, copied into the home where `cozy` publishes it.
fn tiny(home: &Home) -> PathBuf {
    let from = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny");
    let to = home.path().join("src/tiny");
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

/// The events a followed `--json` run printed.
fn events(out: &common::Out) -> Vec<Value> {
    out.stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}")))
        .collect()
}

/// A published function runs to completion on its machine and its result comes back; a field
/// its interface lacks is dropped with a warning, so the package's strict struct never sees it.
#[test]
fn a_published_function_runs_to_completion_on_a_real_machine() {
    let Some(home) = Home::with_hub() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    let account = home.sign_in("runner@cozy.test");
    let project = tiny(&home);
    let locked = common::out(
        home.command()
            .current_dir(&project)
            .args(["--json", "package", "lock"])
            .output()
            .unwrap(),
    );
    assert_eq!(locked.code, 0, "{locked:?}");
    let published = common::out(
        home.command()
            .current_dir(&project)
            .args(["--json", "package", "publish"])
            .output()
            .unwrap(),
    );
    assert_eq!(published.code, 0, "{published:?}");
    let _machine = common::tensord::added(&home, "m1");
    let target = format!("{account}/tiny/echo");

    let out = home.cozy([
        "--json",
        "run",
        &target,
        "hi there",
        "mood=calm",
        "--on",
        "m1",
    ]);
    assert_eq!(out.code, 0, "{out:?}");
    assert!(out.stderr.contains("ignored unknown field mood"), "{out:?}");
    let seen = events(&out);
    let end = seen
        .iter()
        .rev()
        .find(|e| e["type"] == "run.completed")
        .unwrap_or_else(|| panic!("{seen:?}"));
    let run = &end["run"];
    assert_eq!(
        (run["machine"].as_str(), run["result"]["text"].as_str()),
        (Some("m1"), Some("hi there")),
        "{run}"
    );
    let number = run["number"].as_u64().unwrap().to_string();

    let human = home.cozy(["run", &target, "again", "--on", "m1"]);
    assert_eq!(human.code, 0, "{human:?}");
    assert!(
        human.stdout.contains("completed") && human.stdout.contains("again"),
        "{human:?}"
    );

    let shown = home.json(["run", "show", &number, "--json"]);
    assert_eq!(shown["status"], "completed", "{shown}");
    assert_eq!(shown["warnings"][0]["code"], "request_fields_ignored");
    let listed = home.json(["run", "list", "--state", "completed", "--json"]);
    assert_eq!(listed["runs"].as_array().unwrap().len(), 2, "{listed}");
}
