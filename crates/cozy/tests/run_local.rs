//! Local code on a real machine: `cozy run ./script.py` and `cozy run ./dir/fn`. TensorD
//! (COZY_TEST_TENSORD) installs the code it is sent, answers its interface, and runs it; an
//! edited script runs as edited. Nothing Python runs on this computer.
#![cfg(unix)]
mod common;

use std::path::Path;

use common::{Home, Out};
use serde_json::Value;

const SCRIPT: &str = r#"# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.25.1"]
# [tool.cozy]
# accelerator = false
# ///
def main() -> str:
    return "hello from a script"
"#;

fn cozy(home: &Home, dir: &Path, args: &[&str]) -> Out {
    common::out(home.command().current_dir(dir).args(args).output().unwrap())
}

/// A followed `--json` run's end, which must be a completion.
fn completed(out: &Out) -> Value {
    assert_eq!(out.code, 0, "{out:?}");
    let end = out
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .rfind(|e| e["type"] == "run.completed");
    end.unwrap_or_else(|| panic!("no completion: {out:?}"))["run"].clone()
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

#[test]
fn local_code_runs_on_a_real_machine_and_runs_again_as_edited() {
    let Some(home) = Home::new() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    std::fs::create_dir_all(&work).unwrap();

    // A script: the machine installs its generated project, answers `main`'s interface, runs it.
    std::fs::write(work.join("greet.py"), SCRIPT).unwrap();
    let described = cozy(
        &home,
        &work,
        &["run", "./greet.py", "--describe", "--on", "m1"],
    );
    assert_eq!(described.code, 0, "{described:?}");
    assert!(
        described.stdout.contains("request: none") && described.stdout.contains("runs on: CPU"),
        "{described:?}"
    );
    let run = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "greet.py", "--on", "m1"],
    ));
    assert_eq!(run["machine"], "m1");
    assert_eq!(run["result"]["value"], "hello from a script", "{run}");
    assert!(
        run["target"].as_str().unwrap().starts_with("local/"),
        "{run}"
    );

    // Edited, it runs as edited: only the changed file travels.
    std::fs::write(
        work.join("greet.py"),
        SCRIPT.replace("hello from a script", "edited, and run again"),
    )
    .unwrap();
    let again = completed(&cozy(
        &home,
        &work,
        &["--json", "run", "greet.py", "--on", "m1"],
    ));
    assert_eq!(again["result"]["value"], "edited, and run again", "{again}");

    // A project directory: its functions, then one of them.
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    let listed = cozy(&home, &work, &["--json", "run", "./tiny", "--on", "m1"]);
    assert_eq!(listed.code, 0, "{listed:?}");
    let functions: Value = serde_json::from_str(&listed.stdout).unwrap();
    assert_eq!(functions["functions"][0]["function"], "echo", "{functions}");
    let echoed = completed(&cozy(
        &home,
        &work,
        &[
            "--json",
            "run",
            "./tiny/echo",
            "from a directory",
            "--on",
            "m1",
        ],
    ));
    assert_eq!(echoed["result"]["text"], "from a directory", "{echoed}");
}
