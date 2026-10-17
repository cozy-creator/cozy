//! xs-044: a home that a released `cozy` (`COZY_TEST_OLD_COZY`, v0.9.3) filled with runs, their
//! specs and outcomes kept as protobuf, opened by this build. The store is copied aside first, every
//! row becomes the JSON v0.10 reads, every run reads back as it was, and the older build refuses the
//! store it can no longer read. With `COZY_TEST_NEXT_COZY`, the cohort's build reads them the same too.
#![cfg(unix)]
mod common;

use std::path::Path;

use common::{Home, Out};
use cozy_proto::{machine as v1, machine_json as json};
use prost::Message as _;
use serde_json::Value;

const KEPT: &str = "# /// script\n# requires-python = \">=3.12\"\n# dependencies = [\"cozy-runtime>=0.25.1\"]\n\
# [tool.cozy]\n# accelerator = false\n# ///\ndef main() -> str:\n    return \"kept\"\n";

const FAILS: &str = "# /// script\n# requires-python = \">=3.12\"\n# dependencies = [\"cozy-runtime>=0.25.1\"]\n\
# [tool.cozy]\n# accelerator = false\n# ///\ndef main() -> str:\n    raise RuntimeError(\"kept failing\")\n";

/// A followed `--json` run's last event: its type and its run.
fn ended(out: &Out) -> (String, Value) {
    let end = out
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .rfind(|e| e["type"].as_str().is_some_and(|t| t.starts_with("run.")));
    let end = end.unwrap_or_else(|| panic!("no end: {out:?}"));
    (end["type"].as_str().unwrap().to_owned(), end["run"].clone())
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

/// Each run's spec and outcome bytes as the store holds them.
fn rows(db: &Path) -> Vec<(String, Vec<u8>, Option<Vec<u8>>)> {
    let conn = rusqlite::Connection::open(db).unwrap();
    let mut rows = conn
        .prepare("SELECT id, spec, outcome FROM runs ORDER BY number")
        .unwrap();
    rows.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn a_released_homes_runs_read_back_the_same_once_its_rows_are_json() {
    let Some(home) = Home::new() else { return };
    let Some(old) = std::env::var_os("COZY_TEST_OLD_COZY") else {
        eprintln!("skipped: no released cozy (COZY_TEST_OLD_COZY is unset)");
        return;
    };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    home.use_binary(&old);
    let _machine = common::tensord::added(&home, "m1");
    let work = home.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("kept.py"), KEPT).unwrap();
    std::fs::write(work.join("fails.py"), FAILS).unwrap();
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny-output"),
        &work.join("tiny-output"),
    );
    let run = |args: &[&str]| {
        let out = common::out(
            home.command()
                .current_dir(&work)
                .args(args)
                .output()
                .unwrap(),
        );
        ended(&out)
    };
    let (kept, value) = run(&["--json", "run", "kept.py", "--on", "m1"]);
    assert_eq!(
        (kept.as_str(), &value["result"]["value"]),
        ("run.completed", &Value::from("kept"))
    );
    let (wrote, file) = run(&[
        "--json",
        "run",
        "./tiny-output/write",
        "text=a file",
        "--on",
        "m1",
    ]);
    assert_eq!(wrote, "run.completed", "{file}");
    let (failed, _) = run(&["--json", "run", "fails.py", "--on", "m1"]);
    assert_eq!(failed, "run.failed");
    assert_eq!(home.cozy(["down"]).code, 0);

    // What a device would have measured, written into the released build's protobuf row.
    let db = home.root().join("calcifer/state.db");
    let before = rows(&db);
    assert_eq!(before.len(), 3, "{before:?}");
    let measured = br#"{"execution":{"degree":1,"ranks":[{"gpu":0,"attention":"sdpa"}]}}"#;
    {
        let conn = rusqlite::Connection::open(&db).unwrap();
        let (id, _, outcome) = &before[1];
        let mut outcome = v1::Outcome::decode(outcome.as_deref().unwrap()).unwrap();
        outcome.measurements = measured.to_vec();
        conn.execute(
            "UPDATE runs SET outcome = ?2 WHERE id = ?1",
            rusqlite::params![id, outcome.encode_to_vec()],
        )
        .unwrap();
    }
    let before = rows(&db);
    let ids: Vec<&str> = before.iter().map(|(id, ..)| id.as_str()).collect();
    let shown = |ids: &[&str]| -> Vec<Value> {
        ids.iter()
            .map(|id| home.json(["run", "show", id, "--json"]))
            .collect()
    };
    let shown_before = shown(&ids);
    assert_eq!(
        shown_before[1]["measurements"]["execution"]["degree"], 1,
        "{}",
        shown_before[1]
    );
    assert_eq!(home.cozy(["down"]).code, 0);

    // This build: the store is copied aside, then its rows are JSON, read back as they were.
    home.use_binary(env!("CARGO_BIN_EXE_cozy"));
    assert_eq!(shown(&ids), shown_before);
    let calcifer = std::fs::read_dir(home.root().join("calcifer")).unwrap();
    let copies: Vec<_> = calcifer
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|name| name.starts_with("state.db.pre-json-"))
        .collect();
    assert_eq!(copies.len(), 1, "{copies:?}");
    let conn = rusqlite::Connection::open(&db).unwrap();
    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap();
    assert_eq!(version, 5);
    for ((id, spec, outcome), (_, json_spec, json_outcome)) in before.iter().zip(rows(&db)) {
        let spec = v1::RunSpec::decode(spec.as_slice()).unwrap();
        let read: json::RunSpec = serde_json::from_slice(&json_spec).unwrap();
        assert_eq!(v1::RunSpec::from(&read), spec, "run {id}'s spec");
        let outcome = v1::Outcome::decode(outcome.as_deref().unwrap()).unwrap();
        let read: json::Outcome = serde_json::from_slice(&json_outcome.unwrap()).unwrap();
        assert_eq!(v1::Outcome::from(&read), outcome, "run {id}'s outcome");
    }
    let read: json::Outcome = serde_json::from_slice(rows(&db)[1].2.as_deref().unwrap()).unwrap();
    assert_eq!(read.measurements.unwrap().get().as_bytes(), measured);
    drop(conn);

    // A second open converts nothing again; the released build refuses a store it cannot read.
    assert_eq!(home.cozy(["down"]).code, 0);
    assert_eq!(shown(&ids), shown_before);
    let calcifer = std::fs::read_dir(home.root().join("calcifer")).unwrap();
    let again = calcifer
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|name| name.starts_with("state.db.pre-json-"))
        .count();
    assert_eq!(again, 1);

    // The cohort's build (`COZY_TEST_NEXT_COZY`, v0.10 on the HTTP API) reads the same runs.
    if let Some(next) = std::env::var_os("COZY_TEST_NEXT_COZY") {
        assert_eq!(home.cozy(["down"]).code, 0);
        home.use_binary(&next);
        assert_eq!(shown(&ids), shown_before);
        assert_eq!(home.cozy(["down"]).code, 0);
    }
    home.use_binary(&old);
    let refused = home.cozy(["run", "show", ids[0], "--json"]);
    assert_ne!(refused.code, 0, "{refused:?}");
    home.use_binary(env!("CARGO_BIN_EXE_cozy"));
}
