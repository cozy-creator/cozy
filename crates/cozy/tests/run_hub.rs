//! Runs that complete on a real machine: the job's Tensorhub (Tensorhub's dev-build `cozy_ref`
//! job: TENSORHUB_URL, TENSORHUB_LOG) holds just-published torch-free packages, and TensorD from
//! the cozy-runtime wheel (COZY_TEST_TENSORD) installs and runs them. Elsewhere it skips.
#![cfg(unix)]
mod common;

use std::{
    io::BufRead,
    path::{Path, PathBuf},
    process::Stdio,
};

use common::Home;
use serde_json::Value;
use sha2::{Digest as _, Sha256};

/// Locks and publishes a fixture package from a copy in the home, as its author would.
fn publish(home: &Home, fixture: &str) {
    let from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/package")
        .join(fixture);
    let project = home.path().join("src").join(fixture);
    copy(&from, &project);
    for verb in ["lock", "publish"] {
        let out = common::out(
            home.command()
                .current_dir(&project)
                .args(["--json", "package", verb])
                .output()
                .unwrap(),
        );
        assert_eq!(out.code, 0, "package {verb} {fixture}: {out:?}");
    }
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

/// Reads NDJSON lines until one satisfies `want`.
fn until(reader: &mut impl BufRead, want: impl Fn(&Value) -> bool) -> Value {
    loop {
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0, "stdout closed");
        let event: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line}"));
        if want(&event) {
            return event;
        }
    }
}

/// Every file under `dir`.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![];
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        match entry.file_type().unwrap().is_dir() {
            true => out.extend(files(&entry.path())),
            false => out.push(entry.path()),
        }
    }
    out
}

/// One account (the Hub paces sign-ups), two packages:
/// - a published function runs to completion on its machine and its result comes back; a field
///   its interface lacks is dropped with a warning, so the package's strict struct never sees it;
/// - an output the machine writes lands once in `--out`, its bytes the digest the run records,
///   though Calcifer stopped while the run went on.
#[test]
fn published_functions_run_to_completion_on_a_real_machine() {
    let Some(home) = Home::with_hub() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    let account = home.sign_in("runner@cozy.test");
    publish(&home, "tiny");
    publish(&home, "tiny-output");
    let _machine = common::tensord::added(&home, "m1");

    let echo = format!("{account}/tiny/echo");
    let out = home.cozy([
        "--json",
        "run",
        &echo,
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
    let human = home.cozy(["run", &echo, "again", "--on", "m1"]);
    assert_eq!(human.code, 0, "{human:?}");
    assert!(
        human.stdout.contains("completed") && human.stdout.contains("again"),
        "{human:?}"
    );
    let shown = home.json(["run", "show", &number, "--json"]);
    assert_eq!(
        (
            shown["status"].as_str(),
            shown["warnings"][0]["code"].as_str()
        ),
        (Some("completed"), Some("request_fields_ignored"))
    );

    // The writer waits on its machine while Calcifer stops; the follower starts it again and
    // reattaches, and the file lands once.
    let landing = home.path().join("landing");
    let write = format!("{account}/tiny-output/write");
    let text = "a file the machine wrote";
    let mut follower = home
        .command()
        .args([
            "--json",
            "run",
            &write,
            &format!("text={text}"),
            "seconds=8",
            "--on",
            "m1",
            "--out",
        ])
        .arg(&landing)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = std::io::BufReader::new(follower.stdout.take().unwrap());
    let accepted = until(&mut stdout, |e| e["type"] == "run.in_progress");
    let number = accepted["run"]["number"].as_u64().unwrap().to_string();
    let down = home.cozy(["down"]);
    assert_eq!(down.code, 0, "{down:?}");
    let ended = until(&mut stdout, |e| {
        e.get("type").and_then(Value::as_str).is_some_and(|t| {
            t.starts_with("run.") && ["run.completed", "run.failed", "run.canceled"].contains(&t)
        })
    });
    assert_eq!(ended["type"], "run.completed", "{ended}");
    assert_eq!(follower.wait().unwrap().code(), Some(0));

    let shown = home.json(["run", "show", &number, "--json"]);
    let item = &shown["output"][0];
    assert_eq!(
        (item["name"].as_str(), item["status"].as_str()),
        (Some("file"), Some("completed")),
        "{shown}"
    );
    let landed = files(&landing);
    assert_eq!(landed.len(), 1, "the output landed once: {landed:?}");
    assert_eq!(
        item["path"].as_str().map(PathBuf::from),
        Some(landed[0].clone()),
        "{shown}"
    );
    let bytes = std::fs::read(&landed[0]).unwrap();
    assert_eq!(bytes, text.as_bytes());
    assert_eq!(
        item["sha256"].as_str(),
        Some(format!("sha256:{}", hex::encode(Sha256::digest(&bytes))).as_str()),
        "{shown}"
    );
}
