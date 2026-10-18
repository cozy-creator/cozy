//! `cozy machine enable|disable` on this computer. A test's home is not the login home, so the
//! machine runs detached (the login home's service is the CI-home rehearsal's to prove): with one
//! warning that names why, under a user manager or without one. Disable keeps every file, and
//! enable again serves the same machine warm.
#![cfg(target_os = "linux")]
mod common;

use std::{
    path::Path,
    process::{Command, Output},
};

use common::{Home, Out};
use serde_json::Value;

fn systemctl(args: &[&str]) -> Output {
    Command::new("systemctl")
        .arg("--user")
        .args(args)
        .output()
        .unwrap()
}

/// `cozy` in this home with this session's user manager, which the fixture's commands leave out:
/// Calcifer and the machine run as its services.
fn managed(home: &Home) -> Command {
    let mut command = home.command();
    for var in ["XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"] {
        if let Some(value) = std::env::var_os(var) {
            command.env(var, value);
        }
    }
    command
}

fn cozy(home: &Home, args: &[&str]) -> Out {
    common::out(managed(home).args(args).output().unwrap())
}

/// Ends what this test started, however it ended, in the environment it ran in.
struct Cleanup<'a>(&'a Home, bool);

impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        let Cleanup(home, managed) = *self;
        let command = |args: &[&str]| match managed {
            true => drop(cozy(home, args)),
            false => drop(home.cozy(args)),
        };
        command(&["machine", "disable", "--json"]);
        command(&["down", "--json"]);
    }
}

/// Whether a machine holds `root` (its `agent.lock`).
fn held(root: &Path) -> bool {
    let lock = root.join("var/lib/cozy/machine/agent.lock");
    std::fs::File::open(lock).is_ok_and(|file| file.try_lock().is_err())
}

fn json(out: &Out) -> Value {
    assert_eq!(out.code, 0, "{out:?}");
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

/// A followed `--json` run's end, which must complete, and whether it installed or synced.
fn ran(mut command: Command, work: &Path, text: &str) -> bool {
    let out = common::out(
        command
            .current_dir(work)
            .args(["--json", "run", "./tiny/echo", text, "--on", "local"])
            .output()
            .unwrap(),
    );
    assert_eq!(out.code, 0, "{out:?}");
    let events: Vec<Value> = out
        .stdout
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let end = events.iter().rfind(|e| e["type"] == "run.completed");
    assert_eq!(
        end.unwrap_or_else(|| panic!("{out:?}"))["run"]["result"]["text"],
        text
    );
    events
        .iter()
        .filter_map(|e| e["progress"]["stage"].as_str())
        .any(|stage| stage.contains("install") || stage.contains("sync"))
}

/// Under the session's user manager (the job enables linger), a home other than the login one
/// gets no service: the machine runs detached and says why, once.
#[test]
fn under_a_user_manager_another_home_runs_detached() {
    let Some(home) = Home::new() else { return };
    if !systemctl(&["show-environment"]).status.success() {
        eprintln!("skipped: no systemd user manager in this session");
        return;
    }
    let _cleanup = Cleanup(&home, true);
    let first = cozy(&home, &["machine", "enable", "--json"]);
    assert_eq!(json(&first)["changed"].as_bool(), Some(true), "{first:?}");
    assert_eq!(
        first.stderr.matches("not your login home").count(),
        1,
        "{first:?}"
    );
    let root = home.root().join("machine");
    assert!(held(&root), "no machine holds {}", root.display());
    let disabled = json(&cozy(&home, &["machine", "disable", "--json"]));
    assert_eq!(disabled["changed"].as_bool(), Some(true), "{disabled}");
    assert!(!held(&root), "the machine still holds its root");
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

/// Where no service manager answers (the fixture's commands leave the session's out), a machine
/// that cannot start is refused; enable runs it detached with one warning; disable ends it and
/// keeps every file; enable again serves the same machine warm.
#[test]
fn without_a_service_manager_the_machine_runs_detached() {
    let Some(home) = Home::new() else { return };
    let _cleanup = Cleanup(&home, false);
    let root = home.root().join("machine");
    // Its start log, where it says anything before its own log opens, cannot be opened.
    let start_log = root.join("var/lib/cozy/machine/start.log");
    std::fs::create_dir_all(&start_log).unwrap();
    let failed = home.cozy(["machine", "enable", "--json"]);
    assert_eq!(
        (failed.code, failed.refusal()["code"].as_str()),
        (1, Some("machine_start_failed")),
        "{failed:?}"
    );
    assert!(!held(&root), "a refused machine holds its root");
    std::fs::remove_dir(&start_log).unwrap();

    let first = home.cozy(["machine", "enable", "--json"]);
    let enabled = json(&first);
    assert_eq!(
        (
            enabled["changed"].as_bool(),
            enabled["machine"]["name"].as_str()
        ),
        (Some(true), Some("local")),
        "{enabled}"
    );
    assert_eq!(
        first.stderr.matches("no service manager").count(),
        1,
        "{first:?}"
    );
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    ran(home.command(), &work, "first");
    assert!(
        !ran(home.command(), &work, "again"),
        "a second run installed again"
    );
    let again = json(&home.cozy(["machine", "enable", "--json"]));
    assert_eq!(again["changed"].as_bool(), Some(false), "{again}");

    let disabled = json(&home.cozy(["machine", "disable", "--json"]));
    assert_eq!(disabled["changed"].as_bool(), Some(true), "{disabled}");
    assert!(!held(&root), "the machine still holds its root");
    let back = json(&home.cozy(["machine", "enable", "--json"]));
    assert_eq!(back["changed"].as_bool(), Some(true), "{back}");
    assert!(
        !ran(home.command(), &work, "back"),
        "the re-enabled machine installed again"
    );
}
