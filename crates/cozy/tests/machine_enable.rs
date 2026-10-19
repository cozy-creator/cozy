//! `cozy machine enable|disable` on this computer. A test's home is not the login home, so the
//! machine runs detached (the login home's service is the CI-home rehearsal's to prove): with one
//! warning that names why, under a user manager or without one. Disable keeps every file, and
//! enable again serves the same machine warm. An update runs a `cozy` renamed over the machine's
//! launch path, and rolls a broken one back.
#![cfg(target_os = "linux")]
mod common;

use std::{
    os::unix::fs::MetadataExt as _,
    path::{Path, PathBuf},
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
    assert!(
        systemctl(&["show-environment"]).status.success(),
        "no systemd user manager: the job enables linger and exports XDG_RUNTIME_DIR and \
         DBUS_SESSION_BUS_ADDRESS"
    );
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

/// A file's identity: a swap renames another file into place.
fn id(path: &Path) -> (u64, u64) {
    let meta = std::fs::metadata(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    (meta.dev(), meta.ino())
}

/// The file the machine on `root` runs: its lock holder's executable.
fn running(root: &Path) -> (u64, u64) {
    let lock = platform::lock::LockFile::new(root.join("var/lib/cozy/machine/agent.lock"));
    let record = lock.holder().unwrap().expect("a machine holds its root");
    let birth: platform::process::Birth = serde_json::from_slice(&record).unwrap();
    id(Path::new(&format!("/proc/{}/exe", birth.pid)))
}

/// A minimal wheel of `distribution` at 0.0.1: its METADATA alone.
fn wheel(dir: &Path, distribution: &str) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{distribution}-0.0.1-py3-none-any.whl"));
    let made = Command::new("python3")
        .args([
            "-I",
            "-c",
            "import sys, zipfile\n\
             path, distribution = sys.argv[1:]\n\
             with zipfile.ZipFile(path, 'w') as w:\n\
             \x20   w.writestr(f'{distribution}-0.0.1.dist-info/METADATA', \
             f'Metadata-Version: 2.1\\nName: {distribution}\\nVersion: 0.0.1\\n')",
        ])
        .arg(&path)
        .arg(distribution)
        .status()
        .unwrap();
    assert!(made.success());
    path
}

/// `cozy update` leaves a new `cozy` at the launch path and the one it replaced beside it as
/// `.cozy-previous`. The machine's next update drains, its parent execs the new file in place, and
/// that binary serves. One whose service never proves readiness (`COZY_TEST_BROKEN_COZY`: a build
/// that exits once ready) is rolled back: the previous file is renamed back and serves. Under the
/// job's user manager, as the first test; a test's home runs its machine detached.
#[test]
fn an_update_runs_a_renamed_binary_and_rolls_a_broken_one_back() {
    let Some(home) = Home::new() else { return };
    let broken = std::env::var_os("COZY_TEST_BROKEN_COZY").expect(
        "COZY_TEST_BROKEN_COZY: a cozy build whose machine exits once ready (the job builds it)",
    );
    // cozy installed in the home, as the installer puts it: the machine's launch path.
    let bin = home.root().join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let launch = bin.join("cozy");
    std::fs::copy(env!("CARGO_BIN_EXE_cozy"), &launch).unwrap();
    home.use_binary(&launch);
    let _cleanup = Cleanup(&home, true);
    json(&cozy(&home, &["machine", "enable", "--json"]));
    let root = home.root().join("machine");
    assert_eq!(running(&root), id(&launch));
    // Its software: a minimal pair, which later updates name again unchanged.
    let wheels = home.path().join("wheels");
    let (runtime, tensorfs) = (wheel(&wheels, "cozy_runtime"), wheel(&wheels, "tensorfs"));
    let update = || {
        let out = cozy(
            &home,
            &[
                "machine",
                "update",
                "local",
                "--runtime-wheel",
                runtime.to_str().unwrap(),
                "--tensorfs-wheel",
                tensorfs.to_str().unwrap(),
                "--json",
            ],
        );
        let last: Value = serde_json::from_str(out.stdout.lines().last().unwrap_or_default())
            .unwrap_or_else(|e| panic!("{e}: {out:?}"));
        (out, last)
    };
    let (out, last) = update();
    assert_eq!(
        (out.code, last["type"].as_str()),
        (0, Some("run.completed")),
        "{out:?}"
    );
    let swap = |to: &Path| {
        let previous = bin.join(".cozy-previous");
        let _ = std::fs::remove_file(&previous);
        std::fs::hard_link(&launch, &previous).unwrap();
        let staged = bin.join(".cozy-staged");
        std::fs::copy(to, &staged).unwrap();
        std::fs::rename(&staged, &launch).unwrap();
        id(&launch)
    };

    // Another file of this build, renamed over the launch path: the update runs it.
    let first = id(&launch);
    let second = swap(Path::new(env!("CARGO_BIN_EXE_cozy")));
    assert_ne!(second, first);
    let (out, last) = update();
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["result"]["to"]["cozy"].as_str()
        ),
        (0, Some("run.completed"), Some(env!("CARGO_PKG_VERSION"))),
        "{out:?}"
    );
    assert_eq!(
        running(&root),
        second,
        "the machine does not run the renamed binary"
    );

    // A build whose machine never proves readiness: rolled back to the one it replaced.
    swap(Path::new(&broken));
    let (out, last) = update();
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["reason"]["code"].as_str()
        ),
        (1, Some("run.failed"), Some("update_rolled_back")),
        "{out:?}"
    );
    assert_eq!(
        id(&launch),
        second,
        "the previous binary is not back at the launch path"
    );
    assert!(!bin.join(".cozy-previous").exists());
    assert_eq!(
        running(&root),
        second,
        "the machine does not run the previous binary"
    );
    json(&cozy(&home, &["machine", "show", "local", "--json"]));
}
