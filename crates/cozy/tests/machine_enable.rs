//! `cozy machine enable|disable` on this computer. Under the session's systemd user manager (the
//! job enables linger), a machine an older cozy started (a transient `tensord-*` unit with the
//! laptop's environment verbatim; its program is COZY_TEST_TENSORD, since PyPI no longer serves the
//! Runtime that bundled 0.7.3, which the CI-home rehearsal runs) is taken over on the same
//! root, store, ports, worker id and keys. A test's home is not the login home, so the new machine
//! runs detached (the login home's service is the CI-home rehearsal's to prove). Disable keeps every
//! file, and enable again serves the same machine warm.
#![cfg(target_os = "linux")]
mod common;

use std::{
    net::TcpListener,
    os::unix::fs::PermissionsExt as _,
    path::Path,
    process::{Command, Output},
    time::{Duration, Instant},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use common::{Home, Out};
use ed25519_dalek::pkcs8::EncodePrivateKey as _;
use serde_json::Value;

const LEGACY: &str = "tensord-enable-test";
const WORKER: &str = "om-enable-test";

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

fn property(unit: &str, name: &str) -> String {
    let out = systemctl(&["show", unit, "-p", name, "--value"]);
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

fn port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Ends what this test started, however it ended: under the manager too when it used one.
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
        if managed {
            for unit in [LEGACY, "cozy-machine", "cozy-calcifer"] {
                let _ = systemctl(&["disable", "--now", &format!("{unit}.service")]);
            }
        }
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

#[test]
fn enable_takes_over_an_older_machine_and_disable_keeps_its_state() {
    let Some(home) = Home::new() else { return };
    let Some(tensord) = std::env::var_os("COZY_TEST_TENSORD") else {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    };
    if !systemctl(&["show-environment"]).status.success() {
        eprintln!("skipped: no systemd user manager in this session");
        return;
    }
    let _cleanup = Cleanup(&home, true);

    // The laptop's layout: TensorD under the machine root, the store in ~/.tensorfs, and the
    // controller's owner key, whose public half the unit's environment admits.
    let root = home.root().join("machine/root");
    let program = root.join("usr/local/bin/tensord");
    std::fs::create_dir_all(program.parent().unwrap()).unwrap();
    std::fs::copy(&tensord, &program).unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    let store = home.path().join(".tensorfs");
    let keys = home.root().join("keys");
    std::fs::create_dir_all(&keys).unwrap();
    let owner = ed25519_dalek::SigningKey::from_bytes(&[5; 32]);
    std::fs::write(
        keys.join("owner.pem"),
        owner.to_pkcs8_pem(Default::default()).unwrap().as_bytes(),
    )
    .unwrap();
    std::fs::set_permissions(
        keys.join("owner.pem"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    let public = URL_SAFE_NO_PAD.encode(owner.verifying_key().as_bytes());
    let receipt = home.root().join("machine/receipt-key");
    std::fs::write(&receipt, URL_SAFE_NO_PAD.encode([7u8; 32])).unwrap();
    std::fs::set_permissions(&receipt, std::fs::Permissions::from_mode(0o600)).unwrap();
    let (api, media) = (port(), port());
    let env = [
        format!("COZY_MACHINE_ROOT={}", root.display()),
        "COZY_LISTEN_HOST=127.0.0.1".into(),
        format!("COZY_WORKER_ID={WORKER}"),
        "COZY_MACHINE_LIFETIME=persistent".into(),
        format!("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE={}", receipt.display()),
        format!("COZY_AUTHORIZED_KEYS={public}"),
        format!("COZY_TENSORFS_ROOT={}", store.display()),
        format!("COZY_WORKER_INTERNAL_PORT={api}"),
        format!("COZY_MEDIA_INTERNAL_PORT={media}"),
        format!("HOME={}", home.path().display()),
        "CUDA_VISIBLE_DEVICES=".into(),
    ];
    let mut start = Command::new("systemd-run");
    start.args([
        "--user",
        &format!("--unit={LEGACY}"),
        "-p",
        "CollectMode=inactive-or-failed",
    ]);
    start.arg(format!("--working-directory={}", root.display()));
    for pair in &env {
        start.arg(format!("--setenv={pair}"));
    }
    let started = start
        .arg(&program)
        .args(["run", "--persistent"])
        .output()
        .unwrap();
    assert!(started.status.success(), "{started:?}");
    // A hang bound for a machine that never starts; the product waits without one.
    let until = Instant::now() + Duration::from_secs(120);
    let ready = root.join("var/lib/cozy/tensord/api-ready.json");
    while !(ready.exists() && std::net::TcpStream::connect(("127.0.0.1", api)).is_ok()) {
        assert_eq!(
            property(LEGACY, "ActiveState"),
            "active",
            "the older machine stopped"
        );
        assert!(Instant::now() < until, "the older machine never served");
        std::thread::sleep(Duration::from_millis(100));
    }

    // A machine that does not start leaves the older one running as it did: its start log,
    // where the new machine says anything before its own log opens, cannot be opened.
    let start_log = root.join("var/lib/cozy/machine/start.log");
    std::fs::create_dir_all(&start_log).unwrap();
    let failed = cozy(&home, &["machine", "enable", "--json"]);
    assert_eq!(
        (failed.code, failed.refusal()["code"].as_str()),
        (1, Some("machine_start_failed")),
        "{failed:?}"
    );
    assert!(
        failed.refusal()["remedy"]
            .as_str()
            .is_some_and(|r| r.contains("runs again")),
        "{failed:?}"
    );
    // A hang bound for the revived unit's start; the product waits without one.
    let until = Instant::now() + Duration::from_secs(60);
    while std::net::TcpStream::connect(("127.0.0.1", api)).is_err() {
        assert!(
            Instant::now() < until,
            "the older machine did not run again"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(property(LEGACY, "ActiveState"), "active");
    std::fs::remove_dir(&start_log).unwrap();

    // Enable takes it over: the unit is gone, and its settings are the config's.
    let enabled = json(&cozy(&home, &["machine", "enable", "--json"]));
    assert_eq!(
        (
            enabled["changed"].as_bool(),
            enabled["machine"]["name"].as_str()
        ),
        (Some(true), Some("local")),
        "{enabled}"
    );
    assert_ne!(
        property(LEGACY, "ActiveState"),
        "active",
        "{LEGACY} still runs"
    );
    // A test's home is not the login home: the machine runs detached, and holds its root.
    assert!(held(&root), "no machine holds {}", root.display());
    let config = std::fs::read_to_string(home.config_path()).unwrap();
    for line in [
        format!("root = \"{}\"", root.display()),
        format!("store = \"{}\"", store.display()),
        format!("port = {api}"),
        format!("webrtc_port = {media}"),
        format!("worker_id = \"{WORKER}\""),
    ] {
        assert!(config.contains(&line), "{line} not in {config}");
    }
    let shown = json(&cozy(&home, &["machine", "show", "local", "--json"]));
    assert_eq!(shown["address"], format!("127.0.0.1:{api}"), "{shown}");

    // Local code installs once, then runs warm.
    let work = home.path().join("work");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/package/tiny"),
        &work.join("tiny"),
    );
    ran(managed(&home), &work, "first");
    assert!(
        !ran(managed(&home), &work, "again"),
        "a second run installed again"
    );

    // Enabling an enabled machine changes nothing.
    let again = json(&cozy(&home, &["machine", "enable", "--json"]));
    assert_eq!(again["changed"].as_bool(), Some(false), "{again}");

    // Disable stops it and keeps every file; enable serves the same machine warm.
    let disabled = json(&cozy(&home, &["machine", "disable", "--json"]));
    assert_eq!(disabled["changed"].as_bool(), Some(true), "{disabled}");
    assert!(!held(&root), "the machine still holds its root");
    assert!(ready.exists() && store.read_dir().unwrap().next().is_some());
    let back = json(&cozy(&home, &["machine", "enable", "--json"]));
    assert_eq!(back["changed"].as_bool(), Some(true), "{back}");
    assert!(
        !ran(managed(&home), &work, "back"),
        "the re-enabled machine installed again"
    );
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

/// Where no service manager answers (the fixture's commands leave the session's out), enable runs
/// the machine detached with one warning; disable ends it and keeps every file; enable again
/// serves the same machine warm.
#[test]
fn without_a_service_manager_the_machine_runs_detached() {
    let Some(home) = Home::new() else { return };
    let _cleanup = Cleanup(&home, false);
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
    assert!(
        !held(&home.root().join("machine")),
        "the machine still holds its root"
    );
    let back = json(&home.cozy(["machine", "enable", "--json"]));
    assert_eq!(back["changed"].as_bool(), Some(true), "{back}");
    assert!(
        !ran(home.command(), &work, "back"),
        "the re-enabled machine installed again"
    );
}
