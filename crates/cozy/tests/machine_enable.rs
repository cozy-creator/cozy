//! `cozy machine enable|disable` under this computer's systemd user manager. A machine an older cozy
//! started (TensorD 0.7.3, COZY_TEST_TENSORD, as a transient `tensord-*` unit with the laptop's
//! environment verbatim) is taken over on the same root, store, ports, worker id and keys; disable
//! keeps every file; enable again serves the same machine warm. Runs where a user manager answers
//! (the job enables linger); skips elsewhere.
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

/// Ends what this test started under the manager, however it ended.
struct Cleanup<'a>(&'a Home);

impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        let _ = cozy(self.0, &["machine", "disable", "--json"]);
        let _ = cozy(self.0, &["down", "--json"]);
        for unit in [LEGACY, "cozy-machine", "cozy-calcifer"] {
            let _ = systemctl(&["disable", "--now", &format!("{unit}.service")]);
        }
    }
}

fn json(out: &Out) -> Value {
    assert_eq!(out.code, 0, "{out:?}");
    serde_json::from_str(&out.stdout).unwrap_or_else(|e| panic!("{e}: {out:?}"))
}

/// A followed `--json` run's end, which must complete, and whether it installed or synced.
fn ran(home: &Home, work: &Path, text: &str) -> bool {
    let out = common::out(
        managed(home)
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
    let Some(released) = std::env::var_os("COZY_TEST_TENSORD") else {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    };
    if !systemctl(&["show-environment"]).status.success() {
        eprintln!("skipped: no systemd user manager in this session");
        return;
    }
    let _cleanup = Cleanup(&home);

    // The laptop's layout: TensorD under the machine root, the store in ~/.tensorfs, and the
    // controller's owner key, whose public half the unit's environment admits.
    let root = home.root().join("machine/root");
    let program = root.join("usr/local/bin/tensord");
    std::fs::create_dir_all(program.parent().unwrap()).unwrap();
    std::fs::copy(&released, &program).unwrap();
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
    let started = start.arg(&program).output().unwrap();
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
    assert_eq!(property("cozy-machine.service", "ActiveState"), "active");
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
    ran(&home, &work, "first");
    assert!(!ran(&home, &work, "again"), "a second run installed again");

    // Enabling an enabled machine changes nothing.
    let again = json(&cozy(&home, &["machine", "enable", "--json"]));
    assert_eq!(again["changed"].as_bool(), Some(false), "{again}");

    // Disable stops it and keeps every file; enable serves the same machine warm.
    let disabled = json(&cozy(&home, &["machine", "disable", "--json"]));
    assert_eq!(disabled["changed"].as_bool(), Some(true), "{disabled}");
    assert_ne!(property("cozy-machine.service", "ActiveState"), "active");
    assert!(ready.exists() && store.read_dir().unwrap().next().is_some());
    let back = json(&cozy(&home, &["machine", "enable", "--json"]));
    assert_eq!(back["changed"].as_bool(), Some(true), "{back}");
    assert!(
        !ran(&home, &work, "back"),
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
