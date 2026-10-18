//! `cozy machine toolchain install` on a real machine without nvcc (this workspace's TensorD,
//! COZY_TEST_TENSORD): its Status says the toolchain is absent, the install is an update run that
//! puts CUDA's compiler wheels beside its software, and `cozy doctor` then shows it ready. Linux.
#![cfg(target_os = "linux")]
mod common;

use std::time::{Duration, Instant};

use common::{Home, tensord};
use serde_json::Value;

#[test]
fn a_machine_without_nvcc_gets_its_toolchain_on_request() {
    let Some(home) = Home::new() else { return };
    let _machine = tensord::added(&home, "m1");
    let toolchain =
        |home: &Home| home.json(["machine", "show", "m1", "--json"])["status"]["toolchain"].clone();
    let before = toolchain(&home);
    assert_eq!(before["state"], "absent", "{before}");
    assert!(
        before["missing"]
            .as_array()
            .is_some_and(|m| m.iter().any(|t| t == "nvcc")),
        "{before}"
    );
    let doctor = home.cozy(["doctor", "--on", "m1"]);
    assert!(
        doctor.stdout.contains("cozy machine toolchain install"),
        "{doctor:?}"
    );

    let out = home.cozy(["machine", "toolchain", "install", "--on", "m1", "--json"]);
    let last: Value = serde_json::from_str(out.stdout.lines().last().unwrap_or_default())
        .unwrap_or_else(|e| panic!("{e}: {out:?}"));
    assert_eq!(
        (
            out.code,
            last["type"].as_str(),
            last["run"]["kind"].as_str()
        ),
        (0, Some("run.completed"), Some("update")),
        "{out:?}"
    );
    // A hang bound for the Status that follows the install; the product waits without one.
    let until = Instant::now() + Duration::from_secs(60);
    let after = loop {
        let now = toolchain(&home);
        if now["state"] != "absent" || Instant::now() > until {
            break now;
        }
        std::thread::sleep(Duration::from_millis(500));
    };
    assert_eq!(after["state"], "ready", "{after}");
    assert!(
        after["nvcc"].as_str().is_some_and(|v| v.contains("13.0")),
        "{after}"
    );
    let doctor = home.cozy(["doctor", "--on", "m1"]);
    assert!(doctor.stdout.contains("ready: "), "{doctor:?}");
    let again = home.json(["machine", "toolchain", "install", "--on", "m1", "--json"]);
    assert_eq!(again["changed"].as_bool(), Some(false), "{again}");
}
