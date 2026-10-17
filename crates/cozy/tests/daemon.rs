//! Calcifer's lifecycle through the real binary: started on demand, one per home, serving until
//! `down` (which stops no work), replaced when killed, and a refusal to start said in its own words.
mod common;

#[cfg(unix)]
use std::io::Write as _;

use common::Home;
use serde_json::Value;

fn calcifer(home: &Home, verb: &str) -> Value {
    home.json([verb, "--json"])["calcifer"].clone()
}

fn kill(pid: u64) {
    let pid = pid.to_string();
    let status = if cfg!(windows) {
        std::process::Command::new("taskkill")
            .args(["/F", "/PID", &pid])
            .status()
    } else {
        std::process::Command::new("kill")
            .args(["-9", &pid])
            .status()
    };
    assert!(status.unwrap().success());
}

#[test]
fn calcifer_starts_on_demand_and_down_stops_it() {
    let Some(home) = Home::new() else { return };
    let log = home.cozy(["logs", "calcifer", "--json"]);
    assert_eq!(
        (log.code, log.refusal()["code"].as_str()),
        (1, Some("log_not_found")),
        "{log:?}"
    );
    assert_eq!(calcifer(&home, "status")["running"], false);
    assert!(
        !home.root().join("calcifer").exists(),
        "status started Calcifer"
    );
    assert_eq!(home.json(["down", "--json"])["changed"], false);

    let up = calcifer(&home, "up");
    assert_eq!(up["running"], true, "{up}");
    let holder = home.root().join("calcifer/calcifer.lock.holder");
    // Owing nothing, with no client connected, it keeps serving.
    std::thread::sleep(std::time::Duration::from_secs(1));
    assert_eq!(
        calcifer(&home, "up")["pid"],
        up["pid"],
        "a second up started another"
    );
    let status = home.json(["status", "--json"]);
    assert_eq!(status["calcifer"]["pid"], up["pid"], "{status}");
    assert!(status["runs"].as_array().unwrap().is_empty(), "{status}");
    assert_eq!(home.cozy(["logs", "calcifer"]).code, 0);

    let down = home.json(["down", "--json"]);
    assert_eq!(down["changed"], true, "{down}");
    assert!(!holder.exists(), "down returned before Calcifer let go");
    assert_eq!(calcifer(&home, "status")["running"], false);
    assert_ne!(calcifer(&home, "up")["pid"], up["pid"]);
}

/// A Calcifer killed before it could erase its record is not mistaken for a live one.
#[test]
fn a_killed_calcifer_is_replaced() {
    let Some(home) = Home::new() else { return };
    let first = calcifer(&home, "up");
    kill(first["pid"].as_u64().unwrap());
    assert!(home.root().join("calcifer/calcifer.lock.holder").exists());
    assert_eq!(calcifer(&home, "status")["running"], false);
    let next = calcifer(&home, "up");
    assert_eq!(next["running"], true);
    assert_ne!(next["pid"], first["pid"]);
}

/// A Calcifer that cannot open its store refuses in its own words, and the CLI says them.
#[test]
fn a_calcifer_that_cannot_start_says_why() {
    let Some(home) = Home::new() else { return };
    std::fs::create_dir_all(home.root().join("calcifer/state.db")).unwrap();
    let out = home.cozy(["up", "--json"]);
    assert_eq!(
        (out.code, out.refusal()["code"].as_str()),
        (1, Some("calcifer_store")),
        "{out:?}"
    );
    assert_eq!(calcifer(&home, "status")["running"], false);
}

/// Commands sent while `down` is in flight are answered by a live Calcifer: the stopping one is
/// waited out, a connection it closes under a request is sent again, and a new one starts. Never
/// `calcifer_stopping` or `calcifer_unreachable`. A keepalive for no machine is a mutation (it
/// meets the stopping gate) whose live answer is `not_found`.
#[test]
fn commands_racing_down_reach_a_live_calcifer() {
    let Some(home) = Home::new() else { return };
    let quiet = || std::process::Stdio::null();
    for round in 0..8 {
        calcifer(&home, "up");
        let mut down = home
            .command()
            .args(["down"])
            .stdout(quiet())
            .spawn()
            .unwrap();
        let racers: Vec<_> = (0..3)
            .map(|i| {
                std::thread::sleep(std::time::Duration::from_millis(5 * i));
                home.command()
                    .args(["machine", "keepalive", "nosuch", "--json"])
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect();
        assert!(down.wait().unwrap().success());
        for racer in racers {
            let out = common::out(racer.wait_with_output().unwrap());
            assert_eq!(
                (out.code, out.refusal()["kind"].as_str()),
                (1, Some("not_found")),
                "round {round}: {out:?}"
            );
        }
    }
}

/// `status` shows the machines Calcifer holds, from its own picture, beside the unsettled runs.
#[cfg(unix)]
#[test]
fn status_shows_the_machines() {
    let Some(home) = Home::new() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        let _ = writeln!(
            std::io::stderr(),
            "skipped: no TensorD (COZY_TEST_TENSORD is unset)"
        );
        return;
    }
    let _machine = common::tensord::added(&home, "box");
    let status = home.json(["status", "--json"]);
    let machines = status["machines"].as_array().unwrap();
    assert!(
        machines
            .iter()
            .any(|m| m["name"] == "box" && m["kind"] == "added" && m["state"] == "ready"),
        "{status}"
    );
    let human = home.cozy(["status"]);
    assert!(
        human.code == 0 && human.stdout.contains("MACHINE") && human.stdout.contains("box"),
        "{human:?}"
    );
}
