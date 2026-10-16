//! Calcifer's lifecycle through the real binary: started on demand, one per home, stopped by
//! `down` without stopping work, replaced when killed, and a refusal to start said in its own
//! words. Calcifer stops once it owes nothing and no client is connected or wants it, so a test
//! that needs it up holds `wanted.lock` as a client would.
mod common;

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

    // Nobody wants it after `up` returns: it stops by itself.
    let first = calcifer(&home, "up");
    assert_eq!(first["running"], true, "{first}");
    let holder = home.root().join("calcifer/calcifer.lock.holder");
    while holder.exists() {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let _wanted = home.want_calcifer();
    let up = calcifer(&home, "up");
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
    let _wanted = home.want_calcifer();
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
