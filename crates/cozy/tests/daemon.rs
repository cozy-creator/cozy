//! Calcifer's lifecycle through the real binary: started on demand, one per home, serving until
//! `down` (which stops no work), replaced when killed, and a refusal to start said in its own words.
mod common;

use std::{
    io::{BufRead as _, BufReader, Write as _},
    process::Stdio,
    sync::mpsc,
    time::Duration,
};

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

/// Without systemctl, `cozy up` starts Calcifer detached and says once that no login will start
/// it; an `up` that finds it running says nothing of it.
#[cfg(target_os = "linux")]
#[test]
fn up_without_systemctl_serves_and_warns_once() {
    let Some(home) = Home::new() else { return };
    let empty = home.path().join("empty-path");
    std::fs::create_dir_all(&empty).unwrap();
    let up = || {
        let mut command = home.command();
        common::out(
            command
                .env("PATH", &empty)
                .args(["up", "--json"])
                .output()
                .unwrap(),
        )
    };
    let warned = |out: &common::Out| out.stderr.matches("systemctl not found").count();
    let first = up();
    assert_eq!((first.code, warned(&first)), (0, 1), "{first:?}");
    let served: Value = serde_json::from_str(&first.stdout).unwrap();
    assert_eq!(served["calcifer"]["running"], true, "{served}");
    let again = up();
    assert_eq!((again.code, warned(&again)), (0, 0), "{again:?}");
    assert_eq!(home.json(["down", "--json"])["changed"], true);
}

/// `cozy logs calcifer --follow` prints each line once as it is written, across two rotations to
/// `calcifer.log.1`: woken by a watch on the log's directory, on every OS.
#[test]
fn calcifer_s_log_is_followed_across_rotations_without_a_line_missed_or_repeated() {
    let Some(home) = Home::new() else { return };
    let dir = home.root().join("calcifer");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("calcifer.log");
    let open = || {
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true).open(&path).unwrap()
    };
    // Written as Calcifer writes it: line by line through a handle it keeps open, which a
    // rotation leaves on the renamed file.
    let mut file = open();
    let mut write = |n: usize| {
        if n > 0 && n.is_multiple_of(400) {
            std::fs::rename(&path, dir.join("calcifer.log.1")).unwrap();
            file = open();
        }
        file.write_all(format!("line {n}\n").as_bytes()).unwrap();
    };
    write(0);
    let mut follower = home
        .command()
        .args(["logs", "calcifer", "--follow"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let (sent, read) = mpsc::channel();
    let out = BufReader::new(follower.stdout.take().unwrap());
    std::thread::spawn(move || {
        for line in out.lines() {
            if sent.send(line.unwrap()).is_err() {
                break;
            }
        }
    });
    // A hang bound for a follower that stopped; the product waits without one.
    let next = || read.recv_timeout(Duration::from_secs(60)).ok();
    assert_eq!(
        next().as_deref(),
        Some("line 0"),
        "the follower printed the log"
    );
    (1..1000).for_each(write);
    let mut got = vec![];
    while got.len() < 999
        && let Some(line) = next()
    {
        got.push(line);
    }
    follower.kill().unwrap();
    follower.wait().unwrap();
    let expected: Vec<String> = (1..1000).map(|n| format!("line {n}")).collect();
    assert_eq!(got, expected);
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
