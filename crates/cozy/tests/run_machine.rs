//! `cozy run` against a real machine: TensorD from the cozy-runtime wheel (`COZY_TEST_TENSORD`),
//! added to the test home's Calcifer the way a person adds one.
#![cfg(unix)]
mod common;

use std::{
    io::BufRead,
    process::{Child, Stdio},
};

use common::{Home, silent::SilentHub};
use serde_json::Value;

fn follow(home: &Home, args: &[&str]) -> (Child, std::io::BufReader<std::process::ChildStdout>) {
    let mut child = home
        .command()
        .args(args)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = std::io::BufReader::new(child.stdout.take().unwrap());
    (child, stdout)
}

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

/// A run its machine accepted is followed across a Calcifer stop, and a cancel ends it with the
/// machine's own outcome.
#[test]
fn a_run_on_a_real_machine_survives_calcifer_stopping_and_cancels_on_the_machine() {
    let Some(home) = Home::new() else { return };
    let hub = SilentHub::for_home(&home);
    let machine = common::tensord::added(&home, "m1");

    let (mut follower, mut stdout) = follow(
        &home,
        &[
            "run",
            "fidika/sdxl/generate@2.10.0",
            "a cat",
            "--on",
            "m1",
            "--json",
        ],
    );
    let accepted = until(&mut stdout, |e| e["type"] == "run.in_progress");
    let run = &accepted["run"];
    assert_eq!(run["machine"], "m1", "{accepted}");
    assert_eq!(run["hub"], hub.origin.as_str());
    let number = run["number"].as_u64().unwrap().to_string();

    // A player link to the run's output, straight from its machine: the machine's own listener,
    // its pinned certificate, and a capability for that output alone, all in the fragment.
    let play = home.json(["run", "play", &number, "--json"]);
    let link = play["link"].as_str().unwrap();
    let (page, fragment) = link.split_once('#').unwrap();
    assert_eq!(page, "https://cozy-creator.github.io/cozy/play/");
    let pairs: Vec<(String, String)> = url::form_urlencoded::parse(fragment.as_bytes())
        .into_owned()
        .collect();
    let get = |k: &str| {
        pairs
            .iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    };
    assert_eq!(
        (get("v"), get("r"), get("o")),
        (
            "1".into(),
            run["id"].as_str().unwrap().to_owned(),
            "video".into()
        ),
        "{link}"
    );
    assert_eq!(
        format!("sha256:{}", get("f")),
        machine.fingerprint,
        "the media certificate is the machine's pinned one"
    );
    assert!(!get("c").is_empty(), "{link}");
    std::net::TcpStream::connect(get("a"))
        .expect("the machine's WebRTC listener takes a connection");

    let down = home.cozy(["down"]);
    assert_eq!(down.code, 0, "{down:?}");
    home.json(["run", "cancel", &number, "--json"]);
    let ended = until(&mut stdout, |e| e["type"] == "run.canceled");
    assert_eq!(ended["run"]["machine"], "m1");
    assert_eq!(follower.wait().unwrap().code(), Some(1));

    // An ended run that never wrote the output has nothing to play.
    let ended = home.cozy(["run", "play", &number, "--output", "film", "--json"]);
    assert_eq!(
        (ended.code, ended.refusal()["code"].as_str()),
        (1, Some("output_not_found")),
        "{ended:?}"
    );

    let shown = home.json(["run", "show", &number, "--json"]);
    assert_eq!(
        (shown["status"].as_str(), shown["machine"].as_str()),
        (Some("canceled"), Some("m1"))
    );
}
