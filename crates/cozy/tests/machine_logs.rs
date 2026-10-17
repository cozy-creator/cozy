//! `cozy machine logs` through the real binary, a real Calcifer and a real TensorD
//! (COZY_TEST_TENSORD) added by address. A model download writes the machine's TensorFS
//! transport log; reading and following it needs no Hub. Elsewhere it skips.
#![cfg(unix)]
mod common;

use std::{io::BufRead as _, process::Stdio, sync::mpsc, time::Duration};

use common::Home;
use serde_json::Value;

const TINY: &str =
    "hf://hf-internal-testing/tiny-random-gpt2@71034c5d8bde858ff824298bdedc65515b97d2b9";

fn lines(out: &common::Out) -> Vec<Value> {
    assert_eq!(out.code, 0, "{out:?}");
    out.stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}")))
        .collect()
}

#[test]
fn a_machines_logs_read_and_follow_what_it_did() {
    let Some(home) = Home::new() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    let machine = common::tensord::added(&home, "m1");

    let unknown = home.cozy(["machine", "logs", "nosuch", "--json"]);
    assert_eq!(
        (unknown.code, unknown.refusal()["code"].as_str()),
        (1, Some("machine_not_found")),
        "{unknown:?}"
    );
    // TensorD keeps no log of its own yet: a refusal that names the machine and what to do.
    let own = home.cozy(["machine", "logs", "m1", "--json"]);
    let refusal = own.refusal();
    assert_eq!(own.code, 1, "{own:?}");
    assert_eq!(refusal["code"], "machine_log_unsupported", "{own:?}");
    assert!(
        refusal["message"].as_str().unwrap().contains("m1"),
        "{own:?}"
    );
    assert_eq!(refusal["next"][1], "cozy machine update m1", "{own:?}");

    // A download writes the transport log; its lines read back whole, newest last.
    let out = home.cozy([
        "--json",
        "model",
        "download",
        TINY,
        "local/tiny",
        "--on",
        "m1",
    ]);
    assert_eq!(out.code, 0, "{out:?}");
    let read = lines(&home.cozy(["machine", "logs", "m1", "tensorfs", "-n", "0", "--json"]));
    assert!(
        read.iter()
            .any(|l| l["text"].as_str().unwrap().contains(" walk ")),
        "{read:?}"
    );
    assert!(
        read.iter()
            .all(|l| l["machine"] == "m1" && l["log"] == "tensorfs")
    );
    let newest = lines(&home.cozy(["machine", "logs", "m1", "tensorfs", "-n", "1", "--json"]));
    assert_eq!(newest, read[read.len() - 1..], "-n 1 is the newest line");
    let recent = home.cozy([
        "machine", "logs", "m1", "tensorfs", "-n", "0", "--since", "1h", "--json",
    ]);
    assert_eq!(lines(&recent), read, "every line is from the last hour");
    let bad = home.cozy(["machine", "logs", "m1", "--since", "soon", "--json"]);
    assert_eq!(bad.code, 2, "{bad:?}");

    // Following prints what it kept, then each line as the machine writes it.
    let mut follower = home
        .command()
        .args([
            "machine", "logs", "m1", "tensorfs", "-n", "0", "--follow", "--json",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = follower.stdout.take().unwrap();
    let (sent, followed) = mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout)
            .lines()
            .map_while(Result::ok)
        {
            if sent.send(line).is_err() {
                return;
            }
        }
    });
    let next = |what: &str| {
        let line = followed
            .recv_timeout(Duration::from_secs(120))
            .unwrap_or_else(|e| panic!("no {what} line from the follower: {e}"));
        serde_json::from_str::<Value>(&line).unwrap()
    };
    for kept in &read {
        assert_eq!(&next("kept"), kept);
    }
    let out = home.cozy([
        "--json",
        "model",
        "download",
        "hf://hf-internal-testing/tiny-random-bert@f171d7baecaf37b5da5a3616d8833b9969753535",
        "local/bert",
        "--on",
        "m1",
    ]);
    assert_eq!(out.code, 0, "{out:?}");
    let fresh = next("new");
    assert!(
        fresh["text"].as_str().unwrap().contains(" walk "),
        "{fresh}"
    );
    follower.kill().unwrap();
    let _ = follower.wait();

    // A machine that stopped answering is refused by name.
    drop(machine);
    let gone = home.cozy(["machine", "logs", "m1", "tensorfs", "--json"]);
    let refusal = gone.refusal();
    assert_eq!(gone.code, 1, "{gone:?}");
    assert!(
        refusal["message"].as_str().unwrap().contains("m1"),
        "{gone:?}"
    );
}
