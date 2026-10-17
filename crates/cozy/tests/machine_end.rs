//! `cozy machine end` on a real machine (COZY_TEST_TENSORD) whose link is slow: it waits while a
//! completed run's output is still landing, saying so, and ends the machine once it landed.
#![cfg(unix)]
mod common;

use std::{
    io::{Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use common::{Home, tensord};
use serde_json::Value;

const OUTPUT: usize = 24 << 20;

const SCRIPT: &str = r#"# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime"]
# [tool.cozy]
# accelerator = false
# ///
from cozy_runtime.author import FileAsset, Outputs

def main(*, out: Outputs) -> FileAsset:
    return out.save_bytes(bytes(range(256)) * (96 << 10), media_type="application/octet-stream")
"#;

/// A TCP relay to `upstream` passing about `rate` bytes a second over all its connections, both
/// ways: Calcifer reads an output over several connections side by side.
fn throttled(upstream: String, rate: f64) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let next = Arc::new(Mutex::new(Instant::now()));
    std::thread::spawn(move || {
        for client in listener.incoming().flatten() {
            let Ok(server) = TcpStream::connect(&upstream) else {
                continue;
            };
            let ways = [
                (client.try_clone().unwrap(), server.try_clone().unwrap()),
                (server, client),
            ];
            for (mut from, mut to) in ways {
                let next = next.clone();
                std::thread::spawn(move || {
                    let mut buf = vec![0; 64 << 10];
                    while let Ok(n @ 1..) = from.read(&mut buf) {
                        // Each chunk takes its turn on one shared clock.
                        let turn = {
                            let mut next = next.lock().unwrap();
                            let turn = (*next).max(Instant::now());
                            *next = turn + Duration::from_secs_f64(n as f64 / rate);
                            turn
                        };
                        std::thread::sleep(turn.saturating_duration_since(Instant::now()));
                        if to.write_all(&buf[..n]).is_err() {
                            break;
                        }
                    }
                    let _ = to.shutdown(Shutdown::Write);
                });
            }
        }
    });
    address
}

#[test]
fn ending_a_machine_waits_for_its_runs_outputs_to_land() {
    let Some(home) = Home::new() else { return };
    if std::env::var_os("COZY_TEST_TENSORD").is_none() {
        eprintln!("skipped: no TensorD (COZY_TEST_TENSORD is unset)");
        return;
    }
    // Added as a person adds one, then reached only through a 4 MB/s link.
    let mut machine = tensord::Tensord::start(&[tensord::stranger()]);
    // Its TLS leaf outlives a restart.
    let fingerprint = machine.fingerprint.clone();
    let add = |address: &str| {
        home.cozy([
            "machine",
            "add",
            "slow",
            address,
            "--fingerprint",
            &fingerprint,
            "--json",
        ])
    };
    let key = tensord::key_to_authorize(&add(&machine.address));
    machine.restart(&[key]);
    let relay = throttled(machine.address.clone(), f64::from(4u32 << 20));
    let added = add(&relay);
    assert_eq!(added.code, 0, "{added:?}");

    let work = home.path().join("work");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("big.py"), SCRIPT).unwrap();
    let sent = home
        .command()
        .current_dir(&work)
        .args([
            "--json", "run", "big.py", "--on", "slow", "--out", "saved", "--detach",
        ])
        .output()
        .unwrap();
    let sent: Value =
        serde_json::from_slice(&sent.stdout).unwrap_or_else(|e| panic!("{e}: {sent:?}"));
    let id = sent["id"]
        .as_str()
        .unwrap_or_else(|| panic!("{sent}"))
        .to_owned();
    // A follower's terminal event comes only after the outputs landed: read the record instead.
    // A hang bound for a run that never completes; the product waits without one.
    let until = Instant::now() + Duration::from_secs(600);
    loop {
        let run = home.json(["run", "show", &id, "--json"]);
        match run["status"].as_str() {
            Some("completed") => break,
            Some("queued" | "in_progress") => {}
            _ => panic!("the run did not complete: {run}"),
        }
        assert!(Instant::now() < until, "the run never completed: {run}");
        std::thread::sleep(Duration::from_millis(100));
    }

    // Completed, its output still crossing the link: the end waits for it, then ends.
    let ended = home.cozy(["machine", "end", "slow"]);
    assert_eq!(ended.code, 0, "{ended:?}");
    assert!(ended.stderr.contains("Waiting to end slow"), "{ended:?}");
    let saved = std::fs::read_dir(work.join("saved")).unwrap();
    let landed: Vec<_> = saved.flatten().map(|e| e.path()).collect();
    assert_eq!(landed.len(), 1, "{landed:?}");
    assert_eq!(std::fs::metadata(&landed[0]).unwrap().len(), OUTPUT as u64);
    let listed = home.json(["machine", "list", "--json"]);
    let names: Vec<_> = listed["machines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].clone())
        .collect();
    assert!(!names.contains(&"slow".into()), "{listed}");
}
