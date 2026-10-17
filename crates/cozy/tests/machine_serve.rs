//! `cozy machine serve --persistent`: this computer's machine from the same binary, in the home the
//! CLI and Calcifer use. It answers Status over `cozy.machine.v1` to a Cozy-Cap its home admits, and
//! Calcifer reaches it through `cozy machine add` with nothing authorized by hand: Calcifer's owner
//! key is in the home, and the machine reads the home's keys again when a signer is new to it.
#![cfg(target_os = "linux")]
mod common;

use std::{
    os::unix::process::CommandExt as _,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use calcifer::{link::Link, store::Endpoint};
use common::Home;
use sha2::{Digest as _, Sha256};

/// The machine's process group, ended whole when the test ends.
struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = Command::new("kill").args(["-KILL", "--", &format!("-{}", self.0.id())]).status();
        let _ = self.0.wait();
    }
}

#[test]
fn this_computers_machine_serves_status_and_calcifer_adds_it() {
    let Some(home) = Home::new() else { return };
    // A controller of the test's own, authorized as `cozy machine authorize` would.
    let key = ed25519_dalek::SigningKey::from_bytes(&[11; 32]);
    let keys = home.root().join("keys");
    std::fs::create_dir_all(&keys).unwrap();
    std::fs::write(keys.join("authorized"), URL_SAFE_NO_PAD.encode(key.verifying_key().as_bytes()) + "\n").unwrap();

    let mut machine = Machine(
        home.command()
            .args(["machine", "serve", "--persistent"])
            .env("CUDA_VISIBLE_DEVICES", "")
            .process_group(0)
            .stdout(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let ready_file = home.root().join("machine/var/lib/cozy/tensord/api-ready.json");
    let until = Instant::now() + Duration::from_secs(300);
    while !ready_file.exists() {
        if let Ok(Some(status)) = machine.0.try_wait() {
            panic!("cozy machine serve exited before it served: {status}");
        }
        assert!(Instant::now() < until, "no {}", ready_file.display());
        std::thread::sleep(Duration::from_millis(100));
    }
    let comm = std::fs::read_to_string(format!("/proc/{}/comm", machine.0.id())).unwrap();
    assert_eq!(comm.trim(), "tensord");
    let ready: serde_json::Value = serde_json::from_slice(&std::fs::read(&ready_file).unwrap()).unwrap();
    let address = ready["address"].as_str().unwrap().to_owned();
    let body: String = ready["cert_pem"].as_str().unwrap().lines().filter(|l| !l.starts_with("-----")).collect();
    let pin: [u8; 32] = Sha256::digest(STANDARD.decode(body).unwrap()).into();

    // Status over the machine API, under a Cozy-Cap the home admits.
    let endpoint = Endpoint { worker_id: "local".into(), address: address.clone(), pin };
    let frame = tokio::runtime::Runtime::new().unwrap().block_on(async {
        Link::new(&endpoint, key).unwrap().status().await.unwrap()
    });
    assert_eq!((frame.worker_id.as_str(), frame.phase.as_str()), ("local", "ready"), "{frame:?}");
    assert!(frame.gpus.is_empty(), "CUDA_VISIBLE_DEVICES hides every GPU: {:?}", frame.gpus);

    // Calcifer adds it by address and fingerprint; its owner key, minted now, is admitted at once.
    let fingerprint = format!("sha256:{}", hex::encode(pin));
    let out = home.cozy(["machine", "add", "box", &address, "--fingerprint", &fingerprint, "--json"]);
    assert_eq!(out.code, 0, "{out:?}");
    let listed = home.json(["machine", "list", "--json"]);
    let row = listed["machines"].as_array().unwrap().iter().find(|m| m["name"] == "box").cloned();
    let row = row.unwrap_or_else(|| panic!("box is not listed: {listed}"));
    assert_eq!((row["kind"].as_str(), row["state"].as_str()), (Some("added"), Some("ready")), "{row}");
}
