//! SSH is optional on a development rental: granted a developer key on a machine with no sshd,
//! the machine serves anyway and says so once, naming sshd and `cozy doctor`.
//! A rental is a Linux pod.
#![cfg(target_os = "linux")]
mod common;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::SigningKey;
use std::{
    io::{BufRead, BufReader},
    net::TcpListener,
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tensord::api::v1::client::{Auth, Machine};

#[test]
fn a_dev_rental_without_sshd_serves_and_warns_once() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/machine-ssh").join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("home")).unwrap();
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let owner = SigningKey::from_bytes(&[3; 32]).verifying_key();
    let mut child = Command::new(env!("CARGO_BIN_EXE_tensord"))
        .env_clear()
        .arg("--rental")
        // No sbin: this machine has no sshd.
        .env("PATH", "/usr/bin:/bin")
        .env("COZY_MACHINE_ROOT", &root)
        .env("HOME", root.join("home"))
        .env("COZY_WORKER_ID", "no-sshd")
        .env("COZY_WORKER_AUTH_TOKEN", URL_SAFE_NO_PAD.encode([1; 32]))
        .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
        .env("COZY_LISTEN_HOST", "127.0.0.1")
        .env("COZY_AUTHORIZED_KEYS", URL_SAFE_NO_PAD.encode(owner.as_bytes()))
        .env("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL", URL_SAFE_NO_PAD.encode([7; 32]))
        .env("COZY_SSH_PUBLIC_KEY", "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIA developer")
        .env("TENSORHUB_ORIGIN", "https://hub.invalid")
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let stderr = child.stderr.take().unwrap();
    let kept = lines.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            kept.lock().unwrap().push(line);
        }
    });
    let serves = || {
        let pin = common::pin(&root)?;
        std::thread::spawn(move || {
            tokio::runtime::Runtime::new().ok()?.block_on(async {
                let machine = Machine::new(&format!("127.0.0.1:{port}"), pin, Auth::None).ok()?;
                machine.status().await.ok()
            })
        })
        .join()
        .ok()?
    };
    let start = Instant::now();
    while serves().is_none() {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("the machine exited ({status}): {:#?}", lines.lock().unwrap());
        }
        assert!(start.elapsed() < Duration::from_secs(120), "the machine never served: {:#?}", lines.lock().unwrap());
        std::thread::sleep(Duration::from_millis(200));
    }
    let _ = child.kill();
    let _ = child.wait();
    let lines = lines.lock().unwrap();
    let warned: Vec<_> = lines.iter().filter(|line| line.contains("sshd")).collect();
    assert_eq!(warned.len(), 1, "{lines:#?}");
    assert!(
        warned[0].contains("WARN") && warned[0].contains("sshd not found: no SSH into this dev machine; see `cozy doctor`"),
        "{warned:?}"
    );
    drop(lines);
    std::fs::remove_dir_all(&root).unwrap();
}
