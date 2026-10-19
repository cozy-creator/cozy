//! A rental booting over a store from before TensorFS 0.12's format cut (an older image's machine
//! made it, or a live rental updated in place): its own store's old contents go and a fresh store
//! serves; the pod's models re-pull. A store an operator named elsewhere (`COZY_TENSORFS_ROOT`,
//! perhaps shared) is never touched: that machine keeps refusing it. The real binary.
#![cfg(target_os = "linux")]
mod common;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::SigningKey;
use std::{
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn root(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join(name).join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// A pre-cut store at `store`: its catalog and an object.
fn pre_cut(store: &Path) {
    std::fs::create_dir_all(store.join("blobs/ab/cd")).unwrap();
    std::fs::write(store.join("tensorfs.sqlite"), b"").unwrap();
    std::fs::write(store.join("blobs/ab/cd/abcd"), b"an old object").unwrap();
}

fn launch(root: &Path, port: u16, store: Option<&Path>, stderr: Stdio) -> Machine {
    let owner = SigningKey::from_bytes(&[3; 32]).verifying_key();
    let mut command = Command::new(env!("CARGO_BIN_EXE_tensord"));
    command
        .env_clear()
        .arg("--rental")
        .env("PATH", "/usr/bin:/bin")
        .env("COZY_MACHINE_ROOT", root)
        .env("HOME", root.join("home"))
        .env("COZY_WORKER_ID", "pre-cut")
        .env("COZY_WORKER_AUTH_TOKEN", URL_SAFE_NO_PAD.encode([1; 32]))
        .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
        .env("COZY_LISTEN_HOST", "127.0.0.1")
        .env("COZY_AUTHORIZED_KEYS", URL_SAFE_NO_PAD.encode(owner.as_bytes()))
        .env("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL", URL_SAFE_NO_PAD.encode([7; 32]))
        .env("TENSORHUB_ORIGIN", "https://hub.invalid")
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(Stdio::null())
        .stderr(stderr);
    if let Some(store) = store {
        command.env("COZY_TENSORFS_ROOT", store);
    }
    Machine(command.spawn().unwrap())
}

#[test]
fn a_rental_boots_over_a_pre_cut_store_with_a_fresh_one() {
    let (root, port) = (root("rental-pre-cut"), port());
    let store = root.join("var/lib/tensorfs");
    pre_cut(&store);
    let mut machine = launch(&root, port, None, Stdio::inherit());
    let start = Instant::now();
    while common::receipt(&root, port).is_none() {
        assert!(start.elapsed() < Duration::from_secs(120), "the rental never proved readiness");
        assert!(machine.0.try_wait().unwrap().is_none(), "the rental exited over a pre-cut store");
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(!store.join("tensorfs.sqlite").exists() && !store.join("blobs/ab/cd/abcd").exists(), "the old store's contents stayed");
    assert!(tensord::storage::open(&store).is_ok(), "no fresh store serves");
}

#[test]
fn a_store_an_operator_named_elsewhere_is_never_touched() {
    let (root, port) = (root("rental-named-store"), port());
    let store = root.join("shared/tensorfs");
    pre_cut(&store);
    let mut machine = launch(&root, port, Some(&store), Stdio::piped());
    // Its refusal, as it says it: the hang bound is the reader's end of stderr.
    let lines = std::io::BufReader::new(machine.0.stderr.take().unwrap());
    let refused = std::io::BufRead::lines(lines)
        .map_while(Result::ok)
        .any(|line| line.contains("this store predates TensorFS 0.12's format cut"));
    assert!(refused, "the machine never refused the named pre-cut store");
    assert!(store.join("tensorfs.sqlite").exists() && store.join("blobs/ab/cd/abcd").exists(), "a named store was touched");
}
