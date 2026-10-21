//! The normal rental startup converts supported store1, and refuses unsupported store2
//! before changing any existing data. This uses the real binary, without a provider.
#![cfg(target_os = "linux")]
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::SigningKey;
use std::{
    io::Read,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
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

/// A store in TensorFS 0.12's format at `store`, with an object.
fn older(store: &Path) {
    std::fs::create_dir_all(store.join("blobs/ab/cd")).unwrap();
    std::fs::write(store.join("format"), br#"{"format":"tensorfs.store/2"}"#).unwrap();
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
        .env("COZY_WORKER_ID", "older-store")
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
fn a_rental_refuses_unsupported_store2_without_changing_its_contents() {
    let (root, port) = (root("rental-older-store"), port());
    let store = root.join("var/lib/tensorfs");
    older(&store);
    let mut machine = launch(&root, port, None, Stdio::piped());
    // Read through process exit: the supervisor must stop after refused startup,
    // not silently start on another store or leave a partial conversion serving.
    let mut refused = String::new();
    machine.0.stderr.take().unwrap().read_to_string(&mut refused).unwrap();
    assert!(refused.contains("not a store1 format record"), "the rental never refused unsupported store2: {refused}");
    assert!(!machine.0.wait().unwrap().success(), "an unsupported store was accepted");
    assert!(store.join("blobs/ab/cd/abcd").exists(), "the older store was touched");
    assert_eq!(std::fs::read(store.join("format")).unwrap(), br#"{"format":"tensorfs.store/2"}"#);
    assert_eq!(std::fs::read(store.join("blobs/ab/cd/abcd")).unwrap(), b"an old object");
    assert!(!store.join(".store1-migration").exists(), "unsupported data entered conversion");
}
