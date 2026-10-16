//! A real TensorD (`COZY_TEST_TENSORD`, the binary a cozy-runtime wheel carries) on a temp root, for
//! tests that add it as another computer's machine. Its TLS identity outlives a restart.
use std::{
    os::unix::fs::PermissionsExt as _,
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use sha2::{Digest as _, Sha256};

pub struct Tensord {
    child: Option<Child>,
    root: tempfile::TempDir,
    /// Extra environment for the process, such as where to find a test Hub's CA.
    env: Vec<(String, String)>,
    /// `host:port` it listens on.
    pub address: String,
    /// `sha256:<hex>` of its TLS leaf.
    pub fingerprint: String,
}

/// A controller key no Calcifer holds, base64url as TensorD's keys file spells it.
pub fn stranger() -> String {
    let key = ed25519_dalek::SigningKey::from_bytes(&[9; 32]).verifying_key();
    URL_SAFE_NO_PAD.encode(key.as_bytes())
}

impl Tensord {
    /// Starts a machine that admits the controllers holding `keys` (base64url Ed25519 public keys).
    pub fn start(keys: &[String]) -> Self {
        Self::start_with(keys, &[])
    }

    /// `start`, with extra environment for the process.
    pub fn start_with(keys: &[String], env: &[(&str, &str)]) -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut machine = Self {
            child: None,
            root,
            env: env
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            address: String::new(),
            fingerprint: String::new(),
        };
        machine.restart(keys);
        machine
    }

    /// Restarts it admitting `keys`: same identity and leaf, a new port.
    pub fn restart(&mut self, keys: &[String]) {
        self.stop();
        let config = self.root.path().join("config");
        std::fs::create_dir_all(&config).unwrap();
        let write = |name: &str, value: serde_json::Value| {
            let path = config.join(name);
            std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        };
        write("keys.json", serde_json::json!({ "keys": keys }));
        write(
            "readiness.json",
            serde_json::json!({ "key_b64url": URL_SAFE_NO_PAD.encode([7u8; 32]) }),
        );
        write(
            "machine.json",
            serde_json::json!({"worker_id": "cli-test", "identity_directory": "identity",
                "authorized_keys_file": "keys.json", "readiness_hmac_key_file": "readiness.json"}),
        );
        let state = self.root.path().join("state");
        let _ = std::fs::remove_file(state.join("api-ready.json"));
        let binary = std::env::var_os("COZY_TEST_TENSORD")
            .expect("COZY_TEST_TENSORD names the tensord binary these tests run against");
        let mut child = Command::new(binary)
            .envs(self.env.iter().map(|(k, v)| (k, v)))
            .args(["serve", "--state"])
            .arg(&state)
            .arg("--machine-config")
            .arg(config.join("machine.json"))
            .args(["--listen", "127.0.0.1:0", "--host-bytes", "0"])
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let ready = wait_ready(&mut child, &state);
        self.child = Some(child);
        self.address = ready["address"].as_str().unwrap().into();
        let body: String = ready["cert_pem"]
            .as_str()
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with("-----"))
            .collect();
        let der = STANDARD.decode(body).unwrap();
        self.fingerprint = format!("sha256:{}", hex::encode(Sha256::digest(&der)));
    }

    /// Ends the exact process this test started.
    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Tensord {
    fn drop(&mut self) {
        self.stop();
    }
}

fn wait_ready(child: &mut Child, state: &Path) -> serde_json::Value {
    // A hang bound for a machine that never starts; the product waits without one.
    let until = Instant::now() + Duration::from_secs(120);
    loop {
        if let Ok(bytes) = std::fs::read(state.join("api-ready.json")) {
            return serde_json::from_slice(&bytes).unwrap();
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("tensord exited before it served: {status}");
        }
        assert!(
            Instant::now() < until,
            "tensord did not publish api-ready.json"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A TensorD added to `home`'s Calcifer as `name`, the way a person adds one: `cozy machine add`
/// answers the key this controller signs with, the machine admits it, and the add is made again.
pub fn added(home: &super::Home, name: &str) -> Tensord {
    added_with(home, name, &[])
}

/// `added`, with extra environment for the TensorD process.
pub fn added_with(home: &super::Home, name: &str, env: &[(&str, &str)]) -> Tensord {
    let mut machine = Tensord::start_with(&[stranger()], env);
    let add = |machine: &Tensord| {
        home.cozy([
            "machine",
            "add",
            name,
            machine.address.as_str(),
            "--fingerprint",
            machine.fingerprint.as_str(),
            "--json",
        ])
    };
    machine.restart(&[key_to_authorize(&add(&machine))]);
    let out = add(&machine);
    assert_eq!(out.code, 0, "{out:?}");
    machine
}

/// The controller key a refused `cozy machine add --json` asks that computer to authorize.
pub fn key_to_authorize(refused: &super::Out) -> String {
    let remedy = refused.refusal()["remedy"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    remedy
        .split("cozy machine authorize ")
        .nth(1)
        .and_then(|rest| rest.split(|c: char| c == '`' || c.is_whitespace()).next())
        .unwrap_or_else(|| panic!("the add names no key to authorize: {refused:?}"))
        .to_string()
}
