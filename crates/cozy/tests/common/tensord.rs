//! A real TensorD (`COZY_TEST_TENSORD`: this workspace's, `cargo build -p tensord`) on a temp root, for
//! tests that add it as another computer's machine. It runs as a persistent machine from a grant in
//! its environment, as a computer's machine does, so its Status carries its store, disk, models and
//! environments. Its TLS identity outlives a restart.
use std::{
    os::unix::{fs::PermissionsExt as _, process::CommandExt as _},
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
    /// The port it listens on across restarts, when a test names one (a shaped link's).
    port: Option<u16>,
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
        Self::start_on(None, keys, env)
    }

    /// `start_with`, listening on `port` (a new free one each start when `None`).
    pub fn start_on(port: Option<u16>, keys: &[String], env: &[(&str, &str)]) -> Self {
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
            port,
        };
        machine.restart(keys);
        machine
    }

    /// Its machine root (`COZY_MACHINE_ROOT`).
    pub fn root(&self) -> std::path::PathBuf {
        self.root.path().join("machine")
    }

    /// Restarts it admitting `keys`: same identity and leaf, a new port.
    pub fn restart(&mut self, keys: &[String]) {
        self.stop();
        let root = self.root.path().join("machine");
        let key_file = self.root.path().join("receipt-key");
        std::fs::write(&key_file, URL_SAFE_NO_PAD.encode([7u8; 32])).unwrap();
        std::fs::set_permissions(&key_file, std::fs::Permissions::from_mode(0o600)).unwrap();
        let port = self.port.unwrap_or_else(|| {
            std::net::TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port()
        });
        let binary = std::env::var_os("COZY_TEST_TENSORD")
            .expect("COZY_TEST_TENSORD names the tensord binary these tests run against");
        let mut child = Command::new(binary)
            .args(["run", "--persistent"])
            .env_remove("COZY_TEST_TENSORD")
            // A home of its own: this computer's keys and config are not the test machine's.
            .env("HOME", self.root.path())
            .env("COZY_MACHINE_ROOT", &root)
            .env("COZY_LISTEN_HOST", "127.0.0.1")
            .env("COZY_WORKER_ID", "cli-test")
            .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
            .env("COZY_AUTHORIZED_KEYS", keys.join(","))
            .env("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE", &key_file)
            .env("CUDA_VISIBLE_DEVICES", "")
            // It trusts a private Hub only as a run's Hub access names its CA.
            .env_remove("SSL_CERT_FILE")
            .env_remove("SSL_CERT_DIR")
            .envs(self.env.iter().map(|(k, v)| (k, v)))
            // Its own process group: the machine's stable parent starts children, and stop ends them all.
            .process_group(0)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        self.address = format!("127.0.0.1:{port}");
        let leaf = root.join("run/cozy/bootstrap/tls.crt");
        wait_ready(&mut child, &leaf, &self.address);
        self.child = Some(child);
        let body: String = std::fs::read_to_string(&leaf)
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with("-----"))
            .collect();
        let der = STANDARD.decode(body).unwrap();
        self.fingerprint = format!("sha256:{}", hex::encode(Sha256::digest(&der)));
    }

    /// Ends the exact processes this test started: its process group. Its service lets go of the
    /// root's owner lock only as it exits, after its parent's wait returns, so this waits for that
    /// lock too: a restart on the same root never finds it held.
    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{}", child.id())])
                .status();
            let _ = child.wait();
            let lock = self
                .root
                .path()
                .join("machine/var/lib/cozy/tensord/owner.lock");
            if let Ok(file) = std::fs::File::open(lock) {
                let _ = file.lock();
            }
        }
    }
}

impl Drop for Tensord {
    fn drop(&mut self) {
        self.stop();
    }
}

fn wait_ready(child: &mut Child, leaf: &Path, address: &str) {
    // A hang bound for a machine that never starts; the product waits without one.
    let until = Instant::now() + Duration::from_secs(120);
    while !(leaf.exists() && std::net::TcpStream::connect(address).is_ok()) {
        if let Ok(Some(status)) = child.try_wait() {
            panic!("tensord exited before it served: {status}");
        }
        assert!(
            Instant::now() < until,
            "tensord did not listen on {address}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Whether COZY_TEST_TENSORD is this workspace's TensorD rather than a released cozy-runtime
/// wheel's: Tensorhub's job says so when it builds it (`cozy_tensord=workspace` sets COZY_TENSORD).
pub fn workspace() -> bool {
    std::env::var("COZY_TENSORD").as_deref() == Ok("workspace")
}

/// TensorD 0.7.x, released and this workspace's alike, keeps no log of its own for `cozy
/// machine logs`; the v0.10.0 cohort's does. The cohort deletes this.
pub fn serves_own_log() -> bool {
    false
}

/// Released TensorD 0.7.3 drops the executor's adjustment rows; this workspace's keeps them on the
/// outcome (cr-197 U2, #1383). The cohort deletes this.
pub fn reports_adjustments() -> bool {
    workspace()
}

/// A TensorD added to `home`'s Calcifer as `name`, the way a person adds one: `cozy machine add`
/// answers the key this controller signs with, the machine admits it, and the add is made again.
pub fn added(home: &super::Home, name: &str) -> Tensord {
    added_with(home, name, &[])
}

/// `added`, with extra environment for the TensorD process.
pub fn added_with(home: &super::Home, name: &str, env: &[(&str, &str)]) -> Tensord {
    added_on(home, name, None, env)
}

/// `added_with`, listening on `port`.
pub fn added_on(
    home: &super::Home,
    name: &str,
    port: Option<u16>,
    env: &[(&str, &str)],
) -> Tensord {
    let mut machine = Tensord::start_on(port, &[stranger()], env);
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
