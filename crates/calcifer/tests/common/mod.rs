//! What the tests share: a real TensorD process (`COZY_TEST_TENSORD`: this workspace's,
//! `cargo build -p tensord`) on a temp root that admits the test's key, and run specs it accepts with no Runtime.
#![allow(dead_code)]
use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use calcifer::store::{Endpoint, MachineRow};
use cozy_platform::{
    durable,
    lock::{Held, Kind as LockKind, LockFile},
    paths::Home,
};
use cozy_proto::{
    calcifer::{Event, MachineKind, MachineState, WANTED},
    ids::{Digest, RunId},
    machine as v1,
};
use ed25519_dalek::{SigningKey, pkcs8::EncodePrivateKey as _};
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use tokio::sync::broadcast;

pub const WORKER: &str = "calcifer-test";

/// A client's claim that it is starting Calcifer on `home` (`wanted.lock`, shared): the CLI holds
/// it until Calcifer's first answer, so a Calcifer owing nothing waits for that client.
pub fn want(home: &Path) -> Held {
    let dir = Home::at(home).calcifer();
    durable::private_dir(&dir).unwrap();
    let wanted = LockFile::new(dir.join(WANTED)).try_lock(LockKind::Shared);
    wanted.unwrap().expect("a shared lock")
}

pub fn key() -> SigningKey {
    SigningKey::from_bytes(&[33; 32])
}

/// Writes the test key where a machine row names it.
pub fn key_file(dir: &Path) -> PathBuf {
    let path = dir.join("key.pem");
    let pem = key().to_pkcs8_pem(Default::default()).unwrap();
    std::fs::write(&path, pem.as_bytes()).unwrap();
    path
}

pub struct Tensord {
    child: Child,
    pub root: tempfile::TempDir,
    pub endpoint: Endpoint,
    /// More `serve` arguments, kept for a restart.
    extra: Vec<std::ffi::OsString>,
}

impl Tensord {
    pub fn start() -> Self {
        Self::start_args(&[])
    }

    /// As `start`, with more `serve` arguments.
    pub fn start_args(extra: &[&std::ffi::OsStr]) -> Self {
        let root = tempfile::tempdir().unwrap();
        let config = root.path().join("config");
        // TensorD refuses configuration others may read.
        durable::private_dir(root.path()).unwrap();
        std::fs::create_dir_all(&config).unwrap();
        let write = |name: &str, value: serde_json::Value| {
            let bytes = serde_json::to_vec(&value).unwrap();
            durable::replace(&config.join(name), &bytes, 0o600).unwrap();
        };
        let public = URL_SAFE_NO_PAD.encode(key().verifying_key().to_bytes());
        write("keys.json", serde_json::json!({ "keys": [public] }));
        write(
            "readiness.json",
            serde_json::json!({ "key_b64url": URL_SAFE_NO_PAD.encode([7u8; 32]) }),
        );
        write(
            "machine.json",
            serde_json::json!({"worker_id": WORKER, "identity_directory": "identity",
                "authorized_keys_file": "keys.json", "readiness_hmac_key_file": "readiness.json"}),
        );
        let extra: Vec<_> = extra.iter().map(|arg| arg.to_os_string()).collect();
        let (child, endpoint) = serve(root.path(), "127.0.0.1:0", &extra);
        Self {
            child,
            root,
            endpoint,
            extra,
        }
    }

    /// Stops the process and starts another on the same root and address: the same machine,
    /// restarted.
    pub fn restart(&mut self) {
        self.stop();
        let _ = std::fs::remove_file(self.root.path().join("state/api-ready.json"));
        let (child, endpoint) = serve(self.root.path(), &self.endpoint.address, &self.extra);
        assert_eq!(endpoint, self.endpoint, "the same machine");
        self.child = child;
    }

    /// Ends the exact process this test started.
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    pub fn row(&self, name: &str, key: &Path) -> MachineRow {
        machine(name, Some(self.endpoint.clone()), key)
    }
}

/// Serves TensorD on `root` at `listen`, and answers it once it published its endpoint.
fn serve(root: &Path, listen: &str, extra: &[std::ffi::OsString]) -> (Child, Endpoint) {
    let binary = std::env::var_os("COZY_TEST_TENSORD")
        .expect("COZY_TEST_TENSORD names the tensord binary these tests run against");
    let state = root.join("state");
    let mut child = Command::new(binary)
        .args(["serve", "--state"])
        .arg(&state)
        .arg("--machine-config")
        .arg(root.join("config").join("machine.json"))
        .args(["--listen", listen, "--host-bytes", "0"])
        .args(extra)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    // A hang bound for a machine that never starts; the product waits without one.
    let until = Instant::now() + Duration::from_secs(120);
    let ready: serde_json::Value = loop {
        if let Ok(bytes) = std::fs::read(state.join("api-ready.json")) {
            break serde_json::from_slice(&bytes).unwrap();
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("tensord exited before it served: {status}");
        }
        if Instant::now() >= until {
            let _ = child.kill();
            panic!("tensord did not publish api-ready.json");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let pem = ready["cert_pem"].as_str().unwrap();
    let body: String = pem
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect();
    let der = base64::engine::general_purpose::STANDARD
        .decode(body)
        .unwrap();
    let endpoint = Endpoint {
        worker_id: WORKER.into(),
        address: ready["address"].as_str().unwrap().into(),
        pin: Sha256::digest(&der).into(),
    };
    (child, endpoint)
}

impl Drop for Tensord {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn machine(name: &str, endpoint: Option<Endpoint>, key: &Path) -> MachineRow {
    MachineRow {
        name: name.into(),
        kind: MachineKind::Added,
        hub: String::new(),
        rental_id: None,
        state: if endpoint.is_some() {
            MachineState::Ready
        } else {
            MachineState::Booting
        },
        served: endpoint.is_some(),
        endpoint,
        key: key.to_owned(),
        reason: None,
    }
}

/// A Hub that accepts connections and never answers: a release run from it stays preparing
/// until it is canceled, and fails once the listener is gone.
pub struct SilentHub(pub std::net::TcpListener);

impl SilentHub {
    pub fn new() -> Self {
        Self(std::net::TcpListener::bind("127.0.0.1:0").unwrap())
    }

    pub fn origin(&self) -> String {
        format!("http://{}", self.0.local_addr().unwrap())
    }
}

/// A warm run of a release from `hub`, with `inputs` written first.
pub fn warm_spec(hub: &str, inputs: Vec<v1::InputFile>) -> Vec<u8> {
    v1::RunSpec {
        kind: v1::RunKind::Warm as i32,
        source: Some(v1::run_spec::Source::Release(v1::Release {
            package: "cozy-test/held".into(),
            release: String::new(),
        })),
        hub: Some(v1::HubAccess {
            origin: hub.into(),
            ..Default::default()
        }),
        inputs,
        ..Default::default()
    }
    .encode_to_vec()
}

/// A file the run names as an input, and what the spec says of it.
pub fn input_file(dir: &Path, bytes: &[u8]) -> (cozy_proto::calcifer::StagedInput, v1::InputFile) {
    let path = dir.join(format!("input-{}", bytes.len()));
    std::fs::write(&path, bytes).unwrap();
    let digest = Digest::of(bytes);
    let staged = cozy_proto::calcifer::StagedInput {
        path,
        digest,
        length: bytes.len() as u64,
    };
    let spec = v1::InputFile {
        field: "image".into(),
        digest: digest.to_string(),
        length: bytes.len() as u64,
        media_type: "image/png".into(),
        order: 0,
    };
    (staged, spec)
}

/// The run's next event that `wanted` accepts.
pub async fn until(
    events: &mut broadcast::Receiver<(RunId, Event)>,
    run: &RunId,
    wanted: impl Fn(&Event) -> bool,
) -> Event {
    // A hang bound for a test whose run never gets there; the product waits without one.
    tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            match events.recv().await {
                Ok((id, event)) if &id == run && wanted(&event) => return event,
                Ok(_) => {}
                Err(e) => panic!("events: {e}"),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("run {run}: the awaited event never came"))
}

/// Local code a machine accepts with no Runtime: a manifest naming a package and nothing else,
/// staged as the run's input, and a warm (install-only) spec of it.
pub fn local_spec(dir: &Path) -> (cozy_proto::calcifer::StagedInput, v1::RunSpec) {
    let bytes = br#"{"package":"local/hello"}"#;
    let path = dir.join("manifest.json");
    std::fs::write(&path, bytes).unwrap();
    let digest = Digest::of(bytes);
    let staged = cozy_proto::calcifer::StagedInput {
        path,
        digest,
        length: bytes.len() as u64,
    };
    let spec = v1::RunSpec {
        kind: v1::RunKind::Warm as i32,
        source: Some(v1::run_spec::Source::Local(v1::LocalSource {
            manifest: digest.to_string(),
        })),
        ..Default::default()
    };
    (staged, spec)
}

/// Whether the machine holds run `id` under exactly `spec`: the same id and spec answer the
/// run, another spec is `run_id_conflict` (the machine's digest of what it was sent).
pub async fn holds_as(link: &calcifer::Link, id: &RunId, spec: v1::RunSpec) -> bool {
    match link.run(id.as_str(), 0, Some(spec)).await {
        Ok(_) => true,
        Err(calcifer::LinkError::Refused { code, .. }) if code == "run_id_conflict" => false,
        Err(e) => panic!("run {id}: {e}"),
    }
}

/// A TensorD running as a computer's persistent machine from a grant (`tensord run`): unlike
/// `serve`, it installs local packages. It admits the test's key.
pub struct Machine {
    tree: cozy_platform::process::Tree,
    pub root: tempfile::TempDir,
    pub endpoint: Endpoint,
}

impl Machine {
    pub fn start() -> Self {
        let binary = std::env::var_os("COZY_TEST_TENSORD")
            .expect("COZY_TEST_TENSORD names the tensord binary these tests run against");
        let root = tempfile::tempdir().unwrap();
        durable::private_dir(root.path()).unwrap();
        let receipt = root.path().join("receipt-key");
        let receipt_key = URL_SAFE_NO_PAD.encode([7u8; 32]);
        durable::replace(&receipt, receipt_key.as_bytes(), 0o600).unwrap();
        let address = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let machine = root.path().join("machine");
        let mut command = Command::new(binary);
        command
            .args(["run", "--persistent"])
            .env_remove("COZY_TEST_TENSORD")
            // A home of its own: this computer's keys and config are not the test machine's.
            .env("HOME", root.path())
            .env("COZY_MACHINE_ROOT", &machine)
            .env("COZY_LISTEN_HOST", "127.0.0.1")
            .env("COZY_WORKER_ID", WORKER)
            .env("COZY_WORKER_INTERNAL_PORT", address.port().to_string())
            .env(
                "COZY_AUTHORIZED_KEYS",
                URL_SAFE_NO_PAD.encode(key().verifying_key().to_bytes()),
            )
            .env("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE", &receipt)
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdout(Stdio::null());
        let process = cozy_platform::process::Lifetime::Owned;
        let mut tree = cozy_platform::process::spawn(command, process).unwrap();
        let leaf = machine.join("run/cozy/bootstrap/tls.crt");
        // A hang bound for a machine that never starts; the product waits without one.
        let until = Instant::now() + Duration::from_secs(120);
        while !(leaf.exists() && std::net::TcpStream::connect(address).is_ok()) {
            if let Ok(Some(status)) = tree.try_wait() {
                panic!("tensord exited before it served: {status}");
            }
            assert!(
                Instant::now() < until,
                "tensord did not listen on {address}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        let endpoint = Endpoint {
            worker_id: WORKER.into(),
            address: address.to_string(),
            pin: Sha256::digest(pem_der(&std::fs::read_to_string(&leaf).unwrap())).into(),
        };
        Self {
            tree,
            root,
            endpoint,
        }
    }

    /// Its TLS leaf's private key (PKCS#8 DER): the key it proves itself with at a Hub.
    pub fn leaf_key(&self) -> Vec<u8> {
        let key = self.root.path().join("machine/run/cozy/bootstrap/tls.key");
        pem_der(&std::fs::read_to_string(key).unwrap())
    }

    pub fn row(&self, name: &str, key: &Path) -> MachineRow {
        machine(name, Some(self.endpoint.clone()), key)
    }
}

impl Drop for Machine {
    /// Ends the exact processes this test started.
    fn drop(&mut self) {
        let _ = self.tree.kill();
        let _ = self.tree.wait();
    }
}

/// A PEM block's DER.
fn pem_der(pem: &str) -> Vec<u8> {
    let body: String = pem
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect();
    base64::engine::general_purpose::STANDARD
        .decode(body)
        .unwrap()
}
