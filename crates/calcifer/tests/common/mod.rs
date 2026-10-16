//! What the tests share: a real TensorD process (`COZY_TEST_TENSORD`, the binary a Runtime wheel
//! carries) on a temp root that admits the test's key, and run specs it accepts with no Runtime.
#![allow(dead_code)]
use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use calcifer::store::{Endpoint, MachineRow};
use cozy_platform::durable;
use cozy_proto::{
    calcifer::{Event, MachineKind, MachineState},
    ids::{Digest, RunId},
    machine as v1,
};
use ed25519_dalek::{SigningKey, pkcs8::EncodePrivateKey as _};
use prost::Message as _;
use sha2::{Digest as _, Sha256};
use tokio::sync::broadcast;

pub const WORKER: &str = "calcifer-test";

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
}

impl Tensord {
    pub fn start() -> Self {
        let binary = std::env::var_os("COZY_TEST_TENSORD")
            .expect("COZY_TEST_TENSORD names the tensord binary these tests run against");
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
        let state = root.path().join("state");
        let mut child = Command::new(binary)
            .args(["serve", "--state"])
            .arg(&state)
            .arg("--machine-config")
            .arg(config.join("machine.json"))
            .args(["--listen", "127.0.0.1:0", "--host-bytes", "0"])
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
        Self {
            child,
            root,
            endpoint,
        }
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
