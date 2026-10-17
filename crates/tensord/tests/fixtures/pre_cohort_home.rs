//! A machine home as a v0.9 machine leaves it (its journal's stored values in protobuf), for
//! journal_cohort.rs. Copied into a v0.9.1 checkout's crates/tensord/tests and run there with
//! COZY_TEST_KEEP_HOME=<empty dir>: the machine runs `--persistent` from its environment as
//! v0.9's did, installs local code (cpu_memo) and runs a call that writes a file output, then
//! stops. The home stays, with `pre-cohort.json` naming the code's manifest.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::SigningKey;
use std::{
    os::unix::{fs::PermissionsExt, process::CommandExt},
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tensord::api::{
    capability::{self, Grant},
    v1::{self, machine_client::MachineClient},
};
use tonic::{
    metadata::MetadataValue,
    transport::{Certificate, ClientTlsConfig, Endpoint},
    Request,
};

const OWNER: [u8; 32] = [5; 32];

fn authorized<T>(message: T) -> Request<T> {
    let expires = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64 + 600;
    let grant = Grant { machine: "local".into(), action: capability::MACHINE.into(), expires, ..Default::default() };
    let mut request = Request::new(message);
    let value = format!("Cozy-Cap {}", capability::mint(&SigningKey::from_bytes(&OWNER), grant));
    request.metadata_mut().insert("authorization", MetadataValue::try_from(value).unwrap());
    request
}

#[tokio::test]
async fn a_v0_9_machine_home() {
    let Some(home) = std::env::var_os("COZY_TEST_KEEP_HOME") else { return };
    let home = Path::new(&home);
    let root = home.join(".cozy/machine");
    // A laptop's machine root has the uv the CLI installed there.
    std::fs::create_dir_all(root.join("usr/local/bin")).unwrap();
    let uv = Command::new("sh").args(["-c", "command -v uv"]).output().unwrap();
    std::fs::copy(String::from_utf8(uv.stdout).unwrap().trim(), root.join("usr/local/bin/uv")).unwrap();
    let key_file = home.join("receipt-key");
    std::fs::write(&key_file, URL_SAFE_NO_PAD.encode([7u8; 32])).unwrap();
    std::fs::set_permissions(&key_file, std::fs::Permissions::from_mode(0o600)).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let owner = URL_SAFE_NO_PAD.encode(SigningKey::from_bytes(&OWNER).verifying_key().as_bytes());
    let mut machine = Command::new(env!("CARGO_BIN_EXE_tensord"))
        .arg("run")
        .env("HOME", home)
        .env("COZY_MACHINE_ROOT", &root)
        .env("COZY_MACHINE_LIFETIME", "persistent")
        .env("COZY_LISTEN_HOST", "127.0.0.1")
        .env("COZY_WORKER_ID", "local")
        .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
        .env("COZY_AUTHORIZED_KEYS", owner)
        .env("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE", &key_file)
        .env("CUDA_VISIBLE_DEVICES", "")
        .process_group(0)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let ready_path = root.join("var/lib/cozy/tensord/api-ready.json");
    let start = Instant::now();
    let ready: serde_json::Value = loop {
        assert!(machine.try_wait().unwrap().is_none(), "the machine exited");
        if let Ok(bytes) = std::fs::read(&ready_path) {
            break serde_json::from_slice(&bytes).unwrap();
        }
        assert!(start.elapsed() < Duration::from_secs(300), "no api-ready.json");
        std::thread::sleep(Duration::from_millis(100));
    };
    let tls = ClientTlsConfig::new()
        .ca_certificate(Certificate::from_pem(ready["cert_pem"].as_str().unwrap()))
        .domain_name("cozy-worker");
    let channel = Endpoint::from_shared(format!("https://{}", ready["address"].as_str().unwrap()))
        .unwrap()
        .tls_config(tls)
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = MachineClient::new(channel);

    let mut write = async |bytes: &[u8]| -> String {
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(bytes));
        let mut frames = vec![v1::WriteFrame { digest: digest.clone(), length: bytes.len() as u64, offset: 0, data: vec![] }];
        frames.extend(bytes.chunks(64 << 10).map(|chunk| v1::WriteFrame { data: chunk.to_vec(), ..Default::default() }));
        client.write(authorized(tokio_stream::iter(frames))).await.unwrap();
        digest
    };
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_memo");
    let mut files = vec![];
    for name in ["pyproject.toml", "package.toml", "cpu_memo/__init__.py"] {
        let bytes = std::fs::read(fixture.join(name)).unwrap();
        files.push(serde_json::json!({"name": name, "digest": write(&bytes).await, "length": bytes.len()}));
    }
    let manifest = serde_json::json!({"package": "local/tensord-cpu-memo", "release": "0.1.0", "files": files});
    let manifest = write(manifest.to_string().as_bytes()).await;

    let counter = home.join("counter");
    let payload = serde_json::json!({"value": 3, "counter": counter}).to_string().into_bytes();
    for (id, kind, entrypoint) in [("warm", v1::RunKind::Warm, ""), ("note", v1::RunKind::Call, "note")] {
        let spec = v1::RunSpec {
            kind: kind as i32,
            source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest: manifest.clone() })),
            entrypoint: entrypoint.into(),
            payload: if entrypoint.is_empty() { vec![] } else { payload.clone() },
            ..Default::default()
        };
        let request = authorized(v1::RunRequest { id: id.into(), after: 0, spec: Some(spec) });
        let mut events = client.run(request).await.unwrap().into_inner();
        let mut outcome = None;
        while let Some(event) = events.message().await.unwrap() {
            if let Some(v1::run_event::Event::Outcome(done)) = event.event {
                outcome = Some(done);
            }
        }
        let outcome = outcome.expect("an outcome");
        assert_eq!(outcome.status, "succeeded", "{id}: {outcome:?}");
        if id == "note" {
            assert!(!outcome.outputs.is_empty(), "{outcome:?}");
        }
    }
    std::fs::write(home.join("pre-cohort.json"), serde_json::json!({"manifest": manifest}).to_string()).unwrap();
    // Its process group: the stable parent and the service it forked.
    let _ = Command::new("kill").args(["-KILL", "--", &format!("-{}", machine.id())]).status();
    let _ = machine.wait();
}
