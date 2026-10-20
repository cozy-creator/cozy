//! A call that raises: its traceback travels with the run, in its log lines and its outcome's
//! `reason.detail`, so `cozy run show` explains it once the machine is gone.
#[path = "common/child.rs"]
mod child;
#[path = "common/named_runtime.rs"]
mod named_runtime;
use child::BareEnv;
use ed25519_dalek::{pkcs8::EncodePrivateKey, SigningKey};
use std::{
    path::Path,
    process::{Child, Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};
use tensord::api::v1::{
    self,
    client::{Auth, Machine as Client},
};
use tokio_stream::StreamExt;

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn a_raising_calls_traceback_travels_with_its_run() {
    let home = tempfile::tempdir().unwrap();
    let keys = home.path().join(".cozy/keys");
    std::fs::create_dir_all(&keys).unwrap();
    let owner = SigningKey::from_bytes(&[5; 32]);
    std::fs::write(keys.join("owner.pem"), owner.to_pkcs8_pem(Default::default()).unwrap().as_bytes()).unwrap();
    named_runtime::image(&home.path().join(".cozy/machine"));
    let mut machine = Machine(
        Command::new(env!("CARGO_BIN_EXE_tensord"))
            .arg("--persistent")
            .bare_env()
            .user_home(home.path())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let path = home.path().join(".cozy/machine/var/lib/cozy/tensord/api-ready.json");
    let start = Instant::now();
    let ready: serde_json::Value = loop {
        assert!(machine.0.try_wait().unwrap().is_none(), "the machine exited");
        if let Ok(bytes) = std::fs::read(&path) {
            break serde_json::from_slice(&bytes).unwrap();
        }
        assert!(start.elapsed() < Duration::from_secs(300), "no api-ready.json");
        std::thread::sleep(Duration::from_millis(100));
    };
    let pem = ready["cert_pem"].as_str().unwrap();
    let leaf = rustls_pemfile::certs(&mut pem.as_bytes()).next().unwrap().unwrap();
    let auth = Auth::Key { key: Arc::new(owner), worker: "local".into() };
    let client = Client::new(ready["address"].as_str().unwrap(), tensorfs::sha256::digest(&leaf), auth).unwrap();

    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_publish");
    let bulk = client.bulk().unwrap();
    let write = async |bytes: Vec<u8>| -> String {
        let digest = format!("sha256:{}", tensorfs::sha256::hex_digest(&bytes));
        let length = bytes.len() as u64;
        bulk.put(&digest, (0, length, length), bytes).await.unwrap();
        digest
    };
    let mut files = vec![];
    for name in ["pyproject.toml", "package.toml", "cpu_publish/__init__.py"] {
        let bytes = std::fs::read(fixture.join(name)).unwrap();
        let length = bytes.len();
        files.push(serde_json::json!({"name": name, "digest": write(bytes).await, "length": length}));
    }
    let manifest = serde_json::json!({"package": "local/tensord-cpu-publish", "release": "0.1.0", "files": files});
    let manifest = write(manifest.to_string().into_bytes()).await;
    let spec = v1::RunSpec {
        kind: v1::RunKind::Call,
        source: Some(v1::Source::Local(v1::LocalSource { manifest })),
        entrypoint: "explode".into(),
        payload: Some(v1::Json::parse("{}").unwrap()),
        ..Default::default()
    };
    client.submit(&v1::RunRequest { id: "explode".into(), spec }).await.unwrap();
    let mut events = Box::pin(client.events("explode", 0).await.unwrap());
    let (mut logged, mut outcome) = (vec![], None);
    while let Some(event) = events.next().await {
        match event.unwrap().event {
            v1::Event::Log(line) => logged.push(line.text),
            v1::Event::Outcome(done) => {
                outcome = Some(done);
                break;
            }
            _ => {}
        }
    }
    let outcome = outcome.expect("an outcome");
    assert_eq!(outcome.status, "failed", "{outcome:?}");
    let reason = outcome.reason.unwrap();
    assert!(reason.detail.contains("Traceback") || reason.detail.contains("explode"), "{reason:?}");
    assert!(reason.detail.contains("ValueError: boom from the fixture"), "{reason:?}");
    assert!(logged.iter().any(|line| line.contains("ValueError: boom from the fixture")), "{logged:?}");
}
