//! A queued run says what it waits for, in its state, while it waits: another run's install of
//! the same code, or an executor slot this machine's other runs hold.
#[path = "common/child.rs"]
mod child;
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

/// This computer's machine from `home`, with the client its owner signs with.
fn boot(home: &Path) -> (Machine, Client) {
    let keys = home.join(".cozy/keys");
    std::fs::create_dir_all(&keys).unwrap();
    let owner = SigningKey::from_bytes(&[5; 32]);
    std::fs::write(keys.join("owner.pem"), owner.to_pkcs8_pem(Default::default()).unwrap().as_bytes()).unwrap();
    let mut machine = Machine(
        Command::new(env!("CARGO_BIN_EXE_tensord"))
            .arg("--persistent")
            .bare_env()
            .user_home(home)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let path = home.join(".cozy/machine/var/lib/cozy/tensord/api-ready.json");
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
    let client = Client::new(ready["address"].as_str().unwrap(), tensorfs_core::sha256::digest(&leaf), auth).unwrap();
    (machine, client)
}

/// The cpu_bulk fixture written to the machine as a local tree: its manifest's digest.
async fn tree(client: &Client) -> String {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_bulk");
    let bulk = client.bulk().unwrap();
    let write = async |bytes: Vec<u8>| -> String {
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&bytes));
        let length = bytes.len() as u64;
        bulk.put(&digest, (0, length, length), bytes).await.unwrap();
        digest
    };
    let mut files = vec![];
    for name in ["pyproject.toml", "package.toml", "cpu_bulk/__init__.py"] {
        let bytes = std::fs::read(fixture.join(name)).unwrap();
        let length = bytes.len();
        files.push(serde_json::json!({"name": name, "digest": write(bytes).await, "length": length}));
    }
    let manifest = serde_json::json!({"package": "local/tensord-cpu-bulk", "release": "0.1.0", "files": files});
    write(manifest.to_string().into_bytes()).await
}

fn spec(manifest: &str, entrypoint: &str, payload: &str) -> v1::RunSpec {
    v1::RunSpec {
        kind: if entrypoint.is_empty() { v1::RunKind::Warm } else { v1::RunKind::Call },
        source: Some(v1::Source::Local(v1::LocalSource { manifest: manifest.into() })),
        entrypoint: entrypoint.into(),
        payload: (!payload.is_empty()).then(|| v1::Json::parse(payload).unwrap()),
        ..Default::default()
    }
}

/// Run `id`'s events to its outcome: every waiting reason its state named, and the outcome.
async fn follow(client: &Client, id: &str) -> (Vec<String>, v1::Outcome) {
    let mut events = Box::pin(client.events(id, 0).await.unwrap());
    let mut reasons = vec![];
    while let Some(event) = events.next().await {
        match event.unwrap().event {
            v1::Event::State(state) if !state.waiting.is_empty() => reasons.push(state.waiting),
            v1::Event::Outcome(outcome) => return (reasons, outcome),
            _ => {}
        }
    }
    panic!("run {id}'s log ended without an outcome");
}

#[tokio::test]
async fn a_queued_run_names_the_install_or_the_executor_slot_it_waits_for() {
    let home = tempfile::tempdir().unwrap();
    let (_machine, client) = boot(home.path());
    let manifest = tree(&client).await;

    // Runs a and b of one tree, sent together: one builds its environment, and the other waits
    // for that run's install, by name.
    let nap = r#"{"seconds":0.0}"#;
    client.submit(&v1::RunRequest { id: "a".into(), spec: spec(&manifest, "", "") }).await.unwrap();
    client.submit(&v1::RunRequest { id: "b".into(), spec: spec(&manifest, "nap", nap) }).await.unwrap();
    let ((a_reasons, a), (b_reasons, b)) = tokio::join!(follow(&client, "a"), follow(&client, "b"));
    println!("a waited for {a_reasons:?}; b waited for {b_reasons:?}");
    assert_eq!((a.status.as_str(), b.status.as_str()), ("succeeded", "succeeded"), "{a:?} {b:?}");
    let waited = |reasons: &[String], other: &str| {
        reasons.iter().any(|r| r == &format!("waiting for run {other} to install local/tensord-cpu-bulk"))
    };
    assert!(waited(&b_reasons, "a") ^ waited(&a_reasons, "b"), "{a_reasons:?} {b_reasons:?}");

    // This computer's machine runs one deviceless call at a time: d, sent once c runs, waits
    // for c's slot.
    let long = r#"{"seconds":6.0}"#;
    client.submit(&v1::RunRequest { id: "c".into(), spec: spec(&manifest, "nap", long) }).await.unwrap();
    let mut c = Box::pin(client.events("c", 0).await.unwrap());
    while let Some(event) = c.next().await {
        match event.unwrap().event {
            v1::Event::State(state) if state.state == "running" => break,
            v1::Event::Outcome(outcome) => panic!("c ended before d was sent: {outcome:?}"),
            _ => {}
        }
    }
    client.submit(&v1::RunRequest { id: "d".into(), spec: spec(&manifest, "nap", nap) }).await.unwrap();
    let (reasons, outcome) = follow(&client, "d").await;
    println!("d waited for {reasons:?}");
    assert_eq!(outcome.status, "succeeded", "{outcome:?}");
    assert!(reasons.iter().any(|r| r == "waiting for an executor slot (1 busy)"), "{reasons:?}");
    assert_eq!(follow(&client, "c").await.1.status, "succeeded");
}
