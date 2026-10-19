//! A queued run says what it waits for, in its state, while it waits: another run's install of
//! the same code, or its place in the queue. The queue serves a model's runs together, cancels a
//! queued run, and keeps its runs through a restart (xs-047).
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

/// The cpu_bulk fixture written to the machine as a local tree, its code marked `variant`: its
/// manifest's digest.
async fn tree(client: &Client, variant: &str) -> String {
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
        let mut bytes = std::fs::read(fixture.join(name)).unwrap();
        if name.ends_with(".py") {
            bytes.extend(format!("# {variant}\n").bytes());
        }
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

/// Run `id`'s events to its outcome: every waiting reason its state named, when it started
/// running, and the outcome.
async fn follow(client: &Client, id: &str) -> (Vec<String>, Option<i64>, v1::Outcome) {
    let mut events = Box::pin(client.events(id, 0).await.unwrap());
    let (mut reasons, mut started) = (vec![], None);
    while let Some(event) = events.next().await {
        let event = event.unwrap();
        match event.event {
            v1::Event::State(state) if state.state == "running" => started = Some(event.at_ms),
            v1::Event::State(state) if !state.waiting.is_empty() => reasons.push(state.waiting),
            v1::Event::Outcome(outcome) => return (reasons, started, outcome),
            _ => {}
        }
    }
    panic!("run {id}'s log ended without an outcome");
}

/// Until run `id` is in a state `seen` accepts.
async fn until(client: &Client, id: &str, seen: impl Fn(&v1::RunState) -> bool) {
    let mut events = Box::pin(client.events(id, 0).await.unwrap());
    while let Some(event) = events.next().await {
        match event.unwrap().event {
            v1::Event::State(state) if seen(&state) => return,
            v1::Event::Outcome(outcome) => panic!("{id} ended first: {outcome:?}"),
            _ => {}
        }
    }
}
fn place(state: &v1::RunState) -> bool {
    state.waiting.starts_with("position ")
}

#[tokio::test]
async fn a_queued_run_names_the_install_or_the_executor_slot_it_waits_for() {
    let home = tempfile::tempdir().unwrap();
    let (_machine, client) = boot(home.path());
    let manifest = tree(&client, "").await;

    // Runs a and b of one tree, sent together: one builds its environment, and the other waits
    // for that run's install, by name.
    let nap = r#"{"seconds":0.0}"#;
    client.submit(&v1::RunRequest { id: "a".into(), spec: spec(&manifest, "", "") }).await.unwrap();
    client.submit(&v1::RunRequest { id: "b".into(), spec: spec(&manifest, "nap", nap) }).await.unwrap();
    let ((a_reasons, _, a), (b_reasons, _, b)) = tokio::join!(follow(&client, "a"), follow(&client, "b"));
    println!("a waited for {a_reasons:?}; b waited for {b_reasons:?}");
    assert_eq!((a.status.as_str(), b.status.as_str()), ("succeeded", "succeeded"), "{a:?} {b:?}");
    let waited = |reasons: &[String], other: &str| {
        reasons.iter().any(|r| r == &format!("waiting for run {other} to install local/tensord-cpu-bulk"))
    };
    assert!(waited(&b_reasons, "a") ^ waited(&a_reasons, "b"), "{a_reasons:?} {b_reasons:?}");

    // This computer's machine runs one deviceless call at a time: d, sent once c runs, waits
    // first in line for c's slot.
    let long = r#"{"seconds":6.0}"#;
    client.submit(&v1::RunRequest { id: "c".into(), spec: spec(&manifest, "nap", long) }).await.unwrap();
    until(&client, "c", |state| state.state == "running").await;
    client.submit(&v1::RunRequest { id: "d".into(), spec: spec(&manifest, "nap", nap) }).await.unwrap();
    let (reasons, _, outcome) = follow(&client, "d").await;
    println!("d waited for {reasons:?}");
    assert_eq!(outcome.status, "succeeded", "{outcome:?}");
    assert!(reasons.iter().any(|r| r == "position 1 in the queue"), "{reasons:?}");
    assert_eq!(follow(&client, "c").await.2.status, "succeeded");
}

#[tokio::test]
async fn queued_runs_start_a_model_at_a_time_show_their_place_and_cancel() {
    let home = tempfile::tempdir().unwrap();
    let (_machine, client) = boot(home.path());
    let manifest = tree(&client, "").await;
    client.submit(&v1::RunRequest { id: "w".into(), spec: spec(&manifest, "", "") }).await.unwrap();
    assert_eq!(follow(&client, "w").await.2.status, "succeeded");

    // c holds the one slot while runs of two models (`nap`, `bulk`) queue behind it, interleaved.
    let (nap, bulk) = (r#"{"seconds":0.5}"#, r#"{"size":1024}"#);
    client.submit(&v1::RunRequest { id: "c".into(), spec: spec(&manifest, "nap", r#"{"seconds":8.0}"#) }).await.unwrap();
    until(&client, "c", |state| state.state == "running").await;
    let queued = [("n1", "nap", nap), ("b1", "bulk", bulk), ("n2", "nap", nap), ("x", "nap", nap), ("b2", "bulk", bulk), ("n3", "nap", nap)];
    let mut followed = vec![];
    for (id, entrypoint, payload) in queued {
        client.submit(&v1::RunRequest { id: id.into(), spec: spec(&manifest, entrypoint, payload) }).await.unwrap();
        let client = client.clone();
        followed.push((id, entrypoint, tokio::spawn(async move { follow(&client, id).await })));
    }
    // x is cancelled in line: it never starts.
    until(&client, "x", place).await;
    client.cancel("x").await.unwrap();

    let mut runs = vec![];
    for (id, entrypoint, followed) in followed {
        let (reasons, started, outcome) = followed.await.unwrap();
        println!("{id} ({entrypoint}) waited for {reasons:?}, started at {started:?}");
        if id == "x" {
            assert_eq!((outcome.status.as_str(), started), ("canceled", None), "{outcome:?}");
            continue;
        }
        assert_eq!(outcome.status, "succeeded", "{id}: {outcome:?}");
        assert!(reasons.iter().any(|r| r.starts_with("position ") && r.ends_with(" in the queue")), "{id}: {reasons:?}");
        runs.push((started.unwrap(), entrypoint));
    }
    // From c's model, one switch to the other and one back, never one per run as in arrival order.
    runs.sort();
    let order: Vec<_> = std::iter::once("nap").chain(runs.iter().map(|(_, model)| *model)).collect();
    let switches = order.windows(2).filter(|pair| pair[0] != pair[1]).count();
    assert_eq!(switches, 2, "{order:?}");
}

#[tokio::test]
async fn a_restart_keeps_the_queue_and_prepares_an_interrupted_run_again() {
    let home = tempfile::tempdir().unwrap();
    let (machine, client) = boot(home.path());
    let manifest = tree(&client, "").await;
    client.submit(&v1::RunRequest { id: "w".into(), spec: spec(&manifest, "", "") }).await.unwrap();
    assert_eq!(follow(&client, "w").await.2.status, "succeeded");
    // q waits in line behind c; p, of a release not installed yet, is preparing when the
    // machine dies.
    client.submit(&v1::RunRequest { id: "c".into(), spec: spec(&manifest, "nap", r#"{"seconds":60.0}"#) }).await.unwrap();
    until(&client, "c", |state| state.state == "running").await;
    client.submit(&v1::RunRequest { id: "q".into(), spec: spec(&manifest, "nap", r#"{"seconds":0.0}"#) }).await.unwrap();
    until(&client, "q", place).await;
    let fresh = tree(&client, "fresh").await;
    client.submit(&v1::RunRequest { id: "p".into(), spec: spec(&fresh, "nap", r#"{"seconds":0.0}"#) }).await.unwrap();
    let first = Box::pin(client.events("p", 0).await.unwrap()).next().await.unwrap().unwrap().event;
    assert!(matches!(&first, v1::Event::State(state) if state.state == "preparing"), "{first:?}");
    drop(machine);
    std::fs::remove_file(home.path().join(".cozy/machine/var/lib/cozy/tensord/api-ready.json")).unwrap();

    let (_machine, client) = boot(home.path());
    let (c, q, p) = tokio::join!(follow(&client, "c"), follow(&client, "q"), follow(&client, "p"));
    println!("after the restart: c {c:?}\nq {q:?}\np {p:?}");
    assert_ne!(c.2.status, "succeeded", "c was running: a restart ends it");
    assert_eq!((q.2.status.as_str(), p.2.status.as_str()), ("succeeded", "succeeded"), "{q:?} {p:?}");
}
