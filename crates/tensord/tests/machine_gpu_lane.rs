//! A machine's GPU lanes, on a runner whose envelope names two devices: a call that constructs
//! no model (a script) is given the GPUs it asks for, all in its one process, and a count wider
//! than the machine is refused by name; a call that ignores its cancel is killed once its
//! patience is spent, ends CANCELED saying so, and its GPU serves the next run.
#[path = "common/named_runtime.rs"]
mod named_runtime;
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

/// This computer's machine from `home`, its GPU envelope two devices, with its owner's client.
fn boot(home: &Path) -> (Machine, Client) {
    let keys = home.join(".cozy/keys");
    std::fs::create_dir_all(&keys).unwrap();
    let owner = SigningKey::from_bytes(&[6; 32]);
    std::fs::write(keys.join("owner.pem"), owner.to_pkcs8_pem(Default::default()).unwrap().as_bytes()).unwrap();
    // The operator's GPU settings over the probe's: this runner has no GPU, so the envelope is
    // named here. Nothing in the fixture touches a device.
    let state = home.join(".cozy/machine/var/lib/cozy/machine");
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(state.join("gpu-config.json"), r#"{"devices": "0,1", "prespawn": false}"#).unwrap();
    named_runtime::image(&home.join(".cozy/machine"));
    let mut machine = Machine(
        Command::new(env!("CARGO_BIN_EXE_tensord"))
            .arg("--persistent")
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", home)
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
    let client = Client::new(ready["address"].as_str().unwrap(), tensorfs::sha256::digest(&leaf), auth).unwrap();
    (machine, client)
}

/// The cpu_bulk fixture written to the machine as a local tree: its manifest's digest.
async fn tree(client: &Client) -> String {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_bulk");
    let bulk = client.bulk().unwrap();
    let write = async |bytes: Vec<u8>| -> String {
        let digest = format!("sha256:{}", tensorfs::sha256::hex_digest(&bytes));
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

/// Run `id` of `devices` asking for `gpus`, followed to its outcome.
async fn run(client: &Client, manifest: &str, id: &str, gpus: u32) -> v1::Outcome {
    let spec = v1::RunSpec {
        kind: v1::RunKind::Call,
        source: Some(v1::Source::Local(v1::LocalSource { manifest: manifest.into() })),
        entrypoint: "devices".into(),
        payload: Some(v1::Json::parse("{}").unwrap()),
        gpus,
        ..Default::default()
    };
    client.submit(&v1::RunRequest { id: id.into(), spec }).await.unwrap();
    let mut events = Box::pin(client.events(id, 0).await.unwrap());
    while let Some(event) = events.next().await {
        let event = event.unwrap().event;
        eprintln!("{id}: {event:?}");
        if let v1::Event::Outcome(outcome) = event {
            return outcome;
        }
    }
    panic!("run {id}'s log ended without an outcome");
}

fn devices(outcome: &v1::Outcome) -> String {
    assert_eq!(outcome.status, "succeeded", "{outcome:?}");
    let result: serde_json::Value = serde_json::from_str(outcome.result.as_ref().unwrap().get()).unwrap();
    result["devices"].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn a_script_holds_the_gpus_it_asks_for_in_its_one_process() {
    let home = tempfile::tempdir().unwrap();
    let (_machine, client) = boot(home.path());
    let manifest = tree(&client).await;
    // Asked for none, it takes one GPU; asked for two, its process is sealed to both.
    assert_eq!(devices(&run(&client, &manifest, "one", 0).await), "0");
    assert_eq!(devices(&run(&client, &manifest, "two", 2).await), "0,1");
    // Wider than the machine: refused by name, never narrowed.
    let three = run(&client, &manifest, "three", 3).await;
    let reason = three.reason.as_ref().map(|r| (r.code.as_str(), r.message.as_str()));
    assert!(three.status != "succeeded" && format!("{reason:?}").contains("gpu_count_unavailable"), "{three:?}");
}

#[tokio::test]
async fn a_call_that_ignores_its_cancel_is_killed_and_its_gpu_serves_the_next_run() {
    let home = tempfile::tempdir().unwrap();
    let (_machine, client) = boot(home.path());
    let manifest = tree(&client).await;
    let pidfile = home.path().join("stubborn.pid");
    let spec = v1::RunSpec {
        kind: v1::RunKind::Call,
        source: Some(v1::Source::Local(v1::LocalSource { manifest: manifest.clone() })),
        entrypoint: "stubborn".into(),
        payload: Some(v1::Json::parse(serde_json::json!({"label": pidfile}).to_string()).unwrap()),
        ..Default::default()
    };
    client.submit(&v1::RunRequest { id: "stuck".into(), spec }).await.unwrap();
    // Canceled once its pid is written; a run that ends before then fails here, not a clock.
    let mut events = Box::pin(client.events("stuck", 0).await.unwrap());
    let (mut lines, mut pid) = (vec![], None::<i32>);
    let outcome = loop {
        if pid.is_none() {
            pid = std::fs::read_to_string(&pidfile).ok().and_then(|text| text.trim().parse().ok());
            if pid.is_some() {
                client.cancel("stuck").await.unwrap();
            }
        }
        let event = tokio::select! {
            event = events.next() => event.expect("the run's log ended without an outcome").unwrap().event,
            _ = tokio::time::sleep(Duration::from_millis(100)) => continue,
        };
        eprintln!("stuck: {event:?}");
        match event {
            v1::Event::Log(line) => lines.push(line.text),
            v1::Event::Outcome(ended) => break ended,
            _ => {}
        }
    };
    let pid = pid.unwrap_or_else(|| panic!("the stubborn call ended before it ran: {outcome:?}"));
    assert_eq!(outcome.status, "canceled", "{outcome:?}");
    assert!(lines.iter().any(|l| l.contains("it did not stop for its cancel")), "{lines:?}");
    assert!(!Path::new(&format!("/proc/{pid}")).exists(), "the stubborn executor still runs");
    // Its GPU is free: the next call takes it.
    assert_eq!(devices(&run(&client, &manifest, "after", 0).await), "0");
}
