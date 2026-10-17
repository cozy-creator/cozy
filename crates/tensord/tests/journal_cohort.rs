//! A machine home v0.9 left (COZY_TEST_PRE_COHORT_HOME, built by tests/fixtures/pre_cohort_home.rs
//! at v0.9.1), opened by this machine on a copy: its journal of protobuf values is archived
//! beside it, never read, and a fresh journal starts. The machine keeps its identity (leaf,
//! worker id, owner key), its store and its installed environment, whose code runs again with
//! no reinstall.
use ed25519_dalek::{pkcs8::EncodePrivateKey, SigningKey};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
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

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn authorized<T>(message: T) -> Request<T> {
    let expires = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64 + 600;
    let grant = Grant { machine: "local".into(), action: capability::MACHINE.into(), expires, ..Default::default() };
    let mut request = Request::new(message);
    let value = format!("Cozy-Cap {}", capability::mint(&SigningKey::from_bytes(&OWNER), grant));
    request.metadata_mut().insert("authorization", MetadataValue::try_from(value).unwrap());
    request
}

/// Every file under `dir`, relative.
fn files(dir: &Path) -> BTreeSet<PathBuf> {
    let mut found = BTreeSet::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).into_iter().flatten().flatten() {
            match entry.file_type().unwrap().is_dir() {
                true => pending.push(entry.path()),
                false => drop(found.insert(entry.path().strip_prefix(dir).unwrap().to_path_buf())),
            }
        }
    }
    found
}

fn installations(journal: &Path) -> Vec<String> {
    let db = rusqlite::Connection::open(journal).unwrap();
    let mut rows = db.prepare("SELECT alias FROM installations ORDER BY alias").unwrap();
    rows.query_map([], |r| r.get(0)).unwrap().map(Result::unwrap).collect()
}

#[tokio::test]
async fn a_v0_9_home_keeps_its_identity_store_and_environments_across_the_journal_reset() {
    let Some(built) = std::env::var_os("COZY_TEST_PRE_COHORT_HOME") else {
        eprintln!("skipped: no pre-cohort home (COZY_TEST_PRE_COHORT_HOME is unset)");
        return;
    };
    let home = tempfile::tempdir().unwrap();
    let copied = Command::new("cp").arg("-a").arg(Path::new(&built).join(".")).arg(home.path()).status().unwrap();
    assert!(copied.success());
    let root = home.path().join(".cozy/machine");
    let engine = root.join("var/lib/cozy/tensord");
    let journal = files(&root)
        .into_iter()
        .find(|path| path.file_name().is_some_and(|name| name == "executions.sqlite3"))
        .map(|path| root.join(path))
        .expect("a v0.9 journal");
    let before: serde_json::Value =
        serde_json::from_slice(&std::fs::read(engine.join("api-ready.json")).unwrap()).unwrap();
    std::fs::remove_file(engine.join("api-ready.json")).unwrap();
    let installed = installations(&journal);
    assert!(!installed.is_empty(), "the v0.9 home installed its code");
    let store = root.join("var/lib/tensorfs");
    let held = files(&store.join("blobs"));
    let generations = files(&engine.join("generations")).len();
    assert!(generations > 0, "the v0.9 home holds its environment");
    // Calcifer's owner key, as the home holds it.
    std::fs::create_dir_all(home.path().join(".cozy/keys")).unwrap();
    let pem = SigningKey::from_bytes(&OWNER).to_pkcs8_pem(Default::default()).unwrap();
    std::fs::write(home.path().join(".cozy/keys/owner.pem"), pem.as_bytes()).unwrap();

    let mut machine = Machine(
        Command::new(env!("CARGO_BIN_EXE_tensord"))
            .arg("--persistent")
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", home.path())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    let ready: serde_json::Value = loop {
        assert!(machine.0.try_wait().unwrap().is_none(), "the machine exited");
        if let Ok(bytes) = std::fs::read(engine.join("api-ready.json")) {
            break serde_json::from_slice(&bytes).unwrap();
        }
        assert!(start.elapsed() < Duration::from_secs(300), "no api-ready.json");
        std::thread::sleep(Duration::from_millis(100));
    };
    assert_eq!(ready["cert_pem"], before["cert_pem"], "the pinned leaf survives");
    assert_eq!(ready["worker_id"], "local");

    // The old journal lies beside the fresh one, whole; the fresh one carried the installations.
    let beside: Vec<_> = std::fs::read_dir(journal.parent().unwrap())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("executions.format-1-") && name.ends_with(".sqlite3"))
        .collect();
    assert_eq!(beside.len(), 1, "{beside:?}");
    assert_eq!(installations(&journal.parent().unwrap().join(&beside[0])), installed);
    assert_eq!(installations(&journal), installed);
    let format: String = rusqlite::Connection::open(&journal)
        .unwrap()
        .query_row("SELECT value FROM machine_metadata WHERE key='journal_format'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(format, tensord::journal::JOURNAL_FORMAT.to_string());

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
    let mut frames = client.status(authorized(v1::StatusRequest { keepalive: false })).await.unwrap().into_inner();
    assert_eq!(frames.message().await.unwrap().unwrap().worker_id, "local");

    // The installed code runs again from the objects the owner wrote before, with no reinstall.
    let pre: serde_json::Value =
        serde_json::from_slice(&std::fs::read(home.path().join("pre-cohort.json")).unwrap()).unwrap();
    let counter = home.path().join("counter-after");
    let spec = v1::RunSpec {
        kind: v1::RunKind::Call as i32,
        source: Some(v1::run_spec::Source::Local(v1::LocalSource { manifest: pre["manifest"].as_str().unwrap().into() })),
        entrypoint: "note".into(),
        payload: serde_json::json!({"value": 4, "counter": counter}).to_string().into_bytes(),
        ..Default::default()
    };
    let request = authorized(v1::RunRequest { id: "after".into(), after: 0, spec: Some(spec) });
    let mut events = client.run(request).await.unwrap().into_inner();
    let mut outcome = None;
    while let Some(event) = events.message().await.unwrap() {
        if let Some(v1::run_event::Event::Outcome(done)) = event.event {
            outcome = Some(done);
        }
    }
    let outcome = outcome.expect("an outcome");
    assert_eq!(outcome.status, "succeeded", "{outcome:?}");
    assert!(!outcome.outputs.is_empty(), "{outcome:?}");
    assert_eq!(files(&engine.join("generations")).len(), generations, "no environment was rebuilt");
    assert!(held.is_subset(&files(&store.join("blobs"))), "the store kept every object");
    drop(machine);
}
