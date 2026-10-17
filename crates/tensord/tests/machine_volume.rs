//! The datacenter volume on the real binary in machine mode: two machines (two Stores) given one
//! cache mount by `COZY_TENSORFS_CACHE`, and a test Hub. The second machine's download comes from
//! the volume, as its run's record says; a machine without the variable downloads from the Hub.
mod common;
#[path = "common/hub_oauth.rs"]
mod hub_oauth;
#[path = "common/names_hub.rs"]
mod names_hub;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::SigningKey;
use std::{
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
};

const WORKER: &str = "volume-test";
const OWNER: [u8; 32] = [33; 32];

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn boot(root: &Path, port: u16, cache: Option<&Path>) -> Machine {
    let owner = SigningKey::from_bytes(&OWNER).verifying_key();
    let mut command = Command::new(env!("CARGO_BIN_EXE_tensord"));
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("COZY_MACHINE_ROOT", root)
        .env("COZY_WORKER_ID", WORKER)
        .env("COZY_WORKER_AUTH_TOKEN", URL_SAFE_NO_PAD.encode([1; 32]))
        .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
        .env("COZY_LISTEN_HOST", "127.0.0.1")
        .env("COZY_AUTHORIZED_KEYS", URL_SAFE_NO_PAD.encode(owner.as_bytes()))
        .env("COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL", URL_SAFE_NO_PAD.encode([7; 32]))
        .env("TENSORHUB_ORIGIN", "https://hub.invalid")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("COZY_MACHINE_LIFETIME", "persistent")
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(cache) = cache {
        command.env("COZY_TENSORFS_CACHE", cache);
    }
    Machine(command.spawn().unwrap())
}

/// A machine root, booted and answering, and a client of it under the owner's machine cap.
async fn start(name: &str, cache: Option<&Path>) -> (Machine, MachineClient<tonic::transport::Channel>, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/machine-volume")
        .join(format!("{name}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut machine = boot(&root, port, cache);
    let began = Instant::now();
    while common::receipt(&root, port).is_none() {
        assert!(machine.0.try_wait().unwrap().is_none(), "the machine exited");
        assert!(began.elapsed() < Duration::from_secs(300), "no receipt");
        std::thread::sleep(Duration::from_millis(200));
    }
    let pem = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")).unwrap();
    let tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(pem)).domain_name("cozy-worker");
    let endpoint = Endpoint::from_shared(format!("https://127.0.0.1:{port}")).unwrap().tls_config(tls).unwrap();
    (machine, MachineClient::new(endpoint.connect().await.unwrap()), root)
}

fn cap() -> String {
    let expires = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64 + 600;
    let grant = Grant { machine: WORKER.into(), action: capability::MACHINE.into(), expires, ..Default::default() };
    capability::mint(&SigningKey::from_bytes(&OWNER), grant)
}

/// `cozy model download acme/model@1.0.0 bf16` on `client`: its model row.
async fn download(client: &mut MachineClient<tonic::transport::Channel>, id: &str, hub: &str) -> serde_json::Value {
    let spec = v1::RunSpec {
        kind: v1::RunKind::Warm as i32,
        models: vec![v1::ModelChoice {
            parameter: "model".into(),
            repository: "acme/model".into(),
            release: "1.0.0".into(),
            lane: "bf16".into(),
            ..Default::default()
        }],
        hub: Some(v1::HubAccess { origin: hub.into(), object_hosts: vec!["localhost".into()], ..Default::default() }),
        catalog_revision: "r1".into(),
        owner: "alice".into(),
        ..Default::default()
    };
    let mut request = tonic::Request::new(v1::RunRequest { id: id.into(), after: 0, spec: Some(spec) });
    let header = MetadataValue::try_from(format!("Cozy-Cap {}", cap())).unwrap();
    request.metadata_mut().insert("authorization", header);
    let mut events = client.run(request).await.unwrap().into_inner();
    let mut last = None;
    while let Some(event) = events.message().await.unwrap() {
        last = event.event;
    }
    let Some(v1::run_event::Event::Outcome(outcome)) = last else { panic!("no outcome: {last:?}") };
    assert_eq!(outcome.status, "succeeded", "{outcome:?}");
    let result: serde_json::Value = serde_json::from_slice(&outcome.result).unwrap();
    result["models"][0].clone()
}

/// Files under the volume's `objects/`.
fn on_volume(cache: &Path) -> usize {
    let dirs = |dir: &Path| std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).collect::<Vec<_>>();
    dirs(&cache.join("objects")).iter().flat_map(|a| dirs(a)).map(|b| dirs(&b).len()).sum()
}

#[tokio::test]
async fn a_second_machine_downloads_from_the_volume_the_first_filled() {
    let source = std::env::temp_dir().join(format!("tensord-volume-source-{}", uuid::Uuid::new_v4()));
    let store = tensorfs_core::store::Store::ensure(&source).unwrap();
    let manifest = hub_oauth::checkpoint(&store, &vec![5; 1 << 20]);
    let objects = store.objects().unwrap().len() + 1;
    let model = names_hub::Model {
        repository: "acme/model".into(),
        release: "1.0.0".into(),
        lane: "bf16".into(),
        manifest,
    };
    let (hub, heard) = names_hub::serve(&store, model, Vec::<names_hub::Package>::new());
    // A mount point, as the Hub's: a tmpfs directory named from a disk directory.
    let tmpfs = PathBuf::from(format!("/dev/shm/tensord-volume-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&tmpfs).unwrap();
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("target/volume-{}", uuid::Uuid::new_v4()));
    std::os::unix::fs::symlink(&tmpfs, &cache).unwrap();

    let (a, mut client, a_root) = start("a", Some(&cache)).await;
    let first = download(&mut client, "volume-a", &hub).await;
    let total = first["bytes"]["hub"].as_u64().unwrap();
    assert!(total > 1 << 20 && first["bytes"]["volume"] == 0, "{first}");
    let began = Instant::now();
    while on_volume(&cache) < objects {
        assert!(began.elapsed() < Duration::from_secs(60), "A never filled the volume");
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(a);
    heard.lock().unwrap().clear();

    let (b, mut client, b_root) = start("b", Some(&cache)).await;
    let second = download(&mut client, "volume-b", &hub).await;
    assert_eq!((&second["bytes"]["hub"], &second["bytes"]["volume"]), (&0.into(), &total.into()), "{second}");
    let asked = std::mem::take(&mut *heard.lock().unwrap());
    assert!(asked.iter().all(|heard| heard.starts_with("POST /v1/tensorfs/closure")), "{asked:?}");
    let log = std::fs::read_to_string(b_root.join("var/lib/tensorfs/logs/transport.log")).unwrap();
    assert!(log.contains(&format!("hub 0 B volume {total} B")), "{log}");
    drop(b);

    // No variable, no volume: the Hub, as before.
    let (c, mut client, c_root) = start("c", None).await;
    let third = download(&mut client, "volume-c", &hub).await;
    assert_eq!((&third["bytes"]["hub"], &third["bytes"]["volume"]), (&total.into(), &0.into()), "{third}");
    drop(c);
    for dir in [a_root, b_root, c_root, tmpfs, source] {
        let _ = std::fs::remove_dir_all(dir);
    }
    let _ = std::fs::remove_file(cache);
}
