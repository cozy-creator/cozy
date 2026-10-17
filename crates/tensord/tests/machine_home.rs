//! This computer's machine (`--persistent`) from nothing but its user's home: its root, port,
//! keys and log come from `~/.cozy`. A key `cozy machine authorize` appends is admitted at the
//! next call, a restart keeps the port, and the machine's own log reads back over Read.
mod common;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::{pkcs8::EncodePrivateKey, SigningKey};
use std::{
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tensord::api::{
    capability::{self, Grant},
    v1::{self, machine_client::MachineClient},
};
use tonic::{
    metadata::MetadataValue,
    transport::{Certificate, Channel, ClientTlsConfig, Endpoint},
    Code, Request,
};

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn boot(home: &Path) -> Machine {
    Machine(
        Command::new(env!("CARGO_BIN_EXE_tensord"))
            .arg("--persistent")
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", home)
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}

/// The endpoint the machine published once its API listens.
fn ready(machine: &mut Machine, home: &Path) -> serde_json::Value {
    let path = home.join(".cozy/machine/var/lib/cozy/machine/api-ready.json");
    let start = Instant::now();
    loop {
        assert!(machine.0.try_wait().unwrap().is_none(), "the machine exited");
        if let Ok(bytes) = std::fs::read(&path) {
            return serde_json::from_slice(&bytes).unwrap();
        }
        assert!(start.elapsed() < Duration::from_secs(300), "no api-ready.json");
        std::thread::sleep(Duration::from_millis(100));
    }
}

async fn client(ready: &serde_json::Value) -> MachineClient<Channel> {
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
    MachineClient::new(channel)
}

fn authorized<T>(message: T, key: &SigningKey) -> Request<T> {
    let expires = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64 + 600;
    let grant = Grant { machine: "local".into(), action: capability::MACHINE.into(), expires, ..Default::default() };
    let mut request = Request::new(message);
    let value = format!("Cozy-Cap {}", capability::mint(key, grant));
    request.metadata_mut().insert("authorization", MetadataValue::try_from(value).unwrap());
    request
}

/// A machine cap's Status: the first frame, or the refusal.
async fn status(client: &mut MachineClient<Channel>, key: &SigningKey) -> Result<v1::StatusFrame, tonic::Status> {
    let mut frames = client.status(authorized(v1::StatusRequest { keepalive: false }, key)).await?.into_inner();
    Ok(frames.message().await?.expect("a frame"))
}

#[tokio::test]
async fn this_computers_machine_runs_from_its_home() {
    let home = tempfile::tempdir().unwrap();
    let keys = home.path().join(".cozy/keys");
    std::fs::create_dir_all(&keys).unwrap();
    let owner = SigningKey::from_bytes(&[5; 32]);
    std::fs::write(keys.join("owner.pem"), owner.to_pkcs8_pem(Default::default()).unwrap().as_bytes()).unwrap();

    let mut machine = boot(home.path());
    let first = ready(&mut machine, home.path());
    assert_eq!(first["worker_id"], "local");
    assert!(first["address"].as_str().unwrap().starts_with("127.0.0.1:"), "{first}");
    let mut client = client(&first).await;

    // The owner key Calcifer signs with is admitted; another is not, until it is authorized.
    let frame = status(&mut client, &owner).await.unwrap();
    assert_eq!(frame.worker_id, "local");
    let other = SigningKey::from_bytes(&[6; 32]);
    assert_eq!(status(&mut client, &other).await.unwrap_err().code(), Code::Unauthenticated);
    let line = URL_SAFE_NO_PAD.encode(other.verifying_key().as_bytes());
    std::fs::write(keys.join("authorized"), format!("# a laptop\n{line}\n")).unwrap();
    assert_eq!(status(&mut client, &other).await.unwrap().worker_id, "local");

    // Its own log rolls beside the store's and reads back by name.
    let log = v1::ReadRequest { target: Some(v1::read_request::Target::Log("tensord".into())), ..Default::default() };
    let mut frames = client.read(authorized(log, &owner)).await.unwrap().into_inner();
    let mut text = vec![];
    while let Some(frame) = frames.message().await.unwrap() {
        text.extend(frame.data);
    }
    let text = String::from_utf8(text).unwrap();
    assert!(text.contains("readiness sealed"), "{text}");
    let logs = home.path().join(".cozy/machine/var/lib/tensorfs/logs");
    assert!(std::fs::read_dir(&logs).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with("tensord.")));

    // A restart keeps the port Calcifer recorded.
    drop(client);
    drop(machine);
    std::fs::remove_file(home.path().join(".cozy/machine/var/lib/cozy/machine/api-ready.json")).unwrap();
    let mut machine = boot(home.path());
    let second = ready(&mut machine, home.path());
    assert_eq!(second["address"], first["address"]);
    assert_eq!(second["cert_pem"], first["cert_pem"], "the pinned leaf survives a restart");
}
