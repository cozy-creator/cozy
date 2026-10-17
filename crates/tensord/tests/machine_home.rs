//! This computer's machine (`--persistent`) from nothing but its user's home: its root, port,
//! keys and log come from `~/.cozy`. A key `cozy machine authorize` appends is admitted at the
//! next call, the machine's own log reads back by name, local code installs on a root with no
//! uv (the pinned one is fetched), and a restart keeps the port.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
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
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}

/// The endpoint the machine published once its API listens.
fn ready(machine: &mut Machine, home: &Path) -> serde_json::Value {
    let path = home.join(".cozy/machine/var/lib/cozy/tensord/api-ready.json");
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

/// A client of the published endpoint, pinned to its leaf, signing with `key`.
fn client(ready: &serde_json::Value, key: &SigningKey) -> Client {
    let pem = ready["cert_pem"].as_str().unwrap();
    let leaf = rustls_pemfile::certs(&mut pem.as_bytes()).next().unwrap().unwrap();
    let auth = Auth::Key { key: Arc::new(key.clone()), worker: "local".into() };
    Client::new(ready["address"].as_str().unwrap(), tensorfs_core::sha256::digest(&leaf), auth).unwrap()
}

/// `bytes` written to the machine as one object: its digest.
async fn write(client: &Client, bytes: &[u8]) -> String {
    let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(bytes));
    let length = bytes.len() as u64;
    let held = client.bulk().unwrap().put(&digest, (0, length, length), bytes.to_vec()).await.unwrap();
    assert_eq!(held, [(0, length)]);
    digest
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
    let client = client(&first, &owner);

    // The owner key Calcifer signs with is admitted; another is not, until it is authorized.
    let frame = client.status().await.unwrap();
    assert_eq!(frame.worker_id, "local");
    // Every GPU names its tensor-core family and memory class; a computer with no NVIDIA driver
    // (a hosted runner) has none to name.
    if !Path::new("/dev/nvidiactl").exists() {
        assert!(frame.gpus.is_empty(), "{:?}", frame.gpus);
    }
    for gpu in &frame.gpus {
        assert!(gpu.family.starts_with("sm"), "{gpu:?}");
        assert!(["discrete", "coherent", "unified"].contains(&gpu.memory_class.as_str()), "{gpu:?}");
    }
    let other = self::client(&first, &SigningKey::from_bytes(&[6; 32]));
    assert_eq!(other.status().await.unwrap_err().status(), Some(401));
    let line = URL_SAFE_NO_PAD.encode(SigningKey::from_bytes(&[6; 32]).verifying_key().as_bytes());
    std::fs::write(keys.join("authorized"), format!("# a laptop\n{line}\n")).unwrap();
    assert_eq!(other.status().await.unwrap().worker_id, "local");

    // Its own log rolls in its state directory and reads back by name.
    let text = String::from_utf8(client.log("tensord", 0).await.unwrap().to_vec()).unwrap();
    assert!(text.contains("readiness sealed"), "{text}");
    let logs = home.path().join(".cozy/machine/var/lib/cozy/machine/logs");
    assert!(std::fs::read_dir(&logs).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with("tensord.")));

    // Local code on a fresh root: the pinned uv and the installer helper are made at this first
    // install, and the warm run returns the code's interface.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_memo");
    let mut files = vec![];
    for name in ["pyproject.toml", "package.toml", "cpu_memo/__init__.py"] {
        let bytes = std::fs::read(fixture.join(name)).unwrap();
        let digest = write(&client, &bytes).await;
        files.push(serde_json::json!({"name": name, "digest": digest, "length": bytes.len()}));
    }
    let manifest = serde_json::json!({"package": "local/tensord-cpu-memo", "release": "0.1.0", "files": files});
    let manifest = write(&client, manifest.to_string().as_bytes()).await;
    let spec = v1::RunSpec {
        kind: v1::RunKind::Warm,
        source: Some(v1::Source::Local(v1::LocalSource { manifest })),
        ..Default::default()
    };
    client.submit(&v1::RunRequest { id: "local".into(), spec }).await.unwrap();
    let events: Vec<_> = tokio_stream::StreamExt::collect(client.events("local", 0).await.unwrap()).await;
    let outcome = match events.last() {
        Some(Ok(v1::RunEvent { event: v1::Event::Outcome(done), .. })) => done.clone(),
        last => panic!("no outcome: {last:?}"),
    };
    assert_eq!(outcome.status, "succeeded", "{outcome:?}");
    let result: serde_json::Value = serde_json::from_str(outcome.result.as_ref().unwrap().get()).unwrap();
    assert!(result["interface"]["entrypoints"].as_array().unwrap().iter().any(|e| e["name"] == "measure"), "{result}");
    let uv = home.path().join(".cozy/machine/usr/local/bin/uv");
    let version = Command::new(&uv).arg("--version").output().unwrap();
    assert!(String::from_utf8_lossy(&version.stdout).starts_with(&format!("uv {}", tensord::machine::client::UV)));

    // A restart keeps the port Calcifer recorded.
    drop(client);
    drop(machine);
    std::fs::remove_file(home.path().join(".cozy/machine/var/lib/cozy/tensord/api-ready.json")).unwrap();
    let mut machine = boot(home.path());
    let second = ready(&mut machine, home.path());
    assert_eq!(second["address"], first["address"]);
    assert_eq!(second["cert_pem"], first["cert_pem"], "the pinned leaf survives a restart");
}

/// What is held of 1,000 objects is one request: each stored one whole, a partial upload its
/// ranges, an unknown one nothing, in the order asked; a digest that is not one is refused.
#[tokio::test]
async fn a_thousand_objects_answer_what_is_held_in_one_request() {
    let home = tempfile::tempdir().unwrap();
    let keys = home.path().join(".cozy/keys");
    std::fs::create_dir_all(&keys).unwrap();
    let owner = SigningKey::from_bytes(&[5; 32]);
    std::fs::write(keys.join("owner.pem"), owner.to_pkcs8_pem(Default::default()).unwrap().as_bytes()).unwrap();
    let mut machine = boot(home.path());
    let client = client(&ready(&mut machine, home.path()), &owner);

    let mut asked = vec![];
    for i in 0..998u32 {
        asked.push(write(&client, format!("object {i}").as_bytes()).await);
    }
    let partial = b"0123456789".to_vec();
    let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&partial));
    client.bulk().unwrap().put(&digest, (6, 4, 10), partial[6..].to_vec()).await.unwrap();
    asked.push(digest);
    asked.push(format!("sha256:{}", "0".repeat(64)));

    let started = Instant::now();
    let held = client.held_many(&asked).await.unwrap();
    eprintln!("1,000 objects asked in {:?}", started.elapsed());
    let digests: Vec<&str> = held.iter().map(|h| h.digest.as_str()).collect();
    assert_eq!(digests, asked.iter().map(String::as_str).collect::<Vec<_>>());
    for (i, object) in held[..998].iter().enumerate() {
        assert_eq!(object.ranges, [(0, format!("object {i}").len() as u64)], "{object:?}");
    }
    assert_eq!(held[998].ranges, [(6, 10)]);
    assert_eq!(held[999].ranges, []);

    let refused = client.held_many(&["sha256:nope".into()]).await.unwrap_err();
    assert_eq!(refused.status(), Some(400), "{refused:?}");
}
