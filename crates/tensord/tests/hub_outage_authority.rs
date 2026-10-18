//! A rental keeps the Hub's last answer about its keys. A Hub it cannot reach for longer than
//! the lease it named revokes nothing: the owner's open stream stays open and a new call is
//! admitted. The Hub's own answer without the key ends that stream at once. The real binary,
//! and a Hub over real HTTPS.
//! A rental is a Linux pod.
#![cfg(target_os = "linux")]
mod common;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ed25519_dalek::SigningKey;
use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_rustls::rustls;
use tokio_stream::StreamExt;

const WORKER: &str = "outage-test";
const OWNER: [u8; 32] = [41; 32];
const OTHER: [u8; 32] = [42; 32];

struct Machine(Child);
impl Drop for Machine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// The Hub's rental authority: while up it answers `keys` with a one-second lease; while down
/// it drops every connection, as an unreachable Hub does.
struct Hub {
    up: AtomicBool,
    keys: Mutex<Vec<[u8; 32]>>,
    answered: AtomicUsize,
}

async fn serve_hub(hub: Arc<Hub>) -> (u16, Vec<u8>) {
    let key = rcgen::KeyPair::generate().unwrap();
    let cert = rcgen::CertificateParams::new(vec!["localhost".into()])
        .unwrap()
        .self_signed(&key)
        .unwrap();
    let der = cert.der().to_vec();
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::aws_lc_rs::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(
        vec![der.clone().into()],
        rustls::pki_types::PrivateKeyDer::Pkcs8(key.serialize_der().into()),
    )
    .unwrap();
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        while let Ok((tcp, _)) = listener.accept().await {
            if !hub.up.load(Ordering::Acquire) {
                continue; // dropped: the machine's call fails as weather
            }
            let (acceptor, hub) = (acceptor.clone(), hub.clone());
            tokio::spawn(async move {
                let Ok(mut tls) = acceptor.accept(tcp).await else {
                    return;
                };
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") && tls.read(&mut byte).await.unwrap_or(0) == 1 {
                    head.push(byte[0]);
                }
                let keys: Vec<String> = hub
                    .keys
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|key| {
                        URL_SAFE_NO_PAD
                            .encode(SigningKey::from_bytes(key).verifying_key().as_bytes())
                    })
                    .collect();
                let body = serde_json::json!({"worker_id": WORKER, "authorized_keys": keys, "lease_seconds": 1}).to_string();
                let answer = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len());
                if tls.write_all(answer.as_bytes()).await.is_ok() {
                    let _ = tls.shutdown().await;
                    hub.answered.fetch_add(1, Ordering::AcqRel);
                }
            });
        }
    });
    (port, der)
}

async fn until(what: &str, mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < Duration::from_secs(300), "{what}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn an_unreachable_hub_revokes_nothing_and_its_answer_revokes_at_once() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/hub-outage-authority")
        .join(std::process::id().to_string());
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let hub = Arc::new(Hub {
        up: AtomicBool::new(true),
        keys: Mutex::new(vec![OWNER]),
        answered: AtomicUsize::new(0),
    });
    let (hub_port, ca) = serve_hub(hub.clone()).await;
    let free = || {
        std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    };
    let (port, webrtc) = (free(), free());
    let owner = SigningKey::from_bytes(&OWNER).verifying_key();
    let mut machine = Machine(
        Command::new(env!("CARGO_BIN_EXE_tensord"))
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("COZY_MACHINE_ROOT", &root)
            .env("HOME", root.join("home"))
            .arg("--rental")
            .env("COZY_WORKER_ID", WORKER)
            .env("COZY_WORKER_AUTH_TOKEN", URL_SAFE_NO_PAD.encode([1; 32]))
            .env("COZY_WORKER_INTERNAL_PORT", port.to_string())
            .env("COZY_LISTEN_HOST", "127.0.0.1")
            .env(
                "COZY_AUTHORIZED_KEYS",
                URL_SAFE_NO_PAD.encode(owner.as_bytes()),
            )
            .env(
                "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL",
                URL_SAFE_NO_PAD.encode([7; 32]),
            )
            .env("TENSORHUB_ORIGIN", format!("https://localhost:{hub_port}"))
            .env("TENSORHUB_CA_DER_B64URL", URL_SAFE_NO_PAD.encode(&ca))
            .env("COZY_WEBRTC_INTERNAL_PORT", webrtc.to_string())
            .env("RUNPOD_PUBLIC_IP", "203.0.113.7")
            .env(format!("RUNPOD_TCP_PORT_{webrtc}"), "30001")
            .env("CUDA_VISIBLE_DEVICES", "")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    while common::receipt(&root, port).is_none() {
        assert!(
            machine.0.try_wait().unwrap().is_none(),
            "the machine exited"
        );
        assert!(start.elapsed() < Duration::from_secs(300), "no receipt");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    // The Hub's answer, not the boot grant, now admits the owner.
    until("the machine never asked the Hub", || {
        hub.answered.load(Ordering::Acquire) >= 2
    })
    .await;
    let client = common::client(&root, port, common::signed(OWNER, WORKER));
    let mut frames = Box::pin(client.watch().await.unwrap());
    assert_eq!(frames.next().await.unwrap().unwrap().worker_id, WORKER);

    // Down for three leases: the owner's stream stays open and a new call is admitted.
    hub.up.store(false, Ordering::Release);
    let outage = Instant::now();
    while outage.elapsed() < Duration::from_secs(3) {
        if let Ok(next) = tokio::time::timeout(Duration::from_millis(250), frames.next()).await {
            let frame = next.expect("an unreachable Hub ended the stream");
            frame.unwrap_or_else(|refused| panic!("an unreachable Hub ended the stream: {refused}"));
        }
    }
    let during = client.status().await.expect("a new call during the outage");
    assert_eq!(during.worker_id, WORKER);

    // The Hub answers again without the owner's key: its streams end, its calls are refused.
    *hub.keys.lock().unwrap() = vec![OTHER];
    hub.up.store(true, Ordering::Release);
    tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(Ok(_)) = frames.next().await {}
    })
    .await
    .expect("the Hub's revocation did not end the stream");
    assert_eq!(client.status().await.unwrap_err().status(), Some(401));
    let other = client.with_auth(common::signed(OTHER, WORKER)).status().await.unwrap();
    assert_eq!(other.worker_id, WORKER);
    let _ = std::fs::remove_dir_all(&root);
}
