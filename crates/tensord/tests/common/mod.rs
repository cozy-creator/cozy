//! What every readiness test asks a machine root's listener, with the machine API's client.
#![allow(dead_code)]
use std::path::Path;
use tensord::api::v1::{
    self,
    client::{Auth, Machine},
};

/// The pin of the leaf a machine root's listener presents.
pub fn pin(root: &Path) -> Option<[u8; 32]> {
    let pem = std::fs::read(root.join("run/cozy/bootstrap/tls.crt")).ok()?;
    let leaf = rustls_pemfile::certs(&mut pem.as_slice()).next()?.ok()?;
    Some(tensorfs::sha256::digest(&leaf))
}

/// A client of the machine root's listener on `port`, authorized by `auth`.
pub fn client(root: &Path, port: u16, auth: Auth) -> Machine {
    Machine::new(&format!("127.0.0.1:{port}"), pin(root).expect("the machine's leaf"), auth).unwrap()
}

/// The sealed receipt `GET /v1/status` carries with no capability, as the Hub reads it; None
/// until the listener answers and the boot is sealed. Runs on its own thread, so sync and async
/// tests both call it.
pub fn receipt(root: &Path, port: u16) -> Option<Vec<u8>> {
    let pin = pin(root)?;
    let read = move || -> Option<Vec<u8>> {
        tokio::runtime::Runtime::new().ok()?.block_on(async {
            let machine = Machine::new(&format!("127.0.0.1:{port}"), pin, Auth::None).ok()?;
            let frame = machine.status().await.ok()?;
            (!frame.receipt.is_empty()).then_some(frame.receipt)
        })
    };
    std::thread::spawn(read).join().ok()?
}

/// A fresh machine cap from `key` on every request.
pub fn signed(key: [u8; 32], worker: &str) -> Auth {
    Auth::Key { key: std::sync::Arc::new(ed25519_dalek::SigningKey::from_bytes(&key)), worker: worker.into() }
}

/// Submits run `id` and follows its events to the end.
pub async fn run(client: &Machine, id: &str, spec: v1::RunSpec) -> Vec<v1::RunEvent> {
    client.submit(&v1::RunRequest { id: id.into(), spec }).await.unwrap();
    events(client, id, 0).await
}

/// Run `id`'s events after `after`, to the end of its stream.
pub async fn events(client: &Machine, id: &str, after: u64) -> Vec<v1::RunEvent> {
    use tokio_stream::StreamExt;
    let events = client.events(id, after).await.unwrap();
    events.map(Result::unwrap).collect().await
}

/// The outcome that ends `events`.
pub fn outcome(events: &[v1::RunEvent]) -> v1::Outcome {
    match events.last().map(|e| &e.event) {
        Some(v1::Event::Outcome(outcome)) => outcome.clone(),
        last => panic!("no outcome: {last:?}"),
    }
}
