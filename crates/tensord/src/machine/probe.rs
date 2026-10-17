//! The two listener observations the Hub's readiness reader requires, made against the real
//! listener over loopback with the machine API's own client: it presents exactly this machine's
//! leaf, and it refuses a foreign credential.
use cozy_proto::machine::client::{presented_leaf, Auth, Machine};
use std::sync::Arc;

fn address(port: u16) -> String {
    format!("127.0.0.1:{port}")
}

/// True when a TLS handshake on the listener presents exactly `leaf`.
pub async fn presents_leaf(port: u16, leaf: &[u8]) -> bool {
    presented_leaf(&address(port)).await.is_ok_and(|presented| presented == leaf)
}

/// True when the listener refuses a call whose Cozy-Cap is well formed, names this machine and
/// is unexpired, but is signed by a key made for this one probe.
pub async fn refuses_foreign_capability(port: u16, leaf: &[u8], worker_id: &str) -> bool {
    match super::identity::random::<32>() {
        Ok(seed) => {
            let foreign = ed25519_dalek::SigningKey::from_bytes(&seed);
            refuses_capability_of(port, leaf, worker_id, &foreign).await
        }
        Err(_) => false,
    }
}

/// True when `GET /v1/status` carrying a machine-scope Cozy-Cap signed by `key` is refused as a
/// credential while the same call with no capability answers: the refusal is the credential's,
/// not a closed door.
pub async fn refuses_capability_of(
    port: u16,
    leaf: &[u8],
    worker_id: &str,
    key: &ed25519_dalek::SigningKey,
) -> bool {
    let pin = tensorfs_core::sha256::digest(leaf);
    let Ok(machine) = Machine::new(&address(port), pin, Auth::None) else {
        return false;
    };
    let signed = Auth::Key { key: Arc::new(key.clone()), worker: worker_id.into() };
    let open = machine.status().await;
    let refused = machine.with_auth(signed).status().await;
    open.is_ok() && refused.is_err_and(|error| error.status() == Some(401))
}
