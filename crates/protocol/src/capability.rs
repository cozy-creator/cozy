//! Cozy-Cap: the signed grant a client presents to one machine (`authorization: Cozy-Cap <token>`,
//! or a media session's hello). An authorized key permits reading a run's outputs, or every
//! `cozy.machine.v1` call, until an expiry. Machines verify offline and store no bearer.
//!
//! Token: `base64url(payload) "." base64url(ed25519(DOMAIN || payload))`, payload the grant's
//! JSON. Byte-identical with the Go client's former `internal/capability`.
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use ed25519_dalek::{Signature, Signer as _, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

const DOMAIN: &[u8] = b"cozy-capability/1\0";

/// The scheme of the `authorization` metadata value.
pub const SCHEME: &str = "Cozy-Cap ";

/// The machine-wide scope: every `cozy.machine.v1` call.
pub const MACHINE: &str = "machine";

/// What a capability allows. A member this verifier does not know refuses the whole grant: a
/// restriction an older machine cannot read must not be dropped.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    /// The machine's worker id.
    #[serde(rename = "m")]
    pub machine: String,
    /// A run id on that machine, or empty for an action.
    #[serde(rename = "r", default, skip_serializing_if = "String::is_empty")]
    pub run: String,
    /// A machine-wide action ([`MACHINE`]), never with `run`.
    #[serde(rename = "a", default, skip_serializing_if = "String::is_empty")]
    pub action: String,
    /// `name` (every index) or `name/i`; empty: every output of `run`.
    #[serde(rename = "p", default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<String>,
    /// Unix seconds.
    #[serde(rename = "e")]
    pub expires: i64,
    /// The signer's [`key_id`]; [`mint`] fills it.
    #[serde(rename = "k")]
    pub key: String,
    /// The client's DTLS certificate (`sha-256 AB:…`), WebRTC only.
    #[serde(rename = "x", default, skip_serializing_if = "String::is_empty")]
    pub binding: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    #[error("the capability is malformed or not signed by an authorized key")]
    Invalid,
    #[error("the capability has expired")]
    Expired,
    #[error("the capability does not grant this")]
    Scope,
}

/// A public key's compact name: base64url of the first 16 bytes of its SHA-256.
pub fn key_id(key: &VerifyingKey) -> String {
    URL_SAFE_NO_PAD.encode(&Sha256::digest(key.as_bytes())[..16])
}

/// Signs `grant` with `key`.
pub fn mint(key: &SigningKey, mut grant: Grant) -> String {
    grant.key = key_id(&key.verifying_key());
    let payload = serde_json::to_vec(&grant).expect("a grant serializes");
    let signature = key.sign(&[DOMAIN, &payload].concat());
    format!(
        "{}.{}",
        URL_SAFE_NO_PAD.encode(&payload),
        URL_SAFE_NO_PAD.encode(signature.to_bytes())
    )
}

/// Admits `token` for `machine` at `now` (unix seconds), signed by one of `keys`, and answers
/// the grant with the key that signed it (the run's actor). `binding` is the client's DTLS
/// fingerprint on WebRTC and empty elsewhere: a grant naming a binding holds only there.
pub fn verify(
    token: &str,
    machine: &str,
    keys: &[VerifyingKey],
    now: i64,
    binding: &str,
) -> Result<(Grant, VerifyingKey), Refusal> {
    let (encoded, signature) = token.split_once('.').ok_or(Refusal::Invalid)?;
    let payload = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|_| Refusal::Invalid)?;
    let signature = URL_SAFE_NO_PAD
        .decode(signature)
        .ok()
        .and_then(|bytes| Signature::from_slice(&bytes).ok())
        .ok_or(Refusal::Invalid)?;
    let grant: Grant = serde_json::from_slice(&payload).map_err(|_| Refusal::Invalid)?;
    if grant.machine.is_empty()
        || grant.run.is_empty() == grant.action.is_empty()
        || grant.expires == 0
    {
        return Err(Refusal::Invalid);
    }
    let signed = [DOMAIN, &payload].concat();
    let signer = *keys
        .iter()
        .find(|key| key_id(key) == grant.key && key.verify_strict(&signed, &signature).is_ok())
        .ok_or(Refusal::Invalid)?;
    if grant.machine != machine || !grant.binding.is_empty() && grant.binding != binding {
        return Err(Refusal::Invalid);
    }
    if now >= grant.expires {
        return Err(Refusal::Expired);
    }
    Ok((grant, signer))
}

/// The token of an `authorization` value, when it carries one.
pub fn from_authorization(value: &str) -> Option<&str> {
    value.strip_prefix(SCHEME)
}

impl Grant {
    /// Whether the grant covers one output of a run; `index` is a list item's 1-based index.
    pub fn allows(&self, run: &str, output: &str, index: Option<u32>) -> bool {
        run == self.run
            && (self.outputs.is_empty()
                || self.outputs.iter().any(|granted| {
                    granted == output || index.is_some_and(|i| *granted == format!("{output}/{i}"))
                }))
    }
}
