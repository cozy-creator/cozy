//! Sender-constrained access (RFC 9449 DPoP) in the one profile AuthKit accepts: ES256 over
//! P-256. A DPoP-bound token is worthless without its key: every request carries
//! `Authorization: DPoP <token>` beside a fresh proof that signs the method, the URL and the
//! token's hash, so a token lifted off the wire or out of a log replays nowhere.
//!
//! The holder of the token owns its life (redemption, refresh, rotation); this module only
//! presents it, keeps each host's server nonce, and reports a refusal back to the holder.

use crate::err::{refuse, Code, Result};
use crate::transport::credential::{Answer, Ask, CredentialProvider};
use base64::{engine::general_purpose::STANDARD, engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{EcdsaKeyPair, KeyPair, ECDSA_P256_SHA256_FIXED_SIGNING};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// A P-256 key that signs DPoP proofs.
pub struct DpopKey {
    pair: EcdsaKeyPair,
    rng: SystemRandom,
    x: String,
    y: String,
}

impl DpopKey {
    /// An unencrypted PKCS#8 P-256 key (DER).
    pub fn from_pkcs8(der: &[u8]) -> Result<DpopKey> {
        let rng = SystemRandom::new();
        let pair = EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, der, &rng)
            .map_err(|_| unusable("not an unencrypted PKCS#8 P-256 key"))?;
        let point = pair.public_key().as_ref();
        if point.len() != 65 || point[0] != 4 {
            return refuse(
                Code::CREDENTIAL_REQUIRED,
                "a DPoP key's public point is not uncompressed P-256",
            );
        }
        let (x, y) = (
            URL_SAFE_NO_PAD.encode(&point[1..33]),
            URL_SAFE_NO_PAD.encode(&point[33..]),
        );
        Ok(DpopKey { pair, rng, x, y })
    }

    /// The same key in PEM (`-----BEGIN PRIVATE KEY-----`).
    pub fn from_pem(pem: &str) -> Result<DpopKey> {
        let body = pem
            .split("-----BEGIN PRIVATE KEY-----")
            .nth(1)
            .and_then(|rest| rest.split("-----END PRIVATE KEY-----").next())
            .ok_or_else(|| unusable("no PKCS#8 `PRIVATE KEY` block"))?;
        let der = STANDARD
            .decode(body.split_whitespace().collect::<String>())
            .map_err(|_| unusable("the PRIVATE KEY block is not base64"))?;
        DpopKey::from_pkcs8(&der)
    }

    /// A fresh key, held only in memory.
    pub fn generate() -> Result<DpopKey> {
        let rng = SystemRandom::new();
        let der = EcdsaKeyPair::generate_pkcs8(&ECDSA_P256_SHA256_FIXED_SIGNING, &rng)
            .map_err(|_| unusable("the system random source failed"))?;
        DpopKey::from_pkcs8(der.as_ref())
    }

    /// The public key's RFC 7638 thumbprint, unpadded base64url: an authorization request's
    /// `dpop_jkt` and a bound token's `cnf.jkt`.
    pub fn thumbprint(&self) -> String {
        let canonical = format!(
            r#"{{"crv":"P-256","kty":"EC","x":"{}","y":"{}"}}"#,
            self.x, self.y
        );
        URL_SAFE_NO_PAD.encode(crate::sha256::digest(canonical.as_bytes()))
    }

    /// One proof for `method htu`. A resource request names its `access_token` (the proof
    /// carries its hash, `ath`); a token request names none. `nonce` is the server's latest.
    pub fn proof(
        &self,
        method: &str,
        htu: &str,
        access_token: Option<&str>,
        nonce: Option<&str>,
    ) -> Result<String> {
        let mut jti = [0u8; 24];
        self.rng
            .fill(&mut jti)
            .map_err(|_| unusable("the system random source failed"))?;
        let iat = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs());
        let header = serde_json::json!({
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": {"kty": "EC", "crv": "P-256", "x": self.x, "y": self.y},
        });
        let mut claims = serde_json::json!({
            "jti": URL_SAFE_NO_PAD.encode(jti),
            "htm": method,
            "htu": htu,
            "iat": iat,
        });
        if let Some(token) = access_token {
            claims["ath"] = URL_SAFE_NO_PAD
                .encode(crate::sha256::digest(token.as_bytes()))
                .into();
        }
        if let Some(nonce) = nonce {
            claims["nonce"] = nonce.into();
        }
        let input = format!(
            "{}.{}",
            URL_SAFE_NO_PAD.encode(header.to_string()),
            URL_SAFE_NO_PAD.encode(claims.to_string())
        );
        let signature = self
            .pair
            .sign(&self.rng, input.as_bytes())
            .map_err(|_| unusable("signing the proof failed"))?;
        Ok(format!(
            "{input}.{}",
            URL_SAFE_NO_PAD.encode(signature.as_ref())
        ))
    }
}

fn unusable(why: &str) -> crate::err::Refusal {
    crate::err::Refusal {
        code: Code::CREDENTIAL_REQUIRED,
        detail: format!("DPoP key: {why}"),
    }
}

/// Where a DPoP credential's access token comes from. The holder keeps it fresh.
pub trait AccessToken: Sync {
    /// The token to present now. `None` presents the proof alone: what a token endpoint
    /// wants, and at a resource server a request it answers with its own refusal.
    fn current(&self) -> Option<String>;
    /// A resource server refused `token`; the next `current` should not answer it.
    fn refused(&self, _token: &str) {}
}

/// No access token: proofs for a token endpoint.
impl AccessToken for () {
    fn current(&self) -> Option<String> {
        None
    }
}

impl<T: AccessToken + ?Sized> AccessToken for &T {
    fn current(&self) -> Option<String> {
        (**self).current()
    }
    fn refused(&self, token: &str) {
        (**self).refused(token)
    }
}

impl<T: AccessToken + Send + ?Sized> AccessToken for Arc<T> {
    fn current(&self) -> Option<String> {
        (**self).current()
    }
    fn refused(&self, token: &str) {
        (**self).refused(token)
    }
}

/// A DPoP-bound token from `tokens`, presented only to `hosts`, with a fresh proof per
/// request. A `401` is answered by one retry: with the server's nonce when it asked for
/// one (`use_dpop_nonce`), else after reporting the token refused so its holder refreshes.
/// A token endpoint asks for a nonce with a `400`, which is retried the same way.
pub struct DpopCredential<T: AccessToken> {
    key: Arc<DpopKey>,
    tokens: T,
    hosts: Vec<String>,
    origin: Option<String>,
    nonces: Mutex<HashMap<String, String>>,
}

impl<T: AccessToken> DpopCredential<T> {
    pub fn new(key: Arc<DpopKey>, tokens: T, hosts: Vec<String>) -> Self {
        DpopCredential {
            key,
            tokens,
            hosts,
            origin: None,
            nonces: Mutex::new(HashMap::new()),
        }
    }

    /// Proofs name this origin (`scheme://host[:port]`) instead of the one requests reach:
    /// a resource server's public origin, reached here by another route (a loopback hop).
    pub fn proving_origin(mut self, origin: &str) -> Self {
        self.origin = Some(origin.trim_end_matches('/').to_string());
        self
    }

    pub fn tokens(&self) -> &T {
        &self.tokens
    }

    fn names(&self, ask: &Ask) -> bool {
        self.hosts
            .iter()
            .any(|h| h.eq_ignore_ascii_case(&ask.url.host))
    }

    fn htu(&self, ask: &Ask) -> String {
        let htu = ask.htu();
        match &self.origin {
            None => htu,
            Some(origin) => {
                let path = htu
                    .find("://")
                    .and_then(|at| htu[at + 3..].find('/').map(|p| at + 3 + p));
                format!("{origin}{}", path.map_or("/", |p| &htu[p..]))
            }
        }
    }
}

impl<T: AccessToken> CredentialProvider for DpopCredential<T> {
    fn headers(&self, ask: &Ask) -> Vec<(String, String)> {
        if !self.names(ask) {
            return Vec::new();
        }
        let token = self.tokens.current();
        let nonce = self
            .nonces
            .lock()
            .unwrap()
            .get(&ask.url.host.to_ascii_lowercase())
            .cloned();
        let Ok(proof) = self.key.proof(
            ask.method,
            &self.htu(ask),
            token.as_deref(),
            nonce.as_deref(),
        ) else {
            // No proof, no credential: the server's own refusal says so.
            return Vec::new();
        };
        let mut headers = Vec::with_capacity(2);
        if let Some(token) = token {
            headers.push(("authorization".to_string(), format!("DPoP {token}")));
        }
        headers.push(("dpop".to_string(), proof));
        headers
    }

    fn answered(&self, ask: &Ask, answer: &Answer) -> bool {
        let Some(proof) = answer.presented("dpop").filter(|_| self.names(ask)) else {
            return false;
        };
        let fresh = answer
            .header("dpop-nonce")
            .map(str::trim)
            .filter(|n| !n.is_empty());
        if let Some(nonce) = fresh {
            self.nonces
                .lock()
                .unwrap()
                .insert(ask.url.host.to_ascii_lowercase(), nonce.to_string());
        }
        // A nonce is worth a retry only when the refused proof did not already carry it.
        let new_nonce = fresh.is_some() && fresh != proof_nonce(proof).as_deref();
        let token = answer
            .presented("authorization")
            .and_then(|v| v.strip_prefix("DPoP "));
        let challenged = answer
            .header("www-authenticate")
            .is_some_and(|value| value.contains("use_dpop_nonce"));
        match (answer.status, token) {
            (401, _) if challenged => new_nonce,
            (401, Some(token)) => {
                self.tokens.refused(token);
                true
            }
            (400 | 401, _) => new_nonce,
            _ => false,
        }
    }
}

/// The `nonce` a proof carried.
fn proof_nonce(proof: &str) -> Option<String> {
    let claims = URL_SAFE_NO_PAD.decode(proof.split('.').nth(1)?).ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&claims).ok()?;
    claims["nonce"].as_str().map(str::to_string)
}
