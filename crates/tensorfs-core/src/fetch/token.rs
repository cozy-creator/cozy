//! What a request presents (tensorfs-api §1): a token provider is asked per request with its
//! method, URL, headers and payload digest, so a bearer token, DPoP proofs and S3 SigV4 all fit.
//! [`default_for`] picks the provider an endpoint gets when the caller gives none, by host.

use std::sync::Arc;

use async_trait::async_trait;

pub use hyper::HeaderMap;
use super::policy::base_host;
use super::tensorhub::TENSORHUB;
use crate::err::Result;

/// A request about to be sent.
pub struct Request<'a> {
    pub method: &'a str,
    pub url: &'a str,
    /// The headers it carries besides the provider's own (after a send: with them).
    pub headers: &'a [(String, String)],
    /// The hex SHA-256 of the body, when it has one and it is known.
    pub payload: Option<&'a str>,
}

#[async_trait]
pub trait TokenProvider: Send + Sync {
    /// The headers that authorize `request`.
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>>;

    /// The answer to `sent` was `status` with `headers`: true asks once more (a DPoP nonce, a
    /// refreshed token).
    fn retry(&self, sent: &Request<'_>, status: u16, headers: &HeaderMap) -> bool {
        let _ = (sent, status, headers);
        false
    }
}

/// `Authorization: Bearer <token>`.
pub struct Bearer(pub String);

#[async_trait]
impl TokenProvider for Bearer {
    async fn headers(&self, _: &Request<'_>) -> Result<Vec<(String, String)>> {
        Ok(vec![("authorization".into(), format!("Bearer {}", self.0))])
    }
}

/// A bearer provider for `token`.
pub fn bearer(token: impl Into<String>) -> Arc<dyn TokenProvider> {
    Arc::new(Bearer(token.into()))
}

/// The endpoint a name resolves against when none is given: `TENSORFS_ENDPOINT`, else Tensorhub.
pub fn default_endpoint() -> String {
    env("TENSORFS_ENDPOINT").unwrap_or_else(|| TENSORHUB.into())
}

/// The provider the default chain gives endpoint `url`: `TENSORFS_TOKEN` for the default
/// endpoint's host, `HF_TOKEN` for huggingface.co, `CIVITAI_TOKEN` for civitai.com, `AWS_*` for
/// `s3://`; else none (anonymous).
pub fn default_for(url: &str) -> Option<Arc<dyn TokenProvider>> {
    let host = base_host(url).ok()?;
    let var = match host.as_str() {
        "huggingface.co" | "hf.co" => "HF_TOKEN",
        "civitai.com" => "CIVITAI_TOKEN",
        h if base_host(&default_endpoint()).is_ok_and(|d| d == h) => "TENSORFS_TOKEN",
        _ => return None,
    };
    env(var).map(bearer)
}

fn env(var: &str) -> Option<String> {
    std::env::var(var).ok().map(|v| v.trim().to_owned()).filter(|v| !v.is_empty())
}
