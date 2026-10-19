//! What a request presents (tensorfs-api §1): a token provider is asked per request with its
//! method, URL, headers and payload digest, so a bearer token, DPoP proofs ([`super::dpop`])
//! and S3 SigV4 all fit. [`default_for`] picks the provider an endpoint gets when the caller
//! gives none, by host.

use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    sign, PayloadChecksumKind, PercentEncodingMode, SignableBody, SignableRequest, SigningSettings,
    UriPathNormalizationMode,
};
use aws_sigv4::sign::v4;

pub use hyper::HeaderMap;
use super::policy::base_host;
use super::tensorhub::TENSORHUB;
use crate::err::{refuse, Code, Refusal, Result};

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
    if url.starts_with("s3://") {
        return SigV4::from_env().map(|s| Arc::new(s) as Arc<dyn TokenProvider>);
    }
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

/// AWS Signature Version 4 for S3 and its kin (R2, MinIO): every header but `range` signed, the
/// payload by its digest when known.
pub struct SigV4 {
    credentials: Credentials,
    region: String,
}

impl SigV4 {
    pub fn new(access_key: &str, secret: &str, session: Option<String>, region: &str) -> SigV4 {
        let credentials = Credentials::new(access_key, secret, session, None, "tensorfs");
        SigV4 { credentials, region: region.to_owned() }
    }

    /// From `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` and `AWS_REGION`
    /// (else `AWS_DEFAULT_REGION`, else `us-east-1`).
    pub fn from_env() -> Option<SigV4> {
        let region = env("AWS_REGION").or_else(|| env("AWS_DEFAULT_REGION")).unwrap_or_else(|| "us-east-1".into());
        Some(SigV4::new(&env("AWS_ACCESS_KEY_ID")?, &env("AWS_SECRET_ACCESS_KEY")?, env("AWS_SESSION_TOKEN"), &region))
    }
}

#[async_trait]
impl TokenProvider for SigV4 {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>> {
        let failed = |e: &dyn std::fmt::Display| Refusal { code: Code::CREDENTIAL_REQUIRED, detail: format!("SigV4: {e}") };
        let identity = self.credentials.clone().into();
        let mut settings = SigningSettings::default();
        settings.percent_encoding_mode = PercentEncodingMode::Single;
        settings.uri_path_normalization_mode = UriPathNormalizationMode::Disabled;
        settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;
        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name("s3")
            .time(SystemTime::now())
            .settings(settings)
            .build()
            .map_err(|e| failed(&e))?
            .into();
        let body = request.payload.map_or(SignableBody::UnsignedPayload, |hex| SignableBody::Precomputed(hex.to_owned()));
        let signed = (request.headers.iter())
            .filter(|(name, _)| !name.eq_ignore_ascii_case("range"))
            .map(|(name, value)| (name.as_str(), value.as_str()));
        let signable = SignableRequest::new(request.method, request.url, signed, body).map_err(|e| failed(&e))?;
        let (instructions, _) = sign(signable, &params).map_err(|e| failed(&e))?.into_parts();
        Ok(instructions.headers().map(|(n, v)| (n.to_owned(), v.to_owned())).collect())
    }
}

/// The HTTP base of `s3://<bucket>[/<prefix>]`, path-style: at `AWS_ENDPOINT_URL_S3`, else
/// `AWS_ENDPOINT_URL`, else AWS's regional endpoint.
pub fn s3_base(url: &str) -> Result<String> {
    let Some(path) = url.strip_prefix("s3://").filter(|p| !p.is_empty() && !p.starts_with('/')) else {
        return refuse(Code::SOURCE_NOT_ALLOWED, format!("{url}: not s3://<bucket>[/<prefix>]"));
    };
    let service = env("AWS_ENDPOINT_URL_S3").or_else(|| env("AWS_ENDPOINT_URL")).unwrap_or_else(|| {
        let region = env("AWS_REGION").or_else(|| env("AWS_DEFAULT_REGION")).unwrap_or_else(|| "us-east-1".into());
        format!("https://s3.{region}.amazonaws.com")
    });
    Ok(format!("{}/{path}", service.trim_end_matches('/')))
}
