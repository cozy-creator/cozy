//! What a request presents (tensorfs-api §1): a token provider is asked per request with its
//! method, URL, headers and payload digest, so a bearer token and S3 SigV4 both fit.
//! [`default_for`] picks the provider an endpoint gets when the caller gives none, by host.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

pub use super::net::HeaderMap;
use super::policy::base_host;
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
    fn retry(&self, sent: &Request<'_>, status: u16, headers: &HeaderMap) -> Result<bool> {
        let _ = (sent, status, headers);
        Ok(false)
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

/// Tensorhub, the default endpoint.
const TENSORHUB: &str = "https://tensorhub.com";

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
    env_token(url).map(bearer)
}

/// The bearer token the environment holds for `url`'s host, and for no other host.
pub(crate) fn env_token(url: &str) -> Option<String> {
    let host = base_host(url).ok()?;
    let var = match host.as_str() {
        "huggingface.co" | "hf.co" => "HF_TOKEN",
        "civitai.com" => "CIVITAI_TOKEN",
        h if base_host(&default_endpoint()).is_ok_and(|d| d == h) => "TENSORFS_TOKEN",
        _ => return None,
    };
    env(var)
}

fn env(var: &str) -> Option<String> {
    std::env::var(var).ok().map(|v| v.trim().to_owned()).filter(|v| !v.is_empty())
}

/// AWS Signature Version 4 for S3 and its kin (R2, MinIO): every header but `range` signed, the
/// payload by its digest when known, the path as sent (S3's is encoded once, never normalized).
pub struct SigV4 {
    access_key: String,
    secret: String,
    session: Option<String>,
    region: String,
}

impl SigV4 {
    pub fn new(access_key: &str, secret: &str, session: Option<String>, region: &str) -> SigV4 {
        SigV4 { access_key: access_key.into(), secret: secret.into(), session, region: region.into() }
    }

    /// From `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` and `AWS_REGION`
    /// (else `AWS_DEFAULT_REGION`, else `us-east-1`).
    pub fn from_env() -> Option<SigV4> {
        let region = env("AWS_REGION").or_else(|| env("AWS_DEFAULT_REGION")).unwrap_or_else(|| "us-east-1".into());
        Some(SigV4::new(&env("AWS_ACCESS_KEY_ID")?, &env("AWS_SECRET_ACCESS_KEY")?, env("AWS_SESSION_TOKEN"), &region))
    }

    /// The headers that sign `request` (all of its headers) at `now`, seconds since the epoch.
    fn sign(&self, request: &Request<'_>, now: u64) -> Result<Vec<(String, String)>> {
        let failed = |detail: String| Refusal { code: Code::CREDENTIAL_REQUIRED, detail: format!("SigV4: {detail}") };
        let uri: hyper::Uri = request.url.parse().map_err(|e| failed(format!("{}: {e}", request.url)))?;
        let authority = uri.authority().ok_or_else(|| failed(format!("{} names no host", request.url)))?;
        let default_port = matches!((uri.scheme_str(), authority.port_u16()), (Some("https"), Some(443)) | (Some("http"), Some(80)));
        let stamp = utc_stamp(now);
        let scope = format!("{}/{}/s3/aws4_request", &stamp[..8], self.region);
        let payload = request.payload.unwrap_or("UNSIGNED-PAYLOAD");
        let mut own = vec![("x-amz-date".to_owned(), stamp.clone()), ("x-amz-content-sha256".to_owned(), payload.to_owned())];
        if let Some(token) = &self.session {
            own.push(("x-amz-security-token".into(), token.clone()));
        }
        // Names lowercased and sorted; a repeated header's trimmed values joined by commas.
        let mut headers: BTreeMap<String, Vec<String>> = BTreeMap::new();
        // What a proxy or the transport may change is never signed (as AWS's SDKs leave it).
        const UNSIGNED: [&str; 4] = ["authorization", "user-agent", "x-amzn-trace-id", "transfer-encoding"];
        for (name, value) in request.headers.iter().filter(|(name, _)| !UNSIGNED.iter().any(|u| name.eq_ignore_ascii_case(u))) {
            headers.entry(name.to_ascii_lowercase()).or_default().push(value.split(' ').filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" "));
        }
        let host = if default_port { authority.host() } else { authority.as_str() };
        headers.entry("host".into()).or_insert_with(|| vec![host.to_owned()]);
        for (name, value) in &own {
            headers.insert(name.clone(), vec![value.clone()]);
        }
        let signed = headers.keys().map(String::as_str).collect::<Vec<_>>().join(";");
        let mut canonical = format!("{}\n{}\n{}\n", request.method, uri.path(), canonical_query(uri.query().unwrap_or("")));
        for (name, values) in &headers {
            canonical += &format!("{name}:{}\n", values.join(","));
        }
        canonical += &format!("\n{signed}\n{payload}");
        let to_sign = format!("AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{:x}", Sha256::digest(canonical.as_bytes()));
        let mut key = hmac(format!("AWS4{}", self.secret).as_bytes(), &stamp.as_bytes()[..8]);
        for part in [self.region.as_str(), "s3", "aws4_request"] {
            key = hmac(&key, part.as_bytes());
        }
        let signature = hmac(&key, to_sign.as_bytes());
        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed}, Signature={}",
            self.access_key,
            signature.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
        own.push(("authorization".into(), authorization));
        Ok(own)
    }
}

#[async_trait]
impl TokenProvider for SigV4 {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>> {
        let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let headers: Vec<_> = request.headers.iter().filter(|(name, _)| !name.eq_ignore_ascii_case("range")).cloned().collect();
        self.sign(&Request { headers: &headers, ..*request }, now)
    }
}

fn hmac(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC takes a key of any size");
    mac.update(message);
    mac.finalize().into_bytes().into()
}

/// `YYYYMMDDTHHMMSSZ` for `secs` since the Unix epoch (days to the civil date, proleptic Gregorian).
fn utc_stamp(secs: u64) -> String {
    let (days, time) = ((secs / 86_400) as i64 + 719_468, secs % 86_400);
    let (era, day_of_era) = (days.div_euclid(146_097), days.rem_euclid(146_097));
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 { shifted_month + 3 } else { shifted_month - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z", time / 3600, time / 60 % 60, time % 60)
}

/// The query as SigV4 signs it: each pair decoded as a form, encoded again with only the
/// unreserved characters bare, the pairs sorted.
fn canonical_query(query: &str) -> String {
    let mut pairs: Vec<(String, String)> = (query.split('&').filter(|pair| !pair.is_empty()))
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (encode(&decode(key)), encode(&decode(value)))
        })
        .collect();
    pairs.sort();
    pairs.iter().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>().join("&")
}

/// Form decoding: `+` is a space, `%XX` a byte, anything else itself.
fn decode(text: &str) -> String {
    let (bytes, mut out, mut at) = (text.as_bytes(), Vec::with_capacity(text.len()), 0);
    while at < bytes.len() {
        let hex = |i: usize| bytes.get(i).and_then(|b| (*b as char).to_digit(16));
        match (bytes[at], hex(at + 1), hex(at + 2)) {
            (b'+', ..) => out.push(b' '),
            (b'%', Some(high), Some(low)) => {
                out.push((high * 16 + low) as u8);
                at += 2;
            }
            (byte, ..) => out.push(byte),
        }
        at += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn encode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
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
