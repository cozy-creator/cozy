//! DPoP (RFC 9449) with a JWT-bearer grant (RFC 7523), standards only: a P-256 key proves
//! every request, and the token it presents is either given (`DPoP::token`) or traded once, on
//! first use, by an assertion that key signs (`DPoP::grant`). The token goes only to its
//! resource's origin; the Endpoint never forwards it across a redirect.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use aws_lc_rs::rand::{SecureRandom, SystemRandom};
use aws_lc_rs::signature::{EcdsaKeyPair, ECDSA_P256_SHA256_FIXED_SIGNING};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use bytes::Bytes;
use hyper::Uri;
use serde_json::{json, Map, Value};

use super::net::{Body, Client};
use super::token::{HeaderMap, Request, TokenProvider};
use crate::err::{refuse, Code, Refusal, Result};
use crate::sha256::digest;

const JWT_BEARER: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";

/// Where and how the token is traded: the authorization server's token endpoint, the client id
/// the assertion is issued as, the resource (the token's audience, and the only origin it is
/// sent to), and the assertion's extra claims (Cozy's: `{"capability": <JWS>}`).
pub struct JwtBearer {
    pub token_endpoint: String,
    pub client_id: String,
    pub resource: String,
    pub claims: Map<String, Value>,
}

pub struct DPoP {
    key: EcdsaKeyPair,
    jwk: Value,
    resource: Option<String>,
    endpoint: Option<String>,
    grant: Option<(JwtBearer, Client)>,
    token: tokio::sync::Mutex<Option<String>>,
    /// The last DPoP nonce each origin sent.
    nonces: Mutex<HashMap<String, String>>,
}

impl DPoP {
    /// `key` (a private P-256 JWK) proving a token `grant` trades at its first use, over `client`.
    pub fn grant(client: &Client, key: &str, grant: JwtBearer) -> Result<DPoP> {
        let resource = Some(grant.resource.trim_end_matches('/').to_owned());
        DPoP::new(key, resource, Some((grant, client.clone())), None)
    }

    /// `key` proving `access_token`, already minted for `resource` (every origin when `None`).
    pub fn token(key: &str, access_token: &str, resource: Option<&str>) -> Result<DPoP> {
        DPoP::new(key, resource.map(|r| r.trim_end_matches('/').to_owned()), None, Some(access_token.to_owned()))
    }

    /// Reach the resource at another origin (for example a private proxy). Proofs still name
    /// the resource's public origin; credentials go only to this explicit endpoint.
    pub fn at(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(origin(endpoint));
        self
    }

    fn new(key: &str, resource: Option<String>, grant: Option<(JwtBearer, Client)>, token: Option<String>) -> Result<DPoP> {
        let bad = |why: &str| Refusal { code: Code::CREDENTIAL_REQUIRED, detail: format!("the DPoP key {why}") };
        let jwk: Value = serde_json::from_str(key).map_err(|_| bad("is not a JWK"))?;
        let field = |name: &str| jwk[name].as_str().and_then(|v| URL_SAFE_NO_PAD.decode(v).ok()).ok_or_else(|| bad(&format!("has no {name}")));
        if (jwk["kty"].as_str(), jwk["crv"].as_str()) != (Some("EC"), Some("P-256")) {
            return Err(bad("is not a P-256 key"));
        }
        let point = [&[4u8][..], &field("x")?, &field("y")?].concat();
        let key = EcdsaKeyPair::from_private_key_and_public_key(&ECDSA_P256_SHA256_FIXED_SIGNING, &field("d")?, &point)
            .map_err(|_| bad("is not a consistent P-256 key pair"))?;
        let jwk = json!({"kty": "EC", "crv": "P-256", "x": jwk["x"], "y": jwk["y"]});
        let endpoint = resource.as_deref().map(origin);
        Ok(DPoP { key, jwk, resource, endpoint, grant, token: tokio::sync::Mutex::new(token), nonces: Mutex::default() })
    }

    /// RFC 7638 thumbprint of the key: what a token's `cnf.jkt` names.
    pub fn thumbprint(&self) -> String {
        let canonical = format!(r#"{{"crv":"P-256","kty":"EC","x":{},"y":{}}}"#, self.jwk["x"], self.jwk["y"]);
        b64(digest(canonical.as_bytes()))
    }

    /// The token already held, if any. Observing it never redeems a grant.
    pub async fn issued_token(&self) -> Option<String> {
        self.token.lock().await.clone()
    }

    /// A compact ES256 JWS of `claims` under `header`.
    fn sign(&self, mut header: Value, claims: &Value) -> String {
        header["alg"] = "ES256".into();
        let input = format!("{}.{}", b64(header.to_string()), b64(claims.to_string()));
        let signature = self.key.sign(&SystemRandom::new(), input.as_bytes()).expect("ES256 signing");
        format!("{input}.{}", b64(signature.as_ref()))
    }

    /// A proof for `method url`, presenting `token` when given.
    fn proof(&self, method: &str, url: &str, token: Option<&str>, nonce_at: &str) -> String {
        let htu = url.split(['?', '#']).next().unwrap_or(url);
        let mut claims = json!({"jti": jti(), "htm": method, "htu": htu, "iat": now()});
        if let Some(token) = token {
            claims["ath"] = b64(digest(token.as_bytes())).into();
        }
        if let Some(nonce) = self.nonces.lock().unwrap().get(&origin(nonce_at)) {
            claims["nonce"] = nonce.clone().into();
        }
        self.sign(json!({"typ": "dpop+jwt", "jwk": self.jwk}), &claims)
    }

    /// Keeps the nonce `headers` carry for `url`'s origin: true when it is new.
    fn keep_nonce(&self, url: &str, headers: &HeaderMap) -> bool {
        let Some(nonce) = headers.get("dpop-nonce").and_then(|v| v.to_str().ok()).map(str::trim).filter(|n| !n.is_empty()) else {
            return false;
        };
        self.nonces.lock().unwrap().insert(origin(url), nonce.to_owned()).as_deref() != Some(nonce)
    }

    /// The token, traded on first need; one trade at a time.
    async fn access_token(&self) -> Result<String> {
        let mut held = self.token.lock().await;
        if let Some(token) = held.as_ref() {
            return Ok(token.clone());
        }
        let Some((grant, client)) = &self.grant else {
            return refuse(Code::CREDENTIAL_REQUIRED, "a DPoP provider with neither a token nor a grant");
        };
        Ok(held.insert(self.trade(grant, client).await?).clone())
    }

    async fn trade(&self, grant: &JwtBearer, client: &Client) -> Result<String> {
        let issued = now();
        let mut claims = json!({"iss": grant.client_id, "sub": self.thumbprint(), "aud": grant.token_endpoint,
            "iat": issued, "exp": issued + 60, "jti": jti()});
        claims.as_object_mut().expect("an object").extend(grant.claims.clone());
        let assertion = self.sign(json!({"typ": "JWT", "jwk": self.jwk}), &claims);
        let form = [("grant_type", JWT_BEARER), ("assertion", &assertion), ("client_id", &grant.client_id), ("resource", &grant.resource)];
        let form: Vec<String> = form.iter().map(|(k, v)| format!("{k}={}", form_encode(v))).collect();
        let form = Bytes::from(form.join("&"));
        let url = &grant.token_endpoint;
        let mut nonce_retried = false;
        loop {
            let headers = vec![
                ("content-type".to_owned(), "application/x-www-form-urlencoded".to_owned()),
                ("accept".to_owned(), "application/json".to_owned()),
                ("dpop".to_owned(), self.proof("POST", url, None, url)),
            ];
            let mut answer = client.send("POST", url, &headers, Some(Body::Bytes(form.clone()))).await?;
            let (status, fresh) = (answer.status(), self.keep_nonce(url, answer.response.headers()));
            let mut body = Vec::new();
            while let Some(frame) = answer.frame().await? {
                body.extend_from_slice(&frame);
                if body.len() > 1 << 20 {
                    return refuse(Code::SIZE_CAP, format!("{url}: a token answer over 1 MiB"));
                }
            }
            let answered: Value = serde_json::from_slice(&body).unwrap_or_default();
            if status == 200 {
                let token = answered["access_token"].as_str().filter(|t| !t.is_empty());
                return match (token, answered["token_type"].as_str()) {
                    (Some(token), Some(kind)) if kind.eq_ignore_ascii_case("DPoP") => Ok(token.to_owned()),
                    _ => refuse(Code::CREDENTIAL_REQUIRED, format!("{url}: no DPoP-bound token in the answer")),
                };
            }
            if answered["error"] == "use_dpop_nonce" && fresh && !nonce_retried {
                nonce_retried = true;
                continue;
            }
            let why: Vec<&str> = ["error", "reason", "error_description"].iter().filter_map(|k| answered[k].as_str()).collect();
            let code = match status {
                408 | 425 | 429 | 500..=599 => Code::TRANSFER_FAILED,
                _ => Code::CREDENTIAL_REQUIRED,
            };
            return refuse(code, format!("{url}: {status} {}", why.join(": ")));
        }
    }
}

#[async_trait]
impl TokenProvider for DPoP {
    async fn headers(&self, request: &Request<'_>) -> Result<Vec<(String, String)>> {
        if self.endpoint.as_deref().is_some_and(|r| r != origin(request.url)) {
            return Ok(vec![]);
        }
        let token = self.access_token().await?;
        let public = self.resource.as_deref().map(|resource| {
            let path = request.url.parse::<Uri>().ok().and_then(|uri| uri.path_and_query().map(|p| p.as_str().to_owned())).unwrap_or_else(|| "/".into());
            format!("{}{path}", origin(resource))
        });
        let proof = self.proof(request.method, public.as_deref().unwrap_or(request.url), Some(&token), request.url);
        Ok(vec![("authorization".into(), format!("DPoP {token}")), ("dpop".into(), proof)])
    }

    fn retry(&self, sent: &Request<'_>, status: u16, headers: &HeaderMap) -> bool {
        let fresh = self.keep_nonce(sent.url, headers);
        let challenged = headers.get_all("www-authenticate").iter().any(|v| v.to_str().is_ok_and(|v| v.contains("use_dpop_nonce")));
        let proved = sent.headers.iter().any(|(name, _)| name.eq_ignore_ascii_case("dpop"));
        proved && fresh && (status == 400 || (status == 401 && challenged))
    }
}

fn origin(url: &str) -> String {
    match url.parse::<Uri>() {
        Ok(uri) => format!("{}://{}", uri.scheme_str().unwrap_or_default(), uri.authority().map_or("", |a| a.as_str())).to_ascii_lowercase(),
        Err(_) => url.to_ascii_lowercase(),
    }
}

fn b64(bytes: impl AsRef<[u8]>) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

fn jti() -> String {
    let mut raw = [0u8; 24];
    SystemRandom::new().fill(&mut raw).expect("the system random source");
    b64(raw)
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// `application/x-www-form-urlencoded` of one value.
fn form_encode(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'*' => (b as char).to_string(),
            b' ' => "+".to_owned(),
            b => format!("%{b:02X}"),
        })
        .collect()
}
