//! A run's private access on a machine (th-241): the owner's capability, traded once through
//! AuthKit's JWT-bearer grant for a DPoP-bound token its leaf key proves. No refresh, no second
//! trade: a refused token ends the capability.
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::Error;
use crate::jws::{P256Key, claims, jti, now};

/// The AuthKit client every machine asserts as.
const CLIENT_ID: &str = "tensord";
const JWT_BEARER: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";
/// After a trade nobody answered, the next request asks again only after this long.
const QUIET: Duration = Duration::from_secs(15);

/// One private operation, as a capability names it.
#[derive(Clone, Copy, Debug)]
pub enum Op<'a> {
    /// Read one checkpoint: `model` is `org/name`, `manifest` `sha256:<hex>`.
    Read { model: &'a str, manifest: &'a str },
    /// Publish checkpoints to a model.
    Publish { model: &'a str },
}

/// Why a capability no longer serves.
#[derive(Clone, Debug, thiserror::Error)]
pub enum Ended {
    #[error("the run's capability expired")]
    Expired,
    #[error("the Hub refused the run's capability: {0}")]
    Refused(String),
}

pub struct RunToken {
    jws: String,
    expires: u64,
    operations: Vec<Value>,
    /// The Hub's public origin: the token's audience, and the origin every proof names.
    resource: String,
    token_endpoint: String,
    leaf: Arc<P256Key>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    token: Option<(String, Instant)>,
    ended: Option<Ended>,
    quiet_until: Option<Instant>,
    /// The last DPoP nonce each host sent.
    nonces: HashMap<String, String>,
}

#[derive(Deserialize)]
struct Issued {
    access_token: String,
    token_type: String,
    expires_in: u64,
}

#[derive(Deserialize)]
struct OAuthError {
    error: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    error_description: String,
}

impl RunToken {
    /// `jws` as the run carries it, traded at `token_endpoint` with `leaf`. Its claims are read
    /// only to know the operations, the Hub and the expiry; AuthKit and the Hub verify them.
    pub fn new(jws: &str, token_endpoint: &str, leaf: Arc<P256Key>) -> Result<Self, Error> {
        let invalid = |why: &str| Error::Capability(format!("the run's capability {why}"));
        let claims = claims(jws)
            .filter(|_| jws.len() <= 64 << 10)
            .ok_or_else(|| invalid("is not a compact JWS"))?;
        if claims["cnf"]["jkt"].as_str() != Some(leaf.thumbprint().as_str()) {
            return Err(invalid("names another machine key"));
        }
        let expires = claims["exp"]
            .as_u64()
            .ok_or_else(|| invalid("has no exp"))?;
        let resource = claims["aud"]
            .as_str()
            .unwrap_or_default()
            .trim_end_matches('/')
            .to_string();
        let operations = claims["authorization_details"]
            .as_array()
            .filter(|ops| !ops.is_empty())
            .cloned()
            .ok_or_else(|| invalid("names no operations"))?;
        if !resource.starts_with("https://") && !resource.starts_with("http://") {
            return Err(invalid("names no Hub origin"));
        }
        Ok(Self {
            jws: jws.to_string(),
            expires,
            operations,
            resource,
            token_endpoint: token_endpoint.to_string(),
            leaf,
            state: Mutex::default(),
        })
    }

    /// The Hub's public origin the capability names: the token's audience.
    pub fn resource(&self) -> &str {
        &self.resource
    }

    /// Whether the capability names `op`.
    pub fn names(&self, op: Op) -> bool {
        self.operations.iter().any(|named| match op {
            Op::Read { model, manifest } => {
                named["type"] == "tensorhub_model_read"
                    && named["model"] == model
                    && named["manifest"] == manifest
            }
            Op::Publish { model } => {
                named["type"] == "tensorhub_model_publish" && named["model"] == model
            }
        })
    }

    /// Why the capability no longer serves, if it does not.
    pub fn ended(&self) -> Option<Ended> {
        self.state.lock().unwrap().ended.clone()
    }

    /// The token, traded over `http` on first need; an error once the capability ended or while
    /// a trade nobody answered rests.
    pub async fn token(&self, http: &reqwest::Client) -> Result<String, Error> {
        {
            let mut state = self.state.lock().unwrap();
            if now() >= self.expires {
                state.ended = Some(Ended::Expired);
            }
            if let Some(ended) = &state.ended {
                return Err(Error::Ended(ended.clone()));
            }
            if let Some((token, until)) = &state.token {
                if Instant::now() < *until {
                    return Ok(token.clone());
                }
                return Err(Error::Ended(Ended::Expired));
            }
            if state
                .quiet_until
                .is_some_and(|until| Instant::now() < until)
            {
                return Err(Error::Capability(
                    "the token endpoint did not answer; waiting to ask again".into(),
                ));
            }
        }
        let traded = self.trade(http).await;
        let mut state = self.state.lock().unwrap();
        match traded {
            Ok(issued) => {
                state.token = Some((
                    issued.access_token.clone(),
                    Instant::now() + Duration::from_secs(issued.expires_in),
                ));
                state.quiet_until = None;
                Ok(issued.access_token)
            }
            Err(Error::Ended(ended)) => {
                tracing::warn!(%ended, "run capability");
                state.ended = Some(ended.clone());
                Err(Error::Ended(ended))
            }
            Err(other) => {
                tracing::warn!(error = %other, "run capability trade unanswered");
                state.quiet_until = Some(Instant::now() + QUIET);
                Err(other)
            }
        }
    }

    async fn trade(&self, http: &reqwest::Client) -> Result<Issued, Error> {
        let issued = now();
        let assertion = self.leaf.sign(
            json!({"typ": "JWT", "jwk": self.leaf.jwk()}),
            &json!({"iss": CLIENT_ID, "sub": self.leaf.thumbprint(), "aud": self.token_endpoint,
                "iat": issued, "exp": issued + 60, "jti": jti(), "capability": self.jws}),
        );
        let form = [
            ("grant_type", JWT_BEARER),
            ("assertion", &assertion),
            ("client_id", CLIENT_ID),
            ("resource", &self.resource),
        ];
        let unanswered = |why: String| Error::Capability(format!("{}: {why}", self.token_endpoint));
        let mut nonce = None;
        for _ in 0..2 {
            let proof = self
                .leaf
                .proof("POST", &self.token_endpoint, None, nonce.as_deref());
            let answer = http
                .post(&self.token_endpoint)
                .header("DPoP", proof)
                .header("Accept", "application/json")
                .form(&form)
                .send()
                .await
                .map_err(|e| unanswered(e.to_string()))?;
            let status = answer.status().as_u16();
            let fresh = answer
                .headers()
                .get("dpop-nonce")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let body = answer
                .bytes()
                .await
                .map_err(|e| unanswered(e.to_string()))?;
            if status == 200 {
                let issued: Issued = serde_json::from_slice(&body)
                    .map_err(|_| unanswered("an unreadable token answer".into()))?;
                if !issued.token_type.eq_ignore_ascii_case("DPoP") || issued.access_token.is_empty()
                {
                    return Err(unanswered(format!(
                        "a {} token, not a DPoP-bound one",
                        issued.token_type
                    )));
                }
                return Ok(issued);
            }
            let refused = serde_json::from_slice::<OAuthError>(&body)
                .ok()
                .filter(|_| (400..500).contains(&status));
            match refused {
                Some(refused)
                    if refused.error == "use_dpop_nonce" && fresh.is_some() && nonce.is_none() =>
                {
                    nonce = fresh
                }
                Some(refused) if refused.reason == "capability_expired" => {
                    return Err(Error::Ended(Ended::Expired));
                }
                Some(refused) => {
                    let why = [
                        refused.error,
                        refused.reason,
                        refused.error_description.chars().take(256).collect(),
                    ]
                    .into_iter()
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>()
                    .join(": ");
                    return Err(Error::Ended(Ended::Refused(why)));
                }
                None => return Err(unanswered(format!("HTTP {status}"))),
            }
        }
        Err(unanswered("the DPoP nonce changed twice".into()))
    }

    /// The `Authorization` and `DPoP` headers for one request to the Hub: `path` is the
    /// request's path, which the proof names under the Hub's public origin.
    pub fn present(
        &self,
        token: &str,
        method: &str,
        host: &str,
        path: &str,
    ) -> [(&'static str, String); 2] {
        let nonce = self
            .state
            .lock()
            .unwrap()
            .nonces
            .get(&host.to_ascii_lowercase())
            .cloned();
        let proof = self.leaf.proof(
            method,
            &format!("{}{}", self.resource, path),
            Some(token),
            nonce.as_deref(),
        );
        [("authorization", format!("DPoP {token}")), ("dpop", proof)]
    }

    /// What `host` answered a request that presented `token`: its DPoP nonce (`nonce`), and
    /// whether `www-authenticate` asked for one (`challenged`). Keeps the nonce; a refused
    /// token ends the capability. True when the request should go again with a fresh proof.
    pub fn observe(
        &self,
        host: &str,
        token: &str,
        status: u16,
        nonce: Option<&str>,
        challenged: bool,
    ) -> bool {
        let fresh = nonce.map(str::trim).filter(|n| !n.is_empty());
        let mut state = self.state.lock().unwrap();
        let changed = fresh.is_some_and(|fresh| {
            state
                .nonces
                .insert(host.to_ascii_lowercase(), fresh.to_string())
                .as_deref()
                != Some(fresh)
        });
        match status {
            401 if challenged => changed,
            401 => {
                if state.token.as_ref().is_some_and(|(held, _)| held == token) {
                    state.token = None;
                    state.ended = Some(if now() >= self.expires {
                        Ended::Expired
                    } else {
                        Ended::Refused("the Hub refused the run's token".into())
                    });
                }
                false
            }
            400 => changed,
            _ => false,
        }
    }

    pub(crate) fn answered(
        &self,
        url: &reqwest::Url,
        token: &str,
        response: &reqwest::Response,
    ) -> bool {
        let header = |name: &str| response.headers().get(name).and_then(|v| v.to_str().ok());
        let challenged = header("www-authenticate").is_some_and(|v| v.contains("use_dpop_nonce"));
        let host = url.host_str().unwrap_or_default();
        self.observe(
            host,
            token,
            response.status().as_u16(),
            header("dpop-nonce"),
            challenged,
        )
    }
}

impl crate::Client {
    /// Where a machine trades a run capability for this Hub: its authorization server's token
    /// endpoint (RFC 9728 resource metadata, then the issuer's metadata). The CLI finds it once.
    pub async fn token_endpoint(&self) -> Result<String, Error> {
        let resource = self.get_protected_resource_metadata().await?.into_inner();
        let issuer = resource
            .authorization_servers
            .first()
            .map(|s| s.trim_end_matches('/').to_string());
        let server = self.get_openid_configuration().await?.into_inner();
        let usable = issuer.as_deref() == Some(server.issuer.trim_end_matches('/'))
            && server
                .grant_types_supported
                .iter()
                .any(|grant| grant == JWT_BEARER);
        if !usable {
            return Err(Error::Answer(
                "this Hub names no authorization server that takes run capabilities".into(),
            ));
        }
        Ok(server.token_endpoint)
    }
}
