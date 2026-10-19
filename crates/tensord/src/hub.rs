//! Per-request Hub access. Package runs pass their capability and key to the Runtime.
//! Code-less runs use TensorFS's lazy DPoP provider and distribution endpoint themselves.
use base64::Engine as _;
use hub_client::Credential;
use serde_json::{Value, json};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tensorfs_core::fetch::token::{HeaderMap, Request};
use tensorfs_core::fetch::{Client as FetchClient, DPoP, JwtBearer, SourcePolicy, TokenProvider};

pub use hub_client::types;

#[derive(Clone, Copy, Debug)]
pub enum Op<'a> {
    Read { model: &'a str, manifest: &'a str },
    Publish { model: &'a str },
}

/// A run's reasons when its capability does not carry it (th-241's typed refusals).
pub mod reason {
    /// A private operation the run's capability does not name.
    pub const REQUIRED: &str = "capability_required";
    /// The Hub refused the capability: its trade, or its token since (a revoked device key).
    pub const REFUSED: &str = "capability_refused";
    /// The capability's `exp` passed.
    pub const EXPIRED: &str = "capability_expired";
    /// The Hub narrowed the capability below this operation.
    pub const EXCEEDED: &str = "capability_exceeded";
}

/// `scheme://host:port`, lowercase, default port spelled: one key per origin.
pub fn origin_key(origin: &str) -> Option<String> {
    let origin = origin.trim_end_matches('/');
    let (scheme, rest) = origin.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if rest.is_empty() || rest.contains(['/', '?', '#', '@']) {
        return None;
    }
    let (host, port) = match rest.rsplit_once(':') {
        Some((host, port)) if !port.contains(']') => (host, Some(port.parse::<u16>().ok()?)),
        _ => (rest, None),
    };
    let port = match (port, scheme.as_str()) {
        (Some(port), _) => port,
        (None, "https") => 443,
        (None, "http") => 80,
        _ => return None,
    };
    Some(format!("{scheme}://{}:{port}", host.to_ascii_lowercase()))
}

fn loopback_host(key: &str) -> bool {
    let host = key
        .split_once("://")
        .and_then(|(_, rest)| rest.rsplit_once(':'))
        .map(|(host, _)| host)
        .unwrap_or_default();
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

/// An HTTPS origin, or plaintext HTTP on loopback only.
pub fn valid_origin(origin: &str) -> bool {
    match origin_key(origin) {
        Some(key) => {
            key.starts_with("https://") || key.starts_with("http://") && loopback_host(&key)
        }
        None => false,
    }
}

/// The origin of an absolute URL: everything before its path.
fn origin_of(url: &str) -> &str {
    let after = url.find("://").map_or(0, |at| at + 3);
    url[after..]
        .find(['/', '?', '#'])
        .map_or(url, |end| &url[..after + end])
}

/// Whether `url` is a file the run's Hub publishes (its index's file door).
pub fn publishes(source: &Source, url: &str) -> bool {
    origin_key(origin_of(url)).is_some_and(|key| origin_key(&source.origin) == Some(key))
}

/// An absolute URL at a valid origin, without query or fragment.
fn valid_url(url: &str) -> bool {
    valid_origin(origin_of(url)) && !url.contains(['?', '#'])
}

pub(crate) fn pem(der: &[u8]) -> String {
    let body = base64::engine::general_purpose::STANDARD.encode(der);
    let lines: Vec<_> = body
        .as_bytes()
        .chunks(64)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect();
    format!(
        "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
        lines.join("\n")
    )
}

/// A Hub refusal: the HTTP-level detail, never a token.
#[derive(Debug, Clone)]
pub struct Refusal(pub String);

/// Blocks on one Hub call. TensorD's run code is synchronous; its Hub calls share one
/// current-thread runtime, made once. A thread already inside an async context (TensorFS's
/// mover asking for the token) hands the call to a helper thread instead.
///
/// The seam xs-040 U11's runtime unification folds into TensorD's main runtime.
pub(crate) fn block<T: Send>(call: impl Future<Output = T> + Send) -> T {
    static HUB: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    let runtime = HUB.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("the Hub runtime")
    });
    if tokio::runtime::Handle::try_current().is_err() {
        return runtime.block_on(call);
    }
    std::thread::scope(|scope| {
        scope
            .spawn(|| runtime.block_on(call))
            .join()
            .expect("a Hub call")
    })
}

/// HTTPS to one Hub: the public roots and its granted CA, no redirects. A live Hub that is
/// merely slow is waited for; a dead one is found by TCP keepalive (probes after 30 s of
/// silence, 5 s apart), as TensorFS's downloader finds its own. For calls on the Hub runtime
/// (`block`).
pub fn http(ca_der: Option<&[u8]>) -> Result<reqwest::Client, Refusal> {
    static CLIENTS: Mutex<Vec<(Option<Vec<u8>>, reqwest::Client)>> = Mutex::new(Vec::new());
    pooled(&CLIENTS, ca_der)
}

type Pool = Mutex<Vec<(Option<Vec<u8>>, reqwest::Client)>>;

fn pooled(pool: &Pool, ca_der: Option<&[u8]>) -> Result<reqwest::Client, Refusal> {
    let mut clients = pool.lock().unwrap();
    if let Some((_, client)) = clients.iter().find(|(ca, _)| ca.as_deref() == ca_der) {
        return Ok(client.clone());
    }
    let unusable = |why: String| Refusal(format!("the Hub's TLS: {why}"));
    let mut roots = rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    if let Some(der) = ca_der {
        roots
            .add(der.to_vec().into())
            .map_err(|e| unusable(e.to_string()))?;
    }
    let tls = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::aws_lc_rs::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| unusable(e.to_string()))?
    .with_root_certificates(roots)
    .with_no_client_auth();
    const PROBE: Duration = Duration::from_secs(5);
    let client = reqwest::Client::builder()
        .tls_backend_preconfigured(tls)
        .redirect(reqwest::redirect::Policy::none())
        .tcp_keepalive(PROBE * 6)
        .tcp_keepalive_interval(PROBE)
        .tcp_keepalive_retries(6)
        .build()
        .map_err(|e| unusable(e.to_string()))?;
    clients.push((ca_der.map(<[u8]>::to_vec), client.clone()));
    Ok(client)
}

/// Per-request access. The generic TensorFS provider holds the key and trades lazily once.
pub struct Capability {
    access: protocol::machine::HubAccess,
    claims: Value,
    provider: Mutex<Option<Arc<DPoP>>>,
    refused: AtomicBool,
    forbidden: AtomicBool,
}

impl Capability {
    pub fn new(jws: &str, token_endpoint: &str, key: &str) -> Result<Self, Refusal> {
        if !valid_url(token_endpoint) {
            return Err(Refusal(
                "the run's capability comes with no token endpoint".into(),
            ));
        }
        let claims = jws
            .split('.')
            .nth(1)
            .filter(|_| jws.len() <= 64 << 10 && jws.split('.').count() == 3)
            .and_then(|c| {
                base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .decode(c)
                    .ok()
            })
            .and_then(|c| serde_json::from_slice::<Value>(&c).ok())
            .ok_or_else(|| Refusal("the run's capability is not a compact JWS".into()))?;
        let resource = claims["aud"].as_str().unwrap_or_default();
        if !valid_origin(resource)
            || claims["exp"].as_u64().is_none()
            || !claims["authorization_details"]
                .as_array()
                .is_some_and(|ops| !ops.is_empty())
        {
            return Err(Refusal(
                "the run's capability lacks its origin, expiry or operations".into(),
            ));
        }
        let check = DPoP::token(key, "", Some(resource)).map_err(|e| Refusal(e.to_string()))?;
        if claims["cnf"]["jkt"].as_str() != Some(check.thumbprint().as_str()) {
            return Err(Refusal(
                "the run's capability names another request key".into(),
            ));
        }
        Ok(Self {
            access: protocol::machine::HubAccess {
                capability: jws.into(),
                key: key.into(),
                token_endpoint: token_endpoint.into(),
                ..Default::default()
            },
            claims,
            provider: Mutex::default(),
            refused: AtomicBool::new(false),
            forbidden: AtomicBool::new(false),
        })
    }

    pub fn names(&self, op: Op) -> bool {
        self.claims["authorization_details"]
            .as_array()
            .is_some_and(|ops| {
                ops.iter().any(|named| match op {
                    Op::Read { model, manifest } => {
                        named["type"] == "tensorhub_model_read"
                            && named["model"] == model.to_ascii_lowercase()
                            && named["revision"] == manifest
                    }
                    Op::Publish { model } => {
                        named["type"] == "tensorhub_model_upload"
                            && named["model"] == model.to_ascii_lowercase()
                    }
                })
            })
    }

    pub fn expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now >= self.claims["exp"].as_u64().unwrap_or_default()
    }

    fn ended(&self) -> Option<(&'static str, String)> {
        if self.expired() {
            Some((reason::EXPIRED, "the run's capability expired".into()))
        } else if self.refused.load(Ordering::Relaxed) {
            Some((
                reason::REFUSED,
                "the Hub refused the run's capability".into(),
            ))
        } else {
            None
        }
    }

    fn provider(&self, source: &Source) -> Result<Arc<DPoP>, Refusal> {
        let mut held = self.provider.lock().unwrap();
        if let Some(provider) = held.as_ref() {
            return Ok(provider.clone());
        }
        let client = source.fetch_client()?;
        let grant = JwtBearer {
            token_endpoint: self.access.token_endpoint.clone(),
            client_id: "tensord".into(),
            resource: self.claims["aud"].as_str().unwrap_or_default().into(),
            claims: json!({"capability": self.access.capability})
                .as_object()
                .unwrap()
                .clone(),
        };
        let provider = DPoP::grant(&client, &self.access.key, grant)
            .map_err(|e| Refusal(e.to_string()))?
            .at(&source.origin);
        Ok(held.insert(Arc::new(provider)).clone())
    }
}

impl std::fmt::Debug for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Capability").finish_non_exhaustive()
    }
}

/// The request access passed on the executor control channel, never process credentials.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RuntimeHubAccess {
    pub endpoint: String,
    pub capability: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub key: String,
    pub token_endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ca_der: Option<String>,
}

impl std::fmt::Debug for RuntimeHubAccess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeHubAccess").field("endpoint", &self.endpoint).finish_non_exhaustive()
    }
}

/// Where a run reads a Hub, and the owner's capability there, if the run has private work.
#[derive(Clone)]
pub struct Source {
    pub origin: String,
    pub ca_der: Option<Vec<u8>>,
    pub object_hosts: Vec<String>,
    pub capability: Option<Arc<Capability>>,
}

impl Source {
    fn fetch_client(&self) -> Result<FetchClient, Refusal> {
        let local = origin_key(&self.origin).is_some_and(|key| loopback_host(&key));
        let policy = SourcePolicy {
            any_host: true,
            allow_local: local,
            max_redirects: 5,
            ..Default::default()
        };
        match &self.ca_der {
            Some(der) => FetchClient::with_roots(policy, pem(der).as_bytes())
                .map_err(|e| Refusal(e.to_string())),
            None => Ok(FetchClient::new(policy)),
        }
    }

    /// The original per-request access, passed through to the Runtime without a trade.
    pub fn runtime_access(&self) -> RuntimeHubAccess {
        let access = self.capability.as_ref().map(|c| c.access.clone()).unwrap_or_default();
        RuntimeHubAccess {
            endpoint: self.origin.clone(), capability: access.capability, key: access.key, token_endpoint: access.token_endpoint,
            ca_der: self.ca_der.as_ref().map(|der| base64::engine::general_purpose::STANDARD.encode(der)),
        }
    }

    /// Public prefetch never redeems a package run's capability.
    pub fn anonymous(&self) -> Self {
        Self {
            capability: None,
            ..self.clone()
        }
    }

    pub fn new(
        origin: &str,
        ca_der: Option<Vec<u8>>,
        object_hosts: Vec<String>,
        capability: Option<Capability>,
    ) -> Result<Self, Refusal> {
        if !valid_origin(origin) {
            return Err(Refusal(format!("{origin} is not an HTTPS origin")));
        }
        Ok(Self {
            origin: origin.trim_end_matches('/').to_string(),
            ca_der,
            object_hosts,
            capability: capability.map(Arc::new),
        })
    }
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source")
            .field("origin", &self.origin)
            .finish()
    }
}

#[async_trait::async_trait]
impl TokenProvider for Capability {
    async fn headers(&self, request: &Request<'_>) -> tensorfs_core::err::Result<Vec<(String, String)>> {
        if let Some((code, detail)) = self.ended() {
            return tensorfs_core::err::refuse(tensorfs_core::err::Code::CREDENTIAL_REQUIRED, format!("{code}: {detail}"));
        }
        let provider = self.provider.lock().unwrap().clone().expect("a scoped provider");
        let answer = provider.headers(request).await;
        if answer.as_ref().is_err_and(|e| e.code == tensorfs_core::err::Code::CREDENTIAL_REQUIRED) {
            self.refused.store(true, Ordering::Relaxed);
        }
        answer
    }

    fn retry(&self, sent: &Request<'_>, status: u16, headers: &HeaderMap) -> bool {
        let provider = self.provider.lock().unwrap().clone().expect("a scoped provider");
        let retry = provider.retry(sent, status, headers);
        if sent.headers.iter().any(|(name, _)| name.eq_ignore_ascii_case("authorization")) {
            if status == 403 { self.forbidden.store(true, Ordering::Relaxed); }
            if status == 401 && !retry { self.refused.store(true, Ordering::Relaxed); }
        }
        retry
    }
}

/// Reads and publications at a Hub, credentials presented only to the Hub host; presigned
/// object hosts see nothing. Anonymous unless built for an operation the capability names.
pub struct Catalog {
    origin: String,
    hub: hub_client::Client,
    /// The Hub as the run's capability, for an operation it names.
    credential: Option<Arc<Capability>>,
    ca_der: Option<Vec<u8>>,
}

impl Catalog {
    /// Public reads: anonymous.
    pub fn new(source: &Source) -> Result<Self, Refusal> {
        let origin = source.origin.clone();
        let http = http(source.ca_der.as_deref())?;
        let hub = hub_client::Client::connect(&origin, http, Credential::Anonymous);
        Ok(Self {
            origin,
            hub,
            credential: None,
            ca_der: source.ca_der.clone(),
        })
    }

    /// Scope a lazy provider to the operation without trading its capability.
    pub fn for_op(source: &Source, op: Op) -> Result<Self, (&'static str, String)> {
        let catalog = Self::scoped(source, op)?;
        if let Some(ended) = catalog
            .credential
            .as_ref()
            .and_then(|p| p.ended())
        {
            return Err(ended);
        }
        Ok(catalog)
    }

    /// `for_op` with nothing traded until a request needs the token (a fetch whose objects
    /// are all here asks nothing).
    pub fn scoped(source: &Source, op: Op) -> Result<Self, (&'static str, String)> {
        let mut catalog = Self::new(source).map_err(|e| ("catalog_read_failed", e.0))?;
        match &source.capability {
            Some(capability) if capability.names(op) => {
                capability.provider(source).map_err(|e| ("hub_access_invalid", e.0))?;
                catalog.credential = Some(capability.clone());
            }
            _ => {
                if let Op::Publish { model } = op {
                    return Err((
                        reason::REQUIRED,
                        format!("the run's capability does not let it publish to {model}"),
                    ));
                }
            }
        }
        Ok(catalog)
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// Whether its calls present the run's capability.
    pub(crate) fn presents(&self) -> bool {
        self.credential.is_some()
    }

    pub(crate) fn endpoint(&self) -> tensorfs_core::err::Result<tensorfs_core::fetch::Endpoint> {
        let source = Source::new(&self.origin, self.ca_der.clone(), vec![], None).map_err(|e| {
            tensorfs_core::err::Refusal {
                code: tensorfs_core::err::Code::CREDENTIAL_REQUIRED,
                detail: e.0,
            }
        })?;
        let client = source
            .fetch_client()
            .map_err(|e| tensorfs_core::err::Refusal {
                code: tensorfs_core::err::Code::CREDENTIAL_REQUIRED,
                detail: e.0,
            })?;
        let token = self
            .credential
            .as_ref()
            .map(|p| p.clone() as Arc<dyn TokenProvider>);
        let endpoint = tensorfs_core::fetch::Endpoint::new(&client, &self.origin, token)?;
        Ok(if self.credential.is_none() { endpoint.anonymous() } else { endpoint })
    }

    /// Headers presenting the capability on `method url`; none off the Hub host.
    pub fn present(&self, method: &str, url: &str) -> Vec<(String, String)> {
        let Some(presenter) = &self.credential else {
            return Vec::new();
        };
        block(presenter.headers(&Request { method, url, headers: &[], payload: None })).unwrap_or_default()
    }

    /// A package and its releases.
    pub fn package(&self, org: &str, name: &str) -> Result<types::PackageCard, hub_client::Error> {
        block(async { Ok(self.hub.get_package(org, name, None).await?.into_inner()) })
    }

    /// The exact files of one committed release (`python_version` picks its project wheel):
    /// its documents' bytes and its wheels behind presigned URLs, which expire.
    pub fn install_plan(
        &self,
        org: &str,
        name: &str,
        release: &str,
        python_version: Option<&str>,
    ) -> Result<types::PackageInstallPlan, hub_client::Error> {
        let empty = types::Empty(Default::default());
        block(async {
            let plan = self
                .hub
                .get_package_download_plan(org, name, python_version, Some(release), &empty)
                .await?;
            Ok(plan.into_inner())
        })
    }

    /// A failed Hub call as a run's reason: the capability's typed refusal when it no longer
    /// serves or the Hub narrowed it, else `code`.
    pub fn reason(
        &self,
        code: &'static str,
        detail: impl std::fmt::Display,
    ) -> (&'static str, String) {
        let Some(presenter) = &self.credential else {
            return (code, detail.to_string());
        };
        if let Some(ended) = presenter.ended() {
            return ended;
        }
        if presenter.forbidden.load(Ordering::Relaxed) {
            return (reason::EXCEEDED, format!("{}: {detail}", self.origin));
        }
        (code, detail.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use serde_json::{Value, json};

    fn leaf() -> Arc<hub_client::P256Key> {
        let pem = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
            .unwrap()
            .serialize_pem();
        Arc::new(hub_client::P256Key::from_pem(&pem).unwrap())
    }

    fn signed(leaf: &hub_client::P256Key, ops: Value) -> String {
        let part = |value: Value| URL_SAFE_NO_PAD.encode(value.to_string());
        let exp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 600;
        format!(
            "{}.{}.sig",
            part(json!({"alg": "EdDSA", "kid": "dk-1"})),
            part(
                json!({"sub": "alice", "aud": "https://hub.example", "cnf": {"jkt": leaf.thumbprint()},
                "exp": exp, "authorization_details": ops})
            )
        )
    }

    /// Public reads are anonymous; only an operation the capability names carries it, and a
    /// publication it does not name is refused before the Hub hears of it.
    #[test]
    fn only_named_operations_carry_the_capability() {
        let leaf = leaf();
        let manifest = format!("sha256:{}", "ab".repeat(32));
        let jws = signed(
            &leaf,
            json!([
                {"type": "tensorhub_model_read", "model": "alice/private", "revision": manifest},
                {"type": "tensorhub_model_upload", "model": "alice/out"},
            ]),
        );
        // Nothing answers here: a trade goes unanswered and the operation keeps its capability.
        const TOKENS: &str = "http://127.0.0.1:9/v1/auth/oauth2/token";
        let capability = Capability::new(&jws, TOKENS, &leaf.private_jwk()).unwrap();
        assert!(
            Capability::new(&jws, "", &leaf.private_jwk()).is_err(),
            "a capability comes with its token endpoint"
        );
        let source = Source::new(
            "https://hub.example/",
            None,
            vec!["objects.example".into()],
            Some(capability),
        )
        .unwrap();
        let public = Catalog::new(&source).unwrap();
        assert_eq!(public.origin(), "https://hub.example");
        assert!(public.credential.is_none());
        let read =
            |model, manifest| Catalog::for_op(&source, Op::Read { model, manifest }).unwrap();
        assert!(read("alice/private", &manifest).credential.is_some());
        let other_manifest = format!("sha256:{}", "cd".repeat(32));
        assert!(read("alice/private", &other_manifest).credential.is_none());
        assert!(read("public/model", &manifest).credential.is_none());
        let publish = |model| Catalog::for_op(&source, Op::Publish { model });
        assert!(publish("alice/out").unwrap().credential.is_some());
        assert_eq!(publish("alice/other").err().unwrap().0, reason::REQUIRED);
        let anonymous = Source::new("https://hub.example", None, vec![], None).unwrap();
        assert_eq!(
            Catalog::for_op(&anonymous, Op::Publish { model: "alice/out" })
                .err()
                .unwrap()
                .0,
            reason::REQUIRED
        );
        assert!(Source::new("http://hub.example", None, vec![], None).is_err());
        let other = signed(
            &self::leaf(),
            json!([{"type": "tensorhub_model_upload", "model": "alice/out"}]),
        );
        assert!(
            Capability::new(&other, TOKENS, &leaf.private_jwk()).is_err(),
            "a capability for another machine key"
        );
        assert!(Capability::new("not-a-jws", TOKENS, &leaf.private_jwk()).is_err());
    }

    #[test]
    fn origins_compare_by_scheme_host_and_port() {
        assert_eq!(
            origin_key("https://Hub.Example/"),
            origin_key("HTTPS://hub.example:443")
        );
        assert_ne!(
            origin_key("https://hub.example"),
            origin_key("http://hub.example")
        );
        assert!(valid_origin("http://127.0.0.1:8819"));
        assert!(!valid_origin("http://hub.example"));
        assert!(!valid_origin("https://hub.example/path"));
        assert!(!valid_origin("https://user@hub.example"));
        assert_eq!(
            origin_of("https://hub.example/v1/auth"),
            "https://hub.example"
        );
        assert_eq!(origin_of("https://hub.example"), "https://hub.example");
    }
}
