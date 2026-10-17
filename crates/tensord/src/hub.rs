//! Hub access (th-241, th-245). Released content is read anonymously, by name. A run's private
//! operations (reading one of the owner's unpublished checkpoints, publishing to the owner's
//! repositories) are named by a capability the owner's CLI signed with its device key for this
//! machine's TLS leaf. At the run's first private operation the machine trades it, once, for a
//! DPoP-bound token (hub-client's `RunToken`: AuthKit's JWT-bearer grant, an assertion the leaf
//! signs). TensorFS's mover carries the bytes; it presents the token, proved per request, to
//! the Hub host only.
use base64::Engine as _;
use hub_client::{Credential, Ended, P256Key, RunToken};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tensorfs_core::transport::{
    self, Anonymous, Answer, Ask, Client, Closure, CredentialProvider, Deadline, Ledger,
    SourcePolicy, SAMPLE_SECONDS, STILL_SAMPLES,
};

pub use hub_client::{types, Op};
pub use publication::{Publication, Published};

mod publication;

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

/// Where its published files are read: the file door answers with one redirect to its
/// object store.
pub fn files_policy(source: &Source) -> Result<SourcePolicy, Refusal> {
    Ok(SourcePolicy {
        max_redirects: 1,
        ..Catalog::new(source)?.policy
    })
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
fn block<T: Send>(call: impl Future<Output = T> + Send) -> T {
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
/// merely slow is waited for; a dead one is found by TCP keepalive at the transport's own
/// resolution (probes after its noise floor of silence, a sample apart), as TensorFS's are.
pub fn http(ca_der: Option<&[u8]>) -> Result<reqwest::Client, Refusal> {
    static CLIENTS: Mutex<Vec<(Option<Vec<u8>>, reqwest::Client)>> = Mutex::new(Vec::new());
    let mut clients = CLIENTS.lock().unwrap();
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
    let client = reqwest::Client::builder()
        .tls_backend_preconfigured(tls)
        .redirect(reqwest::redirect::Policy::none())
        .tcp_keepalive(Duration::from_secs_f64(SAMPLE_SECONDS) * STILL_SAMPLES)
        .tcp_keepalive_interval(Duration::from_secs_f64(SAMPLE_SECONDS))
        .tcp_keepalive_retries(STILL_SAMPLES)
        .build()
        .map_err(|e| unusable(e.to_string()))?;
    clients.push((ca_der.map(<[u8]>::to_vec), client.clone()));
    Ok(client)
}

/// The machine's TLS leaf key (P-256): it proves DPoP and signs JWT-bearer assertions. A
/// capability names its thumbprint (`cnf.jkt`).
pub struct Leaf(Arc<P256Key>);

impl Leaf {
    /// The identity's PKCS#8 PEM (`-----BEGIN PRIVATE KEY-----`).
    pub fn from_pem(pem: &str) -> Result<Self, Refusal> {
        P256Key::from_pem(pem)
            .map(|key| Self(Arc::new(key)))
            .map_err(|e| Refusal(format!("the machine key: {e}")))
    }

    /// RFC 7638 thumbprint: a capability's and every token's `cnf.jkt`.
    pub fn thumbprint(&self) -> String {
        self.0.thumbprint()
    }
}

/// The owner's capability for one run (th-241), and the one token it trades for. Child runs
/// share it. Nothing is asked of the Hub until a private operation needs it.
pub struct Capability {
    run: Arc<RunToken>,
}

impl Capability {
    /// `jws` as the run spec carries it, traded at `token_endpoint`. Its claims are the CLI's
    /// statement, read here only to know which operations are private, for which Hub and until
    /// when; AuthKit and the Hub verify it.
    pub fn new(jws: &str, token_endpoint: &str, leaf: Arc<Leaf>) -> Result<Self, Refusal> {
        if !valid_url(token_endpoint) {
            return Err(Refusal(
                "the run's capability comes with no token endpoint".into(),
            ));
        }
        let run = RunToken::new(jws, token_endpoint, leaf.0.clone())
            .map_err(|e| Refusal(e.to_string()))?;
        if !valid_origin(run.resource()) {
            return Err(Refusal("the run's capability names no Hub origin".into()));
        }
        Ok(Self { run: Arc::new(run) })
    }

    /// Whether the capability names `op`.
    pub fn names(&self, op: Op) -> bool {
        self.run.names(op)
    }

    /// Why the capability no longer serves, if it does not.
    fn ended(&self) -> Option<(&'static str, String)> {
        self.run.ended().map(|ended| match ended {
            Ended::Expired => (reason::EXPIRED, ended.to_string()),
            Ended::Refused(_) => (reason::REFUSED, ended.to_string()),
        })
    }

    /// The token, traded over `http` on first need; none while it cannot be had.
    fn token(&self, http: &reqwest::Client) -> Option<String> {
        block(self.run.token(http)).ok()
    }
}

impl std::fmt::Debug for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Capability").finish_non_exhaustive()
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

/// The capability's token, as TensorFS's mover presents it: to the Hub host only, with a fresh
/// proof per request naming the Hub's public origin.
struct Presenter {
    capability: Arc<Capability>,
    host: String,
    http: reqwest::Client,
    /// The Hub answered a token-bearing request 403: the operation is outside the token.
    forbidden: AtomicBool,
}

impl CredentialProvider for Presenter {
    fn headers(&self, ask: &Ask) -> Vec<(String, String)> {
        if !ask.url.host.eq_ignore_ascii_case(&self.host) {
            return Vec::new();
        }
        let Some(token) = self.capability.token(&self.http) else {
            return Vec::new();
        };
        let path = ask
            .url
            .target
            .split(['?', '#'])
            .next()
            .filter(|p| !p.is_empty())
            .unwrap_or("/");
        let presented = self
            .capability
            .run
            .present(&token, ask.method, &self.host, path);
        presented
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect()
    }

    fn answered(&self, ask: &Ask, answer: &Answer) -> bool {
        let token = answer
            .presented("authorization")
            .and_then(|v| v.strip_prefix("DPoP "));
        let Some(token) = token.filter(|_| ask.url.host.eq_ignore_ascii_case(&self.host)) else {
            return false;
        };
        if answer.status == 403 {
            self.forbidden.store(true, Ordering::Relaxed);
        }
        let challenged = answer
            .header("www-authenticate")
            .is_some_and(|v| v.contains("use_dpop_nonce"));
        self.capability.run.observe(
            &self.host,
            token,
            answer.status,
            answer.header("dpop-nonce"),
            challenged,
        )
    }
}

/// Reads and publications at a Hub, credentials presented only to the Hub host; presigned
/// object hosts see nothing. Anonymous unless built for an operation the capability names.
pub struct Catalog {
    origin: String,
    host: String,
    hub: hub_client::Client,
    /// The Hub as the run's capability, for an operation it names.
    run: Option<hub_client::Client>,
    credential: Option<Presenter>,
    policy: SourcePolicy,
}

impl Catalog {
    /// Public reads: anonymous.
    pub fn new(source: &Source) -> Result<Self, Refusal> {
        let origin = source.origin.clone();
        let host = transport::base_host(&origin).map_err(|e| Refusal(e.to_string()))?;
        if let Some(der) = &source.ca_der {
            transport::trust_roots(pem(der).as_bytes()).map_err(|e| Refusal(e.to_string()))?;
        }
        let mut allowed = vec![host.clone()];
        allowed.extend(source.object_hosts.iter().cloned());
        let policy = SourcePolicy {
            allowed_hosts: allowed,
            allow_local: origin_key(&origin).is_some_and(|key| loopback_host(&key)),
            ..Default::default()
        };
        let hub = hub_client::Client::connect(
            &origin,
            http(source.ca_der.as_deref())?,
            Credential::Anonymous,
        );
        Ok(Self {
            origin,
            host,
            hub,
            run: None,
            credential: None,
            policy,
        })
    }

    /// `op` under the run's capability when it names it; a read it does not name is public.
    /// A publication it does not name is refused before the Hub hears of it. The token is
    /// traded here, so a capability the Hub refused, or that expired, ends the operation
    /// before any request goes out without it.
    pub fn for_op(source: &Source, op: Op) -> Result<Self, (&'static str, String)> {
        let catalog = Self::scoped(source, op)?;
        if let Some(presenter) = &catalog.credential {
            if presenter.capability.token(&presenter.http).is_none() {
                if let Some(ended) = presenter.capability.ended() {
                    return Err(ended);
                }
            }
        }
        Ok(catalog)
    }

    /// `for_op` with nothing traded until a request needs the token (a fetch whose objects
    /// are all here asks nothing).
    pub fn scoped(source: &Source, op: Op) -> Result<Self, (&'static str, String)> {
        let mut catalog = Self::new(source).map_err(|e| ("catalog_read_failed", e.0))?;
        match &source.capability {
            Some(capability) if capability.names(op) => {
                let http = http(source.ca_der.as_deref()).map_err(|e| ("catalog_read_failed", e.0))?;
                let credential = Credential::Run(capability.run.clone());
                catalog.run = Some(hub_client::Client::connect(&source.origin, http.clone(), credential));
                catalog.credential = Some(Presenter {
                    capability: capability.clone(),
                    host: catalog.host.clone(),
                    http,
                    forbidden: AtomicBool::new(false),
                });
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
    pub fn credential(&self) -> &dyn CredentialProvider {
        match &self.credential {
            Some(presenter) => presenter,
            None => &Anonymous,
        }
    }
    pub fn policy(&self) -> &SourcePolicy {
        &self.policy
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
            let plan = self.hub.get_package_download_plan(org, name, python_version, Some(release), &empty).await?;
            Ok(plan.into_inner())
        })
    }

    /// What `refspec` (`org/name[@release][@sha256:<hex>]`) and `lane` name at the Hub: the
    /// first page of its closure, which carries the release, lane and manifest it resolved.
    pub fn closure(
        &self,
        refspec: &str,
        lane: &str,
    ) -> Result<Closure, tensorfs_core::err::Refusal> {
        transport::closure(
            &Client::new(),
            &self.origin,
            refspec,
            lane,
            "",
            self.credential(),
            &self.policy,
            Deadline::none(),
            &Ledger::new(),
        )
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
        if let Some(ended) = presenter.capability.ended() {
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
    use serde_json::{json, Value};

    fn leaf() -> Arc<Leaf> {
        let pem = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
            .unwrap()
            .serialize_pem();
        Arc::new(Leaf::from_pem(&pem).unwrap())
    }

    fn signed(leaf: &Leaf, ops: Value) -> String {
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
                {"type": "tensorhub_model_read", "model": "alice/private", "manifest": manifest},
                {"type": "tensorhub_model_publish", "model": "alice/out"},
            ]),
        );
        // Nothing answers here: a trade goes unanswered and the operation keeps its capability.
        const TOKENS: &str = "http://127.0.0.1:9/v1/auth/oauth2/token";
        let capability = Capability::new(&jws, TOKENS, leaf.clone()).unwrap();
        assert!(
            Capability::new(&jws, "", leaf.clone()).is_err(),
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
        assert_eq!(
            (public.origin(), public.host.as_str()),
            ("https://hub.example", "hub.example")
        );
        assert!(public.policy().allows_host("objects.example") && public.credential.is_none());
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
            json!([{"type": "tensorhub_model_publish", "model": "alice/out"}]),
        );
        assert!(
            Capability::new(&other, TOKENS, leaf.clone()).is_err(),
            "a capability for another machine key"
        );
        assert!(Capability::new("not-a-jws", TOKENS, leaf).is_err());
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
