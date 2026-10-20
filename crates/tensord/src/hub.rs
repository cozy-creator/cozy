//! Hub access (th-245). A run reads its Hub's package catalog anonymously, and its models through
//! TensorFS's distribution `Endpoint` (`origin`) under its owner's newest ordinary bearer when its
//! request carried one, or anonymously without one.
use base64::Engine as _;
use hub_client::Credential;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use crate::device_executor::{DeviceCommand, Line};
use protocol::machine::TensorFSOptions;
use tensorfs::{
    err::{refuse, Code},
    fetch::token::{HeaderMap, Request, TokenProvider},
};

pub use hub_client::types;

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
/// current-thread runtime, made once. A thread already inside an async context hands the call
/// to a helper thread instead.
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

/// One signer's newest ordinary bearer at one Hub: every read under its owner's access presents
/// the one held when it is sent, and each executor given it for a request hears of each newer
/// one (`DeviceCommand::Bearer`). In memory only, for the life of the process.
pub struct Bearer {
    actor: String,
    held: Mutex<(String, i64)>,
    /// The executors given it, each with the endpoint its requests named.
    lines: Mutex<Vec<(Line, String)>>,
}

impl Bearer {
    /// `actor`'s at `origin`, offered `token` (until `expires_unix_ms`; 0: unknown), which it
    /// keeps unless the one it holds expires later. None without a token or a Hub origin.
    pub fn offer(actor: &str, origin: &str, token: &str, expires_unix_ms: i64) -> Option<Arc<Bearer>> {
        static HELD: Mutex<BTreeMap<(String, String), Arc<Bearer>>> = Mutex::new(BTreeMap::new());
        let key = (actor.to_string(), origin_key(origin).filter(|_| !token.is_empty())?);
        let bearer = HELD.lock().unwrap().entry(key).or_insert_with(|| Arc::new(Bearer {
            actor: actor.into(), held: Mutex::default(), lines: Mutex::default(),
        })).clone();
        let mut held = bearer.held.lock().unwrap();
        let newer = (expires_unix_ms == 0 || expires_unix_ms >= held.1) && held.0 != token;
        if newer {
            *held = (token.to_string(), expires_unix_ms);
        }
        drop(held);
        if newer {
            bearer.forward();
        }
        Some(bearer)
    }

    /// The token held now, and when it expires.
    pub fn held(&self) -> (String, i64) {
        self.held.lock().unwrap().clone()
    }

    /// The token held now for a request `line`'s executor serves at `endpoint`, which hears of
    /// each newer one from now on.
    pub fn given(&self, line: Line, endpoint: &str) -> (String, i64) {
        let mut lines = self.lines.lock().unwrap();
        if !lines.iter().any(|(given, at)| given.same(&line) && at == endpoint) {
            lines.push((line, endpoint.to_string()));
        }
        self.held()
    }

    /// Each executor given it hears of the token held now; a gone one is forgotten.
    fn forward(&self) {
        let (token, expires_unix_ms) = self.held();
        self.lines.lock().unwrap().retain(|(line, endpoint)| {
            line.send(&DeviceCommand::Bearer {
                actor: self.actor.clone(),
                tensorfs: TensorFSOptions { endpoint: endpoint.clone(), token: token.clone(), expires_unix_ms, ..Default::default() },
            })
        });
    }
}

/// `unix_ms` as UTC, to the second.
fn utc(unix_ms: i64) -> String {
    match time::OffsetDateTime::from_unix_timestamp(unix_ms / 1000) {
        Ok(t) => format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", t.year(), t.month() as u8, t.day(), t.hour(), t.minute(), t.second()),
        Err(_) => format!("unix {unix_ms} ms"),
    }
}

#[async_trait::async_trait]
impl TokenProvider for Bearer {
    async fn headers(&self, _: &Request<'_>) -> tensorfs::err::Result<Vec<(String, String)>> {
        Ok(vec![("authorization".into(), format!("Bearer {}", self.held().0))])
    }

    /// A refused bearer that a newer one replaced since is asked once more with that; the newest
    /// refused fails the read, naming when it expires.
    fn retry(&self, sent: &Request<'_>, status: u16, _: &HeaderMap) -> tensorfs::err::Result<bool> {
        if status != 401 {
            return Ok(false);
        }
        let (token, expires) = self.held();
        let presented = format!("Bearer {token}");
        if !sent.headers.iter().any(|(name, value)| name.eq_ignore_ascii_case("authorization") && *value == presented) {
            return Ok(true);
        }
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64);
        refuse(Code::CREDENTIAL_REQUIRED, match expires {
            0 => "the Hub refused the owner's bearer".to_string(),
            at if at <= now => format!("the owner's bearer expired at {}", utc(at)),
            at => format!("the Hub refused the owner's bearer, which expires at {}", utc(at)),
        })
    }
}

/// Where a run reads a Hub, and its owner's bearer there when its request carried one.
#[derive(Clone)]
pub struct Source {
    pub origin: String,
    pub ca_der: Option<Vec<u8>>,
    pub bearer: Option<Arc<Bearer>>,
}

impl Source {
    pub fn new(origin: &str, ca_der: Option<Vec<u8>>, bearer: Option<Arc<Bearer>>) -> Result<Self, Refusal> {
        if !valid_origin(origin) {
            return Err(Refusal(format!("{origin} is not an HTTPS origin")));
        }
        Ok(Self { origin: origin.trim_end_matches('/').to_string(), ca_der, bearer })
    }

    /// Its bearer, for `endpoint` only when that is this Hub.
    pub fn bearer_at(&self, endpoint: &str) -> Option<Arc<Bearer>> {
        let here = origin_key(endpoint).is_some_and(|key| origin_key(&self.origin) == Some(key));
        self.bearer.clone().filter(|_| here)
    }
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source")
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

/// A Hub's package catalog, read anonymously.
pub struct Catalog {
    origin: String,
    hub: hub_client::Client,
}

impl Catalog {
    pub fn new(source: &Source) -> Result<Self, Refusal> {
        let http = http(source.ca_der.as_deref())?;
        let hub = hub_client::Client::connect(&source.origin, http, Credential::Anonymous);
        Ok(Self { origin: source.origin.clone(), hub })
    }

    pub fn origin(&self) -> &str {
        &self.origin
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One bearer per signer and Hub: the one that expires latest, never shown in a Source.
    #[test]
    fn a_signer_keeps_its_newest_bearer_per_hub() {
        let offer = |actor, origin, token, expires| Bearer::offer(actor, origin, token, expires).map(|b| b.held());
        assert_eq!(offer("a", "https://hub.example/", "first", 2), Some(("first".into(), 2)));
        assert_eq!(offer("a", "HTTPS://hub.example:443", "older", 1), Some(("first".into(), 2)));
        assert_eq!(offer("a", "https://hub.example", "newer", 3), Some(("newer".into(), 3)));
        assert_eq!(offer("b", "https://hub.example", "other", 1), Some(("other".into(), 1)));
        assert_eq!(offer("a", "https://hub.example", "", 4), None);
        let bearer = Bearer::offer("a", "https://hub.example", "newer", 3);
        let source = Source::new("https://hub.example/", None, bearer).unwrap();
        assert_eq!(source.origin, "https://hub.example");
        assert!(source.bearer_at("https://hub.example:443").is_some());
        assert!(source.bearer_at("https://other.example").is_none());
        assert!(!format!("{source:?}").contains("newer"));
        assert!(Source::new("http://hub.example", None, None).is_err());
        assert_eq!(utc(1_792_480_500_000), "2026-10-20T07:15:00Z");
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
