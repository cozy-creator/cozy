//! The one HTTP client: hyper's pooled HTTP/1.1 client over a connector that dials only the
//! addresses the egress policy checked, TLS (rustls) naming the host. No timeouts here: the
//! downloader judges every request by measured progress and drops the ones that stop.

use std::collections::VecDeque;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Method, Request, Response, Uri};
use tokio::io::AsyncReadExt;
use hyper_util::client::legacy::connect::{Connected, Connection};
use hyper_util::rt::{TokioExecutor, TokioIo};
use rustls::pki_types::{CertificateDer, ServerName};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;
use tokio::task::JoinSet;

use super::policy::{CheckedUrl, SourcePolicy};
use super::token::{Request as Presented, TokenProvider};
use crate::err::{refuse, Code, Refusal, Result};

/// The headers of an answer, for the fetch modules beside the client: hyper is named only here.
pub(crate) use hyper::HeaderMap;

/// RFC 8305's connection attempt delay: the next address is dialed while the last is silent
/// this long. A stagger, never a verdict.
const STAGGER: Duration = Duration::from_millis(250);

/// A GET's head, with its body still on the wire, and the URL that answered it.
pub(crate) struct Answer {
    pub url: String,
    pub response: Response<Incoming>,
    /// When a redirect led here: until when its `Cache-Control: max-age` lets `url` be reused.
    pub until: Option<Instant>,
}

impl Answer {
    pub fn status(&self) -> u16 {
        self.response.status().as_u16()
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.response.headers().get(name)?.to_str().ok()
    }

    /// The wait an answer states (`Retry-After` in seconds).
    pub fn stated_wait(&self) -> Option<Duration> {
        self.header("retry-after")?
            .trim()
            .parse::<u64>()
            .ok()
            .map(Duration::from_secs)
    }

    /// The next frame of body bytes, `None` at the end.
    pub async fn frame(&mut self) -> Result<Option<Bytes>> {
        loop {
            match self.response.body_mut().frame().await {
                None => return Ok(None),
                Some(Ok(frame)) => match frame.into_data() {
                    Ok(data) if !data.is_empty() => return Ok(Some(data)),
                    _ => continue,
                },
                Some(Err(error)) => {
                    return refuse(Code::TRANSFER_FAILED, format!("the body failed: {error}"))
                }
            }
        }
    }
}

/// A request body: bytes, or the first `length` bytes of a file, counted into `sent` as they go.
pub(crate) enum Body {
    Bytes(Bytes),
    File {
        file: std::fs::File,
        length: u64,
        sent: Arc<AtomicU64>,
    },
}

impl Body {
    fn boxed(self) -> BoxBody<Bytes, std::io::Error> {
        let (file, length, sent) = match self {
            Body::Bytes(bytes) => return Full::new(bytes).map_err(|never| match never {}).boxed(),
            Body::File { file, length, sent } => (tokio::fs::File::from_std(file), length, sent),
        };
        let frames = futures::stream::unfold((file, length), move |(mut file, left)| {
            let sent = Arc::clone(&sent);
            async move {
                if left == 0 {
                    return None;
                }
                let mut buf = vec![0; left.min(1 << 20) as usize];
                let (frame, left) = match file.read(&mut buf).await {
                    Ok(0) => (Err(std::io::ErrorKind::UnexpectedEof.into()), 0),
                    Ok(n) => {
                        buf.truncate(n);
                        sent.fetch_add(n as u64, Ordering::Relaxed);
                        (Ok(Frame::data(Bytes::from(buf))), left - n as u64)
                    }
                    Err(error) => (Err(error), 0),
                };
                Some((frame, (file, left)))
            }
        });
        StreamBody::new(frames).boxed()
    }
}

/// A POST's answer, read whole.
pub(crate) struct Posted {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

/// The client every fetch uses. Cheap to clone: clones share one connection pool.
#[derive(Clone)]
pub struct Client {
    inner: hyper_util::client::legacy::Client<Connector, BoxBody<Bytes, std::io::Error>>,
    policy: SourcePolicy,
}

impl Client {
    /// A client bound by `policy`, trusting the webpki roots.
    pub fn new(policy: SourcePolicy) -> Client {
        Client::trusting(policy, &[])
    }

    /// A client that also trusts the CA certificates in `pem` (a private origin's).
    pub fn with_roots(policy: SourcePolicy, pem: &[u8]) -> Result<Client> {
        let roots: Vec<CertificateDer<'static>> =
            rustls::pki_types::pem::PemObject::pem_slice_iter(pem)
                .collect::<std::result::Result<_, _>>()
                .map_err(|error| Refusal {
                    code: Code::UNKNOWN_FORMAT,
                    detail: format!("not a PEM certificate bundle: {error}"),
                })?;
        if roots.is_empty() {
            return refuse(Code::UNKNOWN_FORMAT, "the PEM bundle holds no certificate");
        }
        Ok(Client::trusting(policy, &roots))
    }

    fn trusting(policy: SourcePolicy, extra: &[CertificateDer<'static>]) -> Client {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        for root in extra {
            let _ = roots.add(root.clone());
        }
        let tls = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .expect("rustls ships its default protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
        let connector = Connector {
            policy: policy.clone(),
            tls: Arc::new(tls),
        };
        let inner =
            hyper_util::client::legacy::Client::builder(TokioExecutor::new()).build(connector);
        Client { inner, policy }
    }

    pub fn policy(&self) -> &SourcePolicy {
        &self.policy
    }

    /// GET `url`, following redirects the policy allows, each hop checked again. Credentials
    /// go only to the origin they were given for: another origin gets only the `range`.
    pub(crate) async fn get_authorized(&self, url: &str, headers: &[(String, String)], mut provider: Option<&dyn TokenProvider>) -> Result<Answer> {
        let mut url = url.to_string();
        let mut headers = headers.to_vec();
        let (mut hops, mut until) = (0, None::<Instant>);
        let mut challenged = false;
        loop {
            let mut presented = headers.clone();
            if let Some(provider) = provider {
                presented.extend(provider.headers(&Presented { method: "GET", url: &url, headers: &headers, payload: None }).await?);
            }
            let response = self.send("GET", &url, &presented, None).await?.response;
            let uri: Uri = url.parse().expect("sent");
            let status = response.status().as_u16();
            let sent = Presented { method: "GET", url: &url, headers: &presented, payload: None };
            if provider.is_some_and(|provider| provider.retry(&sent, status, response.headers())) && !challenged {
                challenged = true;
                continue;
            }
            if matches!(status, 301 | 302 | 303 | 307 | 308) {
                let Some(next) = response
                    .headers()
                    .get("location")
                    .and_then(|v| v.to_str().ok())
                else {
                    return refuse(
                        Code::REDIRECT_REFUSED,
                        format!("{url}: {status} without a Location"),
                    );
                };
                let next = join(&uri, next);
                hops += 1;
                if hops > self.policy.max_redirects {
                    return refuse(
                        Code::REDIRECT_REFUSED,
                        format!(
                            "{url}: redirected to {next}, past the policy's {} hops",
                            self.policy.max_redirects
                        ),
                    );
                }
                if !same_origin(&url, &next) {
                    headers.retain(|(name, _)| name.eq_ignore_ascii_case("range"));
                    provider = None;
                }
                let age = max_age(response.headers()).map(|age| Instant::now() + age);
                until = match (until, age) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                url = next;
                challenged = false;
                continue;
            }
            if let Some(encoding) = response
                .headers()
                .get("content-encoding")
                .and_then(|v| v.to_str().ok())
            {
                if !encoding.trim().is_empty() && !encoding.trim().eq_ignore_ascii_case("identity")
                {
                    return refuse(Code::TRANSFER_FAILED, format!("{url}: the origin sent content-encoding {encoding:?} against `accept-encoding: identity`"));
                }
            }
            return Ok(Answer { url, response, until });
        }
    }

    /// One request, redirects not followed: `body` (none: empty) with `headers`.
    pub(crate) async fn send(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        body: Option<Body>,
    ) -> Result<Answer> {
        self.policy.admit(url)?;
        let bad = |detail: String| Refusal {
            code: Code::SOURCE_NOT_ALLOWED,
            detail,
        };
        let uri: Uri = url.parse().map_err(|_| bad(format!("{url}: not a URL")))?;
        let method = Method::from_bytes(method.as_bytes()).map_err(|e| bad(format!("{method}: {e}")))?;
        let mut request = Request::builder()
            .method(method)
            .uri(uri)
            .header("accept-encoding", "identity");
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }
        let body = body.unwrap_or(Body::Bytes(Bytes::new())).boxed();
        let request = request.body(body).map_err(|e| bad(format!("{url}: {e}")))?;
        let response = self
            .inner
            .request(request)
            .await
            .map_err(|error| refusal(url, &error))?;
        Ok(Answer {
            url: url.to_string(),
            response,
            until: None,
        })
    }

    /// POST `body` (JSON) to `url`, credentials in `headers`; redirects are not followed. The
    /// answer is read whole, at most `cap` bytes.
    pub(crate) async fn post(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: Vec<u8>,
        cap: u64,
    ) -> Result<Posted> {
        let mut headers = headers.to_vec();
        headers.push(("content-type".into(), "application/json".into()));
        let mut answer = (self.send("POST", url, &headers, Some(Body::Bytes(body.into())))).await?;
        let (status, headers) = (answer.status(), answer.response.headers().clone());
        let mut body = Vec::new();
        while let Some(frame) = answer.frame().await? {
            body.extend_from_slice(&frame);
            if body.len() as u64 > cap {
                return refuse(Code::SIZE_CAP, format!("{url}: over {cap} B"));
            }
        }
        Ok(Posted {
            status,
            headers,
            body,
        })
    }
}

pub(crate) fn same_origin(a: &str, b: &str) -> bool {
    let origin = |url: &str| url.parse::<Uri>().ok().map(|url| (url.scheme_str().map(str::to_owned), url.authority().cloned()));
    origin(a) == origin(b)
}

/// A response's `Cache-Control: max-age`.
pub(crate) fn max_age(headers: &HeaderMap) -> Option<Duration> {
    let control = headers.get("cache-control")?.to_str().ok()?;
    let age = control.split(',').find_map(|d| d.trim().strip_prefix("max-age="))?;
    age.trim().parse().ok().map(Duration::from_secs)
}

/// `location` against the URL that answered with it.
pub(crate) fn join(base: &Uri, location: &str) -> String {
    if location.starts_with("http://") || location.starts_with("https://") {
        return location.to_string();
    }
    let scheme = base.scheme_str().unwrap_or("https");
    if let Some(rest) = location.strip_prefix("//") {
        return format!("{scheme}://{rest}");
    }
    let authority = base.authority().map_or("", |a| a.as_str());
    if location.starts_with('/') {
        return format!("{scheme}://{authority}{location}");
    }
    let dir = base.path().rsplit_once('/').map_or("", |(dir, _)| dir);
    format!("{scheme}://{authority}{dir}/{location}")
}

/// A failed exchange as a refusal: the connector's own refusal where it made one.
fn refusal(url: &str, error: &hyper_util::client::legacy::Error) -> Refusal {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(cause) = source {
        if let Some(refusal) = cause.downcast_ref::<Refusal>() {
            return refusal.clone();
        }
        source = cause.source();
    }
    Refusal {
        code: Code::TRANSFER_FAILED,
        detail: format!("{url}: {error}"),
    }
}

/// Dials the checked addresses of a URL's host, then TLS when the URL is https.
#[derive(Clone)]
struct Connector {
    policy: SourcePolicy,
    tls: Arc<rustls::ClientConfig>,
}

impl tower_service::Service<Uri> for Connector {
    type Response = TokioIo<Stream>;
    type Error = Refusal;
    type Future = Pin<Box<dyn Future<Output = Result<TokioIo<Stream>>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        let (policy, tls) = (self.policy.clone(), Arc::clone(&self.tls));
        Box::pin(async move {
            let (url, checking) = (uri.to_string(), policy.clone());
            let checked = tokio::task::spawn_blocking(move || checking.check(&url))
                .await
                .map_err(|error| Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("name resolution: {error}"),
                })??;
            let socket = dial(&checked).await;
            if socket.is_err() {
                // Answers nothing connects to are asked of the resolver again next time.
                policy.resolved.forget(&checked.host, checked.port);
            }
            let socket = socket?;
            if !checked.https {
                return Ok(TokioIo::new(Stream::Plain(socket)));
            }
            let name = ServerName::try_from(checked.host.clone()).map_err(|_| Refusal {
                code: Code::SOURCE_NOT_ALLOWED,
                detail: format!("{}: not a TLS server name", checked.host),
            })?;
            let tls = tokio_rustls::TlsConnector::from(tls)
                .connect(name, socket)
                .await
                .map_err(|error| tls_refusal(&checked, error))?;
            Ok(TokioIo::new(Stream::Tls(Box::new(tls))))
        })
    }
}

/// The first checked address that accepts a TCP connection. Another is dialed whenever the
/// newest has been silent a stagger, families alternating (RFC 8305).
async fn dial(checked: &CheckedUrl) -> Result<TcpStream> {
    let first = checked.addrs.first().is_some_and(SocketAddr::is_ipv4);
    let (mut family, mut other): (VecDeque<SocketAddr>, VecDeque<SocketAddr>) = checked
        .addrs
        .iter()
        .partition(|addr| addr.is_ipv4() == first);
    let mut order = Vec::with_capacity(checked.addrs.len());
    while !family.is_empty() || !other.is_empty() {
        order.extend(family.pop_front());
        order.extend(other.pop_front());
    }
    let mut addrs = order.into_iter();
    let mut pending: JoinSet<std::io::Result<TcpStream>> = JoinSet::new();
    let mut last = String::new();
    let mut fd_full = false;
    loop {
        let more = match addrs.next() {
            Some(addr) => {
                pending.spawn(TcpStream::connect(addr));
                true
            }
            None => false,
        };
        if pending.is_empty() {
            let code = if fd_full {
                Code::FD_HEADROOM
            } else {
                Code::TRANSFER_FAILED
            };
            return refuse(
                code,
                format!(
                    "{}: no checked address accepted a connection ({last})",
                    checked.host
                ),
            );
        }
        let joined = match more {
            true => match tokio::time::timeout(STAGGER, pending.join_next()).await {
                Ok(joined) => joined,
                Err(_) => continue,
            },
            false => pending.join_next().await,
        };
        match joined {
            Some(Ok(Ok(socket))) => {
                let _ = socket.set_nodelay(true);
                return Ok(socket);
            }
            Some(Ok(Err(error))) => {
                fd_full |= crate::descriptors::exhausted(&error);
                last = error.to_string();
            }
            Some(Err(error)) => last = error.to_string(),
            None => {}
        }
    }
}

/// A certificate or protocol verdict is final (`TLS_UNTRUSTED`); anything else is the link's.
fn tls_refusal(checked: &CheckedUrl, error: std::io::Error) -> Refusal {
    let verdict = (error.get_ref())
        .and_then(|inner| inner.downcast_ref::<rustls::Error>())
        .is_some_and(|error| {
            matches!(
                error,
                rustls::Error::InvalidCertificate(_)
                    | rustls::Error::NoCertificatesPresented
                    | rustls::Error::UnsupportedNameType
                    | rustls::Error::PeerIncompatible(_)
            )
        });
    match verdict {
        true => Refusal {
            code: Code::TLS_UNTRUSTED,
            detail: format!("{}: TLS refused: {error}", checked.host),
        },
        false => Refusal {
            code: Code::TRANSFER_FAILED,
            detail: format!("{}: TLS handshake failed: {error}", checked.host),
        },
    }
}

/// A connection, plain or TLS.
enum Stream {
    Plain(TcpStream),
    Tls(Box<tokio_rustls::client::TlsStream<TcpStream>>),
}

impl Connection for Stream {
    fn connected(&self) -> Connected {
        Connected::new()
    }
}

impl AsyncRead for Stream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Stream::Plain(socket) => Pin::new(socket).poll_read(cx, buf),
            Stream::Tls(socket) => Pin::new(socket.as_mut()).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for Stream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            Stream::Plain(socket) => Pin::new(socket).poll_write(cx, buf),
            Stream::Tls(socket) => Pin::new(socket.as_mut()).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Stream::Plain(socket) => Pin::new(socket).poll_flush(cx),
            Stream::Tls(socket) => Pin::new(socket.as_mut()).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Stream::Plain(socket) => Pin::new(socket).poll_shutdown(cx),
            Stream::Tls(socket) => Pin::new(socket.as_mut()).poll_shutdown(cx),
        }
    }
}
