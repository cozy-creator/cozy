//! The transport's async half: every object GET in flight is a task on one small runtime, so a
//! download's requests cost sockets and memory, never threads (a vast pod's container caps its
//! tasks at 1,280; 512 stream threads per pull hit it and wedged, 2026-10-15).
//!
//! The rules are `http`'s, kept exactly: the connection goes to the addresses the predicate
//! CHECKED, with SNI and Host carrying the name; every wait is judged by the ledger (`Judge`),
//! never by an invented constant, its tick only the sampling resolution; a connection is
//! reused only by its origin and the credentials sent on it. HTTP/1.1 framing is hyper's.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use bytes::Bytes;
use http_body_util::{BodyExt, Empty};
use hyper::client::conn::http1::SendRequest;
use hyper_util::rt::TokioIo;
use rustls::pki_types::ServerName;
use tokio::net::TcpStream;
use tokio::task::JoinSet;

use super::http::{self, Client, Judge, Scope};
use super::ledger::Ledger;
use super::policy::CheckedUrl;
use crate::err::{refuse, Code, Refusal, Result};

/// Threads its own runtime polls sockets and timers on, whatever the number of requests.
const NET_THREADS: usize = 2;

/// The runtime its host process lent it ([`use_runtime`]: TensorD's).
static HOST: OnceLock<tokio::runtime::Handle> = OnceLock::new();

/// Runs every later request on `runtime`, the host process's own, instead of the transport's.
pub fn use_runtime(runtime: tokio::runtime::Handle) {
    let _ = HOST.set(runtime);
}

/// The process's one network runtime: its host's, else the transport's own (the wheel's
/// extension, `tfs`). One that cannot start (its threads refused) is asked for again by the next
/// request, never remembered as broken.
pub(crate) fn runtime() -> Result<tokio::runtime::Handle> {
    if let Some(host) = HOST.get() {
        return Ok(host.clone());
    }
    static RUNTIME: Mutex<Option<tokio::runtime::Runtime>> = Mutex::new(None);
    let mut held = RUNTIME.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(runtime) = held.as_ref() {
        return Ok(runtime.handle().clone());
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(NET_THREADS)
        .max_blocking_threads(1)
        .thread_name("tensorfs-net")
        .enable_io()
        .enable_time()
        .build()
        .map_err(|error| Refusal {
            code: Code::TRANSFER_FAILED,
            detail: format!("the network runtime could not start: {error}"),
        })?;
    Ok(held.insert(runtime).handle().clone())
}

/// `future` to its end on the calling thread, its sockets and timers driven by the runtime.
/// Callable from any thread, a runtime's own included.
pub(crate) fn block_on<T>(future: impl Future<Output = T>) -> Result<T> {
    let runtime = runtime()?;
    let _entered = runtime.enter();
    Ok(futures::executor::block_on(future))
}

/// A socket kept for the next ask to the same origin with the same credentials.
struct Idle {
    scope: Scope,
    peer: SocketAddr,
    sender: SendRequest<Empty<Bytes>>,
}

static IDLE: Mutex<Vec<Idle>> = Mutex::new(Vec::new());

/// What idle connections may number: one per request the machine keeps in flight.
const MAX_IDLE: usize = super::PULL_STREAMS;

/// An answer's head, and its body still on the wire.
pub(crate) struct Response<'l> {
    pub status: u16,
    headers: Vec<(String, String)>,
    pub body: Body<'l>,
}

impl Response<'_> {
    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        (self.headers.iter())
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    /// The wait this answer states (`Retry-After`), in the origin's own clock.
    pub fn stated_wait(&self) -> Option<Duration> {
        http::stated_wait(&self.headers)
    }

    /// A small non-artifact body (a redirect's, a refusal's) read to its end, so the
    /// connection serves the next ask. A larger one is left, and its connection closes.
    pub async fn discard(mut self) {
        let mut left: usize = 64 * 1024;
        while left > 0 {
            match self.body.chunk().await {
                Ok(Some(data)) => left = left.saturating_sub(data.len()),
                _ => return,
            }
        }
    }
}

/// A response body, read in the frames it arrives in. Each is counted into the ledger where
/// it crosses into this process; when `judge` is armed, silence beyond the pull's measured
/// patience is a stall.
pub(crate) struct Body<'l> {
    incoming: hyper::body::Incoming,
    ledger: &'l Ledger,
    judge: Judge,
    since: Instant,
    recycle: Option<Idle>,
}

impl Body<'_> {
    /// The next frame of bytes, or `None` at the body's end.
    pub async fn chunk(&mut self) -> Result<Option<Bytes>> {
        // Time the consumer spent writing the last frame is not network silence.
        let advanced = Instant::now();
        loop {
            self.ledger.check_running()?;
            match tokio::time::timeout(self.ledger.tick(), self.incoming.frame()).await {
                Ok(Some(Ok(frame))) => {
                    let Ok(data) = frame.into_data() else { continue };
                    if data.is_empty() {
                        continue;
                    }
                    let now = Instant::now();
                    match self.judge {
                        Judge::Stream => {
                            self.ledger.taught(now - advanced);
                            self.ledger.moved(data.len() as u64);
                        }
                        _ => self.ledger.answered(),
                    }
                    return Ok(Some(data));
                }
                Ok(Some(Err(error))) => {
                    self.recycle = None;
                    return refuse(Code::TRANSFER_FAILED, format!("the connection failed mid-body: {error}"));
                }
                Ok(None) => {
                    if let Some(idle) = self.recycle.take() {
                        keep(idle);
                    }
                    return Ok(None);
                }
                Err(_) => {
                    http::transfer_verdict(self.judge, self.ledger, self.since, "a response body")?;
                    let silent = advanced.elapsed();
                    if self.judge.silent_is_stall() && silent > self.ledger.patience() {
                        self.recycle = None;
                        return refuse(
                            Code::TRANSFER_FAILED,
                            format!(
                                "the stream stopped moving bytes: silent {:.1} s against the {:.1} s \
                                 this pull's own measured pace allows",
                                silent.as_secs_f64(),
                                self.ledger.patience().as_secs_f64(),
                            ),
                        );
                    }
                }
            }
        }
    }
}

/// A full descriptor table: idle connections give their sockets back.
pub(crate) fn shed() {
    IDLE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
}

fn keep(idle: Idle) {
    let mut held = IDLE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if held.len() >= MAX_IDLE {
        held.remove(0);
    }
    held.push(idle);
}

/// An idle connection to `scope` whose peer is still in `checked`'s set: every ask re-checks
/// its policy and DNS, and a socket serves only while its exact peer remains in that set.
fn reuse(scope: &Scope, checked: &CheckedUrl) -> Option<(SocketAddr, SendRequest<Empty<Bytes>>)> {
    let mut held = IDLE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    held.retain(|idle| !idle.sender.is_closed());
    let at = held.iter().rposition(|idle| idle.scope == *scope && checked.addrs.contains(&idle.peer))?;
    let idle = held.remove(at);
    Some((idle.peer, idle.sender))
}

/// One bodiless GET under `judge`: connect to the CHECKED addresses (or reuse a connection
/// to the same origin and credentials), write the request, read the head.
pub(crate) async fn get<'l>(
    client: &Client,
    checked: &CheckedUrl,
    headers: &[(String, String)],
    ledger: &'l Ledger,
    judge: Judge,
    fail: Code,
) -> Result<Response<'l>> {
    ledger.check_running()?;
    let scoped: Vec<(String, String)> =
        (headers.iter()).filter(|(name, _)| !name.eq_ignore_ascii_case("range")).cloned().collect();
    let scope = Scope::of(checked, &scoped);
    // An idle connection its origin closed meanwhile fails before any answer: a GET is
    // idempotent, so it is asked again on a fresh connection.
    if let Some((peer, mut sender)) = reuse(&scope, checked) {
        // Kept the moment its body ended, a connection takes the next request once its driver
        // on the runtime has seen that end. Passing it over until then dialed a second
        // connection whenever the runtime was busy: a handshake per chunk on a loaded machine.
        if let Ok(Ok(())) = judged(sender.ready(), ledger, judge, Instant::now(), "an idle connection", fail).await {
            let held = Idle { scope: scope.clone(), peer, sender };
            match exchange(held, checked, headers, ledger, judge, Instant::now(), fail).await {
                Err(refusal) if refusal.code == fail && ledger.check_running().is_ok() => {}
                answered => return answered,
            }
        }
    }
    let opened = Instant::now();
    let socket = connect(checked, ledger, judge, fail).await?;
    let peer = socket.peer_addr().map_err(|error| Refusal { code: fail, detail: format!("peer: {error}") })?;
    let sender = match checked.https {
        true => {
            let name = ServerName::try_from(checked.host.clone()).map_err(|_| Refusal {
                code: Code::SOURCE_NOT_ALLOWED,
                detail: format!("{}: not a valid TLS server name", checked.host),
            })?;
            let connector = tokio_rustls::TlsConnector::from(client.tls());
            let handshake = connector.connect(name, socket);
            let tls = judged(handshake, ledger, judge, opened, "TLS handshake", fail)
                .await?
                .map_err(|error| tls_refusal(checked, error, ledger, fail))?;
            http::verified(checked);
            handshake_http(TokioIo::new(tls), ledger, judge, opened, fail).await?
        }
        false => handshake_http(TokioIo::new(socket), ledger, judge, opened, fail).await?,
    };
    exchange(Idle { scope, peer, sender }, checked, headers, ledger, judge, opened, fail).await
}

/// An HTTP/1.1 connection over `io`, its driver on the runtime.
async fn handshake_http<T>(io: TokioIo<T>, ledger: &Ledger, judge: Judge, opened: Instant, fail: Code) -> Result<SendRequest<Empty<Bytes>>>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (sender, connection) = judged(hyper::client::conn::http1::handshake(io), ledger, judge, opened, "HTTP handshake", fail)
        .await?
        .map_err(|error| Refusal { code: fail, detail: format!("HTTP handshake: {error}") })?;
    runtime()?.spawn(async move {
        let _ = connection.await;
    });
    Ok(sender)
}

#[allow(clippy::too_many_arguments)]
async fn exchange<'l>(
    mut held: Idle,
    checked: &CheckedUrl,
    headers: &[(String, String)],
    ledger: &'l Ledger,
    judge: Judge,
    opened: Instant,
    fail: Code,
) -> Result<Response<'l>> {
    let default_port = if checked.https { 443 } else { 80 };
    let host = match checked.port == default_port {
        true => checked.host.clone(),
        false => format!("{}:{}", checked.host, checked.port),
    };
    let mut request = hyper::Request::get(&checked.target)
        .header("host", host)
        .header("accept-encoding", "identity");
    for (name, value) in headers {
        request = request.header(name.as_str(), value.as_str());
    }
    let request = request
        .body(Empty::new())
        .map_err(|error| Refusal { code: fail, detail: format!("request: {error}") })?;
    let since = Instant::now();
    let answer = judged(held.sender.send_request(request), ledger, judge, since, "response head", fail).await?;
    let answer = answer.map_err(|error| Refusal { code: fail, detail: format!("the connection failed: {error}") })?;
    // A head that arrived is progress for a call; an object's progress is its bytes, and a
    // head that brings none must not teach the pull to be patient.
    match judge {
        Judge::Stream => ledger.taught(opened.elapsed()),
        _ => ledger.answered(),
    }
    let (parts, incoming) = answer.into_parts();
    let headers: Vec<(String, String)> = (parts.headers.iter())
        .filter_map(|(name, value)| Some((name.as_str().to_string(), value.to_str().ok()?.to_string())))
        .collect();
    let encoding = headers.iter().find(|(name, _)| name == "content-encoding").map(|(_, value)| value.trim());
    if let Some(encoding) = encoding.filter(|e| !e.is_empty() && !e.eq_ignore_ascii_case("identity")) {
        return refuse(
            fail,
            format!("origin transformed an exact-byte response with content-encoding {encoding:?} against `accept-encoding: identity`"),
        );
    }
    let status = parts.status.as_u16();
    Ok(Response {
        status,
        headers,
        body: Body {
            incoming,
            ledger,
            judge,
            since: Instant::now(),
            recycle: Some(held),
        },
    })
}

/// `future` under the ledger's rules: cancellation, the pull's progress, and (for a judged
/// exchange) this phase's own silence past the measured patience.
async fn judged<T>(
    future: impl Future<Output = T>,
    ledger: &Ledger,
    judge: Judge,
    since: Instant,
    phase: &str,
    fail: Code,
) -> Result<T> {
    tokio::pin!(future);
    loop {
        ledger.check_running()?;
        match tokio::time::timeout(ledger.tick(), &mut future).await {
            Ok(answer) => return Ok(answer),
            Err(_) => {
                http::transfer_verdict(judge, ledger, since, phase)?;
                http::handshake_verdict(judge, ledger, since, phase, fail)?;
            }
        }
    }
}

/// One TCP connection to the first checked address that answers. Another is started
/// whenever the newest has been silent a tick, families alternating, and the first to
/// connect wins (RFC 8305).
async fn connect(checked: &CheckedUrl, ledger: &Ledger, judge: Judge, fail: Code) -> Result<TcpStream> {
    let first = checked.addrs.first().is_some_and(SocketAddr::is_ipv4);
    let (mut family, mut other) = (
        checked.addrs.iter().filter(|addr| addr.is_ipv4() == first),
        checked.addrs.iter().filter(|addr| addr.is_ipv4() != first),
    );
    let mut addrs = std::iter::from_fn(|| Some([family.next(), other.next()]))
        .take_while(|pair| pair.iter().any(Option::is_some))
        .flatten()
        .flatten()
        .copied();
    let mut pending: JoinSet<std::io::Result<TcpStream>> = JoinSet::new();
    let (mut newest, mut last): (Option<Instant>, Option<String>) = (None, None);
    let mut exhausted = false;
    loop {
        ledger.check_running()?;
        if !exhausted && newest.is_none_or(|since| since.elapsed() >= ledger.tick()) {
            match addrs.next() {
                Some(addr) => {
                    pending.spawn(TcpStream::connect(addr));
                    newest = Some(Instant::now());
                    continue;
                }
                None => exhausted = true,
            }
        }
        if pending.is_empty() {
            let last = last.unwrap_or_default();
            return refuse(fail, format!("{}: no checked address accepted a connection ({last})", checked.host));
        }
        match tokio::time::timeout(ledger.tick(), pending.join_next()).await {
            Ok(Some(Ok(Ok(socket)))) => {
                let _ = socket.set_nodelay(true);
                http::keepalive(&socket, ledger);
                return Ok(socket);
            }
            // A refusal starts the next address at once.
            Ok(Some(Ok(Err(error)))) => {
                if crate::descriptors::exhausted(&error) {
                    shed();
                }
                (last, newest) = (Some(error.to_string()), None);
            }
            Ok(Some(Err(error))) => (last, newest) = (Some(error.to_string()), None),
            Ok(None) => {}
            Err(_) => {
                let since = newest.unwrap_or_else(Instant::now);
                http::transfer_verdict(judge, ledger, since, "a TCP connection")?;
                // Every address asked and the newest silent past the measured patience.
                if exhausted && http::handshake_verdict(judge, ledger, since, "TCP connect", fail).is_err() {
                    return refuse(fail, format!("{}: no checked address answered", checked.host));
                }
            }
        }
    }
}

/// A TLS failure as `http` classifies it: a certificate or protocol verdict is this client's
/// and typed; anything else is the link's weather.
fn tls_refusal(checked: &CheckedUrl, error: std::io::Error, ledger: &Ledger, fail: Code) -> Refusal {
    let verdict = (error.get_ref())
        .and_then(|inner| inner.downcast_ref::<rustls::Error>())
        .is_some_and(|error| {
            matches!(
                error,
                rustls::Error::InvalidCertificate(_)
                    | rustls::Error::NoCertificatesPresented
                    | rustls::Error::UnsupportedNameType
                    | rustls::Error::InvalidCertRevocationList(_)
                    | rustls::Error::PeerIncompatible(_)
            )
        });
    let refusal = match verdict {
        true => Refusal {
            code: Code::TLS_UNTRUSTED,
            detail: format!("{}: TLS refused: {error}; a private hub names its CA with --ca-file", checked.host),
        },
        false => Refusal { code: fail, detail: format!("{}: TLS handshake failed: {error}", checked.host) },
    };
    http::recanted_now(checked, refusal, ledger, fail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::SourcePolicy;
    use std::io::{Read, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A loopback HTTP/1.1 origin answering every GET with six bytes on a kept-alive connection,
    /// counting the connections it accepts; with `close_first` it shuts its first one unanswered.
    fn origin(close_first: bool) -> (CheckedUrl, Arc<AtomicUsize>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let accepted = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&accepted);
        std::thread::spawn(move || {
            for socket in listener.incoming() {
                let Ok(mut socket) = socket else { return };
                if counted.fetch_add(1, Ordering::SeqCst) == 0 && close_first {
                    let _ = socket.shutdown(std::net::Shutdown::Both);
                    continue;
                }
                std::thread::spawn(move || {
                    let (mut head, mut byte) = (Vec::new(), [0]);
                    while socket.read_exact(&mut byte).is_ok() {
                        head.push(byte[0]);
                        if head.ends_with(b"\r\n\r\n") {
                            head.clear();
                            let _ = socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nanswer");
                        }
                    }
                });
            }
        });
        let policy = SourcePolicy { allowed_hosts: vec!["localhost".into()], allow_local: true, ..Default::default() };
        (policy.check(&format!("http://localhost:{port}/object")).unwrap(), accepted)
    }

    /// A connection to `checked` kept idle before its driver has ever run, as a kept connection is
    /// on a busy runtime: its driver is returned, for the test to start when it chooses.
    fn kept_before_its_driver(checked: &CheckedUrl) -> impl Future<Output = ()> + Send + 'static {
        let socket = std::net::TcpStream::connect(("127.0.0.1", checked.port)).unwrap();
        socket.set_nonblocking(true).unwrap();
        let (peer, scope) = (socket.peer_addr().unwrap(), Scope::of(checked, &[]));
        let (sender, connection) = block_on(async {
            let socket = TcpStream::from_std(socket).unwrap();
            hyper::client::conn::http1::handshake::<_, Empty<Bytes>>(TokioIo::new(socket)).await.unwrap()
        })
        .unwrap();
        assert!(!sender.is_ready(), "a driver that never ran has not asked for a request");
        keep(Idle { scope, peer, sender });
        async move {
            let _ = connection.await;
        }
    }

    fn held(checked: &CheckedUrl) -> bool {
        let scope = Scope::of(checked, &[]);
        IDLE.lock().unwrap().iter().any(|idle| idle.scope == scope)
    }

    /// One GET of `checked` read to its end; meanwhile the kept connection's driver starts once the
    /// ask has taken it from the pool or dialed past it.
    fn get_starting(checked: &CheckedUrl, driver: impl Future<Output = ()> + Send + 'static, accepted: &AtomicUsize) -> Result<Vec<u8>> {
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while held(checked) && accepted.load(Ordering::SeqCst) < 2 {
                    std::thread::yield_now();
                }
                runtime().unwrap().spawn(driver);
            });
            let ledger = Ledger::new();
            block_on(async {
                let mut answer = get(&Client::new(), checked, &[], &ledger, Judge::Transfer, Code::TRANSFER_FAILED).await?;
                let mut body = Vec::new();
                while let Some(data) = answer.body.chunk().await? {
                    body.extend_from_slice(&data);
                }
                Ok(body)
            })?
        })
    }

    #[test]
    fn a_connection_kept_before_its_driver_ran_serves_the_next_ask() {
        let (checked, accepted) = origin(false);
        let driver = kept_before_its_driver(&checked);
        assert_eq!(get_starting(&checked, driver, &accepted).unwrap(), b"answer");
        assert_eq!(accepted.load(Ordering::SeqCst), 1, "the next ask dialed past its kept connection");
    }

    #[test]
    fn a_kept_connection_its_peer_closed_gives_way_to_a_fresh_one_at_once() {
        let (checked, accepted) = origin(true);
        let driver = kept_before_its_driver(&checked);
        let started = Instant::now();
        assert_eq!(get_starting(&checked, driver, &accepted).unwrap(), b"answer");
        assert_eq!(accepted.load(Ordering::SeqCst), 2);
        // A dead sender is no wait: nothing here came near a verdict on silence.
        assert!(started.elapsed() < Ledger::new().patience(), "{:?}", started.elapsed());
    }
}
