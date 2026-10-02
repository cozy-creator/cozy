//! The one socket this crate opens (tfs-049). A deliberately small HTTP/1.1 client — GET,
//! POST, PUT, bounded reuse for ranged GETs and uploads, Content-Length or chunked bodies, no compression, no
//! proxy, redirects surfaced to the caller rather than followed — written directly over
//! `TcpStream`/rustls for exactly two properties no shelf client offers together:
//!
//! - **the connection goes to the addresses the predicate CHECKED** (`CheckedUrl.addrs`),
//!   with SNI/Host still carrying the hostname, which closes the resolve-then-reconnect
//!   rebinding window `egress.py` could only name;
//! - **every socket wait is bounded by the caller's deadline and judged by the liveness
//!   ledger, never by an invented constant.** The socket read timeout here is the ledger's
//!   sampling RESOLUTION; each empty tick re-consults the deadline and the measured stall
//!   rule. Nothing in this file owns a number of seconds.
//!
//! That second claim used to hold for the BYTE plane only, and tfs-106 is the repair. A
//! caller with no wall deadline — which is every pod fetch, deliberately, because a module
//! that substituted a number there would be deciding on the caller's behalf that a large
//! artifact on a slow link is a failure — could wait here forever: `read_head` and the TLS
//! handshake looped on an empty tick with nothing to consult, and a hub exchange left the
//! body rule disarmed. That is how a presign to a black-holed tunnel held `Leases::minting`
//! while sixteen streams parked behind it and two paid H100s billed fifteen minutes at
//! exactly zero bytes. A CONTROL-plane exchange now answers to `Judge::Transfer`: the
//! pull's own progress, which is a measurement, and never to this call's own duration,
//! which would be a constant.
//!
//! Object GETs also apply their existing measured stream-silence rule while
//! handshaking, writing and reading headers (h3a-070). Previously Judge::Stream
//! was ignored in those phases, so body retries were unreachable for a silent
//! TLS peer. Phase progress resets that phase's clock; no object wall limit is added.

use crate::err::{refuse, Code, Refusal, Result};
use crate::transport::ledger::{Deadline, Ledger};
use crate::transport::policy::CheckedUrl;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName};
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const HEAD_MAX_BYTES: usize = 64 * 1024;
const WRITE_CHUNK: usize = 64 * 1024;

fn would_block(error: &std::io::Error) -> bool {
    matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

/// What an exchange's SILENCE answers to, beyond the caller's deadline.
///
/// The two planes measure different things because they are different things, and the whole
/// defect tfs-106 repairs is that only one of them was measured at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Judge {
    /// The BYTE plane. This stream's own chunks, against the pull's measured pace: an
    /// object GET that accepted a connection and stopped sending is stalled, closed, and
    /// re-asked, which the plan already made cost one object.
    Stream,
    /// The CONTROL plane. A hub call has no bytes of its own worth judging — a mint that
    /// takes a minute while sixteen streams keep landing objects is a call, not a wedge —
    /// so it is judged by the PULL'S OWN PROGRESS instead. It faults exactly when nothing
    /// has landed anywhere in this pull for longer than the pull's measured pace allows,
    /// which is the owner's rule read literally: the lack of measured progress, never a
    /// number of seconds.
    Transfer,
    /// An UPLOAD. Every wait — connect, handshake, each write, and the answer after the
    /// last body byte — is judged by this exchange's own silence against the ledger's
    /// patience. Nothing it moves lands here, so it neither counts as progress nor teaches
    /// the pace.
    Silence,
    /// Neither. The caller's deadline is the only bound, which is what a foreign-source
    /// read with its own per-part retry loop above it already had.
    Deadline,
}

/// The CONTROL plane's verdict, quoting exactly the figures a reader needs and no others:
/// how long this call has been waiting, what the pull has actually landed, how long ago,
/// and the term the pull itself taught. Nobody has to read this file to learn whether the
/// number was earned.
///
/// It refuses `TRANSFER_FAILED` and deliberately not the exchange's own `fail` code. The hub
/// may be perfectly healthy; what has been established is that the transfer this call serves
/// has stopped, and calling that "the hub did not answer" would send the next reader to the
/// wrong machine. It is also what makes the verdict terminal in `hub::call` instead of
/// re-asked three more times: a re-ask reaches the identical conclusion on its first tick and
/// would only overwrite this diagnosis with a duller one.
fn transfer_verdict(
    judge: Judge,
    ledger: &Ledger,
    since: Instant,
    waiting_for: &str,
) -> Result<()> {
    ledger.check_running()?;
    if judge != Judge::Transfer {
        return Ok(());
    }
    let still = ledger.still();
    // NOTHING MOVED IS NOT THE SAME AS STOPPED MOVING, and the owner's rule turns on exactly
    // that distinction: kill or error on the lack of MEASURED progress. A pull that has not
    // yet landed a byte has measured nothing about its own pace, so there is no term here
    // that is not a number somebody made up — and a made-up number is the thing this whole
    // module refuses. A hub call that wedges before the first byte of a pull is therefore
    // still bounded by the caller's deadline alone. That is a real remaining silence and it
    // is stated rather than papered over with a constant.
    if still.moved == 0 {
        return Ok(());
    }
    let patience = ledger.stillness();
    // TWO CLOCKS, AND BOTH MUST BE PAST THE MEASURED TERM. `silent` is how long the pull has
    // moved nothing; `outstanding` is how long THIS call has been the thing it waits on.
    //
    // A call must never inherit stillness that predates it. Between one object landing and
    // the next mint going out, a pull is legitimately still — it is hashing, fsyncing and
    // committing what just arrived — and a mint that opened into the tail of that would
    // otherwise be condemned on its first tick for local work it had nothing to do with.
    // Requiring the call to have been outstanding for the term as well gives every call a
    // full measured window of its own, and leaves the verdict meaning what it says: this
    // call has been waiting this long, and the transfer moved nothing in any of it.
    let outstanding = since.elapsed();
    if still.silent <= patience || outstanding <= patience {
        return Ok(());
    }
    refuse(
        Code::TRANSFER_FAILED,
        format!(
            "the transfer this call serves has stopped: {:.1} s waiting for {waiting_for}, \
             {} B landed in this pull and none of them for {:.1} s, against the {:.1} s \
             this pull's own measured pace allows ({} times the longest it has ever gone \
             still and recovered, and never less than one stream's own term)",
            outstanding.as_secs_f64(),
            still.moved,
            still.silent.as_secs_f64(),
            patience.as_secs_f64(),
            crate::transport::ledger::STALL_FACTOR,
        ),
    )
}

/// Object-stream silence is judged before the body too. Each phase starts with
/// its own window and advances only when transport bytes actually move; slow
/// progress may span arbitrarily many windows without exhausting a wall budget.
fn stream_verdict(judge: Judge, ledger: &Ledger, advanced: Instant, phase: &str) -> Result<()> {
    ledger.check_running()?;
    if !judge.silent_is_stall() || advanced.elapsed() <= ledger.patience() {
        return Ok(());
    }
    refuse(
        Code::TRANSFER_FAILED,
        format!(
            "the object stream stopped during {phase}: silent {:.1} s against the \
             {:.1} s this pull's measured pace allows",
            advanced.elapsed().as_secs_f64(),
            ledger.patience().as_secs_f64(),
        ),
    )
}

/// Connect, TLS and the request write are phases a live peer answers at once, so on the
/// CONTROL plane too their silence past the ledger's patience is the peer's verdict: the
/// exchange's own failure code, which `hub::call` re-asks. Before a byte of the pull has
/// moved this is the only bound a black-holed hub meets. The head wait is never judged this
/// way: a live hub may take its time, and a dead one is found by TCP keepalive.
fn handshake_verdict(
    judge: Judge,
    ledger: &Ledger,
    advanced: Instant,
    phase: &str,
    fail: Code,
) -> Result<()> {
    stream_verdict(judge, ledger, advanced, phase)?;
    if judge != Judge::Transfer || advanced.elapsed() <= ledger.patience() {
        return Ok(());
    }
    refuse(
        fail,
        format!(
            "the peer was silent {:.1} s during {phase} against the {:.1} s this pull's \
             measured pace allows",
            advanced.elapsed().as_secs_f64(),
            ledger.patience().as_secs_f64(),
        ),
    )
}

impl Judge {
    fn silent_is_stall(self) -> bool {
        matches!(self, Judge::Stream | Judge::Silence)
    }
}

pub struct Client {
    tls: Mutex<Tls>,
    /// Trusts exactly the roots it was built with, never the process's later additions.
    fixed: bool,
    idle: IdlePool,
}

/// A client's TLS configuration and the process trust generation it was built from.
struct Tls {
    generation: u64,
    config: Arc<rustls::ClientConfig>,
}

type IdlePool = Arc<Mutex<Vec<Idle>>>;

// This bounds retained descriptors, not in-flight transfers (the caller owns those).
// `descriptors::PER_STREAM` counts one idle socket per stream.
const MAX_IDLE: usize = 32;

/// A connection belongs to its origin and the credentials sent on it, never to one
/// request target: a signed CDN URL changes per resolve, and keying on it meant the
/// delivery hop never reused a socket.
#[derive(PartialEq, Eq)]
struct Scope {
    https: bool,
    host: String,
    port: u16,
    // Do not retain credentials in an idle connection's key.
    request: [u8; 32],
}

impl Scope {
    fn of(checked: &CheckedUrl, headers: &[(String, String)]) -> Self {
        let mut hash = crate::sha256::Sha256::new();
        for (name, value) in headers {
            // Preserve duplicate-header ordering as well as values: an origin
            // can assign different meaning to their first or last occurrence.
            let name = name.to_ascii_lowercase();
            // Which bytes are asked for is not who is asking: chunks share a connection.
            if name == "range" {
                continue;
            }
            hash.update(&(name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(&(value.len() as u64).to_le_bytes());
            hash.update(value.as_bytes());
        }
        Self {
            https: checked.https,
            host: checked.host.clone(),
            port: checked.port,
            request: hash.finish(),
        }
    }
}

#[cfg(test)]
#[path = "http_reuse_tests.rs"]
mod reuse_tests;

struct Idle {
    scope: Scope,
    wire: Wire,
}

/// Authorities this process trusts beside the webpki set: a private hub's CA, named by its
/// caller as a location (`tfs --ca-file`, `ensure(ca_file=)`). Additive: a root once
/// trusted stays trusted, and every shared client picks up an addition on its next exchange.
static EXTRA_ROOTS: Mutex<Vec<CertificateDer<'static>>> = Mutex::new(Vec::new());
static TRUST_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Trust the PEM certificates in `pem` for every client this process uses.
pub fn trust_roots(pem: &[u8]) -> Result<usize> {
    let roots: Vec<_> = CertificateDer::pem_slice_iter(pem)
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| Refusal {
            code: Code::UNKNOWN_FORMAT,
            detail: format!("not a PEM certificate bundle: {e}"),
        })?;
    let (usable, _) = rustls::RootCertStore::empty().add_parsable_certificates(roots.clone());
    if roots.is_empty() || usable != roots.len() {
        return refuse(Code::UNKNOWN_FORMAT, "holds no usable CA certificate");
    }
    let mut extra = EXTRA_ROOTS.lock().unwrap();
    let before = extra.len();
    for root in roots {
        if !extra.contains(&root) {
            extra.push(root);
        }
    }
    if extra.len() != before {
        TRUST_GENERATION.fetch_add(1, Ordering::AcqRel);
    }
    Ok(usable)
}

fn tls_config(extra_roots: &[CertificateDer<'static>]) -> Arc<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    for root in extra_roots {
        let _ = roots.add(root.clone());
    }
    Arc::new(
        rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .expect("rustls ships its own default protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth(),
    )
}

impl Client {
    /// Trusts the webpki set plus every root this process trusts, now or later.
    pub fn new() -> Client {
        let generation = TRUST_GENERATION.load(Ordering::Acquire);
        let config = tls_config(&EXTRA_ROOTS.lock().unwrap());
        Client {
            tls: Mutex::new(Tls { generation, config }),
            fixed: false,
            idle: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Trusts the webpki set plus `extra_roots`, and nothing the process adds later.
    pub fn with_extra_roots(extra_roots: &[CertificateDer<'static>]) -> Client {
        Client {
            tls: Mutex::new(Tls {
                generation: 0,
                config: tls_config(extra_roots),
            }),
            fixed: true,
            idle: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn tls(&self) -> Arc<rustls::ClientConfig> {
        let mut tls = self.tls.lock().unwrap();
        let generation = TRUST_GENERATION.load(Ordering::Acquire);
        if !self.fixed && tls.generation != generation {
            *tls = Tls {
                generation,
                config: tls_config(&EXTRA_ROOTS.lock().unwrap()),
            };
        }
        Arc::clone(&tls.config)
    }

    fn take(&self, scope: &Scope, checked: &CheckedUrl) -> Option<Wire> {
        let mut idle = self.idle.lock().ok()?;
        let at = idle.iter().rposition(|entry| entry.scope == *scope)?;
        let mut entry = idle.remove(at);
        // Every request has already rechecked its policy and DNS. An established
        // socket is usable only while its exact peer remains in that checked set.
        if !checked.addrs.contains(&entry.wire.sock().peer_addr().ok()?) || !entry.wire.idle_ready()
        {
            return None;
        }
        Some(entry.wire)
    }
}

impl Default for Client {
    fn default() -> Client {
        Client::new()
    }
}

// ---------------------------------------------------------------- the wire

enum Wire {
    Plain(TcpStream),
    Tls {
        sock: TcpStream,
        conn: Box<rustls::ClientConnection>,
    },
}

impl Wire {
    fn idle_ready(&mut self) -> bool {
        // No unread application bytes or close-notify may cross response boundaries.
        if let Wire::Tls { conn, .. } = self {
            match conn.reader().read(&mut [0]) {
                Err(error) if error.kind() == ErrorKind::WouldBlock => {}
                _ => return false,
            }
        }
        let socket = self.sock();
        if socket.set_nonblocking(true).is_err() {
            return false;
        }
        let clean =
            matches!(socket.peek(&mut [0]), Err(error) if error.kind() == ErrorKind::WouldBlock);
        socket.set_nonblocking(false).is_ok() && clean
    }

    fn sock(&self) -> &TcpStream {
        match self {
            Wire::Plain(sock) => sock,
            Wire::Tls { sock, .. } => sock,
        }
    }

    fn set_read_timeout(&self, timeout: Duration) {
        let _ = self
            .sock()
            .set_read_timeout(Some(timeout.max(Duration::from_millis(1))));
    }
    fn set_write_timeout(&self, timeout: Duration) {
        let _ = self
            .sock()
            .set_write_timeout(Some(timeout.max(Duration::from_millis(1))));
    }

    /// One read attempt. `WouldBlock` is an empty sampling tick, surfaced for the caller
    /// to judge; 0 is end-of-stream.
    fn read_step(&mut self, buf: &mut [u8], ledger: &Ledger) -> std::io::Result<usize> {
        match self {
            Wire::Plain(sock) => sock.read(buf),
            Wire::Tls { sock, conn } => loop {
                // Partial encrypted records can keep arriving without yielding
                // plaintext or a socket timeout. Cancellation must still win.
                ledger
                    .check_running()
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                while conn.wants_write() {
                    conn.write_tls(sock)?;
                }
                match conn.reader().read(buf) {
                    Ok(n) => return Ok(n),
                    Err(e) if e.kind() == ErrorKind::WouldBlock => {
                        let n = conn.read_tls(sock)?;
                        if n == 0 {
                            return Ok(0);
                        }
                        conn.process_new_packets()
                            .map_err(|e| std::io::Error::other(e.to_string()))?;
                    }
                    Err(e) => return Err(e),
                }
            },
        }
    }

    /// One write attempt. Queued TLS bytes are flushed FIRST, so a surfaced `WouldBlock`
    /// always means "none of `buf` was taken" and a retry with the same slice is safe.
    fn write_step(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Wire::Plain(sock) => sock.write(buf),
            Wire::Tls { sock, conn } => {
                while conn.wants_write() {
                    conn.write_tls(sock)?;
                }
                let n = conn.writer().write(buf)?;
                while conn.wants_write() {
                    match conn.write_tls(sock) {
                        Ok(_) => {}
                        Err(e) if would_block(&e) => break,
                        Err(e) => return Err(e),
                    }
                }
                Ok(n)
            }
        }
    }

    /// Drain everything queued. `WouldBlock` surfaces for the caller's tick.
    fn flush_step(&mut self) -> std::io::Result<()> {
        match self {
            Wire::Plain(sock) => sock.flush(),
            Wire::Tls { sock, conn } => {
                while conn.wants_write() {
                    conn.write_tls(sock)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(not(target_vendor = "apple"))]
fn nonblocking_socket(
    family: rustix::net::AddressFamily,
) -> rustix::io::Result<std::os::fd::OwnedFd> {
    use rustix::net::{self, SocketFlags, SocketType};
    net::socket_with(
        family,
        SocketType::STREAM,
        SocketFlags::NONBLOCK | SocketFlags::CLOEXEC,
        Some(net::ipproto::TCP),
    )
}

/// Darwin's socket(2) takes no type flags; they are set before the fd is used.
#[cfg(target_vendor = "apple")]
fn nonblocking_socket(
    family: rustix::net::AddressFamily,
) -> rustix::io::Result<std::os::fd::OwnedFd> {
    use rustix::io::{fcntl_setfd, ioctl_fionbio, FdFlags};
    use rustix::net::{self, SocketType};
    let socket = net::socket(family, SocketType::STREAM, Some(net::ipproto::TCP))?;
    fcntl_setfd(&socket, FdFlags::CLOEXEC)?;
    ioctl_fionbio(&socket, true)?;
    Ok(socket)
}

/// One TCP connection to the first checked address that answers. Another address is
/// started whenever the newest has been silent for a tick, families alternating, and the
/// first handshake to complete wins (RFC 8305): an IPv6 route that drops SYNs while IPv4
/// answers costs a tick, not its patience times every address ahead of the one that works.
fn connect(
    checked: &CheckedUrl,
    deadline: Deadline,
    ledger: &Ledger,
    judge: Judge,
    fail: Code,
) -> Result<TcpStream> {
    use rustix::event::{poll, PollFd, PollFlags, Timespec};
    use rustix::net::{self, AddressFamily};
    let connected = |socket: std::os::fd::OwnedFd| -> Result<TcpStream> {
        let socket: TcpStream = socket.into();
        socket.set_nonblocking(false).map_err(|error| Refusal {
            code: fail,
            detail: format!("configure connected socket: {error}"),
        })?;
        let _ = socket.set_nodelay(true);
        keepalive(&socket, ledger);
        Ok(socket)
    };
    let first = checked.addrs.first().is_some_and(|addr| addr.is_ipv4());
    let (mut family, mut other) = (
        checked.addrs.iter().filter(|addr| addr.is_ipv4() == first),
        checked.addrs.iter().filter(|addr| addr.is_ipv4() != first),
    );
    let mut addrs = std::iter::from_fn(|| Some([family.next(), other.next()]))
        .take_while(|pair| pair.iter().any(Option::is_some))
        .flatten()
        .flatten();
    let mut pending: Vec<(std::os::fd::OwnedFd, Instant)> = Vec::new();
    let mut last: Option<std::io::Error> = None;
    loop {
        ledger.check_running()?;
        let quiet = pending
            .last()
            .is_none_or(|(_, since)| since.elapsed() >= ledger.tick());
        if let Some(addr) = quiet.then(|| addrs.next()).flatten() {
            let family = match addr.is_ipv4() {
                true => AddressFamily::INET,
                false => AddressFamily::INET6,
            };
            let socket = match nonblocking_socket(family) {
                Ok(socket) => socket,
                Err(error) => {
                    let error: std::io::Error = error.into();
                    if crate::descriptors::exhausted(&error) {
                        // Another address needs another descriptor it will not get either.
                        return refuse(
                            Code::FD_HEADROOM,
                            format!("{}: no descriptor for a socket ({error})", checked.host),
                        );
                    }
                    last = Some(error);
                    continue;
                }
            };
            match net::connect(&socket, addr) {
                Ok(()) => return connected(socket),
                Err(rustix::io::Errno::INPROGRESS) => pending.push((socket, Instant::now())),
                Err(error) => last = Some(error.into()),
            }
            continue;
        }
        let Some(oldest) = pending.first().map(|(_, since)| *since) else {
            return refuse(
                fail,
                format!(
                    "{}: no checked address accepted a connection ({})",
                    checked.host,
                    last.map(|error| error.to_string()).unwrap_or_default()
                ),
            );
        };
        // Cancellation and the silence rules stay active without a total connect deadline.
        transfer_verdict(judge, ledger, oldest, "a TCP connection")?;
        // A silent address is that address's verdict, not the URL's.
        if let Err(silent) = handshake_verdict(judge, ledger, oldest, "TCP connect", fail) {
            ledger.check_running()?;
            last = Some(std::io::Error::new(ErrorKind::TimedOut, silent.detail));
            pending.remove(0);
            continue;
        }
        let tick = deadline
            .remaining()?
            .map_or(ledger.tick(), |left| left.min(ledger.tick()));
        let timeout = Timespec {
            tv_sec: tick.as_secs() as _,
            tv_nsec: tick.subsec_nanos() as _,
        };
        let mut waiting: Vec<PollFd<'_>> = pending
            .iter()
            .map(|(socket, _)| PollFd::new(socket, PollFlags::OUT))
            .collect();
        let ready: Vec<bool> = match poll(&mut waiting, Some(&timeout)) {
            Ok(0) | Err(rustix::io::Errno::INTR) => continue,
            Ok(_) => waiting.iter().map(|fd| !fd.revents().is_empty()).collect(),
            Err(error) => {
                return refuse(fail, format!("{}: poll: {error}", checked.host));
            }
        };
        drop(waiting);
        let mut at = 0;
        for ready in ready {
            if !ready {
                at += 1;
                continue;
            }
            let (socket, _) = pending.remove(at);
            match net::sockopt::socket_error(&socket) {
                Ok(Ok(())) => return connected(socket),
                Ok(Err(error)) | Err(error) => last = Some(error.into()),
            }
        }
    }
}

/// TCP keepalive at the ledger's own resolution: probes start after the noise floor of
/// silence and a peer that answers none of `STILL_SAMPLES` probes, a sample apart, is gone.
/// A live hub that is merely slow ACKs every probe and is waited for; a dead host or a
/// dropped tunnel fails the read as weather, which the caller asks again.
fn keepalive(socket: &TcpStream, ledger: &Ledger) {
    use rustix::net::sockopt;
    let second = Duration::from_secs(1);
    let _ = sockopt::set_socket_keepalive(socket, true);
    let _ = sockopt::set_tcp_keepidle(socket, ledger.floor().max(second));
    let _ = sockopt::set_tcp_keepintvl(socket, ledger.sample().max(second));
    let _ = sockopt::set_tcp_keepcnt(socket, crate::transport::ledger::STILL_SAMPLES);
}

#[allow(clippy::too_many_arguments)]
fn tls_handshake(
    host: &str,
    sock: &mut TcpStream,
    conn: &mut rustls::ClientConnection,
    deadline: Deadline,
    ledger: &Ledger,
    judge: Judge,
    since: Instant,
    fail: Code,
) -> Result<()> {
    let sample = ledger.tick();
    let mut advanced = Instant::now();
    while conn.is_handshaking() {
        transfer_verdict(judge, ledger, since, "a TLS handshake")?;
        let tick = match deadline.remaining()? {
            Some(left) => left.min(sample),
            None => sample,
        };
        let _ = sock.set_read_timeout(Some(tick.max(Duration::from_millis(1))));
        let _ = sock.set_write_timeout(Some(tick.max(Duration::from_millis(1))));
        if conn.wants_write() {
            match conn.write_tls(sock) {
                Ok(n) if n > 0 => advanced = Instant::now(),
                Ok(_) => {}
                Err(e) if would_block(&e) => {
                    handshake_verdict(judge, ledger, advanced, "TLS handshake", fail)?;
                }
                Err(e) => return refuse(fail, format!("TLS handshake write: {e}")),
            }
            continue;
        }
        match conn.read_tls(sock) {
            Ok(0) => return refuse(fail, "origin closed the connection mid-handshake"),
            Ok(_) => {
                advanced = Instant::now();
                // A certificate or protocol verdict is this client's, and re-asking reaches
                // it again: refused typed, never retried as weather. The peer's own alert or
                // a garbled record is the link's weather and is asked again. The alert goes
                // out so the origin logs why.
                if let Err(e) = conn.process_new_packets() {
                    let _ = conn.write_tls(sock);
                    let verdict = matches!(
                        e,
                        rustls::Error::InvalidCertificate(_)
                            | rustls::Error::NoCertificatesPresented
                            | rustls::Error::UnsupportedNameType
                            | rustls::Error::InvalidCertRevocationList(_)
                            | rustls::Error::PeerIncompatible(_)
                    );
                    if !verdict {
                        return refuse(fail, format!("{host}: TLS handshake failed: {e}"));
                    }
                    return refuse(
                        Code::TLS_UNTRUSTED,
                        format!(
                            "{host}: TLS refused: {e}; a private hub names its CA with --ca-file"
                        ),
                    );
                }
            }
            Err(e) if would_block(&e) => {
                handshake_verdict(judge, ledger, advanced, "TLS handshake", fail)?;
            }
            Err(e) => return refuse(fail, format!("TLS handshake read: {e}")),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- the exchange

pub struct Response<'l> {
    pub status: u16,
    headers: Vec<(String, String)>,
    body: BodyReader<'l>,
}

impl<'l> Response<'l> {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn into_body(self) -> BodyReader<'l> {
        self.body
    }

    /// Drop a small non-artifact body (a redirect's, a refusal's) by reading it to its end,
    /// so the connection can serve the next ask. Nothing read here is counted; a body
    /// larger than a response head is simply left, and its connection closes.
    pub fn discard(self) {
        let mut body = self.body;
        let mut buf = [0u8; 4096];
        let mut left = HEAD_MAX_BYTES;
        while left > 0 {
            match body.read_framed(&mut buf[..left.min(4096)]) {
                Ok(0) | Err(_) => return,
                Ok(n) => left -= n,
            }
        }
    }

    /// The wait this answer states (`Retry-After`), read in the origin's own clock:
    /// delta-seconds, or an HTTP-date against the answer's own `Date`. `None` when it
    /// states nothing readable.
    pub fn stated_wait(&self) -> Option<Duration> {
        let value = self.header("retry-after")?.trim();
        if let Ok(seconds) = value.parse::<u64>() {
            return Some(Duration::from_secs(seconds));
        }
        let at = http_date(value)?;
        let now = http_date(self.header("date")?)?;
        Some(Duration::from_secs(at.saturating_sub(now)))
    }

    /// The whole body, refused above `cap` rather than truncated.
    pub fn read_capped(mut self, cap: usize) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = self.body.read_refusing(&mut buf)?;
            if n == 0 {
                return Ok(out);
            }
            if out.len() + n > cap {
                return refuse(
                    Code::SIZE_CAP,
                    format!("response body passed the {cap} B cap and was cut there"),
                );
            }
            out.extend_from_slice(&buf[..n]);
        }
    }
}

/// RFC 9110 IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`) as Unix seconds.
fn http_date(text: &str) -> Option<u64> {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut parts = text.split_whitespace().skip(1);
    let day: i64 = parts.next()?.parse().ok()?;
    let month = parts.next()?;
    let month = MONTHS.iter().position(|name| *name == month)? as i64 + 1;
    let year: i64 = parts.next()?.parse().ok()?;
    let clock: Vec<i64> = parts
        .next()?
        .split(':')
        .map(|part| part.parse().ok())
        .collect::<Option<_>>()?;
    if parts.next()? != "GMT" || clock.len() != 3 {
        return None;
    }
    // Days from the civil calendar (Hinnant).
    let (y, m) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + (153 * m + 2) / 5 + day - 1;
    let days = era * 146_097 + doe - 719_468;
    u64::try_from(days * 86_400 + clock[0] * 3600 + clock[1] * 60 + clock[2]).ok()
}

#[derive(Clone, Copy)]
enum Framing {
    Length { remaining: u64 },
    Chunked { in_chunk: u64, done: bool },
    Eof,
}

/// The response body as a plain `Read`, so `DeliveryGrant::admit` consumes it like any
/// other source. Every arriving chunk is COUNTED into the ledger where it crosses into
/// this process — which is what makes "this stream is not moving" a measurement instead
/// of an opinion — and when `judge` is armed, an empty tick beyond the pull's own measured
/// patience is a stall, refused as the request's own failure code.
pub struct BodyReader<'l> {
    wire: Option<Wire>,
    recycle: Option<Box<(IdlePool, Scope)>>,
    framing: Framing,
    raw: Vec<u8>,
    raw_at: usize,
    deadline: Deadline,
    ledger: &'l Ledger,
    judge: Judge,
    advanced: Instant,
    /// When this body's reader was built — the start of the wait a control-plane verdict
    /// measures, so a body that opens into a pull's local pause is not condemned for it.
    since: Instant,
    fail: Code,
}

impl Drop for BodyReader<'_> {
    fn drop(&mut self) {
        self.recycle_if_consumed();
    }
}

impl BodyReader<'_> {
    /// Hand the connection back the moment its last body byte is read, not when the
    /// reader is dropped: the consumer may still spend seconds hashing, fsyncing and
    /// committing what arrived, and the next object's GET should find this socket idle.
    fn recycle_if_consumed(&mut self) {
        // Only a fully consumed, unambiguous Content-Length response is reusable.
        // Dropped/failed bodies, chunked bodies and EOF-delimited replies close.
        if !matches!(self.framing, Framing::Length { remaining: 0 }) || self.raw_len() != 0 {
            return;
        }
        let Some(recycle) = self.recycle.take() else {
            return;
        };
        let (pool, scope) = *recycle;
        let Some(wire) = self.wire.take() else {
            return;
        };
        let Ok(mut idle) = pool.lock() else {
            return;
        };
        // Concurrent streams to one origin each keep their socket, within the bound.
        if idle.len() == MAX_IDLE {
            idle.remove(0);
        }
        idle.push(Idle { scope, wire });
    }
}

impl BodyReader<'_> {
    /// One raw read off the wire under the sampling rule: the socket timeout is the
    /// RESOLUTION; the deadline and (when judged) the measured patience carry the verdict.
    fn step(&mut self, buf: &mut [u8]) -> Result<usize> {
        // Time spent by the consumer hashing or writing the prior chunk is not
        // network silence. This wait starts when it asks for another chunk.
        self.advanced = Instant::now();
        loop {
            self.ledger.check_running()?;
            let tick = match self.deadline.remaining()? {
                Some(left) => left.min(self.ledger.tick()),
                None => self.ledger.tick(),
            };
            let wire = self.wire.as_mut().expect("body owns its wire");
            wire.set_read_timeout(tick);
            match wire.read_step(buf, self.ledger) {
                Ok(n) => {
                    if n > 0 {
                        let now = Instant::now();
                        if self.judge == Judge::Stream {
                            self.ledger.taught(now - self.advanced);
                        }
                        self.advanced = now;
                    }
                    return Ok(n);
                }
                Err(e) if would_block(&e) => {
                    transfer_verdict(self.judge, self.ledger, self.since, "a response body")?;
                    let silent = self.advanced.elapsed();
                    if self.judge.silent_is_stall() && silent > self.ledger.patience() {
                        let _ = self
                            .wire
                            .as_ref()
                            .expect("body owns its wire")
                            .sock()
                            .shutdown(std::net::Shutdown::Both);
                        return refuse(
                            Code::TRANSFER_FAILED,
                            format!(
                                "the stream stopped moving bytes: silent {:.1} s against the \
                                 {:.1} s this pull's own measured pace allows ({} times \
                                 its longest gap, floored at the sampling noise)",
                                silent.as_secs_f64(),
                                self.ledger.patience().as_secs_f64(),
                                crate::transport::ledger::STALL_FACTOR,
                            ),
                        );
                    }
                }
                Err(e) => return refuse(self.fail, format!("the connection failed mid-body: {e}")),
            }
        }
    }

    fn raw_len(&self) -> usize {
        self.raw.len() - self.raw_at
    }

    fn fill_raw(&mut self) -> Result<usize> {
        if self.raw_at < self.raw.len() {
            return Ok(self.raw_len());
        }
        let mut buf = vec![0u8; 64 * 1024];
        let n = self.step(&mut buf)?;
        buf.truncate(n);
        self.raw = buf;
        self.raw_at = 0;
        Ok(n)
    }

    fn take_raw(&mut self, out: &mut [u8]) -> usize {
        let have = &self.raw[self.raw_at..];
        let n = have.len().min(out.len());
        out[..n].copy_from_slice(&have[..n]);
        self.raw_at += n;
        n
    }

    fn raw_line(&mut self) -> Result<String> {
        let mut line = Vec::new();
        loop {
            while self.raw_at < self.raw.len() {
                let byte = self.raw[self.raw_at];
                self.raw_at += 1;
                if byte == b'\n' {
                    while line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    return String::from_utf8(line).map_err(|_| Refusal {
                        code: self.fail,
                        detail: "chunked framing line is not ASCII".into(),
                    });
                }
                line.push(byte);
                if line.len() > 1024 {
                    return refuse(self.fail, "chunked framing line over 1 KiB");
                }
            }
            if self.fill_raw()? == 0 {
                return refuse(self.fail, "the body ended inside its chunked framing");
            }
        }
    }

    /// One read of the framed body, and the ONE place the pull's own progress is counted.
    ///
    /// It is counted here and not at the socket because `read_head` fills its buffer with
    /// the head AND whatever of the body arrived behind it, so a small object can land in
    /// its entirety before this reader exists. Counting at the socket missed exactly those,
    /// which left the control plane believing a busy pull had moved nothing — the one state
    /// in which it must not judge at all.
    ///
    /// THE PULL'S PROGRESS IS ITS ARTIFACT BYTES. A judged body is an object crossing the
    /// door and is counted; anything else — a hub's JSON answer, a foreign source read —
    /// advances the pull without adding to the figure a refusal quotes, because a reader
    /// checking that figure against the transfer's own report must find the same number.
    pub fn read_refusing(&mut self, out: &mut [u8]) -> Result<usize> {
        let n = match self.read_framed(out) {
            Ok(n) => n,
            Err(error) => {
                self.recycle = None;
                return Err(error);
            }
        };
        if n > 0 {
            match self.judge {
                Judge::Stream => self.ledger.moved(n as u64),
                _ => self.ledger.answered(),
            }
        }
        Ok(n)
    }

    fn read_framed(&mut self, out: &mut [u8]) -> Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        match self.framing {
            Framing::Length { remaining } => {
                if remaining == 0 {
                    return Ok(0);
                }
                let want = remaining.min(out.len() as u64) as usize;
                if self.raw_at >= self.raw.len() && self.fill_raw()? == 0 {
                    return refuse(
                        self.fail,
                        "the body ended short of its declared content-length — a short read \
                         is a refusal, not an empty file",
                    );
                }
                let n = self.take_raw(&mut out[..want]);
                self.framing = Framing::Length {
                    remaining: remaining - n as u64,
                };
                self.recycle_if_consumed();
                Ok(n)
            }
            Framing::Chunked { in_chunk, done } => {
                if done {
                    return Ok(0);
                }
                let mut in_chunk = in_chunk;
                if in_chunk == 0 {
                    let line = self.raw_line()?;
                    let size_text = line.split(';').next().unwrap_or("").trim();
                    let size = u64::from_str_radix(size_text, 16).map_err(|_| Refusal {
                        code: self.fail,
                        detail: format!("chunk size {size_text:?} is not hex"),
                    })?;
                    if size == 0 {
                        loop {
                            if self.raw_line()?.is_empty() {
                                break;
                            }
                        }
                        self.framing = Framing::Chunked {
                            in_chunk: 0,
                            done: true,
                        };
                        return Ok(0);
                    }
                    in_chunk = size;
                }
                let want = in_chunk.min(out.len() as u64) as usize;
                if self.raw_at >= self.raw.len() && self.fill_raw()? == 0 {
                    return refuse(self.fail, "the body ended inside a chunk");
                }
                let n = self.take_raw(&mut out[..want]);
                in_chunk -= n as u64;
                if in_chunk == 0 {
                    let boundary = self.raw_line()?;
                    if !boundary.is_empty() {
                        return refuse(self.fail, "chunk did not end at its own CRLF");
                    }
                }
                self.framing = Framing::Chunked {
                    in_chunk,
                    done: false,
                };
                Ok(n)
            }
            Framing::Eof => {
                if self.raw_at >= self.raw.len() && self.fill_raw()? == 0 {
                    return Ok(0);
                }
                Ok(self.take_raw(out))
            }
        }
    }
}

impl Read for BodyReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        self.read_refusing(out)
            .map_err(|refusal| std::io::Error::other(refusal.to_string()))
    }
}

/// A request body. A store object streams from its verified descriptor in bounded chunks,
/// so an upload's memory does not grow with the object.
#[derive(Clone, Copy)]
pub enum Payload<'b> {
    Bytes(&'b [u8]),
    Object(&'b crate::store::VerifiedFile),
}

impl Payload<'_> {
    fn len(&self) -> u64 {
        match self {
            Payload::Bytes(bytes) => bytes.len() as u64,
            Payload::Object(file) => file.len(),
        }
    }
}

/// The largest slice of a streamed body held in memory at once.
const BODY_CHUNK: u64 = 1 << 20;

/// One exchange with an in-memory body. See [`send`].
#[allow(clippy::too_many_arguments)]
pub fn request<'l>(
    client: &Client,
    checked: &CheckedUrl,
    method: &str,
    headers: &[(String, String)],
    body: Option<&[u8]>,
    deadline: Deadline,
    ledger: &'l Ledger,
    judge: Judge,
    write_silence: Option<Duration>,
    fail: Code,
) -> Result<Response<'l>> {
    // Every bodiless GET may reuse a connection to its origin: a pull's objects are many
    // GETs to one host, and a fresh TCP+TLS connection per object spent a descriptor, a
    // handshake and a slow start on each. Metadata writes keep one connection each.
    let pooled = method == "GET"
        && body.is_none()
        && !headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("connection"));
    let scoped: Vec<(String, String)> = headers
        .iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case("range"))
        .cloned()
        .collect();
    send(
        client,
        checked,
        method,
        headers,
        body.map(Payload::Bytes),
        pooled.then_some(scoped.as_slice()),
        deadline,
        ledger,
        judge,
        write_silence,
        fail,
    )
}

/// One exchange: connect to the CHECKED addresses, write the request, read the head, hand
/// back the framed body. `judge` says which plane this exchange belongs to and therefore
/// what its silence answers to — see `Judge`. It governs EVERY wait in the exchange, not
/// only the body: the handshake, the request write and the response head are exactly where
/// a black-holed tunnel parks a hub call, and leaving them unjudged is what made a
/// deadline-free caller wait forever. `write_silence` additionally bounds one WRITE
/// operation that moves nothing — the push plane derives it from the ledger's floor, the
/// port of egress.py's one derived silence bound; `None` leaves the write to `judge`.
/// `pool` keeps the connection alive for the next exchange with this origin that names
/// the same headers — the credentials, never a request's own target or signed
/// conditions; `None` closes it after this exchange.
#[allow(clippy::too_many_arguments)]
pub fn send<'l>(
    client: &Client,
    checked: &CheckedUrl,
    method: &str,
    headers: &[(String, String)],
    body: Option<Payload<'_>>,
    pool: Option<&[(String, String)]>,
    deadline: Deadline,
    ledger: &'l Ledger,
    judge: Judge,
    write_silence: Option<Duration>,
    fail: Code,
) -> Result<Response<'l>> {
    ledger.check_running()?;
    let opened = Instant::now();
    let scope = pool.map(|scoped| Scope::of(checked, scoped));
    let cached = scope.as_ref().and_then(|scope| client.take(scope, checked));
    let mut wire = if let Some(wire) = cached {
        wire
    } else {
        let sock = connect(checked, deadline, ledger, judge, fail)?;
        if checked.https {
            let name = ServerName::try_from(checked.host.clone()).map_err(|_| Refusal {
                code: Code::SOURCE_NOT_ALLOWED,
                detail: format!("{}: not a valid TLS server name", checked.host),
            })?;
            let conn = rustls::ClientConnection::new(client.tls(), name).map_err(|e| Refusal {
                code: fail,
                detail: format!("TLS setup: {e}"),
            })?;
            let mut sock = sock;
            let mut conn = Box::new(conn);
            tls_handshake(
                &checked.host,
                &mut sock,
                &mut conn,
                deadline,
                ledger,
                judge,
                opened,
                fail,
            )?;
            Wire::Tls { sock, conn }
        } else {
            Wire::Plain(sock)
        }
    };

    let default_port = if checked.https { 443 } else { 80 };
    let mut head = format!("{method} {} HTTP/1.1\r\n", checked.target);
    if checked.port == default_port {
        head.push_str(&format!("host: {}\r\n", checked.host));
    } else {
        head.push_str(&format!("host: {}:{}\r\n", checked.host, checked.port));
    }
    head.push_str(if scope.is_some() {
        "connection: keep-alive\r\naccept-encoding: identity\r\n"
    } else {
        "connection: close\r\naccept-encoding: identity\r\n"
    });
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    if let Some(body) = body {
        head.push_str(&format!("content-length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    let mut request_bytes = head.into_bytes();
    if let Some(Payload::Bytes(body)) = body {
        request_bytes.extend_from_slice(body);
    }
    write_all(
        &mut wire,
        &request_bytes,
        deadline,
        ledger,
        judge,
        opened,
        write_silence,
        fail,
    )?;
    if let Some(Payload::Object(file)) = body {
        let mut chunk = vec![0u8; file.len().min(BODY_CHUNK) as usize];
        let mut at = 0u64;
        while at < file.len() {
            let n = chunk.len().min((file.len() - at) as usize);
            file.read_exact_at(&mut chunk[..n], at)
                .map_err(|error| Refusal {
                    code: Code::IO_FAILED,
                    detail: format!("read request body at byte {at}: {error}"),
                })?;
            write_all(
                &mut wire,
                &chunk[..n],
                deadline,
                ledger,
                judge,
                opened,
                write_silence,
                fail,
            )?;
            at += n as u64;
        }
    }

    let (status, response_headers, spill, http11) =
        read_head(&mut wire, deadline, ledger, judge, opened, fail)?;
    if judge == Judge::Stream {
        ledger.taught(opened.elapsed());
    }
    // A HEAD THAT ARRIVED IS PROGRESS. Without this, the second page of a paginated closure
    // would be judged from the start of the first, and a pull whose hub is merely slow
    // would condemn its own second call for the first call's honest latency.
    ledger.answered();

    let framing = framing_of(status, &response_headers, fail)?;
    let reusable = http11
        && status >= 200
        && matches!(framing, Framing::Length { .. })
        && response_headers
            .iter()
            .filter(|(name, _)| name == "content-length")
            .count()
            == 1
        && !response_headers.iter().any(|(name, value)| {
            name == "transfer-encoding"
                || name == "connection"
                    && value
                        .split(',')
                        .any(|part| part.trim().eq_ignore_ascii_case("close"))
        });
    Ok(Response {
        status,
        headers: response_headers,
        body: BodyReader {
            wire: Some(wire),
            recycle: scope
                .filter(|_| reusable)
                .map(|scope| Box::new((Arc::clone(&client.idle), scope))),
            framing,
            raw: spill,
            raw_at: 0,
            deadline,
            ledger,
            judge,
            advanced: Instant::now(),
            since: Instant::now(),
            fail,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn write_all(
    wire: &mut Wire,
    bytes: &[u8],
    deadline: Deadline,
    ledger: &Ledger,
    judge: Judge,
    since: Instant,
    silence: Option<Duration>,
    fail: Code,
) -> Result<()> {
    let mut at = 0;
    let mut advanced = Instant::now();
    loop {
        transfer_verdict(judge, ledger, since, "a request write")?;
        let tick = match deadline.remaining()? {
            Some(left) => left.min(ledger.tick()),
            None => ledger.tick(),
        };
        wire.set_write_timeout(tick);
        if at >= bytes.len() {
            match wire.flush_step() {
                Ok(()) => return Ok(()),
                Err(e) if would_block(&e) => {}
                Err(e) => return refuse(fail, format!("the connection failed at flush: {e}")),
            }
        } else {
            let window = bytes.len().min(at + WRITE_CHUNK);
            match wire.write_step(&bytes[at..window]) {
                Ok(0) => return refuse(fail, "the connection accepted no bytes"),
                Ok(n) => {
                    at += n;
                    advanced = Instant::now();
                    continue;
                }
                Err(e) if would_block(&e) => {}
                Err(e) => return refuse(fail, format!("the connection failed mid-write: {e}")),
            }
        }
        handshake_verdict(judge, ledger, advanced, "request write", fail)?;
        if let Some(bound) = silence {
            if advanced.elapsed() > bound {
                return refuse(
                    fail,
                    format!(
                        "the socket moved nothing for {:.1} s of a write (bound {:.1} s)",
                        advanced.elapsed().as_secs_f64(),
                        bound.as_secs_f64()
                    ),
                );
            }
        }
    }
}

/// Status, headers (lowercased names), and body bytes read past the head.
type Head = (u16, Vec<(String, String)>, Vec<u8>, bool);

fn read_head(
    wire: &mut Wire,
    deadline: Deadline,
    ledger: &Ledger,
    judge: Judge,
    since: Instant,
    fail: Code,
) -> Result<Head> {
    let mut collected: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; 8 * 1024];
    let split = loop {
        if let Some(at) = collected
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
        {
            break at;
        }
        if collected.len() > HEAD_MAX_BYTES {
            return refuse(fail, "response head passed 64 KiB");
        }
        let n = read_step_ticking(wire, &mut buf, deadline, ledger, judge, since, fail)?;
        if n == 0 {
            return refuse(
                fail,
                "the origin closed the connection before a response head",
            );
        }
        collected.extend_from_slice(&buf[..n]);
    };
    let head_text = String::from_utf8_lossy(&collected[..split]).to_string();
    let spill = collected[split + 4..].to_vec();
    let mut lines = head_text.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let status = status_line
        .strip_prefix("HTTP/1.1 ")
        .or_else(|| status_line.strip_prefix("HTTP/1.0 "))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| Refusal {
            code: fail,
            detail: format!("not an HTTP/1.x status line: {status_line:?}"),
        })?;
    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    Ok((status, headers, spill, status_line.starts_with("HTTP/1.1 ")))
}

fn read_step_ticking(
    wire: &mut Wire,
    buf: &mut [u8],
    deadline: Deadline,
    ledger: &Ledger,
    judge: Judge,
    since: Instant,
    fail: Code,
) -> Result<usize> {
    let advanced = Instant::now();
    loop {
        transfer_verdict(judge, ledger, since, "a response head")?;
        let tick = match deadline.remaining()? {
            Some(left) => left.min(ledger.tick()),
            None => ledger.tick(),
        };
        wire.set_read_timeout(tick);
        match wire.read_step(buf, ledger) {
            Ok(n) => return Ok(n),
            Err(e) if would_block(&e) => {
                stream_verdict(judge, ledger, advanced, "response head")?;
            }
            Err(e) => return refuse(fail, format!("the connection failed: {e}")),
        }
    }
}

fn framing_of(status: u16, headers: &[(String, String)], fail: Code) -> Result<Framing> {
    let header = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    };
    if let Some(encoding) = header("content-encoding") {
        if !encoding.trim().is_empty() && !encoding.trim().eq_ignore_ascii_case("identity") {
            return refuse(
                fail,
                format!(
                    "origin transformed an exact-byte response with content-encoding \
                     {encoding:?} against `accept-encoding: identity`"
                ),
            );
        }
    }
    if status == 204 || status == 304 || (100..200).contains(&status) {
        return Ok(Framing::Length { remaining: 0 });
    }
    if let Some(te) = header("transfer-encoding") {
        if te.trim().eq_ignore_ascii_case("chunked") {
            return Ok(Framing::Chunked {
                in_chunk: 0,
                done: false,
            });
        }
        return refuse(fail, format!("transfer-encoding {te:?} is not chunked"));
    }
    match header("content-length") {
        Some(text) => {
            let remaining = text.trim().parse::<u64>().map_err(|_| Refusal {
                code: fail,
                detail: format!("content-length {text:?} is not a number"),
            })?;
            Ok(Framing::Length { remaining })
        }
        None => Ok(Framing::Eof),
    }
}
