//! The one socket this crate opens (tfs-049). A deliberately small HTTP/1.1 client — GET,
//! POST, PUT, bounded reuse for exact ranged GETs, Content-Length or chunked bodies, no compression, no
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
use rustls::pki_types::{CertificateDer, ServerName};
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
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
    if judge != Judge::Stream || advanced.elapsed() <= ledger.patience() {
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

pub struct Client {
    tls: Arc<rustls::ClientConfig>,
    idle: IdlePool,
}

type IdlePool = Arc<Mutex<Vec<Idle>>>;

// This bounds retained descriptors, not in-flight transfers (the caller owns those).
const MAX_IDLE: usize = 8;

#[derive(PartialEq, Eq)]
struct Scope {
    https: bool,
    host: String,
    port: u16,
    // Do not retain signed targets or credentials in an idle connection's key.
    request: [u8; 32],
}

impl Scope {
    fn ranged(checked: &CheckedUrl, headers: &[(String, String)]) -> Self {
        let mut hash = crate::sha256::Sha256::new();
        hash.update(&(checked.target.len() as u64).to_le_bytes());
        hash.update(checked.target.as_bytes());
        for (name, value) in headers
            .iter()
            .filter(|(name, _)| !name.eq_ignore_ascii_case("range"))
        {
            // Preserve duplicate-header ordering as well as values: an origin
            // can assign different meaning to their first or last occurrence.
            let name = name.to_ascii_lowercase();
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

impl Client {
    pub fn new() -> Client {
        Client::with_extra_roots(&[])
    }

    /// `extra_roots` exists for tests that stand up a loopback TLS origin under a
    /// self-signed authority; production callers pass none and trust the webpki set.
    pub fn with_extra_roots(extra_roots: &[CertificateDer<'static>]) -> Client {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        for root in extra_roots {
            let _ = roots.add(root.clone());
        }
        let config = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .expect("rustls ships its own default protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
        Client {
            tls: Arc::new(config),
            idle: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn take(&self, scope: &Scope, checked: &CheckedUrl) -> Option<Wire> {
        let mut idle = self.idle.lock().ok()?;
        let at = idle.iter().position(|entry| entry.scope == *scope)?;
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

fn connect(
    checked: &CheckedUrl,
    deadline: Deadline,
    ledger: &Ledger,
    judge: Judge,
    fail: Code,
) -> Result<TcpStream> {
    use rustix::event::{poll, PollFd, PollFlags, Timespec};
    use rustix::net::{self, AddressFamily};
    let mut last: Option<std::io::Error> = None;
    'addresses: for addr in &checked.addrs {
        ledger.check_running()?;
        let family = if addr.is_ipv4() {
            AddressFamily::INET
        } else {
            AddressFamily::INET6
        };
        let socket = match nonblocking_socket(family) {
            Ok(socket) => socket,
            Err(error) => {
                last = Some(error.into());
                continue;
            }
        };
        let since = Instant::now();
        match net::connect(&socket, addr) {
            Ok(()) => {}
            Err(rustix::io::Errno::INPROGRESS) => loop {
                // Poll one pending connection instead of restarting it at each
                // sample. Cancellation and existing silence rules remain active
                // without imposing a new total connection deadline.
                transfer_verdict(judge, ledger, since, "a TCP connection")?;
                stream_verdict(judge, ledger, since, "TCP connect")?;
                let tick = deadline
                    .remaining()?
                    .map_or(ledger.sample(), |left| left.min(ledger.sample()));
                let timeout = Timespec {
                    tv_sec: tick.as_secs() as _,
                    tv_nsec: tick.subsec_nanos() as _,
                };
                let mut descriptors = [PollFd::new(&socket, PollFlags::OUT)];
                match poll(&mut descriptors, Some(&timeout)) {
                    Ok(0) | Err(rustix::io::Errno::INTR) => continue,
                    Ok(_) => match net::sockopt::socket_error(&socket) {
                        Ok(Ok(())) => break,
                        Ok(Err(error)) | Err(error) => {
                            last = Some(error.into());
                            continue 'addresses;
                        }
                    },
                    Err(error) => {
                        last = Some(error.into());
                        continue 'addresses;
                    }
                }
            },
            Err(error) => {
                last = Some(error.into());
                continue;
            }
        }
        let socket: TcpStream = socket.into();
        socket.set_nonblocking(false).map_err(|error| Refusal {
            code: fail,
            detail: format!("configure connected socket: {error}"),
        })?;
        let _ = socket.set_nodelay(true);
        return Ok(socket);
    }
    refuse(
        fail,
        format!(
            "{}: no checked address accepted a connection ({})",
            checked.host,
            last.map(|error| error.to_string()).unwrap_or_default()
        ),
    )
}

fn tls_handshake(
    sock: &mut TcpStream,
    conn: &mut rustls::ClientConnection,
    deadline: Deadline,
    ledger: &Ledger,
    judge: Judge,
    since: Instant,
    fail: Code,
) -> Result<()> {
    let sample = ledger.sample();
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
                    stream_verdict(judge, ledger, advanced, "TLS handshake")?;
                }
                Err(e) => return refuse(fail, format!("TLS handshake write: {e}")),
            }
            continue;
        }
        match conn.read_tls(sock) {
            Ok(0) => return refuse(fail, "origin closed the connection mid-handshake"),
            Ok(_) => {
                advanced = Instant::now();
                conn.process_new_packets().map_err(|e| Refusal {
                    code: fail,
                    detail: format!("TLS refused: {e}"),
                })?;
            }
            Err(e) if would_block(&e) => {
                stream_verdict(judge, ledger, advanced, "TLS handshake")?;
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
/// patience is a stall: the typed refusal is stashed here (a `Read` can only carry
/// `io::Error`) and `failure()`/`stalled()` hand it back to the retry loop.
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
    failure: Option<Refusal>,
    stalled: bool,
    fail: Code,
}

impl Drop for BodyReader<'_> {
    fn drop(&mut self) {
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
        idle.retain(|entry| entry.scope != scope);
        if idle.len() == MAX_IDLE {
            idle.remove(0);
        }
        idle.push(Idle { scope, wire });
    }
}

impl BodyReader<'_> {
    pub fn failure(&mut self) -> Option<Refusal> {
        self.failure.take()
    }
    pub fn stalled(&self) -> bool {
        self.stalled
    }

    /// One raw read off the wire under the sampling rule: the socket timeout is the
    /// RESOLUTION; the deadline and (when judged) the measured patience carry the verdict.
    fn step(&mut self, buf: &mut [u8]) -> Result<usize> {
        // Time spent by the consumer hashing or writing the prior chunk is not
        // network silence. This wait starts when it asks for another chunk.
        self.advanced = Instant::now();
        loop {
            self.ledger.check_running()?;
            let tick = match self.deadline.remaining()? {
                Some(left) => left.min(self.ledger.sample()),
                None => self.ledger.sample(),
            };
            let wire = self.wire.as_mut().expect("body owns its wire");
            wire.set_read_timeout(tick);
            let waiting = Instant::now();
            if self.judge == Judge::Stream {
                self.ledger.network_read_begins();
            }
            let result = wire.read_step(buf, self.ledger);
            if self.judge == Judge::Stream {
                self.ledger.network_waited(waiting.elapsed());
            }
            match result {
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
                    if self.judge == Judge::Stream && silent > self.ledger.patience() {
                        self.stalled = true;
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
        self.read_refusing(out).map_err(|refusal| {
            let text = refusal.to_string();
            self.failure = Some(refusal);
            std::io::Error::other(text)
        })
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
    send(
        client,
        checked,
        method,
        headers,
        body.map(Payload::Bytes),
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
#[allow(clippy::too_many_arguments)]
pub fn send<'l>(
    client: &Client,
    checked: &CheckedUrl,
    method: &str,
    headers: &[(String, String)],
    body: Option<Payload<'_>>,
    deadline: Deadline,
    ledger: &'l Ledger,
    judge: Judge,
    write_silence: Option<Duration>,
    fail: Code,
) -> Result<Response<'l>> {
    ledger.check_running()?;
    let opened = Instant::now();
    // Retained-source ranges already own their retry/resume budget. Other GETs,
    // metadata writes and uploads retain their existing one-connection behavior.
    let scope = (method == "GET"
        && body.is_none()
        && headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("range"))
        && !headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("connection")))
    .then(|| Scope::ranged(checked, headers));
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
            let conn =
                rustls::ClientConnection::new(Arc::clone(&client.tls), name).map_err(|e| {
                    Refusal {
                        code: fail,
                        detail: format!("TLS setup: {e}"),
                    }
                })?;
            let mut sock = sock;
            let mut conn = Box::new(conn);
            tls_handshake(&mut sock, &mut conn, deadline, ledger, judge, opened, fail)?;
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
            failure: None,
            stalled: false,
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
            Some(left) => left.min(ledger.sample()),
            None => ledger.sample(),
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
        stream_verdict(judge, ledger, advanced, "request write")?;
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
            Some(left) => left.min(ledger.sample()),
            None => ledger.sample(),
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
