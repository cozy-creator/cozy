//! The one socket this crate opens (tfs-049). A deliberately small HTTP/1.1 client — GET,
//! POST, PUT, `Connection: close`, Content-Length or chunked bodies, no compression, no
//! proxy, redirects surfaced to the caller rather than followed — written directly over
//! `TcpStream`/rustls for exactly two properties no shelf client offers together:
//!
//! - **the connection goes to the addresses the predicate CHECKED** (`CheckedUrl.addrs`),
//!   with SNI/Host still carrying the hostname, which closes the resolve-then-reconnect
//!   rebinding window `egress.py` could only name;
//! - **every socket wait is bounded by the caller's deadline and judged by the liveness
//!   ledger, never by an invented constant.** The socket read timeout here is the ledger's
//!   sampling RESOLUTION; each empty tick re-consults the deadline and, on a judged body,
//!   the measured stall rule. Nothing in this file owns a number of seconds.

use crate::err::{refuse, Code, Refusal, Result};
use crate::transport::ledger::{Deadline, Ledger};
use crate::transport::policy::CheckedUrl;
use rustls::pki_types::{CertificateDer, ServerName};
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

const HEAD_MAX_BYTES: usize = 64 * 1024;
const WRITE_CHUNK: usize = 64 * 1024;

fn would_block(error: &std::io::Error) -> bool {
    matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

pub struct Client {
    tls: Arc<rustls::ClientConfig>,
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
        }
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
    fn read_step(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Wire::Plain(sock) => sock.read(buf),
            Wire::Tls { sock, conn } => loop {
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

fn connect(checked: &CheckedUrl, deadline: Deadline, fail: Code) -> Result<TcpStream> {
    let mut last: Option<std::io::Error> = None;
    for addr in &checked.addrs {
        let attempt = match deadline.remaining()? {
            Some(left) => TcpStream::connect_timeout(addr, left),
            None => TcpStream::connect(addr),
        };
        match attempt {
            Ok(sock) => {
                let _ = sock.set_nodelay(true);
                return Ok(sock);
            }
            Err(error) => last = Some(error),
        }
    }
    refuse(
        fail,
        format!(
            "{}: no checked address accepted a connection ({})",
            checked.host,
            last.map(|e| e.to_string()).unwrap_or_default()
        ),
    )
}

fn tls_handshake(
    sock: &mut TcpStream,
    conn: &mut rustls::ClientConnection,
    deadline: Deadline,
    sample: Duration,
    fail: Code,
) -> Result<()> {
    while conn.is_handshaking() {
        let tick = match deadline.remaining()? {
            Some(left) => left.min(sample),
            None => sample,
        };
        let _ = sock.set_read_timeout(Some(tick.max(Duration::from_millis(1))));
        let _ = sock.set_write_timeout(Some(tick.max(Duration::from_millis(1))));
        if conn.wants_write() {
            match conn.write_tls(sock) {
                Ok(_) => {}
                Err(e) if would_block(&e) => {}
                Err(e) => return refuse(fail, format!("TLS handshake write: {e}")),
            }
            continue;
        }
        match conn.read_tls(sock) {
            Ok(0) => return refuse(fail, "origin closed the connection mid-handshake"),
            Ok(_) => {
                conn.process_new_packets().map_err(|e| Refusal {
                    code: fail,
                    detail: format!("TLS refused: {e}"),
                })?;
            }
            Err(e) if would_block(&e) => {}
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
    wire: Wire,
    framing: Framing,
    raw: Vec<u8>,
    raw_at: usize,
    deadline: Deadline,
    ledger: &'l Ledger,
    judge: bool,
    advanced: Instant,
    failure: Option<Refusal>,
    stalled: bool,
    fail: Code,
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
        loop {
            let tick = match self.deadline.remaining()? {
                Some(left) => left.min(self.ledger.sample()),
                None => self.ledger.sample(),
            };
            self.wire.set_read_timeout(tick);
            match self.wire.read_step(buf) {
                Ok(n) => {
                    if n > 0 {
                        let now = Instant::now();
                        self.ledger.taught(now - self.advanced);
                        self.advanced = now;
                    }
                    return Ok(n);
                }
                Err(e) if would_block(&e) => {
                    let silent = self.advanced.elapsed();
                    if self.judge && silent > self.ledger.patience() {
                        self.stalled = true;
                        let _ = self.wire.sock().shutdown(std::net::Shutdown::Both);
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

    pub fn read_refusing(&mut self, out: &mut [u8]) -> Result<usize> {
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

/// One exchange: connect to the CHECKED addresses, write the request, read the head, hand
/// back the framed body. `judge_body` arms the ledger's stall verdict on body reads (the
/// object plane); hub JSON exchanges leave it off and answer to the deadline alone, as
/// `pull.py`'s `_json` did. `write_silence` bounds one WRITE operation that moves nothing
/// — the push plane derives it from the ledger's floor, the port of egress.py's one
/// derived silence bound; `None` answers to the deadline alone.
#[allow(clippy::too_many_arguments)]
pub fn request<'l>(
    client: &Client,
    checked: &CheckedUrl,
    method: &str,
    headers: &[(String, String)],
    body: Option<&[u8]>,
    deadline: Deadline,
    ledger: &'l Ledger,
    judge_body: bool,
    write_silence: Option<Duration>,
    fail: Code,
) -> Result<Response<'l>> {
    let opened = Instant::now();
    let sock = connect(checked, deadline, fail)?;
    let mut wire = if checked.https {
        let name = ServerName::try_from(checked.host.clone()).map_err(|_| Refusal {
            code: Code::SOURCE_NOT_ALLOWED,
            detail: format!("{}: not a valid TLS server name", checked.host),
        })?;
        let conn =
            rustls::ClientConnection::new(Arc::clone(&client.tls), name).map_err(|e| Refusal {
                code: fail,
                detail: format!("TLS setup: {e}"),
            })?;
        let mut sock = sock;
        let mut conn = Box::new(conn);
        tls_handshake(&mut sock, &mut conn, deadline, ledger.sample(), fail)?;
        Wire::Tls { sock, conn }
    } else {
        Wire::Plain(sock)
    };

    let default_port = if checked.https { 443 } else { 80 };
    let mut head = format!("{method} {} HTTP/1.1\r\n", checked.target);
    if checked.port == default_port {
        head.push_str(&format!("host: {}\r\n", checked.host));
    } else {
        head.push_str(&format!("host: {}:{}\r\n", checked.host, checked.port));
    }
    head.push_str("connection: close\r\naccept-encoding: identity\r\n");
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    if let Some(body) = body {
        head.push_str(&format!("content-length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");
    let mut request_bytes = head.into_bytes();
    if let Some(body) = body {
        request_bytes.extend_from_slice(body);
    }
    write_all(
        &mut wire,
        &request_bytes,
        deadline,
        ledger,
        write_silence,
        fail,
    )?;

    let (status, response_headers, spill) = read_head(&mut wire, deadline, ledger, fail)?;
    ledger.taught(opened.elapsed());

    let framing = framing_of(status, &response_headers, fail)?;
    Ok(Response {
        status,
        headers: response_headers,
        body: BodyReader {
            wire,
            framing,
            raw: spill,
            raw_at: 0,
            deadline,
            ledger,
            judge: judge_body,
            advanced: Instant::now(),
            failure: None,
            stalled: false,
            fail,
        },
    })
}

fn write_all(
    wire: &mut Wire,
    bytes: &[u8],
    deadline: Deadline,
    ledger: &Ledger,
    silence: Option<Duration>,
    fail: Code,
) -> Result<()> {
    let mut at = 0;
    let mut advanced = Instant::now();
    loop {
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
type Head = (u16, Vec<(String, String)>, Vec<u8>);

fn read_head(wire: &mut Wire, deadline: Deadline, ledger: &Ledger, fail: Code) -> Result<Head> {
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
        let n = read_step_ticking(wire, &mut buf, deadline, ledger, fail)?;
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
    Ok((status, headers, spill))
}

fn read_step_ticking(
    wire: &mut Wire,
    buf: &mut [u8],
    deadline: Deadline,
    ledger: &Ledger,
    fail: Code,
) -> Result<usize> {
    loop {
        let tick = match deadline.remaining()? {
            Some(left) => left.min(ledger.sample()),
            None => ledger.sample(),
        };
        wire.set_read_timeout(tick);
        match wire.read_step(buf) {
            Ok(n) => return Ok(n),
            Err(e) if would_block(&e) => {}
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
