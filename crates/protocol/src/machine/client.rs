//! A typed client of one machine's API. TLS admits exactly the machine's pinned leaf. Control
//! calls and SSE share one HTTP/2 connection, which a PING every 20 s keeps proven alive; bulk
//! bytes go on [`Bulk`] HTTP/1.1 connections of their own, so parts and ranges move side by side
//! on separate TCP connections. Nothing here times out on a clock: a dead control connection
//! fails its PING, and bulk callers judge their lanes by measured progress.
use std::{
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use ed25519_dalek::SigningKey;
use eventsource_stream::Eventsource as _;
use futures_util::{Stream, StreamExt as _, TryStreamExt as _};
use reqwest::{Method, RequestBuilder, Response, StatusCode, header};
use rustls::{
    DigitallySignedStruct, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    crypto::{CryptoProvider, verify_tls12_signature, verify_tls13_signature},
    pki_types::{CertificateDer, ServerName, UnixTime},
};
use serde::de::DeserializeOwned;
use sha2::{Digest as _, Sha256};
use url::Url;

use super::{
    Event, HELD, HeldAnswer, HeldBytes, HeldObject, HeldRequest, LogLines, OutputTarget, Ranges,
    RunEvent, RunRequest, RunState, StatusFrame, TensorFSOptions, parse_held, path,
};
use crate::{
    capability,
    error::{Envelope, Kind},
};

/// The name every machine's self-signed leaf carries.
pub const SERVER_NAME: &str = "cozy-worker";
/// A cap is checked when a request opens; an open stream outlives it.
const CAP_SECONDS: i64 = 300;
const PING: Duration = Duration::from_secs(20);

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    /// The machine refused: its HTTP status and envelope.
    #[error("the machine refused it ({}): {}", .error.code, .error.message)]
    Refused {
        status: u16,
        error: Box<crate::error::Error>,
    },
    /// Not reached, or the connection broke.
    #[error("{0}")]
    Transport(String),
    /// It answered what this API does not say: another service on its port, a proxy's error page.
    #[error("{0}")]
    Answer(String),
}

impl Error {
    /// The HTTP status of a refusal.
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::Refused { status, .. } => Some(*status),
            _ => None,
        }
    }
    /// The refusal's stable code.
    pub fn code(&self) -> Option<&str> {
        match self {
            Error::Refused { error, .. } => Some(&error.code),
            _ => None,
        }
    }
    pub fn is_not_found(&self) -> bool {
        self.status() == Some(404)
    }
}

/// How each request is authorized.
#[derive(Clone)]
pub enum Auth {
    /// Nothing: `status` answers identity and the readiness receipt only.
    None,
    /// A fresh machine-scope cap for every request, signed by `key` for the machine `worker`.
    Key {
        key: Arc<SigningKey>,
        worker: String,
    },
    /// One token for every request (a run-scope play link's).
    Token(String),
}

impl Auth {
    fn header(&self) -> Option<String> {
        let token = match self {
            Auth::None => return None,
            Auth::Token(token) => token.clone(),
            Auth::Key { key, worker } => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs() as i64);
                let grant = capability::Grant {
                    machine: worker.clone(),
                    action: capability::MACHINE.into(),
                    expires: now + CAP_SECONDS,
                    ..Default::default()
                };
                capability::mint(key, grant)
            }
        };
        Some(format!("{}{token}", capability::SCHEME))
    }
}

/// One machine, reached at `address` (`host:port`) through its pinned leaf.
#[derive(Clone)]
pub struct Machine {
    origin: Url,
    pin: [u8; 32],
    auth: Auth,
    control: reqwest::Client,
}

/// What a ranged `GET` of an output answered: the revision, the output's whole length, the
/// range its body carries (`[first, end)`), and the bytes.
pub struct Output {
    pub rev: u64,
    pub length: u64,
    pub range: (u64, u64),
    pub media_type: String,
    pub body: std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
}

impl Machine {
    pub fn new(address: &str, pin: [u8; 32], auth: Auth) -> Result<Self, Error> {
        let origin = Url::parse(&format!("https://{address}/"))
            .map_err(|e| Error::Transport(format!("the machine's address {address}: {e}")))?;
        let control = builder(pin, &[b"h2"])
            .http2_keep_alive_interval(PING)
            .http2_keep_alive_timeout(PING)
            .http2_keep_alive_while_idle(true)
            .build()
            .map_err(transport)?;
        Ok(Self {
            origin,
            pin,
            auth,
            control,
        })
    }

    /// This machine with requests authorized by `auth` instead, on the same connections.
    pub fn with_auth(&self, auth: Auth) -> Self {
        Self {
            auth,
            ..self.clone()
        }
    }

    /// A bulk lane: HTTP/1.1 connections of its own. Dropping it closes them.
    pub fn bulk(&self) -> Result<Bulk, Error> {
        let client = builder(self.pin, &[])
            .http1_only()
            .tcp_keepalive(PING)
            .build()
            .map_err(transport)?;
        Ok(Bulk {
            machine: self.clone(),
            client,
        })
    }

    /// The machine's frame. Without a credential: identity, version, capabilities and receipt.
    pub async fn status(&self) -> Result<StatusFrame, Error> {
        let request = self.request(Method::GET, path::STATUS, &[]);
        json(send(request.header(header::ACCEPT, "application/json")).await?).await
    }

    /// The machine's frame, then again whenever it changes.
    pub async fn watch(
        &self,
    ) -> Result<impl Stream<Item = Result<StatusFrame, Error>> + Send + 'static, Error> {
        let request = self.request(Method::GET, path::STATUS, &[]);
        let events = sse(send(request.header(header::ACCEPT, "text/event-stream")).await?)?;
        Ok(events.filter_map(|event| async move {
            match event {
                Ok(event) if event.event == "status" => Some(parse(&event.data)),
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            }
        }))
    }

    /// Starts a rental's idle clock over; answers the frame with the new deadline.
    pub async fn keepalive(&self) -> Result<StatusFrame, Error> {
        json(send(self.request(Method::POST, path::KEEPALIVE, &[])).await?).await
    }

    /// Submits a run; its state, also when it was accepted before.
    pub async fn submit(&self, run: &RunRequest) -> Result<RunState, Error> {
        json(send(self.request(Method::POST, path::RUNS, &[]).json(run)).await?).await
    }

    /// The run's log after `after`: its state first, then its events until its outcome. A run
    /// the machine does not have is a 404 refusal.
    pub async fn events(
        &self,
        id: &str,
        after: u64,
    ) -> Result<impl Stream<Item = Result<RunEvent, Error>> + Send + 'static, Error> {
        let mut request = self.request(Method::GET, path::EVENTS, &[id]);
        request = request.header(header::ACCEPT, "text/event-stream");
        if after > 0 {
            request = request.header("last-event-id", after.to_string());
        }
        let events = sse(send(request).await?)?;
        Ok(events.filter_map(|event| async move {
            match event {
                Ok(event) if Event::NAMES.contains(&event.event.as_str()) => {
                    Some(parse(&event.data))
                }
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            }
        }))
    }

    /// Cancels the run; an id the machine never received is recorded canceled.
    pub async fn cancel(&self, id: &str) -> Result<RunState, Error> {
        json(send(self.request(Method::POST, path::CANCEL, &[id])).await?).await
    }

    /// A failed attempt's triage bundle (JSON).
    pub async fn triage(&self, id: &str) -> Result<Bytes, Error> {
        bytes(send(self.request(Method::GET, path::TRIAGE, &[id])).await?).await
    }

    /// The newest `lines` whole lines (0: all) of a log the machine keeps.
    pub async fn log(&self, name: &str, lines: u64) -> Result<Bytes, Error> {
        let request = self.request(Method::GET, path::LOG, &[name]);
        bytes(send(request.query(&[("lines", lines)])).await?).await
    }

    /// A log's lines as [`Self::log`] reads them, or those after `after` (a batch's place), then
    /// each batch as the machine appends it.
    pub async fn follow_log(
        &self,
        name: &str,
        lines: u64,
        after: Option<&str>,
    ) -> Result<impl Stream<Item = Result<LogLines, Error>> + Send + 'static, Error> {
        let mut request = self.request(Method::GET, path::LOG, &[name]);
        request = match after {
            Some(after) => request.header("last-event-id", after),
            None => request.query(&[("lines", lines)]),
        };
        let events = sse(send(request.header(header::ACCEPT, "text/event-stream")).await?)?;
        Ok(events.filter_map(|event| async move {
            match event {
                Ok(event) if event.event == "lines" => Some(parse(&event.data)),
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            }
        }))
    }

    /// The ranges of the object (`sha256:<hex>`) the machine holds for this signer; a stored
    /// object is held whole.
    pub async fn held(&self, digest: &str) -> Result<Ranges, Error> {
        held(&send(self.request(Method::HEAD, path::OBJECT, &[digest])).await?)
    }

    /// What the machine holds of the object (`sha256:<hex>`, `length` bytes) as its upload lands,
    /// until it holds all of it.
    pub async fn follow_held(
        &self,
        digest: &str,
        length: u64,
    ) -> Result<impl Stream<Item = Result<u64, Error>> + Send + 'static, Error> {
        let request = self.request(Method::GET, path::OBJECT, &[digest]);
        let request = request.query(&[("length", length)]);
        let events = sse(send(request.header(header::ACCEPT, "text/event-stream")).await?)?;
        Ok(events.filter_map(|event| async move {
            match event {
                Ok(event) if event.event == "held" => {
                    Some(parse::<HeldBytes>(&event.data).map(|h| h.held))
                }
                Ok(_) => None,
                Err(e) => Some(Err(e)),
            }
        }))
    }

    /// What the machine holds of each of `digests`, in one request, in the order asked.
    pub async fn held_many(&self, digests: &[String]) -> Result<Vec<HeldObject>, Error> {
        let ask = HeldRequest {
            digests: digests.to_vec(),
        };
        let request = self
            .request(Method::POST, path::OBJECTS_HELD, &[])
            .json(&ask);
        let answer: HeldAnswer = json(send(request).await?).await?;
        Ok(answer.held)
    }

    /// Gives the machine this signer's newest bearer at the Hub `bearer.endpoint` names.
    pub async fn bearer(&self, bearer: &TensorFSOptions) -> Result<(), Error> {
        send(self.request(Method::PUT, path::BEARER, &[]).json(bearer))
            .await
            .map(drop)
    }

    fn request(&self, method: Method, template: &str, args: &[&str]) -> RequestBuilder {
        authorized(
            self.control.request(method, self.url(template, args)),
            &self.auth,
        )
    }

    /// `template` with each `{parameter}` filled from `args`, in order, as one path segment.
    fn url(&self, template: &str, args: &[&str]) -> Url {
        let mut url = self.origin.clone();
        let mut args = args.iter();
        url.path_segments_mut()
            .expect("an https URL has a path")
            .clear()
            .extend(template.trim_start_matches('/').split('/').map(|part| {
                match part.starts_with('{') {
                    true => *args.next().expect("an argument for each parameter"),
                    false => part,
                }
            }));
        url
    }
}

/// HTTP/1.1 connections of one bulk lane.
#[derive(Clone)]
pub struct Bulk {
    machine: Machine,
    client: reqwest::Client,
}

impl Bulk {
    /// Lands `[first, first + length)` of the object (`sha256:<hex>`, `total` bytes) from `body`;
    /// answers the ranges held after it.
    pub async fn put(
        &self,
        digest: &str,
        (first, length, total): (u64, u64, u64),
        body: impl Into<reqwest::Body>,
    ) -> Result<Ranges, Error> {
        let mut request = self.request(Method::PUT, path::OBJECT, &[digest]);
        if total > 0 {
            let last = first + length.max(1) - 1;
            request = request.header(
                header::CONTENT_RANGE,
                format!("bytes {first}-{last}/{total}"),
            );
        }
        let request = request
            .header(header::CONTENT_LENGTH, length)
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(body);
        held(&send(request).await?)
    }

    /// `range` (`[first, end)`; an empty one: all of it) of an output at revision `rev` (0: its
    /// newest). A moved revision is a 412 refusal, a range past its end a 416.
    pub async fn get(
        &self,
        target: &OutputTarget,
        (first, end): (u64, u64),
        rev: u64,
    ) -> Result<Output, Error> {
        let mut request = self.request(Method::GET, path::OUTPUT, &[&target.run, &target.output]);
        if target.index > 0 {
            request = request.query(&[("index", target.index)]);
        }
        if !target.member.is_empty() {
            request = request.query(&[("member", &target.member)]);
        }
        if end > first {
            request = request.header(header::RANGE, format!("bytes={first}-{}", end - 1));
        }
        if rev > 0 {
            request = request.header(header::IF_MATCH, format!("\"{rev}\""));
        }
        let response = send(request).await?;
        let headers = response.headers();
        let text = |name: header::HeaderName| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_owned()
        };
        let unexpected = |what: &str| Error::Answer(format!("an output read answered {what}"));
        let rev = text(header::ETAG)
            .trim_matches('"')
            .parse()
            .map_err(|_| unexpected("no revision"))?;
        let media_type = text(header::CONTENT_TYPE);
        let (range, length) = match response.status() {
            StatusCode::PARTIAL_CONTENT => {
                let spelled = text(header::CONTENT_RANGE);
                content_range(&spelled)
                    .ok_or_else(|| unexpected(&format!("Content-Range {spelled:?}")))?
            }
            _ => {
                let length = response
                    .content_length()
                    .ok_or_else(|| unexpected("no length"))?;
                ((0, length), length)
            }
        };
        let body = response.bytes_stream().map_err(transport);
        Ok(Output {
            rev,
            length,
            range,
            media_type,
            body: Box::pin(body),
        })
    }

    fn request(&self, method: Method, template: &str, args: &[&str]) -> RequestBuilder {
        let url = self.machine.url(template, args);
        authorized(self.client.request(method, url), &self.machine.auth)
    }
}

/// `bytes first-last/length` as (`[first, end)`, length).
fn content_range(value: &str) -> Option<((u64, u64), u64)> {
    let (range, length) = value.strip_prefix("bytes ")?.split_once('/')?;
    let (first, last) = range.split_once('-')?;
    let (first, last): (u64, u64) = (first.parse().ok()?, last.parse().ok()?);
    Some(((first, last + 1), length.parse().ok()?))
}

fn authorized(request: RequestBuilder, auth: &Auth) -> RequestBuilder {
    match auth.header() {
        Some(value) => request.header(header::AUTHORIZATION, value),
        None => request,
    }
}

fn builder(pin: [u8; 32], alpn: &[&[u8]]) -> reqwest::ClientBuilder {
    let mut tls = tls(Arc::new(Pinned::new(pin)));
    tls.alpn_protocols = alpn.iter().map(|p| p.to_vec()).collect();
    reqwest::Client::builder()
        .tls_backend_preconfigured(tls)
        .tcp_nodelay(true)
        .no_proxy()
}

/// Sends `request`: a 2xx response, or the machine's refusal.
async fn send(request: RequestBuilder) -> Result<Response, Error> {
    let response = request.send().await.map_err(transport)?;
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.bytes().await.unwrap_or_default();
    let error = match serde_json::from_slice::<Envelope>(&body) {
        Ok(envelope) => envelope.error,
        // A HEAD's refusal has no body; another service's is not ours.
        Err(_) if body.is_empty() => crate::error::Error::new(
            kind(status),
            format!("http_{}", status.as_u16()),
            status.canonical_reason().unwrap_or("refused"),
        ),
        Err(_) => {
            return Err(Error::Answer(format!(
                "the machine answered {status} without the error envelope: {}",
                String::from_utf8_lossy(&body[..body.len().min(200)])
            )));
        }
    };
    Err(Error::Refused {
        status: status.as_u16(),
        error: Box::new(error),
    })
}

fn kind(status: StatusCode) -> Kind {
    match status.as_u16() {
        401 => Kind::Credential,
        403 => Kind::Forbidden,
        404 => Kind::NotFound,
        409 | 412 => Kind::Conflict,
        507 => Kind::Capacity,
        400..=499 => Kind::Validation,
        503 => Kind::Unavailable,
        _ => Kind::Internal,
    }
}

async fn json<T: DeserializeOwned>(response: Response) -> Result<T, Error> {
    parse(&String::from_utf8_lossy(&bytes(response).await?))
}

fn parse<T: DeserializeOwned>(text: &str) -> Result<T, Error> {
    serde_json::from_str(text)
        .map_err(|e| Error::Answer(format!("the machine answered unreadable JSON: {e}")))
}

async fn bytes(response: Response) -> Result<Bytes, Error> {
    response.bytes().await.map_err(transport)
}

fn held(response: &Response) -> Result<Ranges, Error> {
    let value = response
        .headers()
        .get(HELD)
        .map(|v| v.to_str().unwrap_or("-"));
    value
        .and_then(parse_held)
        .ok_or_else(|| Error::Answer(format!("the machine answered {HELD} {value:?}")))
}

fn sse(
    response: Response,
) -> Result<impl Stream<Item = Result<eventsource_stream::Event, Error>> + Send + 'static, Error> {
    let streams = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    if !streams {
        return Err(Error::Answer(
            "the machine answered a stream that is not SSE".into(),
        ));
    }
    Ok(response.bytes_stream().eventsource().map_err(|e| match e {
        eventsource_stream::EventStreamError::Transport(e) => transport(e),
        e => Error::Answer(format!("the machine's SSE: {e}")),
    }))
}

/// A transport failure with its causes.
fn transport(error: reqwest::Error) -> Error {
    let mut message = error.to_string();
    let mut source = std::error::Error::source(&error);
    while let Some(cause) = source {
        message = format!("{message}: {cause}");
        source = cause.source();
    }
    Error::Transport(message)
}

/// The TLS leaf (DER) the machine at `address` presents, whatever it is. Nothing is sent: the
/// handshake is refused once the leaf is seen.
pub async fn presented_leaf(address: &str) -> Result<Vec<u8>, Error> {
    let unreachable = |e: std::io::Error| Error::Transport(format!("{address}: {e}"));
    let pinned = Arc::new(Pinned::new([0; 32]));
    let tcp = tokio::net::TcpStream::connect(address)
        .await
        .map_err(unreachable)?;
    let name = ServerName::try_from(SERVER_NAME).expect("a DNS name");
    let refused = tokio_rustls::TlsConnector::from(Arc::new(tls(pinned.clone())))
        .connect(name, tcp)
        .await;
    let seen = pinned.seen.lock().unwrap_or_else(|e| e.into_inner()).take();
    match (seen, refused) {
        (Some(leaf), _) => Ok(leaf),
        (None, Err(e)) => Err(unreachable(e)),
        (None, Ok(_)) => Err(Error::Transport(format!("{address} presented no TLS leaf"))),
    }
}

/// TLS that admits exactly the leaf whose sha256 is the pin, whatever its name or dates say.
fn tls(pinned: Arc<Pinned>) -> rustls::ClientConfig {
    rustls::ClientConfig::builder_with_provider(pinned.provider.clone())
        .with_safe_default_protocol_versions()
        .expect("aws-lc-rs offers the default protocol versions")
        .dangerous()
        .with_custom_certificate_verifier(pinned)
        .with_no_client_auth()
}

#[derive(Debug)]
struct Pinned {
    pin: [u8; 32],
    provider: Arc<CryptoProvider>,
    /// The leaf the peer presented last.
    seen: Mutex<Option<Vec<u8>>>,
}

impl Pinned {
    fn new(pin: [u8; 32]) -> Self {
        Self {
            pin,
            provider: Arc::new(rustls::crypto::aws_lc_rs::default_provider()),
            seen: Mutex::default(),
        }
    }
}

impl ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &ServerName<'_>,
        _: &[u8],
        _: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        *self.seen.lock().unwrap_or_else(|e| e.into_inner()) = Some(end_entity.to_vec());
        if Sha256::digest(end_entity.as_ref()).as_slice() == self.pin {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General(
                "the machine presented another TLS leaf than the one pinned".into(),
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}
