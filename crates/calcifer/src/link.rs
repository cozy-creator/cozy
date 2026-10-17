//! One machine's `cozy.machine.v1` API: a TLS connection that admits only the machine's pinned
//! leaf, every call carrying a short machine-scope Cozy-Cap signed by the key the machine scopes
//! its runs to.
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use cozy_proto::{capability, ids::Digest, machine as v1};
use ed25519_dalek::SigningKey;
use hyper_util::rt::TokioIo;
use rustls::{
    DigitallySignedStruct, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    crypto::{CryptoProvider, verify_tls12_signature, verify_tls13_signature},
    pki_types::{CertificateDer, ServerName, UnixTime},
};
use sha2::{Digest as _, Sha256};
use tokio::io::{AsyncReadExt as _, AsyncSeekExt as _};
use tonic::{
    Code, Status, Streaming,
    codegen::http::Uri,
    transport::{Channel, Endpoint},
};
use v1::machine_client::MachineClient;

use crate::store;

/// The name every machine's self-signed leaf carries.
const SERVER_NAME: &str = "cozy-worker";
/// An upload's frame: small, so what waits to be taken is little.
const FRAME: usize = 64 << 10;
/// The most a connection keeps unsent in the kernel.
const UNSENT: u32 = 256 << 10;
/// How often an upload's progress is told.
const TELL_EVERY: Duration = Duration::from_millis(250);
/// A cap is checked when a call opens; an open stream outlives it.
const CAP_SECONDS: i64 = 300;

#[derive(Clone, Debug, thiserror::Error)]
pub enum LinkError {
    #[error("the machine has no such run")]
    NotFound,
    /// A refusal the same request cannot change: the machine will not take it.
    #[error("the machine refused it ({code}): {message}")]
    Refused { code: String, message: String },
    /// The machine released itself (a rental's idle end): it is gone.
    #[error("the machine released itself: {0}")]
    Released(String),
    /// The machine does not admit the key (yet).
    #[error("the machine does not admit this key: {0}")]
    Unauthorized(String),
    /// Unreachable, starting, updating, or not admitting the key yet: try again.
    #[error("{0}")]
    Unreachable(String),
}

#[derive(Clone)]
pub struct Link {
    client: MachineClient<Channel>,
    key: Arc<SigningKey>,
    worker: String,
}

impl Link {
    /// A connection to `endpoint`, opened on first use and again whenever it breaks.
    pub fn new(endpoint: &store::Endpoint, key: SigningKey) -> Result<Self, LinkError> {
        let tls = Arc::new(tls(Arc::new(Pinned::new(endpoint.pin))));
        let address = endpoint.address.clone();
        let connector = tower::service_fn(move |_: Uri| {
            let (tls, address) = (tls.clone(), address.clone());
            async move {
                let tcp = tokio::net::TcpStream::connect(&address).await?;
                tcp.set_nodelay(true)?;
                // Little waits unsent in the kernel, so an upload's count of what it handed over
                // is close to what crossed (`Link::write`).
                if let Err(e) = cozy_platform::net::bound_unsent(&tcp, UNSENT)
                    && !e.is_unsupported()
                {
                    tracing::warn!("{e}");
                }
                let name = ServerName::try_from(SERVER_NAME).expect("a DNS name");
                let stream = tokio_rustls::TlsConnector::from(tls)
                    .connect(name, tcp)
                    .await?;
                Ok::<_, std::io::Error>(TokioIo::new(stream))
            }
        });
        let channel = Endpoint::from_shared(format!("http://{}", endpoint.address))
            .map_err(|e| {
                LinkError::Unreachable(format!("the machine's address {}: {e}", endpoint.address))
            })?
            .initial_stream_window_size(16 << 20)
            .initial_connection_window_size(32 << 20)
            // A ping a silent peer does not answer ends the connection; a quiet run is not that.
            .http2_keep_alive_interval(Duration::from_secs(20))
            .keep_alive_timeout(Duration::from_secs(20))
            .keep_alive_while_idle(true)
            .connect_with_connector_lazy(connector);
        Ok(Self {
            client: MachineClient::new(channel).max_decoding_message_size(16 << 20),
            key: Arc::new(key),
            worker: endpoint.worker_id.clone(),
        })
    }

    fn request<T>(&self, message: T) -> tonic::Request<T> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as i64);
        let grant = capability::Grant {
            machine: self.worker.clone(),
            action: capability::MACHINE.into(),
            expires: now + CAP_SECONDS,
            ..Default::default()
        };
        let mut request = tonic::Request::new(message);
        let value = format!(
            "{}{}",
            capability::SCHEME,
            capability::mint(&self.key, grant)
        );
        request
            .metadata_mut()
            .insert("authorization", value.parse().expect("a cap is ASCII"));
        request
    }

    /// Sends the run (with `spec`) or attaches to it, and answers its first frame, the run's
    /// state, with the stream of the events after `after`. A refusal is the stream's first item.
    pub async fn run(
        &self,
        id: &str,
        after: u64,
        spec: Option<v1::RunSpec>,
    ) -> Result<(v1::RunState, Streaming<v1::RunEvent>), LinkError> {
        let request = self.request(v1::RunRequest {
            id: id.into(),
            after,
            spec,
        });
        let mut stream = self
            .client
            .clone()
            .run(request)
            .await
            .map_err(classify)?
            .into_inner();
        match stream.message().await {
            Ok(Some(v1::RunEvent {
                event: Some(v1::run_event::Event::State(state)),
                ..
            })) => Ok((state, stream)),
            Ok(_) => Err(LinkError::Unreachable(
                "the machine's stream did not open with the run's state".into(),
            )),
            Err(status) => Err(classify(status)),
        }
    }

    /// Reads `length` bytes (0: to its end) of an output from `offset`, at revision `if_rev` (0:
    /// its newest). Answers the first frame (the output's revision, length and where this read
    /// ends) and the stream of its data frames. The machine's answer is the caller's to judge.
    pub async fn read(
        &self,
        target: v1::OutputTarget,
        offset: u64,
        length: u64,
        if_rev: u64,
    ) -> Result<(v1::ReadFrame, Streaming<v1::ReadFrame>), Status> {
        let request = self.request(v1::ReadRequest {
            target: Some(v1::read_request::Target::Output(target)),
            offset,
            length,
            if_rev,
            tail: 0,
        });
        let mut frames = self.client.clone().read(request).await?.into_inner();
        match frames.message().await? {
            Some(first) => Ok((first, frames)),
            None => Err(Status::unavailable(
                "the machine ended the read before its first frame",
            )),
        }
    }

    /// The machine's Status: its whole picture, then again whenever it changes. With `keepalive`,
    /// a rental's idle clock starts over first.
    pub async fn status(&self, keepalive: bool) -> Result<Streaming<v1::StatusFrame>, LinkError> {
        let request = self.request(v1::StatusRequest { keepalive });
        Ok(self
            .client
            .clone()
            .status(request)
            .await
            .map_err(classify)?
            .into_inner())
    }

    /// The identity frame the machine at `address` gives anyone who reaches it: its worker id,
    /// boot and version.
    pub async fn identity(address: &str, pin: [u8; 32]) -> Result<v1::StatusFrame, LinkError> {
        let endpoint = store::Endpoint {
            worker_id: String::new(),
            address: address.into(),
            pin,
        };
        let link = Self::new(&endpoint, SigningKey::from_bytes(&[0; 32]))?;
        let request = tonic::Request::new(v1::StatusRequest::default());
        let mut frames = link
            .client
            .clone()
            .status(request)
            .await
            .map_err(classify)?
            .into_inner();
        frames
            .message()
            .await
            .map_err(classify)?
            .ok_or_else(|| LinkError::Unreachable("the machine sent no identity".into()))
    }

    /// The last `tail` bytes (0: all) of a log the machine keeps.
    pub async fn read_log(&self, log: &str, tail: u64) -> Result<Vec<u8>, LinkError> {
        let request = self.request(v1::ReadRequest {
            target: Some(v1::read_request::Target::Log(log.into())),
            tail,
            ..Default::default()
        });
        let mut frames = self
            .client
            .clone()
            .read(request)
            .await
            .map_err(classify)?
            .into_inner();
        let mut bytes = vec![];
        while let Some(frame) = frames.message().await.map_err(classify)? {
            bytes.extend(frame.data);
        }
        Ok(bytes)
    }

    /// Asks the machine to cancel the run. A machine that never received it records it canceled,
    /// so a late send of it is answered with that.
    pub async fn cancel(&self, id: &str) -> Result<v1::RunState, LinkError> {
        let request = self.request(v1::ControlRequest {
            id: id.into(),
            action: v1::Action::Cancel as i32,
        });
        Ok(self
            .client
            .clone()
            .control(request)
            .await
            .map_err(classify)?
            .into_inner())
    }

    /// Uploads one object from `file`, continuing from what the machine already holds.
    /// How many bytes of the object the machine holds for this signer.
    pub async fn held(&self, digest: &Digest, length: u64) -> Result<u64, LinkError> {
        let header = v1::WriteFrame {
            digest: digest.to_string(),
            length,
            ..Default::default()
        };
        let request = self.request(tokio_stream::once(header));
        let asked = self.client.clone().write(request).await;
        Ok(asked.map_err(classify)?.into_inner().held)
    }

    /// Uploads one object from `file` in one stream, from what the machine already holds, so a
    /// long round trip costs one turn, not one per piece. `progress` hears the bytes handed over
    /// (at most a few times a second) with the rate, then what the machine holds once it answers;
    /// with the socket's unsent bytes bounded (`UNSENT`) and HTTP/2 buffering at most 1 MiB per
    /// stream, what was handed over runs only that far ahead of what crossed. An object the
    /// machine already holds is not written and `progress` hears nothing.
    pub async fn write(
        &self,
        digest: &Digest,
        length: u64,
        mut file: tokio::fs::File,
        mut progress: impl FnMut(u64, Option<f64>),
    ) -> Result<(), LinkError> {
        let held = self.held(digest, length).await?;
        if held >= length {
            return Ok(());
        }
        progress(held, None);
        let (frames, receiver) = tokio::sync::mpsc::channel::<v1::WriteFrame>(1);
        let started = Instant::now();
        let rate =
            move |offset: u64| (offset - held) as f64 / started.elapsed().as_secs_f64().max(1e-3);
        // Owns the sender: when the object is read, the stream ends and the machine answers.
        let tell = &mut progress;
        let send = async move {
            file.seek(std::io::SeekFrom::Start(held)).await?;
            let (mut offset, mut told) = (held, Instant::now());
            loop {
                let mut data = vec![0; FRAME];
                let read = file.read(&mut data).await?;
                if read == 0 {
                    return Ok::<_, std::io::Error>(());
                }
                data.truncate(read);
                let frame = v1::WriteFrame {
                    digest: digest.to_string(),
                    length,
                    offset,
                    data,
                };
                if frames.send(frame).await.is_err() {
                    return Ok(());
                }
                offset += read as u64;
                if told.elapsed() >= TELL_EVERY {
                    told = Instant::now();
                    tell(offset, Some(rate(offset)));
                }
            }
        };
        let request = self.request(tokio_stream::wrappers::ReceiverStream::new(receiver));
        let mut client = self.client.clone();
        let (sent, answered) = tokio::join!(send, client.write(request));
        if let Err(e) = sent {
            return Err(LinkError::Refused {
                code: "input_unreadable".into(),
                message: e.to_string(),
            });
        }
        match answered.map_err(classify)?.into_inner().held {
            now if now == length => {
                progress(now, Some(rate(now)));
                Ok(())
            }
            now => Err(LinkError::Unreachable(format!(
                "the machine holds {now} of the {length} bytes of {digest}"
            ))),
        }
    }
}

/// A machine's refusal as the driver acts on it.
pub fn classify(status: Status) -> LinkError {
    let message = status.message().to_owned();
    let code = status
        .metadata()
        .get("cozy-error-code")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    match status.code() {
        Code::NotFound => LinkError::NotFound,
        Code::Unavailable if message.starts_with("machine_released") => {
            LinkError::Released(message)
        }
        Code::Unauthenticated => LinkError::Unauthorized(message),
        Code::InvalidArgument
        | Code::AlreadyExists
        | Code::PermissionDenied
        | Code::Unimplemented
        | Code::FailedPrecondition
        | Code::OutOfRange => LinkError::Refused {
            code: code.unwrap_or_else(|| format!("{:?}", status.code())),
            message,
        },
        other => LinkError::Unreachable(format!("{other:?}: {message}")),
    }
}

/// The TLS leaf (DER) the machine at `address` presents, whatever it is. Nothing is sent: the
/// handshake is refused once the leaf is seen.
pub async fn presented_leaf(address: &str) -> Result<Vec<u8>, LinkError> {
    let unreachable = |e: std::io::Error| LinkError::Unreachable(format!("{address}: {e}"));
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
        (None, Ok(_)) => Err(LinkError::Unreachable(format!(
            "{address} presented no TLS leaf"
        ))),
    }
}

/// TLS that admits exactly the leaf whose sha256 is the pin, whatever its name or dates say.
fn tls(pinned: Arc<Pinned>) -> rustls::ClientConfig {
    let mut config = rustls::ClientConfig::builder_with_provider(pinned.provider.clone())
        .with_safe_default_protocol_versions()
        .expect("aws-lc-rs offers the default protocol versions")
        .dangerous()
        .with_custom_certificate_verifier(pinned)
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h2".to_vec()];
    config
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
