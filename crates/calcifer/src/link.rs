//! One machine's `cozy.machine.v1` API: a TLS connection that admits only the machine's pinned
//! leaf, every call carrying a short machine-scope Cozy-Cap signed by the key the machine scopes
//! its runs to.
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
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
use tokio::io::AsyncReadExt as _;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{
    Code, Status, Streaming,
    codegen::http::Uri,
    transport::{Channel, Endpoint},
};
use v1::machine_client::MachineClient;

use crate::store;

/// The name every machine's self-signed leaf carries.
const SERVER_NAME: &str = "cozy-worker";
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
        let tls = Arc::new(tls(endpoint.pin));
        let address = endpoint.address.clone();
        let connector = tower::service_fn(move |_: Uri| {
            let (tls, address) = (tls.clone(), address.clone());
            async move {
                let tcp = tokio::net::TcpStream::connect(&address).await?;
                tcp.set_nodelay(true)?;
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
    pub async fn write(
        &self,
        digest: &Digest,
        length: u64,
        mut file: tokio::fs::File,
    ) -> Result<(), LinkError> {
        let header = v1::WriteFrame {
            digest: digest.to_string(),
            length,
            ..Default::default()
        };
        let asked = self
            .client
            .clone()
            .write(self.request(tokio_stream::once(header)))
            .await;
        let held = asked.map_err(classify)?.into_inner().held;
        if held >= length {
            return Ok(());
        }
        let (frames, receiver) = tokio::sync::mpsc::channel(4);
        let name = digest.to_string();
        let reader = tokio::spawn(async move {
            use tokio::io::AsyncSeekExt as _;
            file.seek(std::io::SeekFrom::Start(held)).await?;
            let mut offset = held;
            loop {
                let mut data = vec![0; 1 << 20];
                let read = file.read(&mut data).await?;
                if read == 0 {
                    return Ok::<_, std::io::Error>(());
                }
                data.truncate(read);
                let frame = v1::WriteFrame {
                    digest: name.clone(),
                    length,
                    offset,
                    data,
                };
                if frames.send(frame).await.is_err() {
                    return Ok(());
                }
                offset += read as u64;
            }
        });
        let written = self
            .client
            .clone()
            .write(self.request(ReceiverStream::new(receiver)))
            .await;
        if let Ok(Err(e)) = reader.await {
            return Err(LinkError::Refused {
                code: "input_unreadable".into(),
                message: e.to_string(),
            });
        }
        match written.map_err(classify)?.into_inner().held {
            held if held == length => Ok(()),
            held => Err(LinkError::Unreachable(format!(
                "the machine holds {held} of the {length} bytes of {digest}"
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

/// TLS that admits exactly the leaf whose sha256 is `pin`, whatever its name or dates say.
fn tls(pin: [u8; 32]) -> rustls::ClientConfig {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut config = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("aws-lc-rs offers the default protocol versions")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Pinned { pin, provider }))
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h2".to_vec()];
    config
}

#[derive(Debug)]
struct Pinned {
    pin: [u8; 32],
    provider: Arc<CryptoProvider>,
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
