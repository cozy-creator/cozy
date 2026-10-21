use super::{auth::Authority, backend::MachineBackend};
use crate::machine::receipt::{self, Readiness};
use hyper_util::{
    rt::{TokioExecutor, TokioIo, TokioTimer},
    server::conn::auto,
    service::TowerToHyperService,
};
use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_rustls::{rustls, TlsAcceptor};

pub struct MachineIdentity {
    pub authority: Authority,
    pub cert_pem: String,
    pub key_pem: String,
    pub cert_der: Vec<u8>,
    pub readiness: Arc<Readiness>,
    pub started_at_ms: u64,
    /// A rental's idle ledger; None on development front doors.
    pub lifecycle: Option<Arc<crate::machine::lifecycle::Lifecycle>>,
    /// The Hubs this machine is registered with: (origin, worker id there).
    pub hubs: Vec<(String, String)>,
    /// `runtime-update/1`; None on development front doors.
    pub updates: Option<Arc<crate::machine::update::Updates>>,
    /// The WebRTC listener: `cozy/1` for browsers (ICE-TCP).
    pub webrtc: Option<std::net::TcpListener>,
    /// Where browsers reach it; Status reports it with the port `serve` bound.
    pub player: crate::machine::player::Reach,
    pub(crate) webrtc_port: Option<u16>,
    /// The store's directory; Status reports its filesystem. None on development front doors.
    pub store: Option<std::path::PathBuf>,
    /// The GPUs this boot measured, which its readiness receipt names.
    pub gpus: Vec<receipt::Gpu>,
    /// The most memory each of them offers (`machine.gpu_memory_limit`).
    pub gpu_memory_limit: Option<u64>,
}

impl MachineIdentity {
    pub fn ephemeral(
        worker_id: String,
        keys: Vec<ed25519_dalek::VerifyingKey>,
        receipt_key: Vec<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if keys.is_empty() || receipt_key.len() < 32 {
            return Err("authorized owner keys and a >=32-byte readiness key are required".into());
        }
        let boot_id = uuid::Uuid::new_v4().to_string();
        let rcgen::CertifiedKey { cert, signing_key } =
            rcgen::generate_simple_self_signed(vec!["cozy.worker".into(), "localhost".into()])?;
        let cert_der = cert.der().to_vec();
        let authority = Authority {
            worker_id,
            boot_id,
            keys: keys.into(),
        };
        Ok(Self {
            authority,
            cert_pem: cert.pem(),
            key_pem: signing_key.serialize_pem(),
            cert_der,
            readiness: Readiness::open(None, Some(receipt_key), false)?,
            started_at_ms: now_ms(),
            lifecycle: None,
            hubs: vec![],
            updates: None,
            store: None,
            gpus: vec![],
            gpu_memory_limit: None,
            webrtc: None,
            player: Default::default(),
            webrtc_port: None,
        })
    }
}

pub async fn serve<B: MachineBackend>(
    listener: tokio::net::TcpListener,
    mut identity: MachineIdentity,
    backend: Arc<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let port = listener.local_addr()?.port();
    let readiness = identity.readiness.clone();
    let webrtc = match identity.webrtc.take() {
        Some(listener) => {
            listener.set_nonblocking(true)?;
            Some(tokio::net::TcpListener::from_std(listener)?)
        }
        None => None,
    };
    identity.webrtc_port = match &webrtc {
        Some(listener) => Some(listener.local_addr()?.port()),
        None => None,
    };
    let mut measured = MeasuredIdentity::of(&identity);
    measured.webrtc_port = identity.webrtc_port;
    tokio::spawn(prove_readiness(port, readiness, measured));
    let tls = TlsAcceptor::from(Arc::new(tls(&identity)?));
    let identity = Arc::new(identity);
    if let Some(listener) = webrtc {
        let media = super::cozy1::Media::new(identity.clone(), backend.clone())?;
        tokio::spawn(super::cozy1::serve(media, listener));
    }
    let service = TowerToHyperService::new(super::http::router(identity, backend));
    let mut http = auto::Builder::new(TokioExecutor::new());
    // A ping every 20 s keeps NAT mappings on a client's path alive through a quiet run, and one
    // left unanswered for 20 s ends a dead connection (its runs go on; a client attaches again).
    http.http2()
        .timer(TokioTimer::new())
        .keep_alive_interval(LIVENESS)
        .keep_alive_timeout(LIVENESS);
    let http = Arc::new(http);
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _)) => stream,
            // Out of descriptors, say: accept fails again at once, so pause rather than spin.
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        live(&stream);
        let (tls, http, service) = (tls.clone(), http.clone(), service.clone());
        tokio::spawn(async move {
            if let Ok(stream) = tls.accept(stream).await {
                let _ = http.serve_connection(TokioIo::new(stream), service).await;
            }
        });
    }
}

const LIVENESS: Duration = Duration::from_secs(20);

/// The machine's leaf, offering HTTP/2 for control and HTTP/1.1 for bulk lanes.
fn tls(identity: &MachineIdentity) -> std::io::Result<rustls::ServerConfig> {
    let key = rustls_pemfile::private_key(&mut identity.key_pem.as_bytes())?
        .ok_or_else(|| std::io::Error::other("the machine leaf has no private key"))?;
    let leaf = rustls::pki_types::CertificateDer::from(identity.cert_der.clone());
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .and_then(|config| config.with_no_client_auth().with_single_cert(vec![leaf], key))
        .map_err(std::io::Error::other)?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(config)
}

/// Nagle off, as Go's net package sets every TCP connection: a response's small trailing frames
/// otherwise wait for the ACK of the frames before them, one round trip each on a far link. TCP
/// keepalive judges an HTTP/1.1 peer as PINGs judge an HTTP/2 one, so a lane whose client
/// vanished ends.
fn live(stream: &tokio::net::TcpStream) {
    let _ = stream.set_nodelay(true);
    #[cfg(unix)]
    keepalive(stream);
}

#[cfg(unix)]
fn keepalive(stream: &tokio::net::TcpStream) {
    use nix::sys::socket::{setsockopt, sockopt};
    let seconds = LIVENESS.as_secs() as u32;
    let _ = setsockopt(stream, sockopt::KeepAlive, &true);
    #[cfg(target_os = "linux")]
    let _ = setsockopt(stream, sockopt::TcpKeepIdle, &seconds);
    #[cfg(target_os = "macos")]
    let _ = setsockopt(stream, sockopt::TcpKeepAlive, &seconds);
    let _ = setsockopt(stream, sockopt::TcpKeepInterval, &seconds);
    let _ = setsockopt(stream, sockopt::TcpKeepCount, &1);
}

struct MeasuredIdentity {
    worker_id: String,
    boot_id: String,
    cert_der: Vec<u8>,
    webrtc_port: Option<u16>,
    capabilities: Vec<String>,
    gpus: Vec<receipt::Gpu>,
}
impl MeasuredIdentity {
    fn of(identity: &MachineIdentity) -> Self {
        Self {
            worker_id: identity.authority.worker_id.clone(),
            boot_id: identity.authority.boot_id.clone(),
            cert_der: identity.cert_der.clone(),
            webrtc_port: None,
            capabilities: super::machine_status::capabilities(identity),
            gpus: identity.gpus.clone(),
        }
    }
}

/// Observes this listener as the Hub requires and seals the result once. A retained envelope
/// of this boot is kept as is; a measurement that contradicts it is reported, never signed.
async fn prove_readiness(port: u16, readiness: Arc<Readiness>, id: MeasuredIdentity) {
    let listener_bound = crate::machine::probe::presents_leaf(port, &id.cert_der).await;
    let foreign_credential_refused =
        crate::machine::probe::refuses_foreign_capability(port, &id.cert_der, &id.worker_id).await;
    let gpus = match readiness.measures_gpus() {
        false => vec![],
        true => id.gpus,
    };
    let result = readiness.seal(
        receipt::Measured {
            boot_id: &id.boot_id,
            worker_port: port,
            cert_der: &id.cert_der,
            gpus,
            listener_bound,
            foreign_credential_refused,
            capabilities: id.capabilities,
            webrtc_port: id.webrtc_port,
        }
        .payload(),
    );
    match result {
        Ok(true) => tracing::info!("readiness sealed (listener {listener_bound}, foreign capability refused {foreign_credential_refused})"),
        Ok(false) => tracing::info!("this boot's retained readiness still holds"),
        Err(error) => {
            // A service that cannot prove its boot ends, so its parent restarts or rolls it back
            // and whoever launched it hears the reason, instead of waiting on a machine that
            // will never be ready.
            tracing::error!("readiness not proved: {error}");
            std::process::exit(78);
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
