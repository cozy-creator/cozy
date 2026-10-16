//! Helpers for tests against a real Tensorhub (the `testing` feature): the CI hub's emailed codes,
//! a signed-up user, and TLS in front of the hub's plain listener for its public origin.
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{Bearer, Client, DeviceKey, Enrollment, Error};

/// The newest code the hub's development mail sender logged for `email` (its JSON log at `log`).
pub fn emailed_code(log: &Path, email: &str) -> String {
    let started = Instant::now();
    loop {
        let text = std::fs::read_to_string(log).unwrap_or_default();
        let code = text
            .lines()
            .rev()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|line| line["email"] == email)
            .find_map(|line| line["code"].as_str().map(str::to_string));
        if let Some(code) = code {
            return code;
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the hub logged no code for {email}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Enrolls a device key for `email`, reading the emailed code from the hub's log.
pub async fn sign_up(
    anonymous: &Client,
    email: &str,
    log: &Path,
) -> Result<(DeviceKey, Bearer), Error> {
    let enrollment = Enrollment::begin(anonymous, email, "test", None).await?;
    enrollment
        .finish(anonymous, &emailed_code(log, email))
        .await
}

/// TLS for a hub's public `origin` (`https://host`), in front of its plain listener at `backend`,
/// as the proxy before a real hub is. `http` trusts it and reaches the origin through it.
pub struct TlsFront {
    pub http: reqwest::Client,
    serving: tokio::task::JoinHandle<()>,
}

impl TlsFront {
    pub async fn start(origin: &str, backend: SocketAddr) -> Self {
        let host = origin
            .trim_start_matches("https://")
            .trim_end_matches('/')
            .to_string();
        let issued =
            rcgen::generate_simple_self_signed(vec![host.clone()]).expect("a test certificate");
        let cert = issued.cert.der().clone();
        let key =
            rustls::pki_types::PrivateKeyDer::Pkcs8(issued.signing_key.serialize_der().into());
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .and_then(|config| {
            config
                .with_no_client_auth()
                .with_single_cert(vec![cert.clone()], key)
        })
        .expect("a TLS server config");
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a loopback port");
        let front = listener.local_addr().expect("its address");
        let serving = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    let (Ok(mut tls), Ok(mut plain)) = (
                        acceptor.accept(stream).await,
                        tokio::net::TcpStream::connect(backend).await,
                    ) else {
                        return;
                    };
                    let _ = tokio::io::copy_bidirectional(&mut tls, &mut plain).await;
                });
            }
        });
        let http = reqwest::Client::builder()
            .tls_certs_only([reqwest::Certificate::from_der(&cert).expect("its certificate")])
            .resolve(&host, front)
            .build()
            .expect("a client");
        Self { http, serving }
    }
}

impl Drop for TlsFront {
    fn drop(&mut self) {
        self.serving.abort();
    }
}
