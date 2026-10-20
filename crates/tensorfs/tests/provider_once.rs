//! Public DPoP on a real loopback HTTP issuer; no GPU or model download.
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use aws_lc_rs::signature::{UnparsedPublicKey, ECDSA_P256_SHA256_FIXED};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde_json::{json, Value};
use tensorfs::format::{Header, ObjectRef, Repo, Tree};
use tensorfs::{Client, Config, DPoP, JwtBearer, Kind, Open, Request, TokenProvider};

fn point() -> Vec<u8> {
    fn bytes(hex: &str) -> Vec<u8> {
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }
    let mut point = vec![4];
    point.extend(bytes(
        "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296",
    ));
    point.extend(bytes(
        "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5",
    ));
    point
}

fn key() -> String {
    // P-256 scalar one, fixture-only. The native constructor verifies the point/scalar pair.
    let point = point();
    let mut scalar = [0u8; 32];
    scalar[31] = 1;
    json!({"kty":"EC", "crv":"P-256", "x":URL_SAFE_NO_PAD.encode(&point[1..33]),
        "y":URL_SAFE_NO_PAD.encode(&point[33..]), "d":URL_SAFE_NO_PAD.encode(scalar)})
    .to_string()
}

fn claims(jwt: &str) -> Value {
    let parts: Vec<_> = jwt.split('.').collect();
    assert_eq!(parts.len(), 3);
    UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, point())
        .verify(
            format!("{}.{}", parts[0], parts[1]).as_bytes(),
            &URL_SAFE_NO_PAD.decode(parts[2]).unwrap(),
        )
        .expect("the actual HTTP request carries a valid ES256 signature");
    serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[1]).unwrap()).unwrap()
}

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "provider-once-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("create own scratch: {error}"),
            }
        }
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Success,
    Nonce,
    RepeatedNonce,
    ServerNonce,
    DropReply,
    HoldReply,
}

#[derive(Clone)]
struct Seen {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: String,
}
fn receive(stream: &mut TcpStream) -> std::io::Result<Seen> {
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    let mut bytes = Vec::new();
    let end = loop {
        let mut buf = [0u8; 4096];
        let got = stream.read(&mut buf)?;
        if got == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        bytes.extend_from_slice(&buf[..got]);
        if let Some(end) = bytes.windows(4).position(|b| b == b"\r\n\r\n") {
            break end + 4;
        }
        if bytes.len() > 64 << 10 {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
    };
    let head = std::str::from_utf8(&bytes[..end]).map_err(std::io::Error::other)?;
    let mut lines = head.lines();
    let mut first = lines.next().unwrap().split_whitespace();
    let (method, path) = (
        first.next().unwrap().to_owned(),
        first.next().unwrap().to_owned(),
    );
    let headers: BTreeMap<_, _> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_owned()))
        .collect();
    let length = headers
        .get("content-length")
        .map_or(Ok(0), |n| n.parse::<usize>())
        .map_err(std::io::Error::other)?;
    if length > 1 << 20 {
        return Err(std::io::ErrorKind::InvalidData.into());
    }
    while bytes.len() < end + length {
        let mut buf = [0u8; 4096];
        let got = stream.read(&mut buf)?;
        if got == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        bytes.extend_from_slice(&buf[..got]);
    }
    Ok(Seen {
        method,
        path,
        headers,
        body: String::from_utf8(bytes[end..end + length].to_vec())
            .map_err(std::io::Error::other)?,
    })
}

fn answer(
    stream: &mut TcpStream,
    method: &str,
    status: u16,
    body: &[u8],
    nonce: Option<&str>,
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} fixture\r\ncontent-length: {}\r\nconnection: close\r\n",
        body.len()
    )?;
    if let Some(nonce) = nonce {
        write!(stream, "dpop-nonce: {nonce}\r\n")?;
    }
    write!(stream, "\r\n")?;
    if method != "HEAD" {
        stream.write_all(body)?;
    }
    Ok(())
}

fn object_answer(stream: &mut TcpStream, request: &Seen, body: &[u8]) -> std::io::Result<()> {
    let Some(range) = request
        .headers
        .get("range")
        .and_then(|value| value.strip_prefix("bytes="))
    else {
        return answer(stream, &request.method, 200, body, None);
    };
    let (lo, hi) = range.split_once('-').expect("a byte range");
    let lo: usize = lo.parse().unwrap();
    let hi = if hi.is_empty() {
        body.len()
    } else {
        hi.parse::<usize>().unwrap() + 1
    }
    .min(body.len());
    assert!(lo < hi);
    write!(stream, "HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\ncontent-range: bytes {}-{}/{}\r\nconnection: close\r\n\r\n", hi-lo, lo, hi-1, body.len())?;
    if request.method != "HEAD" {
        stream.write_all(&body[lo..hi])?;
    }
    Ok(())
}

struct Origin {
    url: String,
    posts: Arc<AtomicUsize>,
    redeemed: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Seen>>>,
    files: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Origin {
    fn new(mode: Mode) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (posts, redeemed, requests, files, stop) = (
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(BTreeMap::<String, Vec<u8>>::new())),
            Arc::new(AtomicBool::new(false)),
        );
        let (p, used, seen, objects, done) = (
            posts.clone(),
            redeemed.clone(),
            requests.clone(),
            files.clone(),
            stop.clone(),
        );
        let worker = thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(error) => panic!("accept fixture: {error}"),
                };
                let request = receive(&mut stream).expect("read actual HTTP request");
                seen.lock().unwrap().push(request.clone());
                if request.method == "POST" && request.path == "/token" {
                    let count = p.fetch_add(1, Ordering::AcqRel) + 1;
                    match mode {
                        Mode::Nonce if count == 1 => {
                            let _ = answer(
                                &mut stream,
                                &request.method,
                                400,
                                br#"{"error":"use_dpop_nonce"}"#,
                                Some("nonce-one"),
                            );
                        }
                        Mode::RepeatedNonce => {
                            let _ = answer(
                                &mut stream,
                                &request.method,
                                400,
                                br#"{"error":"use_dpop_nonce"}"#,
                                Some(if count == 1 { "nonce-one" } else { "nonce-two" }),
                            );
                        }
                        Mode::ServerNonce => {
                            used.store(true, Ordering::Release);
                            let _ = answer(
                                &mut stream,
                                &request.method,
                                503,
                                br#"{"error":"use_dpop_nonce"}"#,
                                Some("nonce-one"),
                            );
                        }
                        Mode::DropReply | Mode::HoldReply => {
                            used.store(true, Ordering::Release);
                            if matches!(mode, Mode::HoldReply) {
                                while !done.load(Ordering::Acquire) {
                                    thread::sleep(Duration::from_millis(1));
                                }
                            }
                        }
                        _ if used.swap(true, Ordering::AcqRel) => {
                            let _ = answer(
                                &mut stream,
                                &request.method,
                                400,
                                br#"{"error":"invalid_grant","reason":"already redeemed"}"#,
                                None,
                            );
                        }
                        _ => {
                            let _ = answer(
                                &mut stream,
                                &request.method,
                                200,
                                br#"{"access_token":"issued-once","token_type":"DPoP"}"#,
                                None,
                            );
                        }
                    }
                } else {
                    let body = objects.lock().unwrap().get(&request.path).cloned();
                    let _ = match body {
                        Some(body) => object_answer(&mut stream, &request, &body),
                        None => answer(&mut stream, &request.method, 404, b"missing", None),
                    };
                }
            }
        });
        Self {
            url,
            posts,
            redeemed,
            requests,
            files,
            stop,
            worker: Some(worker),
        }
    }
    fn provider(&self, root: &Path) -> DPoP {
        DPoP::grant(
            &self.config(root),
            &key(),
            JwtBearer {
                token_endpoint: format!("{}/token", self.url),
                client_id: "public-provider-test".into(),
                resource: "https://resource.example".into(),
                claims: json!({"purpose":"fixture-one-time"})
                    .as_object()
                    .unwrap()
                    .clone(),
            },
        )
        .unwrap()
    }
    fn config(&self, root: &Path) -> Config {
        let mut config = Config::from_env();
        config.endpoint = self.url.clone();
        config.cache_dir = root.to_path_buf();
        config.token = None;
        config.ca_der = None;
        config.mirror = None;
        config
    }
    async fn consumed(&self) {
        tokio::time::timeout(Duration::from_secs(3), async {
            while !self.redeemed.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("issuer did not consume the actual HTTP grant");
    }
    fn metadata(&self) {
        let header = Header::new(vec![], None, vec![], Default::default())
            .unwrap()
            .to_cbor()
            .unwrap();
        let h = ObjectRef::of(&header);
        let mut tree = Tree::default();
        tree.set_variant("", h.clone());
        let tree = tree.to_json().unwrap();
        let t = ObjectRef::of(&tree);
        let mut repo = Repo::new("acme/model");
        repo.set_release("1.0", t.clone());
        self.files.lock().unwrap().extend([
            (format!("/{}", h.key()), header),
            (format!("/{}", t.key()), tree),
            ("/refs/acme/model.json".into(), repo.to_json().unwrap()),
        ]);
    }
}
impl Drop for Origin {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

async fn headers(provider: &DPoP, url: &str) -> tensorfs::Result<Vec<(String, String)>> {
    provider
        .headers(&Request {
            method: "GET",
            url,
            headers: &[],
            payload: None,
        })
        .await
}

#[tokio::test]
async fn success_and_clones_share_one_signed_exchange() {
    let scratch = Scratch::new();
    let origin = Origin::new(Mode::Success);
    let provider = origin.provider(&scratch.0);
    assert!(provider.issued_token().await.is_none());
    assert_eq!(origin.posts.load(Ordering::Acquire), 0);
    let url = format!("{}/refs/acme/model.json", origin.url);
    let clone = provider.clone();
    let (a, b) = tokio::join!(headers(&provider, &url), headers(&clone, &url));
    let (a, b) = (a.unwrap(), b.unwrap());
    assert_eq!(
        provider.issued_token().await.as_deref(),
        Some("issued-once")
    );
    assert_eq!(origin.posts.load(Ordering::Acquire), 1);
    for presented in [a, b] {
        assert!(presented
            .iter()
            .any(|(name, value)| name == "authorization" && value == "DPoP issued-once"));
        let proof = &presented.iter().find(|(name, _)| name == "dpop").unwrap().1;
        let signed = claims(proof);
        assert_eq!(
            signed["htu"],
            "https://resource.example/refs/acme/model.json"
        );
        assert_eq!(signed["htm"], "GET");
    }
    // The server observed a genuinely signed JWT-bearer request, not only a client call count.
    let requests = origin.requests.lock().unwrap();
    let sent = requests.iter().find(|r| r.path == "/token").unwrap();
    assert!(sent
        .body
        .contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"));
    let assertion = sent
        .body
        .split('&')
        .find_map(|part| part.strip_prefix("assertion="))
        .unwrap();
    assert_eq!(claims(assertion)["purpose"], "fixture-one-time");
    assert_eq!(
        claims(&sent.headers["dpop"])["htu"],
        format!("{}/token", origin.url)
    );
}

#[tokio::test]
async fn only_one_explicit_nonce_challenge_can_retry_the_same_attempt() {
    for (mode, count, succeeds) in [
        (Mode::Nonce, 2, true),
        (Mode::RepeatedNonce, 2, false),
        (Mode::ServerNonce, 1, false),
    ] {
        let scratch = Scratch::new();
        let origin = Origin::new(mode);
        let provider = origin.provider(&scratch.0);
        let url = format!("{}/metadata", origin.url);
        let result = headers(&provider, &url).await;
        assert_eq!(result.is_ok(), succeeds);
        if !succeeds {
            assert_eq!(result.unwrap_err().kind, Kind::Denied);
            assert_eq!(
                headers(&provider, &url).await.unwrap_err().kind,
                Kind::Denied
            );
        }
        assert_eq!(origin.posts.load(Ordering::Acquire), count);
        let requests = origin.requests.lock().unwrap();
        if count == 2 {
            assert_eq!(
                requests[0].body, requests[1].body,
                "nonce retry must stay inside one signed assertion attempt"
            );
            assert_eq!(claims(&requests[1].headers["dpop"])["nonce"], "nonce-one");
        }
    }
}

#[tokio::test]
async fn dropped_reply_is_terminal_and_cannot_hide_behind_a_cached_ref() {
    let scratch = Scratch::new();
    let origin = Origin::new(Mode::DropReply);
    origin.metadata();
    let client = Client::new(origin.config(&scratch.0)).unwrap();
    let cached = client.open("acme/model", &Open::default()).await.unwrap();
    drop(cached);
    let provider = origin.provider(&scratch.0);
    let mut config = origin.config(&scratch.0);
    config.token = Some(Arc::new(provider.clone()));
    let guarded = Client::new(config).unwrap();
    let first = guarded
        .open("acme/model", &Open::default())
        .await
        .unwrap_err();
    assert_eq!(first.kind, Kind::Denied);
    assert!(origin.redeemed.load(Ordering::Acquire));
    let again = guarded
        .open("acme/model", &Open::default())
        .await
        .unwrap_err();
    assert_eq!(again.kind, Kind::Denied);
    assert_eq!(again.detail, first.detail);
    assert_eq!(origin.posts.load(Ordering::Acquire), 1);
    assert!(provider.issued_token().await.is_none());
    assert!(
        client
            .open(
                "acme/model",
                &Open {
                    local_files_only: true,
                    ..Open::default()
                }
            )
            .await
            .is_ok(),
        "the cached positive control must remain usable"
    );
}

#[tokio::test]
async fn cancellation_after_redemption_cannot_redeem_again() {
    let scratch = Scratch::new();
    let origin = Origin::new(Mode::HoldReply);
    let provider = origin.provider(&scratch.0);
    let clone = provider.clone();
    let url = format!("{}/metadata", origin.url);
    let started_url = url.clone();
    let pending = tokio::spawn(async move { headers(&clone, &started_url).await });
    origin.consumed().await;
    pending.abort();
    assert!(pending.await.unwrap_err().is_cancelled());
    let refused = tokio::time::timeout(Duration::from_secs(1), headers(&provider, &url))
        .await
        .expect("canceled exchange retried or kept its mutex locked")
        .unwrap_err();
    assert_eq!(refused.kind, Kind::Denied);
    assert_eq!(origin.posts.load(Ordering::Acquire), 1);
    assert!(provider.issued_token().await.is_none());
}

#[tokio::test]
async fn issued_token_survives_a_later_metadata_failure_and_reconstructs_without_trade() {
    let scratch = Scratch::new();
    let origin = Origin::new(Mode::Success);
    let provider = origin.provider(&scratch.0);
    let mut config = origin.config(&scratch.0);
    config.token = Some(Arc::new(provider.clone()));
    let client = Client::new(config).unwrap();
    let error = client
        .open("acme/missing", &Open::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind, Kind::NotFound);
    let token = provider
        .issued_token()
        .await
        .expect("metadata failure discarded an issued token");
    assert_eq!(origin.posts.load(Ordering::Acquire), 1);
    let reused = DPoP::token(
        &key(),
        &token,
        Some("https://resource.example"),
        Some(&origin.url),
    )
    .unwrap();
    assert!(headers(&reused, &format!("{}/metadata", origin.url))
        .await
        .is_ok());
    assert_eq!(origin.posts.load(Ordering::Acquire), 1);
}
