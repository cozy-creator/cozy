use super::*;
use crate::transport::{Anonymous, CredentialProvider, SourceDownload, SourcePolicy};
use std::net::{Shutdown, TcpListener};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

trait Io: Read + Write {}
impl<T: Read + Write> Io for T {}

#[derive(Clone)]
struct Ask {
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}
impl Ask {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}
struct Reply {
    bytes: Vec<u8>,
    close: bool,
}
impl Reply {
    fn body(body: &[u8]) -> Self {
        let mut bytes =
            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
        bytes.extend_from_slice(body);
        Self {
            bytes,
            close: false,
        }
    }
}

struct Origin {
    url: String,
    certificate: Option<CertificateDer<'static>>,
    connections: Arc<AtomicUsize>,
    asks: Arc<Mutex<Vec<Ask>>>,
    stop: Arc<AtomicBool>,
    sockets: Arc<Mutex<Vec<TcpStream>>>,
    task: Option<thread::JoinHandle<()>>,
}
impl Origin {
    fn new(
        tls: bool,
        setup: Duration,
        handle: impl Fn(&Ask, usize) -> Reply + Send + Sync + 'static,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut certificate = None;
        let config = tls.then(|| {
            let key = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
            let cert = CertificateDer::from(key.cert.der().to_vec());
            let private =
                rustls::pki_types::PrivateKeyDer::try_from(key.key_pair.serialize_der()).unwrap();
            let config = rustls::ServerConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert.clone()], private)
            .unwrap();
            certificate = Some(cert);
            Arc::new(config)
        });
        let connections = Arc::new(AtomicUsize::new(0));
        let asks = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let sockets = Arc::new(Mutex::new(Vec::<TcpStream>::new()));
        let (count, records, stopping, opened) = (
            connections.clone(),
            asks.clone(),
            stop.clone(),
            sockets.clone(),
        );
        let handle = Arc::new(handle);
        let task = thread::spawn(move || {
            let mut children = Vec::new();
            while !stopping.load(Ordering::Relaxed) {
                let (socket, _) = match listener.accept() {
                    Ok(value) => value,
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(error) => panic!("accept: {error}"),
                };
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                socket
                    .set_write_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                socket.set_nodelay(true).unwrap();
                opened.lock().unwrap().push(socket.try_clone().unwrap());
                count.fetch_add(1, Ordering::Relaxed);
                let (config, records, handle) = (config.clone(), records.clone(), handle.clone());
                children.push(thread::spawn(move || {
                    thread::sleep(setup);
                    let mut stream: Box<dyn Io> = match config {
                        Some(config) => Box::new(rustls::StreamOwned::new(
                            rustls::ServerConnection::new(config).unwrap(),
                            socket,
                        )),
                        None => Box::new(socket),
                    };
                    loop {
                        let mut header = Vec::new();
                        while !header.ends_with(b"\r\n\r\n") {
                            let mut byte = [0];
                            if stream.read_exact(&mut byte).is_err() {
                                return;
                            }
                            header.push(byte[0]);
                            assert!(header.len() < 64 << 10);
                        }
                        let text = String::from_utf8(header).unwrap();
                        let ask = Ask {
                            path: text
                                .lines()
                                .next()
                                .unwrap()
                                .split_whitespace()
                                .nth(1)
                                .unwrap()
                                .into(),
                            headers: text
                                .lines()
                                .skip(1)
                                .filter_map(|line| line.split_once(':'))
                                .map(|(name, value)| {
                                    (name.to_ascii_lowercase(), value.trim().into())
                                })
                                .collect(),
                            body: Vec::new(),
                        };
                        let mut ask = ask;
                        if let Some(length) = ask.header("content-length") {
                            let mut body = vec![0; length.parse().unwrap()];
                            if stream.read_exact(&mut body).is_err() {
                                return;
                            }
                            ask.body = body;
                        }
                        let number = {
                            let mut rows = records.lock().unwrap();
                            rows.push(ask.clone());
                            rows.len()
                        };
                        let reply = handle(&ask, number);
                        if stream
                            .write_all(&reply.bytes)
                            .and_then(|()| stream.flush())
                            .is_err()
                            || reply.close
                            || ask.header("connection") == Some("close")
                        {
                            return;
                        }
                    }
                }));
            }
            for child in children {
                child.join().unwrap();
            }
        });
        Self {
            url: format!("{}://localhost:{port}", if tls { "https" } else { "http" }),
            certificate,
            connections,
            asks,
            stop,
            sockets,
            task: Some(task),
        }
    }

    fn client(&self) -> Client {
        Client::with_extra_roots(&self.certificate.iter().cloned().collect::<Vec<_>>())
    }
    fn checked(&self, path: &str) -> CheckedUrl {
        Self::policy()
            .check(&format!("{}{path}", self.url))
            .unwrap()
    }
    fn policy() -> SourcePolicy {
        SourcePolicy {
            allowed_hosts: vec!["localhost".into()],
            allow_local: true,
            max_redirects: 2,
            ..Default::default()
        }
    }
    fn count(&self) -> usize {
        self.connections.load(Ordering::Relaxed)
    }
}
impl Drop for Origin {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for socket in self.sockets.lock().unwrap().iter() {
            let _ = socket.shutdown(Shutdown::Both);
        }
        if let Some(task) = self.task.take() {
            task.join().unwrap();
        }
    }
}

fn ranged() -> Vec<(String, String)> {
    vec![("range".into(), "bytes=0-31".into())]
}
fn ask(client: &Client, checked: &CheckedUrl, headers: &[(String, String)]) -> Result<Vec<u8>> {
    let ledger = Ledger::with_resolution(Duration::from_millis(10));
    let response = request(
        client,
        checked,
        "GET",
        headers,
        None,
        Deadline::none(),
        &ledger,
        Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    )?;
    response.read_capped(2 << 20)
}

#[test]
fn complete_ranged_http_and_tls_bodies_reuse_one_connection() {
    for tls in [false, true] {
        let origin = Origin::new(tls, Duration::ZERO, |_, _| Reply::body(b"range bytes"));
        let client = origin.client();
        for index in 0..4 {
            let headers = vec![("Range".into(), format!("bytes={index}-{}", index + 31))];
            assert_eq!(
                ask(&client, &origin.checked("/object"), &headers).unwrap(),
                b"range bytes"
            );
        }
        assert_eq!(origin.count(), 1, "tls={tls}");
        assert_eq!(origin.asks.lock().unwrap().len(), 4);
    }
}

#[test]
fn cached_peer_must_remain_checked_and_credentials_stay_scoped_but_signed_targets_share() {
    let origin = Origin::new(false, Duration::ZERO, |_, _| Reply::body(b"checked"));
    let client = origin.client();
    let checked = origin.checked("/object?signature=one");
    let mut headers = ranged();
    headers.push(("authorization".into(), "Bearer first".into()));
    ask(&client, &checked, &headers).unwrap();
    let mut changed = checked.clone();
    changed.addrs.clear();
    assert_eq!(
        ask(&client, &changed, &headers).unwrap_err().code,
        Code::TRANSFER_FAILED
    );
    assert_eq!(origin.count(), 1);
    ask(&client, &checked, &headers).unwrap();
    headers[1].1 = "Bearer second".into();
    ask(&client, &checked, &headers).unwrap();
    // A re-signed delivery URL on the same origin and credentials rides the same socket.
    ask(&client, &origin.checked("/object?signature=two"), &headers).unwrap();
    assert_eq!(origin.count(), 3);
    let calls = origin.asks.lock().unwrap();
    assert_eq!(
        calls
            .iter()
            .map(|row| row.header("authorization").unwrap())
            .collect::<Vec<_>>(),
        [
            "Bearer first",
            "Bearer first",
            "Bearer second",
            "Bearer second"
        ]
    );
}

#[test]
fn duplicate_credential_header_order_is_not_collapsed_into_one_scope() {
    let origin = Origin::new(false, Duration::ZERO, |request, _| {
        Reply::body(request.header("authorization").unwrap().as_bytes())
    });
    let client = origin.client();
    let mut headers = ranged();
    headers.extend([
        ("authorization".into(), "Bearer first".into()),
        ("authorization".into(), "Bearer second".into()),
    ]);
    assert_eq!(
        ask(&client, &origin.checked("/object"), &headers).unwrap(),
        b"Bearer first"
    );
    headers.swap(1, 2);
    assert_eq!(
        ask(&client, &origin.checked("/object"), &headers).unwrap(),
        b"Bearer second"
    );
    assert_eq!(origin.count(), 2);
}

#[test]
fn only_complete_unambiguous_http11_length_bodies_enter_the_pool() {
    for raw in [
        "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc",
        "HTTP/1.0 200 OK\r\nContent-Length: 3\r\n\r\nabc",
        "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nContent-Length: 3\r\n\r\nabc",
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n",
        "HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabcunsolicited",
    ] {
        let origin = Origin::new(false, Duration::ZERO, move |_, _| Reply {
            bytes: raw.as_bytes().to_vec(),
            close: false,
        });
        let client = origin.client();
        for _ in 0..2 {
            assert_eq!(
                ask(&client, &origin.checked("/object"), &ranged()).unwrap(),
                b"abc"
            );
        }
        assert_eq!(origin.count(), 2, "{raw}");
    }
    let origin = Origin::new(false, Duration::ZERO, |_, _| {
        Reply::body(&vec![42; 1 << 20])
    });
    let client = origin.client();
    let ledger = Ledger::new();
    let response = request(
        &client,
        &origin.checked("/object"),
        "GET",
        &ranged(),
        None,
        Deadline::none(),
        &ledger,
        Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    )
    .unwrap();
    let mut body = response.into_body();
    body.read_exact(&mut [0; 32]).unwrap();
    drop(body);
    assert_eq!(
        ask(&client, &origin.checked("/object"), &ranged())
            .unwrap()
            .len(),
        1 << 20
    );
    assert_eq!(origin.count(), 2);
}

#[test]
fn idle_descriptor_retention_is_bounded_and_plain_gets_pool_too() {
    let origin = Origin::new(false, Duration::ZERO, |_, _| Reply::body(b"bytes"));
    let client = origin.client();
    let scopes = MAX_IDLE + 8;
    for index in 0..scopes {
        let mut headers = ranged();
        headers.push(("authorization".into(), format!("Bearer {index}")));
        ask(&client, &origin.checked("/object"), &headers).unwrap();
    }
    assert_eq!(client.idle.lock().unwrap().len(), MAX_IDLE);
    assert_eq!(origin.count(), scopes);
    // A whole-object GET is a pull's common case: the second rides the first's socket.
    for _ in 0..2 {
        ask(&client, &origin.checked("/object"), &[]).unwrap();
    }
    assert_eq!(origin.count(), scopes + 1);
    assert_eq!(client.idle.lock().unwrap().len(), MAX_IDLE);
}

#[test]
fn cached_connection_cannot_bypass_cancellation_or_tls_authority() {
    let origin = Origin::new(true, Duration::ZERO, |_, _| Reply::body(b"verified"));
    let client = origin.client();
    let checked = origin.checked("/object");
    ask(&client, &checked, &ranged()).unwrap();
    let stopped = Ledger::new();
    stopped.cancel();
    assert!(request(
        &client,
        &checked,
        "GET",
        &ranged(),
        None,
        Deadline::none(),
        &stopped,
        Judge::Stream,
        None,
        Code::TRANSFER_FAILED
    )
    .is_err());
    assert_eq!(origin.asks.lock().unwrap().len(), 1);
    // A different Client owns its own roots and cannot borrow this trusted socket.
    assert!(ask(&Client::new(), &checked, &ranged()).is_err());
    assert_eq!(ask(&client, &checked, &ranged()).unwrap(), b"verified");
    assert_eq!(origin.asks.lock().unwrap().len(), 2);
    assert_eq!(origin.count(), 2); // original trusted connection + refused TLS handshake
}

#[test]
fn truncated_responses_and_closed_idle_peers_do_not_poison_the_next_range() {
    let origin = Origin::new(false, Duration::ZERO, |_, number| {
        if number == 1 {
            Reply {
                bytes: b"HTTP/1.1 206 Partial Content\r\nContent-Length: 6\r\n\r\nabc".to_vec(),
                close: true,
            }
        } else {
            Reply::body(b"abcdef")
        }
    });
    let client = origin.client();
    let checked = origin.checked("/object");
    assert!(ask(&client, &checked, &ranged()).is_err());
    assert_eq!(ask(&client, &checked, &ranged()).unwrap(), b"abcdef");
    // Simulate a peer retiring its idle keep-alive connection. The retained entry
    // must be discarded without manufacturing bytes or escaping the checked peer.
    for socket in origin.sockets.lock().unwrap().iter() {
        socket.shutdown(Shutdown::Both).unwrap();
    }
    assert_eq!(ask(&client, &checked, &ranged()).unwrap(), b"abcdef");
    assert_eq!(origin.count(), 3);
}

#[test]
fn cancellation_of_a_reused_partial_body_closes_it_and_uses_the_new_ledger() {
    let origin = Origin::new(false, Duration::ZERO, |_, number| {
        if number == 2 {
            Reply {
                bytes: b"HTTP/1.1 206 Partial Content\r\nContent-Length: 128\r\n\r\nabc".to_vec(),
                close: false,
            }
        } else {
            Reply::body(b"whole")
        }
    });
    let client = origin.client();
    let checked = origin.checked("/object");
    ask(&client, &checked, &ranged()).unwrap();
    let ledger = Ledger::with_resolution(Duration::from_millis(10));
    ledger.taught(Duration::from_secs(10)); // silence is not this test's cancellation mechanism
    let response = request(
        &client,
        &checked,
        "GET",
        &ranged(),
        None,
        Deadline::after_seconds(Some(5.0)),
        &ledger,
        Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    )
    .unwrap();
    let mut body = response.into_body();
    body.read_exact(&mut [0; 3]).unwrap();
    assert_eq!(origin.count(), 1);
    thread::scope(|scope| {
        let (entered, ready) = std::sync::mpsc::channel();
        let cancel = &ledger;
        scope.spawn(move || {
            ready.recv().unwrap();
            cancel.cancel();
        });
        entered.send(()).unwrap();
        let error = body.read_refusing(&mut [0; 16]).unwrap_err();
        assert_eq!(error.code, Code::TRANSFER_FAILED);
        assert!(error.detail.contains("stopped"));
    });
    drop(body);
    assert_eq!(ask(&client, &checked, &ranged()).unwrap(), b"whole");
    assert_eq!(origin.count(), 2);
}

#[test]
fn tls_range_fixture_measures_setup_reuse_with_identical_bytes_and_requests() {
    let payload = Arc::new(vec![0x73; 1 << 20]);
    let digest = crate::sha256::digest(&payload);
    let mut timings = Vec::new();
    for close in [true, false] {
        let served = payload.clone();
        let origin = Origin::new(true, Duration::from_millis(30), move |_, _| {
            Reply::body(&served)
        });
        let client = origin.client();
        let mut headers = ranged();
        if close {
            headers.push(("connection".into(), "close".into()));
        }
        let start = Instant::now();
        for index in 0..8 {
            headers[0].1 = format!("bytes={}-{}", index << 20, ((index + 1) << 20) - 1);
            assert_eq!(
                crate::sha256::digest(&ask(&client, &origin.checked("/object"), &headers).unwrap()),
                digest
            );
        }
        timings.push((close, start.elapsed().as_secs_f64(), origin.count()));
        assert_eq!(origin.count(), if close { 8 } else { 1 });
        assert_eq!(origin.asks.lock().unwrap().len(), 8);
    }
    println!("TLS 8MiB, eight requests, 30ms connection setup: {timings:?}");
}

struct CloseRequests;
impl CredentialProvider for CloseRequests {
    fn headers(&self, _: &str) -> Vec<(String, String)> {
        vec![("connection".into(), "close".into())]
    }
}

#[test]
fn source_chunks_ask_the_redirect_target_on_one_connection() {
    use crate::{
        ids::ObjectRef,
        providers::{self, Provenance, Resolution, ResolvedMember},
        store::{Fault, Store},
    };
    let bytes = Arc::new(vec![0x5a; 8 << 20]);
    let object = ObjectRef::of(&bytes);
    let mut timings = Vec::new();
    for close in [true, false] {
        let payload = bytes.clone();
        let origin = Origin::new(false, Duration::from_millis(30), move |request, _| {
            if request.path == "/source" {
                return Reply {
                    bytes: b"HTTP/1.1 302 Found\r\nLocation: /blob\r\nContent-Length: 0\r\n\r\n"
                        .to_vec(),
                    close: false,
                };
            }
            let (first, last) = request
                .header("range")
                .unwrap()
                .strip_prefix("bytes=")
                .unwrap()
                .split_once('-')
                .unwrap();
            let (first, last): (usize, usize) = (first.parse().unwrap(), last.parse().unwrap());
            let mut response = format!("HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {first}-{last}/{}\r\n\r\n", last-first+1, payload.len()).into_bytes();
            response.extend_from_slice(&payload[first..=last]);
            Reply {
                bytes: response,
                close: false,
            }
        });
        // Isolate transport setup from this host's unrelated disk journal load.
        // The same durable prefix/CAS APIs still run; this is not a disk benchmark.
        let temporary = if std::path::Path::new("/dev/shm").is_dir() {
            std::path::PathBuf::from("/dev/shm")
        } else {
            std::env::temp_dir()
        };
        let root = temporary.join(format!(
            "tfs-reuse-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let resolution = Resolution {
            canonical: "hf://fixture/model@revision".into(),
            selection_sha256: "accepted".into(),
            members: vec![ResolvedMember {
                member: "model.safetensors".into(),
                object: object.clone(),
                url: format!("{}/source", origin.url),
                provenance: Provenance::Declared,
                carrier: true,
                requires: vec![],
                companion: false,
            }],
        };
        let owner = ObjectRef::of(format!("reuse-{close}").as_bytes()).id();
        let credentials: &dyn CredentialProvider = if close { &CloseRequests } else { &Anonymous };
        let start = Instant::now();
        let (_, rows) = providers::materialize_selected_with_options(
            &store,
            &owner,
            &resolution,
            &Origin::policy(),
            credentials,
            Deadline::none(),
            &|_, _| {},
            &SourceDownload {
                checkpoint_bytes: 1 << 20,
                streams: 1,
                fault: Fault::default(),
                ..Default::default()
            },
        )
        .unwrap();
        let elapsed = start.elapsed();
        assert_eq!(rows[0].transferred, bytes.len() as u64);
        assert!(store.record_valid(&object.sha256).is_ok());
        // One ask walks the redirect and the eight 1 MiB chunks ask its target directly,
        // all on the drained redirect's socket unless each ask closes its own.
        assert_eq!(origin.asks.lock().unwrap().len(), 9);
        assert_eq!(origin.count(), if close { 9 } else { 1 });
        timings.push((close, elapsed.as_secs_f64(), origin.count()));
        drop(store);
        std::fs::remove_dir_all(root).unwrap();
    }
    println!("source 8 MiB, one redirect, eight 1 MiB chunks, 30 ms connection setup: {timings:?}");
}

#[test]
fn concurrent_pushes_to_one_origin_keep_their_sockets_across_signed_objects() {
    use crate::transport::push::{push_object, UploadGrant};
    let origin = Origin::new(false, Duration::ZERO, |_, _| Reply::body(b""));
    let root = std::env::temp_dir().join(format!(
        "tfs-push-reuse-{}-{}",
        std::process::id(),
        origin.url.rsplit(':').next().unwrap()
    ));
    let store = crate::store::Store::init(&root).unwrap();
    let objects: Vec<_> = (0..12u8)
        .map(|index| {
            let body = vec![index; 256 << 10];
            let object = crate::ids::ObjectRef::of(&body);
            store
                .put_stream(&mut body.as_slice(), Some(&object), &Default::default())
                .unwrap();
            object
        })
        .collect();
    let streams = 3;
    thread::scope(|scope| {
        for lane in objects.chunks(objects.len() / streams) {
            let (store, origin) = (&store, &origin);
            scope.spawn(move || {
                for object in lane {
                    // Each grant is its own signed target with its own signed checksum.
                    let grant = UploadGrant {
                        url: format!(
                            "{}/o/{}?X-Amz-Signature={}",
                            origin.url, object.sha256, object.sha256
                        ),
                        headers: vec![("x-amz-checksum-sha256".into(), object.sha256.clone())],
                    };
                    let pushed = push_object(
                        store,
                        &object.sha256,
                        false,
                        &grant,
                        &Origin::policy(),
                        &Anonymous,
                        Deadline::none(),
                        // Test socket reuse at the production sampling cadence. A
                        // 10ms cadence gives a 60ms silence floor and can retry an
                        // otherwise healthy PUT when the fixture thread is delayed.
                        &Ledger::new(),
                        "",
                    )
                    .unwrap();
                    assert_eq!(
                        (pushed.http_status, pushed.bytes_sent),
                        (200, object.length)
                    );
                }
            });
        }
    });
    let asks = origin.asks.lock().unwrap();
    assert_eq!(asks.len(), objects.len());
    for ask in asks.iter() {
        let object = crate::ids::ObjectRef::of(&ask.body);
        assert_eq!(
            ask.path,
            format!("/o/{}?X-Amz-Signature={}", object.sha256, object.sha256)
        );
        assert_eq!(
            ask.header("x-amz-checksum-sha256"),
            Some(object.sha256.as_str())
        );
        assert_eq!(ask.header("connection"), Some("keep-alive"));
    }
    // No stream ever waits on another's socket, and none opens a second one.
    assert!(origin.count() <= streams, "{} connections", origin.count());
    drop(asks);
    let _ = std::fs::remove_dir_all(root);
}
