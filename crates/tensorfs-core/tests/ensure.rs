//! `tfs ensure` against a REAL Store and a REAL loopback hub (plain and TLS): closure,
//! presign and object GETs over sockets this test starts, driven through the shipped binary.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tensorfs_core::canon::{self, Value};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::header::{Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::jcs::{self, Json};
use tensorfs_core::manifest::{Draft, Entry, Manifest};
use tensorfs_core::store::{Fault, Store};
use tensorfs_core::{gc, registry, source_artifact};

const TFS: &str = env!("CARGO_BIN_EXE_tfs");

fn temporary(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "tfs-ensure-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn store(dir: &Path) -> PathBuf {
    let root = dir.join("store");
    Store::init(&root).unwrap();
    root
}

// ---------------------------------------------------------------- a real checkpoint

struct Checkpoint {
    bodies: HashMap<String, Vec<u8>>,
    manifest: ObjectRef,
    /// Header and blobs, manifest excluded: what the hub's closure lists.
    objects: Vec<ObjectRef>,
    blobs: Vec<ObjectRef>,
}

fn checkpoint(blobs: usize, size: usize, salt: u8) -> Checkpoint {
    let spec = registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let mut bodies = HashMap::new();
    let (mut tensors, mut refs) = (Vec::new(), Vec::new());
    for index in 0..blobs {
        let body: Vec<u8> = (0..size)
            .map(|at| ((at * 31 + index * 7 + salt as usize) % 251) as u8)
            .collect();
        let object = ObjectRef::of(&body);
        tensors.push((
            format!("blocks.{index}.weight"),
            Tensor {
                dtype: Dtype::F32,
                shape: vec![(size / 4) as u64],
                encoding: spec.object_id(),
                parts: vec![(
                    "value".into(),
                    Part {
                        dtype: Dtype::F32,
                        shape: vec![(size / 4) as u64],
                        body: Body::Segments(vec![object.clone()]),
                    },
                )],
            },
        ));
        refs.push(object.clone());
        bodies.insert(object.sha256.clone(), body);
    }
    let header = Header {
        configs: Vec::new(),
        assets: Vec::new(),
        encodings: vec![spec],
        components: vec![("model".into(), tensors)],
    };
    header.validate(&Closure::default()).unwrap();
    let header_bytes = header.canonical_bytes().unwrap();
    let header_ref = ObjectRef::of(&header_bytes);
    bodies.insert(header_ref.sha256.clone(), header_bytes);
    let manifest = Draft {
        entries: vec![("model.cozytensors".into(), Entry::CozyTensors(header_ref))],
    }
    .seal()
    .unwrap()
    .canonical_bytes();
    let manifest_ref = ObjectRef::of(&manifest);
    bodies.insert(manifest_ref.sha256.clone(), manifest);
    let mut objects: Vec<ObjectRef> = bodies
        .iter()
        .filter(|(hex, _)| **hex != manifest_ref.sha256)
        .map(|(hex, body)| ObjectRef {
            sha256: hex.clone(),
            length: body.len() as u64,
        })
        .collect();
    objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    Checkpoint {
        bodies,
        manifest: manifest_ref,
        objects,
        blobs: refs,
    }
}

fn row(object: &ObjectRef) -> Value {
    Value::obj(vec![
        ("length", Value::uint(object.length)),
        ("sha256", Value::str(object.sha256.clone())),
    ])
}

// ---------------------------------------------------------------- the loopback hub

/// How one object GET is answered: its body, or `Truncate` (declare it, send half, close).
enum Answer {
    Body,
    Truncate,
}

type Rule = Arc<dyn Fn(&str, usize) -> Answer + Send + Sync>;

struct Hub {
    base: String,
    gets: Arc<Mutex<HashMap<String, usize>>>,
}

impl Hub {
    fn gets(&self, hex: &str) -> usize {
        *self.gets.lock().unwrap().get(hex).unwrap_or(&0)
    }
    fn total_gets(&self) -> usize {
        self.gets.lock().unwrap().values().sum()
    }
}

/// A hub serving `checkpoint` as `acme/model@r1` lane `public`. `rule` answers each object
/// GET by (digest, how many times it was asked); `gap` paces every object body.
fn hub(
    checkpoint: Arc<Checkpoint>,
    tls: Option<Arc<rustls::ServerConfig>>,
    gap: Duration,
    rule: Rule,
) -> Hub {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let scheme = if tls.is_some() { "https" } else { "http" };
    let base = format!(
        "{scheme}://127.0.0.1:{}",
        listener.local_addr().unwrap().port()
    );
    let gets: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    let (served_base, served_gets) = (base.clone(), Arc::clone(&gets));
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(socket) = socket else { return };
            let (checkpoint, tls, rule) = (Arc::clone(&checkpoint), tls.clone(), Arc::clone(&rule));
            let (base, gets) = (served_base.clone(), Arc::clone(&served_gets));
            std::thread::spawn(move || match tls {
                Some(config) => {
                    let conn = rustls::ServerConnection::new(config).unwrap();
                    let mut stream = rustls::StreamOwned::new(conn, socket);
                    serve(&mut stream, &checkpoint, &base, &gets, gap, &rule);
                    stream.conn.send_close_notify();
                    let _ = stream.flush();
                }
                None => serve(&mut { socket }, &checkpoint, &base, &gets, gap, &rule),
            });
        }
    });
    Hub { base, gets }
}

fn serve(
    stream: &mut (impl Read + Write),
    checkpoint: &Checkpoint,
    base: &str,
    gets: &Mutex<HashMap<String, usize>>,
    gap: Duration,
    rule: &Rule,
) {
    let Some((method, path, body)) = request(stream) else {
        return;
    };
    let json = |value: Value| (200, canon::write(&value));
    let (status, reply, truncate) = match (method.as_str(), path.as_str()) {
        ("POST", "/v1/tensorfs/closure") => {
            let (status, reply) = json(Value::obj(vec![
                ("complete", Value::Bool(true)),
                ("lane", Value::str("public")),
                ("manifest", row(&checkpoint.manifest)),
                ("model", Value::str("acme/model")),
                (
                    "objects",
                    Value::arr(checkpoint.objects.iter().map(row).collect()),
                ),
                ("presign_max_digests", Value::uint(1024)),
                ("release", Value::str("r1")),
                ("scope", Value::str("runtime")),
            ]));
            (status, reply, false)
        }
        ("POST", "/v1/tensorfs/presign") => {
            let Value::Obj(fields) = canon::parse(&body, 1 << 20).unwrap() else {
                panic!("presign body")
            };
            let Some((_, Value::Arr(digests))) = fields.into_iter().find(|(k, _)| k == "digests")
            else {
                panic!("presign digests")
            };
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let urls = digests
                .into_iter()
                .map(|d| {
                    let Value::Str(hex) = d else { panic!() };
                    let url = Value::str(format!("{base}/obj/{hex}"));
                    (hex, url)
                })
                .collect();
            let (status, reply) = json(Value::obj(vec![
                ("expires_at_unix", Value::uint(now + 3600)),
                ("server_time_unix", Value::uint(now)),
                ("urls", Value::map(urls)),
            ]));
            (status, reply, false)
        }
        ("GET", path) if path.starts_with("/obj/") => {
            let hex = path["/obj/".len()..].to_string();
            let asked = {
                let mut gets = gets.lock().unwrap();
                let count = gets.entry(hex.clone()).or_insert(0);
                *count += 1;
                *count
            };
            match checkpoint.bodies.get(&hex) {
                Some(body) => (
                    200,
                    body.clone(),
                    matches!(rule(&hex, asked), Answer::Truncate),
                ),
                None => (404, b"no such object".to_vec(), false),
            }
        }
        _ => (404, b"no such route".to_vec(), false),
    };
    let head = format!(
        "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        reply.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let sent = if truncate {
        &reply[..reply.len() / 2]
    } else {
        &reply[..]
    };
    if gap.is_zero() || !path.starts_with("/obj/") {
        let _ = stream.write_all(sent);
        return;
    }
    for piece in sent.chunks(sent.len().div_ceil(4).max(1)) {
        std::thread::sleep(gap);
        if stream
            .write_all(piece)
            .and_then(|()| stream.flush())
            .is_err()
        {
            return;
        }
    }
}

fn request(stream: &mut impl Read) -> Option<(String, String, Vec<u8>)> {
    let (mut collected, mut buf) = (Vec::new(), [0u8; 4096]);
    let split = loop {
        if let Some(at) = collected.windows(4).position(|w| w == b"\r\n\r\n") {
            break at;
        }
        let n = stream.read(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        collected.extend_from_slice(&buf[..n]);
    };
    let head = String::from_utf8_lossy(&collected[..split]).to_string();
    let mut body = collected[split + 4..].to_vec();
    let mut line = head.lines().next()?.split_whitespace();
    let (method, target) = (line.next()?.to_string(), line.next()?.to_string());
    let length: usize = head
        .lines()
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.trim().parse().ok())
        .unwrap_or(0);
    while body.len() < length {
        let n = stream.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }
    Some((method, target.split('?').next()?.to_string(), body))
}

fn serve_all() -> Rule {
    Arc::new(|_, _| Answer::Body)
}

// ---------------------------------------------------------------- the binary

struct Run {
    ok: bool,
    lines: Vec<Json>,
    stdout: String,
}

impl Run {
    fn events(&self, name: &str) -> Vec<&Json> {
        self.lines
            .iter()
            .filter(|line| text(line, "event") == name)
            .collect()
    }
    fn last(&self) -> &Json {
        self.lines.last().expect("ensure printed nothing")
    }
}

fn field<'a>(line: &'a Json, name: &str) -> Option<&'a Json> {
    let Json::Obj(fields) = line else { return None };
    fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn num(line: &Json, name: &str) -> u64 {
    match field(line, name) {
        Some(Json::Num(n)) if *n >= 0.0 && n.fract() == 0.0 => *n as u64,
        other => panic!("{name} is {other:?} in {line:?}"),
    }
}

fn text(line: &Json, name: &str) -> String {
    match field(line, name) {
        Some(Json::Str(s)) => s.clone(),
        other => panic!("{name} is {other:?} in {line:?}"),
    }
}

fn flag(line: &Json, name: &str) -> bool {
    match field(line, name) {
        Some(Json::Bool(b)) => *b,
        other => panic!("{name} is {other:?} in {line:?}"),
    }
}

fn parse(line: &str) -> Json {
    jcs::parse(line.as_bytes(), 1 << 20).unwrap_or_else(|e| panic!("{e}: {line}"))
}

fn ensure(root: &Path, base: &str, extra: &[&str]) -> Command {
    let mut command = Command::new(TFS);
    command
        .args([
            "ensure",
            root.to_str().unwrap(),
            "acme/model@r1",
            "--hub",
            base,
        ])
        .args([
            "--lane",
            "public",
            "--step",
            "load weights",
            "--sample-seconds",
            "0.05",
        ])
        .args(extra)
        .env("TFS_CREDENTIAL", "");
    command
}

fn finish(output: std::process::Output) -> Run {
    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines = stdout.lines().map(parse).collect();
    Run {
        ok: output.status.success(),
        lines,
        stdout,
    }
}

fn run(root: &Path, base: &str, extra: &[&str]) -> Run {
    finish(ensure(root, base, extra).output().unwrap())
}

fn assert_result(run: &Run, checkpoint: &Checkpoint) -> Json {
    assert!(run.ok, "{}", run.stdout);
    let result = run.last().clone();
    assert_eq!(text(&result, "event"), "ensure.result", "{}", run.stdout);
    assert_eq!(text(&result, "manifest"), checkpoint.manifest.id());
    assert_eq!(text(&result, "model"), "acme/model");
    assert_eq!(text(&result, "lane"), "public");
    let total: u64 = checkpoint.bodies.values().map(|b| b.len() as u64).sum();
    assert_eq!(num(&result, "bytes_total"), total);
    result
}

fn refused(run: &Run) -> (String, bool) {
    assert!(!run.ok, "{}", run.stdout);
    let last = run.last();
    assert_eq!(text(last, "event"), "ensure.refused", "{}", run.stdout);
    (text(last, "code"), flag(last, "resumable"))
}

// ---------------------------------------------------------------- the proofs

#[test]
fn a_cold_ensure_reports_and_lands_and_a_warm_one_moves_nothing() {
    let dir = temporary("cold-warm");
    let root = store(&dir);
    let checkpoint = Arc::new(checkpoint(3, 64 << 10, 1));
    let hub = hub(Arc::clone(&checkpoint), None, Duration::ZERO, serve_all());
    let total: u64 = checkpoint.bodies.values().map(|b| b.len() as u64).sum();

    let cold = run(&root, &hub.base, &[]);
    let result = assert_result(&cold, &checkpoint);
    assert_eq!(num(&result, "bytes_held"), 0);
    assert_eq!(num(&result, "bytes_fetched"), total);
    assert_eq!(num(&result, "attempts"), 1);
    assert!(!flag(&result, "joined"));
    let progress = cold.events("ensure.progress");
    assert_eq!(text(progress[0], "phase"), "resolving");
    assert!(progress
        .iter()
        .all(|e| text(e, "step") == "load weights" && text(e, "model") == "acme/model"));
    let last = progress.last().unwrap();
    assert_eq!(
        (text(last, "phase").as_str(), num(last, "bytes_done")),
        ("fetching", total)
    );
    assert_eq!(num(last, "bytes_total"), total);
    assert!(checkpoint.bodies.keys().all(|hex| hub.gets(hex) == 1));

    let warm = run(&root, &hub.base, &[]);
    let result = assert_result(&warm, &checkpoint);
    assert_eq!(
        (num(&result, "bytes_held"), num(&result, "bytes_fetched")),
        (total, 0)
    );
    assert_eq!(
        hub.total_gets(),
        checkpoint.bodies.len(),
        "a warm ensure moved bytes"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_ranged_get_streams_exact_bytes_under_a_lease() {
    let dir = temporary("get-range");
    let root = store(&dir);
    let checkpoint = Arc::new(checkpoint(1, 256 << 10, 2));
    let hub = hub(Arc::clone(&checkpoint), None, Duration::ZERO, serve_all());
    assert_result(&run(&root, &hub.base, &[]), &checkpoint);
    let blob = &checkpoint.blobs[0];
    let body = &checkpoint.bodies[&blob.sha256];
    let get = |range: &str| {
        Command::new(TFS)
            .args([
                "get",
                root.to_str().unwrap(),
                &blob.id(),
                "--range",
                range,
                "--out",
                "-",
            ])
            .output()
            .unwrap()
    };
    let out = get("5:100000");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, body[5..100_005]);
    assert_eq!(get("200000:").stdout, body[200_000..]);
    let whole = Command::new(TFS)
        .args(["get", root.to_str().unwrap(), &blob.id(), "--out", "-"])
        .output()
        .unwrap();
    assert_eq!(&whole.stdout, body);
    let beyond = get(&format!("{}:1", body.len()));
    assert!(!beyond.status.success() && beyond.stdout.is_empty());
    assert!(String::from_utf8_lossy(&beyond.stderr).contains("RANGE_BOUNDS"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn concurrent_ensures_of_one_manifest_share_one_fetch() {
    let dir = temporary("single-flight");
    let root = store(&dir);
    let checkpoint = Arc::new(checkpoint(4, 64 << 10, 3));
    // Paced so the leader is still fetching when the follower asks.
    let hub = hub(
        Arc::clone(&checkpoint),
        None,
        Duration::from_millis(150),
        serve_all(),
    );
    let mut leader = ensure(
        &root,
        &hub.base,
        &["--streams", "1", "--streams-start", "1"],
    )
    .stdout(Stdio::piped())
    .spawn()
    .unwrap();
    let mut lines = BufReader::new(leader.stdout.take().unwrap()).lines();
    let mut seen = Vec::new();
    for line in lines.by_ref() {
        let line = line.unwrap();
        let fetching = parse(&line);
        seen.push(line);
        if field(&fetching, "phase") == Some(&Json::Str("fetching".into())) {
            break;
        }
    }
    let follower = run(&root, &hub.base, &[]);
    seen.extend(lines.map(|l| l.unwrap()));
    assert!(leader.wait().unwrap().success(), "{}", seen.join("\n"));

    let result = assert_result(&follower, &checkpoint);
    assert!(flag(&result, "joined"), "{}", follower.stdout);
    assert_eq!(num(&result, "bytes_fetched"), 0);
    assert!(follower
        .events("ensure.progress")
        .iter()
        .any(|e| text(e, "phase") == "waiting"));
    assert!(
        checkpoint.bodies.keys().all(|hex| hub.gets(hex) == 1),
        "an object was fetched twice: {:?}",
        hub.gets.lock().unwrap()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_attempt_that_landed_objects_is_retried_and_one_that_landed_none_is_stalled() {
    let dir = temporary("retry");
    let checkpoint = Arc::new(checkpoint(3, 64 << 10, 4));
    let stubborn = checkpoint.blobs[0].sha256.clone();
    // One object's first four answers die mid-body. Nothing counts asks, so the pull asks
    // it again while it lands bytes, and the first attempt brings everything home.
    let rule: Rule = Arc::new(move |hex, asked| {
        if hex == stubborn && asked <= 4 {
            Answer::Truncate
        } else {
            Answer::Body
        }
    });
    let hub_a = hub(Arc::clone(&checkpoint), None, Duration::ZERO, rule);
    let recovered = run(&store(&dir), &hub_a.base, &[]);
    let result = assert_result(&recovered, &checkpoint);
    assert_eq!(num(&result, "attempts"), 1, "{}", recovered.stdout);

    let never: Rule = Arc::new(|_, _| Answer::Truncate);
    let hub_b = hub(Arc::clone(&checkpoint), None, Duration::ZERO, never);
    let root = dir.join("stalled");
    Store::init(&root).unwrap();
    let stalled = run(&root, &hub_b.base, &[]);
    assert_eq!(
        refused(&stalled),
        ("STALLED".to_string(), true),
        "{}",
        stalled.stdout
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A loopback TLS server config under a fresh self-signed authority, and that authority.
fn tls_config() -> (Arc<rustls::ServerConfig>, String) {
    let key = rcgen::generate_simple_self_signed(vec!["127.0.0.1".into()]).unwrap();
    let cert = rustls::pki_types::CertificateDer::from(key.cert.der().to_vec());
    let private = rustls::pki_types::PrivateKeyDer::try_from(key.key_pair.serialize_der())
        .unwrap()
        .clone_key();
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(vec![cert], private)
    .unwrap();
    (Arc::new(config), key.cert.pem())
}

#[test]
fn an_untrusted_hub_is_terminal_and_its_named_ca_admits_it() {
    let dir = temporary("tls");
    let root = store(&dir);
    let (config, pem) = tls_config();
    let checkpoint = Arc::new(checkpoint(2, 16 << 10, 5));
    let hub = hub(
        Arc::clone(&checkpoint),
        Some(config),
        Duration::ZERO,
        serve_all(),
    );

    let untrusted = run(&root, &hub.base, &[]);
    assert_eq!(refused(&untrusted), ("TLS_UNTRUSTED".to_string(), false));
    let ca = dir.join("hub-ca.pem");
    std::fs::write(&ca, pem).unwrap();
    assert_result(
        &run(&root, &hub.base, &["--ca-file", ca.to_str().unwrap()]),
        &checkpoint,
    );
    let garbage = dir.join("garbage.pem");
    std::fs::write(&garbage, "not a certificate").unwrap();
    let bad = run(&root, &hub.base, &["--ca-file", garbage.to_str().unwrap()]);
    assert_eq!(refused(&bad).0, "UNKNOWN_FORMAT");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The library path, in one process: a shared client that already refused a hub trusts it
/// once its CA is added, with no new client and no restart.
#[test]
fn trust_added_after_a_refusal_reaches_the_same_process() {
    use tensorfs_core::ensure::{self, Event, Phase, Request};
    use tensorfs_core::transport::{self, Anonymous, SourcePolicy};
    let dir = temporary("trust-in-process");
    let store = Store::init(&dir.join("store")).unwrap();
    let (config, pem) = tls_config();
    let checkpoint = Arc::new(checkpoint(2, 16 << 10, 7));
    let hub = hub(
        Arc::clone(&checkpoint),
        Some(config),
        Duration::ZERO,
        serve_all(),
    );
    let policy = SourcePolicy::default();
    let events = Mutex::new(Vec::new());
    let on_event = |event: &Event| events.lock().unwrap().push(event.clone());
    let mut request = Request::new(&store, &hub.base, "acme/model@r1", &Anonymous, &policy);
    request.lane = "public";
    request.step = "warm";
    request.on_event = Some(&on_event);
    let refused = ensure::ensure(&request).unwrap_err();
    assert_eq!(refused.code, tensorfs_core::err::Code::TLS_UNTRUSTED);
    assert!(!ensure::resumable(refused.code));
    transport::trust_roots(pem.as_bytes()).unwrap();
    transport::trust_roots(pem.as_bytes()).unwrap();
    let ensured = ensure::ensure(&request).unwrap();
    assert_eq!(ensured.manifest, checkpoint.manifest);
    assert_eq!(ensured.bytes_fetched, ensured.bytes_total);
    let events = events.into_inner().unwrap();
    assert!(events
        .iter()
        .all(|e| e.step == "warm" && e.model == "acme/model"));
    assert_eq!(events.last().unwrap().phase, Phase::Fetching);
    assert_eq!(events.last().unwrap().bytes_done, ensured.bytes_total);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_closure_the_disk_cannot_hold_is_refused_before_a_byte_moves() {
    let dir = temporary("capacity");
    let root = store(&dir);
    let mut huge = checkpoint(1, 16 << 10, 6);
    // The hub declares a petabyte object: no disk this test runs on admits it.
    let petabyte = ObjectRef {
        sha256: "ab".repeat(32),
        length: 1 << 50,
    };
    huge.objects.push(petabyte);
    huge.objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    let hub = hub(Arc::new(huge), None, Duration::ZERO, serve_all());
    let full = run(&root, &hub.base, &[]);
    assert_eq!(
        refused(&full),
        ("CAPACITY_EXHAUSTED".to_string(), true),
        "{}",
        full.stdout
    );
    let detail = text(full.last(), "detail");
    assert!(
        detail.contains("this download needs 11258999068") && detail.contains("GC freed"),
        "{detail}"
    );
    assert_eq!(hub.total_gets(), 0, "admission let bytes move");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delivered_products_are_evictable_oldest_first_and_release_clears_the_mark() {
    let dir = temporary("deliver");
    let store = Store::init(&dir.join("store")).unwrap();
    let owners: Vec<String> = ["a", "b"]
        .iter()
        .map(|l| ObjectRef::of(l.as_bytes()).id())
        .collect();
    for (owner, bytes) in owners
        .iter()
        .zip([b"first product".as_slice(), b"second product"])
    {
        let object = ObjectRef::of(bytes);
        store
            .put_stream(&mut &bytes[..], Some(&object), &Fault::default())
            .unwrap();
        let tree = Manifest::from_files(vec![("out.bin".into(), object)]).unwrap();
        source_artifact::create(&store, owner, &tree).unwrap();
    }
    assert!(source_artifact::delivered(&store).unwrap().is_empty());
    source_artifact::deliver(&store, &owners[1]).unwrap();
    std::thread::sleep(Duration::from_millis(20));
    source_artifact::deliver(&store, &owners[0]).unwrap();
    source_artifact::deliver(&store, &owners[1]).unwrap();
    assert_eq!(
        source_artifact::delivered(&store).unwrap(),
        vec![owners[1].clone(), owners[0].clone()]
    );
    // Delivered is not released: GC keeps its bytes until the policy releases it.
    assert_eq!(gc::collect(store.root(), false).unwrap().reclaimed_bytes, 0);
    source_artifact::release(&store, &owners[1]).unwrap();
    assert_eq!(
        source_artifact::delivered(&store).unwrap(),
        vec![owners[0].clone()]
    );
    assert!(gc::collect(store.root(), false).unwrap().reclaimed_bytes > 0);
    let unknown = ObjectRef::of(b"never retained").id();
    assert!(source_artifact::deliver(&store, &unknown).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn capabilities_name_ensure() {
    let out = Command::new(TFS).arg("capabilities").output().unwrap();
    let names = String::from_utf8(out.stdout).unwrap();
    assert!(
        out.status.success() && names.lines().any(|n| n == "ensure/1"),
        "{names}"
    );
    assert!(names.lines().any(|n| n == "get-range/1"));
}

#[test]
fn a_pressure_pass_reports_the_disk_and_keeps_what_it_is_told() {
    let dir = temporary("pressure");
    let root = store(&dir);
    let keep = ObjectRef::of(b"serving").id();
    let out = Command::new(TFS)
        .args(["pressure", root.to_str().unwrap(), "--keep", &keep])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let line = parse(String::from_utf8(out.stdout).unwrap().trim());
    assert_eq!(text(&line, "event"), "pressure");
    assert!(num(&line, "capacity_bytes") >= num(&line, "available_bytes"));
    // This box is not under pressure, and a pass that finds none collects nothing.
    if !flag(&line, "pressure") {
        assert_eq!(num(&line, "collected_bytes"), 0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
