//! Transport proofs against a REAL store and REAL sockets — every origin here is a
//! loopback listener the test starts itself, so the whole suite runs with no network and
//! no hub. The ledger tests run at millisecond resolution: the cadence is configuration,
//! the RULE under test is production's.

use super::credential::{credential_from_spec, Anonymous, ScopedHeaders};
use super::download::{fetch_ranged, Ranged};
use super::ledger::{Deadline, Ledger};
use super::policy::{self, blocked, SourcePolicy};
use super::pull::{absolute_location, Fetched};

use crate::canon::Value;
use crate::dtype::Dtype;
use crate::err::Code;
use crate::fetch_v1::{DeliveryGrant, FetchPlan};
use crate::header::{Body, Closure, Header, Part, Tensor};
use crate::ids::{Doc, ObjectRef};
use crate::manifest::{Draft, Entry};
use crate::registry;
use crate::store::Store;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-transport-{name}-{}-{}",
        std::process::id(),
        crate::meta::now_nanos_unique()
    ))
}

// ---------------------------------------------------------------- the loopback origin

struct Request {
    method: String,
    path: String,
    /// Everything after `?`. Hits are counted by PATH, so a URL that carries its own expiry
    /// here is still the same object to every existing assertion.
    query: String,
    headers: Vec<(String, String)>,
    #[allow(dead_code)]
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

pub(crate) enum Serve {
    /// Read the request, then hold the socket without sending response headers.
    StallHead,
    /// A controlled real-socket exchange for cancellation races.
    Script(Box<dyn FnOnce(&mut TcpStream) + Send>),
    /// Content-Length framed whole answer.
    Whole {
        status: u16,
        body: Vec<u8>,
        headers: Vec<(String, String)>,
    },
    /// Declare `total`, send only `prefix`, then hold the socket open until the client
    /// gives up on it — a black-holed stream.
    Stall { total: u64, prefix: Vec<u8> },
    /// Declare `total`, send only `prefix`, close — a connection that died mid-object.
    Truncate { total: u64, prefix: Vec<u8> },
    /// A chosen status and headers, `declared` bytes promised, `prefix` actually sent, then
    /// the socket closes — a ranged answer that died part way through its own range.
    Short {
        status: u16,
        headers: Vec<(String, String)>,
        declared: u64,
        prefix: Vec<u8>,
    },
    /// Sleep `ttfb`, then send the body in `chunk`-sized pieces with `gap` between them —
    /// a slow link that is honestly moving.
    Drip {
        body: Vec<u8>,
        chunk: usize,
        gap: Duration,
        ttfb: Duration,
    },
}

fn ok_json(value: &Value) -> Serve {
    Serve::Whole {
        status: 200,
        body: crate::canon::write(value),
        headers: vec![("Content-Type".into(), "application/json".into())],
    }
}

fn status(code: u16, body: &[u8]) -> Serve {
    Serve::Whole {
        status: code,
        body: body.to_vec(),
        headers: Vec::new(),
    }
}

type Dispatch = Arc<dyn Fn(&Request) -> Serve + Send + Sync>;

pub(crate) struct Origin {
    pub(crate) base: String,
    hits: Arc<Mutex<HashMap<String, usize>>>,
}

impl Origin {
    pub(crate) fn hits(&self, path: &str) -> usize {
        *self.hits.lock().unwrap().get(path).unwrap_or(&0)
    }
    fn total_hits(&self) -> usize {
        self.hits.lock().unwrap().values().sum()
    }
}

/// One handler thread per connection; the accept loop is detached and dies with the
/// process, which for a test binary is the end of the test run.
fn origin(dispatch: Dispatch) -> Origin {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let hits: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    let counted = Arc::clone(&hits);
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(socket) = socket else { break };
            let dispatch = Arc::clone(&dispatch);
            let counted = Arc::clone(&counted);
            std::thread::spawn(move || handle(socket, dispatch, counted));
        }
    });
    Origin { base, hits }
}

fn handle(mut socket: TcpStream, dispatch: Dispatch, hits: Arc<Mutex<HashMap<String, usize>>>) {
    let Some(request) = read_request(&mut socket) else {
        return;
    };
    *hits
        .lock()
        .unwrap()
        .entry(request.path.clone())
        .or_insert(0) += 1;
    match dispatch(&request) {
        Serve::Script(run) => run(&mut socket),
        Serve::StallHead => {
            let mut sink = [0u8; 16];
            let _ = socket.set_read_timeout(Some(Duration::from_secs(30)));
            let _ = socket.read(&mut sink);
        }
        Serve::Whole {
            status,
            body,
            headers,
        } => {
            let mut head = format!(
                "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n",
                body.len()
            );
            for (name, value) in headers {
                head.push_str(&format!("{name}: {value}\r\n"));
            }
            head.push_str("\r\n");
            let _ = socket.write_all(head.as_bytes());
            let _ = socket.write_all(&body);
        }
        Serve::Stall { total, prefix } => {
            let head =
                format!("HTTP/1.1 200 OK\r\ncontent-length: {total}\r\nconnection: close\r\n\r\n");
            let _ = socket.write_all(head.as_bytes());
            let _ = socket.write_all(&prefix);
            let _ = socket.flush();
            // Hold the stream open, sending nothing, until the client shuts it down.
            let mut sink = [0u8; 16];
            let _ = socket.set_read_timeout(Some(Duration::from_secs(30)));
            let _ = socket.read(&mut sink);
        }
        Serve::Truncate { total, prefix } => {
            let head =
                format!("HTTP/1.1 200 OK\r\ncontent-length: {total}\r\nconnection: close\r\n\r\n");
            let _ = socket.write_all(head.as_bytes());
            let _ = socket.write_all(&prefix);
        }
        Serve::Short {
            status,
            headers,
            declared,
            prefix,
        } => {
            let mut head = format!(
                "HTTP/1.1 {status} X\r\ncontent-length: {declared}\r\nconnection: close\r\n"
            );
            for (name, value) in headers {
                head.push_str(&format!("{name}: {value}\r\n"));
            }
            head.push_str("\r\n");
            let _ = socket.write_all(head.as_bytes());
            let _ = socket.write_all(&prefix);
        }
        Serve::Drip {
            body,
            chunk,
            gap,
            ttfb,
        } => {
            std::thread::sleep(ttfb);
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(head.as_bytes());
            let _ = socket.flush();
            // An ABSOLUTE schedule, so sleep overshoot on a loaded box does not compound:
            // chunk i is due at start + (i+1)*gap, which is what a paced link looks like.
            let start = std::time::Instant::now();
            for (index, piece) in body.chunks(chunk.max(1)).enumerate() {
                let due = start + gap * (index as u32 + 1);
                let now = std::time::Instant::now();
                if due > now {
                    std::thread::sleep(due - now);
                }
                if socket.write_all(piece).is_err() {
                    return;
                }
                let _ = socket.flush();
            }
        }
    }
}

fn read_request(socket: &mut TcpStream) -> Option<Request> {
    let mut collected = Vec::new();
    let mut buf = [0u8; 4096];
    let split = loop {
        if let Some(at) = collected.windows(4).position(|w| w == b"\r\n\r\n") {
            break at;
        }
        let n = socket.read(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        collected.extend_from_slice(&buf[..n]);
    };
    let head = String::from_utf8_lossy(&collected[..split]).to_string();
    let mut body = collected[split + 4..].to_vec();
    let mut lines = head.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?;
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), query.to_string()),
        None => (target.to_string(), String::new()),
    };
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let length: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    while body.len() < length {
        let n = socket.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }
    Some(Request {
        method,
        path,
        query,
        headers,
        body,
    })
}

// ---------------------------------------------------------------- a real checkpoint

/// A synthetic checkpoint whose runtime closure is `blobs + header`: one tensor per blob,
/// each carried as a segment, exactly the shape a hub `closure` answer declares.
pub(crate) struct Checkpoint {
    /// Every object body the ORIGIN can serve, by bare hex digest (manifest included).
    pub(crate) bodies: HashMap<String, Vec<u8>>,
    pub(crate) manifest: ObjectRef,
    /// The closure rows: header + blobs, sorted by digest, manifest EXCLUDED.
    pub(crate) objects: Vec<ObjectRef>,
}

fn checkpoint(blobs: usize, size: usize) -> Checkpoint {
    checkpoint_sizes(&vec![size; blobs])
}

fn checkpoint_sizes(sizes: &[usize]) -> Checkpoint {
    assert!(sizes.iter().all(|size| size.is_multiple_of(4) && *size > 0));
    let spec = registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let mut bodies = HashMap::new();
    let mut tensors = Vec::new();
    for (index, &size) in sizes.iter().enumerate() {
        let mut body = vec![0u8; size];
        for (at, byte) in body.iter_mut().enumerate() {
            *byte = ((at * 31 + index * 7 + 3) % 251) as u8;
        }
        // The pattern above repeats every 251 indices, and a repeated body is the SAME
        // object — a checkpoint of a few thousand blobs would silently collapse to 251 of
        // them, which is why no test in this file has ever exercised a large closure.
        // Stamping the index in keeps one blob one object at any count.
        for (at, byte) in (index as u32).to_le_bytes().iter().enumerate() {
            body[at] ^= byte;
        }
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
    .unwrap();
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_ref = ObjectRef::of(&manifest_bytes);
    bodies.insert(manifest_ref.sha256.clone(), manifest_bytes);
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
    }
}

fn closure_value(checkpoint: &Checkpoint, base: &str) -> Value {
    let _ = base;
    let row = |o: &ObjectRef| {
        Value::obj(vec![
            ("length", Value::uint(o.length)),
            ("sha256", Value::str(o.sha256.clone())),
        ])
    };
    Value::obj(vec![
        ("lane", Value::str("public")),
        ("manifest", row(&checkpoint.manifest)),
        ("model", Value::str("acme/model")),
        (
            "objects",
            Value::arr(checkpoint.objects.iter().map(row).collect()),
        ),
        ("release", Value::str("r1")),
        ("scope", Value::str("runtime")),
        ("complete", Value::Bool(true)),
        ("presign_max_digests", Value::uint(1956)),
    ])
}

fn presign_value(request_body: &[u8], base: &str) -> Value {
    let asked = crate::canon::parse(request_body, 1 << 20).unwrap();
    let Value::Obj(fields) = asked else { panic!() };
    let Some((_, Value::Arr(digests))) = fields.into_iter().find(|(k, _)| k == "digests") else {
        panic!()
    };
    let urls: Vec<(String, Value)> = digests
        .into_iter()
        .map(|d| {
            let Value::Str(hex) = d else { panic!() };
            let url = format!("{base}/obj/{hex}");
            (hex, Value::str(url))
        })
        .collect();
    let server_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    Value::obj(vec![
        ("expires_at_unix", Value::uint(server_now + 3600)),
        ("server_time_unix", Value::uint(server_now)),
        ("urls", Value::map(urls)),
    ])
}

/// The standard hub emulation: closure + presign + object GETs, with per-object override.
pub(crate) fn hub_origin(
    checkpoint: Arc<Checkpoint>,
    special: impl Fn(&str, usize) -> Option<Serve> + Send + Sync + 'static,
) -> Origin {
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let hits: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    let dispatch_hits = Arc::clone(&hits);
    let dispatch_base = Arc::clone(&base_slot);
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = dispatch_base.lock().unwrap().clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/tensorfs/closure") => ok_json(&closure_value(&checkpoint, &base)),
            ("POST", "/v1/tensorfs/presign") => ok_json(&presign_value(&request.body, &base)),
            ("GET", path) if path.starts_with("/obj/") => {
                let hex = &path["/obj/".len()..];
                let attempt = *dispatch_hits.lock().unwrap().get(path).unwrap_or(&1);
                if let Some(serve) = special(hex, attempt) {
                    return serve;
                }
                match checkpoint.bodies.get(hex) {
                    Some(body) => status(200, body),
                    None => status(404, b"no such object"),
                }
            }
            _ => status(404, b"no such route"),
        }
    });
    let origin = self::origin_with_hits(dispatch, hits);
    *base_slot.lock().unwrap() = origin.base.clone();
    origin
}

fn origin_with_hits(dispatch: Dispatch, hits: Arc<Mutex<HashMap<String, usize>>>) -> Origin {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let counted = Arc::clone(&hits);
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(socket) = socket else { break };
            let dispatch = Arc::clone(&dispatch);
            let counted = Arc::clone(&counted);
            std::thread::spawn(move || handle(socket, dispatch, counted));
        }
    });
    Origin { base, hits }
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

/// A hub whose presigned URLs DIE, and an object store that enforces it.
///
/// Every URL carries the wall-clock instant it stops working, the presign answer declares
/// the same life in the hub's own clock fields (`expires_at_unix`/`server_time_unix`), and a
/// GET presented after that instant is refused the way an object store refuses a signature
/// that has aged out. This is the only origin here that models the one fact the hub's cap
/// exists to state, and it is what makes the walk-outlives-the-cap proof mean anything.
struct Expiring {
    origin: Origin,
    lifetime: Duration,
    /// How many times a URL was presented after it died — the proof that the origin is
    /// actually enforcing, rather than a test that could not fail.
    rejected: Arc<AtomicUsize>,
}

fn expiring_presign_value(request_body: &[u8], base: &str, lifetime: Duration) -> Value {
    let asked = crate::canon::parse(request_body, 1 << 20).unwrap();
    let Value::Obj(fields) = asked else { panic!() };
    let Some((_, Value::Arr(digests))) = fields.into_iter().find(|(k, _)| k == "digests") else {
        panic!()
    };
    let dies = now_millis() + lifetime.as_millis();
    let server_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let urls: Vec<(String, Value)> = digests
        .into_iter()
        .map(|d| {
            let Value::Str(hex) = d else { panic!() };
            let url = format!("{base}/obj/{hex}?dies={dies}");
            (hex, Value::str(url))
        })
        .collect();
    Value::obj(vec![
        (
            "expires_at_unix",
            Value::uint(server_now + lifetime.as_secs()),
        ),
        ("server_time_unix", Value::uint(server_now)),
        ("urls", Value::map(urls)),
    ])
}

fn expiring_hub_origin(
    checkpoint: Arc<Checkpoint>,
    lifetime: Duration,
    ttfb: Duration,
    gap: Duration,
) -> Expiring {
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let hits: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    let rejected = Arc::new(AtomicUsize::new(0));
    let dispatch_base = Arc::clone(&base_slot);
    let dispatch_rejected = Arc::clone(&rejected);
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = dispatch_base.lock().unwrap().clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/tensorfs/closure") => ok_json(&closure_value(&checkpoint, &base)),
            ("POST", "/v1/tensorfs/presign") => {
                ok_json(&expiring_presign_value(&request.body, &base, lifetime))
            }
            ("GET", path) if path.starts_with("/obj/") => {
                let hex = &path["/obj/".len()..];
                let dies: u128 = request
                    .query
                    .strip_prefix("dies=")
                    .and_then(|stamp| stamp.parse().ok())
                    .expect("every URL this hub mints carries its own expiry");
                if now_millis() > dies {
                    dispatch_rejected.fetch_add(1, Ordering::Relaxed);
                    return status(403, b"{\"error\":{\"code\":\"AccessDenied\"}}");
                }
                match checkpoint.bodies.get(hex) {
                    Some(body) => Serve::Drip {
                        body: body.clone(),
                        chunk: (body.len() / 4).max(1),
                        gap,
                        ttfb,
                    },
                    None => status(404, b"no such object"),
                }
            }
            _ => status(404, b"no such route"),
        }
    });
    let origin = origin_with_hits(dispatch, hits);
    *base_slot.lock().unwrap() = origin.base.clone();
    Expiring {
        origin,
        lifetime,
        rejected,
    }
}

fn local_policy() -> SourcePolicy {
    SourcePolicy {
        allowed_hosts: vec!["127.0.0.1".into()],
        allow_local: true,
        max_redirects: 0,
        ..Default::default()
    }
}

/// At a 0.25 s sample the watch restarts a request only once it is 100 ms (two ticks) behind its
/// peers. At 25 ms that was 10 ms, which a loopback answer under a loaded runner often misses:
/// healthy objects were restarted and asked twice.
fn request_for<'a>(store: &'a Store, base: &'a str, policy: &'a SourcePolicy) -> PullRequest<'a> {
    let mut request = PullRequest::new(store, base, "acme/model", Anonymous, policy);
    request.sample_seconds = 0.25;
    request
}

/// One whole checkpoint the way the machine downloads one: its closure asked of the hub, then
/// [`crate::fetch_v1::whole`] on a Fetcher of this request's streams and sample.
struct PullRequest<'a> {
    store: &'a Store,
    base: &'a str,
    refspec: &'a str,
    lane: &'a str,
    access: Arc<dyn crate::fetch_v1::Access>,
    streams: usize,
    sample_seconds: f64,
    cancellation: Option<super::PullCancellation>,
    /// Told the bytes here and in all as the download runs.
    on_progress: Option<&'a (dyn Fn(u64, u64) + Sync)>,
    on_plan: Option<&'a (dyn Fn(&FetchPlan) + Sync)>,
}

impl<'a> PullRequest<'a> {
    fn new(
        store: &'a Store,
        base: &'a str,
        refspec: &'a str,
        credential: impl super::CredentialProvider + Send + 'static,
        policy: &SourcePolicy,
    ) -> PullRequest<'a> {
        PullRequest {
            store,
            base,
            refspec,
            lane: "",
            access: Arc::new(crate::fetch_v1::Plain { credential, policy: policy.clone() }),
            streams: super::PULL_STREAMS,
            sample_seconds: super::SAMPLE_SECONDS,
            cancellation: None,
            on_progress: None,
            on_plan: None,
        }
    }
}

/// What one whole download did: the plan it began from and its walks' own counts.
#[derive(Debug, Default)]
struct Pulled {
    model: String,
    release: String,
    lane: String,
    declared: u64,
    held: u64,
    bytes_held: u64,
    bytes_total: u64,
    fetched: u64,
    bytes_moved: u64,
    streams: usize,
    writers: usize,
    requests: u64,
    restarts: u64,
}

fn pull(request: &PullRequest<'_>) -> crate::err::Result<Pulled> {
    let sample = Duration::from_secs_f64(request.sample_seconds);
    let mut ledger = Ledger::with_resolution(sample);
    if let Some(stop) = &request.cancellation {
        ledger = ledger.with_cancellation(stop.clone());
    }
    let closure = super::hub::closure_all(
        super::shared_client(),
        request.base,
        request.refspec,
        request.lane,
        request.access.credential(),
        &super::hub_policy(request.base, request.access.policy()),
        Deadline::none(),
        &ledger,
    )?;
    let (plan, _) = FetchPlan::of(request.store, "pull", &closure.manifest, &closure.objects)?;
    if let Some(on_plan) = request.on_plan {
        on_plan(&plan);
    }
    let mut pulled = Pulled {
        model: closure.model.clone(),
        release: closure.release.clone(),
        lane: closure.lane.clone(),
        declared: (plan.held.len() + plan.wanted.len()) as u64,
        held: plan.held.len() as u64,
        bytes_held: plan.held_bytes(),
        bytes_total: plan.declared_bytes(),
        ..Pulled::default()
    };
    let walks = walk_lines(request.store).len();
    let fetcher = crate::fetch_v1::Fetcher::new(request.store, request.streams, sample)?;
    let origin = crate::fetch_v1::Origin {
        hub: request.base.to_string(),
        refspec: request.refspec.to_string(),
        lane: request.lane.to_string(),
        manifest: closure.manifest.clone(),
    };
    fetcher.seed(&origin, closure);
    let stop = request.cancellation.clone().unwrap_or_default();
    let progress = |done: u64, total: u64| {
        if let Some(on_progress) = request.on_progress {
            on_progress(done, total);
        }
    };
    let priority = crate::fetch_v1::Priority::Now;
    let landed = crate::fetch_v1::whole(&fetcher, &origin, Arc::clone(&request.access), priority, &stop, &progress);
    // Its last handle waits for its downloader, whose walks are then on record.
    drop(fetcher);
    landed?;
    for line in walk_lines(request.store).into_iter().skip(walks) {
        let field = |name: &str| -> u64 {
            (line.split_whitespace())
                .find_map(|f| f.strip_prefix(name)?.strip_prefix('=')?.parse().ok())
                .unwrap_or(0)
        };
        pulled.fetched += field("fetched");
        pulled.bytes_moved += field("moved");
        pulled.requests += field("requests");
        pulled.restarts += field("restarts");
        pulled.streams = pulled.streams.max(field("streams") as usize);
        pulled.writers = pulled.writers.max(field("writers") as usize);
    }
    Ok(pulled)
}

/// The Store's download walks so far, oldest first.
fn walk_lines(store: &Store) -> Vec<String> {
    std::fs::read_to_string(store.root().join("logs/transport.log"))
        .unwrap_or_default()
        .lines()
        .filter(|line| line.contains(" walk "))
        .map(str::to_string)
        .collect()
}

// ---------------------------------------------------------------- pulls

#[test]
fn cold_pull_lands_and_warm_pull_moves_nothing() {
    let root = temporary("cold-warm");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(6, 32 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default(); // the hub base declares itself; loopback is local
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(report.declared, 8); // 6 blobs + header + manifest
    assert_eq!(report.held, 0);
    assert_eq!(report.fetched, 8);
    assert_eq!(
        report.bytes_moved,
        checkpoint.bodies.values().map(|b| b.len() as u64).sum::<u64>()
    );
    assert_eq!(report.model, "acme/model");
    assert_eq!(report.release, "r1");
    // Cold acquisition may retry a delayed response under concurrent CI load.
    // Warm reuse must add exactly the closure request, regardless of those retries.
    let calls_after_cold = origin.total_hits();
    let closures_after_cold = origin.hits("/v1/tensorfs/closure");
    let presigns_after_cold = origin.hits("/v1/tensorfs/presign");

    // Warm: one closure question, no presign, no GET, nothing moved — and still PROVEN.
    let warm = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(warm.held, 8);
    assert_eq!(warm.fetched, 0);
    assert_eq!(warm.bytes_moved, 0);
    assert_eq!(origin.hits("/v1/tensorfs/closure") - closures_after_cold, 1);
    assert_eq!(origin.hits("/v1/tensorfs/presign") - presigns_after_cold, 0);
    assert_eq!(origin.total_hits(), calls_after_cold + 1);
    let _ = std::fs::remove_dir_all(root);
}

/// Tensorhub is a Go peer. `encoding/json` writes raw UTF-8, escapes `&` as `\u0026`, says
/// `null` for an absent value and may add fields; rows may repeat. A pull reads the fields it
/// consumes and nothing else stops it.
fn go_hub(
    checkpoint: Arc<Checkpoint>,
    closure: impl Fn(&Checkpoint) -> Serve + Send + Sync + 'static,
) -> Origin {
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let slot = Arc::clone(&base_slot);
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = slot.lock().unwrap().clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/tensorfs/closure") => closure(&checkpoint),
            ("POST", "/v1/tensorfs/presign") => ok_json(&presign_value(&request.body, &base)),
            ("GET", path) if path.starts_with("/obj/") => {
                match checkpoint.bodies.get(&path["/obj/".len()..]) {
                    Some(body) => status(200, body),
                    None => status(404, b"no such object"),
                }
            }
            _ => status(404, b"no such route"),
        }
    });
    let origin = origin(dispatch);
    *base_slot.lock().unwrap() = origin.base.clone();
    origin
}

fn go_row(object: &ObjectRef) -> String {
    format!(
        "{{\n      \"sha256\": \"{}\",\n      \"length\": {}\n    }}",
        object.sha256, object.length
    )
}

#[test]
fn a_go_hubs_ordinary_json_pulls() {
    let root = temporary("go-json");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(4, 1024));
    let origin = go_hub(Arc::clone(&checkpoint), |c| {
        let mut rows: Vec<String> = c.objects.iter().rev().map(go_row).collect();
        rows.push(go_row(&c.objects[0]));
        rows.push(go_row(&c.manifest));
        let body = format!(
            "{{\n  \"model\": \"acme/model\",\n  \"release\": \"r1\",\n  \"lane\": \"public\",\n  \
             \"scope\": \"runtime\",\n  \"note\": \"weights \\u0026 configs \u{2014} ready\",\n  \
             \"ratio\": 0.5,\n  \"retired\": null,\n  \"complete\": true,\n  \
             \"presign_max_digests\": 1956,\n  \
             \"manifest\": {},\n  \"objects\": [\n    {}\n  ]\n}}",
            go_row(&c.manifest),
            rows.join(",\n    ")
        );
        Serve::Whole {
            status: 200,
            body: body.into_bytes(),
            headers: vec![("Content-Type".into(), "application/json".into())],
        }
    });
    let policy = SourcePolicy::default();
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(report.declared, checkpoint.objects.len() as u64 + 1);
    assert_eq!(report.fetched, checkpoint.objects.len() as u64 + 1);
    let _ = std::fs::remove_dir_all(root);
}

/// Go's `json.MarshalIndent` envelope escapes `<` and writes an em dash raw. It is still the
/// hub's typed verdict, not weather to retry.
#[test]
fn a_go_hub_refusal_envelope_is_its_verdict() {
    let root = temporary("go-envelope");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(1, 1024));
    let origin = go_hub(Arc::clone(&checkpoint), |_| {
        Serve::Whole {
        status: 403,
        body: "{\n  \"error\": {\n    \"code\": \"model.private\",\n    \"message\": \"acme/model is private \u{2014} \\u003cowner only\\u003e\",\n    \"remedy\": \"present an org token\"\n  }\n}"
            .as_bytes()
            .to_vec(),
        headers: vec![("Content-Type".into(), "application/json".into())],
    }
    });
    let policy = SourcePolicy::default();
    let error = pull(&request_for(&store, &origin.base, &policy)).unwrap_err();
    assert_eq!(error.code, Code::CREDENTIAL_REQUIRED, "{error:?}");
    assert!(error.detail.contains("<owner only>"), "{}", error.detail);
    assert_eq!(
        origin.hits("/v1/tensorfs/closure"),
        1,
        "a verdict is asked once"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn runtime_only_pull_refuses_before_retaining_incomplete_snapshot() {
    let root = temporary("audit-runtime-sibling");
    let store = Store::init(&root).unwrap();
    let mut checkpoint = checkpoint(4, 16 * 1024);
    let original = checkpoint
        .bodies
        .remove(&checkpoint.manifest.sha256)
        .unwrap();
    let parsed = crate::manifest::Manifest::parse(&original).unwrap();
    let sibling = ObjectRef::of(b"unselected source snapshot readme");
    let manifest = Draft {
        entries: vec![
            ("README.md".into(), Entry::File(sibling.clone())),
            (
                "model.cozytensors".into(),
                Entry::CozyTensors(parsed.header().unwrap().clone()),
            ),
        ],
    }
    .seal()
    .unwrap();
    let bytes = manifest.canonical_bytes();
    checkpoint.manifest = ObjectRef::of(&bytes);
    checkpoint
        .bodies
        .insert(checkpoint.manifest.sha256.clone(), bytes);
    // The real Hub response declares scope=runtime and only selected tensor objects.
    let origin = hub_origin(Arc::new(checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let result = pull(&PullRequest::new(
        &store,
        &origin.base,
        "acme/model",
        Anonymous,
        &policy,
    ));
    // Runtime-only Hub closures and repository retention currently disagree about
    // ordinary siblings. Keep the refusal before recording a root that GC cannot read.
    assert_eq!(result.unwrap_err().code, Code::OBJECT_ABSENT);
    assert!(!store.contains(&sibling.sha256));
    let repo = crate::repository::RepositoryName::new("acme", "model").unwrap();
    assert!(!store.repository_path(&repo).exists());
    let _ = std::fs::remove_dir_all(root);
}

/// proto-061: a warm fetch of a held closure is one closure call. The plan answers from the
/// process's trust index, so no catalog is opened and nothing is re-walked.
#[test]
fn a_warm_fetch_opens_no_catalog() {
    let root = temporary("warm-held");
    let pod = root.join("pod");
    let checkpoint = Arc::new(checkpoint(200, 4096));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let store = Store::init(&pod).unwrap();
    let cold = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(cold.fetched, 202);

    let store = Store::open(&pod).unwrap();
    let opened = crate::catalog::connections_opened(&pod);
    let calls = origin.total_hits();
    let warm = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!((warm.held, warm.fetched), (202, 0));
    assert_eq!(origin.total_hits() - calls, 1, "the closure question only");
    assert_eq!(
        crate::catalog::connections_opened(&pod) - opened,
        0,
        "trust checks must not open the catalog per object"
    );
    let _ = std::fs::remove_dir_all(root);
}

/// A pull leaves a Store that can ANSWER for what it holds, and whose contents are not
/// garbage. The closure carries model/release/lane for the local index row; nothing wrote
/// it, so a pod that had just fetched a checkpoint refused `REPOSITORY_ABSENT` when it
/// resolved the release, and `gc_plan` — which seeds liveness from `repos/` alone — planned
/// the whole freshly-admitted closure for deletion.
#[test]
fn pull_records_the_release_it_resolved() {
    let root = temporary("pull-records");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(3, 8 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();

    let repo = crate::repository::RepositoryName::new("acme", "model").unwrap();
    let recorded = crate::repository::Repository::parse(
        &std::fs::read(store.repository_path(&repo)).expect("the pull recorded no repository"),
    )
    .unwrap();
    let release = recorded
        .releases
        .iter()
        .find(|release| release.version == report.release)
        .expect("the pull recorded no release");
    let lane = release
        .lanes
        .iter()
        .find(|lane| lane.lane == report.lane)
        .expect("the pull recorded no lane");
    assert_eq!(lane.manifest, checkpoint.manifest);
    assert!(recorded
        .checkpoints
        .iter()
        .any(|entry| entry.manifest == checkpoint.manifest));

    // THE SILENT HALF: liveness is computed from `repos/`, so an unrecorded pull leaves
    // every object it just admitted collectable.
    let census = crate::storage::Census::open(&root).unwrap();
    assert!(
        census.gc_plan(&[]).unwrap().is_empty(),
        "a recorded pull still planned its own closure for deletion"
    );

    // A warm re-pull records the same row again without conflicting on it.
    pull(&request_for(&store, &origin.base, &policy)).unwrap();
    let again =
        crate::repository::Repository::parse(&std::fs::read(store.repository_path(&repo)).unwrap())
            .unwrap();
    assert_eq!(again.checkpoints.len(), 1);
    assert_eq!(again.releases.len(), 1);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_download_stopped_with_every_object_here_is_recorded_by_the_next() {
    let root = temporary("canceled-custody");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(3, 8 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let token = super::PullCancellation::default();
    let on_progress = |done: u64, total: u64| {
        if total > 0 && done == total {
            token.cancel();
        }
    };
    let mut canceled = request_for(&store, &origin.base, &policy);
    canceled.cancellation = Some(token.clone());
    canceled.on_progress = Some(&on_progress);
    let _ = pull(&canceled);
    let (held, _) = FetchPlan::of(&store, "after-cancel", &checkpoint.manifest, &checkpoint.objects).unwrap();
    assert!(held.wanted.is_empty(), "the stop must retain every byte");

    // The next download moves nothing and records the source the stopped one may not have.
    let presigns = origin.hits("/v1/tensorfs/presign");
    let resumed = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!((resumed.fetched, resumed.bytes_held), (0, held.declared_bytes()));
    assert_eq!(origin.hits("/v1/tensorfs/presign"), presigns);
    let census = crate::storage::Census::open(&root).unwrap();
    assert!(census.gc_plan(&[]).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn corrupt_object_heals_into_wanted_and_refetches() {
    let root = temporary("heal");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(3, 8 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    pull(&request_for(&store, &origin.base, &policy)).unwrap();

    // Rot one blob on disk, same length.
    let victim = &checkpoint.objects[0];
    let path = store.blob_path(&victim.sha256);
    platform::durable::set_mode(&path, 0o644).unwrap();
    std::fs::write(&path, vec![b'x'; victim.length as usize]).unwrap();

    let healed = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(healed.fetched, 1);
    assert_eq!(healed.bytes_moved, victim.length);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn black_holed_stream_is_stalled_by_the_ledger_and_reasked() {
    let root = temporary("stall");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(2, 64 * 1024));
    let victim = checkpoint.objects[0].sha256.clone();
    let victim_for = victim.clone();
    let bodies = checkpoint.bodies.clone();
    // First ask: half the body, then silence forever. Second ask: everything.
    let origin = hub_origin(Arc::clone(&checkpoint), move |hex, attempt| {
        if hex == victim_for && attempt == 1 {
            let body = &bodies[hex];
            return Some(Serve::Stall {
                total: body.len() as u64,
                prefix: body[..body.len() / 2].to_vec(),
            });
        }
        None
    });
    let policy = SourcePolicy::default();
    let started = std::time::Instant::now();
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(report.fetched, 4);
    assert_eq!(origin.hits(&format!("/obj/{victim}")), 2);
    // The verdict came from the measured rule at test resolution (floor 6×25 ms), not
    // from any constant: well under the 30 s the origin was willing to hold the socket.
    assert!(started.elapsed() < Duration::from_secs(10));
    let _ = std::fs::remove_dir_all(root);
}

fn prebody_silence_is_judged(https: bool) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        // Accept HTTP request bytes or a TLS ClientHello, but never answer.
        let mut bytes = [0u8; 8192];
        while matches!(socket.read(&mut bytes), Ok(n) if n > 0) {}
    });
    let scheme = if https { "https" } else { "http" };
    let checked = local_policy()
        .check(&format!("{scheme}://127.0.0.1:{port}/object"))
        .unwrap();
    let ledger = Ledger::with_resolution(Duration::from_millis(25));
    ledger.moved(1024); // A pull that has already moved bytes, as on the real pod.
    let response = super::http::request(
        &super::http::Client::new(),
        &checked,
        "GET",
        &[],
        None,
        Deadline::after_seconds(Some(1.0)),
        &ledger,
        super::http::Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    );
    let refusal = match response {
        Err(refusal) => refusal,
        Ok(_) => panic!("silent origin unexpectedly answered"),
    };
    assert_eq!(refusal.code, Code::TRANSFER_FAILED, "{refusal}");
    assert!(
        refusal.detail.contains(if https {
            "TLS handshake"
        } else {
            "response head"
        }),
        "{refusal}"
    );
}

#[test]
fn an_object_stalled_before_headers_is_judged_by_stream_progress() {
    prebody_silence_is_judged(false);
}

#[test]
fn an_object_stalled_during_tls_is_judged_by_stream_progress() {
    prebody_silence_is_judged(true);
}

#[test]
fn a_prebody_stall_is_reasked_without_discarding_completed_objects() {
    let root = temporary("head-stall-retry");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(2, 64 * 1024));
    let victim = checkpoint.objects[0].sha256.clone();
    let selected = victim.clone();
    let origin = hub_origin(Arc::clone(&checkpoint), move |hex, attempt| {
        (hex == selected && attempt == 1).then_some(Serve::StallHead)
    });
    let policy = SourcePolicy::default();
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(report.fetched, 4);
    for object in &checkpoint.objects {
        assert!(store.record_valid(&object.sha256).is_ok());
        assert_eq!(
            origin.hits(&format!("/obj/{}", object.sha256)),
            if object.sha256 == victim { 2 } else { 1 }
        );
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn fragmented_headers_can_progress_across_multiple_stream_windows() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        // Each line its own segment: Nagle would hold them behind a delayed ACK (macOS).
        socket.set_nodelay(true).unwrap();
        read_request(&mut socket).unwrap();
        socket.write_all(b"HTTP/1.1 200 OK\r\n").unwrap();
        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(200));
            socket.write_all(b"X-Progress: yes\r\n").unwrap();
        }
        socket.write_all(b"Content-Length: 3\r\n\r\nabc").unwrap();
    });
    let checked = local_policy()
        .check(&format!("http://127.0.0.1:{port}/object"))
        .unwrap();
    // Lines 200 ms apart against a 600 ms floor: a loaded runner's late wakeups are not stalls.
    let ledger = Ledger::with_resolution(Duration::from_millis(100));
    let started = std::time::Instant::now();
    let response = super::http::request(
        &super::http::Client::new(),
        &checked,
        "GET",
        &[],
        None,
        Deadline::none(),
        &ledger,
        super::http::Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    )
    .unwrap();
    assert_eq!(response.read_capped(3).unwrap(), b"abc");
    assert!(started.elapsed() > ledger.floor() * 2);
}

/// A first address that never completes the TCP handshake (an IPv6 route that drops SYNs
/// while IPv4 works; here a listener whose accept queue is full, so the kernel drops every
/// SYN) is that address's verdict, not the URL's: the next checked address is asked a tick
/// later, not after the silent one's whole patience.
/// Only Linux drops the handshakes past a full accept queue in silence (macOS may still
/// complete them, Windows resets them).
#[cfg(target_os = "linux")]
#[test]
fn a_silent_first_address_gives_way_to_the_next() {
    let hole = TcpListener::bind("127.0.0.1:0").unwrap();
    rustix::net::listen(&hole, 0).unwrap();
    let filled: Vec<_> = (0..4)
        .filter_map(|_| {
            TcpStream::connect_timeout(&hole.local_addr().unwrap(), Duration::from_millis(100)).ok()
        })
        .collect();
    let live = TcpListener::bind("127.0.0.1:0").unwrap();
    let live_addr = live.local_addr().unwrap();
    std::thread::spawn(move || {
        let (mut socket, _) = live.accept().unwrap();
        read_request(&mut socket).unwrap();
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc")
            .unwrap();
    });
    let checked = policy::CheckedUrl {
        https: false,
        host: "127.0.0.1".into(),
        port: live_addr.port(),
        target: "/object".into(),
        addrs: vec![hole.local_addr().unwrap(), live_addr],
    };
    let ledger = Ledger::with_resolution(Duration::from_millis(250));
    let started = std::time::Instant::now();
    let response = super::http::request(
        &super::http::Client::new(),
        &checked,
        "GET",
        &[],
        None,
        Deadline::none(),
        &ledger,
        super::http::Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    )
    .unwrap();
    assert_eq!(response.read_capped(3).unwrap(), b"abc");
    let took = started.elapsed();
    assert!(took >= ledger.tick(), "the silent address was never tried");
    assert!(
        took < ledger.floor(),
        "the silent address held the live one {took:?}"
    );
    drop(filled);
}

#[test]
fn a_slow_origin_is_given_longer_each_ask_and_then_left_alone() {
    let root = temporary("slow");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(1, 6 * 1024));
    let slow = checkpoint.objects[0].sha256.clone();
    let slow_for = slow.clone();
    let bodies = checkpoint.bodies.clone();
    // A successful 400 ms first answer teaches this origin's latency; subsequent
    // 800 ms gaps exceed the initial floor but remain below that measured term.
    // Unlike the old body-only rule,
    // initial pre-header silence is now bounded, so this fixture gives the first
    // answer a sufficient 600 ms floor. Production timing constants are unchanged.
    let origin = hub_origin(Arc::clone(&checkpoint), move |hex, _| {
        if hex == slow_for {
            return Some(Serve::Drip {
                body: bodies[hex].clone(),
                chunk: 1024,
                gap: Duration::from_millis(800),
                ttfb: Duration::from_millis(400),
            });
        }
        None
    });
    let policy = SourcePolicy::default();
    let mut request = request_for(&store, &origin.base, &policy);
    request.sample_seconds = 0.1;
    let report = pull(&request).unwrap();
    assert_eq!(report.fetched, 3);
    // Its first byte takes 400 ms where its peers' took none, so it may be dropped once. Its
    // next ask has no peer still running to be slower than, and is left to the origin's pace.
    assert!(origin.hits(&format!("/obj/{slow}")) <= 2);
    let _ = std::fs::remove_dir_all(root);
}

/// A hub whose PRESIGN can be made to HOLD, and whose closure declares the mint window, so a
/// test can put a slow mint at an exact point in a walk. `hold(n)` is how long the hub keeps
/// presign ask number `n` before answering it — normally, with real URLs.
///
/// The hold is FINITE on purpose. A hub that never answers would prove only that something
/// eventually gave up; a hub that answers late lets one assertion say WHICH clock ended the
/// wait, because a refusal that arrives before the hub would have spoken cannot have been
/// caused by the hub speaking.
fn minting_hub(
    checkpoint: Arc<Checkpoint>,
    window: usize,
    hold: impl Fn(usize) -> Duration + Send + Sync + 'static,
    serve_object: impl Fn(&[u8]) -> Serve + Send + Sync + 'static,
) -> (Origin, Arc<AtomicUsize>) {
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let hits: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    let mints = Arc::new(AtomicUsize::new(0));
    let dispatch_base = Arc::clone(&base_slot);
    let minted = Arc::clone(&mints);
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = dispatch_base.lock().unwrap().clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/tensorfs/closure") => {
                let Value::Obj(mut fields) = closure_value(&checkpoint, &base) else {
                    panic!("the closure emulation is an object")
                };
                for (key, value) in &mut fields {
                    if key == "presign_max_digests" {
                        *value = Value::uint(window as u64);
                    }
                }
                ok_json(&Value::Obj(fields))
            }
            ("POST", "/v1/tensorfs/presign") => {
                let ask = minted.fetch_add(1, Ordering::Relaxed) + 1;
                std::thread::sleep(hold(ask));
                ok_json(&presign_value(&request.body, &base))
            }
            ("GET", path) if path.starts_with("/obj/") => {
                match checkpoint.bodies.get(&path["/obj/".len()..]) {
                    Some(body) => serve_object(body),
                    None => status(404, b"no such object"),
                }
            }
            _ => status(404, b"no such route"),
        }
    });
    let origin = origin_with_hits(dispatch, hits);
    *base_slot.lock().unwrap() = origin.base.clone();
    (origin, mints)
}

/// **THE CONTROL PLANE (tfs-106): a mint that goes silent while the transfer is standing
/// still faults, and says what it measured.**
///
/// The byte-plane arms above judge a stream that stopped sending. On 2026-09-08 the streams
/// were not sending and were not being judged, because they were not reading: fifteen of
/// them were parked on `Leases::minting` behind one thread inside `hub::presign`, and a hub
/// call answered to the caller's wall deadline alone — which a pod fetch deliberately does
/// not set. `gayong` and `weed` each moved ~82 GB, stopped dead, and billed fifteen minutes
/// at exactly zero with nothing anywhere raising anything.
///
/// The mint here is reached WITH BYTES ALREADY LANDED, which is the production shape and the
/// reason the refusal has a real figure to quote. The verdict must arrive well before the
/// hub would have answered: that is what proves it came from the measurement rather than
/// from the hub finally speaking.
#[test]
fn a_mint_that_goes_silent_on_a_still_pull_faults_with_what_it_measured() {
    // Far longer than any patience this pull can earn at 25 ms resolution, and the whole
    // assertion below is that the walk did not wait for it.
    const HOLD: Duration = Duration::from_secs(5);
    let root = temporary("still-mint");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(3, 8 * 1024));
    // One digest per mint, so the walk asks the hub again between objects and the second ask
    // is reached only after the first object's bytes are already in the store.
    let (origin, mints) = minting_hub(
        Arc::clone(&checkpoint),
        1,
        |ask| {
            if ask >= 2 {
                HOLD
            } else {
                Duration::ZERO
            }
        },
        |body| status(200, body),
    );
    let policy = SourcePolicy::default();
    let mut request = request_for(&store, &origin.base, &policy);
    request.streams = 1;
    request.sample_seconds = 0.025;
    // NO DEADLINE — `Deadline::none()` is `PullRequest`'s default and is exactly what the pod
    // supervisor passes, deliberately. Before tfs-106 this pull could not end.
    let started = std::time::Instant::now();
    let refusal = pull(&request).unwrap_err();
    let waited = started.elapsed();

    // Asked again once, then refused as stalled, quoting the mint's own verdict.
    assert_eq!(refusal.code, Code::STALLED, "{refusal}");
    assert!(refusal.detail.contains("nothing landed"), "{refusal}");
    assert!(
        refusal
            .detail
            .contains("the transfer this call serves has stopped"),
        "{refusal}"
    );
    assert!(
        refusal.detail.contains("s waiting for a response head"),
        "the verdict names the wait it was in: {refusal}"
    );
    // THE FIGURE IS REAL. A fault that could not say what had landed would leave the next
    // reader exactly where the stale progress line left them.
    let quoted = refusal
        .detail
        .split(" B landed in this pull")
        .next()
        .unwrap();
    let landed: u64 = quoted.rsplit(' ').next().unwrap().parse().unwrap();
    let whole: u64 =
        checkpoint.objects.iter().map(|o| o.length).sum::<u64>() + checkpoint.manifest.length;
    assert!(
        landed > 0,
        "the pull had moved bytes before it stopped: {refusal}"
    );
    assert!(
        landed < whole,
        "the pull had NOT finished when it stopped: {refusal}"
    );
    // WHICH CLOCK ENDED IT. The hub was still holding its answer when the walk refused.
    assert!(
        waited < HOLD,
        "the walk waited {waited:?} for a hub holding {HOLD:?} — that is the hub answering, \
         not the pull measuring"
    );
    assert!(mints.load(Ordering::Relaxed) >= 3, "the silent mint was not asked again");
    let _ = std::fs::remove_dir_all(root);
}

/// **The arm that proves this is the owner's rule and not a timeout: the SAME slow mint,
/// over a transfer that is still moving, is left completely alone.**
///
/// Every mint after the first is held far longer than this pull's own measured patience — at
/// 20 ms drip gaps the pull earns roughly 160 ms and the hub holds 1.2 s, seven times it. A
/// bound on how long a mint may take would kill this walk. Nothing here is killed, because
/// the question asked is not how long the call has taken; it is whether the transfer the
/// call serves has stopped. Bytes are landing on the other stream the whole time, so it has
/// not, and a slow hub over a working link is exactly what must not be condemned.
#[test]
fn a_slow_mint_over_a_moving_transfer_is_left_alone() {
    const HOLD: Duration = Duration::from_millis(1200);
    // On tmpfs: a write stalled by another process's disk pressure holds the reading
    // thread off its socket, and the control plane rightly counts that as stillness.
    // This test is about the hub, so the store must not depend on the box's disk.
    let shm = std::path::Path::new("/dev/shm");
    let root = if shm.is_dir() {
        shm.join(format!(
            "tensorfs-slow-mint-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ))
    } else {
        temporary("slow-mint")
    };
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(3, 40 * 1024));
    let (origin, mints) = minting_hub(
        Arc::clone(&checkpoint),
        1,
        |ask| if ask >= 2 { HOLD } else { Duration::ZERO },
        // EVERY object drips, in the same number of pieces whatever its size, so one stream
        // is always moving bytes while the other is waiting on the hub. The 20 ms gap is far
        // inside the pull's own measured pace; the hub's hold is far outside it.
        |body| Serve::Drip {
            body: body.to_vec(),
            chunk: (body.len() / 80).max(1),
            gap: Duration::from_millis(20),
            ttfb: Duration::ZERO,
        },
    );
    let policy = SourcePolicy::default();
    let mut request = request_for(&store, &origin.base, &policy);
    request.streams = 2;
    let started = std::time::Instant::now();
    let report = pull(&request).unwrap();
    let took = started.elapsed();

    assert_eq!(report.fetched, checkpoint.objects.len() as u64 + 1);
    // The slow mint really happened — otherwise this arm proves nothing about patience.
    assert!(
        took >= HOLD,
        "the walk finished in {took:?} without ever waiting out a {HOLD:?} mint"
    );
    assert!(
        mints.load(Ordering::Relaxed) >= 2,
        "only one mint was ever asked for"
    );
    for object in checkpoint.objects.iter() {
        assert!(
            origin.hits(&format!("/obj/{}", object.sha256)) <= 2,
            "object {} was re-asked",
            object.sha256
        );
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_stopped_download_keeps_what_landed_and_the_next_asks_for_the_rest() {
    let root = temporary("deadline");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(4, 8 * 1024));
    let bodies = checkpoint.bodies.clone();
    let origin = hub_origin(Arc::clone(&checkpoint), move |hex, _| {
        Some(Serve::Drip {
            body: bodies[hex].clone(),
            chunk: 8 * 1024,
            gap: Duration::ZERO,
            ttfb: Duration::from_millis(120),
        })
    });
    let policy = SourcePolicy::default();
    let stop = super::PullCancellation::default();
    let stopping = |done: u64, _: u64| {
        if done > 0 {
            stop.cancel();
        }
    };
    let mut rushed = request_for(&store, &origin.base, &policy);
    rushed.streams = 1;
    // A 120 ms first byte is pace, not a stall, even on a slow runner: the stop is the verdict.
    rushed.sample_seconds = 0.1;
    rushed.cancellation = Some(stop.clone());
    rushed.on_progress = Some(&stopping);
    let refusal = pull(&rushed).unwrap_err();
    assert_eq!(refusal.code, Code::TRANSFER_FAILED);

    // Whatever landed before the stop stays landed; the re-plan asks only for the rest.
    let resumed = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(resumed.held + resumed.fetched, 6);
    assert!(
        resumed.held > 0,
        "per-object resume kept the landed objects"
    );
    let _ = std::fs::remove_dir_all(root);
}

/// **The decisive one: a walk that outlives the URL lifetime and still completes.**
///
/// The hub caps how long a presigned URL lives. What that cap BOUNDS depends entirely on
/// when the URL is minted — and minting the whole wanted set before the walk starts spends
/// the cap on the closure rather than on an object, so the last object's URL is already dead
/// when a stream reaches it. The cap was never wrong; the moment of minting was.
///
/// One origin proves both halves, because the only difference between them is that moment.
/// Its URLs live one second and it refuses a URL presented later, exactly as an object store
/// refuses a signature that has aged out. Minted up front, the walk is refused partway
/// through. Minted where they are used, the same URLs — same cap, same closure, same origin
/// — carry a walk several times longer than one URL's life to completion.
///
/// (No production run has yet been observed failing this way. SDXL run 192 is NOT evidence
/// for it: that run died on `tensorfs.delegation_unauthorized`, its URLs fresh on every one
/// of ~2,400 retries because each retry re-ran closure and presign together.)
#[test]
fn a_walk_that_outlives_the_url_lifetime_completes() {
    let checkpoint = Arc::new(checkpoint(62, 4 * 1024)); // 62 blobs + header + manifest
    let hub = expiring_hub_origin(
        Arc::clone(&checkpoint),
        Duration::from_secs(2),
        Duration::from_millis(80),
        Duration::from_millis(30),
    );

    // The same closure, the same origin, the same one-second URLs — minted where they are
    // used. Nothing about the cap changed; only when the URL is asked for.
    let root = temporary("expiry-lazy");
    let store = Store::init(&root).unwrap();
    let policy = SourcePolicy::default();
    let mut request = request_for(&store, &hub.origin.base, &policy);
    request.streams = 4;
    request.sample_seconds = 0.2;
    let started = std::time::Instant::now();
    let report = pull(&request).unwrap();
    let walked = started.elapsed();

    assert_eq!(report.fetched, 64);
    assert_eq!(report.held, 0);
    assert!(
        walked > hub.lifetime,
        "the walk finished inside one URL's life ({walked:?}) — it never crossed the \
         boundary this test is about"
    );
    assert!(
        hub.origin.hits("/v1/tensorfs/presign") > 1,
        "one presign covered the whole walk — the URLs were still minted up front"
    );
    assert_eq!(
        hub.rejected.load(Ordering::Relaxed),
        0,
        "a dead URL was presented — every fetch must carry an authorization minted for it"
    );
    let _ = std::fs::remove_dir_all(root);
}

/// **A resume across an expiry boundary.** The second reason to mint at the point of use: a
/// single up-front mint stamps every unfetched object with the SAME expiry, so an
/// interruption longer than what is left of it poisons all of them at once. A URL minted at
/// use survives a resume by construction — the resumed walk mints its own, and never
/// inherits one the interrupted walk was holding.
#[test]
fn a_resume_across_an_expiry_boundary_completes() {
    let checkpoint = Arc::new(checkpoint(18, 4 * 1024)); // 18 blobs + header + manifest
    let hub = expiring_hub_origin(
        Arc::clone(&checkpoint),
        Duration::from_secs(1),
        Duration::from_millis(50),
        Duration::from_millis(15),
    );
    let root = temporary("expiry-resume");
    let store = Store::init(&root).unwrap();
    let policy = SourcePolicy::default();

    // Cut off only after a verified object has actually landed. A fixed wall
    // deadline can expire before the first admission on a busy test runner.
    let cancellation = crate::transport::PullCancellation::default();
    let observer = |done: u64, _: u64| {
        if done > 0 {
            cancellation.cancel();
        }
    };
    let mut rushed = request_for(&store, &hub.origin.base, &policy);
    rushed.streams = 2;
    rushed.sample_seconds = 0.2;
    rushed.cancellation = Some(cancellation.clone());
    rushed.on_progress = Some(&observer);
    let refusal = pull(&rushed).unwrap_err();
    assert_eq!(refusal.code, Code::TRANSFER_FAILED);
    assert!(refusal.detail.contains("cancelled"));
    let (landed, _) = FetchPlan::of(&store, "rushed", &checkpoint.manifest, &checkpoint.objects).unwrap();
    assert!(!landed.held.is_empty() && !landed.wanted.is_empty(), "{} held", landed.held.len());

    // Wait out the life of every URL that walk was holding, then resume.
    std::thread::sleep(hub.lifetime + Duration::from_millis(300));

    let mut resumed_request = request_for(&store, &hub.origin.base, &policy);
    resumed_request.streams = 2;
    resumed_request.sample_seconds = 0.2;
    let resumed = pull(&resumed_request).unwrap();

    assert_eq!(resumed.held + resumed.fetched, 20);
    assert!(
        resumed.held > 0,
        "per-object resume kept what the interrupted walk landed"
    );
    assert!(
        resumed.fetched > 0,
        "nothing was left to fetch — the walk never crossed the boundary"
    );
    assert_eq!(
        hub.rejected.load(Ordering::Relaxed),
        0,
        "the resume presented a URL minted before the interruption"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn wrong_bytes_are_refused_at_the_door_with_no_retry_and_nothing_installed() {
    let root = temporary("imposter");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(2, 4 * 1024));
    let victim = checkpoint.objects[0].clone();
    let victim_for = victim.sha256.clone();
    let origin = hub_origin(Arc::clone(&checkpoint), move |hex, _| {
        if hex == victim_for {
            return Some(status(200, &vec![b'z'; 4 * 1024]));
        }
        None
    });
    let policy = SourcePolicy::default();
    let refusal = pull(&request_for(&store, &origin.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::OBJECT_ID_MISMATCH);
    assert_eq!(origin.hits(&format!("/obj/{}", victim.sha256)), 1);
    assert!(!store.contains(&victim.sha256));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_connection_that_dies_midobject_costs_that_object_and_nothing_else() {
    let root = temporary("truncate");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(2, 32 * 1024));
    let victim = checkpoint.objects[0].sha256.clone();
    let victim_for = victim.clone();
    let bodies = checkpoint.bodies.clone();
    let origin = hub_origin(Arc::clone(&checkpoint), move |hex, attempt| {
        if hex == victim_for && attempt == 1 {
            let body = &bodies[hex];
            return Some(Serve::Truncate {
                total: body.len() as u64,
                prefix: body[..body.len() / 3].to_vec(),
            });
        }
        None
    });
    let policy = SourcePolicy::default();
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(report.fetched, 4);
    assert_eq!(origin.hits(&format!("/obj/{victim}")), 2);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn retryable_status_buys_a_reask_and_authorization_answers_do_not() {
    let root = temporary("retry");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(2, 4 * 1024));
    let flaky = checkpoint.objects[0].sha256.clone();
    let expired = checkpoint.objects[1].sha256.clone();
    let flaky_for = flaky.clone();
    let origin = hub_origin(Arc::clone(&checkpoint), move |hex, attempt| {
        if hex == flaky_for && attempt == 1 {
            return Some(status(503, b"try later"));
        }
        None
    });
    let policy = SourcePolicy::default();
    let request = request_for(&store, &origin.base, &policy);
    let report = pull(&request).unwrap();
    assert_eq!(report.fetched, 4);
    assert_eq!(origin.hits(&format!("/obj/{flaky}")), 2);

    // A 403 is asked again once on a URL minted afresh; the origin's refusal of that is final.
    let store2root = temporary("retry-expired");
    let store2 = Store::init(&store2root).unwrap();
    let expired_for = expired.clone();
    let origin2 = hub_origin(Arc::clone(&checkpoint), move |hex, _| {
        if hex == expired_for {
            return Some(status(403, b"presign expired"));
        }
        None
    });
    let refusal = pull(&request_for(&store2, &origin2.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::TRANSFER_FAILED);
    assert!(refusal.detail.contains("403 to a URL minted afresh"), "{refusal}");
    assert_eq!(origin2.hits(&format!("/obj/{expired}")), 2);
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(store2root);
}

/// Many requests in flight, few writers: network parallelism stays and disk writes are
/// bounded (`download::WRITERS`).
#[test]
fn a_wide_walk_writes_through_a_bounded_number_of_writers() {
    let root = temporary("writers");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(96, 512 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let mut request = request_for(&store, &origin.base, &policy);
    request.streams = 64;
    let report = pull(&request).unwrap();
    assert_eq!(report.fetched, 98);
    assert!(
        report.streams > crate::transport::download::WRITERS,
        "{report:?}"
    );
    assert!(
        (1..=crate::transport::download::WRITERS).contains(&report.writers),
        "{report:?}"
    );
    assert_eq!(pull(&request).unwrap().fetched, 0);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn sixteen_way_walk_lands_everything_through_the_one_door() {
    let root = temporary("nway");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(24, 16 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let plans = Mutex::new(Vec::new());
    let planned = |plan: &FetchPlan| {
        plans
            .lock()
            .unwrap()
            .push((plan.declared_bytes(), plan.held_bytes()));
    };
    let mut request = request_for(&store, &origin.base, &policy);
    request.streams = 16;
    request.on_plan = Some(&planned);
    let report = pull(&request).unwrap();
    assert_eq!(report.fetched, 26);
    assert_eq!(*plans.lock().unwrap(), vec![(report.bytes_total, 0)]);
    let warm = pull(&request).unwrap();
    assert_eq!(warm.fetched, 0);
    assert_eq!(
        *plans.lock().unwrap(),
        vec![
            (report.bytes_total, 0),
            (report.bytes_total, report.bytes_total)
        ]
    );
    let _ = std::fs::remove_dir_all(root);
}

// ---------------------------------------------------------------- the hub calls

#[test]
fn hub_envelope_crosses_verbatim_as_typed_refusals() {
    let store_root = temporary("envelope");
    let store = Store::init(&store_root).unwrap();
    let not_found: Dispatch = Arc::new(|request: &Request| {
        assert_eq!(request.method, "POST");
        status(
            404,
            br#"{"error":{"code":"models.unknown_ref","message":"no model x","remedy":"check the ref"}}"#,
        )
    });
    let origin = origin(not_found);
    let policy = SourcePolicy::default();
    let refusal = pull(&request_for(&store, &origin.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::REF_NOT_FOUND);
    assert!(refusal.detail.contains("models.unknown_ref"));
    assert!(refusal.detail.contains("check the ref"));

    let denied: Dispatch = Arc::new(|_: &Request| {
        status(
            401,
            br#"{"error":{"code":"auth.credential_required","message":"no credential"}}"#,
        )
    });
    let origin = self::origin(denied);
    let refusal = pull(&request_for(&store, &origin.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::CREDENTIAL_REQUIRED);
    let _ = std::fs::remove_dir_all(store_root);
}

// tfs-101. A status is evidence about the CATALOG only when the hub wrote it. On
// 2026-09-08 the public origin's tunnel agent was seconds out of a reconnect loop and its
// edge answered `POST /v1/tensorfs/presign` with its own 404 page; the status alone was
// read as REF_NOT_FOUND, which the pod supervisor treats as poisoned, and a paid H100
// journaled a permanent refusal for a placement set the hub was holding in full.
#[test]
fn a_status_with_no_hub_envelope_is_the_edge_and_stays_resumable() {
    let store_root = temporary("unenveloped");
    let store = Store::init(&store_root).unwrap();
    let policy = SourcePolicy::default();
    // Exactly what an ngrok endpoint answers while its agent session is down.
    let edge_page: Dispatch = Arc::new(|_: &Request| {
        status(
            404,
            b"<!DOCTYPE html><html><body>ERR_NGROK_3200 endpoint offline</body></html>",
        )
    });
    let origin = origin(edge_page);
    let refusal = pull(&request_for(&store, &origin.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::HUB_UNREACHABLE);
    assert!(refusal.detail.contains("no hub envelope"));
    assert!(refusal.detail.contains("404"));

    // A 401 page from a gateway is not this hub asking for a credential it does not take.
    let gateway: Dispatch = Arc::new(|_: &Request| status(401, b"<html>Unauthorized</html>"));
    let origin = self::origin(gateway);
    let refusal = pull(&request_for(&store, &origin.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::HUB_UNREACHABLE);

    // And a well-formed JSON body that is simply not the envelope stays weather too: the
    // test is the hub's own code, not the content type.
    let shaped: Dispatch = Arc::new(|_: &Request| status(404, br#"{"detail":"not found"}"#));
    let origin = self::origin(shaped);
    let refusal = pull(&request_for(&store, &origin.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::HUB_UNREACHABLE);
    let _ = std::fs::remove_dir_all(store_root);
}

#[test]
fn credential_headers_reach_the_hub_and_not_the_object_host() {
    let root = temporary("cred");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(1, 4 * 1024));
    let seen: Arc<Mutex<Vec<(String, bool)>>> = Arc::default();
    let seen_for = Arc::clone(&seen);
    let checkpoint_for = Arc::clone(&checkpoint);
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let dispatch_base = Arc::clone(&base_slot);
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = dispatch_base.lock().unwrap().clone();
        seen_for.lock().unwrap().push((
            request.path.clone(),
            request.header("authorization").is_some(),
        ));
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/tensorfs/closure") => ok_json(&closure_value(&checkpoint_for, &base)),
            ("POST", "/v1/tensorfs/presign") => ok_json(&presign_value(&request.body, &base)),
            ("GET", path) => match checkpoint_for.bodies.get(&path["/obj/".len()..]) {
                Some(body) => status(200, body),
                None => status(404, b"absent"),
            },
            _ => status(404, b"no route"),
        }
    });
    let origin = origin(dispatch);
    *base_slot.lock().unwrap() = origin.base.clone();
    let token = ScopedHeaders {
        hosts: vec!["127.0.0.1".into()],
        headers: vec![("authorization".into(), "Bearer worker-token".into())],
    };
    let policy = SourcePolicy::default();
    let mut request = PullRequest::new(&store, &origin.base, "acme/model", token, &policy);
    request.sample_seconds = 0.025;
    pull(&request).unwrap();
    let seen = seen.lock().unwrap();
    for (path, carried) in seen.iter() {
        // The token's provider scopes it to the host, and here hub and object host are
        // the same loopback — so it reaches both. The scoping proof is in the policy
        // tests; this asserts it reached the hub calls at all.
        if path.starts_with("/v1/") {
            assert!(carried, "{path} lost the credential");
        }
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_worker_capability_reaches_the_hub_calls_as_its_two_headers() {
    let root = temporary("worker-cred");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(1, 4 * 1024));
    let checkpoint_for = Arc::clone(&checkpoint);
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let dispatch_base = Arc::clone(&base_slot);
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = dispatch_base.lock().unwrap().clone();
        let machine = request.header("x-cozy-worker-id") == Some("om-machine")
            && request.header("x-cozy-worker-token") == Some("worker-token");
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", _) if !machine => status(
                401,
                br#"{"error":{"code":"worker.unauthorized","message":"m"}}"#,
            ),
            ("POST", "/v1/tensorfs/closure") => ok_json(&closure_value(&checkpoint_for, &base)),
            ("POST", "/v1/tensorfs/presign") => ok_json(&presign_value(&request.body, &base)),
            ("GET", path) => match checkpoint_for.bodies.get(&path["/obj/".len()..]) {
                Some(body) => status(200, body),
                None => status(404, b"absent"),
            },
            _ => status(404, b"no route"),
        }
    });
    let origin = origin(dispatch);
    *base_slot.lock().unwrap() = origin.base.clone();
    let policy = SourcePolicy::default();
    let credential =
        credential_from_spec("worker om-machine worker-token", vec!["127.0.0.1".into()]).unwrap();
    let mut request = PullRequest::new(&store, &origin.base, "acme/model", credential, &policy);
    request.sample_seconds = 0.025;
    pull(&request).unwrap();
    let anonymous_root = temporary("worker-cred-anon");
    let anonymous = Store::init(&anonymous_root).unwrap();
    let mut refused = PullRequest::new(&anonymous, &origin.base, "acme/model", Anonymous, &policy);
    refused.sample_seconds = 0.025;
    assert!(pull(&refused)
        .unwrap_err()
        .detail
        .contains("worker.unauthorized"));
    for spec in ["worker om-machine", "worker a b c", "machine a b"] {
        let refusal = credential_from_spec(spec, vec![]).err().unwrap();
        assert_eq!(refusal.code, Code::CREDENTIAL_REQUIRED, "{spec}");
    }
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(anonymous_root);
}

// ---------------------------------------------------------------- the predicate

#[test]
fn the_address_predicate_names_every_blocked_class() {
    for (address, expected) in [
        ("10.1.2.3", "private"),
        ("192.168.1.1", "private"),
        ("100.64.0.9", "private"),
        ("127.0.0.1", "loopback"),
        ("169.254.169.254", "link-local"),
        ("224.0.0.1", "multicast"),
        ("240.0.0.1", "reserved"),
        ("0.0.0.0", "unspecified"),
        ("::1", "loopback"),
        ("fe80::1", "link-local"),
        ("fc00::1", "private"),
        ("ff02::1", "multicast"),
        ("::", "unspecified"),
        ("100::1", "reserved"), // discard-only space: outside 2000::/3, fail-closed
    ] {
        let why = blocked(address.parse().unwrap());
        assert!(
            why.contains(expected),
            "{address}: expected {expected:?} in {why:?}"
        );
    }
    // The v4-mapped spelling of the metadata service is the metadata service.
    assert!(blocked("::ffff:169.254.169.254".parse().unwrap()).contains("link-local"));
    for public in ["93.184.216.34", "1.1.1.1", "2606:4700::1111"] {
        assert_eq!(blocked(public.parse().unwrap()), "", "{public}");
    }
}

#[test]
fn the_policy_fails_closed() {
    let policy = SourcePolicy {
        allowed_hosts: vec!["bucket.example.com".into(), ".cdn.example.net".into()],
        allow_local: false,
        max_redirects: 0,
        ..Default::default()
    };
    // Host outside the declaration.
    let refused = policy.check("https://evil.example.org/x").unwrap_err();
    assert_eq!(refused.code, Code::SOURCE_NOT_ALLOWED);
    // A dot entry admits subdomains only at a label boundary; a bare entry only itself.
    assert!(policy.allows_host("a.cdn.example.net"));
    assert!(!policy.allows_host("cdn.example.net"));
    assert!(!policy.allows_host("evilcdn.example.net"));
    assert!(policy.allows_host("bucket.example.com"));
    assert!(!policy.allows_host("sub.bucket.example.com"));
    // A `*` entry is a host name like any other; `any_host` admits every host, and every
    // address is still checked.
    let star = SourcePolicy {
        allowed_hosts: vec!["*".into()],
        ..Default::default()
    };
    assert!(!star.allows_host("evil.example.org"));
    let any = SourcePolicy {
        any_host: true,
        ..Default::default()
    };
    assert!(any.allows_host("evil.example.org"));
    let refused = any.check("https://127.0.0.1/x").unwrap_err();
    assert_eq!(refused.code, Code::SOURCE_NOT_ALLOWED);
    // Plaintext http without a declared-local deployment.
    assert_eq!(
        policy
            .check("http://bucket.example.com/x")
            .unwrap_err()
            .code,
        Code::SOURCE_NOT_ALLOWED
    );
    // Userinfo and fragments are not one clean origin.
    for url in [
        "https://user:pw@bucket.example.com/x",
        "https://bucket.example.com/x#frag",
        "ftp://bucket.example.com/x",
        "https:///nohost",
    ] {
        assert_eq!(
            policy.check(url).unwrap_err().code,
            Code::SOURCE_NOT_ALLOWED,
            "{url}"
        );
    }
    // A host resolving to a blocked class refuses even when allowlisted.
    let localhost_allowed = SourcePolicy {
        allowed_hosts: vec!["localhost".into(), "169.254.169.254".into()],
        allow_local: false,
        max_redirects: 0,
        ..Default::default()
    };
    let refused = localhost_allowed.check("https://localhost/x").unwrap_err();
    assert_eq!(refused.code, Code::SOURCE_NOT_ALLOWED);
    assert!(refused.detail.contains("loopback"));
    let refused = localhost_allowed
        .check("https://169.254.169.254/latest/meta-data/")
        .unwrap_err();
    assert!(refused.detail.contains("link-local"));
    // An empty allowlist is no egress at all.
    assert_eq!(
        SourcePolicy::default()
            .check("https://anything.example.com/")
            .unwrap_err()
            .code,
        Code::SOURCE_NOT_ALLOWED
    );
}

#[test]
fn a_fake_ip_answer_is_admitted_only_for_a_named_https_origin() {
    let policy = SourcePolicy {
        allowed_hosts: vec![".ngrok.app".into(), "198.18.0.5".into()],
        allow_local: false,
        max_redirects: 0,
        ..Default::default()
    };
    let answer = |ips: &[&str], port: u16| -> Vec<std::net::SocketAddr> {
        ips.iter().map(|ip| std::net::SocketAddr::new(ip.parse().unwrap(), port)).collect()
    };
    // Fake-IP DNS answers every name in 198.18/15; TLS then verifies the certificate for the name.
    policy.resolved.seed("hub.ngrok.app", 443, answer(&["198.18.0.148"], 443));
    let checked = policy.check("https://hub.ngrok.app/healthz").unwrap();
    assert_eq!(checked.addrs, answer(&["198.18.0.148"], 443));
    policy.resolved.seed("top.ngrok.app", 443, answer(&["198.19.255.254"], 443));
    assert!(policy.check("https://top.ngrok.app/x").is_ok());
    // Every other blocked answer still refuses, whatever rides beside the fake one.
    for (name, ips, why) in [
        ("meta.ngrok.app", ["198.18.0.9", "169.254.169.254"], "link-local"),
        ("lan.ngrok.app", ["198.18.0.9", "10.0.0.7"], "private"),
        ("loop.ngrok.app", ["198.18.0.9", "127.0.0.1"], "loopback"),
    ] {
        policy.resolved.seed(name, 443, answer(&ips, 443));
        let refused = policy.check(&format!("https://{name}/x")).unwrap_err();
        assert_eq!(refused.code, Code::SOURCE_NOT_ALLOWED, "{name}");
        assert!(refused.detail.contains(why), "{name}: {}", refused.detail);
    }
    // Plaintext stays strict, and so does a literal 198.18 address (no name to verify).
    policy.resolved.seed("hub.ngrok.app", 80, answer(&["198.18.0.148"], 80));
    assert_eq!(policy.check("http://hub.ngrok.app/x").unwrap_err().code, Code::SOURCE_NOT_ALLOWED);
    let refused = policy.check("https://198.18.0.5/x").unwrap_err();
    assert!(refused.detail.contains("private"), "{}", refused.detail);
}

#[test]
fn redirects_are_refused_or_revalidated_never_followed_blind() {
    let root = temporary("redirect");
    let store = Store::init(&root).unwrap();
    let body = b"redirected".to_vec();
    let object = ObjectRef::of(&body);
    let redirect: Dispatch = Arc::new(|_: &Request| Serve::Whole {
        status: 302,
        body: Vec::new(),
        headers: vec![("location".into(), "https://evil.example.org/steal".into())],
    });
    let origin = origin(redirect);
    let (plan, _) =
        FetchPlan::of_objects(&store, "redirect", std::slice::from_ref(&object)).unwrap();
    let grant = DeliveryGrant::mint(&plan, &object).unwrap();
    let url = format!("{}/obj/{}", origin.base, object.sha256);
    let ledger = Ledger::with_resolution(Duration::from_millis(25));

    // Budget 0: a presigned GET has no business redirecting.
    let refusal = fetch_ranged(
        &store,
        &grant,
        &url,
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        Ranged::default(),
        None,
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::REDIRECT_REFUSED);

    // Budget 1: the hop faces the whole predicate, and the target host is outside the
    // declaration — the PRECISE refusal, exactly as egress.py validated the hop it then
    // refused anyway.
    let mut hopping = local_policy();
    hopping.max_redirects = 1;
    let ledger = Ledger::with_resolution(Duration::from_millis(25));
    let refusal = fetch_ranged(
        &store,
        &grant,
        &url,
        &hopping,
        &Anonymous,
        Deadline::none(),
        &ledger,
        Ranged::default(),
        None,
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::SOURCE_NOT_ALLOWED);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_relative_location_is_resolved_against_the_hop_that_sent_it() {
    // HuggingFace answers `resolve/<commit>/<path>` for a member with NO LFS pointer -- a
    // sharded repo's `model.safetensors.index.json` -- with a same-origin 307 whose
    // Location is `/api/resolve-cache/...`, while every LFS member gets an ABSOLUTE
    // redirect onto a delivery host. Refusing the relative form therefore refused exactly
    // the index members of every sharded repo, on every pod, forever, while their shards
    // succeeded: MiniMax-H3 moved all 210.3 GB of its 44 shards and then sat at 44 of 48
    // for hours on four files totalling 249 KB.
    let root = temporary("relative-redirect");
    let store = Store::init(&root).unwrap();
    let body = b"the member behind a relative redirect".to_vec();
    let object = ObjectRef::of(&body);
    let served = body.clone();
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        if request.path.starts_with("/api/resolve-cache/") {
            return Serve::Whole {
                status: 200,
                body: served.clone(),
                headers: Vec::new(),
            };
        }
        // Relative, and carrying a query -- the shape HuggingFace actually sends.
        Serve::Whole {
            status: 307,
            body: Vec::new(),
            headers: vec![(
                "location".into(),
                "/api/resolve-cache/models/org/repo/deadbeef/member.json?etag=%22abc%22".into(),
            )],
        }
    });
    let origin = origin(dispatch);
    let (plan, _) =
        FetchPlan::of_objects(&store, "relative", std::slice::from_ref(&object)).unwrap();
    let grant = DeliveryGrant::mint(&plan, &object).unwrap();
    let url = format!("{}/org/repo/resolve/deadbeef/member.json", origin.base);
    let ledger = Ledger::with_resolution(Duration::from_millis(25));
    let mut policy = local_policy();
    policy.max_redirects = 5;

    fetch_ranged(
        &store,
        &grant,
        &url,
        &policy,
        &Anonymous,
        Deadline::none(),
        &ledger,
        Ranged::default(),
        None,
    )
    .unwrap();

    // The object is in the store, admitted through the same digest door as every other
    // byte: the redirect changed which socket served it and nothing about what was accepted.
    assert!(store.contains(&object.sha256));
    assert_eq!(origin.hits("/org/repo/resolve/deadbeef/member.json"), 1);
    assert_eq!(
        origin.hits("/api/resolve-cache/models/org/repo/deadbeef/member.json"),
        1
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn resolving_a_relative_location_cannot_leave_the_origin_the_predicate_admitted() {
    // The safety argument for following a relative reference, asserted rather than
    // reasoned: a relative Location resolves against the hop just walked, so it names the
    // SAME origin the fence already admitted. Only an ABSOLUTE Location can name another
    // host, and that one still faces the whole predicate on the next hop -- which is what
    // `redirects_are_refused_or_revalidated_never_followed_blind` pins.
    let base = policy::CheckedUrl {
        https: true,
        host: "huggingface.co".into(),
        port: 443,
        target: "/org/repo/resolve/deadbeef/dir/member.json?v=1".into(),
        addrs: Vec::new(),
    };
    let cases = [
        (
            "/api/resolve-cache/x?e=1",
            "https://huggingface.co/api/resolve-cache/x?e=1",
        ),
        (
            "sibling.json",
            "https://huggingface.co/org/repo/resolve/deadbeef/dir/sibling.json",
        ),
        (
            "https://cdn.example.org/blob",
            "https://cdn.example.org/blob",
        ),
        ("//cdn.example.org/blob", "https://cdn.example.org/blob"),
    ];
    for (location, want) in cases {
        assert_eq!(
            absolute_location(&base, location).as_deref(),
            Some(want),
            "resolving {location:?}"
        );
    }
    // Nothing usable is not silently turned into the base URL: an empty Location is a
    // refusal, not a re-fetch of the same target, which would be a hot loop.
    assert_eq!(absolute_location(&base, "   "), None);

    // A non-default port is carried, so a relative hop cannot be redirected to :443.
    let odd = policy::CheckedUrl {
        port: 8443,
        ..base.clone()
    };
    assert_eq!(
        absolute_location(&odd, "/x").as_deref(),
        Some("https://huggingface.co:8443/x")
    );
}

// ---------------------------------------------------------------- the push half

// ---------------------------------------------------------------- TLS

#[test]
fn the_tls_pump_carries_a_real_exchange() {
    let key = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let cert_der = rustls::pki_types::CertificateDer::from(key.cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::try_from(key.signing_key.serialize_der())
        .unwrap()
        .clone_key();
    let server_config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::aws_lc_rs::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(vec![cert_der.clone()], key_der)
    .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let payload = b"tls bytes, verified".to_vec();
    let served = payload.clone();
    let server_config = Arc::new(server_config);
    std::thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        let conn = rustls::ServerConnection::new(server_config).unwrap();
        let mut stream = rustls::StreamOwned::new(conn, socket);
        let mut sink = [0u8; 4096];
        let mut collected = Vec::new();
        loop {
            let n = stream.read(&mut sink).unwrap();
            collected.extend_from_slice(&sink[..n]);
            if collected.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            served.len()
        );
        stream.write_all(head.as_bytes()).unwrap();
        stream.write_all(&served).unwrap();
        stream.conn.send_close_notify();
        let _ = stream.flush();
    });

    let client = super::http::Client::with_extra_roots(&[cert_der]);
    let policy = SourcePolicy {
        allowed_hosts: vec!["localhost".into()],
        allow_local: true,
        max_redirects: 0,
        ..Default::default()
    };
    let checked = policy
        .check(&format!("https://localhost:{port}/object"))
        .unwrap();
    let ledger = Ledger::with_resolution(Duration::from_millis(25));
    let response = super::http::request(
        &client,
        &checked,
        "GET",
        &[],
        None,
        Deadline::none(),
        &ledger,
        super::http::Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    )
    .unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(response.read_capped(1 << 20).unwrap(), payload);
}

/// Run 5438: one handshake of a long upload failed certificate verification from a host that
/// had verified many times in the same process. A certificate verdict from an origin this
/// process verified is weather, asked again after the transfer's measured patience; once the
/// transfer moves nothing for longer than its measured stillness, the verdict stands.
#[test]
fn a_certificate_refused_by_an_origin_verified_before_is_asked_again() {
    let identity = || {
        let key = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let cert = rustls::pki_types::CertificateDer::from(key.cert.der().to_vec());
        let private = rustls::pki_types::PrivateKeyDer::try_from(key.signing_key.serialize_der())
            .unwrap()
            .clone_key();
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![cert.clone()], private)
        .unwrap();
        (cert, Arc::new(config))
    };
    let (trusted, good) = identity();
    let (_, stranger) = identity();
    // Connection 1 presents a certificate nobody trusts, between two trusted ones; every one
    // from connection 3 on presents it again.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let accepted = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&accepted);
    std::thread::spawn(move || {
        for (index, socket) in listener.incoming().enumerate() {
            let Ok(socket) = socket else { return };
            counted.fetch_add(1, Ordering::SeqCst);
            let config = if index == 0 || index == 2 {
                &good
            } else {
                &stranger
            };
            let conn = rustls::ServerConnection::new(Arc::clone(config)).unwrap();
            let mut stream = rustls::StreamOwned::new(conn, socket);
            let mut head = Vec::new();
            let mut chunk = [0u8; 4096];
            while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                match stream.read(&mut chunk) {
                    Ok(n) if n > 0 => head.extend_from_slice(&chunk[..n]),
                    _ => break,
                }
            }
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}");
            stream.conn.send_close_notify();
            let _ = stream.flush();
        }
    });
    let client = super::http::Client::with_extra_roots(&[trusted]);
    let policy = SourcePolicy {
        allowed_hosts: vec!["localhost".into()],
        allow_local: true,
        max_redirects: 0,
        ..Default::default()
    };
    let base = format!("https://localhost:{port}");
    let ask = |ledger: &Ledger| {
        super::hub::call_method(
            &client,
            "GET",
            &base,
            "/v1/ping",
            b"",
            &Anonymous,
            &policy,
            Deadline::none(),
            ledger,
        )
    };
    let ledger = Ledger::with_resolution(Duration::from_millis(10));
    ask(&ledger).unwrap();
    ask(&ledger).expect("a verified origin's certificate verdict is asked again");
    assert_eq!(accepted.load(Ordering::SeqCst), 3);

    // A transfer that moves nothing while it re-asks keeps the verdict.
    let still = Ledger::with_resolution(Duration::from_millis(10));
    let refused = ask(&still).unwrap_err();
    assert_eq!(refused.code, Code::TLS_UNTRUSTED, "{}", refused.detail);
    assert!(
        refused.detail.contains("verified localhost before"),
        "{}",
        refused.detail
    );
    assert_eq!(accepted.load(Ordering::SeqCst), 5);
}

// ---------------------------------------------------------------- the parity benchmark

/// Serves one object's body by digest.
type Bodies = Arc<dyn Fn(&str) -> Vec<u8> + Send + Sync>;

/// Distinct `size`-byte objects sharing one pseudo-random block, and a way to serve each.
fn paced_objects(size: usize, count: usize) -> (Vec<ObjectRef>, Bodies) {
    let mut block = vec![0u8; size];
    let mut x: u64 = 0x9e3779b97f4a7c15;
    for byte in block.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *byte = (x >> 24) as u8;
    }
    let block = Arc::new(block);
    let body_of = {
        let block = Arc::clone(&block);
        move |index: u64| -> Vec<u8> {
            let mut body = block.as_ref().clone();
            body[..8].copy_from_slice(&index.to_le_bytes());
            body
        }
    };
    let mut declared = Vec::new();
    let mut index_of = HashMap::new();
    for index in 0..count as u64 {
        let object = ObjectRef::of(&body_of(index));
        index_of.insert(object.sha256.clone(), index);
        declared.push(object);
    }
    declared.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    let serve = move |hex: &str| body_of(index_of[hex]);
    (declared, Arc::new(serve))
}

// ------------------------------------------------- proto-037: the wire does not grow

/// What one emulated hub was ASKED, so a test can assert about the requests and not only
/// about the outcome.
struct Wire {
    origin: Origin,
    /// The largest presign body this hub was asked to read.
    largest_presign: Arc<AtomicUsize>,
    /// How many presign calls it answered.
    mints: Arc<AtomicUsize>,
    /// How many closure pages it answered.
    pages: Arc<AtomicUsize>,
}

/// A hub that behaves like tensorhub does: it PAGINATES its closure, DECLARES the presign
/// ceiling it derived from its own body cap, and REFUSES an oversized body rather than
/// truncating it.
///
/// `ceiling` of `None` is a hub from before the ceiling was declared, which a client refuses.
fn wire_hub_origin(
    checkpoint: Arc<Checkpoint>,
    request_cap: usize,
    page: usize,
    ceiling: Option<usize>,
) -> Wire {
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let hits: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    let largest_presign = Arc::new(AtomicUsize::new(0));
    let mints = Arc::new(AtomicUsize::new(0));
    let pages = Arc::new(AtomicUsize::new(0));
    let dispatch_base = Arc::clone(&base_slot);
    let (seen, minted, paged) = (
        Arc::clone(&largest_presign),
        Arc::clone(&mints),
        Arc::clone(&pages),
    );
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = dispatch_base.lock().unwrap().clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/tensorfs/closure") => {
                paged.fetch_add(1, Ordering::Relaxed);
                let after = match crate::canon::parse(&request.body, 1 << 20) {
                    Ok(Value::Obj(fields)) => fields
                        .into_iter()
                        .find(|(k, _)| k == "after")
                        .and_then(|(_, v)| match v {
                            Value::Str(hex) => Some(hex),
                            _ => None,
                        })
                        .unwrap_or_default(),
                    _ => String::new(),
                };
                let rows: Vec<&ObjectRef> = checkpoint
                    .objects
                    .iter()
                    .filter(|o| o.sha256 > after)
                    .collect();
                let complete = rows.len() <= page;
                let taken = if complete { &rows[..] } else { &rows[..page] };
                let row = |o: &ObjectRef| {
                    Value::obj(vec![
                        ("length", Value::uint(o.length)),
                        ("sha256", Value::str(o.sha256.clone())),
                    ])
                };
                let mut fields = vec![
                    ("complete", Value::Bool(complete)),
                    ("lane", Value::str("public")),
                    ("manifest", row(&checkpoint.manifest)),
                    ("model", Value::str("acme/model")),
                    (
                        "objects",
                        Value::arr(taken.iter().map(|o| row(o)).collect()),
                    ),
                    ("release", Value::str("r1")),
                    ("scope", Value::str("runtime")),
                ];
                if let Some(ceiling) = ceiling {
                    fields.push(("presign_max_digests", Value::uint(ceiling as u64)));
                }
                ok_json(&Value::obj(fields))
            }
            ("POST", "/v1/tensorfs/presign") => {
                minted.fetch_add(1, Ordering::Relaxed);
                seen.fetch_max(request.body.len(), Ordering::Relaxed);
                // Exactly what tensorhub does: the size check fires before the decoder, so
                // an oversized body is a size refusal and never a shape one.
                if request.body.len() > request_cap {
                    return status(
                        413,
                        b"{\"error\":{\"code\":\"tensorfs.request_too_large\",\"message\":\"over the cap\"}}",
                    );
                }
                ok_json(&presign_value(&request.body, &base))
            }
            ("GET", path) if path.starts_with("/obj/") => {
                match checkpoint.bodies.get(&path["/obj/".len()..]) {
                    Some(body) => status(200, body),
                    None => status(404, b"no such object"),
                }
            }
            _ => status(404, b"no such route"),
        }
    });
    let origin = origin_with_hits(dispatch, hits);
    *base_slot.lock().unwrap() = origin.base.clone();
    Wire {
        origin,
        largest_presign,
        mints,
        pages,
    }
}

/// **THE DECISIVE ONE: no presign body the walk sends can exceed what the hub will read.**
///
/// The mint window was bounded by bytes to TRANSFER (8 GiB declared) and by digest COUNT
/// (4,096) and by nothing else, so a closure of several thousand small objects fell entirely
/// inside one window and serialised into a body far larger than the hub accepts. Captured on
/// the wire against the live hub, 2026-09-03: `paul/wai-illustrious@17.0.0`, 2,598 objects
/// and ~6.9 GB — under the byte budget — asked for 2,599 digests in a 174,146-byte body at a
/// route that reads 131,072, sixteen times, once per stream.
///
/// Neither existing bound could ever have caught it, which is the point. A digest weighs 67
/// bytes on the wire, so 4,096 of them weigh 274,445: the count cap sat at more than twice
/// the size the hub will read and was unreachable through this route. The missing bound was
/// the request's own size — and it is not a bound this side may hold a copy of, because the
/// number belongs to the hub. The hub declares it; the window is it.
#[test]
fn no_presign_body_exceeds_what_the_hub_will_read() {
    // tensorhub `internal/workerdownloads/service.go`, verbatim.
    const HUB_REQUEST_CAP: usize = 128 << 10;
    // What that cap buys, by the hub's own derivation: (131072 - 13) / 67.
    const HUB_CEILING: usize = 1956;
    // Production's own object count. It MUST exceed the ceiling or the test proves nothing:
    // a closure that fits in one window fits under any bound at all.
    const OBJECTS: usize = 2598;
    // Checked at COMPILE time, not at run time: shrinking the closure under the ceiling
    // would make every assertion below pass vacuously, and that must fail the build rather
    // than pass quietly. At 1,601 objects the old code passes this test.
    const _: () = assert!(OBJECTS > HUB_CEILING);

    let root = temporary("wire-window");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(OBJECTS - 2, 1024)); // blobs + header + manifest
    assert!(
        checkpoint.objects.len() > HUB_CEILING,
        "the fixture collapsed to {} distinct objects",
        checkpoint.objects.len()
    );
    let hub = wire_hub_origin(
        Arc::clone(&checkpoint),
        HUB_REQUEST_CAP,
        usize::MAX,
        Some(HUB_CEILING),
    );
    let policy = SourcePolicy::default();
    let mut request = request_for(&store, &hub.origin.base, &policy);
    request.streams = 4;
    let report = pull(&request).unwrap();

    assert_eq!(report.fetched, checkpoint.objects.len() as u64 + 1);
    let largest = hub.largest_presign.load(Ordering::Relaxed);
    assert!(
        largest <= HUB_REQUEST_CAP,
        "the walk sent a {largest} B presign body at a hub that reads {HUB_REQUEST_CAP} B"
    );
    // And the window is the hub's ceiling, not something smaller that happens to fit: a
    // full window weighs 13 + 67 * 1956 = 131,065 B, seven bytes inside the cap.
    assert_eq!(
        largest,
        13 + 67 * HUB_CEILING,
        "the window is not the ceiling the hub declared"
    );
    // Several windows, which is what a closure larger than one window must cost.
    assert!(hub.mints.load(Ordering::Relaxed) >= 2);
    let _ = std::fs::remove_dir_all(root);
}

/// **A closure larger than one page is walked page by page, and the pages name one tree.**
#[test]
fn a_closure_is_paginated_and_every_page_names_one_tree() {
    let root = temporary("wire-pages");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(40, 1024));
    let hub = wire_hub_origin(Arc::clone(&checkpoint), 128 << 10, 7, Some(1956));
    let policy = SourcePolicy::default();
    let report = pull(&request_for(&store, &hub.origin.base, &policy)).unwrap();
    // Every object arrived, across the seams, with nothing repeated or skipped.
    assert_eq!(report.fetched, checkpoint.objects.len() as u64 + 1);
    assert_eq!(report.declared, checkpoint.objects.len() as u64 + 1);
    let pages = hub.pages.load(Ordering::Relaxed);
    assert!(
        pages >= checkpoint.objects.len().div_ceil(7),
        "{pages} pages for {} objects at 7 per page",
        checkpoint.objects.len()
    );
    let _ = std::fs::remove_dir_all(root);
}

/// **A hub that declares no ceiling predates the protocol and is refused by name.**
#[test]
fn a_hub_that_declares_no_ceiling_is_refused() {
    let root = temporary("wire-noceiling");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(30, 1024));
    let hub = wire_hub_origin(Arc::clone(&checkpoint), 128 << 10, usize::MAX, None);
    let policy = SourcePolicy::default();
    let refusal = pull(&request_for(&store, &hub.origin.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::HUB_REFUSED);
    assert!(
        refusal.detail.contains("presign_max_digests"),
        "{}",
        refusal.detail
    );
    assert!(
        refusal.detail.contains("update Tensorhub"),
        "{}",
        refusal.detail
    );
    assert_eq!(hub.mints.load(Ordering::Relaxed), 0);
    let _ = std::fs::remove_dir_all(root);
}

/// **A SLOW MINT MUST NOT POISON ITS OWN ANSWER.**
///
/// tfs-102, from two paid H100s on 2026-09-08. `gayong` and `weed` each moved ~82 GB of
/// `paul/minimax-h3` at 378 MB/s, then stopped dead and refused with "a URL whose declared
/// life is already spent when it is granted authorizes nothing". Nothing was wrong with the
/// objects — the same closure landed clean on two other pods within the hour.
///
/// `minted` was stamped BEFORE the presign call, and a lease is spent at half the life the
/// hub declared. The hub's life is ten minutes, so a presign that took five — one flaky
/// tunnel, or two of `call`'s own retries — returned URLs that were unusable the instant
/// they arrived, and the walk refused terminally with 82 GB already on disk.
///
/// The URL does not exist for the duration of the call that fetches it. The hub signs it
/// while handling the request and states the life it granted in its own clock; the request
/// leg, the queueing and the signing are time the URL had not begun. Stamping on ARRIVAL
/// charges it only for the return leg, which is one hop.
///
/// Reaching the decision needs a mint slower than half a declared life, so the fixture
/// shortens the LIFE rather than lengthening the call. Everything else is production.
#[test]
fn a_slow_mint_does_not_poison_the_urls_it_returns() {
    let root = temporary("slow-mint");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(3, 4 * 1024));
    // The shortest life the wire can express is one second — the hub states its clocks in
    // whole seconds — so the call need only take past half of it. Stamped before the call,
    // every URL in the answer is over half its life on arrival and the walk has none it
    // may use; stamped on arrival, every one is whole.
    const DECLARED: u64 = 1;
    let slow = Duration::from_millis(700);
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let dispatch_base = Arc::clone(&base_slot);
    let mints = Arc::new(AtomicUsize::new(0));
    let minted = Arc::clone(&mints);
    let served = Arc::clone(&checkpoint);
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = dispatch_base.lock().unwrap().clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/tensorfs/closure") => ok_json(&closure_value(&served, &base)),
            ("POST", "/v1/tensorfs/presign") => {
                minted.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(slow);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let Value::Obj(mut fields) = presign_value(&request.body, &base) else {
                    panic!()
                };
                for (key, value) in &mut fields {
                    match key.as_str() {
                        "expires_at_unix" => *value = Value::uint(now + DECLARED),
                        "server_time_unix" => *value = Value::uint(now),
                        _ => {}
                    }
                }
                ok_json(&Value::Obj(fields))
            }
            ("GET", path) if path.starts_with("/obj/") => {
                match served.bodies.get(&path["/obj/".len()..]) {
                    Some(body) => status(200, body),
                    None => status(404, b"no such object"),
                }
            }
            _ => status(404, b"no such route"),
        }
    });
    let origin = origin(dispatch);
    *base_slot.lock().unwrap() = origin.base.clone();
    let policy = SourcePolicy::default();
    let report = pull(&request_for(&store, &origin.base, &policy))
        .expect("a mint slower than half a declared life still returns usable URLs");
    assert_eq!(report.fetched, checkpoint.objects.len() as u64 + 1);
    assert!(mints.load(Ordering::Relaxed) >= 1);
    let _ = std::fs::remove_dir_all(root);
}

/// **A hub that says "wait" is waited for, and the same request is then answered.**
///
/// A rate limit is the purest resumable refusal in the protocol — the identical request gets
/// a different answer once the window slides — and it used to abort the whole pull through
/// `HUB_REFUSED` after however many gigabytes had landed. The wait is the hub's own stated
/// instant, read off the pair of clocks the hub sent, so a skewed local clock cannot shorten
/// or lengthen it.
#[test]
fn a_hub_that_says_wait_is_waited_for_and_then_answers() {
    let root = temporary("wire-429");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(4, 1024));
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let refusals = Arc::new(AtomicUsize::new(0));
    let dispatch_base = Arc::clone(&base_slot);
    let refused = Arc::clone(&refusals);
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = dispatch_base.lock().unwrap().clone();
        match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/tensorfs/closure") => ok_json(&closure_value(&checkpoint, &base)),
            ("POST", "/v1/tensorfs/presign") => {
                // The first ask is rate limited, and says WHEN — one second on the hub's
                // own clock, stated beside that clock.
                if refused.fetch_add(1, Ordering::Relaxed) == 0 {
                    let now = 1_800_000_000u64;
                    return Serve::Whole {
                        status: 429,
                        body: crate::canon::write(&Value::obj(vec![
                            (
                                "error",
                                Value::obj(vec![
                                    ("code", Value::str("tensorfs.rate_limited")),
                                    ("message", Value::str("the budget is spent")),
                                ]),
                            ),
                            ("retry_after_unix", Value::uint(now + 1)),
                            ("server_time_unix", Value::uint(now)),
                        ])),
                        headers: vec![("Content-Type".into(), "application/json".into())],
                    };
                }
                ok_json(&presign_value(&request.body, &base))
            }
            ("GET", path) if path.starts_with("/obj/") => {
                match checkpoint.bodies.get(&path["/obj/".len()..]) {
                    Some(body) => status(200, body),
                    None => status(404, b"no such object"),
                }
            }
            _ => status(404, b"no such route"),
        }
    });
    let origin = origin_with_hits(dispatch, Arc::default());
    *base_slot.lock().unwrap() = origin.base.clone();

    let policy = SourcePolicy::default();
    let started = std::time::Instant::now();
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    let waited = started.elapsed();

    assert_eq!(report.fetched, 6);
    assert!(
        refusals.load(Ordering::Relaxed) >= 2,
        "the ask was not repeated"
    );
    // The hub asked for a second, and a second is what it got — not a number this side chose.
    assert!(
        waited >= Duration::from_secs(1),
        "the pull did not wait the instant the hub named: {waited:?}"
    );
    let _ = std::fs::remove_dir_all(root);
}

// ------------------------------------------------- one source object (tfs-062, tfs-127)
//
// `fetch_ranged` is the pull's downloader for one object from a URL the caller holds.
// Every origin here is the same loopback listener the rest of this file uses, one thread
// per connection. Nothing is mocked: the client, the fence, the ledger and the admission
// door are production's.

/// A body whose every byte depends on its offset, so an object assembled at the wrong
/// offsets cannot accidentally hash to the right id.
fn ranged_body(bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    let mut state: u32 = 0x9E37_79B9;
    for (at, slot) in out.iter_mut().enumerate() {
        state = state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223 ^ at as u32);
        *slot = (state >> 24) as u8;
    }
    out
}

/// Temp files the store has not cleaned up: the `put-` names its reaper covers.
fn temps(root: &std::path::Path) -> usize {
    std::fs::read_dir(root.join("tmp"))
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("put-"))
        .count()
}

fn range_of(request: &Request) -> Option<(usize, usize)> {
    let spec = request.header("range")?.trim().strip_prefix("bytes=")?;
    let (first, last) = spec.split_once('-')?;
    Some((first.trim().parse().ok()?, last.trim().parse().ok()?))
}

/// The 206 an origin that honours ranges answers with, or the whole body when nothing was
/// asked for.
fn partial(body: &[u8], request: &Request) -> Serve {
    let Some((first, last)) = range_of(request) else {
        return Serve::Whole {
            status: 200,
            body: body.to_vec(),
            headers: Vec::new(),
        };
    };
    let last = last.min(body.len() - 1);
    Serve::Whole {
        status: 206,
        body: body[first..=last].to_vec(),
        headers: vec![(
            "content-range".into(),
            format!("bytes {first}-{last}/{}", body.len()),
        )],
    }
}

/// The byte each `Range` ask ENDS at, deduplicated: a range asked again after a failure
/// ends where its first ask did, so this names the chunks without counting the weather.
fn part_ends(asked: &[String]) -> Vec<u64> {
    let mut ends: Vec<u64> = asked
        .iter()
        .filter_map(|value| {
            value
                .strip_prefix("bytes=")?
                .split_once('-')?
                .1
                .parse()
                .ok()
        })
        .collect();
    ends.sort_unstable();
    ends.dedup();
    ends
}

/// A real Store, one object it wants, and a loopback origin that logs every `Range` asked.
struct Source {
    root: PathBuf,
    store: Store,
    object: ObjectRef,
    grant: DeliveryGrant,
    url: String,
    asked: Arc<Mutex<Vec<String>>>,
}

impl Source {
    fn new(
        name: &str,
        body: &[u8],
        serve: impl Fn(&Request) -> Serve + Send + Sync + 'static,
    ) -> Source {
        let root = temporary(name);
        let store = Store::init(&root).unwrap();
        let object = ObjectRef::of(body);
        let asked: Arc<Mutex<Vec<String>>> = Arc::default();
        let log = Arc::clone(&asked);
        let listener = origin(Arc::new(move |request: &Request| {
            log.lock()
                .unwrap()
                .push(request.header("range").unwrap_or("").to_string());
            serve(request)
        }));
        let (plan, _) = FetchPlan::of_objects(&store, name, std::slice::from_ref(&object)).unwrap();
        let grant = DeliveryGrant::mint(&plan, &object).unwrap();
        Source {
            root,
            store,
            object,
            grant,
            url: format!("{}/member.safetensors", listener.base),
            asked,
        }
    }

    /// `streams` requests of 1 MiB. A whole second of resolution unless the proof is about
    /// the pull's own patience: 25 ms on a loaded box buys asks the origin never earned.
    fn fetch(&self, streams: usize, policy: &SourcePolicy) -> crate::err::Result<Fetched> {
        self.fetch_with(
            streams,
            policy,
            &Ledger::with_resolution(Duration::from_secs(1)),
        )
    }

    fn fetch_with(
        &self,
        streams: usize,
        policy: &SourcePolicy,
        ledger: &Ledger,
    ) -> crate::err::Result<Fetched> {
        let ranged = Ranged {
            streams,
            part: 1 << 20,
        };
        fetch_ranged(
            &self.store,
            &self.grant,
            &self.url,
            policy,
            &Anonymous,
            Deadline::none(),
            ledger,
            ranged,
            None,
        )
    }

    fn asked(&self) -> Vec<String> {
        self.asked.lock().unwrap().clone()
    }

    fn held(&self) -> bool {
        self.store.record_valid(&self.object.sha256).is_ok()
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Six chunks on four connections: the first goes alone, its 206 queues the rest, and the
/// object is committed at the id its bytes hash to.
#[test]
fn a_source_object_crosses_in_parallel_byte_ranges() {
    let body = ranged_body(6 << 20);
    let served = body.clone();
    let source = Source::new("ranged-parallel", &body, move |request| {
        partial(&served, request)
    });
    let mut totals: Vec<u64> = Vec::new();
    let fetched = fetch_ranged(
        &source.store,
        &source.grant,
        &source.url,
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &Ledger::with_resolution(Duration::from_secs(1)),
        Ranged {
            streams: 4,
            part: 1 << 20,
        },
        Some(&mut |moved: u64| totals.push(moved)),
    )
    .unwrap();

    assert_eq!(fetched.object, source.object);
    assert_eq!(fetched.transferred, source.object.length);
    assert!(source.held());
    let asked = source.asked();
    assert_eq!(asked[0], "bytes=0-1048575", "the first chunk goes alone");
    assert_eq!(
        part_ends(&asked),
        vec![1048575, 2097151, 3145727, 4194303, 5242879, 6291455],
        "{asked:?}"
    );
    // Progress counts bytes on this machine, whatever order the chunks land in.
    assert!(
        totals.windows(2).all(|pair| pair[0] <= pair[1]),
        "moved went backwards: {totals:?}"
    );
    assert_eq!(totals.last().copied(), Some(source.object.length));
    assert_eq!(temps(&source.root), 0);
}

/// A source object no larger than one request is still asked for by range, as every source
/// fetch before 0.3.87 was: an origin that answers only ranges serves it.
#[test]
fn a_small_source_object_is_asked_for_by_range() {
    let body = ranged_body(300 << 10);
    let served = body.clone();
    let source = Source::new("ranged-small", &body, move |request| {
        match range_of(request) {
            Some(_) => partial(&served, request),
            None => status(400, b"a range is required"),
        }
    });
    let fetched = source.fetch(4, &local_policy()).unwrap();

    assert_eq!(fetched.transferred, source.object.length);
    assert!(source.held());
    assert_eq!(source.asked(), vec![format!("bytes=0-{}", body.len() - 1)]);
}

/// A source that will not range must still work: the first chunk's ask is answered with
/// the whole object, and that one request is the object.
#[test]
fn an_origin_that_will_not_range_still_delivers_the_object() {
    let body = ranged_body(6 << 20);
    let served = body.clone();
    let source = Source::new("ranged-refused", &body, move |_| Serve::Whole {
        status: 200,
        body: served.clone(),
        headers: Vec::new(),
    });
    let fetched = source.fetch(4, &local_policy()).unwrap();

    assert_eq!(fetched.transferred, source.object.length);
    assert!(source.held());
    assert_eq!(source.asked(), vec!["bytes=0-1048575".to_string()]);
}

/// The same origin, and its whole answer dies half way. The ask for the rest is answered
/// with the whole object again, which is taken from its first byte.
#[test]
fn a_whole_answer_that_dies_starts_the_object_over() {
    let body = ranged_body(6 << 20);
    let (served, cuts) = (body.clone(), Arc::new(AtomicUsize::new(0)));
    let source = Source::new("ranged-refused-cut", &body, move |_| {
        if cuts.fetch_add(1, Ordering::Relaxed) == 0 {
            return Serve::Truncate {
                total: served.len() as u64,
                prefix: served[..3 << 20].to_vec(),
            };
        }
        Serve::Whole {
            status: 200,
            body: served.clone(),
            headers: Vec::new(),
        }
    });
    let fetched = source.fetch(4, &local_policy()).unwrap();

    assert!(source.held());
    assert_eq!(
        source.asked(),
        vec![
            "bytes=0-1048575".to_string(),
            "bytes=3145728-6291455".to_string()
        ]
    );
    // The half that arrived twice was moved twice, and says so.
    assert_eq!(fetched.transferred, source.object.length + (3 << 20));
    assert_eq!(temps(&source.root), 0);
}

/// A connection that dies half way through a chunk is asked again for the half it is
/// missing, and moves nothing it already holds.
#[test]
fn a_request_that_dies_mid_range_resumes_at_the_byte_it_reached() {
    let body = ranged_body(4 << 20);
    let (served, cuts) = (body.clone(), Arc::new(AtomicUsize::new(0)));
    let source = Source::new("ranged-resume", &body, move |request| {
        let Some((first, last)) = range_of(request) else {
            return partial(&served, request);
        };
        // The first ask for the second chunk dies with half its bytes delivered.
        if first == (1 << 20) && cuts.fetch_add(1, Ordering::Relaxed) == 0 {
            let half = (last - first).div_ceil(2);
            return Serve::Short {
                status: 206,
                headers: vec![(
                    "content-range".into(),
                    format!("bytes {first}-{last}/{}", served.len()),
                )],
                declared: (last - first + 1) as u64,
                prefix: served[first..first + half].to_vec(),
            };
        }
        partial(&served, request)
    });
    let fetched = source.fetch(2, &local_policy()).unwrap();

    assert!(source.held());
    let asked = source.asked();
    assert!(
        asked.contains(&"bytes=1572864-2097151".to_string()),
        "expected a resumed range, saw {asked:?}"
    );
    assert_eq!(fetched.transferred, source.object.length);
}

/// Run 2322's shape at a source: one range crawls for its first ask while its peers come
/// home at once. The crawl is dropped and asked again; the fetch does not wait it out.
#[test]
fn a_source_request_far_slower_than_its_peers_is_restarted() {
    const RATE: usize = 16 << 10;
    let body = ranged_body(12 << 20);
    let (served, crawled) = (body.clone(), AtomicUsize::new(0));
    let source = Source::new("ranged-slow", &body, move |request| {
        let Some((first, last)) = range_of(request) else {
            return partial(&served, request);
        };
        if first != (8 << 20) || crawled.fetch_add(1, Ordering::Relaxed) != 0 {
            return partial(&served, request);
        }
        let head = format!(
            "HTTP/1.1 206 X\r\ncontent-length: {}\r\ncontent-range: bytes {first}-{last}/{}\r\n\
             connection: close\r\n\r\n",
            last + 1 - first,
            served.len()
        );
        let slow = served[first..=last].to_vec();
        Serve::Script(Box::new(move |socket| {
            if socket.write_all(head.as_bytes()).is_err() {
                return;
            }
            for piece in slow.chunks(4096) {
                if socket.write_all(piece).is_err() {
                    return;
                }
                std::thread::sleep(Duration::from_secs_f64(4096.0 / RATE as f64));
            }
        }))
    });
    let started = std::time::Instant::now();
    let ledger = Ledger::with_resolution(Duration::from_millis(25));
    let fetched = source.fetch_with(4, &local_policy(), &ledger).unwrap();
    let took = started.elapsed();

    assert!(source.held());
    assert!(fetched.transferred >= source.object.length);
    println!("a 12 MiB object with one crawling MiB: {took:?}");
    // Alone, the crawling megabyte takes 64 s.
    let crawl = Duration::from_secs_f64((1 << 20) as f64 / RATE as f64);
    assert!(took < crawl / 4, "the fetch waited out the crawl: {took:?}");
    let log = std::fs::read_to_string(source.root.join("logs/transport.log")).unwrap();
    assert!(log.contains(" restart "), "{log}");
}

/// An origin behind one link that every answer shares: its first `burst` bytes unpaced, then
/// `rate` B/s, answering at most `answering` requests at once while the rest wait their turn
/// before any header, as on a pod whose link or origin turned slow for every request alike.
/// The range at `crawl` is sent at 16 KiB/s on its first ask, beside the link.
fn shared_link(
    name: &str,
    body: &[u8],
    (rate, burst): (f64, usize),
    answering: usize,
    crawl: Option<usize>,
) -> Source {
    let link = Arc::new(Mutex::new((std::time::Instant::now(), 0)));
    let turns = Arc::new((Mutex::new(answering), std::sync::Condvar::new()));
    let (served, crawled) = (body.to_vec(), Arc::new(AtomicUsize::new(0)));
    Source::new(name, body, move |request| {
        let Some((first, last)) = range_of(request) else {
            return partial(&served, request);
        };
        let crawling = crawl == Some(first) && crawled.fetch_add(1, Ordering::Relaxed) == 0;
        let head = format!(
            "HTTP/1.1 206 X\r\ncontent-length: {}\r\ncontent-range: bytes {first}-{last}/{}\r\n\
             connection: close\r\n\r\n",
            last + 1 - first,
            served.len()
        );
        let range = served[first..=last].to_vec();
        let (link, turns) = (Arc::clone(&link), Arc::clone(&turns));
        Serve::Script(Box::new(move |socket| {
            let (free, freed) = &*turns;
            let mut left = freed
                .wait_while(free.lock().unwrap(), |left| *left == 0)
                .unwrap();
            *left -= 1;
            drop(left);
            let mut sent = socket.write_all(head.as_bytes());
            for block in range.chunks(16 << 10) {
                let time = |rate: f64| Duration::from_secs_f64(block.len() as f64 / rate);
                if crawling {
                    sent = sent.and_then(|()| socket.write_all(block));
                    std::thread::sleep(time((16 << 10) as f64));
                } else {
                    let due = {
                        let (next, carried) = &mut *link.lock().unwrap();
                        *carried += block.len();
                        if *carried > burst {
                            *next = (*next).max(std::time::Instant::now()) + time(rate);
                        }
                        *next
                    };
                    std::thread::sleep(due.saturating_duration_since(std::time::Instant::now()));
                    sent = sent.and_then(|()| socket.write_all(block));
                }
                if sent.is_err() {
                    break;
                }
            }
            *free.lock().unwrap() += 1;
            freed.notify_one();
        }))
    })
}

fn transport_log(source: &Source) -> String {
    std::fs::read_to_string(source.root.join("logs/transport.log")).unwrap_or_default()
}

/// The first range comes home at once, then the link turns slow for every range alike:
/// twelve share it and nineteen wait for a turn. None is slower than the requests beside
/// it, so none is restarted (vast 5090s, 2026-10-10: 885 restarts in 42 s, every range
/// judged against the few early asks that came home).
#[test]
fn a_pod_whose_every_range_is_slow_restarts_none() {
    let body = ranged_body(32 << 20);
    let link = ((12 << 20) as f64, 1 << 20);
    let source = shared_link("uniform-slow", &body, link, 12, None);
    let fetched = source.fetch(32, &local_policy()).unwrap();

    assert!(source.held());
    assert_eq!(fetched.transferred, source.object.length);
    let log = transport_log(&source);
    assert!(!log.contains(" restart "), "{log}");
    assert_eq!(source.asked().len(), 32, "{log}");
}

/// One range crawls while its peers move at the link's shared pace: that range, and only
/// it, is restarted, and the object lands long before the crawl would have.
#[test]
fn the_one_range_far_slower_than_the_ranges_beside_it_is_restarted() {
    let body = ranged_body(16 << 20);
    let link = ((8 << 20) as f64, 0);
    let source = shared_link("one-crawls", &body, link, 16, Some(8 << 20));
    let started = std::time::Instant::now();
    source.fetch(16, &local_policy()).unwrap();
    let took = started.elapsed();

    assert!(source.held());
    let log = transport_log(&source);
    let restarts: Vec<&str> = log
        .lines()
        .filter(|line| line.contains(" restart "))
        .collect();
    assert_eq!(restarts.len(), 1, "{log}");
    assert!(restarts[0].contains(" bytes=8388608-9437183 "), "{log}");
    // Alone, the crawling megabyte takes 64 s.
    assert!(
        took < Duration::from_secs(16),
        "the fetch waited out the crawl: {took:?}"
    );
}

/// A name the resolver does not answer is weather like any failed request: it is asked
/// again (a pod's Docker resolver answered some of 512 lookups only after 4-8 s, and the
/// first 0.3.85 walks ended on it), and the fetch ends on its own patience, quoting it.
#[test]
fn a_name_that_does_not_resolve_is_asked_again_until_the_pull_stops() {
    let root = temporary("unresolved");
    let store = Store::init(&root).unwrap();
    let object = ObjectRef::of(&ranged_body(1 << 20));
    let (plan, _) =
        FetchPlan::of_objects(&store, "unresolved", std::slice::from_ref(&object)).unwrap();
    let grant = DeliveryGrant::mint(&plan, &object).unwrap();
    let policy = SourcePolicy {
        allowed_hosts: vec!["tensorfs-test.invalid".into()],
        allow_local: true,
        ..Default::default()
    };
    let ledger = Ledger::with_resolution(Duration::from_millis(25));
    let started = std::time::Instant::now();
    let refusal = fetch_ranged(
        &store,
        &grant,
        "http://tensorfs-test.invalid/member.safetensors",
        &policy,
        &Anonymous,
        Deadline::none(),
        &ledger,
        Ranged::default(),
        None,
    )
    .unwrap_err();

    assert_eq!(refusal.code, Code::TRANSFER_FAILED, "{refusal}");
    assert!(refusal.detail.contains("nothing landed"), "{refusal}");
    assert!(
        refusal.detail.contains("name resolution failed"),
        "{refusal}"
    );
    assert!(
        started.elapsed() >= ledger.floor(),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(temps(&root), 0);
    let _ = std::fs::remove_dir_all(root);
}

/// The door does not soften because the bytes arrived in pieces: one flipped byte and the
/// object is refused at the digest, with nothing left behind.
#[test]
fn ranged_bytes_that_hash_wrong_are_refused_at_the_same_door() {
    let body = ranged_body(6 << 20);
    let mut corrupt = body.clone();
    corrupt[3 << 20] ^= 0xFF;
    let source = Source::new("ranged-corrupt", &body, move |request| {
        partial(&corrupt, request)
    });
    let refusal = source.fetch(4, &local_policy()).unwrap_err();

    assert_eq!(refusal.code, Code::OBJECT_ID_MISMATCH);
    assert!(!source.held());
    assert_eq!(temps(&source.root), 0, "a temp file was left behind");
}

/// An origin whose object is another length than the plan declares is refused on its own
/// `Content-Range`, before its bytes are taken.
#[test]
fn an_object_of_another_length_is_refused() {
    let body = ranged_body(6 << 20);
    let longer = ranged_body(7 << 20);
    let source = Source::new("ranged-length", &body, move |request| {
        partial(&longer, request)
    });
    let refusal = source.fetch(4, &local_policy()).unwrap_err();

    assert_eq!(refusal.code, Code::LENGTH_MISMATCH, "{refusal}");
    assert!(!source.held());
    assert_eq!(temps(&source.root), 0);
}

/// A chunk that never lands means the object never lands: the fetch ends on its own
/// patience, quotes the origin, and admits nothing.
#[test]
fn a_partly_downloaded_object_is_never_admitted() {
    let body = ranged_body(6 << 20);
    let served = body.clone();
    let source = Source::new("ranged-partial", &body, move |request| {
        match range_of(request) {
            // The fourth chunk is answered 503 forever; every other one lands.
            Some((first, _)) if first == (3 << 20) => status(503, b"busy"),
            _ => partial(&served, request),
        }
    });
    let ledger = Ledger::with_resolution(Duration::from_millis(100));
    let refusal = source.fetch_with(4, &local_policy(), &ledger).unwrap_err();

    assert_eq!(refusal.code, Code::TRANSFER_FAILED);
    assert!(refusal.detail.contains("503"), "{}", refusal.detail);
    assert!(!source.held());
    assert_eq!(temps(&source.root), 0, "a temp file was left behind");
}

/// The fence is per REQUEST, not per object: a chunk redirected off the declared hosts is
/// refused by the predicate a whole-object GET would have met.
#[test]
fn a_ranged_request_may_not_reach_a_host_the_predicate_never_saw() {
    let body = ranged_body(6 << 20);
    let served = body.clone();
    let source = Source::new("ranged-fence", &body, move |request| {
        match range_of(request) {
            // The first chunk is answered honestly; the others are sent somewhere else.
            Some((0, _)) => partial(&served, request),
            _ => Serve::Whole {
                status: 302,
                body: Vec::new(),
                headers: vec![("location".into(), "https://evil.example.org/steal".into())],
            },
        }
    });
    let mut policy = local_policy();
    policy.max_redirects = 5;
    let refusal = source.fetch(4, &policy).unwrap_err();

    assert_eq!(refusal.code, Code::SOURCE_NOT_ALLOWED);
    assert!(
        refusal.detail.contains("evil.example.org"),
        "{}",
        refusal.detail
    );
    assert!(!source.held());
}

/// A scoped credential is consulted PER REQUEST against the host that request reaches.
/// Both arms: scoped to the host the chunks reach, every one carries it; scoped to any
/// other host, none does.
#[test]
fn every_request_re_asks_the_credential_for_the_host_it_reaches() {
    for (scoped_to, expected) in [
        ("127.0.0.1", Some("Bearer hf_secret".to_string())),
        ("cdn.elsewhere.example", None),
    ] {
        let body = ranged_body(4 << 20);
        let served = body.clone();
        let bearers: Arc<Mutex<Vec<Option<String>>>> = Arc::default();
        let seen = Arc::clone(&bearers);
        let source = Source::new("ranged-credential", &body, move |request| {
            seen.lock()
                .unwrap()
                .push(request.header("authorization").map(str::to_string));
            partial(&served, request)
        });
        let credential = ScopedHeaders {
            hosts: vec![scoped_to.into()],
            headers: vec![("authorization".into(), "Bearer hf_secret".into())],
        };
        fetch_ranged(
            &source.store,
            &source.grant,
            &source.url,
            &local_policy(),
            &credential,
            Deadline::none(),
            &Ledger::with_resolution(Duration::from_secs(1)),
            Ranged {
                streams: 2,
                part: 1 << 20,
            },
            None,
        )
        .unwrap();

        let bearers = bearers.lock().unwrap();
        assert!(bearers.len() >= 4, "saw only {} requests", bearers.len());
        assert!(
            bearers.iter().all(|value| *value == expected),
            "scoped to {scoped_to}, the requests carried {bearers:?}"
        );
    }
}

// ------------------------------------------------- the disk budget (tfs-064)
//
// Every expected byte figure below is a BANKED LITERAL measured from the fixture, never
// the expression under test evaluated twice. th-152 shipped an arm that computed its
// expected value from the constants it was checking — it asserted the code equalled itself
// and passed review twice (proto-038, decisions.md 698). `the_fixture_is_the_size_it_was
// _banked_at` is what keeps these literals honest: change the fixture and it says so.

/// checkpoint(6, 32 * 1024): six 32,768-byte blobs, one header, one manifest.
const SIX_BLOB_CLOSURE_BYTES: u64 = 197_312;
/// checkpoint(3, 32 * 1024): the first three of those same six blobs, its own header and
/// manifest. `checkpoint` derives each body from its index alone, so the blobs a smaller
/// closure carries ARE the larger one's first blobs, byte for byte and digest for digest.
const THREE_BLOB_CLOSURE_BYTES: u64 = 98_783;
/// What a store already holding the three-blob closure must still buy for the six-blob one:
/// three new blobs, a new header, a new manifest. NOT the difference of the two totals —
/// the headers and manifests do not cancel — so this is measured too.
const SIX_AFTER_THREE_WANTED_BYTES: u64 = 99_008;

fn closure_bytes(checkpoint: &Checkpoint) -> u64 {
    checkpoint.bodies.values().map(|b| b.len() as u64).sum()
}

#[test]
fn the_fixture_is_the_size_it_was_banked_at() {
    assert_eq!(
        closure_bytes(&checkpoint(6, 32 * 1024)),
        SIX_BLOB_CLOSURE_BYTES
    );
    assert_eq!(
        closure_bytes(&checkpoint(3, 32 * 1024)),
        THREE_BLOB_CLOSURE_BYTES
    );
    // The sharing this file's dedup arm depends on is asserted, not assumed: an arm over a
    // fixture with no shared objects proves nothing about dedup at all.
    let small = checkpoint(3, 32 * 1024);
    let large = checkpoint(6, 32 * 1024);
    let shared = small
        .objects
        .iter()
        .filter(|object| {
            large
                .objects
                .iter()
                .any(|other| other.sha256 == object.sha256)
        })
        .count();
    assert_eq!(
        shared, 3,
        "the two fixtures share no objects; the dedup arm would be vacuous"
    );
}

#[test]
fn a_budget_one_byte_short_refuses_before_a_byte_moves_and_the_same_byte_admits() {
    let root = temporary("budget-two-sided");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(6, 32 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();

    // ONE BYTE SHORT.
    let store = store
        .bind_disk_budget(Some(SIX_BLOB_CLOSURE_BYTES - 1))
        .unwrap();
    // One request at a time, each object would land before the next is admitted: only the
    // whole closure's admission refuses before a byte moves.
    let mut one_at_a_time = request_for(&store, &origin.base, &policy);
    one_at_a_time.streams = 1;
    let refusal = pull(&one_at_a_time).unwrap_err();
    assert_eq!(refusal.code, Code::CAPACITY_EXHAUSTED, "{refusal}");
    // "Before a byte moves" is proved by the origin and the store, not by the exit code: no
    // object was asked for, none landed and no put- temp was even opened.
    for object in &checkpoint.objects {
        assert_eq!(origin.hits(&format!("/obj/{}", object.sha256)), 0, "an object was asked for");
    }
    assert!(
        store.objects().unwrap().is_empty(),
        "the refusal still admitted objects"
    );
    let temps: Vec<_> = std::fs::read_dir(root.join("tmp"))
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("put-"))
        .collect();
    assert!(
        temps.is_empty(),
        "the refusal left scratch behind: {temps:?}"
    );

    // EXACTLY ENOUGH. A budget that refuses everything passes a one-sided test just as
    // happily as one that refuses nothing, so both sides are pinned.
    let store = store
        .bind_disk_budget(Some(SIX_BLOB_CLOSURE_BYTES))
        .unwrap();
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(report.fetched, 8);
    assert_eq!(store.occupancy().unwrap(), SIX_BLOB_CLOSURE_BYTES);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn the_budget_charges_what_the_pull_adds_not_what_it_declares() {
    // THE ARM THAT MATTERS ON A CONTENT-ADDRESSED STORE. A store already holding three of
    // the six blobs must be admitted under a budget that could never hold the six-blob
    // closure declared whole. Charging `declared_bytes()` instead of `wanted_bytes()`
    // refuses this pull — and the two agree exactly when nothing is shared, which is
    // precisely the fixture a careless proof would reach for.
    let root = temporary("budget-dedup");
    let store = Store::init(&root).unwrap();
    let policy = SourcePolicy::default();

    let small = Arc::new(checkpoint(3, 32 * 1024));
    let small_origin = hub_origin(Arc::clone(&small), |_, _| None);
    pull(&request_for(&store, &small_origin.base, &policy)).unwrap();
    assert_eq!(store.occupancy().unwrap(), THREE_BLOB_CLOSURE_BYTES);

    let budget = THREE_BLOB_CLOSURE_BYTES + SIX_AFTER_THREE_WANTED_BYTES;
    assert!(
        budget < THREE_BLOB_CLOSURE_BYTES + SIX_BLOB_CLOSURE_BYTES,
        "the budget must be one no `declared_bytes()` gate could pass, or the arm is vacuous"
    );
    let store = store.bind_disk_budget(Some(budget)).unwrap();
    let large = Arc::new(checkpoint(6, 32 * 1024));
    let large_origin = hub_origin(Arc::clone(&large), |_, _| None);
    let report = pull(&request_for(&store, &large_origin.base, &policy)).unwrap();
    assert_eq!(report.held, 3, "the three shared blobs were not credited");
    assert_eq!(store.occupancy().unwrap(), budget);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_pull_that_fits_once_gc_evicts_an_idle_cache_root_proceeds() {
    let root = temporary("budget-after-gc");
    let store = Store::init(&root).unwrap();
    let policy = SourcePolicy::default();
    // Its blobs are 16 KiB, so it shares nothing with the six-blob closure.
    let idle = Arc::new(checkpoint_sizes(&[16 * 1024; 2]));
    let idle_origin = hub_origin(Arc::clone(&idle), |_, _| None);
    pull(&request_for(&store, &idle_origin.base, &policy)).unwrap();
    assert_eq!(store.occupancy().unwrap(), closure_bytes(&idle));

    // Room for the six-blob closure alone: it fits only once the idle root is gone.
    let store = store.bind_disk_budget(Some(SIX_BLOB_CLOSURE_BYTES)).unwrap();
    let wanted = Arc::new(checkpoint(6, 32 * 1024));
    let wanted_origin = hub_origin(Arc::clone(&wanted), |_, _| None);
    let report = pull(&request_for(&store, &wanted_origin.base, &policy)).unwrap();
    assert_eq!(report.fetched, 8);
    // Exactly the six-blob closure: the idle root's objects made the room.
    assert_eq!(store.occupancy().unwrap(), SIX_BLOB_CLOSURE_BYTES);
    assert!(idle.objects.iter().all(|object| !store.contains(&object.sha256)));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_store_with_no_budget_admits_everything() {
    let root = temporary("budget-absent");
    let store = Store::init(&root).unwrap();
    assert_eq!(store.disk_budget(), None);
    let checkpoint = Arc::new(checkpoint(6, 32 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let report = pull(&request_for(&store, &origin.base, &SourcePolicy::default())).unwrap();
    assert_eq!(report.fetched, 8);

    // And clearing a budget really clears it, rather than leaving a stale number behind.
    let store = store.bind_disk_budget(Some(1)).unwrap();
    assert_eq!(store.disk_budget(), Some(1));
    let store = store.bind_disk_budget(None).unwrap();
    assert_eq!(store.disk_budget(), None);
    assert!(!root.join(crate::store::DISK_BUDGET).exists());
    assert_eq!(Store::open(&root).unwrap().disk_budget(), None);
    let _ = std::fs::remove_dir_all(root);
}

/// `/dev/full` is Linux's.
#[cfg(target_os = "linux")]
#[test]
fn a_real_enospc_from_the_kernel_is_a_capacity_verdict_not_an_io_failure() {
    // The error here is not constructed and not simulated: `/dev/full` is a real device
    // whose write returns a real ENOSPC from the real kernel, which is the exact errno a
    // full container disk delivers. That error is then carried through the ordinary Store
    // write path's own failure handling.
    let mut full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .expect("/dev/full is required for this arm");
    let error = full
        .write_all(&[0u8; 4096])
        .expect_err("/dev/full accepted a write");
    assert_eq!(error.raw_os_error(), Some(28), "ENOSPC is errno 28");

    let refusal = crate::store::classify_io("write blob", error);
    assert_eq!(
        refusal.code,
        Code::CAPACITY_EXHAUSTED,
        "a full filesystem must be a verdict, not weather: {refusal}"
    );

    // The other side: an ordinary I/O failure must NOT be a capacity verdict, or the
    // classification would poison every retryable fault on the pod.
    let ordinary = std::fs::File::open("/definitely/not/here").unwrap_err();
    assert_eq!(
        crate::store::classify_io("open blob", ordinary).code,
        Code::IO_FAILED
    );
}

/// The same kernel ENOSPC met by a module that wraps its own I/O errors (here, writing a
/// checkpoint part out) is the same verdict: every I/O error goes through one classifier.
#[cfg(target_os = "linux")]
#[test]
fn a_full_disk_met_outside_the_store_is_still_a_capacity_verdict() {
    let root = temporary("full-outside-store");
    let store = Store::init(&root).unwrap();
    let part = crate::header::Part::plan(crate::dtype::Dtype::U8, vec![16], &[7u8; 16]);
    let mut full = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .expect("/dev/full is required for this arm");
    let refusal = crate::checkpoint::materialize(&store, "w", &part, &mut full).unwrap_err();
    assert_eq!(refusal.code, Code::CAPACITY_EXHAUSTED, "{refusal}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn downloaded_cache_root_is_protected_then_evicted_and_refetched() {
    let checkpoint = Arc::new(checkpoint(2, 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let root = temporary("pressure-cached-root");
    let store = Store::init(&root).unwrap();
    let request = PullRequest::new(&store, &origin.base, "acme/model", Anonymous, &policy);
    let cold = pull(&request).unwrap();
    assert!(cold.bytes_moved > 0);
    let repository = crate::repository::RepositoryName::new("acme", "model").unwrap();
    let body = std::fs::read(store.repository_path(&repository)).unwrap();
    assert!(crate::cache_roots::matches(&store, &repository, &body).unwrap());
    let parsed = crate::repository::Repository::parse(&body).unwrap();
    let keep = vec![parsed.checkpoints[0].manifest.sha256.clone()];
    assert_eq!(
        crate::gc::collect_cached(&root, &keep)
            .unwrap()
            .reclaimed_bytes,
        0
    );
    let report = crate::gc::collect_cached(&root, &[]).unwrap();
    assert!(report.reclaimed_bytes > 0);
    assert!(!store.repository_path(&repository).exists());
    let again = pull(&request).unwrap();
    assert_eq!(again.bytes_moved, cold.bytes_moved);
    assert!(store.repository_path(&repository).exists());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn complete_content_length_body_does_not_wait_for_connection_close() {
    let root = temporary("complete-keepalive");
    let store = Store::init(&root).unwrap();
    let bytes = b"complete bounded body";
    let object = ObjectRef::of(bytes);
    let (plan, _) =
        FetchPlan::of_objects(&store, "keepalive", std::slice::from_ref(&object)).unwrap();
    let grant = DeliveryGrant::mint(&plan, &object).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (close_peer, hold_open) = std::sync::mpsc::channel();
    let peer = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        read_request(&mut socket).unwrap();
        write!(
            socket,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
            bytes.len()
        )
        .unwrap();
        socket.write_all(bytes).unwrap();
        // Keep TCP open until the complete Store admission returns. Disk fsync
        // time is unrelated to HTTP framing and may exceed a test socket timer.
        let _ = hold_open.recv();
    });
    // No wall deadline: admission's fsync takes what a loaded disk takes. A fetch that waited
    // for the close would get no byte, and the ledger's own stall rule would refuse it.
    let ledger = Ledger::with_resolution(Duration::from_millis(20));
    let result = fetch_ranged(
        &store,
        &grant,
        &format!("http://{address}/body"),
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        Ranged::default(),
        None,
    );
    close_peer.send(()).unwrap();
    peer.join().unwrap();
    let result = result.unwrap();
    assert_eq!(result.object, object);
    assert!(store.record_valid(&object.sha256).is_ok());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn slow_control_answer_does_not_teach_object_stream_patience() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let peer = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        read_request(&mut socket).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
            .unwrap();
    });
    let ledger = Ledger::with_resolution(Duration::from_millis(10));
    let checked = local_policy()
        .check(&format!("http://{address}/presign"))
        .unwrap();
    let response = super::http::request(
        &super::http::Client::new(),
        &checked,
        "GET",
        &[],
        None,
        Deadline::after_seconds(Some(1.0)),
        &ledger,
        super::http::Judge::Transfer,
        None,
        Code::HUB_UNREACHABLE,
    )
    .unwrap();
    assert_eq!(response.read_capped(2).unwrap(), b"{}");
    peer.join().unwrap();
    assert_eq!(
        ledger.patience(),
        ledger.floor(),
        "Hub latency inflated the unrelated object-stream timeout"
    );
}

#[test]
fn consumer_pause_does_not_teach_object_stream_patience() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (resume, paused) = std::sync::mpsc::channel();
    let peer = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        read_request(&mut socket).unwrap();
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\na")
            .unwrap();
        paused.recv().unwrap();
        socket.write_all(b"b").unwrap();
    });
    let ledger = Ledger::with_resolution(Duration::from_millis(10));
    let checked = local_policy()
        .check(&format!("http://{address}/object"))
        .unwrap();
    let mut body = super::http::request(
        &super::http::Client::new(),
        &checked,
        "GET",
        &[],
        None,
        Deadline::after_seconds(Some(1.0)),
        &ledger,
        super::http::Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    )
    .unwrap()
    .into_body();
    let mut byte = [0];
    body.read_exact(&mut byte).unwrap();
    assert_eq!(&byte, b"a");
    // Simulate time spent hashing or writing between successive body reads.
    std::thread::sleep(Duration::from_millis(200));
    resume.send(()).unwrap();
    body.read_exact(&mut byte).unwrap();
    assert_eq!(&byte, b"b");
    peer.join().unwrap();
    assert!(
        ledger.patience() < Duration::from_millis(200),
        "consumer work inflated the unrelated object-stream timeout"
    );
}

#[test]
fn pull_cancellation_interrupts_tls_headers_and_partial_body_waits() {
    for phase in ["tls", "headers", "body"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (entered, waiting) = std::sync::mpsc::channel();
        let peer = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut input = [0u8; 8192];
            if phase == "tls" {
                assert!(socket.read(&mut input).unwrap() > 0);
            } else {
                read_request(&mut socket).unwrap();
                if phase == "body" {
                    socket
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\nx")
                        .unwrap();
                }
            }
            entered.send(()).unwrap();
            while matches!(socket.read(&mut input),Ok(n) if n>0) {}
        });
        let ledger = Arc::new(Ledger::with_resolution(Duration::from_millis(20)));
        ledger.taught(Duration::from_secs(5));
        let abort = Arc::clone(&ledger);
        let cancelling = std::thread::spawn(move || {
            waiting.recv_timeout(Duration::from_secs(1)).unwrap();
            abort.cancel();
        });
        let scheme = if phase == "tls" { "https" } else { "http" };
        let checked = local_policy()
            .check(&format!("{scheme}://{address}/object"))
            .unwrap();
        let started = std::time::Instant::now();
        let result = super::http::request(
            &super::http::Client::new(),
            &checked,
            "GET",
            &[],
            None,
            Deadline::after_seconds(Some(1.0)),
            &ledger,
            super::http::Judge::Stream,
            None,
            Code::TRANSFER_FAILED,
        );
        let refusal = match result {
            Ok(response) => response.read_capped(99).unwrap_err(),
            Err(refusal) => refusal,
        };
        assert!(
            refusal.detail.contains("another object failed"),
            "{phase}: {refusal}"
        );
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "{phase} ignored pull cancellation"
        );
        cancelling.join().unwrap();
        peer.join().unwrap();
    }
}

#[test]
#[cfg(target_os = "linux")]
fn pull_cancellation_interrupts_a_pending_tcp_connection() {
    use rustix::net::{self, AddressFamily, SocketFlags, SocketType};
    let socket = net::socket_with(
        AddressFamily::INET,
        SocketType::STREAM,
        SocketFlags::CLOEXEC,
        Some(net::ipproto::TCP),
    )
    .unwrap();
    net::bind(
        &socket,
        &std::net::SocketAddrV4::new(std::net::Ipv4Addr::LOCALHOST, 0),
    )
    .unwrap();
    net::listen(&socket, 0).unwrap();
    let listener: TcpListener = socket.into();
    let address = listener.local_addr().unwrap();
    // Fill Linux's one-entry backlog and deliberately never accept it. A second
    // connection must keep its same nonblocking socket while polling cancellation.
    let _occupied = TcpStream::connect(address).unwrap();
    let ledger = Arc::new(Ledger::with_resolution(Duration::from_millis(20)));
    ledger.taught(Duration::from_secs(5));
    let abort = Arc::clone(&ledger);
    let cancelling = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        abort.cancel();
    });
    let checked = local_policy()
        .check(&format!("http://{address}/object"))
        .unwrap();
    let started = std::time::Instant::now();
    let refusal = match super::http::request(
        &super::http::Client::new(),
        &checked,
        "GET",
        &[],
        None,
        Deadline::after_seconds(Some(1.0)),
        &ledger,
        super::http::Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    ) {
        Err(refusal) => refusal,
        Ok(_) => panic!("full listener unexpectedly answered"),
    };
    assert!(
        refusal.detail.contains("another object failed"),
        "{refusal}"
    );
    assert!(started.elapsed() < Duration::from_millis(500));
    cancelling.join().unwrap();
}

#[test]
fn attempt_cancellation_interrupts_a_partial_tls_record() {
    let key = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let cert = rustls::pki_types::CertificateDer::from(key.cert.der().to_vec());
    let private = rustls::pki_types::PrivateKeyDer::try_from(key.signing_key.serialize_der()).unwrap();
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::aws_lc_rs::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(vec![cert.clone()], private)
    .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    // The cancel lands only with the head in the client's hands and part of the body's record
    // on the wire, the rest withheld: the client is inside that record, whatever the load.
    let (partial, receiving) = std::sync::mpsc::channel();
    let (headed, head_seen) = std::sync::mpsc::channel();
    let peer = std::thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        socket.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let conn = rustls::ServerConnection::new(Arc::new(config)).unwrap();
        let mut stream = rustls::StreamOwned::new(conn, socket);
        let mut request = Vec::new();
        let mut buffer = [0u8; 1024];
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            let n = stream.read(&mut buffer).unwrap();
            request.extend_from_slice(&buffer[..n]);
        }
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1024\r\n\r\n")
            .unwrap();
        stream.flush().unwrap();
        stream.conn.writer().write_all(&[b'x'; 1024]).unwrap();
        let mut encrypted = Vec::new();
        while stream.conn.wants_write() {
            stream.conn.write_tls(&mut encrypted).unwrap();
        }
        stream.sock.write_all(&encrypted[..41]).unwrap();
        partial.send(()).unwrap();
        // Nothing more until the client gives up the connection.
        let _ = stream.sock.read(&mut [0u8; 16]);
    });
    let root = Ledger::with_resolution(Duration::from_millis(100));
    let attempt = Arc::new(super::ledger::Attempt::default());
    let ledger = root.for_request(Arc::clone(&attempt), root.sample());
    let cancel = std::thread::spawn(move || {
        receiving.recv().unwrap();
        head_seen.recv().unwrap();
        attempt.stopped.store(true, Ordering::Release);
    });
    let client = super::http::Client::with_extra_roots(&[cert]);
    let policy = SourcePolicy {
        allowed_hosts: vec!["localhost".into()],
        allow_local: true,
        max_redirects: 0,
        ..Default::default()
    };
    let checked = policy
        .check(&format!("https://localhost:{port}/object"))
        .unwrap();
    let response = super::http::request(
        &client,
        &checked,
        "GET",
        &[],
        None,
        Deadline::after_seconds(Some(2.0)),
        &ledger,
        super::http::Judge::Stream,
        None,
        Code::TRANSFER_FAILED,
    )
    .unwrap();
    headed.send(()).unwrap();
    let started = std::time::Instant::now();
    let refusal = response.read_capped(1024).unwrap_err();
    assert!(
        refusal.detail.contains("object attempt was stopped"),
        "{refusal}"
    );
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "partial TLS record hid cancellation"
    );
    root.check_running().unwrap();
    cancel.join().unwrap();
    peer.join().unwrap();
}

/// Where the door spends its time at pull concurrency: 64 MiB objects from memory, so no
/// network, through the same stream_verified -> admit_verified pair every pull uses. One
/// real disk, three durability disciplines: per-object fsync (every admission before, and a
/// persistent disk outside a pull now), a persistent pull's epoch (its closing sync is in the
/// wall time), and an ephemeral disk.
#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture"]
fn bench_door_concurrency() {
    use platform::durable::DiskClass;
    const SIZE: usize = 64 << 20;
    let (declared, body) = paced_objects(SIZE, 32);
    // Interleaved, so the three share whatever else the disk is doing at the time.
    for writers in [1usize, 16, 32] {
        for (discipline, class, epoch) in [
            ("per-object fsync", DiskClass::Persistent, false),
            ("persistent pull", DiskClass::Persistent, true),
            ("ephemeral", DiskClass::Ephemeral, false),
        ] {
            let root = temporary(&format!("door-{writers}"));
            let plain = Store::init(&root).unwrap().with_disk_class(class);
            let opened = epoch.then(|| crate::unsynced::Epoch::begin(&plain).unwrap().unwrap());
            let store = plain.syncing(opened.clone());
            let next = AtomicUsize::new(0);
            let streamed = Mutex::new(Duration::ZERO);
            let admitted = Mutex::new(Duration::ZERO);
            // A busy catalog is re-asked, as a pull re-asks it.
            let busy = AtomicUsize::new(0);
            let count = if writers == 1 { 8 } else { declared.len() };
            let before = crate::stats::snapshot();
            let started = std::time::Instant::now();
            std::thread::scope(|scope| {
                for _ in 0..writers {
                    scope.spawn(|| loop {
                        let at = next.fetch_add(1, Ordering::Relaxed);
                        let Some(object) = declared[..count].get(at) else {
                            return;
                        };
                        let bytes = body(&object.sha256);
                        loop {
                            let t0 = std::time::Instant::now();
                            let verified = store
                                .stream_verified(
                                    &mut bytes.as_slice(),
                                    Some(object),
                                    &Default::default(),
                                    crate::store::PUT_TEMP,
                                )
                                .unwrap();
                            let t1 = std::time::Instant::now();
                            let outcome = store.admit_verified(verified, &Default::default(), None);
                            let t2 = std::time::Instant::now();
                            *streamed.lock().unwrap() += t1 - t0;
                            *admitted.lock().unwrap() += t2 - t1;
                            match outcome {
                                Ok(_) => break,
                                Err(refusal) if refusal.code == Code::IO_FAILED => {
                                    busy.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(refusal) => panic!("{refusal:?}"),
                            }
                        }
                    });
                }
            });
            let sync = opened.map_or(0.0, |epoch| epoch.finish().unwrap().1);
            let seconds = started.elapsed().as_secs_f64();
            let after = crate::stats::snapshot();
            println!(
                "{discipline:>16} {writers:3} writers: {count} x 64 MiB in {seconds:6.2} s = \
                 {:7.1} MB/s; per object: stream+hash(+fsync) {:6.0} ms, admission (wait + \
                 turn) {:6.0} ms; {} batches, {:.2} s committing; closing sync {sync:.2} s; \
                 {} busy re-asks",
                (count * SIZE) as f64 / 1e6 / seconds,
                streamed.lock().unwrap().as_secs_f64() * 1e3 / count as f64,
                admitted.lock().unwrap().as_secs_f64() * 1e3 / count as f64,
                after.admission_batches - before.admission_batches,
                (after.admission_nanos - before.admission_nanos) as f64 / 1e9,
                busy.load(Ordering::Relaxed),
            );
            let _ = std::fs::remove_dir_all(root);
        }
    }
}

// ---------------------------------------------------------------- durability by disk

#[cfg(target_os = "linux")]
#[test]
fn an_ephemeral_store_downloads_without_a_marker() {
    let root = std::path::Path::new("/dev/shm").join(format!(
        "tensorfs-ephemeral-pull-{}-{}",
        std::process::id(),
        crate::meta::now_nanos_unique()
    ));
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(6, 32 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(report.fetched, 8);
    assert_eq!(store.disk_class(), platform::durable::DiskClass::Ephemeral);
    assert!(!root.join("tmp/unsynced").exists());
    for object in &checkpoint.objects {
        assert!(store.record_valid(&object.sha256).is_ok());
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_persistent_store_downloads_and_retires_its_marker() {
    let root = crate::unsynced::testing::persistent_root("persistent-pull");
    let store = Store::init(&root).unwrap();
    assert_eq!(store.disk_class(), platform::durable::DiskClass::Persistent);
    let checkpoint = Arc::new(checkpoint(6, 32 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let cold = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(cold.fetched, 8);
    assert!(crate::unsynced::testing::markers(&root).is_empty());
    let warm = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(warm.fetched, 0);
    assert!(crate::unsynced::testing::markers(&root).is_empty());
    let _ = std::fs::remove_dir_all(root);
}

const DURABILITY_CHILD: &str = "TENSORFS_DURABILITY_CHILD";

/// The child of `a_pull_killed_mid_walk_is_repaired_after_an_unclean_shutdown`: one serial
/// pull, run until the parent kills it.
#[test]
#[ignore = "child process of a_pull_killed_mid_walk_is_repaired_after_an_unclean_shutdown"]
fn durability_child_pull() {
    let Ok(spec) = std::env::var(DURABILITY_CHILD) else {
        return;
    };
    let (root, base) = spec.split_once('\n').unwrap();
    let store = Store::open(std::path::Path::new(root)).unwrap();
    let policy = SourcePolicy::default();
    let mut request = request_for(&store, base, &policy);
    request.streams = 1;
    let _ = pull(&request);
}

/// A real process is SIGKILLed in the middle of a pull on a persistent disk. Within the
/// same boot nothing is lost and nothing is rehashed. Across a reboot — here: its marker
/// renamed to a boot that has ended, and one admitted object torn behind a record that
/// still binds, which is what a lost page cache can leave — the next open rehashes what the
/// dead pull admitted and removes the torn object, and the next pull fetches it again.
#[test]
fn a_pull_killed_mid_walk_is_repaired_after_an_unclean_shutdown() {
    use crate::unsynced::testing::{from_an_earlier_boot, markers, persistent_root, tear};
    let root = persistent_root("killed-pull");
    let store = Store::init(&root).unwrap();
    assert_eq!(store.disk_class(), platform::durable::DiskClass::Persistent);
    let checkpoint = Arc::new(checkpoint(8, 64 * 1024));
    let served = Arc::new(AtomicUsize::new(0));
    let stalling = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let origin = hub_origin(Arc::clone(&checkpoint), {
        let served = Arc::clone(&served);
        let stalling = Arc::clone(&stalling);
        move |_, _| {
            if !stalling.load(Ordering::SeqCst) || served.fetch_add(1, Ordering::SeqCst) < 5 {
                None
            } else {
                Some(Serve::StallHead)
            }
        }
    });
    let fence = crate::fork_fence::spawning();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "transport::tests::durability_child_pull",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env(
            DURABILITY_CHILD,
            format!("{}\n{}", root.display(), origin.base),
        )
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    drop(fence);
    let recorded = |store: &Store| {
        checkpoint
            .objects
            .iter()
            .filter(|object| store.record_valid(&object.sha256).is_ok())
            .count()
    };
    let waited = std::time::Instant::now();
    while served.load(Ordering::SeqCst) < 6 || recorded(&store) < 4 {
        assert!(
            waited.elapsed() < Duration::from_secs(60),
            "the child admitted {} objects",
            recorded(&store)
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    child.kill().unwrap();
    child.wait().unwrap();

    let left = markers(&root);
    assert_eq!(left.len(), 1, "the dead pull's marker stands");
    let admitted: Vec<ObjectRef> = checkpoint
        .objects
        .iter()
        .filter(|object| store.record_valid(&object.sha256).is_ok())
        .cloned()
        .collect();
    let victim = admitted[0].clone();
    tear(&store.blob_path(&victim.sha256));
    assert!(
        store.record_valid(&victim.sha256).is_ok(),
        "without recovery the torn bytes would be trusted"
    );
    Store::open(&root).unwrap();
    assert!(
        store.blob_path(&victim.sha256).exists(),
        "the same boot rehashes nothing"
    );

    from_an_earlier_boot(&left[0]);
    let before = crate::stats::snapshot();
    let reopened = Store::open(&root).unwrap();
    let after = crate::stats::snapshot();
    assert!(!reopened.blob_path(&victim.sha256).exists());
    assert!(after.recovery_rehashed - before.recovery_rehashed >= admitted.len() as u64);
    assert!(after.recovery_removed > before.recovery_removed);
    assert!(markers(&root).is_empty());
    for object in &admitted[1..] {
        assert!(reopened.open_verified(&object.sha256).is_ok());
    }

    stalling.store(false, Ordering::SeqCst);
    let policy = SourcePolicy::default();
    let resumed = pull(&request_for(&reopened, &origin.base, &policy)).unwrap();
    assert_eq!(resumed.held + resumed.fetched, resumed.declared);
    assert!(resumed.held >= admitted.len() as u64 - 1);
    let file = reopened.open_verified(&victim.sha256).unwrap();
    assert_eq!(file.len(), victim.length);
    assert_eq!(reopened.hash_object(&victim.sha256).unwrap(), victim.sha256);
    assert!(markers(&root).is_empty());
    let _ = std::fs::remove_dir_all(root);
}

mod tail;
