//! Transport proofs against a REAL store and REAL sockets — every origin here is a
//! loopback listener the test starts itself, so the whole suite runs with no network and
//! no hub. The ledger tests run at millisecond resolution: the cadence is configuration,
//! the RULE under test is production's.

use super::hub::{Anonymous, HostToken};
use super::ledger::{Deadline, Ledger};
use super::policy::{blocked, SourcePolicy};
use super::pull::{fetch_url, pull, PullRequest};
use super::push::{push_object, UploadGrant};
use crate::canon::Value;
use crate::dtype::Dtype;
use crate::err::Code;
use crate::fetch::{DeliveryGrant, FetchPlan};
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

enum Serve {
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

struct Origin {
    base: String,
    hits: Arc<Mutex<HashMap<String, usize>>>,
}

impl Origin {
    fn hits(&self, path: &str) -> usize {
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
struct Checkpoint {
    /// Every object body the ORIGIN can serve, by bare hex digest (manifest included).
    bodies: HashMap<String, Vec<u8>>,
    manifest: ObjectRef,
    /// The closure rows: header + blobs, sorted by digest, manifest EXCLUDED.
    objects: Vec<ObjectRef>,
}

fn checkpoint(blobs: usize, size: usize) -> Checkpoint {
    assert!(size.is_multiple_of(4) && size > 0);
    let spec = registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let mut bodies = HashMap::new();
    let mut tensors = Vec::new();
    for index in 0..blobs {
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
        (
            "stamps",
            Value::obj(vec![("family", Value::arr(vec![Value::str("dit")]))]),
        ),
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
    Value::obj(vec![("urls", Value::map(urls))])
}

/// The standard hub emulation: closure + presign + object GETs, with per-object override.
fn hub_origin(
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
    }
}

fn request_for<'a>(store: &'a Store, base: &'a str, policy: &'a SourcePolicy) -> PullRequest<'a> {
    let mut request = PullRequest::new(store, base, "acme/model", &Anonymous, policy);
    request.sample_seconds = 0.025;
    request
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
        checkpoint.bodies.values().map(|b| b.len() as u64).sum()
    );
    assert_eq!(report.model, "acme/model");
    assert_eq!(report.release, "r1");
    assert_eq!(
        report.stamps,
        vec![("family".to_string(), vec!["dit".to_string()])]
    );
    let object_gets = origin.total_hits() - 2; // closure + presign
    assert_eq!(object_gets, 8);

    // Warm: one closure question, no presign, no GET, nothing moved — and still PROVEN.
    let warm = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(warm.held, 8);
    assert_eq!(warm.fetched, 0);
    assert_eq!(warm.bytes_moved, 0);
    assert_eq!(origin.hits("/v1/tensorfs/closure"), 2);
    assert_eq!(origin.hits("/v1/tensorfs/presign"), 1);
    assert_eq!(origin.total_hits(), 8 + 3);
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
    let mut permissions = std::fs::metadata(&path).unwrap().permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o644);
    std::fs::set_permissions(&path, permissions).unwrap();
    std::fs::write(&path, vec![b'x'; victim.length as usize]).unwrap();

    let healed = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(healed.fetched, 1);
    assert_eq!(healed.bytes_moved, victim.length);
    assert!(healed
        .presence
        .iter()
        .any(|(id, why)| id == &victim.id() && *why == "corrupt_removed"));
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

#[test]
fn slow_origin_teaches_the_ledger_and_is_left_alone() {
    let root = temporary("slow");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(1, 6 * 1024));
    let slow = checkpoint.objects[0].sha256.clone();
    let slow_for = slow.clone();
    let bodies = checkpoint.bodies.clone();
    // 400 ms to first byte teaches the ledger this origin pauses; the 200 ms drip gaps
    // that follow are far over the 150 ms noise floor and must NOT be called stalls.
    let origin = hub_origin(Arc::clone(&checkpoint), move |hex, _| {
        if hex == slow_for {
            return Some(Serve::Drip {
                body: bodies[hex].clone(),
                chunk: 1024,
                gap: Duration::from_millis(200),
                ttfb: Duration::from_millis(400),
            });
        }
        None
    });
    let policy = SourcePolicy::default();
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(report.fetched, 3);
    assert_eq!(origin.hits(&format!("/obj/{slow}")), 1); // one ask; never condemned
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn deadline_refuses_midpull_and_the_next_pull_resumes_per_object() {
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
    let mut rushed = request_for(&store, &origin.base, &policy);
    rushed.streams = 1;
    rushed.deadline = Deadline::after_seconds(Some(0.4));
    let refusal = pull(&rushed).unwrap_err();
    assert_eq!(refusal.code, Code::DEADLINE_EXCEEDED);

    // Whatever landed before the deadline stays landed; the re-plan asks only for the rest.
    let resumed = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(resumed.held + resumed.fetched, 6);
    assert!(
        resumed.held > 0,
        "per-object resume kept the landed objects"
    );
    resumed
        .presence
        .iter()
        .for_each(|(_, why)| assert_ne!(*why, "corrupt_removed"));
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

    // Today's shape, spelled out: every URL stamped once, at T=0, with the hub's cap on it.
    let eager_root = temporary("expiry-eager");
    let eager = Store::init(&eager_root).unwrap();
    let (eager_plan, _) =
        FetchPlan::of(&eager, "eager", &checkpoint.manifest, &checkpoint.objects).unwrap();
    let dies = now_millis() + hub.lifetime.as_millis();
    let up_front: std::collections::BTreeMap<String, String> = eager_plan
        .wanted
        .iter()
        .map(|o| {
            (
                o.sha256.clone(),
                format!("{}/obj/{}?dies={dies}", hub.origin.base, o.sha256),
            )
        })
        .collect();
    let refusal = super::pull::fetch_wanted(
        &eager,
        &eager_plan,
        &up_front,
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &Ledger::with_resolution(Duration::from_millis(200)),
        4,
        None,
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::TRANSFER_FAILED);
    assert!(refusal.detail.contains("403"), "{}", refusal.detail);
    let refused_up_front = hub.rejected.load(Ordering::Relaxed);
    assert!(
        refused_up_front > 0,
        "the origin never actually refused an aged URL — the proof would be vacuous"
    );
    let _ = std::fs::remove_dir_all(eager_root);

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
        refused_up_front,
        "a dead URL was presented during the lazy walk — every fetch must carry an \
         authorization minted for it"
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

    // Cut the first walk off with objects still to move.
    let mut rushed = request_for(&store, &hub.origin.base, &policy);
    rushed.streams = 2;
    rushed.sample_seconds = 0.2;
    rushed.deadline = Deadline::after_seconds(Some(0.6));
    let refusal = pull(&rushed).unwrap_err();
    assert_eq!(refusal.code, Code::DEADLINE_EXCEEDED);

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
    let report = pull(&request_for(&store, &origin.base, &policy)).unwrap();
    assert_eq!(report.fetched, 4);
    assert_eq!(origin.hits(&format!("/obj/{flaky}")), 2);

    // A 403 against a URL minted moments ago: one ask, a typed refusal, no retry. The
    // reason is the LEASE, not the status — this hub declares no lifetime, so there is
    // nothing to argue that time was the cause, and re-minting would ask the same
    // question. A 403 against a lease the hub's own declared life says is spent is
    // re-minted instead; that half is
    // `a_403_is_judged_by_the_leases_age_and_not_by_a_count`.
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
    assert!(refusal.detail.contains("403"));
    assert_eq!(origin2.hits(&format!("/obj/{expired}")), 1);
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(store2root);
}

#[test]
fn sixteen_way_walk_lands_everything_through_the_one_door() {
    let root = temporary("nway");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(24, 16 * 1024));
    let origin = hub_origin(Arc::clone(&checkpoint), |_, _| None);
    let policy = SourcePolicy::default();
    let counted = AtomicUsize::new(0);
    let observer = |_: &ObjectRef, _: u64| {
        counted.fetch_add(1, Ordering::Relaxed);
    };
    let mut request = request_for(&store, &origin.base, &policy);
    request.streams = 16;
    request.on_object = Some(&observer);
    let report = pull(&request).unwrap();
    assert_eq!(report.fetched, 26);
    assert_eq!(counted.load(Ordering::Relaxed), 26);
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

    let denied: Dispatch = Arc::new(|_: &Request| status(401, b"{}"));
    let origin = self::origin(denied);
    let refusal = pull(&request_for(&store, &origin.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::CREDENTIAL_REQUIRED);
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
    let token = HostToken {
        hosts: vec!["127.0.0.1".into()],
        token: "worker-token".into(),
    };
    let policy = SourcePolicy::default();
    let mut request = PullRequest::new(&store, &origin.base, "acme/model", &token, &policy);
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
    let refusal = fetch_url(
        &store,
        &grant,
        &url,
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        None,
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::REDIRECT_REFUSED);

    // Budget 1: the hop faces the whole predicate, and the target host is outside the
    // declaration — the PRECISE refusal, exactly as egress.py validated the hop it then
    // refused anyway.
    let mut hopping = local_policy();
    hopping.max_redirects = 1;
    let refusal = fetch_url(
        &store,
        &grant,
        &url,
        &hopping,
        &Anonymous,
        Deadline::none(),
        &ledger,
        None,
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::SOURCE_NOT_ALLOWED);
    let _ = std::fs::remove_dir_all(root);
}

// ---------------------------------------------------------------- the push half

#[test]
fn push_streams_a_verified_object_and_412_is_the_object_being_durable() {
    let root = temporary("push");
    let store = Store::init(&root).unwrap();
    let body: Vec<u8> = (0u8..=255).cycle().take(64 * 1024).collect();
    let object = ObjectRef::of(&body);
    store
        .put_stream(&mut body.as_slice(), Some(&object), &Default::default())
        .unwrap();

    let seen: Arc<Mutex<Vec<(String, String, usize)>>> = Arc::default();
    let seen_for = Arc::clone(&seen);
    let accept: Dispatch = Arc::new(move |request: &Request| {
        seen_for.lock().unwrap().push((
            request.header("if-none-match").unwrap_or("").to_string(),
            request.header("content-length").unwrap_or("").to_string(),
            request.body.len(),
        ));
        status(200, b"")
    });
    let origin = origin(accept);
    let ledger = Ledger::with_resolution(Duration::from_millis(25));
    let pushed = push_object(
        &store,
        &object.sha256,
        false,
        &UploadGrant::to(&format!("{}/put/{}", origin.base, object.sha256)),
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        "",
    )
    .unwrap();
    assert_eq!(pushed.http_status, 200);
    assert_eq!(pushed.bytes_sent, object.length);
    assert!(!pushed.landed_precondition);
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "*");
        assert_eq!(seen[0].1, object.length.to_string());
        assert_eq!(seen[0].2, object.length as usize);
    }

    // A retryable answer buys a re-send; the second lands.
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for = Arc::clone(&attempts);
    let flaky: Dispatch = Arc::new(move |_request: &Request| {
        if attempts_for.fetch_add(1, Ordering::SeqCst) == 0 {
            status(503, b"busy")
        } else {
            status(200, b"")
        }
    });
    let origin = self::origin(flaky);
    let pushed = push_object(
        &store,
        &object.sha256,
        false,
        &UploadGrant::to(&format!("{}/put/x", origin.base)),
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        "",
    )
    .unwrap();
    assert_eq!(pushed.http_status, 200);
    assert_eq!(attempts.load(Ordering::SeqCst), 2);

    // 412 under If-None-Match: the immutable key already holds these bytes. Success.
    let precondition: Dispatch = Arc::new(|_: &Request| status(412, b""));
    let origin = self::origin(precondition);
    let pushed = push_object(
        &store,
        &object.sha256,
        false,
        &UploadGrant::to(&format!("{}/put/x", origin.base)),
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        "",
    )
    .unwrap();
    assert!(pushed.landed_precondition);

    // A redirect can never replace the exact granted destination — even one whose
    // target passes the whole predicate.
    let hijack: Dispatch = Arc::new(|_: &Request| Serve::Whole {
        status: 307,
        body: Vec::new(),
        headers: vec![("location".into(), "http://127.0.0.1:9/p".into())],
    });
    let origin = self::origin(hijack);
    let refusal = push_object(
        &store,
        &object.sha256,
        false,
        &UploadGrant::to(&format!("{}/put/x", origin.base)),
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        "",
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::REDIRECT_REFUSED);
    let _ = std::fs::remove_dir_all(root);
}

/// th-132's other half, proven at the socket: a tensorhub upload grant is a presigned URL
/// AND the exact headers its signature covers — `if-none-match: *` and
/// `x-amz-checksum-sha256` (podweights.validateGrant refuses a grant without them, and a
/// SigV4 store refuses a PUT without them) — so the mover sends them verbatim, once each,
/// on every attempt. The origin here DEMANDS them: without the checksum header it answers
/// 400 and the push fails, which is exactly what made an upload grant unspendable.
#[test]
fn push_spends_a_grant_the_origin_refuses_without_its_signed_headers() {
    let root = temporary("push-grant-headers");
    let store = Store::init(&root).unwrap();
    let body = b"granted bytes".to_vec();
    let object = ObjectRef::of(&body);
    store
        .put_stream(&mut body.as_slice(), Some(&object), &Default::default())
        .unwrap();
    let raw: Vec<u8> = object
        .sha256
        .as_bytes()
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    let checksum = crate::b64::encode(&raw);

    type SeenHeaders = Arc<Mutex<Vec<Vec<(String, String)>>>>;
    let seen: SeenHeaders = Arc::default();
    let seen_for = Arc::clone(&seen);
    let checksum_for = checksum.clone();
    let demanding: Dispatch = Arc::new(move |request: &Request| {
        seen_for.lock().unwrap().push(request.headers.clone());
        // The store's side of the contract, exactly as S3 enforces it: no signed
        // checksum, no PUT.
        if request.header("x-amz-checksum-sha256") != Some(checksum_for.as_str())
            || request.header("if-none-match") != Some("*")
        {
            return status(400, b"InvalidRequest: the signed headers are required");
        }
        status(200, b"")
    });
    let origin = origin(demanding);
    let ledger = Ledger::with_resolution(Duration::from_millis(25));
    let destination = format!("{}/put/{}", origin.base, object.sha256);

    // Without the grant's headers this origin refuses — the gap th-132 recorded.
    let refusal = push_object(
        &store,
        &object.sha256,
        false,
        &UploadGrant::to(&destination),
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        "",
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::TRANSFER_FAILED);
    assert!(refusal.detail.contains("400"), "{refusal}");

    // The grant as its minter writes it — url and headers in one document, so the URL
    // cannot be spent without its conditions.
    let document = format!("{destination}\nif-none-match: *\nx-amz-checksum-sha256: {checksum}\n");
    let grant = UploadGrant::parse(&document).unwrap();
    assert_eq!(grant.url, destination);
    assert_eq!(grant.headers.len(), 2);
    seen.lock().unwrap().clear();
    let pushed = push_object(
        &store,
        &object.sha256,
        false,
        &grant,
        &local_policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        "",
    )
    .unwrap();
    assert_eq!(pushed.http_status, 200);
    assert_eq!(pushed.bytes_sent, object.length);
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        let of = |name: &str| -> Vec<&str> {
            seen[0]
                .iter()
                .filter(|(held, _)| held == name)
                .map(|(_, value)| value.as_str())
                .collect()
        };
        // The grant's own if-none-match replaced the built-in — sent exactly once.
        assert_eq!(of("if-none-match"), vec!["*"]);
        assert_eq!(of("x-amz-checksum-sha256"), vec![checksum.as_str()]);
    }
    let _ = std::fs::remove_dir_all(root);
}

/// A grant document is refused for the first thing it gets wrong, before any byte moves —
/// and the grammar survives what a real grant actually contains: a presigned URL full of
/// `&` and `=`, and a base64 checksum ending in `=`.
#[test]
fn a_grant_document_is_refused_for_the_first_disagreeing_thing() {
    let presigned = "https://bucket.s3.amazonaws.com/o/k?X-Amz-Algorithm=AWS4-HMAC-SHA256\
                     &X-Amz-Signature=deadbeef&X-Amz-SignedHeaders=host%3Bif-none-match";
    let checksum = "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=";
    let grant = UploadGrant::parse(&format!(
        "{presigned}\n\nif-none-match: *\nx-amz-checksum-sha256: {checksum}\n"
    ))
    .unwrap();
    assert_eq!(grant.url, presigned);
    assert_eq!(
        grant.headers,
        vec![
            ("if-none-match".to_string(), "*".to_string()),
            ("x-amz-checksum-sha256".to_string(), checksum.to_string()),
        ]
    );

    // A name that is not its own canonical lowercase self is not the name the signature
    // covers; an empty value, a line that is not `name: value`, a repeat, and an empty
    // document are each the grant saying something this mover will not guess at.
    for bad in [
        "https://s3.example/put\nX-Amz-Checksum-Sha256: abc",
        "https://s3.example/put\nif-none-match:",
        "https://s3.example/put\nif-none-match",
        "https://s3.example/put\nif-none-match: *\nif-none-match: *",
        "   \n\n",
    ] {
        let refusal = UploadGrant::parse(bad).unwrap_err();
        assert_eq!(refusal.code, Code::SOURCE_NOT_ALLOWED, "{bad:?}");
    }
    // The cap is the parser's, so every door gets it — file, environment, and any caller.
    let refusal = UploadGrant::parse(&"x".repeat(super::MAX_GRANT_BYTES + 1)).unwrap_err();
    assert_eq!(refusal.code, Code::SIZE_CAP);
}

// ---------------------------------------------------------------- TLS

#[test]
fn the_tls_pump_carries_a_real_exchange() {
    let key = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let cert_der = rustls::pki_types::CertificateDer::from(key.cert.der().to_vec());
    let key_der = rustls::pki_types::PrivateKeyDer::try_from(key.key_pair.serialize_der())
        .unwrap()
        .clone_key();
    let server_config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
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
        true,
        None,
        Code::TRANSFER_FAILED,
    )
    .unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(response.read_capped(1 << 20).unwrap(), payload);
}

// ---------------------------------------------------------------- the parity benchmark

/// The tfs-049 parity benchmark, opt-in:
///
///   nice -n 19 cargo test --release -p tensorfs-core --lib bench_walk -- --ignored --nocapture
///
/// The recorded figures (#566b, 92.42 GiB on a rented pod, 141 MB/s transport ceiling):
/// serial 49.8 MB/s, 16-way 117.6 MB/s through the same admission door. A WAN cannot be
/// conjured locally, so objects here are production-shaped (the 64 MiB grid) and the
/// throttled phase emulates the link's SHAPE — 100 ms to first byte, ~8.7 MB/s per stream,
/// 16 x 8.7 = 139 nominal, mirroring the 141 ceiling. It proves the MECHANISM: the 16-way
/// walk through this door clears the recorded 117.6 MB/s aggregate on a link whose single
/// stream cannot reach a tenth of it. The unthrottled phase then proves the door itself
/// (hash + fsync + no-clobber admission) is not the bottleneck at that figure.
#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture"]
fn bench_walk_16way_parity() {
    const SIZE: usize = 64 * 1024 * 1024; // the production object grid
    const COUNT: usize = 32;
    let mut block = vec![0u8; SIZE];
    let mut x: u64 = 0x243f6a8885a308d3;
    for byte in block.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *byte = (x >> 24) as u8;
    }
    let block = Arc::new(block);
    let body_of = |index: u64| -> Vec<u8> {
        let mut body = block.as_ref().clone();
        body[..8].copy_from_slice(&index.to_le_bytes());
        body
    };
    let mut declared: Vec<ObjectRef> = Vec::new();
    let mut index_of: HashMap<String, u64> = HashMap::new();
    for index in 0..COUNT as u64 {
        let object = ObjectRef::of(&body_of(index));
        index_of.insert(object.sha256.clone(), index);
        declared.push(object);
    }
    declared.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    let index_of = Arc::new(index_of);

    let serve = |throttled: bool| -> Origin {
        let index_of = Arc::clone(&index_of);
        let block = Arc::clone(&block);
        origin(Arc::new(move |request: &Request| {
            let hex = request.path.trim_start_matches("/obj/");
            let mut body = block.as_ref().clone();
            body[..8].copy_from_slice(&index_of[hex].to_le_bytes());
            if throttled {
                Serve::Drip {
                    body,
                    chunk: 512 * 1024,
                    gap: Duration::from_millis(60),
                    ttfb: Duration::from_millis(100),
                }
            } else {
                status(200, &body)
            }
        }))
    };

    let run = |origin: &Origin, streams: usize, subset: &[ObjectRef]| -> f64 {
        let root = temporary(&format!("bench-{streams}-{}", subset.len()));
        let store = Store::init(&root).unwrap();
        let (plan, _) = FetchPlan::of_objects(&store, "bench", subset).unwrap();
        let urls: std::collections::BTreeMap<String, String> = subset
            .iter()
            .map(|o| {
                (
                    o.sha256.clone(),
                    format!("{}/obj/{}", origin.base, o.sha256),
                )
            })
            .collect();
        let ledger = Ledger::new(); // PRODUCTION resolution; nothing here should stall
        let started = std::time::Instant::now();
        let walked = super::pull::fetch_wanted(
            &store,
            &plan,
            &urls,
            &local_policy(),
            &Anonymous,
            Deadline::none(),
            &ledger,
            streams,
            None,
        )
        .unwrap();
        let seconds = started.elapsed().as_secs_f64();
        plan.complete(&store).unwrap();
        let _ = std::fs::remove_dir_all(root);
        assert_eq!(walked.fetched as usize, subset.len());
        walked.bytes_moved as f64 / 1e6 / seconds
    };

    let throttled = serve(true);
    let serial = run(&throttled, 1, &declared[..3]);
    let sixteen = run(&throttled, 16, &declared);
    let open = serve(false);
    let unthrottled = run(&open, 16, &declared);

    println!(
        "bench_walk_16way_parity (recorded: serial 49.8 MB/s, 16-way 117.6 MB/s, ceiling 141)"
    );
    println!("  emulated link (100 ms TTFB, ~8.7 MB/s per stream, 139 MB/s nominal aggregate):");
    println!("    serial  {serial:8.1} MB/s   (one stream of that link, as expected)");
    println!("    16-way  {sixteen:8.1} MB/s   vs the recorded 117.6");
    println!("  unthrottled loopback, 16-way through the full door: {unthrottled:8.1} MB/s");
    assert!(
        sixteen >= 117.6,
        "16-way walk {sixteen:.1} MB/s under the recorded 117.6 MB/s"
    );
    assert!(
        sixteen >= 2.4 * serial,
        "16-way {sixteen:.1} not even 2.4x serial {serial:.1} — the regression the spec forbids"
    );
    assert!(
        unthrottled >= 117.6,
        "the door itself moved {unthrottled:.1} MB/s — below the recorded WAN figure"
    );
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
/// `ceiling` of `None` is a hub that declares nothing, which is the skew a client must also
/// survive.
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

/// **A hub that declares no ceiling gets a window derived from a fact this side owns.**
///
/// The fallback is `streams`: mint exactly what is in flight. It is a degradation and not a
/// guess — a few hundred bytes per call, safe under any body cap a hub could have — and it
/// is deliberately not a number picked to be "probably under" whatever the hub's cap is.
#[test]
fn a_hub_that_declares_no_ceiling_mints_one_wave_at_a_time() {
    let root = temporary("wire-noceiling");
    let store = Store::init(&root).unwrap();
    let checkpoint = Arc::new(checkpoint(30, 1024));
    let hub = wire_hub_origin(Arc::clone(&checkpoint), 128 << 10, usize::MAX, None);
    let policy = SourcePolicy::default();
    let mut request = request_for(&store, &hub.origin.base, &policy);
    request.streams = 4;
    let report = pull(&request).unwrap();
    assert_eq!(report.fetched, checkpoint.objects.len() as u64 + 1);
    // A window of 4 over 32 objects is many calls, and each one is tiny.
    let largest = hub.largest_presign.load(Ordering::Relaxed);
    assert!(
        largest <= 13 + 67 * 4,
        "a fallback window sent a {largest} B body"
    );
    assert!(hub.mints.load(Ordering::Relaxed) >= 8);
    let _ = std::fs::remove_dir_all(root);
}

/// **A 403 is judged by the LEASE'S AGE against the life the hub declared, never by a count
/// of how often this object has been re-authorized.**
///
/// The two agree on a fast link and diverge exactly where a slow one lives. An object that
/// begins its transfer holding half a URL's life and takes longer than that to move is a
/// 64 MiB object at ~107 KB/s: slow, and the transport exists to say slow is not broken. A
/// count would call its second expiry terminal after gigabytes had landed. The age test
/// re-mints for it, and still refuses at the FIRST 403 against a URL minted moments ago —
/// because a refusal of a fresh capability is not about time, and asking to be authorized
/// again would be asking the same question.
///
/// Reaching this decision from the wire means racing a clock, so the lease's age is stated
/// here rather than waited for. Everything else is the production type.
#[test]
fn a_403_is_judged_by_the_leases_age_and_not_by_a_count() {
    use std::collections::BTreeMap;
    use std::time::Instant;

    let checkpoint = Arc::new(checkpoint(3, 1024));
    let hub = wire_hub_origin(Arc::clone(&checkpoint), 128 << 10, usize::MAX, Some(1956));
    let policy = local_policy();
    let ledger = Ledger::with_resolution(Duration::from_millis(50));
    let wanted: Vec<ObjectRef> = checkpoint.objects.clone();
    let object = wanted[0].clone();
    let declared = Duration::from_secs(600);

    let leases = super::pull::Leases {
        client: super::pull::shared_client(),
        base: &hub.origin.base,
        credential: &Anonymous,
        policy: &policy,
        deadline: Deadline::none(),
        ledger: &ledger,
        wanted: &wanted,
        window: 1956,
        held: Mutex::new(BTreeMap::new()),
        minting: Mutex::new(()),
    };

    // A lease whose declared life is more than half gone: the origin's refusal IS about
    // time, and being authorized again is a different question with a different answer.
    leases.held.lock().unwrap().insert(
        object.sha256.clone(),
        super::pull::Lease {
            url: "http://127.0.0.1:1/stale".into(),
            minted: Instant::now() - declared,
            lifetime: Some(declared),
        },
    );
    let minted = super::pull::ObjectUrls::url_for(&leases, 0, &object, true)
        .expect("a spent lease refused by the origin must be re-minted");
    assert!(
        minted.starts_with(&hub.origin.base) && minted.contains(&object.sha256),
        "re-authorizing did not produce a fresh URL: {minted}"
    );
    assert_eq!(hub.mints.load(Ordering::Relaxed), 1);

    // A lease minted moments ago, refused by the origin: terminal at the first answer, with
    // no attempt spent and no second mint.
    leases.held.lock().unwrap().insert(
        object.sha256.clone(),
        super::pull::Lease {
            url: "http://127.0.0.1:1/fresh".into(),
            minted: Instant::now(),
            lifetime: Some(declared),
        },
    );
    let refusal = super::pull::ObjectUrls::url_for(&leases, 0, &object, true)
        .expect_err("a URL minted moments ago is not refused because of time");
    assert_eq!(refusal.code, Code::TRANSFER_FAILED);
    assert!(refusal.detail.contains("minted moments ago"), "{refusal}");
    assert_eq!(
        hub.mints.load(Ordering::Relaxed),
        1,
        "a fresh lease's refusal bought a second mint"
    );
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
