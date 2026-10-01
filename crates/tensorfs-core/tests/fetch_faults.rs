//! Model weights through `ensure` — the Runtime's one download entry — while a REAL loopback
//! hub and a REAL object server inject faults on chosen routes, objects, asks and bytes.
//!
//! Every proof ends one of two ways: the exact bytes resident, verified, with no `put-` temp
//! left behind; or a typed refusal. None is bounded by a wall clock. `Watch` fails a test only
//! when the ensure has stopped making measured progress — no phase, byte or attempt moved —
//! for longer than the faults it was dealt explain, counted in the ledger's own samples.
//! Progress is held to the store: an event may never claim more than the objects resident
//! plus what the server has actually sent for the objects still in flight.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tensorfs_core::canon::{self, Value};
use tensorfs_core::dtype::Dtype;
use tensorfs_core::ensure::{self, Ensured, Event, Phase, Request};
use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::header::{Body, Closure, Header, Part, Tensor};
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::{Draft, Entry};
use tensorfs_core::registry;
use tensorfs_core::store::Store;
use tensorfs_core::transport::{Anonymous, PullCancellation, SourcePolicy};

/// The ledger's sampling resolution in these proofs. Every patience the transport derives
/// is a multiple of it, so the proofs judge by the production rule at test speed.
const SAMPLE: f64 = 0.05;

fn sample() -> Duration {
    Duration::from_secs_f64(SAMPLE)
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

fn temporary(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "tfs-fetch-faults-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------- a real checkpoint

struct Checkpoint {
    /// Every servable body by bare hex, manifest included.
    bodies: HashMap<String, Vec<u8>>,
    manifest: ObjectRef,
    /// The closure rows: header and blobs, sorted, manifest excluded.
    objects: Vec<ObjectRef>,
    /// The blobs in tensor order, for choosing a fault's target.
    blobs: Vec<String>,
}

impl Checkpoint {
    fn total(&self) -> u64 {
        self.bodies.values().map(|b| b.len() as u64).sum()
    }
}

fn checkpoint(blobs: usize, size: usize, salt: u8) -> Arc<Checkpoint> {
    let spec = registry::seeds()
        .into_iter()
        .find(|seed| seed.alias == "plain/1")
        .unwrap()
        .spec;
    let (mut bodies, mut tensors, mut order) = (HashMap::new(), Vec::new(), Vec::new());
    for index in 0..blobs {
        let body: Vec<u8> = (0..size)
            .map(|at| ((at * 31 + index * 7 + salt as usize) % 251) as u8)
            .collect();
        let object = ObjectRef::of(&body);
        let part = Part {
            dtype: Dtype::F32,
            shape: vec![(size / 4) as u64],
            body: Body::Segments(vec![object.clone()]),
        };
        tensors.push((
            format!("blocks.{index}.weight"),
            Tensor {
                dtype: Dtype::F32,
                shape: vec![(size / 4) as u64],
                encoding: spec.object_id(),
                parts: vec![("value".into(), part)],
            },
        ));
        order.push(object.sha256.clone());
        bodies.insert(object.sha256, body);
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
    let manifest = {
        let reference = ObjectRef::of(&manifest);
        bodies.insert(reference.sha256.clone(), manifest);
        reference
    };
    let mut objects: Vec<ObjectRef> = bodies
        .iter()
        .filter(|(hex, _)| **hex != manifest.sha256)
        .map(|(hex, body)| ObjectRef {
            sha256: hex.clone(),
            length: body.len() as u64,
        })
        .collect();
    objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    Arc::new(Checkpoint {
        bodies,
        manifest,
        objects,
        blobs: order,
    })
}

// ---------------------------------------------------------------- faults

/// What one matching ask is answered with instead of the honest answer.
#[derive(Clone, Debug)]
enum Fault {
    /// This status, these headers, this body.
    Status(u16, Vec<(String, String)>, String),
    /// After `n` body bytes, abort the connection with a TCP reset.
    Reset(u64),
    /// After `n` body bytes, go silent and hold the connection until the client leaves.
    Stall(u64),
    /// After `n` body bytes, close cleanly short of the declared length.
    Truncate(u64),
    /// The declared body, then these extra bytes on the same connection.
    Overlong(u64),
    /// The declared length with one byte changed.
    Corrupt,
    /// Wait this long, then answer honestly.
    Delay(Duration),
    /// The honest body spread evenly over this long: a slow link that keeps moving.
    Slow(Duration),
}

fn status(code: u16) -> Fault {
    Fault::Status(code, Vec::new(), String::new())
}

fn with_header(fault: Fault, name: &str, value: &str) -> Fault {
    let Fault::Status(code, mut headers, body) = fault else {
        panic!("headers ride on a status fault")
    };
    headers.push((name.into(), value.into()));
    Fault::Status(code, headers, body)
}

/// The hub's own typed refusal, as Tensorhub writes it.
fn enveloped(code: u16, error: &str) -> Fault {
    Fault::Status(
        code,
        vec![("content-type".into(), "application/json".into())],
        format!("{{\"error\":{{\"code\":\"{error}\",\"message\":\"injected\",\"remedy\":\"\"}}}}"),
    )
}

/// Which asks a fault answers: a route (`closure`, `presign`, an object's hex, or `*` for
/// every object) and the 1-based asks of that route it covers.
struct Rule {
    route: String,
    asks: std::ops::RangeInclusive<usize>,
    fault: Fault,
}

fn on(route: &str, asks: std::ops::RangeInclusive<usize>, fault: Fault) -> Rule {
    Rule {
        route: route.into(),
        asks,
        fault,
    }
}

// ---------------------------------------------------------------- the hub and the store

struct Asked {
    method: String,
    path: String,
    query: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Asked {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

fn read_request(socket: &mut TcpStream) -> Option<Asked> {
    let (mut collected, mut buf) = (Vec::new(), [0u8; 8192]);
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
    let mut line = lines.next()?.split_whitespace();
    let (method, target) = (line.next()?.to_string(), line.next()?.to_string());
    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
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
    Some(Asked {
        method,
        path: path.to_string(),
        query: query.to_string(),
        headers,
        body,
    })
}

#[derive(Default)]
struct Ledger {
    /// Asks per route.
    asks: HashMap<String, usize>,
    /// Body bytes written for each object's LATEST ask: what may honestly be in flight.
    sent: HashMap<String, u64>,
    /// URLs presented after they died.
    expired: usize,
    log: Vec<String>,
}

struct Hub {
    base: String,
    checkpoint: Arc<Checkpoint>,
    ledger: Arc<Mutex<Ledger>>,
}

impl Hub {
    fn asks(&self, route: &str) -> usize {
        *self.ledger.lock().unwrap().asks.get(route).unwrap_or(&0)
    }
    fn expired(&self) -> usize {
        self.ledger.lock().unwrap().expired
    }
    fn sent(&self, hex: &str) -> u64 {
        *self.ledger.lock().unwrap().sent.get(hex).unwrap_or(&0)
    }
    fn log(&self) -> String {
        self.ledger.lock().unwrap().log.join("\n")
    }
}

struct Config {
    rules: Vec<Rule>,
    /// How long a presigned URL lives.
    lifetime: Duration,
    /// The life the presign answer declares, when it is not the truth: a hub clock that
    /// disagrees with the object store's.
    declared: Option<Duration>,
    /// Closure rows per page; 0 answers the whole closure on one page.
    page: usize,
    /// The presign batch ceiling the closure declares.
    presign_max: u64,
    /// Closure rows beyond the checkpoint's own: an object the disk cannot hold.
    extra: Vec<ObjectRef>,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            rules: Vec::new(),
            lifetime: Duration::from_secs(3600),
            declared: None,
            page: 0,
            presign_max: 1024,
            extra: Vec::new(),
        }
    }
}

/// A hub (`closure`, `presign`) and an object store on one loopback listener. URLs carry the
/// instant they die and the store refuses them after it, as R2 refuses an aged signature.
fn hub(checkpoint: Arc<Checkpoint>, config: Config) -> Hub {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let ledger: Arc<Mutex<Ledger>> = Arc::default();
    let config = Arc::new(config);
    let (served, counted, origin) = (Arc::clone(&checkpoint), Arc::clone(&ledger), base.clone());
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(socket) = socket else { return };
            let (checkpoint, ledger, config, origin) = (
                Arc::clone(&served),
                Arc::clone(&counted),
                Arc::clone(&config),
                origin.clone(),
            );
            std::thread::spawn(move || serve(socket, &checkpoint, &ledger, &config, &origin));
        }
    });
    Hub {
        base,
        checkpoint,
        ledger,
    }
}

fn serve(
    mut socket: TcpStream,
    checkpoint: &Checkpoint,
    ledger: &Mutex<Ledger>,
    config: &Config,
    base: &str,
) {
    let Some(asked) = read_request(&mut socket) else {
        return;
    };
    let route = match (asked.method.as_str(), asked.path.as_str()) {
        ("POST", "/v1/tensorfs/closure") => "closure".to_string(),
        ("POST", "/v1/tensorfs/presign") => "presign".to_string(),
        ("GET", path) if path.starts_with("/obj/") => path["/obj/".len()..].to_string(),
        _ => return answer(&mut socket, 404, &[], b"no such route", None),
    };
    let object = route != "closure" && route != "presign";
    let (ask, fault) = {
        let mut ledger = ledger.lock().unwrap();
        let ask = {
            let count = ledger.asks.entry(route.clone()).or_insert(0);
            *count += 1;
            *count
        };
        let fault = config
            .rules
            .iter()
            .find(|rule| {
                (rule.route == route || (object && rule.route == "*")) && rule.asks.contains(&ask)
            })
            .map(|rule| rule.fault.clone());
        ledger.log.push(format!(
            "{:>6} ms {route} ask {ask} range {:?}: {fault:?}",
            now_ms() % 1_000_000,
            asked.header("range")
        ));
        if object {
            ledger.sent.insert(route.clone(), 0);
        }
        (ask, fault)
    };
    let _ = ask;
    if let Some(Fault::Delay(wait)) = fault {
        std::thread::sleep(wait);
    }
    if let Some(Fault::Status(code, headers, body)) = &fault {
        return answer(&mut socket, *code, headers, body.as_bytes(), None);
    }
    let body = match route.as_str() {
        "closure" => closure(checkpoint, &asked.body, config),
        "presign" => presign(&asked.body, base, config.lifetime, config.declared),
        hex => {
            let dies: u128 = asked
                .query
                .strip_prefix("dies=")
                .and_then(|stamp| stamp.parse().ok())
                .expect("every URL this hub mints carries its death");
            if now_ms() > dies {
                ledger.lock().unwrap().expired += 1;
                let denied = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>\
                              AccessDenied</Code><Message>Request has expired</Message></Error>";
                return answer(&mut socket, 403, &[], denied.as_bytes(), None);
            }
            match checkpoint.bodies.get(hex) {
                Some(body) => body.clone(),
                None => return answer(&mut socket, 404, &[], b"NoSuchKey", None),
            }
        }
    };
    let sent = object.then(|| (ledger, route.as_str()));
    let (code, body, range) = ranged(&asked, body);
    // A fault's offset is within the answer; a ranged answer may be shorter than it.
    let within = |at: u64| (at as usize).min(body.len());
    let headers = range
        .map(|r| vec![("content-range".to_string(), r)])
        .unwrap_or_default();
    match fault {
        None | Some(Fault::Delay(_)) => answer(&mut socket, code, &headers, &body, sent),
        Some(Fault::Corrupt) => {
            let mut wrong = body;
            let last = wrong.len() - 1;
            wrong[last] ^= 0xff;
            answer(&mut socket, code, &headers, &wrong, sent)
        }
        Some(Fault::Overlong(extra)) => {
            head(&mut socket, code, &headers, body.len() as u64);
            write_counted(&mut socket, &body, sent);
            let _ = socket.write_all(&vec![0x5a; extra as usize]);
        }
        Some(Fault::Truncate(at)) => {
            head(&mut socket, code, &headers, body.len() as u64);
            write_counted(&mut socket, &body[..within(at)], sent);
        }
        Some(Fault::Reset(at)) => {
            head(&mut socket, code, &headers, body.len() as u64);
            write_counted(&mut socket, &body[..within(at)], sent);
            reset(socket);
        }
        Some(Fault::Stall(at)) => {
            head(&mut socket, code, &headers, body.len() as u64);
            write_counted(&mut socket, &body[..within(at)], sent);
            let mut sink = [0u8; 64];
            while matches!(socket.read(&mut sink), Ok(n) if n > 0) {}
        }
        Some(Fault::Slow(over)) => {
            head(&mut socket, code, &headers, body.len() as u64);
            let pieces = (over.as_secs_f64() / (SAMPLE / 4.0)).ceil().max(1.0) as usize;
            for piece in body.chunks(body.len().div_ceil(pieces).max(1)) {
                std::thread::sleep(over / pieces as u32);
                if !write_counted(&mut socket, piece, sent) {
                    return;
                }
            }
        }
        Some(Fault::Status(..)) => unreachable!(),
    }
}

/// A `Range: bytes=a-b` or `bytes=a-` ask answered 206; anything else whole.
fn ranged(asked: &Asked, body: Vec<u8>) -> (u16, Vec<u8>, Option<String>) {
    let Some(spec) = asked.header("range").and_then(|r| r.strip_prefix("bytes=")) else {
        return (200, body, None);
    };
    let (start, end) = spec.split_once('-').unwrap();
    let start: usize = start.parse().unwrap();
    let end: usize = end
        .parse()
        .map_or(body.len() - 1, |e: usize| e.min(body.len() - 1));
    let range = format!("bytes {start}-{end}/{}", body.len());
    (206, body[start..=end].to_vec(), Some(range))
}

fn head(socket: &mut TcpStream, code: u16, headers: &[(String, String)], length: u64) {
    let mut text =
        format!("HTTP/1.1 {code} X\r\ncontent-length: {length}\r\nconnection: close\r\n");
    for (name, value) in headers {
        text.push_str(&format!("{name}: {value}\r\n"));
    }
    text.push_str("\r\n");
    let _ = socket.write_all(text.as_bytes());
}

fn answer(
    socket: &mut TcpStream,
    code: u16,
    headers: &[(String, String)],
    body: &[u8],
    sent: Option<(&Mutex<Ledger>, &str)>,
) {
    head(socket, code, headers, body.len() as u64);
    write_counted(socket, body, sent);
}

/// Write `bytes`, counting them against the object's latest ask BEFORE they leave, so the
/// count is never below what the client could have received.
fn write_counted(
    socket: &mut TcpStream,
    bytes: &[u8],
    sent: Option<(&Mutex<Ledger>, &str)>,
) -> bool {
    for piece in bytes.chunks(64 << 10) {
        if let Some((ledger, hex)) = sent {
            *ledger
                .lock()
                .unwrap()
                .sent
                .entry(hex.to_string())
                .or_insert(0) += piece.len() as u64;
        }
        if socket
            .write_all(piece)
            .and_then(|()| socket.flush())
            .is_err()
        {
            return false;
        }
    }
    true
}

/// Close with RST rather than FIN: SO_LINGER on with a zero timeout.
fn reset(socket: TcpStream) {
    rustix::net::sockopt::set_socket_linger(&socket, Some(Duration::ZERO)).unwrap();
    drop(socket);
}

fn field(body: &[u8], name: &str) -> Option<Value> {
    let Ok(Value::Obj(fields)) = canon::parse(body, 1 << 20) else {
        return None;
    };
    fields.into_iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn row(object: &ObjectRef) -> Value {
    Value::obj(vec![
        ("length", Value::uint(object.length)),
        ("sha256", Value::str(object.sha256.clone())),
    ])
}

fn closure(checkpoint: &Checkpoint, asked: &[u8], config: &Config) -> Vec<u8> {
    let after = match field(asked, "after") {
        Some(Value::Str(after)) => after,
        _ => String::new(),
    };
    let mut rest: Vec<&ObjectRef> = checkpoint
        .objects
        .iter()
        .chain(&config.extra)
        .filter(|o| o.sha256 > after)
        .collect();
    rest.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    let take = if config.page == 0 {
        rest.len()
    } else {
        config.page.min(rest.len())
    };
    canon::write(&Value::obj(vec![
        ("complete", Value::Bool(take == rest.len())),
        ("lane", Value::str("public")),
        ("manifest", row(&checkpoint.manifest)),
        ("model", Value::str("acme/model")),
        (
            "objects",
            Value::arr(rest[..take].iter().map(|o| row(o)).collect()),
        ),
        ("presign_max_digests", Value::uint(config.presign_max)),
        ("release", Value::str("r1")),
        ("scope", Value::str("runtime")),
    ]))
}

fn presign(asked: &[u8], base: &str, lifetime: Duration, declared: Option<Duration>) -> Vec<u8> {
    let Some(Value::Arr(digests)) = field(asked, "digests") else {
        panic!("a presign names its digests")
    };
    let dies = now_ms() + lifetime.as_millis();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let urls = digests
        .into_iter()
        .map(|digest| {
            let Value::Str(hex) = digest else { panic!() };
            let url = Value::str(format!("{base}/obj/{hex}?dies={dies}"));
            (hex, url)
        })
        .collect();
    canon::write(&Value::obj(vec![
        (
            "expires_at_unix",
            Value::uint(now + declared.unwrap_or(lifetime).as_secs()),
        ),
        ("server_time_unix", Value::uint(now)),
        ("urls", Value::map(urls)),
    ]))
}

// ---------------------------------------------------------------- ensure, watched

struct Outcome {
    result: Result<Ensured, Refusal>,
    events: Vec<Event>,
    /// Events whose `bytes_done` exceeded what the store held plus what the server had sent
    /// for the objects still missing.
    overclaims: Vec<String>,
}

struct Run<'a> {
    hub: &'a Hub,
    root: PathBuf,
    /// Silence the faults this run was dealt explain, beyond the ledger's own terms.
    allowance: Duration,
    cancellation: Option<PullCancellation>,
    streams: usize,
}

impl<'a> Run<'a> {
    fn new(hub: &'a Hub, root: &Path) -> Run<'a> {
        Run {
            hub,
            root: root.to_path_buf(),
            allowance: Duration::ZERO,
            cancellation: None,
            streams: 4,
        }
    }

    /// Run `ensure` on its own thread and watch its events. Fails the test only when the
    /// ensure stops making measured progress — never because a wall clock ran out.
    fn ensure(self) -> Outcome {
        // A hang guard, not a verdict: the ledger may legitimately learn a patience of
        // STALL_FACTOR times a slow ask's gap on a loaded box, so silence must outlast that.
        let quiet = sample() * 1200 + self.allowance;
        let (tx, rx) = mpsc::channel::<Result<Ensured, Refusal>>();
        let events: Arc<Mutex<Vec<(Instant, Event)>>> = Arc::default();
        let overclaims: Arc<Mutex<Vec<String>>> = Arc::default();
        let cancellation = self.cancellation.clone().unwrap_or_default();
        let (root, hub_base) = (self.root.clone(), self.hub.base.clone());
        let checkpoint = Arc::clone(&self.hub.checkpoint);
        let ledger = Arc::clone(&self.hub.ledger);
        let (seen, claimed, stop, streams) = (
            Arc::clone(&events),
            Arc::clone(&overclaims),
            cancellation.clone(),
            self.streams,
        );
        std::thread::spawn(move || {
            let store = Store::open(&root).unwrap();
            let policy = SourcePolicy {
                allowed_hosts: vec!["127.0.0.1".into()],
                allow_local: true,
                ..Default::default()
            };
            let on_event = |event: &Event| {
                if event.phase == Phase::Fetching && event.bytes_total > 0 {
                    let (held, flight) =
                        checkpoint.bodies.iter().fold((0, 0), |(h, f), (hex, b)| {
                            if store.contains(hex) || store.manifest_path(hex).is_file() {
                                (h + b.len() as u64, f)
                            } else {
                                let sent = *ledger.lock().unwrap().sent.get(hex).unwrap_or(&0);
                                (h, f + sent)
                            }
                        });
                    if event.bytes_done > held + flight {
                        claimed.lock().unwrap().push(format!(
                            "{} B claimed with {held} B resident and {flight} B in flight",
                            event.bytes_done
                        ));
                    }
                }
                seen.lock().unwrap().push((Instant::now(), event.clone()));
            };
            let mut request = Request::new(&store, &hub_base, "acme/model@r1", &Anonymous, &policy);
            request.lane = "public";
            request.step = "fault proof";
            request.cancellation = Some(stop);
            request.streams = streams;
            request.sample_seconds = SAMPLE;
            request.on_event = Some(&on_event);
            let _ = tx.send(ensure::ensure(&request));
        });
        let mut last = (Instant::now(), None::<(Phase, u64, u32)>);
        let result = loop {
            match rx.recv_timeout(sample()) {
                Ok(result) => break result,
                Err(mpsc::RecvTimeoutError::Disconnected) => panic!("ensure panicked"),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            let latest = events
                .lock()
                .unwrap()
                .last()
                .map(|(_, e)| (e.phase, e.bytes_done, e.attempt));
            if latest != last.1 {
                last = (Instant::now(), latest);
            } else if last.0.elapsed() > quiet {
                cancellation.cancel();
                panic!(
                    "ensure made no measured progress for {:?} (last {:?})\n{}",
                    last.0.elapsed(),
                    last.1,
                    self.hub.log()
                );
            }
        };
        let events = events.lock().unwrap().drain(..).map(|(_, e)| e).collect();
        let overclaims = overclaims.lock().unwrap().clone();
        Outcome {
            result,
            events,
            overclaims,
        }
    }
}

fn store(dir: &Path) -> PathBuf {
    let root = dir.join("store");
    Store::init(&root).unwrap();
    root
}

/// An object's or a manifest's resident bytes.
fn resident(store: &Store, hex: &str) -> Option<Vec<u8>> {
    std::fs::read(store.object_path(hex))
        .or_else(|_| std::fs::read(store.manifest_path(hex)))
        .ok()
}

/// No admission temp survives an ensure that has returned, whatever it returned.
fn temps(root: &Path) -> Vec<String> {
    std::fs::read_dir(root.join("tmp"))
        .map(|dir| {
            dir.filter_map(|e| e.ok()?.file_name().into_string().ok())
                .filter(|name| name.starts_with("put-"))
                .collect()
        })
        .unwrap_or_default()
}

/// The exact bytes resident, nothing stranded, and progress that never claimed more than
/// it had.
fn assert_landed(outcome: &Outcome, hub: &Hub, root: &Path) -> Ensured {
    let ensured = match &outcome.result {
        Ok(ensured) => ensured.clone(),
        Err(refusal) => panic!("refused {refusal}\n{}", hub.log()),
    };
    let checkpoint = &hub.checkpoint;
    assert_eq!(ensured.manifest, checkpoint.manifest);
    assert_eq!(ensured.bytes_total, checkpoint.total());
    let store = Store::open(root).unwrap();
    for (hex, body) in &checkpoint.bodies {
        assert!(
            resident(&store, hex).as_ref() == Some(body),
            "{hex} is not resident whole"
        );
    }
    assert_eq!(
        temps(root),
        Vec::<String>::new(),
        "a put- temp outlived the ensure"
    );
    assert_eq!(
        outcome.overclaims,
        Vec::<String>::new(),
        "progress claimed bytes it did not have"
    );
    let last = outcome.events.last().unwrap();
    assert_eq!(
        (last.bytes_done, last.bytes_total),
        (ensured.bytes_total, ensured.bytes_total)
    );
    ensured
}

fn assert_refused(outcome: &Outcome, hub: &Hub, root: &Path, code: Code) -> Refusal {
    let refusal = match &outcome.result {
        Err(refusal) => refusal.clone(),
        Ok(_) => panic!("landed but {code:?} was expected\n{}", hub.log()),
    };
    assert_eq!(refusal.code, code, "{refusal}\n{}", hub.log());
    assert_eq!(
        temps(root),
        Vec::<String>::new(),
        "a put- temp outlived the refusal"
    );
    assert_eq!(
        outcome.overclaims,
        Vec::<String>::new(),
        "progress claimed bytes it did not have"
    );
    refusal
}

fn blob(checkpoint: &Checkpoint, index: usize) -> String {
    checkpoint.blobs[index].clone()
}

// ---------------------------------------------------------------- the proofs: objects

#[test]
fn a_connection_reset_mid_object_costs_that_object_and_completes() {
    let dir = temporary("reset");
    let checkpoint = checkpoint(4, 256 << 10, 1);
    let victim = blob(&checkpoint, 1);
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(&victim, 1..=2, Fault::Reset(100 << 10))],
            ..Config::default()
        },
    );
    let root = store(&dir);
    let outcome = Run::new(&hub, &root).ensure();
    assert_landed(&outcome, &hub, &root);
    assert_eq!(hub.asks(&victim), 3, "{}", hub.log());
    for other in checkpoint.blobs.iter().filter(|hex| **hex != victim) {
        assert_eq!(
            hub.asks(other),
            1,
            "a reset of one object re-bought another"
        );
    }
}

#[test]
fn a_truncated_body_and_a_clean_close_short_of_length_are_re_asked() {
    let dir = temporary("truncate");
    let checkpoint = checkpoint(3, 256 << 10, 2);
    let victim = blob(&checkpoint, 0);
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![
                on(&victim, 1..=1, Fault::Truncate(255 << 10)),
                on(&victim, 2..=2, Fault::Truncate(0)),
            ],
            ..Config::default()
        },
    );
    let root = store(&dir);
    assert_landed(&Run::new(&hub, &root).ensure(), &hub, &root);
    assert_eq!(hub.asks(&victim), 3);
}

#[test]
fn a_stream_silent_mid_body_is_judged_by_the_ledger_and_re_asked() {
    let dir = temporary("stall");
    let checkpoint = checkpoint(3, 256 << 10, 3);
    let victim = blob(&checkpoint, 2);
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![
                on(&victim, 1..=1, Fault::Stall(0)),
                on(&victim, 2..=2, Fault::Stall(128 << 10)),
            ],
            ..Config::default()
        },
    );
    let root = store(&dir);
    assert_landed(&Run::new(&hub, &root).ensure(), &hub, &root);
    assert_eq!(hub.asks(&victim), 3);
}

#[test]
fn object_store_5xx_answers_are_weather_and_re_asked() {
    for code in [500, 502, 503, 504] {
        let dir = temporary("5xx");
        let checkpoint = checkpoint(2, 64 << 10, 4);
        let victim = blob(&checkpoint, 0);
        let hub = hub(
            Arc::clone(&checkpoint),
            Config {
                rules: vec![on(&victim, 1..=2, status(code))],
                ..Config::default()
            },
        );
        let root = store(&dir);
        assert_landed(&Run::new(&hub, &root).ensure(), &hub, &root);
        assert_eq!(hub.asks(&victim), 3, "HTTP {code}");
    }
}

#[test]
fn a_stated_retry_after_is_waited_out_in_either_spelling() {
    let date = {
        // RFC 9110 IMF-fixdate, two seconds out.
        let at = SystemTime::now() + Duration::from_secs(2);
        httpdate(at)
    };
    for (code, after) in [(429, "1".to_string()), (503, date)] {
        let dir = temporary("retry-after");
        let checkpoint = checkpoint(2, 64 << 10, 5);
        let victim = blob(&checkpoint, 1);
        let hub = hub(
            Arc::clone(&checkpoint),
            Config {
                rules: vec![on(
                    &victim,
                    1..=1,
                    with_header(status(code), "retry-after", &after),
                )],
                ..Config::default()
            },
        );
        let root = store(&dir);
        let mut run = Run::new(&hub, &root);
        run.allowance = Duration::from_secs(3);
        assert_landed(&run.ensure(), &hub, &root);
        assert_eq!(hub.asks(&victim), 2, "HTTP {code} retry-after {after}");
    }
}

fn httpdate(at: SystemTime) -> String {
    let secs = at.duration_since(UNIX_EPOCH).unwrap().as_secs();
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let weekday = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"][(days % 7) as usize];
    // Civil-from-days (Howard Hinnant).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let name = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][(month - 1) as usize];
    format!(
        "{weekday}, {day:02} {name} {year} {:02}:{:02}:{:02} GMT",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// A Retry-After no clock can hold is no statement: it must not panic the pull, and the
/// ask is judged like any unstated retryable answer.
#[test]
fn an_unrepresentable_retry_after_is_no_statement() {
    let dir = temporary("retry-after-overflow");
    let checkpoint = checkpoint(2, 64 << 10, 23);
    let victim = blob(&checkpoint, 0);
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(
                &victim,
                1..=1,
                with_header(status(503), "retry-after", &u64::MAX.to_string()),
            )],
            ..Config::default()
        },
    );
    let root = store(&dir);
    assert_landed(&Run::new(&hub, &root).ensure(), &hub, &root);
    assert_eq!(hub.asks(&victim), 2, "{}", hub.log());
}

/// URLs that die before their GET. The hub's declared life either tells the truth (the
/// client re-mints by age before asking) or disagrees with the object store's clock (the
/// client learns only from the 403). One stream, each first ask slower than a URL's life,
/// so later objects are asked for after their URLs died whatever the order.
#[test]
fn urls_that_die_before_their_get_are_re_minted_and_every_object_lands() {
    for declared in [None, Some(Duration::from_secs(3600))] {
        let dir = temporary("expiry");
        let checkpoint = checkpoint(3, 64 << 10, 6);
        let lifetime = Duration::from_secs(1);
        let hub = hub(
            Arc::clone(&checkpoint),
            Config {
                rules: vec![on("*", 1..=1, Fault::Slow(lifetime * 3 / 4))],
                lifetime,
                declared,
                ..Config::default()
            },
        );
        let root = store(&dir);
        let mut run = Run::new(&hub, &root);
        run.streams = 1;
        run.allowance = lifetime * 4;
        assert_landed(&run.ensure(), &hub, &root);
        assert!(
            hub.asks("presign") >= 2,
            "the dead URLs were never re-minted\n{}",
            hub.log()
        );
        if declared.is_some() {
            assert!(
                hub.expired() >= 1,
                "no ask ever met a dead URL\n{}",
                hub.log()
            );
        }
    }
}

#[test]
fn a_body_longer_than_its_length_never_reaches_the_store_as_more_bytes() {
    let dir = temporary("overlong");
    let checkpoint = checkpoint(2, 64 << 10, 7);
    let victim = blob(&checkpoint, 0);
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(&victim, 1..=1, Fault::Overlong(4096))],
            ..Config::default()
        },
    );
    let root = store(&dir);
    assert_landed(&Run::new(&hub, &root).ensure(), &hub, &root);
}

#[test]
fn wrong_bytes_once_are_re_asked_and_wrong_bytes_always_are_a_typed_verdict() {
    let dir = temporary("corrupt");
    let checkpoint = checkpoint(3, 64 << 10, 8);
    let victim = blob(&checkpoint, 2);
    let once = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(&victim, 1..=1, Fault::Corrupt)],
            ..Config::default()
        },
    );
    let root = store(&dir);
    let outcome = Run::new(&once, &root).ensure();
    // The door refuses the bytes; whether this build re-asks or stops, it never installs
    // them and never claims them.
    match &outcome.result {
        Ok(_) => {
            assert_landed(&outcome, &once, &root);
        }
        Err(_) => {
            assert_refused(&outcome, &once, &root, Code::OBJECT_ID_MISMATCH);
            assert!(!Store::open(&root).unwrap().contains(&victim));
        }
    }

    let always = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(&victim, 1..=usize::MAX, Fault::Corrupt)],
            ..Config::default()
        },
    );
    let root = store(&temporary("corrupt-always"));
    let outcome = Run::new(&always, &root).ensure();
    let refusal = assert_refused(&outcome, &always, &root, Code::OBJECT_ID_MISMATCH);
    assert!(!ensure::resumable(refusal.code));
    assert!(!Store::open(&root).unwrap().contains(&victim));
    assert!(always.asks(&victim) <= tensorfs_core::transport::FETCH_ATTEMPTS as usize);
}

/// A blip is asked through: the pull asks again, a sample apart, until the link answers.
/// STALLED needs the ledger's floor of no landing.
#[test]
fn a_link_blip_before_any_object_lands_is_re_attempted_and_only_stillness_stalls() {
    let checkpoint = checkpoint(3, 64 << 10, 20);
    let blip = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on("*", 1..=4, Fault::Reset(0))],
            ..Config::default()
        },
    );
    let root = store(&temporary("blip"));
    // Nothing counts asks: the pull itself asks again and the first attempt lands it all.
    let ensured = assert_landed(&Run::new(&blip, &root).ensure(), &blip, &root);
    assert_eq!(ensured.attempts, 1, "{}", blip.log());

    let down = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on("*", 1..=usize::MAX, Fault::Reset(0))],
            ..Config::default()
        },
    );
    let root = store(&temporary("down"));
    let started = Instant::now();
    let outcome = Run::new(&down, &root).ensure();
    assert_refused(&outcome, &down, &root, Code::STALLED);
    let floor = sample() * tensorfs_core::transport::STILL_SAMPLES;
    assert!(
        started.elapsed() >= floor,
        "STALLED after {:?}",
        started.elapsed()
    );
}

/// A pull killed mid-object strands its partial temp; the next ensure takes it, because GC,
/// the only other reaper, never runs while a model is leased.
#[test]
fn a_killed_pulls_partial_temp_is_reaped_by_the_next_ensure() {
    let checkpoint = checkpoint(2, 1 << 20, 21);
    let victim = blob(&checkpoint, 0);
    let stuck = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(&victim, 1..=usize::MAX, Fault::Stall(512 << 10))],
            ..Config::default()
        },
    );
    let root = store(&temporary("killed"));
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_tfs"))
        .args(["ensure", root.to_str().unwrap(), "acme/model@r1"])
        .args([
            "--hub",
            &stuck.base,
            "--lane",
            "public",
            "--sample-seconds",
            "60",
        ])
        .env("TFS_CREDENTIAL", "")
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let partial = |root: &Path| {
        temps(root).iter().any(|name| {
            std::fs::metadata(root.join("tmp").join(name)).is_ok_and(|m| m.len() >= 512 << 10)
        })
    };
    while !partial(&root) {
        assert!(
            child.try_wait().unwrap().is_none(),
            "tfs ensure exited early"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(!temps(&root).is_empty());

    let healed = hub(Arc::clone(&checkpoint), Config::default());
    assert_landed(&Run::new(&healed, &root).ensure(), &healed, &root);
}

/// An ensure counts what a cancelled pull already landed as held. When its admission is
/// short and GC runs, those objects, unreferenced until a pull is recorded, must survive:
/// collecting them would re-buy them past the space the pull was admitted for.
#[test]
fn objects_a_flight_holds_survive_the_gc_its_admission_runs() {
    let checkpoint = checkpoint(3, 256 << 10, 22);
    let stuck = blob(&checkpoint, 2);
    let first = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(&stuck, 1..=usize::MAX, Fault::Stall(0))],
            ..Config::default()
        },
    );
    let root = store(&temporary("held-gc"));
    let cancellation = PullCancellation::default();
    let (cancel, landed_root) = (cancellation.clone(), root.clone());
    let landed: Vec<String> = checkpoint.blobs[..2].to_vec();
    let watched = landed.clone();
    std::thread::spawn(move || {
        let store = Store::open(&landed_root).unwrap();
        while !watched.iter().all(|hex| store.contains(hex)) {
            std::thread::sleep(sample());
        }
        cancel.cancel();
    });
    let mut run = Run::new(&first, &root);
    run.cancellation = Some(cancellation);
    assert!(run.ensure().result.is_err());

    // The same checkpoint with one more object than any disk holds: admission collects.
    let short = hub(
        Arc::clone(&checkpoint),
        Config {
            extra: vec![ObjectRef {
                sha256: "f".repeat(64),
                length: 1 << 50,
            }],
            ..Config::default()
        },
    );
    let outcome = Run::new(&short, &root).ensure();
    assert_refused(&outcome, &short, &root, Code::CAPACITY_EXHAUSTED);
    let store = Store::open(&root).unwrap();
    for hex in &landed {
        assert!(store.contains(hex), "GC took {hex}, which the flight held");
    }

    let healed = hub(Arc::clone(&checkpoint), Config::default());
    assert_landed(&Run::new(&healed, &root).ensure(), &healed, &root);
    assert!(
        landed.iter().all(|hex| healed.asks(hex) == 0),
        "{}",
        healed.log()
    );
}

/// A follower whose leader has not yet written its record (still planning or admitting)
/// still knows the closure's size: its bar is never 0/0.
#[test]
fn a_follower_waiting_on_a_record_less_leader_reports_the_closures_size() {
    use fs2::FileExt;
    let checkpoint = checkpoint(2, 64 << 10, 24);
    let hub = hub(Arc::clone(&checkpoint), Config::default());
    let root = store(&temporary("record-less-leader"));
    let flights = root.join("tmp/ensure");
    std::fs::create_dir_all(&flights).unwrap();
    let lock = std::fs::File::create(flights.join(format!("{}.lock", checkpoint.manifest.sha256)))
        .unwrap();
    lock.lock_exclusive().unwrap();
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(2));
        drop(lock);
    });
    let outcome = Run::new(&hub, &root).ensure();
    release.join().unwrap();
    assert_landed(&outcome, &hub, &root);
    let waiting: Vec<&Event> = outcome
        .events
        .iter()
        .filter(|e| e.phase == Phase::Waiting)
        .collect();
    assert!(!waiting.is_empty(), "the follower never waited");
    assert!(
        waiting.iter().all(|e| e.bytes_total == checkpoint.total()),
        "{waiting:?}"
    );
}

#[test]
fn a_missing_object_is_a_typed_refusal_and_the_rest_stay_resident() {
    let dir = temporary("missing");
    let checkpoint = checkpoint(3, 64 << 10, 9);
    let victim = blob(&checkpoint, 1);
    let broken = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(
                &victim,
                1..=usize::MAX,
                Fault::Status(
                    404,
                    Vec::new(),
                    "<Error><Code>NoSuchKey</Code></Error>".into(),
                ),
            )],
            ..Config::default()
        },
    );
    let root = store(&dir);
    let outcome = Run::new(&broken, &root).ensure();
    let refusal = outcome
        .result
        .as_ref()
        .err()
        .expect("an absent object cannot land");
    assert!(
        refusal.detail.contains("404"),
        "{refusal}\n{}",
        broken.log()
    );
    assert_eq!(temps(&root), Vec::<String>::new());
    assert_eq!(outcome.overclaims, Vec::<String>::new());
    assert!(!Store::open(&root).unwrap().contains(&victim));
    // What did land stays landed: a second ensure against a healed origin buys only the rest.
    let healed = hub(Arc::clone(&checkpoint), Config::default());
    let resident: Vec<String> = checkpoint
        .blobs
        .iter()
        .filter(|hex| Store::open(&root).unwrap().contains(hex))
        .cloned()
        .collect();
    assert_landed(&Run::new(&healed, &root).ensure(), &healed, &root);
    assert!(
        resident.iter().all(|hex| healed.asks(hex) == 0),
        "{}",
        healed.log()
    );
    eprintln!("absent object: {refusal}");
}

// ---------------------------------------------------------------- the proofs: the hub

#[test]
fn a_hub_that_resets_or_answers_without_its_envelope_is_weather_and_re_asked() {
    let mut failed = Vec::new();
    for fault in [
        Fault::Reset(0),
        status(502),
        Fault::Status(404, Vec::new(), "ERR_NGROK_3200 tunnel not found".into()),
        enveloped(503, "catalog.unavailable"),
    ] {
        for route in ["closure", "presign"] {
            let dir = temporary("hub-weather");
            let checkpoint = checkpoint(2, 64 << 10, 10);
            let hub = hub(
                Arc::clone(&checkpoint),
                Config {
                    rules: vec![on(route, 1..=1, fault.clone())],
                    ..Config::default()
                },
            );
            let root = store(&dir);
            let outcome = Run::new(&hub, &root).ensure();
            match &outcome.result {
                Ok(_) => drop(assert_landed(&outcome, &hub, &root)),
                Err(refusal) => failed.push(format!("{route} {fault:?}: {refusal}")),
            }
        }
    }
    assert!(
        failed.is_empty(),
        "weather was taken for a verdict:\n{}",
        failed.join("\n")
    );
}

/// A hub that keeps saying "unavailable" through every re-ask is still weather: the pull
/// that already landed objects is re-attempted, never ended by a verdict the hub never gave.
#[test]
fn a_hub_unavailable_through_every_re_ask_is_weather_not_a_verdict() {
    let dir = temporary("hub-exhausted");
    let checkpoint = checkpoint(3, 64 << 10, 19);
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on("presign", 2..=5, enveloped(503, "tensorfs.unavailable"))],
            presign_max: 1,
            ..Config::default()
        },
    );
    let root = store(&dir);
    let ensured = assert_landed(&Run::new(&hub, &root).ensure(), &hub, &root);
    assert!(ensured.attempts >= 2, "{}", hub.log());
}

#[test]
fn a_hub_that_says_wait_is_waited_for() {
    let dir = temporary("hub-429");
    let checkpoint = checkpoint(2, 64 << 10, 11);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let body = format!(
        "{{\"error\":{{\"code\":\"rate_limited\",\"message\":\"slow down\",\"remedy\":\"\"}},\
         \"retry_after_unix\":{},\"server_time_unix\":{now}}}",
        now + 1
    );
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on("closure", 1..=1, Fault::Status(429, Vec::new(), body))],
            ..Config::default()
        },
    );
    let root = store(&dir);
    let mut run = Run::new(&hub, &root);
    run.allowance = Duration::from_secs(2);
    assert_landed(&run.ensure(), &hub, &root);
    assert_eq!(hub.asks("closure"), 2);
}

/// A hub is entitled to be slow: before the first byte the pull has measured nothing, so
/// no silence rule may condemn a call that is merely taking its time.
#[test]
fn a_slow_hub_is_waited_for_not_timed_out() {
    let dir = temporary("hub-slow");
    let checkpoint = checkpoint(2, 64 << 10, 18);
    let slow = sample() * tensorfs_core::transport::STILL_SAMPLES * 4;
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![
                on("closure", 1..=1, Fault::Delay(slow)),
                on("presign", 1..=1, Fault::Delay(slow)),
            ],
            ..Config::default()
        },
    );
    let root = store(&dir);
    let mut run = Run::new(&hub, &root);
    run.allowance = slow * 2;
    assert_landed(&run.ensure(), &hub, &root);
    assert_eq!(
        (hub.asks("closure"), hub.asks("presign")),
        (1, 1),
        "{}",
        hub.log()
    );
}

/// A hub that has not answered is waited for (no timer on a live hub's answer), but the
/// wait is visible, one Resolving event a tick, and the caller's cancellation ends it.
#[test]
fn a_hub_that_never_answers_is_a_visible_wait_that_cancellation_ends() {
    let dir = temporary("hub-mute");
    let checkpoint = checkpoint(1, 4096, 25);
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(
                "closure",
                1..=usize::MAX,
                Fault::Delay(Duration::from_secs(3600)),
            )],
            ..Config::default()
        },
    );
    let root = store(&dir);
    let cancellation = PullCancellation::default();
    let cancelled: Arc<Mutex<Option<Instant>>> = Arc::default();
    let (cancel, at) = (cancellation.clone(), Arc::clone(&cancelled));
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs_f64(ensure::EVENT_SECONDS * 3.5));
        *at.lock().unwrap() = Some(Instant::now());
        cancel.cancel();
    });
    let mut run = Run::new(&hub, &root);
    run.cancellation = Some(cancellation);
    let outcome = run.ensure();
    let waited = cancelled.lock().unwrap().expect("cancelled").elapsed();
    assert!(outcome.result.is_err());
    let resolving = outcome
        .events
        .iter()
        .filter(|e| e.phase == Phase::Resolving)
        .count();
    assert!(resolving >= 3, "{resolving} Resolving events in 3.5 ticks");
    let floor = sample() * tensorfs_core::transport::STILL_SAMPLES;
    assert!(waited < floor, "cancellation took {waited:?}");
}

#[test]
fn the_hubs_own_verdicts_are_typed_and_terminal() {
    for (status, error, code) in [
        (404, "release.not_found", Code::REF_NOT_FOUND),
        (403, "model.forbidden", Code::CREDENTIAL_REQUIRED),
    ] {
        let dir = temporary("hub-verdict");
        let checkpoint = checkpoint(1, 4096, 12);
        let hub = hub(
            Arc::clone(&checkpoint),
            Config {
                rules: vec![on("closure", 1..=usize::MAX, enveloped(status, error))],
                ..Config::default()
            },
        );
        let root = store(&dir);
        let outcome = Run::new(&hub, &root).ensure();
        let refusal = assert_refused(&outcome, &hub, &root, code);
        assert!(refusal.detail.contains(error), "{refusal}");
        assert_eq!(hub.asks("closure"), 1, "a verdict was re-asked");
    }
}

#[test]
fn a_hub_that_never_answers_its_envelope_is_unreachable_never_not_found() {
    let dir = temporary("hub-down");
    let checkpoint = checkpoint(1, 4096, 13);
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(
                "closure",
                1..=usize::MAX,
                Fault::Status(404, Vec::new(), "<html>tunnel offline</html>".into()),
            )],
            ..Config::default()
        },
    );
    let root = store(&dir);
    let outcome = Run::new(&hub, &root).ensure();
    let refusal = assert_refused(&outcome, &hub, &root, Code::HUB_UNREACHABLE);
    assert!(ensure::resumable(refusal.code));
}

#[test]
fn a_closure_paged_across_many_asks_lands_whole() {
    let dir = temporary("pages");
    let checkpoint = checkpoint(40, 4096, 14);
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on("closure", 3..=3, status(502))],
            page: 7,
            ..Config::default()
        },
    );
    let root = store(&dir);
    assert_landed(&Run::new(&hub, &root).ensure(), &hub, &root);
    assert!(hub.asks("closure") >= 7, "{}", hub.log());
}

// ---------------------------------------------------------------- cancellation and resume

/// Run 2322: a prefetch paused by real demand 0.2 s in still returned 30.6 s later, the
/// ledger's whole floor, because the hub's retry wait slept through its cancellation.
#[test]
fn cancellation_ends_a_hub_wait_within_a_sample_not_after_it() {
    let dir = temporary("cancel-hub-wait");
    let checkpoint = checkpoint(1, 4096, 15);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let wait = format!(
        "{{\"error\":{{\"code\":\"rate_limited\",\"message\":\"slow down\",\"remedy\":\"\"}},\
         \"retry_after_unix\":{},\"server_time_unix\":{now}}}",
        now + 30
    );
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(
                "closure",
                1..=usize::MAX,
                Fault::Status(429, Vec::new(), wait),
            )],
            ..Config::default()
        },
    );
    let root = store(&dir);
    let cancellation = PullCancellation::default();
    let cancelled: Arc<Mutex<Option<Instant>>> = Arc::default();
    let (cancel, at, ledger) = (
        cancellation.clone(),
        Arc::clone(&cancelled),
        Arc::clone(&hub.ledger),
    );
    std::thread::spawn(move || {
        while !ledger.lock().unwrap().asks.contains_key("closure") {
            std::thread::sleep(Duration::from_millis(1));
        }
        std::thread::sleep(sample());
        *at.lock().unwrap() = Some(Instant::now());
        cancel.cancel();
    });
    let mut run = Run::new(&hub, &root);
    run.cancellation = Some(cancellation);
    run.allowance = Duration::from_secs(30);
    let outcome = run.ensure();
    let waited = cancelled.lock().unwrap().expect("cancelled").elapsed();
    assert!(outcome.result.is_err());
    // A cancelled wait costs at most a sample of resolution, never the wait itself.
    let floor = sample() * tensorfs_core::transport::STILL_SAMPLES;
    assert!(
        waited < floor,
        "cancellation took {waited:?}\n{}",
        hub.log()
    );
    assert_eq!(hub.asks("closure"), 1, "a cancelled ensure asked again");
}

#[test]
fn a_cancelled_pull_resumes_with_a_new_token_and_buys_only_what_is_missing() {
    let dir = temporary("cancel-resume");
    let checkpoint = checkpoint(4, 128 << 10, 16);
    let stuck = blob(&checkpoint, 3);
    let first = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(&stuck, 1..=usize::MAX, Fault::Stall(0))],
            ..Config::default()
        },
    );
    let root = store(&dir);
    let cancellation = PullCancellation::default();
    let (cancel, landed_root) = (cancellation.clone(), root.clone());
    let others: Vec<String> = checkpoint
        .blobs
        .iter()
        .filter(|h| **h != stuck)
        .cloned()
        .collect();
    std::thread::spawn(move || {
        let store = Store::open(&landed_root).unwrap();
        while !others.iter().all(|hex| store.contains(hex)) {
            std::thread::sleep(sample());
        }
        cancel.cancel();
    });
    let mut run = Run::new(&first, &root);
    run.cancellation = Some(cancellation);
    let outcome = run.ensure();
    assert!(outcome.result.is_err());
    assert_eq!(temps(&root), Vec::<String>::new());

    let second = hub(Arc::clone(&checkpoint), Config::default());
    let ensured = assert_landed(&Run::new(&second, &root).ensure(), &second, &root);
    assert_eq!(second.asks(&stuck), 1);
    for other in checkpoint.blobs.iter().filter(|hex| **hex != stuck) {
        assert_eq!(
            second.asks(other),
            0,
            "a resumed ensure re-bought a resident object"
        );
    }
    assert!(ensured.bytes_held > 0);
}

#[test]
fn a_failed_asks_bytes_are_kept_and_the_next_ask_is_for_the_rest() {
    let dir = temporary("resume");
    let checkpoint = checkpoint(2, 1 << 20, 17);
    let victim = blob(&checkpoint, 0);
    // The first answer dies at 90%. Those bytes stay, the second ask is for the last 10%,
    // and the progress `assert_landed` checks never passes the total on the way.
    let hub = hub(
        Arc::clone(&checkpoint),
        Config {
            rules: vec![on(&victim, 1..=1, Fault::Truncate(900 << 10))],
            ..Config::default()
        },
    );
    let root = store(&dir);
    assert_landed(&Run::new(&hub, &root).ensure(), &hub, &root);
    assert_eq!(hub.asks(&victim), 2, "{}", hub.log());
    assert_eq!(
        hub.sent(&victim),
        (1 << 20) - (900 << 10),
        "the second ask sent more than the rest"
    );
}
