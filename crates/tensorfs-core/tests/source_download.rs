//! Source ingest against a real loopback origin: an HF-like `resolve` redirect onto a
//! ranged, keep-alive delivery host, and an R2-like PUT endpoint. Nothing is mocked.
use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use tensorfs_core::{
    err::Code,
    ids::ObjectRef,
    providers::{self, Provenance, Pulled, Resolution, ResolvedMember},
    store::{Fault, Store},
    transport::{
        memory_available, push_object, streams_within, transfer_streams, Anonymous, Deadline,
        Ledger, PullCancellation, SourceDownload, SourcePolicy, UploadGrant, STREAM_MEMORY,
    },
};

const MIB: usize = 1 << 20;

fn root(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "tfs-download-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}
fn owner(label: &str) -> String {
    ObjectRef::of(label.as_bytes()).id()
}
fn body(seed: u64, length: usize) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

/// What the origin does with one request, decided per request.
enum Act {
    Serve,
    /// Answer this status with these headers and a short body.
    Status(u16, Vec<(String, String)>),
    /// Send the head and this many body bytes, then drop the connection.
    Cut(usize),
    /// Send the body at this many bytes a second.
    Crawl(u64),
    /// Read a PUT body, then say nothing, ever.
    Silent,
}

struct Request {
    method: String,
    path: String,
    range: Option<(u64, u64)>,
}

type Decide = dyn Fn(&Request, usize) -> Act + Send + Sync;

struct Origin {
    base: String,
    objects: HashMap<String, Arc<Vec<u8>>>,
    /// Body bytes the delivery host wrote for ranged answers, and the ranges asked.
    served: AtomicU64,
    asked: Mutex<Vec<(String, u64, u64)>>,
    asks: AtomicUsize,
    connections: AtomicUsize,
    live: AtomicUsize,
    most_live: AtomicUsize,
    puts: AtomicUsize,
    /// Per-connection pace, bytes per second; 0 is unpaced.
    pace: u64,
    decide: Box<Decide>,
    stop: AtomicBool,
}

impl Origin {
    fn start(
        objects: Vec<(&str, Vec<u8>)>,
        pace: u64,
        decide: impl Fn(&Request, usize) -> Act + Send + Sync + 'static,
    ) -> Arc<Origin> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = Arc::new(Origin {
            base: format!("http://{}", listener.local_addr().unwrap()),
            objects: objects
                .into_iter()
                .map(|(name, bytes)| (name.to_string(), Arc::new(bytes)))
                .collect(),
            served: AtomicU64::new(0),
            asked: Mutex::new(vec![]),
            asks: AtomicUsize::new(0),
            connections: AtomicUsize::new(0),
            live: AtomicUsize::new(0),
            most_live: AtomicUsize::new(0),
            puts: AtomicUsize::new(0),
            pace,
            decide: Box::new(decide),
            stop: AtomicBool::new(false),
        });
        let serving = origin.clone();
        thread::spawn(move || {
            for socket in listener.incoming() {
                if serving.stop.load(Ordering::Relaxed) {
                    return;
                }
                let Ok(socket) = socket else { return };
                serving.connections.fetch_add(1, Ordering::Relaxed);
                let origin = serving.clone();
                thread::spawn(move || origin.connection(socket));
            }
        });
        origin
    }

    fn resolution(&self, members: &[&str]) -> Resolution {
        Resolution {
            canonical: "hf://fixture/model@revision".into(),
            selection_sha256: "accepted".into(),
            members: members
                .iter()
                .map(|name| ResolvedMember {
                    member: name.to_string(),
                    object: ObjectRef::of(&self.objects[*name]),
                    url: format!("{}/resolve/{name}", self.base),
                    provenance: Provenance::Declared,
                    carrier: true,
                    requires: vec![],
                    companion: false,
                })
                .collect(),
        }
    }

    fn connection(&self, mut socket: TcpStream) {
        let _ = socket.set_read_timeout(Some(Duration::from_secs(60)));
        loop {
            let mut head = Vec::new();
            let mut byte = [0u8];
            while !head.ends_with(b"\r\n\r\n") {
                if !matches!(socket.read(&mut byte), Ok(1)) {
                    return;
                }
                head.push(byte[0]);
            }
            let text = String::from_utf8_lossy(&head).to_string();
            let mut lines = text.lines();
            let mut first = lines.next().unwrap().split_whitespace();
            let method = first.next().unwrap().to_string();
            let target = first.next().unwrap().to_string();
            let headers: Vec<(String, String)> = lines
                .filter_map(|line| line.split_once(':'))
                .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                .collect();
            let header = |name: &str| {
                headers
                    .iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v.clone())
            };
            let range = header("range").and_then(|value| {
                let (a, b) = value.strip_prefix("bytes=")?.split_once('-')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            });
            let request = Request {
                method,
                path: target.split('?').next().unwrap().to_string(),
                range,
            };
            let number = self.asks.fetch_add(1, Ordering::Relaxed) + 1;
            let act = (self.decide)(&request, number);
            if request.method == "PUT" {
                let length: u64 = header("content-length").unwrap().parse().unwrap();
                // Discarded as it arrives: this origin's memory is not the client's.
                let taken = std::io::copy(&mut (&mut socket).take(length), &mut std::io::sink());
                if taken.ok() != Some(length) {
                    return;
                }
                self.puts.fetch_add(1, Ordering::Relaxed);
                match act {
                    Act::Silent => {
                        // Hold the socket until the client gives up on it.
                        let _ = socket.read(&mut [0u8; 1]);
                        return;
                    }
                    Act::Status(status, extra) => {
                        if !reply(&mut socket, status, &extra, b"retry") {
                            return;
                        }
                    }
                    _ => {
                        if !reply(&mut socket, 200, &[], b"") {
                            return;
                        }
                    }
                }
                continue;
            }
            if let Act::Status(status, extra) = act {
                if !reply(&mut socket, status, &extra, b"slow down") {
                    return;
                }
                continue;
            }
            if let Some(name) = request.path.strip_prefix("/resolve/") {
                // HuggingFace answers with a small body; draining it keeps the socket.
                let location = format!("/cdn/{name}?sig={number}");
                if !reply(
                    &mut socket,
                    302,
                    &[("location".into(), location)],
                    b"Found. Redirecting",
                ) {
                    return;
                }
                continue;
            }
            let name = request.path.strip_prefix("/cdn/").unwrap();
            let bytes = self.objects[name].clone();
            let (a, b) = request.range.unwrap_or((0, bytes.len() as u64 - 1));
            self.asked.lock().unwrap().push((name.into(), a, b));
            let slice = &bytes[a as usize..=b as usize];
            let head = match request.range {
                Some(_) => format!(
                    "HTTP/1.1 206 Partial Content\r\ncontent-range: bytes {a}-{b}/{}\r\ncontent-length: {}\r\n\r\n",
                    bytes.len(),
                    slice.len()
                ),
                None => format!("HTTP/1.1 200 OK\r\ncontent-length: {}\r\n\r\n", slice.len()),
            };
            if socket.write_all(head.as_bytes()).is_err() {
                return;
            }
            let live = self.live.fetch_add(1, Ordering::SeqCst) + 1;
            self.most_live.fetch_max(live, Ordering::SeqCst);
            let cut = match act {
                Act::Cut(n) => Some(n.min(slice.len())),
                _ => None,
            };
            let pace = match act {
                Act::Crawl(rate) => rate,
                _ => self.pace,
            };
            let complete = self.send(&mut socket, &slice[..cut.unwrap_or(slice.len())], pace);
            self.live.fetch_sub(1, Ordering::SeqCst);
            if !complete || cut.is_some() {
                let _ = socket.shutdown(std::net::Shutdown::Both);
                return;
            }
        }
    }

    /// Paced on an absolute schedule so a loaded box does not compound sleep overshoot.
    fn send(&self, socket: &mut TcpStream, bytes: &[u8], pace: u64) -> bool {
        let started = Instant::now();
        let mut sent = 0usize;
        for chunk in bytes.chunks(if pace > 0 && pace < 64 << 10 {
            4 << 10
        } else {
            64 << 10
        }) {
            if pace > 0 {
                let due = started + Duration::from_secs_f64(sent as f64 / pace as f64);
                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                    thread::sleep(wait);
                }
            }
            if socket.write_all(chunk).is_err() {
                return false;
            }
            sent += chunk.len();
            self.served.fetch_add(chunk.len() as u64, Ordering::Relaxed);
        }
        true
    }

    fn requested(&self) -> u64 {
        self.asked
            .lock()
            .unwrap()
            .iter()
            .map(|(_, a, b)| b - a + 1)
            .sum()
    }
}
impl Drop for Origin {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = TcpStream::connect(self.base.trim_start_matches("http://"));
    }
}

fn reply(socket: &mut TcpStream, status: u16, extra: &[(String, String)], body: &[u8]) -> bool {
    let mut head = format!("HTTP/1.1 {status} X\r\ncontent-length: {}\r\n", body.len());
    for (name, value) in extra {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    socket
        .write_all(head.as_bytes())
        .and_then(|()| socket.write_all(body))
        .is_ok()
}

fn policy() -> SourcePolicy {
    SourcePolicy {
        allowed_hosts: vec!["127.0.0.1".into()],
        allow_local: true,
        max_redirects: 5,
        ..Default::default()
    }
}

fn materialize(
    store: &Store,
    owner: &str,
    resolution: &Resolution,
    options: &SourceDownload,
    progress: &(dyn Fn(u64, u64) + Sync),
) -> Result<Vec<Pulled>, tensorfs_core::err::Refusal> {
    providers::materialize_selected_with_options(
        store,
        owner,
        resolution,
        &policy(),
        &Anonymous,
        Deadline::none(),
        progress,
        options,
    )
    .map(|(root, rows)| {
        assert!(root.complete);
        rows
    })
}

fn verify(store: &Store, origin: &Origin, members: &[&str]) {
    for name in members {
        let bytes = &origin.objects[*name];
        let object = ObjectRef::of(bytes);
        assert!(
            store.record_valid(&object.sha256).is_ok(),
            "{name} not admitted"
        );
        assert_eq!(
            &store.read_range(&object.sha256, 0, object.length).unwrap(),
            bytes.as_ref()
        );
    }
}

fn options(checkpoint: usize, streams: usize) -> SourceDownload {
    SourceDownload {
        checkpoint_bytes: checkpoint as u64,
        streams,
        ..Default::default()
    }
}

#[test]
fn members_download_concurrently_once_each_and_report_progress() {
    let names = [
        "a.safetensors",
        "b.safetensors",
        "c.safetensors",
        "config.json",
    ];
    let objects = vec![
        (names[0], body(1, 3 * MIB + 5)),
        (names[1], body(2, 3 * MIB)),
        (names[2], body(3, 2 * MIB + 7)),
        (names[3], body(4, 911)),
    ];
    let total: u64 = objects.iter().map(|(_, bytes)| bytes.len() as u64).sum();
    // Paced so every member is still in flight when the next one starts.
    let origin = Origin::start(objects, 8 << 20, |_, _| Act::Serve);
    let dir = root("concurrent");
    let store = Store::init(&dir).unwrap();
    let reports = Mutex::new(vec![]);
    let rows = materialize(
        &store,
        &owner("concurrent"),
        &origin.resolution(&names),
        // Chunks larger than any member: each member is one ask.
        &options(8 * MIB, 16),
        &|present, whole| reports.lock().unwrap().push((present, whole)),
    )
    .unwrap();
    verify(&store, &origin, &names);
    assert_eq!(rows.iter().map(|row| row.transferred).sum::<u64>(), total);
    assert_eq!(origin.requested(), total, "a byte was asked for twice");
    assert_eq!(origin.asked.lock().unwrap().len(), names.len());
    assert!(
        origin.most_live.load(Ordering::SeqCst) >= 3,
        "members were fetched serially"
    );
    let reports = reports.into_inner().unwrap();
    assert_eq!(reports.first(), Some(&(0, total)));
    assert_eq!(reports.last(), Some(&(total, total)));
    assert!(reports.windows(2).all(|pair| pair[0].0 <= pair[1].0));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn one_large_member_is_asked_in_chunks_across_streams_and_every_byte_lands_once() {
    let name = "model.safetensors";
    let bytes = body(7, 24 * MIB + 3);
    let length = bytes.len() as u64;
    let origin = Origin::start(vec![(name, bytes)], 12 << 20, |_, _| Act::Serve);
    let dir = root("split");
    let store = Store::init(&dir).unwrap();
    let rows = materialize(
        &store,
        &owner("split"),
        &origin.resolution(&[name]),
        &options(MIB, 8),
        &|_, _| {},
    )
    .unwrap();
    verify(&store, &origin, &[name]);
    assert_eq!(rows[0].transferred, length, "a byte was written twice");
    assert!(
        origin.most_live.load(Ordering::SeqCst) >= 4,
        "the member was not split"
    );
    // Every chunk asks the delivery URL the source resolved to; the source is walked once.
    let resolves = origin.asks.load(Ordering::Relaxed) - origin.asked.lock().unwrap().len();
    assert_eq!(resolves, 1, "the redirect was re-walked per ask");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_connection_cut_mid_body_resumes_at_the_byte_it_reached() {
    let name = "model.safetensors";
    let bytes = body(9, 5 * MIB + 1);
    let length = bytes.len() as u64;
    // The first delivery answer dies 2.5 MiB in; the next ask must start there.
    let origin = Origin::start(vec![(name, bytes)], 0, |request, _| match request.range {
        Some((0, _)) if request.path.starts_with("/cdn/") => Act::Cut(5 * MIB / 2),
        _ => Act::Serve,
    });
    let dir = root("cut");
    let store = Store::init(&dir).unwrap();
    let rows = materialize(
        &store,
        &owner("cut"),
        &origin.resolution(&[name]),
        // One chunk: the member is one GET, and its resume one more.
        &options(8 * MIB, 1),
        &|_, _| {},
    )
    .unwrap();
    verify(&store, &origin, &[name]);
    assert_eq!(rows[0].transferred, length);
    let asked = origin.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 2, "{asked:?}");
    assert_eq!(asked[1].1, 5 * MIB as u64 / 2);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn download_child() {
    let Some(dir) = std::env::var_os("TFS_DOWNLOAD_CHILD") else {
        return;
    };
    let base = std::env::var("TFS_DOWNLOAD_ORIGIN").unwrap();
    let names: Vec<String> = std::env::var("TFS_DOWNLOAD_MEMBERS")
        .unwrap()
        .split(',')
        .map(String::from)
        .collect();
    let store = Store::open(std::path::Path::new(&dir)).unwrap();
    let resolution = Resolution {
        canonical: "hf://fixture/model@revision".into(),
        selection_sha256: "accepted".into(),
        members: names
            .iter()
            .enumerate()
            .map(|(seed, name)| ResolvedMember {
                member: name.clone(),
                object: ObjectRef::of(&body(seed as u64 + 20, 8 * MIB + seed)),
                url: format!("{base}/resolve/{name}"),
                provenance: Provenance::Declared,
                carrier: true,
                requires: vec![],
                companion: false,
            })
            .collect(),
    };
    let _ = materialize(
        &store,
        &owner("killed"),
        &resolution,
        &options(MIB, 16),
        &|_, _| {},
    );
}

/// Recorded chunks, read from disk: (object, first, length). A line a kill cut short is
/// not a record.
fn retained(dir: &std::path::Path) -> Vec<(String, u64, u64)> {
    let area = dir
        .join("roots/source-downloads")
        .join(&owner("killed")[7..]);
    let mut chunks = vec![];
    for object in fs::read_dir(area).into_iter().flatten() {
        let object = object.unwrap().path();
        let name = object.file_name().unwrap().to_string_lossy().to_string();
        let text = fs::read_to_string(object.join("chunks")).unwrap_or_default();
        for line in text.split_inclusive('\n').skip(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if line.ends_with('\n') && fields.len() == 3 {
                let (first, end): (u64, u64) =
                    (fields[0].parse().unwrap(), fields[1].parse().unwrap());
                chunks.push((name.clone(), first, end - first));
            }
        }
    }
    chunks
}

#[test]
fn a_killed_download_fetches_only_what_it_had_not_retained() {
    let names = ["a.safetensors", "b.safetensors", "c.safetensors"];
    let objects: Vec<(&str, Vec<u8>)> = names
        .iter()
        .enumerate()
        .map(|(seed, name)| (*name, body(seed as u64 + 20, 8 * MIB + seed)))
        .collect();
    let total: u64 = objects.iter().map(|(_, bytes)| bytes.len() as u64).sum();
    let origin = Origin::start(objects.clone(), 1 << 20, |_, _| Act::Serve);
    let dir = root("killed");
    let store = Store::init(&dir).unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "download_child", "--nocapture"])
        .env("TFS_DOWNLOAD_CHILD", &dir)
        .env("TFS_DOWNLOAD_ORIGIN", &origin.base)
        .env("TFS_DOWNLOAD_MEMBERS", names.join(","))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // SIGKILL once a whole chunk past an object's first is recorded and a good share has
    // crossed.
    let later_retained = || {
        retained(&dir)
            .iter()
            .any(|(_, first, length)| *first > 0 && *length == MIB as u64)
    };
    while origin.served.load(Ordering::Relaxed) < total / 3 || !later_retained() {
        assert!(
            child.try_wait().unwrap().is_none(),
            "the child finished before the kill"
        );
        thread::sleep(Duration::from_millis(5));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    let runs = retained(&dir);
    assert!(
        runs.iter().any(|(_, first, _)| *first > 0),
        "no later chunk was retained: {runs:?}"
    );
    // What is still owed: every member the child did not finish, less its retained runs.
    let mut owed = 0;
    let mut kept = 0;
    for (_, bytes) in &objects {
        let object = ObjectRef::of(bytes);
        if store.record_valid(&object.sha256).is_ok() {
            continue;
        }
        let held: u64 = runs
            .iter()
            .filter(|(sha, _, _)| *sha == object.sha256)
            .map(|(_, _, extent)| extent)
            .sum();
        kept += held;
        owed += object.length - held;
    }
    assert!(kept >= MIB as u64, "nothing was retained: {runs:?}");

    let resumed = Origin::start(objects, 0, |_, _| Act::Serve);
    let rows = materialize(
        &store,
        &owner("killed"),
        &resumed.resolution(&names),
        // Large chunks: each missing range is one ask, so asks are exactly the remainder.
        &options(1 << 30, 16),
        &|_, _| {},
    )
    .unwrap();
    verify(&store, &resumed, &names);
    assert_eq!(resumed.requested(), owed);
    assert_eq!(rows.iter().map(|row| row.transferred).sum::<u64>(), owed);
    fs::remove_dir_all(dir).unwrap();
}

/// Run 2322's shape at a source: the first ask that reaches one MiB of a member crawls at
/// 16 KiB/s while the rest come home at once. That request is restarted; the ingest does
/// not wait out the crawl (64 s for the MiB alone).
#[test]
fn a_crawling_request_is_restarted_and_does_not_hold_the_ingest() {
    const RATE: u64 = 16 << 10;
    let name = "model.safetensors";
    let bytes = body(15, 24 * MIB + 5);
    let crawled = AtomicBool::new(false);
    let origin = Origin::start(vec![(name, bytes)], 0, move |request, _| {
        let reaches = request
            .range
            .is_some_and(|(a, b)| a <= 8 * MIB as u64 && 8 * MIB as u64 <= b);
        match request.path.starts_with("/cdn/") && reaches && !crawled.swap(true, Ordering::SeqCst)
        {
            true => Act::Crawl(RATE),
            false => Act::Serve,
        }
    });
    let dir = root("crawl");
    let store = Store::init(&dir).unwrap();
    let started = Instant::now();
    let rows = materialize(
        &store,
        &owner("crawl"),
        &origin.resolution(&[name]),
        &options(MIB, 4),
        &|_, _| {},
    )
    .unwrap();
    let took = started.elapsed();
    verify(&store, &origin, &[name]);
    assert!(rows[0].transferred >= 24 * MIB as u64 + 5);
    let crawl = Duration::from_secs_f64(MIB as f64 / RATE as f64);
    assert!(
        took < crawl / 4,
        "the ingest waited out the crawl: {took:?}"
    );
    let log = fs::read_to_string(dir.join("logs/transport.log")).unwrap();
    assert!(log.contains(" restart "), "{log}");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_stated_retry_after_is_waited_out_and_spends_no_attempt() {
    let name = "model.safetensors";
    let bytes = body(11, MIB + 9);
    let refusals = Arc::new(AtomicUsize::new(0));
    let counted = refusals.clone();
    // Four stated refusals in a row: as many as there are attempts.
    let origin = Origin::start(vec![(name, bytes)], 0, move |request, _| {
        if request.path.starts_with("/resolve/") && counted.fetch_add(1, Ordering::SeqCst) < 4 {
            Act::Status(429, vec![("retry-after".into(), "1".into())])
        } else {
            Act::Serve
        }
    });
    let dir = root("retry-after");
    let store = Store::init(&dir).unwrap();
    let started = Instant::now();
    materialize(
        &store,
        &owner("retry-after"),
        &origin.resolution(&[name]),
        &options(MIB, 4),
        &|_, _| {},
    )
    .unwrap();
    verify(&store, &origin, &[name]);
    assert!(
        started.elapsed() >= Duration::from_secs(4),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(refusals.load(Ordering::SeqCst), 5);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_caller_cancellation_stops_the_download_and_keeps_retained_bytes() {
    let name = "model.safetensors";
    let bytes = body(13, 8 * MIB);
    let origin = Origin::start(vec![(name, bytes)], 8 << 20, |_, _| Act::Serve);
    let dir = root("cancel");
    let store = Store::init(&dir).unwrap();
    let cancellation = PullCancellation::default();
    let stop = cancellation.clone();
    let cancellable = SourceDownload {
        cancellation: Some(cancellation),
        ..options(MIB, 2)
    };
    let refusal = materialize(
        &store,
        &owner("cancel"),
        &origin.resolution(&[name]),
        &cancellable,
        &|present, _| {
            if present >= 2 * MIB as u64 {
                stop.cancel();
            }
        },
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::TRANSFER_FAILED);
    assert!(refusal.detail.contains("cancelled"), "{refusal}");
    let asked = origin.requested();
    let rows = materialize(
        &store,
        &owner("cancel"),
        &origin.resolution(&[name]),
        &options(1 << 30, 2),
        &|_, _| {},
    )
    .unwrap();
    verify(&store, &origin, &[name]);
    assert!(
        rows[0].transferred <= 6 * MIB as u64,
        "retained bytes were fetched again"
    );
    assert!(origin.requested() > asked);
    fs::remove_dir_all(dir).unwrap();
}

fn pushed_store(name: &str) -> (std::path::PathBuf, Store, ObjectRef) {
    let dir = root(name);
    let store = Store::init(&dir).unwrap();
    let bytes = body(17, 256 << 10);
    let object = ObjectRef::of(&bytes);
    store
        .put_stream(&mut bytes.as_slice(), Some(&object), &Fault::default())
        .unwrap();
    (dir, store, object)
}

#[test]
fn a_push_whose_store_goes_silent_after_the_body_is_judged_not_waited_on() {
    let (dir, store, object) = pushed_store("push-silent");
    let origin = Origin::start(vec![], 0, |request, _| {
        if request.method == "PUT" {
            Act::Silent
        } else {
            Act::Serve
        }
    });
    let ledger = Ledger::with_resolution(Duration::from_millis(20));
    let started = Instant::now();
    let refusal = push_object(
        &store,
        &object.sha256,
        false,
        &UploadGrant::to(&format!("{}/put/object", origin.base)),
        &policy(),
        &Anonymous,
        Deadline::none(),
        &ledger,
        "",
    )
    .unwrap_err();
    assert_eq!(refusal.code, Code::TRANSFER_FAILED);
    assert!(refusal.detail.contains("response head"), "{refusal}");
    assert_eq!(
        origin.puts.load(Ordering::Relaxed),
        4,
        "each attempt sent the whole body"
    );
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "{:?}",
        started.elapsed()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_push_told_to_wait_waits_and_then_lands() {
    let (dir, store, object) = pushed_store("push-wait");
    let origin = Origin::start(vec![], 0, |_, number| {
        if number <= 4 {
            Act::Status(503, vec![("retry-after".into(), "1".into())])
        } else {
            Act::Serve
        }
    });
    let started = Instant::now();
    let pushed = push_object(
        &store,
        &object.sha256,
        false,
        &UploadGrant::to(&format!("{}/put/object", origin.base)),
        &policy(),
        &Anonymous,
        Deadline::none(),
        &Ledger::with_resolution(Duration::from_millis(20)),
        "",
    )
    .unwrap();
    assert_eq!(pushed.http_status, 200);
    assert_eq!(origin.puts.load(Ordering::Relaxed), 5);
    assert!(started.elapsed() >= Duration::from_secs(4));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn concurrent_admissions_queue_instead_of_refusing() {
    let dir = root("admissions");
    let store = Store::init(&dir).unwrap();
    thread::scope(|scope| {
        for worker in 0..16u64 {
            let store = &store;
            scope.spawn(move || {
                for index in 0..4 {
                    let bytes = body(1000 + worker * 4 + index, 64 << 10);
                    let object = ObjectRef::of(&bytes);
                    store
                        .put_stream(&mut bytes.as_slice(), Some(&object), &Fault::default())
                        .unwrap();
                    assert!(store.record_valid(&object.sha256).is_ok());
                }
            });
        }
    });
    fs::remove_dir_all(dir).unwrap();
}

/// Peak resident memory since `/proc/self/clear_refs` was last reset, in bytes.
fn peak_resident() -> u64 {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let kib: u64 = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .unwrap()
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .unwrap();
    kib * 1024
}

/// Run one `*_child` test of this binary in its own process, optionally inside `scope`.
fn run_child(test: &str, dir: &std::path::Path, scope: &[&str]) -> std::process::Output {
    let exe = std::env::current_exe().unwrap();
    let mut command = match scope.split_first() {
        Some((program, args)) => {
            let mut command = Command::new(program);
            command.args(args).arg(&exe);
            command
        }
        None => Command::new(&exe),
    };
    command
        .args(["--exact", test, "--nocapture"])
        .env("TFS_DOWNLOAD_CHILD", dir)
        .output()
        .unwrap()
}

#[test]
fn push_memory_child() {
    let Some(dir) = std::env::var_os("TFS_DOWNLOAD_CHILD") else {
        return;
    };
    let store = Store::init(std::path::Path::new(&dir)).unwrap();
    let objects: Vec<ObjectRef> = (0..16u64)
        .map(|seed| {
            let bytes = body(300 + seed, 16 * MIB);
            let object = ObjectRef::of(&bytes);
            store
                .put_stream(&mut bytes.as_slice(), Some(&object), &Fault::default())
                .unwrap();
            object
        })
        .collect();
    let origin = Origin::start(vec![], 0, |_, _| Act::Serve);
    // Forget the objects' construction; measure only the pushes.
    fs::write("/proc/self/clear_refs", "5").unwrap();
    let before = peak_resident();
    thread::scope(|scope| {
        for object in &objects {
            let (store, origin) = (&store, &origin);
            scope.spawn(move || {
                push_object(
                    store,
                    &object.sha256,
                    false,
                    &UploadGrant::to(&format!("{}/put/{}", origin.base, object.sha256)),
                    &policy(),
                    &Anonymous,
                    Deadline::none(),
                    &Ledger::with_resolution(Duration::from_millis(200)),
                    "",
                )
                .unwrap();
            });
        }
    });
    let grew = peak_resident().saturating_sub(before);
    println!("PUSHED 16 x 16 MiB, peak resident grew {grew} B");
    assert_eq!(origin.puts.load(Ordering::Relaxed), 16);
    // 256 MiB crossed the sockets; the pushes may hold their streams' buffers and no more.
    assert!(
        grew < 16 * STREAM_MEMORY + (16 << 20),
        "pushes buffered {grew} B"
    );
}

#[test]
fn sixteen_concurrent_pushes_hold_stream_buffers_not_objects() {
    let dir = root("push-memory");
    let output = run_child("push_memory_child", &dir, &[]);
    let text = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success() && text.contains("PUSHED"), "{text}");
    println!(
        "{}",
        text.lines().find(|line| line.contains("PUSHED")).unwrap()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn streams_are_sized_to_the_memory_budget() {
    assert_eq!(streams_within(16, None), 16);
    assert_eq!(streams_within(16, Some(1 << 40)), 16);
    assert_eq!(streams_within(16, Some(5 * STREAM_MEMORY)), 5);
    assert_eq!(
        streams_within(16, Some(0)),
        1,
        "a starved host still moves one stream"
    );
    assert!(memory_available().is_some_and(|bytes| bytes > 0));
}

#[test]
fn bounded_memory_child() {
    let Some(dir) = std::env::var_os("TFS_DOWNLOAD_CHILD") else {
        return;
    };
    let available = memory_available().unwrap();
    let streams = transfer_streams(1024);
    println!("BUDGET {available} B allows {streams} streams");
    // The scope's 64 MiB, not the host's gigabytes, decides.
    assert!(
        available < 64 << 20 && streams < 32,
        "{available} B, {streams} streams"
    );
    let name = "model.safetensors";
    let origin = Origin::start(vec![(name, body(77, 16 * MIB))], 16 << 20, |_, _| {
        Act::Serve
    });
    let store = Store::init(std::path::Path::new(&dir)).unwrap();
    materialize(
        &store,
        &owner("bounded"),
        &origin.resolution(&[name]),
        &options(MIB, 1024),
        &|_, _| {},
    )
    .unwrap();
    verify(&store, &origin, &[name]);
    let live = origin.most_live.load(Ordering::SeqCst);
    println!("DOWNLOADED with at most {live} live streams");
    assert!(
        live <= streams,
        "{live} streams against a budget of {streams}"
    );
}

#[test]
fn a_memory_limited_scope_downloads_with_fewer_streams() {
    let scope = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        "MemoryMax=64M",
    ];
    let usable = Command::new(scope[0])
        .args(&scope[1..])
        .arg("true")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !usable {
        eprintln!("SKIPPED: no systemd user manager to create a memory-limited cgroup");
        return;
    }
    let dir = root("bounded-memory");
    let output = run_child("bounded_memory_child", &dir, &scope);
    let text = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && text.contains("DOWNLOADED"),
        "{text}"
    );
    for line in text
        .lines()
        .filter(|line| line.contains("BUDGET") || line.contains("DOWNLOADED"))
    {
        println!("{line}");
    }
    let _ = fs::remove_dir_all(dir);
}

/// Throughput on a paced loopback origin (each connection capped like one CDN stream).
/// `cargo test --test source_download -- --ignored --nocapture throughput`
#[test]
#[ignore]
fn throughput() {
    let names = [
        "a.safetensors",
        "b.safetensors",
        "c.safetensors",
        "d.safetensors",
    ];
    let objects: Vec<(&str, Vec<u8>)> = names
        .iter()
        .enumerate()
        .map(|(seed, name)| (*name, body(seed as u64 + 40, 48 * MIB)))
        .collect();
    let total: u64 = objects.iter().map(|(_, bytes)| bytes.len() as u64).sum();
    for streams in [1, 16] {
        let origin = Origin::start(objects.clone(), 32 << 20, |_, _| Act::Serve);
        let dir = root("throughput");
        let store = Store::init(&dir).unwrap();
        let started = Instant::now();
        materialize(
            &store,
            &owner("throughput"),
            &origin.resolution(&names),
            &options(8 * MIB, streams),
            &|_, _| {},
        )
        .unwrap();
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "streams={streams}: {total} B in {seconds:.2} s = {:.1} MB/s, {} asks, {} connections",
            total as f64 / seconds / 1e6,
            origin.asks.load(Ordering::Relaxed),
            origin.connections.load(Ordering::Relaxed),
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
