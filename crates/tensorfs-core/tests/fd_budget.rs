//! Run 1297 (2026-09-27): a pull growing toward 128 streams under a 1,024 soft
//! RLIMIT_NOFILE ran out of descriptors mid-pull. This binary sets the limit itself and
//! downloads with as many requests as the table allows. Its own binary, because
//! RLIMIT_NOFILE is process-wide.
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rustix::process::{getrlimit, setrlimit, Resource, Rlimit};
use tensorfs_core::fetch::FetchPlan;
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::Store;
use tensorfs_core::transport::{
    fetch_wanted, Anonymous, Deadline, Ledger, SourcePolicy, WalkOutcome, PULL_STREAMS,
};

const SIZE: usize = 512 << 10;
const CHUNK: usize = 16 << 10;
const GAP: Duration = Duration::from_millis(10);

/// A keep-alive HTTP/1.1 object store that serves each connection at ~1.6 MB/s, so only
/// more streams buy more bytes. It counts connections and requests separately.
struct Origin {
    base: String,
    connections: Arc<AtomicUsize>,
    requests: Arc<AtomicUsize>,
}

fn origin(bodies: Arc<HashMap<String, Vec<u8>>>) -> Origin {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let connections = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(AtomicUsize::new(0));
    let (accepted, asked) = (Arc::clone(&connections), Arc::clone(&requests));
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(socket) = socket else { break };
            accepted.fetch_add(1, Ordering::Relaxed);
            let (bodies, asked) = (Arc::clone(&bodies), Arc::clone(&asked));
            std::thread::spawn(move || serve(socket, &bodies, &asked));
        }
    });
    Origin {
        base,
        connections,
        requests,
    }
}

fn serve(mut socket: TcpStream, bodies: &HashMap<String, Vec<u8>>, asked: &AtomicUsize) {
    let mut pending = Vec::new();
    loop {
        let head_end = loop {
            if let Some(at) = pending.windows(4).position(|w| w == b"\r\n\r\n") {
                break at + 4;
            }
            let mut buf = [0u8; 4096];
            match socket.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => pending.extend_from_slice(&buf[..n]),
            }
        };
        let head = String::from_utf8_lossy(&pending[..head_end]).into_owned();
        pending.drain(..head_end);
        asked.fetch_add(1, Ordering::Relaxed);
        let path = head.split_whitespace().nth(1).unwrap_or("");
        let Some(body) = bodies.get(path.trim_start_matches("/obj/")) else {
            let _ = socket.write_all(b"HTTP/1.1 404 X\r\ncontent-length: 0\r\n\r\n");
            continue;
        };
        let answer = format!("HTTP/1.1 200 OK\r\ncontent-length: {}\r\n\r\n", body.len());
        if socket.write_all(answer.as_bytes()).is_err() {
            return;
        }
        let start = std::time::Instant::now();
        for (index, piece) in body.chunks(CHUNK).enumerate() {
            let due = start + GAP * (index as u32 + 1);
            let now = std::time::Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
            if socket.write_all(piece).is_err() {
                return;
            }
        }
    }
}

fn objects(count: usize) -> (Vec<ObjectRef>, Arc<HashMap<String, Vec<u8>>>) {
    let mut x: u64 = 0x2545f4914f6cdd1d;
    let mut bodies = HashMap::new();
    let mut declared = Vec::new();
    for _ in 0..count {
        let body: Vec<u8> = (0..SIZE)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect();
        let object = ObjectRef::of(&body);
        bodies.insert(object.sha256.clone(), body);
        declared.push(object);
    }
    declared.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    (declared, Arc::new(bodies))
}

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-fd-budget-{name}-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}

/// Walk everything at the default ceiling; every object is re-hashed from the Store.
fn walk(origin: &Origin, declared: &[ObjectRef]) -> WalkOutcome {
    let root = temporary("walk");
    let store = Store::init(&root).unwrap();
    let (plan, _) = FetchPlan::of_objects(&store, "fd-budget", declared).unwrap();
    let urls: BTreeMap<String, String> = declared
        .iter()
        .map(|o| {
            (
                o.sha256.clone(),
                format!("{}/obj/{}", origin.base, o.sha256),
            )
        })
        .collect();
    let policy = SourcePolicy {
        allowed_hosts: vec!["127.0.0.1".into()],
        allow_local: true,
        ..Default::default()
    };
    let walked = fetch_wanted(
        &store,
        &plan,
        &urls,
        &policy,
        &Anonymous,
        Deadline::after_seconds(Some(300.0)),
        &Ledger::with_resolution(Duration::from_secs(1)),
        PULL_STREAMS,
        None,
    )
    .unwrap_or_else(|refusal| panic!("the walk refused: {refusal}"));
    plan.complete(&store).unwrap();
    for object in declared {
        let mut bytes = Vec::new();
        store
            .open_verified(&object.sha256)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(
            ObjectRef::of(&bytes),
            *object,
            "the door admitted other bytes"
        );
    }
    let _ = std::fs::remove_dir_all(root);
    assert_eq!(walked.fetched as usize, declared.len());
    walked
}

fn set_limit(soft: u64, hard: Option<u64>) {
    setrlimit(
        Resource::Nofile,
        Rlimit {
            current: Some(soft),
            maximum: hard,
        },
    )
    .unwrap();
}

#[test]
fn a_low_descriptor_limit_bounds_the_walk_and_never_exhausts_it() {
    let original = getrlimit(Resource::Nofile);
    let hard = original.maximum.unwrap_or(u64::MAX);
    assert!(
        hard >= 4096,
        "this proof needs a hard limit of at least 4096"
    );
    // More objects a walk than requests in flight, so a connection has a second request.
    let (declared, bodies) = objects(3 * 640);
    let origin = origin(bodies);

    // A Go parent hands its child the original soft limit. The pull raises it to the hard
    // limit and grows as far as the link rewards, with no descriptor error.
    set_limit(256, original.maximum);
    let before = tensorfs_core::descriptors::exhaustions();
    let raised = walk(&origin, &declared[..640]);
    assert!(
        getrlimit(Resource::Nofile).current.unwrap() > 256,
        "the soft limit was not raised"
    );
    // The raised table no longer bounds the walk: the budget it measures now admits the
    // whole ceiling.
    assert_eq!(
        tensorfs_core::descriptors::streams_within(PULL_STREAMS),
        PULL_STREAMS,
        "a raised table still capped the budget: {raised:?}"
    );
    assert_eq!(
        tensorfs_core::descriptors::exhaustions(),
        before,
        "EMFILE after the raise"
    );
    let (connections, requests) = (
        origin.connections.load(Ordering::Relaxed),
        origin.requests.load(Ordering::Relaxed),
    );
    assert!(
        connections < requests,
        "every object opened its own connection ({connections} for {requests} requests)"
    );

    // The table shrinks under a running walk. The squeeze is driven by the walk's own
    // progress, never by a clock: once the origin has served 16 more GETs, the soft limit
    // drops just below what is open, and it comes back after the walk has met a full
    // table three times. EMFILE is then backpressure: the request is asked again and
    // every byte lands.
    let requests = Arc::clone(&origin.requests);
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished = Arc::clone(&done);
    let squeeze = std::thread::spawn(move || {
        let start = requests.load(Ordering::Relaxed);
        while requests.load(Ordering::Relaxed) < start + 16 && !finished.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let open = std::fs::read_dir("/proc/self/fd").unwrap().count() as u64;
        let pressed = tensorfs_core::descriptors::exhaustions();
        set_limit(open.saturating_sub(4), original.maximum);
        while tensorfs_core::descriptors::exhaustions() < pressed + 3
            && !finished.load(Ordering::Relaxed)
        {
            std::thread::sleep(Duration::from_millis(1));
        }
        set_limit(hard, original.maximum);
        open
    });
    let pressed = walk(&origin, &declared[640..1280]);
    done.store(true, Ordering::Relaxed);
    let open = squeeze.join().unwrap();
    println!("pressed (limit {open}-4 mid-walk): {pressed:?}");
    assert!(
        tensorfs_core::descriptors::exhaustions() > before && pressed.requests > pressed.fetched,
        "the squeeze never pressed the walk: {pressed:?}"
    );

    // A hard limit of 256 cannot be raised past. The walk sizes itself to the table and
    // completes with every object verified. The test's own origin shares this process's
    // table, so a transient EMFILE is possible and is absorbed as backpressure; the walk
    // never fails on one and never grows past the budget.
    set_limit(256, Some(256));
    let bounded = walk(&origin, &declared[1280..]);
    let budget = tensorfs_core::descriptors::streams_within(PULL_STREAMS);
    println!("raised: {raised:?}\nbounded to {budget}: {bounded:?}");
    assert!(
        bounded.streams <= budget.max(1),
        "grew past the descriptor budget: {bounded:?}"
    );
}
