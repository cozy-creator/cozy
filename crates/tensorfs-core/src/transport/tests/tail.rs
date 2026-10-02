//! The tail of a hub pull against an origin that answers like a cold R2 object: most of an
//! object's MiBs come at once, and some crawl for whoever asks them first while a fresh ask
//! for the same bytes is answered at once (measured on a pod, 2026-10-01). A whole GET
//! pays every slow MiB in a row: run 2322, sasori, dietfried and chieri ended on a few
//! objects at 30-180 KB/s for minutes after a 130-700 MB/s body.

use super::super::CHUNK_BYTES;
use super::*;
use std::time::Instant;

const MIB: usize = 1 << 20;
const OBJECT: usize = 16 * MIB;
/// What a slow MiB's first ask is answered at: 8 s for the MiB.
const CRAWL: u64 = 128 * 1024;
const DRIP: usize = 4096;

struct Cold {
    origin: Origin,
    /// Asks answered with a range, and asks that reached a slow MiB first and crawled.
    ranged: Arc<AtomicUsize>,
    crawled: Arc<AtomicUsize>,
    /// The first byte of every ranged ask.
    firsts: Arc<Mutex<Vec<usize>>>,
}

/// A hub whose store honours ranges. Every fourth MiB of each `slow` object crawls for the
/// first ask that reaches it and is answered at once to every later one.
fn cold_hub(checkpoint: Arc<Checkpoint>, slow: Vec<String>) -> Cold {
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let (ranged, crawled) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let firsts: Arc<Mutex<Vec<usize>>> = Arc::default();
    let touched: Arc<Mutex<HashMap<(String, usize), usize>>> = Arc::default();
    let (slot, ranges, crawls, starts) = (
        Arc::clone(&base_slot),
        Arc::clone(&ranged),
        Arc::clone(&crawled),
        Arc::clone(&firsts),
    );
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = slot.lock().unwrap().clone();
        let path = request.path.as_str();
        match request.method.as_str() {
            "POST" if path == "/v1/tensorfs/closure" => {
                return ok_json(&closure_value(&checkpoint, &base))
            }
            "POST" if path == "/v1/tensorfs/presign" => {
                return ok_json(&presign_value(&request.body, &base))
            }
            "GET" if path.starts_with("/obj/") => {}
            _ => return status(404, b"no such route"),
        }
        let hex = path["/obj/".len()..].to_string();
        let Some(body) = checkpoint.bodies.get(&hex).cloned() else {
            return status(404, b"no such object");
        };
        let asked = range_of(request);
        if let Some((first, _)) = asked {
            ranges.fetch_add(1, Ordering::Relaxed);
            starts.lock().unwrap().push(first);
        }
        if !slow.contains(&hex) {
            return partial(&body, request);
        }
        let (first, last) = asked.map_or((0, body.len() - 1), |(first, last)| {
            (first, last.min(body.len() - 1))
        });
        let head = match asked {
            Some(_) => format!(
                "HTTP/1.1 206 X\r\ncontent-length: {}\r\ncontent-range: bytes {first}-{last}/{}\r\n\
                 connection: close\r\n\r\n",
                last + 1 - first,
                body.len()
            ),
            None => format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            ),
        };
        let (touched, crawls) = (Arc::clone(&touched), Arc::clone(&crawls));
        Serve::Script(Box::new(move |socket| {
            if socket.write_all(head.as_bytes()).is_err() {
                return;
            }
            let mut at = first;
            while at <= last {
                let mib = at / MIB;
                let end = ((mib + 1) * MIB - 1).min(last);
                let first_ask = {
                    let mut touched = touched.lock().unwrap();
                    let asks = touched.entry((hex.clone(), mib)).or_insert(0);
                    *asks += 1;
                    *asks == 1
                };
                if mib % 4 == 1 && first_ask {
                    crawls.fetch_add(1, Ordering::Relaxed);
                    for piece in body[at..=end].chunks(DRIP) {
                        std::thread::sleep(Duration::from_secs_f64(DRIP as f64 / CRAWL as f64));
                        if socket.write_all(piece).is_err() {
                            return;
                        }
                    }
                } else if socket.write_all(&body[at..=end]).is_err() {
                    return;
                }
                at = end + 1;
            }
        }))
    });
    let origin = origin_with_hits(dispatch, Arc::default());
    *base_slot.lock().unwrap() = origin.base.clone();
    Cold {
        origin,
        ranged,
        crawled,
        firsts,
    }
}

/// The pull's wall time, its report, and the Store's transport log.
fn timed_pull(
    hub: &Cold,
    checkpoint: &Checkpoint,
    name: &str,
) -> (Duration, crate::transport::PullReport, String) {
    timed_pull_sampling(hub, checkpoint, name, 0.025)
}

fn timed_pull_sampling(
    hub: &Cold,
    checkpoint: &Checkpoint,
    name: &str,
    sample_seconds: f64,
) -> (Duration, crate::transport::PullReport, String) {
    let root = temporary(name);
    let store = Store::init(&root).unwrap();
    let policy = SourcePolicy::default();
    let mut request = request_for(&store, &hub.origin.base, &policy);
    request.sample_seconds = sample_seconds;
    let started = Instant::now();
    let report = pull(&request).unwrap_or_else(|refusal| panic!("{name}: {refusal}"));
    let elapsed = started.elapsed();
    assert_eq!(report.held + report.fetched, report.declared, "{name}");
    let (warm, _) = FetchPlan::of_objects(&store, name, &checkpoint.objects).unwrap();
    assert!(warm.wanted.is_empty(), "{name}: every object verified");
    let log = std::fs::read_to_string(root.join("logs/transport.log")).unwrap_or_default();
    let _ = std::fs::remove_dir_all(root);
    (elapsed, report, log)
}

/// The release gate. Half the objects have MiBs that crawl for their first ask: read whole,
/// each such object takes 32 s. The pull must finish at about the throughput it has with no
/// slow MiB at all, because a request that crawls is dropped and asked again.
#[test]
fn a_pull_with_slow_chunk_answers_finishes_at_body_throughput() {
    let checkpoint = Arc::new(checkpoint(24, OBJECT));
    let (body, _, _) = timed_pull(
        &cold_hub(Arc::clone(&checkpoint), Vec::new()),
        &checkpoint,
        "cold-body",
    );
    let slow: Vec<String> = checkpoint
        .objects
        .iter()
        .filter(|object| object.length == OBJECT as u64)
        .step_by(2)
        .map(|object| object.sha256.clone())
        .collect();
    assert_eq!(slow.len(), 12);
    let hub = cold_hub(Arc::clone(&checkpoint), slow);
    let (tail, report, log) = timed_pull(&hub, &checkpoint, "cold-tail");
    let sample = Duration::from_secs_f64(0.025);
    eprintln!(
        "body {:.2} s; with slow chunks {:.2} s; {} requests, {} restarts, {} asks crawled; one \
         slow object read whole {:.0} s",
        body.as_secs_f64(),
        tail.as_secs_f64(),
        report.requests,
        report.restarts,
        hub.crawled.load(Ordering::Relaxed),
        4.0 * MIB as f64 / CRAWL as f64
    );
    eprintln!("{log}");
    assert!(
        hub.crawled.load(Ordering::Relaxed) > 0,
        "no ask reached a slow MiB"
    );
    assert!(
        tail < body * 3 + sample * 20,
        "the tail crawled: {tail:?} against a {body:?} body"
    );
    // Each restart is on disk, and the pull's one summary line counts them.
    assert!(report.restarts > 0 && log.contains(" restart "), "{log}");
    assert!(
        log.contains(" walk ") && log.contains(&format!("restarts={}", report.restarts)),
        "{log}"
    );
    // A dropped request's bytes stay: its chunk is asked again from the byte it reached.
    assert!(
        hub.firsts
            .lock()
            .unwrap()
            .iter()
            .any(|first| first % CHUNK_BYTES as usize != 0),
        "no chunk was resumed mid-way"
    );
    assert!(
        report.bytes_moved
            < checkpoint
                .bodies
                .values()
                .map(|b| b.len() as u64)
                .sum::<u64>()
                * 2,
        "restarts re-bought the objects"
    );
}

/// An object larger than a chunk is asked for as ranged chunks, and one that is not, as one
/// plain GET.
#[test]
fn large_objects_are_ranged_chunks_and_small_ones_one_get() {
    let checkpoint = Arc::new(checkpoint(3, OBJECT));
    let hub = cold_hub(Arc::clone(&checkpoint), Vec::new());
    // A coarse sample: this counts requests, and a loaded host must not add a restart.
    let (_, report, _) = timed_pull_sampling(&hub, &checkpoint, "chunks", 1.0);
    let chunks = 3 * OBJECT / CHUNK_BYTES as usize;
    assert_eq!(hub.ranged.load(Ordering::Relaxed), chunks);
    // The header and the manifest are each smaller than a chunk.
    assert_eq!(report.requests as usize, chunks + 2, "{report:?}");
    assert_eq!(report.restarts, 0);
}

/// An unstalled pull asks the hub for URLs once (cozy-runtime's models-only prepare asserts
/// exactly one presign).
#[test]
fn an_unstalled_small_pull_presigns_once() {
    for _ in 0..20 {
        let checkpoint = Arc::new(checkpoint(2, 16 * 1024));
        let asked: Arc<Mutex<Vec<usize>>> = Arc::default();
        let base_slot: Arc<Mutex<String>> = Arc::default();
        let (slot, log_asks, served) = (
            Arc::clone(&base_slot),
            Arc::clone(&asked),
            Arc::clone(&checkpoint),
        );
        let dispatch: Dispatch = Arc::new(move |request: &Request| {
            let base = slot.lock().unwrap().clone();
            match (request.method.as_str(), request.path.as_str()) {
                ("POST", "/v1/tensorfs/closure") => ok_json(&closure_value(&served, &base)),
                ("POST", "/v1/tensorfs/presign") => {
                    let value = presign_value(&request.body, &base);
                    let Value::Obj(fields) = &value else { panic!() };
                    let count = fields
                        .iter()
                        .find(|(k, _)| k == "urls")
                        .map(|(_, v)| match v {
                            Value::Obj(m) => m.len(),
                            _ => 0,
                        })
                        .unwrap_or(0);
                    log_asks.lock().unwrap().push(count);
                    ok_json(&value)
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
        let origin = origin_with_hits(dispatch, Arc::default());
        *base_slot.lock().unwrap() = origin.base.clone();
        let root = temporary("presign-once");
        let store = Store::init(&root).unwrap();
        let policy = SourcePolicy::default();
        let mut request = PullRequest::new(&store, &origin.base, "acme/model", &Anonymous, &policy);
        request.sample_seconds = super::super::SAMPLE_SECONDS;
        pull(&request).unwrap();
        let asks = asked.lock().unwrap().clone();
        assert_eq!(
            asks.len(),
            1,
            "presign asks and their digest counts: {asks:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}

/// An origin whose every answer dies half way and that honours no range: each ask starts
/// its object over, so nothing new ever lands. The pull ends on its own measured patience
/// and says what the origin last did; it does not ask forever.
#[test]
fn a_pull_where_nothing_new_lands_ends_and_quotes_the_last_answer() {
    let checkpoint = Arc::new(checkpoint(2, 64 * 1024));
    let bodies = checkpoint.bodies.clone();
    let origin = hub_origin(Arc::clone(&checkpoint), move |hex, _| {
        let body = &bodies[hex];
        Some(Serve::Truncate {
            total: body.len() as u64,
            prefix: body[..body.len() / 2].to_vec(),
        })
    });
    let root = temporary("never-lands");
    let store = Store::init(&root).unwrap();
    let policy = SourcePolicy::default();
    let started = Instant::now();
    let refusal = pull(&request_for(&store, &origin.base, &policy)).unwrap_err();
    assert_eq!(refusal.code, Code::TRANSFER_FAILED, "{refusal}");
    assert!(refusal.detail.contains("nothing landed"), "{refusal}");
    assert!(refusal.detail.contains("last answer"), "{refusal}");
    // An ask that adds nothing waits a sample before the next: about one ask an object a
    // sample for as long as the pull's own patience ran, never a hot loop.
    let samples = started.elapsed().as_secs_f64() / 0.025;
    let objects = checkpoint.objects.len() + 1;
    assert!(
        (origin.total_hits() as f64) < objects as f64 * (samples + 8.0) * 2.0,
        "{} asks in {samples:.0} samples for {objects} objects",
        origin.total_hits()
    );
    let _ = std::fs::remove_dir_all(root);
}
