//! The tail of a hub pull against an R2-like origin that answers a few requests
//! application-limited: bytes keep arriving, so no silence rule fires, but at a
//! rate that would hold the whole pull for minutes (run 2322 on yuuichirou: eight
//! 64 MiB objects at 6-50 KB/s for 15+ minutes after the body landed at ~400 MB/s).

use super::*;
use std::time::Instant;

const BLOB: usize = 4 << 20;
/// The application-limited answer's rate. One object at this rate is the crawl.
const CRAWL: u64 = 128 * 1024;
const DRIP: usize = 4096;

struct Tail {
    origin: Origin,
    /// URLs presented after they died.
    rejected: Arc<AtomicUsize>,
    /// Asks for slow objects beyond their first whole GET.
    hedged: Arc<AtomicUsize>,
    /// Of those, ranged asks.
    ranged: Arc<AtomicUsize>,
}

fn drip_gap() -> Duration {
    Duration::from_secs_f64(DRIP as f64 / CRAWL as f64)
}

/// A hub whose store honours ranges and drip-feeds the first whole GET of each `slow`
/// object. With `lifetime`, URLs die as `expiring_hub_origin`'s do, and the first ranged
/// ask for each slow part keeps answering until its URL is dead, then drops — a hedged
/// range that outlives its presign.
fn tail_hub(checkpoint: Arc<Checkpoint>, slow: Vec<String>, lifetime: Option<Duration>) -> Tail {
    let base_slot: Arc<Mutex<String>> = Arc::default();
    let rejected = Arc::new(AtomicUsize::new(0));
    let hedged = Arc::new(AtomicUsize::new(0));
    let ranges = Arc::new(AtomicUsize::new(0));
    let whole_asks: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    let part_asks: Arc<Mutex<HashMap<(String, usize), usize>>> = Arc::default();
    let (slot, dead, again, ranged) = (
        Arc::clone(&base_slot),
        Arc::clone(&rejected),
        Arc::clone(&hedged),
        Arc::clone(&ranges),
    );
    let dispatch: Dispatch = Arc::new(move |request: &Request| {
        let base = slot.lock().unwrap().clone();
        let path = request.path.as_str();
        match request.method.as_str() {
            "POST" if path == "/v1/tensorfs/closure" => {
                return ok_json(&closure_value(&checkpoint, &base))
            }
            "POST" if path == "/v1/tensorfs/presign" => {
                return ok_json(&match lifetime {
                    Some(life) => expiring_presign_value(&request.body, &base, life),
                    None => presign_value(&request.body, &base),
                })
            }
            "GET" if path.starts_with("/obj/") => {}
            _ => return status(404, b"no such route"),
        }
        let hex = path["/obj/".len()..].to_string();
        let dies: Option<u128> = request
            .query
            .strip_prefix("dies=")
            .and_then(|stamp| stamp.parse().ok());
        if dies.is_some_and(|dies| now_millis() > dies) {
            dead.fetch_add(1, Ordering::Relaxed);
            return status(403, b"{\"error\":{\"code\":\"AccessDenied\"}}");
        }
        let Some(body) = checkpoint.bodies.get(&hex).cloned() else {
            return status(404, b"no such object");
        };
        if !slow.contains(&hex) {
            return partial(&body, request);
        }
        let Some((first, last)) = range_of(request) else {
            let mut asks = whole_asks.lock().unwrap();
            let count = asks.entry(hex).or_insert(0);
            *count += 1;
            if *count > 1 {
                again.fetch_add(1, Ordering::Relaxed);
                return status(200, &body);
            }
            return Serve::Drip {
                body,
                chunk: DRIP,
                gap: drip_gap(),
                ttfb: Duration::ZERO,
            };
        };
        again.fetch_add(1, Ordering::Relaxed);
        ranged.fetch_add(1, Ordering::Relaxed);
        let first_ask = {
            let mut asks = part_asks.lock().unwrap();
            let count = asks.entry((hex, first)).or_insert(0);
            *count += 1;
            *count == 1
        };
        let (Some(dies), true) = (dies, first_ask) else {
            return partial(&body, request);
        };
        let last = last.min(body.len() - 1);
        let range = body[first..=last].to_vec();
        Serve::Script(Box::new(move |socket| {
            let head = format!(
                "HTTP/1.1 206 X\r\ncontent-length: {}\r\ncontent-range: bytes {first}-{last}/{}\r\n\
                 connection: close\r\n\r\n",
                range.len(),
                body.len()
            );
            let _ = socket.write_all(head.as_bytes());
            // Expiry is judged when a request starts, as R2 judges it, so an answer in
            // flight keeps going after its URL died; this one then drops with bytes left.
            for piece in range.chunks(DRIP).take(range.len() / DRIP - 1) {
                if now_millis() > dies + 50 || socket.write_all(piece).is_err() {
                    break;
                }
                std::thread::sleep(drip_gap());
            }
        }))
    });
    let origin = origin_with_hits(dispatch, Arc::default());
    *base_slot.lock().unwrap() = origin.base.clone();
    Tail {
        origin,
        rejected,
        hedged,
        ranged: ranges,
    }
}

fn slow_objects(checkpoint: &Checkpoint, count: usize) -> Vec<String> {
    checkpoint
        .objects
        .iter()
        .filter(|object| object.length == BLOB as u64)
        .take(count)
        .map(|object| object.sha256.clone())
        .collect()
}

fn timed_pull(
    tail: &Tail,
    checkpoint: &Checkpoint,
    name: &str,
) -> (Duration, crate::transport::PullReport) {
    let root = temporary(name);
    let store = Store::init(&root).unwrap();
    let policy = SourcePolicy::default();
    let started = Instant::now();
    let report = pull(&request_for(&store, &tail.origin.base, &policy))
        .unwrap_or_else(|refusal| panic!("{name}: {refusal}"));
    let elapsed = started.elapsed();
    assert_eq!(report.held + report.fetched, report.declared, "{name}");
    let (warm, _) = FetchPlan::of_objects(&store, name, &checkpoint.objects).unwrap();
    assert!(warm.wanted.is_empty(), "{name}: every object verified");
    let _ = std::fs::remove_dir_all(root);
    (elapsed, report)
}

/// The regression gate for run 2322. The body throughput is measured on this host first,
/// then the same checkpoint is pulled with three objects whose first answer crawls. The
/// pull must finish at a throughput comparable to the body, not at the crawl.
#[test]
fn an_application_limited_tail_finishes_at_body_throughput() {
    let checkpoint = Arc::new(checkpoint(24, BLOB));
    let (body, _) = timed_pull(
        &tail_hub(Arc::clone(&checkpoint), Vec::new(), None),
        &checkpoint,
        "tail-body",
    );

    let slow = slow_objects(&checkpoint, 3);
    let hub = tail_hub(Arc::clone(&checkpoint), slow, None);
    let (tail, report) = timed_pull(&hub, &checkpoint, "tail-crawl");
    let crawl = Duration::from_secs_f64(BLOB as f64 / CRAWL as f64);
    let sample = Duration::from_secs_f64(0.025);
    let bytes: u64 = checkpoint.bodies.values().map(|b| b.len() as u64).sum();
    let rate = |elapsed: Duration| bytes as f64 / elapsed.as_secs_f64() / 1e6;
    eprintln!(
        "body {:.2} s ({:.1} MB/s); tail {:.2} s ({:.1} MB/s); one crawling object alone {:.0} s",
        body.as_secs_f64(),
        rate(body),
        tail.as_secs_f64(),
        rate(tail),
        crawl.as_secs_f64()
    );
    assert!(
        hub.hedged.load(Ordering::Relaxed) > 0,
        "no slow object was hedged"
    );
    assert!(
        tail < body * 3 + sample * 40,
        "the tail crawled: {tail:?} against a {body:?} body"
    );
    // 0.3.80 takes the crawl itself on top of the body; the bound above is the gate.
    assert!(tail < crawl / 2, "the tail crawled: {tail:?}");
    assert!(
        report.bytes_moved < bytes + 3 * BLOB as u64 * 2,
        "hedging re-bought more than the slow objects"
    );
}

/// A hedged range that outlives its presign re-authorizes instead of presenting the dead
/// URL again, and the object still finishes far ahead of the crawl.
#[test]
fn a_hedged_range_outliving_its_presign_reauthorizes() {
    // Few objects, so the slow ones are a tail with idle lanes to split them over.
    let checkpoint = Arc::new(checkpoint(3, BLOB));
    let slow = slow_objects(&checkpoint, 2);
    let hub = tail_hub(Arc::clone(&checkpoint), slow, Some(Duration::from_secs(1)));
    let (tail, _) = timed_pull(&hub, &checkpoint, "tail-expiry");
    let crawl = Duration::from_secs_f64(BLOB as f64 / CRAWL as f64);
    eprintln!(
        "expiring tail {:.2} s, {} ranged asks, {} dead URLs presented",
        tail.as_secs_f64(),
        hub.ranged.load(Ordering::Relaxed),
        hub.rejected.load(Ordering::Relaxed)
    );
    assert!(
        hub.ranged.load(Ordering::Relaxed) > 0,
        "no slow object was split"
    );
    assert!(tail < crawl / 2, "the tail crawled: {tail:?}");
}
