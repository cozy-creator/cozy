//! The tail of a pull against origins that answer like a cold R2 object, stall, fail or are
//! slow everywhere. A pull must finish at about the rate it has with no fault at all, because a
//! request that crawls, stalls or fails is dropped and asked again from the byte it reached;
//! and a fetch whose every request is slow restarts none. Run with `--test-threads=1`: the
//! verdicts are relative to the fetch's own measurements, and a starved test thread is one.

mod pull_support;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pull_support::*;
use tensorfs::fetch::{Config, Fetcher, Endpoint, ObjectStore, Origin, Want};

async fn pull(server: &Server, name: &str, config: Config) -> (Duration, Fetcher, Scratch) {
    let root = temporary(&format!("{}-store", name.replace('/', "-")));
    let fetcher = Fetcher::new(
        open_store(&root) as Arc<dyn ObjectStore>,
        client(),
        None,
        config,
    );
    let origin: Arc<dyn Origin> = Arc::new(Endpoint::new(&client(), &server.base, None).unwrap().anonymous());
    let started = Instant::now();
    let pulled = fetcher
        .pull(&origin, name, None, Some("int8"), &Want::default())
        .await;
    pulled.unwrap_or_else(|refusal| panic!("{name}: {refusal}"));
    (started.elapsed(), fetcher, root)
}

fn config(tick: Duration) -> Config {
    Config {
        tick,
        ..Config::default()
    }
}

/// The release gate. Half the objects have MiBs that crawl at 128 KB/s for their first ask: read
/// whole, each such object would take 16 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cold_mibs_finish_at_about_the_body_rate() {
    let tick = Duration::from_millis(20);
    let origin_dir = temporary("cold-origin");
    let model = model(&origin_dir, "acme/cold", &[("unet", &[8 * MIB; 12])], 11);
    let (body, _, _) = pull(
        &Server::serve(&origin_dir, plain()),
        "acme/cold",
        config(tick),
    )
    .await;
    let slow: HashSet<String> = model.chunks["unet"]
        .iter()
        .step_by(2)
        .map(|o| format!("/{}", o.key()))
        .collect();
    let server = Server::serve(&origin_dir, cold(slow.clone(), 128 << 10));
    let (tail, fetcher, root) = pull(&server, "acme/cold", config(tick)).await;
    let stats = fetcher.stats();
    eprintln!(
        "body {:.2} s, with cold MiBs {:.2} s; {} requests, {} restarts",
        body.as_secs_f64(),
        tail.as_secs_f64(),
        stats.requests,
        stats.restarts
    );
    assert!(
        tail < body * 3 + tick * 50,
        "the tail crawled: {tail:?} against a {body:?} body"
    );
    assert!(stats.restarts > 0, "no request was judged");
    let resumed = server
        .blob_asks()
        .iter()
        .filter(|a| slow.contains(&a.path) && a.first > 0)
        .count();
    assert!(
        resumed > 0,
        "no object was resumed from the byte it reached"
    );
    assert_eq!(blobs(&root), blobs(&origin_dir));
}

/// A body that stops mid-way, an ask never answered, an answer with no body, a 503 and a
/// connection cut mid-body, each on an object's first ask: all are asked again, the stalled
/// ones from the byte they reached.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stalled_silent_and_failing_requests_are_asked_again() {
    let tick = Duration::from_millis(20);
    let origin_dir = temporary("faults-origin");
    let model = model(&origin_dir, "acme/faults", &[("unet", &[4 * MIB; 8])], 12);
    let paths: Vec<String> = model.chunks["unet"]
        .iter()
        .map(|o| format!("/{}", o.key()))
        .collect();
    let faulty = paths.clone();
    let server = Server::serve(
        &origin_dir,
        Arc::new(move |ask, body, socket| {
            let Some(which) = faulty
                .iter()
                .position(|p| *p == ask.path)
                .filter(|_| ask.n == 1)
            else {
                return false;
            };
            match which {
                0 => {
                    let _ = head(socket, body.len(), 0)
                        .and_then(|()| std::io::Write::write_all(socket, &body[..MIB]));
                    hold(socket);
                }
                1 => hold(socket),
                2 => {
                    let _ = head(socket, body.len(), 0);
                    hold(socket);
                }
                3 => {
                    let _ = std::io::Write::write_all(
                        socket,
                        b"HTTP/1.1 503 Busy\r\nretry-after: 0\r\ncontent-length: 0\r\n\r\n",
                    );
                }
                4 => {
                    let _ = head(socket, body.len(), 0)
                        .and_then(|()| std::io::Write::write_all(socket, &body[..2 * MIB]));
                }
                _ => return false,
            }
            true
        }),
    );
    let (took, fetcher, root) = pull(&server, "acme/faults", config(tick)).await;
    let stats = fetcher.stats();
    eprintln!(
        "{:.2} s; {} requests, {} restarts",
        took.as_secs_f64(),
        stats.requests,
        stats.restarts
    );
    assert_eq!(blobs(&root), blobs(&origin_dir));
    assert!(
        stats.restarts >= 3,
        "the stalled, silent and bodiless asks were judged: {stats:?}"
    );
    let asks = server.blob_asks();
    assert!(
        asks.iter().any(|a| a.path == paths[0] && a.first == MIB),
        "the stalled body resumed from its MiB"
    );
    assert!(
        asks.iter()
            .any(|a| a.path == paths[4] && a.first == 2 * MIB),
        "the cut body resumed from where it was cut"
    );
    assert!(took < tick * 200, "{took:?}");
}

/// Every request moves at the same slow rate (a slow pod): none is slow against its peers, so
/// none is restarted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fetch_whose_every_request_is_slow_restarts_none() {
    let origin_dir = temporary("slow-origin");
    let model = model(&origin_dir, "acme/slow", &[("unet", &[2 * MIB; 6])], 13);
    let chunks: HashSet<String> = model.chunks["unet"]
        .iter()
        .map(|o| format!("/{}", o.key()))
        .collect();
    let server = Server::serve(
        &origin_dir,
        Arc::new(move |ask, body, socket| {
            if !chunks.contains(&ask.path) {
                return false;
            }
            let _ = head(socket, body.len(), ask.first)
                .and_then(|()| drip(socket, &body[ask.first..], 4 << 20));
            true
        }),
    );
    let (took, fetcher, root) =
        pull(&server, "acme/slow", config(Duration::from_millis(100))).await;
    eprintln!("{:.2} s; {:?}", took.as_secs_f64(), fetcher.stats());
    assert_eq!(fetcher.stats().restarts, 0);
    assert_eq!(blobs(&root), blobs(&origin_dir));
}

/// Each connection is capped at 1 MB/s, so more streams buy more goodput: the pace doubles while
/// objects wait, up to its ceiling (tfs-122).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streams_grow_while_more_of_them_pay() {
    let origin_dir = temporary("pace-origin");
    let model = model(&origin_dir, "acme/pace", &[("unet", &[256 << 10; 96])], 14);
    let chunks: HashSet<String> = model.chunks["unet"]
        .iter()
        .map(|o| format!("/{}", o.key()))
        .collect();
    let server = Server::serve(
        &origin_dir,
        Arc::new(move |ask, body, socket| {
            if !chunks.contains(&ask.path) {
                return false;
            }
            let _ = head(socket, body.len(), ask.first)
                .and_then(|()| drip(socket, &body[ask.first..], 1 << 20));
            true
        }),
    );
    let config = Config {
        tick: Duration::from_millis(25),
        streams: 4,
        max_streams: 32,
    };
    let (took, fetcher, root) = pull(&server, "acme/pace", config).await;
    eprintln!("{:.2} s; {:?}", took.as_secs_f64(), fetcher.stats());
    assert!(fetcher.stats().streams > 4, "{:?}", fetcher.stats());
    assert_eq!(blobs(&root), blobs(&origin_dir));
}
