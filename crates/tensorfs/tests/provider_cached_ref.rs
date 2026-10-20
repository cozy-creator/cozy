//! Supplied provider failures are not transport weather, even with a usable cached repo.
#![cfg(feature = "store")]

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use async_trait::async_trait;
use tensorfs::format::{Dtype, ObjectRef, Repo};
use tensorfs::{
    export, Client, Config, Endpoint, Failure, Fetch, Format, Kind, Open, Request, TokenProvider,
    WriterOptions,
};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "provider-cached-ref-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&root) {
                Ok(()) => return Self(root),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => panic!("create owned root: {e}"),
            }
        }
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Http {
    url: String,
    requests: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Http {
    fn new(root: PathBuf) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(AtomicUsize::new(0));
        let (done, seen) = (stop.clone(), requests.clone());
        let worker = thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => serve(stream, &root, &seen),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1))
                    }
                    Err(e) => panic!("accept healthy fixture: {e}"),
                }
            }
        });
        Self {
            url,
            requests,
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for Http {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
fn serve(mut stream: TcpStream, root: &Path, requests: &AtomicUsize) {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut header = Vec::new();
    let mut byte = [0];
    while !header.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).unwrap_or(0) == 0 {
            return;
        }
        header.push(byte[0]);
        assert!(header.len() < 32768);
    }
    requests.fetch_add(1, Ordering::AcqRel);
    let header = String::from_utf8(header).unwrap();
    let mut first = header.lines().next().unwrap().split_whitespace();
    let method = first.next().unwrap();
    let path = first
        .next()
        .unwrap()
        .split('?')
        .next()
        .unwrap()
        .trim_start_matches('/');
    assert!(!path.contains(".."));
    let Ok(bytes) = fs::read(root.join(path)) else {
        let _ = stream
            .write_all(b"HTTP/1.1 404 Missing\r\ncontent-length:0\r\nconnection:close\r\n\r\n");
        return;
    };
    let range = header.lines().find_map(|line| {
        let (k, v) = line.split_once(':')?;
        if !k.eq_ignore_ascii_case("range") {
            return None;
        }
        let (lo, hi) = v.trim().strip_prefix("bytes=")?.split_once('-')?;
        Some((
            lo.parse::<usize>().unwrap(),
            if hi.is_empty() {
                bytes.len()
            } else {
                hi.parse::<usize>().unwrap() + 1
            }
            .min(bytes.len()),
        ))
    });
    let (lo, hi, status) = range.map_or((0, bytes.len(), 200), |(lo, hi)| (lo, hi, 206));
    let range_header = if range.is_some() {
        format!("content-range:bytes {}-{}/{}\r\n", lo, hi - 1, bytes.len())
    } else {
        String::new()
    };
    let answer=format!("HTTP/1.1 {status} fixture\r\ncontent-length:{}\r\naccept-ranges:bytes\r\netag:\"{}\"\r\n{range_header}connection:close\r\n\r\n",hi-lo,ObjectRef::of(&bytes).id());
    if stream.write_all(answer.as_bytes()).is_ok() && method != "HEAD" {
        let _ = stream.write_all(&bytes[lo..hi]);
    }
}

const DETAIL: &str = "literal supplied provider network refusal";
struct Provider {
    fail: AtomicUsize,
    calls: AtomicUsize,
}
#[async_trait]
impl TokenProvider for Provider {
    async fn headers(&self, _: &Request<'_>) -> tensorfs::Result<Vec<(String, String)>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        if self.fail.load(Ordering::Acquire) == 1 {
            Err(Failure {
                kind: Kind::Network,
                detail: DETAIL.into(),
            })
        } else {
            Ok(Vec::new())
        }
    }

    fn retry(&self, _: &Request<'_>, _: u16, _: &tensorfs::HeaderMap) -> tensorfs::Result<bool> {
        if self.fail.load(Ordering::Acquire) == 2 {
            Err(Failure {
                kind: Kind::Network,
                detail: DETAIL.into(),
            })
        } else {
            Ok(false)
        }
    }
}
fn config(cache: &Path, endpoint: &str) -> Config {
    let mut config = Config::from_env();
    config.cache_dir = cache.to_path_buf();
    config.endpoint = endpoint.into();
    config.token = None;
    config.mirror = None;
    config
}

#[tokio::test(flavor = "current_thread")]
async fn provider_network_error_cannot_become_cached_ref_success() {
    const WEIGHT: &[u8] = &[3, 7, 11, 19];
    let scratch = Scratch::new();
    let origin = scratch.0.join("origin");
    let producer = Client::new(config(
        &scratch.0.join("producer"),
        origin.to_str().unwrap(),
    ))
    .unwrap();
    let mut writer = producer.writer(WriterOptions::default()).unwrap();
    writer
        .component("model", "test.Model", Default::default())
        .unwrap()
        .put("w", Dtype::U8, &[4], &mut &WEIGHT[..])
        .unwrap();
    let checkpoint = writer.commit().await.unwrap();
    let folder = scratch.0.join("portable");
    export(&checkpoint, &folder, Format::Cozytensors, None).unwrap();
    let tree = producer
        .upload_folder(&"acme/provider@1".parse().unwrap(), &folder, None)
        .await
        .unwrap();
    let http = Http::new(origin);
    let cache = scratch.0.join("reader");
    let provider = Arc::new(Provider {
        fail: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
    });
    let mut options = config(&cache, &http.url);
    options.token = Some(provider.clone());
    let client = Client::new(options).unwrap();
    let warm = tokio::time::timeout(
        Duration::from_secs(2),
        client.open("acme/provider", &Open::default()),
    )
    .await
    .unwrap()
    .unwrap();
    warm.fetch(&Fetch::default()).await.unwrap();
    assert_eq!(warm.component("model").unwrap().read("w").unwrap(), WEIGHT);
    assert_eq!(warm.tree(), Some(&tree));
    assert!(
        http.requests.load(Ordering::Acquire) > 0,
        "no actual HTTP source read"
    );
    let repo = Repo::from_json(
        &fs::read(cache.join("refs/acme/provider.json")).expect("first open did not cache a repo"),
    )
    .unwrap();
    assert_eq!(repo.release(None).unwrap().tree, tree);
    drop(warm);

    // Independent ordinary anonymous read proves the endpoint is still healthy. This does
    // not invoke the supplied provider or preflight the operation under test.
    let endpoint = Endpoint::new(&http.url, None).unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), endpoint.repo("acme/provider"))
            .await
            .unwrap()
            .unwrap()
            .is_some()
    );
    for phase in [1, 2] {
        provider.fail.store(phase, Ordering::Release);
        let (calls, requests) = (
            provider.calls.load(Ordering::Acquire),
            http.requests.load(Ordering::Acquire),
        );
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            client.open("acme/provider", &Open::default()),
        )
        .await
        .expect("supplied refusal retried as transport weather");
        assert_eq!(
            provider.calls.load(Ordering::Acquire),
            calls + 1,
            "provider error was replayed"
        );
        assert_eq!(
            http.requests.load(Ordering::Acquire),
            requests + usize::from(phase == 2),
            "provider failure sent an unexpected HTTP request"
        );
        let error = result
            .expect_err("supplied Network failure was swallowed into cached repository success");
        assert_eq!((error.kind, error.detail.as_str()), (Kind::Network, DETAIL));
    }
}
