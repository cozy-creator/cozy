//! Real HTTP import cancellation and caller-runtime ownership; generated tiny source bytes.
#![cfg(feature = "store")]

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use async_trait::async_trait;
use tensorfs::client::ImportOptions;
use tensorfs::err::{Code, Result};
use tensorfs::fetch::token::{HeaderMap, TokenProvider};
use tensorfs::format::{ObjectRef, Value};
use tensorfs::import::{self, Access, Request};
use tensorfs::store2::Store;

const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const DATA: &[u8] = b"0123456789abcdef";
const CONFIG: &[u8] = br#"{"_class_name":"Fixture","gain":3}"#;
const ASSET: &[u8] = br#"{"vocab":["one","two"]}"#;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "import-control-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&root) {
                Ok(()) => return Self(root),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("own test root: {error}"),
            }
        }
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn safetensors() -> Vec<u8> {
    let header = serde_json::to_vec(&serde_json::json!({"weight":{"dtype":"U8","shape":[DATA.len()],"data_offsets":[0,DATA.len()]}})).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(DATA);
    bytes
}

#[derive(Clone, Copy)]
enum Stall {
    Listing,
    Member,
}
#[derive(Clone)]
struct Seen {
    path: String,
    range: Option<String>,
    authorized: bool,
}
struct Origin {
    url: String,
    stalled: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<Seen>>>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Origin {
    fn new(stall: Option<Stall>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (stalled, release, stop, requests) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(Vec::new())),
        );
        let (waiting, go, done, record) = (
            stalled.clone(),
            release.clone(),
            stop.clone(),
            requests.clone(),
        );
        let tensor = safetensors();
        let members = [
            ("model.safetensors", tensor.clone()),
            ("scheduler_config.json", CONFIG.to_vec()),
            ("tokenizer.json", ASSET.to_vec()),
            ("README.md", b"tiny import proof\n".to_vec()),
        ];
        let tree=serde_json::to_vec(&members.iter().map(|(name,bytes)| serde_json::json!({"type":"file","path":name,"size":bytes.len(),"lfs":{"oid":ObjectRef::of(bytes).sha256,"size":bytes.len()}})).collect::<Vec<_>>()).unwrap();
        let mut files = BTreeMap::from([
            (
                "/api/models/proof/model/revision/main".into(),
                serde_json::to_vec(&serde_json::json!({"sha":COMMIT})).unwrap(),
            ),
            (format!("/api/models/proof/model/tree/{COMMIT}"), tree),
        ]);
        for (name, bytes) in members {
            files.insert(format!("/proof/model/resolve/{COMMIT}/{name}"), bytes);
        }
        let worker = thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(error) => panic!("accept: {error}"),
                };
                let Some(request) = receive(&mut stream) else {
                    continue;
                };
                let path = request.path.split('?').next().unwrap().to_owned();
                record.lock().unwrap().push(request.clone());
                let Some(body) = files.get(&path) else {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Missing\r\ncontent-length:0\r\nconnection:close\r\n\r\n",
                    );
                    continue;
                };
                let should_stall = match stall {
                    Some(Stall::Listing) => path.ends_with("/revision/main"),
                    Some(Stall::Member) => path.ends_with("/model.safetensors"),
                    None => false,
                };
                if should_stall && !waiting.load(Ordering::Acquire) {
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\ncontent-length:{}\r\nconnection:close\r\n\r\n",
                        body.len()
                    )
                    .unwrap();
                    let prefix = 16.min(body.len() / 2);
                    stream.write_all(&body[..prefix]).unwrap();
                    stream.flush().unwrap();
                    waiting.store(true, Ordering::Release);
                    while !go.load(Ordering::Acquire) && !done.load(Ordering::Acquire) {
                        thread::sleep(Duration::from_millis(1));
                    }
                    let _ = stream.write_all(&body[prefix..]);
                    continue;
                }
                let (lo, hi, status) = match request
                    .range
                    .as_deref()
                    .and_then(|range| range.strip_prefix("bytes="))
                    .and_then(|range| range.split_once('-'))
                {
                    Some((lo, hi)) => (
                        lo.parse::<usize>().unwrap(),
                        if hi.is_empty() {
                            body.len()
                        } else {
                            hi.parse::<usize>().unwrap() + 1
                        }
                        .min(body.len()),
                        206,
                    ),
                    None => (0, body.len(), 200),
                };
                let range = if status == 206 {
                    format!("content-range:bytes {}-{}/{}\r\n", lo, hi - 1, body.len())
                } else {
                    String::new()
                };
                let _=write!(stream,"HTTP/1.1 {status} fixture\r\ncontent-length:{}\r\n{range}connection:close\r\n\r\n",hi-lo);
                let _ = stream.write_all(&body[lo..hi]);
            }
        });
        Self {
            url,
            stalled,
            release,
            stop,
            requests,
            worker: Some(worker),
        }
    }
    async fn wait_stalled(&self) {
        tokio::time::timeout(Duration::from_secs(3), async {
            while !self.stalled.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(1)).await
            }
        })
        .await
        .expect("actual HTTP body never stalled");
    }
}
impl Drop for Origin {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.release.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn receive(stream: &mut TcpStream) -> Option<Seen> {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut one = [0u8];
    while !bytes.ends_with(b"\r\n\r\n") {
        if stream.read(&mut one).ok()? == 0 {
            return None;
        }
        bytes.push(one[0]);
        assert!(bytes.len() < 32768)
    }
    let request = String::from_utf8(bytes).unwrap();
    let path = request
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .to_owned();
    let headers: BTreeMap<_, _> = request
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_owned()))
        .collect();
    Some(Seen {
        path,
        range: headers.get("range").cloned(),
        authorized: headers
            .get("authorization")
            .is_some_and(|value| value == "Bearer import-proof"),
    })
}

struct OnCaller(Arc<AtomicUsize>);
#[async_trait]
impl TokenProvider for OnCaller {
    async fn headers(
        &self,
        _: &tensorfs::fetch::token::Request<'_>,
    ) -> Result<Vec<(String, String)>> {
        assert_eq!(
            tokio::runtime::Handle::current().runtime_flavor(),
            tokio::runtime::RuntimeFlavor::CurrentThread,
            "HTTP/provider work escaped to the library's global runtime"
        );
        self.0.fetch_add(1, Ordering::AcqRel);
        Ok(vec![])
    }
    fn retry(
        &self,
        _: &tensorfs::fetch::token::Request<'_>,
        _: u16,
        _: &HeaderMap,
    ) -> Result<bool> {
        Ok(false)
    }
}
fn access(
    origin: &Origin,
    handle: Option<tokio::runtime::Handle>,
    cancelled: Option<Arc<AtomicBool>>,
    calls: Option<Arc<AtomicUsize>>,
) -> Access {
    Access {
        huggingface: origin.url.clone(),
        huggingface_token: "bearer import-proof".into(),
        allow_local: true,
        runtime: handle,
        cancelled,
        token: calls.map(|calls| Arc::new(OnCaller(calls)) as Arc<dyn TokenProvider>),
        ..Access::default()
    }
}
fn convert(
    store: &Store,
    access: &Access,
    stop: Option<&(dyn Fn() -> bool + Sync)>,
) -> (Result<ObjectRef>, BTreeMap<String, ObjectRef>, bool) {
    let sources = vec!["hf://proof/model".to_owned()];
    let tmp = store.root().join("tmp");
    let mut writer = store.writer(None, None).unwrap();
    let request = Request {
        sources: &sources,
        converter: None,
        variant: None,
        access,
        tmp: &tmp,
        progress: &|_, _, _| {},
        stop,
    };
    let result =
        import::convert(&request, &mut writer).map(|_| writer.committed().unwrap().clone());
    (
        result,
        writer.repo_files().clone(),
        writer.committed().is_some(),
    )
}
fn assert_bytes(store: &Store, id: &ObjectRef, files: &BTreeMap<String, ObjectRef>) {
    let ck = store.checkpoint(id).unwrap();
    assert_eq!(ck.read("model", "weight").unwrap(), DATA);
    let component = ck.component("model").unwrap();
    assert_eq!(
        component.config_file.as_deref(),
        Some("scheduler_config.json")
    );
    assert_eq!(
        store.read(&component.assets["tokenizer.json"]).unwrap(),
        ASSET
    );
    assert_eq!(
        store.read(&files["README.md"]).unwrap(),
        b"tiny import proof\n"
    );
    assert_eq!(
        ck.header().metadata.get("source"),
        Some(&Value::Str(format!("hf://proof/model@{COMMIT}")))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn stalled_listing_and_member_cancel_on_caller_runtime_and_resume_exact_bytes() {
    for stall in [Stall::Listing, Stall::Member] {
        let scratch = Scratch::new();
        let origin = Origin::new(Some(stall));
        let store = Store::open(scratch.0.join("store")).unwrap();
        let flag = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        // Exercise the shared drop flag and the independently borrowed Request.stop path.
        let shared = matches!(stall, Stall::Listing).then(|| flag.clone());
        let worker_access = access(
            &origin,
            Some(tokio::runtime::Handle::current()),
            shared,
            Some(calls.clone()),
        );
        let worker_store = store.clone();
        let worker_flag = flag.clone();
        let pending = tokio::task::spawn_blocking(move || {
            let stop = || worker_flag.load(Ordering::Acquire);
            convert(
                &worker_store,
                &worker_access,
                matches!(stall, Stall::Member).then_some(&stop as &(dyn Fn() -> bool + Sync)),
            )
        });
        origin.wait_stalled().await;
        flag.store(true, Ordering::Release);
        let (result, _, committed) = tokio::time::timeout(Duration::from_secs(2), pending)
            .await
            .expect("stalled transfer ignored cancellation")
            .unwrap();
        let error = result.unwrap_err();
        assert_eq!(error.code, Code::TRANSFER_FAILED);
        assert!(error.detail.contains("stopped"));
        assert!(!committed, "canceled transfer committed a header");
        assert!(calls.load(Ordering::Acquire) > 0);
        origin.release.store(true, Ordering::Release);
        flag.store(false, Ordering::Release);
        let resume_access = access(
            &origin,
            Some(tokio::runtime::Handle::current()),
            Some(flag),
            Some(calls),
        );
        let resumed_store = store.clone();
        let (result, files, committed) =
            tokio::task::spawn_blocking(move || convert(&resumed_store, &resume_access, None))
                .await
                .unwrap();
        let id = result.unwrap();
        assert!(committed);
        assert_bytes(&store, &id, &files);
        let seen = origin.requests.lock().unwrap();
        assert!(
            seen.iter().all(|request| request.authorized),
            "source credentials were lost"
        );
        if matches!(stall, Stall::Member) {
            assert!(
                seen.iter()
                    .any(|request| request.range.as_deref() == Some("bytes=16-")),
                "the real partial member did not resume"
            );
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_caller_and_legacy_sync_paths_produce_the_identical_header() {
    let scratch = Scratch::new();
    let origin = Origin::new(None);
    let caller = Store::open(scratch.0.join("caller")).unwrap();
    let legacy = Store::open(scratch.0.join("legacy")).unwrap();
    let caller_access = access(
        &origin,
        Some(tokio::runtime::Handle::current()),
        None,
        Some(Arc::new(AtomicUsize::new(0))),
    );
    let copied = caller.clone();
    let (result, files, _) =
        tokio::task::spawn_blocking(move || convert(&copied, &caller_access, None))
            .await
            .unwrap();
    let first = result.unwrap();
    assert_bytes(&caller, &first, &files);
    let legacy_access = access(&origin, None, None, None);
    let copied = legacy.clone();
    let (result, files, _) =
        tokio::task::spawn_blocking(move || convert(&copied, &legacy_access, None))
            .await
            .unwrap();
    let second = result.unwrap();
    assert_bytes(&legacy, &second, &files);
    assert_eq!(
        first, second,
        "runtime/control plumbing changed deterministic ingest"
    );
}

#[test]
fn cancellation_after_progress_stops_before_tensor_write() {
    let scratch = Scratch::new();
    let file = scratch.0.join("model.safetensors");
    fs::write(&file, safetensors()).unwrap();
    let sources = vec![file.to_str().unwrap().to_owned()];
    let store = Store::open(scratch.0.join("store")).unwrap();
    let mut writer = store.writer(None, None).unwrap();
    let stop = AtomicBool::new(false);
    let progress = |stage: &str, _, _| {
        if stage == "converting" {
            stop.store(true, Ordering::Release)
        }
    };
    let stopping = || stop.load(Ordering::Acquire);
    let access = Access::default();
    let tmp = store.root().join("tmp");
    let error = import::convert(
        &Request {
            sources: &sources,
            converter: None,
            variant: None,
            access: &access,
            tmp: &tmp,
            progress: &progress,
            stop: Some(&stopping),
        },
        &mut writer,
    )
    .unwrap_err();
    assert_eq!(error.code, Code::TRANSFER_FAILED);
    assert!(!writer.has("model", "weight"));
    assert!(writer.committed().is_none());
}

struct CancelWriter<'a> {
    writer: &'a mut tensorfs::writer::Writer,
    stop: &'a AtomicBool,
    during_commit: bool,
}
impl import::Sink for CancelWriter<'_> {
    fn begin(&mut self, variant: &str, layout: &import::Layout) -> Result<()> {
        import::Sink::begin(self.writer, variant, layout)
    }
    fn put(
        &mut self,
        component: &str,
        key: &str,
        dtype: &tensorfs::format::Dtype,
        shape: &[u64],
        encoding: Option<&tensorfs::format::Encoding>,
        parts: Vec<import::Part<'_>>,
    ) -> Result<()> {
        import::Sink::put(self.writer, component, key, dtype, shape, encoding, parts)?;
        if !self.during_commit {
            self.stop.store(true, Ordering::Release);
        }
        Ok(())
    }
    fn commit(&mut self, metadata: tensorfs::format::Metadata) -> Result<()> {
        if self.during_commit {
            self.stop.store(true, Ordering::Release);
        }
        import::Sink::commit(self.writer, metadata)
    }
}

#[test]
fn stop_after_real_tensor_write_prevents_final_commit() {
    let scratch = Scratch::new();
    let file = scratch.0.join("model.safetensors");
    fs::write(&file, safetensors()).unwrap();
    let sources = vec![file.to_str().unwrap().to_owned()];
    let store = Store::open(scratch.0.join("store")).unwrap();
    let mut writer = store.writer(None, None).unwrap();
    let stop = AtomicBool::new(false);
    let stopping = || stop.load(Ordering::Acquire);
    let access = Access::default();
    let tmp = store.root().join("tmp");
    let error = import::convert(
        &Request {
            sources: &sources,
            converter: None,
            variant: None,
            access: &access,
            tmp: &tmp,
            progress: &|_, _, _| {},
            stop: Some(&stopping),
        },
        &mut CancelWriter {
            writer: &mut writer,
            stop: &stop,
            during_commit: false,
        },
    )
    .unwrap_err();
    assert_eq!(error.code, Code::TRANSFER_FAILED);
    assert!(
        writer.has("model", "weight"),
        "real tensor write did not finish"
    );
    assert!(
        writer.committed().is_none(),
        "cancellation committed the header"
    );
}

#[test]
fn final_progress_stop_prevents_header_tree_and_ref_publication() {
    let scratch = Scratch::new();
    let file = scratch.0.join("model.safetensors");
    fs::write(&file, safetensors()).unwrap();
    let sources = vec![file.to_str().unwrap().to_owned()];
    let store = Store::open(scratch.0.join("store")).unwrap();
    let stop = AtomicBool::new(false);
    let progress = |stage: &str, _, _| {
        if stage == "imported" {
            stop.store(true, Ordering::Release)
        }
    };
    let stopping = || stop.load(Ordering::Acquire);
    let error = store
        .import_named(
            &sources,
            &ImportOptions {
                name: Some("local/no-ref"),
                converter: None,
                variant: None,
                progress: &progress,
                stop: Some(&stopping),
            },
        )
        .unwrap_err();
    assert_eq!(error.kind, tensorfs::Kind::Network);
    assert!(store.get_ref("local/no-ref").unwrap().is_none());
    assert!(
        store.get_pin("import.local.no-ref").unwrap().is_none(),
        "final progress cancellation committed a header"
    );
}

#[test]
fn stop_after_finalization_begins_cannot_undo_committed_header() {
    let scratch = Scratch::new();
    let file = scratch.0.join("model.safetensors");
    fs::write(&file, safetensors()).unwrap();
    let sources = vec![file.to_str().unwrap().to_owned()];
    let store = Store::open(scratch.0.join("store")).unwrap();
    let mut writer = store.writer(None, None).unwrap();
    let stop = AtomicBool::new(false);
    let stopping = || stop.load(Ordering::Acquire);
    let progress = |_: &str, _, _| {
        assert!(
            !stop.load(Ordering::Acquire),
            "callback ran after finalization began"
        );
    };
    let access = Access::default();
    let tmp = store.root().join("tmp");
    import::convert(
        &Request {
            sources: &sources,
            converter: None,
            variant: None,
            access: &access,
            tmp: &tmp,
            progress: &progress,
            stop: Some(&stopping),
        },
        &mut CancelWriter {
            writer: &mut writer,
            stop: &stop,
            during_commit: true,
        },
    )
    .unwrap();
    assert!(stop.load(Ordering::Acquire));
    let ck = store
        .checkpoint(writer.committed().expect("finalization did not finish"))
        .unwrap();
    assert_eq!(ck.read("model", "weight").unwrap(), DATA);
}

#[test]
fn owned_drop_cancels_stalled_worker_after_caller_stops_driving_io() {
    struct CancelOnDrop(Arc<AtomicBool>, Arc<tokio::sync::Notify>);
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
            self.1.notify_waiters();
        }
    }
    let scratch = Scratch::new();
    let origin = Origin::new(Some(Stall::Member));
    let store = Store::open(scratch.0.join("store")).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let notify = Arc::new(tokio::sync::Notify::new());
    let cancel = CancelOnDrop(flag.clone(), notify.clone());
    let mut worker_access = access(&origin, Some(runtime.handle().clone()), Some(flag), None);
    worker_access.cancel_notify = Some(notify);
    let (send, recv) = std::sync::mpsc::sync_channel(1);
    runtime.block_on(async {
        tokio::task::spawn_blocking(move || {
            let _ = send.send(convert(&store, &worker_access, None));
        });
        origin.wait_stalled().await;
        // Give the first body frame time to reach the real worker. Its next frame remains
        // pending while this current-thread runtime is driven, then all I/O driving stops.
        tokio::time::sleep(Duration::from_millis(75)).await;
    });
    drop(cancel);
    let bounded = recv.recv_timeout(Duration::from_millis(500));
    let needed_driver = bounded.is_err();
    let (result, _, committed) = match bounded {
        Ok(result) => result,
        Err(_) => {
            // Rescue the owned worker before reporting red; runtime Drop must never hang.
            runtime.block_on(async { tokio::time::sleep(Duration::from_millis(100)).await });
            recv.recv_timeout(Duration::from_secs(2))
                .expect("worker failed to drain even after I/O resumed")
        }
    };
    assert_eq!(result.unwrap_err().code, Code::TRANSFER_FAILED);
    assert!(!committed);
    assert!(
        !needed_driver,
        "owned cancellation required restarting the caller runtime's I/O driver"
    );
}
