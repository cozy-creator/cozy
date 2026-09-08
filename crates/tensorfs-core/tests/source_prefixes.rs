use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};
use tensorfs_core::{
    gc,
    ids::{Doc, ObjectRef},
    providers::{self, Provenance, Resolution, ResolvedMember},
    source_artifact,
    store::{Fault, Store},
    transport::{Anonymous, Deadline, SourceDownload, SourcePolicy},
};
const BLOCK: u64 = 1 << 20;
fn bytes() -> Vec<u8> {
    (0..3 * BLOCK + 19)
        .map(|i| (i.wrapping_mul(17) % 251) as u8)
        .collect()
}
fn owner(label: &str) -> String {
    ObjectRef::of(label.as_bytes()).id()
}
fn root() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "tfs-prefix-{}-{}",
        std::process::id(),
        tensorfs_core::meta::now_nanos_unique()
    ))
}
struct Origin {
    url: String,
    asks: Arc<Mutex<Vec<u64>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Origin {
    fn new(ignore_range: bool) -> Self {
        Self::mode(ignore_range, false, false)
    }
    fn mode(ignore_range: bool, expired: bool, encoded: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let asks = Arc::new(Mutex::new(vec![]));
        let stop = Arc::new(AtomicBool::new(false));
        let seen = asks.clone();
        let quitting = stop.clone();
        let task = thread::spawn(move || {
            let body = bytes();
            for socket in listener.incoming() {
                let mut socket = socket.unwrap();
                if quitting.load(Ordering::Relaxed) {
                    break;
                }
                let mut request = vec![];
                let mut buffer = [0u8; 1024];
                while !request.ends_with(b"\r\n\r\n") {
                    let n = socket.read(&mut buffer).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..n]);
                }
                // A killed/retrying client may close while sending its request headers.
                if !request.ends_with(b"\r\n\r\n") {
                    continue;
                }
                let text = String::from_utf8_lossy(&request).to_lowercase();
                let range = text
                    .lines()
                    .find_map(|line| line.strip_prefix("range: bytes="))
                    .unwrap();
                let (a, b) = range.trim().split_once('-').unwrap();
                let start = a.parse::<u64>().unwrap();
                let end = b.parse::<u64>().unwrap();
                seen.lock().unwrap().push(start);
                if expired {
                    let _ = socket.write_all(
                        b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                }
                let response = if ignore_range {
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                } else {
                    format!("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len(),end-start+1)
                };
                let response = if encoded {
                    response.replace(
                        "Connection: close",
                        "Content-Encoding: gzip\r\nConnection: close",
                    )
                } else {
                    response
                };
                let _ = socket.write_all(response.as_bytes());
                let _ = socket.write_all(if ignore_range {
                    &body
                } else {
                    &body[start as usize..=end as usize]
                });
            }
        });
        Self {
            url,
            asks,
            stop,
            thread: Some(task),
        }
    }
}
impl Drop for Origin {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = TcpStream::connect(self.url.trim_start_matches("http://"));
        let joined = self.thread.take().unwrap().join();
        if !thread::panicking() {
            joined.unwrap();
        }
    }
}
fn resolution(url: &str) -> Resolution {
    Resolution {
        canonical: "civitai://123".into(),
        selection_sha256: "previous accepted identity".into(),
        members: vec![ResolvedMember {
            member: "model.safetensors".into(),
            object: ObjectRef::of(&bytes()),
            url: format!("{url}/model"),
            provenance: Provenance::Declared,
            carrier: true,
            requires: vec![],
        }],
    }
}
fn run(
    store: &Store,
    url: &str,
    id: &str,
    fault: Fault,
) -> Result<(source_artifact::TreeRoot, Vec<providers::Pulled>), tensorfs_core::err::Refusal> {
    providers::materialize_selected_with_options(
        store,
        id,
        &resolution(url),
        &SourcePolicy {
            allowed_hosts: vec!["127.0.0.1".into()],
            allow_local: true,
            max_redirects: 5,
        },
        &Anonymous,
        Deadline::none(),
        &mut |_, _| {},
        &SourceDownload {
            checkpoint_bytes: BLOCK,
            fault,
        },
    )
}
#[test]
fn transfer_child() {
    let Some(dir) = std::env::var_os("TFS_PREFIX_CHILD") else {
        return;
    };
    let store = Store::open(std::path::Path::new(&dir)).unwrap();
    let url = std::env::var("TFS_PREFIX_ORIGIN").unwrap();
    let stage = std::env::var("TFS_PREFIX_STAGE").unwrap();
    run(
        &store,
        &url,
        &owner("producer"),
        Fault {
            stage: Some(stage),
            ready: Some(store.root().join("ready")),
        },
    )
    .unwrap();
}
fn kill_at(dir: &std::path::Path, origin: &Origin, stage: &str) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "transfer_child", "--nocapture"])
        .env("TFS_PREFIX_CHILD", dir)
        .env("TFS_PREFIX_ORIGIN", &origin.url)
        .env("TFS_PREFIX_STAGE", stage)
        .stdout(Stdio::null())
        .stderr(fs::File::create(dir.join("child-stderr")).unwrap())
        .spawn()
        .unwrap();
    for _ in 0..2000 {
        if dir.join("ready").is_file() {
            break;
        }
        assert!(
            child.try_wait().unwrap().is_none(),
            "child exited before {stage}: {}",
            fs::read_to_string(dir.join("child-stderr")).unwrap_or_default()
        );
        thread::sleep(Duration::from_millis(5));
    }
    assert!(dir.join("ready").is_file(), "never reached {stage}");
    child.kill().unwrap();
    child.wait().unwrap();
}
#[test]
fn sigkill_resumes_only_verified_prefix_and_never_downloads_completed_objects() {
    for (stage, expected) in [
        ("prefix-written", 0),
        ("prefix-fsync", 0),
        ("prefix-checkpoint", BLOCK),
        ("prefix-next-written", BLOCK),
        ("prefix-verified", 3 * BLOCK + 19),
        ("linked", 3 * BLOCK + 19),
        ("record", 3 * BLOCK + 19),
        ("prefix-admitted", 3 * BLOCK + 19),
    ] {
        let dir = root();
        let store = Store::init(&dir).unwrap();
        let origin = Origin::new(false);
        kill_at(&dir, &origin, stage);
        let old = origin.asks.lock().unwrap().len();
        assert!(source_artifact::read(&store, &owner("producer"))
            .unwrap()
            .is_some());
        if expected < 3 * BLOCK + 19 {
            assert!(!store.contains(&ObjectRef::of(&bytes()).sha256));
        }
        store.reap().unwrap(); // generic temp cleanup cannot erase owned progress
        let (_, rows) = run(&store, &origin.url, &owner("producer"), Fault::default()).unwrap();
        let asks = origin.asks.lock().unwrap();
        if expected < 3 * BLOCK + 19 {
            assert_eq!(asks[old], expected, "stage={stage}");
        } else {
            assert_eq!(asks.len(), old, "stage={stage}");
        }
        drop(asks);
        assert_eq!(
            rows[0].transferred,
            3 * BLOCK + 19 - expected,
            "stage={stage}"
        );
        assert_eq!(
            store
                .read_range(&ObjectRef::of(&bytes()).sha256, 0, 3 * BLOCK + 19)
                .unwrap(),
            bytes()
        );
        let before = origin.asks.lock().unwrap().len();
        run(&store, &origin.url, &owner("producer"), Fault::default()).unwrap();
        assert_eq!(origin.asks.lock().unwrap().len(), before);
        fs::remove_dir_all(dir).unwrap();
    }
}
#[test]
fn tampered_prefix_restarts_and_ignored_range_restarts_honestly() {
    for (tamper, ignore_range) in [(true, false), (false, true)] {
        let dir = root();
        let store = Store::init(&dir).unwrap();
        let origin = Origin::new(false);
        kill_at(&dir, &origin, "prefix-checkpoint");
        if tamper {
            let path = dir
                .join("roots/source-downloads")
                .join(&owner("producer")[7..])
                .join(ObjectRef::of(&bytes()).sha256)
                .join("bytes.part");
            let mut file = fs::OpenOptions::new().write(true).open(path).unwrap();
            file.write_all(b"corrupt").unwrap();
        }
        let replacement = Origin::new(ignore_range);
        if tamper {
            let refusal = run(
                &store,
                &replacement.url,
                &owner("producer"),
                Fault::default(),
            )
            .unwrap_err();
            assert_eq!(refusal.code, tensorfs_core::err::Code::OBJECT_CORRUPT);
            assert!(replacement.asks.lock().unwrap().is_empty());
        }
        let (_, rows) = run(
            &store,
            &replacement.url,
            &owner("producer"),
            Fault::default(),
        )
        .unwrap();
        assert_eq!(rows[0].transferred, 3 * BLOCK + 19);
        assert_eq!(
            replacement.asks.lock().unwrap()[0],
            if tamper { 0 } else { BLOCK }
        );
        assert_eq!(
            store
                .read_range(&ObjectRef::of(&bytes()).sha256, 0, 3 * BLOCK + 19)
                .unwrap(),
            bytes()
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
#[test]
fn partial_adoption_survives_source_release_and_gc_and_disk_budget_counts_prefixes() {
    let dir = root();
    let store = Store::init(&dir).unwrap();
    let origin = Origin::new(false);
    kill_at(&dir, &origin, "prefix-checkpoint");
    assert_eq!(store.occupancy().unwrap(), BLOCK);
    source_artifact::adopt_partial(&store, &owner("producer"), &owner("recipient")).unwrap();
    assert_eq!(store.occupancy().unwrap(), 2 * BLOCK);
    source_artifact::release(&store, &owner("producer")).unwrap();
    assert_eq!(store.occupancy().unwrap(), BLOCK);
    gc::collect(&dir, false).unwrap();
    let replacement = Origin::new(false);
    let (_, rows) = run(
        &store,
        &replacement.url,
        &owner("recipient"),
        Fault::default(),
    )
    .unwrap();
    assert_eq!(replacement.asks.lock().unwrap()[0], BLOCK);
    assert_eq!(rows[0].transferred, 2 * BLOCK + 19);
    let tree = providers::content_manifest(&resolution(&replacement.url)).unwrap();
    assert_eq!(
        store.occupancy().unwrap(),
        3 * BLOCK + 19 + tree.canonical_bytes().len() as u64
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn expired_authority_and_transformed_responses_preserve_verified_identity_prefix() {
    for encoded in [false, true] {
        let dir = root();
        let store = Store::init(&dir).unwrap();
        let origin = Origin::new(false);
        kill_at(&dir, &origin, "prefix-checkpoint");
        let rejected = Origin::mode(false, !encoded, encoded);
        assert!(run(&store, &rejected.url, &owner("producer"), Fault::default()).is_err());
        let replacement = Origin::new(false);
        let (_, rows) = run(
            &store,
            &replacement.url,
            &owner("producer"),
            Fault::default(),
        )
        .unwrap();
        let expected = BLOCK;
        assert_eq!(replacement.asks.lock().unwrap()[0], expected);
        assert_eq!(rows[0].transferred, 3 * BLOCK + 19 - expected);
        fs::remove_dir_all(dir).unwrap();
    }
}
