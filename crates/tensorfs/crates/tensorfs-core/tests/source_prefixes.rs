use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    os::unix::fs::MetadataExt,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};
use tensorfs_core::{
    catalog::WriterGuard,
    gc,
    ids::{Doc, ObjectRef},
    providers::{self, Provenance, Resolution, ResolvedMember},
    source_artifact,
    store::{Fault, Store},
    transport::{Anonymous, Deadline, PullCancellation, SourceDownload, SourcePolicy},
};
const BLOCK: u64 = 1 << 20;
// These controls deliberately fork/kill owners. A sibling fork can temporarily
// inherit another test's shared recovery lock until exec, even with CLOEXEC.
// Serializing fixture lifetimes keeps each GC assertion's no-other-owner premise true.
static PROCESS_FIXTURE: Mutex<()> = Mutex::new(());
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
            companion: false,
        }],
    }
}
fn run(
    store: &Store,
    url: &str,
    id: &str,
    fault: Fault,
) -> Result<(source_artifact::TreeRoot, Vec<providers::Pulled>), tensorfs_core::err::Refusal> {
    run_with(store, url, id, fault, None)
}
fn run_with(
    store: &Store,
    url: &str,
    id: &str,
    fault: Fault,
    cancellation: Option<PullCancellation>,
) -> Result<(source_artifact::TreeRoot, Vec<providers::Pulled>), tensorfs_core::err::Refusal> {
    providers::materialize_selected_with_options(
        store,
        id,
        &resolution(url),
        &SourcePolicy {
            allowed_hosts: vec!["127.0.0.1".into()],
            allow_local: true,
            max_redirects: 5,
            ..Default::default()
        },
        &Anonymous,
        Deadline::none(),
        &|_, _| {},
        // One stream: these kill points prove the order a chunk is made durable in.
        &SourceDownload {
            checkpoint_bytes: BLOCK,
            streams: 1,
            fault,
            cancellation,
        },
    )
}
#[test]
fn transfer_child() {
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
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
    // The fault cut is authorized by its actual marker, not by how long a busy
    // filesystem took to admit the prefix. A terminal child still fails loudly.
    loop {
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
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
    for (stage, expected) in [
        ("chunk-written", 0),
        ("chunk-fsync", 0),
        ("chunk-recorded", BLOCK),
        ("later-chunk-written", BLOCK),
        ("verified", 3 * BLOCK + 19),
        ("linked", 3 * BLOCK + 19),
        ("record", 3 * BLOCK + 19),
        ("admitted", 3 * BLOCK + 19),
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
fn tampered_chunks_restart_the_object_and_an_ignored_range_takes_it_whole() {
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
    for (tamper, ignore_range) in [(true, false), (false, true)] {
        let dir = root();
        let store = Store::init(&dir).unwrap();
        let origin = Origin::new(false);
        kill_at(&dir, &origin, "chunk-recorded");
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
        // A tampered chunk no longer hashes to its record: it is forgotten and fetched
        // again. An origin that ignores ranges gives the object whole from byte 0.
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
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
    let dir = root();
    let store = Store::init(&dir).unwrap();
    let origin = Origin::new(false);
    kill_at(&dir, &origin, "chunk-recorded");
    assert_eq!(store.occupancy().unwrap(), BLOCK);
    let pending = source_artifact::read(&store, &owner("producer"))
        .unwrap()
        .unwrap();
    assert_eq!(pending.objects, vec![ObjectRef::of(&bytes())]);
    assert!(
        pending.receipt().is_err(),
        "liveness intent cannot produce a reusable receipt"
    );
    gc::collect(&dir, false).unwrap();
    let adopted =
        source_artifact::adopt_partial(&store, &owner("producer"), &owner("recipient")).unwrap();
    assert!(
        !adopted.complete && adopted.objects.is_empty(),
        "missing reserved object became completed work"
    );
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
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
    for encoded in [false, true] {
        let dir = root();
        let store = Store::init(&dir).unwrap();
        let origin = Origin::new(false);
        kill_at(&dir, &origin, "chunk-recorded");
        let rejected = Origin::mode(false, !encoded, encoded);
        // An expired URL ends the run at once. A transformed answer is asked again until
        // the pull's patience, so the caller cancels it once it has been asked twice.
        let cancellation = PullCancellation::default();
        let asks = rejected.asks.clone();
        let stop = cancellation.clone();
        let watcher = thread::spawn(move || {
            while asks.lock().unwrap().len() < 2 && !stop.is_cancelled() {
                thread::sleep(Duration::from_millis(5));
            }
            stop.cancel();
        });
        assert!(run_with(
            &store,
            &rejected.url,
            &owner("producer"),
            Fault::default(),
            Some(cancellation.clone()),
        )
        .is_err());
        cancellation.cancel();
        watcher.join().unwrap();
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

fn fresh_owner_after_admission(stage: &str, remove_cas: bool) {
    let dir = root();
    let store = Store::init(&dir).unwrap();
    let origin = Origin::new(false);
    kill_at(&dir, &origin, stage);
    let object = ObjectRef::of(&bytes());
    let prefix = dir
        .join("roots/source-downloads")
        .join(&owner("producer")[7..])
        .join(&object.sha256);
    let before_state = fs::read(prefix.join("chunks")).unwrap();
    let before_bytes = fs::metadata(prefix.join("bytes.part")).ok();
    if stage == "linked" {
        assert_eq!(
            before_bytes.as_ref().unwrap().ino(),
            fs::metadata(store.object_path(&object.sha256))
                .unwrap()
                .ino()
        );
        assert_eq!(before_bytes.as_ref().unwrap().mode() & 0o222, 0);
    }
    if remove_cas {
        // The link existed before its directory fsync; simulate the surviving
        // durable prefix without that CAS name, keeping the exact donor inode.
        fs::remove_file(store.object_path(&object.sha256)).unwrap();
    }
    let prior = source_artifact::read(&store, &owner("producer"))
        .unwrap()
        .unwrap();
    assert!(
        !prior.complete && prior.objects == vec![object.clone()],
        "incomplete source must durably reserve the exact member before admission"
    );
    gc::collect(&dir, false).unwrap(); // a dead source writer no longer excludes GC
    let adopted = source_artifact::adopt_partial(&store, &owner("producer"), &owner("recipient"))
        .unwrap_or_else(|error| panic!("fresh owner after {stage}: {error:?}"));
    assert!(
        adopted.objects.contains(&object),
        "verified CAS member needs independent recipient custody"
    );
    assert_eq!(fs::read(prefix.join("chunks")).unwrap(), before_state);
    if let Some(before) = before_bytes {
        let after = fs::metadata(prefix.join("bytes.part")).unwrap();
        assert_eq!(
            (after.ino(), after.len(), after.mode()),
            (before.ino(), before.len(), before.mode())
        );
        assert_eq!(fs::read(prefix.join("bytes.part")).unwrap(), bytes());
    } else {
        assert!(
            !prefix.join("bytes.part").exists(),
            "adoption recreated removed donor bytes"
        );
    }
    assert!(
        !dir.join("roots/source-downloads")
            .join(&owner("recipient")[7..])
            .exists(),
        "complete objects must not become copied recipient prefixes"
    );
    source_artifact::release(&store, &owner("producer")).unwrap();
    gc::collect(&dir, false).unwrap();
    let requests = origin.asks.lock().unwrap().len();
    let (_, rows) = run(&store, &origin.url, &owner("recipient"), Fault::default()).unwrap();
    assert_eq!(rows[0].transferred, 0);
    assert_eq!(origin.asks.lock().unwrap().len(), requests);
    assert_eq!(
        store.read_range(&object.sha256, 0, object.length).unwrap(),
        bytes()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn fresh_owner_after_link_before_record() {
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
    fresh_owner_after_admission("linked", false);
}
#[test]
fn fresh_owner_after_record_before_landed() {
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
    fresh_owner_after_admission("record", false);
}
#[test]
fn fresh_owner_after_admission_before_landed() {
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
    fresh_owner_after_admission("admitted", false);
}

#[test]
fn fresh_owner_admits_readonly_prefix_without_cas_link() {
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
    fresh_owner_after_admission("chmod", false);
    fresh_owner_after_admission("linked", true);
}

#[test]
fn forked_descriptor_keeps_recovery_excluded_after_parent_guard_drop() {
    let _process_fixture = PROCESS_FIXTURE.lock().unwrap();
    let dir = root();
    let _store = Store::init(&dir).unwrap();
    let writer = WriterGuard::acquire(&dir).unwrap();
    let mut signal = [0; 2];
    assert_eq!(
        unsafe { libc::pipe2(signal.as_mut_ptr(), libc::O_CLOEXEC) },
        0
    );
    let child = unsafe { libc::fork() };
    assert!(child >= 0);
    if child == 0 {
        // Only async-signal-safe syscalls between fork and exit.
        unsafe {
            libc::close(signal[1]);
            let mut byte = 0u8;
            libc::read(signal[0], (&mut byte as *mut u8).cast(), 1);
            libc::_exit(0);
        }
    }
    unsafe {
        libc::close(signal[0]);
    }
    drop(writer);
    let held = gc::collect(&dir, false);
    // Always release/reap the child before asserting the result.
    unsafe {
        libc::write(signal[1], b"x".as_ptr().cast(), 1);
        libc::close(signal[1]);
        let mut status = 0;
        assert_eq!(libc::waitpid(child, &mut status, 0), child);
        assert_eq!(status, 0);
    }
    let refused = held.unwrap_err();
    assert_eq!(refused.code, tensorfs_core::err::Code::STORE_BUSY);
    assert!(refused.detail.contains("recovery lock shared"));
    gc::collect(&dir, false).unwrap();
    fs::remove_dir_all(dir).unwrap();
}
