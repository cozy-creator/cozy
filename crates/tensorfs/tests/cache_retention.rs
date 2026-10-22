//! Holds keep unread objects across collectors in another process, and cache expires on its own.
//! No network or GPU.
#![cfg(all(unix, feature = "store"))]

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tensorfs::format::{
    Body, Component, Dtype, Header, Location, ObjectRef, Repo, Tag, Tensor, Tree,
};
use tensorfs::io::{
    source::{IoTally, Item, ItemSource, Mode, Source},
    Dst,
};
use tensorfs::io::ObjectRange;
use tensorfs::store::Store;
use tensorfs::writer::{Data, Source as WriterSource};

const LATE: &[u8] = b"\x85raw unread bytes, deliberately resembling a CBOR header";
const NEW: &[u8] = b"an admitted asset held by an idle writer";

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "retention-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn store(&self) -> Store {
        Store::open(self.0.join("store"))
            .unwrap()
            .with_grace(Duration::ZERO)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Serialize, Deserialize)]
struct Fixture {
    tree: ObjectRef,
    header: ObjectRef,
    held: Vec<ObjectRef>,
    late: ObjectRef,
    unused: ObjectRef,
    reused: ObjectRef,
    local: ObjectRef,
}

fn header(chunks: Vec<ObjectRef>) -> Header {
    let tensors = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| Tensor {
            key: format!("part{i}"),
            dtype: Dtype::U8,
            shape: vec![c.length],
            body: Body::Data(Location {
                chunk: i as u64,
                offset: 0,
                length: c.length,
            }),
        })
        .collect();
    Header::new(
        chunks,
        None,
        vec![Component {
            name: "c".into(),
            tensors,
            ..Default::default()
        }],
        Default::default(),
    )
    .unwrap()
}

fn fixture(scratch: &Scratch) -> Fixture {
    let store = scratch.store();
    let first = store
        .put_bytes(b"first payload, never opened by the source")
        .unwrap();
    let late = ObjectRef::of(LATE); // Intentionally absent until AFTER the lazy owner is opened.
    let h = store
        .put_bytes(&header(vec![first.clone(), late.clone()]).to_cbor().unwrap())
        .unwrap();
    let unused = store.put_bytes(b"unselected variant payload").unwrap();
    let other = store
        .put_bytes(&header(vec![unused.clone()]).to_cbor().unwrap())
        .unwrap();
    let readme = store
        .put_bytes(b"repository bytes outside the header")
        .unwrap();
    let mut tree = Tree::default();
    tree.set_variant("main", h.clone());
    tree.set_variant("other", other.clone());
    tree.files.insert("README.md".into(), readme.clone());
    let tree = store.put_tree(&tree).unwrap();
    let reused = store
        .put_bytes(b"existing bytes referenced by an idle writer")
        .unwrap();
    let local = store.put_bytes(b"durable local repository bytes").unwrap();
    let local_tree = store
        .put_tree(&Tree {
            files: [("README.md".into(), local.clone())].into(),
        })
        .unwrap();
    store
        .update_ref("local/keep", |_| {
            Some(Repo {
                name: "local/keep".into(),
                tags: vec![Tag {
                    name: "1.0".into(),
                    commit: local_tree,
                    public: true,
                    yanked: false,
                }],
                ..Repo::default()
            })
        })
        .unwrap();
    let fixture = Fixture {
        tree: tree.clone(),
        header: h.clone(),
        held: vec![first, late.clone(), h, tree, other, readme],
        late,
        unused,
        reused,
        local,
    };
    fs::write(
        scratch.0.join("fixture.json"),
        serde_json::to_vec(&fixture).unwrap(),
    )
    .unwrap();
    fixture
}

fn age(store: &Store) {
    fn walk(path: &Path, then: SystemTime) {
        for entry in fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                walk(&entry.path(), then);
            } else {
                File::open(entry.path())
                    .unwrap()
                    .set_modified(then)
                    .unwrap();
            }
        }
    }
    walk(
        &store.root().join("blobs"),
        SystemTime::now() - Duration::from_secs(7200),
    );
}
/// `object` reads through the store as its own bytes.
fn intact(store: &Store, object: &ObjectRef) -> bool {
    store.read(object).is_ok_and(|bytes| ObjectRef::of(&bytes) == *object)
}

fn present(store: &Store, objects: &[ObjectRef]) {
    for object in objects {
        assert!(intact(store, object), "lost {}", object.sha256);
    }
}
fn collected(store: &Store, objects: &[ObjectRef]) {
    for object in objects {
        assert!(
            !store.path(object).unwrap().exists(),
            "still retained {}",
            object.sha256
        );
    }
    let holds = fs::read_dir(store.root().join("holds")).unwrap();
    assert_eq!(holds.count(), 1, "a dead holder's record was not reaped");
}
fn pressure(store: &Store) {
    age(store);
    store.collect(u64::MAX, Duration::ZERO).unwrap();
}

struct Owner {
    child: Child,
    input: ChildStdin,
    lines: Receiver<String>,
}
impl Owner {
    fn start(scratch: &Scratch, mode: &str) -> Self {
        let mut command = Command::new(std::env::current_exe().unwrap());
        // The harness owns this process group, including the real fork child, on timeout.
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            });
        }
        let mut child = command
            .args([
                "--exact",
                "native_owner_process",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("TENSORFS_RETENTION_TEST", &scratch.0)
            .env("TENSORFS_RETENTION_MODE", mode)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        let (send, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                if send.send(line.unwrap()).is_err() {
                    break;
                }
            }
        });
        let mut owner = Self {
            child,
            input,
            lines,
        };
        owner.expect("ready");
        owner
    }
    fn expect(&mut self, expected: &str) -> String {
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            let line = self
                .lines
                .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
                .expect("native owner did not respond before deadline");
            if let Some((_, message)) = line.split_once("TFS:") {
                assert!(
                    message.starts_with(expected),
                    "expected {expected:?}, got {line:?}"
                );
                return message.to_owned();
            }
        }
    }
    fn send(&mut self, line: &str) {
        writeln!(self.input, "{line}").unwrap();
        self.input.flush().unwrap();
    }
    fn finish(mut self) {
        self.send("exit");
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success());
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "native owner failed to exit"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            unsafe {
                libc::killpg(self.child.id() as i32, libc::SIGKILL);
            }
            let _ = self.child.wait();
        }
    }
}
fn report(message: &str) {
    println!("TFS:{message}");
    std::io::stdout().flush().unwrap();
}
fn command() -> String {
    let mut line = String::new();
    assert!(std::io::stdin().read_line(&mut line).unwrap() > 0);
    line.trim().to_owned()
}
fn read_late(source: &Source, object: &ObjectRef) {
    let mut bytes = vec![0; object.length as usize];
    let item = Item {
        offset: 0,
        len: object.length,
        part_end: true,
        source: ItemSource::Object(ObjectRange {
            obj: tensorfs::format::ObjectRef {
                sha256: object.sha256.clone(),
                length: object.length,
            },
            off: 0,
            len: object.length,
        }),
    };
    unsafe {
        source
            .read_items(
                &[item],
                0,
                Dst::host(&mut bytes),
                1,
                Mode::Buffered,
                &IoTally::default(),
            )
            .unwrap();
    }
    assert_eq!(bytes, LATE);
}

#[test]
fn lazy_source_keeps_unread_and_future_selected_objects_across_process_gc() {
    let scratch = Scratch::new();
    let f = fixture(&scratch);
    let store = scratch.store();
    let mut owner = Owner::start(&scratch, "source");
    assert!(!store.path(&f.late).unwrap().exists());
    assert_eq!(store.put_bytes(LATE).unwrap(), f.late);
    pressure(&store);
    present(&store, &f.held);
    assert!(
        !store.path(&f.unused).unwrap().exists(),
        "an unselected payload was kept"
    );
    assert!(
        intact(&store, &f.local),
        "local repository ref was evicted"
    );
    owner.send("read");
    owner.expect("read");
    owner.send("release");
    owner.expect("released");
    pressure(&store);
    collected(&store, &f.held);
    assert!(intact(&store, &f.local));
    owner.finish();
}

#[test]
fn idle_writer_keeps_base_reused_and_admitted_bytes_until_commit() {
    let scratch = Scratch::new();
    let f = fixture(&scratch);
    let store = scratch.store();
    store.put_bytes(LATE).unwrap();
    let mut owner = Owner::start(&scratch, "writer");
    pressure(&store);
    present(&store, &f.held);
    present(&store, &[f.reused.clone(), ObjectRef::of(NEW)]);
    owner.send("commit");
    let message = owner.expect("committed ");
    let id: ObjectRef = serde_json::from_str(message.strip_prefix("committed ").unwrap()).unwrap();
    let ck = store.checkpoint(&id).unwrap();
    assert_eq!(ck.read("c", "part1").unwrap(), LATE);
    assert_eq!(ck.read("c", "fresh").unwrap(), b"tiny spool");
    assert_eq!(ck.asset("c", "new.bin").unwrap(), NEW);
    assert_eq!(
        ck.asset("c", "reused.bin").unwrap(),
        store.read(&f.reused).unwrap()
    );
    owner.finish();
}

#[test]
fn process_death_releases_idle_writer_roots_without_drop_cleanup() {
    let scratch = Scratch::new();
    let f = fixture(&scratch);
    let store = scratch.store();
    store.put_bytes(LATE).unwrap();
    let mut owner = Owner::start(&scratch, "writer");
    pressure(&store);
    present(&store, &[f.reused.clone(), ObjectRef::of(NEW)]);
    owner.child.kill().unwrap();
    owner.child.wait().unwrap();
    pressure(&store);
    collected(&store, &f.held);
    collected(&store, &[f.reused, ObjectRef::of(NEW)]);
    assert!(intact(&store, &f.local));
    store.sweep_writers(Duration::ZERO).unwrap();
    assert_eq!(
        fs::read_dir(store.root().join("tmp/writers"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn fork_inherited_owner_survives_parent_release_then_becomes_collectible() {
    let scratch = Scratch::new();
    let f = fixture(&scratch);
    let store = scratch.store();
    store.put_bytes(LATE).unwrap();
    let mut owner = Owner::start(&scratch, "fork");
    pressure(&store);
    present(&store, &f.held);
    owner.send("read");
    owner.expect("read");
    owner.send("release");
    owner.expect("released");
    pressure(&store);
    collected(&store, &f.held);
    owner.finish();
}

#[test]
#[ignore = "subprocess helper invoked by the four real process tests"]
fn native_owner_process() {
    let scratch = PathBuf::from(std::env::var_os("TENSORFS_RETENTION_TEST").unwrap());
    let f: Fixture =
        serde_json::from_slice(&fs::read(scratch.join("fixture.json")).unwrap()).unwrap();
    let store = Store::open(scratch.join("store")).unwrap();
    let ck = store.variant(&f.tree, Some("main")).unwrap();
    assert_eq!(ck.id(), &f.header);
    match std::env::var("TENSORFS_RETENTION_MODE").unwrap().as_str() {
        "source" => {
            let client = tensorfs::Client::new(tensorfs::Config {
                cache_dir: scratch.join("store"),
                ..tensorfs::Config::from_env()
            })
            .unwrap();
            let open = tensorfs::Open {
                variant: Some("main".into()),
                local_files_only: true,
                ..Default::default()
            };
            let reference = format!("sha256:{}", f.tree.sha256);
            let opened = tensorfs::raw::block_on(client.open(&reference, &open)).unwrap();
            let source = opened.source(false);
            drop((ck, opened, client));
            report("ready");
            assert_eq!(command(), "read");
            read_late(&source, &f.late);
            report("read");
            assert_eq!(command(), "release");
            drop(source);
            report("released");
            assert_eq!(command(), "exit");
        }
        "writer" => {
            let mut writer = store.writer(Some(&ck), None).unwrap();
            drop(ck);
            let mut c = writer.get("c");
            c.asset("new.bin", NEW).unwrap();
            c.reuse_asset("reused.bin", &f.reused).unwrap();
            c.put(
                "fresh",
                &Dtype::U8,
                &[10],
                Data::Plain(WriterSource::Read(&mut &b"tiny spool"[..])),
            )
            .unwrap();
            report("ready");
            assert_eq!(command(), "commit");
            let id = writer.commit().unwrap();
            report(&format!(
                "committed {}",
                serde_json::to_string(&id).unwrap()
            ));
            assert_eq!(command(), "exit");
        }
        "fork" => {
            let (mut parent, mut child) = UnixStream::pair().unwrap();
            // Fork occurs inside a fresh helper with no TensorFS background threads or activity.
            let pid = unsafe { libc::fork() };
            assert!(pid >= 0);
            if pid == 0 {
                drop(parent);
                let mut cmd = [0];
                child.read_exact(&mut cmd).unwrap();
                assert_eq!(cmd, [b'r']);
                assert_eq!(ck.read("c", "part1").unwrap(), LATE);
                child.write_all(b"r").unwrap();
                child.read_exact(&mut cmd).unwrap();
                assert_eq!(cmd, [b'q']);
                drop(ck);
                child.write_all(b"q").unwrap();
                unsafe { libc::_exit(0) };
            }
            drop(child);
            drop(ck); // Only the fork child's inherited OFD owns the native roots.
            report("ready");
            assert_eq!(command(), "read");
            parent.write_all(b"r").unwrap();
            let mut reply = [0];
            parent.read_exact(&mut reply).unwrap();
            assert_eq!(reply, [b'r']);
            report("read");
            assert_eq!(command(), "release");
            parent.write_all(b"q").unwrap();
            parent.read_exact(&mut reply).unwrap();
            assert_eq!(reply, [b'q']);
            let mut status = 0;
            assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
            assert_eq!(status, 0);
            report("released");
            assert_eq!(command(), "exit");
        }
        mode => panic!("unknown owner mode {mode}"),
    }
}

/// Ages every use past the TTL and collects by the TTL.
fn expire(store: &Store) {
    let then = SystemTime::now() - tensorfs::store::TTL - Duration::from_secs(60);
    for entry in fs::read_dir(store.root().join("uses")).unwrap() {
        File::open(entry.unwrap().path()).unwrap().set_modified(then).unwrap();
    }
    age(store);
    store.collect(u64::MAX, tensorfs::store::TTL).unwrap();
}

#[test]
fn writer_results_expire_but_local_replays_remain() {
    use tensorfs::fetch::{Client, Endpoint, SourcePolicy, Stream};

    let repo_bytes: [&[u8]; 2] = [
        b"\x85raw repository file, not a header",
        br#"{"format":"cozytensors.tree/1","files":{}}"#,
    ];
    for mode in ["anonymous", "keyed", "streamed"] {
        let scratch = Scratch::new();
        let store = scratch.store();
        let key = (mode == "keyed").then_some("managed-resume");
        if key.is_some() {
            let mut writer = store.writer(None, key).unwrap();
            writer
                .get("c")
                .put(
                    "weight",
                    &Dtype::U8,
                    &[5],
                    Data::Plain(WriterSource::Read(&mut &b"saved"[..])),
                )
                .unwrap();
            drop(writer); // Existing keyed Drop journals the small spool before resume.
        }
        let mut result = None;
        for (i, bytes) in repo_bytes.iter().enumerate() {
            let mut writer = if mode == "streamed" {
                let client = Client::new(SourcePolicy {
                    allow_local: true,
                    any_host: true,
                    ..Default::default()
                });
                let endpoint = Arc::new(
                    Endpoint::new(&client, scratch.0.join("remote").to_str().unwrap(), None)
                        .unwrap(),
                );
                let runtime = tensorfs::fetch::runtime().unwrap();
                store
                    .streaming_writer(
                        Arc::new(Stream::new(endpoint, runtime.handle().clone())),
                        "acme/result",
                        &format!("run-{i}"),
                        "",
                        None,
                    )
                    .unwrap()
            } else {
                store.writer(None, key).unwrap()
            };
            if mode == "keyed" && i == 0 {
                assert!(
                    writer.has("c", "weight"),
                    "the recent journal did not resume"
                );
            } else {
                assert!(
                    !writer.has("c", "weight"),
                    "a completed key is not a journal lookup"
                );
            }
            if !writer.has("c", "weight") {
                writer
                    .get("c")
                    .put(
                        "weight",
                        &Dtype::U8,
                        &[5],
                        Data::Plain(WriterSource::Read(&mut &b"saved"[..])),
                    )
                    .unwrap();
            }
            writer.repo_file("README.bin", bytes).unwrap();
            let id = writer.commit().unwrap();
            if let Some(previous) = &result {
                assert_eq!(
                    previous, &id,
                    "repo-only changes must not change the header"
                );
            }
            result = Some(id.clone());
            let ck = store.checkpoint(&id).unwrap();
            assert_eq!(ck.read("c", "weight").unwrap(), b"saved");
            assert_eq!(store.read(&ObjectRef::of(bytes)).unwrap(), *bytes);
            drop(ck); // Reopening is a use of the header; the repo files' use must survive it.
        }
        for bytes in repo_bytes {
            assert_eq!(store.read(&ObjectRef::of(bytes)).unwrap(), bytes);
        }
        let id = result.unwrap();
        let h = Header::from_cbor(&store.read(&id).unwrap()).unwrap();
        expire(&store);
        collected(&store, &[id]);
        collected(&store, &h.chunks);
        for bytes in repo_bytes {
            collected(&store, &[ObjectRef::of(bytes)]);
        }
        if mode == "streamed" {
            let remote = Store::open(scratch.0.join("remote")).unwrap();
            let repo = remote.get_ref("acme/result").unwrap().unwrap();
            assert_eq!(repo.tags.len(), 2);
            for tag in repo.tags {
                for object in remote.closure(&tag.commit).unwrap() {
                    assert!(intact(&remote, &object));
                }
            }
        }
    }

    let scratch = Scratch::new();
    let raw = scratch.store();
    let mut writer = raw.writer(None, Some("operator-key")).unwrap();
    writer
        .get("c")
        .put(
            "weight",
            &Dtype::U8,
            &[5],
            Data::Plain(WriterSource::Read(&mut &b"saved"[..])),
        )
        .unwrap();
    let id = writer.commit().unwrap();
    let mut tree = Tree::default();
    tree.set_variant("", id.clone());
    let root = raw.put_tree(&tree).unwrap();
    raw.update_ref("local/derived", |_| {
        Some(Repo {
            name: "local/derived".into(),
            tags: vec![Tag {
                name: "replay-key".into(),
                commit: root,
                public: false,
                yanked: false,
            }],
            ..Repo::default()
        })
    })
    .unwrap();
    expire(&raw);
    let repo = raw.get_ref("local/derived").unwrap().unwrap();
    let ck = raw.variant(&repo.tags[0].commit, None).unwrap();
    assert_eq!(ck.read("c", "weight").unwrap(), b"saved");
}

#[test]
fn failed_stream_retains_local_bytes_until_the_last_holder() {
    use tensorfs::fetch::{Client, Endpoint, SourcePolicy, Stream};

    let scratch = Scratch::new();
    let store = scratch.store();
    // A real file cannot be an endpoint directory. The asynchronous upload fails at flush.
    let destination = scratch.0.join("not-a-directory");
    fs::write(&destination, b"an actual filesystem failure").unwrap();
    let client = Client::new(SourcePolicy {
        allow_local: true,
        any_host: true,
        ..Default::default()
    });
    let endpoint = Arc::new(Endpoint::new(&client, destination.to_str().unwrap(), None).unwrap());
    let runtime = tensorfs::fetch::runtime().unwrap();
    let mut writer = store
        .streaming_writer(
            Arc::new(Stream::new(endpoint, runtime.handle().clone())),
            "acme/failed",
            "main",
            "",
            None,
        )
        .unwrap();
    writer
        .get("c")
        .put(
            "weight",
            &Dtype::U8,
            &[5],
            Data::Plain(WriterSource::Read(&mut &b"saved"[..])),
        )
        .unwrap();
    writer.repo_file("README.bin", LATE).unwrap();
    assert!(writer.commit().is_err());
    assert!(writer.published().is_none());
    let id = writer
        .committed()
        .expect("local commit must precede failed upload")
        .clone();
    expire(&store); // The failed Writer is the only holder at this point.
    assert!(
        intact(&store, &id),
        "failed Writer lost its committed header before an error checkpoint was opened"
    );
    let ck = store.checkpoint(&id).unwrap();
    let chunks = ck.header().chunks.clone();
    expire(&store);
    assert_eq!(ck.read("c", "weight").unwrap(), b"saved");
    assert_eq!(
        store.read(&ObjectRef::of(LATE)).unwrap(),
        LATE,
        "failed Writer still owns its repo bytes"
    );
    drop(writer);
    expire(&store);
    assert_eq!(
        ck.read("c", "weight").unwrap(),
        b"saved",
        "the retained checkpoint survives the failed Writer"
    );
    drop(ck);
    expire(&store);
    collected(&store, &[id, ObjectRef::of(LATE)]);
    collected(&store, &chunks);
}
