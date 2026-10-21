//! The store through real processes and files: SIGKILL at each kill point of a put, a record
//! commit and a GC, then reopen and check; a writer racing a collector; least recently used
//! first over shared chunks; and the two durability classes, counted by strace.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tensorfs::format::{
    variant_of, Body, Component, Dtype, Header, Location, ObjectRef, Repo, Tag, Tensor, Tree,
};
use tensorfs::store::{fault, Durability, Store};

const HOUR: Duration = Duration::from_secs(3600);

struct Scratch(PathBuf);

impl Scratch {
    fn new(under: &Path, name: &str) -> Scratch {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = under.join(format!("store-v2-{name}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }

    fn here(name: &str) -> Scratch {
        Scratch::new(Path::new(env!("CARGO_TARGET_TMPDIR")), name)
    }

    fn root(&self) -> PathBuf {
        self.0.join("store")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn bytes(tag: &str, n: usize) -> Vec<u8> {
    format!("{tag}:")
        .into_bytes()
        .into_iter()
        .cycle()
        .take(n)
        .collect()
}

/// A tree: its variant `int8`'s chunks and header over them, and the repo `files`; its ref.
fn checkpoint(store: &Store, chunks: &[Vec<u8>], files: &[(&str, Vec<u8>)]) -> ObjectRef {
    let chunks: Vec<ObjectRef> = chunks.iter().map(|c| store.put_bytes(c).unwrap()).collect();
    let tensors = chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            let at = Location {
                chunk: i as u64,
                offset: 0,
                length: chunk.length,
            };
            Tensor {
                key: format!("blocks.{i}.weight"),
                dtype: Dtype::U8,
                shape: vec![chunk.length],
                body: Body::Data(at),
            }
        })
        .collect();
    let unet = Component {
        name: "unet".into(),
        tensors,
        ..Default::default()
    };
    let header = Header::new(chunks, None, vec![unet], Default::default()).unwrap();
    let mut tree = Tree::default();
    tree.set_variant("int8", store.put_bytes(&header.to_cbor().unwrap()).unwrap());
    for (path, file) in files {
        tree.files
            .insert((*path).into(), store.put_bytes(file).unwrap());
    }
    store.put_tree(&tree).unwrap()
}

fn repo(name: &str, tree: ObjectRef) -> Repo {
    Repo {
        name: name.into(),
        tags: vec![Tag {
            name: "1.0".into(),
            commit: tree,
            public: true,
            yanked: false,
        }],
    }
}

/// `root`, and when it is a tree, its files and its headers' chunks: walked here, apart from the
/// store's own GC.
fn closure(store: &Store, root: &ObjectRef) -> BTreeSet<ObjectRef> {
    let mut reach = BTreeSet::from([root.clone()]);
    let Ok(Ok(tree)) = store.read(root).map(|b| Tree::from_json(&b)) else {
        return reach;
    };
    for (path, file) in tree.files {
        if variant_of(&path).is_some() {
            let header = Header::from_cbor(&store.read(&file).unwrap()).unwrap();
            reach.extend(header.chunks);
        }
        reach.insert(file);
    }
    reach
}

/// Everything a `local/` ref reaches.
fn live(store: &Store) -> BTreeSet<ObjectRef> {
    let local = store.refs().unwrap().into_iter().filter(|r| r.name.starts_with("local/"));
    let roots: Vec<ObjectRef> = local.flat_map(|r| r.tags).map(|t| t.commit).collect();
    roots.iter().flat_map(|root| closure(store, root)).collect()
}

/// Roots `objects` under the local repo `name`: a tree of them, released.
fn name_locally(store: &Store, name: &str, objects: &[ObjectRef]) {
    let files = objects.iter().enumerate().map(|(i, o)| (format!("{i}.bin"), o.clone()));
    let tree = store.put_tree(&Tree { files: files.collect() }).unwrap();
    store.update_ref(name, |_| Some(repo(name, tree))).unwrap();
}

/// Every object under `blobs/`, as named on disk.
fn blobs(store: &Store) -> BTreeSet<ObjectRef> {
    let mut found = BTreeSet::new();
    for top in fs::read_dir(store.root().join("blobs")).unwrap() {
        for shard in fs::read_dir(top.unwrap().path()).unwrap() {
            for entry in fs::read_dir(shard.unwrap().path()).unwrap() {
                let entry = entry.unwrap();
                found.insert(ObjectRef {
                    sha256: entry.file_name().into_string().unwrap(),
                    length: entry.metadata().unwrap().len(),
                });
            }
        }
    }
    found
}

/// The invariants after any crash: no object is torn, and everything live is whole.
fn check(store: &Store) {
    for object in blobs(store) {
        assert!(
            store.verify(&object).unwrap(),
            "torn object {}",
            object.sha256
        );
    }
    for object in live(store) {
        assert!(
            store.verify(&object).unwrap(),
            "live object {} lost",
            object.sha256
        );
    }
}

fn age(store: &Store, objects: impl IntoIterator<Item = ObjectRef>, by: Duration) {
    let then = SystemTime::now() - by;
    for object in objects {
        File::open(store.path(&object).unwrap())
            .unwrap()
            .set_modified(then)
            .unwrap();
    }
}

fn tmp_files(store: &Store) -> Vec<String> {
    fs::read_dir(store.root().join("tmp"))
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| e.file_type().unwrap().is_file())
        .map(|e| e.file_name().into_string().unwrap())
        .collect()
}

fn trash(store: &Store) -> usize {
    fs::read_dir(store.root().join("tmp/gc")).map_or(0, |d| d.count())
}

#[derive(Serialize, Deserialize)]
struct Request {
    root: PathBuf,
    durability: Option<String>,
    grace: u64,
    arm: Option<String>,
    job: Job,
}

#[derive(Serialize, Deserialize)]
enum Job {
    Put(Vec<u8>),
    Ref(String, ObjectRef),
    Gc,
    /// Each round: a fresh object and a reused old one, rooted together, and a loose object.
    Writer {
        rounds: usize,
        old: Vec<Vec<u8>>,
    },
    /// One GC per line on stdin, until it closes.
    Collector,
    /// Puts each object and a tree of them, and names that tree in two refs.
    Classes(Vec<Vec<u8>>),
}

fn request(root: &Path, arm: Option<&str>, job: Job) -> Request {
    Request {
        root: root.into(),
        durability: None,
        grace: 600,
        arm: arm.map(Into::into),
        job,
    }
}

#[test]
#[ignore = "a child process of the tests in this file"]
fn child() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let request: Request = serde_json::from_str(&line).unwrap();
    if let Some(point) = &request.arm {
        fault::arm(point);
    }
    let mut store = Store::open(&request.root)
        .unwrap()
        .with_grace(Duration::from_secs(request.grace));
    match request.durability.as_deref() {
        Some("durable") => store = store.with_durability(Durability::Durable),
        Some("ephemeral") => store = store.with_durability(Durability::Ephemeral),
        _ => {}
    }
    match request.job {
        Job::Put(object) => {
            store.put_bytes(&object).unwrap();
        }
        Job::Ref(name, manifest) => {
            store
                .update_ref(&name, |_| Some(repo(&name, manifest)))
                .unwrap();
        }
        Job::Gc => {
            store.gc().unwrap();
        }
        Job::Writer { rounds, old } => {
            for i in 0..rounds {
                let fresh = store
                    .put_bytes(&bytes(&format!("fresh-{i}"), 3000))
                    .unwrap();
                let again = store.put_bytes(&old[i % old.len()]).unwrap();
                name_locally(&store, &format!("local/w-{i}"), &[fresh, again]);
                store
                    .put_bytes(&bytes(&format!("loose-{i}"), 2000))
                    .unwrap();
                println!("ROUND {i}");
            }
        }
        Job::Collector => {
            let mut runs = 0;
            while std::io::stdin().read_line(&mut String::new()).unwrap() > 0 {
                store.gc().unwrap();
                runs += 1;
            }
            println!("RUNS {runs}");
        }
        Job::Classes(objects) => {
            let roots: Vec<ObjectRef> = objects
                .iter()
                .map(|o| store.put_bytes(o).unwrap())
                .collect();
            name_locally(&store, "local/a", &roots);
            let tree = store.get_ref("local/a").unwrap().unwrap().tags[0].commit.clone();
            store.update_ref("local/b", |_| Some(repo("local/b", tree))).unwrap();
        }
    }
    println!("DONE");
}

struct Child {
    process: std::process::Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Child {
    fn spawn(request: &Request) -> Child {
        Child::start(Command::new(std::env::current_exe().unwrap()), request)
    }

    /// Under strace, which logs the child's `fsync`, `fdatasync` and `syncfs` calls to `log`.
    fn traced(request: &Request, log: &Path) -> Child {
        let mut command = Command::new("strace");
        command
            .args(["-f", "-qq", "-e", "trace=fsync,fdatasync,syncfs", "-o"])
            .arg(log)
            .arg(std::env::current_exe().unwrap());
        Child::start(command, request)
    }

    fn start(mut command: Command, request: &Request) -> Child {
        let mut process = command
            .args([
                "--exact",
                "child",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut input = process.stdin.take().unwrap();
        writeln!(input, "{}", serde_json::to_string(request).unwrap()).unwrap();
        let output = BufReader::new(process.stdout.take().unwrap());
        Child {
            process,
            input: Some(input),
            output,
        }
    }

    fn wait_for(&mut self, prefix: &str) -> String {
        loop {
            let mut line = String::new();
            let n = self.output.read_line(&mut line).unwrap();
            assert_ne!(n, 0, "the child ended before {prefix}");
            // libtest prints `test child ... ` with no newline before the child's own lines.
            if let Some(at) = line.find(prefix) {
                return line[at..].trim().to_owned();
            }
        }
    }

    fn resume(&mut self) {
        writeln!(self.input.as_mut().unwrap()).unwrap();
    }

    fn kill(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }

    fn finish(mut self) -> String {
        let done = self.wait_for("DONE");
        self.input.take();
        assert!(self.process.wait().unwrap().success());
        done
    }
}

impl Drop for Child {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Runs `request` until its armed point, then SIGKILLs it there.
fn kill_at(request: &Request) {
    let point = request.arm.as_deref().unwrap();
    let mut child = Child::spawn(request);
    assert_eq!(child.wait_for("FAULT"), format!("FAULT {point}"));
    child.kill();
}

#[test]
fn kill_during_put_leaves_no_torn_object() {
    let scratch = Scratch::here("put");
    let root = scratch.root();
    let object = bytes("put", 300_000);
    let expect = ObjectRef::of(&object);

    kill_at(&request(
        &root,
        Some("put.written"),
        Job::Put(object.clone()),
    ));
    let store = Store::open(&root).unwrap();
    check(&store);
    assert!(!store.reuse(&expect).unwrap(), "a temp is never an object");
    assert_eq!(
        tmp_files(&store).len(),
        1,
        "the dead put's temp stays until a GC"
    );

    kill_at(&request(
        &root,
        Some("put.renamed"),
        Job::Put(object.clone()),
    ));
    check(&store);
    assert!(store.verify(&expect).unwrap());
    assert_eq!(store.put_bytes(&object).unwrap(), expect);

    let collected = store.clone().with_grace(Duration::ZERO).gc().unwrap();
    assert_eq!(collected.objects, 1, "nothing references the object");
    assert!(tmp_files(&store).is_empty(), "the dead put's temp is swept");
}

#[test]
fn kill_during_a_record_commit_keeps_the_old_record_or_the_new() {
    let scratch = Scratch::here("records");
    let root = scratch.root();
    let store = Store::open(&root).unwrap();
    let first = checkpoint(&store, &[bytes("one", 5000)], &[]);
    let second = checkpoint(
        &store,
        &[bytes("two", 5000)],
        &[("config.json", b"{}".to_vec())],
    );
    store
        .update_ref("local/model", |_| Some(repo("local/model", first.clone())))
        .unwrap();
    let nowhere = ObjectRef::of(b"nowhere");
    assert_eq!(
        store
            .update_ref("local/q", |_| Some(repo("local/q", nowhere)))
            .unwrap_err()
            .code(),
        "absent",
        "a root must be here"
    );

    let lane = |store: &Store| {
        store.get_ref("LOCAL/Model").unwrap().unwrap().tags[0]
            .commit
            .clone()
    };
    let job = Job::Ref("local/model".into(), second.clone());
    kill_at(&request(&root, Some("record.written"), job));
    check(&store);
    assert_eq!(lane(&store), first);
    let job = Job::Ref("local/model".into(), second.clone());
    kill_at(&request(&root, Some("record.renamed"), job));
    check(&store);
    assert_eq!(lane(&store), second);

    let names = |dir: &str| -> Vec<String> {
        let entries = fs::read_dir(root.join(dir)).unwrap();
        entries
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect()
    };
    assert_eq!(names("refs/local"), ["model.json"]);
    assert_eq!(tmp_files(&store).len(), 1, "the dead commit's temp");
    store.clone().with_grace(Duration::ZERO).gc().unwrap();
    assert!(tmp_files(&store).is_empty());
    check(&store);
}

/// A ref'd checkpoint, a rooted blob, and five garbage objects; all of it two hours old.
fn populated(root: &Path) -> (Store, Vec<ObjectRef>) {
    let store = Store::open(root).unwrap();
    let chunks: Vec<Vec<u8>> = (0..3)
        .map(|i| bytes(&format!("chunk-{i}"), 70_000))
        .collect();
    let manifest = checkpoint(
        &store,
        &chunks,
        &[("unet/config.json", b"{\"a\":1}".to_vec())],
    );
    store
        .update_ref("local/sdxl", |_| Some(repo("local/sdxl", manifest)))
        .unwrap();
    let media = store.put_bytes(&bytes("media", 50_000)).unwrap();
    name_locally(&store, "local/run-1", &[media]);
    let garbage: Vec<ObjectRef> = (0..5)
        .map(|i| {
            store
                .put_bytes(&bytes(&format!("garbage-{i}"), 10_000))
                .unwrap()
        })
        .collect();
    age(&store, blobs(&store), 2 * HOUR);
    (store, garbage)
}

#[test]
fn kill_during_gc_never_loses_a_live_object() {
    for point in ["gc.marked", "gc.listed", "gc.trashed", "gc.swept"] {
        let scratch = Scratch::here("gc-kill");
        let root = scratch.root();
        let (store, garbage) = populated(&root);
        let kept = live(&store);
        assert_eq!(
            kept.len(),
            8,
            "3 chunks, header, config, tree; the rooted blob and its tree"
        );

        kill_at(&request(&root, Some(point), Job::Gc));
        check(&store);
        assert_eq!(trash(&store), usize::from(point == "gc.trashed"), "{point}");

        let collected = store.gc().unwrap();
        assert_eq!(trash(&store), 0, "{point}");
        assert_eq!(
            blobs(&store),
            kept,
            "{point}: exactly the live objects remain"
        );
        let swept_before = match point {
            "gc.swept" => 1,
            _ => 0,
        };
        assert_eq!(
            collected.objects as usize + swept_before,
            garbage.len(),
            "{point}"
        );
    }
}

#[test]
fn an_object_reused_while_gc_sweeps_it_survives() {
    // Reused after the sweep listed it: the touch makes it young, and GC puts it back.
    let scratch = Scratch::here("gc-race");
    let root = scratch.root();
    let store = Store::open(&root).unwrap();
    let reused = bytes("reused", 9000);
    let other = store.put_bytes(&bytes("other", 9000)).unwrap();
    let object = store.put_bytes(&reused).unwrap();
    age(&store, [object.clone(), other.clone()], 2 * HOUR);
    let mut gc = Child::spawn(&request(&root, Some("gc.listed"), Job::Gc));
    gc.wait_for("FAULT gc.listed");
    assert_eq!(store.put_bytes(&reused).unwrap(), object);
    gc.resume();
    gc.finish();
    assert!(store.verify(&object).unwrap());
    assert!(
        !store.reuse(&other).unwrap(),
        "the untouched one is collected"
    );

    // Reused or put while it sits in GC's trash: taken back, young, and the sweep leaves it.
    for put in [false, true] {
        age(&store, [object.clone()], 2 * HOUR);
        let mut gc = Child::spawn(&request(&root, Some("gc.trashed"), Job::Gc));
        gc.wait_for("FAULT gc.trashed");
        assert_eq!(trash(&store), 1);
        // The collector holds an exclusive inode fence until it finishes its trash decision.
        // A real reader waits for that fence. Keep the controller free to restore the
        // namespace and resume GC; a synchronous read here would deadlock the fixture.
        let (sent, received) = std::sync::mpsc::channel();
        let (started, waiting) = std::sync::mpsc::sync_channel(0);
        let reading = store.clone();
        let read_object = object.clone();
        let reader = std::thread::spawn(move || {
            if started.send(()).is_ok() {
                let _ = sent.send(reading.read(&read_object));
            }
        });
        waiting
            .recv_timeout(Duration::from_secs(10))
            .expect("the reader thread never started");
        let probe = File::open(root.join("tmp/gc").join(&object.sha256)).unwrap();
        assert!(
            matches!(
                probe.try_lock_shared(),
                Err(std::fs::TryLockError::WouldBlock)
            ),
            "GC did not hold the actual trash inode's exclusive fence"
        );
        drop(probe);
        assert!(
            matches!(
                received.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ),
            "the reader completed while GC held the exclusive inode fence"
        );
        if put {
            assert_eq!(store.put_bytes(&reused).unwrap(), object);
        } else {
            assert!(store.reuse(&object).unwrap());
        }
        assert_eq!(trash(&store), 0);
        gc.resume();
        let read = received
            .recv_timeout(Duration::from_secs(10))
            .expect("the reader stayed blocked after GC resumed");
        reader.join().expect("the reader thread panicked");
        assert_eq!(
            read.unwrap(),
            reused,
            "the concurrent reader gets the exact bytes"
        );
        gc.finish();
        assert!(store.verify(&object).unwrap());
    }
}

#[test]
fn a_writer_racing_a_collector_loses_nothing_young_or_rooted() {
    let scratch = Scratch::here("gc-writer");
    let root = scratch.root();
    let store = Store::open(&root).unwrap();
    let old: Vec<Vec<u8>> = (0..40).map(|i| bytes(&format!("old-{i}"), 4000)).collect();
    let old_refs: Vec<ObjectRef> = old.iter().map(|o| store.put_bytes(o).unwrap()).collect();
    age(&store, old_refs.clone(), 2 * HOUR);
    let (reused, abandoned) = old_refs.split_at(20);

    // The collector sweeps once per round the writer finishes, while the next round runs.
    let mut collector = Child::spawn(&request(&root, None, Job::Collector));
    let rounds = 40;
    let writer = Job::Writer {
        rounds,
        old: old[..20].to_vec(),
    };
    let mut writer = Child::spawn(&request(&root, None, writer));
    for _ in 0..rounds {
        writer.wait_for("ROUND");
        collector.resume();
    }
    writer.finish();
    collector.input.take();
    assert_eq!(collector.wait_for("RUNS"), format!("RUNS {rounds}"));
    collector.finish();

    store.gc().unwrap();
    check(&store);
    for i in 0..rounds {
        let named = store.get_ref(&format!("local/w-{i}")).unwrap().unwrap();
        let files = store.tree(&named.tags[0].commit).unwrap().files;
        assert_eq!(files["1.bin"], reused[i % 20]);
        for object in files.values() {
            assert!(store.verify(object).unwrap(), "rooted object lost");
        }
        let loose = ObjectRef::of(&bytes(&format!("loose-{i}"), 2000));
        assert!(store.verify(&loose).unwrap(), "a young object was swept");
    }
    for object in abandoned {
        assert!(!store.reuse(object).unwrap(), "old garbage outlived GC");
    }
}

/// Uses go least recently used first, until enough would be free, and only those unused for
/// long enough; what a root keeps never goes.
#[test]
fn collect_drops_the_least_recently_used_first_and_never_a_root() {
    let scratch = Scratch::here("lru");
    let store = Store::open(scratch.root()).unwrap();
    let chunk = |name: &str, n: usize| bytes(name, n);
    let config = ("unet/config.json", b"{\"layers\":4}".to_vec());
    let m1 = checkpoint(
        &store,
        &[chunk("A", 100), chunk("B", 200), chunk("C", 400)],
        std::slice::from_ref(&config),
    );
    let m2 = checkpoint(
        &store,
        &[chunk("B", 200), chunk("C", 400), chunk("D", 800)],
        std::slice::from_ref(&config),
    );
    let m3 = checkpoint(&store, &[chunk("C", 400)], &[]);
    let e = checkpoint(&store, &[chunk("E", 1600)], &[]);
    let g = store.put_bytes(&chunk("G", 3200)).unwrap();
    store
        .update_ref("local/m3", |_| Some(repo("local/m3", m3.clone())))
        .unwrap();
    // One use of each but m3, the oldest first.
    for (tree, hours) in [(&m1, 3), (&m2, 2), (&e, 1)] {
        drop(store.variant(tree, None).unwrap());
        let used = File::open(store.root().join("uses").join(&tree.sha256)).unwrap();
        used.set_modified(SystemTime::now() - hours * HOUR).unwrap();
    }
    age(&store, blobs(&store), 2 * HOUR);

    let size = |objects: &BTreeSet<ObjectRef>| objects.iter().map(|o| o.length).sum::<u64>();
    let (c1, c2, c3, ce) = (
        closure(&store, &m1),
        closure(&store, &m2),
        closure(&store, &m3),
        closure(&store, &e),
    );
    let only = |mine: &BTreeSet<ObjectRef>, others: &[&BTreeSet<ObjectRef>]| -> BTreeSet<ObjectRef> {
        let shared = |o: &&ObjectRef| others.iter().any(|s| s.contains(*o));
        mine.iter().filter(|o| !shared(o)).cloned().collect()
    };
    let m1_only = only(&c1, &[&c2, &c3, &ce]);
    assert_eq!(m1_only.len(), 3, "its tree, its header, and A");

    let collected = store.collect(g.length + 1, Duration::ZERO).unwrap();
    assert_eq!(collected.uses, std::slice::from_ref(&m1.sha256));
    assert_eq!(collected.objects, 4, "G and m1's own three");
    assert_eq!(collected.bytes, g.length + size(&m1_only));
    check(&store);

    let stale = store.collect(u64::MAX, HOUR * 3 / 2).unwrap();
    assert_eq!(stale.uses, std::slice::from_ref(&m2.sha256), "a use within the hour and a half stays");
    let m2_only = only(&c2, &[&c3, &ce]);
    assert_eq!((stale.objects, stale.bytes), (m2_only.len() as u64, size(&m2_only)));
    for object in c3.union(&ce) {
        assert!(store.verify(object).unwrap(), "{} is kept", object.sha256);
    }
    check(&store);
}

#[test]
fn admission_refuses_other_bytes_and_verify_removes_a_corrupt_copy() {
    let scratch = Scratch::here("admit");
    let store = Store::open(scratch.root()).unwrap();
    let good = bytes("good", 5000);
    let expect = ObjectRef::of(&good);
    let other = bytes("bad!", 5000);
    let refused = store.put(&other[..], Some(&expect)).unwrap_err();
    assert_eq!(refused.code(), "mismatch");
    let longer = [good.clone(), b"x".to_vec()].concat();
    assert_eq!(
        store.put(&longer[..], Some(&expect)).unwrap_err().code(),
        "mismatch"
    );
    assert!(
        tmp_files(&store).is_empty(),
        "a refused admission leaves nothing"
    );
    assert!(!store.reuse(&expect).unwrap());

    assert_eq!(store.put(&good[..], Some(&expect)).unwrap(), expect);
    assert_eq!(store.read(&expect).unwrap(), good);
    let path = store.path(&expect).unwrap();
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    permissions.set_readonly(false);
    fs::set_permissions(&path, permissions).unwrap();
    fs::write(&path, &other).unwrap();
    assert!(!store.verify(&expect).unwrap());
    assert_eq!(store.file(&expect).unwrap_err().code(), "absent");
}

#[test]
fn open_creates_only_in_an_empty_place_and_refuses_another_format() {
    let scratch = Scratch::here("open");
    let v1 = scratch.0.join("v1");
    fs::create_dir_all(&v1).unwrap();
    fs::write(v1.join("format"), "store-format 1\nclaim-protocol 1\n").unwrap();
    assert_eq!(Store::open(&v1).unwrap_err().code(), "invalid");
    let busy = scratch.0.join("busy");
    fs::create_dir_all(&busy).unwrap();
    fs::write(busy.join("notes.txt"), "mine").unwrap();
    assert_eq!(Store::open(&busy).unwrap_err().code(), "invalid");
    assert!(!busy.join("blobs").exists());
    let old = scratch.0.join("old");
    fs::create_dir_all(&old).unwrap();
    fs::write(old.join("format"), r#"{"format":"tensorfs.store/2"}"#).unwrap();
    let refused = Store::open(&old).unwrap_err();
    assert!(refused.to_string().contains("pull again"), "{refused}");
    let fresh = Store::open(scratch.root()).unwrap();
    assert_eq!(
        fs::read_to_string(fresh.root().join("format")).unwrap(),
        r#"{"format":"tensorfs.store/3"}"#
    );
    Store::open(scratch.root()).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn durable_flushes_each_object_and_ephemeral_syncs_once() {
    let calls = |log: &Path, call: &str| {
        let text = fs::read_to_string(log).unwrap();
        text.lines()
            .filter(|l| l.contains(&format!("{call}(")))
            .count()
    };
    let objects: Vec<Vec<u8>> = (0..3).map(|i| bytes(&format!("class-{i}"), 8000)).collect();
    assert!(
        Command::new("strace")
            .arg("-V")
            .output()
            .is_ok_and(|o| o.status.success()),
        "this test counts syscalls with strace; install it"
    );

    let disk = Scratch::here("durable");
    let store = Store::open(disk.root()).unwrap();
    assert_eq!(store.durability(), Durability::Durable);
    let log = disk.0.join("strace.log");
    let mut job = request(&disk.root(), None, Job::Classes(objects.clone()));
    job.durability = Some("durable".into());
    Child::traced(&job, &log).finish();
    assert_eq!(
        calls(&log, "fdatasync"),
        6,
        "3 objects, their tree and 2 refs"
    );
    assert!(calls(&log, "fsync") >= 5, "each one's directory");
    assert_eq!(calls(&log, "syncfs"), 0);

    let memory = Scratch::new(Path::new("/dev/shm"), "ephemeral");
    let store = Store::open(memory.root()).unwrap();
    assert_eq!(
        store.durability(),
        Durability::Ephemeral,
        "tmpfs is ephemeral"
    );
    let log = disk.0.join("strace-ephemeral.log");
    Child::traced(&request(&memory.root(), None, Job::Classes(objects)), &log).finish();
    assert_eq!(calls(&log, "fdatasync") + calls(&log, "fsync"), 0);
    assert_eq!(
        calls(&log, "syncfs"),
        1,
        "one syncfs, before the first ref names the objects"
    );
    check(&store);
}

/// TensorD forks executors while it uses the store. A child forked by another thread holds the
/// store's lock descriptors until it execs; that may delay a record commit, never refuse one, and
/// GC takes no lock at all.
#[test]
fn children_forked_meanwhile_never_make_the_store_busy() {
    let scratch = Scratch::here("fork");
    let root = scratch.root();
    let store = Store::open(&root).unwrap();
    let anchor = store.put_bytes(b"anchor").unwrap();
    store
        .update_ref("local/fork", |_| Some(repo("local/fork", anchor)))
        .unwrap();
    let stop = std::sync::atomic::AtomicBool::new(false);
    let rounds = 150;
    std::thread::scope(|scope| {
        let forks = scope.spawn(|| {
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                assert!(Command::new("true").status().unwrap().success());
                n += 1;
            }
            n
        });
        let sweeps = scope.spawn(|| {
            let mut n = 0u64;
            while !stop.load(Ordering::Relaxed) {
                store.gc().unwrap();
                n += 1;
            }
            n
        });
        let writer = Job::Writer {
            rounds: 40,
            old: vec![bytes("fork-old", 1000)],
        };
        let other = Child::spawn(&request(&root, None, writer));
        for i in 0..rounds {
            let object = store.put_bytes(&bytes(&format!("fork-{i}"), 1500)).unwrap();
            name_locally(&store, &format!("local/f-{}", i % 8), std::slice::from_ref(&object));
            store
                .update_ref("local/fork", |_| Some(repo("local/fork", object)))
                .unwrap();
        }
        other.finish();
        stop.store(true, Ordering::Relaxed);
        let (forks, sweeps) = (forks.join().unwrap(), sweeps.join().unwrap());
        assert!(forks > 0 && sweeps > 0, "{forks} forks, {sweeps} sweeps");
    });
    check(&store);
}
