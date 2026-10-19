//! The v2 store through real processes and files: SIGKILL at each kill point of a put, a record
//! commit and a GC, then reopen and check; a writer racing a collector; exact usage over shared
//! chunks; and the two durability classes, counted by strace.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tensorfs_core::format::{
    variant_of, Body, Component, Dtype, Header, Location, ObjectRef, Release, Repo, Tensor, Tree,
};
use tensorfs_core::store2::{fault, Durability, Pin, Store};

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
        releases: vec![Release {
            version: "1.0".into(),
            tree,
            yanked: false,
            revision: 0,
        }],
        ..Default::default()
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

/// Everything a pin or a ref reaches.
fn live(store: &Store) -> BTreeSet<ObjectRef> {
    let mut roots: Vec<ObjectRef> = store
        .pins()
        .unwrap()
        .into_iter()
        .flat_map(|p| p.1.roots)
        .collect();
    for repo in store.refs().unwrap() {
        roots.extend(repo.releases.into_iter().map(|r| r.tree));
    }
    roots.iter().flat_map(|root| closure(store, root)).collect()
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
    Pin(String, Vec<ObjectRef>),
    Ref(String, ObjectRef),
    Gc,
    /// Each round: a fresh object and a reused old one, pinned together, and a loose object.
    Writer {
        rounds: usize,
        old: Vec<Vec<u8>>,
    },
    /// One GC per line on stdin, until it closes.
    Collector,
    /// Puts each object, pins them twice, touches the first pin.
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
        Job::Pin(id, roots) => store.pin(&id, &Pin::new(roots)).unwrap(),
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
                store
                    .pin(&format!("w-{i}"), &Pin::new(vec![fresh, again]))
                    .unwrap();
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
            store.pin("a", &Pin::new(roots.clone())).unwrap();
            store.pin("b", &Pin::new(roots)).unwrap();
            assert!(store.touch_pin("a").unwrap());
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
    let a = store.put_bytes(b"a").unwrap();
    let b = store.put_bytes(b"b").unwrap();
    store.pin("p", &Pin::new(vec![a.clone()])).unwrap();
    let first = checkpoint(&store, &[bytes("one", 5000)], &[]);
    let second = checkpoint(
        &store,
        &[bytes("two", 5000)],
        &[("config.json", b"{}".to_vec())],
    );
    store
        .update_ref("acme/model", |_| Some(repo("acme/model", first.clone())))
        .unwrap();
    let nowhere = Pin::new(vec![ObjectRef::of(b"nowhere")]);
    assert_eq!(
        store.pin("q", &nowhere).unwrap_err().code(),
        "absent",
        "a root must be here"
    );

    let both = Job::Pin("p".into(), vec![a.clone(), b.clone()]);
    kill_at(&request(&root, Some("record.written"), both));
    check(&store);
    assert_eq!(store.get_pin("p").unwrap().unwrap().roots, vec![a.clone()]);
    let both = Job::Pin("p".into(), vec![a.clone(), b.clone()]);
    kill_at(&request(&root, Some("record.renamed"), both));
    check(&store);
    assert_eq!(store.get_pin("p").unwrap().unwrap().roots, vec![a, b]);

    let lane = |store: &Store| {
        store.get_ref("ACME/Model").unwrap().unwrap().releases[0]
            .tree
            .clone()
    };
    let job = Job::Ref("acme/model".into(), second.clone());
    kill_at(&request(&root, Some("record.written"), job));
    check(&store);
    assert_eq!(lane(&store), first);
    let job = Job::Ref("acme/model".into(), second.clone());
    kill_at(&request(&root, Some("record.renamed"), job));
    check(&store);
    assert_eq!(lane(&store), second);

    let names = |dir: &str| -> Vec<String> {
        let entries = fs::read_dir(root.join(dir)).unwrap();
        entries
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect()
    };
    assert_eq!(names("pins"), ["p.json"]);
    assert_eq!(names("refs/acme"), ["model.json"]);
    assert_eq!(tmp_files(&store).len(), 2, "two dead commits' temps");
    store.clone().with_grace(Duration::ZERO).gc().unwrap();
    assert!(tmp_files(&store).is_empty());
    check(&store);
}

/// A ref'd checkpoint, a pinned blob, and five garbage objects; all of it two hours old.
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
        .update_ref("acme/sdxl", |_| Some(repo("acme/sdxl", manifest)))
        .unwrap();
    let media = store.put_bytes(&bytes("media", 50_000)).unwrap();
    store.pin("run-1", &Pin::new(vec![media])).unwrap();
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
            7,
            "3 chunks, header, config, manifest; the pinned blob"
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
        assert_eq!(store.usage().unwrap().unreferenced, 0);
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
        // A real reader restores the namespace first, then waits for that fence. Keep the
        // controller free to resume GC; a synchronous read here would deadlock the fixture.
        let (sent, received) = std::sync::mpsc::channel();
        let reading = store.clone();
        let read_object = object.clone();
        let reader = std::thread::spawn(move || {
            let _ = sent.send(reading.read(&read_object));
        });
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
fn a_writer_racing_a_collector_loses_nothing_young_or_pinned() {
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
        let pin = store.get_pin(&format!("w-{i}")).unwrap().unwrap();
        assert_eq!(pin.roots[1], reused[i % 20]);
        for root in &pin.roots {
            assert!(store.verify(root).unwrap(), "pinned object lost");
        }
        let loose = ObjectRef::of(&bytes(&format!("loose-{i}"), 2000));
        assert!(store.verify(&loose).unwrap(), "a young object was swept");
    }
    for object in abandoned {
        assert!(!store.reuse(object).unwrap(), "old garbage outlived GC");
    }
}

#[test]
fn usage_counts_shared_objects_once_and_collect_drops_the_least_recent_pin() {
    let scratch = Scratch::here("usage");
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
    let e = store.put_bytes(&chunk("E", 1600)).unwrap();
    let g = store.put_bytes(&chunk("G", 3200)).unwrap();
    store
        .update_ref("acme/m3", |_| Some(repo("acme/m3", m3.clone())))
        .unwrap();
    let pin = |id: &str, roots: Vec<ObjectRef>, last_use: u64| {
        let pin = Pin {
            last_use,
            owner: Some("tensord".into()),
            roots,
        };
        store.pin(id, &pin).unwrap();
    };
    pin("p1", vec![m1.clone(), m1.clone()], 1_000);
    pin("p2", vec![m2.clone()], 2_000);
    pin("p3", vec![e.clone()], 3_000);

    let size = |objects: &BTreeSet<ObjectRef>| objects.iter().map(|o| o.length).sum::<u64>();
    let (c1, c2, c3) = (
        closure(&store, &m1),
        closure(&store, &m2),
        closure(&store, &m3),
    );
    let only =
        |mine: &BTreeSet<ObjectRef>, others: &[&BTreeSet<ObjectRef>]| -> BTreeSet<ObjectRef> {
            mine.iter()
                .filter(|o| others.iter().all(|s| !s.contains(*o)))
                .cloned()
                .collect()
        };
    let p1_only = only(&c1, &[&c2, &c3]);
    assert_eq!(p1_only.len(), 3, "M1, its header, and A");
    let all = blobs(&store);
    assert_eq!(
        all.len(),
        13,
        "6 blobs, 3 headers, 1 shared config, 3 manifests"
    );

    let usage = store.usage().unwrap();
    assert_eq!(usage.objects, 13);
    assert_eq!(usage.bytes, size(&all));
    assert_eq!(usage.refs, size(&c3));
    assert_eq!(usage.unreferenced, g.length);
    assert_eq!(usage.pins["p1"].bytes, size(&c1));
    assert_eq!(usage.pins["p1"].marginal, size(&p1_only));
    assert_eq!(usage.pins["p2"].bytes, size(&c2));
    assert_eq!(usage.pins["p2"].marginal, size(&only(&c2, &[&c1, &c3])));
    assert_eq!(usage.pins["p3"].marginal, 1600);
    assert_eq!(usage.pins["p3"].owner.as_deref(), Some("tensord"));
    let sum: u64 =
        usage.pins.values().map(|p| p.marginal).sum::<u64>() + usage.refs + usage.unreferenced;
    assert!(sum < usage.bytes, "shared objects belong to no single pin");

    age(&store, all, 2 * HOUR);
    let collected = store.collect(g.length + 1, Duration::ZERO).unwrap();
    assert_eq!(collected.pins, ["p1"]);
    assert_eq!(collected.objects, 4, "G and p1's own three");
    assert_eq!(collected.bytes, g.length + size(&p1_only));
    let left = store.usage().unwrap();
    assert_eq!(left.pins.keys().collect::<Vec<_>>(), ["p2", "p3"]);
    assert_eq!(left.bytes, usage.bytes - collected.bytes);
    check(&store);

    assert!(store.touch_pin("p2").unwrap());
    let stale = store.collect(u64::MAX, HOUR).unwrap();
    assert_eq!(stale.pins, ["p3"], "a pin used within the hour stays");
    assert_eq!(stale.bytes, 1600);
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
        5,
        "3 objects and 2 pins; a touch is not flushed"
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
        "one syncfs, before the first pin names the objects"
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
        .update_ref("acme/fork", |_| Some(repo("acme/fork", anchor)))
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
            store
                .pin(&format!("f-{}", i % 8), &Pin::new(vec![object.clone()]))
                .unwrap();
            assert!(store.touch_pin(&format!("f-{}", i % 8)).unwrap());
            store
                .update_ref("acme/fork", |_| Some(repo("acme/fork", object)))
                .unwrap();
        }
        other.finish();
        stop.store(true, Ordering::Relaxed);
        let (forks, sweeps) = (forks.join().unwrap(), sweeps.join().unwrap());
        assert!(forks > 0 && sweeps > 0, "{forks} forks, {sweeps} sweeps");
    });
    check(&store);
}
