//! The Store's fetcher against a REAL Store and a REAL loopback hub: tickets, priorities,
//! structure, holds, disk admission and the executor's channel, over sockets this test starts.

mod common;

use platform::ipc::{self, LocalStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{components, hub, serve_all, store, temporary, Answer, Checkpoint, Rule};
use tensorfs_core::err::Code;
use tensorfs_core::fetch_v1::{self as fetch, Access, Fetcher, Origin, Plain, Priority};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::Store;
use tensorfs_core::transport::{Anonymous, PullCancellation, SourcePolicy};

const SAMPLE: Duration = Duration::from_millis(50);

fn origin(hub: &common::Hub, checkpoint: &Checkpoint) -> Origin {
    Origin {
        hub: hub.base.clone(),
        refspec: checkpoint.refspec(),
        lane: checkpoint.lane.clone(),
        manifest: checkpoint.manifest.clone(),
    }
}

fn access() -> Arc<dyn Access> {
    Arc::new(Plain {
        credential: Anonymous,
        policy: SourcePolicy::default(),
    })
}

fn never() -> PullCancellation {
    PullCancellation::default()
}

/// Until `ready`, at most a few seconds.
fn until(ready: impl Fn() -> bool) {
    let began = Instant::now();
    while !ready() {
        assert!(began.elapsed() < Duration::from_secs(20), "never ready");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Two tickets for the same objects, waited on at once from two threads: each object crosses
/// the wire once, under one closure, and both waits answer.
#[test]
fn two_tickets_fetch_each_object_once() {
    let dir = temporary("fetch-dedupe");
    let store = Store::open(&store(&dir)).unwrap();
    let checkpoint = Arc::new(components("acme/model", &[4], 64 << 10, 1, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::from_millis(20),
        serve_all(),
    );
    let fetcher = Fetcher::new(&store, 64, SAMPLE).unwrap();
    let origin = origin(&hub, &checkpoint);
    let first = fetcher.want(
        &origin,
        access(),
        &checkpoint.blobs,
        Priority::Next { run: 1, rank: 0 },
    );
    let second = fetcher.want(&origin, access(), &checkpoint.blobs, Priority::Behind);
    std::thread::scope(|scope| {
        let a = scope.spawn(|| first.wait(&[], &never()));
        let b = scope.spawn(|| second.wait(&checkpoint.blobs, &never()));
        a.join().unwrap().unwrap();
        b.join().unwrap().unwrap();
    });
    assert!(checkpoint
        .bodies
        .keys()
        .all(|hex| store.contains(hex) || store.manifest_path(hex).is_file()));
    assert!(
        checkpoint.bodies.keys().all(|hex| hub.gets(hex) == 1),
        "{:?}",
        hub.gets.lock().unwrap()
    );
    assert_eq!(hub.gets("closure"), 1);
    assert_eq!(first.progress(), (checkpoint.total(), checkpoint.total()));
    let _ = std::fs::remove_dir_all(&dir);
}

/// With one request in flight, objects go Now first, then Next by run and rank, then Behind.
#[test]
fn now_goes_before_next_by_run_and_rank_and_behind_goes_last() {
    let dir = temporary("fetch-order");
    let store = Store::open(&store(&dir)).unwrap();
    let checkpoint = Arc::new(components("acme/model", &[5], 16 << 10, 2, &[]));
    // Every GET waits for the gate, so all the asks are queued before the first answer.
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::ZERO,
        Arc::new(|_, _| Answer::Hold),
    );
    let fetcher = Fetcher::new(&store, 1, SAMPLE).unwrap();
    let origin = origin(&hub, &checkpoint);
    let blob = |at: usize| vec![checkpoint.blobs[at].clone()];
    let structure = fetcher.want(&origin, access(), &[], Priority::Now);
    until(|| hub.total_gets() == 1);
    let tickets = [
        fetcher.want(&origin, access(), &blob(4), Priority::Behind),
        fetcher.want(
            &origin,
            access(),
            &blob(3),
            Priority::Next { run: 2, rank: 0 },
        ),
        fetcher.want(
            &origin,
            access(),
            &blob(2),
            Priority::Next { run: 1, rank: 5 },
        ),
        fetcher.want(
            &origin,
            access(),
            &blob(1),
            Priority::Next { run: 1, rank: 1 },
        ),
        fetcher.want(&origin, access(), &blob(0), Priority::Now),
    ];
    hub.open();
    // Watched, not waited on: a wait would raise its objects to Now.
    until(|| {
        tickets
            .iter()
            .all(|ticket| ticket.progress().0 == ticket.progress().1)
    });
    structure.wait(&[], &never()).unwrap();
    let blobs: Vec<String> = checkpoint.blobs.iter().map(|o| o.sha256.clone()).collect();
    let order: Vec<String> = hub
        .log()
        .into_iter()
        .filter(|hex| blobs.contains(hex))
        .collect();
    let name = |hex: &String| match blobs.iter().position(|b| b == hex) {
        Some(at) => format!("b{at}"),
        None if *hex == checkpoint.manifest.sha256 => "manifest".into(),
        None => "header".into(),
    };
    let log: Vec<String> = hub.log().iter().map(name).collect();
    assert_eq!(
        order, blobs,
        "blobs in the order the hub was asked: {log:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A dropped ticket's objects are not fetched; one on the wire stops and never lands.
#[test]
fn dropping_a_ticket_cancels_its_fetches() {
    let dir = temporary("fetch-drop");
    let store = Store::open(&store(&dir)).unwrap();
    let checkpoint = Arc::new(components("acme/model", &[3], 64 << 10, 3, &[]));
    let held = checkpoint.blobs[0].sha256.clone();
    let rule: Rule = Arc::new(move |hex, _| {
        if hex == held {
            Answer::Hold
        } else {
            Answer::Body
        }
    });
    let hub = hub(vec![Arc::clone(&checkpoint)], None, Duration::ZERO, rule);
    // One request at a time, so what is queued behind the held one stays queued.
    let fetcher = Fetcher::new(&store, 1, SAMPLE).unwrap();
    let origin = origin(&hub, &checkpoint);
    let structure = fetcher.want(&origin, access(), &[], Priority::Now);
    structure.wait(&[], &never()).unwrap();

    let wire = fetcher.want(
        &origin,
        access(),
        &checkpoint.blobs[..1],
        Priority::Next { run: 1, rank: 0 },
    );
    until(|| hub.gets(&checkpoint.blobs[0].sha256) == 1);
    let queued = fetcher.want(
        &origin,
        access(),
        &checkpoint.blobs[1..],
        Priority::Next { run: 1, rank: 1 },
    );
    std::thread::sleep(Duration::from_millis(50));
    // Dropped while queued: never asked for. Dropped on the wire: its request stops.
    drop(queued);
    drop(wire);
    std::thread::sleep(Duration::from_millis(100));
    hub.open();
    std::thread::sleep(Duration::from_millis(300));
    assert!(checkpoint.blobs.iter().all(|o| !store.contains(&o.sha256)));
    assert_eq!(hub.gets(&checkpoint.blobs[0].sha256), 1);
    assert!(checkpoint.blobs[1..]
        .iter()
        .all(|o| hub.gets(&o.sha256) == 0));

    // Wanted again, it is fetched again.
    let again = fetcher.want(&origin, access(), &checkpoint.blobs[..1], Priority::Now);
    again.wait(&[], &never()).unwrap();
    assert!(store.contains(&checkpoint.blobs[0].sha256));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A ticket lands its origin's structure and attests the hub for the rest; a wait fetches an
/// object no ticket asked for, and only that one; one outside the checkpoint is refused.
#[test]
fn a_wait_fetches_an_absent_object_and_the_hub_keeps_the_rest() {
    let dir = temporary("fetch-wait");
    let root = store(&dir);
    let store = Store::open(&root).unwrap();
    let checkpoint = Arc::new(components("acme/model", &[2, 2], 16 << 10, 4, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let fetcher = Fetcher::new(&store, 8, SAMPLE).unwrap();
    let origin = origin(&hub, &checkpoint);
    let ticket = fetcher.want(&origin, access(), &[], Priority::Now);
    ticket.wait(&[], &never()).unwrap();
    assert!(
        store.manifest_path(&checkpoint.manifest.sha256).is_file()
            && store.contains(&checkpoint.header.sha256)
    );
    assert_eq!(hub.total_gets(), 2, "manifest and header only");
    assert_eq!(
        tensorfs_core::origin::of(&root, &checkpoint.manifest).unwrap(),
        Some(origin.clone())
    );
    let custodied = tensorfs_core::ingest::custody::live(&root).unwrap();
    assert!(checkpoint
        .blobs
        .iter()
        .all(|o| custodied.contains(&o.sha256)));

    let read = checkpoint.components[1][1].clone();
    ticket.wait(std::slice::from_ref(&read), &never()).unwrap();
    assert!(store.contains(&read.sha256));
    assert_eq!(hub.gets(&read.sha256), 1);
    assert!(checkpoint
        .blobs
        .iter()
        .filter(|o| **o != read)
        .all(|o| hub.gets(&o.sha256) == 0));

    let foreign = vec![ObjectRef::of(b"not of this checkpoint")];
    assert_eq!(
        ticket.wait(&foreign, &never()).unwrap_err().code,
        Code::NOT_CONTAINED
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// An object the disk cannot admit is refused before a byte of it moves.
#[test]
fn an_object_the_disk_cannot_hold_is_refused() {
    let dir = temporary("fetch-full");
    let root = store(&dir);
    let checkpoint = Arc::new(components("acme/model", &[1], 256 << 10, 5, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let structure = checkpoint.manifest.length + checkpoint.header.length;
    let store = Store::open(&root)
        .unwrap()
        .bind_disk_budget(Some(structure + 1024))
        .unwrap();
    let fetcher = Fetcher::new(&store, 8, SAMPLE).unwrap();
    let ticket = fetcher.want(
        &origin(&hub, &checkpoint),
        access(),
        &checkpoint.blobs,
        Priority::Now,
    );
    let refused = ticket.wait(&[], &never()).unwrap_err();
    assert_eq!(refused.code, Code::CAPACITY_EXHAUSTED, "{refused}");
    assert!(
        refused.detail.contains(&checkpoint.blobs[0].sha256),
        "{refused}"
    );
    assert_eq!(hub.gets(&checkpoint.blobs[0].sha256), 0);
    assert!(!store.contains(&checkpoint.blobs[0].sha256));
    let _ = std::fs::remove_dir_all(&dir);
}

/// A checkpoint held in part is a cache root. When a fetch finds the disk short, the present
/// objects of the one nothing uses are evicted first, its structure stays, and an object
/// another ticket holds is kept.
#[test]
fn a_short_disk_evicts_an_idle_partial_checkpoint_and_keeps_ticketed_objects() {
    let dir = temporary("fetch-evict");
    let root = store(&dir);
    let size = 64 << 10;
    let idle = Arc::new(components("acme/idle", &[3], size, 6, &[]));
    let kept = idle.blobs[0].clone();
    let shared = vec![(kept.clone(), idle.bodies[&kept.sha256].clone())];
    let busy = Arc::new(components("acme/busy", &[1], 2 * size, 7, &shared));
    let hub = hub(
        vec![Arc::clone(&idle), Arc::clone(&busy)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    // The machine owns its Store: its tickets pin, and its own collection runs beside them.
    let owned = Store::ensure_owned(&root).unwrap();
    {
        let fetcher = Fetcher::new(&owned, 8, SAMPLE).unwrap();
        let all = fetcher.want(&origin(&hub, &idle), access(), &idle.blobs, Priority::Now);
        all.wait(&[], &never()).unwrap();
    }
    // A collection leaves a checkpoint held in part alone.
    tensorfs_core::gc::collect(&root, false).unwrap();
    assert!(idle.blobs.iter().all(|o| owned.contains(&o.sha256)));

    let structure = busy.manifest.length + busy.header.length;
    let own = busy.components[0][0].clone();
    // Room for the structure, and for `own` once both of the idle checkpoint's other objects go.
    let budget = owned.occupancy().unwrap() + structure + own.length - 2 * size as u64 + 1024;
    let store = owned.bind_disk_budget(Some(budget)).unwrap();
    let fetcher = Fetcher::new(&store, 8, SAMPLE).unwrap();
    let ticket = fetcher.want(
        &origin(&hub, &busy),
        access(),
        &[kept.clone(), own.clone()],
        Priority::Now,
    );
    ticket.wait(&[], &never()).unwrap();
    assert!(store.contains(&own.sha256));
    assert!(
        store.contains(&kept.sha256),
        "a ticketed object was evicted"
    );
    assert!(
        idle.blobs[1..].iter().all(|o| !store.contains(&o.sha256)),
        "the idle checkpoint kept its objects"
    );
    assert!(
        store.manifest_path(&idle.manifest.sha256).is_file() && store.contains(&idle.header.sha256)
    );
    assert!(tensorfs_core::origin::of(&root, &idle.manifest)
        .unwrap()
        .is_some());
    assert_eq!(hub.gets(&kept.sha256), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A checkpoint held in part that nothing used for the cache's TTL goes, whole.
#[test]
fn an_idle_partial_checkpoint_goes_at_its_ttl() {
    let dir = temporary("fetch-ttl");
    let root = store(&dir);
    let store = Store::open(&root).unwrap();
    let checkpoint = Arc::new(components("acme/model", &[2], 16 << 10, 9, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    {
        let fetcher = Fetcher::new(&store, 8, SAMPLE).unwrap();
        let ticket = fetcher.want(
            &origin(&hub, &checkpoint),
            access(),
            &checkpoint.blobs[..1],
            Priority::Now,
        );
        ticket.wait(&[], &never()).unwrap();
    }
    let present =
        |o: &ObjectRef| store.contains(&o.sha256) || store.manifest_path(&o.sha256).is_file();
    tensorfs_core::gc::collect_idle(&root, &[], Duration::from_secs(3600)).unwrap();
    assert!(present(&checkpoint.blobs[0]) && present(&checkpoint.manifest));
    std::thread::sleep(Duration::from_millis(20));
    tensorfs_core::gc::collect_idle(&root, &[], Duration::from_millis(1)).unwrap();
    assert!(
        !present(&checkpoint.blobs[0])
            && !present(&checkpoint.manifest)
            && !present(&checkpoint.header)
    );
    assert_eq!(
        tensorfs_core::origin::of(&root, &checkpoint.manifest).unwrap(),
        None
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The executor's channel: a schedule is fetched at Next, `present` answers, a wait answers
/// once its object lands, and a checkpoint the grant does not select is refused.
#[test]
fn the_executor_channel_schedules_waits_and_answers_presence() {
    use tensorfs_core::jcs::{self, Json};
    let dir = temporary("fetch-channel");
    let store = Store::open(&store(&dir)).unwrap();
    let checkpoint = Arc::new(components("acme/model", &[3], 16 << 10, 8, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let fetcher = Fetcher::new(&store, 8, SAMPLE).unwrap();
    let grant = fetch::Grant {
        run: 7,
        origins: vec![(origin(&hub, &checkpoint), access())],
        stop: never(),
    };
    let (client, server) = ipc::pair().unwrap();
    let rows = |objects: &[ObjectRef]| {
        objects
            .iter()
            .map(|o| format!(r#"{{"sha256":"{}","length":{}}}"#, o.sha256, o.length))
            .collect::<Vec<_>>()
            .join(",")
    };
    let manifest = checkpoint.manifest.id();
    std::thread::scope(|scope| {
        let served = scope.spawn(|| fetch::serve(&fetcher, grant, server));
        let mut channel = Peer(client);
        let ask = |channel: &mut Peer, body: String| -> Json {
            channel.send(&body);
            channel.receive()
        };
        let present = ask(
            &mut channel,
            format!(
                r#"{{"id":1,"op":"present","manifest":"{manifest}","objects":[{}]}}"#,
                rows(&checkpoint.blobs)
            ),
        );
        assert_eq!(
            jcs::write(&present),
            br#"{"id":1,"result":[false,false,false]}"#.to_vec()
        );
        let scheduled = ask(
            &mut channel,
            format!(
                r#"{{"id":2,"op":"schedule","manifest":"{manifest}","objects":[{}],"rank":0,"behind":false}}"#,
                rows(&checkpoint.blobs[..2])
            ),
        );
        assert_eq!(
            jcs::write(&scheduled),
            br#"{"id":2,"result":null}"#.to_vec()
        );
        let waited = ask(
            &mut channel,
            format!(
                r#"{{"id":3,"op":"wait","manifest":"{manifest}","objects":[{}]}}"#,
                rows(&checkpoint.blobs)
            ),
        );
        assert_eq!(jcs::write(&waited), br#"{"id":3,"result":null}"#.to_vec());
        assert!(checkpoint
            .blobs
            .iter()
            .all(|o| store.contains(&o.sha256) && hub.gets(&o.sha256) == 1));
        let refused = ask(
            &mut channel,
            format!(
                r#"{{"id":4,"op":"wait","manifest":"sha256:{}","objects":[]}}"#,
                "ab".repeat(32)
            ),
        );
        assert!(String::from_utf8(jcs::write(&refused))
            .unwrap()
            .contains("SOURCE_NOT_ALLOWED"));
        drop(channel);
        served.join().unwrap().unwrap();
    });
    let _ = std::fs::remove_dir_all(&dir);
}

/// Six cold checkpoints of 200 objects each, for many tickets in one process.
fn many(hub_base: &str) -> Vec<(Arc<Checkpoint>, Origin)> {
    (0..6)
        .map(|at| {
            let checkpoint = Arc::new(components(
                &format!("acme/many{at}"),
                &[200],
                4096,
                40 + at,
                &[],
            ));
            let origin = Origin {
                hub: hub_base.to_string(),
                refspec: checkpoint.refspec(),
                lane: checkpoint.lane.clone(),
                manifest: checkpoint.manifest.clone(),
            };
            (checkpoint, origin)
        })
        .collect()
}

/// A pod's container caps its tasks (a vast RTX 5060 Ti: `pids.max` 1280) below the streams
/// one process's Fetcher starts for many tickets at once. A thread the cap refuses is
/// backpressure: every wait answers, and every checkpoint lands. On f869b3f the refused
/// spawn panicked the driver, which never started again, and every wait polled forever.
#[test]
fn many_tickets_in_a_task_capped_process_all_land() {
    let scope = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        "TasksMax=48",
    ];
    let usable = std::process::Command::new(scope[0])
        .args(&scope[1..])
        .arg("true")
        .output();
    if !usable.is_ok_and(|output| output.status.success()) {
        eprintln!("SKIPPED: no systemd user manager to create a task-limited cgroup");
        return;
    }
    let dir = temporary("fetch-tasks");
    store(&dir);
    // The hub runs here, outside the cap: only the fetching process is limited.
    let checkpoints = many("");
    let hub = hub(
        checkpoints.iter().map(|(c, _)| Arc::clone(c)).collect(),
        None,
        Duration::ZERO,
        serve_all(),
    );
    let mut child = std::process::Command::new(scope[0])
        .args(&scope[1..])
        .arg(std::env::current_exe().unwrap())
        .args(["--exact", "task_capped_child", "--nocapture"])
        .env("TFS_FETCH_CHILD", &dir)
        .env("TFS_FETCH_HUB", &hub.base)
        .spawn()
        .unwrap();
    // Test harness bound on a wedged fetcher; the product has no such limit.
    let began = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if began.elapsed() > Duration::from_secs(90) {
            let _ = child.kill();
            panic!("the waits never answered under TasksMax=48");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    assert!(status.success());
    let store = Store::open(&dir.join("store")).unwrap();
    for (checkpoint, _) in &checkpoints {
        assert!(checkpoint
            .bodies
            .keys()
            .all(|hex| store.contains(hex) || store.manifest_path(hex).is_file()));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn task_capped_child() {
    let (Some(dir), Some(base)) = (
        std::env::var_os("TFS_FETCH_CHILD"),
        std::env::var_os("TFS_FETCH_HUB"),
    ) else {
        return;
    };
    let store = Store::open(&std::path::Path::new(&dir).join("store")).unwrap();
    let fetcher = Fetcher::of(&store).unwrap();
    let checkpoints = many(base.to_str().unwrap());
    let tickets: Vec<_> = (checkpoints.iter())
        .map(|(checkpoint, origin)| {
            fetcher.want(origin, access(), &checkpoint.blobs, Priority::Now)
        })
        .collect();
    // Waited one after another on this thread: the cap is for the Fetcher's threads, not the
    // test's, and every ticket is already wanted at once.
    for ticket in &tickets {
        ticket.wait(&[], &never()).unwrap();
    }
}

/// A credential that panics on every object GET: a fault inside a download's own stream.
struct Panicking;

impl tensorfs_core::transport::CredentialProvider for Panicking {
    fn headers(&self, ask: &tensorfs_core::transport::Ask) -> Vec<(String, String)> {
        assert!(
            !ask.url.target.starts_with("/obj/"),
            "a credential that panics on objects"
        );
        vec![]
    }
}

/// A download that panics ends the waits on what it left unlanded with a refusal, and the
/// downloader goes on: a later wait lands the checkpoint the panic took down with it. On
/// f869b3f the panicking stream left its run waiting on it forever.
#[test]
fn a_panicking_download_ends_its_waits_and_the_fetcher_goes_on() {
    let dir = temporary("fetch-panic");
    let store = Store::open(&store(&dir)).unwrap();
    let (bad, good) = (
        Arc::new(components("acme/bad", &[8], 16 << 10, 60, &[])),
        Arc::new(components("acme/good", &[8], 16 << 10, 61, &[])),
    );
    let hub = hub(
        vec![Arc::clone(&bad), Arc::clone(&good)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let fetcher = Fetcher::new(&store, 8, SAMPLE).unwrap();
    let panicking: Arc<dyn Access> = Arc::new(Plain {
        credential: Panicking,
        policy: SourcePolicy::default(),
    });
    let refused = fetcher
        .want(&origin(&hub, &bad), panicking, &bad.blobs, Priority::Now)
        .wait(&[], &never());
    assert!(refused.is_err(), "a panicking download's wait answered Ok");
    let landed = fetcher.want(&origin(&hub, &good), access(), &good.blobs, Priority::Now);
    if landed.wait(&[], &never()).is_err() {
        landed.wait(&[], &never()).unwrap();
    }
    assert!(good
        .bodies
        .keys()
        .all(|hex| store.contains(hex) || store.manifest_path(hex).is_file()));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The executor's side of the framing: a kind byte, a big-endian length, JSON.
struct Peer(LocalStream);

impl Peer {
    fn send(&mut self, body: &str) {
        use std::io::Write;
        let mut head = vec![0u8];
        head.extend((body.len() as u32).to_be_bytes());
        self.0.write_all(&head).unwrap();
        self.0.write_all(body.as_bytes()).unwrap();
    }
    fn receive(&mut self) -> tensorfs_core::jcs::Json {
        use std::io::Read;
        let mut head = [0u8; 5];
        self.0.read_exact(&mut head).unwrap();
        let mut body = vec![0u8; u32::from_be_bytes(head[1..].try_into().unwrap()) as usize];
        self.0.read_exact(&mut body).unwrap();
        tensorfs_core::jcs::parse(&body, 1 << 20).unwrap()
    }
}
