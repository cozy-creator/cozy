//! The datacenter volume as the fetcher's second source: real Stores, a real loopback hub, and
//! a shared directory (or a FUSE mount with a stated capacity) as the volume.

// Helpers the Linux-only FUSE tests share go unused elsewhere.
#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]

mod common;
#[cfg(target_os = "linux")]
#[path = "common/fuse.rs"]
mod fuse;

use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use common::{components, hub, serve_all, store, temporary, Checkpoint, Hub};
use tensorfs_core::fetch::{Access, Fetcher, Origin, Plain, Priority, Ticket, Volume};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::Store;
use tensorfs_core::transport::{Anonymous, PullCancellation, SourcePolicy};

/// Long enough that no healthy volume operation on a loaded runner looks stalled.
const SAMPLE: Duration = Duration::from_millis(500);
const TTL: Duration = Duration::from_secs(3600);
const RESERVE: u64 = 1 << 30;

fn origin(hub: &Hub, checkpoint: &Checkpoint) -> Origin {
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

fn until(what: &str, ready: impl Fn() -> bool) {
    let began = Instant::now();
    while !ready() {
        assert!(began.elapsed() < Duration::from_secs(60), "never: {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A fetcher filling `volume`, which exists: the volume's root is a mount, never made.
fn fetcher(store: &Store, volume: &Path) -> Fetcher {
    fs::create_dir_all(volume).unwrap();
    Fetcher::new(store, 8, SAMPLE)
        .unwrap()
        .with_volume(Volume::new(volume, TTL))
}

/// Every object of `checkpoint`, structure included, landed: the ticket that brought them.
fn fetch(fetcher: &Fetcher, hub: &Hub, checkpoint: &Checkpoint) -> Ticket {
    let ticket = fetcher.want(
        &origin(hub, checkpoint),
        access(),
        &checkpoint.blobs,
        Priority::Now,
    );
    ticket.wait(&[], &PullCancellation::default()).unwrap();
    ticket
}

fn on_volume(volume: &Path, hex: &str) -> Option<Vec<u8>> {
    fs::read(Volume::new(volume, TTL).object(hex)).ok()
}

/// Every object of `checkpoint` is on the volume, byte for byte.
fn filled(volume: &Path, checkpoint: &Checkpoint) -> bool {
    (checkpoint.bodies.iter()).all(|(hex, body)| on_volume(volume, hex).as_ref() == Some(body))
}

/// `body` on the volume at its digest, last used `age` ago.
fn place(volume: &Path, body: &[u8], age: Duration) -> std::path::PathBuf {
    let path = Volume::new(volume, TTL).object(&ObjectRef::of(body).sha256);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, body).unwrap();
    let file = fs::File::options().write(true).open(&path).unwrap();
    file.set_modified(SystemTime::now() - age).unwrap();
    path
}

fn junk(size: usize, salt: u8) -> Vec<u8> {
    (0..size)
        .map(|at| (at as u8).wrapping_mul(31).wrapping_add(salt))
        .collect()
}

fn temps(volume: &Path) -> usize {
    fs::read_dir(volume.join("tmp")).map_or(0, |d| d.count())
}

/// Machine A fetches from the hub and fills the volume; machine B, fresh, gets every object
/// from the volume: no object GET, one closure read, no presign.
#[test]
fn a_second_machine_gets_every_object_from_the_volume() {
    let dir = temporary("volume-second");
    let volume = dir.join("volume");
    let checkpoint = Arc::new(components("acme/model", &[3, 2], 96 << 10, 1, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let a = Store::open(&store(&dir.join("a"))).unwrap();
    let fa = fetcher(&a, &volume);
    assert_eq!(
        fetch(&fa, &hub, &checkpoint).landed(),
        (checkpoint.total(), 0)
    );
    until("A's write-back", || filled(&volume, &checkpoint));
    let (gets, presigns, closures) = (hub.total_gets(), hub.gets("presign"), hub.gets("closure"));

    let b = Store::open(&store(&dir.join("b"))).unwrap();
    let fb = fetcher(&b, &volume);
    assert_eq!(
        fetch(&fb, &hub, &checkpoint).landed(),
        (0, checkpoint.total()),
        "B's bytes by (hub, volume)"
    );
    assert!(checkpoint
        .bodies
        .keys()
        .all(|hex| b.contains(hex) || b.manifest_path(hex).is_file()));
    let log = fs::read_to_string(b.root().join("logs/transport.log")).unwrap();
    let line = format!(
        "fetched acme/model@r1 public hub 0 B volume {} B",
        checkpoint.total()
    );
    assert!(log.contains(&line), "{log}");
    assert_eq!(hub.total_gets(), gets, "B asked the hub for an object");
    assert_eq!(hub.gets("presign"), presigns, "B presigned");
    assert_eq!(
        hub.gets("closure"),
        closures + 1,
        "B's access check is its one closure read"
    );
    assert_eq!(temps(&volume), 0);
    let _ = fs::remove_dir_all(&dir);
}

/// A same-length corrupt entry is refused by the Store's door, fetched from the hub, and the
/// hub's copy replaces it on the volume.
#[test]
fn a_corrupt_entry_is_refused_fetched_from_the_hub_and_replaced() {
    let dir = temporary("volume-corrupt");
    let volume = dir.join("volume");
    let checkpoint = Arc::new(components("acme/model", &[4], 64 << 10, 2, &[]));
    for body in checkpoint.bodies.values() {
        place(&volume, body, Duration::ZERO);
    }
    let bad = &checkpoint.blobs[1];
    fs::write(
        Volume::new(&volume, TTL).object(&bad.sha256),
        vec![7u8; bad.length as usize],
    )
    .unwrap();
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let b = Store::open(&store(&dir.join("b"))).unwrap();
    let fb = fetcher(&b, &volume);
    assert_eq!(
        fetch(&fb, &hub, &checkpoint).landed(),
        (bad.length, checkpoint.total() - bad.length)
    );
    assert_eq!(hub.total_gets(), 1);
    assert_eq!(hub.gets(&bad.sha256), 1);
    assert_eq!(
        b.read_range(&bad.sha256, 0, bad.length).unwrap(),
        checkpoint.bodies[&bad.sha256]
    );
    until("the corrupt entry replaced", || {
        filled(&volume, &checkpoint)
    });
    let _ = fs::remove_dir_all(&dir);
}

/// Two machines fetching one cold model at once both write it back: each object ends on the
/// volume once, whole, and no temp is left.
#[test]
fn two_writers_of_one_object_converge_and_leave_no_temps() {
    let dir = temporary("volume-race");
    let volume = dir.join("volume");
    let checkpoint = Arc::new(components("acme/model", &[8], 128 << 10, 3, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let (a, b) = (
        Store::open(&store(&dir.join("a"))).unwrap(),
        Store::open(&store(&dir.join("b"))).unwrap(),
    );
    let (fa, fb) = (fetcher(&a, &volume), fetcher(&b, &volume));
    std::thread::scope(|scope| {
        scope.spawn(|| fetch(&fa, &hub, &checkpoint));
        scope.spawn(|| fetch(&fb, &hub, &checkpoint));
    });
    until("both write-backs", || {
        filled(&volume, &checkpoint) && temps(&volume) == 0
    });
    let objects = Volume::new(&volume, TTL).object(&checkpoint.manifest.sha256);
    let fanout = objects
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let count: usize = fs::read_dir(fanout)
        .unwrap()
        .flatten()
        .flat_map(|a| fs::read_dir(a.path()).unwrap().flatten())
        .map(|b| fs::read_dir(b.path()).unwrap().count())
        .sum();
    assert_eq!(count, checkpoint.bodies.len());
    let _ = fs::remove_dir_all(&dir);
}

/// A checkpoint only its owner may read never reaches the volume.
#[test]
fn a_private_origin_writes_nothing() {
    let dir = temporary("volume-private");
    let volume = dir.join("volume");
    let mut private = components("acme/private", &[3], 64 << 10, 4, &[]);
    private.public = false;
    let (private, public) = (
        Arc::new(private),
        Arc::new(components("acme/public", &[3], 64 << 10, 5, &[])),
    );
    let hub = hub(
        vec![Arc::clone(&private), Arc::clone(&public)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let a = Store::open(&store(&dir.join("a"))).unwrap();
    let fetcher = fetcher(&a, &volume);
    fetch(&fetcher, &hub, &private);
    fetch(&fetcher, &hub, &public);
    until("the public write-back", || filled(&volume, &public));
    assert!(private
        .bodies
        .keys()
        .all(|hex| on_volume(&volume, hex).is_none()));
    let _ = fs::remove_dir_all(&dir);
}

/// A volume short of room: a write evicts the least recently used objects, oldest first, and
/// only as many as it needs. When even an empty volume could not take an object, nothing is
/// evicted, the write is skipped and counted, and the fetch is unaffected.
#[test]
#[cfg(target_os = "linux")]
fn a_short_volume_evicts_the_least_recently_used_then_skips() {
    let dir = temporary("volume-short");
    const J: u64 = 256 << 10;
    let mounted = fuse::mount(
        &dir.join("backing"),
        &dir.join("volume"),
        RESERVE + 3 * J + (64 << 10),
    );
    let hour = Duration::from_secs(3600);
    let old: Vec<_> = (0..3)
        .map(|n| {
            place(
                &mounted.backing,
                &junk(J as usize, n),
                hour * (3 - n as u32) / 4,
            )
        })
        .collect();
    let checkpoint = Arc::new(components("acme/model", &[2], J as usize, 6, &[]));
    let big = Arc::new(components("acme/big", &[1], 4 << 20, 7, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint), Arc::clone(&big)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let a = Store::open(&store(&dir.join("a"))).unwrap();
    let fetcher = fetcher(&a, &mounted.at);
    fetch(&fetcher, &hub, &checkpoint);
    until("the write-back", || filled(&mounted.at, &checkpoint));
    assert!(!old[0].exists() && !old[1].exists(), "the two oldest went");
    assert!(
        old[2].exists(),
        "the youngest stayed: two evictions made the room"
    );

    let skipped = tensorfs_core::stats::snapshot().volume_skipped_bytes;
    fetch(&fetcher, &hub, &big);
    until("the big write skipped", || {
        tensorfs_core::stats::snapshot().volume_skipped_bytes >= skipped + (4 << 20)
    });
    assert!(on_volume(&mounted.at, &big.blobs[0].sha256).is_none());
    assert!(
        filled(&mounted.at, &checkpoint) && old[2].exists(),
        "a hopeless write evicts nothing"
    );
    assert!(a.contains(&big.blobs[0].sha256));
    drop(mounted);
    let _ = fs::remove_dir_all(&dir);
}

/// A network volume's quota, which `statfs` does not show (RunPod's report the whole cluster):
/// a write refused for space evicts the least recently used objects and is written; one that
/// even an emptied volume could not take is skipped and counted, and the fetch is unaffected.
#[test]
#[cfg(target_os = "linux")]
fn a_quota_statfs_does_not_show_evicts_on_a_refused_write() {
    let dir = temporary("volume-quota");
    const J: u64 = 256 << 10;
    let mounted = fuse::mount(&dir.join("backing"), &dir.join("volume"), 1 << 50);
    mounted
        .control
        .quota
        .store(3 * J + (64 << 10), Ordering::Release);
    let hour = Duration::from_secs(3600);
    let old: Vec<_> = (0..3)
        .map(|n| {
            place(
                &mounted.backing,
                &junk(J as usize, n),
                hour * (3 - n as u32) / 4,
            )
        })
        .collect();
    let checkpoint = Arc::new(components("acme/model", &[2], J as usize, 6, &[]));
    let big = Arc::new(components("acme/big", &[1], 4 << 20, 7, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint), Arc::clone(&big)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let a = Store::open(&store(&dir.join("a"))).unwrap();
    let fetcher = fetcher(&a, &mounted.at);
    fetch(&fetcher, &hub, &checkpoint);
    until("the write-back", || filled(&mounted.at, &checkpoint));
    assert!(!old[0].exists() && !old[1].exists(), "the two oldest went");
    assert!(
        old[2].exists(),
        "the youngest stayed: two evictions made the room"
    );
    assert_eq!(temps(&mounted.at), 0);

    let skipped = tensorfs_core::stats::snapshot().volume_skipped_bytes;
    fetch(&fetcher, &hub, &big);
    until("the big write skipped", || {
        tensorfs_core::stats::snapshot().volume_skipped_bytes >= skipped + (4 << 20)
    });
    assert!(on_volume(&mounted.at, &big.blobs[0].sha256).is_none());
    assert!(
        filled(&mounted.at, &checkpoint) && old[2].exists(),
        "a hopeless write evicts nothing"
    );
    assert!(a.contains(&big.blobs[0].sha256));
    drop(mounted);
    let _ = fs::remove_dir_all(&dir);
}

/// Temps and objects unused for the TTL go on the next short write, before anything in use.
#[test]
#[cfg(target_os = "linux")]
fn temps_and_objects_past_the_ttl_go_on_the_next_short_write() {
    let dir = temporary("volume-ttl");
    const J: u64 = 256 << 10;
    let mounted = fuse::mount(
        &dir.join("backing"),
        &dir.join("volume"),
        RESERVE + 3 * J + (64 << 10),
    );
    let expired = TTL * 2;
    let stale = place(&mounted.backing, &junk(J as usize, 1), expired);
    let young = place(
        &mounted.backing,
        &junk(J as usize, 2),
        Duration::from_secs(60),
    );
    let temp = mounted.backing.join("tmp/abandoned.1.2");
    fs::create_dir_all(temp.parent().unwrap()).unwrap();
    fs::write(&temp, junk(J as usize, 3)).unwrap();
    fs::File::options()
        .write(true)
        .open(&temp)
        .unwrap()
        .set_modified(SystemTime::now() - expired)
        .unwrap();
    let checkpoint = Arc::new(components("acme/model", &[1], J as usize, 8, &[]));
    let hub = hub(
        vec![Arc::clone(&checkpoint)],
        None,
        Duration::ZERO,
        serve_all(),
    );
    let a = Store::open(&store(&dir.join("a"))).unwrap();
    let fa = fetcher(&a, &mounted.at);
    fetch(&fa, &hub, &checkpoint);
    until("the write-back", || filled(&mounted.at, &checkpoint));
    assert!(!stale.exists() && !temp.exists(), "past the TTL");
    assert!(young.exists(), "in use, and not needed");
    assert!(
        Volume::mounted(&mounted.at, TTL).is_some(),
        "a mount is a volume"
    );
    assert!(
        Volume::mounted(&mounted.backing, TTL).is_none(),
        "a plain directory is none"
    );
    assert!(Volume::mounted(dir.join("absent"), TTL).is_none());
    drop(mounted);
    let _ = fs::remove_dir_all(&dir);
}
