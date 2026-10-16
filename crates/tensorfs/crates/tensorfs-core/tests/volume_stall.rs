//! A volume whose reads stop answering, through the real kernel path (FUSE): its own binary, so
//! the threads counted are this test's alone.

mod common;
#[path = "common/fuse.rs"]
mod fuse;

use std::fs;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{components, hub, serve_all, store, temporary, Checkpoint, Hub};
use tensorfs_core::fetch::{Access, Fetcher, Origin, Plain, Priority, Volume};
use tensorfs_core::store::Store;
use tensorfs_core::transport::{Anonymous, PullCancellation, SourcePolicy};

const TTL: Duration = Duration::from_secs(3600);

/// Bytes of `checkpoint` that came from (the hub, the volume).
fn fetch(fetcher: &Fetcher, hub: &Hub, checkpoint: &Checkpoint) -> (u64, u64) {
    let origin = Origin {
        hub: hub.base.clone(),
        refspec: checkpoint.refspec(),
        lane: checkpoint.lane.clone(),
        manifest: checkpoint.manifest.clone(),
    };
    let access: Arc<dyn Access> = Arc::new(Plain {
        credential: Anonymous,
        policy: SourcePolicy::default(),
    });
    let ticket = fetcher.want(&origin, access, &checkpoint.blobs, Priority::Now);
    ticket.wait(&[], &PullCancellation::default()).unwrap();
    ticket.landed()
}

/// This process's volume threads: the tender and every operation not yet returned.
fn volume_threads() -> usize {
    let tasks = fs::read_dir("/proc/self/task").unwrap().flatten();
    tasks
        .filter(|t| {
            fs::read_to_string(t.path().join("comm")).is_ok_and(|c| c.trim() == "tensorfs-volume")
        })
        .count()
}

/// Reads stop: the read in flight is abandoned to the hub, nothing more starts on the volume
/// (every object comes from the hub, and no thread is added however much is fetched), and the
/// volume serves again once the stuck read returns.
#[test]
fn a_volume_that_stops_answering_is_left_alone_until_it_answers() {
    let dir = temporary("volume-stall");
    let mounted = fuse::mount(&dir.join("backing"), &dir.join("volume"), 1 << 40);
    let made: Vec<Arc<Checkpoint>> = (0..3)
        .map(|n| {
            Arc::new(components(
                &format!("acme/model{n}"),
                &[6],
                128 << 10,
                n,
                &[],
            ))
        })
        .collect();
    let volume = Volume::new(&mounted.backing, TTL);
    for body in made.iter().flat_map(|c| c.bodies.values()) {
        let path = volume.object(&tensorfs_core::ids::ObjectRef::of(body).sha256);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }
    let hub = hub(made.clone(), None, Duration::ZERO, serve_all());
    let store = Store::open(&store(&dir)).unwrap();
    let fetcher = Fetcher::new(&store, 8, Duration::from_millis(100))
        .unwrap()
        .with_volume(Volume::new(&mounted.at, TTL));

    mounted.control.block(true);
    let began = Instant::now();
    assert_eq!(fetch(&fetcher, &hub, &made[0]), (made[0].total(), 0));
    assert!(
        mounted.control.reads.load(Ordering::Acquire) > 0,
        "no read reached the volume"
    );
    let threads = volume_threads();
    assert!(threads >= 2, "the tender and the stuck read: {threads}");
    assert_eq!(fetch(&fetcher, &hub, &made[1]), (made[1].total(), 0));
    assert_eq!(
        volume_threads(),
        threads,
        "threads grew while the volume was stuck"
    );
    assert!(made[..2]
        .iter()
        .flat_map(|c| c.bodies.keys())
        .all(|hex| hub.gets(hex) == 1));
    eprintln!("stalled and served from the hub in {:?}", began.elapsed());

    mounted.control.block(false);
    let began = Instant::now();
    while volume_threads() > 1 {
        assert!(
            began.elapsed() < Duration::from_secs(30),
            "the stuck read never returned"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        fetch(&fetcher, &hub, &made[2]),
        (0, made[2].total()),
        "the volume serves again"
    );
    assert!(made[2].bodies.keys().all(|hex| hub.gets(hex) == 0));
    let log = fs::read_to_string(store.root().join("logs/transport.log")).unwrap();
    assert!(
        log.contains("silent past") && log.contains("answers again"),
        "{log}"
    );
    drop(fetcher);
    drop(mounted);
    let _ = fs::remove_dir_all(&dir);
}
