//! Run 1297 (2026-09-27): a pull growing toward 128 streams under a 1,024 soft
//! RLIMIT_NOFILE ran out of descriptors mid-pull. This binary sets the limit itself and
//! downloads whole checkpoints on the Store's Fetcher with as many requests as the table
//! allows. Its own binary, because RLIMIT_NOFILE is process-wide. Windows has no such limit.
#![cfg(unix)]
mod common;

use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

use common::{components, hub, serve_all, temporary, Answer, Checkpoint, Hub, Rule};
use rustix::process::{getrlimit, setrlimit, Resource, Rlimit};
use tensorfs_core::fetch_v1::{whole, Fetcher, Origin, Plain, Priority};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::Store;
use tensorfs_core::transport::{Anonymous, PullCancellation, SourcePolicy, PULL_STREAMS};

const SIZE: usize = 512 << 10;
/// Each 512 KiB answer is written in four pieces this far apart: a connection moves ~1.6 MB/s,
/// so only more requests buy more bytes.
const GAP: Duration = Duration::from_millis(80);

/// Three checkpoints of 640 objects at one hub, one per walk.
fn checkpoints() -> Vec<Arc<Checkpoint>> {
    ["acme/a", "acme/b", "acme/c"]
        .iter()
        .enumerate()
        .map(|(salt, model)| Arc::new(components(model, &[640], SIZE, salt + 30, &[])))
        .collect()
}

/// Download `checkpoint` whole; every object is re-hashed from the Store.
fn walk(hub: &Hub, checkpoint: &Checkpoint) {
    let root = temporary("walk");
    let store = Store::init(&root).unwrap();
    let fetcher = Fetcher::new(&store, PULL_STREAMS, Duration::from_secs(1)).unwrap();
    let origin = Origin {
        hub: hub.base.clone(),
        refspec: checkpoint.refspec(),
        lane: checkpoint.lane.clone(),
        manifest: checkpoint.manifest.clone(),
    };
    let policy = SourcePolicy { allowed_hosts: vec!["127.0.0.1".into()], allow_local: true, ..Default::default() };
    let access = Arc::new(Plain { credential: Anonymous, policy });
    whole(&fetcher, &origin, access, Priority::Now, &PullCancellation::default(), &|_, _| {})
        .unwrap_or_else(|refusal| panic!("the download refused: {refusal}"));
    for object in &checkpoint.blobs {
        let mut bytes = Vec::new();
        store.open_verified(&object.sha256).unwrap().read_to_end(&mut bytes).unwrap();
        assert_eq!(ObjectRef::of(&bytes), *object, "the door admitted other bytes");
    }
    let _ = std::fs::remove_dir_all(root);
}

fn set_limit(soft: u64, hard: Option<u64>) {
    setrlimit(Resource::Nofile, Rlimit { current: Some(soft), maximum: hard }).unwrap();
}

#[test]
fn a_low_descriptor_limit_is_raised_and_a_full_table_is_backpressure() {
    let original = getrlimit(Resource::Nofile);
    let hard = original.maximum.unwrap_or(u64::MAX);
    assert!(hard >= 4096, "this proof needs a hard limit of at least 4096");
    let checkpoints = checkpoints();
    // The second walk's first asks wait at the hub until the squeeze lets them through.
    let squeezed: Arc<std::collections::HashSet<String>> =
        Arc::new(checkpoints[1].bodies.keys().cloned().collect());
    let rule: Rule = Arc::new({
        let squeezed = Arc::clone(&squeezed);
        move |hex, asked| match asked == 1 && squeezed.contains(hex) {
            true => Answer::Hold,
            false => Answer::Body,
        }
    });
    let hub = Arc::new(hub(checkpoints.clone(), None, GAP, rule));

    // A Go parent hands its child the original soft limit. The download raises it to the hard
    // limit and grows as far as the link rewards, with no descriptor error.
    set_limit(256, original.maximum);
    let before = tensorfs_core::descriptors::exhaustions();
    walk(&hub, &checkpoints[0]);
    let raised = getrlimit(Resource::Nofile).current.unwrap();
    assert!(raised > 256, "the soft limit was not raised");
    assert_eq!(tensorfs_core::descriptors::exhaustions(), before, "EMFILE after the raise");

    // The table shrinks under a running walk, driven by the walk itself, never by a clock.
    // Its asks are held at the hub, so nothing it opened can close: the soft limit drops to
    // half of what is open, and one answer is let through, which needs a descriptor to land.
    // Every open then meets a full table until three have, and the limit comes back. EMFILE
    // is backpressure: what met it is asked again and every byte lands.
    let squeeze = std::thread::spawn({
        let hub = Arc::clone(&hub);
        move || {
            while !hub.log().iter().any(|hex| squeezed.contains(hex)) {
                std::thread::sleep(Duration::from_millis(1));
            }
            let open = cozy_platform::process::open_fds().unwrap();
            let pressed = tensorfs_core::descriptors::exhaustions();
            set_limit(open / 2, original.maximum);
            hub.admit(1);
            while tensorfs_core::descriptors::exhaustions() < pressed + 3 {
                std::thread::sleep(Duration::from_millis(1));
            }
            set_limit(raised, original.maximum);
            hub.open();
        }
    });
    walk(&hub, &checkpoints[1]);
    squeeze.join().unwrap();
    assert!(tensorfs_core::descriptors::exhaustions() > before, "the squeeze never pressed the walk");

    // A process started under a hard limit of 256, which cannot be raised past.
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "a_hard_limit_child", "--nocapture"])
        .env("TFS_FD_CHILD", "1")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&child.stdout).to_string() + &String::from_utf8_lossy(&child.stderr);
    assert!(child.status.success() && text.contains("BOUNDED"), "{text}");
}

/// Under a hard limit of 256 the machine's budget is sized to the table, and the download
/// completes with every object verified.
#[test]
fn a_hard_limit_child() {
    if std::env::var_os("TFS_FD_CHILD").is_none() {
        return;
    }
    set_limit(256, Some(256));
    let checkpoints = checkpoints();
    let hub = hub(checkpoints.clone(), None, GAP, serve_all());
    walk(&hub, &checkpoints[2]);
    println!("BOUNDED");
}
