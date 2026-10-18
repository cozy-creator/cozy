//! xs-007 row 21 — `tfs reap` unlinked admission temps by mtime, so a slow live writer's temp could
//! be reaped from under it. Liveness is now the lock the writer holds on the temp.
//!
//! Unix only.
#![cfg(unix)]

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MIB: u64 = 1 << 20;

fn temporary(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tensorfs-liveness-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn tfs() -> Command {
    Command::new(env!("CARGO_BIN_EXE_tfs"))
}

fn assert_ok(output: &std::process::Output, what: &str) -> String {
    assert!(
        output.status.success(),
        "{what} failed:\n{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn reap(store: &Path) -> (u64, u64) {
    let out = assert_ok(&tfs().args(["reap"]).arg(store).output().unwrap(), "reap");
    let words: Vec<&str> = out.split_whitespace().collect();
    let gone = words[words.iter().position(|w| *w == "removed").unwrap() + 1];
    let kept = words[words.iter().position(|w| *w == "kept").unwrap() + 1];
    (gone.parse().unwrap(), kept.parse().unwrap())
}

#[test]
fn reap_decides_on_the_writers_lock_and_never_on_age() {
    let root = temporary("row21");
    fs::create_dir_all(&root).unwrap();
    let store = root.join("store");
    assert_ok(
        &tfs().args(["store", "init"]).arg(&store).output().unwrap(),
        "store init",
    );
    let object = root.join("object");
    fs::write(&object, vec![7u8; 4 * MIB as usize]).unwrap();

    // A live writer parked one link() short of admission, holding its temp.
    let ready = root.join("ready");
    let mut writer = tfs()
        .args(["put"])
        .arg(&store)
        .arg(&object)
        .args(["--fault", "written", "--ready"])
        .arg(&ready)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while !ready.is_file() && started.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(ready.is_file(), "the writer never reached its kill point");
    let temp = fs::read_dir(store.join("tmp"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.file_name().unwrap().to_string_lossy().starts_with("put-"))
        .expect("the writer holds a temp");
    // Older than any clock the old reap would have accepted.
    File::options()
        .write(true)
        .open(&temp)
        .unwrap()
        .set_modified(SystemTime::now() - Duration::from_secs(48 * 3600))
        .unwrap();

    assert_eq!(
        reap(&store),
        (0, 1),
        "a live writer's temp is kept whatever its age"
    );
    assert!(temp.is_file());

    assert!(Command::new("kill")
        .args(["-9", &writer.id().to_string()])
        .status()
        .unwrap()
        .success());
    let _ = writer.wait();
    assert_eq!(reap(&store), (1, 0), "a dead writer's temp is an orphan");
    assert!(!temp.exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn concurrent_marker_timestamp_reservations_are_unique() {
    use std::sync::{Arc, Barrier};
    let barrier = Arc::new(Barrier::new(16));
    let values = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    let mut values = Vec::with_capacity(4096);
                    for _ in 0..4096 {
                        barrier.wait();
                        values.push(tensorfs_core::meta::now_nanos_unique());
                    }
                    values
                })
            })
            .collect();
        threads
            .into_iter()
            .flat_map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
    });
    let count = values.len();
    let unique = values.into_iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(
        unique.len(),
        count,
        "independent live marker owners received the same name"
    );
}
