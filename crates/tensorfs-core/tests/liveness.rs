//! xs-007 rows 4 and 21: nothing on the ingest path ends work on a clock.
//!
//! Row 4 — `tfs ingest run` wrapped its worker in `ulimit -t 3600`, a 400 GB ceiling at the
//! measured ~114 MB/s. The fixture here is sized so the SMALLEST ceiling that mechanism could
//! express (`ulimit -t 1`, one CPU-second, ~114 MB in production and less in this build)
//! kills the worker mid-write; the same run under the shipped supervisor completes. The
//! supervisor is then shown to end a worker that has genuinely stopped — SIGSTOP is exactly
//! the observation a stall presents: no byte moved, no CPU tick, never seen running.
//!
//! Row 21 — `tfs reap` unlinked admission temps by mtime, so a slow live writer's temp could
//! be reaped from under it. Liveness is now the lock the writer holds on the temp.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
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

/// One safetensors carrier holding one F32 tensor of `bytes`, filled with a running
/// counter so no two carriers — and no two 64 MiB segments — dedupe against each other.
fn carrier(path: &Path, key: &str, bytes: u64, seed: u32) {
    let header = format!(
        "{{{key:?}:{{\"data_offsets\":[0,{bytes}],\"dtype\":\"F32\",\"shape\":[{}]}}}}",
        bytes / 4
    );
    let mut f = File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    let mut chunk = vec![0u8; MIB as usize];
    let mut n = seed;
    let mut left = bytes;
    while left > 0 {
        for word in chunk.chunks_exact_mut(4) {
            word.copy_from_slice(&n.to_le_bytes());
            n = n.wrapping_add(1);
        }
        let take = left.min(MIB) as usize;
        f.write_all(&chunk[..take]).unwrap();
        left -= take as u64;
    }
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

/// Bank a reviewed two-carrier profile and write its closed source plan.
fn plan(root: &Path, weight: &Path, bias: &Path) -> PathBuf {
    let registry = root.join("registry.json");
    let order = root.join("order.json");
    fs::write(&order, r#"[["encoder","weight"],["decoder","bias"]]"#).unwrap();
    let mut bank = tfs();
    bank.args([
        "ingest",
        "bank",
        "safetensors.diffusers",
        "diffusers.identity/1",
        "--registry",
    ]);
    bank.arg(&registry);
    bank.args(["--source-profile", "big", "--order"]);
    bank.arg(&order);
    bank.args(["--source", &format!("encoder={}", weight.display())]);
    bank.args(["--target", "encoder=plain/1"]);
    bank.args(["--source", &format!("decoder={}", bias.display())]);
    bank.args(["--target", "decoder=plain/1"]);
    assert_ok(&bank.output().unwrap(), "bank");

    let plan = root.join("source-plan.json");
    let mut command = tfs();
    command.args(["ingest", "source-plan", "--registry"]);
    command.arg(&registry);
    command.args(["--out", plan.to_str().unwrap()]);
    command.args(["--carrier", weight.to_str().unwrap()]);
    command.args(["--carrier", bias.to_str().unwrap()]);
    assert_ok(&command.output().unwrap(), "source-plan");
    plan
}

fn session_of(plan: &Path) -> String {
    let body = fs::read_to_string(plan).unwrap();
    let tail = body.split_once(r#""session":""#).unwrap().1;
    tail.split_once('"').unwrap().0.to_string()
}

fn field(text: &str, prefix: &str) -> u64 {
    text.lines()
        .find_map(|line| line.strip_prefix(prefix))
        .unwrap_or_else(|| panic!("no {prefix:?} line in:\n{text}"))
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(Stdio::null())
        .status()
        .unwrap()
        .success()
}

fn reap(store: &Path) -> (u64, u64) {
    let out = assert_ok(&tfs().args(["reap"]).arg(store).output().unwrap(), "reap");
    let words: Vec<&str> = out.split_whitespace().collect();
    let gone = words[words.iter().position(|w| *w == "removed").unwrap() + 1];
    let kept = words[words.iter().position(|w| *w == "kept").unwrap() + 1];
    (gone.parse().unwrap(), kept.parse().unwrap())
}

#[test]
fn ingest_run_ends_on_observed_inactivity_and_never_on_cpu_time() {
    let root = temporary("row4");
    fs::create_dir_all(&root).unwrap();
    let weight = root.join("weight.safetensors");
    let bias = root.join("bias.safetensors");
    // 256 MiB: 1.4 CPU-seconds in a release build of this crate's pure-Rust SHA-256 and
    // ~10 in a debug build — past `ulimit -t 1` in either, and five 64 MiB segments, so
    // the run shows the supervisor several fsync-scale silences to learn from.
    let (weight_bytes, bias_bytes) = (160 * MIB, 96 * MIB);
    carrier(&weight, "weight", weight_bytes, 0);
    carrier(&bias, "bias", bias_bytes, 1 << 30);
    let plan = plan(&root, &weight, &bias);
    let store = root.join("store");
    assert_ok(
        &tfs().args(["store", "init"]).arg(&store).output().unwrap(),
        "store init",
    );

    // The old fence, at the smallest ceiling it could express: the worker dies mid-write
    // and the run reports it. This is what every ingest over the ceiling looked like.
    let fenced = Command::new("/bin/sh")
        .arg("-c")
        .arg(format!(
            "ulimit -t 1; exec '{}' ingest run '{}' --source-plan '{}'",
            env!("CARGO_BIN_EXE_tfs"),
            store.display(),
            plan.display()
        ))
        .output()
        .unwrap();
    let fenced_err = String::from_utf8_lossy(&fenced.stderr);
    assert!(
        !fenced.status.success() && fenced_err.contains("worker exited"),
        "one CPU-second must have killed the fenced worker:\n{}{fenced_err}",
        String::from_utf8_lossy(&fenced.stdout)
    );

    // A worker that has genuinely stopped. SIGSTOP presents exactly what a stall does —
    // no byte moved, no CPU tick, never seen running — and the supervisor must end it.
    let temps = || {
        fs::read_dir(store.join("tmp"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("put-"))
            .count()
    };
    let orphans = temps();
    let mut run = tfs()
        .args(["ingest", "run"])
        .arg(&store)
        .arg("--source-plan")
        .arg(&plan)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(run.stdout.take().unwrap()).lines();
    let worker: u32 = loop {
        let line = lines
            .next()
            .expect("run ended before naming its worker")
            .unwrap();
        if let Some(pid) = line.strip_prefix("worker       pid ") {
            break pid.trim().parse().unwrap();
        }
    };
    // Stopped mid-stream: once its own temp is open, not during the preamble.
    let started = Instant::now();
    while temps() <= orphans && started.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        temps() > orphans,
        "the worker never opened an admission temp"
    );
    assert!(Command::new("kill")
        .args(["-STOP", &worker.to_string()])
        .status()
        .unwrap()
        .success());
    let drained: Vec<String> = lines.map(Result::unwrap).collect();
    let stalled = run.wait_with_output().unwrap();
    let stalled_err = String::from_utf8_lossy(&stalled.stderr);
    assert!(
        !stalled.status.success() && stalled_err.contains("worker stalled"),
        "a stopped worker must be ended as stalled:\n{}\n{stalled_err}",
        drained.join("\n")
    );
    assert!(!alive(worker), "the stalled worker must be gone");
    let report = drained
        .iter()
        .find(|l| l.starts_with("progress     moved"))
        .expect("the run reports what it observed");
    assert!(report.contains("patience "), "{report}");

    // Both dead workers left an orphaned temp; nothing alive holds one.
    let (gone, kept) = reap(&store);
    assert!(gone >= 1 && kept == 0, "reap removed {gone}, kept {kept}");

    // The same session, resumed, runs to completion — the retry that used to hit the
    // same wall. Nothing about its duration is consulted.
    let out = assert_ok(
        &tfs()
            .args(["ingest", "run"])
            .arg(&store)
            .arg("--source-plan")
            .arg(&plan)
            .output()
            .unwrap(),
        "ingest run",
    );
    assert!(out.contains("candidate    manifest sha256:"), "{out}");
    assert!(out.contains("worker exited 0"), "{out}");
    assert!(
        field(&out, "progress     moved ") >= weight_bytes + bias_bytes,
        "{out}"
    );
    assert!(
        !out.contains("progress\t"),
        "records are consumed, not relayed:\n{out}"
    );

    let session = session_of(&plan);
    let install = tfs()
        .args(["ingest", "install"])
        .arg(&store)
        .args([
            &session,
            "local",
            "big-model",
            &"11".repeat(32),
            "local",
            "--observed",
            "absent",
            "--source-uri",
            "hf://reviewed/source@1111111111111111111111111111111111111111",
            "--declared-license",
            "test-only",
        ])
        .output()
        .unwrap();
    assert!(assert_ok(&install, "install").contains("installed    sha256:"));
    let _ = fs::remove_dir_all(root);
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
