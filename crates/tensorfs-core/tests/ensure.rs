//! `tfs ensure` against a REAL Store and a REAL loopback hub (plain and TLS): closure,
//! presign and object GETs over sockets this test starts, driven through the shipped binary.

mod common;

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{checkpoint, hub, serve_all, store, temporary, tls_config, Answer, Checkpoint, Rule};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::jcs::{self, Json};
use tensorfs_core::store::Store;

const TFS: &str = env!("CARGO_BIN_EXE_tfs");

// ---------------------------------------------------------------- the binary

struct Run {
    ok: bool,
    lines: Vec<Json>,
    stdout: String,
}

impl Run {
    fn events(&self, name: &str) -> Vec<&Json> {
        self.lines
            .iter()
            .filter(|line| text(line, "event") == name)
            .collect()
    }
    fn last(&self) -> &Json {
        self.lines.last().expect("ensure printed nothing")
    }
}

fn field<'a>(line: &'a Json, name: &str) -> Option<&'a Json> {
    let Json::Obj(fields) = line else { return None };
    fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn num(line: &Json, name: &str) -> u64 {
    match field(line, name) {
        Some(Json::Num(n)) if *n >= 0.0 && n.fract() == 0.0 => *n as u64,
        other => panic!("{name} is {other:?} in {line:?}"),
    }
}

fn text(line: &Json, name: &str) -> String {
    match field(line, name) {
        Some(Json::Str(s)) => s.clone(),
        other => panic!("{name} is {other:?} in {line:?}"),
    }
}

fn flag(line: &Json, name: &str) -> bool {
    match field(line, name) {
        Some(Json::Bool(b)) => *b,
        other => panic!("{name} is {other:?} in {line:?}"),
    }
}

fn parse(line: &str) -> Json {
    jcs::parse(line.as_bytes(), 1 << 20).unwrap_or_else(|e| panic!("{e}: {line}"))
}

fn ensure(root: &Path, base: &str, extra: &[&str]) -> Command {
    let mut command = Command::new(TFS);
    command
        .args([
            "ensure",
            root.to_str().unwrap(),
            "acme/model@r1",
            "--hub",
            base,
        ])
        .args([
            "--lane",
            "public",
            "--step",
            "load weights",
            "--sample-seconds",
            "0.05",
        ])
        .args(extra)
        .env("TFS_CREDENTIAL", "");
    command
}

fn finish(output: std::process::Output) -> Run {
    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines = stdout.lines().map(parse).collect();
    Run {
        ok: output.status.success(),
        lines,
        stdout,
    }
}

fn run(root: &Path, base: &str, extra: &[&str]) -> Run {
    finish(ensure(root, base, extra).output().unwrap())
}

fn assert_result(run: &Run, checkpoint: &Checkpoint) -> Json {
    assert!(run.ok, "{}", run.stdout);
    let result = run.last().clone();
    assert_eq!(text(&result, "event"), "ensure.result", "{}", run.stdout);
    assert_eq!(text(&result, "manifest"), checkpoint.manifest.id());
    assert_eq!(text(&result, "model"), "acme/model");
    assert_eq!(text(&result, "lane"), "public");
    assert_eq!(num(&result, "bytes_total"), checkpoint.total());
    result
}

fn refused(run: &Run) -> (String, bool) {
    assert!(!run.ok, "{}", run.stdout);
    let last = run.last();
    assert_eq!(text(last, "event"), "ensure.refused", "{}", run.stdout);
    (text(last, "code"), flag(last, "resumable"))
}

// ---------------------------------------------------------------- the proofs

#[test]
fn a_cold_ensure_reports_and_lands_and_a_warm_one_moves_nothing() {
    let dir = temporary("cold-warm");
    let root = store(&dir);
    let checkpoint = Arc::new(checkpoint(3, 64 << 10, 1));
    let hub = hub(vec![Arc::clone(&checkpoint)], None, Duration::ZERO, serve_all());
    let total: u64 = checkpoint.bodies.values().map(|b| b.len() as u64).sum();

    let cold = run(&root, &hub.base, &[]);
    let result = assert_result(&cold, &checkpoint);
    assert_eq!(num(&result, "bytes_held"), 0);
    assert_eq!(num(&result, "bytes_fetched"), total);
    assert!(!flag(&result, "joined"));
    let progress = cold.events("ensure.progress");
    assert_eq!(text(progress[0], "phase"), "resolving");
    assert!(progress
        .iter()
        .all(|e| text(e, "step") == "load weights" && text(e, "model") == "acme/model"));
    let last = progress.last().unwrap();
    assert_eq!(
        (text(last, "phase").as_str(), num(last, "bytes_done")),
        ("fetching", total)
    );
    assert_eq!(num(last, "bytes_total"), total);
    assert!(checkpoint.bodies.keys().all(|hex| hub.gets(hex) == 1));

    let warm = run(&root, &hub.base, &[]);
    let result = assert_result(&warm, &checkpoint);
    assert_eq!(
        (num(&result, "bytes_held"), num(&result, "bytes_fetched")),
        (total, 0)
    );
    assert_eq!(
        hub.total_gets(),
        checkpoint.bodies.len(),
        "a warm ensure moved bytes"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A checkpoint the Store holds whole, named by its digest alone (`org/name@sha256:<hex>`, a
/// machine's exact model choice, lane or not), answers its own closure: a warm ensure asks the
/// hub nothing, not even the closure.
#[test]
fn a_held_checkpoint_named_by_digest_asks_the_hub_nothing() {
    let dir = temporary("held-digest");
    let root = store(&dir);
    let checkpoint = Arc::new(checkpoint(2, 16 << 10, 7));
    let hub = hub(vec![Arc::clone(&checkpoint)], None, Duration::ZERO, serve_all());
    let refspec = format!("acme/model@{}", checkpoint.manifest.id());
    let by_digest = |lane: &str| {
        let mut command = Command::new(TFS);
        command
            .args(["ensure", root.to_str().unwrap(), &refspec, "--hub", &hub.base])
            .args(["--lane", lane, "--sample-seconds", "0.05"])
            .env("TFS_CREDENTIAL", "");
        finish(command.output().unwrap())
    };
    let cold = by_digest("");
    assert!(cold.ok, "{}", cold.stdout);
    assert_eq!(hub.gets("closure"), 1);
    for lane in ["", "fp8"] {
        let warm = by_digest(lane);
        assert!(warm.ok, "{}", warm.stdout);
        assert_eq!(hub.gets("closure"), 1, "a held checkpoint asked the hub (lane {lane:?})");
    }
    assert!(checkpoint.bodies.keys().all(|hex| hub.gets(hex) == 1));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_ranged_get_streams_exact_bytes_under_a_lease() {
    let dir = temporary("get-range");
    let root = store(&dir);
    let checkpoint = Arc::new(checkpoint(1, 256 << 10, 2));
    let hub = hub(vec![Arc::clone(&checkpoint)], None, Duration::ZERO, serve_all());
    assert_result(&run(&root, &hub.base, &[]), &checkpoint);
    let blob = &checkpoint.blobs[0];
    let body = &checkpoint.bodies[&blob.sha256];
    let get = |range: &str| {
        Command::new(TFS)
            .args([
                "get",
                root.to_str().unwrap(),
                &blob.id(),
                "--range",
                range,
                "--out",
                "-",
            ])
            .output()
            .unwrap()
    };
    let out = get("5:100000");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, body[5..100_005]);
    assert_eq!(get("200000:").stdout, body[200_000..]);
    let whole = Command::new(TFS)
        .args(["get", root.to_str().unwrap(), &blob.id(), "--out", "-"])
        .output()
        .unwrap();
    assert_eq!(&whole.stdout, body);
    let beyond = get(&format!("{}:1", body.len()));
    assert!(!beyond.status.success() && beyond.stdout.is_empty());
    assert!(String::from_utf8_lossy(&beyond.stderr).contains("RANGE_BOUNDS"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn concurrent_ensures_of_one_manifest_share_one_fetch() {
    let dir = temporary("single-flight");
    let root = store(&dir);
    let checkpoint = Arc::new(checkpoint(4, 64 << 10, 3));
    // Paced so the leader is still fetching when the follower asks.
    let hub = hub(vec![Arc::clone(&checkpoint)], None, Duration::from_millis(250), serve_all());
    let mut leader = ensure(&root, &hub.base, &[])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(leader.stdout.take().unwrap()).lines();
    let mut seen = Vec::new();
    for line in lines.by_ref() {
        let line = line.unwrap();
        let fetching = parse(&line);
        seen.push(line);
        if field(&fetching, "phase") == Some(&Json::Str("fetching".into())) {
            break;
        }
    }
    let follower = run(&root, &hub.base, &[]);
    seen.extend(lines.map(|l| l.unwrap()));
    assert!(leader.wait().unwrap().success(), "{}", seen.join("\n"));

    let result = assert_result(&follower, &checkpoint);
    assert!(flag(&result, "joined"), "{}", follower.stdout);
    assert_eq!(num(&result, "bytes_fetched"), 0);
    assert!(follower
        .events("ensure.progress")
        .iter()
        .any(|e| text(e, "phase") == "waiting"));
    assert!(
        checkpoint.bodies.keys().all(|hex| hub.gets(hex) == 1),
        "an object was fetched twice: {:?}",
        hub.gets.lock().unwrap()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_attempt_that_landed_objects_is_retried_and_one_that_landed_none_is_stalled() {
    let dir = temporary("retry");
    let checkpoint = Arc::new(checkpoint(3, 64 << 10, 4));
    let stubborn = checkpoint.blobs[0].sha256.clone();
    // One object's first four answers die mid-body. Nothing counts asks, so the pull asks
    // it again while it lands bytes, and the first attempt brings everything home.
    let rule: Rule = Arc::new(move |hex, asked| {
        if hex == stubborn && asked <= 4 {
            Answer::Truncate
        } else {
            Answer::Body
        }
    });
    let hub_a = hub(vec![Arc::clone(&checkpoint)], None, Duration::ZERO, rule);
    let recovered = run(&store(&dir), &hub_a.base, &[]);
    assert_result(&recovered, &checkpoint);

    // An origin that never finishes an answer: asked again with URLs minted afresh, and
    // with nothing else moving, refused as stalled.
    let never: Rule = Arc::new(|_, _| Answer::Truncate);
    let hub_b = hub(vec![Arc::clone(&checkpoint)], None, Duration::ZERO, never);
    let root = dir.join("stalled");
    Store::init(&root).unwrap();
    let stalled = run(&root, &hub_b.base, &[]);
    assert_eq!(
        refused(&stalled),
        ("STALLED".to_string(), true),
        "{}",
        stalled.stdout
    );
    assert!(hub_b.gets("presign") >= 2, "never minted afresh");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_untrusted_hub_is_terminal_and_its_named_ca_admits_it() {
    let dir = temporary("tls");
    let root = store(&dir);
    let (config, pem) = tls_config();
    let checkpoint = Arc::new(checkpoint(2, 16 << 10, 5));
    let hub = hub(vec![Arc::clone(&checkpoint)], Some(config), Duration::ZERO, serve_all());

    let untrusted = run(&root, &hub.base, &[]);
    assert_eq!(refused(&untrusted), ("TLS_UNTRUSTED".to_string(), false));
    let ca = dir.join("hub-ca.pem");
    std::fs::write(&ca, pem).unwrap();
    assert_result(
        &run(&root, &hub.base, &["--ca-file", ca.to_str().unwrap()]),
        &checkpoint,
    );
    let garbage = dir.join("garbage.pem");
    std::fs::write(&garbage, "not a certificate").unwrap();
    let bad = run(&root, &hub.base, &["--ca-file", garbage.to_str().unwrap()]);
    assert_eq!(refused(&bad).0, "UNKNOWN_FORMAT");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The library path, in one process: a shared client that already refused a hub trusts it
/// once its CA is added, with no new client and no restart.
#[test]
fn trust_added_after_a_refusal_reaches_the_same_process() {
    use tensorfs_core::ensure::{self, Event, Phase, Request};
    use tensorfs_core::fetch::Plain;
    use tensorfs_core::transport::{self, Anonymous, SourcePolicy};
    let dir = temporary("trust-in-process");
    let store = Store::init(&dir.join("store")).unwrap();
    let (config, pem) = tls_config();
    let checkpoint = Arc::new(checkpoint(2, 16 << 10, 7));
    let hub = hub(vec![Arc::clone(&checkpoint)], Some(config), Duration::ZERO, serve_all());
    let access = Arc::new(Plain { credential: Anonymous, policy: SourcePolicy::default() });
    let events = Mutex::new(Vec::new());
    let on_event = |event: &Event| events.lock().unwrap().push(event.clone());
    let mut request = Request::new(&store, &hub.base, "acme/model@r1", access);
    request.lane = "public";
    request.step = "warm";
    request.on_event = Some(&on_event);
    let refused = ensure::ensure(&request).unwrap_err();
    assert_eq!(refused.code, tensorfs_core::err::Code::TLS_UNTRUSTED);
    assert!(!ensure::resumable(refused.code));
    transport::trust_roots(pem.as_bytes()).unwrap();
    transport::trust_roots(pem.as_bytes()).unwrap();
    let ensured = ensure::ensure(&request).unwrap();
    assert_eq!(ensured.manifest, checkpoint.manifest);
    assert_eq!(ensured.bytes_fetched, ensured.bytes_total);
    let events = events.into_inner().unwrap();
    assert!(events
        .iter()
        .all(|e| e.step == "warm" && e.model == "acme/model"));
    assert_eq!(events.last().unwrap().phase, Phase::Fetching);
    assert_eq!(events.last().unwrap().bytes_done, ensured.bytes_total);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_closure_the_disk_cannot_hold_is_refused_before_a_byte_moves() {
    let dir = temporary("capacity");
    let root = store(&dir);
    let mut huge = checkpoint(1, 16 << 10, 6);
    // The hub declares a petabyte object: no disk this test runs on admits it.
    let petabyte = ObjectRef {
        sha256: "ab".repeat(32),
        length: 1 << 50,
    };
    huge.objects.push(petabyte);
    huge.objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    let hub = hub(vec![Arc::new(huge)], None, Duration::ZERO, serve_all());
    let full = run(&root, &hub.base, &[]);
    assert_eq!(
        refused(&full),
        ("CAPACITY_EXHAUSTED".to_string(), true),
        "{}",
        full.stdout
    );
    let detail = text(full.last(), "detail");
    assert!(
        detail.contains("this download needs 11258999068") && detail.contains("GC freed"),
        "{detail}"
    );
    assert_eq!(hub.total_gets(), 0, "admission let bytes move");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn capabilities_name_ensure() {
    for name in ["ensure/1", "get-range/1"] {
        assert!(tensorfs_core::CAPABILITIES.contains(&name), "{name}");
    }
}

#[test]
fn a_pressure_pass_reports_the_disk_and_keeps_what_it_is_told() {
    let dir = temporary("pressure");
    let root = store(&dir);
    let keep = ObjectRef::of(b"serving").id();
    let relief = tensorfs_core::ensure::relieve(&Store::open(&root).unwrap(), &[keep]).unwrap();
    assert!(relief.capacity_bytes >= relief.available_bytes);
    // Low is at or under the reserve, and a pass over a disk that is not low collects nothing.
    let disk = tensorfs_core::ensure::Disk::measure(&root).unwrap();
    assert!(relief.pressure || relief.available_bytes > disk.reserve());
    if !relief.pressure {
        assert_eq!(relief.collected_bytes, 0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}


/// A pod's container caps its tasks (a vast RTX 5060 Ti: `pids.max` 1280). Requests in flight
/// are tasks, not threads: a pull keeps more of them in flight than the cap allows threads,
/// and lands. On 0.9.0 a refused stream thread panicked past the pull's sampler and `tfs
/// ensure` never returned, holding the manifest's flight (TensorD 0.5.30 prepare wedge).
#[test]
fn an_ensure_whose_stream_threads_the_container_refuses_lands() {
    const TASKS: u64 = 48;
    let cap = format!("TasksMax={TASKS}");
    let scope = ["systemd-run", "--user", "--scope", "-q", "-p", cap.as_str()];
    let usable = Command::new(scope[0])
        .args(&scope[1..])
        .arg("true")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !usable {
        eprintln!("SKIPPED: no systemd user manager to create a task-limited cgroup");
        return;
    }
    let dir = temporary("tasks");
    let root = store(&dir);
    // 251 objects: the pull asks for a stream for each.
    let checkpoint = Arc::new(checkpoint(251, 4096, 9));
    let hub = hub(vec![Arc::clone(&checkpoint)], None, Duration::ZERO, serve_all());
    let inner = ensure(&root, &hub.base, &[]);
    let out = dir.join("stdout");
    let mut child = Command::new(scope[0])
        .args(&scope[1..])
        .arg(inner.get_program())
        .args(inner.get_args())
        .env("TFS_CREDENTIAL", "")
        .stdout(std::fs::File::create(&out).unwrap())
        .spawn()
        .unwrap();
    // Test harness bound on a wedged pull; the product has no such limit.
    let until = std::time::Instant::now() + Duration::from_secs(90);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if std::time::Instant::now() > until {
            let _ = child.kill();
            panic!("the ensure never ended under {cap}");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let stdout = std::fs::read_to_string(&out).unwrap();
    let run = Run { ok: status.success(), lines: stdout.lines().map(parse).collect(), stdout };
    assert_result(&run, &checkpoint);
    let log = std::fs::read_to_string(root.join("logs/transport.log")).unwrap();
    let walk = log.lines().rfind(|line| line.contains(" walk ")).unwrap();
    let streams: u64 = walk
        .split_whitespace()
        .find_map(|field| field.strip_prefix("streams="))
        .unwrap()
        .parse()
        .unwrap();
    assert!(streams > TASKS, "more requests in flight than the cap allows threads: {walk}");
    let _ = std::fs::remove_dir_all(&dir);
}
