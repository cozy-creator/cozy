//! The real `tensord serve` process behind its HTTPS machine API, called as Calcifer calls it.
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use tensord::journal::Journal;
use ed25519_dalek::SigningKey;
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

#[path = "common/hub_oauth.rs"]
mod hub_oauth;
#[path = "common/names_hub.rs"]
mod names_hub;

struct Machine {
    child: Child,
    root: PathBuf,
    address: String,
}

const SIGNER: [u8; 32] = [33; 32];
/// A second admitted key: another submitter on the same machine.
const OTHER: [u8; 32] = [44; 32];
const WORKER: &str = "surface-test";

impl Machine {
    async fn start() -> Self {
        Self::start_with(|_, _| ()).await
    }
    /// `prepare` sees the state root and the test actor's journal id before the machine starts.
    async fn start_with(prepare: impl FnOnce(&Path, &str)) -> Self {
        Self::start_args(prepare, &[]).await
    }
    /// As `start_with`, with more `serve` arguments.
    async fn start_args(prepare: impl FnOnce(&Path, &str), extra: &[std::ffi::OsString]) -> Self {
        let root = std::env::temp_dir().join(format!("cm-surface-{}", uuid::Uuid::new_v4()));
        let config = root.join("config");
        fs::create_dir_all(&config).unwrap();
        #[cfg(unix)]
        fs::set_permissions(&root, std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
        let keys = [SIGNER, OTHER].map(|key| {
            URL_SAFE_NO_PAD.encode(SigningKey::from_bytes(&key).verifying_key().to_bytes())
        });
        let write = |name: &str, value: serde_json::Value| {
            let path = config.join(name);
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            #[cfg(unix)]
            fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
        };
        write(
            "keys.json",
            serde_json::json!({ "keys": keys }),
        );
        write(
            "readiness.json",
            serde_json::json!({"key_b64url":URL_SAFE_NO_PAD.encode([7u8; 32])}),
        );
        write(
            "machine.json",
            serde_json::json!({"worker_id":WORKER,"identity_directory":"identity",
                "authorized_keys_file":"keys.json","readiness_hmac_key_file":"readiness.json"}),
        );
        prepare(&root.join("state"), &actor());
        let (child, address) = launch(&root, extra);
        Self {
            child,
            root,
            address,
        }
    }
    /// End the exact process this test started; its state stays for inspection.
    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
    fn store(&self) -> PathBuf {
        self.root.join("state/tensorfs")
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        self.stop();
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn actor() -> String {
    actor_of(&SIGNER)
}
fn actor_of(key: &[u8; 32]) -> String {
    tensorfs_core::sha256::hex(&SigningKey::from_bytes(key).verifying_key().to_bytes())
}

fn launch(root: &Path, extra: &[std::ffi::OsString]) -> (Child, String) {
    launch_in(root, extra, &[])
}

/// `launch` under `scope`, a command that runs the machine as its arguments
/// (`systemd-run --user --scope …`).
fn launch_in(root: &Path, extra: &[std::ffi::OsString], scope: &[&str]) -> (Child, String) {
    let state = root.join("state");
    let _ = fs::remove_file(state.join("api-ready.json"));
    let tensord = env!("CARGO_BIN_EXE_tensord");
    let (program, prefix) = match scope {
        [] => (tensord, &[][..]),
        [program, prefix @ ..] => (*program, prefix),
    };
    let mut child = Command::new(program)
        .args(prefix)
        .args((!scope.is_empty()).then_some(tensord))
        .args(["serve", "--state"])
        .arg(&state)
        .arg("--machine-config")
        .arg(root.join("config/machine.json"))
        .args(["--listen", "127.0.0.1:0"])
        .args(extra)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    // Test harness bound on a broken start; the product has no such limit.
    let until = Instant::now() + Duration::from_secs(60);
    let ready = loop {
        if let Ok(bytes) = fs::read(state.join("api-ready.json")) {
            break serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
        }
        if Instant::now() >= until {
            let _ = child.kill(); // never leave a started machine behind
            panic!("machine did not publish api-ready.json");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    (child, ready["address"].as_str().unwrap().to_owned())
}

fn hold_classifier(state: &Path, actor: &str) {
    const INSTALL: &str = r#"
import json, subprocess, sys
from pathlib import Path
from tensord_client.packages import install
fixture, out = Path(sys.argv[1]), Path(sys.argv[2])
subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out / "client"), "."], check=True, capture_output=True)
generation = install(fixture, out / "generations", next((out / "client").glob("*.whl")), python="3.12")
print(json.dumps({"identity": generation.identity}))
"#;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::create_dir_all(state).unwrap();
    let output = Command::new("uv")
        .current_dir(repo)
        .args([
            "run", "--locked", "--extra", "test", "python", "-c", INSTALL,
        ])
        .arg(repo.join("tests/fixtures/cpu_classifier"))
        .arg(state)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = String::from_utf8(output.stdout).unwrap();
    let value: serde_json::Value = serde_json::from_str(line.lines().last().unwrap()).unwrap();
    let identity = value["identity"].as_str().unwrap();
    let generation: serde_json::Value = serde_json::from_slice(
        &fs::read(
            state
                .join("generations")
                .join(identity)
                .join("generation.json"),
        )
        .unwrap(),
    )
    .unwrap();
    Journal::open(&state.join("execution"))
        .unwrap()
        .bind_installation(tensord::journal::Installation {
            actor: actor.into(),
            alias: "fixture".into(),
            generation: identity.into(),
            package: generation["package"].as_str().unwrap().into(),
            release: generation["version"].as_str().unwrap().into(),
            interface: serde_json::to_vec(&generation["interface"]).unwrap(),
            hub: String::new(),
        })
        .unwrap();
}

mod v1_api {
    use super::*;
    use tensord::api::{
        capability::{mint, Grant, MACHINE},
        v1,
    };
    use tensord::api::v1::client::{Auth, Error as ClientError, Machine as Client};
    use tokio_stream::{Stream, StreamExt};

    fn cap(scope: Grant) -> String {
        cap_by(&SIGNER, scope)
    }
    fn cap_by(key: &[u8; 32], scope: Grant) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        mint(
            &SigningKey::from_bytes(key),
            Grant {
                machine: WORKER.into(),
                expires: now + 600,
                ..scope
            },
        )
    }
    /// A client of `machine`'s published endpoint, pinned to its leaf.
    fn client(machine: &Machine) -> Client {
        let ready: serde_json::Value =
            serde_json::from_slice(&fs::read(machine.root.join("state/api-ready.json")).unwrap())
                .unwrap();
        let pem = ready["cert_pem"].as_str().unwrap();
        let leaf = rustls_pemfile::certs(&mut pem.as_bytes()).next().unwrap().unwrap();
        Client::new(&machine.address, tensorfs_core::sha256::digest(&leaf), Auth::None).unwrap()
    }
    fn by(client: &Client, cap: &str) -> Client {
        client.with_auth(Auth::Token(cap.into()))
    }
    /// One Status frame as `token`'s holder sees it.
    async fn status_frame(client: &Client, token: &str) -> v1::StatusFrame {
        by(client, token).status().await.unwrap()
    }
    /// Status as `token`'s holder watches it.
    async fn watch(client: &Client, token: &str) -> Events<v1::StatusFrame> {
        Box::pin(by(client, token).watch().await.unwrap())
    }

    type Events<T = v1::RunEvent> = std::pin::Pin<Box<dyn Stream<Item = Result<T, ClientError>> + Send>>;
    /// A submit (with a spec) and an attach after `after`, as one call.
    struct RunRequest {
        id: String,
        after: u64,
        spec: Option<v1::RunSpec>,
    }
    /// Submits `request.spec` when it names one, then follows the run's events.
    async fn run_events(client: &Client, cap: &str, request: RunRequest) -> Result<Events, ClientError> {
        let client = by(client, cap);
        if let Some(spec) = request.spec {
            client.submit(&v1::RunRequest { id: request.id.clone(), spec }).await?;
        }
        Ok(Box::pin(client.events(&request.id, request.after).await?))
    }
    /// `request` as `cap`'s holder makes it.
    fn authorized(request: RunRequest, cap: &str) -> (RunRequest, String) {
        (request, cap.into())
    }
    async fn run_as(client: &Client, (request, cap): (RunRequest, String)) -> Result<Events, ClientError> {
        run_events(client, &cap, request).await
    }
    async fn cancel(client: &Client, cap: &str, id: &str) -> Result<v1::RunState, ClientError> {
        by(client, cap).cancel(id).await
    }
    async fn collect(mut stream: Events) -> Result<Vec<v1::RunEvent>, ClientError> {
        let mut events = vec![];
        while let Some(event) = stream.next().await.transpose()? {
            events.push(event);
        }
        Ok(events)
    }
    /// A result or measurements as JSON (null when absent).
    fn json(raw: &Option<v1::Json>) -> serde_json::Value {
        serde_json::from_str(raw.as_ref().map_or("null", |raw| raw.get())).unwrap()
    }
    fn raw(value: &serde_json::Value) -> Option<v1::Json> {
        v1::Json::parse(value.to_string()).ok()
    }
    fn spec(iterations: u64) -> v1::RunSpec {
        v1::RunSpec {
            source: Some(v1::Source::Installation("fixture".into())),
            entrypoint: "classify".into(),
            payload: raw(&serde_json::json!({
                "samples": [[5.1, 3.5, 1.4, 0.2], [6.0, 2.7, 5.1, 1.6]],
                "iterations": iterations,
            })),
            ..Default::default()
        }
    }
    /// What a test reads: one output (from `offset`, `length` bytes, refused unless at `if_rev`)
    /// or a machine log's newest `tail` lines.
    #[derive(Clone, Default)]
    struct ReadRequest {
        target: Option<Target>,
        offset: u64,
        length: u64,
        if_rev: u64,
        tail: u64,
    }
    #[derive(Clone)]
    enum Target {
        Output(v1::OutputTarget),
        Log(String),
    }
    /// An output's revision and whole length, and the digest of the bytes read.
    #[derive(Debug)]
    struct Meta {
        rev: u64,
        length: u64,
        digest: String,
    }
    async fn read(client: &Client, cap: &str, request: ReadRequest) -> Result<(Meta, Vec<u8>), ClientError> {
        let client = by(client, cap);
        let target = match request.target.expect("a read names a target") {
            Target::Log(name) => return Ok((Meta { rev: 0, length: 0, digest: String::new() }, client.log(&name, request.tail).await?.to_vec())),
            Target::Output(target) => target,
        };
        let range = match (request.offset, request.length) {
            (0, 0) => (0, 0),
            (offset, 0) => (offset, u64::MAX),
            (offset, length) => (offset, offset + length),
        };
        let output = client.bulk()?.get(&target, range, request.if_rev).await?;
        let mut body = output.body;
        let mut data = vec![];
        while let Some(chunk) = body.next().await {
            data.extend_from_slice(&chunk?);
        }
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&data));
        Ok((Meta { rev: output.rev, length: output.length, digest }, data))
    }

    /// Read returns the store's transport and memory logs, each's rotated part first, or its
    /// newest lines; only a machine capability reads them.
    #[tokio::test]
    async fn read_returns_the_stores_logs_with_rotation_and_newest_lines() {
        let machine = Machine::start().await;
        let client = client(&machine);
        let owner = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let log = |name: &str, tail| ReadRequest {
            target: Some(Target::Log(name.into())),
            tail,
            ..Default::default()
        };
        let logs = machine.store().join("logs");
        let rotated = "1.000 pull a 1 old\n";
        let current: String = (0..3000).map(|n| format!("{n}.000 hedge object-{n} 4096 grant\n")).collect();
        let written = format!("{rotated}{current}").into_bytes();
        // Nothing here fetches, so the transport log is the test's alone: its reads are exact.
        assert!(read(&client, &owner, log("tensorfs-transport", 0)).await.unwrap().1.is_empty());
        fs::create_dir_all(&logs).unwrap();
        fs::write(logs.join("transport.log.1"), rotated).unwrap();
        fs::write(logs.join("transport.log"), &current).unwrap();
        let (_, whole) = read(&client, &owner, log("tensorfs-transport", 0)).await.unwrap();
        assert_eq!(whole, written);
        let (_, newest) = read(&client, &owner, log("tensorfs-transport", 2)).await.unwrap();
        assert_eq!(newest, b"2998.000 hedge object-2998 4096 grant\n2999.000 hedge object-2999 4096 grant\n");
        let (_, all) = read(&client, &owner, log("tensorfs-transport", 3001)).await.unwrap();
        assert_eq!(all, whole, "lines reach back into the rotated part");
        // The memory log is the machine's own: a loaded host's pressure watcher may append a line
        // to it at any moment, so it reads as what was written, then whole lines.
        fs::write(logs.join("memory.log.1"), rotated).unwrap();
        fs::write(logs.join("memory.log"), &current).unwrap();
        let (_, whole) = read(&client, &owner, log("memory", 0)).await.unwrap();
        assert!(whole.starts_with(&written) && whole.ends_with(b"\n"), "{}", String::from_utf8_lossy(&whole[written.len().min(whole.len())..]));
        let run = cap(Grant { run: "1".into(), ..Default::default() });
        for name in ["tensorfs-transport", "memory"] {
            assert_eq!(read(&client, &run, log(name, 0)).await.unwrap_err().status(), Some(403));
        }
        let unknown = read(&client, &owner, log("other", 0)).await.unwrap_err();
        assert_eq!(unknown.status(), Some(404));
    }

    /// A follow sends the kept lines, then each line as it is written, across two rotations with
    /// none missed or repeated; one resumed from a batch's place goes on with the next line.
    #[tokio::test]
    async fn a_followed_log_crosses_rotations_and_resumes_where_it_left() {
        let machine = Machine::start().await;
        let (client, owner) = (client(&machine), cap(Grant { action: MACHINE.into(), ..Default::default() }));
        let follower = by(&client, &owner);
        let logs = machine.store().join("logs");
        fs::create_dir_all(&logs).unwrap();
        let line = |n: usize| format!("{n}.000 pull object-{n} 4096\n");
        let write = |n: usize| {
            use std::io::Write as _;
            if n > 0 && n.is_multiple_of(50) {
                fs::rename(logs.join("transport.log"), logs.join("transport.log.1")).unwrap();
            }
            let mut log = fs::OpenOptions::new().create(true).append(true).open(logs.join("transport.log")).unwrap();
            log.write_all(line(n).as_bytes()).unwrap();
        };
        async fn until(lines: &mut (impl Stream<Item = Result<v1::LogLines, ClientError>> + Unpin), count: usize) -> (String, String) {
            let (mut text, mut after) = (String::new(), String::new());
            while text.lines().count() < count {
                let batch = tokio::time::timeout(Duration::from_secs(30), lines.next()).await;
                let batch = batch.expect("lines arrive as written").expect("the follow stays open").unwrap();
                text.push_str(&batch.text);
                after = batch.after;
            }
            (text, after)
        }

        write(0);
        let mut follow = Box::pin(follower.follow_log("tensorfs-transport", 0, None).await.unwrap());
        (1..120).for_each(write);
        let (text, after) = until(&mut follow, 120).await;
        assert_eq!(text, (0..120).map(line).collect::<String>());
        drop(follow);

        (120..125).for_each(write);
        let mut resumed = Box::pin(follower.follow_log("tensorfs-transport", 0, Some(&after)).await.unwrap());
        assert_eq!(until(&mut resumed, 5).await.0, (120..125).map(line).collect::<String>());
    }

    /// An upload followed from before its first byte says what the machine holds as its parts
    /// land, after each MiB more, and ends at the whole object; a stored object answers at once.
    #[tokio::test]
    async fn an_upload_followed_says_what_lands_and_ends_whole() {
        let machine = Machine::start().await;
        let (client, owner) = (client(&machine), cap(Grant { action: MACHINE.into(), ..Default::default() }));
        let body: Vec<u8> = (0..4u32 << 20).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        let (digest, length) = (format!("sha256:{}", tensorfs_core::sha256::hex_digest(&body)), body.len() as u64);
        let follower = by(&client, &owner);
        let mut held = Box::pin(follower.follow_held(&digest, length).await.unwrap());
        assert_eq!(held.next().await.unwrap().unwrap(), 0, "nothing held before the first part");
        let mut seen = vec![];
        for part in body.chunks(1 << 20).enumerate() {
            let (index, data) = part;
            write(&client, &owner, &digest, length, (index << 20) as u64, data).await.unwrap();
            let now = tokio::time::timeout(Duration::from_secs(30), held.next()).await;
            seen.push(now.expect("held follows each part").unwrap().unwrap());
        }
        assert_eq!(seen, [1 << 20, 2 << 20, 3 << 20, 4 << 20], "after each MiB, to the whole");
        assert!(held.next().await.is_none(), "the stream ends at the whole object");
        let mut stored = Box::pin(follower.follow_held(&digest, length).await.unwrap());
        assert_eq!(stored.next().await.unwrap().unwrap(), length, "a stored object answers whole");
        assert!(stored.next().await.is_none());
    }

    #[tokio::test]
    async fn run_streams_a_real_run_to_its_outcome_and_read_returns_its_output() {
        let machine = Machine::start_with(hold_classifier).await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let request = |spec: Option<v1::RunSpec>, after| RunRequest {
            id: "run-1".into(),
            after,
            spec,
        };
        let events = collect(
            run_events(&client, &all, request(Some(spec(3)), 0))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        // The first frame is the run's state (a snapshot, sequence 0); the log ends at the outcome.
        let v1::Event::State(first) = &events[0].event else {
            panic!("{events:?}")
        };
        assert_eq!((events[0].sequence, first.id.as_str()), (0, "run-1"));
        let number = first.number;
        assert!(number > 0);
        let v1::Event::Outcome(outcome) = &events.last().unwrap().event else {
            panic!("{events:?}")
        };
        assert_eq!(outcome.status, "succeeded", "{outcome:?}");
        // Its own time running the callable, on the machine's clock.
        assert!(outcome.execution_ms > 0, "{outcome:?}");
        let running = events
            .iter()
            .find(|e| {
                matches!(&e.event, v1::Event::State(s) if s.state == "running")
            })
            .expect("the run is seen running");
        let finished = events.last().unwrap().at_ms;
        assert!(running.at_ms > 0 && running.at_ms <= finished, "{events:?}");
        assert!(outcome.execution_ms <= (finished - running.at_ms) as u64 + 1, "{outcome:?}");
        let result: serde_json::Value = json(&outcome.result);
        assert_eq!(result["predictions"], serde_json::json!([0, 2]));
        assert_eq!(result["iterations"], 3);
        let report = outcome
            .outputs
            .iter()
            .find(|p| p.output == "report")
            .unwrap();
        assert_eq!((report.index, report.rev), (0, 1));
        let logged: Vec<u64> = events.iter().skip(1).map(|e| e.sequence).collect();
        assert!(logged.windows(2).all(|w| w[0] < w[1]), "{logged:?}");

        // Read: the output's bytes, digest and revision; an offset resumes.
        let target = |offset| ReadRequest {
            target: Some(Target::Output(v1::OutputTarget {
                run: "run-1".into(),
                output: "report".into(),
                index: 0,
                ..Default::default()
            })),
            offset,
            ..Default::default()
        };
        let (meta, bytes) = read(&client, &all, target(0)).await.unwrap();
        assert_eq!((meta.rev, meta.length), (1, bytes.len() as u64));
        assert_eq!(meta.length, report.length);
        let digest = format!(
            "sha256:{}",
            tensorfs_core::sha256::hex(&tensorfs_core::sha256::digest(&bytes))
        );
        assert_eq!(
            (meta.digest.as_str(), report.digest.as_str()),
            (digest.as_str(), digest.as_str())
        );
        let report_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(report_json["predictions"], serde_json::json!([0, 2]));
        let (_, tail) = read(&client, &all, target(5)).await.unwrap();
        assert_eq!(tail, bytes[5..]);
        let stale = read(
            &client,
            &all,
            ReadRequest {
                if_rev: 7,
                ..target(0)
            },
        )
        .await
        .unwrap_err();
        assert_eq!(stale.status(), Some(412));

        // The same id and spec again is the same run; attach without a spec replays the log.
        let again = collect(
            run_events(&client, &all, request(Some(spec(3)), 0))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let v1::Event::State(state) = &again[0].event else {
            panic!()
        };
        assert_eq!(state.number, number);
        let tail_log = collect(
            run_events(&client, &all, request(None, events[events.len() - 2].sequence))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(
            tail_log.len(),
            2,
            "a snapshot, then only the outcome after the cursor"
        );

        // A run-scope cap may attach to and read its run, and nothing else.
        let scoped = cap(Grant {
            run: "run-1".into(),
            ..Default::default()
        });
        assert!(collect(
            run_events(&client, &scoped, request(None, 0))
                .await
                .unwrap()
        )
        .await
        .is_ok());
        assert!(read(&client, &scoped, target(0)).await.is_ok());
        let submit = run_events(&client, &scoped, RunRequest {
                    id: "run-2".into(),
                    after: 0,
                    spec: Some(spec(1)),
                })
            .await
            .err()
            .unwrap();
        assert_eq!(submit.status(), Some(403));
        let cancel = cancel(&client, &scoped, "run-1")
            .await
            .unwrap_err();
        assert_eq!(cancel.status(), Some(403));
        let bare = client.events("run-1", 0).await.err().unwrap();
        assert_eq!(bare.status(), Some(401));
    }

    #[tokio::test]
    async fn submitters_of_one_package_see_only_their_own_runs() {
        let machine = Machine::start_with(|state, actor| {
            hold_classifier(state, actor);
            // The other submitter holds the same package: a row of its own, the same generation.
            let mut journal = Journal::open(&state.join("execution")).unwrap();
            let held = journal.installations(actor).unwrap().remove(0);
            journal
                .bind_installation(tensord::journal::Installation {
                    actor: actor_of(&OTHER),
                    ..held
                })
                .unwrap();
        })
        .await;
        let client = client(&machine);
        let whole = |key| {
            cap_by(
                key,
                Grant {
                    action: MACHINE.into(),
                    ..Default::default()
                },
            )
        };
        let (ours, theirs) = (whole(&SIGNER), whole(&OTHER));
        let request = |id: &str, spec| RunRequest {
            id: id.into(),
            after: 0,
            spec,
        };
        // Both name their run "run-1": an id is its submitter's own.
        let mut settled = vec![];
        for (cap, iterations) in [(&ours, 3), (&theirs, 4)] {
            let run = request("run-1", Some(spec(iterations)));
            let events = collect(run_events(&client, cap, run).await.unwrap())
                .await
                .unwrap();
            let v1::Event::State(state) = &events[0].event else {
                panic!("{events:?}")
            };
            let v1::Event::Outcome(outcome) = &events.last().unwrap().event else {
                panic!("{events:?}")
            };
            assert_eq!(outcome.status, "succeeded", "{outcome:?}");
            let result: serde_json::Value = json(&outcome.result);
            assert_eq!(result["iterations"], iterations);
            settled.push(state.number);
        }
        let (our_run, their_run) = (settled[0], settled[1]);
        assert_ne!(our_run, their_run, "two runs, not one shared record");
        // Reading "run-1" answers each submitter with its own run's output.
        let target = ReadRequest {
            target: Some(Target::Output(v1::OutputTarget {
                run: "run-1".into(),
                output: "report".into(),
                index: 0,
                ..Default::default()
            })),
            ..Default::default()
        };
        for (cap, iterations) in [(&ours, 3), (&theirs, 4)] {
            let (_, bytes) = read(&client, cap, target.clone()).await.unwrap();
            let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(report["iterations"], iterations);
        }
        // A run only one of them submitted does not exist for the other: attach, read, control.
        let only_ours = request("only-ours", Some(spec(2)));
        collect(
            run_events(&client, &ours, only_ours)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let attach = run_events(&client, &theirs, request("only-ours", None))
            .await;
        let attached = match attach {
            Ok(stream) => collect(stream).await.map(drop),
            Err(status) => Err(status),
        };
        assert_eq!(attached.unwrap_err().status(), Some(404));
        let other = ReadRequest {
            target: Some(Target::Output(v1::OutputTarget {
                run: "only-ours".into(),
                output: "report".into(),
                index: 0,
                ..Default::default()
            })),
            ..Default::default()
        };
        let refused = read(&client, &theirs, other).await.unwrap_err();
        assert_eq!(refused.status(), Some(404));
        // Their cancel of the id fences it in their own namespace (canceled before acceptance)
        // and never reaches our run, which stays succeeded.
        let fenced = cancel(&client, &theirs, "only-ours")
            .await
            .unwrap();
        assert_eq!(fenced.state, "canceled", "{fenced:?}");
        let ours_after = collect(
            run_events(&client, &ours, request("only-ours", None))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let v1::Event::State(state) = &ours_after[0].event else {
            panic!("{ours_after:?}")
        };
        assert_ne!(state.number, fenced.number, "their fence is a record of their own");
        assert_eq!(outcome(&ours_after).status, "succeeded", "{ours_after:?}");
    }

    #[tokio::test]
    async fn control_cancels_a_running_run_cooperatively() {
        let machine = Machine::start_with(hold_classifier).await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let mut stream = run_events(&client, &all, RunRequest {
                    id: "long".into(),
                    after: 0,
                    spec: Some(spec(50_000_000)),
                })
            .await
            .unwrap();
        // Cancel once the running run reports progress: it is executing package code.
        let mut running = false;
        loop {
            let event = stream
                .next().await.transpose()
                .unwrap()
                .expect("run ended before progress");
            match event.event {
                v1::Event::State(state) if state.state == "running" => {
                    running = true
                }
                v1::Event::Progress(_) if running => break,
                _ => {}
            }
        }
        cancel(&client, &all, "long")
            .await
            .unwrap();
        let rest = collect(stream).await.unwrap();
        let v1::Event::Outcome(outcome) = &rest.last().unwrap().event else {
            panic!("{rest:?}")
        };
        assert_eq!(outcome.status, "canceled", "{outcome:?}");
        assert_eq!(outcome.reason.as_ref().unwrap().code, "canceled");
    }

    #[tokio::test]
    async fn status_lists_the_callers_live_runs_and_held_environments() {
        let machine = Machine::start_with(hold_classifier).await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let request = RunRequest {
            id: "live".into(),
            after: 0,
            spec: Some(spec(50_000_000)),
        };
        let mut stream = run_events(&client, &all, request)
            .await
            .unwrap();
        // Progress once it runs (a run also reports progress while it prepares).
        let mut running = false;
        loop {
            match stream.next().await.transpose().unwrap().expect("run ended").event {
                v1::Event::State(state) if state.state == "running" => {
                    running = true
                }
                v1::Event::Progress(_) if running => break,
                _ => {}
            }
        }
        let mut status = watch(&client, &all).await;
        let frame = status.next().await.transpose().unwrap().unwrap();
        let live: Vec<_> = frame
            .runs
            .iter()
            .map(|r| (r.id.as_str(), r.state.as_str()))
            .collect();
        assert_eq!(live, [("live", "running")], "{frame:?}");
        let held: Vec<_> = frame
            .environments
            .iter()
            .map(|e| e.installation.as_str())
            .collect();
        assert_eq!(held, ["fixture"], "{frame:?}");
        cancel(&client, &all, "live").await.unwrap();
        collect(stream).await.unwrap();
        // The held Status stream sends the change: no live run remains.
        let until = Instant::now() + Duration::from_secs(30);
        loop {
            let next = tokio::time::timeout_at(until.into(), status.next())
                .await
                .expect("Status sent no frame without the run")
                .unwrap()
                .unwrap();
            if next.runs.is_empty() {
                break;
            }
        }
    }

    /// Lands `data` at `offset` of the object (`digest`, `length` bytes); nothing asks what is
    /// held. Answers the bytes held from its start.
    async fn write(
        client: &Client,
        cap: &str,
        digest: &str,
        length: u64,
        offset: u64,
        data: &[u8],
    ) -> Result<u64, ClientError> {
        let client = by(client, cap);
        let held = match data.is_empty() && length > 0 {
            true => client.held(digest).await?,
            false => client.bulk()?.put(digest, (offset, data.len() as u64, length), data.to_vec()).await?,
        };
        Ok(held.first().filter(|(start, _)| *start == 0).map_or(0, |(_, end)| *end))
    }

    /// A serve process that installs unpublished code (the client wheel built from this repo).
    async fn installing_machine() -> (Machine, PathBuf) {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tools = std::env::temp_dir().join(format!("cm-tools-{}", uuid::Uuid::new_v4()));
        assert!(Command::new("uv")
            .current_dir(repo)
            .args(["build", "--wheel", "--out-dir"])
            .arg(&tools)
            .status()
            .unwrap()
            .success());
        NAMED_RUNTIME.get_or_init(|| named_runtime_loads(&tools)).await;
        let machine = Machine::start_args(|_, _| (), &installing_args(&tools)).await;
        (machine, tools)
    }

    /// Once per test process: a named Runtime wheel (`COZY_TEST_RUNTIME_WHEEL`) is the one an
    /// installing machine's packages load, by the commit stamped in it, on a machine of its own so
    /// no test's machine holds the probe's package.
    static NAMED_RUNTIME: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
    async fn named_runtime_loads(tools: &Path) {
        let Some(wheel) = std::env::var_os("COZY_TEST_RUNTIME_WHEEL") else { return };
        let want = stamped(Path::new(&wheel));
        assert!(
            !want.is_empty() && want != "unknown",
            "{wheel:?} names no commit (COMMIT = {want:?}): a wheel without provenance proves nothing"
        );
        let machine = Machine::start_args(|_, _| (), &installing_args(tools)).await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let manifest = write_package(&client, &all, "cpu_bulk", "local/tensord-cpu-bulk").await;
        let built = call(&client, &all, &manifest, ("provenance", "provenance"), serde_json::json!({})).await;
        assert_eq!(built["commit"], want.as_str(), "the machine loaded another Runtime than {wheel:?}");
    }

    /// The TensorFS wheel a Runtime wheel accepts, from PyPI, into `dir`.
    const ACCEPTED_TENSORFS: &str = "import re, subprocess, sys, zipfile, pathlib
wheel, out = sys.argv[1], pathlib.Path(sys.argv[2])
archive = zipfile.ZipFile(wheel)
meta = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
req = next(l[15:].split(';')[0].replace(' ', '') for l in archive.read(meta).decode().splitlines()
           if re.match(r'Requires-Dist: tensorfs[^A-Za-z0-9._-]', l))
subprocess.run([sys.executable, '-m', 'pip', 'download', '-q', '--no-deps', '--only-binary=:all:', '--python-version', '3.12',
                '--platform', 'manylinux_2_28_x86_64', '--platform', 'manylinux_2_17_x86_64', '-d', str(out), req], check=True)
print(next(out.glob('tensorfs-*.whl')))
";

    /// The commit a Runtime wheel was built from: its `cozy_runtime._build_provenance.COMMIT`.
    fn stamped(wheel: &Path) -> String {
        let read = "import re, sys, zipfile\n\
            text = zipfile.ZipFile(sys.argv[1]).read('cozy_runtime/_build_provenance.py').decode()\n\
            print(re.search(r'^COMMIT = \"([^\"]*)\"', text, re.M).group(1))\n";
        let out = Command::new("python3").args(["-I", "-c", read]).arg(wheel).output().unwrap();
        assert!(out.status.success(), "reading {wheel:?}: {out:?}");
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    /// `serve`'s installer arguments: the test helper Python and the client wheel in `tools`.
    fn installing_args(tools: &Path) -> Vec<std::ffi::OsString> {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let wheel = fs::read_dir(tools)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "whl"))
            .unwrap();
        let helper = Command::new("uv")
            .current_dir(repo)
            .args([
                "run",
                "--locked",
                "--extra",
                "test",
                "python",
                "-c",
                "import sys; print(sys.executable)",
            ])
            .output()
            .unwrap();
        let helper = String::from_utf8(helper.stdout).unwrap().trim().to_string();
        let mut args: Vec<std::ffi::OsString> =
            vec!["--installer-python".into(), helper.into(), "--client-wheel".into(), wheel.into()];
        // A Runtime wheel not yet published, installed as this machine's own (a dev build), with
        // the TensorFS it accepts, as an update that names a Runtime brings.
        if let Some(runtime) = std::env::var_os("COZY_TEST_RUNTIME_WHEEL") {
            let tensorfs = std::env::var_os("COZY_TEST_TENSORFS_WHEEL").unwrap_or_else(|| {
                let out = Command::new("python3").args(["-I", "-c", ACCEPTED_TENSORFS]).arg(&runtime).arg(tools.join("tensorfs-wheel"))
                    .output().unwrap();
                assert!(out.status.success(), "the TensorFS {runtime:?} accepts: {out:?}");
                String::from_utf8(out.stdout).unwrap().trim().into()
            });
            args.extend(["--package-sdk".into(), runtime, "--package-sdk".into(), tensorfs]);
        }
        args
    }

    /// A Python that describes a project as the machine's Runtime does: the locked test
    /// environment, its Runtime replaced by a dev wheel (`COZY_TEST_RUNTIME_WHEEL`) when one is
    /// named. Kept in `tools`.
    fn describing_python(tools: &Path) -> PathBuf {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let Some(runtime) = std::env::var_os("COZY_TEST_RUNTIME_WHEEL") else {
            return installing_args(tools)[1].clone().into();
        };
        let env = tools.join("describing");
        let uv = |args: &[&std::ffi::OsStr]| {
            assert!(Command::new("uv").current_dir(repo).args(args).status().unwrap().success(), "uv {args:?}");
        };
        let requirements = env.join("requirements.txt");
        uv(&["venv".as_ref(), "--python".as_ref(), "3.12".as_ref(), env.as_os_str()]);
        uv(&["export".as_ref(), "--locked".as_ref(), "--extra".as_ref(), "test".as_ref(), "--no-emit-project".as_ref(),
            "--no-hashes".as_ref(), "-o".as_ref(), requirements.as_os_str()]);
        let python = env.join("bin/python");
        uv(&["pip".as_ref(), "install".as_ref(), "--python".as_ref(), python.as_os_str(), "-r".as_ref(), requirements.as_os_str()]);
        uv(&["pip".as_ref(), "install".as_ref(), "--python".as_ref(), python.as_os_str(), runtime.as_os_str(),
            "-e".as_ref(), repo.as_os_str()]);
        python
    }

    /// A live project's files written with Write, as `cozy run ./dir` sends them: its manifest.
    async fn write_live(
        client: &Client,
        cap: &str,
        project: &Path,
        package: &str,
        names: &[&str],
    ) -> String {
        let mut files = vec![];
        for name in names {
            let bytes = fs::read(project.join(name)).unwrap();
            let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&bytes));
            write(client, cap, &digest, bytes.len() as u64, 0, &bytes).await.unwrap();
            files.push(serde_json::json!({"name": name, "digest": digest, "length": bytes.len()}));
        }
        let manifest = serde_json::to_vec(&serde_json::json!({"package": package, "release": "0.1.0",
            "python_version": "3.12", "files": files}))
        .unwrap();
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&manifest));
        write(client, cap, &digest, manifest.len() as u64, 0, &manifest).await.unwrap();
        digest
    }

    /// A fixture copied where a test may edit it.
    fn project(fixture: &str, names: &[&str]) -> PathBuf {
        let from = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(fixture);
        let to = std::env::temp_dir().join(format!("cm-project-{}", uuid::Uuid::new_v4()));
        for name in names {
            fs::create_dir_all(to.join(name).parent().unwrap()).unwrap();
            fs::copy(from.join(name), to.join(name)).unwrap();
        }
        to
    }

    /// A fixture package written with Write: its manifest's digest.
    async fn write_package(
        client: &Client,
        cap: &str,
        fixture: &str,
        package: &str,
    ) -> String {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(fixture);
        let mut archive = tar::Builder::new(Vec::new());
        for name in [
            "pyproject.toml",
            "package.toml",
            &format!("{fixture}/__init__.py"),
        ] {
            archive
                .append_path_with_name(directory.join(name), name)
                .unwrap();
        }
        let source = archive.into_inner().unwrap();
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&source));
        let length = source.len() as u64;
        assert_eq!(
            write(client, cap, &digest, length, 0, &source)
                .await
                .unwrap(),
            length
        );
        let manifest = serde_json::to_vec(&serde_json::json!({
            "package": package, "release": "0.1.0",
            "python_version": "3.12", "source": {"digest": digest, "length": length}}))
        .unwrap();
        let manifest_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&manifest));
        let size = manifest.len() as u64;
        assert_eq!(
            write(client, cap, &manifest_digest, size, 0, &manifest)
                .await
                .unwrap(),
            size
        );
        manifest_digest
    }

    /// The thumbprint of the leaf the machine's API presents: what the CLI pins and names as a
    /// capability's `cnf.jkt`.
    fn leaf_jkt(machine: &Machine) -> String {
        let identity: serde_json::Value =
            serde_json::from_slice(&fs::read(machine.root.join("config/identity/identity.json")).unwrap()).unwrap();
        hub_client::P256Key::from_pem(identity["private_key_pem"].as_str().unwrap())
            .unwrap()
            .thumbprint()
    }

    /// `cozy model quantize`'s shape on CPU: a job writes its declared weights output through
    /// the machine's native writer channel and adopts it; the output is a product of its log
    /// (the manifest). A warm run keeps that held checkpoint under a local alias.
    #[tokio::test]
    #[ignore = "U10: the Runtime's derive client speaks the derive channel of #1494"]
    async fn a_job_writes_a_weights_output_kept_under_a_local_alias() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(&client, &all, "cpu_weights", "local/tensord-cpu-weights").await;
        let job = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "table".into(),
            payload: v1::Json::parse(r#"{"size":64}"#).ok(),
            owner: "alice".into(),
            ..Default::default()
        };
        let run = |id: &str, spec: v1::RunSpec| RunRequest {
            id: id.into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(run_events(&client, &all, run("weights", job.clone())).await.unwrap())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let written = done
            .outputs
            .iter()
            .find(|p| p.output == "model")
            .unwrap_or_else(|| panic!("{done:?}"));
        assert_eq!(written.media_type, "application/vnd.cozy.model-manifest");
        let result: serde_json::Value = json(&done.result);
        assert!(result.to_string().contains(&written.digest), "{result} names {written:?}");
        // A writer is never memoized: the same call runs again. Its derivation replays the
        // committed transaction of the same code and declaration, so its output is the same.
        assert!(!done.memoized && done.computation_digest.is_empty(), "{done:?}");
        let events = collect(run_events(&client, &all, run("weights-again", job.clone())).await.unwrap())
            .await
            .unwrap();
        let again = outcome(&events);
        assert_eq!(again.status, "succeeded", "{again:?}");
        assert!(!again.memoized && again.computation_digest.is_empty(), "{again:?}");
        let rewritten = again.outputs.iter().find(|p| p.output == "model").unwrap();
        assert_eq!((&rewritten.digest, rewritten.length), (&written.digest, written.length));
        let journal = Journal::open(&machine.root.join("state/execution")).unwrap();
        let transaction = |id: &str| {
            let record = journal.get_public(&actor(), id).unwrap();
            journal.derived_outputs(&record.id).unwrap().into_iter().map(|b| b.transaction).collect::<Vec<_>>()
        };
        assert_eq!(transaction("weights-again").len(), 1);
        assert!(transaction("weights").is_empty(), "the transaction passed to the later run");
        drop(journal);
        let held = tensorfs_core::store2::Store::open(machine.store()).unwrap();
        let reference = tensorfs_core::format::ObjectRef {
            sha256: written.digest.trim_start_matches("sha256:").into(),
            length: written.length,
        };
        held.checkpoint(&reference).unwrap();

        // The held checkpoint kept under a local alias, as `cozy model download … local/tiny`
        // keeps a model: no Hub, no publication.
        let keep = v1::RunSpec {
            kind: v1::RunKind::Warm,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                manifest: written.digest.clone(),
                manifest_length: written.length,
                ..Default::default()
            }],
            weights_destination: "local/tiny".into(),
            owner: "alice".into(),
            ..Default::default()
        };
        let events = collect(run_events(&client, &all, run("keep", keep)).await.unwrap())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["models"][0]["published"]["destination"], "local/tiny", "{result}");
        assert_eq!(result["models"][0]["published"]["checkpoint"], written.digest, "{result}");

        // A file that was never written is refused at submit: the run takes custody of none.
        let unwritten = v1::RunSpec {
            kind: v1::RunKind::Warm,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                source: format!("object://sha256:{}/stray.safetensors", "0".repeat(64)),
                ..Default::default()
            }],
            weights_destination: "local/stray".into(),
            owner: "alice".into(),
            ..Default::default()
        };
        let refused = match run_events(&client, &all, run("unwritten", unwritten)).await {
            Ok(stream) => collect(stream).await.unwrap_err(),
            Err(status) => status,
        };
        assert!(refused.to_string().contains("was not written to this machine"), "{refused:?}");

        let _ = fs::remove_dir_all(tools);
    }

    /// A deviceless function (`accelerator=False`) runs on a pooled executor: its second call
    /// is served by the executor of its first, which paid the imports once.
    #[tokio::test]
    async fn a_second_deviceless_call_reuses_its_executor() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let manifest = write_package(&client, &all, "cpu_memo", "local/tensord-cpu-memo").await;
        let mut served = vec![];
        for id in ["process-1", "process-2"] {
            let spec = v1::RunSpec {
                kind: v1::RunKind::Call,
                source: Some(v1::Source::Local(v1::LocalSource { manifest: manifest.clone() })),
                entrypoint: "process".into(),
                payload: raw(&serde_json::json!({"text": id})),
                owner: "alice".into(),
                ..Default::default()
            };
            let request = RunRequest { id: id.into(), after: 0, spec: Some(spec) };
            let done = outcome(&collect(run_events(&client, &all, request).await.unwrap()).await.unwrap());
            assert_eq!(done.status, "succeeded", "{done:?}");
            let result: serde_json::Value = json(&done.result);
            served.push((result["pid"].as_u64().unwrap(), result["served"].as_u64().unwrap()));
        }
        assert_eq!(served[0].0, served[1].0, "one executor served both calls: {served:?}");
        assert_eq!((served[0].1, served[1].1), (1, 2));
        let _ = fs::remove_dir_all(tools);
    }

    /// One run of `function` of the package `manifest` names, to its outcome: its result.
    async fn call(
        client: &Client,
        cap: &str,
        manifest: &str,
        (id, function): (&str, &str),
        payload: serde_json::Value,
    ) -> serde_json::Value {
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest: manifest.into() })),
            entrypoint: function.into(),
            payload: raw(&payload),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = authorized(RunRequest { id: id.into(), after: 0, spec: Some(spec) }, cap);
        let done = outcome(&collect(run_as(client, request).await.unwrap()).await.unwrap());
        assert_eq!(done.status, "succeeded", "{id}: {done:?}");
        json(&done.result)
    }

    /// On a GPU machine every call runs where its declaration puts it, on pooled executors: a
    /// deviceless root (`accelerator=False`) never opens CUDA and calls a GPU child; a GPU root
    /// calls a child on its own GPU, in another process; a raw-GPU call runs beside a warm
    /// executor without ending it, and that executor's next call finds its CUDA memory intact.
    #[tokio::test]
    #[ignore = "GPU: one card (a rental)"]
    async fn on_one_gpu_every_call_runs_where_its_declaration_puts_it() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tools = std::env::temp_dir().join(format!("cm-tools-{}", uuid::Uuid::new_v4()));
        assert!(Command::new("uv").current_dir(repo).args(["build", "--wheel", "--out-dir"]).arg(&tools)
            .status().unwrap().success());
        let home = tools.join("executor-home");
        fs::create_dir_all(&home).unwrap();
        let config = tools.join("gpu.json");
        fs::write(&config, serde_json::json!({"devices": "0", "environment": {
            "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": home}}).to_string()).unwrap();
        let mut args = installing_args(&tools);
        args.extend(["--gpu-config".into(), config.into_os_string()]);
        let machine = Machine::start_args(|_, _| (), &args).await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let manifest = write_package(&client, &all, "gpu_calls", "local/tensord-gpu-calls").await;
        let hold = serde_json::json!({"hold_s": 0.5});

        let deviceless = call(&client, &all, &manifest, ("cpu-root", "cpu_root"), hold.clone()).await;
        assert_eq!((deviceless["root"]["device"].as_str(), deviceless["child"]["device"].as_str()), (Some("cpu"), Some("cuda")));
        assert_eq!(deviceless["child"]["total"], 8192.0);
        let root = deviceless["root"]["pid"].as_u64().unwrap();
        assert_ne!(Some(root), deviceless["child"]["pid"].as_u64());
        // Its executor is pooled, still alive: it never mapped a CUDA driver.
        let maps = fs::read_to_string(format!("/proc/{root}/maps")).unwrap();
        assert!(!maps.contains("libcuda"), "a deviceless executor opened CUDA");

        let on_gpu = call(&client, &all, &manifest, ("gpu-root", "gpu_root"), hold).await;
        assert_eq!((on_gpu["root"]["device"].as_str(), on_gpu["child"]["device"].as_str()), (Some("cuda"), Some("cuda")));
        assert_ne!(on_gpu["root"]["pid"], on_gpu["child"]["pid"]);

        let none = serde_json::json!({});
        let warm = call(&client, &all, &manifest, ("warm-1", "warm"), none.clone()).await;
        let raw = call(&client, &all, &manifest, ("raw", "raw"), none.clone()).await;
        assert_eq!(raw["device"], "cuda");
        assert_ne!(raw["pid"], warm["pid"]);
        let again = call(&client, &all, &manifest, ("warm-2", "warm"), none).await;
        assert_eq!((&again["pid"], again["served"].as_u64()), (&warm["pid"], Some(2)), "the warm executor was ended");
        let _ = fs::remove_dir_all(tools);
    }

    /// A one-tensor checkpoint of `n` f32 elements a model slot constructs from, and its config.
    fn constructible(store: &tensorfs_core::store2::Store, n: usize) -> tensorfs_core::format::ObjectRef {
        use tensorfs_core::{format::Dtype, writer::{Data, Source}};
        let mut writer = store.writer(None, None).unwrap();
        writer.put("model", "weight", &Dtype::F32, &[n as u64], Data::Plain(Source::Read(&mut vec![5u8; 4 * n].as_slice()))).unwrap();
        writer.file("model/config.json", serde_json::json!({"n": n}).to_string().as_bytes()).unwrap();
        writer.commit().unwrap()
    }

    /// Run 6353: a deviceless job run with `--gpus 2` holds both GPUs, and its children, whose
    /// model slot declares one GPU, run on one each. The caller's count caps a child's group; it
    /// is never refused, nor is a count wider than the machine.
    #[tokio::test]
    #[ignore = "GPU: two cards (a rental)"]
    async fn on_two_gpus_a_runs_gpu_count_caps_its_one_gpu_children() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tools = std::env::temp_dir().join(format!("cm-tools-{}", uuid::Uuid::new_v4()));
        assert!(Command::new("uv").current_dir(repo).args(["build", "--wheel", "--out-dir"]).arg(&tools)
            .status().unwrap().success());
        let home = tools.join("executor-home");
        fs::create_dir_all(&home).unwrap();
        let config = tools.join("gpu.json");
        let mut args = installing_args(&tools);
        args.extend(["--gpu-config".into(), config.clone().into_os_string()]);
        let machine = Machine::start_args(|state, _| {
            let store = tensorfs_core::store2::Store::open(state.join("tensorfs")).unwrap();
            let manifest = constructible(&store, 2048);
            fs::write(&config, serde_json::json!({"devices": "0,1", "environment": {
                "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": home},
                "models": [{"package": "local/tensord-gpu-calls", "slot": "modeled.models.model",
                    "repository": "local/tiny", "release": "1", "lane": "f32",
                    "manifest": manifest.id(), "components": ["model"]}]}).to_string()).unwrap();
        }, &args).await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let manifest = write_package(&client, &all, "gpu_calls", "local/tensord-gpu-calls").await;
        for (id, gpus, hold_s) in [("job-2", 2, 0.5), ("job-4", 4, 0.4)] {
            let spec = v1::RunSpec {
                kind: v1::RunKind::Call,
                source: Some(v1::Source::Local(v1::LocalSource { manifest: manifest.clone() })),
                entrypoint: "cpu_job".into(),
                payload: raw(&serde_json::json!({"hold_s": hold_s})),
                owner: "alice".into(),
                gpus,
                ..Default::default()
            };
            let request = authorized(RunRequest { id: id.into(), after: 0, spec: Some(spec) }, &all);
            let events = collect(run_as(&client, request).await.unwrap()).await.unwrap();
            let done = outcome(&events);
            assert_eq!(done.status, "succeeded", "{id}: {done:?}");
            let result: serde_json::Value = json(&done.result);
            let children = result["children"].as_array().unwrap();
            assert_eq!(children.len(), 2, "{result}");
            assert!(children.iter().all(|c| c["device"].as_str().unwrap().starts_with("cuda")), "{result}");
            // Each call names the model it ran as the machine resolved it (`run show`).
            for event in &events {
                if let v1::Event::Call(call) = &event.event {
                    let measured = json(&call.measurements);
                    assert_eq!((&measured["models"][0]["repository"], &measured["models"][0]["lane"]),
                        (&"local/tiny".into(), &"f32".into()), "{measured}");
                }
            }
        }
        let _ = fs::remove_dir_all(tools);
    }

    /// GPU 0's memory in use, by anyone (MiB).
    fn gpu_used_mib() -> u64 {
        let out = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"])
            .output()
            .unwrap();
        String::from_utf8(out.stdout).unwrap().trim().parse().unwrap()
    }

    /// Driver descriptors `pid` holds: custody keeps one per exported chunk.
    fn driver_fds(pid: u32) -> usize {
        let fds = fs::read_dir(format!("/proc/{pid}/fd")).unwrap().flatten();
        let driver = |fd: &fs::DirEntry| fs::read_link(fd.path()).is_ok_and(|to| to.to_string_lossy().starts_with("/dev/nvidia"));
        fds.filter(driver).count()
    }

    /// A removed package's GPU weights go back to the GPU. Custody (Degree 2) keeps what its
    /// executor offered past the executor, and its removal ended the executor but revoked
    /// nothing (5090: the machine held 16.8 GB once both packages were removed, no process
    /// listed). What no installed package read is let go with the last one that did.
    #[tokio::test]
    #[ignore = "GPU: one card (a rental)"]
    async fn a_removed_packages_gpu_weights_leave_the_gpu() {
        const WEIGHT_MIB: u64 = 256;
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tools = std::env::temp_dir().join(format!("cm-tools-{}", uuid::Uuid::new_v4()));
        assert!(Command::new("uv").current_dir(repo).args(["build", "--wheel", "--out-dir"]).arg(&tools)
            .status().unwrap().success());
        let home = tools.join("executor-home");
        fs::create_dir_all(&home).unwrap();
        let config = tools.join("gpu.json");
        let mut args = installing_args(&tools);
        args.extend(["--gpu-config".into(), config.clone().into_os_string()]);
        let machine = Machine::start_args(|state, _| {
            let store = tensorfs_core::store2::Store::open(state.join("tensorfs")).unwrap();
            let manifest = constructible(&store, (WEIGHT_MIB << 18) as usize);
            fs::write(&config, serde_json::json!({"devices": "0", "environment": {
                "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": home},
                "models": [{"package": "local/tensord-gpu-calls", "slot": "resident.models.model",
                    "repository": "local/weight", "release": "1", "lane": "f32",
                    "manifest": manifest.id(), "components": ["model"]}]}).to_string()).unwrap();
        }, &args).await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let manifest = write_package(&client, &all, "gpu_calls", "local/tensord-gpu-calls").await;
        let (pid, idle_mib) = (machine.child.id(), gpu_used_mib());
        let idle_fds = driver_fds(pid);

        // The second call ends with its executor's weights offered to custody, if the first did not.
        for id in ["resident-1", "resident-2"] {
            let ran = call(&client, &all, &manifest, (id, "resident"), serde_json::json!({})).await;
            assert_eq!(ran["device"], "cuda:0", "{ran}");
        }
        assert!(driver_fds(pid) > idle_fds, "custody kept none of the executor's weights");
        assert!(gpu_used_mib() >= idle_mib + WEIGHT_MIB);

        let spec = v1::RunSpec {
            kind: v1::RunKind::Remove,
            source: Some(v1::Source::Release(v1::Release {
                package: "local/tensord-gpu-calls".into(),
                ..Default::default()
            })),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = authorized(RunRequest { id: "remove".into(), after: 0, spec: Some(spec) }, &all);
        let done = outcome(&collect(run_as(&client, request).await.unwrap()).await.unwrap());
        assert_eq!(done.status, "succeeded", "{done:?}");
        // The ended executor's context leaves with its process; the weights with custody's fds.
        assert_eq!(driver_fds(pid), idle_fds, "custody still holds the removed package's weights");
        let until = Instant::now() + Duration::from_secs(10);
        while gpu_used_mib() >= idle_mib + WEIGHT_MIB / 2 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(100));
        }
        let used = gpu_used_mib();
        assert!(used < idle_mib + WEIGHT_MIB / 2, "{used} MiB in use after the removal, {idle_mib} before the install");
        let _ = fs::remove_dir_all(tools);
    }

    /// An idle tenant gives its GPU to a request that needs it: an older release kept warm
    /// beside a newer one (both installed) is evicted on demand, and the newer release's
    /// request gets a budget for all its weights, its next one paging nothing. Klein bench
    /// (runs 7772, 7779): 0.1.0's idle executor kept its VRAM, and 0.2.0's executor got a
    /// 0.4 GiB budget and paged 18.6 GiB on every request. The weights take 45% of the card:
    /// one release's fit, two do not.
    #[tokio::test]
    #[ignore = "GPU: one card (a rental)"]
    async fn an_idle_older_release_gives_its_gpu_to_the_newer_ones_request() {
        let total: u64 = String::from_utf8(Command::new("nvidia-smi").args(["--query-gpu=memory.total",
            "--format=csv,noheader,nounits", "-i", "0"]).output().unwrap().stdout).unwrap().trim().parse().unwrap();
        let weight_mib = total * 45 / 100;
        const FIXTURE: (&str, &str) = ("gpu_calls", "tensord-gpu-calls");
        let tools = std::env::temp_dir().join(format!("cm-tools-{}", uuid::Uuid::new_v4()));
        assert!(Command::new("uv").current_dir(env!("CARGO_MANIFEST_DIR")).args(["build", "--wheel", "--out-dir"]).arg(&tools)
            .status().unwrap().success());
        let home = tools.join("executor-home");
        fs::create_dir_all(&home).unwrap();
        let config = tools.join("gpu.json");
        fs::write(&config, serde_json::json!({"devices": "0", "environment": {
            "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": home}}).to_string()).unwrap();
        let mut args = installing_args(&tools);
        args.extend(["--gpu-config".into(), config.into_os_string()]);
        let machine = Machine::start_args(|_, _| (), &args).await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        // The release's interface as the machine describes the code it installs.
        let local = write_package(&client, &all, FIXTURE.0, &format!("local/{}", FIXTURE.1)).await;
        let install = v1::RunSpec { kind: v1::RunKind::Warm, owner: "alice".into(),
            source: Some(v1::Source::Local(v1::LocalSource { manifest: local })), ..Default::default() };
        assert_eq!(settle(&client, &all, "describe", install).await.status, "succeeded");
        let generation = fs::read_dir(machine.root.join("state/generations")).unwrap().flatten()
            .find_map(|g| fs::read(g.path().join("generation.json")).ok()).unwrap();
        let interface = serde_json::from_slice::<serde_json::Value>(&generation).unwrap()["interface"].clone();
        let built = tools.join("hub-wheels");
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(FIXTURE.0);
        assert!(Command::new("uv").args(["build", "--wheel", "--out-dir"]).arg(&built).arg(&fixture).status().unwrap().success());
        let wheel = fs::read_dir(&built).unwrap().flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|e| e == "whl")).unwrap();
        let requirements = fs::read_to_string(fixture.join("pyproject.toml")).unwrap();
        let requires: Vec<String> = requirements.split("dependencies = [").nth(1).unwrap().split(']').next().unwrap()
            .split('"').skip(1).step_by(2).map(str::to_string).collect();
        let source = std::env::temp_dir().join(format!("cm-update-{}", uuid::Uuid::new_v4()));
        let store = tensorfs_core::store2::Store::open(&source).unwrap();
        let model = names_hub::Model { repository: "acme/weight".into(), release: "1".into(), lane: "f32".into(),
            manifest: constructible(&store, (weight_mib << 18) as usize) };
        let release = |release: &str| names_hub::Package {
            name: format!("acme/{}", FIXTURE.1), release: release.into(), interface: interface.clone(),
            wheel: wheel.file_name().unwrap().to_str().unwrap().into(), bytes: fs::read(&wheel).unwrap(), callees: vec![],
            requires: requires.clone(),
        };
        let (origin, _) = names_hub::serve(&store, model, [release("0.1.0"), release("0.2.0")]);
        let spec = |kind: v1::RunKind, release: &str, entrypoint: &str| v1::RunSpec {
            kind, entrypoint: entrypoint.into(), owner: "alice".into(),
            source: Some(v1::Source::Release(v1::Release { package: format!("acme/{}", FIXTURE.1), release: release.into() })),
            models: vec![v1::ModelChoice { parameter: "model".into(), repository: "acme/weight".into(),
                release: "1".into(), lane: "f32".into(), ..Default::default() }],
            hub: Some(v1::HubAccess { origin: origin.clone(), object_hosts: vec!["localhost".into()], ..Default::default() }),
            catalog_revision: "r1".into(),
            ..Default::default()
        };
        let memory = |done: &v1::Outcome| json(&done.measurements)["memory"].clone();

        // 0.1.0 warm and idle: its executor keeps the weights on the GPU.
        for id in ["old-1", "old-2"] {
            let done = settle(&client, &all, id, spec(v1::RunKind::Call, "0.1.0", "resident")).await;
            assert_eq!(done.status, "succeeded", "{done:?}");
        }
        // 0.2.0 installs beside it; both stay installed.
        let done = settle(&client, &all, "update", spec(v1::RunKind::Warm, "0.2.0", "")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        let releases: Vec<String> = status_frame(&client, &all).await.environments.into_iter()
            .filter(|e| e.package == format!("acme/{}", FIXTURE.1)).map(|e| e.release).collect();
        assert_eq!(releases, ["0.1.0", "0.2.0"]);

        // 0.2.0's request takes the idle tenant's room: every weight resident, nothing paged
        // out; its next request pages nothing in.
        for (id, first) in [("new-1", true), ("new-2", false)] {
            let done = settle(&client, &all, id, spec(v1::RunKind::Call, "0.2.0", "resident")).await;
            assert_eq!(done.status, "succeeded", "{done:?}");
            let memory = memory(&done);
            assert!(memory["budget_bytes"].as_u64() >= memory["weights_bytes"].as_u64(), "{id}: {memory}");
            assert_eq!(memory["paged_out_bytes"].as_u64().unwrap_or(0), 0, "{id}: {memory}");
            if !first {
                assert_eq!(memory["paged_in_bytes"].as_u64().unwrap_or(0), 0, "{id}: {memory}");
            }
        }
        let _ = fs::remove_dir_all(source);
        let _ = fs::remove_dir_all(tools);
    }

    /// A writer's weights output reaches its run's destination, the requester's repository, under
    /// the run's capability; the publication is a call event of the run.
    #[tokio::test]
    #[ignore = "real AuthKit: needs go and AUTHKIT_TEST_DATABASE_URL"]
    async fn a_writers_weights_output_reaches_its_destination() {
        let (upstream, hub) = hub_oauth::test_hub();
        let authkit = hub_oauth::AuthKit::start(&upstream);
        let (machine, tools) = installing_machine().await;
        let jkt = leaf_jkt(&machine);
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let manifest = write_package(&client, &all, "cpu_weights", "local/tensord-cpu-weights").await;
        let ops = serde_json::json!([{"type": "tensorhub_model_publish", "model": "acme/tiny"}]);
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "table".into(),
            payload: v1::Json::parse(r#"{"size":128}"#).ok(),
            weights_destination: "acme/tiny".into(),
            hub: Some(v1::HubAccess {
                origin: authkit.hub.clone(),
                object_hosts: vec!["localhost".into()],
                capability: authkit.capability(&jkt, &ops, 600),
                token_endpoint: authkit.token_endpoint.clone(),
                ..Default::default()
            }),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = RunRequest { id: "published-table".into(), after: 0, spec: Some(spec) };
        let events = collect(run_events(&client, &all, request).await.unwrap()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let written = done.outputs.iter().find(|p| p.output == "model").unwrap();
        assert_eq!(hub.lock().unwrap().finalized, std::slice::from_ref(&written.digest));
        let uploads: Vec<_> = events.iter().filter_map(|e| match &e.event {
            v1::Event::Call(call) if call.function.ends_with("upload_checkpoint") => Some(call.status.clone()),
            _ => None,
        }).collect();
        assert_eq!(uploads, ["succeeded"]);
        assert_eq!(authkit.exchanges(), 1, "one trade for the run");
        let _ = fs::remove_dir_all(tools);
    }

    /// `cozy model upload local/tiny acme/tiny` under the run's capability (th-241): the spec
    /// names the Hub and carries the owner's device-key-signed capability for this machine's
    /// leaf; the machine trades it once at the upload and AuthKit verifies every proof. With no
    /// capability, or one signed by a revoked device key, the run ends typed.
    #[tokio::test]
    #[ignore = "real AuthKit: needs go and AUTHKIT_TEST_DATABASE_URL"]
    async fn a_warm_run_publishes_a_local_alias_under_its_capability() {
        let (upstream, hub) = hub_oauth::test_hub();
        let authkit = hub_oauth::AuthKit::start(&upstream);
        let mut held = None;
        let machine = Machine::start_with(|state, _| {
            let store = tensorfs_core::store2::Store::open(state.join("tensorfs")).unwrap();
            let manifest = hub_oauth::checkpoint(&store, &[7; 8192]);
            tensord::storage::set_local(&store, "tiny", &manifest).unwrap();
            held = Some(manifest.id());
        })
        .await;
        let manifest = held.unwrap();
        let jkt = leaf_jkt(&machine);
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let ops = serde_json::json!([{"type": "tensorhub_model_publish", "model": "acme/tiny"}]);
        let upload = |id: &str, capability: String| RunRequest {
            id: id.into(),
            after: 0,
            spec: Some(v1::RunSpec {
                kind: v1::RunKind::Warm,
                models: vec![v1::ModelChoice {
                    parameter: "model".into(),
                    repository: "local/tiny".into(),
                    ..Default::default()
                }],
                weights_destination: "acme/tiny".into(),
                hub: Some(v1::HubAccess {
                    origin: authkit.hub.clone(),
                    object_hosts: vec!["localhost".into()],
                    capability,
                    token_endpoint: authkit.token_endpoint.clone(),
                    ..Default::default()
                }),
                owner: "alice".into(),
                ..Default::default()
            }),
        };
        let done = |events: Vec<v1::RunEvent>| outcome(&events);
        let refused = done(collect(run_events(&client, &all, upload("upload-0", String::new())).await.unwrap()).await.unwrap());
        assert_eq!(refused.status, "failed", "{refused:?}");
        assert_eq!(refused.reason.unwrap().code, "capability_required");
        assert_eq!(authkit.exchanges(), 0);

        for id in ["upload-1", "upload-2"] {
            let signed = authkit.capability(&jkt, &ops, 600);
            let done = done(collect(run_events(&client, &all, upload(id, signed)).await.unwrap()).await.unwrap());
            assert_eq!(done.status, "succeeded", "{id}: {done:?}");
            let result: serde_json::Value = json(&done.result);
            assert_eq!(result["models"][0]["published"]["checkpoint"], manifest, "{result}");
        }
        {
            let hub = hub.lock().unwrap();
            assert_eq!(hub.finalized, [manifest.clone(), manifest.clone()]);
            assert_eq!(hub.credentialed_uploads, 0);
        }
        assert_eq!(authkit.exchanges(), 2, "one trade per run");
        let owner = authkit.owner();
        let verified = authkit.verified();
        assert!(!verified.is_empty() && verified.iter().all(|s| s["owner"] == owner.as_str()), "{verified:?}");

        let signed = authkit.capability(&jkt, &ops, 600);
        authkit.revoke_device_key();
        let revoked = done(collect(run_events(&client, &all, upload("upload-3", signed)).await.unwrap()).await.unwrap());
        assert_eq!(revoked.status, "failed", "{revoked:?}");
        assert_eq!(revoked.reason.unwrap().code, "capability_refused");
        assert_eq!(hub.lock().unwrap().finalized.len(), 2);
    }

    /// A wheel with one module and nothing else: `name` `version`.
    fn wheel(name: &str, version: &str) -> Vec<u8> {
        use std::io::Write;
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        let info = format!("{name}-{version}.dist-info");
        let files = [
            (format!("{name}/__init__.py"), "app = None\n".to_string()),
            (format!("{info}/METADATA"), format!("Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")),
            (format!("{info}/WHEEL"), "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n".into()),
        ];
        let mut record = String::new();
        for (path, body) in &files {
            zip.start_file(path.as_str(), options).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
            record.push_str(&format!("{path},,\n"));
        }
        zip.start_file(format!("{info}/RECORD"), options).unwrap();
        zip.write_all(record.as_bytes()).unwrap();
        zip.finish().unwrap().into_inner()
    }

    /// One warm run to its outcome.
    async fn settle(
        client: &Client,
        cap: &str,
        id: &str,
        spec: v1::RunSpec,
    ) -> v1::Outcome {
        let request = RunRequest { id: id.into(), after: 0, spec: Some(spec) };
        outcome(&collect(run_events(client, cap, request).await.unwrap()).await.unwrap())
    }

    /// th-245: a run names its release and model and the machine resolves both at its Hub by
    /// name, anonymously: the newest release's install plan (its uv.lock), the model's lane by one closure
    /// call, then the checkpoint pulled pinned to that lane. Under the same catalog revision a
    /// second run asks the Hub nothing; a moved revision resolves once more and moves no bytes;
    /// a hash alone is the owner's, refused without the capability. Real uv, Python and TensorFS.
    #[tokio::test]
    async fn runs_name_their_release_and_model_and_a_warm_run_asks_the_hub_nothing() {
        let source = std::env::temp_dir().join(format!("cm-names-{}", uuid::Uuid::new_v4()));
        let store = tensorfs_core::store2::Store::open(&source).unwrap();
        let manifest = hub_oauth::checkpoint(&store, &[5; 8192]);
        let model = names_hub::Model {
            repository: "acme/model".into(),
            release: "1.0.0".into(),
            lane: "bf16".into(),
            manifest: manifest.clone(),
        };
        let package = names_hub::Package {
            name: "acme/pkg".into(),
            release: "1.0.0".into(),
            interface: serde_json::json!({"application": "pkg:app", "entrypoints": []}),
            wheel: "pkg-1.0.0-py3-none-any.whl".into(),
            bytes: wheel("pkg", "1.0.0"),
            callees: vec![],
            requires: vec![],
        };
        let (origin, heard) = names_hub::serve(&store, model, Some(package));
        let machine = Machine::start().await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let spec = |release: Option<&str>, choice: v1::ModelChoice, revision: &str| v1::RunSpec {
            kind: v1::RunKind::Warm,
            source: release.map(|release| {
                v1::Source::Release(v1::Release { package: "acme/pkg".into(), release: release.into() })
            }),
            models: vec![v1::ModelChoice { parameter: "model".into(), ..choice }],
            hub: Some(v1::HubAccess {
                origin: origin.clone(),
                object_hosts: vec!["localhost".into()],
                ..Default::default()
            }),
            catalog_revision: revision.into(),
            owner: "alice".into(),
            ..Default::default()
        };
        let named = v1::ModelChoice {
            repository: "acme/model".into(),
            release: "1.0.0".into(),
            lane: "bf16".into(),
            ..Default::default()
        };
        let take = || std::mem::take(&mut *heard.lock().unwrap());

        // Cold: the newest release installs by name; the model resolves and downloads by name.
        let done = settle(&client, &all, "names-1", spec(Some(""), named.clone(), "r1")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        let row = &result["models"][0];
        assert_eq!(result["release"], "1.0.0");
        assert_eq!((&row["release"], &row["lane"], &row["manifest"]), (&"1.0.0".into(), &"bf16".into(), &manifest.id().into()));
        let cold = take();
        for asked in [
            "GET /v1/packages/acme/pkg",
            "POST /v1/packages/acme/pkg/download",
            "POST /v1/tensorfs/closure acme/model@1.0.0 bf16",
            "POST /v1/tensorfs/presign",
        ] {
            assert!(cold.iter().any(|heard| heard == asked), "{asked} not in {cold:?}");
        }
        assert!(cold.iter().all(|heard| !heard.ends_with("+credential")), "{cold:?}");

        // Warm: the release's installation and the name's resolution are held, across a
        // restart too.
        let done = settle(&client, &all, "names-2", spec(Some("1.0.0"), named.clone(), "r1")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(take(), Vec::<String>::new());
        let mut machine = machine;
        machine.stop();
        (machine.child, machine.address) = launch(&machine.root, &[]);
        let client = self::client(&machine);
        let done = settle(&client, &all, "names-2b", spec(Some("1.0.0"), named.clone(), "r1")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(take(), Vec::<String>::new());
        // A run naming no release runs the one the Hub named under its revision: not asked again.
        let done = settle(&client, &all, "names-2c", spec(Some(""), named.clone(), "r1")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(take(), Vec::<String>::new());

        // A name alone is the newest release in the lane this machine's GPUs run (none here:
        // bf16): one closure, nothing to move.
        let bare = v1::ModelChoice { repository: "acme/model".into(), ..Default::default() };
        let done = settle(&client, &all, "names-3", spec(None, bare, "r1")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!((&result["models"][0]["release"], &result["models"][0]["lane"]), (&"1.0.0".into(), &"bf16".into()));
        assert_eq!(take(), ["POST /v1/tensorfs/closure acme/model bf16"]);

        // A revision the caller moved resolves the name once more; the bytes are held.
        let done = settle(&client, &all, "names-4", spec(Some("1.0.0"), named.clone(), "r2")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(take(), ["POST /v1/tensorfs/closure acme/model@1.0.0 bf16"]);

        // Another SDK is another environment of the same release: its card (pyproject, uv.lock)
        // is the one its first installation kept, never read again.
        let links = machine.root.join("links");
        fs::create_dir_all(&links).unwrap();
        machine.stop();
        (machine.child, machine.address) = launch(&machine.root, &["--package-find-links".into(), links.into()]);
        let client = self::client(&machine);
        let done = settle(&client, &all, "names-4b", spec(Some("1.0.0"), named.clone(), "r2")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(take(), Vec::<String>::new());
        let generations = fs::read_dir(machine.root.join("state/generations")).unwrap().flatten();
        let kept = generations.filter(|g| g.path().join("release.json").is_file()).count();
        assert_eq!(kept, 2, "each environment of the release keeps its card");
        // A moved revision asks once more which release is newest, and nothing else.
        let done = settle(&client, &all, "names-4c", spec(Some(""), named, "r2")).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(take(), ["GET /v1/packages/acme/pkg"]);

        // A hash alone is the owner's: absent without the run's capability.
        let other = format!("sha256:{}", "ab".repeat(32));
        let hashed = v1::ModelChoice { repository: "acme/model".into(), manifest: other.clone(), manifest_length: 9, ..Default::default() };
        let done = settle(&client, &all, "names-5", spec(None, hashed, "r2")).await;
        assert_eq!(done.status, "failed", "{done:?}");
        assert_eq!(take(), [format!("POST /v1/tensorfs/closure acme/model@{other} ")]);
        let _ = fs::remove_dir_all(source);
    }

    /// A pod's container caps its tasks (a vast RTX 5060 Ti: `pids.max` 1280) below what two
    /// cold downloads of up to 512 streams each could start together. Both land. On TensorFS
    /// 0.9.0 a refused thread panicked past the pull's sampler: the run stayed preparing for good
    /// and held its flight, so a resubmit joined the dead flight and cancel reached nothing (the
    /// 2026-10-15 prepare wedge).
    #[tokio::test]
    async fn concurrent_cold_downloads_under_a_task_cap_land() {
        const TASKS: u64 = 160;
        let limit = format!("TasksMax={TASKS}");
        let scope = ["systemd-run", "--user", "--scope", "-q", "-p", limit.as_str()];
        let usable = Command::new(scope[0]).args(&scope[1..]).arg("true").stdout(Stdio::null()).stderr(Stdio::null()).status();
        if !usable.is_ok_and(|status| status.success()) {
            eprintln!("SKIPPED: no systemd user manager to create a task-limited cgroup");
            return;
        }
        let source = std::env::temp_dir().join(format!("cm-tasks-{}", uuid::Uuid::new_v4()));
        let store = tensorfs_core::store2::Store::open(&source).unwrap();
        // 300 objects each: a pull would start a stream for every one.
        let model = |name: &str, tag: u32| {
            let bodies: Vec<Vec<u8>> = (0..300u32)
                .map(|at| (0..1024u32).flat_map(|i| [i, at, tag, 0]).flat_map(u32::to_le_bytes).collect())
                .collect();
            let weights: Vec<&[u8]> = bodies.iter().map(Vec::as_slice).collect();
            let manifest = hub_oauth::tensors(&store, &weights);
            let model = names_hub::Model { repository: name.into(), release: "1.0.0".into(), lane: "bf16".into(), manifest };
            names_hub::serve(&store, model, None).0
        };
        let origins = [model("acme/first", 1), model("acme/second", 2)];
        let mut machine = Machine::start().await;
        machine.stop();
        (machine.child, machine.address) = launch_in(&machine.root, &[], &scope);
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let download = |at: usize, name: &str| {
            let spec = v1::RunSpec {
                kind: v1::RunKind::Warm,
                models: vec![v1::ModelChoice {
                    parameter: "model".into(),
                    repository: name.into(),
                    release: "1.0.0".into(),
                    lane: "bf16".into(),
                    ..Default::default()
                }],
                hub: Some(v1::HubAccess { origin: origins[at].clone(), object_hosts: vec!["localhost".into()], ..Default::default() }),
                owner: "alice".into(),
                ..Default::default()
            };
            let (client, all, id) = (client.clone(), all.clone(), format!("tasks-{at}"));
            async move { settle(&client, &all, &id, spec).await }
        };
        // Test harness bound on a wedged preparation; the product has no such limit.
        let both = tokio::time::timeout(Duration::from_secs(120), async {
            tokio::join!(download(0, "acme/first"), download(1, "acme/second"))
        });
        let (first, second) = both.await.expect("a download under the task cap never settled");
        assert_eq!((first.status.as_str(), second.status.as_str()), ("succeeded", "succeeded"), "{first:?} {second:?}");
        let _ = fs::remove_dir_all(source);
    }

    /// th-245 with th-241: the owner's checkpoint by hash reads only under a run capability that
    /// names it. The machine trades it once through a real AuthKit and presents the token on the
    /// closure and presign; a capability naming another checkpoint reads anonymously and finds it
    /// absent, with no trade; one signed for another machine key is refused at submission.
    #[tokio::test]
    #[ignore = "real AuthKit: go and AUTHKIT_TEST_DATABASE_URL"]
    async fn the_owners_checkpoint_by_hash_reads_under_the_runs_capability() {
        let source = std::env::temp_dir().join(format!("cm-private-{}", uuid::Uuid::new_v4()));
        let store = tensorfs_core::store2::Store::open(&source).unwrap();
        let manifest = hub_oauth::checkpoint(&store, &[9; 8192]);
        let model = names_hub::Model {
            repository: "acme/private".into(),
            release: "1.0.0".into(),
            lane: "bf16".into(),
            manifest: manifest.clone(),
        };
        let (upstream, heard) = names_hub::serve(&store, model, None);
        let authkit = hub_oauth::AuthKit::start(&upstream);
        let machine = Machine::start().await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let spec = |capability: String| v1::RunSpec {
            kind: v1::RunKind::Warm,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                repository: "acme/private".into(),
                manifest: manifest.id(),
                manifest_length: manifest.length,
                ..Default::default()
            }],
            hub: Some(v1::HubAccess {
                origin: authkit.hub.clone(),
                object_hosts: vec!["localhost".into()],
                capability,
                token_endpoint: authkit.token_endpoint.clone(),
                ..Default::default()
            }),
            owner: "alice".into(),
            ..Default::default()
        };
        let read = |manifest: &str| serde_json::json!([{"type": "tensorhub_model_read", "model": "acme/private", "manifest": manifest}]);
        let jkt = leaf_jkt(&machine);

        let other = authkit.capability(&jkt, &read(&format!("sha256:{}", "ab".repeat(32))), 600);
        let done = settle(&client, &all, "private-1", spec(other)).await;
        assert_eq!(done.status, "failed", "{done:?}");
        assert_eq!(authkit.exchanges(), 0);

        let named = authkit.capability(&jkt, &read(&manifest.id()), 600);
        let done = settle(&client, &all, "private-2", spec(named)).await;
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(authkit.exchanges(), 1);
        let verified = authkit.verified();
        assert!(verified.iter().any(|seen| seen["path"] == "/v1/tensorfs/closure"), "{verified:?}");
        let closures: Vec<_> = heard.lock().unwrap().iter().filter(|h| h.contains("closure")).cloned().collect();
        assert_eq!(closures.len(), 2, "{closures:?}");

        let forged = authkit.capability(&hub_client::P256Key::generate().thumbprint(), &read(&manifest.id()), 600);
        let request = RunRequest { id: "private-3".into(), after: 0, spec: Some(spec(forged)) };
        let refused = run_events(&client, &all, request).await;
        let refused = match refused {
            Err(status) => status,
            Ok(stream) => collect(stream).await.unwrap_err(),
        };
        assert_eq!(refused.code(), Some("hub_access_invalid"), "{refused:?}");
        let _ = fs::remove_dir_all(source);
    }

    /// `cozy model upload` on the serve process: a warm run of one provider source makes it
    /// with TensorFS and puts its checkpoint in the destination under the machine's authority;
    /// every object reaches the Hub, then finalization.
    #[tokio::test]
    #[ignore = "real network: huggingface.co; real AuthKit: go and AUTHKIT_TEST_DATABASE_URL"]
    async fn a_warm_run_uploads_its_source_model_to_its_destination() {
        let (upstream, hub) = hub_oauth::test_hub();
        let authkit = hub_oauth::AuthKit::start(&upstream);
        let machine = Machine::start().await;
        let ops = serde_json::json!([{"type": "tensorhub_model_publish", "model": "acme/tiny"}]);
        let capability = authkit.capability(&leaf_jkt(&machine), &ops, 600);
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let spec = v1::RunSpec {
            kind: v1::RunKind::Warm,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                source: "hf://hf-internal-testing/tiny-sdxl-pipe@20594cbc343cfcfe447af5c87cdaf6c436b453f2/unet/diffusion_pytorch_model.safetensors".into(),
                ..Default::default()
            }],
            weights_destination: "acme/tiny".into(),
            hub: Some(v1::HubAccess {
                origin: authkit.hub.clone(),
                object_hosts: vec!["localhost".into()],
                capability,
                token_endpoint: authkit.token_endpoint.clone(),
                ..Default::default()
            }),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = RunRequest {
            id: "upload-1".into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(run_events(&client, &all, request).await.unwrap())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        let model = &result["models"][0];
        assert_eq!(model["published"]["destination"], "acme/tiny");
        assert_eq!(model["published"]["checkpoint"], model["manifest"]);
        let held = hub.lock().unwrap();
        assert_eq!(held.credentialed_uploads, 0);
        assert_eq!(held.finalized, [model["manifest"].as_str().unwrap()]);
        assert!(held.declared.contains(&held.finalized[0]));
        let mut uploaded: Vec<_> = held.uploaded.keys().cloned().collect();
        let mut declared = held.declared.clone();
        uploaded.sort();
        declared.sort();
        assert_eq!(uploaded, declared);
    }

    /// `cozy model upload <file>` and `cozy model download … local/name` on the serve process:
    /// a file the client wrote is made into a model with no provider and kept under a local
    /// alias. Real network for the fixture and its profile's reference configs.
    #[tokio::test]
    #[ignore = "real network: huggingface.co"]
    async fn a_written_file_is_made_and_kept_under_a_local_alias() {
        let machine = Machine::start().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let file = Command::new("curl")
            .args(["-sSfL", "https://huggingface.co/hf-internal-testing/tiny-sdxl-pipe/resolve/20594cbc343cfcfe447af5c87cdaf6c436b453f2/unet/diffusion_pytorch_model.safetensors"])
            .output()
            .unwrap();
        assert!(file.status.success());
        let sha256 = tensorfs_core::sha256::hex_digest(&file.stdout);
        let length = file.stdout.len() as u64;
        let held = write(&client, &all, &format!("sha256:{sha256}"), length, 0, &file.stdout)
            .await
            .unwrap();
        assert_eq!(held, length);
        let spec = v1::RunSpec {
            kind: v1::RunKind::Warm,
            models: vec![v1::ModelChoice {
                parameter: "model".into(),
                source: format!("object://sha256:{sha256}/unet.safetensors"),
                ..Default::default()
            }],
            weights_destination: "local/unet".into(),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = RunRequest {
            id: "keep-file".into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(run_events(&client, &all, request).await.unwrap())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        let model = &result["models"][0];
        assert_eq!(model["published"]["destination"], "local/unet", "{result}");
        let store = tensorfs_core::store2::Store::open(machine.store()).unwrap();
        let manifest = tensord::storage::local(&store, "unet").unwrap().unwrap();
        assert_eq!(model["manifest"], manifest.id());
    }

    /// A job's input tree (`--asset field=dir`, a `Tree` field): each file and the tree's manifest
    /// written with Write, the manifest named as an input of the tree media type under the
    /// payload's ref; the machine materializes the directory and the callable reads it.
    #[tokio::test]
    async fn a_written_input_tree_reaches_the_callable_as_a_directory() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(&client, &all, "cpu_tree", "local/tensord-cpu-tree").await;
        let files: [(&str, &[u8]); 2] = [("a.txt", b"alpha"), ("nested/b.bin", b"\x00\x01beta")];
        let mut entries = vec![];
        for (path, body) in files {
            let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(body));
            let length = body.len() as u64;
            assert_eq!(write(&client, &all, &digest, length, 0, body).await.unwrap(), length);
            entries.push(serde_json::json!({"kind": "file", "path": path,
                "blob": {"sha256": digest.trim_start_matches("sha256:"), "length": length}}));
        }
        let tree = serde_json::to_vec(&serde_json::json!({"entries": entries})).unwrap();
        let tree_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&tree));
        let size = tree.len() as u64;
        assert_eq!(write(&client, &all, &tree_digest, size, 0, &tree).await.unwrap(), size);
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "count".into(),
            // As `cozy run --asset data=<dir>` sends it: the payload names the tree's manifest.
            payload: raw(&serde_json::json!({ "data": tree_digest })),
            inputs: vec![v1::InputFile {
                field: "data".into(),
                digest: tree_digest.clone(),
                length: size,
                media_type: "application/vnd.cozy.tree-manifest".into(),
                order: 0,
            }],
            owner: "alice".into(),
            ..Default::default()
        };
        let request = RunRequest {
            id: "tree".into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(run_events(&client, &all, request).await.unwrap())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["files"], serde_json::json!(["a.txt", "nested/b.bin"]), "{result}");
        let mut whole = b"alpha".to_vec();
        whole.extend_from_slice(b"\x00\x01beta");
        assert_eq!(result["sha256"], tensorfs_core::sha256::hex_digest(&whole));
        let _ = fs::remove_dir_all(tools);
    }

    /// An executor holds no descriptor the machine did not hand it: not one the machine itself
    /// inherited (as a service inherits its parent's readiness, key and credential pipes).
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn an_executor_inherits_no_descriptor_the_machine_did_not_hand_it() {
        use std::os::fd::AsRawFd;
        // Not close-on-exec: the machine started below inherits it, as from a parent.
        let (planted, _write) = nix::unistd::pipe().unwrap();
        let target = fs::read_link(format!("/proc/self/fd/{}", planted.as_raw_fd()))
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let (machine, tools) = installing_machine().await;
        let held = fs::read_dir(format!("/proc/{}/fd", machine.child.id()))
            .unwrap()
            .filter_map(|e| fs::read_link(e.unwrap().path()).ok())
            .any(|t| t.to_string_lossy() == target);
        assert!(held, "the machine did not inherit the planted pipe {target}");
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(&client, &all, "cpu_tree", "local/tensord-cpu-tree").await;
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "descriptors".into(),
            payload: v1::Json::parse("{}").ok(),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = RunRequest {
            id: "descriptors".into(),
            after: 0,
            spec: Some(spec),
        };
        let events = collect(run_events(&client, &all, request).await.unwrap())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        let targets = result["targets"].as_array().unwrap();
        assert!(!targets.is_empty(), "{result}");
        assert!(
            !targets.iter().any(|t| t == &serde_json::json!(target)),
            "the executor inherited the machine's pipe {target}: {result}"
        );
        let _ = fs::remove_dir_all(tools);
    }

    /// A tree a child made (`out.save_tree`) reaches its parent as a directory, goes to a
    /// sibling as that one's input tree, and is the job's own output: its manifest and each
    /// member file are read back over Read.
    #[tokio::test]
    async fn a_childs_tree_is_handed_on_and_returned_as_the_jobs_output() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(&client, &all, "cpu_tree", "local/tensord-cpu-tree").await;
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "survey".into(),
            payload: raw(&serde_json::json!({ "text": "verified" })),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = RunRequest { id: "survey".into(), after: 0, spec: Some(spec) };
        let events = collect(run_events(&client, &all, request).await.unwrap())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["files"], serde_json::json!(["b.txt", "empty.txt", "nested/a.txt"]), "{result}");
        let product = done.outputs.iter().find(|p| p.output == "tree").expect("the tree is an output");
        assert_eq!(product.media_type, "application/vnd.cozy.tree-manifest");
        assert_eq!(result["tree"]["digest"], product.digest, "{result}");
        let target = |member: &str| ReadRequest {
            target: Some(Target::Output(v1::OutputTarget {
                run: "survey".into(),
                output: "tree".into(),
                index: 0,
                member: member.into(),
            })),
            ..Default::default()
        };
        let (meta, tree) = read(&client, &all, target("")).await.unwrap();
        assert_eq!(meta.digest, product.digest);
        let tree: serde_json::Value = serde_json::from_slice(&tree).unwrap();
        let paths: Vec<_> = tree["entries"].as_array().unwrap().iter().map(|e| e["path"].clone()).collect();
        assert_eq!(paths, ["b.txt", "empty.txt", "nested/a.txt"], "{tree}");
        let (meta, member) = read(&client, &all, target("nested/a.txt")).await.unwrap();
        assert_eq!(member, b"verified");
        assert_eq!(meta.digest, format!("sha256:{}", tensorfs_core::sha256::hex_digest(b"verified")));
        assert_eq!(read(&client, &all, target("empty.txt")).await.unwrap().1, b"");
        assert_eq!(read(&client, &all, target("absent")).await.unwrap_err().status(), Some(404));
        let _ = fs::remove_dir_all(tools);
    }

    fn outcome(events: &[v1::RunEvent]) -> v1::Outcome {
        match &events.last().unwrap().event {
            v1::Event::Outcome(outcome) => outcome.clone(),
            _ => panic!("{events:?}"),
        }
    }

    /// `cozy run ./dir` and `cozy run script.py` first ask the machine what the code declares: a
    /// warm run of the local source, at `installed`, installs it (editable, unlocked here, as the
    /// controller sends it) and its result carries the interface the environment's Runtime reads
    /// statically, with the installation a call may name. An edit written again returns the new
    /// interface.
    #[tokio::test]
    async fn a_warm_install_of_local_code_returns_its_interface() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let runner = client.clone();
        let installed = async |id: &str, manifest: &str| -> serde_json::Value {
            let spec = v1::RunSpec {
                kind: v1::RunKind::Warm,
                source: Some(v1::Source::Local(v1::LocalSource { manifest: manifest.into() })),
                owner: "alice".into(),
                ..Default::default()
            };
            let request = authorized(RunRequest { id: id.into(), after: 0, spec: Some(spec) }, &all);
            let done = outcome(&collect(run_as(&runner, request).await.unwrap()).await.unwrap());
            assert_eq!(done.status, "succeeded", "{done:?}");
            json(&done.result)
        };
        let names = |result: &serde_json::Value| -> Vec<String> {
            let rows = result["interface"]["entrypoints"].as_array().unwrap_or_else(|| panic!("{result}"));
            rows.iter().map(|row| row["name"].as_str().unwrap().to_owned()).collect()
        };

        // A project directory.
        let files = ["pyproject.toml", "package.toml", "cpu_memo/__init__.py"];
        let dir = project("cpu_memo", &files);
        let package = "local/tensord-cpu-memo";
        let manifest = write_live(&client, &all, &dir, package, &files).await;
        let first = installed("dir-1", &manifest).await;
        assert_eq!((first["package"].as_str(), first["release"].as_str()), (Some(package), Some("0.1.0")), "{first}");
        assert!(names(&first).contains(&"measure".to_string()), "{first}");
        assert!(first["installation"].as_str().is_some_and(|a| a.starts_with("local-")), "{first}");
        // A live edit adds a function: written again, the interface names it.
        let code = dir.join("cpu_memo/__init__.py");
        let added = "\n\nasync def tally(ctx: Context, payload: Probe) -> Measured:\n    return Measured(square=0)\n\n\napp.entrypoint(tally, accelerator=False)\n";
        fs::write(&code, fs::read_to_string(&code).unwrap() + added).unwrap();
        let edited = write_live(&client, &all, &dir, package, &files).await;
        assert_ne!(edited, manifest);
        let second = installed("dir-2", &edited).await;
        assert!(names(&second).contains(&"tally".to_string()) && !names(&first).contains(&"tally".to_string()), "{second}");

        // A script: the project the CLI generates around it (PEP 723 metadata, never locked).
        let script = std::env::temp_dir().join(format!("cm-script-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&script).unwrap();
        let generated = [
            ("pyproject.toml", concat!(
                "[project]\nname = \"cozy-script-0123456789ab\"\nversion = \"0.1.0\"\nrequires-python = \">=3.12\"\n",
                "dependencies = [\"cozy-runtime>=0.4.0,<1\"]\n\n",
                "[project.entry-points.\"cozy.application\"]\ndefault = \"cozy_script_entry:app\"\n\n",
                "[build-system]\nrequires = [\"hatchling>=1.25\"]\nbuild-backend = \"hatchling.build\"\n\n",
                "[tool.hatch.metadata]\nallow-direct-references = true\n\n",
                "[tool.hatch.build.targets.wheel]\nonly-include = [\"cozy_script.py\", \"cozy_script_entry.py\"]\n",
            )),
            ("package.toml", "[application]\nobject = \"cozy_script_entry:app\"\n"),
            ("cozy_script_entry.py", "from cozy_runtime.author import script_app\napp = script_app(\"cozy_script\")\n"),
            ("cozy_script.py", "# /// script\n# requires-python = \">=3.12\"\n# [tool.cozy]\n# accelerator = false\n# ///\ndef main() -> int:\n    return 1\n"),
        ];
        for (name, text) in generated {
            fs::write(script.join(name), text).unwrap();
        }
        let files: Vec<&str> = generated.iter().map(|(name, _)| *name).collect();
        let package = "local/cozy-script-0123456789ab";
        let manifest = write_live(&client, &all, &script, package, &files).await;
        let third = installed("script-1", &manifest).await;
        assert_eq!(names(&third), ["main"], "{third}");
        fs::write(script.join("cozy_script.py"), generated[3].1.replace("-> int:\n    return 1", "-> str:\n    return \"one\"")).unwrap();
        let edited = write_live(&client, &all, &script, package, &files).await;
        let fourth = installed("script-2", &edited).await;
        assert_eq!(names(&fourth), ["main"], "{fourth}");
        assert_ne!(third["interface"], fourth["interface"], "the edited return type");
        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_dir_all(script);
        let _ = fs::remove_dir_all(tools);
    }

    /// A job's memoized call that ran is held by this machine for its signer: a later job has
    /// the same call answered from it, never run, and a new one runs. The client carries none.
    #[tokio::test]
    async fn a_memoized_call_is_answered_from_this_machines_memo() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest =
            write_package(&client, &all, "cpu_memo", "local/tensord-cpu-memo").await;
        let counter = machine.root.join("measured");
        let survey = |id: &str, values: &[i64]| {
            let spec = v1::RunSpec {
                kind: v1::RunKind::Call,
                source: Some(v1::Source::Local(v1::LocalSource {
                    manifest: manifest.clone(),
                })),
                entrypoint: "survey".into(),
                payload: raw(&serde_json::json!({"values": values, "counter": counter})),
                owner: "alice".into(),
                ..Default::default()
            };
            authorized(RunRequest { id: id.into(), after: 0, spec: Some(spec) }, &all)
        };
        let measured = || std::fs::read_to_string(&counter).unwrap().parse::<u32>().unwrap();
        let squares = |events: &[v1::RunEvent]| {
            let done = outcome(events);
            assert_eq!(done.status, "succeeded", "{done:?}");
            json(&done.result)["squares"].clone()
        };
        let first = collect(run_as(&client, survey("survey-1", &[3, 4])).await.unwrap())
            .await
            .unwrap();
        assert_eq!(squares(&first), serde_json::json!([9, 16]));
        assert_eq!(measured(), 2);
        // The call for 3 is answered from the machine's memo; the one for 5 runs.
        let second = collect(run_as(&client, survey("survey-2", &[3, 5])).await.unwrap())
            .await
            .unwrap();
        assert_eq!(squares(&second), serde_json::json!([9, 25]));
        assert_eq!(measured(), 3, "the held call ran again");
        // Each call says how it was answered: the second job's call for 3 is memoized, with the
        // first job's computation; its call for 5 ran.
        let calls = |events: &[v1::RunEvent]| -> Vec<(bool, String)> {
            events
                .iter()
                .filter_map(|e| match &e.event {
                    v1::Event::Call(call) => Some((call.memoized, call.computation_digest.clone())),
                    _ => None,
                })
                .collect()
        };
        let (first, second) = (calls(&first), calls(&second));
        assert!(first.iter().all(|(memoized, digest)| !memoized && digest.starts_with("sha256:")), "{first:?}");
        assert_eq!(second[0], (true, first[0].1.clone()), "{second:?}");
        assert!(!second[1].0 && second[1].1.starts_with("sha256:") && second[1].1 != first[1].1, "{second:?}");
        let _ = fs::remove_dir_all(tools);
    }

    /// A memoized function run directly is answered from this machine's memo as a job's calls
    /// are: the same request again starts nothing and returns the same result and files; another
    /// request, or an edit of the package, runs. A function not declared memoized always runs,
    /// and a failed run is never held.
    #[tokio::test]
    async fn a_memoized_function_run_directly_is_answered_from_this_machines_memo() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let names = ["pyproject.toml", "package.toml", "cpu_memo/__init__.py"];
        let project = project("cpu_memo", &names);
        let package = "local/tensord-cpu-memo";
        let mut manifest = write_live(&client, &all, &project, package, &names).await;
        let (counter, noted) = (machine.root.join("measured"), machine.root.join("noted"));
        let runner = client.clone();
        let run = async |id: &str, manifest: &str, kind: v1::RunKind, entrypoint: &str, payload: serde_json::Value| {
            let spec = v1::RunSpec {
                kind,
                source: Some(v1::Source::Local(v1::LocalSource { manifest: manifest.into() })),
                entrypoint: entrypoint.into(),
                payload: raw(&payload),
                owner: "alice".into(),
                ..Default::default()
            };
            let request = authorized(RunRequest { id: id.into(), after: 0, spec: Some(spec) }, &all);
            collect(run_as(&runner, request).await.unwrap()).await.unwrap()
        };
        let probe = |value: i64, counter: &Path| serde_json::json!({"payload": {"value": value, "counter": counter}});
        let count = |path: &Path| fs::read_to_string(path).unwrap();
        let ran = |events: &[v1::RunEvent]| {
            events.iter().any(|e| matches!(&e.event, v1::Event::State(s) if s.state == "running"))
        };
        let entrypoint = v1::RunKind::Call;
        let first = run("direct-1", &manifest, entrypoint, "measure", probe(3, &counter)).await;
        let computed = outcome(&first);
        assert_eq!(computed.status, "succeeded", "{computed:?}");
        assert!(!computed.memoized && computed.computation_digest.starts_with("sha256:"), "{computed:?}");
        assert_eq!(json(&computed.result), serde_json::json!({"square": 9}));
        assert_eq!(count(&counter), "1");
        // The same request: answered from the memo, nothing started.
        let second = run("direct-2", &manifest, entrypoint, "measure", probe(3, &counter)).await;
        let answered = outcome(&second);
        assert_eq!(answered.status, "succeeded", "{answered:?}");
        assert!(answered.memoized && answered.computation_digest == computed.computation_digest, "{answered:?}");
        assert_eq!(answered.result, computed.result);
        assert!(!ran(&second) && answered.execution_ms == 0, "{second:?}");
        assert_eq!(count(&counter), "1", "the answered request ran");
        // Another request runs.
        let other = outcome(&run("direct-3", &manifest, entrypoint, "measure", probe(4, &counter)).await);
        assert!(!other.memoized && other.computation_digest != computed.computation_digest, "{other:?}");
        assert_eq!(count(&counter), "2");
        // A memoized job whose result is a file: answered with the same file.
        let job = v1::RunKind::Call;
        let mut notes = vec![];
        for id in ["note-1", "note-2"] {
            let events = run(id, &manifest, job, "note", probe(5, &noted)).await;
            let done = outcome(&events);
            assert_eq!(done.status, "succeeded", "{done:?}");
            let target = v1::OutputTarget { run: id.into(), output: "file".into(), ..Default::default() };
            let request = ReadRequest { target: Some(Target::Output(target)), ..Default::default() };
            notes.push((done.memoized, read(&client, &all, request).await.unwrap().1));
        }
        assert_eq!(notes, [(false, b"note 5".to_vec()), (true, b"note 5".to_vec())]);
        assert_eq!(count(&noted), "1", "the answered job ran");
        // A function not declared memoized always runs.
        for id in ["greet-1", "greet-2"] {
            let events = run(id, &manifest, entrypoint, "greet", serde_json::json!({"text": "hi"})).await;
            let done = outcome(&events);
            assert_eq!(done.status, "succeeded", "{done:?}");
            assert!(ran(&events) && !done.memoized && done.computation_digest.is_empty(), "{done:?}");
        }
        // A failed run is never held: the same request fails again by running again.
        let unwritable = machine.root.join("absent/measured");
        for id in ["failing-1", "failing-2"] {
            let events = run(id, &manifest, entrypoint, "measure", probe(3, &unwritable)).await;
            let done = outcome(&events);
            assert!(done.status == "failed" && ran(&events) && !done.memoized, "{done:?}");
        }
        // An edit of the package is another computation: it runs, then its own memo answers.
        let source = project.join("cpu_memo/__init__.py");
        fs::write(&source, fs::read_to_string(&source).unwrap() + "\nEDITED = True\n").unwrap();
        manifest = write_live(&client, &all, &project, package, &names).await;
        let edited = outcome(&run("edited-1", &manifest, entrypoint, "measure", probe(3, &counter)).await);
        assert!(!edited.memoized && edited.computation_digest != computed.computation_digest, "{edited:?}");
        assert_eq!(count(&counter), "3");
        let again = outcome(&run("edited-2", &manifest, entrypoint, "measure", probe(3, &counter)).await);
        assert!(again.memoized && again.computation_digest == edited.computation_digest, "{again:?}");
        assert_eq!(count(&counter), "3");
        let _ = fs::remove_dir_all(project);
        let _ = fs::remove_dir_all(tools);
    }

    /// An author's condition (`Memoize(requires=("seed",))`): a call that gives its seed is
    /// memoized, top-level and as a job's call alike; one that omits it runs every time,
    /// nothing is held, and the run says why.
    #[tokio::test]
    async fn a_call_is_memoized_only_when_it_gives_the_seed_its_author_requires() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let names = ["pyproject.toml", "package.toml", "cpu_seeded/__init__.py"];
        let project = project("cpu_seeded", &names);
        let manifest = write_live(&client, &all, &project, "local/tensord-cpu-seeded", &names).await;
        let counter = machine.root.join("painted");
        let runner = client.clone();
        let run = async |id: &str, kind: v1::RunKind, entrypoint: &str, payload: serde_json::Value| {
            let spec = v1::RunSpec {
                kind,
                source: Some(v1::Source::Local(v1::LocalSource { manifest: manifest.clone() })),
                entrypoint: entrypoint.into(),
                payload: raw(&payload),
                owner: "alice".into(),
                ..Default::default()
            };
            let request = authorized(RunRequest { id: id.into(), after: 0, spec: Some(spec) }, &all);
            collect(run_as(&runner, request).await.unwrap()).await.unwrap()
        };
        let count = || fs::read_to_string(&counter).unwrap();
        let paint = |prompt: &str, seed: Option<u64>| {
            let mut request = serde_json::json!({"prompt": prompt, "counter": counter});
            if let Some(seed) = seed {
                request["seed"] = seed.into();
            }
            request
        };
        let entrypoint = v1::RunKind::Call;
        // Seeded twice: the second is answered from the memo.
        let first = outcome(&run("seeded-1", entrypoint, "paint", paint("fox", Some(7))).await);
        let second = outcome(&run("seeded-2", entrypoint, "paint", paint("fox", Some(7))).await);
        assert!(first.status == "succeeded" && !first.memoized && first.computation_digest.starts_with("sha256:"), "{first:?}");
        assert!(second.memoized && second.computation_digest == first.computation_digest, "{second:?}");
        assert_eq!(second.result, first.result);
        assert_eq!(count(), "1");
        // The same seed, another prompt: another computation.
        let owl = outcome(&run("seeded-3", entrypoint, "paint", paint("owl", Some(7))).await);
        assert!(!owl.memoized && owl.computation_digest != first.computation_digest, "{owl:?}");
        assert_eq!(count(), "2");
        // Unseeded twice: two runs, each its own seed, nothing held, and each says why.
        let mut seeds = vec![];
        for id in ["unseeded-1", "unseeded-2"] {
            let done = outcome(&run(id, entrypoint, "paint", paint("fox", None)).await);
            assert_eq!(done.status, "succeeded", "{done:?}");
            assert!(!done.memoized && done.computation_digest.is_empty() && done.unmemoized == "seed not given", "{done:?}");
            seeds.push(json(&done.result)["seed"].clone());
        }
        assert_ne!(seeds[0], seeds[1], "each unseeded run drew its own seed");
        assert_eq!(count(), "4");
        // A job's calls alike: seeded, its second call is answered; unseeded, both run.
        let calls = |events: &[v1::RunEvent]| -> Vec<(bool, String)> {
            events
                .iter()
                .filter_map(|e| match &e.event {
                    v1::Event::Call(call) => Some((call.memoized, call.unmemoized.clone())),
                    _ => None,
                })
                .collect()
        };
        // `gallery` declares no accelerator: on this GPU-less machine it runs on CPU.
        let job = v1::RunKind::Call;
        let seeded = run("gallery-1", job, "gallery", paint("heron", Some(9))).await;
        assert_eq!(outcome(&seeded).status, "succeeded", "{seeded:?}");
        assert_eq!(calls(&seeded), [(false, String::new()), (true, String::new())]);
        assert_eq!(count(), "5");
        let unseeded = run("gallery-2", job, "gallery", paint("heron", None)).await;
        assert_eq!(outcome(&unseeded).status, "succeeded", "{unseeded:?}");
        let why = "seed not given".to_string();
        assert_eq!(calls(&unseeded), [(false, why.clone()), (false, why)]);
        assert_eq!(count(), "7");
        let _ = fs::remove_dir_all(project);
        let _ = fs::remove_dir_all(tools);
    }

    /// A job calls another package's invocable that its environment holds as a dependency, as a
    /// published package's callee is: the call is a child run of that package, counted where it
    /// truly runs.
    #[tokio::test]
    #[ignore = "U10: the Runtime's derive client speaks the derive channel of #1494"]
    async fn a_job_calls_a_package_its_environment_depends_on() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let built = tools.join("callee-wheel");
        let status = std::process::Command::new("uv")
            .args(["build", "--wheel", "--out-dir"])
            .arg(&built)
            .arg(fixtures.join("cpu_memo"))
            .status()
            .unwrap();
        assert!(status.success());
        // uv also writes a .gitignore into the out dir.
        let wheel = fs::read_dir(&built).unwrap().map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "whl")).unwrap();
        let bytes = fs::read(&wheel).unwrap();
        let wheel_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&bytes));
        write(&client, &all, &wheel_digest, bytes.len() as u64, 0, &bytes).await.unwrap();
        let mut archive = tar::Builder::new(Vec::new());
        for name in ["pyproject.toml", "package.toml", "cpu_caller/__init__.py"] {
            archive
                .append_path_with_name(fixtures.join("cpu_caller").join(name), name)
                .unwrap();
        }
        let source = archive.into_inner().unwrap();
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&source));
        write(&client, &all, &digest, source.len() as u64, 0, &source).await.unwrap();
        let manifest = serde_json::to_vec(&serde_json::json!({
            "package": "local/tensord-cpu-caller", "release": "0.1.0", "python_version": "3.12",
            "source": {"digest": digest, "length": source.len()},
            "wheels": [{"name": wheel.file_name().unwrap().to_str().unwrap(), "digest": wheel_digest, "length": bytes.len()}],
            "callees": {"tensord-cpu-memo": "local/tensord-cpu-memo"}}))
        .unwrap();
        let manifest_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&manifest));
        write(&client, &all, &manifest_digest, manifest.len() as u64, 0, &manifest).await.unwrap();
        let counter = machine.root.join("measured");
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource {
                manifest: manifest_digest,
            })),
            entrypoint: "relay".into(),
            payload: raw(&serde_json::json!({"values": [3, 4], "counter": counter})),
            owner: "alice".into(),
            ..Default::default()
        };
        let request = authorized(
            RunRequest {
                id: "relay-1".into(),
                after: 0,
                spec: Some(spec.clone()),
            },
            &all,
        );
        let events = collect(run_as(&client, request).await.unwrap())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["squares"], serde_json::json!([9, 16]));
        assert_eq!(result["greeting"], "squared squared", "{result}");
        assert_eq!(fs::read_to_string(&counter).unwrap(), "2");
        // Each call is a child run of the callee's own function, and memoized as its own.
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Call(call) => Some(call.function.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, ["measure", "measure", "greet"], "{events:?}");
        // Package log lines are the run's log: a call's lead with its function.
        let logs: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Log(line) => Some((line.level.clone(), line.text.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            logs,
            [
                ("warning".to_string(), "greet: greeting times=2".to_string()),
                ("info".to_string(), "relayed greeting=\"squared squared\" squares=2".to_string()),
            ],
            "{events:?}"
        );
        // H3's references (run 4830): the job shows each image another package's entrypoint
        // returns as its own output the moment it exists, and returns them all at the end.
        let mut sitting = spec.clone();
        sitting.entrypoint = "portrait".into();
        sitting.payload = raw(&serde_json::json!({"names": ["Lighthouse", "Keeper"]}));
        let events = collect(run_events(&client, &all, RunRequest {
            id: "portrait-1".into(), after: 0, spec: Some(sitting),
        }).await.unwrap()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Call(call) => Some(call.function.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, ["paint", "paint"], "{events:?}");
        let shown: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Product(p) if p.output == "references" => Some(p.label.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(shown, ["Reference: Lighthouse", "Reference: Keeper"], "{events:?}");
        let returned = done.result.as_ref().map(|r| r.get().to_string()).unwrap_or_default();
        let mut seen = Vec::new();
        for index in 1..=2u32 {
            let (_, image) = read(
                &client,
                &all,
                ReadRequest {
                    target: Some(Target::Output(v1::OutputTarget {
                        run: "portrait-1".into(),
                        output: "references".into(),
                        index,
                        ..Default::default()
                    })),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            assert!(image.starts_with(b"\x89PNG"), "reference {index} is no PNG");
            let digest = tensorfs_core::sha256::hex_digest(&image);
            assert!(returned.contains(&digest), "reference {index} ({digest}) is not the one returned: {returned}");
            seen.push(digest);
        }
        assert_ne!(seen[0], seen[1], "each reference is its own image");
        // The Runtime's own operations are callees of every environment: a job quantizes
        // another package's model with the Runtime's `quantize`, as a child run of its own.
        let mut requantize = spec.clone();
        requantize.entrypoint = "requantize".into();
        requantize.payload = v1::Json::parse("{}").ok();
        let events = collect(run_events(&client, &all, RunRequest {
            id: "requantize-1".into(), after: 0, spec: Some(requantize),
        }).await.unwrap()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Call(call) => Some(call.function.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, ["produce", "quantize"], "{events:?}");
        let result: serde_json::Value = json(&done.result);
        let (source, quantized) = (result["source"].as_str().unwrap(), result["result"].as_str().unwrap());
        assert!(source.starts_with("sha256:") && quantized.starts_with("sha256:"), "{result}");
        assert_ne!(source, quantized, "quantize wrote a model of its own");
        // A writer's checkpoint handed to a memoized consumer: each job writes it again as a new
        // producer, and the consumer, keyed on the checkpoint's manifest, is answered from memo.
        let weighed = machine.root.join("weighed");
        let mut derived = vec![];
        for id in ["weigh-1", "weigh-2"] {
            let mut spec = spec.clone();
            spec.entrypoint = "derive_and_weigh".into();
            spec.payload = raw(&serde_json::json!({"values": [], "counter": weighed}));
            let events = collect(run_events(&client, &all, RunRequest { id: id.into(), after: 0, spec: Some(spec) })
                .await.unwrap()).await.unwrap();
            let done = outcome(&events);
            assert_eq!(done.status, "succeeded", "{done:?}");
            let calls: Vec<_> = events.iter().filter_map(|e| match &e.event {
                v1::Event::Call(call) => Some((call.function.clone(), call.memoized)),
                _ => None,
            }).collect();
            derived.push((json(&done.result), calls));
        }
        assert_ne!(derived[0].0["producer"], derived[1].0["producer"], "{derived:?}");
        assert_eq!(derived[0].0["length"], derived[1].0["length"], "{derived:?}");
        assert_eq!(derived[1].1, [("produce".to_string(), false), ("weigh".to_string(), true)], "{derived:?}");
        assert_eq!(fs::read_to_string(&weighed).unwrap(), "1", "the consumer ran once");
        // A memoized call whose result is a file is answered from this machine's own memo while
        // it holds the file, for any later job of its signer; once the file is gone, it runs.
        let noted = machine.root.join("noted");
        let annotate = |id: &str| {
            let mut spec = spec.clone();
            spec.entrypoint = "annotate".into();
            spec.payload = raw(&serde_json::json!({"values": [5, 6], "counter": noted}));
            authorized(RunRequest { id: id.into(), after: 0, spec: Some(spec) }, &all)
        };
        let mut notes = vec![];
        for id in ["annotate-1", "annotate-2"] {
            let events = collect(run_as(&client, annotate(id)).await.unwrap()).await.unwrap();
            assert_eq!(outcome(&events).status, "succeeded", "{events:?}");
            let mut files = vec![];
            for index in 1..=2u32 {
                let target = v1::OutputTarget { run: id.into(), output: "files".into(), index, ..Default::default() };
                let request = ReadRequest { target: Some(Target::Output(target)), ..Default::default() };
                files.push(read(&client, &all, request).await.unwrap().1);
            }
            notes.push(files);
        }
        assert_eq!(notes[0], [b"note 5".to_vec(), b"note 6".to_vec()]);
        assert_eq!(notes[1], notes[0], "the reused result is the same files");
        assert_eq!(fs::read_to_string(&noted).unwrap(), "2", "the second job ran its calls again");
        // The store loses one file (as its GC does once reclaim released it): that call runs.
        let hex = tensorfs_core::sha256::hex_digest(b"note 5");
        let blob = walkdir(&machine.root.join("state/tensorfs"))
            .into_iter()
            .find(|p| {
                p.components().any(|c| c.as_os_str() == "blobs")
                    && p.file_name().is_some_and(|n| n.to_string_lossy().contains(&hex))
            })
            .expect("the note's bytes are in the store");
        fs::remove_file(blob).unwrap();
        let events = collect(run_as(&client, annotate("annotate-3")).await.unwrap()).await.unwrap();
        assert_eq!(outcome(&events).status, "succeeded", "{events:?}");
        assert_eq!(fs::read_to_string(&noted).unwrap(), "3", "only the call whose file is gone ran");
        let mut nested = spec.clone();
        nested.entrypoint = "relay_nested".into();
        let events = collect(run_events(&client, &all, RunRequest {
            id: "relay-nested".into(), after: 0, spec: Some(nested),
        }).await.unwrap()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["squares"], serde_json::json!([9, 16]));
        assert_eq!(fs::read_to_string(&counter).unwrap(), "4");
        // A served entrypoint calls children as a job does: each a child run that truly runs.
        let mut served = spec.clone();
        (served.kind, served.entrypoint) = (v1::RunKind::Call, "relay_served".into());
        served.payload = raw(&serde_json::json!({"values": [5, 6], "counter": counter}));
        let events = collect(run_events(&client, &all, RunRequest {
            id: "relay-served".into(), after: 0, spec: Some(served),
        }).await.unwrap()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["squares"], serde_json::json!([25, 36]));
        assert_eq!(fs::read_to_string(&counter).unwrap(), "6");
        let mut restricted = spec;
        restricted.entrypoint = "relay_restricted".into();
        let events = collect(run_events(&client, &all, RunRequest {
            id: "relay-restricted".into(), after: 0, spec: Some(restricted),
        }).await.unwrap()).await.unwrap();
        assert_eq!(outcome(&events).status, "failed", "{events:?}");
        assert_eq!(fs::read_to_string(&counter).unwrap(), "6", "a foreign internal export must run no effects");
        // A published release's environment is described the same way, inside it.
        let python = fs::read_dir(machine.root.join("state/generations"))
            .unwrap()
            .flatten()
            .map(|g| g.path().join("env/bin/python"))
            .find(|p| p.exists())
            .unwrap();
        let (digest, callees) = tensord::published::describe_environment(
            python.to_str().unwrap(),
            "tensord-cpu-caller",
            &Default::default(),
        ).unwrap();
        assert!(digest.starts_with("sha256:"), "{digest}");
        assert_eq!(callees[0].application, "cpu_memo:app", "{callees:?}");
        assert_eq!(callees[0].package, "local/tensord-cpu-memo");
        // The root and callee can both be immutable wheels; no source archive is needed.
        assert!(std::process::Command::new("uv").args(["build", "--wheel", "--out-dir"])
            .arg(&built).arg(fixtures.join("cpu_caller")).status().unwrap().success());
        let root_wheel = fs::read_dir(&built).unwrap().flatten()
            .map(|entry| entry.path()).find(|path| path.file_name().unwrap().to_str().unwrap().starts_with("tensord_cpu_caller-")).unwrap();
        let root_bytes = fs::read(&root_wheel).unwrap();
        let root_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&root_bytes));
        write(&client, &all, &root_digest, root_bytes.len() as u64, 0, &root_bytes).await.unwrap();
        let manifest = serde_json::to_vec(&serde_json::json!({
            "package":"local/tensord-cpu-caller","release":"0.1.0","python_version":"3.12",
            "wheels":[
                {"name":root_wheel.file_name().unwrap().to_str().unwrap(),"digest":root_digest,"length":root_bytes.len()},
                {"name":wheel.file_name().unwrap().to_str().unwrap(),"digest":wheel_digest,"length":bytes.len()}],
            "callees":{"tensord-cpu-memo":"local/tensord-cpu-memo"}
        })).unwrap();
        let manifest_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&manifest));
        write(&client, &all, &manifest_digest, manifest.len() as u64, 0, &manifest).await.unwrap();
        let wheel_counter = machine.root.join("wheel-measured");
        let events = collect(run_events(&client, &all, RunRequest {
            id:"relay-wheels".into(),after:0,spec:Some(v1::RunSpec {
                kind:v1::RunKind::Call,entrypoint:"relay".into(),owner:"alice".into(),
                source:Some(v1::Source::Local(v1::LocalSource { manifest:manifest_digest })),
                payload:raw(&serde_json::json!({"values":[3,4],"counter":wheel_counter})),
                ..Default::default()
            }),
        }).await.unwrap()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        assert_eq!(fs::read_to_string(&wheel_counter).unwrap(), "2");
        let _ = fs::remove_dir_all(tools);
    }

    /// A published release calls another package its uv.lock pins at the Hub (long_form calling
    /// qwen-image-2's generate_image): the release installs from that lock, the callee from the
    /// Hub's file door through TensorFS, and runs as a child under its own Hub name, `<org>/<name>`, so the run's model
    /// choices addressed to it reach it. Named `local/<name>` (TensorD 0.4.1-0.5.0), a choice
    /// for generate_image never arrived: model_choice_absent (runs 5141, 5142).
    #[tokio::test]
    async fn a_published_release_calls_its_hub_callee_under_the_callees_own_name() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant { action: MACHINE.into(), ..Default::default() });
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let built = tools.join("hub-wheels");
        for fixture in ["cpu_caller", "cpu_memo"] {
            assert!(Command::new("uv").args(["build", "--wheel", "--out-dir"]).arg(&built)
                .arg(fixtures.join(fixture)).status().unwrap().success());
        }
        let wheel = |prefix: &str| {
            let path = fs::read_dir(&built).unwrap().flatten().map(|e| e.path())
                .find(|p| p.file_name().unwrap().to_str().unwrap().starts_with(prefix)).unwrap();
            (path.file_name().unwrap().to_str().unwrap().to_string(), fs::read(&path).unwrap())
        };
        let ((root, root_bytes), callee) = (wheel("tensord_cpu_caller-"), wheel("tensord_cpu_memo-"));
        let described = Command::new(describing_python(&tools))
            .args(["-m", "tensord_client.runtime_describe"])
            .env("PYTHONPATH", Path::new(env!("CARGO_MANIFEST_DIR")).join("python"))
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped())
            .spawn().and_then(|mut child| {
                use std::io::Write as _;
                let project = fixtures.join("cpu_caller");
                child.stdin.take().unwrap().write_all(serde_json::json!({"kind": "describe", "project": project}).to_string().as_bytes())?;
                child.wait_with_output()
            }).unwrap();
        let described: serde_json::Value = serde_json::from_slice(&described.stdout).unwrap();
        let source = std::env::temp_dir().join(format!("cm-callee-{}", uuid::Uuid::new_v4()));
        let store = tensorfs_core::store2::Store::open(&source).unwrap();
        let model = names_hub::Model {
            repository: "acme/model".into(), release: "1.0.0".into(), lane: "bf16".into(),
            manifest: hub_oauth::checkpoint(&store, &[5; 8192]),
        };
        let package = names_hub::Package {
            name: "acme/tensord-cpu-caller".into(), release: "0.1.0".into(),
            interface: described["interface"].clone(), wheel: root, bytes: root_bytes, callees: vec![callee],
            // The rest of its lock is PyPI's: a real Runtime and its dependencies.
            requires: vec!["cozy-runtime".into(), "msgspec>=0.19,<1".into()],
        };
        let (origin, _) = names_hub::serve(&store, model, Some(package));
        let counter = machine.root.join("published-measured");
        let events = collect(run_events(&client, &all, RunRequest {
            id: "published-relay".into(), after: 0, spec: Some(v1::RunSpec {
                kind: v1::RunKind::Call, entrypoint: "relay".into(), owner: "alice".into(),
                source: Some(v1::Source::Release(v1::Release {
                    package: "acme/tensord-cpu-caller".into(), release: "0.1.0".into() })),
                payload: raw(&serde_json::json!({"values": [3, 4], "counter": counter})),
                hub: Some(v1::HubAccess { origin, object_hosts: vec!["localhost".into()], ..Default::default() }),
                catalog_revision: "r1".into(),
                ..Default::default()
            }),
        }).await.unwrap()).await.unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["squares"], serde_json::json!([9, 16]));
        let calls: Vec<_> = events.iter().filter_map(|e| match &e.event {
            v1::Event::Call(call) => Some(call.function.clone()),
            _ => None,
        }).collect();
        assert_eq!(calls, ["measure", "measure", "greet"], "{events:?}");
        let named: Vec<String> = fs::read_dir(machine.root.join("state/generations")).unwrap().flatten()
            .filter_map(|g| fs::read(g.path().join("generation.json")).ok())
            .filter_map(|raw| serde_json::from_slice::<serde_json::Value>(&raw).ok())
            .flat_map(|record| record["callees"].as_array().cloned().unwrap_or_default())
            .filter(|callee| callee["distribution"] == "tensord-cpu-memo" || callee["distribution"] == "tensord_cpu_memo")
            .map(|callee| callee["package"].as_str().unwrap_or_default().to_string())
            .collect();
        assert_eq!(named, ["acme/tensord-cpu-memo"]);
        let _ = fs::remove_dir_all(source);
        let _ = fs::remove_dir_all(tools);
    }

    /// Uploads and run sources on the real serve process: an object resumes from what is held,
    /// and unpublished code uploaded to the machine prepares inside its run (warm, then a call).
    #[tokio::test]
    async fn written_local_code_prepares_inside_its_run() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });

        let mut archive = tar::Builder::new(Vec::new());
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_lifecycle");
        for name in [
            "pyproject.toml",
            "package.toml",
            "cpu_lifecycle/__init__.py",
        ] {
            archive
                .append_path_with_name(fixture.join(name), name)
                .unwrap();
        }
        let source = archive.into_inner().unwrap();
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&source));
        let length = source.len() as u64;
        // A probe holds nothing; half, then the rest from the held ranges.
        assert_eq!(
            write(&client, &all, &digest, length, 0, &[])
                .await
                .unwrap(),
            0
        );
        let half = length / 2;
        assert_eq!(
            write(
                &client,
                &all,
                &digest,
                length,
                0,
                &source[..half as usize]
            )
            .await
            .unwrap(),
            half
        );
        // A part past a gap lands too: the held ranges say so, and the gap is sent after it.
        let bulk = by(&client, &all).bulk().unwrap();
        let ahead = bulk.put(&digest, (half + 1, 1, length), source[half as usize + 1..][..1].to_vec()).await;
        assert_eq!(ahead.unwrap(), [(0, half), (half + 1, half + 2)]);
        assert_eq!(
            write(
                &client,
                &all,
                &digest,
                length,
                half,
                &source[half as usize..]
            )
            .await
            .unwrap(),
            length
        );
        let manifest = serde_json::to_vec(&serde_json::json!({
            "package": "local/tensord-cpu-lifecycle", "release": "0.1.0",
            "python_version": "3.12", "source": {"digest": digest, "length": length}}))
        .unwrap();
        let manifest_digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&manifest));
        let size = manifest.len() as u64;
        assert_eq!(
            write(&client, &all, &manifest_digest, size, 0, &manifest)
                .await
                .unwrap(),
            size
        );

        let local = |kind: v1::RunKind| v1::RunSpec {
            kind,
            source: Some(v1::Source::Local(v1::LocalSource {
                manifest: manifest_digest.clone(),
            })),
            entrypoint: "steps".into(),
            payload: v1::Json::parse(r#"{"steps":2,"seconds":0.01}"#).ok(),
            owner: "alice".into(),
            ..Default::default()
        };
        let run = |id: &str, spec: v1::RunSpec| RunRequest {
            id: id.into(),
            after: 0,
            spec: Some(spec),
        };
        let warm = collect(
            run_events(&client, &all, run("warm-1", local(v1::RunKind::Warm)))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(outcome(&warm).status, "succeeded", "{warm:?}");
        // A warm run naming no entrypoint installs the code alone (`cozy package install`).
        let mut install = local(v1::RunKind::Warm);
        install.entrypoint.clear();
        install.payload = None;
        let installed = collect(
            run_events(&client, &all, run("warm-2", install))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let installed = outcome(&installed);
        assert_eq!(installed.status, "succeeded", "{installed:?}");
        let result: serde_json::Value = json(&installed.result);
        assert_eq!(result["package"], "local/tensord-cpu-lifecycle");
        assert_eq!(result["models"], serde_json::json!([]));
        // A warm run's set is the caller's warm set: Status shows each member as it was sent,
        // with what it holds now. This function binds no model, so it holds its code.
        let frame = status_frame(&client, &all).await;
        assert!(frame.capabilities.iter().any(|c| c == "warm/2"), "{frame:?}");
        let member = |entrypoint: &str| v1::WarmItem {
            source: Some(v1::WarmSource::Installation(
                frame.environments[0].installation.clone(),
            )),
            entrypoint: entrypoint.into(),
            level: v1::WarmLevel::Host,
            ..Default::default()
        };
        let set = |items: Vec<v1::WarmItem>| v1::RunSpec {
            kind: v1::RunKind::Warm,
            set: Some(v1::WarmSet { items }),
            owner: "alice".into(),
            ..Default::default()
        };
        let kept = run_events(&client, &all, run("set-1", set(vec![member("steps")])));
        let kept = collect(kept.await.unwrap()).await.unwrap();
        assert_eq!(outcome(&kept).status, "succeeded", "{kept:?}");
        let result: serde_json::Value = json(&outcome(&kept).result);
        assert_eq!(result["set"][0]["level"], "installed", "{result}");
        let frame = status_frame(&client, &all).await;
        let [shown] = &frame.warm[..] else {
            panic!("one member: {frame:?}")
        };
        assert_eq!(
            (shown.entrypoint.as_str(), shown.level, shown.holds.as_str()),
            ("steps", v1::WarmLevel::Host, "installed")
        );
        assert!(shown.held_back.contains("binds no model"), "{shown:?}");
        assert_eq!(frame.environments[0].level, "installed");
        // A member naming no entrypoint of its package refuses the whole set: the last one stays.
        let unknown = run_events(&client, &all, run("set-2", set(vec![member("absent")])));
        let unknown = collect(unknown.await.unwrap()).await.unwrap();
        assert_eq!(outcome(&unknown).status, "failed", "{unknown:?}");
        assert_eq!(status_frame(&client, &all).await.warm.len(), 1);
        // An empty set clears it.
        let cleared = run_events(&client, &all, run("set-3", set(vec![])));
        let cleared = collect(cleared.await.unwrap()).await.unwrap();
        assert_eq!(outcome(&cleared).status, "succeeded", "{cleared:?}");
        assert!(status_frame(&client, &all).await.warm.is_empty());
        let called = collect(
            run_events(&client, &all, run("call-1", local(v1::RunKind::Call)))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let done = outcome(&called);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["steps"], 2);
        // The id names that spec: another spec under it is refused.
        let mut other = local(v1::RunKind::Call);
        other.payload = v1::Json::parse(r#"{"steps":3}"#).ok();
        let conflict = run_events(&client, &all, run("call-1", other)).await.err().unwrap();
        assert_eq!((conflict.status(), conflict.code()), (Some(409), Some("run_id_conflict")));
        let _ = fs::remove_dir_all(tools);
    }

    /// H3 long-form's shape on CPU: a job (`kind: job`) renders each segment through a child run
    /// of its own invocable, handing it the job's file input; each child's file comes back into
    /// the job's spool, and the job publishes the film after every segment.
    #[tokio::test]
    async fn a_job_renders_its_segments_through_child_runs() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(
            &client,
            &all,
            "cpu_longform",
            "local/tensord-cpu-longform",
        )
        .await;
        let mut reference = vec![];
        {
            let mut encoder = png::Encoder::new(&mut reference, 2, 2);
            encoder.set_color(png::ColorType::Rgb);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[7; 12])
                .unwrap();
        }
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&reference));
        let length = reference.len() as u64;
        assert_eq!(
            write(&client, &all, &digest, length, 0, &reference)
                .await
                .unwrap(),
            length
        );
        let segments = ["a dawn", "a storm", "a calm"];
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "long_form".into(),
            payload: raw(&serde_json::json!({"reference": digest, "segments": segments})),
            inputs: vec![v1::InputFile {
                field: "reference".into(),
                digest: digest.clone(),
                length,
                media_type: "image/png".into(),
                order: 0,
            }],
            owner: "alice".into(),
            ..Default::default()
        };
        let events = collect(
            run_events(&client, &all, RunRequest {
                        id: "film".into(),
                        after: 0,
                        spec: Some(spec.clone()),
                    })
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["segments"], 3, "{result}");
        // What the job's executor measured comes with its outcome, as `run show` lists it: the
        // stage it bracketed and its steps, each with a count and a time.
        let measured: serde_json::Value = json(&done.measurements);
        let assemble = &measured["attribution"]["stages"]["assemble"];
        assert_eq!(assemble["count"], 3, "{measured}");
        assert!(assemble["total_ms"].as_f64().unwrap() > 0.0, "{measured}");
        let steps = &measured["attribution"]["steps"]["segments"];
        assert_eq!(steps["count"], 3, "{measured}");
        assert_eq!(steps["series"].as_array().unwrap().len(), 3, "{measured}");

        // Each segment saw the job's reference and continued the previous segment's context.
        let seen = tensorfs_core::sha256::hex_digest(&reference);
        let mut film = String::new();
        let mut context = String::new();
        for (index, prompt) in segments.iter().enumerate() {
            let body = format!("{index}|{prompt}|{seen}|{context}\n");
            context = tensorfs_core::sha256::hex_digest(body.as_bytes());
            film.push_str(&body);
        }
        // Each child run is a settled call in the job's log, with what its executor measured.
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Call(call) => Some(call.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls
                .iter()
                .map(|c| (c.index, c.function.as_str(), c.status.as_str()))
                .collect::<Vec<_>>(),
            (0..3)
                .map(|i| (i, "render_segment", "succeeded"))
                .collect::<Vec<_>>(),
            "{events:?}"
        );
        for call in &calls {
            assert!(call.run.ends_with(&format!("/{}", call.index)), "{call:?}");
            assert!(call.finished_at_ms >= call.called_at_ms && call.called_at_ms > 0, "{call:?}");
            assert!(call.reason.is_none(), "{call:?}");
            let measured = json(&call.measurements);
            assert_eq!(measured["models"], serde_json::json!([]), "a call with no model slot ran none");
        }
        // The film after every segment is a product of the job's `video` output.
        let products: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Product(p) if p.output == "video" => Some(p.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            products
                .iter()
                .map(|p| p.label.as_str())
                .collect::<Vec<_>>(),
            [
                "Video (segments 1-1)",
                "Video (segments 1-2)",
                "Video (segments 1-3)"
            ],
            "{events:?}"
        );
        let (meta, bytes) = read(
            &client,
            &all,
            ReadRequest {
                target: Some(Target::Output(v1::OutputTarget {
                    run: "film".into(),
                    output: "video".into(),
                    index: 0,
                    ..Default::default()
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(String::from_utf8(bytes).unwrap(), film, "{meta:?}");
        // A child's returned file is published by the job as it is (H3 shows each reference its
        // qwen-image-2 callee generates this way): one `rendered` product per segment, its bytes.
        let rendered: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Product(p) if p.output == "rendered" => Some(p.label.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            rendered,
            ["Segment 1 as rendered", "Segment 2 as rendered", "Segment 3 as rendered"],
            "{events:?}"
        );
        let bodies: Vec<&str> = film.split_inclusive('\n').collect();
        for (index, body) in bodies.iter().enumerate() {
            let (_, bytes) = read(
                &client,
                &all,
                ReadRequest {
                    target: Some(Target::Output(v1::OutputTarget {
                        run: "film".into(),
                        output: "rendered".into(),
                        // A list item's index is 1-based.
                        index: index as u32 + 1,
                        ..Default::default()
                    })),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            assert_eq!(String::from_utf8(bytes).unwrap(), *body, "rendered {index}");
        }

        // Every child is a run of its own under the job, settled.
        for index in 0..3 {
            let child = collect(
                run_events(&client, &all, RunRequest {
                            id: format!("film/{index}"),
                            after: 0,
                            spec: None,
                        })
                    .await
                    .unwrap(),
            )
            .await
            .unwrap();
            let child = outcome(&child);
            assert_eq!(child.status, "succeeded", "{child:?}");
            // A call's measurements name the process that executed it: on its GPUs' rows (a
            // Runtime before 0.28 names them `ranks`, as `run show` reads them).
            let measured: serde_json::Value = json(&child.measurements);
            let execution = &measured["execution"];
            let pid = execution["gpus"][0]["pid"].as_u64().or(execution["ranks"][0]["pid"].as_u64());
            assert!(pid > Some(0), "{measured}");
        }

        // A canceled job ends its running child with it.
        let mut held = spec.clone();
        held.payload = raw(&serde_json::json!({"reference": digest, "segments": segments, "hold": 60.0}));
        let stream = run_events(&client, &all, RunRequest {
                    id: "held".into(),
                    after: 0,
                    spec: Some(held),
                })
            .await
            .unwrap();
        let watch = |id: &str| RunRequest {
            id: id.into(),
            after: 0,
            spec: None,
        };
        let until = Instant::now() + Duration::from_secs(60);
        loop {
            let child = run_events(&client, &all, watch("held/0")).await;
            let running = match child {
                Ok(mut child) => match child.next().await.transpose() {
                    Ok(Some(v1::RunEvent {
                        event: v1::Event::State(s),
                        ..
                    })) => s.state == "running",
                    _ => false,
                },
                Err(_) => false,
            };
            if running {
                break;
            }
            assert!(Instant::now() < until, "the first child never ran");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        cancel(&client, &all, "held").await.unwrap();
        let parent = collect(stream).await.unwrap();
        assert_eq!(outcome(&parent).status, "canceled", "{parent:?}");
        let child = collect(
            run_events(&client, &all, watch("held/0"))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(outcome(&child).status, "canceled", "{child:?}");

        // A job's model choices address its callables (`<entrypoint>.models.<parameter>`), and
        // each child prepares with the ones addressed to it.
        let chose = |id: &str, parameter: &str| {
            let mut chosen = spec.clone();
            chosen.models = vec![v1::ModelChoice {
                parameter: parameter.into(),
                repository: "alice/model".into(),
                ..Default::default()
            }];
            RunRequest {
                id: id.into(),
                after: 0,
                spec: Some(chosen),
            }
        };
        let stray = collect(
            run_events(&client, &all, chose("stray", "nope.models.base"))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        // A choice naming no slot anywhere is a warning, never a refusal: the run goes on.
        let warned = stray.iter().any(|e| matches!(&e.event,
            v1::Event::Log(l) if l.level == "warning" && l.text.contains("nope.models.base")));
        let stray = outcome(&stray);
        assert_eq!(stray.status, "succeeded", "{stray:?}");
        assert!(warned, "the stray choice is not in the run's log as a warning");
        collect(
            run_events(&client, &all, chose("chosen", "render_segment.models.base"))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let child = collect(
            run_events(&client, &all, watch("chosen/0"))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        // render_segment declares no model slot, so the choice never reaches it: it runs.
        let child = outcome(&child);
        assert_eq!(child.status, "succeeded", "{child:?}");
        let _ = fs::remove_dir_all(tools);
    }

    /// The run's state as Run's first frame (a snapshot); None while it does not exist.
    async fn snapshot(
        client: &Client,
        cap: &str,
        id: &str,
    ) -> Option<v1::RunState> {
        let request = RunRequest {
            id: id.into(),
            after: 0,
            spec: None,
        };
        match run_events(client, cap, request).await {
            Ok(mut stream) => match stream.next().await.transpose() {
                Ok(Some(v1::RunEvent {
                    event: v1::Event::State(state),
                    ..
                })) => Some(state),
                _ => None,
            },
            Err(_) => None,
        }
    }

    /// Polls until the run shows `wanted`.
    async fn until(
        client: &Client,
        cap: &str,
        id: &str,
        wanted: &str,
    ) -> v1::RunState {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let seen = snapshot(client, cap, id).await;
            if let Some(state) = seen.as_ref().filter(|s| s.state == wanted) {
                return state.clone();
            }
            assert!(Instant::now() < deadline, "{id} never showed {wanted}: {seen:?}");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Cancel and submit again: a film canceled while its second segment runs ends with that
    /// segment, its first one finished. The same film submitted again (after a machine restart)
    /// finds that segment answered from the memo, held when it completed: nothing of it runs
    /// again, the rest run once, and the film is whole.
    #[tokio::test]
    async fn a_canceled_film_submitted_again_skips_its_finished_segments() {
        let (mut machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(
            &client,
            &all,
            "cpu_longform",
            "local/tensord-cpu-longform",
        )
        .await;
        let mut reference = vec![];
        {
            let mut encoder = png::Encoder::new(&mut reference, 2, 2);
            encoder.set_color(png::ColorType::Rgb);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[9; 12])
                .unwrap();
        }
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&reference));
        let length = reference.len() as u64;
        write(&client, &all, &digest, length, 0, &reference)
            .await
            .unwrap();
        let segments = ["a dawn", "a storm", "a calm"];
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "long_form".into(),
            payload: raw(&serde_json::json!({
                "reference": digest, "segments": segments, "hold": 4.0, "hold_at": 1})),
            inputs: vec![v1::InputFile {
                field: "reference".into(),
                digest: digest.clone(),
                length,
                media_type: "image/png".into(),
                order: 0,
            }],
            owner: "alice".into(),
            ..Default::default()
        };
        let submit = |id: &str| {
            authorized(
                RunRequest {
                    id: id.into(),
                    after: 0,
                    spec: Some(spec.clone()),
                },
                &all,
            )
        };
        drop(run_as(&client, submit("film")).await.unwrap());
        until(&client, &all, "film/1", "running").await;
        let canceled = cancel(&client, &all, "film")
            .await
            .unwrap();
        assert!(["running", "canceled"].contains(&canceled.state.as_str()), "{canceled:?}");
        until(&client, &all, "film", "canceled").await;
        until(&client, &all, "film/1", "canceled").await;
        assert_eq!(snapshot(&client, &all, "film/0").await.unwrap().state, "succeeded");
        assert!(snapshot(&client, &all, "film/2").await.is_none());

        // The memo is the machine's: a restart keeps it.
        machine.stop();
        let extra = installing_args(&tools);
        let (child, address) = launch(&machine.root, &extra);
        machine.child = child;
        machine.address = address;
        let client = self::client(&machine);
        let events = collect(run_as(&client, submit("again")).await.unwrap())
            .await
            .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        let result: serde_json::Value = json(&done.result);
        assert_eq!(result["segments"].as_u64(), Some(3), "{result}");
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Call(call) => Some((call.index, call.memoized)),
                _ => None,
            })
            .collect();
        assert_eq!(calls, [(0, true), (1, false), (2, false)], "only the finished segment is answered");
        for (index, memoized) in [(0, true), (1, false), (2, false)] {
            let attached = authorized(
                RunRequest { id: format!("again/{index}"), after: 0, spec: None },
                &all,
            );
            let child = collect(run_as(&client, attached).await.unwrap()).await.unwrap();
            let ended = outcome(&child);
            assert_eq!((ended.status.as_str(), ended.memoized), ("succeeded", memoized), "{ended:?}");
        }
        let seen = tensorfs_core::sha256::hex_digest(&reference);
        let (mut film, mut context) = (String::new(), String::new());
        for (index, prompt) in segments.iter().enumerate() {
            let body = format!("{index}|{prompt}|{seen}|{context}\n");
            context = tensorfs_core::sha256::hex_digest(body.as_bytes());
            film.push_str(&body);
        }
        let (_, bytes) = read(
            &client,
            &all,
            ReadRequest {
                target: Some(Target::Output(v1::OutputTarget {
                    run: "again".into(),
                    output: "video".into(),
                    index: 0,
                    ..Default::default()
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(String::from_utf8(bytes).unwrap(), film);
        let _ = fs::remove_dir_all(tools);
    }

    /// A job's root sleeps while a segment runs; H3's take minutes on a GPU and say nothing
    /// between denoise steps. That stillness is its child's work, so the job is not ended as
    /// wedged (run 3858 was, 85 s into its first segment; here the root's patience is 40 s).
    #[tokio::test]
    async fn a_job_waits_through_a_segment_longer_than_its_own_patience() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(
            &client,
            &all,
            "cpu_longform",
            "local/tensord-cpu-longform",
        )
        .await;
        let mut reference = vec![];
        {
            let mut encoder = png::Encoder::new(&mut reference, 2, 2);
            encoder.set_color(png::ColorType::Rgb);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[9; 12])
                .unwrap();
        }
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&reference));
        let length = reference.len() as u64;
        write(&client, &all, &digest, length, 0, &reference)
            .await
            .unwrap();
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "long_form".into(),
            payload: raw(&serde_json::json!({"reference": digest, "segments": ["a long night"], "hold": 50.0})),
            inputs: vec![v1::InputFile {
                field: "reference".into(),
                digest: digest.clone(),
                length,
                media_type: "image/png".into(),
                order: 0,
            }],
            owner: "alice".into(),
            ..Default::default()
        };
        let events = collect(
            run_events(&client, &all, RunRequest {
                        id: "patient".into(),
                        after: 0,
                        spec: Some(spec),
                    })
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "succeeded", "{done:?}");
        drop(machine);
        let _ = fs::remove_dir_all(tools);
    }

    /// H3 long-form's failure contract on CPU: a segment that fails in authored code fails its
    /// child run with that reason, the job fails with `segment_failed` naming the segment, and
    /// the film it published before the failure stays readable.
    #[tokio::test]
    async fn a_failed_segment_fails_the_job_and_keeps_the_film_so_far() {
        let (machine, tools) = installing_machine().await;
        let client = client(&machine);
        let all = cap(Grant {
            action: MACHINE.into(),
            ..Default::default()
        });
        let manifest = write_package(
            &client,
            &all,
            "cpu_longform",
            "local/tensord-cpu-longform",
        )
        .await;
        let mut reference = vec![];
        {
            let mut encoder = png::Encoder::new(&mut reference, 2, 2);
            encoder.set_color(png::ColorType::Rgb);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[9; 12])
                .unwrap();
        }
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(&reference));
        let length = reference.len() as u64;
        write(&client, &all, &digest, length, 0, &reference)
            .await
            .unwrap();
        let segments = ["a dawn", "a storm", "a calm"];
        let spec = v1::RunSpec {
            kind: v1::RunKind::Call,
            source: Some(v1::Source::Local(v1::LocalSource { manifest })),
            entrypoint: "long_form".into(),
            payload: raw(&serde_json::json!({"reference": digest, "segments": segments, "fail_at": 1})),
            inputs: vec![v1::InputFile {
                field: "reference".into(),
                digest: digest.clone(),
                length,
                media_type: "image/png".into(),
                order: 0,
            }],
            owner: "alice".into(),
            ..Default::default()
        };
        let events = collect(
            run_events(&client, &all, RunRequest {
                        id: "broken".into(),
                        after: 0,
                        spec: Some(spec),
                    })
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let done = outcome(&events);
        assert_eq!(done.status, "failed", "{done:?}");
        let reason = done.reason.clone().unwrap_or_default();
        assert_eq!(reason.code, "segment_failed", "{reason:?}");
        assert!(!reason.message.starts_with("segment_failed:"), "{reason:?}");
        assert!(reason.message.contains("segment 2 of 3"), "{reason:?}");
        // The child that failed is a failed run with the authored reason; the one before it
        // succeeded; no third segment ran.
        let watch = |id: &str| RunRequest {
            id: id.into(),
            after: 0,
            spec: None,
        };
        let first = collect(
            run_events(&client, &all, watch("broken/0"))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(outcome(&first).status, "succeeded", "{first:?}");
        let second = collect(
            run_events(&client, &all, watch("broken/1"))
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        let failed = outcome(&second);
        assert_eq!(failed.status, "failed", "{second:?}");
        let child_reason = failed.reason.as_ref().unwrap();
        assert_eq!(child_reason.code, "unhandled_exception");
        assert!(!child_reason.message.starts_with("unhandled_exception:"), "{child_reason:?}");
        assert!(
            format!("{:?}", failed.reason).contains("cannot be rendered"),
            "{failed:?}"
        );
        // No third segment ran.
        match run_events(&client, &all, watch("broken/2")).await {
            Err(_) => {}
            Ok(stream) => assert!(collect(stream).await.is_err()),
        }
        // The film through the first segment was published and stays readable.
        let labels: Vec<_> = events
            .iter()
            .filter_map(|e| match &e.event {
                v1::Event::Product(p) if p.output == "video" => {
                    Some(p.label.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(labels, ["Video (segments 1-1)"], "{events:?}");
        let seen = tensorfs_core::sha256::hex_digest(&reference);
        let (meta, bytes) = read(
            &client,
            &all,
            ReadRequest {
                target: Some(Target::Output(v1::OutputTarget {
                    run: "broken".into(),
                    output: "video".into(),
                    index: 0,
                    ..Default::default()
                })),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            format!("0|a dawn|{seen}|\n"),
            "{meta:?}"
        );
        let _ = fs::remove_dir_all(tools);
    }
}

/// Every file under `root`.
fn walkdir(root: &Path) -> Vec<std::path::PathBuf> {
    let mut found = vec![];
    for entry in fs::read_dir(root).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(walkdir(&path));
        } else {
            found.push(path);
        }
    }
    found
}
