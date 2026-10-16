//! Output export and the outputs route. The export tests serve Calcifer (`serve`, over real HTTP
//! and SSE on loopback) and run a real CPU package (`tests/fixtures/cpu_outputs`) on a real
//! TensorD (`COZY_TEST_TENSORD`), which installs it with its own installer: the client wheel its
//! binary embeds, over uv and Python 3.12, as on a pod.
mod common;

use std::{
    collections::BTreeSet,
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
    time::{Duration, Instant},
};

use calcifer::{
    Calcifer, Link,
    api::Api,
    serve::{Stopped, serve},
    store::{Batch, NewRun, Store},
};
use common::Tensord;
use cozy_platform::{lock::LockFile, paths::Home};
use cozy_proto::{
    calcifer::{Event, Holder, Kind, On, OutputItem, Run, StagedInput, Status, SubmitRun},
    config::{Config, Loaded},
    error::Envelope,
    ids::{Digest, RunId},
    machine as v1,
};
use prost::Message as _;
use reqwest::{Client, RequestBuilder, StatusCode, header};
use tokio::task::JoinHandle;

const BIG: u64 = 20 << 20;
/// A hang bound for a test whose awaited state never comes (a package's first install included);
/// the product waits without one.
const PATIENCE: Duration = Duration::from_secs(900);

/// `cozy calcifer` serving a home, and a client of its API.
struct Served {
    task: JoinHandle<Result<Stopped, cozy_proto::error::Error>>,
    holder: Holder,
    client: Client,
}

impl Served {
    async fn start(home: &Path) -> Self {
        let loaded = Loaded {
            config: Config::default(),
            home: Home::at(home),
            user_home: home.to_owned(),
            warnings: vec![],
        };
        let lock = LockFile::new(loaded.home.calcifer_lock());
        let task = tokio::spawn(async move { serve(&loaded).await });
        let holder = tokio::time::timeout(PATIENCE, async {
            loop {
                let held = lock.holder().unwrap();
                let held = held.and_then(|bytes| serde_json::from_slice::<Holder>(&bytes).ok());
                if let Some(holder) = held.filter(|h| h.pid == std::process::id()) {
                    return holder;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        });
        let holder = holder.await.expect("Calcifer never published its record");
        Self {
            task,
            holder,
            client: Client::new(),
        }
    }

    fn get(&self, path: &str) -> RequestBuilder {
        let url = format!("http://{}{path}", self.holder.address);
        self.client.get(url).bearer_auth(&self.holder.token)
    }

    async fn submit(&self, submit: &SubmitRun) -> Run {
        let url = format!("http://{}/v1/runs", self.holder.address);
        let request = self.client.post(url).bearer_auth(&self.holder.token);
        let response = request.header("idempotency-key", "k").json(submit);
        let response = response.send().await.unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let submitted: cozy_proto::calcifer::Submitted = response.json().await.unwrap();
        submitted.run
    }

    async fn run(&self, id: &RunId) -> Run {
        let response = self.get(&format!("/v1/runs/{id}")).send().await.unwrap();
        response.json().await.unwrap()
    }

    async fn events(&self, id: &RunId) -> Events {
        let path = format!("/v1/runs/{id}/events?starting_after=0");
        let response = self.get(&path).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        Events {
            response,
            buffer: String::new(),
        }
    }

    /// `POST /v1/down`; machines keep what they accepted.
    async fn down(self) {
        let url = format!("http://{}/v1/down", self.holder.address);
        let request = self.client.post(url).bearer_auth(&self.holder.token);
        assert_eq!(request.send().await.unwrap().status(), StatusCode::OK);
        let stopped = tokio::time::timeout(PATIENCE, self.task).await.unwrap();
        assert_eq!(stopped.unwrap().unwrap(), Stopped::Down);
    }
}

/// One run's SSE stream, parsed.
struct Events {
    response: reqwest::Response,
    buffer: String,
}

impl Events {
    /// The next event; none once the server ends the stream.
    async fn next(&mut self) -> Option<Event> {
        tokio::time::timeout(PATIENCE, async {
            loop {
                if let Some(end) = self.buffer.find("\n\n") {
                    let block: String = self.buffer.drain(..end + 2).collect();
                    // A block with no data is the heartbeat comment.
                    let data = block.lines().find_map(|line| line.strip_prefix("data:"));
                    if let Some(data) = data {
                        return Some(serde_json::from_str(data.trim()).unwrap());
                    }
                    continue;
                }
                let chunk = self.response.chunk().await.unwrap()?;
                self.buffer.push_str(std::str::from_utf8(&chunk).unwrap());
            }
        })
        .await
        .expect("the stream stalled")
    }

    /// Every event until `wanted` accepts one (included); with no `wanted`, to the stream's end.
    async fn until(&mut self, wanted: impl Fn(&Event) -> bool) -> Vec<Event> {
        let mut seen = vec![];
        while let Some(event) = self.next().await {
            let done = wanted(&event);
            seen.push(event);
            if done {
                break;
            }
        }
        seen
    }
}

fn landed(event: &Event) -> Option<&OutputItem> {
    match event {
        Event::OutputLanded { item, .. } => Some(item),
        _ => None,
    }
}

fn run_ok(command: &mut Command) {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A TensorD that installs local packages: its installer helper made as TensorD makes it on a
/// pod (a uv environment over Python 3.12 holding the client wheel its binary embeds).
fn installing_machine() -> Tensord {
    static TOOLS: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();
    let (python, wheel) = TOOLS.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap().keep();
        let tensord = std::env::var_os("COZY_TEST_TENSORD").expect("COZY_TEST_TENSORD");
        let wheel = client_wheel(Path::new(&tensord), &dir);
        let helper = dir.join("helper");
        run_ok(
            Command::new("uv")
                .args(["venv", "--no-project", "--no-config", "--python", "3.12"])
                .arg(&helper),
        );
        let python = helper.join("bin/python");
        run_ok(
            Command::new("uv")
                .args(["pip", "install", "--no-config", "--python"])
                .arg(&python)
                .arg(format!("{}[installer]", wheel.display())),
        );
        (python, wheel)
    });
    Tensord::start_args(&[
        OsStr::new("--installer-python"),
        python.as_os_str(),
        OsStr::new("--client-wheel"),
        wheel.as_os_str(),
    ])
}

/// The client wheel the TensorD binary embeds: the zip in it whose dist-info is
/// `tensord_client`'s, named as its dist-info says.
fn client_wheel(tensord: &Path, dir: &Path) -> PathBuf {
    let binary = std::fs::read(tensord).unwrap();
    let int = |at: usize, width: usize| {
        let bytes = binary.get(at..at + width)?;
        Some(bytes.iter().rev().fold(0usize, |n, b| n << 8 | *b as usize))
    };
    let zip_ends = binary
        .windows(4)
        .enumerate()
        .filter(|(_, w)| w == b"PK\x05\x06");
    for (end, _) in zip_ends {
        let (Some(size), Some(offset), Some(comment)) =
            (int(end + 12, 4), int(end + 16, 4), int(end + 20, 2))
        else {
            continue;
        };
        let Some(start) = end.checked_sub(size + offset) else {
            continue;
        };
        let mut entry = start + offset;
        let mut dist = None;
        while binary.get(entry..entry + 4) == Some(b"PK\x01\x02") {
            let (Some(name), Some(extra), Some(note)) =
                (int(entry + 28, 2), int(entry + 30, 2), int(entry + 32, 2))
            else {
                break;
            };
            let file = binary
                .get(entry + 46..entry + 46 + name)
                .unwrap_or_default();
            let file = std::str::from_utf8(file).unwrap_or_default();
            dist = dist.or(file.strip_suffix(".dist-info/WHEEL").map(str::to_owned));
            entry += 46 + name + extra + note;
        }
        if let Some(dist) = dist.filter(|d| d.starts_with("tensord_client-")) {
            let wheel = dir.join(format!("{dist}-py3-none-any.whl"));
            std::fs::write(&wheel, &binary[start..end + 22 + comment]).unwrap();
            return wheel;
        }
    }
    panic!("{} embeds no client wheel", tensord.display());
}

/// The fixture package as `cozy run ./dir` sends it (its files in a tar and the manifest naming
/// that, both staged with the run), calling its `make`.
fn package_run(dir: &Path, gate: &Path) -> SubmitRun {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpu_outputs");
    let source = dir.join("source.tar");
    run_ok(
        Command::new("tar")
            .args(["--format=ustar", "-cf"])
            .arg(&source)
            .arg("-C")
            .arg(&fixture)
            .args(["pyproject.toml", "package.toml", "cpu_outputs/__init__.py"]),
    );
    let tar = std::fs::read(&source).unwrap();
    let manifest = serde_json::json!({"package": "local/calcifer-cpu-outputs", "release": "0.1.0",
        "python_version": "3.12", "source": {"digest": Digest::of(&tar), "length": tar.len()}});
    let manifest = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(dir.join("manifest.json"), &manifest).unwrap();
    let staged = |path: PathBuf, bytes: &[u8]| StagedInput {
        path,
        digest: Digest::of(bytes),
        length: bytes.len() as u64,
    };
    let payload = serde_json::json!({"revisions": 3, "big": BIG, "gate": gate});
    let spec = v1::RunSpec {
        kind: v1::RunKind::Call as i32,
        source: Some(v1::run_spec::Source::Local(v1::LocalSource {
            manifest: Digest::of(&manifest).to_string(),
        })),
        entrypoint: "make".into(),
        payload: serde_json::to_vec(&payload).unwrap(),
        owner: "alice".into(),
        ..Default::default()
    };
    SubmitRun {
        on: On::Machine("m".into()),
        hub: String::new(),
        kind: Kind::Call,
        target: "local/calcifer-cpu-outputs/make".into(),
        spec: spec.encode_to_vec(),
        inputs: vec![
            staged(source, &tar),
            staged(dir.join("manifest.json"), &manifest),
        ],
        out: None,
        warnings: vec![],
    }
}

/// A home whose store knows the machine `m`.
fn with_machine(home: &Path, machine: &Tensord) {
    let key = common::key_file(home);
    let store = Store::open(&Home::at(home).calcifer_db()).unwrap();
    store.put_machine(&machine.row("m", &key)).unwrap();
}

/// The output's bytes as its machine holds them now, read with a link of the test's.
async fn on_machine(machine: &Tensord, run: &RunId, name: &str, index: u32) -> Vec<u8> {
    let link = Link::new(&machine.endpoint, common::key()).unwrap();
    let target = v1::OutputTarget {
        run: run.to_string(),
        output: name.into(),
        index,
        member: String::new(),
    };
    let (_, mut frames) = link.read(target, 0, 0, 0).await.unwrap();
    let mut bytes = vec![];
    while let Some(frame) = frames.message().await.unwrap() {
        bytes.extend(frame.data);
    }
    bytes
}

/// What a directory holds, by name.
fn listed(dir: &Path) -> BTreeSet<String> {
    let names = std::fs::read_dir(dir).unwrap();
    names
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

/// The run completed once, and each of its outputs landed whole: its stream said so at the
/// item's final revision, its file is the machine's final bytes, and nothing else lands beside.
async fn landed_whole(served: &Served, machine: &Tensord, id: &RunId, events: &[Event]) {
    assert_eq!(events.iter().filter(|e| e.is_terminal()).count(), 1);
    let run = served.run(id).await;
    assert_eq!(run.status, Status::Completed, "{run:?}");
    let mut names = BTreeSet::new();
    for item in &run.output {
        let said = events
            .iter()
            .filter_map(landed)
            .find(|i| i.id == item.id && i.rev == item.rev);
        assert_eq!(
            said.and_then(|i| i.path.clone()),
            item.path,
            "{item:?} landed"
        );
        let path = item.path.clone().unwrap();
        names.insert(path.file_name().unwrap().to_string_lossy().into_owned());
        if item.media_type == "application/vnd.cozy.tree-manifest" {
            let files = [
                ("nested/a.txt", "same"),
                ("b.txt", "same"),
                ("c.txt", "other"),
            ];
            for (member, text) in files {
                assert_eq!(std::fs::read_to_string(path.join(member)).unwrap(), text);
            }
            assert_eq!(listed(&path).len(), 3, "{:?}", listed(&path));
            continue;
        }
        let held = std::fs::read(&path).unwrap();
        assert_eq!(Some(Digest::of(&held)), item.sha256, "{item:?}");
        let machine_holds = on_machine(machine, id, &item.name, item.index).await;
        assert!(held == machine_holds, "{item:?}");
    }
    let n = run.number;
    for name in [
        "video.mp4",
        "frames-1.mp4",
        "frames-2.mp4",
        "frames-3.mp4",
        "big.mp4",
        "tree",
    ] {
        assert!(
            names.contains(&format!("{n}-{name}")),
            "{name} in {names:?}"
        );
    }
    let out = run.output[0].path.as_ref().unwrap().parent().unwrap();
    assert_eq!(listed(out), names, "nothing else lands beside them");
}

fn product(output: &str, index: u32, rev: u64, bytes: &[u8]) -> v1::Product {
    v1::Product {
        output: output.into(),
        index,
        rev,
        length: bytes.len() as u64,
        digest: Digest::of(bytes).to_string(),
        media_type: "video/mp4".into(),
        ..Default::default()
    }
}

async fn code(response: reqwest::Response) -> (StatusCode, String) {
    let status = response.status();
    let envelope: Envelope = response.json().await.unwrap();
    (status, envelope.error.code)
}

/// A settled run whose machine never got an endpoint: its outputs cannot land. The route answers
/// honestly before an output's file holds a revision and serves it whole or by one range once it
/// does; an output whose machine ended is given up with that reason, across a reopen.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_route_serves_a_landed_output_and_says_why_another_has_not_landed() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let out = home.path().join("outputs");
    let db = home.path().join("calcifer/state.db");
    let store = Store::open(&db).unwrap();
    store
        .put_machine(&common::machine("m", None, &key))
        .unwrap();
    let id = RunId::mint([5; 12]);
    let new = NewRun {
        id: id.clone(),
        kind: Kind::Call,
        target: "local/calcifer-cpu-outputs/make".into(),
        hub: String::new(),
        machine: Some("m".into()),
        pinned: true,
        spec: common::warm_spec("https://tensorhub.com", vec![]),
        out_dir: out.clone(),
        warnings: vec![],
        inputs: vec![],
    };
    store.submit("k", "sha256:k", new).unwrap();
    assert!(store.mark_sent(id.as_str(), "m").unwrap());
    let state = v1::RunState {
        id: id.to_string(),
        number: 1,
        state: "running".into(),
        ..Default::default()
    };
    store.accept(id.as_str(), "m", &state).unwrap();
    let batch = Batch {
        cursor: 1 << 32,
        products: vec![product("video", 0, 1, b"segment 1\n")],
        ..Default::default()
    };
    store.observe(id.as_str(), batch).unwrap();
    let video = b"segment 1\nsegment 2\n";
    let outcome = v1::Outcome {
        status: "succeeded".into(),
        outputs: vec![
            product("video", 0, 2, video),
            product("frames", 1, 1, b"frame 1\n"),
        ],
        ..Default::default()
    };
    store.settle(id.as_str(), &outcome, 2 << 32).unwrap();
    drop(store);

    let calcifer = Calcifer::open(&home.path().join("calcifer"), &out).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = axum::Router::new()
        .route(
            "/v1/runs/{id}/outputs/{name}",
            axum::routing::get(calcifer::outputs::serve),
        )
        .with_state(Api::new(calcifer.clone(), address.port(), "token"));
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let client = Client::new();
    let get = |path: &str, range: Option<&str>| {
        let request = client.get(format!("http://{address}{path}"));
        match range {
            Some(range) => request.header(header::RANGE, range),
            None => request,
        }
        .send()
    };
    let early = get("/v1/runs/1/outputs/video", None).await.unwrap();
    assert_eq!(
        code(early).await,
        (StatusCode::CONFLICT, "output_not_ready".into())
    );
    let unknown = get("/v1/runs/1/outputs/nothing", None).await.unwrap();
    assert_eq!(code(unknown).await.1, "output_not_found");
    assert_eq!(
        code(get("/v1/runs/9/outputs/video", None).await.unwrap())
            .await
            .1,
        "run_not_found"
    );
    assert_eq!(calcifer.owed(), ["outputs of run 1"]);

    // The bytes land as the exporter lands them: the file, then its record.
    let path = out.join(calcifer::outputs::file_name(1, "video", 0, "video/mp4"));
    assert_eq!(path.file_name().unwrap(), "1-video.mp4");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(&path, video).unwrap();
    let said = calcifer
        .store()
        .exported(id.as_str(), "video", 0, 2, &path)
        .unwrap();
    assert!(
        matches!(&said[..], [Event::OutputLanded { item, .. }] if item.path.as_ref() == Some(&path))
    );
    let whole = get(&format!("/v1/runs/{id}/outputs/video"), None)
        .await
        .unwrap();
    assert_eq!(whole.status(), StatusCode::OK);
    assert_eq!(whole.headers()[header::ETAG], "\"r2\"");
    assert_eq!(whole.headers()[header::CONTENT_TYPE], "video/mp4");
    assert_eq!(
        whole.headers()["x-cozy-digest"],
        Digest::of(video).to_string().as_str()
    );
    assert_eq!(whole.bytes().await.unwrap(), &video[..]);
    let part = get("/v1/runs/1/outputs/video", Some("bytes=10-"))
        .await
        .unwrap();
    assert_eq!(part.status(), StatusCode::PARTIAL_CONTENT);
    assert_eq!(part.headers()[header::CONTENT_RANGE], "bytes 10-19/20");
    assert_eq!(part.bytes().await.unwrap(), &b"segment 2\n"[..]);
    let tail = get("/v1/runs/1/outputs/video", Some("bytes=-3"))
        .await
        .unwrap();
    assert_eq!(tail.bytes().await.unwrap(), &video[17..]);
    for refused in ["bytes=0-1,4-5", "bytes=20-", "bytes=-0", "bytes=5-2"] {
        let answer = get("/v1/runs/1/outputs/video", Some(refused))
            .await
            .unwrap();
        assert_eq!(
            answer.status(),
            StatusCode::RANGE_NOT_SATISFIABLE,
            "{refused}"
        );
        assert_eq!(
            answer.headers()[header::CONTENT_RANGE],
            "bytes */20",
            "{refused}"
        );
    }

    calcifer.machine_ended("m", "its rental ended").unwrap();
    let until = Instant::now() + PATIENCE;
    while !calcifer.owed().is_empty() {
        assert!(Instant::now() < until, "the list item is never given up");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let lost = get("/v1/runs/1/outputs/frames?index=1", None)
        .await
        .unwrap();
    assert_eq!(
        code(lost).await,
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "output_not_exported".into()
        )
    );
    calcifer.close();
    drop(calcifer);

    let reopened = Calcifer::open(&home.path().join("calcifer"), &out).unwrap();
    assert!(reopened.owed().is_empty());
    let replay = reopened.store().events_after(id.as_str(), 0).unwrap();
    let video = replay.iter().filter_map(landed).find(|i| i.name == "video");
    assert_eq!(video.and_then(|i| i.path.as_deref()), Some(path.as_path()));
    let frames = replay.iter().find_map(|e| match e {
        Event::OutputDone { item, .. } if item.name == "frames" => Some(item),
        _ => None,
    });
    let reason = frames.and_then(|i| i.export_error.as_deref()).unwrap();
    assert!(reason.contains("its rental ended"), "{reason}");
    reopened.close();
}

/// A real run's outputs land as they are published, each file rewritten in place by its newer
/// revisions, a list as one file per item, a big file read in parallel ranges and a tree as a
/// folder. The run's stream says when each lands and ends once all have; the route serves them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_runs_outputs_land_as_published_and_are_rewritten_in_place() {
    let home = tempfile::tempdir().unwrap();
    let machine = installing_machine();
    with_machine(home.path(), &machine);
    let served = Served::start(home.path()).await;
    let gate = machine.root.path().join("gate");
    let run = served.submit(&package_run(home.path(), &gate)).await;
    let (id, n) = (run.id, run.number);
    let mut events = served.events(&id).await;
    let mut seen = events
        .until(|e| landed(e).is_some_and(|i| i.name == "video"))
        .await;
    let first = landed(seen.last().unwrap()).unwrap().clone();
    assert_eq!(first.rev, 1, "the first revision lands while the run waits");
    let video = first.path.unwrap();
    assert_eq!(std::fs::read(&video).unwrap(), b"segment 1\n");
    std::fs::write(&gate, b"").unwrap();
    seen.extend(events.until(|_| false).await);
    landed_whole(&served, &machine, &id, &seen).await;
    let rev = served
        .run(&id)
        .await
        .output
        .iter()
        .find(|i| i.name == "video")
        .unwrap()
        .rev;
    assert!(rev >= 3, "rewritten in place: revision {rev}");
    assert_eq!(
        std::fs::read(&video).unwrap(),
        b"segment 1\nsegment 2\nsegment 3\n"
    );

    let whole = served
        .get(&format!("/v1/runs/{n}/outputs/video"))
        .send()
        .await
        .unwrap();
    assert_eq!(whole.status(), StatusCode::OK);
    assert_eq!(
        whole.headers()[header::ETAG],
        format!("\"r{rev}\"").as_str()
    );
    assert_eq!(whole.headers()[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
    assert_eq!(whole.bytes().await.unwrap(), std::fs::read(&video).unwrap());
    let big = std::fs::read(video.with_file_name(format!("{n}-big.mp4"))).unwrap();
    assert_eq!(big.len() as u64, BIG);
    let tail = served
        .get(&format!("/v1/runs/{id}/outputs/big"))
        .header(header::RANGE, "bytes=-5");
    let tail = tail.send().await.unwrap();
    assert_eq!(tail.status(), StatusCode::PARTIAL_CONTENT);
    assert_eq!(tail.bytes().await.unwrap(), &big[big.len() - 5..]);
    let item = served
        .get(&format!("/v1/runs/{n}/outputs/frames?index=2"))
        .send()
        .await
        .unwrap();
    assert_eq!(item.bytes().await.unwrap(), &b"frame 2\n"[..]);
    let tree = served
        .get(&format!("/v1/runs/{n}/outputs/tree"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        code(tree).await,
        (StatusCode::CONFLICT, "output_is_a_tree".into())
    );
    served.down().await;
}

/// Calcifer stops (`down`) mid-run, after the run's first revision landed; the run goes on
/// without it; the next Calcifer follows it to its end and lands its outputs once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_completes_and_its_outputs_land_once_across_a_calcifer_restart() {
    let home = tempfile::tempdir().unwrap();
    let machine = installing_machine();
    with_machine(home.path(), &machine);
    let first = Served::start(home.path()).await;
    let gate = machine.root.path().join("gate");
    let id = first.submit(&package_run(home.path(), &gate)).await.id;
    let mut events = first.events(&id).await;
    events
        .until(|e| landed(e).is_some_and(|i| i.name == "video"))
        .await;
    drop(events);
    first.down().await;
    std::fs::write(&gate, b"").unwrap();

    let second = Served::start(home.path()).await;
    let all = second.events(&id).await.until(|_| false).await;
    landed_whole(&second, &machine, &id, &all).await;
    second.down().await;
}
