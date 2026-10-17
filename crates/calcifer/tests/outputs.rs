//! Output export and the outputs route. The export tests serve Calcifer (`serve`, over real HTTP
//! and SSE on loopback) and run a real CPU package (`tests/fixtures/cpu_outputs`) on a real
//! TensorD (`COZY_TEST_TENSORD`), which installs it with its own installer: the client wheel its
//! binary embeds, over uv and Python 3.12, as on a pod.
mod common;

use std::{
    collections::BTreeSet,
    path::Path,
    time::{Duration, Instant},
};

use calcifer::{
    Calcifer, Link,
    api::Api,
    machines::Machines,
    serve::{Stopped, serve},
    store::{Batch, NewRun, Store},
};
use common::{Tensord, installing_machine, package_run};
use cozy_platform::{lock::LockFile, paths::Home};
use cozy_proto::{
    calcifer::{
        CHECKPOINT, Event, Holder, Kind, MachinesEvent, OutputItem, Run, RunsEvent, Status,
        SubmitRun,
    },
    config::{Config, Loaded},
    error::Envelope,
    ids::{Digest, RunId},
    machine as v1,
};
use futures_util::StreamExt as _;
use reqwest::{Client, RequestBuilder, StatusCode, header};
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    task::JoinHandle,
};

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
    /// Starts `serve` as the CLI does, and waits for its first answer.
    async fn start(home: &Path) -> Self {
        let loaded = Loaded {
            config: Config::default(),
            home: Home::at(home),
            user_home: home.to_owned(),
            warnings: vec![],
        };
        let lock = LockFile::new(loaded.home.calcifer_lock());
        let task = tokio::spawn(async move { serve(&loaded).await });
        let until = Instant::now() + PATIENCE;
        let holder = loop {
            let held = lock.holder().unwrap();
            let held = held.and_then(|bytes| serde_json::from_slice::<Holder>(&bytes).ok());
            if let Some(holder) = held.filter(|h| h.pid == std::process::id()) {
                break holder;
            }
            if task.is_finished() {
                panic!(
                    "Calcifer stopped before it published its record: {:?}",
                    task.await
                );
            }
            assert!(
                Instant::now() < until,
                "Calcifer never published its record"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        let served = Self {
            task,
            holder,
            client: Client::new(),
        };
        let first = served.get("/v1/runs").send().await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        served
    }

    fn get(&self, path: &str) -> RequestBuilder {
        let url = format!("http://{}{path}", self.holder.address);
        self.client.get(url).bearer_auth(&self.holder.token)
    }

    async fn submit(&self, submit: &SubmitRun) -> Run {
        self.submit_as("k", submit).await
    }

    async fn submit_as(&self, key: &str, submit: &SubmitRun) -> Run {
        let url = format!("http://{}/v1/runs", self.holder.address);
        let request = self.client.post(url).bearer_auth(&self.holder.token);
        let response = request.header("idempotency-key", key).json(submit);
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
        Events::of(response)
    }

    /// A list-level stream (`/v1/runs/events`, `/v1/machines/events`), subscribed now.
    async fn feed<T: serde::de::DeserializeOwned>(&self, path: &str) -> Events<T> {
        let response = self.get(path).send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        Events::of(response)
    }

    /// `DELETE /v1/machines/{name}`, as `cozy machine end [--force]` asks it.
    async fn end(&self, name: &str, force: bool) -> (StatusCode, serde_json::Value) {
        let url = format!(
            "http://{}/v1/machines/{name}?force={force}",
            self.holder.address
        );
        let request = self.client.delete(url).bearer_auth(&self.holder.token);
        let response = request.send().await.unwrap();
        (response.status(), response.json().await.unwrap())
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

/// One SSE stream, parsed: a run's events, or a list-level stream's.
struct Events<T = Event> {
    response: reqwest::Response,
    buffer: String,
    of: std::marker::PhantomData<T>,
}

impl<T: serde::de::DeserializeOwned> Events<T> {
    fn of(response: reqwest::Response) -> Self {
        Self {
            response,
            buffer: String::new(),
            of: std::marker::PhantomData,
        }
    }

    /// The next event; none once the server ends the stream.
    async fn next(&mut self) -> Option<T> {
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
    async fn until(&mut self, wanted: impl Fn(&T) -> bool) -> Vec<T> {
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
    let mut output = link.bulk().unwrap().get(&target, (0, 0), 0).await.unwrap();
    let mut bytes = vec![];
    while let Some(data) = output.body.next().await {
        bytes.extend(data.unwrap());
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
        wants: None,
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
    calcifer.resume().unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = axum::Router::new()
        .route(
            "/v1/runs/{id}/outputs/{name}",
            axum::routing::get(calcifer::outputs::serve),
        )
        .with_state(Api::new(
            calcifer.clone(),
            Machines::start(calcifer.clone(), Home::at(home.path()), &Default::default()),
            address.port(),
            "token",
        ));
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
    reopened.resume().unwrap();
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
    let run = served.submit(&package_run(home.path(), &gate, BIG)).await;
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
    let id = first.submit(&package_run(home.path(), &gate, BIG)).await.id;
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

/// `GET /v1/machines/{name}?fresh=true` answers the machine as it is now: the environment a warm
/// run installed is there the moment the run settles, with no wait for a Status tick. A machine
/// that cannot answer is refused, never shown stale as fresh.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fresh_machine_read_shows_what_a_settled_warm_run_installed() {
    let home = tempfile::tempdir().unwrap();
    let mut machine = installing_machine();
    with_machine(home.path(), &machine);
    let served = Served::start(home.path()).await;
    let mut warm = package_run(home.path(), &home.path().join("gate"), BIG);
    warm.spec.kind = v1::RunKind::Warm;
    warm.spec.payload = None;
    warm.kind = Kind::Warm;
    let run = served.submit(&warm).await;
    let mut events = served.events(&run.id).await;
    events.until(|e| e.is_terminal()).await;
    assert_eq!(served.run(&run.id).await.status, Status::Completed);

    let fresh = served
        .get("/v1/machines/m?fresh=true")
        .send()
        .await
        .unwrap();
    assert_eq!(fresh.status(), StatusCode::OK);
    let fresh: cozy_proto::calcifer::Machine = fresh.json().await.unwrap();
    let status = fresh.status.expect("a Status read now");
    assert!(
        status
            .environments
            .iter()
            .any(|env| env.package.ends_with("calcifer-cpu-outputs")),
        "{status:?}"
    );
    machine.stop();
    let gone = served
        .get("/v1/machines/m?fresh=true")
        .send()
        .await
        .unwrap();
    assert_eq!(
        code(gone).await,
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "machine_unavailable".into()
        )
    );
    served.down().await;
}

/// A TCP relay to `to` that carries each connection's bytes back at `rate` bytes a second: a
/// slow link between Calcifer and its machine (TLS stays end to end).
async fn throttled(to: String, rate: f64) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        while let Ok((inbound, _)) = listener.accept().await {
            let to = to.clone();
            tokio::spawn(async move {
                let Ok(outbound) = tokio::net::TcpStream::connect(&to).await else {
                    return;
                };
                let (mut from_calcifer, mut to_calcifer) = inbound.into_split();
                let (mut from_machine, mut to_machine) = outbound.into_split();
                let up = tokio::io::copy(&mut from_calcifer, &mut to_machine);
                let down = async {
                    let (began, mut sent, mut buffer) = (Instant::now(), 0f64, vec![0; 16 << 10]);
                    while let Ok(n @ 1..) = from_machine.read(&mut buffer).await {
                        if to_calcifer.write_all(&buffer[..n]).await.is_err() {
                            return;
                        }
                        sent += n as f64;
                        let due = began + Duration::from_secs_f64(sent / rate);
                        tokio::time::sleep_until(due.into()).await;
                    }
                };
                tokio::select! {
                    _ = up => {}
                    () = down => {}
                }
            });
        }
    });
    address
}

/// `cozy machine end` waits while a run's outputs are still on their way over a slow link,
/// saying how much is in place, and ends the machine once they landed; `--force` ends it at
/// once, names what it gives up, and the export records it lost without leaving a part file.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ending_a_machine_waits_for_its_runs_outputs_and_force_gives_them_up() {
    let home = tempfile::tempdir().unwrap();
    let machine = installing_machine();
    let relay = throttled(machine.endpoint.address.clone(), f64::from(1u32 << 20)).await;
    let key = common::key_file(home.path());
    let store = Store::open(&Home::at(home.path()).calcifer_db()).unwrap();
    for name in ["m", "n"] {
        let mut row = machine.row(name, &key);
        row.endpoint.as_mut().unwrap().address = relay.clone();
        store.put_machine(&row).unwrap();
    }
    drop(store);
    let served = Served::start(home.path()).await;
    let gate = home.path().join("gate");
    std::fs::write(&gate, b"").unwrap();

    // Ending waits until the outputs landed, then ends.
    let run = served
        .submit(&package_run(home.path(), &gate, 12 << 20))
        .await;
    let mut events = served.events(&run.id).await;
    events.until(Event::is_terminal).await;
    let (status, refused) = served.end("m", false).await;
    assert_eq!(status, StatusCode::CONFLICT, "{refused}");
    assert_eq!(refused["error"]["code"], "outputs_exporting", "{refused}");
    assert!(
        refused["error"]["message"]
            .as_str()
            .unwrap()
            .contains("in place"),
        "{refused}"
    );
    let until = Instant::now() + PATIENCE;
    let ended = loop {
        let (status, answer) = served.end("m", false).await;
        if status == StatusCode::OK {
            break answer;
        }
        assert_eq!(answer["error"]["code"], "outputs_exporting", "{answer}");
        assert!(Instant::now() < until, "the outputs never landed");
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    assert_eq!(ended["changed"], true, "{ended}");
    assert!(ended.get("abandoned").is_none(), "{ended}");
    let detail = served.run(&run.id).await;
    let big = (detail.output.iter()).find(|o| o.name == "big");
    let path = big.and_then(|big| big.path.clone());
    let path = path.unwrap_or_else(|| panic!("{detail:?}"));
    assert_eq!(std::fs::metadata(path).unwrap().len(), 12 << 20);

    // `force` ends it mid-export: the output is given up, recorded lost, and no part remains.
    let out = home.path().join("forced");
    let mut forced = package_run(home.path(), &gate, 64 << 20);
    forced.on = cozy_proto::calcifer::On::Machine("n".into());
    forced.out = Some(out.clone());
    let run = served.submit_as("forced", &forced).await;
    let mut events = served.events(&run.id).await;
    events.until(Event::is_terminal).await;
    let until = Instant::now() + PATIENCE;
    loop {
        let (status, answer) = served.end("n", false).await;
        assert_eq!(status, StatusCode::CONFLICT, "{answer}");
        let said = answer["error"]["message"].as_str().unwrap();
        if !said.contains(": 0 B of") {
            break;
        }
        assert!(Instant::now() < until, "the export never moved: {said}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let (status, ended) = served.end("n", true).await;
    assert_eq!(status, StatusCode::OK, "{ended}");
    let abandoned = ended["abandoned"].as_array().unwrap();
    assert!(
        abandoned
            .iter()
            .any(|a| a.as_str().unwrap().contains(" big ")),
        "{ended}"
    );
    let until = Instant::now() + PATIENCE;
    let lost = loop {
        let detail = served.run(&run.id).await;
        let big = (detail.output.into_iter())
            .find(|o| o.name == "big")
            .unwrap();
        if big.export_error.is_some() || big.path.is_some() {
            break big;
        }
        assert!(Instant::now() < until, "the forced export never settled");
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    assert!(lost.path.is_none(), "{lost:?}");
    let error = lost.export_error.as_deref().unwrap_or_default();
    assert!(error.contains("its machine n ended"), "{lost:?}");
    let left: Vec<_> = std::fs::read_dir(&out)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        !left
            .iter()
            .any(|name| name.ends_with(".part") || name.contains("big")),
        "{left:?}"
    );
    served.down().await;
}

/// `GET /v1/runs/events` carries each run whole as it changes: a local package's run is seen
/// queued, in progress (held at its gate), then completed with its outputs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_runs_stream_carries_a_run_whole_from_queued_to_completed() {
    let home = tempfile::tempdir().unwrap();
    let machine = installing_machine();
    with_machine(home.path(), &machine);
    let served = Served::start(home.path()).await;
    let mut runs: Events<RunsEvent> = served.feed("/v1/runs/events").await;
    let gate = machine.root.path().join("gate");
    let run = served
        .submit(&package_run(home.path(), &gate, 1 << 20))
        .await;
    let status = |event: &RunsEvent| match event {
        RunsEvent::Updated { run: seen } if seen.id == run.id => Some(seen.status),
        _ => None,
    };
    let started = runs.until(|e| status(e) == Some(Status::InProgress)).await;
    assert!(
        started.iter().any(|e| status(e) == Some(Status::Queued)),
        "{started:?}"
    );
    std::fs::write(&gate, b"").unwrap();
    let done = runs.until(|e| status(e) == Some(Status::Completed)).await;
    let RunsEvent::Updated { run: completed } = done.last().unwrap();
    assert!(!completed.output.is_empty(), "the run whole: {completed:?}");
    served.down().await;
}

/// `GET /v1/machines/events` says when a machine's row changes: a run landing on it and settling
/// (its counts), what its Status reports (the run's environment), and its removal once the run's
/// outputs have landed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_machines_stream_says_when_a_machine_changes_and_when_it_is_removed() {
    let home = tempfile::tempdir().unwrap();
    let machine = installing_machine();
    with_machine(home.path(), &machine);
    let served = Served::start(home.path()).await;
    let mut machines: Events<MachinesEvent> = served.feed("/v1/machines/events").await;
    let gate = home.path().join("gate");
    std::fs::write(&gate, b"").unwrap();
    let run = served
        .submit(&package_run(home.path(), &gate, 1 << 20))
        .await;
    let (mut busy, mut settled, mut installed) = (false, false, false);
    while !(settled && installed) {
        let event = machines.next().await.expect("the stream ended first");
        let MachinesEvent::Updated { machine: m } = &event else {
            panic!("only updates while the machine lives: {event:?}");
        };
        assert_eq!(m.name, "m");
        match m.running + m.queued {
            0 => settled |= busy,
            _ => busy = true,
        }
        installed |= m
            .status
            .as_ref()
            .is_some_and(|s| !s.environments.is_empty());
    }
    let mut events = served.events(&run.id).await;
    events.until(Event::is_terminal).await;
    let until = Instant::now() + PATIENCE;
    while served.end("m", false).await.0 != StatusCode::OK {
        assert!(Instant::now() < until, "the machine never ended");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let removed = machines
        .until(|e| matches!(e, MachinesEvent::Removed { name } if name == "m"))
        .await;
    assert!(
        matches!(removed.last(), Some(MachinesEvent::Removed { .. })),
        "{removed:?}"
    );
    served.down().await;
}

/// Every output a run makes ends in one terminal state: a file, a list's items and a tree land
/// here; a checkpoint stays on its machine and is final at once, so it holds up no follower and no
/// `machine end`; an output its machine refuses once the run has ended is recorded lost, never
/// asked for again and again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_output_ends_landed_kept_or_lost_and_a_checkpoint_waits_on_nothing() {
    let home = tempfile::tempdir().unwrap();
    let mut machine = installing_machine();
    with_machine(home.path(), &machine);
    // A settled run on a machine nothing can be read from, whose output is a checkpoint.
    let key = common::key_file(home.path());
    let store = Store::open(&Home::at(home.path()).calcifer_db()).unwrap();
    store
        .put_machine(&common::machine("c", None, &key))
        .unwrap();
    let kept = RunId::mint([6; 12]);
    let new = NewRun {
        id: kept.clone(),
        kind: Kind::Call,
        target: "quantize".into(),
        hub: String::new(),
        machine: Some("c".into()),
        pinned: true,
        spec: common::warm_spec("https://tensorhub.com", vec![]),
        out_dir: home.path().join("kept"),
        warnings: vec![],
        inputs: vec![],
        wants: None,
    };
    store.submit("kept", "sha256:kept", new).unwrap();
    assert!(store.mark_sent(kept.as_str(), "c").unwrap());
    let state = v1::RunState {
        id: kept.to_string(),
        number: 1,
        state: "running".into(),
        ..Default::default()
    };
    store.accept(kept.as_str(), "c", &state).unwrap();
    let outcome = v1::Outcome {
        status: "succeeded".into(),
        outputs: vec![v1::Product {
            media_type: CHECKPOINT.into(),
            ..product("fp8", 0, 1, br#"{"entries":[]}"#)
        }],
        ..Default::default()
    };
    store.settle(kept.as_str(), &outcome, 1 << 32).unwrap();
    drop(store);

    let served = Served::start(home.path()).await;
    let gate = home.path().join("gate");
    std::fs::write(&gate, b"").unwrap();
    let made = served
        .submit_as("made", &package_run(home.path(), &gate, 1 << 20))
        .await;
    let until = Instant::now() + PATIENCE;
    let landed = loop {
        let run = served.run(&made.id).await;
        if run.status == Status::Completed && run.output.iter().all(|o| o.path.is_some()) {
            break run;
        }
        assert!(Instant::now() < until, "{run:?}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    let kinds: BTreeSet<&str> = landed.output.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(kinds, BTreeSet::from(["big", "frames", "tree", "video"]));

    let run = served.run(&kept).await;
    let fp8 = &run.output[0];
    assert!(
        fp8.is_checkpoint() && fp8.path.is_none() && fp8.export_error.is_none(),
        "{fp8:?}"
    );
    let route = served.get(&format!("/v1/runs/{kept}/outputs/fp8")).send();
    assert_eq!(
        code(route.await.unwrap()).await,
        (StatusCode::CONFLICT, "output_is_a_checkpoint".into())
    );
    let (status, ended) = served.end("c", false).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "a checkpoint holds up no end: {ended}"
    );

    // `big` is owed again while its machine restarts after the run ended: the export waits for it,
    // never giving the output up, and lands once it serves again.
    served.down().await;
    let db = rusqlite::Connection::open(Home::at(home.path()).calcifer_db()).unwrap();
    let owed = db
        .execute(
            "UPDATE outputs SET exported_rev = NULL, path = NULL WHERE run = ?1 AND name = 'big'",
            [made.id.as_str()],
        )
        .unwrap();
    assert_eq!(owed, 1);
    drop(db);
    machine.stop();
    let served = Served::start(home.path()).await;
    tokio::time::sleep(Duration::from_secs(3)).await;
    let big = |run: Run| run.output.into_iter().find(|o| o.name == "big").unwrap();
    let waiting = big(served.run(&made.id).await);
    assert!(
        waiting.path.is_none() && waiting.export_error.is_none(),
        "{waiting:?}"
    );
    machine.start_again();
    let until = Instant::now() + PATIENCE;
    loop {
        let now = big(served.run(&made.id).await);
        assert!(now.export_error.is_none(), "{now:?}");
        if now.path.is_some() {
            break;
        }
        assert!(Instant::now() < until, "{now:?}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // The store says `big` is at a revision its machine never had: once the run has ended, the
    // machine's refusal is final.
    served.down().await;
    let db = rusqlite::Connection::open(Home::at(home.path()).calcifer_db()).unwrap();
    let bumped = db
        .execute(
            "UPDATE outputs SET rev = rev + 1 WHERE run = ?1 AND name = 'big'",
            [made.id.as_str()],
        )
        .unwrap();
    assert_eq!(bumped, 1);
    drop(db);
    let served = Served::start(home.path()).await;
    let until = Instant::now() + PATIENCE;
    let big = loop {
        let big = big(served.run(&made.id).await);
        if big.export_error.is_some() || big.path.is_some() {
            break big;
        }
        assert!(Instant::now() < until, "{big:?}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    let error = big.export_error.as_deref().unwrap_or_default();
    assert!(error.contains("the machine refused it"), "{big:?}");
    served.down().await;
}
