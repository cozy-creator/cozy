//! Calcifer's loopback API and its lifecycle, over real HTTP against `serve` on a temp home. Tests
//! that reach a machine run a real TensorD (`COZY_TEST_TENSORD`); its runs are warm runs of a
//! release from a Hub that never answers, so they stay preparing until they are canceled.
mod common;

use std::{
    path::Path,
    process::{Command, Stdio},
    time::Duration,
};

use calcifer::{
    Link,
    serve::{Stopped, serve},
    store::{MachineRow, Store},
};
use common::{SilentHub, Tensord, want};
use cozy_platform::{
    lock::{Kind as LockKind, LockFile},
    paths::Home,
};
use cozy_proto::{
    calcifer::{Down, Event, Holder, Run, RunList, Status, Submitted},
    config::{Config, Loaded},
    error::Envelope,
};
use reqwest::{Client, Method, RequestBuilder, StatusCode, header};
use serde_json::json;
use tokio::task::JoinHandle;

/// A hang bound for a test whose awaited state never comes; the product waits without one.
const PATIENCE: Duration = Duration::from_secs(120);

/// A serving Calcifer on `home`, and a client of it.
struct Calcifer {
    task: JoinHandle<Result<Stopped, cozy_proto::error::Error>>,
    holder: Holder,
    client: Client,
}

impl Calcifer {
    /// Starts `serve` as the CLI does: wanting it (`wanted.lock`) until its first answer.
    async fn start(home: &Path) -> Self {
        let wanted = want(home);
        let loaded = loaded(home);
        let watched = loaded.home.clone();
        let task = tokio::spawn(async move { serve(&loaded).await });
        let holder = record(&watched, |h| h.pid == std::process::id()).await;
        let calcifer = Self {
            task,
            holder,
            client: Client::new(),
        };
        let first = calcifer.request(Method::GET, "/v1/runs").send().await;
        assert_eq!(first.unwrap().status(), StatusCode::OK);
        drop(wanted);
        calcifer
    }

    fn request(&self, method: Method, path: &str) -> RequestBuilder {
        self.client
            .request(method, format!("http://{}{path}", self.holder.address))
            .bearer_auth(&self.holder.token)
    }

    async fn submit(&self, key: &str, body: serde_json::Value) -> (StatusCode, Submitted) {
        let response = self
            .request(Method::POST, "/v1/runs")
            .header("idempotency-key", key)
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = response.status();
        (status, response.json().await.unwrap())
    }

    async fn cancel(&self, id: &str) -> (StatusCode, Run) {
        let response = self
            .request(Method::POST, &format!("/v1/runs/{id}/cancel"))
            .send()
            .await
            .unwrap();
        (response.status(), response.json().await.unwrap())
    }

    async fn events(&self, id: &str, after: u64) -> Events {
        let response = self
            .request(
                Method::GET,
                &format!("/v1/runs/{id}/events?starting_after={after}"),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        Events {
            response,
            buffer: String::new(),
        }
    }

    async fn down(&self) -> Down {
        let response = self.request(Method::POST, "/v1/down").send().await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        response.json().await.unwrap()
    }

    /// Lets go of the client's connections and waits for Calcifer to stop on its own.
    async fn stopped(self) -> Stopped {
        drop(self.client);
        tokio::time::timeout(PATIENCE, self.task)
            .await
            .expect("Calcifer never stopped")
            .unwrap()
            .unwrap()
    }
}

/// A home's load, as `cozy calcifer` reads it, with the defaults (any free port).
fn loaded(home: &Path) -> Loaded {
    Loaded {
        config: Config::default(),
        home: Home::at(home),
        user_home: home.to_owned(),
        warnings: vec![],
    }
}

/// The lock record, once `wanted` accepts it.
async fn record(home: &Home, wanted: impl Fn(&Holder) -> bool) -> Holder {
    let lock = LockFile::new(home.calcifer_lock());
    tokio::time::timeout(PATIENCE, async {
        loop {
            if let Some(holder) = lock
                .holder()
                .unwrap()
                .and_then(|bytes| serde_json::from_slice::<Holder>(&bytes).ok())
                .filter(&wanted)
            {
                return holder;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("no Calcifer published its record")
}

/// One run's SSE stream, parsed.
struct Events {
    response: reqwest::Response,
    buffer: String,
}

impl Events {
    /// The next event and its `id:`; `None` once the server ends the stream.
    async fn next(&mut self) -> Option<(Option<u64>, Event)> {
        tokio::time::timeout(PATIENCE, async {
            loop {
                if let Some(end) = self.buffer.find("\n\n") {
                    let block: String = self.buffer.drain(..end + 2).collect();
                    let (mut id, mut data) = (None, None);
                    for line in block.lines() {
                        if let Some(value) = line.strip_prefix("id:") {
                            id = Some(value.trim().parse().unwrap());
                        } else if let Some(value) = line.strip_prefix("data:") {
                            data = Some(value.trim().to_owned());
                        }
                    }
                    // A block with no data is the heartbeat comment.
                    if let Some(data) = data {
                        return Some((id, serde_json::from_str(&data).unwrap()));
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

    /// Every event until `wanted` accepts one (included).
    async fn until(&mut self, wanted: impl Fn(&Event) -> bool) -> Vec<(Option<u64>, Event)> {
        let mut seen = vec![];
        loop {
            let next = self.next().await.expect("the stream ended first");
            let done = wanted(&next.1);
            seen.push(next);
            if done {
                return seen;
            }
        }
    }

    /// Every event left, the stream's end included.
    async fn rest(&mut self) -> Vec<(Option<u64>, Event)> {
        let mut seen = vec![];
        while let Some(next) = self.next().await {
            seen.push(next);
        }
        seen
    }
}

fn run_of(event: &Event) -> Option<&Run> {
    match event {
        Event::Created { run, .. }
        | Event::Queued { run, .. }
        | Event::InProgress { run, .. }
        | Event::Completed { run, .. }
        | Event::Failed { run, .. }
        | Event::Canceled { run, .. } => Some(run),
        _ => None,
    }
}

fn is_status(status: Status) -> impl Fn(&Event) -> bool {
    move |event| run_of(event).is_some_and(|run| run.status == status)
}

/// A warm run held preparing by a Hub that never answers.
fn held(on: &str, hub: &SilentHub) -> serde_json::Value {
    submission(on, &common::warm_spec(&hub.origin(), vec![]))
}

fn submission(on: &str, spec: &[u8]) -> serde_json::Value {
    use base64::Engine as _;
    json!({
        "on": on,
        "hub": "https://tensorhub.com",
        "kind": "warm",
        "target": "cozy-test/held",
        "spec": base64::engine::general_purpose::STANDARD.encode(spec),
    })
}

/// Records a machine in the home's store before Calcifer opens it.
fn add_machine(home: &Path, row: &MachineRow) {
    Store::open(&Home::at(home).calcifer_db())
        .unwrap()
        .put_machine(row)
        .unwrap();
}

async fn refusal(request: RequestBuilder) -> (StatusCode, String) {
    let response = request.send().await.unwrap();
    let status = response.status();
    assert!(
        !response
            .headers()
            .keys()
            .any(|name| name.as_str().starts_with("access-control-")),
        "no CORS header is ever sent"
    );
    let envelope: Envelope = response.json().await.unwrap();
    (status, envelope.error.code)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_guard_chain_refuses_a_foreign_host_origin_bearer_or_missing_key() {
    let home = tempfile::tempdir().unwrap();
    let calcifer = Calcifer::start(home.path()).await;
    let port = calcifer.holder.address.port();
    let refused = |status: StatusCode, code: &str| (status, code.to_owned());

    let rebound = calcifer
        .request(Method::GET, "/v1/runs")
        .header(header::HOST, format!("evil.test:{port}"));
    assert_eq!(
        refusal(rebound).await,
        refused(StatusCode::FORBIDDEN, "host_not_allowed")
    );
    for path in ["/v1/runs", "/v1/runs/1/events"] {
        let method = if path == "/v1/runs" {
            Method::POST
        } else {
            Method::GET
        };
        let foreign = calcifer
            .request(method, path)
            .header("idempotency-key", "k")
            .header(header::ORIGIN, "http://evil.test");
        assert_eq!(
            refusal(foreign).await,
            refused(StatusCode::FORBIDDEN, "origin_not_allowed"),
            "{path}"
        );
    }
    let url = format!("http://{}/v1/runs", calcifer.holder.address);
    let response = calcifer.client.get(&url).send().await.unwrap();
    assert_eq!(
        response.headers()[header::WWW_AUTHENTICATE],
        r#"Bearer realm="cozy""#
    );
    assert_eq!(
        refusal(calcifer.client.get(&url)).await,
        refused(StatusCode::UNAUTHORIZED, "unauthenticated")
    );
    assert_eq!(
        refusal(calcifer.client.get(&url).bearer_auth("not-the-token")).await,
        refused(StatusCode::UNAUTHORIZED, "unauthenticated")
    );
    let unkeyed = calcifer
        .request(Method::POST, "/v1/runs")
        .json(&submission("any", &[]));
    assert_eq!(
        refusal(unkeyed).await,
        refused(StatusCode::BAD_REQUEST, "idempotency_key_required")
    );
    let unknown = calcifer
        .client
        .get(format!("http://{}/v1/nothing", calcifer.holder.address));
    assert_eq!(
        refusal(unknown).await,
        refused(StatusCode::NOT_FOUND, "unknown_route")
    );

    let good = calcifer
        .request(Method::GET, "/v1/runs")
        .header(header::HOST, format!("localhost:{port}"))
        .header(header::ORIGIN, format!("http://127.0.0.1:{port}"))
        .send()
        .await
        .unwrap();
    assert_eq!(good.status(), StatusCode::OK);
    assert_eq!(good.headers()[header::CACHE_CONTROL], "no-store");
    let list: RunList = good.json().await.unwrap();
    assert!(list.runs.is_empty());
    assert_eq!(calcifer.stopped().await, Stopped::Idle);
}

/// Unplaced runs (`on: any`) wait in the outbox without a machine: no TensorD is needed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_submission_is_accepted_replayed_and_refused_under_another_body_and_calcifer_exits_once_it_is_settled()
 {
    let home = tempfile::tempdir().unwrap();
    let calcifer = Calcifer::start(home.path()).await;
    let mut body = submission("any", &common::warm_spec("http://127.0.0.1:9", vec![]));
    body["from_a_newer_cli"] = json!(true);

    let (status, first) = calcifer.submit("k1", body.clone()).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert!(!first.replay && first.run.status == Status::Queued);
    let (status, again) = calcifer.submit("k1", body.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(again.replay && again.run.id == first.run.id);
    body["target"] = json!("cozy-test/other");
    let conflict = calcifer
        .request(Method::POST, "/v1/runs")
        .header("idempotency-key", "k1")
        .json(&body);
    assert_eq!(
        refusal(conflict).await,
        (StatusCode::CONFLICT, "idempotency_conflict".to_owned())
    );

    let number = first.run.number.to_string();
    let shown: Run = calcifer
        .request(Method::GET, &format!("/v1/runs/{number}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(shown.id, first.run.id);
    let listed: RunList = calcifer
        .request(Method::GET, "/v1/runs?status=queued&limit=5")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed.runs.len(), 1);

    let (status, canceled) = calcifer.cancel(first.run.id.as_str()).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(canceled.status, Status::Canceled);
    let (status, _) = calcifer.cancel(&number).await;
    assert_eq!(status, StatusCode::OK, "already settled");
    assert_eq!(calcifer.stopped().await, Stopped::Idle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_calcifer_exits_while_the_first_holds_the_home_whose_record_is_private() {
    let home = tempfile::tempdir().unwrap();
    let calcifer = Calcifer::start(home.path()).await;
    let lock = LockFile::new(Home::at(home.path()).calcifer_lock());
    let published = lock.holder().unwrap().unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let path = home.path().join("calcifer/calcifer.lock.holder");
        let mode = std::fs::metadata(path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "the record holds the bearer");
    }
    assert_eq!(
        serve(&loaded(home.path())).await.unwrap(),
        Stopped::AnotherHolds
    );
    assert_eq!(lock.holder().unwrap().unwrap(), published);

    let down = calcifer.down().await;
    assert!(down.stopping && down.owed.is_empty());
    assert_eq!(calcifer.stopped().await, Stopped::Down);
    assert_eq!(
        lock.holder().unwrap(),
        None,
        "a clean stop erases its record"
    );
}

/// The process `a_killed_calcifer_leaves_a_record_the_next_one_replaces` kills.
#[tokio::test]
#[ignore = "a child process of another test"]
async fn serve_until_killed() {
    if let Some(home) = std::env::var_os("CALCIFER_TEST_HOME") {
        serve(&loaded(Path::new(&home))).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_killed_calcifer_leaves_a_record_the_next_one_replaces() {
    let home = tempfile::tempdir().unwrap();
    let wanted = want(home.path());
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "serve_until_killed",
            "--exact",
            "--ignored",
            "--test-threads",
            "1",
        ])
        .env("CALCIFER_TEST_HOME", home.path())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let pid = child.id();
    let stale = record(&Home::at(home.path()), |h| h.pid == pid).await;
    child.kill().unwrap();
    child.wait().unwrap();
    drop(wanted);
    let lock = LockFile::new(Home::at(home.path()).calcifer_lock());
    let left: Holder = serde_json::from_slice(&lock.holder().unwrap().unwrap()).unwrap();
    assert_eq!(left, stale, "SIGKILL leaves the record behind");
    // Shared, so the check leaves the stale record (an exclusive taker erases it).
    let free = lock.try_lock(LockKind::Shared).unwrap();
    assert!(
        free.is_some(),
        "the kernel let go of the dead holder's lock"
    );
    drop(free);
    assert!(lock.holder().unwrap().is_some());

    let calcifer = Calcifer::start(home.path()).await;
    assert_ne!(
        calcifer.holder.token, stale.token,
        "a fresh bearer per start"
    );
    let ok = calcifer
        .request(Method::GET, "/v1/runs")
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), StatusCode::OK);
    let old = calcifer
        .client
        .get(format!("http://{}/v1/runs", calcifer.holder.address))
        .bearer_auth(&stale.token);
    assert_eq!(
        refusal(old).await,
        (StatusCode::UNAUTHORIZED, "unauthenticated".to_owned())
    );
    calcifer.down().await;
    assert_eq!(calcifer.stopped().await, Stopped::Down);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_is_followed_to_the_machines_outcome_and_a_reconnect_gets_only_what_is_new() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = Tensord::start();
    let hub = SilentHub::new();
    add_machine(home.path(), &machine.row("m", &key));
    let calcifer = Calcifer::start(home.path()).await;
    let (status, submitted) = calcifer.submit("k", held("m", &hub)).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let id = submitted.run.id.to_string();

    let mut events = calcifer.events(&id, 0).await;
    let first = events.until(is_status(Status::InProgress)).await;
    let accepted = first.last().unwrap().0.unwrap();
    let (status, canceling) = calcifer.cancel(&id).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert!(canceling.cancel_requested);
    let rest = events.rest().await;
    let (end, terminal) = rest.last().unwrap().clone();
    assert!(terminal.is_terminal(), "the stream ends at the terminal");
    let run = run_of(&terminal).unwrap();
    assert_eq!(run.status, Status::Canceled);
    assert_eq!(
        run.reason.as_ref().unwrap().code,
        "canceled",
        "the machine's own outcome"
    );
    let seen: Vec<u64> = first
        .iter()
        .chain(&rest)
        .filter_map(|(id, _)| *id)
        .collect();
    assert!(seen.windows(2).all(|w| w[0] < w[1]), "{seen:?}");

    let replay = calcifer.events(&id, accepted).await.rest().await;
    assert!(
        replay
            .iter()
            .all(|(id, _)| id.is_some_and(|id| id > accepted))
    );
    assert_eq!(
        replay.last().unwrap().0,
        end,
        "it ends at the same terminal"
    );
    let after_end = calcifer.events(&id, end.unwrap()).await.rest().await;
    assert!(after_end.is_empty(), "{after_end:?}");
    assert_eq!(calcifer.cancel(&id).await.0, StatusCode::OK);
    assert_eq!(calcifer.stopped().await, Stopped::Idle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn down_names_what_is_owed_the_machine_keeps_running_it_and_the_next_calcifer_resumes_it() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = Tensord::start();
    let hub = SilentHub::new();
    add_machine(home.path(), &machine.row("m", &key));
    let calcifer = Calcifer::start(home.path()).await;
    let (_, submitted) = calcifer.submit("k", held("m", &hub)).await;
    let id = submitted.run.id.to_string();
    let mut events = calcifer.events(&id, 0).await;
    events.until(is_status(Status::InProgress)).await;

    let down = calcifer.down().await;
    assert!(down.stopping);
    assert_eq!(down.owed, vec![id.clone()]);
    assert!(
        events.rest().await.iter().all(|(_, e)| !e.is_terminal()),
        "down ends an open stream without a terminal"
    );
    assert_eq!(calcifer.stopped().await, Stopped::Down);
    let link = Link::new(&machine.endpoint, common::key()).unwrap();
    let (state, _) = link.run(&id, 0, None).await.unwrap();
    assert_eq!(state.state, "preparing", "the machine keeps the run");

    let calcifer = Calcifer::start(home.path()).await;
    let shown: Run = calcifer
        .request(Method::GET, &format!("/v1/runs/{id}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(shown.status, Status::InProgress);
    let mut events = calcifer.events(&id, 0).await;
    calcifer.cancel(&id).await;
    let end = events.rest().await.pop().unwrap().1;
    assert_eq!(run_of(&end).unwrap().status, Status::Canceled);
    assert_eq!(
        calcifer.stopped().await,
        Stopped::Idle,
        "it exits by itself once nothing is owed"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_calcifer_owing_nothing_stops_at_once_unless_a_client_wants_it() {
    let home = tempfile::tempdir().unwrap();
    let unwanted = tokio::time::timeout(PATIENCE, serve(&loaded(home.path()))).await;
    assert_eq!(unwanted.unwrap().unwrap(), Stopped::Idle, "a login start");
    let lock = LockFile::new(Home::at(home.path()).calcifer_lock());
    assert_eq!(lock.holder().unwrap(), None);

    let calcifer = Calcifer::start(home.path()).await;
    let again = calcifer.request(Method::GET, "/v1/runs").send().await;
    assert_eq!(
        again.unwrap().status(),
        StatusCode::OK,
        "its client is connected"
    );
    assert_eq!(calcifer.stopped().await, Stopped::Idle);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_for_a_disabled_local_machine_or_an_unknown_one_is_refused_unrecorded() {
    let home = tempfile::tempdir().unwrap();
    let calcifer = Calcifer::start(home.path()).await;
    let spec = common::warm_spec("http://127.0.0.1:9", vec![]);
    let submit = |on: &str| {
        calcifer
            .request(Method::POST, "/v1/runs")
            .header("idempotency-key", on)
            .json(&submission(on, &spec))
            .send()
    };
    let local = submit("local").await.unwrap();
    assert_eq!(local.status(), StatusCode::NOT_FOUND);
    let local = local.json::<Envelope>().await.unwrap().error;
    assert_eq!(local.code, "machine_not_found");
    assert_eq!(local.message, "this computer's machine is not enabled");
    assert_eq!(
        local.remedy.as_deref(),
        Some("`cozy machine enable` makes this computer a machine")
    );
    assert_eq!(local.next, ["cozy run … --on any", "cozy machine list"]);
    let other = submit("kiyui").await.unwrap().json::<Envelope>().await;
    assert_eq!(other.unwrap().error.message, r#"no machine named "kiyui""#);
    let listed: RunList = calcifer
        .request(Method::GET, "/v1/runs")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(listed.runs.is_empty(), "nothing was recorded");
    assert_eq!(calcifer.stopped().await, Stopped::Idle);
}
