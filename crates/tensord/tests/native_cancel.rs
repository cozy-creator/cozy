//! Native cancellation uses real SQLite transactions and the machine API's real server and client.
//! No SDK/GPU needed.
use tensord::{
    api::{
        self,
        auth::VerifiedActor,
        capability::{self, Grant},
        domain, v1, MachineBackend, MachineIdentity,
    },
    execution::Engine,
    journal::{Invocation, Journal, Outcome, ResultRecord, State},
    machine_api::{actor_id, NativeBackend},
    objects::Objects,
    runs::Runs,
    service::Service,
};
use ed25519_dalek::SigningKey;
use fs2::FileExt;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Barrier},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_stream::StreamExt;
use v1::client::{Auth, Machine};

const ALICE: [u8; 32] = [51; 32];
const BOB: [u8; 32] = [52; 32];

struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("cm-native-cancel-{}", uuid::Uuid::new_v4())))
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn actor(key: [u8; 32]) -> String {
    actor_id(VerifiedActor {
        public_key: SigningKey::from_bytes(&key).verifying_key().to_bytes(),
    })
}
/// `key`'s cap: machine scope, or run scope for `run`.
fn token(key: [u8; 32], run: Option<&str>) -> Auth {
    let expires = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + 120;
    let mut grant = Grant {
        machine: "cancel-proof".into(),
        expires,
        ..Default::default()
    };
    if let Some(id) = run {
        grant.run = id.into();
    } else {
        grant.action = capability::MACHINE.into();
    }
    Auth::Token(capability::mint(&SigningKey::from_bytes(&key), grant))
}
fn spec() -> v1::RunSpec {
    v1::RunSpec {
        kind: v1::RunKind::Call,
        source: Some(v1::Source::Installation(
            "unused-installation".into(),
        )),
        entrypoint: "must_not_prepare".into(),
        payload: v1::Json::parse("{}").ok(),
        ..Default::default()
    }
}

struct Server {
    client: Machine,
    service: Arc<Service>,
    backend: Arc<NativeBackend>,
    task: tokio::task::JoinHandle<()>,
}
impl Server {
    async fn open(root: &Path) -> Self {
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let store = Arc::new(tensorfs_core::store::Store::ensure(&root.join("store")).unwrap());
        let objects = Arc::new(
            Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap(),
        );
        let mut identity = MachineIdentity::ephemeral(
            "cancel-proof".into(),
            [ALICE, BOB]
                .map(|key| SigningKey::from_bytes(&key).verifying_key())
                .to_vec(),
            vec![7; 32],
        )
        .unwrap();
        let paths =
            tensord::machine::update::Paths::new(&root.join("state"), &root.join("image"));
        identity.updates = Some(
            tensord::machine::update::Updates::open(paths, Box::new(|| true), None, None).unwrap(),
        );
        let pin = tensorfs_core::sha256::digest(&identity.cert_der);
        let mut backend = NativeBackend::new(service.clone(), identity.authority.clone(), store);
        backend.runs = Some(Arc::new(Runs {
            service: service.clone(),
            objects,
            publisher: None,
            local: None,
            own_hub: None,
            leaf: None,
            contexts: Default::default(),
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let backend = Arc::new(backend);
        let served = backend.clone();
        let task =
            tokio::spawn(async move { api::serve(listener, identity, served).await.unwrap() });
        Self {
            client: Machine::new(&address.to_string(), pin, token(ALICE, None)).unwrap(),
            service,
            backend,
            task,
        }
    }
    /// The client as `key`'s holder, machine scope or run scope for `run`.
    fn by(&self, key: [u8; 32], run: Option<&str>) -> Machine {
        self.client.with_auth(token(key, run))
    }
    async fn cancel(&self, id: &str, key: [u8; 32]) -> v1::RunState {
        self.by(key, None).cancel(id).await.unwrap()
    }
    /// The run `id` (submitted again with `sent`) is canceled: its state and outcome say so.
    async fn canceled(&self, id: &str, sent: Option<v1::RunSpec>) {
        if let Some(spec) = sent {
            let state = self.client.submit(&v1::RunRequest { id: id.into(), spec }).await.unwrap();
            assert_eq!(state.state, "canceled");
        }
        assert_canceled(self.client.events(id, 0).await.unwrap()).await;
    }
    async fn stop(self) {
        self.task.abort();
        let _ = self.task.await;
    }
}

/// Events of a run canceled before acceptance: its state and outcome, nothing else.
async fn assert_canceled(events: impl tokio_stream::Stream<Item = Result<v1::RunEvent, v1::client::Error>>) {
    let mut events = Box::pin(events);
    let mut outcome = false;
    while let Some(event) = tokio::time::timeout(Duration::from_secs(5), events.next()).await.unwrap() {
        match event.unwrap().event {
            v1::Event::State(state) => assert_eq!(state.state, "canceled"),
            v1::Event::Outcome(seen) => {
                assert_eq!(seen.status, "canceled");
                outcome = true;
            }
            other => panic!("canceled before acceptance emitted {other:?}"),
        }
    }
    assert!(outcome);
}

#[tokio::test]
async fn canceled_ids_refuse_updates_and_another_actors_update_never_shadows_them() {
    let root = Root::new();
    let server = Server::open(&root.0).await;
    let canceled = server.cancel("shared", ALICE).await;
    let update = v1::RunSpec {
        kind: v1::RunKind::Update,
        payload: v1::Json::parse(r#"{"runtime":"0.19.0"}"#).ok(),
        ..Default::default()
    };
    let error = server.client.submit(&v1::RunRequest { id: "shared".into(), spec: update }).await.unwrap_err();
    assert_eq!((error.status(), error.code()), (Some(409), Some("run_id_conflict")));
    server
        .service
        .engine
        .with_journal(|journal| journal.reserve_update_run(&actor(BOB), "shared"))
        .unwrap();
    // If update admission wins after the API's ownership read, the transactional backend
    // still refuses this operation without presenting it as a missing/older API method.
    let error = server
        .backend
        .control(
            VerifiedActor {
                public_key: SigningKey::from_bytes(&BOB).verifying_key().to_bytes(),
            },
            domain::MachineExecutionControl {
                execution: Some(domain::MachineExecutionQuery {
                    request_id: "shared".into(),
                    expected_execution_workspace_id: server.service.engine.workspace_id(),
                }),
                action: domain::MachineExecutionAction::Cancel as i32,
            },
        )
        .unwrap_err();
    assert_eq!(error.code, "update_control_unsupported");
    // Over the API a cancel reaches the update itself; this reserved one never started.
    let error = server.by(BOB, None).cancel("shared").await.unwrap_err();
    assert_eq!(error.status(), Some(404));
    assert_eq!(server.cancel("shared", ALICE).await, canceled);
    server.canceled("shared", None).await;
    server.stop().await;
    // The update payload journal remains machine-wide; its retained status must not shadow
    // another signer's canceled ordinary run with the same text ID.
    let status = tensord::machine::update::Status {
        operation: "shared".into(),
        state: "succeeded".into(),
        history: vec![tensord::machine::update::Step {
            state: "succeeded".into(),
            at_ms: 1,
        }],
        ..Default::default()
    };
    fs::write(
        root.0.join("state/update/status.json"),
        serde_json::to_vec(&status).unwrap(),
    )
    .unwrap();
    let restarted = Server::open(&root.0).await;
    // A cancel of a finished update answers its end.
    assert_eq!(restarted.cancel("shared", BOB).await.state, "succeeded");
    assert_eq!(
        restarted.cancel("shared", ALICE).await.number,
        canceled.number
    );
    restarted.canceled("shared", None).await;
    let error = restarted.by(BOB, Some("shared")).events("shared", 0).await.err().unwrap();
    assert_eq!(error.status(), Some(403));
    let mut update = Box::pin(restarted.by(BOB, None).events("shared", 0).await.unwrap());
    let mut observed = false;
    while let Some(event) = update.next().await {
        if let v1::Event::Outcome(outcome) = event.unwrap().event {
            assert_eq!(outcome.status, "succeeded");
            observed = true;
        }
    }
    assert!(observed);
    restarted.stop().await;
}

#[tokio::test]
async fn a_busy_ownership_lookup_does_not_block_the_api_executor() {
    let root = Root::new();
    let server = Server::open(&root.0).await;
    let engine = server.service.engine.clone();
    let (held, locked) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    let writer = std::thread::spawn(move || {
        engine
            .with_journal(|_| {
                held.send(()).unwrap();
                // The watchdog releases only this fixture's lock if an old blocking handler stalls
                // the single API executor, so the failure cannot hang the test process.
                let _ = released.recv_timeout(Duration::from_secs(3));
                Ok(())
            })
            .unwrap()
    });
    locked.recv().unwrap();
    let blocked = server.client.clone();
    let lookup = tokio::spawn(async move { blocked.events("absent", 0).await.err() });
    tokio::time::sleep(Duration::from_millis(30)).await;
    let responsive =
        tokio::time::timeout(Duration::from_millis(500), server.client.cancel(&"x".repeat(257))).await;
    let _ = release.send(());
    writer.join().unwrap();
    let refused = responsive.expect("journal ownership I/O stalled the API executor").unwrap_err();
    assert_eq!(refused.code(), Some("invalid_run_id"));
    assert_eq!(lookup.await.unwrap().unwrap().code(), Some("run_not_found"));
    server.stop().await;
}

#[test]
fn update_and_cancel_admission_serialize_across_real_sqlite_connections() {
    let root = Root::new();
    let mut journal = Journal::open(&root.0).unwrap();
    journal.reserve_update_run("alice", "update-first").unwrap();
    assert_eq!(
        journal
            .reserve_run_cancellation("alice", "update-first")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::Unsupported
    );
    assert_eq!(
        journal
            .accept_run("alice", "update-first", "spec", Invocation::default())
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::AlreadyExists
    );
    journal
        .reserve_run_cancellation("alice", "cancel-first")
        .unwrap();
    assert_eq!(
        journal
            .reserve_update_run("alice", "cancel-first")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::AlreadyExists
    );
    journal.reserve_update_run("bob", "cancel-first").unwrap();
    drop(journal);
    for n in 0..24 {
        let id = format!("update-race-{n}");
        let mut updater = Journal::open(&root.0).unwrap();
        let mut canceller = Journal::open(&root.0).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let start = barrier.clone();
        let update_id = id.clone();
        let update = std::thread::spawn(move || {
            start.wait();
            updater.reserve_update_run("alice", &update_id)
        });
        barrier.wait();
        let canceled = canceller.reserve_run_cancellation("alice", &id);
        let updated = update.join().unwrap();
        match (updated, canceled) {
            (Ok(()), Err(error)) => assert_eq!(error.kind(), std::io::ErrorKind::Unsupported),
            (Err(error), Ok(record)) => {
                assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
                assert!(record.canceled_before_acceptance && record.state == State::Canceled);
            }
            results => panic!("both admissions won or failed: {results:?}"),
        }
    }
    let mut restarted = Journal::open(&root.0).unwrap();
    assert!(restarted
        .update_run_reserved("alice", "update-first")
        .unwrap());
    assert!(restarted
        .update_run_reserved("bob", "cancel-first")
        .unwrap());
    assert!(
        restarted
            .get_public("alice", "cancel-first")
            .unwrap()
            .canceled_before_acceptance
    );
    assert_eq!(
        restarted
            .reserve_run_cancellation("alice", "update-first")
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::Unsupported
    );
}

#[tokio::test]
async fn absent_cancel_is_observable_scoped_repeatable_and_survives_restart() {
    let root = Root::new();
    let server = Server::open(&root.0).await;
    // Cancel is the only control: a run is canceled, and submitted again.
    for id in ["", &"x".repeat(257)] {
        assert!(server.client.cancel(id).await.is_err(), "a run id is 1-256 bytes");
    }
    let error = server.by(ALICE, Some("never-sent")).cancel("never-sent").await.unwrap_err();
    assert_eq!(error.status(), Some(403));
    assert!(server
        .service
        .engine
        .get_public(&actor(ALICE), "never-sent")
        .is_err());
    let first = server.cancel("never-sent", ALICE).await;
    assert_eq!(first.state, "canceled");
    assert_eq!(server.cancel("never-sent", ALICE).await, first);
    assert!(server
        .service
        .engine
        .get_public(&actor(BOB), "never-sent")
        .is_err());
    let (bob, new) = server
        .service
        .engine
        .accept_run(&actor(BOB), "never-sent", "bob-spec", Invocation::default())
        .unwrap();
    assert!(new);
    assert_eq!(bob.state, State::Queued);
    server.canceled("never-sent", None).await;
    server.canceled("never-sent", Some(spec())).await;
    server.stop().await;
    let restarted = Server::open(&root.0).await;
    restarted.canceled("never-sent", None).await;
    restarted.canceled("never-sent", Some(spec())).await;
    assert_eq!(
        restarted.cancel("never-sent", ALICE).await.number,
        first.number
    );
    restarted.stop().await;
}

#[tokio::test]
async fn cancellation_wins_while_the_real_run_acceptance_waits_on_store_custody() {
    let root = Root::new();
    let server = Server::open(&root.0).await;
    let path = root.0.join("store/tmp/writers/recovery.lock");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let lock = fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap();
    lock.lock_exclusive().unwrap();
    let client = server.client.clone();
    let submitted =
        tokio::spawn(async move { client.submit(&v1::RunRequest { id: "blocked".into(), spec: spec() }).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(server
        .service
        .engine
        .get_public(&actor(ALICE), "blocked")
        .is_err());
    assert_eq!(server.cancel("blocked", ALICE).await.state, "canceled");
    FileExt::unlock(&lock).unwrap();
    let state = tokio::time::timeout(Duration::from_secs(5), submitted).await.unwrap().unwrap().unwrap();
    assert_eq!(state.state, "canceled", "the blocked acceptance escaped cancellation");
    assert_canceled(server.client.events("blocked", 0).await.unwrap()).await;
    let record = server
        .service
        .engine
        .get_public(&actor(ALICE), "blocked")
        .unwrap();
    assert!(record.canceled_before_acceptance);
    assert_eq!(record.attempt, 0);
    assert!(record.process.is_none() && record.invocation.generation.is_empty());
    assert!(server.service.engine.ready().unwrap().is_empty());
    server.stop().await;
}

#[test]
fn acceptance_and_cancel_have_one_durable_winner_and_preserve_completed_outcomes() {
    let root = Root::new();
    let engine = Engine::open(&root.0).unwrap();
    for n in 0..16 {
        let id = format!("race-{n}");
        let barrier = Arc::new(Barrier::new(2));
        let accepting = engine.clone();
        let start = barrier.clone();
        let sent = id.clone();
        let accept = std::thread::spawn(move || {
            start.wait();
            accepting
                .accept_run("alice", &sent, "spec", Invocation::default())
                .unwrap()
        });
        barrier.wait();
        let canceled = engine.cancel_run("alice", &id).unwrap();
        let (accepted, new) = accept.join().unwrap();
        assert_eq!(accepted.id, canceled.id);
        assert_eq!(
            engine.get_public("alice", &id).unwrap().state,
            State::Canceled
        );
        assert_eq!(new, !canceled.canceled_before_acceptance);
    }
    let record = engine
        .accept_run("alice", "completed", "authored-spec", Invocation::default())
        .unwrap()
        .0;
    engine
        .end_preparation(
            &record.id,
            Outcome::Completed(ResultRecord {
                value: serde_json::json!(42),
                artifacts: vec![],
                asset_bindings: vec![],
            }),
        )
        .unwrap();
    let canceled = engine.cancel_run("alice", "completed").unwrap();
    assert_eq!(canceled.state, State::Completed);
    assert_eq!(canceled.result.unwrap().value, serde_json::json!(42));
    assert!(!canceled.canceled_before_acceptance);
    drop(engine);
    let mut journal = Journal::open(&root.0).unwrap();
    let before = journal.reserve_run_cancellation("alice", "before").unwrap();
    drop(journal);
    let mut journal = Journal::open(&root.0).unwrap();
    let (after, new) = journal
        .accept_run("alice", "before", "late-spec", Invocation::default())
        .unwrap();
    assert!(!new && after.canceled_before_acceptance && after.state == State::Canceled);
    assert_eq!(before.id, after.id);
}
