//! A machine Calcifer cannot reach, against real TensorD processes (`COZY_TEST_TENSORD`) and a
//! loopback stand-in Hub answering its rental routes as Tensorhub does: a rental its Hub says its
//! provider stopped, or one that answers nothing with no known cause, is refused at once and says
//! why; a booting rental queues; a rental back at a new address is followed.
mod common;

use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::State,
    http::{StatusCode, Uri},
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use calcifer::{
    Calcifer,
    machines::Machines,
    store::{MachineRow, State as RunState},
};
use common::{SilentHub, Tensord, until};
use platform::{durable, paths::Home};
use protocol::{
    calcifer::{Event, Kind, MachineKind, MachineState, Status, SubmitRun},
    config,
};
use serde_json::{Value, json};

/// The stand-in Hub: each rental as `GET /v1/rentals/{id}` answers it, and how many it answered.
struct Hub {
    origin: String,
    rentals: Mutex<HashMap<String, Value>>,
    reads: AtomicUsize,
}

impl Hub {
    async fn start() -> Arc<Self> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let hub = Arc::new(Hub {
            origin: format!("http://{}", listener.local_addr().unwrap()),
            rentals: Mutex::default(),
            reads: AtomicUsize::new(0),
        });
        let router = axum::Router::new().fallback(answer).with_state(hub.clone());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        hub
    }

    /// The Hub's word on rental `id` from now on: `state`, with `more` fields.
    fn says(&self, id: &str, state: &str, more: Value) {
        let mut rental = json!({
            "rental_id": id, "name": "rent-0a1b", "state": state,
            "requested_accelerator_model": "rtx-4090", "accelerator_count": 1,
            "creator_public_key": "k", "base_worker_image_digest": "sha256:0",
            "hourly_rate_usd_micros": 400000, "spend_usd_micros": 0, "spend_basis": "charged",
            "container_disk_gb": 100, "created_at": "2026-10-22T10:00:00Z",
        });
        for (key, value) in more.as_object().unwrap() {
            rental[key] = value.clone();
        }
        self.rentals.lock().unwrap().insert(id.into(), rental);
    }

    fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }
}

async fn answer(State(hub): State<Arc<Hub>>, uri: Uri) -> Response {
    let rentals = hub.rentals.lock().unwrap();
    let body = match uri.path() {
        "/v1/rentals" => json!({
            "rentals": rentals.values().cloned().collect::<Vec<_>>(),
            "bindings_revision": 0, "credit": null,
        }),
        path => match path.strip_prefix("/v1/rentals/").and_then(|id| rentals.get(id)) {
            Some(rental) => {
                hub.reads.fetch_add(1, Ordering::SeqCst);
                rental.clone()
            }
            None => return StatusCode::NOT_FOUND.into_response(),
        },
    };
    (StatusCode::OK, serde_json::to_vec(&body).unwrap()).into_response()
}

/// This computer's sign-in at `hub`: a synthetic device key and a kept bearer, live an hour.
fn sign_in(home: &Home, hub: &str) {
    let hub = config::Hub {
        name: None,
        origin: config::Origin::parse(hub).unwrap(),
    };
    durable::private_dir(&home.keys()).unwrap();
    let key = json!({"version": 1, "hub": hub.origin.as_str(), "email": "owner@example.test",
        "device_key_id": "offline-test-device", "private_key": URL_SAFE_NO_PAD.encode([17u8; 32])});
    durable::replace(&hub.device_key(home), &serde_json::to_vec(&key).unwrap(), 0o600).unwrap();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let bearer = hub_client::Bearer {
        token: "t1".into(),
        expires: now + 3600,
        device_key_id: "offline-test-device".into(),
    };
    durable::replace(&hub.bearer(home), &serde_json::to_vec(&bearer).unwrap(), 0o600).unwrap();
}

fn rental(hub: &Hub, name: &str, id: &str, machine: Option<&Tensord>, key: &Path) -> MachineRow {
    MachineRow {
        kind: MachineKind::Rental,
        hub: hub.origin.clone(),
        rental_id: Some(id.into()),
        ..common::machine(name, machine.map(|m| m.endpoint.clone()), key)
    }
}

/// What the Hub publishes of a ready machine: its address, worker id and leaf.
fn ready_at(machine: &Tensord) -> Value {
    let ready = std::fs::read(machine.root.path().join("state/api-ready.json")).unwrap();
    let ready: Value = serde_json::from_slice(&ready).unwrap();
    json!({"worker_id": common::WORKER, "worker_address": machine.endpoint.address,
        "cert_pem": ready["cert_pem"], "provider": "vast"})
}

fn submit(on: &str, hub: &str) -> SubmitRun {
    SubmitRun {
        agreed: None,
        on: on.to_owned().into(),
        hub: "https://tensorhub.com".into(),
        kind: Kind::Warm,
        target: "cozy-test/held".into(),
        spec: common::warm_spec(hub, vec![]),
        inputs: vec![],
        out: None,
        warnings: vec![],
    }
}

/// Waits for `found` to answer something; a hang bound only, the product waits without one.
async fn eventually<T>(what: &str, mut found: impl AsyncFnMut() -> Option<T>) -> T {
    for _ in 0..1200 {
        if let Some(found) = found().await {
            return found;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("{what} never happened");
}

/// Calcifer with `rows` recorded, as a Calcifer that recorded them before it stopped starts.
fn start(home: &Path, rows: &[MachineRow]) -> (Arc<Calcifer>, Arc<Machines>) {
    let calcifer = Calcifer::open(&home.join("calcifer"), &home.join("outputs")).unwrap();
    calcifer.resume().unwrap();
    for row in rows {
        calcifer.put_machine(row).unwrap();
    }
    let machines = Machines::start(calcifer.clone(), Home::at(home), &Default::default());
    (calcifer, machines)
}

fn refusal(submitted: Result<protocol::calcifer::Submitted, calcifer::Error>) -> protocol::error::Error {
    match submitted {
        Err(calcifer::Error::Refused(error)) => error,
        other => panic!("a refusal, not {:?}", other.map(|s| s.run.number)),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rental_its_provider_stopped_is_refused_at_once_and_followed_back_at_a_new_address() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let (hub, run_hub) = (Hub::start().await, SilentHub::new());
    sign_in(&Home::at(home.path()), &hub.origin);
    let mut stopped = Tensord::start();
    stopped.stop();
    hub.says("r1", "degraded", json!({"provider": "vast",
        "failure": {"code": "readiness.provider_stopped.account_balance", "base_worker_image_digest": "sha256:0"}}));
    hub.says("r2", "booting", json!({}));
    let (calcifer, machines) = start(
        home.path(),
        &[rental(&hub, "nemuri", "r1", Some(&stopped), &key), rental(&hub, "kiyui", "r2", None, &key)],
    );
    machines.list(true).await.unwrap();

    // Its Hub's word is known: refused before anything is recorded, saying why and what to do.
    let error = refusal(calcifer.submit("k1", submit("nemuri", &run_hub.origin())).await);
    assert_eq!(error.code, "machine_offline", "{error:?}");
    assert_eq!(
        error.message,
        "nemuri is offline: vast stopped it (the Hub's balance there is spent)"
    );
    assert!(error.remedy.unwrap().contains("--on new"));
    assert!(!calcifer.store().recorded("k1").unwrap());
    let shown = machines.show("nemuri").await.unwrap();
    assert_eq!(shown.state, MachineState::Offline);
    assert_eq!(shown.reason.as_deref(), Some("vast stopped it (the Hub's balance there is spent)"));

    // A booting rental queues, as ever.
    let queued = calcifer.submit("k2", submit("kiyui", &run_hub.origin())).await.unwrap().run;
    assert_eq!(queued.status, Status::Queued);

    // Its provider resumes it at a new address: the Hub's word is followed there, and it serves.
    let resumed = Tensord::start();
    hub.says("r1", "ready", ready_at(&resumed));
    eventually("nemuri back at its new address", async || {
        let m = machines.list(false).await.ok()?.machines.into_iter().find(|m| m.name == "nemuri")?;
        (m.state == MachineState::Ready && m.address.as_deref() == Some(&resumed.endpoint.address))
            .then_some(())
    })
    .await;
    let back = calcifer.submit("k3", submit("nemuri", &run_hub.origin())).await.unwrap().run.id;
    eventually("the run accepted", async || {
        let state = calcifer.store().run(back.as_str()).unwrap()?.state;
        (state == RunState::Accepted).then_some(())
    })
    .await;
    calcifer.cancel(&back).unwrap();
    calcifer.cancel(&queued.id).unwrap();
    machines.close();
    calcifer.close();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_machine_that_answers_nothing_with_no_known_cause_is_refused_saying_since_when() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let (hub, run_hub) = (Hub::start().await, SilentHub::new());
    sign_in(&Home::at(home.path()), &hub.origin);
    let mut gone = Tensord::start();
    gone.stop();
    hub.says("r1", "ready", ready_at(&gone));
    let (calcifer, machines) =
        start(home.path(), &[rental(&hub, "nemuri", "r1", Some(&gone), &key), gone.row("box", &key)]);

    for (name, hub_word) in [("nemuri", true), ("box", false)] {
        let error = refusal(calcifer.submit(name, submit(name, &run_hub.origin())).await);
        assert_eq!(error.code, "machine_unreachable", "{error:?}");
        assert!(error.message.starts_with(&format!("{name} is unreachable (for ")), "{error:?}");
        assert_eq!(
            error.message.ends_with("; its Hub says its provider has not stopped it"),
            hub_word,
            "{error:?}"
        );
        assert!(!calcifer.store().recorded(name).unwrap());
    }
    machines.close();
    calcifer.close();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_warm_run_to_a_machine_that_answers_reads_no_hub_and_an_accepted_run_waits_out_its_stop() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let (hub, run_hub) = (Hub::start().await, SilentHub::new());
    sign_in(&Home::at(home.path()), &hub.origin);
    let mut machine = Tensord::start();
    hub.says("r1", "ready", ready_at(&machine));
    let (calcifer, machines) = start(home.path(), &[rental(&hub, "nemuri", "r1", Some(&machine), &key)]);
    eventually("its Status", async || {
        let m = machines.list(false).await.ok()?.machines.into_iter().find(|m| m.name == "nemuri")?;
        m.status.filter(|_| m.unreachable.is_none())
    })
    .await;
    let mut events = calcifer.subscribe();
    let before = hub.reads();
    let run = calcifer.submit("k1", submit("nemuri", &run_hub.origin())).await.unwrap().run.id;
    eventually("the run accepted", async || {
        let state = calcifer.store().run(run.as_str()).unwrap()?.state;
        (state == RunState::Accepted).then_some(())
    })
    .await;
    assert_eq!(hub.reads(), before, "a run to a machine that answers asks its Hub nothing");

    // Its provider stops it mid-run: the run waits, saying why, and is neither failed nor sent again.
    hub.says("r1", "degraded", json!({"provider": "vast",
        "failure": {"code": "readiness.provider_stopped", "base_worker_image_digest": "sha256:0"}}));
    machine.stop();
    until(&mut events, &run, |e| {
        matches!(e, Event::Waiting { reason } if reason == "nemuri is offline: vast stopped it")
    })
    .await;
    assert_eq!(calcifer.store().run(run.as_str()).unwrap().unwrap().state, RunState::Accepted);

    // The Hub ends it: the run ends with it.
    hub.says("r1", "released", json!({"release_cause": "idle_unreached"}));
    let failed = until(&mut events, &run, |e| matches!(e, Event::Failed { .. })).await;
    let Event::Failed { run: ended, .. } = failed else { unreachable!() };
    assert_eq!(ended.reason.unwrap().code, "machine_lost");
    machines.close();
    calcifer.close();
}
