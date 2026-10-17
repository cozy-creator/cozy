//! Machines against real TensorD processes (`COZY_TEST_TENSORD`): added by address, watched
//! through their Status, and rentals that runs are placed on and wait for. No Hub is involved:
//! a rental is the row its Hub's answer would have recorded. The store tests need no TensorD.
mod common;

use std::{path::Path, sync::Arc, time::Duration};

use calcifer::{
    Calcifer,
    machines::Machines,
    store::{Ask, AskKind, AskState, MachineRow, State, Store},
};
use common::{SilentHub, Tensord, until};
use cozy_platform::{durable, lock::LockFile, paths::Home};
use cozy_proto::{
    calcifer::{
        Ended, Event, Holder, Kind, Machine, MachineKind, MachineList, MachineLog, MachineState,
        Status, SubmitRun,
    },
    error::Envelope,
    ids::{Digest, RunId},
    machine as v1,
};
use ed25519_dalek::pkcs8::EncodePrivateKey as _;

/// A Hub no test signs in to: what a rental's Hub would be had a Hub answered for it.
const NO_HUB: &str = "http://127.0.0.1:9";

fn calcifer(home: &Path) -> Arc<Calcifer> {
    let calcifer = Calcifer::open(&home.join("calcifer"), &home.join("outputs")).unwrap();
    calcifer.resume().unwrap();
    calcifer
}

fn machines(calcifer: &Arc<Calcifer>, home: &Path) -> Arc<Machines> {
    Machines::start(calcifer.clone(), Home::at(home), &Default::default())
}

/// Makes the test key, which every test TensorD admits, this controller's owner key.
fn admit(home: &Path) {
    let path = Home::at(home).owner_key();
    durable::private_dir(path.parent().unwrap()).unwrap();
    let pem = common::key().to_pkcs8_pem(Default::default()).unwrap();
    durable::replace(&path, pem.as_bytes(), 0o600).unwrap();
}

fn rental(name: &str, id: &str, machine: Option<&Tensord>, key: &Path) -> MachineRow {
    MachineRow {
        kind: MachineKind::Rental,
        hub: NO_HUB.into(),
        rental_id: Some(id.into()),
        ..common::machine(name, machine.map(|m| m.endpoint.clone()), key)
    }
}

fn submit(on: &str, spec: v1::RunSpec) -> SubmitRun {
    SubmitRun {
        on: on.to_owned().into(),
        hub: "https://tensorhub.com".into(),
        kind: Kind::Warm,
        target: "cozy-test/held".into(),
        spec,
        inputs: vec![],
        out: None,
        warnings: vec![],
    }
}

fn gpus(spec: v1::RunSpec, gpus: u32) -> v1::RunSpec {
    v1::RunSpec { gpus, ..spec }
}

fn run_of(event: &Event) -> Option<&cozy_proto::calcifer::Run> {
    match event {
        Event::Queued { run, .. }
        | Event::InProgress { run, .. }
        | Event::Completed { run, .. }
        | Event::Failed { run, .. }
        | Event::Canceled { run, .. } => Some(run),
        _ => None,
    }
}

fn reason(event: &Event) -> String {
    run_of(event).unwrap().reason.clone().unwrap().code
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

async fn listed(machines: &Arc<Machines>, name: &str) -> Option<Machine> {
    let list = machines.list(false).await.unwrap();
    list.machines.into_iter().find(|m| m.name == name)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_machine_is_added_only_at_its_fingerprint_with_this_controllers_key_admitted() {
    let home = tempfile::tempdir().unwrap();
    let machine = Tensord::start();
    let calcifer = calcifer(home.path());
    let machines = machines(&calcifer, home.path());
    let address = machine.endpoint.address.clone();
    let fingerprint = Digest::from_bytes(machine.endpoint.pin);
    let refused = |e: cozy_proto::error::Error| e.code;

    let wrong = machines
        .add("box", &address, &Digest::of(b"another leaf"))
        .await;
    assert_eq!(wrong.map_err(refused).unwrap_err(), "fingerprint_mismatch");
    let unauthorized = machines
        .add("box", &address, &fingerprint)
        .await
        .unwrap_err();
    assert_eq!(unauthorized.code, "machine_unauthorized");
    let public = machines.controller_key().unwrap();
    assert_eq!(
        unauthorized.remedy.as_deref(),
        Some(format!("on that computer: cozy machine authorize {public}").as_str())
    );
    let unreachable = machines.add("box", "127.0.0.1:9", &fingerprint).await;
    assert_eq!(
        unreachable.map_err(refused).unwrap_err(),
        "machine_unreachable"
    );
    let invalid = machines.add("any", &address, &fingerprint).await;
    assert_eq!(
        invalid.map_err(refused).unwrap_err(),
        "machine_name_invalid"
    );
    assert!(
        calcifer.store().machine("box").unwrap().is_none(),
        "nothing is recorded before the machine proves itself"
    );

    admit(home.path());
    let added = machines.add("box", &address, &fingerprint).await.unwrap();
    assert_eq!(
        (added.kind, added.state),
        (MachineKind::Added, MachineState::Ready)
    );
    assert_eq!(added.status.unwrap().worker_id, common::WORKER);
    let taken = machines.add("box", &address, &fingerprint).await;
    assert_eq!(taken.map_err(refused).unwrap_err(), "machine_name_taken");
    machines
        .logs("box", MachineLog::Memory, 4096)
        .await
        .unwrap();

    let ended = machines.end("box", false).await.unwrap();
    assert!(ended.changed);
    assert_eq!(ended.machine.state, MachineState::Ended);
    assert!(
        !machines.end("box", false).await.unwrap().changed,
        "an end is said once"
    );
    assert!(listed(&machines, "box").await.is_none());

    let local = machines.put_local(machine.endpoint.clone()).unwrap();
    assert_eq!(local.kind, MachineKind::Local);
    let local_end = machines.end("local", false).await.unwrap_err();
    assert_eq!(
        (local_end.code.as_str(), local_end.next.as_slice()),
        ("machine_local", &["cozy machine disable".to_string()][..])
    );
    machines.close();
    calcifer.close();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_status_watch_shows_the_machine_and_survives_its_restart() {
    let home = tempfile::tempdir().unwrap();
    admit(home.path());
    let mut machine = Tensord::start();
    let calcifer = calcifer(home.path());
    let machines = machines(&calcifer, home.path());
    let fingerprint = Digest::from_bytes(machine.endpoint.pin);
    machines
        .add("box", &machine.endpoint.address, &fingerprint)
        .await
        .unwrap();
    let first = eventually("a Status", async || listed(&machines, "box").await?.status).await;
    assert!(
        ["booting", "ready"].contains(&first.phase.as_str()),
        "{first:?}"
    );
    assert_eq!(
        first.idle_deadline_ms, None,
        "a machine that never releases itself"
    );
    assert!(first.started_at_ms > 0);

    machine.stop();
    eventually("the watch to see the machine gone", async || {
        listed(&machines, "box").await?.unreachable
    })
    .await;
    let row = listed(&machines, "box").await.unwrap();
    assert_eq!(
        row.state,
        MachineState::Ready,
        "an unreachable machine is never lost"
    );

    machine.restart();
    let again = eventually("the restarted machine's Status", async || {
        let row = listed(&machines, "box").await?;
        row.status.filter(|_| row.unreachable.is_none())
    })
    .await;
    assert_eq!(again.worker_id, common::WORKER);
    machines.close();
    calcifer.close();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_any_run_waits_for_a_rental_that_fits_and_goes_to_it() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let hub = SilentHub::new();
    let machine = Tensord::start();
    let calcifer = calcifer(home.path());
    let mut events = calcifer.subscribe();
    let machines = machines(&calcifer, home.path());
    let spec = || common::warm_spec(&hub.origin(), vec![]);
    let state = |id: &RunId| calcifer.store().run(id.as_str()).unwrap().unwrap();

    // No rental yet: the run waits in the outbox, unplaced.
    let early = calcifer
        .submit("k0", submit("any", spec()))
        .await
        .unwrap()
        .run
        .id;
    until(&mut events, &early, |e| matches!(e, Event::Waiting { .. })).await;
    assert_eq!(state(&early).machine, None);
    let bare = calcifer
        .submit("k1", submit("new", spec()))
        .await
        .unwrap()
        .run
        .id;
    // A bare `new` asks its Hub where to run it, and this computer is signed in to none.
    let end = until(&mut events, &bare, Event::is_terminal).await;
    assert_eq!(reason(&end), "signed_out");
    machines.close();

    // A rental its Hub answered for, recorded before this Calcifer started: the waiting run
    // goes to it, and so does the next one.
    calcifer
        .put_machine(&rental("kiyui", "r1", Some(&machine), &key))
        .unwrap();
    let machines = self::machines(&calcifer, home.path());
    let fits = calcifer
        .submit("k2", submit("any", spec()))
        .await
        .unwrap()
        .run
        .id;
    eventually("both runs accepted on the rental", async || {
        [&early, &fits]
            .iter()
            .all(|id| state(id).state == State::Accepted)
            .then_some(())
    })
    .await;
    assert_eq!(state(&early).machine.as_deref(), Some("kiyui"));
    assert_eq!(state(&fits).machine.as_deref(), Some("kiyui"));
    let row = listed(&machines, "kiyui").await.unwrap();
    assert_eq!((row.running, row.recorded), (2, true));

    // A machine with no GPUs fits no run of 4: that one waits, and a cancel ends it unsent.
    eventually("the rental to be ready", async || {
        listed(&machines, "kiyui")
            .await?
            .status
            .filter(|s| s.phase == "ready")
    })
    .await;
    let too_big = calcifer
        .submit("k3", submit("any", gpus(spec(), 4)))
        .await
        .unwrap()
        .run
        .id;
    until(&mut events, &too_big, |e| {
        matches!(e, Event::Waiting { .. })
    })
    .await;
    calcifer.cancel(&too_big).unwrap();
    let canceled = state(&too_big);
    assert_eq!(
        (canceled.machine, canceled.status),
        (None, Some(Status::Canceled))
    );

    for id in [&early, &fits] {
        calcifer.cancel(id).unwrap();
    }
    eventually("both runs ended", async || {
        [&early, &fits]
            .iter()
            .all(|id| state(id).state == State::Settled)
            .then_some(())
    })
    .await;
    machines.close();
    calcifer.close();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_wait_for_a_booting_rental_through_its_rename_and_go_when_its_address_arrives() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let hub = SilentHub::new();
    let calcifer = calcifer(home.path());
    calcifer
        .put_machine(&rental("rent-0a1b", "r2", None, &key))
        .unwrap();
    let mut events = calcifer.subscribe();
    let machines = machines(&calcifer, home.path());
    let early = calcifer
        .submit(
            "k1",
            submit("rent-0a1b", common::warm_spec(&hub.origin(), vec![])),
        )
        .await
        .unwrap()
        .run
        .id;
    let booting = |e: &Event| matches!(e, Event::Waiting { reason } if reason.contains("start"));
    until(&mut events, &early, booting).await;

    // The Hub names the pod while it boots: the run moves with it, and the old name still works.
    assert!(calcifer.rename_machine("rent-0a1b", "kiyui").unwrap());
    let moved = calcifer.store().run(early.as_str()).unwrap().unwrap();
    assert_eq!(moved.machine.as_deref(), Some("kiyui"));
    let late = calcifer
        .submit(
            "k2",
            submit("rent-0a1b", common::warm_spec(&hub.origin(), vec![])),
        )
        .await
        .unwrap()
        .run;
    assert_eq!(late.machine.as_deref(), Some("kiyui"));
    let by_id = calcifer
        .submit("k3", submit("r2", common::warm_spec(&hub.origin(), vec![])))
        .await
        .unwrap()
        .run;
    assert_eq!(by_id.machine.as_deref(), Some("kiyui"));
    let shown = machines.show("rent-0a1b").await.unwrap();
    assert_eq!(
        (shown.name.as_str(), shown.alias.as_deref()),
        ("kiyui", Some("rent-0a1b"))
    );
    assert_eq!((shown.state, shown.queued), (MachineState::Booting, 3));

    // Its Hub publishes the address: every run waiting for it is sent.
    let machine = Tensord::start();
    calcifer
        .put_machine(&rental("kiyui", "r2", Some(&machine), &key))
        .unwrap();
    let ids = [&early, &late.id, &by_id.id];
    let state = |id: &RunId| calcifer.store().run(id.as_str()).unwrap().unwrap().state;
    eventually("every run accepted", async || {
        ids.iter()
            .all(|id| state(id) == State::Accepted)
            .then_some(())
    })
    .await;
    for id in ids {
        calcifer.cancel(id).unwrap();
    }
    eventually("every run ended", async || {
        ids.iter()
            .all(|id| state(id) == State::Settled)
            .then_some(())
    })
    .await;
    machines.close();
    calcifer.close();
}

#[test]
fn asks_and_renames_survive_a_restart() {
    let home = tempfile::tempdir().unwrap();
    let path = home.path().join("state.db");
    let store = Store::open(&path).unwrap();
    store
        .put_machine(&rental("rent-0a1b", "r3", None, Path::new("/k")))
        .unwrap();
    let ask = Ask {
        key: "key-1".into(),
        kind: AskKind::Rent,
        hub: NO_HUB.into(),
        digest: "intent".into(),
        body: b"{}".to_vec(),
        machine: Some("rent-0a1b".into()),
        run: None,
        state: AskState::Pending,
        reason: None,
    };
    assert!(
        store.begin_ask(&ask).unwrap().1,
        "recorded before it is asked"
    );
    assert_eq!(store.begin_ask(&ask).unwrap(), (ask.clone(), false));
    let other = Ask {
        digest: "another intent".into(),
        ..ask.clone()
    };
    assert!(store.begin_ask(&other).is_err(), "one key, one intent");
    assert!(store.rename_machine("rent-0a1b", "kiyui").unwrap());
    drop(store);

    let store = Store::open(&path).unwrap();
    assert_eq!(
        store.pending_asks().unwrap(),
        vec![Ask {
            machine: Some("kiyui".into()),
            ..ask
        }]
    );
    let renamed = store.find_machine("rent-0a1b").unwrap().unwrap();
    assert_eq!(
        (renamed.name.as_str(), renamed.rental_id.as_deref()),
        ("kiyui", Some("r3"))
    );
    assert_eq!(store.find_machine("r3").unwrap().unwrap().name, "kiyui");
    store.settle_ask("key-1", Ok(None)).unwrap();
    assert!(store.pending_asks().unwrap().is_empty());
    assert_eq!(
        store.rent_ask("kiyui").unwrap().unwrap().state,
        AskState::Done
    );
    store
        .put_machine(&MachineRow {
            served: true,
            ..rental("kiyui", "r3", None, Path::new("/k"))
        })
        .unwrap();
    assert!(
        !store.rename_machine("kiyui", "other").unwrap(),
        "a machine that served keeps its name"
    );

    // Its Hub frees the name once it ends, and the next rental its host boots takes it.
    store.machine_ended("kiyui", "released").unwrap();
    store
        .put_machine(&rental("rent-ffff", "r4", None, Path::new("/k")))
        .unwrap();
    assert!(store.rename_machine("rent-ffff", "kiyui").unwrap());
    assert_eq!(
        store
            .find_machine("kiyui")
            .unwrap()
            .unwrap()
            .rental_id
            .as_deref(),
        Some("r4")
    );
    let history = store.find_machine("r3").unwrap().unwrap();
    assert!(history.name.starts_with("kiyui@"), "{}", history.name);
    assert_eq!(history.state, MachineState::Ended);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rent_at_a_hub_this_computer_is_not_signed_in_to_records_nothing() {
    let home = tempfile::tempdir().unwrap();
    let calcifer = calcifer(home.path());
    let machines = machines(&calcifer, home.path());
    let refused = machines
        .rent("key-1", NO_HUB, "h100", None, None)
        .await
        .unwrap_err();
    assert_eq!(refused.code, "signed_out");
    assert!(calcifer.store().ask("key-1").unwrap().is_none());
    assert!(
        !Home::at(home.path()).keys().exists(),
        "no rental key was minted"
    );
    machines.close();
    calcifer.close();
}

/// The machine routes, over loopback HTTP against a serving Calcifer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_machine_routes_answer_over_loopback() {
    let home = tempfile::tempdir().unwrap();
    admit(home.path());
    let machine = Tensord::start();
    let loaded = cozy_proto::config::Loaded {
        config: Default::default(),
        home: Home::at(home.path()),
        user_home: home.path().to_owned(),
        warnings: vec![],
    };
    let served = tokio::spawn(async move { calcifer::serve::serve(&loaded).await });
    let lock = LockFile::new(Home::at(home.path()).calcifer_lock());
    let holder: Holder = eventually("Calcifer's record", async || {
        let bytes = lock.holder().ok()??;
        serde_json::from_slice(&bytes).ok()
    })
    .await;
    let client = reqwest::Client::new();
    let call = |method: reqwest::Method, path: &str| {
        client
            .request(method, format!("http://{}{path}", holder.address))
            .bearer_auth(&holder.token)
    };
    use reqwest::{Method, StatusCode};

    let added = call(Method::POST, "/v1/machines")
        .json(
            &serde_json::json!({"kind": "add", "name": "box", "address": machine.endpoint.address,
            "fingerprint": Digest::from_bytes(machine.endpoint.pin)}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(added.status(), StatusCode::OK);
    assert_eq!(added.json::<Machine>().await.unwrap().name, "box");
    let list: MachineList = call(Method::GET, "/v1/machines?refresh=true")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list.machines.len(), 1);
    assert_eq!(list.signed_out, vec!["https://tensorhub.com".to_string()]);
    let shown: Machine = call(Method::GET, "/v1/machines/box")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(shown.status.unwrap().worker_id, common::WORKER);
    let log = call(Method::GET, "/v1/machines/box/logs/memory?lines=10")
        .send()
        .await
        .unwrap();
    assert_eq!(log.status(), StatusCode::OK);
    let refused = |status: StatusCode, code: &'static str| {
        move |response: reqwest::Response| async move {
            assert_eq!(response.status(), status);
            let envelope: Envelope = response.json().await.unwrap();
            assert_eq!(envelope.error.code, code);
        }
    };
    let unknown = call(Method::GET, "/v1/machines/box/logs/kernel").send();
    refused(StatusCode::NOT_FOUND, "machine_log_unknown")(unknown.await.unwrap()).await;
    let keyless = call(Method::POST, "/v1/machines")
        .json(&serde_json::json!({"kind": "rent", "sku": "h100", "hub": "https://tensorhub.com"}))
        .send();
    refused(StatusCode::BAD_REQUEST, "idempotency_key_required")(keyless.await.unwrap()).await;
    let nobody = call(Method::DELETE, "/v1/machines/kiyui").send();
    refused(StatusCode::NOT_FOUND, "machine_not_found")(nobody.await.unwrap()).await;
    let ended = call(Method::DELETE, "/v1/machines/box")
        .send()
        .await
        .unwrap();
    assert_eq!(ended.status(), StatusCode::OK);
    assert!(ended.json::<Ended>().await.unwrap().changed);

    call(Method::POST, "/v1/down").send().await.unwrap();
    drop(client);
    served.await.unwrap().unwrap();
}

/// A run that publishes into a Hub model carries its owner's capability for it; on a Hub this
/// computer is not signed in to there is none, so the run fails before its machine gets it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_publishing_run_without_its_hub_sign_in_fails_unsent() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = Tensord::start();
    let calcifer = calcifer(home.path());
    calcifer.put_machine(&machine.row("m", &key)).unwrap();
    let mut events = calcifer.subscribe();
    let machines = machines(&calcifer, home.path());
    let mut spec = common::warm_spec(NO_HUB, vec![]);
    spec.weights_destination = "fidika/uploaded".into();
    let mut publish = submit("m", spec);
    publish.hub = NO_HUB.into();
    let id = calcifer.submit("k", publish).await.unwrap().run.id;
    let end = until(&mut events, &id, Event::is_terminal).await;
    assert_eq!(reason(&end), "hub_sign_in_required");
    let link = calcifer::Link::new(&machine.endpoint, common::key()).unwrap();
    assert!(
        matches!(
            link.attach(id.as_str(), 0).await,
            Err(calcifer::LinkError::NotFound)
        ),
        "never sent"
    );
    machines.close();
    calcifer.close();
}

/// Local code's org-relative references name the run Hub's account, which a computer signed out
/// there does not have: the run reaches its machine with no owner.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_signed_out_local_run_reaches_its_machine_with_no_owner() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = common::Machine::start();
    let calcifer = calcifer(home.path());
    calcifer.put_machine(&machine.row("m", &key)).unwrap();
    let machines = machines(&calcifer, home.path());
    let (staged, spec) = common::local_spec(home.path());
    let mut run = submit("m", spec.clone());
    run.inputs = vec![staged];
    let id = calcifer.submit("k", run).await.unwrap().run.id;
    eventually("the run accepted", async || {
        let row = calcifer.store().run(id.as_str()).unwrap().unwrap();
        row.accepted_at_ms.map(drop)
    })
    .await;
    let sent = |owner: &str| v1::RunSpec {
        owner: owner.into(),
        hub: Some(v1::HubAccess {
            origin: "https://tensorhub.com".into(),
            ..Default::default()
        }),
        ..spec.clone()
    };
    let link = calcifer::Link::new(&machine.endpoint, common::key()).unwrap();
    assert!(common::holds_as(&link, &id, sent("")).await);
    assert!(!common::holds_as(&link, &id, sent("someone")).await);
    machines.close();
    calcifer.close();
}
