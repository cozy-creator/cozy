//! The outbox and its followers against a real TensorD process (`COZY_TEST_TENSORD`). Runs are
//! warm runs of a release from a Hub that never answers, so they stay preparing until they are
//! canceled or the Hub goes away; no Runtime is involved.
mod common;

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
    time::Duration,
};

use calcifer::{
    Calcifer, Link, LinkError,
    store::{ImportedRun, State, Store},
};
use common::{SilentHub, Tensord, until};
use cozy_proto::{
    calcifer::{Event, Kind, On, Status, SubmitRun},
    ids::RunId,
    machine as v1,
};
use prost::Message as _;

/// Calcifer on `home`, taking up what an earlier one left, as `serve` does once it serves.
fn open(home: &Path) -> Arc<Calcifer> {
    let calcifer = Calcifer::open(&home.join("calcifer"), &home.join("outputs")).unwrap();
    calcifer.resume().unwrap();
    calcifer
}

fn submit(on: &str, spec: Vec<u8>, inputs: Vec<cozy_proto::calcifer::StagedInput>) -> SubmitRun {
    SubmitRun {
        on: On::from(on.to_owned()),
        hub: "https://tensorhub.com".into(),
        kind: Kind::Warm,
        target: "cozy-test/held".into(),
        spec,
        inputs,
        out: None,
        warnings: vec![],
    }
}

fn is_status(status: Status) -> impl Fn(&Event) -> bool {
    move |event| match event {
        Event::Queued { run, .. }
        | Event::InProgress { run, .. }
        | Event::Completed { run, .. }
        | Event::Failed { run, .. }
        | Event::Canceled { run, .. } => run.status == status,
        _ => false,
    }
}

fn reason_of(event: &Event) -> String {
    match event {
        Event::Failed { run, .. } | Event::Canceled { run, .. } => run.reason.clone().unwrap().code,
        other => panic!("not an end: {other:?}"),
    }
}

/// Waits until the store records the run settled.
async fn settled(calcifer: &Calcifer, id: &RunId) {
    for _ in 0..1200 {
        if calcifer.store().run(id.as_str()).unwrap().unwrap().state == State::Settled {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("run {id} never settled");
}

/// The machine's own answer about the run, with a link of the test's.
async fn on_machine(machine: &Tensord, id: &RunId) -> Result<String, LinkError> {
    let link = Link::new(&machine.endpoint, common::key())?;
    link.run(id.as_str(), 0, None)
        .await
        .map(|(state, _)| state.state)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_is_sent_with_its_input_accepted_and_canceled_by_its_machine() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = Tensord::start();
    let hub = SilentHub::new();
    let calcifer = open(home.path());
    calcifer.put_machine(&machine.row("m", &key)).unwrap();
    let mut events = calcifer.subscribe();
    let (staged, input) = common::input_file(home.path(), b"not really a png");
    // A file the spec refers to without naming it an input (a model's `object://` source).
    let (object, _) = common::input_file(home.path(), b"not really safetensors either");
    let spec = common::warm_spec(&hub.origin(), vec![input]);
    let given = vec![staged.clone(), object.clone()];
    let submitted = calcifer
        .submit("k1", submit("m", spec.clone(), given.clone()))
        .await
        .unwrap();
    let id = submitted.run.id.clone();
    assert!(!submitted.replay);
    until(&mut events, &id, is_status(Status::InProgress)).await;
    let staged_copy = home
        .path()
        .join("calcifer/inputs")
        .join(staged.digest.to_string().trim_start_matches("sha256:"));
    assert!(
        staged_copy.exists(),
        "the input stays staged while its run is unsettled"
    );
    let link = Link::new(&machine.endpoint, common::key()).unwrap();
    for written in [&staged, &object] {
        let held = link.held(&written.digest, written.length).await.unwrap();
        assert_eq!(held, written.length, "written before the send");
    }
    let again = calcifer
        .submit("k1", submit("m", spec, given))
        .await
        .unwrap();
    assert!(again.replay && again.run.id == id);

    calcifer.cancel(&id).unwrap();
    let end = until(&mut events, &id, |e| e.is_terminal()).await;
    assert_eq!(
        reason_of(&end),
        "canceled",
        "the machine's own outcome: {end:?}"
    );
    assert_eq!(on_machine(&machine, &id).await.unwrap(), "canceled");
    for _ in 0..100 {
        if !staged_copy.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(!staged_copy.exists(), "no unsettled run names the input");
}

/// Calcifer stops while its machine holds the run; the run ends without it (canceled at the
/// machine directly); the next Calcifer attaches from the durable cursor and records that end once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_settles_once_after_calcifer_restarts() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = Tensord::start();
    let hub = SilentHub::new();
    let first = open(home.path());
    first.put_machine(&machine.row("m", &key)).unwrap();
    let mut events = first.subscribe();
    let id = first
        .submit(
            "k",
            submit("m", common::warm_spec(&hub.origin(), vec![]), vec![]),
        )
        .await
        .unwrap()
        .run
        .id;
    until(&mut events, &id, is_status(Status::InProgress)).await;
    first.close();
    assert_eq!(
        first.store().run(id.as_str()).unwrap().unwrap().state,
        State::Accepted
    );
    drop(first);
    let link = Link::new(&machine.endpoint, common::key()).unwrap();
    link.cancel(id.as_str()).await.unwrap();
    assert_eq!(on_machine(&machine, &id).await.unwrap(), "canceled");

    let second = open(home.path());
    settled(&second, &id).await;
    let all = second.store().events_after(id.as_str(), 0).unwrap();
    assert_eq!(all.iter().filter(|e| e.is_terminal()).count(), 1, "{all:?}");
    assert_eq!(
        reason_of(all.last().unwrap()),
        "canceled",
        "the machine's own outcome"
    );
    let row = second.store().run(id.as_str()).unwrap().unwrap();
    assert!(row.cursor > 0, "the cursor reached the outcome");
    second.close();
    let third = open(home.path());
    assert_eq!(
        third.store().events_after(id.as_str(), 0).unwrap(),
        all,
        "nothing more was recorded"
    );
    third.close();
}

/// A run to a machine with no endpoint waits in the outbox, unsent, and goes as soon as the
/// machine is reachable; one canceled while it waits is never sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runs_wait_for_a_booting_machine_and_a_cancel_there_never_sends() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let hub = SilentHub::new();
    let calcifer = open(home.path());
    calcifer
        .put_machine(&common::machine("m", None, &key))
        .unwrap();
    let mut events = calcifer.subscribe();
    let waits = calcifer
        .submit(
            "k1",
            submit("m", common::warm_spec(&hub.origin(), vec![]), vec![]),
        )
        .await
        .unwrap()
        .run
        .id;
    let canceled = calcifer
        .submit(
            "k2",
            submit("m", common::warm_spec(&hub.origin(), vec![]), vec![]),
        )
        .await
        .unwrap()
        .run
        .id;
    let waiting = until(&mut events, &waits, |e| matches!(e, Event::Waiting { .. })).await;
    assert!(matches!(waiting, Event::Waiting { reason } if reason.contains("start")));
    calcifer.cancel(&canceled).unwrap();
    assert_eq!(
        calcifer
            .store()
            .run(canceled.as_str())
            .unwrap()
            .unwrap()
            .status,
        Some(Status::Canceled)
    );
    assert_eq!(
        calcifer.store().run(waits.as_str()).unwrap().unwrap().state,
        State::Queued
    );

    let machine = Tensord::start();
    calcifer.put_machine(&machine.row("m", &key)).unwrap();
    until(&mut events, &waits, is_status(Status::InProgress)).await;
    assert!(
        matches!(
            on_machine(&machine, &canceled).await,
            Err(LinkError::NotFound)
        ),
        "never sent"
    );
    calcifer.cancel(&waits).unwrap();
    until(&mut events, &waits, |e| e.is_terminal()).await;
    calcifer.close();
}

/// A Calcifer that crossed the send boundary and stopped before the machine got the run: the
/// next one attaches, hears NotFound, and sends it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sent_run_its_machine_never_got_is_sent_again_after_a_restart() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let hub = SilentHub::new();
    let first = open(home.path());
    first
        .put_machine(&common::machine("m", None, &key))
        .unwrap();
    // The spec names no Hub: the run's own Hub is filled in at the send.
    let spec = v1::RunSpec {
        kind: v1::RunKind::Warm as i32,
        source: Some(v1::run_spec::Source::Release(v1::Release {
            package: "cozy-test/held".into(),
            release: String::new(),
        })),
        ..Default::default()
    };
    let submitted = SubmitRun {
        hub: hub.origin(),
        ..submit("m", spec.encode_to_vec(), vec![])
    };
    let id = first.submit("k", submitted).await.unwrap().run.id;
    first.close();
    assert!(first.store().mark_sent(id.as_str(), "m").unwrap());
    drop(first);

    let machine = Tensord::start();
    let second = open(home.path());
    let mut events = second.subscribe();
    second.put_machine(&machine.row("m", &key)).unwrap();
    until(&mut events, &id, is_status(Status::InProgress)).await;
    assert_eq!(
        on_machine(&machine, &id).await.unwrap(),
        "preparing",
        "held by its Hub, not failed for want of one"
    );
    second.cancel(&id).unwrap();
    until(&mut events, &id, |e| e.is_terminal()).await;
    second.close();
}

/// A machine that served and is proven gone before it confirmed a run Calcifer placed: the run
/// returns to the outbox and is accepted elsewhere under the same id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_lost_before_acceptance_is_placed_again_under_its_id() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let hub = SilentHub::new();
    let mut gone = Tensord::start();
    let row = gone.row("a", &key);
    gone.stop();
    let other = Tensord::start();
    let calcifer = open(home.path());
    calcifer.put_machine(&row).unwrap();
    calcifer.put_machine(&other.row("b", &key)).unwrap();
    let mut events = calcifer.subscribe();
    let id = calcifer
        .submit(
            "k",
            submit("any", common::warm_spec(&hub.origin(), vec![]), vec![]),
        )
        .await
        .unwrap()
        .run
        .id;
    assert!(calcifer.place(&id, "a").unwrap());
    until(
        &mut events,
        &id,
        |e| matches!(e, Event::Waiting { reason } if reason.contains("unreachable")),
    )
    .await;
    assert_eq!(
        calcifer.store().run(id.as_str()).unwrap().unwrap().state,
        State::Sent
    );

    calcifer.machine_ended("a", "its rental ended").unwrap();
    let back = calcifer.store().run(id.as_str()).unwrap().unwrap();
    assert_eq!((back.state, back.machine), (State::Queued, None));
    assert!(calcifer.place(&id, "b").unwrap());
    until(&mut events, &id, is_status(Status::InProgress)).await;
    assert_eq!(on_machine(&other, &id).await.unwrap(), "preparing");
    calcifer.cancel(&id).unwrap();
    until(&mut events, &id, |e| e.is_terminal()).await;
    calcifer.close();
}

/// A machine proven gone after it accepted a run: the run ends FAILED and is never sent again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_lost_after_acceptance_fails_and_is_not_run_again() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let hub = SilentHub::new();
    let mut machine = Tensord::start();
    let calcifer = open(home.path());
    calcifer.put_machine(&machine.row("a", &key)).unwrap();
    let mut events = calcifer.subscribe();
    let id = calcifer
        .submit(
            "k",
            submit("any", common::warm_spec(&hub.origin(), vec![]), vec![]),
        )
        .await
        .unwrap()
        .run
        .id;
    calcifer.place(&id, "a").unwrap();
    until(&mut events, &id, is_status(Status::InProgress)).await;
    machine.stop();
    calcifer.machine_ended("a", "its rental ended").unwrap();
    let end = until(&mut events, &id, |e| e.is_terminal()).await;
    assert_eq!(reason_of(&end), "machine_lost");
    let row = calcifer.store().run(id.as_str()).unwrap().unwrap();
    assert_eq!(
        (row.state, row.machine.as_deref()),
        (State::Settled, Some("a"))
    );
    calcifer.close();
}

/// Runs imported from the Go CLI keep no spec: one its machine holds is followed from its cursor
/// and ends with the machine's own outcome; one its machine never got settles failed, never sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn imported_runs_are_followed_and_never_sent() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = Tensord::start();
    let hub = SilentHub::new();
    let held = RunId::try_from("req-held".to_owned()).unwrap();
    let never = RunId::try_from("req-never".to_owned()).unwrap();
    // The Go CLI sent this one and its machine accepted it.
    let spec = v1::RunSpec::decode(common::warm_spec(&hub.origin(), vec![]).as_slice()).unwrap();
    let link = Link::new(&machine.endpoint, common::key()).unwrap();
    link.run(held.as_str(), 0, Some(spec)).await.unwrap();
    let imported = |id: &RunId, number: u64, state: State| ImportedRun {
        id: id.clone(),
        number,
        kind: Kind::Warm,
        target: "cozy-test/held".into(),
        hub: hub.origin(),
        machine: Some("m".into()),
        out_dir: home.path().join("outputs"),
        state,
        cancel_requested: false,
        cursor: 0,
        machine_number: None,
        machine_state: None,
        status: None,
        reason: None,
        outcome: None,
        created_at_ms: 1,
        accepted_at_ms: None,
        settled_at_ms: None,
        outputs: vec![],
    };
    Store::open(&home.path().join("calcifer/state.db"))
        .unwrap()
        .import(
            &[machine.row("m", &key)],
            &[
                imported(&held, 41, State::Accepted),
                imported(&never, 42, State::Sent),
            ],
        )
        .unwrap();
    let calcifer = open(home.path());
    let mut events = calcifer.subscribe();
    settled(&calcifer, &never).await;
    let row = calcifer.store().run(never.as_str()).unwrap().unwrap();
    assert_eq!(row.reason.unwrap().code, "spec_not_kept");
    assert!(
        matches!(on_machine(&machine, &never).await, Err(LinkError::NotFound)),
        "never sent"
    );
    calcifer.cancel(&held).unwrap();
    let end = until(&mut events, &held, |e| e.is_terminal()).await;
    assert_eq!(
        reason_of(&end),
        "canceled",
        "the machine's own outcome: {end:?}"
    );
    assert_eq!(
        calcifer.store().run(held.as_str()).unwrap().unwrap().number,
        41
    );
    calcifer.close();
}

/// A run's inputs reach its machine as a visible stage, in steps of what the machine holds, and an
/// object two runs name is written to one machine once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inputs_are_written_as_a_visible_stage_and_once_per_machine() {
    let home = tempfile::tempdir().unwrap();
    let key = common::key_file(home.path());
    let machine = Tensord::start();
    let hub = SilentHub::new();
    let calcifer = open(home.path());
    calcifer.put_machine(&machine.row("m", &key)).unwrap();
    let mut events = calcifer.subscribe();
    let bytes: Vec<u8> = (0..24u32 << 20)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    let (staged, input) = common::input_file(home.path(), &bytes);
    let spec = common::warm_spec(&hub.origin(), vec![input]);
    let (first, second) = tokio::join!(
        calcifer.submit("k1", submit("m", spec.clone(), vec![staged.clone()])),
        calcifer.submit("k2", submit("m", spec, vec![staged.clone()])),
    );
    let ids = [first.unwrap().run.id, second.unwrap().run.id];
    let (mut writing, mut accepted) = (HashMap::<RunId, Vec<String>>::new(), HashSet::new());
    // A hang bound for a test whose runs never get there; the product waits without one.
    tokio::time::timeout(Duration::from_secs(120), async {
        while accepted.len() < 2 {
            let (id, event) = events.recv().await.unwrap();
            match event {
                Event::Waiting { reason } if reason.starts_with("writing image: ") => {
                    writing.entry(id).or_default().push(reason);
                }
                event if is_status(Status::InProgress)(&event) => {
                    accepted.insert(id);
                }
                _ => {}
            }
        }
    })
    .await
    .expect("both runs were accepted");
    assert_eq!(
        writing.len(),
        1,
        "one run wrote the object, the other found it held: {writing:?}"
    );
    let shown = writing.values().next().unwrap();
    assert_eq!(shown[0], "writing image: 0 B/24.0 MiB", "{shown:?}");
    assert!(shown.len() >= 2, "from nothing to all of it: {shown:?}");
    assert!(
        shown
            .last()
            .unwrap()
            .starts_with("writing image: 24.0 MiB/24.0 MiB at "),
        "{shown:?}"
    );
    let link = Link::new(&machine.endpoint, common::key()).unwrap();
    assert_eq!(
        link.held(&staged.digest, staged.length).await.unwrap(),
        staged.length
    );
    for id in &ids {
        calcifer.cancel(id).unwrap();
    }
    for id in &ids {
        settled(&calcifer, id).await;
    }
    calcifer.close();
}
