//! Calcifer's store on real SQLite files, each check across a close and reopen.
mod common;

use std::{path::Path, sync::Arc};

use calcifer::store::{Acceptance, Batch, NewRun, RunFilter, State, Store, StoreError};
use cozy_proto::{
    calcifer::{Call, Event, ItemStatus, Kind, Status, Warning},
    ids::{Digest, RunId},
    machine as v1,
};

fn new_run(n: u8, machine: Option<&str>, pinned: bool) -> NewRun {
    NewRun {
        id: RunId::mint([n; 12]),
        kind: Kind::Call,
        target: "fidika/sdxl/generate".into(),
        hub: "https://tensorhub.com".into(),
        machine: machine.map(str::to_owned),
        pinned,
        spec: common::warm_spec("https://tensorhub.com", vec![]),
        out_dir: "/tmp/outputs".into(),
        warnings: vec![],
        inputs: vec![],
        wants: None,
    }
}

fn open(dir: &Path) -> Store {
    Store::open(&dir.join("state.db")).unwrap()
}

fn state(id: &str) -> v1::RunState {
    v1::RunState {
        id: id.into(),
        number: 7,
        state: "preparing".into(),
        ..Default::default()
    }
}

#[test]
fn a_key_names_one_submission_forever_across_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store
        .put_machine(&common::machine("m", None, Path::new("/k")))
        .unwrap();
    let staged = Digest::of(b"an input");
    let warned = NewRun {
        warnings: vec![Warning {
            code: "unknown_field".into(),
            message: "dropped `steps`".into(),
        }],
        inputs: vec![(staged, 8)],
        ..new_run(1, Some("m"), true)
    };
    let (first, fresh) = store.submit("key-1", "sha256:a", warned).unwrap();
    assert!(fresh);
    assert_eq!((first.number, first.state), (1, State::Queued));
    let (second, _) = store
        .submit("key-2", "sha256:b", new_run(2, None, false))
        .unwrap();
    assert_eq!(second.number, 2);
    drop(store);

    let store = open(dir.path());
    let (again, fresh) = store
        .submit("key-1", "sha256:a", new_run(3, Some("m"), true))
        .unwrap();
    assert!(!fresh);
    assert_eq!(again.id, first.id);
    assert!(matches!(
        store.submit("key-1", "sha256:other", new_run(4, Some("m"), true)),
        Err(StoreError::Conflict(_))
    ));
    assert!(matches!(
        store.submit("key-5", "d", new_run(5, Some("nowhere"), true)),
        Err(StoreError::UnknownMachine(_))
    ));
    let listed = |filter: RunFilter| {
        let (rows, older) = store.runs(&filter).unwrap();
        (rows.iter().map(|r| r.number).collect::<Vec<_>>(), older)
    };
    let all = || RunFilter {
        limit: 10,
        ..Default::default()
    };
    assert_eq!(listed(all()), (vec![2, 1], 0));
    assert_eq!(listed(RunFilter { limit: 1, ..all() }), (vec![2], 1));
    assert_eq!(
        listed(RunFilter {
            before: Some(2),
            ..all()
        }),
        (vec![1], 0)
    );
    assert_eq!(
        listed(RunFilter {
            machine: Some("m".into()),
            ..all()
        }),
        (vec![1], 0)
    );
    assert_eq!(
        listed(RunFilter {
            target: Some("fidika/sdxl".into()),
            ..all()
        }),
        (vec![2, 1], 0)
    );
    assert_eq!(
        listed(RunFilter {
            target: Some("fidika/flux".into()),
            ..all()
        }),
        (vec![], 0)
    );
    assert_eq!(
        listed(RunFilter {
            status: Some(Status::Queued),
            ..all()
        }),
        (vec![2, 1], 0)
    );
    assert_eq!(
        listed(RunFilter {
            status: Some(Status::InProgress),
            ..all()
        }),
        (vec![], 0)
    );
    assert_eq!(store.run_by_number(2).unwrap().unwrap().id, second.id);
    let record = store.record(&first).unwrap();
    assert_eq!(record.warnings[0].code, "unknown_field");
    assert!(store.input_in_use(&staged).unwrap());
    store.request_cancel(first.id.as_str()).unwrap();
    assert!(
        !store.input_in_use(&staged).unwrap(),
        "a settled run holds no staged input"
    );
    assert_eq!(store.unsettled().unwrap(), vec![second.id]);
}

/// The send boundary and a cancel race on two threads over many runs: every run is either sent
/// (and its cancel waits for its machine) or canceled having never been sent.
#[test]
fn the_send_boundary_and_a_cancel_never_both_win() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(open(dir.path()));
    store
        .put_machine(&common::machine("m", None, Path::new("/k")))
        .unwrap();
    let ids: Vec<String> = (0..200u8)
        .map(|n| {
            let (row, _) = store
                .submit(&format!("k{n}"), "d", new_run(n, Some("m"), true))
                .unwrap();
            row.id.to_string()
        })
        .collect();
    let sender = {
        let (store, ids) = (store.clone(), ids.clone());
        std::thread::spawn(move || {
            ids.iter()
                .map(|id| store.mark_sent(id, "m").unwrap())
                .collect::<Vec<_>>()
        })
    };
    let canceler = {
        let (store, ids) = (store.clone(), ids.clone());
        std::thread::spawn(move || {
            for id in ids.iter().rev() {
                store.request_cancel(id).unwrap();
            }
        })
    };
    let sent = sender.join().unwrap();
    canceler.join().unwrap();
    drop(store);

    let store = open(dir.path());
    let (mut won, mut lost) = (0, 0);
    for (id, sent) in ids.iter().zip(sent) {
        let row = store.run(id).unwrap().unwrap();
        assert!(row.cancel_requested);
        if sent {
            won += 1;
            assert_eq!(
                row.state,
                State::Sent,
                "sent, so only its machine can end it"
            );
        } else {
            lost += 1;
            assert_eq!(
                (row.state, row.status),
                (State::Settled, Some(Status::Canceled))
            );
            assert!(
                !store.mark_sent(id, "m").unwrap(),
                "a canceled run is never sent"
            );
        }
    }
    assert_eq!(won + lost, 200);
}

#[test]
fn a_lost_machine_returns_unconfirmed_runs_under_their_ids_and_fails_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    let key = Path::new("/k");
    let served = calcifer::store::Endpoint {
        worker_id: "w".into(),
        address: "127.0.0.1:1".into(),
        pin: [0; 32],
    };
    store
        .put_machine(&common::machine("a", Some(served.clone()), key))
        .unwrap();
    store
        .put_machine(&common::machine("b", Some(served), key))
        .unwrap();
    store
        .put_machine(&common::machine("never-served", None, key))
        .unwrap();
    let submit = |n: u8, machine: &str, pinned: bool| {
        store
            .submit(&format!("k{n}"), "d", new_run(n, Some(machine), pinned))
            .unwrap()
            .0
            .id
            .to_string()
    };
    let (placed, pinned, accepted, canceled, queued) = (
        submit(1, "a", false),
        submit(2, "a", true),
        submit(3, "a", false),
        submit(4, "a", false),
        submit(5, "a", false),
    );
    let unserved = submit(6, "never-served", false);
    for id in [&placed, &pinned, &accepted, &canceled] {
        assert!(store.mark_sent(id, "a").unwrap());
    }
    assert!(matches!(
        store.accept(&accepted, "a", &state(&accepted)).unwrap(),
        Acceptance::Fresh(_)
    ));
    store.request_cancel(&canceled).unwrap();
    drop(store);

    let store = open(dir.path());
    store.machine_ended("a", "its rental ended").unwrap();
    store
        .machine_ended("never-served", "it never booted")
        .unwrap();
    drop(store);

    let store = open(dir.path());
    let row = |id: &str| store.run(id).unwrap().unwrap();
    for id in [&placed, &queued] {
        let back = row(id);
        assert_eq!(
            (back.state, back.machine.as_deref(), back.cursor),
            (State::Queued, None, 0),
            "back in the outbox"
        );
    }
    for (id, status, code) in [
        (&pinned, Status::Failed, "machine_lost"),
        (&accepted, Status::Failed, "machine_lost"),
        (&canceled, Status::Canceled, "canceled"),
        (&unserved, Status::Failed, "machine_lost"),
    ] {
        let ended = row(id);
        assert_eq!(
            (ended.state, ended.status),
            (State::Settled, Some(status)),
            "{id}"
        );
        assert_eq!(ended.reason.unwrap().code, code);
    }
    // The run placed again under its id; a late acceptance from the lost machine is refused.
    assert!(store.place(&placed, "b").unwrap());
    assert!(store.mark_sent(&placed, "b").unwrap());
    assert_eq!(
        store.accept(&placed, "a", &state(&placed)).unwrap(),
        Acceptance::Stale
    );
    assert!(matches!(
        store.accept(&placed, "b", &state(&placed)).unwrap(),
        Acceptance::Fresh(_)
    ));
    assert_eq!(
        store.accept(&placed, "b", &state(&placed)).unwrap(),
        Acceptance::Again
    );
}

fn product(output: &str, index: u32, rev: u64) -> v1::Product {
    v1::Product {
        output: output.into(),
        index,
        rev,
        length: rev * 10,
        digest: format!("sha256:{}", "ab".repeat(32)),
        media_type: "image/png".into(),
        ..Default::default()
    }
}

/// What a reconnecting stream replays comes from the columns, survives a reopen, and never
/// repeats what the client already holds.
#[test]
fn a_reconnect_replays_the_newest_picture_after_its_cursor() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store
        .put_machine(&common::machine("m", None, Path::new("/k")))
        .unwrap();
    let id = store
        .submit("k", "d", new_run(1, Some("m"), true))
        .unwrap()
        .0
        .id
        .to_string();
    assert!(store.mark_sent(&id, "m").unwrap());
    store.accept(&id, "m", &state(&id)).unwrap();
    let call = Call {
        run: id.clone(),
        index: 0,
        function: "generate".into(),
        label: String::new(),
        status: "succeeded".into(),
        called_at_ms: 1,
        finished_at_ms: 2,
        measurements: Some(serde_json::json!({"execution": {"gpus": [0]}})),
        reason: None,
        memoized: false,
    };
    let logged = Batch {
        cursor: 1 << 33,
        logs: vec![
            (1, "info".into(), "installing".into()),
            (2, "call".into(), serde_json::to_string(&call).unwrap()),
        ],
        products: vec![product("image", 0, 1)],
        machine_state: Some("running".into()),
    };
    let events = store.observe(&id, logged).unwrap();
    let seen = events
        .iter()
        .filter_map(Event::sequence_number)
        .max()
        .unwrap();
    store
        .observe(
            &id,
            Batch {
                cursor: 2 << 33,
                products: vec![product("image", 0, 2), product("image", 0, 1)],
                ..Default::default()
            },
        )
        .unwrap();
    drop(store);

    let store = open(dir.path());
    let row = store.run(&id).unwrap().unwrap();
    assert_eq!(
        (row.cursor, row.machine_state.as_deref()),
        (2 << 33, Some("running"))
    );
    let after = store.events_after(&id, seen).unwrap();
    assert!(
        matches!(&after[..], [Event::OutputDelta { rev: 2, .. }]),
        "{after:?}"
    );
    let outcome = v1::Outcome {
        status: "succeeded".into(),
        outputs: vec![product("image", 0, 3)],
        result: br#"{"caption":"a cat"}"#.to_vec(),
        measurements: br#"{"stages":[]}"#.to_vec(),
        execution_ms: 1234,
        ..Default::default()
    };
    let settled = store.settle(&id, &outcome, 3 << 33).unwrap();
    assert!(settled.last().unwrap().is_terminal());
    assert!(
        store.settle(&id, &outcome, 3 << 33).unwrap().is_empty(),
        "an outcome settles once"
    );
    drop(store);

    let store = open(dir.path());
    let all = store.events_after(&id, 0).unwrap();
    let kinds: Vec<_> = all
        .iter()
        .map(|e| serde_json::to_value(e).unwrap()["type"].clone())
        .collect();
    assert_eq!(
        kinds,
        ["run.log", "run.call", "output_item.done", "run.completed"]
    );
    let Event::OutputDone { item, .. } = &all[2] else {
        unreachable!()
    };
    assert_eq!(
        (item.rev, item.status, item.id.as_str()),
        (3, ItemStatus::Completed, "1/image")
    );
    assert!(item.sha256.is_some());
    let row = store.run(&id).unwrap().unwrap();
    let detail = store.detail(&row).unwrap();
    assert_eq!(detail.result, Some(serde_json::json!({"caption": "a cat"})));
    assert_eq!(detail.execution_ms, Some(1234));
    assert_eq!(detail.measurements, Some(serde_json::json!({"stages": []})));
    assert_eq!(detail.calls, vec![call]);
    let plain = store.record(&row).unwrap();
    assert!(plain.calls.is_empty() && plain.measurements.is_none() && plain.result.is_some());
    let numbers: Vec<_> = all.iter().filter_map(Event::sequence_number).collect();
    assert!(numbers.windows(2).all(|w| w[0] < w[1]));
    assert!(
        store
            .events_after(&id, *numbers.last().unwrap())
            .unwrap()
            .is_empty()
    );
}
