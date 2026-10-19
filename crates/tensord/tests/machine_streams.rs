//! The machine API's streams through the real TLS server and client, over a backend with a fixed
//! log: what a reattaching or output-limited observer sees, when a stream's authority ends, and
//! how a short output read fails.
use tensord::api::{
    self,
    auth::{Keys, VerifiedActor},
    backend::OutputSnapshot,
    capability::{self, Grant},
    domain, v1, MachineBackend, MachineIdentity,
};
use protocol::error::Error;
use ed25519_dalek::SigningKey;
use std::{
    io::{Seek, Write},
    pin::Pin,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_stream::{Stream, StreamExt};
use v1::{
    client::{Auth, Machine},
    Event,
};

/// A finished run whose log revises `image` twice, sets `private`, reports author progress
/// and fails with a result and a triage bundle; a live run's log only grows.
struct Run {
    live: bool,
    /// Bytes `open_output` holds of the 8 it declares.
    held: usize,
}
impl MachineBackend for Run {
    fn workspace(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionWorkspaceQuery,
    ) -> Result<domain::MachineExecutionWorkspace, Error> {
        Ok(domain::MachineExecutionWorkspace { execution_workspace_id: "w".into(), ..Default::default() })
    }
    fn get(
        &self,
        _: VerifiedActor,
        _: domain::MachineExecutionQuery,
    ) -> Result<domain::MachineExecutionState, Error> {
        let state = if self.live { "running" } else { "failed" };
        Ok(domain::MachineExecutionState { number: 1, state: state.into(), ..Default::default() })
    }
    fn events(
        &self,
        _: VerifiedActor,
        query: domain::MachineExecutionEventsQuery,
    ) -> Result<domain::MachineExecutionEventPage, Error> {
        let progress = |sequence| domain::MachineExecutionEvent {
            sequence,
            kind: "progress".into(),
            body_canonical_bytes: br#"{"payload":{"stage":"the author's prompt"}}"#.to_vec(),
            ..Default::default()
        };
        let events = if self.live {
            std::thread::sleep(Duration::from_millis(20));
            vec![progress(query.after + 1)]
        } else {
            let product = |sequence, output: &str, byte| domain::MachineExecutionEvent {
                sequence,
                kind: "product".into(),
                product: Some(domain::RunProduct {
                    output: output.into(),
                    op: domain::RunProductOp::Set as i32,
                    content: Some(domain::Ref { digest: vec![byte; 32], length: 8 }),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let outcome = br#"{"status":3,"safe_message":"author_failed: the author's secret","result":{"inline_result":"InNlY3JldCI="},"triage_bundle":{}}"#;
            vec![
                product(1, "image", 1),
                product(2, "image", 2),
                product(3, "private", 3),
                progress(4),
                domain::MachineExecutionEvent {
                    sequence: 5,
                    kind: "outcome".into(),
                    outcome: Some(domain::AttemptOutcome {
                        outcome_canonical_bytes: outcome.to_vec(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ]
            .into_iter()
            .filter(|event| event.sequence > query.after)
            .collect()
        };
        Ok(domain::MachineExecutionEventPage { events, next_after: query.after + 1, ..Default::default() })
    }
    fn measurements(&self, _: VerifiedActor, _: domain::MachineExecutionQuery) -> Result<Option<Vec<u8>>, Error> {
        Ok(Some(br#"{"device":"the owner's"}"#.to_vec()))
    }
    fn open_output(
        &self,
        _: VerifiedActor,
        _: u64,
        _: &str,
        _: Option<u32>,
    ) -> Result<OutputSnapshot, Error> {
        let path = std::env::temp_dir().join(format!("cm-v1-output-{}", uuid::Uuid::new_v4()));
        let mut file = std::fs::File::options().read(true).write(true).create_new(true).open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        file.write_all(&b"8 bytes!"[..self.held]).unwrap();
        // A whole output is 128 MiB, sparse: a read stays open behind TCP flow control.
        let length = if self.held == 8 { 128 << 20 } else { 8 };
        if self.held == 8 {
            file.set_len(length).unwrap();
        }
        file.rewind().unwrap();
        Ok(OutputSnapshot {
            parts: vec![(file, length)],
            length,
            rev: 1,
            media_type: "application/octet-stream".into(),
            sha256: None,
        })
    }
}

const SIGNER: [u8; 32] = [91; 32];

fn cap(grant: Grant) -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
    let expires = if grant.expires == 0 { now + 60 } else { grant.expires };
    capability::mint(&SigningKey::from_bytes(&SIGNER), Grant { machine: "m".into(), expires, ..grant })
}
fn machine_cap() -> String {
    cap(Grant { action: capability::MACHINE.into(), ..Default::default() })
}

async fn serve(run: Run) -> (Machine, Keys) {
    let identity =
        MachineIdentity::ephemeral("m".into(), vec![SigningKey::from_bytes(&SIGNER).verifying_key()], vec![7; 32])
            .unwrap();
    let (keys, pin) = (identity.authority.keys.clone(), tensorfs::sha256::digest(&identity.cert_der));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { api::serve(listener, identity, Arc::new(run)).await.unwrap() });
    let machine = Machine::new(&address.to_string(), pin, Auth::Token(machine_cap())).unwrap();
    (machine, keys)
}

type Events = Pin<Box<dyn Stream<Item = Result<v1::RunEvent, v1::client::Error>> + Send>>;

async fn attach(client: &Machine, token: &str, after: u64) -> Events {
    let mut events: Events =
        Box::pin(client.with_auth(Auth::Token(token.into())).events("run", after).await.unwrap());
    let snapshot = events.next().await.unwrap().unwrap();
    assert!(matches!(snapshot.event, Event::State(_)) && snapshot.sequence == 0);
    events
}

/// The bytes of `image` in `[first, end)` (empty: all of it) and the output's length, or how the
/// read failed: refused, or its body broken off.
async fn read(client: &Machine, (first, end): (u64, u64)) -> Result<(u64, Vec<u8>), v1::client::Error> {
    let target = v1::OutputTarget { run: "run".into(), output: "image".into(), ..Default::default() };
    let output = client.bulk().unwrap().get(&target, (first, end), 0).await?;
    let mut body = output.body;
    let mut bytes = vec![];
    while let Some(chunk) = body.next().await {
        bytes.extend_from_slice(&chunk?);
    }
    Ok((output.length, bytes))
}

/// A reattach (every daemon restart) past earlier products keeps each output's revision and
/// the outcome's whole inventory; a cursor at or past the outcome gets the snapshot only.
#[tokio::test]
async fn a_reattach_keeps_revisions_and_the_outcome_keeps_every_output() {
    let (client, _) = serve(Run { live: false, held: 8 }).await;
    let token = machine_cap();
    let mut events = attach(&client, &token, 1).await;
    let Event::Product(product) = events.next().await.unwrap().unwrap().event else {
        panic!("a product follows the snapshot")
    };
    assert_eq!((product.output.as_str(), product.rev), ("image", 2));

    let mut events = attach(&client, &token, 3).await;
    let outcome = loop {
        if let Event::Outcome(outcome) = events.next().await.unwrap().unwrap().event {
            break outcome;
        }
    };
    let outputs: Vec<_> = outcome.outputs.iter().map(|p| (p.output.as_str(), p.rev)).collect();
    assert_eq!(outputs, [("image", 2), ("private", 1)]);
    assert!(outcome.measurements.is_some() && outcome.result.is_some());

    for after in [5, 100] {
        assert!(attach(&client, &token, after).await.next().await.is_none());
    }
}

/// `run play`'s output-limited link sees its outputs and the outcome's status, never the
/// author's progress, result, failure text, measurements or triage.
#[tokio::test]
async fn an_output_limited_grant_sees_only_its_outputs() {
    let (client, _) = serve(Run { live: false, held: 8 }).await;
    let token = cap(Grant { run: "run".into(), outputs: vec!["image".into()], ..Default::default() });
    let mut events = attach(&client, &token, 0).await;
    let mut outcome = None;
    while let Some(event) = events.next().await {
        match event.unwrap().event {
            Event::Product(product) => assert_eq!(product.output, "image"),
            Event::Outcome(seen) => outcome = Some(seen),
            Event::State(state) => assert!(state.waiting.is_empty()),
            other => panic!("the limited grant saw {other:?}"),
        }
    }
    let outcome = outcome.unwrap();
    assert_eq!(outcome.status, "failed");
    assert_eq!(outcome.outputs.iter().map(|p| p.output.as_str()).collect::<Vec<_>>(), ["image"]);
    assert!(outcome.result.is_none() && outcome.reason.is_none() && !outcome.triage);
    assert!(outcome.measurements.is_none());
    let limited = client.with_auth(Auth::Token(token));
    assert_eq!(limited.triage("run").await.unwrap_err().status(), Some(403));
}

/// A run observer and a read outlive their capability's expiry, which held when they opened;
/// revoking the key that opened them ends both.
#[tokio::test]
async fn open_streams_end_on_revocation_not_at_expiry() {
    let (client, keys) = serve(Run { live: true, held: 8 }).await;
    let expires = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64 + 2;
    let token = cap(Grant { action: capability::MACHINE.into(), expires, ..Default::default() });
    let mut events = attach(&client, &token, 0).await;
    let target = v1::OutputTarget { run: "run".into(), output: "image".into(), ..Default::default() };
    let bulk = client.with_auth(Auth::Token(token)).bulk().unwrap();
    let output = bulk.get(&target, (0, 0), 0).await.unwrap();
    assert_eq!(output.length, 128 << 20);
    let mut bytes = output.body;
    let past_expiry = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < past_expiry {
        events.next().await.expect("the observer outlived its cap").unwrap();
    }
    keys.revoke();
    // What was already queued may still arrive; nothing new does, and the stream fails.
    let mut queued = 0;
    loop {
        match events.next().await {
            Some(Ok(_)) => queued += 1,
            Some(Err(_)) => break,
            None => panic!("a revoked stream ends as a failure, not cleanly"),
        }
        assert!(queued < 300, "the revoked observer still receives new events");
    }
    let mut received = 0;
    loop {
        match bytes.next().await {
            Some(Ok(chunk)) => received += chunk.len(),
            Some(Err(_)) => break,
            None => panic!("a revoked read ends as a failure, not a short success"),
        }
        assert!(received < 64 << 20, "the revoked read went on for {received} bytes");
    }
}

/// An output whose file is shorter than its revision fails its read, never a short success.
#[tokio::test]
async fn a_short_output_read_fails() {
    let (client, _) = serve(Run { live: false, held: 3 }).await;
    for range in [(0, 0), (5, 8)] {
        let read = read(&client, range).await;
        assert!(read.is_err(), "{range:?}: {read:?}");
    }
}

/// A range is one part of a parallel read: exactly its bytes, clamped at the output's end, and
/// a short output is short only where the range reaches; a range past the end is refused.
#[tokio::test]
async fn a_range_reads_its_bytes_alone() {
    let (client, _) = serve(Run { live: false, held: 8 }).await;
    assert_eq!(read(&client, (2, 6)).await.unwrap(), (128 << 20, b"byte".to_vec()));
    let end = 128 << 20;
    assert_eq!(read(&client, (end - 3, end + (1 << 20))).await.unwrap(), (end, vec![0; 3]));
    assert_eq!(read(&client, (end - 1, end)).await.unwrap().1.len(), 1);
    let target = v1::OutputTarget { run: "run".into(), output: "image".into(), ..Default::default() };
    let past = client.bulk().unwrap().get(&target, (end, end + 1), 0).await.err().unwrap();
    assert_eq!((past.status(), past.code()), (Some(416), Some("range_unsatisfiable")));
    let moved = client.bulk().unwrap().get(&target, (0, 1), 2).await.err().unwrap();
    assert_eq!((moved.status(), moved.code()), (Some(412), Some("output_revision_moved")));

    let (client, _) = serve(Run { live: false, held: 3 }).await;
    assert_eq!(read(&client, (0, 3)).await.unwrap(), (8, b"8 b".to_vec()));
    assert!(read(&client, (1, 5)).await.is_err());
}
