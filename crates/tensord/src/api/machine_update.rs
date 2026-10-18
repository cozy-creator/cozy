//! `Run kind: update` (G/API.md, D2): the payload names the software cohort, `{"runtime": …,
//! "tensorfs": …}`; the machine's own binary moves only with the `cozy` release. A member is a
//! published version (fetched from PyPI, digest checked); when an input binds that field to an
//! object PUT to the machine, that wheel's file name; or `{"url": "https://…/<wheel>", "sha256": …}`,
//! a wheel the machine fetches itself, held to that digest. The update stages and verifies the
//! pair, waits for measured idleness, restarts the service on it in place and commits once the new
//! service proves readiness, or rolls back. Its log is the update's state history, so a client
//! that lost the stream across the restart follows its events again after its cursor and reads the
//! outcome.
use super::{auth::VerifiedActor, backend::MachineBackend, v1, MachineIdentity};
use crate::machine::update::{Choice, Request, Status as Update, Updates};
use cozy_proto::error::Error;
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tokio_stream::{wrappers::ReceiverStream, Stream};

/// Whether run `id` is an update this machine holds or reserved.
pub(super) fn owns(
    identity: &MachineIdentity,
    backend: &impl MachineBackend,
    actor: VerifiedActor,
    id: &str,
) -> Result<bool, Error> {
    if let Some(runs) = backend.runs() {
        let actor = crate::machine_api::actor_id(actor);
        // Another signer's machine-wide update never shadows this actor's own run.
        match runs.service.engine.get_public(&actor, id) {
            Ok(_) => return Ok(false),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(Error::internal("journal_unreadable", error.to_string())),
        }
        if runs
            .service
            .engine
            .with_journal(|journal| journal.update_run_reserved(&actor, id))
            .map_err(|error| Error::internal("journal_unreadable", error.to_string()))?
        {
            return Ok(true);
        }
    }
    Ok(identity.updates.as_ref().is_some_and(|u| u.update(id).is_some()))
}

#[derive(serde::Deserialize)]
struct Cohort {
    runtime: Option<Member>,
    tensorfs: Option<Member>,
    /// `"install"`: the build toolchain, alone (`cozy machine toolchain install`).
    toolchain: Option<String>,
    #[serde(flatten)]
    unknown: BTreeMap<String, serde_json::Value>,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum Member {
    Named(String),
    Fetched { url: String, sha256: String },
}

static SUBMITTING: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn updates(identity: &MachineIdentity) -> Result<Arc<Updates>, Error> {
    identity
        .updates
        .clone()
        .ok_or_else(|| Error::conflict("update_unsupported", "this machine does not update in place"))
}

/// Submits update `request` (whether it is new, and its state), or answers the one its id
/// names.
pub(super) async fn submit<B: MachineBackend>(
    identity: Arc<MachineIdentity>,
    backend: Arc<B>,
    actor: VerifiedActor,
    request: v1::RunRequest,
) -> Result<(bool, v1::RunState), Error> {
    let updates = updates(&identity)?;
    let id = request.id.clone();
    // SQLite serializes this ID's ownership with ordinary acceptance and cancellation.
    let runs = backend
        .runs()
        .ok_or_else(|| Error::not_found("capability_unavailable", "this machine reserves no update run ids"))?;
    let owner = crate::machine_api::actor_id(actor);
    let stopped = |_| Error::internal("machine_operation_stopped", "machine operation stopped");
    tokio::task::spawn_blocking(move || {
        runs.service.engine.with_journal(|journal| journal.reserve_update_run(&owner, &request.id))
    })
    .await
    .map_err(stopped)?
    .map_err(|error| match error.kind() {
        std::io::ErrorKind::AlreadyExists => Error::conflict("run_id_conflict", error.to_string()),
        _ => Error::internal("journal_unreadable", error.to_string()),
    })?;
    tokio::task::spawn_blocking(move || {
        // One submission at a time, its check and its staging together: a client resending
        // an update it lost attaches to the first, even while that one is still staging
        // and after its own client has gone.
        let _one = SUBMITTING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(update) = updates.update(&id) {
            return Ok((false, state(&id, &update)));
        }
        if !identity.readiness.proved() {
            return Err(Error::unavailable("machine_booting", "the machine is starting; update it once it is ready"));
        }
        stage(&updates, &*backend, actor, &id, request.spec)?;
        let update = updates.update(&id).ok_or_else(|| Error::internal("update_unrecorded", "the update was not recorded"))?;
        Ok((true, state(&id, &update)))
    })
    .await
    .map_err(stopped)?
}

/// Update `id`'s log after `after`, its state first, until it ends.
pub(super) async fn events(
    identity: Arc<MachineIdentity>,
    id: String,
    after: u64,
) -> Result<impl Stream<Item = Result<v1::RunEvent, Error>>, Error> {
    let updates = updates(&identity)?;
    let absent = || Error::not_found("run_not_found", "this machine has no such run");
    let first = updates.update(&id).ok_or_else(absent)?;
    let (sender, receiver) = tokio::sync::mpsc::channel(16);
    let _ = sender.try_send(Ok(event(0, 0, v1::Event::State(state(&id, &first)))));
    tokio::spawn(async move {
        let mut sent = after;
        loop {
            let Some(update) = updates.update(&id) else {
                let _ = sender.send(Err(absent())).await;
                return;
            };
            for (index, step) in update.history.iter().enumerate().skip(sent as usize) {
                if sender.send(Ok(event(index as u64 + 1, step.at_ms, step_event(&update, step)))).await.is_err() {
                    return;
                }
            }
            sent = sent.max(update.history.len() as u64);
            if update.terminal() {
                return;
            }
            if let Some(error) = observation_error(&update) {
                let _ = sender.send(Err(error)).await;
                return;
            }
            tokio::select! {
                _ = sender.closed() => return,
                _ = tokio::time::sleep(Duration::from_millis(250)) => {}
            }
        }
    });
    Ok(ReceiverStream::new(receiver))
}

fn stage(
    updates: &Arc<Updates>,
    backend: &impl MachineBackend,
    actor: VerifiedActor,
    id: &str,
    spec: v1::RunSpec,
) -> Result<(), Error> {
    let payload = spec.payload.as_ref().map_or("{}", |payload| payload.get());
    let cohort: Cohort = serde_json::from_str(payload).map_err(|e| {
        Error::validation("invalid_payload", format!("an update payload is a JSON object: {e}"))
    })?;
    for name in cohort.unknown.keys() {
        tracing::warn!("update {id}: ignoring unknown payload field {name:?}");
    }
    let choose = |member: &str, value: Option<Member>| -> Result<Option<Choice>, Error> {
        let value = match value {
            None => return Ok(None),
            Some(Member::Fetched { url, sha256 }) => {
                let distribution = if member == "runtime" { "cozy_runtime" } else { member };
                return Choice::fetched(distribution, &url, &sha256)
                    .map(Some)
                    .map_err(|why| super::http::refused("wheel_refused", why));
            }
            Some(Member::Named(value)) => value,
        };
        let Some(input) = spec.inputs.iter().find(|i| i.field == member) else {
            return Ok(Some(Choice {
                version: value,
                ..Default::default()
            }));
        };
        let sha256 = input
            .digest
            .strip_prefix("sha256:")
            .filter(|h| h.len() == 64)
            .ok_or_else(|| Error::validation("invalid_digest", "an input digest is sha256:<64 hex>"))?;
        let mut object = backend.object(actor, sha256).map_err(|e| {
            Error::conflict(
                "input_unwritten",
                format!("the {member} wheel {} is not held here; PUT it first: {}", input.digest, e.message),
            )
        })?;
        let (staged, _) = updates
            .stage(&value, &mut object)
            .map_err(|(code, message)| refusal(code, message))?;
        if staged != sha256 {
            return Err(Error::internal("custody_corrupt", "the staged wheel differs from its object"));
        }
        Ok(Some(Choice {
            file: value,
            sha256: staged,
            ..Default::default()
        }))
    };
    let toolchain = match cohort.toolchain.as_deref() {
        None => false,
        Some("install") => true,
        Some(other) => {
            return Err(Error::validation("toolchain_unknown", format!("the toolchain is installed, not {other:?}")));
        }
    };
    let request = Request {
        operation: id.into(),
        pin: None,
        runtime: choose("runtime", cohort.runtime)?,
        tensorfs: choose("tensorfs", cohort.tensorfs)?,
        toolchain,
    };
    updates
        .request(request, |code| std::process::exit(code))
        .map(|_| ())
        .map_err(|(code, message)| refusal(code, message))
}

/// Cancels update `id` before it activates; one activating refuses (`cancel_unavailable`).
pub(super) async fn cancel(identity: &MachineIdentity, id: &str) -> Result<v1::RunState, Error> {
    let updates = updates(identity)?;
    let operation = id.to_string();
    let canceled = tokio::task::spawn_blocking(move || updates.cancel(&operation))
        .await
        .map_err(|_| Error::internal("machine_operation_stopped", "machine operation stopped"))?;
    match canceled {
        Ok(update) => Ok(state(id, &update)),
        Err(error) => Err(match error.kind() {
            std::io::ErrorKind::NotFound => Error::not_found("run_not_found", error.to_string()),
            std::io::ErrorKind::ResourceBusy => Error::conflict("cancel_unavailable", error.to_string()),
            _ => Error::internal("update_unreadable", error.to_string()),
        }),
    }
}

/// An update's refusal, by the HTTP status `Updates` names it with.
fn refusal(code: u16, message: String) -> Error {
    match code {
        400 => Error::validation("invalid_update", message),
        409 => Error::conflict("update_refused", message),
        503 => Error::unavailable("update_unavailable", message),
        _ => Error::internal("update_failed", message),
    }
}

fn event(sequence: u64, at_ms: i64, event: v1::Event) -> v1::RunEvent {
    v1::RunEvent { sequence, at_ms, event }
}

fn state(id: &str, update: &Update) -> v1::RunState {
    v1::RunState {
        id: id.into(),
        number: 0,
        state: match update.state.as_str() {
            "waiting" => "queued",
            "succeeded" => "succeeded",
            "canceled" => "canceled",
            "rolled_back" | "failed" => "failed",
            _ => "running",
        }
        .into(),
        sequence: update.history.len() as u64,
        attempt: 1,
        waiting: if observation_error(update).is_some() {
            update.error.clone()
        } else if update.state == "waiting_activation" {
            "waiting for the machine to be idle".into()
        } else {
            String::new()
        },
    }
}

/// A persistence blockage is an observation error, not a terminal run outcome. Unlike
/// Unavailable, FailedPrecondition lets an update observer stop retrying and report it.
fn observation_error(update: &Update) -> Option<Error> {
    (!update.terminal() && update.error.starts_with("update_commit_blocked:"))
        .then(|| Error::conflict("update_commit_blocked", &update.error))
}

fn step_event(update: &Update, step: &crate::machine::update::Step) -> v1::Event {
    let outcome = |status: &str, code: &str| {
        let result = serde_json::json!({"from": update.from, "to": update.to});
        v1::Event::Outcome(v1::Outcome {
            status: status.into(),
            reason: (!code.is_empty()).then(|| v1::Reason {
                code: code.into(),
                message: update.error.clone(),
                origin: "machine".into(),
                ..Default::default()
            }),
            result: v1::Json::parse(result.to_string()).ok(),
            ..Default::default()
        })
    };
    match step.state.as_str() {
        "succeeded" => outcome("succeeded", ""),
        "canceled" => outcome("canceled", "canceled"),
        "rolled_back" => outcome("failed", "update_rolled_back"),
        "failed" if update.code.is_empty() => outcome("failed", "update_failed"),
        "failed" => outcome("failed", &update.code),
        stage => v1::Event::Progress(v1::Progress {
            stage: stage.into(),
            fraction: -1.0,
            ..Default::default()
        }),
    }
}


#[cfg(test)]
mod commit_observation_tests {
    use super::*;

    #[test]
    fn a_blocked_commit_stops_observation_without_reporting_a_failed_run() {
        let mut update = Update {
            operation: "blocked".into(), state: "starting".into(),
            error: "update_commit_blocked: no space left".into(), ..Default::default()
        };
        let error = observation_error(&update).unwrap();
        assert_eq!(error.code, "update_commit_blocked");
        assert_eq!(state("blocked", &update).state, "running");
        assert_eq!(state("blocked", &update).waiting, update.error);
        update.state = "succeeded".into();
        assert!(observation_error(&update).is_none(), "cleanup cannot undo committed success");
    }
}
