//! A checkpoint put in a Hub repository's custody (the Hub's incremental publication): its
//! closure is declared under one idempotent operation, only the objects the Hub lacks are
//! pushed under its signed grants (TensorFS's mover carries the bytes) and verified, and the
//! publication is finalized. The Hub's finalization job is the commit's only authority, so the
//! call follows it to its end. A checkpoint the repository already holds converges with no
//! transfer.
//!
//! The caller may stop it until it finalizes: pushes in flight end within a sampling tick and
//! nothing is finalized, so no checkpoint appears. The Hub keeps every object it verified (the
//! publication's holds last until it idles out), so asking again resumes from there.
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};

use hub_client::{types, ResponseValue};
use tensorfs_core::checkpoint::walk_cozytensors;
use tensorfs_core::err::{refuse, Code, Refusal, Result};
use tensorfs_core::ids::ObjectRef;
use tensorfs_core::store::Store;
use tensorfs_core::transport::{
    push_object, Deadline, Ledger, PullCancellation, UploadGrant, FETCH_ATTEMPTS,
    RETRYABLE_STATUS,
};

use super::{block, Catalog};

/// Renew stale authority at most twice per object; fresh permission refusals are terminal.
const GRANT_RENEWALS: usize = 2;
/// How often a wait looks at the caller's stop.
const TICK: Duration = Duration::from_millis(200);

pub struct Publication<'a> {
    pub store: &'a Store,
    /// `org/name`, the model repository that takes the checkpoint.
    pub destination: &'a str,
    pub manifest: &'a ObjectRef,
    /// Names this publication: the same operation resumes it, a new one starts another.
    pub operation: &'a str,
    /// Bytes the Hub holds of the closure, and its total.
    pub progress: &'a (dyn Fn(u64, u64) + Sync),
    /// Pushes in flight.
    pub streams: usize,
    /// Stops the publication before it finalizes (the module note).
    pub stop: Option<&'a PullCancellation>,
    /// Asked once every object is held, just before finalizing, which cannot be taken back:
    /// false stops the publication there, as `stop` does.
    pub commit: &'a (dyn Fn() -> bool + Sync),
}

/// What the Hub committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Published {
    /// `sha256:<hex>` of the manifest, the checkpoint's identity in the repository.
    pub checkpoint: String,
    /// The repository already held it: nothing was pushed or finalized.
    pub converged: bool,
}

struct Grant {
    upload: UploadGrant,
    received: Instant,
    lifetime: Option<Duration>,
}

impl Grant {
    fn stale(&self) -> bool {
        self.lifetime.is_some_and(|life| self.received.elapsed() >= life / 2)
    }

    fn expired(&self) -> bool {
        self.lifetime.is_some_and(|life| self.received.elapsed() >= life)
    }
}

#[derive(Clone, Copy)]
struct Heartbeat {
    period: Duration,
    next: Instant,
}

enum UploadEvent {
    Granted(String, Option<Heartbeat>),
    Finished(String),
}

fn heartbeat(grant: &Grant) -> Option<Heartbeat> {
    grant.lifetime.filter(|life| !life.is_zero()).map(|life| {
        let period = (life / 2).max(Duration::from_millis(1));
        Heartbeat { period, next: grant.received + period }
    })
}

/// The Hub's answer to one selection: whether it already holds `id`, else its grant.
fn granted(answer: types::PublicationGrants, id: &str, received: Instant) -> Result<Option<Grant>> {
    let held = answer.held.unwrap_or_default().iter().any(|row| {
        row.object_id == id && matches!(row.state.as_str(), "accepted" | "verifying")
    });
    if held {
        return Ok(None);
    }
    let server_time = answer.server_time_unix;
    let grant = answer
        .grants
        .unwrap_or_default()
        .into_iter()
        .find(|grant| grant.object_id == id)
        .ok_or_else(|| hub_refused("the Hub omitted a selected object's grant"))?;
    let mut headers: Vec<(String, String)> =
        grant.required_headers.into_iter().map(|(k, v)| (k.to_lowercase(), v)).collect();
    headers.sort();
    Ok(Some(Grant {
        upload: UploadGrant { url: grant.url, headers },
        received,
        // Never compare the signer's time to our wall clock.
        lifetime: (server_time > 0 && grant.expires_at_unix > 0)
            .then(|| Duration::from_secs(grant.expires_at_unix.saturating_sub(server_time) as u64)),
    }))
}

fn hub_refused(detail: impl Into<String>) -> Refusal {
    Refusal { code: Code::HUB_REFUSED, detail: detail.into() }
}

fn stopped<T>() -> Result<T> {
    refuse(Code::TRANSFER_FAILED, "the publication was stopped before it finalized; the Hub keeps what it verified")
}

/// The destination's repository, as the Hub folds names.
fn repository(destination: &str) -> Result<(String, String)> {
    let lowered = destination.trim().to_lowercase();
    match lowered.split_once('/') {
        Some((org, name))
            if !org.is_empty()
                && !name.is_empty()
                && org != "local"
                && !name.contains('/')
                && lowered.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_./".contains(&b)) =>
        {
            Ok((org.to_string(), name.to_string()))
        }
        _ => refuse(Code::SOURCE_NOT_ALLOWED, format!("{destination:?} is not an org/name model repository")),
    }
}

type Answer<T> = std::result::Result<ResponseValue<T>, hub_client::CallError>;

impl Catalog {
    /// One call of the publication, re-asked only while the Hub (or the link to it) says "not
    /// now": no answer, an answer without the Hub's envelope (a tunnel or proxy in front of it),
    /// or a status that means try again. Waits are the ledger's patience, a multiple of the gap
    /// this publication has actually shown, and end at the caller's stop.
    fn ask<T: Send, F: Future<Output = Answer<T>> + Send>(
        &self,
        what: &str,
        stop: Option<&PullCancellation>,
        ledger: &Ledger,
        ask: impl Fn() -> F,
    ) -> Result<ResponseValue<T>> {
        let mut last = String::new();
        for attempt in 1..=FETCH_ATTEMPTS {
            if stop.is_some_and(PullCancellation::is_cancelled) {
                return stopped();
            }
            let error = match block(ask()) {
                Ok(answer) => return Ok(answer),
                Err(error) => hub_client::Error::from(error),
            };
            let weather = match &error {
                hub_client::Error::Call(_) => true,
                hub_client::Error::Hub { status, code, .. } => {
                    code.starts_with("http_") || RETRYABLE_STATUS.contains(status)
                }
                _ => false,
            };
            if !weather {
                return Err(hub_refused(format!("{what}: {error}")));
            }
            last = format!("{what}: {error}");
            if attempt < FETCH_ATTEMPTS {
                pause(ledger.patience(), stop)?;
            }
        }
        refuse(Code::HUB_UNREACHABLE, format!("{last} — on all {FETCH_ATTEMPTS} asks"))
    }

    /// Put `p.manifest` in `p.destination`'s custody, under the run's capability.
    pub fn publish(&self, p: &Publication) -> Result<Published> {
        let hub = self.run.as_ref().ok_or_else(|| hub_refused("the run's capability does not let it publish"))?;
        let ledger = match p.stop {
            Some(stop) => Ledger::new().with_cancellation(stop.clone()),
            None => Ledger::new(),
        };
        let (org, name) = repository(p.destination)?;
        let (org, name) = (org.as_str(), name.as_str());
        let checkpoint = p.manifest.id();
        let operation = p.operation;

        // The closure the checkpoint commits to: its manifest and everything its header reaches.
        let manifest = p.store.read_manifest(p.manifest)?;
        let mut objects: BTreeMap<String, ObjectRef> = BTreeMap::new();
        objects.insert(p.manifest.id(), p.manifest.clone());
        for reached in walk_cozytensors(p.store, &manifest)?.objects {
            if objects.get(&reached.obj.id()).is_some_and(|held| held.length != reached.obj.length) {
                return refuse(Code::LENGTH_MISMATCH, format!("{} is named with two lengths", reached.obj.id()));
            }
            objects.insert(reached.obj.id(), reached.obj);
        }
        let total: u64 = objects.values().map(|o| o.length).sum();

        // A repository that already holds this checkpoint converges with no transfer.
        if block(hub.get_model_checkpoint(org, name, &checkpoint)).is_ok() {
            (p.progress)(total, total);
            return Ok(Published { checkpoint, converged: true });
        }

        // Declare the closure; the Hub answers which objects it already holds.
        let declared = types::PublicationDeclaration {
            objects: objects
                .values()
                .map(|o| types::DeclaredObject { object_id: o.id(), length: o.length as i64 })
                .collect(),
        };
        let opened = self.ask("the publication's declaration", p.stop, &ledger, || {
            hub.open_model_publication(org, name, operation, &declared)
        })?;
        let mut pending = vec![];
        for row in &opened.publication.objects {
            if !objects.contains_key(&row.object_id) {
                return Err(hub_refused(format!("the publication names {}, not of this closure", row.object_id)));
            }
            if !matches!(row.state.as_str(), "accepted" | "verifying") {
                pending.push(row.object_id.clone());
            }
        }
        let landed = Mutex::new(total - pending.iter().map(|id| objects[id].length).sum::<u64>());
        (p.progress)(*landed.lock().unwrap(), total);

        // A worker asks for authority only for the object it can dispatch now: finishing a small
        // tensor frees its slot at once, so a slow neighbour cannot hold a batch. The coordinator
        // keeps active holds alive independently of PUTs and verification; its event queue and
        // live grant set are bounded by the worker count.
        let slots = p.streams.clamp(1, 32).min(pending.len());
        let queue = Mutex::new(pending.into_iter());
        let failed: Mutex<Option<Refusal>> = Mutex::new(None);
        let (send, receive) = mpsc::sync_channel(slots.max(1) * 2);
        let select = |id: &str| types::PublicationObjectIds { object_ids: vec![id.to_string()] };
        let grants = |id: &str| -> Result<(types::PublicationGrants, Instant)> {
            let received = Instant::now();
            let selected = select(id);
            let answer = self.ask("the publication's grants", p.stop, &ledger, || {
                hub.grant_publication_objects(org, name, operation, &selected)
            })?;
            Ok((answer.into_inner(), received))
        };
        std::thread::scope(|scope| {
            for _ in 0..slots {
                let (send, queue, failed, landed, objects, grants, ledger) =
                    (send.clone(), &queue, &failed, &landed, &objects, &grants, &ledger);
                scope.spawn(move || loop {
                    if p.stop.is_some_and(PullCancellation::is_cancelled) {
                        failed.lock().unwrap().get_or_insert_with(|| stopped::<()>().unwrap_err());
                    }
                    if failed.lock().unwrap().is_some() {
                        break;
                    }
                    let Some(id) = queue.lock().unwrap().next() else {
                        break;
                    };
                    let object = &objects[&id];
                    let pushed = (|| {
                        let (answer, received) = grants(&id)?;
                        let Some(mut grant) = granted(answer, &id, received)? else {
                            return Ok(true);
                        };
                        let _ = send.send(UploadEvent::Granted(id.clone(), heartbeat(&grant)));
                        let mut renewals = 0;
                        loop {
                            if failed.lock().unwrap().is_some() {
                                return Ok(false);
                            }
                            // This also rechecks authority after bounded event-queue backpressure.
                            if grant.stale() {
                                if renewals == GRANT_RENEWALS {
                                    return Err(hub_refused("the Hub repeatedly issued stale upload grants"));
                                }
                                renewals += 1;
                                let (answer, received) = grants(&id)?;
                                let Some(renewed) = granted(answer, &id, received)? else {
                                    return Ok(true);
                                };
                                grant = renewed;
                                let _ = send.send(UploadEvent::Granted(id.clone(), heartbeat(&grant)));
                                continue;
                            }
                            match push_object(
                                p.store,
                                &object.sha256,
                                object.sha256 == p.manifest.sha256,
                                &grant.upload,
                                self.policy(),
                                self.credential(),
                                Deadline::none(),
                                ledger,
                                "",
                            ) {
                                Ok(_) => return Ok(true),
                                Err(refusal)
                                    if grant.expired()
                                        && renewals < GRANT_RENEWALS
                                        && refusal.code == Code::TRANSFER_FAILED
                                        && refusal.detail == "the object store answered HTTP 403 to the PUT" =>
                                {
                                    // Re-mint through the authority; fresh permission refusals stay terminal.
                                    continue;
                                }
                                Err(refusal) => return Err(refusal),
                            }
                        }
                    })()
                    .and_then(|pushed| {
                        if pushed {
                            {
                                let mut landed = landed.lock().unwrap();
                                *landed += object.length;
                                (p.progress)(*landed, total);
                            }
                            // Verification is bounded by the slots too, without stopping the others.
                            let selected = select(&id);
                            self.ask("the publication's verification", p.stop, ledger, || {
                                hub.verify_publication_objects(org, name, operation, &selected)
                            })?;
                        }
                        Ok(())
                    });
                    if let Err(refusal) = pushed {
                        failed.lock().unwrap().get_or_insert(refusal);
                    }
                    let _ = send.send(UploadEvent::Finished(id));
                });
            }
            drop(send);
            let mut active = BTreeMap::<String, Heartbeat>::new();
            loop {
                let due = active.iter().min_by_key(|(_, beat)| beat.next).map(|(id, beat)| (id.clone(), *beat));
                if let Some((id, beat)) = &due {
                    if beat.next <= Instant::now() {
                        // Before more completions: a stream of small tensors must not starve a
                        // long PUT's heartbeat.
                        if let Err(refusal) = grants(id) {
                            failed.lock().unwrap().get_or_insert(refusal);
                        }
                        active.insert(id.clone(), Heartbeat { next: Instant::now() + beat.period, ..*beat });
                        continue;
                    }
                }
                let event = match &due {
                    Some((_, beat)) => receive.recv_timeout(beat.next.saturating_duration_since(Instant::now())),
                    None => receive.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
                };
                match event {
                    Ok(UploadEvent::Granted(id, Some(beat))) => {
                        active.insert(id, beat);
                    }
                    Ok(UploadEvent::Granted(id, None) | UploadEvent::Finished(id)) => {
                        active.remove(&id);
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                }
            }
        });
        if let Some(refusal) = failed.into_inner().unwrap() {
            return Err(refusal);
        }
        if p.stop.is_some_and(PullCancellation::is_cancelled) || !(p.commit)() {
            return stopped();
        }

        // Finalize, then follow the Hub's finalization job to its end: committed, it is no
        // longer the caller's to stop.
        let ledger = Ledger::new();
        let finalize = types::PublicationFinalize { manifest_id: checkpoint.clone(), manifest_length: p.manifest.length as i64 };
        let mut view = self.ask("the publication's finalization", None, &ledger, || {
            hub.finalize_model_publication(org, name, operation, &finalize)
        })?;
        loop {
            match view.state.as_str() {
                "completed" => return Ok(Published { checkpoint, converged: false }),
                "failed" => {
                    let code = view.error.as_ref().and_then(|e| e["code"].as_str()).unwrap_or("no reason");
                    return Err(hub_refused(format!("the Hub's finalization failed: {code}")));
                }
                "queued" | "running" => {
                    let stated = view
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.trim().parse::<u64>().ok());
                    pause(Duration::from_secs(stated.unwrap_or(2).max(2)), None)?;
                    view = self.ask("the publication's finalization", None, &ledger, || {
                        hub.get_model_finalization(org, name, operation)
                    })?;
                }
                other => return Err(hub_refused(format!("finalization state {other:?}"))),
            }
        }
    }
}

/// Waits `wait`, ended early by the caller's stop.
fn pause(wait: Duration, stop: Option<&PullCancellation>) -> Result<()> {
    let until = Instant::now() + wait;
    while Instant::now() < until {
        if stop.is_some_and(PullCancellation::is_cancelled) {
            return stopped();
        }
        std::thread::sleep(TICK.min(until.saturating_duration_since(Instant::now())));
    }
    Ok(())
}
