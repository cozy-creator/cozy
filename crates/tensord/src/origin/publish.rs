//! A tree put in a Hub repository's custody (the Hub's incremental publication): its closure
//! (the tree, its repo files, each cozytensors header and its chunks and assets) is declared
//! under one idempotent operation, only the objects the Hub lacks are PUT to their presigned
//! final keys and verified, and the publication is finalized. Only those need be here. The Hub's
//! finalization job is the commit's only authority, so the call follows it to its end. A tree
//! the repository already holds converges with no transfer. The Hub's wire names the tree its
//! `manifest`.
//!
//! The caller may stop it until it finalizes: PUTs in flight end within a tick and nothing is
//! finalized. The Hub keeps every object it verified, so asking again resumes from there.
use std::{
    collections::BTreeMap,
    future::Future,
    sync::{mpsc, Mutex},
    time::{Duration, Instant},
};

use hub_client::{types, ResponseValue};
use tensorfs::{
    err::{refuse, Code, Refusal, Result},
    format::ObjectRef,
    store::Store,
};

use crate::{
    hub::{block, Catalog},
    stop::Stop,
};

/// Renew stale authority at most twice per object; fresh permission refusals are terminal.
const GRANT_RENEWALS: usize = 2;
/// Asks of one Hub call, and sends of one object.
const ATTEMPTS: u32 = 4;
/// Statuses that mean "not now".
const RETRYABLE: [u16; 7] = [408, 425, 429, 500, 502, 503, 504];
/// How often a wait looks at the caller's stop.
const TICK: Duration = Duration::from_millis(200);
const FORBIDDEN: &str = "the object store answered HTTP 403 to the PUT";

pub struct Publication<'a> {
    pub store: &'a Store,
    /// `org/name`, the model repository that takes the checkpoint.
    pub destination: &'a str,
    pub tree: &'a ObjectRef,
    /// Names this publication: the same operation resumes it, a new one starts another.
    pub operation: &'a str,
    /// Bytes the Hub holds of the closure, and its total.
    pub progress: &'a (dyn Fn(u64, u64) + Sync),
    /// PUTs in flight.
    pub streams: usize,
    /// Stops the publication before it finalizes.
    pub stop: Option<&'a Stop>,
    /// Asked once every object is held, just before finalizing, which cannot be taken back:
    /// false stops the publication there, as `stop` does.
    pub commit: &'a (dyn Fn() -> bool + Sync),
}

/// What the Hub committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Published {
    /// `sha256:<hex>` of the tree, its identity in the repository.
    pub checkpoint: String,
    /// The repository already held it: nothing was pushed or finalized.
    pub converged: bool,
}

struct Grant {
    url: String,
    headers: Vec<(String, String)>,
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

enum Event {
    Granted(String, Option<Heartbeat>),
    Finished(String),
}

fn heartbeat(grant: &Grant) -> Option<Heartbeat> {
    grant.lifetime.filter(|life| !life.is_zero()).map(|life| {
        let period = (life / 2).max(Duration::from_millis(1));
        Heartbeat { period, next: grant.received + period }
    })
}

fn id(object: &ObjectRef) -> String {
    format!("sha256:{}", object.sha256)
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
        url: grant.url,
        headers,
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

fn is_stopped(stop: Option<&Stop>) -> bool {
    stop.is_some_and(Stop::is_cancelled)
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

/// One call of the publication, re-asked only while the Hub (or the link to it) says "not now":
/// no answer, an answer without the Hub's envelope (a tunnel or proxy in front of it), or a
/// status that means try again; at a doubling backoff, ended by the caller's stop.
fn ask<T: Send, F: Future<Output = Answer<T>> + Send>(
    what: &str,
    stop: Option<&Stop>,
    ask: impl Fn() -> F,
) -> Result<ResponseValue<T>> {
    let mut last = String::new();
    let mut wait = Duration::from_secs(1);
    for attempt in 1..=ATTEMPTS {
        if is_stopped(stop) {
            return stopped();
        }
        let error = match block(ask()) {
            Ok(answer) => return Ok(answer),
            Err(error) => hub_client::Error::from(error),
        };
        let weather = match &error {
            hub_client::Error::Call(_) => true,
            hub_client::Error::Hub { status, code, .. } => code.starts_with("http_") || RETRYABLE.contains(status),
            _ => false,
        };
        if !weather {
            return Err(hub_refused(format!("{what}: {error}")));
        }
        last = format!("{what}: {error}");
        if attempt < ATTEMPTS {
            pause(wait, stop)?;
            wait *= 2;
        }
    }
    refuse(Code::HUB_UNREACHABLE, format!("{last} — on all {ATTEMPTS} asks"))
}

/// Waits `wait`, ended early by the caller's stop.
fn pause(wait: Duration, stop: Option<&Stop>) -> Result<()> {
    let until = Instant::now() + wait;
    while Instant::now() < until {
        if is_stopped(stop) {
            return stopped();
        }
        std::thread::sleep(TICK.min(until.saturating_duration_since(Instant::now())));
    }
    Ok(())
}

/// One verified object PUT to its grant: the URL and exactly the headers its signature covers.
/// `If-None-Match: *` makes it idempotent (412 is an earlier send that landed). Re-sent on
/// weather; a dead peer is found by TCP keepalive, a live slow one is waited for.
fn put(http: &reqwest::Client, store: &Store, object: &ObjectRef, grant: &Grant, stop: Option<&Stop>) -> Result<()> {
    let runtime = tensorfs::fetch::runtime()?;
    let mut wait = Duration::from_secs(1);
    let mut last = String::new();
    for attempt in 1..=ATTEMPTS {
        if is_stopped(stop) {
            return stopped();
        }
        let file = store.file(object).map_err(|e| e.refusal())?;
        let mut headers = vec![
            ("content-type".to_string(), "application/octet-stream".to_string()),
            ("content-length".to_string(), object.length.to_string()),
            ("if-none-match".to_string(), "*".to_string()),
        ];
        for (name, value) in &grant.headers {
            match headers.iter_mut().find(|(held, _)| held == name) {
                Some(slot) => slot.1 = value.clone(),
                None => headers.push((name.clone(), value.clone())),
            }
        }
        let mut request = http.put(&grant.url).body(reqwest::Body::from(tokio::fs::File::from_std(file)));
        for (name, value) in &headers {
            request = request.header(name, value);
        }
        let sent = runtime.block_on(async {
            let stopped = async {
                let mut tick = tokio::time::interval(TICK);
                while !is_stopped(stop) {
                    tick.tick().await;
                }
            };
            tokio::select! {
                answer = request.send() => Some(answer),
                () = stopped => None,
            }
        });
        let answer = match sent {
            None => return stopped(),
            Some(Ok(answer)) => answer,
            Some(Err(error)) => {
                last = format!("PUT {}: {error}", object.sha256);
                if attempt < ATTEMPTS {
                    pause(wait, stop)?;
                    wait *= 2;
                }
                continue;
            }
        };
        let status = answer.status().as_u16();
        match status {
            200..=299 | 412 => return Ok(()),
            300..=399 => return refuse(Code::REDIRECT_REFUSED, format!("HTTP {status} tried to replace the exact granted URL")),
            403 => return refuse(Code::TRANSFER_FAILED, FORBIDDEN),
            _ if RETRYABLE.contains(&status) => {
                let stated = answer.headers().get("retry-after").and_then(|v| v.to_str().ok()?.trim().parse().ok());
                last = format!("the object store answered HTTP {status} to the PUT");
                if attempt < ATTEMPTS {
                    pause(stated.map_or(wait, Duration::from_secs), stop)?;
                    wait *= 2;
                }
            }
            _ => return refuse(Code::TRANSFER_FAILED, format!("the object store answered HTTP {status} to the PUT")),
        }
    }
    refuse(Code::TRANSFER_FAILED, format!("{last} — on all {ATTEMPTS} sends"))
}

impl Catalog {
    /// Put the tree `p.tree` in `p.destination`'s custody, under the run's capability.
    pub fn publish(&self, p: &Publication) -> Result<Published> {
        let hub = self.run().ok_or_else(|| hub_refused("the run's capability does not let it publish"))?;
        let http = crate::hub::fetch_http(self.ca()).map_err(|e| hub_refused(e.0))?;
        let (org, name) = repository(p.destination)?;
        let (org, name) = (org.as_str(), name.as_str());
        let checkpoint = id(p.tree);
        let operation = p.operation;

        // The closure the tree commits to: its files and each header's chunks and assets.
        let objects: BTreeMap<String, ObjectRef> = p
            .store
            .closure(p.tree)
            .map_err(|e| e.refusal())?
            .into_iter()
            .map(|o| (id(&o), o))
            .collect();
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
                .map(|o| types::DeclaredObject { object_id: id(o), length: o.length as i64 })
                .collect(),
        };
        let opened = ask("the publication's declaration", p.stop, || {
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
        // Only what the Hub lacks must be here: what it holds (large parts other checkpoints
        // share) is neither read nor sent.
        for id in &pending {
            if !p.store.reuse(&objects[id]).map_err(|e| e.refusal())? {
                return refuse(Code::OBJECT_ABSENT, format!("{id} of {checkpoint} is not here"));
            }
        }
        let landed = Mutex::new(total - pending.iter().map(|id| objects[id].length).sum::<u64>());
        (p.progress)(*landed.lock().unwrap(), total);

        // A worker asks for authority only for the object it can send now: finishing a small one
        // frees its slot at once, so a slow neighbour cannot hold a batch. The coordinator keeps
        // active holds alive independently of PUTs and verification.
        let slots = p.streams.clamp(1, 32).min(pending.len());
        let queue = Mutex::new(pending.into_iter());
        let failed: Mutex<Option<Refusal>> = Mutex::new(None);
        let (send, receive) = mpsc::sync_channel(slots.max(1) * 2);
        let select = |id: &str| types::PublicationObjectIds { object_ids: vec![id.to_string()] };
        let grants = |id: &str| -> Result<(types::PublicationGrants, Instant)> {
            let received = Instant::now();
            let selected = select(id);
            let answer = ask("the publication's grants", p.stop, || {
                hub.grant_publication_objects(org, name, operation, &selected)
            })?;
            Ok((answer.into_inner(), received))
        };
        std::thread::scope(|scope| {
            for _ in 0..slots {
                let (send, queue, failed, landed, objects, grants, http) =
                    (send.clone(), &queue, &failed, &landed, &objects, &grants, &http);
                scope.spawn(move || loop {
                    if is_stopped(p.stop) {
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
                        let _ = send.send(Event::Granted(id.clone(), heartbeat(&grant)));
                        let mut renewals = 0;
                        loop {
                            if failed.lock().unwrap().is_some() {
                                return Ok(false);
                            }
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
                                let _ = send.send(Event::Granted(id.clone(), heartbeat(&grant)));
                                continue;
                            }
                            match put(http, p.store, object, &grant, p.stop) {
                                Ok(()) => return Ok(true),
                                // Re-mint through the authority; fresh permission refusals stay terminal.
                                Err(refusal) if grant.expired() && renewals < GRANT_RENEWALS && refusal.detail == FORBIDDEN => continue,
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
                            let selected = select(&id);
                            ask("the publication's verification", p.stop, || {
                                hub.verify_publication_objects(org, name, operation, &selected)
                            })?;
                        }
                        Ok(())
                    });
                    if let Err(refusal) = pushed {
                        failed.lock().unwrap().get_or_insert(refusal);
                    }
                    let _ = send.send(Event::Finished(id));
                });
            }
            drop(send);
            let mut active = BTreeMap::<String, Heartbeat>::new();
            loop {
                let due = active.iter().min_by_key(|(_, beat)| beat.next).map(|(id, beat)| (id.clone(), *beat));
                if let Some((id, beat)) = &due {
                    if beat.next <= Instant::now() {
                        // Before more completions: a stream of small objects must not starve a
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
                    Ok(Event::Granted(id, Some(beat))) => {
                        active.insert(id, beat);
                    }
                    Ok(Event::Granted(id, None) | Event::Finished(id)) => {
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
        if is_stopped(p.stop) || !(p.commit)() {
            return stopped();
        }

        // Finalize, then follow the Hub's finalization job to its end: committed, it is no
        // longer the caller's to stop.
        let finalize = types::PublicationFinalize { manifest_id: checkpoint.clone(), manifest_length: p.tree.length as i64 };
        let mut view = ask("the publication's finalization", None, || {
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
                    view = ask("the publication's finalization", None, || {
                        hub.get_model_finalization(org, name, operation)
                    })?;
                }
                other => return Err(hub_refused(format!("finalization state {other:?}"))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_destination_is_an_org_repository() {
        assert_eq!(repository(" Acme/Model ").unwrap(), ("acme".into(), "model".into()));
        for bad in ["local/mine", "acme", "acme/a/b", "/x", "acme/m?"] {
            assert!(repository(bad).is_err(), "{bad}");
        }
    }
}
