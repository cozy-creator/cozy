//! A checkpoint put in a Hub repository's custody (the Hub's incremental publication): its
//! closure is declared under one idempotent operation, only the objects the Hub lacks are
//! pushed under its signed grants and verified, and the publication is finalized. The Hub's
//! finalization job is the commit's only authority, so the call follows it to its end. A
//! checkpoint the repository already holds converges with no transfer.

use std::collections::BTreeMap;
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};

use crate::canon::{self, Value};
use crate::checkpoint::walk_cozytensors;
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::ObjectRef;
use crate::store::Store;
use crate::transport::credential::CredentialProvider;
use crate::transport::hub::call_method;
use crate::transport::ledger::{Deadline, Ledger};
use crate::transport::policy::SourcePolicy;
use crate::transport::push::{push_object, UploadGrant};

/// Renew stale authority at most twice per object; fresh permission refusals are terminal.
const GRANT_RENEWALS: usize = 2;

struct Grant {
    upload: UploadGrant,
    received: Instant,
    lifetime: Option<Duration>,
}

impl Grant {
    fn stale(&self) -> bool {
        self.lifetime
            .is_some_and(|life| self.received.elapsed() >= life / 2)
    }

    fn expired(&self) -> bool {
        self.lifetime
            .is_some_and(|life| self.received.elapsed() >= life)
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
        Heartbeat {
            period,
            next: grant.received + period,
        }
    })
}

fn grants(answer: &Value, received: Instant) -> Result<BTreeMap<String, Grant>> {
    let server_time = number(answer, "server_time_unix");
    rows(answer, "grants")
        .iter()
        .map(|grant| {
            let headers = match field(grant, "required_headers") {
                Some(Value::Obj(pairs)) => pairs
                    .iter()
                    .map(|(k, v)| match v {
                        Value::Str(v) => Ok((k.to_lowercase(), v.clone())),
                        _ => Err(hub_refused("a grant header is not text")),
                    })
                    .collect::<Result<Vec<_>>>()?,
                _ => vec![],
            };
            let expires = number(grant, "expires_at_unix");
            Ok((
                text(grant, "object_id").to_string(),
                Grant {
                    upload: UploadGrant {
                        url: text(grant, "url").to_string(),
                        headers,
                    },
                    received,
                    // Older Hubs may omit clocks. Never compare the signer's time to our wall clock.
                    lifetime: (server_time > 0 && expires > 0)
                        .then(|| Duration::from_secs(expires.saturating_sub(server_time))),
                },
            ))
        })
        .collect()
}

fn is_held(answer: &Value, id: &str) -> bool {
    rows(answer, "held").iter().any(|row| {
        text(row, "object_id") == id && matches!(text(row, "state"), "accepted" | "verifying")
    })
}

pub struct Publication<'a> {
    pub store: &'a Store,
    /// The Hub's base URL.
    pub hub: &'a str,
    /// `org/name`, the model repository that takes the checkpoint.
    pub destination: &'a str,
    pub manifest: &'a ObjectRef,
    /// Names this publication: the same operation resumes it, a new one starts another.
    pub operation: &'a str,
    pub credential: &'a dyn CredentialProvider,
    /// The Hub and its object-storage hosts.
    pub policy: &'a SourcePolicy,
    /// Bytes the Hub holds of the closure, and its total.
    pub progress: &'a (dyn Fn(u64, u64) + Sync),
    /// Pushes in flight.
    pub streams: usize,
}

/// What the Hub committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Published {
    /// `sha256:<hex>` of the manifest, the checkpoint's identity in the repository.
    pub checkpoint: String,
    /// The repository already held it: nothing was pushed or finalized.
    pub converged: bool,
}

fn field<'v>(value: &'v Value, name: &str) -> Option<&'v Value> {
    match value {
        Value::Obj(pairs) => pairs.iter().find(|(k, _)| k == name).map(|(_, v)| v),
        _ => None,
    }
}

fn text<'v>(value: &'v Value, name: &str) -> &'v str {
    match field(value, name) {
        Some(Value::Str(s)) => s,
        _ => "",
    }
}

fn rows<'v>(value: &'v Value, name: &str) -> &'v [Value] {
    match field(value, name) {
        Some(Value::Arr(rows)) => rows,
        _ => &[],
    }
}

fn number(value: &Value, name: &str) -> u64 {
    match field(value, name) {
        Some(Value::Int(n)) if *n >= 0 => *n as u64,
        _ => 0,
    }
}

fn hub_refused(detail: impl Into<String>) -> Refusal {
    Refusal {
        code: Code::HUB_REFUSED,
        detail: detail.into(),
    }
}

/// The destination's repository path, as the Hub folds names.
fn repository(destination: &str) -> Result<String> {
    let lowered = destination.trim().to_lowercase();
    match lowered.split_once('/') {
        Some((org, name))
            if !org.is_empty()
                && !name.is_empty()
                && org != "local"
                && !name.contains('/')
                && lowered
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_./".contains(&b)) =>
        {
            Ok(format!("/v1/models/{org}/{name}"))
        }
        _ => refuse(
            Code::SOURCE_NOT_ALLOWED,
            format!("{destination:?} is not an org/name model repository"),
        ),
    }
}

fn body(value: Value) -> Vec<u8> {
    canon::write(&value)
}

fn ids(objects: &[String]) -> Vec<u8> {
    body(Value::obj(vec![(
        "object_ids",
        Value::arr(objects.iter().map(Value::str).collect()),
    )]))
}

/// Put `p.manifest` in `p.destination`'s custody at `p.hub`.
pub fn publish(p: &Publication) -> Result<Published> {
    let client = crate::transport::http::Client::new();
    let ledger = Ledger::new();
    let deadline = Deadline::none();
    let repository = repository(p.destination)?;
    let checkpoint = p.manifest.id();
    let call = |method: &str, path: &str, payload: &[u8]| {
        call_method(
            &client,
            method,
            p.hub,
            path,
            payload,
            p.credential,
            p.policy,
            deadline,
            &ledger,
        )
    };

    // The closure the checkpoint commits to: its manifest and everything its header reaches.
    let manifest = p.store.read_manifest(p.manifest)?;
    let mut objects: BTreeMap<String, ObjectRef> = BTreeMap::new();
    objects.insert(p.manifest.id(), p.manifest.clone());
    for reached in walk_cozytensors(p.store, &manifest)?.objects {
        if let Some(held) = objects.get(&reached.obj.id()) {
            if held.length != reached.obj.length {
                return refuse(
                    Code::LENGTH_MISMATCH,
                    format!("{} is named with two lengths", reached.obj.id()),
                );
            }
        }
        objects.insert(reached.obj.id(), reached.obj);
    }
    let total: u64 = objects.values().map(|o| o.length).sum();

    // A repository that already holds this checkpoint converges with no transfer.
    let held = call(
        "GET",
        &format!("{repository}/checkpoints/{checkpoint}"),
        &[],
    );
    if held.is_ok() {
        (p.progress)(total, total);
        return Ok(Published {
            checkpoint,
            converged: true,
        });
    }

    // Declare the closure; the Hub answers which objects it already holds.
    let path = format!("{repository}/publications/{}", p.operation);
    let declared = body(Value::obj(vec![(
        "objects",
        Value::arr(
            objects
                .values()
                .map(|o| {
                    Value::obj(vec![
                        ("object_id", Value::str(o.id())),
                        ("length", Value::uint(o.length)),
                    ])
                })
                .collect(),
        ),
    )]));
    let opened = call("PUT", &path, &declared)?;
    let publication = field(&opened, "publication").ok_or_else(|| hub_refused("no publication"))?;
    let mut pending = vec![];
    for row in rows(publication, "objects") {
        let id = text(row, "object_id");
        if !objects.contains_key(id) {
            return Err(hub_refused(format!(
                "the publication names {id}, not of this closure"
            )));
        }
        if !matches!(text(row, "state"), "accepted" | "verifying") {
            pending.push(id.to_string());
        }
    }
    let landed = Mutex::new(total - pending.iter().map(|id| objects[id].length).sum::<u64>());
    (p.progress)(*landed.lock().unwrap(), total);

    // A worker requests authority only for the object it can dispatch now. Finishing
    // a small tensor immediately frees that slot: a slow neighbour cannot hold a batch.
    // The coordinator keeps active publication holds alive independently of PUTs and
    // verification. Its event queue and live grant set are bounded by the worker count.
    let slots = p.streams.clamp(1, 32).min(pending.len());
    let queue = Mutex::new(pending.into_iter());
    let failed: Mutex<Option<Refusal>> = Mutex::new(None);
    let (send, receive) = mpsc::sync_channel(slots.max(1) * 2);
    std::thread::scope(|scope| {
        for _ in 0..slots {
            let (send, queue, failed, call, path, objects, landed, ledger) = (
                send.clone(),
                &queue,
                &failed,
                &call,
                &path,
                &objects,
                &landed,
                &ledger,
            );
            scope.spawn(move || loop {
                if failed.lock().unwrap().is_some() {
                    break;
                }
                let Some(id) = queue.lock().unwrap().next() else {
                    break;
                };
                let object = &objects[&id];
                let selected = ids(std::slice::from_ref(&id));
                let pushed = (|| {
                    let received = Instant::now();
                    let answer = call("POST", &format!("{path}/grants"), &selected)?;
                    if is_held(&answer, &id) {
                        return Ok(true);
                    }
                    let mut grant = grants(&answer, received)?
                        .remove(&id)
                        .ok_or_else(|| hub_refused("the Hub omitted a selected object's grant"))?;
                    let _ = send.send(UploadEvent::Granted(id.clone(), heartbeat(&grant)));
                    let mut renewals = 0;
                    loop {
                        if failed.lock().unwrap().is_some() {
                            return Ok(false);
                        }
                        // This also rechecks authority after bounded event-queue backpressure.
                        if grant.stale() {
                            if renewals == GRANT_RENEWALS {
                                return Err(hub_refused(
                                    "the Hub repeatedly issued stale upload grants",
                                ));
                            }
                            renewals += 1;
                            let received = Instant::now();
                            let answer = call("POST", &format!("{path}/grants"), &selected)?;
                            if is_held(&answer, &id) {
                                return Ok(true);
                            }
                            grant = grants(&answer, received)?.remove(&id).ok_or_else(|| {
                                hub_refused("renewal omitted the selected object")
                            })?;
                            let _ = send.send(UploadEvent::Granted(id.clone(), heartbeat(&grant)));
                            continue;
                        }
                        match push_object(
                            p.store,
                            &object.sha256,
                            object.sha256 == p.manifest.sha256,
                            &grant.upload,
                            p.policy,
                            p.credential,
                            deadline,
                            ledger,
                            "",
                        ) {
                            Ok(_) => return Ok(true),
                            Err(refusal)
                                if grant.expired()
                                    && renewals < GRANT_RENEWALS
                                    && refusal.code == Code::TRANSFER_FAILED
                                    && refusal.detail
                                        == "the object store answered HTTP 403 to the PUT" =>
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
                        // Bound verification concurrency too, without stopping unrelated slots.
                        call("POST", &format!("{path}/verify"), &selected)?;
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
            let due = active
                .iter()
                .min_by_key(|(_, beat)| beat.next)
                .map(|(id, beat)| (id.clone(), *beat));
            if let Some((id, beat)) = &due {
                if beat.next <= Instant::now() {
                    // Check the absolute deadline before accepting more completions: a
                    // stream of small tensors must not starve a long PUT's heartbeat.
                    if let Err(refusal) = call(
                        "POST",
                        &format!("{path}/grants"),
                        &ids(std::slice::from_ref(id)),
                    ) {
                        failed.lock().unwrap().get_or_insert(refusal);
                    }
                    active.insert(
                        id.clone(),
                        Heartbeat {
                            next: Instant::now() + beat.period,
                            ..*beat
                        },
                    );
                    continue;
                }
            }
            let event = match &due {
                Some((_, beat)) => {
                    receive.recv_timeout(beat.next.saturating_duration_since(Instant::now()))
                }
                None => receive
                    .recv()
                    .map_err(|_| mpsc::RecvTimeoutError::Disconnected),
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

    // Finalize, then follow the Hub's finalization job to its end.
    let finalize = body(Value::obj(vec![
        ("manifest_id", Value::str(checkpoint.clone())),
        ("manifest_length", Value::uint(p.manifest.length)),
    ]));
    let mut view = call("POST", &format!("{path}/finalize"), &finalize)?;
    loop {
        match text(&view, "state") {
            "completed" => {
                return Ok(Published {
                    checkpoint,
                    converged: false,
                })
            }
            "failed" => {
                let error = field(&view, "error");
                let code = error.map(|e| text(e, "code")).unwrap_or_default();
                return Err(hub_refused(format!(
                    "the Hub's finalization failed: {}",
                    if code.is_empty() { "no reason" } else { code }
                )));
            }
            "queued" | "running" => {
                ledger.pause(
                    std::time::Duration::from_secs(number(&view, "retry_after").max(2)),
                    deadline,
                )?;
                view = call("GET", &format!("{path}/finalization"), &[])?;
            }
            other => return Err(hub_refused(format!("finalization state {other:?}"))),
        }
    }
}
