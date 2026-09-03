//! The pull: closure → plan → N-way walk → completion proof (tfs-049, `pull.py` ported
//! with its evidence).
//!
//! **The wanted list is walked by N streams, not one.** An artifact is already cut on the
//! object grid, so the objects are independent and the serial loop was leaving the whole
//! link on the table — measured, not supposed: at 92.42 GiB on a rented pod the serial
//! loop moved 49.8 MB/s against a 141 MB/s transport ceiling on the same pods and the same
//! presigned GETs, and a 16-way walk through this same door landed 117.6 MB/s. Concurrency
//! changes the ORDER objects are asked for and nothing else: the plan is still
//! `FetchPlan::of`'s, every byte still enters at `DeliveryGrant::admit` → `put_stream`,
//! and the proof is still `complete`.
//!
//! **Resume is PER OBJECT**, which is what the plan makes free: a dropped connection loses
//! one object's partial temp file and nothing else, and the next invocation re-plans
//! against the store and asks only for what is still missing. There is no byte-range
//! resume inside an object — the artifact is already cut on the object grid, so the
//! recovery unit is an object.
//!
//! **Which failures deserve a re-ask.** A stall the ledger NAMED, a connection that died,
//! and the object store's own retryable answers do; an expired presign and an absent key
//! do not — those are facts about authorization, and asking again about authorization is
//! a lie about time. A refusal at the admission door propagates untouched: the bytes are
//! wrong at the only authority there is.

use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{DeliveryGrant, FetchPlan};
use crate::ids::ObjectRef;
use crate::store::Store;
use crate::transport::http::{self, Client};
use crate::transport::hub::{self, CredentialProvider};
use crate::transport::ledger::{Deadline, Ledger, FETCH_ATTEMPTS, STALL_FACTOR, STREAMS};
use crate::transport::policy::{self, SourcePolicy};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// The object-store answers a re-ask is worth attempting for. Everything else is a fact
/// about the request or the grant, and retrying it is a lie about time.
pub const RETRYABLE_STATUS: [u16; 7] = [408, 425, 429, 500, 502, 503, 504];

/// A caller's per-object observer: the object that landed and the bytes this call moved.
pub type ObjectObserver<'a> = &'a (dyn Fn(&ObjectRef, u64) + Sync);

fn shared_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(Client::new)
}

/// What one fetch moved. `transferred` counts the bytes this CALL pulled off the wire; a
/// held object never reaches here — `DeliveryGrant::mint` refuses first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    pub admitted: crate::fetch::Admitted,
    pub transferred: u64,
}

struct Observed<'l, F> {
    inner: http::BodyReader<'l>,
    seen: u64,
    notify: F,
}

impl<F: FnMut(u64)> std::io::Read for Observed<'_, F> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = std::io::Read::read(&mut self.inner, out)?;
        if n > 0 {
            self.seen += n as u64;
            (self.notify)(self.seen);
        }
        Ok(n)
    }
}

/// Move ONE granted object across the door, re-asking while re-asking is honest.
#[allow(clippy::too_many_arguments)]
pub fn fetch_url(
    store: &Store,
    grant: &DeliveryGrant,
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> Result<Fetched> {
    let client = shared_client();
    let id = grant.object.id();
    let mut last_refusal: Option<Refusal> = None;
    let mut stalled_attempts = 0u32;
    for attempt in 1..=FETCH_ATTEMPTS {
        let last = attempt == FETCH_ATTEMPTS;
        let opened = match open_object(client, url, policy, credential, deadline, ledger) {
            Ok(response) => response,
            Err(refusal) => match refusal.code {
                Code::TRANSFER_FAILED if !last => {
                    last_refusal = Some(refusal);
                    continue;
                }
                _ => {
                    return Err(named(refusal, &id));
                }
            },
        };
        let response = match opened {
            Opened::Body(response) => response,
            Opened::Retryable(status) => {
                if last {
                    return refuse(
                        Code::TRANSFER_FAILED,
                        format!("{id}: the object store answered {status} on every attempt"),
                    );
                }
                continue;
            }
            // An origin's own non-retryable answer is a fact, not weather: no re-ask.
            Opened::Refused(refusal) => return Err(named(refusal, &id)),
        };
        let mut body = Observed {
            inner: response.into_body(),
            seen: 0,
            notify: |total| {
                if let Some(observer) = progress.as_mut() {
                    observer(total);
                }
            },
        };
        match grant.admit(store, &mut body) {
            Ok(admitted) => {
                return Ok(Fetched {
                    admitted,
                    transferred: body.seen,
                })
            }
            Err(door) => {
                let stalled = body.inner.stalled();
                let refusal = body.inner.failure().unwrap_or(door);
                if stalled {
                    stalled_attempts += 1;
                }
                match refusal.code {
                    Code::TRANSFER_FAILED if !last => {
                        last_refusal = Some(refusal);
                        continue;
                    }
                    Code::TRANSFER_FAILED if stalled_attempts > 0 => {
                        return refuse(
                            Code::TRANSFER_FAILED,
                            format!(
                                "{id}: the stream stopped moving bytes {stalled_attempts} \
                                 time(s) of {FETCH_ATTEMPTS} asks, each judged against the \
                                 {:.1} s this pull's own measured pace allowed \
                                 ({STALL_FACTOR} times its longest gap) — this link drops \
                                 streams rather than slowing them; re-run the pull, and if \
                                 it repeats the route between this machine and the object \
                                 store is the subject, not the artifact",
                                ledger.patience().as_secs_f64(),
                            ),
                        );
                    }
                    _ => return Err(named(refusal, &id)),
                }
            }
        }
    }
    Err(last_refusal.unwrap_or(Refusal {
        code: Code::TRANSFER_FAILED,
        detail: format!("{id}: the object store did not answer"),
    }))
}

enum Opened<'l> {
    Body(http::Response<'l>),
    Retryable(u16),
    Refused(Refusal),
}

/// One GET, redirect hops walked BY HAND so every hop faces the same predicate and the
/// credential is re-asked per host (a scoped credential cannot leak onto a CDN).
fn open_object<'l>(
    client: &Client,
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &'l Ledger,
) -> Result<Opened<'l>> {
    let mut target = url.to_string();
    for _hop in 0..=policy.max_redirects {
        let checked = policy.check(&target)?;
        let headers = credential.headers(&checked.host);
        let response = http::request(
            client,
            &checked,
            "GET",
            &headers,
            None,
            deadline,
            ledger,
            true,
            None,
            Code::TRANSFER_FAILED,
        )?;
        let status = response.status;
        if (300..400).contains(&status) {
            let Some(location) = response.header("location").map(str::to_string) else {
                return refuse(Code::REDIRECT_REFUSED, "origin redirected with no Location");
            };
            if !location.contains("://") {
                return refuse(
                    Code::REDIRECT_REFUSED,
                    "origin redirected to a non-absolute URL",
                );
            }
            if policy.max_redirects == 0 {
                return refuse(
                    Code::REDIRECT_REFUSED,
                    "a presigned GET has no business redirecting, and following one would \
                     open a URL the predicate never saw",
                );
            }
            target = location;
            continue;
        }
        if status == 200 {
            return Ok(Opened::Body(response));
        }
        if RETRYABLE_STATUS.contains(&status) {
            return Ok(Opened::Retryable(status));
        }
        return Ok(Opened::Refused(Refusal {
            code: Code::TRANSFER_FAILED,
            detail: format!(
                "the object store answered {status} — a presigned read expires; re-run \
                 the pull to be re-authorized"
            ),
        }));
    }
    refuse(
        Code::REDIRECT_REFUSED,
        format!(
            "more than {} redirect hop(s); a longer chain is a loop or an evasion",
            policy.max_redirects
        ),
    )
}

fn named(refusal: Refusal, id: &str) -> Refusal {
    Refusal {
        code: refusal.code,
        detail: format!("{id}: {}", refusal.detail),
    }
}

/// What the walk moved — counted, never estimated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalkOutcome {
    pub fetched: u64,
    pub bytes_moved: u64,
}

/// Walk `plan.wanted` with `streams` objects in flight. One is the serial walk and it
/// stays reachable, because a link where concurrency does not help is a real link. The
/// first refusal wins; streams already moving finish their object and stop.
#[allow(clippy::too_many_arguments)]
pub fn fetch_wanted(
    store: &Store,
    plan: &FetchPlan,
    urls: &BTreeMap<String, String>,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    streams: usize,
    on_object: Option<ObjectObserver<'_>>,
) -> Result<WalkOutcome> {
    for object in &plan.wanted {
        if !urls.contains_key(&object.sha256) {
            return refuse(
                Code::TRANSFER_FAILED,
                format!(
                    "{}: wanted by the plan and no URL was minted for it — the grant set \
                     covers the wanted set exactly",
                    object.id()
                ),
            );
        }
    }
    let streams = streams.clamp(1, 128).min(plan.wanted.len().max(1));
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let fetched = AtomicU64::new(0);
    let bytes_moved = AtomicU64::new(0);
    let first: Mutex<Option<Refusal>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..streams {
            scope.spawn(|| loop {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                let at = next.fetch_add(1, Ordering::Relaxed);
                let Some(object) = plan.wanted.get(at) else {
                    return;
                };
                let outcome = DeliveryGrant::mint(plan, object).and_then(|grant| {
                    fetch_url(
                        store,
                        &grant,
                        &urls[&object.sha256],
                        policy,
                        credential,
                        deadline,
                        ledger,
                        None,
                    )
                });
                match outcome {
                    Ok(one) => {
                        fetched.fetch_add(1, Ordering::Relaxed);
                        bytes_moved.fetch_add(one.transferred, Ordering::Relaxed);
                        if let Some(observer) = on_object {
                            observer(object, one.transferred);
                        }
                    }
                    Err(refusal) => {
                        stop.store(true, Ordering::Relaxed);
                        let mut slot = first.lock().unwrap();
                        if slot.is_none() {
                            *slot = Some(refusal);
                        }
                        return;
                    }
                }
            });
        }
    });
    if let Some(refusal) = first.lock().unwrap().take() {
        return Err(refusal);
    }
    Ok(WalkOutcome {
        fetched: fetched.load(Ordering::Relaxed),
        bytes_moved: bytes_moved.load(Ordering::Relaxed),
    })
}

/// One pull, described before it runs. `sample_seconds` is the ledger's RESOLUTION —
/// configuration, never a verdict; the default is the runtime's one sampling cadence.
pub struct PullRequest<'a> {
    pub store: &'a Store,
    pub base: &'a str,
    pub refspec: &'a str,
    pub lane: &'a str,
    pub credential: &'a dyn CredentialProvider,
    pub policy: &'a SourcePolicy,
    pub deadline: Deadline,
    pub streams: usize,
    pub session: &'a str,
    pub sample_seconds: f64,
    pub on_object: Option<ObjectObserver<'a>>,
}

impl<'a> PullRequest<'a> {
    pub fn new(
        store: &'a Store,
        base: &'a str,
        refspec: &'a str,
        credential: &'a dyn CredentialProvider,
        policy: &'a SourcePolicy,
    ) -> PullRequest<'a> {
        PullRequest {
            store,
            base,
            refspec,
            lane: "",
            credential,
            policy,
            deadline: Deadline::none(),
            streams: STREAMS,
            session: "",
            sample_seconds: crate::transport::ledger::SAMPLE_SECONDS,
            on_object: None,
        }
    }
}

/// What a pull actually did — counted, never estimated.
#[derive(Debug, Clone, Default)]
pub struct PullReport {
    pub refspec: String,
    pub model: String,
    pub release: String,
    pub lane: String,
    pub manifest: String,
    pub session: String,
    pub streams: usize,
    pub declared: u64,
    pub held: u64,
    pub fetched: u64,
    pub bytes_moved: u64,
    pub bytes_held: u64,
    pub bytes_total: u64,
    pub presence: Vec<(String, &'static str)>,
    pub stamps: Vec<(String, Vec<String>)>,
}

/// Pull one hub checkpoint into a local store: closure, `FetchPlan::of` against the REAL
/// store, presign for exactly `wanted`, the N-way walk, and the completion proof — a local
/// walk over locally verified bytes, never the transfer's own success report.
///
/// A warm store asks the hub one question (`closure`) and moves nothing: no presign call
/// is made when `wanted` is empty.
pub fn pull(request: &PullRequest<'_>) -> Result<PullReport> {
    let ledger = Ledger::with_resolution(std::time::Duration::from_secs_f64(
        request.sample_seconds.clamp(0.001, 3600.0),
    ));
    // Naming a hub base IS declaring it, and a loopback hub declares a local deployment —
    // the port of pull.py's `local_hub` carve-out. The policy itself stays the caller's.
    let mut policy = request.policy.clone();
    if let Ok(host) = policy::base_host(request.base) {
        if !policy.allows_host(&host) {
            policy.allowed_hosts.push(host);
        }
    }
    if policy::is_loopback_origin(request.base) {
        policy.allow_local = true;
    }
    let client = shared_client();
    let closure = hub::closure(
        client,
        request.base,
        request.refspec,
        request.lane,
        request.credential,
        &policy,
        request.deadline,
        &ledger,
    )?;
    let session = if request.session.is_empty() {
        format!("pull-{}", &closure.manifest.sha256[..12])
    } else {
        request.session.to_string()
    };
    let (plan, why) = FetchPlan::of(request.store, &session, &closure.manifest, &closure.objects)?;
    let mut report = PullReport {
        refspec: request.refspec.to_string(),
        model: closure.model.clone(),
        release: closure.release.clone(),
        lane: closure.lane.clone(),
        manifest: closure.manifest.id(),
        session: session.clone(),
        streams: request.streams,
        declared: (plan.held.len() + plan.wanted.len()) as u64,
        held: plan.held.len() as u64,
        bytes_held: plan.held_bytes(),
        bytes_total: plan.declared_bytes(),
        presence: why
            .iter()
            .map(|(object, presence)| (object.id(), presence.as_str()))
            .collect(),
        stamps: closure.stamps.clone(),
        ..Default::default()
    };
    if !plan.wanted.is_empty() {
        let digests: Vec<String> = plan.wanted.iter().map(|o| o.sha256.clone()).collect();
        let urls: BTreeMap<String, String> = hub::presign(
            client,
            request.base,
            &digests,
            request.credential,
            &policy,
            request.deadline,
            &ledger,
        )?
        .into_iter()
        .collect();
        let walked = fetch_wanted(
            request.store,
            &plan,
            &urls,
            &policy,
            request.credential,
            request.deadline,
            &ledger,
            request.streams,
            request.on_object,
        )?;
        report.fetched = walked.fetched;
        report.bytes_moved = walked.bytes_moved;
    }
    match closure.scope.as_str() {
        "runtime" => plan.complete_cozytensors(request.store)?,
        _ => plan.complete(request.store)?,
    }
    Ok(report)
}
