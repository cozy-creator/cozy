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
//! **A presigned URL is minted just before the object it authorizes is fetched.** The
//! hub caps a URL's life, and the question that cap answers depends entirely on WHEN the
//! URL is minted. Minting the whole wanted set before the walk starts makes the cap a
//! bound on the WHOLE WALK — the last object's URL is minted at T=0 beside the first, and
//! dies while the walk is still in its opening objects — so the cap silently becomes a cap
//! on how large a closure may be pulled at all, which is not what it was written to be.
//! Minting at the point of use makes the same cap a bound on ONE OBJECT'S transfer, which
//! is what it was sized for and is generous for, and makes resume free: a walk that stops
//! and starts again mints its own URLs and inherits none of the dying ones.
//!
//! **The window is the bound the HUB declares and nothing else (proto-037).** It was two
//! numbers this side chose — 4,096 digests copied from a hub constant that the hub's own
//! body cap made unreachable, and an 8 GiB byte budget. Every model we own is smaller than
//! 8 GiB, so that window never once closed: minting "at the point of use" minted the entire
//! closure at T=0, exactly as the release before it did, and the protection this module was
//! rewritten for was inert from the day it shipped. Meanwhile the count cap was twice the
//! size the hub would read, so `paul/wai-illustrious@17.0.0` — 2,598 objects — passed both
//! bounds and sent a 174,146 B body at a route that reads 131,072.
//!
//! A byte budget is gone rather than corrected. Window size is pure GRANULARITY: an
//! over-large window costs a re-mint, an under-large one costs a round trip, and neither
//! costs a transfer. The only bound with a correctness role is how many digests fit in one
//! request, and that number belongs to the hub, which now says it on every closure.
//!
//! **Which failures deserve a re-ask.** A stall the ledger NAMED, a connection that died,
//! the object store's own retryable answers, and a LOCAL store that was merely busy do; an
//! absent key does not. Asking the same question with the same expired URL is a lie about
//! time — but re-AUTHORIZING is a different act, and it is the one the origin's 403 asked
//! for. So a 403 is judged by the LEASE it refused: one the hub's own declared life says was
//! already spent is re-minted, and one minted moments ago is terminal, because a refusal of
//! a fresh capability is not about time. That gate is an observed fact and not a count — a
//! count would call a genuinely slow object's second 403 terminal, and a slow origin is not
//! a broken one.
//!
//! A refusal at the admission door propagates untouched: the bytes are wrong at the only
//! authority there is. `IO_FAILED` is the one exception, and it is not an exception to the
//! rule but an application of it: the catalog is a rollback-journal SQLite with a busy
//! timeout, and sixteen concurrent writers make "database is locked" a fact about this
//! second. A lock held five seconds ago is not held when we ask again, and it was killing
//! whole multi-gigabyte walks after the bytes had landed. It re-asks with no wait at all,
//! because the wait already happened inside SQLite'''s busy timeout.

use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{DeliveryGrant, FetchPlan};
use crate::ids::{Doc, ObjectRef};
use crate::store::Store;
use crate::transport::http::{self, Client};
use crate::transport::hub::{self, CredentialProvider};
use crate::transport::ledger::{Deadline, Ledger, FETCH_ATTEMPTS, STALL_FACTOR, STREAMS};
use crate::transport::policy::{self, SourcePolicy};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The object-store answers a re-ask is worth attempting for. Everything else is a fact
/// about the request or the grant, and retrying it is a lie about time.
pub const RETRYABLE_STATUS: [u16; 7] = [408, 425, 429, 500, 502, 503, 504];

/// A caller's per-object observer: the object that landed and the bytes this call moved.
pub type ObjectObserver<'a> = &'a (dyn Fn(&ObjectRef, u64) + Sync);

pub(super) fn shared_client() -> &'static Client {
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

/// Move ONE granted object across the door from a URL the caller already holds. The
/// caller then owns the question of whether that URL is still alive — which is why `pull`
/// uses `fetch_authorized` instead.
#[allow(clippy::too_many_arguments)]
pub fn fetch_url(
    store: &Store,
    grant: &DeliveryGrant,
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    progress: Option<&mut dyn FnMut(u64)>,
) -> Result<Fetched> {
    fetch_authorized(
        store,
        grant,
        &|stale: bool| {
            if stale {
                return refuse(
                    Code::TRANSFER_FAILED,
                    "the object store refused this URL and the caller supplied only one — \
                     re-run with a freshly minted URL",
                );
            }
            Ok(url.to_string())
        },
        policy,
        credential,
        deadline,
        ledger,
        progress,
    )
}

/// Move ONE granted object across the door, re-asking while re-asking is honest, and
/// re-AUTHORIZING on every ask.
///
/// `authorize` is consulted once per attempt rather than once per object, because the two
/// are not the same thing when a URL has a life: four attempts against one aging URL can
/// outlast it, and the fourth would then be refused for a reason the first three were not.
/// Re-asking with the same expired URL is a lie about time — but re-minting is not asking
/// again, it is being authorized again, which is honest and is exactly what the object
/// store's refusal asked for.
#[allow(clippy::too_many_arguments)]
pub fn fetch_authorized(
    store: &Store,
    grant: &DeliveryGrant,
    authorize: &dyn Fn(bool) -> Result<String>,
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
    // Set when the origin refused the URL this loop last held. The next ask says so, and
    // the supplier decides whether re-authorizing is honest — it holds the lease and its
    // declared life, which is the only place that question can be answered.
    let mut stale: Option<u16> = None;
    for attempt in 1..=FETCH_ATTEMPTS {
        let last = attempt == FETCH_ATTEMPTS;
        // A refused re-authorization must still name the origin's answer. It is the whole
        // diagnosis: "403, and re-minting was not honest" and "403, and the hub would not
        // mint" are different problems with different owners.
        let url = match authorize(stale.is_some()) {
            Ok(url) => url,
            Err(refusal) => match stale {
                Some(status) => {
                    return refuse(
                        Code::TRANSFER_FAILED,
                        format!(
                            "{id}: the object store answered {status} and re-authorizing \
                             was refused — {}",
                            refusal.detail
                        ),
                    )
                }
                None => return Err(refusal),
            },
        };
        stale = None;
        let opened = match open_object(client, &url, policy, credential, deadline, ledger) {
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
            Opened::Expired(status) => {
                if last {
                    return refuse(
                        Code::TRANSFER_FAILED,
                        format!(
                            "{id}: the object store answered {status} on every attempt, \
                             re-authorizing between them"
                        ),
                    );
                }
                // Ask to be authorized AGAIN — a different act from asking again, and the
                // only one the origin's refusal can be answered with. The supplier refuses
                // if the URL it just handed out was too young for time to be the cause.
                stale = Some(status);
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
                    // IO_FAILED is the LOCAL store, and at sixteen concurrent writers
                    // against a rollback-journal SQLite it is routinely "database is
                    // locked" — a fact about this second, not about the object. It killed
                    // whole multi-gigabyte walks after the bytes had landed. No wait is
                    // added: the wait already happened inside SQLite's busy timeout, and
                    // by the time this object is re-fetched the contention has moved.
                    Code::TRANSFER_FAILED | Code::IO_FAILED if !last => {
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
    /// The origin refused the signature. Whether that is weather or a verdict depends on
    /// the URL's age, which only the supplier knows.
    Expired(u16),
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
        // 401/403 is the object store saying the SIGNATURE is no good — the one answer
        // that a fresh URL can change, and the one that used to abort the walk outright.
        if status == 401 || status == 403 {
            return Ok(Opened::Expired(status));
        }
        return Ok(Opened::Refused(Refusal {
            code: Code::TRANSFER_FAILED,
            detail: format!("the object store answered {status}"),
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

/// Where the walk gets the URL for the object it is ABOUT to fetch — asked at the point of
/// use, and asked again for every re-ask.
///
/// The walk does not know what a presign is, and does not need to: it names the object it
/// has reached and is handed something it may GET. `at` is that object's index in
/// `plan.wanted`, which is the whole reason this is not a bare `Fn(&ObjectRef)` — an
/// implementation that mints has to know where the walk has got to in order to mint AHEAD
/// of it rather than one call per object.
pub trait ObjectUrls: Sync {
    /// `stale` is true when the caller is asking BECAUSE the URL it last held was refused
    /// by the origin. A supplier that can re-mint must not answer with the same URL; one
    /// that cannot must refuse, because handing back the dead URL would turn a re-ask into
    /// an infinite one.
    fn url_for(&self, at: usize, object: &ObjectRef, stale: bool) -> Result<String>;
}

/// URLs already in hand: the shape a caller that did its own authorizing passes. Nothing
/// here re-mints, so the lifetime of these URLs bounds the whole walk — which is precisely
/// why `pull` does not use this impl.
impl ObjectUrls for BTreeMap<String, String> {
    fn url_for(&self, _at: usize, object: &ObjectRef, stale: bool) -> Result<String> {
        if stale {
            return refuse(
                Code::TRANSFER_FAILED,
                format!(
                    "{}: the object store refused the only URL this caller holds — these \
                     URLs were minted elsewhere and nothing here can re-authorize them",
                    object.id()
                ),
            );
        }
        match self.get(&object.sha256) {
            Some(url) => Ok(url.clone()),
            None => refuse(
                Code::TRANSFER_FAILED,
                format!(
                    "{}: wanted by the plan and no URL was minted for it — the grant set \
                     covers the wanted set exactly",
                    object.id()
                ),
            ),
        }
    }
}

/// How wide a mint window is when the hub declared no ceiling: exactly what the streams
/// have in flight.
///
/// This is a DEGRADATION, derived from a fact this side owns, not a copy of a fact the hub
/// owns. A window of `streams` digests is a few hundred bytes and is safe under any body cap
/// a hub could plausibly have; it costs one presign call per wave of objects, which is a
/// round trip and never a transfer. There is deliberately no constant here that guesses at
/// the hub's cap — guessing at it is the defect this module is being repaired for.
fn window_without_a_declared_ceiling(streams: usize) -> usize {
    streams.max(1)
}

/// One minted URL and the life the hub declared for it.
///
/// `pub(super)` for one reason: the rule that decides whether a 403 is re-mintable is the
/// lease's AGE against that declared life, and reaching that decision from the wire means
/// racing a clock. The rule is worth testing directly, with a lease whose age is stated
/// rather than waited for.
pub(super) struct Lease {
    pub(super) url: String,
    pub(super) minted: Instant,
    /// What the HUB said this URL is good for, or `None` when it said nothing.
    pub(super) lifetime: Option<Duration>,
}

impl Lease {
    /// A lease is spent once less than half the life the hub granted it remains.
    ///
    /// Half of the hub's OWN number is not a magic duration: it carries no opinion about how
    /// many seconds anything should take, it moves when the hub moves, and what it buys is a
    /// stated guarantee — every object begins its transfer holding at least half a URL's
    /// life, whatever that life happens to be. A hub that declares nothing leaves nothing to
    /// judge, and then the mint window's own advance is the only bound on a URL's age.
    fn spent(&self) -> bool {
        self.lifetime
            .is_some_and(|life| self.minted.elapsed() * 2 >= life)
    }
}

/// The walk's URL supply: `hub::presign` called just ahead of the workers instead of once
/// for the whole closure.
///
/// Two locks, and they are different questions. `held` is the map, taken for as long as a
/// lookup; `minting` is the right to CALL, held across the network so that N streams
/// arriving at the same exhausted window cost one presign rather than N — which is also
/// what the hub wants, since it single-flights presign per rental.
pub(super) struct Leases<'a> {
    pub(super) client: &'a Client,
    pub(super) base: &'a str,
    pub(super) credential: &'a dyn CredentialProvider,
    pub(super) policy: &'a SourcePolicy,
    pub(super) deadline: Deadline,
    pub(super) ledger: &'a Ledger,
    pub(super) wanted: &'a [ObjectRef],
    /// How many digests one mint asks for — the ceiling the hub declared on this pull's
    /// own closure, or `window_without_a_declared_ceiling` when it declared none. Never a
    /// constant compiled in here.
    pub(super) window: usize,
    pub(super) held: Mutex<BTreeMap<String, Lease>>,
    pub(super) minting: Mutex<()>,
}

impl ObjectUrls for Leases<'_> {
    fn url_for(&self, at: usize, object: &ObjectRef, stale: bool) -> Result<String> {
        if stale && !self.retire(&object.sha256)? {
            // The refused lease was NOT spent: the hub's own declared life still had more
            // than half of it left, so the origin's refusal is not about time and asking to
            // be authorized again is asking the same question. Terminal, at the first 403,
            // with no attempt spent guessing.
            return refuse(
                Code::TRANSFER_FAILED,
                format!(
                    "{}: the object store refused a URL minted moments ago, with most of \
                     the life the hub declared for it still to run — that is a fact about \
                     the object, not about time, and re-minting would ask the same question",
                    object.id()
                ),
            );
        }
        if let Some(url) = self.fresh(&object.sha256) {
            return Ok(url);
        }
        let _minting = self.minting.lock().unwrap();
        // Another stream may have minted this window while this one waited for the right
        // to call. Asking again before calling is what makes one window cost one call.
        if let Some(url) = self.fresh(&object.sha256) {
            return Ok(url);
        }
        self.mint_from(at)?;
        self.fresh(&object.sha256).ok_or_else(|| Refusal {
            code: Code::HUB_REFUSED,
            detail: format!(
                "{}: the mint answered without leaving a usable URL for it — a URL whose \
                 declared life is already spent when it is granted authorizes nothing",
                object.id()
            ),
        })
    }
}

impl Leases<'_> {
    fn fresh(&self, digest: &str) -> Option<String> {
        let held = self.held.lock().unwrap();
        held.get(digest)
            .filter(|lease| !lease.spent())
            .map(|lease| lease.url.clone())
    }

    /// Drop the lease the origin just refused, and answer whether re-minting is honest.
    ///
    /// The gate is the LEASE'S AGE against the life the hub declared for it, which is an
    /// observed fact about this URL and carries no opinion about seconds. It is not a count
    /// of how many times this object has been re-authorized: a count and a freshness test
    /// agree on a fast link and diverge exactly where a slow one lives. An object holding
    /// half a URL's life that takes longer than that to move is slow, not broken, and its
    /// second expiry is as honest as its first.
    ///
    /// A hub that declared no lifetime leaves nothing to judge, and then a refusal is taken
    /// at face value: the window's own advance is the only bound on a URL's age, so a
    /// re-mint could not be argued for.
    fn retire(&self, digest: &str) -> Result<bool> {
        let mut held = self.held.lock().unwrap();
        let Some(lease) = held.remove(digest) else {
            // Nothing held: another stream retired it, and whoever mints next is honest.
            return Ok(true);
        };
        Ok(lease.spent())
    }

    /// Mint from the walk's cursor forward: the object the caller has reached, and enough of
    /// what follows that the streams behind it find their URLs already waiting.
    ///
    /// Objects still holding a fresh lease are skipped. `plan.wanted` is sorted unique by
    /// digest, so any subsequence of it is still the sorted unique list presign requires —
    /// no re-sorting, and no chance of asking for a digest twice.
    ///
    /// The call itself crosses the network to the hub, so it goes out on the pull's own
    /// `Deadline` and teaches the same `Ledger` the object fetches teach: a mint that hangs
    /// must be as visible to liveness as a stream that stops sending, and a cheap call that
    /// hangs is still a hang.
    fn mint_from(&self, at: usize) -> Result<()> {
        let mut digests: Vec<String> = Vec::new();
        for object in &self.wanted[at..] {
            if digests.len() >= self.window {
                break;
            }
            if self.fresh(&object.sha256).is_some() {
                continue;
            }
            digests.push(object.sha256.clone());
        }
        if digests.is_empty() {
            return Ok(());
        }
        // Stamped BEFORE the call, so the call's own latency is spent out of the lease it
        // is fetching rather than granted back to it.
        let minted = Instant::now();
        let answer = hub::presign(
            self.client,
            self.base,
            &digests,
            self.credential,
            self.policy,
            self.deadline,
            self.ledger,
        )?;
        let mut held = self.held.lock().unwrap();
        for (digest, url) in answer.urls {
            held.insert(
                digest,
                Lease {
                    url,
                    minted,
                    lifetime: answer.lifetime,
                },
            );
        }
        Ok(())
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
    urls: &dyn ObjectUrls,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    streams: usize,
    on_object: Option<ObjectObserver<'_>>,
) -> Result<WalkOutcome> {
    // Nothing is checked up front any more, because there is no up-front set to check. The
    // invariant it proved — the grant set covers the wanted set exactly — is enforced per
    // mint instead: presign authorizes every digest it was asked or refuses, never a silent
    // subset, and a supplier with no URL for an object refuses in the same words.
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
                let authorize = |stale: bool| urls.url_for(at, object, stale);
                let outcome = DeliveryGrant::mint(plan, object).and_then(|grant| {
                    fetch_authorized(
                        store, &grant, &authorize, policy, credential, deadline, ledger, None,
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
/// store, the N-way walk presigning `wanted` as it reaches it, and the completion proof — a
/// local walk over locally verified bytes, never the transfer's own success report.
///
/// A warm store asks the hub one question (`closure`) and moves nothing: no presign call is
/// made when `wanted` is empty, and a URL is never minted for an object the walk does not
/// go on to fetch.
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
    // The WHOLE closure, however many pages the hub takes to say it. A closure that fits
    // one page is one call, which is every model we own today; nothing about the walk
    // changes when one day it is not.
    let closure = hub::closure_all(
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
        let leases = Leases {
            client,
            base: request.base,
            credential: request.credential,
            policy: &policy,
            deadline: request.deadline,
            ledger: &ledger,
            wanted: &plan.wanted,
            window: closure
                .presign_max_digests
                .unwrap_or_else(|| window_without_a_declared_ceiling(request.streams)),
            held: Mutex::new(BTreeMap::new()),
            minting: Mutex::new(()),
        };
        let walked = fetch_wanted(
            request.store,
            &plan,
            &leases,
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
    // The proof and the index row both read and write the catalog, and a walk that has just
    // landed gigabytes must not be thrown away because the store was busy for a moment while
    // its own streams were closing. Same rule as the walk's: a contended local store is a
    // fact about this second.
    while_store_busy(|| match closure.scope.as_str() {
        "runtime" => plan.complete_cozytensors(request.store),
        _ => plan.complete(request.store),
    })?;
    while_store_busy(|| record_locally(request.store, &closure))?;
    Ok(report)
}

/// Re-run while the LOCAL store answers `IO_FAILED`, which at this concurrency is dominated
/// by SQLite's "database is locked". Bounded by the same `FETCH_ATTEMPTS` the walk uses, and
/// with no wait of its own: the wait already happened inside the busy timeout, and a lock
/// held a moment ago is not held now. Every other code is taken at face value on the first
/// answer — proto-035's polarity, an allow-list of transient codes.
fn while_store_busy<T>(mut attempt: impl FnMut() -> Result<T>) -> Result<T> {
    let mut last = None;
    for _ in 0..FETCH_ATTEMPTS {
        match attempt() {
            Ok(value) => return Ok(value),
            Err(refusal) if refusal.code == Code::IO_FAILED => last = Some(refusal),
            Err(refusal) => return Err(refusal),
        }
    }
    Err(last.unwrap_or(Refusal {
        code: Code::IO_FAILED,
        detail: "the local store stayed busy".into(),
    }))
}

/// RECORD WHAT THE PULL RESOLVED. The closure carries `model`, `release` and `lane` for
/// exactly this — "the resolved identity the local index row needs" — and until now nothing
/// wrote that row, so a pull left a Store holding a checkpoint it could not answer for.
///
/// Two things broke on that, one loudly and one silently. `Store::resolve_release` reads
/// `repos/<org>/<name>.json`, so a pod that had just fetched a whole checkpoint refused
/// `REPOSITORY_ABSENT` when it asked which manifest a release lane names. And `gc_plan`
/// seeds liveness from `repos/` alone — an owner ruling, "the filesystem itself is
/// self-describing" — so every object the pull had just admitted was formally garbage: a
/// measured 135 MB proof store planned its entire contents for deletion after a successful
/// fetch.
///
/// Both mutations are idempotent, which is what a warm re-fetch needs: an identical
/// checkpoint returns the repository unchanged, and re-setting a lane to the manifest it
/// already names is a no-op the revision fence accepts because the revision is read fresh.
fn record_locally(store: &Store, closure: &hub::Closure) -> Result<()> {
    use crate::repository::{Mutation, ReleaseLane, Repository, RepositoryName};

    let Some((org, name)) = closure.model.split_once('/') else {
        return refuse(
            Code::HUB_REFUSED,
            format!("closure model {:?} is not org/name", closure.model),
        );
    };
    let repo = RepositoryName::new(org, name)?;
    let fault = crate::store::Fault::default();
    let mut observed = std::fs::read(store.repository_path(&repo)).ok();
    let mut latest = observed.as_deref().map(Repository::parse).transpose()?;

    let holds_checkpoint = |repository: Option<&Repository>| {
        repository.is_some_and(|repository| {
            repository
                .checkpoints
                .iter()
                .any(|entry| entry.manifest == closure.manifest)
        })
    };
    let holds_lane = |repository: Option<&Repository>| {
        closure.lane.is_empty()
            || repository.is_some_and(|repository| {
                repository.releases.iter().any(|release| {
                    release.version == closure.release
                        && !release.yanked
                        && release.lanes.iter().any(|lane| {
                            lane.lane == closure.lane && lane.manifest == closure.manifest
                        })
                })
            })
    };
    // A warm pull records nothing, for the same reason it moves nothing: the row already
    // says what this pull would say.
    if holds_checkpoint(latest.as_ref()) && holds_lane(latest.as_ref()) {
        return Ok(());
    }
    if !holds_checkpoint(latest.as_ref()) {
        latest = store.apply_repository(
            observed.as_deref(),
            &Mutation::PutCheckpoint {
                repo: repo.clone(),
                manifest: closure.manifest.clone(),
            },
            &fault,
        )?;
        observed = latest.as_ref().map(Doc::canonical_bytes);
    }
    // A closure that named no lane still gets its checkpoint, which is what keeps the
    // objects live; there is simply no release row to point at them.
    if closure.lane.is_empty() {
        return Ok(());
    }
    let expected_revision = latest
        .as_ref()
        .and_then(|repository| {
            repository
                .releases
                .iter()
                .find(|release| release.version == closure.release)
        })
        .map_or(0, |release| release.revision);
    store.apply_repository(
        observed.as_deref(),
        &Mutation::UpdateRelease {
            expected_revision,
            repo,
            remove: Vec::new(),
            set: vec![ReleaseLane {
                lane: closure.lane.clone(),
                manifest: closure.manifest.clone(),
            }],
            version: closure.release.clone(),
        },
        &fault,
    )?;
    Ok(())
}
