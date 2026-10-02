//! The pull: closure → plan → download → completion proof (tfs-049).
//!
//! The plan is `FetchPlan::of`'s, the wanted objects are downloaded by `download` as ranged
//! chunks, every byte is hashed against its digest before the Store commits it, and the
//! proof is still `complete`. Resume across pulls is per object: the next invocation
//! re-plans against the store and asks only for what is still missing.
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
//! **A 403 is judged by the LEASE it refused.** One the hub's own declared life says was
//! already spent is re-minted, and one minted moments ago is terminal, because a refusal of
//! a fresh capability is not about time.

use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{DeliveryGrant, FetchPlan};
use crate::ids::{Doc, ObjectRef};
use crate::store::Store;
use crate::transport::http::{self, Client};
use crate::transport::hub::{self, CredentialProvider};
use crate::transport::ledger::{Deadline, Ledger, PULL_STREAMS};
use crate::transport::policy::{self, SourcePolicy};
use std::collections::BTreeMap;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The object-store answers a re-ask is worth attempting for. Everything else is a fact
/// about the request or the grant, and retrying it is a lie about time.
pub const RETRYABLE_STATUS: [u16; 7] = [408, 425, 429, 500, 502, 503, 504];

/// A caller's per-object observer: the object that landed and the bytes this call moved.
pub type ObjectObserver<'a> = &'a (dyn Fn(&ObjectRef, u64, ObjectSource) + Sync);

/// Which door answered for one object, as told to an observer. It is on the callback and
/// not only in the final report because the two doors cost different things and only one
/// of them is an invoice: without the split, a datacenter whose cache has silently gone
/// cold is indistinguishable, while the walk runs, from one that is working — the closure
/// still completes and the totals still add up, and the bytes are quietly re-bought.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectSource {
    Cache,
    Origin,
}

/// Hears how many of ONE object's bytes are home, while its fetch runs. The count goes back
/// when an origin that does not range starts the object over.
pub trait FetchObserver {
    fn moved(&mut self, total: u64);
}

impl<F: FnMut(u64)> FetchObserver for F {
    fn moved(&mut self, total: u64) {
        self(total)
    }
}

pub(super) fn shared_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(Client::new)
}

/// What one fetch moved: the object now in the Store (or the staging area), and the bytes
/// this call pulled off the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    pub object: ObjectRef,
    pub transferred: u64,
}

pub(super) enum Opened<'l> {
    Body(http::Response<'l>),
    /// A transient answer, and the wait the origin stated for it, if any.
    Retryable(u16, Option<Duration>),
    /// The origin refused the signature. Whether that is weather or a verdict depends on
    /// the URL's age, which only the supplier knows.
    Expired(u16),
    Refused(Refusal),
}

/// One GET, redirect hops walked BY HAND so every hop faces the same predicate and the
/// credential is re-asked per host (a scoped credential cannot leak onto a CDN).
///
/// `extra` is appended AFTER the credential, and is how a caller asks for one byte range
/// of the object rather than all of it. It changes nothing else about the walk, and that
/// is the point: a ranged fetch is THIS function called once per part, so the hop budget,
/// the per-hop predicate, the per-host credential re-ask, the relative-`Location`
/// resolution and the `Retryable`/`Expired` status arms are not re-implemented alongside
/// it and cannot drift from it. The hop callback names the whole CHECKED url rather than
/// its host so a caller can report the hop it just walked.
#[allow(clippy::too_many_arguments)]
pub(super) fn open_object<'l>(
    client: &Client,
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &'l Ledger,
    extra: &[(String, String)],
    hop: &mut dyn FnMut(&policy::CheckedUrl, u16),
) -> Result<Opened<'l>> {
    let mut target = url.to_string();
    for _hop in 0..=policy.max_redirects {
        let checked = policy.check(&target)?;
        let mut headers = credential.headers(&checked.host);
        headers.extend_from_slice(extra);
        let response = http::request(
            client,
            &checked,
            "GET",
            &headers,
            None,
            deadline,
            ledger,
            http::Judge::Stream,
            None,
            Code::TRANSFER_FAILED,
        )?;
        let status = response.status;
        hop(&checked, status);
        if (300..400).contains(&status) {
            let Some(location) = response.header("location").map(str::to_string) else {
                return refuse(Code::REDIRECT_REFUSED, "origin redirected with no Location");
            };
            let Some(absolute) = absolute_location(&checked, &location) else {
                return refuse(
                    Code::REDIRECT_REFUSED,
                    "origin redirected with an unusable Location",
                );
            };
            if policy.max_redirects == 0 {
                return refuse(
                    Code::REDIRECT_REFUSED,
                    "a presigned GET has no business redirecting, and following one would \
                     open a URL the predicate never saw",
                );
            }
            target = absolute;
            // Read the few redirect bytes so this connection serves the next hop's ask.
            response.discard();
            continue;
        }
        // 206 joins 200 as "the origin answered with a body". A caller that asked for no
        // range never sees one, and if a broken origin sent one anyway the admission door
        // refuses the short body on length. A caller that DID ask for a range needs the
        // partial answer to arrive as a body rather than as an unclassified status.
        if status == 200 || status == 206 {
            return Ok(Opened::Body(response));
        }
        if RETRYABLE_STATUS.contains(&status) {
            let wait = response.stated_wait();
            response.discard();
            return Ok(Opened::Retryable(status, wait));
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

/// RFC 9110 §10.2.2: a `Location` MAY be a relative reference, and resolving one is the
/// client's job. It is not an edge case here — HuggingFace answers a `resolve` URL for a
/// member with NO LFS pointer (a sharded repo's `model.safetensors.index.json`) with a
/// same-origin 307 to `/api/resolve-cache/...`, while every LFS member gets an absolute
/// redirect onto a delivery host. Refusing the relative form therefore refused exactly the
/// index members of every sharded repo, forever, on every pod, while their shards succeeded.
///
/// Resolution is against the hop just walked, so a relative reference cannot leave the
/// origin the predicate already admitted — strictly narrower than the absolute redirects
/// this walk has always followed. The caller re-checks the result regardless: the fence is
/// applied per hop, and this function decides a URL's SHAPE, never its permission.
pub(super) fn absolute_location(base: &policy::CheckedUrl, location: &str) -> Option<String> {
    let location = location.trim();
    if location.is_empty() {
        return None;
    }
    if location.contains("://") {
        return Some(location.to_string());
    }
    let scheme = if base.https { "https" } else { "http" };
    // A protocol-relative reference carries its own authority; only the scheme is inherited.
    if let Some(rest) = location.strip_prefix("//") {
        return (!rest.is_empty()).then(|| format!("{scheme}://{rest}"));
    }
    let default_port = if base.https { 443 } else { 80 };
    let origin = if base.port == default_port {
        format!("{scheme}://{}", base.host)
    } else {
        format!("{scheme}://{}:{}", base.host, base.port)
    };
    if location.starts_with('/') {
        return Some(format!("{origin}{location}"));
    }
    // A path-relative reference resolves against the base's directory. The base query is
    // dropped, which is what RFC 3986 §5.3 does and what every browser does.
    let path = base.target.split('?').next().unwrap_or("/");
    let directory = match path.rfind('/') {
        Some(cut) => &path[..=cut],
        None => "/",
    };
    Some(format!("{origin}{directory}{location}"))
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

    /// Cooperative cancellation for an individual redundant attempt. Providers
    /// that wait for remote authority must consult this ledger during that wait.
    fn url_for_while(
        &self,
        at: usize,
        object: &ObjectRef,
        stale: bool,
        ledger: &Ledger,
    ) -> Result<String> {
        ledger.check_running()?;
        let result = self.url_for(at, object, stale);
        ledger.check_running()?;
        result
    }
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

/// One minted URL and the life the hub declared for it.
///
/// `pub(super)` for one reason: the rule that decides whether a 403 is re-mintable is the
/// lease's AGE against that declared life, and reaching that decision from the wire means
/// racing a clock. The rule is worth testing directly, with a lease whose age is stated
/// rather than waited for.
pub(super) struct Lease {
    pub(super) url: String,
    pub(super) minted: Instant,
    /// What the HUB said this URL is good for.
    pub(super) lifetime: Duration,
}

impl Lease {
    /// A lease is spent once less than half the life the hub granted it remains.
    ///
    /// Half of the hub's OWN number is not a magic duration: it carries no opinion about how
    /// many seconds anything should take, it moves when the hub moves, and what it buys is a
    /// stated guarantee — every object begins its transfer holding at least half a URL's
    /// life, whatever that life happens to be.
    fn spent(&self) -> bool {
        self.minted.elapsed() * 2 >= self.lifetime
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
    /// own closure. Never a constant compiled in here.
    pub(super) window: usize,
    pub(super) held: Mutex<BTreeMap<String, Lease>>,
    pub(super) minting: Minting,
}

#[derive(Default)]
pub(super) struct Minting {
    active: Mutex<bool>,
    ready: Condvar,
}

impl Minting {
    fn acquire(&self, ledger: &Ledger, deadline: Deadline) -> Result<MintOwner<'_>> {
        let mut active = self.active.lock().unwrap();
        while *active {
            ledger.check_running()?;
            let tick = deadline
                .remaining()?
                .map_or(ledger.sample(), |left| left.min(ledger.sample()));
            active = self.ready.wait_timeout(active, tick).unwrap().0;
        }
        ledger.check_running()?;
        deadline.remaining()?;
        *active = true;
        Ok(MintOwner(self))
    }
}

struct MintOwner<'a>(&'a Minting);
impl Drop for MintOwner<'_> {
    fn drop(&mut self) {
        *self.0.active.lock().unwrap() = false;
        self.0.ready.notify_all();
    }
}

impl ObjectUrls for Leases<'_> {
    fn url_for(&self, at: usize, object: &ObjectRef, stale: bool) -> Result<String> {
        self.url_for_while(at, object, stale, self.ledger)
    }

    fn url_for_while(
        &self,
        at: usize,
        object: &ObjectRef,
        stale: bool,
        ledger: &Ledger,
    ) -> Result<String> {
        ledger.check_running()?;
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
        let _minting = self.minting.acquire(ledger, self.deadline)?;
        ledger.check_running()?;
        // Another stream may have minted this window while this one waited for the right
        // to call. Asking again before calling is what makes one window cost one call.
        if let Some(url) = self.fresh(&object.sha256) {
            return Ok(url);
        }
        self.mint_from(at, ledger)?;
        if let Some(url) = self.fresh(&object.sha256) {
            return Ok(url);
        }
        // The mint answered and the URL was STILL not usable. With `minted` stamped on
        // arrival this can only mean the hub granted a life it had already spent — it
        // declared `expires_at_unix` at or before its own `server_time_unix` — so say
        // that, and say the number, instead of leaving the reader to infer which of the
        // two clocks was wrong.
        refuse(
            Code::HUB_REFUSED,
            format!(
                "{}: the mint answered but the URL was unusable on arrival; the hub \
                 declared a life of {:?} for it, and a URL whose declared life is already \
                 spent when it is granted authorizes nothing",
                object.id(),
                self.declared_lifetime(&object.sha256)
            ),
        )
    }
}

impl Leases<'_> {
    /// What the hub said this digest's URL was good for, for a refusal to quote.
    fn declared_lifetime(&self, digest: &str) -> Option<Duration> {
        let held = self.held.lock().unwrap();
        held.get(digest).map(|lease| lease.lifetime)
    }

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
    ///
    /// That was an aspiration until tfs-106 and is now the mechanism. The call runs under
    /// `Judge::Transfer`, so it faults once THIS PULL has landed nothing for longer than its
    /// own measured pace allows — never because the mint itself took long. It matters that
    /// the verdict is reached HERE, holding `minting`: the fifteen streams parked on that
    /// mutex cannot observe anything, so the only thread that can end their wait is the one
    /// that owns it.
    fn mint_from(&self, at: usize, ledger: &Ledger) -> Result<()> {
        let mut digests: Vec<String> = Vec::new();
        // From the asker's place onward, then the unminted objects before it: whichever
        // worker reaches the hub first, one window covers every object it can hold. Starting
        // at `at` alone left the objects of workers that started earlier but asked later for
        // a second presign on a pull the window could have covered in one.
        for object in self.wanted[at..].iter().chain(&self.wanted[..at]) {
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
        let answer = hub::presign(
            self.client,
            self.base,
            &digests,
            self.credential,
            self.policy,
            self.deadline,
            ledger,
        )?;
        // Stamped when the ANSWER ARRIVES (tfs-102). It used to be stamped before the
        // call, "so the call's own latency is spent out of the lease it is fetching
        // rather than granted back to it" — but the URL does not exist for any of that
        // time. The hub signs it while handling the call and states, in its own clock,
        // the life it granted; the request leg, the hub's queueing, custody's signing and
        // every retry `call` makes inside `presign` are all time the URL had not begun.
        //
        // Charging them to it made a slow mint POISON ITS OWN ANSWER. The hub's lifetime
        // is ten minutes and a lease is spent at half of it, so a presign that took five
        // minutes — one flaky tunnel, or two of `call`'s own retries — returned URLs that
        // were unusable the instant they arrived, and the walk refused terminally on a
        // paid pod with "a URL whose declared life is already spent when it is granted
        // authorizes nothing". Twice on 2026-09-08, on gayong and weed, each after ~82 GB
        // had already moved at 378 MB/s.
        //
        // What is left uncharged is the return leg alone, which is one hop and bounded by
        // the call's own deadline.
        let minted = Instant::now();
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

/// What the walk moved — counted, never estimated, and attributed to the source that
/// actually answered. `cached` objects crossed the same admission door as `fetched` ones;
/// they simply came off a mounted cache instead of the origin, and saying which is the
/// only way an operator can tell a warm datacenter from a cold one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WalkOutcome {
    pub fetched: u64,
    pub bytes_moved: u64,
    pub cached: u64,
    pub bytes_cached: u64,
    /// Requests the download kept in flight.
    pub streams: usize,
    /// GETs sent, and how many of them were dropped as silent or slow and asked again.
    pub requests: u64,
    pub restarts: u64,
}

/// Walk `plan.wanted` with `streams` objects in flight. One is the serial walk and it
/// stays reachable, because a link where concurrency does not help is a real link. The
/// first refusal wins and cancels in-flight network waits. Verified objects remain local.
///
/// **`cache` is the source order, and it is the whole reason a second pod is cheap.** The
/// order is local Store, then the optional mounted cache, then the origin — the Store half
/// is already spent by the time this is called, because `FetchPlan::of` planned against the
/// real store and `wanted` is what it does not hold. So this walk owns the other two hops,
/// and it owns them per object rather than per pull: a half-warm cache answers the half it
/// holds and the origin answers the rest, in one walk, with no mode to choose between.
///
/// A hit skips the origin ENTIRELY — no presigned URL is minted for an object that never
/// crosses the network, which is the same rule the walk already applies to a held object.
/// A miss, a corrupt final, or an unreadable mount is weather: it falls through to the
/// origin and the pull is unharmed, which is the property that lets a disposable cache be
/// deleted at any moment without making a working grant fail.
///
/// Verified origin objects are offered to the existing bounded cache write pool.
/// Saturation retains only declared object references; origin workers never wait
/// for the cache. Once acquisition finishes, rejected offers are retried while
/// cache copies make progress. The CLI drains accepted work and reports incomplete
/// replication on cache failure. A failed pull does not promise a complete cache.
/// `streams` is the most requests in flight (`download`).
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
    fetch_wanted_bounded(
        store, plan, urls, policy, credential, deadline, ledger, streams, on_object, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fetch_wanted_bounded(
    store: &Store,
    plan: &FetchPlan,
    urls: &dyn ObjectUrls,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    streams: usize,
    on_object: Option<ObjectObserver<'_>>,
    cache_writes: Option<&crate::repo_cache::CacheWriteReceipt>,
) -> Result<WalkOutcome> {
    let backfill = store.repo_cache().map(|cache| {
        crate::repo_cache::BackfillBatch::new(cache.clone(), store, plan.wanted.len())
            .observe(cache_writes.cloned().unwrap_or_default())
    });
    // Descriptors and memory are budgets: the soft descriptor limit is raised to the hard
    // one, and the download never runs more requests than either can hold (run 1297:
    // EMFILE under a 1,024 soft limit at 128 streams).
    crate::descriptors::raise();
    let streams = crate::descriptors::streams_within(super::memory::transfer_streams(
        streams.clamp(1, PULL_STREAMS),
    ));
    let walked = super::download::Download {
        store,
        wanted: &plan.wanted,
        grant: &|object| DeliveryGrant::mint(plan, object),
        urls,
        chunk: super::download::CHUNK_BYTES,
        ranged: true,
        policy,
        credential,
        deadline,
        ledger,
        on_object,
        backfill: backfill.as_ref(),
        keep: None,
    }
    .run(streams, None)?;
    if let Some(backfill) = backfill {
        backfill.finish_offers();
    }
    Ok(walked)
}

/// One pull, described before it runs. `sample_seconds` is the ledger's RESOLUTION —
/// configuration, never a verdict; the default is the runtime's one sampling cadence.
pub struct PullRequest<'a> {
    pub cancellation: Option<super::PullCancellation>,
    pub store: &'a Store,
    pub base: &'a str,
    pub refspec: &'a str,
    pub lane: &'a str,
    pub credential: &'a dyn CredentialProvider,
    pub policy: &'a SourcePolicy,
    pub deadline: Deadline,
    /// The most requests in flight.
    pub streams: usize,
    pub session: &'a str,
    pub sample_seconds: f64,
    /// Remaining admission allowance supplied by the host: new bytes and objects,
    /// excluding content already held. Does not change the Store's durable quota.
    pub download_budget: Option<(u64, u64)>,
    /// The native plan, once, before body transfers. Observability only; consumers can
    /// report the exact declared and already-held bytes without resolving a second plan.
    pub on_plan: Option<&'a (dyn Fn(&FetchPlan) + Sync)>,
    pub on_object: Option<ObjectObserver<'a>>,
    /// A closure the caller already resolved with [`resolve_closure`]; the pull asks the
    /// hub nothing more about it.
    pub closure: Option<&'a hub::Closure>,
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
            cancellation: None,
            store,
            base,
            refspec,
            lane: "",
            credential,
            policy,
            deadline: Deadline::none(),
            streams: PULL_STREAMS,
            session: "",
            sample_seconds: crate::transport::ledger::SAMPLE_SECONDS,
            download_budget: None,
            on_plan: None,
            on_object: None,
            closure: None,
        }
    }
}

/// The pull's egress policy: the caller's, plus the hub it named.
fn pull_policy(request: &PullRequest<'_>) -> SourcePolicy {
    let mut policy = request.policy.clone();
    // One resolution per host for this pull, however many objects it asks for.
    policy.resolved = Default::default();
    // Naming a hub base IS declaring it, and a loopback hub declares a local deployment.
    if let Ok(host) = policy::base_host(request.base) {
        if !policy.allows_host(&host) {
            policy.allowed_hosts.push(host);
        }
    }
    if policy::is_loopback_origin(request.base) {
        policy.allow_local = true;
    }
    policy
}

fn pull_ledger(request: &PullRequest<'_>) -> Ledger {
    let ledger = Ledger::with_resolution(std::time::Duration::from_secs_f64(
        request.sample_seconds.clamp(0.001, 3600.0),
    ));
    match &request.cancellation {
        Some(cancellation) => ledger.with_cancellation(cancellation.clone()),
        None => ledger,
    }
}

/// What `request` names, exactly: the Store's own answer when the ref is digest-pinned to a
/// release lane it already holds whole, else the hub's closure.
pub fn resolve_closure(request: &PullRequest<'_>) -> Result<hub::Closure> {
    let ledger = pull_ledger(request);
    ledger.check_running()?;
    if let Some(closure) = pinned_local_closure(request) {
        if matches!(plan_against_store(request, &closure), Ok((plan, _)) if plan.wanted.is_empty())
        {
            return Ok(closure);
        }
    }
    hub::closure_all(
        shared_client(),
        request.base,
        request.refspec,
        request.lane,
        request.credential,
        &pull_policy(request),
        request.deadline,
        &ledger,
    )
}

/// What a pull actually did — counted, never estimated.
#[derive(Debug, Clone, Default)]
pub struct PullReport {
    pub scope: String,
    pub cache_writes: crate::repo_cache::CacheWriteReceipt,
    pub refspec: String,
    pub model: String,
    pub release: String,
    pub lane: String,
    pub manifest: String,
    pub session: String,
    /// Requests the download kept in flight, the GETs it sent, and how many of those it
    /// dropped as silent or slow and asked again.
    pub streams: usize,
    pub requests: u64,
    pub restarts: u64,
    pub declared: u64,
    pub held: u64,
    pub fetched: u64,
    pub bytes_moved: u64,
    /// What the mounted cache answered for, and what that spared the origin. Zero with no
    /// cache and zero on a cold one; a warm datacenter is the case where these carry the
    /// closure and `bytes_moved` is small.
    pub cached: u64,
    pub bytes_cached: u64,
    pub bytes_held: u64,
    pub bytes_total: u64,
    pub presence: Vec<(String, &'static str)>,
    /// The disk the Store lives on and what making this pull durable cost.
    pub durability: crate::unsynced::SyncCost,
}

/// Pull one hub checkpoint into a local store: closure, `FetchPlan::of` against the REAL
/// store, the N-way walk presigning `wanted` as it reaches it, and the completion proof — a
/// local walk over locally verified bytes, never the transfer's own success report.
///
/// A warm store asks the hub one question (`closure`) and moves nothing: no presign call is
/// made when `wanted` is empty, and a URL is never minted for an object the walk does not
/// go on to fetch. A digest-pinned ask whose release lane the local index already maps to
/// that manifest, over a closure this Store trusts, asks the hub nothing at all.
pub fn pull(request: &PullRequest<'_>) -> Result<PullReport> {
    // Keep the presence proof, transfer and retained repository in one GC-safe span.
    // The shared native writer fence permits concurrent pulls; GC refuses STORE_BUSY.
    let ledger = pull_ledger(request);
    ledger.check_running()?;
    let _retention = crate::catalog::WriterGuard::acquire(request.store.root())?;
    let policy = pull_policy(request);
    // A digest-pinned ask for a release lane this Store already records, over a closure
    // it already trusts, has nothing to learn from the hub (proto-061 A).
    if let Some(closure) = request
        .closure
        .cloned()
        .or_else(|| pinned_local_closure(request))
    {
        if let Ok((plan, report)) = plan_against_store(request, &closure) {
            // A canceled pull can land every object before recording its source.
            // Supplied closures still need that custody committed below.
            if plan.wanted.is_empty() && records_checkpoint(request.store, &closure) {
                if let Some(observer) = request.on_plan {
                    observer(&plan);
                }
                ledger.check_running()?;
                return Ok(report);
            }
        }
    }
    let client = shared_client();
    // The WHOLE closure, however many pages the hub takes to say it. A closure that fits
    // one page is one call, which is every model we own today; nothing about the walk
    // changes when one day it is not.
    // A Store-answered closure authorizes no presign; if its objects went since, ask.
    let closure = match request.closure {
        Some(closure) if closure.presign_max_digests > 0 => closure.clone(),
        _ => hub::closure_all(
            client,
            request.base,
            request.refspec,
            request.lane,
            request.credential,
            &policy,
            request.deadline,
            &ledger,
        )?,
    };
    let (plan, mut report) = plan_against_store(request, &closure)?;
    if let Some((bytes, objects)) = request.download_budget {
        let wanted_bytes = plan.wanted_bytes();
        let wanted_objects = plan.wanted.len() as u64;
        if wanted_bytes > bytes || wanted_objects > objects {
            return Err(Refusal {
                code: Code::CAPACITY_EXHAUSTED,
                detail: format!(
                    "download needs {wanted_bytes} new bytes and {wanted_objects} new objects; \
                     host allowance is {bytes} bytes and {objects} objects"
                ),
            });
        }
    }
    if let Some(observer) = request.on_plan {
        observer(&plan);
    }
    ledger.check_running()?;
    // THE ADMISSION GATE, and it stands here for two reasons (tfs-064).
    //
    // It is BEFORE the walk, so a pull that cannot fit costs a closure call and no bytes —
    // a refusal at the door instead of a filesystem that blocks writes at 60% of a
    // transfer, which is indistinguishable from a stall and has cost two nights.
    //
    // And it charges `wanted_bytes()`, never `declared_bytes()`: the plan has already
    // split the closure against the real store, so an object this store already holds is
    // not bought twice. On a content-addressed store that difference is the whole point —
    // a second model sharing a text encoder and a VAE with the first may add almost
    // nothing — and the two figures agree exactly when nothing is shared, which is exactly
    // the fixture a careless proof would use.
    request.store.admits(plan.wanted_bytes())?;
    // THE TRANSFER STARTS HERE, and the control plane's clock starts with it. Everything
    // above is local — opening the store, splitting the closure against what it already
    // holds, the disk-budget check — and charging that to the transfer would let a slow
    // planning pass condemn the very first mint that followed it.
    ledger.transfer_begins();
    if !plan.wanted.is_empty() {
        // On a persistent disk the walk admits without fsync and fsyncs its objects in
        // batches, the last one before anything records the checkpoint.
        let epoch = crate::unsynced::Epoch::begin(request.store)?;
        let store = request.store.syncing(epoch.clone());
        let leases = Leases {
            client,
            base: request.base,
            credential: request.credential,
            policy: &policy,
            deadline: request.deadline,
            ledger: &ledger,
            wanted: &plan.wanted,
            window: closure.presign_max_digests,
            held: Mutex::new(BTreeMap::new()),
            minting: Minting::default(),
        };
        let walked = fetch_wanted_bounded(
            &store,
            &plan,
            &leases,
            &policy,
            request.credential,
            request.deadline,
            &ledger,
            request.streams,
            request.on_object,
            Some(&report.cache_writes),
        )?;
        report.streams = walked.streams;
        report.requests = walked.requests;
        report.restarts = walked.restarts;
        report.fetched = walked.fetched;
        report.bytes_moved = walked.bytes_moved;
        report.cached = walked.cached;
        report.bytes_cached = walked.bytes_cached;
        if let Some(epoch) = epoch {
            (report.durability.syncs, report.durability.seconds) = epoch.finish()?;
        }
    }
    ledger.check_running()?;
    // Nothing moved and this Store already records the checkpoint, which it does only once
    // the closure is local: the plan's presence pass was the whole proof, so there is no
    // re-walk, no index write, and no cache offer. Held objects are never offered to the
    // cache; it is filled by the pull that fetched them from the origin.
    if plan.wanted.is_empty() && records_checkpoint(request.store, &closure) {
        return Ok(report);
    }
    // The proof and the index row both write the catalog; a contended catalog waits for its
    // live holder (`catalog::lock`), so a busy moment never throws a landed walk away.
    match closure.scope.as_str() {
        "runtime" => plan.complete_cozytensors(request.store),
        _ => plan.complete(request.store),
    }?;
    record_locally(request.store, &closure)?;
    Ok(report)
}

/// Split `closure` against the store and build the report a pull that moves nothing returns.
fn plan_against_store(
    request: &PullRequest<'_>,
    closure: &hub::Closure,
) -> Result<(FetchPlan, PullReport)> {
    let session = if request.session.is_empty() {
        format!("pull-{}", &closure.manifest.sha256[..12])
    } else {
        request.session.to_string()
    };
    let (plan, why) = FetchPlan::of(request.store, &session, &closure.manifest, &closure.objects)?;
    let report = PullReport {
        scope: closure.scope.clone(),
        refspec: request.refspec.to_string(),
        model: closure.model.clone(),
        release: closure.release.clone(),
        lane: closure.lane.clone(),
        manifest: closure.manifest.id(),
        session,
        declared: (plan.held.len() + plan.wanted.len()) as u64,
        held: plan.held.len() as u64,
        bytes_held: plan.held_bytes(),
        bytes_total: plan.declared_bytes(),
        presence: why
            .iter()
            .map(|(object, presence)| (object.id(), presence.as_str()))
            .collect(),
        durability: crate::unsynced::SyncCost {
            class: request.store.disk_class(),
            filesystem: request.store.disk().kind,
            ..Default::default()
        },
        ..Default::default()
    };
    Ok((plan, report))
}

/// The closure this Store can answer for itself: the refspec is `org/name@release@sha256:<hex>`,
/// a lane is named, and the local release index maps (model, release, lane) to exactly that
/// manifest. Anything else — an unpinned ref, a moved pointer, a manifest or header this
/// Store cannot read — is a question for the hub.
fn pinned_local_closure(request: &PullRequest<'_>) -> Option<hub::Closure> {
    let mut parts = request.refspec.split('@');
    let (model, release, digest) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() || release.is_empty() || request.lane.is_empty() {
        return None;
    }
    let pinned = digest.strip_prefix("sha256:")?;
    let (org, name) = model.split_once('/')?;
    let repo = crate::repository::RepositoryName::new(org, name).ok()?;
    let bytes = std::fs::read(request.store.repository_path(&repo)).ok()?;
    let repository = crate::repository::Repository::parse(&bytes).ok()?;
    let manifest = repository
        .releases
        .iter()
        .find(|entry| entry.version == release && !entry.yanked)?
        .lanes
        .iter()
        .find(|lane| lane.lane == request.lane && lane.manifest.sha256 == pinned)?
        .manifest
        .clone();
    if !repository
        .checkpoints
        .iter()
        .any(|entry| entry.manifest == manifest)
    {
        return None;
    }
    let document = request.store.read_manifest(&manifest).ok()?;
    let walked = crate::checkpoint::walk_cozytensors(request.store, &document).ok()?;
    let objects: Vec<ObjectRef> = walked
        .distinct()
        .into_iter()
        .filter(|object| object.sha256 != manifest.sha256)
        .cloned()
        .collect();
    Some(hub::Closure {
        model: model.to_string(),
        release: release.to_string(),
        lane: request.lane.to_string(),
        scope: "runtime".to_string(),
        manifest,
        objects,
        complete: true,
        // Served only when nothing is wanted, so it authorizes no presign.
        presign_max_digests: 0,
    })
}

/// The local index row already says what this pull would record.
fn records_checkpoint(store: &Store, closure: &hub::Closure) -> bool {
    let Some(repo) = closure
        .model
        .split_once('/')
        .and_then(|(org, name)| crate::repository::RepositoryName::new(org, name).ok())
    else {
        return false;
    };
    let latest = std::fs::read(store.repository_path(&repo))
        .ok()
        .and_then(|bytes| crate::repository::Repository::parse(&bytes).ok());
    holds_checkpoint(latest.as_ref(), closure) && holds_lane(latest.as_ref(), closure)
}

fn holds_checkpoint(
    repository: Option<&crate::repository::Repository>,
    closure: &hub::Closure,
) -> bool {
    repository.is_some_and(|repository| {
        repository
            .checkpoints
            .iter()
            .any(|entry| entry.manifest == closure.manifest)
    })
}

fn holds_lane(repository: Option<&crate::repository::Repository>, closure: &hub::Closure) -> bool {
    closure.lane.is_empty()
        || repository.is_some_and(|repository| {
            repository.releases.iter().any(|release| {
                release.version == closure.release
                    && !release.yanked
                    && release
                        .lanes
                        .iter()
                        .any(|lane| lane.lane == closure.lane && lane.manifest == closure.manifest)
            })
        })
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
/// Record the verified checkpoint and lane in one transaction. Concurrent additions to
/// other lanes merge; a conflicting change to this lane retains its refusal.
fn record_locally(store: &Store, closure: &hub::Closure) -> Result<()> {
    use crate::repository::{Mutation, ReleaseLane, Repository, RepositoryName};

    let Some((org, name)) = closure.model.split_once('/') else {
        return refuse(
            Code::HUB_REFUSED,
            format!("closure model {:?} is not org/name", closure.model),
        );
    };
    let repo = RepositoryName::new(org, name)?;
    let path = store.repository_path(&repo);
    let observed = match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(crate::store::classify_io("read pull repository", error)),
    };
    let latest = observed.as_deref().map(Repository::parse).transpose()?;
    if holds_checkpoint(latest.as_ref(), closure) && holds_lane(latest.as_ref(), closure) {
        return Ok(());
    }
    // Keep the full intent even when the observed checkpoint is already present: a stale
    // commit may converge only if both the exact checkpoint and requested lane survive.
    let mut mutations = vec![Mutation::PutCheckpoint {
        repo: repo.clone(),
        manifest: closure.manifest.clone(),
    }];
    if !closure.lane.is_empty() {
        let expected_revision = latest
            .as_ref()
            .and_then(|repository| {
                repository
                    .releases
                    .iter()
                    .find(|release| release.version == closure.release)
            })
            .map_or(0, |release| release.revision);
        mutations.push(Mutation::UpdateRelease {
            expected_revision,
            repo,
            remove: Vec::new(),
            set: vec![ReleaseLane {
                extra: Default::default(),
                lane: closure.lane.clone(),
                manifest: closure.manifest.clone(),
            }],
            version: closure.release.clone(),
        });
    }
    store.apply_cached_repository(
        observed.as_deref(),
        &mutations,
        &crate::store::Fault::default(),
    )?;
    Ok(())
}

#[cfg(test)]
mod mint_wait_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn mint_owner_completion_wakes_waiters_and_attempt_stop_is_local() {
        for cancel in [false, true] {
            let mint = Minting::default();
            let root = Ledger::with_resolution(if cancel {
                Duration::from_millis(10)
            } else {
                Duration::from_secs(5)
            });
            let owner = mint.acquire(&root, Deadline::none()).unwrap();
            let attempt = std::sync::Arc::new(super::super::ledger::Attempt::default());
            let local = root.for_request(std::sync::Arc::clone(&attempt), root.sample());
            std::thread::scope(|scope| {
                let (ready, waiting) = std::sync::mpsc::channel();
                let (done, finished) = std::sync::mpsc::channel();
                let mint = &mint;
                let local = &local;
                scope.spawn(move || {
                    ready.send(()).unwrap();
                    let result = mint.acquire(local, Deadline::none());
                    done.send(result.is_ok()).unwrap();
                });
                waiting.recv_timeout(Duration::from_secs(1)).unwrap();
                std::thread::sleep(Duration::from_millis(20));
                let owner = if cancel {
                    attempt.stopped.store(true, Ordering::Release);
                    Some(owner)
                } else {
                    drop(owner);
                    None
                };
                let result = finished.recv_timeout(Duration::from_millis(200));
                drop(owner); // release even when the negative control times out
                assert_eq!(result.unwrap(), !cancel);
            });
            root.check_running().unwrap();
        }
    }
}
