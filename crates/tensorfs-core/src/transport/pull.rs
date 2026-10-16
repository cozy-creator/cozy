//! What every download shares: the object-store answers worth asking again, where an
//! object came from, a batch's fixed URLs, the Hub's minted ones (`Lease`, `Minting`), a run's
//! counts, and the record of a checkpoint made whole under its repository.
//!
//! **A presigned URL is minted just before the object it authorizes is fetched**, a window
//! at a time, and the window is the bound the HUB declares (proto-037): a URL's capped life
//! then bounds one object's transfer, not the whole download.

use crate::err::{refuse, Code, Result};
use crate::ids::{Doc, ObjectRef};
use crate::store::Store;
use crate::transport::http::Client;
use crate::transport::hub;
use crate::transport::ledger::{Deadline, Ledger};
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
    Volume,
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

pub(crate) fn shared_client() -> &'static Client {
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
pub(crate) struct Lease {
    pub(crate) url: String,
    pub(crate) minted: Instant,
    /// What the HUB said this URL is good for.
    pub(crate) lifetime: Duration,
}

impl Lease {
    /// A lease is spent once less than half the life the hub granted it remains.
    ///
    /// Half of the hub's OWN number is not a magic duration: it carries no opinion about how
    /// many seconds anything should take, it moves when the hub moves, and what it buys is a
    /// stated guarantee — every object begins its transfer holding at least half a URL's
    /// life, whatever that life happens to be.
    pub(crate) fn spent(&self) -> bool {
        self.minted.elapsed() * 2 >= self.lifetime
    }
}


#[derive(Default)]
pub(crate) struct Minting {
    active: Mutex<bool>,
    ready: Condvar,
}

impl Minting {
    /// The right to mint, unless another caller holds it.
    pub(crate) fn try_acquire(&self) -> Option<MintOwner<'_>> {
        let mut active = self.active.lock().unwrap();
        (!*active).then(|| {
            *active = true;
            MintOwner(self)
        })
    }

    pub(crate) fn acquire(&self, ledger: &Ledger, deadline: Deadline) -> Result<MintOwner<'_>> {
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

pub(crate) struct MintOwner<'a>(&'a Minting);
impl Drop for MintOwner<'_> {
    fn drop(&mut self) {
        *self.0.active.lock().unwrap() = false;
        self.0.ready.notify_all();
    }
}



/// What the walk moved — counted, never estimated.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WalkOutcome {
    pub fetched: u64,
    pub bytes_moved: u64,
    /// Requests the download kept in flight.
    pub streams: usize,
    /// The most writes into temp files at once (at most `download::WRITERS`).
    pub writers: usize,
    /// GETs sent, and how many of them were dropped as silent or slow and asked again.
    pub requests: u64,
    pub restarts: u64,
}





/// `policy`, plus the hub at `base`: naming a hub declares it, and a loopback hub a local
/// deployment. One resolution per host for whatever uses it.
pub(crate) fn hub_policy(base: &str, policy: &SourcePolicy) -> SourcePolicy {
    let mut policy = policy.clone();
    policy.resolved = Default::default();
    if let Ok(host) = policy::base_host(base) {
        if !policy.allows_host(&host) {
            policy.allowed_hosts.push(host);
        }
    }
    if policy::is_loopback_origin(base) {
        policy.allow_local = true;
    }
    policy
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
pub(crate) fn record_locally(store: &Store, closure: &hub::Closure) -> Result<()> {
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
