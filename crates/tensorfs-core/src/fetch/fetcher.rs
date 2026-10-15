//! The Store's one downloader. Runtime and TensorD say what they want and how urgently; the
//! [`Fetcher`] fetches each object once, most urgent first, through the pull's downloader:
//! 8 MiB ranged chunks, slow ones asked again against their peers, every byte through the
//! Store's door.
//!
//! - **Tickets.** [`Fetcher::want`] answers a [`Ticket`]: interest in objects of one origin at
//!   one [`Priority`]. Objects are deduplicated by digest, and an object is as urgent as the
//!   most urgent ticket that wants it. Dropping a ticket releases its interest: nothing is
//!   fetched without a live ticket, and a request already on the wire stops.
//! - **Structure.** A ticket also wants its origin's structure (manifest, header, assets),
//!   which lands first. The origin is then attested ([`crate::origin`]): a checkpoint held in
//!   part is a cache root whose absent objects its hub holds.
//! - **Origins.** One closure per origin, resolved when something must be fetched and kept.
//!   Presigns are batched, and the next window is minted while the current one downloads.
//! - **Holds.** A ticket's objects are held from GC while it lives.
//! - **Disk.** An object is admitted before it is opened. A short disk is relieved by GC
//!   (garbage, idle partial checkpoints' objects, least recently used cache roots); one still
//!   short refuses the object `CAPACITY_EXHAUSTED`.
//! - **Failure.** A refused object fails the waits on it; a later want or wait asks again.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, PoisonError, RwLock, Weak};
use std::time::Duration;

use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{DeliveryGrant, FetchPlan};
use crate::ids::{Doc, ObjectRef};
use crate::meta::{Hold, Meta};
use crate::store::Store;
use crate::transport::{
    self, Cell, CredentialProvider, Deadline, Download, Feed, Lease, Ledger, Minting, ObjectSource, Opening,
    PullCancellation, Pulse, SourcePolicy, Supply, BEHIND, CHUNK_BYTES, NEXT, NOW,
};

/// A checkpoint at its hub: where the objects absent here come from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Origin {
    /// The hub's base URL.
    pub hub: String,
    /// `org/name@release`.
    pub refspec: String,
    pub lane: String,
    pub manifest: ObjectRef,
}

impl Origin {
    fn evidence(&self) -> String {
        format!("{} {}", self.hub, self.refspec)
    }
}

/// What fetches from an origin present: its credential and egress policy. The hub itself is
/// always allowed.
pub trait Access: Send + Sync {
    fn credential(&self) -> &dyn CredentialProvider;
    fn policy(&self) -> &SourcePolicy;
}

/// A credential and egress policy held together: the plainest [`Access`].
pub struct Plain<C> {
    pub credential: C,
    pub policy: SourcePolicy,
}

impl<C: CredentialProvider + Send + Sync> Access for Plain<C> {
    fn credential(&self) -> &dyn CredentialProvider {
        &self.credential
    }
    fn policy(&self) -> &SourcePolicy {
        &self.policy
    }
}

/// How urgent a ticket's objects are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    /// A reader is blocked on them.
    Now,
    /// A run will read them: runs in acceptance order, then by rank within a run.
    Next { run: u64, rank: u32 },
    /// Nothing waits on them (warm sets, hints): fetched only while nothing more urgent is out.
    Behind,
}

impl Priority {
    /// The downloader's order: smaller is sooner. Runs past 2^30 order as the last.
    fn key(self) -> u64 {
        match self {
            Priority::Now => NOW,
            Priority::Next { run, rank } => NEXT | (run.min((1 << 30) - 1) << 32) | u64::from(rank),
            Priority::Behind => BEHIND,
        }
    }
}

/// The Store's downloader; clones share it. [`Fetcher::of`] is this process's for a Store.
#[derive(Clone)]
pub struct Fetcher(Arc<Handle>);

struct Handle {
    shared: Arc<Shared>,
}

impl Drop for Handle {
    /// The last handle ends the downloader and waits for it: no request or temp file of this
    /// fetcher outlives it.
    fn drop(&mut self) {
        self.shared.stop.cancel();
        self.shared.changed.notify_all();
        self.shared.pulse.beat();
        if let Some(driver) = self.shared.driver.lock().unwrap().take() {
            let _ = driver.join();
        }
    }
}

struct Shared {
    store: Store,
    meta: Meta,
    streams: usize,
    /// The downloader ledger's resolution; configuration, never a verdict.
    sample: Duration,
    state: Mutex<State>,
    /// Waiters and the driver wait here for objects to land, fail or be wanted.
    changed: Condvar,
    pulse: Pulse,
    stop: PullCancellation,
    /// Admissions hold it shared, a collection to make room exclusively.
    gate: RwLock<()>,
    /// One disk admission at a time.
    admitting: Mutex<()>,
    driver: Mutex<Option<std::thread::JoinHandle<()>>>,
    me: Weak<Shared>,
}

#[derive(Default)]
struct State {
    entries: HashMap<String, Entry>,
    /// The queued entries by (key, seq, digest): the next to open first.
    queue: BTreeSet<(u64, u64, String)>,
    origins: HashMap<Origin, Source>,
    tickets: HashMap<u64, Interest>,
    next_ticket: u64,
    seq: u64,
    /// Bytes and count of admitted objects not yet landed: what admission sets aside.
    reserved: u64,
    admitted: u64,
    driving: bool,
}

/// The queue holds `sha256` at its key while it is queued, and not otherwise.
fn requeue(entries: &mut HashMap<String, Entry>, queue: &mut BTreeSet<(u64, u64, String)>, sha256: &str) {
    let Some(entry) = entries.get_mut(sha256) else {
        return;
    };
    let at = (entry.phase == Phase::Queued).then(|| (entry.cell.key(), entry.seq));
    if entry.queued == at {
        return;
    }
    if let Some((key, seq)) = std::mem::replace(&mut entry.queued, at) {
        queue.remove(&(key, seq, sha256.to_string()));
    }
    if let Some((key, seq)) = at {
        queue.insert((key, seq, sha256.to_string()));
    }
}

impl State {
    fn requeue(&mut self, sha256: &str) {
        requeue(&mut self.entries, &mut self.queue, sha256);
    }

    fn refuse(&mut self, sha256: &str, refusal: Refusal) {
        if let Some(entry) = self.entries.get_mut(sha256) {
            entry.phase = Phase::Refused;
            entry.failures += 1;
            entry.refusal = Some(refusal);
        }
        self.requeue(sha256);
    }

    /// Every queued entry `origin` alone supplies ends refused.
    fn refuse_origin(&mut self, origin: &Origin, refusal: &Refusal) {
        let only: Vec<String> = (self.queue.iter())
            .filter(|(_, _, sha256)| self.entries[sha256].origins == [origin.clone()])
            .map(|(_, _, sha256)| sha256.clone())
            .collect();
        for sha256 in only {
            self.refuse(&sha256, refusal.clone());
        }
    }

    /// Every entry not landed ends refused: no ticket waits on a downloader that cannot run.
    fn refuse_unlanded(&mut self, refusal: &Refusal) {
        let unlanded: Vec<String> = (self.entries.iter())
            .filter(|(_, entry)| entry.phase != Phase::Refused)
            .map(|(sha256, _)| sha256.clone())
            .collect();
        for sha256 in unlanded {
            self.unreserve(&sha256);
            self.refuse(&sha256, refusal.clone());
        }
    }

    /// What `sha256` set aside on the disk is free again.
    fn unreserve(&mut self, sha256: &str) {
        if let Some(entry) = self.entries.get_mut(sha256).filter(|e| e.reserved) {
            entry.reserved = false;
            self.reserved -= entry.object.length;
            self.admitted -= 1;
        }
    }

    fn remove(&mut self, sha256: &str) -> Option<Entry> {
        self.unreserve(sha256);
        let entry = self.entries.remove(sha256)?;
        if let Some((key, seq)) = entry.queued {
            self.queue.remove(&(key, seq, sha256.to_string()));
        }
        Some(entry)
    }
}

/// One absent object some ticket wants.
struct Entry {
    object: ObjectRef,
    seq: u64,
    cell: Arc<Cell>,
    /// Each interested ticket and its key.
    interest: HashMap<u64, u64>,
    /// The origins its tickets want it from.
    origins: Vec<Origin>,
    phase: Phase,
    /// Admitted against the disk: its bytes are set aside until it ends.
    reserved: bool,
    /// Its place in the queue, while queued.
    queued: Option<(u64, u64)>,
    /// Times it was refused, and the last refusal: a waiter fails on a refusal after it began.
    failures: u64,
    refusal: Option<Refusal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queued,
    Open,
    Refused,
}

/// One ticket's interest.
struct Interest {
    origin: Origin,
    key: u64,
    /// Its objects and the key it wants each at.
    objects: HashMap<String, (ObjectRef, u64)>,
    holds: Vec<Hold>,
    fetched: u64,
    cached: u64,
}

/// One origin: its closure, the supply its fetches use, and how far its structure landed.
#[derive(Default)]
struct Source {
    tickets: usize,
    /// The first live ticket's.
    access: Option<Arc<dyn Access>>,
    closure: Option<Arc<transport::Closure>>,
    supply: Option<Arc<OriginSupply>>,
    resolving: bool,
    structure: Structure,
}

#[derive(Clone, Default)]
enum Structure {
    #[default]
    Manifest,
    Header(ObjectRef),
    Assets(Vec<ObjectRef>),
    Done,
    Refused(Refusal),
}

impl Structure {
    /// The structure objects still to land.
    fn pending(&self, manifest: &ObjectRef) -> Vec<ObjectRef> {
        match self {
            Structure::Manifest => vec![manifest.clone()],
            Structure::Header(header) => vec![header.clone()],
            Structure::Assets(assets) => assets.clone(),
            Structure::Done | Structure::Refused(_) => vec![],
        }
    }
}

static FETCHERS: Mutex<Vec<(PathBuf, Weak<Handle>)>> = Mutex::new(Vec::new());

impl Fetcher {
    /// This process's fetcher for `store`'s root, with the transport's full streams.
    pub fn of(store: &Store) -> Result<Fetcher> {
        let root = std::fs::canonicalize(store.root()).unwrap_or_else(|_| store.root().to_path_buf());
        let mut all = FETCHERS.lock().unwrap();
        all.retain(|(_, handle)| handle.strong_count() > 0);
        if let Some(handle) = all
            .iter()
            .find(|(at, _)| *at == root)
            .and_then(|(_, h)| h.upgrade())
        {
            return Ok(Fetcher(handle));
        }
        let sample = Duration::from_secs_f64(transport::SAMPLE_SECONDS);
        let fetcher = Fetcher::new(store, transport::PULL_STREAMS, sample)?;
        all.push((root, Arc::downgrade(&fetcher.0)));
        Ok(fetcher)
    }

    /// A fetcher of its own for `store`, with at most `streams` requests in flight and its
    /// downloader's ledger sampling at `sample`.
    pub fn new(store: &Store, streams: usize, sample: Duration) -> Result<Fetcher> {
        let meta = Meta::open(store)?;
        let shared = Arc::new_cyclic(|me| Shared {
            store: store.clone(),
            meta,
            streams: streams.clamp(1, transport::PULL_STREAMS),
            sample,
            state: Mutex::default(),
            changed: Condvar::new(),
            pulse: Pulse::default(),
            stop: PullCancellation::default(),
            gate: RwLock::default(),
            admitting: Mutex::default(),
            driver: Mutex::default(),
            me: me.clone(),
        });
        Ok(Fetcher(Arc::new(Handle { shared })))
    }

    pub fn store(&self) -> &Store {
        &self.0.shared.store
    }

    /// Interest in `objects` of `origin`, and in its structure, at `at`. Never blocks on the
    /// network: what is absent is queued, and the origin's closure resolved on its own thread.
    pub fn want(
        &self,
        origin: &Origin,
        access: Arc<dyn Access>,
        objects: &[ObjectRef],
        at: Priority,
    ) -> Ticket {
        let shared = &self.0.shared;
        let id = {
            let mut state = shared.state.lock().unwrap();
            state.next_ticket += 1;
            let id = state.next_ticket;
            state.tickets.insert(
                id,
                Interest {
                    origin: origin.clone(),
                    key: at.key(),
                    objects: HashMap::new(),
                    holds: vec![],
                    fetched: 0,
                    cached: 0,
                },
            );
            let source = state.origins.entry(origin.clone()).or_default();
            source.tickets += 1;
            source.access.get_or_insert(access);
            id
        };
        let ticket = Ticket {
            fetcher: self.clone(),
            id,
            origin: origin.clone(),
        };
        // Its last use, for the TTL of a checkpoint held in part.
        crate::origin::touch(&shared.store, &origin.manifest);
        let structure = shared.pending_structure(origin);
        shared.add(id, &[structure, objects.to_vec()].concat(), at.key());
        shared.advance(origin);
        ticket
    }

    pub(crate) fn seed(&self, origin: &Origin, closure: transport::Closure) {
        let mut state = self.0.shared.state.lock().unwrap();
        let source = state.origins.entry(origin.clone()).or_default();
        if source.closure.is_none() {
            source.closure = Some(Arc::new(closure));
        }
    }
}

/// Interest in some objects of one origin; dropping it releases the interest.
pub struct Ticket {
    fetcher: Fetcher,
    id: u64,
    origin: Origin,
}

impl Ticket {
    pub fn origin(&self) -> &Origin {
        &self.origin
    }

    /// Wait until `objects` are present, raising them to [`Priority::Now`] and adding any this
    /// ticket lacks. Empty: everything the ticket wants, its structure included. A refusal of
    /// one of them after the wait began fails it; so do `stop` and the fetcher's end.
    pub fn wait(&self, objects: &[ObjectRef], stop: &PullCancellation) -> Result<()> {
        let shared = &self.fetcher.0.shared;
        let whole = objects.is_empty();
        let objects = match whole {
            true => self.objects(),
            false => objects.to_vec(),
        };
        let since: HashMap<String, u64> = {
            let state = shared.state.lock().unwrap();
            (objects.iter())
                .map(|o| {
                    (
                        o.sha256.clone(),
                        state.entries.get(&o.sha256).map_or(0, |e| e.failures),
                    )
                })
                .collect()
        };
        shared.add(self.id, &objects, NOW);
        let mut state = shared.state.lock().unwrap();
        loop {
            let wanted: Vec<ObjectRef> = match whole {
                true => state
                    .tickets
                    .get(&self.id)
                    .map_or(vec![], |t| t.objects.values().map(|(o, _)| o.clone()).collect()),
                false => objects.clone(),
            };
            let mut absent = false;
            for object in &wanted {
                let Some(entry) = state.entries.get(&object.sha256) else {
                    continue;
                };
                if entry.failures > since.get(&object.sha256).copied().unwrap_or(0) {
                    let refusal = entry.refusal.clone().unwrap_or_else(|| Refusal {
                        code: Code::TRANSFER_FAILED,
                        detail: "refused".into(),
                    });
                    return Err(named(refusal, &object.id(), &self.origin));
                }
                absent = true;
            }
            let structure = &state
                .origins
                .get(&self.origin)
                .map(|s| s.structure.clone())
                .unwrap_or_default();
            if whole {
                if let Structure::Refused(refusal) = structure {
                    return Err(refusal.clone());
                }
                absent |= !matches!(structure, Structure::Done);
            }
            if !absent {
                return Ok(());
            }
            if stop.is_cancelled() {
                return refuse(Code::TRANSFER_FAILED, "the wait was cancelled by its caller");
            }
            if shared.stop.is_cancelled() {
                return refuse(Code::TRANSFER_FAILED, "the fetcher stopped");
            }
            // Resolution only: a caller's stop has no other way to be seen.
            state = shared.changed.wait_timeout(state, POLL).unwrap().0;
        }
    }

    /// Bytes present and in all, over what the ticket wants.
    pub fn progress(&self) -> (u64, u64) {
        let state = self.fetcher.0.shared.state.lock().unwrap();
        let Some(interest) = state.tickets.get(&self.id) else {
            return (0, 0);
        };
        interest
            .objects
            .values()
            .fold((0, 0), |(done, total), (object, _)| {
                let present = !state.entries.contains_key(&object.sha256);
                (
                    done + if present { object.length } else { 0 },
                    total + object.length,
                )
            })
    }

    /// Bytes of this ticket's objects that landed from the origin and from the repo cache.
    pub(crate) fn landed(&self) -> (u64, u64) {
        let state = self.fetcher.0.shared.state.lock().unwrap();
        state
            .tickets
            .get(&self.id)
            .map_or((0, 0), |t| (t.fetched, t.cached))
    }

    fn objects(&self) -> Vec<ObjectRef> {
        let state = self.fetcher.0.shared.state.lock().unwrap();
        state
            .tickets
            .get(&self.id)
            .map_or(vec![], |t| t.objects.values().map(|(o, _)| o.clone()).collect())
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.fetcher.0.shared.release(self.id);
    }
}

/// How often a waiter looks at its caller's stop: resolution only.
const POLL: Duration = Duration::from_millis(100);

/// `refusal`, naming the object and the origin it was fetched from.
fn named(refusal: Refusal, id: &str, origin: &Origin) -> Refusal {
    let what = match refusal.detail.contains(id) {
        true => String::new(),
        false => format!("{id} "),
    };
    Refusal {
        detail: format!("{what}from {}: {}", origin.evidence(), refusal.detail),
        ..refusal
    }
}

impl Shared {
    fn pending_structure(&self, origin: &Origin) -> Vec<ObjectRef> {
        let state = self.state.lock().unwrap();
        state
            .origins
            .get(origin)
            .map_or(vec![origin.manifest.clone()], |s| {
                s.structure.pending(&origin.manifest)
            })
    }

    /// `objects` join ticket `id` at `key` (raising any it holds at a later key): held from GC,
    /// and queued when absent. Starts the origin's resolution and the driver as needed.
    fn add(&self, id: u64, objects: &[ObjectRef], key: u64) {
        if objects.is_empty() {
            return;
        }
        // Held before presence is asked, so nothing present can be collected in between.
        let owner = crate::meta::owner(self.store.root());
        let hold = {
            let _admitting = owner.as_ref().map(|owner| owner.gate.read().unwrap());
            self.meta.read_hold(owner.as_ref(), objects)
        };
        let present: Vec<bool> = objects.iter().map(|o| self.present(o)).collect();
        let mut resolve = None;
        {
            let mut state = self.state.lock().unwrap();
            let state = &mut *state;
            let Some(interest) = state.tickets.get_mut(&id) else {
                drop(hold.map(|hold| hold.release(&self.meta)));
                return;
            };
            if let Ok(hold) = hold {
                interest.holds.push(hold);
            }
            let origin = interest.origin.clone();
            for (object, present) in objects.iter().zip(present) {
                let wanted = interest
                    .objects
                    .entry(object.sha256.clone())
                    .or_insert((object.clone(), key));
                wanted.1 = wanted.1.min(key);
                let key = wanted.1;
                if present && !state.entries.contains_key(&object.sha256) {
                    continue;
                }
                state.seq += 1;
                let seq = state.seq;
                let entry = state
                    .entries
                    .entry(object.sha256.clone())
                    .or_insert_with(|| Entry {
                        object: object.clone(),
                        seq,
                        cell: Cell::new(key),
                        interest: HashMap::new(),
                        origins: vec![],
                        phase: Phase::Queued,
                        reserved: false,
                        queued: None,
                        failures: 0,
                        refusal: None,
                    });
                let ticket_key = entry.interest.entry(id).or_insert(key);
                *ticket_key = (*ticket_key).min(key);
                if !entry.origins.contains(&origin) {
                    entry.origins.push(origin.clone());
                }
                if entry.phase == Phase::Refused {
                    entry.phase = Phase::Queued;
                    entry.cell = Cell::new(key);
                }
                entry
                    .cell
                    .set(entry.interest.values().copied().min().unwrap_or(key));
                requeue(&mut state.entries, &mut state.queue, &object.sha256);
            }
            let source = state.origins.entry(origin.clone()).or_default();
            if let Some(closure) = source.closure.clone() {
                if source.supply.is_none() {
                    if let Some(access) = source.access.clone() {
                        source.supply = Some(OriginSupply::new(&origin, closure, access, self.me.clone()));
                    }
                }
                self.contain(state, &origin);
            } else if !source.resolving && source.access.is_some() {
                let waiting = (state.queue.iter())
                    .any(|(_, _, sha256)| state.entries[sha256].origins.contains(&origin));
                let source = state.origins.get_mut(&origin).unwrap();
                if waiting {
                    source.resolving = true;
                    resolve = Some(origin.clone());
                }
            }
            if !state.driving && !state.queue.is_empty() {
                state.driving = true;
                if let Some(me) = self.me.upgrade() {
                    let spawned = std::thread::Builder::new()
                        .name("tensorfs-fetch".into())
                        .spawn(move || me.drive());
                    match spawned {
                        Ok(driver) => *self.driver.lock().unwrap() = Some(driver),
                        Err(error) => {
                            state.driving = false;
                            let detail = format!("no downloader could start: {error}");
                            state.refuse_unlanded(&Refusal { code: Code::TRANSFER_FAILED, detail });
                        }
                    }
                }
            }
        }
        if let (Some(origin), Some(me)) = (resolve, self.me.upgrade()) {
            let resolving = origin.clone();
            let spawned = std::thread::Builder::new().name("tensorfs-closure".into()).spawn(move || {
                // A resolution that panics ends what waits on it, as one that fails does.
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Arc::clone(&me).resolve(&resolving))).is_err() {
                    let mut state = me.state.lock().unwrap_or_else(PoisonError::into_inner);
                    if let Some(source) = state.origins.get_mut(&resolving) {
                        source.resolving = false;
                    }
                    let detail = "the closure resolution panicked".to_string();
                    state.refuse_origin(&resolving, &Refusal { code: Code::IO_FAILED, detail });
                    drop(state);
                    me.changed.notify_all();
                    me.pulse.beat();
                }
            });
            if let Err(error) = spawned {
                let mut state = self.state.lock().unwrap();
                state.origins.get_mut(&origin).unwrap().resolving = false;
                let detail = format!("no closure resolution could start: {error}");
                state.refuse_origin(&origin, &Refusal { code: Code::TRANSFER_FAILED, detail });
            }
        }
        self.changed.notify_all();
        self.pulse.beat();
    }

    fn present(&self, object: &ObjectRef) -> bool {
        match self.store.manifest_path(&object.sha256).is_file() {
            true => true,
            false => self.store.contains(&object.sha256),
        }
    }

    /// Queued objects of `origin` its closure does not name are refused.
    fn contain(&self, state: &mut State, origin: &Origin) {
        let Some(closure) = state.origins.get(origin).and_then(|s| s.closure.clone()) else {
            return;
        };
        let contained = |o: &ObjectRef| {
            *o == closure.manifest
                || closure
                    .objects
                    .binary_search_by(|c| c.sha256.cmp(&o.sha256))
                    .is_ok_and(|at| closure.objects[at] == *o)
        };
        let foreign: Vec<String> = (state.queue.iter())
            .map(|(_, _, sha256)| &state.entries[sha256])
            .filter(|e| e.origins == [origin.clone()] && !contained(&e.object))
            .map(|e| e.object.sha256.clone())
            .collect();
        for sha256 in foreign {
            let detail = format!("sha256:{sha256} is not a part of this checkpoint");
            state.refuse(
                &sha256,
                Refusal {
                    code: Code::NOT_CONTAINED,
                    detail,
                },
            );
        }
    }

    /// Resolve `origin`'s closure from its hub, on its own thread, and mint its first window.
    fn resolve(self: Arc<Self>, origin: &Origin) {
        let access = self
            .state
            .lock()
            .unwrap()
            .origins
            .get(origin)
            .and_then(|s| s.access.clone());
        let resolved = access.as_ref().map_or_else(
            || refuse(Code::CREDENTIAL_REQUIRED, "no live ticket names this origin"),
            |access| {
                let ledger = Ledger::new().with_cancellation(self.stop.clone());
                let policy = transport::hub_policy(&origin.hub, access.policy());
                let closure = transport::closure_all(
                    transport::shared_client(),
                    &origin.hub,
                    &origin.refspec,
                    &origin.lane,
                    access.credential(),
                    &policy,
                    Deadline::none(),
                    &ledger,
                )?;
                if closure.manifest != origin.manifest {
                    return refuse(
                        Code::NOT_CONTAINED,
                        format!(
                            "{} lane {:?} now names {}, not {}",
                            origin.refspec,
                            origin.lane,
                            closure.manifest.id(),
                            origin.manifest.id()
                        ),
                    );
                }
                Ok(closure)
            },
        );
        let supply = {
            let mut state = self.state.lock().unwrap();
            let state = &mut *state;
            let source = state.origins.get_mut(origin).unwrap();
            source.resolving = false;
            match resolved {
                Ok(closure) => {
                    let closure = Arc::new(closure);
                    source.closure = Some(Arc::clone(&closure));
                    let live = access.filter(|_| source.tickets > 0);
                    let supply =
                        live.map(|access| OriginSupply::new(origin, closure, access, self.me.clone()));
                    source.supply = supply.clone();
                    self.contain(state, origin);
                    supply
                }
                Err(refusal) => {
                    state.refuse_origin(origin, &refusal);
                    None
                }
            }
        };
        self.changed.notify_all();
        self.pulse.beat();
        if let Some(supply) = supply {
            let ledger = Ledger::new().with_cancellation(self.stop.clone());
            if let Ok(_minting) = supply.minting.acquire(&ledger, Deadline::none()) {
                let _ = supply.mint(None, &ledger);
            }
        }
    }

    /// The structure of `origin` as far as it is here: each stage's objects join its tickets;
    /// once header and assets are here, the origin is attested.
    fn advance(&self, origin: &Origin) {
        loop {
            let structure = {
                let state = self.state.lock().unwrap();
                match state.origins.get(origin) {
                    Some(source) => source.structure.clone(),
                    None => return,
                }
            };
            let pending = structure.pending(&origin.manifest);
            if matches!(structure, Structure::Done | Structure::Refused(_))
                || !pending.iter().all(|o| self.present(o))
            {
                return;
            }
            let next = match &structure {
                Structure::Manifest => self
                    .store
                    .read_manifest(&origin.manifest)
                    .map(|m| match m.header() {
                        Some(header) => Structure::Header(header.clone()),
                        None => Structure::Done,
                    }),
                Structure::Header(header) => {
                    crate::checkpoint::load_header(&self.store, header).map(|header| {
                        let mut assets: Vec<ObjectRef> = header
                            .assets
                            .iter()
                            .flat_map(|(_, a)| a.segments.clone())
                            .collect();
                        assets.sort_by(|a, b| a.sha256.cmp(&b.sha256));
                        assets.dedup();
                        Structure::Assets(assets)
                    })
                }
                _ => Ok(Structure::Done),
            };
            let next = match next.and_then(|next| match next {
                Structure::Done => self.attest(origin).map(|()| next),
                next => Ok(next),
            }) {
                Ok(next) => next,
                Err(refusal) => Structure::Refused(refusal),
            };
            let add: Vec<(u64, u64)> = {
                let mut state = self.state.lock().unwrap();
                let Some(source) = state.origins.get_mut(origin) else {
                    return;
                };
                source.structure = next.clone();
                state
                    .tickets
                    .iter()
                    .filter(|(_, t)| t.origin == *origin)
                    .map(|(id, t)| (*id, t.key))
                    .collect()
            };
            let pending = next.pending(&origin.manifest);
            for (id, key) in add {
                self.add(id, &pending, key);
            }
            self.changed.notify_all();
        }
    }

    /// Record `origin`'s hub as holding what of its checkpoint is absent here, unless a
    /// repository here already records the checkpoint whole.
    fn attest(&self, origin: &Origin) -> Result<()> {
        let model = origin.refspec.split('@').next().unwrap_or_default();
        let recorded = model
            .split_once('/')
            .and_then(|(org, name)| crate::repository::RepositoryName::new(org, name).ok())
            .and_then(|repo| std::fs::read(self.store.repository_path(&repo)).ok())
            .and_then(|bytes| crate::repository::Repository::parse(&bytes).ok())
            .is_some_and(|repo| repo.checkpoints.iter().any(|c| c.manifest == origin.manifest));
        if recorded {
            return Ok(());
        }
        crate::origin::attest(&self.store, &origin.manifest, origin)
    }

    /// Ticket `id` ends: its interest goes, objects nothing else wants leave the queue (or the
    /// wire), and its holds are released.
    fn release(&self, id: u64) {
        let holds = {
            let mut state = self.state.lock().unwrap();
            let state = &mut *state;
            let Some(interest) = state.tickets.remove(&id) else {
                return;
            };
            for sha256 in interest.objects.keys() {
                let Some(entry) = state.entries.get_mut(sha256) else {
                    continue;
                };
                entry.interest.remove(&id);
                if let Some(key) = entry.interest.values().copied().min() {
                    entry.cell.set(key);
                    state.requeue(sha256);
                    continue;
                }
                if entry.phase == Phase::Open {
                    entry.cell.leave();
                }
                state.remove(sha256);
            }
            if let Some(source) = state.origins.get_mut(&interest.origin) {
                source.tickets -= 1;
                if source.tickets == 0 {
                    // The credential lives no longer than the tickets that brought it.
                    source.access = None;
                    source.supply = None;
                }
            }
            interest.holds
        };
        for hold in holds {
            let _ = hold.release(&self.meta);
        }
        self.changed.notify_all();
        self.pulse.beat();
    }

    /// The most urgent queued entry whose origin can supply it.
    fn best<'s>(&self, state: &'s State) -> Option<&'s (u64, u64, String)> {
        let supplied = |o: &Origin| state.origins.get(o).is_some_and(|s| s.supply.is_some());
        (state.queue.iter()).find(|(_, _, sha256)| state.entries[sha256].origins.iter().any(supplied))
    }

    /// The downloader, while anything is queued, until the fetcher ends.
    fn drive(self: Arc<Self>) {
        loop {
            {
                let mut state = self.state.lock().unwrap();
                while self.best(&state).is_none() {
                    if self.stop.is_cancelled() {
                        state.driving = false;
                        return;
                    }
                    state = self.changed.wait_timeout(state, POLL).unwrap().0;
                }
            }
            // A run that fails or panics (a thread the container refused) ends what it left
            // unlanded with its reason: its tickets answer, and the driver never spins on it.
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run()));
            if self.stop.is_cancelled() {
                self.state.lock().unwrap_or_else(PoisonError::into_inner).driving = false;
                return;
            }
            let refusal = match ran {
                Ok(Ok(())) => continue,
                Ok(Err(refusal)) => refusal,
                Err(_) => Refusal { code: Code::IO_FAILED, detail: "the downloader panicked".into() },
            };
            eprintln!("tensorfs fetch: {refusal}");
            self.state.lock().unwrap_or_else(PoisonError::into_inner).refuse_unlanded(&refusal);
            self.changed.notify_all();
            self.pulse.beat();
        }
    }

    /// One downloader run: until nothing is queued or open.
    fn run(&self) -> Result<()> {
        crate::descriptors::raise();
        let streams = self.streams;
        let ledger = Ledger::with_resolution(self.sample).with_cancellation(self.stop.clone());
        ledger.transfer_begins();
        let epoch = crate::unsynced::Epoch::begin(&self.store)?;
        let store = self.store.syncing(epoch.clone());
        let ran = Download {
            store: &store,
            feed: self,
            chunk: CHUNK_BYTES,
            ranged: true,
            deadline: Deadline::none(),
            ledger: &ledger,
            keep: None,
        }
        .run(streams, None);
        if let Some(epoch) = epoch {
            epoch.finish()?;
        }
        ran.map(drop)
    }

    /// Whether `bytes` more fit the disk and the Store's budget, beyond its reserve.
    fn short(&self, bytes: u64, objects: u64) -> Result<Option<u64>> {
        let need = crate::ensure::Space::of(bytes, objects);
        let disk = crate::ensure::Disk::measure(self.store.root())?;
        let free = crate::ensure::free(&disk, crate::ensure::Space::default());
        let mut short = (need.bytes > free.bytes || need.inodes > free.inodes)
            .then(|| need.bytes.saturating_sub(free.bytes).max(1));
        if let Some(budget) = self.store.disk_budget() {
            let occupancy = self.store.occupancy()?;
            let over = (occupancy + bytes).saturating_sub(budget);
            if over > 0 {
                short = Some(short.unwrap_or(0).max(over));
            }
        }
        Ok(short)
    }

    /// Admit `object`, which is open, against the disk beside every object admitted and not
    /// yet landed, and set its bytes aside; relieve a short disk by GC first. One at a time,
    /// so what one is refused never counts against another.
    fn admit(&self, object: &ObjectRef, cell: &Cell) -> Result<()> {
        let _one = self.admitting.lock().unwrap();
        let (reserved, admitted) = {
            let state = self.state.lock().unwrap();
            (state.reserved, state.admitted)
        };
        let need = || self.short(reserved + object.length, admitted + 1);
        if let Some(short) = need()? {
            let protected = self.protected();
            let collected = {
                let _collecting = self.gate.write().unwrap();
                crate::gc::collect_cached_for(self.store.root(), &protected, short)
            };
            if let Some(short) = need()? {
                let why = collected
                    .err()
                    .map_or(String::new(), |r| format!("; GC could not run: {r}"));
                return refuse(
                    Code::CAPACITY_EXHAUSTED,
                    format!(
                        "{} needs {} B beside {reserved} B admitted; the disk under {} is {short} B \
                         short after GC{why}",
                        object.id(),
                        object.length,
                        self.store.root().display(),
                    ),
                );
            }
        }
        let mut state = self.state.lock().unwrap();
        let state = &mut *state;
        if let Some(entry) = state.entries.get_mut(&object.sha256) {
            if std::ptr::eq(Arc::as_ptr(&entry.cell), cell) && entry.phase == Phase::Open && !entry.reserved {
                entry.reserved = true;
                state.reserved += object.length;
                state.admitted += 1;
            }
        }
        Ok(())
    }

    /// The manifests of every origin a live ticket names: GC keeps their present objects.
    fn protected(&self) -> Vec<String> {
        let state = self.state.lock().unwrap();
        state
            .origins
            .iter()
            .filter(|(_, s)| s.tickets > 0)
            .map(|(o, _)| o.manifest.sha256.clone())
            .collect()
    }
}

impl<'a> Feed<'a> for Shared {
    fn open(&self, before: u64) -> Result<Option<Opening<'a>>> {
        loop {
            let (object, cell, supply) = {
                let mut state = self.state.lock().unwrap();
                let Some((key, _, sha256)) = self.best(&state).cloned() else {
                    return Ok(None);
                };
                if key >= before {
                    return Ok(None);
                }
                let state = &mut *state;
                let entry = state.entries.get_mut(&sha256).unwrap();
                let supply = (entry.origins.iter())
                    .find_map(|o| state.origins.get(o).and_then(|s| s.supply.clone()))
                    .expect("the best entry has a supply");
                entry.phase = Phase::Open;
                let opened = (entry.object.clone(), Arc::clone(&entry.cell), supply);
                state.requeue(&sha256);
                opened
            };
            if self.present(&object) {
                self.landed(&object, &cell, None);
                continue;
            }
            let manifest = (supply.origin.manifest == object).then_some(&object);
            let granted = self.admit(&object, &cell).and_then(|()| {
                let (plan, _) = match manifest {
                    Some(manifest) => {
                        FetchPlan::of(&self.store, "fetch", manifest, std::slice::from_ref(manifest))?
                    }
                    None => FetchPlan::of_objects(&self.store, "fetch", std::slice::from_ref(&object))?,
                };
                match plan.wanted.is_empty() {
                    true => Ok(None),
                    false => DeliveryGrant::mint(&plan, &object).map(Some),
                }
            });
            match granted {
                Ok(Some(grant)) => {
                    let supply: Arc<dyn Supply + 'a> = supply;
                    return Ok(Some(Opening {
                        at: 0,
                        grant,
                        supply,
                        cell,
                    }));
                }
                Ok(None) => self.landed(&object, &cell, None),
                Err(refusal) => self.ended(&object, &cell, Err(refusal)),
            }
        }
    }

    fn waiting(&self) -> Option<u64> {
        self.best(&self.state.lock().unwrap()).map(|(key, _, _)| *key)
    }

    fn ended(&self, object: &ObjectRef, cell: &Cell, outcome: Result<(ObjectSource, u64)>) {
        match outcome {
            Ok((door, _)) => self.landed(object, cell, Some(door)),
            Err(refusal) => {
                {
                    let mut state = self.state.lock().unwrap();
                    let state = &mut *state;
                    let Some(entry) = state.entries.get_mut(&object.sha256) else {
                        return;
                    };
                    if !std::ptr::eq(Arc::as_ptr(&entry.cell), cell) {
                        return;
                    }
                    state.unreserve(&object.sha256);
                    state.refuse(&object.sha256, refusal);
                }
                self.changed.notify_all();
                self.pulse.beat();
            }
        }
    }

    fn lasting(&self) -> bool {
        true
    }

    fn pulse(&self) -> &Pulse {
        &self.pulse
    }

    fn gate(&self) -> Option<&RwLock<()>> {
        Some(&self.gate)
    }
}

impl Shared {
    /// `object` is here: its entry ends, its tickets are credited, and any origin whose
    /// structure waited on it moves on.
    fn landed(&self, object: &ObjectRef, cell: &Cell, door: Option<ObjectSource>) {
        let origins: Vec<Origin> = {
            let mut state = self.state.lock().unwrap();
            let state = &mut *state;
            if let Some(entry) = state.entries.get(&object.sha256) {
                if std::ptr::eq(Arc::as_ptr(&entry.cell), cell) {
                    let entry = state.remove(&object.sha256).unwrap();
                    for id in entry.interest.keys() {
                        if let Some(interest) = state.tickets.get_mut(id) {
                            match door {
                                Some(ObjectSource::Origin) => interest.fetched += object.length,
                                Some(ObjectSource::Cache) => interest.cached += object.length,
                                None => {}
                            }
                        }
                    }
                }
            }
            (state.origins.iter())
                .filter(|(origin, source)| source.structure.pending(&origin.manifest).contains(object))
                .map(|(origin, _)| origin.clone())
                .collect()
        };
        self.changed.notify_all();
        self.pulse.beat();
        for origin in origins {
            self.advance(&origin);
        }
    }
}

/// One origin's URLs: its closure's presign window minted ahead of the downloader, in the
/// order the fetcher will open its objects.
struct OriginSupply {
    origin: Origin,
    closure: Arc<transport::Closure>,
    access: Arc<dyn Access>,
    policy: SourcePolicy,
    leases: Mutex<HashMap<String, Lease>>,
    minting: Minting,
    shared: Weak<Shared>,
}

impl OriginSupply {
    fn new(
        origin: &Origin,
        closure: Arc<transport::Closure>,
        access: Arc<dyn Access>,
        shared: Weak<Shared>,
    ) -> Arc<OriginSupply> {
        Arc::new(OriginSupply {
            origin: origin.clone(),
            policy: transport::hub_policy(&origin.hub, access.policy()),
            closure,
            access,
            leases: Mutex::default(),
            minting: Minting::default(),
            shared,
        })
    }

    fn fresh(&self, sha256: &str) -> Option<String> {
        let leases = self.leases.lock().unwrap();
        leases
            .get(sha256)
            .filter(|lease| !lease.spent())
            .map(|lease| lease.url.clone())
    }

    /// The objects of this origin the fetcher will open next, most urgent first, that hold
    /// no fresh URL; and how many waiting ones do.
    /// `open`: those already opened first, which need theirs now.
    fn upcoming(&self, open: bool) -> (Vec<String>, usize) {
        let Some(shared) = self.shared.upgrade() else {
            return (vec![], 0);
        };
        let state = shared.state.lock().unwrap();
        let leases = self.leases.lock().unwrap();
        let (mut unminted, mut fresh) = (vec![], 0);
        let opened = (state.entries.values())
            .filter(|e| open && e.phase == Phase::Open)
            .map(|e| &e.object.sha256);
        for sha256 in opened.chain(state.queue.iter().map(|(_, _, sha256)| sha256)) {
            if state.entries[sha256].origins.first() != Some(&self.origin) {
                continue;
            }
            match leases.get(sha256).is_some_and(|lease| !lease.spent()) {
                true => fresh += 1,
                false => unminted.push(sha256.clone()),
            }
        }
        (unminted, fresh)
    }

    /// Mint a window: `first`, then what the fetcher opens next.
    fn mint(&self, first: Option<&ObjectRef>, ledger: &Ledger) -> Result<()> {
        let window = self.closure.presign_max_digests.max(1);
        let mut digests: Vec<String> = first
            .filter(|o| self.fresh(&o.sha256).is_none())
            .map(|o| o.sha256.clone())
            .into_iter()
            .collect();
        for sha256 in self.upcoming(true).0 {
            if digests.len() >= window {
                break;
            }
            if !digests.contains(&sha256) {
                digests.push(sha256);
            }
        }
        if digests.is_empty() {
            return Ok(());
        }
        digests.sort();
        let answer = transport::presign(
            transport::shared_client(),
            &self.origin.hub,
            &digests,
            self.access.credential(),
            &self.policy,
            Deadline::none(),
            ledger,
        )?;
        let minted = std::time::Instant::now();
        let mut leases = self.leases.lock().unwrap();
        for (digest, url) in answer.urls {
            leases.insert(
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

    /// When fewer than half a window of this origin's waiting objects hold fresh URLs, the
    /// asker that sees it mints the next window, unless another already is.
    fn ahead(&self, ledger: &Ledger) {
        let (unminted, fresh) = self.upcoming(false);
        if unminted.is_empty() || fresh * 2 >= self.closure.presign_max_digests.max(1) {
            return;
        }
        if let Some(_minting) = self.minting.try_acquire() {
            let _ = self.mint(None, ledger);
        }
    }
}

impl Supply for OriginSupply {
    fn url(&self, _at: usize, object: &ObjectRef, stale: bool, ledger: &Ledger) -> Result<String> {
        ledger.check_running()?;
        // A refused URL is minted afresh whatever its declared life: the origin's clock rules.
        if stale {
            self.leases.lock().unwrap().remove(&object.sha256);
        }
        if let Some(url) = self.fresh(&object.sha256) {
            self.ahead(ledger);
            return Ok(url);
        }
        let _minting = self.minting.acquire(ledger, Deadline::none())?;
        if let Some(url) = self.fresh(&object.sha256) {
            return Ok(url);
        }
        self.mint(Some(object), ledger)?;
        self.fresh(&object.sha256).ok_or_else(|| Refusal {
            code: Code::HUB_REFUSED,
            detail: format!(
                "{}: the mint answered but the URL was unusable on arrival",
                object.id()
            ),
        })
    }
    fn policy(&self) -> &SourcePolicy {
        &self.policy
    }
    fn credential(&self) -> &dyn CredentialProvider {
        self.access.credential()
    }
}
