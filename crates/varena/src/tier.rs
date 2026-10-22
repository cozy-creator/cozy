//! The machine's RAM tier: one host layout per layout digest, a ready word per region, mapped
//! read-only (and page-locked, region by region) by executors, held by whichever participant
//! binds the machine's tier name (TensorD when present). An executor fills a region it misses
//! through `io::read` under the region's write lock, and copies from a ready region under its
//! read lock, so a region in flight is locked. Under pressure (an event, or a headroom check
//! before a fill) [`host_decide`] picks what goes: orphaned layouts (least recently used), then
//! layouts with no declared use, then the furthest declared; never a region in flight. Each
//! executor mapping a chosen layout is asked first, through its row's ask thread, to unpin and
//! unmap it, so the pages really return when the tier punches them out.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use platform::handles::Owned;
use serde::{Deserialize, Serialize};

use crate::ledger;
use crate::os::{self, Map, Stream};
use crate::{Error, Result};

/// Where pressure comes from.
pub enum Pressure {
    /// The host's PSI trigger and cgroup events.
    Host(platform::contain::host_pressure::Pressure),
    /// A descriptor that turns readable on pressure (an eventfd).
    Events(Owned),
    /// No event source: only the headroom check at every host decision.
    Headroom,
}

const NAME: &str = "tier";

/// What the host must keep free for the calls running on it: their measured host growth still
/// to come (each one's recorded high-water less its process's anonymous bytes now). `None`: a
/// running call is unmeasured, and the tier fills nothing beside it. TensorD sums its GPUs'
/// ([`crate::Ledger::host_need`]); an executor serving the tier itself, its own GPU's.
pub type Need = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

// The live OS source in production; deterministic headroom in isolated service tests.
type Budget = Arc<dyn Fn() -> i128 + Send + Sync>;

/// TensorD's need source: what the requests running on its GPUs keep free, summed (`None`
/// while any is unmeasured).
pub fn need(ledgers: &[&crate::Ledger]) -> Need {
    let tables: Vec<Arc<ledger::Shared>> = ledgers.iter().map(|l| l.table()).collect();
    Arc::new(move || tables.iter().map(|t| ledger::host_need(t)).sum())
}

/// Discretionary headroom includes the shipped measured non-tier growth reserve. This is
/// observed demand, not a fixed floor; shmem growth from the tier itself is excluded.
fn measured_budget() -> Budget {
    let growth = Mutex::new(platform::contain::host_pressure::Growth::default());
    Arc::new(move || {
        let mut growth = growth.lock();
        let host = platform::contain::host_memory::read();
        let reserved = growth.observe(host);
        i128::from(host.available) - i128::from(reserved)
    })
}

// One named claim file for this tier's host-pressure domain, shared by every GPU and
// surviving service replacements. Each live client owns a read lock on one bytes record;
// the metadata lock serializes scan, budget check and new lock/record publication.
const CLAIMS: &str = "tier-claims";
const CLAIM_MAGIC: u64 = u64::from_le_bytes(*b"vhrsv001");
const CLAIM_WORD: u64 = 8;

struct Claims {
    fd: Owned,
    local: Mutex<()>,
}

struct ClaimLock<'a> {
    claims: &'a Claims,
    _local: parking_lot::MutexGuard<'a, ()>,
}

impl Drop for ClaimLock<'_> {
    fn drop(&mut self) { let _ = os::unlock(&self.claims.fd, 0, CLAIM_WORD); }
}

enum Admission {
    Granted(Owned),
    Short { bytes: u64, headroom: u64 },
    Unmeasured,
}

fn claim_word(map: &Map, offset: u64) -> &AtomicU64 {
    // SAFETY: callers walk aligned complete words inside this mapping under the metadata lock.
    unsafe { &*(map.base.add(offset as usize).cast::<AtomicU64>()) }
}

impl Claims {
    fn open() -> Result<Self> {
        let claims = Self { fd: os::shared(CLAIMS, true)?, local: Mutex::new(()) };
        os::own(&claims.fd)?;
        {
            let _lock = claims.lock()?;
            let size = os::size(&claims.fd)?;
            if size == 0 { os::resize(&claims.fd, platform_page())?; }
            let size = os::size(&claims.fd)?;
            if size < platform_page() || !size.is_multiple_of(CLAIM_WORD) {
                return Err(Error::Closed("invalid host-reservation claim table".into()));
            }
            let map = os::map(&claims.fd, size as usize, true)?;
            let magic = claim_word(&map, 0);
            if magic.load(Acquire) == 0 && size == platform_page() {
                // Complete an interrupted first creation only when no client owns a row.
                for offset in (CLAIM_WORD..size).step_by(CLAIM_WORD as usize) {
                    if os::held(&claims.fd, offset, CLAIM_WORD)? {
                        return Err(Error::Closed("uninitialized host claim table has live owners".into()));
                    }
                }
                magic.store(CLAIM_MAGIC, Release);
            } else if magic.load(Acquire) != CLAIM_MAGIC {
                return Err(Error::Closed("incompatible host-reservation claim table".into()));
            }
        }
        Ok(claims)
    }

    fn lock(&self) -> Result<ClaimLock<'_>> {
        let local = self.local.lock();
        os::lock(&self.fd, 0, CLAIM_WORD, true)?;
        Ok(ClaimLock { claims: self, _local: local })
    }

    fn view(&self) -> Result<Map> {
        let size = os::size(&self.fd)?;
        if size < platform_page() || !size.is_multiple_of(CLAIM_WORD) {
            return Err(Error::Closed("invalid host-reservation claim table".into()));
        }
        let map = os::map(&self.fd, size as usize, true)?;
        if claim_word(&map, 0).load(Acquire) != CLAIM_MAGIC {
            return Err(Error::Closed("incompatible host-reservation claim table".into()));
        }
        Ok(map)
    }

    fn scan(&self, map: &Map) -> Result<(u64, Option<u64>)> {
        let (mut live, mut vacant) = (0u64, None);
        for offset in (CLAIM_WORD..map.len as u64).step_by(CLAIM_WORD as usize) {
            if os::held(&self.fd, offset, CLAIM_WORD)? {
                live = live.saturating_add(claim_word(map, offset).load(Acquire));
            } else if vacant.is_none() {
                vacant = Some(offset);
            }
        }
        Ok((live, vacant))
    }

    fn live_bytes(&self) -> Result<u64> {
        let _lock = self.lock()?;
        self.scan(&self.view()?).map(|(live, _)| live)
    }

    fn admit(&self, need: &Need, bytes: u64, budget: &Budget) -> Result<Admission> {
        let _lock = self.lock()?;
        let Some(keep) = need() else { return Ok(Admission::Unmeasured) };
        let mut map = self.view()?;
        let (pending, vacant) = self.scan(&map)?;
        let want = keep.saturating_add(pending).saturating_add(bytes);
        let mut room = budget();
        let mut metadata = if vacant.is_none() { platform_page() } else { 0 };
        if room >= i128::from(want) + i128::from(metadata) {
            let offset = if let Some(offset) = vacant { offset } else {
                let offset = map.len as u64;
                let size = offset.checked_add(platform_page()).ok_or_else(|| Error::Internal("host claim table overflow".into()))?;
                os::resize(&self.fd, size)?;
                map = os::map(&self.fd, size as usize, true)?;
                // Fault the new metadata page before re-reading actual headroom.
                claim_word(&map, offset).store(0, Release);
                metadata = 0; // This page is now in the observed OS budget, not future bytes.
                room = budget();
                offset
            };
            if room >= i128::from(want) {
                // Read-only rights are enough for a lifetime lock. The client receives no
                // write access to another client's claim, and no explicit stale-ID free exists.
                let lease = os::reopen(&self.fd, false)?;
                if !os::share(&lease, offset, CLAIM_WORD, false)? {
                    return Err(Error::Internal("host claim lock refused".into()));
                }
                claim_word(&map, offset).store(bytes, Release);
                return Ok(Admission::Granted(lease));
            }
        }
        let deficit = u64::try_from((-room).max(0)).unwrap_or(u64::MAX);
        Ok(Admission::Short { bytes: want.saturating_add(metadata).saturating_add(deficit), headroom: u64::try_from(room).unwrap_or(0) })
    }
}

/// Atomically admit a fill into measured surplus. The record lock already exists before
/// its fd is sent; queued SCM_RIGHTS keeps it alive even if the service dies before receipt.
fn admit(state: &Shared, need: &Need, bytes: u64, budget: &Budget, claims: &Claims) -> Result<Option<Owned>> {
    for attempt in 0..2 {
        match claims.admit(need, bytes, budget)? {
            Admission::Granted(fd) => return Ok(Some(fd)),
            Admission::Unmeasured => return Ok(None),
            Admission::Short { bytes, headroom } if attempt == 0 => { relieve(state, HostTrigger::Room { bytes, headroom }); }
            Admission::Short { .. } => return Ok(None),
        }
    }
    Ok(None)
}

/// A pressure event only says look. Preserve signed growth deficits and every still-live
/// future claim, including those granted by an earlier service holder.
fn answer(state: &Shared, need: &Need, budget: &Budget, claims: &Claims) {
    while budget() < i128::from(need().unwrap_or(u64::MAX)).saturating_add(i128::from(claims.live_bytes().unwrap_or(u64::MAX))) {
        if relieve(state, HostTrigger::Pressure) == 0 { return; }
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum Call {
    /// Regions as (offset, span) in the layout; `ask` names the opener's ask socket.
    Open { digest: String, regions: Vec<(u64, u64)>, ask: String },
    /// A layout this user holds, handed (with its file) to a tier that does not know it (a
    /// restarted one): its ready regions are the tier's again.
    Adopt { digest: String, regions: Vec<(u64, u64)>, ask: String },
    /// This process's next use of the layout, in steps; `None`: no declared use.
    Use { digest: String, distance: Option<u32> },
    Close { digest: String },
    /// Make `bytes` of host headroom before a fill, giving back in order.
    Room { bytes: u64 },
    /// Every layout's resident bytes, together.
    Resident,
}

#[derive(Debug, Serialize, Deserialize)]
enum Reply {
    Done,
    /// The layout's data starts at this offset of the file handed with the reply.
    Opened { data: u64 },
    Room { ok: bool },
    Resident { bytes: u64 },
}

struct Layout {
    fd: Owned,
    words: Map,
    regions: Vec<(u64, u64)>,
    data: u64,
    /// Connection → its declared distance and its ask socket.
    users: HashMap<u64, (Option<u32>, String)>,
    used: u64,
    /// A release asked of its users, punched when they have all answered.
    pending: Option<Pending>,
}

struct Pending {
    regions: Vec<u32>,
    forget: bool,
    waiting: HashSet<u64>,
}

impl Layout {
    fn ready(&self, r: usize) -> Option<&AtomicU32> {
        (r * 4 + 4 <= self.words.len).then(|| {
            // SAFETY: inside the words page, aligned to 4.
            unsafe { &*(self.words.base.add(r * 4) as *const AtomicU32) }
        })
    }

    fn holds(&self) -> bool {
        (0..self.regions.len()).any(|r| self.ready(r).is_some_and(|w| w.load(Acquire) == 1))
    }

    /// Its facts for [`host_decide`]: a ready region another description has locked is in
    /// flight (a copy from it, or a fill into it).
    fn facts(&self, digest: &str) -> HostLayout {
        let regions = self
            .regions
            .iter()
            .enumerate()
            .map(|(r, (off, span))| {
                let ready = self.ready(r).is_some_and(|w| w.load(Acquire) == 1);
                // A layout already being asked back is in flight whole: it is not chosen twice.
                let in_flight = ready && (self.pending.is_some() || os::held(&self.fd, self.data + off, *span).unwrap_or(true));
                (*span, ready, in_flight)
            })
            .collect();
        HostLayout { digest: digest.to_string(), regions, users: self.users.values().map(|(d, _)| *d).collect(), used: self.used }
    }

    /// A failed or dead filler may leave physical pages behind ready=0. Reclaim only
    /// ranges no operation locks; active and quarantined transfers keep their OFD leases.
    fn discard_unready(&self) -> u64 {
        let before = os::allocated(&self.fd);
        for r in 0..self.regions.len() {
            if self.ready(r).is_some_and(|w| w.load(Acquire) != 1) {
                self.release(r as u32);
            }
        }
        before.saturating_sub(os::allocated(&self.fd))
    }

    /// Gives back region `r` unless a copy took it meanwhile.
    fn release(&self, r: u32) -> u64 {
        let Some(&(off, span)) = self.regions.get(r as usize) else { return 0 };
        let Some(word) = self.ready(r as usize) else { return 0 };
        if !matches!(os::lock(&self.fd, self.data + off, span, false), Ok(true)) {
            return 0;
        }
        word.store(0, Release);
        let _ = os::punch(&self.fd, self.data + off, span);
        let _ = os::unlock(&self.fd, self.data + off, span);
        span
    }
}

/// A layout as [`host_decide`] reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostLayout {
    pub digest: String,
    /// Per region: (bytes, ready, in flight).
    pub regions: Vec<(u64, bool, bool)>,
    /// Each user's next declared use, in steps; `None`: no declared use.
    pub users: Vec<Option<u32>>,
    /// When it was last used (a logical clock).
    pub used: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostTrigger {
    /// A pressure event: give back one rung (one layout's regions not in flight).
    Pressure,
    /// A fill needs this many bytes of host headroom beyond `headroom`.
    Room { bytes: u64, headroom: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostAction {
    Release { digest: String, region: u32 },
    /// An orphaned layout with nothing left: its file goes.
    Forget(String),
}

/// The tier's one decision procedure, pure: orphaned layouts (least recently used first), then
/// layouts with no declared use, then the furthest declared; never a region in flight. TensorFS
/// owns its staging buffers, so the tier has no staging rung of its own.
pub fn host_decide(layouts: &[HostLayout], trigger: HostTrigger) -> Vec<HostAction> {
    let rank = |l: &HostLayout| -> (u8, i64) {
        if l.users.is_empty() {
            return (0, l.used as i64);
        }
        match l.users.iter().filter_map(|d| *d).min() {
            None => (1, l.used as i64),
            Some(d) => (2, -i64::from(d)),
        }
    };
    let mut order: Vec<&HostLayout> = layouts.iter().filter(|l| l.regions.iter().any(|(_, ready, _)| *ready)).collect();
    order.sort_by(|a, b| rank(a).cmp(&rank(b)).then(a.digest.cmp(&b.digest)));
    let free = |l: &HostLayout| l.regions.iter().enumerate().rev().filter(|(_, (_, ready, flight))| *ready && !flight).map(|(r, (b, ..))| (r as u32, *b)).collect::<Vec<_>>();
    let mut out = vec![];
    match trigger {
        HostTrigger::Pressure => {
            if let Some(l) = order.into_iter().find(|l| !free(l).is_empty()) {
                let regions = free(l);
                out.extend(regions.iter().map(|(r, _)| HostAction::Release { digest: l.digest.clone(), region: *r }));
                if l.users.is_empty() && regions.len() == l.regions.iter().filter(|(_, ready, _)| *ready).count() {
                    out.push(HostAction::Forget(l.digest.clone()));
                }
            }
        }
        HostTrigger::Room { bytes, headroom } => {
            let mut have = headroom;
            for l in order {
                for (r, b) in free(l) {
                    if have >= bytes {
                        return out;
                    }
                    out.push(HostAction::Release { digest: l.digest.clone(), region: r });
                    have += b;
                }
            }
        }
    }
    out
}

#[derive(Default)]
struct State {
    layouts: HashMap<String, Layout>,
    clock: u64,
    releases: u64,
    /// Bytes executors reported unpinned and unmapped when asked.
    acked: u64,
    /// The host-short event's signalling end: nothing is left to give back.
    short: Option<Owned>,
    /// Its users' open connections: a tier that stops serving closes them.
    conns: HashMap<u64, Stream>,
}

type Shared = Arc<(Mutex<State>, Condvar)>;

/// Applies the host decision for `trigger` without waiting on anyone: a chosen layout nobody
/// maps is punched out now; one that executors map is asked back from each of them (on a thread
/// per ask), and punched once every one has answered or died. Nothing left to give back raises
/// the host-short event. Returns the bytes given back now.
fn relieve(state: &Shared, trigger: HostTrigger) -> u64 {
    let (lock, cv) = &**state;
    let mut s = lock.lock();
    let discarded: u64 = s.layouts.values().map(Layout::discard_unready).sum();
    let trigger = match trigger {
        HostTrigger::Pressure if discarded > 0 => {
            s.releases += 1;
            drop(s);
            cv.notify_all();
            return discarded;
        }
        HostTrigger::Room { bytes, headroom } => HostTrigger::Room { bytes, headroom: headroom.saturating_add(discarded) },
        other => other,
    };
    let facts: Vec<HostLayout> = s.layouts.iter().map(|(d, l)| l.facts(d)).collect();
    let actions = host_decide(&facts, trigger);
    if actions.is_empty() {
        // Under pressure with nothing left to give (a layout being asked back of a stalled user
        // waits for TensorD to end it): the host is short past the tier. A staging Room is no
        // pressure.
        if trigger == HostTrigger::Pressure {
            if let Some(w) = &s.short {
                os::signal(w);
            }
        }
        if discarded > 0 { s.releases += 1; }
        drop(s);
        cv.notify_all();
        return discarded;
    }
    let mut chosen: BTreeMap<String, (Vec<u32>, bool)> = BTreeMap::new();
    for a in actions {
        match a {
            HostAction::Release { digest, region } => chosen.entry(digest).or_default().0.push(region),
            HostAction::Forget(digest) => chosen.entry(digest).or_default().1 = true,
        }
    }
    let (mut gave, mut unasked) = (discarded, vec![]);
    for (digest, (regions, forget)) in chosen {
        let Some(l) = s.layouts.get_mut(&digest) else { continue };
        let users: Vec<(u64, String)> = l.users.iter().map(|(c, (_, ask))| (*c, ask.clone())).collect();
        if users.is_empty() {
            gave += regions.iter().map(|r| l.release(*r)).sum::<u64>();
            if forget {
                s.layouts.remove(&digest);
            }
            continue;
        }
        l.pending = Some(Pending { regions, forget, waiting: users.iter().map(|(c, _)| *c).collect() });
        for (conn, ask) in users {
            let (st, d) = (state.clone(), digest.clone());
            let asking = std::thread::Builder::new().name("varena-tier-ask".into()).spawn(move || {
                let freed = ledger::release_host(&ask, &d);
                acked(&st, &d, conn, freed);
            });
            // No thread, no answer: the user is not waited for.
            if asking.is_err() {
                unasked.push((digest.clone(), conn));
            }
        }
    }
    if gave > 0 {
        s.releases += 1;
    }
    drop(s);
    unasked.into_iter().for_each(|(digest, conn)| acked(state, &digest, conn, 0));
    cv.notify_all();
    gave
}

/// A user answered (or died): once every user of the layout has, its regions are punched out.
fn acked(state: &Shared, digest: &str, conn: u64, freed: u64) {
    let (lock, cv) = &**state;
    let mut s = lock.lock();
    s.acked += freed;
    let Some(l) = s.layouts.get_mut(digest) else { return };
    l.users.remove(&conn);
    let Some(p) = l.pending.as_mut() else { return };
    p.waiting.remove(&conn);
    if !p.waiting.is_empty() {
        return;
    }
    let Some(p) = l.pending.take() else { return };
    let gave: u64 = p.regions.iter().map(|r| l.release(*r)).sum();
    if p.forget && l.users.is_empty() {
        s.layouts.remove(digest);
    }
    if gave > 0 {
        s.releases += 1;
    }
    drop(s);
    cv.notify_all();
}

/// The machine's RAM-tier service, served from this process until dropped.
pub struct Tier {
    state: Shared,
    stop: Arc<std::sync::atomic::AtomicBool>,
    /// The host-short event's waiting end.
    short: Owned,
    budget: Budget,
    claims: Arc<Claims>,
    /// The accept loop: the name is free once it ends.
    accepting: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Tier {
    /// Gives the name back (the accept loop wakes, sees the stop, and drops its listener) and
    /// closes its users' connections, as its process's end would.
    fn drop(&mut self) {
        self.stop.store(true, Release);
        let _ = os::connect(os::euid(), NAME);
        if let Some(accepting) = self.accepting.take() {
            let _ = accepting.join();
        }
        let conns = std::mem::take(&mut self.state.0.lock().conns);
        conns.values().for_each(os::close);
    }
}

impl Tier {
    /// Binds the machine's tier name and serves it, admitting fills against `need`; `None` when
    /// another participant does.
    pub fn serve(pressure: Pressure, need: Need) -> Result<Option<Tier>> {
        let Some(tier) = Tier::bind(need.clone())? else { return Ok(None) };
        tier.watch(pressure, need)?;
        Ok(Some(tier))
    }

    fn watch(&self, mut pressure: Pressure, need: Need) -> Result<()> {
        if matches!(pressure, Pressure::Headroom) { return Ok(()); }
        let (state, stop, budget, claims) = (self.state.clone(), self.stop.clone(), self.budget.clone(), self.claims.clone());
        std::thread::Builder::new()
            .name("varena-pressure".into())
            .spawn(move || loop {
                let woke = match &mut pressure {
                    Pressure::Host(p) => p.wait().map_err(Error::from),
                    Pressure::Events(e) => os::await_event(e),
                    Pressure::Headroom => return,
                };
                if woke.is_err() || stop.load(Acquire) {
                    return;
                }
                answer(&state, &need, &budget, &claims);
            })
            .map_err(|e| Error::io("spawn pressure", e))?;
        Ok(())
    }

    fn bind(need: Need) -> Result<Option<Tier>> {
        Self::bind_with(need, measured_budget())
    }

    fn bind_with(need: Need, budget: Budget) -> Result<Option<Tier>> {
        let Some(listener) = os::listen(os::euid(), NAME)? else { return Ok(None) };
        let claims = Arc::new(Claims::open()?);
        let short = os::event()?;
        let signal = os::dup(&short)?;
        let state = Arc::new((Mutex::new(State { short: Some(signal), ..State::default() }), Condvar::new()));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (s, halt, available, held_claims) = (state.clone(), stop.clone(), budget.clone(), claims.clone());
        let accepting = std::thread::Builder::new()
            .name("varena-tier".into())
            .spawn(move || {
                for stream in os::incoming(listener) {
                    if halt.load(Acquire) {
                        return;
                    }
                    let (s, need, budget, claims) = (s.clone(), need.clone(), available.clone(), held_claims.clone());
                    let _ = std::thread::Builder::new().name("varena-tier-conn".into()).spawn(move || serve(stream, &s, &need, &budget, &claims));
                }
            })
            .map_err(|e| Error::io("spawn tier", e))?;
        Ok(Some(Tier { state, stop, short, budget, claims, accepting: Some(accepting) }))
    }

    /// The layouts holding any ready region (status).
    pub fn held(&self) -> Vec<String> {
        let mut out: Vec<String> = self.state.0.lock().layouts.iter().filter(|(_, l)| l.holds()).map(|(d, _)| d.clone()).collect();
        out.sort();
        out
    }

    /// Each held layout's resident bytes: what its file really holds (status).
    pub fn resident(&self) -> Vec<(String, u64)> {
        let s = self.state.0.lock();
        let mut out: Vec<(String, u64)> = s.layouts.iter().map(|(d, l)| (d.clone(), os::allocated(&l.fd))).collect();
        out.sort();
        out
    }

    /// Blocks until the tier has nothing left to give back and the host is still short: TensorD
    /// then ends idle executors, least recently used first (the last host rung, a lifecycle
    /// action, so TensorD's).
    pub fn await_short(&self) -> Result<()> {
        os::await_event(&self.short)
    }

    /// The host-short event, for a caller that waits on several (signalled; a wait clears it).
    pub fn short(&self) -> &Owned {
        &self.short
    }

    /// Bytes executors unpinned and unmapped when the tier asked them to (status).
    pub fn acked(&self) -> u64 {
        self.state.0.lock().acked
    }

    /// Blocks until the tier has given back `n` rungs since it started; returns the count.
    /// Each layout's nearest declared use over its users (`None`: none declares one), as the
    /// executors last told it (status).
    pub fn uses(&self) -> Vec<(String, Option<u32>)> {
        let s = self.state.0.lock();
        let mut out: Vec<(String, Option<u32>)> = s.layouts.iter().map(|(d, l)| (d.clone(), l.users.values().filter_map(|(d, _)| *d).min())).collect();
        out.sort();
        out
    }

    /// Blocks until `seen` holds of [`Tier::uses`] (executors tell the tier asynchronously).
    pub fn await_uses(&self, seen: impl Fn(&[(String, Option<u32>)]) -> bool) {
        let (lock, cv) = &*self.state;
        let mut s = lock.lock();
        loop {
            let mut uses: Vec<(String, Option<u32>)> = s.layouts.iter().map(|(d, l)| (d.clone(), l.users.values().filter_map(|(d, _)| *d).min())).collect();
            uses.sort();
            if seen(&uses) {
                return;
            }
            cv.wait(&mut s);
        }
    }

    pub fn await_releases(&self, n: u64) -> u64 {
        let (lock, cv) = &*self.state;
        let mut s = lock.lock();
        while s.releases < n {
            cv.wait(&mut s);
        }
        s.releases
    }
}

fn serve(stream: Stream, state: &Shared, need: &Need, budget: &Budget, claims: &Claims) {
    // Only this user's processes: an abstract name has no permissions of its own.
    if os::peer(&stream).is_err() {
        return;
    }
    let (lock, cv) = &**state;
    let conn = {
        let mut s = lock.lock();
        s.clock += 1;
        let conn = s.clock;
        if let Ok(c) = stream.try_clone() {
            s.conns.insert(conn, c);
        }
        conn
    };
    while let Ok(Some((bytes, fd))) = os::recv(&stream) {
        let Ok(call) = os::decode::<Call>(&bytes) else { break };
        let mut s = lock.lock();
        s.clock += 1;
        let clock = s.clock;
        let (reply, out) = match call {
            Call::Open { digest, regions, ask } => match open(&mut s, &digest, regions) {
                Ok((data, fd)) => {
                    if let Some(l) = s.layouts.get_mut(&digest) {
                        l.users.insert(conn, (None, ask));
                        l.used = clock;
                    }
                    (Reply::Opened { data }, Some(fd))
                }
                Err(_) => (Reply::Room { ok: false }, None),
            },
            Call::Adopt { digest, regions, ask } => {
                if !s.layouts.contains_key(&digest) {
                    if let Some(Ok(l)) = fd.map(|f| Layout::adopt(f, regions)) {
                        s.layouts.insert(digest.clone(), l);
                    }
                }
                if let Some(l) = s.layouts.get_mut(&digest) {
                    l.users.insert(conn, (None, ask));
                    l.used = clock;
                }
                cv.notify_all();
                (Reply::Done, None)
            }
            Call::Use { digest, distance } => {
                if let Some((d, _)) = s.layouts.get_mut(&digest).and_then(|l| {
                    l.used = clock;
                    l.users.get_mut(&conn)
                }) {
                    *d = distance;
                }
                cv.notify_all();
                (Reply::Done, None)
            }
            Call::Close { digest } => {
                if let Some(l) = s.layouts.get_mut(&digest) {
                    l.users.remove(&conn);
                }
                (Reply::Done, None)
            }
            Call::Room { bytes } => {
                drop(s);
                // Never waits: what nobody maps goes now; what executors map is asked back.
                let lease = admit(state, need, bytes, budget, claims).unwrap_or_else(|e| {
                    tracing::warn!("varena tier admission refused: {e}");
                    None
                });
                cv.notify_all();
                s = lock.lock();
                (Reply::Room { ok: lease.is_some() }, lease)
            }
            Call::Resident => (Reply::Resident { bytes: s.layouts.values().map(|l| os::allocated(&l.fd)).sum() }, None),
        };
        drop(s);
        let granted = matches!(reply, Reply::Room { ok: true });
        if os::encode(&reply).and_then(|b| os::send(&stream, &(b, out))).is_err() {
            break;
        }
        // The temporary reply dropped our claim fd. A receiver may finish recvmsg before
        // sendmsg returns here, so acknowledge that it is now the sole owner before letting
        // reserve return. Permit drop then releases the promise without an IPC round trip.
        if granted && os::encode(&Reply::Done).and_then(|b| os::send(&stream, &(b, None))).is_err() {
            break;
        }
    }
    let mut s = lock.lock();
    s.conns.remove(&conn);
    s.layouts.values_mut().for_each(|l| {
        l.users.remove(&conn);
    });
    let discarded: u64 = s.layouts.values().map(Layout::discard_unready).sum();
    if discarded > 0 { s.releases += 1; }
    drop(s);
    cv.notify_all();
}

/// The layout's file and data offset, made when absent.
fn open(s: &mut State, digest: &str, regions: Vec<(u64, u64)>) -> Result<(u64, Owned)> {
    if let Some(l) = s.layouts.get(digest) {
        return Ok((l.data, os::dup(&l.fd)?));
    }
    let data = words_bytes(&regions);
    let end = regions.iter().map(|(o, s)| o + s).max().unwrap_or(0);
    let fd = os::memfd(&format!("varena-tier-{digest}"))?;
    os::resize(&fd, data + end)?;
    let words = os::map(&fd, data as usize, true)?;
    let handed = os::dup(&fd)?;
    s.layouts.insert(digest.to_string(), Layout { fd, words, regions, data, users: HashMap::new(), used: 0, pending: None });
    Ok((data, handed))
}

/// The layout file's ready-words page(s) ahead of its data.
fn words_bytes(regions: &[(u64, u64)]) -> u64 {
    let page = platform_page();
    ((regions.len() as u64 * 4).max(1)).div_ceil(page) * page
}

impl Layout {
    /// A layout from its file, its ready words as they are.
    fn adopt(fd: Owned, regions: Vec<(u64, u64)>) -> Result<Layout> {
        let data = words_bytes(&regions);
        let words = os::map(&fd, data as usize, true)?;
        Ok(Layout { fd, words, regions, data, users: HashMap::new(), used: 0, pending: None })
    }
}

fn platform_page() -> u64 {
    #[cfg(unix)]
    return platform::shm::page_size() as u64;
    #[cfg(not(unix))]
    return 1 << 16;
}

/// An executor's view of one layout in the tier.
pub(crate) struct Held {
    /// This process's own description: its locks are its own.
    fd: Owned,
    ro: Map,
    data: u64,
    regions: Vec<(u64, u64)>,
    /// Regions page-locked for the copy engine (once each, while ready), regions a copy is
    /// reading now, and whether the tier asked for the layout back.
    pins: parking_lot::Mutex<Pins>,
}

#[derive(Default)]
struct Pins {
    registered: std::collections::HashSet<u32>,
    reading: std::collections::HashSet<u32>,
    released: bool,
    /// Regions still read when the layout was released: unlocked when their copy ends.
    deferred: Vec<u32>,
}

impl Held {
    /// Its file and regions, to hand to a tier that does not know the layout.
    pub fn handover(&self) -> Result<(Owned, Vec<(u64, u64)>)> {
        Ok((os::dup(&self.fd)?, self.regions.clone()))
    }

    fn ptr(&self, r: u32) -> Result<(u64, u64)> {
        let (off, span) = self.region(r)?;
        Ok((self.ro.base as u64 + self.data + off, span))
    }

    /// The tier wants the layout back: every region not being read is unlocked now, the rest
    /// when their copy ends; the mapping goes with the last holder. Returns the bytes it had
    /// page-locked.
    pub fn release(&self, mut unregister: impl FnMut(u64) -> Result<()>) -> u64 {
        let mut pins = self.pins.lock();
        pins.released = true;
        let regions: Vec<u32> = pins.registered.drain().collect();
        let mut bytes = 0;
        for r in regions {
            let Ok((ptr, span)) = self.ptr(r) else { continue };
            bytes += span;
            if pins.reading.contains(&r) {
                pins.deferred.push(r);
            } else {
                let _ = unregister(ptr);
            }
        }
        bytes
    }

    /// After a copy from region `r`: unlocks it if the layout was released meanwhile.
    pub fn settle(&self, r: u32, unregister: impl FnOnce(u64) -> Result<()>) {
        let mut pins = self.pins.lock();
        pins.reading.remove(&r);
        if let Some(i) = pins.deferred.iter().position(|d| *d == r) {
            pins.deferred.swap_remove(i);
            if let Ok((ptr, _)) = self.ptr(r) {
                let _ = unregister(ptr);
            }
        }
    }

    /// Page-locks region `r` once it is ready, so fills from it are pinned copies; a region
    /// the tier gave back is unlocked first. A released layout is never locked again.
    pub fn pin(&self, r: u32, register: impl FnOnce(u64, u64) -> Result<()>, unregister: impl FnOnce(u64) -> Result<()>) -> Result<()> {
        let (ptr, span) = self.ptr(r)?;
        let ready = self.ready(r).is_some_and(|w| w.load(Acquire) == 1);
        let mut pins = self.pins.lock();
        if pins.released {
            return Ok(());
        }
        let registered = &mut pins.registered;
        match (ready, registered.contains(&r)) {
            (true, false) => {
                register(ptr, span)?;
                registered.insert(r);
            }
            (false, true) => {
                unregister(ptr)?;
                registered.remove(&r);
            }
            _ => {}
        }
        Ok(())
    }

    /// Unlocks every region (the layout is closing).
    pub fn unpin(&self, unregister: impl FnMut(u64) -> Result<()>) {
        self.release(unregister);
    }
    fn ready(&self, r: u32) -> Option<&AtomicU32> {
        let at = r as usize * 4;
        (at + 4 <= self.data as usize).then(|| {
            // SAFETY: inside the read-only words page, aligned to 4.
            unsafe { &*(self.ro.base.add(at) as *const AtomicU32) }
        })
    }

    fn region(&self, r: u32) -> Result<(u64, u64)> {
        self.regions.get(r as usize).copied().ok_or_else(|| Error::Invalid(format!("region {r}")))
    }

    /// Whether the tier holds region `r` (filled, and not taken back).
    pub fn holds(&self, r: u32) -> bool {
        self.ready(r).is_some_and(|w| w.load(Acquire) == 1)
    }

    /// Copies each ready region into its destination, all in one engine read, under their
    /// read locks; whether each was copied. The caller settles each afterwards
    /// ([`Held::settle`]).
    pub fn read(self: &Arc<Self>, regions: &[(u32, tensorfs::io::Dst)], drain: impl FnOnce() -> Result<()>) -> Result<Vec<bool>> {
        self.read_with(regions, |copies| {
            // SAFETY: the operation's locks and exact source mapping outlive the copy and
            // any required error drain; the caller owns every destination exclusively.
            unsafe { tensorfs::io::read(copies, &tensorfs::io::ReadOpts::default()) }.map(|_| ()).map_err(Error::from)
        }, drain)
    }

    fn read_with(self: &Arc<Self>, regions: &[(u32, tensorfs::io::Dst)],
        copy: impl FnOnce(&[tensorfs::io::Copy]) -> Result<()>, drain: impl FnOnce() -> Result<()>) -> Result<Vec<bool>> {
        let spans = regions.iter().map(|(r, dst)| self.region(*r).and_then(|(off, span)| if dst.len() as u64 > span { Err(Error::Invalid("tier read past its region".into())) } else { Ok((off, span)) })).collect::<Result<Vec<_>>>()?;
        let locks = self.lock_regions(regions.iter().map(|(r, _)| *r), false)?;
        self.pins.lock().reading.extend(regions.iter().map(|(r, _)| *r));
        let ready: Vec<bool> = regions.iter().map(|(r, _)| self.ready(*r).is_some_and(|w| w.load(Acquire) == 1)).collect();
        // SAFETY: each ready region is read-locked and inside this exact retained mapping.
        let src = |off: u64| tensorfs::io::Src::Host(unsafe { self.ro.base.add((self.data + off) as usize) });
        let copies: Vec<_> = regions.iter().zip(&spans).zip(&ready).filter(|(_, ready)| **ready).map(|(((_, dst), (off, _)), _)| tensorfs::io::Copy { src: src(*off), dst: *dst }).collect();
        if !copies.is_empty() {
            let copied = crate::caught(|| copy(&copies));
            if let Err(error) = copied {
                if matches!(error, Error::Ownership(_)) {
                    std::mem::forget((self.clone(), locks));
                    return Err(error);
                }
                // A direct host source may still be borrowed after CUDA failure. Drain while
                // BOTH its mapping and read lock are alive; an error is not completion.
                if let Err(draining) = crate::caught(drain) {
                    std::mem::forget((self.clone(), locks));
                    return Err(Error::Ownership(format!("RAM-tier read completion is unknown after {error}; drain failed: {draining}")));
                }
                return Err(error);
            }
        }
        Ok(ready)
    }

    /// Fill owns distinct write-lock descriptions, so concurrent calls through one Held
    /// exclude one another too. Ordinary Err means the caller proved its I/O drained;
    /// Ownership means completion is unknown and destination mappings/locks are quarantined.
    pub fn fill(&self, regions: &[u32], fill: impl FnOnce(&[(u32, tensorfs::io::Dst)]) -> Result<()>) -> Result<()> {
        let locks = self.lock_regions(regions.iter().copied(), true)?;
        let todo: Vec<u32> = regions.iter().copied().collect::<std::collections::BTreeSet<_>>().into_iter()
            .filter(|r| self.ready(*r).is_none_or(|w| w.load(Acquire) != 1)).collect();
        if todo.is_empty() { return Ok(()); }
        let words = os::map_at(&locks, 0, self.data as usize, true)?;
        let maps = todo.iter().map(|r| self.region(*r).and_then(|(off, span)| os::map_at(&locks, self.data + off, span as usize, true))).collect::<Result<Vec<_>>>()?;
        let dsts: Vec<_> = todo.iter().zip(&maps).map(|(r, m)| (*r, tensorfs::io::Dst::Host { ptr: m.base, len: m.len })).collect();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fill(&dsts)))
            .unwrap_or_else(|_| Err(Error::Ownership("RAM-tier fill panicked before proving completion".into())));
        if let Err(error) = outcome {
            if matches!(error, Error::Ownership(_)) {
                std::mem::forget((maps, words, locks));
                return Err(error);
            }
            // A drained failure must not strand dirty pages behind ready=0, where the
            // pressure policy cannot see them. No ready region is punched by this cleanup.
            for r in todo {
                let (off, span) = self.region(r)?;
                if let Err(e) = os::punch(&locks, self.data + off, span) {
                    std::mem::forget((maps, words, locks));
                    return Err(Error::Ownership(format!("RAM-tier failed-fill cleanup could not return its pages: {e}")));
                }
            }
            return Err(error);
        }
        for r in todo {
            // SAFETY: inside the words page, aligned to 4, with this region write-locked.
            let word = unsafe { &*(words.base.add(r as usize * 4) as *const AtomicU32) };
            word.store(1, Release);
        }
        Ok(())
    }

    /// A separate OFD description per operation: locks on Held's shared description would
    /// not exclude two local threads, and could not outlive Held after quarantining a fill.
    fn lock_regions(&self, regions: impl Iterator<Item = u32>, write: bool) -> Result<Owned> {
        let mut order: Vec<u32> = regions.collect();
        order.sort_unstable();
        order.dedup();
        let fd = os::reopen(&self.fd, true)?;
        for r in order {
            let (off, span) = self.region(r)?;
            match write {
                true => os::lock(&fd, self.data + off, span, true)?,
                false => os::share(&fd, self.data + off, span, true)?,
            };
        }
        Ok(fd)
    }


}

/// A fill's host reservation, owned in the client. Its record's read-lock fd survives a
/// service restart and closes on completion/client death. Never unlink the named claim file.
pub(crate) struct Permit {
    // Drop the kernel claim before closing the socket that wakes the current service.
    _claim: Owned,
    _stream: Stream,
}

/// An executor's connection to the machine's tier.
pub(crate) struct Client {
    stream: Mutex<Stream>,
    /// The namespace of the tier it found (the table owner's).
    ns: u32,
    /// The tier this client serves from this process, when it found none.
    service: Mutex<Option<Tier>>,
    /// Connections made: a new one may be a new tier (its server restarted), which knows none
    /// of this user's layouts, so its user hands them over again.
    holder: AtomicU64,
    /// What a tier served here admits fills against.
    need: Need,
}

impl Client {
    /// Connects to the tier in user `ns`'s namespace (the table owner's), serving it from this
    /// process (with the host's pressure trigger when it arms) when nobody does and it is this
    /// user's; another user's tier is its TensorD's, never ours.
    /// Serving it here, the tier admits fills against `need`.
    pub fn open(ns: u32, need: Need) -> Result<Client> {
        let (stream, service) = connect(ns, &need)?;
        Ok(Client { stream: Mutex::new(stream), ns, service: Mutex::new(service), holder: AtomicU64::new(1), need })
    }

    /// Which connection this is (it changes when the tier may have).
    pub fn holder(&self) -> u64 {
        self.holder.load(Acquire)
    }

    /// Hands layout `digest` (this user's file of it) to the tier, which takes it as it is
    /// unless it knows it.
    pub fn adopt_layout(&self, digest: &str, regions: Vec<(u64, u64)>, ask: &str, file: Owned) -> Result<()> {
        self.call_with(&Call::Adopt { digest: digest.into(), regions, ask: ask.into() }, Some(file)).map(|_| ())
    }

    /// One call; a broken connection (its tier's server went) is replaced once. The stream's
    /// lock is a leaf: connecting (which may start a tier) happens outside it.
    fn call(&self, call: &Call) -> Result<(Reply, Option<Owned>)> {
        self.call_with(call, None)
    }

    fn call_with(&self, call: &Call, file: Option<Owned>) -> Result<(Reply, Option<Owned>)> {
        let handle = file.as_ref().map(os::dup).transpose()?;
        if let Ok(reply) = os::call(&self.stream.lock(), call, handle) {
            return Ok(reply);
        }
        let (stream, service) = connect(self.ns, &self.need)?;
        if let Some(service) = service {
            *self.service.lock() = Some(service);
        }
        self.holder.fetch_add(1, AcqRel);
        let mut current = self.stream.lock();
        *current = stream;
        os::call(&current, call, file)
    }

    /// Maps `digest`'s layout; the tier asks `ask` (this process's row) to give it back.
    pub fn open_layout(&self, digest: &str, regions: Vec<(u64, u64)>, ask: &str) -> Result<Held> {
        let (reply, fd) = self.call(&Call::Open { digest: digest.into(), regions: regions.clone(), ask: ask.into() })?;
        let (Reply::Opened { data }, Some(fd)) = (reply, fd) else { return Err(Error::Io("the tier refused a layout".into())) };
        let fd = os::reopen(&fd, true)?;
        let len = data + regions.iter().map(|(o, s)| o + s).max().unwrap_or(0);
        let ro = os::map(&fd, len as usize, false)?;
        Ok(Held { fd, ro, data, regions, pins: Default::default() })
    }

    pub fn use_layout(&self, digest: &str, distance: Option<u32>) -> Result<()> {
        self.call(&Call::Use { digest: digest.into(), distance }).map(|_| ())
    }

    pub fn close(&self, digest: &str) -> Result<()> {
        self.call(&Call::Close { digest: digest.into() }).map(|_| ())
    }

    /// Host headroom for `bytes`, the tier giving back in order; false when it cannot now. A
    /// connection of its own: a fill never waits behind another call.
    pub fn reserve(&self, bytes: u64) -> Option<Permit> {
        let stream = os::connect(self.ns, NAME).ok()??;
        match os::call::<_, Reply>(&stream, &Call::Room { bytes }, None) {
            Ok((Reply::Room { ok: true }, Some(claim))) => {
                let (done, extra) = os::recv(&stream).ok()??;
                if !matches!(os::decode::<Reply>(&done), Ok(Reply::Done)) || extra.is_some() { return None; }
                Some(Permit { _claim: claim, _stream: stream })
            }
            _ => None,
        }
    }

    /// Every layout's resident bytes in the tier, together (0 when it cannot be asked).
    pub fn resident(&self) -> u64 {
        let Ok(Some(stream)) = os::connect(self.ns, NAME) else { return 0 };
        match os::call::<_, Reply>(&stream, &Call::Resident, None) {
            Ok((Reply::Resident { bytes }, _)) => bytes,
            _ => 0,
        }
    }
}

/// Connects to the tier in user `ns`'s namespace, serving it from this process when nobody does
/// and it is this user's (returned with the stream).
fn connect(ns: u32, need: &Need) -> Result<(Stream, Option<Tier>)> {
    let mut service = None;
    loop {
        if let Some(stream) = os::connect(ns, NAME)? {
            return Ok((stream, service));
        }
        if ns != os::euid() {
            return Err(Error::Closed("the RAM tier (its TensorD serves none)".into()));
        }
        let mode = platform::contain::host_pressure::HostMode::Shared;
        service = match platform::contain::host_pressure::Pressure::arm(mode) {
            Ok(p) => Tier::serve(Pressure::Host(p), need.clone())?,
            Err(_) => Tier::serve(Pressure::Headroom, need.clone())?,
        };
    }
}


#[cfg(all(test, target_os = "linux"))]
mod reservation_tests {
    #![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
    use super::*;

    // Each service case gets a subprocess and namespace; a lib-test thread never changes
    // another test's process-wide namespace or connects to the user's actual tier.
    fn isolated(name: &str, run: impl FnOnce()) {
        const ROLE: &str = "VARENA_TIER_RESERVATION_TEST";
        if std::env::var(ROLE).as_deref() == Ok(name) {
            assert!(os::namespace(&format!("tier-reservation-{}-", std::process::id())));
            run();
            return;
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([&format!("tier::reservation_tests::{name}"), "--exact", "--nocapture", "--test-threads=1"])
            .env(ROLE, name).status().unwrap();
        assert!(status.success(), "isolated tier case {name}: {status}");
    }

    fn room(bytes: u64) -> (Option<Permit>, bool) {
        let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
        let permit = client.reserve(bytes);
        let admitted = permit.is_some();
        (permit, admitted)
    }

    #[test]
    fn concurrent_fills_cannot_spend_the_same_surplus() {
        isolated("concurrent_fills_cannot_spend_the_same_surplus", || {
            let _tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(|| 6)).unwrap().unwrap();
            let (_first, admitted) = room(4);
            assert!(admitted);
            // The first fill has been admitted but has not touched its pages yet. Another
            // connection sees the same OS headroom, but four bytes are already promised.
            let (_second, admitted) = room(4);
            assert!(!admitted, "two pending fills spent the same six bytes of host surplus");
        });
    }

    fn await_promised(tier: &Tier, bytes: u64) {
        let (lock, changed) = &*tier.state;
        let mut state = lock.lock();
        while tier.claims.live_bytes().unwrap() != bytes {
            assert!(!changed.wait_for(&mut state, std::time::Duration::from_secs(5)).timed_out(), "reservation was not released");
        }
    }

    #[test]
    fn closing_a_permit_returns_only_the_promise_not_resident_pages() {
        isolated("closing_a_permit_returns_only_the_promise_not_resident_pages", || {
            let available = Arc::new(AtomicU64::new(6));
            let sample = available.clone();
            let tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(move || i128::from(sample.load(Acquire)))).unwrap().unwrap();
            let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
            let permit = client.reserve(4).unwrap();
            assert!(client.reserve(4).is_none());
            available.store(2, Release); // The completed fill now occupies four physical bytes.
            drop(permit);
            await_promised(&tier, 0);
            assert!(client.reserve(4).is_none(), "resident pages remain charged by live headroom");
            available.store(6, Release); // Only actual reclaim restores them.
            assert!(client.reserve(4).is_some());
        });
    }

    #[test]
    fn a_dead_client_returns_its_unspent_reservation() {
        isolated("a_dead_client_returns_its_unspent_reservation", || {
            use std::io::BufRead;
            let tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(|| 6)).unwrap().unwrap();
            let address = format!("cozy-varena-{}-tier-reservation-{}-tier", os::euid(), std::process::id());
            let script = r#"import array,json,socket,struct,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
s.connect('\0'+sys.argv[1])
b=json.dumps({'Room':{'bytes':4}}).encode()
s.sendall(struct.pack('<I',len(b))+b'\0'+b)
def read(n):
    b=b''
    while len(b)<n:
        p=s.recv(n-len(b))
        if not p: raise RuntimeError('tier closed')
        b+=p
    return b
head=read(5)
r=json.loads(read(struct.unpack('<I',head[:4])[0]))
assert r['Room']['ok'] and head[4]==1,r
_,ancillary,_,_=s.recvmsg(1,socket.CMSG_SPACE(array.array('i').itemsize))
claim=array.array('i');claim.frombytes(ancillary[0][2]);assert len(claim)==1
head=read(5);assert json.loads(read(struct.unpack('<I',head[:4])[0]))=='Done' and head[4]==0
print('reserved',flush=True)
sys.stdin.read()
"#;
            struct Child(std::process::Child);
            impl Drop for Child {
                fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); }
            }
            let mut child = Child(std::process::Command::new("python3").args(["-c", script, &address])
                .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn().unwrap());
            let mut line = String::new();
            std::io::BufReader::new(child.0.stdout.take().unwrap()).read_line(&mut line).unwrap();
            assert_eq!(line.trim(), "reserved");
            assert!(!room(4).1);
            child.0.kill().unwrap();
            child.0.wait().unwrap();
            await_promised(&tier, 0);
            assert!(room(4).1, "a dead client cannot strand its admission");
        });
    }

    #[test]
    fn unmeasured_calls_and_pending_fills_keep_their_headroom() {
        isolated("unmeasured_calls_and_pending_fills_keep_their_headroom", || {
            let need = Arc::new(AtomicU64::new(3));
            let read = need.clone();
            let tier = Tier::bind_with(Arc::new(move || match read.load(Acquire) { u64::MAX => None, n => Some(n) }), Arc::new(|| 8)).unwrap().unwrap();
            let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
            assert!(client.reserve(6).is_none());
            let permit = client.reserve(5).unwrap();
            assert!(client.reserve(1).is_none());
            drop(permit);
            await_promised(&tier, 0);
            need.store(u64::MAX, Release);
            assert!(client.reserve(1).is_none(), "an unmeasured call permits no speculative fill");
        });
    }

    #[test]
    fn a_drained_fill_failure_returns_only_its_unfinished_pages() {
        isolated("a_drained_fill_failure_returns_only_its_unfinished_pages", || {
            let page = platform_page();
            let tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(move || i128::from(8 * page))).unwrap().unwrap();
            let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
            let held = client.open_layout("failed", vec![(0, page), (page, page)], "").unwrap();
            let first = client.reserve(page).unwrap();
            held.fill(&[0], |dsts| {
                for (_, dst) in dsts {
                    let tensorfs::io::Dst::Host { ptr, len } = dst else { unreachable!() };
                    // SAFETY: the fill owns this mapped host destination exclusively.
                    unsafe { std::ptr::write_bytes(*ptr, 7, *len) };
                }
                Ok(())
            }).unwrap();
            drop(first);
            await_promised(&tier, 0);
            let next = client.reserve(page).unwrap();
            let result = held.fill(&[0, 1], |dsts| {
                assert_eq!(dsts.len(), 1, "ready pages must not be overwritten");
                for (_, dst) in dsts {
                    let tensorfs::io::Dst::Host { ptr, len } = dst else { unreachable!() };
                    // SAFETY: the fill owns this mapped host destination exclusively.
                    unsafe { std::ptr::write_bytes(*ptr, 9, *len) };
                }
                Err(Error::Io("drained read failure".into()))
            });
            assert!(matches!(result, Err(Error::Io(_))));
            drop(next);
            await_promised(&tier, 0);
            assert!(held.holds(0) && !held.holds(1));
            assert!(os::allocated(&held.fd) <= held.data + page, "dirty unready pages were stranded");
            // SAFETY: the first ready region remains mapped by Held.
            assert_eq!(unsafe { *(held.ptr(0).unwrap().0 as *const u8) }, 7);
        });
    }

    #[test]
    fn an_undrained_fill_keeps_its_destination_mapping_and_region_lock() {
        isolated("an_undrained_fill_keeps_its_destination_mapping_and_region_lock", || {
            let page = platform_page();
            let _tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(move || i128::from(8 * page))).unwrap().unwrap();
            let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
            let held = client.open_layout("quarantined", vec![(0, page)], "").unwrap();
            let inspection = os::reopen(&held.fd, true).unwrap();
            let permit = client.reserve(page).unwrap();
            let pointer = AtomicU64::new(0);
            let result = held.fill(&[0], |dsts| {
                let tensorfs::io::Dst::Host { ptr, .. } = dsts[0].1 else { unreachable!() };
                pointer.store(ptr as u64, Release);
                Err(Error::Ownership("copy completion unknown".into()))
            });
            assert!(matches!(result, Err(Error::Ownership(_))));
            let data = held.data;
            assert!(!held.holds(0));
            drop(held);
            assert!(os::held(&inspection, data, page).unwrap(), "quarantine lost the write lock");
            let mut residency = [0u8; 1];
            // SAFETY: mincore only queries the saved page-aligned mapping; it does not dereference it.
            assert_eq!(unsafe { libc::mincore(pointer.load(Acquire) as *mut libc::c_void, page as usize, residency.as_mut_ptr()) }, 0,
                "quarantine unmapped the asynchronous destination");
            // The caller quarantines its host permit alongside the unresolved GPU source.
            std::mem::forget(permit);
        });
    }

    #[test]
    fn local_fillers_use_distinct_lock_descriptions() {
        isolated("local_fillers_use_distinct_lock_descriptions", || {
            let page = platform_page();
            let _tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(move || i128::from(8 * page))).unwrap().unwrap();
            let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
            let held = Arc::new(client.open_layout("local-locks", vec![(0, page)], "").unwrap());
            let (entered, ready) = std::sync::mpsc::channel();
            let (release, done) = std::sync::mpsc::channel();
            let first = held.clone();
            let writer = std::thread::spawn(move || first.fill(&[0], |_| {
                entered.send(()).unwrap();
                done.recv().unwrap();
                Ok(())
            }));
            ready.recv().unwrap();
            assert!(os::held(&held.fd, held.data, page).unwrap(), "a shared OFD does not exclude local writers");
            let second = held.clone();
            let follower = std::thread::spawn(move || second.fill(&[0], |_| panic!("a ready region must not be filled twice")));
            release.send(()).unwrap();
            writer.join().unwrap().unwrap();
            follower.join().unwrap().unwrap();
        });
    }


    #[test]
    fn observed_growth_deficit_reclaims_even_with_zero_call_need() {
        isolated("observed_growth_deficit_reclaims_even_with_zero_call_need", || {
            let page = platform_page();
            let budget = Arc::new(AtomicU64::new(8 * page));
            let available = budget.clone();
            let tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(move || i128::from(available.load(Acquire)) - 1)).unwrap().unwrap();
            let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
            let held = client.open_layout("growth", vec![(0, page)], "").unwrap();
            held.fill(&[0], |dsts| {
                let tensorfs::io::Dst::Host { ptr, len } = dsts[0].1 else { unreachable!() };
                // SAFETY: this fill exclusively owns its mapped destination.
                unsafe { std::ptr::write_bytes(ptr, 1, len) };
                Ok(())
            }).unwrap();
            client.close("growth").unwrap();
            budget.store(0, Release); // Need0, but measured growth exceeds live headroom.
            let need: Need = Arc::new(|| Some(0));
            answer(&tier.state, &need, &tier.budget, &tier.claims);
            assert!(tier.held().is_empty(), "saturating a growth deficit to0 suppressed pressure reclaim");
        });
    }

    #[test]
    fn a_failed_read_drains_before_releasing_its_exact_source() {
        isolated("a_failed_read_drains_before_releasing_its_exact_source", || {
            for completes in [true, false] {
                let page = platform_page();
                let _tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(move || i128::from(8 * page))).unwrap().unwrap();
                let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
                let held = Arc::new(client.open_layout("read-failure", vec![(0, page)], "").unwrap());
                held.fill(&[0], |_| Ok(())).unwrap();
                let observed = held.clone();
                let inspection = os::reopen(&held.fd, true).unwrap();
                let data = held.data;
                let original_mapping = held.ro.base;
                let pointer = AtomicU64::new(0);
                let result = held.read_with(&[(0, tensorfs::io::Dst::Device { device: 0, ptr: 1, len: page as usize, stream: 0 })], |copies| {
                    let tensorfs::io::Src::Host(ptr) = copies[0].src else { unreachable!() };
                    pointer.store(ptr as u64, Release);
                    Err(Error::Io("in-flight copy failed".into()))
                }, || {
                    assert!(os::held(&inspection, data, page).unwrap(), "read lock released before drain");
                    assert_eq!(pointer.load(Acquire), original_mapping as u64 + data, "drain retained a different source VA");
                    if completes { Ok(()) } else { Err(Error::Io("completion check failed".into())) }
                });
                if completes {
                    assert!(matches!(result, Err(Error::Io(_))));
                    assert!(!os::held(&inspection, data, page).unwrap());
                    held.settle(0, |_| Ok(()));
                } else {
                    assert!(matches!(result, Err(Error::Ownership(_))));
                    assert!(observed.pins.lock().reading.contains(&0));
                    drop(observed);
                    drop(held);
                    assert!(os::held(&inspection, data, page).unwrap(), "quarantine released its read lock");
                    let mut residency = [0u8; 2];
                    // SAFETY: mincore queries, without dereferencing, this exact old mapping.
                    assert_eq!(unsafe { libc::mincore(original_mapping.cast(), (data + page) as usize, residency.as_mut_ptr()) }, 0);
                }
            }
        });
    }


    #[test]
    fn a_dead_filler_returns_dirty_unready_pages() {
        isolated("a_dead_filler_returns_dirty_unready_pages", || {
            use std::io::BufRead;
            use std::os::fd::AsRawFd;
            use std::os::unix::process::CommandExt;
            let page = platform_page();
            let tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(move || i128::from(8 * page))).unwrap().unwrap();
            let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
            let held = client.open_layout("dead-fill", vec![(0, page)], "").unwrap();
            let words = os::map_at(&held.fd, 0, held.data as usize, true).unwrap();
            // SAFETY: the ready word is mapped and no writer has begun this fixture yet.
            unsafe { (&*words.base.cast::<AtomicU32>()).store(0, Release) };
            let address = format!("cozy-varena-{}-tier-reservation-{}-tier", os::euid(), std::process::id());
            let script = r#"import array,ctypes,fcntl,json,mmap,os,socket,struct,sys
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.connect('\0'+sys.argv[1])
size=int(sys.argv[3]);data=int(sys.argv[2])
b=json.dumps({'Room':{'bytes':size}}).encode();s.sendall(struct.pack('<I',len(b))+b'\0'+b)
def read(n):
    b=b''
    while len(b)<n:
        p=s.recv(n-len(b))
        if not p:raise RuntimeError('tier closed')
        b+=p
    return b
head=read(5);assert json.loads(read(struct.unpack('<I',head[:4])[0]))['Room']['ok'] and head[4]==1
_,ancillary,_,_=s.recvmsg(1,socket.CMSG_SPACE(array.array('i').itemsize))
claim=array.array('i');claim.frombytes(ancillary[0][2]);assert len(claim)==1
head=read(5);assert json.loads(read(struct.unpack('<I',head[:4])[0]))=='Done' and head[4]==0
fd=os.open('/proc/self/fd/3',os.O_RDWR)
class Lock(ctypes.Structure):
    _fields_=[('kind',ctypes.c_short),('whence',ctypes.c_short),('start',ctypes.c_longlong),('length',ctypes.c_longlong),('pid',ctypes.c_int)]
fcntl.fcntl(fd,fcntl.F_OFD_SETLKW,bytes(Lock(fcntl.F_WRLCK,os.SEEK_SET,data,size,0)))
m=mmap.mmap(fd,size,offset=data);m[:]=b'\7'*size
print('dirty',flush=True);sys.stdin.read()
"#;
            struct Child(std::process::Child);
            impl Drop for Child {
                fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); }
            }
            let fd = held.fd.as_raw_fd();
            let mut command = std::process::Command::new("python3");
            command.args(["-c", script, &address, &held.data.to_string(), &page.to_string()])
                .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped());
            // SAFETY: only async-signal-safe descriptor operations between fork and exec.
            unsafe { command.pre_exec(move || {
                if libc::dup2(fd, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 { return Err(std::io::Error::last_os_error()); }
                Ok(())
            }); }
            let mut child = Child(command.spawn().unwrap());
            let mut line = String::new();
            std::io::BufReader::new(child.0.stdout.take().unwrap()).read_line(&mut line).unwrap();
            assert_eq!(line.trim(), "dirty");
            assert!(!held.holds(0));
            assert!(os::allocated(&held.fd) >= held.data + page);
            assert_eq!(held.region(0).unwrap().1, page);
            // Cleanup must respect the live fill's write lease.
            assert_eq!(tier.state.0.lock().layouts.get("dead-fill").unwrap().discard_unready(), 0);
            child.0.kill().unwrap();
            child.0.wait().unwrap();
            await_promised(&tier, 0);
            relieve(&tier.state, HostTrigger::Pressure);
            assert!(os::allocated(&held.fd) <= held.data, "a dead fill stranded dirty pages outside the pressure policy");
        });
    }


    #[test]
    #[ignore = "subprocess service holder for the restart regression"]
    fn claim_holder() {
        use std::io::Read;
        let Ok(prefix) = std::env::var("VARENA_TIER_CLAIM_SERVER") else { return };
        assert!(os::namespace(&prefix));
        let _tier = Tier::bind_with(Arc::new(|| Some(0)), Arc::new(|| 6)).unwrap().unwrap();
        println!("claim-holder-ready");
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input).unwrap();
    }

    #[test]
    fn holder_death_preserves_unfaulted_client_claims() {
        isolated("holder_death_preserves_unfaulted_client_claims", || {
            use std::io::BufRead;
            struct Child(std::process::Child);
            impl Drop for Child {
                fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); }
            }
            let prefix = format!("tier-reservation-{}-", std::process::id());
            let holder = || {
                let mut child = Child(std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["tier::reservation_tests::claim_holder", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
                    .env("VARENA_TIER_CLAIM_SERVER", &prefix)
                    .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn().unwrap());
                let mut output = std::io::BufReader::new(child.0.stdout.take().unwrap());
                loop {
                    let mut line = String::new();
                    assert!(output.read_line(&mut line).unwrap() > 0, "holder exited before serving");
                    if line.contains("claim-holder-ready") { break; }
                }
                child
            };
            let first = holder();
            let client = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
            let permit = client.reserve(4).unwrap(); // No destination page has faulted yet.
            assert_eq!(Claims::open().unwrap().live_bytes().unwrap(), 4);
            drop(first); // Real SIGKILL of the service process, while the client survives.
            let _replacement = holder();
            let next = Client::open(os::euid(), Arc::new(|| Some(0))).unwrap();
            assert!(next.reserve(4).is_none(), "replacement forgot a live future-byte promise");
            drop(client);
            assert!(next.reserve(4).is_none(), "a permit outlives its ordinary client connection");
            drop(permit);
            let settled = next.reserve(6).unwrap();
            assert_eq!(Claims::open().unwrap().live_bytes().unwrap(), 6);
            drop(settled);
            assert_eq!(Claims::open().unwrap().live_bytes().unwrap(), 0, "completion returns the durable claim immediately");
        });
    }

    #[test]
    fn unknown_claim_schema_never_resets_a_live_owner() {
        isolated("unknown_claim_schema_never_resets_a_live_owner", || {
            use std::os::fd::AsRawFd;
            let claims = Claims::open().unwrap();
            let need: Need = Arc::new(|| Some(0));
            let budget: Budget = Arc::new(|| 6);
            let Admission::Granted(lease) = claims.admit(&need, 4, &budget).unwrap() else { panic!("initial claim refused") };
            // SAFETY: querying the flags of this live owned descriptor.
            assert_eq!(unsafe { libc::fcntl(lease.as_raw_fd(), libc::F_GETFL) } & libc::O_ACCMODE, libc::O_RDONLY);
            for invalid in [u64::from_le_bytes(*b"vhrsv999"), 0] {
                {
                    let _lock = claims.lock().unwrap();
                    let map = os::map(&claims.fd, platform_page() as usize, true).unwrap();
                    claim_word(&map, 0).store(invalid, Release);
                }
                assert!(matches!(Claims::open(), Err(Error::Closed(_))));
                let _lock = claims.lock().unwrap();
                let map = os::map(&claims.fd, platform_page() as usize, true).unwrap();
                assert_eq!(claim_word(&map, 0).load(Acquire), invalid, "unknown bytes were rewritten");
                assert!(os::held(&claims.fd, CLAIM_WORD, CLAIM_WORD).unwrap(), "the live claim was lost");
                claim_word(&map, 0).store(CLAIM_MAGIC, Release);
            }
            drop(lease);
            assert_eq!(claims.live_bytes().unwrap(), 0);
        });
    }

    fn filled(client: &Client, name: &str, ask: &str) -> Held {
        let held = client.open_layout(name, vec![(0, platform_page())], ask).unwrap();
        held.fill(&[0], |dsts| {
            let tensorfs::io::Dst::Host { ptr, len } = dsts[0].1 else { unreachable!() };
            // SAFETY: this fill exclusively owns its mapped destination.
            unsafe { std::ptr::write_bytes(ptr, 1, len) };
            Ok(())
        }).unwrap();
        held
    }

    #[test]
    fn pressure_one_event_continues_after_async_layout_releases() {
        isolated("pressure_one_event_continues_after_async_layout_releases", || {
            let page = platform_page();
            let files = Arc::new(Mutex::new(Vec::<(Owned, u64)>::new()));
            let observed = files.clone();
            let need: Need = Arc::new(move || Some(2 * page));
            let budget: Budget = Arc::new(move || {
                let resident: u64 = observed.lock().iter().map(|(fd, data)| os::allocated(fd).saturating_sub(*data)).sum();
                i128::from(2 * page) - i128::from(resident)
            });
            let tier = Tier::bind_with(need.clone(), budget).unwrap().unwrap();
            let client = Client::open(os::euid(), need.clone()).unwrap();
            // The absent endpoint models a departed user: completion is asynchronous on the
            // production ask thread, and real memfd data must be punched before headroom rises.
            let first = filled(&client, "first", "departed-user");
            let second = filled(&client, "second", "departed-user");
            files.lock().extend([(os::reopen(&first.fd, true).unwrap(), first.data), (os::reopen(&second.fd, true).unwrap(), second.data)]);
            let event = os::event().unwrap();
            tier.watch(Pressure::Events(os::dup(&event).unwrap()), need).unwrap();
            os::signal(&event); // Exactly one external pressure event.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let (lock, changed) = &*tier.state;
            let mut state = lock.lock();
            while state.layouts.values().any(Layout::holds) {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                assert!(!remaining.is_zero() && !changed.wait_for(&mut state, remaining).timed_out(),
                    "one event left ready layouts after asynchronous completion: {:?}", state.layouts.iter().filter(|(_, l)| l.holds()).map(|(d, _)| d).collect::<Vec<_>>());
            }
            assert_eq!(state.releases, 2);
            drop(state);
            assert_eq!(tier.budget.as_ref()(), i128::from(2 * page));
        });
    }

    #[test]
    fn pressure_pending_reclaim_does_not_raise_host_short() {
        isolated("pressure_pending_reclaim_does_not_raise_host_short", || {
            use std::os::fd::AsRawFd;
            let need: Need = Arc::new(|| Some(1));
            let tier = Tier::bind_with(need.clone(), Arc::new(|| 0)).unwrap().unwrap();
            let client = Client::open(os::euid(), need.clone()).unwrap();
            let listener = os::listen(os::euid(), "stalled-user").unwrap().unwrap();
            let _held = filled(&client, "pending", "stalled-user");
            answer(&tier.state, &need, &tier.budget, &tier.claims);
            let mut incoming = libc::pollfd { fd: listener.as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: poll only inspects the live listener descriptor; timeout only fails hangs.
            assert_eq!(unsafe { libc::poll(&mut incoming, 1, 5000) }, 1);
            let (ask, _) = listener.accept().unwrap();
            ask.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
            assert!(os::recv(&ask).unwrap().is_some());
            // The first ask remains unanswered. A later event must return promptly without
            // escalating to executor termination while already-requested reclaim is pending.
            answer(&tier.state, &need, &tier.budget, &tier.claims);
            let mut short = libc::pollfd { fd: tier.short.as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: poll only inspects the live event descriptor and does not wait.
            let escalated = unsafe { libc::poll(&mut short, 1, 0) };
            os::send(&ask, &os::encode(&serde_json::json!({"freed": 0})).unwrap(), None).unwrap();
            assert_eq!(escalated, 0, "pending tier reclaim must settle before host-short escalation");
        });
    }

}
