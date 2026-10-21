//! The machine's RAM tier: one host layout per layout digest, a ready word per region, mapped
//! read-only by executors, held by whichever participant binds the machine's tier name
//! (TensorD when present). An executor fills a region it misses through `io::read` under the
//! region's write lock, and copies from a ready region under its read lock, so a region in
//! flight is locked. Under pressure (an event, or a headroom check before a fill) the tier
//! gives back, one layout per rung: orphaned layouts (least recently used), then layouts with
//! no declared use, then the furthest declared; never a region in flight.

use std::collections::HashMap;
use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use platform::handles::Owned;
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Serialize, Deserialize)]
enum Call {
    /// Regions as (offset, span) in the layout.
    Open { digest: String, regions: Vec<(u64, u64)> },
    /// This process's next use of the layout, in steps; `None`: no declared use.
    Use { digest: String, distance: Option<u32> },
    Close { digest: String },
    /// Make `bytes` of host headroom before a fill, giving back in order.
    Room { bytes: u64 },
}

#[derive(Debug, Serialize, Deserialize)]
enum Reply {
    Done,
    /// The layout's data starts at this offset of the file handed with the reply.
    Opened { data: u64 },
    Room { ok: bool },
}

struct Layout {
    fd: Owned,
    words: Map,
    regions: Vec<(u64, u64)>,
    data: u64,
    /// Connection → its declared distance.
    users: HashMap<u64, Option<u32>>,
    used: u64,
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
                let in_flight = ready && os::held(&self.fd, self.data + off, *span).unwrap_or(true);
                (*span, ready, in_flight)
            })
            .collect();
        HostLayout { digest: digest.to_string(), regions, users: self.users.values().copied().collect(), used: self.used }
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
}

impl State {
    /// Applies the host decision for `trigger`; returns the bytes given back.
    fn relieve(&mut self, trigger: HostTrigger) -> u64 {
        let facts: Vec<HostLayout> = self.layouts.iter().map(|(d, l)| l.facts(d)).collect();
        let actions = host_decide(&facts, trigger);
        let mut gave = 0;
        for a in actions {
            match a {
                HostAction::Release { digest, region } => gave += self.layouts.get(&digest).map_or(0, |l| l.release(region)),
                HostAction::Forget(digest) => {
                    self.layouts.remove(&digest);
                }
            }
        }
        if gave > 0 {
            self.releases += 1;
        }
        gave
    }
}

/// The machine's RAM-tier service, served from this process while it lives.
pub struct Tier {
    state: Arc<(Mutex<State>, Condvar)>,
}

impl Tier {
    /// Binds the machine's tier name and serves it; `None` when another participant does.
    pub fn serve(pressure: Pressure) -> Result<Option<Tier>> {
        let Some(tier) = Tier::bind()? else { return Ok(None) };
        if matches!(pressure, Pressure::Headroom) {
            return Ok(Some(tier));
        }
        let state = tier.state.clone();
        std::thread::Builder::new()
            .name("varena-pressure".into())
            .spawn(move || loop {
                let woke = match &pressure {
                    Pressure::Host(p) => p.wait().map_err(Error::from),
                    Pressure::Events(e) => os::await_event(e),
                    Pressure::Headroom => return,
                };
                if woke.is_err() {
                    return;
                }
                let (lock, cv) = &*state;
                lock.lock().relieve(HostTrigger::Pressure);
                cv.notify_all();
            })
            .map_err(|e| Error::io("spawn pressure", e))?;
        Ok(Some(tier))
    }

    fn bind() -> Result<Option<Tier>> {
        let Some(listener) = os::listen(NAME)? else { return Ok(None) };
        let state = Arc::new((Mutex::new(State::default()), Condvar::new()));
        let s = state.clone();
        std::thread::Builder::new()
            .name("varena-tier".into())
            .spawn(move || {
                for stream in os::incoming(listener) {
                    let s = s.clone();
                    let _ = std::thread::Builder::new().name("varena-tier-conn".into()).spawn(move || serve(stream, &s));
                }
            })
            .map_err(|e| Error::io("spawn tier", e))?;
        Ok(Some(Tier { state }))
    }

    /// The layouts holding any ready region (status).
    pub fn held(&self) -> Vec<String> {
        let mut out: Vec<String> = self.state.0.lock().layouts.iter().filter(|(_, l)| l.holds()).map(|(d, _)| d.clone()).collect();
        out.sort();
        out
    }

    /// Blocks until the tier has given back `n` rungs since it started; returns the count.
    pub fn await_releases(&self, n: u64) -> u64 {
        let (lock, cv) = &*self.state;
        let mut s = lock.lock();
        while s.releases < n {
            cv.wait(&mut s);
        }
        s.releases
    }
}

fn serve(stream: Stream, state: &(Mutex<State>, Condvar)) {
    // Only this user's processes: an abstract name has no permissions of its own.
    if os::peer(&stream).is_err() {
        return;
    }
    let (lock, cv) = state;
    let conn = {
        let mut s = lock.lock();
        s.clock += 1;
        s.clock
    };
    while let Ok(Some((bytes, _))) = os::recv(&stream) {
        let Ok(call) = os::decode::<Call>(&bytes) else { break };
        let mut s = lock.lock();
        s.clock += 1;
        let clock = s.clock;
        let (reply, out) = match call {
            Call::Open { digest, regions } => match open(&mut s, &digest, regions) {
                Ok((data, fd)) => {
                    if let Some(l) = s.layouts.get_mut(&digest) {
                        l.users.insert(conn, None);
                        l.used = clock;
                    }
                    (Reply::Opened { data }, Some(fd))
                }
                Err(_) => (Reply::Room { ok: false }, None),
            },
            Call::Use { digest, distance } => {
                if let Some(l) = s.layouts.get_mut(&digest) {
                    l.users.insert(conn, distance);
                    l.used = clock;
                }
                (Reply::Done, None)
            }
            Call::Close { digest } => {
                if let Some(l) = s.layouts.get_mut(&digest) {
                    l.users.remove(&conn);
                }
                (Reply::Done, None)
            }
            Call::Room { bytes } => {
                let headroom = u64::try_from(platform::contain::host_memory::read().available).unwrap_or(0);
                let gave = if headroom >= bytes { 0 } else { s.relieve(HostTrigger::Room { bytes, headroom }) };
                cv.notify_all();
                (Reply::Room { ok: headroom + gave >= bytes }, None)
            }
        };
        drop(s);
        if os::encode(&reply).and_then(|b| os::send(&stream, &(b, out))).is_err() {
            break;
        }
    }
    let mut s = lock.lock();
    s.layouts.values_mut().for_each(|l| {
        l.users.remove(&conn);
    });
}

/// The layout's file and data offset, made when absent.
fn open(s: &mut State, digest: &str, regions: Vec<(u64, u64)>) -> Result<(u64, Owned)> {
    if let Some(l) = s.layouts.get(digest) {
        return Ok((l.data, os::dup(&l.fd)?));
    }
    let page = platform_page();
    let data = ((regions.len() as u64 * 4).max(1)).div_ceil(page) * page;
    let end = regions.iter().map(|(o, s)| o + s).max().unwrap_or(0);
    let fd = os::memfd(&format!("varena-tier-{digest}"))?;
    os::resize(&fd, data + end)?;
    let words = os::map(&fd, data as usize, true)?;
    let handed = os::dup(&fd)?;
    s.layouts.insert(digest.to_string(), Layout { fd, words, regions, data, users: HashMap::new(), used: 0 });
    Ok((data, handed))
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
    /// Regions page-locked for the copy engine (once each, while ready).
    registered: parking_lot::Mutex<std::collections::HashSet<u32>>,
}

impl Held {
    /// Page-locks region `r` once it is ready, so fills from it are pinned copies; a region
    /// the tier gave back is unlocked first.
    pub fn pin(&self, r: u32, register: impl FnOnce(u64, u64) -> Result<()>, unregister: impl FnOnce(u64) -> Result<()>) -> Result<()> {
        let (off, span) = self.region(r)?;
        let ptr = self.ro.base as u64 + self.data + off;
        let ready = self.ready(r).is_some_and(|w| w.load(Acquire) == 1);
        let mut registered = self.registered.lock();
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
    pub fn unpin(&self, mut unregister: impl FnMut(u64) -> Result<()>) {
        for r in self.registered.lock().drain() {
            if let Ok((off, _)) = self.region(r) {
                let _ = unregister(self.ro.base as u64 + self.data + off);
            }
        }
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

    /// Copies region `r` into `dst` if it is ready, under its read lock; false when it is not.
    pub fn read(&self, r: u32, dst: tensorfs::io::Dst) -> Result<bool> {
        let (off, span) = self.region(r)?;
        os::share(&self.fd, self.data + off, span, true)?;
        let copied = (|| {
            if self.ready(r).is_none_or(|w| w.load(Acquire) != 1) {
                return Ok(false);
            }
            let len = dst.len() as u64;
            if len > span {
                return Err(Error::Invalid("tier read past its region".into()));
            }
            // SAFETY: the region is ready and read-locked (nobody frees it), inside our mapping.
            let src = unsafe { self.ro.base.add((self.data + off) as usize) };
            let copy = tensorfs::io::Copy { src: tensorfs::io::Src::Host(src), dst };
            // SAFETY: the caller owns `dst` exclusively for this call.
            unsafe { tensorfs::io::read(&[copy], &tensorfs::io::ReadOpts::default()) }?;
            Ok(true)
        })();
        os::unlock(&self.fd, self.data + off, span)?;
        copied
    }

    /// Fills region `r` from `fill` (which writes the region's host bytes) unless it is ready.
    pub fn fill(&self, r: u32, fill: impl FnOnce(tensorfs::io::Dst) -> Result<()>) -> Result<()> {
        let (off, span) = self.region(r)?;
        os::lock(&self.fd, self.data + off, span, true)?;
        let done = (|| {
            if self.ready(r).is_some_and(|w| w.load(Acquire) == 1) {
                return Ok(());
            }
            let rw = os::map_at(&self.fd, self.data + off, span as usize, true)?;
            fill(tensorfs::io::Dst::Host { ptr: rw.base, len: span as usize })?;
            drop(rw);
            // The words page is the service's to write; the region lock orders this store.
            let words = os::map_at(&self.fd, 0, self.data as usize, true)?;
            // SAFETY: inside the words page, aligned to 4.
            let word = unsafe { &*(words.base.add(r as usize * 4) as *const AtomicU32) };
            word.store(1, Release);
            Ok(())
        })();
        os::unlock(&self.fd, self.data + off, span)?;
        done
    }
}

/// An executor's connection to the machine's tier.
pub(crate) struct Client {
    stream: Mutex<Stream>,
    _service: Option<Tier>,
}

impl Client {
    /// Connects to the tier, serving it from this process (with the host's pressure trigger
    /// when it arms) when nobody does.
    pub fn open() -> Result<Client> {
        let mut service = None;
        loop {
            if let Some(stream) = os::connect(NAME)? {
                return Ok(Client { stream: Mutex::new(stream), _service: service });
            }
            let mode = platform::contain::host_pressure::HostMode::Shared;
            service = match platform::contain::host_pressure::Pressure::arm(mode) {
                Ok(p) => Tier::serve(Pressure::Host(p))?,
                Err(_) => Tier::serve(Pressure::Headroom)?,
            };
        }
    }

    fn call(&self, call: &Call) -> Result<(Reply, Option<Owned>)> {
        os::call(&self.stream.lock(), call, None)
    }

    pub fn open_layout(&self, digest: &str, regions: Vec<(u64, u64)>) -> Result<Held> {
        let (reply, fd) = self.call(&Call::Open { digest: digest.into(), regions: regions.clone() })?;
        let (Reply::Opened { data }, Some(fd)) = (reply, fd) else { return Err(Error::Io("the tier refused a layout".into())) };
        let fd = os::reopen(&fd, true)?;
        let len = data + regions.iter().map(|(o, s)| o + s).max().unwrap_or(0);
        let ro = os::map(&fd, len as usize, false)?;
        Ok(Held { fd, ro, data, regions, registered: Default::default() })
    }

    pub fn use_layout(&self, digest: &str, distance: Option<u32>) -> Result<()> {
        self.call(&Call::Use { digest: digest.into(), distance }).map(|_| ())
    }

    pub fn close(&self, digest: &str) -> Result<()> {
        self.call(&Call::Close { digest: digest.into() }).map(|_| ())
    }

    /// Host headroom for `bytes`, the tier giving back in order; false when it cannot.
    pub fn room(&self, bytes: u64) -> Result<bool> {
        Ok(matches!(self.call(&Call::Room { bytes })?.0, Reply::Room { ok: true }))
    }
}
