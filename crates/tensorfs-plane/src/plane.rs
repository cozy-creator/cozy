//! The plane: weight sets, tiers, budgets, fills, resident leases and telemetry.
//!
//! One lock (`Core`) holds every book; it is held across fast driver calls (map, unmap,
//! record, enqueue) and never across I/O. Waits happen on the condvar or on one targeted event.
//! Priorities are the policy's: eviction takes unpinned, unleased, unheld regions in ascending
//! priority, then later-registered weight set first, then higher region index first (the tail
//! of the address order). Nothing here measures memory or decides what should be resident.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use tensorfs_core::meta::Meta;
use tensorfs_core::read::{ReadLease, ReadPlan};
use tensorfs_core::store::Store;

use crate::cuda::{self, AllocationProp, CUcontext, CUdevice, CUstream, CtxGuard};
use crate::cursor::Cur;
use crate::device::{self, Arena, Chunk, CopyJob, CopySrc, Copied, EvRef, EventPool, Mover, MoverCtx};
use crate::host::HostMem;
use crate::io::{Batch, Dst, IoTally, Mode, Readers, Source};
use crate::layout::{Layout, REGION_ALIGN};
use crate::{Error, Result};

/// Everything the plane is told at open. Values come from the runtime's central config.
#[derive(Debug, Clone)]
pub struct PlaneConfig {
    /// CUDA ordinals this plane serves. Empty: host tier only, libcuda never loaded.
    pub devices: Vec<i32>,
    /// Persistent reader threads.
    pub readers: usize,
    /// Copy streams (each with its own mover thread) per device.
    pub copy_streams: usize,
    /// Physical VRAM handle size. 0: one handle per region.
    pub slab_bytes: u64,
    /// Pinned bounce buffers per copy stream for device fills the host tier does not serve.
    pub staging_buffers: usize,
    pub staging_bytes: u64,
    /// O_DIRECT for pinned-tier fills of bytes the page cache does not hold.
    pub direct_io: bool,
}

impl Default for PlaneConfig {
    fn default() -> Self {
        PlaneConfig {
            devices: Vec::new(),
            readers: 8,
            copy_streams: 1,
            slab_bytes: 64 << 20,
            staging_buffers: 4,
            staging_bytes: 64 << 20,
            direct_io: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Pinned,
    Device(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WsId(pub u32);

#[derive(Debug, Clone, Default)]
pub struct Trim {
    pub freed: u64,
    pub evicted: Vec<(String, u32)>,
    pub over_budget_unreleasable: u64,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub kind: &'static str,
    pub ws: Option<String>,
    pub region: Option<u32>,
    pub device: Option<i32>,
    pub bytes: u64,
    pub ns: u64,
    pub detail: String,
}

impl Event {
    fn new(kind: &'static str, ws: Option<String>, region: Option<u32>, device: Option<i32>, bytes: u64, ns: u64, detail: impl Into<String>) -> Event {
        Event { kind, ws, region, device, bytes, ns, detail: detail.into() }
    }
}

const EVENT_RING: usize = 4096;

// ---------------------------------------------------------------- books

#[derive(Debug, Clone)]
pub(crate) enum HState {
    Absent,
    Filling,
    Ready,
    /// The fill failed; the typed cause is kept for every later reader. Like `Absent`, it
    /// holds no claim, no RAM and no budget.
    Invalid(Error),
}

pub(crate) struct HReg {
    pub state: HState,
    pub prio: i64,
    pub pin: bool,
    pub registered: bool,
    /// Cursors streaming from this region and copies in flight from it.
    pub holds: u32,
    pub pending: Vec<EvRef>,
    /// Device fills waiting for this host fill to finish (device slots).
    pub then: Vec<usize>,
}

pub(crate) enum DState {
    Absent,
    /// Submitted; the mover has not recorded its event yet.
    Queued,
    Filling(EvRef),
    Ready,
    Invalid(Error),
}

pub(crate) struct DReg {
    pub chunks: Vec<Chunk>,
    pub state: DState,
    pub prio: i64,
    pub pin: bool,
    pub live: u32,
    pub holds: u32,
    /// Release events of finished leases the device may not have reached.
    pub pending: Vec<EvRef>,
}

pub(crate) struct DevWs {
    pub base: u64,
    pub regs: Vec<DReg>,
    pub views: u64,
    /// Bumped before any unmap: a view exported under an older epoch may name unmapped pages.
    pub epoch: u64,
}

pub(crate) struct Ws {
    pub name: String,
    pub layout: Arc<Layout>,
    pub source: Arc<Source>,
    pub host: Arc<HostMem>,
    pub hregs: Vec<HReg>,
    pub devs: Vec<Option<DevWs>>,
    pub evictions: u64,
    pub evicted_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DevCounters {
    pub copies: u64,
    pub host_copy_bytes: u64,
    pub host_copy_ns: u64,
    pub disk_copy_bytes: u64,
    pub disk_copy_ns: u64,
    /// 0.3.92's copies from page-cache mappings, removed in 0.3.93: always 0, kept for callers
    /// that read them.
    pub mapped_copy_bytes: u64,
    pub mapped_copy_ns: u64,
    pub misses: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    pub trims: u64,
}

#[derive(Debug, Clone, Default)]
pub struct HostCounters {
    pub fills: u64,
    pub fill_bytes: u64,
    pub fill_ns: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    pub register_failed: u64,
    pub trims: u64,
}

/// Per weight set × device × stage tag ("" for resident acquires).
#[derive(Debug, Clone, Default)]
pub struct TagCounters {
    pub acquires: u64,
    /// Scheduled positions the consumer skipped (their ring bytes were handed on unread).
    pub skipped: u64,
    pub misses: u64,
    pub late: u64,
    pub late_bytes: u64,
    pub stall_ns: u64,
    pub streamed_bytes: u64,
}

pub(crate) enum TimingKind {
    Copy { from_host: bool },
    /// The consumer stream's wait on a late copy.
    Stall { tag: String, region: u32 },
    /// The consumer's use of a region: end of the wait to the release event.
    Compute { tag: String, region: u32 },
}

/// Per weight set × device × stage tag × region: GPU-timed facts for the cost model.
#[derive(Debug, Clone, Default)]
pub struct RegionCounters {
    pub uses: u64,
    pub compute_ns: u64,
    pub stall_ns: u64,
    pub late: u64,
}

pub(crate) struct Timing {
    pub dev: usize,
    pub ws: u32,
    pub kind: TimingKind,
    pub a: EvRef,
    pub b: EvRef,
    pub bytes: u64,
}

pub(crate) struct Core {
    /// Latched on the first ambiguous driver outcome or lost lease: every later call refuses
    /// and the runtime replaces the executor.
    pub poisoned: Option<String>,
    pub sets: Vec<Option<Ws>>,
    pub arenas: Vec<Arena>,
    pub host_budget: u64,
    pub host_used: u64,
    pub cursors: HashMap<u64, Cur>,
    pub next_cursor: u64,
    pub dstats: Vec<DevCounters>,
    pub hstats: HostCounters,
    pub tags: BTreeMap<(u32, usize, String), TagCounters>,
    pub regions: BTreeMap<(u32, usize, String, u32), RegionCounters>,
    pub events: VecDeque<Event>,
    pub events_dropped: u64,
    pub timings: VecDeque<Timing>,
}

pub(crate) struct DeviceRt {
    pub ordinal: i32,
    pub dev: CUdevice,
    pub ctx: CUcontext,
    pub uuid: String,
    pub exportable: bool,
    pub granularity: u64,
    pub events: Arc<EventPool>,
    pub movers: Vec<Mover>,
    next_mover: AtomicUsize,
}
unsafe impl Send for DeviceRt {}
unsafe impl Sync for DeviceRt {}

impl DeviceRt {
    pub fn mover(&self) -> usize {
        self.next_mover.fetch_add(1, Ordering::Relaxed) % self.movers.len()
    }
    pub fn bind(&self) -> Result<CtxGuard> {
        CtxGuard::enter(self.ctx)
    }
}

pub(crate) struct Inner {
    pub cfg: PlaneConfig,
    pub devices: Vec<DeviceRt>,
    pub readers: Arc<Readers>,
    pub tally: Arc<IoTally>,
    pub core: Mutex<Core>,
    pub cv: Condvar,
}

impl Core {
    pub fn poison(&mut self, cause: String) -> Error {
        Error::Poisoned(self.poisoned.get_or_insert(cause).clone())
    }
    /// Latch a `Poisoned` outcome reported by a copy or a driver call.
    pub fn latch(&mut self, e: &Error) {
        if let Error::Poisoned(p) = e {
            self.poison(p.clone());
        }
    }
    pub fn check(&self) -> Result<()> {
        match &self.poisoned {
            Some(p) => Err(Error::Poisoned(p.clone())),
            None => Ok(()),
        }
    }
    pub fn ws(&self, id: WsId) -> Result<&Ws> {
        self.sets.get(id.0 as usize).and_then(Option::as_ref).ok_or(Error::Closed)
    }
    pub fn ws_mut(&mut self, id: WsId) -> Result<&mut Ws> {
        self.sets.get_mut(id.0 as usize).and_then(Option::as_mut).ok_or(Error::Closed)
    }
    pub fn event(&mut self, e: Event) {
        if self.events.len() == EVENT_RING {
            self.events.pop_front();
            self.events_dropped += 1;
        }
        self.events.push_back(e);
    }
    pub fn tag(&mut self, ws: u32, dev: usize, tag: &str) -> &mut TagCounters {
        self.tags.entry((ws, dev, tag.to_string())).or_default()
    }
}

/// Fold resolved timing pairs into counters; keep the unresolved. `all: false` stops at the
/// first pair still in flight (the hot-path sweep: a query or two per release).
pub(crate) fn resolve_timings(c: &mut Core, all: bool) {
    let mut keep = VecDeque::new();
    while let Some(t) = c.timings.pop_front() {
        if !(t.b.done().unwrap_or(false)) {
            keep.push_back(t);
            if all {
                continue;
            }
            break;
        }
        let ns = (t.b.since(&t.a).unwrap_or(0.0) as f64 * 1e6) as u64;
        match &t.kind {
            TimingKind::Copy { from_host } => {
                let s = &mut c.dstats[t.dev];
                if *from_host {
                    s.host_copy_bytes += t.bytes;
                    s.host_copy_ns += ns;
                } else {
                    s.disk_copy_bytes += t.bytes;
                    s.disk_copy_ns += ns;
                }
            }
            TimingKind::Compute { tag, region } => {
                let k = (t.ws, t.dev, tag.clone(), *region);
                let rc = c.regions.entry(k).or_default();
                rc.uses += 1;
                rc.compute_ns += ns;
            }
            TimingKind::Stall { tag, region } => {
                let tag = tag.clone();
                c.tag(t.ws, t.dev, &tag).stall_ns += ns;
                c.regions.entry((t.ws, t.dev, tag.clone(), *region)).or_default().stall_ns += ns;
                if ns > 0 {
                    let name = c.sets.get(t.ws as usize).and_then(Option::as_ref).map(|w| w.name.clone());
                    c.event(Event::new("late", name, None, None, t.bytes, ns, tag));
                }
            }
        }
    }
    keep.append(&mut c.timings);
    c.timings = keep;
}

/// Commit `Filling -> Ready` when the copy event has completed.
fn commit(reg: &mut DReg) -> Result<()> {
    if let DState::Filling(ev) = &reg.state {
        if ev.done()? {
            reg.state = DState::Ready;
        }
    }
    Ok(())
}

pub(crate) fn prune(pending: &mut Vec<EvRef>) {
    pending.retain(|e| !e.done().unwrap_or(false));
}

fn sync_all(pending: &mut Vec<EvRef>) -> Result<()> {
    for e in pending.drain(..) {
        e.sync()?;
    }
    Ok(())
}

// ---------------------------------------------------------------- the plane

pub(crate) fn devws(c: &mut Core, ws: WsId, d: usize) -> Result<&mut DevWs> {
    if let Some(p) = &c.poisoned {
        return Err(Error::Poisoned(p.clone()));
    }
    let w = c.ws_mut(ws)?;
    if w.devs[d].is_none() {
        let base = device::reserve_va(w.layout.nbytes)?;
        let n = w.layout.regions.len();
        w.devs[d] = Some(DevWs {
            base,
            regs: (0..n)
                .map(|_| DReg {
                    chunks: Vec::new(),
                    state: DState::Absent,
                    prio: 0,
                    pin: false,
                    live: 0,
                    holds: 0,
                    pending: Vec::new(),
                })
                .collect(),
            views: 0,
            epoch: 1,
        });
    }
    Ok(w.devs[d].as_mut().unwrap())
}


impl Inner {
    /// Evict device regions until `need` mapped bytes are freed, taking only candidates below
    /// `below` (None: any unpinned priority). Records blockers.
    pub(crate) fn evict_dev(self: &Arc<Self>, c: &mut Core, d: usize, need: u64, below: Option<i64>, t: &mut Trim) -> Result<u64> {
        let mut cands: Vec<(i64, u32, u32)> = Vec::new();
        for (wi, w) in c.sets.iter_mut().enumerate() {
            let Some(w) = w.as_mut() else { continue };
            let Some(dw) = w.devs[d].as_mut() else { continue };
            for (r, reg) in dw.regs.iter_mut().enumerate() {
                if reg.chunks.is_empty() || reg.pin || below.is_some_and(|b| reg.prio >= b) {
                    continue;
                }
                commit(reg)?;
                if reg.live > 0 || reg.holds > 0 || matches!(reg.state, DState::Queued) {
                    t.blockers.push(format!("{}#{r}@dev{}", w.name, self.devices[d].ordinal));
                    continue;
                }
                cands.push((reg.prio, wi as u32, r as u32));
            }
        }
        cands.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
        let mut freed = 0u64;
        for (_, wi, r) in cands {
            if freed >= need {
                break;
            }
            freed += self.unmap_region(c, WsId(wi), d, r, true)?;
            let name = c.ws(WsId(wi))?.name.clone();
            t.evicted.push((name, r));
        }
        Ok(freed)
    }

    /// Unmap one region (its leases are clear): wait for its pending events, unmap, recycle.
    /// `evict` counts it as an eviction (and emits the event); closing a weight set does not.
    pub(crate) fn unmap_region(self: &Arc<Self>, c: &mut Core, ws: WsId, d: usize, r: u32, evict: bool) -> Result<u64> {
        let ordinal = self.devices[d].ordinal;
        let w = c.ws_mut(ws)?;
        let region = w.layout.regions[r as usize].clone();
        let name = w.name.clone();
        let dw = w.devs[d].as_mut().expect("mapped region has a device reservation");
        let reg = &mut dw.regs[r as usize];
        if let DState::Filling(ev) = &reg.state {
            ev.sync()?;
        }
        sync_all(&mut reg.pending)?;
        dw.epoch += 1;
        if let Err(e) = device::unmap_span(dw.base + region.offset, region.span) {
            return Err(c.poison(format!("unmap of {name}#{r} failed: {e}")));
        }
        let w = c.ws_mut(ws)?;
        let reg = &mut w.devs[d].as_mut().unwrap().regs[r as usize];
        let chunks = std::mem::take(&mut reg.chunks);
        reg.state = DState::Absent;
        c.arenas[d].mapped -= region.span;
        c.arenas[d].put_idle(chunks);
        if evict {
            let w = c.ws_mut(ws)?;
            w.evictions += 1;
            w.evicted_bytes += region.span;
            c.dstats[d].evictions += 1;
            c.dstats[d].evicted_bytes += region.span;
            c.event(Event::new("evicted", Some(name), Some(r), Some(ordinal), region.span, 0, ""));
        }
        Ok(region.span)
    }

    pub(crate) fn evict_host(self: &Arc<Self>, c: &mut Core, need: u64, below: Option<i64>, t: &mut Trim) -> Result<u64> {
        let mut cands: Vec<(i64, u32, u32)> = Vec::new();
        for (wi, w) in c.sets.iter_mut().enumerate() {
            let Some(w) = w.as_mut() else { continue };
            for (r, h) in w.hregs.iter_mut().enumerate() {
                if !matches!(h.state, HState::Ready) || h.pin || below.is_some_and(|b| h.prio >= b) {
                    continue;
                }
                prune(&mut h.pending);
                if h.holds > 0 {
                    t.blockers.push(format!("{}#{r}@pinned", w.name));
                    continue;
                }
                cands.push((h.prio, wi as u32, r as u32));
            }
        }
        cands.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
        let mut freed = 0u64;
        for (_, wi, r) in cands {
            if freed >= need {
                break;
            }
            let n = self.release_host(c, WsId(wi), r, true)?;
            freed += n;
            t.freed += n;
            let name = c.ws(WsId(wi))?.name.clone();
            t.evicted.push((name, r));
        }
        Ok(freed)
    }

    /// Let a Ready host region go: wait for copies reading it, unregister, end this process's
    /// claim. Its RAM is punched unless another process still claims it (that process counts
    /// it). Returns the span this plane's budget regains.
    pub(crate) fn release_host(self: &Arc<Self>, c: &mut Core, ws: WsId, r: u32, evict: bool) -> Result<u64> {
        let first = self.devices.first().map(|d| d.ctx);
        let w = c.ws_mut(ws)?;
        let region = w.layout.regions[r as usize].clone();
        let host = w.host.clone();
        let name = w.name.clone();
        let h = &mut w.hregs[r as usize];
        debug_assert!(matches!(h.state, HState::Ready) && h.holds == 0);
        sync_all(&mut h.pending)?;
        if h.registered {
            let _g = first.map(CtxGuard::enter).transpose()?;
            host.unregister(region.offset)?;
            h.registered = false;
        }
        let punched = host.release(r, region.offset, region.span)?;
        h.state = HState::Absent;
        if evict {
            w.evictions += 1;
            w.evicted_bytes += region.span;
        }
        c.host_used -= region.span;
        if evict {
            c.hstats.evictions += 1;
            c.hstats.evicted_bytes += region.span;
            let detail = if punched { "pinned" } else { "pinned; claimed elsewhere, not punched" };
            c.event(Event::new("evicted", Some(name), Some(r), None, region.span, 0, detail));
        }
        Ok(region.span)
    }

    pub(crate) fn want_host(self: &Arc<Self>, c: &mut Core, ws: WsId, regions: &[u32], prio: i64, pin: bool) -> Result<()> {
        let w = c.ws_mut(ws)?;
        let mut need = 0u64;
        for &r in regions {
            let span = w.layout.regions[r as usize].span;
            let h = &mut w.hregs[r as usize];
            h.prio = prio;
            h.pin = pin;
            if matches!(h.state, HState::Absent | HState::Invalid(_)) {
                need += span;
            }
        }
        if need > c.host_budget {
            return Err(Error::BelowFloor { pool: "pinned", need, budget: c.host_budget });
        }
        if c.host_used + need > c.host_budget {
            let deficit = c.host_used + need - c.host_budget;
            let mut t = Trim::default();
            let freed = self.evict_host(c, deficit, Some(prio), &mut t)?;
            if freed < deficit {
                return Err(Error::Shortfall {
                    pool: "pinned",
                    need,
                    available: c.host_budget.saturating_sub(c.host_used),
                    blockers: t.blockers,
                });
            }
        }
        for &r in regions {
            self.fill_host(c, ws, r, false)?;
        }
        Ok(())
    }

    /// Claim and start the host fill of one region (no-op unless Absent/Invalid). Adopts bytes
    /// another process already made Ready; `adopt_only` claims nothing else.
    pub(crate) fn fill_host(self: &Arc<Self>, c: &mut Core, ws: WsId, r: u32, adopt_only: bool) -> Result<()> {
        let w = c.ws_mut(ws)?;
        if !matches!(w.hregs[r as usize].state, HState::Absent | HState::Invalid(_)) {
            return Ok(());
        }
        let region = w.layout.regions[r as usize].clone();
        let (host, source, name) = (w.host.clone(), w.source.clone(), w.name.clone());
        host.claim(r)?;
        let adopted = host.ready(r);
        if adopt_only && !adopted {
            host.release(r, region.offset, region.span)?;
            return Ok(());
        }
        w.hregs[r as usize].state = HState::Filling;
        c.host_used += region.span;
        let inner = self.clone();
        let t0 = std::time::Instant::now();
        let host2 = host.clone();
        let done = move |res: Result<()>| {
            let mut registered = false;
            if res.is_ok() {
                if let Some(dev) = inner.devices.first() {
                    let reg = CtxGuard::enter(dev.ctx).and_then(|_g| {
                        host2.register(region.offset, crate::layout::align_up(region.nbytes, crate::layout::PART_ALIGN))
                    });
                    registered = reg.is_ok();
                    if let Err(e) = reg {
                        let mut c = inner.core.lock().unwrap();
                        c.hstats.register_failed += 1;
                        c.event(Event::new("register_failed", Some(name.clone()), Some(r), None, region.nbytes, 0, e.to_string()));
                    }
                }
                host2.set_ready(r, true);
            }
            // A failed fill holds nothing: its partial pages go unless another process claims
            // the region (and fills or holds it).
            let let_go = match &res {
                Ok(()) => Ok(false),
                Err(_) => host2.release(r, region.offset, region.span),
            };
            let mut c = inner.core.lock().unwrap();
            if let Err(e) = let_go {
                c.event(Event::new("release_failed", Some(name.clone()), Some(r), None, region.span, 0, e.to_string()));
            }
            let ns = t0.elapsed().as_nanos() as u64;
            let Some(Some(w)) = c.sets.get_mut(ws.0 as usize) else { return };
            let h = &mut w.hregs[r as usize];
            let then = std::mem::take(&mut h.then);
            match &res {
                Ok(()) => {
                    h.state = HState::Ready;
                    h.registered = registered;
                }
                Err(e) => h.state = HState::Invalid(e.clone()),
            }
            if res.is_err() {
                c.host_used -= region.span;
            } else if !adopted {
                c.hstats.fills += 1;
                c.hstats.fill_bytes += region.nbytes;
                c.hstats.fill_ns += ns;
            }
            let (kind, detail) = match &res {
                Ok(()) => ("fill_done", if adopted { "pinned<-adopted" } else { "pinned<-disk" }.to_string()),
                Err(e) => ("fill_failed", e.to_string()),
            };
            c.event(Event::new(kind, Some(name.clone()), Some(r), None, region.nbytes, ns, detail));
            for d in then {
                if let Err(e) = inner.submit_dev_fill(&mut c, ws, d, r) {
                    c.latch(&e);
                    if let Some(Some(w)) = c.sets.get_mut(ws.0 as usize) {
                        if let Some(dw) = w.devs[d].as_mut() {
                            dw.regs[r as usize].state = DState::Invalid(e);
                        }
                    }
                }
            }
            drop(c);
            inner.cv.notify_all();
        };
        if adopted || region.items.is_empty() {
            self.readers.submit(false, move || done(Ok(())));
            return Ok(());
        }
        let batch = Batch::new(region.items.len(), done);
        for it in region.items.iter().cloned() {
            let (source, batch, tally) = (source.clone(), batch.clone(), self.tally.clone());
            // SAFETY: the region's host range is exclusively this fill's until it completes
            // (state Filling; the OFD fill lock keeps other processes from punching it).
            let dst = Dst(unsafe { host.base().add(it.offset as usize) });
            self.readers.submit(false, move || {
                let dst = dst;
                let r = unsafe { source.read_item(&it, dst.0, Mode::Cache, &tally) };
                // Drop the source before signalling: the waiter may then close the last
                // weight set, and its lease must end then, not when this thread unwinds.
                drop(source);
                batch.finish(r);
            });
        }
        Ok(())
    }


    /// Map + fill `regions` on device slot `d`. The caller binds the device context.
    pub(crate) fn want_dev(self: &Arc<Self>, c: &mut Core, ws: WsId, d: usize, regions: &[u32], prio: i64, pin: bool) -> Result<()> {
        let layout = c.ws(ws)?.layout.clone();
        let mut need = 0u64;
        {
            let dw = devws(c, ws, d)?;
            for &r in regions {
                let reg = &mut dw.regs[r as usize];
                reg.prio = prio;
                reg.pin = pin;
                if reg.chunks.is_empty() {
                    need += layout.regions[r as usize].span;
                }
            }
        }
        let a = &c.arenas[d];
        if need > a.budget {
            return Err(Error::BelowFloor { pool: "vram", need, budget: a.budget });
        }
        if a.mapped + need > a.budget {
            let deficit = a.mapped + need - a.budget;
            let mut t = Trim::default();
            let freed = self.evict_dev(c, d, deficit, Some(prio), &mut t)?;
            if freed < deficit {
                return Err(Error::Shortfall {
                    pool: "vram",
                    need,
                    available: c.arenas[d].budget.saturating_sub(c.arenas[d].mapped),
                    blockers: t.blockers,
                });
            }
        }
        for &r in regions {
            self.map_region(c, ws, d, r)?;
            let reg = &mut c.ws_mut(ws)?.devs[d].as_mut().unwrap().regs[r as usize];
            commit(reg)?;
            if matches!(reg.state, DState::Absent | DState::Invalid(_)) {
                self.submit_dev_fill(c, ws, d, r)?;
            }
        }
        Ok(())
    }

    pub(crate) fn map_region(self: &Arc<Self>, c: &mut Core, ws: WsId, d: usize, r: u32) -> Result<()> {
        let span = c.ws(ws)?.layout.regions[r as usize].span;
        let off = c.ws(ws)?.layout.regions[r as usize].offset;
        if !devws(c, ws, d)?.regs[r as usize].chunks.is_empty() {
            return Ok(());
        }
        let sizes = c.arenas[d].chunk_sizes(span);
        let chunks = c.arenas[d].take(&sizes)?;
        let base = devws(c, ws, d)?.base;
        if let Err(e) = device::map_span(base + off, &chunks, self.devices[d].dev) {
            if let Error::Poisoned(p) = e {
                return Err(c.poison(p));
            }
            c.arenas[d].put_idle(chunks);
            return Err(e);
        }
        c.arenas[d].mapped += span;
        devws(c, ws, d)?.regs[r as usize].chunks = chunks;
        Ok(())
    }

    /// Queue the copy that fills region `r` on device slot `d` (mapped) from the pinned tier when
    /// it holds the region, after it when it is filling, else from disk through staging.
    pub(crate) fn submit_dev_fill(self: &Arc<Self>, c: &mut Core, ws: WsId, d: usize, r: u32) -> Result<()> {
        c.check()?;
        let w = c.ws_mut(ws)?;
        let region = w.layout.regions[r as usize].clone();
        let hstate = w.hregs[r as usize].state.clone();
        if matches!(hstate, HState::Filling) {
            w.hregs[r as usize].then.push(d);
            w.devs[d].as_mut().unwrap().regs[r as usize].state = DState::Queued;
            return Ok(());
        }
        let from_host = matches!(hstate, HState::Ready);
        let src = if from_host {
            w.hregs[r as usize].holds += 1;
            // SAFETY: inside the live mapping; the hold keeps the region from eviction.
            CopySrc::Host(unsafe { w.host.base().add(region.offset as usize) })
        } else {
            CopySrc::Disk {
                source: w.source.clone(),
                items: region.items.clone(),
                base: region.offset,
            }
        };
        let dw = w.devs[d].as_mut().unwrap();
        dw.regs[r as usize].state = DState::Queued;
        let dst = dw.base + region.offset;
        let inner = self.clone();
        let name = w.name.clone();
        let job = CopyJob {
            waits: Vec::new(),
            dst,
            len: region.nbytes,
            src,
            done: Box::new(move |res: Result<Copied>| {
                let mut c = inner.core.lock().unwrap();
                let ordinal = inner.devices[d].ordinal;
                let Some(Some(w)) = c.sets.get_mut(ws.0 as usize) else { return };
                if from_host {
                    let h = &mut w.hregs[r as usize];
                    h.holds -= 1;
                    if let Ok(cp) = &res {
                        prune(&mut h.pending);
                        h.pending.push(cp.ready.clone());
                    }
                }
                let Some(dw) = w.devs[d].as_mut() else { return };
                let reg = &mut dw.regs[r as usize];
                match res {
                    Ok(cp) => {
                        reg.state = DState::Filling(cp.ready.clone());
                        c.dstats[d].copies += 1;
                        c.timings.push_back(Timing {
                            dev: d,
                            ws: ws.0,
                            kind: TimingKind::Copy { from_host },
                            a: cp.start,
                            b: cp.ready,
                            bytes: cp.bytes,
                        });
                    }
                    Err(e) => {
                        reg.state = DState::Invalid(e.clone());
                        c.latch(&e);
                        c.event(Event::new("fill_failed", Some(name), Some(r), Some(ordinal), region.nbytes, 0, e.to_string()));
                    }
                }
                drop(c);
                inner.cv.notify_all();
            }),
        };
        let m = self.devices[d].mover();
        if let Err(e) = self.devices[d].movers[m].submit(job) {
            let w = c.ws_mut(ws)?;
            if from_host {
                w.hregs[r as usize].holds -= 1;
            }
            w.devs[d].as_mut().unwrap().regs[r as usize].state = DState::Invalid(e.clone());
            return Err(e);
        }
        Ok(())
    }

    /// A stream-scoped lease on a resident region. A region not resident is filled on demand
    /// (a miss) at its last wanted priority. No host block unless the fill is not yet queued.
    pub(crate) fn acquire_home(self: &Arc<Self>, ws: WsId, d: usize, r: u32, stream: u64, tag: &str, prio: Option<i64>) -> Result<Lease> {
        let dev = &self.devices[d];
        let _g = dev.bind()?;
        let mut c = self.core.lock().unwrap();
        let region = c.ws(ws)?.layout.region(r)?.clone();
        {
            let dw = devws(&mut c, ws, d)?;
            let reg = &mut dw.regs[r as usize];
            if let Some(p) = prio {
                reg.prio = p;
            }
            commit(reg)?;
            if matches!(reg.state, DState::Absent) {
                let prio = reg.prio;
                let pin = reg.pin;
                c.dstats[d].misses += 1;
                c.tag(ws.0, d, tag).misses += 1;
                self.want_dev(&mut c, ws, d, &[r], prio, pin)?;
            }
        }
        let t0 = std::time::Instant::now();
        loop {
            let dw = devws(&mut c, ws, d)?;
            match &dw.regs[r as usize].state {
                DState::Queued => c = self.cv.wait(c).unwrap(),
                DState::Invalid(e) => return Err(e.clone()),
                DState::Absent => return Err(Error::Invalid(format!("region {r} was evicted before use"))),
                _ => break,
            }
        }
        let host_wait = t0.elapsed().as_nanos() as u64;
        let base = devws(&mut c, ws, d)?.base;
        let pending = match &devws(&mut c, ws, d)?.regs[r as usize].state {
            DState::Filling(ev) => Some(ev.clone()),
            _ => None,
        };
        let start = self.begin_use(&mut c, d, ws, r, tag, stream, pending.as_ref(), host_wait, region.nbytes)?;
        devws(&mut c, ws, d)?.regs[r as usize].live += 1;
        Ok(Lease {
            inner: self.clone(),
            ptr: base + region.offset,
            nbytes: region.nbytes,
            region: r,
            ring_offset: None,
            ws,
            dev: d,
            tag: tag.to_string(),
            start: Mutex::new(Some(start)),
            fill: pending,
            kind: LeaseKind::Home,
            released: AtomicBool::new(false),
        })
    }

    /// Make the consumer stream wait for `ready` (no host block), measuring a late arrival
    /// with a timing pair around the wait, then mark the start of the use. Returns that mark.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_use(
        &self,
        c: &mut Core,
        d: usize,
        ws: WsId,
        r: u32,
        tag: &str,
        stream: u64,
        ready: Option<&EvRef>,
        host_wait: u64,
        nbytes: u64,
    ) -> Result<EvRef> {
        let dev = &self.devices[d];
        let s = stream as CUstream;
        let late = match ready {
            Some(ev) => !ev.done()?,
            None => false,
        };
        let a = if late { Some(dev.events.get(true)?) } else { None };
        if let Some(a) = &a {
            a.record(s)?;
        }
        if let Some(ev) = ready {
            let drv = cuda::driver()?;
            // SAFETY: consumer stream of this context; the event is live.
            cuda::check("cuStreamWaitEvent", unsafe { (drv.stream_wait_event)(s, ev.raw(), 0) })?;
        }
        let start = dev.events.get(true)?;
        start.record(s)?;
        let tc = c.tag(ws.0, d, tag);
        tc.acquires += 1;
        tc.stall_ns += host_wait;
        if late {
            tc.late += 1;
            tc.late_bytes += nbytes;
            c.regions.entry((ws.0, d, tag.to_string(), r)).or_default().late += 1;
        }
        if let Some(a) = a {
            c.timings.push_back(Timing {
                dev: d,
                ws: ws.0,
                kind: TimingKind::Stall { tag: tag.to_string(), region: r },
                a,
                b: start.clone(),
                bytes: nbytes,
            });
        }
        Ok(start)
    }

    pub(crate) fn view(self: &Arc<Self>, ws: WsId, d: usize) -> Result<View> {
        let _g = self.devices[d].bind()?;
        let mut c = self.core.lock().unwrap();
        let nbytes = c.ws(ws)?.layout.nbytes;
        let dw = devws(&mut c, ws, d)?;
        dw.views += 1;
        Ok(View {
            inner: self.clone(),
            ws,
            dev: d,
            ptr: dw.base,
            nbytes,
            epoch: dw.epoch,
        })
    }
}

pub struct Plane {
    pub(crate) inner: Arc<Inner>,
}

impl Plane {
    pub fn open(cfg: PlaneConfig) -> Result<Plane> {
        let readers = Arc::new(Readers::new(cfg.readers));
        let tally = Arc::new(IoTally::default());
        let mut devices = Vec::new();
        let mut arenas = Vec::new();
        for &ordinal in &cfg.devices {
            let (dev, ctx) = cuda::primary_context(ordinal)?;
            let _g = CtxGuard::enter(ctx)?;
            if cuda::attribute(dev, cuda::ATTR_VMM_SUPPORTED)? != 1 {
                return Err(Error::Invalid(format!("device {ordinal} has no virtual memory management")));
            }
            let exportable = cuda::attribute(dev, cuda::ATTR_POSIX_FD_SUPPORTED)? == 1;
            let prop = AllocationProp::device(dev, exportable);
            let d = cuda::driver()?;
            let mut g = 0usize;
            // SAFETY: out-param.
            cuda::check("cuMemGetAllocationGranularity", unsafe {
                (d.mem_get_allocation_granularity)(&mut g, &prop, cuda::MEM_GRANULARITY_MINIMUM)
            })?;
            let g = g as u64;
            if g == 0 || !REGION_ALIGN.is_multiple_of(g) || !cfg.slab_bytes.is_multiple_of(REGION_ALIGN) {
                return Err(Error::Invalid(format!(
                    "device {ordinal}: granularity {g} and slab {} must divide into {REGION_ALIGN}",
                    cfg.slab_bytes
                )));
            }
            let events = Arc::new(EventPool::new(ctx));
            let movers = (0..cfg.copy_streams.max(1))
                .map(|i| {
                    Mover::start(
                        format!("tfs-plane-copy-{ordinal}.{i}"),
                        MoverCtx {
                            ctx,
                            events: events.clone(),
                            readers: readers.clone(),
                            tally: tally.clone(),
                            staging: (cfg.staging_buffers, cfg.staging_bytes),
                        },
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            devices.push(DeviceRt {
                ordinal,
                dev,
                ctx,
                uuid: cuda::uuid(dev)?,
                exportable,
                granularity: g,
                events,
                movers,
                next_mover: AtomicUsize::new(0),
            });
            arenas.push(Arena::new(prop, cfg.slab_bytes));
        }
        let n = devices.len();
        Ok(Plane {
            inner: Arc::new(Inner {
                cfg,
                devices,
                readers,
                tally,
                core: Mutex::new(Core {
                    poisoned: None,
                    sets: Vec::new(),
                    arenas,
                    host_budget: 0,
                    host_used: 0,
                    cursors: HashMap::new(),
                    next_cursor: 1,
                    dstats: vec![DevCounters::default(); n],
                    hstats: HostCounters::default(),
                    tags: BTreeMap::new(),
                    regions: BTreeMap::new(),
                    events: VecDeque::new(),
                    events_dropped: 0,
                    timings: VecDeque::new(),
                }),
                cv: Condvar::new(),
            }),
        })
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, Core> {
        self.inner.core.lock().unwrap()
    }

    /// Err(Poisoned) once the plane has latched a poison.
    pub fn check(&self) -> Result<()> {
        self.lock().check()
    }

    pub(crate) fn dev(&self, ordinal: i32) -> Result<usize> {
        self.inner
            .devices
            .iter()
            .position(|d| d.ordinal == ordinal)
            .ok_or_else(|| Error::Invalid(format!("device {ordinal} is not served by this plane")))
    }

    /// Wrap a TensorFS read lease as a byte source. The source owns the lease (its GC hold and
    /// verified door) and ends it when the last weight set and handle using it are gone, so
    /// one lease serves every component of a manifest.
    pub fn source(&self, store: Arc<Store>, meta: Arc<Meta>, lease: ReadLease) -> Arc<Source> {
        Arc::new(Source::new(store, meta, lease, self.inner.cfg.direct_io, Arc::as_ptr(&self.inner) as usize))
    }

    /// Register one weight set: `plan`'s parts grouped into `regions`. `host_fd` adopts a
    /// memfd another process holds for the same layout; otherwise the plane creates one.
    pub fn register(
        &self,
        name: &str,
        source: Arc<Source>,
        plan: &ReadPlan,
        regions: &[Vec<String>],
        host_fd: Option<RawFd>,
    ) -> Result<WsId> {
        self.check()?;
        if !source.made_by(Arc::as_ptr(&self.inner) as usize) {
            return Err(Error::Invalid("the source was made by another plane".into()));
        }
        let layout = Layout::build(plan, regions)?;
        let host = match host_fd {
            Some(fd) => HostMem::adopt(fd, &layout)?,
            None => HostMem::create(name, &layout)?,
        };
        let n = layout.regions.len();
        let ws = Ws {
            name: name.to_string(),
            layout: Arc::new(layout),
            source,
            host: Arc::new(host),
            hregs: (0..n)
                .map(|_| HReg {
                    state: HState::Absent,
                    prio: 0,
                    pin: false,
                    registered: false,
                    holds: 0,
                    pending: Vec::new(),
                    then: Vec::new(),
                })
                .collect(),
            devs: (0..self.inner.devices.len()).map(|_| None).collect(),
            evictions: 0,
            evicted_bytes: 0,
        };
        let mut c = self.lock();
        c.sets.push(Some(ws));
        let id = WsId(c.sets.len() as u32 - 1);
        if host_fd.is_some() {
            // Bytes another process left Ready in the memfd join this plane's books (claimed,
            // counted, cheapest to evict): RAM the memfd holds is always inside a live budget.
            for r in 0..n as u32 {
                if c.ws(id)?.host.ready(r) {
                    c.ws_mut(id)?.hregs[r as usize].prio = i64::MIN;
                    self.inner.fill_host(&mut c, id, r, true)?;
                }
            }
        }
        Ok(id)
    }

    pub fn layout(&self, ws: WsId) -> Result<Arc<Layout>> {
        Ok(self.lock().ws(ws)?.layout.clone())
    }
    pub fn name(&self, ws: WsId) -> Result<String> {
        Ok(self.lock().ws(ws)?.name.clone())
    }
    pub fn host_fd(&self, ws: WsId) -> Result<RawFd> {
        Ok(self.lock().ws(ws)?.host.fd())
    }
    pub fn devices(&self) -> Vec<(i32, String)> {
        self.inner.devices.iter().map(|d| (d.ordinal, d.uuid.clone())).collect()
    }

    fn regions_or_all(c: &Core, ws: WsId, regions: Option<&[u32]>) -> Result<Vec<u32>> {
        let n = c.ws(ws)?.layout.regions.len() as u32;
        match regions {
            None => Ok((0..n).collect()),
            Some(r) => {
                if let Some(bad) = r.iter().find(|&&x| x >= n) {
                    return Err(Error::Invalid(format!("region {bad} of {n}")));
                }
                Ok(r.to_vec())
            }
        }
    }

    // ------------------------------------------------------------ budgets

    pub fn set_pinned_budget(&self, bytes: u64) -> Result<Trim> {
        let mut c = self.lock();
        c.check()?;
        c.host_budget = bytes;
        let mut t = Trim::default();
        if c.host_used > bytes {
            let need = c.host_used - bytes;
            self.inner.evict_host(&mut c, need, None, &mut t)?;
        }
        t.over_budget_unreleasable = c.host_used.saturating_sub(bytes);
        c.hstats.trims += 1;
        c.event(Event::new("budget", None, None, None, bytes, 0, format!("pinned; freed {}", t.freed)));
        Ok(t)
    }

    pub fn set_vram_budget(&self, ordinal: i32, bytes: u64) -> Result<Trim> {
        let d = self.dev(ordinal)?;
        let _g = self.inner.devices[d].bind()?;
        let mut c = self.lock();
        c.check()?;
        c.arenas[d].budget = bytes;
        let mut t = Trim::default();
        t.freed += c.arenas[d].trim()?;
        let over = c.arenas[d].committed.saturating_sub(bytes);
        if over > 0 {
            self.inner.evict_dev(&mut c, d, over, None, &mut t)?;
            t.freed += c.arenas[d].trim()?;
        }
        t.over_budget_unreleasable = c.arenas[d].committed.saturating_sub(bytes);
        c.dstats[d].trims += 1;
        c.event(Event::new("budget", None, None, Some(ordinal), bytes, 0, format!("vram; freed {}", t.freed)));
        Ok(t)
    }

    // ------------------------------------------------------------ residency intent

    /// Make `regions` (all when None) resident in `tier` at `prio`, evicting only strictly
    /// lower-priority unpinned, unleased regions; `Shortfall` otherwise. Returns at once; the
    /// ticket reports completion.
    pub fn want(&self, ws: WsId, tier: Tier, regions: Option<&[u32]>, prio: i64, pin: bool) -> Result<Ticket> {
        let mut c = self.lock();
        c.check()?;
        let regions = Self::regions_or_all(&c, ws, regions)?;
        match tier {
            Tier::Pinned => self.inner.want_host(&mut c, ws, &regions, prio, pin)?,
            Tier::Device(o) => {
                let d = self.dev(o)?;
                let _g = self.inner.devices[d].bind()?;
                self.inner.want_dev(&mut c, ws, d, &regions, prio, pin)?;
            }
        }
        Ok(Ticket {
            inner: self.inner.clone(),
            ws,
            tier,
            regions,
        })
    }

    /// One host read, then a device fill per device from it (the pinned tier is filled first
    /// when its budget allows; otherwise each device reads through its own staging).
    pub fn replicate(&self, ws: WsId, ordinals: &[i32], regions: Option<&[u32]>, prio: i64, pin: bool) -> Result<Vec<Ticket>> {
        let mut out = Vec::new();
        {
            let mut c = self.lock();
            c.check()?;
            let rs = Self::regions_or_all(&c, ws, regions)?;
            if self.inner.want_host(&mut c, ws, &rs, prio, pin).is_ok() {
                out.push(Ticket { inner: self.inner.clone(), ws, tier: Tier::Pinned, regions: rs });
            }
        }
        for &o in ordinals {
            out.push(self.want(ws, Tier::Device(o), regions, prio, pin)?);
        }
        Ok(out)
    }

    /// Unmap (device) or unpin (pinned) `regions`; refuses while any is leased or held.
    pub fn drop_regions(&self, ws: WsId, tier: Tier, regions: Option<&[u32]>) -> Result<u64> {
        let mut c = self.lock();
        c.check()?;
        let regions = Self::regions_or_all(&c, ws, regions)?;
        let mut freed = 0;
        match tier {
            Tier::Pinned => {
                for &r in &regions {
                    let h = &mut c.ws_mut(ws)?.hregs[r as usize];
                    h.pin = false;
                    if h.holds > 0 || matches!(h.state, HState::Filling) {
                        return Err(Error::LeaseViolation {
                            op: "drop",
                            live: h.holds as u64,
                            pending: 0,
                            what: format!("pinned region {r}"),
                        });
                    }
                    match h.state {
                        HState::Ready => freed += self.inner.release_host(&mut c, ws, r, true)?,
                        HState::Invalid(_) => h.state = HState::Absent,
                        _ => {}
                    }
                }
            }
            Tier::Device(o) => {
                let d = self.dev(o)?;
                let _g = self.inner.devices[d].bind()?;
                for &r in &regions {
                    let Some(dw) = c.ws_mut(ws)?.devs[d].as_mut() else { continue };
                    let reg = &mut dw.regs[r as usize];
                    reg.pin = false;
                    if reg.chunks.is_empty() {
                        continue;
                    }
                    if reg.live > 0 || reg.holds > 0 || matches!(reg.state, DState::Queued) {
                        return Err(Error::LeaseViolation {
                            op: "drop",
                            live: (reg.live + reg.holds) as u64,
                            pending: reg.pending.len() as u64,
                            what: format!("region {r} on device {o}"),
                        });
                    }
                    freed += self.inner.unmap_region(&mut c, ws, d, r, true)?;
                }
                c.arenas[d].trim()?;
            }
        }
        Ok(freed)
    }

    // ------------------------------------------------------------ resident access

    /// A stream-scoped lease on a resident region. A region not resident is filled on demand
    /// (a miss) at its last wanted priority. No host block unless the fill is not yet queued.
    /// `prio`, when given, becomes the region's priority first (the demand fill evicts only
    /// below it).
    pub fn acquire(&self, ws: WsId, ordinal: i32, r: u32, stream: u64, prio: Option<i64>) -> Result<Lease> {
        let d = self.dev(ordinal)?;
        self.inner.acquire_home(ws, d, r, stream, "", prio)
    }

    /// Set priority (and pin, when given) of regions in a tier without moving a byte. Regions
    /// not resident there keep the value for their next fill.
    pub fn prioritise(&self, ws: WsId, tier: Tier, regions: Option<&[u32]>, prio: i64, pin: Option<bool>) -> Result<()> {
        let mut c = self.lock();
        c.check()?;
        let regions = Self::regions_or_all(&c, ws, regions)?;
        let d = match tier {
            Tier::Pinned => None,
            Tier::Device(o) => Some(self.dev(o)?),
        };
        let w = c.ws_mut(ws)?;
        for r in regions {
            let (p, q) = match d {
                None => {
                    let h = &mut w.hregs[r as usize];
                    (&mut h.prio, &mut h.pin)
                }
                Some(d) => match w.devs[d].as_mut() {
                    Some(dw) => {
                        let g = &mut dw.regs[r as usize];
                        (&mut g.prio, &mut g.pin)
                    }
                    None => continue,
                },
            };
            *p = prio;
            if let Some(pin) = pin {
                *q = pin;
            }
        }
        Ok(())
    }

    /// The weight set's stable device VA (reserved on first use). Bytes are valid only for
    /// resident regions under a lease. The view blocks `close_ws`, not eviction.
    pub fn view(&self, ws: WsId, ordinal: i32) -> Result<View> {
        let d = self.dev(ordinal)?;
        self.inner.view(ws, d)
    }

    /// Driver truth: every chunk the books call mapped must be the driver's mapping at that VA.
    /// A mismatch poisons the weight set on that device. Returns chunks checked.
    pub fn verify(&self, ws: WsId, ordinal: i32) -> Result<u64> {
        let d = self.dev(ordinal)?;
        let _g = self.inner.devices[d].bind()?;
        let mut c = self.lock();
        c.check()?;
        let layout = c.ws(ws)?.layout.clone();
        let Some(dw) = c.ws(ws)?.devs[d].as_ref() else { return Ok(0) };
        let mut n = 0;
        let mut bad = None;
        'regions: for (r, reg) in dw.regs.iter().enumerate() {
            let mut at = dw.base + layout.regions[r].offset;
            for ch in &reg.chunks {
                if device::driver_handle(at)? != Some(ch.handle) {
                    bad = Some(r);
                    break 'regions;
                }
                at += ch.size;
                n += 1;
            }
        }
        match bad {
            Some(r) => Err(c.poison(format!("the driver's mapping of region {r} disagrees with the books"))),
            None => Ok(n),
        }
    }

    // ------------------------------------------------------------ lifecycle

    /// Close a weight set: refuses while leases, holds, views or cursors remain. Its device
    /// memory returns to the arena and its pinned RAM is punched unless another process claims
    /// it. A poisoned plane refuses: copies may still touch its ranges, so they stay mapped.
    pub fn close_ws(&self, ws: WsId) -> Result<()> {
        let mut c = self.lock();
        c.check()?;
        if c.cursors.values().any(|k| k.ws == ws) {
            return Err(Error::LeaseViolation { op: "close", live: 1, pending: 0, what: "an open cursor".into() });
        }
        let ndev = self.inner.devices.len();
        {
            let w = c.ws(ws)?;
            for dw in w.devs.iter().flatten() {
                let live: u32 = dw.regs.iter().map(|r| r.live + r.holds).sum();
                if live > 0 || dw.views > 0 {
                    return Err(Error::LeaseViolation {
                        op: "close",
                        live: live as u64 + dw.views,
                        pending: 0,
                        what: format!("weight set {}", w.name),
                    });
                }
            }
            if w.hregs.iter().any(|h| h.holds > 0 || matches!(h.state, HState::Filling)) {
                return Err(Error::LeaseViolation { op: "close", live: 1, pending: 0, what: "pinned tier in use".into() });
            }
            let queued = w.devs.iter().flatten().any(|dw| dw.regs.iter().any(|r| matches!(r.state, DState::Queued)));
            if queued {
                return Err(Error::LeaseViolation { op: "close", live: 0, pending: 1, what: "a queued fill".into() });
            }
        }
        for d in 0..ndev {
            let _g = self.inner.devices[d].bind()?;
            let n = c.ws(ws)?.layout.regions.len() as u32;
            if c.ws(ws)?.devs[d].is_none() {
                continue;
            }
            for r in 0..n {
                if !c.ws(ws)?.devs[d].as_ref().unwrap().regs[r as usize].chunks.is_empty() {
                    self.inner.unmap_region(&mut c, ws, d, r, false)?;
                }
            }
            let dw = c.ws_mut(ws)?.devs[d].take().unwrap();
            device::free_va(dw.base, c.ws(ws)?.layout.nbytes)?;
            c.arenas[d].trim()?;
        }
        let n = c.ws(ws)?.hregs.len() as u32;
        for r in 0..n {
            if matches!(c.ws(ws)?.hregs[r as usize].state, HState::Ready) {
                self.inner.release_host(&mut c, ws, r, false)?;
            }
        }
        let w = c.sets[ws.0 as usize].take();
        drop(c);
        // Outside the lock: the last weight set on a source ends its TensorFS lease (file I/O).
        drop(w);
        Ok(())
    }

    pub fn close(&self) -> Result<()> {
        let ids: Vec<WsId> = {
            let c = self.lock();
            (0..c.sets.len() as u32).filter(|&i| c.sets[i as usize].is_some()).map(WsId).collect()
        };
        for id in ids {
            self.close_ws(id)?;
        }
        for (d, dev) in self.inner.devices.iter().enumerate() {
            for m in &dev.movers {
                m.stop();
            }
            let _g = dev.bind()?;
            self.lock().arenas[d].release_idle()?;
        }
        self.inner.readers.shutdown();
        Ok(())
    }

    // ------------------------------------------------------------ telemetry

    pub fn events(&self) -> Vec<Event> {
        let mut c = self.lock();
        resolve_timings(&mut c, true);
        c.events.drain(..).collect()
    }

    pub fn stats(&self) -> Stats {
        let mut c = self.lock();
        resolve_timings(&mut c, true);
        let t = &self.inner.tally;
        let mut host = HostStats {
            budget: c.host_budget,
            used: c.host_used,
            counters: c.hstats.clone(),
            cached_bytes: t.cached_bytes.load(Ordering::Relaxed),
            direct_bytes: t.direct_bytes.load(Ordering::Relaxed),
            buffered_bytes: t.buffered_bytes.load(Ordering::Relaxed),
            inline_bytes: t.inline_bytes.load(Ordering::Relaxed),
            direct_refused: t.direct_refused.load(Ordering::Relaxed),
            disk_read_bytes: disk_read_bytes(),
            ..Default::default()
        };
        let mut devs: Vec<DevStats> = self
            .inner
            .devices
            .iter()
            .enumerate()
            .map(|(d, dev)| {
                let a = &c.arenas[d];
                DevStats {
                    ordinal: dev.ordinal,
                    uuid: dev.uuid.clone(),
                    granularity: dev.granularity,
                    exportable: dev.exportable,
                    budget: a.budget,
                    committed: a.committed,
                    mapped: a.mapped,
                    idle: a.idle_bytes(),
                    over_budget_unreleasable: a.committed.saturating_sub(a.budget),
                    counters: c.dstats[d].clone(),
                    ..Default::default()
                }
            })
            .collect();
        let mut sets = Vec::new();
        for (wi, w) in c.sets.iter_mut().enumerate() {
            let Some(w) = w.as_mut() else { continue };
            let mut s = WsStats {
                id: wi as u32,
                name: w.name.clone(),
                nbytes: w.layout.nbytes,
                regions: w.layout.regions.len() as u32,
                host_memfd_bytes: w.host.resident_bytes(),
                evictions: w.evictions,
                evicted_bytes: w.evicted_bytes,
                ..Default::default()
            };
            for (r, h) in w.hregs.iter().enumerate() {
                if matches!(h.state, HState::Ready) {
                    s.host_ready_bytes += w.layout.regions[r].nbytes;
                    host.ready_bytes += w.layout.regions[r].nbytes;
                    if h.registered {
                        host.registered_bytes += w.layout.regions[r].nbytes;
                    }
                }
            }
            host.memfd_bytes += s.host_memfd_bytes;
            for (d, dw) in w.devs.iter_mut().enumerate() {
                let Some(dw) = dw.as_mut() else { continue };
                let mut ready = 0;
                for (r, reg) in dw.regs.iter_mut().enumerate() {
                    let _ = commit(reg);
                    if matches!(reg.state, DState::Ready) {
                        ready += w.layout.regions[r].nbytes;
                    }
                    if reg.live > 0 {
                        devs[d].leased_bytes += w.layout.regions[r].span;
                    }
                    if reg.holds > 0 {
                        devs[d].held_bytes += w.layout.regions[r].span;
                    }
                }
                devs[d].reserved_va += w.layout.nbytes;
                s.devices.push((self.inner.devices[d].ordinal, ready));
            }
            sets.push(s);
        }
        for k in c.cursors.values() {
            devs[k.dev].reserved_va += k.ring_len;
        }
        let tags = c
            .tags
            .iter()
            .map(|((w, d, tag), t)| {
                let name = c.sets.get(*w as usize).and_then(Option::as_ref).map_or(String::new(), |x| x.name.clone());
                (name, self.inner.devices[*d].ordinal, tag.clone(), t.clone())
            })
            .collect();
        let regions = c
            .regions
            .iter()
            .map(|((w, d, tag, r), rc)| {
                let name = c.sets.get(*w as usize).and_then(Option::as_ref).map_or(String::new(), |x| x.name.clone());
                (name, self.inner.devices[*d].ordinal, tag.clone(), *r, rc.clone())
            })
            .collect();
        Stats {
            host,
            devices: devs,
            sets,
            tags,
            regions,
            events_dropped: c.events_dropped,
            poisoned: c.poisoned.clone(),
        }
    }
}

impl Drop for Plane {
    fn drop(&mut self) {
        // Movers and readers hold `Inner`; stop them when the last user handle goes.
        if Arc::strong_count(&self.inner) == 1 {
            let _ = self.close();
        }
    }
}

impl Drop for Inner {
    /// The last reference can go without `close()`: handles collected in any order, a plane
    /// dropped with a fill queued (it ends inside that copy's completion), a poisoned plane.
    /// Its threads stop, and once every device has run all queued work its ranges are unmapped,
    /// its handles released and its pinned tier let go, exactly as `close()` would. A device
    /// that cannot be synchronized keeps its ranges and the pinned tier for the life of the
    /// process: a copy may still touch them.
    fn drop(&mut self) {
        self.readers.stop();
        for dev in &self.devices {
            for m in &dev.movers {
                m.detach();
            }
        }
        let c = self.core.get_mut().unwrap_or_else(|p| p.into_inner());
        if teardown(&self.devices, c).is_err() {
            for w in c.sets.iter().flatten() {
                std::mem::forget(w.host.clone());
            }
        }
    }
}

/// Free everything the books still hold, after the devices ran all queued work.
fn teardown(devices: &[DeviceRt], c: &mut Core) -> Result<()> {
    for dev in devices {
        let _g = dev.bind()?;
        // SAFETY: the device's context is bound.
        cuda::check("cuCtxSynchronize", unsafe { (cuda::driver()?.ctx_synchronize)() })?;
    }
    let mut rings: Vec<(usize, u64, u64, Vec<Chunk>)> = c.cursors.drain().map(|(_, k)| k.into_ring()).collect();
    for (d, dev) in devices.iter().enumerate() {
        let _g = dev.bind()?;
        let mut chunks = Vec::new();
        for (_, base, len, held) in rings.iter_mut().filter(|r| r.0 == d && r.2 > 0) {
            device::unmap_span(*base, *len)?;
            device::free_va(*base, *len)?;
            chunks.append(held);
        }
        for w in c.sets.iter_mut().flatten() {
            let Some(dw) = w.devs[d].take() else { continue };
            for (reg, region) in dw.regs.into_iter().zip(&w.layout.regions) {
                if !reg.chunks.is_empty() {
                    device::unmap_span(dw.base + region.offset, region.span)?;
                    chunks.extend(reg.chunks);
                }
            }
            device::free_va(dw.base, w.layout.nbytes)?;
        }
        c.arenas[d].put_idle(chunks);
        c.arenas[d].release_idle()?;
    }
    let first = devices.first().map(|d| d.ctx);
    for w in c.sets.iter_mut().flatten() {
        for (r, (h, region)) in w.hregs.iter_mut().zip(&w.layout.regions).enumerate() {
            if h.registered {
                let _g = first.map(CtxGuard::enter).transpose()?;
                w.host.unregister(region.offset)?;
                h.registered = false;
            }
            if matches!(h.state, HState::Ready) {
                w.host.release(r as u32, region.offset, region.span)?;
                h.state = HState::Absent;
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct HostStats {
    pub budget: u64,
    pub used: u64,
    pub ready_bytes: u64,
    pub registered_bytes: u64,
    /// Kernel truth: RAM held by this plane's memfds (st_blocks).
    pub memfd_bytes: u64,
    pub cached_bytes: u64,
    pub direct_bytes: u64,
    pub buffered_bytes: u64,
    pub inline_bytes: u64,
    pub direct_refused: u64,
    /// Always 0 since 0.3.93 (see `mapped_copy_bytes`).
    pub map_refused: u64,
    /// Bytes this process caused the block layer to read (`/proc/self/io` read_bytes).
    pub disk_read_bytes: u64,
    pub counters: HostCounters,
}

fn disk_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("read_bytes: ")?.trim().parse().ok()))
        .unwrap_or(0)
}

#[derive(Debug, Clone, Default)]
pub struct DevStats {
    pub ordinal: i32,
    pub uuid: String,
    pub granularity: u64,
    pub exportable: bool,
    pub budget: u64,
    pub committed: u64,
    pub mapped: u64,
    pub idle: u64,
    pub leased_bytes: u64,
    pub held_bytes: u64,
    pub over_budget_unreleasable: u64,
    pub reserved_va: u64,
    pub counters: DevCounters,
}

#[derive(Debug, Clone, Default)]
pub struct WsStats {
    pub id: u32,
    pub name: String,
    pub nbytes: u64,
    pub regions: u32,
    pub host_ready_bytes: u64,
    pub host_memfd_bytes: u64,
    /// (ordinal, resident Ready bytes)
    pub devices: Vec<(i32, u64)>,
    pub evictions: u64,
    pub evicted_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub host: HostStats,
    pub devices: Vec<DevStats>,
    pub sets: Vec<WsStats>,
    /// (weight set, ordinal, stage tag, counters)
    pub tags: Vec<(String, i32, String, TagCounters)>,
    /// (weight set, ordinal, stage tag, region, counters)
    pub regions: Vec<(String, i32, String, u32, RegionCounters)>,
    pub events_dropped: u64,
    pub poisoned: Option<String>,
}

// ---------------------------------------------------------------- tickets, leases, views

/// Completion of a `want`. Reads the books; holds nothing.
pub struct Ticket {
    inner: Arc<Inner>,
    ws: WsId,
    tier: Tier,
    regions: Vec<u32>,
}

impl Ticket {
    /// Ok(true) when every region is resident in the tier.
    pub fn done(&self) -> Result<bool> {
        self.check(false)
    }
    /// Host wait until done: the condvar for host-side steps, then each copy's own event.
    pub fn wait(&self) -> Result<()> {
        self.check(true).map(|_| ())
    }

    fn check(&self, block: bool) -> Result<bool> {
        let mut c = self.inner.core.lock().unwrap();
        loop {
            let w = c.ws_mut(self.ws)?;
            let mut busy = false;
            let mut evs = Vec::new();
            for &r in &self.regions {
                match self.tier {
                    Tier::Pinned => match &w.hregs[r as usize].state {
                        HState::Ready => {}
                        HState::Filling => busy = true,
                        HState::Invalid(e) => return Err(e.clone()),
                        HState::Absent => return Err(Error::Invalid(format!("region {r} left the pinned tier"))),
                    },
                    Tier::Device(o) => {
                        let d = self.inner.devices.iter().position(|x| x.ordinal == o).unwrap();
                        let Some(dw) = w.devs[d].as_mut() else { return Err(Error::Closed) };
                        let reg = &mut dw.regs[r as usize];
                        commit(reg)?;
                        match &reg.state {
                            DState::Ready => {}
                            DState::Queued => busy = true,
                            DState::Filling(ev) => evs.push(ev.clone()),
                            DState::Invalid(e) => return Err(e.clone()),
                            DState::Absent => return Err(Error::Invalid(format!("region {r} was evicted"))),
                        }
                    }
                }
            }
            if !block {
                return Ok(!busy && evs.is_empty());
            }
            if busy {
                c = self.inner.cv.wait(c).unwrap();
                continue;
            }
            drop(c);
            for e in &evs {
                e.sync()?;
            }
            return Ok(true);
        }
    }
}

pub(crate) enum LeaseKind {
    Home,
    Ring { cursor: u64, k: u64 },
}

/// Use of one region's bytes on one consumer stream. `release(stream)` exactly once; dropping
/// an unreleased lease poisons the plane (its consumer may still run on those bytes).
pub struct Lease {
    pub(crate) inner: Arc<Inner>,
    pub ptr: u64,
    pub nbytes: u64,
    pub region: u32,
    /// Where a streamed lease's bytes sit in its cursor's ring; None for a resident one.
    pub ring_offset: Option<u64>,
    pub(crate) ws: WsId,
    pub(crate) dev: usize,
    pub(crate) tag: String,
    pub(crate) start: Mutex<Option<EvRef>>,
    /// The copy this lease's bytes came from, when it was still in flight at acquire.
    pub(crate) fill: Option<EvRef>,
    pub(crate) kind: LeaseKind,
    pub(crate) released: AtomicBool,
}

impl Lease {
    pub fn device(&self) -> i32 {
        self.inner.devices[self.dev].ordinal
    }

    /// Record the release event on `stream`. The bytes (resident or in the ring) are reusable once the
    /// device passes it; the span from acquire to here is the region's GPU-timed use.
    pub fn release(&self, stream: u64) -> Result<()> {
        if self.released.swap(true, Ordering::SeqCst) {
            return Err(Error::Invalid("lease released twice".into()));
        }
        let rt = &self.inner.devices[self.dev];
        let _g = rt.bind()?;
        let ev = rt.events.get(true)?;
        ev.record(stream as CUstream)?;
        let start = self.start.lock().unwrap().take();
        let mut c = self.inner.core.lock().unwrap();
        resolve_timings(&mut c, false);
        if let Some(a) = start {
            c.timings.push_back(Timing {
                dev: self.dev,
                ws: self.ws.0,
                kind: TimingKind::Compute { tag: self.tag.clone(), region: self.region },
                a,
                b: ev.clone(),
                bytes: self.nbytes,
            });
        }
        match self.kind {
            LeaseKind::Home => {
                let reg = &mut c.ws_mut(self.ws)?.devs[self.dev].as_mut().ok_or(Error::Closed)?.regs[self.region as usize];
                reg.live -= 1;
                prune(&mut reg.pending);
                reg.pending.push(ev);
                Ok(())
            }
            LeaseKind::Ring { cursor, k } => crate::cursor::release_ring(&self.inner, &mut c, cursor, k, ev),
        }
    }

    /// Diagnostic: compare this lease's device bytes with the source, part by part, and return
    /// the parts that differ. Waits for the copy and reads the store: debugging runs only.
    pub fn verify_bytes(&self) -> Result<Vec<String>> {
        if self.released.load(Ordering::SeqCst) {
            return Err(Error::Invalid("verify_bytes after release: the bytes may be unmapped".into()));
        }
        let rt = &self.inner.devices[self.dev];
        let _g = rt.bind()?;
        if let Some(ev) = &self.fill {
            ev.sync()?;
        }
        let (layout, source) = {
            let c = self.inner.core.lock().unwrap();
            let w = c.ws(self.ws)?;
            (w.layout.clone(), w.source.clone())
        };
        let region = &layout.regions[self.region as usize];
        let d = cuda::driver()?;
        let mut got = vec![0u8; region.nbytes as usize];
        let mut s: CUstream = std::ptr::null_mut();
        // SAFETY: a private stream; the copy reads the leased range into our buffer.
        unsafe {
            cuda::check("cuStreamCreate", (d.stream_create)(&mut s, cuda::STREAM_NON_BLOCKING))?;
            let r = cuda::check(
                "cuMemcpyDtoHAsync",
                (d.memcpy_dtoh_async)(got.as_mut_ptr() as *mut libc::c_void, self.ptr, got.len(), s),
            )
            .and_then(|_| cuda::check("cuStreamSynchronize", (d.stream_synchronize)(s)));
            (d.stream_destroy)(s);
            r?;
        }
        let mut want = vec![0u8; region.nbytes as usize];
        let tally = IoTally::default();
        for it in &region.items {
            // SAFETY: the item lies inside `want` (layout offsets relative to the region).
            unsafe { source.read_item(it, want.as_mut_ptr().add((it.offset - region.offset) as usize), Mode::Buffered, &tally)? };
        }
        Ok(layout
            .parts
            .iter()
            .filter(|p| p.region == self.region)
            .filter(|p| {
                let a = (p.offset - region.offset) as usize;
                got[a..a + p.nbytes as usize] != want[a..a + p.nbytes as usize]
            })
            .map(|p| p.what.clone())
            .collect())
    }

    /// A zero-copy view of this lease's bytes. It keeps the weight set's VA (or the cursor's
    /// ring) alive, not the bytes: they are valid only while a lease covers them.
    pub fn view(&self) -> Result<LeaseView> {
        let hold = match self.kind {
            LeaseKind::Home => Hold::Ws(self.inner.view(self.ws, self.dev)?),
            LeaseKind::Ring { cursor, .. } => Hold::Ring(crate::cursor::ring_view(&self.inner, cursor)?),
        };
        Ok(LeaseView {
            ptr: self.ptr,
            nbytes: self.nbytes,
            device: self.device(),
            hold,
        })
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if !self.released.load(Ordering::SeqCst) {
            let cause = format!("lease on region {} dropped without release(stream)", self.region);
            self.inner.core.lock().unwrap().poison(cause);
        }
    }
}

pub enum Hold {
    Ws(View),
    Ring(crate::cursor::RingView),
}

pub struct LeaseView {
    pub ptr: u64,
    pub nbytes: u64,
    pub device: i32,
    hold: Hold,
}

impl LeaseView {
    /// A resident region's view is stale once any region of its weight set was unmapped;
    /// a ring view never is (the ring stays mapped for the cursor's life).
    pub fn stale(&self) -> bool {
        match &self.hold {
            Hold::Ws(v) => v.stale(),
            Hold::Ring(_) => false,
        }
    }
}

/// A weight set's device VA, for zero-copy views of resident regions.
pub struct View {
    inner: Arc<Inner>,
    ws: WsId,
    dev: usize,
    pub ptr: u64,
    pub nbytes: u64,
    epoch: u64,
}

impl View {
    pub fn device(&self) -> i32 {
        self.inner.devices[self.dev].ordinal
    }
    /// Whether any region was unmapped since this view was taken.
    pub fn stale(&self) -> bool {
        let c = self.inner.core.lock().unwrap();
        c.ws(self.ws)
            .ok()
            .and_then(|w| w.devs[self.dev].as_ref())
            .is_none_or(|dw| dw.epoch != self.epoch)
    }
}

impl Drop for View {
    fn drop(&mut self) {
        let mut c = self.inner.core.lock().unwrap();
        if let Ok(w) = c.ws_mut(self.ws) {
            if let Some(dw) = w.devs[self.dev].as_mut() {
                dw.views -= 1;
            }
        }
    }
}
