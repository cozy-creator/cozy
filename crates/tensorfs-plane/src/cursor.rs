//! The streaming cursor: one stage's region schedule on one device.
//!
//! Regions resident at `stream()` are served in place and held for the cursor's life. The rest
//! stream through a byte ring: one device range, backed by slab-sized handles, where each
//! streamed position is placed right after the previous one (wrapping to 0 when it does not
//! fit before the end). Its copy waits on the release events of exactly the earlier positions
//! whose bytes it overwrites and records `ready[k]`. `acquire` enqueues
//! `cuStreamWaitEvent(consumer, ready[k])` (no host block once the mover has recorded it);
//! `release` records the position's release event and queues what now fits. Handle size, ring
//! size and region size are independent: many fine regions share a ring of a few slabs, and
//! the floor is the largest streamed region, not `window × largest`.
//!
//! The schedule is the caller's and is never reordered. A consumer that departs from it is
//! served anyway: a region further ahead in the current cycle skips the positions before it
//! (their bytes are handed on unread); any other region is a demand acquire (a miss when not
//! resident) and leaves the position unchanged.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::device::{self, Chunk, CopyJob, CopySrc, Copied, EvRef};
use crate::layout::{align_up, PART_ALIGN, REGION_ALIGN};
use crate::plane::{devws, Core, DState, HState, Inner, Lease, LeaseKind, Plane, Timing, TimingKind, Trim, WsId};
use crate::{Error, Result};

/// One streamed position's bytes in the ring, until a later placement overwrites them.
struct Place {
    k: u64,
    off: u64,
    len: u64,
    /// None while leased or unread; Some(ev): the bytes may be overwritten after `ev`
    /// (None inside: at once).
    freed: Option<Option<EvRef>>,
}

pub(crate) struct Cur {
    pub ws: WsId,
    pub dev: usize,
    pub tag: String,
    order: Vec<u32>,
    /// Per order index: served in place (resident at creation).
    home: Vec<bool>,
    /// Per order index: rank among the streamed entries.
    rank: Vec<u64>,
    streamed: Vec<u32>,
    /// Per streamed entry: bytes it occupies in the ring (4 KiB aligned).
    sizes: Vec<u64>,
    /// Positions in the schedule; u64::MAX cycles until close.
    total: u64,
    /// Most streamed positions queued ahead of the consumer.
    window: u64,
    pos: u64,
    pub ring_base: u64,
    pub ring_len: u64,
    ring_chunks: Vec<Chunk>,
    head: u64,
    placed: Vec<Place>,
    /// k -> Ok(ready event) once the mover recorded it, Err(cause) if the copy failed.
    ready: BTreeMap<u64, Result<EvRef>>,
    /// Streamed positions released or skipped.
    done: u64,
    next_submit: u64,
    /// Queued copies the mover has not recorded yet.
    inflight: u64,
    live: u32,
    views: u32,
    mover: usize,
    /// Dropped while leases or views were live: stops prefetching and closes itself when
    /// the last of them and every copy in flight are gone.
    orphaned: bool,
}

impl Cur {
    fn streamed_total(&self) -> u64 {
        if self.total == u64::MAX {
            return u64::MAX;
        }
        self.streamed.len() as u64 * (self.total / self.order.len() as u64)
    }
    fn k_of(&self, p: u64) -> Option<u64> {
        let n = self.order.len() as u64;
        let i = (p % n) as usize;
        (!self.home[i]).then(|| (p / n) * self.streamed.len() as u64 + self.rank[i])
    }
    /// The next position at or after `pos`, within one cycle, that schedules `r`.
    fn ahead(&self, r: u32) -> Option<u64> {
        let n = self.order.len() as u64;
        (self.pos..self.pos.saturating_add(n).min(self.total)).find(|&p| self.order[(p % n) as usize] == r)
    }
}

/// Queue copies for streamed positions, in order, while the window and the ring allow.
fn pump(inner: &Arc<Inner>, c: &mut Core, id: u64) -> Result<()> {
    loop {
        if c.poisoned.is_some() {
            return Ok(()); // nothing new goes to a device whose state is ambiguous
        }
        let cur = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
        let k = cur.next_submit;
        if cur.orphaned || k >= cur.streamed_total() || k - cur.done >= cur.window {
            return Ok(());
        }
        let size = cur.sizes[(k % cur.streamed.len() as u64) as usize];
        let off = if cur.head + size > cur.ring_len { 0 } else { cur.head };
        let hit = |p: &Place| p.off < off + size && off < p.off + p.len;
        if cur.placed.iter().any(|p| hit(p) && p.freed.is_none()) {
            return Ok(()); // bytes still leased or unread: wait for their release
        }
        let waits: Vec<EvRef> = cur.placed.iter().filter(|p| hit(p)).filter_map(|p| p.freed.clone().flatten()).collect();
        cur.placed.retain(|p| !hit(p));
        cur.placed.push(Place { k, off, len: size, freed: None });
        cur.head = off + size;
        submit(inner, c, id, k, off, waits)?;
    }
}

/// Queue the copy for streamed position `k` into the ring at `off`, after `waits`.
fn submit(inner: &Arc<Inner>, c: &mut Core, id: u64, k: u64, off: u64, waits: Vec<EvRef>) -> Result<()> {
    let cur = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
    let (ws, d, mover) = (cur.ws, cur.dev, cur.mover);
    let r = cur.streamed[(k % cur.streamed.len() as u64) as usize];
    let dst = cur.ring_base + off;
    cur.next_submit = k + 1;
    cur.inflight += 1;
    let w = c.ws_mut(ws)?;
    let region = w.layout.regions[r as usize].clone();
    let from_host = matches!(w.hregs[r as usize].state, HState::Ready);
    let src = if from_host {
        w.hregs[r as usize].holds += 1;
        // SAFETY: inside the live mapping; the hold keeps the region until the copy's event.
        CopySrc::Host(unsafe { w.host.base().add(region.offset as usize) })
    } else {
        CopySrc::Disk {
            source: w.source.clone(),
            items: region.items.clone(),
            base: region.offset,
        }
    };
    let inner2 = inner.clone();
    let job = CopyJob {
        waits,
        dst,
        len: region.nbytes,
        src,
        done: Box::new(move |res: Result<Copied>| {
            let mut c = inner2.core.lock().unwrap();
            if from_host {
                if let Ok(w) = c.ws_mut(ws) {
                    let h = &mut w.hregs[r as usize];
                    h.holds -= 1;
                    if let Ok(cp) = &res {
                        crate::plane::prune(&mut h.pending);
                        h.pending.push(cp.ready.clone());
                    }
                }
            }
            let tag = c.cursors.get(&id).map(|k| k.tag.clone()).unwrap_or_default();
            if let Ok(cp) = &res {
                c.dstats[d].copies += 1;
                c.tag(ws.0, d, &tag).streamed_bytes += cp.bytes;
                c.timings.push_back(Timing {
                    dev: d,
                    ws: ws.0,
                    kind: TimingKind::Copy { from_host },
                    a: cp.start.clone(),
                    b: cp.ready.clone(),
                    bytes: cp.bytes,
                });
            }
            if let Err(e) = &res {
                c.latch(e);
            }
            if let Some(cur) = c.cursors.get_mut(&id) {
                cur.inflight -= 1;
                cur.ready.insert(k, res.map(|cp| cp.ready));
            }
            reap(&inner2, &mut c, id);
            drop(c);
            inner2.cv.notify_all();
        }),
    };
    if let Err(e) = inner.devices[d].movers[mover].submit(job) {
        if from_host {
            c.ws_mut(ws)?.hregs[r as usize].holds -= 1;
        }
        let cur = c.cursors.get_mut(&id).unwrap();
        cur.inflight -= 1;
        cur.ready.insert(k, Err(e));
    }
    Ok(())
}

/// Mark position `k`'s ring bytes free after `freed`, count it done, and queue what fits.
fn hand_on(inner: &Arc<Inner>, c: &mut Core, id: u64, k: u64, freed: Option<EvRef>) -> Result<()> {
    let cur = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
    if let Some(p) = cur.placed.iter_mut().find(|p| p.k == k) {
        p.freed = Some(freed);
    }
    cur.done += 1;
    pump(inner, c, id)
}

/// A streamed lease ended: its bytes are reusable after `ev`.
pub(crate) fn release_ring(inner: &Arc<Inner>, c: &mut Core, id: u64, k: u64, ev: EvRef) -> Result<()> {
    c.cursors.get_mut(&id).ok_or(Error::Closed)?.live -= 1;
    hand_on(inner, c, id, k, Some(ev))?;
    reap(inner, c, id);
    Ok(())
}

/// Finish an orphaned cursor once nothing uses its ring.
fn reap(inner: &Arc<Inner>, c: &mut Core, id: u64) {
    let Some(cur) = c.cursors.get(&id) else { return };
    if cur.orphaned && cur.live == 0 && cur.views == 0 && cur.inflight == 0 {
        if let Err(e) = finish(inner, c, id) {
            c.poison(format!("closing a dropped cursor failed: {e}"));
        }
    }
}

/// Unmap a cursor's ring and return its holds. No lease, view or copy in flight remains (a
/// failed copy is drained before it is reported). A poisoned plane keeps the ring mapped.
fn finish(inner: &Arc<Inner>, c: &mut Core, id: u64) -> Result<()> {
    c.check()?;
    let cur = c.cursors.remove(&id).ok_or(Error::Closed)?;
    let d = cur.dev;
    let _g = inner.devices[d].bind()?;
    let freed = cur.placed.iter().filter_map(|p| p.freed.as_ref().and_then(Option::as_ref));
    if let Err(e) = cur.ready.values().flatten().chain(freed).try_for_each(|e| e.sync()) {
        return Err(c.poison(format!("waiting for a ring's copies and readers failed: {e}")));
    }
    if cur.ring_len > 0 {
        if let Err(e) = device::unmap_span(cur.ring_base, cur.ring_len) {
            return Err(c.poison(format!("ring unmap failed: {e}")));
        }
        c.arenas[d].mapped -= cur.ring_len;
        c.arenas[d].put_idle(cur.ring_chunks);
        device::free_va(cur.ring_base, cur.ring_len)?;
    }
    if let Ok(w) = c.ws_mut(cur.ws) {
        if let Some(dw) = w.devs[d].as_mut() {
            for (i, &r) in cur.order.iter().enumerate() {
                if cur.home[i] {
                    dw.regs[r as usize].holds -= 1;
                }
            }
        }
    }
    c.arenas[d].trim()?;
    Ok(())
}

/// Hand position `k` on unread: if queued, its bytes are free once its copy lands.
fn skip<'a>(inner: &'a Arc<Inner>, mut c: MutexGuard<'a, Core>, id: u64, k: u64) -> Result<MutexGuard<'a, Core>> {
    let cur = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
    if k >= cur.next_submit {
        // Never queued: it takes no ring bytes.
        cur.next_submit = k + 1;
        cur.done += 1;
        pump(inner, &mut c, id)?;
        return Ok(c);
    }
    while !c.cursors.get(&id).ok_or(Error::Closed)?.ready.contains_key(&k) {
        c = inner.cv.wait(c).unwrap();
    }
    let ev = c.cursors.get_mut(&id).unwrap().ready.remove(&k).unwrap().ok();
    hand_on(inner, &mut c, id, k, ev)?;
    Ok(c)
}

/// A stage's streaming schedule. Close it (after releasing every lease) to return its ring
/// and holds.
pub struct Cursor {
    inner: Arc<Inner>,
    id: u64,
    pub window: u32,
    pub ring_ptr: u64,
    pub ring_nbytes: u64,
    /// Per order index: served in place (true) or streamed through the ring.
    pub home: Vec<bool>,
}

impl Plane {
    /// Start streaming `order` (`repeat` times; 0 cycles until `close`) on device `ordinal`,
    /// queuing up to `window` streamed positions ahead through a ring of `ring_bytes` (default:
    /// `window` × the largest streamed region). The ring may evict regions below `prio`.
    /// `tag` names the stage.
    #[allow(clippy::too_many_arguments)]
    pub fn stream(
        &self,
        ws: WsId,
        ordinal: i32,
        order: &[u32],
        window: u32,
        repeat: u32,
        prio: i64,
        tag: &str,
        ring_bytes: Option<u64>,
    ) -> Result<Cursor> {
        if order.is_empty() || window == 0 {
            return Err(Error::Invalid("stream needs a non-empty order and window >= 1".into()));
        }
        let d = self.dev(ordinal)?;
        let inner = &self.inner;
        let _g = inner.devices[d].bind()?;
        let mut c = self.lock();
        let layout = c.ws(ws)?.layout.clone();
        if let Some(bad) = order.iter().find(|&&r| r as usize >= layout.regions.len()) {
            return Err(Error::Invalid(format!("region {bad} of {}", layout.regions.len())));
        }
        let dw = devws(&mut c, ws, d)?;
        let home: Vec<bool> = order
            .iter()
            .map(|&r| {
                let reg = &dw.regs[r as usize];
                !reg.chunks.is_empty() && matches!(reg.state, DState::Ready | DState::Filling(_) | DState::Queued)
            })
            .collect();
        let mut rank = vec![0u64; order.len()];
        let mut streamed = Vec::new();
        for (i, &r) in order.iter().enumerate() {
            if !home[i] {
                rank[i] = streamed.len() as u64;
                streamed.push(r);
            }
        }
        let sizes: Vec<u64> = streamed.iter().map(|&r| align_up(layout.regions[r as usize].nbytes.max(1), PART_ALIGN)).collect();
        let largest = sizes.iter().copied().max().unwrap_or(0);
        let want = ring_bytes.unwrap_or(window as u64 * largest);
        if want < largest {
            return Err(Error::Invalid(format!("a {want}-byte ring cannot hold the largest streamed region ({largest} bytes)")));
        }
        let need = if streamed.is_empty() { 0 } else { align_up(want, REGION_ALIGN) };
        let (budget, mapped) = (c.arenas[d].budget, c.arenas[d].mapped);
        if need > budget {
            return Err(Error::BelowFloor { pool: "vram", need, budget });
        }
        if mapped + need > budget {
            let deficit = mapped + need - budget;
            let mut t = Trim::default();
            let freed = inner.evict_dev(&mut c, d, deficit, Some(prio), &mut t)?;
            if freed < deficit {
                return Err(Error::Shortfall {
                    pool: "vram",
                    need,
                    available: c.arenas[d].budget.saturating_sub(c.arenas[d].mapped),
                    blockers: t.blockers,
                });
            }
        }
        let (ring_base, ring_chunks) = if need > 0 {
            let base = device::reserve_va(need)?;
            let sizes = c.arenas[d].chunk_sizes(need);
            let mapped = c.arenas[d].take(&sizes).and_then(|ch| match device::map_span(base, &ch, inner.devices[d].dev) {
                Ok(()) => Ok(ch),
                Err(e) => {
                    c.arenas[d].put_idle(ch);
                    Err(e)
                }
            });
            match mapped {
                Ok(ch) => {
                    c.arenas[d].mapped += need;
                    (base, ch)
                }
                Err(e) => {
                    let _ = device::free_va(base, need);
                    return Err(match e {
                        Error::Poisoned(p) => c.poison(p),
                        e => e,
                    });
                }
            }
        } else {
            (0, Vec::new())
        };
        let dw = devws(&mut c, ws, d)?;
        for (i, &r) in order.iter().enumerate() {
            if home[i] {
                dw.regs[r as usize].holds += 1;
            }
        }
        let id = c.next_cursor;
        c.next_cursor += 1;
        let mover = inner.devices[d].mover();
        c.cursors.insert(
            id,
            Cur {
                ws,
                dev: d,
                tag: tag.to_string(),
                order: order.to_vec(),
                home: home.clone(),
                rank,
                streamed,
                sizes,
                total: if repeat == 0 { u64::MAX } else { order.len() as u64 * repeat as u64 },
                window: window as u64,
                pos: 0,
                ring_base,
                ring_len: need,
                ring_chunks,
                head: 0,
                placed: Vec::new(),
                ready: BTreeMap::new(),
                done: 0,
                next_submit: 0,
                inflight: 0,
                live: 0,
                views: 0,
                mover,
                orphaned: false,
            },
        );
        pump(inner, &mut c, id)?;
        Ok(Cursor {
            inner: inner.clone(),
            id,
            window,
            ring_ptr: ring_base,
            ring_nbytes: need,
            home,
        })
    }
}

impl Cursor {
    /// Schedule positions consumed so far.
    pub fn position(&self) -> u64 {
        self.inner.core.lock().unwrap().cursors.get(&self.id).map_or(0, |k| k.pos)
    }
    /// Completed cycles over the order.
    pub fn passes(&self) -> u64 {
        self.inner.core.lock().unwrap().cursors.get(&self.id).map_or(0, |k| k.pos / k.order.len() as u64)
    }

    /// Lease region `r` for `stream`: the next scheduled position when it names `r`, else as
    /// described in the module doc.
    pub fn acquire(&self, r: u32, stream: u64) -> Result<Lease> {
        let inner = &self.inner;
        let d = inner.core.lock().unwrap().cursors.get(&self.id).ok_or(Error::Closed)?.dev;
        let _g = inner.devices[d].bind()?;
        let mut c = inner.core.lock().unwrap();
        c.check()?;
        let cur = c.cursors.get(&self.id).ok_or(Error::Closed)?;
        let (ws, tag) = (cur.ws, cur.tag.clone());
        let from = cur.pos;
        let Some(p) = cur.ahead(r) else {
            drop(c);
            return inner.acquire_home(ws, d, r, stream, &tag, None);
        };
        for q in from..p {
            if let Some(k) = c.cursors[&self.id].k_of(q) {
                c = skip(inner, c, self.id, k)?;
            }
            c.tag(ws.0, d, &tag).skipped += 1;
        }
        let cur = c.cursors.get_mut(&self.id).unwrap();
        cur.pos = p;
        let Some(k) = cur.k_of(p) else {
            // Served in place: the region is held, so it is still mapped.
            cur.pos += 1;
            drop(c);
            return inner.acquire_home(ws, d, r, stream, &tag, None);
        };
        if k >= cur.next_submit {
            return Err(Error::Invalid(format!(
                "ring exhausted: streamed position {k} needs leased ring bytes released first"
            )));
        }
        let t0 = std::time::Instant::now();
        let ev = loop {
            let cur = c.cursors.get(&self.id).ok_or(Error::Closed)?;
            match cur.ready.get(&k) {
                Some(Ok(e)) => break e.clone(),
                Some(Err(e)) => return Err(e.clone()),
                None => c = inner.cv.wait(c).unwrap(),
            }
        };
        let host_wait = t0.elapsed().as_nanos() as u64;
        let nbytes = c.ws(ws)?.layout.regions[r as usize].nbytes;
        let start = inner.begin_use(&mut c, d, ws, r, &tag, stream, Some(&ev), host_wait, nbytes)?;
        let cur = c.cursors.get_mut(&self.id).unwrap();
        cur.ready.remove(&k);
        cur.pos += 1;
        cur.live += 1;
        let off = cur.placed.iter().find(|x| x.k == k).expect("a queued position is placed").off;
        Ok(Lease {
            inner: inner.clone(),
            ptr: cur.ring_base + off,
            nbytes,
            region: r,
            ring_offset: Some(off),
            ws,
            dev: d,
            tag,
            start: Mutex::new(Some(start)),
            fill: Some(ev),
            kind: LeaseKind::Ring { cursor: self.id, k },
            released: AtomicBool::new(false),
        })
    }

    pub fn ring_view(&self) -> Result<RingView> {
        ring_view(&self.inner, self.id)
    }

    /// Return the ring and holds. Refuses while a streamed lease or ring view is live; waits
    /// (targeted) for copies in flight and for the device to pass every release.
    pub fn close(&self) -> Result<()> {
        let inner = &self.inner;
        let mut c = inner.core.lock().unwrap();
        let Some(cur) = c.cursors.get(&self.id) else { return Ok(()) };
        if cur.live > 0 || cur.views > 0 {
            return Err(Error::LeaseViolation {
                op: "close cursor",
                live: (cur.live + cur.views) as u64,
                pending: 0,
                what: "streamed leases or ring views".into(),
            });
        }
        // Every queued copy must be recorded before the ring can be unmapped.
        while c.cursors[&self.id].inflight > 0 {
            c = inner.cv.wait(c).unwrap();
        }
        finish(inner, &mut c, self.id)?;
        drop(c);
        inner.cv.notify_all();
        Ok(())
    }
}

impl Drop for Cursor {
    fn drop(&mut self) {
        if let Err(Error::LeaseViolation { .. }) = self.close() {
            if let Some(k) = self.inner.core.lock().unwrap().cursors.get_mut(&self.id) {
                k.orphaned = true;
            }
        }
    }
}

/// The ring's device bytes for zero-copy views. Keeps the cursor from closing.
pub struct RingView {
    inner: Arc<Inner>,
    id: u64,
    pub device: i32,
    pub ptr: u64,
    pub nbytes: u64,
}

impl Drop for RingView {
    fn drop(&mut self) {
        let mut c = self.inner.core.lock().unwrap();
        if let Some(k) = c.cursors.get_mut(&self.id) {
            k.views -= 1;
        }
        reap(&self.inner, &mut c, self.id);
    }
}

pub(crate) fn ring_view(inner: &Arc<Inner>, id: u64) -> Result<RingView> {
    let mut c = inner.core.lock().unwrap();
    let k = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
    k.views += 1;
    let d = k.dev;
    Ok(RingView {
        inner: inner.clone(),
        id,
        device: inner.devices[d].ordinal,
        ptr: k.ring_base,
        nbytes: k.ring_len,
    })
}
