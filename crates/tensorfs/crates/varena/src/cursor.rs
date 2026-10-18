//! The streaming cursor: one schedule of regions on one device.
//!
//! Regions mapped at `stream()` are served in place (they stay droppable: an acquire of one
//! that was dropped since is a miss and refills it). The rest stream through a byte ring: one
//! device range where each streamed position is placed right after the previous one (wrapping
//! to 0 when it does not fit before the end). Its copy waits on the release events of exactly the
//! earlier positions whose bytes it overwrites and records `ready[k]`; copies run ahead as far
//! as the ring has room. `acquire` enqueues `cuStreamWaitEvent(consumer, ready[k])` (no host
//! block once the mover has reported it); `release` records the position's release event and
//! queues what now fits. The floor is the largest streamed region, not the schedule.
//!
//! The schedule is the caller's and is never reordered. A consumer that departs from it is
//! served anyway: a region further ahead in the current cycle skips the positions before it
//! (their bytes are handed on unread); any other region is a demand acquire (a miss when not
//! mapped) and leaves the position unchanged.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, MutexGuard};

use crate::device::{self, Chunk, CopyJob, CopySrc, Copied, EvRef};
use crate::layout::{align_up, PART_ALIGN, REGION_ALIGN};
use crate::plane::{devws, runs, wait_on, Core, DState, Inner, Lease, LeaseKind, Plane, WsId};
use crate::{Error, Result};

/// One streamed position's bytes in the ring, until a later placement overwrites them.
struct Place {
    k: u64,
    off: u64,
    len: u64,
    /// The copy that wrote these bytes, once it was taken by a lease.
    copied: Option<EvRef>,
    /// None while leased or unread; Some(ev): the bytes may be overwritten after `ev`
    /// (None inside: at once).
    freed: Option<Option<EvRef>>,
}

pub(crate) struct Cur {
    pub ws: WsId,
    pub dev: usize,
    order: Vec<u32>,
    /// Per order index: served in place (mapped at creation).
    home: Vec<bool>,
    /// Per order index: rank among the streamed entries.
    rank: Vec<u64>,
    streamed: Vec<u32>,
    /// Per streamed entry: bytes it occupies in the ring (4 KiB aligned).
    sizes: Vec<u64>,
    /// Positions in the schedule; u64::MAX cycles until close.
    total: u64,
    pos: u64,
    pub ring_base: u64,
    pub ring_len: u64,
    ring_mem: Vec<Chunk>,
    head: u64,
    placed: Vec<Place>,
    /// k -> Ok(ready event) once the mover reported it, Err(cause) if the copy failed.
    ready: BTreeMap<u64, Result<EvRef>>,
    next_submit: u64,
    /// Queued copies the mover has not reported yet.
    inflight: u64,
    live: u32,
    views: u32,
    mover: usize,
    /// Dropped while leases or views were live: stops prefetching and closes itself when the
    /// last of them and every copy in flight are gone.
    orphaned: bool,
}

impl Cur {
    /// (device slot, ring base, ring length, ring memory), for a plane torn down unclosed.
    pub(crate) fn into_ring(self) -> (usize, u64, u64, Vec<Chunk>) {
        (self.dev, self.ring_base, self.ring_len, self.ring_mem)
    }
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

fn closed(id: u64) -> Error {
    Error::Closed(format!("cursor {id} was closed"))
}

/// Queue copies for streamed positions, in order, while the ring has room.
fn pump(inner: &Arc<Inner>, c: &mut Core, id: u64) -> Result<()> {
    loop {
        if c.poisoned.is_some() {
            return Ok(()); // nothing new goes to a device whose state is ambiguous
        }
        let cur = c.cursors.get_mut(&id).ok_or_else(|| closed(id))?;
        let k = cur.next_submit;
        if cur.orphaned || k >= cur.streamed_total() {
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
        cur.placed.push(Place { k, off, len: size, copied: None, freed: None });
        cur.head = off + size;
        submit(inner, c, id, k, off, waits)?;
    }
}

/// Queue the copy for streamed position `k` into the ring at `off`, after `waits`.
fn submit(inner: &Arc<Inner>, c: &mut Core, id: u64, k: u64, off: u64, waits: Vec<EvRef>) -> Result<()> {
    let cur = c.cursors.get_mut(&id).ok_or_else(|| closed(id))?;
    let (ws, d, mover) = (cur.ws, cur.dev, cur.mover);
    let r = cur.streamed[(k % cur.streamed.len() as u64) as usize];
    let dst = cur.ring_base + off;
    cur.next_submit = k + 1;
    cur.inflight += 1;
    let w = c.ws(ws)?;
    let region = &w.layout.regions[r as usize];
    let src = match &w.sealed {
        Some(host) if !host.hole(r) => CopySrc::Sealed { host: host.clone(), region: r, offset: region.offset },
        _ => CopySrc::Files { files: w.files.clone(), runs: runs(&w.layout, r) },
    };
    let inner2 = inner.clone();
    let job = CopyJob {
        waits,
        dst,
        len: region.nbytes,
        src,
        done: Box::new(move |res: Result<Copied>| {
            let mut c = inner2.core.lock().unwrap();
            match &res {
                Ok(cp) => c.copied(d, cp),
                Err(e) => c.latch(e),
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
        let cur = c.cursors.get_mut(&id).unwrap();
        cur.inflight -= 1;
        cur.ready.insert(k, Err(e));
    }
    Ok(())
}

/// Mark position `k`'s ring bytes free after `freed`, and queue what fits.
fn hand_on(inner: &Arc<Inner>, c: &mut Core, id: u64, k: u64, freed: Option<EvRef>) -> Result<()> {
    let cur = c.cursors.get_mut(&id).ok_or_else(|| closed(id))?;
    if let Some(p) = cur.placed.iter_mut().find(|p| p.k == k) {
        p.freed = Some(freed);
    }
    pump(inner, c, id)
}

/// A streamed lease ended: its ring bytes are reusable as `freed` says (None: not before close).
/// The books are settled before the next copies are queued; Err is only that queueing.
pub(crate) fn release_ring(inner: &Arc<Inner>, c: &mut Core, id: u64, k: u64, freed: Option<Option<EvRef>>) -> Result<()> {
    let Some(cur) = c.cursors.get_mut(&id) else { return Ok(()) };
    cur.live -= 1;
    let pumped = match freed {
        Some(ev) => hand_on(inner, c, id, k, ev),
        None => Ok(()),
    };
    reap(inner, c, id);
    pumped
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

/// Unmap a cursor's ring and release its memory. No lease, view or copy in flight remains (a
/// failed copy is drained before it is reported). A poisoned plane keeps the ring mapped.
fn finish(inner: &Arc<Inner>, c: &mut Core, id: u64) -> Result<()> {
    c.check()?;
    let d = c.cursors.get(&id).ok_or_else(|| closed(id))?.dev;
    let _g = inner.devices[d].bind()?;
    let cur = c.cursors.remove(&id).unwrap();
    // Every copy into the ring (taken or not) and every reader's release.
    let placed = cur.placed.iter().flat_map(|p| p.copied.iter().chain(p.freed.iter().flatten()));
    if let Err(e) = cur.ready.values().flatten().chain(placed).try_for_each(|e| e.sync()) {
        return Err(c.poison(format!("waiting for a ring's copies and readers failed: {e}")));
    }
    if cur.ring_len > 0 {
        if let Err(e) = device::unmap_span(cur.ring_base, cur.ring_len) {
            return Err(c.poison(format!("ring unmap failed: {e}")));
        }
        for m in cur.ring_mem {
            if let Err(e) = device::release(m.handle) {
                return Err(c.poison(format!("ring release failed: {e}")));
            }
        }
        c.mapped[d] -= cur.ring_len;
        device::free_va(cur.ring_base, cur.ring_len)?;
    }
    Ok(())
}

/// Hand position `k` on unread: if queued, its bytes are free once its copy lands.
fn skip<'a>(inner: &'a Arc<Inner>, mut c: MutexGuard<'a, Core>, id: u64, k: u64) -> Result<MutexGuard<'a, Core>> {
    let cur = c.cursors.get_mut(&id).ok_or_else(|| closed(id))?;
    if k >= cur.next_submit {
        // Never queued: it takes no ring bytes.
        cur.next_submit = k + 1;
        pump(inner, &mut c, id)?;
        return Ok(c);
    }
    while !c.cursors.get(&id).ok_or_else(|| closed(id))?.ready.contains_key(&k) {
        c = inner.cv.wait(c).unwrap();
    }
    let ev = c.cursors.get_mut(&id).unwrap().ready.remove(&k).unwrap().ok();
    hand_on(inner, &mut c, id, k, ev)?;
    Ok(c)
}

/// A streaming schedule. Close it (after releasing every lease) to return its ring.
pub struct Cursor {
    inner: Arc<Inner>,
    id: u64,
    pub ring_ptr: u64,
    pub ring_nbytes: u64,
    /// Per order index: served in place (true) or streamed through the ring.
    pub home: Vec<bool>,
}

impl Plane {
    /// Stream `order` (`repeat` times; 0 cycles until `close`) on device `ordinal` through a
    /// ring of `ring_bytes` (default: twice the largest streamed region; at least the largest).
    /// Mapping the ring may raise `DeviceFull`/`Floor`.
    pub fn stream(&self, ws: WsId, ordinal: i32, order: &[u32], repeat: u32, ring_bytes: Option<u64>) -> Result<Cursor> {
        if order.is_empty() {
            return Err(Error::Invalid("stream needs a non-empty order".into()));
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
                !reg.mem.is_empty() && matches!(reg.state, DState::Ready | DState::Filling(_) | DState::Queued)
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
        let want = ring_bytes.unwrap_or(2 * largest);
        if want < largest {
            return Err(Error::Invalid(format!("a {want}-byte ring cannot hold the largest streamed region ({largest} bytes)")));
        }
        let need = if streamed.is_empty() { 0 } else { align_up(want, REGION_ALIGN) };
        let (ring_base, ring_mem) = if need > 0 {
            inner.room(need)?;
            let base = device::reserve_va(need)?;
            match inner.map_span(d, base, need) {
                Ok(mem) => {
                    c.mapped[d] += need;
                    (base, mem)
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
        let id = c.next_cursor;
        c.next_cursor += 1;
        let mover = inner.devices[d].mover();
        let total = if repeat == 0 { u64::MAX } else { order.len() as u64 * repeat as u64 };
        let cur = Cur {
            ws,
            dev: d,
            order: order.to_vec(),
            home: home.clone(),
            rank,
            streamed,
            sizes,
            total,
            pos: 0,
            ring_base,
            ring_len: need,
            ring_mem,
            head: 0,
            placed: Vec::new(),
            ready: BTreeMap::new(),
            next_submit: 0,
            inflight: 0,
            live: 0,
            views: 0,
            mover,
            orphaned: false,
        };
        c.cursors.insert(id, cur);
        pump(inner, &mut c, id)?;
        Ok(Cursor { inner: inner.clone(), id, ring_ptr: ring_base, ring_nbytes: need, home })
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
        let d = inner.core.lock().unwrap().cursors.get(&self.id).ok_or_else(|| closed(self.id))?.dev;
        let _g = inner.devices[d].bind()?;
        let mut c = inner.core.lock().unwrap();
        c.check()?;
        let cur = c.cursors.get(&self.id).ok_or_else(|| closed(self.id))?;
        let (ws, from) = (cur.ws, cur.pos);
        let Some(p) = cur.ahead(r) else {
            drop(c);
            return inner.acquire_home(ws, d, r, stream);
        };
        for q in from..p {
            if let Some(k) = c.cursors[&self.id].k_of(q) {
                c = skip(inner, c, self.id, k)?;
            }
        }
        let cur = c.cursors.get_mut(&self.id).unwrap();
        cur.pos = p;
        let Some(k) = cur.k_of(p) else {
            // Served in place (refilled first if it was dropped since).
            cur.pos += 1;
            drop(c);
            return inner.acquire_home(ws, d, r, stream);
        };
        if k >= cur.next_submit {
            return Err(Error::Invalid(format!("ring exhausted: streamed position {k} needs leased ring bytes released first")));
        }
        let ev = loop {
            let cur = c.cursors.get(&self.id).ok_or_else(|| closed(self.id))?;
            match cur.ready.get(&k) {
                Some(Ok(e)) => break e.clone(),
                Some(Err(e)) => return Err(e.clone()),
                None => c = inner.cv.wait(c).unwrap(),
            }
        };
        wait_on(stream, &ev)?;
        let nbytes = c.ws(ws)?.layout.regions[r as usize].nbytes;
        let cur = c.cursors.get_mut(&self.id).unwrap();
        cur.ready.remove(&k);
        cur.pos += 1;
        cur.live += 1;
        let place = cur.placed.iter_mut().find(|x| x.k == k).expect("a queued position is placed");
        place.copied = Some(ev.clone());
        let off = place.off;
        Ok(Lease {
            inner: inner.clone(),
            ptr: cur.ring_base + off,
            nbytes,
            region: r,
            ring_offset: Some(off),
            ws,
            dev: d,
            kind: LeaseKind::Ring { cursor: self.id, k },
            released: AtomicBool::new(false),
        })
    }

    /// Return the ring. Refuses while a streamed lease or view is live; waits (targeted) for
    /// copies in flight and for the device to pass every release.
    pub fn close(&self) -> Result<()> {
        let inner = &self.inner;
        let mut c = inner.core.lock().unwrap();
        let Some(cur) = c.cursors.get(&self.id) else { return Ok(()) };
        if cur.live > 0 || cur.views > 0 {
            return Err(Error::Invalid(format!("cursor {} has {} live leases or views", self.id, cur.live + cur.views)));
        }
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
        if self.close().is_err() {
            if let Some(k) = self.inner.core.lock().unwrap().cursors.get_mut(&self.id) {
                k.orphaned = true;
            }
        }
    }
}

/// The ring's device bytes, held by a streamed lease's view. Keeps the cursor from closing.
pub struct RingView {
    inner: Arc<Inner>,
    id: u64,
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
    c.cursors.get_mut(&id).ok_or_else(|| closed(id))?.views += 1;
    Ok(RingView { inner: inner.clone(), id })
}
