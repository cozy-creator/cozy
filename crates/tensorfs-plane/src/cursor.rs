//! The streaming cursor: one stage's region schedule on one device.
//!
//! Regions resident at `stream()` are served in place and held for the cursor's life; the rest
//! stream through a ring of `window` device slots. The k-th streamed position lands in slot
//! `k % window`; its copy waits on the release event of position `k - window` (the slot's last
//! reader) and records `ready[k]`. `acquire` enqueues `cuStreamWaitEvent(consumer, ready[k])`
//! (no host block once the mover has recorded it); `release` records `freed[k]` and queues the
//! copy for `k + window`.
//!
//! The schedule is the caller's and is never reordered. A consumer that departs from it is
//! served anyway: a region further ahead in the current cycle skips the positions before it
//! (their slots are handed on unread); any other region is a demand acquire (a miss when not
//! resident) and leaves the position unchanged.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::device::{self, Chunk, CopyJob, CopySrc, Copied, EvRef};
use crate::plane::{devws, Core, DState, HState, Inner, Lease, LeaseKind, Plane, Timing, TimingKind, Trim, WsId};
use crate::{Error, Result};

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
    /// Positions in the schedule; u64::MAX cycles until close.
    total: u64,
    window: u64,
    pos: u64,
    pub slot_base: u64,
    pub slot_span: u64,
    pub slots: Vec<Vec<Chunk>>,
    /// k -> Ok(ready event) once the mover recorded it, Err(cause) if the copy failed.
    ready: BTreeMap<u64, Result<EvRef>>,
    /// k -> the event after which slot `k % window` is free (None: already free), kept until
    /// the copy that reuses the slot is queued.
    freed: BTreeMap<u64, Option<EvRef>>,
    /// Streamed positions acquired-and-released or skipped.
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

/// Queue copies for every streamed position whose slot is free, in order.
fn pump(inner: &Arc<Inner>, c: &mut Core, id: u64) -> Result<()> {
    loop {
        let cur = c.cursors.get(&id).ok_or(Error::Closed)?;
        let k = cur.next_submit;
        if cur.orphaned || k >= cur.streamed_total() || (k >= cur.window && !cur.freed.contains_key(&(k - cur.window))) {
            return Ok(());
        }
        submit(inner, c, id, k)?;
    }
}

/// Queue the copy for streamed position `k`; its slot is free after `freed[k - window]`.
fn submit(inner: &Arc<Inner>, c: &mut Core, id: u64, k: u64) -> Result<()> {
    let cur = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
    let (ws, d, window, mover) = (cur.ws, cur.dev, cur.window, cur.mover);
    let r = cur.streamed[(k % cur.streamed.len() as u64) as usize];
    let waits: Vec<EvRef> = if k >= window { cur.freed.remove(&(k - window)).flatten().into_iter().collect() } else { Vec::new() };
    let dst = cur.slot_base + (k % window) * cur.slot_span;
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

/// A streamed lease ended: its slot is free after `ev`.
pub(crate) fn release_slot(inner: &Arc<Inner>, c: &mut Core, id: u64, k: u64, ev: EvRef) -> Result<()> {
    let cur = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
    cur.freed.insert(k, Some(ev));
    cur.live -= 1;
    cur.done += 1;
    pump(inner, c, id)?;
    reap(inner, c, id);
    Ok(())
}

/// Finish an orphaned cursor once nothing uses its slots.
fn reap(inner: &Arc<Inner>, c: &mut Core, id: u64) {
    let Some(cur) = c.cursors.get(&id) else { return };
    if cur.orphaned && cur.live == 0 && cur.views == 0 && cur.inflight == 0 {
        if let Err(e) = finish(inner, c, id) {
            c.poison(format!("closing a dropped cursor failed: {e}"));
        }
    }
}

/// Unmap a cursor's slots and return its holds. No lease, view or copy in flight remains.
fn finish(inner: &Arc<Inner>, c: &mut Core, id: u64) -> Result<()> {
    let cur = c.cursors.remove(&id).ok_or(Error::Closed)?;
    let d = cur.dev;
    let _g = inner.devices[d].bind()?;
    for e in cur.ready.values().flatten().chain(cur.freed.values().flatten()) {
        e.sync()?;
    }
    for (i, ch) in cur.slots.into_iter().enumerate() {
        if let Err(e) = device::unmap_span(cur.slot_base + i as u64 * cur.slot_span, cur.slot_span) {
            return Err(c.poison(format!("slot unmap failed: {e}")));
        }
        c.arenas[d].mapped -= cur.slot_span;
        c.arenas[d].put_idle(ch);
    }
    if cur.slot_span * cur.window > 0 {
        device::free_va(cur.slot_base, cur.slot_span * cur.window)?;
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

/// Hand position `k`'s slot on unread: free once its copy lands (or at once if never queued).
fn skip<'a>(inner: &'a Arc<Inner>, mut c: MutexGuard<'a, Core>, id: u64, k: u64) -> Result<MutexGuard<'a, Core>> {
    let cur = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
    if k >= cur.next_submit {
        // Never queued: the slot is free when its previous reader was.
        if k >= cur.window && !cur.freed.contains_key(&(k - cur.window)) {
            return Err(Error::Invalid(format!("cannot skip streamed position {k}: slot {} is still leased", k % cur.window)));
        }
        let prev = if k >= cur.window { cur.freed.remove(&(k - cur.window)).flatten() } else { None };
        cur.freed.insert(k, prev);
        cur.next_submit = k + 1;
    } else {
        while !c.cursors.get(&id).ok_or(Error::Closed)?.ready.contains_key(&k) {
            c = inner.cv.wait(c).unwrap();
        }
        let cur = c.cursors.get_mut(&id).unwrap();
        let ev = cur.ready.remove(&k).unwrap().ok();
        cur.freed.insert(k, ev);
    }
    c.cursors.get_mut(&id).unwrap().done += 1;
    pump(inner, &mut c, id)?;
    Ok(c)
}

/// A stage's streaming schedule. Close it (after releasing every lease) to return its slots
/// and holds.
pub struct Cursor {
    inner: Arc<Inner>,
    id: u64,
    pub window: u32,
    pub slot_nbytes: u64,
    pub slot_ptrs: Vec<u64>,
    /// Per order index: served in place (true) or streamed through slots.
    pub home: Vec<bool>,
}

impl Plane {
    /// Start streaming `order` (`repeat` times; 0 cycles until `close`) on device `ordinal`
    /// through `window` slots. Slots may evict regions below `prio`. `tag` names the stage.
    #[allow(clippy::too_many_arguments)]
    pub fn stream(&self, ws: WsId, ordinal: i32, order: &[u32], window: u32, repeat: u32, prio: i64, tag: &str) -> Result<Cursor> {
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
        let slot_span = streamed.iter().map(|&r| layout.regions[r as usize].span).max().unwrap_or(0);
        let window = if streamed.is_empty() { 0 } else { window as u64 };
        let need = slot_span * window;
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
        let slot_base = if need > 0 { device::reserve_va(need)? } else { 0 };
        let mut slots = Vec::new();
        for i in 0..window {
            let sizes = c.arenas[d].chunk_sizes(slot_span);
            let r = c.arenas[d].take(&sizes).and_then(|ch| match device::map_span(slot_base + i * slot_span, &ch, inner.devices[d].dev) {
                Ok(()) => Ok(ch),
                Err(e) => {
                    c.arenas[d].put_idle(ch);
                    Err(e)
                }
            });
            match r {
                Ok(ch) => {
                    c.arenas[d].mapped += slot_span;
                    slots.push(ch);
                }
                Err(e) => {
                    for (j, ch) in slots.drain(..).enumerate() {
                        if device::unmap_span(slot_base + j as u64 * slot_span, slot_span).is_err() {
                            return Err(c.poison(format!("slot rollback unmap failed after {e}")));
                        }
                        c.arenas[d].mapped -= slot_span;
                        c.arenas[d].put_idle(ch);
                    }
                    let _ = device::free_va(slot_base, need);
                    return Err(match e {
                        Error::Poisoned(p) => c.poison(p),
                        e => e,
                    });
                }
            }
        }
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
                total: if repeat == 0 { u64::MAX } else { order.len() as u64 * repeat as u64 },
                window,
                pos: 0,
                slot_base,
                slot_span,
                slots,
                ready: BTreeMap::new(),
                freed: BTreeMap::new(),
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
            window: window as u32,
            slot_nbytes: slot_span,
            slot_ptrs: (0..window).map(|i| slot_base + i * slot_span).collect(),
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
                "window exhausted: streamed position {k} needs slot {} released first",
                k % cur.window
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
        let slot = k % cur.window;
        Ok(Lease {
            inner: inner.clone(),
            ptr: cur.slot_base + slot * cur.slot_span,
            nbytes,
            region: r,
            slot: Some(slot as u32),
            ws,
            dev: d,
            tag,
            start: Mutex::new(Some(start)),
            kind: LeaseKind::Slot { cursor: self.id, k },
            released: AtomicBool::new(false),
        })
    }

    pub fn slot_view(&self, i: u32) -> Result<SlotView> {
        slot_view(&self.inner, self.id, i)
    }

    /// Return slots and holds. Refuses while a streamed lease or slot view is live; waits
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
                what: "slot leases or slot views".into(),
            });
        }
        // Every queued copy must be recorded before its slot can be unmapped.
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

/// A slot's device bytes for zero-copy views. Keeps the cursor from closing.
pub struct SlotView {
    inner: Arc<Inner>,
    id: u64,
    pub device: i32,
    pub ptr: u64,
    pub nbytes: u64,
}

impl Drop for SlotView {
    fn drop(&mut self) {
        let mut c = self.inner.core.lock().unwrap();
        if let Some(k) = c.cursors.get_mut(&self.id) {
            k.views -= 1;
        }
        reap(&self.inner, &mut c, self.id);
    }
}

pub(crate) fn slot_view(inner: &Arc<Inner>, id: u64, i: u32) -> Result<SlotView> {
    let mut c = inner.core.lock().unwrap();
    let k = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
    if i as u64 >= k.window {
        return Err(Error::Invalid(format!("slot {i} of {}", k.window)));
    }
    k.views += 1;
    let d = k.dev;
    Ok(SlotView {
        inner: inner.clone(),
        id,
        device: inner.devices[d].ordinal,
        ptr: k.slot_base + i as u64 * k.slot_span,
        nbytes: k.slot_span,
    })
}
