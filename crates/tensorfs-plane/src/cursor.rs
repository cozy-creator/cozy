//! The streaming cursor: one stage's region schedule on one device.
//!
//! Regions resident at `stream()` are served in place and held for the cursor's life; the rest
//! stream through a ring of `window` device slots. The k-th streamed position lands in slot
//! `k % window`; its copy waits on the release event of position `k - window` (the slot's last
//! reader) and records `ready[k]`. `acquire` enqueues `cuStreamWaitEvent(consumer, ready[k])`
//! (no host block once the mover has recorded it); `release` records `freed[k]` and queues the
//! copy for `k + window`. The schedule is the caller's: `acquire` of any other region than the
//! next one is refused.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::cuda::{self, CUstream};
use crate::device::{self, Chunk, CopyJob, CopySrc, Copied, EvRef};
use crate::plane::{
    devws, DState, HState, Inner, Lease, LeaseKind, Plane, Timing, TimingKind, Trim, WsId,
};
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
    total: u64,
    window: u64,
    pos: u64,
    pub slot_base: u64,
    pub slot_span: u64,
    pub slots: Vec<Vec<Chunk>>,
    /// k -> Ok(ready event) once the mover recorded it, Err(cause) if the copy failed.
    ready: BTreeMap<u64, std::result::Result<EvRef, String>>,
    /// k -> release event, kept until the copy that reuses its slot has been queued.
    freed: BTreeMap<u64, EvRef>,
    released: u64,
    next_submit: u64,
    live: u32,
    views: u32,
    mover: usize,
    pub poisoned: Option<String>,
}

impl Cur {
    fn streamed_total(&self) -> u64 {
        self.streamed.len() as u64 * (self.total / self.order.len() as u64)
    }
    fn k_of(&self, p: u64) -> Option<u64> {
        let n = self.order.len() as u64;
        let i = (p % n) as usize;
        (!self.home[i]).then(|| (p / n) * self.streamed.len() as u64 + self.rank[i])
    }
}

/// Queue the copy for streamed position `k` (its slot is free once `freed[k - window]`).
fn submit(inner: &Arc<Inner>, c: &mut crate::plane::Core, id: u64, k: u64) -> Result<()> {
    let cur = c.cursors.get(&id).ok_or(Error::Closed)?;
    let (ws, d, window) = (cur.ws, cur.dev, cur.window);
    let r = cur.streamed[(k % cur.streamed.len() as u64) as usize];
    let waits: Vec<EvRef> = if k >= window {
        vec![cur.freed.get(&(k - window)).expect("slot released before reuse").clone()]
    } else {
        Vec::new()
    };
    let dst = cur.slot_base + (k % window) * cur.slot_span;
    let mover = cur.mover;
    let w = c.ws_mut(ws)?;
    let region = w.layout.regions[r as usize].clone();
    let from_host = w.hregs[r as usize].state == HState::Ready;
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
                        h.pending.push(cp.ready.clone());
                    }
                }
            }
            let tag = c.cursors.get(&id).map(|k| k.tag.clone()).unwrap_or_default();
            if let Ok(cp) = &res {
                c.dstats[d].copies += 1;
                c.tag(ws.0, d, &tag).streamed_bytes += cp.bytes;
                c.timings.push(Timing {
                    dev: d,
                    ws: ws.0,
                    kind: TimingKind::Copy { from_host },
                    a: cp.start.clone(),
                    b: cp.ready.clone(),
                    bytes: cp.bytes,
                });
            }
            if let Some(cur) = c.cursors.get_mut(&id) {
                cur.ready.insert(k, res.map(|cp| cp.ready).map_err(|e| e.to_string()));
            }
            drop(c);
            inner2.cv.notify_all();
        }),
    };
    if k >= window {
        c.cursors.get_mut(&id).unwrap().freed.remove(&(k - window));
    }
    c.cursors.get_mut(&id).unwrap().next_submit = k + 1;
    if let Err(e) = inner.devices[d].movers[mover].submit(job) {
        if from_host {
            c.ws_mut(ws)?.hregs[r as usize].holds -= 1;
        }
        c.cursors.get_mut(&id).unwrap().ready.insert(k, Err(e.to_string()));
    }
    Ok(())
}

pub(crate) fn release_slot(inner: &Arc<Inner>, id: u64, k: u64, stream: u64) -> Result<()> {
    let d = inner.core.lock().unwrap().cursors.get(&id).ok_or(Error::Closed)?.dev;
    let rt = &inner.devices[d];
    let _g = rt.bind()?;
    let ev = rt.events.get(false)?;
    ev.record(stream as CUstream)?;
    let mut c = inner.core.lock().unwrap();
    let cur = c.cursors.get_mut(&id).ok_or(Error::Closed)?;
    cur.freed.insert(k, ev);
    cur.ready.remove(&k);
    cur.live -= 1;
    cur.released += 1;
    let next = k + cur.window;
    if next < cur.streamed_total() && next == cur.next_submit {
        // Queue every copy whose slot is now free, in order.
        let mut k2 = next;
        while k2 < c.cursors[&id].streamed_total()
            && k2 == c.cursors[&id].next_submit
            && c.cursors[&id].freed.contains_key(&(k2 - c.cursors[&id].window))
        {
            submit(inner, &mut c, id, k2)?;
            k2 += 1;
        }
    }
    Ok(())
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
    /// Start streaming `order` (repeated `repeat` times) on device `ordinal` through `window`
    /// slots. Slots may evict regions below `prio`. `tag` names the stage in telemetry.
    pub fn stream(&self, ws: WsId, ordinal: i32, order: &[u32], window: u32, repeat: u32, prio: i64, tag: &str) -> Result<Cursor> {
        if order.is_empty() || window == 0 || repeat == 0 {
            return Err(Error::Invalid("stream needs a non-empty order, window >= 1, repeat >= 1".into()));
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
            let r = c.arenas[d].take(&sizes).and_then(|ch| {
                match device::map_span(slot_base + i * slot_span, &ch, inner.devices[d].dev) {
                    Ok(()) => Ok(ch),
                    Err(e) => {
                        c.arenas[d].put_idle(ch);
                        Err(e)
                    }
                }
            });
            match r {
                Ok(ch) => {
                    c.arenas[d].mapped += slot_span;
                    slots.push(ch);
                }
                Err(e) => {
                    for (j, ch) in slots.drain(..).enumerate() {
                        let _ = device::unmap_span(slot_base + j as u64 * slot_span, slot_span);
                        c.arenas[d].mapped -= slot_span;
                        c.arenas[d].put_idle(ch);
                    }
                    let _ = device::free_va(slot_base, need);
                    return Err(e);
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
                total: order.len() as u64 * repeat as u64,
                window,
                pos: 0,
                slot_base,
                slot_span,
                slots,
                ready: BTreeMap::new(),
                freed: BTreeMap::new(),
                released: 0,
                next_submit: 0,
                live: 0,
                views: 0,
                mover,
                poisoned: None,
            },
        );
        let first = window.min(c.cursors[&id].streamed_total());
        for k in 0..first {
            submit(inner, &mut c, id, k)?;
        }
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
    pub fn position(&self) -> u64 {
        self.inner.core.lock().unwrap().cursors.get(&self.id).map_or(0, |k| k.pos)
    }

    /// Lease the next scheduled region for `stream`.
    pub fn acquire(&self, r: u32, stream: u64) -> Result<Lease> {
        let inner = &self.inner;
        let d = inner.core.lock().unwrap().cursors.get(&self.id).ok_or(Error::Closed)?.dev;
        let dev = &inner.devices[d];
        let _g = dev.bind()?;
        let mut c = inner.core.lock().unwrap();
        let cur = c.cursors.get(&self.id).ok_or(Error::Closed)?;
        if let Some(p) = &cur.poisoned {
            return Err(Error::Poisoned(p.clone()));
        }
        let p = cur.pos;
        if p >= cur.total {
            return Err(Error::Invalid(format!("cursor finished its {} positions", cur.total)));
        }
        let expected = cur.order[(p % cur.order.len() as u64) as usize];
        if r != expected {
            return Err(Error::ScheduleMismatch { position: p, expected, got: r });
        }
        let (ws, tag) = (cur.ws, cur.tag.clone());
        let Some(k) = cur.k_of(p) else {
            // Served in place: the region is held, so it is still mapped.
            drop(c);
            let lease = inner.acquire_home(ws, d, r, stream, &tag)?;
            inner.core.lock().unwrap().cursors.get_mut(&self.id).ok_or(Error::Closed)?.pos += 1;
            return Ok(lease);
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
                Some(Err(e)) => return Err(Error::Invalid(format!("streamed copy of region {r} failed: {e}"))),
                None => c = inner.cv.wait(c).unwrap(),
            }
        };
        let host_wait = t0.elapsed().as_nanos() as u64;
        let nbytes = c.ws(ws)?.layout.regions[r as usize].nbytes;
        let late = !ev.done()?;
        let s = stream as CUstream;
        let stall = if late { Some((dev.events.get(true)?, dev.events.get(true)?)) } else { None };
        if let Some((a, _)) = &stall {
            a.record(s)?;
        }
        let drv = cuda::driver()?;
        // SAFETY: consumer stream of this context; the event is live.
        cuda::check("cuStreamWaitEvent", unsafe { (drv.stream_wait_event)(s, ev.raw(), 0) })?;
        if let Some((_, b)) = &stall {
            b.record(s)?;
        }
        let tc = c.tag(ws.0, d, &tag);
        tc.acquires += 1;
        tc.stall_ns += host_wait;
        if late {
            tc.late += 1;
            tc.late_bytes += nbytes;
        }
        if let Some((a, b)) = stall {
            c.timings.push(Timing { dev: d, ws: ws.0, kind: TimingKind::Stall { tag }, a, b, bytes: nbytes });
        }
        let cur = c.cursors.get_mut(&self.id).unwrap();
        cur.pos += 1;
        cur.live += 1;
        let slot = k % cur.window;
        Ok(Lease {
            inner: inner.clone(),
            ptr: cur.slot_base + slot * cur.slot_span,
            nbytes,
            region: r,
            slot: Some(slot as u32),
            kind: LeaseKind::Slot { cursor: self.id, k },
            released: AtomicBool::new(false),
        })
    }

    /// Return slots and holds. Refuses while a streamed lease is live; waits (targeted) for
    /// copies in flight and for the device to pass every release.
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
        let d = cur.dev;
        // Every queued copy must have been recorded before its slot can be unmapped.
        loop {
            let cur = &c.cursors[&self.id];
            if (cur.released..cur.next_submit).all(|k| cur.ready.contains_key(&k)) {
                break;
            }
            c = inner.cv.wait(c).unwrap();
        }
        let cur = c.cursors.remove(&self.id).unwrap();
        let _g = inner.devices[d].bind()?;
        for e in cur.ready.values().flatten().chain(cur.freed.values()) {
            e.sync()?;
        }
        for (i, ch) in cur.slots.into_iter().enumerate() {
            if let Err(e) = device::unmap_span(cur.slot_base + i as u64 * cur.slot_span, cur.slot_span) {
                c.arenas[d].poisoned = Some(format!("slot unmap failed: {e}"));
                return Err(e);
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
        drop(c);
        inner.cv.notify_all();
        Ok(())
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
        if let Some(k) = self.inner.core.lock().unwrap().cursors.get_mut(&self.id) {
            k.views -= 1;
        }
    }
}

impl Cursor {
    pub fn slot_view(&self, i: u32) -> Result<SlotView> {
        let mut c = self.inner.core.lock().unwrap();
        let k = c.cursors.get_mut(&self.id).ok_or(Error::Closed)?;
        if i as u64 >= k.window {
            return Err(Error::Invalid(format!("slot {i} of {}", k.window)));
        }
        k.views += 1;
        let d = k.dev;
        Ok(SlotView {
            inner: self.inner.clone(),
            id: self.id,
            device: self.inner.devices[d].ordinal,
            ptr: k.slot_base + i as u64 * k.slot_span,
            nbytes: k.slot_span,
        })
    }
}

impl Drop for Cursor {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
