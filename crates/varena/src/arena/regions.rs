//! The arenas' hooks for regions (varena-leases §5, §11).
//! - Residency (WDDM Evict/MakeResident): a region arena's chunks go back to the source and come
//!   again at the same addresses, so its blocks' pointers never change. An evicted span keeps its
//!   blocks and directory entry with no chunks. One that comes whole free stays until it is made
//!   resident again, or gives its addresses back once its arena ended.
//! - A graph region's use of its pool, for its life.

use std::collections::BTreeSet;
use std::sync::atomic::Ordering::{AcqRel, Acquire};
use std::sync::Arc;

use platform::handles::Owned;
use platform::memory;

use super::{Arena, Chunk, Kind, Owner, PoolId, Shared, State, Torch};
use crate::backend::{Event, Va};
use crate::{Error, Result};

/// A swapped region's bytes: a RAM file (a memfd, a pagefile section) mapped here.
struct SwapFile {
    _file: Owned,
    base: u64,
    len: u64,
}

impl SwapFile {
    fn new(len: u64) -> Result<SwapFile> {
        let g = memory::granularity().max(1);
        let len = len.max(1).div_ceil(g) * g;
        let file = memory::page(len).map_err(host)?;
        let base = memory::reserve(len).map_err(host)?;
        if let Err(e) = memory::map_page(base, &file, len, true) {
            memory::release(base, len);
            return Err(host(e));
        }
        Ok(SwapFile { _file: file, base, len })
    }

    fn at(&mut self, off: u64, len: u64) -> Result<&mut [u8]> {
        if off.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(Error::Internal(format!("a swap of {} bytes has no {len} at {off}", self.len)));
        }
        // SAFETY: inside this file's live read-write mapping, which only `&mut self` reaches.
        Ok(unsafe { std::slice::from_raw_parts_mut((self.base + off) as *mut u8, len as usize) })
    }
}

impl Drop for SwapFile {
    fn drop(&mut self) {
        let _ = memory::unmap_page(self.base, self.len);
        memory::release(self.base, self.len);
    }
}

fn host(e: platform::Error) -> Error {
    Error::io("a region's swap file", e)
}

/// What an eviction took: each span (base, length, offset in the swap file) and, swapped, its bytes.
#[derive(Default)]
pub struct Evicted {
    pub bytes: u64,
    spans: Vec<(u64, u64, u64)>,
    swap: Option<SwapFile>,
}

impl Evicted {
    pub fn swapped(&self) -> bool {
        self.swap.is_some()
    }
}

impl Torch {
    /// Evicts region `owner`'s spans once their streams passed everything enqueued so far: their
    /// chunks go back to the source, and with `swap` their bytes go to a RAM file first. The caller
    /// guarantees nothing allocates into the region meanwhile (it is unpinned). `None` while a
    /// capture is open: no epoch can be recorded then (R2), so the caller asks again later.
    pub fn evict(&self, owner: Owner, swap: bool) -> Result<Option<Evicted>> {
        if !matches!(owner, Owner::Region(_)) {
            return Err(Error::Invalid(format!("only a region's arena is evicted, not {owner:?}")));
        }
        let sh = &self.sh;
        let Some(a) = sh.arenas.lock().get(&owner).cloned() else { return Ok(Some(Evicted::default())) };
        let streams: Vec<usize> = {
            let st = a.st.lock();
            let spans = st.spans.values().filter_map(|s| s.clock.as_ref().map(|c| c.stream));
            spans.chain(st.uses.values().flatten().copied()).collect::<BTreeSet<_>>().into_iter().collect()
        };
        if !sh.quiesce(&streams)? {
            return Ok(None);
        }
        let spans: Vec<(u64, u64)> = a.st.lock().spans.iter().filter(|(_, s)| !s.chunks.is_empty()).map(|(b, s)| (*b, s.len)).collect();
        let mut file = if swap { Some(SwapFile::new(spans.iter().map(|(_, l)| l).sum())?) } else { None };
        let mut out = Evicted::default();
        for (base, len) in spans {
            if let Some(f) = file.as_mut() {
                // The region's streams passed their epochs: nothing to order the copy after.
                sh.backend.copy_out(Va { base, len }, 0, f.at(out.bytes, len)?, Event::DONE)?;
            }
            let chunks = {
                let mut st = a.st.lock();
                let Some(span) = st.spans.get_mut(&base) else { continue };
                let chunks = std::mem::take(&mut span.chunks);
                if !chunks.is_empty() {
                    st.mapped = st.mapped.saturating_sub(len);
                    st.hide(base);
                }
                chunks
            };
            if chunks.is_empty() {
                continue;
            }
            // A chunk that cannot go back stays mapped and counted, its range the span's still.
            sh.unmap(Va { base, len }, chunks, false, true);
            out.spans.push((base, len, out.bytes));
            out.bytes += len;
        }
        out.swap = file;
        Ok(Some(out))
    }

    /// Maps `evicted`'s spans again at their addresses, all or none, and copies swapped bytes back;
    /// `evicted` is empty after. Never on a capturing thread (R6). True when nothing was swapped:
    /// the bytes are new.
    pub fn make_resident(&self, owner: Owner, evicted: &mut Evicted) -> Result<bool> {
        let sh = &self.sh;
        if sh.capturing() {
            return Err(Error::Invalid("a region is made resident before a capture, never during one (R6)".into()));
        }
        let fresh = !evicted.swapped();
        let Some(a) = sh.arenas.lock().get(&owner).cloned() else {
            *evicted = Evicted::default();
            return Ok(fresh);
        };
        let mut got: Vec<(u64, u64, u64, Vec<Chunk>)> = Vec::new();
        for &(base, len, off) in &evicted.spans {
            if !a.st.lock().spans.get(&base).is_some_and(|s| s.chunks.is_empty()) {
                continue;
            }
            match sh.map_again(Va { base, len }) {
                Ok(chunks) => got.push((base, len, off, chunks)),
                Err(e) => {
                    got.into_iter().for_each(|(base, len, _, chunks)| {
                        sh.unmap(Va { base, len }, chunks, false, false);
                    });
                    return Err(e);
                }
            }
        }
        let Evicted { mut swap, .. } = std::mem::take(evicted);
        for (base, len, off, chunks) in got {
            {
                let mut st = a.st.lock();
                let Some(span) = st.spans.get_mut(&base) else {
                    drop(st);
                    sh.unmap(Va { base, len }, chunks, false, false);
                    continue;
                };
                span.chunks = chunks;
                st.mapped += len;
                st.show(base);
                sh.mapped.fetch_add(len, AcqRel);
            }
            if let Some(f) = swap.as_mut() {
                sh.backend.copy_in(Va { base, len }, 0, f.at(off, len)?)?;
            }
            let mut st = a.st.lock();
            if st.spans.get(&base).is_some_and(|s| s.live == 0) {
                sh.whole(&a, &mut st, base);
            }
        }
        Ok(fresh)
    }

    /// One more use of `pool` that is no capture: a graph region's, for its life ([`Torch::pool_release`]
    /// ends it). The pool's spans stay while any use is left (R4/R5).
    pub fn pool_retain(&self, pool: PoolId) {
        let a = self.sh.arena(Owner::Pool(pool));
        let mut st = a.st.lock();
        st.users += 1;
        self.sh.revive(&a, &mut st);
    }

    /// Every span mapped now: the arenas' and the idle list's (an atomic read).
    pub fn mapped(&self) -> u64 {
        self.sh.mapped.load(Acquire)
    }

    /// The bytes `owner`'s arena maps now.
    pub fn mapped_by(&self, owner: Owner) -> u64 {
        self.sh.arenas.lock().get(&owner).cloned().map_or(0, |a| a.st.lock().mapped)
    }

    /// The source's slots of the chunks `owner`'s arena maps now.
    pub fn slots(&self, owner: Owner) -> Vec<u32> {
        let Some(a) = self.sh.arenas.lock().get(&owner).cloned() else { return Vec::new() };
        let st = a.st.lock();
        st.spans.values().flat_map(|s| s.chunks.iter()).filter_map(|c| self.sh.source.slot(c)).collect()
    }

    /// Before a region's arena ends: its evicted spans with no live block give their addresses back.
    pub fn forget_evicted(&self, owner: Owner) {
        let Some(a) = self.sh.arenas.lock().get(&owner).cloned() else { return };
        let mut st = a.st.lock();
        let gone: Vec<u64> = st.spans.iter().filter(|(_, s)| s.chunks.is_empty() && s.live == 0).map(|(b, _)| *b).collect();
        gone.into_iter().for_each(|b| self.sh.unreserve(&mut st, b));
    }
}

impl State {
    /// An evicted span's free runs and open slot pages leave the free lists: nothing is carved there
    /// (its frees keep them out: [`State::unfit`], [`State::unslot`]).
    fn hide(&mut self, base: u64) {
        let Some((stream, tier, end)) = self.spans.get(&base).map(|s| (s.stream(), s.tier, base + s.len)) else { return };
        let runs: Vec<(u64, u64)> = self.blocks.range(base..end).filter(|(_, b)| b.free && b.span == base).map(|(at, b)| (*at, b.len)).collect();
        runs.into_iter().for_each(|(at, len)| {
            self.free.remove(&(stream, tier, len, at));
        });
        let pages: Vec<(u64, (usize, usize))> = self.pages.range(base..end).map(|(at, p)| (*at, (p.stream, p.class))).collect();
        for (at, key) in pages {
            if let Some(v) = self.open.get_mut(&key) {
                v.retain(|p| *p != at);
            }
        }
    }

    /// A span resident again: its free runs and slot pages with a free slot are carved from again.
    fn show(&mut self, base: u64) {
        let Some((stream, tier, end)) = self.spans.get(&base).map(|s| (s.stream(), s.tier, base + s.len)) else { return };
        let runs: Vec<(u64, u64)> = self.blocks.range(base..end).filter(|(_, b)| b.free && b.span == base).map(|(at, b)| (*at, b.len)).collect();
        runs.into_iter().for_each(|(at, len)| {
            self.free.insert((stream, tier, len, at));
        });
        let pages: Vec<(u64, (usize, usize))> = self.pages.range(base..end).filter(|(_, p)| p.free != 0).map(|(at, p)| (*at, (p.stream, p.class))).collect();
        for (at, key) in pages {
            let v = self.open.entry(key).or_default();
            if !v.contains(&at) {
                v.push(at);
            }
        }
    }
}

impl Shared {
    /// Waits until every one of `streams` passed an epoch after its work so far; false while a capture
    /// is open. A stream whose recording was lost can prove nothing: an error.
    fn quiesce(&self, streams: &[usize]) -> Result<bool> {
        let Some(need) = self.stamps(streams) else { return Ok(false) };
        loop {
            let seen = *self.progress.0.lock();
            if let Some((c, _)) = need.iter().find(|(c, n)| !c.passed(*n) && c.lost.load(Acquire)) {
                return Err(Error::Internal(format!("stream {:#x} lost its epochs: a region on it cannot be proven idle", c.stream)));
            }
            if need.iter().all(|(c, n)| c.passed(*n)) {
                return Ok(true);
            }
            self.kick();
            let mut p = self.progress.0.lock();
            while *p == seen {
                self.progress.1.wait(&mut p);
            }
        }
    }

    /// The source's chunks for an evicted span's addresses; a short source first gets idle spans back.
    fn map_again(&self, va: Va) -> Result<Vec<Chunk>> {
        // Never over a chunk that could not go back: it is still mapped there.
        if self.stuck_in(va) {
            self.retry_stuck();
            if self.stuck_in(va) {
                return Err(Error::Internal(format!("a chunk of the span at {va:?} could not go back: it is not made resident over it")));
            }
        }
        loop {
            let mut chunks = Vec::new();
            match self.source.map(va, Kind::Torch, &mut chunks) {
                Ok(()) => return Ok(chunks),
                Err(e) => {
                    self.unmap(va, chunks, false, false);
                    if !(matches!(e, Error::Exhausted(_)) && self.give_back()) {
                        return Err(e);
                    }
                }
            }
        }
    }

    /// An evicted span came whole free: it stays for the region's next residency, or gives its
    /// addresses back once its arena ended.
    pub(super) fn evicted_whole(&self, a: &Arc<Arena>, st: &mut State, base: u64) {
        if a.ended.load(Acquire) {
            self.unreserve(st, base);
        }
    }

    /// An evicted span leaves its arena: nothing is mapped there, so its addresses go back.
    fn unreserve(&self, st: &mut State, base: u64) {
        let Some(span) = st.spans.remove(&base) else { return };
        st.free.remove(&(span.stream(), span.tier, span.len, base));
        st.blocks.remove(&base);
        self.dir.write().remove(&base);
        self.release_va(Va { base, len: span.len });
    }
}
