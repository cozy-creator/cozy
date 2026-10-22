//! Step 2's watermark (audit §4). Once `make_room` has to take the running set's own pages, that set streams its upper
//! blocks (those at or above its watermark) through [`SLOTS`] physical pages. Each slot moves to its next block's own
//! VA in turn, so a graph captured against a block's VA stays valid. Weights then never cause an OOM: the floor is the
//! activations plus the slots. The watermark resets when its request ends.

use std::cmp::Reverse;
use std::sync::Arc;

use parking_lot::MutexGuard;

use super::{Home, Inner, Job, Mech, State, COPY};
use crate::backend::{Access, Event, Physical, Va};
use crate::table::{Class, Key};
use crate::{Origin, Result};

/// One being filled, one being read, one queued: the slots a forward needs to overlap its copies and its compute.
pub(super) const SLOTS: usize = 3;

/// A set streaming its blocks at or above `watermark`, for the request that made it stream.
pub(super) struct Streaming {
    pub(super) watermark: u32,
    pub(super) req: Option<u64>,
}

/// What [`Inner::stream`] took: pages unmapped now, and pages whose pending release events (`events`) the caller
/// waits on before a later round takes them.
#[derive(Default)]
pub(super) struct Streamed {
    pub(super) unmapped: u64,
    pub(super) pending: u64,
    pub(super) events: Vec<u64>,
}

/// What [`Inner::stream_in`] did for a bind.
pub(super) enum StreamIn {
    /// A slot moves to the block: it has a page now, filling.
    Moved,
    /// Every slot is in use by another thread: wait for a release and look again.
    Wait,
    /// Not a streamed block, or too few slots yet: the page is made the usual way.
    No,
}

impl State {
    /// A page that is the set's slot now: mapped at an upper block, never custody's (its memory is overwritten next).
    fn slot(&self, key: Key, watermark: u32) -> bool {
        key.1 >= watermark && self.homes.get(&key).is_some_and(|h| !h.exported && !h.keeping)
    }

    /// No pin, and nothing writes or protects it: a slot that may move.
    fn movable(&self, key: Key) -> bool {
        self.table.page(key).is_some_and(|p| p.pins.is_empty())
            && self.homes.get(&key).is_some_and(|h| h.phys.is_some() && !h.filling && !h.landing && !h.transforming && !h.protecting)
    }

    fn last_use(&self, key: Key) -> u64 {
        self.table.page(key).map_or(0, |p| p.last_use)
    }
}

impl Inner {
    /// `make_room`'s rung before the running set's own pages. It lowers `set`'s watermark until the upper pages free
    /// `want` (unmapped now, or once their pending release events complete: `events`, waited on by the caller with
    /// no lock held), or until the whole set streams. It keeps the [`SLOTS`] most recently used upper pages that are
    /// not custody's as slots, since a slot's memory is overwritten next. The hook (`hook`) never takes a kept page:
    /// it makes no custody call.
    pub(super) fn stream(&self, st: &mut State, set: u32, want: u64, req: Option<u64>, hook: bool) -> Result<Streamed> {
        let blocks = st.mech(set)?.spec.layout.regions.len() as u32;
        let mut watermark = st.streams.get(&set).map_or(blocks, |s| s.watermark);
        let mut taken = Streamed::default();
        loop {
            let mut upper: Vec<Key> = st.homes.keys().filter(|k| k.0 == set && k.1 >= watermark).copied().collect();
            upper.sort_by_key(|k| Reverse(st.last_use(*k)));
            let slots: Vec<Key> = upper.iter().filter(|k| st.slot(**k, watermark)).take(SLOTS).copied().collect();
            for key in upper.into_iter().filter(|k| !slots.contains(k)) {
                if taken.unmapped + taken.pending >= want {
                    break;
                }
                let Some(page) = st.table.page(key) else { continue };
                // A page in use stays until its release; the hook leaves custody's pages alone.
                let busy = !page.pins.is_empty() || st.homes.get(&key).is_some_and(|h| h.filling || h.landing || h.transforming || h.protecting);
                if busy || (hook && page.kept) {
                    continue;
                }
                let (releasing, span) = (page.releasing.clone(), page.bytes);
                if !releasing.is_empty() {
                    // Taken in a later round, once its consumers are done.
                    taken.events.extend(releasing);
                    taken.pending += span;
                    continue;
                }
                self.unmap(st, key, Class::W, req, !hook)?;
                taken.unmapped += span;
            }
            if taken.unmapped + taken.pending >= want || watermark == 0 {
                break;
            }
            watermark -= 1;
        }
        st.streams.insert(set, Streaming { watermark, req });
        Ok(taken)
    }

    /// A streamed set's upper block with no page: the least recently used movable slot of its set moves to it, after
    /// the consumers of the block it backed. It falls back to the usual path while the set has fewer than [`SLOTS`]
    /// slots, or when this thread itself pins every one of them (waiting would wait on itself).
    pub(super) fn stream_in(&self, st: &mut MutexGuard<'_, State>, key: Key, req: Option<u64>) -> Result<StreamIn> {
        let Some(watermark) = st.streams.get(&key.0).map(|s| s.watermark) else { return Ok(StreamIn::No) };
        if key.1 < watermark || st.homes.contains_key(&key) {
            return Ok(StreamIn::No);
        }
        let slots: Vec<Key> = st.homes.keys().filter(|k| k.0 == key.0 && st.slot(**k, watermark)).copied().collect();
        if slots.len() < SLOTS {
            return Ok(StreamIn::No);
        }
        let Some(from) = slots.iter().filter(|k| st.movable(**k)).min_by_key(|k| st.last_use(**k)).copied() else {
            let me = std::thread::current().id();
            let mine = slots.iter().all(|k| st.pins.values().any(|(p, _, t)| p == k && *t == me));
            return Ok(if mine { StreamIn::No } else { StreamIn::Wait });
        };
        let (va, from_off, span) = st.place(from)?;
        let (_, off, need) = st.place(key)?;
        // A slot backs a block of its own size only: another size gets a page the usual way.
        if need != span {
            return Ok(StreamIn::No);
        }
        // The block it backed: its release events are the move's to wait on, then retire.
        let after: Vec<Event> = st.table.page_mut(from).map(|p| std::mem::take(&mut p.releasing)).unwrap_or_default().into_iter().map(Event).collect();
        st.events.values_mut().for_each(|q| q.retain(|(_, k)| *k != from));
        let Some(Home { phys: Some(phys), ready, .. }) = st.homes.remove(&from) else { return Ok(StreamIn::Wait) };
        self.backend.retire(ready);
        st.sync(from);
        let gen = st.tick();
        let identity = st.mech(key.0)?.key.clone();
        // Counted mapped, filling, now: the mechanism moves the slot (no new memory), then queues the fill.
        st.homes.insert(key, Home { phys: None, ready: Event::DONE, ready_on: COPY, gen, filled: false, filling: true, transforming: false, exported: false, origin: Origin::Store, speculative: false, identity, keep_after: false, keeping: false, protecting: false, reclaiming: false, landing: false });
        st.sync(key);
        self.mech.post(Mech::Move { key, gen, va, from: from_off, off, span, phys, after, job: Job { key, gen, req, span, after: vec![] } }, 0);
        Ok(StreamIn::Moved)
    }

    /// Its request ended: the sets it made stream go back to binding the usual way.
    pub(super) fn unstream(&self, st: &mut State, req: u64) {
        st.streams.retain(|_, s| s.req != Some(req));
    }

    /// [`Mech::Move`] on the mechanism thread: after the old block's consumers, its VA goes, and the slot maps
    /// read-write at the new block's VA; then the new block's fill is queued. A block unmapped meanwhile lets the
    /// slot go.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn move_slot(self: &Arc<Self>, key: Key, gen: u64, va: Va, from: u64, off: u64, span: u64, phys: Physical, after: Vec<Event>, job: Job) {
        let moved = after.iter().try_for_each(|e| self.backend.finish(*e))
            .and_then(|()| self.backend.unmap(va, from, span))
            .and_then(|()| self.backend.map(va, off, &phys, Access::ReadWrite));
        if let Err(error) = moved {
            let mut st = self.state.lock();
            self.lost_backing(&mut st, span, false, error);
            return;
        }
        after.into_iter().for_each(|e| self.backend.retire(e));
        let mut st = self.state.lock();
        if !st.homes.get(&key).is_some_and(|h| h.gen == gen) {
            drop(st);
            if let Err(error) = self.backend.unmap(va, off, span).and_then(|()| self.backend.destroy(phys)) {
                let mut st = self.state.lock();
                self.lost_backing(&mut st, span, false, error);
            }
            return;
        }
        if let Some(h) = st.homes.get_mut(&key) {
            h.phys = Some(phys);
        }
        drop(st);
        if self.queue_fills(vec![job], false).is_err() {
            tracing::warn!("varena: the fill threads are gone");
        }
    }
}
