//! The order `make_room` takes a process's own pages in, pure (no device, clock, I/O or thread):
//! other sets' idle pages (those no open plan uses, least recently used first; then those a
//! later step uses, furthest first); then the current step's own pages, furthest next use first
//! (Belady). Never a pinned or filling page; a page whose release event is pending is taken once
//! it completes. Also what prefetch maps and where a group's sets are placed ([`place`]).

use std::cmp::Reverse;

use crate::table::{Key, Table, SLOTS};

/// An own page `make_room` may take: its bytes, and whether a release event is still pending on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Victim {
    pub key: Key,
    pub bytes: u64,
    pub releasing: bool,
}

/// A next use, furthest first: none at all, then the most steps, passes, blocks ahead.
type Far = Reverse<(bool, Option<(u32, u32, u32)>)>;

fn far(d: Option<(u32, u32, u32)>) -> Far {
    Reverse((d.is_none(), d))
}

/// Everything `make_room` may take, in its order; `keep` (the pages being bound) never.
pub fn victims(t: &Table, keep: &[Key]) -> Vec<Victim> {
    let pages = || t.pages().filter(|(k, p)| p.mapped && !p.filling && p.pins.is_empty() && !keep.contains(k));
    let page = |(key, p): (Key, &crate::table::Page)| Victim { key, bytes: p.bytes, releasing: !p.releasing.is_empty() };
    let mut idle: Vec<(Key, &crate::table::Page)> = pages().filter(|(k, _)| t.distance(*k).is_none()).collect();
    idle.sort_by_key(|(k, p)| (p.last_use, *k));
    let mut later: Vec<((u32, u32, u32), Key, &crate::table::Page)> = pages().filter_map(|(k, p)| t.distance(k).filter(|d| d.0 > 0).map(|d| (d, k, p))).collect();
    later.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    // The current step's pages, by (next use, the page).
    let mut own: Vec<((Far, Key), Victim)> = pages().filter_map(|(k, p)| t.distance(k).filter(|d| d.0 == 0).map(|d| ((far(Some(d)), k), page((k, p))))).collect();
    own.sort_by_key(|(rank, _)| *rank);
    idle.into_iter()
        .chain(later.into_iter().map(|(_, k, p)| (k, p)))
        .map(page)
        .chain(own.into_iter().map(|(_, v)| v))
        .collect()
}

/// What prefetch maps for `req` into `free` bytes only: the next [`SLOTS`] unmapped blocks of its
/// current step's sets past their cursors, nearest first, while they fit.
pub fn prefetch(t: &Table, req: u64, free: u64) -> Vec<Key> {
    let Some(r) = t.request(req) else { return vec![] };
    let mut ahead: Vec<((u32, u32, u32), Key, u64)> = t
        .pages()
        .filter(|(_, p)| !p.mapped)
        .filter_map(|(k, p)| r.distance(k).filter(|d| d.0 == 0 && d.1 == 0 && u64::from(d.2) < SLOTS).map(|d| (d, k, p.bytes)))
        .collect();
    ahead.sort();
    let mut free = free;
    ahead
        .into_iter()
        .take_while(|(.., b)| {
            let fits = *b <= free;
            free = free.saturating_sub(*b);
            fits
        })
        .map(|(_, k, _)| k)
        .collect()
}

/// Where each set of a group's package lives, as an index into the group's GPUs. `rooms[g]` is
/// what GPU g holds beside what cannot move there (on GPU 0, the package's measured working set
/// too); `sets[i]` is set i's bytes, `None` when unknown. Largest first: GPU 0 keeps a set that
/// fits there, else the GPU with the most room that holds it takes it, else it stays on GPU 0
/// (it pages there). An unknown set goes where the most room is left.
pub fn place(rooms: &[u64], sets: &[Option<u64>]) -> Vec<u32> {
    let mut left = rooms.to_vec();
    let mut out = vec![0; sets.len()];
    let mut order: Vec<usize> = (0..sets.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(sets.get(i).copied().flatten().map_or(0, |b| b + 1)));
    let most = |left: &[u64], holds: &dyn Fn(u64) -> bool| left.iter().enumerate().filter(|(_, r)| holds(**r)).max_by_key(|(g, r)| (**r, std::cmp::Reverse(*g))).map(|(g, _)| g);
    for i in order {
        let g = match sets.get(i).copied().flatten() {
            Some(bytes) if left.first().is_some_and(|r| *r >= bytes) => 0,
            Some(bytes) => most(&left, &|r| r >= bytes).unwrap_or(0),
            None => most(&left, &|_| true).unwrap_or(0),
        };
        if let (Some(r), Some(bytes)) = (left.get_mut(g), sets.get(i).copied().flatten()) {
            *r = r.saturating_sub(bytes);
        }
        if let Some(o) = out.get_mut(i) {
            *o = g as u32;
        }
    }
    out
}
