//! The order `make_room` takes a process's own pages in, pure (no device, clock, I/O or thread):
//! other sets' idle pages (those no open plan uses, least recently used first; then those a
//! later step uses, furthest first); then the current step's own pages, furthest next use first
//! (Belady). Never a pinned or filling page; a page whose release event is pending is taken once
//! it completes. Also what prefetch maps, and the ledger's admission ([`fits`]) and placement
//! ([`place`]) until the lease manager's replace them.

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

/// A row's facts as admission reads them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Admitted {
    pub context: u64,
    /// Pinned bytes outside any running W.
    pub outside: u64,
    /// Each running request's largest remaining W, summed.
    pub remaining: u64,
    pub running: u32,
    /// Running requests with an unmeasured step left.
    pub unmeasured: u32,
    /// The remaining W of its requests waiting on child calls, per family they wait for
    /// (family, W); family 0: any (beyond the slots), counted for every child.
    pub yielded: [(u64, u64); YIELDED],
}

/// Families a row publishes yielded W for, each apart; more fold into one counted for all.
pub const YIELDED: usize = 3;

/// A row's yielded W as a child of `family` sees it: all but its own family's parents'.
pub fn yielded_beside(yielded: &[(u64, u64); YIELDED], family: Option<u64>) -> u64 {
    yielded.iter().filter(|(f, _)| family != Some(*f) || *f == 0).map(|(_, w)| w).sum()
}

/// Whether a request whose W is `w` (its recorded peak; `None`: unmeasured) fits beside `rows`
/// and the dispatched requests not yet running: contexts + pins outside a running W + each
/// running request's peak + this one's ≤ capacity less foreign bytes. An idle GPU always fits (a
/// request larger than the card streams); anything unmeasured fits only there. A request
/// waiting on a child call counts too, except for its own family's child (`family`).
pub fn fits(capacity: u64, foreign: u64, rows: &[Admitted], dispatched: &[Option<u64>], w: Option<u64>, family: Option<u64>) -> bool {
    if rows.iter().all(|r| r.running == 0) && dispatched.is_empty() {
        return true;
    }
    if w.is_none() || rows.iter().any(|r| r.unmeasured > 0) || dispatched.iter().any(Option::is_none) {
        return false;
    }
    let rows: u64 = rows.iter().map(|r| r.context + r.outside + r.remaining + yielded_beside(&r.yielded, family)).sum();
    let dispatched: u64 = dispatched.iter().flatten().sum();
    rows + dispatched + w.unwrap_or(0) <= capacity.saturating_sub(foreign)
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
