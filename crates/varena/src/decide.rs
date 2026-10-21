//! The decision procedure: [`decide`] maps a [`Table`] and a [`Trigger`] to [`Action`]s, pure
//! (no device, clock, I/O or thread), in the design's §4 order. Own evictions are certain, so
//! one call plans them through; a remote rung (orphans, asks) is uncertain, so a call ends
//! there and the applier decides again once it is answered, with the rung marked [`Tried`].
//! Admission ([`fits`]) lives here too, as its only implementation.

use std::collections::BTreeSet;

use crate::table::{Class, Key, Table};

/// Remote rungs already tried in this demand.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tried {
    /// The trim hook ran.
    pub trimmed: bool,
    pub orphans: bool,
    /// (row, ceiling) asked.
    pub asked: BTreeSet<(u32, Class)>,
    /// Sites cut at this pass start (each at most once).
    pub cut: BTreeSet<String>,
    /// Bytes the remote rungs freed for this demand.
    pub freed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// A bind of an unmapped page, for `req` when a request's pass binds it.
    Bind { page: Key, req: Option<u64>, tried: Tried },
    /// A pass of `req` starts needing `room`: its room and its step's unmapped pages must fit.
    Pass { req: u64, room: u64, tried: Tried },
    /// An op of `req` ran short by `need` (at least one page's worth when unknown).
    Oom { req: u64, need: Option<u64>, site: Option<String>, tried: Tried },
    /// Another row asks for `bytes` under `ceiling`.
    Ask { ceiling: Class, bytes: u64 },
    /// Fetch ahead for `req` into the surplus.
    Speculate { req: u64, tried: Tried },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Run the trim hook: the allocator's slack (part of F) becomes free, before any eviction.
    Trim(u64),
    /// Unmap an own page, of this class now.
    Unmap { page: Key, class: Class },
    /// Ask custody to drop this many orphan bytes, least recently used first.
    ReleaseOrphans(u64),
    Ask { row: u32, ceiling: Class, bytes: u64 },
    /// Ask an own cache to give back bytes.
    Shrink { cache: u32, bytes: u64 },
    /// The request's step starves: no more speculation until the next step.
    Starve(u64),
    /// Map and fill a page for use now.
    Map(Key),
    /// Map and fill a page ahead of its use; fails soft.
    Prefetch(Key),
    /// The freed bytes become the request's room for its failing op.
    Claim { req: u64, bytes: u64 },
    /// Cut `site` of the request's pass one level finer.
    Cut { req: u64, site: String },
    /// Lower the request's watermark for the rest of it: its W becomes evictable.
    Lower(u64),
    Park(u64),
    Exhausted { need: u64 },
}

/// The actions for `trigger`; the last one ends a demand (enough, a remote rung, a cut, a park
/// or exhaustion).
pub fn decide(t: &Table, trigger: &Trigger) -> Vec<Action> {
    #[cfg(feature = "faults")]
    crate::faults::hit(crate::faults::Point::Decide);
    match trigger {
        Trigger::Bind { page, req, tried } => {
            let need = t.page(*page).map_or(0, |p| p.bytes);
            let mut d = Demand::new(t, need, *req, Some(*page), tried);
            if d.run(Rungs::Bind) == Some(Done::Enough) {
                d.out.push(Action::Map(*page));
            }
            d.out
        }
        Trigger::Pass { req, room, tried } => {
            let Some(r) = t.request(*req) else { return vec![] };
            let extra = room.saturating_sub(r.room + r.claimed);
            let unmapped = |lowered: bool| {
                let step = r.steps.get(r.cursor.step as usize);
                let pages = step.into_iter().flat_map(|(sets, _)| sets.iter()).filter_map(|s| t.sets.get(*s as usize)?.as_ref().map(|set| (*s, set)));
                let sizes = pages.flat_map(|(s, set)| set.pages.iter().enumerate().filter(|(_, p)| !p.mapped).map(move |(b, p)| ((s, b as u32), p.bytes)));
                let sizes: Vec<u64> = sizes.map(|(_, b)| b).collect();
                if lowered { sizes.iter().copied().max().unwrap_or(0) } else { sizes.iter().sum() }
            };
            let mut d = Demand::new(t, extra + unmapped(r.lowered), Some(*req), None, tried);
            if let Some(Done::Lowered) = d.run(Rungs::Pass) {
                // Lowered: the step streams, so one page beside the room is enough.
                d.need = extra + unmapped(true);
                if d.run(Rungs::Lowered).is_none() {
                    d.exhausted();
                }
            }
            d.out
        }
        Trigger::Oom { req, need, site, tried } => {
            let Some(r) = t.request(*req) else { return vec![Action::Exhausted { need: need.unwrap_or(0) }] };
            // Unknown: any page's worth is progress.
            let mut d = Demand::new(t, need.unwrap_or(1), Some(*req), None, tried);
            d.oom = true;
            // Free bytes count once per episode (the cap was stale); after that only freed ones do.
            if r.episode.claimed_free {
                d.got = d.got.min(tried.freed);
            }
            match d.run(Rungs::Oom) {
                Some(Done::Remote) => return d.out,
                Some(Done::Enough) => {}
                Some(Done::Lowered) | None => {
                    let cuttable = site.as_ref().filter(|s| !r.sites.iter().any(|x| &x.name == *s && x.stuck));
                    if let Some(site) = cuttable {
                        d.out.push(Action::Cut { req: *req, site: site.clone() });
                    } else {
                        if !r.lowered {
                            d.out.push(Action::Lower(*req));
                        }
                        d.lowered = true;
                        if d.run(Rungs::Lowered).is_none() && d.got == 0 {
                            d.park_or_exhaust(r.episode.parked);
                        }
                    }
                }
            }
            if d.got > 0 {
                // A known need claims what it needs; an unknown one everything gathered.
                let bytes = if need.is_some() { d.got.min(d.need) } else { d.got };
                d.out.push(Action::Claim { req: *req, bytes });
            }
            d.out
        }
        Trigger::Ask { ceiling, bytes } => answer(t, *ceiling, *bytes),
        Trigger::Speculate { req, tried } => speculate(t, *req, tried),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Done {
    Enough,
    /// Ended on a remote rung: decide again once answered.
    Remote,
    /// Nothing left but lowering.
    Lowered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rungs {
    Bind,
    Pass,
    Oom,
    /// Only the lowered request's W, after everything else.
    Lowered,
}

struct Demand<'a> {
    t: &'a Table,
    need: u64,
    /// Bytes available so far: F plus what the planned actions free.
    got: u64,
    req: Option<u64>,
    /// The page being bound: never a victim.
    keep: Option<Key>,
    tried: &'a Tried,
    oom: bool,
    lowered: bool,
    gone: BTreeSet<Key>,
    out: Vec<Action>,
}

impl<'a> Demand<'a> {
    fn new(t: &'a Table, need: u64, req: Option<u64>, keep: Option<Key>, tried: &'a Tried) -> Demand<'a> {
        Demand { t, need, got: t.free(), req, keep, tried, oom: false, lowered: false, gone: BTreeSet::new(), out: vec![] }
    }

    fn enough(&self) -> bool {
        self.got >= self.need
    }

    /// Own pages of `class`, in eviction order.
    fn own(&self, class: Class) -> Vec<(Key, u64)> {
        let t = self.t;
        type Candidate = (Key, u64, u64, Option<(u32, u32, u32)>);
        let mut pages: Vec<Candidate> = t
            .pages()
            .filter(|(k, _)| Some(*k) != self.keep && !self.gone.contains(k) && t.class(*k) == Some(class))
            .map(|(k, p)| (k, p.bytes, p.last_use, t.distance(k)))
            .collect();
        match class {
            // Least recently used first.
            Class::N => pages.sort_by_key(|(k, _, used, _)| (*used, *k)),
            // Furthest next use first (Belady).
            _ => pages.sort_by(|a, b| b.3.cmp(&a.3).then(a.0.cmp(&b.0))),
        }
        if class == Class::W {
            // Only a lowered request's working set, after lowering.
            let lowered = |k: Key| t.user(k).is_some_and(|r| r.lowered || (self.lowered && Some(r.id) == self.req));
            pages.retain(|(k, ..)| lowered(*k));
        }
        pages.into_iter().map(|(k, b, ..)| (k, b)).collect()
    }

    /// Unmaps own pages of `class` until enough.
    fn evict(&mut self, class: Class) {
        for (page, bytes) in self.own(class) {
            if self.enough() {
                return;
            }
            if class == Class::S {
                if let Some((r, _)) = self.t.page(page).and_then(|p| p.speculative) {
                    if !self.out.contains(&Action::Starve(r)) {
                        self.out.push(Action::Starve(r));
                    }
                }
            }
            self.out.push(Action::Unmap { page, class });
            self.gone.insert(page);
            self.got += bytes;
        }
    }

    /// Own caches, least recently used first (an OOM: their bytes are torch's).
    fn shrink(&mut self) {
        let mut caches: Vec<_> = self.t.caches.iter().filter(|c| c.bytes > 0).collect();
        caches.sort_by_key(|c| (c.last_use, c.id));
        for c in caches {
            if self.enough() {
                return;
            }
            let bytes = c.bytes.min(self.need - self.got);
            self.out.push(Action::Shrink { cache: c.id, bytes });
            self.got += bytes;
        }
    }

    /// Asks every untried row with bytes under `ceiling`; true when it asked.
    fn ask(&mut self, ceiling: Class) -> bool {
        let mut rows: Vec<_> = self.t.others.iter().filter(|r| !self.tried.asked.contains(&(r.row, ceiling)) && r.evictable(ceiling) > 0).collect();
        rows.sort_by_key(|r| (std::cmp::Reverse(r.evictable(ceiling)), r.row));
        let mut asked = false;
        let mut left = self.need - self.got;
        for r in rows {
            if left == 0 {
                break;
            }
            let bytes = left.min(r.evictable(ceiling));
            self.out.push(Action::Ask { row: r.row, ceiling, bytes });
            left -= bytes;
            asked = true;
        }
        asked
    }

    /// Runs the demand order to enough, a remote rung, or the end of eviction.
    fn run(&mut self, rungs: Rungs) -> Option<Done> {
        if self.enough() {
            return Some(Done::Enough);
        }
        if rungs == Rungs::Lowered {
            self.evict(Class::W);
            return self.enough().then_some(Done::Enough);
        }
        if !self.tried.trimmed && self.t.slack() > 0 {
            self.out.push(Action::Trim((self.need - self.got).min(self.t.slack())));
            return Some(Done::Remote);
        }
        if !self.tried.orphans && self.t.orphans > 0 {
            self.out.push(Action::ReleaseOrphans((self.need - self.got).min(self.t.orphans)));
            return Some(Done::Remote);
        }
        self.evict(Class::S);
        self.evict(Class::N);
        if self.oom {
            self.shrink();
        }
        if self.enough() {
            return Some(Done::Enough);
        }
        if self.ask(Class::N) {
            return Some(Done::Remote);
        }
        self.evict(Class::D);
        if self.enough() {
            return Some(Done::Enough);
        }
        if self.ask(Class::D) {
            return Some(Done::Remote);
        }
        if rungs == Rungs::Bind {
            // Mid-pass nothing is cut: the binding request lowers and streams its own W.
            if let Some(r) = self.req.and_then(|id| self.t.request(id)).filter(|r| !r.lowered) {
                self.out.push(Action::Lower(r.id));
            }
            self.lowered = true;
            self.evict(Class::W);
            if self.enough() {
                return Some(Done::Enough);
            }
            self.exhausted();
            return None;
        }
        if rungs == Rungs::Pass {
            return self.cut_or_lower();
        }
        Some(Done::Lowered)
    }

    /// Pass start: cut the site with the highest measured peak, else lower.
    fn cut_or_lower(&mut self) -> Option<Done> {
        let Some(r) = self.req.and_then(|id| self.t.request(id)) else {
            self.exhausted();
            return None;
        };
        let fresh = |s: &&crate::table::CutSite| !s.stuck && s.peak > 0 && !self.tried.cut.contains(&s.name);
        let site = r.sites.iter().filter(fresh).max_by_key(|s| (s.peak, std::cmp::Reverse(s.name.clone())));
        if let Some(site) = site {
            self.out.push(Action::Cut { req: r.id, site: site.name.clone() });
            return None;
        }
        if r.lowered {
            return Some(Done::Lowered);
        }
        self.out.push(Action::Lower(r.id));
        self.lowered = true;
        Some(Done::Lowered)
    }

    fn park_or_exhaust(&mut self, parked: bool) {
        match (self.req, parked || self.t.headroom == 0) {
            (Some(r), false) => self.out.push(Action::Park(r)),
            _ => self.exhausted(),
        }
    }

    fn exhausted(&mut self) {
        let need = self.need.saturating_sub(self.got);
        self.out.push(Action::Exhausted { need });
    }
}

/// The own ask thread's answer: S, then N (least recently used), then under a D ceiling the
/// furthest D. With no request open here, the trim hook and the caches go first. Never W or P.
fn answer(t: &Table, ceiling: Class, bytes: u64) -> Vec<Action> {
    let tried = Tried::default();
    let mut d = Demand { got: 0, ..Demand::new(t, bytes, None, None, &tried) };
    if t.requests.is_empty() {
        if t.slack() > 0 {
            d.out.push(Action::Trim(t.slack().min(bytes)));
            d.got += t.slack().min(bytes);
        }
        d.shrink();
    }
    d.evict(Class::S);
    d.evict(Class::N);
    if ceiling >= Class::D {
        d.evict(Class::D);
    }
    d.out
}

/// Prefetch for `req` into the surplus: F, then orphans, then own N, never anything with a
/// declared use. Needs a measured pass, a declared fill and no starvation; each page once per
/// pass, nearest use first. A page for a later step leaves that step's extra room free, so a
/// prefetch never makes a step's room come from D or W: demand evicts what it would without it.
fn speculate(t: &Table, req: u64, tried: &Tried) -> Vec<Action> {
    let Some(r) = t.request(req) else { return vec![] };
    if !r.measured || r.starved || t.requests.iter().any(|o| !o.measured) {
        return vec![];
    }
    let declared = |set: u32| t.sets.get(set as usize).and_then(|s| s.as_ref()).is_some_and(|s| s.declared);
    let mut wanted: Vec<(Key, u64, (u32, u32, u32))> = t
        .pages()
        .filter(|(k, p)| !p.mapped && declared(k.0) && !r.speculated.contains(k))
        .filter_map(|(k, p)| r.distance(k).map(|d| (k, p.bytes, d)))
        .collect();
    wanted.sort_by_key(|(k, _, d)| (*d, *k));
    // The extra room each step ahead needs beyond the current one; `None`: unmeasured.
    let reserve = |ahead: u32| -> Option<u64> {
        (1..=ahead).map(|i| r.rooms.get((r.cursor.step + i) as usize).copied().flatten().map(|room| room.saturating_sub(r.room))).try_fold(0, |m, x| x.map(|x| m.max(x)))
    };
    let mut d = Demand { got: t.free(), ..Demand::new(t, 0, Some(req), None, tried) };
    let n: Vec<(Key, u64)> = d.own(Class::N);
    let mut n = n.into_iter();
    for (page, bytes, distance) in wanted {
        let Some(keep) = reserve(distance.0) else { return d.out };
        while d.got < bytes + keep {
            if !d.tried.orphans && t.orphans > 0 {
                d.out.push(Action::ReleaseOrphans(bytes + keep - d.got));
                return d.out;
            }
            let Some((victim, b)) = n.next() else { return d.out };
            d.out.push(Action::Unmap { page: victim, class: Class::N });
            d.got += b;
        }
        d.out.push(Action::Prefetch(page));
        d.got -= bytes;
    }
    d.out
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
}

/// Whether a request whose W is `w` (`None`: unmeasured) fits beside `rows` and the dispatched
/// requests not yet running: contexts + pins outside a running W + each running request's
/// largest remaining W + this one's ≤ capacity less foreign bytes. Anything unmeasured fits
/// only on an idle GPU.
pub fn fits(capacity: u64, foreign: u64, rows: &[Admitted], dispatched: &[Option<u64>], w: Option<u64>) -> bool {
    let idle = rows.iter().all(|r| r.running == 0) && dispatched.is_empty();
    let unmeasured = w.is_none() || rows.iter().any(|r| r.unmeasured > 0) || dispatched.iter().any(Option::is_none);
    if unmeasured {
        return idle;
    }
    let rows: u64 = rows.iter().map(|r| r.context + r.outside + r.remaining).sum();
    let dispatched: u64 = dispatched.iter().flatten().sum();
    rows + dispatched + w.unwrap_or(0) <= capacity.saturating_sub(foreign)
}
