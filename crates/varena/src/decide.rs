//! The decision procedure: [`decide`] maps a [`Table`] and a [`Trigger`] to [`Action`]s, pure
//! (no device, clock, I/O or thread), in the design's §4 order. Own evictions are certain, so
//! one call plans them through; a remote rung (the trim hook, orphans, asks) is uncertain, so
//! a call ends there and the applier decides again once it is answered, with the rung marked
//! [`Tried`] and what it freed counted. Every placement choice is made here: a pass's room and
//! cuts, relaxing a cut, an OOM's answer, prefetch, a load's placement, an ask's answer, and
//! admission ([`fits`]).

use std::collections::BTreeSet;

use crate::table::{Class, CutSite, Key, Table};

/// Remote rungs already tried in this demand, and what this demand's earlier rounds freed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tried {
    /// The trim hook ran.
    pub trimmed: bool,
    pub orphans: bool,
    /// (row, ceiling) asked.
    pub asked: BTreeSet<(u64, Class)>,
    /// Sites cut at this pass start (each at most once), or proposed at this OOM.
    pub cut: BTreeSet<String>,
    /// Bytes this demand's earlier rounds freed, own unmaps and remote answers alike.
    pub freed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// A bind of an unmapped page, for `req` when a request's pass binds it.
    Bind { page: Key, req: Option<u64>, tried: Tried },
    /// A pass of `req` over `set` starts needing `room` (measured at its cut levels), or,
    /// unmeasured, the reserve `unmeasured` sets aside: the room and its step's unmapped pages
    /// must fit.
    Pass { req: u64, set: u32, room: Option<u64>, unmeasured: u64, tried: Tried },
    /// An op of `req`'s pass over `set` ran short by `need` (at least one page's worth when
    /// unknown) at `site`; a collective site recovers locally and proposes its finer level for
    /// the next pass.
    Oom { req: u64, set: u32, need: Option<u64>, site: Option<String>, collective: bool, tried: Tried },
    /// Another row asks for `bytes` under `ceiling`.
    Ask { ceiling: Class, bytes: u64 },
    /// Fetch ahead for `req` into the surplus.
    Speculate { req: u64, tried: Tried },
    /// Place a loaded set's unmapped pages into free bytes, evicting nothing.
    Load { set: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Run the trim hook: the allocator's slack (part of F) becomes free, before any eviction.
    Trim(u64),
    /// Unmap an own page, of this class now.
    Unmap { page: Key, class: Class },
    /// Ask custody to drop this many orphan bytes, least recently used first.
    ReleaseOrphans(u64),
    Ask { row: u64, ceiling: Class, bytes: u64 },
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
    /// The pass's room, set in the same apply as the evictions that made it.
    Reserve { req: u64, room: u64, measured: bool },
    /// Cut `site` of the pass over `set` one level finer; the cut level last had `failed_at`
    /// bytes of room.
    Cut { req: u64, set: u32, site: String, failed_at: u64 },
    /// Relax `site` one level: its measured room fits the surplus.
    Relax { req: u64, set: u32, site: String },
    /// A collective site's level proposed for the next agreement: finer after an OOM (whose
    /// room was `failed_at`), or one coarser when its measured room fits the surplus.
    Propose { req: u64, set: u32, site: String, level: u32, failed_at: Option<u64> },
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
        Trigger::Pass { req, set, room, unmeasured, tried } => pass(t, *req, *set, *room, *unmeasured, tried),
        Trigger::Oom { req, set, need, site, collective, tried } => oom(t, *req, *set, *need, site.as_deref(), *collective, tried),
        Trigger::Ask { ceiling, bytes } => answer(t, *ceiling, *bytes),
        Trigger::Speculate { req, tried } => speculate(t, *req, tried),
        Trigger::Load { set } => {
            let mut free = t.free();
            let mut out = vec![];
            for (k, p) in t.pages().filter(|(k, p)| k.0 == *set && !p.mapped) {
                if free < p.bytes {
                    break;
                }
                free -= p.bytes;
                out.push(Action::Prefetch(k));
            }
            out
        }
    }
}

/// A pass start: its room and its step's unmapped pages (one page, lowered) must fit. When
/// eviction can't, the uncut site with the highest measured peak is cut (never a collective
/// one, nor one whose flag is unknown: ranks change those together), the room becoming the
/// measured room at the finer level or, unmeasured there, the reserve; then the request lowers.
/// With room to spare, a cut relaxes one level when its measured room fits and beats the room it
/// failed at; a collective one is proposed one level coarser for the ranks' agreement.
fn pass(t: &Table, req: u64, set: u32, room: Option<u64>, unmeasured: u64, tried: &Tried) -> Vec<Action> {
    let Some(r) = t.request(req) else { return vec![] };
    let sites = r.sites.get(&set).map(Vec::as_slice).unwrap_or_default();
    let held = r.room + r.claimed;
    let unmapped = |lowered: bool| {
        let step = r.steps.get(r.cursor.step as usize);
        let sets = step.into_iter().flat_map(|(sets, _)| sets.iter()).filter_map(|s| t.sets.get(*s as usize)?.as_ref());
        let sizes: Vec<u64> = sets.flat_map(|s| s.pages.iter()).filter(|p| !p.mapped).map(|p| p.bytes).collect();
        if lowered { sizes.iter().copied().max().unwrap_or(0) } else { sizes.iter().sum() }
    };
    let (mut room, mut measured) = (room.unwrap_or(unmeasured), room.is_some());
    let mut d = Demand::new(t, room.saturating_sub(held) + unmapped(r.lowered), Some(req), None, tried);
    let mut lowered = r.lowered;
    let mut cut: BTreeSet<&str> = tried.cut.iter().map(String::as_str).collect();
    loop {
        match d.run(Rungs::Pass) {
            Some(Done::Enough) => break,
            Some(Done::Remote) => return d.out,
            _ => {}
        }
        let fresh = |s: &&CutSite| !s.stuck && !s.collective && s.peak > 0 && !cut.contains(s.name.as_str());
        if let Some(site) = sites.iter().filter(fresh).max_by_key(|s| (s.peak, std::cmp::Reverse(s.name.clone()))) {
            cut.insert(&site.name);
            d.out.push(Action::Cut { req, set, site: site.name.clone(), failed_at: d.got + held });
            (room, measured) = match site.finer {
                Some(finer) => (finer, true),
                None => (unmeasured, false),
            };
            d.need = room.saturating_sub(held) + unmapped(lowered);
            continue;
        }
        if lowered {
            d.lowered = true;
            d.need = room.saturating_sub(held) + unmapped(true);
            if d.run(Rungs::Lowered) != Some(Done::Enough) {
                d.exhausted();
                return d.out;
            }
            break;
        }
        d.out.push(Action::Lower(req));
        lowered = true;
        d.need = room.saturating_sub(held) + unmapped(true);
    }
    // Relaxing is speculation: only into what is left once this pass has its room.
    let spare = d.got - d.need;
    let coarser = sites.iter().filter(|s| s.level > 0).filter_map(|s| s.coarser.map(|c| (s, c)));
    let fitting = |(_, (c, failed)): &(&CutSite, (u64, Option<u64>))| c.saturating_sub(room) <= spare && failed.is_none_or(|f| room + spare > f);
    let relax = coarser.filter(fitting).min_by_key(|(s, (c, _))| (*c, s.name.clone()));
    if let (true, Some((site, (coarse, _)))) = (measured, relax) {
        d.out.push(match site.collective {
            true => Action::Propose { req, set, site: site.name.clone(), level: site.level - 1, failed_at: None },
            false => Action::Relax { req, set, site: site.name.clone() },
        });
        room = room.max(coarse);
    }
    d.out.push(Action::Reserve { req, room, measured });
    d.out
}

/// An out-of-memory: free bytes count once per episode (the cap was stale); after that only
/// what this demand freed. Then the demand order; at its end, a cut of the op's site (never a
/// collective's mid-pass: its finer level is proposed for the next pass), lowering, parking,
/// exhaustion. Each answer is progress, so a sequence ends.
fn oom(t: &Table, req: u64, set: u32, need: Option<u64>, site: Option<&str>, collective: bool, tried: &Tried) -> Vec<Action> {
    let Some(r) = t.request(req) else { return vec![Action::Exhausted { need: need.unwrap_or(0) }] };
    let sites = r.sites.get(&set).map(Vec::as_slice).unwrap_or_default();
    // Unknown: any page's worth is progress.
    let mut d = Demand::new(t, need.unwrap_or(1), Some(req), None, tried);
    d.oom = true;
    if r.episode.claimed_free {
        d.got = t.free().min(tried.freed);
    }
    let mut out = vec![];
    if let (Some(site), true) = (site, collective) {
        if !tried.cut.contains(site) {
            let level = sites.iter().find(|s| s.name == site).map_or(0, |s| s.level) + 1;
            out.push(Action::Propose { req, set, site: site.to_string(), level, failed_at: Some(r.room + r.claimed + t.free()) });
        }
    }
    match d.run(Rungs::Oom) {
        Some(Done::Remote) => {
            out.append(&mut d.out);
            return out;
        }
        Some(Done::Enough) => {}
        Some(Done::Lowered) | None => {
            let cuttable = site.filter(|s| !collective && !sites.iter().any(|x| x.name == *s && x.stuck));
            if let Some(site) = cuttable {
                d.out.push(Action::Cut { req, set, site: site.to_string(), failed_at: r.room + r.claimed + t.free() });
            } else {
                if !r.lowered {
                    d.out.push(Action::Lower(req));
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
        d.out.push(Action::Claim { req, bytes });
    }
    out.append(&mut d.out);
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Done {
    Enough,
    /// Ended on a remote rung: decide again once answered.
    Remote,
    /// Nothing left but cutting or lowering.
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

/// The own ask thread's answer, from pages only (the thread runs no caller code): S, then N
/// (least recently used), then under a D ceiling the furthest D; never W or P. An idle process
/// also wants its trim and caches run, which its trim thread does after the answer.
fn answer(t: &Table, ceiling: Class, bytes: u64) -> Vec<Action> {
    let tried = Tried::default();
    let mut d = Demand { got: 0, ..Demand::new(t, bytes, None, None, &tried) };
    d.evict(Class::S);
    d.evict(Class::N);
    if ceiling >= Class::D {
        d.evict(Class::D);
    }
    if t.requests.is_empty() && !d.enough() {
        if t.slack() > 0 {
            d.out.push(Action::Trim(t.slack()));
        }
        d.shrink();
    }
    d.out
}

/// Prefetch for `req` into the surplus: F, then orphans, then own N, never anything with a
/// declared use; in shared mode only own N (the footprint never grows). Needs a measured pass,
/// a declared fill and no starvation; each page once per pass, nearest use first. A page for a
/// later step leaves that step's extra room free, so a prefetch never makes a step's room come
/// from D or W: demand evicts what it would without it.
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
    let mut d = Demand { got: if t.shared { 0 } else { t.free() }, ..Demand::new(t, 0, Some(req), None, tried) };
    let n: Vec<(Key, u64)> = d.own(Class::N);
    let mut n = n.into_iter();
    for (page, bytes, distance) in wanted {
        let Some(keep) = reserve(distance.0) else { return d.out };
        while d.got < bytes + keep {
            if !t.shared && !d.tried.orphans && t.orphans > 0 {
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
/// largest remaining W + this one's ≤ capacity less foreign bytes. An idle GPU always fits (a
/// request larger than the card streams); anything unmeasured fits only there.
pub fn fits(capacity: u64, foreign: u64, rows: &[Admitted], dispatched: &[Option<u64>], w: Option<u64>) -> bool {
    if rows.iter().all(|r| r.running == 0) && dispatched.is_empty() {
        return true;
    }
    if w.is_none() || rows.iter().any(|r| r.unmeasured > 0) || dispatched.iter().any(Option::is_none) {
        return false;
    }
    let rows: u64 = rows.iter().map(|r| r.context + r.outside + r.remaining).sum();
    let dispatched: u64 = dispatched.iter().flatten().sum();
    rows + dispatched + w.unwrap_or(0) <= capacity.saturating_sub(foreign)
}
