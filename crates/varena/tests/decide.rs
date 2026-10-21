//! The decision procedure over tables: real plans at real capacities, each demand rung, Belady,
//! the speculation invariant and termination. A small simulator applies the actions the way
//! the applier does (maps fill at once; asks free what the other row offers).

use varena::decide::{decide, fits, Action, Admitted, OomSite, Trigger, Tried};
use varena::table::{Class, Key, Page, Remote, Req, Set, Table};

const MB: u64 = 1 << 20;
const GB: u64 = 1 << 30;

fn set(name: &str, pages: usize, bytes: u64) -> Option<Set> {
    Some(Set { name: name.into(), pages: vec![Page { bytes, ..Page::default() }; pages], declared: true })
}

/// A table with these sets and one request over `steps` (set indices, passes).
fn table(capacity: u64, sets: Vec<Option<Set>>, steps: Vec<(Vec<u32>, u32)>) -> Table {
    let req = Req { id: 1, name: "r".into(), rooms: steps.iter().map(|_| Some(0)).collect(), steps, measured: true, ..Req::default() };
    Table { capacity, sets, requests: vec![req], ..Table::default() }
}

/// Each step's measured room, as the applier reads it from the ledger.
fn with_rooms(mut t: Table, rooms: &[u64]) -> Table {
    t.requests[0].rooms = rooms.iter().map(|r| Some(*r)).collect();
    t
}

#[derive(Default)]
struct Run {
    evicted: Vec<(Key, Class)>,
    exhausted: bool,
    lowered: bool,
    speculative: Vec<Key>,
    maps: u64,
}

/// Applies own actions to the table; asks are answered by removing the other row's bytes.
fn apply(t: &mut Table, actions: &[Action], run: &mut Run, tried: &mut Tried, clock: &mut u64) -> bool {
    let mut remote = false;
    for a in actions {
        match a {
            Action::Unmap { page, class } => {
                run.evicted.push((*page, *class));
                tried.freed += t.page(*page).unwrap().bytes;
                let p = t.page_mut(*page).unwrap();
                p.mapped = false;
                p.filled = false;
                p.speculative = None;
            }
            Action::Map(k) | Action::Prefetch(k) => {
                *clock += 1;
                let step = t.requests[0].distance(*k).map(|d| t.requests[0].cursor.step + d.0);
                let p = t.page_mut(*k).unwrap();
                p.mapped = true;
                p.filled = true;
                p.last_use = *clock;
                run.maps += p.bytes;
                if matches!(a, Action::Prefetch(_)) {
                    p.speculative = step.map(|s| (1, s));
                    run.speculative.push(*k);
                    t.requests[0].speculated.insert(*k);
                }
            }
            Action::Starve(_) => t.requests[0].starved = true,
            Action::Lower(_) => {
                t.requests[0].lowered = true;
                run.lowered = true;
            }
            Action::Exhausted { .. } => run.exhausted = true,
            Action::ReleaseOrphans(b) => {
                tried.orphans = true;
                t.orphans = t.orphans.saturating_sub(*b);
                remote = true;
            }
            Action::Ask { row, ceiling, bytes } => {
                tried.asked.insert((*row, *ceiling));
                let r = t.others.iter_mut().find(|r| r.row == *row).unwrap();
                let mut left = *bytes;
                for c in [Class::S, Class::N, Class::D].into_iter().filter(|c| *c <= *ceiling) {
                    let take = left.min(r.classes[c.index()]);
                    r.classes[c.index()] -= take;
                    left -= take;
                    tried.freed += take;
                }
                remote = true;
            }
            Action::Claim { req, bytes } => {
                let r = t.request_mut(*req).unwrap();
                r.claimed += bytes;
                r.episode.claimed_free = true;
            }
            Action::Trim(_) => {
                tried.trimmed = true;
                remote = true;
            }
            Action::Reserve { req, room, measured } => {
                let r = t.request_mut(*req).unwrap();
                r.room = *room;
                r.measured = *measured;
            }
            Action::Shrink { .. } | Action::Cut { .. } | Action::Relax { .. } | Action::Propose { .. } | Action::Park(_) => {}
        }
    }
    remote
}

/// Decides and applies until the trigger ends (remote rungs decided again).
fn demand(t: &mut Table, mut trigger: Trigger, run: &mut Run, clock: &mut u64) -> Vec<Action> {
    loop {
        let actions = decide(t, &trigger);
        let mut tried = match &trigger {
            Trigger::Bind { tried, .. } | Trigger::Pass { tried, .. } | Trigger::Oom { tried, .. } | Trigger::Speculate { tried, .. } => tried.clone(),
            Trigger::Ask { .. } | Trigger::Load { .. } => Tried::default(),
        };
        if !apply(t, &actions, run, &mut tried, clock) {
            return actions;
        }
        match &mut trigger {
            Trigger::Bind { tried: t, .. } | Trigger::Pass { tried: t, .. } | Trigger::Oom { tried: t, .. } | Trigger::Speculate { tried: t, .. } => *t = tried,
            Trigger::Ask { .. } | Trigger::Load { .. } => return actions,
        }
    }
}

/// Runs a request's plan: per pass a pass trigger, then a bind (and, `speculate`, a prefetch
/// decision) per block in order.
fn play(t: &mut Table, rooms: &[u64], speculate: bool) -> Run {
    let mut run = Run::default();
    let mut clock = 0;
    let steps = t.requests[0].steps.clone();
    for (i, (sets, passes)) in steps.iter().enumerate() {
        {
            let r = &mut t.requests[0];
            r.cursor = Default::default();
            r.cursor.step = i as u32;
            r.starved = false;
        }
        for _ in 0..*passes {
            for &s in sets {
                let r = &mut t.requests[0];
                *r.cursor.passes.entry(s).or_insert(0) += 1;
                r.cursor.next.insert(s, 0);
                r.speculated.clear();
                let room = rooms[i];
                let a = demand(t, Trigger::Pass { req: 1, set: s, room: Some(room), unmeasured: 0, tried: Tried::default() }, &mut run, &mut clock);
                if matches!(a.last(), Some(Action::Exhausted { .. })) {
                    return run;
                }
                let blocks = t.sets[s as usize].as_ref().unwrap().pages.len() as u32;
                for b in 0..blocks {
                    let key = (s, b);
                    if !t.page(key).unwrap().mapped {
                        let a = demand(t, Trigger::Bind { page: key, req: Some(1), tried: Tried::default() }, &mut run, &mut clock);
                        if matches!(a.last(), Some(Action::Exhausted { .. })) {
                            return run;
                        }
                    }
                    clock += 1;
                    let p = t.page_mut(key).unwrap();
                    p.last_use = clock;
                    p.speculative = None;
                    t.requests[0].cursor.next.insert(s, b + 1);
                    if speculate {
                        demand(t, Trigger::Speculate { req: 1, tried: Tried::default() }, &mut run, &mut clock);
                    }
                }
            }
        }
    }
    run
}

/// H3: [te, {avae, vae}, dit×8, avae, vae] (8-bit DiT in 40 blocks).
fn h3(capacity: u64) -> (Table, Vec<u64>) {
    let sets = vec![set("te", 24, 256 * MB), set("avae", 2, 256 * MB), set("vae", 2, 256 * MB), set("dit", 40, 352 * MB)];
    let steps = vec![(vec![0], 1), (vec![1, 2], 1), (vec![3], 8), (vec![1], 1), (vec![2], 1)];
    let rooms = vec![GB, GB, 3 * GB, 2 * GB, 4 * GB];
    (with_rooms(table(capacity, sets, steps), &rooms), rooms)
}

/// SDXL: [te, unet×30, vae].
fn sdxl(capacity: u64) -> (Table, Vec<u64>) {
    let sets = vec![set("te", 8, 200 * MB), set("unet", 26, 100 * MB), set("vae", 2, 80 * MB)];
    let steps = vec![(vec![0], 1), (vec![1], 30), (vec![2], 1)];
    let rooms = vec![GB / 2, 2 * GB, 3 * GB];
    (with_rooms(table(capacity, sets, steps), &rooms), rooms)
}

#[test]
fn h3_and_sdxl_complete_at_24_12_and_8_gib() {
    for capacity in [24 * GB, 12 * GB, 8 * GB] {
        for (name, (mut t, rooms)) in [("h3", h3(capacity)), ("sdxl", sdxl(capacity))] {
            let run = play(&mut t, &rooms, true);
            assert!(!run.exhausted, "{name} at {} GiB exhausted", capacity / GB);
            let w_or_p = run.evicted.iter().filter(|(_, c)| matches!(c, Class::W | Class::P)).count();
            assert!(run.lowered || w_or_p == 0, "{name} at {} GiB evicted W without lowering", capacity / GB);
            assert!(run.evicted.iter().all(|(_, c)| *c != Class::P), "{name}: a pinned page went");
        }
    }
    // Whole at 24 GiB: the DiT (14 GiB) stays resident across its 8 passes.
    let (mut t, rooms) = h3(24 * GB);
    let run = play(&mut t, &rooms, false);
    assert!(!run.lowered);
    let dit = 40 * 352 * MB;
    assert!(run.maps < dit * 2, "the DiT was read {} times", run.maps / dit);
    // At 8 GiB the DiT streams: lowered.
    let (mut t, rooms) = h3(8 * GB);
    assert!(play(&mut t, &rooms, false).lowered);
}

/// A table at capacity 10 pages with one request at step 1 of [a, b, a+c], one page of each
/// class an own demand can take.
fn ladder() -> Table {
    let sets = vec![set("a", 4, MB), set("b", 2, MB), set("c", 2, MB), set("n", 2, MB)];
    let mut t = table(10 * MB, sets, vec![(vec![0], 1), (vec![1], 1), (vec![0, 2], 1)]);
    t.requests[0].cursor.step = 1;
    t.requests[0].cursor.passes.insert(1, 1);
    t
}

fn map(t: &mut Table, k: Key) {
    let p = t.page_mut(k).unwrap();
    p.mapped = true;
    p.filled = true;
}

#[test]
fn every_rung_of_the_demand_order() {
    let bind = |t: &Table| decide(t, &Trigger::Bind { page: (1, 1), req: Some(1), tried: Tried::default() });
    // F: enough free, map at once.
    let t = ladder();
    assert_eq!(bind(&t), vec![Action::Map((1, 1))]);

    // Fill the card: W = b0, D = a0..a3 and c0, c1, N = n0, n1, S = c? (speculated for step 2).
    let mut full = ladder();
    for k in [(1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (3, 0), (3, 1)] {
        map(&mut full, k);
    }
    full.page_mut((2, 1)).unwrap().speculative = Some((1, 2));
    full.capacity = 9 * MB;
    assert_eq!(full.class((1, 0)), Some(Class::W));
    assert_eq!(full.class((0, 0)), Some(Class::D));
    assert_eq!(full.class((2, 1)), Some(Class::S));
    assert_eq!(full.class((3, 0)), Some(Class::N));

    // O: orphans go first.
    let mut t = full.clone();
    t.orphans = MB;
    t.capacity += MB;
    assert_eq!(bind(&t), vec![Action::ReleaseOrphans(MB)]);
    // Own S: the step starves.
    let a = bind(&full);
    assert_eq!(a[..2], [Action::Starve(1), Action::Unmap { page: (2, 1), class: Class::S }]);
    // Own N, least recently used first.
    let mut t = full.clone();
    t.page_mut((2, 1)).unwrap().speculative = None;
    t.page_mut((3, 0)).unwrap().last_use = 5;
    t.page_mut((3, 1)).unwrap().last_use = 2;
    assert_eq!(bind(&t)[0], Action::Unmap { page: (3, 1), class: Class::N });
    // Others' O/S/N: asked under N.
    let mut t = full.clone();
    for k in [(2, 1), (3, 0), (3, 1)] {
        t.page_mut(k).unwrap().mapped = false;
    }
    t.capacity = 6 * MB + 2 * MB;
    t.others = vec![Remote { row: 7, classes: [0, 0, 0, 2 * MB, 0, 0, 0], ..Remote::default() }];
    assert_eq!(bind(&t), vec![Action::Ask { row: 7, ceiling: Class::N, bytes: MB }]);
    // Own D, furthest first: a3 (next used at step 2, block 3) before c0.
    let mut tried = Tried::default();
    tried.asked.insert((7, Class::N));
    let a = decide(&t, &Trigger::Bind { page: (1, 1), req: Some(1), tried: tried.clone() });
    assert_eq!(a[0], Action::Unmap { page: (0, 3), class: Class::D });
    // Others' D: asked under D once own D is gone.
    let mut t2 = t.clone();
    for k in [(0, 0), (0, 1), (0, 2), (0, 3), (2, 0)] {
        t2.page_mut(k).unwrap().mapped = false;
    }
    t2.capacity = MB + MB;
    t2.others = vec![Remote { row: 7, classes: [0, 0, 0, 0, 2 * MB, 0, 0], ..Remote::default() }];
    assert_eq!(decide(&t2, &Trigger::Bind { page: (1, 1), req: Some(1), tried: Tried::default() }), vec![Action::Ask { row: 7, ceiling: Class::D, bytes: MB }]);
    // Cut at a pass start: the site with the highest measured peak.
    let mut t3 = t2.clone();
    t3.others.clear();
    t3.requests[0].sites.insert(
        1,
        vec![
            varena::table::CutSite { name: "ffn".into(), peak: 3 * MB, finest: 4, ..Default::default() },
            varena::table::CutSite { name: "attn".into(), peak: 2 * MB, finest: 4, ..Default::default() },
        ],
    );
    let a = decide(&t3, &Trigger::Pass { req: 1, set: 1, room: Some(4 * MB), unmeasured: 0, tried: Tried::default() });
    assert!(a.iter().any(|a| matches!(a, Action::Cut { site, .. } if site == "ffn")), "{a:?}");
    // Lower: no cut left, the request lowers and streams its own W (but never the page bound).
    let mut tried = Tried::default();
    tried.cut.extend(["ffn".to_string(), "attn".to_string()]);
    let a = decide(&t3, &Trigger::Pass { req: 1, set: 1, room: Some(MB), unmeasured: 0, tried });
    assert!(a.contains(&Action::Lower(1)), "{a:?}");
    // Park, then Exhausted, at an OOM with nothing left.
    let mut t4 = t3.clone();
    t4.page_mut((1, 0)).unwrap().pins = vec![1];
    t4.requests[0].lowered = true;
    t4.requests[0].episode.claimed_free = true;
    t4.headroom = GB;
    let oom = |t: &Table| decide(t, &Trigger::Oom { req: 1, set: Some(1), need: Some(MB), site: None, tried: Tried::default() });
    assert_eq!(oom(&t4).last(), Some(&Action::Park(1)));
    t4.requests[0].episode.parked = true;
    assert!(matches!(oom(&t4).last(), Some(Action::Exhausted { .. })));
}

#[test]
fn belady_evicts_the_furthest_use() {
    // Plan [a, b, c, d]; at step 0, b/c/d are D at distances 1/2/3: d goes first, then c.
    let sets = vec![set("a", 1, MB), set("b", 1, MB), set("c", 1, MB), set("d", 1, MB), set("x", 1, MB)];
    let mut t = table(4 * MB, sets, vec![(vec![0], 1), (vec![1], 1), (vec![2], 1), (vec![3], 1), (vec![4], 1)]);
    for k in [(1, 0), (2, 0), (3, 0), (4, 0)] {
        map(&mut t, k);
    }
    let a = decide(&t, &Trigger::Bind { page: (0, 0), req: Some(1), tried: Tried::default() });
    assert_eq!(a, vec![Action::Unmap { page: (4, 0), class: Class::D }, Action::Map((0, 0))]);
    // Within a repeating step the block just used is the furthest (most recently used).
    let mut t = table(3 * MB, vec![set("dit", 4, MB)], vec![(vec![0], 8)]);
    t.requests[0].lowered = true;
    t.requests[0].cursor.passes.insert(0, 1);
    t.requests[0].cursor.next.insert(0, 3);
    for k in [(0, 0), (0, 1), (0, 2)] {
        map(&mut t, k);
    }
    let a = decide(&t, &Trigger::Bind { page: (0, 3), req: Some(1), tried: Tried::default() });
    assert_eq!(a[0], Action::Unmap { page: (0, 2), class: Class::W });
}

#[test]
fn speculation_changes_no_demand_eviction() {
    for (name, capacity) in [("24", 24 * GB), ("12", 12 * GB), ("8", 8 * GB)] {
        let (mut off, rooms) = h3(capacity);
        let (mut on, _) = h3(capacity);
        let plain = play(&mut off, &rooms, false);
        let ahead = play(&mut on, &rooms, true);
        let valuable = |r: &Run| r.evicted.iter().filter(|(_, c)| matches!(c, Class::D | Class::W | Class::P)).copied().collect::<Vec<_>>();
        assert_eq!(valuable(&plain), valuable(&ahead), "h3 at {name} GiB");
        assert!(!ahead.speculative.is_empty() || capacity == 8 * GB, "h3 at {name} GiB fetched nothing ahead");
    }
    // Speculation itself never evicts anything with a declared use.
    let (mut t, rooms) = sdxl(12 * GB);
    play(&mut t, &rooms, false);
    let a = decide(&t, &Trigger::Speculate { req: 1, tried: Tried::default() });
    assert!(a.iter().all(|a| !matches!(a, Action::Unmap { class: Class::D | Class::W | Class::P | Class::S, .. })), "{a:?}");
}

#[test]
fn any_oom_sequence_ends_within_pages_plus_cuts_plus_two() {
    let pages = 6u32;
    let cut_levels = 3u32;
    let mut t = table(pages as u64 * MB, vec![set("dit", pages as usize, MB)], vec![(vec![0], 1)]);
    t.requests[0].cursor.passes.insert(0, 1);
    for b in 0..pages {
        map(&mut t, (0, b));
    }
    t.page_mut((0, 0)).unwrap().pins = vec![9];
    let mut answers = 0;
    let mut cuts = 0;
    let mut clock = 0;
    let mut run = Run::default();
    loop {
        answers += 1;
        assert!(answers <= pages + cut_levels + 2, "{answers} answers");
        let site = (cuts < cut_levels).then(|| OomSite { name: "ffn".into(), collective: false, finest: cut_levels });
        let a = demand(&mut t, Trigger::Oom { req: 1, set: Some(0), need: Some(64 * MB), site, tried: Tried::default() }, &mut run, &mut clock);
        match a.iter().rev().find(|a| matches!(a, Action::Cut { .. } | Action::Park(_) | Action::Exhausted { .. })) {
            Some(Action::Cut { .. }) => cuts += 1,
            Some(Action::Exhausted { .. }) => break,
            Some(_) => t.requests[0].episode.parked = true,
            None => assert!(a.iter().any(|a| matches!(a, Action::Claim { .. })), "a retry without progress: {a:?}"),
        }
    }
    assert_eq!(cuts, cut_levels);
    assert!(run.evicted.iter().all(|(k, _)| *k != (0, 0)), "the pinned page went");
}

#[test]
fn admission_counts_each_byte_once() {
    let row = |remaining, running, unmeasured| Admitted { context: GB, outside: 0, remaining, running, unmeasured };
    assert!(fits(24 * GB, 0, &[row(10 * GB, 1, 0)], &[], Some(12 * GB)));
    assert!(!fits(24 * GB, 0, &[row(10 * GB, 1, 0)], &[], Some(13 * GB + 1)));
    assert!(!fits(24 * GB, 2 * GB, &[row(10 * GB, 1, 0)], &[], Some(12 * GB)), "foreign bytes are taken off");
    assert!(!fits(24 * GB, 0, &[row(10 * GB, 1, 0)], &[Some(4 * GB)], Some(12 * GB)), "a dispatched request counts");
    // Unmeasured: only on an idle GPU.
    assert!(!fits(24 * GB, 0, &[row(GB, 1, 0)], &[], None));
    assert!(fits(24 * GB, 0, &[row(0, 0, 0)], &[], None));
    assert!(!fits(24 * GB, 0, &[row(GB, 1, 1)], &[], Some(GB)));
}

// ---- regressions from review 1

/// Review 1, finding 5 (the reviewer's probe): an OOM whose own N unmap and another row's
/// answer together cover the need re-decides with both counted, and evicts no D page.
#[test]
fn an_oom_counts_its_own_unmaps_after_an_ask() {
    let g = 1 << 30;
    let page = || Page { bytes: g, mapped: true, filled: true, ..Page::default() };
    let set = |name: &str, n: usize| Some(Set { name: name.into(), pages: (0..n).map(|_| page()).collect(), declared: true });
    let mut req = Req { id: 1, name: "r".into(), steps: vec![(vec![0], 1), (vec![1], 1)], rooms: vec![Some(0), Some(0)], measured: true, ..Req::default() };
    req.episode.claimed_free = true;
    let mut t = Table { capacity: 6 * g, sets: vec![set("w", 2), set("d", 2), set("n", 1)], requests: vec![req], others: vec![Remote { row: 99, running: true, classes: [0, 0, 0, g, 0, 0, 0], ..Remote::default() }], ..Table::default() };
    let (mut run, mut clock) = (Run::default(), 0);
    let a = demand(&mut t, Trigger::Oom { req: 1, set: Some(0), need: Some(2 * g), site: None, tried: Tried::default() }, &mut run, &mut clock);
    assert!(run.evicted.iter().all(|(_, c)| *c != Class::D), "evicted a D page: {:?}", run.evicted);
    assert!(a.iter().any(|a| matches!(a, Action::Claim { bytes, .. } if *bytes == 2 * g)), "{a:?}");
}

/// Review 1, finding 6: an idle GPU admits anything (it streams), and a lowered request's
/// remaining W counts its streaming slots, not all its pages.
#[test]
fn an_idle_gpu_admits_a_model_larger_than_the_card() {
    let idle = Admitted { context: GB, ..Admitted::default() };
    assert!(fits(8 * GB, 0, &[idle], &[], Some(20 * GB)));
    let (mut t, rooms) = h3(8 * GB);
    play(&mut t, &rooms, false);
    let r = &t.requests[0];
    assert!(r.lowered);
    let remaining = t.remaining(r).unwrap();
    assert!(remaining < 8 * GB, "a streamed step's W is its room and slots: {remaining}");
}

fn cut_table(level: u32, coarser: Option<(u64, Option<u64>)>, collective: bool) -> Table {
    let mut t = table(16 * MB, vec![set("dit", 4, MB)], vec![(vec![0], 1)]);
    t.requests[0].cursor.passes.insert(0, 1);
    t.requests[0].sites.insert(0, vec![varena::table::CutSite { name: "ffn".into(), level, finest: 4, peak: 4 * MB, collective, finer: Some(2 * MB), coarser, ..Default::default() }]);
    t
}

/// Review 1, finding 7: a cut relaxes only into what is left once the pass has its room and
/// its step's pages, and never back to a room it failed at.
#[test]
fn relaxing_takes_only_the_surplus_and_remembers_failures() {
    let pass = |t: &Table, room: u64| decide(t, &Trigger::Pass { req: 1, set: 0, room: Some(room), unmeasured: 0, tried: Tried::default() });
    // 16 MB: 4 MB of pages unmapped, a 4 MB room; coarser needs 6 MB: 8 MB spare, so it relaxes.
    let t = cut_table(1, Some((6 * MB, None)), false);
    assert!(pass(&t, 4 * MB).contains(&Action::Relax { req: 1, set: 0, site: "ffn".into() }));
    // The same with the coarser room larger than the spare: it stays cut.
    let t = cut_table(1, Some((13 * MB, None)), false);
    assert!(!pass(&t, 4 * MB).iter().any(|a| matches!(a, Action::Relax { .. })));
    // It failed at 12 MB of room before; 12 MB is all there is: it stays cut.
    let t = cut_table(1, Some((6 * MB, Some(12 * MB))), false);
    assert!(!pass(&t, 4 * MB).iter().any(|a| matches!(a, Action::Relax { .. })));
    // Relaxing never evicts: with the pages taking the surplus it waits.
    let mut t = cut_table(1, Some((6 * MB, None)), false);
    t.capacity = 9 * MB;
    assert!(!pass(&t, 4 * MB).iter().any(|a| matches!(a, Action::Relax { .. } | Action::Unmap { .. })));
}

/// Review 1, finding 4: a pass start never cuts a collective site on its own, and an OOM at one
/// proposes its finer level for the next pass instead of cutting mid-pass.
#[test]
fn collective_sites_change_only_by_agreement() {
    let mut t = cut_table(0, None, true);
    t.capacity = 5 * MB;
    let a = decide(&t, &Trigger::Pass { req: 1, set: 0, room: Some(4 * MB), unmeasured: 0, tried: Tried::default() });
    assert!(!a.iter().any(|a| matches!(a, Action::Cut { .. })), "{a:?}");
    assert!(a.contains(&Action::Lower(1)));
    let a = decide(&t, &Trigger::Oom { req: 1, set: Some(0), need: Some(MB), site: Some(OomSite { name: "ffn".into(), collective: true, finest: 4 }), tried: Tried::default() });
    assert!(a.iter().any(|a| matches!(a, Action::Propose { req: 1, set: 0, site, level: 1, failed_at: Some(_) } if site == "ffn")), "{a:?}");
    assert!(!a.iter().any(|a| matches!(a, Action::Cut { .. })), "{a:?}");
}

/// Review 1, finding 8: a pass's room is set by the same decision, in the same apply, as the
/// evictions that made it.
#[test]
fn a_pass_reserves_its_room_with_its_evictions() {
    let mut t = table(4 * MB, vec![set("a", 2, MB), set("n", 2, MB)], vec![(vec![0], 1)]);
    map(&mut t, (1, 0));
    map(&mut t, (1, 1));
    let a = decide(&t, &Trigger::Pass { req: 1, set: 0, room: Some(2 * MB), unmeasured: 0, tried: Tried::default() });
    assert_eq!(a.last(), Some(&Action::Reserve { req: 1, room: 2 * MB, measured: true }));
    assert_eq!(a.iter().filter(|a| matches!(a, Action::Unmap { class: Class::N, .. })).count(), 2, "{a:?}");
}

/// The tier's order, pure: orphaned layouts first (least recently used), then unused, then the
/// furthest declared; a region in flight is never given back.
#[test]
fn the_tier_never_releases_a_region_in_flight() {
    use varena::tier::{host_decide, HostAction, HostLayout, HostTrigger};
    let layout = |digest: &str, users: Vec<Option<u32>>, used: u64, regions: Vec<(u64, bool, bool)>| HostLayout { digest: digest.into(), regions, users, used };
    let layouts = vec![
        layout("near", vec![Some(1)], 1, vec![(MB, true, false)]),
        layout("orphan", vec![], 9, vec![(MB, true, true), (MB, true, false)]),
        layout("unused", vec![None], 5, vec![(MB, true, false)]),
        layout("far", vec![Some(4)], 2, vec![(MB, true, false)]),
    ];
    let first = host_decide(&layouts, HostTrigger::Pressure);
    assert_eq!(first, vec![HostAction::Release { digest: "orphan".into(), region: 1 }], "the region in flight stays, so the layout stays");
    let room = host_decide(&layouts, HostTrigger::Room { bytes: 4 * MB, headroom: 0 });
    let order: Vec<&str> = room.iter().map(|a| match a { HostAction::Release { digest, .. } | HostAction::Forget(digest) => digest.as_str() }).collect();
    assert_eq!(order, vec!["orphan", "unused", "far", "near"]);
}
