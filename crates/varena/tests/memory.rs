//! One process's Memory over host pages: rooms bounded by measured shapes, the cut memo, and
//! the OOM answers for cut and collective sites.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use tensorfs::format::ObjectRef;
use tensorfs::io::source::Source;
use tensorfs::io::{ItemSource, ObjectRange};
use varena::layout::{Item, Layout, Part, Region, REGION_ALIGN};
use varena::{Fill, Fingerprint, GpuId, HostBackend, Level, Measured, Memory, Next, Plan, SetSpec, Site, Step, Torch};

const MB: u64 = 1 << 20;

fn open(case: &str, memory: u64) -> (varena::Ledger, Memory, GpuId) {
    let g = GpuId { uuid: format!("m{}-{case}", std::process::id()), model: "host".into(), memory };
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(memory))).unwrap();
    (ledger, mem, g)
}

fn set(dir: &Path, name: &str, regions: u64, uuid: &str) -> SetSpec {
    let path = dir.join(format!("{name}.bin"));
    std::fs::write(&path, vec![3u8; (regions * REGION_ALIGN) as usize]).unwrap();
    let obj = ObjectRef { sha256: format!("obj-{name}"), length: regions * REGION_ALIGN };
    let layout = Layout {
        parts: (0..regions).map(|r| Part { what: format!("t{r}"), region: r as u32, offset: r * REGION_ALIGN, nbytes: REGION_ALIGN }).collect(),
        regions: (0..regions)
            .map(|r| Region {
                offset: r * REGION_ALIGN,
                nbytes: REGION_ALIGN,
                span: REGION_ALIGN,
                items: vec![Item { offset: r * REGION_ALIGN, len: REGION_ALIGN, source: ItemSource::Object(ObjectRange { obj: obj.clone(), off: r * REGION_ALIGN, len: REGION_ALIGN }), part_end: true }],
            })
            .collect(),
        nbytes: regions * REGION_ALIGN,
        digest: format!("layout-{name}-{uuid}"),
        names: (0..regions).map(|r| vec![format!("t{r}")]).collect(),
    };
    let source = Source::files(vec![(obj, File::open(&path).unwrap())], false, 0);
    let fingerprint = Fingerprint::new(&layout, &[], Fill::Stored, "release-1", "0/1", uuid);
    SetSpec { name: format!("pkg@1/{name}"), layout: Arc::new(layout), source: Arc::new(source), fingerprint }
}

fn plan(name: &str) -> Plan {
    Plan { steps: vec![Step { sets: vec![format!("pkg@1/{name}")], passes: 1 }] }
}

/// One pass at `shape`, its peak as W less the set's pages.
fn room(mem: &Memory, id: varena::SetId, name: &str, shape: &str, pages: u64, measured: Option<u64>) -> u64 {
    let r = mem.request(&format!("{name}-{shape}"), plan(name)).unwrap();
    let pass = r.pass(id, "fwd", shape, Torch::default()).unwrap();
    let w = r.stats().peak;
    if let Some(peak) = measured {
        pass.end(Measured { allocated_peak: peak, slack: 0, sites: vec![varena::SitePeak { name: "ffn".into(), peak: peak / 2, collective: false, finest: 4 }] }, Torch::default());
    }
    drop(r);
    w - pages
}

#[test]
fn a_new_shape_is_bounded_by_a_measured_one() {
    let dir = tempfile::tempdir().unwrap();
    let (_l, mem, g) = open("shape", 64 * MB);
    let id = mem.load(set(dir.path(), "s", 1, &g.uuid), Level::Ram).unwrap();
    let pages = REGION_ALIGN;
    // Unmeasured: no room but the largest context growth (none yet).
    assert_eq!(room(&mem, id, "s", "1x8x8", pages, Some(8 * MB)), 0);
    assert_eq!(room(&mem, id, "s", "1x8x8", pages, None), 8 * MB, "measured");
    assert_eq!(room(&mem, id, "s", "1x4x8", pages, None), 8 * MB, "bounded by the nearest larger shape");
    assert_eq!(room(&mem, id, "s", "1x16x8", pages, None), 16 * MB, "scaled from the nearest smaller by its largest ratio");
}

#[test]
fn an_oom_at_a_cut_site_cuts_and_a_collective_one_waits_for_the_next_pass() {
    let dir = tempfile::tempdir().unwrap();
    let (_l, mem, g) = open("cut", 4 * MB);
    let id = mem.load(set(dir.path(), "c", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", plan("c")).unwrap();
    let pass = r.pass(id, "fwd", "1x8", Torch::default()).unwrap();
    let pins: Vec<_> = (0..2).map(|b| mem.bind(id, b, 0).unwrap()).collect();
    let site = Some(Site { name: "ffn", collective: false, finest: 4 });
    // The free bytes went to the pages; every page is pinned: the op cuts, level by level.
    assert!(!pass.measured() && !r.measured(), "no room measured at this shape yet");
    assert!(matches!(r.oom(Some(id), Some(MB), site, Torch::default()), Next::Cut(1)));
    assert_eq!(r.cut(Some(id), "ffn"), 1);
    assert!(matches!(r.oom(Some(id), Some(MB), site, Torch::default()), Next::Cut(2)));
    // Review 1, finding 4: a collective site never cuts mid-pass. It recovers locally, its
    // finer level is proposed, and it takes effect when the ranks agree at the next pass.
    let collective = Some(Site { name: "allgather", collective: true, finest: 4 });
    let next = r.oom(Some(id), Some(MB), collective, Torch::default());
    assert!(matches!(next, Next::Park(_) | Next::Exhausted(_)), "{next:?}");
    assert_eq!(r.cut(Some(id), "allgather"), 0, "unchanged within the pass");
    drop(pins);
    drop(pass);
    let next = r.pass(id, "fwd", "1x8", Torch::default()).unwrap();
    assert_eq!(next.levels(), vec![("allgather".to_string(), 1)], "proposed for the agreement");
    assert_eq!(r.cut(Some(id), "allgather"), 0);
    next.agree(vec![("allgather".into(), 2)]);
    assert_eq!(r.cut(Some(id), "allgather"), 2, "the ranks' maximum");
}

/// Review 1, finding 11: dropping an earlier pass after the next one started keeps the next
/// one's state (binds stay its request's).
#[test]
fn a_late_dropped_pass_leaves_the_next_one() {
    let dir = tempfile::tempdir().unwrap();
    let (_l, mem, g) = open("late", 16 * MB);
    let id = mem.load(set(dir.path(), "l", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["pkg@1/l".into()], passes: 2 }] }).unwrap();
    let first = r.pass(id, "fwd", "1x8", Torch::default()).unwrap();
    let second = r.pass(id, "fwd", "1x8", Torch::default()).unwrap();
    drop(first);
    drop(mem.bind(id, 0, 0).unwrap());
    assert_eq!(r.stats().trace.len(), 1, "the bind is still the request's");
    drop(second);
}

/// Review 1, finding 3: a Memory that served custody leaves no row behind once dropped:
/// custody's row is its own and publishes only orphans.
#[test]
fn a_dropped_custody_holder_leaves_no_zombie_row() {
    let dir = tempfile::tempdir().unwrap();
    let g = GpuId { uuid: format!("m{}-zombie", std::process::id()), model: "host".into(), memory: 16 * MB };
    let mem = Memory::open(g.clone(), None, Box::new(HostBackend::new(g.memory))).unwrap();
    mem.load(set(dir.path(), "z", 2, &g.uuid), Level::Gpu).unwrap();
    let ledger = varena::Ledger::create(&g, None).unwrap();
    let held = |l: &varena::Ledger| l.status().rows.iter().map(|r| r.classes.iter().enumerate().filter(|(c, _)| *c != 1).map(|(_, b)| b).sum::<u64>()).sum::<u64>();
    assert_eq!(held(&ledger), 4 * MB);
    drop(mem);
    assert_eq!(held(&ledger), 0, "{:?}", ledger.status().rows);
}

/// S5: after its holder ends, its kept pages show as class O on custody's row.
#[test]
fn orphans_show_as_class_o() {
    let dir = tempfile::tempdir().unwrap();
    let (ledger, mem, g) = open("orphans", 16 * MB);
    mem.load(set(dir.path(), "o", 2, &g.uuid), Level::Gpu).unwrap();
    drop(mem);
    // The custody service sees the connection end on its own thread; it rings the ledger.
    while ledger.status().rows.iter().map(|r| r.classes[1]).sum::<u64>() < 4 * MB {
        ledger.await_event().unwrap();
    }
}

/// S5: `fits(package, "")` means the package's widest measured shape.
#[test]
fn an_empty_shape_fits_by_the_widest_measured() {
    let (ledger, mem, _g) = open("widest", 16 * MB);
    ledger.restore(&[varena::Peak::Request { key: "pkg|1x1".into(), bytes: 2 * MB }, varena::Peak::Request { key: "pkg|4x4".into(), bytes: 17 * MB }]).unwrap();
    let busy = mem.request("busy", Plan::default()).unwrap();
    assert!(ledger.fits("pkg", "1x1"));
    assert!(!ledger.fits("pkg", ""), "the widest (17 MB) beside a running request does not fit");
    drop(busy);
    assert!(ledger.fits("pkg", ""), "an idle GPU fits anything");
}

/// Round 2, finding 4: in a joint step an OOM names its set, and cuts only that set's site,
/// whichever pass started last.
#[test]
fn a_joint_steps_oom_cuts_only_its_own_sets_site() {
    let dir = tempfile::tempdir().unwrap();
    let (_l, mem, g) = open("joint", 4 * MB);
    let a = mem.load(set(dir.path(), "a", 1, &g.uuid), Level::Ram).unwrap();
    let b = mem.load(set(dir.path(), "b", 1, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["pkg@1/a".into(), "pkg@1/b".into()], passes: 1 }] }).unwrap();
    let pa = r.pass(a, "fwd", "1x8", Torch::default()).unwrap();
    let pb = r.pass(b, "fwd", "1x8", Torch::default()).unwrap();
    let pins = [mem.bind(a, 0, 0).unwrap(), mem.bind(b, 0, 0).unwrap()];
    let site = Some(Site { name: "ffn", collective: false, finest: 4 });
    assert!(matches!(r.oom(Some(a), Some(MB), site, Torch::default()), Next::Cut(1)));
    assert_eq!(r.cut(Some(a), "ffn"), 1, "a's pass cut");
    assert_eq!(r.cut(Some(b), "ffn"), 0, "b's pass, started last, untouched");
    drop(pins);
    drop((pa, pb));
}

/// Round 2, finding 6: one unit for rooms. Latents carried across steps (allocated > 0 at every
/// pass start) are part of every absolute room: the remaining working set a row publishes and
/// the peak it records count them, and speculation keeps the next step's absolute room free,
/// so the plan runs with no D page evicted.
#[test]
fn rooms_count_what_earlier_steps_left_allocated() {
    let dir = tempfile::tempdir().unwrap();
    let (ledger, mem, g) = open("units", 16 * MB);
    let a = mem.load(set(dir.path(), "a", 1, &g.uuid), Level::Ram).unwrap();
    let b = mem.load(set(dir.path(), "b", 2, &g.uuid), Level::Ram).unwrap();
    let page = REGION_ALIGN;
    let latents = 6 * MB;
    let torch = Torch { allocated: latents, reserved: latents };
    let plan = || Plan { steps: ["a", "b", "a"].iter().map(|s| Step { sets: vec![format!("pkg@1/{s}")], passes: 1 }).collect() };
    // Measured once: a's pass grows nothing beyond the latents, b's grows 4 MB.
    let first = mem.request("first", plan()).unwrap();
    for (id, grows) in [(a, 0), (b, 4 * MB), (a, 0)] {
        let p = first.pass(id, "fwd", "1x8", torch).unwrap();
        p.end(Measured { allocated_peak: latents + grows, slack: 0, sites: vec![] }, torch);
    }
    let peak = first.end().peak;
    assert_eq!(peak, 2 * page + latents + 4 * MB, "the first request's peak is b's pages and its absolute room");
    let second = mem.request("second", plan()).unwrap();
    let remaining = |l: &varena::Ledger| l.status().rows.iter().map(|r| r.remaining).sum::<u64>();
    assert_eq!(remaining(&ledger), 2 * page + latents + 4 * MB, "published absolute: pages, start allocation, growth");
    let p = second.pass(a, "fwd", "1x8", torch).unwrap();
    assert!(p.measured() && second.measured(), "a's room at 1x8 was measured");
    // Speculation fetched ahead only into what leaves b's absolute room beyond a's free.
    let free = ledger.status().free;
    assert!(free >= 4 * MB, "b's extra room stays free: {free}");
    p.end(Measured { allocated_peak: latents, slack: 0, sites: vec![] }, torch);
    for id in [b, a] {
        let p = second.pass(id, "fwd", "1x8", torch).unwrap();
        (0..if id == b { 2 } else { 1 }).for_each(|block| drop(mem.bind(id, block, 0).unwrap()));
        p.end(Measured { allocated_peak: latents, slack: 0, sites: vec![] }, torch);
    }
    let stats = second.end();
    assert_eq!(stats.evicted[varena::table::Class::D.index()], 0, "{stats:?}");
}

/// Round 2, finding 5: a collective site's flag is kept with its measured room, so a first
/// pass offers it for agreement; and a collective level relaxes only by agreement, when this
/// rank's surplus allows (it proposes the coarser level, the ranks' maximum is in effect).
#[test]
fn collective_levels_relax_by_agreement() {
    let dir = tempfile::tempdir().unwrap();
    let (ledger, mem, g) = open("relax", 16 * MB);
    let c = mem.load(set(dir.path(), "c", 1, &g.uuid), Level::Ram).unwrap();
    let x = mem.load(set(dir.path(), "x", 4, &g.uuid), Level::Ram).unwrap();
    let ag = |peak| varena::SitePeak { name: "ag".into(), peak, collective: true, finest: 4 };
    let measured = |peak, site| Measured { allocated_peak: peak, slack: 0, sites: vec![site] };
    let first = mem.request("first", plan("c")).unwrap();
    first.pass(c, "fwd", "1x8", Torch::default()).unwrap().end(measured(4 * MB, ag(2 * MB)), Torch::default());
    drop(first);
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["pkg@1/c".into()], passes: 4 }] }).unwrap();
    let p = r.pass(c, "fwd", "1x8", Torch::default()).unwrap();
    assert_eq!(p.levels(), vec![("ag".to_string(), 0)], "known collective from its measured room");
    // An OOM at it while the card is tight: the finer level is proposed, for the next pass.
    let pins: Vec<_> = (0..4).map(|b| mem.bind(x, b, 0).unwrap()).collect();
    r.oom(Some(c), Some(MB), Some(Site { name: "ag", collective: true, finest: 4 }), Torch::default());
    p.end(measured(4 * MB, ag(2 * MB)), Torch::default());
    drop(pins);
    mem.unload(x);
    // x's kept pages are free once custody has their Withdraws (they read as P until then).
    while ledger.status().rows.iter().any(|r| r.classes[varena::table::Class::P.index()] > 0) {
        ledger.await_event().unwrap();
    }
    let p = r.pass(c, "fwd", "1x8", Torch::default()).unwrap();
    assert_eq!(p.levels(), vec![("ag".to_string(), 1)]);
    p.agree(vec![("ag".into(), 1)]);
    assert_eq!(r.cut(Some(c), "ag"), 1);
    p.end(measured(3 * MB, ag(MB)), Torch::default());
    // With room to spare this rank proposes the coarser level; another rank still needs the
    // finer one: the maximum stays.
    let p = r.pass(c, "fwd", "1x8", Torch::default()).unwrap();
    assert_eq!(p.levels(), vec![("ag".to_string(), 0)], "relaxed, proposed");
    p.agree(vec![("ag".into(), 1)]);
    assert_eq!(r.cut(Some(c), "ag"), 1, "the ranks' maximum");
    p.end(measured(3 * MB, ag(MB)), Torch::default());
    // Every rank proposes it: relaxed.
    let p = r.pass(c, "fwd", "1x8", Torch::default()).unwrap();
    assert_eq!(p.levels(), vec![("ag".to_string(), 0)]);
    p.agree(vec![("ag".into(), 0)]);
    assert_eq!(r.cut(Some(c), "ag"), 0, "a transient OOM does not cut forever");
}

/// S5 review, 1: a restarted TensorD drops the dispatches it finds (a dead TensorD's), so the
/// GPU never reads as busy forever.
#[test]
fn a_restarted_tensord_drops_stale_dispatches() {
    let g = GpuId { uuid: format!("m{}-dispatch", std::process::id()), model: "host".into(), memory: 16 * MB };
    let ledger = varena::Ledger::create(&g, None).unwrap();
    ledger.dispatch("r1", "pkg", "1x1").unwrap();
    assert_eq!(ledger.status().dispatched.len(), 1);
    drop(ledger);
    let ledger = varena::Ledger::create(&g, None).unwrap();
    assert!(ledger.status().dispatched.is_empty());
}

/// Python review 1: the entry is its own field. Rooms match on (set, entry) and interpolate over
/// shapes within one entry; an entry's name never breaks the shape's parse.
#[test]
fn rooms_interpolate_within_their_entry() {
    let dir = tempfile::tempdir().unwrap();
    let (_l, mem, g) = open("entry", 64 * MB);
    let id = mem.load(set(dir.path(), "e", 1, &g.uuid), Level::Ram).unwrap();
    let room = |entry: &str, shape: &str, measured: Option<u64>| {
        let r = mem.request(&format!("{entry}-{shape}"), plan("e")).unwrap();
        let pass = r.pass(id, entry, shape, Torch::default()).unwrap();
        let w = r.stats().peak;
        if let Some(peak) = measured {
            pass.end(Measured { allocated_peak: peak, slack: 0, sites: vec![] }, Torch::default());
        }
        w - REGION_ALIGN
    };
    room("enc|x", "1x8", Some(4 * MB));
    room("dec", "1x8", Some(8 * MB));
    assert_eq!(room("enc|x", "1x16", None), 8 * MB, "scaled from its own entry's 1x8");
    assert_eq!(room("dec", "1x4", None), 8 * MB, "bounded by its own entry's 1x8");
    assert_eq!(room("other", "1x8", None), 0, "another entry is unmeasured");
}

/// Python review 1: a site never cuts beyond its finest level; at it, the next rung follows.
#[test]
fn a_site_at_its_finest_takes_the_next_rung() {
    let dir = tempfile::tempdir().unwrap();
    let (_l, mem, g) = open("finest", 4 * MB);
    let id = mem.load(set(dir.path(), "f", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", plan("f")).unwrap();
    let _pass = r.pass(id, "fwd", "1x8", Torch::default()).unwrap();
    let _pins: Vec<_> = (0..2).map(|b| mem.bind(id, b, 0).unwrap()).collect();
    let site = Some(Site { name: "ffn", collective: false, finest: 1 });
    assert!(matches!(r.oom(Some(id), Some(MB), site, Torch::default()), Next::Cut(1)));
    let next = r.oom(Some(id), Some(MB), site, Torch::default());
    assert!(matches!(next, Next::Park(_) | Next::Exhausted(_)), "{next:?}");
    assert_eq!(r.cut(Some(id), "ffn"), 1);
}

/// S8: an OOM in the package's own code between passes is answered from the demand order (a
/// cold page goes), and the next pass is unaffected.
#[test]
fn an_oom_between_passes_evicts_cold_pages() {
    let dir = tempfile::tempdir().unwrap();
    let (ledger, mem, g) = open("between", 8 * MB);
    let a = mem.load(set(dir.path(), "a", 2, &g.uuid), Level::Ram).unwrap();
    mem.load(set(dir.path(), "n", 1, &g.uuid), Level::Gpu).unwrap();
    let r = mem.request("r", Plan { steps: vec![Step { sets: vec!["pkg@1/a".into()], passes: 2 }] }).unwrap();
    let pass = r.pass(a, "fwd", "1x8", Torch::default()).unwrap();
    (0..2).for_each(|b| drop(mem.bind(a, b, 0).unwrap()));
    pass.end(Measured::default(), Torch::default());
    // Free bytes are claimed once; then the cold page (no declared use) goes.
    let free = ledger.status().free;
    assert!(matches!(r.oom(None, Some(free + REGION_ALIGN), None, Torch::default()), Next::Retry));
    assert_eq!(ledger.status().rows.iter().map(|r| r.classes[varena::table::Class::N.index()]).sum::<u64>(), 0, "n went");
    let pass = r.pass(a, "fwd", "1x8", Torch::default()).unwrap();
    assert!(!mem.bind(a, 1, 0).unwrap().fresh);
    assert_eq!(r.cut(Some(a), "ffn"), 0, "nothing was cut");
    pass.end(Measured::default(), Torch::default());
    assert!(!r.stats().lowered, "nothing was lowered");
}

/// Review 3 queue, item 3: an op site outside any pass (the package's own split) has the
/// request's own cut memo: an OOM there takes the demand order, then cuts it, up to its finest;
/// `cut(None, site)` answers it.
#[test]
fn a_site_outside_any_pass_has_the_requests_memo() {
    let dir = tempfile::tempdir().unwrap();
    let (_l, mem, g) = open("outside", 4 * MB);
    let id = mem.load(set(dir.path(), "o", 2, &g.uuid), Level::Ram).unwrap();
    let r = mem.request("r", plan("o")).unwrap();
    let _pins: Vec<_> = (0..2).map(|b| mem.bind(id, b, 0).unwrap()).collect();
    let split = Some(Site { name: "split", collective: false, finest: 2 });
    assert!(matches!(r.oom(None, Some(MB), split, Torch::default()), Next::Cut(1)));
    assert_eq!(r.cut(None, "split"), 1);
    assert!(matches!(r.oom(None, Some(MB), split, Torch::default()), Next::Cut(2)));
    let next = r.oom(None, Some(MB), split, Torch::default());
    assert!(matches!(next, Next::Park(_) | Next::Exhausted(_)), "at its finest the next rung: {next:?}");
    assert_eq!(r.cut(None, "split"), 2);
}

/// Review 3, N7: a room is the latest completed measurement (it comes down when a pass needs
/// less); an OOM at a measured room records what it needed at once; a restore never overrides
/// what this run measured.
#[test]
fn the_latest_measured_room_wins() {
    let dir = tempfile::tempdir().unwrap();
    let (ledger, mem, g) = open("latest", 64 * MB);
    let id = mem.load(set(dir.path(), "l", 1, &g.uuid), Level::Ram).unwrap();
    let pages = REGION_ALIGN;
    room(&mem, id, "l", "1x8", pages, Some(8 * MB));
    assert_eq!(room(&mem, id, "l", "1x8", pages, Some(4 * MB)), 8 * MB);
    assert_eq!(room(&mem, id, "l", "1x8", pages, None), 4 * MB, "came down");
    let r = mem.request("oom", plan("l")).unwrap();
    let pass = r.pass(id, "fwd", "1x8", Torch::default()).unwrap();
    r.oom(Some(id), Some(6 * MB), None, Torch::default());
    drop(pass);
    drop(r);
    assert_eq!(room(&mem, id, "l", "1x8", pages, None), 6 * MB, "the room the OOM needed");
    let saved = ledger.peaks().into_iter().find_map(|p| match p {
        varena::Peak::Pass { key, mut room } if room.shape == "1x8" => {
            room.delta = 16 * MB;
            Some(varena::Peak::Pass { key, room })
        }
        _ => None,
    });
    ledger.restore(&[saved.unwrap()]).unwrap();
    assert_eq!(room(&mem, id, "l", "1x8", pages, None), 6 * MB, "this run's measurement is newer");
}

